// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Private test-only bridge from the production coverage-aware head plan to
//! the v18 adaptive replay. It deliberately contains no codec/public policy.
const std = @import("std");
const builtin = @import("builtin");
const platform = @import("antfly_platform");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const primary_probe = @import("attention_match_primary_train_candidate.zig");
const probe_select = @import("attention_match_train_critical_policy_probe.zig");
const train_cv = @import("attention_match_train_domain_cv_split.zig");
const gram_provider = @import("../../ops/metal_attention_match_train_gram.zig");
const gram_delta = @import("attention_match_gram_delta_accelerate.zig");
const validation_accelerate = @import("../../ops/metal_attention_match_validation.zig");
const reference_cache = @import("attention_match_fp64_reference_summary_cache.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const bounded_run = @import("attention_match_joint_bounded_run.zig");
const tsqr_value = @import("attention_match_e2_tsqr_value_backend.zig");
const prepared_mod = @import("attention_match_joint_prepared_input.zig");
const metal = @import("../../ops/metal_attention_match_fit.zig");

pub const strategy = "identity_chunk_coefficients_adaptive_v18";
pub const bounded_geometry_policy = "flat_gram_le96_else_bounded_page_tile8_builder256mib_cap1gib_v1";
pub const bounded_accelerated_geometry_policy = "flat_gram_le96_else_bounded_page_tile64_accelerate_builder256mib_cap1gib_v1";

fn admissionDiagnosticEnabled() bool {
    return platform.env.getenvBool("ANTFLY_METAL_QUALITY_CRITICAL_PHASE_ADMISSION_DIAGNOSTIC");
}

fn admissionFailure(step: []const u8, err: anyerror, caller_external: usize, normalized_and_ranges: usize, postfit: usize, combined_cap: usize, query_rows: usize, source_rows: usize, dim: usize, ranges: usize) anyerror {
    if (admissionDiagnosticEnabled()) std.debug.print(
        "attention_match_adaptive_admission_failure step={s} error={s} q={d} n={d} d={d} ranges={d} caller_external_bytes={d} normalized_and_ranges_bytes={d} postfit_bytes={d} combined_cap={d} available_after_caller={d} available_after_normalized={d} diagnostic_only=true\n",
        .{ step, @errorName(err), query_rows, source_rows, dim, ranges, caller_external, normalized_and_ranges, postfit, combined_cap, combined_cap -| caller_external, combined_cap -| tryAddDiagnostic(caller_external, normalized_and_ranges) },
    );
    return err;
}

fn tryAddDiagnostic(a: usize, b: usize) usize {
    return std.math.add(usize, a, b) catch std.math.maxInt(usize);
}

fn admissionSnapshot(step: []const u8, caller_external: usize, normalized_and_ranges: usize, postfit: usize, combined_cap: usize, query_rows: usize, source_rows: usize, dim: usize, ranges: usize) void {
    if (admissionDiagnosticEnabled()) std.debug.print(
        "attention_match_adaptive_admission step={s} q={d} n={d} d={d} ranges={d} caller_external_bytes={d} normalized_and_ranges_bytes={d} postfit_bytes={d} replay_external_bytes={d} combined_cap={d} available_after_replay_external={d} diagnostic_only=true\n",
        .{ step, query_rows, source_rows, dim, ranges, caller_external, normalized_and_ranges, postfit, tryAddDiagnostic(tryAddDiagnostic(caller_external, normalized_and_ranges), postfit), combined_cap, combined_cap -| tryAddDiagnostic(tryAddDiagnostic(caller_external, normalized_and_ranges), postfit) },
    );
}

test "adaptive admission diagnostic preserves error and saturates display arithmetic" {
    try std.testing.expectEqual(std.math.maxInt(usize), tryAddDiagnostic(std.math.maxInt(usize), 1));
    try std.testing.expectEqual(@as(anyerror, error.V18TrainingCacheUnsupported), admissionFailure("test", error.V18TrainingCacheUnsupported, 2, 3, 5, 7, 11, 13, 17, 19));
}
pub const full_train_reduction_batch_policy = "cached_train_fp64_reduction_full_query_batch_v1";

fn trainingReductionBatchRows(policy: engine.Options.PrivatePerformancePolicy, query_count: usize) !usize {
    if (query_count == 0) return error.InvalidCompactionQueries;
    return if (policy == .owned_capture_summary_incremental_reduction_v1 or policy == .owned_capture_summary_incremental_reduction_critical025_v1 or policy == .diagnostic_capture_gram_batch256_v1) @min(query_count, 768) else 64;
}

test "private reduction batches the complete authenticated TRAIN matrix" {
    // The qualified E4 route has 768 TRAIN rows. Keeping this decision at the
    // caller makes both workspace planning and backend dispatch observe the
    // same batch size; no query is removed or moved to heldout.
    try std.testing.expectEqual(@as(usize, 768), try trainingReductionBatchRows(.owned_capture_summary_incremental_reduction_v1, 12_288));
    try std.testing.expectEqual(@as(usize, 767), try trainingReductionBatchRows(.owned_capture_summary_incremental_reduction_v1, 767));
    try std.testing.expectEqual(@as(usize, 64), try trainingReductionBatchRows(.owned_capture_summary_incremental_v1, 768));
    try std.testing.expectError(error.InvalidCompactionQueries, trainingReductionBatchRows(.owned_capture_summary_incremental_reduction_v1, 0));
}

pub fn boundedGeometryPolicyFor(accelerated: bool) []const u8 {
    return if (accelerated) bounded_accelerated_geometry_policy else bounded_geometry_policy;
}

pub fn boundedGeometryPolicyDigestFor(accelerated: bool) [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(boundedGeometryPolicyFor(accelerated), &digest, .{});
    return digest;
}

pub fn boundedGeometryPolicyDigest() [32]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bounded_geometry_policy, &digest, .{});
    return digest;
}

fn resolvedGeometryRoute(candidate_count: usize, configured: engine.Options.AdaptiveV18GeometryRoute) !engine.Options.AdaptiveV18GeometryRoute {
    if (configured == .bounded_page_wide_cache_v1) return .bounded_page_wide_cache_v1;
    if (candidate_count <= @import("attention_match_joint_gram_geometry.zig").max_candidates) return .flat_gram_only;
    if (configured != .bounded_page_over_96) return error.V18TrainingCacheUnsupported;
    return .bounded_page_over_96;
}

test "32K geometry route is explicit while 8K remains flat" {
    try std.testing.expectEqual(engine.Options.AdaptiveV18GeometryRoute.flat_gram_only, try resolvedGeometryRoute(69, .bounded_page_over_96));
    try std.testing.expectEqual(engine.Options.AdaptiveV18GeometryRoute.bounded_page_over_96, try resolvedGeometryRoute(372, .bounded_page_over_96));
    try std.testing.expectError(error.V18TrainingCacheUnsupported, resolvedGeometryRoute(372, .flat_gram_only));
    try std.testing.expectEqual(engine.Options.AdaptiveV18GeometryRoute.bounded_page_wide_cache_v1, try resolvedGeometryRoute(72, .bounded_page_wide_cache_v1));
}

/// Prior codec payloads are indexed in current-head row space. Adaptive replay
/// may remove later exact rows, but it must keep every already encoded row at
/// the same index before cloning the immutable payload metadata.
fn validatePriorCodecMap(current: checkpoint.Head, source_indices: []const usize, coverage_indices: []const usize) !void {
    if (source_indices.len != coverage_indices.len) return error.InvalidAttentionShape;
    if (current.fp8.rows != 0) {
        const first: usize = current.fp8.first_row;
        const end = try std.math.add(usize, first, current.fp8.rows);
        if (end > source_indices.len) return error.InvalidAttentionShape;
        for (first..end) |row|
            if (source_indices[row] != row or coverage_indices[row] != row) return error.InvalidAttentionShape;
    }
    for (current.compressed.spans) |span| {
        const end = try std.math.add(usize, span.first, span.rows);
        if (end > source_indices.len) return error.InvalidAttentionShape;
        for (span.first..end) |row|
            if (source_indices[row] != row or coverage_indices[row] != row) return error.InvalidAttentionShape;
    }
}

const Normalized = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    fn deinit(self: *@This()) void {
        self.allocator.free(self.bias);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.* = undefined;
    }
};

fn normalizeHead(allocator: std.mem.Allocator, head: checkpoint.Head) !Normalized {
    const keys = try allocator.alloc(f64, head.keys.len);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, head.values.len);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, head.bias.len);
    errdefer allocator.free(bias);
    for (keys, head.keys) |*out, value| out.* = value;
    for (values, head.values) |*out, value| out.* = value;
    for (bias, head.bias) |*out, value| out.* = value;
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias };
}

pub const Run = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    ranges: []admission.Range,
    result: replay.Result,
    numerical_policy: engine.Options.AdaptiveV18NumericalPolicy,
    gram_backend_calls: usize = 0,
    validation_summary_calls: usize = 0,
    validation_max_batch_rows: usize = 0,
    validation_workspace_peak_bytes: usize = 0,
    reference_cache_counters: reference_cache.Counters = .{},
    phase_timing: replay.PhaseTiming = .{},
    geometry_route: engine.Options.AdaptiveV18GeometryRoute = .flat_gram_only,
    bounded_evidence: ?bounded_run.Evidence = null,
    bounded_tsqr_calls: usize = 0,
    bounded_tsqr_workspace_peak_bytes: usize = 0,
    bounded_tsqr_cancellation_checks: usize = 0,

    pub fn deinit(self: *@This()) void {
        self.result.deinit();
        self.allocator.free(self.ranges);
        self.allocator.free(self.bias);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.* = undefined;
    }
};

fn validateBoundedReductionContract(evidence: bounded_run.Evidence, reduction: replay.TrainingReductionEvidence, range_count: usize, query_rows: usize) !void {
    const expected_scores = try std.math.mul(usize, range_count, 3);
    const expected_rows = try std.math.mul(usize, expected_scores, query_rows);
    const scoring = @import("attention_match_joint_bounded_page_scoring.zig");
    const scalar = std.mem.eql(u8, evidence.bounded_reduction_policy, scoring.scalar_execution_policy);
    const accelerated = std.mem.eql(u8, evidence.bounded_reduction_policy, scoring.accelerate_execution_policy);
    const valid_execution = (scalar and evidence.bounded_score_accelerate_calls == 0 and evidence.bounded_score_accelerate_tiles == 0 and evidence.bounded_score_accelerate_dgemm_calls == 0 and evidence.bounded_score_cancellation_checks == 0) or
        (accelerated and evidence.bounded_score_accelerate_calls == expected_scores and evidence.bounded_score_accelerate_tiles >= expected_scores and evidence.bounded_score_accelerate_dgemm_calls == try std.math.mul(usize, evidence.bounded_score_accelerate_tiles, 2) and evidence.bounded_score_cancellation_checks >= try std.math.mul(usize, evidence.bounded_score_accelerate_tiles, 2));
    if (!valid_execution or evidence.bounded_score_completions != expected_scores or evidence.bounded_score_query_rows != expected_rows or
        !std.meta.eql(reduction, replay.TrainingReductionEvidence{}))
        return error.V18TrainingScoreUnsupported;
}

/// Converts replay maps through the current checkpoint head. The returned head
/// owns all slices and preserves existing storage annotations for the shared
/// engine codec tail; no map is regenerated from a dense logical prefix.
pub fn materializeHead(allocator: std.mem.Allocator, current: checkpoint.Head, run: *const Run) !checkpoint.Head {
    const result = &run.result;
    const rows = result.head.rows;
    if (result.source_indices.len != rows or result.coverage_source_indices.len != rows) return error.InvalidAttentionShape;
    try validatePriorCodecMap(current, result.source_indices, result.coverage_source_indices);
    const positions = try allocator.alloc(u64, rows);
    errdefer allocator.free(positions);
    const ends = try allocator.alloc(u64, rows);
    errdefer allocator.free(ends);
    const keys = try allocator.alloc(f32, try std.math.mul(usize, rows, result.head.dim));
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f32, keys.len);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f32, rows);
    errdefer allocator.free(bias);
    for (result.source_indices, result.coverage_source_indices, 0..) |source_i, coverage_i, row| {
        if (source_i >= current.positions.len or coverage_i >= current.coverage_ends.len) return error.InvalidAttentionShape;
        positions[row] = current.positions[source_i];
        ends[row] = current.coverage_ends[coverage_i];
        bias[row] = @floatCast(result.head.bias.?[row]);
    }
    for (result.head.keys, keys) |value, *out| out.* = @floatCast(value);
    for (result.head.values, values) |value, *out| out.* = @floatCast(value);
    var compressed = try current.compressed.clone(allocator);
    errdefer compressed.deinit(allocator);
    var compacted_until = current.compacted_until;
    if (result.accepted) for (run.ranges) |range| {
        const end = range.start + range.rows;
        if (end == 0 or end > current.coverage_ends.len) return error.InvalidAttentionShape;
        compacted_until = @max(compacted_until, current.coverage_ends[end - 1]);
    };
    return .{ .dim = current.dim, .compacted_until = compacted_until, .positions = positions, .coverage_ends = ends, .keys = keys, .values = values, .bias = bias, .fp8 = current.fp8, .compressed = compressed };
}

fn bytesForNormalized(head: checkpoint.Head, ranges: usize) !usize {
    var elements = try std.math.add(usize, head.keys.len, head.values.len);
    elements = try std.math.add(usize, elements, head.bias.len);
    var bytes = try std.math.mul(usize, elements, @sizeOf(f64));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, ranges, @sizeOf(admission.Range)));
    return bytes;
}

fn constructionBytes(head: checkpoint.Head, plan: engine.HeadPlan) !usize {
    const maximum_ranges = std.math.sub(usize, plan.old_rows, plan.prefix_rows) catch return error.InvalidFitConfig;
    var bytes = try bytesForNormalized(head, 0);
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, maximum_ranges, @sizeOf(engine.FitRange) * 2));
    return bytes;
}

fn exactFitRangeCount(head: checkpoint.Head, plan: engine.HeadPlan, chunk_rows: usize) !usize {
    if (chunk_rows == 0 or head.positions.len != head.coverage_ends.len or head.positions.len != head.bias.len or
        plan.prefix_rows > plan.old_rows or plan.old_rows > head.positions.len)
        return error.InvalidFitConfig;
    var count: usize = 0;
    var at = plan.prefix_rows;
    while (at < plan.old_rows) {
        while (at < plan.old_rows and (head.positions[at] == std.math.maxInt(u64) or
            head.coverage_ends[at] != head.positions[at] + 1 or head.bias[at] != 0)) : (at += 1)
        {}
        if (at == plan.old_rows) break;
        var run_end = at + 1;
        while (run_end < plan.old_rows and head.positions[run_end] != std.math.maxInt(u64) and
            head.coverage_ends[run_end] == head.positions[run_end] + 1 and
            head.positions[run_end - 1] != std.math.maxInt(u64) and head.bias[run_end] == 0 and
            head.positions[run_end] == head.positions[run_end - 1] + 1) : (run_end += 1)
        {}
        count = try std.math.add(usize, count, std.math.divCeil(usize, run_end - at, chunk_rows) catch return error.InvalidFitConfig);
        at = run_end;
    }
    if (count == 0) return error.NoEligibleAttentionRows;
    return count;
}

/// Conservative post-replay peak. The scheduler's resident source+candidate
/// reservation owns the materialized f32 checkpoint arrays, so this counts only
/// worker-local queries, normalized/result f64 payloads, maps/geometry, final
/// validation scratch, and the optional codec scratch which overlap them.
fn authorizationPeakBytes(head: checkpoint.Head, range_count: usize, query_bytes: usize, train_queries: usize, heldout_queries: usize, options: *const engine.Options) !usize {
    const rows = head.positions.len;
    const d: usize = head.dim;
    const f64_head = try std.math.mul(usize, rows, try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, d, 2), 1), @sizeOf(f64)));
    const maps = try std.math.mul(usize, rows, 2 * @sizeOf(usize));
    const geometry = try std.math.mul(usize, 3, try std.math.add(usize, @sizeOf(replay.Branch), try std.math.mul(usize, range_count, @sizeOf(usize) + @sizeOf(f64))));
    var validation_cfg = options.fit;
    var validation_context: validation_accelerate.Context = .{};
    validation_cfg.backend = switch (options.independent_validation_policy) {
        .scalar_reference => null,
        .accelerate_fp64_summary_v1 => validation_accelerate.backend(&validation_context),
    };
    const validation = try solver.validationBytes(rows, rows, d, heldout_queries, validation_cfg);
    var codec: usize = 0;
    if (options.rotated4_checkpoint) {
        const samples = @min(train_queries, 64);
        const packed_bytes = try std.math.mul(usize, rows, try std.math.add(usize, 60, try std.math.mul(usize, d, 6)));
        codec = try std.math.add(usize, packed_bytes, try std.math.add(usize, try std.math.mul(usize, rows, try std.math.add(usize, 17, try std.math.mul(usize, samples, 16))), try std.math.mul(usize, samples, try std.math.add(usize, 16, try std.math.mul(usize, d, 8)))));
        if (options.fit.backend) |backend| codec = try std.math.add(usize, codec, try backend.workspace_fn(backend.context, rows, 1, samples, d));
    } else if (options.fp8_checkpoint) {
        codec = try std.math.mul(usize, rows, try std.math.add(usize, 40, try std.math.mul(usize, d, 4)));
    }
    var peak = try std.math.add(usize, query_bytes, try std.math.mul(usize, f64_head, 2));
    peak = try std.math.add(usize, peak, maps);
    peak = try std.math.add(usize, peak, try std.math.mul(usize, range_count, @sizeOf(admission.Range)));
    peak = try std.math.add(usize, peak, geometry);
    peak = try std.math.add(usize, peak, @max(validation, codec));
    return peak;
}

pub const BoundedValueEvidence = struct {
    calls: usize = 0,
    workspace_peak_bytes: usize = 0,
    cancellation_checks: usize = 0,
};

pub const PrimaryTrainProbeConfig = struct {
    heldout_query_rows: usize,
    critical_policy: replay.CriticalCorePolicy,
    postfit_workspace_bytes: usize,
    validation_backend: solver.Backend,
    selection_train_queries: []const f64 = &.{},
    selection_split_sha256: [32]u8 = @splat(0),
    timing: ?*primary_probe.Timing = null,
    flat_cache_allocation_authority: ?@import("attention_match_joint_training_source.zig").AllocationAuthority = null,
    bounded_value_observer: ?*BoundedValueEvidence = null,
    accelerate_bounded_score: bool = false,
};
pub const PrimaryTrainProbeTiming = primary_probe.Timing;

const PrimaryTrainProbeCapture = struct {
    output: ?primary_probe.Owned = null,
    config: PrimaryTrainProbeConfig,
    replay_phase_timing: replay.PhaseTiming = .{},

    fn digestSource(source: solver.Head) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("antfly.primary-train-probe.source.v1\x00");
        hash.update(std.mem.sliceAsBytes(source.keys));
        hash.update(std.mem.sliceAsBytes(source.values));
        hash.update(std.mem.sliceAsBytes(source.bias.?));
        hash.update(std.mem.asBytes(&source.rows));
        hash.update(std.mem.asBytes(&source.dim));
        hash.update(std.mem.asBytes(&source.scale));
        hash.update(std.mem.asBytes(&source.softcap));
        return hash.finalResult();
    }

    fn digestTrain(train: []const f64, dim: usize) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("antfly.primary-train-probe.train.v1\x00");
        hash.update(std.mem.asBytes(&dim));
        hash.update(std.mem.sliceAsBytes(train));
        return hash.finalResult();
    }

    fn capture(
        raw: ?*anyopaque,
        allocator: std.mem.Allocator,
        source: solver.Head,
        train: []const f64,
        ranges: []const admission.Range,
        cached: *@import("attention_match_joint_training_source.zig").Source,
        adaptive_plan: *const replay.AdaptivePlan,
        fit_cfg: solver.Config,
        external_live_bytes: usize,
        combined_cap: usize,
    ) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        if (self.output != null or self.config.heldout_query_rows == 0 or adaptive_plan.geometry_budget_percent != 91)
            return error.InvalidFitConfig;
        self.output = try primary_probe.buildPrimaryTrainCandidate(allocator, source, train, ranges, adaptive_plan, fit_cfg, cached, .{
            .critical_policy = self.config.critical_policy,
            .primary_budget_percent = 91,
            .postfit_workspace_bytes = self.config.postfit_workspace_bytes,
            .external_live_bytes = external_live_bytes,
            .combined_cap = combined_cap,
            .source_authority_sha256 = digestSource(source),
            .train_domain_sha256 = digestTrain(train, source.dim),
            .validation_backend = self.config.validation_backend,
            .selection_train_queries = self.config.selection_train_queries,
            .selection_split_sha256 = self.config.selection_split_sha256,
            .timing = self.config.timing,
        });
    }

    fn capturePageBacked(raw: ?*anyopaque, view: bounded_run.PrimaryProbeView) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        if (self.output != null or self.config.heldout_query_rows == 0 or view.adaptive_plan.geometry_budget_percent != 91)
            return error.InvalidFitConfig;
        self.output = try primary_probe.buildPrimaryTrainCandidatePageBacked(view.allocator, view.source, view.train, view.ranges, view.adaptive_plan, view.cfg, view.postfit_hook, .{
            .critical_policy = self.config.critical_policy,
            .primary_budget_percent = 91,
            .postfit_workspace_bytes = self.config.postfit_workspace_bytes,
            .external_live_bytes = view.external_live_bytes,
            .combined_cap = view.combined_cap,
            .source_authority_sha256 = digestSource(view.source),
            .train_domain_sha256 = digestTrain(view.train, view.source.dim),
            .validation_backend = self.config.validation_backend,
            .selection_train_queries = self.config.selection_train_queries,
            .selection_split_sha256 = self.config.selection_split_sha256,
            .timing = self.config.timing,
        });
    }

    fn take(self: *@This()) !primary_probe.Owned {
        var value = self.output orelse return error.InvalidFitConfig;
        value.replay_phase_timing = self.replay_phase_timing;
        self.output = null;
        return value;
    }

    fn deinit(self: *@This()) void {
        if (self.output) |*owned| owned.deinit();
        self.output = null;
    }
};
fn transferProbeTiming(capture: ?*PrimaryTrainProbeCapture, timing: replay.PhaseTiming) void {
    if (capture) |value| {
        if (value.config.timing != null) value.replay_phase_timing = timing;
    }
}
test "typed primary probe timing transfer is diagnostic only" {
    var observer: primary_probe.Timing = .{};
    var capture: PrimaryTrainProbeCapture = .{ .config = .{ .heldout_query_rows = 1, .critical_policy = .train_probability_gte_0_025_no_top1_v1, .postfit_workspace_bytes = 1, .validation_backend = undefined, .timing = &observer } };
    transferProbeTiming(&capture, .{ .cache_prepare_ns = 7, .chooser_ns = 11 });
    try std.testing.expectEqual(@as(u64, 7), capture.replay_phase_timing.cache_prepare_ns);
    try std.testing.expectEqual(@as(u64, 11), capture.replay_phase_timing.chooser_ns);
    capture.config.timing = null;
    transferProbeTiming(&capture, .{ .cache_prepare_ns = 99 });
    try std.testing.expectEqual(@as(u64, 7), capture.replay_phase_timing.cache_prepare_ns);
}

/// Runs one current checkpoint head. `caller_external_live_bytes` includes the
/// checkpoint/source state and query matrices but excludes normalization and
/// the returned range array, which are added here before pre-cache admission.
fn runHeadImpl(
    allocator: std.mem.Allocator,
    head: checkpoint.Head,
    plan: engine.HeadPlan,
    train: []const f64,
    heldout: []const f64,
    scale: f64,
    softcap: f64,
    fit_cfg: solver.Config,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    numerical_policy: engine.Options.AdaptiveV18NumericalPolicy,
    geometry_route: engine.Options.AdaptiveV18GeometryRoute,
    bounded_tile_queries: usize,
    bounded_builder_transient_bytes: usize,
    bounded_delta_accelerated: bool,
    bounded_tsqr_value_backend: bool,
    workspace_profile: engine.Options.AdaptiveV18WorkspaceProfile,
    bounded_score_policy: engine.Options.AdaptiveV18BoundedScorePolicy,
    geometry_budget_percent: u8,
    fallback_geometry_budget_percent: u8,
    fallback_authorize_context: ?*anyopaque,
    fallback_authorize_fn: ?*const fn (?*anyopaque) bool,
    validation_policy: engine.Options.IndependentValidationPolicy,
    reference_cache_policy: engine.Options.V18ReferenceSummaryCachePolicy,
    geometry_chooser_policy: engine.Options.GeometryChooserPolicy,
    private_performance_policy: engine.Options.PrivatePerformancePolicy,
    critical_core_policy: replay.CriticalCorePolicy,
    postfit_policy: replay.PostfitPolicy,
    rejected_detail_diagnostic: bool,
    train_probability_census: bool,
    flat_cache_allocation_authority: ?@import("attention_match_joint_training_source.zig").AllocationAuthority,
    primary_train_probe: ?*PrimaryTrainProbeCapture,
    rejected_branch_capture: ?replay.RejectedBranchCapture,
) !Run {
    // `headFitRanges` temporarily owns an upper-bound descriptor scratch plus
    // its exact result. Admit that construction peak before either allocation.
    const construction_bytes = try constructionBytes(head, plan);
    const construction_peak = try std.math.add(usize, caller_external_live_bytes, construction_bytes);
    if (construction_peak > combined_cap) return admissionFailure("construction", error.V18TrainingCacheUnsupported, caller_external_live_bytes, construction_bytes, 0, combined_cap, train.len / head.dim, head.positions.len, head.dim, 0);
    const ranges = blk: {
        const fit_ranges = try engine.headFitRanges(allocator, head, plan, 256);
        defer allocator.free(fit_ranges);
        const owned = try allocator.alloc(admission.Range, fit_ranges.len);
        errdefer allocator.free(owned);
        for (owned, fit_ranges) |*out, source| out.* = .{ .start = source.start, .rows = source.rows };
        break :blk owned;
    };
    errdefer allocator.free(ranges);

    var normalized = try normalizeHead(allocator, head);
    errdefer normalized.deinit();
    const original: solver.Head = .{ .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .rows = head.positions.len, .dim = head.dim, .scale = scale, .softcap = softcap };
    const normalized_and_ranges = try bytesForNormalized(head, ranges.len);
    const external = try std.math.add(usize, caller_external_live_bytes, normalized_and_ranges);
    if (external > combined_cap) return admissionFailure("normalization", error.V18TrainingCacheUnsupported, caller_external_live_bytes, normalized_and_ranges, 0, combined_cap, train.len / original.dim, original.rows, original.dim, ranges.len);
    var cfg = fit_cfg;
    cfg.target_rows = 1;
    cfg.selection = .residual_omp;
    cfg.mass_solver = .active_set;
    cfg.ridge = 1e-8;
    cfg.max_workspace_bytes = combined_cap;
    var gram_context: gram_provider.Context = .{};
    var validation_context: validation_accelerate.Context = .{};
    const validation_backend = switch (validation_policy) {
        .scalar_reference => null,
        .accelerate_fp64_summary_v1 => validation_accelerate.backend(&validation_context),
    };
    const candidate_count = try std.math.mul(usize, ranges.len, 3);
    // Formal E2's q30720 TRAIN domain must never construct the scalar q*n FP64
    // reference matrix, even when its 7936-row shape has only 69 candidates.
    // The sealed E2 workspace profile therefore selects the page route by
    // authority, while every established flat/E4 route keeps candidate-count
    // resolution and scalar reference bit identity unchanged.
    const resolved_route: engine.Options.AdaptiveV18GeometryRoute = if (workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1)
        .bounded_page_over_96
    else
        try resolvedGeometryRoute(candidate_count, geometry_route);
    var cache_counters: reference_cache.Counters = .{};
    const cache_bytes = switch (reference_cache_policy) {
        .disabled => 0,
        .exact_original_train_hho_v1 => blk: {
            if (validation_policy != .accelerate_fp64_summary_v1 or train.len == 0) return error.InvalidFitConfig;
            cache_counters.detailed_observer_bypasses += @intFromBool(rejected_detail_diagnostic);
            break :blk try std.math.add(usize, try reference_cache.retainedBytes(train.len / original.dim, original.dim), try reference_cache.retainedBytes(heldout.len / original.dim, original.dim));
        },
    };
    const admitted_external = std.math.add(usize, external, cache_bytes) catch {
        cache_counters.cap_fallbacks_before_allocation += 1;
        return error.V18TrainingCacheUnsupported;
    };
    if (admitted_external > combined_cap) {
        cache_counters.cap_fallbacks_before_allocation += 1;
        return error.V18TrainingCacheUnsupported;
    }
    // The wide page route cannot lazily materialize scalar q*n FP64
    // probabilities inside the TRAIN reference summary.  Reuse the same
    // independent Accelerate FP64 summary backend as heldout; it computes the
    // identical original-head targets/masses without retaining P.  Flat and
    // legacy routes keep their established scalar bit-identity contract.
    var train_cache: ?reference_cache.Context = if (reference_cache_policy == .exact_original_train_hho_v1)
        if (resolved_route == .bounded_page_over_96 or resolved_route == .bounded_page_wide_cache_v1)
            try reference_cache.Context.init(allocator, .train, original, train, validation_backend orelse return error.AcceleratedValidationUnsupported, &cache_counters)
        else
            try reference_cache.Context.initScalar(allocator, original, train, &cache_counters)
    else
        null;
    defer if (train_cache) |*cache| cache.deinit();
    var heldout_cache: ?reference_cache.Context = if (reference_cache_policy == .exact_original_train_hho_v1 and heldout.len != 0 and !rejected_detail_diagnostic) try reference_cache.Context.init(allocator, .heldout, original, heldout, validation_backend.?, &cache_counters) else null;
    defer if (heldout_cache) |*cache| cache.deinit();
    const capture_backend = if (private_performance_policy != .legacy) blk: {
        const capture_route = resolved_route;
        if (capture_route == .bounded_page_over_96 or capture_route == .bounded_page_wide_cache_v1) break :blk null;
        if (capture_route != .flat_gram_only)
            return admissionFailure("geometry_route", error.V18TrainingCacheUnsupported, caller_external_live_bytes, normalized_and_ranges, 0, combined_cap, train.len / original.dim, original.rows, original.dim, ranges.len);
        const fitter = try metal.Fitter.fromBackend(fit_cfg.backend orelse return admissionFailure("fit_backend", error.V18TrainingCacheUnsupported, caller_external_live_bytes, normalized_and_ranges, 0, combined_cap, train.len / original.dim, original.rows, original.dim, ranges.len));
        break :blk fitter.trainingCaptureBackend();
    } else null;
    var phase_timing: replay.PhaseTiming = .{};
    defer transferProbeTiming(primary_train_probe, phase_timing);
    const postfit_workspace = switch (postfit_policy) {
        .disabled => 0,
        .train_global_residual_range_mass_l2_exact_f32_cache_v1 => try replay.postfitWorkspaceReserve(train.len / original.dim, original.rows, original.rows, ranges.len),
        .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => try replay.postfitLawsonWorkspaceReserve(train.len / original.dim, original.rows, original.rows, ranges.len),
        .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => blk: {
            var bytes = try @import("attention_match_train_l2_surrogate_minimax_postfit.zig").workspaceBytes(original, original, train.len / original.dim, ranges.len, validation_backend orelse return error.InvalidFitConfig, cfg);
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, ranges.len, @sizeOf(@import("attention_match_train_group_mass_postfit.zig").Range)));
            break :blk try std.math.add(usize, bytes, try std.math.mul(usize, original.rows, @sizeOf(usize)));
        },
    };
    admissionSnapshot("pre_replay", caller_external_live_bytes, normalized_and_ranges, postfit_workspace, combined_cap, train.len / original.dim, original.rows, original.dim, ranges.len);
    const reduction_batch_rows = try trainingReductionBatchRows(private_performance_policy, train.len / original.dim);
    if (resolved_route == .bounded_page_over_96 or resolved_route == .bounded_page_wide_cache_v1) {
        if (rejected_branch_capture != null) return error.V18TrainingCacheUnsupported;
        if (postfit_policy == .train_global_residual_range_mass_lawson_exact_rounded_max_v1) return error.V18TrainingCacheUnsupported;
        if (critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1 or critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2) return error.V18TrainingCacheUnsupported;
        var value_context: tsqr_value.Context = .{};
        defer {
            if (primary_train_probe) |probe| {
                if (probe.config.bounded_value_observer) |observer| observer.* = .{
                    .calls = value_context.calls,
                    .workspace_peak_bytes = value_context.workspace_peak_bytes,
                    .cancellation_checks = value_context.cancellation_checks,
                };
            }
        }
        if (bounded_tsqr_value_backend) cfg.value_backend = value_context.valueBackend();
        // Planner admission below observes cfg.value_backend before allocating;
        // the adapter publishes output only after TSQR completes and checks control.
        if (validation_policy != .scalar_reference and validation_policy != .accelerate_fp64_summary_v1) return error.AcceleratedValidationUnsupported;
        if (numerical_policy != .accepted_f32_gram_reuse_v1 or (geometry_route != .bounded_page_over_96 and geometry_route != .bounded_page_wide_cache_v1) or
            bounded_tile_queries != @as(usize, if (bounded_delta_accelerated) 64 else 8) or bounded_builder_transient_bytes != 256 * 1024 * 1024 or fit_cfg.backend == null)
            return error.V18TrainingCacheUnsupported;
        const fitter = try metal.Fitter.fromBackend(fit_cfg.backend.?);
        const transient = try bounded_run.builderTransientBytesForPreparedRoute(original, train.len / original.dim, 256, bounded_tile_queries, cfg, critical_core_policy, geometry_budget_percent);
        if (transient > bounded_builder_transient_bytes) return error.V18TrainingCacheUnsupported;
        const prepared: bounded_run.PreparedView = .{
            .source = original,
            .train = train,
            .heldout = heldout,
            .ranges = ranges,
            .cfg = cfg,
            .external_live_bytes = admitted_external,
            .combined_cap = combined_cap,
            .critical_core_policy = critical_core_policy,
            .postfit_policy = postfit_policy,
            .score_policy = switch (bounded_score_policy) {
                .scalar_reference => .scalar_reference,
                .accelerate_fp64_tiled_v1 => .accelerate_fp64_tiled_v1,
            },
            .validation = .{
                .backend = validation_backend,
                .train_reference_summary_backend = if (train_cache) |*cache| cache.referenceBackend() else null,
                .heldout_reference_summary_backend = if (heldout_cache) |*cache| cache.referenceBackend() else null,
            },
            .budget_profile = if (workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1)
                .e2_formal_p60_no_fallback_v1
            else
                .legacy_e4,
        };
        const acceleration: ?bounded_run.GeometryAcceleration = if (bounded_delta_accelerated)
            .{ .backend = gram_delta.accelerateBackend() orelse return error.AcceleratedDeltaUnsupported }
        else
            null;
        const bounded = if (primary_train_probe) |probe|
            try bounded_run.runPreparedViewBoundedTimedAcceleratedPrimaryProbe(allocator, prepared, fitter, bounded_tile_queries, bounded_builder_transient_bytes, geometry_budget_percent, &phase_timing, acceleration, .{ .context = probe, .invoke_fn = PrimaryTrainProbeCapture.capturePageBacked })
        else
            try bounded_run.runPreparedViewBoundedTimedAcceleratedBudgets(allocator, prepared, fitter, bounded_tile_queries, bounded_builder_transient_bytes, geometry_budget_percent, fallback_geometry_budget_percent, fallback_authorize_context, fallback_authorize_fn, rejected_detail_diagnostic, &phase_timing, acceleration);
        return .{ .allocator = allocator, .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .ranges = ranges, .result = bounded.result, .numerical_policy = numerical_policy, .validation_summary_calls = validation_context.successfulCalls(), .validation_max_batch_rows = validation_context.maximum_batch_rows, .validation_workspace_peak_bytes = validation_context.workspace_peak_bytes, .reference_cache_counters = cache_counters, .phase_timing = phase_timing, .geometry_route = resolved_route, .bounded_evidence = bounded.evidence, .bounded_tsqr_calls = value_context.calls, .bounded_tsqr_workspace_peak_bytes = value_context.workspace_peak_bytes, .bounded_tsqr_cancellation_checks = value_context.cancellation_checks };
    }
    const result = switch (numerical_policy) {
        .legacy_dense_f64 => try replay.runHeadV18WithOptions(allocator, original, train, heldout, ranges, cfg, admitted_external, combined_cap, .{ .phase_timing = &phase_timing, .training_source = true, .mass_strategy = .identity_chunk_coefficients_v18, .validation_backend = validation_backend, .train_reference_summary_backend = if (train_cache) |*cache| cache.referenceBackend() else null, .heldout_reference_summary_backend = if (heldout_cache) |*cache| cache.referenceBackend() else null, .rejected_branch_capture = rejected_branch_capture }),
        .accepted_f32_gram_reuse_v1 => replay.runHeadV18WithOptions(allocator, original, train, heldout, ranges, cfg, admitted_external, combined_cap, .{
            .phase_timing = &phase_timing,
            .training_source = true,
            .mass_strategy = .identity_chunk_coefficients_v18,
            .critical_core_policy = critical_core_policy,
            .postfit_policy = postfit_policy,
            .postfit_workspace_bytes = postfit_workspace,
            .probability_storage = .exact_f32_widened,
            .geometry_storage = .per_query_gram,
            .geometry_budget_percent = geometry_budget_percent,
            .fallback_geometry_budget_percent = fallback_geometry_budget_percent,
            .fallback_authorize_context = fallback_authorize_context,
            .fallback_authorize_fn = fallback_authorize_fn,
            .geometry_chooser_policy = switch (geometry_chooser_policy) {
                .legacy_full_scan => .legacy_full_scan,
                .incremental_gram_v1 => .incremental_gram_v1,
            },
            .training_source_query_batch_rows = if (private_performance_policy == .diagnostic_capture_gram_batch256_v1) 256 else 64,
            .training_capture_backend = capture_backend,
            .training_cache_allocation_authority = flat_cache_allocation_authority orelse if (primary_train_probe) |probe| probe.config.flat_cache_allocation_authority else null,
            .flat_batch_policy = if (private_performance_policy == .diagnostic_capture_gram_batch256_v1) .diagnostic_capture_gram_256_fallback_64_v1 else .established,
            .training_reduction_policy = if (private_performance_policy == .owned_capture_summary_incremental_reduction_v1 or private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 or private_performance_policy == .diagnostic_capture_gram_batch256_v1) .accelerate_fp64_cached_v1 else .scalar_reference,
            .training_reduction_query_batch_rows = reduction_batch_rows,
            .gram_tile_rows = if (private_performance_policy == .diagnostic_capture_gram_batch256_v1) 256 else 64,
            .gram_backend = gram_provider.backend(&gram_context),
            .gram_delta_backend = gram_delta.accelerateBackend() orelse return error.AcceleratedDeltaUnsupported,
            .gram_delta_require_accelerated = true,
            .joint_fixed_original_memo = true,
            .observe_validation_detail = rejected_detail_diagnostic,
            .observe_train_probability_census = train_probability_census,
            .validation_backend = validation_backend,
            .train_reference_summary_backend = if (train_cache) |*cache| cache.referenceBackend() else null,
            .heldout_reference_summary_backend = if (heldout_cache) |*cache| cache.referenceBackend() else null,
            .primary_train_probe_context = if (primary_train_probe) |probe| probe else null,
            .primary_train_probe_heldout_query_rows = if (primary_train_probe) |probe| probe.config.heldout_query_rows else 0,
            .primary_train_probe_fn = if (primary_train_probe != null) PrimaryTrainProbeCapture.capture else null,
            .rejected_branch_capture = rejected_branch_capture,
        }) catch |err| return admissionFailure("replay", err, caller_external_live_bytes, normalized_and_ranges, postfit_workspace, combined_cap, train.len / original.dim, original.rows, original.dim, ranges.len),
    };
    return .{ .allocator = allocator, .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .ranges = ranges, .result = result, .numerical_policy = numerical_policy, .gram_backend_calls = gram_context.successfulCalls(), .validation_summary_calls = validation_context.successfulCalls(), .validation_max_batch_rows = validation_context.maximum_batch_rows, .validation_workspace_peak_bytes = validation_context.workspace_peak_bytes, .reference_cache_counters = cache_counters, .phase_timing = phase_timing, .geometry_route = .flat_gram_only };
}

pub fn runHead(
    allocator: std.mem.Allocator,
    head: checkpoint.Head,
    plan: engine.HeadPlan,
    train: []const f64,
    heldout: []const f64,
    scale: f64,
    softcap: f64,
    fit_cfg: solver.Config,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    numerical_policy: engine.Options.AdaptiveV18NumericalPolicy,
    geometry_route: engine.Options.AdaptiveV18GeometryRoute,
    bounded_tile_queries: usize,
    bounded_builder_transient_bytes: usize,
    bounded_delta_accelerated: bool,
    bounded_tsqr_value_backend: bool,
    geometry_budget_percent: u8,
    fallback_geometry_budget_percent: u8,
    fallback_authorize_context: ?*anyopaque,
    fallback_authorize_fn: ?*const fn (?*anyopaque) bool,
    validation_policy: engine.Options.IndependentValidationPolicy,
    reference_cache_policy: engine.Options.V18ReferenceSummaryCachePolicy,
    geometry_chooser_policy: engine.Options.GeometryChooserPolicy,
    private_performance_policy: engine.Options.PrivatePerformancePolicy,
    critical_core_policy: replay.CriticalCorePolicy,
    postfit_policy: replay.PostfitPolicy,
    rejected_detail_diagnostic: bool,
    train_probability_census: bool,
    rejected_branch_capture: ?replay.RejectedBranchCapture,
) !Run {
    return runHeadImpl(allocator, head, plan, train, heldout, scale, softcap, fit_cfg, caller_external_live_bytes, combined_cap, numerical_policy, geometry_route, bounded_tile_queries, bounded_builder_transient_bytes, bounded_delta_accelerated, bounded_tsqr_value_backend, .standard, .scalar_reference, geometry_budget_percent, fallback_geometry_budget_percent, fallback_authorize_context, fallback_authorize_fn, validation_policy, reference_cache_policy, geometry_chooser_policy, private_performance_policy, critical_core_policy, postfit_policy, rejected_detail_diagnostic, train_probability_census, null, null, rejected_branch_capture);
}

fn runHeadWithAllocationAuthority(
    allocator: std.mem.Allocator,
    head: checkpoint.Head,
    plan: engine.HeadPlan,
    train: []const f64,
    heldout: []const f64,
    scale: f64,
    softcap: f64,
    fit_cfg: solver.Config,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    options: *const engine.Options,
    primary_percent: u8,
    fallback_percent: u8,
    fallback_context: ?*anyopaque,
    fallback_fn: ?*const fn (?*anyopaque) bool,
    selected_private_policy: engine.Options.PrivatePerformancePolicy,
    selected_critical_policy: replay.CriticalCorePolicy,
    rejected_branch_capture: ?replay.RejectedBranchCapture,
    authority: ?@import("attention_match_joint_training_source.zig").AllocationAuthority,
    route: engine.Options.AdaptiveV18GeometryRoute,
) !Run {
    const wide = route == .bounded_page_wide_cache_v1;
    return runHeadImpl(allocator, head, plan, train, heldout, scale, softcap, fit_cfg, caller_external_live_bytes, combined_cap, options.adaptive_v18_numerical_policy, route, if (wide) 64 else options.adaptive_v18_bounded_tile_queries, if (wide) 256 * 1024 * 1024 else options.adaptive_v18_bounded_builder_transient_bytes, if (wide) true else options.adaptive_v18_bounded_delta_accelerated, if (wide) true else options.adaptive_v18_bounded_tsqr_value_backend, options.adaptive_v18_workspace_profile, options.adaptive_v18_bounded_score_policy, primary_percent, fallback_percent, fallback_context, fallback_fn, options.independent_validation_policy, options.v18_reference_summary_cache_policy, options.geometry_chooser_policy, selected_private_policy, selected_critical_policy, options.adaptive_v18_postfit_policy, options.adaptive_v18_rejected_detail_diagnostic, options.adaptive_v18_train_probability_census, authority, null, rejected_branch_capture);
}

/// Executes the real p91 TRAIN preparation and returns its frozen divisor-2
/// candidate. HHO values are deliberately absent; only their authenticated
/// row count is forwarded to planner admission.
pub fn runHeadWithPrimaryProbe(
    allocator: std.mem.Allocator,
    head: checkpoint.Head,
    plan: engine.HeadPlan,
    train: []const f64,
    scale: f64,
    softcap: f64,
    fit_cfg: solver.Config,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    numerical_policy: engine.Options.AdaptiveV18NumericalPolicy,
    geometry_route: engine.Options.AdaptiveV18GeometryRoute,
    bounded_tile_queries: usize,
    bounded_builder_transient_bytes: usize,
    bounded_delta_accelerated: bool,
    bounded_tsqr_value_backend: bool,
    validation_policy: engine.Options.IndependentValidationPolicy,
    reference_cache_policy: engine.Options.V18ReferenceSummaryCachePolicy,
    geometry_chooser_policy: engine.Options.GeometryChooserPolicy,
    private_performance_policy: engine.Options.PrivatePerformancePolicy,
    postfit_policy: replay.PostfitPolicy,
    rejected_detail_diagnostic: bool,
    train_probability_census: bool,
    probe_config: PrimaryTrainProbeConfig,
) !primary_probe.Owned {
    if (probe_config.heldout_query_rows == 0 or numerical_policy != .accepted_f32_gram_reuse_v1 or (geometry_route != .flat_gram_only and geometry_route != .bounded_page_wide_cache_v1))
        return error.InvalidFitConfig;
    var capture: PrimaryTrainProbeCapture = .{ .config = probe_config };
    defer capture.deinit();
    var unexpected = runHeadImpl(allocator, head, plan, train, &.{}, scale, softcap, fit_cfg, caller_external_live_bytes, combined_cap, numerical_policy, geometry_route, bounded_tile_queries, bounded_builder_transient_bytes, bounded_delta_accelerated, bounded_tsqr_value_backend, .resident_native_hybrid_2g_v1, if (probe_config.accelerate_bounded_score) .accelerate_fp64_tiled_v1 else .scalar_reference, 91, 0, null, null, validation_policy, reference_cache_policy, geometry_chooser_policy, private_performance_policy, probe_config.critical_policy, postfit_policy, rejected_detail_diagnostic, train_probability_census, null, &capture, null) catch |err| switch (err) {
        error.TrainPrimaryProbeExported => return capture.take(),
        else => return err,
    };
    unexpected.deinit();
    return error.InvalidFitConfig;
}

test "primary TRAIN probe capture requires typed exit and transfers one owner" {
    var capture: PrimaryTrainProbeCapture = .{ .config = .{
        .heldout_query_rows = 64,
        .critical_policy = .train_probability_gte_0_025_no_top1_v1,
        .postfit_workspace_bytes = 1,
        .validation_backend = undefined,
    } };
    defer capture.deinit();
    try std.testing.expectError(error.InvalidFitConfig, capture.take());
    const source: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .bias = &.{5}, .rows = 1, .dim = 2, .scale = 1, .softcap = 0 };
    const source_digest = PrimaryTrainProbeCapture.digestSource(source);
    const train_digest = PrimaryTrainProbeCapture.digestTrain(&.{ 6, 7 }, 2);
    try std.testing.expect(!std.mem.eql(u8, &source_digest, &train_digest));
    try std.testing.expect(!std.mem.allEqual(u8, &source_digest, 0));
    try std.testing.expect(!std.mem.allEqual(u8, &train_digest, 0));
}

pub const PhysicalBudgetContext = struct {
    fallback_claims: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),

    fn claim(self: *@This(), maximum: u8) bool {
        var current = self.fallback_claims.load(.acquire);
        while (current < maximum) {
            if (self.fallback_claims.cmpxchgWeak(current, current + 1, .acq_rel, .acquire)) |observed| current = observed else return true;
        }
        return false;
    }
};

const FallbackAuthorization = struct {
    context: *PhysicalBudgetContext,
    maximum: u8,

    fn authorize(raw: ?*anyopaque) bool {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        return self.context.claim(self.maximum);
    }
};

test "fallback authorization preserves the global physical-budget claim limit" {
    var context: PhysicalBudgetContext = .{};
    var authorization: FallbackAuthorization = .{ .context = &context, .maximum = 3 };
    try std.testing.expect(FallbackAuthorization.authorize(&authorization));
    try std.testing.expect(FallbackAuthorization.authorize(&authorization));
    try std.testing.expect(FallbackAuthorization.authorize(&authorization));
    try std.testing.expect(!FallbackAuthorization.authorize(&authorization));
}

fn belongs(batch: engine.ReferenceBatch, source: engine.SourceMetadata, layout: engine.Layout, donor: usize, head: usize) bool {
    return source.belongs(layout, batch, donor, head);
}

const RejectedCaptureAdapter = struct {
    allocator: std.mem.Allocator,
    capture: engine.RejectedHeadCapture,
    donor_layer: u32,
    kv_head: usize,
    logical_length: u64,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    scale: f64,
    softcap: f64,
    original_positions: []const u64,
    original_coverage: []const u64,
    train: []const f64,
    heldout: []const f64,
    candidate_positions: []u64,
    candidate_coverage: []u64,
    train_positions: []u64,
    heldout_positions: []u64,
    consumer_layers: []u32,
    query_heads: []u32,
    train_counts: []u32,
    heldout_counts: []u32,

    fn retainedBytes(source_rows: usize, train_rows: usize, heldout_rows: usize, descriptors: usize) !usize {
        var total = try std.math.mul(usize, source_rows, 2 * @sizeOf(u64));
        total = try std.math.add(usize, total, try std.math.mul(usize, try std.math.add(usize, train_rows, heldout_rows), @sizeOf(u64)));
        return std.math.add(usize, total, try std.math.mul(usize, descriptors, 4 * @sizeOf(u32)));
    }

    fn init(allocator: std.mem.Allocator, capture: engine.RejectedHeadCapture, source: engine.SourceMetadata, head: checkpoint.Head, donor: usize, head_index: usize, plan: engine.HeadPlan, recent_rows: usize, chunk_rows: usize, batches: []const engine.ReferenceBatch, layout: engine.Layout, train: []const f64, heldout: []const f64) !RejectedCaptureAdapter {
        const donor_layer = try source.donorLayer(donor);
        var count: usize = 0;
        var train_rows: usize = 0;
        var heldout_rows: usize = 0;
        for (batches) |batch| if (belongs(batch, source, layout, donor, head_index)) {
            count = try std.math.add(usize, count, 1);
            train_rows = try std.math.add(usize, train_rows, batch.train_positions.len);
            heldout_rows = try std.math.add(usize, heldout_rows, batch.heldout_positions.len);
        };
        if (count == 0 or try std.math.mul(usize, train_rows, head.dim) != train.len or try std.math.mul(usize, heldout_rows, head.dim) != heldout.len) return error.InvalidCompactionQueries;
        const candidate_positions = try allocator.alloc(u64, head.positions.len);
        errdefer allocator.free(candidate_positions);
        const candidate_coverage = try allocator.alloc(u64, head.positions.len);
        errdefer allocator.free(candidate_coverage);
        const train_positions = try allocator.alloc(u64, train_rows);
        errdefer allocator.free(train_positions);
        const heldout_positions = try allocator.alloc(u64, heldout_rows);
        errdefer allocator.free(heldout_positions);
        const consumer_layers = try allocator.alloc(u32, count);
        errdefer allocator.free(consumer_layers);
        const query_heads = try allocator.alloc(u32, count);
        errdefer allocator.free(query_heads);
        const train_counts = try allocator.alloc(u32, count);
        errdefer allocator.free(train_counts);
        const heldout_counts = try allocator.alloc(u32, count);
        errdefer allocator.free(heldout_counts);
        var qi: usize = 0;
        var ti: usize = 0;
        var hi: usize = 0;
        for (batches) |batch| if (belongs(batch, source, layout, donor, head_index)) {
            @memcpy(train_positions[ti..][0..batch.train_positions.len], batch.train_positions);
            @memcpy(heldout_positions[hi..][0..batch.heldout_positions.len], batch.heldout_positions);
            consumer_layers[qi] = batch.consumer_layer;
            query_heads[qi] = batch.query_head;
            train_counts[qi] = @intCast(batch.train_positions.len);
            heldout_counts[qi] = @intCast(batch.heldout_positions.len);
            ti += batch.train_positions.len;
            hi += batch.heldout_positions.len;
            qi += 1;
        };
        return .{ .allocator = allocator, .capture = capture, .donor_layer = donor_layer, .kv_head = head_index, .logical_length = source.logical_length, .prefix_rows = plan.prefix_rows, .recent_rows = recent_rows, .chunk_rows = chunk_rows, .scale = layout.scales_by_donor[donor], .softcap = layout.softcaps_by_donor[donor], .original_positions = head.positions, .original_coverage = head.coverage_ends, .train = train, .heldout = heldout, .candidate_positions = candidate_positions, .candidate_coverage = candidate_coverage, .train_positions = train_positions, .heldout_positions = heldout_positions, .consumer_layers = consumer_layers, .query_heads = query_heads, .train_counts = train_counts, .heldout_counts = heldout_counts };
    }

    fn deinit(self: *RejectedCaptureAdapter) void {
        inline for (.{ "candidate_positions", "candidate_coverage", "train_positions", "heldout_positions", "consumer_layers", "query_heads", "train_counts", "heldout_counts" }) |field| self.allocator.free(@field(self, field));
        self.* = undefined;
    }

    fn callback(self: *RejectedCaptureAdapter) replay.RejectedBranchCapture {
        return .{ .context = self, .capture_fn = captureBranch };
    }

    fn captureBranch(raw: ?*anyopaque, view: replay.RejectedBranchView) void {
        const self: *RejectedCaptureAdapter = @ptrCast(@alignCast(raw orelse return));
        if (view.divisor != 2 or view.source_indices.len != view.candidate.rows or view.coverage_source_indices.len != view.candidate.rows or view.candidate.rows > self.candidate_positions.len) return;
        for (view.source_indices, view.coverage_source_indices, 0..) |source_index, coverage_index, row| {
            if (source_index >= self.original_positions.len or coverage_index >= self.original_coverage.len) return;
            self.candidate_positions[row] = self.original_positions[source_index];
            self.candidate_coverage[row] = self.original_coverage[coverage_index];
        }
        self.capture.capture_fn(self.capture.context, .{
            .donor_layer = self.donor_layer,
            .kv_head = self.kv_head,
            .retention_divisor = 2,
            .logical_length = self.logical_length,
            .prefix_rows = self.prefix_rows,
            .recent_rows = self.recent_rows,
            .chunk_rows = self.chunk_rows,
            .scale = self.scale,
            .softcap = self.softcap,
            .original = view.original,
            .candidate = view.candidate,
            .original_positions = self.original_positions,
            .original_coverage_ends = self.original_coverage,
            .candidate_positions = self.candidate_positions[0..view.candidate.rows],
            .candidate_coverage_ends = self.candidate_coverage[0..view.candidate.rows],
            .train_queries = self.train,
            .heldout_queries = self.heldout,
            .train_positions = self.train_positions,
            .heldout_positions = self.heldout_positions,
            .query_consumer_layers = self.consumer_layers,
            .query_heads = self.query_heads,
            .query_train_counts = self.train_counts,
            .query_heldout_counts = self.heldout_counts,
            .metrics = view.metrics,
        });
    }
};

test "V18 rejected adapter maps borrowed candidate metadata only during callback" {
    const Sink = struct {
        calls: usize = 0,
        fn capture(raw: ?*anyopaque, view: engine.RejectedHeadFixtureView) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            std.testing.expectEqualSlices(u64, &.{4}, view.candidate_positions) catch unreachable;
            std.testing.expectEqualSlices(u64, &.{8}, view.candidate_coverage_ends) catch unreachable;
            std.testing.expectEqual(@as(usize, 2), view.retention_divisor) catch unreachable;
        }
    };
    const original_keys = [_]f64{ 1, 2 };
    const original_values = [_]f64{ 3, 4 };
    const original_bias = [_]f64{ 0, 0 };
    const candidate_keys = [_]f64{2};
    const candidate_values = [_]f64{4};
    const candidate_bias = [_]f64{0};
    const original: solver.Head = .{ .keys = &original_keys, .values = &original_values, .bias = &original_bias, .rows = 2, .dim = 1, .scale = 1, .softcap = 0 };
    const candidate: solver.Head = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &candidate_bias, .rows = 1, .dim = 1, .scale = 1, .softcap = 0 };
    var sink: Sink = .{};
    var candidate_positions: [2]u64 = undefined;
    var candidate_coverage: [2]u64 = undefined;
    const original_positions = [_]u64{ 0, 4 };
    const original_coverage = [_]u64{ 1, 8 };
    const train = [_]f64{1};
    const heldout = [_]f64{2};
    var train_positions = [_]u64{8};
    var heldout_positions = [_]u64{9};
    var consumer_layers = [_]u32{5};
    var query_heads = [_]u32{0};
    var train_counts = [_]u32{1};
    var heldout_counts = [_]u32{1};
    var adapter: RejectedCaptureAdapter = .{ .allocator = std.testing.allocator, .capture = .{ .context = &sink, .capture_fn = Sink.capture }, .donor_layer = 5, .kv_head = 0, .logical_length = 8, .prefix_rows = 1, .recent_rows = 2, .chunk_rows = 2, .scale = 1, .softcap = 0, .original_positions = &original_positions, .original_coverage = &original_coverage, .train = &train, .heldout = &heldout, .candidate_positions = &candidate_positions, .candidate_coverage = &candidate_coverage, .train_positions = &train_positions, .heldout_positions = &heldout_positions, .consumer_layers = &consumer_layers, .query_heads = &query_heads, .train_counts = &train_counts, .heldout_counts = &heldout_counts };
    const good_source = [_]usize{1};
    const good_coverage = [_]usize{1};
    RejectedCaptureAdapter.captureBranch(&adapter, .{ .divisor = 2, .original = original, .candidate = candidate, .source_indices = &good_source, .coverage_source_indices = &good_coverage, .train_queries = &.{1}, .heldout_queries = &.{2}, .metrics = .{} });
    try std.testing.expectEqual(@as(usize, 1), sink.calls);
    const bad_source = [_]usize{2};
    RejectedCaptureAdapter.captureBranch(&adapter, .{ .divisor = 2, .original = original, .candidate = candidate, .source_indices = &bad_source, .coverage_source_indices = &good_coverage, .train_queries = &.{1}, .heldout_queries = &.{2}, .metrics = .{} });
    try std.testing.expectEqual(@as(usize, 1), sink.calls);
    try std.testing.expectEqual(@as(usize, 64), try RejectedCaptureAdapter.retainedBytes(2, 1, 1, 1));
}

fn fittedCandidateRun(run: *const Run) !?[2]usize {
    if (!run.result.accepted) return null;
    var branch_index: ?usize = null;
    for (run.result.branches, 0..) |branch, i| if (branch.divisor == run.result.chosen_divisor) {
        branch_index = i;
        break;
    };
    const branch = run.result.branches[branch_index orelse return error.InvalidAttentionShape];
    if (branch.geometry_targets.len != run.ranges.len) return error.InvalidAttentionShape;
    var source_first: ?usize = null;
    var source_end: usize = 0;
    for (run.ranges, branch.geometry_targets) |range, target| {
        if (target == range.rows) {
            if (source_first != null) break;
            continue;
        }
        if (source_first == null) {
            source_first = range.start;
            source_end = range.start + range.rows;
        } else if (range.start == source_end) {
            source_end += range.rows;
        } else break;
    }
    const first_source = source_first orelse return null;
    var first: ?usize = null;
    var end: usize = 0;
    for (run.result.coverage_source_indices, 0..) |source_i, candidate_i| {
        if (source_i >= first_source and source_i < source_end) {
            if (first == null) first = candidate_i;
            end = candidate_i + 1;
        } else if (first != null and source_i >= source_end) break;
    }
    return if (first) |begin| .{ begin, end } else error.InvalidAttentionShape;
}

const SourceEnvelope = struct { first: usize, end: usize };

const CodecEnvelopeMode = enum { first_fitted_run, full_adaptive_envelope };

/// Rotated4 can represent exact rows between packed spans. FP8 has one
/// contiguous wire range, so it may widen only before that range exists.
fn codecEnvelopeMode(fp8: bool, rotated4: bool, prior_fp8_rows: usize) !CodecEnvelopeMode {
    if (fp8 and rotated4) return error.InvalidFitConfig;
    if (rotated4 or (fp8 and prior_fp8_rows == 0)) return .full_adaptive_envelope;
    return .first_fitted_run;
}

/// Returns the source-row envelope from the first fitted island through the
/// last fitted island, including identity-preserved ranges between them. A
/// historical source gap fails closed because it cannot be represented as a
/// contiguous codec candidate without changing row meaning.
fn adaptiveCodecSourceEnvelope(ranges: []const admission.Range, targets: []const usize) !?SourceEnvelope {
    if (ranges.len == 0 or ranges.len != targets.len) return error.InvalidAttentionShape;
    var source_first: ?usize = null;
    var source_end: usize = 0;
    var prior_end: ?usize = null;
    for (ranges, targets) |range, target| {
        if (range.rows == 0 or target == 0 or target > range.rows) return error.InvalidAttentionShape;
        const end = try std.math.add(usize, range.start, range.rows);
        if (prior_end) |previous| if (range.start != previous) return null;
        prior_end = end;
        if (target != range.rows) {
            if (source_first == null) source_first = range.start;
            source_end = end;
        }
    }
    return if (source_first) |first| .{ .first = first, .end = source_end } else null;
}

fn envelopeIntersectsPacked(compressed: checkpoint.packed_storage.Packed, first: usize, end: usize) !bool {
    if (first > end) return error.InvalidAttentionShape;
    for (first..end) |row| switch (compressed.index(row)) {
        .compressed => return true,
        .exact => {},
    };
    return false;
}

fn fittedCandidateAdaptiveEnvelope(run: *const Run, result: *const checkpoint.Head) !?[2]usize {
    if (!run.result.accepted) return null;
    var branch_index: ?usize = null;
    for (run.result.branches, 0..) |branch, i| if (branch.divisor == run.result.chosen_divisor) {
        branch_index = i;
        break;
    };
    const branch = run.result.branches[branch_index orelse return error.InvalidAttentionShape];
    if (branch.geometry_targets.len != run.ranges.len) return error.InvalidAttentionShape;
    const envelope = (try adaptiveCodecSourceEnvelope(run.ranges, branch.geometry_targets)) orelse return fittedCandidateRun(run);
    var first: ?usize = null;
    var end: usize = 0;
    for (run.result.coverage_source_indices, 0..) |source_i, candidate_i| {
        if (source_i >= envelope.first and source_i < envelope.end) {
            if (first == null) first = candidate_i;
            end = candidate_i + 1;
        } else if (first != null and source_i >= envelope.end) break;
    }
    const begin = first orelse return error.InvalidAttentionShape;
    if (try envelopeIntersectsPacked(result.compressed, begin, end)) return fittedCandidateRun(run);
    return .{ begin, end };
}

test "adaptive codec envelope includes identity gaps and excludes protected prefix and recent tail" {
    const ranges = [_]admission.Range{
        .{ .start = 128, .rows = 640 },
        .{ .start = 768, .rows = 640 },
        .{ .start = 1408, .rows = 640 },
    };
    const envelope = (try adaptiveCodecSourceEnvelope(&ranges, &.{ 320, 640, 256 })).?;
    try std.testing.expectEqual(@as(usize, 128), envelope.first);
    try std.testing.expectEqual(@as(usize, 2048), envelope.end);
}

test "adaptive codec envelope rejects history gaps and all identity geometry" {
    const gapped = [_]admission.Range{ .{ .start = 128, .rows = 256 }, .{ .start = 640, .rows = 256 } };
    try std.testing.expect((try adaptiveCodecSourceEnvelope(&gapped, &.{ 64, 64 })) == null);
    const contiguous = [_]admission.Range{ .{ .start = 128, .rows = 256 }, .{ .start = 384, .rows = 256 } };
    try std.testing.expect((try adaptiveCodecSourceEnvelope(&contiguous, &.{ 256, 256 })) == null);
}

test "adaptive codec envelope fails closed on prior packed overlap" {
    const spans = [_]checkpoint.packed_storage.Span{.{ .first = 383, .rows = 2, .offset = 0 }};
    const compressed: checkpoint.packed_storage.Packed = .{ .version = 1, .spans = &spans };
    try std.testing.expect(!(try envelopeIntersectsPacked(compressed, 128, 383)));
    try std.testing.expect(try envelopeIntersectsPacked(compressed, 128, 896));
    try std.testing.expect(!(try envelopeIntersectsPacked(compressed, 385, 896)));
}

test "adaptive codec envelope widens rotated4 and only initial FP8" {
    try std.testing.expectEqual(CodecEnvelopeMode.first_fitted_run, try codecEnvelopeMode(false, false, 0));
    try std.testing.expectEqual(CodecEnvelopeMode.full_adaptive_envelope, try codecEnvelopeMode(true, false, 0));
    try std.testing.expectEqual(CodecEnvelopeMode.first_fitted_run, try codecEnvelopeMode(true, false, 64));
    try std.testing.expectEqual(CodecEnvelopeMode.full_adaptive_envelope, try codecEnvelopeMode(false, true, 0));
    try std.testing.expectError(error.InvalidFitConfig, codecEnvelopeMode(true, true, 0));
}

const ChunkStats = struct { attempted: usize, completed: usize };

/// Maps final-branch work into the established publication counters. Geometry
/// scoring is intentionally separate. Branches with final metrics expose all
/// fitted-range attempts; only the chosen accepted branch is completed.
fn adaptiveChunkStats(range_count: usize, accepted: bool, chosen_divisor: usize, scored_candidates_reused: bool, branches: []const replay.Branch) !ChunkStats {
    // A returned run proves all three training score levels completed.
    var attempted = try std.math.mul(usize, range_count, 3);
    var completed: usize = 0;
    for (branches) |branch| {
        if (branch.geometry_fitted_ranges == 0) continue;
        if (branch.mass_error != null and branch.output_rmse != null) {
            // Reuse composes and validates the retained scored packets; it does
            // not dispatch another selection/value fit for the final branch.
            if (!scored_candidates_reused)
                attempted = try std.math.add(usize, attempted, branch.geometry_fitted_ranges);
            if (accepted and branch.divisor == chosen_divisor and std.mem.eql(u8, branch.rejection, "none"))
                completed = try std.math.add(usize, completed, branch.geometry_fitted_ranges);
        }
    }
    return .{ .attempted = attempted, .completed = completed };
}

/// Callback installed only by the private typed v18 quality policy. The common
/// engine retains task ordering, worker ownership, source/candidate isolation,
/// and publication. Query conversion is charged before allocation/backend use.
pub fn engineCallback(raw: ?*anyopaque, allocator: std.mem.Allocator, source: engine.SourceMetadata, head: checkpoint.Head, donor: usize, head_index: usize, batches: []const engine.ReferenceBatch, layout: engine.Layout, packed_plan: [4]usize, options: *const engine.Options, resident: usize) anyerror!checkpoint.Head {
    if (options.stats) |stats| {
        stats.fp64_validation_policy = options.independent_validation_policy;
        stats.geometry_chooser_policy = options.geometry_chooser_policy;
    }
    const donor_layer = try source.donorLayer(donor);
    var train_len: usize = 0;
    var heldout_len: usize = 0;
    var consumer_count: usize = 0;
    for (batches) |batch| if (belongs(batch, source, layout, donor, head_index)) {
        consumer_count = try std.math.add(usize, consumer_count, 1);
        train_len = try std.math.add(usize, train_len, batch.train.len);
        heldout_len = try std.math.add(usize, heldout_len, batch.heldout.len);
        if (options.adaptive_v18_train_primary_policy_cross_validation and
            (batch.train_positions.len != train_cv.rows_per_domain * train_cv.domains_per_consumer or
                batch.independent_train_split != train_cv.rows_per_domain or
                batch.independent_train_second_split != train_cv.rows_per_domain * 2 or
                batch.train.len != batch.train_positions.len * head.dim)) return error.InvalidCompactionQueries;
    };
    if (train_len == 0 or heldout_len == 0 or consumer_count == 0) return error.MissingCompactionConsumerQueries;
    const comparison = options.adaptive_v18_train_primary_policy_comparison;
    const query_bytes = try std.math.mul(usize, try std.math.add(usize, train_len, heldout_len), @sizeOf(f64));
    const capture_adapter_bytes = if (options.rejected_head_capture != null)
        try RejectedCaptureAdapter.retainedBytes(head.positions.len, train_len / head.dim, heldout_len / head.dim, consumer_count)
    else
        0;
    const admitted_query_bytes = try std.math.add(usize, query_bytes, capture_adapter_bytes);
    const per_head_cap = try engine.v18AvailableHeadWorkspaceBytes(options.*, resident);
    const plan: engine.HeadPlan = .{ .old_rows = packed_plan[0], .prefix_rows = packed_plan[1], .target_rows = packed_plan[2], .block_rows = packed_plan[3] };
    const before_query_alloc = try std.math.add(usize, admitted_query_bytes, try constructionBytes(head, plan));
    // Count the same exact/gapped runs as `headFitRanges` without allocating.
    // Charging one descriptor per eligible row falsely rejects 32K heads, while
    // ceil(span/chunk) undercounts alternating prior-fitted gaps.
    const exact_ranges = exactFitRangeCount(head, plan, 256) catch return error.V18TrainingCacheUnsupported;
    const authorization_peak = try authorizationPeakBytes(head, exact_ranges, admitted_query_bytes, train_len / head.dim, heldout_len / head.dim, options);
    const reference_cache_bytes = switch (options.v18_reference_summary_cache_policy) {
        .disabled => 0,
        .exact_original_train_hho_v1 => try std.math.add(usize, try reference_cache.retainedBytes(train_len / head.dim, head.dim), try reference_cache.retainedBytes(heldout_len / head.dim, head.dim)),
    };
    const authorized_with_reference_cache = try std.math.add(usize, authorization_peak, reference_cache_bytes);
    const policy_cap = try engine.v18HeadWorkspaceBytesForOptions(options.*);
    if (per_head_cap == 0 or per_head_cap > policy_cap or @max(before_query_alloc, authorized_with_reference_cache) > per_head_cap) {
        if (options.stats) |stats| stats.fp64_reference_cap_fallbacks_before_allocation +|= @intFromBool(options.v18_reference_summary_cache_policy != .disabled);
        return error.V18TrainingCacheUnsupported;
    }
    const train = allocator.alloc(f64, train_len) catch return error.V18TrainingCacheUnsupported;
    defer allocator.free(train);
    var train_at: usize = 0;
    for (batches) |batch| if (belongs(batch, source, layout, donor, head_index)) {
        for (batch.train) |value| {
            train[train_at] = value;
            train_at += 1;
        }
    };
    if (options.stats) |stats| stats.v18_planned_head_workspace_bytes = policy_cap;
    const component_ab = options.component_ab_diagnostic;
    const private_perf = options.private_performance_policy != .legacy;
    if (!component_ab and !private_perf and (options.geometry_chooser_policy != .legacy_full_scan or
        options.independent_validation_policy != .scalar_reference)) return error.V18TrainingScoreUnsupported;
    if (private_perf and ((options.adaptive_v18_geometry_route != .flat_gram_only and
        options.adaptive_v18_geometry_route != .bounded_page_over_96 and
        options.adaptive_v18_geometry_route != .bounded_page_wide_cache_v1) or
        options.independent_validation_policy != .accelerate_fp64_summary_v1))
        return error.V18TrainingScoreUnsupported;
    const primary_percent = if (component_ab)
        @as(u8, 100)
    else if (options.adaptive_v18_primary_budget_percent == 0)
        @as(u8, 95)
    else
        options.adaptive_v18_primary_budget_percent;
    if (component_ab and options.adaptive_v18_primary_budget_percent != 0) return error.V18TrainingScoreUnsupported;
    var fallback_authorization: FallbackAuthorization = undefined;
    const fallback_tuple_present = options.adaptive_v18_fallback_budget_percent != 0 or options.adaptive_v18_max_budget_fallbacks != 0;
    const fallback_tuple_complete = options.adaptive_v18_fallback_budget_percent != 0 and options.adaptive_v18_max_budget_fallbacks != 0;
    if (fallback_tuple_present != fallback_tuple_complete) return error.InvalidFitConfig;
    const fallback_enabled = options.adaptive_v18_primary_budget_percent != 0 and fallback_tuple_complete;
    if (fallback_enabled) fallback_authorization = .{ .context = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig)), .maximum = options.adaptive_v18_max_budget_fallbacks };
    var selected_critical_policy = options.adaptive_v18_critical_core_policy;
    var selected_private_performance_policy = options.private_performance_policy;
    if (comparison) {
        const use_cv = options.adaptive_v18_train_primary_policy_cross_validation;
        var split_owned: ?train_cv.Result = if (use_cv)
            try train_cv.split(allocator, train, head.dim, consumer_count, query_bytes, per_head_cap, options.fit.control)
        else
            null;
        defer if (split_owned) |*owned| owned.deinit();
        const probe_train = if (split_owned) |*owned| owned.fit_queries else train;
        const probe_cv = if (split_owned) |*owned| owned.cv_queries else @as([]const f64, &.{});
        const split_digest = if (split_owned) |*owned| owned.digest else @as([32]u8, @splat(0));
        const split_live_bytes = if (split_owned) |*owned| owned.live_bytes else 0;
        const probe_external = try std.math.add(usize, query_bytes, split_live_bytes);
        if (!fallback_enabled or primary_percent != 91 or options.adaptive_v18_fallback_budget_percent != 95 or options.adaptive_v18_max_budget_fallbacks != 3 or options.adaptive_v18_rejected_detail_diagnostic)
            return error.InvalidFitConfig;
        var probe_validation_context: validation_accelerate.Context = .{};
        const probe_validation_backend = validation_accelerate.backend(&probe_validation_context);
        var fixed = try runHeadWithPrimaryProbe(allocator, head, plan, probe_train, layout.scales_by_donor[donor], layout.softcaps_by_donor[donor], options.fit, probe_external, per_head_cap, options.adaptive_v18_numerical_policy, options.adaptive_v18_geometry_route, options.adaptive_v18_bounded_tile_queries, options.adaptive_v18_bounded_builder_transient_bytes, options.adaptive_v18_bounded_delta_accelerated, options.adaptive_v18_bounded_tsqr_value_backend, options.independent_validation_policy, .disabled, options.geometry_chooser_policy, options.private_performance_policy, options.adaptive_v18_postfit_policy, false, options.adaptive_v18_train_probability_census, .{
            .heldout_query_rows = heldout_len / head.dim,
            .critical_policy = .train_probability_gte_0_025_no_top1_v1,
            .postfit_workspace_bytes = try replay.postfitWorkspaceReserve(probe_train.len / head.dim, head.positions.len, head.positions.len, exact_ranges),
            .validation_backend = probe_validation_backend,
            .selection_train_queries = probe_cv,
            .selection_split_sha256 = split_digest,
        });
        var fixed_live = true;
        defer if (fixed_live) fixed.deinit();
        var residual = try runHeadWithPrimaryProbe(allocator, head, plan, probe_train, layout.scales_by_donor[donor], layout.softcaps_by_donor[donor], options.fit, try std.math.add(usize, probe_external, fixed.candidate_live_bytes), per_head_cap, options.adaptive_v18_numerical_policy, options.adaptive_v18_geometry_route, options.adaptive_v18_bounded_tile_queries, options.adaptive_v18_bounded_builder_transient_bytes, options.adaptive_v18_bounded_delta_accelerated, options.adaptive_v18_bounded_tsqr_value_backend, options.independent_validation_policy, .disabled, options.geometry_chooser_policy, .owned_capture_summary_incremental_reduction_v1, options.adaptive_v18_postfit_policy, false, options.adaptive_v18_train_probability_census, .{
            .heldout_query_rows = heldout_len / head.dim,
            .critical_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2,
            .postfit_workspace_bytes = try replay.postfitWorkspaceReserve(probe_train.len / head.dim, head.positions.len, head.positions.len, exact_ranges),
            .validation_backend = probe_validation_backend,
            .selection_train_queries = probe_cv,
            .selection_split_sha256 = split_digest,
        });
        var residual_live = true;
        defer if (residual_live) residual.deinit();
        const selected = try probe_select.select(.{ fixed.probeCandidate(), residual.probeCandidate() }, .{ .candidate_live_bytes_cap = per_head_cap -| probe_external, .control = options.fit.control });
        selected_critical_policy = switch (selected.chosen_variant) {
            .fixed_probability_0_025_no_top1 => .train_probability_gte_0_025_no_top1_v1,
            .calibrated_floor_0_0025_p91_residual16 => .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2,
        };
        selected_private_performance_policy = switch (selected.chosen_variant) {
            .fixed_probability_0_025_no_top1 => .owned_capture_summary_incremental_reduction_critical025_v1,
            .calibrated_floor_0_0025_p91_residual16 => .owned_capture_summary_incremental_reduction_v1,
        };
        std.debug.print("attention_match_train_primary_policy_probe donor={d} head={d} fixed025_error={d} residual16_error={d} fixed025_candidate_sha256={x} residual16_candidate_sha256={x} chosen={s} evidence_sha256={x} probe_candidates=2 recompute_count=1 heldout_observations=0 frozen_before_heldout=true diagnostic_only=true\n", .{ donor_layer, head_index, fixed.train_metrics.max_relative_mass_error, residual.train_metrics.max_relative_mass_error, fixed.candidate_sha256, residual.candidate_sha256, @tagName(selected.chosen_variant), selected.evidence_sha256 });
        if (options.stats) |stats| {
            stats.v18_train_primary_probe_heads += 1;
            stats.v18_train_primary_probe_candidates += 2;
            stats.v18_train_primary_probe_recomputed_heads += 1;
        }
        std.debug.print("attention_match_train_primary_policy_cv donor={d} head={d} metric_domain={s} fit_rows_per_consumer={d} cv_rows_per_consumer={d} split_sha256={x} heldout_observations=0 frozen_before_heldout=true diagnostic_only=true\n", .{ donor_layer, head_index, @tagName(fixed.metrics_domain), if (split_owned) |*owned| owned.fit_rows_per_consumer else train.len / head.dim / consumer_count, if (split_owned) |*owned| owned.cv_rows_per_consumer else 0, split_digest });
        fixed.deinit();
        fixed_live = false;
        residual.deinit();
        residual_live = false;
        if (split_owned) |*owned| owned.deinit();
        split_owned = null;
        // Both probe owners and the split are gone before HHO allocation below.
        try options.fit.control.check();
    }
    const heldout = allocator.alloc(f64, heldout_len) catch return error.V18TrainingCacheUnsupported;
    defer allocator.free(heldout);
    var heldout_at: usize = 0;
    for (batches) |batch| if (belongs(batch, source, layout, donor, head_index)) for (batch.heldout) |value| {
        heldout[heldout_at] = value;
        heldout_at += 1;
    };
    if (heldout_at != heldout_len) return error.InvalidCompactionQueries;
    var rejected_adapter: ?RejectedCaptureAdapter = if (options.rejected_head_capture) |capture|
        try RejectedCaptureAdapter.init(allocator, capture, source, head, donor, head_index, plan, options.recent_rows, options.chunk_rows, batches, layout, train, heldout)
    else
        null;
    defer if (rejected_adapter) |*adapter| adapter.deinit();
    const rejected_branch_capture = if (rejected_adapter) |*adapter| adapter.callback() else null;
    const final_query_rows = train_len / head.dim;
    const probability_bytes = try std.math.mul(usize, try std.math.mul(usize, final_query_rows, head.positions.len), @sizeOf(f32));
    const output_bytes = try std.math.mul(usize, try std.math.mul(usize, final_query_rows, head.dim), @sizeOf(f64));
    const final_authority = options.adaptive_v18_wide_cache_authority;
    const flat_admission = if (final_authority) |authority| try authority.flatAdmission(probability_bytes, output_bytes, options.fit.control) else null;
    const initial_route: engine.Options.AdaptiveV18GeometryRoute = if (flat_admission) |decision| if (decision.fits) .flat_gram_only else .bounded_page_wide_cache_v1 else options.adaptive_v18_geometry_route;
    if (flat_admission) |decision| if (!decision.fits) std.debug.print("attention_match_final_wide_preflight donor={d} head={d} query_rows={d} source_rows={d} ranges={d} cap={d} required_physical_bytes={d} available_physical_bytes={d} route=bounded_page_wide_cache_v1 diagnostic_only=true\n", .{ donor_layer, head_index, final_query_rows, head.positions.len, exact_ranges, per_head_cap, decision.required_physical_bytes, decision.available_physical_bytes });
    const allocation_authority = if (initial_route == .flat_gram_only and final_authority != null) final_authority.?.allocationAuthority() else null;
    var run = runHeadWithAllocationAuthority(allocator, head, plan, train, heldout, layout.scales_by_donor[donor], layout.softcaps_by_donor[donor], options.fit, admitted_query_bytes, per_head_cap, options, primary_percent, if (fallback_enabled) options.adaptive_v18_fallback_budget_percent else 0, if (fallback_enabled) &fallback_authorization else null, if (fallback_enabled) FallbackAuthorization.authorize else null, selected_private_performance_policy, selected_critical_policy, rejected_branch_capture, allocation_authority, initial_route) catch |err| retry: {
        if (initial_route == .flat_gram_only and final_authority != null and err == error.V18TrainingCacheAuthorityChanged) {
            if (try final_authority.?.consumeBoundedRetry(err, probability_bytes, output_bytes, options.fit.control)) |retry_authority| {
                std.debug.print("attention_match_final_wide_retry donor={d} head={d} query_rows={d} source_rows={d} ranges={d} cap={d} required_physical_bytes={d} refused_available_physical_bytes={d} refreshed_available_physical_bytes={d} route=bounded_page_wide_cache_v1 diagnostic_only=true\n", .{ donor_layer, head_index, final_query_rows, head.positions.len, exact_ranges, per_head_cap, retry_authority.refused.required_physical_bytes, retry_authority.refused.available_physical_bytes, retry_authority.refreshed.available_physical_bytes });
                break :retry try runHeadWithAllocationAuthority(allocator, head, plan, train, heldout, layout.scales_by_donor[donor], layout.softcaps_by_donor[donor], options.fit, admitted_query_bytes, per_head_cap, options, primary_percent, if (fallback_enabled) options.adaptive_v18_fallback_budget_percent else 0, if (fallback_enabled) &fallback_authorization else null, if (fallback_enabled) FallbackAuthorization.authorize else null, selected_private_performance_policy, selected_critical_policy, rejected_branch_capture, null, .bounded_page_wide_cache_v1);
            }
        }
        return err;
    };
    if (options.adaptive_v18_phase_timing_diagnostic or options.adaptive_v18_rejected_detail_diagnostic or options.adaptive_v18_train_probability_census) std.debug.print("attention_match_phase_timing donor={d} head={d} geometry_route={s} accepted={} budget_percent={d} fallback_attempted={} fallback_authorizations={d} phase_count=8 unit=ns cache_prepare_ns={d} local_original_rho_y_ns={d} selection_freeze_ns={d} joint_value_fit_ns={d} rounded_harm_ns={d} gram_construction_ns={d} chooser_ns={d} final_heldout_ns={d} diagnostic_only=true\n", .{
        donor_layer,
        head_index,
        @tagName(run.geometry_route),
        run.result.accepted,
        run.result.geometry_budget_percent,
        run.result.fallback_reuse.fallback_attempted,
        run.result.fallback_reuse.authorization_calls,
        run.phase_timing.cache_prepare_ns,
        run.phase_timing.local_original_rho_y_ns,
        run.phase_timing.selection_freeze_ns,
        run.phase_timing.joint_value_fit_ns,
        run.phase_timing.rounded_harm_ns,
        run.phase_timing.gram_construction_ns,
        run.phase_timing.chooser_ns,
        run.phase_timing.final_heldout_ns,
    });
    if (run.reference_cache_counters.retained_bytes_peak != 0) std.debug.print("attention_match_reference_cache_timing donor={d} head={d} unit=ns train_fills={d} train_hits={d} heldout_fills={d} heldout_hits={d} train_init_ns={d} heldout_init_ns={d} train_fill_ns={d} train_hit_ns={d} heldout_fill_ns={d} heldout_hit_ns={d} scalar_fill_ns={d} accelerated_fill_ns={d} retained_bytes_peak={d} candidate_delegates={d} diagnostic_only=true\n", .{
        donor_layer,                                      head_index,
        run.reference_cache_counters.train_fills,         run.reference_cache_counters.train_hits,
        run.reference_cache_counters.heldout_fills,       run.reference_cache_counters.heldout_hits,
        run.reference_cache_counters.train_init_ns,       run.reference_cache_counters.heldout_init_ns,
        run.reference_cache_counters.train_fill_ns,       run.reference_cache_counters.train_hit_ns,
        run.reference_cache_counters.heldout_fill_ns,     run.reference_cache_counters.heldout_hit_ns,
        run.reference_cache_counters.scalar_fill_ns,      run.reference_cache_counters.accelerated_fill_ns,
        run.reference_cache_counters.retained_bytes_peak, run.reference_cache_counters.candidate_delegates,
    });
    if (run.result.geometry_budget_percent != primary_percent) {
        if (options.stats) |stats| stats.v18_budget_fallback_heads += 1;
    }
    defer run.deinit();
    if (options.adaptive_v18_train_probability_census) {
        const census = run.result.critical_core.train_probability_census;
        if (!census.observed) return error.InvalidFitConfig;
        std.debug.print("attention_match_train_probability_census donor={d} head={d} thresholds={any} total_counts={any} range_count={d} sorted_range_counts_005={any} sorted_range_counts_010={any} sorted_range_counts_025={any} training_only=true diagnostic_only=true\n", .{ donor_layer, head_index, census.thresholds, census.total_counts, census.range_count, census.sorted_range_counts[0][0..census.range_count], census.sorted_range_counts[1][0..census.range_count], census.sorted_range_counts[2][0..census.range_count] });
    }
    if (options.adaptive_v18_rejected_detail_diagnostic) for (run.result.branches) |branch| if (branch.heldout_detail) |detail| {
        std.debug.print("attention_match_rejected_detail donor={d} head={d} divisor={d} worst_query={d} worst_relative_mass_error={d} original_peak_source_row={d} original_peak_probability={d} original_peak_retained={} original_peak_exact_core={} original_peak_train_max_probability={any} original_peak_candidate_weight_ratio={any} original_peak_value_relative_l2={any} candidate_peak_source_row={d} candidate_peak_probability={d} candidate_peak_train_max_probability={any} candidate_peak_weight_ratio={any} frozen_before_heldout=true diagnostic_only=true\n", .{ donor_layer, head_index, branch.divisor, detail.validation.worst_query_index, detail.validation.worst_relative_mass_error, detail.validation.original_peak_row, detail.validation.original_peak_probability, detail.original_peak_retained, detail.original_peak_exact_core, detail.original_peak_train_max_probability, detail.original_peak_candidate_weight_ratio, detail.original_peak_value_relative_l2, detail.candidate_peak_source_index, detail.validation.compact_peak_probability, detail.candidate_peak_train_max_probability, detail.candidate_peak_weight_ratio });
    };
    const expected_postfit_digest = replay.postfitPolicyDigest(options.adaptive_v18_postfit_policy);
    if (options.adaptive_v18_postfit_policy != .disabled) std.debug.print("attention_match_train_postfit donor={d} head={d} policy={s} policy_digest={x} attempts={d} completions={d} workspace_bytes={d} max_train_before={d} max_train_after={d} refinement_outcome={s} refinement_attempted_iterations={d} refinement_accepted_improvements={d} refinement_nnls_sweeps={d} minimax_attempts={d} minimax_completions={d} minimax_applied={d} minimax_scale={d} minimax_scale_min={d} minimax_scale_max={d} exact_train_mass_before={d} exact_train_mass_after={d} exact_train_output_rmse_before={d} exact_train_output_rmse_after={d} accepted={} frozen_before_heldout=true\n", .{ donor, head_index, @tagName(run.result.postfit.policy), run.result.postfit.policy_digest, run.result.postfit.attempts, run.result.postfit.completions, run.result.postfit.workspace_bytes, run.result.postfit.max_abs_train_before, run.result.postfit.max_abs_train_after, @tagName(run.result.postfit.refinement_outcome), run.result.postfit.refinement_attempted_iterations, run.result.postfit.refinement_accepted_improvements, run.result.postfit.refinement_nnls_sweeps, run.result.postfit.minimax_attempts, run.result.postfit.minimax_completions, run.result.postfit.minimax_applied, run.result.postfit.minimax_scale, run.result.postfit.minimax_scale_min, run.result.postfit.minimax_scale_max, run.result.postfit.exact_train_mass_before, run.result.postfit.exact_train_mass_after, run.result.postfit.exact_train_output_rmse_before, run.result.postfit.exact_train_output_rmse_after, run.result.accepted });
    var all_geometry_unavailable = run.result.branches.len != 0;
    for (run.result.branches) |branch| all_geometry_unavailable = all_geometry_unavailable and std.mem.eql(u8, branch.rejection, "geometry_budget_infeasible");
    if (run.result.postfit.policy != options.adaptive_v18_postfit_policy or
        (options.adaptive_v18_postfit_policy == .disabled and (run.result.postfit.attempts != 0 or run.result.postfit.completions != 0)) or
        (options.adaptive_v18_postfit_policy != .disabled and
            (!std.mem.eql(u8, &run.result.postfit.policy_digest, &expected_postfit_digest) or
                (run.result.accepted and (run.result.postfit.attempts != 1 or run.result.postfit.completions != 1)) or
                (!run.result.accepted and ((run.result.postfit.attempts == 0 and !all_geometry_unavailable) or run.result.postfit.completions != run.result.postfit.attempts)))))
        return error.InvalidFitConfig;
    if (selected_critical_policy != .disabled) {
        const critical = run.result.critical_core;
        if (critical.policy != selected_critical_policy or
            !std.mem.eql(u8, &critical.policy_digest, &replay.criticalCorePolicyDigest(critical.policy)) or
            critical.unique_rows > head.positions.len)
            return error.InvalidFitConfig;
        const calibrated = critical.calibration;
        const calibrated_policy_digest = switch (critical.policy) {
            .train_probability_budget_calibrated_floor_0_0025_p91_v1 => @import("attention_match_train_budget_calibrated_critical_threshold.zig").policyDigest(),
            .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => @import("attention_match_train_budget_calibrated_critical_threshold_reserve16.zig").policyDigest(),
            else => @as([32]u8, @splat(0)),
        };
        const is_calibrated = critical.policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1 or critical.policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
        if (is_calibrated and
            (!calibrated.observed or calibrated.critical_rows != critical.unique_rows or calibrated.minimum_middle_rows > calibrated.middle_cap_rows or
                !std.mem.eql(u8, &calibrated.policy_sha256, &calibrated_policy_digest)))
            return error.InvalidFitConfig;
        if (!is_calibrated and calibrated.observed) return error.InvalidFitConfig;
        std.debug.print("attention_match_critical_core donor={d} head={d} policy={s} policy_digest={x} unique_rows={d} train_rows={d} calibrated={} threshold={d} minimum_middle_rows={d} middle_cap_rows={d} calibration_policy_sha256={x} frozen_before_heldout=true\n", .{
            donor_layer, head_index, @tagName(critical.policy), critical.policy_digest, critical.unique_rows, train.len / head.dim, calibrated.observed, calibrated.threshold, calibrated.minimum_middle_rows, calibrated.middle_cap_rows, calibrated.policy_sha256,
        });
    }
    if (options.geometry_chooser_policy == .incremental_gram_v1) {
        const expected = if (run.geometry_route == .bounded_page_over_96 or run.geometry_route == .bounded_page_wide_cache_v1)
            if (primary_percent == 100) @import("attention_match_joint_bounded_tiled_replay.zig").policy else @import("attention_match_joint_bounded_tiled_replay.zig").budget_percent_policy
        else if (primary_percent == 100) @import("attention_match_joint_gram_geometry.zig").incremental_policy else @import("attention_match_joint_gram_geometry.zig").incremental_budget_percent_policy;
        if (!std.mem.eql(u8, run.result.geometry_policy, expected)) {
            std.debug.print("attention_match_geometry_policy_rejected donor={d} head={d} route={s} expected={s} actual={s}\n", .{ donor_layer, head_index, @tagName(run.geometry_route), expected, run.result.geometry_policy });
            return error.V18TrainingScoreUnsupported;
        }
        if (options.stats) |stats| stats.incremental_gram_chooser_calls = try std.math.add(usize, stats.incremental_gram_chooser_calls, run.result.incremental_gram_chooser_calls);
    } else if (std.mem.eql(u8, run.result.geometry_policy, @import("attention_match_joint_gram_geometry.zig").incremental_policy)) {
        return error.V18TrainingScoreUnsupported;
    }
    if (run.geometry_route == .bounded_page_over_96 or run.geometry_route == .bounded_page_wide_cache_v1) {
        const effective_wide = run.geometry_route == .bounded_page_wide_cache_v1;
        const effective_delta = effective_wide or options.adaptive_v18_bounded_delta_accelerated;
        const effective_tsqr = effective_wide or options.adaptive_v18_bounded_tsqr_value_backend;
        const evidence = run.bounded_evidence orelse return error.V18TrainingScoreUnsupported;
        const expected_budget_profile: bounded_run.BudgetProfile = if (options.adaptive_v18_workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1)
            .e2_formal_p60_no_fallback_v1
        else
            .legacy_e4;
        if (evidence.budget_profile != expected_budget_profile or evidence.geometry_budget_percent != run.result.geometry_budget_percent or
            evidence.query_batch_rows != 64 or evidence.probability_storage != .exact_f32_widened or
            run.result.training_source.counters.query_batch_rows != evidence.query_batch_rows or
            run.result.probability_storage != evidence.probability_storage or
            evidence.provider_calls == 0 or evidence.provider_rows == 0 or evidence.workspace_bound_bytes > per_head_cap or
            evidence.scoring_cache_prepares != run.ranges.len or evidence.scoring_cache_selection_hits != run.ranges.len * 3 or
            evidence.scoring_cache_fit_hits != run.ranges.len * 3 or evidence.geometry_cache_requests == 0 or
            evidence.geometry_cache_requests != evidence.geometry_cache_hits + evidence.geometry_cache_loads or
            evidence.range_cache_prepares != evidence.scoring_cache_prepares + evidence.geometry_cache_loads + evidence.postfit_range_cache_prepares or
            evidence.geometry_delta_accelerated != effective_delta or evidence.geometry_delta_fallbacks != 0 or
            (effective_delta and (evidence.geometry_delta_tiles == 0 or evidence.geometry_delta_backend_calls != evidence.geometry_delta_tiles * 2 or evidence.geometry_delta_packed_bytes_peak == 0 or evidence.geometry_delta_staged_bytes_peak == 0)) or
            (!effective_delta and (evidence.geometry_delta_tiles != 0 or evidence.geometry_delta_backend_calls != 0)))
        {
            std.debug.print("attention_match_bounded_evidence_rejected donor={d} head={d} route={s} budget={d}/{d} batch={d}/{d} provider_calls={d} provider_rows={d} workspace={d}/{d} scoring_prepares={d} scoring_selection={d} scoring_fit={d} ranges={d} total_prepares={d} geometry_requests={d} geometry_hits={d} geometry_loads={d} postfit_prepares={d} delta={}/{} delta_tiles={d} delta_calls={d} delta_packed={d} delta_staged={d} delta_fallbacks={d}\n", .{ donor_layer, head_index, @tagName(run.geometry_route), evidence.geometry_budget_percent, run.result.geometry_budget_percent, evidence.query_batch_rows, run.result.training_source.counters.query_batch_rows, evidence.provider_calls, evidence.provider_rows, evidence.workspace_bound_bytes, per_head_cap, evidence.scoring_cache_prepares, evidence.scoring_cache_selection_hits, evidence.scoring_cache_fit_hits, run.ranges.len, evidence.range_cache_prepares, evidence.geometry_cache_requests, evidence.geometry_cache_hits, evidence.geometry_cache_loads, evidence.postfit_range_cache_prepares, evidence.geometry_delta_accelerated, effective_delta, evidence.geometry_delta_tiles, evidence.geometry_delta_backend_calls, evidence.geometry_delta_packed_bytes_peak, evidence.geometry_delta_staged_bytes_peak, evidence.geometry_delta_fallbacks });
            return error.V18TrainingScoreUnsupported;
        }
        if ((effective_tsqr and
            (run.bounded_tsqr_calls == 0 or run.bounded_tsqr_workspace_peak_bytes == 0 or run.bounded_tsqr_workspace_peak_bytes > per_head_cap or run.bounded_tsqr_cancellation_checks == 0)) or
            (!effective_tsqr and
                (run.bounded_tsqr_calls != 0 or run.bounded_tsqr_workspace_peak_bytes != 0 or run.bounded_tsqr_cancellation_checks != 0)))
        {
            std.debug.print("attention_match_bounded_tsqr_evidence_rejected donor={d} head={d} route={s} required={} calls={d} workspace={d}/{d} cancellation_checks={d}\n", .{ donor_layer, head_index, @tagName(run.geometry_route), effective_tsqr, run.bounded_tsqr_calls, run.bounded_tsqr_workspace_peak_bytes, per_head_cap, run.bounded_tsqr_cancellation_checks });
            return error.V18TrainingScoreUnsupported;
        }
        if (options.stats) |stats| {
            stats.v18_bounded_heads += 1;
            stats.v18_bounded_provider_calls +|= evidence.provider_calls;
            stats.v18_bounded_provider_rows +|= evidence.provider_rows;
            stats.v18_bounded_workspace_bound_bytes = @max(stats.v18_bounded_workspace_bound_bytes, evidence.workspace_bound_bytes);
            stats.v18_bounded_scoring_cache_prepares += evidence.scoring_cache_prepares;
            stats.v18_bounded_scoring_selection_hits += evidence.scoring_cache_selection_hits;
            stats.v18_bounded_scoring_fit_hits += evidence.scoring_cache_fit_hits;
            stats.v18_bounded_score_accelerate_calls += evidence.bounded_score_accelerate_calls;
            stats.v18_bounded_score_accelerate_tiles += evidence.bounded_score_accelerate_tiles;
            stats.v18_bounded_score_accelerate_dgemm_calls += evidence.bounded_score_accelerate_dgemm_calls;
            stats.v18_bounded_score_cancellation_checks += evidence.bounded_score_cancellation_checks;
            stats.v18_bounded_range_cache_prepares += evidence.range_cache_prepares;
            stats.v18_bounded_geometry_cache_requests += evidence.geometry_cache_requests;
            stats.v18_bounded_geometry_cache_hits += evidence.geometry_cache_hits;
            stats.v18_bounded_geometry_cache_loads += evidence.geometry_cache_loads;
            stats.v18_bounded_delta_heads += @intFromBool(evidence.geometry_delta_accelerated);
            stats.v18_bounded_delta_tiles += evidence.geometry_delta_tiles;
            stats.v18_bounded_delta_backend_calls += evidence.geometry_delta_backend_calls;
            stats.v18_bounded_delta_packed_bytes_peak = @max(stats.v18_bounded_delta_packed_bytes_peak, evidence.geometry_delta_packed_bytes_peak);
            stats.v18_bounded_delta_staged_bytes_peak = @max(stats.v18_bounded_delta_staged_bytes_peak, evidence.geometry_delta_staged_bytes_peak);
            stats.v18_bounded_delta_fallbacks += evidence.geometry_delta_fallbacks;
            stats.v18_bounded_tsqr_heads += @intFromBool(effective_tsqr);
            stats.v18_bounded_tsqr_calls += run.bounded_tsqr_calls;
            stats.v18_bounded_tsqr_workspace_peak_bytes = @max(stats.v18_bounded_tsqr_workspace_peak_bytes, run.bounded_tsqr_workspace_peak_bytes);
            stats.v18_bounded_tsqr_cancellation_checks += run.bounded_tsqr_cancellation_checks;
        }
    }
    const actual_batch_rows = run.result.training_source.counters.query_batch_rows;
    const diagnostic_batch = options.private_performance_policy == .diagnostic_capture_gram_batch256_v1;
    if (actual_batch_rows != 64 and (!diagnostic_batch or actual_batch_rows != 256)) return error.V18TrainingScoreUnsupported;
    if (options.stats) |stats| {
        stats.v18_primary_budget_percent = options.adaptive_v18_primary_budget_percent;
        if (stats.v18_query_batch_rows != 0 and stats.v18_query_batch_rows != actual_batch_rows)
            return error.V18TrainingScoreUnsupported;
        stats.v18_query_batch_rows = actual_batch_rows;
    }
    var result = try materializeHead(allocator, head, &run);
    errdefer {
        allocator.free(@constCast(result.positions));
        allocator.free(@constCast(result.coverage_ends));
        allocator.free(@constCast(result.keys));
        allocator.free(@constCast(result.values));
        allocator.free(@constCast(result.bias));
        result.compressed.deinit(allocator);
    }
    if (options.fp8_checkpoint or options.rotated4_checkpoint) {
        const candidate_run = switch (try codecEnvelopeMode(options.fp8_checkpoint, options.rotated4_checkpoint, result.fp8.rows)) {
            .full_adaptive_envelope => try fittedCandidateAdaptiveEnvelope(&run, &result),
            .first_fitted_run => try fittedCandidateRun(&run),
        };
        if (candidate_run) |authorized| {
            var codec_detail: engine.CodecDiagnostic = .{};
            var codec_options = options.*;
            const local_detail = codec_options.codec_diagnostic == null and admissionDiagnosticEnabled();
            if (local_detail) codec_options.codec_diagnostic = &codec_detail;
            try engine.authorizeAdaptiveCodec(allocator, .{ .keys = run.keys, .values = run.values, .bias = run.bias, .rows = head.positions.len, .dim = head.dim, .scale = layout.scales_by_donor[donor], .softcap = layout.softcaps_by_donor[donor] }, run.result.head, &result, authorized[0], authorized[1], train, heldout, codec_options);
            if (local_detail) std.debug.print("attention_match_adaptive_codec donor={d} head={d} kind={s} outcome={s} retention={s} eligible_rows={d} packed_rows={d} retained_exact_rows={d} quality_evaluated={} mass_error={d} output_rmse={d} diagnostic_only=true\n", .{ donor_layer, head_index, @tagName(codec_detail.kind), @tagName(codec_detail.outcome), @tagName(codec_detail.rotated4_risk_retention), codec_detail.eligible_rows, codec_detail.packed_rows, codec_detail.retained_exact_rows, codec_detail.metrics.quality_evaluated, codec_detail.metrics.max_relative_mass_error, codec_detail.metrics.relative_output_rmse });
        } else if (run.result.accepted) return error.InvalidAttentionShape;
    } else if (run.result.accepted) if (options.stats) |stats| {
        stats.validated_heads += 1;
        stats.rounded_fit.max_relative_mass_error = run.result.final_mass_error;
        stats.rounded_fit.relative_output_rmse = run.result.final_output_rmse;
        stats.max_head_mass_error = @max(stats.max_head_mass_error, run.result.final_mass_error);
        stats.max_head_output_rmse = @max(stats.max_head_output_rmse, run.result.final_output_rmse);
    };
    // The private numerical route is an execution contract, not telemetry.
    // Validate it even when the caller does not request aggregate stats.
    if (run.numerical_policy == .accepted_f32_gram_reuse_v1 and run.geometry_route == .flat_gram_only) {
        const candidates = try std.math.mul(usize, run.ranges.len, 3);
        const expected_tiles = try std.math.mul(usize, candidates, std.math.divCeil(usize, train.len / head.dim, actual_batch_rows) catch return error.InvalidAttentionShape);
        if (run.result.probability_storage != .exact_f32_widened or run.result.geometry_storage != .per_query_gram or
            run.result.geometry_budget_percent != (if (options.adaptive_v18_primary_budget_percent == 0 or run.result.geometry_budget_percent == primary_percent) primary_percent else options.adaptive_v18_fallback_budget_percent) or run.result.training_source.counters.query_batch_rows != actual_batch_rows or
            !run.result.scored_candidates_reused or run.gram_backend_calls == 0 or
            !run.result.gram_delta.requested or !run.result.gram_delta.accelerated or run.result.gram_delta.tile_rows != actual_batch_rows or
            run.result.gram_delta.tiles != expected_tiles or run.result.gram_delta.backend_calls != expected_tiles * 2 or run.result.gram_delta.fallbacks != 0 or
            !run.result.joint_fixed_memo.enabled or run.result.joint_fixed_memo.builds != run.ranges.len or run.result.joint_fixed_memo.hits != run.ranges.len * 3)
            return error.V18TrainingScoreUnsupported;
    }
    const reduction = run.result.training_reduction;
    if (options.private_performance_policy == .owned_capture_summary_incremental_reduction_v1 or options.private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 or options.private_performance_policy == .diagnostic_capture_gram_batch256_v1) {
        if (run.geometry_route == .bounded_page_over_96 or run.geometry_route == .bounded_page_wide_cache_v1) {
            const bounded_reduction = run.bounded_evidence orelse return error.V18TrainingScoreUnsupported;
            try validateBoundedReductionContract(bounded_reduction, reduction, run.ranges.len, train.len / head.dim);
            std.debug.print("attention_match_bounded_reduction_contract donor={d} head={d} policy={s} score_completions={d} score_query_rows={d} flat_reduction_default=true\n", .{ donor_layer, head_index, bounded_reduction.bounded_reduction_policy, bounded_reduction.bounded_score_completions, bounded_reduction.bounded_score_query_rows });
        } else if (!reduction.accelerated or reduction.local_original_calls == 0 or reduction.rounded_harm_calls == 0 or reduction.tiles == 0 or reduction.backend_calls != try std.math.mul(usize, reduction.tiles, 2) or reduction.workspace_bytes_peak == 0)
            return error.V18TrainingScoreUnsupported;
    } else if (reduction.accelerated or reduction.local_original_calls != 0 or reduction.rounded_harm_calls != 0 or reduction.tiles != 0 or reduction.backend_calls != 0 or reduction.workspace_bytes_peak != 0) return error.V18TrainingScoreUnsupported;
    if (options.stats) |stats| {
        stats.training_reduction_local_original_calls = try std.math.add(usize, stats.training_reduction_local_original_calls, reduction.local_original_calls);
        stats.training_reduction_rounded_harm_calls = try std.math.add(usize, stats.training_reduction_rounded_harm_calls, reduction.rounded_harm_calls);
        stats.training_reduction_tiles = try std.math.add(usize, stats.training_reduction_tiles, reduction.tiles);
        stats.training_reduction_backend_calls = try std.math.add(usize, stats.training_reduction_backend_calls, reduction.backend_calls);
        stats.training_reduction_workspace_peak_bytes = @max(stats.training_reduction_workspace_peak_bytes, reduction.workspace_bytes_peak);
        stats.fp64_validation_summary_calls = try std.math.add(usize, stats.fp64_validation_summary_calls, run.validation_summary_calls);
        stats.fp64_validation_dgemm_calls = try std.math.add(usize, stats.fp64_validation_dgemm_calls, try std.math.mul(usize, run.validation_summary_calls, 2));
        stats.fp64_validation_max_batch_rows = @max(stats.fp64_validation_max_batch_rows, run.validation_max_batch_rows);
        stats.fp64_validation_workspace_peak_bytes = @max(stats.fp64_validation_workspace_peak_bytes, run.validation_workspace_peak_bytes);
        const chunk_stats = try adaptiveChunkStats(run.ranges.len, run.result.accepted, run.result.chosen_divisor, run.result.scored_candidates_reused, run.result.branches);
        stats.attempted_chunks = try std.math.add(usize, stats.attempted_chunks, chunk_stats.attempted);
        stats.completed_chunks = try std.math.add(usize, stats.completed_chunks, chunk_stats.completed);
        for (run.result.branches) |branch| if (!std.mem.eql(u8, branch.rejection, "none")) {
            if (branch.mass_error) |mass| stats.max_rejected_mass_error = @max(stats.max_rejected_mass_error, mass);
            if (branch.output_rmse) |output| stats.max_rejected_output_rmse = @max(stats.max_rejected_output_rmse, output);
        };
        stats.v18_cache_ready_heads += @intFromBool(run.result.training_source.outcome == .ready);
        if (run.numerical_policy == .accepted_f32_gram_reuse_v1) {
            stats.v18_exact_f32_probability_heads += 1;
            stats.v18_scored_reuse_heads += 1;
            if (run.geometry_route == .flat_gram_only) {
                stats.v18_gram_heads += 1;
                stats.v18_gram_backend_calls = try std.math.add(usize, stats.v18_gram_backend_calls, run.result.gram_delta.backend_calls);
            }
        }
        stats.v18_accepted_heads += @intFromBool(run.result.accepted);
        stats.v18_preserved_heads += @intFromBool(!run.result.accepted);
        stats.preserved_heads += @intFromBool(!run.result.accepted);
        stats.v18_cache_phase_peak_bound_bytes = @max(stats.v18_cache_phase_peak_bound_bytes, @max(run.result.training_source.counters.capture_peak_bytes, run.result.training_source.counters.steady_peak_bytes));
        stats.v18_cache_prepare_ns +|= run.phase_timing.cache_prepare_ns;
        stats.v18_selection_freeze_ns +|= run.phase_timing.selection_freeze_ns;
        stats.v18_joint_value_fit_ns +|= run.phase_timing.joint_value_fit_ns;
        stats.v18_gram_construction_ns +|= run.phase_timing.gram_construction_ns;
        stats.v18_chooser_ns +|= run.phase_timing.chooser_ns;
        stats.v18_final_heldout_ns +|= run.phase_timing.final_heldout_ns;
        stats.fp64_reference_train_fills +|= run.reference_cache_counters.train_fills;
        stats.fp64_reference_train_hits +|= run.reference_cache_counters.train_hits;
        stats.fp64_reference_hho_fills +|= run.reference_cache_counters.heldout_fills;
        stats.fp64_reference_hho_hits +|= run.reference_cache_counters.heldout_hits;
        stats.fp64_reference_candidate_delegates +|= run.reference_cache_counters.candidate_delegates;
        stats.fp64_reference_candidate_hits +|= run.reference_cache_counters.candidate_hits;
        stats.fp64_reference_retained_bytes_peak = @max(stats.fp64_reference_retained_bytes_peak, run.reference_cache_counters.retained_bytes_peak);
        stats.fp64_reference_cap_fallbacks_before_allocation +|= run.reference_cache_counters.cap_fallbacks_before_allocation;
        stats.fp64_reference_detailed_observer_bypasses +|= run.reference_cache_counters.detailed_observer_bypasses;
        std.debug.print("attention_match_v18_head donor={d} head={d} original_rows={d} candidate_rows={d} accepted={} chosen_divisor={d} ranges={d} training_score_fits={d} attempted_chunks={d} completed_chunks={d} scored_candidates_reused={} scored_packets_reused={d} reselections_avoided={d} value_fit_dispatches_avoided={d} mass_error={d} output_rmse={d}\n", .{ donor, head_index, head.positions.len, run.result.head.rows, run.result.accepted, run.result.chosen_divisor, run.ranges.len, run.ranges.len * 3, chunk_stats.attempted, chunk_stats.completed, run.result.scored_candidates_reused, run.result.scored_packets_reused, run.result.reselections_avoided, run.result.value_fit_dispatches_avoided, run.result.final_mass_error, run.result.final_output_rmse });
        for (run.result.branches, 0..) |branch, branch_index| {
            std.debug.print("attention_match_v18_branch donor={d} head={d} branch={d} divisor={d} rejection={s} candidate_rows={d} mass_error={?d} output_rmse={?d} fitted_ranges={d} preserved_ranges={d}\n", .{ donor, head_index, branch_index, branch.divisor, branch.rejection, branch.candidate_rows, branch.mass_error, branch.output_rmse, branch.geometry_fitted_ranges, branch.geometry_preserved_ranges });
        }
    }
    return result;
}

/// Test compatibility helper for deliberately partial synthetic checkpoints.
/// Production streaming callers never enter this function: they pass
/// authenticated metadata and the currently leased head directly. Requiring
/// production `checkpoint.State.validate` here would reject the gapped-head
/// fixtures before the adapter behavior they are designed to exercise.
fn engineCallbackFromState(raw: ?*anyopaque, allocator: std.mem.Allocator, source: checkpoint.State, donor: usize, head_index: usize, batches: []const engine.ReferenceBatch, layout: engine.Layout, packed_plan: [4]usize, options: *const engine.Options, resident: usize) anyerror!checkpoint.Head {
    if (!builtin.is_test) return error.InvalidFitConfig;
    try options.fit.control.check();
    if (donor >= source.donors.len or head_index >= source.donors[donor].heads.len or
        source.layer_donors.len == 0 or source.donors.len == 0) return error.InvalidKvCheckpoint;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-partial-test-fixture.v1\x00");
    hash.update(&source.identity.weights);
    hash.update(&source.identity.tokenizer);
    hash.update(&source.identity.template);
    hash.update(&source.identity.layout);
    hash.update(std.mem.asBytes(&source.logical_length));
    hash.update(std.mem.sliceAsBytes(source.layer_donors));
    var census: usize = 0;
    for (source.donors, 0..) |fixture_donor, donor_index| {
        hash.update(std.mem.asBytes(&donor_index));
        hash.update(std.mem.asBytes(&fixture_donor.layer));
        hash.update(std.mem.asBytes(&fixture_donor.window));
        for (fixture_donor.heads, 0..) |fixture_head, fixture_head_index| {
            hash.update(std.mem.asBytes(&fixture_head_index));
            hash.update(std.mem.sliceAsBytes(fixture_head.positions));
            hash.update(std.mem.sliceAsBytes(fixture_head.coverage_ends));
            hash.update(std.mem.sliceAsBytes(fixture_head.keys));
            hash.update(std.mem.sliceAsBytes(fixture_head.values));
            hash.update(std.mem.sliceAsBytes(fixture_head.bias));
            census = try std.math.add(usize, census, std.mem.sliceAsBytes(fixture_head.positions).len);
            census = try std.math.add(usize, census, std.mem.sliceAsBytes(fixture_head.coverage_ends).len);
            census = try std.math.add(usize, census, std.mem.sliceAsBytes(fixture_head.keys).len);
            census = try std.math.add(usize, census, std.mem.sliceAsBytes(fixture_head.values).len);
            census = try std.math.add(usize, census, std.mem.sliceAsBytes(fixture_head.bias).len);
        }
    }
    var metadata: engine.SourceMetadata = .{
        .identity = source.identity,
        .logical_length = source.logical_length,
        .tokens = source.tokens,
        .logits = source.logits,
        .layer_donors = source.layer_donors,
        .full_donors = source.donors,
        .structure_sha256 = undefined,
        .state_sha256 = hash.finalResult(),
        .source_census_physical_bytes = census,
    };
    metadata.structure_sha256 = engine.sourceMetadataStructureDigest(metadata);
    return engineCallback(raw, allocator, metadata, source.donors[donor].heads[head_index], donor, head_index, batches, layout, packed_plan, options, resident);
}

test "bounded reduction contract rejects missing scores and fabricated flat reduction" {
    var evidence: bounded_run.Evidence = .{
        .train_queries = 5,
        .query_batch_rows = 5,
        .probability_storage = .exact_f32_widened,
        .provider_calls = 1,
        .provider_rows = 1,
        .workspace_bound_bytes = 1,
        .scoring_cache_prepares = 2,
        .scoring_cache_selection_hits = 6,
        .scoring_cache_fit_hits = 6,
        .range_cache_prepares = 2,
        .range_cache_selection_hits = 6,
        .range_cache_fit_hits = 6,
        .geometry_cache_requests = 1,
        .geometry_cache_hits = 1,
        .geometry_cache_loads = 0,
        .bounded_score_completions = 6,
        .bounded_score_query_rows = 30,
    };
    try validateBoundedReductionContract(evidence, .{}, 2, 5);
    evidence.bounded_score_completions = 5;
    try std.testing.expectError(error.V18TrainingScoreUnsupported, validateBoundedReductionContract(evidence, .{}, 2, 5));
    evidence.bounded_score_completions = 6;
    try std.testing.expectError(error.V18TrainingScoreUnsupported, validateBoundedReductionContract(evidence, .{ .policy = .accelerate_fp64_cached_v1, .accelerated = true, .local_original_calls = 1, .rounded_harm_calls = 1, .tiles = 1, .backend_calls = 2, .workspace_bytes_peak = 1 }, 2, 5));
}

test "coverage-aware v18 ranges preserve fitted rows and logical gaps" {
    const positions = [_]u64{ 0, 1, 2, 8, 9, 10, 20 };
    const ends = [_]u64{ 1, 2, 8, 9, 10, 11, 21 };
    const bias = [_]f32{ 0, 0, 0.5, 0, 0, 0, 0 };
    const values: [positions.len]f32 = @splat(0);
    const head: checkpoint.Head = .{ .dim = 1, .compacted_until = 8, .positions = &positions, .coverage_ends = &ends, .keys = &values, .values = &values, .bias = &bias };
    const ranges = try engine.headFitRanges(std.testing.allocator, head, .{ .old_rows = 7, .prefix_rows = 2, .target_rows = 5, .block_rows = 3 }, 2);
    defer std.testing.allocator.free(ranges);
    try std.testing.expectEqualSlices(engine.FitRange, &.{ .{ .start = 3, .rows = 2 }, .{ .start = 5, .rows = 1 }, .{ .start = 6, .rows = 1 } }, ranges);
    try std.testing.expectEqual(ranges.len, try exactFitRangeCount(head, .{ .old_rows = 7, .prefix_rows = 2, .target_rows = 5, .block_rows = 3 }, 2));
}

fn rangeAllocationFailure(allocator: std.mem.Allocator) !void {
    const positions = [_]u64{ 0, 1, 8, 9 };
    const ends = [_]u64{ 1, 2, 9, 10 };
    const bias = [_]f32{ 0, 0, 0, 0 };
    const values: [positions.len]f32 = @splat(0);
    const head: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &ends, .keys = &values, .values = &values, .bias = &bias };
    const ranges = try engine.headFitRanges(allocator, head, .{ .old_rows = 4, .prefix_rows = 0, .target_rows = 2, .block_rows = 2 }, 2);
    defer allocator.free(ranges);
    try std.testing.expectEqual(@as(usize, 2), ranges.len);
}

test "coverage-aware v18 range ownership unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, rangeAllocationFailure, .{});
}

test "adaptive chunk stats distinguish completed and rejected branch work" {
    var alpha: [0]f64 = .{};
    var targets: [0]usize = .{};
    const accepted = replay.Branch{ .divisor = 4, .candidate_rows = 8, .mass_error = 0.02, .output_rmse = 0.01, .rejection = "none", .alpha = &alpha, .geometry_fitted_ranges = 5, .geometry_targets = &targets };
    const heldout_rejected = replay.Branch{ .divisor = 3, .candidate_rows = 8, .mass_error = 0.2, .output_rmse = 0.1, .rejection = "heldout_gate", .alpha = &alpha, .geometry_fitted_ranges = 4, .geometry_targets = &targets };
    const early_rejected = replay.Branch{ .divisor = 2, .candidate_rows = 8, .mass_error = null, .output_rmse = null, .rejection = "joint_mass_or_value_fit", .alpha = &alpha, .geometry_fitted_ranges = 7, .geometry_targets = &targets };
    const published = try adaptiveChunkStats(6, true, 4, false, &.{ heldout_rejected, early_rejected, accepted });
    try std.testing.expectEqual(@as(usize, 27), published.attempted);
    try std.testing.expectEqual(@as(usize, 5), published.completed);
    const preserved = try adaptiveChunkStats(6, false, 0, false, &.{ heldout_rejected, early_rejected });
    try std.testing.expectEqual(@as(usize, 22), preserved.attempted);
    try std.testing.expectEqual(@as(usize, 0), preserved.completed);

    const reused = try adaptiveChunkStats(6, true, 4, true, &.{ heldout_rejected, early_rejected, accepted });
    try std.testing.expectEqual(@as(usize, 18), reused.attempted);
    try std.testing.expectEqual(@as(usize, 5), reused.completed);
    const reused_preserved = try adaptiveChunkStats(6, false, 0, true, &.{ heldout_rejected, early_rejected });
    try std.testing.expectEqual(@as(usize, 18), reused_preserved.attempted);
    try std.testing.expectEqual(@as(usize, 0), reused_preserved.completed);
}

const PriorCodec = enum { fp8, rotated4 };

fn materializePriorCodecFixture(allocator: std.mem.Allocator, codec_kind: PriorCodec, corrupt_map: bool) !void {
    const d = 64;
    var keys: [6 * d]f32 = @splat(0.25);
    var values: [6 * d]f32 = @splat(2);
    var packed_state: checkpoint.packed_storage.Packed = .{};
    defer packed_state.deinit(allocator);
    var fp8: checkpoint.codec.Fp8 = .{};
    if (codec_kind == .fp8) {
        var rounded_keys: [2 * d]f64 = @splat(0.25);
        var rounded_values: [2 * d]f64 = @splat(2);
        const key_scale = checkpoint.codec.scaleFor(&rounded_keys);
        const value_scale = checkpoint.codec.scaleFor(&rounded_values);
        try checkpoint.codec.round(&rounded_keys, key_scale);
        try checkpoint.codec.round(&rounded_values, value_scale);
        for (&rounded_keys, &rounded_values, keys[d .. 3 * d], values[d .. 3 * d]) |key, value, *out_key, *out_value| {
            out_key.* = @floatCast(key);
            out_value.* = @floatCast(value);
        }
        fp8 = .{ .first_row = 1, .rows = 2, .key_scale = key_scale, .value_scale = value_scale };
    } else {
        packed_state = try packed_state.append(allocator, 1, d, keys[d .. 3 * d], values[d .. 3 * d], .{});
        try packed_state.decodeRow(d, 0, keys[d .. 2 * d], values[d .. 2 * d]);
        try packed_state.decodeRow(d, 1, keys[2 * d .. 3 * d], values[2 * d .. 3 * d]);
    }
    const current: checkpoint.Head = .{ .dim = d, .compacted_until = 5, .positions = &.{ 0, 2, 4, 8, 9, 12 }, .coverage_ends = &.{ 1, 3, 5, 9, 10, 13 }, .keys = &keys, .values = &values, .bias = &.{ 0, 1, 1, 0, 0, 0 }, .fp8 = fp8, .compressed = packed_state };
    var normalized = try normalizeHead(allocator, current);
    defer normalized.deinit();
    for (normalized.keys, normalized.values, current.keys, current.values) |key, value, expected_key, expected_value| {
        try std.testing.expectEqual(@as(f64, expected_key), key);
        try std.testing.expectEqual(@as(f64, expected_value), value);
    }

    var result_keys: [5 * d]f64 = undefined;
    var result_values: [5 * d]f64 = undefined;
    const source_indices = if (corrupt_map) [_]usize{ 0, 2, 1, 3, 5 } else [_]usize{ 0, 1, 2, 3, 5 };
    for (source_indices, 0..) |source_row, candidate_row| {
        for (0..d) |column| {
            result_keys[candidate_row * d + column] = current.keys[source_row * d + column];
            result_values[candidate_row * d + column] = current.values[source_row * d + column];
        }
    }
    var result_bias: [5]f64 = .{ 0, 1, 1, 0, 0 };
    var coverage_indices = source_indices;
    var branches: [0]replay.Branch = .{};
    const result: replay.Result = .{ .allocator = allocator, .head = .{ .keys = &result_keys, .values = &result_values, .bias = &result_bias, .rows = 5, .dim = d, .scale = 1, .softcap = 0 }, .source_indices = @constCast(&source_indices), .coverage_source_indices = &coverage_indices, .branches = &branches, .accepted = true, .chosen_divisor = 2, .final_metrics_valid = true, .final_mass_error = 0, .final_output_rmse = 0 };
    var ranges = [_]admission.Range{.{ .start = 3, .rows = 3 }};
    var run: Run = .{ .allocator = allocator, .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .ranges = &ranges, .result = result, .numerical_policy = .legacy_dense_f64 };
    if (corrupt_map) return std.testing.expectError(error.InvalidAttentionShape, materializeHead(allocator, current, &run));
    var materialized = try materializeHead(allocator, current, &run);
    defer {
        allocator.free(@constCast(materialized.positions));
        allocator.free(@constCast(materialized.coverage_ends));
        allocator.free(@constCast(materialized.keys));
        allocator.free(@constCast(materialized.values));
        allocator.free(@constCast(materialized.bias));
        materialized.compressed.deinit(allocator);
    }
    try std.testing.expectEqualSlices(f32, current.keys[d .. 3 * d], materialized.keys[d .. 3 * d]);
    try std.testing.expectEqualSlices(f32, current.values[d .. 3 * d], materialized.values[d .. 3 * d]);
    try std.testing.expectEqualDeep(current.fp8, materialized.fp8);
    try std.testing.expectEqualSlices(checkpoint.packed_storage.Span, current.compressed.spans, materialized.compressed.spans);
    try std.testing.expectEqualSlices(u8, current.compressed.keys, materialized.compressed.keys);
    try std.testing.expectEqualSlices(u8, current.compressed.values, materialized.compressed.values);
}

test "adaptive materialization preserves prior FP8 and rotated4 payloads through sparse later ranges" {
    inline for (.{ PriorCodec.fp8, PriorCodec.rotated4 }) |codec_kind| {
        try materializePriorCodecFixture(std.testing.allocator, codec_kind, false);
        try materializePriorCodecFixture(std.testing.allocator, codec_kind, true);
        try std.testing.checkAllAllocationFailures(std.testing.allocator, materializePriorCodecFixture, .{ codec_kind, false });
    }
}

test "primary probe wide72 real Metal matches flat oracle with postfit" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const rows: usize = 96;
    const cap: usize = 2 * 1024 * 1024 * 1024;
    var positions: [rows]u64 = undefined;
    var coverage: [rows]u64 = undefined;
    var keys: [rows]f32 = @splat(0);
    var values: [rows]f32 = @splat(1);
    var bias: [rows]f32 = @splat(0);
    for (&positions, &coverage, 0..) |*position, *end, row| {
        // Twenty-four exact runs of four rows produce the production 72
        // divisor candidates while retaining a compact fixture.
        position.* = row + (row / 4) * 4;
        end.* = position.* + 1;
        // Distinct columns avoid a rank-one zero-key fixture that cannot
        // make the OMP progress required by a multi-row primary candidate.
        // Resident source keys are widened from f16. Keep that exact contract
        // so candidate serialization cannot change a selected source key.
        const key_pattern = [_]f32{ -0.5, -0.125, 0.0625, 0.375 };
        keys[row] = key_pattern[row % 4] + @as(f32, @floatFromInt(row / 4)) / 1024.0;
        try std.testing.expectEqual(keys[row], @as(f32, @floatCast(@as(f16, @floatCast(keys[row])))));
    }
    const head: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const plan: engine.HeadPlan = .{ .old_rows = rows, .prefix_rows = 0, .target_rows = 48, .block_rows = 4 };
    var train: [65]f64 = undefined;
    for (&train, 0..) |*query, i| query.* = @as(f64, @floatFromInt(i)) / 64.0 - 0.5;

    var fitter = try metal.Fitter.init(allocator, null, cap);
    defer fitter.deinit();
    var validation: validation_accelerate.Context = .{};
    const validation_backend = validation_accelerate.backend(&validation);
    const postfit_workspace = try replay.postfitWorkspaceReserve(train.len, rows, rows, 24);
    const base_cfg: solver.Config = .{
        .target_rows = 1,
        .selection = .residual_omp,
        .mass_solver = .active_set,
        .max_relative_mass_error = 0.05,
        .max_relative_output_rmse = 0.05,
        .max_workspace_bytes = cap,
        .workspace_admission = .resident_native_hybrid_2g_v1,
        .backend = fitter.backend(),
    };
    var bounded_value_evidence: BoundedValueEvidence = .{};
    const probe_cfg: PrimaryTrainProbeConfig = .{
        .heldout_query_rows = 1,
        .critical_policy = .train_probability_gte_0_025_no_top1_v1,
        .postfit_workspace_bytes = postfit_workspace,
        .validation_backend = validation_backend,
        .bounded_value_observer = &bounded_value_evidence,
    };
    var flat = runHeadWithPrimaryProbe(allocator, head, plan, &train, 1, 0, base_cfg, 0, cap, .accepted_f32_gram_reuse_v1, .flat_gram_only, 0, 0, false, false, .accelerate_fp64_summary_v1, .exact_original_train_hho_v1, .incremental_gram_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, probe_cfg) catch |err| {
        std.debug.print("primary72_fixture phase=flat error={s}\n", .{@errorName(err)});
        return err;
    };
    defer flat.deinit();
    std.debug.print("primary72_fixture phase=flat_complete rows={d} mass={d} rmse={d} validation_calls={d}\n", .{ flat.candidate.head.rows, flat.train_metrics.max_relative_mass_error, flat.train_metrics.relative_output_rmse, validation.successfulCalls() });
    const calls_after_flat = validation.successfulCalls();
    var wide = runHeadWithPrimaryProbe(allocator, head, plan, &train, 1, 0, base_cfg, 0, cap, .accepted_f32_gram_reuse_v1, .bounded_page_wide_cache_v1, 64, 256 * 1024 * 1024, true, true, .accelerate_fp64_summary_v1, .exact_original_train_hho_v1, .incremental_gram_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, probe_cfg) catch |err| {
        std.debug.print("primary72_fixture phase=wide error={s}\n", .{@errorName(err)});
        return err;
    };
    defer wide.deinit();
    std.debug.print("primary72_fixture phase=wide_complete rows={d} mass={d} rmse={d} validation_calls={d}\n", .{ wide.candidate.head.rows, wide.train_metrics.max_relative_mass_error, wide.train_metrics.relative_output_rmse, validation.successfulCalls() });

    try std.testing.expectEqual(@as(usize, 1), wide.postfit.attempts);
    try std.testing.expectEqual(@as(usize, 1), wide.postfit.completions);
    try std.testing.expect(bounded_value_evidence.calls > 0);
    try std.testing.expect(bounded_value_evidence.workspace_peak_bytes > 0);
    try std.testing.expect(bounded_value_evidence.cancellation_checks > 0);
    try std.testing.expect(validation.successfulCalls() > calls_after_flat);
    try std.testing.expect(!std.mem.allEqual(u8, &wide.source_authority_sha256, 0));
    try std.testing.expect(!std.mem.allEqual(u8, &wide.train_domain_sha256, 0));
    try std.testing.expect(!std.mem.allEqual(u8, &wide.candidate_sha256, 0));
    try std.testing.expectEqualSlices(usize, flat.candidate.source_indices, wide.candidate.source_indices);
    try std.testing.expectEqualSlices(usize, flat.candidate.coverage_source_indices, wide.candidate.coverage_source_indices);
    // Flat and page capture use separate f32 reductions. Compare their fit
    // metrics at f32 precision, then check each candidate independently below
    // against the scalar FP64 oracle at the tighter validation tolerance.
    const capture_rounding_tolerance: f64 = 8 * std.math.floatEps(f32);
    try std.testing.expectApproxEqAbs(flat.train_metrics.max_relative_mass_error, wide.train_metrics.max_relative_mass_error, capture_rounding_tolerance);
    try std.testing.expectApproxEqAbs(flat.train_metrics.relative_output_rmse, wide.train_metrics.relative_output_rmse, 1e-12);
    var source_keys: [rows]f64 = undefined;
    for (keys, &source_keys) |key, *out| out.* = key;
    const source_values: [rows]f64 = @splat(1);
    const source_bias: [rows]f64 = @splat(0);
    const source_fp64: solver.Head = .{ .keys = &source_keys, .values = &source_values, .bias = &source_bias, .rows = rows, .dim = 1, .scale = 1, .softcap = 0 };
    const oracle = try solver.validationMetrics(allocator, source_fp64, wide.candidate.head, &train, .{
        .target_rows = wide.candidate.head.rows,
        .max_relative_mass_error = 0.05,
        .max_relative_output_rmse = 0.05,
    });
    const flat_oracle = try solver.validationMetrics(allocator, source_fp64, flat.candidate.head, &train, .{
        .target_rows = flat.candidate.head.rows,
        .max_relative_mass_error = 0.05,
        .max_relative_output_rmse = 0.05,
    });
    try std.testing.expectApproxEqAbs(flat_oracle.max_relative_mass_error, flat.train_metrics.max_relative_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(flat_oracle.relative_output_rmse, flat.train_metrics.relative_output_rmse, 1e-12);
    try std.testing.expectApproxEqAbs(oracle.max_relative_mass_error, wide.train_metrics.max_relative_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(oracle.relative_output_rmse, wide.train_metrics.relative_output_rmse, 1e-12);
    try std.testing.expect(wide.train_metrics.max_relative_mass_error <= 0.05);
    try std.testing.expect(wide.train_metrics.relative_output_rmse <= 0.05);
    try std.testing.expect(flat.train_metrics.max_relative_mass_error <= 0.05);
    try std.testing.expect(flat.train_metrics.relative_output_rmse <= 0.05);
    try std.testing.expectEqual(@as(usize, 65), wide.fit_query_rows);
    try std.testing.expectEqual(@as(usize, 0), wide.cv_query_rows);

    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var cancelled_cfg = base_cfg;
    cancelled_cfg.control = .{ .check_fn = Cancel.check };
    const calls_before_cancel = validation.successfulCalls();
    const dispatches_before_cancel = fitter.dispatchCount();
    bounded_value_evidence = .{};
    try std.testing.expectError(error.Cancelled, runHeadWithPrimaryProbe(allocator, head, plan, &train, 1, 0, cancelled_cfg, 0, cap, .accepted_f32_gram_reuse_v1, .bounded_page_wide_cache_v1, 64, 256 * 1024 * 1024, true, true, .accelerate_fp64_summary_v1, .exact_original_train_hho_v1, .incremental_gram_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, probe_cfg));
    try std.testing.expectEqual(calls_before_cancel, validation.successfulCalls());
    try std.testing.expectEqual(dispatches_before_cancel, fitter.dispatchCount());

    var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
    bounded_value_evidence = .{};
    try std.testing.expectError(error.OutOfMemory, runHeadWithPrimaryProbe(failing.allocator(), head, plan, &train, 1, 0, base_cfg, 0, cap, .accepted_f32_gram_reuse_v1, .bounded_page_wide_cache_v1, 64, 256 * 1024 * 1024, true, true, .accelerate_fp64_summary_v1, .exact_original_train_hho_v1, .incremental_gram_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, probe_cfg));
    bounded_value_evidence = .{};
    var recovered = runHeadWithPrimaryProbe(allocator, head, plan, &train, 1, 0, base_cfg, 0, cap, .accepted_f32_gram_reuse_v1, .bounded_page_wide_cache_v1, 64, 256 * 1024 * 1024, true, true, .accelerate_fp64_summary_v1, .exact_original_train_hho_v1, .incremental_gram_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, probe_cfg) catch |err| {
        std.debug.print("primary72_fixture phase=recovered error={s}\n", .{@errorName(err)});
        return err;
    };
    try std.testing.expect(bounded_value_evidence.calls > 0);
    try std.testing.expect(bounded_value_evidence.workspace_peak_bytes > 0);
    try std.testing.expect(bounded_value_evidence.cancellation_checks > 0);
    recovered.deinit();
}

test "final engine callback exact tracker cap late cancellation refusal recovery matches flat oracle" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const rows: usize = 96;
    const cap: usize = 2 * 1024 * 1024 * 1024;
    var positions: [rows]u64 = undefined;
    var coverage: [rows]u64 = undefined;
    var keys: [rows]f32 = undefined;
    const values: [rows]f32 = @splat(1);
    const bias: [rows]f32 = @splat(0);
    for (&positions, &coverage, &keys, 0..) |*position, *end, *key, row| {
        position.* = row + (row / 4) * 4;
        end.* = position.* + 1;
        const pattern = [_]f32{ -0.5, -0.125, 0.0625, 0.375 };
        key.* = pattern[row % 4] + @as(f32, @floatFromInt(row / 4)) / 1024.0;
    }
    const head: checkpoint.Head = .{ .dim = 1, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const heads = [_]checkpoint.Head{head};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .heads = &heads }};
    const state: checkpoint.State = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 192, .donors = &donors, .layer_donors = &.{0} };
    var train: [65]f32 = undefined;
    var train_positions: [65]u64 = undefined;
    for (&train, &train_positions, 0..) |*query, *position, i| {
        query.* = @as(f32, @floatFromInt(i)) / 64.0 - 0.5;
        position.* = 200 + i;
    }
    const heldout = [_]f32{0.25};
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 0, .query_head = 0, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &.{300} };
    var fitter = try metal.Fitter.init(allocator, null, cap);
    defer fitter.deinit();
    var stats: engine.Stats = .{};
    var options: engine.Options = .{
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .admission = .cumulative_ladder_4_3_2,
        .chunk_rows = 256,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .independent_fp64_publication = true,
        .validation = .whole_head,
        .storage_precision = .f16,
        .max_queries_per_consumer = 768,
        .min_global_savings_ratio = 1.5,
        .v18_reference_summary_cache_policy = .exact_original_train_hho_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .stats = &stats,
    };
    options.fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_workspace_bytes = cap, .workspace_admission = .resident_native_hybrid_2g_v1, .backend = fitter.backend() };
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    var physical_budget: PhysicalBudgetContext = .{};
    var flat = try engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 48, 4 }, &options, 0);
    defer {
        allocator.free(@constCast(flat.positions));
        allocator.free(@constCast(flat.coverage_ends));
        allocator.free(@constCast(flat.keys));
        allocator.free(@constCast(flat.values));
        allocator.free(@constCast(flat.bias));
        flat.compressed.deinit(allocator);
    }
    stats = .{};
    options.adaptive_v18_workspace_profile = .resident_e2_formal_2g_p60_no_fallback_v1;
    options.adaptive_v18_geometry_route = .bounded_page_over_96;
    options.adaptive_v18_primary_budget_percent = 60;
    options.adaptive_v18_fallback_budget_percent = 95;
    options.adaptive_v18_max_budget_fallbacks = 0;
    options.adaptive_v18_bounded_tile_queries = 64;
    options.adaptive_v18_bounded_builder_transient_bytes = 256 * 1024 * 1024;
    options.adaptive_v18_bounded_delta_accelerated = true;
    options.adaptive_v18_bounded_tsqr_value_backend = true;
    try std.testing.expectError(error.InvalidFitConfig, engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 24, 2 }, &options, 0));
    options.adaptive_v18_fallback_budget_percent = 0;
    var e2 = try engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 24, 2 }, &options, 0);
    defer {
        allocator.free(@constCast(e2.positions));
        allocator.free(@constCast(e2.coverage_ends));
        allocator.free(@constCast(e2.keys));
        allocator.free(@constCast(e2.values));
        allocator.free(@constCast(e2.bias));
        e2.compressed.deinit(allocator);
    }
    try std.testing.expectEqual(@as(u8, 60), stats.v18_primary_budget_percent);
    try std.testing.expectEqual(@as(usize, 0), stats.v18_budget_fallback_heads);
    try std.testing.expect(stats.v18_bounded_heads > 0 and stats.v18_bounded_tsqr_calls > 0);
    // E4's 24-range p91 oracle and E2's 36-range p60 authority intentionally
    // select different geometries. Compare the invariant quality contract,
    // not byte identity between non-equivalent plans.
    try std.testing.expectEqual(@as(usize, 1), stats.v18_accepted_heads);
    try std.testing.expect(stats.max_head_mass_error <= options.fit.max_relative_mass_error);
    try std.testing.expect(stats.max_head_output_rmse <= options.fit.max_relative_output_rmse);
    try std.testing.expect(e2.positions.len > 0 and e2.positions.len < head.positions.len);
    for (e2.positions[1..], e2.positions[0 .. e2.positions.len - 1]) |position, previous|
        try std.testing.expect(position > previous);
    for (e2.values) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expectEqual(@as(usize, 0), stats.v18_bounded_score_accelerate_calls);
    try std.testing.expectEqual(@as(usize, 0), stats.v18_bounded_score_accelerate_tiles);
    try std.testing.expectEqual(@as(usize, 0), stats.v18_bounded_score_accelerate_dgemm_calls);
    try std.testing.expectEqual(@as(usize, 0), stats.v18_bounded_score_cancellation_checks);

    stats = .{};
    options.adaptive_v18_bounded_score_policy = .accelerate_fp64_tiled_v1;
    var e2_accelerated = try engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 24, 2 }, &options, 0);
    defer {
        allocator.free(@constCast(e2_accelerated.positions));
        allocator.free(@constCast(e2_accelerated.coverage_ends));
        allocator.free(@constCast(e2_accelerated.keys));
        allocator.free(@constCast(e2_accelerated.values));
        allocator.free(@constCast(e2_accelerated.bias));
        e2_accelerated.compressed.deinit(allocator);
    }
    try std.testing.expectEqualSlices(u64, e2.positions, e2_accelerated.positions);
    try std.testing.expectEqualSlices(u64, e2.coverage_ends, e2_accelerated.coverage_ends);
    for (e2.values, e2_accelerated.values) |scalar, accelerated|
        try std.testing.expectApproxEqRel(scalar, accelerated, 2e-12);
    try std.testing.expect(stats.v18_bounded_score_accelerate_calls > 0);
    try std.testing.expect(stats.v18_bounded_score_accelerate_tiles >= stats.v18_bounded_score_accelerate_calls);
    try std.testing.expectEqual(stats.v18_bounded_score_accelerate_tiles * 2, stats.v18_bounded_score_accelerate_dgemm_calls);
    try std.testing.expect(stats.v18_bounded_score_cancellation_checks >= stats.v18_bounded_score_accelerate_tiles * 2);
    options.adaptive_v18_bounded_score_policy = .scalar_reference;

    options.adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1;
    options.adaptive_v18_geometry_route = .flat_gram_only;
    options.adaptive_v18_primary_budget_percent = 91;
    options.adaptive_v18_fallback_budget_percent = 95;
    options.adaptive_v18_max_budget_fallbacks = 3;
    options.adaptive_v18_bounded_tile_queries = 0;
    options.adaptive_v18_bounded_builder_transient_bytes = 0;
    options.adaptive_v18_bounded_delta_accelerated = false;
    options.adaptive_v18_bounded_tsqr_value_backend = false;
    const probability_bytes = train.len * rows * @sizeOf(f32);
    const output_bytes = train.len * @sizeOf(f64);
    var boundary_tracker = @import("host_allocation_tracker.zig").HostAllocationTracker.initSwitchablePrepare(0);
    const boundary_identity = try boundary_tracker.activateBoundedPrepare(@import("host_allocation_tracker.zig").bounded_prepare_e4_physical_bytes);
    var boundary_authority = try @import("attention_match_native_resident_hybrid_bridge.zig").wideCacheAuthority(&boundary_tracker, boundary_identity);
    const initial_boundary = try boundary_authority.flatAdmission(probability_bytes, output_bytes, .{});
    boundary_tracker.physical_live_bytes = @import("host_allocation_tracker.zig").bounded_prepare_e4_physical_bytes - initial_boundary.required_physical_bytes + 1;
    try std.testing.expect(!(try boundary_authority.flatAdmission(probability_bytes, output_bytes, .{})).fits);
    boundary_tracker.physical_live_bytes -= 1;
    try std.testing.expect((try boundary_authority.flatAdmission(probability_bytes, output_bytes, .{})).fits);
    boundary_tracker.physical_live_bytes -= 1;
    try std.testing.expect((try boundary_authority.flatAdmission(probability_bytes, output_bytes, .{})).fits);

    const LateCancel = struct {
        calls: usize = 0,
        cancelled: bool = false,
        fn admission(raw: ?*anyopaque, p: usize, y: usize) !@import("attention_match_wide_cache_authority.zig").FlatAdmission {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 2) self.cancelled = true;
            return .{ .owner_identity = 8, .probability_logical_bytes = p, .output_logical_bytes = y, .available_physical_bytes = if (self.calls == 2) 0 else std.math.maxInt(usize), .required_physical_bytes = p + y, .fits = self.calls != 2 };
        }
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.cancelled) return error.Cancelled;
        }
    };
    var late_cancel: LateCancel = .{};
    var cancel_authority: @import("attention_match_wide_cache_authority.zig").Authority = .{ .context = &late_cancel, .owner_identity = 8, .policy_sha256 = @import("attention_match_wide_cache_authority.zig").policy_sha256, .flat_admission_fn = LateCancel.admission };
    options.adaptive_v18_wide_cache_authority = &cancel_authority;
    options.fit.control = .{ .context = &late_cancel, .check_fn = LateCancel.check };
    try std.testing.expectError(error.Cancelled, engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 48, 4 }, &options, 0));
    try std.testing.expectEqual(@as(usize, 2), late_cancel.calls);
    try std.testing.expect(cancel_authority.allocation_refusal == null);
    const Refusal = struct {
        calls: usize = 0,
        fn admission(raw: ?*anyopaque, p: usize, y: usize) !@import("attention_match_wide_cache_authority.zig").FlatAdmission {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return .{ .owner_identity = 9, .probability_logical_bytes = p, .output_logical_bytes = y, .available_physical_bytes = if (self.calls == 2) 0 else std.math.maxInt(usize), .required_physical_bytes = p + y, .fits = self.calls != 2 };
        }
    };
    var refusal: Refusal = .{};
    var authority: @import("attention_match_wide_cache_authority.zig").Authority = .{ .context = &refusal, .owner_identity = 9, .policy_sha256 = @import("attention_match_wide_cache_authority.zig").policy_sha256, .flat_admission_fn = Refusal.admission };
    stats = .{};
    options.fit.control = .{};
    options.adaptive_v18_wide_cache_authority = &authority;
    var wide = try engineCallbackFromState(&physical_budget, allocator, state, 0, 0, &.{batch}, layout, .{ rows, 0, 48, 4 }, &options, 0);
    defer {
        allocator.free(@constCast(wide.positions));
        allocator.free(@constCast(wide.coverage_ends));
        allocator.free(@constCast(wide.keys));
        allocator.free(@constCast(wide.values));
        allocator.free(@constCast(wide.bias));
        wide.compressed.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 3), refusal.calls);
    try std.testing.expectEqualSlices(u64, flat.positions, wide.positions);
    try std.testing.expectEqualSlices(u64, flat.coverage_ends, wide.coverage_ends);
    try std.testing.expectEqualSlices(f32, flat.keys, wide.keys);
    try std.testing.expectEqualSlices(f32, flat.values, wide.values);
    try std.testing.expect(stats.v18_bounded_heads > 0 and stats.v18_bounded_tsqr_heads > 0 and stats.v18_bounded_tsqr_calls > 0);
    try std.testing.expectEqual(@as(usize, 72), stats.v18_bounded_scoring_selection_hits);
    try std.testing.expectEqual(@as(usize, 72), stats.v18_bounded_scoring_fit_hits);
}
