// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Complete default-off bounded route after authenticated input preparation.
const std = @import("std");
const builtin = @import("builtin");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const prepared_mod = @import("attention_match_joint_prepared_input.zig");
const scoring = @import("attention_match_joint_bounded_page_scoring.zig");
const page_replay = @import("attention_match_joint_bounded_page_replay.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const selection = @import("attention_match_global_selection_diagnostic.zig");
const streamed_gram = @import("attention_match_streamed_gram_selection.zig");
const joint = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const group_postfit = @import("attention_match_train_group_mass_postfit.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const delta_kernel = @import("attention_match_gram_delta_accelerate.zig");
const validation_accelerate = @import("../../ops/metal_attention_match_validation.zig");
const metal = @import("../../ops/metal_attention_match_fit.zig");
const tsqr_value = @import("attention_match_e2_tsqr_value_backend.zig");

pub const geometry_snapshot_schema = "antfly.attention-match-bounded-train-geometry-snapshot.v1";

pub const Evidence = struct {
    training_route: []const u8 = "legacy_bounded_v1",
    train_queries: usize,
    geometry_budget_percent: u8 = 100,
    budget_profile: BudgetProfile = .legacy_e4,
    query_batch_rows: usize,
    probability_storage: training_source.ProbabilityStorage,
    provider_calls: u64,
    provider_rows: u64,
    workspace_bound_bytes: usize,
    scoring_cache_prepares: usize,
    scoring_cache_selection_hits: usize,
    scoring_cache_fit_hits: usize,
    range_cache_prepares: usize,
    postfit_range_cache_prepares: usize = 0,
    range_cache_selection_hits: usize,
    range_cache_fit_hits: usize,
    geometry_cache_requests: usize,
    geometry_cache_hits: usize,
    geometry_cache_loads: usize,
    bounded_reduction_policy: []const u8 = scoring.execution_policy,
    bounded_score_completions: usize = 0,
    bounded_score_query_rows: usize = 0,
    bounded_score_accelerate_calls: usize = 0,
    bounded_score_accelerate_tiles: usize = 0,
    bounded_score_accelerate_dgemm_calls: usize = 0,
    bounded_score_cancellation_checks: usize = 0,
    geometry_delta_accelerated: bool = false,
    geometry_delta_tiles: usize = 0,
    geometry_delta_backend_calls: usize = 0,
    geometry_delta_packed_bytes_peak: usize = 0,
    geometry_delta_staged_bytes_peak: usize = 0,
    geometry_delta_fallbacks: usize = 0,
    raw_selection_policy: []const u8 = "legacy_dense_qr_raw_omp_v1",
    raw_selection_calls: usize = 0,
    raw_selection_probability_values: usize = 0,
    raw_selection_workspace_peak_bytes: usize = 0,
    raw_selection_provenance_sha256: [32]u8 = @splat(0),
    critical_core_policy: []const u8 = "disabled",
    critical_core_policy_sha256: [32]u8 = @splat(0),
    critical_core_authority_sha256: [32]u8 = @splat(0),
    critical_core_rows_total: usize = 0,
    critical_core_rows_peak: usize = 0,
};

pub const GeometryAcceleration = struct { backend: delta_kernel.Backend, require_accelerated: bool = true };
pub const BudgetProfile = enum { legacy_e4, e2_formal_p60_no_fallback_v1 };

fn validateBudgetProfile(profile: BudgetProfile, geometry_budget_percent: u8, fallback_geometry_budget_percent: u8, fallback_context: bool, fallback_fn: bool, geometry_hook: bool, critical: replay.CriticalCorePolicy, postfit: replay.PostfitPolicy) !void {
    const fallback_configured = fallback_geometry_budget_percent != 0 or fallback_context or fallback_fn;
    switch (profile) {
        .legacy_e4 => {
            if (geometry_budget_percent != 91 and geometry_budget_percent != 95 and geometry_budget_percent != 100)
                return error.InvalidFitConfig;
            if (fallback_configured and (geometry_budget_percent != 91 or fallback_geometry_budget_percent != 95 or !fallback_context or !fallback_fn or geometry_hook))
                return error.InvalidFitConfig;
        },
        .e2_formal_p60_no_fallback_v1 => if (geometry_budget_percent != 60 or fallback_configured or
            critical != .train_probability_gte_0_025_no_top1_v1 or
            postfit != .train_global_residual_range_mass_l2_exact_f32_cache_v1)
            return error.InvalidFitConfig,
    }
}

test "bounded formal E2 budget admits only p60 without fallback" {
    try validateBudgetProfile(.e2_formal_p60_no_fallback_v1, 60, 0, false, false, false, .train_probability_gte_0_025_no_top1_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1);
    try std.testing.expectError(error.InvalidFitConfig, validateBudgetProfile(.legacy_e4, 60, 0, false, false, false, .train_probability_gte_0_025_no_top1_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1));
    try std.testing.expectError(error.InvalidFitConfig, validateBudgetProfile(.e2_formal_p60_no_fallback_v1, 60, 95, true, true, false, .train_probability_gte_0_025_no_top1_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1));
    try std.testing.expectError(error.InvalidFitConfig, validateBudgetProfile(.e2_formal_p60_no_fallback_v1, 61, 0, false, false, false, .train_probability_gte_0_025_no_top1_v1, .train_global_residual_range_mass_l2_exact_f32_cache_v1));
}
pub const GeometryFreezeView = struct {
    source: @import("attention_match.zig").Head,
    source_positions: []const u64,
    source_coverage: []const u64,
    ranges: []const @import("attention_match_cumulative_cache.zig").Range,
    adaptive_plan: *const replay.AdaptivePlan,
    geometry_budget_percent: u8,
    train_queries: usize,
    route_external_bytes: usize,
    scoring_workspace_bound_bytes: usize,
    geometry_workspace_bound_bytes: usize,
    combined_cap: usize,
    critical_core_policy: replay.CriticalCorePolicy = .disabled,
};
pub const GeometryFreezeHook = struct {
    context: ?*anyopaque,
    source_positions: []const u64,
    source_coverage: []const u64,
    invoke_fn: *const fn (?*anyopaque, GeometryFreezeView, @import("attention_match.zig").Control) anyerror!void,
};

pub const PrimaryProbeView = struct {
    allocator: std.mem.Allocator,
    source: @import("attention_match.zig").Head,
    train: []const f64,
    ranges: []const admission.Range,
    adaptive_plan: *const replay.AdaptivePlan,
    cfg: @import("attention_match.zig").Config,
    external_live_bytes: usize,
    combined_cap: usize,
    postfit_hook: replay.CandidatePostfitHook,
};

pub const PrimaryProbeHook = struct {
    context: ?*anyopaque,
    invoke_fn: *const fn (?*anyopaque, PrimaryProbeView) anyerror!void,
};

fn validateGeometryHookAuthority(source_rows: usize, hook: GeometryFreezeHook) !void {
    if (hook.source_positions.len != source_rows or hook.source_coverage.len != source_rows) return error.InvalidFitConfig;
    for (hook.source_positions, hook.source_coverage, 0..) |position, coverage, index| {
        const expected = std.math.cast(u64, index) orelse return error.InvalidFitConfig;
        if (position != expected or coverage != expected + 1) return error.InvalidFitConfig;
    }
}

fn dispatchGeometryHook(hook: GeometryFreezeHook, view: GeometryFreezeView, control: @import("attention_match.zig").Control) !void {
    try control.check();
    try hook.invoke_fn(hook.context, view, control);
}
pub const Output = struct {
    result: replay.Result,
    evidence: Evidence,
    pub fn deinit(self: *Output) void {
        self.result.deinit();
        self.* = undefined;
    }
};

/// Non-owning authenticated input view. This is the production serving seam:
/// the caller retains every slice and the bounded route never deinitializes or
/// transfers them. `PreparedInput` remains the owning diagnostic API below.
pub const PreparedView = struct {
    source: @import("attention_match.zig").Head,
    train: []const f64,
    heldout: []const f64,
    ranges: []const @import("attention_match_cumulative_cache.zig").Range,
    cfg: @import("attention_match.zig").Config,
    external_live_bytes: usize,
    combined_cap: usize,
    critical_core_policy: replay.CriticalCorePolicy = .disabled,
    postfit_policy: replay.PostfitPolicy = .disabled,
    score_policy: page_replay.ScorePolicy = .scalar_reference,
    validation: replay.PageBackedValidation = .{},
    budget_profile: BudgetProfile = .legacy_e4,
    test_budget_accept_context: ?*anyopaque = null,
    test_budget_accept_fn: ?*const fn (?*anyopaque, u8, bool) bool = null,

    pub fn validate(self: PreparedView) !void {
        return self.validateRole(false);
    }

    fn validatePrimaryTrain(self: PreparedView) !void {
        return self.validateRole(true);
    }

    fn validateRole(self: PreparedView, primary_train_only: bool) !void {
        try self.cfg.control.check();
        try self.source.validate();
        const invalid_heldout_role = if (primary_train_only) self.heldout.len != 0 else self.heldout.len == 0;
        if (self.source.dim == 0 or self.train.len == 0 or invalid_heldout_role or
            self.train.len % self.source.dim != 0 or self.heldout.len % self.source.dim != 0 or
            self.ranges.len == 0 or self.combined_cap == 0 or self.external_live_bytes > self.combined_cap)
            return error.InvalidFitConfig;
        const test_override = self.test_budget_accept_context != null or self.test_budget_accept_fn != null;
        if (test_override and (self.test_budget_accept_context == null or self.test_budget_accept_fn == null))
            return error.InvalidFitConfig;
        if (comptime !@import("builtin").is_test) if (test_override) return error.InvalidFitConfig;
        var previous_end: usize = 0;
        for (self.ranges, 0..) |range, index| {
            const end = std.math.add(usize, range.start, range.rows) catch return error.InvalidFitConfig;
            if (range.rows == 0 or end > self.source.rows or (index != 0 and range.start < previous_end))
                return error.InvalidFitConfig;
            previous_end = end;
        }
    }

    pub fn fromOwned(prepared: *const prepared_mod.PreparedInput) PreparedView {
        return .{
            .source = prepared.source,
            .train = prepared.train,
            .heldout = prepared.heldout,
            .ranges = prepared.ranges,
            .cfg = prepared.cfg,
            .external_live_bytes = prepared.external_live_bytes,
            .combined_cap = prepared.combined_cap,
            .critical_core_policy = .disabled,
        };
    }
};

const BoundedPostfitContext = struct {
    page: *page_replay.Context,
    reserve: usize,
    range_cache_prepares: usize = 0,

    fn load(raw: ?*anyopaque, range: group_postfit.Range, control: @import("attention_match.zig").Control) !*const training_source.Source {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const before = self.page.cache_counters.prepare_attempts;
        const source = try self.page.loadRangeForPostfit(.{ .start = range.start, .rows = range.rows }, control);
        const delta = std.math.sub(usize, self.page.cache_counters.prepare_attempts, before) catch
            return error.V18TrainingCacheUnsupported;
        self.range_cache_prepares = std.math.add(usize, self.range_cache_prepares, delta) catch {
            self.page.releaseRangeForPostfit();
            return error.V18TrainingCacheUnsupported;
        };
        return source;
    }
    fn release(raw: ?*anyopaque) void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return));
        self.page.releaseRangeForPostfit();
    }
    fn apply(raw: ?*anyopaque, allocator: std.mem.Allocator, original: @import("attention_match.zig").Head, candidate: *replay.Candidate, input_ranges: []const admission.Range, frozen: replay.FrozenOwned, cfg: @import("attention_match.zig").Config) !replay.PostfitEvidence {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const ranges = try allocator.alloc(group_postfit.Range, input_ranges.len);
        defer allocator.free(ranges);
        for (input_ranges, ranges) |range, *out| out.* = .{ .start = range.start, .rows = range.rows, .target_rows = range.rows };
        var core_count: usize = 0;
        for (frozen.chunks) |chunk| core_count = try std.math.add(usize, core_count, chunk.exact_core_local.len);
        const cores = try allocator.alloc(usize, core_count);
        defer allocator.free(cores);
        var ci: usize = 0;
        for (frozen.ranges, frozen.chunks) |range, chunk| {
            var index: ?usize = null;
            for (input_ranges, 0..) |input, ri| if (input.start == range.start and input.rows == range.rows) {
                index = ri;
                break;
            };
            ranges[index orelse return error.InvalidAttentionShape].target_rows = chunk.selected_local.len;
            for (chunk.exact_core_local) |local| {
                if (local >= range.rows) return error.InvalidAttentionShape;
                cores[ci] = try std.math.add(usize, range.start, local);
                ci += 1;
            }
        }
        const metadata = try std.math.add(usize, try std.math.mul(usize, ranges.len, @sizeOf(group_postfit.Range)), try std.math.mul(usize, cores.len, @sizeOf(usize)));
        if (metadata >= self.reserve) return error.CompactionWorkspaceLimit;
        var post_cfg = cfg;
        post_cfg.max_workspace_bytes = self.reserve - metadata;
        const diagnostics = try group_postfit.applyBounded(allocator, original, candidate.head, candidate.source_indices, ranges, cores, .{ .context = self, .train = self.page.train, .query_rows = self.page.train.len / original.dim, .original_rows = original.rows, .load_fn = load, .release_fn = release }, post_cfg, 0);
        return .{ .policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1, .policy_digest = replay.postfitPolicyDigest(.train_global_residual_range_mass_l2_exact_f32_cache_v1), .attempts = 1, .completions = 1, .nnls_sweeps = diagnostics.nnls_sweeps, .max_abs_train_before = diagnostics.max_abs_before, .max_abs_train_after = diagnostics.max_abs_after, .workspace_bytes = try std.math.add(usize, diagnostics.workspace_bytes, metadata) };
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn acceleratedScoreScratchBytes(policy: page_replay.ScorePolicy, scratch_elements: usize) !usize {
    return if (policy == .accelerate_fp64_tiled_v1) try mul(scratch_elements, @sizeOf(f64)) else 0;
}
fn rounded(bytes: usize) !usize {
    const page = std.heap.pageSize();
    return (try add(bytes, page - 1)) & ~(page - 1);
}

/// Conservative per-range selection + one-chunk joint-fit authority. The
/// caller admits this bound simultaneously with the captured page provider,
/// range cache, normalized Y, and retained scored artifacts.
pub fn builderTransientBytes(source: @import("attention_match.zig").Head, query_count: usize, maximum_range_rows: usize, cfg: @import("attention_match.zig").Config) !usize {
    return builderTransientBytesWithCriticalPolicy(source, query_count, maximum_range_rows, cfg, .disabled);
}

pub fn builderTransientBytesWithCriticalPolicy(source: @import("attention_match.zig").Head, query_count: usize, maximum_range_rows: usize, cfg: @import("attention_match.zig").Config, critical_core_policy: replay.CriticalCorePolicy) !usize {
    if (query_count == 0 or maximum_range_rows < 2) return error.InvalidFitConfig;
    const target = std.math.divCeil(usize, maximum_range_rows, 2) catch return error.InvalidFitConfig;
    var local = cfg;
    local.target_rows = target;
    var selection_cfg = local;
    selection_cfg.backend = null;
    selection_cfg.value_backend = null;
    const select_bytes = try selection.cachedWorkspaceBytes(maximum_range_rows, query_count, selection_cfg);
    const descriptors = @sizeOf(joint.FrozenChunk) + @sizeOf(@import("attention_match_cumulative_cache.zig").Range) + @sizeOf([]usize) + @sizeOf([]f64);
    const output_bytes = try mul(target, @sizeOf(usize) + @sizeOf(f64));
    // freezeBranch retains the descriptor arrays throughout selection, then
    // briefly owns selected output plus its duplicated final arrays.
    const selection_peak = try add(descriptors, @max(select_bytes, try mul(output_bytes, 2)));
    const retained_frozen = try add(descriptors, output_bytes);
    const fit = try joint.workspaceBytesShape(source, query_count, 1, target, target, local, true, .identity_chunk_coefficients_v18, false);
    const recompute_fixed = if (cfg.value_backend) |backend| backend.recompute_fixed_original_tiled else false;
    const fixed_original = if (critical_core_policy == .disabled or recompute_fixed) 0 else try joint.fixedOriginalTermBytes(query_count, source.dim);
    return @max(selection_peak, try add(try add(retained_frozen, fixed_original), fit.required));
}

pub fn streamedGramBuilderTransientBytes(source: @import("attention_match.zig").Head, query_count: usize, maximum_range_rows: usize, query_batch_rows: usize, cfg: @import("attention_match.zig").Config) !usize {
    return streamedGramBuilderTransientBytesWithCriticalPolicy(source, query_count, maximum_range_rows, query_batch_rows, cfg, .disabled);
}

pub fn streamedGramBuilderTransientBytesWithCriticalPolicy(source: @import("attention_match.zig").Head, query_count: usize, maximum_range_rows: usize, query_batch_rows: usize, cfg: @import("attention_match.zig").Config, critical_core_policy: replay.CriticalCorePolicy) !usize {
    if (query_count == 0 or maximum_range_rows < 2) return error.InvalidFitConfig;
    const target = std.math.divCeil(usize, maximum_range_rows, 2) catch return error.InvalidFitConfig;
    var local = cfg;
    local.target_rows = target;
    const descriptors = @sizeOf(joint.FrozenChunk) + @sizeOf(@import("attention_match_cumulative_cache.zig").Range) + @sizeOf([]usize) + @sizeOf([]f64);
    const output_bytes = try mul(target, @sizeOf(usize) + @sizeOf(f64));
    const select_bytes = try streamed_gram.workspaceBytes(query_count, maximum_range_rows, target, query_batch_rows);
    const fit = try joint.workspaceBytesShape(source, query_count, 1, target, target, local, true, .identity_chunk_coefficients_v18, false);
    const recompute_fixed = if (cfg.value_backend) |backend| backend.recompute_fixed_original_tiled else false;
    const fixed_original = if (critical_core_policy == .disabled or recompute_fixed) 0 else try joint.fixedOriginalTermBytes(query_count, source.dim);
    return @max(try add(descriptors, select_bytes), try add(try add(try add(descriptors, output_bytes), fixed_original), fit.required));
}

/// Single admission authority shared by the outer callback preflight and the
/// executing bounded replay. Non-100 budgets execute streamed Gram selection;
/// their planner must never be replaced by the dense cached-QR bound.
pub fn builderTransientBytesForPreparedRoute(source: @import("attention_match.zig").Head, query_count: usize, maximum_range_rows: usize, query_batch_rows: usize, cfg: @import("attention_match.zig").Config, critical_core_policy: replay.CriticalCorePolicy, geometry_budget_percent: u8) !usize {
    if (geometry_budget_percent == 0 or geometry_budget_percent > 100) return error.InvalidFitConfig;
    return if (geometry_budget_percent == 100)
        builderTransientBytesWithCriticalPolicy(source, query_count, maximum_range_rows, cfg, critical_core_policy)
    else
        streamedGramBuilderTransientBytesWithCriticalPolicy(source, query_count, maximum_range_rows, query_batch_rows, cfg, critical_core_policy);
}

test "12K wide critical-core planners charge fixed fit overlap and TSQR fits 256 MiB" {
    const q: usize = 12_288;
    const source: @import("attention_match.zig").Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 8_128, .dim = 512, .scale = 1 };
    const cap: usize = 256 * 1024 * 1024;
    var cfg: @import("attention_match.zig").Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 30 };
    const dense_disabled = try builderTransientBytes(source, q, 256, cfg);
    const dense_critical = try builderTransientBytesWithCriticalPolicy(source, q, 256, cfg, .train_probability_gte_0_025_no_top1_v1);
    try std.testing.expect(dense_critical > dense_disabled);
    try std.testing.expect(dense_critical > cap);
    try std.testing.expectEqual(dense_disabled, try builderTransientBytesWithCriticalPolicy(source, q, 256, cfg, .disabled));

    var tsqr_context: tsqr_value.Context = .{};
    cfg.value_backend = tsqr_context.valueBackend();
    const tsqr_critical = try builderTransientBytesWithCriticalPolicy(source, q, 256, cfg, .train_probability_gte_0_025_no_top1_v1);
    try std.testing.expect(tsqr_critical <= cap);
    try std.testing.expect(tsqr_critical < dense_critical);

    const streamed_disabled = try streamedGramBuilderTransientBytes(source, q, 256, 64, cfg);
    const streamed_critical = try streamedGramBuilderTransientBytesWithCriticalPolicy(source, q, 256, 64, cfg, .train_probability_gte_0_025_no_top1_v1);
    try std.testing.expectEqual(streamed_disabled, streamed_critical);
    try std.testing.expectEqual(streamed_disabled, try streamedGramBuilderTransientBytesWithCriticalPolicy(source, q, 256, 64, cfg, .disabled));
}

test "E2 40-consumer TSQR tiled fixed recomputation fits unchanged builder cap" {
    const q: usize = 30_720;
    const cap: usize = 256 * 1024 * 1024;
    var tsqr_context: tsqr_value.Context = .{};
    const cfg: @import("attention_match.zig").Config = .{
        .target_rows = 1,
        .max_workspace_bytes = 1 << 30,
        .value_backend = tsqr_context.valueBackend(),
    };
    for ([_]usize{ 7_936, 32_512 }) |rows| {
        const source: @import("attention_match.zig").Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = rows, .dim = 512, .scale = 1 };
        const planned = try builderTransientBytesForPreparedRoute(source, q, 256, 64, cfg, .train_probability_gte_0_025_no_top1_v1, 60);
        const without_critical = try streamedGramBuilderTransientBytes(source, q, 256, 64, cfg);
        try std.testing.expect(planned <= cap);
        try std.testing.expectEqual(without_critical, planned);
        try std.testing.expect(without_critical <= cap);
    }
}

test "E2 prepared-route planner authority is shared by preflight and execution" {
    const q: usize = 30_720;
    var tsqr_context: tsqr_value.Context = .{};
    const cfg: @import("attention_match.zig").Config = .{
        .target_rows = 1,
        .max_workspace_bytes = 1 << 30,
        .value_backend = tsqr_context.valueBackend(),
    };
    const source: @import("attention_match.zig").Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 32_512, .dim = 512, .scale = 1 };
    const preflight = try builderTransientBytesForPreparedRoute(source, q, 256, 64, cfg, .train_probability_gte_0_025_no_top1_v1, 60);
    const execution_guard = try builderTransientBytesForPreparedRoute(source, q, 256, 64, cfg, .train_probability_gte_0_025_no_top1_v1, 60);
    try std.testing.expectEqual(preflight, execution_guard);
    try std.testing.expect(preflight <= 256 * 1024 * 1024);
}

test "high fanout planner separates scalar selection from value backend" {
    const Probe = struct {
        var combined_calls: usize = 0;
        var value_calls: usize = 0;
        fn combined(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            combined_calls += 1;
            return error.InvalidAttentionShape;
        }
        fn value(_: ?*anyopaque, q: usize, m: usize, d: usize, _: f64) !usize {
            value_calls += 1;
            if (q != 30_720 or m != 128 or d != 512) return error.InvalidFitConfig;
            return 2 * 1024 * 1024;
        }
    };
    Probe.combined_calls = 0;
    Probe.value_calls = 0;
    const source: @import("attention_match.zig").Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 32_512, .dim = 512, .scale = 1 };
    const cfg: @import("attention_match.zig").Config = .{
        .target_rows = 1,
        .max_workspace_bytes = 1 << 30,
        .backend = .{ .workspace_fn = Probe.combined },
        .value_backend = .{ .workspace_fn = Probe.value, .fit_fn = undefined },
    };
    const bytes = try builderTransientBytes(source, 30_720, 256, cfg);
    try std.testing.expect(bytes > 0 and bytes < 1 << 30);
    try std.testing.expectEqual(@as(usize, 0), Probe.combined_calls);
    try std.testing.expectEqual(@as(usize, 1), Probe.value_calls);
}

pub fn runPreparedViewBoundedTimed(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
) !Output {
    return runPreparedViewBoundedTimedAccelerated(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, observe_validation_detail, phase_timing, null);
}

pub fn runPreparedViewBoundedTimedAccelerated(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
) !Output {
    return runPreparedViewBoundedTimedAcceleratedBudget(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, 100, observe_validation_detail, phase_timing, acceleration);
}

pub fn runPreparedViewBoundedTimedAcceleratedBudget(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    geometry_budget_percent: u8,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
) !Output {
    return runPreparedViewBoundedTimedAcceleratedBudgetInternal(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, geometry_budget_percent, 0, null, null, observe_validation_detail, phase_timing, acceleration, null, null);
}

/// Production budget/fallback route. Scoring is frozen once. A rejected
/// primary may rebuild geometry from that frozen scoring authority only after
/// the caller's bounded fallback authorization succeeds.
pub fn runPreparedViewBoundedTimedAcceleratedBudgets(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    geometry_budget_percent: u8,
    fallback_geometry_budget_percent: u8,
    fallback_authorize_context: ?*anyopaque,
    fallback_authorize_fn: ?*const fn (?*anyopaque) bool,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
) !Output {
    return runPreparedViewBoundedTimedAcceleratedBudgetInternal(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, geometry_budget_percent, fallback_geometry_budget_percent, fallback_authorize_context, fallback_authorize_fn, observe_validation_detail, phase_timing, acceleration, null, null);
}

pub fn runPreparedViewBoundedTimedAcceleratedPrimaryProbe(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    geometry_budget_percent: u8,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
    hook: PrimaryProbeHook,
) !Output {
    return runPreparedViewBoundedTimedAcceleratedBudgetInternal(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, geometry_budget_percent, 0, null, null, false, phase_timing, acceleration, null, hook);
}

/// Diagnostic-only synchronous geometry export. The hook borrows every slice
/// and the adaptive plan only for the duration of the call. Successful return
/// from the callback deliberately terminates before frozen replay or HHO use.
pub fn runPreparedViewBoundedTimedAcceleratedBudgetGeometryHook(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    geometry_budget_percent: u8,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
    hook: GeometryFreezeHook,
) !Output {
    return runPreparedViewBoundedTimedAcceleratedBudgetInternal(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, geometry_budget_percent, 0, null, null, false, phase_timing, acceleration, hook, null);
}

fn runPreparedViewBoundedTimedAcceleratedBudgetInternal(
    allocator: std.mem.Allocator,
    prepared: PreparedView,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    geometry_budget_percent: u8,
    fallback_geometry_budget_percent: u8,
    fallback_authorize_context: ?*anyopaque,
    fallback_authorize_fn: ?*const fn (?*anyopaque) bool,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
    acceleration: ?GeometryAcceleration,
    geometry_hook: ?GeometryFreezeHook,
    primary_probe_hook: ?PrimaryProbeHook,
) !Output {
    if (primary_probe_hook != null) {
        // The primary exporter receives TRAIN values and an authenticated
        // heldout row count through its callback. HHO values stay absent.
        try prepared.validatePrimaryTrain();
    } else try prepared.validate();
    if (comptime builtin.os.tag != .macos) if (prepared.score_policy == .accelerate_fp64_tiled_v1) return error.UnsupportedOperation;
    if (geometry_hook) |hook| try validateGeometryHookAuthority(prepared.source.rows, hook);
    const fallback_configured = fallback_geometry_budget_percent != 0 or fallback_authorize_context != null or fallback_authorize_fn != null;
    try validateBudgetProfile(prepared.budget_profile, geometry_budget_percent, fallback_geometry_budget_percent, fallback_authorize_context != null, fallback_authorize_fn != null, geometry_hook != null, prepared.critical_core_policy, prepared.postfit_policy);
    if (acceleration) |value| {
        if (!value.require_accelerated) return error.InvalidFitConfig;
    }
    if (tile_queries == 0 or builder_transient_bound_bytes == 0 or prepared.train.len % prepared.source.dim != 0)
        return error.InvalidFitConfig;
    const q = prepared.train.len / prepared.source.dim;
    const qd = try mul(q, prepared.source.dim);
    var max_rows: usize = 0;
    for (prepared.ranges) |range| max_rows = @max(max_rows, range.rows);
    const streamed_route = geometry_budget_percent != 100;
    if (prepared.critical_core_policy != .disabled and !streamed_route) return error.InvalidFitConfig;
    if (builder_transient_bound_bytes < try builderTransientBytesForPreparedRoute(prepared.source, q, max_rows, tile_queries, prepared.cfg, prepared.critical_core_policy, geometry_budget_percent))
        return error.CompactionWorkspaceLimit;
    const scratch_elements = try mul(tile_queries, max_rows);
    // Capture dispatch owns its Metal/shared transient concurrently with the
    // authenticated prepared input. Reject the combined physical peak before
    // captureNormalizer can allocate or dispatch.
    const capture_workspace = try fitter.captureWorkspaceBytes(prepared.source, prepared.train.len);
    const captured_retained = try fitter.capturedPageRetainedBytes(prepared.source, prepared.train.len);
    if (try add(prepared.external_live_bytes, capture_workspace) > prepared.combined_cap)
        return error.CompactionWorkspaceLimit;
    // Normalizer owns two f32 scalars/query and q*d f32 outputs. This exact
    // retained bound is admitted before capture or any provider dispatch.
    const y_bytes = try mul(qd, @sizeOf(f64));
    const scratch_bytes = try mul(scratch_elements, @sizeOf(f64));
    const accelerated_scratch_bytes = try acceleratedScoreScratchBytes(prepared.score_policy, scratch_elements);
    const ratio_bytes = if (acceleration != null) try mul(max_rows, @sizeOf(f64)) else 0;
    const range_cache_bytes = try mul(try mul(q, max_rows), @sizeOf(f32));
    const provider_transient = try add(try rounded(try mul(try mul(@min(tile_queries, q), max_rows), @sizeOf(f32))), std.heap.pageSize());
    const postfit_workspace = switch (prepared.postfit_policy) {
        .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => return error.V18TrainingCacheUnsupported,
        .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => return error.V18TrainingCacheUnsupported,
        .disabled => 0,
        .train_global_residual_range_mass_l2_exact_f32_cache_v1 => try replay.postfitWorkspaceReserve(q, prepared.source.rows, prepared.source.rows, prepared.ranges.len),
    };
    if (prepared.postfit_policy != .disabled and prepared.critical_core_policy == .disabled) return error.InvalidFitConfig;
    const route_external = try add(prepared.external_live_bytes, try add(postfit_workspace, try add(captured_retained, try add(y_bytes, try add(scratch_bytes, try add(accelerated_scratch_bytes, try add(ratio_bytes, try add(range_cache_bytes, provider_transient))))))));
    if (route_external > prepared.combined_cap) return error.CompactionWorkspaceLimit;

    var normalizer = try fitter.capturePageSourceWide(allocator, prepared.source, prepared.train, prepared.external_live_bytes, prepared.combined_cap, prepared.cfg.control);
    defer normalizer.deinit();
    if (normalizer.retained_bytes != captured_retained or normalizer.normalizer.outputs.len != qd) return error.InvalidAttentionShape;
    const original_y = try allocator.alloc(f64, qd);
    defer allocator.free(original_y);
    for (normalizer.normalizer.outputs, original_y, 0..) |value, *out, i| {
        if ((i & 4095) == 0) try prepared.cfg.control.check();
        out.* = value;
    }
    const scratch = try allocator.alloc(f64, scratch_elements);
    defer allocator.free(scratch);
    const accelerated_scratch: ?[]f64 = if (prepared.score_policy == .accelerate_fp64_tiled_v1) try allocator.alloc(f64, scratch_elements) else null;
    defer if (accelerated_scratch) |owned| allocator.free(owned);
    const ratios: ?[]f64 = if (acceleration != null) try allocator.alloc(f64, max_rows) else null;
    defer if (ratios) |owned| allocator.free(owned);
    const calls_before = fitter.probability_page_calls;
    const rows_before = fitter.probability_page_rows;
    var context: page_replay.Context = .{ .fitter = fitter, .normalizer = &normalizer, .original = prepared.source, .train = prepared.train, .cfg = prepared.cfg, .tile_queries = tile_queries, .builder_transient_bound_bytes = builder_transient_bound_bytes, .page_scratch = scratch, .accelerated_scratch = if (accelerated_scratch) |owned| owned else &.{}, .score_policy = prepared.score_policy, .allocator = allocator, .original_y = original_y, .range_cache_reserved_bytes = range_cache_bytes, .cache_batch_rows = normalizer.batch_rows, .delta_backend = if (acceleration) |value| value.backend else null, .require_accelerated_delta = acceleration != null, .ratio_scratch = if (ratios) |owned| owned else &.{}, .streamed_gram_selection = streamed_route, .critical_core_policy = prepared.critical_core_policy };
    defer context.deinit();
    var scored = try scoring.build(allocator, prepared.ranges, original_y, q, prepared.source.dim, route_external, prepared.combined_cap, context.builder(), context.scorer(), prepared.cfg.control);
    const scoring_bound = scored.workspace_bound_bytes;
    const bounded_score_completions = scored.score_completions;
    const bounded_score_query_rows = scored.score_query_rows;
    const bounded_reduction_policy = scored.execution_policy;
    const expected_hits = try mul(prepared.ranges.len, 3);
    if (context.cache_counters.prepare_attempts != prepared.ranges.len or context.cache_counters.builds != prepared.ranges.len or
        context.cache_counters.selection_hits != expected_hits or context.cache_counters.fit_hits != expected_hits)
        return error.V18TrainingCacheUnsupported;
    if (streamed_route) {
        var summed_range_rows: usize = 0;
        for (prepared.ranges) |range| summed_range_rows = try add(summed_range_rows, range.rows);
        const expected_probability_values = try mul(try mul(q, 3), summed_range_rows);
        if (context.streamed_gram_calls != expected_hits or
            context.streamed_gram_probability_values != expected_probability_values)
            return error.V18TrainingCacheUnsupported;
    }
    if (prepared.critical_core_policy != .disabled and context.critical_authority_builds != prepared.ranges.len)
        return error.V18TrainingCacheUnsupported;
    // Snapshot the scoring-only authority before finishGeometry deliberately
    // replaces the single range cache while traversing all three branches.
    const scoring_cache_prepares = context.cache_counters.prepare_attempts;
    const scoring_cache_selection_hits = context.cache_counters.selection_hits;
    const scoring_cache_fit_hits = context.cache_counters.fit_hits;
    // Scored artifacts own every exact-core index required by geometry. Drop
    // the per-range extraction authority before geometry changes the cache so
    // it cannot inflate or escape the admitted scoring phase.
    context.clearCriticalAuthority();
    const retained_scored_bytes = scored.retained_artifact_bytes;
    var selected_budget = geometry_budget_percent;
    var adaptive = try page_replay.finishGeometryWithBudgetPercent(allocator, &scored, prepared.ranges, original_y, q, prepared.source.dim, tile_queries, selected_budget, route_external, prepared.combined_cap, &context, prepared.cfg.control);
    var adaptive_live = true;
    defer if (adaptive_live) adaptive.deinit();
    const expected_total_prepares = try add(prepared.ranges.len, context.geometry_cache_loads);
    if (context.cache_counters.prepare_attempts != expected_total_prepares or context.cache_counters.builds != expected_total_prepares or
        context.cache_counters.selection_hits != expected_hits or context.cache_counters.fit_hits != expected_hits)
        return error.V18TrainingCacheUnsupported;
    if (adaptive.geometry_workspace_bound_bytes > prepared.combined_cap) return error.CompactionWorkspaceLimit;
    if (geometry_hook) |hook| {
        try dispatchGeometryHook(hook, .{
            .source = prepared.source,
            .source_positions = hook.source_positions,
            .source_coverage = hook.source_coverage,
            .ranges = prepared.ranges,
            .adaptive_plan = &adaptive,
            .geometry_budget_percent = selected_budget,
            .train_queries = q,
            .route_external_bytes = route_external,
            .scoring_workspace_bound_bytes = scoring_bound,
            .geometry_workspace_bound_bytes = adaptive.geometry_workspace_bound_bytes,
            .combined_cap = prepared.combined_cap,
            .critical_core_policy = prepared.critical_core_policy,
        }, prepared.cfg.control);
        return error.TrainSnapshotExported;
    }
    var replay_cfg = prepared.cfg;
    replay_cfg.max_workspace_bytes = prepared.combined_cap - route_external;
    var postfit_context: BoundedPostfitContext = .{ .page = &context, .reserve = postfit_workspace };
    if (primary_probe_hook) |hook| {
        try prepared.cfg.control.check();
        try hook.invoke_fn(hook.context, .{
            .allocator = allocator,
            .source = prepared.source,
            .train = prepared.train,
            .ranges = prepared.ranges,
            .adaptive_plan = &adaptive,
            .cfg = prepared.cfg,
            .external_live_bytes = route_external,
            .combined_cap = prepared.combined_cap,
            .postfit_hook = .{ .context = &postfit_context, .apply_fn = BoundedPostfitContext.apply },
        });
        try prepared.cfg.control.check();
        return error.TrainPrimaryProbeExported;
    }
    var result = switch (prepared.postfit_policy) {
        .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => return error.V18TrainingCacheUnsupported,
        .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => return error.V18TrainingCacheUnsupported,
        .disabled => if (prepared.critical_core_policy == .disabled)
            try replay.runFrozenAdaptivePageBackedRuntimeValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.validation)
        else
            try replay.runFrozenAdaptivePageBackedRuntimeCriticalValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.critical_core_policy, prepared.validation),
        .train_global_residual_range_mass_l2_exact_f32_cache_v1 => try replay.runFrozenAdaptivePageBackedRuntimePostfitValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.critical_core_policy, postfit_workspace, .{ .context = &postfit_context, .apply_fn = BoundedPostfitContext.apply }, prepared.validation),
    };
    var result_live = true;
    errdefer if (result_live) result.deinit();
    if (prepared.test_budget_accept_fn) |override| result.accepted = override(prepared.test_budget_accept_context, selected_budget, result.accepted);
    var fallback_attempted = false;
    if (!result.accepted and fallback_configured) {
        if (!fallback_authorize_fn.?(fallback_authorize_context)) return error.CompactionPhysicalBudgetUnsupported;
        result.deinit();
        result_live = false;
        const frozen_scores = adaptive.scores;
        const frozen_scored = adaptive.scored;
        adaptive.owns_scored_packets = false;
        adaptive.deinit();
        adaptive_live = false;
        selected_budget = fallback_geometry_budget_percent;
        adaptive = try page_replay.replanFrozenScoringWithBudgetPercent(allocator, frozen_scores, frozen_scored, retained_scored_bytes, prepared.ranges, original_y, q, prepared.source.dim, tile_queries, selected_budget, route_external, prepared.combined_cap, &context, prepared.cfg.control);
        adaptive_live = true;
        result = switch (prepared.postfit_policy) {
            .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => return error.V18TrainingCacheUnsupported,
            .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => return error.V18TrainingCacheUnsupported,
            .disabled => if (prepared.critical_core_policy == .disabled)
                try replay.runFrozenAdaptivePageBackedRuntimeValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.validation)
            else
                try replay.runFrozenAdaptivePageBackedRuntimeCriticalValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.critical_core_policy, prepared.validation),
            .train_global_residual_range_mass_l2_exact_f32_cache_v1 => try replay.runFrozenAdaptivePageBackedRuntimePostfitValidated(allocator, prepared.source, prepared.train, prepared.heldout, prepared.ranges, &adaptive, replay_cfg, observe_validation_detail, phase_timing, prepared.critical_core_policy, postfit_workspace, .{ .context = &postfit_context, .apply_fn = BoundedPostfitContext.apply }, prepared.validation),
        };
        result_live = true;
        fallback_attempted = true;
        if (prepared.test_budget_accept_fn) |override| result.accepted = override(prepared.test_budget_accept_context, selected_budget, result.accepted);
    }
    result.geometry_budget_percent = selected_budget;
    const scored_packets = try mul(prepared.ranges.len, 3);
    result.fallback_reuse = .{ .scored_packets = scored_packets, .geometry_attempts = @as(usize, if (fallback_attempted) 2 else 1), .final_validation_attempts = @as(usize, if (fallback_attempted) 2 else 1), .authorization_calls = @as(usize, @intFromBool(fallback_attempted)), .fallback_attempted = fallback_attempted, .primary_rejected_result_destroyed = fallback_attempted, .reselections_avoided = if (fallback_attempted) scored_packets else 0, .value_fit_dispatches_avoided = if (fallback_attempted) scored_packets else 0 };
    if (fitter.probability_page_calls <= calls_before or fitter.probability_page_rows <= rows_before or
        context.geometry_cache_requests == 0 or
        try add(context.geometry_cache_hits, context.geometry_cache_loads) != context.geometry_cache_requests)
        return error.V18TrainingScoreUnsupported;
    if (acceleration != null and (context.delta_evidence.tiles == 0 or context.delta_evidence.backend_calls != context.delta_evidence.tiles * 2 or context.delta_evidence.fallbacks != 0))
        return error.AcceleratedDeltaUnsupported;
    // This route deliberately has no materialized flat probability matrix.
    // `ready` means the authenticated page provider completed every requested
    // TRAIN page without fallback; the separate bounded evidence below proves
    // actual provider execution and its admitted peak.
    if (context.range_cache) |provider_cache| {
        if (provider_cache.probabilities.len != 0 or
            provider_cache.probabilities_f32.len != try mul(q, context.range_cache_rows))
            return error.V18TrainingScoreUnsupported;
    } else if (prepared.postfit_policy == .disabled) return error.V18TrainingScoreUnsupported;
    const actual_storage: training_source.ProbabilityStorage = .exact_f32_widened;
    result.probability_storage = actual_storage;
    result.training_source = .{ .requested = true, .outcome = .ready, .external_live_bytes = route_external };
    result.training_source.counters.query_batch_rows = normalizer.batch_rows;
    result.training_source.counters.probability_storage = actual_storage;
    return .{ .result = result, .evidence = .{
        .training_route = if (streamed_route) "bounded_exact_f32_streamed_gram_tsqr_v1" else "legacy_bounded_v1",
        .train_queries = q,
        .geometry_budget_percent = selected_budget,
        .budget_profile = prepared.budget_profile,
        .query_batch_rows = normalizer.batch_rows,
        .probability_storage = actual_storage,
        .provider_calls = fitter.probability_page_calls - calls_before,
        .provider_rows = fitter.probability_page_rows - rows_before,
        .workspace_bound_bytes = @max(scoring_bound, adaptive.geometry_workspace_bound_bytes),
        .scoring_cache_prepares = scoring_cache_prepares,
        .scoring_cache_selection_hits = scoring_cache_selection_hits,
        .scoring_cache_fit_hits = scoring_cache_fit_hits,
        .range_cache_prepares = context.cache_counters.prepare_attempts,
        .postfit_range_cache_prepares = postfit_context.range_cache_prepares,
        .range_cache_selection_hits = context.cache_counters.selection_hits,
        .range_cache_fit_hits = context.cache_counters.fit_hits,
        .geometry_cache_requests = context.geometry_cache_requests,
        .geometry_cache_hits = context.geometry_cache_hits,
        .geometry_cache_loads = context.geometry_cache_loads,
        .bounded_reduction_policy = bounded_reduction_policy,
        .bounded_score_completions = bounded_score_completions,
        .bounded_score_query_rows = bounded_score_query_rows,
        .bounded_score_accelerate_calls = context.accelerated_score_calls,
        .bounded_score_accelerate_tiles = context.accelerated_score_tiles,
        .bounded_score_accelerate_dgemm_calls = context.accelerated_score_dgemm_calls,
        .bounded_score_cancellation_checks = context.accelerated_score_cancellation_checks,
        .geometry_delta_accelerated = acceleration != null,
        .geometry_delta_tiles = context.delta_evidence.tiles,
        .geometry_delta_backend_calls = context.delta_evidence.backend_calls,
        .geometry_delta_packed_bytes_peak = context.delta_evidence.packed_bytes_peak,
        .geometry_delta_staged_bytes_peak = context.delta_evidence.staged_bytes_peak,
        .geometry_delta_fallbacks = context.delta_evidence.fallbacks,
        .raw_selection_policy = if (streamed_route) streamed_gram.normalized_policy else "legacy_dense_qr_raw_omp_v1",
        .raw_selection_calls = context.streamed_gram_calls,
        .raw_selection_probability_values = context.streamed_gram_probability_values,
        .raw_selection_workspace_peak_bytes = context.streamed_gram_workspace_peak_bytes,
        .raw_selection_provenance_sha256 = if (streamed_route) context.streamed_gram_hasher.finalResult() else @splat(0),
        .critical_core_policy = @tagName(prepared.critical_core_policy),
        .critical_core_policy_sha256 = replay.criticalCorePolicyDigest(prepared.critical_core_policy),
        .critical_core_authority_sha256 = if (prepared.critical_core_policy != .disabled) context.critical_authority_hasher.finalResult() else @splat(0),
        .critical_core_rows_total = context.critical_rows_total,
        .critical_core_rows_peak = context.critical_rows_peak,
    } };
}

pub fn runPreparedBoundedTimed(
    allocator: std.mem.Allocator,
    prepared: *prepared_mod.PreparedInput,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    observe_validation_detail: bool,
    phase_timing: ?*replay.PhaseTiming,
) !Output {
    return runPreparedViewBoundedTimed(allocator, PreparedView.fromOwned(prepared), fitter, tile_queries, builder_transient_bound_bytes, observe_validation_detail, phase_timing);
}

pub fn runPreparedBounded(
    allocator: std.mem.Allocator,
    prepared: *prepared_mod.PreparedInput,
    fitter: *metal.Fitter,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    observe_validation_detail: bool,
) !Output {
    return runPreparedBoundedTimed(allocator, prepared, fitter, tile_queries, builder_transient_bound_bytes, observe_validation_detail, null);
}

test "primary page probe requires absent heldout values" {
    const source: @import("attention_match.zig").Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 1, 1, 1, 1 }, .rows = 4, .dim = 1, .scale = 1 };
    var view: PreparedView = .{
        .source = source,
        .train = &.{ 0, 1 },
        .heldout = &.{},
        .ranges = &.{.{ .start = 0, .rows = 4 }},
        .cfg = .{ .target_rows = 1 },
        .external_live_bytes = 0,
        .combined_cap = 1024 * 1024,
    };
    try view.validatePrimaryTrain();
    try std.testing.expectError(error.InvalidFitConfig, view.validate());
    view.heldout = &.{0};
    try view.validate();
    try std.testing.expectError(error.InvalidFitConfig, view.validatePrimaryTrain());
}

test "bounded runtime PreparedView rejects overlapping authority before dispatch" {
    const keys = [_]f64{ 0, 1, 2, 3 };
    const values = [_]f64{ 4, 5, 6, 7 };
    const bias = [_]f64{ 0, 0, 0, 0 };
    const train = [_]f64{ 0.5, 1.5 };
    const heldout = [_]f64{2.5};
    const ranges = [_]@import("attention_match_cumulative_cache.zig").Range{ .{ .start = 0, .rows = 3 }, .{ .start = 2, .rows = 2 } };
    const view: PreparedView = .{ .source = .{ .keys = &keys, .values = &values, .bias = &bias, .rows = 4, .dim = 1, .scale = 1 }, .train = &train, .heldout = &heldout, .ranges = &ranges, .cfg = .{ .target_rows = 1 }, .external_live_bytes = 64, .combined_cap = 1024 };
    try std.testing.expectError(error.InvalidFitConfig, view.validate());
}

test "geometry hook authority is contiguous and dispatch is cancellation ordered" {
    const Hook = struct {
        fn invoke(raw: ?*anyopaque, view: GeometryFreezeView, _: @import("attention_match.zig").Control) !void {
            const calls: *usize = @ptrCast(@alignCast(raw.?));
            calls.* += 1;
            try std.testing.expectEqual(@as(u8, 91), view.geometry_budget_percent);
        }
    };
    var calls: usize = 0;
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const hook: GeometryFreezeHook = .{ .context = &calls, .source_positions = &positions, .source_coverage = &coverage, .invoke_fn = Hook.invoke };
    try validateGeometryHookAuthority(2, hook);
    const bad_coverage = [_]u64{ 1, 3 };
    const bad: GeometryFreezeHook = .{ .context = &calls, .source_positions = &positions, .source_coverage = &bad_coverage, .invoke_fn = Hook.invoke };
    try std.testing.expectError(error.InvalidFitConfig, validateGeometryHookAuthority(2, bad));
    var adaptive: replay.AdaptivePlan = undefined;
    const head: @import("attention_match.zig").Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const view: GeometryFreezeView = .{ .source = head, .source_positions = &positions, .source_coverage = &coverage, .ranges = &.{}, .adaptive_plan = &adaptive, .geometry_budget_percent = 91, .train_queries = 4, .route_external_bytes = 10, .scoring_workspace_bound_bytes = 20, .geometry_workspace_bound_bytes = 30, .combined_cap = 40 };
    try dispatchGeometryHook(hook, view, .{});
    try std.testing.expectEqual(@as(usize, 1), calls);
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, dispatchGeometryHook(hook, view, .{ .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 1), calls);
}

test "bounded route executes 72 candidates with accelerated FP64 validation" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const rows: usize = 96;
    const cap: usize = 64 * 1024 * 1024;
    var ownership_transferred = false;
    const keys = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(keys);
    @memset(keys, 0);
    const values = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(values);
    @memset(values, 1);
    const bias = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(bias);
    @memset(bias, 0);
    const train = try allocator.alloc(f64, 65);
    errdefer if (!ownership_transferred) allocator.free(train);
    for (train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64.0 - 0.5;
    const heldout = try allocator.dupe(f64, &.{ -0.25, 0.25 });
    errdefer if (!ownership_transferred) allocator.free(heldout);
    const ranges = try allocator.alloc(@import("attention_match_cumulative_cache.zig").Range, 24);
    errdefer if (!ownership_transferred) allocator.free(ranges);
    for (ranges, 0..) |*range, i| range.* = .{ .start = i * 4, .rows = 4 };
    var prepared: prepared_mod.PreparedInput = .{
        .allocator = allocator,
        .source = .{ .keys = keys, .values = values, .bias = bias, .rows = rows, .dim = 1, .scale = 1 },
        .train = train,
        .heldout = heldout,
        .ranges = ranges,
        .cfg = .{
            .target_rows = 1,
            .max_relative_mass_error = 0.05,
            .max_relative_output_rmse = 0.05,
            .max_workspace_bytes = cap,
        },
        .external_live_bytes = (rows * 3 + train.len + heldout.len) * @sizeOf(f64) + ranges.len * @sizeOf(@import("attention_match_cumulative_cache.zig").Range),
        .combined_cap = cap,
    };
    ownership_transferred = true;
    defer prepared.deinit();

    std.debug.print("wide72_fixture phase=prepared ranges={d} candidates={d} queries={d} rows={d}\n", .{ ranges.len, ranges.len * 3, train.len, rows });
    var fitter = metal.Fitter.init(allocator, null, cap) catch |err| {
        std.debug.print("wide72_fixture phase=scalar_fitter_init error={s}\n", .{@errorName(err)});
        return err;
    };
    defer fitter.deinit();
    var output = runPreparedBounded(allocator, &prepared, &fitter, 2, 1024 * 1024, false) catch |err| {
        std.debug.print("wide72_fixture phase=scalar_run error={s}\n", .{@errorName(err)});
        return err;
    };
    defer output.deinit();
    std.debug.print("wide72_fixture phase=scalar_complete accepted={} divisor={d} prepares={d} selection_hits={d} fit_hits={d} geometry_requests={d} geometry_hits={d} geometry_loads={d} range_prepares={d}\n", .{ output.result.accepted, output.result.chosen_divisor, output.evidence.scoring_cache_prepares, output.evidence.scoring_cache_selection_hits, output.evidence.scoring_cache_fit_hits, output.evidence.geometry_cache_requests, output.evidence.geometry_cache_hits, output.evidence.geometry_cache_loads, output.evidence.range_cache_prepares });

    try std.testing.expectEqual(training_source.ProbabilityStorage.exact_f32_widened, output.result.training_source.counters.probability_storage);
    try std.testing.expectEqual(output.result.probability_storage, output.evidence.probability_storage);
    try std.testing.expectEqual(output.result.probability_storage, output.result.training_source.counters.probability_storage);
    try std.testing.expectEqual(output.evidence.query_batch_rows, output.result.training_source.counters.query_batch_rows);
    try std.testing.expectEqual(ranges.len, output.evidence.scoring_cache_prepares);
    try std.testing.expectEqual(ranges.len * 3, output.evidence.scoring_cache_selection_hits);
    try std.testing.expectEqual(ranges.len * 3, output.evidence.scoring_cache_fit_hits);
    try std.testing.expect(output.evidence.geometry_cache_loads > ranges.len);
    try std.testing.expectEqual(
        output.evidence.geometry_cache_requests,
        output.evidence.geometry_cache_hits + output.evidence.geometry_cache_loads,
    );
    try std.testing.expectEqual(
        ranges.len + output.evidence.geometry_cache_loads + output.evidence.postfit_range_cache_prepares,
        output.evidence.range_cache_prepares,
    );
    try std.testing.expectEqual(@as(usize, 0), output.evidence.postfit_range_cache_prepares);

    // Exercise the complete Context -> tiled PageContext -> Context evidence
    // bridge with a fresh captured source. The scalar arm above is the exact
    // decision/quality oracle; only the numerator implementation changes.
    std.debug.print("wide72_fixture phase=scalar_assertions_complete\n", .{});
    var accelerated_fitter = metal.Fitter.init(allocator, null, cap) catch |err| {
        std.debug.print("wide72_fixture phase=accelerated_fitter_init error={s}\n", .{@errorName(err)});
        return err;
    };
    defer accelerated_fitter.deinit();
    const delta_backend = delta_kernel.accelerateBackend() orelse return error.SkipZigTest;
    var validation_context: validation_accelerate.Context = .{};
    var accelerated_view = PreparedView.fromOwned(&prepared);
    accelerated_view.validation.backend = validation_accelerate.backend(&validation_context);
    var accelerated = runPreparedViewBoundedTimedAccelerated(
        allocator,
        accelerated_view,
        &accelerated_fitter,
        2,
        1024 * 1024,
        false,
        null,
        .{ .backend = delta_backend },
    ) catch |err| {
        std.debug.print("wide72_fixture phase=accelerated_run error={s}\n", .{@errorName(err)});
        return err;
    };
    defer accelerated.deinit();
    std.debug.print("wide72_fixture phase=accelerated_complete accepted={} divisor={d} validation_calls={d} delta_tiles={d} prepares={d} selection_hits={d} fit_hits={d} geometry_requests={d} geometry_hits={d} geometry_loads={d} range_prepares={d}\n", .{ accelerated.result.accepted, accelerated.result.chosen_divisor, validation_context.successfulCalls(), accelerated.evidence.geometry_delta_tiles, accelerated.evidence.scoring_cache_prepares, accelerated.evidence.scoring_cache_selection_hits, accelerated.evidence.scoring_cache_fit_hits, accelerated.evidence.geometry_cache_requests, accelerated.evidence.geometry_cache_hits, accelerated.evidence.geometry_cache_loads, accelerated.evidence.range_cache_prepares });

    try std.testing.expectEqual(training_source.ProbabilityStorage.exact_f32_widened, accelerated.result.training_source.counters.probability_storage);
    try std.testing.expectEqual(accelerated.result.probability_storage, accelerated.evidence.probability_storage);
    try std.testing.expectEqual(accelerated.result.probability_storage, accelerated.result.training_source.counters.probability_storage);
    try std.testing.expectEqual(accelerated.evidence.query_batch_rows, accelerated.result.training_source.counters.query_batch_rows);
    try std.testing.expect(accelerated.evidence.geometry_delta_accelerated);
    try std.testing.expect(accelerated.evidence.geometry_delta_tiles > 0);
    try std.testing.expectEqual(
        accelerated.evidence.geometry_delta_tiles * 2,
        accelerated.evidence.geometry_delta_backend_calls,
    );
    try std.testing.expect(accelerated.evidence.geometry_delta_packed_bytes_peak > 0);
    try std.testing.expect(accelerated.evidence.geometry_delta_staged_bytes_peak > 0);
    try std.testing.expectEqual(@as(usize, 0), accelerated.evidence.geometry_delta_fallbacks);
    try std.testing.expect(validation_context.successfulCalls() > 0);
    std.debug.print("wide72_fixture phase=parity scalar_accepted={} accelerated_accepted={} scalar_divisor={d} accelerated_divisor={d} scalar_rows={d} accelerated_rows={d} scalar_mass={d} accelerated_mass={d} scalar_rmse={d} accelerated_rmse={d}\n", .{ output.result.accepted, accelerated.result.accepted, output.result.chosen_divisor, accelerated.result.chosen_divisor, output.result.head.rows, accelerated.result.head.rows, output.result.final_mass_error, accelerated.result.final_mass_error, output.result.final_output_rmse, accelerated.result.final_output_rmse });
    try std.testing.expectEqual(output.result.accepted, accelerated.result.accepted);
    try std.testing.expectEqual(output.result.chosen_divisor, accelerated.result.chosen_divisor);
    try std.testing.expectEqual(output.result.head.rows, accelerated.result.head.rows);
    try std.testing.expectEqual(output.result.final_metrics_valid, accelerated.result.final_metrics_valid);
    try std.testing.expectApproxEqAbs(output.result.final_mass_error, accelerated.result.final_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(output.result.final_output_rmse, accelerated.result.final_output_rmse, 1e-12);
    try std.testing.expectEqualSlices(usize, output.result.source_indices, accelerated.result.source_indices);
    try std.testing.expectEqualSlices(usize, output.result.coverage_source_indices, accelerated.result.coverage_source_indices);
    try std.testing.expectEqual(ranges.len, accelerated.evidence.scoring_cache_prepares);
    try std.testing.expectEqual(ranges.len * 3, accelerated.evidence.scoring_cache_selection_hits);
    try std.testing.expectEqual(ranges.len * 3, accelerated.evidence.scoring_cache_fit_hits);
    try std.testing.expectEqual(
        accelerated.evidence.geometry_cache_requests,
        accelerated.evidence.geometry_cache_hits + accelerated.evidence.geometry_cache_loads,
    );
    try std.testing.expectEqual(
        ranges.len + accelerated.evidence.geometry_cache_loads + accelerated.evidence.postfit_range_cache_prepares,
        accelerated.evidence.range_cache_prepares,
    );
    try std.testing.expectEqual(@as(usize, 0), accelerated.evidence.postfit_range_cache_prepares);
}

test "bounded critical policy preserves exact cores through scoring and geometry" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    if (!@import("antfly_platform").env.getenvBool("ANTFLY_METAL_ATTENTION_MATCH_FLOATPAIR_DIAGNOSTIC")) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const rows: usize = 16;
    const cap: usize = 64 * 1024 * 1024;
    var ownership_transferred = false;
    const keys = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(keys);
    const values = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(values);
    const bias = try allocator.alloc(f64, rows);
    errdefer if (!ownership_transferred) allocator.free(bias);
    for (keys, values, bias, 0..) |*key, *value, *b, i| {
        key.* = 0;
        value.* = @as(f64, @floatFromInt(100 + i));
        b.* = -20;
    }
    @memcpy(keys[8..14], &[_]f64{ -1, -0.6, -0.2, 0.2, 0.6, 1 });
    @memset(bias[8..14], -1.5);
    @memcpy(keys[14..16], &[_]f64{ -1, 1 });
    @memset(bias[14..16], 3);
    // Symmetric queries keep the threshold union exactly at global rows 14/15
    // while making the residual basis well-conditioned. Protected rows 0..7
    // remain outside the fitted range.
    const train = try allocator.dupe(f64, &.{ -1, -0.8, -0.6, -0.2, 0.2, 0.6, 0.8, 1 });
    errdefer if (!ownership_transferred) allocator.free(train);
    const heldout = try allocator.dupe(f64, &.{ -0.5, 0.5 });
    errdefer if (!ownership_transferred) allocator.free(heldout);
    const ranges = try allocator.dupe(@import("attention_match_cumulative_cache.zig").Range, &.{
        .{ .start = 8, .rows = 8 },
    });
    errdefer if (!ownership_transferred) allocator.free(ranges);
    var prepared: prepared_mod.PreparedInput = .{
        .allocator = allocator,
        .source = .{ .keys = keys, .values = values, .bias = bias, .rows = rows, .dim = 1, .scale = 1 },
        .train = train,
        .heldout = heldout,
        .ranges = ranges,
        .cfg = .{
            .target_rows = 1,
            .max_relative_mass_error = 10,
            .max_relative_output_rmse = 10,
            .max_workspace_bytes = cap,
        },
        .external_live_bytes = (rows * 3 + train.len + heldout.len) * @sizeOf(f64) + ranges.len * @sizeOf(@import("attention_match_cumulative_cache.zig").Range),
        .combined_cap = cap,
    };
    ownership_transferred = true;
    defer prepared.deinit();

    var fitter = try metal.Fitter.init(allocator, null, cap);
    defer fitter.deinit();
    const Gate = struct {
        fallback_claims: usize = 2,
        validation_calls: usize = 0,
        observed_budgets: [2]u8 = @splat(0),
        fn accept(raw: ?*anyopaque, budget: u8, independently_accepted: bool) bool {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return false));
            if (self.validation_calls < self.observed_budgets.len) self.observed_budgets[self.validation_calls] = budget;
            self.validation_calls += 1;
            return budget == 95 and independently_accepted;
        }
        fn authorize(raw: ?*anyopaque) bool {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return false));
            if (self.fallback_claims >= 3) return false;
            self.fallback_claims += 1;
            return true;
        }
    };
    var gate: Gate = .{};
    var view = PreparedView.fromOwned(&prepared);
    view.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    view.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    view.test_budget_accept_context = &gate;
    view.test_budget_accept_fn = Gate.accept;
    const transient = try streamedGramBuilderTransientBytesWithCriticalPolicy(view.source, train.len, 8, 2, view.cfg, view.critical_core_policy);
    var output = try runPreparedViewBoundedTimedAcceleratedBudgets(allocator, view, &fitter, 2, transient, 91, 95, &gate, Gate.authorize, false, null, null);
    defer output.deinit();

    try std.testing.expect(output.result.accepted);
    try std.testing.expectEqual(@as(u8, 95), output.result.geometry_budget_percent);
    try std.testing.expectEqual(@as(u8, 95), output.evidence.geometry_budget_percent);
    try std.testing.expectEqual(@as(usize, 3), gate.fallback_claims);
    try std.testing.expectEqual(@as(usize, 2), gate.validation_calls);
    try std.testing.expectEqualSlices(u8, &.{ 91, 95 }, &gate.observed_budgets);
    try std.testing.expect(output.result.fallback_reuse.fallback_attempted);
    try std.testing.expect(output.result.fallback_reuse.primary_rejected_result_destroyed);
    try std.testing.expectEqual(@as(usize, 1), output.result.fallback_reuse.authorization_calls);
    try std.testing.expectEqual(@as(usize, 2), output.result.fallback_reuse.geometry_attempts);
    try std.testing.expectEqual(@as(usize, 2), output.result.fallback_reuse.final_validation_attempts);
    try std.testing.expectEqual(ranges.len * 3, output.result.fallback_reuse.scored_packets);
    try std.testing.expectEqual(ranges.len * 3, output.result.fallback_reuse.reselections_avoided);
    try std.testing.expectEqual(replay.CriticalCorePolicy.train_probability_gte_0_025_no_top1_v1, output.result.critical_core.policy);
    try std.testing.expectEqual(replay.criticalCorePolicyDigest(.train_probability_gte_0_025_no_top1_v1), output.result.critical_core.policy_digest);
    try std.testing.expect(output.result.critical_core.unique_rows > 0);
    try std.testing.expectEqual(replay.PostfitPolicy.train_global_residual_range_mass_l2_exact_f32_cache_v1, output.result.postfit.policy);
    try std.testing.expectEqual(@as(usize, 1), output.result.postfit.attempts);
    try std.testing.expectEqual(@as(usize, 1), output.result.postfit.completions);
    try std.testing.expect(output.result.head.rows < rows);
    try std.testing.expectEqualStrings("train_probability_gte_0_025_no_top1_v1", output.evidence.critical_core_policy);
    try std.testing.expectEqual(replay.criticalCorePolicyDigest(.train_probability_gte_0_025_no_top1_v1), output.evidence.critical_core_policy_sha256);
    const zero_digest = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &output.evidence.critical_core_authority_sha256, &zero_digest));
    try std.testing.expectEqual(@as(usize, 2), output.evidence.critical_core_rows_total);
    try std.testing.expectEqual(@as(usize, 2), output.evidence.critical_core_rows_peak);
    try std.testing.expectEqual(ranges.len, output.evidence.scoring_cache_prepares);
    try std.testing.expectEqual(ranges.len * 3, output.evidence.scoring_cache_selection_hits);
    try std.testing.expectEqual(ranges.len * 3, output.evidence.scoring_cache_fit_hits);
    try std.testing.expect(output.evidence.postfit_range_cache_prepares > 0);
    try std.testing.expectEqual(
        output.evidence.scoring_cache_prepares + output.evidence.geometry_cache_loads + output.evidence.postfit_range_cache_prepares,
        output.evidence.range_cache_prepares,
    );
    try std.testing.expectEqual(replay.JointFixedMemoEvidence{}, output.result.joint_fixed_memo);
    for ([_]usize{ 14, 15 }) |global| {
        const compact = std.mem.indexOfScalar(usize, output.result.source_indices, global) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(keys[global], output.result.head.keys[compact]);
        try std.testing.expectEqual(values[global], output.result.head.values[compact]);
        try std.testing.expectEqual(bias[global], output.result.head.bias.?[compact]);
    }
    for (keys[0..8]) |value| try std.testing.expectEqual(@as(f64, 0), value);
    try std.testing.expectEqualSlices(f64, &.{ -1, -0.6, -0.2, 0.2, 0.6, 1, -1, 1 }, keys[8..16]);
    for (values, 0..) |value, i| try std.testing.expectEqual(@as(f64, @floatFromInt(100 + i)), value);
    for (bias[0..8]) |value| try std.testing.expectEqual(@as(f64, -20), value);
    for (bias[8..14]) |value| try std.testing.expectEqual(@as(f64, -1.5), value);
    for (bias[14..16]) |value| try std.testing.expectEqual(@as(f64, 3), value);
}
