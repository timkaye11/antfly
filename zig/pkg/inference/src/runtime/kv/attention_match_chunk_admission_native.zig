// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const replay = @import("attention_match_chunk_admission_replay.zig");
const solver = @import("attention_match.zig");

const workspace_bytes: usize = 1024 * 1024 * 1024;

const Plan = struct {
    schema: []const u8,
    input_dir: []const u8,
    augmentation_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    input_manifest_sha256: []const u8,
    augmentation_manifest_sha256: []const u8,
    fixed_source65_manifest_sha256: []const u8,
    parent_study_tokens_sha256: []const u8,
    study_tokens_sha256: []const u8,
    method: []const u8,
    query_samples: u16,
    diagnostic_donor: usize,
    diagnostic_head: usize,
    result_path: []const u8,
    max_workspace_bytes: usize,
    retention_ladder: [3]usize,
    mass_gate: f64,
    output_gate: f64,
    heldout_controls_admission: bool,
    heldout_controls_fit: bool,
    reject_exact_active_fixed_point: bool = false,
    trace_active_nnls: bool = false,
    profile_fitter_phases: bool = false,
    selection_warm_start: bool = true,
    one_chunk_training_cache: bool = false,
    training_page_cache: bool = false,
    chunk_rows: u16 = 1024,
    chunk_rows_explicit: bool = false,
    query_batch_rows: u16 = 16,
    query_batch_rows_explicit: bool = false,
    key_layout: enum { row_major, column_major } = .row_major,
    key_layout_explicit: bool = false,
    raw_selection_solver: enum { householder, fp64_gram_advisory, range_shared_fp64_gram_original_kkt_v1 } = .householder,
    raw_selection_solver_explicit: bool = false,
    raw_selection_policy_sha256: []const u8 = "",
};

const ActiveNnlsTrace = struct {
    const primary_capacity = 56;
    const tail_capacity = 8;
    primary: [primary_capacity]solver.ActiveNnlsTraceEvent = undefined,
    primary_len: usize = 0,
    tail: [tail_capacity]solver.ActiveNnlsTraceEvent = undefined,
    total: usize = 0,

    fn record(raw: ?*anyopaque, incoming: solver.ActiveNnlsTraceEvent) void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        const ordinal = self.total;
        self.total += 1;
        var event = incoming;
        event.ordinal = ordinal;
        const power_of_two = ordinal != 0 and ordinal & (ordinal - 1) == 0;
        if ((ordinal < 8 or power_of_two) and self.primary_len < self.primary.len) {
            self.primary[self.primary_len] = event;
            self.primary_len += 1;
        }
        self.tail[ordinal % self.tail.len] = event;
    }

    fn retained(self: *const ActiveNnlsTrace, allocator: std.mem.Allocator) ![]solver.ActiveNnlsTraceEvent {
        const tail_len = @min(self.total, self.tail.len);
        const tail_start = self.total - tail_len;
        var unique_tail: usize = 0;
        for (0..tail_len) |index| {
            const event = self.tail[(tail_start + index) % self.tail.len];
            var duplicate = false;
            for (self.primary[0..self.primary_len]) |primary| {
                if (primary.ordinal == event.ordinal) duplicate = true;
            }
            if (!duplicate) unique_tail += 1;
        }
        const result = try allocator.alloc(solver.ActiveNnlsTraceEvent, self.primary_len + unique_tail);
        @memcpy(result[0..self.primary_len], self.primary[0..self.primary_len]);
        var count = self.primary_len;
        for (0..tail_len) |index| {
            const event = self.tail[(tail_start + index) % self.tail.len];
            var duplicate = false;
            for (self.primary[0..self.primary_len]) |primary| {
                if (primary.ordinal == event.ordinal) duplicate = true;
            }
            if (!duplicate) {
                result[count] = event;
                count += 1;
            }
        }
        std.mem.sort(solver.ActiveNnlsTraceEvent, result, {}, struct {
            fn less(_: void, a: solver.ActiveNnlsTraceEvent, b: solver.ActiveNnlsTraceEvent) bool {
                return a.ordinal < b.ordinal;
            }
        }.less);
        return result;
    }
};

fn policyShape(plan: Plan) bool {
    const chunk_rows_valid = plan.chunk_rows == 1024 or (plan.chunk_rows == 256 and plan.chunk_rows_explicit);
    const query_batch_valid = plan.query_batch_rows == 16 or (plan.query_batch_rows == 64 and plan.query_batch_rows_explicit);
    const key_layout_valid = plan.key_layout == .row_major or (plan.key_layout == .column_major and plan.key_layout_explicit);
    const raw_solver_valid = plan.raw_selection_solver == .householder or (plan.raw_selection_solver_explicit and (plan.raw_selection_solver != .range_shared_fp64_gram_original_kkt_v1 or plan.raw_selection_policy_sha256.len == 64));
    return chunk_rows_valid and query_batch_valid and key_layout_valid and raw_solver_valid and !(!std.mem.eql(u8, plan.schema, "antfly.attention-match-chunk-admission-plan.v12") or !std.fs.path.isAbsolute(plan.input_dir) or !std.fs.path.isAbsolute(plan.augmentation_dir) or !std.fs.path.isAbsolute(plan.result_path) or !std.mem.eql(u8, plan.method, "top1") or plan.query_samples != 512 or plan.diagnostic_donor != 5 or plan.diagnostic_head != 1 or plan.max_workspace_bytes != workspace_bytes or plan.retention_ladder[0] != 4 or plan.retention_ladder[1] != 3 or plan.retention_ladder[2] != 2 or plan.mass_gate != 0.05 or plan.output_gate != 0.05 or !plan.heldout_controls_admission or plan.heldout_controls_fit);
}

fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    return out;
}

fn finalHeadDigest(head: solver.Head) [32]u8 {
    var state = std.crypto.hash.sha2.Sha256.init(.{});
    state.update(std.mem.asBytes(&head.rows));
    state.update(std.mem.asBytes(&head.dim));
    state.update(std.mem.sliceAsBytes(head.keys));
    state.update(std.mem.sliceAsBytes(head.values));
    if (head.bias) |bias| state.update(std.mem.sliceAsBytes(bias));
    state.update(std.mem.asBytes(&head.scale));
    state.update(std.mem.asBytes(&head.softcap));
    return state.finalResult();
}

fn executionGeometryDigest(manifest_digest: [32]u8, input_chunk_rows: usize, actual_chunk_rows: usize) [32]u8 {
    var state = std.crypto.hash.sha2.Sha256.init(.{});
    state.update("antfly.attention-match-chunk-admission.execution-geometry.v1\x00");
    state.update(&manifest_digest);
    state.update(std.mem.asBytes(&input_chunk_rows));
    state.update(std.mem.asBytes(&actual_chunk_rows));
    return state.finalResult();
}

test "chunk cumulative admission source72 augmented replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_CHUNK_ADMISSION_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidWholeFitInput;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const plan = parsed.value;
    if (plan.raw_selection_solver == .range_shared_fp64_gram_original_kkt_v1) {
        const supplied = try parseDigest(plan.raw_selection_policy_sha256);
        var expected: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash("range_shared_fp64_gram_original_kkt_v1\x00module=metal_attention_match_lapack.prepareRawGram/v1", &expected, .{});
        if (!std.mem.eql(u8, &supplied, &expected)) return error.InvalidWholeFitInput;
    }
    var plan_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &plan_digest, .{});
    if (!policyShape(plan)) return error.InvalidWholeFitInput;
    const manifest_path = try std.fs.path.join(allocator, &.{ plan.input_dir, "manifest.json" });
    defer allocator.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, allocator, .limited(whole.manifest_reservation));
    defer allocator.free(manifest);
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &(try parseDigest(plan.input_manifest_sha256))) or plan.fixed_source65_manifest_sha256.len != 64) return error.InvalidWholeFitInput;
    const augmentation_path = try std.fs.path.join(allocator, &.{ plan.augmentation_dir, "manifest.json" });
    defer allocator.free(augmentation_path);
    const augmentation_manifest = try std.Io.Dir.cwd().readFileAlloc(io, augmentation_path, allocator, .limited(64 * 1024));
    defer allocator.free(augmentation_manifest);
    var augmentation_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(augmentation_manifest, &augmentation_digest, .{});
    if (!std.mem.eql(u8, &augmentation_digest, &(try parseDigest(plan.augmentation_manifest_sha256)))) return error.InvalidWholeFitInput;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(plan.identity.weights), .tokenizer = try parseDigest(plan.identity.tokenizer), .template = try parseDigest(plan.identity.template), .layout = try parseDigest(plan.identity.layout) };
    var input = try whole.read(allocator, io, .{ .output_dir = plan.input_dir, .identity = identity, .config_sha256 = try parseDigest(plan.config_sha256), .source_freeze_sha256 = try parseDigest(plan.source_freeze_sha256), .binary_sha256 = try parseDigest(plan.binary_sha256), .exporter_sha256 = try parseDigest(plan.exporter_sha256), .limits = checkpoint.serving_limits });
    defer input.deinit();
    var augmented = try augmentation.read(allocator, io, .{ .output_dir = plan.augmentation_dir, .parent_manifest_sha256 = manifest_digest, .identity = identity, .source_freeze_sha256 = try parseDigest(plan.source_freeze_sha256), .binary_sha256 = try parseDigest(plan.binary_sha256), .exporter_sha256 = try parseDigest(plan.exporter_sha256), .parent_study_tokens_sha256 = try parseDigest(plan.parent_study_tokens_sha256), .study_tokens_sha256 = try parseDigest(plan.study_tokens_sha256) });
    defer augmented.deinit();
    if (input.query_samples != 512 or input.method != .top1) return error.InvalidWholeFitInput;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, workspace_bytes);
    try fitter.setQueryBatchRows(plan.query_batch_rows);
    try fitter.setKeyLayout(switch (plan.key_layout) {
        .row_major => .row_major,
        .column_major => .column_major,
    });
    try fitter.setRawSelectionSolver(switch (plan.raw_selection_solver) {
        .householder => .householder,
        .fp64_gram_advisory => .fp64_gram_advisory,
        .range_shared_fp64_gram_original_kkt_v1 => .range_shared_fp64_gram_original_kkt_v1,
    });
    try fitter.setPhaseProfiling(plan.profile_fitter_phases);
    var live = true;
    defer if (live) fitter.deinit();
    const started = platform.time.monotonicNs();
    var trace: ActiveNnlsTrace = .{};
    const observer: ?solver.ActiveNnlsObserver = if (plan.trace_active_nnls) .{ .context = &trace, .record_fn = ActiveNnlsTrace.record } else null;
    if ((plan.one_chunk_training_cache and !plan.selection_warm_start) or (plan.training_page_cache and !plan.one_chunk_training_cache)) return error.InvalidWholeFitInput;
    var cache_counters: replay.TrainingCacheCounters = .{};
    const actual_chunk_rows: usize = if (plan.chunk_rows_explicit) plan.chunk_rows else input.fit.chunk_rows;
    var result = try replay.runInputWithPolicyObservedWarmTrainingPageCacheGeometry(allocator, &input, augmented.batches, fitter.backend(), plan.diagnostic_donor, plan.diagnostic_head, workspace_bytes, plan.reject_exact_active_fixed_point, observer, plan.selection_warm_start, plan.one_chunk_training_cache, plan.training_page_cache, &cache_counters, if (plan.chunk_rows_explicit) actual_chunk_rows else null);
    defer result.deinit();
    const fitting_phase_counters: ?Fitter.PhaseCounters = if (plan.trace_active_nnls or plan.profile_fitter_phases or plan.raw_selection_solver != .householder) fitter.phaseCounters() else null;
    const fitter_dispatches = fitter.dispatchCount();
    const actual_training_attention_calls = fitter.attentionCallCount();
    const actual_query_batch_rows = fitter.queryBatchRows();
    if (cache_counters.page_cache_enabled != plan.training_page_cache) return error.InvalidWholeFitInput;
    const actual_key_layout = fitter.keyLayout();
    const actual_raw_selection_solver = fitter.rawSelectionSolver();
    const expected_key_layout: Fitter.KeyLayout = switch (plan.key_layout) {
        .row_major => .row_major,
        .column_major => .column_major,
    };
    if (actual_key_layout != expected_key_layout) return error.InvalidWholeFitInput;
    const expected_raw_selection_solver: Fitter.RawSelectionSolver = switch (plan.raw_selection_solver) {
        .householder => .householder,
        .fp64_gram_advisory => .fp64_gram_advisory,
        .range_shared_fp64_gram_original_kkt_v1 => .range_shared_fp64_gram_original_kkt_v1,
    };
    if (actual_raw_selection_solver != expected_raw_selection_solver) return error.InvalidWholeFitInput;
    if (actual_raw_selection_solver == .fp64_gram_advisory or actual_raw_selection_solver == .range_shared_fp64_gram_original_kkt_v1) {
        const counters = fitting_phase_counters.?.gram_advisory;
        if (counters.calls == 0 or counters.candidate_attempts != counters.native_original_p_kkt_checks)
            return error.InvalidWholeFitInput;
    }
    if (plan.one_chunk_training_cache and cache_counters.full_original_training_attend_calls != actual_training_attention_calls)
        return error.InvalidWholeFitInput;
    cache_counters.full_original_training_attend_calls = actual_training_attention_calls;
    const accelerated_value_solves = fitter.deinitAndTakeValueSolves();
    live = false;
    const elapsed_ns = platform.time.monotonicNs() -| started;
    if (accelerated_value_solves == 0 or elapsed_ns == 0) return error.InvalidWholeFitInput;
    const rows_before = input.checkpoint_owned.state.donors[plan.diagnostic_donor].heads[plan.diagnostic_head].positions.len;
    const dim = input.checkpoint_owned.state.donors[plan.diagnostic_donor].heads[plan.diagnostic_head].dim;
    const source_bytes = try std.math.mul(usize, try std.math.mul(usize, rows_before, dim), 4);
    const candidate_bytes = try std.math.add(usize, try std.math.mul(usize, try std.math.mul(usize, result.head.rows, dim), 4), try std.math.mul(usize, result.head.rows, 20));
    const final_head_sha256 = finalHeadDigest(result.head);
    var trace_events: []solver.ActiveNnlsTraceEvent = &.{};
    if (plan.trace_active_nnls) trace_events = try trace.retained(allocator);
    defer if (plan.trace_active_nnls) allocator.free(trace_events);
    const execution_geometry_sha256 = executionGeometryDigest(manifest_digest, input.fit.chunk_rows, actual_chunk_rows);
    const json = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.attention-match-chunk-admission-result.v12", .authenticated = true, .input_chunk_rows = input.fit.chunk_rows, .chunk_rows = actual_chunk_rows, .chunk_rows_explicit = plan.chunk_rows_explicit, .execution_geometry_sha256 = std.fmt.bytesToHex(execution_geometry_sha256, .lower), .query_batch_rows = actual_query_batch_rows, .query_batch_rows_explicit = plan.query_batch_rows_explicit, .key_layout = actual_key_layout, .key_layout_explicit = plan.key_layout_explicit, .final_head_sha256 = std.fmt.bytesToHex(final_head_sha256, .lower), .plan_sha256 = std.fmt.bytesToHex(plan_digest, .lower), .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = std.fmt.bytesToHex(augmentation_digest, .lower), .fixed_source65_manifest_sha256 = plan.fixed_source65_manifest_sha256, .identity = plan.identity, .config_sha256 = plan.config_sha256, .parent_study_tokens_sha256 = plan.parent_study_tokens_sha256, .study_tokens_sha256 = plan.study_tokens_sha256, .source_freeze_sha256 = plan.source_freeze_sha256, .binary_sha256 = plan.binary_sha256, .exporter_sha256 = plan.exporter_sha256, .method = plan.method, .query_samples = plan.query_samples, .diagnostic_donor = plan.diagnostic_donor, .diagnostic_head = plan.diagnostic_head, .max_workspace_bytes = plan.max_workspace_bytes, .retention_ladder = plan.retention_ladder, .mass_gate = plan.mass_gate, .output_gate = plan.output_gate, .heldout_controls_admission = true, .heldout_controls_fit = false, .reject_exact_active_fixed_point = plan.reject_exact_active_fixed_point, .trace_active_nnls = plan.trace_active_nnls, .selection_warm_start = plan.selection_warm_start, .one_chunk_training_cache = plan.one_chunk_training_cache, .training_cache_counters = cache_counters, .fitter_dispatches = fitter_dispatches, .active_nnls_trace_total_events = trace.total, .active_nnls_trace_retained_events = trace_events.len, .active_nnls_trace = trace_events, .fitting_phase_counters = fitting_phase_counters, .chronological_first_passing = true, .rejected_chunk_policy = "preserve_original", .storage_precision = "f16_keys_values_f32_bias", .final_uncached_crosscheck = true, .final_mass_error = result.final_mass_error, .final_output_rmse = result.final_output_rmse, .rows_before = rows_before, .rows_after = result.head.rows, .source_head_bytes = source_bytes, .candidate_head_bytes = candidate_bytes, .single_head_savings_ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(candidate_bytes)), .global_savings_claim = false, .accelerated_value_solves = accelerated_value_solves, .elapsed_ns = elapsed_ns, .chunks = result.chunks, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false }, .{});
    defer allocator.free(json);
    var file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

test "chunk admission v12 policy shape is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const good: Plan = .{ .schema = "antfly.attention-match-chunk-admission-plan.v12", .input_dir = "/input", .augmentation_dir = "/augmentation", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .augmentation_manifest_sha256 = digest, .fixed_source65_manifest_sha256 = digest, .parent_study_tokens_sha256 = digest, .study_tokens_sha256 = digest, .method = "top1", .query_samples = 512, .diagnostic_donor = 5, .diagnostic_head = 1, .result_path = "/result", .max_workspace_bytes = workspace_bytes, .retention_ladder = .{ 4, 3, 2 }, .mass_gate = 0.05, .output_gate = 0.05, .heldout_controls_admission = true, .heldout_controls_fit = false };
    try std.testing.expect(policyShape(good));
    var legacy = good;
    legacy.query_batch_rows = 16;
    legacy.query_batch_rows_explicit = false;
    try std.testing.expect(policyShape(legacy));
    var batch64 = good;
    batch64.query_batch_rows = 64;
    batch64.query_batch_rows_explicit = true;
    try std.testing.expect(policyShape(batch64));
    var chunk256 = batch64;
    chunk256.chunk_rows = 256;
    chunk256.chunk_rows_explicit = true;
    try std.testing.expect(policyShape(chunk256));
    var implicit256 = chunk256;
    implicit256.chunk_rows_explicit = false;
    try std.testing.expect(!policyShape(implicit256));
    var invalid_chunk = chunk256;
    invalid_chunk.chunk_rows = 128;
    try std.testing.expect(!policyShape(invalid_chunk));
    var implicit64 = batch64;
    implicit64.query_batch_rows_explicit = false;
    try std.testing.expect(!policyShape(implicit64));
    var invalid_batch = good;
    invalid_batch.query_batch_rows = 32;
    invalid_batch.query_batch_rows_explicit = true;
    try std.testing.expect(!policyShape(invalid_batch));
    var advisory = good;
    advisory.raw_selection_solver = .fp64_gram_advisory;
    advisory.raw_selection_solver_explicit = true;
    try std.testing.expect(policyShape(advisory));
    advisory.raw_selection_solver_explicit = false;
    try std.testing.expect(!policyShape(advisory));
    var bad = good;
    bad.mass_gate = 0.051;
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.heldout_controls_fit = true;
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.retention_ladder = .{ 2, 3, 4 };
    try std.testing.expect(!policyShape(bad));
    bad = good;
    bad.diagnostic_head = 0;
    try std.testing.expect(!policyShape(bad));
}

test "chunk admission active NNLS trace is bounded across solve counter resets and retains tail" {
    var trace: ActiveNnlsTrace = .{};
    for (0..100) |index| ActiveNnlsTrace.record(&trace, .{
        .phase = .outer,
        .q = 8,
        .m = 4,
        .solves = index % 5,
        .best = null,
        .active_count = 2,
        .violation = 1,
        .tolerance = 1e-8,
        .weights_sha256 = @splat(@intCast(index)),
        .active_sha256 = @splat(1),
        .fixed_point_enabled = false,
    });
    const retained = try trace.retained(std.testing.allocator);
    defer std.testing.allocator.free(retained);
    try std.testing.expect(retained.len <= 64);
    try std.testing.expectEqual(@as(usize, 100), trace.total);
    try std.testing.expectEqual(@as(usize, 99), retained[retained.len - 1].ordinal);
    try std.testing.expectEqual(@as(usize, 4), retained[retained.len - 1].solves);
    // A primary power-of-two sample can also appear in the rolling tail.
    // Short traces must retain each ordinal once, including the overlap.
    for ([_]usize{ 1, 9, 17 }) |total| {
        var short: ActiveNnlsTrace = .{};
        for (0..total) |index| {
            var event = retained[0];
            event.solves = index % 5;
            ActiveNnlsTrace.record(&short, event);
        }
        const events = try short.retained(std.testing.allocator);
        defer std.testing.allocator.free(events);
        try std.testing.expectEqual(total, events.len);
        for (events, 0..) |event, index| try std.testing.expectEqual(index, event.ordinal);
    }
}
