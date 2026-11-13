const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");
const hybrid = @import("attention_match_recent64_future_primary_hybrid.zig");
const heldout_bundle = @import("attention_match_prepared_heldout_bundle.zig");
const solver = @import("attention_match.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");

pub const Mode = enum { established_a, incremental_reduction_c, optimized_gram_advisory_d, full128_incremental_reduction_e, full128_budget_91_95_3_f, full128_phase_balanced_g };
pub const combined_cap_bytes: usize = 3840 * 1024 * 1024;
pub const head_workspace_bytes: usize = 1024 * 1024 * 1024;
pub const host_fit_cap_bytes: usize = combined_cap_bytes - head_workspace_bytes;

pub fn policy(mode: Mode) []const u8 {
    return switch (mode) {
        .established_a => "recent32_future_primary32_v1|TRAIN=recent8096..8127+future8132+16*i|split=32|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|default_off|builtin_test_only",
        .incremental_reduction_c => "recent32_future_primary32_v1|mode=C_incremental_geometry+Accelerate_FP64_cached_TRAIN_reduction|TRAIN=recent8096..8127+future8132+16*i|split=32|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|default_off|builtin_test_only",
        .optimized_gram_advisory_d => "recent32_future_primary32_v1|mode=D_incremental_reduction+FP64_Gram_advisory_raw_selection|TRAIN=recent8096..8127+future8132+16*i|split=32|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|default_off|builtin_test_only|numerical_artifacts_may_differ",
        .full128_incremental_reduction_e => "recent64_future_primary64_v1|mode=E_incremental_geometry+Accelerate_FP64_cached_TRAIN_reduction|TRAIN=recent8064..8127+future8132+8*i|split=64|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|default_off|builtin_test_only|sample_density_diagnostic",
        .full128_budget_91_95_3_f => "recent64_future_primary64_v1|mode=F_policy_isolation_E_plus_budget91_95_3|TRAIN=recent8064..8127+future8132+8*i|split=64|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|budget=91/95/3|default_off|builtin_test_only|sample_density_diagnostic",
        .full128_phase_balanced_g => "recent64_future_phase_balanced64_v1|mode=G_E_numerics+phase_balanced_future_TRAIN|TRAIN=recent8064..8127+future8129+8*i+i%7|split=64|HHO=independent_future_retained64|critical=disabled|postfit=disabled|workspace=1GiB|aggregate=3840MiB|budget=0/0/0|default_off|builtin_test_only",
    };
}

pub fn policyDigest(mode: Mode) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy(mode), &out, .{});
    return out;
}

fn isFull128(mode: Mode) bool {
    return mode == .full128_incremental_reduction_e or mode == .full128_budget_91_95_3_f or mode == .full128_phase_balanced_g;
}

fn trainRows(mode: Mode) usize {
    return if (isFull128(mode)) 128 else 64;
}

fn trainSplit(mode: Mode) usize {
    return if (isFull128(mode)) 64 else 32;
}

pub const Result = struct {
    candidate: checkpoint.Owned,
    stats: engine.Stats,
    elapsed_ns: u64,
    source_physical_bytes: usize,
    candidate_physical_bytes: usize,
    physical_savings_ratio: f64,
    hybrid_digest: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.candidate.deinit();
        self.* = undefined;
    }
};

fn joined(allocator: std.mem.Allocator, training: hybrid.Owned, heldout: heldout_bundle.Owned, mode: Mode) ![]engine.ReferenceBatch {
    if (training.batches.len != hybrid.batch_count or heldout.batches.len != hybrid.batch_count)
        return error.InvalidRecentFuturePrimaryDiagnostic;
    const out = try allocator.alloc(engine.ReferenceBatch, hybrid.batch_count);
    errdefer allocator.free(out);
    for (training.batches, heldout.batches, out) |train, hho, *batch| {
        if (train.consumer_layer != hho.consumer_layer or train.query_head != hho.query_head or
            train.train_positions.len != trainRows(mode) or train.train.len != trainRows(mode) * hybrid.head_dim or
            train.independent_train_split == null or train.independent_train_split.? != trainSplit(mode) or
            train.independent_train_second_split != null or train.heldout.len != 0 or
            hho.train.len != 0 or hho.heldout_positions.len != 64 or hho.heldout.len != 64 * hybrid.head_dim)
            return error.InvalidRecentFuturePrimaryDiagnostic;
        for (train.train_positions) |p| for (hho.heldout_positions) |h| if (p == h)
            return error.InvalidRecentFuturePrimaryDiagnostic;
        batch.* = .{
            .consumer_layer = train.consumer_layer,
            .query_head = train.query_head,
            .train = train.train,
            .train_positions = train.train_positions,
            .heldout = hho.heldout,
            .heldout_positions = hho.heldout_positions,
            .independent_train_split = trainSplit(mode),
        };
    }
    return out;
}

pub fn runWithRejectedCapture(allocator: std.mem.Allocator, source: checkpoint.State, training: hybrid.Owned, heldout: heldout_bundle.Owned, backend: solver.Backend, control: solver.Control, mode: Mode, rejected_head_capture: ?engine.RejectedHeadCapture) !Result {
    if (comptime !@import("builtin").is_test) return error.UnsupportedRecentFuturePrimaryDiagnostic;
    if (source.logical_length != 8128 or source.donors.len != heldout.scales.len or heldout.scales.len != heldout.softcaps.len)
        return error.InvalidRecentFuturePrimaryDiagnostic;
    try control.check();
    const batches = try joined(allocator, training, heldout, mode);
    defer allocator.free(batches);
    var query_heads: [256]u32 = @splat(8);
    if (source.layer_donors.len > query_heads.len) return error.InvalidRecentFuturePrimaryDiagnostic;
    const layout: engine.Layout = .{ .query_heads_by_layer = query_heads[0..source.layer_donors.len], .scales_by_donor = heldout.scales, .softcaps_by_donor = heldout.softcaps };
    var stats: engine.Stats = .{};
    var physical_budget_context: adaptive.PhysicalBudgetContext = .{};
    const budget_policy_f = mode == .full128_budget_91_95_3_f;
    const options: engine.Options = .{
        .admission = .cumulative_ladder_4_3_2,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_fn = adaptive.engineCallback,
        .adaptive_v18_context = if (budget_policy_f) &physical_budget_context else null,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .training_position_policy = switch (mode) {
            .established_a => .recent32_future_primary32_full_source_replay_v1,
            .full128_incremental_reduction_e, .full128_budget_91_95_3_f => .recent64_future_primary64_incremental_reduction_diagnostic_v1,
            .full128_phase_balanced_g => .recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1,
            else => .recent32_future_primary32_incremental_reduction_diagnostic_v1,
        },
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .private_performance_policy = if (mode == .established_a) .owned_capture_summary_v1 else .owned_capture_summary_incremental_reduction_v1,
        .geometry_chooser_policy = if (mode == .established_a) .legacy_full_scan else .incremental_gram_v1,
        .adaptive_v18_phase_timing_diagnostic = true,
        .stats = &stats,
        .rejected_head_capture = rejected_head_capture,
        .retention_divisor = 4,
        .min_retention_divisor = 2,
        .rejected_head_policy = .preserve_original,
        .min_global_savings_ratio = 1.5,
        .storage_precision = .f16,
        .recent_rows = 2048,
        .prefix_rows = 128,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .max_parallel_validations = 1,
        .chunk_rows = 256,
        .max_queries_per_consumer = trainRows(mode),
        .max_transient_host_bytes = host_fit_cap_bytes,
        .adaptive_v18_primary_budget_percent = if (budget_policy_f) 91 else 0,
        .adaptive_v18_fallback_budget_percent = if (budget_policy_f) 95 else 0,
        .adaptive_v18_max_budget_fallbacks = if (budget_policy_f) 3 else 0,
        .checkpoint_limits = checkpoint.serving_limits,
        .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = head_workspace_bytes, .backend = backend, .control = control },
    };
    const started = platform.time.monotonicNs();
    var candidate = engine.prepare(allocator, source, batches, layout, options) catch |err| {
        std.debug.print("recent-future-primary-fit-failure-stats-json {f}\n", .{std.json.fmt(.{ .schema = "antfly.recent-future-primary-fit-failure-stats.v1", .mode = @tagName(mode), .policy = policy(mode), .error_name = @errorName(err), .elapsed_ns = platform.time.monotonicNs() -| started, .hybrid_digest = training.digest, .stats = stats, .diagnostic_only = true, .qualification_credit = false }, .{})});
        return err;
    };
    errdefer candidate.deinit();
    const source_bytes = try metal_storage.ordinaryGlobalAllocationBytesWithControl(source, checkpoint.serving_limits, control);
    const candidate_bytes = try metal_storage.globalAllocationBytesWithControl(candidate.state, checkpoint.serving_limits, control);
    if (candidate_bytes == 0) return error.InvalidRecentFuturePrimaryDiagnostic;
    const ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(candidate_bytes));
    if (stats.v18_eligible_heads != 8 or stats.v18_attempted_heads != 8 or stats.v18_accepted_heads + stats.v18_preserved_heads != 8 or
        stats.max_head_mass_error > 0.05 or stats.max_head_output_rmse > 0.05 or stats.actual_global_savings_ratio < 1.5 or ratio < 1.5)
        return error.InvalidRecentFuturePrimaryDiagnostic;
    return .{ .candidate = candidate, .stats = stats, .elapsed_ns = platform.time.monotonicNs() -| started, .source_physical_bytes = source_bytes, .candidate_physical_bytes = candidate_bytes, .physical_savings_ratio = ratio, .hybrid_digest = training.digest };
}

pub fn run(allocator: std.mem.Allocator, source: checkpoint.State, training: hybrid.Owned, heldout: heldout_bundle.Owned, backend: solver.Backend, control: solver.Control, mode: Mode) !Result {
    return runWithRejectedCapture(allocator, source, training, heldout, backend, control, mode, null);
}

test "mode E binds full128 rows split and distinct policy" {
    try std.testing.expectEqual(@as(usize, 128), trainRows(.full128_incremental_reduction_e));
    try std.testing.expectEqual(@as(usize, 64), trainSplit(.full128_incremental_reduction_e));
    try std.testing.expect(std.mem.indexOf(u8, policy(.full128_incremental_reduction_e), "recent64_future_primary64_v1") != null);
    try std.testing.expectEqual(@as(usize, 64), trainRows(.incremental_reduction_c));
    try std.testing.expectEqual(@as(usize, 32), trainSplit(.incremental_reduction_c));
}

test "mode F changes only the authenticated budget tuple from mode E" {
    try std.testing.expectEqual(trainRows(.full128_incremental_reduction_e), trainRows(.full128_budget_91_95_3_f));
    try std.testing.expectEqual(trainSplit(.full128_incremental_reduction_e), trainSplit(.full128_budget_91_95_3_f));
    try std.testing.expect(!std.mem.eql(u8, policy(.full128_incremental_reduction_e), policy(.full128_budget_91_95_3_f)));
    try std.testing.expect(std.mem.indexOf(u8, policy(.full128_incremental_reduction_e), "budget=91/95/3") == null);
    try std.testing.expect(std.mem.indexOf(u8, policy(.full128_budget_91_95_3_f), "budget=91/95/3") != null);
    const e_digest = policyDigest(.full128_incremental_reduction_e);
    const f_digest = policyDigest(.full128_budget_91_95_3_f);
    try std.testing.expect(!std.mem.eql(u8, &e_digest, &f_digest));
}

test "mode G changes only future TRAIN phase policy from mode E" {
    try std.testing.expectEqual(trainRows(.full128_incremental_reduction_e), trainRows(.full128_phase_balanced_g));
    try std.testing.expectEqual(trainSplit(.full128_incremental_reduction_e), trainSplit(.full128_phase_balanced_g));
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), head_workspace_bytes);
    try std.testing.expect(std.mem.indexOf(u8, policy(.full128_phase_balanced_g), "budget=0/0/0") != null);
    try std.testing.expect(std.mem.indexOf(u8, policy(.full128_phase_balanced_g), "phase_balanced") != null);
    try std.testing.expect(!std.mem.eql(u8, &policyDigest(.full128_incremental_reduction_e), &policyDigest(.full128_phase_balanced_g)));
}

test "mixed policy identities are distinct and retain unchanged caps" {
    try std.testing.expect(!std.mem.eql(u8, policy(.established_a), policy(.incremental_reduction_c)));
    try std.testing.expect(!std.mem.eql(u8, policy(.incremental_reduction_c), policy(.optimized_gram_advisory_d)));
    try std.testing.expect(std.mem.indexOf(u8, policy(.optimized_gram_advisory_d), "numerical_artifacts_may_differ") != null);
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), head_workspace_bytes);
    try std.testing.expectEqual(@as(usize, 3840 * 1024 * 1024), combined_cap_bytes);
}
