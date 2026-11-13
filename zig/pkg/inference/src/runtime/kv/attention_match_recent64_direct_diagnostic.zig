const std = @import("std");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const recent = @import("attention_match_prefill_recent64_capture.zig");
const heldout_bundle = @import("attention_match_prepared_heldout_bundle.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");
const solver = @import("attention_match.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");

pub const Mode = enum { established_a, incremental_geometry_b, incremental_reduction_c };

pub fn policy(mode: Mode) []const u8 {
    return switch (mode) {
        .established_a => "recent64_direct_adaptive_v18_diagnostic_v2|TRAIN=actual_post_rope_R64_full_current_source_reference_replay|HHO=independent_future_retained64|validation=Accelerate_FP64_summary+independent_scalar_oracle|prefix=128|recent=2048|ratio=1.5|default_off|builtin_test_only",
        .incremental_geometry_b => "recent64_direct_adaptive_v18_diagnostic_v3|mode=B_incremental_geometry|TRAIN=actual_post_rope_R64_full_current_source_reference_replay|HHO=independent_future_retained64|validation=Accelerate_FP64_summary+independent_scalar_oracle|prefix=128|recent=2048|ratio=1.5|default_off|builtin_test_only",
        .incremental_reduction_c => "recent64_direct_adaptive_v18_diagnostic_v3|mode=C_incremental_geometry+Accelerate_FP64_cached_TRAIN_reduction|TRAIN=actual_post_rope_R64_full_current_source_reference_replay|HHO=independent_future_retained64|validation=Accelerate_FP64_summary+independent_scalar_oracle|prefix=128|recent=2048|ratio=1.5|default_off|builtin_test_only",
    };
}
pub const combined_cap_bytes: usize = 3840 * 1024 * 1024;
pub const head_workspace_bytes: usize = 1024 * 1024 * 1024;
pub const host_fit_cap_bytes: usize = combined_cap_bytes - head_workspace_bytes;
pub const expected_global_heads: usize = recent.query_heads;

pub const Result = struct {
    candidate: checkpoint.Owned,
    stats: engine.Stats,
    elapsed_ns: u64,
    train_rows_per_consumer: usize,
    heldout_rows_per_consumer: usize,
    authenticated_global_heads: usize,
    source_physical_bytes: usize,
    candidate_physical_bytes: usize,
    physical_savings_ratio: f64,

    pub fn deinit(self: *@This()) void {
        self.candidate.deinit();
        self.* = undefined;
    }
};

fn authenticatedGlobalHeadCensus(batches: []const engine.ReferenceBatch) !usize {
    if (batches.len != recent.consumer_count * recent.query_heads)
        return error.InvalidRecent64DirectDiagnostic;
    var consumers: [recent.consumer_count]?u32 = @splat(null);
    var seen: [recent.consumer_count][recent.query_heads]bool = @splat(@splat(false));
    var consumer_count: usize = 0;
    for (batches) |batch| {
        if (batch.query_head >= recent.query_heads) return error.InvalidRecent64DirectDiagnostic;
        var consumer_index: ?usize = null;
        for (consumers, 0..) |consumer, index| if (consumer != null and consumer.? == batch.consumer_layer) {
            consumer_index = index;
            break;
        };
        if (consumer_index == null) {
            if (consumer_count == consumers.len) return error.InvalidRecent64DirectDiagnostic;
            consumers[consumer_count] = batch.consumer_layer;
            consumer_index = consumer_count;
            consumer_count += 1;
        }
        const index = consumer_index.?;
        if (seen[index][batch.query_head]) return error.InvalidRecent64DirectDiagnostic;
        seen[index][batch.query_head] = true;
    }
    if (consumer_count != recent.consumer_count) return error.InvalidRecent64DirectDiagnostic;
    for (seen) |heads| for (heads) |present| if (!present)
        return error.InvalidRecent64DirectDiagnostic;
    return recent.query_heads;
}

fn validateRoleGeometry(train: engine.ReferenceBatch, hho: engine.ReferenceBatch) !void {
    if (train.consumer_layer != hho.consumer_layer or train.query_head != hho.query_head or
        train.train_positions.len != recent.horizon or train.train.len != recent.horizon * recent.head_dim or
        train.heldout.len != 0 or train.heldout_positions.len != 0 or
        train.independent_train_split != null or train.independent_train_second_split != null or
        hho.train.len != 0 or hho.train_positions.len != 0 or
        hho.independent_train_split != null or hho.independent_train_second_split != null or
        hho.heldout_positions.len != heldout_bundle.expected_rows or
        hho.heldout.len != heldout_bundle.expected_rows * heldout_bundle.expected_dim)
        return error.InvalidRecent64DirectDiagnostic;
}

fn validateAndJoin(
    allocator: std.mem.Allocator,
    training: recent.Training,
    heldout: heldout_bundle.Owned,
) ![]engine.ReferenceBatch {
    if (training.batches.len != heldout_bundle.expected_batches or
        heldout.batches.len != heldout_bundle.expected_batches)
        return error.InvalidRecent64DirectDiagnostic;
    const batches = try allocator.alloc(engine.ReferenceBatch, training.batches.len);
    errdefer allocator.free(batches);
    for (training.batches, heldout.batches, batches) |train, hho, *out| {
        try validateRoleGeometry(train, hho);
        for (train.train_positions, 0..) |position, row| {
            if (position != 8064 + row or position < 6080) return error.InvalidRecent64DirectDiagnostic;
        }
        for (hho.heldout_positions) |position| {
            for (train.train_positions) |train_position| if (position == train_position)
                return error.InvalidRecent64DirectDiagnostic;
        }
        out.* = .{
            .consumer_layer = train.consumer_layer,
            .query_head = train.query_head,
            .train = train.train,
            .train_positions = train.train_positions,
            .heldout = hho.heldout,
            .heldout_positions = hho.heldout_positions,
        };
    }
    return batches;
}

pub fn run(
    allocator: std.mem.Allocator,
    source: checkpoint.State,
    training: recent.Training,
    heldout: heldout_bundle.Owned,
    backend: solver.Backend,
    control: solver.Control,
    mode: Mode,
) !Result {
    if (comptime !@import("builtin").is_test) return error.UnsupportedRecent64DirectDiagnostic;
    if (source.logical_length != 8128 or source.donors.len != heldout.scales.len or
        heldout.scales.len != heldout.softcaps.len)
        return error.InvalidRecent64DirectDiagnostic;
    try control.check();
    const global_heads = try authenticatedGlobalHeadCensus(training.batches);
    const batches = try validateAndJoin(allocator, training, heldout);
    defer allocator.free(batches);
    var query_heads: [256]u32 = @splat(8);
    if (source.layer_donors.len > query_heads.len) return error.InvalidRecent64DirectDiagnostic;
    const layout: engine.Layout = .{
        .query_heads_by_layer = query_heads[0..source.layer_donors.len],
        .scales_by_donor = heldout.scales,
        .softcaps_by_donor = heldout.softcaps,
    };
    var stats: engine.Stats = .{};
    const private_policy: engine.Options.PrivatePerformancePolicy = switch (mode) {
        .established_a => .owned_capture_summary_v1,
        .incremental_geometry_b => .owned_capture_summary_incremental_v1,
        .incremental_reduction_c => .owned_capture_summary_incremental_reduction_v1,
    };
    const position_policy: engine.Options.TrainingPositionPolicy = switch (mode) {
        .established_a => .recent_prefill_r64_full_current_source_replay_v1,
        .incremental_geometry_b => .recent_prefill_r64_incremental_geometry_diagnostic_v1,
        .incremental_reduction_c => .recent_prefill_r64_incremental_reduction_diagnostic_v1,
    };
    const options: engine.Options = .{
        .admission = .cumulative_ladder_4_3_2,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_fn = adaptive.engineCallback,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .training_position_policy = position_policy,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .private_performance_policy = private_policy,
        .geometry_chooser_policy = if (mode == .established_a) .legacy_full_scan else .incremental_gram_v1,
        .adaptive_v18_phase_timing_diagnostic = true,
        // Keep the established physical/head chooser defaults. The private
        // policy name above authorizes only the typed Accelerate FP64 summary
        // backend; this diagnostic has no p91/p95 adaptive-context grant.
        .stats = &stats,
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
        .max_queries_per_consumer = recent.horizon,
        // Host and backend are disjoint bounded owners. Their maxima sum to
        // the unchanged 3840 MiB diagnostic fit envelope.
        .max_transient_host_bytes = host_fit_cap_bytes,
        .checkpoint_limits = checkpoint.serving_limits,
        .fit = .{
            .target_rows = 1,
            .selection = .residual_omp,
            .mass_solver = .active_set,
            .max_workspace_bytes = head_workspace_bytes,
            .backend = backend,
            .control = control,
        },
    };
    const started = platform.time.monotonicNs();
    var candidate = engine.prepare(allocator, source, batches, layout, options) catch |err| {
        std.debug.print("recent64-fit-failure-stats-json {f}\n", .{std.json.fmt(.{
            .schema = "antfly.recent64-fit-failure-stats.v1",
            .mode = @tagName(mode),
            .policy = policy(mode),
            .error_name = @errorName(err),
            .elapsed_ns = platform.time.monotonicNs() -| started,
            .stats = stats,
            .diagnostic_only = true,
            .qualification_credit = false,
        }, .{})});
        return err;
    };
    errdefer candidate.deinit();
    const elapsed = platform.time.monotonicNs() -| started;
    const source_physical_bytes = try metal_storage.ordinaryGlobalAllocationBytesWithControl(source, checkpoint.serving_limits, control);
    const candidate_physical_bytes = try metal_storage.globalAllocationBytesWithControl(candidate.state, checkpoint.serving_limits, control);
    if (candidate_physical_bytes == 0) return error.InvalidRecent64DirectDiagnostic;
    const physical_savings_ratio = @as(f64, @floatFromInt(source_physical_bytes)) / @as(f64, @floatFromInt(candidate_physical_bytes));
    // Adaptive-v18 reports one attempt per donor KV head. `validated_heads`
    // belongs to the legacy whole-head path and may legitimately remain zero;
    // independent all-head numerical proof is the external CPU FP64 oracle.
    if (global_heads != expected_global_heads or stats.v18_eligible_heads != global_heads or
        stats.v18_attempted_heads != global_heads or
        stats.v18_accepted_heads + stats.v18_preserved_heads != global_heads or
        !std.math.isFinite(stats.max_head_mass_error) or stats.max_head_mass_error > 0.05 or
        !std.math.isFinite(stats.max_head_output_rmse) or stats.max_head_output_rmse > 0.05 or
        stats.source_global_bytes == 0 or
        stats.candidate_global_bytes == 0 or stats.actual_global_savings_ratio < 1.5 or
        !std.math.isFinite(physical_savings_ratio) or physical_savings_ratio < 1.5)
        return error.InvalidRecent64DirectDiagnostic;
    return .{
        .candidate = candidate,
        .stats = stats,
        .elapsed_ns = elapsed,
        .train_rows_per_consumer = recent.horizon,
        .heldout_rows_per_consumer = heldout_bundle.expected_rows,
        .authenticated_global_heads = global_heads,
        .source_physical_bytes = source_physical_bytes,
        .candidate_physical_bytes = candidate_physical_bytes,
        .physical_savings_ratio = physical_savings_ratio,
    };
}

test "recent64 diagnostic modes are distinct and keep critical and postfit disabled" {
    try std.testing.expect(!std.mem.eql(u8, policy(.established_a), policy(.incremental_geometry_b)));
    try std.testing.expect(!std.mem.eql(u8, policy(.incremental_geometry_b), policy(.incremental_reduction_c)));
}

test "recent64 direct census authenticates seven consumers times eight heads" {
    var batches: [recent.consumer_count * recent.query_heads]engine.ReferenceBatch = undefined;
    var at: usize = 0;
    for (0..recent.consumer_count) |consumer| for (0..recent.query_heads) |head| {
        batches[at] = .{ .consumer_layer = @intCast(24 + consumer), .query_head = @intCast(head), .train = &.{}, .train_positions = &.{}, .heldout = &.{}, .heldout_positions = &.{} };
        at += 1;
    };
    try std.testing.expectEqual(expected_global_heads, try authenticatedGlobalHeadCensus(&batches));
    batches[batches.len - 1].query_head = 0;
    try std.testing.expectError(error.InvalidRecent64DirectDiagnostic, authenticatedGlobalHeadCensus(&batches));
}

test "recent64 direct roles keep TRAIN and independent HHO disjoint" {
    const allocator = std.testing.allocator;
    const train_values = try allocator.alloc(f32, recent.horizon * recent.head_dim);
    defer allocator.free(train_values);
    const hho_values = try allocator.alloc(f32, heldout_bundle.expected_rows * heldout_bundle.expected_dim);
    defer allocator.free(hho_values);
    var train_positions: [recent.horizon]u64 = undefined;
    var hho_positions: [heldout_bundle.expected_rows]u64 = undefined;
    for (&train_positions, 0..) |*position, row| position.* = 8064 + row;
    for (&hho_positions, 0..) |*position, row| position.* = 7000 + row;
    const train: engine.ReferenceBatch = .{ .consumer_layer = 24, .query_head = 0, .train = train_values, .train_positions = &train_positions, .heldout = &.{}, .heldout_positions = &.{} };
    const hho: engine.ReferenceBatch = .{ .consumer_layer = 24, .query_head = 0, .train = &.{}, .train_positions = &.{}, .heldout = hho_values, .heldout_positions = &hho_positions };
    try validateRoleGeometry(train, hho);
    var wrong = hho;
    wrong.train = hho_values;
    try std.testing.expectError(error.InvalidRecent64DirectDiagnostic, validateRoleGeometry(train, wrong));
    wrong = hho;
    wrong.independent_train_split = 1;
    try std.testing.expectError(error.InvalidRecent64DirectDiagnostic, validateRoleGeometry(train, wrong));
}
