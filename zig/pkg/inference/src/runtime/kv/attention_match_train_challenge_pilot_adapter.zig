const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");
const challenge = @import("attention_match_train_challenge_historical_selection.zig");

pub const policy = "train_challenge_historical_pilot_adapter_v1|pilot=primary91_divisor2_fixed025_l2|pilot_hho_values=absent|challenge=all_eligible_dense512_train|output=h256_chronological|final_hho_ladder=caller_only";
pub const selected_rows: usize = 256;

pub const Config = struct {
    fit: solver.Config,
    caller_external_live_bytes: usize,
    combined_cap: usize,
    heldout_query_rows: usize,
    postfit_workspace_bytes: usize,
    validation_backend: solver.Backend,
    challenge_backend: solver.Backend,
    source_authority_sha256: [32]u8,
    dense_authority_sha256: [32]u8,
    initial_historical_sha256: [32]u8,
    private_performance_policy: engine.Options.PrivatePerformancePolicy = .owned_capture_summary_incremental_reduction_critical025_v1,
    reference_summary_cache_policy: engine.Options.V18ReferenceSummaryCachePolicy = .disabled,
    expected_query_batch_rows: u16 = 64,
};

pub const Evidence = struct {
    eligible_rows: usize,
    excluded_rows: usize,
    maximum_train_challenge_error: f64,
    mean_train_challenge_error: f64,
    pilot_candidate_sha256: [32]u8,
    pilot_train_domain_sha256: [32]u8,
    selection_digest: [32]u8,
    authority_digest: [32]u8,
    pilot_heldout_observations: usize = 0,
    pilot_primary_budget_percent: u8 = 91,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    indices: []u32,
    queries_f32: []f32,
    positions: []u64,
    evidence: Evidence,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.queries_f32);
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}

fn sourceMatchesCheckpoint(source: solver.Head, head: checkpoint.Head) !void {
    try source.validate();
    const rows = head.positions.len;
    const dim: usize = head.dim;
    if (rows == 0 or dim == 0 or source.rows != rows or source.dim != dim or
        head.coverage_ends.len != rows or head.keys.len != try mul(rows, dim) or
        head.values.len != try mul(rows, dim) or head.bias.len != rows or source.bias == null)
        return error.InvalidFitConfig;
    for (source.keys, head.keys) |wide, narrow| if (wide != @as(f64, narrow)) return error.InvalidFitConfig;
    for (source.values, head.values) |wide, narrow| if (wide != @as(f64, narrow)) return error.InvalidFitConfig;
    for (source.bias.?, head.bias) |wide, narrow| if (wide != @as(f64, narrow)) return error.InvalidFitConfig;
}

fn outputBytes(dim: usize) !usize {
    var bytes = try mul(selected_rows, @sizeOf(u32));
    bytes = try add(bytes, try mul(try mul(selected_rows, dim), @sizeOf(f32)));
    return add(bytes, try mul(selected_rows, @sizeOf(u64)));
}

fn digestAuthority(config: Config, pilot_sha: [32]u8, selection_sha: [32]u8, indices: []const u32, queries: []const f32, positions: []const u64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&config.source_authority_sha256);
    hash.update(&config.dense_authority_sha256);
    hash.update(&config.initial_historical_sha256);
    hash.update(&pilot_sha);
    hash.update(&selection_sha);
    hash.update(std.mem.sliceAsBytes(indices));
    hash.update(std.mem.sliceAsBytes(queries));
    hash.update(std.mem.sliceAsBytes(positions));
    return hash.finalResult();
}

/// Builds exactly one p91/divisor2 TRAIN-only pilot, scores every eligible
/// Dense512 historical TRAIN row against it, and returns a replacement H256.
/// No heldout query values are accepted by this API. The caller owns the one
/// unchanged HHO ladder after this owner has been frozen or consumed.
pub fn derive(
    allocator: std.mem.Allocator,
    checkpoint_head: checkpoint.Head,
    source: solver.Head,
    plan: engine.HeadPlan,
    initial_train: []const f64,
    dense_queries_f32: []const f32,
    dense_positions: []const u64,
    heldout_positions: []const u64,
    scale: f64,
    softcap: f64,
    config: Config,
) !Owned {
    try config.fit.control.check();
    if (!nonzero(config.source_authority_sha256) or !nonzero(config.dense_authority_sha256) or
        !nonzero(config.initial_historical_sha256) or config.heldout_query_rows == 0 or
        config.combined_cap == 0 or config.validation_backend.summary_fn == null or
        config.challenge_backend.summary_fn == null or dense_positions.len != 512 or
        heldout_positions.len > dense_positions.len or source.scale != scale or source.softcap != softcap or
        dense_queries_f32.len != try mul(dense_positions.len, source.dim))
        return error.InvalidFitConfig;
    try sourceMatchesCheckpoint(source, checkpoint_head);
    if (initial_train.len == 0 or initial_train.len % source.dim != 0) return error.InvalidFitConfig;
    if (heldout_positions.len > 1) for (heldout_positions[1..], heldout_positions[0 .. heldout_positions.len - 1]) |next, previous| {
        if (next <= previous) return error.InvalidFitConfig;
    };
    var eligible_rows: usize = 0;
    for (dense_positions) |position| {
        eligible_rows += @intFromBool(std.mem.indexOfScalar(u64, heldout_positions, position) == null);
    }
    if (eligible_rows < selected_rows) return error.InvalidFitConfig;

    const final_bytes = try outputBytes(source.dim);
    const challenge_bytes = try challenge.workspaceBytes(eligible_rows, source.dim, selected_rows, source.rows, source.rows, config.challenge_backend);
    const retained_after_probe = try add(challenge_bytes, final_bytes);
    if (config.caller_external_live_bytes > config.combined_cap or retained_after_probe > config.combined_cap - config.caller_external_live_bytes)
        return error.CompactionWorkspaceLimit;

    if ((config.private_performance_policy == .diagnostic_capture_gram_batch256_v1) != (config.expected_query_batch_rows == 256) or
        (config.expected_query_batch_rows != 64 and config.expected_query_batch_rows != 256))
        return error.InvalidFitConfig;
    var pilot = try adaptive.runHeadWithPrimaryProbe(
        allocator,
        checkpoint_head,
        plan,
        initial_train,
        scale,
        softcap,
        config.fit,
        try add(config.caller_external_live_bytes, retained_after_probe),
        config.combined_cap,
        .accepted_f32_gram_reuse_v1,
        .flat_gram_only,
        0,
        0,
        false,
        false,
        .accelerate_fp64_summary_v1,
        config.reference_summary_cache_policy,
        .incremental_gram_v1,
        config.private_performance_policy,
        .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        false,
        true,
        .{
            .heldout_query_rows = config.heldout_query_rows,
            .critical_policy = .train_probability_gte_0_025_no_top1_v1,
            .postfit_workspace_bytes = config.postfit_workspace_bytes,
            .validation_backend = config.validation_backend,
        },
    );
    defer pilot.deinit();
    if (pilot.primary_budget_percent != 91 or pilot.candidate.head.rows > source.rows or pilot.probeCandidate().heldout_observations != 0 or pilot.critical_policy != .train_probability_gte_0_025_no_top1_v1)
        return error.InvalidFitConfig;

    const challenge_external = try add(config.caller_external_live_bytes, try add(pilot.candidate_live_bytes, final_bytes));
    var selected = try challenge.select(allocator, source, pilot.candidate.head, dense_queries_f32, dense_positions, heldout_positions, selected_rows, config.challenge_backend, challenge_external, config.combined_cap, config.fit.control);
    defer selected.deinit();

    const indices = try allocator.dupe(u32, selected.indices);
    errdefer allocator.free(indices);
    const queries = try allocator.alloc(f32, try mul(selected_rows, source.dim));
    errdefer allocator.free(queries);
    const positions = try allocator.alloc(u64, selected_rows);
    errdefer allocator.free(positions);
    for (indices, 0..) |index_u32, output_row| {
        const index: usize = index_u32;
        if (index >= dense_positions.len or (output_row > 0 and index <= indices[output_row - 1])) return error.InvalidFitConfig;
        positions[output_row] = dense_positions[index];
        @memcpy(queries[output_row * source.dim ..][0..source.dim], dense_queries_f32[index * source.dim ..][0..source.dim]);
    }
    const authority = digestAuthority(config, pilot.candidate_sha256, selected.selection_digest, indices, queries, positions);
    return .{
        .allocator = allocator,
        .indices = indices,
        .queries_f32 = queries,
        .positions = positions,
        .evidence = .{
            .eligible_rows = selected.eligible_rows,
            .excluded_rows = selected.excluded_rows,
            .maximum_train_challenge_error = selected.maximum_error,
            .mean_train_challenge_error = selected.mean_error,
            .pilot_candidate_sha256 = pilot.candidate_sha256,
            .pilot_train_domain_sha256 = pilot.train_domain_sha256,
            .selection_digest = selected.selection_digest,
            .authority_digest = authority,
        },
    };
}

test "selected Dense TRAIN rows materialize nonzero queries chronologically" {
    const dense: [12]f32 = .{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const positions: [6]u64 = .{ 10, 20, 30, 40, 50, 60 };
    const indices: [3]u32 = .{ 0, 2, 5 };
    var output: [6]f32 = undefined;
    var output_positions: [3]u64 = undefined;
    for (indices, 0..) |index, row| {
        output_positions[row] = positions[index];
        @memcpy(output[row * 2 ..][0..2], dense[index * 2 ..][0..2]);
    }
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 5, 6, 11, 12 }, &output);
    try std.testing.expectEqualSlices(u64, &.{ 10, 30, 60 }, &output_positions);
}

test "authority and cap preflight rejects before pilot allocation" {
    const source: solver.Head = .{ .keys = &.{1}, .values = &.{1}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    const head: checkpoint.Head = .{ .dim = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &.{1}, .values = &.{1}, .bias = &.{0} };
    const Backend = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {}
    };
    const backend: solver.Backend = .{ .workspace_fn = Backend.workspace, .summary_fn = Backend.summary };
    var dense: [512]f32 = @splat(1);
    var positions: [512]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = i;
    try std.testing.expectError(error.InvalidFitConfig, derive(std.testing.failing_allocator, head, source, .{ .old_rows = 1, .prefix_rows = 0, .target_rows = 1, .block_rows = 1 }, &.{1}, &dense, &positions, &.{}, 1, 0, .{
        .fit = .{ .target_rows = 1 },
        .caller_external_live_bytes = 0,
        .combined_cap = 1,
        .heldout_query_rows = 1,
        .postfit_workspace_bytes = 1,
        .validation_backend = backend,
        .challenge_backend = backend,
        .source_authority_sha256 = @splat(0),
        .dense_authority_sha256 = @splat(1),
        .initial_historical_sha256 = @splat(2),
    }));
}
