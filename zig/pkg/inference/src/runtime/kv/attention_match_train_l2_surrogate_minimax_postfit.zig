const std = @import("std");
const solver = @import("attention_match.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const group = @import("attention_match_train_group_mass_postfit.zig");
const minimax = @import("attention_match_train_surrogate_partition_minimax.zig");

pub const policy = "train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1|cache=exact_f32|exact_and_identity_fixed|validation=independent_summary|heldout=absent";
pub const policy_digest: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const Evidence = struct {
    l2: group.Diagnostics,
    minimax_applied: bool,
    minimax_scale: f64,
    baseline: solver.Metrics,
    selected: solver.Metrics,
    workspace_bytes: usize,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn inActiveRange(key: usize, ranges: []const group.Range) bool {
    for (ranges) |range| if (key >= range.start and key < range.start + range.rows) return range.target_rows != range.rows;
    return false;
}

pub fn workspaceBytes(original: solver.Head, candidate: solver.Head, query_rows: usize, ranges: usize, validation_backend: solver.Backend, cfg: solver.Config) !usize {
    const l2 = try group.workspaceBytes(query_rows, original.rows, candidate.rows, ranges);
    var validation_cfg = cfg;
    validation_cfg.backend = validation_backend;
    const validation = try solver.validationBytes(original.rows, candidate.rows, original.dim, query_rows, validation_cfg);
    var minimax_peak = try mul(query_rows, 2 * @sizeOf(f64));
    minimax_peak = try add(minimax_peak, candidate.rows);
    minimax_peak = try add(minimax_peak, try mul(candidate.rows, @sizeOf(f64)));
    minimax_peak = try add(minimax_peak, validation);
    return @max(l2, minimax_peak);
}

pub fn apply(allocator: std.mem.Allocator, original: solver.Head, candidate: *solver.Head, candidate_source_indices: []const usize, ranges: []const group.Range, exact_core_source_indices: []const usize, cached: *const training_source.Source, cache_cfg: solver.Config, validation_backend: solver.Backend, external_live_bytes: usize, combined_cap: usize) !Evidence {
    try original.validate();
    try candidate.validate();
    const train = cached.train_ptr[0..try mul(cached.query_count, cached.dim)];
    try cached.validate(original, train, cache_cfg);
    if (original.bias == null or candidate.bias == null or candidate.rows != candidate_source_indices.len or ranges.len == 0 or cached.probabilities_f32.len == 0 or cached.probability_row_start != 0 or cached.probability_rows != original.rows or validation_backend.summary_fn == null) return error.InvalidFitConfig;
    const workspace = try workspaceBytes(original, candidate.*, cached.query_count, ranges.len, validation_backend, cache_cfg);
    if (external_live_bytes > combined_cap or workspace > combined_cap - external_live_bytes or workspace > cache_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;

    // Existing proven L2 publication completes and releases its scratch first.
    const l2 = try group.apply(allocator, original, candidate.*, candidate_source_indices, ranges, exact_core_source_indices, cached, cache_cfg, external_live_bytes);

    const mask_bytes = candidate.rows;
    if (mask_bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    const surrogate_mask = try allocator.alloc(bool, candidate.rows);
    defer allocator.free(surrogate_mask);
    @memset(surrogate_mask, false);
    var surrogate_rows: usize = 0;
    for (candidate_source_indices, surrogate_mask) |key, *is_surrogate| {
        if (key >= original.rows) return error.InvalidFitConfig;
        is_surrogate.* = inActiveRange(key, ranges) and std.mem.indexOfScalar(usize, exact_core_source_indices, key) == null;
        surrogate_rows += @intFromBool(is_surrogate.*);
    }
    if (surrogate_rows == 0 or surrogate_rows == candidate.rows) return error.InvalidFitConfig;

    const fixed = try allocator.alloc(f64, cached.query_count);
    defer allocator.free(fixed);
    @memset(fixed, 0);
    const surrogate = try allocator.alloc(f64, cached.query_count);
    defer allocator.free(surrogate);
    @memset(surrogate, 0);
    for (candidate_source_indices, surrogate_mask, candidate.bias.?) |key, is_surrogate, bias| {
        const ratio = std.math.exp(bias - original.bias.?[key]);
        if (!std.math.isFinite(ratio) or ratio <= 0) return error.NonFiniteAttention;
        for (0..cached.query_count) |qi| {
            if ((qi & 31) == 0) try cache_cfg.control.check();
            const contribution = (try cached.probability(qi, key)) * ratio;
            if (is_surrogate) surrogate[qi] += contribution else fixed[qi] += contribution;
        }
    }

    var validation_cfg = cache_cfg;
    validation_cfg.backend = validation_backend;
    const Exact = struct {
        allocator: std.mem.Allocator,
        original: solver.Head,
        candidate: solver.Head,
        train: []const f64,
        cfg: solver.Config,
        fn evaluate(raw: ?*anyopaque, proposed_bias: []const f64, control: solver.Control) !solver.Metrics {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
            var proposed = self.candidate;
            proposed.bias = proposed_bias;
            var cfg = self.cfg;
            cfg.control = control;
            return solver.validationMetrics(self.allocator, self.original, proposed, self.train, cfg);
        }
    };
    var exact: Exact = .{ .allocator = allocator, .original = original, .candidate = candidate.*, .train = train, .cfg = validation_cfg };
    const ratio_live = try add(external_live_bytes, try add(mask_bytes, try mul(cached.query_count, 2 * @sizeOf(f64))));
    const validation = try solver.validationBytes(original.rows, candidate.rows, original.dim, cached.query_count, validation_cfg);
    if (validation > combined_cap - ratio_live) return error.CompactionWorkspaceLimit;
    var refined = try minimax.calibrate(allocator, candidate.bias.?, surrogate_mask, fixed, surrogate, .{ .context = &exact, .evaluate_fn = Exact.evaluate }, ratio_live, combined_cap - validation, .{ .maximum_output_rmse = cache_cfg.max_relative_output_rmse, .minimum_scale = cache_cfg.min_weight, .control = cache_cfg.control });
    defer refined.deinit();
    if (refined.applied) @memcpy(@constCast(candidate.bias.?), refined.bias);
    return .{ .l2 = l2, .minimax_applied = refined.applied, .minimax_scale = refined.scale, .baseline = refined.baseline, .selected = refined.selected, .workspace_bytes = workspace };
}

test "combined adapter applies L2 then surrogate minimax while fixed bias remains exact" {
    const Summary = struct {
        fn workspace(_: ?*anyopaque, rows: usize, _: usize, _: usize, _: usize) !usize {
            return std.math.mul(usize, rows, @sizeOf(f64));
        }
        fn run(_: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
            const probabilities = try allocator.alloc(f64, head.rows);
            defer allocator.free(probabilities);
            for (masses, 0..) |*mass, row| {
                try control.check();
                mass.* = try solver.attend(head, queries[row * head.dim ..][0..head.dim], probabilities, targets[row * head.dim ..][0..head.dim]);
            }
        }
    };
    const original: solver.Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 0, 0, 0 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    var candidate_bias = [_]f64{ 0, @log(@as(f64, 0.5)), 0 };
    var candidate: solver.Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 0, 0, 0 }, .bias = &candidate_bias, .rows = 3, .dim = 1, .scale = 1 };
    const train = [_]f64{0};
    var probabilities = [_]f32{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    var outputs = [_]f64{0};
    var cached: training_source.Source = .{ .allocator = std.testing.allocator, .keys_ptr = original.keys.ptr, .values_ptr = original.values.ptr, .bias_ptr = original.bias.?.ptr, .train_ptr = train[0..].ptr, .rows = 3, .dim = 1, .query_count = 1, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 1, .cache_bytes = 0, .consumer_peak_bytes = 1 << 20, .digest = @splat(1), .probabilities_f32 = &probabilities, .probability_rows = 3, .outputs = &outputs };
    const backend: solver.Backend = .{ .workspace_fn = Summary.workspace, .summary_fn = Summary.run };
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };
    const fixed_before = candidate_bias[0];
    const evidence = try apply(std.testing.allocator, original, &candidate, &.{ 0, 1, 2 }, &.{.{ .start = 0, .rows = 3, .target_rows = 2 }}, &.{0}, &cached, cfg, backend, 0, 1 << 20);
    try std.testing.expectEqual(fixed_before, candidate_bias[0]);
    try std.testing.expect(evidence.selected.max_relative_mass_error <= evidence.baseline.max_relative_mass_error);
}
