//! Bounded TRAIN-only Lawson refinement of an existing group-mass L2 fit.
//!
//! This module does not publish biases or inspect heldout rows. The caller
//! supplies the already-frozen C matrix, residual-mass target, L2 solution,
//! and an evaluator that scores the exact rounded TRAIN candidate bytes.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_group_mass_lawson_exact_rounded_max_v1";
pub const policy_digest: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const Options = struct {
    maximum_iterations: usize = 8,
    maximum_line_search_steps: usize = 8,
    residual_floor: f64 = 1e-12,
};

pub const ExactEvaluator = struct {
    context: ?*anyopaque,
    evaluate_fn: *const fn (?*anyopaque, []const f64, solver.Control) anyerror!f64,

    fn evaluate(self: ExactEvaluator, alpha: []const f64, control: solver.Control) !f64 {
        const score = try self.evaluate_fn(self.context, alpha, control);
        if (!std.math.isFinite(score) or score < 0) return error.NonFiniteAttention;
        return score;
    }
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    alpha: []f64,
    baseline_exact_max: f64,
    best_exact_max: f64,
    attempted_iterations: usize,
    accepted_improvements: usize,
    nnls_sweeps: usize,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.alpha);
        self.* = undefined;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub fn workspaceBytes(query_rows: usize, groups: usize) !usize {
    if (query_rows == 0 or groups == 0) return error.InvalidFitConfig;
    var count = try mul(query_rows, groups);
    count = std.math.add(usize, count, try mul(query_rows, 4)) catch return error.CompactionWorkspaceLimit;
    count = std.math.add(usize, count, try mul(groups, 5)) catch return error.CompactionWorkspaceLimit;
    return mul(count, @sizeOf(f64));
}

fn fillResidual(c: []const f64, rhs: []const f64, q: usize, m: usize, alpha: []const f64, residual: []f64) !void {
    if (c.len != q * m or rhs.len != q or alpha.len != m or residual.len != q) return error.InvalidFitConfig;
    for (rhs, residual, 0..) |target, *out, qi| {
        var fitted: f64 = 0;
        for (alpha, 0..) |value, group| fitted += c[qi * m + group] * value;
        out.* = fitted - target;
        if (!std.math.isFinite(out.*)) return error.NonFiniteAttention;
    }
}

/// Refines `initial_alpha` without mutating it. Returned alpha is always the
/// initial exact-rounded TRAIN solution or a strictly better one. Lawson row
/// weights affect proposal generation only; publication choice is made solely
/// by the exact evaluator.
pub fn refine(allocator: std.mem.Allocator, c: []const f64, rhs: []const f64, query_rows: usize, groups: usize, initial_alpha: []const f64, cfg: solver.Config, evaluator: ExactEvaluator, options: Options) !Result {
    if (query_rows == 0 or groups == 0 or c.len != try mul(query_rows, groups) or rhs.len != query_rows or initial_alpha.len != groups or options.maximum_iterations == 0 or options.maximum_iterations > 32 or options.maximum_line_search_steps == 0 or options.maximum_line_search_steps > 16 or !std.math.isFinite(options.residual_floor) or options.residual_floor <= 0 or cfg.min_weight <= 0) return error.InvalidFitConfig;
    const local_bytes = try workspaceBytes(query_rows, groups);
    const native_bytes = try solver.workspaceBytes(groups, groups, query_rows, 1);
    if (local_bytes > cfg.max_workspace_bytes or native_bytes > cfg.max_workspace_bytes - local_bytes) return error.CompactionWorkspaceLimit;
    for (c) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    for (rhs) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    for (initial_alpha) |value| if (!std.math.isFinite(value) or value < cfg.min_weight) return error.InvalidFitConfig;
    try cfg.control.check();

    const best = try allocator.dupe(f64, initial_alpha);
    errdefer allocator.free(best);
    const current = try allocator.dupe(f64, initial_alpha);
    defer allocator.free(current);
    const previous = try allocator.alloc(f64, groups);
    defer allocator.free(previous);
    const proposed = try allocator.alloc(f64, groups);
    defer allocator.free(proposed);
    const row_weights = try allocator.alloc(f64, query_rows);
    defer allocator.free(row_weights);
    @memset(row_weights, 1);
    const weighted_c = try allocator.alloc(f64, try mul(query_rows, groups));
    defer allocator.free(weighted_c);
    const weighted_rhs = try allocator.alloc(f64, query_rows);
    defer allocator.free(weighted_rhs);
    const residual = try allocator.alloc(f64, query_rows);
    defer allocator.free(residual);
    const solver_residual = try allocator.alloc(f64, query_rows);
    defer allocator.free(solver_residual);
    const norms = try allocator.alloc(f64, groups);
    defer allocator.free(norms);

    const baseline = try evaluator.evaluate(best, cfg.control);
    var best_score = baseline;
    var attempted: usize = 0;
    var accepted: usize = 0;
    var sweeps: usize = 0;
    if (best_score == 0) return .{ .allocator = allocator, .alpha = best, .baseline_exact_max = baseline, .best_exact_max = best_score, .attempted_iterations = 0, .accepted_improvements = 0, .nnls_sweeps = 0 };

    for (0..options.maximum_iterations) |_| {
        try cfg.control.check();
        attempted += 1;
        fillResidual(c, rhs, query_rows, groups, current, residual) catch |err| switch (err) {
            error.NonFiniteAttention => return error.LawsonProposalRejected,
            else => return err,
        };
        var max_weight: f64 = 0;
        for (row_weights, residual) |*weight, value| {
            weight.* *= @max(@abs(value), options.residual_floor);
            if (!std.math.isFinite(weight.*)) return error.LawsonProposalRejected;
            max_weight = @max(max_weight, weight.*);
        }
        if (!(max_weight > 0) or !std.math.isFinite(max_weight)) return error.LawsonProposalRejected;
        for (0..query_rows) |qi| {
            row_weights[qi] /= max_weight;
            const scale = @sqrt(row_weights[qi]);
            weighted_rhs[qi] = rhs[qi] * scale;
            for (0..groups) |group| weighted_c[qi * groups + group] = c[qi * groups + group] * scale;
        }
        @memcpy(previous, current);
        @memcpy(proposed, current);
        const proposal_sweeps = solver.fitMassActiveTargetDiagnostic(allocator, weighted_c, weighted_rhs, query_rows, groups, proposed, solver_residual, norms, cfg, true) catch |err| switch (err) {
            error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention, error.InsufficientSelectionProgress => return error.LawsonProposalRejected,
            else => return err,
        };
        sweeps = @max(sweeps, proposal_sweeps);
        for (proposed) |*value| {
            if (!std.math.isFinite(value.*) or value.* < 0) return error.LawsonProposalRejected;
            value.* = @max(value.*, cfg.min_weight);
        }

        var step: f64 = 1;
        for (0..options.maximum_line_search_steps) |_| {
            try cfg.control.check();
            for (current, previous, proposed) |*out, old, next| out.* = @max(cfg.min_weight, old + step * (next - old));
            const score = evaluator.evaluate(current, cfg.control) catch |err| switch (err) {
                error.NonFiniteAttention => return error.LawsonProposalRejected,
                else => return err,
            };
            if (score < best_score) {
                @memcpy(best, current);
                best_score = score;
                accepted += 1;
            }
            step *= 0.5;
        }
        // Lawson's next row weights follow the full weighted-L2 proposal.
        // Line-search points above are publication candidates only.
        @memcpy(current, proposed);
    }
    try cfg.control.check();
    return .{ .allocator = allocator, .alpha = best, .baseline_exact_max = baseline, .best_exact_max = best_score, .attempted_iterations = attempted, .accepted_improvements = accepted, .nnls_sweeps = sweeps };
}

const MatrixEvaluator = struct {
    c: []const f64,
    rhs: []const f64,
    q: usize,
    m: usize,
    round_f32: bool = true,

    fn evaluate(raw: ?*anyopaque, alpha: []const f64, _: solver.Control) !f64 {
        const self: *MatrixEvaluator = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        var maximum: f64 = 0;
        for (self.rhs, 0..) |target, qi| {
            var fitted: f64 = 0;
            for (alpha, 0..) |value, group| {
                const exact: f64 = if (self.round_f32) @floatCast(@as(f32, @floatCast(value))) else value;
                fitted += self.c[qi * self.m + group] * exact;
            }
            maximum = @max(maximum, @abs(fitted - target));
        }
        return maximum;
    }
};

test "Lawson refinement reduces rounded TRAIN max error on an outlier objective" {
    const c = [_]f64{ 1, 1, 1 };
    const rhs = [_]f64{ 1, 1, 4 };
    var eval: MatrixEvaluator = .{ .c = &c, .rhs = &rhs, .q = 3, .m = 1 };
    var result = try refine(std.testing.allocator, &c, &rhs, 3, 1, &.{2}, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, .{ .context = &eval, .evaluate_fn = MatrixEvaluator.evaluate }, .{});
    defer result.deinit();
    try std.testing.expect(result.best_exact_max < result.baseline_exact_max);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), result.best_exact_max, 2e-5);
    try std.testing.expect(result.accepted_improvements > 0);
}

test "exact rounded evaluator prevents publishing a worse Lawson proposal" {
    const Reject = struct {
        initial: f64,
        fn evaluate(raw: ?*anyopaque, alpha: []const f64, _: solver.Control) !f64 {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
            return if (@as(f32, @floatCast(alpha[0])) == @as(f32, @floatCast(self.initial))) 0.25 else 1;
        }
    };
    var eval: Reject = .{ .initial = 2 };
    var result = try refine(std.testing.allocator, &.{ 1, 1, 1 }, &.{ 1, 1, 4 }, 3, 1, &.{2}, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, .{ .context = &eval, .evaluate_fn = Reject.evaluate }, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 0.25), result.best_exact_max);
    try std.testing.expectEqual(@as(f64, 2), result.alpha[0]);
    try std.testing.expectEqual(@as(usize, 0), result.accepted_improvements);
}

fn allocationFixture(allocator: std.mem.Allocator) !void {
    var eval: MatrixEvaluator = .{ .c = &.{ 1, 1, 1 }, .rhs = &.{ 1, 1, 4 }, .q = 3, .m = 1 };
    var result = try refine(allocator, eval.c, eval.rhs, 3, 1, &.{2}, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, .{ .context = &eval, .evaluate_fn = MatrixEvaluator.evaluate }, .{});
    result.deinit();
}

test "Lawson refinement cancellation and allocation failures are atomic" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var eval: MatrixEvaluator = .{ .c = &.{ 1, 1, 1 }, .rhs = &.{ 1, 1, 4 }, .q = 3, .m = 1 };
    try std.testing.expectError(error.Cancelled, refine(std.testing.allocator, eval.c, eval.rhs, 3, 1, &.{2}, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }, .{ .context = &eval, .evaluate_fn = MatrixEvaluator.evaluate }, .{}));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFixture, .{});
}
