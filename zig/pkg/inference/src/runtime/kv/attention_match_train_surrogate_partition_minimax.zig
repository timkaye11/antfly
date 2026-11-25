const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_surrogate_partition_minimax_v2|ratio=fixed_exact_plus_scale_times_surrogate|exact_core_and_identity_bias_immutable|convex_train_minimax|publish_exact_rounded_mass_improves_and_output_gate_passes|heldout=absent";

pub const ExactEvaluator = struct {
    context: ?*anyopaque,
    evaluate_fn: *const fn (?*anyopaque, []const f64, solver.Control) anyerror!solver.Metrics,
};

pub const Config = struct {
    maximum_output_rmse: f64 = 0.05,
    minimum_scale: f64 = 1e-12,
    maximum_scale: f64 = 1e12,
    control: solver.Control = .{},
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    bias: []f64,
    applied: bool,
    scale: f64,
    baseline: solver.Metrics,
    selected: solver.Metrics,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

fn envelopeSum(fixed: []const f64, surrogate: []const f64, scale: f64) !f64 {
    var minimum = std.math.inf(f64);
    var maximum = -std.math.inf(f64);
    for (fixed, surrogate) |f, g| {
        const ratio = f + scale * g;
        if (!std.math.isFinite(ratio) or ratio <= 0) return error.NonFiniteAttention;
        minimum = @min(minimum, ratio);
        maximum = @max(maximum, ratio);
    }
    return minimum + maximum;
}

/// Minimizes max_i |fixed_i + scale*surrogate_i - 1| for positive surrogate
/// mass. The upper envelope is monotone increasing and the lower envelope is
/// monotone decreasing, so their crossing is the global convex minimum.
pub fn selectScale(fixed: []const f64, surrogate: []const f64, cfg: Config) !f64 {
    if (fixed.len == 0 or fixed.len != surrogate.len or !std.math.isFinite(cfg.minimum_scale) or !std.math.isFinite(cfg.maximum_scale) or cfg.minimum_scale <= 0 or cfg.maximum_scale < cfg.minimum_scale) return error.InvalidFitConfig;
    var any_surrogate = false;
    for (fixed, surrogate) |f, g| {
        if (!std.math.isFinite(f) or !std.math.isFinite(g) or f < 0 or g < 0 or f + g <= 0) return error.NonFiniteAttention;
        any_surrogate = any_surrogate or g > 0;
    }
    if (!any_surrogate) return error.InvalidFitConfig;
    if (try envelopeSum(fixed, surrogate, cfg.minimum_scale) >= 2) return cfg.minimum_scale;
    if (try envelopeSum(fixed, surrogate, cfg.maximum_scale) <= 2) return cfg.maximum_scale;
    var low = cfg.minimum_scale;
    var high = cfg.maximum_scale;
    for (0..80) |_| {
        try cfg.control.check();
        const middle = low + (high - low) * 0.5;
        if (try envelopeSum(fixed, surrogate, middle) < 2) low = middle else high = middle;
    }
    return low + (high - low) * 0.5;
}

pub fn calibrate(allocator: std.mem.Allocator, input_bias: []const f64, surrogate_mask: []const bool, fixed_ratio: []const f64, surrogate_ratio: []const f64, evaluator: ExactEvaluator, external_live_bytes: usize, combined_cap: usize, cfg: Config) !Result {
    try cfg.control.check();
    if (input_bias.len == 0 or surrogate_mask.len != input_bias.len or !std.math.isFinite(cfg.maximum_output_rmse) or cfg.maximum_output_rmse < 0) return error.InvalidFitConfig;
    const bytes = try std.math.mul(usize, input_bias.len, @sizeOf(f64));
    if (external_live_bytes > combined_cap or bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    const scale = try selectScale(fixed_ratio, surrogate_ratio, cfg);
    const baseline = try evaluator.evaluate_fn(evaluator.context, input_bias, cfg.control);
    if (!std.math.isFinite(baseline.max_relative_mass_error) or !std.math.isFinite(baseline.relative_output_rmse)) return error.NonFiniteAttention;
    const output = try allocator.dupe(f64, input_bias);
    errdefer allocator.free(output);
    const shift: f64 = @floatCast(@as(f32, @floatCast(@log(scale))));
    for (output, surrogate_mask) |*bias, is_surrogate| if (is_surrogate) {
        try cfg.control.check();
        bias.* = @floatCast(@as(f32, @floatCast(bias.* + shift)));
    };
    const calibrated = try evaluator.evaluate_fn(evaluator.context, output, cfg.control);
    if (!std.math.isFinite(calibrated.max_relative_mass_error) or !std.math.isFinite(calibrated.relative_output_rmse)) return error.NonFiniteAttention;
    const apply = calibrated.max_relative_mass_error < baseline.max_relative_mass_error and calibrated.relative_output_rmse <= cfg.maximum_output_rmse;
    if (!apply) @memcpy(output, input_bias);
    return .{ .allocator = allocator, .bias = output, .applied = apply, .scale = scale, .baseline = baseline, .selected = if (apply) calibrated else baseline };
}

test "convex surrogate scale centers extreme affine TRAIN ratios" {
    const fixed = [_]f64{ 0.2, 0.4 };
    const surrogate = [_]f64{ 0.4, 0.8 };
    const scale = try selectScale(&fixed, &surrogate, .{ .minimum_scale = 0.01, .maximum_scale = 10 });
    const first = fixed[0] + scale * surrogate[0];
    const second = fixed[1] + scale * surrogate[1];
    try std.testing.expectApproxEqAbs(@as(f64, 2), first + second, 1e-12);
    const optimum = @max(@abs(first - 1), @abs(second - 1));
    for ([_]f64{ scale * 0.9, scale * 1.1 }) |other| try std.testing.expect(optimum < @max(@abs(fixed[0] + other * surrogate[0] - 1), @abs(fixed[1] + other * surrogate[1] - 1)));
}

test "only surrogate biases change and exact output gate controls publication" {
    const E = struct {
        fn score(_: ?*anyopaque, bias: []const f64, _: solver.Control) !solver.Metrics {
            return .{ .max_relative_mass_error = @abs(@exp(bias[1]) * 0.5 - 1), .relative_output_rmse = 0.01 };
        }
    };
    var result = try calibrate(std.testing.allocator, &.{ 7, 0, 9 }, &.{ false, true, false }, &.{0.5}, &.{0.25}, .{ .context = null, .evaluate_fn = E.score }, 0, 1024, .{});
    defer result.deinit();
    try std.testing.expect(result.applied);
    try std.testing.expectEqual(@as(f64, 7), result.bias[0]);
    try std.testing.expectEqual(@as(f64, 9), result.bias[2]);
    try std.testing.expect(result.selected.max_relative_mass_error < result.baseline.max_relative_mass_error);
}

test "surrogate minimax rejects output regression and preserves input atomically" {
    const E = struct {
        fn score(_: ?*anyopaque, bias: []const f64, _: solver.Control) !solver.Metrics {
            return .{ .max_relative_mass_error = @abs(@exp(bias[1]) * 0.5 - 1), .relative_output_rmse = if (bias[1] == 0) 0.01 else 0.06 };
        }
    };
    const original = [_]f64{ 7, 0, 9 };
    var result = try calibrate(std.testing.allocator, &original, &.{ false, true, false }, &.{0.5}, &.{0.25}, .{ .context = null, .evaluate_fn = E.score }, 0, 1024, .{});
    defer result.deinit();
    try std.testing.expect(!result.applied);
    try std.testing.expectEqualSlices(f64, &original, result.bias);
    try std.testing.expectEqual(result.baseline.max_relative_mass_error, result.selected.max_relative_mass_error);
}

test "surrogate minimax allocation failure and mid search cancellation publish nothing" {
    const E = struct {
        fn score(_: ?*anyopaque, bias: []const f64, _: solver.Control) !solver.Metrics {
            return .{ .max_relative_mass_error = @abs(@exp(bias[1]) * 0.5 - 1), .relative_output_rmse = 0.01 };
        }
        fn run(a: std.mem.Allocator) !void {
            var result = try calibrate(a, &.{ 7, 0, 9 }, &.{ false, true, false }, &.{0.5}, &.{0.25}, .{ .context = null, .evaluate_fn = score }, 0, 1024, .{});
            result.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, E.run, .{});
    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 10) return error.Cancelled;
        }
    };
    var state: Cancel = .{};
    try std.testing.expectError(error.Cancelled, calibrate(std.testing.allocator, &.{ 7, 0, 9 }, &.{ false, true, false }, &.{0.5}, &.{0.25}, .{ .context = null, .evaluate_fn = E.score }, 0, 1024, .{ .control = .{ .context = &state, .check_fn = Cancel.check } }));
    try std.testing.expectError(error.CompactionWorkspaceLimit, calibrate(std.testing.failing_allocator, &.{ 7, 0, 9 }, &.{ false, true, false }, &.{0.5}, &.{0.25}, .{ .context = null, .evaluate_fn = E.score }, 1000, 1023, .{}));
}
