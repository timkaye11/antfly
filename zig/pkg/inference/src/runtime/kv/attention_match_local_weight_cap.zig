// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! DEV-only, model-agnostic cap for selected local weights. The caller applies
//! this after selection and before value fitting or scored-candidate retention.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "positive_max_local_weight_before_value_fit_v1";

pub const Evidence = struct {
    configured: ?f64 = null,
    scored_candidates: usize = 0,
    clamped_weights: usize = 0,
    max_before: f64 = 0,
    max_after: f64 = 0,

    pub fn merge(self: *Evidence, other: Evidence) !void {
        if (self.configured != other.configured) return error.InvalidFitConfig;
        self.scored_candidates = std.math.add(usize, self.scored_candidates, other.scored_candidates) catch return error.InvalidAttentionShape;
        self.clamped_weights = std.math.add(usize, self.clamped_weights, other.clamped_weights) catch return error.InvalidAttentionShape;
        self.max_before = @max(self.max_before, other.max_before);
        self.max_after = @max(self.max_after, other.max_after);
    }
};

pub fn validate(maximum: ?f64) !void {
    if (maximum) |cap| if (!std.math.isFinite(cap) or cap <= 0) return error.InvalidFitConfig;
}

/// Null performs no scan or write, preserving the existing path exactly.
pub fn apply(weights: []f64, maximum: ?f64, control: solver.Control) !Evidence {
    try validate(maximum);
    var result: Evidence = .{ .configured = maximum };
    const cap = maximum orelse return result;
    for (weights) |*weight| {
        try control.check();
        // Active-set NNLS may retain an exact zero at its feasibility
        // boundary. A cap constrains only the upper bound; negative weights
        // remain invalid.
        if (!std.math.isFinite(weight.*) or weight.* < 0) return error.NonFiniteAttention;
        result.max_before = @max(result.max_before, weight.*);
        if (weight.* > cap) {
            weight.* = cap;
            result.clamped_weights += 1;
        }
        result.max_after = @max(result.max_after, weight.*);
    }
    try control.check();
    return result;
}

test "null cap preserves weights without scanning semantics" {
    var weights = [_]f64{ 0.5, 2.0 };
    const evidence = try apply(&weights, null, .{});
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 2.0 }, &weights);
    try std.testing.expectEqual(@as(?f64, null), evidence.configured);
    try std.testing.expectEqual(@as(usize, 0), evidence.clamped_weights);
}

test "positive cap records exact pre and post evidence" {
    var weights = [_]f64{ 0.5, 1.05, 2.8437, 9.3046 };
    const evidence = try apply(&weights, 1.05, .{});
    try std.testing.expectEqualSlices(f64, &.{ 0.5, 1.05, 1.05, 1.05 }, &weights);
    try std.testing.expectEqual(@as(usize, 2), evidence.clamped_weights);
    try std.testing.expectEqual(@as(f64, 9.3046), evidence.max_before);
    try std.testing.expectEqual(@as(f64, 1.05), evidence.max_after);
}

test "positive cap accepts an NNLS boundary zero and rejects a negative weight" {
    var weights = [_]f64{ 0, 2 };
    const evidence = try apply(&weights, 1.05, .{});
    try std.testing.expectEqualSlices(f64, &.{ 0, 1.05 }, &weights);
    try std.testing.expectEqual(@as(usize, 1), evidence.clamped_weights);
    weights[0] = -std.math.floatEps(f64);
    try std.testing.expectError(error.NonFiniteAttention, apply(&weights, 1.05, .{}));
}

test "cap rejects invalid and nonfinite inputs" {
    var weights = [_]f64{1};
    try std.testing.expectError(error.InvalidFitConfig, apply(&weights, 0, .{}));
    try std.testing.expectError(error.InvalidFitConfig, apply(&weights, -1, .{}));
    try std.testing.expectError(error.InvalidFitConfig, apply(&weights, std.math.nan(f64), .{}));
    weights[0] = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteAttention, apply(&weights, 1.05, .{}));
}

const Cancel = struct {
    checks: usize = 0,
    cancel_at: usize,
    fn check(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.checks += 1;
        if (self.checks == self.cancel_at) return error.Cancelled;
    }
};

test "cap cancellation stops before later weights are published" {
    var weights = [_]f64{ 2, 3, 4 };
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, apply(&weights, 1.05, .{ .context = &cancel, .check_fn = Cancel.check }));
    try std.testing.expectEqualSlices(f64, &.{ 1.05, 3, 4 }, &weights);
}

test "evidence merge is checked and model agnostic" {
    var total: Evidence = .{ .configured = 1.05 };
    try total.merge(.{ .configured = 1.05, .scored_candidates = 1, .clamped_weights = 2, .max_before = 9.3, .max_after = 1.05 });
    try total.merge(.{ .configured = 1.05, .scored_candidates = 2, .clamped_weights = 1, .max_before = 2.8, .max_after = 1.05 });
    try std.testing.expectEqual(@as(usize, 3), total.scored_candidates);
    try std.testing.expectEqual(@as(usize, 3), total.clamped_weights);
    try std.testing.expectEqual(@as(f64, 9.3), total.max_before);
    try std.testing.expectError(error.InvalidFitConfig, total.merge(.{ .configured = 1.1 }));
}
