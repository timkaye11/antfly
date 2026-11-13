// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! TRAIN-only calibration of one common exact-core probability threshold.
//! The caller authenticates and owns `maximum_probability`; this module never
//! reads heldout values and never changes the supplied physical budget.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");

pub const policy = "train_max_probability_budget_calibrated_common_threshold_v1";
pub const floor_probability: f64 = 0.0025;
pub const ceiling_probability: f64 = 0.025;

pub fn policyDigest() [32]u8 {
    @setEvalBranchQuota(100000);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy ++ ";floor=0.0025;ceiling=0.025;inclusive=true;common=true;base=divisor4;cap=divisor2-times-primary-percent;ties=source-stable;heldout=forbidden");
    return hash.finalResult();
}

pub const Config = struct {
    primary_percent: u8,
    max_workspace_bytes: usize = 64 * 1024,
    control: solver.Control = .{},
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    threshold: f64,
    range_critical_counts: []usize,
    critical_rows: usize,
    minimum_middle_rows: usize,
    middle_cap_rows: usize,
    policy_sha256: [32]u8 = policyDigest(),

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.range_critical_counts);
        self.* = undefined;
    }
};

fn fitRows(ranges: []const admission.Range, source_rows: usize) !usize {
    var total: usize = 0;
    var prior_end: ?usize = null;
    for (ranges) |range| {
        if (range.rows < 2 or range.start > source_rows or range.rows > source_rows - range.start) return error.InvalidAttentionShape;
        if (prior_end) |end| if (range.start < end) return error.InvalidAttentionShape;
        prior_end = range.start + range.rows;
        total = try std.math.add(usize, total, range.rows);
    }
    return total;
}

fn middleCap(ranges: []const admission.Range, percent: u8) !usize {
    if (percent == 0 or percent > 100) return error.InvalidFitConfig;
    var divisor2: usize = 0;
    for (ranges) |range| divisor2 = try std.math.add(usize, divisor2, std.math.divCeil(usize, range.rows, 2) catch return error.InvalidFitConfig);
    return try std.math.mul(usize, divisor2, percent) / 100;
}

const Evaluation = struct { feasible: bool, critical_rows: usize, minimum_middle_rows: usize };

fn evaluate(maximum_probability: []const f64, ranges: []const admission.Range, threshold: f64, cap: usize, counts: []usize, control: solver.Control) !Evaluation {
    @memset(counts, 0);
    var critical_rows: usize = 0;
    var minimum: usize = 0;
    for (ranges, counts, 0..) |range, *count, ri| {
        if (ri % 8 == 0) try control.check();
        for (maximum_probability[range.start..][0..range.rows], 0..) |value, row| {
            if (row % 32 == 0) try control.check();
            if (!std.math.isFinite(value) or value < 0 or value > 1) return error.NonFiniteAttention;
            if (value >= threshold) count.* = try std.math.add(usize, count.*, 1);
        }
        if (count.* >= range.rows - 1) return .{ .feasible = false, .critical_rows = 0, .minimum_middle_rows = 0 };
        critical_rows = try std.math.add(usize, critical_rows, count.*);
        const nominal = std.math.divCeil(usize, range.rows, 4) catch return error.InvalidFitConfig;
        minimum = try std.math.add(usize, minimum, @max(nominal, try std.math.add(usize, count.*, 1)));
    }
    return .{ .feasible = minimum <= cap, .critical_rows = critical_rows, .minimum_middle_rows = minimum };
}

pub fn select(allocator: std.mem.Allocator, maximum_probability: []const f64, ranges: []const admission.Range, cfg: Config) !Result {
    try cfg.control.check();
    if (ranges.len == 0 or maximum_probability.len == 0 or !std.math.isFinite(floor_probability) or !std.math.isFinite(ceiling_probability) or floor_probability <= 0 or floor_probability >= ceiling_probability) return error.InvalidFitConfig;
    const fit_rows = try fitRows(ranges, maximum_probability.len);
    const scratch_bytes = try std.math.mul(usize, fit_rows, @sizeOf(f64));
    const result_bytes = try std.math.mul(usize, ranges.len, @sizeOf(usize));
    if (try std.math.add(usize, scratch_bytes, result_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const cap = try middleCap(ranges, cfg.primary_percent);
    const counts = try allocator.alloc(usize, ranges.len);
    errdefer allocator.free(counts);
    var scratch = try allocator.alloc(f64, fit_rows);
    defer allocator.free(scratch);
    var at: usize = 0;
    for (ranges) |range| {
        for (maximum_probability[range.start..][0..range.rows]) |value| if (!std.math.isFinite(value) or value < 0 or value > 1) return error.NonFiniteAttention;
        @memcpy(scratch[at..][0..range.rows], maximum_probability[range.start..][0..range.rows]);
        at += range.rows;
    }
    std.mem.sort(f64, scratch, {}, std.sort.asc(f64));

    const mandatory = try evaluate(maximum_probability, ranges, ceiling_probability, cap, counts, cfg.control);
    if (!mandatory.feasible) return error.GeometryBudgetInfeasible;
    var chosen = ceiling_probability;
    var chosen_eval = mandatory;
    const floor_eval = try evaluate(maximum_probability, ranges, floor_probability, cap, counts, cfg.control);
    if (floor_eval.feasible) {
        chosen = floor_probability;
        chosen_eval = floor_eval;
    } else {
        var prior: ?f64 = null;
        for (scratch, 0..) |breakpoint, i| {
            if (i % 32 == 0) try cfg.control.check();
            if (breakpoint < floor_probability or breakpoint > ceiling_probability or (prior != null and breakpoint == prior.?)) continue;
            prior = breakpoint;
            const at_breakpoint = try evaluate(maximum_probability, ranges, breakpoint, cap, counts, cfg.control);
            if (at_breakpoint.feasible) {
                chosen = breakpoint;
                chosen_eval = at_breakpoint;
                break;
            }
            const above = std.math.nextAfter(f64, breakpoint, std.math.inf(f64));
            if (above <= ceiling_probability) {
                const above_eval = try evaluate(maximum_probability, ranges, above, cap, counts, cfg.control);
                if (above_eval.feasible) {
                    chosen = above;
                    chosen_eval = above_eval;
                    break;
                }
            }
        }
    }
    const final = try evaluate(maximum_probability, ranges, chosen, cap, counts, cfg.control);
    if (!final.feasible or final.critical_rows != chosen_eval.critical_rows or final.minimum_middle_rows != chosen_eval.minimum_middle_rows) return error.InvalidFitConfig;
    return .{ .allocator = allocator, .threshold = chosen, .range_critical_counts = counts, .critical_rows = final.critical_rows, .minimum_middle_rows = final.minimum_middle_rows, .middle_cap_rows = cap };
}

test "budget calibrated threshold handles ragged tail and inclusive ties" {
    const maxima = [_]f64{ 0.025, 0.001, 0.001, 0.001, 0.025, 0.001, 0.001 };
    const ranges = [_]admission.Range{ .{ .start = 0, .rows = 4 }, .{ .start = 4, .rows = 3 } };
    var result = try select(std.testing.allocator, &maxima, &ranges, .{ .primary_percent = 100 });
    defer result.deinit();
    try std.testing.expectEqual(floor_probability, result.threshold);
    try std.testing.expectEqualSlices(usize, &.{ 1, 1 }, result.range_critical_counts);
}

test "budget raises floor at deterministic tied breakpoint" {
    const maxima = [_]f64{ 0.025, 0.02, 0.01, 0.0025, 0.025, 0.02, 0.01, 0.0025 };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 8 }};
    var result = try select(std.testing.allocator, &maxima, &ranges, .{ .primary_percent = 75 });
    defer result.deinit();
    try std.testing.expectEqual(std.math.nextAfter(f64, 0.02, std.math.inf(f64)), result.threshold);
    try std.testing.expectEqual(@as(usize, 3), result.middle_cap_rows);
    try std.testing.expect(result.minimum_middle_rows <= result.middle_cap_rows);
}

test "mandatory ceiling infeasible is typed" {
    const maxima = [_]f64{ 0.9, 0.8, 0.7, 0.6 };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 4 }};
    try std.testing.expectError(error.GeometryBudgetInfeasible, select(std.testing.allocator, &maxima, &ranges, .{ .primary_percent = 100 }));
}

const Cancel = struct {
    fn check(_: ?*anyopaque) !void {
        return error.Cancelled;
    }
};
test "budget calibration propagates cancellation" {
    const maxima = [_]f64{ 0.025, 0.01, 0.005, 0.0025 };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 4 }};
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &maxima, &ranges, .{ .primary_percent = 100, .control = .{ .check_fn = Cancel.check } }));
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    const maxima = [_]f64{ 0.025, 0.01, 0.005, 0.0025 };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 4 }};
    var result = try select(allocator, &maxima, &ranges, .{ .primary_percent = 100 });
    result.deinit();
}
test "budget calibration allocation failures unwind" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
