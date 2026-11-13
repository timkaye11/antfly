// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Training-only deterministic variable-budget geometry. This module consumes
//! already measured rounded full-head training harm; it never sees heldout.
const std = @import("std");
const solver = @import("attention_match.zig");
pub const policy = "greedy_positive_benefit_per_row_stable_source_level_v1";
pub const training_harm_authority = "cache_gpu_probability_v1_heuristic";
pub const signed_joint_policy = "greedy_positive_joint_train_objective_per_row_stable_source_level_v1";

pub const Level = enum(u2) { divisor4, divisor3, divisor2, identity };
pub const RangeScores = struct {
    start: usize,
    rows: usize,
    target_rows: [4]usize,
    mass_harm: [4]f64,
    output_harm: [4]f64,
};
pub const Weights = struct { mass: f64 = 1, output: f64 = 1 };
pub const Harm = struct { max_relative_mass: f64, relative_output_rmse: f64 };
pub const CandidateDelta = struct { mass: []const f64, numerator: []const f64 };
pub const JointRangeScores = struct {
    start: usize,
    rows: usize,
    target_rows: [4]usize,
    deltas: [4]CandidateDelta,
};
pub const Geometry = struct {
    allocator: std.mem.Allocator,
    /// False means the immutable minimum support exceeded this branch's row
    /// budget.  It is not a workspace failure and later branches remain valid.
    available: bool = true,
    levels: []Level,
    targets: []usize,
    middle_cap_rows: usize,
    middle_rows: usize,
    preserved_ranges: usize,
    fitted_ranges: usize,
    train_harm_before: f64,
    train_harm_after: f64,
    digest: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.levels);
        self.allocator.free(self.targets);
        self.* = undefined;
    }
};
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn target(rows: usize, divisor: usize) !usize {
    return std.math.divCeil(usize, rows, divisor) catch error.InvalidAttentionShape;
}
fn harm(row: RangeScores, level: Level, weights: Weights) !f64 {
    const i: usize = @backingInt(level);
    const value = weights.mass * row.mass_harm[i] + weights.output * row.output_harm[i];
    if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    return value;
}

/// Measures a rounded local candidate in original-denominator units. `rho`
/// and `candidate_rho` are q-vectors; chunk outputs are q-by-d. The exact
/// identity option passes equal inputs and therefore returns zero harm.
pub fn measureRoundedCandidateHarm(original_y: []const f64, rho: []const f64, original_chunk_y: []const f64, candidate_rho: []const f64, candidate_chunk_y: []const f64, q: usize, dim: usize, floor: f64) !Harm {
    if (q == 0 or dim == 0 or rho.len != q or candidate_rho.len != q or original_y.len != try std.math.mul(usize, q, dim) or original_chunk_y.len != original_y.len or candidate_chunk_y.len != original_y.len or !std.math.isFinite(floor) or floor <= 0) return error.InvalidFitConfig;
    var max_mass: f64 = 0;
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    for (0..q) |qi| {
        const delta = candidate_rho[qi] - rho[qi];
        const denominator = 1 + delta;
        if (!std.math.isFinite(rho[qi]) or !std.math.isFinite(candidate_rho[qi]) or rho[qi] < 0 or candidate_rho[qi] < 0 or !std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        // Z_candidate / Z_original = 1 + delta. This is the same
        // partition-mass error measured by the final FP64 validation gate.
        max_mass = @max(max_mass, @abs(delta));
        for (0..dim) |di| {
            const at = qi * dim + di;
            const predicted = (original_y[at] - original_chunk_y[at] + candidate_chunk_y[at]) / denominator;
            if (!std.math.isFinite(predicted) or !std.math.isFinite(original_y[at])) return error.NonFiniteAttention;
            const error_value = predicted - original_y[at];
            squared_error += error_value * error_value;
            squared_reference += original_y[at] * original_y[at];
        }
    }
    return .{ .max_relative_mass = max_mass, .relative_output_rmse = @sqrt(squared_error / @max(squared_reference, floor * floor)) };
}
fn digestGeometry(scores: []const RangeScores, levels: []const Level, targets: []const usize, divisor: usize, cap: usize, actual: usize, weights: Weights) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [8]u8 = undefined;
    for ([_]u64{ divisor, cap, actual, @bitCast(weights.mass), @bitCast(weights.output) }) |v| {
        std.mem.writeInt(u64, &bytes, v, .little);
        h.update(&bytes);
    }
    for (scores, levels, targets) |row, level, rows| {
        for ([_]u64{ row.start, row.rows, @backingInt(level), rows }) |v| {
            std.mem.writeInt(u64, &bytes, v, .little);
            h.update(&bytes);
        }
        for (row.target_rows) |value| {
            std.mem.writeInt(u64, &bytes, @intCast(value), .little);
            h.update(&bytes);
        }
        for (row.mass_harm) |value| {
            std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
            h.update(&bytes);
        }
        for (row.output_harm) |value| {
            std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
            h.update(&bytes);
        }
    }
    return h.finalResult();
}

/// Starts every range at ceil(rows/4), then greedily takes any higher ladder
/// upgrades with the greatest positive training-harm reduction per added row.
/// A candidate which does not fit the predeclared branch budget is skipped;
/// stable ties use the lower original source start.
pub fn choose(allocator: std.mem.Allocator, scores: []const RangeScores, branch_divisor: usize, weights: Weights) !Geometry {
    if (scores.len == 0 or (branch_divisor != 4 and branch_divisor != 3 and branch_divisor != 2) or !std.math.isFinite(weights.mass) or !std.math.isFinite(weights.output) or weights.mass < 0 or weights.output < 0 or weights.mass + weights.output == 0) return error.InvalidFitConfig;
    const levels = try allocator.alloc(Level, scores.len);
    errdefer allocator.free(levels);
    const targets = try allocator.alloc(usize, scores.len);
    errdefer allocator.free(targets);
    var eligible_rows: usize = 0;
    var current_rows: usize = 0;
    var before: f64 = 0;
    var prior_end: usize = 0;
    for (scores, levels, targets, 0..) |row, *level, *rows, i| {
        if (row.rows == 0 or (i != 0 and row.start < prior_end)) return error.InvalidAttentionShape;
        const expected = [4]usize{ try target(row.rows, 4), try target(row.rows, 3), try target(row.rows, 2), row.rows };
        if (!std.mem.eql(usize, &expected, &row.target_rows)) return error.InvalidAttentionShape;
        for (row.mass_harm, row.output_harm) |m, o| if (!std.math.isFinite(m) or m < 0 or !std.math.isFinite(o) or o < 0) return error.NonFiniteAttention;
        level.* = .divisor4;
        rows.* = expected[0];
        eligible_rows = try add(eligible_rows, row.rows);
        current_rows = try add(current_rows, rows.*);
        before += try harm(row, .divisor4, weights);
        prior_end = try add(row.start, row.rows);
    }
    const cap = try target(eligible_rows, branch_divisor);
    if (current_rows > cap) return error.InvalidAttentionShape;
    while (true) {
        var best: ?usize = null;
        var best_level: Level = .divisor4;
        var best_gain: f64 = 0;
        var best_extra: usize = 1;
        for (scores, levels, targets, 0..) |row, level, rows, i| {
            if (level == .identity) continue;
            var next_raw: usize = @as(usize, @backingInt(level)) + 1;
            while (next_raw <= @backingInt(Level.identity)) : (next_raw += 1) {
                const next: Level = @fromBackingInt(@intCast(next_raw));
                const next_rows = row.target_rows[@backingInt(next)];
                const extra = next_rows - rows;
                if (extra == 0 or extra > cap - current_rows) continue;
                const gain = (try harm(row, level, weights)) - (try harm(row, next, weights));
                if (!(gain > 0)) continue;
                const lhs = gain * @as(f64, @floatFromInt(best_extra));
                const rhs = best_gain * @as(f64, @floatFromInt(extra));
                if (best == null or lhs > rhs or (lhs == rhs and (row.start < scores[best.?].start or (row.start == scores[best.?].start and @backingInt(next) < @backingInt(best_level))))) {
                    best = i;
                    best_level = next;
                    best_gain = gain;
                    best_extra = extra;
                }
            }
        }
        const i = best orelse break;
        levels[i] = best_level;
        targets[i] = scores[i].target_rows[@backingInt(levels[i])];
        current_rows += best_extra;
    }
    var after: f64 = 0;
    var preserved: usize = 0;
    for (scores, levels) |row, level| {
        after += try harm(row, level, weights);
        preserved += @intFromBool(level == .identity);
    }
    return .{ .allocator = allocator, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = current_rows, .preserved_ranges = preserved, .fitted_ranges = scores.len - preserved, .train_harm_before = before, .train_harm_after = after, .digest = digestGeometry(scores, levels, targets, branch_divisor, cap, current_rows, weights) };
}

fn jointObjective(original_y: []const f64, mass: []const f64, numerator: []const f64, q: usize, dim: usize, weights: Weights) !f64 {
    const qd = std.math.mul(usize, q, dim) catch return error.InvalidAttentionShape;
    if (q == 0 or dim == 0 or mass.len != q or original_y.len != qd or numerator.len != original_y.len) return error.InvalidFitConfig;
    var maximum: f64 = 0;
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    for (0..q) |qi| {
        const denominator = 1 + mass[qi];
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(mass[qi]));
        for (0..dim) |di| {
            const at = qi * dim + di;
            const predicted = (original_y[at] + numerator[at]) / denominator;
            if (!std.math.isFinite(predicted) or !std.math.isFinite(original_y[at])) return error.NonFiniteAttention;
            const delta = predicted - original_y[at];
            squared_error += delta * delta;
            squared_reference += original_y[at] * original_y[at];
        }
    }
    const output = @sqrt(squared_error / @max(squared_reference, 1e-24));
    const result = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(result) or result < 0) return error.NonFiniteAttention;
    return result;
}

fn addCandidate(mass: []f64, numerator: []f64, candidate: CandidateDelta, sign: f64) !void {
    if ((candidate.mass.len == 0) != (candidate.numerator.len == 0)) return error.InvalidFitConfig;
    if (candidate.mass.len == 0) return;
    if (candidate.mass.len != mass.len or candidate.numerator.len != numerator.len) return error.InvalidFitConfig;
    for (mass, candidate.mass) |*out, value| out.* += sign * value;
    for (numerator, candidate.numerator) |*out, value| out.* += sign * value;
}

/// Training-only signed joint geometry. Candidate deltas are relative to the
/// original full head in original-partition units. Identity uses empty slices.
/// No heldout queries or metrics enter this interface.
pub fn chooseSignedJoint(allocator: std.mem.Allocator, scores: []const JointRangeScores, original_y: []const f64, q: usize, dim: usize, branch_divisor: usize, weights: Weights, control: solver.Control) !Geometry {
    try control.check();
    const qd = std.math.mul(usize, q, dim) catch return error.InvalidAttentionShape;
    if (scores.len == 0 or original_y.len != qd or (branch_divisor != 4 and branch_divisor != 3 and branch_divisor != 2) or !std.math.isFinite(weights.mass) or !std.math.isFinite(weights.output) or weights.mass < 0 or weights.output < 0 or weights.mass + weights.output == 0) return error.InvalidFitConfig;
    const levels = try allocator.alloc(Level, scores.len);
    errdefer allocator.free(levels);
    const targets = try allocator.alloc(usize, scores.len);
    errdefer allocator.free(targets);
    const aggregate_mass = try allocator.alloc(f64, q);
    defer allocator.free(aggregate_mass);
    const aggregate_numerator = try allocator.alloc(f64, original_y.len);
    defer allocator.free(aggregate_numerator);
    const trial_mass = try allocator.alloc(f64, q);
    defer allocator.free(trial_mass);
    const trial_numerator = try allocator.alloc(f64, original_y.len);
    defer allocator.free(trial_numerator);
    @memset(aggregate_mass, 0);
    @memset(aggregate_numerator, 0);
    var eligible: usize = 0;
    var current_rows: usize = 0;
    var prior_end: usize = 0;
    for (scores, levels, targets, 0..) |row, *level, *rows, i| {
        if (row.rows == 0 or (i != 0 and row.start < prior_end)) return error.InvalidAttentionShape;
        for (row.target_rows, 0..) |count, li| {
            if (count == 0 or count > row.rows or (li != 0 and count < row.target_rows[li - 1])) return error.InvalidFitConfig;
            const delta = row.deltas[li];
            if (li == 3) {
                if (delta.mass.len != 0 or delta.numerator.len != 0 or count != row.rows) return error.InvalidFitConfig;
            } else if (delta.mass.len != q or delta.numerator.len != original_y.len) return error.InvalidFitConfig;
        }
        level.* = .divisor4;
        rows.* = row.target_rows[0];
        eligible = try add(eligible, row.rows);
        current_rows = try add(current_rows, rows.*);
        prior_end = try add(row.start, row.rows);
        try addCandidate(aggregate_mass, aggregate_numerator, row.deltas[0], 1);
    }
    const cap = try target(eligible, branch_divisor);
    const before = try jointObjective(original_y, aggregate_mass, aggregate_numerator, q, dim, weights);
    var current_objective = before;
    while (true) {
        try control.check();
        var best: ?usize = null;
        var best_level: Level = .divisor4;
        var best_objective: f64 = 0;
        var best_gain: f64 = 0;
        var best_cost: usize = 1;
        for (scores, levels, targets, 0..) |row, level, rows, i| {
            if (level == .identity) continue;
            for (@backingInt(level) + 1..4) |next_index| {
                try control.check();
                const next: Level = @fromBackingInt(@intCast(next_index));
                const next_rows = row.target_rows[next_index];
                const cost = next_rows - rows;
                const proposed_rows = try add(current_rows, cost);
                if (cost == 0 or proposed_rows > cap) continue;
                @memcpy(trial_mass, aggregate_mass);
                @memcpy(trial_numerator, aggregate_numerator);
                try addCandidate(trial_mass, trial_numerator, row.deltas[@backingInt(level)], -1);
                try addCandidate(trial_mass, trial_numerator, row.deltas[next_index], 1);
                const objective = try jointObjective(original_y, trial_mass, trial_numerator, q, dim, weights);
                const gain = current_objective - objective;
                if (gain <= 0) continue;
                if (best == null or gain * @as(f64, @floatFromInt(best_cost)) > best_gain * @as(f64, @floatFromInt(cost)) or
                    (gain * @as(f64, @floatFromInt(best_cost)) == best_gain * @as(f64, @floatFromInt(cost)) and (row.start < scores[best.?].start or (row.start == scores[best.?].start and @backingInt(next) < @backingInt(best_level)))))
                {
                    best = i;
                    best_level = next;
                    best_objective = objective;
                    best_gain = gain;
                    best_cost = cost;
                }
            }
        }
        const i = best orelse break;
        try control.check();
        try addCandidate(aggregate_mass, aggregate_numerator, scores[i].deltas[@backingInt(levels[i])], -1);
        try addCandidate(aggregate_mass, aggregate_numerator, scores[i].deltas[@backingInt(best_level)], 1);
        current_rows += best_cost;
        levels[i] = best_level;
        targets[i] = scores[i].target_rows[@backingInt(best_level)];
        current_objective = best_objective;
    }
    var preserved: usize = 0;
    for (levels) |level| preserved += @intFromBool(level == .identity);
    var digest: [32]u8 = undefined;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(signed_joint_policy);
    hasher.update(std.mem.sliceAsBytes(levels));
    hasher.update(std.mem.sliceAsBytes(targets));
    for (scores) |row| {
        hasher.update(std.mem.asBytes(&row.start));
        hasher.update(std.mem.asBytes(&row.rows));
    }
    hasher.update(std.mem.asBytes(&q));
    hasher.update(std.mem.asBytes(&dim));
    hasher.update(std.mem.asBytes(&branch_divisor));
    hasher.update(std.mem.asBytes(&weights.mass));
    hasher.update(std.mem.asBytes(&weights.output));
    hasher.update(std.mem.asBytes(&cap));
    hasher.update(std.mem.asBytes(&current_rows));
    for (aggregate_mass) |value| {
        var bits: u64 = @bitCast(value);
        hasher.update(std.mem.asBytes(&bits));
    }
    for (aggregate_numerator) |value| {
        var bits: u64 = @bitCast(value);
        hasher.update(std.mem.asBytes(&bits));
    }
    try control.check();
    hasher.final(&digest);
    return .{ .allocator = allocator, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = current_rows, .preserved_ranges = preserved, .fitted_ranges = scores.len - preserved, .train_harm_before = before, .train_harm_after = current_objective, .digest = digest };
}

test "signed joint geometry detects reinforcement and preserves cancellation" {
    const zero = [_]f64{0};
    const positive = [_]f64{0.04};
    const negative = [_]f64{-0.04};
    const numerator = [_]f64{0};
    const identity: CandidateDelta = .{ .mass = &.{}, .numerator = &.{} };
    const positive_levels = [4]CandidateDelta{ .{ .mass = &positive, .numerator = &numerator }, .{ .mass = &zero, .numerator = &numerator }, .{ .mass = &zero, .numerator = &numerator }, identity };
    const negative_levels = [4]CandidateDelta{ .{ .mass = &negative, .numerator = &numerator }, .{ .mass = &zero, .numerator = &numerator }, .{ .mass = &zero, .numerator = &numerator }, identity };
    const reinforcement = [_]JointRangeScores{
        .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = positive_levels },
        .{ .start = 8, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = positive_levels },
    };
    var upgraded = try chooseSignedJoint(std.testing.allocator, &reinforcement, &.{1}, 1, 1, 2, .{ .mass = 1, .output = 0 }, .{});
    defer upgraded.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, upgraded.targets);
    try std.testing.expect(upgraded.train_harm_after < upgraded.train_harm_before);

    const cancellation = [_]JointRangeScores{
        .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = positive_levels },
        .{ .start = 8, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = negative_levels },
    };
    var preserved = try chooseSignedJoint(std.testing.allocator, &cancellation, &.{1}, 1, 1, 2, .{ .mass = 1, .output = 0 }, .{});
    defer preserved.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, preserved.targets);
    try std.testing.expectEqual(@as(f64, 0), preserved.train_harm_after);
}

test "signed joint geometry scores output through the aggregate denominator" {
    const zero = [_]f64{0};
    const output_delta = [_]f64{0.2};
    const clean = [_]f64{0};
    const identity: CandidateDelta = .{ .mass = &.{}, .numerator = &.{} };
    const levels = [4]CandidateDelta{ .{ .mass = &zero, .numerator = &output_delta }, .{ .mass = &zero, .numerator = &clean }, .{ .mass = &zero, .numerator = &clean }, identity };
    const scores = [_]JointRangeScores{
        .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = levels },
        .{ .start = 8, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = levels },
    };
    var geometry = try chooseSignedJoint(std.testing.allocator, &scores, &.{1}, 1, 1, 2, .{ .mass = 0, .output = 1 }, .{});
    defer geometry.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, geometry.targets);
    try std.testing.expect(geometry.train_harm_before > 0.39);
    try std.testing.expectEqual(@as(f64, 0), geometry.train_harm_after);
}

test "signed joint geometry jumps over a harmful intermediate level" {
    const d4 = [_]f64{0.1};
    const d3 = [_]f64{0.2};
    const clean = [_]f64{0};
    const numerator = [_]f64{0};
    const scores = [_]JointRangeScores{.{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = .{
        .{ .mass = &d4, .numerator = &numerator },
        .{ .mass = &d3, .numerator = &numerator },
        .{ .mass = &clean, .numerator = &numerator },
        .{ .mass = &.{}, .numerator = &.{} },
    } }};
    var geometry = try chooseSignedJoint(std.testing.allocator, &scores, &.{1}, 1, 1, 2, .{ .mass = 1, .output = 0 }, .{});
    defer geometry.deinit();
    try std.testing.expectEqualSlices(usize, &.{4}, geometry.targets);
    try std.testing.expectEqual(@as(f64, 0), geometry.train_harm_after);
}

const CancelAfter = struct {
    calls: usize = 0,
    at: usize,
    fn check(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        if (self.calls == self.at) return error.Cancelled;
    }
    fn control(self: *@This()) solver.Control {
        return .{ .context = self, .check_fn = check };
    }
};

test "signed joint geometry cancellation publishes no geometry" {
    const delta = [_]f64{0.1};
    const clean = [_]f64{0};
    const scores = [_]JointRangeScores{.{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .deltas = .{
        .{ .mass = &delta, .numerator = &clean },
        .{ .mass = &clean, .numerator = &clean },
        .{ .mass = &clean, .numerator = &clean },
        .{ .mass = &.{}, .numerator = &.{} },
    } }};
    var cancel: CancelAfter = .{ .at = 3 };
    try std.testing.expectError(error.Cancelled, chooseSignedJoint(std.testing.allocator, &scores, &.{1}, 1, 1, 2, .{}, cancel.control()));
    try std.testing.expect(cancel.calls >= cancel.at);
}

test "adaptive geometry greedily upgrades training harm with stable source ties" {
    const rows = [_]RangeScores{
        .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .mass_harm = .{ 8, 6, 2, 0 }, .output_harm = .{ 0, 0, 0, 0 } },
        .{ .start = 8, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .mass_harm = .{ 8, 6, 2, 0 }, .output_harm = .{ 0, 0, 0, 0 } },
    };
    var g = try choose(std.testing.allocator, &rows, 2, .{});
    defer g.deinit();
    try std.testing.expectEqual(@as(usize, 8), g.middle_rows);
    try std.testing.expectEqualSlices(Level, &.{ .divisor2, .divisor2 }, g.levels);
    try std.testing.expect(g.train_harm_after < g.train_harm_before);
}
test "adaptive geometry can choose exact identity across sorted gaps and rejects overlap" {
    const row = RangeScores{ .start = 4, .rows = 4, .target_rows = .{ 1, 2, 2, 4 }, .mass_harm = .{ 10, 9, 8, 0 }, .output_harm = .{ 0, 0, 0, 0 } };
    const donor = RangeScores{ .start = 8, .rows = 20, .target_rows = .{ 5, 7, 10, 20 }, .mass_harm = .{ 0, 0, 0, 0 }, .output_harm = .{ 0, 0, 0, 0 } };
    var g = try choose(std.testing.allocator, &.{ row, donor }, 2, .{});
    defer g.deinit();
    try std.testing.expectEqual(Level.identity, g.levels[0]);
    var bad = [_]RangeScores{ row, row };
    bad[1].start = 7;
    try std.testing.expectError(error.InvalidAttentionShape, choose(std.testing.allocator, &bad, 2, .{}));

    bad[1].start = 12;
    var gapped = try choose(std.testing.allocator, &bad, 2, .{});
    defer gapped.deinit();
    try std.testing.expectEqual(@as(usize, 2), gapped.targets.len);
}
test "rounded candidate harm models full-head renormalization and identity is zero" {
    const exact = try measureRoundedCandidateHarm(&.{ 2, 4 }, &.{0.25}, &.{ 0.5, 1 }, &.{0.25}, &.{ 0.5, 1 }, 1, 2, 1e-12);
    try std.testing.expectEqual(@as(f64, 0), exact.max_relative_mass);
    try std.testing.expectEqual(@as(f64, 0), exact.relative_output_rmse);
    const changed = try measureRoundedCandidateHarm(&.{ 2, 4 }, &.{0.25}, &.{ 0.5, 1 }, &.{0.5}, &.{ 1, 3 }, 1, 2, 1e-12);
    try std.testing.expect(changed.max_relative_mass > 0);
    try std.testing.expect(changed.relative_output_rmse > 0);
    try std.testing.expectError(error.NonFiniteAttention, measureRoundedCandidateHarm(&.{1}, &.{2}, &.{1}, &.{0}, &.{1}, 1, 1, 1e-12));
}
test "rounded candidate harm measures partition loss even when output is unchanged" {
    const measured = try measureRoundedCandidateHarm(&.{1}, &.{1}, &.{1}, &.{0.5}, &.{0.5}, 1, 1, 1e-12);
    try std.testing.expectEqual(@as(f64, 0.5), measured.max_relative_mass);
    try std.testing.expectEqual(@as(f64, 0), measured.relative_output_rmse);
}

fn chooseAllocationFailure(allocator: std.mem.Allocator) !void {
    const rows = [_]RangeScores{.{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 }, .mass_harm = .{ 8, 6, 2, 0 }, .output_harm = .{ 1, 1, 1, 0 } }};
    var geometry = try choose(allocator, &rows, 2, .{});
    geometry.deinit();
}
test "adaptive geometry unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, chooseAllocationFailure, .{});
}
