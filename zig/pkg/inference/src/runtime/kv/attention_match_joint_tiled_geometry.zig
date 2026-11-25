// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Private, default-off TRAIN-only signed geometry which streams candidate
//! deltas instead of retaining Q*K mass and Q*triangle(K) Gram matrices.
const std = @import("std");
const solver = @import("attention_match.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");

pub const policy = "greedy_positive_joint_train_objective_per_row_tiled_v1";
pub const budget_percent_policy = "greedy_positive_joint_train_objective_per_row_tiled_divisor2_budget_percent_v1";

pub const Range = struct {
    start: usize,
    rows: usize,
    target_rows: [4]usize,
};

pub const DeltaAccessor = struct {
    /// Fills a query tile of the signed candidate delta in original-partition
    /// units. Identity (level 3) is implicit and is never requested.
    /// Implementations must be synchronous and must not retain output slices.
    context: ?*anyopaque,
    fill_fn: *const fn (
        context: ?*anyopaque,
        range_index: usize,
        level_index: usize,
        first_query: usize,
        query_count: usize,
        mass: []f64,
        numerator: []f64,
        control: solver.Control,
    ) anyerror!void,
};

pub const Plan = struct {
    owned_peak_bytes: usize,
    accessor_workspace_bytes: usize,
    combined_peak_bytes: usize,
    aggregate_bytes: usize,
    trial_bytes: usize,
    tile_bytes: usize,
    result_bytes: usize,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

fn vectorBytes(q: usize, dim: usize) !usize {
    return mul(try add(q, try mul(q, dim)), @sizeOf(f64));
}

/// Peak owned by this chooser. Caller-owned source/page/scored state and any
/// allocations performed inside the accessor are deliberately not included;
/// their conservative simultaneous peak must be supplied as external_live.
pub fn workspaceBytes(q: usize, dim: usize, range_count: usize, tile_rows: usize, accessor_workspace_bytes: usize) !Plan {
    if (q == 0 or dim == 0 or range_count == 0 or tile_rows == 0) return error.InvalidFitConfig;
    const aggregate = try vectorBytes(q, dim);
    const trial = aggregate;
    const tile = try vectorBytes(@min(q, tile_rows), dim);
    const result = try mul(range_count, @sizeOf(adaptive.Level) + @sizeOf(usize));
    const owned = try add(result, try add(aggregate, try add(trial, tile)));
    return .{
        .owned_peak_bytes = owned,
        .accessor_workspace_bytes = accessor_workspace_bytes,
        .combined_peak_bytes = try add(owned, accessor_workspace_bytes),
        .aggregate_bytes = aggregate,
        .trial_bytes = trial,
        .tile_bytes = tile,
        .result_bytes = result,
    };
}

fn validateWeights(weights: adaptive.Weights) !void {
    if (!std.math.isFinite(weights.mass) or !std.math.isFinite(weights.output) or
        weights.mass < 0 or weights.output < 0 or
        (weights.mass == 0 and weights.output == 0)) return error.InvalidFitConfig;
}

fn validateRanges(ranges: []const Range) !void {
    if (ranges.len == 0) return error.InvalidFitConfig;
    var prior_end: usize = 0;
    for (ranges, 0..) |range, ri| {
        if (range.rows == 0 or (ri != 0 and range.start < prior_end)) return error.InvalidAttentionShape;
        for (range.target_rows, 0..) |count, level| {
            if (count == 0 or count > range.rows or
                (level != 0 and count < range.target_rows[level - 1])) return error.InvalidFitConfig;
        }
        if (range.target_rows[3] != range.rows) return error.InvalidFitConfig;
        prior_end = try add(range.start, range.rows);
    }
}

fn objective(
    original_y: []const f64,
    mass: []const f64,
    numerator: []const f64,
    q: usize,
    dim: usize,
    weights: adaptive.Weights,
) !f64 {
    const qd = try mul(q, dim);
    if (original_y.len != qd or mass.len != q or numerator.len != qd) return error.InvalidAttentionShape;
    var maximum: f64 = 0;
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    for (0..q) |qi| {
        const denominator = 1 + mass[qi];
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(mass[qi]));
        for (0..dim) |di| {
            const at = qi * dim + di;
            if (!std.math.isFinite(original_y[at]) or !std.math.isFinite(numerator[at])) return error.NonFiniteAttention;
            const predicted = (original_y[at] + numerator[at]) / denominator;
            if (!std.math.isFinite(predicted)) return error.NonFiniteAttention;
            const delta = predicted - original_y[at];
            squared_error += delta * delta;
            squared_reference += original_y[at] * original_y[at];
        }
    }
    const output = @sqrt(squared_error / @max(squared_reference, 1e-24));
    const value = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    return value;
}

fn applyCandidate(
    destination_mass: []f64,
    destination_numerator: []f64,
    range_index: usize,
    level_index: usize,
    sign: f64,
    q: usize,
    dim: usize,
    tile_rows: usize,
    tile_mass: []f64,
    tile_numerator: []f64,
    accessor: DeltaAccessor,
    control: solver.Control,
) !void {
    if (level_index == 3) return;
    var first: usize = 0;
    while (first < q) {
        try control.check();
        const take = @min(tile_rows, q - first);
        const tile_values = try mul(take, dim);
        @memset(tile_mass[0..take], 0);
        @memset(tile_numerator[0..tile_values], 0);
        try accessor.fill_fn(
            accessor.context,
            range_index,
            level_index,
            first,
            take,
            tile_mass[0..take],
            tile_numerator[0..tile_values],
            control,
        );
        for (0..take) |local| {
            const value = tile_mass[local];
            if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            destination_mass[first + local] += sign * value;
            if (!std.math.isFinite(destination_mass[first + local])) return error.NonFiniteAttention;
            for (0..dim) |di| {
                const at = (first + local) * dim + di;
                const delta = tile_numerator[local * dim + di];
                if (!std.math.isFinite(delta)) return error.NonFiniteAttention;
                destination_numerator[at] += sign * delta;
                if (!std.math.isFinite(destination_numerator[at])) return error.NonFiniteAttention;
            }
        }
        first += take;
    }
}

fn digestGeometry(
    ranges: []const Range,
    levels: []const adaptive.Level,
    targets: []const usize,
    q: usize,
    dim: usize,
    divisor: usize,
    cap: usize,
    actual: usize,
    weights: adaptive.Weights,
    aggregate_mass: []const f64,
    aggregate_numerator: []const f64,
    budget_percent: u8,
) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(if (budget_percent == 100) policy else budget_percent_policy);
    if (budget_percent != 100) h.update(&.{budget_percent});
    var bytes: [8]u8 = undefined;
    for (levels) |level| {
        std.mem.writeInt(u64, &bytes, @backingInt(level), .little);
        h.update(&bytes);
    }
    for (targets) |value| {
        std.mem.writeInt(u64, &bytes, @intCast(value), .little);
        h.update(&bytes);
    }
    for (ranges) |range| {
        for ([_]usize{ range.start, range.rows }) |value| {
            std.mem.writeInt(u64, &bytes, @intCast(value), .little);
            h.update(&bytes);
        }
    }
    for ([_]usize{ q, dim, divisor, cap, actual }) |value| {
        std.mem.writeInt(u64, &bytes, @intCast(value), .little);
        h.update(&bytes);
    }
    for ([_]f64{ weights.mass, weights.output }) |value| {
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        h.update(&bytes);
    }
    for (aggregate_mass) |value| {
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        h.update(&bytes);
    }
    for (aggregate_numerator) |value| {
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        h.update(&bytes);
    }
    return h.finalResult();
}

/// Exact dense-algebra chooser with bounded owned memory. This function never
/// receives heldout data. It is intentionally a separate, default-off seam.
pub fn choose(
    allocator: std.mem.Allocator,
    ranges: []const Range,
    original_y: []const f64,
    q: usize,
    dim: usize,
    branch_divisor: usize,
    weights: adaptive.Weights,
    tile_rows: usize,
    accessor_workspace_bytes: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    accessor: DeltaAccessor,
    control: solver.Control,
) !adaptive.Geometry {
    return chooseWithBudgetPercent(allocator, ranges, original_y, q, dim, branch_divisor, weights, 100, tile_rows, accessor_workspace_bytes, external_live_bytes, combined_cap, accessor, control);
}

pub fn chooseWithBudgetPercent(
    allocator: std.mem.Allocator,
    ranges: []const Range,
    original_y: []const f64,
    q: usize,
    dim: usize,
    branch_divisor: usize,
    weights: adaptive.Weights,
    budget_percent: u8,
    tile_rows: usize,
    accessor_workspace_bytes: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    accessor: DeltaAccessor,
    control: solver.Control,
) !adaptive.Geometry {
    try control.check();
    const qd = try mul(q, dim);
    if (original_y.len != qd or (branch_divisor != 4 and branch_divisor != 3 and branch_divisor != 2) or
        budget_percent < 50 or budget_percent > 100 or (budget_percent != 100 and branch_divisor != 2) or
        external_live_bytes > combined_cap) return error.InvalidFitConfig;
    try validateWeights(weights);
    try validateRanges(ranges);
    const plan = try workspaceBytes(q, dim, ranges.len, tile_rows, accessor_workspace_bytes);
    if (try add(external_live_bytes, plan.combined_peak_bytes) > combined_cap) return error.CompactionWorkspaceLimit;

    const levels = try allocator.alloc(adaptive.Level, ranges.len);
    errdefer allocator.free(levels);
    const targets = try allocator.alloc(usize, ranges.len);
    errdefer allocator.free(targets);
    const aggregate_mass = try allocator.alloc(f64, q);
    defer allocator.free(aggregate_mass);
    const aggregate_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(aggregate_numerator);
    const trial_mass = try allocator.alloc(f64, q);
    defer allocator.free(trial_mass);
    const trial_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(trial_numerator);
    const take_max = @min(q, tile_rows);
    const tile_mass = try allocator.alloc(f64, take_max);
    defer allocator.free(tile_mass);
    const tile_numerator = try allocator.alloc(f64, try mul(take_max, dim));
    defer allocator.free(tile_numerator);
    @memset(aggregate_mass, 0);
    @memset(aggregate_numerator, 0);

    var eligible: usize = 0;
    var current_rows: usize = 0;
    for (ranges, levels, targets, 0..) |range, *level, *target_rows, ri| {
        try control.check();
        level.* = .divisor4;
        target_rows.* = range.target_rows[0];
        eligible = try add(eligible, range.rows);
        current_rows = try add(current_rows, target_rows.*);
        try applyCandidate(aggregate_mass, aggregate_numerator, ri, 0, 1, q, dim, tile_rows, tile_mass, tile_numerator, accessor, control);
    }
    const nominal_cap = std.math.divCeil(usize, eligible, branch_divisor) catch return error.InvalidAttentionShape;
    const cap = if (budget_percent == 100) nominal_cap else try mul(nominal_cap, budget_percent) / 100;
    if (current_rows > cap) return error.GeometryBudgetInfeasible;
    const before = try objective(original_y, aggregate_mass, aggregate_numerator, q, dim, weights);
    var current_objective = before;

    while (true) {
        try control.check();
        var best_range: ?usize = null;
        var best_level: adaptive.Level = .divisor4;
        var best_objective: f64 = 0;
        var best_gain: f64 = 0;
        var best_cost: usize = 1;
        for (ranges, levels, targets, 0..) |range, level, current_target, ri| {
            if (level == .identity) continue;
            for (@as(usize, @backingInt(level)) + 1..4) |next_index| {
                try control.check();
                const cost = range.target_rows[next_index] - current_target;
                const proposed_rows = try add(current_rows, cost);
                if (cost == 0 or proposed_rows > cap) continue;
                @memcpy(trial_mass, aggregate_mass);
                @memcpy(trial_numerator, aggregate_numerator);
                try applyCandidate(trial_mass, trial_numerator, ri, @backingInt(level), -1, q, dim, tile_rows, tile_mass, tile_numerator, accessor, control);
                try applyCandidate(trial_mass, trial_numerator, ri, next_index, 1, q, dim, tile_rows, tile_mass, tile_numerator, accessor, control);
                const value = try objective(original_y, trial_mass, trial_numerator, q, dim, weights);
                const gain = current_objective - value;
                if (gain <= 0) continue;
                if (best_range == null or
                    gain * @as(f64, @floatFromInt(best_cost)) > best_gain * @as(f64, @floatFromInt(cost)) or
                    (gain * @as(f64, @floatFromInt(best_cost)) == best_gain * @as(f64, @floatFromInt(cost)) and
                        (range.start < ranges[best_range.?].start or
                            (range.start == ranges[best_range.?].start and next_index < @backingInt(best_level)))))
                {
                    best_range = ri;
                    best_level = @fromBackingInt(@intCast(next_index));
                    best_objective = value;
                    best_gain = gain;
                    best_cost = cost;
                }
            }
        }
        const ri = best_range orelse break;
        try control.check();
        try applyCandidate(aggregate_mass, aggregate_numerator, ri, @backingInt(levels[ri]), -1, q, dim, tile_rows, tile_mass, tile_numerator, accessor, control);
        try applyCandidate(aggregate_mass, aggregate_numerator, ri, @backingInt(best_level), 1, q, dim, tile_rows, tile_mass, tile_numerator, accessor, control);
        levels[ri] = best_level;
        targets[ri] = ranges[ri].target_rows[@backingInt(best_level)];
        current_rows += best_cost;
        current_objective = best_objective;
    }
    try control.check();
    var preserved: usize = 0;
    for (levels) |level| preserved += @intFromBool(level == .identity);
    return .{
        .allocator = allocator,
        .levels = levels,
        .targets = targets,
        .middle_cap_rows = cap,
        .middle_rows = current_rows,
        .preserved_ranges = preserved,
        .fitted_ranges = ranges.len - preserved,
        .train_harm_before = before,
        .train_harm_after = current_objective,
        .digest = digestGeometry(ranges, levels, targets, q, dim, branch_divisor, cap, current_rows, weights, aggregate_mass, aggregate_numerator, budget_percent),
    };
}

const Fixture = struct {
    q: usize,
    dim: usize,
    range_count: usize,
    mass: []const f64,
    numerator: []const f64,
    calls: usize = 0,

    fn fill(raw: ?*anyopaque, ri: usize, li: usize, first: usize, count: usize, mass: []f64, numerator: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        if (ri >= self.range_count or li >= 3 or first + count > self.q or mass.len != count or numerator.len != count * self.dim) return error.InvalidAttentionShape;
        const candidate = ri * 3 + li;
        for (0..count) |local| {
            try control.check();
            const qi = first + local;
            mass[local] = self.mass[candidate * self.q + qi];
            @memcpy(numerator[local * self.dim ..][0..self.dim], self.numerator[(candidate * self.q + qi) * self.dim ..][0..self.dim]);
        }
    }

    fn dense(self: *@This(), allocator: std.mem.Allocator, ranges: []const Range, original_y: []const f64, divisor: usize, weights: adaptive.Weights) !adaptive.Geometry {
        const scores = try allocator.alloc(adaptive.JointRangeScores, ranges.len);
        defer allocator.free(scores);
        for (ranges, scores, 0..) |range, *score, ri| {
            score.* = .{
                .start = range.start,
                .rows = range.rows,
                .target_rows = range.target_rows,
                .deltas = undefined,
            };
            for (0..3) |li| {
                const candidate = ri * 3 + li;
                score.deltas[li] = .{
                    .mass = self.mass[candidate * self.q ..][0..self.q],
                    .numerator = self.numerator[candidate * self.q * self.dim ..][0 .. self.q * self.dim],
                };
            }
            score.deltas[3] = .{ .mass = &.{}, .numerator = &.{} };
        }
        return adaptive.chooseSignedJoint(allocator, scores, original_y, self.q, self.dim, divisor, weights, .{});
    }
};

const fixture_ranges = [_]Range{
    .{ .start = 4, .rows = 8, .target_rows = .{ 2, 3, 4, 8 } },
    .{ .start = 20, .rows = 8, .target_rows = .{ 2, 3, 4, 8 } },
};
const fixture_y = [_]f64{ 0.25, -0.5, 0.75, 0.125 };
const fixture_mass = [_]f64{
    0.08,  0.04,  0.02,  0.01,  0.0, 0.0,
    -0.06, -0.03, -0.02, -0.01, 0.0, 0.0,
};
const fixture_numerator = [_]f64{
    0.02,   -0.01, 0.01,  0.03, 0.01,   -0.005, 0.005,  0.015, 0, 0, 0, 0,
    -0.015, 0.02,  -0.01, 0.01, -0.005, 0.01,   -0.005, 0.005, 0, 0, 0, 0,
};

fn makeFixture() Fixture {
    return .{ .q = 2, .dim = 2, .range_count = 2, .mass = &fixture_mass, .numerator = &fixture_numerator };
}

test "tiled signed chooser is exactly equivalent to retained dense algebra" {
    for ([_]usize{ 1, 2 }) |tile_rows| for ([_]usize{ 4, 3, 2 }) |divisor| {
        var fixture = makeFixture();
        var dense = try fixture.dense(std.testing.allocator, &fixture_ranges, &fixture_y, divisor, .{});
        defer dense.deinit();
        var tiled = try choose(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, divisor, .{}, tile_rows, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{});
        defer tiled.deinit();
        try std.testing.expectEqualSlices(adaptive.Level, dense.levels, tiled.levels);
        try std.testing.expectEqualSlices(usize, dense.targets, tiled.targets);
        try std.testing.expectEqual(dense.middle_cap_rows, tiled.middle_cap_rows);
        try std.testing.expectEqual(dense.middle_rows, tiled.middle_rows);
        try std.testing.expectEqual(dense.preserved_ranges, tiled.preserved_ranges);
        try std.testing.expectEqual(dense.fitted_ranges, tiled.fitted_ranges);
        try std.testing.expectEqual(dense.train_harm_before, tiled.train_harm_before);
        try std.testing.expectEqual(dense.train_harm_after, tiled.train_harm_after);
    };
}

test "tiled divisor2 budget percent is explicit and legacy 100 is byte-stable" {
    var fixture = makeFixture();
    var legacy = try choose(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{});
    defer legacy.deinit();
    var explicit = try chooseWithBudgetPercent(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 100, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{});
    defer explicit.deinit();
    try std.testing.expectEqualSlices(adaptive.Level, legacy.levels, explicit.levels);
    try std.testing.expectEqualSlices(usize, legacy.targets, explicit.targets);
    try std.testing.expectEqual(legacy.digest, explicit.digest);
    for ([_]u8{ 91, 95 }) |percent| {
        var reduced = try chooseWithBudgetPercent(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, percent, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{});
        defer reduced.deinit();
        try std.testing.expectEqual(legacy.middle_cap_rows * percent / 100, reduced.middle_cap_rows);
        try std.testing.expect(reduced.middle_rows <= reduced.middle_cap_rows);
        try std.testing.expect(!std.mem.eql(u8, &legacy.digest, &reduced.digest));
    }
    try std.testing.expectError(error.InvalidFitConfig, chooseWithBudgetPercent(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 3, .{}, 91, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}));
    try std.testing.expectError(error.InvalidFitConfig, chooseWithBudgetPercent(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 49, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}));
}

test "tiled critical minimum reports an unavailable divisor4 row budget" {
    var fixture = makeFixture();
    var ranges = fixture_ranges;
    ranges[0].target_rows[0] = 3;
    ranges[1].target_rows[0] = 3;
    try std.testing.expectError(error.GeometryBudgetInfeasible, chooseWithBudgetPercent(std.testing.allocator, &ranges, &fixture_y, 2, 2, 4, .{}, 100, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}));
}

test "tiled signed chooser cap rejects before allocation or accessor" {
    var fixture = makeFixture();
    const accessor_workspace: usize = 23;
    const plan = try workspaceBytes(2, 2, 2, 1, accessor_workspace);
    try std.testing.expectEqual(accessor_workspace, plan.accessor_workspace_bytes);
    try std.testing.expectError(error.CompactionWorkspaceLimit, choose(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 1, accessor_workspace, 17, 17 + plan.combined_peak_bytes - 1, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}));
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
}

const Cancel = struct {
    checks: usize = 0,
    cancel_at: usize,
    fn check(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.checks += 1;
        if (self.checks == self.cancel_at) return error.Cancelled;
    }
    fn control(self: *@This()) solver.Control {
        return .{ .context = self, .check_fn = check };
    }
};

test "tiled signed chooser cancellation unwinds before publication" {
    var fixture = makeFixture();
    var cancel: Cancel = .{ .cancel_at = 7 };
    try std.testing.expectError(error.Cancelled, choose(std.testing.allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, cancel.control()));
}

fn allocationFixture(allocator: std.mem.Allocator) !void {
    var fixture = makeFixture();
    var geometry = try choose(allocator, &fixture_ranges, &fixture_y, 2, 2, 2, .{}, 1, 0, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{});
    geometry.deinit();
}

test "tiled signed chooser unwinds every owned allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFixture, .{});
}

test "tiled signed chooser preserves stable source-level ties and nonmonotone jumps" {
    const Local = struct {
        fn fill(_: ?*anyopaque, ri: usize, li: usize, _: usize, _: usize, mass: []f64, numerator: []f64, _: solver.Control) !void {
            mass[0] = if (li == 0) 0.2 else if (li == 1) 0.4 else 0;
            numerator[0] = if (ri == 0) 0 else 0.001;
        }
    };
    var geometry = try choose(std.testing.allocator, &fixture_ranges, &.{1}, 1, 1, 2, .{ .mass = 1, .output = 0 }, 1, 0, 0, 1024 * 1024, .{ .context = null, .fill_fn = Local.fill }, .{});
    defer geometry.deinit();
    try std.testing.expectEqual(adaptive.Level.divisor2, geometry.levels[0]);
    try std.testing.expect(geometry.train_harm_after < geometry.train_harm_before);
}

test "tiled signed chooser resolves a cap-limited equal-gain tie by lower source start" {
    const Local = struct {
        fn fill(_: ?*anyopaque, _: usize, li: usize, _: usize, _: usize, mass: []f64, numerator: []f64, _: solver.Control) !void {
            mass[0] = if (li == 0) 0.2 else 0;
            numerator[0] = 0;
        }
    };
    const ranges = [_]Range{
        .{ .start = 4, .rows = 8, .target_rows = .{ 2, 4, 6, 8 } },
        .{ .start = 20, .rows = 8, .target_rows = .{ 2, 4, 6, 8 } },
    };
    var geometry = try choose(std.testing.allocator, &ranges, &.{1}, 1, 1, 3, .{ .mass = 1, .output = 0 }, 1, 0, 0, 1024 * 1024, .{ .context = null, .fill_fn = Local.fill }, .{});
    defer geometry.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 4, 2 }, geometry.targets);
}
