// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! TRAIN-only reservation of the one already-budgeted exact global head.
//! No heldout query, metric, candidate, or fallback result is accepted here.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const calibrated = @import("attention_match_train_budget_calibrated_critical_threshold_reserve16.zig");

pub const policy = "train_pressure_reserve_one_exact_head_v1|heads=8|primary_percent=91|calibration=reserve16_v2|pressure=highest_threshold|ceiling_infeasible=highest_minimum_middle_rows_then_critical_rows|ties=donor_layer_then_head_index|heldout=forbidden";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const HeadInput = struct {
    donor_layer: u32,
    head_index: usize,
    maximum_probability: []const f64,
};

pub const HeadEvidence = struct {
    donor_layer: u32,
    head_index: usize,
    threshold: f64,
    critical_rows: usize,
    minimum_middle_rows: usize,
    middle_cap_rows: usize,
    feasible: bool,
    calibrated_policy_sha256: [32]u8,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    policy_sha256: [32]u8,
    chosen_donor_layer: u32,
    chosen_head_index: usize,
    pressure_infeasible_count: usize,
    heads: []HeadEvidence,
    evidence_sha256: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.heads);
        self.* = undefined;
    }
};

fn canonicalRanges(ranges: []const admission.Range, source_rows: usize) !void {
    if (source_rows != 7936 or ranges.len != 23) return error.InvalidFitConfig;
    var at: usize = 128;
    var total: usize = 0;
    for (ranges, 0..) |range, index| {
        const expected_rows: usize = if (index + 1 == ranges.len) 128 else 256;
        if (range.start != at or range.rows != expected_rows) return error.InvalidFitConfig;
        at = try std.math.add(usize, at, range.rows);
        total = try std.math.add(usize, total, range.rows);
    }
    if (at != 5888 or total != 5760) return error.InvalidFitConfig;
}

const CeilingPressure = struct { critical_rows: usize, minimum_middle_rows: usize, cap: usize };

fn ceilingPressure(maximum: []const f64, ranges: []const admission.Range, control: solver.Control) !CeilingPressure {
    var critical_rows: usize = 0;
    var minimum: usize = 0;
    var divisor2: usize = 0;
    for (ranges, 0..) |range, ri| {
        if ((ri & 7) == 0) try control.check();
        var count: usize = 0;
        for (maximum[range.start..][0..range.rows], 0..) |value, row| {
            if ((row & 31) == 0) try control.check();
            if (!std.math.isFinite(value) or value < 0 or value > 1) return error.NonFiniteAttention;
            if (value >= calibrated.ceiling_probability) count = try std.math.add(usize, count, 1);
        }
        const reserve = @min(range.rows - 1, calibrated.minimum_residual_rows);
        critical_rows = try std.math.add(usize, critical_rows, count);
        minimum = try std.math.add(usize, minimum, @max(std.math.divCeil(usize, range.rows, 4) catch return error.InvalidFitConfig, try std.math.add(usize, count, reserve)));
        divisor2 = try std.math.add(usize, divisor2, std.math.divCeil(usize, range.rows, 2) catch return error.InvalidFitConfig);
    }
    return .{ .critical_rows = critical_rows, .minimum_middle_rows = minimum, .cap = try std.math.mul(usize, divisor2, 91) / 100 };
}

fn identityBefore(a: HeadEvidence, b: HeadEvidence) bool {
    return a.donor_layer < b.donor_layer or (a.donor_layer == b.donor_layer and a.head_index < b.head_index);
}
fn identityLess(_: void, a: HeadEvidence, b: HeadEvidence) bool {
    return identityBefore(a, b);
}

fn greaterPressure(a: HeadEvidence, b: HeadEvidence) bool {
    if (a.feasible != b.feasible) return !a.feasible;
    if (!a.feasible) {
        if (a.minimum_middle_rows != b.minimum_middle_rows) return a.minimum_middle_rows > b.minimum_middle_rows;
        if (a.critical_rows != b.critical_rows) return a.critical_rows > b.critical_rows;
    } else if (a.threshold != b.threshold) return a.threshold > b.threshold;
    return identityBefore(a, b);
}

pub fn select(allocator: std.mem.Allocator, inputs: []const HeadInput, ranges: []const admission.Range, maximum_workspace_bytes: usize, control: solver.Control) !Result {
    try control.check();
    if (inputs.len != 8) return error.InvalidFitConfig;
    if (maximum_workspace_bytes < 8 * @sizeOf(HeadEvidence)) return error.CompactionWorkspaceLimit;
    try canonicalRanges(ranges, inputs[0].maximum_probability.len);
    for (inputs, 0..) |input, i| {
        if (input.maximum_probability.len != inputs[0].maximum_probability.len) return error.InvalidFitConfig;
        for (inputs[0..i]) |prior| if (input.donor_layer == prior.donor_layer and input.head_index == prior.head_index) return error.InvalidFitConfig;
    }
    const heads = try allocator.alloc(HeadEvidence, inputs.len);
    errdefer allocator.free(heads);
    var infeasible_count: usize = 0;
    for (inputs, heads) |input, *evidence| {
        try control.check();
        var calibrated_result = calibrated.select(allocator, input.maximum_probability, ranges, .{ .primary_percent = 91, .max_workspace_bytes = maximum_workspace_bytes - inputs.len * @sizeOf(HeadEvidence), .control = control }) catch |err| switch (err) {
            error.GeometryBudgetInfeasible => {
                const pressure = try ceilingPressure(input.maximum_probability, ranges, control);
                evidence.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index, .threshold = calibrated.ceiling_probability, .critical_rows = pressure.critical_rows, .minimum_middle_rows = pressure.minimum_middle_rows, .middle_cap_rows = pressure.cap, .feasible = false, .calibrated_policy_sha256 = calibrated.policyDigest() };
                infeasible_count += 1;
                continue;
            },
            else => return err,
        };
        defer calibrated_result.deinit();
        evidence.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index, .threshold = calibrated_result.threshold, .critical_rows = calibrated_result.critical_rows, .minimum_middle_rows = calibrated_result.minimum_middle_rows, .middle_cap_rows = calibrated_result.middle_cap_rows, .feasible = true, .calibrated_policy_sha256 = calibrated_result.policy_sha256 };
    }
    std.mem.sort(HeadEvidence, heads, {}, identityLess);
    var chosen: usize = 0;
    for (heads[1..], 1..) |evidence, index| {
        if (greaterPressure(evidence, heads[chosen])) chosen = index;
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    const calibrated_digest = calibrated.policyDigest();
    hash.update(&calibrated_digest);
    for (heads) |evidence| {
        hash.update(std.mem.asBytes(&evidence.donor_layer));
        hash.update(std.mem.asBytes(&evidence.head_index));
        hash.update(std.mem.asBytes(&evidence.threshold));
        hash.update(std.mem.asBytes(&evidence.critical_rows));
        hash.update(std.mem.asBytes(&evidence.minimum_middle_rows));
        hash.update(std.mem.asBytes(&evidence.middle_cap_rows));
        hash.update(std.mem.asBytes(&evidence.feasible));
    }
    return .{ .allocator = allocator, .policy_sha256 = policy_sha256, .chosen_donor_layer = heads[chosen].donor_layer, .chosen_head_index = heads[chosen].head_index, .pressure_infeasible_count = infeasible_count, .heads = heads, .evidence_sha256 = hash.finalResult() };
}

pub const shape_policy = "train_pressure_reserve_one_exact_head_shape_v1|geometry=caller_authenticated_flat_ranges|calibration=reserve16_v2|pressure=highest_threshold|ties=canonical_identity|heldout=forbidden";
pub const shape_policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(shape_policy, &out, .{});
    break :blk out;
};
pub const formal_bounded_shape_policy = "train_pressure_reserve_one_exact_head_shape_v2|geometry=authenticated_e2_32512_bounded_page|primary60|planned_exact1|physical=captured_source_and_actual_candidate";

pub const HeadIdentity = struct { donor_layer: u32, head_index: usize };
pub const PhysicalPlan = struct {
    flat_geometry: bool,
    primary_percent: u8,
    planned_exact_heads: usize,
    source_physical_bytes: usize,
    maximum_candidate_physical_bytes: usize,
    minimum_global_savings_ratio: f64,
};
pub const Shape = struct {
    source_rows: usize,
    prefix_rows: usize,
    middle_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    ranges: []const admission.Range,
    eligible_head_identities: []const HeadIdentity,
    primary_percent: u8,
    planned_exact_heads: usize,
    physical_plan: PhysicalPlan,
};

fn identityOrder(a: HeadIdentity, b: HeadIdentity) bool {
    return a.donor_layer < b.donor_layer or (a.donor_layer == b.donor_layer and a.head_index < b.head_index);
}

fn validateShape(shape: Shape) !void {
    const formal_bounded = !shape.physical_plan.flat_geometry and shape.source_rows == 32_512 and shape.primary_percent == 60 and
        shape.ranges.len == 119 and shape.eligible_head_identities.len == 3;
    if ((!shape.physical_plan.flat_geometry and !formal_bounded) or shape.source_rows == 0 or shape.middle_rows == 0 or
        shape.chunk_rows == 0 or shape.ranges.len == 0 or shape.eligible_head_identities.len == 0 or
        shape.primary_percent == 0 or shape.primary_percent > 100 or shape.planned_exact_heads != 1 or
        shape.physical_plan.primary_percent != shape.primary_percent or shape.physical_plan.planned_exact_heads != 1 or
        shape.physical_plan.source_physical_bytes == 0 or shape.physical_plan.maximum_candidate_physical_bytes == 0 or
        !std.math.isFinite(shape.physical_plan.minimum_global_savings_ratio) or shape.physical_plan.minimum_global_savings_ratio < 1)
        return error.UnsupportedShape;
    const prefix_middle = try std.math.add(usize, shape.prefix_rows, shape.middle_rows);
    if (try std.math.add(usize, prefix_middle, shape.recent_rows) != shape.source_rows) return error.InvalidFitConfig;
    const actual_ratio = @as(f64, @floatFromInt(shape.physical_plan.source_physical_bytes)) /
        @as(f64, @floatFromInt(shape.physical_plan.maximum_candidate_physical_bytes));
    if (!std.math.isFinite(actual_ratio) or actual_ratio < shape.physical_plan.minimum_global_savings_ratio) return error.PhysicalBudgetUnsupported;
    var at = shape.prefix_rows;
    var middle: usize = 0;
    for (shape.ranges, 0..) |range, i| {
        if (range.start != at or range.rows == 0 or range.rows > shape.chunk_rows or
            (i + 1 != shape.ranges.len and range.rows != shape.chunk_rows)) return error.InvalidFitConfig;
        at = try std.math.add(usize, at, range.rows);
        middle = try std.math.add(usize, middle, range.rows);
    }
    if (middle != shape.middle_rows or at != prefix_middle) return error.InvalidFitConfig;
    for (shape.eligible_head_identities, 0..) |identity, i| {
        if (i != 0 and !identityOrder(shape.eligible_head_identities[i - 1], identity)) return error.InvalidFitConfig;
    }
}

fn shapeCeilingPressure(maximum: []const f64, shape: Shape, control: solver.Control) !CeilingPressure {
    var pressure = try ceilingPressure(maximum, shape.ranges, control);
    var divisor2: usize = 0;
    for (shape.ranges) |range| divisor2 = try std.math.add(usize, divisor2, std.math.divCeil(usize, range.rows, 2) catch return error.InvalidFitConfig);
    pressure.cap = try std.math.mul(usize, divisor2, shape.primary_percent) / 100;
    return pressure;
}

/// Shape-aware successor. The legacy `select` function and its digest remain
/// unchanged. The only non-flat shape is the authenticated E2/32K bounded-page
/// tuple; its physical plan is supplied from the captured native inventory.
pub fn selectForShape(allocator: std.mem.Allocator, inputs: []const HeadInput, shape: Shape, maximum_workspace_bytes: usize, control: solver.Control) !Result {
    try control.check();
    try validateShape(shape);
    if (inputs.len != shape.eligible_head_identities.len) return error.InvalidFitConfig;
    const evidence_bytes = try std.math.mul(usize, inputs.len, @sizeOf(HeadEvidence));
    if (maximum_workspace_bytes < evidence_bytes) return error.CompactionWorkspaceLimit;
    for (inputs, 0..) |input, i| {
        if (input.maximum_probability.len != shape.source_rows) return error.InvalidFitConfig;
        for (inputs[0..i]) |prior| if (input.donor_layer == prior.donor_layer and input.head_index == prior.head_index) return error.InvalidFitConfig;
        var found = false;
        for (shape.eligible_head_identities) |identity| if (identity.donor_layer == input.donor_layer and identity.head_index == input.head_index) {
            found = true;
            break;
        };
        if (!found) return error.InvalidFitConfig;
    }
    const heads = try allocator.alloc(HeadEvidence, inputs.len);
    errdefer allocator.free(heads);
    var infeasible_count: usize = 0;
    for (inputs, heads) |input, *evidence| {
        try control.check();
        var calibrated_result = calibrated.select(allocator, input.maximum_probability, shape.ranges, .{ .primary_percent = shape.primary_percent, .max_workspace_bytes = maximum_workspace_bytes - evidence_bytes, .control = control }) catch |err| switch (err) {
            error.GeometryBudgetInfeasible => {
                const pressure = try shapeCeilingPressure(input.maximum_probability, shape, control);
                evidence.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index, .threshold = calibrated.ceiling_probability, .critical_rows = pressure.critical_rows, .minimum_middle_rows = pressure.minimum_middle_rows, .middle_cap_rows = pressure.cap, .feasible = false, .calibrated_policy_sha256 = calibrated.policyDigest() };
                infeasible_count += 1;
                continue;
            },
            else => return err,
        };
        defer calibrated_result.deinit();
        evidence.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index, .threshold = calibrated_result.threshold, .critical_rows = calibrated_result.critical_rows, .minimum_middle_rows = calibrated_result.minimum_middle_rows, .middle_cap_rows = calibrated_result.middle_cap_rows, .feasible = true, .calibrated_policy_sha256 = calibrated_result.policy_sha256 };
    }
    std.mem.sort(HeadEvidence, heads, {}, identityLess);
    var chosen: usize = 0;
    for (heads[1..], 1..) |evidence, index| if (greaterPressure(evidence, heads[chosen])) {
        chosen = index;
    };
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    if (shape.physical_plan.flat_geometry) {
        hash.update(shape_policy);
        hash.update(&shape_policy_sha256);
    } else {
        hash.update(formal_bounded_shape_policy);
        var formal_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(formal_bounded_shape_policy, &formal_digest, .{});
        hash.update(&formal_digest);
    }
    const calibrated_digest = calibrated.policyDigest();
    hash.update(&calibrated_digest);
    inline for (.{ shape.source_rows, shape.prefix_rows, shape.middle_rows, shape.recent_rows, shape.chunk_rows, shape.primary_percent, shape.planned_exact_heads }) |value| hash.update(std.mem.asBytes(&value));
    hash.update(std.mem.asBytes(&shape.physical_plan.flat_geometry));
    hash.update(std.mem.asBytes(&shape.physical_plan.primary_percent));
    hash.update(std.mem.asBytes(&shape.physical_plan.planned_exact_heads));
    hash.update(std.mem.asBytes(&shape.physical_plan.source_physical_bytes));
    hash.update(std.mem.asBytes(&shape.physical_plan.maximum_candidate_physical_bytes));
    hash.update(std.mem.asBytes(&shape.physical_plan.minimum_global_savings_ratio));
    for (shape.ranges) |range| {
        hash.update(std.mem.asBytes(&range.start));
        hash.update(std.mem.asBytes(&range.rows));
    }
    for (shape.eligible_head_identities) |identity| {
        hash.update(std.mem.asBytes(&identity.donor_layer));
        hash.update(std.mem.asBytes(&identity.head_index));
    }
    for (heads) |evidence| {
        hash.update(std.mem.asBytes(&evidence.donor_layer));
        hash.update(std.mem.asBytes(&evidence.head_index));
        hash.update(std.mem.asBytes(&evidence.threshold));
        hash.update(std.mem.asBytes(&evidence.critical_rows));
        hash.update(std.mem.asBytes(&evidence.minimum_middle_rows));
        hash.update(std.mem.asBytes(&evidence.middle_cap_rows));
        hash.update(std.mem.asBytes(&evidence.feasible));
    }
    return .{ .allocator = allocator, .policy_sha256 = shape_policy_sha256, .chosen_donor_layer = heads[chosen].donor_layer, .chosen_head_index = heads[chosen].head_index, .pressure_infeasible_count = infeasible_count, .heads = heads, .evidence_sha256 = hash.finalResult() };
}

fn fixtureRanges() [23]admission.Range {
    var ranges: [23]admission.Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == ranges.len) 128 else 256 };
        start += range.rows;
    }
    return ranges;
}

test "highest TRAIN threshold wins and canonical identity breaks ties" {
    const low: [7936]f64 = @splat(0);
    var high: [7936]f64 = @splat(0);
    const ranges = fixtureRanges();
    for (ranges) |range| @memset(high[range.start..][0..@min(100, range.rows)], 0.02);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(23 - i), .head_index = 0, .maximum_probability = if (i == 3 or i == 4) &high else &low };
    var result = try select(std.testing.allocator, &inputs, &ranges, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 19), result.chosen_donor_layer);
    try std.testing.expectEqual(@as(usize, 0), result.pressure_infeasible_count);
}

test "ceiling infeasible outranks feasible pressure without heldout evidence" {
    var feasible: [7936]f64 = @splat(0);
    var infeasible: [7936]f64 = @splat(0);
    const ranges = fixtureRanges();
    @memset(feasible[128..228], 0.01);
    @memset(infeasible[128..384], 0.025);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = if (i == 6) &infeasible else &feasible };
    var result = try select(std.testing.allocator, &inputs, &ranges, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 11), result.chosen_donor_layer);
    try std.testing.expectEqual(@as(usize, 1), result.pressure_infeasible_count);
    for (result.heads) |evidence| if (evidence.donor_layer == result.chosen_donor_layer) {
        try std.testing.expect(!evidence.feasible);
        break;
    };
}

test "duplicate or incomplete head tuples reject and cancellation publishes nothing" {
    const ranges = fixtureRanges();
    const maxima: [7936]f64 = @splat(0);
    const short = [_]HeadInput{.{ .donor_layer = 5, .head_index = 0, .maximum_probability = &maxima }};
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &short, &ranges, 1 << 20, .{}));
    var duplicate: [8]HeadInput = @splat(.{ .donor_layer = 5, .head_index = 0, .maximum_probability = &maxima });
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &duplicate, &ranges, 1 << 20, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    for (&duplicate, 0..) |*input, i| input.donor_layer = @intCast(5 + i);
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &duplicate, &ranges, 1 << 20, .{ .check_fn = Cancel.check }));
}

fn allocationCase(allocator: std.mem.Allocator) !void {
    const ranges = fixtureRanges();
    var maxima: [7936]f64 = @splat(0);
    @memset(maxima[128..384], 0.01);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    var result = try select(allocator, &inputs, &ranges, 1 << 20, .{});
    result.deinit();
}

test "allocation failure and mid calibration cancellation publish no partial result" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
    const ranges = fixtureRanges();
    const maxima: [7936]f64 = @splat(0.01);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    const Cancel = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 20) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &inputs, &ranges, 1 << 20, .{ .context = &cancel, .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 20), cancel.checks);
}

test "workspace cap failure is typed separately from head shape" {
    const ranges = fixtureRanges();
    const maxima: [7936]f64 = @splat(0);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, &inputs, &ranges, 1, .{}));
}

fn currentShape(ranges: []const admission.Range, identities: []const HeadIdentity) Shape {
    return .{
        .source_rows = 7936,
        .prefix_rows = 128,
        .middle_rows = 5760,
        .recent_rows = 2048,
        .chunk_rows = 256,
        .ranges = ranges,
        .eligible_head_identities = identities,
        .primary_percent = 91,
        .planned_exact_heads = 1,
        .physical_plan = .{ .flat_geometry = true, .primary_percent = 91, .planned_exact_heads = 1, .source_physical_bytes = 3_000_000, .maximum_candidate_physical_bytes = 1_900_000, .minimum_global_savings_ratio = 1.5 },
    };
}

test "shape entrypoint agrees on current chosen head and binds changed shape evidence" {
    const ranges = fixtureRanges();
    const low: [7936]f64 = @splat(0);
    var high: [7936]f64 = @splat(0);
    for (ranges) |range| @memset(high[range.start..][0..@min(100, range.rows)], 0.02);
    var inputs: [8]HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(23 - i), .head_index = 0, .maximum_probability = if (i == 3 or i == 4) &high else &low };
    const identities = [_]HeadIdentity{ .{ .donor_layer = 16, .head_index = 0 }, .{ .donor_layer = 17, .head_index = 0 }, .{ .donor_layer = 18, .head_index = 0 }, .{ .donor_layer = 19, .head_index = 0 }, .{ .donor_layer = 20, .head_index = 0 }, .{ .donor_layer = 21, .head_index = 0 }, .{ .donor_layer = 22, .head_index = 0 }, .{ .donor_layer = 23, .head_index = 0 } };
    var legacy = try select(std.testing.allocator, &inputs, &ranges, 1 << 20, .{});
    defer legacy.deinit();
    var shaped = try selectForShape(std.testing.allocator, &inputs, currentShape(&ranges, &identities), 1 << 20, .{});
    defer shaped.deinit();
    try std.testing.expectEqual(legacy.chosen_donor_layer, shaped.chosen_donor_layer);
    try std.testing.expectEqual(legacy.chosen_head_index, shaped.chosen_head_index);
    try std.testing.expectEqualSlices(u8, &shape_policy_sha256, &shaped.policy_sha256);
    try std.testing.expect(!std.mem.eql(u8, &legacy.evidence_sha256, &shaped.evidence_sha256));
    var changed = currentShape(&ranges, &identities);
    changed.physical_plan.source_physical_bytes += 1;
    var changed_result = try selectForShape(std.testing.allocator, &inputs, changed, 1 << 20, .{});
    defer changed_result.deinit();
    try std.testing.expectEqual(shaped.chosen_donor_layer, changed_result.chosen_donor_layer);
    try std.testing.expect(!std.mem.eql(u8, &shaped.evidence_sha256, &changed_result.evidence_sha256));
}

fn smallShapeCase(allocator: std.mem.Allocator) !void {
    const ranges = [_]admission.Range{ .{ .start = 4, .rows = 4 }, .{ .start = 8, .rows = 4 } };
    const identities = [_]HeadIdentity{ .{ .donor_layer = 5, .head_index = 0 }, .{ .donor_layer = 6, .head_index = 0 } };
    const maxima: [16]f64 = @splat(0.01);
    const inputs = [_]HeadInput{ .{ .donor_layer = 6, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 5, .head_index = 0, .maximum_probability = &maxima } };
    const shape: Shape = .{ .source_rows = 16, .prefix_rows = 4, .middle_rows = 8, .recent_rows = 4, .chunk_rows = 4, .ranges = &ranges, .eligible_head_identities = &identities, .primary_percent = 91, .planned_exact_heads = 1, .physical_plan = .{ .flat_geometry = true, .primary_percent = 91, .planned_exact_heads = 1, .source_physical_bytes = 300, .maximum_candidate_physical_bytes = 100, .minimum_global_savings_ratio = 1.5 } };
    var result = try selectForShape(allocator, &inputs, shape, 1 << 20, .{});
    result.deinit();
}

test "shape entrypoint cap OOM cancellation and unsupported geometry are typed" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, smallShapeCase, .{});
    const ranges = [_]admission.Range{ .{ .start = 4, .rows = 4 }, .{ .start = 8, .rows = 4 } };
    const identities = [_]HeadIdentity{ .{ .donor_layer = 5, .head_index = 0 }, .{ .donor_layer = 6, .head_index = 0 } };
    const maxima: [16]f64 = @splat(0.01);
    const inputs = [_]HeadInput{ .{ .donor_layer = 5, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 6, .head_index = 0, .maximum_probability = &maxima } };
    var shape: Shape = .{ .source_rows = 16, .prefix_rows = 4, .middle_rows = 8, .recent_rows = 4, .chunk_rows = 4, .ranges = &ranges, .eligible_head_identities = &identities, .primary_percent = 91, .planned_exact_heads = 1, .physical_plan = .{ .flat_geometry = true, .primary_percent = 91, .planned_exact_heads = 1, .source_physical_bytes = 300, .maximum_candidate_physical_bytes = 100, .minimum_global_savings_ratio = 1.5 } };
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectForShape(std.testing.failing_allocator, &inputs, shape, 1, .{}));
    shape.physical_plan.flat_geometry = false;
    try std.testing.expectError(error.UnsupportedShape, selectForShape(std.testing.failing_allocator, &inputs, shape, 1 << 20, .{}));
    shape.physical_plan.flat_geometry = true;
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, selectForShape(std.testing.failing_allocator, &inputs, shape, 1 << 20, .{ .check_fn = Cancel.check }));
}
