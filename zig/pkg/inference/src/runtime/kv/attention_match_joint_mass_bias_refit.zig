// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Diagnostic-only whole-head TRAIN mass refit on immutable selected support.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const fit_workspace_bytes: usize = 512 * 1024 * 1024;
pub const ComponentCap = enum {
    mib512,
    mib768,

    pub fn bytes(self: ComponentCap) usize {
        return switch (self) {
            .mib512 => 512 * 1024 * 1024,
            .mib768 => 768 * 1024 * 1024,
        };
    }
};
pub const combined_cap_bytes: usize = 1024 * 1024 * 1024;
pub const batch_rows: usize = 64;

pub const Range = struct { start: usize, rows: usize, target_budget: usize, exact_core_source_keys: []const usize };

pub const SelectedWeights = struct { weights: []f64, sweeps: usize };
pub const WeightSelector = struct {
    context: ?*anyopaque,
    select_fn: *const fn (?*anyopaque, std.mem.Allocator, []const f64, []const f64, usize, usize, []const f64, solver.Config, usize) anyerror!SelectedWeights,
};
pub const Options = struct { selector: ?WeightSelector = null };

pub const Result = struct {
    allocator: std.mem.Allocator,
    bias: []f64,
    weights: []f64,
    train_before: solver.Metrics,
    train_after: solver.Metrics,
    nnls_sweeps: usize,
    residual_rows: usize,
    fit_peak_bytes: usize,
    validation_peak_bytes: usize,
    floored_weight_count: usize,
    max_floor_perturbation: f64,
    pub fn deinit(self: *Result) void {
        self.allocator.free(self.bias);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

/// Includes caller A plus the active solver's matrix and worst full-active
/// Householder QR/RHS. It deliberately overcounts vectors shared across phases.
pub fn fitPeakBytes(q: usize, m: usize, original_rows: usize, candidate_rows: usize, backend_bytes: usize) !usize {
    if (q == 0 or m == 0) return error.InvalidFitConfig;
    // Caller A + active-solver matrix, then the worst full-active dense QR.
    var scalars = try mul(try mul(q, m), 2);
    scalars = try add(scalars, try mul(try add(q, m), m));
    // rho, caller residual/norms/weights, solver rhs/solution/trial, QR rhs/v/dots.
    scalars = try add(scalars, try add(try mul(q, 5), try mul(m, 8)));
    scalars = try add(scalars, try mul(batch_rows, original_rows));
    var bytes = try mul(scalars, @sizeOf(f64));
    bytes = try add(bytes, candidate_rows); // fixed bool
    bytes = try add(bytes, try mul(candidate_rows, @sizeOf(f64))); // owned replacement bias overlaps fit temps
    bytes = try add(bytes, try mul(m, @sizeOf(usize))); // residual slots
    bytes = try add(bytes, std.mem.alignForward(usize, m, @alignOf(f64))); // solver active bool
    bytes = try add(bytes, try mul(m, @sizeOf(usize))); // solver indices
    bytes = try add(bytes, backend_bytes);
    return bytes;
}

fn legacyValueWorkspaceBytes(q: usize, m: usize, d: usize) !usize {
    const rows = try add(q, m);
    var scalars = try mul(rows, m);
    scalars = try add(scalars, try mul(rows, d));
    scalars = try add(scalars, m);
    scalars = try add(scalars, try add(q, @max(m, d)));
    return mul(scalars, @sizeOf(f64));
}

pub fn valueBackendExtraBytes(q: usize, m: usize, d: usize, cfg: solver.Config) !usize {
    const backend = cfg.value_backend orelse return 0;
    const declared = try backend.workspace_fn(backend.context, q, m, d, cfg.ridge);
    const legacy = try legacyValueWorkspaceBytes(q, m, d);
    return declared -| legacy;
}

fn f32Round(value: f64) !f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
    return @floatCast(@as(f32, @floatCast(value)));
}
fn sameF64(a: f64, b: f64) bool {
    return @as(u64, @bitCast(a)) == @as(u64, @bitCast(b));
}

fn validateRanges(original_rows: usize, candidate_positions: []const u64, ranges: []const Range) !void {
    if (original_rows != 7936 or ranges.len != 23) return error.InvalidFitConfig;
    var middle_total: usize = 0;
    for (ranges, 0..) |range, index| {
        const start = 128 + index * 256;
        const rows = @min(@as(usize, 256), 5888 - start);
        if (range.start != start or range.rows != rows or range.target_budget == 0 or range.target_budget > rows) return error.InvalidFitConfig;
        var retained: usize = 0;
        for (candidate_positions) |position| {
            const key = std.math.cast(usize, position) orelse return error.InvalidFitConfig;
            if (key >= start and key < start + rows) retained += 1;
        }
        if (retained != range.target_budget) return error.InvalidFitConfig;
        for (range.exact_core_source_keys, 0..) |key, i| if (key < start or key >= start + rows or std.mem.indexOfScalar(usize, range.exact_core_source_keys[0..i], key) != null or std.mem.indexOfScalar(u64, candidate_positions, @intCast(key)) == null) return error.InvalidFitConfig;
        middle_total = try add(middle_total, range.target_budget);
    }
    if (candidate_positions.len != 2176 + middle_total) return error.InvalidFitConfig;
}

const RowPolicy = struct { expected_coverage: u64, identity_fixed: bool };
fn rowPolicy(key: usize, original_coverage: []const u64, ranges: []const Range) !RowPolicy {
    if (key >= original_coverage.len) return error.InvalidFitConfig;
    if (key < 128 or key >= 5888) return .{ .expected_coverage = original_coverage[key], .identity_fixed = true };
    for (ranges) |range| if (key >= range.start and key < range.start + range.rows) {
        const identity = range.target_budget == range.rows;
        return .{ .expected_coverage = if (identity) original_coverage[key] else original_coverage[range.start + range.rows - 1], .identity_fixed = identity };
    };
    return error.InvalidFitConfig;
}
fn validateRowCoverage(key: usize, actual: u64, original_coverage: []const u64, ranges: []const Range) !RowPolicy {
    const policy = try rowPolicy(key, original_coverage, ranges);
    if (actual != policy.expected_coverage) return error.InvalidFitConfig;
    return policy;
}
fn fixedRow(key: usize, policy: RowPolicy, ranges: []const Range) bool {
    if (policy.identity_fixed) return true;
    for (ranges) |range| if (std.mem.indexOfScalar(usize, range.exact_core_source_keys, key) != null) return true;
    return false;
}

pub fn fitJointResidualMassBias(
    allocator: std.mem.Allocator,
    original: solver.Head,
    original_positions: []const u64,
    original_coverage: []const u64,
    baseline: solver.Head,
    candidate_positions: []const u64,
    candidate_coverage: []const u64,
    ranges: []const Range,
    train: []const f64,
    cfg: solver.Config,
    external_live_bytes: usize,
) !Result {
    return fitJointResidualMassBiasOptions(allocator, original, original_positions, original_coverage, baseline, candidate_positions, candidate_coverage, ranges, train, cfg, external_live_bytes, .{});
}

pub fn fitJointResidualMassBiasOptions(
    allocator: std.mem.Allocator,
    original: solver.Head,
    original_positions: []const u64,
    original_coverage: []const u64,
    baseline: solver.Head,
    candidate_positions: []const u64,
    candidate_coverage: []const u64,
    ranges: []const Range,
    train: []const f64,
    cfg: solver.Config,
    external_live_bytes: usize,
    options: Options,
) !Result {
    try original.validate();
    try baseline.validate();
    try cfg.control.check();
    const component_cap: ComponentCap = if (cfg.max_workspace_bytes == ComponentCap.mib512.bytes()) .mib512 else if (cfg.max_workspace_bytes == ComponentCap.mib768.bytes()) .mib768 else return error.InvalidFitConfig;
    _ = component_cap;
    if (original.bias == null or baseline.bias == null or original.dim != baseline.dim or original.scale != baseline.scale or original.softcap != baseline.softcap or original_positions.len != original.rows or original_coverage.len != original.rows or candidate_positions.len != baseline.rows or candidate_coverage.len != baseline.rows or train.len == 0 or train.len % original.dim != 0) return error.InvalidFitConfig;
    for (original_positions, 0..) |position, index| if (position != (std.math.cast(u64, index) orelse return error.InvalidFitConfig)) return error.InvalidFitConfig;
    try validateRanges(original.rows, candidate_positions, ranges);
    const q = train.len / original.dim;
    if (external_live_bytes > combined_cap_bytes) return error.CompactionWorkspaceLimit;
    const backend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, original.rows, 1, @min(q, batch_rows), original.dim) else 0;
    const validation_peak = try solver.validationBytes(original.rows, baseline.rows, original.dim, q, cfg);
    if (validation_peak > combined_cap_bytes - external_live_bytes) return error.CompactionWorkspaceLimit;
    const train_before = try solver.validationMetrics(allocator, original, baseline, train, cfg);
    var temps_live = true;
    const fixed = try allocator.alloc(bool, baseline.rows);
    defer if (temps_live) allocator.free(fixed);
    @memset(fixed, false);
    var residual_count: usize = 0;
    for (candidate_positions, candidate_coverage, 0..) |position, coverage, slot| {
        const key = std.math.cast(usize, position) orelse return error.InvalidFitConfig;
        const policy = try validateRowCoverage(key, coverage, original_coverage, ranges);
        if (key >= original.rows or (slot != 0 and position <= candidate_positions[slot - 1])) return error.InvalidFitConfig;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(baseline.keys[slot * baseline.dim ..][0..baseline.dim]), std.mem.sliceAsBytes(original.keys[key * original.dim ..][0..original.dim]))) return error.InvalidFitConfig;
        const is_fixed = fixedRow(key, policy, ranges);
        fixed[slot] = is_fixed;
        if (is_fixed) {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(baseline.values[slot * baseline.dim ..][0..baseline.dim]), std.mem.sliceAsBytes(original.values[key * original.dim ..][0..original.dim])) or !sameF64(baseline.bias.?[slot], original.bias.?[key])) return error.InvalidFitConfig;
        } else residual_count += 1;
    }
    if (residual_count == 0) return error.InvalidFitConfig;
    const value_backend_extra = try valueBackendExtraBytes(q, residual_count, 1, cfg);
    const peak = try fitPeakBytes(q, residual_count, original.rows, baseline.rows, try add(backend_bytes, value_backend_extra));
    if (peak > cfg.max_workspace_bytes or external_live_bytes > combined_cap_bytes or peak > combined_cap_bytes - external_live_bytes) return error.CompactionWorkspaceLimit;
    const retained_result_bytes = try add(try mul(baseline.rows, @sizeOf(f64)), try mul(residual_count, @sizeOf(f64)));
    if (try add(validation_peak, retained_result_bytes) > combined_cap_bytes - external_live_bytes) return error.CompactionWorkspaceLimit;
    const residual_slots = try allocator.alloc(usize, residual_count);
    defer if (temps_live) allocator.free(residual_slots);
    var ri: usize = 0;
    for (fixed, 0..) |is_fixed, slot| if (!is_fixed) {
        residual_slots[ri] = slot;
        ri += 1;
    };
    const a = try allocator.alloc(f64, try mul(q, residual_count));
    defer if (temps_live) allocator.free(a);
    const rho = try allocator.alloc(f64, q);
    defer if (temps_live) allocator.free(rho);
    const tile = try allocator.alloc(f64, try mul(batch_rows, original.rows));
    defer if (temps_live) allocator.free(tile);
    var first: usize = 0;
    while (first < q) {
        try cfg.control.check();
        const take = @min(batch_rows, q - first);
        try solver.attendBatch(allocator, original, train[first * original.dim ..][0 .. take * original.dim], tile[0 .. take * original.rows], null, null, cfg);
        for (0..take) |row| {
            const p = tile[row * original.rows ..][0..original.rows];
            var fixed_mass: f64 = 0;
            for (fixed, candidate_positions) |is_fixed, position| {
                if (is_fixed) fixed_mass += p[@intCast(position)];
            }
            rho[first + row] = 1 - fixed_mass;
            if (!std.math.isFinite(rho[first + row]) or rho[first + row] < 0) return error.CompactionFitRejected;
            for (residual_slots, 0..) |slot, col| a[(first + row) * residual_count + col] = p[@intCast(candidate_positions[slot])];
        }
        first += take;
    }
    const warm = try allocator.alloc(f64, residual_count);
    defer if (temps_live) allocator.free(warm);
    for (residual_slots, warm) |slot, *weight| {
        const key: usize = @intCast(candidate_positions[slot]);
        weight.* = @exp(baseline.bias.?[slot] - original.bias.?[key]);
        if (!std.math.isFinite(weight.*) or weight.* < 0) return error.CompactionFitRejected;
    }
    const selection = if (options.selector) |selector|
        try selector.select_fn(selector.context, allocator, a, rho, q, residual_count, warm, cfg, external_live_bytes)
    else blk: {
        const weights = try allocator.dupe(f64, warm);
        errdefer allocator.free(weights);
        const residual = try allocator.alloc(f64, q);
        defer allocator.free(residual);
        const norms = try allocator.alloc(f64, residual_count);
        defer allocator.free(norms);
        break :blk SelectedWeights{ .weights = weights, .sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, a, rho, q, residual_count, weights, residual, norms, cfg, true) };
    };
    const weights = selection.weights;
    errdefer allocator.free(weights);
    const sweeps = selection.sweeps;
    const bias = try allocator.dupe(f64, baseline.bias.?);
    errdefer allocator.free(bias);
    var floored_weight_count: usize = 0;
    var max_floor_perturbation: f64 = 0;
    for (residual_slots, weights) |slot, weight| {
        const key: usize = @intCast(candidate_positions[slot]);
        if (weight < 0 or !std.math.isFinite(weight)) return error.CompactionFitRejected;
        const stored_weight = @max(weight, cfg.min_weight);
        if (stored_weight != weight) {
            floored_weight_count += 1;
            max_floor_perturbation = @max(max_floor_perturbation, stored_weight - weight);
        }
        bias[slot] = try f32Round(original.bias.?[key] + @log(stored_weight));
    }
    const fitted: solver.Head = .{ .keys = baseline.keys, .values = baseline.values, .bias = bias, .rows = baseline.rows, .dim = baseline.dim, .scale = baseline.scale, .softcap = baseline.softcap };
    allocator.free(warm);
    allocator.free(tile);
    allocator.free(rho);
    allocator.free(a);
    allocator.free(residual_slots);
    allocator.free(fixed);
    temps_live = false;
    const train_after = try solver.validationMetrics(allocator, original, fitted, train, cfg);
    return .{ .allocator = allocator, .bias = bias, .weights = weights, .train_before = train_before, .train_after = train_after, .nnls_sweeps = sweeps, .residual_rows = residual_count, .fit_peak_bytes = peak, .validation_peak_bytes = validation_peak, .floored_weight_count = floored_weight_count, .max_floor_perturbation = max_floor_perturbation };
}

test "joint mass fit admission rejects before backend-sized work" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitPeakBytes(std.math.maxInt(usize), 2, 7936, 4908, 0));
    const peak = try fitPeakBytes(8, 3, 16, 7, 4096);
    try std.testing.expect(peak > 8 * 3 * 8);
    try std.testing.expect(peak <= fit_workspace_bytes);
    try std.testing.expect(try add(peak, fit_workspace_bytes - peak) == fit_workspace_bytes);
    try std.testing.expect((try add(peak, fit_workspace_bytes - peak + 1)) > fit_workspace_bytes);
}

test "value backend workspace charges only the positive legacy delta" {
    const Probe = struct {
        fn workspace(context: ?*anyopaque, _: usize, _: usize, _: usize, _: f64) !usize {
            return @intFromPtr(context.?);
        }
        fn fit(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, _: solver.Control, _: []f64) !void {}
    };
    const legacy = try legacyValueWorkspaceBytes(7, 3, 1);
    var cfg: solver.Config = .{ .target_rows = 1 };
    cfg.value_backend = .{ .context = @ptrFromInt(legacy + 4096), .workspace_fn = Probe.workspace, .fit_fn = Probe.fit };
    try std.testing.expectEqual(@as(usize, 4096), try valueBackendExtraBytes(7, 3, 1, cfg));
    cfg.value_backend.?.context = @ptrFromInt(legacy - 1);
    try std.testing.expectEqual(@as(usize, 0), try valueBackendExtraBytes(7, 3, 1, cfg));
}

test "joint warm NNLS solves a known system before storage flooring" {
    const a = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const rho = [_]f64{ 2, 3, 5 };
    var weights = [_]f64{ 0.5, 0.5 };
    var residual: [3]f64 = undefined;
    var norms: [2]f64 = undefined;
    _ = try solver.fitMassActiveTargetDiagnostic(std.testing.allocator, &a, &rho, 3, 2, &weights, &residual, &norms, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set }, true);
    try std.testing.expectApproxEqAbs(@as(f64, 2), weights[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), weights[1], 1e-12);
}

test "coverage and fixed classification follow compacted versus identity range authority" {
    var coverage: [640]u64 = undefined;
    for (&coverage, 0..) |*value, index| value.* = @intCast(index + 1);
    const compacted = Range{ .start = 128, .rows = 256, .target_budget = 128, .exact_core_source_keys = &.{173} };
    const identity = Range{ .start = 384, .rows = 256, .target_budget = 256, .exact_core_source_keys = &.{} };
    const compacted_policy = try rowPolicy(173, &coverage, &.{ compacted, identity });
    try std.testing.expectEqual(coverage[383], compacted_policy.expected_coverage);
    try std.testing.expect(!compacted_policy.identity_fixed);
    try std.testing.expect(fixedRow(173, compacted_policy, &.{ compacted, identity }));
    try std.testing.expect(!fixedRow(174, try rowPolicy(174, &coverage, &.{ compacted, identity }), &.{ compacted, identity }));
    try std.testing.expect(compacted_policy.expected_coverage != coverage[173]);
    try std.testing.expectError(error.InvalidFitConfig, validateRowCoverage(173, coverage[173], &coverage, &.{ compacted, identity }));
    const identity_policy = try rowPolicy(511, &coverage, &.{ compacted, identity });
    try std.testing.expectEqual(coverage[511], identity_policy.expected_coverage);
    try std.testing.expect(identity_policy.identity_fixed);
    try std.testing.expect(fixedRow(511, identity_policy, &.{ compacted, identity }));
    try std.testing.expectError(error.InvalidFitConfig, rowPolicy(700, &coverage, &.{ compacted, identity }));
}

test "typed component caps cannot raise the combined cap" {
    try std.testing.expectEqual(@as(usize, 512 * 1024 * 1024), ComponentCap.mib512.bytes());
    try std.testing.expectEqual(@as(usize, 768 * 1024 * 1024), ComponentCap.mib768.bytes());
    try std.testing.expect(ComponentCap.mib768.bytes() < combined_cap_bytes);
}

test "fit peak accounts five q and eight m vector scalars" {
    const q: usize = 15360;
    const m: usize = 1822;
    const rows: usize = 7936;
    const candidate_rows: usize = 4907;
    var scalars = q * m * 2 + (q + m) * m;
    scalars += q * 5 + m * 8;
    scalars += batch_rows * rows;
    var expected = scalars * @sizeOf(f64);
    expected += candidate_rows + candidate_rows * @sizeOf(f64);
    expected += m * @sizeOf(usize) + std.mem.alignForward(usize, m, @alignOf(f64)) + m * @sizeOf(usize);
    try std.testing.expectEqual(expected, try fitPeakBytes(q, m, rows, candidate_rows, 0));
}
