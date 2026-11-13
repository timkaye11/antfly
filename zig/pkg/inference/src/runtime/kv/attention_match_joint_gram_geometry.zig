const std = @import("std");
const solver = @import("attention_match.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");

pub const policy = "greedy_positive_per_query_gram_train_objective_per_row_v1";
pub const incremental_policy = "greedy_positive_per_query_gram_incremental_candidate_v1";
pub const incremental_budget_percent_policy = "greedy_positive_per_query_gram_incremental_candidate_budget_percent_v1";
pub const exchange_policy = "train_lex_mass_output_budget_exchange_frontier8_sweeps4_v1";
pub const max_candidates: usize = 96;
pub const exchange_frontier_limit: usize = 8;
pub const exchange_maximum_sweeps: usize = 4;

pub const ExchangeConfig = struct {
    max_relative_mass_error: f64,
    max_relative_output_rmse: f64,
};

pub const DeltaAccessor = struct {
    /// Synchronous, allocation-free view over caller-owned scored packets and
    /// flat TRAIN P/Y. Those objects must be included in external_live_bytes.
    context: ?*anyopaque,
    fill_fn: *const fn (?*anyopaque, usize, usize, usize, usize, []f64, []f64, solver.Control) anyerror!void,
};

pub const GramBackend = struct {
    context: ?*anyopaque,
    /// Pure/stable admission authority: repeated calls for the same `(k,dim)`
    /// must return the same bound and compute_fn must not exceed it.
    workspace_fn: *const fn (?*anyopaque, usize, usize) anyerror!usize,
    compute_fn: *const fn (?*anyopaque, []const f64, usize, usize, []f64, []u8, solver.Control) anyerror!void,
};

pub const Range = struct { start: usize, rows: usize, target_rows: [4]usize };

pub const Representation = struct {
    allocator: std.mem.Allocator,
    query_count: usize,
    dim: usize,
    ranges: []Range,
    mass: []f64, // query-major Q x K
    gram: []f64, // query-major packed lower triangle
    original_y_norm_squared: f64,
    digest: [32]u8,
    peak_bytes: usize,
    chooser_bytes: usize,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.gram);
        self.allocator.free(self.mass);
        self.allocator.free(self.ranges);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn tri(k: usize) !usize {
    return (try mul(k, try add(k, 1))) / 2;
}
fn packedIndex(i: usize, j: usize) usize {
    const row = @max(i, j);
    const column = @min(i, j);
    return row * (row + 1) / 2 + column;
}

fn hashU64Le(h: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    h.update(&bytes);
}

const Fixture = struct {
    calls: usize = 0,
    fn fill(raw: ?*anyopaque, range: usize, level: usize, first: usize, count: usize, dm: []f64, dn: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        for (0..count) |local| {
            try control.check();
            const qi = first + local;
            dm[local] = 0.01 * @as(f64, @floatFromInt((range + 1) * (3 - level) * (qi + 1)));
            for (0..2) |di| dn[local * 2 + di] = 0.02 * @as(f64, @floatFromInt((range + 1) * (3 - level) * (qi + di + 1)));
        }
    }
};

const test_ranges = [_]Range{
    .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 } },
    .{ .start = 12, .rows = 8, .target_rows = .{ 2, 3, 4, 8 } },
};
const test_y = [_]f64{ 0.25, -0.5, 0.75, 0.125 };

fn buildFixture(allocator: std.mem.Allocator, cap: usize, control: solver.Control, fixture: *Fixture) !Representation {
    return build(allocator, &test_ranges, &test_y, 2, 2, 1, 17, cap, .{ .context = fixture, .fill_fn = Fixture.fill }, null, control);
}

fn denseObjective(selected: []const u8, weights: adaptive.Weights) !f64 {
    var fixture: Fixture = .{};
    var maximum: f64 = 0;
    var squared: f64 = 0;
    var reference: f64 = 0;
    for (test_y) |v| reference += v * v;
    var dm: [1]f64 = undefined;
    var dn: [2]f64 = undefined;
    for (0..2) |qi| {
        var aggregate_m: f64 = 0;
        var aggregate_w: [2]f64 = @splat(0);
        for (selected, 0..) |level, ri| if (level < 3) {
            try Fixture.fill(&fixture, ri, level, qi, 1, &dm, &dn, .{});
            aggregate_m += dm[0];
            for (0..2) |di| aggregate_w[di] += dn[di] - dm[0] * test_y[qi * 2 + di];
        };
        maximum = @max(maximum, @abs(aggregate_m));
        const denominator = 1 + aggregate_m;
        for (aggregate_w) |v| squared += v * v / (denominator * denominator);
    }
    return weights.mass * maximum + weights.output * @sqrt(squared / @max(reference, 1e-24));
}

test "per-query Gram agrees with dense W oracle and produces bounded geometry" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    for ([_][2]u8{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 3 }, .{ 3, 3 } }) |selected| {
        try std.testing.expectApproxEqRel(try denseObjective(&selected, .{}), try objective(&rep, &selected, .{}), 256 * std.math.floatEps(f64));
    }
    var geometry = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer geometry.deinit();
    try std.testing.expect(geometry.middle_rows <= geometry.middle_cap_rows);
    try std.testing.expect(geometry.train_harm_after <= geometry.train_harm_before);
}

test "attention_match incremental Gram chooser preserves fixture decisions and bounds owned scratch" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    try std.testing.expectEqual(rep.chooser_bytes + 2 * rep.query_count * @sizeOf(f64), try incrementalChooserBytes(&rep));
    var baseline = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer baseline.deinit();
    var incremental = try chooseIncremental(std.testing.allocator, &rep, 2, .{}, .{});
    defer incremental.deinit();
    try std.testing.expectEqualSlices(adaptive.Level, baseline.levels, incremental.levels);
    try std.testing.expectEqualSlices(usize, baseline.targets, incremental.targets);
    try std.testing.expectEqual(baseline.middle_rows, incremental.middle_rows);
    try std.testing.expectApproxEqRel(baseline.train_harm_before, incremental.train_harm_before, 64 * std.math.floatEps(f64));
    try std.testing.expectApproxEqRel(baseline.train_harm_after, incremental.train_harm_after, 512 * std.math.floatEps(f64));
    try std.testing.expect(!std.mem.eql(u8, &baseline.digest, &incremental.digest));
}

test "incremental refreshed objective is bit identical to full recomputation" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    const mass = try std.testing.allocator.alloc(f64, rep.query_count);
    defer std.testing.allocator.free(mass);
    const norms = try std.testing.allocator.alloc(f64, rep.query_count);
    defer std.testing.allocator.free(norms);
    const weights: adaptive.Weights = .{ .mass = 0.375, .output = 1.625 };
    for ([_][2]u8{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 1 }, .{ 3, 2 }, .{ 3, 3 } }) |selected| {
        try refreshCurrent(&rep, &selected, mass, norms, .{});
        try std.testing.expectEqual(try objective(&rep, &selected, weights), try objectiveFromCurrent(&rep, mass, norms, weights));
    }
}

test "incremental refreshed objective preserves ragged rank-dependent geometry digest" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    // Preserve the gapped source starts while making the two ranges asymmetric.
    rep.ranges[0].target_rows = .{ 1, 3, 5, 8 };
    rep.ranges[1].target_rows = .{ 2, 4, 6, 8 };
    const weights: adaptive.Weights = .{ .mass = 1.75, .output = 0.25 };
    for ([_]usize{ 3, 2 }) |divisor| {
        var reference = try chooseIncrementalWithBudgetPercentMode(std.testing.allocator, &rep, divisor, weights, 100, .{}, false);
        defer reference.deinit();
        var refreshed = try chooseIncrementalWithBudgetPercentMode(std.testing.allocator, &rep, divisor, weights, 100, .{}, true);
        defer refreshed.deinit();
        try std.testing.expectEqualSlices(adaptive.Level, reference.levels, refreshed.levels);
        try std.testing.expectEqualSlices(usize, reference.targets, refreshed.targets);
        try std.testing.expectEqual(reference.middle_rows, refreshed.middle_rows);
        try std.testing.expectEqual(reference.middle_cap_rows, refreshed.middle_cap_rows);
        try std.testing.expectEqual(reference.train_harm_before, refreshed.train_harm_before);
        try std.testing.expectEqual(reference.train_harm_after, refreshed.train_harm_after);
        try std.testing.expectEqualSlices(u8, &reference.digest, &refreshed.digest);
    }
}

fn incrementalAllocationFailure(allocator: std.mem.Allocator, rep: *const Representation) !void {
    var geometry = try chooseIncremental(allocator, rep, 2, .{}, .{});
    geometry.deinit();
}

fn incrementalBudget95AllocationFailure(allocator: std.mem.Allocator, rep: *const Representation) !void {
    var geometry = try chooseIncrementalWithBudgetPercent(allocator, rep, 2, .{}, 95, .{});
    geometry.deinit();
}

test "attention_match incremental Gram chooser cancellation and allocation unwind" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, chooseIncremental(std.testing.allocator, &rep, 2, .{}, cancel.control()));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, incrementalAllocationFailure, .{&rep});
}

test "context incremental Gram budget percent is divisor2-only and canonical" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var legacy = try chooseIncremental(std.testing.allocator, &rep, 2, .{}, .{});
    defer legacy.deinit();
    var budget95 = try chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, .{});
    defer budget95.deinit();
    try std.testing.expectEqual((std.math.divCeil(usize, 16, 2) catch unreachable) * 95 / 100, budget95.middle_cap_rows);
    try std.testing.expect(budget95.middle_rows <= budget95.middle_cap_rows);
    try std.testing.expect(!std.mem.eql(u8, &legacy.digest, &budget95.digest));
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, cancel.control()));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, incrementalBudget95AllocationFailure, .{&rep});
    try std.testing.expectError(error.InvalidFitConfig, chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 3, .{}, 95, .{}));
    try std.testing.expectError(error.InvalidFitConfig, chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 49, .{}));

    const saved0 = rep.ranges[0].target_rows[0];
    const saved1 = rep.ranges[1].target_rows[0];
    rep.ranges[0].target_rows[0] = 4;
    rep.ranges[1].target_rows[0] = 4;
    defer rep.ranges[0].target_rows[0] = saved0;
    defer rep.ranges[1].target_rows[0] = saved1;
    try std.testing.expectError(error.GeometryBudgetInfeasible, chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, .{}));
}

test "incremental geometry reuses one representation from p91 to p95 with zero-cost levels" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    rep.ranges[0].target_rows = .{ 2, 2, 4, 8 };
    rep.ranges[1].target_rows = .{ 2, 3, 3, 8 };
    var primary = try chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 91, .{});
    defer primary.deinit();
    var fallback = try chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, .{});
    defer fallback.deinit();
    try std.testing.expect(primary.middle_rows <= primary.middle_cap_rows);
    try std.testing.expect(fallback.middle_rows <= fallback.middle_cap_rows);
    try std.testing.expect(fallback.middle_cap_rows >= primary.middle_cap_rows);
    try std.testing.expect(!std.mem.eql(u8, &primary.digest, &fallback.digest));
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

test "per-query Gram cancellation unwinds before publication" {
    var fixture: Fixture = .{};
    var cancel: Cancel = .{ .cancel_at = 5 };
    try std.testing.expectError(error.Cancelled, buildFixture(std.testing.allocator, 1024 * 1024, cancel.control(), &fixture));
}

test "per-query Gram chooser cancellation publishes no geometry" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, choose(std.testing.allocator, &rep, 2, .{}, cancel.control()));
}

test "per-query Gram cap rejects before accessor or allocation" {
    var fixture: Fixture = .{};
    try std.testing.expectError(error.CompactionWorkspaceLimit, buildFixture(std.testing.allocator, 18, .{}, &fixture));
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
}

fn allocationFixture(allocator: std.mem.Allocator) !void {
    var fixture: Fixture = .{};
    var rep = try buildFixture(allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
}

test "per-query Gram construction unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFixture, .{});
}

test "per-query Gram fails closed beyond bounded candidate count" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, workspaceBytes(4096, 512, max_candidates / 3 + 1, 8, 0));
}

test "per-query Gram rejects invalid identity and weight policy" {
    var fixture: Fixture = .{};
    var bad = test_ranges;
    bad[0].target_rows[3] = 7;
    try std.testing.expectError(error.InvalidFitConfig, build(std.testing.allocator, &bad, &test_y, 2, 2, 1, 0, 1024 * 1024, .{ .context = &fixture, .fill_fn = Fixture.fill }, null, .{}));
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    try std.testing.expectError(error.InvalidFitConfig, choose(std.testing.allocator, &rep, 2, .{ .mass = -1, .output = 1 }, .{}));
}

test "per-query Gram rejects a materially negative squared norm" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    rep.gram[packedIndex(0, 0)] = -1;
    try std.testing.expectError(error.NonFiniteAttention, objective(&rep, &.{ 0, 3 }, .{}));
}

test "per-query Gram chooser jumps over a harmful intermediate level" {
    const Local = struct {
        fn fill(_: ?*anyopaque, _: usize, level: usize, _: usize, _: usize, dm: []f64, dn: []f64, _: solver.Control) !void {
            dm[0] = ([_]f64{ 0.1, 0.2, 0 })[level];
            dn[0] = 0;
        }
    };
    const ranges = [_]Range{.{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 4, 8 } }};
    var rep = try build(std.testing.allocator, &ranges, &.{0}, 1, 1, 1, 0, 1024 * 1024, .{ .context = null, .fill_fn = Local.fill }, null, .{});
    defer rep.deinit();
    var geometry = try choose(std.testing.allocator, &rep, 2, .{ .mass = 1, .output = 0 }, .{});
    defer geometry.deinit();
    try std.testing.expectEqualSlices(usize, &.{4}, geometry.targets);
}

test "per-query Gram budget exchange escapes a cap-locked TRAIN mass trap" {
    const Local = struct {
        fn fill(_: ?*anyopaque, range: usize, level: usize, _: usize, _: usize, dm: []f64, dn: []f64, _: solver.Control) !void {
            dm[0] = ([_][3]f64{ .{ 0.03, 0.02, 0.02 }, .{ 0.15, 0.15, 0 } })[range][level];
            dn[0] = 0;
        }
    };
    const ranges = [_]Range{
        .{ .start = 0, .rows = 8, .target_rows = .{ 2, 3, 6, 8 } },
        .{ .start = 8, .rows = 8, .target_rows = .{ 2, 3, 6, 8 } },
    };
    var rep = try build(std.testing.allocator, &ranges, &.{1}, 1, 1, 1, 0, 1024 * 1024, .{ .context = null, .fill_fn = Local.fill }, null, .{});
    defer rep.deinit();
    var base = adaptive.Geometry{ .allocator = std.testing.allocator, .levels = try std.testing.allocator.dupe(adaptive.Level, &.{ .divisor2, .divisor4 }), .targets = try std.testing.allocator.dupe(usize, &.{ 6, 2 }), .middle_cap_rows = 8, .middle_rows = 8, .preserved_ranges = 0, .fitted_ranges = 2, .train_harm_before = 0.17, .train_harm_after = 0.17, .digest = @splat(9) };
    defer base.deinit();
    var refined = try refineBudgetExchange(std.testing.allocator, &rep, &base, .{ .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 }, .{ .mass = 1, .output = 0 }, .{});
    defer refined.deinit();
    try std.testing.expectEqualSlices(adaptive.Level, &.{ .divisor4, .divisor2 }, refined.levels);
    try std.testing.expect(refined.middle_rows <= refined.middle_cap_rows);
    try std.testing.expect(refined.train_harm_after < 0.05);
}

test "per-query Gram budget exchange rejects cancellation and invalid limits" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var base = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer base.deinit();
    var cancel: Cancel = .{ .cancel_at = 1 };
    try std.testing.expectError(error.Cancelled, refineBudgetExchange(std.testing.allocator, &rep, &base, .{ .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 }, .{}, cancel.control()));
    try std.testing.expectError(error.InvalidFitConfig, refineBudgetExchange(std.testing.allocator, &rep, &base, .{ .max_relative_mass_error = 0, .max_relative_output_rmse = 0.05 }, .{}, .{}));
}

fn exchangeAllocationFixture(allocator: std.mem.Allocator) !void {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var base = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer base.deinit();
    var refined = try refineBudgetExchange(allocator, &rep, &base, .{ .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 }, .{}, .{});
    defer refined.deinit();
}

test "per-query Gram budget exchange unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exchangeAllocationFixture, .{});
}

pub fn workspaceBytes(q: usize, dim: usize, range_count: usize, tile_rows: usize, backend_scratch: usize) !usize {
    if (q == 0 or dim == 0 or range_count == 0 or tile_rows == 0) return error.InvalidFitConfig;
    const k = try mul(range_count, 3);
    if (k > max_candidates) return error.CompactionWorkspaceLimit;
    const t = @min(q, tile_rows);
    const persistent_values = try mul(try add(try mul(q, k), try mul(q, try tri(k))), @sizeOf(f64));
    const persistent = try add(persistent_values, try mul(range_count, @sizeOf(Range)));
    const w_tile = try mul(try mul(try mul(t, k), dim), @sizeOf(f64));
    const accessor = try mul(try add(t, try mul(t, dim)), @sizeOf(f64));
    const packed_output = try mul(try tri(k), @sizeOf(f64));
    const build_peak = try add(persistent, try add(w_tile, try add(accessor, try add(packed_output, backend_scratch))));
    const chooser = try mul(range_count, @sizeOf(u8) + @sizeOf(adaptive.Level) + @sizeOf(usize));
    return @max(build_peak, try add(persistent, chooser));
}

fn scalarGram(_: ?*anyopaque, matrix: []const f64, k: usize, dim: usize, output: []f64, _: []u8, control: solver.Control) !void {
    if (matrix.len != try mul(k, dim) or output.len != try tri(k)) return error.InvalidAttentionShape;
    for (0..k) |i| for (0..i + 1) |j| {
        try control.check();
        var dot: f64 = 0;
        for (matrix[i * dim ..][0..dim], matrix[j * dim ..][0..dim]) |a, b| dot += a * b;
        if (!std.math.isFinite(dot)) return error.NonFiniteAttention;
        output[packedIndex(i, j)] = dot;
    };
}

pub fn build(allocator: std.mem.Allocator, ranges: []const Range, original_y: []const f64, q: usize, dim: usize, tile_rows: usize, external_live_bytes: usize, combined_cap: usize, accessor: DeltaAccessor, backend: ?GramBackend, control: solver.Control) !Representation {
    try control.check();
    const qd = try mul(q, dim);
    if (ranges.len == 0 or original_y.len != qd or external_live_bytes > combined_cap) return error.InvalidFitConfig;
    const k = try mul(ranges.len, 3);
    if (k > max_candidates) return error.CompactionWorkspaceLimit;
    var prior_end: usize = 0;
    for (ranges, 0..) |range, i| {
        if (range.rows == 0 or (i != 0 and range.start < prior_end)) return error.InvalidAttentionShape;
        for (range.target_rows, 0..) |count, level| {
            if (count == 0 or count > range.rows or (level != 0 and count < range.target_rows[level - 1])) return error.InvalidFitConfig;
        }
        if (range.target_rows[3] != range.rows) return error.InvalidFitConfig;
        prior_end = try add(range.start, range.rows);
    }
    // The scalar implementation exists for pure tests and small diagnostics.
    // Production-sized Q=4096 construction must provide a bounded BLAS/Metal
    // backend rather than accidentally execute O(Q*K*K*D) scalar work.
    if (backend == null and try mul(try mul(q, k), try mul(k, dim)) > 16 * 1024 * 1024) return error.UnsupportedOperation;
    const backend_scratch = if (backend) |b| try b.workspace_fn(b.context, k, dim) else 0;
    const owned_peak = try workspaceBytes(q, dim, ranges.len, tile_rows, backend_scratch);
    if (try add(external_live_bytes, owned_peak) > combined_cap) return error.CompactionWorkspaceLimit;
    const owned_ranges = try allocator.dupe(Range, ranges);
    errdefer allocator.free(owned_ranges);
    const mass = try allocator.alloc(f64, try mul(q, k));
    errdefer allocator.free(mass);
    const triangle = try tri(k);
    const gram = try allocator.alloc(f64, try mul(q, triangle));
    errdefer allocator.free(gram);
    var original_norm_squared: f64 = 0;
    for (original_y) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteAttention;
        original_norm_squared += value * value;
        if (!std.math.isFinite(original_norm_squared)) return error.NonFiniteAttention;
    }
    const take_max = @min(q, tile_rows);
    const w = try allocator.alloc(f64, try mul(try mul(take_max, k), dim));
    defer allocator.free(w);
    const delta_m = try allocator.alloc(f64, take_max);
    defer allocator.free(delta_m);
    const delta_n = try allocator.alloc(f64, try mul(take_max, dim));
    defer allocator.free(delta_n);
    const packed_gram = try allocator.alloc(f64, triangle);
    defer allocator.free(packed_gram);
    const scratch = try allocator.alloc(u8, backend_scratch);
    defer allocator.free(scratch);
    var first: usize = 0;
    while (first < q) : (first += take_max) {
        try control.check();
        const take = @min(take_max, q - first);
        for (0..k) |candidate| {
            const ri = candidate / 3;
            const li = candidate % 3;
            @memset(delta_m[0..take], 0);
            @memset(delta_n[0 .. take * dim], 0);
            try accessor.fill_fn(accessor.context, ri, li, first, take, delta_m[0..take], delta_n[0 .. take * dim], control);
            for (0..take) |qi| {
                const m = delta_m[qi];
                if (!std.math.isFinite(m)) return error.NonFiniteAttention;
                mass[(first + qi) * k + candidate] = m;
                for (0..dim) |di| {
                    const value = delta_n[qi * dim + di] - m * original_y[(first + qi) * dim + di];
                    if (!std.math.isFinite(value)) return error.NonFiniteAttention;
                    w[(qi * k + candidate) * dim + di] = value;
                }
            }
        }
        for (0..take) |qi| {
            try control.check();
            const matrix = w[qi * k * dim ..][0 .. k * dim];
            if (backend) |b| try b.compute_fn(b.context, matrix, k, dim, packed_gram, scratch, control) else try scalarGram(null, matrix, k, dim, packed_gram, scratch, control);
            for (packed_gram) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            @memcpy(gram[(first + qi) * triangle ..][0..triangle], packed_gram);
        }
    }
    try control.check();
    var digest: [32]u8 = undefined;
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(policy);
    hasher.update(std.mem.sliceAsBytes(owned_ranges));
    hasher.update(std.mem.sliceAsBytes(mass));
    hasher.update(std.mem.sliceAsBytes(gram));
    hasher.final(&digest);
    const chooser_bytes = try mul(ranges.len, @sizeOf(u8) + @sizeOf(adaptive.Level) + @sizeOf(usize));
    return .{ .allocator = allocator, .query_count = q, .dim = dim, .ranges = owned_ranges, .mass = mass, .gram = gram, .original_y_norm_squared = original_norm_squared, .digest = digest, .peak_bytes = owned_peak, .chooser_bytes = chooser_bytes };
}

fn normFor(rep: *const Representation, qi: usize, selected: []const u8) !f64 {
    const k = rep.ranges.len * 3;
    const triangle = try tri(k);
    const g = rep.gram[qi * triangle ..][0..triangle];
    var value: f64 = 0;
    var absolute_sum: f64 = 0;
    for (selected, 0..) |level_i, ri| if (level_i < 3) {
        const i = ri * 3 + level_i;
        const d = g[packedIndex(i, i)];
        if (!std.math.isFinite(d) or d < 0) return error.NonFiniteAttention;
        absolute_sum += @abs(d);
        value += d;
        for (selected[0..ri], 0..) |level_j, rj| if (level_j < 3) {
            const pair = 2 * g[packedIndex(i, rj * 3 + level_j)];
            absolute_sum += @abs(pair);
            value += pair;
        };
    };
    const bound = 64 * std.math.floatEps(f64) * @max(absolute_sum, 1e-300);
    if (value < -bound or !std.math.isFinite(value)) return error.NonFiniteAttention;
    return @max(value, 0);
}

fn objective(rep: *const Representation, selected: []const u8, weights: adaptive.Weights) !f64 {
    if (selected.len != rep.ranges.len) return error.InvalidFitConfig;
    if (!std.math.isFinite(weights.mass) or !std.math.isFinite(weights.output) or weights.mass < 0 or weights.output < 0 or (weights.mass == 0 and weights.output == 0)) return error.InvalidFitConfig;
    const k = rep.ranges.len * 3;
    var maximum: f64 = 0;
    var squared: f64 = 0;
    for (0..rep.query_count) |qi| {
        var m: f64 = 0;
        for (selected, 0..) |level, ri| {
            if (level < 3) m += rep.mass[qi * k + ri * 3 + level];
        }
        const denominator = 1 + m;
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(m));
        squared += try normFor(rep, qi, selected) / (denominator * denominator);
    }
    const output = @sqrt(squared / @max(rep.original_y_norm_squared, 1e-24));
    const result = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(result)) return error.NonFiniteAttention;
    return result;
}

fn objectiveFromCurrent(rep: *const Representation, current_mass: []const f64, current_norm: []const f64, weights: adaptive.Weights) !f64 {
    if (current_mass.len != rep.query_count or current_norm.len != rep.query_count) return error.InvalidFitConfig;
    if (!std.math.isFinite(weights.mass) or !std.math.isFinite(weights.output) or weights.mass < 0 or weights.output < 0 or (weights.mass == 0 and weights.output == 0)) return error.InvalidFitConfig;
    var maximum: f64 = 0;
    var squared: f64 = 0;
    for (current_mass, current_norm) |mass, norm| {
        const denominator = 1 + mass;
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(mass));
        squared += norm / (denominator * denominator);
    }
    const output = @sqrt(squared / @max(rep.original_y_norm_squared, 1e-24));
    const result = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(result)) return error.NonFiniteAttention;
    return result;
}

fn replacementNorm(rep: *const Representation, qi: usize, selected: []const u8, changed_range: usize, next_level: u8, current_norm: f64) !f64 {
    const k = rep.ranges.len * 3;
    const triangle = try tri(k);
    const g = rep.gram[qi * triangle ..][0..triangle];
    const old_level = selected[changed_range];
    var value = current_norm;
    var absolute_sum = @abs(current_norm);
    if (old_level < 3) {
        const old = changed_range * 3 + old_level;
        const self = g[packedIndex(old, old)];
        value -= self;
        absolute_sum += @abs(self);
        for (selected, 0..) |level, ri| if (ri != changed_range and level < 3) {
            const pair = 2 * g[packedIndex(old, ri * 3 + level)];
            value -= pair;
            absolute_sum += @abs(pair);
        };
    }
    if (next_level < 3) {
        const next = changed_range * 3 + next_level;
        const self = g[packedIndex(next, next)];
        value += self;
        absolute_sum += @abs(self);
        for (selected, 0..) |level, ri| if (ri != changed_range and level < 3) {
            const pair = 2 * g[packedIndex(next, ri * 3 + level)];
            value += pair;
            absolute_sum += @abs(pair);
        };
    }
    const bound = 128 * std.math.floatEps(f64) * @max(absolute_sum, 1e-300);
    if (value < -bound or !std.math.isFinite(value)) return error.NonFiniteAttention;
    return @max(value, 0);
}

fn replacementObjective(rep: *const Representation, selected: []const u8, changed_range: usize, next_level: u8, current_mass: []const f64, current_norm: []const f64, weights: adaptive.Weights) !f64 {
    if (current_mass.len != rep.query_count or current_norm.len != rep.query_count) return error.InvalidFitConfig;
    const k = rep.ranges.len * 3;
    const old_level = selected[changed_range];
    var maximum: f64 = 0;
    var squared: f64 = 0;
    for (0..rep.query_count) |qi| {
        var mass = current_mass[qi];
        if (old_level < 3) mass -= rep.mass[qi * k + changed_range * 3 + old_level];
        if (next_level < 3) mass += rep.mass[qi * k + changed_range * 3 + next_level];
        const denominator = 1 + mass;
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(mass));
        squared += try replacementNorm(rep, qi, selected, changed_range, next_level, current_norm[qi]) / (denominator * denominator);
    }
    const output = @sqrt(squared / @max(rep.original_y_norm_squared, 1e-24));
    const result = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(result)) return error.NonFiniteAttention;
    return result;
}

fn refreshCurrent(rep: *const Representation, selected: []const u8, mass: []f64, norms: []f64, control: solver.Control) !void {
    const k = rep.ranges.len * 3;
    for (0..rep.query_count) |qi| {
        if (qi % 64 == 0) try control.check();
        var value: f64 = 0;
        for (selected, 0..) |level, ri| {
            if (level < 3) value += rep.mass[qi * k + ri * 3 + level];
        }
        mass[qi] = value;
        norms[qi] = try normFor(rep, qi, selected);
    }
}

pub fn incrementalChooserBytes(rep: *const Representation) !usize {
    return add(rep.chooser_bytes, try mul(try mul(rep.query_count, 2), @sizeOf(f64)));
}

/// Opt-in exact-Gram candidate evaluator. It replaces a full normFor scan for
/// every trial with one O(range_count) Gram update per query. Committed states
/// are refreshed through the original normFor order. Candidate regrouping has
/// its own policy and must not be credited as bit-exact with `policy`.
pub fn chooseIncremental(allocator: std.mem.Allocator, rep: *const Representation, divisor: usize, weights: adaptive.Weights, control: solver.Control) !adaptive.Geometry {
    return chooseIncrementalWithBudgetPercent(allocator, rep, divisor, weights, 100, control);
}

pub fn chooseIncrementalWithBudgetPercent(allocator: std.mem.Allocator, rep: *const Representation, divisor: usize, weights: adaptive.Weights, budget_percent: u8, control: solver.Control) !adaptive.Geometry {
    return chooseIncrementalWithBudgetPercentMode(allocator, rep, divisor, weights, budget_percent, control, true);
}

fn chooseIncrementalWithBudgetPercentMode(allocator: std.mem.Allocator, rep: *const Representation, divisor: usize, weights: adaptive.Weights, budget_percent: u8, control: solver.Control, reuse_refreshed_objective: bool) !adaptive.Geometry {
    try control.check();
    if ((divisor != 4 and divisor != 3 and divisor != 2) or !validBudgetPercent(budget_percent) or (budget_percent != 100 and divisor != 2)) return error.InvalidFitConfig;
    const selected = try allocator.alloc(u8, rep.ranges.len);
    defer allocator.free(selected);
    @memset(selected, 0);
    const levels = try allocator.alloc(adaptive.Level, rep.ranges.len);
    errdefer allocator.free(levels);
    @memset(levels, .divisor4);
    const targets = try allocator.alloc(usize, rep.ranges.len);
    errdefer allocator.free(targets);
    const current_mass = try allocator.alloc(f64, rep.query_count);
    defer allocator.free(current_mass);
    const current_norm = try allocator.alloc(f64, rep.query_count);
    defer allocator.free(current_norm);
    var eligible: usize = 0;
    var rows: usize = 0;
    for (rep.ranges, targets) |range, *out| {
        out.* = range.target_rows[0];
        eligible = try add(eligible, range.rows);
        rows = try add(rows, out.*);
    }
    const nominal_cap = std.math.divCeil(usize, eligible, divisor) catch return error.InvalidAttentionShape;
    const cap = if (budget_percent == 100) nominal_cap else try mul(nominal_cap, budget_percent) / 100;
    if (rows > cap) return error.GeometryBudgetInfeasible;
    try refreshCurrent(rep, selected, current_mass, current_norm, control);
    const before = try objective(rep, selected, weights);
    var current = before;
    while (true) {
        try control.check();
        var best_range: ?usize = null;
        var best_level: u8 = 0;
        var best_gain: f64 = 0;
        var best_cost: usize = 1;
        for (rep.ranges, selected, targets, 0..) |range, level, current_rows, ri| for (@as(usize, level) + 1..4) |next| {
            try control.check();
            const cost = range.target_rows[next] - current_rows;
            if (cost == 0 or try add(rows, cost) > cap) continue;
            const value = try replacementObjective(rep, selected, ri, @intCast(next), current_mass, current_norm, weights);
            const gain = current - value;
            if (gain <= 0) continue;
            if (best_range == null or gain * @as(f64, @floatFromInt(best_cost)) > best_gain * @as(f64, @floatFromInt(cost)) or
                (gain * @as(f64, @floatFromInt(best_cost)) == best_gain * @as(f64, @floatFromInt(cost)) and (range.start < rep.ranges[best_range.?].start or (range.start == rep.ranges[best_range.?].start and next < best_level))))
            {
                best_range = ri;
                best_level = @intCast(next);
                best_gain = gain;
                best_cost = cost;
            }
        };
        const ri = best_range orelse break;
        selected[ri] = best_level;
        levels[ri] = @fromBackingInt(std.math.cast(u2, best_level) orelse return error.InvalidFitConfig);
        targets[ri] = rep.ranges[ri].target_rows[best_level];
        rows += best_cost;
        try refreshCurrent(rep, selected, current_mass, current_norm, control);
        current = if (reuse_refreshed_objective)
            try objectiveFromCurrent(rep, current_mass, current_norm, weights)
        else
            try objective(rep, selected, weights);
    }
    var preserved: usize = 0;
    for (selected) |level| preserved += @intFromBool(level == 3);
    var digest: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(if (budget_percent == 100) incremental_policy else incremental_budget_percent_policy);
    h.update(&rep.digest);
    for (levels) |level| h.update(&.{@backingInt(level)});
    for (targets) |target_rows| hashU64Le(&h, @intCast(target_rows));
    if (budget_percent != 100) h.update(&.{budget_percent});
    h.final(&digest);
    return .{ .allocator = allocator, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = rows, .preserved_ranges = preserved, .fitted_ranges = rep.ranges.len - preserved, .train_harm_before = before, .train_harm_after = current, .digest = digest };
}

const ExchangeMetrics = struct { mass: f64, output: f64, objective: f64 };

fn exchangeMetrics(rep: *const Representation, selected: []const u8, weights: adaptive.Weights) !ExchangeMetrics {
    if (selected.len != rep.ranges.len) return error.InvalidFitConfig;
    const k = rep.ranges.len * 3;
    var maximum: f64 = 0;
    var squared: f64 = 0;
    for (0..rep.query_count) |qi| {
        var m: f64 = 0;
        for (selected, 0..) |level, ri| {
            if (level < 3) m += rep.mass[qi * k + ri * 3 + level];
        }
        const denominator = 1 + m;
        if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
        maximum = @max(maximum, @abs(m));
        squared += try normFor(rep, qi, selected) / (denominator * denominator);
    }
    const output = @sqrt(squared / @max(rep.original_y_norm_squared, 1e-24));
    const combined = weights.mass * maximum + weights.output * output;
    if (!std.math.isFinite(maximum) or !std.math.isFinite(output) or !std.math.isFinite(combined)) return error.NonFiniteAttention;
    return .{ .mass = maximum, .output = output, .objective = combined };
}

fn exchangeBetter(a: ExchangeMetrics, b: ExchangeMetrics, cfg: ExchangeConfig) bool {
    const am = @max(a.mass - cfg.max_relative_mass_error, 0) / cfg.max_relative_mass_error;
    const bm = @max(b.mass - cfg.max_relative_mass_error, 0) / cfg.max_relative_mass_error;
    const ao = @max(a.output - cfg.max_relative_output_rmse, 0) / cfg.max_relative_output_rmse;
    const bo = @max(b.output - cfg.max_relative_output_rmse, 0) / cfg.max_relative_output_rmse;
    return am < bm or (am == bm and (ao < bo or (ao == bo and a.objective < b.objective)));
}

const ExchangeMove = struct { range: usize, level: u8, freed: usize, metrics: ExchangeMetrics };
const ExchangePair = struct { down: ExchangeMove, up_range: usize, up_level: u8, rows: usize, metrics: ExchangeMetrics };

fn moveBefore(a: ExchangeMove, b: ExchangeMove, rep: *const Representation, cfg: ExchangeConfig) bool {
    if (exchangeBetter(a.metrics, b.metrics, cfg)) return true;
    if (exchangeBetter(b.metrics, a.metrics, cfg)) return false;
    if (a.freed != b.freed) return a.freed > b.freed;
    if (rep.ranges[a.range].start != rep.ranges[b.range].start) return rep.ranges[a.range].start < rep.ranges[b.range].start;
    return a.level < b.level;
}

fn insertMove(frontier: *[exchange_frontier_limit]ExchangeMove, count: *usize, move: ExchangeMove, rep: *const Representation, cfg: ExchangeConfig) void {
    var at: usize = 0;
    while (at < count.* and !moveBefore(move, frontier[at], rep, cfg)) : (at += 1) {}
    if (at >= exchange_frontier_limit) return;
    const next = @min(count.* + 1, exchange_frontier_limit);
    var i = next;
    while (i > at + 1) : (i -= 1) frontier[i - 1] = frontier[i - 2];
    frontier[at] = move;
    count.* = next;
}

/// DEV-only refinement of an already-built greedy geometry. It consumes only
/// the TRAIN Gram representation; heldout data cannot enter this interface.
pub fn refineBudgetExchange(allocator: std.mem.Allocator, rep: *const Representation, base: *const adaptive.Geometry, cfg: ExchangeConfig, weights: adaptive.Weights, control: solver.Control) !adaptive.Geometry {
    try control.check();
    if (!std.math.isFinite(cfg.max_relative_mass_error) or cfg.max_relative_mass_error <= 0 or
        !std.math.isFinite(cfg.max_relative_output_rmse) or cfg.max_relative_output_rmse <= 0 or
        base.levels.len != rep.ranges.len or base.targets.len != rep.ranges.len) return error.InvalidFitConfig;
    const selected = try allocator.alloc(u8, rep.ranges.len);
    defer allocator.free(selected);
    const levels = try allocator.dupe(adaptive.Level, base.levels);
    errdefer allocator.free(levels);
    const targets = try allocator.dupe(usize, base.targets);
    errdefer allocator.free(targets);
    var rows: usize = 0;
    for (rep.ranges, levels, targets, selected) |range, level, target_rows, *out| {
        const raw: u8 = @backingInt(level);
        if (target_rows != range.target_rows[raw]) return error.InvalidFitConfig;
        out.* = raw;
        rows = try add(rows, target_rows);
    }
    if (rows > base.middle_cap_rows) return error.InvalidFitConfig;
    var current = try exchangeMetrics(rep, selected, weights);
    for (0..exchange_maximum_sweeps) |_| {
        try control.check();
        var frontier: [exchange_frontier_limit]ExchangeMove = undefined;
        var frontier_count: usize = 0;
        for (rep.ranges, selected, targets, 0..) |range, level, target_rows, ri| for (0..level) |down| {
            try control.check();
            selected[ri] = @intCast(down);
            const candidate = try exchangeMetrics(rep, selected, weights);
            selected[ri] = level;
            insertMove(&frontier, &frontier_count, .{ .range = ri, .level = @intCast(down), .freed = target_rows - range.target_rows[down], .metrics = candidate }, rep, cfg);
        };
        var best: ?ExchangePair = null;
        for (frontier[0..frontier_count]) |down| for (rep.ranges, selected, targets, 0..) |range, level, target_rows, up_ri| {
            const effective_level = if (up_ri == down.range) down.level else level;
            const effective_rows = if (up_ri == down.range) range.target_rows[down.level] else target_rows;
            for (@as(usize, effective_level) + 1..4) |up| {
                try control.check();
                const proposed = rows - down.freed + (range.target_rows[up] - effective_rows);
                if (proposed > base.middle_cap_rows) continue;
                const saved_down = selected[down.range];
                const saved_up = selected[up_ri];
                selected[down.range] = down.level;
                selected[up_ri] = @intCast(up);
                const candidate_metrics = try exchangeMetrics(rep, selected, weights);
                selected[down.range] = saved_down;
                selected[up_ri] = saved_up;
                const candidate: ExchangePair = .{ .down = down, .up_range = up_ri, .up_level = @intCast(up), .rows = proposed, .metrics = candidate_metrics };
                if (!exchangeBetter(candidate.metrics, current, cfg)) continue;
                if (best == null or exchangeBetter(candidate.metrics, best.?.metrics, cfg) or
                    (!exchangeBetter(best.?.metrics, candidate.metrics, cfg) and
                        (rep.ranges[candidate.down.range].start < rep.ranges[best.?.down.range].start or
                            (rep.ranges[candidate.down.range].start == rep.ranges[best.?.down.range].start and rep.ranges[candidate.up_range].start < rep.ranges[best.?.up_range].start)))) best = candidate;
            }
        };
        const chosen = best orelse break;
        selected[chosen.down.range] = chosen.down.level;
        levels[chosen.down.range] = @fromBackingInt(@intCast(chosen.down.level));
        targets[chosen.down.range] = rep.ranges[chosen.down.range].target_rows[chosen.down.level];
        selected[chosen.up_range] = chosen.up_level;
        levels[chosen.up_range] = @fromBackingInt(@intCast(chosen.up_level));
        targets[chosen.up_range] = rep.ranges[chosen.up_range].target_rows[chosen.up_level];
        rows = chosen.rows;
        current = chosen.metrics;
    }
    try control.check();
    var preserved: usize = 0;
    for (selected) |level| preserved += @intFromBool(level == 3);
    var digest: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(exchange_policy);
    h.update(&base.digest);
    for (levels) |level| h.update(&.{@backingInt(level)});
    for (targets) |target_rows| hashU64Le(&h, std.math.cast(u64, target_rows) orelse return error.InvalidAttentionShape);
    hashU64Le(&h, @bitCast(cfg.max_relative_mass_error));
    hashU64Le(&h, @bitCast(cfg.max_relative_output_rmse));
    h.final(&digest);
    return .{ .allocator = allocator, .levels = levels, .targets = targets, .middle_cap_rows = base.middle_cap_rows, .middle_rows = rows, .preserved_ranges = preserved, .fitted_ranges = rep.ranges.len - preserved, .train_harm_before = base.train_harm_before, .train_harm_after = current.objective, .digest = digest };
}

pub const budget95_policy = "greedy_positive_joint_train_objective_per_row_divisor2_budget95_v1";
pub const budget_percent_policy = "greedy_positive_joint_train_objective_per_row_divisor2_budget_percent_v1";

pub fn validBudgetPercent(percent: u8) bool {
    return percent == 100 or percent == 95 or (percent >= 50 and percent <= 94);
}

pub fn policyForBudgetPercent(percent: u8) ?[]const u8 {
    if (percent == 100) return policy;
    if (percent == 95) return budget95_policy;
    if (percent >= 50 and percent <= 94) return budget_percent_policy;
    return null;
}

pub fn choose(allocator: std.mem.Allocator, rep: *const Representation, divisor: usize, weights: adaptive.Weights, control: solver.Control) !adaptive.Geometry {
    return chooseWithBudgetPercent(allocator, rep, divisor, weights, 100, control);
}

pub fn chooseWithBudgetPercent(allocator: std.mem.Allocator, rep: *const Representation, divisor: usize, weights: adaptive.Weights, budget_percent: u8, control: solver.Control) !adaptive.Geometry {
    try control.check();
    if ((divisor != 4 and divisor != 3 and divisor != 2) or !validBudgetPercent(budget_percent) or (budget_percent != 100 and divisor != 2)) return error.InvalidFitConfig;
    const required_chooser = try mul(rep.ranges.len, @sizeOf(u8) + @sizeOf(adaptive.Level) + @sizeOf(usize));
    if (required_chooser != rep.chooser_bytes) return error.InvalidAttentionShape;
    const selected = try allocator.alloc(u8, rep.ranges.len);
    defer allocator.free(selected);
    @memset(selected, 0);
    const levels = try allocator.alloc(adaptive.Level, rep.ranges.len);
    errdefer allocator.free(levels);
    @memset(levels, .divisor4);
    const targets = try allocator.alloc(usize, rep.ranges.len);
    errdefer allocator.free(targets);
    var eligible: usize = 0;
    var rows: usize = 0;
    for (rep.ranges, targets, 0..) |range, *out, i| {
        if (range.rows == 0 or (i != 0 and range.start < rep.ranges[i - 1].start + rep.ranges[i - 1].rows)) return error.InvalidAttentionShape;
        out.* = range.target_rows[0];
        eligible = try add(eligible, range.rows);
        rows = try add(rows, out.*);
    }
    const nominal_cap = std.math.divCeil(usize, eligible, divisor) catch return error.InvalidAttentionShape;
    const cap = if (budget_percent == 100) nominal_cap else try mul(nominal_cap, budget_percent) / 100;
    if (rows > cap) return error.GeometryBudgetInfeasible;
    const before = try objective(rep, selected, weights);
    var current = before;
    while (true) {
        try control.check();
        var best_range: ?usize = null;
        var best_level: u8 = 0;
        var best_value: f64 = 0;
        var best_gain: f64 = 0;
        var best_cost: usize = 1;
        for (rep.ranges, selected, targets, 0..) |range, level, current_rows, ri| for (@as(usize, level) + 1..4) |next| {
            try control.check();
            const cost = range.target_rows[next] - current_rows;
            const proposed_rows = try add(rows, cost);
            if (cost == 0 or proposed_rows > cap) continue;
            selected[ri] = @intCast(next);
            const value = try objective(rep, selected, weights);
            selected[ri] = level;
            const gain = current - value;
            if (gain <= 0) continue;
            if (best_range == null or gain * @as(f64, @floatFromInt(best_cost)) > best_gain * @as(f64, @floatFromInt(cost)) or
                (gain * @as(f64, @floatFromInt(best_cost)) == best_gain * @as(f64, @floatFromInt(cost)) and (range.start < rep.ranges[best_range.?].start or (range.start == rep.ranges[best_range.?].start and next < best_level))))
            {
                best_range = ri;
                best_level = @intCast(next);
                best_value = value;
                best_gain = gain;
                best_cost = cost;
            }
        };
        const ri = best_range orelse break;
        selected[ri] = best_level;
        levels[ri] = @fromBackingInt(@intCast(best_level));
        targets[ri] = rep.ranges[ri].target_rows[best_level];
        rows += best_cost;
        current = best_value;
    }
    try control.check();
    var preserved: usize = 0;
    for (selected) |level| preserved += @intFromBool(level == 3);
    var digest: [32]u8 = undefined;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(policyForBudgetPercent(budget_percent) orelse return error.InvalidFitConfig);
    h.update(&rep.digest);
    h.update(std.mem.sliceAsBytes(levels));
    if (budget_percent == 100) {
        // Preserve the legacy digest byte-for-byte.
        h.update(std.mem.sliceAsBytes(targets));
    } else {
        var encoded: [8]u8 = undefined;
        for (targets) |target_rows| {
            std.mem.writeInt(u64, &encoded, std.math.cast(u64, target_rows) orelse return error.InvalidAttentionShape, .little);
            h.update(&encoded);
        }
        h.update(&.{budget_percent});
    }
    h.final(&digest);
    return .{ .allocator = allocator, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = rows, .preserved_ranges = preserved, .fitted_ranges = rep.ranges.len - preserved, .train_harm_before = before, .train_harm_after = current, .digest = digest };
}

test "divisor2 budget percent floors nominal cap and default is unchanged" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var legacy = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer legacy.deinit();
    var explicit = try chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 100, .{});
    defer explicit.deinit();
    try std.testing.expectEqual(legacy.middle_cap_rows, explicit.middle_cap_rows);
    try std.testing.expectEqualSlices(usize, legacy.targets, explicit.targets);
    try std.testing.expectEqualSlices(u8, &legacy.digest, &explicit.digest);
    var reduced = try chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, .{});
    defer reduced.deinit();
    try std.testing.expectEqual(legacy.middle_cap_rows * 95 / 100, reduced.middle_cap_rows);
    try std.testing.expect(reduced.middle_rows <= reduced.middle_cap_rows);
    try std.testing.expectError(error.InvalidFitConfig, chooseWithBudgetPercent(std.testing.allocator, &rep, 3, .{}, 95, .{}));
}

test "budget95 reports an infeasible row budget before objective" {
    const Local = struct {
        fn fill(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize, _: []f64, _: []f64, _: solver.Control) !void {}
    };
    const ranges = [_]Range{.{ .start = 0, .rows = 1, .target_rows = .{ 1, 1, 1, 1 } }};
    var rep = try build(std.testing.allocator, &ranges, &.{0}, 1, 1, 1, 0, 1024 * 1024, .{ .context = null, .fill_fn = Local.fill }, null, .{});
    defer rep.deinit();
    try std.testing.expectError(error.GeometryBudgetInfeasible, chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, .{}));
}

test "critical minimum may make divisor4 unavailable while divisor3 remains viable" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    rep.ranges[0].target_rows[0] = 3;
    rep.ranges[1].target_rows[0] = 3;
    try std.testing.expectError(error.GeometryBudgetInfeasible, chooseWithBudgetPercent(std.testing.allocator, &rep, 4, .{}, 100, .{}));
    var dense = try chooseWithBudgetPercent(std.testing.allocator, &rep, 3, .{}, 100, .{});
    defer dense.deinit();
    try std.testing.expectEqual(@as(usize, 6), dense.middle_rows);
    try std.testing.expectError(error.GeometryBudgetInfeasible, chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 4, .{}, 100, .{}));
    var incremental = try chooseIncrementalWithBudgetPercent(std.testing.allocator, &rep, 3, .{}, 100, .{});
    defer incremental.deinit();
    try std.testing.expectEqual(@as(usize, 6), incremental.middle_rows);
}

fn budget95AllocationFixture(allocator: std.mem.Allocator) !void {
    var fixture: Fixture = .{};
    var rep = try buildFixture(allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var geometry = try chooseWithBudgetPercent(allocator, &rep, 2, .{}, 95, .{});
    defer geometry.deinit();
}

test "budget95 chooser cancellation and every allocation failure unwind" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 95, cancel.control()));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, budget95AllocationFixture, .{});
}

test "generalized divisor2 budgets floor exactly and preserve sealed legacy policies" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var legacy = try choose(std.testing.allocator, &rep, 2, .{}, .{});
    defer legacy.deinit();
    inline for (.{ @as(u8, 50), 51, 94 }) |percent| {
        var reduced = try chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, percent, .{});
        defer reduced.deinit();
        try std.testing.expectEqual(legacy.middle_cap_rows * percent / 100, reduced.middle_cap_rows);
        try std.testing.expect(reduced.middle_rows <= reduced.middle_cap_rows);
    }
    try std.testing.expectEqualStrings(policy, policyForBudgetPercent(100).?);
    try std.testing.expectEqualStrings(budget95_policy, policyForBudgetPercent(95).?);
    try std.testing.expectEqualStrings(budget_percent_policy, policyForBudgetPercent(50).?);
    inline for (.{ @as(u8, 0), 49, 96, 99 }) |invalid|
        try std.testing.expectError(error.InvalidFitConfig, chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, invalid, .{}));
}

fn generalizedAllocationFixture(allocator: std.mem.Allocator) !void {
    var fixture: Fixture = .{};
    var rep = try buildFixture(allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var geometry = try chooseWithBudgetPercent(allocator, &rep, 2, .{}, 94, .{});
    defer geometry.deinit();
}

test "generalized budget chooser cancellation and every allocation failure unwind" {
    var fixture: Fixture = .{};
    var rep = try buildFixture(std.testing.allocator, 1024 * 1024, .{}, &fixture);
    defer rep.deinit();
    var cancel: Cancel = .{ .cancel_at = 2 };
    try std.testing.expectError(error.Cancelled, chooseWithBudgetPercent(std.testing.allocator, &rep, 2, .{}, 94, cancel.control()));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, generalizedAllocationFixture, .{});
}

test "flat Gram 256 is byte and final-geometry equivalent to 64 with tail 257" {
    const q: usize = 257;
    const dim: usize = 2;
    var original_y: [q * dim]f64 = undefined;
    for (&original_y, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i % 29)) / 29;
    var fixture64: Fixture = .{};
    var rep64 = try build(std.testing.allocator, &test_ranges, &original_y, q, dim, 64, 0, 16 * 1024 * 1024, .{ .context = &fixture64, .fill_fn = Fixture.fill }, null, .{});
    defer rep64.deinit();
    var fixture256: Fixture = .{};
    var rep256 = try build(std.testing.allocator, &test_ranges, &original_y, q, dim, 256, 0, 16 * 1024 * 1024, .{ .context = &fixture256, .fill_fn = Fixture.fill }, null, .{});
    defer rep256.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rep64.mass), std.mem.sliceAsBytes(rep256.mass));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rep64.gram), std.mem.sliceAsBytes(rep256.gram));
    try std.testing.expectEqual(rep64.digest, rep256.digest);
    try std.testing.expectEqual(@as(usize, 5 * test_ranges.len * 3), fixture64.calls);
    try std.testing.expectEqual(@as(usize, 2 * test_ranges.len * 3), fixture256.calls);
    var selected64 = try chooseWithBudgetPercent(std.testing.allocator, &rep64, 2, .{}, 95, .{});
    defer selected64.deinit();
    var selected256 = try chooseWithBudgetPercent(std.testing.allocator, &rep256, 2, .{}, 95, .{});
    defer selected256.deinit();
    try std.testing.expectEqualSlices(adaptive.Level, selected64.levels, selected256.levels);
    try std.testing.expectEqualSlices(usize, selected64.targets, selected256.targets);
    try std.testing.expectEqual(selected64.digest, selected256.digest);
}
