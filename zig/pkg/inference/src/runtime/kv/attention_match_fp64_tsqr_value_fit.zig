// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Isolated, default-off FP64 TSQR value-fit experiment. It solves the same
//! ridge objective as attention_match.fitValues without retaining Q*M or Q*D.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "chronological_fp64_tsqr_value_fit_experiment_v1";

pub const RowSource = struct {
    context: ?*anyopaque,
    workspace_bytes: usize = 0,
    fill_fn: *const fn (?*anyopaque, usize, usize, []f64, []f64, solver.Control) anyerror!void,
};

pub const Workspace = struct {
    owned_bytes: usize,
    source_bytes: usize,
    combined_bytes: usize,
    stack_rows: usize,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub fn workspaceBytes(m: usize, d: usize, batch_rows: usize, source_workspace_bytes: usize) !Workspace {
    if (m == 0 or d == 0 or batch_rows == 0) return error.InvalidFitConfig;
    const stack_rows = try add(m, @max(m, batch_rows));
    var scalars = try add(try mul(stack_rows, m), try mul(stack_rows, d));
    scalars = try add(scalars, stack_rows);
    scalars = try add(scalars, @max(m, d));
    scalars = try add(scalars, try mul(m, d)); // atomic unpublished solution
    const owned = try mul(scalars, @sizeOf(f64));
    return .{ .owned_bytes = owned, .source_bytes = source_workspace_bytes, .combined_bytes = try add(owned, source_workspace_bytes), .stack_rows = stack_rows };
}

fn finite(values: []const f64) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
}

/// Reduces `rows` stacked rows in place. Zero intermediate columns are kept;
/// the ridge merge may make them full-rank later. The final triangular solve
/// is the sole rank authority.
fn reduce(stack_a: []f64, stack_y: []f64, rows: usize, m: usize, d: usize, v: []f64, dots: []f64, control: solver.Control) !void {
    if (rows == 0 or rows > v.len or stack_a.len < try mul(rows, m) or stack_y.len < try mul(rows, d) or dots.len < @max(m, d)) return error.InvalidAttentionShape;
    const columns = @min(rows, m);
    for (0..columns) |j| {
        try control.check();
        var norm: f64 = 0;
        for (j..rows) |i| norm = std.math.hypot(norm, stack_a[i * m + j]);
        if (!std.math.isFinite(norm)) return error.NonFiniteAttention;
        if (norm == 0) continue;
        const alpha = if (stack_a[j * m + j] >= 0) -norm else norm;
        for (j..rows) |i| v[i] = stack_a[i * m + j];
        v[j] -= alpha;
        var vn: f64 = 0;
        for (v[j..rows]) |value| vn = std.math.hypot(vn, value);
        if (!std.math.isFinite(vn) or vn == 0) return error.NonFiniteAttention;
        for (v[j..rows]) |*value| value.* /= vn;
        @memset(dots[0..m], 0);
        for (j..rows) |i| {
            for (j..m) |column| dots[column] += v[i] * stack_a[i * m + column];
        }
        for (j..rows) |i| {
            for (j..m) |column| stack_a[i * m + column] -= 2 * v[i] * dots[column];
        }
        @memset(dots[0..d], 0);
        for (j..rows) |i| {
            for (0..d) |column| dots[column] += v[i] * stack_y[i * d + column];
        }
        for (j..rows) |i| {
            for (0..d) |column| stack_y[i * d + column] -= 2 * v[i] * dots[column];
        }
    }
    try finite(stack_a[0 .. rows * m]);
    try finite(stack_y[0 .. rows * d]);
}

/// Experimental only. The caller supplies the complete Q and immutable source
/// binding through `RowSource`; this function has no heldout input.
pub fn fit(
    allocator: std.mem.Allocator,
    q: usize,
    m: usize,
    d: usize,
    ridge: f64,
    batch_rows: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    source: RowSource,
    control: solver.Control,
    out: []f64,
) !void {
    try control.check();
    if (q == 0 or m == 0 or d == 0 or batch_rows == 0 or out.len != try mul(m, d) or
        !std.math.isFinite(ridge) or ridge < 0 or external_live_bytes > combined_cap) return error.InvalidFitConfig;
    const plan = try workspaceBytes(m, d, batch_rows, source.workspace_bytes);
    if (plan.combined_bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    try control.check();

    const stack_a = try allocator.alloc(f64, try mul(plan.stack_rows, m));
    defer allocator.free(stack_a);
    const stack_y = try allocator.alloc(f64, try mul(plan.stack_rows, d));
    defer allocator.free(stack_y);
    const v = try allocator.alloc(f64, plan.stack_rows);
    defer allocator.free(v);
    const dots = try allocator.alloc(f64, @max(m, d));
    defer allocator.free(dots);
    const solution = try allocator.alloc(f64, try mul(m, d));
    defer allocator.free(solution);

    var rows: usize = 0;
    var first: usize = 0;
    while (first < q) {
        try control.check();
        const take = @min(batch_rows, q - first);
        if (rows + take > plan.stack_rows) return error.CompactionWorkspaceLimit;
        const a_slice = stack_a[rows * m ..][0 .. take * m];
        const y_slice = stack_y[rows * d ..][0 .. take * d];
        try source.fill_fn(source.context, first, take, a_slice, y_slice, control);
        try finite(a_slice);
        try finite(y_slice);
        rows += take;
        first += take;
        if (rows >= m) {
            try reduce(stack_a, stack_y, rows, m, d, v, dots, control);
            rows = m;
        }
    }

    if (rows + m > plan.stack_rows) return error.CompactionWorkspaceLimit;
    @memset(stack_a[rows * m ..][0 .. m * m], 0);
    @memset(stack_y[rows * d ..][0 .. m * d], 0);
    const ridge_scale = @sqrt(ridge);
    for (0..m) |j| stack_a[(rows + j) * m + j] = ridge_scale;
    rows += m;
    try reduce(stack_a, stack_y, rows, m, d, v, dots, control);

    @memset(solution, 0);
    var j = m;
    while (j > 0) {
        j -= 1;
        try control.check();
        const diagonal = stack_a[j * m + j];
        if (!std.math.isFinite(diagonal) or diagonal == 0) return error.SingularValueFit;
        for (0..d) |column| {
            var value = stack_y[j * d + column];
            for (j + 1..m) |k| value -= stack_a[j * m + k] * solution[k * d + column];
            solution[j * d + column] = value / diagonal;
        }
    }
    try finite(solution);
    try control.check();
    @memcpy(out, solution);
}

const Fixture = struct {
    a: []const f64,
    y: []const f64,
    q: usize,
    m: usize,
    d: usize,
    calls: usize = 0,
    nonfinite: bool = false,

    fn fill(raw: ?*anyopaque, first: usize, count: usize, a: []f64, y: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        self.calls += 1;
        if (first > self.q or count > self.q - first or a.len != count * self.m or y.len != count * self.d) return error.InvalidAttentionShape;
        try control.check();
        @memcpy(a, self.a[first * self.m ..][0 .. count * self.m]);
        @memcpy(y, self.y[first * self.d ..][0 .. count * self.d]);
        if (self.nonfinite) a[0] = std.math.nan(f64);
    }
};

fn parityAllocation(allocator: std.mem.Allocator) !void {
    const q = 7;
    const m = 3;
    const d = 2;
    const a = [_]f64{ 1, 0, 2, 0, 1, 1, 1, 1, 0, 2, -1, 1, -1, 2, 0.5, 0.25, 1.5, -0.5, 1.25, 0.75, 2 };
    const y = [_]f64{ 2, -1, 1, 3, 4, 2, -2, 1, 3, 0.5, 1.5, -3, 2.5, 1 };
    var fixture: Fixture = .{ .a = &a, .y = &y, .q = q, .m = m, .d = d };
    var actual: [m * d]f64 = undefined;
    var expected: [m * d]f64 = undefined;
    try fit(allocator, q, m, d, 1e-8, 2, 0, 1 << 20, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}, &actual);
    try solver.fitValuesReferenceForTest(allocator, &a, &y, q, m, d, 1e-8, .{}, &expected);
    for (expected, actual) |want, got| try std.testing.expectApproxEqRel(want, got, 5e-12);
    try std.testing.expectEqual(@as(usize, 4), fixture.calls);
}

test "TSQR agrees with dense ridge Householder and unwinds every allocation" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parityAllocation, .{});
}

test "TSQR ridge repairs a zero column and zero ridge rejects rank deficiency" {
    const a = [_]f64{ 1, 0, 2, 0, 3, 0 };
    const y = [_]f64{ 1, 2, 3 };
    var fixture: Fixture = .{ .a = &a, .y = &y, .q = 3, .m = 2, .d = 1 };
    var out: [2]f64 = .{ -7, -7 };
    try fit(std.testing.allocator, 3, 2, 1, 1e-8, 2, 0, 1 << 20, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}, &out);
    try std.testing.expect(std.math.isFinite(out[0]) and std.math.isFinite(out[1]));
    out = .{ -7, -7 };
    try std.testing.expectError(error.SingularValueFit, fit(std.testing.allocator, 3, 2, 1, 0, 2, 0, 1 << 20, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}, &out));
    try std.testing.expectEqual([2]f64{ -7, -7 }, out);
}

test "TSQR cap cancellation and nonfinite failures never publish output" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const a = [_]f64{ 1, 2 };
    const y = [_]f64{ 3, 4 };
    var fixture: Fixture = .{ .a = &a, .y = &y, .q = 2, .m = 1, .d = 1 };
    var out = [_]f64{-9};
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, 2, 1, 1, 1e-8, 1, 0, 1, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}, &out));
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
    try std.testing.expectError(error.Cancelled, fit(std.testing.allocator, 2, 1, 1, 1e-8, 1, 0, 1 << 20, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{ .check_fn = Cancel.check }, &out));
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
    fixture.nonfinite = true;
    try std.testing.expectError(error.NonFiniteAttention, fit(std.testing.allocator, 2, 1, 1, 1e-8, 1, 0, 1 << 20, .{ .context = &fixture, .fill_fn = Fixture.fill }, .{}, &out));
    try std.testing.expectEqual(@as(f64, -9), out[0]);
}
