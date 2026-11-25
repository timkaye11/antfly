// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Private training-only anchored joint chunk-mass coefficient solve.
const std = @import("std");
const solver = @import("attention_match.zig");

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub const Diagnostics = struct {
    global_max_relative_before: f64,
    global_max_relative_after: f64,
    per_chunk_max_relative_before: ?f64,
    per_chunk_max_relative_after: ?f64,
    worst_chunk_before: ?usize,
    worst_chunk_after: ?usize,
    worst_query_before: ?usize,
    worst_query_after: ?usize,
    zero_columns: usize,
    active_columns: usize,
    augmented_rows: usize,
    admitted_peak_bytes: usize,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    alpha: []f64,
    diagnostics: Diagnostics,
    nnls_sweeps: usize,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.alpha);
        self.* = undefined;
    }
};

pub fn workspaceBytes(q: usize, k: usize, cfg: solver.Config) !usize {
    _ = cfg;
    if (q == 0 or k == 0) return error.InvalidAttentionShape;
    const rows = try add(q, k);
    var retained = try mul(k, @sizeOf(f64));
    retained = try add(retained, try mul(k, @sizeOf(usize)));
    retained = try add(retained, try mul(try mul(rows, k), @sizeOf(f64)));
    retained = try add(retained, try mul(rows, 3 * @sizeOf(f64)));
    retained = try add(retained, try mul(k, @sizeOf(f64)));
    return add(retained, try solver.workspaceBytes(k, k, rows, 1));
}

fn relativeError(actual: f64, expected: f64, floor: f64) f64 {
    return @abs(actual - expected) / @max(@abs(expected), floor);
}

/// C is query-major q-by-k. diag_a[c]=sum_i C[i,c]^2 and
/// diag_b[c]=sum_i C[i,c]*rho[i,c], accumulated in the same scalar order.
/// Optional rho is query-major q-by-k and is used only for diagnostics.
pub fn fit(
    allocator: std.mem.Allocator,
    c: []const f64,
    rhs: []const f64,
    diag_a: []const f64,
    diag_b: []const f64,
    rho: ?[]const f64,
    cfg: solver.Config,
) !Result {
    const q = rhs.len;
    const k = diag_a.len;
    if (q == 0 or k == 0 or diag_b.len != k or c.len != try mul(q, k) or (rho != null and rho.?.len != c.len)) return error.InvalidAttentionShape;
    if (cfg.max_workspace_bytes == 0 or cfg.max_sweeps == 0 or !std.math.isFinite(cfg.nnls_tolerance) or cfg.nnls_tolerance <= 0 or !std.math.isFinite(cfg.min_weight) or cfg.min_weight <= 0) return error.InvalidFitConfig;
    try cfg.control.check();
    for (c) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    for (rhs) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    for (diag_a, diag_b) |a, b| if (!std.math.isFinite(a) or a < 0 or !std.math.isFinite(b)) return error.NonFiniteAttention;
    if (rho) |values| for (values) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;

    var active_count: usize = 0;
    for (diag_a) |a| active_count += @intFromBool(a > 0);
    const rows = try add(q, active_count);
    var retained = try mul(k, @sizeOf(f64)); // result alpha
    retained = try add(retained, try mul(active_count, @sizeOf(usize)));
    retained = try add(retained, try mul(try mul(rows, active_count), @sizeOf(f64)));
    retained = try add(retained, try mul(rows, 3 * @sizeOf(f64))); // target, beta, residual upper bound
    retained = try add(retained, try mul(active_count, @sizeOf(f64))); // norms
    const nested = if (active_count == 0) 0 else try solver.workspaceBytes(active_count, active_count, rows, 1);
    const admitted_peak = try add(retained, nested);
    if (admitted_peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;

    const alpha = try allocator.alloc(f64, k);
    errdefer allocator.free(alpha);
    @memset(alpha, 1);
    if (active_count == 0) return .{ .allocator = allocator, .alpha = alpha, .diagnostics = diagnostics(c, rhs, rho, alpha, cfg.min_weight, 0, rows, admitted_peak), .nnls_sweeps = 0 };
    const active = try allocator.alloc(usize, active_count);
    defer allocator.free(active);
    var at: usize = 0;
    for (diag_a, 0..) |a, index| if (a > 0) {
        active[at] = index;
        at += 1;
    };
    const augmented = try allocator.alloc(f64, try mul(rows, active_count));
    defer allocator.free(augmented);
    @memset(augmented, 0);
    const target = try allocator.alloc(f64, rows);
    defer allocator.free(target);
    for (0..q) |qi| {
        var epsilon_sum: f64 = 0;
        for (active, 0..) |source_column, column| {
            const value = c[qi * k + source_column];
            augmented[qi * active_count + column] = value;
            epsilon_sum += value;
        }
        target[qi] = rhs[qi] - cfg.min_weight * epsilon_sum;
        if (!std.math.isFinite(target[qi])) return error.NonFiniteAttention;
    }
    for (active, 0..) |source_column, column| {
        const root = @sqrt(diag_a[source_column]);
        augmented[(q + column) * active_count + column] = root;
        target[q + column] = diag_b[source_column] / root - cfg.min_weight * root;
        if (!std.math.isFinite(target[q + column])) return error.NonFiniteAttention;
    }
    const beta = try allocator.alloc(f64, active_count);
    defer allocator.free(beta);
    @memset(beta, 1 - cfg.min_weight);
    const residual = try allocator.alloc(f64, rows);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, active_count);
    defer allocator.free(norms);
    var native_cfg = cfg;
    native_cfg.backend = null;
    const sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, augmented, target, rows, active_count, beta, residual, norms, native_cfg, true);
    for (active, beta) |source_column, value| alpha[source_column] = value + cfg.min_weight;
    return .{ .allocator = allocator, .alpha = alpha, .diagnostics = diagnostics(c, rhs, rho, alpha, cfg.min_weight, active_count, rows, admitted_peak), .nnls_sweeps = sweeps };
}

fn diagnostics(c: []const f64, rhs: []const f64, rho: ?[]const f64, alpha: []const f64, floor: f64, active: usize, rows: usize, admitted_peak: usize) Diagnostics {
    const q = rhs.len;
    const k = alpha.len;
    var global_before: f64 = 0;
    var global_after: f64 = 0;
    for (0..q) |qi| {
        var before: f64 = 0;
        var after: f64 = 0;
        for (alpha, 0..) |a, ci| {
            before += c[qi * k + ci];
            after += c[qi * k + ci] * a;
        }
        global_before = @max(global_before, relativeError(before, rhs[qi], floor));
        global_after = @max(global_after, relativeError(after, rhs[qi], floor));
    }
    var chunk_before: ?f64 = null;
    var chunk_after: ?f64 = null;
    var worst_cb: ?usize = null;
    var worst_ca: ?usize = null;
    var worst_qb: ?usize = null;
    var worst_qa: ?usize = null;
    if (rho) |values| for (0..q) |qi| for (alpha, 0..) |a, ci| {
        const expected = values[qi * k + ci];
        const before = relativeError(c[qi * k + ci], expected, floor);
        const after = relativeError(c[qi * k + ci] * a, expected, floor);
        if (chunk_before == null or before > chunk_before.?) {
            chunk_before = before;
            worst_cb = ci;
            worst_qb = qi;
        }
        if (chunk_after == null or after > chunk_after.?) {
            chunk_after = after;
            worst_ca = ci;
            worst_qa = qi;
        }
    };
    return .{ .global_max_relative_before = global_before, .global_max_relative_after = global_after, .per_chunk_max_relative_before = chunk_before, .per_chunk_max_relative_after = chunk_after, .worst_chunk_before = worst_cb, .worst_chunk_after = worst_ca, .worst_query_before = worst_qb, .worst_query_after = worst_qa, .zero_columns = k - active, .active_columns = active, .augmented_rows = rows, .admitted_peak_bytes = admitted_peak };
}

fn loss(c: []const f64, rhs: []const f64, rho: []const f64, alpha: []const f64) struct { global: f64, anchored: f64 } {
    const q = rhs.len;
    const k = alpha.len;
    var global: f64 = 0;
    var anchored: f64 = 0;
    for (0..q) |qi| {
        var total: f64 = 0;
        for (alpha, 0..) |a, ci| {
            const actual = c[qi * k + ci] * a;
            total += actual;
            anchored += std.math.pow(f64, actual - rho[qi * k + ci], 2);
        }
        global += std.math.pow(f64, total - rhs[qi], 2);
    }
    return .{ .global = global, .anchored = global + anchored };
}

test "anchored objective distinguishes correlated chunks erased by global sum" {
    const c = [_]f64{ 1, 1, 1, 1 };
    const rhs = [_]f64{ 2, 2 };
    const rho = [_]f64{ 1, 1, 1, 1 };
    for ([_][2]f64{ .{ 2, 0 }, .{ 1, 1 }, .{ 0, 2 } }, [_]f64{ 4, 0, 4 }) |alpha, expected_anchor| {
        const value = loss(&c, &rhs, &rho, &alpha);
        try std.testing.expectEqual(@as(f64, 0), value.global);
        try std.testing.expectEqual(expected_anchor, value.anchored);
    }
    var result = try fit(std.testing.allocator, &c, &rhs, &.{ 2, 2 }, &.{ 2, 2 }, &rho, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.alpha[0], 1e-8);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.alpha[1], 1e-8);
}

test "anchored objective preserves exact zero column at one" {
    var result = try fit(std.testing.allocator, &.{ 1, 0, 2, 0 }, &.{ 1, 2 }, &.{ 5, 0 }, &.{ 5, 0 }, &.{ 1, 3, 2, 4 }, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1), result.alpha[1]);
    try std.testing.expectEqual(@as(usize, 1), result.diagnostics.zero_columns);
}

test "anchored warm start preserves weak active column at alpha one" {
    const epsilon: f64 = 1e-6;
    const c = [_]f64{ 1, epsilon, 1, epsilon };
    const rhs = [_]f64{ 1 + epsilon, 1 + epsilon };
    const diag_a = [_]f64{ 2, 2 * epsilon * epsilon };
    const diag_b = diag_a;
    var result = try fit(std.testing.allocator, &c, &rhs, &diag_a, &diag_b, &c, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.alpha[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1), result.alpha[1], 1e-8);
    try std.testing.expect(result.diagnostics.global_max_relative_after <= result.diagnostics.global_max_relative_before + 1e-12);
    try std.testing.expect(result.diagnostics.per_chunk_max_relative_after.? <= result.diagnostics.per_chunk_max_relative_before.? + 1e-12);
}

test "compressed diagonal anchors differ from direct anchors by one constant" {
    const c = [_]f64{ 1, 2, 3, 4 };
    const rho = [_]f64{ 1.5, 0.5, 2, 3 };
    const a = [_]f64{ 10, 20 };
    const b = [_]f64{ 7.5, 13 };
    var expected_difference: ?f64 = null;
    for ([_][2]f64{ .{ 0, 0 }, .{ 1, 1 }, .{ 2, 0.25 } }) |alpha| {
        var direct: f64 = 0;
        for (0..2) |qi| {
            for (alpha, 0..) |value, ci|
                direct += std.math.pow(f64, c[qi * 2 + ci] * value - rho[qi * 2 + ci], 2);
        }
        var compressed: f64 = 0;
        for (alpha, 0..) |value, ci| {
            const root = @sqrt(a[ci]);
            compressed += std.math.pow(f64, root * value - b[ci] / root, 2);
        }
        const difference = direct - compressed;
        if (expected_difference) |expected|
            try std.testing.expectApproxEqAbs(expected, difference, 1e-12)
        else
            expected_difference = difference;
    }
}

fn anchoredAllocationFixture(allocator: std.mem.Allocator) !void {
    var result = try fit(allocator, &.{ 1, 1, 1, 1 }, &.{ 2, 2 }, &.{ 2, 2 }, &.{ 2, 2 }, &.{ 1, 1, 1, 1 }, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    result.deinit();
}

test "anchored objective cap cancellation and allocation unwind" {
    var admitted = try fit(std.testing.allocator, &.{ 1, 1, 1, 1 }, &.{ 2, 2 }, &.{ 2, 2 }, &.{ 2, 2 }, null, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    const peak = admitted.diagnostics.admitted_peak_bytes;
    admitted.deinit();
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, &.{ 1, 1, 1, 1 }, &.{ 2, 2 }, &.{ 2, 2 }, &.{ 2, 2 }, null, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = peak - 1 }));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, fit(std.testing.allocator, &.{ 1, 1, 1, 1 }, &.{ 2, 2 }, &.{ 2, 2 }, &.{ 2, 2 }, null, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, anchoredAllocationFixture, .{});
}
