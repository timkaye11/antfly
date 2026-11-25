// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Optional FP64 numerator acceleration over an already-authenticated page.
//! Mass retains the scalar order; both outputs publish atomically.
const std = @import("std");
const solver = @import("attention_match.zig");
const kernel = @import("attention_match_gram_delta_accelerate.zig");

pub const Evidence = struct {
    tiles: usize = 0,
    backend_calls: usize = 0,
    packed_bytes_peak: usize = 0,
    staged_bytes_peak: usize = 0,
    fallbacks: usize = 0,
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}

/// Kernel packed+staged storage, caller numerator staging, and scalar mass
/// staging. The page and q-independent ratios are caller-owned external bytes.
pub fn workspaceBytes(query_rows: usize, selected_rows: usize, dim: usize) !usize {
    const numerator = try mul(try mul(query_rows, dim), @sizeOf(f64));
    const mass = try mul(query_rows, @sizeOf(f64));
    return add(try kernel.workspaceBytes(query_rows, selected_rows, dim), try add(numerator, mass));
}

pub fn fill(
    allocator: std.mem.Allocator,
    page: []const f64,
    original_values: []const f64,
    selected_local: []const usize,
    ratios: []const f64,
    fitted_values: []const f64,
    query_rows: usize,
    range_rows: usize,
    dim: usize,
    backend: kernel.Backend,
    external_live_bytes: usize,
    combined_cap: usize,
    control: solver.Control,
    mass: []f64,
    numerator: []f64,
    evidence: *Evidence,
) !void {
    const qd = try mul(query_rows, dim);
    if (mass.len != query_rows or numerator.len != qd or page.len != try mul(query_rows, range_rows))
        return error.InvalidAttentionShape;
    const owned = try workspaceBytes(query_rows, selected_local.len, dim);
    if (external_live_bytes > combined_cap or owned > combined_cap - external_live_bytes)
        return error.CompactionWorkspaceLimit;
    try control.check();
    const staged_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(staged_numerator);
    const staged_mass = try allocator.alloc(f64, query_rows);
    defer allocator.free(staged_mass);
    var observed: kernel.Evidence = .{};
    const staging_external = try add(external_live_bytes, try add(try mul(qd, @sizeOf(f64)), try mul(query_rows, @sizeOf(f64))));
    try kernel.compute(allocator, page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, .{
        .backend = backend,
        .require_accelerated = true,
        .external_live_bytes = staging_external,
        .max_workspace_bytes = combined_cap,
        .control = control,
    }, staged_numerator, &observed);
    if (!observed.accelerated or observed.backend_calls != 2 or observed.unsupported_fallbacks != 0 or
        observed.backend_error_fallbacks != 0 or observed.nonfinite_fallbacks != 0)
        return error.AcceleratedDeltaUnsupported;
    for (0..query_rows) |qi| {
        try control.check();
        var value: f64 = 0;
        for (0..range_rows) |local| value -= page[qi * range_rows + local];
        for (selected_local, ratios) |local, ratio| value += page[qi * range_rows + local] * ratio;
        if (!std.math.isFinite(value)) return error.NonFiniteAttention;
        staged_mass[qi] = value;
    }
    @memcpy(mass, staged_mass);
    @memcpy(numerator, staged_numerator);
    evidence.tiles = try add(evidence.tiles, 1);
    evidence.backend_calls = try add(evidence.backend_calls, observed.backend_calls);
    evidence.packed_bytes_peak = @max(evidence.packed_bytes_peak, observed.packed_bytes);
    evidence.staged_bytes_peak = @max(evidence.staged_bytes_peak, try add(observed.staged_bytes, try add(try mul(qd, @sizeOf(f64)), try mul(query_rows, @sizeOf(f64)))));
}

const Fake = struct {
    fn gemm(_: ?*anyopaque, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, b: []const f64, beta: f64, c: []f64) !void {
        for (0..m) |i| for (0..n) |j| {
            var sum: f64 = 0;
            for (0..k) |x| sum += a[i * k + x] * b[x * n + j];
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        };
    }
};

test "bounded page acceleration matches direct signed delta and stages mass" {
    var mass = [_]f64{ 77, 88 };
    var numerator = [_]f64{ 11, 22, 33, 44 };
    var evidence: Evidence = .{};
    const page = [_]f64{ 0.1, 0.2, 0.7, 0.4, 0.3, 0.3 };
    try fill(std.testing.allocator, &page, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 0, 2 }, &.{ 2, 0.5 }, &.{ 7, 8, 9, 10 }, 2, 3, 2, .{ .gemm_fn = Fake.gemm }, 0, 1 << 20, .{}, &mass, &numerator, &evidence);
    try std.testing.expectApproxEqAbs(@as(f64, -0.45), mass[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.05), mass[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.35), numerator[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1), numerator[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.15), numerator[2], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 4.1), numerator[3], 1e-12);
    try std.testing.expectEqual(@as(usize, 1), evidence.tiles);
    try std.testing.expectEqual(@as(usize, 2), evidence.backend_calls);
    try std.testing.expectError(error.CompactionWorkspaceLimit, fill(std.testing.allocator, &page, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 0, 2 }, &.{ 2, 0.5 }, &.{ 7, 8, 9, 10 }, 2, 3, 2, .{ .gemm_fn = Fake.gemm }, 0, 1, .{}, &mass, &numerator, &evidence));
    const before_mass = mass;
    const before_numerator = numerator;
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, fill(std.testing.allocator, &page, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 0, 2 }, &.{ 2, 0.5 }, &.{ 7, 8, 9, 10 }, 2, 3, 2, .{ .gemm_fn = Fake.gemm }, 0, 1 << 20, .{ .check_fn = Cancel.check }, &mass, &numerator, &evidence));
    try std.testing.expectEqualSlices(f64, &before_mass, &mass);
    try std.testing.expectEqualSlices(f64, &before_numerator, &numerator);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, fill(failing.allocator(), &page, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 0, 2 }, &.{ 2, 0.5 }, &.{ 7, 8, 9, 10 }, 2, 3, 2, .{ .gemm_fn = Fake.gemm }, 0, 1 << 20, .{}, &mass, &numerator, &evidence));
    try std.testing.expectEqualSlices(f64, &before_mass, &mass);
    try std.testing.expectEqualSlices(f64, &before_numerator, &numerator);
}
