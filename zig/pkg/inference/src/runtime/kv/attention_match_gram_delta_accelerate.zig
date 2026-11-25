// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Default-off FP64 packed numerator delta prototype for the private Gram path.
//! GEMM changes reduction order; callers must compare decisions against the
//! scalar oracle. Results are staged and published only after all checks pass.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("attention_match.zig");

extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

pub const policy = "packed_fp64_accelerate_gram_delta_v1";

pub const Evidence = struct {
    requested: bool = false,
    accelerated: bool = false,
    backend_calls: usize = 0,
    packed_bytes: usize = 0,
    staged_bytes: usize = 0,
    unsupported_fallbacks: usize = 0,
    backend_error_fallbacks: usize = 0,
    nonfinite_fallbacks: usize = 0,
    allocation_failures: usize = 0,
};

pub const Backend = struct {
    context: ?*anyopaque = null,
    gemm_fn: *const fn (?*anyopaque, usize, usize, usize, f64, []const f64, []const f64, f64, []f64) anyerror!void,
};

pub const Options = struct {
    backend: ?Backend = null,
    require_accelerated: bool = false,
    external_live_bytes: usize = 0,
    max_workspace_bytes: usize,
    control: solver.Control = .{},
};

fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn workspaceBytes(query_rows: usize, selected_rows: usize, dim: usize) !usize {
    if (query_rows == 0 or selected_rows == 0 or dim == 0) return error.InvalidAttentionShape;
    const packed_bytes = try checkedMul(try checkedMul(query_rows, selected_rows), @sizeOf(f64));
    const staged = try checkedMul(try checkedMul(query_rows, dim), @sizeOf(f64));
    return std.math.add(usize, packed_bytes, staged) catch error.InvalidAttentionShape;
}

fn validateInputs(page: []const f64, original_values: []const f64, selected_local: []const usize, ratios: []const f64, fitted_values: []const f64, query_rows: usize, range_rows: usize, dim: usize, out: []f64) !void {
    if (query_rows == 0 or range_rows == 0 or dim == 0 or selected_local.len == 0 or
        page.len != try checkedMul(query_rows, range_rows) or
        original_values.len != try checkedMul(range_rows, dim) or
        ratios.len != selected_local.len or
        fitted_values.len != try checkedMul(selected_local.len, dim) or
        out.len != try checkedMul(query_rows, dim)) return error.InvalidAttentionShape;
    for (selected_local, ratios) |local, ratio| if (local >= range_rows or !std.math.isFinite(ratio) or ratio < 0) return error.InvalidAttentionShape;
    for (page) |v| if (!std.math.isFinite(v) or v < 0) return error.NonFiniteAttention;
    for (original_values) |v| if (!std.math.isFinite(v)) return error.NonFiniteAttention;
    for (fitted_values) |v| if (!std.math.isFinite(v)) return error.NonFiniteAttention;
}

fn scalar(page: []const f64, original_values: []const f64, selected_local: []const usize, ratios: []const f64, fitted_values: []const f64, query_rows: usize, range_rows: usize, dim: usize, control: solver.Control, out: []f64) !void {
    @memset(out, 0);
    for (0..query_rows) |qi| {
        try control.check();
        for (0..range_rows) |local| {
            const p = page[qi * range_rows + local];
            for (0..dim) |di| out[qi * dim + di] -= p * original_values[local * dim + di];
        }
        for (selected_local, ratios, 0..) |local, ratio, si| {
            const p = page[qi * range_rows + local] * ratio;
            for (0..dim) |di| out[qi * dim + di] += p * fitted_values[si * dim + di];
        }
    }
}

/// Computes only the signed numerator delta. Denominator/mass accumulation
/// remains on the unchanged scalar path. `out` is unchanged on any error.
pub fn compute(allocator: std.mem.Allocator, page: []const f64, original_values: []const f64, selected_local: []const usize, ratios: []const f64, fitted_values: []const f64, query_rows: usize, range_rows: usize, dim: usize, options: Options, out: []f64, evidence: *Evidence) !void {
    try validateInputs(page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, out);
    try options.control.check();
    evidence.* = .{ .requested = options.require_accelerated or options.backend != null };
    const selected_count = selected_local.len;
    const packed_len = try checkedMul(query_rows, selected_count);
    const staged_len = try checkedMul(query_rows, dim);
    const packed_bytes = try checkedMul(packed_len, @sizeOf(f64));
    const staged_bytes = try checkedMul(staged_len, @sizeOf(f64));
    const owned = std.math.add(usize, packed_bytes, staged_bytes) catch return error.InvalidAttentionShape;
    const peak = std.math.add(usize, options.external_live_bytes, owned) catch return error.InvalidAttentionShape;
    if (peak > options.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    evidence.packed_bytes = packed_bytes;
    evidence.staged_bytes = staged_bytes;

    const staged = allocator.alloc(f64, staged_len) catch |err| {
        evidence.allocation_failures += 1;
        return err;
    };
    defer allocator.free(staged);
    const packed_values = allocator.alloc(f64, packed_len) catch |err| {
        evidence.allocation_failures += 1;
        return err;
    };
    defer allocator.free(packed_values);

    const backend = options.backend orelse {
        if (options.require_accelerated) return error.AcceleratedDeltaUnsupported;
        evidence.unsupported_fallbacks += 1;
        try scalar(page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, options.control, staged);
        @memcpy(out, staged);
        return;
    };
    for (0..query_rows) |qi| {
        try options.control.check();
        for (selected_local, ratios, 0..) |local, ratio, si| packed_values[qi * selected_count + si] = page[qi * range_rows + local] * ratio;
    }
    try options.control.check();
    backend.gemm_fn(backend.context, query_rows, dim, range_rows, -1, page, original_values, 0, staged) catch |err| {
        if (options.require_accelerated) return err;
        evidence.backend_error_fallbacks += 1;
        try scalar(page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, options.control, staged);
        @memcpy(out, staged);
        return;
    };
    evidence.backend_calls += 1;
    try options.control.check();
    backend.gemm_fn(backend.context, query_rows, dim, selected_count, 1, packed_values, fitted_values, 1, staged) catch |err| {
        if (options.require_accelerated) return err;
        evidence.backend_error_fallbacks += 1;
        try scalar(page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, options.control, staged);
        @memcpy(out, staged);
        return;
    };
    evidence.backend_calls += 1;
    try options.control.check();
    var finite = true;
    for (staged) |v| finite = finite and std.math.isFinite(v);
    if (!finite) {
        if (options.require_accelerated) return error.NonFiniteAttention;
        evidence.nonfinite_fallbacks += 1;
        evidence.accelerated = false;
        try scalar(page, original_values, selected_local, ratios, fitted_values, query_rows, range_rows, dim, options.control, staged);
    } else evidence.accelerated = true;
    @memcpy(out, staged);
}

fn accelerateGemm(_: ?*anyopaque, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, b: []const f64, beta: f64, c: []f64) !void {
    const mm = std.math.cast(c_int, m) orelse return error.InvalidAttentionShape;
    const nn = std.math.cast(c_int, n) orelse return error.InvalidAttentionShape;
    const kk = std.math.cast(c_int, k) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(101, 111, 111, mm, nn, kk, alpha, a.ptr, kk, b.ptr, nn, beta, c.ptr, nn);
}

pub fn accelerateBackend() ?Backend {
    return if (builtin.os.tag == .macos) .{ .gemm_fn = accelerateGemm } else null;
}

fn testGemm(_: ?*anyopaque, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, b: []const f64, beta: f64, c: []f64) !void {
    for (0..m) |i| for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |x| sum += a[i * k + x] * b[x * n + j];
        c[i * n + j] = alpha * sum + if (beta == 0) 0 else beta * c[i * n + j];
    };
}

test "packed FP64 delta agrees numerically with independent scalar oracle" {
    const page = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25 };
    const values = [_]f64{ 1, 2, -1, 3, 4, -2, 0.5, 0.25 };
    const selected = [_]usize{ 3, 1 };
    const ratios = [_]f64{ 1.5, 0.75 };
    const fitted = [_]f64{ 0.25, 2.5, -2, 1 };
    var expected: [4]f64 = undefined;
    try scalar(&page, &values, &selected, &ratios, &fitted, 2, 4, 2, .{}, &expected);
    var actual = [_]f64{ 99, 99, 99, 99 };
    var evidence: Evidence = .{};
    try compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 2, 4, 2, .{ .backend = .{ .gemm_fn = testGemm }, .require_accelerated = true, .max_workspace_bytes = 1024 }, &actual, &evidence);
    try std.testing.expect(evidence.accelerated);
    try std.testing.expectEqual(@as(usize, 2), evidence.backend_calls);
    try std.testing.expectEqual(@as(usize, 0), evidence.unsupported_fallbacks);
    try std.testing.expectEqual(@as(usize, 0), evidence.nonfinite_fallbacks);
    try std.testing.expectEqual(try workspaceBytes(2, 2, 2), evidence.packed_bytes + evidence.staged_bytes);
    for (expected, actual) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-14);
}

test "packed FP64 delta rejects a nonfinite consumed range value" {
    const page = [_]f64{ 0.5, 0.5 };
    const values = [_]f64{ 1, std.math.nan(f64) };
    const selected = [_]usize{0};
    const ratios = [_]f64{1};
    const fitted = [_]f64{3};
    var out = [_]f64{77};
    var evidence: Evidence = .{};
    try std.testing.expectError(error.NonFiniteAttention, compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = testGemm }, .require_accelerated = true, .max_workspace_bytes = 1024 }, &out, &evidence));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}

test "packed FP64 delta cap and allocation fail before publication" {
    const page = [_]f64{ 0.5, 0.5 };
    const values = [_]f64{ 1, 2 };
    const selected = [_]usize{0};
    const ratios = [_]f64{1};
    const fitted = [_]f64{3};
    var out = [_]f64{77};
    var evidence: Evidence = .{};
    try std.testing.expectError(error.CompactionWorkspaceLimit, compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = testGemm }, .max_workspace_bytes = 15 }, &out, &evidence));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, compute(failing.allocator(), &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = testGemm }, .max_workspace_bytes = 1024 }, &out, &evidence));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
    try std.testing.expectEqual(@as(usize, 1), evidence.allocation_failures);
}

test "packed FP64 nonfinite backend falls back or fails measured arm" {
    const Bad = struct {
        fn gemm(_: ?*anyopaque, _: usize, _: usize, _: usize, _: f64, _: []const f64, _: []const f64, _: f64, c: []f64) !void {
            @memset(c, std.math.nan(f64));
        }
    };
    const page = [_]f64{ 0.5, 0.5 };
    const values = [_]f64{ 1, 2 };
    const selected = [_]usize{0};
    const ratios = [_]f64{1};
    const fitted = [_]f64{3};
    var out = [_]f64{77};
    var evidence: Evidence = .{};
    try compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = Bad.gemm }, .max_workspace_bytes = 1024 }, &out, &evidence);
    try std.testing.expectEqual(@as(f64, 0), out[0]);
    try std.testing.expectEqual(@as(usize, 1), evidence.nonfinite_fallbacks);
    out[0] = 77;
    try std.testing.expectError(error.NonFiniteAttention, compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = Bad.gemm }, .require_accelerated = true, .max_workspace_bytes = 1024 }, &out, &evidence));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}

test "packed FP64 backend error falls back or fails measured arm" {
    const Bad = struct {
        fn gemm(_: ?*anyopaque, _: usize, _: usize, _: usize, _: f64, _: []const f64, _: []const f64, _: f64, _: []f64) !void {
            return error.BackendFailure;
        }
    };
    const page = [_]f64{ 0.5, 0.5 };
    const values = [_]f64{ 1, 2 };
    const selected = [_]usize{0};
    const ratios = [_]f64{1};
    const fitted = [_]f64{3};
    var out = [_]f64{77};
    var evidence: Evidence = .{};
    try compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = Bad.gemm }, .max_workspace_bytes = 1024 }, &out, &evidence);
    try std.testing.expectEqual(@as(f64, 0), out[0]);
    try std.testing.expectEqual(@as(usize, 1), evidence.backend_error_fallbacks);
    out[0] = 77;
    try std.testing.expectError(error.BackendFailure, compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{ .backend = .{ .gemm_fn = Bad.gemm }, .require_accelerated = true, .max_workspace_bytes = 1024 }, &out, &evidence));
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}

test "packed FP64 cancellation after first GEMM does not publish" {
    const Cancel = struct {
        checks: usize = 0,
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 4) return error.Cancelled;
        }
        fn gemm(raw: ?*anyopaque, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, b: []const f64, beta: f64, c: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return testGemm(null, m, n, k, alpha, a, b, beta, c);
        }
    };
    const page = [_]f64{ 0.5, 0.5 };
    const values = [_]f64{ 1, 2 };
    const selected = [_]usize{0};
    const ratios = [_]f64{1};
    const fitted = [_]f64{3};
    var out = [_]f64{77};
    var evidence: Evidence = .{};
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, compute(std.testing.allocator, &page, &values, &selected, &ratios, &fitted, 1, 2, 1, .{
        .backend = .{ .context = &cancel, .gemm_fn = Cancel.gemm },
        .require_accelerated = true,
        .max_workspace_bytes = 1024,
        .control = .{ .context = &cancel, .check_fn = Cancel.check },
    }, &out, &evidence));
    try std.testing.expectEqual(@as(usize, 1), cancel.calls);
    try std.testing.expectEqual(@as(f64, 77), out[0]);
}
