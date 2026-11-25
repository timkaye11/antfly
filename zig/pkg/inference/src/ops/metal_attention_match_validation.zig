// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Optional macOS FP64 validation summary backend. It accelerates only the
//! query/key logits and probability/value matrices; stable softmax and log
//! mass remain explicit FP64 loops. Fitting callers keep the scalar backend
//! unless they deliberately install this backend for independent validation.
const builtin = @import("builtin");
const std = @import("std");
const solver = @import("../runtime/kv/attention_match.zig");
const challenge = @import("../runtime/kv/attention_match_train_challenge_historical_selection.zig");

extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

pub const Context = struct {
    successful_calls: usize = 0,
    successful_mass_only_calls: usize = 0,
    mass_only_gemm_calls: usize = 0,
    maximum_batch_rows: usize = 0,
    workspace_peak_bytes: usize = 0,
    gemm_fn: ?GemmFn = null,
    test_owner: ?*anyopaque = null,

    pub fn successfulCalls(self: *const Context) usize {
        return self.successful_calls;
    }
};

/// Diagnostic-only TRAIN challenge contract. It cannot be installed as a
/// solver validation backend because it deliberately returns no value output.
pub const GemmFn = *const fn (*Context, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

fn checkedInt(value: usize) !c_int {
    if (value == 0 or value > std.math.maxInt(c_int)) return error.InvalidAttentionShape;
    return @intCast(value);
}

fn workspaceBytes(_: ?*anyopaque, rows: usize, _: usize, queries: usize, dim: usize) !usize {
    _ = try checkedInt(rows);
    _ = try checkedInt(queries);
    _ = try checkedInt(dim);
    const per_query = std.math.add(usize, rows, std.math.add(usize, dim, 1) catch return error.InvalidAttentionShape) catch return error.InvalidAttentionShape;
    return std.math.mul(usize, try std.math.mul(usize, per_query, queries), @sizeOf(f64)) catch error.InvalidAttentionShape;
}

fn massWorkspaceBytes(_: ?*anyopaque, rows: usize, queries: usize, dim: usize) !usize {
    _ = try checkedInt(rows);
    _ = try checkedInt(queries);
    _ = try checkedInt(dim);
    return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows, queries), queries), @sizeOf(f64)) catch error.InvalidAttentionShape;
}

fn finite(values: []const f64) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
}

fn summary(context: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
    if (comptime builtin.os.tag != .macos) return error.UnsupportedOperation;
    const ctx: *Context = @ptrCast(@alignCast(context orelse return error.InvalidAttentionShape));
    try control.check();
    try head.validate();
    if (queries.len == 0 or queries.len % head.dim != 0 or targets.len != queries.len or masses.len != queries.len / head.dim)
        return error.InvalidAttentionShape;
    try finite(queries);
    const q = queries.len / head.dim;
    const nq = try checkedInt(q);
    const nr = try checkedInt(head.rows);
    const nd = try checkedInt(head.dim);
    const logits_len = try std.math.mul(usize, q, head.rows);
    const targets_len = try std.math.mul(usize, q, head.dim);
    const scratch_len = std.math.add(usize, logits_len, std.math.add(usize, targets_len, q) catch return error.InvalidAttentionShape) catch return error.InvalidAttentionShape;
    const scratch = try allocator.alloc(f64, scratch_len);
    defer allocator.free(scratch);
    const logits = scratch[0..logits_len];
    const staged_targets = scratch[logits_len..][0..targets_len];
    const staged_masses = scratch[logits_len + targets_len ..][0..q];
    try control.check();
    if (ctx.gemm_fn) |gemm| gemm(ctx, nq, nr, nd, head.scale, queries.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr) else @"cblas_dgemm$NEWLAPACK"(101, 111, 112, nq, nr, nd, head.scale, queries.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr);
    try control.check();
    try finite(logits);
    for (0..q) |query_index| {
        try control.check();
        const row_logits = logits[query_index * head.rows ..][0..head.rows];
        var maximum: f64 = -std.math.inf(f64);
        for (row_logits, 0..) |*value, row| {
            var x = value.*;
            if (head.softcap > 0) x = head.softcap * std.math.tanh(x / head.softcap);
            x += if (head.bias) |bias| bias[row] else 0;
            if (!std.math.isFinite(x)) return error.NonFiniteAttention;
            value.* = x;
            maximum = @max(maximum, x);
        }
        var sum: f64 = 0;
        for (row_logits) |*value| {
            value.* = @exp(value.* - maximum);
            sum += value.*;
        }
        if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
        for (row_logits) |*value| value.* /= sum;
        staged_masses[query_index] = maximum + @log(sum);
        if (!std.math.isFinite(staged_masses[query_index])) return error.NonFiniteAttention;
    }
    try control.check();
    if (ctx.gemm_fn) |gemm| gemm(ctx, nq, nd, nr, 1, logits.ptr, nr, head.values.ptr, nd, 0, staged_targets.ptr, nd) else @"cblas_dgemm$NEWLAPACK"(101, 111, 111, nq, nd, nr, 1, logits.ptr, nr, head.values.ptr, nd, 0, staged_targets.ptr, nd);
    try finite(staged_targets);
    try control.check();
    @memcpy(targets, staged_targets);
    @memcpy(masses, staged_masses);
    ctx.successful_calls = std.math.add(usize, ctx.successful_calls, 1) catch return error.InvalidAttentionShape;
    ctx.maximum_batch_rows = @max(ctx.maximum_batch_rows, q);
    ctx.workspace_peak_bytes = @max(ctx.workspace_peak_bytes, try workspaceBytes(ctx, head.rows, 1, q, head.dim));
}

fn massesOnly(context: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, masses: []f64, control: solver.Control) !void {
    if (comptime builtin.os.tag != .macos) return error.UnsupportedOperation;
    const ctx: *Context = @ptrCast(@alignCast(context orelse return error.InvalidAttentionShape));
    try control.check();
    try head.validate();
    if (queries.len == 0 or queries.len % head.dim != 0 or masses.len != queries.len / head.dim) return error.InvalidAttentionShape;
    try finite(queries);
    const q = queries.len / head.dim;
    const nq = try checkedInt(q);
    const nr = try checkedInt(head.rows);
    const nd = try checkedInt(head.dim);
    const logits_len = try std.math.mul(usize, q, head.rows);
    const scratch = try allocator.alloc(f64, try std.math.add(usize, logits_len, q));
    defer allocator.free(scratch);
    const logits = scratch[0..logits_len];
    const staged_masses = scratch[logits_len..][0..q];
    try control.check();
    if (ctx.gemm_fn) |gemm| gemm(ctx, nq, nr, nd, head.scale, queries.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr) else @"cblas_dgemm$NEWLAPACK"(101, 111, 112, nq, nr, nd, head.scale, queries.ptr, nd, head.keys.ptr, nd, 0, logits.ptr, nr);
    ctx.mass_only_gemm_calls = std.math.add(usize, ctx.mass_only_gemm_calls, 1) catch return error.InvalidAttentionShape;
    try control.check();
    try finite(logits);
    for (0..q) |query_index| {
        try control.check();
        const row_logits = logits[query_index * head.rows ..][0..head.rows];
        var maximum: f64 = -std.math.inf(f64);
        for (row_logits, 0..) |*value, row| {
            var x = value.*;
            if (head.softcap > 0) x = head.softcap * std.math.tanh(x / head.softcap);
            x += if (head.bias) |bias| bias[row] else 0;
            if (!std.math.isFinite(x)) return error.NonFiniteAttention;
            value.* = x;
            maximum = @max(maximum, x);
        }
        var sum: f64 = 0;
        for (row_logits) |*value| {
            value.* = @exp(value.* - maximum);
            sum += value.*;
        }
        if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
        staged_masses[query_index] = maximum + @log(sum);
        if (!std.math.isFinite(staged_masses[query_index])) return error.NonFiniteAttention;
    }
    try control.check();
    @memcpy(masses, staged_masses);
    ctx.successful_mass_only_calls = std.math.add(usize, ctx.successful_mass_only_calls, 1) catch return error.InvalidAttentionShape;
    ctx.maximum_batch_rows = @max(ctx.maximum_batch_rows, q);
    ctx.workspace_peak_bytes = @max(ctx.workspace_peak_bytes, try massWorkspaceBytes(ctx, head.rows, q, head.dim));
}

pub fn backend(context: *Context) solver.Backend {
    return .{ .validation_batch = .rows_256, .context = context, .workspace_fn = workspaceBytes, .summary_fn = summary };
}

pub fn massBackend(context: *Context) challenge.MassBackend {
    return .{ .context = context, .workspace_fn = massWorkspaceBytes, .masses_fn = massesOnly };
}

const Cancel = struct {
    cancelled: bool,
    fn check(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        if (self.cancelled) return error.Cancelled;
    }
};

test "context Accelerate FP64 validation summary matches scalar attention oracle" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const head: solver.Head = .{
        .keys = &.{ 0.25, -0.5, 1.25, 0.75, -1.5, 0.125 },
        .values = &.{ 1.5, -2, 0.25, 3, -0.75, 0.5 },
        .bias = &.{ 0.1, -0.2, 0.05 },
        .rows = 3,
        .dim = 2,
        .scale = 0.625,
        .softcap = 1.75,
    };
    const queries = [_]f64{ 0.3, -1.1, 2.0, 0.125 };
    var expected_targets: [4]f64 = undefined;
    var expected_masses: [2]f64 = undefined;
    var probabilities: [3]f64 = undefined;
    for (0..2) |i| expected_masses[i] = try solver.attend(head, queries[i * 2 ..][0..2], &probabilities, expected_targets[i * 2 ..][0..2]);
    var context: Context = .{};
    var actual_targets: [4]f64 = undefined;
    var actual_masses: [2]f64 = undefined;
    try summary(&context, std.testing.allocator, head, &queries, &actual_targets, &actual_masses, .{});
    for (expected_targets, actual_targets) |expected, actual| try std.testing.expectApproxEqRel(expected, actual, 2e-14);
    for (expected_masses, actual_masses) |expected, actual| try std.testing.expectApproxEqRel(expected, actual, 2e-14);
    try std.testing.expectEqual(@as(usize, 1), context.successfulCalls());
}

test "TRAIN challenge mass-only backend is bit identical and omits value GEMM" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const Count = struct {
        calls: usize = 0,
        context: Context = .{ .gemm_fn = gemm },
        fn gemm(ctx: *Context, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.test_owner.?));
            const trans_b: c_int = if (self.calls == 0) 112 else 111;
            @"cblas_dgemm$NEWLAPACK"(101, 111, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            self.calls += 1;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0.25, -0.5, 1.25, 0.75, -1.5, 0.125 }, .values = &.{ 1.5, -2, 0.25, 3, -0.75, 0.5 }, .bias = &.{ 0.1, -0.2, 0.05 }, .rows = 3, .dim = 2, .scale = 0.625, .softcap = 1.75 };
    const queries = [_]f64{ 0.3, -1.1, 2.0, 0.125 };
    var count: Count = .{};
    count.context.test_owner = &count;
    var targets: [4]f64 = undefined;
    var expected: [2]f64 = undefined;
    try summary(&count.context, std.testing.allocator, head, &queries, &targets, &expected, .{});
    try std.testing.expectEqual(@as(usize, 2), count.calls);
    count.calls = 0;
    const mass = massBackend(&count.context);
    var actual = [_]f64{ -7, -7 };
    try mass.masses_fn(mass.context, std.testing.allocator, head, &queries, &actual, .{});
    try std.testing.expectEqual(expected, actual);
    try std.testing.expectEqual(@as(usize, 1), count.calls);
    try std.testing.expectEqual(@as(usize, 1), count.context.successful_mass_only_calls);
    try std.testing.expectEqual(@as(usize, 1), count.context.mass_only_gemm_calls);
    try std.testing.expect((try mass.workspace_fn(mass.context, head.rows, 2, head.dim)) < (try workspaceBytes(&count.context, head.rows, 1, 2, head.dim)));
}

test "TRAIN challenge mass-only backend cancellation publishes nothing" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const Stop = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 3) return error.Cancelled;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 1, -1 }, .values = &.{ 3, 4 }, .rows = 2, .dim = 1, .scale = 1 };
    var context: Context = .{};
    const mass = massBackend(&context);
    var stop: Stop = .{};
    var out = [_]f64{ 77, 88 };
    try std.testing.expectError(error.Cancelled, mass.masses_fn(mass.context, std.testing.allocator, head, &.{ 0.25, -0.5 }, &out, .{ .context = &stop, .check_fn = Stop.check }));
    try std.testing.expectEqual([2]f64{ 77, 88 }, out);
    try std.testing.expectEqual(@as(usize, 0), context.successful_mass_only_calls);
    try std.testing.expectEqual(@as(usize, 1), context.mass_only_gemm_calls);
}

fn massAllocationFailure(allocator: std.mem.Allocator) !void {
    const head: solver.Head = .{ .keys = &.{ 1, -1 }, .values = &.{ 3, 4 }, .rows = 2, .dim = 1, .scale = 1 };
    var context: Context = .{};
    const mass = massBackend(&context);
    var out = [_]f64{ 77, 88 };
    mass.masses_fn(mass.context, allocator, head, &.{ 0.25, -0.5 }, &out, .{}) catch |err| {
        try std.testing.expectEqual([2]f64{ 77, 88 }, out);
        try std.testing.expectEqual(@as(usize, 0), context.successful_mass_only_calls);
        return err;
    };
}

test "TRAIN challenge mass-only backend allocation failure publishes nothing" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, massAllocationFailure, .{});
}

test "context Accelerate FP64 validation summary bounds shape finite input and cancellation" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    try std.testing.expectError(error.InvalidAttentionShape, workspaceBytes(null, 0, 1, 1, 1));
    try std.testing.expectError(error.InvalidAttentionShape, workspaceBytes(null, std.math.maxInt(usize), 1, 2, 1));
    const head: solver.Head = .{ .keys = &.{1}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 };
    var context: Context = .{};
    var target: [1]f64 = undefined;
    var mass: [1]f64 = undefined;
    var cancel: Cancel = .{ .cancelled = true };
    const control: solver.Control = .{ .context = &cancel, .check_fn = Cancel.check };
    try std.testing.expectError(error.Cancelled, summary(&context, std.testing.allocator, head, &.{1}, &target, &mass, control));
    try std.testing.expectEqual(@as(usize, 0), context.successfulCalls());
    try std.testing.expectError(error.NonFiniteAttention, summary(&context, std.testing.allocator, head, &.{std.math.nan(f64)}, &target, &mass, .{}));
    try std.testing.expectEqual(@as(usize, 0), context.successfulCalls());
}

test "context Accelerate FP64 validation cancellation after first dispatch publishes no partial output" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const Dispatch = struct {
        checks: usize = 0,
        calls: usize = 0,
        context: Context = .{ .gemm_fn = gemm },
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 3) return error.Cancelled;
        }
        fn gemm(ctx: *Context, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void {
            const self: *@This() = @ptrCast(@alignCast(ctx.test_owner.?));
            self.calls += 1;
            for (0..@intCast(m)) |i| for (0..@intCast(n)) |j| {
                var sum: f64 = 0;
                for (0..@intCast(k)) |x| sum += a[i * @as(usize, @intCast(lda)) + x] * b[x * @as(usize, @intCast(ldb)) + j];
                c[i * @as(usize, @intCast(ldc)) + j] = alpha * sum + beta * c[i * @as(usize, @intCast(ldc)) + j];
            };
        }
    };
    var dispatch: Dispatch = .{};
    dispatch.context.test_owner = &dispatch;
    const head: solver.Head = .{ .keys = &.{1}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 };
    var targets = [_]f64{77};
    var masses = [_]f64{88};
    try std.testing.expectError(error.Cancelled, summary(&dispatch.context, std.testing.allocator, head, &.{3}, &targets, &masses, .{ .context = &dispatch, .check_fn = Dispatch.check }));
    try std.testing.expectEqual(@as(usize, 1), dispatch.calls);
    try std.testing.expectEqual(@as(f64, 77), targets[0]);
    try std.testing.expectEqual(@as(f64, 88), masses[0]);
    try std.testing.expectEqual(@as(usize, 0), dispatch.context.successfulCalls());
}

fn allocationFailureFixture(allocator: std.mem.Allocator) !void {
    const head: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .rows = 2, .dim = 1, .scale = 1 };
    var context: Context = .{};
    var targets: [2]f64 = undefined;
    var masses: [2]f64 = undefined;
    try summary(&context, allocator, head, &.{ 0.25, -0.5 }, &targets, &masses, .{});
    try std.testing.expectEqual(@as(usize, 1), context.successfulCalls());
}

test "context Accelerate FP64 validation summary unwinds allocation failures" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureFixture, .{});
}

test "context Accelerate FP64 validation preserves decisions immediately around the point zero five mass gate" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const source: solver.Head = .{ .keys = &.{0}, .values = &.{3}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    const query = [_]f64{0.25};
    inline for (.{ @as(f64, 1.049999), @as(f64, 1.050001) }, 0..) |ratio, index| {
        const compact: solver.Head = .{ .keys = &.{0}, .values = &.{3}, .bias = &.{@log(ratio)}, .rows = 1, .dim = 1, .scale = 1 };
        const scalar = try solver.validationMetrics(std.testing.allocator, source, compact, &query, .{ .target_rows = 1 });
        var context: Context = .{};
        const accelerated = try solver.validationMetrics(std.testing.allocator, source, compact, &query, .{ .target_rows = 1, .backend = backend(&context) });
        try std.testing.expectApproxEqAbs(scalar.max_relative_mass_error, accelerated.max_relative_mass_error, 2e-14);
        try std.testing.expectApproxEqAbs(scalar.relative_output_rmse, accelerated.relative_output_rmse, 2e-14);
        try std.testing.expectEqual(index == 0, accelerated.max_relative_mass_error <= 0.05 and accelerated.relative_output_rmse <= 0.05);
        try std.testing.expectEqual(@as(usize, 2), context.successfulCalls());
    }
}
