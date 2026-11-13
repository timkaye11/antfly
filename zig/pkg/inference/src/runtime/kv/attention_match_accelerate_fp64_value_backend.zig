// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Diagnostic-only value-phase adapter for Accelerate FP64 Householder QR.
const std = @import("std");
const platform = @import("antfly_platform");
const solver = @import("attention_match.zig");
const lapack = @import("../../ops/metal_attention_match_lapack.zig");

pub const policy = "accelerate_fp64_householder_query_workspace_dtrsm_atomic_v2";
pub const work_cap_multiplier: usize = 64;

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub const Workspace = struct {
    lapack_qr_bytes: usize,
    lapack_rhs_bytes: usize,
    lapack_tau_bytes: usize,
    lapack_work_bytes: usize,
    atomic_output_bytes: usize,
    total_bytes: usize,
};

pub const Stats = struct {
    calls: usize = 0,
    lq_optimal_elements_max: usize = 0,
    apply_optimal_elements_max: usize = 0,
    declared_elements_max: usize = 0,
    chosen_elements_max: usize = 0,
    workspace_peak_bytes: usize = 0,
    scalar_rhs_calls: usize = 0,
    vector_rhs_calls: usize = 0,
    scalar_rhs_ns: u64 = 0,
    vector_rhs_ns: u64 = 0,
};

fn monotonicClock(_: ?*anyopaque) u64 {
    return platform.time.monotonicNs();
}

pub const Context = struct {
    stats: Stats = .{},
    clock_context: ?*anyopaque = null,
    clock_fn: *const fn (?*anyopaque) u64 = monotonicClock,

    fn now(self: *const Context) u64 {
        return self.clock_fn(self.clock_context);
    }

    pub fn valueBackend(self: *Context) solver.ValueBackend {
        return .{ .context = self, .workspace_fn = workspace, .fit_fn = fit };
    }
};

pub fn workspacePlan(q: usize, m: usize, d: usize, ridge: f64) !Workspace {
    if (q == 0 or m == 0 or d == 0 or !std.math.isFinite(ridge) or ridge < 0) return error.InvalidFitConfig;
    const rows = try add(q, m);
    _ = std.math.cast(c_int, rows) orelse return error.InvalidAttentionShape;
    _ = std.math.cast(c_int, m) orelse return error.InvalidAttentionShape;
    _ = std.math.cast(c_int, d) orelse return error.InvalidAttentionShape;
    const qr = try mul(try mul(rows, m), @sizeOf(f64));
    const rhs = try mul(try mul(rows, d), @sizeOf(f64));
    const tau = try mul(m, @sizeOf(f64));
    const work = try mul(try mul(work_cap_multiplier, @max(m, d)), @sizeOf(f64));
    const output = try mul(try mul(m, d), @sizeOf(f64));
    var total = try add(qr, rhs);
    total = try add(total, tau);
    total = try add(total, work);
    total = try add(total, output);
    return .{ .lapack_qr_bytes = qr, .lapack_rhs_bytes = rhs, .lapack_tau_bytes = tau, .lapack_work_bytes = work, .atomic_output_bytes = output, .total_bytes = total };
}

fn workspace(_: ?*anyopaque, q: usize, m: usize, d: usize, ridge: f64) !usize {
    if (comptime !lapack.supported) return error.UnsupportedMetalFp64Values;
    return (try workspacePlan(q, m, d, ridge)).total_bytes;
}

fn fit(raw_context: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
    if (comptime !lapack.supported) return error.UnsupportedMetalFp64Values;
    const context: ?*Context = if (raw_context) |raw| @ptrCast(@alignCast(raw)) else null;
    const started = if (context) |value| value.now() else 0;
    try control.check();
    const plan = try workspacePlan(q, m, d, ridge);
    if (a.len != try mul(q, m) or y.len != try mul(q, d) or out.len != try mul(m, d) or plan.atomic_output_bytes != try mul(out.len, @sizeOf(f64))) return error.InvalidAttentionShape;
    const unpublished = try allocator.alloc(f64, out.len);
    defer allocator.free(unpublished);
    var metadata: lapack.WorkspaceMetadata = undefined;
    try lapack.valuesWithWorkspaceLimit(allocator, a, y, q, m, d, ridge, control, try mul(work_cap_multiplier, @max(m, d)), true, &metadata, unpublished);
    if (metadata.chosen_elements > metadata.declared_cap_elements or metadata.declared_cap_elements != try mul(work_cap_multiplier, @max(m, d)) or !metadata.blas_backsolve) return error.AttentionMatchLapackFailure;
    for (unpublished) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    try control.check();
    if (context) |value| {
        value.stats.calls +|= 1;
        value.stats.lq_optimal_elements_max = @max(value.stats.lq_optimal_elements_max, metadata.lq_optimal_elements);
        value.stats.apply_optimal_elements_max = @max(value.stats.apply_optimal_elements_max, metadata.apply_optimal_elements);
        value.stats.declared_elements_max = @max(value.stats.declared_elements_max, metadata.declared_cap_elements);
        value.stats.chosen_elements_max = @max(value.stats.chosen_elements_max, metadata.chosen_elements);
        value.stats.workspace_peak_bytes = @max(value.stats.workspace_peak_bytes, plan.total_bytes);
        const elapsed = value.now() -| started;
        if (d == 1) {
            value.stats.scalar_rhs_calls +|= 1;
            value.stats.scalar_rhs_ns +|= elapsed;
        } else {
            value.stats.vector_rhs_calls +|= 1;
            value.stats.vector_rhs_ns +|= elapsed;
        }
    }
    @memcpy(out, unpublished);
}

pub fn valueBackend() solver.ValueBackend {
    return .{ .context = null, .workspace_fn = workspace, .fit_fn = fit };
}

test "workspace plan includes exact LAPACK arrays and atomic output" {
    const plan = try workspacePlan(7, 3, 2, 1e-8);
    try std.testing.expectEqual(@as(usize, (10 * 3 + 10 * 2 + 3 + 64 * 3 + 3 * 2) * 8), plan.total_bytes);
    try std.testing.expectEqual(@as(usize, 3 * 2 * 8), plan.atomic_output_bytes);
    try std.testing.expectError(error.CompactionWorkspaceLimit, workspacePlan(std.math.maxInt(usize), 2, 1, 0));
    try std.testing.expectError(error.InvalidFitConfig, workspacePlan(1, 1, 1, -1));
}

test "resident E4 value shapes reserve bounded queried workspace" {
    const ordinary = try workspacePlan(3072, 170, 512, 1e-8);
    const high_fanout = try workspacePlan(12288, 170, 512, 1e-8);
    try std.testing.expectEqual(@as(usize, 64 * 512 * 8), ordinary.lapack_work_bytes);
    try std.testing.expectEqual(ordinary.lapack_work_bytes, high_fanout.lapack_work_bytes);
    try std.testing.expect(ordinary.total_bytes < 64 * 1024 * 1024);
    try std.testing.expect(high_fanout.total_bytes < 128 * 1024 * 1024);
}

test "value backend is context-free and reports the checked plan" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const backend = valueBackend();
    try std.testing.expectEqual(@as(?*anyopaque, null), backend.context);
    try std.testing.expectEqual((try workspacePlan(7, 3, 2, 1e-8)).total_bytes, try backend.workspace_fn(backend.context, 7, 3, 2, 1e-8));
}

test "diagnostic context reports actual queried workspace only after publication" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const Clock = struct {
        value: u64 = 100,
        fn now(raw: ?*anyopaque) u64 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const result = self.value;
            self.value += 7;
            return result;
        }
    };
    var clock: Clock = .{};
    var context: Context = .{ .clock_context = &clock, .clock_fn = Clock.now };
    const backend = context.valueBackend();
    var out = [_]f64{ 0, 0 };
    try backend.fit_fn(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3, 5 }, 3, 2, 1, 0, .{}, &out);
    try std.testing.expectEqual(@as(usize, 1), context.stats.calls);
    try std.testing.expectEqual(@as(usize, 1), context.stats.scalar_rhs_calls);
    try std.testing.expectEqual(@as(usize, 0), context.stats.vector_rhs_calls);
    try std.testing.expectEqual(@as(u64, 7), context.stats.scalar_rhs_ns);
    try std.testing.expectEqual(@as(u64, 0), context.stats.vector_rhs_ns);
    try std.testing.expect(context.stats.chosen_elements_max > 0);
    try std.testing.expect(context.stats.chosen_elements_max <= context.stats.declared_elements_max);
    try std.testing.expectEqual((try workspacePlan(3, 2, 1, 0)).total_bytes, context.stats.workspace_peak_bytes);
}

test "diagnostic context separates scalar and vector RHS clocks" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const Clock = struct {
        value: u64 = 100,
        step: u64 = 7,
        fn now(raw: ?*anyopaque) u64 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const result = self.value;
            self.value += self.step;
            return result;
        }
    };
    var clock: Clock = .{};
    var context: Context = .{ .clock_context = &clock, .clock_fn = Clock.now };
    const backend = context.valueBackend();
    var scalar = [_]f64{ 0, 0 };
    try backend.fit_fn(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3, 5 }, 3, 2, 1, 0, .{}, &scalar);
    clock.step = 13;
    var vector = [_]f64{ 0, 0, 0, 0 };
    try backend.fit_fn(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 4, 3, 6, 5, 10 }, 3, 2, 2, 0, .{}, &vector);
    try std.testing.expectEqual(@as(usize, 2), context.stats.calls);
    try std.testing.expectEqual(@as(usize, 1), context.stats.scalar_rhs_calls);
    try std.testing.expectEqual(@as(usize, 1), context.stats.vector_rhs_calls);
    try std.testing.expectEqual(@as(u64, 7), context.stats.scalar_rhs_ns);
    try std.testing.expectEqual(@as(u64, 13), context.stats.vector_rhs_ns);
}

test "Accelerate value backend is analytic and cancellation atomic" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const a = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const y = [_]f64{ 2, 3, 5 };
    var out = [_]f64{ -9, -9 };
    const backend = valueBackend();
    try backend.fit_fn(backend.context, std.testing.allocator, &a, &y, 3, 2, 1, 0, .{}, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 2), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), out[1], 1e-12);
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    out = .{ -7, -7 };
    try std.testing.expectError(error.Cancelled, backend.fit_fn(backend.context, std.testing.allocator, &a, &y, 3, 2, 1, 0, .{ .check_fn = Cancel.check }, &out));
    try std.testing.expectEqual([2]f64{ -7, -7 }, out);
}

test "Accelerate value backend preserves rank failure and late cancellation atomicity" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const backend = valueBackend();
    var singular = [_]f64{ -3, -3 };
    try std.testing.expectError(error.SingularValueFit, backend.fit_fn(backend.context, std.testing.allocator, &.{ 0, 0, 0, 0 }, &.{ 1, 2 }, 2, 2, 1, 0, .{}, &singular));
    try std.testing.expectEqual([2]f64{ -3, -3 }, singular);

    const State = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 7) return error.Cancelled;
        }
    };
    var state: State = .{};
    var cancelled = [_]f64{ -5, -5 };
    try std.testing.expectError(error.Cancelled, backend.fit_fn(backend.context, std.testing.allocator, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3, 5 }, 3, 2, 1, 0, .{ .context = &state, .check_fn = State.check }, &cancelled));
    try std.testing.expectEqual([2]f64{ -5, -5 }, cancelled);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    if (comptime !lapack.supported) return;
    const a = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const y = [_]f64{ 2, 3, 5 };
    var out = [_]f64{ -5, -5 };
    const backend = valueBackend();
    backend.fit_fn(backend.context, allocator, &a, &y, 3, 2, 1, 0, .{}, &out) catch |err| {
        try std.testing.expectEqual([2]f64{ -5, -5 }, out);
        return err;
    };
}

test "Accelerate value backend allocation failures never publish" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
