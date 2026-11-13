// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const lapack = @import("metal_attention_match_lapack.zig");
const solver = @import("../runtime/kv/attention_match.zig");

fn selectionWorkspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
    return 0;
}

fn compareSelection(input: solver.MatrixInput, target: usize) !void {
    const allocator = std.testing.allocator;
    var config: solver.Config = .{
        .target_rows = target,
        .selection = .residual_omp,
        .mass_solver = .active_set,
        .backend = .{ .workspace_fn = selectionWorkspace, .values_fn = lapack.values },
    };
    var reference = try solver.fitMatrices(allocator, input, config);
    defer reference.deinit();
    config.backend.?.select_fn = lapack.select;
    var accelerated = try solver.fitMatrices(allocator, input, config);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(usize, reference.source_indices, accelerated.source_indices);
    for (reference.bias, accelerated.bias) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    for (reference.values, accelerated.values) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
}

fn compareRawSelection(input: solver.MatrixInput, rhs: []const f64, target: usize) !void {
    var cfg: solver.Config = .{ .target_rows = target, .selection = .residual_omp, .mass_solver = .active_set };
    var reference = try solver.fitMatricesMassTargetPolicyForTest(std.testing.allocator, input, cfg, rhs, .raw);
    defer reference.deinit();
    cfg.backend = .{ .workspace_fn = selectionWorkspace, .values_fn = lapack.values, .raw_select_fn = lapack.selectRawTarget };
    var accelerated = try solver.fitMatricesMassTargetPolicyForTest(std.testing.allocator, input, cfg, rhs, .raw);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(usize, reference.source_indices, accelerated.source_indices);
    for (reference.bias, accelerated.bias) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    for (reference.values, accelerated.values) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
}

test "attention_match Accelerate raw rho selector matches native asymmetric and dependent matrices" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const p = [_]f64{
        0.45, 0.20, 0.35,
        0.35, 0.60, 0.05,
        0.40, 0.40, 0.20,
        0.10, 0.10, 0.80,
        0.60, 0.20, 0.20,
    };
    const y = [_]f64{ 1, 2, 3, 4, 5 };
    try compareRawSelection(.{ .probabilities = &p, .targets = &y, .rows = 3, .dim = 1, .query_rows = 5 }, &.{ 0.2, 2, 0.7, 1.3, 0 }, 2);
    const dependent = [_]f64{ 0.2, 0.2, 0.6, 0.4, 0.4, 0.2, 0.1, 0.1, 0.8 };
    try compareRawSelection(.{ .probabilities = &dependent, .targets = &.{ 1, 2, 3 }, .rows = 3, .dim = 1, .query_rows = 3 }, &.{ 1, 0.5, 2 }, 2);
}

test "attention_match raw rho callback is explicitly gated and rejects malformed output" {
    const Trap = struct {
        var calls: usize = 0;
        fn raw(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: []const f64, _: usize, n: usize, _: []const usize, indices: []usize, weights: []f64, _: solver.Config) !usize {
            calls += 1;
            @memset(indices, n);
            @memset(weights, 1);
            return 0;
        }
    };
    const input: solver.MatrixInput = .{ .probabilities = &.{ 0.8, 0.2, 0.1, 0.9 }, .targets = &.{ 1, 2 }, .rows = 2, .dim = 1, .query_rows = 2 };
    const backend: solver.Backend = .{ .workspace_fn = selectionWorkspace, .raw_select_fn = Trap.raw };
    Trap.calls = 0;
    var normalized = try solver.fitMatrices(std.testing.allocator, input, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend });
    normalized.deinit();
    try std.testing.expectEqual(@as(usize, 0), Trap.calls);
    try std.testing.expectError(error.InvalidSelectionResult, solver.fitMatricesMassTargetPolicyForTest(std.testing.allocator, input, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend }, &.{ 1, 1 }, .raw));
    try std.testing.expectEqual(@as(usize, 1), Trap.calls);
}

test "attention_match Accelerate raw rho selector preserves cancellation and allocation failure" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const p = [_]f64{ 0.8, 0.2, 0.1, 0.9 };
    const importance = [_]f64{ 0.65, 0.85 };
    var indices: [1]usize = undefined;
    var weights: [1]f64 = undefined;
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const cancelled: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .control = .{ .check_fn = Cancel.check } };
    try std.testing.expectError(error.Cancelled, lapack.selectRawTarget(null, std.testing.allocator, &p, &importance, &.{ 1, 0.5 }, 2, 2, &.{}, &indices, &weights, cancelled));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, lapack.selectRawTarget(null, failing.allocator(), &p, &importance, &.{ 1, 0.5 }, 2, 2, &.{}, &indices, &weights, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set }));
}

test "attention_match Metal FP64 selection matches native asymmetric matrices and dependent columns" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const q = 33;
    const n = 12;
    const d = 3;
    var p: [q * n]f64 = undefined;
    var y: [q * d]f64 = @splat(0);
    for (0..q) |row| {
        var sum: f64 = 0;
        for (0..n) |column| {
            const value = 1.2 + @cos(@as(f64, @floatFromInt((row + 1) * (column + 2))) * 0.37) + @as(f64, @floatFromInt(column)) * 0.08;
            p[row * n + column] = value;
            sum += value;
        }
        for (p[row * n ..][0..n]) |*value| value.* /= sum;
        for (0..n) |column| for (0..d) |component| {
            y[row * d + component] += p[row * n + column] * @sin(@as(f64, @floatFromInt(column * d + component)) * 0.19);
        };
    }
    try compareSelection(.{ .probabilities = &p, .targets = &y, .rows = n, .dim = d, .query_rows = q }, 4);
    // Identical columns make incremental QR singular. Active-set fallback and
    // the native final KKT check must still accept the valid mass solution.
    const uniform: [16 * 8]f64 = @splat(0.125);
    const targets: [16]f64 = @splat(2.5);
    try compareSelection(.{ .probabilities = &uniform, .targets = &targets, .rows = 8, .dim = 1, .query_rows = 16 }, 4);
}

test "attention_match Metal FP64 selection preserves mandatory training argmax seeds" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const input: solver.MatrixInput = .{ .probabilities = &.{ 0.2, 0.8, 0, 0.2, 0, 0.8 }, .targets = &.{ 1, 2 }, .rows = 3, .dim = 1, .query_rows = 2 };
    const base: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true };
    var reference = try solver.fitMatrices(std.testing.allocator, input, base);
    defer reference.deinit();
    var accelerated_cfg = base;
    accelerated_cfg.backend = .{ .workspace_fn = selectionWorkspace, .values_fn = lapack.values, .select_fn = lapack.select };
    var accelerated = try solver.fitMatrices(std.testing.allocator, input, accelerated_cfg);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, accelerated.source_indices);
    try std.testing.expectEqualSlices(usize, reference.source_indices, accelerated.source_indices);
    for (reference.bias, accelerated.bias) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    for (reference.values, accelerated.values) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    try std.testing.expect(reference.metrics.nnls_sweeps > 0 and accelerated.metrics.nnls_sweeps > 0);
}

test "attention_match Metal FP64 selection preserves mandatory training top2 seeds" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const input: solver.MatrixInput = .{ .probabilities = &.{ 0.6, 0.3, 0.1, 0, 0.6, 0, 0.1, 0.3 }, .targets = &.{ 1, 2 }, .rows = 4, .dim = 1, .query_rows = 2 };
    const base: solver.Config = .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true, .mandatory_training_top_k = 2 };
    var reference = try solver.fitMatrices(std.testing.allocator, input, base);
    defer reference.deinit();
    var accelerated_cfg = base;
    accelerated_cfg.backend = .{ .workspace_fn = selectionWorkspace, .values_fn = lapack.values, .select_fn = lapack.select };
    var accelerated = try solver.fitMatrices(std.testing.allocator, input, accelerated_cfg);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, accelerated.source_indices);
    try std.testing.expectEqualSlices(usize, reference.source_indices, accelerated.source_indices);
    for (reference.bias, accelerated.bias) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    for (reference.values, accelerated.values) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-9);
    try std.testing.expect(reference.metrics.nnls_sweeps > 0 and accelerated.metrics.nnls_sweeps > 0);
}

test "attention_match Metal FP64 selection preserves the native workspace including one query" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const p = [_]f64{ 0.125, 0.125, 0.25, 0.5 };
    const input: solver.MatrixInput = .{ .probabilities = &p, .targets = &.{3}, .rows = 4, .dim = 1, .query_rows = 1 };
    try compareSelection(input, 1);
    const bytes = try solver.matrixWorkspaceBytes(4, 1, 1, 1);
    const memory = try std.testing.allocator.alignedAlloc(u8, .of(f64), bytes);
    defer std.testing.allocator.free(memory);
    var fixed = std.heap.FixedBufferAllocator.init(memory);
    var result = try solver.fitMatrices(fixed.allocator(), input, .{
        .target_rows = 1,
        .selection = .residual_omp,
        .mass_solver = .active_set,
        .backend = .{ .workspace_fn = selectionWorkspace, .values_fn = lapack.values, .select_fn = lapack.select },
    });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3) / (1 + 1e-8), result.values[0], 1e-10);
}

test "attention_match Metal FP64 selection checks dimensions importance and cancellation" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.4, 0.3 };
    var importance: [4]f64 = @splat(0);
    for (0..3) |i| for (0..4) |j| {
        importance[j] += p[i * 4 + j] * p[i * 4 + j];
    };
    var indices: [2]usize = undefined;
    var weights: [2]f64 = undefined;
    const config: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set };
    try std.testing.expectError(error.InvalidAttentionShape, lapack.select(null, std.testing.failing_allocator, &p, &importance, 0, 4, &.{}, &indices, &weights, config));
    var poisoned = importance;
    poisoned[0] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteAttention, lapack.select(null, std.testing.failing_allocator, &p, &poisoned, 3, 4, &.{}, &indices, &weights, config));
    const Check = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 3) return error.Cancelled;
        }
    };
    var state: Check = .{};
    var cancellable = config;
    cancellable.control = .{ .context = &state, .check_fn = Check.check };
    try std.testing.expectError(error.Cancelled, lapack.select(null, std.testing.allocator, &p, &importance, 3, 4, &.{}, &indices, &weights, cancellable));
    try std.testing.expectEqual(@as(usize, 3), state.calls);
}

test "attention_match Metal FP64 selection releases every failed allocation" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const Case = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const p = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.2, 0.1, 0.4, 0.3 };
            var importance: [4]f64 = @splat(0);
            for (0..3) |i| for (0..4) |j| {
                importance[j] += p[i * 4 + j] * p[i * 4 + j];
            };
            var indices: [2]usize = undefined;
            var weights: [2]f64 = undefined;
            _ = try lapack.select(null, allocator, &p, &importance, 3, 4, &.{}, &indices, &weights, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}

fn compare(a: std.mem.Allocator, q: usize, m: usize, d: usize, ridge: f64) !void {
    const matrix = try a.alloc(f64, q * m);
    defer a.free(matrix);
    const rhs = try a.alloc(f64, q * d);
    defer a.free(rhs);
    const expected = try a.alloc(f64, m * d);
    defer a.free(expected);
    const actual = try a.alloc(f64, m * d);
    defer a.free(actual);
    for (matrix, 0..) |*v, index| v.* = @sin(@as(f64, @floatFromInt(index *% 7919 +% 17)) * 0.021) + if (index / m == index % m) @as(f64, 2) else 0;
    for (rhs, 0..) |*v, index| v.* = @cos(@as(f64, @floatFromInt(index)) * 0.037);
    const started = std.Io.Clock.awake.now(std.testing.io);
    try solver.fitValuesReferenceForTest(a, matrix, rhs, q, m, d, ridge, .{}, expected);
    const middle = std.Io.Clock.awake.now(std.testing.io);
    try lapack.values(null, a, matrix, rhs, q, m, d, ridge, .{}, actual);
    const ended = std.Io.Clock.awake.now(std.testing.io);
    var error_norm: f64 = 0;
    var expected_norm: f64 = 0;
    for (actual, expected) |got, want| {
        error_norm = std.math.hypot(error_norm, got - want);
        expected_norm = std.math.hypot(expected_norm, want);
        try std.testing.expectApproxEqAbs(want, got, 1e-8 * @max(1, @abs(want)));
    }
    std.debug.print("lapack_compare q={d} m={d} d={d} ridge={d} reference_ns={d} accelerated_ns={d} relative_l2={d}\n", .{
        q, m, d, ridge, middle.nanoseconds - started.nanoseconds, ended.nanoseconds - middle.nanoseconds, error_norm / @max(expected_norm, 1e-12),
    });
}

test "attention_match Metal FP64 value solve handles both rectangular orientations and ridge" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const a = std.testing.allocator;
    var out: [2]f64 = undefined;
    try lapack.values(null, a, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3, 5 }, 3, 2, 1, 0, .{}, &out);
    try std.testing.expectApproxEqAbs(@as(f64, 2), out[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), out[1], 1e-12);
    try compare(a, 2, 5, 7, 1e-3);
    try compare(a, 11, 3, 5, 0);
    try compare(a, 512, 32, 16, 1e-8);
}

test "attention_match Metal FP64 value solve stays inside the native QR workspace" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const q = 128;
    const m = 12;
    const d = 17;
    const bytes = ((q + m) * (m + d) + (q + m) + @max(m, d)) * @sizeOf(f64);
    const buffer = try std.testing.allocator.alignedAlloc(u8, .of(f64), bytes);
    defer std.testing.allocator.free(buffer);
    var fixed = std.heap.FixedBufferAllocator.init(buffer);
    var matrix: [q * m]f64 = @splat(0.1);
    var rhs: [q * d]f64 = @splat(2);
    var out: [m * d]f64 = undefined;
    try lapack.values(null, fixed.allocator(), &matrix, &rhs, q, m, d, 1e-8, .{}, &out);
    try std.testing.expect(fixed.end_index == 0);
}

test "attention_match Metal FP64 value solve rejects poison and cancellation and releases failed allocations" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    var out: [2]f64 = undefined;
    try std.testing.expectError(error.NonFiniteAttention, lapack.values(null, std.testing.failing_allocator, &.{ std.math.nan(f64), 0, 0, 1 }, &.{ 1, 2 }, 2, 2, 1, 0, .{}, &out));
    const Case = struct {
        fn cancel(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
        fn run(a: std.mem.Allocator) !void {
            var result: [2]f64 = undefined;
            try lapack.values(null, a, &.{ 1, 0, 0, 1 }, &.{ 2, 3 }, 2, 2, 1, 1e-8, .{}, &result);
        }
    };
    try std.testing.expectError(error.Cancelled, lapack.values(null, std.testing.failing_allocator, &.{ 1, 0, 0, 1 }, &.{ 1, 2 }, 2, 2, 1, 0, .{ .check_fn = Case.cancel }, &out));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}

test "attention_match Metal FP64 active-set value callback preserves native mass solver and independent KKT" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const Backend = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
    };
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.4, 0.25, 0.15, 0.35, 0.25, 0.4, 0.1, 0.2, 0.3, 0.2, 0.4, 0.3, 0.1, 0.15, 0.3, 0.2, 0.35 };
    const y = [_]f64{ 1, 2, 3, 2, 1, 3 };
    const input: solver.MatrixInput = .{ .probabilities = &p, .targets = &y, .rows = 4, .dim = 1, .query_rows = 6 };
    var cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set };
    var native = try solver.fitMatrices(std.testing.allocator, input, cfg);
    defer native.deinit();
    cfg.backend = .{ .workspace_fn = Backend.workspace, .values_fn = lapack.values };
    var accelerated = try solver.fitMatrices(std.testing.allocator, input, cfg);
    defer accelerated.deinit();
    try std.testing.expectEqualSlices(usize, native.source_indices, accelerated.source_indices);
    for (native.values, accelerated.values) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-10);
    for (native.bias, accelerated.bias) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-10);
}

test "attention_match Metal FP64 value performance uses the E2B query geometry" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    try compare(std.testing.allocator, 10240, 256, 1, 0);
    try compare(std.testing.allocator, 10240, 256, 512, 1e-8);
}

test "attention_match Metal FP64 near-collinear ridge solve matches the independent scalar oracle" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const q = 257;
    const m = 8;
    const d = 3;
    var matrix: [q * m]f64 = undefined;
    var rhs: [q * d]f64 = @splat(0);
    for (0..q) |row| for (0..m) |column| {
        matrix[row * m + column] = @sin(@as(f64, @floatFromInt(row)) * 0.001) +
            if (row % m == column) @as(f64, 1e-4) else 0;
        for (0..d) |output| rhs[row * d + output] += matrix[row * m + column] * @as(f64, @floatFromInt(column + output + 1));
    };
    var expected: [m * d]f64 = undefined;
    var actual: [m * d]f64 = undefined;
    try solver.fitValuesReferenceForTest(std.testing.allocator, &matrix, &rhs, q, m, d, 1e-8, .{}, &expected);
    try lapack.values(null, std.testing.allocator, &matrix, &rhs, q, m, d, 1e-8, .{}, &actual);
    for (actual, expected) |got, want| try std.testing.expectApproxEqAbs(want, got, 1e-8);
}

test "attention_match Metal FP64 cancellation after factorization frees every buffer" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const State = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 3) return error.Cancelled;
        }
    };
    var state: State = .{};
    var out: [2]f64 = undefined;
    try std.testing.expectError(error.Cancelled, lapack.values(null, std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 3 }, 2, 2, 1, 1e-8, .{ .context = &state, .check_fn = State.check }, &out));
    try std.testing.expectEqual(@as(usize, 3), state.calls);
}

test "attention_match Metal FP64 accelerator output cannot bypass the native direct KKT check" {
    if (comptime !lapack.supported) return error.SkipZigTest;
    const Bad = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn select(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: []const usize, indices: []usize, weights: []f64, _: solver.Config) !usize {
            for (indices, 0..) |*index, i| index.* = i;
            @memset(weights, 0);
            return 0;
        }
        fn solve(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, _: solver.Control, out: []f64) !void {
            @memset(out, 0);
        }
    };
    const input: solver.MatrixInput = .{ .probabilities = &.{ 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0.4, 0.25, 0.15, 0.35, 0.25 }, .targets = &.{ 1, 2, 3 }, .rows = 4, .dim = 1, .query_rows = 3 };
    try std.testing.expectError(error.SingularValueFit, solver.fitMatrices(std.testing.allocator, input, .{
        .target_rows = 2,
        .selection = .residual_omp,
        .mass_solver = .active_set,
        .max_sweeps = 4,
        .backend = .{ .workspace_fn = Bad.workspace, .select_fn = Bad.select, .values_fn = Bad.solve },
    }));
}
