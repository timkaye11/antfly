// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");

/// A rank-2 dot over independently strided operands, retaining the scalar native dot's F64 left-to-right
/// accumulation for every output. SIMD lanes compute independent columns;
/// there is no reduction across lanes or intermediate F32 rounding.
/// Packing eight output columns at a time bounds scratch to 64 * k bytes and
/// reuses their F64 conversion across all input rows.
pub fn rank2(
    allocator: std.mem.Allocator,
    lhs: []const f32,
    lhs_row_stride: usize,
    lhs_k_stride: usize,
    rhs: []const f32,
    rhs_column_stride: usize,
    rhs_k_stride: usize,
    output: []f32,
    m: usize,
    k: usize,
    n: usize,
) !void {
    @setFloatMode(.strict);
    const lanes = 8;
    const V = @Vector(lanes, f64);
    const panel = try allocator.alloc(f64, try std.math.mul(usize, k, lanes));
    defer allocator.free(panel);
    var column: usize = 0;
    while (column < n) : (column += lanes) {
        const valid = @min(lanes, n - column);
        for (0..k) |ki| {
            inline for (0..lanes) |lane| {
                panel[ki * lanes + lane] = if (lane < valid) rhs[(column + lane) * rhs_column_stride + ki * rhs_k_stride] else 0.0;
            }
        }
        var row: usize = 0;
        while (row + 1 < m) : (row += 2) {
            var acc0: V = @splat(0.0);
            var acc1: V = @splat(0.0);
            for (0..k) |ki| {
                const weights: V = panel[ki * lanes ..][0..lanes].*;
                const a0: V = @splat(@as(f64, lhs[row * lhs_row_stride + ki * lhs_k_stride]));
                const a1: V = @splat(@as(f64, lhs[(row + 1) * lhs_row_stride + ki * lhs_k_stride]));
                acc0 += a0 * weights;
                acc1 += a1 * weights;
            }
            inline for (0..lanes) |lane| {
                if (lane < valid) {
                    output[row * n + column + lane] = @floatCast(acc0[lane]);
                    output[(row + 1) * n + column + lane] = @floatCast(acc1[lane]);
                }
            }
        }
        if (row < m) {
            var acc: V = @splat(0.0);
            for (0..k) |ki| {
                const weights: V = panel[ki * lanes ..][0..lanes].*;
                const a: V = @splat(@as(f64, lhs[row * lhs_row_stride + ki * lhs_k_stride]));
                acc += a * weights;
            }
            inline for (0..lanes) |lane| {
                if (lane < valid) output[row * n + column + lane] = @floatCast(acc[lane]);
            }
        }
    }
}

test "gemma4 native packed dot preserves scalar F64 order across strides and tails" {
    const allocator = std.testing.allocator;
    const m = 3;
    const k = 257;
    const n = 13;
    var lhs: [m * k]f32 = undefined;
    var transposed: [m * k]f32 = undefined;
    var rhs: [n * k]f32 = undefined;
    var rhs_transposed: [n * k]f32 = undefined;
    var output: [m * n]f32 = undefined;
    for (&lhs, 0..) |*v, i| {
        const value: f32 = @floatFromInt(@as(i32, @intCast((i * 17 + 3) % 127)) - 63);
        v.* = value / 19.0;
    }
    for (&rhs, 0..) |*v, i| {
        const value: f32 = @floatFromInt(@as(i32, @intCast((i * 29 + 11) % 251)) - 125);
        v.* = value / 23.0;
    }
    // This product difference is exactly 1 in F64 but rounds to 0 in F32.
    @memset(lhs[0..k], 0.0);
    lhs[0] = 4097.0;
    lhs[1] = 4096.0;
    rhs[0] = 4097.0;
    rhs[1] = -4098.0;
    for (0..n) |column| for (0..k) |ki| {
        rhs_transposed[ki * n + column] = rhs[column * k + ki];
    };
    for (0..m) |row| for (0..k) |ki| {
        transposed[ki * m + row] = lhs[row * k + ki];
    };
    for ([_]bool{ false, true }) |strided| {
        const input: []const f32 = if (strided) &transposed else &lhs;
        for ([_]bool{ false, true }) |rhs_strided| {
            try rank2(allocator, input, if (strided) 1 else k, if (strided) m else 1, if (rhs_strided) &rhs_transposed else &rhs, if (rhs_strided) 1 else k, if (rhs_strided) n else 1, &output, m, k, n);
            for (0..m) |row| for (0..n) |column| {
                var expected: f64 = 0.0;
                for (0..k) |ki| expected += @as(f64, lhs[row * k + ki]) * @as(f64, rhs[column * k + ki]);
                try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, @floatCast(expected)))), @as(u32, @bitCast(output[row * n + column])));
            };
            try std.testing.expectEqual(@as(f32, 1.0), output[0]);
        }
    }
}
