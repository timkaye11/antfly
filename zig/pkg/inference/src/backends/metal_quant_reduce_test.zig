// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Wide-row fixtures for the single-row quantized `1x_reduce` Metal kernels.
//! Those kernels stripe a row's input blocks across SIMD groups. Dispatching
//! fewer groups than the shader's stride silently drops blocks once a row is
//! wider than 512 values, which 256-wide fixtures cannot observe.
const std = @import("std");
const build_options = @import("build_options");
const mr = @import("metal_runtime.zig");
const gguf = @import("../gguf/root.zig");

const Known = gguf.tensor_types.KnownTensorType;
const Case = struct { format: mr.MetalQuantFormat, kind: Known };
const cases = [_]Case{
    .{ .format = .q4_1, .kind = .Q4_1 },
    .{ .format = .q4_k, .kind = .Q4_K },
    .{ .format = .q5_0, .kind = .Q5_0 },
    .{ .format = .q5_1, .kind = .Q5_1 },
    .{ .format = .q5_k, .kind = .Q5_K },
    .{ .format = .q6_k, .kind = .Q6_K },
    .{ .format = .q8_1, .kind = .Q8_1 },
    .{ .format = .q8_k, .kind = .Q8_K },
    .{ .format = .iq4_nl, .kind = .IQ4_NL },
    .{ .format = .iq4_xs, .kind = .IQ4_XS },
    .{ .format = .mxfp4, .kind = .MXFP4 },
};

// Arbitrary quantized payload with finite, block-dependent scales, so a
// dropped or duplicated block always changes the row sum.
fn fillBlock(kind: Known, block: []u8, seed: usize) void {
    var state: u64 = 0x9e3779b97f4a7c15 ^ (@as(u64, seed) *% 0xbf58476d1ce4e5b9);
    for (block) |*byte| {
        state = state *% 6364136223846793005 +% 1442695040888963407;
        byte.* = @truncate(state >> 33);
    }
    const scale: f32 = 0.004 * @as(f32, @floatFromInt(1 + seed % 13));
    const half: u16 = @bitCast(@as(f16, @floatCast(scale)));
    switch (kind) {
        .Q5_0, .IQ4_NL, .IQ4_XS => std.mem.writeInt(u16, block[0..2], half, .little),
        .Q4_1, .Q5_1, .Q8_1, .Q4_K, .Q5_K => {
            std.mem.writeInt(u16, block[0..2], half, .little);
            std.mem.writeInt(u16, block[2..4], @bitCast(@as(f16, @floatCast(scale * 0.5))), .little);
        },
        .Q6_K => std.mem.writeInt(u16, block[208..210], half, .little),
        .Q8_K => std.mem.writeInt(u32, block[0..4], @bitCast(scale * 0.125), .little),
        // E8M0 exponent: 2^(e - 128), kept well inside the finite range.
        .MXFP4 => block[0] = @intCast(118 + seed % 8),
        else => unreachable,
    }
}

test "metal quant reduce single-row kernels sum every block of wide rows" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    const runtime = mr.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer mr.termite_metal_decode_runtime_destroy(runtime);
    // Odd output count: the last threadgroup of every 2- and 4-row kernel is ragged.
    const out_dim: usize = 7;
    const batch_rows: usize = 3;
    for (cases) |case| {
        const tensor_type: gguf.tensor_types.TensorType = .{ .known = case.kind };
        const values = gguf.tensor_types.valuesPerBlock(tensor_type).?;
        const block_bytes = gguf.tensor_types.bytesPerBlock(tensor_type).?;
        for ([_]usize{ 256, 512, 1024, 2048 }) |in_dim| {
            const row_blocks = in_dim / values;
            const raw = try a.alloc(u8, out_dim * row_blocks * block_bytes);
            defer a.free(raw);
            for (0..out_dim * row_blocks) |block| fillBlock(case.kind, raw[block * block_bytes ..][0..block_bytes], block);
            const input = try a.alloc(f32, batch_rows * in_dim);
            defer a.free(input);
            for (input, 0..) |*x, i| x.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7 + i / in_dim) % 31)) - 15)) / 32;
            const weights = try a.alloc(f32, in_dim);
            defer a.free(weights);
            var expected: [batch_rows * out_dim]f64 = undefined;
            var magnitude: [batch_rows * out_dim]f64 = undefined;
            for (0..out_dim) |o| {
                try gguf.quant_codec.dequantizeRow(tensor_type, raw, in_dim, o, weights);
                for (0..batch_rows) |r| {
                    var sum: f64 = 0;
                    var absolute: f64 = 0;
                    for (weights, input[r * in_dim ..][0..in_dim]) |w, x| {
                        try std.testing.expect(std.math.isFinite(w));
                        sum += @as(f64, w) * x;
                        absolute += @abs(@as(f64, w) * x);
                    }
                    expected[r * out_dim + o] = sum;
                    magnitude[r * out_dim + o] = absolute;
                }
            }
            const format = @backingInt(case.format);
            try std.testing.expectEqual(@as(c_int, 0), mr.termite_metal_decode_runtime_prepare_quantized_linear_slot(runtime, format, 0, raw.ptr, raw.len, in_dim, out_dim));
            // rows == 1 selects the SIMD-group reduction; rows == 3 the scalar/batch kernel.
            var output: [batch_rows * out_dim]f32 = undefined;
            for ([_]usize{ 1, batch_rows }) |rows| {
                @memset(&output, std.math.nan(f32));
                try std.testing.expectEqual(@as(c_int, 0), mr.termite_metal_decode_runtime_apply_quantized_linear_slot_host(runtime, format, 0, input.ptr, rows, in_dim, out_dim, &output));
                for (output[0 .. rows * out_dim], expected[0 .. rows * out_dim], magnitude[0 .. rows * out_dim], 0..) |actual, want, scale, i| {
                    // FP32 reduction order differs per kernel; bound by the L1 mass.
                    const tolerance = 1e-4 + scale * 2e-5;
                    if (!(@abs(@as(f64, actual) - want) <= tolerance)) {
                        std.debug.print("metal quant reduce mismatch: format={s} in_dim={d} rows={d} index={d} actual={d} want={d} tolerance={d}\n", .{ @tagName(case.format), in_dim, rows, i, actual, want, tolerance });
                        return error.TestExpectedApproxEqAbs;
                    }
                }
            }
        }
    }
}
