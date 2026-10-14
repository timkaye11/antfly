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

//! Streamed matrix bindings. Unsupported packed formats use bounded FP32 tiles.
const std = @import("std");
const shared = @import("qwen4exp_kernels.zig");
const mr = @import("metal_runtime.zig");
const gguf = @import("../gguf/root.zig");
pub const Buffer = shared.Buffer;
pub const Ref = shared.Ref;
pub const Params = shared.Params;
pub const Profile = shared.Profile;
pub const max_chunk = 1024;
pub const tile_rows = 512;
pub const quant_grid_bytes = gguf.quant_codec.iq3_s_grid.len * @sizeOf(u32);
pub const tile_bytes = tile_rows * 17408 * 4;
pub const tile_result_bytes = max_chunk * tile_rows * 4;
pub const Kernels = struct {
    core: shared.Kernels,
    tile: Buffer,
    tile_result: Buffer,
    quant_grid: Buffer,
    gpu_dequant: bool = true,
    profile_enabled: bool = false,
    gpu_wait_ns: u64 = 0,
    gpu_ns: u64 = 0,
    gpu_batches: u64 = 0,
    cancelled: ?*const std.atomic.Value(bool) = null,
    cpu_dequant_bytes: u64 = 0,
    cpu_dequant_ns: u64 = 0,
    pub fn init() !Kernels {
        var core = try shared.Kernels.initSource(@embedFile("qwen35_streamed.metal"), "qwen35_streamed");
        errdefer core.deinit();
        core.batching_enabled = true;
        var tile = try Buffer.init(core.runtime, tile_bytes);
        errdefer tile.deinit();
        var result = try Buffer.init(core.runtime, tile_result_bytes);
        errdefer result.deinit();
        var grid = try Buffer.init(core.runtime, quant_grid_bytes);
        @memcpy(grid.bytes[0..quant_grid_bytes], std.mem.sliceAsBytes(&gguf.quant_codec.iq3_s_grid));
        return .{ .core = core, .tile = tile, .tile_result = result, .quant_grid = grid, .gpu_dequant = std.c.getenv("ANTFLY_QWEN35_CPU_DEQUANT") == null, .profile_enabled = std.c.getenv("ANTFLY_QWEN35_PROFILE") != null };
    }
    pub fn deinit(self: *Kernels) void {
        self.core.deinit();
        self.tile.deinit();
        self.tile_result.deinit();
        self.quant_grid.deinit();
    }
    // Drains remain mandatory before releasing a layer or CPU-visible buffers.
    pub fn endBatch(self: *Kernels) !void {
        const pending = self.core.batch_active or self.core.batch_submitted;
        const tick = if (self.profile_enabled) shared.Profile.now() else 0;
        try self.core.endBatch();
        if (self.profile_enabled and pending) {
            self.gpu_wait_ns += shared.Profile.now() - tick;
            self.gpu_ns += mr.termite_metal_decode_runtime_last_frame_gpu_nanos(self.core.runtime);
            self.gpu_batches += 1;
        }
    }
    pub fn run(self: *Kernels, p: Params, ins: []const Ref, out: Ref) !void {
        try self.core.run(p, ins, out);
    }
    pub fn matmul(self: *Kernels, kind: gguf.tensor_types.TensorType, raw: []const u8, weight: Ref, input_dim: usize, output_dim: usize, rows: usize, input: Ref, output: Ref) !void {
        if (rows == 0 or rows > max_chunk or input_dim == 0 or output_dim == 0 or input.bytes < rows * input_dim * 4 or output.bytes < rows * output_dim * 4) return error.MetalBufferRange;
        const expected = gguf.tensor_types.byteLen(kind, &.{ input_dim, output_dim }) orelse return error.UnsupportedQwen35Format;
        if (raw.len != expected or weight.bytes != expected) return error.MetalBufferRange;
        // Packed vector kernels are efficient for decode. For prefill, reuse
        // one bounded FP32 tile across every prompt row with matrix GEMM.
        const quant: ?mr.MetalQuantFormat = if (rows >= 8) null else switch (kind.raw()) {
            8 => .q8_0,
            11 => .q3_k,
            12 => .q4_k,
            13 => .q5_k,
            14 => .q6_k,
            0, 20, 21, 23 => null,
            else => return error.UnsupportedQwen35Format,
        };
        const tiled_quant: ?mr.MetalQuantFormat = if (rows < 8) null else switch (kind.raw()) {
            8 => .q8_0,
            12 => .q4_k,
            13 => .q5_k,
            14 => .q6_k,
            else => null,
        };
        const own_tiled = self.gpu_dequant and (kind.raw() == 11 or kind.raw() == 21 or kind.raw() == 23);
        if (quant) |q| {
            // Rebind every dispatch; slot 0 aliases the current encoded bytes and owns no copy.
            if (shared.termite_metal_qwen4exp_prepare_linear(self.core.runtime, weight.handle, @backingInt(q), weight.offset, weight.bytes, input_dim, output_dim) != 0) return error.Qwen35QuantPrepareFailed;
            var row: usize = 0;
            while (row < rows) {
                const count = @min(32, rows - row);
                if (mr.termite_metal_decode_runtime_apply_quantized_linear_slot_device(self.core.runtime, @backingInt(q), 0, input.handle, input.offset + row * input_dim * 4, count, input_dim, output_dim, output.handle, output.offset + row * output_dim * 4) != 0) return error.Qwen35MatmulFailed;
                row += count;
            }
        } else if (rows >= 8 and (kind.raw() == 20 or tiled_quant != null or own_tiled)) {
            // GPU commands own the staging tile until completion. Each tile is
            // decoded once and reused across the prompt by FP32 matrix GEMM.
            if (tiled_quant) |q| {
                if (shared.termite_metal_qwen4exp_prepare_linear(self.core.runtime, weight.handle, @backingInt(q), weight.offset, weight.bytes, input_dim, output_dim) != 0) return error.Qwen35QuantPrepareFailed;
            }
            var start: usize = 0;
            while (start < output_dim) {
                if (self.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
                const count = @min(tile_rows, output_dim - start);
                if (count * input_dim * 4 > self.tile.logical_bytes) return error.Qwen35TileTooLarge;
                if (tiled_quant) |q| {
                    if (mr.termite_metal_decode_runtime_quant_copy_linear_slot_to_f32_device(self.core.runtime, @backingInt(q), 0, start, count, input_dim, output_dim, 1, self.tile.handle, 0) != 0) return error.Qwen35MatmulFailed;
                } else if (own_tiled) {
                    try self.run(.{ .op = 4, .mode = kind.raw(), .n = @intCast(count * input_dim), .dim = @intCast(input_dim), .index = @intCast(start) }, &.{ weight, self.quant_grid.ref() }, self.tile.ref());
                } else {
                    try self.run(.{ .op = 3, .n = @intCast(count * input_dim), .dim = @intCast(input_dim), .index = @intCast(start) }, &.{weight}, self.tile.ref());
                }
                if (mr.termite_metal_decode_runtime_dot_general_2d_f32_device(self.core.runtime, input.handle, input.offset, self.tile.handle, 0, rows, count, input_dim, 1, self.tile_result.handle, 0) != 0) return error.Qwen35MatmulFailed;
                try self.run(.{ .op = 8, .n = @intCast(rows * count), .groups = @intCast(count), .stride = @intCast(output_dim), .index = @intCast(start) }, &.{self.tile_result.ref()}, output);
                start += count;
            }
        } else if (self.gpu_dequant and (kind.raw() == 21 or kind.raw() == 23)) {
            try self.run(.{ .op = 5, .mode = kind.raw(), .n = @intCast(rows * output_dim * 32), .dim = @intCast(input_dim), .groups = @intCast(output_dim) }, &.{ weight, input, self.quant_grid.ref() }, output);
        } else if (kind.raw() == 20) {
            // This bounded path predates the generic IQ4_NL reduce dispatch
            // being given all four SIMD groups; wide shapes are tested here.
            try self.run(.{ .op = 7, .n = @intCast(rows * output_dim * 32), .dim = @intCast(input_dim), .groups = @intCast(output_dim) }, &.{ weight, input }, output);
        } else if (kind.raw() == 0) {
            try self.run(.{ .op = 6, .n = @intCast(rows * output_dim * 32), .dim = @intCast(input_dim), .groups = @intCast(output_dim), .stride = @intCast(output_dim) }, &.{ weight, input }, output);
        } else {
            // Never overwrite CPU-visible staging while any GPU reader is pending.
            try self.endBatch();
            var start: usize = 0;
            while (start < output_dim) {
                if (self.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
                const count = @min(tile_rows, output_dim - start);
                if (count * input_dim * 4 > self.tile.logical_bytes) return error.Qwen35TileTooLarge;
                const clock = shared.Profile.now();
                for (0..count) |r| try gguf.quant_codec.dequantizeRow(kind, raw, input_dim, start + r, self.tile.floats()[r * input_dim ..][0..input_dim]);
                self.cpu_dequant_ns += shared.Profile.now() - clock;
                self.cpu_dequant_bytes += count * input_dim * 4;
                try self.core.beginBatch();
                errdefer self.core.cancelBatch();
                if (rows >= 8 and count >= 128) {
                    // Existing bounded FP32 MPS GEMM reuses tile weights across
                    // prompt rows; scatter from a bounded compact result buffer.
                    if (mr.termite_metal_decode_runtime_dot_general_2d_f32_device(self.core.runtime, input.handle, input.offset, self.tile.handle, 0, rows, count, input_dim, 1, self.tile_result.handle, 0) != 0) return error.Qwen35MatmulFailed;
                    try self.run(.{ .op = 8, .n = @intCast(rows * count), .groups = @intCast(count), .stride = @intCast(output_dim), .index = @intCast(start) }, &.{self.tile_result.ref()}, output);
                } else {
                    try self.run(.{ .op = 6, .n = @intCast(rows * count * 32), .dim = @intCast(input_dim), .groups = @intCast(count), .stride = @intCast(output_dim), .index = @intCast(start) }, &.{ self.tile.ref(), input }, output);
                }
                try self.endBatch();
                start += count;
            }
            try self.core.beginBatch();
        }
    }
};

test "Qwen35 all artifact codecs match independent ggml rows" {
    inline for (.{ 8, 11, 12, 13, 14, 20, 21, 23 }) |tag| {
        const prefix = "fixtures/qwen35_streamed/" ++ std.fmt.comptimePrint("{d}", .{tag});
        const raw = @embedFile(prefix ++ ".bin");
        const expected = @embedFile(prefix ++ ".f32");
        var row: [256]f32 = undefined;
        for (0..3) |r| {
            try gguf.quant_codec.dequantizeRow(gguf.tensor_types.TensorType.fromRaw(tag), raw, 256, r, &row);
            for (row, 0..) |actual, j| {
                const want: f32 = @bitCast(std.mem.readInt(u32, expected[(r * 256 + j) * 4 ..][0..4], .little));
                try std.testing.expectApproxEqAbs(want, actual, 1e-6);
            }
        }
    }
}
test "Qwen35 Metal tiled matrices match independent ggml rows across batch boundaries" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var input = try Buffer.init(kernels.core.runtime, max_chunk * 256 * 4);
    defer input.deinit();
    var output = try Buffer.init(kernels.core.runtime, max_chunk * 3 * 4);
    defer output.deinit();
    for (input.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32;
    inline for (.{ 0, 8, 11, 12, 13, 14, 20, 21, 23 }) |tag| {
        const prefix = "fixtures/qwen35_streamed/" ++ std.fmt.comptimePrint("{d}", .{tag});
        const raw = @embedFile(prefix ++ ".bin");
        const expected = @embedFile(prefix ++ ".f32");
        var weight = try Buffer.init(kernels.core.runtime, raw.len);
        defer weight.deinit();
        @memcpy(weight.bytes[0..raw.len], raw);
        for ([_]usize{ 1, 3, 33, 128, max_chunk }) |rows| {
            try kernels.core.beginBatch();
            try kernels.matmul(gguf.tensor_types.TensorType.fromRaw(tag), raw, weight.ref(), 256, 3, rows, input.ref(), output.ref());
            try kernels.core.endBatch();
            for (0..rows) |r| for (0..3) |column| {
                var want: f32 = 0;
                for (0..256) |j| {
                    const w: f32 = @bitCast(std.mem.readInt(u32, expected[(column * 256 + j) * 4 ..][0..4], .little));
                    want += w * input.floats()[r * 256 + j];
                }
                try std.testing.expectApproxEqAbs(want, output.floats()[r * 3 + column], 0.0001 + @abs(want) * 0.0001);
            };
        }
    }
}

test "Qwen35 wide IQ4_NL IQ3_S IQ4_XS rows include every block in decode and prefill" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    inline for (.{ 20, 21, 23 }) |tag| {
        const prefix = "fixtures/qwen35_streamed/" ++ std.fmt.comptimePrint("{d}", .{tag});
        const raw = @embedFile(prefix ++ ".bin");
        const expected = @embedFile(prefix ++ ".f32");
        const rb = raw.len / 3;
        for ([_]usize{ 5120, 17408 }) |width| {
            const row_bytes = width / 256 * rb;
            var weight = try Buffer.init(kernels.core.runtime, row_bytes * 3);
            defer weight.deinit();
            for (0..3) |r| for (0..width / 256) |block| {
                @memcpy(weight.bytes[r * row_bytes + block * rb ..][0..rb], raw[r * rb ..][0..rb]);
            };
            var input = try Buffer.init(kernels.core.runtime, width * 8 * 4);
            defer input.deinit();
            var output = try Buffer.init(kernels.core.runtime, 8 * 3 * 4);
            defer output.deinit();
            for (input.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 32;
            for ([_]usize{ 1, 3, 8 }) |rows| {
                try kernels.core.beginBatch();
                try kernels.matmul(gguf.tensor_types.TensorType.fromRaw(tag), weight.bytes[0 .. row_bytes * 3], weight.ref(), width, 3, rows, input.ref(), output.ref());
                try kernels.core.endBatch();
                for (0..rows) |r| for (0..3) |col| {
                    var want: f64 = 0;
                    for (0..width) |j| {
                        const w: f32 = @bitCast(std.mem.readInt(u32, expected[(col * 256 + j % 256) * 4 ..][0..4], .little));
                        want += @as(f64, w) * input.floats()[r * width + j];
                    }
                    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(want)), output.floats()[r * 3 + col], 0.0001 + @as(f32, @floatCast(@abs(want))) * 0.0001);
                };
            }
        }
    }
}

test "Qwen35 MPS dequant tiles preserve padded output strides and a short final tile" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var input = try Buffer.init(kernels.core.runtime, max_chunk * 256 * 4);
    defer input.deinit();
    var output = try Buffer.init(kernels.core.runtime, max_chunk * 1041 * 4);
    defer output.deinit();
    for (input.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32;
    inline for (.{ 8, 11, 12, 13, 14, 20, 21, 23 }) |tag| {
        const prefix = "fixtures/qwen35_streamed/" ++ std.fmt.comptimePrint("{d}", .{tag});
        const raw = @embedFile(prefix ++ ".bin");
        const expected = @embedFile(prefix ++ ".f32");
        const rb = raw.len / 3;
        var weight = try Buffer.init(kernels.core.runtime, 1041 * rb);
        defer weight.deinit();
        for (0..1041) |r| @memcpy(weight.bytes[r * rb ..][0..rb], raw[(r % 3) * rb ..][0..rb]);
        for ([_]usize{ 1, 8, 33, 128, max_chunk }) |rows| {
            try kernels.core.beginBatch();
            try kernels.matmul(gguf.tensor_types.TensorType.fromRaw(tag), weight.bytes[0 .. 1041 * rb], weight.ref(), 256, 1041, rows, input.ref(), output.ref());
            try kernels.core.endBatch();
            for (0..rows) |r| for (0..1041) |col| {
                var want: f32 = 0;
                for (0..256) |j| {
                    const w: f32 = @bitCast(std.mem.readInt(u32, expected[((col % 3) * 256 + j) * 4 ..][0..4], .little));
                    want += w * input.floats()[r * 256 + j];
                }
                try std.testing.expectApproxEqAbs(want, output.floats()[r * 1041 + col], 0.0001 + @abs(want) * 0.0001);
            };
        }
    }
}

test "Qwen35 GPU dequant tiles exactly match independent ggml values with offsets" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    inline for (.{ 11, 21, 23 }) |tag| {
        const prefix = "fixtures/qwen35_streamed/" ++ std.fmt.comptimePrint("{d}", .{tag});
        const raw = @embedFile(prefix ++ ".bin");
        const expected = @embedFile(prefix ++ ".f32");
        var weight = try Buffer.init(kernels.core.runtime, raw.len + 256);
        defer weight.deinit();
        @memcpy(weight.bytes[256..][0..raw.len], raw);
        var ref = weight.ref();
        ref.offset = 256;
        ref.bytes = raw.len;
        try kernels.core.beginBatch();
        try kernels.run(.{ .op = 4, .mode = tag, .n = 512, .dim = 256, .index = 1 }, &.{ ref, kernels.quant_grid.ref() }, kernels.tile.ref());
        try kernels.endBatch();
        for (0..512) |j| {
            const want: f32 = @bitCast(std.mem.readInt(u32, expected[(256 + j) * 4 ..][0..4], .little));
            try std.testing.expectEqual(want, kernels.tile.floats()[j]);
        }
    }
}
