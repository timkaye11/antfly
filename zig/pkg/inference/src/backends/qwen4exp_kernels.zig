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

//! Small eager kernel surface for the experimental SSD-backed decoder.
const std = @import("std");
const options = @import("build_options");
const mr = @import("metal_runtime.zig");
pub extern fn termite_metal_qwen4exp_alias(runtime: *mr.RawMetalDecodeRuntime, data: [*]u8, bytes: usize) ?*anyopaque;
pub extern fn termite_metal_buffer_release(handle: *anyopaque) void;
pub extern fn termite_metal_qwen4exp_prepare_linear(runtime: *mr.RawMetalDecodeRuntime, handle: *anyopaque, format: u32, offset: usize, bytes: usize, input: usize, output: usize) c_int;
extern fn termite_metal_qwen4exp_apply_linear(runtime: *mr.RawMetalDecodeRuntime, format: u32, input: *anyopaque, input_offset: usize, rows: usize, input_dim: usize, output_dim: usize, output: *anyopaque, output_offset: usize) c_int;
extern fn termite_metal_qwen4exp_residency_create(runtime: *mr.RawMetalDecodeRuntime, handles: [*]const *anyopaque, count: usize, admitted_bytes: usize) ?*anyopaque;
extern fn termite_metal_qwen4exp_residency_release(runtime: *mr.RawMetalDecodeRuntime, handle: *anyopaque) void;
extern fn termite_metal_qwen4exp_residency_supported() c_int;

pub const Params = extern struct {
    op: u32,
    n: u32 = 0,
    dim: u32 = 0,
    groups: u32 = 0,
    position: u32 = 0,
    stride: u32 = 0,
    mode: u32 = 0,
    index: u32 = 0,
    epsilon: f32 = 1e-6,
    scale: f32 = 1,
    theta: f32 = 10000000,
    reserved: f32 = 0,
};

pub const Ref = struct {
    handle: *anyopaque,
    offset: usize = 0,
    bytes: usize,
    pub fn slice(self: Ref, offset: usize, bytes: usize) !Ref {
        if (offset > self.bytes or bytes > self.bytes - offset) return error.MetalBufferRange;
        return .{ .handle = self.handle, .offset = try std.math.add(usize, self.offset, offset), .bytes = bytes };
    }
};

pub const Buffer = struct {
    bytes: []u8,
    handle: *anyopaque,
    logical_bytes: usize,
    pub fn init(runtime: *mr.RawMetalDecodeRuntime, len: usize) !Buffer {
        if (len == 0) return error.EmptyMetalBuffer;
        const page = std.heap.pageSize();
        const rounded = (try std.math.add(usize, len, page - 1)) & ~(page - 1);
        const bytes = try std.heap.page_allocator.alloc(u8, rounded);
        errdefer std.heap.page_allocator.free(bytes);
        const handle = termite_metal_qwen4exp_alias(runtime, bytes.ptr, bytes.len) orelse return error.MetalBufferAllocationFailed;
        return .{ .bytes = bytes, .handle = handle, .logical_bytes = len };
    }
    pub fn deinit(self: *Buffer) void {
        termite_metal_buffer_release(self.handle);
        std.heap.page_allocator.free(self.bytes);
        self.* = undefined;
    }
    pub fn ref(self: Buffer) Ref {
        return .{ .handle = self.handle, .bytes = self.logical_bytes };
    }
    pub fn floats(self: Buffer) []f32 {
        return std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(self.bytes[0..self.logical_bytes])));
    }
};

pub const BatchKind = enum { attention, router, experts, output };
pub const Profile = struct {
    enabled: bool = false,
    // Host wall time: 0..27 opcodes, 28 quantized matvec, 29 paging, 30 batch wait.
    // 31/32 grouped projection/activation, 33/34 prefill gather/scatter.
    // Inside batches, opcode timings measure encoding, not GPU execution.
    nanoseconds: [35]u64 = @splat(0),
    calls: [35]u64 = @splat(0),
    batch_wait_ns: [4]u64 = @splat(0),
    batch_gpu_ns: [4]u64 = @splat(0),
    pub fn now() u64 {
        var ts: std.c.timespec = undefined;
        if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
        return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
    }
    pub fn record(self: *Profile, op: usize, start: u64) void {
        if (!self.enabled) return;
        self.nanoseconds[op] +|= now() -| start;
        self.calls[op] +|= 1;
    }
    pub fn since(self: Profile, before: Profile) Profile {
        var result = self;
        for (0..result.calls.len) |i| {
            result.nanoseconds[i] -|= before.nanoseconds[i];
            result.calls[i] -|= before.calls[i];
        }
        for (0..4) |i| {
            result.batch_wait_ns[i] -|= before.batch_wait_ns[i];
            result.batch_gpu_ns[i] -|= before.batch_gpu_ns[i];
        }
        return result;
    }
};

var residency_notice = std.atomic.Value(bool).init(false);
pub const Kernels = struct {
    runtime: *mr.RawMetalDecodeRuntime,
    pipeline: *mr.RawMetalGeneratedPipeline,
    dummy: Buffer,
    profile: Profile = .{},
    cancelled: ?*const std.atomic.Value(bool) = null,
    batch_active: bool = false,
    batch_submitted: bool = false,
    batching_enabled: bool = true,
    residency: ?*anyopaque = null,
    pub fn init() !Kernels {
        return initSource(@embedFile("qwen4exp.metal"), "qwen4exp_eager");
    }
    pub fn initSource(source: []const u8, name: [*:0]const u8) !Kernels {
        if (comptime !options.enable_metal) return error.MetalNotEnabled;
        const runtime = mr.termite_metal_decode_runtime_create() orelse return error.MetalRuntimeUnavailable;
        errdefer mr.termite_metal_decode_runtime_destroy(runtime);
        var message: [4096]u8 = @splat(0);
        const pipeline = mr.termite_metal_generated_pipeline_create(source.ptr, source.len, name, 1, &message, message.len) orelse {
            std.log.err("Qwen4Exp Metal kernel: {s}", .{std.mem.sliceTo(&message, 0)});
            return error.MetalKernelCompilationFailed;
        };
        errdefer mr.termite_metal_generated_pipeline_destroy(pipeline);
        return .{
            .runtime = runtime,
            .pipeline = pipeline,
            .dummy = try Buffer.init(runtime, 4),
            .profile = .{ .enabled = std.c.getenv("ANTFLY_QWEN4EXP_PROFILE") != null },
            .batching_enabled = std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_BATCHING") == null and std.c.getenv("ANTFLY_QWEN4EXP_CAPTURE_DIR") == null,
        };
    }
    pub fn beginBatch(self: *Kernels) !void {
        if (self.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
        if (!self.batching_enabled) return;
        if (self.batch_active or self.batch_submitted) return error.Qwen4ExpNestedBatch;
        if (mr.termite_metal_decode_runtime_begin_frame(self.runtime) != 0) return error.MetalFrameBeginFailed;
        self.batch_active = true;
    }
    pub fn submitBatch(self: *Kernels) !void {
        if (!self.batch_active) return;
        if (self.batch_submitted) return error.Qwen4ExpOutstandingBatch;
        if (mr.termite_metal_decode_runtime_submit_frame(self.runtime) != 0) return error.MetalFrameSubmitFailed;
        self.batch_active = false;
        self.batch_submitted = true;
    }
    pub fn waitBatch(self: *Kernels, kind: ?BatchKind) !void {
        if (!self.batch_submitted) return;
        const start = if (self.profile.enabled) Profile.now() else 0;
        defer self.profile.record(30, start);
        // wait_frame drains submitted resources even when execution fails.
        const status = mr.termite_metal_decode_runtime_wait_frame(self.runtime);
        self.batch_submitted = false;
        if (self.profile.enabled) if (kind) |k| {
            const i = @backingInt(k);
            self.profile.batch_wait_ns[i] +|= Profile.now() -| start;
            self.profile.batch_gpu_ns[i] +|= mr.termite_metal_decode_runtime_last_frame_gpu_nanos(self.runtime);
        };
        if (status != 0) return error.MetalFrameSyncFailed;
    }
    pub fn endBatch(self: *Kernels) !void {
        try self.submitBatch();
        try self.waitBatch(null);
    }
    pub fn cancelBatch(self: *Kernels) void {
        if (self.batch_active) {
            _ = mr.termite_metal_decode_runtime_cancel_frame(self.runtime);
            self.batch_active = false;
        }
        self.waitBatch(null) catch {};
    }
    pub fn endProfiledBatch(self: *Kernels, kind: BatchKind) !void {
        try self.submitBatch();
        try self.waitBatch(kind);
    }
    pub fn deinit(self: *Kernels) void {
        self.cancelBatch();
        self.releaseResidency();
        mr.termite_metal_decode_runtime_destroy(self.runtime);
        mr.termite_metal_generated_pipeline_destroy(self.pipeline);
        self.dummy.deinit();
        self.* = undefined;
    }
    /// Pin the admitted allocations in one residency set. Residency sets need
    /// macOS 15; earlier systems run without one (receipts record
    /// `metal_residency`), but a failure where they exist is an error.
    pub fn makeResident(self: *Kernels, handles: []const *anyopaque, admitted_bytes: usize) !void {
        if (self.residency != null or self.batch_active) return error.InvalidResidencyState;
        if (termite_metal_qwen4exp_residency_supported() == 0) {
            if (!residency_notice.swap(true, .monotonic)) std.log.warn("Metal residency sets need macOS 15; running without one", .{});
            return;
        }
        self.residency = termite_metal_qwen4exp_residency_create(self.runtime, handles.ptr, handles.len, admitted_bytes) orelse return error.Qwen4ExpResidencyUnavailable;
    }
    pub fn releaseResidency(self: *Kernels) void {
        std.debug.assert(!self.batch_active);
        if (self.residency) |set| termite_metal_qwen4exp_residency_release(self.runtime, set);
        self.residency = null;
    }
    pub fn run(self: *Kernels, params: Params, inputs: []const Ref, output: Ref) !void {
        if (self.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
        if (params.n == 0 or params.op > 32 or inputs.len > 7) return error.InvalidQwen4ExpDispatch;
        if ((params.op == 13 or params.op == 27) and params.n != 32) return error.InvalidQwen4ExpDispatch;
        if (params.op == 2 and (params.dim == 0 or params.n % 32 != 0)) return error.InvalidQwen4ExpDispatch;
        if (params.op == 30 or params.op == 31) {
            if (inputs.len != 2 or params.dim == 0 or params.n % params.dim != 0 or
                inputs[1].bytes < @as(usize, params.n / params.dim) * 4 or
                (if (params.op == 30) output.bytes else inputs[0].bytes) < @as(usize, params.n) * 4)
                return error.InvalidQwen4ExpDispatch;
        }
        if (params.op == 28) {
            const block: u32 = switch (params.mode) {
                7, 8 => 32,
                12, 13 => 256,
                else => return error.InvalidQwen4ExpDispatch,
            };
            if (inputs.len != 3 or params.dim == 0 or params.dim % block != 0 or params.groups == 0 or params.groups > 2560 or params.n % (params.groups * 32) != 0 or params.n / (params.groups * 32) > 10 or params.index > 1023) return error.InvalidQwen4ExpDispatch;
        }
        const start = if (self.profile.enabled) Profile.now() else 0;
        defer self.profile.record(if (params.op == 32) 28 else if (params.op >= 28) params.op + 3 else params.op, start);
        var handles: [8]?*anyopaque = @splat(self.dummy.handle);
        var offsets: [8]usize = @splat(0);
        var lengths: [8]usize = @splat(4);
        for (inputs, 0..) |input, i| {
            handles[i] = input.handle;
            offsets[i] = input.offset;
            lengths[i] = input.bytes;
        }
        handles[7] = output.handle;
        offsets[7] = output.offset;
        lengths[7] = output.bytes;
        if (mr.termite_metal_qwen4exp_dispatch(self.runtime, self.pipeline, &handles, &offsets, &lengths, 8, &params, @sizeOf(Params), params.n) != 0) return error.Qwen4ExpMetalDispatchFailed;
    }
    pub fn matmul(self: *Kernels, format: u32, weight: Ref, input_dim: usize, output_dim: usize, rows: usize, input: Ref, output: Ref) !void {
        if (rows == 0 or rows > 32 or input.bytes < rows * input_dim * 4 or output.bytes < rows * output_dim * 4) return error.MetalBufferRange;
        const quant: mr.MetalQuantFormat = switch (format) {
            7 => .q5_1,
            8 => .q8_0,
            12 => .q4_k,
            13 => .q5_k,
            else => return error.UnsupportedQwen4ExpMatrixType,
        };
        if (termite_metal_qwen4exp_prepare_linear(self.runtime, weight.handle, @backingInt(quant), weight.offset, weight.bytes, input_dim, output_dim) != 0) return error.Qwen4ExpQuantPrepareFailed;
        if (termite_metal_qwen4exp_apply_linear(self.runtime, @backingInt(quant), input.handle, input.offset, rows, input_dim, output_dim, output.handle, output.offset) != 0) return error.Qwen4ExpQuantMatvecFailed;
        if (!self.batch_active and mr.termite_metal_decode_runtime_flush_active_frame(self.runtime) != 0) return error.MetalFrameSyncFailed;
    }
    pub fn matmulTiled(self: *Kernels, format: u32, weight: Ref, input_dim: usize, output_dim: usize, rows: usize, input: Ref, output: Ref, tile: Ref, result: Ref) !void {
        if (rows < 8) return self.matmul(format, weight, input_dim, output_dim, rows, input, output);
        if (rows > 32 or input.bytes < rows * input_dim * 4 or output.bytes < rows * output_dim * 4 or input_dim > 6144 or result.bytes < rows * 512 * 4 or tile.bytes < 512 * input_dim * 4) return error.MetalBufferRange;
        const quant: mr.MetalQuantFormat = switch (format) {
            7 => .q5_1,
            8 => .q8_0,
            12 => .q4_k,
            13 => .q5_k,
            else => return error.UnsupportedQwen4ExpMatrixType,
        };
        if (termite_metal_qwen4exp_prepare_linear(self.runtime, weight.handle, @backingInt(quant), weight.offset, weight.bytes, input_dim, output_dim) != 0) return error.Qwen4ExpQuantPrepareFailed;
        var at: usize = 0;
        while (at < output_dim) {
            const count = @min(512, output_dim - at);
            if (mr.termite_metal_decode_runtime_quant_copy_linear_slot_to_f32_device(self.runtime, @backingInt(quant), 0, at, count, input_dim, output_dim, 1, tile.handle, tile.offset) != 0) return error.Qwen4ExpQuantMatvecFailed;
            if (mr.termite_metal_decode_runtime_dot_general_2d_f32_device(self.runtime, input.handle, input.offset, tile.handle, tile.offset, rows, count, input_dim, 1, result.handle, result.offset) != 0) return error.Qwen4ExpQuantMatvecFailed;
            try self.run(.{ .op = 32, .n = @intCast(rows * count), .dim = @intCast(count), .stride = @intCast(output_dim), .index = @intCast(at) }, &.{result}, output);
            at += count;
        }
    }
    pub fn matvec(self: *Kernels, format: u32, weight_bytes: usize, input_dim: usize, output_dim: usize, weight: Ref, input: Ref, output: Ref) !void {
        if (self.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
        if (input.bytes < input_dim * 4 or output.bytes < output_dim * 4 or weight.bytes < weight_bytes) return error.MetalBufferRange;
        if (format == 0 or format == 30) {
            return self.run(.{ .op = 6, .n = @intCast(output_dim * 32), .dim = @intCast(input_dim), .mode = format }, &.{ weight, input }, output);
        }
        const quant: mr.MetalQuantFormat = switch (format) {
            7 => .q5_1,
            8 => .q8_0,
            12 => .q4_k,
            13 => .q5_k,
            else => return error.UnsupportedQwen4ExpMatrixType,
        };
        const start = if (self.profile.enabled) Profile.now() else 0;
        defer self.profile.record(28, start);
        if (termite_metal_qwen4exp_prepare_linear(self.runtime, weight.handle, @backingInt(quant), weight.offset, weight_bytes, input_dim, output_dim) != 0) return error.Qwen4ExpQuantPrepareFailed;
        if (mr.termite_metal_decode_runtime_apply_quantized_linear_slot_device(self.runtime, @backingInt(quant), 0, input.handle, input.offset, 1, input_dim, output_dim, output.handle, output.offset) != 0) return error.Qwen4ExpQuantMatvecFailed;
        if (!self.batch_active and mr.termite_metal_decode_runtime_flush_active_frame(self.runtime) != 0) return error.MetalFrameSyncFailed;
    }
};

test "Qwen4Exp Metal router chooses expert 511 and normalizes top ten" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var logits = try Buffer.init(kernels.runtime, 512 * 4);
    defer logits.deinit();
    var route = try Buffer.init(kernels.runtime, 20 * 4);
    defer route.deinit();
    for (logits.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    try kernels.run(.{ .op = 13, .n = 32 }, &.{logits.ref()}, route.ref());
    const ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(route.bytes[0..40])));
    var sum: f32 = 0;
    for (ids, 0..) |id, i| {
        try std.testing.expectEqual(@as(u32, @intCast(511 - i)), id);
        sum += route.floats()[10 + i];
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1), sum, 1e-6);
    @memset(logits.floats(), 0);
    try kernels.run(.{ .op = 13, .n = 32 }, &.{logits.ref()}, route.ref());
    for (ids, 0..) |id, i| try std.testing.expectEqual(@as(u32, @intCast(i)), id);
    // Reject nonfinite values even when they sit outside lane zero's stripe.
    logits.floats()[511] = std.math.nan(f32);
    try kernels.run(.{ .op = 13, .n = 32 }, &.{logits.ref()}, route.ref());
    for (ids) |id| try std.testing.expectEqual(std.math.maxInt(u32), id);
}

test "Qwen4Exp SIMD normalization matches double precision RMS and L2 references" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var input = try Buffer.init(kernels.runtime, 10240 * 4);
    defer input.deinit();
    var gamma = try Buffer.init(kernels.runtime, 10240 * 4);
    defer gamma.deinit();
    var output = try Buffer.init(kernels.runtime, 10240 * 4);
    defer output.deinit();
    for (input.floats(), gamma.floats(), 0..) |*x, *g, i| {
        x.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 127)) - 63)) * 0.017;
        g.* = 0.5 + @as(f32, @floatFromInt(i % 23)) * 0.031;
    }
    for ([_]u32{ 128, 2560 }) |dim| for (0..3) |mode| {
        const count = 10240 / dim;
        try kernels.run(.{ .op = 2, .n = count * 32, .dim = dim, .mode = @intCast(mode) }, &.{ input.ref(), gamma.ref() }, output.ref());
        for (0..count) |group| {
            const base = group * dim;
            var sum: f64 = 0;
            for (input.floats()[base..][0..dim]) |x| sum += @as(f64, x) * x;
            const inverse = 1.0 / @sqrt((if (mode == 2) sum else sum / @as(f64, @floatFromInt(dim))) + 1e-6);
            for (0..dim) |j| {
                const weight: f64 = if (mode == 2) 1 else gamma.floats()[if (mode == 1) j else base + j];
                const expected: f32 = @floatCast(@as(f64, input.floats()[base + j]) * inverse * weight);
                try std.testing.expectApproxEqAbs(expected, output.floats()[base + j], 2e-6);
            }
        }
    };
}

test "Qwen4Exp SIMD greedy selection keeps ties and rejects nonfinite tails" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var logits = try Buffer.init(kernels.runtime, 248320 * 4);
    defer logits.deinit();
    var result = try Buffer.init(kernels.runtime, 4);
    defer result.deinit();
    @memset(logits.floats(), -3);
    logits.floats()[31] = 7;
    logits.floats()[248319] = 7;
    try kernels.run(.{ .op = 27, .n = 32, .dim = 248320 }, &.{logits.ref()}, result.ref());
    try std.testing.expectEqual(@as(u32, 31), std.mem.readInt(u32, result.bytes[0..4], .little));
    logits.floats()[248319] = std.math.inf(f32);
    try kernels.run(.{ .op = 27, .n = 32, .dim = 248320 }, &.{logits.ref()}, result.ref());
    try std.testing.expectEqual(std.math.maxInt(u32), std.mem.readInt(u32, result.bytes[0..4], .little));
    try std.testing.expectError(error.InvalidQwen4ExpDispatch, kernels.run(.{ .op = 27, .n = 1, .dim = 248320 }, &.{logits.ref()}, result.ref()));
}

test "Qwen4Exp Q5_1 down projection covers all 640 inputs and nonzero weight offset" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    const width = 640;
    const rows = 3;
    const prefix = 64;
    var weight = try Buffer.init(kernels.runtime, prefix + rows * 20 * 24);
    defer weight.deinit();
    var input = try Buffer.init(kernels.runtime, width * 4);
    defer input.deinit();
    var output = try Buffer.init(kernels.runtime, rows * 4);
    defer output.deinit();
    @memset(weight.bytes, 0);
    @memset(input.floats(), 1);
    // Constant +1 weights. A two-SIMD-group dispatch misses blocks 16..19
    // and returns 512 instead of 640; odd output count also exercises the tail.
    for (0..rows * 20) |block| {
        const off = prefix + block * 24;
        std.mem.writeInt(u16, weight.bytes[off..][0..2], @bitCast(@as(f16, 1)), .little);
        @memset(weight.bytes[off + 8 ..][0..16], 0x11);
    }
    const range = try weight.ref().slice(prefix, rows * 20 * 24);
    try kernels.matvec(7, range.bytes, width, rows, range, input.ref(), output.ref());
    for (output.floats()) |value| try std.testing.expectApproxEqAbs(@as(f32, 640), value, 1e-4);
}

test "Qwen4Exp batch preserves quantized slot offsets and scratch dependencies and cancels safely" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    kernels.batching_enabled = true;
    const encoded_bytes = 20 * 24;
    var weights = try Buffer.init(kernels.runtime, 2 * encoded_bytes);
    defer weights.deinit();
    var input = try Buffer.init(kernels.runtime, 640 * 4);
    defer input.deinit();
    var scratch = try Buffer.init(kernels.runtime, 4);
    defer scratch.deinit();
    var output = try Buffer.init(kernels.runtime, 4);
    defer output.deinit();
    @memset(input.floats(), 1);
    @memset(weights.bytes, 0);
    for (0..40) |block| {
        const off = block * 24;
        std.mem.writeInt(u16, weights.bytes[off..][0..2], @bitCast(@as(f16, if (block < 20) 1 else 2)), .little);
        @memset(weights.bytes[off + 8 ..][0..16], 0x11);
    }
    const first = try weights.ref().slice(0, encoded_bytes);
    const second = try weights.ref().slice(encoded_bytes, encoded_bytes);
    const allocations = [_]*anyopaque{ weights.handle, input.handle, scratch.handle, output.handle };
    if (termite_metal_qwen4exp_residency_supported() != 0) try std.testing.expectError(error.Qwen4ExpResidencyUnavailable, kernels.makeResident(&allocations, 1));
    try kernels.makeResident(&allocations, weights.bytes.len + input.bytes.len + scratch.bytes.len + output.bytes.len);
    defer kernels.releaseResidency();
    for (0..2) |_| {
        try kernels.beginBatch();
        errdefer kernels.cancelBatch();
        try kernels.matvec(7, encoded_bytes, 640, 1, first, input.ref(), scratch.ref());
        try kernels.run(.{ .op = 1, .n = 1, .dim = 1 }, &.{scratch.ref()}, output.ref());
        try kernels.matvec(7, encoded_bytes, 640, 1, second, input.ref(), scratch.ref());
        try kernels.run(.{ .op = 16, .n = 1 }, &.{ output.ref(), scratch.ref() }, output.ref());
        try kernels.endBatch();
        try std.testing.expectApproxEqAbs(@as(f32, 1920), output.floats()[0], 1e-4);
    }
    try kernels.beginBatch();
    try kernels.run(.{ .op = 0, .n = 1 }, &.{}, output.ref());
    try std.testing.expectEqual(@as(c_int, 1), mr.termite_metal_decode_runtime_active_frame_has_work(kernels.runtime));
    kernels.cancelBatch();
    try std.testing.expectApproxEqAbs(@as(f32, 1920), output.floats()[0], 1e-4);
    try kernels.beginBatch();
    try kernels.run(.{ .op = 0, .n = 1 }, &.{}, output.ref());
    try kernels.endBatch();
    try std.testing.expectEqual(@as(f32, 0), output.floats()[0]);
}

test "Qwen4Exp causal convolution honors PLE dilation across successive tokens" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var input = try Buffer.init(kernels.runtime, 4);
    defer input.deinit();
    var weight = try Buffer.init(kernels.runtime, 16);
    defer weight.deinit();
    var history = try Buffer.init(kernels.runtime, 9 * 4);
    defer history.deinit();
    var output = try Buffer.init(kernels.runtime, 4);
    defer output.deinit();
    // A unit impulse must recur only at positions 0, 3, 6 and 9.
    @memcpy(weight.floats(), &[_]f32{ 1, 2, 3, 4 });
    @memset(history.floats(), 0);
    for (0..12) |position| {
        input.floats()[0] = if (position == 0) 1 else 0;
        try kernels.run(.{ .op = 9, .n = 1, .stride = 3 }, &.{ input.ref(), weight.ref(), history.ref() }, output.ref());
        const impulse: f32 = switch (position) {
            0 => 4,
            3 => 3,
            6 => 2,
            9 => 1,
            else => 0,
        };
        try std.testing.expectApproxEqAbs(impulse / (1 + @exp(-impulse)), output.floats()[0], 1e-6);
    }
}

test "Qwen4Exp GDN preserves recurrent state and converter head order" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var qkv = try Buffer.init(kernels.runtime, 10240 * 4);
    defer qkv.deinit();
    var beta = try Buffer.init(kernels.runtime, 48 * 4);
    defer beta.deinit();
    var alpha = try Buffer.init(kernels.runtime, 48 * 4);
    defer alpha.deinit();
    var decay = try Buffer.init(kernels.runtime, 48 * 4);
    defer decay.deinit();
    var bias = try Buffer.init(kernels.runtime, 48 * 4);
    defer bias.deinit();
    var state = try Buffer.init(kernels.runtime, 48 * 128 * 128 * 4);
    defer state.deinit();
    var output = try Buffer.init(kernels.runtime, 6144 * 4);
    defer output.deinit();
    @memset(qkv.floats(), 0);
    @memset(beta.floats(), 0); // beta = 1/2
    @memset(alpha.floats(), 0);
    @memset(decay.floats(), -1); // exp(-softplus(0)) = 1/2
    @memset(bias.floats(), 0);
    @memset(state.floats(), 0);
    // Key/query head 15 drives value heads 15, 31, 47, not adjacent heads.
    qkv.floats()[15 * 128] = 1;
    qkv.floats()[2048 + 15 * 128] = 1;
    for (0..48) |h| qkv.floats()[4096 + h * 128] = 2;
    for ([_]f32{ 1, 1.25, 1.3125 }) |expected_state| {
        try kernels.run(.{ .op = 10, .n = 6144 }, &.{ qkv.ref(), beta.ref(), alpha.ref(), decay.ref(), bias.ref(), state.ref() }, output.ref());
        for (0..48) |h| {
            const expected: f32 = if (h % 16 == 15) expected_state else 0;
            try std.testing.expectApproxEqAbs(expected, state.floats()[h * 128 * 128], 1e-6);
            try std.testing.expectApproxEqAbs(expected / @sqrt(@as(f32, 128)), output.floats()[h * 128], 1e-6);
        }
    }
}

test "Qwen4Exp QSA selects 512 blocks with causal incomplete tails through context 4096" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var index = try Buffer.init(kernels.runtime, 1024 * 4);
    defer index.deinit();
    var mask = try Buffer.init(kernels.runtime, 1024 * 4);
    defer mask.deinit();
    var query = try Buffer.init(kernels.runtime, 6144 * 4);
    defer query.deinit();
    var keys = try Buffer.init(kernels.runtime, 4096 * 512 * 4);
    defer keys.deinit();
    var scores = try Buffer.init(kernels.runtime, 24 * 4096 * 4);
    defer scores.deinit();
    @memset(query.floats(), 0);
    @memset(keys.floats(), 0);
    for (index.floats(), 0..) |*v, i| v.* = @floatFromInt(i);
    for ([_]u32{ 1, 3, 4, 2048, 2051, 4095, 4096 }) |tokens| {
        try kernels.run(.{ .op = 23, .n = 1, .position = tokens - 1 }, &.{index.ref()}, mask.ref());
        const blocks = tokens / 4;
        for (0..blocks) |block| try std.testing.expectEqual(@as(f32, if (block + 512 >= blocks) 1 else 0), mask.floats()[block]);
        try kernels.run(.{ .op = 24, .n = 24 * tokens, .position = tokens - 1 }, &.{ query.ref(), keys.ref(), mask.ref() }, scores.ref());
        for (0..24 * tokens) |i| {
            const block = (i % tokens) / 4;
            if (block < blocks and block + 512 < blocks) {
                try std.testing.expectEqual(-std.math.inf(f32), scores.floats()[i]);
            } else try std.testing.expectEqual(@as(f32, 0), scores.floats()[i]);
        }
        if (blocks <= 512) {
            // Selection kept every block, so the unmasked mode the runtime
            // uses below 513 blocks scores identically, whatever the mask.
            const masked = try std.testing.allocator.dupe(f32, scores.floats()[0 .. 24 * tokens]);
            defer std.testing.allocator.free(masked);
            @memset(mask.floats(), 0);
            try kernels.run(.{ .op = 24, .n = 24 * tokens, .position = tokens - 1, .mode = 1 }, &.{ query.ref(), keys.ref(), mask.ref() }, scores.ref());
            try std.testing.expectEqualSlices(f32, masked, scores.floats()[0 .. 24 * tokens]);
        }
    }
}

test "Qwen4Exp split submission drains before buffer reuse and rejects nested work" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var buffer = try Buffer.init(kernels.runtime, 256 * 4);
    defer buffer.deinit();
    @memset(buffer.floats(), 42);
    try kernels.beginBatch();
    try kernels.run(.{ .op = 0, .n = 256 }, &.{}, buffer.ref());
    try kernels.submitBatch();
    try std.testing.expectError(error.Qwen4ExpNestedBatch, kernels.beginBatch());
    try kernels.waitBatch(.experts);
    for (buffer.floats()) |v| try std.testing.expectEqual(@as(f32, 0), v);
    try kernels.beginBatch();
    try kernels.run(.{ .op = 0, .n = 256 }, &.{}, buffer.ref());
    try kernels.submitBatch();
    kernels.cancelBatch();
    try std.testing.expect(!kernels.batch_submitted and !kernels.batch_active);
    try kernels.beginBatch();
    kernels.cancelBatch();
}

test "Qwen4Exp prefill gather scatter preserves route order across shared scratch reuse" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    const width = 37;
    const tokens = 128;
    var input = try Buffer.init(kernels.runtime, tokens * width * 4);
    defer input.deinit();
    var indices = try Buffer.init(kernels.runtime, 2 * tokens * 4);
    defer indices.deinit();
    var scratch = try Buffer.init(kernels.runtime, (tokens * width + 16) * 4);
    defer scratch.deinit();
    var output = try Buffer.init(kernels.runtime, tokens * 10 * width * 4);
    defer output.deinit();
    for (input.floats(), 0..) |*value, i| value.* = @floatFromInt(i);
    const locations = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(indices.bytes[0 .. 2 * tokens * 4])));
    for (0..tokens) |i| {
        locations[i] = @intCast((tokens - 1 - i) * 10 + 3);
        locations[tokens + i] = @intCast(i * 10 + 7);
    }
    for ([_]usize{ 1, 7, 32, 128 }) |count| {
        @memset(output.floats(), -999);
        const temporary = try scratch.ref().slice(64, count * width * 4);
        try kernels.beginBatch();
        errdefer kernels.cancelBatch();
        for (0..2) |expert| {
            const index = try indices.ref().slice(expert * tokens * 4, count * 4);
            try kernels.run(.{ .op = 30, .n = @intCast(count * width), .dim = width }, &.{ input.ref(), index }, temporary);
            try kernels.run(.{ .op = 31, .n = @intCast(count * width), .dim = width }, &.{ temporary, index }, output.ref());
        }
        try kernels.endBatch();
        for (0..tokens) |token| for (0..10) |route| for (0..width) |component| {
            const written = (route == 3 and token >= tokens - count) or (route == 7 and token < count);
            try std.testing.expectEqual(if (written) input.floats()[token * width + component] else @as(f32, -999), output.floats()[(token * 10 + route) * width + component]);
        };
    }
}

test "Qwen4Exp grouped experts match independent GGUF dequantization with masks and slot permutations" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    const codec = @import("../gguf/quant_codec.zig");
    var kernels = try Kernels.init();
    defer kernels.deinit();
    for ([_]u32{ 7, 8, 12, 13 }) |format| {
        const width: usize = if (format < 12) 640 else 512;
        const block: usize = if (format < 12) 32 else 256;
        const block_bytes: usize = switch (format) {
            7 => 24,
            8 => 34,
            12 => 144,
            13 => 176,
            else => unreachable,
        };
        const rows = 3;
        const prefix = 64;
        const matrix_bytes = width / block * block_bytes * rows;
        const stride = prefix + matrix_bytes;
        var weights = try Buffer.init(kernels.runtime, 3 * stride);
        defer weights.deinit();
        for (weights.bytes, 0..) |*v, i| v.* = @truncate(i * 17 + 29);
        for (0..3) |slot| for (0..rows * width / block) |b| {
            const off = slot * stride + prefix + b * block_bytes;
            std.mem.writeInt(u16, weights.bytes[off..][0..2], @bitCast(@as(f16, 0.03125)), .little);
            if (format != 8) std.mem.writeInt(u16, weights.bytes[off + 2 ..][0..2], @bitCast(@as(f16, 0.015625)), .little);
        };
        var inputs = try Buffer.init(kernels.runtime, 32 * width * 4);
        defer inputs.deinit();
        for (inputs.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 23)) / 23 - 0.5;
        var slots = try Buffer.init(kernels.runtime, 3 * 4);
        defer slots.deinit();
        const order = [_]u32{ 2, 0, 1 };
        @memcpy(slots.bytes[0..12], std.mem.sliceAsBytes(&order));
        var output = try Buffer.init(kernels.runtime, 3 * rows * 4);
        defer output.deinit();
        for ([_]f32{ 0, 1 }) |per_route| {
            @memset(output.floats(), -999);
            try kernels.run(.{ .op = 28, .n = 3 * rows * 32, .dim = @intCast(width), .groups = rows, .stride = @intCast(stride), .position = prefix, .mode = format, .index = 5, .scale = per_route }, &.{ weights.ref(), inputs.ref(), slots.ref() }, output.ref());
            for (order, 0..) |slot, route| {
                for (0..rows) |row| {
                    if (route == 1) {
                        try std.testing.expectEqual(@as(f32, -999), output.floats()[route * rows + row]);
                        continue;
                    }
                    var decoded: [640]f32 = undefined;
                    try codec.dequantizeRow(.{ .known = @fromBackingInt(@intCast(format)) }, weights.bytes[slot * stride + prefix ..][0..matrix_bytes], width, row, decoded[0..width]);
                    var expected: f64 = 0;
                    const x = inputs.floats()[if (per_route != 0) route * width else 0..][0..width];
                    for (decoded[0..width], x) |w, v| expected += @as(f64, w) * v;
                    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), output.floats()[route * rows + row], 0.002);
                }
            }
        }
        var batched = try Buffer.init(kernels.runtime, 32 * rows * 4);
        defer batched.deinit();
        for ([_]usize{ 1, 2, 7, 32 }) |batch_rows| {
            try kernels.beginBatch();
            try kernels.matmul(format, try weights.ref().slice(prefix, matrix_bytes), width, rows, batch_rows, inputs.ref(), batched.ref());
            try kernels.endBatch();
            for (0..batch_rows) |r| for (0..rows) |col| {
                var decoded: [640]f32 = undefined;
                try codec.dequantizeRow(.{ .known = @fromBackingInt(@intCast(format)) }, weights.bytes[prefix..][0..matrix_bytes], width, col, decoded[0..width]);
                var expected: f64 = 0;
                for (decoded[0..width], inputs.floats()[r * width ..][0..width]) |w, v| expected += @as(f64, w) * v;
                try std.testing.expectApproxEqAbs(@as(f32, @floatCast(expected)), batched.floats()[r * rows + col], 0.002);
            };
        }
    }
}

test "Qwen4Exp batched expert reductions match sequential production projection shapes" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    var kernels = try Kernels.init();
    defer kernels.deinit();
    var tile = try Buffer.init(kernels.runtime, 512 * 6144 * 4);
    defer tile.deinit();
    var compact = try Buffer.init(kernels.runtime, 32 * 512 * 4);
    defer compact.deinit();
    for ([_]u32{ 7, 8, 12, 13 }) |format| {
        const input_dim: usize = if (format == 7) 640 else 2560;
        const output_dim: usize = if (format == 7) 2560 else 640;
        const block: usize = if (format < 12) 32 else 256;
        const block_bytes: usize = switch (format) {
            7 => 24,
            8 => 34,
            12 => 144,
            13 => 176,
            else => unreachable,
        };
        const weight_bytes = input_dim / block * block_bytes * output_dim;
        var weights = try Buffer.init(kernels.runtime, 64 + weight_bytes);
        defer weights.deinit();
        for (weights.bytes, 0..) |*v, i| v.* = @truncate(i * 17 + 29);
        for (0..output_dim * input_dim / block) |b| {
            const offset = 64 + b * block_bytes;
            std.mem.writeInt(u16, weights.bytes[offset..][0..2], @bitCast(@as(f16, 0.03125)), .little);
            if (format != 8) std.mem.writeInt(u16, weights.bytes[offset + 2 ..][0..2], @bitCast(@as(f16, 0.015625)), .little);
        }
        var input = try Buffer.init(kernels.runtime, 64 + 32 * input_dim * 4);
        defer input.deinit();
        for (input.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 23)) / 23 - 0.5;
        var expected = try Buffer.init(kernels.runtime, 64 + 32 * output_dim * 4);
        defer expected.deinit();
        var actual = try Buffer.init(kernels.runtime, 64 + 32 * output_dim * 4);
        defer actual.deinit();
        const w = try weights.ref().slice(64, weight_bytes);
        for ([_]bool{ false, true }) |tiled| for ([_]usize{ 1, 2, 7, 8, 32 }) |rows| {
            try kernels.beginBatch();
            errdefer kernels.cancelBatch();
            const x = try input.ref().slice(64, rows * input_dim * 4);
            const y = try actual.ref().slice(64, rows * output_dim * 4);
            if (tiled) try kernels.matmulTiled(format, w, input_dim, output_dim, rows, x, y, tile.ref(), compact.ref()) else try kernels.matmul(format, w, input_dim, output_dim, rows, x, y);
            for (0..rows) |row| try kernels.matvec(format, weight_bytes, input_dim, output_dim, w, try input.ref().slice(64 + row * input_dim * 4, input_dim * 4), try expected.ref().slice(64 + row * output_dim * 4, output_dim * 4));
            try kernels.endBatch();
            for (actual.floats()[16..][0 .. rows * output_dim], expected.floats()[16..][0 .. rows * output_dim]) |got, want| {
                try std.testing.expect(std.math.isFinite(got));
                try std.testing.expectApproxEqAbs(want, got, @max(0.002, @abs(want) * 0.00002));
            }
        };
    }
}
