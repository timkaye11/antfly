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

//! Developer smoke for the optional direct cuDNN backend integration.
const std = @import("std");
const Context = @import("context.zig").CudaContext;
const Buffer = @import("buffer.zig").DeviceBuffer;
const Cudnn = @import("cudnn_sdpa.zig").Module;

pub fn main(init: std.process.Init) !void {
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    const expect_unavailable = if (args.next()) |arg| std.mem.eql(u8, arg, "--expect-unavailable") else false;
    var ctx = try Context.initDefault();
    defer ctx.deinit();
    var cudnn = Cudnn.init(ctx.stream) catch |err| {
        if (expect_unavailable and err == error.CudnnUnavailable) {
            std.debug.print("cuDNN unavailable fallback smoke passed\n", .{});
            return;
        }
        return err;
    };
    defer cudnn.deinit();
    if (expect_unavailable) return error.CudnnUnexpectedlyAvailable;
    const sequence: usize = 17;
    const bytes = sequence * 12 * 64 * @sizeOf(u16);
    const zeros = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(zeros);
    @memset(zeros, 0);
    var buffers: [4]Buffer = undefined;
    var initialized: usize = 0;
    defer {
        for (buffers[0..initialized]) |*buffer| buffer.free(&ctx);
    }
    while (initialized < buffers.len) : (initialized += 1) {
        buffers[initialized] = try Buffer.alloc(&ctx, bytes);
        try buffers[initialized].copyFromHost(&ctx, zeros);
    }
    if (!try cudnn.execute(sequence, buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, 0))
        return error.CudnnPlanUnavailable;
    try ctx.synchronize();
    const stats = cudnn.stats();
    std.debug.print("cuDNN SDPA smoke passed: executes={} misses={} hits={} workspace={}\n", .{
        stats.executes, stats.plan_misses, stats.plan_hits, stats.workspace_high_water,
    });

    const text_key = @import("cudnn_sdpa.zig").Key{
        .batch = 8,
        .q_heads = 4,
        .kv_heads = 2,
        .sequence = 512,
        .dim = 256,
        .scale = 1,
        .local_window_radius = 512,
    };
    const q_bytes: usize = 8 * 512 * 4 * 256 * @sizeOf(u16);
    const kv_bytes: usize = 8 * 512 * 2 * 256 * @sizeOf(u16);
    var text_buffers: [4]Buffer = undefined;
    const text_sizes = [_]usize{ q_bytes, kv_bytes, kv_bytes, q_bytes };
    const text_zeros = try std.heap.page_allocator.alloc(u8, q_bytes);
    defer std.heap.page_allocator.free(text_zeros);
    @memset(text_zeros, 0);
    var text_initialized: usize = 0;
    defer {
        for (text_buffers[0..text_initialized]) |*buffer| buffer.free(&ctx);
    }
    while (text_initialized < text_buffers.len) : (text_initialized += 1) {
        text_buffers[text_initialized] = try Buffer.alloc(&ctx, text_sizes[text_initialized]);
        try text_buffers[text_initialized].copyFromHost(&ctx, text_zeros[0..text_sizes[text_initialized]]);
    }
    var plain_text_key = text_key;
    plain_text_key.local_window_radius = 0;
    if (!try cudnn.executeText(plain_text_key, text_buffers[0].ptr, text_buffers[1].ptr, text_buffers[2].ptr, text_buffers[3].ptr, 0))
        return error.CudnnPlainTextPlanUnavailable;
    if (!try cudnn.executeText(text_key, text_buffers[0].ptr, text_buffers[1].ptr, text_buffers[2].ptr, text_buffers[3].ptr, 0))
        return error.CudnnTextPlanUnavailable;
    try ctx.synchronize();
    try text_buffers[3].copyToHost(&ctx, text_zeros);
    try ctx.synchronize();
    for (text_zeros) |value| if (value != 0) return error.CudnnPlainOrBandZeroMismatch;
    const text_stats = cudnn.stats();
    std.debug.print("cuDNN text GQA smoke passed: text_executes={} misses={} workspace={}\n", .{
        text_stats.text_executes, text_stats.plan_misses, text_stats.workspace_high_water,
    });
    if (text_stats.text_executes != 2 or text_stats.plan_misses != 3) return error.CudnnKeyCollision;

    try boundarySmoke(&ctx, &cudnn);
}

fn boundarySmoke(ctx: *Context, cudnn: *Cudnn) !void {
    const sequence: usize = 515;
    const query_heads: usize = 4;
    const kv_heads: usize = 2;
    const dim: usize = 256;
    const q_count = sequence * query_heads * dim;
    const kv_count = sequence * kv_heads * dim;
    const q_host = try std.heap.page_allocator.alloc(u16, q_count);
    defer std.heap.page_allocator.free(q_host);
    const k_host = try std.heap.page_allocator.alloc(u16, kv_count);
    defer std.heap.page_allocator.free(k_host);
    const v_host = try std.heap.page_allocator.alloc(u16, kv_count);
    defer std.heap.page_allocator.free(v_host);
    const out_host = try std.heap.page_allocator.alloc(u16, q_count);
    defer std.heap.page_allocator.free(out_host);
    @memset(q_host, 0);
    @memset(k_host, 0);
    @memset(v_host, 0);
    @memset(out_host, 0);
    const sentinel_keys = [_]usize{ 0, 1, 512, 513, 514 };
    for (0..kv_heads) |head| {
        for (sentinel_keys, 0..) |key, channel|
            v_host[(key * kv_heads + head) * dim + channel] = 0x3f80; // BF16 1.0
    }

    const sizes = [_]usize{ q_count * 2, kv_count * 2, kv_count * 2, q_count * 2 };
    var buffers: [4]Buffer = undefined;
    var initialized: usize = 0;
    defer for (buffers[0..initialized]) |*buffer| buffer.free(ctx);
    while (initialized < buffers.len) : (initialized += 1)
        buffers[initialized] = try Buffer.alloc(ctx, sizes[initialized]);
    try buffers[0].copyFromHost(ctx, std.mem.sliceAsBytes(q_host));
    try buffers[1].copyFromHost(ctx, std.mem.sliceAsBytes(k_host));
    try buffers[2].copyFromHost(ctx, std.mem.sliceAsBytes(v_host));
    try ctx.synchronize();
    const key = @import("cudnn_sdpa.zig").Key{
        .batch = 1,
        .q_heads = query_heads,
        .kv_heads = kv_heads,
        .sequence = sequence,
        .dim = dim,
        .scale = 1,
        .local_window_radius = 512,
    };
    if (!try cudnn.executeText(key, buffers[0].ptr, buffers[1].ptr, buffers[2].ptr, buffers[3].ptr, 0))
        return error.CudnnBoundaryPlanUnavailable;
    try buffers[3].copyToHost(ctx, std.mem.sliceAsBytes(out_host));
    try ctx.synchronize();
    const value = struct {
        fn at(output: []const u16, query: usize, channel: usize) u16 {
            return output[(query * query_heads) * dim + channel];
        }
    }.at;
    std.debug.print("cuDNN Zig boundary values: q0={{0x{x},0x{x}}} q513={{0x{x},0x{x}}} q514={{0x{x},0x{x}}}\n", .{
        value(out_host, 0, 2),   value(out_host, 0, 3),   value(out_host, 513, 0),
        value(out_host, 513, 1), value(out_host, 514, 1), value(out_host, 514, 2),
    });
    if (value(out_host, 0, 2) != 0x3b00 or value(out_host, 0, 3) != 0 or
        value(out_host, 513, 0) != 0 or value(out_host, 513, 1) != 0x3aff or
        value(out_host, 514, 1) != 0 or value(out_host, 514, 2) != 0x3b00)
        return error.CudnnBoundaryMismatch;
    std.debug.print("cuDNN Zig boundary smoke passed: q0={{0x{x},0x{x}}} q513={{0x{x},0x{x}}} q514={{0x{x},0x{x}}}\n", .{
        value(out_host, 0, 2),   value(out_host, 0, 3),   value(out_host, 513, 0),
        value(out_host, 513, 1), value(out_host, 514, 1), value(out_host, 514, 2),
    });
}
