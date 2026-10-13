// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Standalone L4 differential for grouped BF16 SDPA with right padding.
//! Generate fixtures with generate_cudnn_padding_goldens.py outside the repo.
const std = @import("std");
const context = @import("context.zig");
const buffer = @import("buffer.zig");
const sdpa = @import("cudnn_sdpa.zig");

fn read(allocator: std.mem.Allocator, io: std.Io, directory: []const u8, name: []const u8, bytes: usize) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ directory, name });
    defer allocator.free(path);
    const value = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(bytes + 1));
    errdefer allocator.free(value);
    if (value.len != bytes) return error.InvalidFixtureSize;
    return value;
}

fn upload(ctx: *context.CudaContext, bytes: []const u8) !buffer.DeviceBuffer {
    var result = try buffer.DeviceBuffer.alloc(ctx, bytes.len);
    errdefer result.free(ctx);
    try result.copyFromHost(ctx, bytes);
    return result;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2) return error.ExpectedGoldenDirectory;
    var ctx = try context.CudaContext.initDefault();
    defer ctx.deinit();
    if (ctx.info.compute_major != 8 or ctx.info.compute_minor != 9) return error.UnqualifiedGpuArchitecture;
    var module = try sdpa.Module.init(ctx.stream);
    defer module.deinit();
    for (0..4) |index| {
        const directory = try std.fmt.allocPrint(allocator, "{s}/case-{d}", .{ args[1], index });
        const raw_shape = try read(allocator, init.io, directory, "shape.u64", 48);
        var shape: [6]u64 = undefined;
        @memcpy(std.mem.asBytes(&shape), raw_shape);
        const batch: usize = @intCast(shape[0]);
        const sequence: usize = @intCast(shape[1]);
        if (batch == 0 or batch > 32 or sequence == 0 or sequence > 8192 or shape[2] != 4 or shape[3] != 2 or shape[4] != 256) return error.InvalidFixtureShape;
        const count = batch * sequence * 4 * 256;
        const q_bytes = try read(allocator, init.io, directory, "q.bf16", count * 2);
        const k_bytes = try read(allocator, init.io, directory, "k.bf16", count);
        const v_bytes = try read(allocator, init.io, directory, "v.bf16", count);
        var q = try upload(&ctx, q_bytes);
        defer q.free(&ctx);
        var k = try upload(&ctx, k_bytes);
        defer k.free(&ctx);
        var v = try upload(&ctx, v_bytes);
        defer v.free(&ctx);
        var output = try buffer.DeviceBuffer.alloc(&ctx, count * 2);
        defer output.free(&ctx);
        var lengths = try buffer.DeviceBuffer.alloc(&ctx, batch * 4);
        defer lengths.free(&ctx);
        const key = sdpa.Key{ .batch = batch, .q_heads = 4, .kv_heads = 2, .sequence = sequence, .dim = 256, .scale = 1, .local_window_radius = @intCast(shape[5]), .padding_mask = true };
        const workspace_bytes = (try module.workspaceBytesFor(key)) orelse return error.PaddedPlanUnavailable;
        var workspace = try buffer.DeviceBuffer.alloc(&ctx, workspace_bytes);
        defer workspace.free(&ctx);
        const output_host = try allocator.alloc(u16, count);
        const zeros = try allocator.alloc(u16, count);
        @memset(zeros, 0);
        // Revisit the same key with different lengths. Reusing a plan must
        // not capture the previous request's padding values or output rows.
        for ([_]usize{ 0, 1, 0 }) |variant| {
            const length_name = try std.fmt.allocPrint(allocator, "lengths-{d}.i32", .{variant});
            const length_bytes = try read(allocator, init.io, directory, length_name, batch * 4);
            const host_lengths = try allocator.alloc(i32, batch);
            @memcpy(std.mem.sliceAsBytes(host_lengths), length_bytes);
            const expected_name = try std.fmt.allocPrint(allocator, "expected-{d}.f32", .{variant});
            const expected_bytes = try read(allocator, init.io, directory, expected_name, count * 4);
            const expected = try allocator.alloc(f32, count);
            @memcpy(std.mem.sliceAsBytes(expected), expected_bytes);
            try lengths.copyFromHost(&ctx, length_bytes);
            try output.copyFromHost(&ctx, std.mem.sliceAsBytes(zeros));
            if (!try module.executeTextPadded(key, q.ptr, k.ptr, v.ptr, output.ptr, workspace.ptr, lengths.ptr)) return error.PaddedExecuteUnavailable;
            try output.copyToHost(&ctx, std.mem.sliceAsBytes(output_host));
            try ctx.synchronize();
            var maximum_error: f64 = 0;
            var minimum_cosine: f64 = 1;
            for (0..batch) |b| for (0..sequence) |s| {
                const valid = s < @as(usize, @intCast(host_lengths[b]));
                var dot: f64 = 0;
                var aa: f64 = 0;
                var bb: f64 = 0;
                for (0..1024) |d| {
                    const at = (b * sequence + s) * 1024 + d;
                    const actual: f32 = @bitCast(@as(u32, output_host[at]) << 16);
                    if (!std.math.isFinite(actual)) return error.NonfiniteAttentionOutput;
                    if (!valid and output_host[at] != 0) return error.NonzeroPaddingOutput;
                    const gold: f64 = expected[at];
                    const value: f64 = actual;
                    maximum_error = @max(maximum_error, @abs(value - gold));
                    dot += value * gold;
                    aa += value * value;
                    bb += gold * gold;
                }
                if (valid) {
                    if (aa <= 0 or bb <= 0) return error.ZeroValidOutput;
                    minimum_cosine = @min(minimum_cosine, dot / @sqrt(aa * bb));
                }
            };
            std.debug.print("padded_sdpa batch={d} sequence={d} radius={d} variant={d} min_cosine={d:.9} max_error={d:.9}\n", .{ batch, sequence, shape[5], variant, minimum_cosine, maximum_error });
            if (minimum_cosine < 0.9999 or maximum_error > 0.02) return error.AttentionDifferentialFailed;
        }
    }
    std.debug.print("PASS: 12 L4 padded-attention runs, including plan reuse and changed lengths; cuDNN={d}\n", .{module.version});
}
