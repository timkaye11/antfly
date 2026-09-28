// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Synchronous, bucketed reduction for host-accessible optimizer gradients.
//! Every rank sends a fixed-width step/inventory header before any f32 data so
//! a mismatched parameter list cannot be interpreted as another tensor.
const std = @import("std");

pub const Block = struct {
    name: []const u8,
    data: []f32,
};

pub const max_bucket_elements: usize = 1024 * 1024;
const header_len = 48;

pub fn reduce(allocator: std.mem.Allocator, collective: anytype, step: u64, local_weight: u64, blocks: []const Block) !void {
    if (local_weight == 0 or blocks.len == 0) return error.InvalidGradientInventory;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var total_elements: usize = 0;
    for (blocks, 0..) |block, index| {
        if (block.name.len == 0 or block.data.len == 0) return error.InvalidGradientInventory;
        if (index > 0 and std.mem.order(u8, blocks[index - 1].name, block.name) != .lt) return error.UnsortedGradientInventory;
        hash.update(block.name);
        hash.update(&.{0});
        var len_bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_bytes, block.data.len, .little);
        hash.update(&len_bytes);
        total_elements = try std.math.add(usize, total_elements, block.data.len);
        for (block.data) |value| if (!std.math.isFinite(value)) return error.NonFiniteGradient;
    }
    if (total_elements == 0) return error.InvalidGradientInventory;
    var header: [header_len]u8 = undefined;
    std.mem.writeInt(u64, header[0..8], step, .little);
    std.mem.writeInt(u64, header[8..16], local_weight, .little);
    hash.final(header[16..48]);
    var gathered: [header_len * 2]u8 = undefined;
    try collective.allGatherBytes(&header, &gathered);
    const peer = gathered[header_len * (1 - collective.rank()) ..][0..header_len];
    if (!std.mem.eql(u8, header[0..8], peer[0..8])) return error.DistributedStepMismatch;
    if (!std.mem.eql(u8, header[16..48], peer[16..48])) return error.GradientInventoryMismatch;
    const remote_weight = std.mem.readInt(u64, peer[8..16], .little);
    if (remote_weight == 0) return error.InvalidGradientInventory;
    const global_weight = try std.math.add(u64, local_weight, remote_weight);
    const own_scale: f32 = @floatFromInt(local_weight);
    const inverse_global: f32 = 1.0 / @as(f32, @floatFromInt(global_weight));

    const input = try allocator.alloc(f32, @min(total_elements, max_bucket_elements));
    defer allocator.free(input);
    const output = try allocator.alloc(f32, input.len);
    defer allocator.free(output);
    var block_index: usize = 0;
    var block_offset: usize = 0;
    var remaining = total_elements;
    while (remaining > 0) {
        const count = @min(remaining, input.len);
        var cursor: usize = 0;
        while (cursor < count) {
            const block = blocks[block_index];
            const n = @min(count - cursor, block.data.len - block_offset);
            for (0..n) |i| input[cursor + i] = block.data[block_offset + i] * own_scale;
            cursor += n;
            block_offset += n;
            if (block_offset == block.data.len) {
                block_index += 1;
                block_offset = 0;
            }
        }
        try collective.allSumF32(input[0..count], output[0..count]);
        for (output[0..count]) |value| if (!std.math.isFinite(value)) return error.NonFiniteGradient;
        // The source slices may span bucket boundaries, so copy out by
        // position rather than assuming a parameter fits in one bucket.
        var output_offset: usize = 0;
        var consumed = total_elements - remaining;
        for (blocks) |block| {
            if (consumed >= block.data.len) {
                consumed -= block.data.len;
                continue;
            }
            const n = @min(count - output_offset, block.data.len - consumed);
            for (0..n) |i| block.data[consumed + i] = output[output_offset + i] * inverse_global;
            output_offset += n;
            consumed = 0;
            if (output_offset == count) break;
        }
        remaining -= count;
    }
}

test "reduction rejects divergent inventories before sending gradients" {
    const Fake = struct {
        fn rank(_: *@This()) u8 {
            return 0;
        }
        fn allGatherBytes(_: *@This(), input: []const u8, output: []u8) !void {
            @memcpy(output[0..input.len], input);
            @memcpy(output[input.len..], input);
            output[input.len + 16] ^= 1;
        }
        fn allSumF32(_: *@This(), _: []const f32, _: []f32) !void {
            return error.UnexpectedCollective;
        }
    };
    var fake = Fake{};
    var values = [_]f32{ 1, 2 };
    try std.testing.expectError(error.GradientInventoryMismatch, reduce(std.testing.allocator, &fake, 1, 2, &.{.{ .name = "a", .data = &values }}));
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, &values);
}

test "weighted two-rank reduction updates all gradient slices" {
    const Fake = struct {
        fn rank(_: *@This()) u8 {
            return 0;
        }
        fn allGatherBytes(_: *@This(), input: []const u8, output: []u8) !void {
            @memcpy(output[0..input.len], input);
            @memcpy(output[input.len..], input);
            std.mem.writeInt(u64, output[input.len + 8 ..][0..8], 3, .little);
        }
        fn allSumF32(_: *@This(), input: []const f32, output: []f32) !void {
            for (input, output) |value, *out| out.* = value + 3 * 4;
        }
    };
    var fake = Fake{};
    var a = [_]f32{2};
    var b = [_]f32{ 2, 2 };
    try reduce(std.testing.allocator, &fake, 1, 1, &.{ .{ .name = "a", .data = &a }, .{ .name = "b", .data = &b } });
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), b[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), b[1], 1e-6);
}
