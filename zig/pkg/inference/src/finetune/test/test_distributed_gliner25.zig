// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const internal = @import("inference_internal");
const distributed = internal.finetune.distributed_runtime;
const run = internal.finetune.gliner_boundary_run;
const bundle = internal.models.gliner_boundary_bundle;

test "distributed GLiNER2.5 replay is disjoint and checkpoints are rank bound" {
    const a = std.testing.allocator;
    const source = bundle.Identity{ .backbone = .small, .precision = .fp32, .weight = bundle.Digest.of("weights"), .sidecars = .{ bundle.Digest.of("model"), bundle.Digest.of("encoder"), bundle.Digest.of("tokenizer"), bundle.Digest.of("tokenizer config") } };
    const data = run.Data{ .examples = 4, .train_sha256 = @splat(2), .schema_sha256 = @splat(3) };
    const config = run.Config{ .mode = .lora, .adapter_config_sha256 = @splat(4), .epochs = 1, .batch_size = 2, .shuffle = true };
    const first = try run.Plan.initDistributed(config, source, data, .{}, .{ .rank = 0 });
    const second = try run.Plan.initDistributed(config, source, data, .{}, .{ .rank = 1 });
    try std.testing.expectEqual(try first.sharedFingerprint(a), try second.sharedFingerprint(a));
    try std.testing.expect(!std.mem.eql(u8, &(try first.fingerprint(a)), &(try second.fingerprint(a))));
    const order = try first.epochOrder(a, 0, null);
    defer a.free(order);
    const other_order = try second.epochOrder(a, 0, null);
    defer a.free(other_order);
    try std.testing.expectEqualSlices(u32, order, other_order);
    var visited = [_]bool{false} ** 8;
    for (order) |index| {
        visited[2 * index] = true;
        visited[2 * index + 1] = true;
    }
    for (visited) |value| try std.testing.expect(value);
}

test "distributed GLiNER2.5 gradients reduce the union of present paths" {
    const Fake = struct {
        pub fn rank(_: *@This()) u8 {
            return 0;
        }
        pub fn allGatherBytes(_: *@This(), input: []const u8, output: []u8) !void {
            @memcpy(output[0..input.len], input);
            @memcpy(output[input.len..], input);
            if (input.len == 3) {
                output[3] = 1;
                output[4] = 1;
            }
        }
        pub fn allSumF32(_: *@This(), input: []const f32, output: []f32) !void {
            try std.testing.expectEqual(@as(usize, 3), input.len);
            for (input, &[_]f32{ 6, 8, 10 }, output) |own, peer, *result| result.* = own + peer;
        }
    };
    var fake = Fake{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var a = [_]f32{ 2, 4 };
    var blocks = [_]distributed.Context.OptionalBlock{
        .{ .name = "a", .data = &a, .elements = 2 },
        .{ .name = "b", .data = null, .elements = 1 },
        .{ .name = "c", .data = null, .elements = 1 },
    };
    try distributed.reduceOptionalCollective(arena.allocator(), &fake, 0, 1, &blocks);
    try std.testing.expectEqualSlices(f32, &.{ 4, 6 }, blocks[0].data.?);
    try std.testing.expectEqualSlices(f32, &.{5}, blocks[1].data.?);
    try std.testing.expect(blocks[2].data == null);
}
