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

const lifecycle = distributed.lifecycle;

test "distributed GLiNER2.5 peer pause and checkpoint requests reach both ranks" {
    const Collective = struct {
        own_rank: usize,
        pub fn allGatherBytes(self: *@This(), input: []const u8, output: []u8) !void {
            @memcpy(output[0..19], input);
            @memcpy(output[19..38], input);
            output[17] = 0;
            output[18] = 0;
            output[36] = 1;
            output[37] = 1;
            try std.testing.expectEqual(input[17], output[self.own_rank * 19 + 17]);
        }
    };
    for (0..2) |rank| {
        var collective = Collective{ .own_rank = rank };
        const agreed = try lifecycle.agreeBoundary(&collective, .{
            .microbatch = 7,
            .optimizer = 2,
            .complete = false,
            .pause = rank == 1,
            .checkpoint = rank == 1,
        });
        try std.testing.expect(agreed.pause and agreed.checkpoint);
        try std.testing.expectEqual(@as(u64, 7), agreed.microbatch);
    }
}

test "distributed GLiNER2.5 pause refuses divergent training boundaries" {
    const Collective = struct {
        pub fn allGatherBytes(_: *@This(), input: []const u8, output: []u8) !void {
            @memcpy(output[0..19], input);
            @memcpy(output[19..38], input);
            output[19] ^= 1;
        }
    };
    var collective = Collective{};
    try std.testing.expectError(error.DistributedBoundaryMismatch, lifecycle.agreeBoundary(&collective, .{
        .microbatch = 7,
        .optimizer = 2,
        .complete = false,
        .pause = true,
        .checkpoint = false,
    }));
}

const CheckpointFixture = struct {
    fail_snapshot: bool = false,
    fail_receipt: bool = false,
    peer_fail_phase: u8 = 0,
    disconnect_phase: u8 = 0,
    snapshots: usize = 0,
    receipts: usize = 0,
    acknowledged_phase: u8 = 0,
    pub fn writeSnapshot(self: *@This()) !void {
        if (self.fail_snapshot) return error.DiskFull;
        self.snapshots += 1;
    }
    pub fn writeReceipt(self: *@This()) !void {
        try std.testing.expectEqual(@as(u8, 1), self.acknowledged_phase);
        if (self.fail_receipt) return error.ReceiptWriteFailed;
        self.receipts += 1;
    }
    pub fn allGatherBytes(self: *@This(), input: []const u8, output: []u8) !void {
        if (input[0] == self.disconnect_phase) return error.PeerDisconnected;
        @memcpy(output[0..2], input);
        @memcpy(output[2..4], input);
        output[3] = @intFromBool(input[0] != self.peer_fail_phase);
        self.acknowledged_phase = input[0];
    }
};

test "distributed GLiNER2.5 checkpoint never acknowledges a one-sided snapshot" {
    var own_failure = CheckpointFixture{ .fail_snapshot = true };
    try std.testing.expectError(error.DiskFull, lifecycle.publishCheckpoint(&own_failure, &own_failure));
    try std.testing.expectEqual(@as(usize, 0), own_failure.receipts);
    var peer_failure = CheckpointFixture{ .peer_fail_phase = 1 };
    try std.testing.expectError(error.DistributedPeerCheckpointFailed, lifecycle.publishCheckpoint(&peer_failure, &peer_failure));
    try std.testing.expectEqual(@as(usize, 1), peer_failure.snapshots);
    try std.testing.expectEqual(@as(usize, 0), peer_failure.receipts);
    var lost_peer = CheckpointFixture{ .disconnect_phase = 1 };
    try std.testing.expectError(error.PeerDisconnected, lifecycle.publishCheckpoint(&lost_peer, &lost_peer));
    try std.testing.expectEqual(@as(usize, 0), lost_peer.receipts);
}

test "distributed GLiNER2.5 checkpoint requires receipts from both ranks" {
    var own_failure = CheckpointFixture{ .fail_receipt = true };
    try std.testing.expectError(error.ReceiptWriteFailed, lifecycle.publishCheckpoint(&own_failure, &own_failure));
    var peer_failure = CheckpointFixture{ .peer_fail_phase = 2 };
    try std.testing.expectError(error.DistributedPeerCheckpointFailed, lifecycle.publishCheckpoint(&peer_failure, &peer_failure));
    var success = CheckpointFixture{};
    try lifecycle.publishCheckpoint(&success, &success);
    try std.testing.expectEqual(@as(u8, 2), success.acknowledged_phase);
    try std.testing.expectEqual(@as(usize, 1), success.receipts);
}
