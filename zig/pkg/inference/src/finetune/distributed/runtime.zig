// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in two-rank collective context for RealAutodiffTrainer.
const std = @import("std");
const jaccl = @import("jaccl.zig");
const sync = @import("gradient_sync.zig");
const trainer_mod = @import("../real_autodiff_trainer.zig");
pub const lifecycle = @import("lifecycle.zig");

pub const Context = struct {
    allocator: std.mem.Allocator,
    group: jaccl.Group,
    trainer: ?*trainer_mod.RealAutodiffTrainer = null,
    step: u64 = 0,
    local_weight: u64 = 1,
    total_sync_ns: u64 = 0,

    pub const OptionalBlock = struct { name: []const u8, data: ?[]f32, elements: usize };

    pub fn rank(self: *const Context) u8 {
        return self.group.rank();
    }
    pub fn deinit(self: *Context) void {
        self.group.deinit();
    }

    pub fn reduce(ctx: *anyopaque, grads: []const trainer_mod.GradBlock) anyerror!void {
        const self: *Context = @ptrCast(@alignCast(ctx));
        const started_ns = monotonicNowNs();
        if (self.trainer) |trainer| {
            // GLiNER2's task heads have per-window optimizer gates. Every
            // rank must apply the union of task families seen by the group.
            var flags: [8]u8 = @splat(0);
            for (trainer.conditional_optimizer_families[0..trainer.conditional_optimizer_family_count], 0..) |family, i|
                flags[i] = @intFromBool(family.window_present);
            var gathered_flags: [16]u8 = undefined;
            try self.group.allGatherBytes(&flags, &gathered_flags);
            for (trainer.conditional_optimizer_families[0..trainer.conditional_optimizer_family_count], 0..) |*family, i|
                family.window_present = gathered_flags[i] != 0 or gathered_flags[8 + i] != 0;
        }
        const blocks = try self.allocator.alloc(sync.Block, grads.len);
        defer self.allocator.free(blocks);
        for (grads, blocks) |grad, *block| block.* = .{ .name = grad.name, .data = grad.data };
        std.mem.sort(sync.Block, blocks, {}, struct {
            fn less(_: void, a: sync.Block, b: sync.Block) bool {
                return std.mem.lessThan(u8, a.name, b.name);
            }
        }.less);
        sync.reduce(self.allocator, &self.group, self.step, self.local_weight, blocks) catch |err| {
            std.log.err("distributed gradient reduction at step {d}: {s}: {s}", .{ self.step, @errorName(err), self.group.lastError() });
            return err;
        };
        const ended_ns = monotonicNowNs();
        const duration_ns = ended_ns -| started_ns;
        self.total_sync_ns +|= duration_ns;
        std.debug.print("distributed_sync rank={d} step={d} duration_ns={d} total_sync_ns={d}\n", .{
            self.rank(), self.step, duration_ns, self.total_sync_ns,
        });
        self.step += 1;
    }

    pub fn verifyWeightsAgree(self: *Context, trainer: *const trainer_mod.RealAutodiffTrainer) !void {
        if (trainer.lora_params.items.len == 0) return error.UninitializedDistributedTrainer;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        for (trainer.lora_params.items) |slot| {
            hash.update(slot.name);
            hash.update(std.mem.sliceAsBytes(slot.weights));
        }
        for (trainer.regular_params.items) |slot| {
            hash.update(slot.name);
            hash.update(std.mem.sliceAsBytes(slot.weights));
        }
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        var gathered: [64]u8 = undefined;
        try self.group.allGatherBytes(&digest, &gathered);
        if (!std.mem.eql(u8, gathered[0..32], gathered[32..64])) return error.DistributedWeightsMismatch;
    }

    pub fn verifyDigest(self: *Context, digest: [32]u8) !void {
        var gathered: [64]u8 = undefined;
        try self.group.allGatherBytes(&digest, &gathered);
        if (!std.mem.eql(u8, gathered[0..32], gathered[32..64])) return error.DistributedStateMismatch;
    }

    /// Make the union of trainable paths identical before either rank stages
    /// its optimizer transaction. Missing local paths contribute zero; paths
    /// missing on both ranks remain absent and retain grad=None semantics.
    pub fn reduceOptional(self: *Context, scratch: std.mem.Allocator, step: u64, local_weight: u64, blocks: []OptionalBlock) !void {
        const started_ns = monotonicNowNs();
        try reduceOptionalCollective(scratch, &self.group, step, local_weight, blocks);
        const duration_ns = monotonicNowNs() -| started_ns;
        self.total_sync_ns +|= duration_ns;
        std.debug.print("distributed_sync rank={d} step={d} duration_ns={d} total_sync_ns={d}\n", .{ self.rank(), step, duration_ns, self.total_sync_ns });
    }
};

pub fn reduceOptionalCollective(scratch: std.mem.Allocator, collective: anytype, step: u64, local_weight: u64, blocks: []Context.OptionalBlock) !void {
    if (blocks.len == 0 or blocks.len > 4096 or local_weight == 0) return error.InvalidGradientInventory;
    var header: [40]u8 = undefined;
    std.mem.writeInt(u64, header[0..8], step, .little);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (blocks) |block| {
        if (block.name.len == 0 or block.elements == 0 or (block.data != null and block.data.?.len != block.elements)) return error.InvalidGradientInventory;
        hash.update(block.name);
        hash.update(&.{0});
        var length: [8]u8 = undefined;
        std.mem.writeInt(u64, &length, block.elements, .little);
        hash.update(&length);
    }
    hash.final(header[8..40]);
    var peer_header: [80]u8 = undefined;
    try collective.allGatherBytes(&header, &peer_header);
    if (!std.mem.eql(u8, peer_header[0..40], peer_header[40..80])) return error.GradientInventoryMismatch;
    const local = try scratch.alloc(u8, blocks.len);
    const gathered = try scratch.alloc(u8, blocks.len * 2);
    for (blocks, local) |block, *flag| flag.* = @intFromBool(block.data != null);
    try collective.allGatherBytes(local, gathered);
    const active = try scratch.alloc(sync.Block, blocks.len);
    var count: usize = 0;
    for (blocks, 0..) |*block, index| {
        if (gathered[index] > 1 or gathered[blocks.len + index] > 1) return error.InvalidGradientInventory;
        if (gathered[index] == 0 and gathered[blocks.len + index] == 0) continue;
        if (block.data == null) {
            const zeros = try scratch.alloc(f32, block.elements);
            @memset(zeros, 0);
            block.data = zeros;
        }
        active[count] = .{ .name = block.name, .data = block.data.? };
        count += 1;
    }
    if (count == 0) return;
    std.mem.sort(sync.Block, active[0..count], {}, struct {
        fn less(_: void, a: sync.Block, b: sync.Block) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.less);
    try sync.reduce(scratch, collective, step, local_weight, active[0..count]);
}

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) return 0;
    return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
}

pub fn openFromEnv(allocator: std.mem.Allocator) !?Context {
    const transport_raw = std.c.getenv("ANTFLY_DISTRIBUTED_TRANSPORT");
    const generic_rank = std.c.getenv("ANTFLY_DISTRIBUTED_RANK");
    const generic_coordinator = std.c.getenv("ANTFLY_DISTRIBUTED_COORDINATOR");
    const generic_library = std.c.getenv("ANTFLY_DISTRIBUTED_LIBRARY");
    const generic_devices = std.c.getenv("ANTFLY_DISTRIBUTED_DEVICES_FILE");
    const legacy_rank = std.c.getenv("ANTFLY_JACCL_RANK");
    const legacy_coordinator = std.c.getenv("ANTFLY_JACCL_COORDINATOR");
    const legacy_library = std.c.getenv("ANTFLY_JACCL_LIBRARY");
    const legacy_devices = std.c.getenv("ANTFLY_JACCL_DEVICES_FILE");
    const generic = transport_raw != null or generic_rank != null or generic_coordinator != null or generic_library != null or generic_devices != null;
    const legacy = legacy_rank != null or legacy_coordinator != null or legacy_library != null or legacy_devices != null;
    if (!generic and !legacy) return null;
    if (generic and legacy) return error.MixedDistributedConfiguration;
    const rank_raw = if (generic) generic_rank else legacy_rank;
    const coordinator_raw = if (generic) generic_coordinator else legacy_coordinator;
    const library_raw = if (generic) generic_library else legacy_library;
    const devices_raw = if (generic) generic_devices else legacy_devices;
    const transport = if (transport_raw) |value| std.mem.span(value) else "jaccl";
    if (!std.mem.eql(u8, transport, "jaccl") and !std.mem.eql(u8, transport, "tcp")) return error.InvalidDistributedTransport;
    if (rank_raw == null or coordinator_raw == null or library_raw == null or
        (std.mem.eql(u8, transport, "jaccl") and devices_raw == null)) return error.IncompleteDistributedConfiguration;
    const rank = try std.fmt.parseInt(u8, std.mem.span(rank_raw.?), 10);
    return .{
        .allocator = allocator,
        .group = try jaccl.Group.open(std.mem.span(library_raw.?), rank, std.mem.span(coordinator_raw.?), if (devices_raw) |value| std.mem.span(value) else ""),
    };
}
