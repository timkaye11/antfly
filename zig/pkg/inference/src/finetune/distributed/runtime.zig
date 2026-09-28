// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in two-rank JACCL context for RealAutodiffTrainer. The launcher sets
//! all four variables; partial configuration fails before a training step.
const std = @import("std");
const jaccl = @import("jaccl.zig");
const sync = @import("gradient_sync.zig");
const trainer_mod = @import("../real_autodiff_trainer.zig");

pub const Context = struct {
    allocator: std.mem.Allocator,
    group: jaccl.Group,
    trainer: ?*trainer_mod.RealAutodiffTrainer = null,
    step: u64 = 0,
    local_weight: u64 = 1,
    total_sync_ns: u64 = 0,

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
            std.log.err("JACCL gradient reduction at step {d}: {s}: {s}", .{ self.step, @errorName(err), self.group.lastError() });
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
};

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) return 0;
    return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
}

pub fn openFromEnv(allocator: std.mem.Allocator) !?Context {
    const rank_raw = std.c.getenv("ANTFLY_JACCL_RANK");
    const coordinator_raw = std.c.getenv("ANTFLY_JACCL_COORDINATOR");
    const devices_raw = std.c.getenv("ANTFLY_JACCL_DEVICES_FILE");
    const library_raw = std.c.getenv("ANTFLY_JACCL_LIBRARY");
    if (rank_raw == null and coordinator_raw == null and devices_raw == null and library_raw == null) return null;
    if (rank_raw == null or coordinator_raw == null or devices_raw == null or library_raw == null) return error.IncompleteJacclConfiguration;
    const rank = try std.fmt.parseInt(u8, std.mem.span(rank_raw.?), 10);
    return .{
        .allocator = allocator,
        .group = try jaccl.Group.open(std.mem.span(library_raw.?), rank, std.mem.span(coordinator_raw.?), std.mem.span(devices_raw.?)),
    };
}
