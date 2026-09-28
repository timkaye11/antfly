// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Optional, process-local JACCL transport. The C++ library is loaded only
//! for a distributed job; normal inference builds have no JACCL dependency.
const std = @import("std");

const OpenFn = *const fn (c_int, [*:0]const u8, [*:0]const u8, *?*anyopaque) callconv(.c) c_int;
const RankFn = *const fn (?*anyopaque) callconv(.c) c_int;
const SumFn = *const fn (?*anyopaque, [*]const f32, [*]f32, usize) callconv(.c) c_int;
const GatherFn = *const fn (?*anyopaque, [*]const u8, [*]u8, usize) callconv(.c) c_int;
const BarrierFn = *const fn (?*anyopaque) callconv(.c) c_int;
const CloseFn = *const fn (?*anyopaque) callconv(.c) void;
const LastErrorFn = *const fn () callconv(.c) [*:0]const u8;

pub const Group = struct {
    library: std.DynLib,
    handle: ?*anyopaque,
    rank_value: u8,
    all_sum: SumFn,
    all_gather: GatherFn,
    barrier_fn: BarrierFn,
    close_fn: CloseFn,
    last_error_fn: LastErrorFn,

    pub fn open(library_path: []const u8, rank_id: u8, coordinator: [:0]const u8, device_file: [:0]const u8) !Group {
        if (rank_id >= 2) return error.InvalidRank;
        var library = try std.DynLib.open(library_path);
        errdefer library.close();
        const open_fn = library.lookup(OpenFn, "antfly_jaccl_open") orelse return error.InvalidJacclBridge;
        const rank_fn = library.lookup(RankFn, "antfly_jaccl_rank") orelse return error.InvalidJacclBridge;
        const size_fn = library.lookup(RankFn, "antfly_jaccl_size") orelse return error.InvalidJacclBridge;
        const all_sum = library.lookup(SumFn, "antfly_jaccl_all_sum_f32") orelse return error.InvalidJacclBridge;
        const all_gather = library.lookup(GatherFn, "antfly_jaccl_all_gather") orelse return error.InvalidJacclBridge;
        const barrier_fn = library.lookup(BarrierFn, "antfly_jaccl_barrier") orelse return error.InvalidJacclBridge;
        const close_fn = library.lookup(CloseFn, "antfly_jaccl_close") orelse return error.InvalidJacclBridge;
        const last_error_fn = library.lookup(LastErrorFn, "antfly_jaccl_last_error") orelse return error.InvalidJacclBridge;
        var handle: ?*anyopaque = null;
        if (open_fn(rank_id, coordinator.ptr, device_file.ptr, &handle) != 0 or handle == null) {
            std.log.err("JACCL initialization failed: {s}", .{std.mem.span(last_error_fn())});
            return error.JacclFailure;
        }
        errdefer close_fn(handle);
        if (rank_fn(handle) != rank_id or size_fn(handle) != 2) return error.InvalidJacclGroup;
        return .{
            .library = library,
            .handle = handle,
            .rank_value = rank_id,
            .all_sum = all_sum,
            .all_gather = all_gather,
            .barrier_fn = barrier_fn,
            .close_fn = close_fn,
            .last_error_fn = last_error_fn,
        };
    }

    pub fn deinit(self: *Group) void {
        self.close_fn(self.handle);
        self.library.close();
        self.* = undefined;
    }

    pub fn rank(self: *const Group) u8 {
        return self.rank_value;
    }

    pub fn lastError(self: *const Group) []const u8 {
        return std.mem.span(self.last_error_fn());
    }

    pub fn allSumF32(self: *Group, input: []const f32, output: []f32) !void {
        if (input.len == 0 or input.len != output.len) return error.InvalidCollectiveBuffer;
        if (self.all_sum(self.handle, input.ptr, output.ptr, input.len) != 0) return error.JacclFailure;
    }

    pub fn allGatherBytes(self: *Group, input: []const u8, output: []u8) !void {
        if (input.len == 0 or output.len != input.len * 2) return error.InvalidCollectiveBuffer;
        if (self.all_gather(self.handle, input.ptr, output.ptr, input.len) != 0) return error.JacclFailure;
    }

    pub fn barrier(self: *Group) !void {
        if (self.barrier_fn(self.handle) != 0) return error.JacclFailure;
    }
};

test "JACCL group rejects ranks outside a two-node job before loading the library" {
    try std.testing.expectError(error.InvalidRank, Group.open("missing.dylib", 2, "127.0.0.1:32132", "devices.json"));
}
