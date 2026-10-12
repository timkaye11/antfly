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

//! Synchronous SSD paging for one Qwen4Exp sequence. The executor must release
//! its route lease only after Metal completion. Weight bytes and the directory
//! publish together; failed reads leave even evicted slots unchanged.
const std = @import("std");
const store_mod = @import("../../models/tensor_store.zig");
const model = @import("../../models/qwen4exp.zig");
const c_file = @import("../../util/c_file.zig");
const arena = @import("expert_slot_arena.zig").ArenaTypes(.{ .layers = 1, .experts = 512, .route_width = 10 });
const TensorType = @import("../../gguf/tensor_types.zig").TensorType;
pub const read_lane_count: usize = 6;
pub const read_stack_bytes = 256 * 1024;
pub const read_executor_bytes = 2 * 1024 * 1024;

pub const Projection = struct {
    reader: c_file.RangeReader,
    offset: u64,
    stride: usize,
    format: TensorType,
    input: usize,
    output: usize,

    fn open(allocator: std.mem.Allocator, store: store_mod.TensorStore, name: []const u8, experts: usize) !Projection {
        var range = (try store.describeTensorRange(allocator, name)) orelse return error.MissingTensorRange;
        defer range.deinit(allocator);
        if (range.shape.len != 3 or range.shape[0] != experts or range.byte_len % experts != 0) return error.InvalidExpertRange;
        const reader = try c_file.RangeReader.open(allocator, range.path, true);
        errdefer {
            var owned = reader;
            owned.deinit();
        }
        if (range.byte_offset > reader.size or range.byte_len > reader.size - range.byte_offset) return error.RegionOutOfBounds;
        return .{ .reader = reader, .offset = range.byte_offset, .stride = range.byte_len / experts, .format = range.gguf_type orelse return error.MissingGgufFormat, .input = std.math.cast(usize, range.shape[2]) orelse return error.InvalidExpertRange, .output = std.math.cast(usize, range.shape[1]) orelse return error.InvalidExpertRange };
    }
};

pub const Stats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    bytes_read: u64 = 0,
    read_failures: u64 = 0,
    pub fn since(self: Stats, before: Stats) Stats {
        return .{ .hits = self.hits -| before.hits, .misses = self.misses -| before.misses, .bytes_read = self.bytes_read -| before.bytes_read, .read_failures = self.read_failures -| before.read_failures };
    }
};

pub const Layer = struct {
    projections: [3]Projection,
    directory: arena.Directory,
    /// Page-aligned storage can be shared directly with Metal, with no second
    /// weight allocation. Each slot contains gate, up, down in that order.
    bytes: []u8,
    stride: usize,
    logical_bytes: usize,
    cancelled: ?*const std.atomic.Value(bool) = null,
    leased: bool = false,
    stats: Stats = .{},

    pub fn init(allocator: std.mem.Allocator, store: store_mod.TensorStore, layer: usize, slots: u8) !Layer {
        var projections: [3]Projection = undefined;
        var opened: usize = 0;
        errdefer for (projections[0..opened]) |*p| p.reader.deinit();
        var stride: usize = 0;
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |kind, i| {
            var name_buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_{s}_exps.weight", .{ layer, kind });
            projections[i] = try Projection.open(allocator, store, name, 512);
            opened += 1;
            stride = try std.math.add(usize, stride, projections[i].stride);
        }
        return initOwned(projections, slots, stride);
    }

    // Takes ownership only on success; separated to exercise the transaction
    // against tiny files without allocating production-sized expert matrices.
    fn initOwned(projections: [3]Projection, slots: u8, stride: usize) !Layer {
        const directory = try arena.Directory.init(slots);
        if (slots < 10) return error.InvalidCapacity;
        const logical = try std.math.mul(usize, stride, slots);
        const size = (try std.math.add(usize, logical, std.heap.pageSize() - 1)) & ~(std.heap.pageSize() - 1);
        const bytes = try std.heap.page_allocator.alloc(u8, size);
        return .{ .projections = projections, .directory = directory, .bytes = bytes, .stride = stride, .logical_bytes = logical };
    }

    pub fn deinit(self: *Layer) void {
        std.debug.assert(!self.leased);
        for (&self.projections) |*p| p.reader.deinit();
        std.heap.page_allocator.free(self.bytes);
        self.* = undefined;
    }

    pub const Lease = struct {
        owner: *Layer,
        route: arena.RouteObservation,
        pub fn releaseAfterCompletion(self: *Lease) void {
            std.debug.assert(self.owner.leased);
            self.owner.leased = false;
            self.* = undefined;
        }
        pub fn projection(self: *const Lease, route_index: usize, projection_index: usize) []const u8 {
            std.debug.assert(self.owner.leased and route_index < self.route.count and projection_index < 3);
            var offset = @as(usize, self.route.route_slots[route_index]) * self.owner.stride;
            for (self.owner.projections[0..projection_index]) |p| offset += p.stride;
            return self.owner.bytes[offset..][0..self.owner.projections[projection_index].stride];
        }
    };

    pub fn acquire(self: *Layer, experts: []const u32, staging: []u8) !Lease {
        return self.acquireWithIo(experts, staging, null);
    }

    const ReadLane = struct {
        projection: *const Projection,
        experts: []const u32,
        repaired: []const bool,
        staging: []u8,
        stride: usize,
        projection_offset: usize,
        partition: usize,
        bytes: u64 = 0,
        cancelled: ?*const std.atomic.Value(bool) = null,
        failure: ?anyerror = null,
        fn run(lane: *@This()) void {
            const p = lane.projection;
            var cursor = lane.projection_offset;
            var ordinal: usize = 0;
            for (lane.experts, lane.repaired) |expert, miss| {
                if (lane.cancelled) |stop| if (stop.load(.acquire)) {
                    lane.failure = error.Cancelled;
                    return;
                };
                if (!miss) continue;
                const destination = cursor;
                cursor += lane.stride;
                const selected = ordinal % 2 == lane.partition;
                ordinal += 1;
                if (!selected) continue;
                const offset = std.math.add(u64, p.offset, std.math.mul(u64, expert, p.stride) catch {
                    lane.failure = error.Overflow;
                    return;
                }) catch {
                    lane.failure = error.Overflow;
                    return;
                };
                p.reader.read(offset, lane.staging[destination..][0..p.stride]) catch |err| {
                    lane.failure = err;
                    return;
                };
                lane.bytes += p.stride;
            }
        }
    };

    /// A transaction must stay at one address until finish/abort joins its tasks.
    /// Resident projections remain immutable throughout; misses publish atomically.
    pub const Pending = struct {
        owner: *Layer = undefined,
        next: arena.Directory = undefined,
        route: arena.RouteObservation = undefined,
        experts: [10]u32 = undefined,
        lanes: [read_lane_count]ReadLane = undefined,
        group: std.Io.Group = .init,
        io: ?std.Io = null,
        staging: []u8 = undefined,
        active: bool = false,

        pub fn begin(self: *Pending, owner: *Layer, experts: []const u32, staging: []u8, io: ?std.Io) !void {
            std.debug.assert(!self.active);
            if (owner.leased) return error.ExpertSlotsInUse;
            if (owner.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
            var next = owner.directory;
            const route = try next.observeRoute(0, 512, experts);
            if (try std.math.mul(usize, route.miss_count, owner.stride) > staging.len) return error.ExpertStagingTooSmall;
            self.* = .{ .owner = owner, .next = next, .route = route, .staging = staging, .io = io, .active = true };
            @memcpy(self.experts[0..experts.len], experts);
            owner.leased = true;
            var offset: usize = 0;
            for (&owner.projections, 0..) |*p, projection| {
                for (0..2) |partition| self.lanes[projection * 2 + partition] = .{
                    .projection = p,
                    .cancelled = owner.cancelled,
                    .experts = self.experts[0..experts.len],
                    .repaired = self.route.repaired[0..experts.len],
                    .staging = staging,
                    .stride = owner.stride,
                    .projection_offset = offset,
                    .partition = partition,
                };
                offset += p.stride;
            }
            if (route.miss_count > 0) {
                if (io) |read_io| {
                    for (&self.lanes) |*lane| self.group.async(read_io, ReadLane.run, .{lane});
                } else for (&self.lanes) |*lane| lane.run();
            }
        }

        pub fn residentProjection(self: *const Pending, route_index: usize, projection_index: usize) []const u8 {
            std.debug.assert(self.active and !self.route.repaired[route_index]);
            return (Lease{ .owner = self.owner, .route = self.route }).projection(route_index, projection_index);
        }

        /// Caller drains GPU readers before aborting or releasing the final lease.
        pub fn abort(self: *Pending) void {
            if (!self.active) return;
            if (self.io) |io| self.group.cancel(io);
            for (self.lanes) |lane| self.owner.stats.bytes_read +|= lane.bytes;
            self.owner.leased = false;
            self.active = false;
        }

        pub fn finish(self: *Pending) !Lease {
            std.debug.assert(self.active);
            // Errors leave this transaction active so the caller drains GPU work
            // before abort releases the route's ownership.
            if (self.io) |io| try self.group.await(io);
            if (self.owner.cancelled) |stop| if (stop.load(.acquire)) return error.Cancelled;
            for (self.lanes) |lane| if (lane.failure) |err| {
                self.owner.stats.read_failures +|= 1;
                return err;
            };
            var cursor: usize = 0;
            for (self.route.slots(), self.route.repaired[0..self.route.count]) |slot, miss| {
                if (!miss) continue;
                @memcpy(self.owner.bytes[@as(usize, slot) * self.owner.stride ..][0..self.owner.stride], self.staging[cursor..][0..self.owner.stride]);
                cursor += self.owner.stride;
            }
            for (self.lanes) |lane| self.owner.stats.bytes_read +|= lane.bytes;
            self.owner.directory = self.next;
            self.owner.stats.hits +|= self.route.hit_count;
            self.owner.stats.misses +|= self.route.miss_count;
            self.active = false;
            return .{ .owner = self.owner, .route = self.route };
        }
    };

    pub fn acquireWithIo(self: *Layer, experts: []const u32, staging: []u8, io: ?std.Io) !Lease {
        var pending: Pending = .{};
        try pending.begin(self, experts, staging, io);
        defer pending.abort();
        return pending.finish();
    }
};

/// Fixed-capacity direct-mapped row cache. Collisions are harmless reloads;
/// bytes are published only after a full row read. The 16 output rows are
/// copied into caller-owned staging before Metal sees them.
pub const NgramCache = struct {
    reader: c_file.RangeReader,
    offset: u64,
    rows: u64,
    row_bytes: usize,
    entries: []Entry,
    bytes: []u8,
    allocator: std.mem.Allocator,
    stats: Stats = .{},
    const Entry = struct { row: u64 = std.math.maxInt(u64) };

    pub fn init(allocator: std.mem.Allocator, store: store_mod.TensorStore, budget: usize) !NgramCache {
        var range = (try store.describeTensorRange(allocator, "per_layer_token_embd.weight")) orelse return error.MissingTensorRange;
        defer range.deinit(allocator);
        if (range.shape.len != 2 or range.shape[1] != 160 or range.shape[0] <= 0 or range.gguf_type == null or range.gguf_type.?.raw() != 20) return error.InvalidNgramRange;
        const rows: u64 = @intCast(range.shape[0]);
        if (range.byte_len % rows != 0) return error.InvalidNgramRange;
        const row_bytes = range.byte_len / @as(usize, @intCast(rows));
        const count = budget / (row_bytes + @sizeOf(Entry));
        if (count == 0) return error.NgramCacheBudgetTooSmall;
        const reader = try c_file.RangeReader.open(allocator, range.path, true);
        errdefer {
            var r = reader;
            r.deinit();
        }
        if (range.byte_offset > reader.size or range.byte_len > reader.size - range.byte_offset) return error.RegionOutOfBounds;
        const entries = try allocator.alloc(Entry, count);
        errdefer allocator.free(entries);
        @memset(entries, .{});
        const bytes = try allocator.alloc(u8, count * row_bytes);
        return .{ .reader = reader, .offset = range.byte_offset, .rows = rows, .row_bytes = row_bytes, .entries = entries, .bytes = bytes, .allocator = allocator };
    }

    pub fn deinit(self: *NgramCache) void {
        self.reader.deinit();
        self.allocator.free(self.entries);
        self.allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn readRows(self: *NgramCache, rows: []const u64, out: []u8) !void {
        if (try std.math.mul(usize, rows.len, self.row_bytes) != out.len) return error.InvalidNgramOutput;
        for (rows) |row| if (row >= self.rows) return error.NgramRowOutOfBounds;
        for (rows, 0..) |row, i| {
            const index = row % self.entries.len;
            const entry = &self.entries[index];
            const cached = self.bytes[index * self.row_bytes ..][0..self.row_bytes];
            const destination = out[i * self.row_bytes ..][0..self.row_bytes];
            if (entry.row == row) {
                @memcpy(destination, cached);
                self.stats.hits +|= 1;
            } else {
                self.reader.read(try std.math.add(u64, self.offset, try std.math.mul(u64, row, self.row_bytes)), destination) catch |err| {
                    self.stats.read_failures +|= 1;
                    return err;
                };
                @memcpy(cached, destination);
                entry.row = row;
                self.stats.misses +|= 1;
                self.stats.bytes_read +|= self.row_bytes;
            }
        }
    }
};

test "Qwen4Exp pager holds leases and rolls back late short reads without corrupting victims" {
    const a = std.testing.allocator;
    var executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(read_lane_count), .concurrent_limit = .limited(read_lane_count), .stack_size = read_stack_bytes });
    defer executor.deinit();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var data: [512 * 4]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @truncate(i / 4);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts", .data = &data });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "experts" });
    defer a.free(path);
    var ps: [3]Projection = undefined;
    var opened: usize = 0;
    errdefer for (ps[0..opened]) |*p| p.reader.deinit();
    for (&ps) |*p| {
        p.* = .{ .reader = try c_file.RangeReader.open(a, path, false), .offset = 0, .stride = 4, .format = .{ .known = .F32 }, .input = 1, .output = 1 };
        opened += 1;
    }
    var layer = try Layer.initOwned(ps, 10, 12);
    opened = 0;
    defer layer.deinit();
    var staging: [120]u8 = undefined;
    const route = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 511 };
    var lease = try layer.acquire(&route, &staging);
    try std.testing.expectEqual(@as(usize, 10), lease.route.miss_count);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255, 255 }, lease.projection(9, 2));
    try std.testing.expectError(error.ExpertSlotsInUse, layer.acquire(&route, &staging));
    lease.releaseAfterCompletion();
    lease = try layer.acquire(&route, &staging);
    try std.testing.expectEqual(@as(usize, 10), lease.route.hit_count);
    lease.releaseAfterCompletion();
    // A mixed in-flight route protects every resident expert, including when
    // the transaction is abandoned. No unpublished slot may become visible.
    var pending: Layer.Pending = .{};
    const resident_before = layer.directory;
    try pending.begin(&layer, &.{ 0, 10, 2, 11, 4, 12, 6, 13, 8, 14 }, &staging, executor.io());
    try std.testing.expectEqual(@as(usize, 5), pending.route.hit_count);
    try std.testing.expectEqualDeep(resident_before, layer.directory);
    try std.testing.expectEqualSlices(u8, &.{ 2, 2, 2, 2 }, pending.residentProjection(2, 1));
    try std.testing.expectError(error.ExpertSlotsInUse, layer.acquire(&route, &staging));
    pending.abort();
    try std.testing.expect(!layer.leased);
    try std.testing.expectEqualDeep(resident_before, layer.directory);
    try pending.begin(&layer, &.{ 0, 10, 2, 11, 4, 12, 6, 13, 8, 14 }, &staging, executor.io());
    lease = try pending.finish();
    try std.testing.expectEqualSlices(u8, &.{ 14, 14, 14, 14 }, lease.projection(9, 2));
    try std.testing.expectEqualSlices(u8, &.{ 2, 2, 2, 2 }, lease.projection(2, 1));
    lease.releaseAfterCompletion();
    const before = layer.directory;
    const bytes = try a.dupe(u8, layer.bytes[0..layer.logical_bytes]);
    defer a.free(bytes);
    var stop: std.atomic.Value(bool) = .init(false);
    layer.cancelled = &stop;
    try pending.begin(&layer, &.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, &staging, executor.io());
    stop.store(true, .release);
    try std.testing.expectError(error.Cancelled, pending.finish());
    pending.abort();
    try std.testing.expect(!layer.leased);
    try std.testing.expectEqualDeep(before, layer.directory);
    try std.testing.expectEqualSlices(u8, bytes, layer.bytes[0..layer.logical_bytes]);
    try std.testing.expectError(error.Cancelled, layer.acquire(&route, &staging));
    stop.store(false, .release);
    // Make the last projection fail after gate and up have been read.
    layer.projections[2].offset = data.len;
    try std.testing.expectError(error.RegionOutOfBounds, layer.acquireWithIo(&.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, &staging, executor.io()));
    try std.testing.expectEqualDeep(before, layer.directory);
    try std.testing.expectEqualSlices(u8, bytes, layer.bytes[0..layer.logical_bytes]);
    layer.projections[2].offset = 0;
    // The cached file size cannot hide an actual short read after truncation.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts", .data = data[0..4] });
    try std.testing.expectError(error.IncompleteRead, layer.acquireWithIo(&.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, &staging, executor.io()));
    try std.testing.expect(!layer.leased);
    try std.testing.expectEqualDeep(before, layer.directory);
    try std.testing.expectEqualSlices(u8, bytes, layer.bytes[0..layer.logical_bytes]);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "experts", .data = &data });
    // Simulate an unavailable read endpoint without touching the model volume.
    {
        const fd = layer.projections[2].reader.fd;
        layer.projections[2].reader.fd = -1;
        defer layer.projections[2].reader.fd = fd;
        try std.testing.expectError(error.IncompleteRead, layer.acquireWithIo(&.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, &staging, executor.io()));
        try std.testing.expect(!layer.leased);
        try std.testing.expectEqualDeep(before, layer.directory);
        try std.testing.expectEqualSlices(u8, bytes, layer.bytes[0..layer.logical_bytes]);
    }
    lease = try layer.acquireWithIo(&.{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }, &staging, executor.io());
    defer lease.releaseAfterCompletion();
    try std.testing.expectEqualSlices(u8, &.{ 19, 19, 19, 19 }, lease.projection(9, 2));
}

test "Qwen4Exp ngram cache copies colliding rows and preserves a victim after read failure" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "rows", .data = &.{ 11, 12, 21, 22 } });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path, "rows" });
    defer a.free(path);
    var entries = [_]NgramCache.Entry{.{}};
    var bytes: [2]u8 = undefined;
    var cache = NgramCache{
        .reader = try c_file.RangeReader.open(a, path, false),
        .offset = 0,
        .rows = 2,
        .row_bytes = 2,
        .entries = &entries,
        .bytes = &bytes,
        .allocator = a,
    };
    defer cache.reader.deinit();
    var out: [6]u8 = undefined;
    try cache.readRows(&.{ 0, 1, 0 }, &out);
    try std.testing.expectEqualSlices(u8, &.{ 11, 12, 21, 22, 11, 12 }, &out);
    cache.offset = 4;
    try std.testing.expectError(error.RegionOutOfBounds, cache.readRows(&.{1}, out[0..2]));
    try std.testing.expectEqual(@as(u64, 0), cache.entries[0].row);
    try cache.readRows(&.{0}, out[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 11, 12 }, out[0..2]);
    try std.testing.expectError(error.NgramRowOutOfBounds, cache.readRows(&.{2}, out[0..2]));
}
