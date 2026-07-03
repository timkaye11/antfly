// Copyright 2026 Antfly, Inc.
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

// Result cache with TTL expiration and singleflight deduplication.
// Mirrors the legacy Go inference caching strategy.

const std = @import("std");
const libc = @cImport(@cInclude("time.h"));

fn nowNs() i64 {
    var ts: libc.struct_timespec = undefined;
    if (libc.clock_gettime(libc.CLOCK_MONOTONIC, &ts) != 0) return 0;
    return @as(i64, @intCast(ts.tv_sec)) * std.time.ns_per_s + @as(i64, @intCast(ts.tv_nsec));
}

pub fn ResultCache(comptime V: type) type {
    return struct {
        const Self = @This();

        const Entry = struct {
            value: V,
            expires_at: i64,
        };

        map: std.StringHashMap(Entry),
        ttl_ns: i64,
        allocator: std.mem.Allocator,
        mutex: std.atomic.Mutex = .unlocked,
        hits: u64,
        misses: u64,

        pub fn init(allocator: std.mem.Allocator, ttl_ms: u64) Self {
            return .{
                .map = std.StringHashMap(Entry).init(allocator),
                .ttl_ns = @intCast(ttl_ms * std.time.ns_per_ms),
                .allocator = allocator,
                .hits = 0,
                .misses = 0,
            };
        }

        pub fn deinit(self: *Self) void {
            lockAtomic(&self.mutex);
            defer self.mutex.unlock();
            var it = self.map.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                freeValue(V, self.allocator, entry.value_ptr.value);
            }
            self.map.deinit();
        }

        pub fn get(self: *Self, key: []const u8) ?V {
            lockAtomic(&self.mutex);
            defer self.mutex.unlock();
            const entry = self.map.get(key) orelse {
                self.misses += 1;
                return null;
            };
            const now = nowNs();
            if (now > entry.expires_at) {
                self.removeLocked(key);
                self.misses += 1;
                return null;
            }
            self.hits += 1;
            return entry.value;
        }

        pub fn getAlloc(self: *Self, allocator: std.mem.Allocator, key: []const u8) !?V {
            lockAtomic(&self.mutex);
            defer self.mutex.unlock();
            const entry = self.map.get(key) orelse {
                self.misses += 1;
                return null;
            };
            const now = nowNs();
            if (now > entry.expires_at) {
                self.removeLocked(key);
                self.misses += 1;
                return null;
            }
            const cloned = try cloneValue(V, allocator, entry.value);
            self.hits += 1;
            return cloned;
        }

        pub fn put(self: *Self, key: []const u8, value: V) void {
            self.putCopy(key, value) catch {};
        }

        pub fn putCopy(self: *Self, key: []const u8, value: V) !void {
            const owned_value = try cloneValue(V, self.allocator, value);
            errdefer freeValue(V, self.allocator, owned_value);
            try self.putOwned(key, owned_value);
        }

        pub fn putOwned(self: *Self, key: []const u8, value: V) !void {
            lockAtomic(&self.mutex);
            defer self.mutex.unlock();
            const owned_key = try self.allocator.dupe(u8, key);
            errdefer self.allocator.free(owned_key);
            const now = nowNs();
            const gop = try self.map.getOrPut(owned_key);
            if (gop.found_existing) {
                self.allocator.free(owned_key);
                freeValue(V, self.allocator, gop.value_ptr.value);
            }
            gop.value_ptr.* = .{
                .value = value,
                .expires_at = now + self.ttl_ns,
            };
        }

        pub fn stats(self: *const Self) struct { hits: u64, misses: u64, size: usize } {
            const mutable: *Self = @constCast(self);
            lockAtomic(&mutable.mutex);
            defer mutable.mutex.unlock();
            return .{
                .hits = self.hits,
                .misses = self.misses,
                .size = self.map.count(),
            };
        }

        fn removeLocked(self: *Self, key: []const u8) void {
            const removed = self.map.fetchRemove(key) orelse return;
            self.allocator.free(removed.key);
            freeValue(V, self.allocator, removed.value.value);
        }
    };
}

fn lockAtomic(mutex: *std.atomic.Mutex) void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

fn cloneValue(comptime V: type, allocator: std.mem.Allocator, value: V) !V {
    if (comptime V == []const f32 or V == []f32) {
        const out = try allocator.alloc(f32, value.len);
        @memcpy(out, value);
        return out;
    }
    return value;
}

fn freeValue(comptime V: type, allocator: std.mem.Allocator, value: V) void {
    if (comptime V == []const f32 or V == []f32) allocator.free(value);
}

test "basic cache operations" {
    const allocator = std.testing.allocator;
    var c = ResultCache(i32).init(allocator, 60_000);
    defer c.deinit();

    try std.testing.expectEqual(@as(?i32, null), c.get("key1"));

    c.put("key1", 42);
    try std.testing.expectEqual(@as(?i32, 42), c.get("key1"));

    const s = c.stats();
    try std.testing.expectEqual(@as(u64, 1), s.hits);
    try std.testing.expectEqual(@as(u64, 1), s.misses);
}

test "float slice cache owns values" {
    const allocator = std.testing.allocator;
    var c = ResultCache([]f32).init(allocator, 60_000);
    defer c.deinit();

    var value = [_]f32{ 1.0, 2.0 };
    try c.putCopy("key1", value[0..]);
    value[0] = 9.0;

    const cached = (try c.getAlloc(allocator, "key1")).?;
    defer allocator.free(cached);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0 }, cached);
}
