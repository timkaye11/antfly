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

//! Two fixed slots. Publication and reuse require completed I/O / GPU leases.
const std = @import("std");
const model = @import("../models/qwen35_streamed.zig");
const c_file = @import("../util/c_file.zig");
pub const State = enum { empty, reading, ready, gpu, failed };
pub const Slot = struct {
    bytes: []u8,
    state: State = .empty,
    layer: usize = 0,
    span: model.Span = .{ .offset = 0, .bytes = 0 },
    group: std.Io.Group = .init,
    io: ?std.Io = null,
    reader: *const c_file.RangeReader = undefined,
    cancelled: *const std.atomic.Value(bool) = undefined,
    failure: ?anyerror = null,
    bytes_read: u64 = 0,
    total_bytes_read: u64 = 0,
    generation: u64 = 0,
    // Stable address required until finish/drain. One bounded read worker.
    pub fn begin(self: *Slot, reader: *const c_file.RangeReader, layer: usize, span: model.Span, io: ?std.Io, cancelled: *const std.atomic.Value(bool)) !void {
        if (self.state != .empty) return error.StreamedSlotBusy;
        if (span.bytes == 0 or span.bytes > self.bytes.len) return error.StreamedSlotTooSmall;
        self.state = .reading;
        self.layer = layer;
        self.span = span;
        self.reader = reader;
        self.cancelled = cancelled;
        self.io = io;
        self.failure = null;
        self.bytes_read = 0;
        self.generation += 1;
        if (io) |i| self.group.async(i, read, .{self}) else self.read();
    }
    fn read(self: *Slot) void {
        var off: usize = 0;
        while (off < self.span.bytes) {
            if (self.cancelled.load(.acquire)) {
                self.failure = error.Cancelled;
                return;
            }
            const end = @min(off + 3 * model.MiB, self.span.bytes);
            self.reader.read(self.span.offset + off, self.bytes[off..end]) catch |err| {
                self.failure = err;
                return;
            };
            self.bytes_read += end - off;
            off = end;
        }
    }
    pub fn finish(self: *Slot) !void {
        if (self.state != .reading) return error.InvalidStreamedSlotState;
        if (self.io) |io| try self.group.await(io);
        self.total_bytes_read += self.bytes_read;
        if (self.failure) |err| {
            self.state = .failed;
            return err;
        }
        self.state = .ready;
    }
    pub fn acquire(self: *Slot) !Lease {
        if (self.state != .ready) return error.InvalidStreamedSlotState;
        self.state = .gpu;
        return .{ .slot = self, .generation = self.generation };
    }
    /// GPU must have been drained by the execution owner first.
    pub fn drain(self: *Slot) void {
        if (self.state == .reading) {
            if (self.io) |io| self.group.cancel(io);
            self.total_bytes_read += self.bytes_read;
        }
        self.state = .empty;
    }
};
pub const Lease = struct {
    slot: *Slot,
    generation: u64,
    pub fn releaseAfterCompletion(self: *Lease) !void {
        if (self.slot.state != .gpu or self.slot.generation != self.generation) return error.StaleStreamedLease;
        self.slot.state = .empty;
    }
};
test "Qwen35 streamed pager rejects reuse until GPU completion and stale releases" {
    var bytes: [32]u8 = undefined;
    var slot: Slot = .{ .bytes = &bytes, .state = .ready, .generation = 1 };
    var lease = try slot.acquire();
    try std.testing.expectError(error.InvalidStreamedSlotState, slot.acquire());
    try lease.releaseAfterCompletion();
    try std.testing.expectError(error.StaleStreamedLease, lease.releaseAfterCompletion());
    slot.state = .ready;
    slot.generation = 2;
    var next = try slot.acquire();
    try std.testing.expectError(error.StaleStreamedLease, lease.releaseAfterCompletion());
    try next.releaseAfterCompletion();
}

test "Qwen35 pager joins asynchronous reads and rejects truncation and cancellation" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "weights", .data = "0123456789abcdef" });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "weights" });
    defer a.free(path);
    var reader = try c_file.RangeReader.open(a, path, false);
    defer reader.deinit();
    var bytes: [16]u8 = undefined;
    var stop: std.atomic.Value(bool) = .init(false);
    var slot: Slot = .{ .bytes = &bytes };
    defer slot.drain();
    try slot.begin(&reader, 0, .{ .offset = 4, .bytes = 8 }, std.testing.io, &stop);
    try std.testing.expectError(error.StreamedSlotBusy, slot.begin(&reader, 1, .{ .offset = 0, .bytes = 8 }, null, &stop));
    try slot.finish();
    try std.testing.expectEqualStrings("456789ab", bytes[0..8]);
    var lease = try slot.acquire();
    try lease.releaseAfterCompletion();
    // Simulate an SSD file truncated after its descriptor was opened.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "weights", .data = "x" });
    try slot.begin(&reader, 1, .{ .offset = 0, .bytes = 8 }, std.testing.io, &stop);
    try std.testing.expectError(error.IncompleteRead, slot.finish());
    try std.testing.expectEqual(State.failed, slot.state);
    try std.testing.expectError(error.InvalidStreamedSlotState, slot.acquire());
    slot.drain();
    stop.store(true, .release);
    try slot.begin(&reader, 2, .{ .offset = 0, .bytes = 1 }, std.testing.io, &stop);
    try std.testing.expectError(error.Cancelled, slot.finish());
    try std.testing.expectEqual(@as(u64, 8), slot.total_bytes_read);
}
