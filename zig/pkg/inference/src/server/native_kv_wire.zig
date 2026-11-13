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

//! Narrow streaming framing for one native KV checkpoint in AFATT001.
//! The caller parses metadata and admits decoded state before reading the file.
//! It must call finish after the checkpoint decoder and before publishing state.
const std = @import("std");
pub const content_type = "application/vnd.antfly.attachments.v1";
pub const v2_content_type = "application/vnd.antfly.attachments.v2";
pub const checkpoint_mime = "application/vnd.antfly.kv-checkpoint";
const magic = "AFATT001";

pub const Admission = struct {
    context: *anyopaque,
    acquire: *const fn (*anyopaque, usize) bool,
    release: *const fn (*anyopaque, usize) void,
};
pub const Limits = struct {
    max_metadata_bytes: usize = 2 * 1024 * 1024,
    max_checkpoint_bytes: usize = 2 * 1024 * 1024 * 1024,
    admission: ?Admission = null,
};
pub const Start = struct {
    allocator: std.mem.Allocator,
    metadata: []u8,
    checkpoint_bytes: usize,
    frame_bytes: usize,
    admission: ?Admission,
    pub fn deinit(self: *@This()) void {
        const bytes = self.metadata.len;
        self.allocator.free(self.metadata);
        if (self.admission) |admission| admission.release(admission.context, bytes);
        self.* = undefined;
    }
};

fn totalBytes(metadata: usize, payload: usize) !usize {
    const prefix = std.math.add(usize, 40 + checkpoint_mime.len, metadata) catch return error.NativeKvFrameLimit;
    return std.math.add(usize, prefix, payload) catch error.NativeKvFrameLimit;
}
fn validateLengths(metadata: usize, payload: usize, limits: Limits) !usize {
    if (metadata == 0 or metadata > limits.max_metadata_bytes or payload < 184 or payload > limits.max_checkpoint_bytes) return error.NativeKvFrameLimit;
    return totalBytes(metadata, payload);
}
fn readExact(reader: anytype, bytes: []u8, control: anytype) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        try control.check();
        const requested = @min(16384, bytes.len - offset);
        const count = try reader.read(bytes[offset..][0..requested]);
        if (count == 0 or count > requested) return error.InvalidNativeKvFrame;
        offset += count;
    }
}

/// Allocates metadata only; the checkpoint body remains in the reader.
/// Control must provide check()!void, as the inference work-control type does.
pub fn readStart(allocator: std.mem.Allocator, reader: anytype, limits: Limits, control: anytype) !Start {
    try control.check();
    var header: [40]u8 = undefined;
    try readExact(reader, &header, control);
    if (!std.mem.eql(u8, header[0..8], magic) or
        std.mem.readInt(u32, header[16..20], .little) != 1 or
        std.mem.readInt(u32, header[20..24], .little) != 0 or
        std.mem.readInt(u32, header[24..28], .little) != checkpoint_mime.len or
        std.mem.readInt(u32, header[28..32], .little) != 0) return error.InvalidNativeKvFrame;
    const metadata_bytes = std.math.cast(usize, std.mem.readInt(u64, header[8..16], .little)) orelse return error.NativeKvFrameLimit;
    const checkpoint_bytes = std.math.cast(usize, std.mem.readInt(u64, header[32..40], .little)) orelse return error.NativeKvFrameLimit;
    const frame_bytes = try validateLengths(metadata_bytes, checkpoint_bytes, limits);
    if (limits.admission) |admission| if (!admission.acquire(admission.context, metadata_bytes)) return error.NativeKvFrameBusy;
    errdefer if (limits.admission) |admission| admission.release(admission.context, metadata_bytes);
    const metadata = try allocator.alloc(u8, metadata_bytes);
    errdefer allocator.free(metadata);
    try readExact(reader, metadata, control);
    var mime: [checkpoint_mime.len]u8 = undefined;
    try readExact(reader, &mime, control);
    if (!std.mem.eql(u8, &mime, checkpoint_mime)) return error.InvalidNativeKvFrame;
    try control.check();
    return .{ .allocator = allocator, .metadata = metadata, .checkpoint_bytes = checkpoint_bytes, .frame_bytes = frame_bytes, .admission = limits.admission };
}

/// Stream this prefix, then exportV2 directly into the same sink.
pub fn writeStart(sink: anytype, metadata: []const u8, checkpoint_bytes: usize, limits: Limits, control: anytype) !usize {
    try control.check();
    const size = try validateLengths(metadata.len, checkpoint_bytes, limits);
    var header: [40]u8 = @splat(0);
    @memcpy(header[0..8], magic);
    std.mem.writeInt(u64, header[8..16], @intCast(metadata.len), .little);
    std.mem.writeInt(u32, header[16..20], 1, .little);
    std.mem.writeInt(u32, header[24..28], checkpoint_mime.len, .little);
    std.mem.writeInt(u64, header[32..40], @intCast(checkpoint_bytes), .little);
    try sink.writeAll(&header);
    var offset: usize = 0;
    while (offset < metadata.len) {
        try control.check();
        const end = offset + @min(16384, metadata.len - offset);
        try sink.writeAll(metadata[offset..end]);
        offset = end;
    }
    try control.check();
    try sink.writeAll(checkpoint_mime);
    return size;
}

pub fn finish(reader: anytype, control: anytype) !void {
    try control.check();
    var extra: [1]u8 = undefined;
    if (try reader.read(&extra) != 0) return error.InvalidNativeKvFrame;
    try control.check();
}

pub const v2_magic = "AFATT002";
pub const v2_trailer_bytes: usize = 32;
pub const max_prepared_metadata_bytes: usize = 16 * 1024;
pub const PreparedOutcome = enum { compacted, unchanged_insufficient_savings };

pub const PreparedProducer = struct {
    schema: []const u8 = "antfly.native-kv-prepared-producer.v1",
    model: []const u8,
    model_artifact_sha256_hex: []const u8,
    model_tokenizer_sha256_hex: []const u8,
    model_template_sha256_hex: []const u8,
    model_layout_sha256_hex: []const u8,
    profile_sha256_hex: []const u8,
    profile_semantic_sha256_hex: []const u8,
    source_sha256_hex: []const u8,
    binary_sha256_hex: []const u8,
    policy_name: []const u8,
    policy_sha256_hex: []const u8,
    /// Digest of the private prepared-quality evidence core. Ordinary
    /// producers leave this null.
    quality_evidence_core_sha256_hex: ?[]const u8 = null,
    admission: []const u8,
    chunk_rows: u16,
    reject_exact_active_fixed_point: bool,
    encoding: []const u8,
    logical_revision: u64,
    logical_tokens: u64,
    checkpoint_bytes: u64,
    outcome: PreparedOutcome,
};

pub const V2Metadata = struct {
    schema: []const u8 = "antfly.native-kv-envelope.v2",
    /// Base64 of canonical PreparedProducer JSON, decoded size <= 16 KiB.
    /// Resume clients forward this string byte-for-byte.
    prepared_metadata_base64: []const u8,
    /// Optional new generation request. This is not part of producer authority.
    request_json: ?[]const u8 = null,
};

pub const V2Start = struct {
    inner: Start,
    pub fn deinit(self: *@This()) void {
        self.inner.deinit();
        self.* = undefined;
    }
};

fn v2FrameBytes(metadata: usize, payload: usize, limits: Limits) !usize {
    return std.math.add(usize, try validateLengths(metadata, payload, limits), v2_trailer_bytes) catch error.NativeKvFrameLimit;
}

fn v2Header(metadata: usize, payload: usize) [40]u8 {
    var header: [40]u8 = @splat(0);
    @memcpy(header[0..8], v2_magic);
    std.mem.writeInt(u64, header[8..16], @intCast(metadata), .little);
    // AFATT header word 16 is attachment count, not framing version.
    std.mem.writeInt(u32, header[16..20], 1, .little);
    std.mem.writeInt(u32, header[24..28], checkpoint_mime.len, .little);
    std.mem.writeInt(u64, header[32..40], @intCast(payload), .little);
    return header;
}

pub fn writeV2Start(sink: anytype, metadata: []const u8, payload: usize, limits: Limits, control: anytype) !usize {
    try control.check();
    const size = try v2FrameBytes(metadata.len, payload, limits);
    const header = v2Header(metadata.len, payload);
    try sink.writeAll(&header);
    var offset: usize = 0;
    while (offset < metadata.len) {
        try control.check();
        const end = offset + @min(16384, metadata.len - offset);
        try sink.writeAll(metadata[offset..end]);
        offset = end;
    }
    try control.check();
    try sink.writeAll(checkpoint_mime);
    return size;
}

pub fn readV2Start(allocator: std.mem.Allocator, reader: anytype, limits: Limits, control: anytype) !V2Start {
    try control.check();
    var header: [40]u8 = undefined;
    try readExact(reader, &header, control);
    if (!std.mem.eql(u8, header[0..8], v2_magic) or std.mem.readInt(u32, header[16..20], .little) != 1 or
        std.mem.readInt(u32, header[20..24], .little) != 0 or std.mem.readInt(u32, header[24..28], .little) != checkpoint_mime.len or
        std.mem.readInt(u32, header[28..32], .little) != 0) return error.InvalidNativeKvFrame;
    const metadata_bytes = std.math.cast(usize, std.mem.readInt(u64, header[8..16], .little)) orelse return error.NativeKvFrameLimit;
    const payload = std.math.cast(usize, std.mem.readInt(u64, header[32..40], .little)) orelse return error.NativeKvFrameLimit;
    const frame = try v2FrameBytes(metadata_bytes, payload, limits);
    if (limits.admission) |admission| if (!admission.acquire(admission.context, metadata_bytes)) return error.NativeKvFrameBusy;
    errdefer if (limits.admission) |admission| admission.release(admission.context, metadata_bytes);
    const metadata = try allocator.alloc(u8, metadata_bytes);
    errdefer allocator.free(metadata);
    try readExact(reader, metadata, control);
    var mime: [checkpoint_mime.len]u8 = undefined;
    try readExact(reader, &mime, control);
    if (!std.mem.eql(u8, &mime, checkpoint_mime)) return error.InvalidNativeKvFrame;
    return .{ .inner = .{ .allocator = allocator, .metadata = metadata, .checkpoint_bytes = payload, .frame_bytes = frame, .admission = limits.admission } };
}

pub fn V2HashingWriter(comptime SinkType: type) type {
    return struct {
        sink: *SinkType,
        remaining: usize,
        hash: std.crypto.hash.sha2.Sha256,
        control: *const fn (?*anyopaque) anyerror!void,
        control_context: ?*anyopaque,
        finished: bool = false,

        pub fn init(sink: *SinkType, metadata: []const u8, payload: usize, control_context: ?*anyopaque, control: *const fn (?*anyopaque) anyerror!void) @This() {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            const header = v2Header(metadata.len, payload);
            hash.update(&header);
            hash.update(metadata);
            hash.update(checkpoint_mime);
            return .{ .sink = sink, .remaining = payload, .hash = hash, .control = control, .control_context = control_context };
        }
        pub fn writeAll(self: *@This(), bytes: []const u8) !void {
            if (bytes.len > self.remaining) return error.NativeKvFrameLimit;
            if (self.finished) return error.InvalidNativeKvFrame;
            var offset: usize = 0;
            while (offset < bytes.len) {
                try self.control(self.control_context);
                const end = offset + @min(16384, bytes.len - offset);
                try self.sink.writeAll(bytes[offset..end]);
                self.hash.update(bytes[offset..end]);
                offset = end;
            }
            self.remaining -= bytes.len;
        }
        pub fn finish(self: *@This()) ![32]u8 {
            if (self.finished or self.remaining != 0) return error.InvalidNativeKvFrame;
            try self.control(self.control_context);
            var digest: [32]u8 = undefined;
            self.hash.final(&digest);
            try self.sink.writeAll(&digest);
            self.finished = true;
            return digest;
        }
    };
}

pub fn V2HashingReader(comptime ReaderType: type) type {
    return struct {
        reader: *ReaderType,
        remaining: usize,
        hash: std.crypto.hash.sha2.Sha256,
        finished: bool = false,
        pub fn init(reader: *ReaderType, metadata: []const u8, payload: usize) @This() {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            const header = v2Header(metadata.len, payload);
            hash.update(&header);
            hash.update(metadata);
            hash.update(checkpoint_mime);
            return .{ .reader = reader, .remaining = payload, .hash = hash };
        }
        pub fn read(self: *@This(), output: []u8) !usize {
            if (self.remaining == 0) return 0;
            if (self.finished) return error.InvalidNativeKvFrame;
            const requested = @min(output.len, self.remaining, 16384);
            const count = try self.reader.read(output[0..requested]);
            if (count == 0 or count > requested) return error.InvalidNativeKvFrame;
            self.hash.update(output[0..count]);
            self.remaining -= count;
            return count;
        }
        pub fn finish(self: *@This(), control: anytype) ![32]u8 {
            if (self.finished or self.remaining != 0) return error.InvalidNativeKvFrame;
            var expected: [32]u8 = undefined;
            try readExact(self.reader, &expected, control);
            var actual: [32]u8 = undefined;
            self.hash.final(&actual);
            if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidNativeKvFrame;
            try wire.finish(self.reader, control);
            self.finished = true;
            return actual;
        }
    };
}

const wire = @This();
const Control = struct {
    calls: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    pub fn check(self: *@This()) !void {
        const call = self.calls;
        self.calls += 1;
        if (call == self.fail_at) return error.Cancelled;
    }
};
const Reader = struct {
    bytes: []const u8,
    offset: usize = 0,
    calls: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    pub fn read(self: *@This(), dest: []u8) !usize {
        const call = self.calls;
        self.calls += 1;
        if (call == self.fail_at) return error.ReadFailed;
        if (dest.len > 16384) return error.UnboundedRead;
        const n = @min(dest.len, self.bytes.len - self.offset, 7);
        @memcpy(dest[0..n], self.bytes[self.offset..][0..n]);
        self.offset += n;
        return n;
    }
};
const Sink = struct {
    bytes: [65536]u8 = undefined,
    offset: usize = 0,
    calls: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    pub fn writeAll(self: *@This(), bytes: []const u8) !void {
        const call = self.calls;
        self.calls += 1;
        if (call == self.fail_at) return error.WriteFailed;
        if (bytes.len > 16384) return error.UnboundedWrite;
        @memcpy(self.bytes[self.offset..][0..bytes.len], bytes);
        self.offset += bytes.len;
    }
};
const Ledger = struct {
    live: usize = 0,
    allowed: bool = true,
    fn acquire(raw: *anyopaque, bytes: usize) bool {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (!self.allowed) return false;
        self.live += bytes;
        return true;
    }
    fn release(raw: *anyopaque, bytes: usize) void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        std.debug.assert(bytes <= self.live);
        self.live -= bytes;
    }
    fn admission(self: *@This()) wire.Admission {
        return .{ .context = self, .acquire = acquire, .release = release };
    }
};
fn exercise(allocator: std.mem.Allocator) !void {
    var sink: Sink = .{};
    var control: Control = .{};
    const size = try wire.writeStart(&sink, "{}", 184, .{}, &control);
    const prefix = sink.offset;
    try sink.writeAll(&@as([184]u8, @splat(42)));
    try std.testing.expectEqual(size, sink.offset);
    var ledger: Ledger = .{};
    var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
    {
        var start = try wire.readStart(allocator, &reader, .{ .admission = ledger.admission() }, &control);
        defer start.deinit();
        try std.testing.expectEqual(2, ledger.live);
        try std.testing.expectEqual(prefix, reader.offset);
        try std.testing.expectEqual(size, start.frame_bytes);
        try std.testing.expectEqual(184, start.checkpoint_bytes);
        try std.testing.expectEqualStrings("{}", start.metadata);
        try std.testing.expectError(error.InvalidNativeKvFrame, wire.finish(&reader, &control));
        reader.offset = reader.bytes.len;
        try wire.finish(&reader, &control);
    }
    try std.testing.expectEqual(0, ledger.live);
}
test "streaming native checkpoint framing leaves payload unbuffered and balances admission" {
    try exercise(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exercise, .{});
}
test "streaming native checkpoint framing rejects every truncated prefix and malformed descriptor" {
    var sink: Sink = .{};
    var control: Control = .{};
    _ = try wire.writeStart(&sink, "{}", 184, .{}, &control);
    var ledger: Ledger = .{};
    for (0..sink.offset) |end| {
        var reader: Reader = .{ .bytes = sink.bytes[0..end] };
        try std.testing.expectError(error.InvalidNativeKvFrame, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
        try std.testing.expectEqual(0, ledger.live);
    }
    for ([_]usize{ 0, 16, 20, 24, 28, 42 }) |index| {
        sink.bytes[index] ^= 1;
        defer sink.bytes[index] ^= 1;
        var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
        try std.testing.expectError(error.InvalidNativeKvFrame, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
        try std.testing.expectEqual(0, ledger.live);
    }
    for ([_]usize{ 8, 32 }) |index| {
        const saved = sink.bytes[index..][0..8].*;
        @memset(sink.bytes[index..][0..8], 255);
        defer @memcpy(sink.bytes[index..][0..8], &saved);
        var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
        try std.testing.expectError(error.NativeKvFrameLimit, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
        try std.testing.expectEqual(0, ledger.live);
    }
    ledger.allowed = false;
    var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
    try std.testing.expectError(error.NativeKvFrameBusy, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
    try std.testing.expectEqual(40, reader.offset);
    try std.testing.expectEqual(0, ledger.live);
}
test "streaming native checkpoint framing unwinds every read and cancellation boundary" {
    var sink: Sink = .{};
    var control: Control = .{};
    _ = try wire.writeStart(&sink, &@as([32769]u8, @splat('a')), 184, .{}, &control);
    var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
    var ledger: Ledger = .{};
    control = .{};
    var start = try wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control);
    start.deinit();
    const checks = control.calls;
    const reads = reader.calls;
    for (0..checks) |fail_at| {
        control = .{ .fail_at = fail_at };
        reader = .{ .bytes = sink.bytes[0..sink.offset] };
        try std.testing.expectError(error.Cancelled, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
        try std.testing.expectEqual(0, ledger.live);
    }
    for (0..reads) |fail_at| {
        control = .{};
        reader = .{ .bytes = sink.bytes[0..sink.offset], .fail_at = fail_at };
        try std.testing.expectError(error.ReadFailed, wire.readStart(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
        try std.testing.expectEqual(0, ledger.live);
    }
}
test "streaming native checkpoint framing bounds export writes and propagates sink failure" {
    const metadata = &@as([32769]u8, @splat('b'));
    var sink: Sink = .{};
    var control: Control = .{};
    _ = try wire.writeStart(&sink, metadata, 184, .{}, &control);
    const writes = sink.calls;
    const checks = control.calls;
    for (0..writes) |fail_at| {
        sink = .{ .fail_at = fail_at };
        control = .{};
        try std.testing.expectError(error.WriteFailed, wire.writeStart(&sink, metadata, 184, .{}, &control));
    }
    for (0..checks) |fail_at| {
        sink = .{};
        control = .{ .fail_at = fail_at };
        try std.testing.expectError(error.Cancelled, wire.writeStart(&sink, metadata, 184, .{}, &control));
    }
    for ([_]usize{ 0, 183, 2147483649 }) |bytes| {
        sink = .{};
        control = .{};
        try std.testing.expectError(error.NativeKvFrameLimit, wire.writeStart(&sink, metadata, bytes, .{}, &control));
        try std.testing.expectEqual(0, sink.offset);
    }
}

test "AFATT002 streams one pass and authenticates metadata plus exact payload" {
    const Check = struct {
        fn call(raw: ?*anyopaque) !void {
            const control: *Control = @ptrCast(@alignCast(raw orelse return error.Cancelled));
            try control.check();
        }
    };
    const metadata: [32769]u8 = @splat('m');
    const payload: [184]u8 = @splat(37);
    var sink: Sink = .{};
    var control: Control = .{};
    const frame_bytes = try wire.writeV2Start(&sink, &metadata, payload.len, .{}, &control);
    var hashing = wire.V2HashingWriter(Sink).init(&sink, &metadata, payload.len, &control, Check.call);
    try hashing.writeAll(payload[0..91]);
    try hashing.writeAll(payload[91..]);
    const written_digest = try hashing.finish();
    try std.testing.expectEqual(frame_bytes, sink.offset);

    var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
    var start = try wire.readV2Start(std.testing.allocator, &reader, .{}, &control);
    defer start.deinit();
    try std.testing.expectEqual(payload.len, start.inner.checkpoint_bytes);
    try std.testing.expectEqualSlices(u8, &metadata, start.inner.metadata);
    var verified = wire.V2HashingReader(Reader).init(&reader, start.inner.metadata, start.inner.checkpoint_bytes);
    var decoded: [184]u8 = undefined;
    var offset: usize = 0;
    while (offset < decoded.len) {
        const count = try verified.read(decoded[offset..]);
        if (count == 0) return error.InvalidNativeKvFrame;
        offset += count;
    }
    const read_digest = try verified.finish(&control);
    try std.testing.expectEqual(written_digest, read_digest);
    try std.testing.expectEqual(payload, decoded);

    const payload_offset = 40 + metadata.len + checkpoint_mime.len;
    sink.bytes[payload_offset + 17] ^= 1;
    reader = .{ .bytes = sink.bytes[0..sink.offset] };
    var corrupt = try wire.readV2Start(std.testing.allocator, &reader, .{}, &control);
    defer corrupt.deinit();
    var corrupt_reader = wire.V2HashingReader(Reader).init(&reader, corrupt.inner.metadata, corrupt.inner.checkpoint_bytes);
    offset = 0;
    while (offset < decoded.len) offset += try corrupt_reader.read(decoded[offset..]);
    try std.testing.expectError(error.InvalidNativeKvFrame, corrupt_reader.finish(&control));
    sink.bytes[payload_offset + 17] ^= 1;

    sink.bytes[40 + 19] ^= 1;
    reader = .{ .bytes = sink.bytes[0..sink.offset] };
    var corrupt_metadata = try wire.readV2Start(std.testing.allocator, &reader, .{}, &control);
    defer corrupt_metadata.deinit();
    var metadata_reader = wire.V2HashingReader(Reader).init(&reader, corrupt_metadata.inner.metadata, corrupt_metadata.inner.checkpoint_bytes);
    offset = 0;
    while (offset < decoded.len) offset += try metadata_reader.read(decoded[offset..]);
    try std.testing.expectError(error.InvalidNativeKvFrame, metadata_reader.finish(&control));
    sink.bytes[40 + 19] ^= 1;

    for (0..v2_trailer_bytes) |missing| {
        reader = .{ .bytes = sink.bytes[0 .. sink.offset - missing - 1] };
        var truncated = try wire.readV2Start(std.testing.allocator, &reader, .{}, &control);
        defer truncated.deinit();
        var truncated_reader = wire.V2HashingReader(Reader).init(&reader, truncated.inner.metadata, truncated.inner.checkpoint_bytes);
        offset = 0;
        while (offset < decoded.len) offset += try truncated_reader.read(decoded[offset..]);
        try std.testing.expectError(error.InvalidNativeKvFrame, truncated_reader.finish(&control));
    }

    try std.testing.expectError(error.InvalidNativeKvFrame, hashing.finish());
    try std.testing.expectError(error.InvalidNativeKvFrame, verified.finish(&control));
}

test "AFATT002 bounds admission allocation and cancellation" {
    var sink: Sink = .{};
    var control: Control = .{};
    _ = try wire.writeV2Start(&sink, "{}", 184, .{}, &control);
    var ledger: Ledger = .{};
    var reader: Reader = .{ .bytes = sink.bytes[0..sink.offset] };
    var tiny: [1]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&tiny);
    try std.testing.expectError(error.OutOfMemory, wire.readV2Start(fixed.allocator(), &reader, .{ .admission = ledger.admission() }, &control));
    try std.testing.expectEqual(0, ledger.live);

    ledger.allowed = false;
    reader = .{ .bytes = sink.bytes[0..sink.offset] };
    try std.testing.expectError(error.NativeKvFrameBusy, wire.readV2Start(std.testing.allocator, &reader, .{ .admission = ledger.admission() }, &control));
    try std.testing.expectEqual(0, ledger.live);

    sink = .{};
    control = .{ .fail_at = 0 };
    try std.testing.expectError(error.Cancelled, wire.writeV2Start(&sink, "{}", 184, .{}, &control));
    try std.testing.expectEqual(0, sink.offset);
    control = .{};
    try std.testing.expectError(error.NativeKvFrameLimit, wire.writeV2Start(&sink, "{}", 183, .{}, &control));
    try std.testing.expectEqual(0, sink.offset);
}
