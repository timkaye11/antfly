// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, test/development-only retention of the exact HHO domain that was
//! alive during prepared-quality fitting. The canonical digest is identical to
//! the resident final-heldout authority placed in the prepared closure.
const std = @import("std");
const engine = @import("attention_match_engine.zig");

pub const schema = "antfly.attention-match-prepared-heldout-bundle.v1";
const magic = "AFHHO001";
pub const maximum_bytes: usize = 16 * 1024 * 1024;
pub const expected_batches: usize = 56;
pub const expected_rows: usize = 64;
pub const expected_dim: usize = 512;
pub const StreamResult = struct { bytes: usize, sha256: [32]u8 };
pub const Cohort = enum { e4_8128_v1, e2_8128_v1 };

fn expectedLayer(cohort: Cohort, batch_index: usize) u32 {
    const stratum = batch_index / 8;
    return switch (cohort) {
        .e4_8128_v1 => @intCast(5 + stratum * 6),
        .e2_8128_v1 => @intCast(4 + stratum * 5),
    };
}

pub fn validateConsumerTuples(batches: []const engine.ReferenceBatch, cohort: Cohort) !void {
    if (batches.len != expected_batches) return error.InvalidPreparedHeldoutBundle;
    for (batches, 0..) |batch, index| {
        if (batch.consumer_layer != expectedLayer(cohort, index) or batch.query_head != index % 8 or
            batch.heldout_positions.len != expected_rows or batch.heldout.len != expected_rows * expected_dim)
            return error.InvalidPreparedHeldoutBundle;
    }
}

pub const Owned = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    authority_sha256: [32]u8,
    scales: []f64,
    softcaps: []f64,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.allocator.free(self.softcaps);
        self.allocator.free(self.scales);
        self.allocator.free(self.bytes);
        self.* = undefined;
    }
};

fn update(hash: *std.crypto.hash.sha2.Sha256, index: usize, batch: engine.ReferenceBatch) !void {
    if (batch.heldout_positions.len != expected_rows or batch.heldout.len != expected_rows * expected_dim) return error.InvalidPreparedHeldoutBundle;
    var scalar: [8]u8 = undefined;
    inline for (.{ index, @as(usize, batch.consumer_layer), @as(usize, batch.query_head), batch.heldout_positions.len, @as(usize, expected_dim) }) |value| {
        std.mem.writeInt(u64, &scalar, @intCast(value), .little);
        hash.update(&scalar);
    }
    hash.update(std.mem.sliceAsBytes(batch.heldout));
    hash.update(std.mem.sliceAsBytes(batch.heldout_positions));
}

pub fn canonicalDigest(batches: []const engine.ReferenceBatch) ![32]u8 {
    if (batches.len != expected_batches) return error.InvalidPreparedHeldoutBundle;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.resident-final-heldout-domain.v1\x00");
    for (batches, 0..) |batch, index| try update(&hash, index, batch);
    return hash.finalResult();
}

fn streamHashed(sink: anytype, hash_state: *std.crypto.hash.sha2.Sha256, count: *usize, bytes: []const u8) !void {
    try sink.writeAll(bytes);
    hash_state.update(bytes);
    count.* = try std.math.add(usize, count.*, bytes.len);
}

/// Emits the exact encodeAlloc wire image without retaining a second HHO-sized
/// allocation. The caller owns an exclusive temporary sink and publishes it
/// only after this function returns successfully.
pub fn writeStream(sink: anytype, batches: []const engine.ReferenceBatch, authority_sha256: [32]u8, scales: []const f64, softcaps: []const f64) !StreamResult {
    if (!std.mem.eql(u8, &try canonicalDigest(batches), &authority_sha256)) return error.PreparedHeldoutAuthorityMismatch;
    if (scales.len == 0 or scales.len != softcaps.len or scales.len > 32) return error.InvalidPreparedHeldoutBundle;
    var hash_state = std.crypto.hash.sha2.Sha256.init(.{});
    var count: usize = 0;
    try streamHashed(sink, &hash_state, &count, magic);
    var scalar4: [4]u8 = undefined;
    inline for (.{ @as(u32, 1), @as(u32, expected_batches), @as(u32, expected_rows), @as(u32, expected_dim), @as(u32, @intCast(scales.len)) }) |value| {
        std.mem.writeInt(u32, &scalar4, value, .little);
        try streamHashed(sink, &hash_state, &count, &scalar4);
    }
    try streamHashed(sink, &hash_state, &count, &authority_sha256);
    var scalar8: [8]u8 = undefined;
    for (scales, softcaps) |scale, softcap| {
        std.mem.writeInt(u64, &scalar8, @bitCast(scale), .little);
        try streamHashed(sink, &hash_state, &count, &scalar8);
        std.mem.writeInt(u64, &scalar8, @bitCast(softcap), .little);
        try streamHashed(sink, &hash_state, &count, &scalar8);
    }
    var descriptor: [16]u8 = undefined;
    for (batches) |batch| {
        std.mem.writeInt(u32, descriptor[0..4], batch.consumer_layer, .little);
        std.mem.writeInt(u32, descriptor[4..8], batch.query_head, .little);
        std.mem.writeInt(u32, descriptor[8..12], expected_rows, .little);
        std.mem.writeInt(u32, descriptor[12..16], expected_dim, .little);
        try streamHashed(sink, &hash_state, &count, &descriptor);
    }
    for (batches) |batch| try streamHashed(sink, &hash_state, &count, std.mem.sliceAsBytes(batch.heldout));
    for (batches) |batch| try streamHashed(sink, &hash_state, &count, std.mem.sliceAsBytes(batch.heldout_positions));
    const digest = hash_state.finalResult();
    try sink.writeAll(&digest);
    count = try std.math.add(usize, count, digest.len);
    if (count > maximum_bytes) return error.PreparedHeldoutBundleLimit;
    return .{ .bytes = count, .sha256 = digest };
}

/// The production publication entry point. Cohort validation is intentionally
/// outside `writeStream` so the retained E4 writer and its byte image remain
/// unchanged for existing diagnostic fixtures.
pub fn writeStreamForCohort(sink: anytype, batches: []const engine.ReferenceBatch, authority_sha256: [32]u8, scales: []const f64, softcaps: []const f64, cohort: Cohort) !StreamResult {
    try validateConsumerTuples(batches, cohort);
    return writeStream(sink, batches, authority_sha256, scales, softcaps);
}

pub fn encodeAlloc(allocator: std.mem.Allocator, batches: []const engine.ReferenceBatch, authority_sha256: [32]u8, scales: []const f64, softcaps: []const f64) ![]u8 {
    if (!std.mem.eql(u8, &try canonicalDigest(batches), &authority_sha256)) return error.PreparedHeldoutAuthorityMismatch;
    if (scales.len == 0 or scales.len != softcaps.len or scales.len > 32) return error.InvalidPreparedHeldoutBundle;
    const descriptor_bytes = expected_batches * 16;
    const value_bytes = expected_batches * expected_rows * expected_dim * @sizeOf(f32);
    const position_bytes = expected_batches * expected_rows * @sizeOf(u64);
    const total = 8 + 4 * 5 + 32 + scales.len * 16 + descriptor_bytes + value_bytes + position_bytes + 32;
    if (total > maximum_bytes) return error.PreparedHeldoutBundleLimit;
    const out = try allocator.alloc(u8, total);
    errdefer allocator.free(out);
    var at: usize = 0;
    @memcpy(out[at..][0..8], magic);
    at += 8;
    inline for (.{ @as(u32, 1), @as(u32, expected_batches), @as(u32, expected_rows), @as(u32, expected_dim), @as(u32, @intCast(scales.len)) }) |value| {
        std.mem.writeInt(u32, out[at..][0..4], value, .little);
        at += 4;
    }
    @memcpy(out[at..][0..32], &authority_sha256);
    at += 32;
    for (scales, softcaps) |scale, softcap| {
        std.mem.writeInt(u64, out[at..][0..8], @bitCast(scale), .little);
        at += 8;
        std.mem.writeInt(u64, out[at..][0..8], @bitCast(softcap), .little);
        at += 8;
    }
    for (batches) |batch| {
        std.mem.writeInt(u32, out[at..][0..4], batch.consumer_layer, .little);
        at += 4;
        std.mem.writeInt(u32, out[at..][0..4], batch.query_head, .little);
        at += 4;
        std.mem.writeInt(u32, out[at..][0..4], expected_rows, .little);
        at += 4;
        std.mem.writeInt(u32, out[at..][0..4], expected_dim, .little);
        at += 4;
    }
    for (batches) |batch| {
        const raw = std.mem.sliceAsBytes(batch.heldout);
        @memcpy(out[at..][0..raw.len], raw);
        at += raw.len;
    }
    for (batches) |batch| {
        const raw = std.mem.sliceAsBytes(batch.heldout_positions);
        @memcpy(out[at..][0..raw.len], raw);
        at += raw.len;
    }
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(out[0..at], &digest, .{});
    @memcpy(out[at..][0..32], &digest);
    at += 32;
    if (at != out.len) return error.InvalidPreparedHeldoutBundle;
    return out;
}

pub fn decodeOwned(allocator: std.mem.Allocator, encoded: []const u8, expected_authority: [32]u8) !Owned {
    if (encoded.len > maximum_bytes or encoded.len < 88 or !std.mem.eql(u8, encoded[0..8], magic)) return error.InvalidPreparedHeldoutBundle;
    var at: usize = 8;
    inline for (.{ @as(u32, 1), @as(u32, expected_batches), @as(u32, expected_rows), @as(u32, expected_dim) }) |expected| {
        if (std.mem.readInt(u32, encoded[at..][0..4], .little) != expected) return error.InvalidPreparedHeldoutBundle;
        at += 4;
    }
    const donor_count = std.mem.readInt(u32, encoded[at..][0..4], .little);
    at += 4;
    if (donor_count == 0 or donor_count > 32) return error.InvalidPreparedHeldoutBundle;
    var exact_bytes = try std.math.add(usize, at, 32);
    exact_bytes = try std.math.add(usize, exact_bytes, try std.math.mul(usize, donor_count, 16));
    exact_bytes = try std.math.add(usize, exact_bytes, expected_batches * 16);
    exact_bytes = try std.math.add(usize, exact_bytes, expected_batches * expected_rows * expected_dim * @sizeOf(f32));
    exact_bytes = try std.math.add(usize, exact_bytes, expected_batches * expected_rows * @sizeOf(u64));
    exact_bytes = try std.math.add(usize, exact_bytes, 32);
    if (encoded.len != exact_bytes) return error.InvalidPreparedHeldoutBundle;
    if (!std.mem.eql(u8, encoded[at..][0..32], &expected_authority)) return error.PreparedHeldoutAuthorityMismatch;
    at += 32;
    const scales = try allocator.alloc(f64, donor_count);
    errdefer allocator.free(scales);
    const softcaps = try allocator.alloc(f64, donor_count);
    errdefer allocator.free(softcaps);
    for (scales, softcaps) |*scale, *softcap| {
        scale.* = @bitCast(std.mem.readInt(u64, encoded[at..][0..8], .little));
        at += 8;
        softcap.* = @bitCast(std.mem.readInt(u64, encoded[at..][0..8], .little));
        at += 8;
    }
    const descriptors = encoded[at..][0 .. expected_batches * 16];
    at += descriptors.len;
    const values = try allocator.alloc(f32, expected_batches * expected_rows * expected_dim);
    errdefer allocator.free(values);
    const value_raw = std.mem.sliceAsBytes(values);
    @memcpy(value_raw, encoded[at..][0..value_raw.len]);
    at += value_raw.len;
    const positions = try allocator.alloc(u64, expected_batches * expected_rows);
    errdefer allocator.free(positions);
    const position_raw = std.mem.sliceAsBytes(positions);
    @memcpy(position_raw, encoded[at..][0..position_raw.len]);
    at += position_raw.len;
    if (at + 32 != encoded.len) return error.InvalidPreparedHeldoutBundle;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(encoded[0..at], &digest, .{});
    if (!std.mem.eql(u8, &digest, encoded[at..][0..32])) return error.PreparedHeldoutBundleChanged;
    const batches = try allocator.alloc(engine.ReferenceBatch, expected_batches);
    errdefer allocator.free(batches);
    for (batches, 0..) |*batch, i| {
        const d = descriptors[i * 16 ..][0..16];
        batch.* = .{ .train = &.{}, .train_positions = &.{}, .consumer_layer = @intCast(std.mem.readInt(u32, d[0..4], .little)), .query_head = @intCast(std.mem.readInt(u32, d[4..8], .little)), .heldout = values[i * expected_rows * expected_dim ..][0 .. expected_rows * expected_dim], .heldout_positions = positions[i * expected_rows ..][0..expected_rows] };
    }
    if (!std.mem.eql(u8, &try canonicalDigest(batches), &expected_authority)) return error.PreparedHeldoutAuthorityMismatch;
    return .{ .allocator = allocator, .bytes = try allocator.dupe(u8, encoded), .batches = batches, .values = values, .positions = positions, .authority_sha256 = expected_authority, .scales = scales, .softcaps = softcaps };
}

test "prepared heldout bundle round trips raw f32 bits positions and resident authority" {
    const allocator = std.testing.allocator;
    const values = try allocator.alloc(f32, expected_batches * expected_rows * expected_dim);
    defer allocator.free(values);
    const positions = try allocator.alloc(u64, expected_batches * expected_rows);
    defer allocator.free(positions);
    var batches: [expected_batches]engine.ReferenceBatch = undefined;
    for (&batches, 0..) |*batch, i| {
        const v = values[i * expected_rows * expected_dim ..][0 .. expected_rows * expected_dim];
        const p = positions[i * expected_rows ..][0..expected_rows];
        for (v, 0..) |*x, j| x.* = @bitCast(@as(u32, @intCast(0x3f000000 + (i * 17 + j) % 1024)));
        for (p, 0..) |*x, j| x.* = 8000 + j;
        batch.* = .{ .train = &.{}, .train_positions = &.{}, .consumer_layer = @intCast(i / 8), .query_head = @intCast(i % 8), .heldout = v, .heldout_positions = p };
    }
    const authority = try canonicalDigest(&batches);
    const encoded = try encodeAlloc(allocator, &batches, authority, &.{ 0.125, 0.25 }, &.{ 50, 0 });
    defer allocator.free(encoded);
    const Sink = struct {
        bytes: []u8,
        at: usize = 0,
        fn writeAll(self: *@This(), value: []const u8) !void {
            if (value.len > self.bytes.len - self.at) return error.TestUnexpectedResult;
            @memcpy(self.bytes[self.at..][0..value.len], value);
            self.at += value.len;
        }
    };
    const streamed = try allocator.alloc(u8, encoded.len);
    defer allocator.free(streamed);
    var sink: Sink = .{ .bytes = streamed };
    const stream_result = try writeStream(&sink, &batches, authority, &.{ 0.125, 0.25 }, &.{ 50, 0 });
    try std.testing.expectEqual(encoded.len, stream_result.bytes);
    try std.testing.expectEqualSlices(u8, encoded, streamed);
    var decoded = try decodeOwned(allocator, encoded, authority);
    defer decoded.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(values), std.mem.sliceAsBytes(decoded.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(positions), std.mem.sliceAsBytes(decoded.positions));
    try std.testing.expectEqualSlices(f64, &.{ 0.125, 0.25 }, decoded.scales);
    var changed = try allocator.dupe(u8, encoded);
    defer allocator.free(changed);
    changed[changed.len - 33] ^= 1;
    try std.testing.expectError(error.PreparedHeldoutBundleChanged, decodeOwned(allocator, changed, authority));
}

test "prepared heldout bundle validates exact E4 and E2 consumer tuples without changing E4 digest" {
    const values: [expected_rows * expected_dim]f32 = @splat(0);
    const positions: [expected_rows]u64 = @splat(8128);
    var e4: [expected_batches]engine.ReferenceBatch = undefined;
    var e2: [expected_batches]engine.ReferenceBatch = undefined;
    for (&e4, &e2, 0..) |*e4_batch, *e2_batch, index| {
        e4_batch.* = .{ .train = &.{}, .train_positions = &.{}, .consumer_layer = expectedLayer(.e4_8128_v1, index), .query_head = @intCast(index % 8), .heldout = &values, .heldout_positions = &positions };
        e2_batch.* = .{ .train = &.{}, .train_positions = &.{}, .consumer_layer = expectedLayer(.e2_8128_v1, index), .query_head = @intCast(index % 8), .heldout = &values, .heldout_positions = &positions };
    }
    const before = try canonicalDigest(&e4);
    try validateConsumerTuples(&e4, .e4_8128_v1);
    try validateConsumerTuples(&e2, .e2_8128_v1);
    try std.testing.expectEqualSlices(u8, &before, &try canonicalDigest(&e4));
    e2[55].consumer_layer = 29;
    try std.testing.expectError(error.InvalidPreparedHeldoutBundle, validateConsumerTuples(&e2, .e2_8128_v1));
    try std.testing.expectError(error.InvalidPreparedHeldoutBundle, validateConsumerTuples(&e4, .e2_8128_v1));
}

test "prepared heldout streaming failure leaves publication to caller" {
    const Sink = struct {
        writes: usize = 0,
        fn writeAll(self: *@This(), _: []const u8) !void {
            self.writes += 1;
            if (self.writes == 3) return error.InjectedWriteFailure;
        }
    };
    var values: [expected_rows * expected_dim]f32 = @splat(0);
    var positions: [expected_rows]u64 = @splat(8128);
    var batches: [expected_batches]engine.ReferenceBatch = undefined;
    for (&batches, 0..) |*batch, i| batch.* = .{ .train = &.{}, .train_positions = &.{}, .consumer_layer = @intCast(i / 8), .query_head = @intCast(i % 8), .heldout = &values, .heldout_positions = &positions };
    const authority = try canonicalDigest(&batches);
    var sink: Sink = .{};
    try std.testing.expectError(error.InjectedWriteFailure, writeStream(&sink, &batches, authority, &.{0.125}, &.{50}));
}
