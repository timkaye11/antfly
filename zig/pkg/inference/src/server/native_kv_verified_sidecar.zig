// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Constructs a qualification sidecar only from a completely verified
//! AFATT002 frame. It never accepts a naked checkpoint hash or claimed policy.
const std = @import("std");
const wire = @import("native_kv_wire.zig");
const metadata = @import("native_kv_v2_metadata.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");

pub const ExpectedProducer = struct {
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    binary_sha256: [32]u8,
    policy_sha256: [32]u8,
    policy_name: []const u8,
};

pub const Sidecar = struct {
    /// Existing soak consumers recognize this schema; unlike historical
    /// hand-written files, this value is emitted only by verifyFrame.
    schema: []const u8 = "antfly.native-kv-prepared-sidecar.v1",
    afatt002_sha256_hex: [64]u8,
    footer_sha256_hex: [64]u8,
    checkpoint_sha256_hex: [64]u8,
    producer_sha256_hex: [64]u8,
    policy_sha256_hex: [64]u8,
    model: []const u8,
    logical_tokens: u64,
    checkpoint_bytes: u64,
    encoding: []const u8,
};

pub const Verified = struct {
    allocator: std.mem.Allocator,
    decoded: checkpoint.Decoded,
    producer: metadata.Owned,
    sidecar: Sidecar,

    pub fn deinit(self: *Verified) void {
        self.decoded.checkpoint.deinit();
        self.producer.deinit();
        self.* = undefined;
    }
};

const SliceReader = struct {
    bytes: []const u8,
    offset: usize = 0,
    pub fn read(self: *@This(), output: []u8) !usize {
        if (self.offset == self.bytes.len) return 0;
        const count = @min(output.len, self.bytes.len - self.offset);
        @memcpy(output[0..count], self.bytes[self.offset..][0..count]);
        self.offset += count;
        return count;
    }
};

const Control = struct {
    pub fn check(_: *@This()) !void {}
};

pub fn verifyFrame(allocator: std.mem.Allocator, frame: []const u8, expected: checkpoint.Identity, expected_producer: ExpectedProducer, limits: checkpoint.Limits) !Verified {
    var control: Control = .{};
    var reader: SliceReader = .{ .bytes = frame };
    var start = try wire.readV2Start(allocator, &reader, .{}, &control);
    defer start.deinit();
    var producer = try metadata.Owned.parse(allocator, start.inner.metadata);
    errdefer producer.deinit();
    const producer_value = producer.producer.value;
    const profile = try digest(producer_value.profile_sha256_hex);
    const profile_semantic = try digest(producer_value.profile_semantic_sha256_hex);
    const implementation = try digest(producer_value.source_sha256_hex);
    const binary = try digest(producer_value.binary_sha256_hex);
    const policy = try digest(producer_value.policy_sha256_hex);
    if (!std.mem.eql(u8, producer_value.policy_name, expected_producer.policy_name) or
        !std.mem.eql(u8, &profile, &expected_producer.profile_sha256) or
        !std.mem.eql(u8, &profile_semantic, &expected_producer.profile_semantic_sha256) or
        !std.mem.eql(u8, &implementation, &expected_producer.implementation_semantic_sha256) or
        !std.mem.eql(u8, &binary, &expected_producer.binary_sha256) or
        !std.mem.eql(u8, &policy, &expected_producer.policy_sha256))
        return error.NativeKvProducerIdentityMismatch;
    var hashing = wire.V2HashingReader(SliceReader).init(&reader, start.inner.metadata, start.inner.checkpoint_bytes);
    var decoded = try checkpoint.decodeReaderWithEncodingAndControl(allocator, &hashing, start.inner.checkpoint_bytes, expected, limits, .{});
    errdefer decoded.checkpoint.deinit();
    const frame_footer = try hashing.finish(&control);
    try metadata.validateCheckpointIdentity(producer.producer.value, decoded.checkpoint.state.identity);
    if (producer.producer.value.logical_tokens != decoded.checkpoint.state.logical_length or
        producer.producer.value.checkpoint_bytes != start.inner.checkpoint_bytes or
        !std.mem.eql(u8, producer.producer.value.encoding, @tagName(decoded.encoding)))
        return error.NativeKvProducerIdentityMismatch;
    var checkpoint_hash: [32]u8 = undefined;
    const payload_offset = 40 + start.inner.metadata.len + wire.checkpoint_mime.len;
    std.crypto.hash.sha2.Sha256.hash(frame[payload_offset..][0..start.inner.checkpoint_bytes], &checkpoint_hash, .{});
    var producer_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(producer.producer_bytes, &producer_hash, .{});
    var frame_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(frame, &frame_hash, .{});
    const policy_hash = try digest(producer.producer.value.policy_sha256_hex);
    return .{
        .allocator = allocator,
        .decoded = decoded,
        .producer = producer,
        .sidecar = .{
            .afatt002_sha256_hex = std.fmt.bytesToHex(frame_hash, .lower),
            .footer_sha256_hex = std.fmt.bytesToHex(frame_footer, .lower),
            .checkpoint_sha256_hex = std.fmt.bytesToHex(checkpoint_hash, .lower),
            .producer_sha256_hex = std.fmt.bytesToHex(producer_hash, .lower),
            .policy_sha256_hex = std.fmt.bytesToHex(policy_hash, .lower),
            .model = producer.producer.value.model,
            .logical_tokens = decoded.checkpoint.state.logical_length,
            .checkpoint_bytes = start.inner.checkpoint_bytes,
            .encoding = producer.producer.value.encoding,
        },
    };
}

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidNativeKvFrame;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, text) catch return error.InvalidNativeKvFrame;
    return result;
}
