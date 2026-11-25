// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const wire = @import("native_kv_wire.zig");
const fit_policy = @import("../runtime/kv/attention_match_fit_policy.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");

pub const Owned = struct {
    allocator: std.mem.Allocator,
    outer: std.json.Parsed(wire.V2Metadata),
    producer_bytes: []u8,
    producer: std.json.Parsed(wire.PreparedProducer),

    pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Owned {
        var outer = try std.json.parseFromSlice(wire.V2Metadata, allocator, bytes, .{
            .allocate = .alloc_always,
            .ignore_unknown_fields = false,
        });
        errdefer outer.deinit();
        if (!std.mem.eql(u8, outer.value.schema, "antfly.native-kv-envelope.v2")) return error.InvalidNativeKvFrame;
        const decoded_size = std.base64.standard.Decoder.calcSizeForSlice(outer.value.prepared_metadata_base64) catch
            return error.InvalidNativeKvFrame;
        if (decoded_size == 0 or decoded_size > wire.max_prepared_metadata_bytes) return error.NativeKvFrameLimit;
        const producer_bytes = try allocator.alloc(u8, decoded_size);
        errdefer allocator.free(producer_bytes);
        std.base64.standard.Decoder.decode(producer_bytes, outer.value.prepared_metadata_base64) catch
            return error.InvalidNativeKvFrame;
        var producer = try std.json.parseFromSlice(wire.PreparedProducer, allocator, producer_bytes, .{
            .allocate = .alloc_always,
            .ignore_unknown_fields = false,
        });
        errdefer producer.deinit();
        try validateProducer(producer.value);
        if (outer.value.request_json) |request| {
            var parsed = try std.json.parseFromSlice(std.json.Value, allocator, request, .{ .allocate = .alloc_always });
            defer parsed.deinit();
            if (parsed.value != .object) return error.InvalidNativeCheckpointRequest;
        }
        return .{ .allocator = allocator, .outer = outer, .producer_bytes = producer_bytes, .producer = producer };
    }

    pub fn deinit(self: *Owned) void {
        self.producer.deinit();
        self.allocator.free(self.producer_bytes);
        self.outer.deinit();
        self.* = undefined;
    }
};

fn hex64(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |byte| if (!std.ascii.isDigit(byte) and !(byte >= 'a' and byte <= 'f')) return false;
    return true;
}

pub fn validateProducer(value: wire.PreparedProducer) !void {
    if (!std.mem.eql(u8, value.schema, "antfly.native-kv-prepared-producer.v1") or
        value.model.len == 0 or value.policy_name.len == 0 or value.admission.len == 0 or
        value.encoding.len == 0 or value.logical_revision == 0 or value.logical_tokens == 0 or
        value.checkpoint_bytes < 192 or
        !hex64(value.model_artifact_sha256_hex) or !hex64(value.model_tokenizer_sha256_hex) or
        !hex64(value.model_template_sha256_hex) or !hex64(value.model_layout_sha256_hex) or
        !hex64(value.profile_sha256_hex) or !hex64(value.profile_semantic_sha256_hex) or
        !hex64(value.source_sha256_hex) or !hex64(value.binary_sha256_hex) or
        !hex64(value.policy_sha256_hex)) return error.InvalidNativeKvFrame;
    if (value.quality_evidence_core_sha256_hex) |quality_digest|
        if (!hex64(quality_digest) or std.mem.allEqual(u8, quality_digest, '0')) return error.InvalidNativeKvFrame;
    const selected = try producerPolicy(value);
    const encoding = checkpoint.Encoding.parse(value.encoding) catch return error.InvalidNativeKvFrame;
    selected.validate(encoding, true) catch return error.InvalidNativeKvFrame;
    const declared = try digest(value.policy_sha256_hex);
    const derived = selected.digest();
    if (!std.mem.eql(u8, &declared, &derived)) return error.NativeKvProducerIdentityMismatch;
}

fn producerPolicy(value: wire.PreparedProducer) !fit_policy.Policy {
    const selected = if (std.mem.eql(u8, value.policy_name, "legacy_fixed"))
        fit_policy.Policy.legacy_fixed
    else if (std.mem.eql(u8, value.policy_name, "cumulative256_exact_cycle_v1"))
        fit_policy.Policy.cumulative256_exact_cycle_v1
    else
        return error.InvalidNativeKvFrame;
    if (!std.mem.eql(u8, value.admission, @tagName(selected.admission)) or value.chunk_rows != selected.chunk_rows or
        value.reject_exact_active_fixed_point != selected.reject_exact_active_fixed_point)
        return error.NativeKvProducerIdentityMismatch;
    return selected;
}

fn digest(text: []const u8) ![32]u8 {
    if (!hex64(text)) return error.InvalidNativeKvFrame;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, text) catch return error.InvalidNativeKvFrame;
    return result;
}

pub fn validateCheckpointIdentity(value: wire.PreparedProducer, actual: @import("../runtime/kv/attention_match_checkpoint.zig").Identity) !void {
    const artifact = try digest(value.model_artifact_sha256_hex);
    const tokenizer = try digest(value.model_tokenizer_sha256_hex);
    const template = try digest(value.model_template_sha256_hex);
    const layout = try digest(value.model_layout_sha256_hex);
    if (!std.mem.eql(u8, &artifact, &actual.weights) or !std.mem.eql(u8, &tokenizer, &actual.tokenizer) or
        !std.mem.eql(u8, &template, &actual.template) or !std.mem.eql(u8, &layout, &actual.layout))
        return error.NativeKvProducerIdentityMismatch;
}

pub fn validateAuthority(value: wire.PreparedProducer, authority: @import("native_kv_producer_authority.zig").TrustedProducer) !void {
    const selected = try producerPolicy(value);
    const profile = try digest(value.profile_sha256_hex);
    const semantic = try digest(value.profile_semantic_sha256_hex);
    const implementation = try digest(value.source_sha256_hex);
    const binary = try digest(value.binary_sha256_hex);
    const policy = try digest(value.policy_sha256_hex);
    const authority_policy = authority.policy.digest();
    if (!std.mem.eql(u8, &profile, &authority.identity.profile_sha256) or
        !std.mem.eql(u8, &semantic, &authority.identity.profile_semantic_sha256) or
        !std.mem.eql(u8, &implementation, &authority.implementation_semantic_sha256) or
        !std.meta.eql(selected, authority.policy) or !std.mem.eql(u8, &binary, &authority.binary_sha256) or
        !std.mem.eql(u8, &policy, &authority_policy)) return error.NativeKvProducerIdentityMismatch;
}

test "AFATT002 producer metadata is bounded opaque and strict" {
    const digest_hex = "1111111111111111111111111111111111111111111111111111111111111111";
    const policy_digest_hex = std.fmt.bytesToHex(fit_policy.Policy.legacy_fixed.digest(), .lower);
    const producer = try std.json.Stringify.valueAlloc(std.testing.allocator, wire.PreparedProducer{
        .model = "gemma",
        .model_artifact_sha256_hex = digest_hex,
        .model_tokenizer_sha256_hex = digest_hex,
        .model_template_sha256_hex = digest_hex,
        .model_layout_sha256_hex = digest_hex,
        .profile_sha256_hex = digest_hex,
        .profile_semantic_sha256_hex = digest_hex,
        .source_sha256_hex = digest_hex,
        .binary_sha256_hex = digest_hex,
        .policy_name = "legacy_fixed",
        .policy_sha256_hex = &policy_digest_hex,
        .admission = "fixed",
        .chunk_rows = 1024,
        .reject_exact_active_fixed_point = false,
        .encoding = "f16",
        .logical_revision = 1,
        .logical_tokens = 2,
        .checkpoint_bytes = 192,
        .outcome = .compacted,
    }, .{});
    defer std.testing.allocator.free(producer);
    const encoded_size = std.base64.standard.Encoder.calcSize(producer.len);
    const encoded = try std.testing.allocator.alloc(u8, encoded_size);
    defer std.testing.allocator.free(encoded);
    _ = std.base64.standard.Encoder.encode(encoded, producer);
    const outer = try std.json.Stringify.valueAlloc(std.testing.allocator, wire.V2Metadata{ .prepared_metadata_base64 = encoded }, .{});
    defer std.testing.allocator.free(outer);
    var parsed = try Owned.parse(std.testing.allocator, outer);
    defer parsed.deinit();
    try std.testing.expectEqual(wire.PreparedOutcome.compacted, parsed.producer.value.outcome);
}

test "AFATT002 producer derives named policy tuple and digest" {
    const candidate = fit_policy.Policy.cumulative256_exact_cycle_v1;
    const candidate_digest = std.fmt.bytesToHex(candidate.digest(), .lower);
    var producer: wire.PreparedProducer = .{
        .model = "gemma",
        .model_artifact_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .model_tokenizer_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .model_template_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .model_layout_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .profile_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .profile_semantic_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .source_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .binary_sha256_hex = "1111111111111111111111111111111111111111111111111111111111111111",
        .policy_name = "cumulative256_exact_cycle_v1",
        .policy_sha256_hex = &candidate_digest,
        .admission = @tagName(candidate.admission),
        .chunk_rows = candidate.chunk_rows,
        .reject_exact_active_fixed_point = candidate.reject_exact_active_fixed_point,
        .encoding = "f16",
        .logical_revision = 1,
        .logical_tokens = 2,
        .checkpoint_bytes = 192,
        .outcome = .compacted,
    };
    try validateProducer(producer);
    producer.chunk_rows = 1024;
    try std.testing.expectError(error.NativeKvProducerIdentityMismatch, validateProducer(producer));
    producer.chunk_rows = candidate.chunk_rows;
    producer.policy_name = "legacy_fixed";
    try std.testing.expectError(error.NativeKvProducerIdentityMismatch, validateProducer(producer));
    producer.policy_name = "cumulative256_exact_cycle_v1";
    producer.policy_sha256_hex = "2222222222222222222222222222222222222222222222222222222222222222";
    try std.testing.expectError(error.NativeKvProducerIdentityMismatch, validateProducer(producer));
}
