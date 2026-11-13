// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Offline qualification constructor for checkpoint+sidecar artifacts. Both
//! outputs are published exclusively only after the original AFATT002 frame,
//! footer, Producer, checkpoint checksum and expected identity all verify.
const std = @import("std");
const verified = @import("../server/native_kv_verified_sidecar.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");

const Binding = struct {
    schema: []const u8,
    envelope_path: []const u8,
    envelope_sha256: []const u8,
    checkpoint_output_path: []const u8,
    sidecar_output_path: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    producer: struct {
        profile_sha256: []const u8,
        profile_semantic_sha256: []const u8,
        implementation_semantic_sha256: []const u8,
        binary_sha256: []const u8,
        policy_sha256: []const u8,
        policy_name: []const u8,
    },
};

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidNativeKvSidecarBinding;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidNativeKvSidecarBinding;
    return out;
}

fn publishPair(
    io: std.Io,
    checkpoint_output_path: []const u8,
    sidecar_output_path: []const u8,
    payload: []const u8,
    sidecar_json: []const u8,
    fail_sidecar_write_for_test: bool,
) !void {
    // Reserve both final names before publishing either. Every error after a
    // successful create removes only files owned by this invocation; a
    // pre-existing sidecar therefore prevents any checkpoint publication.
    var sidecar_owned = false;
    var checkpoint_owned = false;
    errdefer {
        if (checkpoint_owned) std.Io.Dir.deleteFileAbsolute(io, checkpoint_output_path) catch {};
        if (sidecar_owned) std.Io.Dir.deleteFileAbsolute(io, sidecar_output_path) catch {};
    }
    var sidecar_file = try std.Io.Dir.createFileAbsolute(io, sidecar_output_path, .{ .exclusive = true, .truncate = false });
    sidecar_owned = true;
    defer sidecar_file.close(io);
    var checkpoint_file = try std.Io.Dir.createFileAbsolute(io, checkpoint_output_path, .{ .exclusive = true, .truncate = false });
    checkpoint_owned = true;
    defer checkpoint_file.close(io);
    try checkpoint_file.writeStreamingAll(io, payload);
    try checkpoint_file.sync(io);
    if (fail_sidecar_write_for_test) return error.InjectedSidecarWriteFailure;
    try sidecar_file.writeStreamingAll(io, sidecar_json);
    try sidecar_file.writeStreamingAll(io, "\n");
    try sidecar_file.sync(io);
    checkpoint_owned = false;
    sidecar_owned = false;
}

pub fn main(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (args.len != 2 or !std.mem.eql(u8, args[0], "--binding") or !std.fs.path.isAbsolute(args[1]))
        return error.InvalidArguments;
    const binding_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], allocator, .limited(64 * 1024));
    defer allocator.free(binding_bytes);
    var parsed = try std.json.parseFromSlice(Binding, allocator, binding_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const binding = parsed.value;
    if (!std.mem.eql(u8, binding.schema, "antfly.native-kv-sidecar-constructor.v1") or
        !std.fs.path.isAbsolute(binding.envelope_path) or !std.fs.path.isAbsolute(binding.checkpoint_output_path) or
        !std.fs.path.isAbsolute(binding.sidecar_output_path) or std.mem.eql(u8, binding.checkpoint_output_path, binding.sidecar_output_path))
        return error.InvalidNativeKvSidecarBinding;
    const frame = try std.Io.Dir.cwd().readFileAlloc(io, binding.envelope_path, allocator, .limited(2 * 1024 * 1024 * 1024 + 2 * 1024 * 1024 + 128));
    defer allocator.free(frame);
    var frame_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(frame, &frame_hash, .{});
    if (!std.mem.eql(u8, &frame_hash, &try digest(binding.envelope_sha256))) return error.InvalidNativeKvSidecarBinding;
    const expected: checkpoint.Identity = .{
        .weights = try digest(binding.identity.weights),
        .tokenizer = try digest(binding.identity.tokenizer),
        .template = try digest(binding.identity.template),
        .layout = try digest(binding.identity.layout),
    };
    var result = try verified.verifyFrame(allocator, frame, expected, .{
        .profile_sha256 = try digest(binding.producer.profile_sha256),
        .profile_semantic_sha256 = try digest(binding.producer.profile_semantic_sha256),
        .implementation_semantic_sha256 = try digest(binding.producer.implementation_semantic_sha256),
        .binary_sha256 = try digest(binding.producer.binary_sha256),
        .policy_sha256 = try digest(binding.producer.policy_sha256),
        .policy_name = binding.producer.policy_name,
    }, checkpoint.serving_limits);
    defer result.deinit();
    const metadata_bytes = std.math.cast(usize, std.mem.readInt(u64, frame[8..16], .little)) orelse return error.InvalidNativeKvFrame;
    const payload_offset = 40 + metadata_bytes + @import("../server/native_kv_wire.zig").checkpoint_mime.len;
    const payload_bytes = std.math.cast(usize, result.sidecar.checkpoint_bytes) orelse return error.InvalidNativeKvFrame;
    const payload = frame[payload_offset..][0..payload_bytes];
    const sidecar_json = try std.json.Stringify.valueAlloc(allocator, result.sidecar, .{});
    defer allocator.free(sidecar_json);
    try publishPair(io, binding.checkpoint_output_path, binding.sidecar_output_path, payload, sidecar_json, false);
}

test "native KV verified sidecar publication is paired and rolls back owned outputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, "checkpoint.bin" });
    defer allocator.free(checkpoint_path);
    const sidecar_path = try std.fs.path.join(allocator, &.{ root, "sidecar.json" });
    defer allocator.free(sidecar_path);

    try tmp.dir.writeFile(io, .{ .sub_path = "sidecar.json", .data = "existing" });
    try std.testing.expectError(error.PathAlreadyExists, publishPair(io, checkpoint_path, sidecar_path, "checkpoint", "{}", false));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.openFileAbsolute(io, checkpoint_path, .{}));
    const existing = try tmp.dir.readFileAlloc(io, "sidecar.json", allocator, .limited(32));
    defer allocator.free(existing);
    try std.testing.expectEqualStrings("existing", existing);

    try tmp.dir.deleteFile(io, "sidecar.json");
    try std.testing.expectError(error.InjectedSidecarWriteFailure, publishPair(io, checkpoint_path, sidecar_path, "checkpoint", "{}", true));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.openFileAbsolute(io, checkpoint_path, .{}));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.openFileAbsolute(io, sidecar_path, .{}));
}
