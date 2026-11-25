// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Process-local AFATT002 authority. HTTP inputs cannot construct it.
const std = @import("std");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
const fit_policy = @import("../runtime/kv/attention_match_fit_policy.zig");
const compaction_identity = @import("compaction_identity");
const release = @import("../runtime/kv/attention_match_release.zig");

pub const Identity = struct {
    checkpoint: checkpoint.Identity,
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
};

pub const Authority = struct {
    binary_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    identity: Identity,
    policy: fit_policy.Policy,
    execution: release.Execution,

    pub fn trusted(self: Authority) TrustedProducer {
        return .{ .binary_sha256 = self.binary_sha256, .implementation_semantic_sha256 = self.implementation_semantic_sha256, .identity = self.identity, .policy = self.policy };
    }

    pub fn initVerified(allocator: std.mem.Allocator, io: std.Io, expected_binary_sha256: [32]u8, expected_implementation_semantic_sha256: [32]u8, identity: Identity, policy: fit_policy.Policy, execution: release.Execution) !Authority {
        if (!std.mem.eql(u8, &expected_implementation_semantic_sha256, &compaction_identity.digest))
            return error.NativeKvProducerImplementationMismatch;
        const path = try std.process.executablePathAlloc(io, allocator);
        defer allocator.free(path);
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
        defer file.close(io);
        const stat = try file.stat(io);
        if (stat.kind != .file) return error.InvalidNativeKvProducerBinary;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var buffer: [64 * 1024]u8 = undefined;
        var offset: u64 = 0;
        while (offset < stat.size) {
            const requested: usize = @intCast(@min(stat.size - offset, buffer.len));
            const read = try file.readPositionalAll(io, buffer[0..requested], offset);
            if (read == 0) return error.InvalidNativeKvProducerBinary;
            hash.update(buffer[0..read]);
            offset += read;
        }
        var actual: [32]u8 = undefined;
        hash.final(&actual);
        if (!std.mem.eql(u8, &actual, &expected_binary_sha256)) return error.NativeKvProducerBinaryMismatch;
        try policy.validate(.f16, true);
        try execution.validate();
        return .{ .binary_sha256 = actual, .implementation_semantic_sha256 = compaction_identity.digest, .identity = identity, .policy = policy, .execution = execution };
    }

    pub fn validate(self: Authority, actual: checkpoint.Identity, policy: fit_policy.Policy, profile_sha256: [32]u8, profile_semantic_sha256: [32]u8) !void {
        if (!checkpoint.Identity.eql(self.identity.checkpoint, actual) or !std.meta.eql(self.policy, policy) or
            !std.mem.eql(u8, &self.identity.profile_sha256, &profile_sha256) or
            !std.mem.eql(u8, &self.identity.profile_semantic_sha256, &profile_semantic_sha256))
            return error.NativeKvProducerIdentityMismatch;
    }
};

pub const TrustedProducer = struct {
    binary_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    identity: Identity,
    policy: fit_policy.Policy,
};

test "producer authority rejects an implementation digest not built into the executable" {
    var wrong = compaction_identity.digest;
    wrong[0] ^= 1;
    try std.testing.expectError(error.NativeKvProducerImplementationMismatch, Authority.initVerified(
        std.testing.allocator,
        std.testing.io,
        @splat(0),
        wrong,
        .{ .checkpoint = undefined, .profile_sha256 = @splat(0), .profile_semantic_sha256 = @splat(0) },
        .legacy_fixed,
        .{},
    ));
}
