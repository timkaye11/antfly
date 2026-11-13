// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");

pub const VerifiedEnrollment = struct {
    raw_verification_sha256: [32]u8,
    target_verification_sha256: [32]u8,
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    full_context_release_sha256: [32]u8,
    source_archive_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    runtime_policy_sha256: [32]u8,
    kernel_caps_sha256: [32]u8,
    device_registry_id: u64,
    physical_memory_bytes: u64,
    device_name_sha256: [32]u8,
    compiler_identity_sha256: [32]u8,
    backend_environment_sha256: [32]u8,
};
pub const Expected = struct {
    artifact_sha256: [32]u8,
    raw_verification_sha256: [32]u8,
    target_verification_sha256: [32]u8,
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    full_context_release_sha256: [32]u8,
    source_archive_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    runtime_policy_sha256: [32]u8,
    kernel_caps_sha256: [32]u8,
    device_registry_id: u64,
    physical_memory_bytes: u64,
    device_name_sha256: [32]u8,
    compiler_identity_sha256: [32]u8,
    backend_environment_sha256: [32]u8,
};
const Wire = struct {
    schema: []const u8,
    raw_verification_sha256: []const u8,
    target_verification_sha256: []const u8,
    profile_sha256: []const u8,
    profile_semantic_sha256: []const u8,
    binary_sha256: []const u8,
    source_freeze_sha256: []const u8,
    full_context_release_sha256: []const u8,
    source_archive_sha256: []const u8,
    implementation_semantic_sha256: []const u8,
    runtime_policy_sha256: []const u8,
    kernel_caps_sha256: []const u8,
    device_registry_id: u64,
    physical_memory_bytes: u64,
    device_name_sha256: []const u8,
    compiler_identity_sha256: []const u8,
    backend_environment_sha256: []const u8,
    mode: []const u8,
    qualification_credit: bool,
    production_qualified: bool,
    enrollment_applied: bool,
};

fn observedHardwareMatches(value: VerifiedEnrollment, expected: Expected) bool {
    return value.device_registry_id == expected.device_registry_id and
        value.physical_memory_bytes == expected.physical_memory_bytes and
        std.mem.eql(u8, &value.device_name_sha256, &expected.device_name_sha256) and
        std.mem.eql(u8, &value.compiler_identity_sha256, &expected.compiler_identity_sha256) and
        std.mem.eql(u8, &value.backend_environment_sha256, &expected.backend_environment_sha256);
}
fn digest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidPrivateDenseEnrollment;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidPrivateDenseEnrollment;
    if (std.mem.allEqual(u8, &out, 0)) return error.InvalidPrivateDenseEnrollment;
    return out;
}
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !VerifiedEnrollment {
    var parsed = std.json.parseFromSlice(Wire, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidPrivateDenseEnrollment;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.qualified-private-native-checkpoint-enrollment.v4") or
        !std.mem.eql(u8, value.mode, "qualified_private_dynamic_budget") or
        !value.qualification_credit or !value.production_qualified or !value.enrollment_applied or
        value.device_registry_id == 0 or value.physical_memory_bytes == 0)
        return error.InvalidPrivateDenseEnrollment;
    return .{
        .raw_verification_sha256 = try digest(value.raw_verification_sha256),
        .target_verification_sha256 = try digest(value.target_verification_sha256),
        .profile_sha256 = try digest(value.profile_sha256),
        .profile_semantic_sha256 = try digest(value.profile_semantic_sha256),
        .binary_sha256 = try digest(value.binary_sha256),
        .source_freeze_sha256 = try digest(value.source_freeze_sha256),
        .full_context_release_sha256 = try digest(value.full_context_release_sha256),
        .source_archive_sha256 = try digest(value.source_archive_sha256),
        .implementation_semantic_sha256 = try digest(value.implementation_semantic_sha256),
        .runtime_policy_sha256 = try digest(value.runtime_policy_sha256),
        .kernel_caps_sha256 = try digest(value.kernel_caps_sha256),
        .device_registry_id = value.device_registry_id,
        .physical_memory_bytes = value.physical_memory_bytes,
        .device_name_sha256 = try digest(value.device_name_sha256),
        .compiler_identity_sha256 = try digest(value.compiler_identity_sha256),
        .backend_environment_sha256 = try digest(value.backend_environment_sha256),
    };
}

pub fn parseVerified(allocator: std.mem.Allocator, bytes: []const u8, expected: Expected) !VerifiedEnrollment {
    var artifact: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &artifact, .{});
    if (!std.mem.eql(u8, &artifact, &expected.artifact_sha256))
        return error.InvalidPrivateDenseEnrollment;
    const value = try parse(allocator, bytes);
    inline for (.{
        .{ value.raw_verification_sha256, expected.raw_verification_sha256 },
        .{ value.target_verification_sha256, expected.target_verification_sha256 },
        .{ value.profile_sha256, expected.profile_sha256 },
        .{ value.profile_semantic_sha256, expected.profile_semantic_sha256 },
        .{ value.binary_sha256, expected.binary_sha256 },
        .{ value.source_freeze_sha256, expected.source_freeze_sha256 },
        .{ value.full_context_release_sha256, expected.full_context_release_sha256 },
        .{ value.source_archive_sha256, expected.source_archive_sha256 },
        .{ value.implementation_semantic_sha256, expected.implementation_semantic_sha256 },
        .{ value.runtime_policy_sha256, expected.runtime_policy_sha256 },
        .{ value.kernel_caps_sha256, expected.kernel_caps_sha256 },
        .{ value.device_name_sha256, expected.device_name_sha256 },
        .{ value.compiler_identity_sha256, expected.compiler_identity_sha256 },
        .{ value.backend_environment_sha256, expected.backend_environment_sha256 },
    }) |pair| if (!std.mem.eql(u8, &pair[0], &pair[1]))
        return error.InvalidPrivateDenseEnrollment;
    if (!observedHardwareMatches(value, expected))
        return error.InvalidPrivateDenseEnrollment;
    return value;
}

test "qualified private observed hardware rejects every identity mutation" {
    var value: VerifiedEnrollment = undefined;
    value.device_registry_id = 1;
    value.physical_memory_bytes = 24 * 1024 * 1024 * 1024;
    value.device_name_sha256 = @splat(1);
    value.compiler_identity_sha256 = @splat(2);
    value.backend_environment_sha256 = @splat(3);
    var expected: Expected = undefined;
    expected.device_registry_id = value.device_registry_id;
    expected.physical_memory_bytes = value.physical_memory_bytes;
    expected.device_name_sha256 = value.device_name_sha256;
    expected.compiler_identity_sha256 = value.compiler_identity_sha256;
    expected.backend_environment_sha256 = value.backend_environment_sha256;
    try std.testing.expect(observedHardwareMatches(value, expected));
    value.device_registry_id += 1;
    try std.testing.expect(!observedHardwareMatches(value, expected));
    value.device_registry_id -= 1;
    value.physical_memory_bytes -= 1;
    try std.testing.expect(!observedHardwareMatches(value, expected));
    value.physical_memory_bytes += 1;
    value.device_name_sha256[0] ^= 1;
    try std.testing.expect(!observedHardwareMatches(value, expected));
    value.device_name_sha256[0] ^= 1;
    value.compiler_identity_sha256[0] ^= 1;
    try std.testing.expect(!observedHardwareMatches(value, expected));
    value.compiler_identity_sha256[0] ^= 1;
    value.backend_environment_sha256[0] ^= 1;
    try std.testing.expect(!observedHardwareMatches(value, expected));
}

test "context qualified private enrollment rejects pre-enrollment evidence" {
    try std.testing.expectError(error.InvalidPrivateDenseEnrollment, parse(std.testing.allocator, "{\"schema\":\"antfly.qualified-private-native-checkpoint-enrollment.v2\",\"qualification_credit\":false,\"production_qualified\":false,\"enrollment_applied\":false}"));
}

test "context qualified private enrollment requires independently configured artifact digest" {
    const bytes = "{}";
    const expected: Expected = .{
        .artifact_sha256 = @splat(1),
        .raw_verification_sha256 = @splat(14),
        .target_verification_sha256 = @splat(15),
        .profile_sha256 = @splat(2),
        .profile_semantic_sha256 = @splat(3),
        .binary_sha256 = @splat(4),
        .source_freeze_sha256 = @splat(5),
        .full_context_release_sha256 = @splat(9),
        .source_archive_sha256 = @splat(10),
        .implementation_semantic_sha256 = @splat(6),
        .runtime_policy_sha256 = @splat(7),
        .kernel_caps_sha256 = @splat(8),
        .device_registry_id = 1,
        .physical_memory_bytes = 1,
        .device_name_sha256 = @splat(11),
        .compiler_identity_sha256 = @splat(12),
        .backend_environment_sha256 = @splat(13),
    };
    try std.testing.expectError(error.InvalidPrivateDenseEnrollment, parseVerified(std.testing.allocator, bytes, expected));
}
