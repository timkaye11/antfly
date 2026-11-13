const std = @import("std");
const platform = @import("antfly_platform");
const profile = @import("context_resident_hybrid_compatibility_profile.zig");
const shape = @import("runtime/kv/attention_match_e2_flat_shape_authority.zig");
const resident = @import("runtime/kv/attention_match_e2_resident_authority.zig");
const bridge = @import("runtime/kv/attention_match_e2_native_resident_hybrid_bridge.zig");

fn digestFile(path: []const u8, maximum: u64) ![32]u8 {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidObservationInput;
    var file = try std.Io.Dir.cwd().openFile(std.testing.io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(std.testing.io);
    const stat = try file.stat(std.testing.io);
    if (stat.kind != .file or stat.size == 0 or stat.size > maximum) return error.InvalidObservationInput;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const count: usize = @intCast(@min(stat.size - offset, buffer.len));
        const read = try file.readPositionalAll(std.testing.io, buffer[0..count], offset);
        if (read == 0) return error.InvalidObservationInput;
        hash.update(buffer[0..read]);
        offset += read;
    }
    return hash.finalResult();
}

fn executableDigest(allocator: std.mem.Allocator) ![32]u8 {
    const path = try std.process.executablePathAlloc(std.testing.io, allocator);
    defer allocator.free(path);
    return digestFile(path, std.math.maxInt(u64));
}

fn ownerCensusForJson() [3]u32 {
    return .{ shape.canonical_owner_consumer_counts[0], shape.canonical_owner_consumer_counts[1], shape.canonical_owner_consumer_counts[2] };
}

test "context E2 compiled owner census serializes as numeric JSON array" {
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, ownerCensusForJson(), .{});
    defer std.testing.allocator.free(body);
    try std.testing.expectEqualStrings("[8,8,40]", body);
}

fn observe(source_freeze_sha256: [32]u8, binary_sha256: [32]u8) !struct {
    profile_semantic_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    guarded_policy_module_sha256: [32]u8,
    formal_policy_module_sha256: [32]u8,
    runtime_implementation_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
} {
    if (std.mem.allEqual(u8, &source_freeze_sha256, 0) or std.mem.allEqual(u8, &binary_sha256, 0)) return error.InvalidObservationInput;
    const validated = try shape.validate(.{ .authority_sha256 = source_freeze_sha256 });
    var owner_total: usize = 0;
    for (shape.canonical_owner_consumer_counts) |count| owner_total += count;
    if (validated.range_count != 24 or validated.candidate_count != 72 or validated.shape.batch_count != owner_total)
        return error.InvalidObservationInput;
    return .{
        .profile_semantic_sha256 = profile.expectedE2ProfileSemantic(),
        .shape_authority_sha256 = validated.digest,
        .guarded_policy_module_sha256 = resident.guardedTrialDigest(),
        .formal_policy_module_sha256 = resident.formalFitDigest(),
        .runtime_implementation_sha256 = profile.runtimeImplementationDigest(),
        .implementation_semantic_sha256 = bridge.implementation_semantic_sha256,
    };
}

test "context E2 8128 compiled authority observation derives typed digests" {
    const got = try observe(@splat(1), @splat(2));
    try std.testing.expect(!std.mem.allEqual(u8, &got.profile_semantic_sha256, 0));
    try std.testing.expect(!std.mem.eql(u8, &got.profile_semantic_sha256, &got.shape_authority_sha256));
    try std.testing.expect(!std.mem.eql(u8, &got.guarded_policy_module_sha256, &got.formal_policy_module_sha256));
    try std.testing.expectError(error.InvalidObservationInput, observe(@splat(0), @splat(2)));
    try std.testing.expectError(error.InvalidObservationInput, observe(@splat(1), @splat(0)));
}

test "context E2 8128 compiled authority observation emitter" {
    const allocator = std.testing.allocator;
    const enabled = platform.env.getenv("ANTFLY_E2_8128_COMPILED_AUTHORITY_OBSERVATION") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, enabled, "1")) return error.InvalidObservationInput;
    const freeze_path = platform.env.getenv("ANTFLY_E2_8128_SOURCE_FREEZE") orelse return error.InvalidObservationInput;
    const output_path = platform.env.getenv("ANTFLY_E2_8128_OBSERVATION_OUTPUT") orelse return error.InvalidObservationInput;
    if (!std.fs.path.isAbsolute(output_path)) return error.InvalidObservationInput;
    const freeze_sha = try digestFile(freeze_path, 16 * 1024 * 1024);
    const binary_sha = try executableDigest(allocator);
    const got = try observe(freeze_sha, binary_sha);
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.e2-8128-compiled-authority-observation.v1",
        .observed_by_compiled_code = true,
        .diagnostic_only = true,
        .qualification_credit = false,
        .serving_activation = false,
        .model_forward_steps = 0,
        .source_freeze_sha256 = std.fmt.bytesToHex(freeze_sha, .lower),
        .binary_sha256 = std.fmt.bytesToHex(binary_sha, .lower),
        .cohort = "e2_8128_v1",
        .source_rows = 8128,
        .range_count = 24,
        .candidate_count = 72,
        .owners = .{ .{ .layer = 4, .heads = .{.{ .head = 0, .retained_rows = 512 }} }, .{ .layer = 9, .heads = .{.{ .head = 0, .retained_rows = 512 }} }, .{ .layer = 14, .heads = .{.{ .head = 0, .retained_rows = 512 }} } },
        .eligible_heads = .{ .{ .donor_layer = 4, .head_index = 0 }, .{ .donor_layer = 9, .head_index = 0 }, .{ .donor_layer = 14, .head_index = 0 } },
        .consumer_owner_counts = ownerCensusForJson(),
        .profile_semantic_sha256 = std.fmt.bytesToHex(got.profile_semantic_sha256, .lower),
        .shape_authority_sha256 = std.fmt.bytesToHex(got.shape_authority_sha256, .lower),
        .guarded_policy_module_sha256 = std.fmt.bytesToHex(got.guarded_policy_module_sha256, .lower),
        .formal_policy_module_sha256 = std.fmt.bytesToHex(got.formal_policy_module_sha256, .lower),
        .runtime_implementation_sha256 = std.fmt.bytesToHex(got.runtime_implementation_sha256, .lower),
        .implementation_semantic_sha256 = std.fmt.bytesToHex(got.implementation_semantic_sha256, .lower),
        .candidate_schema = "antfly.native-kv-operator-candidate.v6",
        .execution_name = "gemma4_native_coalesced_owner_v1",
        .resident_gated_ffn_fusion = false,
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(body);
    var output = try std.Io.Dir.cwd().createFile(std.testing.io, output_path, .{ .exclusive = true, .truncate = false });
    defer output.close(std.testing.io);
    try output.writeStreamingAll(std.testing.io, body);
    try output.writeStreamingAll(std.testing.io, "\n");
    try output.sync(std.testing.io);
}
