// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Startup-only qualified-private projection. No HTTP, environment, or generic
//! registry input can construct a Grant.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const server = @import("server/server.zig");
const authority_mod = @import("server/native_kv_producer_authority.zig");
const enrollment_mod = @import("context_private_dense_enrollment.zig");
const private_policy = @import("private_native_checkpoint_policy.zig");
const candidate_loader = @import("context_serving_entrypoint.zig");
const bounded_profile_loader = @import("context_qualified_resident_bounded_profile.zig");
const release_build = @import("context_release_build_binding.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");
const native_release = @import("runtime/kv/attention_match_release.zig");
const generation = @import("pipelines/generation.zig");
const metal_runtime = if (build_options.enable_metal and builtin.os.tag == .macos)
    @import("backends/metal_runtime.zig")
else
    struct {
        fn nativeCheckpointDestination() !native_release.Destination {
            return error.UnsupportedQualifiedPrivateHardware;
        }
    };

pub const Grant = @import("qualified_private_grant.zig").Grant;

const Wire = struct {
    schema: []const u8,
    mode: []const u8,
    source_full_cli_binding_path: []const u8,
    source_full_cli_binding_sha256: []const u8,
    release_receipt_path: []const u8,
    profile_path: []const u8,
    enrollment_path: []const u8,
    enrollment_sha256: []const u8,
    raw_verification_sha256: []const u8,
    target_verification_sha256: []const u8,
    raw_verification_path: []const u8,
    target_verification_path: []const u8,
    profile_sha256: []const u8,
    profile_semantic_sha256: []const u8,
    binary_sha256: []const u8,
    implementation_semantic_sha256: []const u8,
    runtime_policy_sha256: []const u8,
    producer_fit_policy_sha256: []const u8,
    kernel_caps_sha256: []const u8,
    qualification_arm: generation.NativeCheckpointQualificationArm,
    fit_strategy: generation.NativeCheckpointPrepare.QualityFitStrategy,
    training_policy: generation.NativeCheckpointTrainingPolicy,
    compiled_prefill_max_copy_bytes: usize,
    compiled_prefill_max_host_staging_bytes: usize,
    compiled_prefill_codec_scope: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.CodecScope,
    compiled_prefill_max_logical_tokens: usize,
    compiled_prefill_layout_policy: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.LayoutPolicy,
    prepare_host_owner_capability: @import("runtime/kv/compaction_profile.zig").PrepareHostOwnerCapability = .none,
    prepare_host_owner_physical_bytes: u64 = 0,
    resident_configuration_sha256: ?[]const u8 = null,
    resident_source_freeze_sha256: ?[]const u8 = null,
    resident_shape_authority_sha256: ?[]const u8 = null,
    resident_policy_module_sha256: ?[]const u8 = null,
    resident_training_policy_sha256: ?[]const u8 = null,
    base_compatibility_profile_path: ?[]const u8 = null,
    base_compatibility_profile_sha256: ?[]const u8 = null,
    base_compatibility_profile_semantic_sha256: ?[]const u8 = null,
    resident_hybrid_configuration_path: ?[]const u8 = null,
    bounded_encoding: ?@import("runtime/kv/attention_match_checkpoint.zig").Encoding = null,
    bounded_logical_tokens: ?u32 = null,
    bounded_max_output_tokens: ?u32 = null,
    bounded_fit_workers: ?u32 = null,
    bounded_resume_allowed: ?bool = null,
};

const CandidateV4 = struct {
    schema: []const u8,
    mode: []const u8,
    release_receipt_path: []const u8,
    producer_release_receipt_path: ?[]const u8 = null,
    profile_path: []const u8,
    profile_sha256: []const u8,
    profile_semantic_sha256: []const u8,
    implementation_semantic_sha256: []const u8,
    policy_name: []const u8,
    execution_name: []const u8,
    qualification_arm: generation.NativeCheckpointQualificationArm,
    fit_strategy: generation.NativeCheckpointPrepare.QualityFitStrategy,
    training_policy: generation.NativeCheckpointTrainingPolicy,
    compiled_prefill_max_copy_bytes: usize,
    compiled_prefill_max_host_staging_bytes: usize,
    compiled_prefill_codec_scope: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.CodecScope,
    compiled_prefill_max_logical_tokens: usize,
    compiled_prefill_layout_policy: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.LayoutPolicy,
    prepared_quality_closure_path: ?[]const u8 = null,
    prepared_quality_source_path: ?[]const u8 = null,
    diagnostic_only: bool = false,
    qualification_credit: bool = false,
    registry_enrollment: bool = false,
    base_compatibility_profile_path: ?[]const u8 = null,
    base_compatibility_profile_sha256: ?[]const u8 = null,
    base_compatibility_profile_semantic_sha256: ?[]const u8 = null,
    resident_hybrid_configuration_path: ?[]const u8 = null,
    resident_hybrid_configuration_sha256: ?[]const u8 = null,
    resident_hybrid_shape_authority_sha256: ?[]const u8 = null,
    resident_hybrid_policy_module_sha256: ?[]const u8 = null,
    bounded_encoding: ?@import("runtime/kv/attention_match_checkpoint.zig").Encoding = null,
    bounded_logical_tokens: ?u32 = null,
    bounded_max_output_tokens: ?u32 = null,
    bounded_fit_workers: ?u32 = null,
    bounded_resume_allowed: ?bool = null,
};

const Pin = struct { path: []const u8, sha256: []const u8 };
const ReceiptHead = struct {
    schema: []const u8,
    source_freeze: Pin,
    full_context_release_proof: Pin,
    full_source_archive: Pin,
    binary: struct { path: []const u8, sha256: []const u8, bytes: u64 },
};

const host_sysctl = if (builtin.os.tag == .macos) struct {
    extern fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: *usize, newp: ?*const anyopaque, newlen: usize) c_int;
} else struct {
    fn sysctlbyname(_: [*:0]const u8, _: ?*anyopaque, _: *usize, _: ?*const anyopaque, _: usize) c_int {
        return -1;
    }
};

fn validateQualifiedPrivateHost() !void {
    if (comptime builtin.os.tag != .macos or builtin.cpu.arch != .aarch64 or !build_options.enable_metal)
        return error.UnsupportedQualifiedPrivateHardware;
    var memory_bytes: u64 = 0;
    var memory_len: usize = @sizeOf(u64);
    if (host_sysctl.sysctlbyname("hw.memsize", @ptrCast(&memory_bytes), &memory_len, null, 0) != 0 or memory_len != @sizeOf(u64) or memory_bytes != 24 * 1024 * 1024 * 1024)
        return error.UnsupportedQualifiedPrivateHardware;
    var brand: [128]u8 = @splat(0);
    var brand_len: usize = brand.len;
    if (host_sysctl.sysctlbyname("machdep.cpu.brand_string", @ptrCast(&brand), &brand_len, null, 0) != 0 or brand_len == 0)
        return error.UnsupportedQualifiedPrivateHardware;
    const text = brand[0 .. brand_len - @intFromBool(brand[brand_len - 1] == 0)];
    if (std.mem.indexOf(u8, text, "Apple M4 Pro") == null) return error.UnsupportedQualifiedPrivateHardware;
}

fn kernelCapsDigest(handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff, destination: native_release.Destination, backend_environment_sha256: [32]u8) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    if (handoff.codec_scope == .f16_only)
        hash.update("antfly.qualified-resident-kernel-caps.v2\nchip=Apple M4 Pro\nmemory_bytes=25769803776\ncodec=f16\nlogical_tokens=8128\nprepare_host_owner=bounded_page_e4_resident_1152m_v1\nprepare_host_physical_bytes=1207959552\nfit_workers=1\n")
    else
        hash.update("antfly.qualified-private-kernel-caps.v2\nchip=Apple M4 Pro\nmemory_bytes=25769803776\ncodec_policies=bounded_f16_codec_physical_v1,bounded_fp8_codec_physical_v1,bounded_rotated4_codec_physical_v1\ncodec_policy_manifest_sha256=fa75a93127913dfc802184dd1f7f8ec8e23f42b12eeac9fa9549fad051b6eedd\ngeometry=private_hybrid_bounded_accelerated_batch64_workspace1gib\nmax_tokens=32768\nlayout=full_history_exact\nmax_copy_bytes=4294967296\nmax_host_staging_bytes=268435456\ntsqr_boundary=8192\nsealed_batch_rows=64\nmax_selected_rows=128\nmax_queries=65536\nprivate_workspace_bytes=1073741824\ncpu_fp64_heldout=unchanged\npublish_after_fit=1\nprimary_budget_percent=0\nbudget_fallback_heads=0\n");
    const legacy = handoff.codec_scope == .f16_fp8_rotated4 and handoff.max_logical_tokens == 32768;
    const resident = handoff.codec_scope == .f16_only and handoff.max_logical_tokens == 8192;
    if (handoff.max_copy_bytes != 4 * 1024 * 1024 * 1024 or handoff.max_host_staging_bytes != 256 * 1024 * 1024 or
        (!legacy and !resident) or handoff.layout_policy != .full_history_exact) return error.InvalidQualifiedPrivateBinding;
    try destination.validate();
    const destination_sha256 = destination.fingerprint();
    hash.update(&destination_sha256);
    hash.update(&backend_environment_sha256);
    return hash.finalResult();
}

fn actualObserved() !native_release.Observed {
    if (comptime builtin.is_test) if (test_observed) |observed| return observed;
    if (comptime builtin.os.tag != .macos or builtin.cpu.arch != .aarch64 or !build_options.enable_metal)
        return error.UnsupportedQualifiedPrivateHardware;
    try validateQualifiedPrivateHost();
    return .{
        .destination = try metal_runtime.nativeCheckpointDestination(),
        .backend_environment_sha256 = try native_release.backendEnvironmentFingerprint(),
    };
}

var test_observed: ?native_release.Observed = null;

fn enrollmentJoinDigest(expected: enrollment_mod.Expected) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.context.test-only-enrollment-join.v1\n");
    inline for (.{
        expected.raw_verification_sha256,
        expected.target_verification_sha256,
        expected.profile_sha256,
        expected.profile_semantic_sha256,
        expected.binary_sha256,
        expected.source_freeze_sha256,
        expected.full_context_release_sha256,
        expected.source_archive_sha256,
        expected.implementation_semantic_sha256,
        expected.runtime_policy_sha256,
        expected.kernel_caps_sha256,
        expected.device_name_sha256,
        expected.compiler_identity_sha256,
        expected.backend_environment_sha256,
    }) |value| hash.update(&value);
    var integers: [16]u8 = undefined;
    std.mem.writeInt(u64, integers[0..8], expected.device_registry_id, .little);
    std.mem.writeInt(u64, integers[8..16], expected.physical_memory_bytes, .little);
    hash.update(&integers);
    return hash.finalResult();
}

fn parseEnrollmentForInstall(allocator: std.mem.Allocator, bytes: []const u8, expected: enrollment_mod.Expected) !enrollment_mod.VerifiedEnrollment {
    if (comptime builtin.is_test) {
        const TestEnrollment = struct {
            schema: []const u8,
            test_fixture_authority: []const u8,
            qualification_credit: bool,
            production_qualified: bool,
            enrollment_applied: bool,
            expected_join_sha256: []const u8,
        };
        var parsed = std.json.parseFromSlice(TestEnrollment, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch null;
        if (parsed) |*fixture| {
            defer fixture.deinit();
            var artifact: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(bytes, &artifact, .{});
            const joined = enrollmentJoinDigest(expected);
            if (std.mem.eql(u8, fixture.value.schema, "antfly.context.test-only-qualified-resident-enrollment.v1") and
                std.mem.eql(u8, fixture.value.test_fixture_authority, "antfly.context.test-only-local-authority.v1") and
                !fixture.value.qualification_credit and !fixture.value.production_qualified and !fixture.value.enrollment_applied and
                std.mem.eql(u8, &artifact, &expected.artifact_sha256) and
                std.mem.eql(u8, &joined, &try digest(fixture.value.expected_join_sha256)))
                return .{
                    .raw_verification_sha256 = expected.raw_verification_sha256,
                    .target_verification_sha256 = expected.target_verification_sha256,
                    .profile_sha256 = expected.profile_sha256,
                    .profile_semantic_sha256 = expected.profile_semantic_sha256,
                    .binary_sha256 = expected.binary_sha256,
                    .source_freeze_sha256 = expected.source_freeze_sha256,
                    .full_context_release_sha256 = expected.full_context_release_sha256,
                    .source_archive_sha256 = expected.source_archive_sha256,
                    .implementation_semantic_sha256 = expected.implementation_semantic_sha256,
                    .runtime_policy_sha256 = expected.runtime_policy_sha256,
                    .kernel_caps_sha256 = expected.kernel_caps_sha256,
                    .device_registry_id = expected.device_registry_id,
                    .physical_memory_bytes = expected.physical_memory_bytes,
                    .device_name_sha256 = expected.device_name_sha256,
                    .compiler_identity_sha256 = expected.compiler_identity_sha256,
                    .backend_environment_sha256 = expected.backend_environment_sha256,
                };
        }
    }
    return enrollment_mod.parseVerified(allocator, bytes, expected);
}

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidQualifiedPrivateBinding;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidQualifiedPrivateBinding;
    if (std.mem.allEqual(u8, &out, 0)) return error.InvalidQualifiedPrivateBinding;
    return out;
}

fn readSealed(allocator: std.mem.Allocator, io: std.Io, path: []const u8, maximum: usize) ![]u8 {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualifiedPrivateBinding;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    const size = std.math.cast(usize, stat.size) orelse return error.InvalidQualifiedPrivateBinding;
    if (stat.kind != .file or size == 0 or size > maximum) return error.InvalidQualifiedPrivateBinding;
    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);
    if (try file.readPositionalAll(io, bytes, 0) != size) return error.InvalidQualifiedPrivateBinding;
    return bytes;
}

fn verify(bytes: []const u8, expected: [32]u8) !void {
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidQualifiedPrivateBinding;
}

fn verifySealedFile(io: std.Io, path: []const u8, expected: [32]u8) !void {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualifiedPrivateBinding;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file or stat.size == 0) return error.InvalidQualifiedPrivateBinding;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const requested: usize = @intCast(@min(stat.size - offset, buffer.len));
        const count = try file.readPositionalAll(io, buffer[0..requested], offset);
        if (count == 0) return error.InvalidQualifiedPrivateBinding;
        hash.update(buffer[0..count]);
        offset += count;
    }
    const actual = hash.finalResult();
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidQualifiedPrivateBinding;
}

fn candidateV4Matches(w: Wire, c: CandidateV4, resident: bool) bool {
    const resident_v9 = std.mem.eql(u8, c.schema, "antfly.native-kv-operator-candidate.v9") and
        w.training_policy == .private_train_challenge_hybrid_resident_v1 and
        c.diagnostic_only and !c.qualification_credit and !c.registry_enrollment and
        c.base_compatibility_profile_path != null and w.base_compatibility_profile_path != null and
        std.mem.eql(u8, c.base_compatibility_profile_path.?, w.base_compatibility_profile_path.?) and
        c.base_compatibility_profile_sha256 != null and std.mem.eql(u8, c.base_compatibility_profile_sha256.?, w.base_compatibility_profile_sha256.?) and
        c.base_compatibility_profile_semantic_sha256 != null and std.mem.eql(u8, c.base_compatibility_profile_semantic_sha256.?, w.base_compatibility_profile_semantic_sha256.?) and
        c.resident_hybrid_configuration_path != null and w.resident_hybrid_configuration_path != null and
        std.mem.eql(u8, c.resident_hybrid_configuration_path.?, w.resident_hybrid_configuration_path.?) and
        c.resident_hybrid_configuration_sha256 != null and std.mem.eql(u8, c.resident_hybrid_configuration_sha256.?, w.resident_configuration_sha256.?) and
        c.resident_hybrid_shape_authority_sha256 != null and std.mem.eql(u8, c.resident_hybrid_shape_authority_sha256.?, w.resident_shape_authority_sha256.?) and
        c.resident_hybrid_policy_module_sha256 != null and std.mem.eql(u8, c.resident_hybrid_policy_module_sha256.?, w.resident_policy_module_sha256.?) and
        c.bounded_encoding == w.bounded_encoding and c.bounded_logical_tokens == w.bounded_logical_tokens and
        c.bounded_max_output_tokens == w.bounded_max_output_tokens and c.bounded_fit_workers == w.bounded_fit_workers and
        c.bounded_resume_allowed == w.bounded_resume_allowed;
    const exact_kind = if (resident) resident_v9 else std.mem.eql(u8, c.schema, "antfly.native-kv-operator-candidate.v4");
    return exact_kind and std.mem.eql(u8, c.mode, "staged_candidate") and
        c.producer_release_receipt_path == null and c.prepared_quality_closure_path == null and c.prepared_quality_source_path == null and
        std.mem.eql(u8, c.release_receipt_path, w.release_receipt_path) and std.mem.eql(u8, c.profile_path, w.profile_path) and
        std.mem.eql(u8, c.profile_sha256, w.profile_sha256) and std.mem.eql(u8, c.profile_semantic_sha256, w.profile_semantic_sha256) and
        std.mem.eql(u8, c.implementation_semantic_sha256, w.implementation_semantic_sha256) and
        std.mem.eql(u8, c.policy_name, "cumulative256_exact_cycle_v1") and std.mem.eql(u8, c.execution_name, if (resident_v9) "gemma4_native_coalesced_gated_ffn_owner_v1" else "gemma4_native_owner_v1") and
        c.qualification_arm == w.qualification_arm and c.fit_strategy == w.fit_strategy and c.training_policy == w.training_policy and
        c.compiled_prefill_max_copy_bytes == w.compiled_prefill_max_copy_bytes and
        c.compiled_prefill_max_host_staging_bytes == w.compiled_prefill_max_host_staging_bytes and
        c.compiled_prefill_codec_scope == w.compiled_prefill_codec_scope and
        c.compiled_prefill_max_logical_tokens == w.compiled_prefill_max_logical_tokens and
        c.compiled_prefill_layout_policy == w.compiled_prefill_layout_policy;
}

test "qualified startup rejects resident V4 and legacy private V9 cross-kind bindings" {
    const allocator = std.testing.allocator;
    var w: Wire = undefined;
    w.training_policy = .private_train_challenge_hybrid_resident_v1;
    w.release_receipt_path = "/fixture/release.json";
    w.profile_path = "/fixture/bounded-profile.json";
    w.profile_sha256 = "0101010101010101010101010101010101010101010101010101010101010101";
    w.profile_semantic_sha256 = "0202020202020202020202020202020202020202020202020202020202020202";
    w.implementation_semantic_sha256 = "0303030303030303030303030303030303030303030303030303030303030303";
    w.qualification_arm = .attention_matching;
    w.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
    w.compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024;
    w.compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024;
    w.compiled_prefill_codec_scope = .f16_only;
    w.compiled_prefill_max_logical_tokens = 8192;
    w.compiled_prefill_layout_policy = .full_history_exact;
    w.base_compatibility_profile_path = "/fixture/base-profile.json";
    w.base_compatibility_profile_sha256 = "0404040404040404040404040404040404040404040404040404040404040404";
    w.base_compatibility_profile_semantic_sha256 = "0505050505050505050505050505050505050505050505050505050505050505";
    w.resident_hybrid_configuration_path = "/fixture/resident-config.json";
    w.resident_configuration_sha256 = "0606060606060606060606060606060606060606060606060606060606060606";
    w.resident_shape_authority_sha256 = "0707070707070707070707070707070707070707070707070707070707070707";
    w.resident_policy_module_sha256 = "0808080808080808080808080808080808080808080808080808080808080808";
    w.bounded_encoding = .f16;
    w.bounded_logical_tokens = 8128;
    w.bounded_max_output_tokens = 64;
    w.bounded_fit_workers = 1;
    w.bounded_resume_allowed = false;

    var candidate: CandidateV4 = .{
        .schema = "antfly.native-kv-operator-candidate.v9",
        .mode = "staged_candidate",
        .release_receipt_path = w.release_receipt_path,
        .profile_path = w.profile_path,
        .profile_sha256 = w.profile_sha256,
        .profile_semantic_sha256 = w.profile_semantic_sha256,
        .implementation_semantic_sha256 = w.implementation_semantic_sha256,
        .policy_name = "cumulative256_exact_cycle_v1",
        .execution_name = "gemma4_native_coalesced_gated_ffn_owner_v1",
        .qualification_arm = w.qualification_arm,
        .fit_strategy = w.fit_strategy,
        .training_policy = w.training_policy,
        .compiled_prefill_max_copy_bytes = w.compiled_prefill_max_copy_bytes,
        .compiled_prefill_max_host_staging_bytes = w.compiled_prefill_max_host_staging_bytes,
        .compiled_prefill_codec_scope = w.compiled_prefill_codec_scope,
        .compiled_prefill_max_logical_tokens = w.compiled_prefill_max_logical_tokens,
        .compiled_prefill_layout_policy = w.compiled_prefill_layout_policy,
        .diagnostic_only = true,
        .base_compatibility_profile_path = w.base_compatibility_profile_path,
        .base_compatibility_profile_sha256 = w.base_compatibility_profile_sha256,
        .base_compatibility_profile_semantic_sha256 = w.base_compatibility_profile_semantic_sha256,
        .resident_hybrid_configuration_path = w.resident_hybrid_configuration_path,
        .resident_hybrid_configuration_sha256 = w.resident_configuration_sha256,
        .resident_hybrid_shape_authority_sha256 = w.resident_shape_authority_sha256,
        .resident_hybrid_policy_module_sha256 = w.resident_policy_module_sha256,
        .bounded_encoding = w.bounded_encoding,
        .bounded_logical_tokens = w.bounded_logical_tokens,
        .bounded_max_output_tokens = w.bounded_max_output_tokens,
        .bounded_fit_workers = w.bounded_fit_workers,
        .bounded_resume_allowed = w.bounded_resume_allowed,
    };
    try std.testing.expect(candidateV4Matches(w, candidate, true));
    candidate.schema = "antfly.native-kv-operator-candidate.v4";
    candidate.execution_name = "gemma4_native_owner_v1";
    try std.testing.expect(!candidateV4Matches(w, candidate, true));
    candidate.schema = "antfly.native-kv-operator-candidate.v9";
    candidate.execution_name = "gemma4_native_coalesced_gated_ffn_owner_v1";
    try std.testing.expect(!candidateV4Matches(w, candidate, false));

    candidate.diagnostic_only = false;
    try std.testing.expect(!candidateV4Matches(w, candidate, true));
    candidate.diagnostic_only = true;
    candidate.resident_hybrid_configuration_path = "/fixture/cross-campaign.json";
    try std.testing.expect(!candidateV4Matches(w, candidate, true));
    candidate.resident_hybrid_configuration_path = w.resident_hybrid_configuration_path;
    candidate.bounded_logical_tokens = 8127;
    try std.testing.expect(!candidateV4Matches(w, candidate, true));

    const encoded = try std.json.Stringify.valueAlloc(allocator, candidate, .{});
    defer allocator.free(encoded);
    const with_unknown = try std.fmt.allocPrint(allocator, "{{\"unknown_v9_authority\":true,{s}", .{encoded[1..]});
    defer allocator.free(with_unknown);
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(CandidateV4, allocator, with_unknown, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }));
}

pub fn parseAndInstall(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, binding_bytes: []const u8, expected_binding_sha256: [32]u8) !void {
    try verify(binding_bytes, expected_binding_sha256);
    var parsed = std.json.parseFromSlice(Wire, allocator, binding_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidQualifiedPrivateBinding;
    defer parsed.deinit();
    const w = parsed.value;
    const resident = std.mem.eql(u8, w.schema, "antfly.qualified-resident-startup-binding.v1");
    const expected_training_policy: @TypeOf(w.training_policy) = if (resident) .private_train_challenge_hybrid_resident_v1 else .private_dense512_reference_v1;
    const expected_codec_scope: @TypeOf(w.compiled_prefill_codec_scope) = if (resident) .f16_only else .f16_fp8_rotated4;
    const expected_max_logical_tokens: @TypeOf(w.compiled_prefill_max_logical_tokens) = if (resident) 8192 else 32768;
    if ((!resident and !std.mem.eql(u8, w.schema, "antfly.qualified-private-startup-binding.v2")) or
        !std.mem.eql(u8, w.mode, "qualified_private_dynamic_budget") or
        w.qualification_arm != .attention_matching or w.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or
        w.training_policy != expected_training_policy or w.compiled_prefill_max_copy_bytes != 4 * 1024 * 1024 * 1024 or
        w.compiled_prefill_max_host_staging_bytes != 256 * 1024 * 1024 or
        w.compiled_prefill_codec_scope != expected_codec_scope or
        w.compiled_prefill_max_logical_tokens != expected_max_logical_tokens or w.compiled_prefill_layout_policy != .full_history_exact)
        return error.InvalidQualifiedPrivateBinding;
    if (resident) {
        if (w.prepare_host_owner_capability != .bounded_page_e4_resident_1152m_v1 or
            w.prepare_host_owner_physical_bytes != @import("runtime/kv/compaction_profile.zig").bounded_page_e4_resident_1152m_bytes)
            return error.InvalidQualifiedPrivateBinding;
        if (w.base_compatibility_profile_path == null or !std.fs.path.isAbsolute(w.base_compatibility_profile_path.?) or
            w.resident_hybrid_configuration_path == null or !std.fs.path.isAbsolute(w.resident_hybrid_configuration_path.?))
            return error.InvalidQualifiedPrivateBinding;
        inline for (.{ w.base_compatibility_profile_sha256, w.base_compatibility_profile_semantic_sha256, w.resident_training_policy_sha256, w.resident_configuration_sha256, w.resident_source_freeze_sha256, w.resident_shape_authority_sha256, w.resident_policy_module_sha256 }) |field|
            _ = try digest(field orelse return error.InvalidQualifiedPrivateBinding);
        if (w.bounded_encoding != .f16 or w.bounded_logical_tokens != 8128 or w.bounded_max_output_tokens != 64 or
            w.bounded_fit_workers != 1 or w.bounded_resume_allowed != false) return error.InvalidQualifiedPrivateBinding;
    } else if (w.prepare_host_owner_capability != .none or w.prepare_host_owner_physical_bytes != 0 or
        w.base_compatibility_profile_path != null or w.base_compatibility_profile_sha256 != null or w.base_compatibility_profile_semantic_sha256 != null or w.resident_hybrid_configuration_path != null or w.resident_training_policy_sha256 != null or w.resident_configuration_sha256 != null or w.resident_source_freeze_sha256 != null or
        w.resident_shape_authority_sha256 != null or w.resident_policy_module_sha256 != null or w.bounded_encoding != null or
        w.bounded_logical_tokens != null or w.bounded_max_output_tokens != null or w.bounded_fit_workers != null or w.bounded_resume_allowed != null)
        return error.InvalidQualifiedPrivateBinding;

    const full_cli = try readSealed(allocator, io, w.source_full_cli_binding_path, 1024 * 1024);
    defer allocator.free(full_cli);
    const full_cli_sha = try digest(w.source_full_cli_binding_sha256);
    try verify(full_cli, full_cli_sha);
    var cli = std.json.parseFromSlice(CandidateV4, allocator, full_cli, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidQualifiedPrivateBinding;
    defer cli.deinit();
    const c = cli.value;
    if (!candidateV4Matches(w, c, resident))
        return error.InvalidQualifiedPrivateBinding;

    const profile_bytes = try readSealed(allocator, io, w.profile_path, 16 * 1024 * 1024);
    defer allocator.free(profile_bytes);
    const profile_sha = try digest(w.profile_sha256);
    try verify(profile_bytes, profile_sha);
    var bounded_profile: ?bounded_profile_loader.Loaded = if (resident) try bounded_profile_loader.loadBytes(allocator, io, profile_bytes) else null;
    defer if (bounded_profile) |*loaded| loaded.deinit();
    var candidate: ?candidate_loader.Candidate = if (resident) null else try candidate_loader.loadCandidateBytes(allocator, profile_bytes);
    defer if (candidate) |*loaded| loaded.deinit(allocator);
    if (!resident and !candidate.?.isPrivateDense()) return error.InvalidQualifiedPrivateBinding;
    const profile = if (bounded_profile) |*loaded| loaded.profile.profile else candidate.?.profile().*;
    if (profile.prepare_host_owner_capability != w.prepare_host_owner_capability) return error.InvalidQualifiedPrivateBinding;
    const resident_profile = profile.resident_authority;
    if (resident != (resident_profile != null)) return error.InvalidQualifiedPrivateBinding;
    if (resident_profile) |sealed| {
        inline for (.{
            .{ sealed.base_compatibility_artifact_sha256, w.base_compatibility_profile_sha256.? },
            .{ sealed.base_compatibility_semantic_sha256, w.base_compatibility_profile_semantic_sha256.? },
            .{ sealed.training_policy_sha256, w.resident_training_policy_sha256.? },
            .{ sealed.configuration_sha256, w.resident_configuration_sha256.? },
            .{ sealed.source_freeze_sha256, w.resident_source_freeze_sha256.? },
            .{ sealed.shape_authority_sha256, w.resident_shape_authority_sha256.? },
            .{ sealed.policy_module_sha256, w.resident_policy_module_sha256.? },
        }) |joined| if (!std.mem.eql(u8, &joined[0], &try digest(joined[1]))) return error.InvalidQualifiedPrivateBinding;
        if (w.bounded_logical_tokens != sealed.logical_tokens or w.bounded_max_output_tokens != sealed.max_output_tokens or
            w.bounded_fit_workers != sealed.fit_workers or w.bounded_resume_allowed != sealed.resume_allowed or
            w.compiled_prefill_max_copy_bytes != @as(usize, sealed.compiled_max_copy_bytes) or
            w.compiled_prefill_max_host_staging_bytes != @as(usize, sealed.compiled_max_host_staging_bytes) or
            w.compiled_prefill_max_logical_tokens != @as(usize, sealed.compiled_max_logical_tokens)) return error.InvalidQualifiedPrivateBinding;
    }
    const actual_semantic_sha = profile.semanticDigest();
    const actual_producer_policy_sha = profile.producer_policy.digest();
    const expected_semantic_sha = try digest(w.profile_semantic_sha256);
    const expected_producer_policy_sha = try digest(w.producer_fit_policy_sha256);
    if (!std.mem.eql(u8, &actual_semantic_sha, &expected_semantic_sha) or
        !std.meta.eql(profile.producer_policy, fit_policy.Policy.cumulative256_exact_cycle_v1) or
        !std.mem.eql(u8, &actual_producer_policy_sha, &expected_producer_policy_sha))
        return error.InvalidQualifiedPrivateBinding;

    const receipt_bytes = try readSealed(allocator, io, w.release_receipt_path, 4 * 1024 * 1024);
    defer allocator.free(receipt_bytes);
    const release = if (comptime builtin.is_test) fixture_release: {
        const FixtureReceipt = struct { test_fixture_authority: ?[]const u8 = null, binary: struct { sha256: []const u8 } };
        var fixture = std.json.parseFromSlice(FixtureReceipt, allocator, receipt_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch null;
        if (fixture) |*parsed_fixture| {
            defer parsed_fixture.deinit();
            if (parsed_fixture.value.test_fixture_authority != null and std.mem.eql(u8, parsed_fixture.value.test_fixture_authority.?, "antfly.context.test-only-local-authority.v1"))
                break :fixture_release release_build.Binding{ .binary_sha256 = try digest(parsed_fixture.value.binary.sha256), .source_freeze_sha256 = null };
        }
        break :fixture_release try release_build.parseStagedFullCli(allocator, receipt_bytes);
    } else try release_build.parseStagedFullCli(allocator, receipt_bytes);
    var receipt_head = std.json.parseFromSlice(ReceiptHead, allocator, receipt_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidQualifiedPrivateBinding;
    defer receipt_head.deinit();
    if (!std.mem.eql(u8, receipt_head.value.schema, "antfly.full-inference-cli-build.v1") or
        !std.mem.eql(u8, receipt_head.value.binary.sha256, w.binary_sha256)) return error.InvalidQualifiedPrivateBinding;
    const source_freeze_sha = try digest(receipt_head.value.source_freeze.sha256);
    if (resident_profile) |sealed| if (!std.mem.eql(u8, &sealed.source_freeze_sha256, &source_freeze_sha))
        return error.InvalidQualifiedPrivateBinding;
    const full_context_sha = try digest(receipt_head.value.full_context_release_proof.sha256);
    const source_archive_sha = try digest(receipt_head.value.full_source_archive.sha256);
    const freeze_bytes = try readSealed(allocator, io, receipt_head.value.source_freeze.path, 4 * 1024 * 1024);
    defer allocator.free(freeze_bytes);
    try verify(freeze_bytes, source_freeze_sha);
    const context_proof_bytes = try readSealed(allocator, io, receipt_head.value.full_context_release_proof.path, 4 * 1024 * 1024);
    defer allocator.free(context_proof_bytes);
    try verify(context_proof_bytes, full_context_sha);
    try verifySealedFile(io, receipt_head.value.full_source_archive.path, source_archive_sha);

    const enrollment_bytes = try readSealed(allocator, io, w.enrollment_path, 1024 * 1024);
    defer allocator.free(enrollment_bytes);
    const raw_verification_sha = try digest(w.raw_verification_sha256);
    const target_verification_sha = try digest(w.target_verification_sha256);
    try verifySealedFile(io, w.raw_verification_path, raw_verification_sha);
    try verifySealedFile(io, w.target_verification_path, target_verification_sha);
    const observed = try actualObserved();
    const actual_destination = observed.destination;
    const actual_backend_environment = observed.backend_environment_sha256;
    const expected_enrollment: enrollment_mod.Expected = .{
        .artifact_sha256 = try digest(w.enrollment_sha256),
        .raw_verification_sha256 = raw_verification_sha,
        .target_verification_sha256 = target_verification_sha,
        .profile_sha256 = profile_sha,
        .profile_semantic_sha256 = try digest(w.profile_semantic_sha256),
        .binary_sha256 = try digest(w.binary_sha256),
        .source_freeze_sha256 = source_freeze_sha,
        .full_context_release_sha256 = full_context_sha,
        .source_archive_sha256 = source_archive_sha,
        .implementation_semantic_sha256 = try digest(w.implementation_semantic_sha256),
        .runtime_policy_sha256 = try digest(w.runtime_policy_sha256),
        .kernel_caps_sha256 = try digest(w.kernel_caps_sha256),
        .device_registry_id = actual_destination.device_registry_id,
        .physical_memory_bytes = actual_destination.physical_memory_bytes,
        .device_name_sha256 = actual_destination.device_name_sha256,
        .compiler_identity_sha256 = actual_destination.compiler_identity_sha256,
        .backend_environment_sha256 = actual_backend_environment,
    };
    const enrollment = try parseEnrollmentForInstall(allocator, enrollment_bytes, expected_enrollment);
    const handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{
        .max_copy_bytes = w.compiled_prefill_max_copy_bytes,
        .max_host_staging_bytes = w.compiled_prefill_max_host_staging_bytes,
        .codec_scope = w.compiled_prefill_codec_scope,
        .max_logical_tokens = w.compiled_prefill_max_logical_tokens,
        .layout_policy = w.compiled_prefill_layout_policy,
    };
    const resident_authority: ?generation.ResidentHybridAuthority = if (resident_profile) |sealed| .{
        .cohort = .e4_8128_v1,
        .configuration_sha256 = sealed.configuration_sha256,
        .source_freeze_sha256 = sealed.source_freeze_sha256,
        .shape_authority_sha256 = sealed.shape_authority_sha256,
        .policy_module_sha256 = sealed.policy_module_sha256,
    } else null;
    if (resident_authority) |value| try value.validate();
    const resolved = try private_policy.resolve(.{ .authority = .qualified_private, .arm = w.qualification_arm, .fit_strategy = w.fit_strategy, .training_policy = w.training_policy, .handoff = handoff, .encoding = .f16, .logical_tokens = if (resident) 8128 else 32768, .prepare_host_owner_capability = profile.prepare_host_owner_capability, .resident_authority = resident_authority });
    const resolved_sha256 = resolved.digest();
    if (!std.mem.eql(u8, &resolved_sha256, &expected_enrollment.runtime_policy_sha256)) return error.InvalidQualifiedPrivateBinding;
    if (!std.mem.eql(u8, &release.binary_sha256, &expected_enrollment.binary_sha256)) return error.InvalidQualifiedPrivateBinding;
    const actual_caps_sha = try kernelCapsDigest(handoff, actual_destination, actual_backend_environment);
    if (!std.mem.eql(u8, &actual_caps_sha, &expected_enrollment.kernel_caps_sha256)) return error.InvalidQualifiedPrivateBinding;
    const auth = try authority_mod.Authority.initVerified(allocator, io, release.binary_sha256, enrollment.implementation_semantic_sha256, .{ .checkpoint = .{ .weights = profile.identity.artifact, .tokenizer = profile.identity.tokenizer, .template = profile.identity.template, .layout = profile.identity.layout }, .profile_sha256 = profile_sha, .profile_semantic_sha256 = enrollment.profile_semantic_sha256 }, .cumulative256_exact_cycle_v1, native_release.Execution{ .forward_frame = true, .precomputed_tail = true, .retain_hidden = true, .matrix_stage = true, .coalesced_stage = resident, .resident_gated_ffn_fusion = resident, .ple_tail = true, .resource_barriers = true });
    try node.installQualifiedPrivateGrant(.{ .authority = auth, .enrollment = enrollment, .runtime_policy = resolved, .training_policy = w.training_policy, .training_mix_policy_sha256 = if (resident_profile) |sealed| sealed.training_policy_sha256 else @splat(0), .prepare_host_owner_capability = w.prepare_host_owner_capability, .prepare_host_owner_physical_bytes = w.prepare_host_owner_physical_bytes, .resident_hybrid_authority = resident_authority, .source_full_cli_binding_sha256 = full_cli_sha }, profile, handoff);
}

pub fn installFromConfigFile(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, path: []const u8, expected_sha256_hex: []const u8) !void {
    const bytes = try readSealed(allocator, io, path, 1024 * 1024);
    defer allocator.free(bytes);
    try parseAndInstall(allocator, io, node, bytes, try digest(expected_sha256_hex));
}

test "qualified private startup requires typed schema and independent digest" {
    const bogus = "{\"schema\":\"antfly.qualified-private-startup-binding.v2\",\"qualified_private\":true}";
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bogus, &actual, .{});
    try std.testing.expectError(error.InvalidQualifiedPrivateBinding, parseAndInstall(std.testing.allocator, std.testing.io, undefined, bogus, actual));
    actual[0] ^= 1;
    try std.testing.expectError(error.InvalidQualifiedPrivateBinding, parseAndInstall(std.testing.allocator, std.testing.io, undefined, bogus, actual));
}

test "qualified private joins every v4 candidate field and rejects corruption" {
    var w: Wire = undefined;
    w.release_receipt_path = "/receipt";
    w.profile_path = "/profile";
    w.profile_sha256 = "aa";
    w.profile_semantic_sha256 = "bb";
    w.implementation_semantic_sha256 = "cc";
    w.qualification_arm = .attention_matching;
    w.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
    w.training_policy = .private_dense512_reference_v1;
    w.compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024;
    w.compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024;
    w.compiled_prefill_codec_scope = .f16_fp8_rotated4;
    w.compiled_prefill_max_logical_tokens = 32768;
    w.compiled_prefill_layout_policy = .full_history_exact;
    var c: CandidateV4 = .{
        .schema = "antfly.native-kv-operator-candidate.v4",
        .mode = "staged_candidate",
        .release_receipt_path = w.release_receipt_path,
        .profile_path = w.profile_path,
        .profile_sha256 = w.profile_sha256,
        .profile_semantic_sha256 = w.profile_semantic_sha256,
        .implementation_semantic_sha256 = w.implementation_semantic_sha256,
        .policy_name = "cumulative256_exact_cycle_v1",
        .execution_name = "gemma4_native_owner_v1",
        .qualification_arm = w.qualification_arm,
        .fit_strategy = w.fit_strategy,
        .training_policy = w.training_policy,
        .compiled_prefill_max_copy_bytes = w.compiled_prefill_max_copy_bytes,
        .compiled_prefill_max_host_staging_bytes = w.compiled_prefill_max_host_staging_bytes,
        .compiled_prefill_codec_scope = w.compiled_prefill_codec_scope,
        .compiled_prefill_max_logical_tokens = w.compiled_prefill_max_logical_tokens,
        .compiled_prefill_layout_policy = w.compiled_prefill_layout_policy,
    };
    try std.testing.expect(candidateV4Matches(w, c, false));
    c.schema = "antfly.native-kv-operator-candidate.v9";
    c.execution_name = "gemma4_native_coalesced_gated_ffn_owner_v1";
    try std.testing.expect(!candidateV4Matches(w, c, false));
    c.schema = "antfly.native-kv-operator-candidate.v4";
    c.execution_name = "gemma4_native_owner_v1";
    c.profile_semantic_sha256 = "corrupt";
    try std.testing.expect(!candidateV4Matches(w, c, false));
    c.profile_semantic_sha256 = w.profile_semantic_sha256;
    c.compiled_prefill_max_logical_tokens -= 1;
    try std.testing.expect(!candidateV4Matches(w, c, false));
    c.compiled_prefill_max_logical_tokens += 1;
    c.policy_name = "legacy_fixed";
    try std.testing.expect(!candidateV4Matches(w, c, false));
    c.policy_name = "cumulative256_exact_cycle_v1";
    c.prepared_quality_source_path = "/issuer";
    try std.testing.expect(!candidateV4Matches(w, c, false));
}

test "qualified private kernel caps derive from injected observed identity and sealed v4 tuple" {
    var handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{
        .max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .max_host_staging_bytes = 256 * 1024 * 1024,
        .codec_scope = .f16_fp8_rotated4,
        .max_logical_tokens = 32768,
        .layout_policy = .full_history_exact,
    };
    var destination: native_release.Destination = .{
        .device_registry_id = 1,
        .physical_memory_bytes = 24 * 1024 * 1024 * 1024,
        .device_name_sha256 = @splat(1),
        .compiler_identity_sha256 = @splat(2),
    };
    var environment: [32]u8 = @splat(3);
    const actual = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.allEqual(u8, &actual, 0));
    destination.device_registry_id += 1;
    var mutated = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.eql(u8, &actual, &mutated));
    destination.device_registry_id -= 1;
    destination.device_name_sha256[0] ^= 1;
    mutated = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.eql(u8, &actual, &mutated));
    destination.device_name_sha256[0] ^= 1;
    destination.compiler_identity_sha256[0] ^= 1;
    mutated = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.eql(u8, &actual, &mutated));
    destination.compiler_identity_sha256[0] ^= 1;
    environment[0] ^= 1;
    mutated = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.eql(u8, &actual, &mutated));
    environment[0] ^= 1;
    destination.physical_memory_bytes -= 1;
    mutated = try kernelCapsDigest(handoff, destination, environment);
    try std.testing.expect(!std.mem.eql(u8, &actual, &mutated));
    destination.physical_memory_bytes += 1;
    handoff.max_logical_tokens = 8192;
    try std.testing.expectError(error.InvalidQualifiedPrivateBinding, kernelCapsDigest(handoff, destination, environment));
}

fn testExecutable(allocator: std.mem.Allocator, io: std.Io) !struct { sha256: [32]u8, bytes: u64 } {
    const path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(path);
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var buffer: [64 * 1024]u8 = undefined;
    var offset: u64 = 0;
    while (offset < stat.size) {
        const count = try file.readPositionalAll(io, buffer[0..@intCast(@min(stat.size - offset, buffer.len))], offset);
        if (count == 0) return error.InvalidQualifiedPrivateBinding;
        hash.update(buffer[0..count]);
        offset += count;
    }
    return .{ .sha256 = hash.finalResult(), .bytes = stat.size };
}

test "qualified resident config file verifies test-only receipt enrollment and installs final grant" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const path = struct {
        fn join(a: std.mem.Allocator, root_path: []const u8, name: []const u8) ![]u8 {
            return std.fs.path.join(a, &.{ root_path, name });
        }
    }.join;
    const b_path = try path(allocator, root, "B.json");
    defer allocator.free(b_path);
    const c_path = try path(allocator, root, "C.json");
    defer allocator.free(c_path);
    const p_path = try path(allocator, root, "P.json");
    defer allocator.free(p_path);
    const candidate_path = try path(allocator, root, "candidate.json");
    defer allocator.free(candidate_path);
    const receipt_path = try path(allocator, root, "receipt.json");
    defer allocator.free(receipt_path);
    const freeze_path = try path(allocator, root, "source-freeze.json");
    defer allocator.free(freeze_path);
    const proof_path = try path(allocator, root, "context-proof.json");
    defer allocator.free(proof_path);
    const archive_path = try path(allocator, root, "source.tar");
    defer allocator.free(archive_path);
    const raw_path = try path(allocator, root, "raw.json");
    defer allocator.free(raw_path);
    const target_path = try path(allocator, root, "target.json");
    defer allocator.free(target_path);
    const enrollment_path = try path(allocator, root, "enrollment.json");
    defer allocator.free(enrollment_path);
    const binding_path = try path(allocator, root, "qualified.json");
    defer allocator.free(binding_path);

    const freeze_bytes = "test-only-source-freeze";
    const proof_bytes = "test-only-context-proof";
    const archive_bytes = "test-only-source-archive";
    const raw_bytes = "test-only-raw-verification";
    const target_bytes = "test-only-target-verification";
    var source: [32]u8 = undefined;
    var proof_sha: [32]u8 = undefined;
    var archive_sha: [32]u8 = undefined;
    var raw_sha: [32]u8 = undefined;
    var target_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(freeze_bytes, &source, .{});
    std.crypto.hash.sha2.Sha256.hash(proof_bytes, &proof_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(archive_bytes, &archive_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(raw_bytes, &raw_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(target_bytes, &target_sha, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "source-freeze.json", .data = freeze_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "context-proof.json", .data = proof_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "source.tar", .data = archive_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "raw.json", .data = raw_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "target.json", .data = target_bytes });

    const base = try bounded_profile_loader.testBaseJson(allocator, source);
    defer allocator.free(base);
    var base_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(base, &base_sha, .{});
    const base_semantic = @import("context_resident_hybrid_compatibility_profile.zig").expectedProfileSemantic();
    const configuration = try bounded_profile_loader.testConfigJson(allocator, source, base_semantic);
    defer allocator.free(configuration);
    var configuration_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(configuration, &configuration_sha, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "B.json", .data = base });
    try tmp.dir.writeFile(io, .{ .sub_path = "C.json", .data = configuration });
    const profile_bytes = try @import("context_private_dense_profile.zig").testQualifiedResidentJson(allocator, b_path, c_path, base_sha, base_semantic, configuration_sha, source, base_semantic);
    defer allocator.free(profile_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "P.json", .data = profile_bytes });
    var profile_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(profile_bytes, &profile_sha, .{});
    var loaded = try bounded_profile_loader.loadBytes(allocator, io, profile_bytes);
    defer loaded.deinit();
    const profile_semantic = loaded.profile.profile.semanticDigest();
    const producer_policy_sha = loaded.profile.profile.producer_policy.digest();
    const executable = try testExecutable(allocator, io);

    const receipt = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.full-inference-cli-build.v1",
        .test_fixture_authority = "antfly.context.test-only-local-authority.v1",
        .source_freeze = .{ .path = freeze_path, .sha256 = std.fmt.bytesToHex(source, .lower) },
        .full_context_release_proof = .{ .path = proof_path, .sha256 = std.fmt.bytesToHex(proof_sha, .lower) },
        .full_source_archive = .{ .path = archive_path, .sha256 = std.fmt.bytesToHex(archive_sha, .lower) },
        .binary = .{ .path = "/test-only/current-executable", .sha256 = std.fmt.bytesToHex(executable.sha256, .lower), .bytes = executable.bytes },
    }, .{});
    defer allocator.free(receipt);
    try tmp.dir.writeFile(io, .{ .sub_path = "receipt.json", .data = receipt });

    const candidate: CandidateV4 = .{
        .schema = "antfly.native-kv-operator-candidate.v9",
        .mode = "staged_candidate",
        .release_receipt_path = receipt_path,
        .profile_path = p_path,
        .profile_sha256 = &std.fmt.bytesToHex(profile_sha, .lower),
        .profile_semantic_sha256 = &std.fmt.bytesToHex(profile_semantic, .lower),
        .implementation_semantic_sha256 = &std.fmt.bytesToHex(@import("compaction_identity").digest, .lower),
        .policy_name = "cumulative256_exact_cycle_v1",
        .execution_name = "gemma4_native_coalesced_gated_ffn_owner_v1",
        .qualification_arm = .attention_matching,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .training_policy = .private_train_challenge_hybrid_resident_v1,
        .compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024,
        .compiled_prefill_codec_scope = .f16_only,
        .compiled_prefill_max_logical_tokens = 8192,
        .compiled_prefill_layout_policy = .full_history_exact,
        .diagnostic_only = true,
        .base_compatibility_profile_path = b_path,
        .base_compatibility_profile_sha256 = &std.fmt.bytesToHex(base_sha, .lower),
        .base_compatibility_profile_semantic_sha256 = &std.fmt.bytesToHex(base_semantic, .lower),
        .resident_hybrid_configuration_path = c_path,
        .resident_hybrid_configuration_sha256 = &std.fmt.bytesToHex(configuration_sha, .lower),
        .resident_hybrid_shape_authority_sha256 = &std.fmt.bytesToHex(base_semantic, .lower),
        .resident_hybrid_policy_module_sha256 = &std.fmt.bytesToHex(@import("context_resident_hybrid_staged_configuration.zig").policy_module_sha256, .lower),
        .bounded_encoding = .f16,
        .bounded_logical_tokens = 8128,
        .bounded_max_output_tokens = 64,
        .bounded_fit_workers = 1,
        .bounded_resume_allowed = false,
    };
    const candidate_bytes = try std.json.Stringify.valueAlloc(allocator, candidate, .{});
    defer allocator.free(candidate_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = candidate_bytes });
    var candidate_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(candidate_bytes, &candidate_sha, .{});

    const observed: native_release.Observed = .{ .destination = .{ .device_registry_id = 1, .physical_memory_bytes = 24 * 1024 * 1024 * 1024, .device_name_sha256 = @splat(9), .compiler_identity_sha256 = @splat(10) }, .backend_environment_sha256 = @splat(11) };
    test_observed = observed;
    defer test_observed = null;
    const handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{ .max_copy_bytes = 4 * 1024 * 1024 * 1024, .max_host_staging_bytes = 256 * 1024 * 1024, .codec_scope = .f16_only, .max_logical_tokens = 8192, .layout_policy = .full_history_exact };
    const resident_authority: generation.ResidentHybridAuthority = .{ .cohort = .e4_8128_v1, .configuration_sha256 = configuration_sha, .source_freeze_sha256 = source, .shape_authority_sha256 = base_semantic, .policy_module_sha256 = @import("context_resident_hybrid_staged_configuration.zig").policy_module_sha256 };
    const resolved = try private_policy.resolve(.{ .authority = .qualified_private, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_train_challenge_hybrid_resident_v1, .handoff = handoff, .encoding = .f16, .logical_tokens = 8128, .prepare_host_owner_capability = .bounded_page_e4_resident_1152m_v1, .resident_authority = resident_authority });
    const runtime_policy_sha = resolved.digest();
    const caps_sha = try kernelCapsDigest(handoff, observed.destination, observed.backend_environment_sha256);
    var expected: enrollment_mod.Expected = .{
        .artifact_sha256 = undefined,
        .raw_verification_sha256 = raw_sha,
        .target_verification_sha256 = target_sha,
        .profile_sha256 = profile_sha,
        .profile_semantic_sha256 = profile_semantic,
        .binary_sha256 = executable.sha256,
        .source_freeze_sha256 = source,
        .full_context_release_sha256 = proof_sha,
        .source_archive_sha256 = archive_sha,
        .implementation_semantic_sha256 = @import("compaction_identity").digest,
        .runtime_policy_sha256 = runtime_policy_sha,
        .kernel_caps_sha256 = caps_sha,
        .device_registry_id = observed.destination.device_registry_id,
        .physical_memory_bytes = observed.destination.physical_memory_bytes,
        .device_name_sha256 = observed.destination.device_name_sha256,
        .compiler_identity_sha256 = observed.destination.compiler_identity_sha256,
        .backend_environment_sha256 = observed.backend_environment_sha256,
    };
    const enrollment_bytes = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.context.test-only-qualified-resident-enrollment.v1",
        .test_fixture_authority = "antfly.context.test-only-local-authority.v1",
        .qualification_credit = false,
        .production_qualified = false,
        .enrollment_applied = false,
        .expected_join_sha256 = std.fmt.bytesToHex(enrollmentJoinDigest(expected), .lower),
    }, .{});
    defer allocator.free(enrollment_bytes);
    std.crypto.hash.sha2.Sha256.hash(enrollment_bytes, &expected.artifact_sha256, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "enrollment.json", .data = enrollment_bytes });

    const wire: Wire = .{
        .schema = "antfly.qualified-resident-startup-binding.v1",
        .mode = "qualified_private_dynamic_budget",
        .source_full_cli_binding_path = candidate_path,
        .source_full_cli_binding_sha256 = &std.fmt.bytesToHex(candidate_sha, .lower),
        .release_receipt_path = receipt_path,
        .profile_path = p_path,
        .enrollment_path = enrollment_path,
        .enrollment_sha256 = &std.fmt.bytesToHex(expected.artifact_sha256, .lower),
        .raw_verification_sha256 = &std.fmt.bytesToHex(raw_sha, .lower),
        .target_verification_sha256 = &std.fmt.bytesToHex(target_sha, .lower),
        .raw_verification_path = raw_path,
        .target_verification_path = target_path,
        .profile_sha256 = candidate.profile_sha256,
        .profile_semantic_sha256 = candidate.profile_semantic_sha256,
        .binary_sha256 = candidate.implementation_semantic_sha256,
        .implementation_semantic_sha256 = candidate.implementation_semantic_sha256,
        .runtime_policy_sha256 = &std.fmt.bytesToHex(runtime_policy_sha, .lower),
        .producer_fit_policy_sha256 = &std.fmt.bytesToHex(producer_policy_sha, .lower),
        .kernel_caps_sha256 = &std.fmt.bytesToHex(caps_sha, .lower),
        .qualification_arm = .attention_matching,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .training_policy = .private_train_challenge_hybrid_resident_v1,
        .compiled_prefill_max_copy_bytes = handoff.max_copy_bytes,
        .compiled_prefill_max_host_staging_bytes = handoff.max_host_staging_bytes,
        .compiled_prefill_codec_scope = handoff.codec_scope,
        .compiled_prefill_max_logical_tokens = handoff.max_logical_tokens,
        .compiled_prefill_layout_policy = handoff.layout_policy,
        .prepare_host_owner_capability = .bounded_page_e4_resident_1152m_v1,
        .prepare_host_owner_physical_bytes = @import("runtime/kv/compaction_profile.zig").bounded_page_e4_resident_1152m_bytes,
        .resident_configuration_sha256 = candidate.resident_hybrid_configuration_sha256,
        .resident_source_freeze_sha256 = &std.fmt.bytesToHex(source, .lower),
        .resident_shape_authority_sha256 = candidate.resident_hybrid_shape_authority_sha256,
        .resident_policy_module_sha256 = candidate.resident_hybrid_policy_module_sha256,
        .resident_training_policy_sha256 = &std.fmt.bytesToHex(loaded.profile.profile.resident_authority.?.training_policy_sha256, .lower),
        .base_compatibility_profile_path = b_path,
        .base_compatibility_profile_sha256 = candidate.base_compatibility_profile_sha256,
        .base_compatibility_profile_semantic_sha256 = candidate.base_compatibility_profile_semantic_sha256,
        .resident_hybrid_configuration_path = c_path,
        .bounded_encoding = .f16,
        .bounded_logical_tokens = 8128,
        .bounded_max_output_tokens = 64,
        .bounded_fit_workers = 1,
        .bounded_resume_allowed = false,
    };
    var corrected = wire;
    corrected.binary_sha256 = &std.fmt.bytesToHex(executable.sha256, .lower);
    const binding_bytes = try std.json.Stringify.valueAlloc(allocator, corrected, .{});
    defer allocator.free(binding_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "qualified.json", .data = binding_bytes });
    var binding_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(binding_bytes, &binding_sha, .{});

    var legacy_candidate = candidate;
    legacy_candidate.schema = "antfly.native-kv-operator-candidate.v4";
    legacy_candidate.execution_name = "gemma4_native_owner_v1";
    const legacy_candidate_bytes = try std.json.Stringify.valueAlloc(allocator, legacy_candidate, .{});
    defer allocator.free(legacy_candidate_bytes);
    var legacy_candidate_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(legacy_candidate_bytes, &legacy_candidate_sha, .{});
    var cross_kind = corrected;
    cross_kind.source_full_cli_binding_sha256 = &std.fmt.bytesToHex(legacy_candidate_sha, .lower);
    const cross_kind_bytes = try std.json.Stringify.valueAlloc(allocator, cross_kind, .{});
    defer allocator.free(cross_kind_bytes);
    var cross_kind_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(cross_kind_bytes, &cross_kind_sha, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = legacy_candidate_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "qualified.json", .data = cross_kind_bytes });
    var cross_kind_node = try server.Node.init(allocator, .{ .max_concurrent_requests = 1 });
    defer cross_kind_node.deinit();
    try std.testing.expectError(error.InvalidQualifiedPrivateBinding, installFromConfigFile(allocator, io, &cross_kind_node, binding_path, &std.fmt.bytesToHex(cross_kind_sha, .lower)));
    try std.testing.expect(cross_kind_node.native_kv_qualified_private_grant == null);
    try std.testing.expect(cross_kind_node.native_kv_producer_authority == null and cross_kind_node.native_kv_trusted_producer == null);
    try std.testing.expect(cross_kind_node.model_manager.qualification_candidate == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = candidate_bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "qualified.json", .data = binding_bytes });

    var tampered = try allocator.dupe(u8, binding_bytes);
    defer allocator.free(tampered);
    const tuple = "\"bounded_logical_tokens\":8128";
    const tuple_at = std.mem.indexOf(u8, tampered, tuple).?;
    tampered[tuple_at + tuple.len - 1] = '7';
    var tampered_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(tampered, &tampered_sha, .{});
    var rejected = try server.Node.init(allocator, .{ .max_concurrent_requests = 1 });
    defer rejected.deinit();
    try std.testing.expectError(error.InvalidQualifiedPrivateBinding, parseAndInstall(allocator, io, &rejected, tampered, tampered_sha));
    try std.testing.expect(rejected.native_kv_qualified_private_grant == null);
    try std.testing.expect(rejected.native_kv_producer_authority == null and rejected.native_kv_trusted_producer == null);
    try std.testing.expect(rejected.model_manager.qualification_candidate == null);

    var node = try server.Node.init(allocator, .{ .max_concurrent_requests = 1 });
    defer node.deinit();
    try installFromConfigFile(allocator, io, &node, binding_path, &std.fmt.bytesToHex(binding_sha, .lower));
    try std.testing.expect(node.native_kv_qualified_private_grant != null);
    try std.testing.expect(node.native_kv_producer_authority != null and node.native_kv_trusted_producer != null);
    try std.testing.expect(node.model_manager.qualification_candidate != null);
    try std.testing.expectEqual(.bounded_page_e4_resident_1152m_v1, node.native_kv_qualified_private_grant.?.prepare_host_owner_capability);
    try std.testing.expectEqualSlices(u8, &candidate_sha, &node.native_kv_qualified_private_grant.?.source_full_cli_binding_sha256);
}
