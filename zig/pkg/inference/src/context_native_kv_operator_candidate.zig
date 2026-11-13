// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Full-CLI entrypoint for a staged, non-enrolled AFATT002 producer.
//! The command binding supplies a hash from a sealed release-build receipt;
//! the running executable is rehashed before the candidate becomes reachable.
const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const server = @import("server/server.zig");
const producer_authority = @import("server/native_kv_producer_authority.zig");
const profile_mod = @import("runtime/kv/compaction_profile.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const release_build = @import("context_release_build_binding.zig");
const native_release = @import("runtime/kv/attention_match_release.zig");
const candidate_loader = @import("context_serving_entrypoint.zig");
const staged_exact_baseline = @import("context_native_kv_staged_exact_baseline.zig");
const generation = @import("pipelines/generation.zig");
const resident_hybrid = @import("context_resident_hybrid_staged_configuration.zig");
const resident_profile = @import("context_resident_hybrid_compatibility_profile.zig");
const bounded_resident_profile = @import("context_qualified_resident_bounded_profile.zig");
const formal_inventory = @import("runtime/kv/attention_match_e2_formal_owner_inventory.zig");
const metal_runtime = if (build_options.enable_metal and builtin.os.tag == .macos)
    @import("backends/metal_runtime.zig")
else
    struct {
        fn nativeCheckpointDestination() !native_release.Destination {
            return error.UnsupportedQualifiedPrivateHardware;
        }
    };

const ContextTestCandidateReceipt = struct {
    schema: []const u8,
    diagnostic_only: bool,
    qualification_credit: bool,
    serving_qualified: bool,
    registry_enrollment: bool,
    parent_full_cli_receipt: struct { path: []const u8, sha256: []const u8 },
    source_freeze: struct { path: []const u8, sha256: []const u8 },
    context_release_proof: struct { path: []const u8, sha256: []const u8 },
    binary: struct { path: []const u8, sha256: []const u8, bytes: u64 },
};

const ContextTestReceiptAuthorities = struct {
    parent_full_cli: release_build.Binding,
    context_candidate: release_build.Binding,
};

fn parseContextTestCandidateReceiptAuthorities(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !ContextTestReceiptAuthorities {
    var parsed = std.json.parseFromSlice(ContextTestCandidateReceipt, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch
        return error.InvalidContextReleaseBuildReceipt;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.context-test-candidate-build.v1") or
        !value.diagnostic_only or value.qualification_credit or value.serving_qualified or value.registry_enrollment or
        !std.fs.path.isAbsolute(value.parent_full_cli_receipt.path) or !std.fs.path.isAbsolute(value.source_freeze.path) or !std.fs.path.isAbsolute(value.context_release_proof.path) or
        !std.fs.path.isAbsolute(value.binary.path) or value.binary.bytes == 0)
        return error.InvalidContextReleaseBuildReceipt;
    const source_bytes = try readNoFollow(allocator, io, value.source_freeze.path, 16 * 1024 * 1024);
    defer allocator.free(source_bytes);
    const parent_bytes = try readNoFollow(allocator, io, value.parent_full_cli_receipt.path, 4 * 1024 * 1024);
    defer allocator.free(parent_bytes);
    const proof_bytes = try readNoFollow(allocator, io, value.context_release_proof.path, 4 * 1024 * 1024);
    defer allocator.free(proof_bytes);
    const binary_bytes = try readNoFollow(allocator, io, value.binary.path, 2 * 1024 * 1024 * 1024);
    defer allocator.free(binary_bytes);
    var source_sha: [32]u8 = undefined;
    var parent_sha: [32]u8 = undefined;
    var proof_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source_bytes, &source_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(parent_bytes, &parent_sha, .{});
    std.crypto.hash.sha2.Sha256.hash(proof_bytes, &proof_sha, .{});
    if (!std.crypto.timing_safe.eql([32]u8, parent_sha, try parseDigest(value.parent_full_cli_receipt.sha256)) or
        !std.crypto.timing_safe.eql([32]u8, source_sha, try parseDigest(value.source_freeze.sha256)) or
        !std.crypto.timing_safe.eql([32]u8, proof_sha, try parseDigest(value.context_release_proof.sha256)))
        return error.InvalidContextReleaseBuildReceipt;
    const parent = try release_build.parseStagedFullCli(allocator, parent_bytes);
    if (parent.source_freeze_sha256 == null or
        !std.crypto.timing_safe.eql([32]u8, parent.source_freeze_sha256.?, source_sha))
        return error.InvalidContextReleaseBuildReceipt;
    const ParentJoin = struct {
        source_freeze: struct { path: []const u8, sha256: []const u8 },
        full_context_release_proof: struct { path: []const u8, sha256: []const u8 },
    };
    var parent_join = std.json.parseFromSlice(ParentJoin, allocator, parent_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch
        return error.InvalidContextReleaseBuildReceipt;
    defer parent_join.deinit();
    if (!std.mem.eql(u8, parent_join.value.source_freeze.path, value.source_freeze.path) or
        !std.mem.eql(u8, parent_join.value.source_freeze.sha256, value.source_freeze.sha256) or
        !std.mem.eql(u8, parent_join.value.full_context_release_proof.path, value.context_release_proof.path) or
        !std.mem.eql(u8, parent_join.value.full_context_release_proof.sha256, value.context_release_proof.sha256))
        return error.InvalidContextReleaseBuildReceipt;
    const proof = try release_build.parseStagedFullContextProof(allocator, proof_bytes);
    const binary_sha = try parseDigest(value.binary.sha256);
    var observed_binary_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(binary_bytes, &observed_binary_sha, .{});
    if (binary_bytes.len != value.binary.bytes or
        !std.crypto.timing_safe.eql([32]u8, observed_binary_sha, binary_sha) or
        !std.crypto.timing_safe.eql([32]u8, proof.binary_sha256, binary_sha))
        return error.InvalidContextReleaseBuildReceipt;
    return .{
        .parent_full_cli = parent,
        .context_candidate = .{ .binary_sha256 = binary_sha, .source_freeze_sha256 = source_sha },
    };
}

fn parseContextTestCandidateReceipt(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !release_build.Binding {
    return (try parseContextTestCandidateReceiptAuthorities(allocator, io, bytes)).context_candidate;
}

/// Capture-only authority for context-model diagnostics. This deliberately
/// returns both the verified parent full-CLI binary/source binding and the
/// digest of the executable that is running the context test. The two files
/// are distinct authorities and must never be represented as one binary.
pub const ContextTestCaptureAuthority = struct {
    parent: release_build.Binding,
    context_binary_sha256: [32]u8,
};

pub fn contextTestCaptureAuthority(
    allocator: std.mem.Allocator,
    io: std.Io,
    receipt_path: []const u8,
) !ContextTestCaptureAuthority {
    if (!std.fs.path.isAbsolute(receipt_path)) return error.InvalidContextReleaseBuildReceipt;
    const receipt = try readNoFollow(allocator, io, receipt_path, 4 * 1024 * 1024);
    defer allocator.free(receipt);
    const authorities = try parseContextTestCandidateReceiptAuthorities(allocator, io, receipt);
    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);
    const executable = try readNoFollow(allocator, io, executable_path, 2 * 1024 * 1024 * 1024);
    defer allocator.free(executable);
    var context_binary_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(executable, &context_binary_sha256, .{});
    if (std.mem.allEqual(u8, &context_binary_sha256, 0)) return error.InvalidContextReleaseBuildReceipt;
    if (!std.crypto.timing_safe.eql([32]u8, context_binary_sha256, authorities.context_candidate.binary_sha256))
        return error.InvalidContextReleaseBuildReceipt;
    return .{ .parent = authorities.parent_full_cli, .context_binary_sha256 = context_binary_sha256 };
}

fn parseStagedReceiptForInstall(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !release_build.Binding {
    if (comptime builtin.is_test) {
        const TestReceipt = struct {
            schema: []const u8,
            test_fixture_authority: ?[]const u8 = null,
            source_freeze: struct { sha256: []const u8 },
            binary: struct { sha256: []const u8 },
        };
        var fixture = std.json.parseFromSlice(TestReceipt, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch null;
        if (fixture) |*parsed| {
            defer parsed.deinit();
            if (parsed.value.test_fixture_authority != null and
                std.mem.eql(u8, parsed.value.test_fixture_authority.?, "antfly.context.test-only-local-authority.v1") and
                std.mem.eql(u8, parsed.value.schema, "antfly.full-inference-cli-build.v1"))
                return .{
                    .binary_sha256 = try parseDigest(parsed.value.binary.sha256),
                    .source_freeze_sha256 = try parseDigest(parsed.value.source_freeze.sha256),
                };
        }
    }
    if (comptime builtin.is_test) {
        var schema = std.json.parseFromSlice(struct { schema: []const u8 }, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch
            return error.InvalidStagedFullCliBuildReceipt;
        defer schema.deinit();
        if (std.mem.eql(u8, schema.value.schema, "antfly.context-test-candidate-build.v1"))
            return parseContextTestCandidateReceipt(allocator, io, bytes);
    }
    return release_build.parseStagedFullCli(allocator, bytes);
}

fn loadFormalResidentAuthority(
    allocator: std.mem.Allocator,
    io: std.Io,
    parsed_configuration: resident_hybrid.Authority,
    parent_receipt_bytes: []const u8,
    value: FileBinding,
) !resident_hybrid.Authority {
    const inventory_bytes = try readNoFollow(allocator, io, value.formal_owner_inventory_path.?, 4 * 1024 * 1024);
    defer allocator.free(inventory_bytes);
    var inventory_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(inventory_bytes, &inventory_sha256, .{});
    if (!std.crypto.timing_safe.eql([32]u8, inventory_sha256, parsed_configuration.formal_owner_inventory_sha256.?))
        return error.InvalidQualificationCandidate;
    const Capture = struct {
        schema: []const u8,
        capture_only: bool,
        fit_authority: bool,
        release_authority: bool,
        artifact: formal_inventory.Artifact,
        final_configuration_sha256: []const u8,
    };
    var captured = std.json.parseFromSlice(Capture, allocator, inventory_bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    }) catch return error.InvalidQualificationCandidate;
    defer captured.deinit();
    if (!std.mem.eql(u8, captured.value.schema, "antfly.e2.formal-owner-inventory.capture.v1") or
        !captured.value.capture_only or captured.value.fit_authority or captured.value.release_authority)
        return error.InvalidQualificationCandidate;
    try captured.value.artifact.validate();
    const final_configuration = formal_inventory.finalConfigurationDigest(captured.value.artifact);
    if (!std.crypto.timing_safe.eql([32]u8, final_configuration, try parseDigest(captured.value.final_configuration_sha256)) or
        @as(usize, captured.value.artifact.source_rows) != parsed_configuration.e2_fit_source_rows)
        return error.InvalidQualificationCandidate;

    const context_receipt_bytes = try readNoFollow(allocator, io, value.formal_context_candidate_receipt_path.?, 4 * 1024 * 1024);
    defer allocator.free(context_receipt_bytes);
    var context_receipt_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(context_receipt_bytes, &context_receipt_sha256, .{});
    if (!std.crypto.timing_safe.eql([32]u8, context_receipt_sha256, parsed_configuration.formal_context_candidate_receipt_sha256.?))
        return error.InvalidQualificationCandidate;
    const context_authorities = try parseContextTestCandidateReceiptAuthorities(allocator, io, context_receipt_bytes);
    const parent = try release_build.parseStagedFullCli(allocator, parent_receipt_bytes);
    if (parent.source_freeze_sha256 == null or context_authorities.parent_full_cli.source_freeze_sha256 == null or
        !std.crypto.timing_safe.eql([32]u8, parent.binary_sha256, context_authorities.parent_full_cli.binary_sha256) or
        !std.crypto.timing_safe.eql([32]u8, parent.source_freeze_sha256.?, context_authorities.parent_full_cli.source_freeze_sha256.?))
        return error.InvalidQualificationCandidate;
    const destination = try metal_runtime.nativeCheckpointDestination();
    try destination.validate();
    const authority = try generation.formalE2ResidentAuthority(captured.value.artifact, .{
        .parent_full_cli_binary_sha256 = parent.binary_sha256,
        .source_freeze_sha256 = parent.source_freeze_sha256.?,
        .context_binary_sha256 = context_authorities.context_candidate.binary_sha256,
        .hardware_sha256 = destination.fingerprint(),
    });
    return .{
        .configuration_sha256 = authority.configuration_sha256,
        .source_freeze_sha256 = authority.source_freeze_sha256,
        .shape_authority_sha256 = authority.shape_authority_sha256,
        .policy_module_sha256 = authority.policy_module_sha256,
        .profile_semantic_sha256 = parsed_configuration.profile_semantic_sha256,
        .cohort = .e2_8128_v1,
        .guarded_development_trial = true,
        .e2_fit_source_rows = authority.e2_fit_source_rows,
        .formal_owner_inventory_sha256 = authority.e2_formal_owner_inventory_sha256,
        .formal_context_candidate_receipt_sha256 = parsed_configuration.formal_context_candidate_receipt_sha256,
        .formal_owner_inventory = captured.value.artifact,
    };
}

fn validateResidentCompatibilityBinding(profile_semantic: [32]u8, profile_source: [32]u8, authority: resident_hybrid.Authority, receipt_source: [32]u8) !void {
    if (!std.mem.eql(u8, &profile_semantic, &authority.profile_semantic_sha256) or
        !std.mem.eql(u8, &profile_source, &authority.source_freeze_sha256) or
        !std.mem.eql(u8, &receipt_source, &authority.source_freeze_sha256))
        return error.InvalidQualificationCandidate;
}

fn validateFormalFileTuple(
    resident: resident_hybrid.Authority,
    inventory_path: ?[]const u8,
    inventory_sha256: ?[]const u8,
    receipt_path: ?[]const u8,
    receipt_sha256: ?[]const u8,
) !bool {
    const formal = resident.formal_owner_inventory_sha256 != null;
    const any = inventory_path != null or inventory_sha256 != null or receipt_path != null or receipt_sha256 != null;
    const all = inventory_path != null and inventory_sha256 != null and receipt_path != null and receipt_sha256 != null;
    if (any != all or formal != all) return error.InvalidQualificationCandidate;
    if (!formal) return false;
    if (!std.fs.path.isAbsolute(inventory_path.?) or !std.fs.path.isAbsolute(receipt_path.?) or
        std.mem.eql(u8, inventory_path.?, receipt_path.?) or
        !std.mem.eql(u8, &resident.formal_owner_inventory_sha256.?, &try parseDigest(inventory_sha256.?)) or
        !std.mem.eql(u8, &resident.formal_context_candidate_receipt_sha256.?, &try parseDigest(receipt_sha256.?)))
        return error.InvalidQualificationCandidate;
    return true;
}

test "formal operational loader rejects partial and caller-mutated file tuples before IO" {
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    var resident: resident_hybrid.Authority = .{
        .configuration_sha256 = @splat(1),
        .source_freeze_sha256 = @splat(2),
        .shape_authority_sha256 = @splat(3),
        .policy_module_sha256 = resident_hybrid.e2_formal_policy_module_sha256,
        .profile_semantic_sha256 = @splat(4),
        .cohort = .e2_8128_v1,
        .e2_fit_source_rows = 7_936,
        .formal_owner_inventory_sha256 = try parseDigest(one),
        .formal_context_candidate_receipt_sha256 = try parseDigest(two),
    };
    try std.testing.expectError(error.InvalidQualificationCandidate, validateFormalFileTuple(resident, "/inventory.json", one, null, null));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateFormalFileTuple(resident, "/inventory.json", two, "/receipt.json", two));
    try std.testing.expect(try validateFormalFileTuple(resident, "/inventory.json", one, "/receipt.json", two));
    resident.formal_owner_inventory_sha256 = null;
    resident.formal_context_candidate_receipt_sha256 = null;
    try std.testing.expect(!(try validateFormalFileTuple(resident, null, null, null, null)));
}

pub const Binding = struct {
    /// Exact bytes of the parent-audited full release receipt. The receipt
    /// parser requires release_eligible=true, serving_qualified=false and the
    /// compiler profile of this executable.
    release_receipt_json: []const u8,
    implementation_semantic_sha256: [32]u8,
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    policy: fit_policy.Policy,
    execution: native_release.Execution,
    qualification_arm: @import("pipelines/generation.zig").NativeCheckpointQualificationArm,
    fit_strategy: @import("runtime/kv/metal_checkpoint_quality.zig").FitStrategy = .legacy,
    training_policy: @import("pipelines/generation.zig").NativeCheckpointTrainingPolicy = .primary_only,
    compiled_prefill_handoff: ?@import("pipelines/generation.zig").NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = null,
    prepared_quality_closure_path: ?[]const u8 = null,
    prepared_quality_source_path: ?[]const u8 = null,
    resident_hybrid_authority: ?resident_hybrid.Authority = null,
};

const FileBinding = struct {
    const Mode = enum { staged_candidate, enrolled_producer, trusted_consumer };
    schema: []const u8,
    mode: Mode,
    release_receipt_path: []const u8,
    producer_release_receipt_path: ?[]const u8 = null,
    profile_path: []const u8,
    profile_sha256: []const u8,
    profile_semantic_sha256: []const u8,
    implementation_semantic_sha256: []const u8,
    policy_name: []const u8,
    execution_name: []const u8,
    qualification_arm: @import("pipelines/generation.zig").NativeCheckpointQualificationArm,
    fit_strategy: @import("runtime/kv/metal_checkpoint_quality.zig").FitStrategy = .legacy,
    training_policy: @import("pipelines/generation.zig").NativeCheckpointTrainingPolicy = .primary_only,
    compiled_prefill_max_copy_bytes: ?usize = null,
    compiled_prefill_max_host_staging_bytes: ?usize = null,
    compiled_prefill_codec_scope: ?@import("pipelines/generation.zig").NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.CodecScope = null,
    compiled_prefill_max_logical_tokens: ?usize = null,
    compiled_prefill_layout_policy: ?@import("pipelines/generation.zig").NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.LayoutPolicy = null,
    prepared_quality_closure_path: ?[]const u8 = null,
    prepared_quality_source_path: ?[]const u8 = null,
    resident_hybrid_configuration_path: ?[]const u8 = null,
    resident_hybrid_configuration_sha256: ?[]const u8 = null,
    resident_hybrid_shape_authority_sha256: ?[]const u8 = null,
    resident_hybrid_policy_module_sha256: ?[]const u8 = null,
    formal_owner_inventory_path: ?[]const u8 = null,
    formal_owner_inventory_sha256: ?[]const u8 = null,
    formal_context_candidate_receipt_path: ?[]const u8 = null,
    formal_context_candidate_receipt_sha256: ?[]const u8 = null,
    base_compatibility_profile_path: ?[]const u8 = null,
    base_compatibility_profile_sha256: ?[]const u8 = null,
    base_compatibility_profile_semantic_sha256: ?[]const u8 = null,
    bounded_encoding: ?checkpoint.Encoding = null,
    bounded_logical_tokens: ?u32 = null,
    bounded_max_output_tokens: ?u32 = null,
    bounded_fit_workers: ?u32 = null,
    bounded_resume_allowed: ?bool = null,
    diagnostic_only: bool = false,
    qualification_credit: bool = false,
    registry_enrollment: bool = false,
};

fn candidateExecution(schema: []const u8, execution_name: []const u8) !native_release.Execution {
    const v6 = std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v6");
    const v7 = std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v7");
    const v8 = std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v8");
    const v9 = std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v9");
    const legacy = std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v1") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v2") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v3") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v4") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v5");
    if ((!v9 and !v8 and !v7 and !v6 and !legacy) or
        ((v7 or v8 or v9) and !std.mem.eql(u8, execution_name, "gemma4_native_coalesced_gated_ffn_owner_v1")) or
        (v6 and !std.mem.eql(u8, execution_name, "gemma4_native_coalesced_owner_v1")) or
        (legacy and !std.mem.eql(u8, execution_name, "gemma4_native_owner_v1")))
        return error.InvalidQualificationCandidate;
    return .{
        .forward_frame = true,
        .precomputed_tail = true,
        .retain_hidden = true,
        .matrix_stage = true,
        .coalesced_stage = v6 or v7 or v8 or v9,
        .resident_gated_ffn_fusion = v7 or v8 or v9,
        .ple_tail = true,
        .resource_barriers = true,
    };
}

fn validateResidentCohortExecution(cohort: resident_profile.Cohort, schema: []const u8, execution: native_release.Execution) !void {
    switch (cohort) {
        .e4_8128_v1 => {},
        .e2_8128_v1 => if (!std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v6") or execution.resident_gated_ffn_fusion)
            return error.InvalidQualificationCandidate,
    }
}

test "E2 resident cohort accepts only non-FFN coalesced V6 execution" {
    const v6 = try candidateExecution("antfly.native-kv-operator-candidate.v6", "gemma4_native_coalesced_owner_v1");
    const v7 = try candidateExecution("antfly.native-kv-operator-candidate.v7", "gemma4_native_coalesced_gated_ffn_owner_v1");
    const v8 = try candidateExecution("antfly.native-kv-operator-candidate.v8", "gemma4_native_coalesced_gated_ffn_owner_v1");
    try validateResidentCohortExecution(.e2_8128_v1, "antfly.native-kv-operator-candidate.v6", v6);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateResidentCohortExecution(.e2_8128_v1, "antfly.native-kv-operator-candidate.v7", v7));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateResidentCohortExecution(.e2_8128_v1, "antfly.native-kv-operator-candidate.v8", v8));
    try validateResidentCohortExecution(.e4_8128_v1, "antfly.native-kv-operator-candidate.v8", v8);
}

test "candidate execution successors bind only their authenticated features" {
    const legacy = try candidateExecution("antfly.native-kv-operator-candidate.v5", "gemma4_native_owner_v1");
    try std.testing.expect(!legacy.coalesced_stage);
    const coalesced = try candidateExecution("antfly.native-kv-operator-candidate.v6", "gemma4_native_coalesced_owner_v1");
    try std.testing.expect(coalesced.coalesced_stage);
    try std.testing.expect(!coalesced.resident_gated_ffn_fusion);
    const fused = try candidateExecution("antfly.native-kv-operator-candidate.v7", "gemma4_native_coalesced_gated_ffn_owner_v1");
    try std.testing.expect(fused.coalesced_stage and fused.resident_gated_ffn_fusion);
    const long_context = try candidateExecution("antfly.native-kv-operator-candidate.v8", "gemma4_native_coalesced_gated_ffn_owner_v1");
    try std.testing.expectEqualDeep(fused, long_context);
    const bounded_v9 = try candidateExecution("antfly.native-kv-operator-candidate.v9", "gemma4_native_coalesced_gated_ffn_owner_v1");
    try std.testing.expectEqualDeep(fused, bounded_v9);
    try std.testing.expect(coalesced.forward_frame and coalesced.precomputed_tail and coalesced.retain_hidden and coalesced.matrix_stage and coalesced.ple_tail and coalesced.resource_barriers);
    try std.testing.expectError(error.InvalidQualificationCandidate, candidateExecution("antfly.native-kv-operator-candidate.v5", "gemma4_native_coalesced_owner_v1"));
    try std.testing.expectError(error.InvalidQualificationCandidate, candidateExecution("antfly.native-kv-operator-candidate.v6", "gemma4_native_owner_v1"));
    try std.testing.expectError(error.InvalidQualificationCandidate, candidateExecution("antfly.native-kv-operator-candidate.v7", "gemma4_native_coalesced_owner_v1"));
    try std.testing.expectError(error.InvalidQualificationCandidate, candidateExecution("antfly.native-kv-operator-candidate.v4", "unknown"));
}

fn validateCandidateV6Envelope(value: FileBinding) !void {
    const v6 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v6");
    const v7 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v7");
    const v8 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v8");
    const v9 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v9");
    if ((!v6 and !v7 and !v8 and !v9) or
        value.mode != .staged_candidate or
        !std.mem.eql(u8, value.policy_name, "cumulative256_exact_cycle_v1") or
        (v6 and !std.mem.eql(u8, value.execution_name, "gemma4_native_coalesced_owner_v1")) or
        ((v7 or v8 or v9) and !std.mem.eql(u8, value.execution_name, "gemma4_native_coalesced_gated_ffn_owner_v1")) or
        value.training_policy != .private_train_challenge_hybrid_resident_v1 or
        value.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or
        value.qualification_arm != .attention_matching or
        !value.diagnostic_only or value.qualification_credit or value.registry_enrollment)
        return error.InvalidQualificationCandidate;
}

test "candidate v6 rejects old schema wrong mode policy and execution" {
    var value: FileBinding = undefined;
    value.schema = "antfly.native-kv-operator-candidate.v6";
    value.mode = .staged_candidate;
    value.policy_name = "cumulative256_exact_cycle_v1";
    value.execution_name = "gemma4_native_coalesced_owner_v1";
    value.training_policy = .private_train_challenge_hybrid_resident_v1;
    value.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
    value.qualification_arm = .attention_matching;
    value.diagnostic_only = true;
    value.qualification_credit = false;
    value.registry_enrollment = false;
    try validateCandidateV6Envelope(value);
    value.schema = "antfly.native-kv-operator-candidate.v7";
    value.execution_name = "gemma4_native_coalesced_gated_ffn_owner_v1";
    try validateCandidateV6Envelope(value);
    value.schema = "antfly.native-kv-operator-candidate.v8";
    try validateCandidateV6Envelope(value);
    value.execution_name = "gemma4_native_coalesced_owner_v1";
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV6Envelope(value));
    value.schema = "antfly.native-kv-operator-candidate.v6";
    value.execution_name = "gemma4_native_coalesced_owner_v1";
    value.schema = "antfly.native-kv-operator-candidate.v5";
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV6Envelope(value));
    value.schema = "antfly.native-kv-operator-candidate.v6";
    value.mode = .enrolled_producer;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV6Envelope(value));
    value.mode = .staged_candidate;
    value.policy_name = "wrong";
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV6Envelope(value));
    value.policy_name = "cumulative256_exact_cycle_v1";
    value.execution_name = "gemma4_native_owner_v1";
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV6Envelope(value));
}

const MatrixBinding = struct {
    schema: []const u8,
    candidate_binding_path: []const u8,
    optimal_compiled_exact_binding_path: []const u8,
    packed_exact_binding_path: []const u8,
};

fn parseDigest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidQualificationCandidate;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, value) catch return error.InvalidQualificationCandidate;
    return result;
}

fn readNoFollow(allocator: std.mem.Allocator, io: std.Io, path: []const u8, maximum: usize) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const stat = try file.stat(io);
    const size = std.math.cast(usize, stat.size) orelse return error.InvalidQualificationCandidate;
    if (stat.kind != .file or size == 0 or size > maximum) return error.InvalidQualificationCandidate;
    const bytes = try allocator.alloc(u8, size);
    errdefer allocator.free(bytes);
    if (try file.readPositionalAll(io, bytes, 0) != size) return error.InvalidQualificationCandidate;
    return bytes;
}

fn validateCompiledPrefillPolicy(
    candidate_v1: bool,
    candidate_v2: bool,
    candidate_v3: bool,
    max_copy_bytes: ?usize,
    max_host_staging_bytes: ?usize,
    codec_scope: ?@import("pipelines/generation.zig").NativeCheckpointPrepare.CompiledPrefillExactKvHandoff.CodecScope,
    max_logical_tokens: ?usize,
) !void {
    if (candidate_v1 and (max_copy_bytes != null or max_host_staging_bytes != null or codec_scope != null or max_logical_tokens != null))
        return error.InvalidQualificationCandidate;
    if (candidate_v2 and (codec_scope != null or max_logical_tokens != null))
        return error.InvalidQualificationCandidate;
    if (candidate_v3 and (max_copy_bytes == null or max_host_staging_bytes == null or
        max_copy_bytes.? == 0 or max_host_staging_bytes.? == 0 or
        codec_scope != .f16_fp8_rotated4 or max_logical_tokens != 8192))
        return error.InvalidQualificationCandidate;
}

test "candidate v3 seals all-codec 8K policy and older schemas stay narrow" {
    try validateCompiledPrefillPolicy(false, false, true, 1, 1, .f16_fp8_rotated4, 8192);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPrefillPolicy(false, false, true, 1, 1, .f16_only, 8192));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPrefillPolicy(false, false, true, 1, 1, .f16_fp8_rotated4, 8193));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPrefillPolicy(false, false, true, 0, 1, .f16_fp8_rotated4, 8192));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPrefillPolicy(false, false, true, 1, 0, .f16_fp8_rotated4, 8192));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPrefillPolicy(false, true, false, 1, 1, .f16_fp8_rotated4, 8192));
    try validateCompiledPrefillPolicy(false, true, false, 1, 1, null, null);
    try validateCompiledPrefillPolicy(true, false, false, null, null, null, null);
}

fn validateCandidateV4Policy(value: FileBinding) !void {
    if (value.compiled_prefill_max_copy_bytes != 4 * 1024 * 1024 * 1024 or
        value.compiled_prefill_max_host_staging_bytes != 256 * 1024 * 1024 or
        value.compiled_prefill_codec_scope != .f16_fp8_rotated4 or
        value.compiled_prefill_max_logical_tokens != 32768 or
        value.compiled_prefill_layout_policy != .full_history_exact)
        return error.InvalidQualificationCandidate;
}

fn validateCandidateV8Policy(value: FileBinding) !void {
    if (value.compiled_prefill_max_copy_bytes != 4 * 1024 * 1024 * 1024 or
        value.compiled_prefill_max_host_staging_bytes != 256 * 1024 * 1024 or
        value.compiled_prefill_codec_scope != .f16_fp8_rotated4 or
        value.compiled_prefill_max_logical_tokens != 32768 or
        value.compiled_prefill_layout_policy != .full_history_exact_8k_split_swa_32k)
        return error.InvalidQualificationCandidate;
}

test "candidate v8 alone authenticates the long-context split layout" {
    var value: FileBinding = undefined;
    value.compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024;
    value.compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024;
    value.compiled_prefill_codec_scope = .f16_fp8_rotated4;
    value.compiled_prefill_max_logical_tokens = 32768;
    value.compiled_prefill_layout_policy = .full_history_exact_8k_split_swa_32k;
    try validateCandidateV8Policy(value);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV4Policy(value));
    value.compiled_prefill_layout_policy = .full_history_exact;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV8Policy(value));
    try validateCandidateV4Policy(value);
}

test "candidate v4 seals caps codec boundary and full-history layout while v3 stays narrow" {
    var value: FileBinding = undefined;
    value.compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024;
    value.compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024;
    value.compiled_prefill_codec_scope = .f16_fp8_rotated4;
    value.compiled_prefill_max_logical_tokens = 32768;
    value.compiled_prefill_layout_policy = .full_history_exact;
    try validateCandidateV4Policy(value);
    value.compiled_prefill_max_copy_bytes.? -= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV4Policy(value));
    value.compiled_prefill_max_copy_bytes.? += 1;
    value.compiled_prefill_layout_policy = .split_swa_permitted;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV4Policy(value));
    value.compiled_prefill_layout_policy = .full_history_exact;
    value.compiled_prefill_codec_scope = .f16_only;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCandidateV4Policy(value));
}

fn isStagedExactBaselineSchema(schema: []const u8) bool {
    return std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v1") or
        std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v2") or
        std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v3") or
        std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v4");
}

fn installCandidateFromConfigBytes(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, config_bytes: []const u8) !void {
    var schema = std.json.parseFromSlice(struct { schema: []const u8 }, allocator, config_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidQualificationCandidate;
    defer schema.deinit();
    if (std.mem.eql(u8, schema.value.schema, "antfly.native-kv-qualification-matrix.v1") or
        isStagedExactBaselineSchema(schema.value.schema))
        return error.InvalidQualificationCandidate;
    var config = try std.json.parseFromSlice(FileBinding, allocator, config_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer config.deinit();
    const value = config.value;
    const candidate_v1 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v1");
    const candidate_v2 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v2");
    const candidate_v3 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v3");
    const candidate_v4 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v4");
    const candidate_v5 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v5");
    const candidate_v6 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v6");
    const candidate_v7 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v7");
    const candidate_v8 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v8");
    const candidate_v9 = std.mem.eql(u8, value.schema, "antfly.native-kv-operator-candidate.v9");
    const candidate_execution = try candidateExecution(value.schema, value.execution_name);
    if (candidate_v6 or candidate_v7 or candidate_v8 or candidate_v9) try validateCandidateV6Envelope(value);
    if ((!candidate_v1 and !candidate_v2 and !candidate_v3 and !candidate_v4 and !candidate_v5 and !candidate_v6 and !candidate_v7 and !candidate_v8 and !candidate_v9) or
        !std.fs.path.isAbsolute(value.release_receipt_path) or !std.fs.path.isAbsolute(value.profile_path) or
        !std.mem.eql(u8, value.policy_name, "cumulative256_exact_cycle_v1"))
        return error.InvalidQualificationCandidate;
    try validateCompiledPrefillPolicy(candidate_v1, candidate_v2, candidate_v3, value.compiled_prefill_max_copy_bytes, value.compiled_prefill_max_host_staging_bytes, value.compiled_prefill_codec_scope, value.compiled_prefill_max_logical_tokens);
    if (candidate_v9) {
        if (value.compiled_prefill_max_copy_bytes != 4 * 1024 * 1024 * 1024 or value.compiled_prefill_max_host_staging_bytes != 256 * 1024 * 1024 or
            value.compiled_prefill_codec_scope != .f16_only or value.compiled_prefill_max_logical_tokens != 8192 or
            value.compiled_prefill_layout_policy != .full_history_exact) return error.InvalidQualificationCandidate;
        if (value.bounded_encoding != .f16 or value.bounded_logical_tokens != 8128 or value.bounded_max_output_tokens != 64 or
            value.bounded_fit_workers != 1 or value.bounded_resume_allowed != false) return error.InvalidQualificationCandidate;
    } else if (candidate_v8) {
        try validateCandidateV8Policy(value);
    } else if (candidate_v4 or candidate_v5 or candidate_v6 or candidate_v7) {
        try validateCandidateV4Policy(value);
    } else if (value.compiled_prefill_layout_policy != null) return error.InvalidQualificationCandidate;
    if ((value.prepared_quality_closure_path == null) != (value.prepared_quality_source_path == null)) return error.InvalidQualificationCandidate;
    const resident_fields_present = value.resident_hybrid_configuration_path != null or value.resident_hybrid_configuration_sha256 != null or value.resident_hybrid_shape_authority_sha256 != null or value.resident_hybrid_policy_module_sha256 != null or value.formal_owner_inventory_path != null or value.formal_owner_inventory_sha256 != null or value.formal_context_candidate_receipt_path != null or value.formal_context_candidate_receipt_sha256 != null or value.base_compatibility_profile_path != null or value.base_compatibility_profile_sha256 != null or value.base_compatibility_profile_semantic_sha256 != null or value.bounded_encoding != null or value.bounded_logical_tokens != null or value.bounded_max_output_tokens != null or value.bounded_fit_workers != null or value.bounded_resume_allowed != null or value.diagnostic_only or value.qualification_credit or value.registry_enrollment;
    if (candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8 or candidate_v9) {
        if (value.mode != .staged_candidate or value.training_policy != .private_train_challenge_hybrid_resident_v1 or
            value.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or value.qualification_arm != .attention_matching or
            value.resident_hybrid_configuration_path == null or !std.fs.path.isAbsolute(value.resident_hybrid_configuration_path.?) or
            value.resident_hybrid_configuration_sha256 == null or value.resident_hybrid_shape_authority_sha256 == null or
            value.resident_hybrid_policy_module_sha256 == null or !value.diagnostic_only or value.qualification_credit or value.registry_enrollment)
            return error.InvalidQualificationCandidate;
        if (candidate_v9 and (value.base_compatibility_profile_path == null or !std.fs.path.isAbsolute(value.base_compatibility_profile_path.?) or
            value.base_compatibility_profile_sha256 == null or value.base_compatibility_profile_semantic_sha256 == null)) return error.InvalidQualificationCandidate;
    } else if (resident_fields_present) return error.InvalidQualificationCandidate;
    if (value.prepared_quality_closure_path) |closure_path|
        if (value.mode != .staged_candidate or !std.fs.path.isAbsolute(closure_path)) return error.InvalidQualificationCandidate;
    if (value.prepared_quality_source_path) |source_path|
        if (value.mode != .staged_candidate or !std.fs.path.isAbsolute(source_path) or
            std.mem.eql(u8, source_path, value.prepared_quality_closure_path.?)) return error.InvalidQualificationCandidate;
    const receipt = try std.Io.Dir.cwd().readFileAlloc(io, value.release_receipt_path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(receipt);
    const profile_bytes = try readNoFollow(allocator, io, value.profile_path, 16 * 1024 * 1024);
    defer allocator.free(profile_bytes);
    var profile_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(profile_bytes, &profile_hash, .{});
    if (!std.mem.eql(u8, &profile_hash, &try parseDigest(value.profile_sha256))) return error.InvalidQualificationCandidate;
    var resident_authority: ?resident_hybrid.Authority = if (candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8) blk: {
        const configuration_bytes = try readNoFollow(allocator, io, value.resident_hybrid_configuration_path.?, 128 * 1024);
        defer allocator.free(configuration_bytes);
        break :blk try resident_hybrid.parse(allocator, configuration_bytes, .{
            .configuration_sha256 = try parseDigest(value.resident_hybrid_configuration_sha256.?),
            .implementation_semantic_sha256 = try parseDigest(value.implementation_semantic_sha256),
            .shape_authority_sha256 = try parseDigest(value.resident_hybrid_shape_authority_sha256.?),
            .policy_module_sha256 = try parseDigest(value.resident_hybrid_policy_module_sha256.?),
            .profile_semantic_sha256 = try parseDigest(value.profile_semantic_sha256),
        });
    } else null;
    if (resident_authority) |resident| {
        if (try validateFormalFileTuple(resident, value.formal_owner_inventory_path, value.formal_owner_inventory_sha256, value.formal_context_candidate_receipt_path, value.formal_context_candidate_receipt_sha256)) {
            resident_authority = try loadFormalResidentAuthority(allocator, io, resident, receipt, value);
        }
    }
    var bounded_candidate: ?bounded_resident_profile.Loaded = if (candidate_v9) try bounded_resident_profile.loadBytes(allocator, io, profile_bytes) else null;
    defer if (bounded_candidate) |*loaded| loaded.deinit();
    if (bounded_candidate) |loaded| {
        const sealed = loaded.profile.profile.resident_authority orelse return error.InvalidQualificationCandidate;
        const artifact = loaded.profile.parsed.value;
        if (!std.mem.eql(u8, value.base_compatibility_profile_path.?, artifact.base_compatibility_profile_path.?) or
            !std.mem.eql(u8, &try parseDigest(value.base_compatibility_profile_sha256.?), &sealed.base_compatibility_artifact_sha256) or
            !std.mem.eql(u8, &try parseDigest(value.base_compatibility_profile_semantic_sha256.?), &sealed.base_compatibility_semantic_sha256) or
            !std.mem.eql(u8, value.resident_hybrid_configuration_path.?, artifact.resident_configuration_path.?) or
            !std.mem.eql(u8, &try parseDigest(value.resident_hybrid_configuration_sha256.?), &sealed.configuration_sha256) or
            !std.mem.eql(u8, &try parseDigest(value.resident_hybrid_shape_authority_sha256.?), &sealed.shape_authority_sha256) or
            !std.mem.eql(u8, &try parseDigest(value.resident_hybrid_policy_module_sha256.?), &sealed.policy_module_sha256) or
            value.bounded_encoding != .f16 or value.bounded_logical_tokens != sealed.logical_tokens or
            value.bounded_max_output_tokens != sealed.max_output_tokens or value.bounded_fit_workers != sealed.fit_workers or
            value.bounded_resume_allowed != sealed.resume_allowed or value.compiled_prefill_max_copy_bytes != @as(usize, sealed.compiled_max_copy_bytes) or
            value.compiled_prefill_max_host_staging_bytes != @as(usize, sealed.compiled_max_host_staging_bytes) or
            value.compiled_prefill_max_logical_tokens != @as(usize, sealed.compiled_max_logical_tokens)) return error.InvalidQualificationCandidate;
    }
    // Parse the exact retained bytes whose digest was verified. Do not reopen
    // the pathname: pre/post pathname hashes cannot exclude an A->B->A swap
    // while a shared loader performs its own open.
    var resident_candidate: ?resident_profile.Loaded = if (candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8) try resident_profile.parseBytes(allocator, profile_bytes) else null;
    defer if (resident_candidate) |*loaded| loaded.deinit();
    if (resident_candidate) |loaded| try validateResidentCohortExecution(loaded.cohort, value.schema, candidate_execution);
    var candidate: ?candidate_loader.Candidate = if (candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8 or candidate_v9) null else try candidate_loader.loadCandidateBytes(allocator, profile_bytes);
    defer if (candidate) |*loaded| loaded.deinit(allocator);
    const private_dynamic_budget = candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8 or candidate_v9 or candidate.?.isPrivateDense();
    if (candidate_v9) {
        // V9's declared training policy is sealed by P and already joined to
        // B/C above. It intentionally has no generic Candidate value.
        if (bounded_candidate.?.profile.parsed.value.training_policy.len == 0 or
            !std.mem.eql(u8, bounded_candidate.?.profile.parsed.value.training_policy, "private_train_challenge_hybrid_resident_v1"))
            return error.InvalidQualificationCandidate;
    } else if (candidate_v5 or candidate_v6 or candidate_v7 or candidate_v8) {
        const authority = resident_authority.?;
        // The receipt source is joined again after its authenticated parser
        // runs; this early join prevents profile/config disagreement.
        if (!std.mem.eql(u8, &resident_candidate.?.profile_semantic_sha256, &authority.profile_semantic_sha256) or
            !std.mem.eql(u8, &resident_candidate.?.source_freeze_sha256, &authority.source_freeze_sha256) or
            @backingInt(resident_candidate.?.cohort) != @backingInt(authority.cohort)) return error.InvalidQualificationCandidate;
    } else if (private_dynamic_budget) {
        const declared = switch (candidate.?) {
            .private_dense => |loaded| loaded.parsed.value.training_policy,
            else => unreachable,
        };
        const expected = switch (value.training_policy) {
            .private_dense512_reference_v1 => "private_dense512_reference_v1",
            .private_future_decode_mix_v1 => "private_future_decode_mix_v1",
            .private_train_challenge_hybrid_resident_v1 => "private_train_challenge_hybrid_resident_v1",
            else => return error.InvalidQualificationCandidate,
        };
        if (!std.mem.eql(u8, declared, expected)) return error.InvalidQualificationCandidate;
    }
    // C binds the immutable base B. After the complete B/C/P join, this
    // installation projection selects P and therefore carries P's semantic.
    const v9_authority: ?resident_hybrid.Authority = if (bounded_candidate) |loaded| .{ .configuration_sha256 = loaded.authority.configuration_sha256, .source_freeze_sha256 = loaded.authority.source_freeze_sha256, .shape_authority_sha256 = loaded.authority.shape_authority_sha256, .policy_module_sha256 = loaded.authority.policy_module_sha256, .profile_semantic_sha256 = loaded.profile.profile.semanticDigest(), .cohort = .e4_8128_v1 } else null;
    const compiled_prefill_handoff: ?@import("pipelines/generation.zig").NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = if (value.compiled_prefill_max_copy_bytes) |device| .{
        .max_copy_bytes = device,
        .max_host_staging_bytes = value.compiled_prefill_max_host_staging_bytes orelse return error.InvalidQualificationCandidate,
        .codec_scope = value.compiled_prefill_codec_scope orelse .f16_only,
        .max_logical_tokens = value.compiled_prefill_max_logical_tokens orelse std.math.maxInt(usize),
        .layout_policy = value.compiled_prefill_layout_policy orelse .split_swa_permitted,
    } else if (value.compiled_prefill_max_host_staging_bytes != null)
        return error.InvalidQualificationCandidate
    else
        null;
    if (compiled_prefill_handoff) |handoff| {
        if (handoff.max_copy_bytes > 4 * 1024 * 1024 * 1024 or handoff.max_host_staging_bytes > 256 * 1024 * 1024 or
            (candidate_v3 and handoff.max_logical_tokens != 8192) or
            ((candidate_v4 or candidate_v5 or candidate_v6 or candidate_v7) and (handoff.max_logical_tokens != 32768 or handoff.layout_policy != .full_history_exact)) or
            (candidate_v8 and (handoff.max_logical_tokens != 32768 or handoff.layout_policy != .full_history_exact_8k_split_swa_32k)) or
            (candidate_v9 and (handoff.codec_scope != .f16_only or handoff.max_logical_tokens != 8192 or handoff.layout_policy != .full_history_exact)))
            return error.InvalidQualificationCandidate;
    }
    var profile = if (bounded_candidate) |*loaded| loaded.profile.profile else if (resident_candidate) |*loaded| loaded.profile else candidate.?.profile().*;
    profile.producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1;
    const binding: Binding = .{
        .release_receipt_json = receipt,
        .implementation_semantic_sha256 = try parseDigest(value.implementation_semantic_sha256),
        .profile_sha256 = profile_hash,
        .profile_semantic_sha256 = try parseDigest(value.profile_semantic_sha256),
        .policy = .cumulative256_exact_cycle_v1,
        .execution = candidate_execution,
        .qualification_arm = value.qualification_arm,
        .fit_strategy = value.fit_strategy,
        .training_policy = value.training_policy,
        .compiled_prefill_handoff = compiled_prefill_handoff,
        .prepared_quality_closure_path = value.prepared_quality_closure_path,
        .prepared_quality_source_path = value.prepared_quality_source_path,
        .resident_hybrid_authority = v9_authority orelse resident_authority,
    };
    if (value.mode == .staged_candidate) {
        if (private_dynamic_budget and (value.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or
            (value.training_policy != .private_dense512_reference_v1 and value.training_policy != .private_future_decode_mix_v1 and
                value.training_policy != .private_train_challenge_hybrid_resident_v1) or
            value.qualification_arm != .attention_matching))
            return error.InvalidQualificationCandidate;
        try installStagedFullCli(allocator, io, node, profile, binding, private_dynamic_budget);
    } else if (value.mode == .enrolled_producer) {
        if (private_dynamic_budget) return error.InvalidQualificationCandidate;
        if (value.fit_strategy != .legacy or value.training_policy != .primary_only) return error.InvalidQualificationCandidate;
        const trusted = try verifiedAuthority(allocator, io, profile, binding);
        try node.installNativeKvProducerAuthority(trusted);
    } else {
        if (private_dynamic_budget) return error.InvalidQualificationCandidate;
        if (value.fit_strategy != .legacy or value.training_policy != .primary_only) return error.InvalidQualificationCandidate;
        const producer_receipt_path = value.producer_release_receipt_path orelse return error.InvalidQualificationCandidate;
        if (!std.fs.path.isAbsolute(producer_receipt_path)) return error.InvalidQualificationCandidate;
        // Verify this receiver against its own receipt first; producer trust is
        // then derived from a distinct parent-audited receipt.
        _ = try verifiedAuthority(allocator, io, profile, binding);
        const producer_receipt = try std.Io.Dir.cwd().readFileAlloc(io, producer_receipt_path, allocator, .limited(4 * 1024 * 1024));
        defer allocator.free(producer_receipt);
        const producer_release = try release_build.parse(allocator, producer_receipt);
        try node.installNativeKvTrustedProducer(.{
            .binary_sha256 = producer_release.binary_sha256,
            .implementation_semantic_sha256 = binding.implementation_semantic_sha256,
            .identity = .{
                .checkpoint = .{ .weights = profile.identity.artifact, .tokenizer = profile.identity.tokenizer, .template = profile.identity.template, .layout = profile.identity.layout },
                .profile_sha256 = binding.profile_sha256,
                .profile_semantic_sha256 = binding.profile_semantic_sha256,
            },
            .policy = binding.policy,
        });
    }
}

test "resident compatibility binding joins config profile and receipt" {
    const semantic: [32]u8 = @splat(1);
    const source: [32]u8 = @splat(2);
    const authority: resident_hybrid.Authority = .{
        .configuration_sha256 = @splat(3),
        .source_freeze_sha256 = source,
        .shape_authority_sha256 = @splat(4),
        .policy_module_sha256 = @splat(5),
        .profile_semantic_sha256 = semantic,
    };
    try validateResidentCompatibilityBinding(semantic, source, authority, source);
    var wrong = source;
    wrong[0] ^= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateResidentCompatibilityBinding(semantic, wrong, authority, source));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateResidentCompatibilityBinding(semantic, source, authority, wrong));
    var wrong_semantic = semantic;
    wrong_semantic[0] ^= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateResidentCompatibilityBinding(wrong_semantic, source, authority, source));
}

fn isMatrixCandidateSchema(schema: []const u8) bool {
    return std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v3") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v4") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v5") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v6") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v7") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v8") or
        std.mem.eql(u8, schema, "antfly.native-kv-operator-candidate.v9");
}

test "context qualification matrix routes gated FFN V7 through child authority loading" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = "{}" });
    const candidate_path = try tmp.dir.realPathFileAlloc(io, "candidate.json", allocator);
    defer allocator.free(candidate_path);
    const parent = std.fs.path.dirname(candidate_path).?;
    const optimal_path = try std.fs.path.join(allocator, &.{ parent, "missing-optimal.json" });
    defer allocator.free(optimal_path);
    const packed_path = try std.fs.path.join(allocator, &.{ parent, "missing-packed.json" });
    defer allocator.free(packed_path);
    const matrix = try std.json.Stringify.valueAlloc(allocator, MatrixBinding{
        .schema = "antfly.native-kv-qualification-matrix.v1",
        .candidate_binding_path = candidate_path,
        .optimal_compiled_exact_binding_path = optimal_path,
        .packed_exact_binding_path = packed_path,
    }, .{});
    defer allocator.free(matrix);
    // The missing baseline must still fail before any Node authority mutation.
    // V7 must reach that read, while unsupported candidate versions must reject
    // before loading baseline authorities.
    var unused_node: server.Node = undefined;
    const Case = struct { schema: []const u8, expected: anyerror };
    for ([_]Case{
        .{ .schema = "antfly.native-kv-operator-candidate.v6", .expected = error.FileNotFound },
        .{ .schema = "antfly.native-kv-operator-candidate.v7", .expected = error.FileNotFound },
        .{ .schema = "antfly.native-kv-operator-candidate.v8", .expected = error.FileNotFound },
        .{ .schema = "antfly.native-kv-staged-exact-baseline.v3", .expected = error.InvalidQualificationCandidate },
    }) |case| {
        const candidate = try std.json.Stringify.valueAlloc(allocator, .{ .schema = case.schema }, .{});
        defer allocator.free(candidate);
        try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = candidate });
        try std.testing.expectError(case.expected, installMatrixFromConfigBytes(allocator, io, &unused_node, matrix));
    }
}

fn installMatrixFromConfigBytes(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, bytes: []const u8) !void {
    var parsed = std.json.parseFromSlice(MatrixBinding, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidQualificationCandidate;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.native-kv-qualification-matrix.v1") or
        !std.fs.path.isAbsolute(value.candidate_binding_path) or
        !std.fs.path.isAbsolute(value.optimal_compiled_exact_binding_path) or
        !std.fs.path.isAbsolute(value.packed_exact_binding_path) or
        std.mem.eql(u8, value.candidate_binding_path, value.optimal_compiled_exact_binding_path) or
        std.mem.eql(u8, value.candidate_binding_path, value.packed_exact_binding_path) or
        std.mem.eql(u8, value.optimal_compiled_exact_binding_path, value.packed_exact_binding_path))
        return error.InvalidQualificationCandidate;
    const candidate_bytes = try readNoFollow(allocator, io, value.candidate_binding_path, 64 * 1024);
    defer allocator.free(candidate_bytes);
    var candidate_schema = std.json.parseFromSlice(struct { schema: []const u8 }, allocator, candidate_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidQualificationCandidate;
    defer candidate_schema.deinit();
    if (!isMatrixCandidateSchema(candidate_schema.value.schema))
        return error.InvalidQualificationCandidate;
    const optimal_bytes = try readNoFollow(allocator, io, value.optimal_compiled_exact_binding_path, 64 * 1024);
    defer allocator.free(optimal_bytes);
    const packed_bytes = try readNoFollow(allocator, io, value.packed_exact_binding_path, 64 * 1024);
    defer allocator.free(packed_bytes);
    const optimal = try staged_exact_baseline.loadFromConfigBytes(allocator, io, optimal_bytes);
    const packed_loaded = try staged_exact_baseline.loadFromConfigBytes(allocator, io, packed_bytes);
    if (optimal.arm != .optimal_compiled_exact or packed_loaded.arm != .packed_exact)
        return error.InvalidQualificationCandidate;
    try installCandidateFromConfigBytes(allocator, io, node, candidate_bytes);
    try node.installNativeKvQualificationMatrix(optimal.authority, packed_loaded.authority, packed_loaded.compiled_packed_prefill_bridge);
}

pub fn installFromConfigFile(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, path: []const u8) !void {
    return installConfigFile(allocator, io, node, path, null);
}

pub fn installFromConfigFileWithDigest(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, path: []const u8, expected: [32]u8) !void {
    return installConfigFile(allocator, io, node, path, expected);
}

fn installConfigFile(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, path: []const u8, expected: ?[32]u8) !void {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualificationCandidate;
    const config_bytes = try readNoFollow(allocator, io, path, 64 * 1024);
    defer allocator.free(config_bytes);
    if (expected) |digest| {
        var actual: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(config_bytes, &actual, .{});
        if (!std.crypto.timing_safe.eql([32]u8, actual, digest)) return error.InvalidQualificationCandidate;
    }
    var schema = std.json.parseFromSlice(struct { schema: []const u8 }, allocator, config_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidQualificationCandidate;
    defer schema.deinit();
    if (std.mem.eql(u8, schema.value.schema, "antfly.native-kv-qualification-matrix.v1"))
        return installMatrixFromConfigBytes(allocator, io, node, config_bytes);
    if (isStagedExactBaselineSchema(schema.value.schema))
        return staged_exact_baseline.installFromConfigBytes(allocator, io, node, config_bytes);
    return installCandidateFromConfigBytes(allocator, io, node, config_bytes);
}

test "staged exact baseline router recognizes bridge authority v4" {
    try std.testing.expect(isStagedExactBaselineSchema("antfly.native-kv-staged-exact-baseline.v1"));
    try std.testing.expect(isStagedExactBaselineSchema("antfly.native-kv-staged-exact-baseline.v2"));
    try std.testing.expect(isStagedExactBaselineSchema("antfly.native-kv-staged-exact-baseline.v3"));
    try std.testing.expect(isStagedExactBaselineSchema("antfly.native-kv-staged-exact-baseline.v4"));
    try std.testing.expect(!isStagedExactBaselineSchema("antfly.native-kv-staged-exact-baseline.v5"));
    try std.testing.expect(!isStagedExactBaselineSchema("antfly.native-kv-operator-candidate.v6"));
}

pub fn install(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, profile: profile_mod.Profile, binding: Binding) !void {
    return installWithReceipt(allocator, io, node, profile, binding, .release);
}

const ReceiptAuthority = enum { release, staged_full_cli };

fn installStagedFullCli(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, profile: profile_mod.Profile, binding: Binding, private_dynamic_budget: bool) !void {
    const trusted = try verifiedAuthorityWithReceipt(allocator, io, profile, binding, .staged_full_cli);
    try validateStagedPolicy(binding);
    // V9's f16-only bounded resident envelope is installed as one validated
    // transaction.  In particular, do not publish resident authority before
    // model-manager configuration and prepared-quality ownership succeed.
    if (binding.compiled_prefill_handoff != null and binding.compiled_prefill_handoff.?.codec_scope == .f16_only and
        binding.resident_hybrid_authority != null and binding.resident_hybrid_authority.?.cohort == .e4_8128_v1)
    {
        const resident = binding.resident_hybrid_authority orelse return error.InvalidQualificationCandidate;
        return node.installNativeKvBoundedResidentQualificationCandidateAtomic(
            trusted,
            profile,
            binding.qualification_arm,
            binding.fit_strategy,
            binding.training_policy,
            private_dynamic_budget,
            binding.compiled_prefill_handoff.?,
            .{
                .configuration_sha256 = resident.configuration_sha256,
                .source_freeze_sha256 = resident.source_freeze_sha256,
                .shape_authority_sha256 = resident.shape_authority_sha256,
                .policy_module_sha256 = resident.policy_module_sha256,
                .e2_guarded_development_trial = resident.guarded_development_trial,
                .e2_fit_source_rows = resident.e2_fit_source_rows,
                .e2_formal_owner_inventory_sha256 = resident.formal_owner_inventory_sha256 orelse @splat(0),
                .e2_formal_owner_inventory = resident.formal_owner_inventory,
                .cohort = .e4_8128_v1,
            },
            binding.prepared_quality_closure_path,
            binding.prepared_quality_source_path,
        );
    }
    if (binding.resident_hybrid_authority) |resident|
        try node.installNativeKvResidentHybridStagedAuthority(.{
            .configuration_sha256 = resident.configuration_sha256,
            .source_freeze_sha256 = resident.source_freeze_sha256,
            .shape_authority_sha256 = resident.shape_authority_sha256,
            .policy_module_sha256 = resident.policy_module_sha256,
            .e2_guarded_development_trial = resident.guarded_development_trial,
            .e2_fit_source_rows = resident.e2_fit_source_rows,
            .e2_formal_owner_inventory_sha256 = resident.formal_owner_inventory_sha256 orelse @splat(0),
            .e2_formal_owner_inventory = resident.formal_owner_inventory,
            .cohort = switch (resident.cohort) {
                .e4_8128_v1 => .e4_8128_v1,
                .e2_8128_v1 => .e2_8128_v1,
            },
        });
    if (binding.prepared_quality_closure_path) |closure_path|
        try node.installNativeKvQualificationCandidateWithPreparedQuality(trusted, profile, binding.qualification_arm, binding.fit_strategy, binding.training_policy, private_dynamic_budget, binding.compiled_prefill_handoff, closure_path, binding.prepared_quality_source_path.?)
    else
        try node.installNativeKvQualificationCandidate(trusted, profile, binding.qualification_arm, binding.fit_strategy, binding.training_policy, private_dynamic_budget, binding.compiled_prefill_handoff);
}

fn installWithReceipt(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, profile: profile_mod.Profile, binding: Binding, receipt_authority: ReceiptAuthority) !void {
    const trusted = try verifiedAuthorityWithReceipt(allocator, io, profile, binding, receipt_authority);
    try validateStagedPolicy(binding);
    try node.installNativeKvQualificationCandidate(trusted, profile, binding.qualification_arm, binding.fit_strategy, binding.training_policy, false, null);
}

fn validateStagedPolicy(binding: Binding) !void {
    const resident = binding.training_policy == .private_train_challenge_hybrid_resident_v1;
    if (resident != (binding.resident_hybrid_authority != null)) return error.InvalidQualificationCandidate;
    if (resident and (binding.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or binding.qualification_arm != .attention_matching or binding.compiled_prefill_handoff == null)) return error.InvalidQualificationCandidate;
    if (binding.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and binding.qualification_arm != .attention_matching) return error.InvalidQualificationCandidate;
    if (generation.isPrivateAdaptiveTraining(binding.training_policy) and
        (binding.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or binding.qualification_arm != .attention_matching))
        return error.InvalidQualificationCandidate;
    if (binding.compiled_prefill_handoff != null and
        (binding.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !generation.isPrivateAdaptiveTraining(binding.training_policy) or
            binding.qualification_arm != .attention_matching))
        return error.InvalidQualificationCandidate;
    if (binding.training_policy == .private_future_decode_mix_v1 and binding.compiled_prefill_handoff == null)
        return error.InvalidQualificationCandidate;
}

test "future staged policy requires and preserves authenticated compiled handoff" {
    var binding: Binding = undefined;
    binding.qualification_arm = .attention_matching;
    binding.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
    binding.training_policy = .private_future_decode_mix_v1;
    binding.resident_hybrid_authority = null;
    binding.compiled_prefill_handoff = null;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateStagedPolicy(binding));
    binding.compiled_prefill_handoff = .{ .max_copy_bytes = 1, .max_host_staging_bytes = 1, .codec_scope = .f16_fp8_rotated4, .max_logical_tokens = 8192 };
    try validateStagedPolicy(binding);
}

fn verifiedAuthority(allocator: std.mem.Allocator, io: std.Io, profile: profile_mod.Profile, binding: Binding) !producer_authority.Authority {
    return verifiedAuthorityWithReceipt(allocator, io, profile, binding, .release);
}

fn verifiedAuthorityWithReceipt(allocator: std.mem.Allocator, io: std.Io, profile: profile_mod.Profile, binding: Binding, receipt_authority: ReceiptAuthority) !producer_authority.Authority {
    const artifact_sha = try profile_mod.profileArtifactDigest(profile);
    const semantic_sha = profile.semanticDigest();
    const release = switch (receipt_authority) {
        .release => try release_build.parse(allocator, binding.release_receipt_json),
        .staged_full_cli => try parseStagedReceiptForInstall(allocator, io, binding.release_receipt_json),
    };
    if (binding.resident_hybrid_authority) |resident| {
        if (receipt_authority != .staged_full_cli or release.source_freeze_sha256 == null) return error.InvalidQualificationCandidate;
        try validateResidentCompatibilityBinding(semantic_sha, resident.source_freeze_sha256, resident, release.source_freeze_sha256.?);
    }
    if (!std.mem.eql(u8, &artifact_sha, &binding.profile_sha256) or
        !std.mem.eql(u8, &semantic_sha, &binding.profile_semantic_sha256) or
        !std.meta.eql(profile.producer_policy, binding.policy))
        return error.InvalidQualificationCandidate;
    return producer_authority.Authority.initVerified(
        allocator,
        io,
        release.binary_sha256,
        binding.implementation_semantic_sha256,
        .{
            .checkpoint = checkpoint.Identity{
                .weights = profile.identity.artifact,
                .tokenizer = profile.identity.tokenizer,
                .template = profile.identity.template,
                .layout = profile.identity.layout,
            },
            .profile_sha256 = artifact_sha,
            .profile_semantic_sha256 = semantic_sha,
        },
        binding.policy,
        binding.execution,
    );
}

test "native KV operator config loader fails closed before candidate installation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "binding.json", .data = "{\"schema\":\"wrong\"}" });
    const path = try tmp.dir.realPathFileAlloc(io, "binding.json", allocator);
    defer allocator.free(path);
    var unused_node: server.Node = undefined;
    try std.testing.expectError(error.MissingField, installFromConfigFile(allocator, io, &unused_node, path));
    try std.testing.expectError(error.InvalidQualificationCandidate, installFromConfigFileWithDigest(allocator, io, &unused_node, path, @splat(0)));
    var expected: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("{\"schema\":\"wrong\"}", &expected, .{});
    try std.testing.expectError(error.MissingField, installFromConfigFileWithDigest(allocator, io, &unused_node, path, expected));
}

fn testExecutableDigest(allocator: std.mem.Allocator, io: std.Io) !struct { digest: [32]u8, bytes: u64 } {
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
        if (count == 0) return error.InvalidNativeKvProducerBinary;
        hash.update(buffer[0..count]);
        offset += count;
    }
    return .{ .digest = hash.finalResult(), .bytes = stat.size };
}

test "CandidateV9 config file installs the exact bounded B C P authority" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    const b_path = try std.fs.path.join(allocator, &.{ root, "B.json" });
    defer allocator.free(b_path);
    const c_path = try std.fs.path.join(allocator, &.{ root, "C.json" });
    defer allocator.free(c_path);
    const p_path = try std.fs.path.join(allocator, &.{ root, "P.json" });
    defer allocator.free(p_path);
    const receipt_path = try std.fs.path.join(allocator, &.{ root, "receipt.json" });
    defer allocator.free(receipt_path);
    const candidate_path = try std.fs.path.join(allocator, &.{ root, "candidate.json" });
    defer allocator.free(candidate_path);

    const source: [32]u8 = @splat(7);
    const base = try bounded_resident_profile.testBaseJson(allocator, source);
    defer allocator.free(base);
    var base_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(base, &base_sha, .{});
    const base_semantic = resident_profile.expectedProfileSemantic();
    const configuration = try bounded_resident_profile.testConfigJson(allocator, source, base_semantic);
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
    var loaded = try bounded_resident_profile.loadBytes(allocator, io, profile_bytes);
    defer loaded.deinit();
    const profile_semantic = loaded.profile.profile.semanticDigest();

    const executable = try testExecutableDigest(allocator, io);
    const receipt = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.full-inference-cli-build.v1",
        .test_fixture_authority = "antfly.context.test-only-local-authority.v1",
        .source_freeze = .{ .sha256 = std.fmt.bytesToHex(source, .lower) },
        .binary = .{ .sha256 = std.fmt.bytesToHex(executable.digest, .lower), .bytes = executable.bytes },
    }, .{});
    defer allocator.free(receipt);
    try tmp.dir.writeFile(io, .{ .sub_path = "receipt.json", .data = receipt });

    const candidate: FileBinding = .{
        .schema = "antfly.native-kv-operator-candidate.v9",
        .mode = .staged_candidate,
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
        .resident_hybrid_configuration_path = c_path,
        .resident_hybrid_configuration_sha256 = &std.fmt.bytesToHex(configuration_sha, .lower),
        .resident_hybrid_shape_authority_sha256 = &std.fmt.bytesToHex(base_semantic, .lower),
        .resident_hybrid_policy_module_sha256 = &std.fmt.bytesToHex(@import("context_resident_hybrid_staged_configuration.zig").policy_module_sha256, .lower),
        .base_compatibility_profile_path = b_path,
        .base_compatibility_profile_sha256 = &std.fmt.bytesToHex(base_sha, .lower),
        .base_compatibility_profile_semantic_sha256 = &std.fmt.bytesToHex(base_semantic, .lower),
        .bounded_encoding = .f16,
        .bounded_logical_tokens = 8128,
        .bounded_max_output_tokens = 64,
        .bounded_fit_workers = 1,
        .bounded_resume_allowed = false,
        .diagnostic_only = true,
    };
    const candidate_bytes = try std.json.Stringify.valueAlloc(allocator, candidate, .{});
    defer allocator.free(candidate_bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = candidate_bytes });

    var tampered = try allocator.dupe(u8, candidate_bytes);
    defer allocator.free(tampered);
    const tuple = "\"bounded_logical_tokens\":8128";
    const tuple_at = std.mem.indexOf(u8, tampered, tuple).?;
    tampered[tuple_at + tuple.len - 1] = '7';
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = tampered });
    var rejected = try server.Node.init(allocator, .{ .max_concurrent_requests = 1 });
    defer rejected.deinit();
    try std.testing.expectError(error.InvalidQualificationCandidate, installFromConfigFile(allocator, io, &rejected, candidate_path));
    try std.testing.expect(rejected.native_kv_operator_qualification_authority == null);
    try std.testing.expect(rejected.native_kv_resident_hybrid_authority == null);
    try std.testing.expect(rejected.model_manager.qualification_candidate == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "candidate.json", .data = candidate_bytes });

    var node = try server.Node.init(allocator, .{ .max_concurrent_requests = 1 });
    defer node.deinit();
    try installFromConfigFile(allocator, io, &node, candidate_path);
    try std.testing.expect(node.native_kv_operator_qualification_authority != null);
    try std.testing.expect(node.native_kv_resident_hybrid_authority != null);
    try std.testing.expect(node.model_manager.qualification_candidate != null);
    try std.testing.expectEqualSlices(u8, &profile_semantic, &node.native_kv_operator_qualification_authority.?.identity.profile_semantic_sha256);
    try std.testing.expectEqual(.f16_only, node.native_kv_operator_compiled_prefill_handoff.?.codec_scope);
    try std.testing.expectEqual(@as(usize, 8192), node.native_kv_operator_compiled_prefill_handoff.?.max_logical_tokens);
    try std.testing.expectEqualSlices(u8, &configuration_sha, &node.native_kv_resident_hybrid_authority.?.configuration_sha256);
}

test "context resident v5 startup ingress is semantically compiled" {
    std.mem.doNotOptimizeAway(&installCandidateFromConfigBytes);
}
