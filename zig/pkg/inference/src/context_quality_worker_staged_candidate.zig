// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test-binary-only staged-private candidate installation for the maintained
//! quality worker. The worker and full CLI deliberately have distinct binary
//! identities. Cross-binary source provenance remains a parent-audited join;
//! this module authenticates both receipts and only trusts the running worker.
const std = @import("std");
const builtin = @import("builtin");
const server = @import("server/server.zig");
const authority = @import("server/native_kv_producer_authority.zig");
const release_build = @import("context_release_build_binding.zig");
const profile_mod = @import("runtime/kv/compaction_profile.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const generation = @import("pipelines/generation.zig");
const quality = @import("runtime/kv/metal_checkpoint_quality.zig");

pub const InstallEvidence = struct {
    worker_binary_sha256: [32]u8,
    full_cli_binary_sha256: [32]u8,
    profile_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    authority_implementation_sha256: [32]u8,
    execution: @import("runtime/kv/attention_match_release.zig").Execution,
    cross_binary_source_join_verified: bool = false,
};
const FullCliProofJoin = struct { full_context_release_proof: struct { sha256: []const u8 } };

const WorkerV4Receipt = struct {
    schema: []const u8,
    evidence_scope: []const u8,
    release_eligible: bool,
    serving_qualified: bool,
    source_verified_after_build: bool,
    compiler_profile: struct { strip_debug_info: bool, have_error_return_tracing: bool },
    binary: struct { path: []const u8, sha256: []const u8, bytes: u64 },
    resource: struct { path: []const u8, sha256: []const u8 },
    tests: struct { failed: usize, leaked: usize, passed: usize, selected: usize, skipped: usize },
    build: struct { target: []const u8, metal: bool, cuda: bool, optimize: []const u8 },
};

const FocusedDiagnosticReceipt = struct {
    schema: []const u8,
    diagnostic_only: bool,
    release_eligible: bool,
    qualification_credit: bool,
    source_verified_after_build: bool,
    binary: struct { path: []const u8, sha256: []const u8, bytes: u64 },
    source_freeze: struct { path: []const u8, sha256: []const u8 },
    source_archive: struct { path: []const u8, sha256: []const u8, entries: usize },
    build_helper: struct { path: []const u8, sha256: []const u8 },
    source351_manifest_sha256: []const u8,
    tests: struct { failed: usize, leaked: usize, selected: usize, passed: usize, skipped: usize },
    bootstrap_tests: []const []const u8,
    build: struct { target: []const u8, metal: bool, cuda: bool, optimize: []const u8, filter: []const u8 },
};

const focused_build_helper_sha256 = "5b2e57f6e2e2534acc782217cc06f2388bf70b171f4deb03eaf014469af16aa0";

fn verifyDiagnosticFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected_hex: []const u8, limit: usize) !void {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualityWorkerAuthority;
    const expected = try parseDigest(expected_hex);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(limit)) catch return error.InvalidQualityWorkerAuthority;
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidQualityWorkerAuthority;
}

fn parseFocusedDiagnostic(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !release_build.Binding {
    var parsed = std.json.parseFromSlice(FocusedDiagnosticReceipt, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidQualityWorkerAuthority;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.focused-future-probe-diagnostic-worker.v1") or
        !value.diagnostic_only or value.release_eligible or value.qualification_credit or !value.source_verified_after_build or
        value.binary.bytes == 0 or !std.fs.path.isAbsolute(value.binary.path) or value.source_archive.entries == 0 or
        value.tests.failed != 0 or value.tests.leaked != 0 or value.tests.selected != 1 or value.tests.passed != 0 or
        value.tests.skipped != 1 or value.bootstrap_tests.len != 0 or
        !std.mem.eql(u8, value.build.target, "inference-test-context") or !value.build.metal or value.build.cuda or
        !std.mem.eql(u8, value.build.optimize, "ReleaseFast") or
        !std.mem.eql(u8, value.build.filter, "context Metal focused retained E4 future probe diagnostic") or
        !std.mem.eql(u8, value.build_helper.sha256, focused_build_helper_sha256) or
        !std.mem.eql(u8, value.source351_manifest_sha256, "2bf9f1d8ceaa46c798826061dfbe024927bfaecada22a54a8d3718fe8f1ed773"))
        return error.InvalidQualityWorkerAuthority;
    try verifyDiagnosticFile(allocator, io, value.source_freeze.path, value.source_freeze.sha256, 16 * 1024 * 1024);
    try verifyDiagnosticFile(allocator, io, value.source_archive.path, value.source_archive.sha256, 1024 * 1024 * 1024);
    try verifyDiagnosticFile(allocator, io, value.build_helper.path, value.build_helper.sha256, 4 * 1024 * 1024);
    return .{ .binary_sha256 = try parseDigest(value.binary.sha256) };
}

fn parseDigest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidQualityWorkerAuthority;
    var digest: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&digest, value) catch return error.InvalidQualityWorkerAuthority;
    if (std.mem.allEqual(u8, &digest, 0)) return error.InvalidQualityWorkerAuthority;
    return digest;
}

/// The maintained quality worker is a context-test executable and therefore
/// uses the parent-audited V4 receipt. Keep this parser private to the worker;
/// the generic release parser intentionally continues rejecting V4.
fn parseWorkerV4(allocator: std.mem.Allocator, bytes: []const u8) !release_build.Binding {
    var parsed = std.json.parseFromSlice(WorkerV4Receipt, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch return error.InvalidQualityWorkerAuthority;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.context-release-build.v4") or
        !std.mem.eql(u8, value.evidence_scope, "full_context_release_build") or
        !value.release_eligible or value.serving_qualified or !value.source_verified_after_build or
        !value.compiler_profile.strip_debug_info or value.compiler_profile.have_error_return_tracing or
        value.compiler_profile.strip_debug_info != builtin.strip_debug_info or
        value.compiler_profile.have_error_return_tracing != builtin.have_error_return_tracing or
        value.binary.bytes == 0 or !std.mem.eql(u8, value.binary.path, "context-tests") or
        value.resource.path.len == 0 or value.resource.path[0] == '/' or
        value.tests.failed != 0 or value.tests.leaked != 0 or value.tests.passed == 0 or
        value.tests.selected != value.tests.passed + value.tests.skipped or
        !std.mem.eql(u8, value.build.target, "inference-test-context") or !value.build.metal or value.build.cuda or
        !std.mem.eql(u8, value.build.optimize, "ReleaseFast"))
        return error.InvalidQualityWorkerAuthority;
    _ = try parseDigest(value.resource.sha256);
    return .{ .binary_sha256 = try parseDigest(value.binary.sha256) };
}

fn validateDigestNamespaces(profile_runtime: [32]u8, compiled_runtime: [32]u8, authority_implementation: [32]u8) !void {
    if (!std.meta.eql(profile_runtime, compiled_runtime) or
        std.mem.allEqual(u8, &authority_implementation, 0) or
        std.meta.eql(compiled_runtime, authority_implementation))
        return error.InvalidQualityWorkerAuthority;
}

/// Installs the private policy into exactly one test Node before model load.
/// The caller must independently audit that both build receipts name the same
/// frozen source; neither receipt format alone proves that cross-binary join.
pub fn install(
    allocator: std.mem.Allocator,
    io: std.Io,
    node: *server.Node,
    profile: profile_mod.Profile,
    worker_receipt_json: []const u8,
    full_cli_receipt_json: []const u8,
) !InstallEvidence {
    return installWithTraining(allocator, io, node, profile, worker_receipt_json, full_cli_receipt_json, .private_dense512_reference_v1, null);
}

/// New authenticated staged profile entrypoint. There is no HTTP/env/default
/// path to this parameter; callers must have already authenticated the profile
/// and both binary receipts.
pub fn installWithTraining(
    allocator: std.mem.Allocator,
    io: std.Io,
    node: *server.Node,
    profile: profile_mod.Profile,
    worker_receipt_json: []const u8,
    full_cli_receipt_json: []const u8,
    training_policy: generation.NativeCheckpointTrainingPolicy,
    resident_hybrid_authority: ?generation.ResidentHybridAuthority,
) !InstallEvidence {
    if (comptime !builtin.is_test) @compileError("quality worker staged candidate is test-binary-only");
    const worker = try parseWorkerV4(allocator, worker_receipt_json);
    const full_cli = try release_build.parseStagedFullCli(allocator, full_cli_receipt_json);
    if (training_policy == .private_train_challenge_hybrid_resident_v1) {
        const resident = resident_hybrid_authority orelse return error.InvalidQualityWorkerAuthority;
        const cli_source = full_cli.source_freeze_sha256 orelse return error.InvalidQualityWorkerAuthority;
        if (!std.mem.eql(u8, &cli_source, &resident.source_freeze_sha256)) return error.InvalidQualityWorkerAuthority;
    }
    var full_cli_join = try std.json.parseFromSlice(FullCliProofJoin, allocator, full_cli_receipt_json, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer full_cli_join.deinit();
    var worker_receipt_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(worker_receipt_json, &worker_receipt_sha256, .{});
    const joined_worker_receipt = try parseDigest(full_cli_join.value.full_context_release_proof.sha256);
    if (std.mem.eql(u8, &worker.binary_sha256, &full_cli.binary_sha256))
        return error.InvalidQualityWorkerAuthority;
    if (!std.mem.eql(u8, &worker_receipt_sha256, &joined_worker_receipt)) return error.InvalidQualityWorkerAuthority;
    if (profile.engine != .native_whole_head_v2 or !std.meta.eql(profile.producer_policy, fit_policy.Policy.cumulative256_exact_cycle_v1) or
        profile.query_protocols.len != 0 or profile.max_prompt_tokens != 32768 or profile.max_output_tokens != 256)
        return error.InvalidQualityWorkerAuthority;
    const profile_sha = try profile_mod.profileArtifactDigest(profile);
    const semantic_sha = profile.semanticDigest();
    const implementation_sha = profile_mod.implementationDigest(profile_mod.servingImplementation(.native_whole_head_v2, true, true));
    const authority_implementation_sha = @import("compaction_identity").digest;
    try validateDigestNamespaces(profile.identity.runtime, implementation_sha, authority_implementation_sha);
    const execution: @import("runtime/kv/attention_match_release.zig").Execution = .{ .forward_frame = true, .precomputed_tail = true, .retain_hidden = true, .matrix_stage = true, .ple_tail = true, .resource_barriers = true };
    const trusted = try authority.Authority.initVerified(
        allocator,
        io,
        worker.binary_sha256,
        authority_implementation_sha,
        .{
            .checkpoint = checkpoint.Identity{
                .weights = profile.identity.artifact,
                .tokenizer = profile.identity.tokenizer,
                .template = profile.identity.template,
                .layout = profile.identity.layout,
            },
            .profile_sha256 = profile_sha,
            .profile_semantic_sha256 = semantic_sha,
        },
        .cumulative256_exact_cycle_v1,
        execution,
    );
    if (training_policy == .private_train_challenge_hybrid_resident_v1) {
        const resident = resident_hybrid_authority orelse return error.InvalidQualityWorkerAuthority;
        try resident.validate();
        try node.installNativeKvResidentHybridStagedAuthority(resident);
    } else if (resident_hybrid_authority != null) return error.InvalidQualityWorkerAuthority;
    try node.installNativeKvQualificationCandidate(
        trusted,
        profile,
        .attention_matching,
        quality.FitStrategy.identity_chunk_coefficients_adaptive_v18,
        training_policy,
        true,
        if (training_policy == .private_future_decode_mix_v1 or training_policy == .private_train_challenge_hybrid_resident_v1) .{
            .max_copy_bytes = 4 * 1024 * 1024 * 1024,
            .max_host_staging_bytes = 256 * 1024 * 1024,
            .codec_scope = .f16_fp8_rotated4,
            .max_logical_tokens = 32768,
            .layout_policy = .full_history_exact,
        } else null,
    );
    return .{
        .worker_binary_sha256 = worker.binary_sha256,
        .full_cli_binary_sha256 = full_cli.binary_sha256,
        .profile_sha256 = profile_sha,
        .profile_semantic_sha256 = semantic_sha,
        .implementation_semantic_sha256 = implementation_sha,
        .authority_implementation_sha256 = authority_implementation_sha,
        .execution = trusted.execution,
        .cross_binary_source_join_verified = true,
    };
}

/// Diagnostic-only authority for fast retained-model iteration.  Unlike the
/// qualification entrypoint above, this deliberately has no full-CLI join and
/// therefore cannot produce qualification or release evidence.
pub fn installFocusedFutureDiagnostic(
    allocator: std.mem.Allocator,
    io: std.Io,
    node: *server.Node,
    profile: profile_mod.Profile,
    worker_receipt_json: []const u8,
) !InstallEvidence {
    if (comptime !builtin.is_test) @compileError("focused future diagnostic is test-binary-only");
    const worker = try parseFocusedDiagnostic(allocator, io, worker_receipt_json);
    if (profile.engine != .native_whole_head_v2 or !std.meta.eql(profile.producer_policy, fit_policy.Policy.cumulative256_exact_cycle_v1) or
        profile.query_protocols.len != 0 or profile.max_prompt_tokens != 32768 or profile.max_output_tokens != 256)
        return error.InvalidQualityWorkerAuthority;
    const profile_sha = try profile_mod.profileArtifactDigest(profile);
    const semantic_sha = profile.semanticDigest();
    const implementation_sha = profile_mod.implementationDigest(profile_mod.servingImplementation(.native_whole_head_v2, true, true));
    const authority_implementation_sha = @import("compaction_identity").digest;
    try validateDigestNamespaces(profile.identity.runtime, implementation_sha, authority_implementation_sha);
    const execution: @import("runtime/kv/attention_match_release.zig").Execution = .{ .forward_frame = true, .precomputed_tail = true, .retain_hidden = true, .matrix_stage = true, .ple_tail = true, .resource_barriers = true };
    const trusted = try authority.Authority.initVerified(allocator, io, worker.binary_sha256, authority_implementation_sha, .{
        .checkpoint = .{ .weights = profile.identity.artifact, .tokenizer = profile.identity.tokenizer, .template = profile.identity.template, .layout = profile.identity.layout },
        .profile_sha256 = profile_sha,
        .profile_semantic_sha256 = semantic_sha,
    }, .cumulative256_exact_cycle_v1, execution);
    try node.installNativeKvQualificationCandidate(trusted, profile, .attention_matching, .identity_chunk_coefficients_adaptive_v18, .private_future_decode_mix_v1, true, .{
        .max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .max_host_staging_bytes = 256 * 1024 * 1024,
        .codec_scope = .f16_fp8_rotated4,
        .max_logical_tokens = 32768,
        .layout_policy = .full_history_exact,
    });
    return .{
        .worker_binary_sha256 = worker.binary_sha256,
        .full_cli_binary_sha256 = @splat(0),
        .profile_sha256 = profile_sha,
        .profile_semantic_sha256 = semantic_sha,
        .implementation_semantic_sha256 = implementation_sha,
        .authority_implementation_sha256 = authority_implementation_sha,
        .execution = trusted.execution,
        .cross_binary_source_join_verified = false,
    };
}

test "context quality worker V4 receipt and digest namespaces fail closed" {
    const good =
        \\{"schema":"antfly.context-release-build.v4","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false},"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"resource":{"path":"build-resource.json","sha256":"2222222222222222222222222222222222222222222222222222222222222222"},"tests":{"failed":0,"leaked":0,"passed":3,"selected":4,"skipped":1},"build":{"target":"inference-test-context","metal":true,"cuda":false,"optimize":"ReleaseFast"}}
    ;
    const parsed = try parseWorkerV4(std.testing.allocator, good);
    try std.testing.expectEqual(@as(u8, 0x11), parsed.binary_sha256[0]);
    const wrong_schema =
        \\{"schema":"antfly.context-release-build.v3","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false},"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"resource":{"path":"build-resource.json","sha256":"2222222222222222222222222222222222222222222222222222222222222222"},"tests":{"failed":0,"leaked":0,"passed":3,"selected":4,"skipped":1},"build":{"target":"inference-test-context","metal":true,"cuda":false,"optimize":"ReleaseFast"}}
    ;
    try std.testing.expectError(error.InvalidQualityWorkerAuthority, parseWorkerV4(std.testing.allocator, wrong_schema));
    const runtime_digest: [32]u8 = @splat(0x33);
    const authority_digest: [32]u8 = @splat(0x44);
    try validateDigestNamespaces(runtime_digest, runtime_digest, authority_digest);
    try std.testing.expectError(error.InvalidQualityWorkerAuthority, validateDigestNamespaces(@splat(0x55), runtime_digest, authority_digest));
    try std.testing.expectError(error.InvalidQualityWorkerAuthority, validateDigestNamespaces(runtime_digest, runtime_digest, runtime_digest));
    try std.testing.expectError(error.InvalidQualityWorkerAuthority, validateDigestNamespaces(runtime_digest, runtime_digest, @splat(0)));
}
