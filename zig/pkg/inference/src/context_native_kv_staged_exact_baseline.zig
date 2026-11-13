// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Operator-only authority for ordinary exact baseline measurement.
//! It never installs a compaction profile, producer, trusted consumer, or
//! checkpoint route and carries no fitted-quality or savings assertion.
const std = @import("std");
const server = @import("server/server.zig");
const release_build = @import("context_release_build_binding.zig");
const authority_mod = @import("server/native_kv_producer_authority.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");
const release = @import("runtime/kv/attention_match_release.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");
const generation = @import("pipelines/generation.zig");
const compaction_identity = @import("compaction_identity");

const FileBinding = struct {
    schema: []const u8,
    mode: []const u8,
    release_receipt_path: []const u8,
    implementation_semantic_sha256: []const u8,
    authority_sha256: []const u8,
    execution_name: []const u8,
    execution: release.Execution,
    qualification_arm: generation.NativeCheckpointQualificationArm,
    compiled_packed_prefill_bridge: ?struct {
        observation_path: []const u8,
        observation_sha256: []const u8,
        max_copy_bytes: usize,
        max_host_staging_bytes: usize,
        max_logical_tokens: usize,
    } = null,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
};

fn digest32(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidQualificationCandidate;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidQualificationCandidate;
    return out;
}

fn loadFullCliReleaseProof(allocator: std.mem.Allocator, io: std.Io, full_cli_bytes: []const u8) !release_build.Binding {
    const Join = struct { full_context_release_proof: struct { path: []const u8, sha256: []const u8 } };
    var parsed = std.json.parseFromSlice(Join, allocator, full_cli_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true }) catch return error.InvalidQualificationCandidate;
    defer parsed.deinit();
    const pin = parsed.value.full_context_release_proof;
    if (!std.fs.path.isAbsolute(pin.path)) return error.InvalidQualificationCandidate;
    const expected = try digest32(pin.sha256);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, pin.path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidQualificationCandidate;
    return release_build.parseStagedFullContextProof(allocator, bytes) catch return error.InvalidQualificationCandidate;
}

const compiled_observation = @import("context_compiled_packed_observation.zig");
const CompiledPackedObservation = compiled_observation.Document;

fn validateCompiledPackedObservation(observed: CompiledPackedObservation, identity: checkpoint.Identity, full_cli_binary_sha256: [32]u8, release_proof_binary_sha256: [32]u8, source_freeze_sha256: [32]u8, implementation_sha256: [32]u8, maximum_host_staging_bytes: usize, maximum_tokens: usize) !void {
    const source = observed.result.source_authority;
    const expected_owners = if (observed.layout_census) |census| blk: {
        if (census.attention_layers == 0 or census.owner_count == 0 or census.owner_count > census.attention_layers or std.mem.allEqual(u8, &census.owner_projection_sha256, 0) or
            !std.mem.eql(u8, &census.context_layout_sha256, &source.context_layout_sha256)) return error.InvalidQualificationCandidate;
        break :blk census.owner_count;
    } else 24; // Immutable V3 E4 observations predate the compiled census.
    if (!std.mem.eql(u8, observed.schema, compiled_observation.schema_name) or
        !observed.diagnostic_only or observed.qualification_credit or observed.serving_activation or
        !observed.compiled_prefill_completed or !observed.exact_export_completed or
        !std.mem.eql(u8, observed.exact_export_abi, compiled_observation.export_abi_name) or
        observed.exact_export_peak_host_staging_bytes > maximum_host_staging_bytes or
        std.mem.allEqual(u8, &observed.test_binary_sha256, 0) or
        std.mem.eql(u8, &observed.test_binary_sha256, &full_cli_binary_sha256) or
        !std.mem.eql(u8, &observed.test_binary_sha256, &release_proof_binary_sha256) or
        !std.mem.eql(u8, &observed.source_freeze_sha256, &source_freeze_sha256) or
        !std.mem.eql(u8, &observed.implementation_semantic_sha256, &implementation_sha256) or
        !observed.model_identity.eql(identity) or source.source_revision == 0 or source.source_sequence == 0 or
        source.logical_tokens == 0 or source.logical_tokens > maximum_tokens or
        !std.mem.eql(u8, &source.model_weights_sha256, &identity.weights) or
        !std.mem.eql(u8, &source.model_identity_layout_sha256, &identity.layout) or
        std.mem.allEqual(u8, &source.context_layout_sha256, 0) or std.mem.allEqual(u8, &source.token_stream_sha256, 0) or
        observed.result.copied_bytes == 0 or observed.result.imported_owners != expected_owners or !observed.result.source_unchanged or !observed.result.release_complete or
        observed.result.compared_owners != expected_owners or observed.result.total_key_mismatches != 0 or
        observed.result.total_value_mismatches != 0 or !observed.result.all_f16_identical)
        return error.InvalidQualificationCandidate;
}

fn loadCompiledPackedObservation(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected_sha256: [32]u8, identity: checkpoint.Identity, full_cli_binary_sha256: [32]u8, release_proof_binary_sha256: [32]u8, source_freeze_sha256: [32]u8, implementation_sha256: [32]u8, maximum_host_staging_bytes: usize, maximum_tokens: usize) !void {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualificationCandidate;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected_sha256)) return error.InvalidQualificationCandidate;
    var parsed = compiled_observation.parse(allocator, bytes) catch return error.InvalidQualificationCandidate;
    defer parsed.deinit();
    try validateCompiledPackedObservation(parsed.value, identity, full_cli_binary_sha256, release_proof_binary_sha256, source_freeze_sha256, implementation_sha256, maximum_host_staging_bytes, maximum_tokens);
}

test "compiled packed observation binds completed revision identity and token boundary" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const full_cli: [32]u8 = @splat(5);
    const proof: [32]u8 = @splat(12);
    const freeze: [32]u8 = @splat(6);
    const implementation: [32]u8 = @splat(7);
    var observed: CompiledPackedObservation = .{
        .schema = compiled_observation.schema_name,
        .diagnostic_only = true,
        .qualification_credit = false,
        .serving_activation = false,
        .compiled_prefill_completed = true,
        .exact_export_completed = true,
        .exact_export_abi = compiled_observation.export_abi_name,
        .exact_export_peak_host_staging_bytes = 0,
        .source_checkpoint_sha256 = @splat(13),
        .source_freeze_sha256 = freeze,
        .test_binary_sha256 = proof,
        .implementation_semantic_sha256 = implementation,
        .model_identity = identity,
        .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024,
        .result = .{
            .source_authority = .{ .model_weights_sha256 = identity.weights, .model_identity_layout_sha256 = identity.layout, .context_layout_sha256 = @splat(9), .token_stream_sha256 = @splat(10), .source_revision = 7, .source_sequence = 11, .logical_tokens = 8 },
            .copied_bytes = 1,
            .imported_owners = 24,
            .source_unchanged = true,
            .release_complete = true,
            .compared_owners = 24,
            .total_key_mismatches = 0,
            .total_value_mismatches = 0,
            .all_f16_identical = true,
        },
    };
    try validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8);
    observed.exact_export_peak_host_staging_bytes = 2;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.exact_export_peak_host_staging_bytes = 0;
    observed.result.source_authority.source_revision = 0;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.result.source_authority.source_revision = 7;
    observed.result.source_authority.logical_tokens = 9;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.result.source_authority.logical_tokens = 8;
    observed.test_binary_sha256 = @splat(8);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.test_binary_sha256 = full_cli;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.test_binary_sha256 = proof;
    observed.source_freeze_sha256[0] ^= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
    observed.source_freeze_sha256 = freeze;
    observed.implementation_semantic_sha256[0] ^= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, full_cli, proof, freeze, implementation, 1, 8));
}

test "compiled packed observation consumes authenticated layout-derived owner census" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var observed: CompiledPackedObservation = .{ .schema = compiled_observation.schema_name, .diagnostic_only = true, .qualification_credit = false, .serving_activation = false, .compiled_prefill_completed = true, .exact_export_completed = true, .exact_export_abi = compiled_observation.export_abi_name, .exact_export_peak_host_staging_bytes = 0, .source_checkpoint_sha256 = @splat(5), .source_freeze_sha256 = @splat(6), .test_binary_sha256 = @splat(12), .implementation_semantic_sha256 = @splat(7), .model_identity = identity, .process_rss_limit_bytes = 8 * 1024 * 1024 * 1024, .layout_census = .{ .attention_layers = 35, .owner_count = 15, .context_layout_sha256 = @splat(9), .owner_projection_sha256 = @splat(11) }, .result = .{ .source_authority = .{ .model_weights_sha256 = identity.weights, .model_identity_layout_sha256 = identity.layout, .context_layout_sha256 = @splat(9), .token_stream_sha256 = @splat(10), .source_revision = 1, .source_sequence = 1, .logical_tokens = 8128 }, .copied_bytes = 1, .imported_owners = 15, .source_unchanged = true, .release_complete = true, .compared_owners = 15, .total_key_mismatches = 0, .total_value_mismatches = 0, .all_f16_identical = true } };
    try validateCompiledPackedObservation(observed, identity, @splat(5), @splat(12), @splat(6), @splat(7), 1, 8128);
    observed.layout_census.?.owner_count = 24;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, @splat(5), @splat(12), @splat(6), @splat(7), 1, 8128));
    observed.layout_census.?.owner_count = 15;
    observed.layout_census.?.context_layout_sha256[0] ^= 1;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateCompiledPackedObservation(observed, identity, @splat(5), @splat(12), @splat(6), @splat(7), 1, 8128));
}

fn authorityDigest(binary: [32]u8, implementation: [32]u8, identity: checkpoint.Identity, arm: generation.NativeCheckpointQualificationArm, execution: release.Execution, bridge: ?generation.CompiledPackedPrefillBridgeAuthority, release_proof_binary: ?[32]u8) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(if (bridge == null) "antfly.staged-exact-baseline-authority.v1\n" else "antfly.staged-exact-baseline-authority.v2\n");
    const execution_digest = execution.fingerprint();
    hash.update(&binary);
    hash.update(&implementation);
    hash.update(&identity.weights);
    hash.update(&identity.tokenizer);
    hash.update(&identity.template);
    hash.update(&identity.layout);
    hash.update(@tagName(arm));
    hash.update(&execution_digest);
    if (bridge) |capability| {
        const capability_digest = capability.digest();
        hash.update(&capability_digest);
        const proof_binary = release_proof_binary orelse unreachable;
        hash.update(&proof_binary);
    }
    return hash.finalResult();
}

fn semanticDigest(implementation: [32]u8, arm: generation.NativeCheckpointQualificationArm) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.staged-exact-baseline-semantic.v1\n");
    hash.update(&implementation);
    hash.update(@tagName(arm));
    return hash.finalResult();
}

pub const Loaded = struct {
    authority: authority_mod.Authority,
    arm: generation.NativeCheckpointQualificationArm,
    compiled_packed_prefill_bridge: ?generation.CompiledPackedPrefillBridgeAuthority,
};

fn validateExecutionEnvelope(schema: []const u8, execution_name: []const u8, execution: release.Execution) !void {
    const baseline_v1 = std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v1") and
        std.mem.eql(u8, execution_name, "gemma4_exact_owner_v1");
    const baseline_v2 = std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v2") and
        std.mem.eql(u8, execution_name, "gemma4_exact_coalesced_owner_v1");
    const baseline_v3 = std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v3") and
        std.mem.eql(u8, execution_name, "gemma4_exact_coalesced_fused_half_owner_v1");
    const baseline_v4 = std.mem.eql(u8, schema, "antfly.native-kv-staged-exact-baseline.v4") and
        std.mem.eql(u8, execution_name, "gemma4_exact_compiled_packed_bridge_v1");
    if (!baseline_v1 and !baseline_v2 and !baseline_v3 and !baseline_v4) return error.InvalidQualificationCandidate;
    try execution.validate();
    if (!execution.forward_frame or !execution.precomputed_tail or
        !execution.retain_hidden or !execution.matrix_stage or
        !execution.ple_tail or !execution.resource_barriers or
        execution.coalesced_stage != (baseline_v2 or baseline_v3 or baseline_v4) or
        execution.resident_gated_ffn_fusion != (baseline_v3 or baseline_v4) or execution.concurrent_frame)
        return error.InvalidQualificationCandidate;
}

fn parseAndValidateFileBinding(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(FileBinding) {
    var parsed = std.json.parseFromSlice(FileBinding, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch return error.InvalidQualificationCandidate;
    errdefer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.mode, "staged_exact_baseline") or
        !std.fs.path.isAbsolute(value.release_receipt_path) or value.qualification_arm == .attention_matching)
        return error.InvalidQualificationCandidate;
    try validateExecutionEnvelope(value.schema, value.execution_name, value.execution);
    if ((value.compiled_packed_prefill_bridge != null) != std.mem.eql(u8, value.schema, "antfly.native-kv-staged-exact-baseline.v4") or
        (value.compiled_packed_prefill_bridge != null and value.qualification_arm != .packed_exact))
        return error.InvalidQualificationCandidate;
    return parsed;
}

pub fn loadFromConfigBytes(allocator: std.mem.Allocator, io: std.Io, bytes: []const u8) !Loaded {
    var parsed = try parseAndValidateFileBinding(allocator, bytes);
    defer parsed.deinit();
    const value = parsed.value;
    const receipt_bytes = try std.Io.Dir.cwd().readFileAlloc(io, value.release_receipt_path, allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(receipt_bytes);
    const receipt = try release_build.parseStagedFullCli(allocator, receipt_bytes);
    const implementation = try digest32(value.implementation_semantic_sha256);
    if (!std.mem.eql(u8, &implementation, &compaction_identity.digest)) return error.NativeKvProducerImplementationMismatch;
    const identity = checkpoint.Identity{ .weights = try digest32(value.identity.weights), .tokenizer = try digest32(value.identity.tokenizer), .template = try digest32(value.identity.template), .layout = try digest32(value.identity.layout) };
    const bridge: ?generation.CompiledPackedPrefillBridgeAuthority = if (value.compiled_packed_prefill_bridge) |capability| .{
        .observation_sha256 = try digest32(capability.observation_sha256),
        .max_copy_bytes = capability.max_copy_bytes,
        .max_host_staging_bytes = capability.max_host_staging_bytes,
        .max_logical_tokens = capability.max_logical_tokens,
    } else null;
    const release_proof = if (bridge != null) try loadFullCliReleaseProof(allocator, io, receipt_bytes) else null;
    if (bridge) |capability| {
        try capability.validate();
        try loadCompiledPackedObservation(
            allocator,
            io,
            value.compiled_packed_prefill_bridge.?.observation_path,
            capability.observation_sha256,
            identity,
            receipt.binary_sha256,
            release_proof.?.binary_sha256,
            receipt.source_freeze_sha256 orelse return error.InvalidQualificationCandidate,
            implementation,
            capability.max_host_staging_bytes,
            capability.max_logical_tokens,
        );
    }
    const sealed = authorityDigest(receipt.binary_sha256, implementation, identity, value.qualification_arm, value.execution, bridge, if (release_proof) |proof| proof.binary_sha256 else null);
    if (!std.mem.eql(u8, &sealed, &try digest32(value.authority_sha256))) return error.InvalidQualificationCandidate;
    const authority = try authority_mod.Authority.initVerified(allocator, io, receipt.binary_sha256, implementation, .{
        .checkpoint = identity,
        .profile_sha256 = sealed,
        .profile_semantic_sha256 = semanticDigest(implementation, value.qualification_arm),
    }, fit_policy.Policy.legacy_fixed, value.execution);
    return .{ .authority = authority, .arm = value.qualification_arm, .compiled_packed_prefill_bridge = bridge };
}

test "staged exact baseline v2 requires only the authenticated coalesced execution pair" {
    const ordinary: release.Execution = .{ .forward_frame = true, .precomputed_tail = true, .retain_hidden = true, .matrix_stage = true, .ple_tail = true, .resource_barriers = true };
    var coalesced = ordinary;
    coalesced.coalesced_stage = true;
    try validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v1", "gemma4_exact_owner_v1", ordinary);
    try validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_coalesced_owner_v1", coalesced);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v1", "gemma4_exact_coalesced_owner_v1", ordinary));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_owner_v1", coalesced));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_coalesced_owner_v1", ordinary));
    var gated_v1 = ordinary;
    gated_v1.resident_gated_ffn_fusion = true;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v1", "gemma4_exact_owner_v1", gated_v1));
    var gated_v2 = coalesced;
    gated_v2.resident_gated_ffn_fusion = true;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_coalesced_owner_v1", gated_v2));
    try validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v3", "gemma4_exact_coalesced_fused_half_owner_v1", gated_v2);
    try validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v4", "gemma4_exact_compiled_packed_bridge_v1", gated_v2);
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v3", "gemma4_exact_coalesced_fused_half_owner_v1", coalesced));
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v4", "gemma4_exact_compiled_packed_bridge_v1", coalesced));
    var concurrent = coalesced;
    concurrent.concurrent_frame = true;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_coalesced_owner_v1", concurrent));
    var missing = coalesced;
    missing.resource_barriers = false;
    try std.testing.expectError(error.InvalidQualificationCandidate, validateExecutionEnvelope("antfly.native-kv-staged-exact-baseline.v2", "gemma4_exact_coalesced_owner_v1", missing));
}

test "staged exact baseline full binding parser accepts v2 and rejects mixed or unknown policy" {
    const valid =
        \\{"schema":"antfly.native-kv-staged-exact-baseline.v2","mode":"staged_exact_baseline","release_receipt_path":"/receipt.json","implementation_semantic_sha256":"00","authority_sha256":"00","execution_name":"gemma4_exact_coalesced_owner_v1","execution":{"forward_frame":true,"precomputed_tail":true,"retain_hidden":true,"matrix_stage":true,"ple_tail":true,"resource_barriers":true,"coalesced_stage":true,"concurrent_frame":false},"qualification_arm":"optimal_compiled_exact","identity":{"weights":"00","tokenizer":"00","template":"00","layout":"00"}}
    ;
    var parsed = try parseAndValidateFileBinding(std.testing.allocator, valid);
    defer parsed.deinit();
    try std.testing.expect(parsed.value.execution.coalesced_stage);
    const mixed = std.mem.replaceOwned(u8, std.testing.allocator, valid, "gemma4_exact_coalesced_owner_v1", "gemma4_exact_owner_v1") catch unreachable;
    defer std.testing.allocator.free(mixed);
    try std.testing.expectError(error.InvalidQualificationCandidate, parseAndValidateFileBinding(std.testing.allocator, mixed));
    const unknown = std.mem.replaceOwned(u8, std.testing.allocator, valid, "\"identity\":", "\"unknown\":1,\"identity\":") catch unreachable;
    defer std.testing.allocator.free(unknown);
    try std.testing.expectError(error.InvalidQualificationCandidate, parseAndValidateFileBinding(std.testing.allocator, unknown));
}

test "staged exact baseline v4 requires packed compiled observation capability" {
    const valid =
        \\{"schema":"antfly.native-kv-staged-exact-baseline.v4","mode":"staged_exact_baseline","release_receipt_path":"/receipt.json","implementation_semantic_sha256":"00","authority_sha256":"00","execution_name":"gemma4_exact_compiled_packed_bridge_v1","execution":{"forward_frame":true,"precomputed_tail":true,"retain_hidden":true,"matrix_stage":true,"ple_tail":true,"resource_barriers":true,"coalesced_stage":true,"resident_gated_ffn_fusion":true,"concurrent_frame":false},"qualification_arm":"packed_exact","compiled_packed_prefill_bridge":{"observation_path":"/compiled-observation.json","observation_sha256":"00","max_copy_bytes":1,"max_host_staging_bytes":1,"max_logical_tokens":1},"identity":{"weights":"00","tokenizer":"00","template":"00","layout":"00"}}
    ;
    var parsed = try parseAndValidateFileBinding(std.testing.allocator, valid);
    defer parsed.deinit();
    try std.testing.expect(parsed.value.compiled_packed_prefill_bridge != null);
    const missing = try std.mem.replaceOwned(u8, std.testing.allocator, valid, ",\"compiled_packed_prefill_bridge\":{\"observation_path\":\"/compiled-observation.json\",\"observation_sha256\":\"00\",\"max_copy_bytes\":1,\"max_host_staging_bytes\":1,\"max_logical_tokens\":1}", "");
    defer std.testing.allocator.free(missing);
    try std.testing.expectError(error.InvalidQualificationCandidate, parseAndValidateFileBinding(std.testing.allocator, missing));
    const wrong_arm = try std.mem.replaceOwned(u8, std.testing.allocator, valid, "\"packed_exact\"", "\"optimal_compiled_exact\"");
    defer std.testing.allocator.free(wrong_arm);
    try std.testing.expectError(error.InvalidQualificationCandidate, parseAndValidateFileBinding(std.testing.allocator, wrong_arm));
}

pub fn installFromConfigBytes(allocator: std.mem.Allocator, io: std.Io, node: *server.Node, bytes: []const u8) !void {
    const loaded = try loadFromConfigBytes(allocator, io, bytes);
    try node.installNativeKvStagedExactBaseline(loaded.authority, loaded.arm, loaded.compiled_packed_prefill_bridge);
}

test "context staged exact baseline authority digest separates arms and identity" {
    const identity = checkpoint.Identity{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const execution: release.Execution = .{ .forward_frame = true, .precomputed_tail = true, .retain_hidden = true, .matrix_stage = true, .ple_tail = true, .resource_barriers = true };
    const exact = authorityDigest(@splat(5), compaction_identity.digest, identity, .optimal_compiled_exact, execution, null, null);
    const packed_digest = authorityDigest(@splat(5), compaction_identity.digest, identity, .packed_exact, execution, null, null);
    try std.testing.expect(!std.mem.eql(u8, &exact, &packed_digest));
    var changed = identity;
    changed.layout[0] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &exact, &authorityDigest(@splat(5), compaction_identity.digest, changed, .optimal_compiled_exact, execution, null, null)));
    var changed_execution = execution;
    changed_execution.resource_barriers = false;
    try std.testing.expect(!std.mem.eql(u8, &exact, &authorityDigest(@splat(5), compaction_identity.digest, identity, .optimal_compiled_exact, changed_execution, null, null)));
    changed_execution.concurrent_frame = true;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, changed_execution.validate());
    const bridge: generation.CompiledPackedPrefillBridgeAuthority = .{ .observation_sha256 = @splat(8), .max_copy_bytes = 1, .max_host_staging_bytes = 1, .max_logical_tokens = 1 };
    const proof_a = authorityDigest(@splat(5), compaction_identity.digest, identity, .packed_exact, execution, bridge, @splat(9));
    const proof_b = authorityDigest(@splat(5), compaction_identity.digest, identity, .packed_exact, execution, bridge, @splat(10));
    try std.testing.expect(!std.mem.eql(u8, &proof_a, &proof_b));
}
