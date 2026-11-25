// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Process-local bridge from the actual public AFATT002 producer to the
//! retained quality worker.  It authenticates already-produced bytes and
//! evidence; it never constructs or refits a candidate.
const std = @import("std");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
const verified_sidecar = @import("native_kv_verified_sidecar.zig");

pub const schema = "antfly.native-kv-prepared-quality-closure.v1";

pub const FutureTrainingAuthority = struct {
    policy_sha256: [64]u8,
    source_identity_sha256: [64]u8,
    source_tokens_sha256: [64]u8,
    source_revision: u64,
    generated_ids_sha256: [64]u8,
    early_eos_observations: u16,
    primary_queries_sha256: [64]u8,
    historical_queries_sha256: [64]u8,
    future_queries_sha256: [64]u8,
    original_heldout_sha256: [64]u8,
};

pub const CodecEvidence = struct {
    fp8_heads: usize = 0,
    fp8_fallback_heads: usize = 0,
    fp8_eligible_rows: usize = 0,
    fp8_encoded_rows: usize = 0,
    fp8_exact_rows: usize = 0,
    rotated4_heads: usize = 0,
    rotated4_fallback_heads: usize = 0,
    rotated4_eligible_rows: usize = 0,
    rotated4_compressed_rows: usize = 0,
    rotated4_exact_rows: usize = 0,
};

pub const Record = struct {
    schema: []const u8 = schema,
    checkpoint_sha256: [64]u8,
    producer_sha256: [64]u8,
    prompt_tokens_sha256: [64]u8,
    source_checkpoint_sha256: [64]u8,
    profile_sha256: [64]u8,
    profile_semantic_sha256: [64]u8,
    implementation_semantic_sha256: [64]u8,
    binary_sha256: [64]u8,
    policy_sha256: [64]u8,
    checkpoint_layout_sha256: [64]u8,
    encoding: []const u8,
    logical_tokens: usize,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    validated_heads: usize,
    preserved_heads: usize,
    completed_chunks: usize,
    max_head_mass_error: f64,
    max_head_output_rmse: f64,
    cache_ready_heads: usize,
    exact_f32_probability_heads: usize,
    incremental_gram_chooser_calls: usize,
    training_reduction_local_original_calls: usize,
    training_reduction_rounded_harm_calls: usize,
    training_reduction_tiles: usize,
    training_reduction_backend_calls: usize,
    training_reduction_workspace_peak_bytes: usize,
    exact_local_window_capacity: bool,
    v18_policy_sha256: [64]u8,
    v18_eligible_heads: usize,
    v18_attempted_heads: usize,
    v18_accepted_heads: usize,
    v18_preserved_heads: usize,
    v18_fallback_heads: usize,
    v18_cache_ready_heads: usize,
    v18_planned_exact_heads: usize = 0,
    v18_query_batch_rows: usize,
    v18_planned_head_workspace_bytes: usize,
    v18_cache_phase_peak_bound_bytes: usize,
    v18_bounded_heads: usize,
    v18_bounded_delta_heads: usize,
    v18_bounded_delta_tiles: usize,
    v18_bounded_delta_backend_calls: usize,
    v18_bounded_delta_packed_bytes_peak: usize,
    v18_bounded_delta_staged_bytes_peak: usize,
    v18_bounded_delta_fallbacks: usize,
    v18_primary_budget_percent: u8,
    v18_budget_fallback_heads: usize,
    v18_evidence_profile: []const u8,
    v18_evidence_profile_sha256: [64]u8,
    v18_codec_physical_profile: ?[]const u8 = null,
    v18_codec_physical_profile_sha256: [64]u8 = @splat('0'),
    v18_bounded_tsqr_policy_sha256: [64]u8 = @splat('0'),
    v18_bounded_tsqr_heads: usize = 0,
    v18_bounded_tsqr_calls: usize = 0,
    v18_bounded_tsqr_workspace_peak_bytes: usize = 0,
    v18_bounded_tsqr_cancellation_checks: usize = 0,
    request_private_publish_complete: bool,
    validation_complete: bool,
    /// Absent preserves the v1 closure digest byte-for-byte. Present only for
    /// the authenticated staged future-mixture profile.
    future_training_authority: ?FutureTrainingAuthority = null,
    /// Absent preserves the legacy closure digest byte-for-byte. Resident
    /// packed profiles require this actual-result evidence.
    codec_evidence: ?CodecEvidence = null,
};

/// Production emitter join for the independent bounded codec/TSQR seal. Kept
/// generic so the server's concrete generation result remains the authority.
pub fn applyBoundedSeal(record: *Record, result: anytype) void {
    record.v18_codec_physical_profile = if (result.v18_codec_physical_profile == .none) null else @tagName(result.v18_codec_physical_profile);
    record.v18_codec_physical_profile_sha256 = std.fmt.bytesToHex(result.v18_codec_physical_profile_sha256, .lower);
    record.v18_bounded_tsqr_policy_sha256 = std.fmt.bytesToHex(result.v18_bounded_tsqr_policy_sha256, .lower);
    record.v18_bounded_tsqr_heads = result.v18_bounded_tsqr_heads;
    record.v18_bounded_tsqr_calls = result.v18_bounded_tsqr_calls;
    record.v18_bounded_tsqr_workspace_peak_bytes = result.v18_bounded_tsqr_workspace_peak_bytes;
    record.v18_bounded_tsqr_cancellation_checks = result.v18_bounded_tsqr_cancellation_checks;
}

pub const Envelope = struct {
    record: Record,
    closure_sha256: [64]u8,
};

fn hex64(value: []const u8) bool {
    if (value.len != 64 or std.mem.allEqual(u8, value, '0')) return false;
    for (value) |byte| if (!std.ascii.isDigit(byte) and !(byte >= 'a' and byte <= 'f')) return false;
    return true;
}

fn addInt(hash: *std.crypto.hash.sha2.Sha256, value: usize) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    hash.update(&bytes);
}

fn hashCore(record: Record, hash: *std.crypto.hash.sha2.Sha256) void {
    hash.update("antfly.native-kv-prepared-quality-core.v1\x00");
    inline for (.{ record.prompt_tokens_sha256, record.source_checkpoint_sha256, record.profile_sha256, record.profile_semantic_sha256, record.implementation_semantic_sha256, record.binary_sha256, record.policy_sha256, record.checkpoint_layout_sha256, record.v18_policy_sha256, record.v18_evidence_profile_sha256 }) |value| hash.update(&value);
    hash.update(record.encoding);
    hash.update("\x00");
    inline for (.{ record.logical_tokens, record.source_global_bytes, record.candidate_global_bytes, record.validated_heads, record.preserved_heads, record.completed_chunks, record.cache_ready_heads, record.exact_f32_probability_heads, record.incremental_gram_chooser_calls, record.training_reduction_local_original_calls, record.training_reduction_rounded_harm_calls, record.training_reduction_tiles, record.training_reduction_backend_calls, record.training_reduction_workspace_peak_bytes }) |value| addInt(hash, value);
    inline for (.{ record.v18_eligible_heads, record.v18_attempted_heads, record.v18_accepted_heads, record.v18_preserved_heads, record.v18_fallback_heads, record.v18_cache_ready_heads, record.v18_query_batch_rows, record.v18_planned_head_workspace_bytes, record.v18_cache_phase_peak_bound_bytes, record.v18_bounded_heads, record.v18_bounded_delta_heads, record.v18_bounded_delta_tiles, record.v18_bounded_delta_backend_calls, record.v18_bounded_delta_packed_bytes_peak, record.v18_bounded_delta_staged_bytes_peak, record.v18_bounded_delta_fallbacks, record.v18_primary_budget_percent, record.v18_budget_fallback_heads }) |value| addInt(hash, value);
    if (record.v18_planned_exact_heads != 0) {
        hash.update("resident-planned-exact-heads.v1\x00");
        addInt(hash, record.v18_planned_exact_heads);
    }
    hash.update(record.v18_evidence_profile);
    if (record.v18_codec_physical_profile) |profile| {
        hash.update(&record.v18_codec_physical_profile_sha256);
        hash.update(&record.v18_bounded_tsqr_policy_sha256);
        hash.update(profile);
        inline for (.{ record.v18_bounded_tsqr_heads, record.v18_bounded_tsqr_calls, record.v18_bounded_tsqr_workspace_peak_bytes, record.v18_bounded_tsqr_cancellation_checks }) |value| addInt(hash, value);
    }
    hash.update(std.mem.asBytes(&record.max_head_mass_error));
    hash.update(std.mem.asBytes(&record.max_head_output_rmse));
    hash.update(&.{ @intFromBool(record.exact_local_window_capacity), @intFromBool(record.request_private_publish_complete), @intFromBool(record.validation_complete) });
    if (record.future_training_authority) |authority| {
        hash.update("future-training-authority.v1\x00");
        inline for (.{ authority.policy_sha256, authority.source_identity_sha256, authority.source_tokens_sha256, authority.generated_ids_sha256, authority.primary_queries_sha256, authority.historical_queries_sha256, authority.future_queries_sha256, authority.original_heldout_sha256 }) |value| hash.update(&value);
        var revision: [8]u8 = undefined;
        std.mem.writeInt(u64, &revision, authority.source_revision, .little);
        hash.update(&revision);
        var eos: [2]u8 = undefined;
        std.mem.writeInt(u16, &eos, authority.early_eos_observations, .little);
        hash.update(&eos);
    }
    if (record.codec_evidence) |codec| {
        hash.update("resident-codec-evidence.v1\x00");
        inline for (.{ codec.fp8_heads, codec.fp8_fallback_heads, codec.fp8_eligible_rows, codec.fp8_encoded_rows, codec.fp8_exact_rows, codec.rotated4_heads, codec.rotated4_fallback_heads, codec.rotated4_eligible_rows, codec.rotated4_compressed_rows, codec.rotated4_exact_rows }) |value| addInt(hash, value);
    }
}

pub fn coreDigest(record: Record) ![32]u8 {
    try validateCore(record);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hashCore(record, &hash);
    var result: [32]u8 = undefined;
    hash.final(&result);
    return result;
}

pub fn digest(record: Record) ![32]u8 {
    try validate(record);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-kv-prepared-quality-closure.v1\x00");
    hash.update(&record.checkpoint_sha256);
    hash.update(&record.producer_sha256);
    hashCore(record, &hash);
    var result: [32]u8 = undefined;
    hash.final(&result);
    return result;
}

fn formalProfileLogicalTokensValid(profile: @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile, logical_tokens: usize) bool {
    return switch (profile) {
        .resident_e4_32512_f16_page_tsqr_v1, .resident_e4_32512_fp8_page_tsqr_v1, .resident_e4_32512_rotated4_page_tsqr_v1 => logical_tokens == 32512,
        .resident_e2_formal_f16_page_tsqr_v1, .resident_e2_formal_fp8_page_tsqr_v1, .resident_e2_formal_rotated4_page_tsqr_v1 => logical_tokens == 7936 or logical_tokens == 32512,
        else => true,
    };
}

fn validateCore(record: Record) !void {
    const candidate_scaled = std.math.mul(usize, record.candidate_global_bytes, 3) catch return error.InvalidPreparedServingQualityClosure;
    const source_scaled = std.math.mul(usize, record.source_global_bytes, 2) catch return error.InvalidPreparedServingQualityClosure;
    const total_heads = std.math.add(usize, record.validated_heads, record.preserved_heads) catch return error.InvalidPreparedServingQualityClosure;
    const profile = std.meta.stringToEnum(@import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile, record.v18_evidence_profile) orelse return error.InvalidPreparedServingQualityClosure;
    const resident_e4_bounded = profile == .resident_e4_32512_f16_page_tsqr_v1 or profile == .resident_e4_32512_fp8_page_tsqr_v1 or profile == .resident_e4_32512_rotated4_page_tsqr_v1;
    const resident_e2_bounded = profile == .resident_e2_formal_f16_page_tsqr_v1 or profile == .resident_e2_formal_fp8_page_tsqr_v1 or profile == .resident_e2_formal_rotated4_page_tsqr_v1;
    if (!formalProfileLogicalTokensValid(profile, record.logical_tokens))
        return error.InvalidPreparedServingQualityClosure;
    const resident_e4 = profile == .resident_f16_fixed91_batch64_workspace2gib or profile == .resident_fp8_fixed91_batch64_workspace2gib or profile == .resident_rotated4_fixed91_batch64_workspace2gib or resident_e4_bounded;
    const resident_e2 = profile == .resident_e2_f16_fixed60_batch64_workspace2gib or profile == .resident_e2_fp8_fixed60_batch64_workspace2gib or profile == .resident_e2_rotated4_fixed60_batch64_workspace2gib or resident_e2_bounded;
    const resident = resident_e4 or resident_e2;
    const formal_bounded = resident_e4_bounded or resident_e2_bounded;
    const bounded = profile == .private_hybrid_bounded_batch64_workspace1gib or profile == .private_hybrid_bounded_accelerated_batch64_workspace1gib or formal_bounded;
    const formal_bounded_complete = formal_bounded and record.v18_bounded_heads != 0 and record.v18_bounded_heads == record.v18_accepted_heads and
        record.v18_bounded_delta_heads == record.v18_bounded_heads and record.v18_bounded_delta_tiles != 0 and
        record.v18_bounded_delta_backend_calls == (std.math.mul(usize, record.v18_bounded_delta_tiles, 2) catch return error.InvalidPreparedServingQualityClosure) and
        record.v18_bounded_delta_packed_bytes_peak != 0 and record.v18_bounded_delta_staged_bytes_peak != 0 and record.v18_bounded_delta_fallbacks == 0;
    const effective_heads = std.math.add(usize, total_heads, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    const v18_total_heads = std.math.add(usize, std.math.add(usize, record.v18_accepted_heads, record.v18_preserved_heads) catch return error.InvalidPreparedServingQualityClosure, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    const v18_ready_heads = std.math.add(usize, std.math.add(usize, record.v18_cache_ready_heads, record.v18_fallback_heads) catch return error.InvalidPreparedServingQualityClosure, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    const expected_chooser_calls = std.math.mul(usize, record.cache_ready_heads, 3) catch return error.InvalidPreparedServingQualityClosure;
    const encoding = std.meta.stringToEnum(checkpoint.Encoding, record.encoding) orelse return error.InvalidPreparedServingQualityClosure;
    if (!std.mem.eql(u8, record.schema, schema) or (encoding != .f16 and encoding != .fp8 and encoding != .rotated4) or record.logical_tokens == 0 or
        record.source_global_bytes == 0 or record.candidate_global_bytes == 0 or candidate_scaled > source_scaled or
        record.validated_heads == 0 or total_heads == 0 or (record.completed_chunks == 0 and !formal_bounded_complete) or !std.math.isFinite(record.max_head_mass_error) or !std.math.isFinite(record.max_head_output_rmse) or
        record.max_head_mass_error < 0 or record.max_head_output_rmse < 0 or record.max_head_mass_error > 0.05 or record.max_head_output_rmse > 0.05 or
        record.cache_ready_heads != total_heads or record.exact_f32_probability_heads != total_heads or
        record.v18_eligible_heads != effective_heads or record.v18_attempted_heads != effective_heads or
        v18_total_heads != effective_heads or v18_ready_heads != effective_heads or
        record.incremental_gram_chooser_calls != expected_chooser_calls or
        (!formal_bounded and (record.training_reduction_local_original_calls == 0 or record.training_reduction_rounded_harm_calls == 0 or record.training_reduction_tiles == 0 or
            record.training_reduction_backend_calls != try std.math.mul(usize, record.training_reduction_tiles, 2) or record.training_reduction_workspace_peak_bytes == 0)) or
        (formal_bounded and (record.training_reduction_local_original_calls != 0 or record.training_reduction_rounded_harm_calls != 0 or record.training_reduction_tiles != 0 or
            record.training_reduction_backend_calls != 0 or record.training_reduction_workspace_peak_bytes != 0 or !formal_bounded_complete)) or !record.exact_local_window_capacity or
        !record.request_private_publish_complete or !record.validation_complete)
        return error.InvalidPreparedServingQualityClosure;
    inline for (.{ record.prompt_tokens_sha256, record.source_checkpoint_sha256, record.profile_sha256, record.profile_semantic_sha256, record.implementation_semantic_sha256, record.binary_sha256, record.policy_sha256, record.checkpoint_layout_sha256, record.v18_policy_sha256, record.v18_evidence_profile_sha256 }) |value|
        if (!hex64(&value)) return error.InvalidPreparedServingQualityClosure;
    if (resident and record.future_training_authority == null) return error.InvalidPreparedServingQualityClosure;
    if (record.future_training_authority) |authority| {
        inline for (.{ authority.policy_sha256, authority.source_identity_sha256, authority.source_tokens_sha256, authority.generated_ids_sha256, authority.primary_queries_sha256, authority.historical_queries_sha256, authority.future_queries_sha256, authority.original_heldout_sha256 }) |value|
            if (!hex64(&value)) return error.InvalidPreparedServingQualityClosure;
        const observed = try digestHex(authority.policy_sha256);
        const policies = @import("../runtime/kv/attention_match_future_decode_training.zig");
        const legacy = std.mem.eql(u8, &observed, &policies.policyDigest());
        const resident_policy = std.mem.eql(u8, &observed, &policies.residentHybridPolicyDigest());
        const resident_e2_policy = std.mem.eql(u8, &observed, &policies.residentE2HybridPolicyDigest());
        if ((!legacy and !resident_policy and !resident_e2_policy) or authority.source_revision == 0 or
            (resident_e4 and !resident_policy) or (resident_e2 and !resident_e2_policy) or (!resident and !legacy))
            return error.InvalidPreparedServingQualityClosure;
    }
    const observed_profile_digest = try digestHex(record.v18_evidence_profile_sha256);
    const expected_profile_digest = @import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(profile);
    if (!std.mem.eql(u8, &observed_profile_digest, &expected_profile_digest)) return error.InvalidPreparedServingQualityClosure;
    const quality = @import("../runtime/kv/metal_checkpoint_quality.zig");
    if (resident_e4 and (record.v18_query_batch_rows != 64 or record.v18_planned_head_workspace_bytes != 2 * 1024 * 1024 * 1024 or
        record.v18_primary_budget_percent != 91 or record.v18_budget_fallback_heads > 3 or
        record.v18_eligible_heads != 8 or record.v18_attempted_heads != 8 or record.v18_accepted_heads != 7 or
        record.v18_preserved_heads != 0 or record.v18_fallback_heads != 0 or record.v18_cache_ready_heads != 7 or
        record.v18_planned_exact_heads != 1 or record.validated_heads != 7 or record.preserved_heads != 0 or
        record.exact_f32_probability_heads != 7 or record.incremental_gram_chooser_calls != 21))
        return error.InvalidPreparedServingQualityClosure;
    if (resident_e2 and (record.v18_query_batch_rows != 64 or record.v18_planned_head_workspace_bytes != 2 * 1024 * 1024 * 1024 or
        record.v18_primary_budget_percent != 60 or record.v18_budget_fallback_heads != 0 or
        record.v18_eligible_heads != 3 or record.v18_attempted_heads != 3 or record.v18_accepted_heads != 2 or
        record.v18_preserved_heads != 0 or record.v18_fallback_heads != 0 or record.v18_cache_ready_heads != 2 or
        record.v18_planned_exact_heads != 1 or record.validated_heads != 2 or record.preserved_heads != 0 or
        record.exact_f32_probability_heads != 2 or record.incremental_gram_chooser_calls != 6))
        return error.InvalidPreparedServingQualityClosure;
    switch (encoding) {
        .fp8 => if (resident) {
            const codec = record.codec_evidence orelse return error.InvalidPreparedServingQualityClosure;
            const covered_heads = std.math.add(usize, codec.fp8_heads, codec.fp8_fallback_heads) catch return error.InvalidPreparedServingQualityClosure;
            const covered_rows = std.math.add(usize, codec.fp8_encoded_rows, codec.fp8_exact_rows) catch return error.InvalidPreparedServingQualityClosure;
            if (codec.fp8_heads == 0 or covered_heads != record.validated_heads or codec.fp8_eligible_rows == 0 or
                codec.fp8_encoded_rows == 0 or covered_rows != codec.fp8_eligible_rows or
                codec.rotated4_heads != 0 or codec.rotated4_fallback_heads != 0 or codec.rotated4_eligible_rows != 0 or codec.rotated4_compressed_rows != 0 or codec.rotated4_exact_rows != 0)
                return error.InvalidPreparedServingQualityClosure;
        },
        .rotated4 => if (resident) {
            const codec = record.codec_evidence orelse return error.InvalidPreparedServingQualityClosure;
            const covered_heads = std.math.add(usize, codec.rotated4_heads, codec.rotated4_fallback_heads) catch return error.InvalidPreparedServingQualityClosure;
            const covered_rows = std.math.add(usize, codec.rotated4_compressed_rows, codec.rotated4_exact_rows) catch return error.InvalidPreparedServingQualityClosure;
            if (codec.rotated4_heads == 0 or covered_heads != record.validated_heads or
                covered_rows != codec.rotated4_eligible_rows or
                (codec.rotated4_heads > 0) != (codec.rotated4_compressed_rows > 0) or codec.rotated4_eligible_rows < record.validated_heads or
                codec.fp8_heads != 0 or codec.fp8_fallback_heads != 0 or codec.fp8_eligible_rows != 0 or codec.fp8_encoded_rows != 0 or codec.fp8_exact_rows != 0)
                return error.InvalidPreparedServingQualityClosure;
        },
        .f16 => if (record.codec_evidence != null) return error.InvalidPreparedServingQualityClosure,
        else => return error.InvalidPreparedServingQualityClosure,
    }
    if (!resident and record.v18_planned_exact_heads != 0) return error.InvalidPreparedServingQualityClosure;
    if (bounded) {
        const codec_name = record.v18_codec_physical_profile orelse return error.InvalidPreparedServingQualityClosure;
        const codec_profile = std.meta.stringToEnum(quality.V18EvidenceProfile, codec_name) orelse return error.InvalidPreparedServingQualityClosure;
        const bounded_codec_profile = codec_profile == .bounded_f16_codec_physical_v1 or codec_profile == .bounded_fp8_codec_physical_v1 or codec_profile == .bounded_rotated4_codec_physical_v1;
        const codec_digest = try digestHex(record.v18_codec_physical_profile_sha256);
        const tsqr_digest = try digestHex(record.v18_bounded_tsqr_policy_sha256);
        const expected_codec_digest = quality.v18EvidenceProfileDigest(codec_profile);
        const expected_tsqr_digest = quality.boundedTsqrValuePolicyDigest();
        if (!bounded_codec_profile or !hex64(&record.v18_codec_physical_profile_sha256) or !hex64(&record.v18_bounded_tsqr_policy_sha256) or
            !std.mem.eql(u8, &codec_digest, &expected_codec_digest) or
            !quality.physicalProfileEncodingMatches(codec_profile, encoding) or
            !std.mem.eql(u8, &tsqr_digest, &expected_tsqr_digest) or
            ((!resident_e4_bounded and !resident_e2_bounded) and (record.v18_primary_budget_percent != 0 or record.v18_budget_fallback_heads != 0)) or
            (resident_e4_bounded and (record.v18_primary_budget_percent != 91 or record.v18_budget_fallback_heads > 3)) or
            (resident_e2_bounded and (record.v18_primary_budget_percent != 60 or record.v18_budget_fallback_heads != 0)) or
            record.v18_bounded_tsqr_heads != record.v18_bounded_heads or record.v18_bounded_tsqr_heads == 0 or record.v18_bounded_tsqr_calls < record.v18_bounded_tsqr_heads or
            record.v18_bounded_tsqr_workspace_peak_bytes == 0 or record.v18_bounded_tsqr_workspace_peak_bytes > record.v18_planned_head_workspace_bytes or record.v18_bounded_tsqr_cancellation_checks < record.v18_bounded_tsqr_calls)
            return error.InvalidPreparedServingQualityClosure;
    } else if (record.v18_codec_physical_profile != null or !std.mem.allEqual(u8, &record.v18_codec_physical_profile_sha256, '0') or !std.mem.allEqual(u8, &record.v18_bounded_tsqr_policy_sha256, '0') or record.v18_bounded_tsqr_heads != 0 or record.v18_bounded_tsqr_calls != 0 or record.v18_bounded_tsqr_workspace_peak_bytes != 0 or record.v18_bounded_tsqr_cancellation_checks != 0) {
        return error.InvalidPreparedServingQualityClosure;
    } else if (!quality.physicalProfileEncodingMatches(profile, encoding)) return error.InvalidPreparedServingQualityClosure;
}

fn globalHeadCount(state: checkpoint.State) !usize {
    var count: usize = 0;
    for (state.donors) |donor| {
        if (donor.window != 0) continue;
        count = std.math.add(usize, count, donor.heads.len) catch return error.InvalidPreparedServingQualityClosure;
    }
    if (count == 0) return error.InvalidPreparedServingQualityClosure;
    return count;
}

test "context formal prepared closure profiles admit only their sealed logical lengths" {
    const quality = @import("../runtime/kv/metal_checkpoint_quality.zig");
    inline for (.{ quality.V18EvidenceProfile.resident_e4_32512_f16_page_tsqr_v1, .resident_e4_32512_fp8_page_tsqr_v1, .resident_e4_32512_rotated4_page_tsqr_v1 }) |profile| {
        try std.testing.expect(formalProfileLogicalTokensValid(profile, 32512));
        try std.testing.expect(!formalProfileLogicalTokensValid(profile, 7936));
        try std.testing.expect(!formalProfileLogicalTokensValid(profile, 8128));
    }
    inline for (.{ quality.V18EvidenceProfile.resident_e2_formal_f16_page_tsqr_v1, .resident_e2_formal_fp8_page_tsqr_v1, .resident_e2_formal_rotated4_page_tsqr_v1 }) |profile| {
        try std.testing.expect(formalProfileLogicalTokensValid(profile, 7936));
        try std.testing.expect(formalProfileLogicalTokensValid(profile, 32512));
        try std.testing.expect(!formalProfileLogicalTokensValid(profile, 8128));
    }
    try std.testing.expect(formalProfileLogicalTokensValid(.resident_e2_f16_fixed60_batch64_workspace2gib, 8128));
}

fn validateModelHeadCount(record: Record, source: checkpoint.State, candidate: checkpoint.State) !void {
    const expected = try globalHeadCount(source);
    const record_heads = std.math.add(usize, std.math.add(usize, record.validated_heads, record.preserved_heads) catch return error.InvalidPreparedServingQualityClosure, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    const cache_heads = std.math.add(usize, record.cache_ready_heads, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    const probability_heads = std.math.add(usize, record.exact_f32_probability_heads, record.v18_planned_exact_heads) catch return error.InvalidPreparedServingQualityClosure;
    if (try globalHeadCount(candidate) != expected or
        record_heads != expected or
        cache_heads != expected or probability_heads != expected or
        record.v18_eligible_heads != expected or record.v18_attempted_heads != expected)
        return error.InvalidPreparedServingQualityClosure;
}

fn validateCodecGeometry(record: Record, candidate: checkpoint.State) !void {
    const codec = record.codec_evidence orelse return;
    var fp8_heads: usize = 0;
    var fp8_rows: usize = 0;
    var rotated4_heads: usize = 0;
    var rotated4_rows: usize = 0;
    for (candidate.donors) |donor| {
        if (donor.window != 0) continue;
        for (donor.heads) |head| {
            if (head.fp8.rows != 0) {
                fp8_heads += 1;
                fp8_rows = std.math.add(usize, fp8_rows, @intCast(head.fp8.rows)) catch return error.InvalidPreparedServingQualityClosure;
            }
            var packed_rows: usize = 0;
            for (head.compressed.spans) |span|
                packed_rows = std.math.add(usize, packed_rows, @intCast(span.rows)) catch return error.InvalidPreparedServingQualityClosure;
            if (packed_rows != 0) {
                rotated4_heads += 1;
                rotated4_rows = std.math.add(usize, rotated4_rows, packed_rows) catch return error.InvalidPreparedServingQualityClosure;
            }
        }
    }
    if (fp8_heads != codec.fp8_heads or fp8_rows != codec.fp8_encoded_rows or
        rotated4_heads != codec.rotated4_heads or rotated4_rows != codec.rotated4_compressed_rows)
        return error.InvalidPreparedServingQualityClosure;
}

test "context prepared quality head count is authenticated by model checkpoint layout" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const head: checkpoint.Head = .{};
    const e2_donors = [_]checkpoint.Donor{
        .{ .layer = 4, .window = 0, .heads = &.{head} },
        .{ .layer = 9, .window = 0, .heads = &.{head} },
        .{ .layer = 14, .window = 0, .heads = &.{head} },
        .{ .layer = 1, .window = 1024, .heads = &.{ head, head } },
    };
    const e4_donors = [_]checkpoint.Donor{.{ .layer = 5, .window = 0, .heads = &.{ head, head, head, head, head, head, head, head } }};
    const e2: checkpoint.State = .{ .identity = identity, .logical_length = 1, .donors = &e2_donors };
    const e4: checkpoint.State = .{ .identity = identity, .logical_length = 1, .donors = &e4_donors };
    var record: Record = undefined;
    record.validated_heads = 2;
    record.preserved_heads = 1;
    record.cache_ready_heads = 3;
    record.exact_f32_probability_heads = 3;
    record.v18_eligible_heads = 3;
    record.v18_attempted_heads = 3;
    try validateModelHeadCount(record, e2, e2);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, validateModelHeadCount(record, e2, e4));
    record.cache_ready_heads = 8;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, validateModelHeadCount(record, e2, e2));
    record.validated_heads = 7;
    record.cache_ready_heads = 8;
    record.exact_f32_probability_heads = 8;
    record.v18_eligible_heads = 8;
    record.v18_attempted_heads = 8;
    try validateModelHeadCount(record, e4, e4);
    record.validated_heads = 7;
    record.preserved_heads = 0;
    record.cache_ready_heads = 7;
    record.exact_f32_probability_heads = 7;
    record.v18_planned_exact_heads = 1;
    try validateModelHeadCount(record, e4, e4);
}

pub fn validate(record: Record) !void {
    try validateCore(record);
    inline for (.{ record.checkpoint_sha256, record.producer_sha256 }) |value|
        if (!hex64(&value)) return error.InvalidPreparedServingQualityClosure;
}

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !std.json.Parsed(Envelope) {
    var parsed = try std.json.parseFromSlice(Envelope, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    errdefer parsed.deinit();
    const actual = try digest(parsed.value.record);
    var expected: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&expected, &parsed.value.closure_sha256) catch return error.InvalidPreparedServingQualityClosure;
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidPreparedServingQualityClosure;
    return parsed;
}

pub const Prepared = struct {
    closure: std.json.Parsed(Envelope),
    verified: verified_sidecar.Verified,
    source_checkpoint: checkpoint.Owned,
    retained_input_bytes: usize,
    checkpoint_taken: bool = false,

    pub fn deinit(self: *Prepared) void {
        if (self.checkpoint_taken) {
            self.verified.producer.deinit();
        } else {
            self.verified.deinit();
            self.source_checkpoint.deinit();
        }
        self.closure.deinit();
        self.* = undefined;
    }

    pub fn takeCandidate(self: *Prepared) !@import("../runtime/kv/metal_checkpoint_quality.zig").PreparedServingCandidate {
        if (self.checkpoint_taken) return error.InvalidPreparedServingQualityCandidate;
        const record = self.closure.value.record;
        self.checkpoint_taken = true;
        return .{
            .source_checkpoint = self.source_checkpoint,
            .checkpoint = self.verified.decoded.checkpoint,
            .checkpoint_sha256 = try digestHex(record.checkpoint_sha256),
            .closure_sha256 = try digest(self.closure.value.record),
            .source_checkpoint_sha256 = try digestHex(record.source_checkpoint_sha256),
            .encoding = self.verified.decoded.encoding,
            .prompt_tokens_sha256 = try digestHex(record.prompt_tokens_sha256),
            .logical_tokens = record.logical_tokens,
            .source_global_bytes = record.source_global_bytes,
            .candidate_global_bytes = record.candidate_global_bytes,
            .validated_heads = record.validated_heads,
            .preserved_heads = record.preserved_heads,
            .completed_chunks = record.completed_chunks,
            .max_head_mass_error = record.max_head_mass_error,
            .max_head_output_rmse = record.max_head_output_rmse,
            .exact_local_window_capacity = record.exact_local_window_capacity,
            .retained_input_bytes = self.retained_input_bytes,
            .future_training_authority_sha256 = if (record.future_training_authority != null) try futureAuthorityDigest(record.future_training_authority.?) else @splat(0),
            .producer_v18 = .{
                .policy_sha256 = try digestHex(record.v18_policy_sha256),
                .eligible_heads = record.v18_eligible_heads,
                .attempted_heads = record.v18_attempted_heads,
                .accepted_heads = record.v18_accepted_heads,
                .preserved_heads = record.v18_preserved_heads,
                .fallback_heads = record.v18_fallback_heads,
                .cache_ready_heads = record.v18_cache_ready_heads,
                .planned_exact_heads = record.v18_planned_exact_heads,
                .exact_f32_probability_heads = record.exact_f32_probability_heads,
                .incremental_gram_chooser_calls = record.incremental_gram_chooser_calls,
                .query_batch_rows = record.v18_query_batch_rows,
                .planned_head_workspace_bytes = record.v18_planned_head_workspace_bytes,
                .cache_phase_peak_bound_bytes = record.v18_cache_phase_peak_bound_bytes,
                .bounded_heads = record.v18_bounded_heads,
                .bounded_delta_heads = record.v18_bounded_delta_heads,
                .bounded_delta_tiles = record.v18_bounded_delta_tiles,
                .bounded_delta_backend_calls = record.v18_bounded_delta_backend_calls,
                .bounded_delta_packed_bytes_peak = record.v18_bounded_delta_packed_bytes_peak,
                .bounded_delta_staged_bytes_peak = record.v18_bounded_delta_staged_bytes_peak,
                .bounded_delta_fallbacks = record.v18_bounded_delta_fallbacks,
                .primary_budget_percent = record.v18_primary_budget_percent,
                .budget_fallback_heads = record.v18_budget_fallback_heads,
                .evidence_profile = std.meta.stringToEnum(@import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile, record.v18_evidence_profile) orelse return error.InvalidPreparedServingQualityClosure,
                .evidence_profile_sha256 = try digestHex(record.v18_evidence_profile_sha256),
                .codec_physical_profile = if (record.v18_codec_physical_profile) |value| std.meta.stringToEnum(@import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile, value) orelse return error.InvalidPreparedServingQualityClosure else .none,
                .codec_physical_profile_sha256 = if (record.v18_codec_physical_profile != null) try digestHex(record.v18_codec_physical_profile_sha256) else @splat(0),
                .bounded_tsqr_policy_sha256 = if (record.v18_codec_physical_profile != null) try digestHex(record.v18_bounded_tsqr_policy_sha256) else @splat(0),
                .bounded_tsqr_heads = record.v18_bounded_tsqr_heads,
                .bounded_tsqr_calls = record.v18_bounded_tsqr_calls,
                .bounded_tsqr_workspace_peak_bytes = record.v18_bounded_tsqr_workspace_peak_bytes,
                .bounded_tsqr_cancellation_checks = record.v18_bounded_tsqr_cancellation_checks,
            },
        };
    }
};

fn futureAuthorityDigest(authority: FutureTrainingAuthority) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    inline for (.{ authority.policy_sha256, authority.source_identity_sha256, authority.source_tokens_sha256, authority.generated_ids_sha256, authority.primary_queries_sha256, authority.historical_queries_sha256, authority.future_queries_sha256, authority.original_heldout_sha256 }) |value| {
        if (!hex64(&value)) return error.InvalidPreparedServingQualityClosure;
        hash.update(&value);
    }
    var revision: [8]u8 = undefined;
    std.mem.writeInt(u64, &revision, authority.source_revision, .little);
    hash.update(&revision);
    var eos: [2]u8 = undefined;
    std.mem.writeInt(u16, &eos, authority.early_eos_observations, .little);
    hash.update(&eos);
    return hash.finalResult();
}

/// Decode and authenticate the exact public AFATT002 bytes before the quality
/// worker allocates a candidate owner or submits Metal work.
pub fn verifyPrepared(
    allocator: std.mem.Allocator,
    frame: []const u8,
    source_wire: []const u8,
    closure_bytes: []const u8,
    identity: checkpoint.Identity,
    producer: verified_sidecar.ExpectedProducer,
    prompt_tokens_sha256: [32]u8,
) !Prepared {
    const retained_input_bytes = try retainedInputBytes(frame.len, source_wire.len, closure_bytes.len);
    var closure = try parse(allocator, closure_bytes);
    errdefer closure.deinit();
    var verified = try verified_sidecar.verifyFrame(allocator, frame, identity, producer, checkpoint.serving_limits);
    errdefer verified.deinit();
    const record = closure.value.record;
    const expected_checkpoint = try digestHex(record.checkpoint_sha256);
    const expected_producer = try digestHex(record.producer_sha256);
    const expected_prompt = try digestHex(record.prompt_tokens_sha256);
    const actual_checkpoint = try digestHex(verified.sidecar.checkpoint_sha256_hex);
    const actual_producer = try digestHex(verified.sidecar.producer_sha256_hex);
    var actual_source: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source_wire, &actual_source, .{});
    const expected_source = try digestHex(record.source_checkpoint_sha256);
    if (!std.mem.eql(u8, &actual_source, &expected_source)) return error.InvalidPreparedServingQualityClosure;
    if (try checkpoint.encodedWireEncoding(source_wire) != .f16) return error.InvalidPreparedServingQualityClosure;
    var source_checkpoint = try checkpoint.decodeWithControl(allocator, source_wire, identity, checkpoint.serving_limits, .{});
    errdefer source_checkpoint.deinit();
    const profile = try digestHex(record.profile_sha256);
    const profile_semantic = try digestHex(record.profile_semantic_sha256);
    const implementation = try digestHex(record.implementation_semantic_sha256);
    const binary = try digestHex(record.binary_sha256);
    const policy = try digestHex(record.policy_sha256);
    const producer_core_text = verified.producer.producer.value.quality_evidence_core_sha256_hex orelse
        return error.InvalidPreparedServingQualityClosure;
    try verifyAuthenticatedCore(record, producer_core_text, verified.producer.producer.value.model_layout_sha256_hex);
    if (!std.mem.eql(u8, &expected_checkpoint, &actual_checkpoint) or
        !std.mem.eql(u8, &expected_producer, &actual_producer) or
        !std.mem.eql(u8, &expected_prompt, &prompt_tokens_sha256) or
        !std.mem.eql(u8, &profile, &producer.profile_sha256) or
        !std.mem.eql(u8, &profile_semantic, &producer.profile_semantic_sha256) or
        !std.mem.eql(u8, &implementation, &producer.implementation_semantic_sha256) or
        !std.mem.eql(u8, &binary, &producer.binary_sha256) or
        !std.mem.eql(u8, &policy, &producer.policy_sha256) or
        !std.mem.eql(u8, record.encoding, @tagName(verified.decoded.encoding)) or
        record.logical_tokens != verified.sidecar.logical_tokens)
        return error.InvalidPreparedServingQualityClosure;
    if (source_checkpoint.state.logical_length != record.logical_tokens or source_checkpoint.state.tokens.len != record.logical_tokens)
        return error.InvalidPreparedServingQualityClosure;
    try validateModelHeadCount(record, source_checkpoint.state, verified.decoded.checkpoint.state);
    try validateCodecGeometry(record, verified.decoded.checkpoint.state);
    return .{ .closure = closure, .verified = verified, .source_checkpoint = source_checkpoint, .retained_input_bytes = retained_input_bytes };
}

fn retainedInputBytes(frame_bytes: usize, source_bytes: usize, closure_bytes: usize) !usize {
    const frame_and_source = std.math.add(usize, frame_bytes, source_bytes) catch
        return error.InvalidPreparedServingQualityClosure;
    return std.math.add(usize, frame_and_source, closure_bytes) catch
        return error.InvalidPreparedServingQualityClosure;
}

test "context prepared quality retained input bytes are checked before decode" {
    try std.testing.expectEqual(@as(usize, 21), try retainedInputBytes(5, 7, 9));
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, retainedInputBytes(std.math.maxInt(usize), 1, 0));
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, retainedInputBytes(std.math.maxInt(usize) - 1, 1, 1));
}

test "context prepared codec evidence matches decoded checkpoint geometry" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const values: [24]f32 = .{ -0.0, 0.000000059604644775390625, 65504, -65504, 1.0625, 2, 3, 4 } ++ @as([8]f32, @splat(1.125)) ++ @as([8]f32, @splat(1.0625));
    const state: checkpoint.State = .{
        .identity = identity,
        .logical_length = 3,
        .tokens = &.{ 0, 0, 0 },
        .logits = &.{0.1234567},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{.{ .dim = 8, .compacted_until = 2, .positions = &.{ 0, 1, 2 }, .coverage_ends = &.{ 1, 2, 3 }, .keys = &values, .values = &values, .bias = &.{ 0, 0.69, 0 }, .fp8 = .{ .first_row = 1, .rows = 1 } }} }},
    };
    const wire = try checkpoint.encodeWithEncoding(std.testing.allocator, state, checkpoint.serving_limits, .fp8, .{});
    defer std.testing.allocator.free(wire);
    var decoded = try checkpoint.decode(std.testing.allocator, wire, identity, checkpoint.serving_limits);
    defer decoded.deinit();
    var record: Record = undefined;
    try std.testing.expectEqual(@as(u32, 1), decoded.state.donors[0].heads[0].fp8.rows);
    record.codec_evidence = .{ .fp8_heads = 1, .fp8_eligible_rows = 3, .fp8_encoded_rows = 1, .fp8_exact_rows = 2 };
    try validateCodecGeometry(record, decoded.state);
    record.codec_evidence.?.fp8_encoded_rows = 2;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, validateCodecGeometry(record, decoded.state));
}

fn verifyAuthenticatedCore(record: Record, producer_core_text: []const u8, producer_layout_text: []const u8) !void {
    const producer_core = try digestText(producer_core_text);
    const actual_core = try coreDigest(record);
    const record_layout = try digestHex(record.checkpoint_layout_sha256);
    const producer_layout = try digestText(producer_layout_text);
    if (!std.mem.eql(u8, &producer_core, &actual_core) or !std.mem.eql(u8, &record_layout, &producer_layout))
        return error.InvalidPreparedServingQualityClosure;
}

fn digestHex(value: [64]u8) ![32]u8 {
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, &value) catch return error.InvalidPreparedServingQualityClosure;
    return result;
}

fn digestText(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidPreparedServingQualityClosure;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, value) catch return error.InvalidPreparedServingQualityClosure;
    return result;
}

test "context prepared serving closure rejects policy and artifact mutations" {
    const nonzero: [64]u8 = @splat('1');
    var record: Record = .{
        .checkpoint_sha256 = nonzero,
        .producer_sha256 = nonzero,
        .prompt_tokens_sha256 = nonzero,
        .source_checkpoint_sha256 = nonzero,
        .profile_sha256 = nonzero,
        .profile_semantic_sha256 = nonzero,
        .implementation_semantic_sha256 = nonzero,
        .binary_sha256 = nonzero,
        .policy_sha256 = nonzero,
        .checkpoint_layout_sha256 = nonzero,
        .encoding = "f16",
        .logical_tokens = 7936,
        .source_global_bytes = 130023424,
        .candidate_global_bytes = 86212608,
        .validated_heads = 7,
        .preserved_heads = 1,
        .completed_chunks = 21,
        .max_head_mass_error = 0.04,
        .max_head_output_rmse = 0.02,
        .cache_ready_heads = 8,
        .exact_f32_probability_heads = 8,
        .incremental_gram_chooser_calls = 24,
        .training_reduction_local_original_calls = 21,
        .training_reduction_rounded_harm_calls = 21,
        .training_reduction_tiles = 192,
        .training_reduction_backend_calls = 384,
        .training_reduction_workspace_peak_bytes = 1024,
        .exact_local_window_capacity = true,
        .v18_policy_sha256 = nonzero,
        .v18_eligible_heads = 8,
        .v18_attempted_heads = 8,
        .v18_accepted_heads = 7,
        .v18_preserved_heads = 1,
        .v18_fallback_heads = 0,
        .v18_cache_ready_heads = 8,
        .v18_query_batch_rows = 64,
        .v18_planned_head_workspace_bytes = 1024 * 1024 * 1024,
        .v18_cache_phase_peak_bound_bytes = 900 * 1024 * 1024,
        .v18_bounded_heads = 0,
        .v18_bounded_delta_heads = 0,
        .v18_bounded_delta_tiles = 0,
        .v18_bounded_delta_backend_calls = 0,
        .v18_bounded_delta_packed_bytes_peak = 0,
        .v18_bounded_delta_staged_bytes_peak = 0,
        .v18_bounded_delta_fallbacks = 0,
        .v18_primary_budget_percent = 91,
        .v18_budget_fallback_heads = 2,
        .v18_evidence_profile = "private_physical_budget_batch64_workspace1gib",
        .v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib), .lower),
        .request_private_publish_complete = true,
        .validation_complete = true,
    };
    _ = try digest(record);
    const authenticated_core = try coreDigest(record);
    const authenticated_core_text = std.fmt.bytesToHex(authenticated_core, .lower);
    try verifyAuthenticatedCore(record, &authenticated_core_text, &record.checkpoint_layout_sha256);
    record.source_global_bytes += 1;
    _ = try digest(record); // A caller can recompute the outer self-hash.
    const mutated_core = try coreDigest(record);
    try std.testing.expect(!std.mem.eql(u8, &authenticated_core, &mutated_core));
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, verifyAuthenticatedCore(record, &authenticated_core_text, &record.checkpoint_layout_sha256));
    record.source_global_bytes -= 1;
    record.max_head_mass_error = -0.001;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.max_head_mass_error = 0.04;
    record.candidate_global_bytes = 86682283;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.candidate_global_bytes = 86212608;
    record.training_reduction_backend_calls -= 1;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.training_reduction_backend_calls += 1;
    record.preserved_heads = 0;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.preserved_heads = 1;
    record.incremental_gram_chooser_calls -= 1;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.incremental_gram_chooser_calls += 1;
    record.profile_sha256[0] = '2';
    _ = try digest(record); // Outer hash can again be recomputed by an attacker.
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, verifyAuthenticatedCore(record, &authenticated_core_text, &record.checkpoint_layout_sha256));
    record.profile_sha256[0] = '1';
    var wrong_layout = record.checkpoint_layout_sha256;
    wrong_layout[0] = '2';
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, verifyAuthenticatedCore(record, &authenticated_core_text, &wrong_layout));

    record.validated_heads = 2;
    record.preserved_heads = 1;
    record.cache_ready_heads = 3;
    record.exact_f32_probability_heads = 3;
    record.incremental_gram_chooser_calls = 9;
    record.v18_eligible_heads = 3;
    record.v18_attempted_heads = 3;
    record.v18_accepted_heads = 2;
    record.v18_preserved_heads = 1;
    record.v18_cache_ready_heads = 3;
    _ = try digest(record);
    record.cache_ready_heads = 8;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.validated_heads = 7;
    record.preserved_heads = 1;
    record.cache_ready_heads = 8;
    record.exact_f32_probability_heads = 8;
    record.incremental_gram_chooser_calls = 24;
    record.v18_eligible_heads = 8;
    record.v18_attempted_heads = 8;
    record.v18_accepted_heads = 7;
    record.v18_preserved_heads = 1;
    record.v18_cache_ready_heads = 8;

    inline for (.{
        .{ "f16", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.private_physical_budget_batch64_workspace1gib },
        .{ "fp8", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.private_fp8_physical_budget_batch64_workspace1gib },
        .{ "rotated4", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.private_rotated4_physical_budget_batch64_workspace1gib },
    }) |row| {
        record.encoding = row[0];
        record.v18_evidence_profile = @tagName(row[1]);
        record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(row[1]), .lower);
        _ = try digest(record);
    }
    record.encoding = "fp8";
    record.v18_evidence_profile = "private_rotated4_physical_budget_batch64_workspace1gib";
    record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.private_rotated4_physical_budget_batch64_workspace1gib), .lower);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));

    record.v18_planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024;
    record.v18_primary_budget_percent = 91;
    record.validated_heads = 7;
    record.preserved_heads = 0;
    record.cache_ready_heads = 7;
    record.exact_f32_probability_heads = 7;
    record.incremental_gram_chooser_calls = 21;
    record.v18_accepted_heads = 7;
    record.v18_preserved_heads = 0;
    record.v18_cache_ready_heads = 7;
    record.v18_planned_exact_heads = 1;
    const future_policy = @import("../runtime/kv/attention_match_future_decode_training.zig");
    const resident_future = FutureTrainingAuthority{
        .policy_sha256 = std.fmt.bytesToHex(future_policy.residentHybridPolicyDigest(), .lower),
        .source_identity_sha256 = nonzero,
        .source_tokens_sha256 = nonzero,
        .source_revision = 1,
        .generated_ids_sha256 = nonzero,
        .early_eos_observations = 0,
        .primary_queries_sha256 = nonzero,
        .historical_queries_sha256 = nonzero,
        .future_queries_sha256 = nonzero,
        .original_heldout_sha256 = nonzero,
    };
    record.future_training_authority = resident_future;
    inline for (.{
        .{ "f16", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.resident_f16_fixed91_batch64_workspace2gib },
        .{ "fp8", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.resident_fp8_fixed91_batch64_workspace2gib },
        .{ "rotated4", @import("../runtime/kv/metal_checkpoint_quality.zig").V18EvidenceProfile.resident_rotated4_fixed91_batch64_workspace2gib },
    }) |row| {
        record.encoding = row[0];
        record.v18_evidence_profile = @tagName(row[1]);
        record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(row[1]), .lower);
        record.codec_evidence = if (std.mem.eql(u8, row[0], "fp8")) .{
            .fp8_heads = 1,
            .fp8_fallback_heads = 6,
            .fp8_eligible_rows = 100,
            .fp8_encoded_rows = 40,
            .fp8_exact_rows = 60,
        } else if (std.mem.eql(u8, row[0], "rotated4")) .{
            .rotated4_heads = 2,
            .rotated4_fallback_heads = 5,
            .rotated4_eligible_rows = 100,
            .rotated4_compressed_rows = 84,
            .rotated4_exact_rows = 16,
        } else null;
        _ = try digest(record);
        const valid_resident = record;
        if (record.codec_evidence) |*codec| {
            if (std.mem.eql(u8, row[0], "fp8")) codec.fp8_fallback_heads -= 1 else codec.rotated4_fallback_heads -= 1;
            try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
            record = valid_resident;
            if (record.codec_evidence) |*coverage| {
                if (std.mem.eql(u8, row[0], "fp8")) coverage.rotated4_heads = 1 else coverage.rotated4_compressed_rows -= 1;
            }
            try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
            record = valid_resident;
            record.codec_evidence = null;
            try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
            record = valid_resident;
        }
        record.future_training_authority = null;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.v18_accepted_heads = 6;
        record.v18_preserved_heads = 1;
        record.validated_heads = 6;
        record.preserved_heads = 1;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.v18_cache_ready_heads = 6;
        record.v18_fallback_heads = 1;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.source_global_bytes = 0;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.max_head_mass_error = 0.050001;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.v18_primary_budget_percent = 90;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
        record.v18_evidence_profile_sha256[0] = if (record.v18_evidence_profile_sha256[0] == '0') '1' else '0';
        try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
        record = valid_resident;
    }
    const resident_record = record;
    var wrong_resident_policy = resident_record;
    wrong_resident_policy.future_training_authority.?.policy_sha256 = std.fmt.bytesToHex(future_policy.policyDigest(), .lower);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(wrong_resident_policy));
    var unknown_resident_policy = resident_record;
    unknown_resident_policy.future_training_authority.?.policy_sha256 = @splat('a');
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(unknown_resident_policy));

    var legacy_record = resident_record;
    legacy_record.encoding = "f16";
    legacy_record.codec_evidence = null;
    legacy_record.validated_heads = 7;
    legacy_record.preserved_heads = 1;
    legacy_record.cache_ready_heads = 8;
    legacy_record.exact_f32_probability_heads = 8;
    legacy_record.incremental_gram_chooser_calls = 24;
    legacy_record.v18_accepted_heads = 7;
    legacy_record.v18_preserved_heads = 1;
    legacy_record.v18_cache_ready_heads = 8;
    legacy_record.v18_planned_exact_heads = 0;
    legacy_record.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
    legacy_record.v18_evidence_profile = "private_physical_budget_batch64_workspace1gib";
    legacy_record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib), .lower);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(legacy_record));
    legacy_record.future_training_authority.?.policy_sha256 = std.fmt.bytesToHex(future_policy.policyDigest(), .lower);
    _ = try digest(legacy_record);

    // Restore the resident tuple for the pre-existing resident mutation tests.
    record = resident_record;
    record.v18_primary_budget_percent = 90;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.v18_primary_budget_percent = 91;
    record.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.v18_planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024;
    record.v18_planned_exact_heads = 0;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.v18_planned_exact_heads = 2;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));

    record.future_training_authority = legacy_record.future_training_authority;
    record.codec_evidence = null;
    record.encoding = "fp8";
    record.validated_heads = 7;
    record.preserved_heads = 1;
    record.cache_ready_heads = 8;
    record.exact_f32_probability_heads = 8;
    record.incremental_gram_chooser_calls = 24;
    record.v18_preserved_heads = 1;
    record.v18_cache_ready_heads = 8;
    record.v18_planned_exact_heads = 0;
    record.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
    record.v18_evidence_profile = "private_hybrid_bounded_accelerated_batch64_workspace1gib";
    record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.private_hybrid_bounded_accelerated_batch64_workspace1gib), .lower);
    record.v18_bounded_heads = 8;
    record.v18_bounded_delta_heads = 8;
    record.v18_bounded_delta_tiles = 24;
    record.v18_bounded_delta_backend_calls = 48;
    record.v18_bounded_delta_packed_bytes_peak = 4096;
    record.v18_bounded_delta_staged_bytes_peak = 8192;
    record.v18_primary_budget_percent = 0;
    record.v18_budget_fallback_heads = 0;
    const quality = @import("../runtime/kv/metal_checkpoint_quality.zig");
    const emitted = .{
        .v18_codec_physical_profile = quality.V18EvidenceProfile.bounded_fp8_codec_physical_v1,
        .v18_codec_physical_profile_sha256 = quality.v18EvidenceProfileDigest(.bounded_fp8_codec_physical_v1),
        .v18_bounded_tsqr_policy_sha256 = quality.boundedTsqrValuePolicyDigest(),
        .v18_bounded_tsqr_heads = @as(usize, 8),
        .v18_bounded_tsqr_calls = @as(usize, 9), // A truthful retry is allowed.
        .v18_bounded_tsqr_workspace_peak_bytes = @as(usize, 4096),
        .v18_bounded_tsqr_cancellation_checks = @as(usize, 12),
    };
    applyBoundedSeal(&record, emitted);
    _ = try digest(record);
    record.v18_bounded_tsqr_calls = 0;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.v18_bounded_tsqr_calls = emitted.v18_bounded_tsqr_calls;
    record.v18_bounded_tsqr_policy_sha256[0] = if (record.v18_bounded_tsqr_policy_sha256[0] == '0') '1' else '0';
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
    record.v18_bounded_tsqr_policy_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").boundedTsqrValuePolicyDigest(), .lower);
    record.v18_codec_physical_profile = "bounded_rotated4_codec_physical_v1";
    record.v18_codec_physical_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.bounded_rotated4_codec_physical_v1), .lower);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, digest(record));
}

test "context verifyPrepared authenticates AFATT002 payload source and producer core" {
    const wire = @import("native_kv_wire.zig");
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var keys: [64]f32 = @splat(0.25);
    var values: [64]f32 = @splat(0.5);
    const state: checkpoint.State = .{
        .identity = identity,
        .logical_length = 1,
        .tokens = &.{0},
        .logits = &.{1},
        .layer_donors = &.{ 0, 0 },
        .donors = &.{.{ .heads = &.{.{ .dim = 64, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &keys, .values = &values, .bias = &.{0} }} }},
    };
    const source = try checkpoint.encodeWithEncoding(std.testing.allocator, state, checkpoint.serving_limits, .f16, .{});
    defer std.testing.allocator.free(source);
    var payload_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(source, &payload_sha, .{});
    const one: [64]u8 = @splat('1');
    const canonical_policy = @import("../runtime/kv/attention_match_fit_policy.zig").Policy.cumulative256_exact_cycle_v1.digest();
    const canonical_policy_text = std.fmt.bytesToHex(canonical_policy, .lower);
    var record: Record = .{
        .checkpoint_sha256 = std.fmt.bytesToHex(payload_sha, .lower),
        .producer_sha256 = one,
        .prompt_tokens_sha256 = one,
        .source_checkpoint_sha256 = std.fmt.bytesToHex(payload_sha, .lower),
        .profile_sha256 = one,
        .profile_semantic_sha256 = one,
        .implementation_semantic_sha256 = one,
        .binary_sha256 = one,
        .policy_sha256 = canonical_policy_text,
        .checkpoint_layout_sha256 = std.fmt.bytesToHex(identity.layout, .lower),
        .encoding = "f16",
        .logical_tokens = 1,
        .source_global_bytes = 300,
        .candidate_global_bytes = 200,
        .validated_heads = 1,
        .preserved_heads = 0,
        .completed_chunks = 3,
        .max_head_mass_error = 0.01,
        .max_head_output_rmse = 0.01,
        .cache_ready_heads = 1,
        .exact_f32_probability_heads = 1,
        .incremental_gram_chooser_calls = 3,
        .training_reduction_local_original_calls = 1,
        .training_reduction_rounded_harm_calls = 1,
        .training_reduction_tiles = 1,
        .training_reduction_backend_calls = 2,
        .training_reduction_workspace_peak_bytes = 1,
        .exact_local_window_capacity = true,
        .v18_policy_sha256 = one,
        .v18_eligible_heads = 1,
        .v18_attempted_heads = 1,
        .v18_accepted_heads = 1,
        .v18_preserved_heads = 0,
        .v18_fallback_heads = 0,
        .v18_cache_ready_heads = 1,
        .v18_query_batch_rows = 64,
        .v18_planned_head_workspace_bytes = 1024 * 1024 * 1024,
        .v18_cache_phase_peak_bound_bytes = 900 * 1024 * 1024,
        .v18_bounded_heads = 0,
        .v18_bounded_delta_heads = 0,
        .v18_bounded_delta_tiles = 0,
        .v18_bounded_delta_backend_calls = 0,
        .v18_bounded_delta_packed_bytes_peak = 0,
        .v18_bounded_delta_staged_bytes_peak = 0,
        .v18_bounded_delta_fallbacks = 0,
        .v18_primary_budget_percent = 91,
        .v18_budget_fallback_heads = 0,
        .v18_evidence_profile = "private_physical_budget_batch64_workspace1gib",
        .v18_evidence_profile_sha256 = std.fmt.bytesToHex(@import("../runtime/kv/metal_checkpoint_quality.zig").v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib), .lower),
        .request_private_publish_complete = true,
        .validation_complete = true,
    };
    const quality = @import("../runtime/kv/metal_checkpoint_quality.zig");
    record.v18_evidence_profile = "private_hybrid_bounded_accelerated_batch64_workspace1gib";
    record.v18_evidence_profile_sha256 = std.fmt.bytesToHex(quality.v18EvidenceProfileDigest(.private_hybrid_bounded_accelerated_batch64_workspace1gib), .lower);
    record.v18_bounded_heads = 1;
    record.v18_bounded_delta_heads = 1;
    record.v18_bounded_delta_tiles = 3;
    record.v18_bounded_delta_backend_calls = 6;
    record.v18_bounded_delta_packed_bytes_peak = 4096;
    record.v18_bounded_delta_staged_bytes_peak = 8192;
    record.v18_primary_budget_percent = 0;
    record.v18_budget_fallback_heads = 0;
    applyBoundedSeal(&record, .{
        .v18_codec_physical_profile = quality.V18EvidenceProfile.bounded_f16_codec_physical_v1,
        .v18_codec_physical_profile_sha256 = quality.v18EvidenceProfileDigest(.bounded_f16_codec_physical_v1),
        .v18_bounded_tsqr_policy_sha256 = quality.boundedTsqrValuePolicyDigest(),
        .v18_bounded_tsqr_heads = @as(usize, 1),
        .v18_bounded_tsqr_calls = @as(usize, 2),
        .v18_bounded_tsqr_workspace_peak_bytes = @as(usize, 4096),
        .v18_bounded_tsqr_cancellation_checks = @as(usize, 3),
    });
    const core_text = std.fmt.bytesToHex(try coreDigest(record), .lower);
    const weights_text = std.fmt.bytesToHex(identity.weights, .lower);
    const tokenizer_text = std.fmt.bytesToHex(identity.tokenizer, .lower);
    const template_text = std.fmt.bytesToHex(identity.template, .lower);
    const layout_text = std.fmt.bytesToHex(identity.layout, .lower);
    const producer = try std.json.Stringify.valueAlloc(std.testing.allocator, wire.PreparedProducer{
        .model = "fixture",
        .model_artifact_sha256_hex = &weights_text,
        .model_tokenizer_sha256_hex = &tokenizer_text,
        .model_template_sha256_hex = &template_text,
        .model_layout_sha256_hex = &layout_text,
        .profile_sha256_hex = &one,
        .profile_semantic_sha256_hex = &one,
        .source_sha256_hex = &one,
        .binary_sha256_hex = &one,
        .policy_name = "cumulative256_exact_cycle_v1",
        .policy_sha256_hex = &canonical_policy_text,
        .quality_evidence_core_sha256_hex = &core_text,
        .admission = "cumulative_ladder_4_3_2",
        .chunk_rows = 256,
        .reject_exact_active_fixed_point = true,
        .encoding = "f16",
        .logical_revision = 1,
        .logical_tokens = 1,
        .checkpoint_bytes = source.len,
        .outcome = .compacted,
    }, .{});
    defer std.testing.allocator.free(producer);
    var producer_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(producer, &producer_sha, .{});
    record.producer_sha256 = std.fmt.bytesToHex(producer_sha, .lower);
    const encoded_producer = try std.testing.allocator.alloc(u8, std.base64.standard.Encoder.calcSize(producer.len));
    defer std.testing.allocator.free(encoded_producer);
    _ = std.base64.standard.Encoder.encode(encoded_producer, producer);
    const metadata = try std.json.Stringify.valueAlloc(std.testing.allocator, wire.V2Metadata{ .prepared_metadata_base64 = encoded_producer }, .{});
    defer std.testing.allocator.free(metadata);
    const Sink = struct {
        bytes: [1024 * 1024]u8 = undefined,
        len: usize = 0,
        pub fn writeAll(self: *@This(), value: []const u8) !void {
            if (value.len > self.bytes.len - self.len) return error.NoSpaceLeft;
            @memcpy(self.bytes[self.len..][0..value.len], value);
            self.len += value.len;
        }
    };
    const Control = struct {
        pub fn check(_: *@This()) !void {}
        fn callback(_: ?*anyopaque) !void {}
    };
    var sink: Sink = .{};
    var control: Control = .{};
    _ = try wire.writeV2Start(&sink, metadata, source.len, .{}, &control);
    var hashing = wire.V2HashingWriter(Sink).init(&sink, metadata, source.len, null, Control.callback);
    try hashing.writeAll(source);
    _ = try hashing.finish();
    const closure_bytes = try std.json.Stringify.valueAlloc(std.testing.allocator, Envelope{ .record = record, .closure_sha256 = std.fmt.bytesToHex(try digest(record), .lower) }, .{});
    defer std.testing.allocator.free(closure_bytes);
    const expected: verified_sidecar.ExpectedProducer = .{
        .profile_sha256 = @splat(0x11),
        .profile_semantic_sha256 = @splat(0x11),
        .implementation_semantic_sha256 = @splat(0x11),
        .binary_sha256 = @splat(0x11),
        .policy_sha256 = canonical_policy,
        .policy_name = "cumulative256_exact_cycle_v1",
    };
    var prepared = try verifyPrepared(std.testing.allocator, sink.bytes[0..sink.len], source, closure_bytes, identity, expected, @splat(0x11));
    var candidate = try prepared.takeCandidate();
    try std.testing.expectEqual(quality.V18EvidenceProfile.private_hybrid_bounded_accelerated_batch64_workspace1gib, candidate.producer_v18.evidence_profile);
    try std.testing.expectEqual(quality.V18EvidenceProfile.bounded_f16_codec_physical_v1, candidate.producer_v18.codec_physical_profile);
    try std.testing.expectEqual(@as(usize, 2), candidate.producer_v18.bounded_tsqr_calls);
    var frozen: quality.Frozen = undefined;
    frozen.encoding = .f16;
    var report: quality.Report = .{ .allocator = std.testing.allocator, .frozen = frozen, .fit_strategy = .identity_chunk_coefficients_adaptive_v18 };
    candidate.producer_v18.apply(&report);
    try report.validateV18Evidence();
    candidate.deinit();
    prepared.deinit();
    var mutated = record;
    mutated.source_global_bytes += 1;
    const mutated_bytes = try std.json.Stringify.valueAlloc(std.testing.allocator, Envelope{ .record = mutated, .closure_sha256 = std.fmt.bytesToHex(try digest(mutated), .lower) }, .{});
    defer std.testing.allocator.free(mutated_bytes);
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, verifyPrepared(std.testing.allocator, sink.bytes[0..sink.len], source, mutated_bytes, identity, expected, @splat(0x11)));
    var wrong_expected = expected;
    wrong_expected.binary_sha256 = @splat(0x22);
    try std.testing.expectError(error.NativeKvProducerIdentityMismatch, verifyPrepared(std.testing.allocator, sink.bytes[0..sink.len], source, closure_bytes, identity, wrong_expected, @splat(0x11)));
    var wrong_identity = identity;
    wrong_identity.layout = @splat(9);
    try std.testing.expectError(error.KvCheckpointModelMismatch, verifyPrepared(std.testing.allocator, sink.bytes[0..sink.len], source, closure_bytes, wrong_identity, expected, @splat(0x11)));
    var changed_source = try std.testing.allocator.dupe(u8, source);
    defer std.testing.allocator.free(changed_source);
    changed_source[changed_source.len / 2] ^= 1;
    try std.testing.expectError(error.InvalidPreparedServingQualityClosure, verifyPrepared(std.testing.allocator, sink.bytes[0..sink.len], changed_source, closure_bytes, identity, expected, @splat(0x11)));
}
