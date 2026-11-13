// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const generation = @import("pipelines/generation.zig");
const checkpoint = @import("runtime/kv/attention_match_checkpoint.zig");

pub const Authority = enum { staged_candidate, qualified_private };
pub const Input = struct {
    authority: Authority,
    arm: generation.NativeCheckpointQualificationArm,
    fit_strategy: generation.NativeCheckpointPrepare.QualityFitStrategy,
    training_policy: generation.NativeCheckpointTrainingPolicy,
    handoff: ?generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff,
    encoding: checkpoint.Encoding,
    logical_tokens: usize,
    prepare_host_owner_capability: @import("runtime/kv/compaction_profile.zig").PrepareHostOwnerCapability = .none,
    resident_authority: ?generation.ResidentHybridAuthority = null,
};
pub const Resolved = struct {
    private_dynamic_budget: bool = true,
    staged_operator_strategy: bool = true,
    bounded_delta_accelerated: bool,
    compiled_prefill: bool,
    capture_summary: bool,
    incremental_budget: bool,
    training_reduction: bool,
    future_decode_training_v1: bool = false,
    resident_train_challenge_hybrid_v1: bool = false,
    exact_local_window_capacity: bool = true,

    pub fn digest(self: @This()) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("antfly.private-native-checkpoint-policy.v1\n");
        // Preserve the exact legacy v1 byte stream. Future training is a
        // conditional extension and contributes no field name when false.
        inline for (.{ "private_dynamic_budget", "staged_operator_strategy", "bounded_delta_accelerated", "compiled_prefill", "capture_summary", "incremental_budget", "training_reduction", "exact_local_window_capacity" }) |name| {
            hash.update(name);
            hash.update(if (@field(self, name)) "=1\n" else "=0\n");
        }
        if (self.future_decode_training_v1)
            hash.update("extension=future_decode_training_v1\n");
        if (self.resident_train_challenge_hybrid_v1)
            hash.update("extension=private_train_challenge_hybrid_resident_v1;head_workspace=2147483648;transient_host=4026531840;process_guard=8589934592;fit_workers=1;geometry=flat_gram_only;numerical=accepted_f32_gram_reuse_v1;critical=train_probability_gte_0_025_no_top1_v1;postfit=train_global_residual_range_mass_l2_exact_f32_cache_v1;chooser=incremental_gram_v1;validation=accelerate_fp64_summary_v1;performance=owned_capture_summary_incremental_reduction_critical025_v1;max_queries=768;adaptive_heads=7;planned_exact_heads=1\n");
        return hash.finalResult();
    }
};

pub fn resolve(input: Input) !Resolved {
    if (input.arm != .attention_matching or
        input.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or
        (input.training_policy != .private_dense512_reference_v1 and input.training_policy != .private_future_decode_mix_v1 and input.training_policy != .private_train_challenge_hybrid_resident_v1))
        return error.UnqualifiedCompactionArtifact;
    if (input.authority == .qualified_private and input.training_policy == .private_future_decode_mix_v1)
        return error.UnqualifiedCompactionArtifact;
    const resident = input.training_policy == .private_train_challenge_hybrid_resident_v1;
    const legacy_resident_candidate = resident and input.authority == .staged_candidate and
        input.prepare_host_owner_capability == .none and input.resident_authority == null;
    const typed_resident = resident and !legacy_resident_candidate;
    if (resident) {
        if (!legacy_resident_candidate and (input.prepare_host_owner_capability != .bounded_page_e4_resident_1152m_v1 or input.resident_authority == null or
            input.resident_authority.?.cohort != .e4_8128_v1 or input.encoding != .f16 or input.logical_tokens != 8128))
            return error.UnqualifiedCompactionArtifact;
        if (typed_resident) try input.resident_authority.?.validate();
    } else if (input.prepare_host_owner_capability != .none or input.resident_authority != null) return error.UnqualifiedCompactionArtifact;
    const handoff = input.handoff orelse return error.UnqualifiedCompactionArtifact;
    if (handoff.max_copy_bytes != 4 * 1024 * 1024 * 1024 or handoff.max_host_staging_bytes != 256 * 1024 * 1024)
        return error.InvalidQualificationCandidate;
    const v3 = handoff.max_logical_tokens == 8192 and handoff.layout_policy == .split_swa_permitted;
    const v4 = handoff.max_logical_tokens == 32768 and handoff.layout_policy == .full_history_exact;
    const resident_scoped = typed_resident and handoff.codec_scope == .f16_only and
        handoff.max_logical_tokens == 8192 and handoff.layout_policy == .full_history_exact;
    if ((!v3 and !v4 and !resident_scoped) or (!resident_scoped and handoff.codec_scope != .f16_fp8_rotated4)) return error.InvalidQualificationCandidate;
    const compiled = handoff.allows(input.encoding, input.logical_tokens);
    if (!compiled) return error.UnqualifiedCompactionArtifact;
    const capture_summary = input.logical_tokens <= 8192;
    return .{
        .bounded_delta_accelerated = input.encoding == .f16,
        .compiled_prefill = true,
        .capture_summary = capture_summary,
        .incremental_budget = capture_summary,
        .training_reduction = capture_summary,
        .future_decode_training_v1 = input.training_policy == .private_future_decode_mix_v1,
        .resident_train_challenge_hybrid_v1 = input.training_policy == .private_train_challenge_hybrid_resident_v1,
    };
}

test "resident policy requires typed capability and is identical for candidate and qualified authority" {
    const handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{ .max_copy_bytes = 4 * 1024 * 1024 * 1024, .max_host_staging_bytes = 256 * 1024 * 1024, .codec_scope = .f16_only, .max_logical_tokens = 8192, .layout_policy = .full_history_exact };
    var module: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&module, "e0a407063023d89c40f516270081012329a8981dc5c64de2dd8a985d8469afee");
    const resident_authority: generation.ResidentHybridAuthority = .{ .configuration_sha256 = @splat(1), .source_freeze_sha256 = @splat(2), .shape_authority_sha256 = @splat(3), .policy_module_sha256 = module };
    const common: Input = .{ .authority = .staged_candidate, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_train_challenge_hybrid_resident_v1, .handoff = handoff, .encoding = .f16, .logical_tokens = 8128, .prepare_host_owner_capability = .bounded_page_e4_resident_1152m_v1, .resident_authority = resident_authority };
    const candidate = try resolve(common);
    var qualified_input = common;
    qualified_input.authority = .qualified_private;
    const qualified = try resolve(qualified_input);
    try std.testing.expectEqualDeep(candidate, qualified);
    var forged = qualified_input;
    forged.prepare_host_owner_capability = .none;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, resolve(forged));
    const legacy_handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{ .max_copy_bytes = 4 * 1024 * 1024 * 1024, .max_host_staging_bytes = 256 * 1024 * 1024, .codec_scope = .f16_fp8_rotated4, .max_logical_tokens = 32768, .layout_policy = .full_history_exact };
    _ = try resolve(.{ .authority = .staged_candidate, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_train_challenge_hybrid_resident_v1, .handoff = legacy_handoff, .encoding = .f16, .logical_tokens = 8128 });
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, resolve(.{ .authority = .qualified_private, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_train_challenge_hybrid_resident_v1, .handoff = legacy_handoff, .encoding = .f16, .logical_tokens = 8128 }));
}

test "legacy runtime policy digest remains fixed when future extension is absent" {
    const legacy: Resolved = .{
        .bounded_delta_accelerated = false,
        .compiled_prefill = true,
        .capture_summary = false,
        .incremental_budget = false,
        .training_reduction = false,
        .future_decode_training_v1 = false,
    };
    var expected: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&expected, "c9efdae1a41f0bed4f2fdbee5bbbd0d0385ea0304950054024f01e8084b72c8c");
    const legacy_digest = legacy.digest();
    try std.testing.expectEqualSlices(u8, &expected, &legacy_digest);
    var future = legacy;
    future.future_decode_training_v1 = true;
    const future_digest = future.digest();
    try std.testing.expect(!std.mem.eql(u8, &legacy_digest, &future_digest));
}

test "context v4 staged and qualified private resolve identical 32K policy" {
    const handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{
        .max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .max_host_staging_bytes = 256 * 1024 * 1024,
        .codec_scope = .f16_fp8_rotated4,
        .max_logical_tokens = 32768,
        .layout_policy = .full_history_exact,
    };
    inline for (.{ checkpoint.Encoding.f16, .fp8, .rotated4 }) |encoding| {
        const staged = try resolve(.{ .authority = .staged_candidate, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_dense512_reference_v1, .handoff = handoff, .encoding = encoding, .logical_tokens = 32768 });
        const qualified = try resolve(.{ .authority = .qualified_private, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_dense512_reference_v1, .handoff = handoff, .encoding = encoding, .logical_tokens = 32768 });
        try std.testing.expectEqualDeep(staged, qualified);
        try std.testing.expectEqual(staged.digest(), qualified.digest());
        try std.testing.expect(!qualified.capture_summary);
    }
}

test "context staged and qualified private resolve identical sealed policy" {
    const handoff: generation.NativeCheckpointPrepare.CompiledPrefillExactKvHandoff = .{
        .max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .max_host_staging_bytes = 256 * 1024 * 1024,
        .codec_scope = .f16_fp8_rotated4,
        .max_logical_tokens = 8192,
    };
    inline for (.{ checkpoint.Encoding.f16, .fp8, .rotated4 }) |encoding| {
        const staged = try resolve(.{ .authority = .staged_candidate, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_dense512_reference_v1, .handoff = handoff, .encoding = encoding, .logical_tokens = 8192 });
        const qualified = try resolve(.{ .authority = .qualified_private, .arm = .attention_matching, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .training_policy = .private_dense512_reference_v1, .handoff = handoff, .encoding = encoding, .logical_tokens = 8192 });
        try std.testing.expectEqualDeep(staged, qualified);
        try std.testing.expectEqual(staged.digest(), qualified.digest());
    }
}
