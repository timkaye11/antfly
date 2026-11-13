// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test-driver contracts for independent native whole-head answer and NLL
//! measurements. These receipts never confer serving or release authority.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const teacher = @import("../../backends/context_teacher_forced.zig");
const allocations = @import("host_allocation_tracker.zig");
const queries = @import("attention_match_queries.zig");
const native_release = @import("attention_match_release.zig");
const engine = @import("attention_match_engine.zig");
const future_augmentation = @import("attention_match_future_augmentation.zig");

pub const max_output_tokens = 256;
pub const accepted_snapshot_name = "accepted-candidate.f16-v3";

pub const FitStrategy = enum { legacy, identity_chunk_coefficients_adaptive_v18 };

/// Test-only destination for retaining the exact fitted wire after every
/// numerical and task gate has accepted it. The directory is already the
/// sealed native-quality output scope; the filename is intentionally fixed.
pub const AcceptedSnapshotCapture = struct {
    output: std.Io.Dir,
};

pub const AcceptedSnapshot = struct {
    path: []const u8,
    bytes: usize,
    sha256: [32]u8,
};

pub fn writeAcceptedSnapshot(
    io: std.Io,
    capture: AcceptedSnapshotCapture,
    wire: []const u8,
    expected_sha256: [32]u8,
) !AcceptedSnapshot {
    if (wire.len == 0 or wire.len > checkpoint.serving_limits.max_file_bytes)
        return error.KvCheckpointLimit;
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(wire, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected_sha256)) return error.KvCheckpointResumeMismatch;
    var file = try capture.output.createFile(io, accepted_snapshot_name, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, wire);
    try file.sync(io);
    return .{ .path = accepted_snapshot_name, .bytes = wire.len, .sha256 = actual };
}

pub const Request = struct {
    pub const ResidentHybridAuthority = struct {
        pub const Cohort = enum { e4_8128_v1, e2_8128_v1, e4_formal_7936_256_v1, e4_formal_32512_256_v1 };
        configuration_sha256: [32]u8,
        source_freeze_sha256: [32]u8,
        shape_authority_sha256: [32]u8,
        policy_module_sha256: [32]u8,
        cohort: Cohort = .e4_8128_v1,
        e4_formal_authority: ?@import("attention_match_e4_formal_authority.zig").Authority = null,
        e2_guarded_development_trial: bool = false,
        e2_fit_source_rows: usize = 8128,
        e2_formal_owner_inventory_sha256: [32]u8 = @splat(0),
        e2_formal_owner_inventory: ?@import("attention_match_e2_formal_owner_inventory.zig").Artifact = null,
    };
    encoding: checkpoint.Encoding,
    output_tokens: usize = max_output_tokens,
    minimum_savings_ratio: f64 = 1.5,
    admission: engine.Options.Admission = .fixed,
    chunk_rows: u16 = 1024,
    reject_exact_active_fixed_point: bool = false,
    fit_strategy: FitStrategy = .legacy,
    /// Authenticated test-worker mirror of the staged private operator tuple.
    /// This cannot be set by public requests or the legacy quality manifest.
    private_dense512_policy: bool = false,
    /// Staged-only replacement mixture: primary256 + independent256 +
    /// exact-source future-decode256. Default false and unqualified.
    private_future_decode_training_v1: bool = false,
    /// Exact staged resident mode. Authority is copied only from the
    /// authenticated candidate-v5 binding by the quality worker.
    private_resident_hybrid_policy: bool = false,
    resident_hybrid_authority: ?ResidentHybridAuthority = null,
    /// Diagnostic-only comparison of raw and prepared continuations cloned
    /// from the same authenticated exact source. It never changes fitting,
    /// publication, or qualification decisions.
    route_parity_diagnostic: bool = false,

    pub fn capacity(self: Request, prompt_tokens: usize, model_limit: usize) !usize {
        if (self.encoding == .f32 or self.output_tokens == 0 or self.output_tokens > max_output_tokens or
            !std.math.isFinite(self.minimum_savings_ratio) or self.minimum_savings_ratio < 1.5 or self.minimum_savings_ratio > 4)
            return error.InvalidNativeQualityRequest;
        if (self.admission == .fixed and self.chunk_rows != 1024) return error.InvalidNativeQualityRequest;
        if (self.reject_exact_active_fixed_point and self.admission != .cumulative_ladder_4_3_2)
            return error.InvalidNativeQualityRequest;
        if (self.admission == .cumulative_ladder_4_3_2 and
            ((self.encoding != .f16 and self.encoding != .fp8 and self.encoding != .rotated4) or
                self.minimum_savings_ratio != 1.5 or (self.chunk_rows != 128 and self.chunk_rows != 256 and self.chunk_rows != 1024)))
            return error.InvalidNativeQualityRequest;
        if (self.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and
            (!@import("builtin").is_test or self.admission != .cumulative_ladder_4_3_2 or self.chunk_rows != 256 or
                !self.reject_exact_active_fixed_point or (self.encoding != .f16 and self.encoding != .fp8 and self.encoding != .rotated4)))
            return error.InvalidNativeQualityRequest;
        if (self.private_dense512_policy and
            (self.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or
                self.admission != .cumulative_ladder_4_3_2 or self.chunk_rows != 256 or
                !self.reject_exact_active_fixed_point))
            return error.InvalidNativeQualityRequest;
        if (self.private_future_decode_training_v1 and
            (!@import("builtin").is_test or !self.private_dense512_policy or self.route_parity_diagnostic))
            return error.InvalidNativeQualityRequest;
        if (self.private_resident_hybrid_policy and
            (!@import("builtin").is_test or !self.private_dense512_policy or self.private_future_decode_training_v1 or
                self.route_parity_diagnostic or self.resident_hybrid_authority == null))
            return error.InvalidNativeQualityRequest;
        if (!self.private_resident_hybrid_policy and self.resident_hybrid_authority != null)
            return error.InvalidNativeQualityRequest;
        if (self.resident_hybrid_authority) |authority| inline for (.{ authority.configuration_sha256, authority.source_freeze_sha256, authority.shape_authority_sha256, authority.policy_module_sha256 }) |digest|
            if (std.mem.allEqual(u8, &digest, 0)) return error.InvalidNativeQualityRequest;
        if (self.resident_hybrid_authority) |authority| {
            const expected_formal_cohort: ?@import("attention_match_e4_formal_authority.zig").Cohort = switch (authority.cohort) {
                .e4_formal_7936_256_v1 => .formal_7936_256_v1,
                .e4_formal_32512_256_v1 => .formal_32512_256_v1,
                else => null,
            };
            if (expected_formal_cohort) |formal_cohort| {
                const formal = authority.e4_formal_authority orelse return error.InvalidNativeQualityRequest;
                formal.validate() catch return error.InvalidNativeQualityRequest;
                if (formal.cohort != formal_cohort or
                    !std.mem.eql(u8, &formal.configuration_sha256, &authority.configuration_sha256) or
                    !std.mem.eql(u8, &formal.source_freeze_sha256, &authority.source_freeze_sha256) or
                    !std.mem.eql(u8, &formal.shape_authority_sha256, &authority.shape_authority_sha256) or
                    !std.mem.eql(u8, &formal.policy_module_sha256, &authority.policy_module_sha256))
                    return error.InvalidNativeQualityRequest;
            } else if (authority.e4_formal_authority != null) return error.InvalidNativeQualityRequest;
            if (authority.e2_fit_source_rows == 8128) {
                if (authority.e2_formal_owner_inventory != null or !std.mem.allEqual(u8, &authority.e2_formal_owner_inventory_sha256, 0))
                    return error.InvalidNativeQualityRequest;
            } else {
                if (authority.cohort != .e2_8128_v1 or !authority.e2_guarded_development_trial or
                    (authority.e2_fit_source_rows != 7_936 and authority.e2_fit_source_rows != 32_512)) return error.InvalidNativeQualityRequest;
                const inventory = authority.e2_formal_owner_inventory orelse return error.InvalidNativeQualityRequest;
                inventory.validate() catch return error.InvalidNativeQualityRequest;
                if (!std.mem.eql(u8, &inventory.digest, &authority.e2_formal_owner_inventory_sha256)) return error.InvalidNativeQualityRequest;
            }
        }
        if (self.route_parity_diagnostic and (!@import("builtin").is_test or !self.private_dense512_policy))
            return error.InvalidNativeQualityRequest;
        const capacity_value = try std.math.add(usize, prompt_tokens, self.output_tokens);
        // The private query repeat may exceed the wire window, but must stay
        // inside the actual model's position limit. It is never published.
        const query_end = try std.math.add(usize, prompt_tokens, 512);
        if (prompt_tokens < 2 or capacity_value > checkpoint.max_context_tokens or query_end > model_limit)
            return error.CompactionQueryLimit;
        return capacity_value;
    }
};

test "context attention_match staged private quality request is an explicit v18 tuple" {
    inline for (.{ checkpoint.Encoding.f16, .fp8, .rotated4 }) |encoding| {
        const request: Request = .{
            .encoding = encoding,
            .admission = .cumulative_ladder_4_3_2,
            .chunk_rows = 256,
            .reject_exact_active_fixed_point = true,
            .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
            .private_dense512_policy = true,
        };
        try std.testing.expectEqual(@as(usize, 8192), try request.capacity(7936, 32768));
    }
    var invalid: Request = .{
        .encoding = .f16,
        .admission = .cumulative_ladder_4_3_2,
        .chunk_rows = 256,
        .reject_exact_active_fixed_point = true,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .private_dense512_policy = true,
    };
    invalid.chunk_rows = 128;
    try std.testing.expectError(error.InvalidNativeQualityRequest, invalid.capacity(7936, 32768));
}

test "context route parity diagnostic requires the authenticated staged private tuple" {
    var request: Request = .{ .encoding = .f16, .route_parity_diagnostic = true };
    try std.testing.expectError(error.InvalidNativeQualityRequest, request.capacity(7936, 32768));
    request.admission = .cumulative_ladder_4_3_2;
    request.chunk_rows = 256;
    request.reject_exact_active_fixed_point = true;
    request.fit_strategy = .identity_chunk_coefficients_adaptive_v18;
    request.private_dense512_policy = true;
    try std.testing.expectEqual(@as(usize, 8192), try request.capacity(7936, 32768));
}

pub fn admissionConfigurationFingerprint(base: [32]u8, admission: engine.Options.Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool) [32]u8 {
    if (admission == .fixed and chunk_rows == 1024 and !reject_exact_active_fixed_point) return base;
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update(&base);
    hash.update(if (admission == .fixed) "attention_match_admission=fixed\n" else "attention_match_admission=cumulative_ladder_4_3_2\n");
    // Preserve the already-qualified cumulative-1024 identity. Only an
    // explicit non-default geometry extends that established fingerprint.
    if (chunk_rows != 1024) {
        var chunk_text: [32]u8 = undefined;
        const chunk = std.fmt.bufPrint(&chunk_text, "attention_match_chunk_rows={d}\n", .{chunk_rows}) catch unreachable;
        hash.update(chunk);
    }
    if (reject_exact_active_fixed_point) hash.update("attention_match_reject_exact_active_fixed_point=1\n");
    return hash.finalResult();
}

/// Preserve every historical fixed and cumulative-F16 identity. New cumulative
/// codec routes receive a distinct domain only after their explicit opt-in.
pub fn admissionEncodingConfigurationFingerprint(base: [32]u8, admission: engine.Options.Admission, chunk_rows: u16, reject_exact_active_fixed_point: bool, encoding: checkpoint.Encoding) [32]u8 {
    const established = admissionConfigurationFingerprint(base, admission, chunk_rows, reject_exact_active_fixed_point);
    if (admission == .fixed or encoding == .f16) return established;
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update(&established);
    const domain: []const u8 = switch (encoding) {
        .fp8 => "attention_match_cumulative_encoding=fp8\n",
        .rotated4 => "attention_match_cumulative_encoding=rotated4\n",
        else => return established,
    };
    hash.update(domain);
    return hash.finalResult();
}

pub fn fitStrategyConfigurationFingerprint(established: [32]u8, strategy: FitStrategy) [32]u8 {
    if (strategy == .legacy) return established;
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update(&established);
    hash.update("attention_match_fit_strategy=identity_chunk_coefficients_adaptive_v18\n");
    hash.update("attention_match_adaptive_policy=greedy_positive_benefit_per_row_stable_source_level_v1\n");
    hash.update("attention_match_score_authority=cache_gpu_probability_v1_heuristic\n");
    hash.update("attention_match_identity_coefficients=1\nattention_match_identity_sweeps=0\n");
    hash.update("attention_match_chunk_rows=256\nattention_match_branch_caps=4,3,2\nattention_match_cache_query_batch=16\n");
    hash.update("attention_match_selection=residual_omp\nattention_match_value_fit=local_fp64_ls\nattention_match_rounding=checkpoint_codec\n");
    hash.update("attention_match_identity_omission=1\nattention_match_branch_selection_heldout_gates=1\n");
    hash.update("attention_match_accepted_publication_reauthorization=backend_null_fp64_plus_codec\n");
    hash.update(@embedFile("attention_match_joint_chunk_mass_diagnostic.zig"));
    hash.update(@embedFile("attention_match_joint_adaptive_geometry.zig"));
    hash.update(@embedFile("attention_match_joint_gram_geometry.zig"));
    hash.update(@embedFile("attention_match_local_weight_cap.zig"));
    hash.update(@embedFile("../../ops/metal_attention_match_train_gram.zig"));
    hash.update(@embedFile("attention_match_joint_replay_planner.zig"));
    hash.update(@embedFile("attention_match_joint_chunk_mass_replay.zig"));
    hash.update(@embedFile("attention_match_joint_adaptive_quality.zig"));
    return hash.finalResult();
}

pub const V18EvidenceProfile = enum { none, legacy_batch16_workspace512mib, private_budget95_batch64_workspace1gib, private_hybrid_bounded_batch64_workspace1gib, private_hybrid_bounded_accelerated_batch64_workspace1gib, private_physical_budget_batch64_workspace1gib, private_fp8_physical_budget_batch64_workspace1gib, private_rotated4_physical_budget_batch64_workspace1gib, resident_f16_fixed91_batch64_workspace2gib, resident_fp8_fixed91_batch64_workspace2gib, resident_rotated4_fixed91_batch64_workspace2gib, resident_e4_32512_f16_page_tsqr_v1, resident_e4_32512_fp8_page_tsqr_v1, resident_e4_32512_rotated4_page_tsqr_v1, resident_e2_formal_f16_page_tsqr_v1, resident_e2_formal_fp8_page_tsqr_v1, resident_e2_formal_rotated4_page_tsqr_v1, bounded_f16_codec_physical_v1, bounded_fp8_codec_physical_v1, bounded_rotated4_codec_physical_v1, resident_e2_f16_fixed60_batch64_workspace2gib, resident_e2_fp8_fixed60_batch64_workspace2gib, resident_e2_rotated4_fixed60_batch64_workspace2gib };

pub fn v18EvidenceProfileDigest(profile: V18EvidenceProfile) [32]u8 {
    if (profile == .none) return @splat(0);
    var hash: std.crypto.hash.sha2.Sha256 = .init(.{});
    hash.update(switch (profile) {
        .none => unreachable,
        .legacy_batch16_workspace512mib => "antfly.v18-evidence-profile.v1;legacy=1;query-batch=16;workspace=536870912",
        .private_budget95_batch64_workspace1gib => "antfly.v18-evidence-profile.v1;private=1;budget=95;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=per-query-gram;gram-delta-tile=64;joint-fixed-original-memo=1",
        .private_hybrid_bounded_batch64_workspace1gib => "antfly.v18-evidence-profile.v2;private=1;budget=95;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=flat-gram-le96-else-bounded-page;bounded-tile=8;builder-transient=268435456;joint-fixed-original-memo-flat=1",
        .private_hybrid_bounded_accelerated_batch64_workspace1gib => "antfly.v18-evidence-profile.v4;private=1;budget=95;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=flat-gram-le96-else-bounded-page;bounded-tile=64;bounded-delta=packed-fp64-accelerate;builder-transient=268435456;joint-fixed-original-memo-flat=1",
        .private_physical_budget_batch64_workspace1gib => "antfly.v18-evidence-profile.v3;private=1;budget=derived-physical-primary;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=per-query-gram;gram-delta-tile=64;joint-fixed-original-memo=1",
        .private_fp8_physical_budget_batch64_workspace1gib => "antfly.v18-evidence-profile.v5;private=1;encoding=fp8;precodec=f16;budget=derived-physical-primary;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;codec-quality=independent;postcodec-physical=independent;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=per-query-gram;gram-delta-tile=64;joint-fixed-original-memo=1",
        .private_rotated4_physical_budget_batch64_workspace1gib => "antfly.v18-evidence-profile.v5;private=1;encoding=rotated4;precodec=f16;budget=derived-physical-primary;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;codec-quality=independent;postcodec-physical=independent;query-batch=64;workspace=1073741824;probability=exact-f32-widened;geometry=per-query-gram;gram-delta-tile=64;joint-fixed-original-memo=1",
        .resident_f16_fixed91_batch64_workspace2gib => "antfly.v18-evidence-profile.v6;resident-hybrid=1;encoding=f16;budget=91;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;planned-exact=1",
        .resident_fp8_fixed91_batch64_workspace2gib => "antfly.v18-evidence-profile.v6;resident-hybrid=1;encoding=fp8;precodec=f16;budget=91;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;planned-exact=1",
        .resident_rotated4_fixed91_batch64_workspace2gib => "antfly.v18-evidence-profile.v6;resident-hybrid=1;encoding=rotated4;precodec=f16;budget=91;fallback=95;max-fallbacks=3;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;planned-exact=1",
        .resident_e4_32512_f16_page_tsqr_v1 => "antfly.v18-evidence-profile.e4-32512.v1;resident-hybrid=1;encoding=f16;source-runtime=f16;source=32512;ranges=119;candidates=357;budget=91;fallback=95;max-fallbacks=3;query-batch=64;workspace=2147483648;host=1207959552;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;heldout=accelerate-fp64;device-candidate=streamed",
        .resident_e4_32512_fp8_page_tsqr_v1 => "antfly.v18-evidence-profile.e4-32512.v1;resident-hybrid=1;encoding=fp8;source-runtime=f16;source=32512;ranges=119;candidates=357;budget=91;fallback=95;max-fallbacks=3;query-batch=64;workspace=2147483648;host=1207959552;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;heldout=accelerate-fp64;device-candidate=streamed",
        .resident_e4_32512_rotated4_page_tsqr_v1 => "antfly.v18-evidence-profile.e4-32512.v1;resident-hybrid=1;encoding=rotated4;source-runtime=f16;source=32512;ranges=119;candidates=357;budget=91;fallback=95;max-fallbacks=3;query-batch=64;workspace=2147483648;host=1207959552;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;heldout=accelerate-fp64;device-candidate=streamed",
        .resident_e2_formal_f16_page_tsqr_v1 => "antfly.v18-evidence-profile.e2-formal.v1;encoding=f16;source-runtime=f16;source-set=7936,32512;budget=60;fallback=0;max-fallbacks=0;query-batch=64;workspace=2147483648;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;adaptive=2;planned-exact=1",
        .resident_e2_formal_fp8_page_tsqr_v1 => "antfly.v18-evidence-profile.e2-formal.v1;encoding=fp8;source-runtime=f16;source-set=7936,32512;budget=60;fallback=0;max-fallbacks=0;query-batch=64;workspace=2147483648;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;adaptive=2;planned-exact=1",
        .resident_e2_formal_rotated4_page_tsqr_v1 => "antfly.v18-evidence-profile.e2-formal.v1;encoding=rotated4;source-runtime=f16;source-set=7936,32512;budget=60;fallback=0;max-fallbacks=0;query-batch=64;workspace=2147483648;builder=268435456;geometry=bounded-page;values=tsqr;train-reference=fp64-page;adaptive=2;planned-exact=1",
        .resident_e2_f16_fixed60_batch64_workspace2gib => "antfly.v18-evidence-profile.e2.v1;resident-hybrid=1;cohort=e2-8128;encoding=f16;budget=60;fallback=0;max-fallbacks=0;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;adaptive=2;planned-exact=1",
        .resident_e2_fp8_fixed60_batch64_workspace2gib => "antfly.v18-evidence-profile.e2.v1;resident-hybrid=1;cohort=e2-8128;encoding=fp8;precodec=f16;budget=60;fallback=0;max-fallbacks=0;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;adaptive=2;planned-exact=1",
        .resident_e2_rotated4_fixed60_batch64_workspace2gib => "antfly.v18-evidence-profile.e2.v1;resident-hybrid=1;cohort=e2-8128;encoding=rotated4;precodec=f16;budget=60;fallback=0;max-fallbacks=0;minimum-global-savings=1.5;query-batch=64;workspace=2147483648;probability=exact-f32-widened;geometry=flat-gram-only;adaptive=2;planned-exact=1",
        .bounded_f16_codec_physical_v1 => "antfly.bounded-codec-physical.v1;encoding=f16;precodec=f16;postcodec-physical=independent;global-savings-gate=unchanged",
        .bounded_fp8_codec_physical_v1 => "antfly.bounded-codec-physical.v1;encoding=fp8;precodec=f16;postcodec-physical=independent;global-savings-gate=unchanged",
        .bounded_rotated4_codec_physical_v1 => "antfly.bounded-codec-physical.v1;encoding=rotated4;precodec=f16;postcodec-physical=independent;global-savings-gate=unchanged",
    });
    return hash.finalResult();
}

pub fn physicalProfileEncodingMatches(profile: V18EvidenceProfile, encoding: checkpoint.Encoding) bool {
    return switch (profile) {
        .private_physical_budget_batch64_workspace1gib => encoding == .f16,
        .private_fp8_physical_budget_batch64_workspace1gib => encoding == .fp8,
        .private_rotated4_physical_budget_batch64_workspace1gib => encoding == .rotated4,
        .resident_f16_fixed91_batch64_workspace2gib => encoding == .f16,
        .resident_fp8_fixed91_batch64_workspace2gib => encoding == .fp8,
        .resident_rotated4_fixed91_batch64_workspace2gib => encoding == .rotated4,
        .resident_e4_32512_f16_page_tsqr_v1 => encoding == .f16,
        .resident_e4_32512_fp8_page_tsqr_v1 => encoding == .fp8,
        .resident_e4_32512_rotated4_page_tsqr_v1 => encoding == .rotated4,
        .resident_e2_formal_f16_page_tsqr_v1 => encoding == .f16,
        .resident_e2_formal_fp8_page_tsqr_v1 => encoding == .fp8,
        .resident_e2_formal_rotated4_page_tsqr_v1 => encoding == .rotated4,
        .resident_e2_f16_fixed60_batch64_workspace2gib => encoding == .f16,
        .resident_e2_fp8_fixed60_batch64_workspace2gib => encoding == .fp8,
        .resident_e2_rotated4_fixed60_batch64_workspace2gib => encoding == .rotated4,
        .bounded_f16_codec_physical_v1 => encoding == .f16,
        .bounded_fp8_codec_physical_v1 => encoding == .fp8,
        .bounded_rotated4_codec_physical_v1 => encoding == .rotated4,
        else => true,
    };
}

pub fn boundedTsqrValuePolicyDigest() [32]u8 {
    const backend = @import("attention_match_e2_tsqr_value_backend.zig");
    var text: [256]u8 = undefined;
    const policy_text = std.fmt.bufPrint(&text, "antfly.bounded-tsqr-value-policy.v1;batch={d};max-selected={d};max-queries={d};workspace-cap=1073741824;cpu-fp64-heldout=unchanged;publish-after-fit=1", .{ backend.sealed_batch_rows, backend.max_selected_rows, backend.max_queries }) catch unreachable;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy_text);
    return hash.finalResult();
}

test "context private physical evidence profiles bind the checkpoint encoding" {
    try std.testing.expect(physicalProfileEncodingMatches(.private_physical_budget_batch64_workspace1gib, .f16));
    try std.testing.expect(physicalProfileEncodingMatches(.private_fp8_physical_budget_batch64_workspace1gib, .fp8));
    try std.testing.expect(physicalProfileEncodingMatches(.private_rotated4_physical_budget_batch64_workspace1gib, .rotated4));
    try std.testing.expect(!physicalProfileEncodingMatches(.private_fp8_physical_budget_batch64_workspace1gib, .f16));
    try std.testing.expect(!physicalProfileEncodingMatches(.private_rotated4_physical_budget_batch64_workspace1gib, .fp8));
    try std.testing.expect(!std.mem.eql(u8, &v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib), &v18EvidenceProfileDigest(.private_fp8_physical_budget_batch64_workspace1gib)));
}

test "native quality cumulative admission has distinct explicit configuration identity" {
    const base: [32]u8 = @splat(17);
    try std.testing.expectEqualSlices(u8, &base, &admissionConfigurationFingerprint(base, .fixed, 1024, false));
    const cumulative = admissionConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, false);
    const cumulative_256 = admissionConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 256, false);
    const cumulative_fixed_point = admissionConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, true);
    try std.testing.expect(!std.mem.eql(u8, &base, &cumulative));
    try std.testing.expect(!std.mem.eql(u8, &cumulative, &cumulative_256));
    try std.testing.expect(!std.mem.eql(u8, &cumulative, &cumulative_fixed_point));
    try std.testing.expectEqualSlices(u8, &cumulative, &admissionConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, false));
    const request: Request = .{ .encoding = .f16, .minimum_savings_ratio = 1.5, .admission = .cumulative_ladder_4_3_2 };
    try std.testing.expectEqual(@as(usize, 8192), try request.capacity(7936, 131072));
    for ([_]checkpoint.Encoding{ .f16, .fp8, .rotated4 }) |encoding| {
        for ([_]u16{ 128, 256, 1024 }) |chunk_rows| {
            var codec_request = request;
            codec_request.encoding = encoding;
            codec_request.chunk_rows = chunk_rows;
            codec_request.reject_exact_active_fixed_point = true;
            try std.testing.expectEqual(@as(usize, 8192), try codec_request.capacity(7936, 131072));
        }
    }
    var invalid = request;
    invalid = request;
    invalid.chunk_rows = 64;
    try std.testing.expectError(error.InvalidNativeQualityRequest, invalid.capacity(7936, 131072));
    invalid = .{ .encoding = .f16, .chunk_rows = 256 };
    try std.testing.expectError(error.InvalidNativeQualityRequest, invalid.capacity(7936, 131072));
    var cumulative_256_request = request;
    cumulative_256_request.chunk_rows = 256;
    try std.testing.expectEqual(@as(usize, 8192), try cumulative_256_request.capacity(7936, 131072));
    var fixed_point_request = cumulative_256_request;
    fixed_point_request.reject_exact_active_fixed_point = true;
    try std.testing.expectEqual(@as(usize, 8192), try fixed_point_request.capacity(7936, 131072));
    fixed_point_request.admission = .fixed;
    fixed_point_request.chunk_rows = 1024;
    try std.testing.expectError(error.InvalidNativeQualityRequest, fixed_point_request.capacity(7936, 131072));
    const fixed_base = admissionEncodingConfigurationFingerprint(base, .fixed, 1024, false, .f16);
    try std.testing.expectEqualSlices(u8, &base, &fixed_base);
    try std.testing.expectEqualSlices(u8, &fixed_base, &admissionEncodingConfigurationFingerprint(base, .fixed, 1024, false, .fp8));
    try std.testing.expectEqualSlices(u8, &cumulative, &admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, false, .f16));
    const cumulative_fp8 = admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, false, .fp8);
    const cumulative_rotated4 = admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 1024, false, .rotated4);
    try std.testing.expect(!std.mem.eql(u8, &cumulative, &cumulative_fp8));
    try std.testing.expect(!std.mem.eql(u8, &cumulative, &cumulative_rotated4));
    try std.testing.expect(!std.mem.eql(u8, &cumulative_fp8, &cumulative_rotated4));
    for ([_]u16{ 256, 1024 }) |chunk_rows| {
        for ([_]bool{ false, true }) |fixed_point| {
            const historical_f16 = admissionConfigurationFingerprint(base, .cumulative_ladder_4_3_2, chunk_rows, fixed_point);
            try std.testing.expectEqualSlices(u8, &historical_f16, &admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, chunk_rows, fixed_point, .f16));
            const encoded_fp8 = admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, chunk_rows, fixed_point, .fp8);
            const encoded_rotated4 = admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, chunk_rows, fixed_point, .rotated4);
            try std.testing.expect(!std.mem.eql(u8, &historical_f16, &encoded_fp8));
            try std.testing.expect(!std.mem.eql(u8, &historical_f16, &encoded_rotated4));
            try std.testing.expect(!std.mem.eql(u8, &encoded_fp8, &encoded_rotated4));
        }
    }
}

test "private v18 fit strategy is typed and preserves every legacy fingerprint" {
    const base: [32]u8 = @splat(29);
    inline for (.{ checkpoint.Encoding.f16, .fp8, .rotated4 }) |encoding| {
        const established = admissionEncodingConfigurationFingerprint(base, .cumulative_ladder_4_3_2, 256, true, encoding);
        try std.testing.expectEqualSlices(u8, &established, &fitStrategyConfigurationFingerprint(established, .legacy));
        const v18 = fitStrategyConfigurationFingerprint(established, .identity_chunk_coefficients_adaptive_v18);
        try std.testing.expect(!std.mem.eql(u8, &established, &v18));
    }
    var request: Request = .{ .encoding = .f16, .admission = .cumulative_ladder_4_3_2, .chunk_rows = 256, .reject_exact_active_fixed_point = true, .fit_strategy = .identity_chunk_coefficients_adaptive_v18 };
    _ = try request.capacity(7936, 131072);
    request.chunk_rows = 1024;
    try std.testing.expectError(error.InvalidNativeQualityRequest, request.capacity(7936, 131072));
}

/// Authenticated provenance for a checkpoint fitted by a separately sealed
/// diagnostic campaign. The caller owns `wire`; this view never retains it.
pub const SuppliedCandidate = struct {
    wire: []const u8,
    wire_sha256: [32]u8,
    closure_sha256: [32]u8,
    source_checkpoint_sha256: [32]u8,
    prompt_tokens_sha256: [32]u8,
    identity: checkpoint.Identity,
    logical_tokens: usize,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    completed_chunks: usize,
    validated_heads: usize,
    preserved_heads: usize,
    max_head_mass_error: f64,
    max_head_output_rmse: f64,
    global_savings_gate: f64,
    closure_global_savings_gate_pass: bool,

    pub fn validate(self: SuppliedCandidate, expected: checkpoint.Identity, prompt: []const i64) !void {
        if (self.wire.len == 0 or self.wire.len > checkpoint.serving_limits.max_file_bytes or
            std.mem.allEqual(u8, &self.wire_sha256, 0) or std.mem.allEqual(u8, &self.closure_sha256, 0) or
            std.mem.allEqual(u8, &self.source_checkpoint_sha256, 0) or
            !self.identity.eql(expected) or self.logical_tokens != prompt.len or
            !std.mem.eql(u8, &self.prompt_tokens_sha256, &try teacher.promptTokenDigest(prompt)) or
            self.source_global_bytes == 0 or self.candidate_global_bytes == 0 or
            self.candidate_global_bytes >= self.source_global_bytes or self.completed_chunks == 0 or
            self.validated_heads == 0 or self.preserved_heads > self.validated_heads or
            !std.math.isFinite(self.max_head_mass_error) or self.max_head_mass_error < 0 or self.max_head_mass_error > 0.05 or
            !std.math.isFinite(self.max_head_output_rmse) or self.max_head_output_rmse < 0 or self.max_head_output_rmse > 0.05 or
            !std.math.isFinite(self.global_savings_gate) or self.global_savings_gate < 1.5 or
            !self.closure_global_savings_gate_pass or
            @as(f64, @floatFromInt(self.source_global_bytes)) / @as(f64, @floatFromInt(self.candidate_global_bytes)) < self.global_savings_gate)
            return error.InvalidSuppliedCheckpointQualityCandidate;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(self.wire, &digest, .{});
        if (!std.mem.eql(u8, &digest, &self.wire_sha256)) return error.SuppliedCheckpointChanged;
    }

    pub fn decode(self: SuppliedCandidate, allocator: std.mem.Allocator, expected: checkpoint.Identity, prompt: []const i64, control: @import("attention_match.zig").Control) !checkpoint.Owned {
        try self.validate(expected, prompt);
        try control.check();
        return checkpoint.decodeWithControl(allocator, self.wire, expected, checkpoint.serving_limits, control);
    }
};

/// Exact public AFATT002 candidate. Unlike SuppliedCandidate, its checkpoint
/// is already decoded by the maintained frame verifier and its metrics are
/// copied only from the producer-minted closure.
pub const PreparedServingCandidate = struct {
    source_checkpoint: checkpoint.Owned,
    checkpoint: checkpoint.Owned,
    checkpoint_sha256: [32]u8,
    closure_sha256: [32]u8,
    source_checkpoint_sha256: [32]u8,
    encoding: checkpoint.Encoding,
    prompt_tokens_sha256: [32]u8,
    logical_tokens: usize,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    validated_heads: usize,
    preserved_heads: usize,
    completed_chunks: usize,
    max_head_mass_error: f64,
    max_head_output_rmse: f64,
    exact_local_window_capacity: bool,
    /// Authenticated AFATT002 candidate, exact-source, and closure bytes held
    /// by the caller until the complete prepared quality run returns.
    retained_input_bytes: usize,
    producer_v18: ProducerV18Evidence,
    /// Authenticated digest of the exact future-mixture authority carried by
    /// the prepared closure; zero for every legacy/default profile.
    future_training_authority_sha256: [32]u8 = @splat(0),
    live: bool = true,

    pub fn deinit(self: *PreparedServingCandidate) void {
        if (self.live) {
            self.checkpoint.deinit();
            self.source_checkpoint.deinit();
        }
        self.live = false;
    }

    pub const Pair = struct { source: checkpoint.Owned, candidate: checkpoint.Owned };

    pub fn take(self: *PreparedServingCandidate, expected: checkpoint.Identity, prompt: []const i64) !Pair {
        const candidate_scaled = std.math.mul(usize, self.candidate_global_bytes, 3) catch return error.InvalidPreparedServingQualityCandidate;
        const source_scaled = std.math.mul(usize, self.source_global_bytes, 2) catch return error.InvalidPreparedServingQualityCandidate;
        if (!self.live or !self.checkpoint.state.identity.eql(expected) or !self.source_checkpoint.state.identity.eql(expected) or
            self.source_checkpoint.state.logical_length != prompt.len or self.source_checkpoint.state.tokens.len != prompt.len or
            self.logical_tokens != prompt.len or
            !std.mem.eql(u8, &self.prompt_tokens_sha256, &try teacher.operatorPromptTokenDigest(&.{}, prompt)) or
            (self.encoding != .f16 and self.encoding != .fp8 and self.encoding != .rotated4) or
            !preparedEvidenceProfileMatchesEncoding(self.producer_v18, self.encoding) or
            std.mem.allEqual(u8, &self.checkpoint_sha256, 0) or std.mem.allEqual(u8, &self.closure_sha256, 0) or std.mem.allEqual(u8, &self.source_checkpoint_sha256, 0) or
            self.source_global_bytes == 0 or self.candidate_global_bytes == 0 or candidate_scaled > source_scaled or
            self.validated_heads == 0 or self.preserved_heads > self.validated_heads or (self.completed_chunks == 0 and !producerBoundedComplete(self.producer_v18)) or
            !std.math.isFinite(self.max_head_mass_error) or self.max_head_mass_error < 0 or self.max_head_mass_error > 0.05 or
            !std.math.isFinite(self.max_head_output_rmse) or self.max_head_output_rmse < 0 or self.max_head_output_rmse > 0.05 or
            !self.exact_local_window_capacity or self.retained_input_bytes == 0)
            return error.InvalidPreparedServingQualityCandidate;
        for (self.source_checkpoint.state.tokens, prompt) |actual, wanted|
            if (wanted < 0 or actual != @as(u32, @intCast(wanted))) return error.InvalidPreparedServingQualityCandidate;
        self.live = false;
        return .{ .source = self.source_checkpoint, .candidate = self.checkpoint };
    }
};

fn producerBoundedComplete(evidence: ProducerV18Evidence) bool {
    const bounded_profile = switch (evidence.evidence_profile) {
        .resident_e4_32512_f16_page_tsqr_v1,
        .resident_e4_32512_fp8_page_tsqr_v1,
        .resident_e4_32512_rotated4_page_tsqr_v1,
        .resident_e2_formal_f16_page_tsqr_v1,
        .resident_e2_formal_fp8_page_tsqr_v1,
        .resident_e2_formal_rotated4_page_tsqr_v1,
        => true,
        else => false,
    };
    const expected_backend_calls = std.math.mul(usize, evidence.bounded_delta_tiles, 2) catch return false;
    return bounded_profile and evidence.bounded_heads != 0 and evidence.bounded_heads == evidence.accepted_heads and
        evidence.bounded_delta_heads == evidence.bounded_heads and evidence.bounded_delta_tiles != 0 and
        evidence.bounded_delta_backend_calls == expected_backend_calls and evidence.bounded_delta_packed_bytes_peak != 0 and
        evidence.bounded_delta_staged_bytes_peak != 0 and evidence.bounded_delta_fallbacks == 0 and
        evidence.bounded_tsqr_heads == evidence.bounded_heads and evidence.bounded_tsqr_calls >= evidence.bounded_tsqr_heads and
        evidence.bounded_tsqr_workspace_peak_bytes != 0 and evidence.bounded_tsqr_workspace_peak_bytes <= evidence.planned_head_workspace_bytes and
        evidence.bounded_tsqr_cancellation_checks >= evidence.bounded_tsqr_calls;
}

test "formal bounded producer closure requires complete delta TSQR and codec-bound profile" {
    const rows = [_]struct { checkpoint.Encoding, V18EvidenceProfile, V18EvidenceProfile }{
        .{ .f16, .resident_e2_formal_f16_page_tsqr_v1, .bounded_f16_codec_physical_v1 },
        .{ .fp8, .resident_e2_formal_fp8_page_tsqr_v1, .bounded_fp8_codec_physical_v1 },
        .{ .rotated4, .resident_e2_formal_rotated4_page_tsqr_v1, .bounded_rotated4_codec_physical_v1 },
        .{ .f16, .resident_e4_32512_f16_page_tsqr_v1, .bounded_f16_codec_physical_v1 },
        .{ .fp8, .resident_e4_32512_fp8_page_tsqr_v1, .bounded_fp8_codec_physical_v1 },
        .{ .rotated4, .resident_e4_32512_rotated4_page_tsqr_v1, .bounded_rotated4_codec_physical_v1 },
    };
    for (rows) |row| {
        var evidence: ProducerV18Evidence = .{
            .policy_sha256 = @splat(1),
            .eligible_heads = 3,
            .attempted_heads = 3,
            .accepted_heads = 2,
            .preserved_heads = 0,
            .fallback_heads = 0,
            .cache_ready_heads = 2,
            .planned_exact_heads = 1,
            .query_batch_rows = 64,
            .planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024,
            .cache_phase_peak_bound_bytes = 1024,
            .bounded_heads = 2,
            .bounded_delta_heads = 2,
            .bounded_delta_tiles = 3,
            .bounded_delta_backend_calls = 6,
            .bounded_delta_packed_bytes_peak = 1,
            .bounded_delta_staged_bytes_peak = 1,
            .bounded_delta_fallbacks = 0,
            .primary_budget_percent = 60,
            .budget_fallback_heads = 0,
            .evidence_profile = row[1],
            .evidence_profile_sha256 = v18EvidenceProfileDigest(row[1]),
            .codec_physical_profile = row[2],
            .codec_physical_profile_sha256 = v18EvidenceProfileDigest(row[2]),
            .bounded_tsqr_policy_sha256 = boundedTsqrValuePolicyDigest(),
            .bounded_tsqr_heads = 2,
            .bounded_tsqr_calls = 2,
            .bounded_tsqr_workspace_peak_bytes = 1,
            .bounded_tsqr_cancellation_checks = 2,
        };
        try std.testing.expect(physicalProfileEncodingMatches(row[1], row[0]));
        try std.testing.expect(physicalProfileEncodingMatches(row[2], row[0]));
        try std.testing.expect(producerBoundedComplete(evidence));
        evidence.bounded_tsqr_calls = 1;
        try std.testing.expect(!producerBoundedComplete(evidence));
        evidence.bounded_tsqr_calls = 2;
        evidence.bounded_delta_backend_calls = 5;
        try std.testing.expect(!producerBoundedComplete(evidence));
        try std.testing.expect(!physicalProfileEncodingMatches(row[2], switch (row[0]) {
            .f16 => .fp8,
            else => .f16,
        }));
    }
    try std.testing.expect(!std.mem.eql(u8, &v18EvidenceProfileDigest(.resident_e2_formal_f16_page_tsqr_v1), &v18EvidenceProfileDigest(.resident_e2_f16_fixed60_batch64_workspace2gib)));
}

pub fn preparedEvidenceProfileMatchesEncoding(evidence: ProducerV18Evidence, encoding: checkpoint.Encoding) bool {
    return physicalProfileEncodingMatches(evidence.evidence_profile, encoding) and
        std.mem.eql(u8, &evidence.evidence_profile_sha256, &v18EvidenceProfileDigest(evidence.evidence_profile));
}

test "context prepared serving evidence profile binds all candidate codecs" {
    inline for (.{
        .{ checkpoint.Encoding.f16, V18EvidenceProfile.private_physical_budget_batch64_workspace1gib },
        .{ checkpoint.Encoding.fp8, V18EvidenceProfile.private_fp8_physical_budget_batch64_workspace1gib },
        .{ checkpoint.Encoding.rotated4, V18EvidenceProfile.private_rotated4_physical_budget_batch64_workspace1gib },
    }) |row| {
        var evidence: ProducerV18Evidence = std.mem.zeroes(ProducerV18Evidence);
        evidence.evidence_profile = row[1];
        evidence.evidence_profile_sha256 = v18EvidenceProfileDigest(row[1]);
        try std.testing.expect(preparedEvidenceProfileMatchesEncoding(evidence, row[0]));
        evidence.evidence_profile_sha256[0] ^= 1;
        try std.testing.expect(!preparedEvidenceProfileMatchesEncoding(evidence, row[0]));
    }
    var fp8: ProducerV18Evidence = std.mem.zeroes(ProducerV18Evidence);
    fp8.evidence_profile = .private_fp8_physical_budget_batch64_workspace1gib;
    fp8.evidence_profile_sha256 = v18EvidenceProfileDigest(fp8.evidence_profile);
    try std.testing.expect(!preparedEvidenceProfileMatchesEncoding(fp8, .f16));
    try std.testing.expect(!preparedEvidenceProfileMatchesEncoding(fp8, .rotated4));
}

pub const ProducerV18Evidence = struct {
    policy_sha256: [32]u8,
    eligible_heads: usize,
    attempted_heads: usize,
    accepted_heads: usize,
    preserved_heads: usize,
    fallback_heads: usize,
    cache_ready_heads: usize,
    planned_exact_heads: usize = 0,
    exact_f32_probability_heads: usize = 0,
    incremental_gram_chooser_calls: usize = 0,
    query_batch_rows: usize,
    planned_head_workspace_bytes: usize,
    cache_phase_peak_bound_bytes: usize,
    bounded_heads: usize,
    bounded_delta_heads: usize,
    bounded_delta_tiles: usize,
    bounded_delta_backend_calls: usize,
    bounded_delta_packed_bytes_peak: usize,
    bounded_delta_staged_bytes_peak: usize,
    bounded_delta_fallbacks: usize,
    primary_budget_percent: u8,
    budget_fallback_heads: usize,
    evidence_profile: V18EvidenceProfile,
    evidence_profile_sha256: [32]u8,
    codec_physical_profile: V18EvidenceProfile = .none,
    codec_physical_profile_sha256: [32]u8 = @splat(0),
    bounded_tsqr_policy_sha256: [32]u8 = @splat(0),
    bounded_tsqr_heads: usize = 0,
    bounded_tsqr_calls: usize = 0,
    bounded_tsqr_workspace_peak_bytes: usize = 0,
    bounded_tsqr_cancellation_checks: usize = 0,

    pub fn apply(self: ProducerV18Evidence, report: *Report) void {
        report.v18_policy_sha256 = self.policy_sha256;
        report.v18_eligible_heads = self.eligible_heads;
        report.v18_attempted_heads = self.attempted_heads;
        report.v18_accepted_heads = self.accepted_heads;
        report.v18_preserved_heads = self.preserved_heads;
        report.v18_fallback_heads = self.fallback_heads;
        report.v18_cache_ready_heads = self.cache_ready_heads;
        report.v18_planned_exact_heads = self.planned_exact_heads;
        report.v18_exact_f32_probability_heads = self.exact_f32_probability_heads;
        report.incremental_gram_chooser_calls = self.incremental_gram_chooser_calls;
        report.v18_query_batch_rows = self.query_batch_rows;
        report.v18_planned_head_workspace_bytes = self.planned_head_workspace_bytes;
        report.v18_cache_phase_peak_bound_bytes = self.cache_phase_peak_bound_bytes;
        report.v18_bounded_heads = self.bounded_heads;
        report.v18_bounded_delta_heads = self.bounded_delta_heads;
        report.v18_bounded_delta_tiles = self.bounded_delta_tiles;
        report.v18_bounded_delta_backend_calls = self.bounded_delta_backend_calls;
        report.v18_bounded_delta_packed_bytes_peak = self.bounded_delta_packed_bytes_peak;
        report.v18_bounded_delta_staged_bytes_peak = self.bounded_delta_staged_bytes_peak;
        report.v18_bounded_delta_fallbacks = self.bounded_delta_fallbacks;
        report.v18_primary_budget_percent = self.primary_budget_percent;
        report.v18_budget_fallback_heads = self.budget_fallback_heads;
        report.v18_evidence_profile = self.evidence_profile;
        report.v18_evidence_profile_sha256 = self.evidence_profile_sha256;
        report.v18_codec_physical_profile = self.codec_physical_profile;
        report.v18_codec_physical_profile_sha256 = self.codec_physical_profile_sha256;
        report.v18_bounded_tsqr_policy_sha256 = self.bounded_tsqr_policy_sha256;
        report.v18_bounded_tsqr_heads = self.bounded_tsqr_heads;
        report.v18_bounded_tsqr_calls = self.bounded_tsqr_calls;
        report.v18_bounded_tsqr_workspace_peak_bytes = self.bounded_tsqr_workspace_peak_bytes;
        report.v18_bounded_tsqr_cancellation_checks = self.bounded_tsqr_cancellation_checks;
    }
};

test "supplied quality candidate authenticates identity prompt shape and global gate without fitting" {
    const prompt = [_]i64{ 7, 11, 50 };
    const identity: checkpoint.Identity = .{
        .weights = @splat(1),
        .tokenizer = @splat(2),
        .template = @splat(3),
        .layout = @splat(4),
    };
    const wire = "authenticated-f16-v3-fixture";
    var wire_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(wire, &wire_sha256, .{});
    const valid: SuppliedCandidate = .{
        .wire = wire,
        .wire_sha256 = wire_sha256,
        .closure_sha256 = @splat(5),
        .source_checkpoint_sha256 = @splat(6),
        .prompt_tokens_sha256 = try teacher.promptTokenDigest(&prompt),
        .identity = identity,
        .logical_tokens = prompt.len,
        .source_global_bytes = 300,
        .candidate_global_bytes = 100,
        .completed_chunks = 1,
        .validated_heads = 2,
        .preserved_heads = 1,
        .max_head_mass_error = 0.01,
        .max_head_output_rmse = 0.02,
        .global_savings_gate = 1.5,
        .closure_global_savings_gate_pass = true,
    };
    try valid.validate(identity, &prompt);

    var changed = valid;
    changed.identity.layout = @splat(9);
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityCandidate, changed.validate(identity, &prompt));
    changed = valid;
    changed.logical_tokens += 1;
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityCandidate, changed.validate(identity, &prompt));
    changed = valid;
    changed.candidate_global_bytes = 250;
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityCandidate, changed.validate(identity, &prompt));
    changed = valid;
    changed.max_head_mass_error = 0.051;
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityCandidate, changed.validate(identity, &prompt));
    changed = valid;
    changed.max_head_output_rmse = 0.051;
    try std.testing.expectError(error.InvalidSuppliedCheckpointQualityCandidate, changed.validate(identity, &prompt));
    changed = valid;
    changed.wire_sha256 = @splat(7);
    try std.testing.expectError(error.SuppliedCheckpointChanged, changed.validate(identity, &prompt));

    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, valid.decode(std.testing.allocator, identity, &prompt, .{ .check_fn = Cancel.check }));

    const one_prompt = [_]i64{0};
    const tokens = [_]u32{0};
    const logits = [_]f32{ 0, 1 };
    const layer_donors = [_]u32{0};
    const positions = [_]u64{0};
    const coverage_ends = [_]u64{1};
    const keys = [_]f32{0};
    const values = [_]f32{0};
    const bias = [_]f32{0};
    const heads = [_]checkpoint.Head{.{ .dim = 1, .compacted_until = 1, .positions = &positions, .coverage_ends = &coverage_ends, .keys = &keys, .values = &values, .bias = &bias }};
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .heads = &heads }};
    const state: checkpoint.State = .{ .identity = identity, .logical_length = 1, .tokens = &tokens, .logits = &logits, .layer_donors = &layer_donors, .donors = &donors };
    const encoded = try checkpoint.encodeWithEncoding(std.testing.allocator, state, checkpoint.serving_limits, .f16, .{});
    defer std.testing.allocator.free(encoded);
    var encoded_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(encoded, &encoded_digest, .{});
    var allocation_candidate = valid;
    allocation_candidate.wire = encoded;
    allocation_candidate.wire_sha256 = encoded_digest;
    allocation_candidate.prompt_tokens_sha256 = try teacher.promptTokenDigest(&one_prompt);
    allocation_candidate.logical_tokens = one_prompt.len;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, allocation_candidate.decode(failing.allocator(), identity, &one_prompt, .{}));
}

pub const Frozen = struct {
    identity: checkpoint.Identity,
    observed: native_release.Observed,
    execution: native_release.Execution,
    configuration_sha256: [32]u8,
    implementation_sha256: [32]u8,
    prompt_sha256: [32]u8,
    checkpoint_sha256: [32]u8,
    installed_sha256: [32]u8,
    source_checkpoint_sha256: [32]u8,
    calibration: ?queries.Seal,
    candidate_source: enum { fitted, supplied, prepared_serving } = .fitted,
    fit_calls: usize = 1,
    calibration_executed: bool = true,
    admission: engine.Options.Admission = .fixed,
    chunk_rows: u16 = 1024,
    reject_exact_active_fixed_point: bool = false,
    encoding: checkpoint.Encoding = .f16,
    fit_strategy: FitStrategy = .legacy,
    source_revision: u64,
    destination_revision_before_install: u64,
    installed_revision: u64,
    logical_tokens: usize,
    completed_chunks: usize,
    validated_heads: usize,
    preserved_heads: usize = 0,
    protected_prefix_rows: usize,
    protected_recent_rows: usize,
    fp8_heads: usize = 0,
    fp8_fallback_heads: usize = 0,
    rotated4_heads: usize = 0,
    rotated4_fallback_heads: usize = 0,

    pub fn validate(self: Frozen) !void {
        self.observed.destination.validate() catch return error.InvalidNativeQualityPublication;
        self.execution.validate() catch return error.InvalidNativeQualityPublication;
        if (self.encoding == .f32 or
            (self.admission == .fixed and (self.chunk_rows != 1024 or self.reject_exact_active_fixed_point)) or
            (self.admission == .cumulative_ladder_4_3_2 and self.chunk_rows != 128 and self.chunk_rows != 256 and self.chunk_rows != 1024) or
            std.mem.allEqual(u8, &self.observed.backend_environment_sha256, 0) or
            !std.mem.eql(u8, &self.configuration_sha256, &fitStrategyConfigurationFingerprint(admissionEncodingConfigurationFingerprint(native_release.configurationFingerprint(self.execution, self.observed.backend_environment_sha256), self.admission, self.chunk_rows, self.reject_exact_active_fixed_point, self.encoding), self.fit_strategy)))
            return error.InvalidNativeQualityPublication;
        inline for (.{ "weights", "tokenizer", "template", "layout" }) |field|
            if (std.mem.allEqual(u8, &@field(self.identity, field), 0)) return error.InvalidNativeQualityPublication;
        inline for (.{ "implementation_sha256", "prompt_sha256", "checkpoint_sha256", "installed_sha256", "source_checkpoint_sha256" }) |field|
            if (std.mem.allEqual(u8, &@field(self, field), 0)) return error.InvalidNativeQualityPublication;
        if (self.calibration_executed) {
            if (self.candidate_source != .fitted or self.fit_calls != 1) return error.InvalidNativeQualityPublication;
            const calibration = self.calibration orelse return error.InvalidNativeQualityPublication;
            calibration.validate() catch return error.InvalidNativeQualityPublication;
            if (calibration.source_length != self.logical_tokens or calibration.repeat_length != 512 or
                calibration.train_positions != 256 or calibration.heldout_positions != 64) return error.InvalidNativeQualityPublication;
        } else {
            if (self.fit_calls != 0 or self.calibration != null) return error.InvalidNativeQualityPublication;
            switch (self.candidate_source) {
                .supplied => if (self.admission != .fixed or self.chunk_rows != 1024 or self.reject_exact_active_fixed_point or self.fit_strategy != .legacy) return error.InvalidNativeQualityPublication,
                .prepared_serving => if (self.admission != .cumulative_ladder_4_3_2 or self.chunk_rows != 256 or !self.reject_exact_active_fixed_point or self.fit_strategy != .identity_chunk_coefficients_adaptive_v18) return error.InvalidNativeQualityPublication,
                .fitted => return error.InvalidNativeQualityPublication,
            }
        }
        if (!std.mem.eql(u8, &self.checkpoint_sha256, &self.installed_sha256) or
            self.source_revision == std.math.maxInt(u64) or self.destination_revision_before_install == std.math.maxInt(u64) or
            self.installed_revision != self.destination_revision_before_install + 1 or
            self.logical_tokens < 2 or self.logical_tokens > checkpoint.max_context_tokens or
            self.protected_prefix_rows < 128 or self.protected_recent_rows < 2048 or
            self.protected_prefix_rows >= self.logical_tokens or
            self.protected_recent_rows >= self.logical_tokens - self.protected_prefix_rows or
            self.completed_chunks == 0 or self.validated_heads == 0 or
            (self.admission == .cumulative_ladder_4_3_2 and self.encoding == .fp8 and self.fp8_heads == 0) or
            (self.admission == .cumulative_ladder_4_3_2 and self.encoding == .rotated4 and self.rotated4_heads == 0) or
            self.fp8_heads > self.validated_heads or self.rotated4_heads > self.validated_heads - self.fp8_heads or
            self.fp8_fallback_heads > self.validated_heads or self.rotated4_fallback_heads > self.validated_heads)
            return error.InvalidNativeQualityPublication;
    }
};

pub const Reference = struct {
    count: usize,
    sha256: [32]u8,
    provider: teacher.ReferenceProvider,

    pub fn reveal(self: Reference, frozen: Frozen, capacity: usize) ![]const i64 {
        try frozen.validate();
        if (self.count < teacher.min_reference_tokens or self.count > capacity or
            std.mem.allEqual(u8, &self.sha256, 0)) return error.InvalidNativeQualityReference;
        const tokens = try self.provider.reveal();
        if (tokens.len != self.count or
            !std.mem.eql(u8, &self.sha256, &try teacher.referenceSequenceDigest(tokens)))
            return error.InvalidNativeQualityReference;
        return tokens;
    }
};

pub const Greedy = struct {
    tokens: [max_output_tokens]i64 = @splat(0),
    count: usize = 0,
    hit_eos: bool = false,
    forward_steps: usize = 0,
    prepared_forward_steps: usize = 0,
    serving_forward_steps: usize = 0,
    ordinary_serving_forward_steps: usize = 0,
    ordinary_backend_frames: usize = 0,
    cold_staged_forward_steps: usize = 0,
    prepared_layer_blocks: usize = 0,
    lm_head_q4_q6_refine_dispatches: u64 = 0,
    lm_head_q6_full_argmax_dispatches: u64 = 0,
    lm_head_q4_resident_sampling_rejections: u64 = 0,
    e4_fused_half_attempts: u64 = 0,
    e4_fused_half_hits: u64 = 0,
    e4_fused_half_pair_dispatches: u64 = 0,
    e4_fused_half_down_dispatches: u64 = 0,
    e4_fused_half_fallbacks: u64 = 0,
    e4_fused_half_fatal_failures: u64 = 0,
    owner_id: u64 = 0,
    public_text: []u8 = &.{},
    release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
};

pub const Scores = struct {
    log_probabilities: [max_output_tokens]f64 = @splat(0),
    count: usize = 0,
    nll: f64 = 0,
    forward_steps: usize = 0,
    prepared_forward_steps: usize = 0,
    serving_forward_steps: usize = 0,
    ordinary_serving_forward_steps: usize = 0,
    ordinary_backend_frames: usize = 0,
    cold_staged_forward_steps: usize = 0,
    prepared_layer_blocks: usize = 0,
    attention_calls: usize = 0,
    donor_appends: usize = 0,
    owner_id: u64 = 0,
    release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
};

/// Public checkpoint serving can use its staged installed-attention path on
/// the first cold step while model slots become ready. Later steps must use
/// every prepared layer tail; partial execution never proves the route.
pub fn validatePreparedStep(blocks: u64, layers: usize, first: bool) !void {
    if (layers == 0 or (blocks != layers and !(first and blocks == 0)))
        return error.InvalidNativeQualityPreparedRoute;
}

/// Parent-clock durations of the real serial collector phases. These are
/// diagnostics, separate from serving throughput or destination-soak evidence.
pub const PhaseDurations = struct {
    geometry_prepare: u64 = 0,
    exact_prefill: u64 = 0,
    query_collection: u64 = 0,
    fit: u64 = 0,
    transport: u64 = 0,
    install: u64 = 0,
    model_slots: u64 = 0,
    exact_greedy: u64 = 0,
    compacted_greedy: u64 = 0,
    reference_reveal: u64 = 0,
    exact_teacher: u64 = 0,
    compacted_teacher: u64 = 0,
    teardown: u64 = 0,
};

pub const QueryCollectionProvenance = enum { none, primary_reference, dense512_reference };

pub const QueryCollectionDiagnostic = struct {
    provenance: QueryCollectionProvenance = .none,
    setup_ns: u64 = 0,
    prefix_clone_ns: u64 = 0,
    repeat_forward_ns: u64 = 0,
    cleanup_ns: u64 = 0,
    forward_calls: usize = 0,
    samples: usize = 0,
    repeat_rows: usize = 0,
    chunk_rows: usize = 0,
    consumers: usize = 0,
    query_bytes: usize = 0,
    cloned_bytes: usize = 0,
    shadow_allocation_bound_bytes: usize = 0,
    shadow_owner_peak_bytes: u64 = 0,
    ragged_attention_calls: usize = 0,
    scope_released: bool = false,
    scope_delta_bytes: ?u64 = null,

    pub fn validate(self: @This()) !void {
        if (self.provenance == .none) {
            if (!std.meta.eql(self, @This(){})) return error.InvalidNativeQualityPublication;
        } else if (self.samples < 2 or self.repeat_rows < 2 or self.chunk_rows == 0 or self.consumers == 0 or
            self.query_bytes == 0 or self.forward_calls == 0 or !self.scope_released or self.scope_delta_bytes != 0)
            return error.InvalidNativeQualityPublication;
    }
};

test "query collection diagnostics remain absent unless explicitly populated" {
    try (QueryCollectionDiagnostic{}).validate();
    var actual: QueryCollectionDiagnostic = .{ .provenance = .primary_reference };
    try std.testing.expectError(error.InvalidNativeQualityPublication, actual.validate());
    actual.samples = 320;
    actual.repeat_rows = 512;
    actual.chunk_rows = 256;
    actual.consumers = 42;
    actual.query_bytes = 64 * 1024 * 1024;
    actual.forward_calls = 2;
    actual.scope_released = true;
    actual.scope_delta_bytes = 0;
    try actual.validate();
}

const RemovedFutureTrainingCaptureEvidenceLegacy = struct {
    generated_ids: [256]i64,
    generated_ids_count: u16,
    generated_ids_sha256: [32]u8,
    source_before: @import("attention_match_self_study_training.zig").SourceBoundary,
    source_after: @import("attention_match_self_study_training.zig").SourceBoundary,
    early_eos_observations: u16,
    backend_owned_full_logits_calls: u16,
    backend_owned_full_logits_ns: u64,
    backend_owned_frame_successes: u16,
    generic_last_logits_calls: u16,
    backend_owned_full_logits_selected: bool,
    primary_rows: u16 = 256,
    historical_rows: u16 = 256,
    future_rows: u16 = 256,
    heldout_rows: u16 = 64,
    heldout_sha256_before_fit: [32]u8 = @splat(0),

    pub fn validate(self: @This(), authority: @import("attention_match_future_decode_training.zig").Authority) !void {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        for (self.generated_ids) |id| hash.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, id)));
        const digest = hash.finalResult();
        var source_hash = std.crypto.hash.sha2.Sha256.init(.{});
        inline for (.{ self.source_before.identity.weights, self.source_before.identity.tokenizer, self.source_before.identity.template, self.source_before.identity.layout }) |value| source_hash.update(&value);
        source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, self.source_before.sequence_id)));
        source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, @intCast(self.source_before.logical_tokens))));
        const source_identity_sha256 = source_hash.finalResult();
        const domain_sha256 = [_][32]u8{ authority.primary_sha256, authority.historical_sha256, authority.future_sha256, authority.heldout_sha256 };
        for (domain_sha256, 0..) |value, index| {
            if (std.mem.allEqual(u8, &value, 0)) return error.InvalidNativeQualityPublication;
            for (domain_sha256[0..index]) |previous|
                if (std.mem.eql(u8, &value, &previous)) return error.InvalidNativeQualityPublication;
        }
        if (self.generated_ids_count != self.generated_ids.len or
            !std.mem.eql(u8, &digest, &self.generated_ids_sha256) or
            !std.mem.eql(u8, &digest, &authority.generated_ids_sha256) or
            !std.mem.eql(u8, &source_identity_sha256, &authority.source_identity_sha256) or
            !std.meta.eql(self.source_before, self.source_after) or
            self.source_before.kv_revision != authority.source_revision or
            !std.mem.eql(u8, &self.source_before.tokens_sha256, &authority.source_tokens_sha256) or
            self.early_eos_observations != authority.early_eos_observations or
            !self.backend_owned_full_logits_selected or self.backend_owned_full_logits_calls != 256 or
            self.backend_owned_frame_successes != 256 or self.generic_last_logits_calls != 0 or
            self.backend_owned_full_logits_ns == 0 or self.primary_rows != 256 or
            self.historical_rows != 256 or self.future_rows != 256 or self.heldout_rows != 64 or
            !std.mem.eql(u8, &self.heldout_sha256_before_fit, &authority.heldout_sha256))
            return error.InvalidNativeQualityPublication;
    }
};

pub const FutureTrainingCaptureEvidence = future_augmentation.FutureTrainingCaptureEvidence;

test "context future training capture evidence binds generated frame source and domains" {
    var ids: [256]i64 = undefined;
    for (&ids, 0..) |*id, index| id.* = @intCast(index);
    var ids_hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (ids) |id| ids_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(i64, id)));
    const ids_sha = ids_hash.finalResult();
    const boundary: @import("attention_match_self_study_training.zig").SourceBoundary = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .sequence_id = 7,
        .logical_tokens = 7936,
        .kv_revision = 11,
        .tokens_sha256 = @splat(5),
    };
    var source_hash = std.crypto.hash.sha2.Sha256.init(.{});
    inline for (.{ boundary.identity.weights, boundary.identity.tokenizer, boundary.identity.template, boundary.identity.layout }) |value| source_hash.update(&value);
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.sequence_id)));
    source_hash.update(&std.mem.toBytes(std.mem.nativeToLittle(u64, boundary.logical_tokens)));
    const authority: @import("attention_match_future_decode_training.zig").Authority = .{ .source_identity_sha256 = source_hash.finalResult(), .source_tokens_sha256 = boundary.tokens_sha256, .source_revision = boundary.kv_revision, .generation_contract_sha256 = @splat(6), .generated_ids_sha256 = ids_sha, .primary_sha256 = @splat(7), .historical_sha256 = @splat(8), .future_sha256 = @splat(9), .heldout_sha256 = @splat(10), .early_eos_observations = 3 };
    var evidence: FutureTrainingCaptureEvidence = .{ .generated_ids = ids, .generated_ids_count = 256, .generated_ids_sha256 = ids_sha, .source_before = boundary, .source_after = boundary, .early_eos_observations = 3, .backend_owned_full_logits_calls = 256, .backend_owned_full_logits_ns = 1, .backend_owned_frame_successes = 256, .generic_last_logits_calls = 0, .backend_owned_full_logits_selected = true, .heldout_sha256_before_fit = authority.heldout_sha256 };
    try evidence.validate(authority);
    evidence.source_after.kv_revision += 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, evidence.validate(authority));
    evidence.source_after = boundary;
    evidence.backend_owned_frame_successes = 255;
    try std.testing.expectError(error.InvalidNativeQualityPublication, evidence.validate(authority));
}

pub const Report = struct {
    allocator: std.mem.Allocator,
    frozen: Frozen,
    calibration_consumers: [256]queries.Consumer = @splat(std.mem.zeroes(queries.Consumer)),
    phase_durations_ns: PhaseDurations = .{},
    primary_query_collection: QueryCollectionDiagnostic = .{},
    dense512_query_collection: QueryCollectionDiagnostic = .{},
    query_collection_combined_before_fit_ns: u64 = 0,
    future_training_authority: ?@import("attention_match_future_decode_training.zig").Authority = null,
    future_training_capture: ?FutureTrainingCaptureEvidence = null,
    source_serving_prefill: bool = false,
    source_route: enum { generated_exact, producer_retained_exact } = .generated_exact,
    source_prefill_full_chunk_rows: usize = 0,
    source_prefill_full_chunk_count: usize = 0,
    source_prefill_tail_rows: usize = 0,
    source_prefill_final_token_rows: usize = 0,
    source_prefill_e4_fused_half_attempts: u64 = 0,
    source_prefill_e4_fused_half_hits: u64 = 0,
    source_prefill_e4_fused_half_pair_dispatches: u64 = 0,
    source_prefill_e4_fused_half_down_dispatches: u64 = 0,
    source_prefill_e4_fused_half_fallbacks: u64 = 0,
    source_prefill_e4_fused_half_fatal_failures: u64 = 0,
    source_prefill_e2_fused_half_attempts: u64 = 0,
    source_prefill_e2_fused_half_hits: u64 = 0,
    source_prefill_e2_fused_half_pair_6144_dispatches: u64 = 0,
    source_prefill_e2_fused_half_pair_12288_dispatches: u64 = 0,
    source_prefill_e2_fused_half_down_6144_dispatches: u64 = 0,
    source_prefill_e2_fused_half_down_12288_dispatches: u64 = 0,
    source_prefill_e2_fused_half_fallbacks: u64 = 0,
    source_prefill_e2_fused_half_fatal_failures: u64 = 0,
    source_prefill_prepared_geometry: u8 = 0,
    source_prefill_prepared_route: bool = false,
    source_prefill_prepared_runtime_total_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_total_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_private_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_private_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_shared_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_shared_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_managed_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_managed_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_norm_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_norm_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_dense_linear_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_dense_linear_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_quant_linear_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_quant_linear_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_scratch_bytes_before: u64 = 0,
    source_prefill_prepared_runtime_scratch_bytes_after: u64 = 0,
    source_prefill_prepared_runtime_buffer_count_before: u64 = 0,
    source_prefill_prepared_runtime_buffer_count_after: u64 = 0,
    source_prefill_prepared_retained_device_bytes: u64 = 0,
    collection_elapsed_ns: u64 = 0,
    model_slot_runtime_kv_bytes_before: u64 = 0,
    model_slot_runtime_kv_bytes_after: u64 = 0,
    continuation_runtime_kv_bytes: u64 = 0,
    compacted_greedy_runtime_kv_bytes_before: u64 = 0,
    compacted_greedy_runtime_kv_bytes_after: u64 = 0,
    compacted_teacher_runtime_kv_bytes_before: u64 = 0,
    compacted_teacher_runtime_kv_bytes_after: u64 = 0,
    model_slots_prepared_without_kv_allocation: bool = false,
    exact_greedy: Greedy = .{},
    compacted_greedy: Greedy = .{},
    exact_teacher: Scores = .{},
    compacted_teacher: Scores = .{},
    /// Populated only by Request.route_parity_diagnostic. Both values use the
    /// exact retained source; they are evidence only and confer no credit.
    exact_raw_greedy_diagnostic: ?Greedy = null,
    exact_prepared_teacher_diagnostic: ?Scores = null,
    route_parity_diagnostic_unqualified: bool = false,
    reference_sha256: [32]u8 = @splat(0),
    reference_revealed_after_both_greedy_branches: bool = false,
    source_unchanged: bool = false,
    source_owner_id: u64 = 0,
    compacted_owner_id: u64 = 0,
    query_scope_released: bool = false,
    query_scope_delta_bytes: ?u64 = null,
    admitted_peak_bytes: usize = 0,
    source_global_bytes: usize = 0,
    candidate_global_bytes: usize = 0,
    max_head_mass_error: f64 = 0,
    max_head_output_rmse: f64 = 0,
    fp64_accelerated_value_solves: u64 = 0,
    source_release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
    compacted_release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
    collection_complete: bool = false,
    candidate_source: enum { fitted, supplied, prepared_serving } = .fitted,
    fit_calls: usize = 1,
    calibration_executed: bool = true,
    admission: engine.Options.Admission = .fixed,
    chunk_rows: u16 = 1024,
    reject_exact_active_fixed_point: bool = false,
    fit_strategy: FitStrategy = .legacy,
    v18_policy_sha256: [32]u8 = @splat(0),
    v18_eligible_heads: usize = 0,
    v18_attempted_heads: usize = 0,
    v18_accepted_heads: usize = 0,
    v18_preserved_heads: usize = 0,
    v18_fallback_heads: usize = 0,
    v18_cache_ready_heads: usize = 0,
    v18_planned_exact_heads: usize = 0,
    v18_exact_f32_probability_heads: usize = 0,
    incremental_gram_chooser_calls: usize = 0,
    v18_query_batch_rows: usize = 0,
    v18_planned_head_workspace_bytes: usize = 0,
    v18_cache_phase_peak_bound_bytes: usize = 0,
    v18_bounded_heads: usize = 0,
    v18_bounded_delta_heads: usize = 0,
    v18_bounded_delta_tiles: usize = 0,
    v18_bounded_delta_backend_calls: usize = 0,
    v18_bounded_delta_packed_bytes_peak: usize = 0,
    v18_bounded_delta_staged_bytes_peak: usize = 0,
    v18_bounded_delta_fallbacks: usize = 0,
    v18_primary_budget_percent: u8 = 0,
    v18_budget_fallback_heads: usize = 0,
    v18_evidence_profile: V18EvidenceProfile = .none,
    v18_evidence_profile_sha256: [32]u8 = @splat(0),
    v18_codec_physical_profile: V18EvidenceProfile = .none,
    v18_codec_physical_profile_sha256: [32]u8 = @splat(0),
    v18_bounded_tsqr_policy_sha256: [32]u8 = @splat(0),
    v18_bounded_tsqr_heads: usize = 0,
    v18_bounded_tsqr_calls: usize = 0,
    v18_bounded_tsqr_workspace_peak_bytes: usize = 0,
    v18_bounded_tsqr_cancellation_checks: usize = 0,
    supplied_closure_sha256: ?[32]u8 = null,
    accepted_snapshot: ?AcceptedSnapshot = null,

    pub fn validateV18Evidence(self: Report) !void {
        if (self.fit_strategy == .legacy) {
            if (self.v18_eligible_heads != 0 or self.v18_attempted_heads != 0 or self.v18_accepted_heads != 0 or self.v18_preserved_heads != 0 or self.v18_fallback_heads != 0 or self.v18_cache_ready_heads != 0 or self.v18_planned_exact_heads != 0 or self.v18_query_batch_rows != 0 or self.v18_planned_head_workspace_bytes != 0 or self.v18_cache_phase_peak_bound_bytes != 0 or self.v18_bounded_heads != 0 or self.v18_bounded_delta_heads != 0 or self.v18_bounded_delta_tiles != 0 or self.v18_bounded_delta_backend_calls != 0 or self.v18_bounded_delta_packed_bytes_peak != 0 or self.v18_bounded_delta_staged_bytes_peak != 0 or self.v18_bounded_delta_fallbacks != 0 or self.v18_primary_budget_percent != 0 or self.v18_budget_fallback_heads != 0 or self.v18_evidence_profile != .none or !std.mem.allEqual(u8, &self.v18_evidence_profile_sha256, 0) or self.v18_codec_physical_profile != .none or !std.mem.allEqual(u8, &self.v18_codec_physical_profile_sha256, 0) or !std.mem.allEqual(u8, &self.v18_bounded_tsqr_policy_sha256, 0) or self.v18_bounded_tsqr_heads != 0 or self.v18_bounded_tsqr_calls != 0 or self.v18_bounded_tsqr_workspace_peak_bytes != 0 or self.v18_bounded_tsqr_cancellation_checks != 0)
                return error.InvalidNativeQualityPublication;
            return;
        }
        const expected: struct { batch: usize, workspace: usize } = switch (self.v18_evidence_profile) {
            .none => return error.InvalidNativeQualityPublication,
            .legacy_batch16_workspace512mib => .{ .batch = 16, .workspace = 512 * 1024 * 1024 },
            .private_budget95_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .private_hybrid_bounded_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .private_hybrid_bounded_accelerated_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .private_physical_budget_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .private_fp8_physical_budget_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .private_rotated4_physical_budget_batch64_workspace1gib => .{ .batch = 64, .workspace = 1024 * 1024 * 1024 },
            .resident_f16_fixed91_batch64_workspace2gib, .resident_fp8_fixed91_batch64_workspace2gib, .resident_rotated4_fixed91_batch64_workspace2gib => .{ .batch = 64, .workspace = 2 * 1024 * 1024 * 1024 },
            .resident_e4_32512_f16_page_tsqr_v1, .resident_e4_32512_fp8_page_tsqr_v1, .resident_e4_32512_rotated4_page_tsqr_v1 => .{ .batch = 64, .workspace = 2 * 1024 * 1024 * 1024 },
            .resident_e2_formal_f16_page_tsqr_v1, .resident_e2_formal_fp8_page_tsqr_v1, .resident_e2_formal_rotated4_page_tsqr_v1 => .{ .batch = 64, .workspace = 2 * 1024 * 1024 * 1024 },
            .resident_e2_f16_fixed60_batch64_workspace2gib, .resident_e2_fp8_fixed60_batch64_workspace2gib, .resident_e2_rotated4_fixed60_batch64_workspace2gib => .{ .batch = 64, .workspace = 2 * 1024 * 1024 * 1024 },
            .bounded_f16_codec_physical_v1, .bounded_fp8_codec_physical_v1, .bounded_rotated4_codec_physical_v1 => return error.InvalidNativeQualityPublication,
        };
        const physical_profile = self.v18_evidence_profile == .private_physical_budget_batch64_workspace1gib or
            self.v18_evidence_profile == .private_fp8_physical_budget_batch64_workspace1gib or
            self.v18_evidence_profile == .private_rotated4_physical_budget_batch64_workspace1gib;
        const resident_e4_flat_profile = self.v18_evidence_profile == .resident_f16_fixed91_batch64_workspace2gib or
            self.v18_evidence_profile == .resident_fp8_fixed91_batch64_workspace2gib or
            self.v18_evidence_profile == .resident_rotated4_fixed91_batch64_workspace2gib;
        const resident_e4_bounded_profile = self.v18_evidence_profile == .resident_e4_32512_f16_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e4_32512_fp8_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e4_32512_rotated4_page_tsqr_v1;
        const resident_e4_profile = resident_e4_flat_profile or resident_e4_bounded_profile;
        const resident_e2_profile = self.v18_evidence_profile == .resident_e2_f16_fixed60_batch64_workspace2gib or
            self.v18_evidence_profile == .resident_e2_fp8_fixed60_batch64_workspace2gib or
            self.v18_evidence_profile == .resident_e2_rotated4_fixed60_batch64_workspace2gib or
            self.v18_evidence_profile == .resident_e2_formal_f16_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e2_formal_fp8_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e2_formal_rotated4_page_tsqr_v1;
        const resident_e2_bounded_profile = self.v18_evidence_profile == .resident_e2_formal_f16_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e2_formal_fp8_page_tsqr_v1 or
            self.v18_evidence_profile == .resident_e2_formal_rotated4_page_tsqr_v1;
        const resident_profile = resident_e4_profile or resident_e2_profile;
        const resident_fused_prefill = resident_e4_flat_profile and self.frozen.execution.resident_gated_ffn_fusion;
        const bounded_profile = self.v18_evidence_profile == .private_hybrid_bounded_batch64_workspace1gib or self.v18_evidence_profile == .private_hybrid_bounded_accelerated_batch64_workspace1gib or resident_e4_bounded_profile or resident_e2_bounded_profile;
        const bounded_accelerated_profile = self.v18_evidence_profile == .private_hybrid_bounded_accelerated_batch64_workspace1gib or resident_e4_bounded_profile or resident_e2_bounded_profile;
        const codec_profile = self.v18_codec_physical_profile == .bounded_f16_codec_physical_v1 or self.v18_codec_physical_profile == .bounded_fp8_codec_physical_v1 or self.v18_codec_physical_profile == .bounded_rotated4_codec_physical_v1;
        const budget_profile = physical_profile or resident_profile;
        const adaptive_total = std.math.add(usize, std.math.add(usize, self.v18_accepted_heads, self.v18_preserved_heads) catch return error.InvalidNativeQualityPublication, self.v18_planned_exact_heads) catch return error.InvalidNativeQualityPublication;
        const ready_total = std.math.add(usize, std.math.add(usize, self.v18_cache_ready_heads, self.v18_fallback_heads) catch return error.InvalidNativeQualityPublication, self.v18_planned_exact_heads) catch return error.InvalidNativeQualityPublication;
        const candidate_scaled = if (resident_profile) std.math.mul(usize, self.candidate_global_bytes, 3) catch return error.InvalidNativeQualityPublication else 0;
        const source_scaled = if (resident_profile) std.math.mul(usize, self.source_global_bytes, 2) catch return error.InvalidNativeQualityPublication else 0;
        if (resident_e4_profile and (self.v18_eligible_heads != 8 or self.v18_attempted_heads != 8 or
            self.v18_accepted_heads != 7 or self.v18_preserved_heads != 0 or self.v18_fallback_heads != 0 or
            self.v18_cache_ready_heads != 7 or self.v18_planned_exact_heads != 1 or
            self.frozen.validated_heads != 7 or self.frozen.preserved_heads != 0 or
            self.v18_exact_f32_probability_heads != 7 or self.incremental_gram_chooser_calls != 21 or
            self.source_global_bytes == 0 or self.candidate_global_bytes == 0 or candidate_scaled > source_scaled or
            !std.math.isFinite(self.max_head_mass_error) or !std.math.isFinite(self.max_head_output_rmse) or
            self.max_head_mass_error < 0 or self.max_head_output_rmse < 0 or
            self.max_head_mass_error > 0.05 or self.max_head_output_rmse > 0.05))
            return error.InvalidNativeQualityPublication;
        if (resident_e2_profile and (self.v18_eligible_heads != 3 or self.v18_attempted_heads != 3 or
            self.v18_accepted_heads != 2 or self.v18_preserved_heads != 0 or self.v18_fallback_heads != 0 or
            self.v18_cache_ready_heads != 2 or self.v18_planned_exact_heads != 1 or
            self.frozen.validated_heads != 2 or self.frozen.preserved_heads != 0 or
            self.v18_exact_f32_probability_heads != 2 or self.incremental_gram_chooser_calls != 6 or
            self.source_global_bytes == 0 or self.candidate_global_bytes == 0 or candidate_scaled > source_scaled or
            !std.math.isFinite(self.max_head_mass_error) or !std.math.isFinite(self.max_head_output_rmse) or
            self.max_head_mass_error < 0 or self.max_head_output_rmse < 0 or
            self.max_head_mass_error > 0.05 or self.max_head_output_rmse > 0.05))
            return error.InvalidNativeQualityPublication;
        if (resident_fused_prefill) {
            const expected_fused_calls = std.math.mul(u64, @intCast(self.source_prefill_full_chunk_count), 42) catch
                return error.InvalidNativeQualityPublication;
            // The retained 8128-row serving plan is fifteen full 512-row
            // chunks plus one 448-row tail. The typed resident route admits
            // only 511/512 rows; the short tail cleanly returns to the
            // existing Gemma vector fallback before entering the C backend.
            const expected_attempts = std.math.mul(u64, 15, 42) catch unreachable;
            const expected_hits = std.math.mul(u64, 15, 42) catch unreachable;
            if (!self.source_serving_prefill or self.source_prefill_full_chunk_rows != 512 or
                self.source_prefill_full_chunk_count != 15 or self.source_prefill_tail_rows != 448 or
                self.source_prefill_final_token_rows != 0 or expected_fused_calls != expected_hits or
                self.source_prefill_e4_fused_half_attempts != expected_attempts or
                self.source_prefill_e4_fused_half_hits != expected_hits or
                self.source_prefill_e4_fused_half_pair_dispatches != expected_hits or
                self.source_prefill_e4_fused_half_down_dispatches != expected_hits or
                self.source_prefill_e4_fused_half_fallbacks != 0 or
                self.source_prefill_e4_fused_half_fatal_failures != 0)
                return error.InvalidNativeQualityPublication;
        }
        if (bounded_profile) {
            if (!codec_profile or !physicalProfileEncodingMatches(self.v18_codec_physical_profile, self.frozen.encoding) or
                !std.mem.eql(u8, &self.v18_codec_physical_profile_sha256, &v18EvidenceProfileDigest(self.v18_codec_physical_profile)) or
                !std.mem.eql(u8, &self.v18_bounded_tsqr_policy_sha256, &boundedTsqrValuePolicyDigest()) or
                self.v18_bounded_tsqr_heads != self.v18_bounded_heads or self.v18_bounded_tsqr_heads == 0 or self.v18_bounded_tsqr_calls < self.v18_bounded_tsqr_heads or
                self.v18_bounded_tsqr_workspace_peak_bytes == 0 or self.v18_bounded_tsqr_workspace_peak_bytes > self.v18_planned_head_workspace_bytes or self.v18_bounded_tsqr_cancellation_checks < self.v18_bounded_tsqr_calls)
                return error.InvalidNativeQualityPublication;
        } else if (self.v18_codec_physical_profile != .none or !std.mem.allEqual(u8, &self.v18_codec_physical_profile_sha256, 0) or
            !std.mem.allEqual(u8, &self.v18_bounded_tsqr_policy_sha256, 0) or self.v18_bounded_tsqr_heads != 0 or self.v18_bounded_tsqr_calls != 0 or
            self.v18_bounded_tsqr_workspace_peak_bytes != 0 or self.v18_bounded_tsqr_cancellation_checks != 0)
            return error.InvalidNativeQualityPublication;
        if (std.mem.allEqual(u8, &self.v18_policy_sha256, 0) or self.v18_eligible_heads == 0 or
            !std.mem.eql(u8, &self.v18_evidence_profile_sha256, &v18EvidenceProfileDigest(self.v18_evidence_profile)) or
            self.v18_eligible_heads != self.v18_attempted_heads or
            self.v18_attempted_heads != adaptive_total or ready_total != self.v18_attempted_heads or
            (resident_profile and self.v18_planned_exact_heads != 1) or (!resident_profile and self.v18_planned_exact_heads != 0) or
            self.v18_fallback_heads > self.v18_preserved_heads or self.v18_query_batch_rows != expected.batch or
            self.v18_planned_head_workspace_bytes != expected.workspace or
            self.v18_cache_phase_peak_bound_bytes > self.v18_planned_head_workspace_bytes or
            self.v18_bounded_heads > self.v18_attempted_heads or self.v18_budget_fallback_heads > 3 or
            (budget_profile and (self.v18_primary_budget_percent < 50 or self.v18_primary_budget_percent >= 95)) or
            (resident_e4_profile and self.v18_primary_budget_percent != 91) or
            (resident_e2_profile and (self.v18_primary_budget_percent != 60 or self.v18_budget_fallback_heads != 0)) or
            (!budget_profile and (self.v18_primary_budget_percent != 0 or self.v18_budget_fallback_heads != 0)) or
            !physicalProfileEncodingMatches(self.v18_evidence_profile, self.frozen.encoding) or
            (bounded_profile and self.v18_bounded_heads == 0) or
            (!bounded_profile and self.v18_bounded_heads != 0) or
            (bounded_accelerated_profile and (self.v18_bounded_delta_heads != self.v18_bounded_heads or self.v18_bounded_delta_tiles == 0 or self.v18_bounded_delta_backend_calls != self.v18_bounded_delta_tiles * 2 or self.v18_bounded_delta_packed_bytes_peak == 0 or self.v18_bounded_delta_staged_bytes_peak == 0 or self.v18_bounded_delta_fallbacks != 0)) or
            (!bounded_accelerated_profile and (self.v18_bounded_delta_heads != 0 or self.v18_bounded_delta_tiles != 0 or self.v18_bounded_delta_backend_calls != 0 or self.v18_bounded_delta_packed_bytes_peak != 0 or self.v18_bounded_delta_staged_bytes_peak != 0 or self.v18_bounded_delta_fallbacks != 0)))
            return error.InvalidNativeQualityPublication;
    }

    pub fn deinit(self: *Report) void {
        self.allocator.free(self.exact_greedy.public_text);
        self.allocator.free(self.compacted_greedy.public_text);
        if (self.exact_raw_greedy_diagnostic) |*diagnostic|
            self.allocator.free(diagnostic.public_text);
        self.exact_greedy.public_text = &.{};
        self.compacted_greedy.public_text = &.{};
        if (self.exact_raw_greedy_diagnostic) |*diagnostic|
            diagnostic.public_text = &.{};
    }
};

test "context route parity diagnostic report releases the independent raw greedy text" {
    var report: Report = .{ .allocator = std.testing.allocator, .frozen = undefined };
    report.exact_greedy.public_text = try std.testing.allocator.dupe(u8, "prepared");
    report.compacted_greedy.public_text = try std.testing.allocator.dupe(u8, "candidate");
    report.exact_raw_greedy_diagnostic = .{
        .public_text = try std.testing.allocator.dupe(u8, "raw"),
    };
    report.deinit();
    try std.testing.expectEqual(@as(usize, 0), report.exact_raw_greedy_diagnostic.?.public_text.len);
}

test "context v18 evidence validator accepts only complete legacy or private profile tuples" {
    var frozen: Frozen = undefined;
    frozen.encoding = .f16;
    var report: Report = .{
        .allocator = std.testing.allocator,
        .frozen = frozen,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .v18_policy_sha256 = @splat(1),
        .v18_eligible_heads = 2,
        .v18_attempted_heads = 2,
        .v18_accepted_heads = 2,
        .v18_cache_ready_heads = 2,
        .v18_evidence_profile = .legacy_batch16_workspace512mib,
        .v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.legacy_batch16_workspace512mib),
        .v18_query_batch_rows = 16,
        .v18_planned_head_workspace_bytes = 512 * 1024 * 1024,
        .v18_cache_phase_peak_bound_bytes = 512 * 1024 * 1024,
    };
    try report.validateV18Evidence();
    report.v18_evidence_profile = .private_budget95_batch64_workspace1gib;
    report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_budget95_batch64_workspace1gib);
    report.v18_query_batch_rows = 64;
    report.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
    try report.validateV18Evidence();
    report.v18_bounded_heads = 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_evidence_profile = .private_hybrid_bounded_batch64_workspace1gib;
    report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_hybrid_bounded_batch64_workspace1gib);
    report.v18_codec_physical_profile = .bounded_f16_codec_physical_v1;
    report.v18_codec_physical_profile_sha256 = v18EvidenceProfileDigest(.bounded_f16_codec_physical_v1);
    report.v18_bounded_tsqr_policy_sha256 = boundedTsqrValuePolicyDigest();
    report.v18_bounded_tsqr_heads = 1;
    report.v18_bounded_tsqr_calls = 1;
    report.v18_bounded_tsqr_workspace_peak_bytes = 1024;
    report.v18_bounded_tsqr_cancellation_checks = 1;
    report.v18_primary_budget_percent = 0;
    try report.validateV18Evidence();
    report.v18_bounded_heads = 0;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_bounded_heads = 1;
    report.v18_query_batch_rows = 16;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_query_batch_rows = 64;
    report.v18_evidence_profile = .private_hybrid_bounded_accelerated_batch64_workspace1gib;
    report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_hybrid_bounded_accelerated_batch64_workspace1gib);
    report.v18_bounded_delta_heads = 1;
    report.v18_bounded_delta_tiles = 7;
    report.v18_bounded_delta_backend_calls = 14;
    report.v18_bounded_delta_packed_bytes_peak = 4096;
    report.v18_bounded_delta_staged_bytes_peak = 8192;
    try report.validateV18Evidence();
    report.v18_bounded_delta_backend_calls = 13;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_bounded_delta_backend_calls = 14;
    report.v18_bounded_delta_fallbacks = 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_bounded_delta_fallbacks = 0;
    report.v18_evidence_profile_sha256[0] ^= 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
}

test "resident two GiB evidence profiles are encoding specific and fixed p91" {
    inline for (.{
        .{ checkpoint.Encoding.f16, V18EvidenceProfile.resident_f16_fixed91_batch64_workspace2gib },
        .{ checkpoint.Encoding.fp8, V18EvidenceProfile.resident_fp8_fixed91_batch64_workspace2gib },
        .{ checkpoint.Encoding.rotated4, V18EvidenceProfile.resident_rotated4_fixed91_batch64_workspace2gib },
    }) |row| {
        var frozen: Frozen = undefined;
        frozen.encoding = row[0];
        frozen.execution = .{};
        frozen.validated_heads = 7;
        frozen.preserved_heads = 0;
        var report: Report = .{
            .allocator = std.testing.allocator,
            .frozen = frozen,
            .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
            .v18_policy_sha256 = @splat(1),
            .v18_eligible_heads = 8,
            .v18_attempted_heads = 8,
            .v18_accepted_heads = 7,
            .v18_cache_ready_heads = 7,
            .v18_planned_exact_heads = 1,
            .v18_exact_f32_probability_heads = 7,
            .incremental_gram_chooser_calls = 21,
            .source_global_bytes = 300,
            .candidate_global_bytes = 200,
            .v18_query_batch_rows = 64,
            .v18_planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024,
            .v18_cache_phase_peak_bound_bytes = 1024,
            .v18_primary_budget_percent = 91,
            .v18_evidence_profile = row[1],
            .v18_evidence_profile_sha256 = v18EvidenceProfileDigest(row[1]),
        };
        try report.validateV18Evidence();
        // V7 is a distinct execution policy: it must prove every one of the
        // fifteen eligible 512-row chunks across all 42 transformer layers,
        // The 448-row tail cleanly declines before the C backend.
        report.frozen.execution.resident_gated_ffn_fusion = true;
        report.source_serving_prefill = true;
        report.source_prefill_full_chunk_rows = 512;
        report.source_prefill_full_chunk_count = 15;
        report.source_prefill_tail_rows = 448;
        report.source_prefill_e4_fused_half_attempts = 630;
        report.source_prefill_e4_fused_half_hits = 630;
        report.source_prefill_e4_fused_half_pair_dispatches = 630;
        report.source_prefill_e4_fused_half_down_dispatches = 630;
        report.source_prefill_e4_fused_half_fallbacks = 0;
        try report.validateV18Evidence();
        inline for (.{ "attempts", "hits", "pair", "down", "fallback", "fatal" }) |mutation| {
            var corrupt = report;
            if (std.mem.eql(u8, mutation, "attempts")) corrupt.source_prefill_e4_fused_half_attempts -= 1;
            if (std.mem.eql(u8, mutation, "hits")) corrupt.source_prefill_e4_fused_half_hits -= 1;
            if (std.mem.eql(u8, mutation, "pair")) corrupt.source_prefill_e4_fused_half_pair_dispatches -= 1;
            if (std.mem.eql(u8, mutation, "down")) corrupt.source_prefill_e4_fused_half_down_dispatches -= 1;
            if (std.mem.eql(u8, mutation, "fallback")) corrupt.source_prefill_e4_fused_half_fallbacks = 1;
            if (std.mem.eql(u8, mutation, "fatal")) corrupt.source_prefill_e4_fused_half_fatal_failures = 1;
            try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt.validateV18Evidence());
        }
        var corrupt_shape = report;
        corrupt_shape.source_serving_prefill = false;
        try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt_shape.validateV18Evidence());
        corrupt_shape = report;
        corrupt_shape.source_prefill_full_chunk_rows = 256;
        try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt_shape.validateV18Evidence());
        corrupt_shape = report;
        corrupt_shape.source_prefill_full_chunk_count = 16;
        try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt_shape.validateV18Evidence());
        corrupt_shape = report;
        corrupt_shape.source_prefill_tail_rows = 447;
        try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt_shape.validateV18Evidence());
        corrupt_shape = report;
        corrupt_shape.source_prefill_final_token_rows = 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, corrupt_shape.validateV18Evidence());
        report.frozen.execution.resident_gated_ffn_fusion = false;
        report.source_serving_prefill = false;
        report.source_prefill_full_chunk_rows = 0;
        report.source_prefill_full_chunk_count = 0;
        report.source_prefill_tail_rows = 0;
        report.source_prefill_e4_fused_half_attempts = 0;
        report.source_prefill_e4_fused_half_hits = 0;
        report.source_prefill_e4_fused_half_pair_dispatches = 0;
        report.source_prefill_e4_fused_half_down_dispatches = 0;
        report.source_prefill_e4_fused_half_fallbacks = 0;
        report.v18_planned_head_workspace_bytes = 1024 * 1024 * 1024;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024;
        report.v18_primary_budget_percent = 90;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_primary_budget_percent = 91;
        report.v18_evidence_profile_sha256[0] ^= 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(row[1]);
        report.v18_planned_exact_heads = 0;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_planned_exact_heads = 2;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_planned_exact_heads = 1;
        report.v18_accepted_heads = 6;
        report.v18_preserved_heads = 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_accepted_heads = 7;
        report.v18_preserved_heads = 0;
        report.source_global_bytes = 0;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.source_global_bytes = 300;
        report.max_head_mass_error = 0.050001;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    }
    try std.testing.expect(!physicalProfileEncodingMatches(.resident_f16_fixed91_batch64_workspace2gib, .fp8));
    try std.testing.expect(!std.mem.eql(u8, &v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib), &v18EvidenceProfileDigest(.resident_f16_fixed91_batch64_workspace2gib)));
}

test "E2 resident evidence is exactly two adaptive plus one planned exact at fixed p60" {
    inline for (.{
        .{ checkpoint.Encoding.f16, V18EvidenceProfile.resident_e2_f16_fixed60_batch64_workspace2gib },
        .{ checkpoint.Encoding.fp8, V18EvidenceProfile.resident_e2_fp8_fixed60_batch64_workspace2gib },
        .{ checkpoint.Encoding.rotated4, V18EvidenceProfile.resident_e2_rotated4_fixed60_batch64_workspace2gib },
    }) |row| {
        var frozen: Frozen = undefined;
        frozen.encoding = row[0];
        frozen.execution = .{};
        frozen.validated_heads = 2;
        frozen.preserved_heads = 0;
        var report: Report = .{ .allocator = std.testing.allocator, .frozen = frozen, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .v18_policy_sha256 = @splat(1), .v18_eligible_heads = 3, .v18_attempted_heads = 3, .v18_accepted_heads = 2, .v18_cache_ready_heads = 2, .v18_planned_exact_heads = 1, .v18_exact_f32_probability_heads = 2, .incremental_gram_chooser_calls = 6, .source_global_bytes = 49_938_432, .candidate_global_bytes = 33_226_752, .v18_query_batch_rows = 64, .v18_planned_head_workspace_bytes = 2 * 1024 * 1024 * 1024, .v18_cache_phase_peak_bound_bytes = 1_500_000_000, .v18_primary_budget_percent = 60, .v18_evidence_profile = row[1], .v18_evidence_profile_sha256 = v18EvidenceProfileDigest(row[1]) };
        try report.validateV18Evidence();
        report.v18_preserved_heads = 1;
        report.v18_accepted_heads = 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_preserved_heads = 0;
        report.v18_accepted_heads = 2;
        report.v18_primary_budget_percent = 61;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_primary_budget_percent = 60;
        report.v18_planned_exact_heads = 0;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    }
}

test "context E4 32K resident bounded profiles bind distinct destination codecs over f16 source" {
    const cases = [_]struct { checkpoint.Encoding, V18EvidenceProfile }{
        .{ .f16, .resident_e4_32512_f16_page_tsqr_v1 },
        .{ .fp8, .resident_e4_32512_fp8_page_tsqr_v1 },
        .{ .rotated4, .resident_e4_32512_rotated4_page_tsqr_v1 },
    };
    inline for (cases, 0..) |row, index| {
        try std.testing.expect(physicalProfileEncodingMatches(row[1], row[0]));
        inline for (cases, 0..) |other, other_index| {
            if (index != other_index) {
                try std.testing.expect(!physicalProfileEncodingMatches(row[1], other[0]));
                try std.testing.expect(!std.mem.eql(u8, &v18EvidenceProfileDigest(row[1]), &v18EvidenceProfileDigest(other[1])));
            }
        }
    }
}

test "context 32K bounded TSQR seal binds all codecs and rejects each corrupted authority" {
    const cases = [_]struct { checkpoint.Encoding, V18EvidenceProfile }{
        .{ .f16, .bounded_f16_codec_physical_v1 },
        .{ .fp8, .bounded_fp8_codec_physical_v1 },
        .{ .rotated4, .bounded_rotated4_codec_physical_v1 },
    };
    inline for (cases) |row| {
        var frozen: Frozen = undefined;
        frozen.encoding = row[0];
        var report: Report = .{ .allocator = std.testing.allocator, .frozen = frozen, .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .v18_policy_sha256 = @splat(1), .v18_eligible_heads = 3, .v18_attempted_heads = 3, .v18_accepted_heads = 3, .v18_cache_ready_heads = 3, .v18_query_batch_rows = 64, .v18_planned_head_workspace_bytes = 1024 * 1024 * 1024, .v18_cache_phase_peak_bound_bytes = 512 * 1024 * 1024, .v18_bounded_heads = 3, .v18_bounded_delta_heads = 3, .v18_bounded_delta_tiles = 9, .v18_bounded_delta_backend_calls = 18, .v18_bounded_delta_packed_bytes_peak = 4096, .v18_bounded_delta_staged_bytes_peak = 8192, .v18_primary_budget_percent = 0, .v18_evidence_profile = .private_hybrid_bounded_accelerated_batch64_workspace1gib, .v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_hybrid_bounded_accelerated_batch64_workspace1gib), .v18_codec_physical_profile = row[1], .v18_codec_physical_profile_sha256 = v18EvidenceProfileDigest(row[1]), .v18_bounded_tsqr_policy_sha256 = boundedTsqrValuePolicyDigest(), .v18_bounded_tsqr_heads = 3, .v18_bounded_tsqr_calls = 3, .v18_bounded_tsqr_workspace_peak_bytes = 1024, .v18_bounded_tsqr_cancellation_checks = 3 };
        try report.validateV18Evidence();
        report.v18_bounded_tsqr_calls = 0;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_bounded_tsqr_calls = 3;
        report.v18_bounded_tsqr_policy_sha256[0] ^= 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
        report.v18_bounded_tsqr_policy_sha256[0] ^= 1;
        report.v18_codec_physical_profile_sha256[0] ^= 1;
        try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    }
}

test "context FP8 physical evidence requires its encoding-specific p91 tuple" {
    var frozen: Frozen = undefined;
    frozen.encoding = .fp8;
    var report: Report = .{
        .allocator = std.testing.allocator,
        .frozen = frozen,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .v18_policy_sha256 = @splat(1),
        .v18_eligible_heads = 8,
        .v18_attempted_heads = 8,
        .v18_accepted_heads = 7,
        .v18_preserved_heads = 1,
        .v18_cache_ready_heads = 8,
        .v18_query_batch_rows = 64,
        .v18_planned_head_workspace_bytes = 1024 * 1024 * 1024,
        .v18_cache_phase_peak_bound_bytes = 1024 * 1024 * 1024,
        .v18_primary_budget_percent = 91,
        .v18_budget_fallback_heads = 1,
        .v18_evidence_profile = .private_fp8_physical_budget_batch64_workspace1gib,
        .v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_fp8_physical_budget_batch64_workspace1gib),
    };
    try report.validateV18Evidence();

    report.v18_evidence_profile = .private_physical_budget_batch64_workspace1gib;
    report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_physical_budget_batch64_workspace1gib);
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_evidence_profile = .private_fp8_physical_budget_batch64_workspace1gib;
    report.v18_evidence_profile_sha256 = v18EvidenceProfileDigest(.private_fp8_physical_budget_batch64_workspace1gib);
    report.v18_primary_budget_percent = 95;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
    report.v18_primary_budget_percent = 91;
    report.v18_budget_fallback_heads = 4;
    try std.testing.expectError(error.InvalidNativeQualityPublication, report.validateV18Evidence());
}

test "accepted native quality snapshot is hash bound and never overwritten" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const wire = "f16-v3 accepted candidate";
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(wire, &digest, .{});

    var wrong = digest;
    wrong[0] ^= 1;
    try std.testing.expectError(error.KvCheckpointResumeMismatch, writeAcceptedSnapshot(io, .{ .output = tmp.dir }, wire, wrong));

    const receipt = try writeAcceptedSnapshot(io, .{ .output = tmp.dir }, wire, digest);
    try std.testing.expectEqualStrings(accepted_snapshot_name, receipt.path);
    try std.testing.expectEqual(wire.len, receipt.bytes);
    try std.testing.expectEqualSlices(u8, &digest, &receipt.sha256);
    const restored = try tmp.dir.readFileAlloc(io, accepted_snapshot_name, std.testing.allocator, .limited(1024));
    defer std.testing.allocator.free(restored);
    try std.testing.expectEqualStrings(wire, restored);

    const replacement = "replacement";
    var replacement_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(replacement, &replacement_digest, .{});
    var overwrite_rejected = false;
    _ = writeAcceptedSnapshot(io, .{ .output = tmp.dir }, replacement, replacement_digest) catch {
        overwrite_rejected = true;
    };
    try std.testing.expect(overwrite_rejected);
    const unchanged = try tmp.dir.readFileAlloc(io, accepted_snapshot_name, std.testing.allocator, .limited(1024));
    defer std.testing.allocator.free(unchanged);
    try std.testing.expectEqualStrings(wire, unchanged);
}

test "attention_match Metal quality admission keeps query clones private and references opaque" {
    const request: Request = .{ .encoding = .f16 };
    try std.testing.expectEqual(@as(usize, 32768), try request.capacity(32512, 131072));
    try std.testing.expectError(error.CompactionQueryLimit, request.capacity(32512, 32768));
    try std.testing.expectError(error.CompactionQueryLimit, request.capacity(32513, 131072));
    const Provider = struct {
        calls: usize = 0,
        fn reveal(raw: *anyopaque) ![]const i64 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            return &.{ 7, 9 };
        }
    };
    var provider: Provider = .{};
    const reference: Reference = .{ .count = 2, .sha256 = try teacher.referenceSequenceDigest(&.{ 7, 9 }), .provider = .{ .ctx = &provider, .reveal_fn = Provider.reveal } };
    var calibration: queries.Seal = .{
        .source_length = 7936,
        .repeat_length = 512,
        .consumer_count = 1,
        .batch_count = 1,
        .train_positions = 256,
        .heldout_positions = 64,
        .train_elements = 512,
        .heldout_elements = 128,
        .geometry_sha256 = @splat(9),
        .train_positions_sha256 = @splat(10),
        .heldout_positions_sha256 = @splat(11),
        .train_values_sha256 = @splat(12),
        .heldout_values_sha256 = @splat(13),
        .sha256 = @splat(0),
    };
    calibration.sha256 = calibration.digest();
    var frozen: Frozen = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .observed = .{ .destination = .{ .device_registry_id = 1, .physical_memory_bytes = 24 * 1024 * 1024 * 1024, .device_name_sha256 = @splat(13), .compiler_identity_sha256 = @splat(14) }, .backend_environment_sha256 = @splat(15) },
        .execution = .{ .forward_frame = true, .precomputed_tail = true },
        .configuration_sha256 = @splat(0),
        .implementation_sha256 = @splat(5),
        .prompt_sha256 = @splat(6),
        .checkpoint_sha256 = @splat(7),
        .installed_sha256 = @splat(7),
        .source_checkpoint_sha256 = @splat(8),
        .calibration = calibration,
        .source_revision = 1,
        .destination_revision_before_install = 1,
        .installed_revision = 2,
        .logical_tokens = 7936,
        .completed_chunks = 1,
        .validated_heads = 3,
        .protected_prefix_rows = 128,
        .protected_recent_rows = 2048,
    };
    frozen.configuration_sha256 = native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256);
    frozen.installed_sha256 = @splat(0);
    try std.testing.expectError(error.InvalidNativeQualityPublication, reference.reveal(frozen, 256));
    try std.testing.expectEqual(@as(usize, 0), provider.calls);
    frozen.installed_sha256 = @splat(8);
    try std.testing.expectError(error.InvalidNativeQualityPublication, reference.reveal(frozen, 256));
    try std.testing.expectEqual(@as(usize, 0), provider.calls);
    frozen.installed_sha256 = frozen.checkpoint_sha256;
    // The source and installation destination have independent revisions.
    frozen.source_revision = 50;
    try frozen.validate();
    // Historical fixed admission permits an all-F16 fallback even when its
    // requested wire encoding is FP8. Cumulative all-format admission is an
    // explicit codec qualification and requires observed codec activity.
    frozen.encoding = .fp8;
    try frozen.validate();
    frozen.admission = .cumulative_ladder_4_3_2;
    frozen.configuration_sha256 = admissionEncodingConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point, frozen.encoding);
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.fp8_heads = 1;
    try frozen.validate();
    frozen.fp8_heads = 0;
    frozen.encoding = .f16;
    frozen.admission = .fixed;
    frozen.configuration_sha256 = native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256);
    // Supplied candidates are authenticated artifacts, never relabeled as a
    // production fit under a different admission strategy.  This must remain
    // false even when a forged record recomputes the matching fingerprint.
    frozen.candidate_source = .supplied;
    frozen.fit_calls = 0;
    frozen.calibration_executed = false;
    frozen.calibration = null;
    frozen.admission = .cumulative_ladder_4_3_2;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.admission = .fixed;
    frozen.reject_exact_active_fixed_point = true;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.reject_exact_active_fixed_point = false;
    frozen.candidate_source = .fitted;
    frozen.fit_calls = 1;
    frozen.calibration_executed = true;
    frozen.calibration = calibration;
    frozen.admission = .cumulative_ladder_4_3_2;
    frozen.chunk_rows = 128;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    try frozen.validate();
    frozen.chunk_rows = 64;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.admission = .fixed;
    frozen.chunk_rows = 256;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.chunk_rows = 1024;
    frozen.configuration_sha256 = admissionConfigurationFingerprint(native_release.configurationFingerprint(frozen.execution, frozen.observed.backend_environment_sha256), frozen.admission, frozen.chunk_rows, frozen.reject_exact_active_fixed_point);
    frozen.destination_revision_before_install += 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, frozen.validate());
    frozen.destination_revision_before_install -= 1;
    frozen.calibration.?.train_values_sha256[0] ^= 1;
    try std.testing.expectError(error.InvalidNativeQualityPublication, reference.reveal(frozen, 256));
    try std.testing.expectEqual(@as(usize, 0), provider.calls);
    frozen.calibration.?.train_values_sha256[0] ^= 1;
    try std.testing.expectEqualSlices(i64, &.{ 7, 9 }, try reference.reveal(frozen, 256));
    try std.testing.expectEqual(@as(usize, 1), provider.calls);
}

test "attention_match Metal quality route permits observed cold staging and rejects warm or partial fallbacks" {
    try validatePreparedStep(0, 35, true);
    try validatePreparedStep(35, 35, true);
    try validatePreparedStep(35, 35, false);
    try validatePreparedStep(42, 42, false);
    try std.testing.expectError(error.InvalidNativeQualityPreparedRoute, validatePreparedStep(0, 35, false));
    try std.testing.expectError(error.InvalidNativeQualityPreparedRoute, validatePreparedStep(34, 35, true));
    try std.testing.expectError(error.InvalidNativeQualityPreparedRoute, validatePreparedStep(36, 35, false));
    try std.testing.expectError(error.InvalidNativeQualityPreparedRoute, validatePreparedStep(0, 0, true));
}
