// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Host reference orchestration for attention_match_v1. Query capture and device
//! execution remain separate adapters: this module never substitutes cached keys
//! for queries and never silently routes a device request through the CPU.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("attention_match.zig");
const validation_accelerate = @import("../../ops/metal_attention_match_validation.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const cumulative = @import("attention_match_cumulative_admission.zig");
const cumulative_cache = @import("attention_match_cumulative_cache.zig");

pub const ReferenceBatch = struct {
    consumer_layer: u32,
    query_head: u32,
    /// Post-RoPE queries from an isolated repeat-prefill/self-study run. These
    /// positions must follow the represented history and must be disjoint across
    /// training and held-out sets. Each consumer contributes equal sample counts.
    train: []const f32,
    heldout: []const f32,
    train_positions: []const u64,
    heldout_positions: []const u64,
    /// Optional boundaries between independently sampled TRAIN position
    /// domains. Positions are monotonic within a domain, not across domains.
    independent_train_split: ?usize = null,
    /// Requires independent_train_split and introduces a third domain.
    independent_train_second_split: ?usize = null,
};

pub const Layout = struct {
    query_heads_by_layer: []const u32,
    scales_by_donor: []const f64,
    softcaps_by_donor: []const f64,
};

pub const SourceDonorMetadata = struct {
    layer: u32,
    window: u32,
    head_count: usize,
    head_dim: usize,
    rows: usize = 0,
    physical_bytes: usize = 0,
};

pub const SourceMetadata = struct {
    identity: checkpoint.Identity,
    logical_length: u64,
    tokens: []const u32,
    logits: []const f32,
    layer_donors: []const u32,
    donors: []const SourceDonorMetadata = &.{},
    full_donors: []const checkpoint.Donor = &.{},
    structure_sha256: [32]u8,
    state_sha256: [32]u8,
    source_census_physical_bytes: usize,
    source_global_storage_bytes: usize = 0,

    pub fn donorLayer(self: @This(), donor: usize) !u32 {
        if (self.full_donors.len != 0) {
            if (donor >= self.full_donors.len) return error.InvalidKvCheckpoint;
            return self.full_donors[donor].layer;
        }
        if (donor >= self.donors.len) return error.InvalidKvCheckpoint;
        return self.donors[donor].layer;
    }

    pub fn belongs(self: @This(), layout: Layout, batch: ReferenceBatch, donor: usize, head_index: usize) bool {
        const head_count = if (self.full_donors.len != 0)
            if (donor < self.full_donors.len) self.full_donors[donor].heads.len else 0
        else if (donor < self.donors.len) self.donors[donor].head_count else 0;
        if (batch.consumer_layer >= self.layer_donors.len or self.layer_donors[batch.consumer_layer] != donor or head_count == 0)
            return false;
        const query_heads = layout.query_heads_by_layer[batch.consumer_layer];
        if (query_heads == 0 or query_heads % head_count != 0) return false;
        return batch.query_head / (query_heads / head_count) == head_index;
    }
};

pub const StreamingHeadLease = struct {
    context: ?*anyopaque,
    close_fn: *const fn (?*anyopaque, *StreamingHeadLease) void,
    donor_index: usize,
    head_index: usize,
    head: checkpoint.Head,
    closed: bool = false,

    pub fn deinit(self: *@This()) void {
        if (self.closed) @panic("streaming head lease closed twice");
        self.close_fn(self.context, self);
        self.closed = true;
    }
};

pub const StreamingHeadSource = struct {
    metadata: SourceMetadata,
    expected_structure_sha256: [32]u8,
    expected_state_sha256: [32]u8,
    context: ?*anyopaque,
    open_fn: *const fn (?*anyopaque, usize, usize, solver.Control) anyerror!StreamingHeadLease,
    verify_no_live_fn: *const fn (?*anyopaque) anyerror!void,
    external_live_bytes: usize,

    pub fn open(self: *@This(), donor: usize, head: usize, control: solver.Control) !StreamingHeadLease {
        var lease = try self.open_fn(self.context, donor, head, control);
        if (lease.donor_index != donor or lease.head_index != head or lease.closed) {
            if (!lease.closed) lease.deinit();
            return error.InvalidTrainingAugmentationBinding;
        }
        return lease;
    }
};

pub fn sourceMetadataStructureDigest(source: SourceMetadata) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-source-structure.v1\x00");
    hash.update(&source.identity.weights);
    hash.update(&source.identity.tokenizer);
    hash.update(&source.identity.template);
    hash.update(&source.identity.layout);
    hash.update(std.mem.asBytes(&source.logical_length));
    hash.update(std.mem.sliceAsBytes(source.tokens));
    hash.update(std.mem.sliceAsBytes(source.logits));
    hash.update(std.mem.sliceAsBytes(source.layer_donors));
    const count = if (source.full_donors.len != 0) source.full_donors.len else source.donors.len;
    hash.update(std.mem.asBytes(&count));
    for (0..count) |index| {
        const descriptor: SourceDonorMetadata = if (source.full_donors.len != 0) .{
            .layer = source.full_donors[index].layer,
            .window = source.full_donors[index].window,
            .head_count = source.full_donors[index].heads.len,
            .head_dim = source.full_donors[index].heads[0].dim,
            .rows = source.full_donors[index].heads[0].positions.len,
            .physical_bytes = 0,
        } else source.donors[index];
        hash.update(std.mem.asBytes(&descriptor.layer));
        hash.update(std.mem.asBytes(&descriptor.window));
        hash.update(std.mem.asBytes(&descriptor.head_count));
        hash.update(std.mem.asBytes(&descriptor.head_dim));
        hash.update(std.mem.asBytes(&descriptor.rows));
        hash.update(std.mem.asBytes(&descriptor.physical_bytes));
    }
    hash.update(std.mem.asBytes(&source.source_census_physical_bytes));
    hash.update(std.mem.asBytes(&source.source_global_storage_bytes));
    return hash.finalResult();
}

pub fn sourceMetadataFromState(source: checkpoint.State, limits: checkpoint.Limits, control: solver.Control) !SourceMetadata {
    try source.validateWithControl(limits, control);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-source-metadata.v1\x00");
    hash.update(&source.identity.weights);
    hash.update(&source.identity.tokenizer);
    hash.update(&source.identity.template);
    hash.update(&source.identity.layout);
    hash.update(std.mem.asBytes(&source.logical_length));
    hash.update(std.mem.sliceAsBytes(source.tokens));
    hash.update(std.mem.sliceAsBytes(source.logits));
    hash.update(std.mem.sliceAsBytes(source.layer_donors));
    for (source.donors, 0..) |donor, donor_index| {
        const descriptor: SourceDonorMetadata = .{ .layer = donor.layer, .window = donor.window, .head_count = donor.heads.len, .head_dim = donor.heads[0].dim, .rows = donor.heads[0].positions.len };
        hash.update(std.mem.asBytes(&donor_index));
        hash.update(std.mem.asBytes(&descriptor.layer));
        hash.update(std.mem.asBytes(&descriptor.window));
        hash.update(std.mem.asBytes(&descriptor.head_count));
        hash.update(std.mem.asBytes(&descriptor.head_dim));
        for (donor.heads, 0..) |head, head_index| {
            try control.check();
            hash.update(std.mem.asBytes(&head_index));
            hash.update(std.mem.sliceAsBytes(head.positions));
            hash.update(std.mem.sliceAsBytes(head.coverage_ends));
            hash.update(std.mem.sliceAsBytes(head.keys));
            hash.update(std.mem.sliceAsBytes(head.values));
            hash.update(std.mem.sliceAsBytes(head.bias));
            hash.update(std.mem.asBytes(&head.compacted_until));
            hash.update(std.mem.asBytes(&head.fp8));
            hash.update(std.mem.sliceAsBytes(head.compressed.spans));
            hash.update(head.compressed.keys);
            hash.update(head.compressed.values);
        }
    }
    var result: SourceMetadata = .{
        .identity = source.identity,
        .logical_length = source.logical_length,
        .tokens = source.tokens,
        .logits = source.logits,
        .layer_donors = source.layer_donors,
        .full_donors = source.donors,
        .structure_sha256 = undefined,
        .state_sha256 = hash.finalResult(),
        .source_census_physical_bytes = try source.residentBytes(),
    };
    result.structure_sha256 = sourceMetadataStructureDigest(result);
    return result;
}

pub const RejectedHeadFixtureView = struct {
    donor_layer: u32,
    kv_head: usize,
    retention_divisor: usize,
    logical_length: u64,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    scale: f64,
    softcap: f64,
    original: solver.Head,
    candidate: solver.Head,
    original_positions: []const u64,
    original_coverage_ends: []const u64,
    candidate_positions: []const u64,
    candidate_coverage_ends: []const u64,
    train_queries: []const f64,
    heldout_queries: []const f64,
    train_positions: []const u64,
    heldout_positions: []const u64,
    query_consumer_layers: []const u32,
    query_heads: []const u32,
    query_train_counts: []const u32,
    query_heldout_counts: []const u32,
    metrics: solver.Metrics,
};

pub const RejectedHeadCapture = struct {
    context: ?*anyopaque,
    capture_fn: *const fn (?*anyopaque, RejectedHeadFixtureView) void,
};

/// Diagnostic-only detail for an optional codec attempt. This never changes
/// publication: production callers leave `Options.codec_diagnostic` null.
pub const Rotated4RiskRetention = enum { quarter_v1, half_v2 };

pub const CodecDiagnostic = struct {
    pub const Kind = enum { none, fp8, rotated4 };
    pub const Outcome = enum {
        not_attempted,
        accepted,
        noncontiguous_range,
        invalid_quantized_row,
        nonfinite_probability_ratio,
        postcodec_heldout_gate,
    };
    kind: Kind = .none,
    outcome: Outcome = .not_attempted,
    rotated4_risk_retention: Rotated4RiskRetention = .quarter_v1,
    first: usize = 0,
    end: usize = 0,
    eligible_rows: usize = 0,
    retained_exact_rows: usize = 0,
    packed_rows: usize = 0,
    metrics: solver.Metrics = .{},
};

pub const Options = struct {
    pub const Admission = enum { fixed, cumulative_ladder_4_3_2 };
    pub const FitStrategy = enum { legacy, identity_chunk_coefficients_adaptive_v18 };
    pub const AdaptiveV18NumericalPolicy = enum { legacy_dense_f64, accepted_f32_gram_reuse_v1 };
    pub const AdaptiveV18GeometryRoute = enum { flat_gram_only, bounded_page_over_96, bounded_page_wide_cache_v1 };
    pub const AdaptiveV18BoundedScorePolicy = enum { scalar_reference, accelerate_fp64_tiled_v1 };
    pub const IndependentValidationPolicy = enum { scalar_reference, accelerate_fp64_summary_v1 };
    pub const TrainingPositionPolicy = enum {
        future_only,
        /// Builtin-test-only replay of captured post-RoPE queries against the
        /// complete current source. Positions remain their real prefill
        /// positions; this is not a causal past-attention target.
        recent_prefill_r64_full_current_source_replay_v1,
        /// Same authenticated R64 domain with the incremental geometry
        /// chooser. Distinct so the established replay remains byte-stable.
        recent_prefill_r64_incremental_geometry_diagnostic_v1,
        /// Same authenticated R64 domain with incremental geometry and the
        /// cached Accelerate FP64 TRAIN reduction. Builtin-test-only.
        recent_prefill_r64_incremental_reduction_diagnostic_v1,
        /// Builtin-test-only fixed 32 recent + 32 independently teacher-forced
        /// future-primary TRAIN rows. HHO remains a disjoint future domain.
        recent32_future_primary32_full_source_replay_v1,
        /// Same authenticated mixed TRAIN domain with incremental geometry and
        /// cached Accelerate FP64 TRAIN reductions.
        recent32_future_primary32_incremental_reduction_diagnostic_v1,
        /// Builtin-test-only full 64 recent + 64 independently teacher-forced
        /// future-primary TRAIN rows. This is a distinct sample-density
        /// diagnostic; it does not inherit the fixed32 policy identity.
        recent64_future_primary64_incremental_reduction_diagnostic_v1,
        recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1,
    };
    pub const V18ReferenceSummaryCachePolicy = enum { disabled, exact_original_train_hho_v1 };
    pub const PrivatePerformancePolicy = enum { legacy, owned_capture_summary_v1, owned_capture_summary_incremental_v1, owned_capture_summary_incremental_reduction_v1, owned_capture_summary_incremental_reduction_critical025_v1, diagnostic_capture_gram_batch256_v1 };
    pub const GeometryChooserPolicy = enum { legacy_full_scan, incremental_gram_v1 };
    pub const AdaptiveV18WorkspaceProfile = enum {
        standard,
        resident_native_hybrid_2g_v1,
        resident_e4_32512_bounded_2g_v1,
        resident_e2_formal_2g_p60_no_fallback_v1,
    };
    /// Fixed preserves the established per-chunk fitter byte-for-byte. The
    /// cumulative policy must be threaded through preflight and execution.
    admission: Admission = .fixed,
    /// Private test-only quality route. Legacy remains byte-identical/default.
    fit_strategy: FitStrategy = .legacy,
    /// Process-local staged authorization for the private v18 route.
    staged_operator_strategy: bool = false,
    adaptive_v18_context: ?*anyopaque = null,
    adaptive_v18_fn: ?*const fn (?*anyopaque, std.mem.Allocator, SourceMetadata, checkpoint.Head, usize, usize, []const ReferenceBatch, Layout, [4]usize, *const Options, usize) anyerror!checkpoint.Head = null,
    /// Owner-bound final-fit cache authority. Null preserves every established
    /// flat callback. The pointed value outlives the synchronous prepare.
    adaptive_v18_wide_cache_authority: ?*@import("attention_match_wide_cache_authority.zig").Authority = null,
    adaptive_v18_numerical_policy: AdaptiveV18NumericalPolicy = .legacy_dense_f64,
    adaptive_v18_workspace_profile: AdaptiveV18WorkspaceProfile = .standard,
    /// Default-off exact cache for the authenticated resident fallback's
    /// chronological cumulative ladder. Legacy and public routes stay disabled.
    cumulative_page_cache_policy: cumulative.PageCachePolicy = .disabled,
    adaptive_v18_geometry_route: AdaptiveV18GeometryRoute = .flat_gram_only,
    /// Explicit staged TRAIN-critical retention; legacy remains disabled.
    adaptive_v18_critical_core_policy: @import("attention_match_joint_chunk_mass_replay.zig").CriticalCorePolicy = .disabled,
    /// Private staged TRAIN-only bias postfit. No public request surface.
    adaptive_v18_postfit_policy: @import("attention_match_joint_chunk_mass_replay.zig").PostfitPolicy = .disabled,
    /// Test-only observation after the mandatory independent heldout gate.
    adaptive_v18_rejected_detail_diagnostic: bool = false,
    /// Emit phase spans without enabling rejected-HHO detail observation.
    adaptive_v18_phase_timing_diagnostic: bool = false,
    adaptive_v18_train_probability_census: bool = false,
    /// Staged TRAIN-only p91 comparison. Disabled preserves every established
    /// route; HHO is opened only after the selected policy is frozen.
    adaptive_v18_train_primary_policy_comparison: bool = false,
    /// Deterministic TRAIN-domain CV for the staged primary comparison.
    /// Requires comparison; ordinary/default behavior remains unchanged.
    adaptive_v18_train_primary_policy_cross_validation: bool = false,
    adaptive_v18_bounded_tile_queries: usize = 0,
    adaptive_v18_bounded_builder_transient_bytes: usize = 0,
    /// Explicit staged-private acceleration for the >96-candidate bounded
    /// route. False preserves the established scalar page delta path.
    adaptive_v18_bounded_delta_accelerated: bool = false,
    /// Typed private 32K authority for the phase-specific TSQR value solver.
    /// It never replaces attention, selection, or independent FP64 validation.
    /// False preserves all v3, 8K, legacy, and enrolled behavior.
    adaptive_v18_bounded_tsqr_value_backend: bool = false,
    /// Default-off TRAIN scoring policy for the final bounded callback. This
    /// does not alter independent heldout FP64 validation.
    adaptive_v18_bounded_score_policy: AdaptiveV18BoundedScorePolicy = .scalar_reference,
    /// Explicit staged-private physical budget. Zero/zero preserves every
    /// legacy and prior private profile byte-for-byte.
    adaptive_v18_primary_budget_percent: u8 = 0,
    adaptive_v18_fallback_budget_percent: u8 = 0,
    adaptive_v18_max_budget_fallbacks: u8 = 0,
    /// Optional diagnostics survive a rejected candidate; never authorize it.
    stats: ?*Stats = null,
    codec_diagnostic: ?*CodecDiagnostic = null,
    rejected_head_capture: ?RejectedHeadCapture = null,
    retention_divisor: usize = 4,
    /// On numerical rejection, retry with successively more retained rows down
    /// to this divisor. Every attempt uses the original source and the same
    /// whole-head error limits; admission must reserve the largest attempt.
    min_retention_divisor: ?usize = null,
    /// Reject by default. Explicit fallback preserves the original head only
    /// for a rejected numerical fit; cancellation, OOM and corrupt input fail.
    rejected_head_policy: enum { reject, preserve_original } = .reject,
    /// Actual global device bytes, counting candidate bias/position/coverage
    /// metadata. Admission never treats the requested retention as achieved.
    min_global_savings_ratio: ?f64 = null,
    source_has_metadata: bool = false,
    /// Round fitted K/V to the serving precision BEFORE held-out validation.
    /// In-memory checkpoint state represents these exact rounded values as f32.
    storage_precision: enum { f32, f16 } = .f32,
    /// Experimental disk encoding. Validate decoded FP8 against the original
    /// whole head; rejected FP8 attempts retain the already validated f16 fit.
    fp8_checkpoint: bool = false,
    /// Compress an eligible FP16 fit. The byte savings from newly quantized
    /// rows must not turn an otherwise rejected fit into a published cache.
    rotated4_checkpoint: bool = false,
    /// Fixed TRAIN-only risk retention. Quarter is the established production
    /// default; half is an explicit diagnostic policy and never adapts from
    /// heldout results.
    rotated4_risk_retention: Rotated4RiskRetention = .quarter_v1,
    recent_rows: usize = 2048,
    /// Preserve these initial logical positions exactly, including across cycles.
    prefix_rows: usize = 0,
    /// Whole-head validation includes the exact prefix/recent rows and applies
    /// the same thresholds after rounding. Chunk mode retains the strict oracle.
    validation: enum { chunk, whole_head } = .chunk,
    /// Re-authorize final whole-head and encoded publication using the common
    /// scalar FP64 attention path. Off by default so existing CUDA/common
    /// execution and performance remain unchanged; the Metal adapter opts in.
    independent_fp64_publication: bool = false,
    /// Private DEV-only independent validation accelerator. This never replaces
    /// `fit.backend`; scalar reference remains the default.
    independent_validation_policy: IndependentValidationPolicy = .scalar_reference,
    training_position_policy: TrainingPositionPolicy = .future_only,
    /// Default-off exact FP64 ORIGINAL summary reuse. Candidate summaries are
    /// always recomputed and the independent HHO acceptance gate is unchanged.
    v18_reference_summary_cache_policy: V18ReferenceSummaryCachePolicy = .disabled,
    private_performance_policy: PrivatePerformancePolicy = .legacy,
    geometry_chooser_policy: GeometryChooserPolicy = .legacy_full_scan,
    component_ab_diagnostic: bool = false,
    /// Bound concurrent whole-head conversion/validation, independently of fit workers.
    max_parallel_validations: usize = 2,
    chunk_rows: usize = 512,
    max_queries_per_consumer: usize = 256,
    /// Counts source + candidate host state, serialization, query conversion,
    /// replacement buffers and solver scratch. Device memory needs an additional
    /// reservation by the serving adapter and is not hidden in this number.
    max_transient_host_bytes: usize = 2 * 1024 * 1024 * 1024,
    checkpoint_limits: checkpoint.Limits = .{},
    fit: solver.Config = .{ .target_rows = 1 },
};

pub const Stats = struct {
    donor_layer: u32 = 0,
    head: usize = 0,
    chunk_start: usize = 0,
    attempted_chunks: usize = 0,
    completed_chunks: usize = 0,
    max_nnls_sweeps: usize = 0,
    last_fit: solver.Metrics = .{},
    validating_storage: bool = false,
    rounded_fit: solver.Metrics = .{},
    validating_head: bool = false,
    validated_heads: usize = 0,
    preserved_heads: usize = 0,
    v18_eligible_heads: usize = 0,
    v18_attempted_heads: usize = 0,
    v18_accepted_heads: usize = 0,
    v18_preserved_heads: usize = 0,
    v18_fallback_heads: usize = 0,
    v18_train_primary_probe_heads: usize = 0,
    v18_train_primary_probe_candidates: usize = 0,
    v18_train_primary_probe_recomputed_heads: usize = 0,
    v18_cache_ready_heads: usize = 0,
    v18_planned_exact_heads: usize = 0,
    v18_query_batch_rows: usize = 0,
    v18_planned_head_workspace_bytes: usize = 0,
    v18_cache_phase_peak_bound_bytes: usize = 0,
    v18_exact_f32_probability_heads: usize = 0,
    v18_gram_heads: usize = 0,
    v18_scored_reuse_heads: usize = 0,
    v18_gram_backend_calls: usize = 0,
    v18_bounded_heads: usize = 0,
    v18_bounded_provider_calls: u64 = 0,
    v18_bounded_provider_rows: u64 = 0,
    v18_bounded_workspace_bound_bytes: usize = 0,
    v18_bounded_scoring_cache_prepares: usize = 0,
    v18_bounded_scoring_selection_hits: usize = 0,
    v18_bounded_scoring_fit_hits: usize = 0,
    v18_bounded_score_accelerate_calls: usize = 0,
    v18_bounded_score_accelerate_tiles: usize = 0,
    v18_bounded_score_accelerate_dgemm_calls: usize = 0,
    v18_bounded_score_cancellation_checks: usize = 0,
    v18_bounded_range_cache_prepares: usize = 0,
    v18_bounded_geometry_cache_requests: usize = 0,
    v18_bounded_geometry_cache_hits: usize = 0,
    v18_bounded_geometry_cache_loads: usize = 0,
    v18_bounded_delta_heads: usize = 0,
    v18_bounded_delta_tiles: usize = 0,
    v18_bounded_delta_backend_calls: usize = 0,
    v18_bounded_delta_packed_bytes_peak: usize = 0,
    v18_bounded_delta_staged_bytes_peak: usize = 0,
    v18_bounded_delta_fallbacks: usize = 0,
    v18_bounded_tsqr_heads: usize = 0,
    v18_bounded_tsqr_calls: usize = 0,
    v18_bounded_tsqr_workspace_peak_bytes: usize = 0,
    v18_bounded_tsqr_cancellation_checks: usize = 0,
    v18_cache_prepare_ns: u64 = 0,
    v18_selection_freeze_ns: u64 = 0,
    v18_joint_value_fit_ns: u64 = 0,
    v18_gram_construction_ns: u64 = 0,
    v18_chooser_ns: u64 = 0,
    v18_final_heldout_ns: u64 = 0,
    v18_primary_budget_percent: u8 = 0,
    v18_budget_fallback_heads: usize = 0,
    retention_retries: usize = 0,
    rotated4_heads: usize = 0,
    rotated4_fallback_heads: usize = 0,
    rotated4_eligible_rows: usize = 0,
    rotated4_compressed_rows: usize = 0,
    rotated4_exact_rows: usize = 0,
    fp8_heads: usize = 0,
    fp8_fallback_heads: usize = 0,
    fp8_eligible_rows: usize = 0,
    fp8_encoded_rows: usize = 0,
    fp8_exact_rows: usize = 0,
    fp64_validation_summary_calls: usize = 0,
    fp64_validation_dgemm_calls: usize = 0,
    fp64_validation_max_batch_rows: usize = 0,
    fp64_validation_workspace_peak_bytes: usize = 0,
    fp64_validation_policy: Options.IndependentValidationPolicy = .scalar_reference,
    fp64_reference_train_fills: usize = 0,
    fp64_reference_train_hits: usize = 0,
    fp64_reference_hho_fills: usize = 0,
    fp64_reference_hho_hits: usize = 0,
    fp64_reference_candidate_delegates: usize = 0,
    fp64_reference_candidate_hits: usize = 0,
    fp64_reference_retained_bytes_peak: usize = 0,
    fp64_reference_cap_fallbacks_before_allocation: usize = 0,
    fp64_reference_detailed_observer_bypasses: usize = 0,
    geometry_chooser_policy: Options.GeometryChooserPolicy = .legacy_full_scan,
    incremental_gram_chooser_calls: usize = 0,
    training_reduction_local_original_calls: usize = 0,
    training_reduction_rounded_harm_calls: usize = 0,
    training_reduction_tiles: usize = 0,
    training_reduction_backend_calls: usize = 0,
    training_reduction_workspace_peak_bytes: usize = 0,
    cumulative_page_cache_pages: usize = 0,
    cumulative_page_cache_hits: usize = 0,
    cumulative_page_cache_misses: usize = 0,
    cumulative_page_cache_fallbacks_before_gpu: usize = 0,
    cumulative_page_cache_retained_bytes_peak: usize = 0,
    convergence_fallbacks: usize = 0,
    source_global_bytes: usize = 0,
    candidate_global_bytes: usize = 0,
    fitted_global_bytes: usize = 0,
    fitted_global_savings_ratio: f64 = 1,
    actual_global_savings_ratio: f64 = 1,
    max_rejected_mass_error: f64 = 0,
    max_rejected_output_rmse: f64 = 0,
    max_head_mass_error: f64 = 0,
    max_head_output_rmse: f64 = 0,
};

fn privatePerformancePolicyValid(options: Options) bool {
    if (options.private_performance_policy == .legacy) return true;
    const expected_chooser: Options.GeometryChooserPolicy = switch (options.private_performance_policy) {
        .owned_capture_summary_incremental_v1, .owned_capture_summary_incremental_reduction_v1, .owned_capture_summary_incremental_reduction_critical025_v1, .diagnostic_capture_gram_batch256_v1 => .incremental_gram_v1,
        else => .legacy_full_scan,
    };
    const geometry_authorized = options.adaptive_v18_geometry_route == .flat_gram_only or
        ((options.adaptive_v18_workspace_profile == .resident_e4_32512_bounded_2g_v1 or
            options.adaptive_v18_workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1) and options.adaptive_v18_geometry_route == .bounded_page_over_96);
    return options.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and options.staged_operator_strategy and
        options.adaptive_v18_numerical_policy == .accepted_f32_gram_reuse_v1 and geometry_authorized and
        options.independent_validation_policy == .accelerate_fp64_summary_v1 and options.geometry_chooser_policy == expected_chooser and
        options.validation == .whole_head and options.independent_fp64_publication;
}

fn streamingPrivatePerformancePolicyValid(options: Options) bool {
    return options.private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
        options.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and options.staged_operator_strategy and
        options.adaptive_v18_numerical_policy == .accepted_f32_gram_reuse_v1 and
        options.adaptive_v18_geometry_route == .bounded_page_over_96 and options.adaptive_v18_bounded_tsqr_value_backend and
        options.independent_validation_policy == .accelerate_fp64_summary_v1 and options.geometry_chooser_policy == .incremental_gram_v1 and
        options.validation == .whole_head and options.independent_fp64_publication;
}

fn cumulativePageCachePolicyValid(options: Options) bool {
    if (options.cumulative_page_cache_policy == .disabled) return true;
    // Quarantined: adaptive V18 returns through its callback before the legacy
    // cumulative fitter containing this cache. Unsupported V18 fallback is not
    // an authorization to claim the experimental cache ran.
    return false;
}

fn trainingPositionPolicyValid(options: Options) bool {
    if (options.training_position_policy == .future_only) return true;
    const expected_private: Options.PrivatePerformancePolicy = switch (options.training_position_policy) {
        .future_only => unreachable,
        .recent_prefill_r64_full_current_source_replay_v1 => .owned_capture_summary_v1,
        .recent_prefill_r64_incremental_geometry_diagnostic_v1 => .owned_capture_summary_incremental_v1,
        .recent_prefill_r64_incremental_reduction_diagnostic_v1 => .owned_capture_summary_incremental_reduction_v1,
        .recent32_future_primary32_full_source_replay_v1 => .owned_capture_summary_v1,
        .recent32_future_primary32_incremental_reduction_diagnostic_v1 => .owned_capture_summary_incremental_reduction_v1,
        .recent64_future_primary64_incremental_reduction_diagnostic_v1, .recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1 => .owned_capture_summary_incremental_reduction_v1,
    };
    const expected_max_queries: usize = switch (options.training_position_policy) {
        .recent64_future_primary64_incremental_reduction_diagnostic_v1, .recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1 => 128,
        else => 64,
    };
    return builtin.is_test and options.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and
        options.staged_operator_strategy and options.admission == .cumulative_ladder_4_3_2 and
        options.prefix_rows == 128 and options.recent_rows == 2048 and options.max_queries_per_consumer == expected_max_queries and
        options.validation == .whole_head and options.storage_precision == .f16 and
        options.independent_fp64_publication and
        options.independent_validation_policy == .accelerate_fp64_summary_v1 and
        options.private_performance_policy == expected_private;
}

test "recent64 replay position policy requires the complete diagnostic route" {
    try std.testing.expect(trainingPositionPolicyValid(.{}));
    var options: Options = .{
        .training_position_policy = .recent_prefill_r64_full_current_source_replay_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .admission = .cumulative_ladder_4_3_2,
        .prefix_rows = 128,
        .recent_rows = 2048,
        .max_queries_per_consumer = 64,
        .validation = .whole_head,
        .storage_precision = .f16,
        .independent_fp64_publication = true,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .private_performance_policy = .owned_capture_summary_v1,
    };
    try std.testing.expect(trainingPositionPolicyValid(options));
    options.training_position_policy = .recent_prefill_r64_incremental_geometry_diagnostic_v1;
    try std.testing.expect(!trainingPositionPolicyValid(options));
    options.private_performance_policy = .owned_capture_summary_incremental_v1;
    options.geometry_chooser_policy = .incremental_gram_v1;
    try std.testing.expect(trainingPositionPolicyValid(options));
    options.training_position_policy = .recent_prefill_r64_incremental_reduction_diagnostic_v1;
    try std.testing.expect(!trainingPositionPolicyValid(options));
    options.private_performance_policy = .owned_capture_summary_incremental_reduction_v1;
    try std.testing.expect(trainingPositionPolicyValid(options));
    options.independent_validation_policy = .scalar_reference;
    try std.testing.expect(!trainingPositionPolicyValid(options));
    options.independent_validation_policy = .accelerate_fp64_summary_v1;
    options.max_queries_per_consumer = 63;
    try std.testing.expect(!trainingPositionPolicyValid(options));
}

test "context private owned capture summary policy is flat and heldout-only" {
    try std.testing.expect(privatePerformancePolicyValid(.{}));
    const sealed: Options = .{
        .private_performance_policy = .owned_capture_summary_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .validation = .whole_head,
        .independent_fp64_publication = true,
    };
    try std.testing.expect(privatePerformancePolicyValid(sealed));
    var bounded = sealed;
    bounded.adaptive_v18_geometry_route = .bounded_page_over_96;
    try std.testing.expect(!privatePerformancePolicyValid(bounded));
    var scalar = sealed;
    scalar.independent_validation_policy = .scalar_reference;
    try std.testing.expect(!privatePerformancePolicyValid(scalar));
    var incremental = sealed;
    incremental.geometry_chooser_policy = .incremental_gram_v1;
    try std.testing.expect(!privatePerformancePolicyValid(incremental));
    incremental.private_performance_policy = .owned_capture_summary_incremental_v1;
    try std.testing.expect(privatePerformancePolicyValid(incremental));
    incremental.private_performance_policy = .owned_capture_summary_incremental_reduction_v1;
    try std.testing.expect(privatePerformancePolicyValid(incremental));
}

test "attention match rejected-head capture is default disabled" {
    const options: Options = .{};
    try std.testing.expect(options.rejected_head_capture == null);
    try std.testing.expectEqual(Options.Admission.fixed, options.admission);
    try std.testing.expectEqual(Options.V18ReferenceSummaryCachePolicy.disabled, options.v18_reference_summary_cache_policy);
}

test "legacy cumulative page cache is quarantined from adaptive V18" {
    var resident: Options = .{
        .admission = .cumulative_ladder_4_3_2,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .chunk_rows = 256,
        .max_queries_per_consumer = 768,
        .cumulative_page_cache_policy = .exact_1024_v1,
    };
    try std.testing.expect(!cumulativePageCachePolicyValid(resident));
    resident.adaptive_v18_workspace_profile = .standard;
    try std.testing.expect(!cumulativePageCachePolicyValid(resident));
    resident.adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1;
    resident.max_queries_per_consumer = 256;
    try std.testing.expect(!cumulativePageCachePolicyValid(resident));
    var legacy: Options = .{};
    try std.testing.expect(cumulativePageCachePolicyValid(legacy));
    legacy.cumulative_page_cache_policy = .exact_1024_v1;
    try std.testing.expect(!cumulativePageCachePolicyValid(legacy));
}

test "cumulative runtime planning reserves preserve-all rows and declared solver workspace" {
    const fixed = try headWorkspaceBytes(16, 8, 8, 8, 8, 1, .{ .fit = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 } });
    const cumulative_plan = try headWorkspaceBytes(16, 8, 8, 8, 8, 1, .{
        .admission = .cumulative_ladder_4_3_2,
        .validation = .whole_head,
        .storage_precision = .f16,
        .independent_fp64_publication = true,
        .min_global_savings_ratio = 1.5,
        .min_retention_divisor = 2,
        .fit = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 },
    });
    try std.testing.expect(fixed.solver_bytes <= 1 << 20);
    try std.testing.expect(cumulative_plan.solver_bytes <= 1 << 20);
    try std.testing.expect(cumulative_plan.bytes >= 1 << 20);
    try std.testing.expect(cumulative_plan.bytes > fixed.bytes);
}

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn budget(required: usize, maximum: usize) !void {
    if (required > maximum) return error.CompactionWorkspaceLimit;
}

pub const HeadWorkspace = struct {
    /// Additional bytes beyond the source and worst-case candidate states.
    bytes: usize,
    solver_bytes: usize,
};

/// The accelerator is allowed to construct fitting matrices, but publication
/// is authorized against the common FP64 attention semantics. Keeping this a
/// distinct config also makes the extra scalar validation storage visible to
/// preflight admission instead of silently allocating it after admission.
fn finalAuthorizationConfig(fit: solver.Config, independent: bool, trusted_reference_backend: ?solver.Backend) solver.Config {
    var authorization = fit;
    if (independent) authorization.backend = trusted_reference_backend;
    return authorization;
}

/// Codec TRAIN risk selection consumes full probability matrices. Summary-only
/// validation backends are heldout authorization oracles and must never enter
/// this path.
fn codecTrainingConfig(authorization: solver.Config) solver.Config {
    var training = authorization;
    training.backend = null;
    training.metrics = null;
    return training;
}
fn trustedValidationBackend(policy: Options.IndependentValidationPolicy, context: *validation_accelerate.Context) ?solver.Backend {
    return switch (policy) {
        .scalar_reference => null,
        .accelerate_fp64_summary_v1 => validation_accelerate.backend(context),
    };
}

/// Shared by preflight admission and compactHead. Includes the replacement
/// arrays even though the resident bound also reserves a full-size candidate.
pub fn headWorkspaceBytes(
    original_rows: usize,
    target_rows: usize,
    block_rows: usize,
    train_elements: usize,
    heldout_elements: usize,
    d: usize,
    options: Options,
) !HeadWorkspace {
    if (d == 0 or block_rows == 0 or target_rows == 0 or target_rows > original_rows or
        block_rows > original_rows or options.retention_divisor < 2 or options.retention_divisor > 4 or
        train_elements == 0 or heldout_elements == 0 or train_elements % d != 0 or heldout_elements % d != 0)
        return error.InvalidFitConfig;
    const query_bytes = try mul(try add(train_elements, heldout_elements), 8);
    // Cumulative admission can preserve every rejected source chunk, so its
    // largest live candidate is the original row count rather than the fixed
    // plan target. Charge that shape before any worker starts.
    const admitted_rows = if (options.admission == .cumulative_ladder_4_3_2) original_rows else target_rows;
    const replacement_bytes = try mul(admitted_rows, try add(20, try mul(d, 8)));
    // Cloned prior state, encoded candidate and filtered candidate coexist.
    const packed_bytes = if (options.rotated4_checkpoint)
        try mul(admitted_rows, try add(60, try mul(d, 6)))
    else
        try mul(admitted_rows, try add(40, try mul(d, 4)));
    const samples = @min(train_elements / d, 64);
    const selection_bytes = if (options.rotated4_checkpoint)
        try add(try mul(admitted_rows, try add(17, try mul(samples, 16))), try mul(samples, try add(16, try mul(d, 8))))
    else
        0;
    const selection_backend_bytes = if (options.rotated4_checkpoint and options.fit.backend != null)
        try options.fit.backend.?.workspace_fn(options.fit.backend.?.context, admitted_rows, 1, samples, d)
    else
        0;
    const conversion_bytes = if (options.admission == .cumulative_ladder_4_3_2) blk: {
        // Full-original K/V/bias stay live while the inner cumulative cache,
        // trial candidate, and final validation operate. They are deliberately
        // outside the inner max_workspace_bytes cap.
        const original_f64 = try mul(try mul(original_rows, try add(try mul(d, 2), 1)), @sizeOf(f64));
        const range_count = std.math.divCeil(usize, original_rows, options.chunk_rows) catch return error.InvalidFitConfig;
        break :blk try add(original_f64, try mul(range_count, @sizeOf(cumulative_cache.Range)));
    } else try mul(try add(try mul(block_rows, try add(try mul(d, 4), 6)), try mul(d, 2)), 8);
    const scratch_max = try solver.fitWorkspaceBytes(block_rows, std.math.divCeil(usize, block_rows, options.retention_divisor) catch unreachable, train_elements / d, d, heldout_elements / d, options.fit);
    var validation_context: validation_accelerate.Context = .{};
    const trusted_backend = trustedValidationBackend(options.independent_validation_policy, &validation_context);
    const authorization = finalAuthorizationConfig(options.fit, options.independent_fp64_publication, trusted_backend);
    const head_validation_bytes = if (options.validation == .whole_head)
        try add(try mul(try mul(try add(original_rows, admitted_rows), try add(try mul(d, 2), 1)), 8), try solver.validationBytes(original_rows, admitted_rows, d, heldout_elements / d, authorization))
    else
        0;
    return .{
        .bytes = try add(try add(try add(query_bytes, packed_bytes), try add(selection_bytes, selection_backend_bytes)), try add(replacement_bytes, @max(head_validation_bytes, try add(conversion_bytes, if (options.admission == .cumulative_ladder_4_3_2) options.fit.max_workspace_bytes else scratch_max)))),
        .solver_bytes = scratch_max,
    };
}

pub fn parallelResidentBytes(source_bytes: usize, head_count: usize) !usize {
    return add(try mul(source_bytes, 2), try mul(head_count, @sizeOf(HeadTask)));
}

/// The scheduler divides scratch equally, so every worker must accommodate the
/// largest eligible head. A validation concurrency gate does not shrink this
/// conservative bound.
pub fn parallelHostBytes(source_bytes: usize, head_count: usize, head_workspace_bytes: usize, workers: usize) !usize {
    if (workers == 0 or workers > 4) return error.InvalidFitConfig;
    return add(try parallelResidentBytes(source_bytes, head_count), try mul(workers, head_workspace_bytes));
}

fn validatePositions(positions: []const u64, logical_length: u64) !void {
    for (positions, 0..) |pos, i| {
        if (pos < logical_length or (i > 0 and pos <= positions[i - 1])) return error.InvalidCompactionQueries;
    }
}

pub fn validateReferencePositions(batch: ReferenceBatch, logical_length: u64) !void {
    if (batch.independent_train_second_split != null and batch.independent_train_split == null)
        return error.InvalidCompactionQueries;
    if (batch.independent_train_split) |split| {
        if (split == 0 or split >= batch.train_positions.len) return error.InvalidCompactionQueries;
        try validatePositions(batch.train_positions[0..split], logical_length);
        if (batch.independent_train_second_split) |second| {
            if (second <= split or second >= batch.train_positions.len)
                return error.InvalidCompactionQueries;
            try validatePositions(batch.train_positions[split..second], logical_length);
            try validatePositions(batch.train_positions[second..], logical_length);
        } else {
            try validatePositions(batch.train_positions[split..], logical_length);
        }
    } else try validatePositions(batch.train_positions, logical_length);
    try validatePositions(batch.heldout_positions, logical_length);
    var j: usize = 0;
    const primary = if (batch.independent_train_split) |split| batch.train_positions[0..split] else batch.train_positions;
    for (primary) |pos| {
        while (j < batch.heldout_positions.len and batch.heldout_positions[j] < pos) : (j += 1) {}
        if (j < batch.heldout_positions.len and batch.heldout_positions[j] == pos) return error.InvalidCompactionQueries;
    }
}

fn validateReferencePositionsForOptions(batch: ReferenceBatch, logical_length: u64, options: Options) !void {
    if (options.training_position_policy == .future_only)
        return validateReferencePositions(batch, logical_length);
    if (options.training_position_policy == .recent64_future_primary64_incremental_reduction_diagnostic_v1 or
        options.training_position_policy == .recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1)
    {
        if (!builtin.is_test or logical_length != 8128 or options.prefix_rows != 128 or
            options.recent_rows != 2048 or options.max_queries_per_consumer != 128 or
            batch.train_positions.len != 128 or batch.independent_train_split != 64 or
            batch.independent_train_second_split != null)
            return error.InvalidCompactionQueries;
        for (batch.train_positions[0..64], 0..) |position, row| {
            if (position != 8064 + row) return error.InvalidCompactionQueries;
        }
        for (batch.train_positions[64..], 0..) |position, row| {
            const expected: u64 = if (options.training_position_policy == .recent64_future_phase_balanced64_incremental_reduction_diagnostic_v1)
                8129 + 8 * row + row % 7
            else
                8132 + 8 * row;
            if (position != expected) return error.InvalidCompactionQueries;
        }
        var future = batch;
        future.train_positions = batch.train_positions[64..];
        future.independent_train_split = null;
        future.independent_train_second_split = null;
        return validateReferencePositions(future, logical_length);
    }
    if (options.training_position_policy == .recent32_future_primary32_full_source_replay_v1 or
        options.training_position_policy == .recent32_future_primary32_incremental_reduction_diagnostic_v1)
    {
        if (!builtin.is_test or logical_length != 8128 or options.prefix_rows != 128 or
            options.recent_rows != 2048 or options.max_queries_per_consumer != 64 or
            batch.train_positions.len != 64 or batch.independent_train_split != 32 or
            batch.independent_train_second_split != null)
            return error.InvalidCompactionQueries;
        for (batch.train_positions[0..32], 0..) |position, row| {
            if (position != 8096 + row) return error.InvalidCompactionQueries;
        }
        for (batch.train_positions[32..], 0..) |position, row| {
            if (position != 8132 + 16 * row) return error.InvalidCompactionQueries;
        }
        // Only the recent prefix has the full-current-source interpretation.
        // Validate the future TRAIN suffix as the primary domain so every one
        // of its rows is checked for collisions with independent future HHO.
        var future = batch;
        future.train_positions = batch.train_positions[32..];
        future.independent_train_split = null;
        future.independent_train_second_split = null;
        return validateReferencePositions(future, logical_length);
    }
    if (!builtin.is_test or logical_length != 8128 or options.prefix_rows != 128 or
        options.recent_rows != 2048 or logical_length - @as(u64, @intCast(options.recent_rows)) != 6080 or
        options.max_queries_per_consumer != 64 or batch.independent_train_split != null or
        batch.independent_train_second_split != null or batch.train_positions.len != 64)
        return error.InvalidCompactionQueries;
    for (batch.train_positions, 0..) |position, row| {
        if (position != 8064 + row or position < 6080)
            return error.InvalidCompactionQueries;
    }
    // HHO remains genuinely future-only and disjoint. Only TRAIN receives the
    // explicit full-current-source replay interpretation.
    try validatePositions(batch.heldout_positions, logical_length);
}

pub fn validateQueries(source: checkpoint.State, batches: []const ReferenceBatch, layout: Layout, options: Options) !void {
    if (layout.query_heads_by_layer.len != source.layer_donors.len or
        layout.scales_by_donor.len != source.donors.len or layout.softcaps_by_donor.len != source.donors.len)
        return error.InvalidCompactionQueries;
    var expected: usize = 0;
    var train_count: ?usize = null;
    var heldout_count: ?usize = null;
    for (source.donors, layout.scales_by_donor, layout.softcaps_by_donor) |_, scale, cap| {
        if (!std.math.isFinite(scale) or scale <= 0 or !std.math.isFinite(cap) or cap < 0) return error.InvalidCompactionQueries;
    }
    for (source.layer_donors, layout.query_heads_by_layer, 0..) |donor_index, qh, layer| {
        const donor = source.donors[donor_index];
        if (qh == 0 or qh % donor.heads.len != 0) return error.InvalidCompactionQueries;
        if (donor.window != 0) continue;
        expected = try add(expected, qh);
        for (0..qh) |head| {
            var found = false;
            for (batches) |batch| {
                if (batch.consumer_layer != layer or batch.query_head != head) continue;
                if (found) return error.InvalidCompactionQueries;
                found = true;
                const nt = batch.train_positions.len;
                const nh = batch.heldout_positions.len;
                if (nt == 0 or nh == 0 or nt > options.max_queries_per_consumer or nh > options.max_queries_per_consumer or
                    batch.train.len != try mul(nt, donor.heads[0].dim) or batch.heldout.len != try mul(nh, donor.heads[0].dim))
                    return error.InvalidCompactionQueries;
                if (train_count) |count| {
                    if (count != nt) return error.UnbalancedCompactionQueries;
                } else train_count = nt;
                if (heldout_count) |count| {
                    if (count != nh) return error.UnbalancedCompactionQueries;
                } else heldout_count = nh;
                try validateReferencePositionsForOptions(batch, source.logical_length, options);
                for (batch.train) |x| if (!std.math.isFinite(x)) return error.InvalidCompactionQueries;
                for (batch.heldout) |x| if (!std.math.isFinite(x)) return error.InvalidCompactionQueries;
            }
            if (!found) return error.MissingCompactionConsumerQueries;
        }
    }
    if (batches.len != expected) return error.InvalidCompactionQueries;
}

fn validateStreamingMetadata(source: SourceMetadata, limits: checkpoint.Limits, control: solver.Control) !void {
    try control.check();
    if (source.logical_length == 0 or source.logical_length > limits.max_tokens or source.tokens.len != source.logical_length or
        source.logits.len == 0 or source.logits.len > limits.max_vocab or source.layer_donors.len == 0 or
        source.layer_donors.len > limits.max_layers or source.donors.len == 0 or source.full_donors.len != 0 or
        std.mem.allEqual(u8, &source.state_sha256, 0) or source.source_census_physical_bytes == 0 or
        !std.mem.eql(u8, &source.structure_sha256, &sourceMetadataStructureDigest(source)))
        return error.InvalidKvCheckpoint;
    for (source.tokens, 0..) |token, index| {
        if (index % 4096 == 0) try control.check();
        if (token >= source.logits.len) return error.InvalidKvCheckpoint;
    }
    for (source.logits) |value| if (!std.math.isFinite(value)) return error.InvalidKvCheckpoint;
    for (source.layer_donors, 0..) |donor, layer| {
        if (donor >= source.donors.len or source.donors[donor].layer > layer) return error.InvalidKvCheckpoint;
    }
    var inventory_physical: usize = 0;
    for (source.donors, 0..) |donor, index| {
        if (donor.layer >= source.layer_donors.len or source.layer_donors[donor.layer] != index or
            (index > 0 and donor.layer <= source.donors[index - 1].layer) or donor.window > limits.max_tokens or
            donor.head_count == 0 or donor.head_count > limits.max_kv_heads or donor.head_dim == 0 or donor.head_dim > limits.max_head_dim or
            donor.rows == 0 or donor.rows > source.logical_length or donor.physical_bytes == 0)
            return error.InvalidKvCheckpoint;
        inventory_physical = try add(inventory_physical, donor.physical_bytes);
    }
    if (inventory_physical != source.source_census_physical_bytes) return error.InvalidKvCheckpoint;
}

fn validateStreamingQueries(source: SourceMetadata, batches: []const ReferenceBatch, layout: Layout, options: Options) !void {
    if (layout.query_heads_by_layer.len != source.layer_donors.len or layout.scales_by_donor.len != source.donors.len or
        layout.softcaps_by_donor.len != source.donors.len)
        return error.InvalidCompactionQueries;
    var expected: usize = 0;
    var train_count: ?usize = null;
    var heldout_count: ?usize = null;
    for (source.donors, layout.scales_by_donor, layout.softcaps_by_donor) |_, scale, cap|
        if (!std.math.isFinite(scale) or scale <= 0 or !std.math.isFinite(cap) or cap < 0) return error.InvalidCompactionQueries;
    for (source.layer_donors, layout.query_heads_by_layer, 0..) |donor_index, query_heads, layer| {
        const donor = source.donors[donor_index];
        if (query_heads == 0 or query_heads % donor.head_count != 0) return error.InvalidCompactionQueries;
        if (donor.window != 0) continue;
        expected = try add(expected, query_heads);
        for (0..query_heads) |query_head| {
            var found = false;
            for (batches) |batch| {
                if (batch.consumer_layer != layer or batch.query_head != query_head) continue;
                if (found) return error.InvalidCompactionQueries;
                found = true;
                const train_rows = batch.train_positions.len;
                const heldout_rows = batch.heldout_positions.len;
                if (train_rows == 0 or heldout_rows == 0 or train_rows > options.max_queries_per_consumer or
                    heldout_rows > options.max_queries_per_consumer or batch.train.len != try mul(train_rows, donor.head_dim) or
                    batch.heldout.len != try mul(heldout_rows, donor.head_dim)) return error.InvalidCompactionQueries;
                if (train_count) |count| {
                    if (count != train_rows) return error.UnbalancedCompactionQueries;
                } else train_count = train_rows;
                if (heldout_count) |count| {
                    if (count != heldout_rows) return error.UnbalancedCompactionQueries;
                } else heldout_count = heldout_rows;
                try validateReferencePositionsForOptions(batch, source.logical_length, options);
                for (batch.train) |value| if (!std.math.isFinite(value)) return error.InvalidCompactionQueries;
                for (batch.heldout) |value| if (!std.math.isFinite(value)) return error.InvalidCompactionQueries;
            }
            if (!found) return error.MissingCompactionConsumerQueries;
        }
    }
    if (batches.len != expected) return error.InvalidCompactionQueries;
}

fn belongs(batch: ReferenceBatch, source: checkpoint.State, layout: Layout, donor_index: usize, head_index: usize) bool {
    if (source.layer_donors[batch.consumer_layer] != donor_index) return false;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / source.donors[donor_index].heads.len;
    return batch.query_head / group == head_index;
}

fn converted(allocator: std.mem.Allocator, values: []const f32) ![]f64 {
    const result = try allocator.alloc(f64, values.len);
    for (result, values) |*out, value| out.* = value;
    return result;
}

pub const HeadPlan = struct {
    old_rows: usize,
    prefix_rows: usize,
    target_rows: usize,
    block_rows: usize,
};

pub const FitRange = struct { start: usize, rows: usize };
pub const v18_head_workspace_bytes: usize = 512 * 1024 * 1024;

/// Conservative complete physical cap for the private v18 worker. The v18
/// replay planner performs the shape-specific proof again before cache capture.
pub fn v18HeadWorkspaceBytes() usize {
    return v18_head_workspace_bytes;
}

pub fn v18HeadWorkspaceBytesForPolicy(policy: Options.AdaptiveV18NumericalPolicy) usize {
    return switch (policy) {
        .legacy_dense_f64 => v18_head_workspace_bytes,
        .accepted_f32_gram_reuse_v1 => 1024 * 1024 * 1024,
    };
}

/// The larger resident-native cap is a separate staged policy. It cannot be
/// obtained by raising a generic fitter limit or selecting the numerical route.
/// Available workspace for a head never exceeds the separately admitted
/// resident policy, even when serial assembly retains less than the reserved
/// worst-case source/candidate pair.
pub fn v18AvailableHeadWorkspaceBytes(options: Options, resident: usize) !usize {
    const available = std.math.sub(usize, options.max_transient_host_bytes, resident) catch return error.V18TrainingCacheUnsupported;
    // Aggregate host headroom is availability, never authority to enlarge one
    // head. Every profile has a separately authenticated per-head policy cap.
    return @min(available, try v18HeadWorkspaceBytesForOptions(options));
}

pub fn v18HeadWorkspaceBytesForOptions(options: Options) !usize {
    if (options.adaptive_v18_workspace_profile == .standard)
        return v18HeadWorkspaceBytesForPolicy(options.adaptive_v18_numerical_policy);
    if (options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.staged_operator_strategy or
        options.adaptive_v18_numerical_policy != .accepted_f32_gram_reuse_v1 or
        options.adaptive_v18_critical_core_policy != .train_probability_gte_0_025_no_top1_v1 or
        options.adaptive_v18_postfit_policy != .train_global_residual_range_mass_l2_exact_f32_cache_v1 or
        (options.private_performance_policy != .owned_capture_summary_incremental_reduction_critical025_v1 and
            options.private_performance_policy != .diagnostic_capture_gram_batch256_v1) or
        options.geometry_chooser_policy != .incremental_gram_v1 or
        options.independent_validation_policy != .accelerate_fp64_summary_v1 or
        !options.independent_fp64_publication or options.validation != .whole_head or options.storage_precision != .f16 or
        options.max_queries_per_consumer != 768 or options.fit.max_workspace_bytes != 2 * 1024 * 1024 * 1024 or
        options.min_global_savings_ratio != 1.5)
        return error.InvalidFitConfig;
    switch (options.adaptive_v18_workspace_profile) {
        .standard => return error.InvalidFitConfig,
        .resident_native_hybrid_2g_v1 => if (options.adaptive_v18_primary_budget_percent != 91 or
            options.adaptive_v18_fallback_budget_percent != 95 or options.adaptive_v18_max_budget_fallbacks != 3 or
            options.adaptive_v18_geometry_route != .flat_gram_only) return error.InvalidFitConfig,
        .resident_e4_32512_bounded_2g_v1 => if (options.adaptive_v18_primary_budget_percent != 91 or
            options.adaptive_v18_fallback_budget_percent != 95 or options.adaptive_v18_max_budget_fallbacks != 3 or
            options.adaptive_v18_geometry_route != .bounded_page_over_96 or options.adaptive_v18_bounded_tile_queries != 64 or
            options.adaptive_v18_bounded_builder_transient_bytes != 256 * 1024 * 1024 or !options.adaptive_v18_bounded_delta_accelerated or
            !options.adaptive_v18_bounded_tsqr_value_backend) return error.InvalidFitConfig,
        .resident_e2_formal_2g_p60_no_fallback_v1 => if (options.adaptive_v18_primary_budget_percent != 60 or
            options.adaptive_v18_fallback_budget_percent != 0 or options.adaptive_v18_max_budget_fallbacks != 0 or
            options.adaptive_v18_geometry_route != .bounded_page_over_96) return error.InvalidFitConfig,
    }
    return 2 * 1024 * 1024 * 1024;
}

test "resident native workspace cap requires its complete staged policy" {
    try std.testing.expectEqual(v18_head_workspace_bytes, try v18HeadWorkspaceBytesForOptions(.{}));
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), try v18HeadWorkspaceBytesForOptions(.{ .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1 }));
    var resident: Options = .{
        .adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .independent_fp64_publication = true,
        .validation = .whole_head,
        .storage_precision = .f16,
        .max_queries_per_consumer = 768,
        .fit = .{ .target_rows = 1, .max_workspace_bytes = 2 * 1024 * 1024 * 1024 },
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .min_global_savings_ratio = 1.5,
    };
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), try v18HeadWorkspaceBytesForOptions(resident));
    resident.max_transient_host_bytes = 3 * 1024 * 1024 * 1024;
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), try v18AvailableHeadWorkspaceBytes(resident, 128 * 1024 * 1024));
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), try v18AvailableHeadWorkspaceBytes(resident, 2 * 1024 * 1024 * 1024));
    try std.testing.expect(criticalCorePolicyValid(resident));
    var incompatible = resident;
    incompatible.source_has_metadata = true;
    try std.testing.expect(!criticalCorePolicyValid(incompatible));
    incompatible = resident;
    incompatible.adaptive_v18_train_primary_policy_comparison = true;
    try std.testing.expect(!criticalCorePolicyValid(incompatible));
    resident.private_performance_policy = .diagnostic_capture_gram_batch256_v1;
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), try v18HeadWorkspaceBytesForOptions(resident));
    try std.testing.expect(criticalCorePolicyValid(resident));
    resident.private_performance_policy = .owned_capture_summary_incremental_reduction_v1;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(resident));
    resident.private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    resident.staged_operator_strategy = false;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(resident));
    resident.staged_operator_strategy = true;
    resident.adaptive_v18_postfit_policy = .disabled;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(resident));
    resident.adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    resident.fit.max_workspace_bytes = 1024 * 1024 * 1024;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(resident));
}

test "standard v18 availability cannot exceed its authorized per-head cap" {
    var standard: Options = .{
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .max_transient_host_bytes = 3 * 1024 * 1024 * 1024,
    };
    const cap = try v18HeadWorkspaceBytesForOptions(standard);
    try std.testing.expectEqual(@as(usize, 1024 * 1024 * 1024), cap);
    try std.testing.expectEqual(cap, try v18AvailableHeadWorkspaceBytes(standard, 128 * 1024 * 1024));
    // Raising only aggregate headroom cannot silently raise the head policy.
    standard.max_transient_host_bytes = 6 * 1024 * 1024 * 1024;
    try std.testing.expectEqual(cap, try v18AvailableHeadWorkspaceBytes(standard, 128 * 1024 * 1024));
    // Real remaining headroom below the policy cap remains the tighter bound.
    standard.max_transient_host_bytes = 640 * 1024 * 1024;
    try std.testing.expectEqual(@as(usize, 512 * 1024 * 1024), try v18AvailableHeadWorkspaceBytes(standard, 128 * 1024 * 1024));
    standard.max_transient_host_bytes = 127 * 1024 * 1024;
    try std.testing.expectError(error.V18TrainingCacheUnsupported, v18AvailableHeadWorkspaceBytes(standard, 128 * 1024 * 1024));
}

test "formal E2 resident workspace admits only sealed p60 no-fallback tuple" {
    var options: Options = .{
        .adaptive_v18_workspace_profile = .resident_e2_formal_2g_p60_no_fallback_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .bounded_page_over_96,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .independent_fp64_publication = true,
        .validation = .whole_head,
        .storage_precision = .f16,
        .max_queries_per_consumer = 768,
        .fit = .{ .target_rows = 1, .max_workspace_bytes = 2 * 1024 * 1024 * 1024 },
        .adaptive_v18_primary_budget_percent = 60,
        .adaptive_v18_fallback_budget_percent = 0,
        .adaptive_v18_max_budget_fallbacks = 0,
        .min_global_savings_ratio = 1.5,
    };
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), try v18HeadWorkspaceBytesForOptions(options));
    try std.testing.expect(criticalCorePolicyValid(options));
    options.adaptive_v18_primary_budget_percent = 61;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(options));
    options.adaptive_v18_primary_budget_percent = 60;
    options.adaptive_v18_fallback_budget_percent = 95;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(options));
    options.adaptive_v18_fallback_budget_percent = 0;
    options.adaptive_v18_workspace_profile = .resident_native_hybrid_2g_v1;
    try std.testing.expectError(error.InvalidFitConfig, v18HeadWorkspaceBytesForOptions(options));
}

/// Builds the current-head ranges which are still exact and newly eligible.
/// Logical gaps and any prior fitted row split runs; callers must never rebuild
/// a fresh contiguous middle from `prefix_rows..old_rows` for successive cycles.
pub fn headFitRanges(allocator: std.mem.Allocator, original: checkpoint.Head, plan: HeadPlan, chunk_rows: usize) ![]FitRange {
    if (chunk_rows == 0 or original.positions.len != original.coverage_ends.len or original.positions.len != original.bias.len or plan.prefix_rows > plan.old_rows or plan.old_rows > original.positions.len)
        return error.InvalidFitConfig;
    const scratch = try allocator.alloc(FitRange, plan.old_rows - plan.prefix_rows);
    defer allocator.free(scratch);
    var count: usize = 0;
    var at = plan.prefix_rows;
    while (at < plan.old_rows) {
        while (at < plan.old_rows and (original.positions[at] == std.math.maxInt(u64) or original.coverage_ends[at] != original.positions[at] + 1 or original.bias[at] != 0)) : (at += 1) {}
        if (at == plan.old_rows) break;
        var run_end = at + 1;
        while (run_end < plan.old_rows and
            original.positions[run_end] != std.math.maxInt(u64) and original.coverage_ends[run_end] == original.positions[run_end] + 1 and
            original.positions[run_end - 1] != std.math.maxInt(u64) and
            original.bias[run_end] == 0 and
            original.positions[run_end] == original.positions[run_end - 1] + 1) : (run_end += 1)
        {}
        var start = at;
        while (start < run_end) {
            const rows = @min(chunk_rows, run_end - start);
            scratch[count] = .{ .start = start, .rows = rows };
            count += 1;
            start += rows;
        }
        at = run_end;
    }
    if (count == 0) return error.NoEligibleAttentionRows;
    return allocator.dupe(FitRange, scratch[0..count]);
}

/// The admission planner and fitter share the same coverage boundaries. Prior
/// fitted rows remain immutable; only newly eligible exact rows can shrink.
/// The caller validates the checkpoint's complete shape and numerical state.
pub fn headPlan(logical_length: u64, original: checkpoint.Head, options: Options) !?HeadPlan {
    if (options.chunk_rows == 0 or options.retention_divisor < 2 or options.retention_divisor > 4 or
        original.positions.len != original.coverage_ends.len or original.positions.len != original.bias.len)
        return error.InvalidFitConfig;
    const boundary = logical_length - @min(logical_length, options.recent_rows);
    if (boundary < original.compacted_until) return error.CompactionRecentWindowUnavailable;
    var old_rows: usize = 0;
    while (old_rows < original.positions.len and original.coverage_ends[old_rows] <= boundary) : (old_rows += 1) {}
    var prefix = @min(options.prefix_rows, std.math.cast(usize, logical_length) orelse return error.InvalidFitConfig);
    if (prefix > original.positions.len) return error.CompactionPrefixUnavailable;
    for (0..prefix) |i| {
        if (original.positions[i] != i or original.coverage_ends[i] != i + 1 or original.bias[i] != 0)
            return error.CompactionPrefixUnavailable;
    }
    // Already fitted history is immutable across cycles. Only newly eligible
    // exact rows enter the solver, avoiding repeated approximation of the same
    // vectors. Whole-head validation still includes the retained fitted rows.
    while (prefix < original.positions.len and original.coverage_ends[prefix] <= original.compacted_until) : (prefix += 1) {}
    if (old_rows <= prefix or old_rows - prefix < 2) return null;
    var target_rows: usize = original.positions.len - old_rows + prefix;
    var planned_offset: usize = prefix;
    while (planned_offset < old_rows) {
        const count = @min(options.chunk_rows, old_rows - planned_offset);
        target_rows = try add(target_rows, std.math.divCeil(usize, count, options.retention_divisor) catch unreachable);
        planned_offset += count;
    }
    if (target_rows == original.positions.len) return null;
    return .{ .old_rows = old_rows, .prefix_rows = prefix, .target_rows = target_rows, .block_rows = @min(old_rows - prefix, options.chunk_rows) };
}

fn compactHeadCumulative(
    allocator: std.mem.Allocator,
    original: checkpoint.Head,
    plan: HeadPlan,
    train: []const f64,
    heldout: []const f64,
    scale: f64,
    softcap: f64,
    options: Options,
    validation_gate: ?*HeadValidationGate,
) !checkpoint.Head {
    if (options.validation != .whole_head or options.storage_precision != .f16 or
        options.min_global_savings_ratio == null or !options.independent_fp64_publication)
        return error.InvalidFitConfig;
    const d = original.dim;
    const ok = try converted(allocator, original.keys);
    defer allocator.free(ok);
    const ov = try converted(allocator, original.values);
    defer allocator.free(ov);
    const ob = try converted(allocator, original.bias);
    defer allocator.free(ob);
    const source_head: solver.Head = .{ .keys = ok, .values = ov, .bias = ob, .rows = original.positions.len, .dim = d, .scale = scale, .softcap = softcap };
    const range_count = std.math.divCeil(usize, plan.old_rows - plan.prefix_rows, options.chunk_rows) catch return error.InvalidFitConfig;
    const ranges = try allocator.alloc(cumulative_cache.Range, range_count);
    defer allocator.free(ranges);
    var at = plan.prefix_rows;
    for (ranges) |*range| {
        range.* = .{ .start = at, .rows = @min(options.chunk_rows, plan.old_rows - at) };
        at += range.rows;
    }
    var cfg = options.fit;
    cfg.selection = .residual_omp;
    cfg.mass_solver = .active_set;
    cfg.ridge = 1e-8;
    var cache_counters: cumulative.TrainingCacheCounters = .{};
    var fitted = try cumulative.runWithPageCachePolicy(allocator, source_head, train, heldout, ranges, cfg, options.fit.max_workspace_bytes, options.cumulative_page_cache_policy, &cache_counters);
    defer fitted.deinit();
    if (options.stats) |stats| {
        stats.cumulative_page_cache_pages +|= cache_counters.pages_built;
        stats.cumulative_page_cache_hits +|= cache_counters.page_hits;
        stats.cumulative_page_cache_misses +|= cache_counters.page_misses;
        stats.cumulative_page_cache_fallbacks_before_gpu +|= cache_counters.page_fallbacks_before_gpu;
        stats.cumulative_page_cache_retained_bytes_peak = @max(stats.cumulative_page_cache_retained_bytes_peak, cache_counters.retained_bytes_peak);
    }
    const rows = fitted.head.rows;
    const positions = try allocator.alloc(u64, rows);
    errdefer allocator.free(positions);
    const ends = try allocator.alloc(u64, rows);
    errdefer allocator.free(ends);
    const keys = try allocator.alloc(f32, try mul(rows, d));
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f32, keys.len);
    errdefer allocator.free(values);
    const biases = try allocator.alloc(f32, rows);
    errdefer allocator.free(biases);
    for (fitted.source_indices, fitted.coverage_source_indices, 0..) |source_index, coverage_index, row| {
        if (source_index >= original.positions.len or coverage_index >= original.coverage_ends.len) return error.InvalidAttentionShape;
        positions[row] = original.positions[source_index];
        ends[row] = original.coverage_ends[coverage_index];
        biases[row] = @floatCast(fitted.head.bias.?[row]);
    }
    for (fitted.head.keys, fitted.head.values, 0..) |key, value, index| {
        keys[index] = @floatCast(key);
        values[index] = @floatCast(value);
    }
    var any_accepted = false;
    for (fitted.chunks) |chunk| {
        any_accepted = any_accepted or chunk.chosen_divisor != 0;
        if (options.stats) |stats| {
            stats.attempted_chunks += chunk.attempts;
            if (chunk.chosen_divisor != 0) stats.completed_chunks += 1;
            stats.retention_retries += chunk.attempts -| 1;
        }
    }
    if (options.stats) |stats| {
        stats.rounded_fit.max_relative_mass_error = fitted.final_mass_error;
        stats.rounded_fit.relative_output_rmse = fitted.final_output_rmse;
    }
    var result: checkpoint.Head = .{
        .dim = d,
        .compacted_until = if (any_accepted) original.coverage_ends[plan.old_rows - 1] else original.compacted_until,
        .positions = positions,
        .coverage_ends = ends,
        .keys = keys,
        .values = values,
        .bias = biases,
    };
    if (options.fp8_checkpoint) result.fp8 = original.fp8;
    result.compressed = try original.compressed.clone(allocator);
    errdefer result.compressed.deinit(allocator);
    const codec_run = firstAcceptedChunkRun(fitted.chunks);
    var encoded_first = plan.prefix_rows;
    var encoded_end = plan.prefix_rows;
    if (codec_run) |run| {
        while (encoded_first < rows and fitted.coverage_source_indices[encoded_first] < run.start) : (encoded_first += 1) {}
        encoded_end = encoded_first;
        while (encoded_end < rows and fitted.coverage_source_indices[encoded_end] < run.end) : (encoded_end += 1) {}
        if (encoded_first == encoded_end) return error.InvalidAttentionShape;
    }
    if (validation_gate) |gate| try gate.enter(options.fit.control);
    defer if (validation_gate) |gate| gate.leave();
    var authorization_options = options;
    if (encoded_end == encoded_first) {
        authorization_options.fp8_checkpoint = false;
        authorization_options.rotated4_checkpoint = false;
    }
    try authorizeRoundedHead(allocator, source_head, fitted.head, &result, encoded_first, encoded_end, train, heldout, authorization_options);
    return result;
}

const CodecSourceRun = struct { start: usize, end: usize };

/// Select the first contiguous accepted run. Leading exact chunks are skipped;
/// the first exact gap after acceptance terminates codec ownership.
fn firstAcceptedChunkRun(chunks: []const cumulative.ChunkResult) ?CodecSourceRun {
    var result: ?CodecSourceRun = null;
    for (chunks) |chunk| {
        if (chunk.chosen_divisor == 0) {
            if (result != null) break;
            continue;
        }
        if (result) |*run| {
            if (chunk.start != run.end) break;
            run.end = chunk.start + chunk.source_rows;
        } else result = .{ .start = chunk.start, .end = chunk.start + chunk.source_rows };
    }
    return result;
}

fn codecRunTestChunk(start: usize, source_rows: usize, chosen_divisor: usize) cumulative.ChunkResult {
    return .{
        .start = start,
        .source_rows = source_rows,
        .chosen_divisor = chosen_divisor,
        .candidate_rows = if (chosen_divisor == 0) source_rows else source_rows / chosen_divisor,
        .attempts = if (chosen_divisor == 0) 3 else 1,
        .last_candidate_metrics_valid = chosen_divisor != 0,
        .last_rejection_reason = if (chosen_divisor == 0) "preserved" else "",
        .last_candidate_mass_error = 0,
        .last_candidate_output_rmse = 0,
        .cumulative_mass_error = 0,
        .cumulative_output_rmse = 0,
    };
}

test "cumulative codec selects first accepted run after leading preserved chunks" {
    const chunks = [_]cumulative.ChunkResult{
        codecRunTestChunk(0, 1024, 0),
        codecRunTestChunk(1024, 1024, 2),
        codecRunTestChunk(2048, 1024, 3),
        codecRunTestChunk(3072, 1024, 0),
        codecRunTestChunk(4096, 1024, 2),
    };
    try std.testing.expectEqual(CodecSourceRun{ .start = 1024, .end = 3072 }, firstAcceptedChunkRun(&chunks).?);

    const noncontiguous = [_]cumulative.ChunkResult{
        codecRunTestChunk(0, 1024, 0),
        codecRunTestChunk(1024, 1024, 2),
        codecRunTestChunk(3072, 1024, 2),
    };
    try std.testing.expectEqual(CodecSourceRun{ .start = 1024, .end = 2048 }, firstAcceptedChunkRun(&noncontiguous).?);

    const preserved = [_]cumulative.ChunkResult{
        codecRunTestChunk(0, 1024, 0),
        codecRunTestChunk(1024, 1024, 0),
    };
    try std.testing.expect(firstAcceptedChunkRun(&preserved) == null);
}

fn compactHead(allocator: std.mem.Allocator, source: checkpoint.State, donor_index: usize, head_index: usize, batches: []const ReferenceBatch, layout: Layout, options: Options, resident: usize, validation_gate: ?*HeadValidationGate) !?checkpoint.Head {
    const original = source.donors[donor_index].heads[head_index];
    const d = original.dim;
    const plan = (try headPlan(source.logical_length, original, options)) orelse return null;
    const old_rows = plan.old_rows;
    const prefix = plan.prefix_rows;
    const target_rows = plan.target_rows;
    var train_elements: usize = 0;
    var heldout_elements: usize = 0;
    for (batches) |batch| if (belongs(batch, source, layout, donor_index, head_index)) {
        train_elements = try add(train_elements, batch.train.len);
        heldout_elements = try add(heldout_elements, batch.heldout.len);
    };
    const block_rows = plan.block_rows;
    const workspace = try headWorkspaceBytes(original.positions.len, target_rows, block_rows, train_elements, heldout_elements, d, options);
    try budget(try add(resident, workspace.bytes), options.max_transient_host_bytes);
    try budget(workspace.solver_bytes, options.fit.max_workspace_bytes);

    const train = try allocator.alloc(f64, train_elements);
    defer allocator.free(train);
    const heldout = try allocator.alloc(f64, heldout_elements);
    defer allocator.free(heldout);
    var ti: usize = 0;
    var hi: usize = 0;
    var train_positions = std.ArrayList(u64).empty;
    defer train_positions.deinit(allocator);
    var heldout_positions = std.ArrayList(u64).empty;
    defer heldout_positions.deinit(allocator);
    var query_consumer_layers = std.ArrayList(u32).empty;
    defer query_consumer_layers.deinit(allocator);
    var query_heads = std.ArrayList(u32).empty;
    defer query_heads.deinit(allocator);
    var query_train_counts = std.ArrayList(u32).empty;
    defer query_train_counts.deinit(allocator);
    var query_heldout_counts = std.ArrayList(u32).empty;
    defer query_heldout_counts.deinit(allocator);
    for (batches) |batch| if (belongs(batch, source, layout, donor_index, head_index)) {
        for (batch.train) |x| {
            train[ti] = x;
            ti += 1;
        }
        for (batch.heldout) |x| {
            heldout[hi] = x;
            hi += 1;
        }
        if (options.rejected_head_capture != null) {
            try train_positions.appendSlice(allocator, batch.train_positions);
            try heldout_positions.appendSlice(allocator, batch.heldout_positions);
            try query_consumer_layers.append(allocator, batch.consumer_layer);
            try query_heads.append(allocator, batch.query_head);
            try query_train_counts.append(allocator, @intCast(batch.train_positions.len));
            try query_heldout_counts.append(allocator, @intCast(batch.heldout_positions.len));
        }
    };

    if (options.admission == .cumulative_ladder_4_3_2) {
        return try compactHeadCumulative(allocator, original, plan, train, heldout, layout.scales_by_donor[donor_index], layout.softcaps_by_donor[donor_index], options, validation_gate);
    }

    const positions = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(positions);
    const ends = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(ends);
    const keys = try allocator.alloc(f32, try mul(target_rows, d));
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f32, keys.len);
    errdefer allocator.free(values);
    const biases = try allocator.alloc(f32, target_rows);
    errdefer allocator.free(biases);
    @memcpy(positions[0..prefix], original.positions[0..prefix]);
    @memcpy(ends[0..prefix], original.coverage_ends[0..prefix]);
    @memcpy(biases[0..prefix], original.bias[0..prefix]);
    @memcpy(keys[0 .. prefix * d], original.keys[0 .. prefix * d]);
    @memcpy(values[0 .. prefix * d], original.values[0 .. prefix * d]);
    var offset: usize = prefix;
    var out: usize = prefix;
    while (offset < old_rows) {
        try options.fit.control.check();
        const count = @min(options.chunk_rows, old_rows - offset);
        const k64 = try converted(allocator, original.keys[offset * d ..][0 .. count * d]);
        defer allocator.free(k64);
        const v64 = try converted(allocator, original.values[offset * d ..][0 .. count * d]);
        defer allocator.free(v64);
        const b64 = try converted(allocator, original.bias[offset..][0..count]);
        defer allocator.free(b64);
        const h: solver.Head = .{ .keys = k64, .values = v64, .bias = b64, .rows = count, .dim = d, .scale = layout.scales_by_donor[donor_index], .softcap = layout.softcaps_by_donor[donor_index] };
        var cfg = options.fit;
        // These intermediate blocks are private. Whole-head mode rejects the
        // assembled rounded candidate before it can leave this function.
        if (options.validation == .whole_head) {
            cfg.max_relative_mass_error = std.math.floatMax(f64);
            cfg.max_relative_output_rmse = std.math.floatMax(f64);
        }
        cfg.target_rows = std.math.divCeil(usize, count, options.retention_divisor) catch unreachable;
        if (options.stats) |stats| {
            stats.donor_layer = source.donors[donor_index].layer;
            stats.head = head_index;
            stats.chunk_start = offset;
            stats.attempted_chunks += 1;
            stats.validating_storage = false;
            stats.validating_head = false;
            stats.rounded_fit = .{};
            cfg.metrics = &stats.last_fit;
        }
        var fitted = try solver.fit(allocator, h, train, heldout, cfg);
        defer fitted.deinit();
        if (options.stats) |stats| stats.max_nnls_sweeps = @max(stats.max_nnls_sweeps, fitted.metrics.nnls_sweeps);
        for (fitted.source_indices, 0..) |idx, row| {
            positions[out + row] = original.positions[offset + idx];
            ends[out + row] = original.coverage_ends[offset + count - 1];
            biases[out + row] = @floatCast(fitted.bias[row]);
        }
        for (fitted.keys, fitted.values, 0..) |k, v, i| {
            keys[out * d + i] = if (options.storage_precision == .f16) @as(f16, @floatCast(k)) else @floatCast(k);
            values[out * d + i] = if (options.storage_precision == .f16) @as(f16, @floatCast(v)) else @floatCast(v);
        }
        // Check held-out fit again AFTER storage rounding, using the same
        // thresholds. This catches overflowing or unstable fitted f32 values.
        for (fitted.keys, fitted.values, 0..) |*k, *v, i| {
            k.* = keys[out * d + i];
            v.* = values[out * d + i];
        }
        for (fitted.bias, 0..) |*b, i| b.* = biases[out + i];
        var rounded_cfg = cfg;
        if (options.stats) |stats| {
            stats.validating_storage = true;
            rounded_cfg.metrics = &stats.rounded_fit;
        }
        try validateRounded(allocator, h, fitted.head(h), heldout, rounded_cfg);
        if (options.stats) |stats| stats.completed_chunks += 1;
        out += fitted.bias.len;
        offset += count;
    }
    const remaining = original.positions.len - old_rows;
    @memcpy(positions[out..], original.positions[old_rows..]);
    @memcpy(ends[out..], original.coverage_ends[old_rows..]);
    @memcpy(biases[out..], original.bias[old_rows..]);
    @memcpy(keys[out * d ..], original.keys[old_rows * d ..]);
    @memcpy(values[out * d ..], original.values[old_rows * d ..]);
    std.debug.assert(out + remaining == target_rows);
    var result: checkpoint.Head = .{ .dim = d, .compacted_until = original.coverage_ends[old_rows - 1], .positions = positions, .coverage_ends = ends, .keys = keys, .values = values, .bias = biases };
    if (options.fp8_checkpoint) result.fp8 = original.fp8;
    result.compressed = try original.compressed.clone(allocator);
    errdefer result.compressed.deinit(allocator);
    if (options.validation == .whole_head) {
        // Bound simultaneous whole-head FP64 conversions at long contexts.
        // Chunk fitting still runs on all independently owned workers.
        if (validation_gate) |gate| try gate.enter(options.fit.control);
        defer if (validation_gate) |gate| gate.leave();
        if (options.stats) |stats| {
            stats.validating_head = true;
            stats.validating_storage = true;
        }
        const ok = try converted(allocator, original.keys);
        defer allocator.free(ok);
        const ov = try converted(allocator, original.values);
        defer allocator.free(ov);
        const ob = try converted(allocator, original.bias);
        defer allocator.free(ob);
        const ck = try converted(allocator, keys);
        defer allocator.free(ck);
        const cv = try converted(allocator, values);
        defer allocator.free(cv);
        const cb = try converted(allocator, biases);
        defer allocator.free(cb);
        const h: solver.Head = .{ .keys = ok, .values = ov, .bias = ob, .rows = original.positions.len, .dim = d, .scale = layout.scales_by_donor[donor_index], .softcap = layout.softcaps_by_donor[donor_index] };
        var compact = h;
        compact.keys = ck;
        compact.values = cv;
        compact.bias = cb;
        compact.rows = target_rows;
        authorizeRoundedHead(allocator, h, compact, &result, prefix, out, train, heldout, options) catch |err| {
            if (err == error.CompactionFitRejected and
                options.retention_divisor == (options.min_retention_divisor orelse options.retention_divisor))
                if (options.rejected_head_capture) |capture| {
                    capture.capture_fn(capture.context, .{
                        .donor_layer = source.donors[donor_index].layer,
                        .kv_head = head_index,
                        .retention_divisor = options.retention_divisor,
                        .logical_length = source.logical_length,
                        .prefix_rows = prefix,
                        .recent_rows = options.recent_rows,
                        .chunk_rows = options.chunk_rows,
                        .scale = h.scale,
                        .softcap = h.softcap,
                        .original = h,
                        .candidate = compact,
                        .original_positions = original.positions,
                        .original_coverage_ends = original.coverage_ends,
                        .candidate_positions = positions,
                        .candidate_coverage_ends = ends,
                        .train_queries = train,
                        .heldout_queries = heldout,
                        .train_positions = train_positions.items,
                        .heldout_positions = heldout_positions.items,
                        .query_consumer_layers = query_consumer_layers.items,
                        .query_heads = query_heads.items,
                        .query_train_counts = query_train_counts.items,
                        .query_heldout_counts = query_heldout_counts.items,
                        .metrics = if (options.stats) |stats| stats.rounded_fit else .{},
                    });
                };
            return err;
        };
    }
    return result;
}

/// Single final numerical and codec publication authority shared by fixed and
/// cumulative assembly. Callers own source/candidate metadata and rejection
/// capture; this function never publishes a partially authorized head.
fn authorizeRoundedHead(allocator: std.mem.Allocator, original: solver.Head, compact: solver.Head, result: *checkpoint.Head, first: usize, end: usize, train: []const f64, heldout: []const f64, options: Options) !void {
    var validation_context: validation_accelerate.Context = .{};
    const trusted_backend = trustedValidationBackend(options.independent_validation_policy, &validation_context);
    var cfg = finalAuthorizationConfig(options.fit, options.independent_fp64_publication, trusted_backend);
    if (options.stats) |stats| cfg.metrics = &stats.rounded_fit;
    try validateRounded(allocator, original, compact, heldout, cfg);
    if (options.fp8_checkpoint) {
        const accepted = try quantizeCheckpointHead(allocator, original, compact, result, first, end, heldout, cfg, options.codec_diagnostic);
        if (options.stats) |stats| {
            if (accepted) stats.fp8_heads += 1 else stats.fp8_fallback_heads += 1;
            try recordFp8Coverage(stats, end - first, accepted);
        }
    }
    if (options.rotated4_checkpoint and @import("turboquant_rotated.zig").supported(original.dim)) {
        const previous_rows = result.compressed.rows();
        const accepted = try quantizeRotatedHead(allocator, original, compact, result, first, end, train, heldout, cfg, options.rotated4_risk_retention, options.codec_diagnostic);
        if (options.stats) |stats| {
            if (accepted) stats.rotated4_heads += 1 else stats.rotated4_fallback_heads += 1;
            const compressed_rows = result.compressed.rows() - previous_rows;
            stats.rotated4_eligible_rows += end - first;
            stats.rotated4_compressed_rows += compressed_rows;
            stats.rotated4_exact_rows += end - first - compressed_rows;
        }
    }
    if (options.stats) |stats| {
        stats.fp64_validation_summary_calls = try add(stats.fp64_validation_summary_calls, validation_context.successfulCalls());
        stats.fp64_validation_dgemm_calls = try add(stats.fp64_validation_dgemm_calls, try mul(validation_context.successfulCalls(), 2));
        stats.fp64_validation_max_batch_rows = @max(stats.fp64_validation_max_batch_rows, validation_context.maximum_batch_rows);
        stats.fp64_validation_workspace_peak_bytes = @max(stats.fp64_validation_workspace_peak_bytes, validation_context.workspace_peak_bytes);
        stats.validated_heads += 1;
        stats.max_head_mass_error = @max(stats.max_head_mass_error, stats.rounded_fit.max_relative_mass_error);
        stats.max_head_output_rmse = @max(stats.max_head_output_rmse, stats.rounded_fit.relative_output_rmse);
    }
}

fn recordFp8Coverage(stats: *Stats, eligible: usize, accepted: bool) !void {
    stats.fp8_eligible_rows = try std.math.add(usize, stats.fp8_eligible_rows, eligible);
    if (accepted) {
        stats.fp8_encoded_rows = try std.math.add(usize, stats.fp8_encoded_rows, eligible);
    } else {
        stats.fp8_exact_rows = try std.math.add(usize, stats.fp8_exact_rows, eligible);
    }
}

test "FP8 coverage counters aggregate encoded exact fallback and zero-format rows" {
    var stats: Stats = .{};
    try recordFp8Coverage(&stats, 768, true);
    try recordFp8Coverage(&stats, 256, false);
    try recordFp8Coverage(&stats, 0, true);
    try recordFp8Coverage(&stats, 0, false);
    try std.testing.expectEqual(@as(usize, 1024), stats.fp8_eligible_rows);
    try std.testing.expectEqual(@as(usize, 768), stats.fp8_encoded_rows);
    try std.testing.expectEqual(@as(usize, 256), stats.fp8_exact_rows);
}

/// Shared final validation/codec tail for the private adaptive adapter. The
/// supplied range is one contiguous newly fitted run in candidate-row space;
/// prior codec spans and exact identity/gap rows remain untouched.
pub fn authorizeAdaptiveCodec(allocator: std.mem.Allocator, original: solver.Head, compact: solver.Head, result: *checkpoint.Head, first: usize, end: usize, train: []const f64, heldout: []const f64, options: Options) !void {
    if (options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or first >= end or end > compact.rows)
        return error.InvalidFitConfig;
    return authorizeRoundedHead(allocator, original, compact, result, first, end, train, heldout, options);
}

fn quantizeCheckpointHead(allocator: std.mem.Allocator, original: solver.Head, compact: solver.Head, result: *checkpoint.Head, first: usize, end: usize, queries: []const f64, cfg: solver.Config, diagnostic: ?*CodecDiagnostic) !bool {
    if (diagnostic) |value| value.* = .{ .kind = .fp8, .first = first, .end = end };
    const codec = checkpoint.codec;
    const previous = result.fp8;
    // The wire format has one contiguous FP8 range per head. Keep its scales
    // and bytes immutable across fits. A new range can extend it only when
    // adjacent and representable at the same scales; otherwise new rows keep
    // their validated f16 representation and the prior FP8 range survives.
    if (previous.rows != 0 and @as(usize, previous.first_row) + previous.rows != first) {
        if (diagnostic) |value| value.outcome = .noncontiguous_range;
        return false;
    }
    const begin_element = first * result.dim;
    const end_element = end * result.dim;
    const keys = @constCast(compact.keys[begin_element..end_element]);
    const values = @constCast(compact.values[begin_element..end_element]);
    const key_scale = if (previous.rows == 0) codec.scaleFor(keys) else previous.key_scale;
    const value_scale = if (previous.rows == 0) codec.scaleFor(values) else previous.value_scale;
    var offset: usize = 0;
    while (offset < keys.len) {
        try cfg.control.check();
        const count = @min(@as(usize, 4096), keys.len - offset);
        codec.round(keys[offset..][0..count], key_scale) catch {
            if (diagnostic) |value| value.outcome = .invalid_quantized_row;
            return false;
        };
        codec.round(values[offset..][0..count], value_scale) catch {
            if (diagnostic) |value| value.outcome = .invalid_quantized_row;
            return false;
        };
        offset += count;
    }
    var metrics: solver.Metrics = .{};
    var validation = cfg;
    validation.metrics = &metrics;
    validateRounded(allocator, original, compact, queries, validation) catch |err| {
        if (err == error.CompactionFitRejected) {
            if (diagnostic) |value| {
                value.outcome = .postcodec_heldout_gate;
                value.metrics = metrics;
                value.metrics.quality_evaluated = true;
            }
            return false;
        }
        return err;
    };
    // Publication happens only after combined fitting + quantization passes.
    // The unmodified f32 arrays retain the f16 fallback until this point.
    for (keys, values, begin_element..) |k, v, i| {
        if (i % 4096 == 0) try cfg.control.check();
        @constCast(result.keys)[i] = @floatCast(k);
        @constCast(result.values)[i] = @floatCast(v);
    }
    const encoded_first = if (previous.rows == 0) first else previous.first_row;
    result.fp8 = .{ .first_row = @intCast(encoded_first), .rows = @intCast(end - encoded_first), .key_scale = key_scale, .value_scale = value_scale };
    if (cfg.metrics) |out_metrics| out_metrics.* = metrics;
    if (diagnostic) |value| {
        value.outcome = .accepted;
        value.metrics = metrics;
        value.metrics.quality_evaluated = true;
    }
    return true;
}

fn validateRounded(allocator: std.mem.Allocator, original: solver.Head, compact: solver.Head, queries: []const f64, cfg: solver.Config) !void {
    var metrics: solver.Metrics = .{};
    defer {
        if (cfg.metrics) |out| out.* = metrics;
    }
    try compact.validate();
    metrics = try solver.validationMetrics(allocator, original, compact, queries, cfg);
    if (!std.math.isFinite(metrics.relative_output_rmse) or
        metrics.relative_output_rmse > cfg.max_relative_output_rmse or
        metrics.max_relative_mass_error > cfg.max_relative_mass_error) return error.CompactionFitRejected;
}

/// Dense exact CUDA storage has no compaction metadata; staged candidates
/// carry bias/position/coverage arrays even for unchanged heads. Local windows
/// are deliberately excluded from the global-cache saving, but remain counted
/// in the complete source/candidate host and device admission budgets.
pub fn globalStorageBytes(state_value: checkpoint.State, precision: @FieldType(Options, "storage_precision"), staged: bool) !usize {
    var total: usize = 0;
    for (state_value.donors) |donor| {
        if (donor.window != 0) continue;
        for (donor.heads) |head| {
            const exact_elements = try mul(head.positions.len - head.compressed.rows(), head.dim);
            const kv_bytes = try add(try mul(exact_elements, if (precision == .f16) 4 else 8), head.compressed.bytes());
            const metadata = if (staged or head.compacted_until != 0) try mul(head.positions.len, 20) else 0;
            total = try add(total, try add(kv_bytes, metadata));
        }
    }
    return total;
}

/// Cost of the same fitted geometry before this operation's quantization.
/// Existing compressed history is immutable and keeps its actual storage cost;
/// only newly fitted rows are priced at the serving precision. This separates
/// the fit's publication decision from the optional codec's additional savings.
fn fittedGlobalStorageBytes(source: checkpoint.State, candidate: checkpoint.State, precision: @FieldType(Options, "storage_precision")) !usize {
    var total: usize = 0;
    for (source.donors, candidate.donors) |before, after| {
        if (after.window != 0) continue;
        for (before.heads, after.heads) |original, fitted| {
            const exact_rows = std.math.sub(usize, fitted.positions.len, original.compressed.rows()) catch return error.InvalidKvCheckpoint;
            const exact_bytes = try mul(try mul(exact_rows, fitted.dim), if (precision == .f16) 4 else 8);
            total = try add(total, try add(try add(exact_bytes, original.compressed.bytes()), try mul(fitted.positions.len, 20)));
        }
    }
    return total;
}

/// Prepare a complete candidate without mutating source. Commit is a separate
/// operation after device upload has succeeded. Chunking bounds QR work; every
/// compacted row records the coverage of its entire source chunk.
pub fn prepare(allocator: std.mem.Allocator, source: checkpoint.State, batches: []const ReferenceBatch, layout: Layout, options: Options) !checkpoint.Owned {
    return prepareWithBackends(allocator, source, batches, layout, options, &.{});
}

/// Parallel preparation uses a known thread-safe allocator. Backends must be
/// independently owned, remain alive through this call, and share an admitted
/// aggregate device budget. The source and queries are immutable throughout.
pub fn prepareParallel(source: checkpoint.State, batches: []const ReferenceBatch, layout: Layout, options: Options, backends: []const solver.Backend) !checkpoint.Owned {
    if (backends.len < 2 or backends.len > 4) return error.InvalidFitConfig;
    for (backends, 0..) |backend, i| if (backend.context) |context| {
        for (backends[0..i]) |previous| if (previous.context == context) return error.InvalidFitConfig;
    };
    return prepareWithBackends(std.heap.page_allocator, source, batches, layout, options, backends);
}

fn preparedHead(allocator: std.mem.Allocator, source: checkpoint.State, metadata: SourceMetadata, donor_index: usize, head_index: usize, batches: []const ReferenceBatch, layout: Layout, options: Options, resident: usize, validation_gate: ?*HeadValidationGate) !checkpoint.Head {
    try options.fit.control.check();
    const donor = source.donors[donor_index];
    if (donor.window == 0 and options.fit_strategy == .identity_chunk_coefficients_adaptive_v18) {
        const plan = (try headPlan(source.logical_length, donor.heads[head_index], options)) orelse return checkpoint.cloneHead(allocator, donor.heads[head_index]);
        if (options.stats) |stats| {
            stats.v18_eligible_heads += 1;
            stats.v18_attempted_heads += 1;
        }
        const callback = options.adaptive_v18_fn orelse return error.InvalidFitConfig;
        return callback(options.adaptive_v18_context, allocator, metadata, donor.heads[head_index], donor_index, head_index, batches, layout, .{ plan.old_rows, plan.prefix_rows, plan.target_rows, plan.block_rows }, &options, resident) catch |err| switch (err) {
            error.V18TrainingCacheUnsupported => {
                if (options.stats) |stats| {
                    stats.v18_fallback_heads += 1;
                    stats.v18_preserved_heads += 1;
                }
                var legacy = options;
                legacy.fit_strategy = .legacy;
                legacy.staged_operator_strategy = false;
                legacy.adaptive_v18_context = null;
                legacy.adaptive_v18_fn = null;
                legacy.adaptive_v18_numerical_policy = .legacy_dense_f64;
                return preparedHead(allocator, source, metadata, donor_index, head_index, batches, layout, legacy, resident, validation_gate);
            },
            else => return err,
        };
    }
    const replacement = if (donor.window == 0) attempt: {
        var attempt_options = options;
        while (true) {
            try options.fit.control.check();
            break :attempt compactHead(allocator, source, donor_index, head_index, batches, layout, attempt_options, resident, validation_gate) catch |err| {
                if (err != error.CompactionFitRejected and err != error.MassFitDidNotConverge) return err;
                try options.fit.control.check();
                if (options.stats) |stats| {
                    const metrics = if (stats.validating_storage) stats.rounded_fit else stats.last_fit;
                    stats.max_rejected_mass_error = @max(stats.max_rejected_mass_error, metrics.max_relative_mass_error);
                    stats.max_rejected_output_rmse = @max(stats.max_rejected_output_rmse, metrics.relative_output_rmse);
                }
                if (options.admission == .fixed and attempt_options.retention_divisor > (options.min_retention_divisor orelse options.retention_divisor)) {
                    attempt_options.retention_divisor -= 1;
                    if (options.stats) |stats| stats.retention_retries += 1;
                    continue;
                }
                if (options.rejected_head_policy != .preserve_original) return err;
                if (options.stats) |stats| {
                    stats.preserved_heads += 1;
                    if (err == error.MassFitDidNotConverge) stats.convergence_fallbacks += 1;
                }
                break :attempt null;
            };
        }
    } else null;
    return replacement orelse try checkpoint.cloneHead(allocator, donor.heads[head_index]);
}

const HeadTask = struct {
    ordinal: usize,
    score: usize,
    fn less(_: void, a: HeadTask, b: HeadTask) bool {
        return if (a.score == b.score) a.ordinal < b.ordinal else a.score > b.score;
    }
};
const HeadValidationGate = struct {
    maximum: usize = 2,
    active: std.atomic.Value(usize) = .init(0),
    fn enter(self: *@This(), control: solver.Control) !void {
        while (true) {
            try control.check();
            const count = self.active.load(.acquire);
            if (count < self.maximum and self.active.cmpxchgWeak(count, count + 1, .acq_rel, .acquire) == null) {
                errdefer self.leave();
                try control.check();
                return;
            }
            std.Thread.yield() catch {};
        }
    }
    fn leave(self: *@This()) void {
        const before = self.active.fetchSub(1, .acq_rel);
        std.debug.assert(before > 0 and before <= self.maximum);
    }
};

const ParallelControl = struct {
    validation_gate: HeadValidationGate = .{},
    mutex: std.atomic.Mutex = .unlocked,
    control: solver.Control,
    failure: ?anyerror = null,
    next_head: usize = 0,
    tasks: []const HeadTask,

    fn lock(self: *@This()) void {
        while (!self.mutex.tryLock()) std.atomic.spinLoopHint();
    }
    fn check(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.lock();
        defer self.mutex.unlock();
        if (self.failure) |err| return err;
        // Arbitrary caller callbacks are never invoked concurrently.
        self.control.check() catch |err| {
            self.failure = err;
            return err;
        };
    }
    fn fail(self: *@This(), err: anyerror) void {
        self.lock();
        defer self.mutex.unlock();
        if (self.failure == null) self.failure = err;
    }
    fn failHead(self: *@This(), err: anyerror, task: usize, donor: u32, head: usize, diagnostic: bool) void {
        self.lock();
        defer self.mutex.unlock();
        const origin = self.failure == null;
        if (origin) self.failure = err;
        if (diagnostic) std.debug.print("attention_match_parallel_failure stage={s} task={d} donor={d} head={d} error={s} first_error={s} diagnostic_only=true\n", .{ if (origin) "origin" else "propagated", task, donor, head, @errorName(err), if (self.failure) |first| @errorName(first) else "none" });
    }
    fn next(self: *@This()) ?usize {
        self.lock();
        defer self.mutex.unlock();
        if (self.failure != null or self.next_head == self.tasks.len) return null;
        const result = self.tasks[self.next_head].ordinal;
        self.next_head += 1;
        return result;
    }
};
const HeadWorker = struct {
    source: checkpoint.State,
    metadata: SourceMetadata,
    candidate: checkpoint.State,
    batches: []const ReferenceBatch,
    layout: Layout,
    options: Options,
    resident: usize,
    shared: *ParallelControl,
    stats: Stats = .{},

    fn run(self: *@This()) void {
        self.options.stats = &self.stats;
        self.options.fit.control = .{ .context = self.shared, .check_fn = ParallelControl.check };
        // A caller metrics pointer is not worker-owned. compactHead uses its
        // private Stats instead; never let the solver publish into shared data.
        self.options.fit.metrics = null;
        while (self.shared.next()) |task| {
            var first: usize = 0;
            for (self.source.donors, 0..) |donor, di| {
                if (donor.window != 0) continue;
                if (task >= first + donor.heads.len) {
                    first += donor.heads.len;
                    continue;
                }
                const hi = task - first;
                const result = preparedHead(std.heap.page_allocator, self.source, self.metadata, di, hi, self.batches, self.layout, self.options, self.resident, &self.shared.validation_gate) catch |err| {
                    self.shared.failHead(err, task, donor.layer, hi, self.options.adaptive_v18_train_probability_census);
                    return;
                };
                @constCast(self.candidate.donors[di].heads)[hi] = result;
                break;
            }
        }
    }
};
fn mergeStats(total: *Stats, part: Stats) void {
    total.donor_layer = part.donor_layer;
    total.head = part.head;
    total.chunk_start = part.chunk_start;
    total.last_fit = part.last_fit;
    total.rounded_fit = part.rounded_fit;
    total.validating_storage = part.validating_storage;
    total.validating_head = part.validating_head;
    if (part.fp64_validation_policy != .scalar_reference) total.fp64_validation_policy = part.fp64_validation_policy;
    if (part.geometry_chooser_policy != .legacy_full_scan) total.geometry_chooser_policy = part.geometry_chooser_policy;
    inline for (.{ "attempted_chunks", "completed_chunks", "validated_heads", "preserved_heads", "retention_retries", "convergence_fallbacks", "fp8_heads", "fp8_fallback_heads", "fp8_eligible_rows", "fp8_encoded_rows", "fp8_exact_rows", "fp64_validation_summary_calls", "fp64_validation_dgemm_calls", "incremental_gram_chooser_calls", "training_reduction_local_original_calls", "training_reduction_rounded_harm_calls", "training_reduction_tiles", "training_reduction_backend_calls", "rotated4_heads", "rotated4_fallback_heads", "rotated4_eligible_rows", "rotated4_compressed_rows", "rotated4_exact_rows", "v18_eligible_heads", "v18_attempted_heads", "v18_accepted_heads", "v18_preserved_heads", "v18_fallback_heads", "v18_budget_fallback_heads", "v18_train_primary_probe_heads", "v18_train_primary_probe_candidates", "v18_train_primary_probe_recomputed_heads", "v18_cache_ready_heads", "v18_planned_exact_heads", "v18_exact_f32_probability_heads", "v18_gram_heads", "v18_scored_reuse_heads", "v18_gram_backend_calls", "v18_bounded_heads", "v18_bounded_provider_calls", "v18_bounded_provider_rows", "v18_bounded_scoring_cache_prepares", "v18_bounded_scoring_selection_hits", "v18_bounded_scoring_fit_hits", "v18_bounded_score_accelerate_calls", "v18_bounded_score_accelerate_tiles", "v18_bounded_score_accelerate_dgemm_calls", "v18_bounded_score_cancellation_checks", "v18_bounded_range_cache_prepares", "v18_bounded_geometry_cache_requests", "v18_bounded_geometry_cache_hits", "v18_bounded_geometry_cache_loads", "v18_bounded_delta_heads", "v18_bounded_delta_tiles", "v18_bounded_delta_backend_calls", "v18_bounded_delta_fallbacks" }) |field|
        @field(total, field) += @field(part, field);
    inline for (.{ "v18_query_batch_rows", "v18_primary_budget_percent", "v18_planned_head_workspace_bytes", "v18_cache_phase_peak_bound_bytes", "v18_bounded_workspace_bound_bytes", "v18_bounded_delta_packed_bytes_peak", "v18_bounded_delta_staged_bytes_peak", "fp64_validation_max_batch_rows", "fp64_validation_workspace_peak_bytes", "training_reduction_workspace_peak_bytes" }) |field|
        @field(total, field) = @max(@field(total, field), @field(part, field));
    inline for (.{ "v18_cache_prepare_ns", "v18_selection_freeze_ns", "v18_joint_value_fit_ns", "v18_gram_construction_ns", "v18_chooser_ns", "v18_final_heldout_ns" }) |field|
        @field(total, field) +|= @field(part, field);
    inline for (.{ "max_nnls_sweeps", "max_rejected_mass_error", "max_rejected_output_rmse", "max_head_mass_error", "max_head_output_rmse" }) |field|
        @field(total, field) = @max(@field(total, field), @field(part, field));
}

test "parallel stats sum planned exact heads" {
    var total: Stats = .{ .v18_planned_exact_heads = 1 };
    mergeStats(&total, .{ .v18_planned_exact_heads = 2 });
    try std.testing.expectEqual(@as(usize, 3), total.v18_planned_exact_heads);
}
fn prepareHeadsParallel(candidate: checkpoint.State, source: checkpoint.State, metadata: SourceMetadata, batches: []const ReferenceBatch, layout: Layout, options: Options, source_bytes: usize, backends: []const solver.Backend) !void {
    // Reserve source + worst-case candidate once. Divide the remaining host
    // workspace among workers so individually admitted jobs cannot oversubscribe
    // the process. No worker reads another worker's partially built candidate.
    var head_count: usize = 0;
    for (source.donors) |donor| if (donor.window == 0) {
        head_count = try add(head_count, donor.heads.len);
    };
    const resident = try parallelResidentBytes(source_bytes, head_count);
    try budget(resident, options.max_transient_host_bytes);
    const per_worker = (options.max_transient_host_bytes - resident) / backends.len;
    const tasks = try std.heap.page_allocator.alloc(HeadTask, head_count);
    defer std.heap.page_allocator.free(tasks);
    var ordinal: usize = 0;
    for (source.donors, 0..) |donor, di| {
        if (donor.window == 0) {
            for (donor.heads, 0..) |head, hi| {
                var query_elements: usize = 0;
                for (batches) |batch| if (belongs(batch, source, layout, di, hi)) {
                    query_elements = try add(query_elements, batch.train.len);
                };
                tasks[ordinal] = .{ .ordinal = ordinal, .score = try mul(head.positions.len, query_elements) };
                ordinal += 1;
            }
        } else for (donor.heads, 0..) |head, hi| {
            try options.fit.control.check();
            @constCast(candidate.donors[di].heads)[hi] = try checkpoint.cloneHead(std.heap.page_allocator, head);
        }
    }
    std.mem.sort(HeadTask, tasks, {}, HeadTask.less);
    var shared: ParallelControl = .{ .control = options.fit.control, .tasks = tasks, .validation_gate = .{ .maximum = options.max_parallel_validations } };
    var jobs: [4]HeadWorker = undefined;
    for (jobs[0..backends.len], backends) |*job, backend| {
        var worker_options = options;
        worker_options.max_transient_host_bytes = resident + per_worker;
        worker_options.fit.backend = backend;
        job.* = .{ .source = source, .metadata = metadata, .candidate = candidate, .batches = batches, .layout = layout, .options = worker_options, .resident = resident, .shared = &shared };
    }
    {
        var threads: [3]std.Thread = undefined;
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |thread| thread.join();
        for (jobs[1..backends.len]) |*job| {
            threads[spawned] = std.Thread.spawn(.{}, HeadWorker.run, .{job}) catch |err| {
                shared.fail(err);
                break;
            };
            spawned += 1;
        }
        jobs[0].run();
    }
    if (options.stats) |stats| for (jobs[0..backends.len]) |job| mergeStats(stats, job.stats);
    if (shared.failure) |err| return err;
}

pub fn criticalCorePolicyValid(options: Options) bool {
    if (options.adaptive_v18_workspace_profile == .resident_native_hybrid_2g_v1 or
        options.adaptive_v18_workspace_profile == .resident_e4_32512_bounded_2g_v1 or
        options.adaptive_v18_workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1)
    {
        // The resident route has its own complete admission tuple. Offline
        // diagnostic observation flags are not serving authorization.
        _ = v18HeadWorkspaceBytesForOptions(options) catch return false;
        return !options.source_has_metadata and !options.component_ab_diagnostic and
            !options.adaptive_v18_train_primary_policy_comparison and
            !options.adaptive_v18_train_primary_policy_cross_validation and
            !options.adaptive_v18_rejected_detail_diagnostic and
            options.fit.max_relative_mass_error == 0.05 and
            options.fit.max_relative_output_rmse == 0.05;
    }
    if (options.adaptive_v18_critical_core_policy == .disabled) return !options.adaptive_v18_train_primary_policy_comparison and !options.adaptive_v18_train_primary_policy_cross_validation and options.adaptive_v18_postfit_policy != .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1;
    const calibrated = options.adaptive_v18_critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1 or
        options.adaptive_v18_critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    const diagnostic_incremental_reduction = calibrated and
        options.private_performance_policy == .owned_capture_summary_incremental_reduction_v1 and
        options.geometry_chooser_policy == .incremental_gram_v1 and
        options.adaptive_v18_geometry_route == .flat_gram_only and
        options.independent_validation_policy == .accelerate_fp64_summary_v1 and
        (options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 or options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_lawson_exact_rounded_max_v1) and
        !options.adaptive_v18_rejected_detail_diagnostic;
    const diagnostic_incremental_reduction_critical025 =
        options.adaptive_v18_critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
        options.private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
        options.adaptive_v18_primary_budget_percent == 91 and
        options.adaptive_v18_fallback_budget_percent == 95 and
        options.adaptive_v18_max_budget_fallbacks == 3 and
        options.geometry_chooser_policy == .incremental_gram_v1 and
        options.adaptive_v18_geometry_route == .flat_gram_only and
        options.independent_validation_policy == .accelerate_fp64_summary_v1 and
        options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 and
        options.adaptive_v18_train_probability_census and
        !options.adaptive_v18_rejected_detail_diagnostic;
    const diagnostic_surrogate_partition_minimax =
        options.adaptive_v18_critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
        options.private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
        options.adaptive_v18_primary_budget_percent == 91 and options.adaptive_v18_fallback_budget_percent == 95 and
        options.adaptive_v18_max_budget_fallbacks == 3 and options.geometry_chooser_policy == .incremental_gram_v1 and
        options.adaptive_v18_geometry_route == .flat_gram_only and options.independent_validation_policy == .accelerate_fp64_summary_v1 and
        options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 and
        options.adaptive_v18_train_probability_census and !options.adaptive_v18_rejected_detail_diagnostic and
        !options.adaptive_v18_train_primary_policy_comparison and !options.adaptive_v18_train_primary_policy_cross_validation and
        options.fit.max_workspace_bytes == 1024 * 1024 * 1024 and
        options.fit.max_relative_mass_error == 0.05 and options.fit.max_relative_output_rmse == 0.05;
    const train_primary_comparison = options.adaptive_v18_train_primary_policy_comparison and
        options.fit.max_workspace_bytes == 1024 * 1024 * 1024 and
        options.fit.max_relative_mass_error == 0.05 and options.fit.max_relative_output_rmse == 0.05 and
        options.adaptive_v18_critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
        options.private_performance_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
        options.adaptive_v18_primary_budget_percent == 91 and options.adaptive_v18_fallback_budget_percent == 95 and
        options.adaptive_v18_max_budget_fallbacks == 3 and options.geometry_chooser_policy == .incremental_gram_v1 and
        options.adaptive_v18_geometry_route == .flat_gram_only and options.independent_validation_policy == .accelerate_fp64_summary_v1 and
        options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 and
        options.adaptive_v18_train_probability_census and !options.adaptive_v18_rejected_detail_diagnostic;
    if (options.adaptive_v18_train_primary_policy_comparison and !train_primary_comparison) return false;
    if (options.adaptive_v18_train_primary_policy_cross_validation and !train_primary_comparison) return false;
    const diagnostic_legacy_lawson = calibrated and
        options.private_performance_policy == .legacy and
        options.geometry_chooser_policy == .legacy_full_scan and
        options.adaptive_v18_geometry_route == .flat_gram_only and
        options.independent_validation_policy == .scalar_reference and
        options.adaptive_v18_train_probability_census and
        options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_lawson_exact_rounded_max_v1;
    if (options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_lawson_exact_rounded_max_v1 and !diagnostic_incremental_reduction and !diagnostic_legacy_lawson) return false;
    if (options.adaptive_v18_postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 and !diagnostic_surrogate_partition_minimax) return false;
    const common = options.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and
        options.staged_operator_strategy and
        options.adaptive_v18_numerical_policy == .accepted_f32_gram_reuse_v1 and
        options.max_queries_per_consumer == 768 and
        options.validation == .whole_head and options.independent_fp64_publication and
        options.storage_precision == .f16 and !options.component_ab_diagnostic and
        (options.private_performance_policy == .legacy or diagnostic_incremental_reduction or diagnostic_incremental_reduction_critical025 or diagnostic_surrogate_partition_minimax or train_primary_comparison);
    if (!common) return false;
    return switch (options.adaptive_v18_critical_core_policy) {
        .disabled => unreachable,
        .train_probability_gte_0_025_no_top1_v1 => true,
        .train_probability_budget_calibrated_floor_0_0025_p91_v1, .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => options.adaptive_v18_geometry_route == .flat_gram_only and
            options.adaptive_v18_primary_budget_percent == 91 and options.adaptive_v18_fallback_budget_percent == 95 and options.adaptive_v18_max_budget_fallbacks == 3,
    };
}

test "TRAIN surrogate partition minimax admits only sealed critical025 tuple" {
    var options: Options = .{
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1,
        .adaptive_v18_train_probability_census = true,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .storage_precision = .f16,
        .fit = .{ .target_rows = 1, .max_workspace_bytes = 1024 * 1024 * 1024, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 },
    };
    try std.testing.expect(criticalCorePolicyValid(options));
    const saved = options;
    options.adaptive_v18_critical_core_policy = .disabled;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options = saved;
    options.fit.max_workspace_bytes -= 1;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options.fit.max_workspace_bytes += 1;
    options.adaptive_v18_train_primary_policy_comparison = true;
    try std.testing.expect(!criticalCorePolicyValid(options));
}

test "TRAIN critical retention requires staged complete training and whole-head publication" {
    try std.testing.expect(criticalCorePolicyValid(.{}));
    const allowed: Options = .{
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .storage_precision = .f16,
    };
    try std.testing.expect(criticalCorePolicyValid(allowed));
    var wrong = allowed;
    wrong.fit_strategy = .legacy;
    try std.testing.expect(!criticalCorePolicyValid(wrong));
    wrong = allowed;
    wrong.staged_operator_strategy = false;
    try std.testing.expect(!criticalCorePolicyValid(wrong));
    wrong = allowed;
    wrong.max_queries_per_consumer = 256;
    try std.testing.expect(!criticalCorePolicyValid(wrong));
    wrong = allowed;
    wrong.independent_fp64_publication = false;
    try std.testing.expect(!criticalCorePolicyValid(wrong));
    wrong = allowed;
    wrong.adaptive_v18_numerical_policy = .legacy_dense_f64;
    try std.testing.expect(!criticalCorePolicyValid(wrong));
    wrong = allowed;
    wrong.private_performance_policy = .owned_capture_summary_v1;
    try std.testing.expect(!criticalCorePolicyValid(wrong));

    var calibrated = allowed;
    calibrated.adaptive_v18_critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_v1;
    calibrated.adaptive_v18_primary_budget_percent = 91;
    calibrated.adaptive_v18_fallback_budget_percent = 95;
    calibrated.adaptive_v18_max_budget_fallbacks = 3;
    try std.testing.expect(criticalCorePolicyValid(calibrated));
    calibrated.adaptive_v18_primary_budget_percent = 90;
    try std.testing.expect(!criticalCorePolicyValid(calibrated));
    calibrated.adaptive_v18_primary_budget_percent = 91;
    calibrated.adaptive_v18_geometry_route = .bounded_page_over_96;
    try std.testing.expect(!criticalCorePolicyValid(calibrated));
    calibrated.adaptive_v18_geometry_route = .flat_gram_only;
    calibrated.adaptive_v18_critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    try std.testing.expect(criticalCorePolicyValid(calibrated));
}

test "TRAIN critical retention admits only the sealed offline incremental reduction comparison" {
    var comparison: Options = .{
        .adaptive_v18_critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .storage_precision = .f16,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
    };
    try std.testing.expect(criticalCorePolicyValid(comparison));
    comparison.adaptive_v18_rejected_detail_diagnostic = true;
    try std.testing.expect(!criticalCorePolicyValid(comparison));
    comparison.adaptive_v18_rejected_detail_diagnostic = false;
    comparison.adaptive_v18_postfit_policy = .disabled;
    try std.testing.expect(!criticalCorePolicyValid(comparison));
    comparison.adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    comparison.private_performance_policy = .owned_capture_summary_incremental_v1;
    try std.testing.expect(!criticalCorePolicyValid(comparison));
    comparison.private_performance_policy = .owned_capture_summary_incremental_reduction_v1;
    comparison.adaptive_v18_postfit_policy = .train_global_residual_range_mass_lawson_exact_rounded_max_v1;
    try std.testing.expect(criticalCorePolicyValid(comparison));
    comparison.private_performance_policy = .legacy;
    comparison.geometry_chooser_policy = .legacy_full_scan;
    comparison.independent_validation_policy = .scalar_reference;
    comparison.adaptive_v18_train_probability_census = true;
    try std.testing.expect(criticalCorePolicyValid(comparison));
    comparison.adaptive_v18_train_probability_census = false;
    try std.testing.expect(!criticalCorePolicyValid(comparison));
}

fn prepareWithBackends(allocator: std.mem.Allocator, source: checkpoint.State, batches: []const ReferenceBatch, layout: Layout, requested_options: Options, backends: []const solver.Backend) !checkpoint.Owned {
    var options = requested_options;
    if (options.stats) |stats| stats.* = .{};
    try options.fit.control.check();
    if (!criticalCorePolicyValid(options)) return error.InvalidFitConfig;
    if (!cumulativePageCachePolicyValid(options)) return error.InvalidFitConfig;
    if (!trainingPositionPolicyValid(options)) return error.InvalidFitConfig;
    _ = try v18HeadWorkspaceBytesForOptions(options);
    if (options.adaptive_v18_workspace_profile != .standard)
        options.fit.workspace_admission = .resident_native_hybrid_2g_v1;
    if ((options.fit_strategy == .legacy and (options.staged_operator_strategy or options.adaptive_v18_context != null or options.adaptive_v18_fn != null or options.adaptive_v18_wide_cache_authority != null or options.adaptive_v18_numerical_policy != .legacy_dense_f64 or options.v18_reference_summary_cache_policy != .disabled or options.adaptive_v18_geometry_route != .flat_gram_only or options.adaptive_v18_bounded_tile_queries != 0 or options.adaptive_v18_bounded_builder_transient_bytes != 0 or options.adaptive_v18_bounded_delta_accelerated or options.adaptive_v18_bounded_tsqr_value_backend or options.adaptive_v18_primary_budget_percent != 0 or options.adaptive_v18_fallback_budget_percent != 0 or options.adaptive_v18_max_budget_fallbacks != 0)) or
        (options.fit_strategy == .identity_chunk_coefficients_adaptive_v18 and
            (!options.staged_operator_strategy or options.admission != .cumulative_ladder_4_3_2 or options.chunk_rows != 256 or options.adaptive_v18_fn == null or
                options.storage_precision != .f16 or !options.independent_fp64_publication)) or
        (options.independent_validation_policy != .scalar_reference and
            (options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.independent_fp64_publication or options.validation != .whole_head)))
        return error.InvalidFitConfig;
    if (options.adaptive_v18_wide_cache_authority != null and
        (options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.staged_operator_strategy or
            options.adaptive_v18_workspace_profile == .standard or
            (options.adaptive_v18_workspace_profile == .resident_native_hybrid_2g_v1 and options.adaptive_v18_geometry_route != .flat_gram_only) or
            (options.adaptive_v18_workspace_profile == .resident_e4_32512_bounded_2g_v1 and options.adaptive_v18_geometry_route != .bounded_page_over_96) or
            (options.adaptive_v18_workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1 and options.adaptive_v18_geometry_route != .bounded_page_over_96) or
            options.independent_validation_policy != .accelerate_fp64_summary_v1 or options.private_performance_policy == .legacy or
            options.adaptive_v18_numerical_policy != .accepted_f32_gram_reuse_v1))
        return error.InvalidFitConfig;
    if (options.v18_reference_summary_cache_policy != .disabled and
        (options.independent_validation_policy != .accelerate_fp64_summary_v1 or options.adaptive_v18_geometry_route != .flat_gram_only)) return error.InvalidFitConfig;
    const private_perf = options.private_performance_policy != .legacy;
    if (!privatePerformancePolicyValid(options)) return error.InvalidFitConfig;
    if ((options.component_ab_diagnostic and
        (!builtin.is_test or options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.staged_operator_strategy or
            options.validation != .whole_head or options.adaptive_v18_geometry_route != .flat_gram_only or
            options.adaptive_v18_bounded_delta_accelerated)) or
        (!options.component_ab_diagnostic and !private_perf and
            (options.geometry_chooser_policy != .legacy_full_scan or options.independent_validation_policy != .scalar_reference)))
        return error.InvalidFitConfig;
    const budget_tuple_off = options.adaptive_v18_primary_budget_percent == 0 and options.adaptive_v18_fallback_budget_percent == 0 and options.adaptive_v18_max_budget_fallbacks == 0;
    const budget_tuple_private = options.adaptive_v18_primary_budget_percent >= 50 and options.adaptive_v18_primary_budget_percent < 95 and options.adaptive_v18_fallback_budget_percent == 95 and options.adaptive_v18_max_budget_fallbacks == 3;
    const budget_tuple_e2_formal = options.adaptive_v18_workspace_profile == .resident_e2_formal_2g_p60_no_fallback_v1 and
        options.adaptive_v18_primary_budget_percent == 60 and options.adaptive_v18_fallback_budget_percent == 0 and options.adaptive_v18_max_budget_fallbacks == 0;
    if ((!budget_tuple_off and !budget_tuple_private and !budget_tuple_e2_formal) or
        ((budget_tuple_private or budget_tuple_e2_formal) and options.adaptive_v18_context == null)) return error.InvalidFitConfig;
    if (options.component_ab_diagnostic and !budget_tuple_off) return error.InvalidFitConfig;
    if (options.adaptive_v18_geometry_route == .bounded_page_over_96 and
        (options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or options.adaptive_v18_numerical_policy != .accepted_f32_gram_reuse_v1 or
            options.adaptive_v18_bounded_tile_queries != @as(usize, if (options.adaptive_v18_bounded_delta_accelerated) 64 else 8) or options.adaptive_v18_bounded_builder_transient_bytes != 256 * 1024 * 1024))
        return error.InvalidFitConfig;
    if (options.adaptive_v18_bounded_tsqr_value_backend and options.adaptive_v18_geometry_route != .bounded_page_over_96) return error.InvalidFitConfig;
    try source.validateWithControl(options.checkpoint_limits, options.fit.control);
    if (options.max_parallel_validations == 0 or options.max_parallel_validations > 4) return error.InvalidFitConfig;
    if ((options.fp8_checkpoint and options.rotated4_checkpoint) or ((options.fp8_checkpoint or options.rotated4_checkpoint) and (options.storage_precision != .f16 or options.validation != .whole_head))) return error.InvalidFitConfig;
    if (options.fp8_checkpoint) for (source.donors) |donor| for (donor.heads) |head| {
        if (head.compressed.spans.len != 0) return error.InvalidFitConfig;
    };
    if (options.min_retention_divisor) |minimum| {
        if (minimum < 2 or minimum > options.retention_divisor or options.validation != .whole_head) return error.InvalidFitConfig;
    }
    if (options.admission == .cumulative_ladder_4_3_2 and
        (options.validation != .whole_head or options.storage_precision != .f16 or
            !options.independent_fp64_publication or options.min_global_savings_ratio == null or
            options.retention_divisor != 4 or options.min_retention_divisor != 2))
        return error.InvalidFitConfig;
    if (options.chunk_rows < 2 or options.retention_divisor < 2 or options.retention_divisor > 4 or options.max_queries_per_consumer == 0)
        return error.InvalidFitConfig;
    if (options.min_global_savings_ratio) |minimum| {
        if (!std.math.isFinite(minimum) or minimum < 1) return error.InvalidFitConfig;
    }
    try validateQueries(source, batches, layout, options);
    const source_metadata = try sourceMetadataFromState(source, options.checkpoint_limits, options.fit.control);
    const source_bytes = try source.residentBytes();
    // Admit a conservative source + full-size candidate bound before allocating.
    // The actual candidate is assembled head-by-head; no full clone/encoding.
    if (source_bytes > options.checkpoint_limits.max_state_bytes) return error.KvCheckpointLimit;
    try budget(try mul(source_bytes, 2), options.max_transient_host_bytes);
    var candidate = try checkpoint.cloneMetadata(allocator, source);
    errdefer candidate.deinit();
    if (backends.len != 0) {
        try prepareHeadsParallel(candidate.state, source, source_metadata, batches, layout, options, source_bytes, backends);
    } else for (source.donors, 0..) |donor, donor_index| {
        for (donor.heads, 0..) |_, head_index| {
            const resident = try add(source_bytes, try candidate.state.residentBytes());
            @constCast(candidate.state.donors[donor_index].heads)[head_index] = try preparedHead(allocator, source, source_metadata, donor_index, head_index, batches, layout, options, resident, null);
        }
    }
    try candidate.state.validateWithControl(options.checkpoint_limits, options.fit.control);
    const source_global = try globalStorageBytes(source, options.storage_precision, options.source_has_metadata);
    const candidate_global = try globalStorageBytes(candidate.state, options.storage_precision, true);
    const fitted_global = if (options.rotated4_checkpoint)
        try fittedGlobalStorageBytes(source, candidate.state, options.storage_precision)
    else
        candidate_global;
    const ratio = if (candidate_global == 0) @as(f64, 1) else @as(f64, @floatFromInt(source_global)) / @as(f64, @floatFromInt(candidate_global));
    const fitted_ratio = if (fitted_global == 0) @as(f64, 1) else @as(f64, @floatFromInt(source_global)) / @as(f64, @floatFromInt(fitted_global));
    if (options.stats) |stats| {
        stats.source_global_bytes = source_global;
        stats.candidate_global_bytes = candidate_global;
        stats.fitted_global_bytes = fitted_global;
        stats.fitted_global_savings_ratio = fitted_ratio;
        stats.actual_global_savings_ratio = ratio;
    }
    // Diagnostic callers may prepare an identity state without a saving gate.
    // An explicit minimum is a strict acceptance condition, never a hint.
    if (options.min_global_savings_ratio) |minimum| {
        // Preserve the FP16 policy's exact-state fallback. A smaller codec is
        // not evidence that a previously unpublished fit preserves answers.
        if (fitted_ratio < minimum or ratio < minimum) return error.CompactionSavingsInsufficient;
    }
    try options.fit.control.check();
    return candidate;
}

fn allocateStreamingCandidate(allocator: std.mem.Allocator, source: SourceMetadata) !checkpoint.Owned {
    const tokens = try allocator.dupe(u32, source.tokens);
    errdefer allocator.free(tokens);
    const logits = try allocator.dupe(f32, source.logits);
    errdefer allocator.free(logits);
    const layer_donors = try allocator.dupe(u32, source.layer_donors);
    errdefer allocator.free(layer_donors);
    const donors = try allocator.alloc(checkpoint.Donor, source.donors.len);
    @memset(donors, .{});
    var result: checkpoint.Owned = .{ .allocator = allocator, .state = .{
        .identity = source.identity,
        .logical_length = source.logical_length,
        .tokens = tokens,
        .logits = logits,
        .layer_donors = layer_donors,
        .donors = donors,
    } };
    errdefer result.deinit();
    for (source.donors, donors) |descriptor, *donor| {
        const heads = try allocator.alloc(checkpoint.Head, descriptor.head_count);
        @memset(heads, .{});
        donor.* = .{ .layer = descriptor.layer, .window = descriptor.window, .heads = heads };
    }
    return result;
}

fn streamingHeadBytes(head: checkpoint.Head) !usize {
    var bytes = try mul(head.positions.len, 2 * @sizeOf(u64) + @sizeOf(f32));
    bytes = try add(bytes, try mul(head.keys.len, 2 * @sizeOf(f32)));
    return add(bytes, head.compressed.bytes());
}

pub const StreamingDeviceSink = struct {
    allocator: std.mem.Allocator,
    context: ?*anyopaque,
    upload_fn: *const fn (?*anyopaque, usize, checkpoint.Head, solver.Control) anyerror!void,
    candidate_device_max_bytes: usize,
    retained_query_bytes: usize,
    maximum_current_head_bytes: usize,
};

pub const StreamingDeviceEvidence = struct {
    candidate_global_bytes: usize,
    fitted_global_savings_ratio: f64,
    uploaded_heads: usize,
};

fn recordStreamingAdaptiveAttempt(options: *const Options) void {
    if (options.stats) |stats| {
        stats.v18_eligible_heads += 1;
        stats.v18_attempted_heads += 1;
    }
}

fn admitStreamingDevicePeak(source_live: usize, candidate_device: usize, queries: usize, current_head: usize, combined_cap: usize, host_cap: usize) !void {
    const host_peak = try add(source_live, try add(queries, current_head));
    if (host_peak > host_cap) return error.CompactionWorkspaceLimit;
    if (try add(host_peak, candidate_device) > combined_cap) return error.CompactionWorkspaceLimit;
}

fn destroyPreparedHead(allocator: std.mem.Allocator, head: checkpoint.Head) void {
    allocator.free(head.positions);
    allocator.free(head.coverage_ends);
    allocator.free(head.keys);
    allocator.free(head.values);
    allocator.free(head.bias);
    var compressed = head.compressed;
    compressed.deinit(allocator);
}

fn stagedHeadStorageBytes(head: checkpoint.Head, precision: @FieldType(Options, "storage_precision")) !usize {
    const exact_elements = try mul(head.positions.len - head.compressed.rows(), head.dim);
    return add(try add(try mul(exact_elements, if (precision == .f16) 4 else 8), head.compressed.bytes()), try mul(head.positions.len, 20));
}

/// Fits and validates one authenticated head at a time, then transfers it to
/// an unpublished device candidate. No complete host candidate is retained.
pub fn prepareFromHeadSourceIntoDevice(source: *StreamingHeadSource, batches: []const ReferenceBatch, layout: Layout, requested_options: Options, sink: StreamingDeviceSink) !StreamingDeviceEvidence {
    var options = requested_options;
    if (options.stats) |stats| stats.* = .{};
    try options.fit.control.check();
    try validateStreamingMetadata(source.metadata, options.checkpoint_limits, options.fit.control);
    if (!std.mem.eql(u8, &source.metadata.structure_sha256, &source.expected_structure_sha256) or !std.mem.eql(u8, &source.metadata.state_sha256, &source.expected_state_sha256) or !std.mem.eql(u8, &sourceMetadataStructureDigest(source.metadata), &source.expected_structure_sha256)) return error.InvalidTrainingAugmentationBinding;
    try validateStreamingQueries(source.metadata, batches, layout, options);
    if (!criticalCorePolicyValid(options) or !streamingPrivatePerformancePolicyValid(options) or !trainingPositionPolicyValid(options) or !cumulativePageCachePolicyValid(options) or options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.staged_operator_strategy or options.adaptive_v18_fn == null or options.adaptive_v18_geometry_route != .bounded_page_over_96 or !options.adaptive_v18_bounded_tsqr_value_backend or options.storage_precision != .f16 or (options.rotated4_checkpoint and options.fp8_checkpoint) or !options.independent_fp64_publication or options.validation != .whole_head or options.max_parallel_validations != 1 or source.metadata.source_global_storage_bytes == 0 or sink.candidate_device_max_bytes == 0 or sink.maximum_current_head_bytes == 0) return error.InvalidFitConfig;
    try admitStreamingDevicePeak(source.external_live_bytes, sink.candidate_device_max_bytes, sink.retained_query_bytes, sink.maximum_current_head_bytes, options.fit.max_workspace_bytes, options.max_transient_host_bytes);
    options.fit.workspace_admission = .resident_native_hybrid_2g_v1;
    var candidate_global: usize = 0;
    var uploaded: usize = 0;
    for (source.metadata.donors, 0..) |descriptor, donor_index| for (0..descriptor.head_count) |head_index| {
        try options.fit.control.check();
        var lease = try source.open(donor_index, head_index, options.fit.control);
        var lease_live = true;
        defer if (lease_live) lease.deinit();
        try checkpoint.validateHeadForSource(lease.head, source.metadata.logical_length, descriptor.window, options.checkpoint_limits, options.fit.control);
        if (lease.head.dim != descriptor.head_dim or try streamingHeadBytes(lease.head) > sink.maximum_current_head_bytes) return error.CompactionWorkspaceLimit;
        const resident = try add(source.external_live_bytes, try add(sink.retained_query_bytes, try streamingHeadBytes(lease.head)));
        try budget(resident, options.max_transient_host_bytes);
        const prepared = if (descriptor.window != 0)
            try checkpoint.cloneHead(sink.allocator, lease.head)
        else blk: {
            const plan = (try headPlan(source.metadata.logical_length, lease.head, options)) orelse return error.InvalidFitConfig;
            recordStreamingAdaptiveAttempt(&options);
            const callback = options.adaptive_v18_fn orelse unreachable;
            break :blk callback(options.adaptive_v18_context, sink.allocator, source.metadata, lease.head, donor_index, head_index, batches, layout, .{ plan.old_rows, plan.prefix_rows, plan.target_rows, plan.block_rows }, &options, resident) catch |err| switch (err) {
                error.V18TrainingCacheUnsupported => if (options.rejected_head_policy == .preserve_original) preserved: {
                    if (options.stats) |stats| {
                        stats.v18_fallback_heads += 1;
                        stats.v18_preserved_heads += 1;
                    }
                    break :preserved try checkpoint.cloneHead(sink.allocator, lease.head);
                } else return err,
                else => return err,
            };
        };
        var prepared_live = true;
        defer if (prepared_live) destroyPreparedHead(sink.allocator, prepared);
        try checkpoint.validateHeadForSource(prepared, source.metadata.logical_length, descriptor.window, options.checkpoint_limits, options.fit.control);
        const storage_bytes = try stagedHeadStorageBytes(prepared, options.storage_precision);
        if (descriptor.window == 0) candidate_global = try add(candidate_global, storage_bytes);
        try sink.upload_fn(sink.context, uploaded, prepared, options.fit.control);
        destroyPreparedHead(sink.allocator, prepared);
        prepared_live = false;
        uploaded += 1;
        lease.deinit();
        lease_live = false;
    };
    try source.verify_no_live_fn(source.context);
    const ratio = if (candidate_global == 0) @as(f64, 1) else @as(f64, @floatFromInt(source.metadata.source_global_storage_bytes)) / @as(f64, @floatFromInt(candidate_global));
    if (options.stats) |stats| {
        stats.source_global_bytes = source.metadata.source_global_storage_bytes;
        stats.candidate_global_bytes = candidate_global;
        stats.fitted_global_bytes = candidate_global;
        stats.fitted_global_savings_ratio = ratio;
        stats.actual_global_savings_ratio = ratio;
    }
    if (options.min_global_savings_ratio) |minimum| if (ratio < minimum) return error.CompactionSavingsInsufficient;
    try options.fit.control.check();
    return .{ .candidate_global_bytes = candidate_global, .fitted_global_savings_ratio = ratio, .uploaded_heads = uploaded };
}

test "streaming device telemetry counts only actual eligible callback attempts" {
    var stats: Stats = .{};
    var options: Options = .{ .stats = &stats };
    recordStreamingAdaptiveAttempt(&options);
    recordStreamingAdaptiveAttempt(&options);
    try std.testing.expectEqual(@as(usize, 2), stats.v18_eligible_heads);
    try std.testing.expectEqual(@as(usize, 2), stats.v18_attempted_heads);
    options.stats = null;
    recordStreamingAdaptiveAttempt(&options);
    try std.testing.expectEqual(@as(usize, 2), stats.v18_eligible_heads);
    try std.testing.expectEqual(@as(usize, 2), stats.v18_attempted_heads);
}

test "streaming device peak admits exact host and combined boundaries before source open" {
    try admitStreamingDevicePeak(100, 400, 200, 300, 1000, 600);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitStreamingDevicePeak(100, 401, 200, 300, 1000, 600));
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitStreamingDevicePeak(101, 399, 200, 300, 1000, 600));
}

/// Incrementally assembles a complete candidate from an authenticated source
/// that owns at most one leased head. The sealed 32K route is adaptive-only;
/// unsupported adaptive preparation is an error, never a legacy/full-state
/// fallback or an unchanged-success claim.
pub fn prepareFromHeadSource(allocator: std.mem.Allocator, source: *StreamingHeadSource, batches: []const ReferenceBatch, layout: Layout, requested_options: Options) !checkpoint.Owned {
    var options = requested_options;
    if (options.stats) |stats| stats.* = .{};
    try options.fit.control.check();
    try validateStreamingMetadata(source.metadata, options.checkpoint_limits, options.fit.control);
    if (!std.mem.eql(u8, &source.metadata.structure_sha256, &source.expected_structure_sha256) or
        !std.mem.eql(u8, &source.metadata.state_sha256, &source.expected_state_sha256))
        return error.InvalidTrainingAugmentationBinding;
    try validateStreamingQueries(source.metadata, batches, layout, options);
    if (!criticalCorePolicyValid(options) or !streamingPrivatePerformancePolicyValid(options) or
        !trainingPositionPolicyValid(options) or !cumulativePageCachePolicyValid(options) or
        options.fit_strategy != .identity_chunk_coefficients_adaptive_v18 or !options.staged_operator_strategy or
        options.adaptive_v18_fn == null or options.adaptive_v18_geometry_route != .bounded_page_over_96 or
        !options.adaptive_v18_bounded_tsqr_value_backend or options.storage_precision != .f16 or
        (options.rotated4_checkpoint and options.fp8_checkpoint) or !options.independent_fp64_publication or
        options.validation != .whole_head or options.max_parallel_validations != 1 or
        source.metadata.source_global_storage_bytes == 0 or source.external_live_bytes > options.max_transient_host_bytes)
        return error.InvalidFitConfig;
    options.fit.workspace_admission = .resident_native_hybrid_2g_v1;
    var candidate = try allocateStreamingCandidate(allocator, source.metadata);
    errdefer candidate.deinit();
    for (source.metadata.donors, 0..) |descriptor, donor_index| {
        for (0..descriptor.head_count) |head_index| {
            try options.fit.control.check();
            var lease = try source.open(donor_index, head_index, options.fit.control);
            var lease_live = true;
            defer if (lease_live) lease.deinit();
            try checkpoint.validateHeadForSource(lease.head, source.metadata.logical_length, descriptor.window, options.checkpoint_limits, options.fit.control);
            if (lease.head.dim != descriptor.head_dim) return error.InvalidKvCheckpoint;
            const resident = try add(source.external_live_bytes, try add(try candidate.state.residentBytes(), try streamingHeadBytes(lease.head)));
            try budget(resident, options.max_transient_host_bytes);
            const prepared = if (descriptor.window != 0)
                try checkpoint.cloneHead(allocator, lease.head)
            else blk: {
                const plan = (try headPlan(source.metadata.logical_length, lease.head, options)) orelse return error.InvalidFitConfig;
                const callback = options.adaptive_v18_fn orelse unreachable;
                break :blk callback(options.adaptive_v18_context, allocator, source.metadata, lease.head, donor_index, head_index, batches, layout, .{ plan.old_rows, plan.prefix_rows, plan.target_rows, plan.block_rows }, &options, resident) catch |err| switch (err) {
                    error.V18TrainingCacheUnsupported => if (options.rejected_head_policy == .preserve_original) preserved: {
                        if (options.stats) |stats| {
                            stats.v18_fallback_heads += 1;
                            stats.v18_preserved_heads += 1;
                        }
                        break :preserved try checkpoint.cloneHead(allocator, lease.head);
                    } else return err,
                    else => return err,
                };
            };
            lease.deinit();
            lease_live = false;
            @constCast(candidate.state.donors[donor_index].heads)[head_index] = prepared;
        }
    }
    try source.verify_no_live_fn(source.context);
    try candidate.state.validateWithControl(options.checkpoint_limits, options.fit.control);
    const candidate_global = try globalStorageBytes(candidate.state, options.storage_precision, true);
    const ratio = if (candidate_global == 0) @as(f64, 1) else @as(f64, @floatFromInt(source.metadata.source_global_storage_bytes)) / @as(f64, @floatFromInt(candidate_global));
    if (options.stats) |stats| {
        stats.source_global_bytes = source.metadata.source_global_storage_bytes;
        stats.candidate_global_bytes = candidate_global;
        stats.fitted_global_bytes = candidate_global;
        stats.fitted_global_savings_ratio = ratio;
        stats.actual_global_savings_ratio = ratio;
    }
    if (options.min_global_savings_ratio) |minimum| if (ratio < minimum) return error.CompactionSavingsInsufficient;
    try options.fit.control.check();
    return candidate;
}

const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
const global: checkpoint.Head = .{ .dim = 1, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 } };
const local: checkpoint.Head = .{ .dim = 1, .positions = &.{ 2, 3 }, .coverage_ends = &.{ 3, 4 }, .keys = &.{ 1, 2 }, .values = &.{ 6, 8 }, .bias = &.{ 0, 0 } };
const state: checkpoint.State = .{ .identity = identity, .logical_length = 4, .tokens = &.{ 0, 1, 0, 1 }, .logits = &.{ 0.2, 0.8 }, .layer_donors = &.{ 0, 1, 0 }, .donors = &.{ .{ .layer = 0, .heads = &.{ global, global } }, .{ .layer = 1, .window = 2, .heads = &.{ local, local } } } };

test "source metadata binds full payload and preserves existing query mapping" {
    const first = try sourceMetadataFromState(state, .{}, .{});
    try std.testing.expectEqual(state.logical_length, first.logical_length);
    try std.testing.expectEqual(try state.residentBytes(), first.source_census_physical_bytes);
    try std.testing.expect(first.belongs(layout_fixture, .{ .consumer_layer = 0, .query_head = 3, .train = &.{}, .heldout = &.{}, .train_positions = &.{}, .heldout_positions = &.{} }, 0, 1));
    var mutated_keys = [_]f32{ 0, 0, 0, 1 };
    var mutated_head = global;
    mutated_head.keys = &mutated_keys;
    const mutated_state: checkpoint.State = .{ .identity = identity, .logical_length = 4, .tokens = &.{ 0, 1, 0, 1 }, .logits = &.{ 0.2, 0.8 }, .layer_donors = &.{0}, .donors = &.{.{ .layer = 0, .heads = &.{mutated_head} }} };
    const second = try sourceMetadataFromState(mutated_state, .{}, .{});
    try std.testing.expect(!std.mem.eql(u8, &first.state_sha256, &second.state_sha256));
}

test "streaming candidate owns fitted and local heads and closes every singleton lease" {
    const rows: usize = 2304;
    const positions = try std.testing.allocator.alloc(u64, rows);
    defer std.testing.allocator.free(positions);
    const coverage = try std.testing.allocator.alloc(u64, rows);
    defer std.testing.allocator.free(coverage);
    const keys = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(keys);
    const values = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(values);
    const bias = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(bias);
    for (positions, coverage, keys, values, bias, 0..) |*p, *c, *k, *v, *b, index| {
        p.* = index;
        c.* = index + 1;
        k.* = @floatFromInt(index % 7);
        v.* = @floatFromInt(index % 11);
        b.* = 0;
    }
    const global_head: checkpoint.Head = .{ .dim = 1, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias };
    const local_head: checkpoint.Head = .{ .dim = 1, .positions = positions[rows - 4 ..], .coverage_ends = coverage[rows - 4 ..], .keys = keys[rows - 4 ..], .values = values[rows - 4 ..], .bias = bias[rows - 4 ..] };
    const Loader = struct {
        heads: [2]checkpoint.Head,
        open_count: usize = 0,
        close_count: usize = 0,
        live: bool = false,
        fail_fit: bool = false,
        unsupported_fit: bool = false,
        fn open(raw: ?*anyopaque, donor: usize, head_index: usize, _: solver.Control) !StreamingHeadLease {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.live or head_index != 0 or donor >= self.heads.len) return error.TestUnexpectedResult;
            self.live = true;
            self.open_count += 1;
            return .{ .context = self, .close_fn = close, .donor_index = donor, .head_index = head_index, .head = self.heads[donor] };
        }
        fn close(raw: ?*anyopaque, _: *StreamingHeadLease) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            std.debug.assert(self.live);
            self.live = false;
            self.close_count += 1;
        }
        fn verify(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.live) return error.IncompleteContextAllocationScope;
        }
        fn fit(raw: ?*anyopaque, allocator: std.mem.Allocator, _: SourceMetadata, head: checkpoint.Head, _: usize, _: usize, _: []const ReferenceBatch, _: Layout, _: [4]usize, _: *const Options, _: usize) !checkpoint.Head {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.fail_fit) return error.InjectedFailure;
            if (self.unsupported_fit) return error.V18TrainingCacheUnsupported;
            return checkpoint.cloneHead(allocator, head);
        }
    };
    var loader: Loader = .{ .heads = .{ global_head, local_head } };
    const tokens = try std.testing.allocator.alloc(u32, rows);
    defer std.testing.allocator.free(tokens);
    @memset(tokens, 0);
    var metadata: SourceMetadata = .{ .identity = identity, .logical_length = rows, .tokens = tokens, .logits = &.{0}, .layer_donors = &.{ 0, 1 }, .donors = &.{ .{ .layer = 0, .window = 0, .head_count = 1, .head_dim = 1, .rows = rows, .physical_bytes = 32768 }, .{ .layer = 1, .window = 4, .head_count = 1, .head_dim = 1, .rows = 4, .physical_bytes = 4096 } }, .structure_sha256 = undefined, .state_sha256 = @splat(1), .source_census_physical_bytes = 36864, .source_global_storage_bytes = 1 << 20 };
    metadata.structure_sha256 = sourceMetadataStructureDigest(metadata);
    var source: StreamingHeadSource = .{
        .metadata = metadata,
        .expected_structure_sha256 = metadata.structure_sha256,
        .expected_state_sha256 = metadata.state_sha256,
        .context = &loader,
        .open_fn = Loader.open,
        .verify_no_live_fn = Loader.verify,
        .external_live_bytes = 0,
    };
    var stats: Stats = .{};
    var options: Options = .{
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .admission = .cumulative_ladder_4_3_2,
        .adaptive_v18_fn = Loader.fit,
        .adaptive_v18_context = &loader,
        .adaptive_v18_workspace_profile = .resident_e4_32512_bounded_2g_v1,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .bounded_page_over_96,
        .adaptive_v18_bounded_tile_queries = 64,
        .adaptive_v18_bounded_builder_transient_bytes = 256 * 1024 * 1024,
        .adaptive_v18_bounded_delta_accelerated = true,
        .adaptive_v18_bounded_tsqr_value_backend = true,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .prefix_rows = 128,
        .recent_rows = 2048,
        .chunk_rows = 256,
        .retention_divisor = 4,
        .min_retention_divisor = 2,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .storage_precision = .f16,
        .independent_fp64_publication = true,
        .max_parallel_validations = 1,
        .max_transient_host_bytes = 2 * 1024 * 1024 * 1024,
        .min_global_savings_ratio = 1.5,
        .rejected_head_policy = .preserve_original,
        .stats = &stats,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .fit = .{ .target_rows = 1, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_workspace_bytes = 2 * 1024 * 1024 * 1024 },
    };
    const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &.{0}, .heldout = &.{0}, .train_positions = &.{rows}, .heldout_positions = &.{rows + 1} }};
    const layout: Layout = .{ .query_heads_by_layer = &.{ 1, 1 }, .scales_by_donor = &.{ 1, 1 }, .softcaps_by_donor = &.{ 0, 0 } };
    var candidate = try prepareFromHeadSource(std.testing.allocator, &source, &batches, layout, options);
    defer candidate.deinit();
    try std.testing.expectEqual(@as(usize, 2), loader.open_count);
    try std.testing.expectEqual(loader.open_count, loader.close_count);
    try std.testing.expect(!loader.live);
    try std.testing.expectEqualSlices(f32, global_head.keys, candidate.state.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, local_head.values, candidate.state.donors[1].heads[0].values);
    loader.unsupported_fit = true;
    var preserved = try prepareFromHeadSource(std.testing.allocator, &source, &batches, layout, options);
    defer preserved.deinit();
    try std.testing.expectEqualSlices(f32, global_head.values, preserved.state.donors[0].heads[0].values);
    try std.testing.expectEqual(@as(usize, 1), stats.v18_fallback_heads);
    try std.testing.expectEqual(@as(usize, 1), stats.v18_preserved_heads);
    loader.unsupported_fit = false;
    const opens_before_stale = loader.open_count;
    source.expected_state_sha256 = @splat(9);
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, prepareFromHeadSource(std.testing.allocator, &source, &batches, layout, options));
    try std.testing.expectEqual(opens_before_stale, loader.open_count);
    source.expected_state_sha256 = metadata.state_sha256;
    const opens_before_failure = loader.open_count;
    loader.fail_fit = true;
    try std.testing.expectError(error.InjectedFailure, prepareFromHeadSource(std.testing.allocator, &source, &batches, layout, options));
    try std.testing.expectEqual(opens_before_failure + 1, loader.open_count);
    try std.testing.expectEqual(loader.open_count, loader.close_count);
    try std.testing.expect(!loader.live);
    loader.fail_fit = false;
    options.fit.control = .{ .check_fn = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    }.check };
    try std.testing.expectError(error.Cancelled, prepareFromHeadSource(std.testing.allocator, &source, &batches, layout, options));
    try std.testing.expect(!loader.live);
}
const layout_fixture: Layout = .{ .query_heads_by_layer = &.{ 4, 4, 4 }, .scales_by_donor = &.{ 1, 1 }, .softcaps_by_donor = &.{ 0, 0 } };
fn batchesFixture() [8]ReferenceBatch {
    var batches: [8]ReferenceBatch = undefined;
    for (&batches, 0..) |*b, i| b.* = .{ .consumer_layer = if (i < 4) 0 else 2, .query_head = @intCast(i % 4), .train = &.{0}, .heldout = &.{1}, .train_positions = &.{4}, .heldout_positions = &.{5} };
    return batches;
}

test "attention_match head plan preserves prior fitted coverage across cycles" {
    const options: Options = .{ .prefix_rows = 2, .recent_rows = 2, .chunk_rows = 4, .retention_divisor = 2 };
    const head: checkpoint.Head = .{
        .dim = 1,
        .compacted_until = 14,
        .positions = &.{ 0, 1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19 },
        .coverage_ends = &.{ 1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20 },
        .bias = &.{ 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 },
    };
    const plan = (try headPlan(20, head, options)).?;
    try std.testing.expectEqualDeep(HeadPlan{ .old_rows = 12, .prefix_rows = 8, .target_rows = 12, .block_rows = 4 }, plan);
    var recent_only = options;
    recent_only.recent_rows = 6;
    try std.testing.expectEqual(@as(?HeadPlan, null), try headPlan(20, head, recent_only));
    recent_only.recent_rows = 7;
    try std.testing.expectError(error.CompactionRecentWindowUnavailable, headPlan(20, head, recent_only));
    var unavailable = options;
    unavailable.prefix_rows = 3;
    try std.testing.expectError(error.CompactionPrefixUnavailable, headPlan(20, head, unavailable));
    var invalid = options;
    invalid.chunk_rows = 0;
    try std.testing.expectError(error.InvalidFitConfig, headPlan(20, head, invalid));
}

test "attention_match engine compacts shared donors once and preserves exact recent and local rows" {
    const batches = batchesFixture();
    var result = try prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .recent_rows = 1, .chunk_rows = 2 });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.state.donors.len);
    try std.testing.expectEqualSlices(u32, state.layer_donors, result.state.layer_donors);
    try std.testing.expectEqualSlices(f32, local.values, result.state.donors[1].heads[0].values);
    const head = result.state.donors[0].heads[0];
    try std.testing.expectEqualSlices(u64, &.{ 0, 2, 3 }, head.positions);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3, 4 }, head.coverage_ends);
    try std.testing.expectEqual(@as(f32, 8), head.values[2]);
    try std.testing.expectEqual(@as(f32, 0), head.bias[2]);
    try std.testing.expectEqual(@as(u64, 4), result.state.logical_length);
}

test "attention_match engine requires queries from every shared consumer and GQA head" {
    var batches = batchesFixture();
    try std.testing.expectError(error.MissingCompactionConsumerQueries, prepare(std.testing.allocator, state, batches[0..4], layout_fixture, .{ .recent_rows = 0 }));
    batches[1].query_head = 0;
    try std.testing.expectError(error.InvalidCompactionQueries, prepare(std.testing.allocator, state, &batches, layout_fixture, .{}));
    batches = batchesFixture();
    batches[0].heldout_positions = &.{4};
    try std.testing.expectError(error.InvalidCompactionQueries, prepare(std.testing.allocator, state, &batches, layout_fixture, .{}));
    try std.testing.expectEqual(@as(f32, 2), state.donors[0].heads[0].values[0]);
}

test "attention_match engine validates three independent overlapping TRAIN position domains" {
    const values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const positions = [_]u64{ 4, 5, 4, 5, 4, 5 };
    var batch: ReferenceBatch = .{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &values,
        .heldout = &.{7},
        .train_positions = &positions,
        .heldout_positions = &.{6},
        .independent_train_split = 2,
        .independent_train_second_split = 4,
    };
    try validateReferencePositions(batch, 4);

    batch.independent_train_second_split = null;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
    batch.independent_train_split = null;
    batch.independent_train_second_split = 4;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
    batch.independent_train_split = 2;
    inline for (.{ @as(usize, 0), 2, 1, 6 }) |bad| {
        batch.independent_train_second_split = bad;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
    }
    batch.independent_train_second_split = 4;
    const bad_primary = [_]u64{ 5, 4, 4, 5, 4, 5 };
    batch.train_positions = &bad_primary;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
    const bad_historical = [_]u64{ 4, 5, 5, 4, 4, 5 };
    batch.train_positions = &bad_historical;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
    const bad_future = [_]u64{ 4, 5, 4, 5, 5, 4 };
    batch.train_positions = &bad_future;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 4));
}

test "recent64 full-current-source replay keeps real TRAIN positions and future-only HHO" {
    var train_positions: [64]u64 = undefined;
    for (&train_positions, 0..) |*position, row| position.* = 8064 + row;
    var batch: ReferenceBatch = .{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &.{},
        .heldout = &.{},
        .train_positions = &train_positions,
        .heldout_positions = &.{ 8128, 8129 },
    };
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositions(batch, 8128));
    const replay: Options = .{
        .training_position_policy = .recent_prefill_r64_full_current_source_replay_v1,
        .prefix_rows = 128,
        .recent_rows = 2048,
        .max_queries_per_consumer = 64,
    };
    try validateReferencePositionsForOptions(batch, 8128, replay);

    var bad_boundary = train_positions;
    bad_boundary[0] = 8063;
    batch.train_positions = &bad_boundary;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, replay));
    batch.train_positions = &train_positions;
    batch.heldout_positions = &.{8127};
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, replay));
    batch.heldout_positions = &.{8128};
    batch.independent_train_split = 32;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, replay));
    batch.independent_train_split = null;
    var wrong_geometry = replay;
    wrong_geometry.recent_rows = 2047;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, wrong_geometry));
}

test "recent32 future32 policy validates both domains and heldout separation" {
    var positions: [64]u64 = undefined;
    for (positions[0..32], 0..) |*position, row| position.* = 8096 + row;
    for (positions[32..], 0..) |*position, row| position.* = 8132 + 16 * row;
    var batch: ReferenceBatch = .{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &.{},
        .heldout = &.{},
        .train_positions = &positions,
        .heldout_positions = &.{ 8128, 8136, 8144, 8152 },
        .independent_train_split = 32,
    };
    const base: Options = .{ .prefix_rows = 128, .recent_rows = 2048, .max_queries_per_consumer = 64 };
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, base));
    var recent_only = base;
    recent_only.training_position_policy = .recent_prefill_r64_full_current_source_replay_v1;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, recent_only));
    inline for (.{ Options.TrainingPositionPolicy.recent32_future_primary32_full_source_replay_v1, Options.TrainingPositionPolicy.recent32_future_primary32_incremental_reduction_diagnostic_v1 }) |policy| {
        var options = base;
        options.training_position_policy = policy;
        try validateReferencePositionsForOptions(batch, 8128, options);
        positions[0] -= 1;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        positions[0] += 1;
        positions[32] -= 4;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        positions[32] += 4;
        batch.heldout_positions = &.{ 8128, 8132 };
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.heldout_positions = &.{ 8128, 8127 };
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.heldout_positions = &.{ 8136, 8128 };
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.heldout_positions = &.{ 8128, 8136, 8144, 8152 };
        batch.independent_train_split = null;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.independent_train_split = 31;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.independent_train_split = 32;
        batch.independent_train_second_split = 48;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
        batch.independent_train_second_split = null;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8129, options));
        options.recent_rows = 2047;
        try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
    }
}

test "recent64 future64 policy is exact and mutation sensitive" {
    var positions: [128]u64 = undefined;
    for (positions[0..64], 0..) |*position, row| position.* = 8064 + row;
    for (positions[64..], 0..) |*position, row| position.* = 8132 + 8 * row;
    var batch: ReferenceBatch = .{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &.{},
        .heldout = &.{},
        .train_positions = &positions,
        .heldout_positions = &.{ 8128, 8136, 8144, 8152 },
        .independent_train_split = 64,
    };
    var options: Options = .{
        .prefix_rows = 128,
        .recent_rows = 2048,
        .max_queries_per_consumer = 128,
        .training_position_policy = .recent64_future_primary64_incremental_reduction_diagnostic_v1,
    };
    try validateReferencePositionsForOptions(batch, 8128, options);
    positions[0] -= 1;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
    positions[0] += 1;
    positions[64] -= 1;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
    positions[64] += 1;
    batch.heldout_positions = &.{ 8128, 8132 };
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
    batch.heldout_positions = &.{ 8128, 8136, 8144, 8152 };
    batch.independent_train_split = 63;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
    batch.independent_train_split = 64;
    options.max_queries_per_consumer = 64;
    try std.testing.expectError(error.InvalidCompactionQueries, validateReferencePositionsForOptions(batch, 8128, options));
}

test "attention_match engine rejects enlarging exact recent window into fitted history" {
    const batches = batchesFixture();
    var candidate = try prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .recent_rows = 1 });
    defer candidate.deinit();
    try std.testing.expectError(error.CompactionRecentWindowUnavailable, prepare(std.testing.allocator, candidate.state, &batches, layout_fixture, .{ .recent_rows = 3 }));
    try std.testing.expectEqual(@as(u64, 3), candidate.state.donors[0].heads[0].compacted_until);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    const batches = batchesFixture();
    var result = try prepare(allocator, state, &batches, layout_fixture, .{ .recent_rows = 0, .chunk_rows = 2 });
    defer result.deinit();
}
test "attention_match engine failures preserve source and unwind partial candidates" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "attention_match engine twenty actual checkpoint cycles preserve fitted history and original mass" {
    const a = std.testing.allocator;
    const tokens: [44]u32 = @splat(0);
    var current: ?checkpoint.Owned = null;
    defer if (current) |*owned| owned.deinit();
    for (0..20) |cycle| {
        const length: usize = 4 + cycle * 2;
        var positions: [44]u64 = undefined;
        var ends: [44]u64 = undefined;
        var keys: [44]f32 = @splat(0);
        var values: [44]f32 = @splat(3);
        var biases: [44]f32 = @splat(0);
        const retained: usize = if (current) |old| old.state.donors[0].heads[0].positions.len else 0;
        const appended: usize = if (current != null) 2 else length;
        const rows = retained + appended;
        if (current) |old| {
            const h = old.state.donors[0].heads[0];
            @memcpy(positions[0..retained], h.positions);
            @memcpy(ends[0..retained], h.coverage_ends);
            @memcpy(keys[0..retained], h.keys);
            @memcpy(values[0..retained], h.values);
            @memcpy(biases[0..retained], h.bias);
        }
        for (retained..rows) |row| {
            positions[row] = length - appended + row - retained;
            ends[row] = positions[row] + 1;
        }
        const h: checkpoint.Head = .{ .dim = 1, .compacted_until = if (current) |old| old.state.logical_length else 0, .positions = positions[0..rows], .coverage_ends = ends[0..rows], .keys = keys[0..rows], .values = values[0..rows], .bias = biases[0..rows] };
        const local_positions = [_]u64{ length - 2, length - 1 };
        const local_ends = [_]u64{ length - 1, length };
        const lh: checkpoint.Head = .{ .dim = 1, .positions = &local_positions, .coverage_ends = &local_ends, .keys = &.{ 0, 0 }, .values = &.{ 3, 3 }, .bias = &.{ 0, 0 } };
        var source = state;
        source.logical_length = length;
        source.tokens = tokens[0..length];
        source.donors = &.{ .{ .layer = 0, .heads = &.{ h, h } }, .{ .layer = 1, .window = 2, .heads = &.{ lh, lh } } };
        var batches = batchesFixture();
        const train_positions = [_]u64{length};
        const heldout_positions = [_]u64{length + 1};
        for (&batches) |*batch| {
            batch.train_positions = &train_positions;
            batch.heldout_positions = &heldout_positions;
        }
        var candidate = try prepare(a, source, &batches, layout_fixture, .{ .recent_rows = 0, .validation = .whole_head });
        defer candidate.deinit();
        const encoded = try checkpoint.encode(a, candidate.state, .{});
        defer a.free(encoded);
        var restored = try checkpoint.decode(a, encoded, identity, .{});
        errdefer restored.deinit();
        const head = restored.state.donors[0].heads[0];
        try std.testing.expectEqual(length, head.compacted_until);
        try std.testing.expectEqual(retained + 1, head.positions.len);
        try std.testing.expectEqualSlices(u64, h.positions[0..retained], head.positions[0..retained]);
        try std.testing.expectEqualSlices(u64, h.coverage_ends[0..retained], head.coverage_ends[0..retained]);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(h.keys[0..retained]), std.mem.sliceAsBytes(head.keys[0..retained]));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(h.values[0..retained]), std.mem.sliceAsBytes(head.values[0..retained]));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(h.bias[0..retained]), std.mem.sliceAsBytes(head.bias[0..retained]));
        var mass: f64 = 0;
        var numerator: f64 = 0;
        for (head.bias, head.values) |bias, value| {
            const weight = @exp(@as(f64, bias));
            mass += weight;
            numerator += weight * value;
        }
        try std.testing.expectApproxEqAbs(@as(f64, @floatFromInt(length)), mass, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f64, 3), numerator / mass, 1e-6);
        if (current) |*owned| owned.deinit();
        current = restored;
    }
}

test "attention_match fitting validates serving f16 rounding before publication" {
    var rounded_global = global;
    rounded_global.values = &.{ 1.0001, 1.0001, 1.0001, 1.0001 };
    var source = state;
    source.donors = &.{ .{ .layer = 0, .heads = &.{ rounded_global, rounded_global } }, .{ .layer = 1, .window = 2, .heads = &.{ local, local } } };
    const batches = batchesFixture();
    var stats: Stats = .{};
    var options: Options = .{ .stats = &stats, .recent_rows = 0, .chunk_rows = 4, .fit = .{ .target_rows = 1, .max_relative_output_rmse = 1e-6 } };
    var precise = try prepare(std.testing.allocator, source, &batches, layout_fixture, options);
    defer precise.deinit();
    options.storage_precision = .f16;
    try std.testing.expectError(error.CompactionFitRejected, prepare(std.testing.allocator, source, &batches, layout_fixture, options));
    try std.testing.expectEqual(@as(usize, 1), stats.attempted_chunks);
    try std.testing.expectEqual(@as(usize, 0), stats.completed_chunks);
    try std.testing.expectEqual(@as(u32, 0), stats.donor_layer);
    try std.testing.expectEqual(@as(usize, 0), stats.head);
    try std.testing.expect(stats.last_fit.nnls_sweeps > 0);
    try std.testing.expect(stats.validating_storage);
    try std.testing.expect(stats.last_fit.relative_output_rmse < 1e-6);
    try std.testing.expect(stats.rounded_fit.relative_output_rmse > 1e-6);
    options.validation = .whole_head;
    try std.testing.expectError(error.CompactionFitRejected, prepare(std.testing.allocator, source, &batches, layout_fixture, options));
    try std.testing.expect(stats.validating_head);
    try std.testing.expectEqual(@as(usize, 1), stats.completed_chunks);
    try std.testing.expectEqual(@as(usize, 0), stats.validated_heads);
    try std.testing.expect(stats.rounded_fit.relative_output_rmse > 1e-6);
    options.validation = .chunk;
    options.fit.max_relative_output_rmse = 0.01;
    var rounded = try prepare(std.testing.allocator, source, &batches, layout_fixture, options);
    defer rounded.deinit();
    try std.testing.expectEqual(@as(usize, 2), stats.attempted_chunks);
    try std.testing.expectEqual(@as(usize, 2), stats.completed_chunks);
    try std.testing.expectEqual(@as(f32, 1), rounded.state.donors[0].heads[0].values[0]);
    try std.testing.expectEqualSlices(f32, local.values, rounded.state.donors[1].heads[0].values);
    try std.testing.expectEqualSlices(f32, &.{ 1.0001, 1.0001, 1.0001, 1.0001 }, source.donors[0].heads[0].values);
}

test "attention_match fitting admits two resident states without serialized full cache clone" {
    const batches = batchesFixture();
    // No eligible old rows: this exercises the exact-copy path at the new
    // two-state bound, below the former three-state + encoding requirement.
    const bound = try state.residentBytes() * 2;
    var result = try prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .max_transient_host_bytes = bound });
    defer result.deinit();
    try std.testing.expectEqualSlices(f32, global.values, result.state.donors[0].heads[0].values);
    try std.testing.expect(result.state.donors[0].heads[0].values.ptr != global.values.ptr);
    try std.testing.expectError(error.CompactionWorkspaceLimit, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .max_transient_host_bytes = bound - 1 }));
}

test "attention_match whole head validation preserves exact prefix and rejects unavailable prefix" {
    const batches = batchesFixture();
    var stats: Stats = .{};
    var result = try prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .stats = &stats, .recent_rows = 1, .prefix_rows = 1, .validation = .whole_head, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set } });
    defer result.deinit();
    const h = result.state.donors[0].heads[0];
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 3 }, h.positions);
    try std.testing.expectEqualSlices(u64, &.{ 1, 3, 4 }, h.coverage_ends);
    try std.testing.expectEqual(@as(f32, 2), h.values[0]);
    try std.testing.expectEqual(@as(f32, 0), h.bias[0]);
    try std.testing.expectEqual(@as(usize, 2), stats.validated_heads);
    try std.testing.expectError(error.CompactionPrefixUnavailable, prepare(std.testing.allocator, result.state, &batches, layout_fixture, .{ .recent_rows = 1, .prefix_rows = 2 }));
    var second = try prepare(std.testing.allocator, result.state, &batches, layout_fixture, .{ .recent_rows = 1, .prefix_rows = 1, .validation = .whole_head });
    defer second.deinit();
    try std.testing.expectEqualSlices(f32, h.values, second.state.donors[0].heads[0].values);
}

fn wholeHeadAllocationFailure(allocator: std.mem.Allocator) !void {
    const batches = batchesFixture();
    var result = try prepare(allocator, state, &batches, layout_fixture, .{ .recent_rows = 1, .prefix_rows = 1, .validation = .whole_head, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set } });
    defer result.deinit();
}
test "attention_match whole head validation unwinds allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, wholeHeadAllocationFailure, .{});
}

fn rejectedFixture() checkpoint.State {
    var source = state;
    // Return a static backing array; the source and every head remain borrowed.
    const donors = struct {
        const heads = [_]checkpoint.Head{ blk: {
            var h = global;
            h.keys = &.{ -1, -1, 1, 1 };
            break :blk h;
        }, global };
        const values = [_]checkpoint.Donor{ .{ .layer = 0, .heads = &heads }, .{ .layer = 1, .window = 2, .heads = &.{ local, local } } };
    };
    source.donors = &donors.values;
    return source;
}

test "attention_match explicit fallback preserves rejected head and reports actual storage savings" {
    const source = rejectedFixture();
    const batches = batchesFixture();
    var stats: Stats = .{};
    var options: Options = .{ .stats = &stats, .recent_rows = 0, .chunk_rows = 4, .validation = .whole_head };
    try std.testing.expectError(error.CompactionFitRejected, prepare(std.testing.allocator, source, &batches, layout_fixture, options));
    options.rejected_head_policy = .preserve_original;
    var candidate = try prepare(std.testing.allocator, source, &batches, layout_fixture, options);
    defer candidate.deinit();
    try std.testing.expectEqual(@as(usize, 1), stats.preserved_heads);
    try std.testing.expectEqual(@as(usize, 1), stats.validated_heads);
    const kept = candidate.state.donors[0].heads[0];
    try std.testing.expectEqualSlices(f32, source.donors[0].heads[0].keys, kept.keys);
    try std.testing.expectEqualSlices(f32, source.donors[0].heads[0].values, kept.values);
    try std.testing.expectEqualSlices(u64, source.donors[0].heads[0].coverage_ends, kept.coverage_ends);
    try std.testing.expect(kept.keys.ptr != source.donors[0].heads[0].keys.ptr);
    try std.testing.expect(stats.max_rejected_mass_error > 0.05);
    try std.testing.expectEqual(@as(usize, 1), candidate.state.donors[0].heads[1].positions.len);
    // Tiny dim=1 test heads are dominated by metadata. A nominal 4x request
    // must not be mistaken for a real device saving.
    try std.testing.expect(stats.actual_global_savings_ratio < 1);
    options.min_global_savings_ratio = 1;
    try std.testing.expectError(error.CompactionSavingsInsufficient, prepare(std.testing.allocator, source, &batches, layout_fixture, options));
    options.min_global_savings_ratio = std.math.nan(f64);
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, source, &batches, layout_fixture, options));
}

fn preservedAllocationFailure(allocator: std.mem.Allocator) !void {
    const source = rejectedFixture();
    const batches = batchesFixture();
    var result = try prepare(allocator, source, &batches, layout_fixture, .{ .recent_rows = 0, .chunk_rows = 4, .validation = .whole_head, .rejected_head_policy = .preserve_original });
    defer result.deinit();
}
test "attention_match fallback cannot swallow allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, preservedAllocationFailure, .{});
}

test "attention_match fallback cannot swallow cancellation or invalid source" {
    const batches = batchesFixture();
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, prepare(std.testing.allocator, rejectedFixture(), &batches, layout_fixture, .{ .rejected_head_policy = .preserve_original, .fit = .{ .target_rows = 1, .control = .{ .check_fn = Cancel.check } } }));
    var bad = rejectedFixture();
    bad.logical_length = 0;
    try std.testing.expectError(error.InvalidKvCheckpoint, prepare(std.testing.allocator, bad, &batches, layout_fixture, .{ .rejected_head_policy = .preserve_original }));
}

const ParallelTestBackend = struct {
    fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
        return 0;
    }
    fn failWorkspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
        return error.InjectedWorkerFailure;
    }
    const backend: solver.Backend = .{ .workspace_fn = workspace };
};
test "attention_match explicit final publication authorization is independent FP64" {
    const control: solver.Control = .{};
    const accelerated: solver.Config = .{
        .target_rows = 4,
        .backend = ParallelTestBackend.backend,
        .max_relative_mass_error = 0.0125,
        .max_relative_output_rmse = 0.025,
        .control = control,
    };
    const unchanged = finalAuthorizationConfig(accelerated, false, null);
    try std.testing.expect(unchanged.backend != null);
    const authorization = finalAuthorizationConfig(accelerated, true, null);
    try std.testing.expect(authorization.backend == null);
    try std.testing.expectEqual(accelerated.target_rows, authorization.target_rows);
    try std.testing.expectEqual(accelerated.max_relative_mass_error, authorization.max_relative_mass_error);
    try std.testing.expectEqual(accelerated.max_relative_output_rmse, authorization.max_relative_output_rmse);
    try std.testing.expectEqual(accelerated.control.context, authorization.control.context);

    // Scalar validation streams one query at a time, so the independently
    // admitted storage is bounded by source+candidate rows, not query count.
    const expected = try mul(try add(512, try add(256, try add(try mul(64, 2), 2))), 8);
    try std.testing.expectEqual(expected, try solver.validationBytes(512, 256, 64, 4096, authorization));

    var context: validation_accelerate.Context = .{};
    const summary = trustedValidationBackend(.accelerate_fp64_summary_v1, &context).?;
    const accelerated_authorization = finalAuthorizationConfig(accelerated, true, summary);
    try std.testing.expect(accelerated_authorization.backend != null);
    try std.testing.expect(accelerated_authorization.backend.?.summary_fn != null);
    try std.testing.expect(accelerated_authorization.backend.?.attend_fn == null);
    try std.testing.expect(accelerated_authorization.backend.?.context != accelerated.backend.?.context);
    try std.testing.expect(try solver.validationBytes(512, 256, 64, 4096, accelerated_authorization) > expected);
    const codec_training = codecTrainingConfig(accelerated_authorization);
    try std.testing.expect(codec_training.backend == null);
    try std.testing.expect(codec_training.metrics == null);
    try std.testing.expectEqual(accelerated_authorization.max_relative_mass_error, codec_training.max_relative_mass_error);
    try std.testing.expectEqual(accelerated_authorization.max_relative_output_rmse, codec_training.max_relative_output_rmse);
    try std.testing.expectEqual(accelerated_authorization.control.context, codec_training.control.context);
}
test "attention_match planned host bound admits parallel fitting and rejects one byte less" {
    const batches = batchesFixture();
    const backends: [4]solver.Backend = @splat(ParallelTestBackend.backend);
    var options: Options = .{
        .recent_rows = 1,
        .chunk_rows = 2,
        .validation = .whole_head,
        .fit = .{ .target_rows = 1, .backend = ParallelTestBackend.backend },
    };
    // Two consumer layers, each with two query heads per KV head.
    const workspace = try headWorkspaceBytes(4, 3, 2, 4, 4, 1, options);
    const source_bytes = try state.residentBytes();
    var serial = try prepare(std.testing.allocator, state, &batches, layout_fixture, options);
    defer serial.deinit();
    const expected = try checkpoint.encode(std.testing.allocator, serial.state, .{});
    defer std.testing.allocator.free(expected);
    for (2..5) |workers| {
        const bound = try parallelHostBytes(source_bytes, 2, workspace.bytes, workers);
        options.max_transient_host_bytes = bound;
        var parallel = try prepareParallel(state, &batches, layout_fixture, options, backends[0..workers]);
        defer parallel.deinit();
        const actual = try checkpoint.encode(std.testing.allocator, parallel.state, .{});
        defer std.testing.allocator.free(actual);
        try std.testing.expectEqualSlices(u8, expected, actual);
        options.max_transient_host_bytes = bound - 1;
        try std.testing.expectError(error.CompactionWorkspaceLimit, prepareParallel(state, &batches, layout_fixture, options, backends[0..workers]));
    }
}

test "attention_match host planning rejects malformed geometry and overflow" {
    try std.testing.expectError(error.InvalidFitConfig, headWorkspaceBytes(4, 2, 2, 2, 2, 0, .{}));
    try std.testing.expectError(error.InvalidFitConfig, headWorkspaceBytes(4, 2, 2, 3, 2, 2, .{}));
    try std.testing.expectError(error.InvalidFitConfig, parallelHostBytes(1, 1, 1, 0));
    try std.testing.expectError(error.InvalidFitConfig, parallelHostBytes(1, 1, 1, 5));
    try std.testing.expectError(error.CompactionWorkspaceLimit, parallelHostBytes(std.math.maxInt(usize), 1, 1, 4));
    try std.testing.expectError(error.CompactionWorkspaceLimit, parallelHostBytes(1, 1, std.math.maxInt(usize), 4));
}

test "attention_match parallel heads match serial checkpoints and aggregate diagnostics" {
    const batches = batchesFixture();
    var serial_stats: Stats = .{};
    var parallel_stats: Stats = .{};
    var options: Options = .{ .recent_rows = 1, .chunk_rows = 2, .stats = &serial_stats };
    var serial = try prepare(std.testing.allocator, state, &batches, layout_fixture, options);
    defer serial.deinit();
    options.stats = &parallel_stats;
    const backends: [4]solver.Backend = @splat(ParallelTestBackend.backend);
    for (2..5) |workers| {
        var parallel = try prepareParallel(state, &batches, layout_fixture, options, backends[0..workers]);
        defer parallel.deinit();
        const actual = try checkpoint.encode(std.testing.allocator, parallel.state, .{});
        defer std.testing.allocator.free(actual);
        const expected = try checkpoint.encode(std.testing.allocator, serial.state, .{});
        defer std.testing.allocator.free(expected);
        try std.testing.expectEqualSlices(u8, expected, actual);
        try std.testing.expectEqual(serial_stats.completed_chunks, parallel_stats.completed_chunks);
        try std.testing.expectEqual(serial_stats.source_global_bytes, parallel_stats.source_global_bytes);
        try std.testing.expectEqual(serial_stats.candidate_global_bytes, parallel_stats.candidate_global_bytes);
    }
}
test "attention_match parallel cancellation worker failures and aggregate admission preserve source" {
    const batches = batchesFixture();
    const backends: [4]solver.Backend = @splat(ParallelTestBackend.backend);
    var options: Options = .{ .recent_rows = 1, .chunk_rows = 2, .rejected_head_policy = .preserve_original };
    try std.testing.expectError(error.InvalidFitConfig, prepareParallel(state, &batches, layout_fixture, options, backends[0..1]));
    options.max_transient_host_bytes = 2 * try state.residentBytes() + 4;
    try std.testing.expectError(error.CompactionWorkspaceLimit, prepareParallel(state, &batches, layout_fixture, options, &backends));
    options.max_transient_host_bytes = 1024 * 1024;
    const failing: [2]solver.Backend = @splat(.{ .workspace_fn = ParallelTestBackend.failWorkspace });
    try std.testing.expectError(error.InjectedWorkerFailure, prepareParallel(state, &batches, layout_fixture, options, &failing));
    const CancelParallel = struct {
        active: std.atomic.Value(bool) = .init(false),
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.active.swap(true, .acq_rel)) return error.ConcurrentControlCallback;
            defer self.active.store(false, .release);
            self.calls += 1;
            if (self.calls >= 10) return error.CompactionCancelled;
        }
    };
    var cancellation: CancelParallel = .{};
    options.fit.control = .{ .context = &cancellation, .check_fn = CancelParallel.check };
    try std.testing.expectError(error.CompactionCancelled, prepareParallel(state, &batches, layout_fixture, options, &backends));
    try std.testing.expectEqual(@as(usize, 10), cancellation.calls);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, state.donors[0].heads[0].values);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 3 }, state.donors[0].heads[0].positions);
}

test "attention_match parallel rejects aliased mutable backend contexts before execution" {
    const batches = batchesFixture();
    var contexts: [2]u8 = .{ 0, 0 };
    var backends = [_]solver.Backend{ ParallelTestBackend.backend, ParallelTestBackend.backend };
    backends[0].context = &contexts[0];
    backends[1].context = &contexts[0];
    try std.testing.expectError(error.InvalidFitConfig, prepareParallel(state, &batches, layout_fixture, .{ .recent_rows = 1 }, &backends));
    backends[1].context = &contexts[1];
    var candidate = try prepareParallel(state, &batches, layout_fixture, .{ .recent_rows = 1 }, &backends);
    defer candidate.deinit();
}

test "attention_match repeated fit preserves exact prefix fitted history and exact recent tail" {
    const h: checkpoint.Head = .{ .dim = 1, .compacted_until = 4, .positions = &.{ 0, 1, 4, 5, 6, 7 }, .coverage_ends = &.{ 1, 4, 5, 6, 7, 8 }, .keys = &.{ 0, 0, 0, 0, 0, 0 }, .values = &.{ 3, 3, 3, 3, 3, 3 }, .bias = &.{ 0, @log(@as(f32, 3)), 0, 0, 0, 0 } };
    const lh: checkpoint.Head = .{ .dim = 1, .positions = &.{ 6, 7 }, .coverage_ends = &.{ 7, 8 }, .keys = &.{ 0, 0 }, .values = &.{ 3, 3 }, .bias = &.{ 0, 0 } };
    var source = state;
    source.logical_length = 8;
    source.tokens = &.{ 0, 0, 0, 0, 0, 0, 0, 0 };
    source.donors = &.{ .{ .layer = 0, .heads = &.{ h, h } }, .{ .layer = 1, .window = 2, .heads = &.{ lh, lh } } };
    var batches = batchesFixture();
    for (&batches) |*b| {
        b.train_positions = &.{8};
        b.heldout_positions = &.{9};
    }
    var stats: Stats = .{};
    const options: Options = .{ .prefix_rows = 1, .recent_rows = 1, .validation = .whole_head, .stats = &stats };
    var result = try prepare(std.testing.allocator, source, &batches, layout_fixture, options);
    defer result.deinit();
    const actual = result.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(u64, 7), actual.compacted_until);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 4, 7 }, actual.positions);
    try std.testing.expectEqualSlices(u64, &.{ 1, 4, 7, 8 }, actual.coverage_ends);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(h.bias[0..2]), std.mem.sliceAsBytes(actual.bias[0..2]));
    try std.testing.expectEqual(@as(f32, 0), actual.bias[3]);
    var unchanged = try prepare(std.testing.allocator, result.state, &batches, layout_fixture, options);
    defer unchanged.deinit();
    try std.testing.expectEqual(@as(usize, 0), stats.attempted_chunks);
    const first = try checkpoint.encode(std.testing.allocator, result.state, .{});
    defer std.testing.allocator.free(first);
    const second = try checkpoint.encode(std.testing.allocator, unchanged.state, .{});
    defer std.testing.allocator.free(second);
    try std.testing.expectEqualSlices(u8, first, second);
}

test "attention_match whole-head validation gate cancellation preserves permits" {
    var gate: HeadValidationGate = .{};
    const Cancel = struct {
        calls: usize = 0,
        after: usize,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls >= self.after) return error.CompactionCancelled;
        }
    };
    var cancel: Cancel = .{ .after = 2 };
    try std.testing.expectError(error.CompactionCancelled, gate.enter(.{ .context = &cancel, .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 0), gate.active.load(.acquire));
    try gate.enter(.{});
    try gate.enter(.{});
    cancel = .{ .after = 3 };
    try std.testing.expectError(error.CompactionCancelled, gate.enter(.{ .context = &cancel, .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 2), gate.active.load(.acquire));
    gate.leave();
    try gate.enter(.{});
    gate.leave();
    gate.leave();
    try std.testing.expectEqual(@as(usize, 0), gate.active.load(.acquire));
}

test "attention_match one-permit validation gate releases cancellation reservation" {
    var gate: HeadValidationGate = .{ .maximum = 1 };
    try gate.enter(.{});
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, gate.enter(.{ .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 1), gate.active.load(.acquire));
    gate.leave();
    try gate.enter(.{});
    gate.leave();
    try std.testing.expectEqual(@as(usize, 0), gate.active.load(.acquire));
}
test "attention_match validation concurrency rejects unbounded configurations" {
    const batches = batchesFixture();
    for ([_]usize{ 0, 5 }) |maximum| try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .max_parallel_validations = maximum }));
}

fn fp8EngineAllocationFailures(a: std.mem.Allocator) !void {
    const batches = batchesFixture();
    var stats: Stats = .{};
    var result = try prepare(a, state, &batches, layout_fixture, .{ .stats = &stats, .recent_rows = 1, .prefix_rows = 1, .validation = .whole_head, .storage_precision = .f16, .fp8_checkpoint = true });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), stats.fp8_heads);
    const head = result.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(u32, 1), head.fp8.first_row);
    try std.testing.expectEqual(@as(u32, 1), head.fp8.rows);
    try std.testing.expectEqual(global.keys[0], head.keys[0]);
    try std.testing.expectEqual(global.values[0], head.values[0]);
    try std.testing.expectEqual(@as(f32, 8), head.values[head.values.len - 1]);
    try std.testing.expectEqualSlices(f32, local.values, result.state.donors[1].heads[0].values);
    try std.testing.expectEqual(@as(u32, 0), result.state.donors[1].heads[0].fp8.rows);
    const bytes = try checkpoint.encodeWithEncoding(a, result.state, .{}, .fp8, .{});
    defer a.free(bytes);
    var restored = try checkpoint.decode(a, bytes, identity, .{});
    defer restored.deinit();
    try std.testing.expectEqualSlices(f32, head.values, restored.state.donors[0].heads[0].values);
}

test "attention_match FP8 validates all consumers and preserves protected rows under allocation failure" {
    try fp8EngineAllocationFailures(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, fp8EngineAllocationFailures, .{});
}

test "attention_match FP8 rejection measures combined error and keeps the f16 candidate" {
    const original: solver.Head = .{ .dim = 1, .rows = 1, .scale = 1, .keys = &.{0}, .values = &.{1.099609375}, .bias = &.{0} };
    var keys64 = [_]f64{0};
    var values64 = [_]f64{1.0625};
    const compact: solver.Head = .{ .dim = 1, .rows = 1, .scale = 1, .keys = &keys64, .values = &values64, .bias = &.{0} };
    var keys = [_]f32{0};
    var values = [_]f32{1.0625};
    var result: checkpoint.Head = .{ .dim = 1, .compacted_until = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &keys, .values = &values, .bias = &.{0} };
    const cfg: solver.Config = .{ .target_rows = 1 };
    try validateRounded(std.testing.allocator, original, compact, &.{0}, cfg);
    var diagnostic: CodecDiagnostic = .{};
    try std.testing.expect(!try quantizeCheckpointHead(std.testing.allocator, original, compact, &result, 0, 1, &.{0}, cfg, &diagnostic));
    try std.testing.expectEqual(CodecDiagnostic.Kind.fp8, diagnostic.kind);
    try std.testing.expectEqual(CodecDiagnostic.Outcome.postcodec_heldout_gate, diagnostic.outcome);
    try std.testing.expectEqual(@as(f32, 1.0625), values[0]);
    try std.testing.expectEqual(@as(u32, 0), result.fp8.rows);
}

test "attention_match FP8 requires serving half precision and whole-head validation" {
    const batches = batchesFixture();
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .fp8_checkpoint = true }));
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .fp8_checkpoint = true, .storage_precision = .f16 }));
}

test "attention_match FP8 successive fit preserves prior encoded rows and scales" {
    const a = std.testing.allocator;
    for ([_]f32{ 3, 1000 }) |new_value| {
        const values = [_]f32{ 3, 3, new_value, new_value, new_value, new_value, new_value };
        const old: checkpoint.Head = .{
            .dim = 1,
            .compacted_until = 3,
            .positions = &.{ 0, 1, 3, 4, 5, 6, 7 },
            .coverage_ends = &.{ 1, 3, 4, 5, 6, 7, 8 },
            .keys = &.{ 0, 0, 0, 0, 0, 0, 0 },
            .values = &values,
            .bias = &.{ 0, @log(@as(f32, 2)), 0, 0, 0, 0, 0 },
            .fp8 = .{ .first_row = 1, .rows = 1, .key_scale = 1, .value_scale = 1 },
        };
        const source: checkpoint.State = .{
            .identity = identity,
            .logical_length = 8,
            .tokens = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .logits = &.{0},
            .layer_donors = &.{0},
            .donors = &.{.{ .layer = 0, .heads = &.{old} }},
        };
        const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &.{0}, .heldout = &.{0}, .train_positions = &.{8}, .heldout_positions = &.{9} }};
        const layout: Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
        var stats: Stats = .{};
        var result = try prepare(a, source, &batches, layout, .{
            .stats = &stats,
            .recent_rows = 2,
            .prefix_rows = 1,
            .chunk_rows = 3,
            .validation = .whole_head,
            .storage_precision = .f16,
            .fp8_checkpoint = true,
        });
        defer result.deinit();
        const head = result.state.donors[0].heads[0];
        try std.testing.expectEqual(@as(u32, 1), head.fp8.first_row);
        try std.testing.expectEqual(@as(u32, if (new_value == 3) 2 else 1), head.fp8.rows);
        try std.testing.expectEqual(old.fp8.key_scale, head.fp8.key_scale);
        try std.testing.expectEqual(old.fp8.value_scale, head.fp8.value_scale);
        try std.testing.expectEqualSlices(u64, old.coverage_ends[0..2], head.coverage_ends[0..2]);
        try std.testing.expectEqualSlices(f32, old.keys[0..2], head.keys[0..2]);
        try std.testing.expectEqualSlices(f32, old.values[0..2], head.values[0..2]);
        try std.testing.expectEqualSlices(f32, old.bias[0..2], head.bias[0..2]);
        try std.testing.expectEqual(try checkpoint.codec.encode(old.values[1], old.fp8.value_scale), try checkpoint.codec.encode(head.values[1], head.fp8.value_scale));
        try std.testing.expectEqual(@as(usize, @intFromBool(new_value == 1000)), stats.fp8_fallback_heads);
        const encoded = try checkpoint.encodeWithEncoding(a, result.state, .{}, .fp8, .{});
        defer a.free(encoded);
        var decoded = try checkpoint.decode(a, encoded, identity, .{});
        defer decoded.deinit();
        try std.testing.expectEqualDeep(head.fp8, decoded.state.donors[0].heads[0].fp8);
        try std.testing.expectEqualSlices(f32, old.values[0..2], decoded.state.donors[0].heads[0].values[0..2]);
    }
}

test "attention_match FP8 later f16 gap preserves the prior encoded range" {
    const a = std.testing.allocator;
    const old: checkpoint.Head = .{
        .dim = 1,
        .compacted_until = 6,
        .positions = &.{ 0, 1, 3, 6, 7, 8, 9, 10 },
        .coverage_ends = &.{ 1, 3, 6, 7, 8, 9, 10, 11 },
        .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .values = &.{ 3, 3, 1000, 1000, 1000, 1000, 1000, 1000 },
        .bias = &.{ 0, @log(@as(f32, 2)), @log(@as(f32, 3)), 0, 0, 0, 0, 0 },
        .fp8 = .{ .first_row = 1, .rows = 1, .key_scale = 1, .value_scale = 1 },
    };
    const source: checkpoint.State = .{
        .identity = identity,
        .logical_length = 11,
        .tokens = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .heads = &.{old} }},
    };
    const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &.{0}, .heldout = &.{0}, .train_positions = &.{11}, .heldout_positions = &.{12} }};
    const layout: Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    var result = try prepare(a, source, &batches, layout, .{
        .recent_rows = 2,
        .prefix_rows = 1,
        .chunk_rows = 3,
        .validation = .whole_head,
        .storage_precision = .f16,
        .fp8_checkpoint = true,
    });
    defer result.deinit();
    const head = result.state.donors[0].heads[0];
    try std.testing.expectEqualDeep(old.fp8, head.fp8);
    try std.testing.expectEqualSlices(f32, old.keys[0..3], head.keys[0..3]);
    try std.testing.expectEqualSlices(f32, old.values[0..3], head.values[0..3]);
    try std.testing.expectEqualSlices(f32, old.bias[0..3], head.bias[0..3]);
    try std.testing.expect(head.positions.len < old.positions.len);
    const encoded = try checkpoint.encodeWithEncoding(a, result.state, .{}, .fp8, .{});
    defer a.free(encoded);
    var decoded = try checkpoint.decode(a, encoded, identity, .{});
    defer decoded.deinit();
    try std.testing.expectEqualDeep(old.fp8, decoded.state.donors[0].heads[0].fp8);
}

fn retryBatches() [8]ReferenceBatch {
    var batches = batchesFixture();
    for (&batches) |*batch| {
        batch.train = &.{ -1, 1 };
        batch.heldout = &.{ -0.5, 0.5 };
        batch.train_positions = &.{ 4, 5 };
        batch.heldout_positions = &.{ 6, 7 };
    }
    return batches;
}

fn retryOptions() Options {
    return .{ .recent_rows = 0, .chunk_rows = 4, .validation = .whole_head, .min_retention_divisor = 2, .rejected_head_policy = .preserve_original, .storage_precision = .f16, .fp8_checkpoint = true, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set } };
}

fn retentionRetryAllocationFailures(a: std.mem.Allocator) !void {
    const source = rejectedFixture();
    const batches = retryBatches();
    var stats: Stats = .{};
    var options = retryOptions();
    options.stats = &stats;
    var result = try prepare(a, source, &batches, layout_fixture, options);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), stats.retention_retries);
    try std.testing.expectEqual(@as(usize, 2), stats.validated_heads);
    try std.testing.expectEqual(@as(usize, 0), stats.preserved_heads);
    try std.testing.expectEqual(@as(usize, 2), result.state.donors[0].heads[0].positions.len);
    try std.testing.expect(stats.max_rejected_mass_error > 0.05);
    try std.testing.expect(stats.max_head_mass_error <= 0.05);
    try std.testing.expect(stats.max_head_output_rmse <= 0.05);
    try std.testing.expectEqualSlices(f32, &.{ -1, -1, 1, 1 }, source.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, local.values, result.state.donors[1].heads[0].values);
}

test "attention_match bounded retention retries preserve error gates and unwind allocations" {
    try retentionRetryAllocationFailures(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, retentionRetryAllocationFailures, .{});
}

test "attention_match retention retry cancellation cannot preserve or publish a failed head" {
    const Cancel = struct {
        fn check(raw: ?*anyopaque) !void {
            const stats: *Stats = @ptrCast(@alignCast(raw.?));
            if (stats.retention_retries != 0) return error.CompactionCancelled;
        }
    };
    var stats: Stats = .{};
    var options = retryOptions();
    options.stats = &stats;
    options.fit.control = .{ .context = &stats, .check_fn = Cancel.check };
    const batches = retryBatches();
    try std.testing.expectError(error.CompactionCancelled, prepare(std.testing.allocator, rejectedFixture(), &batches, layout_fixture, options));
    try std.testing.expectEqual(@as(usize, 0), stats.validated_heads);
    try std.testing.expectEqual(@as(usize, 0), stats.preserved_heads);
}

test "attention_match retention retries obey parallel admission and match serial results" {
    const source = rejectedFixture();
    const batches = retryBatches();
    var options = retryOptions();
    var serial = try prepare(std.testing.allocator, source, &batches, layout_fixture, options);
    defer serial.deinit();
    const expected = try checkpoint.encodeWithEncoding(std.testing.allocator, serial.state, .{}, .fp8, .{});
    defer std.testing.allocator.free(expected);
    const backends: [4]solver.Backend = @splat(ParallelTestBackend.backend);
    options.fit.backend = ParallelTestBackend.backend;
    var planned = options;
    planned.retention_divisor = 2;
    const workspace = try headWorkspaceBytes(4, 2, 4, 8, 8, 1, planned);
    for (2..5) |workers| {
        var stats: Stats = .{};
        options.stats = &stats;
        options.max_transient_host_bytes = try parallelHostBytes(try source.residentBytes(), 2, workspace.bytes, workers);
        var result = try prepareParallel(source, &batches, layout_fixture, options, backends[0..workers]);
        defer result.deinit();
        const actual = try checkpoint.encodeWithEncoding(std.testing.allocator, result.state, .{}, .fp8, .{});
        defer std.testing.allocator.free(actual);
        try std.testing.expectEqualSlices(u8, expected, actual);
        try std.testing.expectEqual(@as(usize, 1), stats.retention_retries);
        options.max_transient_host_bytes -= 1;
        try std.testing.expectError(error.CompactionWorkspaceLimit, prepareParallel(source, &batches, layout_fixture, options, backends[0..workers]));
    }
}

test "attention_match retention retry configuration is bounded and requires whole-head validation" {
    const batches = batchesFixture();
    for ([_]usize{ 0, 1, 5 }) |minimum| {
        try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .min_retention_divisor = minimum, .validation = .whole_head }));
    }
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .min_retention_divisor = 2 }));
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, state, &batches, layout_fixture, .{ .retention_divisor = 2, .min_retention_divisor = 3, .validation = .whole_head }));
}

fn rotated4RetainedCount(count: usize, retention: Rotated4RiskRetention) usize {
    return switch (retention) {
        .quarter_v1 => count / 4,
        .half_v2 => count / 2,
    };
}

fn quantizeRotatedHead(allocator: std.mem.Allocator, original: solver.Head, compact: solver.Head, result: *checkpoint.Head, first: usize, end: usize, train: []const f64, queries: []const f64, cfg: solver.Config, retention: Rotated4RiskRetention, diagnostic: ?*CodecDiagnostic) !bool {
    if (diagnostic) |value| value.* = .{ .kind = .rotated4, .rotated4_risk_retention = retention, .first = first, .end = end, .eligible_rows = end - first };
    const dim = result.dim;
    var candidate = result.compressed.append(allocator, first, dim, result.keys[first * dim .. end * dim], result.values[first * dim .. end * dim], cfg.control) catch |err| {
        if (err == error.InvalidQuantizedRow or err == error.NonFiniteAttention or err == error.InvalidKvCheckpoint) {
            if (diagnostic) |value| value.outcome = .invalid_quantized_row;
            return false;
        }
        return err;
    };
    defer candidate.deinit(allocator);
    // Training queries select rows. Held-out queries authorize publication only
    // once, against the original whole head with the unchanged error limits.
    const count = end - first;
    const samples = @min(train.len / dim, 64);
    if (samples == 0) return error.InvalidCompactionQueries;
    const sampled = try allocator.alloc(f64, samples * dim);
    defer allocator.free(sampled);
    for (0..samples) |i| {
        const q = if (samples == 1) 0 else i * (train.len / dim - 1) / (samples - 1);
        @memcpy(sampled[i * dim ..][0..dim], train[q * dim ..][0..dim]);
    }
    const baseline = try allocator.alloc(f64, samples * compact.rows);
    defer allocator.free(baseline);
    const probabilities = try allocator.alloc(f64, baseline.len);
    defer allocator.free(probabilities);
    var baseline_mass: [64]f64 = undefined;
    var candidate_mass: [64]f64 = undefined;
    const training_cfg = codecTrainingConfig(cfg);
    try solver.attendBatch(allocator, compact, sampled, baseline, null, baseline_mass[0..samples], training_cfg);
    var keys: [512]f32 = undefined;
    var values: [512]f32 = undefined;
    for (first..end) |row| {
        try cfg.control.check();
        try candidate.decodeRow(dim, candidate.index(row).compressed, keys[0..dim], values[0..dim]);
        for (keys[0..dim], values[0..dim], 0..) |k, v, d| {
            @constCast(compact.keys)[row * dim + d] = k;
            @constCast(compact.values)[row * dim + d] = v;
        }
    }
    try solver.attendBatch(allocator, compact, sampled, probabilities, null, candidate_mass[0..samples], training_cfg);
    const risk = try allocator.alloc(f64, count);
    defer allocator.free(risk);
    @memset(risk, 0);
    for (0..samples) |q| {
        try cfg.control.check();
        const ratio = @exp(candidate_mass[q] - baseline_mass[q]);
        if (!std.math.isFinite(ratio)) {
            if (diagnostic) |value| value.outcome = .nonfinite_probability_ratio;
            return false;
        }
        for (risk, 0..) |*score, i| {
            const offset = q * compact.rows + first + i;
            score.* = @max(score.*, @abs(probabilities[offset] * ratio - baseline[offset]));
        }
    }
    const order = try allocator.alloc(usize, count);
    defer allocator.free(order);
    for (order, 0..) |*row, i| row.* = i;
    const Sort = struct {
        fn less(scores: []const f64, lhs: usize, rhs: usize) bool {
            return if (scores[lhs] == scores[rhs]) lhs < rhs else scores[lhs] > scores[rhs];
        }
    };
    std.mem.sort(usize, order, @as([]const f64, risk), Sort.less);
    const selected = try allocator.alloc(bool, count);
    defer allocator.free(selected);
    @memset(selected, true);
    // Fixed TRAIN policy; never tune the retained fraction on held-out results.
    const retained_count = rotated4RetainedCount(count, retention);
    for (order[0..retained_count]) |i| {
        try cfg.control.check();
        selected[i] = false;
        for (0..dim) |d| {
            const offset = (first + i) * dim + d;
            @constCast(compact.keys)[offset] = result.keys[offset];
            @constCast(compact.values)[offset] = result.values[offset];
        }
    }
    const filtered = try candidate.selectNew(allocator, first, selected, dim, cfg.control);
    candidate.deinit(allocator);
    candidate = filtered;
    if (diagnostic) |value| {
        value.retained_exact_rows = retained_count;
        value.packed_rows = count - retained_count;
    }
    var metrics: solver.Metrics = .{};
    var validation = cfg;
    validation.metrics = &metrics;
    validateRounded(allocator, original, compact, queries, validation) catch |err| {
        if (err == error.CompactionFitRejected) {
            if (diagnostic) |value| {
                value.outcome = .postcodec_heldout_gate;
                value.metrics = metrics;
                value.metrics.quality_evaluated = true;
            }
            return false;
        }
        return err;
    };
    try cfg.control.check();
    for (first * dim..end * dim) |i| {
        @constCast(result.keys)[i] = @floatCast(compact.keys[i]);
        @constCast(result.values)[i] = @floatCast(compact.values[i]);
    }
    result.compressed.deinit(allocator);
    result.compressed = candidate;
    candidate = .{};
    if (cfg.metrics) |out| out.* = metrics;
    if (diagnostic) |value| {
        value.outcome = .accepted;
        value.metrics = metrics;
        value.metrics.quality_evaluated = true;
    }
    return true;
}

fn rotatedEngineAllocationFailures(a: std.mem.Allocator) !void {
    const keys: [4 * 64]f32 = @splat(0);
    const values: [4 * 64]f32 = @splat(2);
    const query: [64]f32 = @splat(0.5);
    const head: checkpoint.Head = .{ .dim = 64, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &keys, .values = &values, .bias = &.{ 0, 0, 0, 0 } };
    const input: checkpoint.State = .{ .identity = identity, .logical_length = 4, .tokens = &.{ 0, 0, 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .heads = &.{head} }} };
    const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &query, .heldout = &query, .train_positions = &.{4}, .heldout_positions = &.{5} }};
    const layout: Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{0.125}, .softcaps_by_donor = &.{0} };
    var stats: Stats = .{};
    const options: Options = .{ .stats = &stats, .prefix_rows = 1, .recent_rows = 1, .validation = .whole_head, .storage_precision = .f16, .rotated4_checkpoint = true };
    var result = try prepare(a, input, &batches, layout, options);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), stats.rotated4_heads);
    const fitted = result.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(usize, 1), fitted.compressed.rows());
    try std.testing.expectEqual(@as(u32, 1), fitted.compressed.spans[0].first);
    try std.testing.expectEqualSlices(f32, keys[0..64], fitted.keys[0..64]);
    try std.testing.expectEqualSlices(f32, values[192..], fitted.values[128..]);
    const bytes = try checkpoint.encodeWithEncoding(a, result.state, .{}, .rotated4, .{});
    defer a.free(bytes);
    var restored = try checkpoint.decode(a, bytes, identity, .{});
    defer restored.deinit();
    // With no newly eligible history the old compressed representation is cloned,
    // not rounded again, even when quantization is disabled for the next cycle.
    var next_options = options;
    next_options.rotated4_checkpoint = false;
    var again = try prepare(a, restored.state, &batches, layout, next_options);
    defer again.deinit();
    try std.testing.expectEqualSlices(u8, fitted.compressed.keys, again.state.donors[0].heads[0].compressed.keys);
    try std.testing.expectEqualSlices(u8, fitted.compressed.values, again.state.donors[0].heads[0].compressed.values);
}

test "rotated4 engine validates quantization preserves protected rows and unwinds allocation failures" {
    try rotatedEngineAllocationFailures(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, rotatedEngineAllocationFailures, .{});
}

test "rotated4 savings cannot publish a fit rejected by the FP16 policy" {
    const a = std.testing.allocator;
    const keys: [4 * 64]f32 = @splat(0);
    const values: [4 * 64]f32 = @splat(2);
    const query: [64]f32 = @splat(0.5);
    const head: checkpoint.Head = .{ .dim = 64, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &keys, .values = &values, .bias = &.{ 0, 0, 0, 0 } };
    const input: checkpoint.State = .{ .identity = identity, .logical_length = 4, .tokens = &.{ 0, 0, 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .heads = &.{head} }} };
    const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &query, .heldout = &query, .train_positions = &.{4}, .heldout_positions = &.{5} }};
    const layout: Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{0.125}, .softcaps_by_donor = &.{0} };
    var stats: Stats = .{};
    var options: Options = .{ .stats = &stats, .prefix_rows = 1, .recent_rows = 1, .validation = .whole_head, .storage_precision = .f16 };
    var baseline = try prepare(a, input, &batches, layout, options);
    defer baseline.deinit();
    const fp16_bytes = stats.candidate_global_bytes;
    const fp16_ratio = stats.actual_global_savings_ratio;
    options.rotated4_checkpoint = true;
    var compressed = try prepare(a, input, &batches, layout, options);
    defer compressed.deinit();
    try std.testing.expect(stats.rotated4_heads > 0);
    try std.testing.expectEqual(fp16_bytes, stats.fitted_global_bytes);
    try std.testing.expectEqual(fp16_ratio, stats.fitted_global_savings_ratio);
    try std.testing.expect(stats.actual_global_savings_ratio > fp16_ratio);
    options.min_global_savings_ratio = (fp16_ratio + stats.actual_global_savings_ratio) / 2;
    try std.testing.expectError(error.CompactionSavingsInsufficient, prepare(a, input, &batches, layout, options));
    const backends: [2]solver.Backend = @splat(ParallelTestBackend.backend);
    try std.testing.expectError(error.CompactionSavingsInsufficient, prepareParallel(input, &batches, layout, options, &backends));
    options.rotated4_checkpoint = false;
    try std.testing.expectError(error.CompactionSavingsInsufficient, prepare(a, input, &batches, layout, options));
    // The same geometry remains eligible when the FP16 fit itself qualifies.
    options.min_global_savings_ratio = fp16_ratio;
    options.rotated4_checkpoint = true;
    var accepted = try prepare(a, input, &batches, layout, options);
    defer accepted.deinit();
    try std.testing.expect(accepted.state.donors[0].heads[0].compressed.rows() > 0);
    try std.testing.expectEqualSlices(f32, &keys, input.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, &values, input.donors[0].heads[0].values);
}

test "rotated4 repeated fitting preserves compressed bytes across an exact fallback gap" {
    const a = std.testing.allocator;
    const query: [64]f32 = @splat(0.5);
    const keys: [4 * 64]f32 = @splat(0);
    const values: [4 * 64]f32 = @splat(2);
    const tokens: [16]u32 = @splat(0);
    const original: checkpoint.Head = .{ .dim = 64, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &keys, .values = &values, .bias = &.{ 0, 0, 0, 0 } };
    const source: checkpoint.State = .{ .identity = identity, .logical_length = 4, .tokens = tokens[0..4], .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .heads = &.{original} }} };
    var train_positions = [_]u64{4};
    var heldout_positions = [_]u64{5};
    const batches = [_]ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &query, .heldout = &query, .train_positions = &train_positions, .heldout_positions = &heldout_positions }};
    const layout: Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{0.125}, .softcaps_by_donor = &.{0} };
    var options: Options = .{ .prefix_rows = 1, .recent_rows = 1, .validation = .whole_head, .storage_precision = .f16, .rotated4_checkpoint = true };
    var current = try prepare(a, source, &batches, layout, options);
    defer current.deinit();
    for ([_]bool{ false, true, true }) |quantize| {
        const old = current.state.donors[0].heads[0];
        const n = old.positions.len;
        var positions: [16]u64 = undefined;
        var ends: [16]u64 = undefined;
        var bias: [16]f32 = @splat(0);
        var k: [16 * 64]f32 = @splat(0);
        var v: [16 * 64]f32 = @splat(2);
        @memcpy(positions[0..n], old.positions);
        @memcpy(ends[0..n], old.coverage_ends);
        @memcpy(bias[0..n], old.bias);
        @memcpy(k[0 .. n * 64], old.keys);
        @memcpy(v[0 .. n * 64], old.values);
        for (0..2) |i| {
            positions[n + i] = current.state.logical_length + i;
            ends[n + i] = positions[n + i] + 1;
        }
        var appended = old;
        appended.positions = positions[0 .. n + 2];
        appended.coverage_ends = ends[0 .. n + 2];
        appended.bias = bias[0 .. n + 2];
        appended.keys = k[0 .. (n + 2) * 64];
        appended.values = v[0 .. (n + 2) * 64];
        var input = current.state;
        input.logical_length += 2;
        input.tokens = tokens[0..@intCast(input.logical_length)];
        input.donors = &.{.{ .heads = &.{appended} }};
        train_positions[0] = input.logical_length;
        heldout_positions[0] = input.logical_length + 1;
        options.rotated4_checkpoint = quantize;
        var next = try prepare(a, input, &batches, layout, options);
        errdefer next.deinit();
        // An independently prepared unquantized fit retains the older packed
        // history, including the gap. Pricing must match it even when new spans
        // merge with a preceding span in the quantized candidate.
        var fp16_options = options;
        fp16_options.rotated4_checkpoint = false;
        var fp16_fit = try prepare(a, input, &batches, layout, fp16_options);
        defer fp16_fit.deinit();
        try std.testing.expectEqual(try globalStorageBytes(fp16_fit.state, .f16, true), try fittedGlobalStorageBytes(input, next.state, .f16));
        const stored = next.state.donors[0].heads[0].compressed;
        try std.testing.expectEqualSlices(u8, old.compressed.keys, stored.keys[0..old.compressed.keys.len]);
        try std.testing.expectEqualSlices(u8, old.compressed.values, stored.values[0..old.compressed.values.len]);
        if (quantize) try std.testing.expectEqual(@as(usize, 2), stored.spans.len);
        const wire = try checkpoint.encodeWithEncoding(a, next.state, .{}, .rotated4, .{});
        defer a.free(wire);
        var restored = try checkpoint.decode(a, wire, identity, .{});
        restored.deinit();
        current.deinit();
        current = next;
    }
}

test "rotated4 numerical rejection preserves independently validated FP16 and prior packed ownership" {
    const a = std.testing.allocator;
    var keys: [128]f32 = undefined;
    var values: [128]f32 = @splat(2);
    for (&keys, 0..) |*k, i| k.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16;
    var stored = try (checkpoint.packed_storage.Packed{}).append(a, 0, 64, keys[0..64], values[0..64], .{});
    defer stored.deinit(a);
    try stored.decodeRow(64, 0, keys[0..64], values[0..64]);
    var result: checkpoint.Head = .{ .dim = 64, .keys = &keys, .values = &values, .bias = &.{ 0, 0 }, .compressed = stored };
    const original_keys = keys;
    var k64: [128]f64 = undefined;
    const v64: [128]f64 = @splat(2);
    for (&k64, keys) |*k, value| k.* = value;
    var candidate_keys = k64;
    var candidate_values = v64;
    const query: [64]f64 = @splat(1);
    const original: solver.Head = .{ .keys = &k64, .values = &v64, .bias = &.{ 0, 0 }, .rows = 2, .dim = 64, .scale = 0.125 };
    var compact = original;
    compact.keys = &candidate_keys;
    compact.values = &candidate_values;
    var diagnostic: CodecDiagnostic = .{};
    const accepted = try quantizeRotatedHead(a, original, compact, &result, 1, 2, &query, &query, .{ .target_rows = 1, .max_relative_mass_error = 0, .max_relative_output_rmse = 0 }, .quarter_v1, &diagnostic);
    try std.testing.expect(!accepted);
    try std.testing.expectEqual(CodecDiagnostic.Kind.rotated4, diagnostic.kind);
    try std.testing.expectEqual(CodecDiagnostic.Outcome.postcodec_heldout_gate, diagnostic.outcome);
    try std.testing.expectEqual(stored.keys.ptr, result.compressed.keys.ptr);
    try std.testing.expectEqualSlices(f32, &original_keys, result.keys);
    try std.testing.expectEqual(@as(usize, 1), result.compressed.rows());
}

fn selectiveRotatedAllocationTest(a: std.mem.Allocator) !void {
    const n = 10;
    const d = 64;
    var keys: [n * d]f32 = undefined;
    var values: [n * d]f32 = undefined;
    var original_keys: [n * d]f64 = undefined;
    var original_values: [n * d]f64 = undefined;
    const bias: [n]f64 = @splat(0);
    const bias32: [n]f32 = @splat(0);
    for (&keys, &values, &original_keys, &original_values, 0..) |*k, *v, *ok, *ov, i| {
        k.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17) % 83)) - 41)) / 16;
        v.* = @as(f32, @floatFromInt(i % 13)) / 8;
        ok.* = k.*;
        ov.* = v.*;
    }
    const saved_keys = keys;
    const saved_values = values;
    var compact_keys = original_keys;
    var compact_values = original_values;
    const train: [d]f64 = @splat(0.5);
    const heldout: [d]f64 = @splat(-0.25);
    const original: solver.Head = .{ .keys = &original_keys, .values = &original_values, .bias = &bias, .rows = n, .dim = d, .scale = 0.125 };
    var compact = original;
    compact.keys = &compact_keys;
    compact.values = &compact_values;
    // Independent scalar oracle ranks the training-query partition contribution.
    var all = try (checkpoint.packed_storage.Packed{}).append(a, 1, d, keys[d .. 9 * d], values[d .. 9 * d], .{});
    defer all.deinit(a);
    var k: [d]f32 = undefined;
    var v: [d]f32 = undefined;
    for (1..9) |row| {
        try all.decodeRow(d, row - 1, &k, &v);
        for (0..d) |j| {
            compact_keys[row * d + j] = k[j];
            compact_values[row * d + j] = v[j];
        }
    }
    var po: [n]f64 = undefined;
    var pc: [n]f64 = undefined;
    const mo = try solver.attend(original, &train, &po, null);
    const mc = try solver.attend(compact, &train, &pc, null);
    var risks: [n]f64 = @splat(-1);
    for (1..9) |row| risks[row] = @abs(pc[row] * @exp(mc - mo) - po[row]);
    var exact: [n]bool = @splat(false);
    for (0..2) |_| {
        var best: usize = 1;
        for (2..9) |row| if (risks[row] > risks[best]) {
            best = row;
        };
        exact[best] = true;
        risks[best] = -1;
    }
    compact_keys = original_keys;
    compact_values = original_values;
    var result: checkpoint.Head = .{ .dim = d, .keys = &keys, .values = &values, .bias = &bias32 };
    defer result.compressed.deinit(a);
    var diagnostic: CodecDiagnostic = .{};
    try std.testing.expect(try quantizeRotatedHead(a, original, compact, &result, 1, 9, &train, &heldout, .{ .target_rows = 1, .max_relative_mass_error = 1e6, .max_relative_output_rmse = 1e6 }, .quarter_v1, &diagnostic));
    try std.testing.expectEqual(Rotated4RiskRetention.quarter_v1, diagnostic.rotated4_risk_retention);
    try std.testing.expectEqual(@as(usize, 2), diagnostic.retained_exact_rows);
    try std.testing.expectEqual(@as(usize, 6), diagnostic.packed_rows);
    try result.compressed.validate(n, d);
    try std.testing.expectEqual(@as(usize, 6), result.compressed.rows());
    for (0..n) |row| {
        if (row == 0 or row == 9 or exact[row]) {
            try std.testing.expect(result.compressed.index(row) == .exact);
            try std.testing.expectEqualSlices(f32, saved_keys[row * d ..][0..d], result.keys[row * d ..][0..d]);
            try std.testing.expectEqualSlices(f32, saved_values[row * d ..][0..d], result.values[row * d ..][0..d]);
        } else try std.testing.expect(result.compressed.index(row) == .compressed);
    }
}

test "rotated4 selective retention follows training risk preserves exact rows and unwinds allocations" {
    try std.testing.expectEqual(@as(usize, 2), rotated4RetainedCount(8, .quarter_v1));
    try std.testing.expectEqual(@as(usize, 4), rotated4RetainedCount(8, .half_v2));
    try selectiveRotatedAllocationTest(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, selectiveRotatedAllocationTest, .{});
}

const CumulativeCodec = enum { f16, fp8, rotated4 };

fn CumulativeFixture(comptime d: usize) type {
    return struct {
        const Self = @This();
        keys: [8 * d]f32,
        values: [8 * d]f32,
        train: [d]f32,
        heldout: [d]f32,
        heads: [1]checkpoint.Head,
        donors: [1]checkpoint.Donor,
        state: checkpoint.State,
        batches: [1]ReferenceBatch,
        layout: Layout,

        fn init(self: *Self) void {
            self.keys = @splat(0);
            self.values = @splat(1);
            self.train = @splat(0);
            self.heldout = @splat(0.25);
            self.heads = .{.{
                .dim = d,
                .positions = &.{ 0, 1, 2, 3, 4, 5, 6, 7 },
                .coverage_ends = &.{ 1, 2, 3, 4, 5, 6, 7, 8 },
                .keys = &self.keys,
                .values = &self.values,
                .bias = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
            }};
            self.donors = .{.{ .layer = 0, .heads = &self.heads }};
            self.state = .{
                .identity = identity,
                .logical_length = 8,
                .tokens = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
                .logits = &.{0},
                .layer_donors = &.{0},
                .donors = &self.donors,
            };
            self.batches = .{.{
                .consumer_layer = 0,
                .query_head = 0,
                .train = &self.train,
                .heldout = &self.heldout,
                .train_positions = &.{8},
                .heldout_positions = &.{9},
            }};
            self.layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
        }
    };
}

fn cumulativeOptions(codec: CumulativeCodec) Options {
    return .{
        .admission = .cumulative_ladder_4_3_2,
        .prefix_rows = 1,
        .recent_rows = 1,
        .chunk_rows = 6,
        .retention_divisor = 4,
        .min_retention_divisor = 2,
        .rejected_head_policy = .preserve_original,
        .validation = .whole_head,
        .storage_precision = .f16,
        .fp8_checkpoint = codec == .fp8,
        .rotated4_checkpoint = codec == .rotated4,
        .independent_fp64_publication = true,
        .min_global_savings_ratio = 1.5,
        .max_transient_host_bytes = 32 << 20,
        .fit = .{
            .target_rows = 1,
            .max_workspace_bytes = 8 << 20,
            .max_relative_mass_error = 0.05,
            .max_relative_output_rmse = 0.05,
        },
    };
}

fn runCumulativeEngineFixture(allocator: std.mem.Allocator, codec: CumulativeCodec) !void {
    var fixture: CumulativeFixture(64) = undefined;
    fixture.init();
    try fixture.state.validate(checkpoint.serving_limits);
    const original_keys = fixture.keys;
    const original_values = fixture.values;
    var stats: Stats = .{};
    var options = cumulativeOptions(codec);
    options.stats = &stats;
    var result = try prepare(allocator, fixture.state, &fixture.batches, fixture.layout, options);
    defer result.deinit();
    const head = result.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(usize, 4), head.positions.len);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 7 }, head.positions);
    try std.testing.expectEqualSlices(u64, &.{ 1, 7, 7, 8 }, head.coverage_ends);
    try std.testing.expectEqual(@as(u64, 7), head.compacted_until);
    try std.testing.expect(stats.actual_global_savings_ratio > 1.5);
    try std.testing.expectEqualSlices(f32, &original_keys, &fixture.keys);
    try std.testing.expectEqualSlices(f32, &original_values, &fixture.values);
    switch (codec) {
        .f16 => try std.testing.expectEqual(@as(usize, 0), stats.fp8_heads + stats.rotated4_heads),
        .fp8 => try std.testing.expectEqual(@as(usize, 1), stats.fp8_heads + stats.fp8_fallback_heads),
        .rotated4 => if (@import("turboquant_rotated.zig").supported(64))
            try std.testing.expectEqual(@as(usize, 1), stats.rotated4_heads + stats.rotated4_fallback_heads),
    }
}

fn cumulativeAllocationFailure(allocator: std.mem.Allocator) !void {
    try runCumulativeEngineFixture(allocator, .f16);
}

test "cumulative prepare dynamically compacts D64 source through common f16 codec tail" {
    try runCumulativeEngineFixture(std.testing.allocator, .f16);
}

test "cumulative prepare routes admitted D64 source through common fp8 and rotated4 codec tails" {
    try runCumulativeEngineFixture(std.testing.allocator, .fp8);
    try runCumulativeEngineFixture(std.testing.allocator, .rotated4);
}

test "cumulative prepare unwinds every allocation failure and preserves source" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, cumulativeAllocationFailure, .{});
}

test "cumulative prepare cancellation preserves source" {
    var fixture: CumulativeFixture(64) = undefined;
    fixture.init();
    try fixture.state.validate(checkpoint.serving_limits);
    const original_keys = fixture.keys;
    const original_values = fixture.values;
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.CompactionCancelled;
        }
    };
    var options = cumulativeOptions(.f16);
    options.fit.control = .{ .check_fn = Cancel.check };
    try std.testing.expectError(error.CompactionCancelled, prepare(std.testing.allocator, fixture.state, &fixture.batches, fixture.layout, options));
    try std.testing.expectEqualSlices(f32, &original_keys, &fixture.keys);
    try std.testing.expectEqualSlices(f32, &original_values, &fixture.values);
}

test "cumulative prepare rejecting global 1.5 savings preserves source" {
    var fixture: CumulativeFixture(1) = undefined;
    fixture.init();
    try fixture.state.validate(checkpoint.serving_limits);
    const original_keys = fixture.keys;
    const original_values = fixture.values;
    // D1 metadata makes even a four-row candidate larger than the
    // metadata-free eight-row source at the real 1.5 gate.
    try std.testing.expectError(error.CompactionSavingsInsufficient, prepare(std.testing.allocator, fixture.state, &fixture.batches, fixture.layout, cumulativeOptions(.f16)));
    try std.testing.expectEqualSlices(f32, &original_keys, &fixture.keys);
    try std.testing.expectEqualSlices(f32, &original_values, &fixture.values);
}

test "cumulative workspace planning charges full source conversion before backend execution" {
    const d = 64;
    const rows = 32 * 1024;
    const queries = 512 * d;
    var options = cumulativeOptions(.f16);
    options.chunk_rows = 1024;
    options.fit.max_workspace_bytes = 512 << 20;
    const planned = try headWorkspaceBytes(rows, rows, 1024, queries, queries, d, options);
    const ranges = std.math.divCeil(usize, rows, options.chunk_rows) catch unreachable;
    const outer = rows * (2 * d + 1) * @sizeOf(f64) + ranges * @sizeOf(cumulative_cache.Range);
    try std.testing.expect(outer > 32 << 20);
    try std.testing.expect(planned.bytes >= outer + options.fit.max_workspace_bytes);
}

test "cumulative prepare repeated cycle preserves prior fitted rows and fp8 payload" {
    const a = std.testing.allocator;
    const d = 64;
    var first_fixture: CumulativeFixture(d) = undefined;
    first_fixture.init();
    try first_fixture.state.validate(checkpoint.serving_limits);
    var first = try prepare(a, first_fixture.state, &first_fixture.batches, first_fixture.layout, cumulativeOptions(.fp8));
    defer first.deinit();
    const old = first.state.donors[0].heads[0];
    try std.testing.expectEqual(@as(usize, 4), old.positions.len);
    try std.testing.expect(old.fp8.rows != 0);

    var positions: [10]u64 = undefined;
    var coverage: [10]u64 = undefined;
    var keys: [10 * d]f32 = undefined;
    var values: [10 * d]f32 = undefined;
    var bias: [10]f32 = undefined;
    @memcpy(positions[0..4], old.positions);
    @memcpy(coverage[0..4], old.coverage_ends);
    @memcpy(keys[0 .. 4 * d], old.keys);
    @memcpy(values[0 .. 4 * d], old.values);
    @memcpy(bias[0..4], old.bias);
    for (4..10) |row| {
        positions[row] = 8 + row - 4;
        coverage[row] = positions[row] + 1;
        bias[row] = 0;
        @memset(keys[row * d ..][0..d], 0);
        @memset(values[row * d ..][0..d], 1);
    }
    const protected_positions = positions[0..3].*;
    const protected_coverage = coverage[0..3].*;
    const protected_keys = keys[0 .. 3 * d].*;
    const protected_values = values[0 .. 3 * d].*;
    const protected_bias = bias[0..3].*;
    const prior_fp8 = old.fp8;

    const appended: checkpoint.Head = .{
        .dim = d,
        .compacted_until = old.compacted_until,
        .positions = &positions,
        .coverage_ends = &coverage,
        .keys = &keys,
        .values = &values,
        .bias = &bias,
        .fp8 = old.fp8,
    };
    const source: checkpoint.State = .{
        .identity = identity,
        .logical_length = 14,
        .tokens = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .heads = &.{appended} }},
    };
    try source.validate(checkpoint.serving_limits);
    const train: [d]f32 = @splat(0);
    const heldout: [d]f32 = @splat(0.25);
    const batches = [_]ReferenceBatch{.{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &train,
        .heldout = &heldout,
        .train_positions = &.{14},
        .heldout_positions = &.{15},
    }};
    var second_options = cumulativeOptions(.fp8);
    second_options.source_has_metadata = true;
    var second = try prepare(a, source, &batches, first_fixture.layout, second_options);
    defer second.deinit();
    const current = second.state.donors[0].heads[0];

    // Coverage <= the prior compacted boundary is an immutable fitted prefix.
    try std.testing.expectEqualSlices(u64, &protected_positions, current.positions[0..3]);
    try std.testing.expectEqualSlices(u64, &protected_coverage, current.coverage_ends[0..3]);
    try std.testing.expectEqualSlices(f32, &protected_keys, current.keys[0 .. 3 * d]);
    try std.testing.expectEqualSlices(f32, &protected_values, current.values[0 .. 3 * d]);
    try std.testing.expectEqualSlices(f32, &protected_bias, current.bias[0..3]);
    try std.testing.expectEqual(prior_fp8.first_row, current.fp8.first_row);
    try std.testing.expectEqual(prior_fp8.key_scale, current.fp8.key_scale);
    try std.testing.expectEqual(prior_fp8.value_scale, current.fp8.value_scale);
    try std.testing.expect(current.compacted_until > old.compacted_until);
    try std.testing.expectEqualSlices(u64, &.{ 0, 1, 2, 7, 8, 13 }, current.positions);
    // Borrowed second-cycle input remains unchanged after publication.
    try std.testing.expectEqualSlices(u64, &protected_positions, positions[0..3]);
    try std.testing.expectEqualSlices(f32, &protected_keys, keys[0 .. 3 * d]);
    try std.testing.expectEqualSlices(f32, &protected_bias, bias[0..3]);
}

test "critical025 optimized tuple is sealed and leaves calibrated tag unchanged" {
    const allowed: Options = .{
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .adaptive_v18_train_probability_census = true,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .max_queries_per_consumer = 768,
        .storage_precision = .f16,
    };
    try std.testing.expect(privatePerformancePolicyValid(allowed) and criticalCorePolicyValid(allowed));
    var invalid = allowed;
    invalid.adaptive_v18_train_probability_census = false;
    try std.testing.expect(!criticalCorePolicyValid(invalid));
    invalid = allowed;
    invalid.adaptive_v18_postfit_policy = .disabled;
    try std.testing.expect(!criticalCorePolicyValid(invalid));
    invalid = allowed;
    invalid.adaptive_v18_rejected_detail_diagnostic = true;
    try std.testing.expect(!criticalCorePolicyValid(invalid));
    invalid = allowed;
    invalid.adaptive_v18_geometry_route = .bounded_page_over_96;
    try std.testing.expect(!privatePerformancePolicyValid(invalid) and !criticalCorePolicyValid(invalid));
    invalid = allowed;
    invalid.adaptive_v18_max_budget_fallbacks = 4;
    try std.testing.expect(!criticalCorePolicyValid(invalid));
    invalid = allowed;
    invalid.private_performance_policy = .owned_capture_summary_incremental_reduction_v1;
    try std.testing.expect(!criticalCorePolicyValid(invalid));
}

test "TRAIN primary policy comparison admits only sealed p91 p95 max3 tuple" {
    var options: Options = .{
        .adaptive_v18_train_primary_policy_comparison = true,
        .fit = .{ .target_rows = 1, .max_workspace_bytes = 1024 * 1024 * 1024 },
        .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .adaptive_v18_postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = .flat_gram_only,
        .adaptive_v18_primary_budget_percent = 91,
        .adaptive_v18_fallback_budget_percent = 95,
        .adaptive_v18_max_budget_fallbacks = 3,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .storage_precision = .f16,
        .private_performance_policy = .owned_capture_summary_incremental_reduction_critical025_v1,
        .geometry_chooser_policy = .incremental_gram_v1,
        .independent_validation_policy = .accelerate_fp64_summary_v1,
        .adaptive_v18_train_probability_census = true,
    };
    try std.testing.expect(criticalCorePolicyValid(options));
    const saved = options;
    options.adaptive_v18_train_primary_policy_cross_validation = true;
    try std.testing.expect(criticalCorePolicyValid(options));
    options.adaptive_v18_train_primary_policy_comparison = false;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options = saved;
    options.adaptive_v18_critical_core_policy = .disabled;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options = saved;
    options.fit.max_relative_mass_error = 0.1;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options = saved;
    options.adaptive_v18_primary_budget_percent = 95;
    try std.testing.expect(!criticalCorePolicyValid(options));
    options.adaptive_v18_primary_budget_percent = 91;
    options.adaptive_v18_rejected_detail_diagnostic = true;
    try std.testing.expect(!criticalCorePolicyValid(options));
}

test "TRAIN primary policy CV cannot activate outside comparison" {
    var options: Options = .{ .adaptive_v18_train_primary_policy_cross_validation = true };
    try std.testing.expect(!criticalCorePolicyValid(options));
    options.adaptive_v18_train_primary_policy_comparison = true;
    try std.testing.expect(!criticalCorePolicyValid(options)); // The rest of the sealed tuple is still mandatory.
}
