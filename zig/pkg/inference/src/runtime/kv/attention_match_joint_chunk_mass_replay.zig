// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only whole-branch replay. Every branch freezes all selections
//! from training data, performs one joint mass correction, refits values with
//! the corrected weights fixed, rounds the complete candidate, and only then
//! consults heldout data once. No heldout result feeds a fit within a branch.
const std = @import("std");
const solver = @import("attention_match.zig");
const selection = @import("attention_match_global_selection_diagnostic.zig");
const critical_adapter = @import("attention_match_critical_core_adapter.zig");
const group_mass_postfit = @import("attention_match_train_group_mass_postfit.zig");
const group_mass_lawson = @import("attention_match_train_group_mass_lawson.zig");
const critical_keys = @import("attention_match_train_critical_source_keys_streaming.zig");
const calibrated_critical = @import("attention_match_train_budget_calibrated_critical_threshold.zig");
const calibrated_critical_v2 = @import("attention_match_train_budget_calibrated_critical_threshold_reserve16.zig");
const joint = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const training_source = @import("attention_match_joint_training_source.zig");
pub const TrainSnapshotDomain = enum(u8) { primary = 0, historical = 1, future = 2 };
pub const TrainSnapshotDomainRun = struct { domain: TrainSnapshotDomain, start: usize, count: usize };
const replay_planner = @import("attention_match_joint_replay_planner.zig");
const anchored = @import("attention_match_joint_chunk_mass_anchored.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");
const gram_geometry = @import("attention_match_joint_gram_geometry.zig");
const gram_delta = @import("attention_match_gram_delta_accelerate.zig");
const gram_delta_page = @import("attention_match_gram_delta_page_adapter.zig");
const train_reduction = @import("attention_match_training_reduction_accelerate.zig");
const local_weight_cap = @import("attention_match_local_weight_cap.zig");
const platform = @import("antfly_platform");
const metal_fit = @import("../../ops/metal_attention_match_fit.zig");

pub const GeometryStorage = replay_planner.GeometryStorage;
pub const CriticalCorePolicy = critical_adapter.Policy;
pub const criticalCorePolicyDigest = critical_adapter.policyDigest;

pub const GramMemoEvidence = struct {
    enabled: bool = false,
    bytes: usize = 0,
    hits: usize = 0,
    misses: usize = 0,
};

pub const GramDeltaEvidence = struct {
    requested: bool = false,
    accelerated: bool = false,
    tiles: usize = 0,
    tile_rows: usize = 0,
    backend_calls: usize = 0,
    packed_bytes_peak: usize = 0,
    staged_bytes_peak: usize = 0,
    page_bytes_peak: usize = 0,
    fallbacks: usize = 0,
};
pub const JointFixedMemoEvidence = struct {
    enabled: bool = false,
    bytes: usize = 0,
    builds: usize = 0,
    hits: usize = 0,
};
pub const TrainingReductionPolicy = enum { scalar_reference, accelerate_fp64_cached_v1 };
pub const TrainingReductionEvidence = struct {
    policy: TrainingReductionPolicy = .scalar_reference,
    accelerated: bool = false,
    local_original_calls: usize = 0,
    rounded_harm_calls: usize = 0,
    tiles: usize = 0,
    backend_calls: usize = 0,
    workspace_bytes_peak: usize = 0,
};

/// DEV-only wall-clock attribution. Null leaves the default path free of clock
/// reads. Counters are inclusive phase totals and are not qualification data.
pub const PhaseTiming = struct {
    cache_prepare_ns: u64 = 0,
    local_original_rho_y_ns: u64 = 0,
    selection_freeze_ns: u64 = 0,
    joint_value_fit_ns: u64 = 0,
    rounded_harm_ns: u64 = 0,
    gram_construction_ns: u64 = 0,
    chooser_ns: u64 = 0,
    final_heldout_ns: u64 = 0,
};

const TimedPhase = enum { cache_prepare, local_original_rho_y, selection_freeze, joint_value_fit, rounded_harm, gram_construction, chooser, final_heldout };

const BranchValidationConfigs = struct { heldout: solver.Config, direct_train: solver.Config };

pub const PageBackedValidation = struct {
    backend: ?solver.Backend = null,
    train_reference_summary_backend: ?solver.ReferenceSummaryBackend = null,
    heldout_reference_summary_backend: ?solver.ReferenceSummaryBackend = null,
};

fn branchValidationConfigs(cfg: solver.Config, backend: ?solver.Backend, observe_detail: bool, train_reference: ?solver.ReferenceSummaryBackend, heldout_reference: ?solver.ReferenceSummaryBackend) !BranchValidationConfigs {
    if (observe_detail and backend != null) return error.InvalidFitConfig;
    var heldout = cfg;
    heldout.backend = backend;
    heldout.reference_summary_backend = heldout_reference;
    var direct_train = heldout;
    direct_train.reference_summary_backend = train_reference;
    direct_train.backend = null;
    return .{ .heldout = heldout, .direct_train = direct_train };
}

test "direct TRAIN observer remains scalar when heldout summary acceleration is enabled" {
    const Counter = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn summary(raw: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (targets.len != queries.len or masses.len != queries.len / head.dim) return error.InvalidAttentionShape;
            @memset(targets, 0);
            @memset(masses, 0);
        }
    };
    var counter: Counter = .{};
    const backend: solver.Backend = .{ .validation_batch = .rows_256, .context = &counter, .workspace_fn = Counter.workspace, .summary_fn = Counter.summary };
    const configs = try branchValidationConfigs(.{ .target_rows = 1 }, backend, false, null, null);
    try std.testing.expect(configs.heldout.backend != null);
    try std.testing.expect(configs.direct_train.backend == null);
    const head: solver.Head = .{ .keys = &.{1}, .values = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    _ = try solver.validationMetrics(std.testing.allocator, head, head, &.{ 1, 2 }, configs.direct_train);
    try std.testing.expectEqual(@as(usize, 0), counter.calls);
    _ = try solver.validationMetrics(std.testing.allocator, head, head, &.{ 1, 2 }, configs.heldout);
    try std.testing.expectEqual(@as(usize, 2), counter.calls);
}

fn phaseStart(timing: ?*PhaseTiming) u64 {
    return if (timing != null) platform.time.monotonicNs() else 0;
}

fn phaseEnd(timing: ?*PhaseTiming, phase: TimedPhase, started: u64) void {
    const out = timing orelse return;
    const elapsed = platform.time.monotonicNs() -| started;
    switch (phase) {
        .cache_prepare => out.cache_prepare_ns +|= elapsed,
        .local_original_rho_y => out.local_original_rho_y_ns +|= elapsed,
        .selection_freeze => out.selection_freeze_ns +|= elapsed,
        .joint_value_fit => out.joint_value_fit_ns +|= elapsed,
        .rounded_harm => out.rounded_harm_ns +|= elapsed,
        .gram_construction => out.gram_construction_ns +|= elapsed,
        .chooser => out.chooser_ns +|= elapsed,
        .final_heldout => out.final_heldout_ns +|= elapsed,
    }
}

const CriticalAdmissionPhase = enum {
    initial_plan,
    cache_prepare,
    critical_extract,
    critical_replan,
    score_bank,
    gram_geometry,
    final_replay,
};

const CriticalPhaseDiagnostic = struct {
    enabled: bool,
    phase: CriticalAdmissionPhase = .initial_plan,
    query_rows: usize,
    source_rows: usize,
    dim: usize,
    range_count: usize,
    external_bytes: usize,
    combined_cap: usize,
    cache_bytes: usize = 0,
    authority_bytes: usize = 0,
    critical_rows: usize = 0,
    target_max: usize = 0,
    residual_max: usize = 0,
    consumer_peak: usize = 0,
    scoring_peak: usize = 0,
    geometry_peak: usize = 0,
    final_peak: usize = 0,

    fn setPlan(self: *@This(), plan: replay_planner.AdaptivePlan) void {
        if (!self.enabled) return;
        self.consumer_peak = plan.consumer_peak_bytes;
        self.scoring_peak = plan.scoring_peak_bytes;
        self.geometry_peak = plan.signed_geometry_peak_bytes;
        self.final_peak = plan.final_replay_peak_bytes;
    }

    fn fail(self: *const @This(), err: anyerror) anyerror {
        if (self.enabled) std.debug.print(
            "attention_match_critical_phase_failure phase={s} error={s} q={} n={} d={} ranges={} external_bytes={} cache_bytes={} authority_bytes={} combined_cap={} consumer_peak={} scoring_peak={} geometry_peak={} final_peak={} critical_rows={} target_max={} residual_max={}\n",
            .{ @tagName(self.phase), @errorName(err), self.query_rows, self.source_rows, self.dim, self.range_count, self.external_bytes, self.cache_bytes, self.authority_bytes, self.combined_cap, self.consumer_peak, self.scoring_peak, self.geometry_peak, self.final_peak, self.critical_rows, self.target_max, self.residual_max },
        );
        return err;
    }
};

fn criticalPhaseDiagnosticEnabled(options: Options) bool {
    return options.observe_critical_phase_admission or platform.env.getenvBool("ANTFLY_METAL_QUALITY_CRITICAL_PHASE_ADMISSION_DIAGNOSTIC");
}

test "critical phase admission diagnostic defaults off and preserves error identity" {
    try std.testing.expect(!(@as(Options, .{})).observe_critical_phase_admission);
    const diagnostic: CriticalPhaseDiagnostic = .{
        .enabled = false,
        .query_rows = 1,
        .source_rows = 1,
        .dim = 1,
        .range_count = 1,
        .external_bytes = 0,
        .combined_cap = 1,
    };
    try std.testing.expect(diagnostic.fail(error.CompactionWorkspaceLimit) == error.CompactionWorkspaceLimit);
}

fn cachedProbabilityCopy(raw: ?*anyopaque, query: usize, start: usize, out: []f64) !void {
    const source: *const training_source.Source = @ptrCast(@alignCast(raw.?));
    try source.copyChunk(query, start, out);
}

const CriticalProbabilityContext = struct {
    source: *const training_source.Source,
    fn copy(raw: ?*anyopaque, query: usize, out: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        try control.check();
        if (out.len != self.source.rows) return error.InvalidAttentionShape;
        try self.source.copyChunk(query, 0, out);
    }
};

fn extractCriticalAuthorityThreshold(allocator: std.mem.Allocator, source: *const training_source.Source, ranges: []const admission.Range, threshold: f64, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !critical_keys.Result {
    var context: CriticalProbabilityContext = .{ .source = source };
    const critical_ranges = try allocator.alloc(critical_keys.Range, ranges.len);
    defer allocator.free(critical_ranges);
    for (ranges, critical_ranges) |range, *out| out.* = .{ .start = range.start, .rows = range.rows };
    return critical_keys.extractStreaming(allocator, .{ .context = &context, .query_rows = source.query_count, .complete_query_rows = source.query_count, .source_rows = source.rows, .external_live_bytes = external_live_bytes, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened, .copy_query_fn = CriticalProbabilityContext.copy }, critical_ranges, .{ .threshold = threshold, .combined_cap = combined_cap, .control = control });
}

fn extractCriticalAuthority(allocator: std.mem.Allocator, source: *const training_source.Source, ranges: []const admission.Range, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !critical_keys.Result {
    return extractCriticalAuthorityThreshold(allocator, source, ranges, calibrated_critical.ceiling_probability, external_live_bytes, combined_cap, control);
}

fn criticalCount(authority: ?*const critical_keys.Result, range: admission.Range) !usize {
    const value = authority orelse return 0;
    for (value.inner.ranges) |entry| if (entry.start == range.start and entry.rows == range.rows) return entry.key_count;
    return error.InvalidAttentionShape;
}

fn criticalResidualReserve(policy: CriticalCorePolicy, rows: usize) !usize {
    if (policy == .disabled) return 0;
    if (rows < 2) return error.InvalidAttentionShape;
    return switch (policy) {
        .disabled => 0,
        .train_probability_gte_0_025_no_top1_v1, .train_probability_budget_calibrated_floor_0_0025_p91_v1 => 1,
        .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => @min(@as(usize, 16), rows - 1),
    };
}

fn criticalAwareTargets(rows: usize, core_count: usize, policy: CriticalCorePolicy) ![4]usize {
    if (rows == 0 or core_count > rows) return error.InvalidAttentionShape;
    const reserve = try criticalResidualReserve(policy, rows);
    if (reserve != 0 and (core_count > rows - reserve or core_count + reserve >= rows)) return error.CompactionFitRejected;
    const minimum = if (reserve != 0) core_count + reserve else 0;
    return .{
        @min(rows, @max(minimum, std.math.divCeil(usize, rows, 4) catch return error.InvalidAttentionShape)),
        @min(rows, @max(minimum, std.math.divCeil(usize, rows, 3) catch return error.InvalidAttentionShape)),
        @min(rows, @max(minimum, std.math.divCeil(usize, rows, 2) catch return error.InvalidAttentionShape)),
        rows,
    };
}

test "critical core raises compacted levels monotonically and rejects implicit identity" {
    try std.testing.expectEqual([4]usize{ 93, 93, 128, 256 }, try criticalAwareTargets(256, 92, .train_probability_gte_0_025_no_top1_v1));
    try std.testing.expectEqual([4]usize{ 64, 86, 128, 256 }, try criticalAwareTargets(256, 0, .disabled));
    try std.testing.expectError(error.CompactionFitRejected, criticalAwareTargets(256, 255, .train_probability_gte_0_025_no_top1_v1));
    try std.testing.expectEqual([4]usize{ 108, 108, 128, 256 }, try criticalAwareTargets(256, 92, .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2));
    try std.testing.expectEqual(@as(usize, 7), try criticalResidualReserve(.train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2, 8));
    try std.testing.expectEqual([4]usize{ 1, 1, 1, 1 }, try criticalAwareTargets(1, 0, .disabled));
}

fn mergeReductionEvidence(out: *TrainingReductionEvidence, value: train_reduction.Evidence, local_original: bool) !void {
    if (!value.accelerated) return error.InvalidFitConfig;
    out.accelerated = true;
    if (local_original) out.local_original_calls = try std.math.add(usize, out.local_original_calls, 1) else out.rounded_harm_calls = try std.math.add(usize, out.rounded_harm_calls, 1);
    out.tiles = try std.math.add(usize, out.tiles, value.tiles);
    out.backend_calls = try std.math.add(usize, out.backend_calls, value.backend_calls);
    out.workspace_bytes_peak = @max(out.workspace_bytes_peak, value.workspace_bytes);
}

pub const Options = struct {
    phase_timing: ?*PhaseTiming = null,
    /// DEV-only failure provenance. It observes admission state and never
    /// changes a bound, allocation, numerical path, or returned error.
    observe_critical_phase_admission: bool = false,
    /// DEV-only bound applied before candidate value fitting and geometry scoring.
    max_local_weight: ?f64 = null,
    training_source: bool = false,
    /// Diagnostic-only atom ranking. Raw preserves the established route.
    selection_score: selection.SelectionScore = .raw,
    /// Null preserves the established score mapping byte-for-byte.
    selection_policy: ?selection.SelectionPolicy = null,
    /// Production-private TRAIN authority. Disabled preserves legacy bytes.
    critical_core_policy: CriticalCorePolicy = .disabled,
    postfit_policy: PostfitPolicy = .disabled,
    postfit_workspace_bytes: usize = 0,
    candidate_postfit_hook: ?CandidatePostfitHook = null,
    mass_strategy: joint.MassStrategy = .global_active_nnls_v16,
    /// DEV-only final refit after v18 has frozen support and geometry.
    final_refit_strategy: FinalRefitStrategy = .identity_reuse,
    retain_training_diagnostics: bool = false,
    adaptive_targets_by_branch: ?[3][]const usize = null,
    adaptive_geometry: ?*[3]adaptive.Geometry = null,
    scored_candidates: ?*const AdaptivePlan = null,
    geometry_policy: []const u8 = adaptive.policy,
    observe_direct_train: bool = false,
    observe_validation_detail: bool = false,
    /// Builtin-test-only synchronous observation of the transient divisor-2
    /// candidate after its scalar/authorized HHO gate rejects it. The view is
    /// borrowed and valid only for the callback. Null is the production path.
    rejected_branch_capture: ?RejectedBranchCapture = null,
    /// Diagnostic-only census of TRAIN maximum probability by source row.
    observe_train_probability_census: bool = false,
    validation_backend: ?solver.Backend = null,
    train_reference_summary_backend: ?solver.ReferenceSummaryBackend = null,
    heldout_reference_summary_backend: ?solver.ReferenceSummaryBackend = null,
    geometry_storage: GeometryStorage = .dense,
    gram_tile_rows: usize = 8,
    gram_backend: ?gram_geometry.GramBackend = null,
    /// DEV-only numerical FP64 GEMM numerator delta. Scalar remains default.
    gram_delta_backend: ?gram_delta.Backend = null,
    gram_delta_require_accelerated: bool = false,
    /// DEV-only exact memo for the common negative original-range term used
    /// by the three Gram candidates for one `(range, query tile)`.
    gram_memoize_original_delta: bool = false,
    /// DEV-only exact memo of original rho/residual reused by d4/d3/d2 fits.
    joint_fixed_original_memo: bool = false,
    /// DEV-only TRAIN Gram budget-preserving downgrade/upgrade refinement.
    train_budget_exchange: bool = false,
    /// DEV-only divisor-2 TRAIN geometry budget. 100 preserves legacy bytes.
    geometry_budget_percent: u8 = 100,
    /// Optional second geometry/final-validation attempt over the same TRAIN
    /// cache and scored packets. Authorization remains caller-owned.
    fallback_geometry_budget_percent: u8 = 0,
    fallback_authorize_context: ?*anyopaque = null,
    fallback_authorize_fn: ?*const fn (?*anyopaque) bool = null,
    /// DEV-only exact-Gram chooser implementation. The default retains the
    /// established full objective scan and digest byte-for-byte.
    geometry_chooser_policy: GeometryChooserPolicy = .legacy_full_scan,
    geometry_workspace_bound_bytes: usize = 0,
    probability_storage: training_source.ProbabilityStorage = .f64,
    /// Private flat-cache capture geometry. Legacy/default remains 16.
    training_source_query_batch_rows: usize = 16,
    training_capture_backend: ?training_source.CaptureBackend = null,
    training_cache_allocation_authority: ?training_source.AllocationAuthority = null,
    flat_batch_policy: FlatBatchPolicy = .established,
    /// DEV-only regrouped FP64 reductions over cached TRAIN probabilities.
    /// Scalar loops remain the byte-compatible default.
    training_reduction_policy: TrainingReductionPolicy = .scalar_reference,
    training_reduction_query_batch_rows: usize = 64,
    /// DEV-only TRAIN snapshot hook. Heldout is intentionally absent.
    train_snapshot_context: ?*anyopaque = null,
    train_snapshot_fn: ?*const fn (?*anyopaque, solver.Head, *const training_source.Source, *const AdaptivePlan, solver.Control) anyerror![32]u8 = null,
    train_snapshot_v2_domain_runs: []const TrainSnapshotDomainRun = &.{},
    train_snapshot_v2_fn: ?*const fn (?*anyopaque, []const TrainSnapshotDomainRun, solver.Head, *const training_source.Source, *const AdaptivePlan, solver.Control) anyerror![32]u8 = null,
    /// Diagnostic TRAIN-only primary probe. The heldout value slice must be
    /// empty; only its authenticated query-row count participates in planner
    /// admission. The hook runs after p91 geometry freezes and before final
    /// replay, fallback authorization, or any HHO validation.
    primary_train_probe_context: ?*anyopaque = null,
    primary_train_probe_heldout_query_rows: usize = 0,
    primary_train_probe_fn: ?*const fn (?*anyopaque, std.mem.Allocator, solver.Head, []const f64, []const admission.Range, *training_source.Source, *const AdaptivePlan, solver.Config, usize, usize) anyerror!void = null,
};

pub const RejectedBranchView = struct {
    divisor: usize,
    original: solver.Head,
    candidate: solver.Head,
    source_indices: []const usize,
    coverage_source_indices: []const usize,
    train_queries: []const f64,
    heldout_queries: []const f64,
    metrics: solver.Metrics,
};

pub const RejectedBranchCapture = struct {
    context: ?*anyopaque,
    capture_fn: *const fn (?*anyopaque, RejectedBranchView) void,
};

fn observeRejectedBranch(options: Options, divisor: usize, pass: bool, original: solver.Head, candidate: *const Candidate, train: []const f64, heldout: []const f64, metrics: solver.Metrics) void {
    if (pass or divisor != 2) return;
    if (options.rejected_branch_capture) |capture| capture.capture_fn(capture.context, .{
        .divisor = divisor,
        .original = original,
        .candidate = candidate.head,
        .source_indices = candidate.source_indices,
        .coverage_source_indices = candidate.coverage_source_indices,
        .train_queries = train,
        .heldout_queries = heldout,
        .metrics = metrics,
    });
}

fn primaryProbeHeldoutRows(options: Options, heldout: []const f64, dim: usize) !usize {
    if (dim == 0 or heldout.len % dim != 0) return error.InvalidFitConfig;
    if (options.primary_train_probe_fn == null) {
        if (heldout.len == 0 or options.primary_train_probe_heldout_query_rows != 0 or options.primary_train_probe_context != null) return error.InvalidFitConfig;
        return heldout.len / dim;
    }
    if (heldout.len != 0 or options.primary_train_probe_heldout_query_rows == 0 or options.geometry_budget_percent != 91 or options.fallback_geometry_budget_percent != 0 or options.fallback_authorize_fn != null or options.fallback_authorize_context != null or options.train_snapshot_fn != null or options.train_snapshot_v2_fn != null)
        return error.InvalidFitConfig;
    return options.primary_train_probe_heldout_query_rows;
}

pub const PostfitPolicy = enum { disabled, train_global_residual_range_mass_l2_exact_f32_cache_v1, train_global_residual_range_mass_lawson_exact_rounded_max_v1, train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 };
pub fn postfitPolicyDigest(policy_value: PostfitPolicy) [32]u8 {
    return switch (policy_value) {
        .disabled => @splat(0),
        .train_global_residual_range_mass_l2_exact_f32_cache_v1 => group_mass_postfit.policy_digest,
        .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => group_mass_lawson.policy_digest,
        .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => @import("attention_match_train_l2_surrogate_minimax_postfit.zig").policy_digest,
    };
}
pub const PostfitEvidence = struct {
    policy: PostfitPolicy = .disabled,
    policy_digest: [32]u8 = @splat(0),
    attempts: usize = 0,
    completions: usize = 0,
    nnls_sweeps: usize = 0,
    max_abs_train_before: f64 = 0,
    max_abs_train_after: f64 = 0,
    workspace_bytes: usize = 0,
    refinement_attempted_iterations: usize = 0,
    refinement_accepted_improvements: usize = 0,
    refinement_nnls_sweeps: usize = 0,
    refinement_outcome: group_mass_postfit.LawsonOutcome = .disabled,
    minimax_attempts: usize = 0,
    minimax_completions: usize = 0,
    minimax_applied: usize = 0,
    minimax_scale: f64 = 1,
    minimax_scale_min: f64 = 1,
    minimax_scale_max: f64 = 1,
    exact_train_mass_before: f64 = 0,
    exact_train_mass_after: f64 = 0,
    exact_train_output_rmse_before: f64 = 0,
    exact_train_output_rmse_after: f64 = 0,
};
pub const CandidatePostfitHook = struct {
    context: ?*anyopaque,
    apply_fn: *const fn (?*anyopaque, std.mem.Allocator, solver.Head, *Candidate, []const admission.Range, FrozenOwned, solver.Config) anyerror!PostfitEvidence,
};
pub fn postfitWorkspaceReserve(query_rows: usize, candidate_rows: usize, maximum_core_rows: usize, group_count: usize) !usize {
    var bytes = try group_mass_postfit.workspaceBytes(query_rows, candidate_rows, candidate_rows, group_count);
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, group_count, @sizeOf(group_mass_postfit.Range)));
    return std.math.add(usize, bytes, try std.math.mul(usize, maximum_core_rows, @sizeOf(usize)));
}
pub fn postfitLawsonWorkspaceReserve(query_rows: usize, candidate_rows: usize, maximum_core_rows: usize, group_count: usize) !usize {
    var bytes = try group_mass_postfit.workspaceBytesLawson(query_rows, candidate_rows, candidate_rows, group_count);
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, group_count, @sizeOf(group_mass_postfit.Range)));
    return std.math.add(usize, bytes, try std.math.mul(usize, maximum_core_rows, @sizeOf(usize)));
}

pub const GeometryChooserPolicy = enum { legacy_full_scan, incremental_gram_v1 };

/// The established route remains 16/64. The diagnostic policy is deliberately
/// not inferred from a row count: capture and Gram must move as one admitted
/// pair, and callers must retain its evidence as non-qualification data.
pub const FlatBatchPolicy = enum {
    established,
    diagnostic_capture_gram_256_fallback_64_v1,
};

pub const FinalRefitStrategy = enum { identity_reuse, global_active_nnls_v16 };
pub const FinalRefitEvidence = struct {
    strategy: FinalRefitStrategy = .identity_reuse,
    attempts: usize = 0,
    completions: usize = 0,
};

fn hashU64Le(hasher: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hasher.update(&bytes);
}

fn hashExactCoreExtension(hasher: *std.crypto.hash.sha2.Sha256, chunk: joint.FrozenChunk) !void {
    if (chunk.exact_core_local.len == 0) return;
    hasher.update("antfly.frozen-scored-support-bank.exact-core.v1\x00");
    hashU64Le(hasher, chunk.exact_core_local.len);
    for (chunk.exact_core_local, 0..) |local, i| {
        if (local >= chunk.rows or (i != 0 and chunk.exact_core_local[i - 1] >= local)) return error.InvalidAttentionShape;
        const selected_index = std.mem.indexOfScalar(usize, chunk.selected_local, local) orelse return error.InvalidAttentionShape;
        if (selected_index >= chunk.weights.len or chunk.weights[selected_index] != 1) return error.InvalidAttentionShape;
        hashU64Le(hasher, local);
    }
}

fn frozenSupportBankDigest(plan: *const AdaptivePlan) ![32]u8 {
    if (plan.scores.len != plan.scored.len) return error.InvalidAttentionShape;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.frozen-scored-support-bank.v1\x00");
    hashU64Le(&h, plan.scores.len);
    for (plan.scores, plan.scored) |score, levels| {
        hashU64Le(&h, score.start);
        hashU64Le(&h, score.rows);
        for (levels, 0..) |candidate, level| {
            const owned = candidate orelse return error.InvalidAttentionShape;
            if (owned.frozen.chunks.len != 1 or owned.frozen.ranges.len != 1) return error.InvalidAttentionShape;
            const chunk = owned.frozen.chunks[0];
            const range = owned.frozen.ranges[0];
            if (range.start != score.start or range.rows != score.rows or chunk.start != score.start or chunk.rows != score.rows or
                chunk.selected_local.len != chunk.weights.len or chunk.selected_local.len != score.target_rows[level])
                return error.InvalidAttentionShape;
            hashU64Le(&h, level);
            hashU64Le(&h, chunk.selected_local.len);
            for (chunk.selected_local, chunk.weights) |local, weight| {
                if (local >= chunk.rows or !std.math.isFinite(weight) or weight < 0) return error.InvalidAttentionShape;
                hashU64Le(&h, local);
                hashU64Le(&h, @bitCast(weight));
            }
            try hashExactCoreExtension(&h, chunk);
        }
    }
    return h.finalResult();
}

test "frozen support digest extension binds exact core membership and preserves empty bytes" {
    const selected = [_]usize{ 0, 2 };
    const weights = [_]f64{ 1, 2 };
    var legacy = std.crypto.hash.sha2.Sha256.init(.{});
    legacy.update("prefix");
    var empty = legacy;
    try hashExactCoreExtension(&empty, .{ .start = 0, .rows = 3, .selected_local = &selected, .weights = &weights });
    const legacy_digest = legacy.finalResult();
    const empty_digest = empty.finalResult();
    try std.testing.expectEqualSlices(u8, &legacy_digest, &empty_digest);
    var core = legacy;
    try hashExactCoreExtension(&core, .{ .start = 0, .rows = 3, .selected_local = &selected, .weights = &weights, .exact_core_local = &.{0} });
    const core_digest = core.finalResult();
    try std.testing.expect(!std.mem.eql(u8, &legacy_digest, &core_digest));
}

pub const TrainingSourceReport = struct {
    requested: bool = false,
    outcome: enum { disabled, ready, fallback_cap, fallback_allocation } = .disabled,
    digest: ?[32]u8 = null,
    counters: training_source.Counters = .{},
    external_live_bytes: usize = 0,
};

pub const AdaptivePlan = struct {
    weight_cap: local_weight_cap.Evidence = .{},
    allocator: std.mem.Allocator,
    scores: []adaptive.RangeScores,
    scored: [][3]?ScoredCandidate,
    branches: [3]adaptive.Geometry,
    geometry_policy: []const u8,
    geometry_storage: GeometryStorage = .dense,
    geometry_budget_percent: u8 = 100,
    geometry_workspace_bound_bytes: usize = 0,
    gram_memo: GramMemoEvidence = .{},
    gram_delta: GramDeltaEvidence = .{},
    joint_fixed_memo: JointFixedMemoEvidence = .{},
    training_reduction: TrainingReductionEvidence = .{},
    incremental_gram_chooser_calls: usize = 0,
    owns_scored_packets: bool = true,
    pub fn deinit(self: *@This()) void {
        for (&self.branches) |*branch| branch.deinit();
        if (self.owns_scored_packets) {
            for (self.scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
            self.allocator.free(self.scored);
            self.allocator.free(self.scores);
        }
        self.* = undefined;
    }
};

pub const Branch = struct {
    divisor: usize,
    candidate_rows: usize,
    mass_error: ?f64,
    output_rmse: ?f64,
    direct_train_metrics: ?solver.Metrics = null,
    heldout_detail: ?HeldoutDetail = null,
    rejection: []const u8,
    alpha: []f64,
    training_diagnostics: ?anchored.Diagnostics = null,
    geometry_digest: ?[32]u8 = null,
    geometry_middle_cap_rows: usize = 0,
    geometry_middle_rows: usize = 0,
    geometry_preserved_ranges: usize = 0,
    geometry_fitted_ranges: usize = 0,
    geometry_train_harm_before: f64 = 0,
    geometry_train_harm_after: f64 = 0,
    geometry_targets: []usize = &.{},
    postfit: PostfitEvidence = .{},
};

pub const HeldoutDetail = struct {
    validation: solver.ValidationDetail,
    candidate_peak_source_index: usize,
    original_peak_bias: f64,
    candidate_peak_bias: f64,
    candidate_peak_original_bias: f64,
    original_peak_bias_weight: ?f64,
    candidate_peak_bias_weight: ?f64,
    candidate_peak_weight_ratio: ?f64,
    original_peak_retained: bool,
    original_peak_exact_core: bool,
    original_peak_train_max_probability: ?f64,
    candidate_peak_train_max_probability: ?f64,
    original_peak_value_relative_l2: ?f64,
    /// exp(candidate bias - original bias) for the original reference peak
    /// when that exact source row survives in the composed candidate.
    original_peak_candidate_weight_ratio: ?f64,
};

fn finiteExp(value: f64) ?f64 {
    const result = @exp(value);
    return if (std.math.isFinite(result)) result else null;
}

const ReferencePeakEvidence = struct { retained: bool, candidate_bias_weight: ?f64, value_relative_l2: ?f64 };

pub const FallbackReuseEvidence = struct {
    live_training_sources: usize = 1,
    scored_bank_builds: usize = 1,
    scored_packets: usize = 0,
    geometry_attempts: usize = 1,
    final_validation_attempts: usize = 1,
    authorization_calls: usize = 0,
    fallback_attempted: bool = false,
    primary_rejected_result_destroyed: bool = false,
    reselections_avoided: usize = 0,
    value_fit_dispatches_avoided: usize = 0,
};

fn referencePeakEvidence(original: solver.Head, candidate: Candidate, original_peak_row: usize) !ReferencePeakEvidence {
    if (original_peak_row >= original.rows or candidate.source_indices.len != candidate.head.rows) return error.InvalidAttentionShape;
    for (candidate.source_indices, 0..) |source_row, candidate_row| if (source_row == original_peak_row) {
        const original_bias = original.bias.?[original_peak_row];
        const candidate_bias = candidate.head.bias.?[candidate_row];
        var delta_sq: f64 = 0;
        var original_sq: f64 = 0;
        const original_first = try std.math.mul(usize, original_peak_row, original.dim);
        const candidate_first = try std.math.mul(usize, candidate_row, candidate.head.dim);
        for (original.values[original_first..][0..original.dim], candidate.head.values[candidate_first..][0..candidate.head.dim]) |a, b| {
            const delta = b - a;
            delta_sq += delta * delta;
            original_sq += a * a;
        }
        const relative = if (std.math.isFinite(delta_sq) and std.math.isFinite(original_sq)) @sqrt(delta_sq / @max(original_sq, std.math.floatEps(f64))) else null;
        return .{ .retained = true, .candidate_bias_weight = finiteExp(candidate_bias - original_bias), .value_relative_l2 = relative };
    };
    return .{ .retained = false, .candidate_bias_weight = null, .value_relative_l2 = null };
}

fn exactCoreContains(frozen: FrozenOwned, source_row: usize) bool {
    for (frozen.ranges, frozen.chunks) |range, chunk| {
        if (source_row < range.start or source_row >= range.start + range.rows) continue;
        return std.mem.indexOfScalar(usize, chunk.exact_core_local, source_row - range.start) != null;
    }
    return false;
}

fn trainMaxProbability(cached: ?*training_source.Source, source_row: usize, control: solver.Control) !?f64 {
    const source = cached orelse return null;
    if (source_row < source.probability_row_start or source_row - source.probability_row_start >= (if (source.probability_rows == 0) source.rows else source.probability_rows)) return null;
    var maximum: f64 = 0;
    for (0..source.query_count) |query| {
        if (query % 32 == 0) try control.check();
        maximum = @max(maximum, try source.probability(query, source_row));
    }
    return maximum;
}

pub const CriticalCoreEvidence = struct {
    policy: CriticalCorePolicy = .disabled,
    policy_digest: [32]u8 = critical_adapter.policyDigest(.disabled),
    unique_rows: usize = 0,
    calibration: CriticalCoreCalibrationEvidence = .{},
    train_probability_census: TrainProbabilityCensus = .{},
};

pub const CriticalCoreCalibrationEvidence = struct {
    observed: bool = false,
    threshold: f64 = 0,
    critical_rows: usize = 0,
    minimum_middle_rows: usize = 0,
    middle_cap_rows: usize = 0,
    policy_sha256: [32]u8 = @splat(0),
};

fn trainMaximumProbabilities(allocator: std.mem.Allocator, cached: *const training_source.Source, external_live_bytes: usize, combined_cap: usize, control: solver.Control) ![]f64 {
    if (cached.rows == 0 or cached.query_count == 0) return error.InvalidFitConfig;
    const bytes = try std.math.mul(usize, cached.rows, @sizeOf(f64));
    if (try std.math.add(usize, external_live_bytes, bytes) > combined_cap) return error.CompactionWorkspaceLimit;
    const maxima = try allocator.alloc(f64, cached.rows);
    errdefer allocator.free(maxima);
    @memset(maxima, 0);
    for (0..cached.query_count) |query| {
        if (query % 32 == 0) try control.check();
        for (maxima, 0..) |*maximum, row| maximum.* = @max(maximum.*, try cached.probability(query, row));
    }
    return maxima;
}

pub const TrainProbabilityCensus = struct {
    observed: bool = false,
    thresholds: [3]f64 = .{ 0.005, 0.01, 0.025 },
    total_counts: [3]usize = @splat(0),
    range_count: usize = 0,
    /// Ascending counts across 256-row fit ranges; only [0..range_count] is live.
    sorted_range_counts: [3][32]u16 = @splat(@splat(0)),
};

fn trainProbabilityCensus(allocator: std.mem.Allocator, cached: *const training_source.Source, ranges: []const admission.Range, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !TrainProbabilityCensus {
    if (ranges.len > 32 or cached.rows == 0) return error.InvalidFitConfig;
    const bytes = try std.math.mul(usize, cached.rows, @sizeOf(f64));
    if (try std.math.add(usize, external_live_bytes, bytes) > combined_cap) return error.CompactionWorkspaceLimit;
    const maxima = try trainMaximumProbabilities(allocator, cached, external_live_bytes, combined_cap, control);
    defer allocator.free(maxima);
    var out: TrainProbabilityCensus = .{ .observed = true, .range_count = ranges.len };
    for (ranges, 0..) |range, ri| {
        const end = try std.math.add(usize, range.start, range.rows);
        if (end > maxima.len or range.rows > std.math.maxInt(u16)) return error.InvalidAttentionShape;
        for (out.thresholds, 0..) |threshold, ti| {
            var count: u16 = 0;
            for (maxima[range.start..end]) |value| if (value >= threshold) {
                count = try std.math.add(u16, count, 1);
                out.total_counts[ti] = try std.math.add(usize, out.total_counts[ti], 1);
            };
            out.sorted_range_counts[ti][ri] = count;
        }
    }
    for (&out.sorted_range_counts) |*counts| std.mem.sort(u16, counts[0..ranges.len], {}, std.sort.asc(u16));
    return out;
}

pub const Result = struct {
    weight_cap: local_weight_cap.Evidence = .{},
    allocator: std.mem.Allocator,
    head: solver.Head,
    source_indices: []usize,
    coverage_source_indices: []usize,
    branches: []Branch,
    accepted: bool,
    chosen_divisor: usize,
    final_metrics_valid: bool,
    final_mass_error: f64,
    final_output_rmse: f64,
    training_source: TrainingSourceReport = .{},
    critical_core: CriticalCoreEvidence = .{},
    postfit: PostfitEvidence = .{},
    scored_candidates_reused: bool = false,
    scored_packets_reused: usize = 0,
    reselections_avoided: usize = 0,
    value_fit_dispatches_avoided: usize = 0,
    fallback_reuse: FallbackReuseEvidence = .{},
    heldout_gate_calls: usize = 0,
    /// Actual storage selected by the completed training-source capture. This
    /// is populated from execution counters, never from the requested option.
    probability_storage: training_source.ProbabilityStorage = .f64,
    geometry_policy: []const u8 = adaptive.policy,
    geometry_storage: GeometryStorage = .dense,
    geometry_budget_percent: u8 = 100,
    geometry_workspace_bound_bytes: usize = 0,
    gram_memo: GramMemoEvidence = .{},
    gram_delta: GramDeltaEvidence = .{},
    joint_fixed_memo: JointFixedMemoEvidence = .{},
    training_reduction: TrainingReductionEvidence = .{},
    incremental_gram_chooser_calls: usize = 0,
    final_refit: FinalRefitEvidence = .{},
    frozen_support_bank_digest: [32]u8 = @splat(0),
    train_snapshot_manifest_digest: ?[32]u8 = null,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(@constCast(self.head.keys));
        self.allocator.free(@constCast(self.head.values));
        self.allocator.free(@constCast(self.head.bias.?));
        self.allocator.free(self.source_indices);
        self.allocator.free(self.coverage_source_indices);
        for (self.branches) |branch| {
            self.allocator.free(branch.alpha);
            if (branch.geometry_targets.len != 0) self.allocator.free(branch.geometry_targets);
        }
        self.allocator.free(self.branches);
        self.* = undefined;
    }
};

fn fallbackDeniedObservationEnabled(phase_timing: ?*PhaseTiming, train_probability_census: bool) bool {
    return phase_timing != null or train_probability_census;
}

fn observePrimaryRejectedFallbackDenied(result: *const Result, budget_percent: u8, enabled: bool) void {
    if (!enabled) return;
    std.debug.print("attention_match_primary_rejected_fallback_denied budget_percent={d} accepted={} final_metrics_valid={} branches={d} postfit_policy={s} postfit_attempts={d} postfit_completions={d} postfit_train_max_before={d} postfit_train_max_after={d} diagnostic_only=true\n", .{ budget_percent, result.accepted, result.final_metrics_valid, result.branches.len, @tagName(result.postfit.policy), result.postfit.attempts, result.postfit.completions, result.postfit.max_abs_train_before, result.postfit.max_abs_train_after });
    for (result.branches, 0..) |branch, index| {
        std.debug.print("attention_match_primary_rejected_fallback_denied_branch budget_percent={d} branch_index={d} divisor={d} candidate_rows={d} mass_error={any} output_rmse={any} rejection={s} direct_train_metrics={any} postfit_policy={s} postfit_attempts={d} postfit_completions={d} postfit_nnls_sweeps={d} postfit_train_max_before={d} postfit_train_max_after={d} diagnostic_only=true\n", .{ budget_percent, index, branch.divisor, branch.candidate_rows, branch.mass_error, branch.output_rmse, branch.rejection, branch.direct_train_metrics, @tagName(branch.postfit.policy), branch.postfit.attempts, branch.postfit.completions, branch.postfit.nnls_sweeps, branch.postfit.max_abs_train_before, branch.postfit.max_abs_train_after });
    }
}

test "fallback denied observation is default off and follows existing diagnostics only" {
    try std.testing.expect(!fallbackDeniedObservationEnabled(null, false));
    try std.testing.expect(fallbackDeniedObservationEnabled(null, true));
    var timing: PhaseTiming = .{};
    try std.testing.expect(fallbackDeniedObservationEnabled(&timing, false));
}

pub const FrozenOwned = struct {
    allocator: std.mem.Allocator,
    chunks: []joint.FrozenChunk,
    ranges: []admission.Range,
    indices: [][]usize,
    weights: [][]f64,
    pub fn deinit(self: *@This()) void {
        for (self.indices) |v| self.allocator.free(v);
        for (self.weights) |v| self.allocator.free(v);
        self.allocator.free(self.indices);
        self.allocator.free(self.weights);
        self.allocator.free(self.chunks);
        self.allocator.free(self.ranges);
        self.* = undefined;
    }
};

pub const ScoredCandidate = struct {
    frozen: FrozenOwned,
    fitted: joint.Result,
    pub fn deinit(self: *@This()) void {
        self.fitted.deinit();
        self.frozen.deinit();
        self.* = undefined;
    }
};

comptime {
    if (@sizeOf(FrozenOwned) != replay_planner.frozen_owned_layout_bytes or
        @alignOf(FrozenOwned) != replay_planner.frozen_owned_layout_alignment or
        @sizeOf(ScoredCandidate) != replay_planner.scored_candidate_layout_bytes or
        @alignOf(ScoredCandidate) != replay_planner.scored_candidate_layout_alignment or
        @sizeOf([3]?ScoredCandidate) != replay_planner.scored_candidate_table_entry_bytes)
        @compileError("adaptive replay planner scored allocation layout drift");
}

/// Adds the signed denominator and numerator change for one retained scored
/// packet. TRAIN probabilities only; callers own/zero q and q*d outputs.
pub fn accumulateScoredDelta(plan: *const AdaptivePlan, range_index: usize, level: usize, original: solver.Head, cached: *const training_source.Source, control: solver.Control, delta_rho: []f64, delta_y: []f64) !void {
    return accumulateScoredDeltaSlice(plan, range_index, level, original, cached, 0, cached.query_count, control, delta_rho, delta_y);
}

/// Allocation-free bounded view used by the per-query Gram builder. `delta_y`
/// is tile-local query-major; no full Q*D temporary is created.
pub fn accumulateScoredDeltaSlice(plan: *const AdaptivePlan, range_index: usize, level: usize, original: solver.Head, cached: *const training_source.Source, first_query: usize, query_count: usize, control: solver.Control, delta_rho: []f64, delta_y: []f64) !void {
    const qd = std.math.mul(usize, query_count, original.dim) catch return error.InvalidAttentionShape;
    const query_end = std.math.add(usize, first_query, query_count) catch return error.InvalidAttentionShape;
    if (range_index >= plan.scores.len or level >= 3 or query_end > cached.query_count or delta_rho.len != query_count or delta_y.len != qd) return error.InvalidAttentionShape;
    const score = plan.scores[range_index];
    const owned = plan.scored[range_index][level] orelse return error.InvalidAttentionShape;
    if (owned.frozen.chunks.len != 1 or owned.fitted.chunks.len != 1) return error.InvalidAttentionShape;
    const chunk = owned.frozen.chunks[0];
    const original_bias = original.bias orelse return error.InvalidAttentionShape;
    for (first_query..query_end, 0..) |qi, local_qi| {
        try control.check();
        for (0..score.rows) |local| {
            const probability = try cached.probability(qi, score.start + local);
            delta_rho[local_qi] -= probability;
            for (0..original.dim) |di| delta_y[local_qi * original.dim + di] -= probability * original.values[(score.start + local) * original.dim + di];
        }
        for (chunk.selected_local, 0..) |local, selected_i| {
            const global = score.start + local;
            const ratio = @exp(owned.fitted.bias[selected_i] - original_bias[global]);
            if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
            const probability = try cached.probability(qi, global) * ratio;
            delta_rho[local_qi] += probability;
            for (0..original.dim) |di| delta_y[local_qi * original.dim + di] += probability * owned.fitted.values[selected_i * original.dim + di];
        }
    }
}

fn gramMemoBytes(tile_rows: usize, dim: usize) !usize {
    if (tile_rows == 0 or dim == 0) return error.InvalidFitConfig;
    const values = std.math.add(usize, tile_rows, try std.math.mul(usize, tile_rows, dim)) catch return error.InvalidAttentionShape;
    return std.math.mul(usize, values, @sizeOf(f64)) catch error.InvalidAttentionShape;
}

/// Private zero-output accessor used only by the DEV Gram path. The public
/// additive accessor above remains unchanged. A hit is valid only for the
/// immediately matching range/query tile; changing either invalidates it.
const GramMemoContext = struct {
    plan: *const AdaptivePlan,
    original: solver.Head,
    cached: *const training_source.Source,
    negative_mass: []f64,
    negative_numerator: []f64,
    cached_range: ?usize = null,
    cached_first: usize = 0,
    cached_count: usize = 0,
    misses: usize = 0,
    hits: usize = 0,

    fn fill(raw: ?*anyopaque, range_index: usize, level: usize, first: usize, count: usize, delta_mass: []f64, delta_numerator: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        const qd = std.math.mul(usize, count, self.original.dim) catch return error.InvalidAttentionShape;
        const query_end = std.math.add(usize, first, count) catch return error.InvalidAttentionShape;
        if (range_index >= self.plan.scores.len or level >= 3 or query_end > self.cached.query_count or
            delta_mass.len != count or delta_numerator.len != qd or self.negative_mass.len < count or self.negative_numerator.len < qd)
            return error.InvalidAttentionShape;
        const score = self.plan.scores[range_index];
        const owned = self.plan.scored[range_index][level] orelse return error.InvalidAttentionShape;
        if (owned.frozen.chunks.len != 1 or owned.fitted.chunks.len != 1) return error.InvalidAttentionShape;
        const hit = self.cached_range == range_index and self.cached_first == first and self.cached_count == count;
        if (!hit) {
            @memset(self.negative_numerator[0..qd], 0);
            for (first..query_end, 0..) |qi, local_qi| {
                try control.check();
                var mass: f64 = 0;
                for (0..score.rows) |local| {
                    const probability = try self.cached.probability(qi, score.start + local);
                    mass -= probability;
                    for (0..self.original.dim) |di| self.negative_numerator[local_qi * self.original.dim + di] -= probability * self.original.values[(score.start + local) * self.original.dim + di];
                }
                self.negative_mass[local_qi] = mass;
            }
            self.cached_range = range_index;
            self.cached_first = first;
            self.cached_count = count;
            self.misses += 1;
        } else self.hits += 1;
        @memcpy(delta_mass, self.negative_mass[0..count]);
        @memcpy(delta_numerator, self.negative_numerator[0..qd]);
        const chunk = owned.frozen.chunks[0];
        const original_bias = self.original.bias orelse return error.InvalidAttentionShape;
        for (first..query_end, 0..) |qi, local_qi| {
            try control.check();
            for (chunk.selected_local, 0..) |local, selected_i| {
                const global = score.start + local;
                const ratio = @exp(owned.fitted.bias[selected_i] - original_bias[global]);
                if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
                const probability = try self.cached.probability(qi, global) * ratio;
                delta_mass[local_qi] += probability;
                for (0..self.original.dim) |di| delta_numerator[local_qi * self.original.dim + di] += probability * owned.fitted.values[selected_i * self.original.dim + di];
            }
        }
    }
};

fn scoredLevel(plan: *const AdaptivePlan, range: admission.Range, target: usize) !*const ScoredCandidate {
    for (plan.scores, plan.scored, 0..) |score, levels, ri| {
        if (score.start != range.start or score.rows != range.rows) continue;
        for (score.target_rows[0..3], levels, 0..) |rows, entry, li| {
            _ = entry;
            if (rows == target) {
                if (plan.scored[ri][li]) |*owned| return owned;
                return error.InvalidAttentionShape;
            }
        }
        return error.InvalidAttentionShape;
    }
    return error.InvalidAttentionShape;
}

pub fn freezeFromScored(allocator: std.mem.Allocator, ranges: []const admission.Range, targets: []const usize, plan: *const AdaptivePlan, cap: usize, control: solver.Control) !FrozenOwned {
    if (ranges.len != targets.len) return error.InvalidFitConfig;
    var count: usize = 0;
    for (ranges, targets) |range, target| if (target < range.rows) {
        if (target == 0) return error.InvalidFitConfig;
        count += 1;
    };
    if (count == 0) return error.InvalidFitConfig;
    var admitted = try std.math.add(usize, try std.math.mul(usize, count, @sizeOf(joint.FrozenChunk) + @sizeOf(admission.Range)), try std.math.mul(usize, count, @sizeOf([]usize) + @sizeOf([]f64)));
    for (ranges, targets) |range, target| {
        if (target < range.rows) admitted = try std.math.add(usize, admitted, try std.math.mul(usize, target, @sizeOf(usize) + @sizeOf(f64)));
    }
    if (admitted > cap) return error.CompactionWorkspaceLimit;
    const chunks = try allocator.alloc(joint.FrozenChunk, count);
    errdefer allocator.free(chunks);
    const compacted_ranges = try allocator.alloc(admission.Range, count);
    errdefer allocator.free(compacted_ranges);
    const indices = try allocator.alloc([]usize, count);
    errdefer allocator.free(indices);
    const weights = try allocator.alloc([]f64, count);
    errdefer allocator.free(weights);
    var made: usize = 0;
    errdefer {
        for (indices[0..made]) |slice| allocator.free(slice);
        for (weights[0..made]) |slice| allocator.free(slice);
    }
    var bytes = try std.math.add(usize, try std.math.mul(usize, count, @sizeOf(joint.FrozenChunk) + @sizeOf(admission.Range)), try std.math.mul(usize, count, @sizeOf([]usize) + @sizeOf([]f64)));
    for (ranges, targets) |range, target| {
        try control.check();
        if (target == range.rows) continue;
        const scored = try scoredLevel(plan, range, target);
        if (scored.frozen.chunks.len != 1 or scored.frozen.ranges.len != 1 or scored.frozen.chunks[0].selected_local.len != target) return error.InvalidAttentionShape;
        const payload = try std.math.add(usize, try std.math.mul(usize, target, @sizeOf(usize) + @sizeOf(f64)), try std.math.mul(usize, scored.frozen.chunks[0].exact_core_local.len, @sizeOf(usize)));
        bytes = try std.math.add(usize, bytes, payload);
        if (bytes > cap) return error.CompactionWorkspaceLimit;
        const source_chunk = scored.frozen.chunks[0];
        indices[made] = try allocator.alloc(usize, source_chunk.selected_local.len + source_chunk.exact_core_local.len);
        errdefer allocator.free(indices[made]);
        @memcpy(indices[made][0..source_chunk.selected_local.len], source_chunk.selected_local);
        @memcpy(indices[made][source_chunk.selected_local.len..], source_chunk.exact_core_local);
        weights[made] = try allocator.dupe(f64, scored.frozen.chunks[0].weights);
        errdefer allocator.free(weights[made]);
        chunks[made] = .{ .start = range.start, .rows = range.rows, .selected_local = indices[made][0..source_chunk.selected_local.len], .weights = weights[made], .exact_core_local = indices[made][source_chunk.selected_local.len..] };
        compacted_ranges[made] = range;
        made += 1;
    }
    return .{ .allocator = allocator, .chunks = chunks, .ranges = compacted_ranges, .indices = indices, .weights = weights };
}

pub fn fitFromScored(allocator: std.mem.Allocator, original: solver.Head, frozen: FrozenOwned, plan: *const AdaptivePlan, cap: usize, control: solver.Control) !joint.Result {
    var selected_total: usize = 0;
    for (frozen.chunks) |chunk| selected_total = try std.math.add(usize, selected_total, chunk.selected_local.len);
    const bytes = try std.math.add(usize, try std.math.mul(usize, frozen.chunks.len, @sizeOf(joint.ChunkResult)), try std.math.mul(usize, selected_total, try std.math.add(usize, try std.math.mul(usize, original.dim, @sizeOf(f64)), @sizeOf(f64))));
    if (bytes > cap) return error.CompactionWorkspaceLimit;
    const chunks = try allocator.alloc(joint.ChunkResult, frozen.chunks.len);
    errdefer allocator.free(chunks);
    const values = try allocator.alloc(f64, try std.math.mul(usize, selected_total, original.dim));
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, selected_total);
    errdefer allocator.free(bias);
    var value_offset: usize = 0;
    var bias_offset: usize = 0;
    for (frozen.ranges, frozen.chunks, 0..) |range, chunk, ci| {
        try control.check();
        const scored = try scoredLevel(plan, range, chunk.selected_local.len);
        if (scored.fitted.chunks.len != 1 or scored.fitted.values.len != chunk.selected_local.len * original.dim or scored.fitted.bias.len != chunk.selected_local.len) return error.InvalidAttentionShape;
        @memcpy(values[value_offset..][0..scored.fitted.values.len], scored.fitted.values);
        @memcpy(bias[bias_offset..][0..scored.fitted.bias.len], scored.fitted.bias);
        chunks[ci] = .{ .start = range.start, .rows = range.rows, .value_offset = value_offset, .selected_count = chunk.selected_local.len, .alpha = 1 };
        value_offset += scored.fitted.values.len;
        bias_offset += scored.fitted.bias.len;
    }
    return .{ .allocator = allocator, .chunks = chunks, .values = values, .bias = bias, .nnls_sweeps = 0 };
}

fn scoredPlanBytes(plan: *const AdaptivePlan) !usize {
    var bytes = try std.math.add(usize, try std.math.mul(usize, plan.scores.len, @sizeOf(adaptive.RangeScores)), try std.math.mul(usize, plan.scored.len, @sizeOf([3]?ScoredCandidate)));
    for (plan.scored) |levels| for (levels) |candidate| if (candidate) |owned| {
        const fitted = try std.math.add(usize, try std.math.mul(usize, owned.fitted.chunks.len, @sizeOf(joint.ChunkResult)), try std.math.add(usize, try std.math.mul(usize, owned.fitted.values.len, @sizeOf(f64)), try std.math.mul(usize, owned.fitted.bias.len, @sizeOf(f64))));
        bytes = try std.math.add(usize, bytes, try std.math.add(usize, try frozenBytes(owned.frozen), fitted));
    };
    return bytes;
}

fn adaptiveGeometryBytes(plan: *const AdaptivePlan) !usize {
    var bytes: usize = 0;
    for (&plan.branches) |*branch| {
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, branch.levels.len, @sizeOf(adaptive.Level)));
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, branch.targets.len, @sizeOf(usize)));
    }
    return bytes;
}

fn freezeBranchPolicy(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, divisor: usize, adaptive_targets: ?[]const usize, cfg: solver.Config, cached: ?*training_source.Source, counters: ?*training_source.Counters, policy: selection.SelectionPolicy, critical_authority: ?*const critical_keys.Result) !FrozenOwned {
    if (adaptive_targets) |targets| if (targets.len != ranges.len) return error.InvalidFitConfig;
    var compacted_count: usize = 0;
    for (ranges, 0..) |range, i| if (adaptive_targets == null or adaptive_targets.?[i] < range.rows) {
        compacted_count += 1;
    };
    if (compacted_count == 0) return error.InvalidFitConfig;
    const chunks = try allocator.alloc(joint.FrozenChunk, compacted_count);
    errdefer allocator.free(chunks);
    const compacted_ranges = try allocator.alloc(admission.Range, compacted_count);
    errdefer allocator.free(compacted_ranges);
    const indices = try allocator.alloc([]usize, compacted_count);
    errdefer allocator.free(indices);
    const weights = try allocator.alloc([]f64, compacted_count);
    errdefer allocator.free(weights);
    var made: usize = 0;
    var live_bytes: usize = try std.math.add(usize, try std.math.mul(usize, chunks.len, @sizeOf(joint.FrozenChunk)), try std.math.add(usize, try std.math.mul(usize, compacted_ranges.len, @sizeOf(admission.Range)), try std.math.add(usize, try std.math.mul(usize, indices.len, @sizeOf([]usize)), try std.math.mul(usize, weights.len, @sizeOf([]f64)))));
    errdefer {
        for (indices[0..made]) |v| allocator.free(v);
        for (weights[0..made]) |v| allocator.free(v);
    }
    for (ranges, 0..) |range, source_ci| {
        const budget = if (adaptive_targets) |targets| targets[source_ci] else std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig;
        if (budget == range.rows) continue;
        if (budget == 0 or budget > range.rows) return error.InvalidFitConfig;
        const ci = made;
        var local = cfg;
        local.target_rows = budget;
        local.selection = .residual_omp;
        local.mass_solver = .active_set;
        local.max_relative_mass_error = std.math.floatMax(f64);
        local.max_relative_output_rmse = std.math.floatMax(f64);
        const output_bytes = try std.math.mul(usize, budget, @sizeOf(usize) + @sizeOf(f64));
        const duplication_peak = try std.math.add(usize, live_bytes, try std.math.mul(usize, output_bytes, 2));
        if (duplication_peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        local.max_workspace_bytes = cfg.max_workspace_bytes - live_bytes;
        if (critical_authority != null and cached == null) return error.InvalidFitConfig;
        if (critical_authority) |authority| {
            var selected = try critical_adapter.selectCached(allocator, original, cached.?, range, budget, local, authority, policy);
            defer selected.deinit();
            indices[ci] = try allocator.alloc(usize, selected.indices.len + selected.exact_core_local.len);
            errdefer allocator.free(indices[ci]);
            @memcpy(indices[ci][0..selected.indices.len], selected.indices);
            @memcpy(indices[ci][selected.indices.len..], selected.exact_core_local);
            weights[ci] = try allocator.dupe(f64, selected.weights);
            errdefer allocator.free(weights[ci]);
            chunks[ci] = .{ .start = range.start, .rows = range.rows, .selected_local = indices[ci][0..selected.indices.len], .weights = weights[ci], .exact_core_local = indices[ci][selected.indices.len..] };
            cached.?.noteSelectionHit(counters);
            compacted_ranges[ci] = range;
            live_bytes = try std.math.add(usize, live_bytes, try std.math.add(usize, output_bytes, try std.math.mul(usize, selected.exact_core_local.len, @sizeOf(usize))));
            made += 1;
            continue;
        }
        var selected = if (cached) |source| blk: {
            const hit = try selection.selectFromTrainingSourceRuntimeNamedPolicy(allocator, original, range.start, range.rows, source, local, policy);
            source.noteSelectionHit(counters);
            break :blk hit;
        } else try selection.selectStreamingRuntimeNamedPolicy(allocator, original, range.start, range.rows, train, local, policy);
        defer selected.deinit();
        if (selected.indices.len != budget or selected.weights.len != budget) return error.InvalidAttentionShape;
        indices[ci] = try allocator.dupe(usize, selected.indices);
        errdefer allocator.free(indices[ci]);
        weights[ci] = try allocator.dupe(f64, selected.weights);
        errdefer allocator.free(weights[ci]);
        chunks[ci] = .{ .start = range.start, .rows = range.rows, .selected_local = indices[ci], .weights = weights[ci] };
        compacted_ranges[ci] = range;
        live_bytes = try std.math.add(usize, live_bytes, output_bytes);
        made += 1;
    }
    return .{ .allocator = allocator, .chunks = chunks, .ranges = compacted_ranges, .indices = indices, .weights = weights };
}

pub fn freezeBranch(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, divisor: usize, adaptive_targets: ?[]const usize, cfg: solver.Config, cached: ?*training_source.Source, counters: ?*training_source.Counters) !FrozenOwned {
    return freezeBranchPolicy(allocator, original, train, ranges, divisor, adaptive_targets, cfg, cached, counters, .raw, null);
}

fn activeSelectionPolicy(options: Options) selection.SelectionPolicy {
    return options.selection_policy orelse switch (options.selection_score) {
        .raw => .raw,
        .normalized => .normalized,
    };
}

pub fn frozenBytes(frozen: FrozenOwned) !usize {
    var bytes = try std.math.add(usize, try std.math.mul(usize, frozen.chunks.len, @sizeOf(joint.FrozenChunk)), try std.math.add(usize, try std.math.mul(usize, frozen.ranges.len, @sizeOf(admission.Range)), try std.math.add(usize, try std.math.mul(usize, frozen.indices.len, @sizeOf([]usize)), try std.math.mul(usize, frozen.weights.len, @sizeOf([]f64)))));
    for (frozen.chunks) |chunk| bytes = try std.math.add(usize, bytes, try std.math.add(usize, try std.math.mul(usize, chunk.selected_local.len, @sizeOf(usize) + @sizeOf(f64)), try std.math.mul(usize, chunk.exact_core_local.len, @sizeOf(usize))));
    return bytes;
}

pub fn round16(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f16)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f16, @floatCast(value.*)));
    }
}
pub fn round32(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f32, @floatCast(value.*)));
    }
}

pub const Candidate = struct {
    allocator: std.mem.Allocator,
    head: solver.Head,
    source_indices: []usize,
    coverage_source_indices: []usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(@constCast(self.head.keys));
        self.allocator.free(@constCast(self.head.values));
        self.allocator.free(@constCast(self.head.bias.?));
        self.allocator.free(self.source_indices);
        self.allocator.free(self.coverage_source_indices);
    }
};

test "rejected branch capture is divisor2 rejection only and borrows live candidate" {
    const Observer = struct {
        calls: usize = 0,
        expected_keys: [*]const f64,
        fn capture(raw: ?*anyopaque, view: RejectedBranchView) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            std.testing.expectEqual(@as(usize, 2), view.divisor) catch unreachable;
            std.testing.expectEqual(self.expected_keys, view.candidate.keys.ptr) catch unreachable;
            std.testing.expectEqual(@as(usize, 1), view.source_indices.len) catch unreachable;
        }
    };
    const keys = [_]f64{1};
    const values = [_]f64{2};
    const bias = [_]f64{0};
    var source_indices = [_]usize{0};
    var coverage = [_]usize{0};
    const head: solver.Head = .{ .keys = &keys, .values = &values, .bias = &bias, .rows = 1, .dim = 1, .scale = 1, .softcap = 0 };
    const candidate: Candidate = .{ .allocator = std.testing.allocator, .head = head, .source_indices = &source_indices, .coverage_source_indices = &coverage };
    var observer: Observer = .{ .expected_keys = keys[0..].ptr };
    const options: Options = .{ .rejected_branch_capture = .{ .context = &observer, .capture_fn = Observer.capture } };
    observeRejectedBranch(options, 4, false, head, &candidate, &.{3}, &.{4}, .{});
    observeRejectedBranch(options, 2, true, head, &candidate, &.{3}, &.{4}, .{});
    try std.testing.expectEqual(@as(usize, 0), observer.calls);
    observeRejectedBranch(options, 2, false, head, &candidate, &.{3}, &.{4}, .{});
    try std.testing.expectEqual(@as(usize, 1), observer.calls);
}

pub fn compose(allocator: std.mem.Allocator, original: solver.Head, ranges: []const admission.Range, frozen: FrozenOwned, fitted: joint.Result) !Candidate {
    var rows = original.rows;
    for (ranges, frozen.chunks) |range, chunk| {
        rows -= range.rows;
        rows += chunk.selected_local.len;
    }
    const elements = try std.math.mul(usize, rows, original.dim);
    const keys = try allocator.alloc(f64, elements);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, elements);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, rows);
    errdefer allocator.free(bias);
    const source_indices = try allocator.alloc(usize, rows);
    errdefer allocator.free(source_indices);
    const coverage = try allocator.alloc(usize, rows);
    errdefer allocator.free(coverage);
    var src: usize = 0;
    var dst: usize = 0;
    var value_offset: usize = 0;
    var bias_offset: usize = 0;
    for (ranges, frozen.chunks) |range, chunk| {
        const exact = range.start - src;
        @memcpy(keys[dst * original.dim ..][0 .. exact * original.dim], original.keys[src * original.dim ..][0 .. exact * original.dim]);
        @memcpy(values[dst * original.dim ..][0 .. exact * original.dim], original.values[src * original.dim ..][0 .. exact * original.dim]);
        @memcpy(bias[dst..][0..exact], original.bias.?[src..][0..exact]);
        for (0..exact) |i| {
            source_indices[dst + i] = src + i;
            coverage[dst + i] = src + i;
        }
        dst += exact;
        const m = chunk.selected_local.len;
        const order = try allocator.alloc(usize, m);
        defer allocator.free(order);
        for (order, 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order, chunk.selected_local, struct {
            fn less(indices: []const usize, a: usize, b: usize) bool {
                return indices[a] < indices[b];
            }
        }.less);
        for (order, 0..) |source_order, j| {
            const global = range.start + chunk.selected_local[source_order];
            @memcpy(keys[(dst + j) * original.dim ..][0..original.dim], original.keys[global * original.dim ..][0..original.dim]);
            @memcpy(values[(dst + j) * original.dim ..][0..original.dim], fitted.values[value_offset + source_order * original.dim ..][0..original.dim]);
            bias[dst + j] = fitted.bias[bias_offset + source_order];
            source_indices[dst + j] = global;
            coverage[dst + j] = range.start + range.rows - 1;
        }
        // Only replacement rows cross a new storage boundary. Protected,
        // preserved and recent rows remain byte-equivalent to the source.
        try round16(keys[dst * original.dim ..][0 .. m * original.dim]);
        try round16(values[dst * original.dim ..][0 .. m * original.dim]);
        try round32(bias[dst..][0..m]);
        dst += m;
        value_offset += m * original.dim;
        bias_offset += m;
        src = range.start + range.rows;
    }
    const tail = original.rows - src;
    @memcpy(keys[dst * original.dim ..][0 .. tail * original.dim], original.keys[src * original.dim ..][0 .. tail * original.dim]);
    @memcpy(values[dst * original.dim ..][0 .. tail * original.dim], original.values[src * original.dim ..][0 .. tail * original.dim]);
    @memcpy(bias[dst..][0..tail], original.bias.?[src..][0..tail]);
    for (0..tail) |i| {
        source_indices[dst + i] = src + i;
        coverage[dst + i] = src + i;
    }
    dst += tail;
    if (dst != rows) return error.InvalidAttentionShape;
    return .{ .allocator = allocator, .head = .{ .keys = keys, .values = values, .bias = bias, .rows = rows, .dim = original.dim, .scale = original.scale, .softcap = original.softcap }, .source_indices = source_indices, .coverage_source_indices = coverage };
}

fn attachGeometry(allocator: std.mem.Allocator, branch: *Branch, options: Options, index: usize) !void {
    const geometries = options.adaptive_geometry orelse return;
    const geometry = &geometries[index];
    branch.geometry_targets = try allocator.dupe(usize, geometry.targets);
    branch.geometry_digest = geometry.digest;
    branch.geometry_middle_cap_rows = geometry.middle_cap_rows;
    branch.geometry_middle_rows = geometry.middle_rows;
    branch.geometry_preserved_ranges = geometry.preserved_ranges;
    branch.geometry_fitted_ranges = geometry.fitted_ranges;
    branch.geometry_train_harm_before = geometry.train_harm_before;
    branch.geometry_train_harm_after = geometry.train_harm_after;
}

fn unavailableGeometry(allocator: std.mem.Allocator, representation: *const gram_geometry.Representation, divisor: usize, budget_percent: u8) !adaptive.Geometry {
    const levels = try allocator.alloc(adaptive.Level, representation.ranges.len);
    errdefer allocator.free(levels);
    @memset(levels, .divisor4);
    const targets = try allocator.alloc(usize, representation.ranges.len);
    errdefer allocator.free(targets);
    var eligible: usize = 0;
    var required: usize = 0;
    for (representation.ranges, targets) |range, *target| {
        target.* = range.target_rows[0];
        eligible = try std.math.add(usize, eligible, range.rows);
        required = try std.math.add(usize, required, target.*);
    }
    const nominal_cap = std.math.divCeil(usize, eligible, divisor) catch return error.InvalidAttentionShape;
    const cap = if (budget_percent == 100) nominal_cap else try std.math.mul(usize, nominal_cap, budget_percent) / 100;
    if (required <= cap) return error.InvalidFitConfig;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.geometry-budget-infeasible.v1\x00");
    hashU64Le(&h, divisor);
    hashU64Le(&h, budget_percent);
    hashU64Le(&h, cap);
    hashU64Le(&h, required);
    for (targets) |target| hashU64Le(&h, target);
    return .{ .allocator = allocator, .available = false, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = required, .preserved_ranges = 0, .fitted_ranges = representation.ranges.len, .train_harm_before = 0, .train_harm_after = 0, .digest = h.finalResult() };
}
fn branchPayloadBytes(branch: Branch) !usize {
    return std.math.add(usize, try std.math.mul(usize, branch.alpha.len, @sizeOf(f64)), try std.math.mul(usize, branch.geometry_targets.len, @sizeOf(usize))) catch error.InvalidAttentionShape;
}

fn overlayPostRanges(input_ranges: []const admission.Range, frozen: FrozenOwned, output: []group_mass_postfit.Range) !void {
    if (input_ranges.len == 0 or output.len != input_ranges.len or frozen.ranges.len != frozen.chunks.len) return error.InvalidFitConfig;
    for (input_ranges, output) |range, *out| out.* = .{ .start = range.start, .rows = range.rows, .target_rows = range.rows };
    for (frozen.ranges, frozen.chunks) |range, chunk| {
        var found: ?usize = null;
        for (input_ranges, 0..) |input, index| if (input.start == range.start and input.rows == range.rows) {
            found = index;
            break;
        };
        output[found orelse return error.InvalidAttentionShape].target_rows = chunk.selected_local.len;
    }
}

pub fn applyTrainGroupPostfit(allocator: std.mem.Allocator, original: solver.Head, candidate: *Candidate, input_ranges: []const admission.Range, frozen: FrozenOwned, cached: *training_source.Source, cfg: solver.Config, reserve: usize, policy_value: PostfitPolicy) !PostfitEvidence {
    if (reserve == 0 or input_ranges.len == 0 or frozen.ranges.len != frozen.chunks.len) return error.InvalidFitConfig;
    var core_count: usize = 0;
    for (frozen.chunks) |chunk| core_count = try std.math.add(usize, core_count, chunk.exact_core_local.len);
    const post_ranges = try allocator.alloc(group_mass_postfit.Range, input_ranges.len);
    defer allocator.free(post_ranges);
    const cores = try allocator.alloc(usize, core_count);
    defer allocator.free(cores);
    try overlayPostRanges(input_ranges, frozen, post_ranges);
    var ci: usize = 0;
    for (frozen.ranges, frozen.chunks) |range, chunk| {
        for (chunk.exact_core_local) |local| {
            if (local >= range.rows) return error.InvalidAttentionShape;
            cores[ci] = try std.math.add(usize, range.start, local);
            ci += 1;
        }
    }
    const metadata = try std.math.add(usize, try std.math.mul(usize, post_ranges.len, @sizeOf(group_mass_postfit.Range)), try std.math.mul(usize, cores.len, @sizeOf(usize)));
    if (metadata >= reserve) return error.CompactionWorkspaceLimit;
    var post_cfg = cfg;
    post_cfg.max_workspace_bytes = reserve - metadata;
    if (policy_value == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1) {
        var independent_context: @import("../../ops/metal_attention_match_validation.zig").Context = .{};
        const backend = @import("../../ops/metal_attention_match_validation.zig").backend(&independent_context);
        const evidence = try @import("attention_match_train_l2_surrogate_minimax_postfit.zig").apply(allocator, original, &candidate.head, candidate.source_indices, post_ranges, cores, cached, post_cfg, backend, metadata, reserve);
        return .{
            .policy = policy_value,
            .policy_digest = postfitPolicyDigest(policy_value),
            .attempts = 1,
            .completions = 1,
            .nnls_sweeps = evidence.l2.nnls_sweeps,
            // Legacy fields retain their established L2 residual semantics.
            .max_abs_train_before = evidence.l2.max_abs_before,
            .max_abs_train_after = evidence.l2.max_abs_after,
            .workspace_bytes = try std.math.add(usize, metadata, evidence.workspace_bytes),
            .minimax_attempts = 1,
            .minimax_completions = 1,
            .minimax_applied = @intFromBool(evidence.minimax_applied),
            .minimax_scale = evidence.minimax_scale,
            .minimax_scale_min = evidence.minimax_scale,
            .minimax_scale_max = evidence.minimax_scale,
            .exact_train_mass_before = evidence.baseline.max_relative_mass_error,
            .exact_train_mass_after = evidence.selected.max_relative_mass_error,
            .exact_train_output_rmse_before = evidence.baseline.relative_output_rmse,
            .exact_train_output_rmse_after = evidence.selected.relative_output_rmse,
        };
    }
    const diagnostics = switch (policy_value) {
        .train_global_residual_range_mass_l2_exact_f32_cache_v1 => try group_mass_postfit.apply(allocator, original, candidate.head, candidate.source_indices, post_ranges, cores, cached, post_cfg, 0),
        .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => try group_mass_postfit.applyLawson(allocator, original, candidate.head, candidate.source_indices, post_ranges, cores, cached, post_cfg, 0),
        .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => unreachable,
        .disabled => return error.InvalidFitConfig,
    };
    return .{ .policy = policy_value, .policy_digest = postfitPolicyDigest(policy_value), .attempts = 1, .completions = 1, .nnls_sweeps = diagnostics.nnls_sweeps, .max_abs_train_before = diagnostics.max_abs_before, .max_abs_train_after = diagnostics.max_abs_after, .workspace_bytes = try std.math.add(usize, diagnostics.workspace_bytes, metadata), .refinement_attempted_iterations = diagnostics.lawson_attempted_iterations, .refinement_accepted_improvements = diagnostics.lawson_accepted_improvements, .refinement_nnls_sweeps = diagnostics.lawson_nnls_sweeps, .refinement_outcome = diagnostics.lawson_outcome };
}

test "postfit range overlay retains omitted identity gaps" {
    const input = [_]admission.Range{ .{ .start = 2, .rows = 2 }, .{ .start = 4, .rows = 2 }, .{ .start = 6, .rows = 2 } };
    const selected = [_]usize{0};
    const weights = [_]f64{1};
    var ranges = [_]admission.Range{ .{ .start = 2, .rows = 2 }, .{ .start = 6, .rows = 2 } };
    var chunks = [_]joint.FrozenChunk{ .{ .start = 2, .rows = 2, .selected_local = &selected, .weights = &weights }, .{ .start = 6, .rows = 2, .selected_local = &selected, .weights = &weights } };
    var indices = [_][]usize{};
    var owned_weights = [_][]f64{};
    const frozen: FrozenOwned = .{ .allocator = std.testing.allocator, .chunks = &chunks, .ranges = &ranges, .indices = &indices, .weights = &owned_weights };
    var output: [3]group_mass_postfit.Range = undefined;
    try overlayPostRanges(&input, frozen, &output);
    try std.testing.expectEqual(@as(usize, 1), output[0].target_rows);
    try std.testing.expectEqual(@as(usize, 2), output[1].target_rows);
    try std.testing.expectEqual(@as(usize, 1), output[2].target_rows);
}

fn aggregatePostfit(branches: []const Branch, policy_value: PostfitPolicy) !PostfitEvidence {
    if (policy_value == .disabled) return .{};
    var out: PostfitEvidence = .{ .policy = policy_value, .policy_digest = postfitPolicyDigest(policy_value) };
    for (branches) |branch| {
        out.attempts = try std.math.add(usize, out.attempts, branch.postfit.attempts);
        out.completions = try std.math.add(usize, out.completions, branch.postfit.completions);
        out.nnls_sweeps = @max(out.nnls_sweeps, branch.postfit.nnls_sweeps);
        out.max_abs_train_before = @max(out.max_abs_train_before, branch.postfit.max_abs_train_before);
        out.max_abs_train_after = @max(out.max_abs_train_after, branch.postfit.max_abs_train_after);
        out.workspace_bytes = @max(out.workspace_bytes, branch.postfit.workspace_bytes);
        out.minimax_attempts = try std.math.add(usize, out.minimax_attempts, branch.postfit.minimax_attempts);
        out.minimax_applied = try std.math.add(usize, out.minimax_applied, branch.postfit.minimax_applied);
        if (branch.postfit.minimax_completions != 0) {
            if (out.minimax_completions == 0) {
                out.minimax_scale_min = branch.postfit.minimax_scale;
                out.minimax_scale_max = branch.postfit.minimax_scale;
            } else {
                out.minimax_scale_min = @min(out.minimax_scale_min, branch.postfit.minimax_scale);
                out.minimax_scale_max = @max(out.minimax_scale_max, branch.postfit.minimax_scale);
            }
            // Last evaluated branch is the selected branch when accepted.
            out.minimax_scale = branch.postfit.minimax_scale;
        }
        out.minimax_completions = try std.math.add(usize, out.minimax_completions, branch.postfit.minimax_completions);
        out.exact_train_mass_before = @max(out.exact_train_mass_before, branch.postfit.exact_train_mass_before);
        out.exact_train_mass_after = @max(out.exact_train_mass_after, branch.postfit.exact_train_mass_after);
        out.exact_train_output_rmse_before = @max(out.exact_train_output_rmse_before, branch.postfit.exact_train_output_rmse_before);
        out.exact_train_output_rmse_after = @max(out.exact_train_output_rmse_after, branch.postfit.exact_train_output_rmse_after);
        out.refinement_attempted_iterations = @max(out.refinement_attempted_iterations, branch.postfit.refinement_attempted_iterations);
        out.refinement_accepted_improvements = @max(out.refinement_accepted_improvements, branch.postfit.refinement_accepted_improvements);
        out.refinement_nnls_sweeps = @max(out.refinement_nnls_sweeps, branch.postfit.refinement_nnls_sweeps);
        if (@backingInt(branch.postfit.refinement_outcome) > @backingInt(out.refinement_outcome)) out.refinement_outcome = branch.postfit.refinement_outcome;
    }
    return out;
}

test "postfit rejected evidence aggregates and all unavailable remains typed zero" {
    const enabled: PostfitPolicy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    const unavailable = [_]Branch{.{ .divisor = 4, .candidate_rows = 8, .mass_error = null, .output_rmse = null, .rejection = "geometry_budget_infeasible", .alpha = &.{} }};
    const zero = try aggregatePostfit(&unavailable, enabled);
    try std.testing.expectEqual(enabled, zero.policy);
    try std.testing.expectEqual(@as(usize, 0), zero.attempts);
    const rejected = [_]Branch{.{ .divisor = 2, .candidate_rows = 4, .mass_error = 0.2, .output_rmse = 0.1, .rejection = "heldout_gate", .alpha = &.{}, .postfit = .{ .policy = enabled, .policy_digest = postfitPolicyDigest(enabled), .attempts = 1, .completions = 1, .nnls_sweeps = 7 } }};
    const aggregate = try aggregatePostfit(&rejected, enabled);
    try std.testing.expectEqual(@as(usize, 1), aggregate.attempts);
    try std.testing.expectEqual(@as(usize, 1), aggregate.completions);
    try std.testing.expectEqual(@as(usize, 7), aggregate.nnls_sweeps);
    var minimax = rejected[0];
    minimax.postfit.minimax_attempts = 1;
    minimax.postfit.minimax_completions = 1;
    minimax.postfit.minimax_applied = 1;
    minimax.postfit.minimax_scale = 1.02;
    minimax.postfit.exact_train_mass_before = 0.04;
    minimax.postfit.exact_train_mass_after = 0.03;
    const measured = try aggregatePostfit(&.{ unavailable[0], minimax }, enabled);
    try std.testing.expectEqual(@as(usize, 1), measured.minimax_attempts);
    try std.testing.expectEqual(@as(usize, 1), measured.minimax_applied);
    try std.testing.expectEqual(@as(f64, 1.02), measured.minimax_scale_min);
    try std.testing.expectEqual(@as(f64, 1.02), measured.minimax_scale_max);
    try std.testing.expectEqual(@as(f64, 0.04), measured.exact_train_mass_before);
    try std.testing.expectEqual(@as(f64, 0.03), measured.exact_train_mass_after);
}

fn primaryProbePlannerExternalBytes(external_bytes: usize, postfit_workspace_bytes: usize, primary_probe: bool) !usize {
    return if (primary_probe) external_bytes else std.math.add(usize, external_bytes, postfit_workspace_bytes);
}

test "primary probe planner separates postfit phase while normal path retains it" {
    try std.testing.expectEqual(@as(usize, 100), try primaryProbePlannerExternalBytes(100, 25, true));
    try std.testing.expectEqual(@as(usize, 125), try primaryProbePlannerExternalBytes(100, 25, false));
    try std.testing.expectError(error.Overflow, primaryProbePlannerExternalBytes(std.math.maxInt(usize), 1, false));
    try std.testing.expectEqual(std.math.maxInt(usize), try primaryProbePlannerExternalBytes(std.math.maxInt(usize), 1, true));
}

fn runWithOptions(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, options: Options, cached: ?*training_source.Source, counters: ?*training_source.Counters) !Result {
    if (options.geometry_storage == .per_query_gram and options.scored_candidates == null) return error.InvalidFitConfig;
    if (options.final_refit_strategy != .identity_reuse and
        (options.scored_candidates == null or cached == null or
            options.mass_strategy != .identity_chunk_coefficients_v18 or !options.training_source))
        return error.InvalidFitConfig;
    try cfg.control.check();
    const divisors = [_]usize{ 4, 3, 2 };
    const branch_struct_bytes = try std.math.mul(usize, 3, @sizeOf(Branch));
    if (branch_struct_bytes > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const branches = try allocator.alloc(Branch, divisors.len);
    errdefer allocator.free(branches);
    var initialized: usize = 0;
    var heldout_gate_calls: usize = 0;
    errdefer for (branches[0..initialized]) |branch| {
        allocator.free(branch.alpha);
        if (branch.geometry_targets.len != 0) allocator.free(branch.geometry_targets);
    };
    var branch_live_bytes = branch_struct_bytes;
    var reused_packets: usize = 0;
    var avoided_dispatches: usize = 0;
    var refit_attempts: usize = 0;
    var refit_completions: usize = 0;
    for (divisors, 0..) |divisor, bi| {
        if (branch_live_bytes > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var branch_cfg = cfg;
        branch_cfg.max_workspace_bytes -= branch_live_bytes;
        const adaptive_targets = if (options.adaptive_targets_by_branch) |targets| targets[bi] else null;
        if (options.adaptive_geometry) |geometries| if (!geometries[bi].available) {
            branches[bi] = .{ .divisor = divisor, .candidate_rows = original.rows, .mass_error = null, .output_rmse = null, .rejection = "geometry_budget_infeasible", .alpha = try allocator.alloc(f64, 0) };
            attachGeometry(allocator, &branches[bi], options, bi) catch |err| {
                allocator.free(branches[bi].alpha);
                return err;
            };
            initialized += 1;
            branch_live_bytes = try std.math.add(usize, branch_live_bytes, try branchPayloadBytes(branches[bi]));
            continue;
        };
        const freeze_started = phaseStart(options.phase_timing);
        var frozen = (if (options.scored_candidates) |plan|
            freezeFromScored(allocator, ranges, adaptive_targets orelse return error.InvalidFitConfig, plan, branch_cfg.max_workspace_bytes, branch_cfg.control)
        else
            freezeBranchPolicy(allocator, original, train, ranges, divisor, adaptive_targets, branch_cfg, cached, counters, activeSelectionPolicy(options), null)) catch |err| switch (err) {
            error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => {
                branches[bi] = .{ .divisor = divisor, .candidate_rows = original.rows, .mass_error = null, .output_rmse = null, .rejection = "selection_or_mass_fit", .alpha = try allocator.alloc(f64, 0) };
                attachGeometry(allocator, &branches[bi], options, bi) catch |attach_err| {
                    allocator.free(branches[bi].alpha);
                    return attach_err;
                };
                initialized += 1;
                branch_live_bytes = try std.math.add(usize, branch_live_bytes, try branchPayloadBytes(branches[bi]));
                continue;
            },
            else => return err,
        };
        phaseEnd(options.phase_timing, .selection_freeze, freeze_started);
        defer frozen.deinit();
        if (options.scored_candidates != null) {
            reused_packets = try std.math.add(usize, reused_packets, frozen.chunks.len);
            if (options.final_refit_strategy == .identity_reuse)
                avoided_dispatches = try std.math.add(usize, avoided_dispatches, 1);
        }
        const frozen_bytes = try frozenBytes(frozen);
        if (frozen_bytes > branch_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var joint_cfg = branch_cfg;
        joint_cfg.max_workspace_bytes -= frozen_bytes;
        const fit_started = phaseStart(options.phase_timing);
        if (options.scored_candidates != null and options.final_refit_strategy == .global_active_nnls_v16)
            refit_attempts = try std.math.add(usize, refit_attempts, 1);
        var fitted = (if (options.scored_candidates) |plan|
            switch (options.final_refit_strategy) {
                .identity_reuse => fitFromScored(allocator, original, frozen, plan, joint_cfg.max_workspace_bytes, joint_cfg.control),
                .global_active_nnls_v16 => joint.fitStrategy(allocator, original, train, frozen.chunks, joint_cfg, cached.?, counters, .global_active_nnls_v16, false),
            }
        else
            joint.fitStrategy(allocator, original, train, frozen.chunks, joint_cfg, cached, counters, options.mass_strategy, options.retain_training_diagnostics)) catch |err| switch (err) {
            error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => {
                branches[bi] = .{ .divisor = divisor, .candidate_rows = original.rows, .mass_error = null, .output_rmse = null, .rejection = "joint_mass_or_value_fit", .alpha = try allocator.alloc(f64, 0) };
                attachGeometry(allocator, &branches[bi], options, bi) catch |attach_err| {
                    allocator.free(branches[bi].alpha);
                    return attach_err;
                };
                initialized += 1;
                branch_live_bytes = try std.math.add(usize, branch_live_bytes, try branchPayloadBytes(branches[bi]));
                continue;
            },
            else => return err,
        };
        if (options.scored_candidates != null and options.final_refit_strategy == .global_active_nnls_v16)
            refit_completions = try std.math.add(usize, refit_completions, 1);
        phaseEnd(options.phase_timing, .joint_value_fit, fit_started);
        defer fitted.deinit();
        var selected_total: usize = 0;
        var max_selected: usize = 0;
        for (frozen.chunks) |chunk| selected_total = try std.math.add(usize, selected_total, chunk.selected_local.len);
        for (frozen.chunks) |chunk| max_selected = @max(max_selected, chunk.selected_local.len);
        const fitted_bytes = try std.math.add(usize, try std.math.mul(usize, fitted.chunks.len, @sizeOf(joint.ChunkResult)), try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, selected_total, original.dim), selected_total), @sizeOf(f64)));
        var candidate_rows = original.rows;
        for (frozen.ranges, frozen.chunks) |range, chunk| candidate_rows = try std.math.add(usize, std.math.sub(usize, candidate_rows, range.rows) catch return error.InvalidAttentionShape, chunk.selected_local.len);
        const candidate_numeric = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, try std.math.mul(usize, candidate_rows, original.dim), 2), candidate_rows), @sizeOf(f64));
        const candidate_maps = try std.math.mul(usize, try std.math.mul(usize, candidate_rows, 2), @sizeOf(usize));
        const compose_peak = try std.math.add(usize, try std.math.add(usize, frozen_bytes, fitted_bytes), try std.math.add(usize, try std.math.add(usize, candidate_numeric, candidate_maps), try std.math.mul(usize, max_selected, @sizeOf(usize))));
        if (try std.math.add(usize, branch_live_bytes, compose_peak) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var candidate = try compose(allocator, original, frozen.ranges, frozen, fitted);
        var keep = false;
        defer if (!keep) candidate.deinit();
        const postfit: PostfitEvidence = switch (options.postfit_policy) {
            .disabled => .{},
            .train_global_residual_range_mass_l2_exact_f32_cache_v1 => if (options.candidate_postfit_hook) |hook|
                try hook.apply_fn(hook.context, allocator, original, &candidate, ranges, frozen, branch_cfg)
            else
                try applyTrainGroupPostfit(allocator, original, &candidate, ranges, frozen, cached orelse return error.InvalidFitConfig, branch_cfg, options.postfit_workspace_bytes, options.postfit_policy),
            .train_global_residual_range_mass_lawson_exact_rounded_max_v1 => if (options.candidate_postfit_hook != null)
                return error.InvalidFitConfig
            else
                try applyTrainGroupPostfit(allocator, original, &candidate, ranges, frozen, cached orelse return error.InvalidFitConfig, branch_cfg, options.postfit_workspace_bytes, options.postfit_policy),
            .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 => if (options.candidate_postfit_hook != null)
                return error.InvalidFitConfig
            else
                try applyTrainGroupPostfit(allocator, original, &candidate, ranges, frozen, cached orelse return error.InvalidFitConfig, branch_cfg, options.postfit_workspace_bytes, options.postfit_policy),
        };
        const validation_cfgs = try branchValidationConfigs(cfg, options.validation_backend, options.observe_validation_detail, options.train_reference_summary_backend, options.heldout_reference_summary_backend);
        const held_cfg = validation_cfgs.heldout;
        const train_observer_cfg = validation_cfgs.direct_train;
        const heldout_rows = heldout.len / original.dim;
        const validation_bytes = try solver.validationBytes(original.rows, candidate_rows, original.dim, heldout_rows, held_cfg);
        const train_validation_bytes = if (options.observe_direct_train)
            try solver.validationBytes(original.rows, candidate_rows, original.dim, train.len / original.dim, train_observer_cfg)
        else
            0;
        const alpha_bytes = try std.math.mul(usize, fitted.chunks.len, @sizeOf(f64));
        const candidate_live = try std.math.add(usize, try std.math.add(usize, frozen_bytes, fitted_bytes), try std.math.add(usize, candidate_numeric, candidate_maps));
        const branch_peak = try std.math.add(usize, branch_live_bytes, try std.math.add(usize, candidate_live, @max(@max(validation_bytes, train_validation_bytes), alpha_bytes)));
        if (branch_peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        // Observational only: same composed candidate, backend-null FP64, and
        // completed before the single independent heldout gate.  It cannot
        // alter selection, fitting, geometry, pass, or branch ordering.
        const direct_train_metrics = if (options.observe_direct_train)
            try solver.validationMetrics(allocator, original, candidate.head, train, train_observer_cfg)
        else
            null;
        const heldout_started = phaseStart(options.phase_timing);
        heldout_gate_calls = try std.math.add(usize, heldout_gate_calls, 1);
        const validation: solver.ValidationObservation = if (options.observe_validation_detail)
            try solver.validationMetricsDetailed(allocator, original, candidate.head, heldout, held_cfg)
        else
            .{ .metrics = try solver.validationMetrics(allocator, original, candidate.head, heldout, held_cfg), .detail = undefined };
        phaseEnd(options.phase_timing, .final_heldout, heldout_started);
        const metrics = validation.metrics;
        const heldout_detail: ?HeldoutDetail = if (options.observe_validation_detail) detail: {
            const observed = validation.detail;
            if (observed.original_peak_row >= original.rows or observed.compact_peak_row >= candidate.head.rows or candidate.source_indices.len != candidate.head.rows) return error.InvalidAttentionShape;
            const source_row = candidate.source_indices[observed.compact_peak_row];
            if (source_row >= original.rows) return error.InvalidAttentionShape;
            const original_bias = original.bias.?[observed.original_peak_row];
            const candidate_bias = candidate.head.bias.?[observed.compact_peak_row];
            const candidate_source_bias = original.bias.?[source_row];
            const reference_peak = try referencePeakEvidence(original, candidate, observed.original_peak_row);
            break :detail .{ .validation = observed, .candidate_peak_source_index = source_row, .original_peak_bias = original_bias, .candidate_peak_bias = candidate_bias, .candidate_peak_original_bias = candidate_source_bias, .original_peak_bias_weight = finiteExp(original_bias), .candidate_peak_bias_weight = finiteExp(candidate_bias), .candidate_peak_weight_ratio = finiteExp(candidate_bias - candidate_source_bias), .original_peak_retained = reference_peak.retained, .original_peak_exact_core = exactCoreContains(frozen, observed.original_peak_row), .original_peak_train_max_probability = try trainMaxProbability(cached, observed.original_peak_row, cfg.control), .candidate_peak_train_max_probability = try trainMaxProbability(cached, source_row, cfg.control), .original_peak_value_relative_l2 = reference_peak.value_relative_l2, .original_peak_candidate_weight_ratio = reference_peak.candidate_bias_weight };
        } else null;
        const alpha = try allocator.alloc(f64, fitted.chunks.len);
        for (fitted.chunks, alpha) |chunk, *a| a.* = chunk.alpha;
        const pass = metrics.max_relative_mass_error <= cfg.max_relative_mass_error and metrics.relative_output_rmse <= cfg.max_relative_output_rmse;
        observeRejectedBranch(options, divisor, pass, original, &candidate, train, heldout, metrics);
        branches[bi] = .{ .divisor = divisor, .candidate_rows = candidate.head.rows, .mass_error = metrics.max_relative_mass_error, .output_rmse = metrics.relative_output_rmse, .direct_train_metrics = direct_train_metrics, .heldout_detail = heldout_detail, .rejection = if (pass) "none" else "heldout_gate", .alpha = alpha, .training_diagnostics = fitted.training_diagnostics, .postfit = postfit };
        attachGeometry(allocator, &branches[bi], options, bi) catch |attach_err| {
            allocator.free(alpha);
            return attach_err;
        };
        initialized += 1;
        if (pass) {
            const duplicate_bytes = try std.math.mul(usize, initialized, @sizeOf(Branch));
            const accepted_peak = try std.math.add(usize, branch_live_bytes, try std.math.add(usize, candidate_live, try std.math.add(usize, try branchPayloadBytes(branches[bi]), duplicate_bytes)));
            if (accepted_peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            const kept_branches = try allocator.dupe(Branch, branches[0..initialized]);
            allocator.free(branches);
            keep = true;
            return .{ .allocator = allocator, .head = candidate.head, .source_indices = candidate.source_indices, .coverage_source_indices = candidate.coverage_source_indices, .branches = kept_branches, .accepted = true, .chosen_divisor = divisor, .final_metrics_valid = true, .final_mass_error = metrics.max_relative_mass_error, .final_output_rmse = metrics.relative_output_rmse, .scored_candidates_reused = options.scored_candidates != null, .scored_packets_reused = reused_packets, .reselections_avoided = reused_packets, .value_fit_dispatches_avoided = avoided_dispatches, .heldout_gate_calls = heldout_gate_calls, .geometry_policy = options.geometry_policy, .geometry_storage = options.geometry_storage, .geometry_workspace_bound_bytes = options.geometry_workspace_bound_bytes, .final_refit = .{ .strategy = options.final_refit_strategy, .attempts = refit_attempts, .completions = refit_completions }, .postfit = postfit };
        }
        branch_live_bytes = try std.math.add(usize, branch_live_bytes, try branchPayloadBytes(branches[bi]));
    }
    if (initialized != branches.len) return error.InvalidAttentionShape;
    const elements = try std.math.mul(usize, original.rows, original.dim);
    const fallback_numeric = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, elements, 2), original.rows), @sizeOf(f64));
    const fallback_maps = try std.math.mul(usize, try std.math.mul(usize, original.rows, 2), @sizeOf(usize));
    if (try std.math.add(usize, branch_live_bytes, try std.math.add(usize, fallback_numeric, fallback_maps)) > cfg.max_workspace_bytes)
        return error.CompactionWorkspaceLimit;
    const keys = try allocator.dupe(f64, original.keys[0..elements]);
    errdefer allocator.free(keys);
    const values = try allocator.dupe(f64, original.values[0..elements]);
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f64, original.bias.?[0..original.rows]);
    errdefer allocator.free(bias);
    const source_indices = try allocator.alloc(usize, original.rows);
    errdefer allocator.free(source_indices);
    const coverage = try allocator.alloc(usize, original.rows);
    errdefer allocator.free(coverage);
    for (source_indices, coverage, 0..) |*source, *end, index| {
        source.* = index;
        end.* = index;
    }
    return .{ .allocator = allocator, .head = .{ .keys = keys, .values = values, .bias = bias, .rows = original.rows, .dim = original.dim, .scale = original.scale, .softcap = original.softcap }, .source_indices = source_indices, .coverage_source_indices = coverage, .branches = branches, .accepted = false, .chosen_divisor = 0, .final_metrics_valid = false, .final_mass_error = 0, .final_output_rmse = 0, .scored_candidates_reused = options.scored_candidates != null, .scored_packets_reused = reused_packets, .reselections_avoided = reused_packets, .value_fit_dispatches_avoided = avoided_dispatches, .heldout_gate_calls = heldout_gate_calls, .geometry_policy = options.geometry_policy, .geometry_storage = options.geometry_storage, .geometry_workspace_bound_bytes = options.geometry_workspace_bound_bytes, .final_refit = .{ .strategy = options.final_refit_strategy, .attempts = refit_attempts, .completions = refit_completions }, .postfit = try aggregatePostfit(branches, options.postfit_policy) };
}

pub fn run(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config) !Result {
    if (comptime !@import("builtin").is_test) @compileError("joint chunk mass replay diagnostic entry is test-only");
    return runWithOptions(allocator, original, train, heldout, ranges, cfg, .{}, null, null);
}

/// v18 low-level entry after training-only scoring has frozen every branch's
/// targets. Identity targets are omitted from joint correction and copied
/// byte-exactly by chronological composition.
pub fn runFrozenAdaptive(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters) !Result {
    if (comptime !@import("builtin").is_test) @compileError("frozen adaptive diagnostic entry is test-only");
    return runFrozenAdaptiveObserved(allocator, original, train, heldout, ranges, adaptive_plan, cfg, cached, counters, false, false);
}

pub fn runFrozenAdaptiveObserved(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, observe_direct_train: bool, observe_validation_detail: bool) !Result {
    if (comptime !@import("builtin").is_test) @compileError("observed frozen adaptive diagnostic entry is test-only");
    return runFrozenAdaptiveObservedTimed(allocator, original, train, heldout, ranges, adaptive_plan, cfg, cached, counters, observe_direct_train, observe_validation_detail, null, .identity_reuse, null, .disabled, 0, null);
}

fn adaptiveCriticalCoreRows(plan: *const AdaptivePlan) !usize {
    var rows: usize = 0;
    for (plan.scored) |levels| {
        const first = levels[0] orelse return error.InvalidAttentionShape;
        if (first.frozen.chunks.len != 1) return error.InvalidAttentionShape;
        const core = first.frozen.chunks[0].exact_core_local;
        rows = try std.math.add(usize, rows, core.len);
        for (levels[1..]) |candidate| {
            const owned = candidate orelse return error.InvalidAttentionShape;
            if (owned.frozen.chunks.len != 1 or !std.mem.eql(usize, core, owned.frozen.chunks[0].exact_core_local)) return error.InvalidAttentionShape;
        }
    }
    return rows;
}

fn runFrozenAdaptivePageBackedChecked(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, critical_core_policy: CriticalCorePolicy, postfit_policy: PostfitPolicy, postfit_workspace_bytes: usize, postfit_hook: ?CandidatePostfitHook, validation: PageBackedValidation) !Result {
    try cfg.control.check();
    const support_digest = try frozenSupportBankDigest(adaptive_plan);
    const targets: [3][]const usize = .{ adaptive_plan.branches[0].targets, adaptive_plan.branches[1].targets, adaptive_plan.branches[2].targets };
    for (targets) |branch| if (branch.len != ranges.len) return error.InvalidFitConfig;
    const retained = try std.math.add(usize, try scoredPlanBytes(adaptive_plan), try adaptiveGeometryBytes(adaptive_plan));
    if (retained > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var replay_cfg = cfg;
    replay_cfg.max_workspace_bytes -= retained;
    var result = try runWithOptions(allocator, original, train, heldout, ranges, replay_cfg, .{
        .phase_timing = phase_timing,
        .mass_strategy = .identity_chunk_coefficients_v18,
        .adaptive_targets_by_branch = targets,
        .adaptive_geometry = &adaptive_plan.branches,
        .scored_candidates = adaptive_plan,
        .geometry_policy = adaptive_plan.geometry_policy,
        .observe_validation_detail = observe_validation_detail,
        .geometry_storage = adaptive_plan.geometry_storage,
        .geometry_workspace_bound_bytes = adaptive_plan.geometry_workspace_bound_bytes,
        .postfit_policy = postfit_policy,
        .postfit_workspace_bytes = postfit_workspace_bytes,
        .candidate_postfit_hook = postfit_hook,
        .validation_backend = validation.backend,
        .train_reference_summary_backend = validation.train_reference_summary_backend,
        .heldout_reference_summary_backend = validation.heldout_reference_summary_backend,
    }, null, null);
    result.weight_cap = adaptive_plan.weight_cap;
    result.frozen_support_bank_digest = support_digest;
    errdefer result.deinit();
    result.critical_core = .{ .policy = critical_core_policy, .policy_digest = criticalCorePolicyDigest(critical_core_policy), .unique_rows = if (critical_core_policy == .disabled) 0 else try adaptiveCriticalCoreRows(adaptive_plan) };
    return result;
}

/// Production entry after the private Engine route has authenticated source,
/// queries, ranges, cap and frozen candidate bank. It shares the exact checked
/// implementation with the diagnostic wrapper below.
pub fn runFrozenAdaptivePageBackedRuntime(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming) !Result {
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, .disabled, .disabled, 0, null, .{});
}

pub fn runFrozenAdaptivePageBackedRuntimeValidated(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, validation: PageBackedValidation) !Result {
    if (observe_validation_detail and validation.backend != null) return error.InvalidFitConfig;
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, .disabled, .disabled, 0, null, validation);
}

pub fn runFrozenAdaptivePageBackedRuntimeCritical(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, critical_core_policy: CriticalCorePolicy) !Result {
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, critical_core_policy, .disabled, 0, null, .{});
}

pub fn runFrozenAdaptivePageBackedRuntimeCriticalValidated(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, critical_core_policy: CriticalCorePolicy, validation: PageBackedValidation) !Result {
    if (critical_core_policy == .disabled or (observe_validation_detail and validation.backend != null)) return error.InvalidFitConfig;
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, critical_core_policy, .disabled, 0, null, validation);
}

pub fn runFrozenAdaptivePageBackedRuntimePostfit(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, critical_core_policy: CriticalCorePolicy, postfit_workspace_bytes: usize, postfit_hook: CandidatePostfitHook) !Result {
    if (critical_core_policy == .disabled) return error.InvalidFitConfig;
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, critical_core_policy, .train_global_residual_range_mass_l2_exact_f32_cache_v1, postfit_workspace_bytes, postfit_hook, .{});
}

pub fn runFrozenAdaptivePageBackedRuntimePostfitValidated(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, critical_core_policy: CriticalCorePolicy, postfit_workspace_bytes: usize, postfit_hook: CandidatePostfitHook, validation: PageBackedValidation) !Result {
    if (critical_core_policy == .disabled or (observe_validation_detail and validation.backend != null)) return error.InvalidFitConfig;
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, critical_core_policy, .train_global_residual_range_mass_l2_exact_f32_cache_v1, postfit_workspace_bytes, postfit_hook, validation);
}

/// DEV-only wrapper retained for existing diagnostic callers and fixtures.
pub fn runFrozenAdaptivePageBacked(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, observe_validation_detail: bool, phase_timing: ?*PhaseTiming) !Result {
    if (comptime !@import("builtin").is_test) @compileError("page-backed frozen adaptive diagnostic entry is test-only");
    return runFrozenAdaptivePageBackedChecked(allocator, original, train, heldout, ranges, adaptive_plan, cfg, observe_validation_detail, phase_timing, .disabled, .disabled, 0, null, .{});
}

fn runFrozenAdaptiveObservedTimed(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, observe_direct_train: bool, observe_validation_detail: bool, phase_timing: ?*PhaseTiming, final_refit_strategy: FinalRefitStrategy, validation_backend: ?solver.Backend, postfit_policy: PostfitPolicy, postfit_workspace_bytes: usize, rejected_branch_capture: ?RejectedBranchCapture) !Result {
    try cfg.control.check();
    try local_weight_cap.validate(adaptive_plan.weight_cap.configured);
    if (!std.math.isFinite(adaptive_plan.weight_cap.max_before) or !std.math.isFinite(adaptive_plan.weight_cap.max_after) or
        adaptive_plan.weight_cap.max_before < adaptive_plan.weight_cap.max_after or adaptive_plan.weight_cap.max_after < 0) return error.InvalidFitConfig;
    if (adaptive_plan.weight_cap.configured) |cap| {
        if (cap < cfg.min_weight or adaptive_plan.weight_cap.scored_candidates != try std.math.mul(usize, adaptive_plan.scored.len, 3) or
            adaptive_plan.weight_cap.max_after > cap or (adaptive_plan.weight_cap.clamped_weights > 0) != (adaptive_plan.weight_cap.max_before > cap)) return error.InvalidFitConfig;
    } else if (adaptive_plan.weight_cap.scored_candidates != 0 or adaptive_plan.weight_cap.clamped_weights != 0 or
        adaptive_plan.weight_cap.max_before != 0 or adaptive_plan.weight_cap.max_after != 0) return error.InvalidFitConfig;
    const targets: [3][]const usize = .{ adaptive_plan.branches[0].targets, adaptive_plan.branches[1].targets, adaptive_plan.branches[2].targets };
    const support_digest = try frozenSupportBankDigest(adaptive_plan);
    for (targets) |branch| if (branch.len != ranges.len) return error.InvalidFitConfig;
    const retained = try std.math.add(usize, try scoredPlanBytes(adaptive_plan), try adaptiveGeometryBytes(adaptive_plan));
    if (retained > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var replay_cfg = cfg;
    replay_cfg.max_workspace_bytes -= retained;
    var result = try runWithOptions(allocator, original, train, heldout, ranges, replay_cfg, .{ .phase_timing = phase_timing, .training_source = true, .mass_strategy = .identity_chunk_coefficients_v18, .final_refit_strategy = final_refit_strategy, .postfit_policy = postfit_policy, .postfit_workspace_bytes = postfit_workspace_bytes, .adaptive_targets_by_branch = targets, .adaptive_geometry = &adaptive_plan.branches, .scored_candidates = adaptive_plan, .geometry_policy = adaptive_plan.geometry_policy, .observe_direct_train = observe_direct_train, .observe_validation_detail = observe_validation_detail, .geometry_storage = adaptive_plan.geometry_storage, .geometry_workspace_bound_bytes = adaptive_plan.geometry_workspace_bound_bytes, .validation_backend = validation_backend, .rejected_branch_capture = rejected_branch_capture }, cached, counters);
    result.gram_memo = adaptive_plan.gram_memo;
    result.joint_fixed_memo = adaptive_plan.joint_fixed_memo;
    result.training_reduction = adaptive_plan.training_reduction;
    result.incremental_gram_chooser_calls = adaptive_plan.incremental_gram_chooser_calls;
    result.frozen_support_bank_digest = support_digest;
    result.weight_cap = adaptive_plan.weight_cap;
    return result;
}

/// DEV A/B oracle only. Repeats selection/value fits using the pre-reuse path;
/// never selected by production authority.
pub fn runFrozenAdaptiveUncachedOracle(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, adaptive_plan: *AdaptivePlan, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters) !Result {
    if (comptime !@import("builtin").is_test) @compileError("uncached frozen adaptive oracle is test-only");
    // The oracle repeats uncached selection and value fitting and therefore
    // cannot represent a cap already applied to retained scored packets.
    if (adaptive_plan.weight_cap.configured != null) return error.InvalidFitConfig;
    const targets: [3][]const usize = .{ adaptive_plan.branches[0].targets, adaptive_plan.branches[1].targets, adaptive_plan.branches[2].targets };
    return runWithOptions(allocator, original, train, heldout, ranges, cfg, .{ .training_source = true, .mass_strategy = .identity_chunk_coefficients_v18, .adaptive_targets_by_branch = targets, .adaptive_geometry = &adaptive_plan.branches, .geometry_policy = adaptive_plan.geometry_policy }, cached, counters);
}

/// Retains scored selections and rounded value fits, then uses their signed
/// complete-head TRAIN errors to freeze three divisor-budget geometries.
/// Heldout is absent by construction.
pub fn planAdaptiveGeometry(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters) !AdaptivePlan {
    return planAdaptiveGeometryStorage(allocator, original, train, ranges, cfg, cached, counters, .dense, 8, null);
}

pub fn planAdaptiveGeometryStorage(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, storage: GeometryStorage, gram_tile_rows: usize, gram_backend: ?gram_geometry.GramBackend) !AdaptivePlan {
    return planAdaptiveGeometryStorageCappedMemoized(allocator, original, train, ranges, cfg, cached, counters, storage, gram_tile_rows, gram_backend, null, false, false, false, 100, .legacy_full_scan, null, false, .scalar_reference, 64, null, .raw);
}

pub fn planAdaptiveGeometryStorageCapped(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, storage: GeometryStorage, gram_tile_rows: usize, gram_backend: ?gram_geometry.GramBackend, maximum: ?f64) !AdaptivePlan {
    return planAdaptiveGeometryStorageCappedMemoized(allocator, original, train, ranges, cfg, cached, counters, storage, gram_tile_rows, gram_backend, maximum, false, false, false, 100, .legacy_full_scan, null, false, .scalar_reference, 64, null, .raw);
}

fn validAcceleratedGramTileRows(capture_rows: usize, gram_rows: usize) bool {
    return gram_rows == 8 or gram_rows == 64 or (capture_rows == 256 and gram_rows == 256);
}

const PreparedScoredBank = struct {
    allocator: std.mem.Allocator,
    scores: []adaptive.RangeScores,
    scored: [][3]?ScoredCandidate,
    retained_bytes: usize,
    joint_memo_bytes: usize,
    joint_memo_builds: usize,
    joint_memo_hits: usize,
    joint_memo_enabled: bool,
    training_reduction: TrainingReductionEvidence,
    weight_cap: local_weight_cap.Evidence,
    owns_packets: bool = true,

    fn disown(self: *@This()) void {
        self.owns_packets = false;
    }

    fn deinit(self: *@This()) void {
        if (self.owns_packets) {
            for (self.scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
            self.allocator.free(self.scored);
            self.allocator.free(self.scores);
        }
        self.* = undefined;
    }
};

fn criticalCoreRows(bank: *const PreparedScoredBank) !usize {
    var rows: usize = 0;
    for (bank.scored) |levels| {
        const first = levels[0] orelse return error.InvalidAttentionShape;
        if (first.frozen.chunks.len != 1) return error.InvalidAttentionShape;
        rows = try std.math.add(usize, rows, first.frozen.chunks[0].exact_core_local.len);
        for (levels[1..]) |candidate| {
            const owned = candidate orelse return error.InvalidAttentionShape;
            if (owned.frozen.chunks.len != 1 or !std.mem.eql(usize, first.frozen.chunks[0].exact_core_local, owned.frozen.chunks[0].exact_core_local)) return error.InvalidAttentionShape;
        }
    }
    return rows;
}

fn digestBytes(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn rangePolicyIdentity() [32]u8 {
    return digestBytes("range_shared_fp64_gram_original_kkt_v1\x00module=metal_attention_match_lapack.prepareRawGram/v1");
}

fn sourceIdentity(original: solver.Head, authenticated_training_source_digest: [32]u8) [32]u8 {
    var state = std.crypto.hash.sha2.Sha256.init(.{});
    state.update("attention-match-range-source-identity.v1\x00");
    state.update(std.mem.asBytes(&original.rows));
    state.update(std.mem.asBytes(&original.dim));
    state.update(std.mem.sliceAsBytes(original.keys));
    state.update(std.mem.sliceAsBytes(original.values));
    if (original.bias) |bias| state.update(std.mem.sliceAsBytes(bias));
    state.update(std.mem.asBytes(&original.scale));
    state.update(std.mem.asBytes(&original.softcap));
    // This digest binds the exact captured TRAIN source revision and backend.
    state.update(&authenticated_training_source_digest);
    return state.finalResult();
}

fn criticalAuthorityIdentity(authority: ?*const critical_keys.Result) [32]u8 {
    var state = std.crypto.hash.sha2.Sha256.init(.{});
    state.update("attention-match-critical-authority-identity.v1\x00");
    if (authority) |value| {
        state.update(&value.probability_sha256);
        state.update(std.mem.sliceAsBytes(value.inner.ranges));
        state.update(std.mem.sliceAsBytes(value.inner.source_keys));
        state.update(std.mem.sliceAsBytes(value.inner.query_offsets));
        state.update(std.mem.sliceAsBytes(value.inner.critical_queries));
        state.update(std.mem.asBytes(&value.inner.query_rows));
        state.update(std.mem.asBytes(&value.inner.complete_query_rows));
        state.update(std.mem.asBytes(&value.inner.source_rows));
        state.update(std.mem.asBytes(&value.inner.threshold));
    }
    return state.finalResult();
}

const RawRangeLease = struct {
    backend: ?solver.Backend = null,
    token: ?solver.Backend.RawRangeSession = null,

    fn open(backend: solver.Backend, allocator: std.mem.Allocator, identity: solver.Backend.RawRangeIdentity, admitted_bytes: usize) !@This() {
        const begin = backend.raw_range_begin_fn orelse return error.UnsupportedOperation;
        if (backend.raw_range_end_fn == null) return error.UnsupportedOperation;
        return .{ .backend = backend, .token = try begin(backend.context, allocator, identity, admitted_bytes) };
    }

    fn deinit(self: *@This()) void {
        const token = self.token orelse return;
        self.token = null;
        self.backend.?.raw_range_end_fn.?(self.backend.?.context, token);
    }
};

fn prepareScoredBank(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, maximum: ?f64, joint_fixed_original_memo: bool, training_reduction_policy: TrainingReductionPolicy, training_reduction_query_batch_rows: usize, phase_timing: ?*PhaseTiming, selection_policy: selection.SelectionPolicy, critical_core_policy: CriticalCorePolicy, critical_authority: ?*const critical_keys.Result) !PreparedScoredBank {
    const raw_range_policy = if (cfg.backend) |backend| backend.raw_range_policy else .off;
    var cap_evidence: local_weight_cap.Evidence = .{ .configured = maximum };
    try cfg.control.check();
    try cached.validate(original, train, cfg);
    const scores = try allocator.alloc(adaptive.RangeScores, ranges.len);
    errdefer allocator.free(scores);
    const score_bytes = try std.math.mul(usize, scores.len, @sizeOf(adaptive.RangeScores));
    const scored = try allocator.alloc([3]?ScoredCandidate, ranges.len);
    errdefer allocator.free(scored);
    @memset(scored, .{ null, null, null });
    var retained: usize = 0;
    errdefer for (scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
    const table_bytes = try std.math.mul(usize, scored.len, @sizeOf([3]?ScoredCandidate));
    const plan_bytes = try std.math.add(usize, score_bytes, table_bytes);
    if (plan_bytes > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var score_cfg = cfg;
    score_cfg.max_workspace_bytes -= plan_bytes;
    var retained_bytes: usize = 0;
    // A critical candidate always needs the fixed-original term. With memo
    // enabled one range-owned term is reused by all three levels; with memo
    // disabled each level builds its own term, but the same bytes must still
    // be admitted while that level is live.
    const fixed_term_bytes = if (joint_fixed_original_memo or critical_authority != null) try joint.fixedOriginalTermBytes(cached.query_count, original.dim) else 0;
    // Public memo evidence remains disabled/zero when the caller did not ask
    // for reuse; fixed_term_bytes still admits the mandatory per-level term.
    const joint_memo_bytes = if (joint_fixed_original_memo) fixed_term_bytes else 0;
    var joint_memo_builds: usize = 0;
    var joint_memo_hits: usize = 0;
    var reduction_evidence: TrainingReductionEvidence = .{ .policy = training_reduction_policy };
    const reduction_backend = switch (training_reduction_policy) {
        .scalar_reference => null,
        .accelerate_fp64_cached_v1 => train_reduction.accelerateBackend() orelse return error.UnsupportedOperation,
    };
    const source_identity = sourceIdentity(original, cached.digest);
    const train_identity = digestBytes(std.mem.sliceAsBytes(train));
    const critical_identity = criticalAuthorityIdentity(critical_authority);
    for (ranges, scores, scored) |range, *score, *retained_levels| {
        const core_count = try criticalCount(critical_authority, range);
        // Scored packets represent compacted candidates. An exact identity
        // level has no FrozenOwned chunk and must never reach chunks[0].
        // Reject this rare authority shape before any candidate allocation;
        // a future identity-level implementation must model it explicitly.
        score.* = .{ .start = range.start, .rows = range.rows, .target_rows = try criticalAwareTargets(range.rows, core_count, critical_core_policy), .mass_harm = .{ 0, 0, 0, 0 }, .output_harm = .{ 0, 0, 0, 0 } };
        const q = cached.query_count;
        const qd = try std.math.mul(usize, q, original.dim);
        const range_vectors = try std.math.mul(usize, try std.math.add(usize, q, qd), @sizeOf(f64));
        if (try std.math.add(usize, retained_bytes, range_vectors) > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var accelerated_range: ?train_reduction.OwnedRange = null;
        defer if (accelerated_range) |*owned| owned.deinit();
        var scalar_range_rho: ?[]f64 = null;
        defer if (scalar_range_rho) |owned| allocator.free(owned);
        var scalar_range_y: ?[]f64 = null;
        defer if (scalar_range_y) |owned| allocator.free(owned);
        const original_started = phaseStart(phase_timing);
        if (reduction_backend == null) {
            scalar_range_rho = try allocator.alloc(f64, q);
            scalar_range_y = try allocator.alloc(f64, qd);
            @memset(scalar_range_rho.?, 0);
            @memset(scalar_range_y.?, 0);
            for (0..q) |qi| {
                try cfg.control.check();
                for (0..range.rows) |local| {
                    const probability = try cached.probability(qi, range.start + local);
                    scalar_range_rho.?[qi] += probability;
                    for (0..original.dim) |di| scalar_range_y.?[qi * original.dim + di] += probability * original.values[(range.start + local) * original.dim + di];
                }
            }
        }
        if (reduction_backend == null) phaseEnd(phase_timing, .local_original_rho_y, original_started);
        var fixed_original: ?joint.FixedOriginalTerm = null;
        defer if (fixed_original) |*owned| owned.deinit();
        if (joint_fixed_original_memo and critical_authority == null) {
            const fixed_external = try std.math.add(usize, retained_bytes, if (reduction_backend == null) range_vectors else 0);
            fixed_original = try joint.prepareFixedOriginalTerm(allocator, original, train, .{ .start = range.start, .rows = range.rows, .selected_local = &.{}, .weights = &.{} }, cfg, cached, fixed_external, score_cfg.max_workspace_bytes);
            joint_memo_builds = try std.math.add(usize, joint_memo_builds, 1);
        }
        const range_gram_bytes = if (raw_range_policy == .range_shared_fp64_gram_original_kkt_v1) blk: {
            const backend = cfg.backend orelse return error.InvalidFitConfig;
            const workspace = backend.raw_range_workspace_fn orelse return error.UnsupportedOperation;
            break :blk try workspace(backend.context, range.rows);
        } else 0;
        const range_live = try std.math.add(usize, try std.math.add(usize, retained_bytes, if (reduction_backend == null) range_vectors else 0), try std.math.add(usize, fixed_term_bytes, range_gram_bytes));
        if (range_live > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var range_lease: RawRangeLease = .{};
        if (raw_range_policy == .range_shared_fp64_gram_original_kkt_v1) {
            const backend = cfg.backend orelse return error.InvalidFitConfig;
            if (range.rows > 1024) return error.UnsupportedOperation;
            range_lease = try RawRangeLease.open(backend, allocator, .{
                .range_start = range.start,
                .range_rows = range.rows,
                .query_rows = cached.query_count,
                .source_rows = original.rows,
                .source_identity = source_identity,
                .train_identity = train_identity,
                .critical_authority_identity = critical_identity,
                .policy_identity = rangePolicyIdentity(),
            }, range_gram_bytes);
        }
        defer range_lease.deinit();
        for (0..3) |li| {
            try cfg.control.check();
            const retained_and_range = try std.math.add(usize, try std.math.add(usize, retained_bytes, if (reduction_backend == null) range_vectors else 0), fixed_term_bytes);
            if (retained_and_range > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            var candidate_cfg = score_cfg;
            candidate_cfg.max_workspace_bytes -= retained_and_range;
            if (range_lease.token) |token| {
                if (candidate_cfg.max_workspace_bytes < range_gram_bytes) return error.CompactionWorkspaceLimit;
                candidate_cfg.max_workspace_bytes -= range_gram_bytes;
                candidate_cfg.raw_range_session = token;
                candidate_cfg.raw_range_policy = raw_range_policy;
            }
            const one_range = [_]admission.Range{range};
            const one_target = [_]usize{score.target_rows[li]};
            var transferred = false;
            const freeze_started = phaseStart(phase_timing);
            var frozen = try freezeBranchPolicy(allocator, original, train, &one_range, 1, &one_target, candidate_cfg, cached, counters, selection_policy, critical_authority);
            phaseEnd(phase_timing, .selection_freeze, freeze_started);
            errdefer if (!transferred) frozen.deinit();
            if (maximum != null) for (frozen.weights) |weights| {
                try cap_evidence.merge(try local_weight_cap.apply(weights, maximum, cfg.control));
            };
            if (maximum != null) cap_evidence.scored_candidates = std.math.add(usize, cap_evidence.scored_candidates, 1) catch return error.InvalidAttentionShape;
            const current_frozen_bytes = try frozenBytes(frozen);
            if (current_frozen_bytes > candidate_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            var fit_cfg = candidate_cfg;
            fit_cfg.max_workspace_bytes -= current_frozen_bytes;
            var candidate_fixed: ?joint.FixedOriginalTerm = null;
            defer if (candidate_fixed) |*owned| owned.deinit();
            if (critical_authority != null) {
                const fixed_external = try std.math.add(usize, retained_bytes, try std.math.add(usize, if (reduction_backend == null) range_vectors else 0, current_frozen_bytes));
                if (joint_fixed_original_memo) {
                    if (fixed_original == null) {
                        fixed_original = try joint.prepareFixedOriginalTerm(allocator, original, train, frozen.chunks[0], cfg, cached, fixed_external, score_cfg.max_workspace_bytes);
                        joint_memo_builds = try std.math.add(usize, joint_memo_builds, 1);
                    }
                } else {
                    candidate_fixed = try joint.prepareFixedOriginalTerm(allocator, original, train, frozen.chunks[0], cfg, cached, fixed_external, score_cfg.max_workspace_bytes);
                    joint_memo_builds = try std.math.add(usize, joint_memo_builds, 1);
                }
            }
            const fit_started = phaseStart(phase_timing);
            var fitted = try joint.fitStrategyWithFixedOriginal(allocator, original, train, frozen.chunks, fit_cfg, cached, counters, .identity_chunk_coefficients_v18, false, if (candidate_fixed) |*owned| owned else if (fixed_original) |*owned| owned else null);
            if (joint_fixed_original_memo and fixed_original != null) joint_memo_hits = try std.math.add(usize, joint_memo_hits, 1);
            phaseEnd(phase_timing, .joint_value_fit, fit_started);
            errdefer if (!transferred) fitted.deinit();
            try round16(fitted.values);
            try round32(fitted.bias);
            const scored_fitted_bytes = try std.math.add(usize, try std.math.mul(usize, fitted.chunks.len, @sizeOf(joint.ChunkResult)), try std.math.add(usize, try std.math.mul(usize, fitted.values.len, @sizeOf(f64)), try std.math.mul(usize, fitted.bias.len, @sizeOf(f64))));
            if (reduction_backend != null) {
                retained_bytes = try std.math.add(usize, retained_bytes, try std.math.add(usize, current_frozen_bytes, scored_fitted_bytes));
                if (retained_bytes > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
                retained_levels[li] = .{ .frozen = frozen, .fitted = fitted };
                transferred = true;
                retained += 1;
                continue;
            }
            const range_rho = scalar_range_rho.?;
            const range_y = scalar_range_y.?;
            // range_rho+range_y are already live and charged by range_vectors.
            // The rounded-harm phase adds only candidate_rho+added_y; the
            // measurement itself is allocation-free. Charging two additional
            // vector sets spuriously rejects tight but valid critical banks.
            const candidate_vectors = if (reduction_backend == null) range_vectors else 0;
            const chunk = frozen.chunks[0];
            const reduction_extra = if (reduction_backend != null)
                try std.math.add(usize, try train_reduction.harmWorkspaceBytes(q, chunk.selected_local.len, original.dim, training_reduction_query_batch_rows), try std.math.mul(usize, chunk.selected_local.len, @sizeOf(f64)))
            else
                0;
            const score_peak = try std.math.add(usize, try std.math.add(usize, retained_bytes, try std.math.add(usize, range_vectors, fixed_term_bytes)), try std.math.add(usize, current_frozen_bytes, try std.math.add(usize, scored_fitted_bytes, try std.math.add(usize, candidate_vectors, reduction_extra))));
            if (score_peak > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            const harm_started = phaseStart(phase_timing);
            var candidate_rho: ?[]f64 = null;
            defer if (candidate_rho) |owned| allocator.free(owned);
            var added_y: ?[]f64 = null;
            defer if (added_y) |owned| allocator.free(owned);
            for (chunk.selected_local) |local| for (0..original.dim) |di| {
                const source_key = original.keys[(range.start + local) * original.dim + di];
                const rounded_key: f64 = @floatCast(@as(f16, @floatCast(source_key)));
                if (@as(u64, @bitCast(source_key)) != @as(u64, @bitCast(rounded_key))) return error.V18TrainingScoreUnsupported;
            };
            const measured = if (reduction_backend) |backend| accelerated: {
                const ratios = try allocator.alloc(f64, chunk.selected_local.len);
                defer allocator.free(ratios);
                for (chunk.selected_local, ratios, 0..) |local, *ratio, selected_i| {
                    const global = range.start + local;
                    ratio.* = @exp(fitted.bias[selected_i] - original.bias.?[global]);
                    if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
                }
                const harm = try train_reduction.measureSelectedHarm(allocator, .{ .context = cached, .query_count = q, .row_count = original.rows, .copy_fn = cachedProbabilityCopy }, range.start, range.rows, chunk.selected_local, ratios, fitted.values, cached.outputs, range_rho, range_y, original.dim, 1e-12, .{ .backend = backend, .query_batch_rows = training_reduction_query_batch_rows, .external_live_bytes = score_peak - reduction_extra + try std.math.mul(usize, ratios.len, @sizeOf(f64)), .max_workspace_bytes = score_cfg.max_workspace_bytes, .control = cfg.control });
                try mergeReductionEvidence(&reduction_evidence, harm.evidence, false);
                break :accelerated adaptive.Harm{ .max_relative_mass = harm.max_relative_mass, .relative_output_rmse = harm.relative_output_rmse };
            } else scalar: {
                candidate_rho = try allocator.alloc(f64, q);
                added_y = try allocator.alloc(f64, qd);
                @memset(candidate_rho.?, 0);
                @memset(added_y.?, 0);
                for (0..q) |qi| {
                    try cfg.control.check();
                    for (chunk.selected_local, 0..) |local, selected_i| {
                        const global = range.start + local;
                        const ratio = @exp(fitted.bias[selected_i] - original.bias.?[global]);
                        if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
                        const probability = try cached.probability(qi, global) * ratio;
                        candidate_rho.?[qi] += probability;
                        for (0..original.dim) |di| added_y.?[qi * original.dim + di] += probability * fitted.values[selected_i * original.dim + di];
                    }
                }
                break :scalar try adaptive.measureRoundedCandidateHarm(cached.outputs, range_rho, range_y, candidate_rho.?, added_y.?, q, original.dim, 1e-12);
            };
            phaseEnd(phase_timing, .rounded_harm, harm_started);
            score.mass_harm[li] = measured.max_relative_mass;
            score.output_harm[li] = measured.relative_output_rmse;
            retained_bytes = try std.math.add(usize, retained_bytes, try std.math.add(usize, current_frozen_bytes, scored_fitted_bytes));
            if (retained_bytes > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            retained_levels[li] = .{ .frozen = frozen, .fitted = fitted };
            transferred = true;
            retained += 1;
        }
        if (reduction_backend) |backend| {
            if (fixed_original) |*owned| owned.deinit();
            fixed_original = null;
            const deferred_original_started = phaseStart(phase_timing);
            accelerated_range = try train_reduction.reduceRangeOwned(allocator, .{ .context = cached, .query_count = q, .row_count = original.rows, .copy_fn = cachedProbabilityCopy }, range.start, range.rows, original.values[range.start * original.dim ..][0 .. range.rows * original.dim], original.dim, .{ .backend = backend, .query_batch_rows = training_reduction_query_batch_rows, .external_live_bytes = retained_bytes, .max_workspace_bytes = score_cfg.max_workspace_bytes, .control = cfg.control });
            try mergeReductionEvidence(&reduction_evidence, accelerated_range.?.evidence, true);
            phaseEnd(phase_timing, .local_original_rho_y, deferred_original_started);
            const range_rho = accelerated_range.?.rho;
            const range_y = accelerated_range.?.y;
            for (retained_levels, 0..) |*stored, li| {
                try cfg.control.check();
                const candidate = stored.* orelse return error.InvalidAttentionShape;
                const chunk = candidate.frozen.chunks[0];
                const harm_workspace = try train_reduction.harmWorkspaceBytes(q, chunk.selected_local.len, original.dim, training_reduction_query_batch_rows);
                const ratio_bytes = try std.math.mul(usize, chunk.selected_local.len, @sizeOf(f64));
                const score_peak = try std.math.add(usize, retained_bytes, try std.math.add(usize, range_vectors, try std.math.add(usize, harm_workspace, ratio_bytes)));
                if (score_peak > score_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
                const harm_started = phaseStart(phase_timing);
                const ratios = try allocator.alloc(f64, chunk.selected_local.len);
                defer allocator.free(ratios);
                for (chunk.selected_local, ratios, 0..) |local, *ratio, selected_i| {
                    const global = range.start + local;
                    ratio.* = @exp(candidate.fitted.bias[selected_i] - original.bias.?[global]);
                    if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
                }
                const harm = try train_reduction.measureSelectedHarm(allocator, .{ .context = cached, .query_count = q, .row_count = original.rows, .copy_fn = cachedProbabilityCopy }, range.start, range.rows, chunk.selected_local, ratios, candidate.fitted.values, cached.outputs, range_rho, range_y, original.dim, 1e-12, .{ .backend = backend, .query_batch_rows = training_reduction_query_batch_rows, .external_live_bytes = score_peak - harm_workspace, .max_workspace_bytes = score_cfg.max_workspace_bytes, .control = cfg.control });
                try mergeReductionEvidence(&reduction_evidence, harm.evidence, false);
                score.mass_harm[li] = harm.max_relative_mass;
                score.output_harm[li] = harm.relative_output_rmse;
                phaseEnd(phase_timing, .rounded_harm, harm_started);
            }
        }
    }
    std.debug.assert(retained == ranges.len * 3);
    return .{ .allocator = allocator, .scores = scores, .scored = scored, .retained_bytes = retained_bytes, .joint_memo_bytes = joint_memo_bytes, .joint_memo_builds = joint_memo_builds, .joint_memo_hits = joint_memo_hits, .joint_memo_enabled = joint_fixed_original_memo, .training_reduction = reduction_evidence, .weight_cap = cap_evidence };
}

test "critical scored bank runtime peak stays within planner bound" {
    const allocator = std.testing.allocator;
    const rows: usize = 64;
    const dim: usize = 64;
    const q: usize = 128;
    const cap: usize = 64 * 1024 * 1024;
    const keys = try allocator.alloc(f64, rows * dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f64, rows * dim);
    defer allocator.free(values);
    const bias = try allocator.alloc(f64, rows);
    defer allocator.free(bias);
    @memset(keys, 0);
    @memset(values, 1);
    @memset(bias, 0);
    const train = try allocator.alloc(f64, q * dim);
    defer allocator.free(train);
    @memset(train, 0);
    const probabilities = try allocator.alloc(f32, q * rows);
    defer allocator.free(probabilities);
    const tail: f32 = @floatCast(0.79 / 57.0);
    for (0..q) |qi| {
        @memset(probabilities[qi * rows ..][0..7], 0.03);
        @memset(probabilities[qi * rows + 7 ..][0 .. rows - 7], tail);
    }
    const outputs = try allocator.alloc(f64, q * dim);
    defer allocator.free(outputs);
    var probability_sum: f64 = 0;
    for (probabilities[0..rows]) |probability| probability_sum += probability;
    @memset(outputs, probability_sum);
    const head: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = rows, .dim = dim, .scale = 1 };
    var source: training_source.Source = .{ .allocator = allocator, .keys_ptr = keys.ptr, .values_ptr = values.ptr, .bias_ptr = bias.ptr, .train_ptr = train.ptr, .rows = rows, .dim = dim, .query_count = q, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 64, .cache_bytes = probabilities.len * @sizeOf(f32) + outputs.len * @sizeOf(f64), .consumer_peak_bytes = cap, .digest = std.mem.zeroes([32]u8), .probabilities_f32 = probabilities, .outputs = outputs, .owns_outputs = false };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 16 }};
    var authority = try extractCriticalAuthority(allocator, &source, &ranges, 0, cap, .{});
    defer authority.deinit();
    try std.testing.expectEqual(@as(usize, 7), try criticalCount(&authority, ranges[0]));
    const UnitColumnValueBackend = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: f64) !usize {
            return 0;
        }
        fn fit(_: ?*anyopaque, _: std.mem.Allocator, a: []const f64, y: []const f64, fit_q: usize, m: usize, fit_d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
            if (m != 1 or a.len != fit_q or y.len != fit_q * fit_d or out.len != fit_d or !std.math.isFinite(ridge) or ridge <= 0) return error.InvalidFitConfig;
            try control.check();
            var denominator = ridge;
            for (a) |value| {
                if (!std.math.isFinite(value)) return error.NonFiniteAttention;
                denominator += value * value;
            }
            if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
            // Validate every result before publishing any output. The second
            // pass is deterministic and cannot fail, preserving atomicity
            // without scratch allocation.
            for (0..fit_d) |di| {
                var numerator: f64 = 0;
                for (0..fit_q) |qi| {
                    const target = y[qi * fit_d + di];
                    if (!std.math.isFinite(target)) return error.NonFiniteAttention;
                    numerator += a[qi] * target;
                }
                if (!std.math.isFinite(numerator) or !std.math.isFinite(numerator / denominator)) return error.NonFiniteAttention;
            }
            try control.check();
            for (out, 0..) |*result, di| {
                var numerator: f64 = 0;
                for (0..fit_q) |qi| numerator += a[qi] * y[qi * fit_d + di];
                result.* = numerator / denominator;
            }
        }
        const backend: solver.ValueBackend = .{ .workspace_fn = workspace, .fit_fn = fit };
    };
    const cfg: solver.Config = .{ .target_rows = 8, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .value_backend = UnitColumnValueBackend.backend, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = cap };
    const totals = [_][3]usize{.{ 8, 8, 8 }};
    const residuals = [_][3]usize{.{ 1, 1, 1 }};
    const fixed_bytes = try joint.fixedOriginalTermBytes(q, dim);
    const planned = try replay_planner.planAdaptiveForStorageRefitJointMemoValidationReductionCritical(allocator, head, q, 2, &ranges, cfg, 3 * @sizeOf(Branch), 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, fixed_bytes, null, 0, .{ .total_rows = &totals, .residual_rows = &residuals });
    try std.testing.expect(planned.scoring_peak_bytes < 1 << 20);
    source.consumer_peak_bytes = planned.scoring_peak_bytes;
    var counters: training_source.Counters = .{};
    var admitted_cfg = cfg;
    admitted_cfg.max_workspace_bytes = planned.scoring_peak_bytes;
    source.consumer_peak_bytes = planned.scoring_peak_bytes;
    counters = .{};
    var bank = try prepareScoredBank(allocator, head, train, &ranges, admitted_cfg, &source, &counters, null, true, .scalar_reference, 64, null, .raw, .train_probability_gte_0_025_no_top1_v1, &authority);
    defer bank.deinit();
    try std.testing.expect(bank.joint_memo_enabled);
    try std.testing.expectEqual(@as(usize, 1), bank.joint_memo_builds);
    try std.testing.expectEqual(@as(usize, 3), bank.joint_memo_hits);
    try std.testing.expectEqual([4]usize{ 8, 8, 8, 16 }, bank.scores[0].target_rows);
    const range_vectors = (q + q * dim) * @sizeOf(f64);
    const plan_bytes = ranges.len * (@sizeOf(adaptive.RangeScores) + @sizeOf([3]?ScoredCandidate));
    var retained_before: usize = 0;
    var runtime_postfit_peak: usize = 0;
    for (bank.scored[0]) |candidate| {
        const owned = candidate orelse return error.TestUnexpectedResult;
        const frozen_bytes = try frozenBytes(owned.frozen);
        const fitted_bytes = @sizeOf(joint.ChunkResult) + (owned.fitted.values.len + owned.fitted.bias.len) * @sizeOf(f64);
        const candidate_peak = plan_bytes + retained_before + fixed_bytes + frozen_bytes + fitted_bytes;
        runtime_postfit_peak = @max(runtime_postfit_peak, candidate_peak);
        retained_before += frozen_bytes + fitted_bytes;
        try std.testing.expectEqual(@as(usize, 7), owned.frozen.chunks[0].exact_core_local.len);
        for (owned.frozen.chunks[0].exact_core_local) |local| {
            const selected = std.mem.indexOfScalar(usize, owned.frozen.chunks[0].selected_local, local) orelse return error.TestUnexpectedResult;
            try std.testing.expectEqual(@as(f64, 1), owned.frozen.chunks[0].weights[selected]);
            try std.testing.expectEqualSlices(f64, head.values[local * dim ..][0..dim], owned.fitted.values[selected * dim ..][0..dim]);
            try std.testing.expectEqual(head.bias.?[local], owned.fitted.bias[selected]);
        }
    }
    try std.testing.expect(runtime_postfit_peak <= planned.scoring_peak_bytes);
    try std.testing.expect(runtime_postfit_peak + range_vectors > runtime_postfit_peak);
    if (train_reduction.accelerateBackend()) |_| {
        const reduction_workspace = try reductionWorkspaceBytes(q, ranges[0].rows, 8, dim, 64);
        const accelerated_plan = try replay_planner.planAdaptiveForStorageRefitJointMemoValidationReductionCritical(allocator, head, q, 2, &ranges, cfg, 3 * @sizeOf(Branch), 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, fixed_bytes, null, reduction_workspace, .{ .total_rows = &totals, .residual_rows = &residuals });
        var accelerated_cfg = cfg;
        accelerated_cfg.max_workspace_bytes = accelerated_plan.scoring_peak_bytes;
        source.consumer_peak_bytes = accelerated_plan.scoring_peak_bytes;
        var accelerated = try prepareScoredBank(allocator, head, train, &ranges, accelerated_cfg, &source, &counters, null, true, .accelerate_fp64_cached_v1, 64, null, .raw, .train_probability_gte_0_025_no_top1_v1, &authority);
        defer accelerated.deinit();
        for (bank.scores[0].mass_harm, accelerated.scores[0].mass_harm) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
        for (bank.scores[0].output_harm, accelerated.scores[0].output_harm) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
    }
    for (keys) |value| try std.testing.expectEqual(@as(f64, 0), value);
    for (values) |value| try std.testing.expectEqual(@as(f64, 1), value);
    for (bias) |value| try std.testing.expectEqual(@as(f64, 0), value);
}

fn planAdaptiveGeometryStorageCappedMemoized(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, counters: *training_source.Counters, storage: GeometryStorage, gram_tile_rows: usize, gram_backend: ?gram_geometry.GramBackend, maximum: ?f64, memoize_original_delta: bool, train_budget_exchange: bool, joint_fixed_original_memo: bool, geometry_budget_percent: u8, geometry_chooser_policy: GeometryChooserPolicy, delta_backend: ?gram_delta.Backend, require_delta_accelerated: bool, training_reduction_policy: TrainingReductionPolicy, training_reduction_query_batch_rows: usize, phase_timing: ?*PhaseTiming, selection_policy: selection.SelectionPolicy) !AdaptivePlan {
    if ((delta_backend != null or require_delta_accelerated) and (storage != .per_query_gram or memoize_original_delta)) return error.InvalidFitConfig;
    if (require_delta_accelerated and delta_backend == null) return error.AcceleratedDeltaUnsupported;
    if ((delta_backend != null or require_delta_accelerated) and !validAcceleratedGramTileRows(cached.query_batch_rows, gram_tile_rows)) return error.InvalidFitConfig;
    try local_weight_cap.validate(maximum);
    if (!gram_geometry.validBudgetPercent(geometry_budget_percent)) return error.InvalidFitConfig;
    if (geometry_chooser_policy == .incremental_gram_v1 and
        (storage != .per_query_gram or train_budget_exchange))
        return error.InvalidFitConfig;
    if (geometry_budget_percent != 100 and (storage != .per_query_gram or train_budget_exchange or maximum != null)) return error.InvalidFitConfig;
    if (maximum) |cap| if (cap < cfg.min_weight) return error.InvalidFitConfig;
    var bank = try prepareScoredBank(allocator, original, train, ranges, cfg, cached, counters, maximum, joint_fixed_original_memo, training_reduction_policy, training_reduction_query_batch_rows, phase_timing, selection_policy, .disabled, null);
    defer bank.deinit();
    var plan = try planAdaptiveGeometryFromPreparedBank(allocator, original, train, ranges, cfg, cached, storage, gram_tile_rows, gram_backend, memoize_original_delta, train_budget_exchange, geometry_budget_percent, geometry_chooser_policy, delta_backend, require_delta_accelerated, phase_timing, &bank);
    plan.owns_scored_packets = true;
    bank.disown();
    return plan;
}

fn planAdaptiveGeometryFromPreparedBank(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, ranges: []const admission.Range, cfg: solver.Config, cached: *training_source.Source, storage: GeometryStorage, gram_tile_rows: usize, gram_backend: ?gram_geometry.GramBackend, memoize_original_delta: bool, train_budget_exchange: bool, geometry_budget_percent: u8, geometry_chooser_policy: GeometryChooserPolicy, delta_backend: ?gram_delta.Backend, require_delta_accelerated: bool, phase_timing: ?*PhaseTiming, bank: *PreparedScoredBank) !AdaptivePlan {
    if (phase_timing != null) std.debug.print("attention_match_gram_geometry_entry budget_percent={d} chooser={s} storage={s} query_batch_rows={d} gram_tile_rows={d} train_budget_exchange={} memoize_original_delta={} delta_backend={} require_delta_accelerated={} score_ranges={d} input_ranges={d} diagnostic_only=true\n", .{ geometry_budget_percent, @tagName(geometry_chooser_policy), @tagName(storage), cached.query_batch_rows, gram_tile_rows, train_budget_exchange, memoize_original_delta, delta_backend != null, require_delta_accelerated, bank.scores.len, ranges.len });
    cached.validate(original, train, cfg) catch |err| {
        if (phase_timing != null) std.debug.print("attention_match_gram_geometry_failure stage=cache_validate budget_percent={d} error={s} diagnostic_only=true\n", .{ geometry_budget_percent, @errorName(err) });
        return err;
    };
    if (bank.scores.len != ranges.len or bank.scored.len != ranges.len) {
        if (phase_timing != null) std.debug.print("attention_match_gram_geometry_failure stage=bank_shape budget_percent={d} scores={d} scored={d} ranges={d} error=InvalidFitConfig diagnostic_only=true\n", .{ geometry_budget_percent, bank.scores.len, bank.scored.len, ranges.len });
        return error.InvalidFitConfig;
    }
    const scores = bank.scores;
    const scored = bank.scored;
    const joint_memo_bytes = bank.joint_memo_bytes;
    const joint_memo_builds = bank.joint_memo_builds;
    const joint_memo_hits = bank.joint_memo_hits;
    const reduction_evidence = bank.training_reduction;
    const cap_evidence = bank.weight_cap;
    const scored_view: AdaptivePlan = .{ .allocator = allocator, .scores = scores, .scored = scored, .branches = undefined, .geometry_policy = adaptive.signed_joint_policy, .geometry_storage = storage, .weight_cap = cap_evidence };
    if (storage == .per_query_gram) {
        const GramContext = struct {
            plan: *const AdaptivePlan,
            original: solver.Head,
            cached: *const training_source.Source,
            fn fill(raw: ?*anyopaque, range_index: usize, level: usize, first: usize, count: usize, delta_mass: []f64, delta_numerator: []f64, control: solver.Control) !void {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                return accumulateScoredDeltaSlice(self.plan, range_index, level, self.original, self.cached, first, count, control, delta_mass, delta_numerator);
            }
        };
        const AcceleratedGramContext = struct {
            allocator: std.mem.Allocator,
            plan: *const AdaptivePlan,
            original: solver.Head,
            train: []const f64,
            cfg: solver.Config,
            cached: *const training_source.Source,
            backend: gram_delta.Backend,
            ratios: []f64,
            external_live_bytes: usize,
            combined_cap: usize,
            budget_percent: u8,
            diagnostic: bool,
            evidence: gram_delta_page.Evidence = .{},

            fn fill(raw: ?*anyopaque, range_index: usize, level: usize, first: usize, count: usize, delta_mass: []f64, delta_numerator: []f64, control: solver.Control) !void {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                if (range_index >= self.plan.scores.len or level >= 3) return error.InvalidAttentionShape;
                const candidate = self.plan.scored[range_index][level] orelse return error.InvalidAttentionShape;
                if (candidate.frozen.chunks.len != 1 or candidate.frozen.ranges.len != 1) return error.InvalidAttentionShape;
                const chunk = candidate.frozen.chunks[0];
                const range = candidate.frozen.ranges[0];
                if (chunk.selected_local.len > self.ratios.len or candidate.fitted.bias.len != chunk.selected_local.len) return error.InvalidAttentionShape;
                const ratios = self.ratios[0..chunk.selected_local.len];
                for (chunk.selected_local, candidate.fitted.bias, ratios) |local, fitted_bias, *ratio| {
                    if (local >= range.rows) return error.InvalidAttentionShape;
                    ratio.* = @exp(fitted_bias - self.original.bias.?[range.start + local]);
                    if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
                }
                gram_delta_page.fill(self.allocator, self.cached, self.original, self.train, self.cfg, range.start, range.rows, chunk.selected_local, ratios, candidate.fitted.values, first, count, self.backend, self.external_live_bytes, self.combined_cap, control, delta_mass, delta_numerator, &self.evidence) catch |err| {
                    if (self.diagnostic) std.debug.print("attention_match_gram_geometry_failure stage=delta_accessor budget_percent={d} range={d} level={d} first={d} count={d} error={s} diagnostic_only=true\n", .{ self.budget_percent, range_index, level, first, count, @errorName(err) });
                    return err;
                };
            }
        };
        const DiagnosticGramBackend = struct {
            inner: gram_geometry.GramBackend,
            budget_percent: u8,
            diagnostic: bool,

            fn workspace(raw: ?*anyopaque, k: usize, dim: usize) !usize {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                return self.inner.workspace_fn(self.inner.context, k, dim);
            }

            fn compute(raw: ?*anyopaque, matrix: []const f64, k: usize, dim: usize, output: []f64, scratch: []u8, control: solver.Control) !void {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                self.inner.compute_fn(self.inner.context, matrix, k, dim, output, scratch, control) catch |err| {
                    if (self.diagnostic) std.debug.print("attention_match_gram_geometry_failure stage=gram_backend budget_percent={d} k={d} dim={d} error={s} diagnostic_only=true\n", .{ self.budget_percent, k, dim, @errorName(err) });
                    return err;
                };
            }

            fn backend(self: *@This()) gram_geometry.GramBackend {
                return .{ .context = self, .workspace_fn = workspace, .compute_fn = compute };
            }
        };
        if (gram_backend == null) return error.V18TrainingScoreUnsupported;
        const scored_bytes = try scoredPlanBytes(&scored_view);
        const temporary_ranges = try std.math.mul(usize, ranges.len, @sizeOf(gram_geometry.Range));
        const memo_bytes = if (memoize_original_delta) try gramMemoBytes(@min(cached.query_count, gram_tile_rows), original.dim) else 0;
        var external_for_build = try std.math.add(usize, scored_bytes, temporary_ranges);
        external_for_build = try std.math.add(usize, external_for_build, memo_bytes);
        const candidate_count = try std.math.mul(usize, ranges.len, 3);
        const backend_scratch = try gram_backend.?.workspace_fn(gram_backend.?.context, candidate_count, original.dim);
        const gram_owned_peak = try gram_geometry.workspaceBytes(cached.query_count, original.dim, ranges.len, gram_tile_rows, backend_scratch);
        const one_geometry_payload = try std.math.mul(usize, ranges.len, @sizeOf(adaptive.Level) + @sizeOf(usize));
        const all_geometry_payload = try std.math.mul(usize, 3, one_geometry_payload);
        const exchange_chooser = if (train_budget_exchange) try std.math.mul(usize, ranges.len, @sizeOf(u8) + @sizeOf(adaptive.Level) + @sizeOf(usize)) else 0;
        const exchange_peak = if (train_budget_exchange) try std.math.add(usize, exchange_chooser, one_geometry_payload) else 0;
        var max_range_rows: usize = 0;
        var max_selected_rows: usize = 0;
        for (ranges, scores) |range, score| {
            max_range_rows = @max(max_range_rows, range.rows);
            max_selected_rows = @max(max_selected_rows, score.target_rows[2]);
        }
        const ratio_bytes = if (delta_backend != null) try std.math.mul(usize, max_selected_rows, @sizeOf(f64)) else 0;
        const delta_workspace = if (delta_backend != null) try gram_delta_page.workspaceBytes(@min(cached.query_count, gram_tile_rows), max_range_rows, max_selected_rows, original.dim) else 0;
        const accelerated_extra = try std.math.add(usize, ratio_bytes, delta_workspace);
        const full_gram_peak = try std.math.add(usize, external_for_build, try std.math.add(usize, accelerated_extra, try std.math.add(usize, gram_owned_peak, try std.math.add(usize, all_geometry_payload, exchange_peak))));
        if (full_gram_peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        const gram_ranges = try allocator.alloc(gram_geometry.Range, ranges.len);
        defer allocator.free(gram_ranges);
        for (scores, gram_ranges) |score, *out| out.* = .{ .start = score.start, .rows = score.rows, .target_rows = score.target_rows };
        var empty_mass: [0]f64 = .{};
        var empty_numerator: [0]f64 = .{};
        const memo_mass: []f64 = if (memoize_original_delta) try allocator.alloc(f64, @min(cached.query_count, gram_tile_rows)) else empty_mass[0..];
        defer if (memoize_original_delta) allocator.free(memo_mass);
        const memo_numerator: []f64 = if (memoize_original_delta) try allocator.alloc(f64, try std.math.mul(usize, @min(cached.query_count, gram_tile_rows), original.dim)) else empty_numerator[0..];
        defer if (memoize_original_delta) allocator.free(memo_numerator);
        var empty_ratios: [0]f64 = .{};
        const ratios: []f64 = if (delta_backend != null) try allocator.alloc(f64, max_selected_rows) else empty_ratios[0..];
        defer if (delta_backend != null) allocator.free(ratios);
        var context: GramContext = .{ .plan = &scored_view, .original = original, .cached = cached };
        var memo_context: GramMemoContext = .{ .plan = &scored_view, .original = original, .cached = cached, .negative_mass = memo_mass, .negative_numerator = memo_numerator };
        const accelerated_external = try std.math.add(usize, external_for_build, try std.math.add(usize, gram_owned_peak, ratio_bytes));
        var accelerated_context: AcceleratedGramContext = .{ .allocator = allocator, .plan = &scored_view, .original = original, .train = train, .cfg = cfg, .cached = cached, .backend = delta_backend orelse undefined, .ratios = ratios, .external_live_bytes = accelerated_external, .combined_cap = cfg.max_workspace_bytes, .budget_percent = geometry_budget_percent, .diagnostic = phase_timing != null };
        const accessor: gram_geometry.DeltaAccessor = if (delta_backend != null)
            .{ .context = &accelerated_context, .fill_fn = AcceleratedGramContext.fill }
        else if (memoize_original_delta)
            .{ .context = &memo_context, .fill_fn = GramMemoContext.fill }
        else
            .{ .context = &context, .fill_fn = GramContext.fill };
        var diagnostic_gram_context: DiagnosticGramBackend = .{ .inner = gram_backend.?, .budget_percent = geometry_budget_percent, .diagnostic = phase_timing != null };
        const effective_gram_backend = diagnostic_gram_context.backend();
        const gram_started = phaseStart(phase_timing);
        var representation = gram_geometry.build(allocator, gram_ranges, cached.outputs, cached.query_count, original.dim, gram_tile_rows, external_for_build, cfg.max_workspace_bytes, accessor, effective_gram_backend, cfg.control) catch |err| {
            if (phase_timing != null) std.debug.print("attention_match_gram_geometry_failure stage=representation_build budget_percent={d} error={s} diagnostic_only=true\n", .{ geometry_budget_percent, @errorName(err) });
            return err;
        };
        phaseEnd(phase_timing, .gram_construction, gram_started);
        defer representation.deinit();
        const admitted = try std.math.add(usize, external_for_build, try std.math.add(usize, accelerated_extra, try std.math.add(usize, representation.peak_bytes, try std.math.add(usize, all_geometry_payload, exchange_peak))));
        std.debug.assert(admitted == full_gram_peak);
        if (admitted > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        var made: usize = 0;
        var branches: [3]adaptive.Geometry = undefined;
        errdefer for (branches[0..made]) |*branch| branch.deinit();
        const chooser_started = phaseStart(phase_timing);
        for ([_]usize{ 4, 3, 2 }, 0..) |divisor, i| {
            try cfg.control.check();
            var greedy = (switch (geometry_chooser_policy) {
                .legacy_full_scan => gram_geometry.chooseWithBudgetPercent(allocator, &representation, divisor, .{}, if (divisor == 2) geometry_budget_percent else 100, cfg.control),
                .incremental_gram_v1 => gram_geometry.chooseIncrementalWithBudgetPercent(allocator, &representation, divisor, .{}, if (divisor == 2) geometry_budget_percent else 100, cfg.control),
            }) catch |err| switch (err) {
                error.GeometryBudgetInfeasible => {
                    branches[i] = unavailableGeometry(allocator, &representation, divisor, if (divisor == 2) geometry_budget_percent else 100) catch |unavailable_err| {
                        if (phase_timing != null) std.debug.print("attention_match_gram_geometry_failure stage=unavailable_geometry budget_percent={d} divisor={d} error={s} diagnostic_only=true\n", .{ geometry_budget_percent, divisor, @errorName(unavailable_err) });
                        return unavailable_err;
                    };
                    made += 1;
                    continue;
                },
                else => {
                    if (phase_timing != null) std.debug.print("attention_match_gram_geometry_failure stage=chooser budget_percent={d} divisor={d} chooser={s} error={s} diagnostic_only=true\n", .{ geometry_budget_percent, divisor, @tagName(geometry_chooser_policy), @errorName(err) });
                    return err;
                },
            };
            if (train_budget_exchange) {
                branches[i] = gram_geometry.refineBudgetExchange(allocator, &representation, &greedy, .{ .max_relative_mass_error = cfg.max_relative_mass_error, .max_relative_output_rmse = cfg.max_relative_output_rmse }, .{}, cfg.control) catch |err| {
                    greedy.deinit();
                    return err;
                };
                greedy.deinit();
            } else branches[i] = greedy;
            made += 1;
        }
        phaseEnd(phase_timing, .chooser, chooser_started);
        const expected_backend_calls = try std.math.mul(usize, accelerated_context.evidence.tiles, 2);
        const delta_evidence: GramDeltaEvidence = if (delta_backend != null) .{ .requested = true, .accelerated = accelerated_context.evidence.tiles > 0 and accelerated_context.evidence.backend_calls == expected_backend_calls and accelerated_context.evidence.fallbacks == 0, .tiles = accelerated_context.evidence.tiles, .tile_rows = gram_tile_rows, .backend_calls = accelerated_context.evidence.backend_calls, .packed_bytes_peak = accelerated_context.evidence.packed_bytes_peak, .staged_bytes_peak = accelerated_context.evidence.staged_bytes_peak, .page_bytes_peak = accelerated_context.evidence.page_bytes_peak, .fallbacks = accelerated_context.evidence.fallbacks } else .{};
        if (require_delta_accelerated and (!delta_evidence.accelerated or delta_evidence.tiles == 0 or delta_evidence.backend_calls != expected_backend_calls or delta_evidence.fallbacks != 0)) return error.AcceleratedDeltaUnsupported;
        const geometry_policy = if (geometry_chooser_policy == .incremental_gram_v1)
            (if (geometry_budget_percent == 100) gram_geometry.incremental_policy else gram_geometry.incremental_budget_percent_policy)
        else if (train_budget_exchange)
            gram_geometry.exchange_policy
        else
            gram_geometry.policyForBudgetPercent(geometry_budget_percent) orelse return error.InvalidFitConfig;
        return .{ .allocator = allocator, .scores = scores, .scored = scored, .branches = branches, .geometry_policy = geometry_policy, .geometry_storage = storage, .geometry_workspace_bound_bytes = admitted, .gram_memo = .{ .enabled = memoize_original_delta, .bytes = memo_bytes, .hits = memo_context.hits, .misses = memo_context.misses }, .gram_delta = delta_evidence, .joint_fixed_memo = .{ .enabled = bank.joint_memo_enabled, .bytes = joint_memo_bytes, .builds = joint_memo_builds, .hits = joint_memo_hits }, .training_reduction = reduction_evidence, .incremental_gram_chooser_calls = if (geometry_chooser_policy == .incremental_gram_v1) made else 0, .weight_cap = cap_evidence, .owns_scored_packets = false };
    }
    if (train_budget_exchange) return error.InvalidFitConfig;
    // Fail before allocating any signed-geometry payload.  This is the exact
    // simultaneous retained-packet, delta, descriptor, chooser, and eventual
    // geometry authority mirrored by replay_planner.planAdaptive.
    const score_vector_bytes = try std.math.mul(usize, try std.math.add(usize, cached.query_count, try std.math.mul(usize, cached.query_count, original.dim)), @sizeOf(f64));
    const signed_descriptors = try std.math.mul(usize, ranges.len, @sizeOf(adaptive.JointRangeScores));
    const signed_deltas = try std.math.mul(usize, try std.math.mul(usize, ranges.len, 3), score_vector_bytes);
    const chooser_vectors = try std.math.mul(usize, 2, score_vector_bytes);
    const geometry_payload = try std.math.mul(usize, 3, try std.math.mul(usize, ranges.len, @sizeOf(adaptive.Level) + @sizeOf(usize)));
    var signed_required = try scoredPlanBytes(&scored_view);
    signed_required = try std.math.add(usize, signed_required, signed_descriptors);
    signed_required = try std.math.add(usize, signed_required, signed_deltas);
    signed_required = try std.math.add(usize, signed_required, chooser_vectors);
    signed_required = try std.math.add(usize, signed_required, geometry_payload);
    if (signed_required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const joint_scores = try allocator.alloc(adaptive.JointRangeScores, ranges.len);
    defer allocator.free(joint_scores);
    var delta_ranges: usize = 0;
    defer for (joint_scores[0..delta_ranges]) |row| for (row.deltas[0..3]) |delta| {
        allocator.free(@constCast(delta.mass));
        allocator.free(@constCast(delta.numerator));
    };
    for (scores, joint_scores, 0..) |score, *joint_score, ri| {
        joint_score.* = .{ .start = score.start, .rows = score.rows, .target_rows = score.target_rows, .deltas = undefined };
        var made_levels: usize = 0;
        var row_transferred = false;
        errdefer if (!row_transferred) for (joint_score.deltas[0..made_levels]) |delta| {
            allocator.free(@constCast(delta.mass));
            allocator.free(@constCast(delta.numerator));
        };
        for (0..3) |li| {
            var level_transferred = false;
            const mass = try allocator.alloc(f64, cached.query_count);
            errdefer if (!level_transferred) allocator.free(mass);
            const numerator = try allocator.alloc(f64, try std.math.mul(usize, cached.query_count, original.dim));
            errdefer if (!level_transferred) allocator.free(numerator);
            @memset(mass, 0);
            @memset(numerator, 0);
            try accumulateScoredDelta(&scored_view, ri, li, original, cached, cfg.control, mass, numerator);
            joint_score.deltas[li] = .{ .mass = mass, .numerator = numerator };
            level_transferred = true;
            made_levels += 1;
        }
        joint_score.deltas[3] = .{ .mass = &.{}, .numerator = &.{} };
        delta_ranges += 1;
        row_transferred = true;
    }
    var made: usize = 0;
    var branches: [3]adaptive.Geometry = undefined;
    errdefer for (branches[0..made]) |*branch| branch.deinit();
    const chooser_started = phaseStart(phase_timing);
    for ([_]usize{ 4, 3, 2 }, 0..) |divisor, i| {
        branches[i] = try adaptive.chooseSignedJoint(allocator, joint_scores, cached.outputs, cached.query_count, original.dim, divisor, .{}, cfg.control);
        made += 1;
    }
    phaseEnd(phase_timing, .chooser, chooser_started);
    return .{ .allocator = allocator, .scores = scores, .scored = scored, .branches = branches, .geometry_policy = adaptive.signed_joint_policy, .geometry_storage = .dense, .geometry_workspace_bound_bytes = signed_required, .joint_fixed_memo = .{ .enabled = bank.joint_memo_enabled, .bytes = joint_memo_bytes, .builds = joint_memo_builds, .hits = joint_memo_hits }, .training_reduction = reduction_evidence, .weight_cap = cap_evidence, .owns_scored_packets = false };
}

fn belongs(batch: engine.ReferenceBatch, source: checkpoint.State, layout: engine.Layout, donor: usize, head: usize) bool {
    if (source.layer_donors[batch.consumer_layer] != donor) return false;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / source.donors[donor].heads.len;
    return batch.query_head / group == head;
}

/// Validate every option whose legality affects the adaptive planner's peak or
/// numerical policy. Both the whole-input diagnostic wrapper and the public
/// normalized-head runtime entry must cross this boundary before allocation or
/// backend dispatch.
fn validateV18RuntimeOptions(options: Options, min_weight: f64) !void {
    if (options.observe_train_probability_census and (!options.training_source or options.critical_core_policy == .disabled)) return error.InvalidFitConfig;
    if (options.postfit_policy != .disabled and
        (!options.training_source or options.critical_core_policy == .disabled or
            options.mass_strategy != .identity_chunk_coefficients_v18 or options.final_refit_strategy != .identity_reuse or
            options.probability_storage != .exact_f32_widened or options.postfit_workspace_bytes == 0 or min_weight != 1e-12))
        return error.InvalidFitConfig;
    if (options.postfit_policy == .disabled and options.postfit_workspace_bytes != 0) return error.InvalidFitConfig;
    if (options.postfit_policy == .disabled and options.candidate_postfit_hook != null) return error.InvalidFitConfig;
    if (options.critical_core_policy != .disabled and
        (!options.training_source or options.mass_strategy != .identity_chunk_coefficients_v18 or
            options.probability_storage != .exact_f32_widened or
            activeSelectionPolicy(options) != .raw or options.max_local_weight != null))
        return error.InvalidFitConfig;
    if (options.training_reduction_query_batch_rows == 0) return error.InvalidFitConfig;
    if (options.training_reduction_policy == .accelerate_fp64_cached_v1 and
        (!options.training_source or options.probability_storage != .exact_f32_widened or options.geometry_storage != .per_query_gram))
        return error.InvalidFitConfig;
    const diagnostic_pair = options.flat_batch_policy == .diagnostic_capture_gram_256_fallback_64_v1 and
        options.training_source_query_batch_rows == 256 and options.gram_tile_rows == 256 and
        options.training_capture_backend != null;
    if (!diagnostic_pair and (options.flat_batch_policy != .established or
        (options.training_source_query_batch_rows != 16 and options.training_source_query_batch_rows != 64)))
        return error.InvalidFitConfig;
    if (options.gram_delta_backend != null or options.gram_delta_require_accelerated) {
        if (options.geometry_storage != .per_query_gram or options.gram_memoize_original_delta or
            (options.gram_tile_rows != 8 and options.gram_tile_rows != 64 and !diagnostic_pair)) return error.InvalidFitConfig;
        if (options.gram_delta_require_accelerated and options.gram_delta_backend == null) return error.AcceleratedDeltaUnsupported;
    }
    if (!gram_geometry.validBudgetPercent(options.geometry_budget_percent)) return error.InvalidFitConfig;
    const fallback_configured = options.fallback_geometry_budget_percent != 0 or options.fallback_authorize_fn != null or options.fallback_authorize_context != null;
    if (fallback_configured and (options.geometry_budget_percent < 50 or options.geometry_budget_percent >= 95 or options.fallback_geometry_budget_percent != 95 or options.fallback_authorize_fn == null or options.fallback_authorize_context == null or options.train_snapshot_fn != null or options.train_snapshot_v2_fn != null))
        return error.InvalidFitConfig;
    if (options.geometry_chooser_policy == .incremental_gram_v1 and
        (options.geometry_storage != .per_query_gram or options.train_budget_exchange))
        return error.InvalidFitConfig;
    const snapshot_pair = (options.flat_batch_policy == .established and options.training_source_query_batch_rows == 64 and options.gram_tile_rows == 64 and options.training_reduction_query_batch_rows == 64 and options.training_capture_backend != null) or (diagnostic_pair and options.training_reduction_query_batch_rows == 256);
    const typed_future_snapshot = options.train_snapshot_v2_fn != null and (options.geometry_budget_percent == 91 or options.geometry_budget_percent == 95) and snapshot_pair and options.geometry_chooser_policy == .incremental_gram_v1 and options.training_reduction_policy == .accelerate_fp64_cached_v1 and options.train_snapshot_v2_domain_runs.len != 0;
    if (options.geometry_budget_percent != 100 and (!options.training_source or options.mass_strategy != .identity_chunk_coefficients_v18 or options.geometry_storage != .per_query_gram or options.train_budget_exchange or options.final_refit_strategy != .identity_reuse or options.max_local_weight != null or options.train_snapshot_fn != null or (options.train_snapshot_v2_fn != null and !typed_future_snapshot))) return error.InvalidFitConfig;
    try local_weight_cap.validate(options.max_local_weight);
    if (options.max_local_weight) |cap| {
        if (!options.training_source or options.mass_strategy != .identity_chunk_coefficients_v18 or cap < min_weight) return error.InvalidFitConfig;
    }
}

test "critical production per-query Gram budget tuple is accepted without weakening numerical guards" {
    const Authorize = struct {
        fn call(_: ?*anyopaque) bool {
            return true;
        }
    };
    var authority: u8 = 0;
    const production: Options = .{
        .critical_core_policy = .train_probability_gte_0_025_no_top1_v1,
        .training_source = true,
        .probability_storage = .exact_f32_widened,
        .geometry_storage = .per_query_gram,
        .geometry_budget_percent = 91,
        .fallback_geometry_budget_percent = 95,
        .fallback_authorize_context = &authority,
        .fallback_authorize_fn = Authorize.call,
        .mass_strategy = .identity_chunk_coefficients_v18,
        .selection_score = .raw,
        .training_source_query_batch_rows = 64,
        .gram_tile_rows = 64,
    };
    try validateV18RuntimeOptions(production, 1e-12);
    var wrong = production;
    wrong.probability_storage = .f64;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(wrong, 1e-12));
    wrong = production;
    wrong.mass_strategy = .global_active_nnls_v16;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(wrong, 1e-12));
    wrong = production;
    wrong.selection_score = .normalized;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(wrong, 1e-12));
    wrong = production;
    wrong.geometry_storage = .dense;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(wrong, 1e-12));
}

fn reductionWorkspaceBytes(query_count: usize, max_range_rows: usize, max_selected_rows: usize, dim: usize, batch_rows: usize) !usize {
    const score_vectors = try std.math.mul(usize, try std.math.add(usize, query_count, try std.math.mul(usize, query_count, dim)), @sizeOf(f64));
    const harm = try std.math.add(usize, try train_reduction.harmWorkspaceBytes(query_count, max_selected_rows, dim, batch_rows), try std.math.mul(usize, max_selected_rows, @sizeOf(f64)));
    return @max(
        try train_reduction.workspaceBytes(query_count, max_range_rows, dim, batch_rows),
        try std.math.add(usize, score_vectors, harm),
    );
}

fn criticalCoreConservativeReserve(original: solver.Head, ranges: []const admission.Range, query_count: usize, cfg: solver.Config) !usize {
    var retained_rows: usize = 0;
    var max_rows: usize = 0;
    for (ranges) |range| {
        retained_rows = try std.math.add(usize, retained_rows, try std.math.mul(usize, 3, range.rows));
        max_rows = @max(max_rows, range.rows);
    }
    const per_row = try std.math.add(usize, @sizeOf(usize) + 2 * @sizeOf(f64), try std.math.mul(usize, original.dim, @sizeOf(f64)));
    const retained = try std.math.mul(usize, retained_rows, per_row);
    const fit = (try joint.workspaceBytesShape(original, query_count, 1, max_rows, max_rows, cfg, true, .identity_chunk_coefficients_v18, false)).required;
    return std.math.add(usize, retained, fit);
}

test "TRAIN reduction planner fallback preserves irregular tail and near-cap admission" {
    const q: usize = 12_289;
    const large = try reductionWorkspaceBytes(q, 256, 128, 512, 768);
    const fallback = try reductionWorkspaceBytes(q, 256, 128, 512, 64);
    try std.testing.expect(large > fallback);
    try std.testing.expectEqual(@as(usize, 17), std.math.divCeil(usize, q, 768) catch unreachable);
    try std.testing.expectEqual(@as(usize, 193), std.math.divCeil(usize, q, 64) catch unreachable);
    const aggregate_without_reduction: usize = 994_003_776;
    const cap = aggregate_without_reduction + fallback;
    try std.testing.expect(aggregate_without_reduction + large > cap);
    try std.testing.expect(aggregate_without_reduction + fallback <= cap);
}

test "diagnostic256 full runtime option tuple is admitted and mixed tuples reject" {
    const Dummy = struct {
        fn configure(_: ?*anyopaque, _: usize) !void {}
        fn workspace(_: ?*anyopaque, _: solver.Head, _: usize) !usize {
            return 0;
        }
        fn begin(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: usize, _: solver.Control) !?*anyopaque {
            return null;
        }
        fn tile(_: ?*anyopaque, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {}
        fn end(_: ?*anyopaque, _: std.mem.Allocator) void {}
        fn backend() training_source.CaptureBackend {
            return .{ .context = null, .configure_fn = configure, .workspace_fn = workspace, .begin_fn = begin, .tile_fn = tile, .end_fn = end };
        }
    };
    var options: Options = .{ .training_source = true, .mass_strategy = .identity_chunk_coefficients_v18, .probability_storage = .exact_f32_widened, .geometry_storage = .per_query_gram, .geometry_budget_percent = 95, .geometry_chooser_policy = .incremental_gram_v1, .training_source_query_batch_rows = 256, .gram_tile_rows = 256, .flat_batch_policy = .diagnostic_capture_gram_256_fallback_64_v1, .training_capture_backend = Dummy.backend(), .training_reduction_policy = .accelerate_fp64_cached_v1, .training_reduction_query_batch_rows = 256 };
    try validateV18RuntimeOptions(options, 1e-12);
    try std.testing.expect(validAcceleratedGramTileRows(256, 256));
    options.gram_tile_rows = 64;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(options, 1e-12));
    options.gram_tile_rows = 256;
    options.training_source_query_batch_rows = 64;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(options, 1e-12));
    try std.testing.expect(!validAcceleratedGramTileRows(64, 256));
}

test "runtime v18 helper rejects option tuples formerly checked only by input wrapper" {
    const accepted: Options = .{
        .training_source = true,
        .mass_strategy = .identity_chunk_coefficients_v18,
        .probability_storage = .exact_f32_widened,
        .geometry_storage = .per_query_gram,
        .geometry_budget_percent = 95,
        .training_source_query_batch_rows = 64,
        .gram_tile_rows = 64,
        .joint_fixed_original_memo = true,
    };
    try validateV18RuntimeOptions(accepted, 1e-12);
    var invalid = accepted;
    invalid.training_source_query_batch_rows = 32;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(invalid, 1e-12));
    invalid = accepted;
    invalid.geometry_storage = .dense;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(invalid, 1e-12));
    invalid = accepted;
    invalid.max_local_weight = 1e-13;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(invalid, 1e-12));
    invalid = accepted;
    invalid.training_source_query_batch_rows = 256;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(invalid, 1e-12));
    invalid.flat_batch_policy = .diagnostic_capture_gram_256_fallback_64_v1;
    invalid.gram_tile_rows = 256;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(invalid, 1e-12));
}

test "fallback reuse accepts the engine primary budget range and rejects partial tuples before backend work" {
    const Authorization = struct {
        fn allow(_: ?*anyopaque) bool {
            return true;
        }
    };
    var context: u8 = 0;
    const base: Options = .{
        .training_source = true,
        .mass_strategy = .identity_chunk_coefficients_v18,
        .probability_storage = .exact_f32_widened,
        .geometry_storage = .per_query_gram,
        .geometry_budget_percent = 90,
        .fallback_geometry_budget_percent = 95,
        .fallback_authorize_context = &context,
        .fallback_authorize_fn = Authorization.allow,
        .training_source_query_batch_rows = 64,
        .gram_tile_rows = 64,
        .joint_fixed_original_memo = true,
    };
    for ([_]u8{ 90, 91, 94 }) |primary| {
        var accepted = base;
        accepted.geometry_budget_percent = primary;
        try validateV18RuntimeOptions(accepted, 1e-12);
    }
    for ([_]u8{ 49, 95, 100 }) |primary| {
        var rejected = base;
        rejected.geometry_budget_percent = primary;
        try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(rejected, 1e-12));
    }
    var partial = base;
    partial.fallback_geometry_budget_percent = 0;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(partial, 1e-12));
    partial = base;
    partial.fallback_authorize_context = null;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(partial, 1e-12));
    partial = base;
    partial.fallback_authorize_fn = null;
    try std.testing.expectError(error.InvalidFitConfig, validateV18RuntimeOptions(partial, 1e-12));
}

pub fn runInputWithOptions(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, branch_chunk_rows: usize, max_workspace_bytes: usize, options: Options) !Result {
    try validateV18RuntimeOptions(options, input.fit.min_weight);
    if (comptime !@import("builtin").is_test) @compileError("joint chunk mass input replay is diagnostic-test-only");
    const state = input.checkpoint_owned.state;
    if (donor >= state.donors.len or head_index >= state.donors[donor].heads.len or state.donors[donor].window != 0) return error.InvalidWholeFitInput;
    const source_head = state.donors[donor].heads[head_index];
    if (source_head.compressed.rows() != 0 or source_head.positions.len != source_head.keys.len / source_head.dim) return error.UnsupportedKvCheckpoint;
    var train_count: usize = 0;
    var heldout_count: usize = 0;
    for (input.batches) |batch| if (belongs(batch, state, input.layout, donor, head_index)) {
        train_count = try std.math.add(usize, train_count, batch.train.len);
        heldout_count = try std.math.add(usize, heldout_count, batch.heldout.len);
    };
    for (augmented) |batch| if (belongs(batch, state, input.layout, donor, head_index)) {
        if (batch.heldout.len != 0) return error.InvalidWholeFitInput;
        train_count = try std.math.add(usize, train_count, batch.train.len);
    };
    if (train_count == 0 or heldout_count == 0) return error.MissingCompactionConsumerQueries;
    const train = try allocator.alloc(f64, train_count);
    defer allocator.free(train);
    const heldout = try allocator.alloc(f64, heldout_count);
    defer allocator.free(heldout);
    var train_at: usize = 0;
    var heldout_at: usize = 0;
    for (input.batches) |batch| if (belongs(batch, state, input.layout, donor, head_index)) {
        for (batch.train) |value| {
            train[train_at] = value;
            train_at += 1;
        }
        for (batch.heldout) |value| {
            heldout[heldout_at] = value;
            heldout_at += 1;
        }
    };
    for (augmented) |batch| if (belongs(batch, state, input.layout, donor, head_index)) for (batch.train) |value| {
        train[train_at] = value;
        train_at += 1;
    };
    if (train_at != train_count or heldout_at != heldout_count) return error.InvalidWholeFitInput;
    const keys = try allocator.alloc(f64, source_head.keys.len);
    defer allocator.free(keys);
    const values = try allocator.alloc(f64, source_head.values.len);
    defer allocator.free(values);
    const bias = try allocator.alloc(f64, source_head.bias.len);
    defer allocator.free(bias);
    for (keys, source_head.keys) |*destination, value| destination.* = value;
    for (values, source_head.values) |*destination, value| destination.* = value;
    for (bias, source_head.bias) |*destination, value| destination.* = value;
    const source: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = source_head.positions.len, .dim = source_head.dim, .scale = input.layout.scales_by_donor[donor], .softcap = input.layout.softcaps_by_donor[donor] };
    const fit_cfg = input.fit;
    if (fit_cfg.prefix_rows >= source.rows or fit_cfg.recent_rows >= source.rows - fit_cfg.prefix_rows or branch_chunk_rows != 256) return error.InvalidWholeFitInput;
    const old_end = source.rows - fit_cfg.recent_rows;
    const middle = old_end - fit_cfg.prefix_rows;
    const chunk_count = std.math.divCeil(usize, middle, branch_chunk_rows) catch return error.InvalidWholeFitInput;
    const ranges = try allocator.alloc(admission.Range, chunk_count);
    defer allocator.free(ranges);
    var start = fit_cfg.prefix_rows;
    for (ranges) |*range| {
        const rows = @min(branch_chunk_rows, old_end - start);
        range.* = .{ .start = start, .rows = rows };
        start += rows;
    }
    if (start != old_end) return error.InvalidWholeFitInput;
    const external_elements = try std.math.add(usize, try std.math.add(usize, try std.math.add(usize, keys.len, values.len), bias.len), try std.math.add(usize, train.len, heldout.len));
    var external_bytes = try std.math.mul(usize, external_elements, @sizeOf(f64));
    external_bytes = try std.math.add(usize, external_bytes, try std.math.mul(usize, ranges.len, @sizeOf(admission.Range)));
    if (external_bytes > max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = fit_cfg.max_relative_mass_error, .max_relative_output_rmse = fit_cfg.max_relative_output_rmse, .max_sweeps = fit_cfg.max_sweeps, .nnls_tolerance = fit_cfg.nnls_tolerance, .min_weight = fit_cfg.min_weight, .max_workspace_bytes = max_workspace_bytes - external_bytes, .backend = backend };
    if (options.mass_strategy == .identity_chunk_coefficients_v18) {
        if (!options.training_source or branch_chunk_rows != 256) return error.InvalidFitConfig;
        var head_cfg = cfg;
        head_cfg.max_workspace_bytes = max_workspace_bytes;
        return runHeadV18WithOptions(allocator, source, train, heldout, ranges, head_cfg, external_bytes, max_workspace_bytes, options);
    }
    if (!options.training_source) return runWithOptions(allocator, source, train, heldout, ranges, cfg, options, null, null);
    var planning_cfg = cfg;
    planning_cfg.max_workspace_bytes = max_workspace_bytes;
    const plan = replay_planner.plan(allocator, source, train.len / source.dim, heldout.len / source.dim, ranges, planning_cfg, try std.math.mul(usize, 3, @sizeOf(Branch)), external_bytes, options.mass_strategy, options.retain_training_diagnostics) catch |err| switch (err) {
        error.CompactionWorkspaceLimit, error.OutOfMemory => {
            var fallback = try runWithOptions(allocator, source, train, heldout, ranges, cfg, .{ .mass_strategy = options.mass_strategy, .retain_training_diagnostics = options.retain_training_diagnostics }, null, null);
            fallback.training_source = .{ .requested = true, .outcome = if (err == error.OutOfMemory) .fallback_allocation else .fallback_cap, .external_live_bytes = external_bytes };
            return fallback;
        },
        else => return err,
    };
    var counters: training_source.Counters = .{};
    const prepared = try training_source.prepare(allocator, source, train, cfg, .{ .combined_cap = max_workspace_bytes, .external_live_bytes = external_bytes, .consumer_peak_bytes = plan.consumer_peak_bytes, .probability_storage = options.probability_storage, .query_batch_rows = options.training_source_query_batch_rows, .capture_backend = options.training_capture_backend, .allocation_authority = options.training_cache_allocation_authority }, &counters);
    switch (prepared) {
        .fallback => |reason| {
            var fallback = try runWithOptions(allocator, source, train, heldout, ranges, cfg, .{ .mass_strategy = options.mass_strategy, .retain_training_diagnostics = options.retain_training_diagnostics }, null, null);
            fallback.training_source = .{ .requested = true, .outcome = if (reason == .allocation) .fallback_allocation else .fallback_cap, .counters = counters, .external_live_bytes = external_bytes };
            return fallback;
        },
        .ready => |owned| {
            var cached = owned;
            defer cached.deinit();
            cfg.max_workspace_bytes = plan.consumer_peak_bytes;
            var result = try runWithOptions(allocator, source, train, heldout, ranges, cfg, options, &cached, &counters);
            result.training_source = .{ .requested = true, .outcome = .ready, .digest = cached.digest, .counters = counters, .external_live_bytes = external_bytes };
            return result;
        },
    }
}

/// Shared runtime and diagnostic entry for a normalized immutable head.
/// Every live source/query/range byte is reserved by the caller before capture.
pub fn runHeadV18WithOptions(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    heldout: []const f64,
    ranges: []const admission.Range,
    fit_cfg: solver.Config,
    external_live_bytes: usize,
    combined_cap: usize,
    options: Options,
) !Result {
    try validateV18RuntimeOptions(options, fit_cfg.min_weight);
    try original.validate();
    if (ranges.len == 0 or train.len == 0 or combined_cap == 0 or external_live_bytes > combined_cap or fit_cfg.max_workspace_bytes != combined_cap or train.len % original.dim != 0 or !options.training_source or options.mass_strategy != .identity_chunk_coefficients_v18)
        return error.InvalidFitConfig;
    const heldout_query_rows = try primaryProbeHeldoutRows(options, heldout, original.dim);
    const source = original;
    var cfg = fit_cfg;
    const external_bytes = external_live_bytes;
    const max_workspace_bytes = combined_cap;
    var critical_phase_diagnostic: CriticalPhaseDiagnostic = .{
        .enabled = criticalPhaseDiagnosticEnabled(options),
        .query_rows = train.len / original.dim,
        .source_rows = original.rows,
        .dim = original.dim,
        .range_count = ranges.len,
        .external_bytes = external_live_bytes,
        .combined_cap = combined_cap,
    };
    var planning_cfg = cfg;
    planning_cfg.max_workspace_bytes = max_workspace_bytes;
    const Scratch = struct {
        fn bytes(opts: Options, source_head: solver.Head, fit_ranges: []const admission.Range, train_queries: usize, tile_rows: usize) !usize {
            if (opts.geometry_storage != .per_query_gram) return 0;
            const gram_backend = opts.gram_backend orelse return error.V18TrainingScoreUnsupported;
            const candidate_count = try std.math.mul(usize, fit_ranges.len, 3);
            var total_bytes = try gram_backend.workspace_fn(gram_backend.context, candidate_count, source_head.dim);
            if (opts.gram_memoize_original_delta) total_bytes = try std.math.add(usize, total_bytes, try gramMemoBytes(@min(tile_rows, train_queries), source_head.dim));
            if (opts.train_budget_exchange) total_bytes = try std.math.add(usize, total_bytes, try std.math.mul(usize, fit_ranges.len, @sizeOf(u8) + 2 * (@sizeOf(adaptive.Level) + @sizeOf(usize))));
            if (opts.gram_delta_backend != null or opts.gram_delta_require_accelerated) {
                var max_range_rows: usize = 0;
                for (fit_ranges) |range| max_range_rows = @max(max_range_rows, range.rows);
                const max_selected_rows = std.math.divCeil(usize, max_range_rows, 2) catch return error.InvalidAttentionShape;
                total_bytes = try std.math.add(usize, total_bytes, try std.math.mul(usize, max_selected_rows, @sizeOf(f64)));
                total_bytes = try std.math.add(usize, total_bytes, try gram_delta_page.workspaceBytes(@min(tile_rows, train_queries), max_range_rows, max_selected_rows, source_head.dim));
            }
            if (opts.geometry_chooser_policy == .incremental_gram_v1)
                total_bytes = try std.math.add(usize, total_bytes, try std.math.mul(usize, try std.math.mul(usize, train_queries, 2), @sizeOf(f64)));
            return total_bytes;
        }
    };
    var active_capture_rows = options.training_source_query_batch_rows;
    var active_gram_rows = options.gram_tile_rows;
    var gram_scratch = try Scratch.bytes(options, source, ranges, train.len / source.dim, active_gram_rows);
    if (options.flat_batch_policy == .diagnostic_capture_gram_256_fallback_64_v1) {
        const backend = options.training_capture_backend orelse return error.InvalidFitConfig;
        const configure = backend.configure_fn orelse return error.InvalidFitConfig;
        // Configuration is itself preflight: it occurs before workspace
        // queries, allocation, capture.begin, or provider dispatch.
        try configure(backend.context, active_capture_rows);
        const capture_transient = try backend.workspace_fn(backend.context, source, active_capture_rows);
        if (capture_transient > 256 * 1024 * 1024 or gram_scratch > 256 * 1024 * 1024) {
            active_capture_rows = 64;
            active_gram_rows = 64;
            gram_scratch = try Scratch.bytes(options, source, ranges, train.len / source.dim, active_gram_rows);
            try configure(backend.context, active_capture_rows);
            if ((try backend.workspace_fn(backend.context, source, active_capture_rows)) > 256 * 1024 * 1024 or gram_scratch > 256 * 1024 * 1024)
                return error.V18TrainingCacheUnsupported;
        }
    }
    const final_mass_strategy: joint.MassStrategy = switch (options.final_refit_strategy) {
        .identity_reuse => .identity_chunk_coefficients_v18,
        .global_active_nnls_v16 => .global_active_nnls_v16,
    };
    const joint_fixed_bytes = if (options.joint_fixed_original_memo or options.critical_core_policy != .disabled) try joint.fixedOriginalTermBytes(train.len / source.dim, source.dim) else 0;
    var max_range_rows: usize = 0;
    for (ranges) |range| max_range_rows = @max(max_range_rows, range.rows);
    const max_selected_rows = std.math.divCeil(usize, max_range_rows, 2) catch return error.InvalidAttentionShape;
    var reduction_batch_rows = options.training_reduction_query_batch_rows;
    var reduction_workspace = if (options.training_reduction_policy == .accelerate_fp64_cached_v1)
        try reductionWorkspaceBytes(train.len / source.dim, max_range_rows, max_selected_rows, source.dim, reduction_batch_rows)
    else
        0;
    const Planner = struct {
        fn run(a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize) !replay_planner.AdaptivePlan {
            return replay_planner.planAdaptiveForStorageRefitJointMemoValidationReduction(a, src, train_queries, heldout_queries, fit_ranges, pcfg, branch_bytes, ext, observe, storage, tile_rows, gram_bytes, mass_strategy, fixed_bytes, validation_backend, reduction_bytes);
        }
        fn runCritical(a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize, shape: replay_planner.CriticalTargetShape) !replay_planner.AdaptivePlan {
            return replay_planner.planAdaptiveForStorageRefitJointMemoValidationReductionCritical(a, src, train_queries, heldout_queries, fit_ranges, pcfg, branch_bytes, ext, observe, storage, tile_rows, gram_bytes, mass_strategy, fixed_bytes, validation_backend, reduction_bytes, shape);
        }
        fn runProbe(a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize) !replay_planner.AdaptivePlan {
            return replay_planner.planAdaptiveForStorageRefitJointMemoValidationReductionProbe(a, src, train_queries, heldout_queries, fit_ranges, pcfg, branch_bytes, ext, observe, storage, tile_rows, gram_bytes, mass_strategy, fixed_bytes, validation_backend, reduction_bytes);
        }
        fn runCriticalProbe(a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize, shape: replay_planner.CriticalTargetShape) !replay_planner.AdaptivePlan {
            return replay_planner.planAdaptiveForStorageRefitJointMemoValidationReductionCriticalProbe(a, src, train_queries, heldout_queries, fit_ranges, pcfg, branch_bytes, ext, observe, storage, tile_rows, gram_bytes, mass_strategy, fixed_bytes, validation_backend, reduction_bytes, shape);
        }
    };
    const branch_bytes = try std.math.mul(usize, 3, @sizeOf(Branch));
    // A TRAIN-only primary probe applies postfit only after the cache/scoring
    // phase. buildPrimaryTrainCandidate performs its own exact admission with
    // the postfit scratch live, so overlapping that scratch with cache/scoring
    // here is a false phase sum. Normal/HHO execution remains unchanged.
    const planner_external = try primaryProbePlannerExternalBytes(external_bytes, options.postfit_workspace_bytes, options.primary_train_probe_fn != null);
    if (planner_external > max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const probe_only = options.primary_train_probe_fn != null;
    const run_planner: *const fn (a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize) anyerror!replay_planner.AdaptivePlan = if (probe_only) Planner.runProbe else Planner.run;
    const run_critical_planner: *const fn (a: std.mem.Allocator, src: solver.Head, train_queries: usize, heldout_queries: usize, fit_ranges: []const admission.Range, pcfg: solver.Config, branch_bytes: usize, ext: usize, observe: bool, storage: GeometryStorage, tile_rows: usize, gram_bytes: usize, mass_strategy: joint.MassStrategy, fixed_bytes: usize, validation_backend: ?solver.Backend, reduction_bytes: usize, shape: replay_planner.CriticalTargetShape) anyerror!replay_planner.AdaptivePlan = if (probe_only) Planner.runCriticalProbe else Planner.runCritical;
    if (probe_only and critical_phase_diagnostic.enabled) {
        var diagnostic_cfg = planning_cfg;
        diagnostic_cfg.max_workspace_bytes = std.math.maxInt(usize);
        const estimated = try Planner.runProbe(allocator, source, train.len / source.dim, heldout_query_rows, ranges, diagnostic_cfg, branch_bytes, 0, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace);
        const q = train.len / source.dim;
        const probability_bytes = try std.math.mul(usize, try std.math.mul(usize, q, source.rows), @sizeOf(f32));
        const output_bytes = try std.math.mul(usize, try std.math.mul(usize, q, source.dim), @sizeOf(f64));
        const expected_cache_bytes = try std.math.add(usize, probability_bytes, output_bytes);
        const admitted_total = try std.math.add(usize, planner_external, try std.math.add(usize, expected_cache_bytes, estimated.consumer_peak_bytes));
        std.debug.print("attention_match_primary_probe_admission_estimate replay_external_bytes={d} expected_cache_bytes={d} scoring_peak_bytes={d} signed_geometry_peak_bytes={d} probe_consumer_peak_bytes={d} admitted_total_bytes={d} combined_cap={d} diagnostic_only=true\n", .{ planner_external, expected_cache_bytes, estimated.scoring_peak_bytes, estimated.signed_geometry_peak_bytes, estimated.consumer_peak_bytes, admitted_total, max_workspace_bytes });
    }
    var plan = run_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, planner_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace) catch |err| switch (err) {
        error.CompactionWorkspaceLimit, error.OutOfMemory => retry: {
            if (options.training_reduction_policy != .accelerate_fp64_cached_v1 or reduction_batch_rows <= 64) {
                if (options.flat_batch_policy != .diagnostic_capture_gram_256_fallback_64_v1) return error.V18TrainingCacheUnsupported;
                active_capture_rows = 64;
                active_gram_rows = 64;
                gram_scratch = try Scratch.bytes(options, source, ranges, train.len / source.dim, active_gram_rows);
                break :retry run_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, planner_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace) catch |fallback_err| switch (fallback_err) {
                    error.CompactionWorkspaceLimit, error.OutOfMemory => return error.V18TrainingCacheUnsupported,
                    else => return fallback_err,
                };
            }
            reduction_batch_rows = 64;
            reduction_workspace = try reductionWorkspaceBytes(train.len / source.dim, max_range_rows, max_selected_rows, source.dim, reduction_batch_rows);
            break :retry run_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, planner_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace) catch |retry_err| switch (retry_err) {
                error.CompactionWorkspaceLimit, error.OutOfMemory => pair_fallback: {
                    if (options.flat_batch_policy != .diagnostic_capture_gram_256_fallback_64_v1) return error.V18TrainingCacheUnsupported;
                    active_capture_rows = 64;
                    active_gram_rows = 64;
                    gram_scratch = try Scratch.bytes(options, source, ranges, train.len / source.dim, active_gram_rows);
                    break :pair_fallback run_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, planner_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace) catch |fallback_err| switch (fallback_err) {
                        error.CompactionWorkspaceLimit, error.OutOfMemory => return error.V18TrainingCacheUnsupported,
                        else => return fallback_err,
                    };
                },
                else => return retry_err,
            };
        },
        else => return err,
    };
    critical_phase_diagnostic.setPlan(plan);
    var counters: training_source.Counters = .{};
    const cache_started = phaseStart(options.phase_timing);
    if (options.training_capture_backend) |backend| if (backend.configure_fn) |configure| try configure(backend.context, active_capture_rows);
    critical_phase_diagnostic.phase = .cache_prepare;
    var prepared = training_source.prepare(allocator, source, train, cfg, .{ .combined_cap = max_workspace_bytes, .external_live_bytes = planner_external, .consumer_peak_bytes = plan.consumer_peak_bytes, .probability_storage = options.probability_storage, .query_batch_rows = active_capture_rows, .capture_backend = options.training_capture_backend, .allocation_authority = options.training_cache_allocation_authority }, &counters) catch |err| return critical_phase_diagnostic.fail(err);
    const initial_fallback = switch (prepared) {
        .fallback => true,
        .ready => false,
    };
    if (initial_fallback and options.flat_batch_policy == .diagnostic_capture_gram_256_fallback_64_v1) {
        // `prepare` returns fallback only before capture.begin. Re-plan the
        // capture/Gram pair together; never continue with mixed tile sizes.
        active_capture_rows = 64;
        active_gram_rows = 64;
        gram_scratch = try Scratch.bytes(options, source, ranges, train.len / source.dim, active_gram_rows);
        plan = try run_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, planner_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, reduction_workspace);
        if (options.training_capture_backend) |backend| if (backend.configure_fn) |configure| try configure(backend.context, active_capture_rows);
        critical_phase_diagnostic.setPlan(plan);
        prepared = training_source.prepare(allocator, source, train, cfg, .{ .combined_cap = max_workspace_bytes, .external_live_bytes = planner_external, .consumer_peak_bytes = plan.consumer_peak_bytes, .probability_storage = options.probability_storage, .query_batch_rows = active_capture_rows, .capture_backend = options.training_capture_backend, .allocation_authority = options.training_cache_allocation_authority }, &counters) catch |err| return critical_phase_diagnostic.fail(err);
    }
    phaseEnd(options.phase_timing, .cache_prepare, cache_started);
    switch (prepared) {
        .fallback => |reason| return if (reason == .authority_changed) error.V18TrainingCacheAuthorityChanged else error.V18TrainingCacheUnsupported,
        .ready => |owned| {
            var cached = owned;
            defer cached.deinit();
            critical_phase_diagnostic.cache_bytes = counters.cache_bytes;
            critical_phase_diagnostic.phase = .critical_extract;
            var calibration: CriticalCoreCalibrationEvidence = .{};
            var critical_authority: ?critical_keys.Result = switch (options.critical_core_policy) {
                .disabled => null,
                .train_probability_gte_0_025_no_top1_v1 => extractCriticalAuthority(allocator, &cached, ranges, try std.math.add(usize, external_bytes, counters.cache_bytes), max_workspace_bytes, cfg.control) catch |err| return critical_phase_diagnostic.fail(err),
                .train_probability_budget_calibrated_floor_0_0025_p91_v1 => calibrated: {
                    if (options.geometry_budget_percent != 91 or options.fallback_geometry_budget_percent != (if (options.primary_train_probe_fn != null) @as(u8, 0) else 95)) return error.InvalidFitConfig;
                    const maxima_external = try std.math.add(usize, external_bytes, counters.cache_bytes);
                    const evidence: CriticalCoreCalibrationEvidence = evidence: {
                        const maxima = trainMaximumProbabilities(allocator, &cached, maxima_external, max_workspace_bytes, cfg.control) catch |err| return critical_phase_diagnostic.fail(err);
                        defer allocator.free(maxima);
                        const selector_peak = try std.math.add(usize, try std.math.add(usize, maxima_external, try std.math.mul(usize, maxima.len, @sizeOf(f64))), 64 * 1024);
                        if (selector_peak > max_workspace_bytes) return error.CompactionWorkspaceLimit;
                        var selected = calibrated_critical.select(allocator, maxima, ranges, .{ .primary_percent = options.geometry_budget_percent, .control = cfg.control }) catch |err| return critical_phase_diagnostic.fail(err);
                        defer selected.deinit();
                        if (!std.mem.eql(u8, &selected.policy_sha256, &calibrated_critical.policyDigest())) return error.InvalidFitConfig;
                        break :evidence .{ .observed = true, .threshold = selected.threshold, .critical_rows = selected.critical_rows, .minimum_middle_rows = selected.minimum_middle_rows, .middle_cap_rows = selected.middle_cap_rows, .policy_sha256 = selected.policy_sha256 };
                    };
                    var authority = extractCriticalAuthorityThreshold(allocator, &cached, ranges, evidence.threshold, maxima_external, max_workspace_bytes, cfg.control) catch |err| return critical_phase_diagnostic.fail(err);
                    errdefer authority.deinit();
                    var verified_rows: usize = 0;
                    var verified_minimum: usize = 0;
                    for (ranges) |range| {
                        const actual = try criticalCount(&authority, range);
                        verified_rows = try std.math.add(usize, verified_rows, actual);
                        verified_minimum = try std.math.add(usize, verified_minimum, @max(try std.math.add(usize, actual, 1), std.math.divCeil(usize, range.rows, 4) catch return error.InvalidFitConfig));
                    }
                    if (verified_rows != evidence.critical_rows or verified_minimum != evidence.minimum_middle_rows or verified_minimum > evidence.middle_cap_rows) return error.InvalidFitConfig;
                    calibration = evidence;
                    break :calibrated authority;
                },
                .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => calibrated: {
                    if (options.geometry_budget_percent != 91 or options.fallback_geometry_budget_percent != (if (options.primary_train_probe_fn != null) @as(u8, 0) else 95)) return error.InvalidFitConfig;
                    const maxima_external = try std.math.add(usize, external_bytes, counters.cache_bytes);
                    const evidence: CriticalCoreCalibrationEvidence = evidence: {
                        const maxima = trainMaximumProbabilities(allocator, &cached, maxima_external, max_workspace_bytes, cfg.control) catch |err| return critical_phase_diagnostic.fail(err);
                        defer allocator.free(maxima);
                        const selector_peak = try std.math.add(usize, try std.math.add(usize, maxima_external, try std.math.mul(usize, maxima.len, @sizeOf(f64))), 64 * 1024);
                        if (selector_peak > max_workspace_bytes) return error.CompactionWorkspaceLimit;
                        var selected = calibrated_critical_v2.select(allocator, maxima, ranges, .{ .primary_percent = options.geometry_budget_percent, .control = cfg.control }) catch |err| return critical_phase_diagnostic.fail(err);
                        defer selected.deinit();
                        if (!std.mem.eql(u8, &selected.policy_sha256, &calibrated_critical_v2.policyDigest())) return error.InvalidFitConfig;
                        break :evidence .{ .observed = true, .threshold = selected.threshold, .critical_rows = selected.critical_rows, .minimum_middle_rows = selected.minimum_middle_rows, .middle_cap_rows = selected.middle_cap_rows, .policy_sha256 = selected.policy_sha256 };
                    };
                    var authority = extractCriticalAuthorityThreshold(allocator, &cached, ranges, evidence.threshold, maxima_external, max_workspace_bytes, cfg.control) catch |err| return critical_phase_diagnostic.fail(err);
                    errdefer authority.deinit();
                    var verified_rows: usize = 0;
                    var verified_minimum: usize = 0;
                    for (ranges) |range| {
                        const actual = try criticalCount(&authority, range);
                        verified_rows = try std.math.add(usize, verified_rows, actual);
                        verified_minimum = try std.math.add(usize, verified_minimum, @max(try std.math.add(usize, actual, try criticalResidualReserve(options.critical_core_policy, range.rows)), std.math.divCeil(usize, range.rows, 4) catch return error.InvalidFitConfig));
                    }
                    if (verified_rows != evidence.critical_rows or verified_minimum != evidence.minimum_middle_rows or verified_minimum > evidence.middle_cap_rows) return error.InvalidFitConfig;
                    calibration = evidence;
                    break :calibrated authority;
                },
            };
            defer if (critical_authority) |*authority| authority.deinit();
            const census_live = try std.math.add(usize, try std.math.add(usize, external_bytes, counters.cache_bytes), if (critical_authority) |*authority| try critical_adapter.authorityBytes(authority) else 0);
            const census = if (options.observe_train_probability_census)
                try trainProbabilityCensus(allocator, &cached, ranges, census_live, max_workspace_bytes, cfg.control)
            else
                TrainProbabilityCensus{};
            if (critical_authority) |*authority| {
                const authority_bytes = try critical_adapter.authorityBytes(authority);
                critical_phase_diagnostic.authority_bytes = authority_bytes;
                const total_rows = try allocator.alloc([3]usize, ranges.len);
                defer allocator.free(total_rows);
                const residual_rows = try allocator.alloc([3]usize, ranges.len);
                defer allocator.free(residual_rows);
                var exact_max_residual: usize = 0;
                for (ranges, total_rows, residual_rows) |range, *totals, *residuals| {
                    const core_count = try criticalCount(authority, range);
                    critical_phase_diagnostic.critical_rows = try std.math.add(usize, critical_phase_diagnostic.critical_rows, core_count);
                    const targets = try criticalAwareTargets(range.rows, core_count, options.critical_core_policy);
                    totals.* = .{ targets[0], targets[1], targets[2] };
                    for (totals.*, 0..) |total, level| {
                        residuals[level] = total - core_count;
                        critical_phase_diagnostic.target_max = @max(critical_phase_diagnostic.target_max, total);
                        exact_max_residual = @max(exact_max_residual, residuals[level]);
                    }
                }
                critical_phase_diagnostic.residual_max = exact_max_residual;
                const exact_reduction = if (options.training_reduction_policy == .accelerate_fp64_cached_v1)
                    try reductionWorkspaceBytes(train.len / source.dim, max_range_rows, exact_max_residual, source.dim, reduction_batch_rows)
                else
                    0;
                const exact_external = try std.math.add(usize, try std.math.add(usize, planner_external, counters.cache_bytes), authority_bytes);
                critical_phase_diagnostic.phase = .critical_replan;
                plan = run_critical_planner(allocator, source, train.len / source.dim, heldout_query_rows, ranges, planning_cfg, branch_bytes, exact_external, options.observe_direct_train, options.geometry_storage, active_gram_rows, gram_scratch, final_mass_strategy, joint_fixed_bytes, options.validation_backend, exact_reduction, .{ .total_rows = total_rows, .residual_rows = residual_rows }) catch |err| return critical_phase_diagnostic.fail(err);
                critical_phase_diagnostic.setPlan(plan);
                cached.consumer_peak_bytes = plan.consumer_peak_bytes;
            }
            cfg.max_workspace_bytes = @max(plan.scoring_peak_bytes, plan.signed_geometry_peak_bytes);
            critical_phase_diagnostic.phase = .score_bank;
            var bank = prepareScoredBank(allocator, source, train, ranges, cfg, &cached, &counters, options.max_local_weight, options.joint_fixed_original_memo, options.training_reduction_policy, reduction_batch_rows, options.phase_timing, activeSelectionPolicy(options), options.critical_core_policy, if (critical_authority) |*authority| authority else null) catch |err| return critical_phase_diagnostic.fail(err);
            defer bank.deinit();
            critical_phase_diagnostic.phase = .gram_geometry;
            var adaptive_plan = planAdaptiveGeometryFromPreparedBank(allocator, source, train, ranges, cfg, &cached, options.geometry_storage, active_gram_rows, options.gram_backend, options.gram_memoize_original_delta, options.train_budget_exchange, options.geometry_budget_percent, options.geometry_chooser_policy, options.gram_delta_backend, options.gram_delta_require_accelerated, options.phase_timing, &bank) catch |err| return critical_phase_diagnostic.fail(err);
            var adaptive_plan_live = true;
            defer if (adaptive_plan_live) adaptive_plan.deinit();
            if (options.train_snapshot_v2_fn) |export_fn| {
                _ = try export_fn(options.train_snapshot_context, options.train_snapshot_v2_domain_runs, source, &cached, &adaptive_plan, cfg.control);
                return error.TrainSnapshotExported;
            }
            if (options.primary_train_probe_fn) |probe_fn| {
                // This metadata was historically left at its default100.
                // The probe already requires the actual planner input p91.
                adaptive_plan.geometry_budget_percent = options.geometry_budget_percent;
                var probe_cfg = fit_cfg;
                probe_cfg.max_workspace_bytes = max_workspace_bytes;
                var probe_external = try std.math.add(usize, external_bytes, counters.cache_bytes);
                if (critical_authority) |*authority| probe_external = try std.math.add(usize, probe_external, try critical_adapter.authorityBytes(authority));
                // Bank packets are borrowed by adaptive_plan: charge once.
                probe_external = try std.math.add(usize, probe_external, try scoredPlanBytes(&adaptive_plan));
                probe_external = try std.math.add(usize, probe_external, try adaptiveGeometryBytes(&adaptive_plan));
                if (probe_external >= max_workspace_bytes) return error.CompactionWorkspaceLimit;
                try probe_fn(options.primary_train_probe_context, allocator, source, train, ranges, &cached, &adaptive_plan, probe_cfg, probe_external, max_workspace_bytes);
                return error.TrainPrimaryProbeExported;
            }
            if (options.train_snapshot_fn) |export_fn| {
                _ = try export_fn(options.train_snapshot_context, source, &cached, &adaptive_plan, cfg.control);
                return error.TrainSnapshotExported;
            }
            cfg.max_workspace_bytes = plan.final_replay_peak_bytes;
            critical_phase_diagnostic.phase = .final_replay;
            var result = runFrozenAdaptiveObservedTimed(allocator, source, train, heldout, ranges, &adaptive_plan, cfg, &cached, &counters, options.observe_direct_train, options.observe_validation_detail, options.phase_timing, options.final_refit_strategy, options.validation_backend, options.postfit_policy, options.postfit_workspace_bytes, options.rejected_branch_capture) catch |err| return critical_phase_diagnostic.fail(err);
            var selected_budget = options.geometry_budget_percent;
            var fallback_attempted = false;
            if (!result.accepted and options.fallback_geometry_budget_percent != 0) {
                const authorize = options.fallback_authorize_fn.?;
                if (!authorize(options.fallback_authorize_context)) {
                    observePrimaryRejectedFallbackDenied(&result, options.geometry_budget_percent, fallbackDeniedObservationEnabled(options.phase_timing, options.observe_train_probability_census));
                    result.deinit();
                    return error.CompactionPhysicalBudgetUnsupported;
                }
                result.deinit();
                adaptive_plan.deinit();
                adaptive_plan_live = false;
                cfg.max_workspace_bytes = @max(plan.scoring_peak_bytes, plan.signed_geometry_peak_bytes);
                critical_phase_diagnostic.phase = .gram_geometry;
                adaptive_plan = planAdaptiveGeometryFromPreparedBank(allocator, source, train, ranges, cfg, &cached, options.geometry_storage, active_gram_rows, options.gram_backend, options.gram_memoize_original_delta, options.train_budget_exchange, options.fallback_geometry_budget_percent, options.geometry_chooser_policy, options.gram_delta_backend, options.gram_delta_require_accelerated, options.phase_timing, &bank) catch |err| return critical_phase_diagnostic.fail(err);
                adaptive_plan_live = true;
                cfg.max_workspace_bytes = plan.final_replay_peak_bytes;
                critical_phase_diagnostic.phase = .final_replay;
                result = runFrozenAdaptiveObservedTimed(allocator, source, train, heldout, ranges, &adaptive_plan, cfg, &cached, &counters, options.observe_direct_train, options.observe_validation_detail, options.phase_timing, options.final_refit_strategy, options.validation_backend, options.postfit_policy, options.postfit_workspace_bytes, options.rejected_branch_capture) catch |err| return critical_phase_diagnostic.fail(err);
                selected_budget = options.fallback_geometry_budget_percent;
                fallback_attempted = true;
            }
            result.weight_cap = adaptive_plan.weight_cap;
            result.probability_storage = counters.probability_storage;
            result.geometry_budget_percent = selected_budget;
            result.gram_delta = adaptive_plan.gram_delta;
            result.training_reduction = adaptive_plan.training_reduction;
            const scored_packets = try std.math.mul(usize, bank.scores.len, 3);
            result.fallback_reuse = .{ .scored_packets = scored_packets, .geometry_attempts = @as(usize, if (fallback_attempted) 2 else 1), .final_validation_attempts = @as(usize, if (fallback_attempted) 2 else 1), .authorization_calls = @as(usize, @intFromBool(fallback_attempted)), .fallback_attempted = fallback_attempted, .primary_rejected_result_destroyed = fallback_attempted, .reselections_avoided = if (fallback_attempted) scored_packets else 0, .value_fit_dispatches_avoided = if (fallback_attempted) scored_packets else 0 };
            result.training_source = .{ .requested = true, .outcome = .ready, .digest = cached.digest, .counters = counters, .external_live_bytes = planner_external };
            result.critical_core = .{ .policy = options.critical_core_policy, .policy_digest = critical_adapter.policyDigest(options.critical_core_policy), .unique_rows = try criticalCoreRows(&bank), .calibration = calibration, .train_probability_census = census };
            return result;
        },
    }
}

pub fn runInput(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, branch_chunk_rows: usize, max_workspace_bytes: usize) !Result {
    return runInputWithOptions(allocator, input, augmented, backend, donor, head_index, branch_chunk_rows, max_workspace_bytes, .{});
}

/// Production-shaped, test-only v18 entry for an already normalized current
/// checkpoint head. The caller owns `original`, query matrices, and sparse
/// coverage-aware ranges and includes all of them in `external_live_bytes`.
/// This function performs the complete pre-cache admission before any backend
/// scoring call and owns the cache for the full adaptive replay.
pub fn runHeadV18(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    heldout: []const f64,
    ranges: []const admission.Range,
    cfg: solver.Config,
    external_live_bytes: usize,
    combined_cap: usize,
) !Result {
    if (comptime !@import("builtin").is_test) @compileError("v18 adaptive head fitting is test-only");
    if (ranges.len == 0 or combined_cap == 0 or external_live_bytes > combined_cap or cfg.max_workspace_bytes != combined_cap or train.len % original.dim != 0 or heldout.len % original.dim != 0)
        return error.InvalidFitConfig;
    const plan = replay_planner.planAdaptive(allocator, original, train.len / original.dim, heldout.len / original.dim, ranges, cfg, try std.math.mul(usize, 3, @sizeOf(Branch)), external_live_bytes, false) catch |err| switch (err) {
        error.CompactionWorkspaceLimit, error.OutOfMemory => return error.V18TrainingCacheUnsupported,
        else => return err,
    };
    var counters: training_source.Counters = .{};
    const prepared = try training_source.prepare(allocator, original, train, cfg, .{ .combined_cap = combined_cap, .external_live_bytes = external_live_bytes, .consumer_peak_bytes = plan.consumer_peak_bytes }, &counters);
    switch (prepared) {
        .fallback => return error.V18TrainingCacheUnsupported,
        .ready => |owned| {
            var cached = owned;
            defer cached.deinit();
            var scoring_cfg = cfg;
            scoring_cfg.max_workspace_bytes = plan.scoring_peak_bytes;
            var adaptive_plan = try planAdaptiveGeometry(allocator, original, train, ranges, scoring_cfg, &cached, &counters);
            defer adaptive_plan.deinit();
            var replay_cfg = cfg;
            replay_cfg.max_workspace_bytes = plan.final_replay_peak_bytes;
            var result = try runFrozenAdaptive(allocator, original, train, heldout, ranges, &adaptive_plan, replay_cfg, &cached, &counters);
            result.training_source = .{ .requested = true, .outcome = .ready, .digest = cached.digest, .counters = counters, .external_live_bytes = external_live_bytes };
            return result;
        },
    }
}

pub fn runInputV18(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, branch_chunk_rows: usize, max_workspace_bytes: usize) !Result {
    return runInputWithOptions(allocator, input, augmented, backend, donor, head_index, branch_chunk_rows, max_workspace_bytes, .{ .training_source = true, .mass_strategy = .identity_chunk_coefficients_v18 });
}

test "joint chunk mass replay uses one heldout gate per complete branch" {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var result = try run(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expect(result.accepted);
    try std.testing.expectEqual(@as(usize, 4), result.chosen_divisor);
    try std.testing.expectEqual(@as(usize, 1), result.branches.len);
    try std.testing.expectEqualStrings("none", result.branches[0].rejection);
}

test "unavailable geometry branches skip fitting and heldout without masking workspace limits" {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var geometries: [3]adaptive.Geometry = undefined;
    var made: usize = 0;
    defer for (geometries[0..made]) |*geometry| geometry.deinit();
    while (made < geometries.len) : (made += 1) {
        const levels = try std.testing.allocator.dupe(adaptive.Level, &.{.divisor4});
        errdefer std.testing.allocator.free(levels);
        const unavailable_targets = try std.testing.allocator.dupe(usize, &.{2});
        geometries[made] = .{ .allocator = std.testing.allocator, .available = false, .levels = levels, .targets = unavailable_targets, .middle_cap_rows = 1, .middle_rows = 2, .preserved_ranges = 0, .fitted_ranges = 1, .train_harm_before = 0, .train_harm_after = 0, .digest = std.mem.zeroes([32]u8) };
    }
    const targets: [3][]const usize = .{ geometries[0].targets, geometries[1].targets, geometries[2].targets };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20 };
    var result = try runWithOptions(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, cfg, .{ .adaptive_targets_by_branch = targets, .adaptive_geometry = &geometries }, null, null);
    defer result.deinit();
    try std.testing.expect(!result.accepted);
    try std.testing.expectEqual(@as(usize, 0), result.heldout_gate_calls);
    for (result.branches) |branch| try std.testing.expectEqualStrings("geometry_budget_infeasible", branch.rejection);
    var rejected_cfg = cfg;
    rejected_cfg.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, runWithOptions(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, rejected_cfg, .{ .adaptive_targets_by_branch = targets, .adaptive_geometry = &geometries }, null, null));
}

test "cached full replay exactly matches legacy and preserves source" {
    var keys = [_]f64{ 0, 0.5, 1, 1.5, 2, 2.5 };
    var values = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var bias = [_]f64{ 0, 0, 0, 0, 0, 0 };
    const keys_before = keys;
    const values_before = values;
    const bias_before = bias;
    const source: solver.Head = .{ .keys = &keys, .values = &values, .bias = &bias, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 0.5, 1 };
    const heldout = [_]f64{ 0.25, 0.75 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20 };
    var legacy = try run(std.testing.allocator, source, &train, &heldout, &ranges, cfg);
    defer legacy.deinit();
    var counters: training_source.Counters = .{};
    const prepared = try training_source.prepare(std.testing.allocator, source, &train, cfg, .{ .combined_cap = 2 << 20, .external_live_bytes = 0, .consumer_peak_bytes = 1 << 20 }, &counters);
    var cached = switch (prepared) {
        .ready => |ready| ready,
        .fallback => return error.TestUnexpectedResult,
    };
    defer cached.deinit();
    var actual = try runWithOptions(std.testing.allocator, source, &train, &heldout, &ranges, cfg, .{ .training_source = true }, &cached, &counters);
    defer actual.deinit();
    try std.testing.expect(counters.selection_hits > 0 and counters.fit_hits > 0);
    try std.testing.expectEqualSlices(f64, legacy.head.keys, actual.head.keys);
    try std.testing.expectEqualSlices(f64, legacy.head.values, actual.head.values);
    try std.testing.expectEqualSlices(f64, legacy.head.bias.?, actual.head.bias.?);
    try std.testing.expectEqualSlices(usize, legacy.source_indices, actual.source_indices);
    try std.testing.expectEqualSlices(usize, legacy.coverage_source_indices, actual.coverage_source_indices);
    try std.testing.expectEqual(legacy.accepted, actual.accepted);
    try std.testing.expectEqual(legacy.chosen_divisor, actual.chosen_divisor);
    try std.testing.expectEqual(legacy.final_metrics_valid, actual.final_metrics_valid);
    try std.testing.expectEqual(legacy.final_mass_error, actual.final_mass_error);
    try std.testing.expectEqual(legacy.final_output_rmse, actual.final_output_rmse);
    try std.testing.expectEqual(legacy.branches.len, actual.branches.len);
    for (legacy.branches, actual.branches) |expected, got| {
        try std.testing.expectEqual(expected.divisor, got.divisor);
        try std.testing.expectEqual(expected.candidate_rows, got.candidate_rows);
        try std.testing.expectEqual(expected.mass_error, got.mass_error);
        try std.testing.expectEqual(expected.output_rmse, got.output_rmse);
        try std.testing.expectEqualStrings(expected.rejection, got.rejection);
        try std.testing.expectEqualSlices(f64, expected.alpha, got.alpha);
    }
    try std.testing.expectEqualSlices(f64, &keys_before, &keys);
    try std.testing.expectEqualSlices(f64, &values_before, &values);
    try std.testing.expectEqualSlices(f64, &bias_before, &bias);
}

test "exact f32 cache preserves full replay selections heads branches and metrics" {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, control: solver.Control) !void {
            for (0..queries.len / head.dim) |qi| {
                try control.check();
                _ = try solver.attend(head, queries[qi * head.dim ..][0..head.dim], probabilities[qi * head.rows ..][0..head.rows], if (outputs) |out| out[qi * head.dim ..][0..head.dim] else null);
            }
            for (probabilities) |*value| value.* = @as(f32, @floatCast(value.*));
            if (outputs) |out| {
                for (out) |*value| value.* = @as(f32, @floatCast(value.*));
            }
        }
    };
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 0.5, 1 };
    const heldout = [_]f64{ 0.25, 0.75 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } };
    var f64_source = (try training_source.prepare(std.testing.allocator, source, &train, cfg, .{ .combined_cap = 2 << 20, .external_live_bytes = 0, .consumer_peak_bytes = 1 << 20 }, null)).ready;
    defer f64_source.deinit();
    var f32_source = (try training_source.prepare(std.testing.allocator, source, &train, cfg, .{ .combined_cap = 2 << 20, .external_live_bytes = 0, .consumer_peak_bytes = 1 << 20, .probability_storage = .exact_f32_widened }, null)).ready;
    defer f32_source.deinit();
    try std.testing.expectEqualSlices(u8, &f64_source.digest, &f32_source.digest);
    var expected = try runWithOptions(std.testing.allocator, source, &train, &heldout, &ranges, cfg, .{ .training_source = true }, &f64_source, null);
    defer expected.deinit();
    var actual = try runWithOptions(std.testing.allocator, source, &train, &heldout, &ranges, cfg, .{ .training_source = true, .probability_storage = .exact_f32_widened }, &f32_source, null);
    defer actual.deinit();
    try std.testing.expectEqualSlices(f64, expected.head.keys, actual.head.keys);
    try std.testing.expectEqualSlices(f64, expected.head.values, actual.head.values);
    try std.testing.expectEqualSlices(f64, expected.head.bias.?, actual.head.bias.?);
    try std.testing.expectEqualSlices(usize, expected.source_indices, actual.source_indices);
    try std.testing.expectEqualSlices(usize, expected.coverage_source_indices, actual.coverage_source_indices);
    try std.testing.expectEqual(expected.accepted, actual.accepted);
    try std.testing.expectEqual(expected.chosen_divisor, actual.chosen_divisor);
    try std.testing.expectEqual(expected.final_mass_error, actual.final_mass_error);
    try std.testing.expectEqual(expected.final_output_rmse, actual.final_output_rmse);
    try std.testing.expectEqual(expected.branches.len, actual.branches.len);
    for (expected.branches, actual.branches) |a, b| {
        try std.testing.expectEqual(a.mass_error, b.mass_error);
        try std.testing.expectEqual(a.output_rmse, b.output_rmse);
        try std.testing.expectEqualStrings(a.rejection, b.rejection);
        try std.testing.expectEqualSlices(f64, a.alpha, b.alpha);
    }
}

test "v18 adaptive freeze omits identity ranges and keeps original starts" {
    const source: solver.Head = .{ .keys = &.{ 0, 1, 2, 3, 4, 5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{ .{ .start = 0, .rows = 3 }, .{ .start = 3, .rows = 3 } };
    var frozen = try freezeBranch(std.testing.allocator, source, &.{ 0, 1 }, &ranges, 2, &.{ 3, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 }, null, null);
    defer frozen.deinit();
    try std.testing.expectEqual(@as(usize, 1), frozen.chunks.len);
    try std.testing.expectEqual(@as(usize, 3), frozen.chunks[0].start);
    try std.testing.expectEqualSlices(admission.Range, &.{.{ .start = 3, .rows = 3 }}, frozen.ranges);
}

test "frozen adaptive E seam observes one live rejected divisor2 candidate" {
    const Capture = struct {
        calls: usize = 0,
        fn observe(raw: ?*anyopaque, view: RejectedBranchView) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            std.testing.expectEqual(@as(usize, 2), view.divisor) catch unreachable;
            std.testing.expect(view.candidate.rows < view.original.rows) catch unreachable;
            std.testing.expectEqual(view.candidate.rows, view.source_indices.len) catch unreachable;
            std.testing.expectEqual(view.candidate.rows, view.coverage_source_indices.len) catch unreachable;
            std.testing.expect(view.metrics.quality_evaluated) catch unreachable;
        }
    };
    const source: solver.Head = .{ .keys = &.{ -2, -1, 0, 1, 2, 3 }, .values = &.{ 1, -3, 2, 7, -5, 11 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ -0.75, 0.25, 1.25 };
    const heldout = [_]f64{ -1.375, 0.875, 2.625 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 0, .max_relative_output_rmse = 0, .max_workspace_bytes = 4 << 20 };
    var cache = (try training_source.prepare(std.testing.allocator, source, &train, cfg, .{ .combined_cap = 4 << 20, .external_live_bytes = 0, .consumer_peak_bytes = 2 << 20 }, null)).ready;
    defer cache.deinit();
    var counters: training_source.Counters = .{};
    var plan = try planAdaptiveGeometry(std.testing.allocator, source, &train, &ranges, cfg, &cache, &counters);
    defer plan.deinit();
    var capture: Capture = .{};
    var result = try runFrozenAdaptiveObservedTimed(std.testing.allocator, source, &train, &heldout, &ranges, &plan, cfg, &cache, &counters, false, false, null, .identity_reuse, null, .disabled, 0, .{ .context = &capture, .capture_fn = Capture.observe });
    defer result.deinit();
    try std.testing.expect(!result.accepted);
    try std.testing.expectEqual(@as(usize, 1), capture.calls);
}

test "TRAIN primary probe admits only empty HHO values with p91 count authority" {
    const Hook = struct {
        fn run(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []const admission.Range, _: *training_source.Source, _: *const AdaptivePlan, _: solver.Config, _: usize, _: usize) anyerror!void {}
    };
    var options: Options = .{
        .training_source = true,
        .mass_strategy = .identity_chunk_coefficients_v18,
        .geometry_budget_percent = 91,
        .primary_train_probe_heldout_query_rows = 64,
        .primary_train_probe_fn = Hook.run,
    };
    try std.testing.expectEqual(@as(usize, 64), try primaryProbeHeldoutRows(options, &.{}, 512));
    const nonempty: [512]f64 = @splat(0);
    try std.testing.expectError(error.InvalidFitConfig, primaryProbeHeldoutRows(options, &nonempty, 512));
    options.fallback_geometry_budget_percent = 95;
    try std.testing.expectError(error.InvalidFitConfig, primaryProbeHeldoutRows(options, &.{}, 512));
    options = .{};
    try std.testing.expectEqual(@as(usize, 2), try primaryProbeHeldoutRows(options, &.{ 0, 0 }, 1));
    try std.testing.expectError(error.InvalidFitConfig, primaryProbeHeldoutRows(options, &.{}, 1));
}

const TestGramBackend = struct {
    fn workspace(_: ?*anyopaque, _: usize, _: usize) !usize {
        return 0;
    }
    fn compute(_: ?*anyopaque, matrix: []const f64, k: usize, dim: usize, output: []f64, _: []u8, control: solver.Control) !void {
        var at: usize = 0;
        for (0..k) |i| for (0..i + 1) |j| {
            try control.check();
            var dot: f64 = 0;
            for (0..dim) |di| dot += matrix[i * dim + di] * matrix[j * dim + di];
            output[at] = dot;
            at += 1;
        };
    }
    const backend: gram_geometry.GramBackend = .{ .context = null, .workspace_fn = workspace, .compute_fn = compute };
};

const TestGramDeltaBackend = struct {
    fn gemm(_: ?*anyopaque, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, b: []const f64, beta: f64, c: []f64) !void {
        for (0..m) |i| for (0..n) |j| {
            var sum: f64 = 0;
            for (0..k) |x| sum += a[i * k + x] * b[x * n + j];
            c[i * n + j] = alpha * sum + if (beta == 0) 0 else beta * c[i * n + j];
        };
    }
    const backend: gram_delta.Backend = .{ .gemm_fn = gemm };
};

fn adaptiveReplayWithControl(allocator: std.mem.Allocator, control: solver.Control) !void {
    const keys = [_]f64{ 0, 0, 0, 0, 0, 0 };
    const values = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const bias = [_]f64{ 0, 0, 0, 0, 0, 0 };
    const train = [_]f64{ 0, 1 };
    var probabilities = [_]f64{ 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0 };
    var outputs = [_]f64{ 3.5, 3.5 };
    const head: solver.Head = .{ .keys = &keys, .values = &values, .bias = &bias, .rows = 6, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20, .control = control };
    var source: training_source.Source = .{ .allocator = std.testing.allocator, .keys_ptr = head.keys.ptr, .values_ptr = head.values.ptr, .bias_ptr = head.bias.?.ptr, .train_ptr = train[0..].ptr, .rows = 6, .dim = 1, .query_count = 2, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 16, .cache_bytes = 14 * @sizeOf(f64), .consumer_peak_bytes = 1 << 20, .digest = std.mem.zeroes([32]u8), .probabilities = &probabilities, .outputs = &outputs };
    var counters: training_source.Counters = .{};
    const ranges = [_]admission.Range{ .{ .start = 0, .rows = 3 }, .{ .start = 3, .rows = 3 } };
    var plan = try planAdaptiveGeometry(allocator, head, &train, &ranges, cfg, &source, &counters);
    defer plan.deinit();
    const expected_support_digest = try frozenSupportBankDigest(&plan);
    try std.testing.expectEqual(@as(?f64, null), plan.weight_cap.configured);
    try std.testing.expectEqual(@as(usize, 0), plan.weight_cap.scored_candidates);
    try std.testing.expectEqual(@as(usize, 0), plan.weight_cap.clamped_weights);
    var full_mass = [_]f64{ 0, 0 };
    var full_numerator = [_]f64{ 0, 0 };
    try accumulateScoredDelta(&plan, 0, 0, head, &source, control, &full_mass, &full_numerator);
    var sliced_mass = [_]f64{ 0, 0 };
    var sliced_numerator = [_]f64{ 0, 0 };
    try accumulateScoredDeltaSlice(&plan, 0, 0, head, &source, 0, 1, control, sliced_mass[0..1], sliced_numerator[0..1]);
    try accumulateScoredDeltaSlice(&plan, 0, 0, head, &source, 1, 1, control, sliced_mass[1..2], sliced_numerator[1..2]);
    try std.testing.expectEqualSlices(f64, &full_mass, &sliced_mass);
    try std.testing.expectEqualSlices(f64, &full_numerator, &sliced_numerator);
    var oracle_counters: training_source.Counters = .{};
    var oracle = try runFrozenAdaptiveUncachedOracle(allocator, head, &train, &.{0.5}, &ranges, &plan, cfg, &source, &oracle_counters);
    defer oracle.deinit();
    if (control.check_fn == null) {
        var gram_counters: training_source.Counters = .{};
        var gram_plan = try planAdaptiveGeometryStorage(allocator, head, &train, &ranges, cfg, &source, &gram_counters, .per_query_gram, 1, TestGramBackend.backend);
        defer gram_plan.deinit();
        var memo_plan = try planAdaptiveGeometryStorageCappedMemoized(allocator, head, &train, &ranges, cfg, &source, &gram_counters, .per_query_gram, 1, TestGramBackend.backend, null, true, false, false, 100, .legacy_full_scan, null, false, .scalar_reference, 64, null, .raw);
        defer memo_plan.deinit();
        var joint_memo_plan = try planAdaptiveGeometryStorageCappedMemoized(allocator, head, &train, &ranges, cfg, &source, &gram_counters, .per_query_gram, 1, TestGramBackend.backend, null, false, false, true, 100, .legacy_full_scan, null, false, .scalar_reference, 64, null, .raw);
        defer joint_memo_plan.deinit();
        var accelerated_plan = try planAdaptiveGeometryStorageCappedMemoized(allocator, head, &train, &ranges, cfg, &source, &gram_counters, .per_query_gram, 8, TestGramBackend.backend, null, false, false, false, 100, .legacy_full_scan, TestGramDeltaBackend.backend, true, .scalar_reference, 64, null, .raw);
        defer accelerated_plan.deinit();
        try std.testing.expectEqual(GeometryStorage.per_query_gram, gram_plan.geometry_storage);
        try std.testing.expectEqual(GramMemoEvidence{}, gram_plan.gram_memo);
        try std.testing.expectEqual(GramMemoEvidence{ .enabled = true, .bytes = 16, .hits = 8, .misses = 4 }, memo_plan.gram_memo);
        try std.testing.expectEqual(JointFixedMemoEvidence{ .enabled = true, .bytes = 32, .builds = 2, .hits = 6 }, joint_memo_plan.joint_fixed_memo);
        try std.testing.expectEqual(GramDeltaEvidence{ .requested = true, .accelerated = true, .tiles = 6, .tile_rows = 8, .backend_calls = 12, .packed_bytes_peak = 32, .staged_bytes_peak = 32, .page_bytes_peak = 48 }, accelerated_plan.gram_delta);
        try std.testing.expectError(error.InvalidFitConfig, planAdaptiveGeometryStorageCappedMemoized(allocator, head, &train, &ranges, cfg, &source, &gram_counters, .per_query_gram, 8, TestGramBackend.backend, null, true, false, false, 100, .legacy_full_scan, TestGramDeltaBackend.backend, true, .scalar_reference, 64, null, .raw));
        try std.testing.expectEqualStrings(gram_geometry.policy, gram_plan.geometry_policy);

        // A prepared bank remains the sole packet owner while two independent
        // budget geometries and final validations borrow it in sequence.
        var reuse_counters: training_source.Counters = .{};
        var reuse_bank = try prepareScoredBank(allocator, head, &train, &ranges, cfg, &source, &reuse_counters, null, false, .scalar_reference, 64, null, .raw, .disabled, null);
        defer reuse_bank.deinit();
        const selections_after_bank = reuse_counters.selection_hits;
        const fits_after_bank = reuse_counters.fit_hits;
        {
            var budget91 = try planAdaptiveGeometryFromPreparedBank(allocator, head, &train, &ranges, cfg, &source, .per_query_gram, 1, TestGramBackend.backend, false, false, 91, .legacy_full_scan, null, false, null, &reuse_bank);
            defer budget91.deinit();
            try std.testing.expect(!budget91.owns_scored_packets);
            var result91 = try runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &budget91, cfg, &source, &reuse_counters, false, false);
            defer result91.deinit();
        }
        try std.testing.expectEqual(selections_after_bank, reuse_counters.selection_hits);
        try std.testing.expectEqual(fits_after_bank, reuse_counters.fit_hits);
        var budget95 = try planAdaptiveGeometryFromPreparedBank(allocator, head, &train, &ranges, cfg, &source, .per_query_gram, 1, TestGramBackend.backend, false, false, 95, .legacy_full_scan, null, false, null, &reuse_bank);
        defer budget95.deinit();
        try std.testing.expect(!budget95.owns_scored_packets);
        var result95 = try runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &budget95, cfg, &source, &reuse_counters, false, false);
        defer result95.deinit();
        try std.testing.expectEqual(selections_after_bank, reuse_counters.selection_hits);
        try std.testing.expectEqual(fits_after_bank, reuse_counters.fit_hits);
        try std.testing.expect(reuse_bank.owns_packets);
        try std.testing.expect(gram_plan.geometry_workspace_bound_bytes > 0);
        for (&plan.branches, &gram_plan.branches) |dense_branch, gram_branch| {
            try std.testing.expectEqualSlices(usize, dense_branch.targets, gram_branch.targets);
            try std.testing.expectApproxEqRel(dense_branch.train_harm_after, gram_branch.train_harm_after, 512 * std.math.floatEps(f64));
        }
        for (&gram_plan.branches, &memo_plan.branches) |uncached_branch, memo_branch| {
            try std.testing.expectEqualSlices(usize, uncached_branch.targets, memo_branch.targets);
            try std.testing.expectEqual(uncached_branch.train_harm_before, memo_branch.train_harm_before);
            try std.testing.expectEqual(uncached_branch.train_harm_after, memo_branch.train_harm_after);
            try std.testing.expectEqualSlices(u8, &uncached_branch.digest, &memo_branch.digest);
        }
        for (&gram_plan.branches, &accelerated_plan.branches) |scalar_branch, accelerated_branch| {
            try std.testing.expectEqualSlices(usize, scalar_branch.targets, accelerated_branch.targets);
            try std.testing.expectApproxEqAbs(scalar_branch.train_harm_before, accelerated_branch.train_harm_before, 1e-14);
            try std.testing.expectApproxEqAbs(scalar_branch.train_harm_after, accelerated_branch.train_harm_after, 1e-14);
        }
        for (gram_plan.scored, joint_memo_plan.scored) |uncached_levels, memo_levels| for (uncached_levels, memo_levels) |uncached_candidate, memo_candidate| {
            const lhs = uncached_candidate.?;
            const rhs = memo_candidate.?;
            try std.testing.expectEqualSlices(f64, lhs.fitted.values, rhs.fitted.values);
            try std.testing.expectEqualSlices(f64, lhs.fitted.bias, rhs.fitted.bias);
        };
        // Exercise hit, range invalidation, and a one-query tail directly;
        // every memoized result is bit-equal to the unchanged public accessor.
        var memo_mass: [1]f64 = undefined;
        var memo_numerator: [1]f64 = undefined;
        var memo_context: GramMemoContext = .{ .plan = &memo_plan, .original = head, .cached = &source, .negative_mass = &memo_mass, .negative_numerator = &memo_numerator };
        for ([_]struct { range: usize, level: usize, first: usize }{
            .{ .range = 0, .level = 0, .first = 0 },
            .{ .range = 0, .level = 1, .first = 0 },
            .{ .range = 1, .level = 0, .first = 1 },
        }) |request| {
            var expected_mass = [_]f64{0};
            var expected_numerator = [_]f64{0};
            var actual_mass = [_]f64{0};
            var actual_numerator = [_]f64{0};
            try accumulateScoredDeltaSlice(&memo_plan, request.range, request.level, head, &source, request.first, 1, .{}, &expected_mass, &expected_numerator);
            try GramMemoContext.fill(&memo_context, request.range, request.level, request.first, 1, &actual_mass, &actual_numerator, .{});
            try std.testing.expectEqualSlices(f64, &expected_mass, &actual_mass);
            try std.testing.expectEqualSlices(f64, &expected_numerator, &actual_numerator);
        }
        try std.testing.expectEqual(@as(usize, 1), memo_context.hits);
        try std.testing.expectEqual(@as(usize, 2), memo_context.misses);
        const MemoCancel = struct {
            checks: usize = 0,
            cancel_at: usize,
            fn check(raw: ?*anyopaque) !void {
                const self: *@This() = @ptrCast(@alignCast(raw.?));
                self.checks += 1;
                if (self.checks == self.cancel_at) return error.Cancelled;
            }
            fn makeControl(self: *@This()) solver.Control {
                return .{ .context = self, .check_fn = check };
            }
        };
        // Cancellation in the negative scan never publishes a valid cache
        // key; cancellation in selected addition returns an error and callers
        // discard the partially written accessor outputs.
        for ([_]usize{ 1, 2 }) |cancel_at| {
            var cancel: MemoCancel = .{ .cancel_at = cancel_at };
            var cancel_mass: [1]f64 = undefined;
            var cancel_numerator: [1]f64 = undefined;
            var cancelled_context: GramMemoContext = .{ .plan = &memo_plan, .original = head, .cached = &source, .negative_mass = &cancel_mass, .negative_numerator = &cancel_numerator };
            var out_mass = [_]f64{0};
            var out_numerator = [_]f64{0};
            try std.testing.expectError(error.Cancelled, GramMemoContext.fill(&cancelled_context, 0, 0, 0, 1, &out_mass, &out_numerator, cancel.makeControl()));
            if (cancel_at == 1) try std.testing.expectEqual(@as(?usize, null), cancelled_context.cached_range);
        }
        var gram_result = try runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &gram_plan, cfg, &source, &gram_counters, false, false);
        defer gram_result.deinit();
        try std.testing.expectEqual(gram_plan.gram_memo, gram_result.gram_memo);
        var memo_result = try runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &memo_plan, cfg, &source, &gram_counters, false, false);
        defer memo_result.deinit();
        try std.testing.expectEqual(memo_plan.gram_memo, memo_result.gram_memo);
        try std.testing.expectEqualSlices(f64, gram_result.head.keys, memo_result.head.keys);
        try std.testing.expectEqualSlices(f64, gram_result.head.values, memo_result.head.values);
        try std.testing.expectEqualSlices(f64, gram_result.head.bias.?, memo_result.head.bias.?);
        try std.testing.expectEqual(GeometryStorage.per_query_gram, gram_result.geometry_storage);
        try std.testing.expectEqualSlices(f64, oracle.head.keys, gram_result.head.keys);
        for (oracle.head.values, gram_result.head.values) |expected, actual| try std.testing.expectApproxEqRel(expected, actual, 512 * std.math.floatEps(f64));

        // The optional cap is applied to every scored packet before that
        // packet's joint value fit and is retained into replay evidence.
        const cap: f64 = 0.25;
        var capped_counters: training_source.Counters = .{};
        var capped_plan = try planAdaptiveGeometryStorageCapped(allocator, head, &train, &ranges, cfg, &source, &capped_counters, .dense, 1, null, cap);
        defer capped_plan.deinit();
        try std.testing.expectEqual(@as(?f64, cap), capped_plan.weight_cap.configured);
        try std.testing.expectEqual(ranges.len * 3, capped_plan.weight_cap.scored_candidates);
        try std.testing.expect(capped_plan.weight_cap.clamped_weights > 0);
        for (capped_plan.scored) |levels| for (levels) |entry| if (entry) |candidate| for (candidate.frozen.weights) |weights| for (weights) |weight| {
            try std.testing.expect(weight >= 0 and weight <= cap);
        };
        var capped_result = try runFrozenAdaptive(allocator, head, &train, &.{0.5}, &ranges, &capped_plan, cfg, &source, &capped_counters);
        defer capped_result.deinit();
        try std.testing.expectEqual(capped_plan.weight_cap, capped_result.weight_cap);
        var rejected_oracle_counters: training_source.Counters = .{};
        try std.testing.expectError(error.InvalidFitConfig, runFrozenAdaptiveUncachedOracle(allocator, head, &train, &.{0.5}, &ranges, &capped_plan, cfg, &source, &rejected_oracle_counters));
    }
    var rejected_cfg = cfg;
    rejected_cfg.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, runFrozenAdaptive(allocator, head, &train, &.{0.5}, &ranges, &plan, rejected_cfg, &source, &counters));
    const CancelReplay = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled_cfg = cfg;
    cancelled_cfg.control = .{ .check_fn = CancelReplay.check };
    try std.testing.expectError(error.Cancelled, runFrozenAdaptive(allocator, head, &train, &.{0.5}, &ranges, &plan, cancelled_cfg, &source, &counters));
    try std.testing.expectError(error.Cancelled, runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &plan, cancelled_cfg, &source, &counters, true, true));
    const selection_hits_before = counters.selection_hits;
    const fit_hits_before = counters.fit_hits;
    var result = try runFrozenAdaptive(allocator, head, &train, &.{0.5}, &ranges, &plan, cfg, &source, &counters);
    defer result.deinit();
    try std.testing.expectEqualSlices(u8, &expected_support_digest, &result.frozen_support_bank_digest);
    try std.testing.expectEqual(FinalRefitEvidence{}, result.final_refit);
    if (control.check_fn == null) {
        var page_backed = try runFrozenAdaptivePageBacked(allocator, head, &train, &.{0.5}, &ranges, &plan, cfg, false, null);
        defer page_backed.deinit();
        try std.testing.expectEqualSlices(u8, &expected_support_digest, &page_backed.frozen_support_bank_digest);
        var refit_counters: training_source.Counters = .{};
        var refit = try runFrozenAdaptiveObservedTimed(allocator, head, &train, &.{0.5}, &ranges, &plan, cfg, &source, &refit_counters, true, false, null, .global_active_nnls_v16, null, .disabled, 0, null);
        defer refit.deinit();
        try std.testing.expectEqual(FinalRefitStrategy.global_active_nnls_v16, refit.final_refit.strategy);
        try std.testing.expect(refit.final_refit.attempts > 0);
        try std.testing.expect(refit.final_refit.completions > 0);
        try std.testing.expect(refit.final_refit.completions <= refit.final_refit.attempts);
        try std.testing.expectEqualSlices(u8, &expected_support_digest, &refit.frozen_support_bank_digest);
        try std.testing.expectEqualSlices(usize, result.source_indices, refit.source_indices);
        try std.testing.expectEqual(result.branches.len, refit.branches.len);
        for (result.branches, refit.branches) |identity, active| {
            try std.testing.expectEqual(identity.geometry_digest, active.geometry_digest);
            try std.testing.expectEqualSlices(usize, identity.geometry_targets, active.geometry_targets);
            for (active.alpha) |alpha| try std.testing.expect(std.math.isFinite(alpha) and alpha >= cfg.min_weight);
        }
    }
    try std.testing.expectEqual(plan.weight_cap, result.weight_cap);
    var observed_counters: training_source.Counters = .{};
    var observed = try runFrozenAdaptiveObserved(allocator, head, &train, &.{0.5}, &ranges, &plan, cfg, &source, &observed_counters, true, true);
    defer observed.deinit();
    try std.testing.expectEqualSlices(f64, oracle.head.keys, result.head.keys);
    try std.testing.expectEqualSlices(f64, oracle.head.values, result.head.values);
    try std.testing.expectEqualSlices(f64, oracle.head.bias.?, result.head.bias.?);
    try std.testing.expectEqualSlices(usize, oracle.source_indices, result.source_indices);
    try std.testing.expectEqualSlices(usize, oracle.coverage_source_indices, result.coverage_source_indices);
    try std.testing.expectEqual(result.accepted, observed.accepted);
    try std.testing.expectEqual(result.chosen_divisor, observed.chosen_divisor);
    try std.testing.expectEqual(result.final_mass_error, observed.final_mass_error);
    try std.testing.expectEqual(result.final_output_rmse, observed.final_output_rmse);
    try std.testing.expectEqualSlices(f64, result.head.keys, observed.head.keys);
    try std.testing.expectEqualSlices(f64, result.head.values, observed.head.values);
    try std.testing.expectEqualSlices(f64, result.head.bias.?, observed.head.bias.?);
    try std.testing.expectEqual(oracle.accepted, result.accepted);
    try std.testing.expectEqual(oracle.chosen_divisor, result.chosen_divisor);
    try std.testing.expectEqual(oracle.final_mass_error, result.final_mass_error);
    try std.testing.expectEqual(oracle.final_output_rmse, result.final_output_rmse);
    try std.testing.expectEqualStrings(adaptive.signed_joint_policy, plan.geometry_policy);
    try std.testing.expectEqualStrings(plan.geometry_policy, oracle.geometry_policy);
    try std.testing.expectEqualStrings(plan.geometry_policy, result.geometry_policy);
    try std.testing.expectEqual(oracle.branches.len, result.branches.len);
    for (oracle.branches, result.branches) |expected, actual| {
        try std.testing.expectEqual(expected.divisor, actual.divisor);
        try std.testing.expectEqual(expected.candidate_rows, actual.candidate_rows);
        try std.testing.expectEqual(expected.mass_error, actual.mass_error);
        try std.testing.expectEqual(expected.output_rmse, actual.output_rmse);
        try std.testing.expectEqualStrings(expected.rejection, actual.rejection);
        try std.testing.expectEqualSlices(f64, expected.alpha, actual.alpha);
        try std.testing.expectEqual(expected.geometry_digest, actual.geometry_digest);
        try std.testing.expectEqualSlices(usize, expected.geometry_targets, actual.geometry_targets);
    }
    try std.testing.expectEqual(result.branches.len, observed.branches.len);
    for (result.branches, observed.branches) |expected, actual| {
        try std.testing.expectEqual(expected.divisor, actual.divisor);
        try std.testing.expectEqual(expected.candidate_rows, actual.candidate_rows);
        try std.testing.expectEqual(expected.mass_error, actual.mass_error);
        try std.testing.expectEqual(expected.output_rmse, actual.output_rmse);
        try std.testing.expectEqualStrings(expected.rejection, actual.rejection);
        try std.testing.expectEqualSlices(f64, expected.alpha, actual.alpha);
        try std.testing.expectEqual(expected.geometry_digest, actual.geometry_digest);
        try std.testing.expectEqualSlices(usize, expected.geometry_targets, actual.geometry_targets);
        try std.testing.expect(expected.direct_train_metrics == null);
        try std.testing.expect(expected.heldout_detail == null);
        const train_metrics = actual.direct_train_metrics orelse return error.TestUnexpectedResult;
        const detail = actual.heldout_detail orelse return error.TestUnexpectedResult;
        try std.testing.expect(std.math.isFinite(train_metrics.max_relative_mass_error));
        try std.testing.expect(std.math.isFinite(train_metrics.relative_output_rmse));
        try std.testing.expectEqual(actual.mass_error.?, detail.validation.worst_relative_mass_error);
        try std.testing.expect(detail.validation.worst_query_index < 1);
        try std.testing.expect(detail.validation.original_peak_row < head.rows);
        try std.testing.expect(detail.validation.compact_peak_row < actual.candidate_rows);
        try std.testing.expect(detail.candidate_peak_source_index < head.rows);
        try std.testing.expect(detail.validation.queries_over_0_9 <= detail.validation.queries_over_0_5);
        try std.testing.expect(detail.validation.queries_over_0_5 <= detail.validation.queries_over_0_05);
        const retained_row = std.mem.indexOfScalar(usize, observed.source_indices, detail.validation.original_peak_row);
        try std.testing.expectEqual(retained_row != null, detail.original_peak_retained);
        if (retained_row) |candidate_row| {
            const expected_weight = @exp(observed.head.bias.?[candidate_row] - head.bias.?[detail.validation.original_peak_row]);
            try std.testing.expectEqual(expected_weight, detail.original_peak_candidate_weight_ratio.?);
        } else try std.testing.expect(detail.original_peak_candidate_weight_ratio == null);
        const candidate_source_bias = head.bias.?[detail.candidate_peak_source_index];
        try std.testing.expectEqual(candidate_source_bias, detail.candidate_peak_original_bias);
        try std.testing.expectEqual(@exp(detail.candidate_peak_bias - candidate_source_bias), detail.candidate_peak_weight_ratio.?);
    }
    try std.testing.expect(oracle_counters.selection_hits > 0 and oracle_counters.fit_hits > 0);
    try std.testing.expectEqual(selection_hits_before, counters.selection_hits);
    try std.testing.expectEqual(fit_hits_before, counters.fit_hits);
    try std.testing.expect(result.scored_candidates_reused);
    try std.testing.expect(result.scored_packets_reused > 0);
    try std.testing.expectEqual(result.scored_packets_reused, result.reselections_avoided);
    try std.testing.expect(result.value_fit_dispatches_avoided > 0);
}
fn adaptiveReplayAllocationFailure(allocator: std.mem.Allocator) !void {
    return adaptiveReplayWithControl(allocator, .{});
}
test "v18 scoring geometry and attached branch targets unwind every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, adaptiveReplayAllocationFailure, .{});
}
test "v18 cancellation during cached scoring unwinds and leaves cache caller-owned" {
    const Cancel = struct {
        var checks: usize = 0;
        fn check(_: ?*anyopaque) !void {
            checks += 1;
            if (checks == 8) return error.Cancelled;
        }
    };
    Cancel.checks = 0;
    try std.testing.expectError(error.Cancelled, adaptiveReplayWithControl(std.testing.allocator, .{ .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 8), Cancel.checks);
}

fn jointReplayAllocationFailure(allocator: std.mem.Allocator) !void {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var result = try run(allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20 });
    result.deinit();
}

test "joint chunk replay cap rejects and unwinds every allocation failure" {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    try std.testing.expectError(error.CompactionWorkspaceLimit, run(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, jointReplayAllocationFailure, .{});
}

const RawRangeLeaseTestContext = struct {
    allocator: std.mem.Allocator = undefined,
    owned: ?*u8 = null,
    begins: usize = 0,
    ends: usize = 0,

    fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
        return 0;
    }
    fn begin(raw: ?*anyopaque, allocator: std.mem.Allocator, _: solver.Backend.RawRangeIdentity, _: usize) !solver.Backend.RawRangeSession {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.begins += 1;
        const owned = try allocator.create(u8);
        owned.* = 0;
        self.allocator = allocator;
        self.owned = owned;
        return .{ .context = owned, .generation = self.begins };
    }
    fn end(raw: ?*anyopaque, token: solver.Backend.RawRangeSession) void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        const owned = self.owned orelse return;
        if (token.context != @as(*anyopaque, @ptrCast(owned))) return;
        self.owned = null;
        self.allocator.destroy(owned);
        self.ends += 1;
    }
    fn backend(self: *@This()) solver.Backend {
        return .{ .context = self, .workspace_fn = workspace, .raw_range_policy = .range_shared_fp64_gram_original_kkt_v1, .raw_range_begin_fn = begin, .raw_range_end_fn = end };
    }
};

fn testRangeIdentity() solver.Backend.RawRangeIdentity {
    return .{ .range_start = 0, .range_rows = 2, .query_rows = 2, .source_rows = 2, .source_identity = @splat(1), .train_identity = @splat(2), .critical_authority_identity = @splat(3), .policy_identity = rangePolicyIdentity() };
}

fn failInsideRangeLease(ctx: *RawRangeLeaseTestContext, err: anyerror) !void {
    var lease = try RawRangeLease.open(ctx.backend(), std.testing.allocator, testRangeIdentity(), 64);
    defer lease.deinit();
    return err;
}

test "range session begin allocation failure publishes no token and calls no end" {
    var ctx: RawRangeLeaseTestContext = .{};
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, RawRangeLease.open(ctx.backend(), failing.allocator(), testRangeIdentity(), 64));
    try std.testing.expectEqual(@as(usize, 1), ctx.begins);
    try std.testing.expectEqual(@as(usize, 0), ctx.ends);
    try std.testing.expect(ctx.owned == null);
}

test "range session cancellation ends exactly once" {
    var ctx: RawRangeLeaseTestContext = .{};
    try std.testing.expectError(error.Cancelled, failInsideRangeLease(&ctx, error.Cancelled));
    try std.testing.expectEqual(@as(usize, 1), ctx.begins);
    try std.testing.expectEqual(@as(usize, 1), ctx.ends);
    try std.testing.expect(ctx.owned == null);
}

test "range session candidate two failure ends exactly once" {
    var ctx: RawRangeLeaseTestContext = .{};
    try std.testing.expectError(error.OutOfMemory, failInsideRangeLease(&ctx, error.OutOfMemory));
    try std.testing.expectEqual(@as(usize, 1), ctx.begins);
    try std.testing.expectEqual(@as(usize, 1), ctx.ends);
    try std.testing.expect(ctx.owned == null);
}

const CriticalThreeLevelRangeObserver = metal_fit.Fitter.CpuRangeTestBackend;

fn criticalThreeLevelFixture(observer: *CriticalThreeLevelRangeObserver, cancel_second: bool) !void {
    const allocator = std.testing.allocator;
    const full = [_]f64{
        0.90, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
        0.88, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
    };
    const range: admission.Range = .{ .start = 1, .rows = 6 };
    var authority = try critical_keys.extractDense(allocator, .{
        .values = &full,
        .query_rows = 2,
        .complete_query_rows = 2,
        .source_rows = 8,
        .source_key_space = .global_original,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
    }, &.{.{ .start = range.start, .rows = range.rows }}, .{ .combined_cap = 1 << 20 });
    defer authority.deinit();
    const head: solver.Head = .{
        .keys = &.{ 0, 1, 2, 3, 4, 5, 6, 7 },
        .values = &.{ 0, 1, 2, 3, 4, 5, 6, 7 },
        .bias = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .rows = 8,
        .dim = 1,
        .scale = 1,
    };
    const train = [_]f64{ 0, 1 };
    const cached_probabilities = try allocator.dupe(f32, &.{
        0.03, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.04, 0.01, 0.01, 0.01, 0.01, 0.01,
    });
    var outputs = [_]f64{ 0, 0 };
    const backend = observer.backend();
    var source = try training_source.initBorrowedRangeF32(allocator, head, &train, .{ .target_rows = 2, .backend = backend }, range.start, range.rows, cached_probabilities, &outputs, 1 << 20);
    defer source.deinit();
    var lease = try RawRangeLease.open(backend, allocator, .{
        .range_start = range.start,
        .range_rows = range.rows,
        .query_rows = 2,
        .source_rows = head.rows,
        .source_identity = sourceIdentity(head, source.digest),
        .train_identity = digestBytes(std.mem.sliceAsBytes(&train)),
        .critical_authority_identity = criticalAuthorityIdentity(&authority),
        .policy_identity = rangePolicyIdentity(),
    }, try backend.raw_range_workspace_fn.?(backend.context, range.rows));
    defer lease.deinit();
    var cfg: solver.Config = .{
        .target_rows = 2,
        .selection = .residual_omp,
        .mass_solver = .active_set,
        .max_workspace_bytes = 1 << 20,
        .backend = backend,
        .raw_range_policy = .range_shared_fp64_gram_original_kkt_v1,
        .raw_range_session = lease.token.?,
    };
    const targets = [_]usize{ 2, 3, 4 };
    for (targets, 0..) |target, level| {
        const Cancel = struct {
            fn check(_: ?*anyopaque) !void {
                return error.Cancelled;
            }
        };
        if (cancel_second and level == 1) cfg.control = .{ .check_fn = Cancel.check };
        var frozen = freezeBranchPolicy(allocator, head, &train, &.{range}, 1, &.{target}, cfg, &source, null, .raw, &authority) catch |err| {
            if (cancel_second and level == 1 and err == error.Cancelled) return;
            return err;
        };
        frozen.deinit();
        cfg.target_rows = target;
    }
    if (cancel_second) return error.TestExpectedError;
}

test "critical three level freeze shares residual map P RHS and one range lease" {
    var observer: CriticalThreeLevelRangeObserver = .{};
    try criticalThreeLevelFixture(&observer, false);
    try std.testing.expectEqual(@as(usize, 1), observer.begins);
    try std.testing.expectEqual(@as(usize, 1), observer.ends);
    try std.testing.expectEqual(@as(u64, 1), observer.counters.gram_advisory.gram_precomputes);
    try std.testing.expectEqual(@as(usize, 3), observer.consumes);
    try std.testing.expectEqual(@as(u64, 3), observer.counters.gram_advisory.candidate_attempts);
    try std.testing.expectEqual(@as(u64, 3), observer.counters.gram_advisory.native_original_p_kkt_checks);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, &observer.q);
    try std.testing.expectEqualSlices(usize, &.{ 5, 5, 5 }, &observer.n);
    try std.testing.expectEqual(observer.p_sha256[0], observer.p_sha256[1]);
    try std.testing.expectEqual(observer.p_sha256[0], observer.p_sha256[2]);
    try std.testing.expectEqual(observer.rhs_sha256[0], observer.rhs_sha256[1]);
    try std.testing.expectEqual(observer.rhs_sha256[0], observer.rhs_sha256[2]);
}

test "critical three level freeze candidate two cancellation ends range lease exactly once" {
    var observer: CriticalThreeLevelRangeObserver = .{};
    try criticalThreeLevelFixture(&observer, true);
    try std.testing.expectEqual(@as(usize, 1), observer.begins);
    try std.testing.expectEqual(@as(usize, 1), observer.ends);
    try std.testing.expectEqual(@as(u64, 1), observer.counters.gram_advisory.gram_precomputes);
    try std.testing.expectEqual(@as(usize, 1), observer.consumes);
    try std.testing.expectEqual(@as(u64, 1), observer.counters.gram_advisory.candidate_attempts);
    try std.testing.expectEqual(@as(u64, 1), observer.counters.gram_advisory.native_original_p_kkt_checks);
}
