// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Test-only offline replay of the exact production768 all-head policy from an
//! authenticated whole-fit parent plus complete-future capture. The caller
//! owns the backend and all artifact authority. Heldout rows are copied once
//! from the parent and never select a policy or fallback assignment.
const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const future = @import("attention_match_future_augmentation.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const historical_coverage = @import("attention_match_production768_historical_coverage.zig");
const probability_accelerate = @import("attention_match_accelerate_fp64_probability_scan.zig");
const fp64_value = @import("attention_match_accelerate_fp64_value_backend.zig");
const challenge_rewrite = @import("attention_match_production768_train_challenge_rewrite.zig");
const checkpoint_phase = @import("attention_match_checkpoint_phase_loader.zig");
const exact_reservation = @import("attention_match_train_pressure_exact_head_reservation.zig");
const planned_exact = @import("attention_match_planned_exact_context.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");
const physical = @import("attention_match_physical_budget.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");
const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
pub const RawSelectionSolver = Fitter.RawSelectionSolver;
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
pub const Rotated4RiskRetention = engine.Rotated4RiskRetention;
pub const PostfitPolicy = @import("attention_match_joint_chunk_mass_replay.zig").PostfitPolicy;
pub const CriticalCorePolicy = @import("attention_match_joint_chunk_mass_replay.zig").CriticalCorePolicy;
pub const PerformanceComparisonPolicy = enum { legacy, owned_capture_summary_incremental_reduction_v1, owned_capture_summary_incremental_reduction_critical025_v1 };
pub const TrainExactReservationPolicy = enum { disabled, highest_budget_calibrated_critical_threshold_v1 };

pub const HistoricalCoverageBinding = struct {
    dense: augmentation.Binding,
    dense_outer: historical_coverage.OuterBinding,
    dense_nested_manifest_sha256: [32]u8,
    /// This is an offline diagnostic derivation. The complete mix and Dense512
    /// artifact must be siblings under the exact same authenticated parent and
    /// current producer; no old-pool/current-pool provenance mix is accepted.
    maximum_workspace_bytes: usize = 128 * 1024 * 1024,
};

pub const policy = "production768_complete_future_critical_p91_p95_all_head_v1";
pub const historical_coverage_accelerate_policy = "historical_coverage_probability_accelerate_fp64_gather_eligible_v1|train_only|scalar_default|no_hho";

pub const HistoricalCoverageProbabilityBackend = enum {
    scalar,
    accelerate_fp64_gather_eligible_v1,
};

pub const CompleteAuthority = struct {
    output_dir: []const u8,
    manifest_sha256: [32]u8,
    primary_parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    authority: @FieldType(future.CompleteReadBinding, "authority"),
    historical_study_tokens_sha256: [32]u8,
};

pub const Binding = struct {
    /// V16-only explicit raw selector. Older bindings retain Householder.
    raw_selection_solver: Fitter.RawSelectionSolver = .householder,
    raw_selection_solver_explicit: bool = false,
    raw_selection_policy_sha256: ?[32]u8 = null,
    encoding: checkpoint.Encoding = .f16,
    rotated4_risk_retention: Rotated4RiskRetention = .quarter_v1,
    critical_core_policy: CriticalCorePolicy = .train_probability_gte_0_025_no_top1_v1,
    postfit_policy: PostfitPolicy = .disabled,
    rejected_detail_diagnostic: bool = false,
    train_probability_census: bool = false,
    performance_comparison_policy: PerformanceComparisonPolicy = .legacy,
    train_exact_reservation_policy: TrainExactReservationPolicy = .disabled,
    train_primary_policy_comparison: bool = false,
    train_primary_policy_cross_validation: bool = false,
    historical_coverage: ?HistoricalCoverageBinding = null,
    /// Default-off retained-development experiment. Builds one TRAIN-only
    /// primary pilot per non-reserved head and rewrites H256 from Dense512
    /// before the ordinary engine receives any heldout query values.
    train_challenge_historical_selection: bool = false,
    /// Default-off v10 diagnostic. Retain authenticated coverage winners and
    /// deterministic Source918 H256 anchors before TRAIN-only challenge fill.
    hybrid_coverage_anchor: bool = false,
    historical_coverage_probability_backend: HistoricalCoverageProbabilityBackend = .scalar,
    train_challenge_mass_only: bool = false,
    phase_profiling: bool = false,
    serial_pilot_fp64_value_backend: bool = false,
    /// Default-off, single-worker diagnostic that routes both TRAIN-only
    /// challenge-pilot and final value fits through the existing bounded
    /// Accelerate FP64 backend. Query bytes, selection and gates are unchanged.
    serial_full_fp64_value_backend: bool = false,
    /// V15 serial control arm. It changes only requested/actual worker count
    /// and phase evidence; the existing Fitter value path remains selected.
    serial_full_fp64_control: bool = false,
    whole_fit: whole.Binding,
    complete_future: CompleteAuthority,
    /// One backend selects the production serial engine path; two through four
    /// select the production task-sorted parallel path. The native wrapper
    /// must construct exactly the worker count admitted by Fitter.parallelPlan.
    maximum_workers: usize = 4,
    max_workspace_bytes: usize = 1024 * 1024 * 1024,
    max_transient_host_bytes: usize,
    query_reservation_bytes: usize,
    control: solver.Control = .{},
};

const Merged = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    fn deinit(self: *@This()) void {
        self.allocator.free(self.batches);
        self.allocator.free(self.values);
        self.allocator.free(self.positions);
        self.* = undefined;
    }
};

fn mergeComplete(allocator: std.mem.Allocator, primary: []const engine.ReferenceBatch, historical: []const engine.ReferenceBatch, future_batches: []const engine.ReferenceBatch, control: solver.Control) !Merged {
    if (primary.len == 0 or primary.len != historical.len or primary.len != future_batches.len) return error.InvalidFutureAugmentation;
    var value_count: usize = 0;
    var position_count: usize = 0;
    for (primary, historical, future_batches) |p, h, f| {
        try control.check();
        if (p.consumer_layer != h.consumer_layer or p.consumer_layer != f.consumer_layer or p.query_head != h.query_head or p.query_head != f.query_head or
            p.train_positions.len != 256 or p.heldout_positions.len != 64 or h.train_positions.len != 256 or f.train_positions.len != 256 or
            h.heldout.len != 0 or h.heldout_positions.len != 0 or f.heldout.len != 0 or f.heldout_positions.len != 0 or
            p.train.len % 256 != 0 or h.train.len != p.train.len or f.train.len != p.train.len)
            return error.InvalidFutureAugmentation;
        value_count = try std.math.add(usize, value_count, try std.math.add(usize, p.train.len, try std.math.add(usize, h.train.len, f.train.len)));
        position_count = try std.math.add(usize, position_count, 768);
    }
    const batches = try allocator.alloc(engine.ReferenceBatch, primary.len);
    errdefer allocator.free(batches);
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    var value_at: usize = 0;
    var position_at: usize = 0;
    for (primary, historical, future_batches, batches) |p, h, f, *out| {
        try control.check();
        const begin = value_at;
        for ([_][]const f32{ p.train, h.train, f.train }) |source| {
            @memcpy(values[value_at..][0..source.len], source);
            value_at += source.len;
        }
        const position_begin = position_at;
        for ([_][]const u64{ p.train_positions, h.train_positions, f.train_positions }) |source| {
            @memcpy(positions[position_at..][0..source.len], source);
            position_at += source.len;
        }
        out.* = .{
            .consumer_layer = p.consumer_layer,
            .query_head = p.query_head,
            .train = values[begin..value_at],
            .heldout = p.heldout,
            .train_positions = positions[position_begin..position_at],
            .heldout_positions = p.heldout_positions,
            .independent_train_split = 256,
            .independent_train_second_split = 512,
        };
        try engine.validateReferencePositions(out.*, 7936);
    }
    if (value_at != values.len or position_at != positions.len) return error.InvalidFutureAugmentation;
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions };
}

fn eligibleShape(state: checkpoint.State, options: engine.Options) !struct { rows: usize, dim: usize, heads: usize } {
    var rows: usize = 0;
    var dim: usize = 0;
    var heads: usize = 0;
    for (state.donors) |donor| if (donor.window == 0) for (donor.heads) |head| {
        if (!physical.freshContiguousHead(head)) return error.InvalidFitConfig;
        if ((rows != 0 and rows != head.positions.len) or (dim != 0 and dim != head.dim)) return error.InvalidFitConfig;
        if ((try engine.headPlan(state.logical_length, head, options)) == null) return error.InvalidFitConfig;
        rows = head.positions.len;
        dim = head.dim;
        heads = try std.math.add(usize, heads, 1);
    };
    if (rows == 0 or dim == 0 or heads != 8) return error.InvalidFitConfig;
    return .{ .rows = rows, .dim = dim, .heads = heads };
}

pub const Result = struct {
    candidate: checkpoint.Owned,
    stats: engine.Stats,
    source_physical_bytes: usize,
    candidate_physical_bytes: usize,
    physical_ratio: f64,
    primary_budget_percent: u8,
    fallback_budget_percent: u8,
    maximum_fallbacks: u8,
    workers: usize,
    source_host_bytes: usize,
    admitted_fit_host_bytes: usize,
    historical_coverage_evidence: ?historical_coverage.Evidence,
    train_challenge_evidence: ?challenge_rewrite.Evidence,
    planned_exact: PlannedExactEvidence,
    accepted: bool,
    train_challenge_mass_only_calls: usize,
    train_challenge_mass_only_gemm_calls: usize,
    train_challenge_validation_summary_calls: usize,
    fitter_phase_counters: [4]?Fitter.PhaseCounters,
    serial_pilot_fp64_value_stats: fp64_value.Stats,
    historical_coverage_wall_ns: u64,
    serial_pilot_rewrite_wall_ns: u64,
    serial_pilot_wall_ns: [8]u64,
    serial_pilot_input_extract_ns: [8]u64,
    serial_pilot_query_frontend_ns: [8]u64,
    serial_pilot_selection_fit_ns: [8]u64,
    serial_pilot_value_fit_ns: [8]u64,
    serial_pilot_validation_and_postfit_ns: [8]u64,
    serial_pilot_reload_ns: [8]u64,
    serial_pilot_encode_rewrite_ns: [8]u64,
    serial_pilot_observer_timing: challenge_rewrite.PilotTiming,
    final_engine_wall_ns: u64,
    pub fn deinit(self: *@This()) void {
        self.candidate.deinit();
        self.* = undefined;
    }
};

pub const PlannedExactEvidence = struct {
    policy: TrainExactReservationPolicy = .disabled,
    policy_sha256: [32]u8 = @splat(0),
    evidence_sha256: [32]u8 = @splat(0),
    donor_layer: u32 = 0,
    head_index: usize = 0,
    adaptive_fit_attempts: usize = 0,
    exact_clone_rows: usize = 0,
    frozen_before_heldout: bool = false,
};

fn trainPrimaryComparisonValid(binding: Binding) bool {
    return !binding.train_primary_policy_comparison or
        (binding.historical_coverage != null and
            binding.performance_comparison_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
            binding.critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
            binding.postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 and
            binding.train_probability_census and !binding.rejected_detail_diagnostic);
}

fn trainPrimaryCrossValidationValid(binding: Binding) bool {
    return !binding.train_primary_policy_cross_validation or
        (binding.train_primary_policy_comparison and trainPrimaryComparisonValid(binding) and
            binding.historical_coverage != null and binding.train_exact_reservation_policy == .highest_budget_calibrated_critical_threshold_v1);
}

fn plannedExactPolicyValid(binding: Binding) bool {
    return binding.train_exact_reservation_policy == .disabled or
        (binding.train_exact_reservation_policy == .highest_budget_calibrated_critical_threshold_v1 and
            binding.historical_coverage != null and
            (if (binding.train_primary_policy_comparison)
                trainPrimaryComparisonValid(binding)
            else if (binding.train_challenge_historical_selection)
                trainChallengeHistoricalValid(binding)
            else if (binding.postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1)
                binding.performance_comparison_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and binding.critical_core_policy == .train_probability_gte_0_025_no_top1_v1
            else
                binding.performance_comparison_policy == .owned_capture_summary_incremental_reduction_v1 and binding.critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2) and
            (binding.postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 or binding.postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1) and
            !binding.rejected_detail_diagnostic and binding.train_probability_census);
}

fn sameExactHead(a: checkpoint.Head, b: checkpoint.Head) bool {
    return a.dim == b.dim and a.compacted_until == b.compacted_until and
        std.mem.eql(u64, a.positions, b.positions) and
        std.mem.eql(u64, a.coverage_ends, b.coverage_ends) and
        std.mem.eql(u8, std.mem.sliceAsBytes(a.keys), std.mem.sliceAsBytes(b.keys)) and std.mem.eql(u8, std.mem.sliceAsBytes(a.values), std.mem.sliceAsBytes(b.values)) and
        std.mem.eql(u8, std.mem.sliceAsBytes(a.bias), std.mem.sliceAsBytes(b.bias)) and
        a.fp8.first_row == b.fp8.first_row and a.fp8.rows == b.fp8.rows and
        @as(u32, @bitCast(a.fp8.key_scale)) == @as(u32, @bitCast(b.fp8.key_scale)) and
        @as(u32, @bitCast(a.fp8.value_scale)) == @as(u32, @bitCast(b.fp8.value_scale)) and
        a.compressed.version == b.compressed.version and
        std.mem.eql(u8, std.mem.sliceAsBytes(a.compressed.spans), std.mem.sliceAsBytes(b.compressed.spans)) and
        std.mem.eql(u8, a.compressed.keys, b.compressed.keys) and std.mem.eql(u8, a.compressed.values, b.compressed.values);
}

fn codecEvidenceValid(encoding: checkpoint.Encoding, stats: engine.Stats, expected_heads: usize) bool {
    return switch (encoding) {
        .f16 => true,
        .fp8 => stats.fp8_heads > 0 and stats.fp8_encoded_rows > 0 and
            stats.fp8_heads + stats.fp8_fallback_heads == expected_heads and
            stats.fp8_eligible_rows > 0 and
            stats.fp8_encoded_rows + stats.fp8_exact_rows == stats.fp8_eligible_rows,
        .rotated4 => stats.rotated4_heads > 0 and stats.rotated4_compressed_rows > 0 and
            stats.rotated4_heads + stats.rotated4_fallback_heads == expected_heads and
            stats.rotated4_eligible_rows > 0 and
            stats.rotated4_compressed_rows + stats.rotated4_exact_rows == stats.rotated4_eligible_rows,
        else => false,
    };
}

fn findHead(state: checkpoint.State, donor_layer: u32, head_index: usize) !checkpoint.Head {
    for (state.donors) |donor| if (donor.layer == donor_layer) {
        if (head_index >= donor.heads.len) return error.InvalidFitConfig;
        return donor.heads[head_index];
    };
    return error.InvalidFitConfig;
}

fn incrementalComparisonValid(binding: Binding) bool {
    const calibrated = binding.critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1 or binding.critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    if (binding.performance_comparison_policy == .legacy) return binding.postfit_policy != .train_global_residual_range_mass_lawson_exact_rounded_max_v1 or (calibrated and binding.train_probability_census);
    if (binding.rejected_detail_diagnostic or !binding.train_probability_census) return false;
    return switch (binding.performance_comparison_policy) {
        .legacy => unreachable,
        .owned_capture_summary_incremental_reduction_v1 => calibrated and (binding.postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 or binding.postfit_policy == .train_global_residual_range_mass_lawson_exact_rounded_max_v1),
        .owned_capture_summary_incremental_reduction_critical025_v1 => binding.critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and (binding.postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 or binding.postfit_policy == .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1),
    };
}

fn surrogatePartitionMinimaxValid(binding: Binding) bool {
    return binding.postfit_policy != .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1 or
        (binding.max_workspace_bytes == 1024 * 1024 * 1024 and binding.historical_coverage != null and
            (binding.train_exact_reservation_policy == .disabled or binding.train_exact_reservation_policy == .highest_budget_calibrated_critical_threshold_v1) and !binding.train_primary_policy_comparison and
            !binding.train_primary_policy_cross_validation and binding.critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
            binding.performance_comparison_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
            binding.train_probability_census and !binding.rejected_detail_diagnostic);
}

fn trainChallengeHistoricalValid(binding: Binding) bool {
    if (!binding.train_challenge_historical_selection)
        return !binding.hybrid_coverage_anchor and binding.historical_coverage_probability_backend == .scalar;
    if (binding.hybrid_coverage_anchor and binding.historical_coverage_probability_backend != .accelerate_fp64_gather_eligible_v1) return false;
    return binding.max_workspace_bytes == 1024 * 1024 * 1024 and
        (binding.historical_coverage != null and binding.train_exact_reservation_policy == .highest_budget_calibrated_critical_threshold_v1 and
            binding.critical_core_policy == .train_probability_gte_0_025_no_top1_v1 and
            binding.postfit_policy == .train_global_residual_range_mass_l2_exact_f32_cache_v1 and
            binding.performance_comparison_policy == .owned_capture_summary_incremental_reduction_critical025_v1 and
            binding.train_probability_census and !binding.rejected_detail_diagnostic and
            !binding.train_primary_policy_comparison and !binding.train_primary_policy_cross_validation);
}

fn referenceOwnerBytes(batches: []const engine.ReferenceBatch) !usize {
    var bytes = try std.math.mul(usize, batches.len, @sizeOf(engine.ReferenceBatch));
    for (batches) |batch| {
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, try std.math.add(usize, batch.train.len, batch.heldout.len), @sizeOf(f32)));
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, try std.math.add(usize, batch.train_positions.len, batch.heldout_positions.len), @sizeOf(u64)));
    }
    return bytes;
}

fn denseOwnerBytes(dense: augmentation.Owned) !usize {
    var bytes = dense.manifest_bytes.len;
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, dense.batches.len, @sizeOf(engine.ReferenceBatch)));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, dense.training.len, @sizeOf(f32)));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, dense.positions.len, @sizeOf(u64)));
    inline for (.{ dense.consumer_layers, dense.query_heads, dense.counts, dense.dimensions }) |slice|
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, slice.len, @sizeOf(u32)));
    return bytes;
}

fn challengedOwnerBytes(owned: challenge_rewrite.Owned) !usize {
    var bytes = try std.math.mul(usize, owned.batches.len, @sizeOf(engine.ReferenceBatch));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, owned.values.len, @sizeOf(f32)));
    return std.math.add(usize, bytes, try std.math.mul(usize, owned.positions.len, @sizeOf(u64))) catch error.CompactionWorkspaceLimit;
}

const ChallengeDenseLoaderContext = struct {
    io: std.Io,
    binding: augmentation.Binding,

    fn load(raw: ?*anyopaque, allocator: std.mem.Allocator, control: solver.Control) anyerror!augmentation.Owned {
        try control.check();
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        return augmentation.readReferenceCoverageDense512(allocator, self.io, self.binding);
    }
};

const CheckpointReloadContext = struct {
    io: std.Io,
    binding: whole.Binding,
    payload_bytes: usize,
    payload_sha256: [32]u8,
    maximum_state_live_bytes: usize,

    fn load(raw: ?*anyopaque, allocator: std.mem.Allocator, control: solver.Control) anyerror!checkpoint_phase.Reloaded {
        try control.check();
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidWholeFitInputBinding));
        const reloaded = try whole.reloadCheckpointOnly(allocator, self.io, self.binding, self.payload_bytes, self.payload_sha256, self.maximum_state_live_bytes, control);
        return .{ .checkpoint_owned = reloaded.owned, .payload_sha256 = reloaded.payload_sha256, .peak_live_bytes = reloaded.peak_live_bytes };
    }
};

const CheckpointReleaseContext = struct {
    input: *whole.OwnedInput,
    fn release(raw: ?*anyopaque) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidWholeFitInput));
        try self.input.releaseCheckpoint();
    }
};

fn valueAbPolicyValid(phase: bool, mass: bool, pilot: bool, control: bool, full: bool, challenge: bool, hybrid: bool, workers: usize) bool {
    const experiments = @as(u8, @intFromBool(mass)) + @as(u8, @intFromBool(pilot)) + @as(u8, @intFromBool(control)) + @as(u8, @intFromBool(full));
    const serial_ab = control or full;
    return (experiments != 0) == phase and experiments <= 1 and (experiments == 0 or (challenge and hybrid)) and (workers == @as(usize, if (serial_ab) 1 else 4));
}

pub const range_shared_policy_sha256: [32]u8 = .{ 0x5b, 0x33, 0x21, 0xf3, 0xdd, 0x61, 0xe1, 0x3c, 0x91, 0x58, 0x96, 0xc6, 0x37, 0xb6, 0xf3, 0x34, 0x23, 0x9c, 0x12, 0x81, 0xdc, 0x2a, 0x04, 0xf0, 0xb9, 0xf0, 0x07, 0x77, 0xec, 0x62, 0xd5, 0xfb };

fn rawSelectionTupleValid(value: RawSelectionSolver, explicit: bool, policy_sha256: ?[32]u8) bool {
    if (!explicit) return value == .householder and policy_sha256 == null;
    return switch (value) {
        .householder => policy_sha256 == null,
        .range_shared_fp64_gram_original_kkt_v1 => if (policy_sha256) |digest| std.mem.eql(u8, &digest, &range_shared_policy_sha256) else false,
        .fp64_gram_advisory => false,
    };
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !Result {
    // Validate before any read, allocation, or backend construction.
    if (!rawSelectionTupleValid(binding.raw_selection_solver, binding.raw_selection_solver_explicit, binding.raw_selection_policy_sha256)) return error.InvalidFitConfig;
    if (binding.encoding != .f16 and binding.encoding != .fp8 and binding.encoding != .rotated4) return error.InvalidFitConfig;
    if (binding.rotated4_risk_retention != .quarter_v1 and binding.encoding != .rotated4) return error.InvalidFitConfig;
    if (binding.max_workspace_bytes != 1024 * 1024 * 1024 or binding.query_reservation_bytes >= binding.max_transient_host_bytes) return error.InvalidFitConfig;
    if (!incrementalComparisonValid(binding)) return error.InvalidFitConfig;
    if (!plannedExactPolicyValid(binding)) return error.InvalidFitConfig;
    if (!trainPrimaryComparisonValid(binding)) return error.InvalidFitConfig;
    if (!trainPrimaryCrossValidationValid(binding)) return error.InvalidFitConfig;
    if (!surrogatePartitionMinimaxValid(binding)) return error.InvalidFitConfig;
    if (!trainChallengeHistoricalValid(binding)) return error.InvalidFitConfig;
    if (!valueAbPolicyValid(binding.phase_profiling, binding.train_challenge_mass_only, binding.serial_pilot_fp64_value_backend, binding.serial_full_fp64_control, binding.serial_full_fp64_value_backend, binding.train_challenge_historical_selection, binding.hybrid_coverage_anchor, binding.maximum_workers)) return error.InvalidFitConfig;
    const incremental_comparison = binding.performance_comparison_policy != .legacy;
    const private_performance_policy: engine.Options.PrivatePerformancePolicy = switch (binding.performance_comparison_policy) {
        .legacy => .legacy,
        .owned_capture_summary_incremental_reduction_v1 => .owned_capture_summary_incremental_reduction_v1,
        .owned_capture_summary_incremental_reduction_critical025_v1 => .owned_capture_summary_incremental_reduction_critical025_v1,
    };
    const calibrated = binding.critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1 or binding.critical_core_policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    if (!std.meta.eql(binding.whole_fit.identity, binding.complete_future.identity) or
        !std.mem.eql(u8, &binding.whole_fit.source_freeze_sha256, &binding.complete_future.source_freeze_sha256) or
        !std.mem.eql(u8, &binding.whole_fit.binary_sha256, &binding.complete_future.binary_sha256) or
        !std.mem.eql(u8, &binding.whole_fit.exporter_sha256, &binding.complete_future.exporter_sha256))
        return error.InvalidFutureAugmentation;
    var input = try whole.read(allocator, io, binding.whole_fit);
    defer input.deinit();
    const source_host_bytes = try input.checkpoint_owned.state.residentBytes();
    const aggregate_limit = binding.max_transient_host_bytes - binding.query_reservation_bytes;
    const worker_plan = try Fitter.parallelPlan(source_host_bytes, 8, binding.max_workspace_bytes, aggregate_limit, binding.maximum_workers);
    std.debug.print("production768_offline_plan workers={d} source_host_bytes={d} admitted_fit_host_bytes={d} head_workspace_bytes={d} query_reservation_bytes={d}\n", .{ worker_plan.workers, source_host_bytes, worker_plan.admitted_bytes, binding.max_workspace_bytes, binding.query_reservation_bytes });
    const fitters = try allocator.alloc(Fitter, worker_plan.workers);
    defer allocator.free(fitters);
    var initialized: usize = 0;
    defer for (fitters[0..initialized]) |*fitter| fitter.deinit();
    const backends = try allocator.alloc(solver.Backend, worker_plan.workers);
    defer allocator.free(backends);
    for (fitters, backends) |*fitter, *backend| {
        fitter.* = try Fitter.initWorker(allocator, null, worker_plan.admitted_bytes, binding.max_workspace_bytes);
        initialized += 1;
        try fitter.setRawSelectionSolver(binding.raw_selection_solver);
        try fitter.setQueryBatchRows(64);
        try fitter.setPhaseProfiling(binding.phase_profiling);
        backend.* = fitter.backend();
    }
    if (input.query_samples != 320 or input.method != .top1 or input.source_has_metadata or input.batches.len != 56) return error.InvalidWholeFitInput;
    var complete = try future.readComplete(allocator, io, .{
        .output_dir = binding.complete_future.output_dir,
        .manifest_sha256 = binding.complete_future.manifest_sha256,
        .primary_parent_manifest_sha256 = binding.complete_future.primary_parent_manifest_sha256,
        .identity = binding.complete_future.identity,
        .source_freeze_sha256 = binding.complete_future.source_freeze_sha256,
        .binary_sha256 = binding.complete_future.binary_sha256,
        .exporter_sha256 = binding.complete_future.exporter_sha256,
        .authority = binding.complete_future.authority,
        .historical_study_tokens_sha256 = binding.complete_future.historical_study_tokens_sha256,
        .primary_batches = input.batches,
    });
    defer complete.deinit();
    var derived_historical: ?historical_coverage.Owned = null;
    var historical_coverage_wall_ns: u64 = 0;
    var serial_pilot_rewrite_wall_ns: u64 = 0;
    var pilot_timing: challenge_rewrite.PilotTiming = .{};
    var serial_value_context: fp64_value.Context = .{};
    defer if (derived_historical) |*owned| owned.deinit();
    var challenged_historical: ?challenge_rewrite.Owned = null;
    var train_challenge_mass_only_calls: usize = 0;
    var train_challenge_mass_only_gemm_calls: usize = 0;
    var train_challenge_validation_summary_calls: usize = 0;
    defer if (challenged_historical) |*owned| owned.deinit();
    if (binding.historical_coverage) |coverage| {
        const coverage_started = std.Io.Clock.awake.now(io).nanoseconds;
        if (!std.meta.eql(coverage.dense.identity, binding.whole_fit.identity) or
            !std.mem.eql(u8, &coverage.dense.parent_manifest_sha256, &binding.complete_future.primary_parent_manifest_sha256) or
            !std.mem.eql(u8, &coverage.dense.source_freeze_sha256, &binding.whole_fit.source_freeze_sha256) or
            !std.mem.eql(u8, &coverage.dense.binary_sha256, &binding.whole_fit.binary_sha256) or
            !std.mem.eql(u8, &coverage.dense.exporter_sha256, &binding.whole_fit.exporter_sha256) or
            coverage.dense.samples_per_stratum != 512 or coverage.maximum_workspace_bytes == 0 or coverage.maximum_workspace_bytes > binding.max_workspace_bytes)
            return error.InvalidTrainingAugmentationBinding;
        // The outer sidecar's retained-primary manifest is a reference-coverage
        // authority only. Dense query bytes come exclusively from the current
        // nested artifact, whose parent is the current whole-fit capture.
        if (!std.meta.eql(coverage.dense_outer.identity, binding.whole_fit.identity) or
            !std.mem.eql(u8, &coverage.dense_outer.nested_manifest_sha256, &coverage.dense_nested_manifest_sha256) or
            !std.mem.eql(u8, &coverage.dense_outer.source_freeze_sha256, &binding.whole_fit.source_freeze_sha256) or
            !std.mem.eql(u8, &coverage.dense_outer.binary_sha256, &binding.whole_fit.binary_sha256) or
            !std.mem.eql(u8, &coverage.dense_outer.exporter_sha256, &binding.whole_fit.exporter_sha256))
            return error.InvalidTrainingAugmentationBinding;
        try historical_coverage.authenticateOuter(allocator, io, coverage.dense_outer);
        var dense = try augmentation.readReferenceCoverageDense512(allocator, io, coverage.dense);
        var dense_live = true;
        defer if (dense_live) dense.deinit();
        var dense_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &dense_digest, .{});
        if (!std.mem.eql(u8, &dense_digest, &coverage.dense_nested_manifest_sha256)) return error.InvalidTrainingAugmentationBinding;
        std.debug.print("production768_historical_coverage phase=begin train_only=true maximum_workspace_bytes={d} diagnostic_only=true\n", .{coverage.maximum_workspace_bytes});
        var probability_context: probability_accelerate.Context = .{};
        derived_historical = switch (binding.historical_coverage_probability_backend) {
            .scalar => try historical_coverage.derive(allocator, input.checkpoint_owned.state, input.layout, input.batches, complete.future.batches, &dense, coverage.maximum_workspace_bytes, binding.control),
            .accelerate_fp64_gather_eligible_v1 => try historical_coverage.deriveWithProbabilityBackend(allocator, input.checkpoint_owned.state, input.layout, input.batches, complete.future.batches, &dense, coverage.maximum_workspace_bytes, binding.control, .{ .backend = probability_accelerate.backend(&probability_context), .gather_eligible_historical = true }),
        };
        const evidence = derived_historical.?.evidence;
        const coverage_elapsed = std.Io.Clock.awake.now(io).nanoseconds - coverage_started;
        if (coverage_elapsed < 0 or coverage_elapsed > std.math.maxInt(u64)) return error.InvalidFitConfig;
        historical_coverage_wall_ns = @intCast(coverage_elapsed);
        std.debug.print("production768_historical_coverage phase=complete batch_count={d} eligible_rows={d} excluded_rows={d} coverage_selected_rows={d} kcenter_selected_rows={d} newly_covered_key_count={d} selection_digest={s} historical_digest={s} coverage_authority_sha256={s} diagnostic_only=true\n", .{ evidence.batch_count, evidence.eligible_rows, evidence.excluded_rows, evidence.coverage_selected_rows, evidence.kcenter_selected_rows, evidence.newly_covered_key_count, std.fmt.bytesToHex(evidence.selection_digest, .lower), std.fmt.bytesToHex(evidence.derived_historical_sha256, .lower), std.fmt.bytesToHex(evidence.coverage_authority_sha256, .lower) });
        if (binding.train_challenge_historical_selection) {
            if (input.batches.len != 56 or derived_historical.?.batches.len != 56 or complete.future.batches.len != 56 or dense.batches.len != 56 or dense.counts.len != 56) return error.InvalidTrainingAugmentation;
            const obsolete_historical_bytes = try denseOwnerBytes(complete.historical);
            try complete.releaseHistorical();
            std.debug.print("production768_train_challenge phase=release_obsolete_complete_historical released_bytes={d} replacement_historical_digest={s} source_authority_unchanged=true diagnostic_only=true\n", .{ obsolete_historical_bytes, std.fmt.bytesToHex(evidence.derived_historical_sha256, .lower) });
            var dense_metadata = try challenge_rewrite.freezeDenseMetadata(allocator, &dense, coverage.maximum_workspace_bytes);
            defer dense_metadata.deinit();
            const dense_payload_bytes = try denseOwnerBytes(dense);
            dense.deinit();
            dense_live = false;
            var dense_loader_context: ChallengeDenseLoaderContext = .{ .io = io, .binding = coverage.dense };
            // readReferenceCoverageDense512 authenticates a 128 MiB bundle;
            // reserve one additional manifest-sized transient for its parsed
            // metadata before invoking the allocating callback.
            const dense_loader_maximum_live = try std.math.add(usize, coverage.maximum_workspace_bytes, 64 * 1024);
            const dense_loader: challenge_rewrite.DenseLoader = .{ .context = &dense_loader_context, .load_fn = ChallengeDenseLoaderContext.load, .maximum_live_bytes = dense_loader_maximum_live };
            std.debug.print("production768_train_challenge phase=release_initial_dense512 released_bytes={d} retained_metadata_bytes={d} reload_manifest_sha256={s} diagnostic_only=true\n", .{ dense_payload_bytes, try dense_metadata.liveBytes(), std.fmt.bytesToHex(dense_metadata.manifest_sha256, .lower) });
            var independent_challenge_context: @import("../../ops/metal_attention_match_validation.zig").Context = .{};
            const independent_challenge_backend = @import("../../ops/metal_attention_match_validation.zig").backend(&independent_challenge_context);
            const independent_mass_backend = @import("../../ops/metal_attention_match_validation.zig").massBackend(&independent_challenge_context);
            var primary_views: [56]challenge_rewrite.TrainingBatchView = undefined;
            var historical_views: [56]challenge_rewrite.TrainingBatchView = undefined;
            var future_views: [56]challenge_rewrite.TrainingBatchView = undefined;
            for (input.batches, derived_historical.?.batches, complete.future.batches, &primary_views, &historical_views, &future_views) |p, h, f, *pv, *hv, *fv| {
                pv.* = .{ .consumer_layer = p.consumer_layer, .query_head = p.query_head, .train = p.train, .train_positions = p.train_positions, .heldout_positions = p.heldout_positions };
                hv.* = .{ .consumer_layer = h.consumer_layer, .query_head = h.query_head, .train = h.train, .train_positions = h.train_positions, .heldout_positions = &.{} };
                fv.* = .{ .consumer_layer = f.consumer_layer, .query_head = f.query_head, .train = f.train, .train_positions = f.train_positions, .heldout_positions = &.{} };
            }
            var phase_authority = try checkpoint_phase.freezeAuthority(allocator, input.checkpoint_owned.state, input.layout, input.batches, input.checkpoint_payload_sha256, binding.control);
            defer phase_authority.deinit();
            var reload_context: CheckpointReloadContext = .{ .io = io, .binding = binding.whole_fit, .payload_bytes = input.checkpoint_payload_bytes, .payload_sha256 = input.checkpoint_payload_sha256, .maximum_state_live_bytes = source_host_bytes };
            var release_context: CheckpointReleaseContext = .{ .input = &input };
            const phase_checkpoint_loader: checkpoint_phase.Loader = .{ .context = &reload_context, .reload_fn = CheckpointReloadContext.load, .maximum_reload_live_bytes = source_host_bytes };
            // The full checkpoint is still resident until rewrite completes all
            // authority checks and invokes the one-shot release callback. This
            // ledger intentionally excludes it only for the later pilot phase.
            const input_live = try input.liveBytes();
            if (input_live < source_host_bytes) return error.InvalidWholeFitInput;
            var external_live: usize = input_live - source_host_bytes;
            external_live = try std.math.add(usize, external_live, try denseOwnerBytes(complete.future));
            // Includes H256 plus the per-consumer coverage CSR retained until
            // the hybrid selector has frozen every replacement batch.
            external_live = try std.math.add(usize, external_live, try derived_historical.?.liveBytes());
            external_live = try std.math.add(usize, external_live, try dense_metadata.liveBytes());
            external_live = try std.math.add(usize, external_live, @sizeOf(@TypeOf(primary_views)) + @sizeOf(@TypeOf(historical_views)) + @sizeOf(@TypeOf(future_views)));
            external_live = try std.math.add(usize, external_live, try phase_authority.liveBytes());
            const prerelease_external = try std.math.add(usize, external_live, phase_authority.resident_bytes);
            const rewrite_started = std.Io.Clock.awake.now(io).nanoseconds;
            challenged_historical = challenge_rewrite.rewrite(allocator, input.checkpoint_owned.state, input.layout, &primary_views, &historical_views, &future_views, dense_metadata, dense_loader, .{ .authority = phase_authority, .loader = phase_checkpoint_loader, .release_context = &release_context, .release_fn = CheckpointReleaseContext.release, .external_without_checkpoint = external_live, .prerelease_external_live_bytes = prerelease_external }, .{
                .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .max_relative_mass_error = input.fit.max_relative_mass_error, .max_relative_output_rmse = input.fit.max_relative_output_rmse, .max_sweeps = input.fit.max_sweeps, .nnls_tolerance = input.fit.nnls_tolerance, .min_weight = input.fit.min_weight, .max_workspace_bytes = binding.max_workspace_bytes, .backend = backends[0], .value_backend = if (binding.serial_pilot_fp64_value_backend or binding.serial_full_fp64_value_backend) serial_value_context.valueBackend() else null, .control = binding.control },
                .backend = independent_challenge_backend,
                .combined_cap = binding.max_workspace_bytes,
                .external_live_bytes = external_live,
                .postfit_workspace_bytes = try replay.postfitWorkspaceReserve(15360, 7936, 7936, 23),
                .source_freeze_sha256 = binding.whole_fit.source_freeze_sha256,
                .dense_manifest_sha256 = coverage.dense_nested_manifest_sha256,
                .initial_historical_sha256 = evidence.derived_historical_sha256,
                .skip_donor_layer = evidence.chosen_exact_donor_layer,
                .skip_head_index = evidence.chosen_exact_head_index,
                .hybrid_coverage_anchor = binding.hybrid_coverage_anchor,
                .coverage_offsets = if (binding.hybrid_coverage_anchor) derived_historical.?.coverage_offsets else &.{},
                .coverage_indices = if (binding.hybrid_coverage_anchor) derived_historical.?.coverage_indices else &.{},
                .coverage_authority_sha256 = if (binding.hybrid_coverage_anchor) evidence.coverage_authority_sha256 else @splat(0),
                .mass_backend = if (binding.train_challenge_mass_only) independent_mass_backend else null,
                .pilot_timing = if (binding.serial_pilot_fp64_value_backend or binding.serial_full_fp64_value_backend) &pilot_timing else null,
            }) catch |err| {
                std.debug.print("production768_train_challenge phase=rewrite_failed error={s} mass_only_calls={d} mass_only_gemm_calls={d} validation_summary_calls={d} diagnostic_only=true\n", .{ @errorName(err), independent_challenge_context.successful_mass_only_calls, independent_challenge_context.mass_only_gemm_calls, independent_challenge_context.successful_calls });
                return err;
            };
            const rewrite_elapsed = std.Io.Clock.awake.now(io).nanoseconds - rewrite_started;
            if (rewrite_elapsed < 0 or rewrite_elapsed > std.math.maxInt(u64)) return error.InvalidFitConfig;
            serial_pilot_rewrite_wall_ns = @intCast(rewrite_elapsed);
            if ((binding.serial_pilot_fp64_value_backend or binding.serial_full_fp64_value_backend) and (pilot_timing.count != 7 or serial_value_context.stats.calls == 0 or serial_value_context.stats.chosen_elements_max == 0 or serial_value_context.stats.chosen_elements_max > serial_value_context.stats.declared_elements_max or serial_value_context.stats.workspace_peak_bytes > binding.max_workspace_bytes)) return error.InvalidTrainingAugmentation;
            train_challenge_mass_only_calls = independent_challenge_context.successful_mass_only_calls;
            train_challenge_mass_only_gemm_calls = independent_challenge_context.mass_only_gemm_calls;
            train_challenge_validation_summary_calls = independent_challenge_context.successful_calls;
            std.debug.print("production768_train_challenge phase=rewrite_complete mass_only_calls={d} mass_only_gemm_calls={d} validation_summary_calls={d} diagnostic_only=true\n", .{ train_challenge_mass_only_calls, train_challenge_mass_only_gemm_calls, train_challenge_validation_summary_calls });
            if (binding.train_challenge_mass_only and train_challenge_mass_only_calls != 104) return error.InvalidMassOnlyCallCount;
            if (binding.train_challenge_mass_only and train_challenge_mass_only_gemm_calls != 104) return error.InvalidMassOnlyGemmCount;
            // Root1346's unchanged path measured 416 calls. Mass-only replaces
            // exactly two selector summaries for each of 52 rewritten batches;
            // the remaining 312 calls are the unchanged seven-pilot validation.
            if (binding.train_challenge_mass_only and train_challenge_validation_summary_calls != 312) return error.InvalidMassOnlyValidationSummaryCount;
            if (input.checkpoint_live) return error.InvalidWholeFitInput;
            var final_external = external_live;
            final_external = try std.math.add(usize, final_external, try challengedOwnerBytes(challenged_historical.?));
            var restored = try checkpoint_phase.loadFinal(allocator, phase_authority, phase_checkpoint_loader, final_external, binding.max_workspace_bytes, binding.control);
            var restored_live = true;
            errdefer if (restored_live) restored.deinit();
            try input.installCheckpoint(restored, input.checkpoint_payload_sha256);
            restored_live = false;
            const challenged = challenged_historical.?.evidence;
            std.debug.print("production768_train_challenge phase=frozen pilot_heads={d} skipped_exact_heads={d} rewritten_batches={d} max_train_challenge_error={d} mean_train_challenge_error={d} selection_digest={s} hybrid_coverage_anchor={} hybrid_mandatory_count={d} hybrid_coverage_count={d} hybrid_anchor_count={d} hybrid_challenge_count={d} coverage_authority_sha256={s} hybrid_authority_sha256={s} frozen_before_heldout={} heldout_value_observations={d} diagnostic_only=true\n", .{ challenged.pilot_head_count, challenged.skipped_exact_head_count, challenged.rewritten_batch_count, challenged.maximum_train_challenge_error, challenged.mean_train_challenge_error, std.fmt.bytesToHex(challenged.selection_digest, .lower), binding.hybrid_coverage_anchor, challenged.hybrid_mandatory_count, challenged.hybrid_coverage_count, challenged.hybrid_anchor_count, challenged.hybrid_challenge_count, std.fmt.bytesToHex(challenged.coverage_authority_sha256, .lower), std.fmt.bytesToHex(challenged.hybrid_authority_sha256, .lower), challenged.frozen_before_heldout, challenged.heldout_value_observations });
        }
    }
    const historical_batches = if (challenged_historical) |*owned| owned.batches else if (derived_historical) |*owned| owned.batches else complete.historical.batches;
    var merged = try mergeComplete(allocator, input.batches, historical_batches, complete.future.batches, binding.control);
    defer merged.deinit();
    // Capture metadata records the legacy export constructor (1024). The
    // authenticated private runtime policy remains separately fixed at 256.
    if (input.fit.prefix_rows != 128 or input.fit.recent_rows != 2048 or input.fit.chunk_rows != 1024 or input.fit.retention_divisor != 4 or input.fit.min_retention_divisor != 2 or
        input.fit.max_relative_mass_error != 0.05 or input.fit.max_relative_output_rmse != 0.05 or input.fit.selection != .residual_omp or input.fit.mass_solver != .active_set or input.fit.validation != .whole_head or input.fit.rejected_head_policy != .preserve_original or input.fit.storage_precision != .f16 or !input.fit.independent_fp64_publication)
        return error.InvalidWholeFitInput;

    var stats: engine.Stats = .{};
    var options: engine.Options = .{
        .admission = .cumulative_ladder_4_3_2,
        .fit_strategy = .identity_chunk_coefficients_adaptive_v18,
        .staged_operator_strategy = true,
        .adaptive_v18_fn = adaptive.engineCallback,
        .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1,
        .adaptive_v18_geometry_route = if (incremental_comparison or calibrated) .flat_gram_only else .bounded_page_over_96,
        .adaptive_v18_critical_core_policy = binding.critical_core_policy,
        .adaptive_v18_postfit_policy = binding.postfit_policy,
        .adaptive_v18_rejected_detail_diagnostic = binding.rejected_detail_diagnostic,
        .adaptive_v18_train_probability_census = binding.train_probability_census,
        .adaptive_v18_train_primary_policy_comparison = binding.train_primary_policy_comparison,
        .adaptive_v18_train_primary_policy_cross_validation = binding.train_primary_policy_cross_validation,
        .geometry_chooser_policy = if (incremental_comparison) .incremental_gram_v1 else .legacy_full_scan,
        .independent_validation_policy = if (incremental_comparison) .accelerate_fp64_summary_v1 else .scalar_reference,
        .private_performance_policy = private_performance_policy,
        .adaptive_v18_bounded_tile_queries = 64,
        .adaptive_v18_bounded_builder_transient_bytes = 256 * 1024 * 1024,
        .adaptive_v18_bounded_delta_accelerated = true,
        .adaptive_v18_bounded_tsqr_value_backend = false,
        .stats = &stats,
        .retention_divisor = 4,
        .min_retention_divisor = 2,
        .rejected_head_policy = .preserve_original,
        .prefix_rows = input.fit.prefix_rows,
        .recent_rows = input.fit.recent_rows,
        .chunk_rows = 256,
        .max_queries_per_consumer = 768,
        .validation = .whole_head,
        .independent_fp64_publication = true,
        .storage_precision = .f16,
        .fp8_checkpoint = binding.encoding == .fp8,
        .rotated4_checkpoint = binding.encoding == .rotated4,
        .rotated4_risk_retention = binding.rotated4_risk_retention,
        .max_parallel_validations = 1,
        .max_transient_host_bytes = worker_plan.admitted_bytes,
        .min_global_savings_ratio = 1.5,
        .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .max_relative_mass_error = input.fit.max_relative_mass_error, .max_relative_output_rmse = input.fit.max_relative_output_rmse, .max_sweeps = input.fit.max_sweeps, .nnls_tolerance = input.fit.nnls_tolerance, .min_weight = input.fit.min_weight, .max_workspace_bytes = binding.max_workspace_bytes, .backend = backends[0], .control = binding.control },
    };
    if (binding.serial_full_fp64_value_backend) options.fit.value_backend = serial_value_context.valueBackend();
    const shape = try eligibleShape(input.checkpoint_owned.state, options);
    const budget = try physical.deriveUniform(shape.rows, shape.dim, shape.heads, options.prefix_rows + options.recent_rows, 3);
    if (budget.primary_percent != 91 or budget.fallback_rows != 4912 or budget.primary_rows != 4796) return error.InvalidFitConfig;
    const planned_evidence = if (binding.train_exact_reservation_policy == .highest_budget_calibrated_critical_threshold_v1) blk: {
        const evidence = if (derived_historical) |owned| owned.evidence else return error.InvalidFitConfig;
        if (!std.mem.eql(u8, &evidence.exact_head_policy_sha256, &exact_reservation.policy_sha256)) return error.InvalidFitConfig;
        break :blk PlannedExactEvidence{
            .policy = binding.train_exact_reservation_policy,
            .policy_sha256 = evidence.exact_head_policy_sha256,
            .evidence_sha256 = evidence.exact_head_evidence_sha256,
            .donor_layer = evidence.chosen_exact_donor_layer,
            .head_index = evidence.chosen_exact_head_index,
            .frozen_before_heldout = true,
        };
    } else PlannedExactEvidence{};
    if (planned_evidence.policy != .disabled) std.debug.print("production768_planned_exact phase=frozen donor={d} head={d} policy_sha256={s} evidence_sha256={s} frozen_before_heldout=true maximum_p95_attempts=3 diagnostic_only=true\n", .{ planned_evidence.donor_layer, planned_evidence.head_index, std.fmt.bytesToHex(planned_evidence.policy_sha256, .lower), std.fmt.bytesToHex(planned_evidence.evidence_sha256, .lower) });
    var budget_context: adaptive.PhysicalBudgetContext = .{};
    var planned_context: planned_exact.Context = .{
        .enabled = planned_evidence.policy != .disabled,
        .donor_layer = planned_evidence.donor_layer,
        .head_index = planned_evidence.head_index,
    };
    options.adaptive_v18_primary_budget_percent = budget.primary_percent;
    options.adaptive_v18_fallback_budget_percent = 95;
    options.adaptive_v18_max_budget_fallbacks = 3;
    if (planned_context.enabled) {
        options.adaptive_v18_fn = planned_exact.Context.callback;
        options.adaptive_v18_context = &planned_context;
    } else options.adaptive_v18_context = &budget_context;
    const final_engine_started = std.Io.Clock.awake.now(io).nanoseconds;
    var candidate = if (backends.len == 1)
        try engine.prepare(allocator, input.checkpoint_owned.state, merged.batches, input.layout, options)
    else
        try engine.prepareParallel(input.checkpoint_owned.state, merged.batches, input.layout, options, backends);
    const final_engine_elapsed = std.Io.Clock.awake.now(io).nanoseconds - final_engine_started;
    if (final_engine_elapsed < 0 or final_engine_elapsed > std.math.maxInt(u64)) return error.InvalidFitConfig;
    const final_engine_wall_ns: u64 = @intCast(final_engine_elapsed);
    errdefer candidate.deinit();
    const source_bytes = try metal_storage.ordinaryGlobalAllocationBytesWithControl(input.checkpoint_owned.state, checkpoint.serving_limits, binding.control);
    const candidate_bytes = try metal_storage.globalAllocationBytesWithControl(candidate.state, checkpoint.serving_limits, binding.control);
    if (candidate_bytes == 0) return error.InvalidFitConfig;
    const ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(candidate_bytes));
    var exact_evidence = planned_evidence;
    if (planned_context.enabled) {
        exact_evidence.adaptive_fit_attempts = planned_context.adaptive_fit_attempts.load(.acquire);
        const source_head = try findHead(input.checkpoint_owned.state, exact_evidence.donor_layer, exact_evidence.head_index);
        const candidate_head = try findHead(candidate.state, exact_evidence.donor_layer, exact_evidence.head_index);
        if (!sameExactHead(source_head, candidate_head)) return error.InvalidFitConfig;
        exact_evidence.exact_clone_rows = source_head.positions.len;
    }
    const expected_probe_heads: usize = if (planned_context.enabled) 7 else 8;
    const probe_evidence_valid = !binding.train_primary_policy_comparison or
        (stats.v18_train_primary_probe_heads == expected_probe_heads and
            stats.v18_train_primary_probe_candidates == expected_probe_heads * 2 and
            stats.v18_train_primary_probe_recomputed_heads == expected_probe_heads);
    const codec_evidence_valid = codecEvidenceValid(binding.encoding, stats, if (planned_context.enabled) 7 else 8);
    const accepted = probe_evidence_valid and codec_evidence_valid and if (planned_context.enabled)
        stats.v18_eligible_heads == 8 and stats.v18_attempted_heads == 8 and stats.v18_accepted_heads == 7 and stats.v18_preserved_heads == 0 and exact_evidence.adaptive_fit_attempts == 7 and stats.v18_budget_fallback_heads <= 3 and ratio >= 1.5
    else
        stats.v18_eligible_heads == 8 and stats.v18_attempted_heads == 8 and stats.v18_accepted_heads >= 7 and stats.v18_preserved_heads <= 1 and stats.v18_budget_fallback_heads <= 3 and ratio >= 1.5;
    var fitter_phase_counters: [4]?Fitter.PhaseCounters = @splat(null);
    for (fitters, 0..) |*fitter, i| fitter_phase_counters[i] = fitter.phaseCounters();
    return .{ .candidate = candidate, .serial_pilot_observer_timing = pilot_timing, .stats = stats, .source_physical_bytes = source_bytes, .candidate_physical_bytes = candidate_bytes, .physical_ratio = ratio, .primary_budget_percent = budget.primary_percent, .fallback_budget_percent = 95, .maximum_fallbacks = 3, .workers = worker_plan.workers, .source_host_bytes = source_host_bytes, .admitted_fit_host_bytes = worker_plan.admitted_bytes, .historical_coverage_evidence = if (derived_historical) |owned| owned.evidence else null, .train_challenge_evidence = if (challenged_historical) |owned| owned.evidence else null, .planned_exact = exact_evidence, .accepted = accepted, .train_challenge_mass_only_calls = train_challenge_mass_only_calls, .train_challenge_mass_only_gemm_calls = train_challenge_mass_only_gemm_calls, .train_challenge_validation_summary_calls = train_challenge_validation_summary_calls, .fitter_phase_counters = fitter_phase_counters, .serial_pilot_fp64_value_stats = serial_value_context.stats, .historical_coverage_wall_ns = historical_coverage_wall_ns, .serial_pilot_rewrite_wall_ns = serial_pilot_rewrite_wall_ns, .serial_pilot_wall_ns = pilot_timing.elapsed_ns, .serial_pilot_input_extract_ns = pilot_timing.input_extract_ns, .serial_pilot_query_frontend_ns = pilot_timing.query_frontend_ns, .serial_pilot_selection_fit_ns = pilot_timing.selection_fit_ns, .serial_pilot_value_fit_ns = pilot_timing.value_fit_ns, .serial_pilot_validation_and_postfit_ns = pilot_timing.validation_and_postfit_ns, .serial_pilot_reload_ns = pilot_timing.reload_ns, .serial_pilot_encode_rewrite_ns = pilot_timing.encode_rewrite_ns, .final_engine_wall_ns = final_engine_wall_ns };
}

test "production768 V16 forwards sealed raw selector before backend publication" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    var fitter = try Fitter.initWorker(std.testing.allocator, null, 64 * 1024 * 1024, 64 * 1024 * 1024);
    defer fitter.deinit();
    try fitter.setRawSelectionSolver(.range_shared_fp64_gram_original_kkt_v1);
    try fitter.setQueryBatchRows(64);
    try fitter.setPhaseProfiling(true);
    const backend = fitter.backend();
    try std.testing.expectEqual(Fitter.RawSelectionSolver.range_shared_fp64_gram_original_kkt_v1, fitter.rawSelectionSolver());
    try std.testing.expectEqual(solver.RawRangePolicy.range_shared_fp64_gram_original_kkt_v1, backend.raw_range_policy);
    try std.testing.expectEqual(Fitter.RawSelectionSolver.range_shared_fp64_gram_original_kkt_v1, fitter.phaseCounters().raw_selection_solver);
}

test "production768 V16 rejects diagnostic Gram selector" {
    try std.testing.expect(rawSelectionTupleValid(.householder, false, null));
    try std.testing.expect(rawSelectionTupleValid(.householder, true, null));
    try std.testing.expect(rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, true, range_shared_policy_sha256));
    try std.testing.expect(!rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, false, range_shared_policy_sha256));
    try std.testing.expect(!rawSelectionTupleValid(.range_shared_fp64_gram_original_kkt_v1, true, null));
    try std.testing.expect(!rawSelectionTupleValid(.fp64_gram_advisory, true, null));
}

test "production768 offline policy is complete-future all-head and heldout cannot select assignment" {
    var options: engine.Options = .{ .fit_strategy = .identity_chunk_coefficients_adaptive_v18, .staged_operator_strategy = true, .adaptive_v18_numerical_policy = .accepted_f32_gram_reuse_v1, .adaptive_v18_critical_core_policy = .train_probability_gte_0_025_no_top1_v1, .max_queries_per_consumer = 768, .validation = .whole_head, .independent_fp64_publication = true, .storage_precision = .f16 };
    try std.testing.expect(engine.criticalCorePolicyValid(options));
    options.max_queries_per_consumer = 960;
    try std.testing.expect(!engine.criticalCorePolicyValid(options));
    const budget = try physical.deriveUniform(7936, 512, 8, 2176, 3);
    try std.testing.expectEqual(@as(u8, 91), budget.primary_percent);
    try std.testing.expectEqual(@as(usize, 4796), budget.primary_rows);
    try std.testing.expectEqual(@as(usize, 4912), budget.fallback_rows);
}

test "full Accelerate value A/B binding is default-off and serial-only" {
    try std.testing.expect(valueAbPolicyValid(false, false, false, false, false, true, true, 4));
    try std.testing.expect(valueAbPolicyValid(true, false, false, true, false, true, true, 1));
    try std.testing.expect(valueAbPolicyValid(true, false, false, false, true, true, true, 1));
    try std.testing.expect(!valueAbPolicyValid(true, false, false, true, true, true, true, 1));
    try std.testing.expect(!valueAbPolicyValid(true, false, false, false, true, true, true, 4));
    try std.testing.expect(!valueAbPolicyValid(false, false, false, false, true, true, true, 1));
}

test "serial A/B aggregate admission produces one actual worker for both arms" {
    const source_host_bytes: usize = 304_758_376;
    const workspace: usize = 1024 * 1024 * 1024;
    const aggregate_limit: usize = 3840 * 1024 * 1024 - 528_492_032;
    inline for (.{ .{ true, false }, .{ false, true } }) |arms| {
        try std.testing.expect(valueAbPolicyValid(true, false, false, arms[0], arms[1], true, true, 1));
        const plan = try Fitter.parallelPlan(source_host_bytes, 8, workspace, aggregate_limit, 1);
        try std.testing.expectEqual(@as(usize, 1), plan.workers);
    }
}

test "codec diagnostic acceptance requires an observed nonempty codec route" {
    var stats: engine.Stats = .{};
    try std.testing.expect(codecEvidenceValid(.f16, stats, 7));
    try std.testing.expect(!codecEvidenceValid(.fp8, stats, 7));
    stats.fp8_heads = 1;
    stats.fp8_fallback_heads = 6;
    stats.fp8_eligible_rows = 100;
    stats.fp8_encoded_rows = 40;
    stats.fp8_exact_rows = 60;
    try std.testing.expect(codecEvidenceValid(.fp8, stats, 7));
    stats.fp8_exact_rows = 59;
    try std.testing.expect(!codecEvidenceValid(.fp8, stats, 7));
    stats = .{};
    stats.rotated4_heads = 2;
    stats.rotated4_fallback_heads = 5;
    stats.rotated4_eligible_rows = 80;
    stats.rotated4_compressed_rows = 64;
    stats.rotated4_exact_rows = 16;
    try std.testing.expect(codecEvidenceValid(.rotated4, stats, 7));
}

test "production768 incremental reduction comparison is diagnostic and exact tuple only" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_v1;
    binding.critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.rejected_detail_diagnostic = false;
    binding.train_probability_census = true;
    try std.testing.expect(incrementalComparisonValid(binding));
    binding.rejected_detail_diagnostic = true;
    try std.testing.expect(!incrementalComparisonValid(binding));
    binding.rejected_detail_diagnostic = false;
    binding.postfit_policy = .disabled;
    try std.testing.expect(!incrementalComparisonValid(binding));
    binding.performance_comparison_policy = .legacy;
    try std.testing.expect(incrementalComparisonValid(binding));
    binding.postfit_policy = .train_global_residual_range_mass_lawson_exact_rounded_max_v1;
    try std.testing.expect(incrementalComparisonValid(binding));
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    try std.testing.expect(!incrementalComparisonValid(binding));
    binding.critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_v1;
    binding.postfit_policy = .train_global_residual_range_mass_lawson_exact_rounded_max_v1;
    try std.testing.expect(incrementalComparisonValid(binding));
}

test "production768 planned exact reservation is an explicit historical-coverage tuple" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.train_primary_policy_comparison = false;
    binding.train_exact_reservation_policy = .disabled;
    try std.testing.expect(plannedExactPolicyValid(binding));
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    binding.historical_coverage = null;
    try std.testing.expect(!plannedExactPolicyValid(binding));
    binding.historical_coverage = .{ .dense = undefined, .dense_outer = undefined, .dense_nested_manifest_sha256 = undefined };
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_v1;
    binding.critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.rejected_detail_diagnostic = false;
    binding.train_probability_census = true;
    try std.testing.expect(plannedExactPolicyValid(binding));
    binding.performance_comparison_policy = .legacy;
    try std.testing.expect(!plannedExactPolicyValid(binding));
}

fn plannedExactAllocationFailure(allocator: std.mem.Allocator) !void {
    const source: checkpoint.Head = .{ .dim = 1, .compacted_until = 2, .positions = &.{ 0, 1 }, .coverage_ends = &.{ 1, 2 }, .keys = &.{ @as(f32, @bitCast(@as(u32, 0x80000000))), 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 } };
    const cloned = try planned_exact.cloneHead(allocator, source, .{});
    defer planned_exact.freeHead(allocator, cloned);
    try std.testing.expect(sameExactHead(source, cloned));
    @constCast(cloned.keys)[0] = 0;
    try std.testing.expect(!sameExactHead(source, cloned));
}

test "production768 planned exact clone is atomic under allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, plannedExactAllocationFailure, .{});
}

test "production768 critical025 optimized comparison is exact and diagnostic only" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.train_probability_census = true;
    binding.rejected_detail_diagnostic = false;
    try std.testing.expect(incrementalComparisonValid(binding));
    binding.critical_core_policy = .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2;
    try std.testing.expect(!incrementalComparisonValid(binding));
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.rejected_detail_diagnostic = true;
    try std.testing.expect(!incrementalComparisonValid(binding));
}

test "production768 TRAIN comparison is optional and preserves planned exact strict tuples" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.train_primary_policy_comparison = false;
    try std.testing.expect(trainPrimaryComparisonValid(binding));
    binding.train_primary_policy_comparison = true;
    binding.historical_coverage = null;
    try std.testing.expect(!trainPrimaryComparisonValid(binding));
    binding.historical_coverage = .{ .dense = undefined, .dense_outer = undefined, .dense_nested_manifest_sha256 = undefined };
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.train_probability_census = true;
    binding.rejected_detail_diagnostic = false;
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    try std.testing.expect(trainPrimaryComparisonValid(binding) and plannedExactPolicyValid(binding));
    binding.rejected_detail_diagnostic = true;
    try std.testing.expect(!trainPrimaryComparisonValid(binding));
}

test "production768 TRAIN primary CV requires the sealed comparison tuple" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.train_primary_policy_cross_validation = false;
    try std.testing.expect(trainPrimaryCrossValidationValid(binding));
    binding.train_primary_policy_cross_validation = true;
    binding.train_primary_policy_comparison = false;
    try std.testing.expect(!trainPrimaryCrossValidationValid(binding));
    binding.train_primary_policy_comparison = true;
    binding.historical_coverage = .{ .dense = undefined, .dense_outer = undefined, .dense_nested_manifest_sha256 = undefined };
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.train_probability_census = true;
    binding.rejected_detail_diagnostic = false;
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    try std.testing.expect(trainPrimaryCrossValidationValid(binding));
    binding.train_exact_reservation_policy = .disabled;
    try std.testing.expect(!trainPrimaryCrossValidationValid(binding));
}

test "production768 surrogate minimax requires sealed original critical tuple" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    binding.postfit_policy = .train_global_residual_range_mass_l2_then_surrogate_partition_minimax_exact_rounded_v1;
    binding.max_workspace_bytes = 1024 * 1024 * 1024;
    binding.historical_coverage = .{ .dense = undefined, .dense_outer = undefined, .dense_nested_manifest_sha256 = undefined };
    binding.train_exact_reservation_policy = .disabled;
    binding.train_primary_policy_comparison = false;
    binding.train_primary_policy_cross_validation = false;
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.train_probability_census = true;
    binding.rejected_detail_diagnostic = false;
    try std.testing.expect(surrogatePartitionMinimaxValid(binding));
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    try std.testing.expect(surrogatePartitionMinimaxValid(binding));
    try std.testing.expect(plannedExactPolicyValid(binding));
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_v1;
    try std.testing.expect(!plannedExactPolicyValid(binding));
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.train_primary_policy_cross_validation = true;
    try std.testing.expect(!surrogatePartitionMinimaxValid(binding));
}

test "production768 TRAIN challenge requires planned exact critical025 L2" {
    var binding: Binding = undefined;
    binding.historical_coverage_probability_backend = .scalar;
    binding.train_challenge_historical_selection = true;
    binding.hybrid_coverage_anchor = false;
    binding.max_workspace_bytes = 1024 * 1024 * 1024;
    binding.historical_coverage = .{ .dense = undefined, .dense_outer = undefined, .dense_nested_manifest_sha256 = undefined };
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    binding.critical_core_policy = .train_probability_gte_0_025_no_top1_v1;
    binding.postfit_policy = .train_global_residual_range_mass_l2_exact_f32_cache_v1;
    binding.performance_comparison_policy = .owned_capture_summary_incremental_reduction_critical025_v1;
    binding.train_probability_census = true;
    binding.rejected_detail_diagnostic = false;
    binding.train_primary_policy_comparison = false;
    binding.train_primary_policy_cross_validation = false;
    try std.testing.expect(trainChallengeHistoricalValid(binding) and plannedExactPolicyValid(binding));
    binding.historical_coverage_probability_backend = .accelerate_fp64_gather_eligible_v1;
    try std.testing.expect(trainChallengeHistoricalValid(binding) and plannedExactPolicyValid(binding));
    binding.hybrid_coverage_anchor = true;
    try std.testing.expect(trainChallengeHistoricalValid(binding) and plannedExactPolicyValid(binding));
    binding.historical_coverage_probability_backend = .scalar;
    try std.testing.expect(!trainChallengeHistoricalValid(binding));
    binding.historical_coverage_probability_backend = .accelerate_fp64_gather_eligible_v1;
    binding.hybrid_coverage_anchor = false;
    binding.train_challenge_historical_selection = false;
    binding.hybrid_coverage_anchor = false;
    try std.testing.expect(!trainChallengeHistoricalValid(binding));
    binding.historical_coverage_probability_backend = .scalar;
    try std.testing.expect(trainChallengeHistoricalValid(binding));
    binding.train_challenge_historical_selection = true;
    binding.hybrid_coverage_anchor = false;
    binding.historical_coverage_probability_backend = .accelerate_fp64_gather_eligible_v1;
    binding.train_primary_policy_cross_validation = true;
    try std.testing.expect(!trainChallengeHistoricalValid(binding));
    binding.train_primary_policy_cross_validation = false;
    binding.train_exact_reservation_policy = .disabled;
    try std.testing.expect(!trainChallengeHistoricalValid(binding));
    binding.train_exact_reservation_policy = .highest_budget_calibrated_critical_threshold_v1;
    binding.max_workspace_bytes -= 1;
    try std.testing.expect(!trainChallengeHistoricalValid(binding));
}
