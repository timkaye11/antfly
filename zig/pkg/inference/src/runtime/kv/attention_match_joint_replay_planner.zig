// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Checked conservative physical upper bound for the complete private Tier-A
//! joint replay consumer peak. Every simultaneously live allocation is counted;
//! non-overlapping phases are combined with `max` where proven below.
const std = @import("std");
const solver = @import("attention_match.zig");
const selection = @import("attention_match_global_selection_diagnostic.zig");
const joint = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");
const gram_geometry = @import("attention_match_joint_gram_geometry.zig");

pub const GeometryStorage = enum { dense, per_query_gram };

const FrozenOwnedLayout = struct {
    allocator: std.mem.Allocator,
    chunks: []joint.FrozenChunk,
    ranges: []admission.Range,
    indices: [][]usize,
    weights: [][]f64,
};
const ScoredCandidateLayout = struct { frozen: FrozenOwnedLayout, fitted: joint.Result };
pub const frozen_owned_layout_bytes = @sizeOf(FrozenOwnedLayout);
pub const frozen_owned_layout_alignment = @alignOf(FrozenOwnedLayout);
pub const scored_candidate_layout_bytes = @sizeOf(ScoredCandidateLayout);
pub const scored_candidate_layout_alignment = @alignOf(ScoredCandidateLayout);
pub const scored_candidate_table_entry_bytes = @sizeOf([3]?ScoredCandidateLayout);

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

fn admitRangeSessionBeforeAllocation(external_bytes: usize, existing_bytes: usize, session_bytes: usize, cap: usize) !usize {
    const admitted = try add(external_bytes, try add(existing_bytes, session_bytes));
    if (admitted > cap) return error.CompactionWorkspaceLimit;
    return admitted;
}

fn admitCombined(external: usize, consumer: usize, cap: usize) !void {
    if (try add(external, consumer) > cap) return error.CompactionWorkspaceLimit;
}

pub const Plan = struct {
    consumer_peak_bytes: usize,
    selection_peak_bytes: usize,
    joint_peak_bytes: usize,
    compose_peak_bytes: usize,
    validation_peak_bytes: usize,
};

pub const AdaptivePlan = struct {
    consumer_peak_bytes: usize,
    scoring_peak_bytes: usize,
    retained_geometry_bytes: usize,
    retained_scored_bytes: usize,
    signed_geometry_peak_bytes: usize,
    final_replay_peak_bytes: usize,
};

pub const CriticalTargetShape = struct {
    /// Actual retained rows for quarter/third/half levels, per range.
    total_rows: []const [3]usize,
    /// Residual-fit rows for the same levels; total-residual is the core count.
    residual_rows: []const [3]usize,
};

pub fn planAdaptive(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool) !AdaptivePlan {
    return planAdaptiveForStorage(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, .dense, 8, 0);
}

pub fn planAdaptiveForStorage(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize) !AdaptivePlan {
    return planAdaptiveForStorageRefit(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, .identity_chunk_coefficients_v18);
}

pub fn planAdaptiveForStorageRefit(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemo(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, 0);
}

pub fn planAdaptiveForStorageRefitJointMemo(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidation(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, null);
}

pub fn planAdaptiveForStorageRefitValidation(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, validation_backend: ?solver.Backend) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidation(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, 0, validation_backend);
}

pub fn planAdaptiveForStorageRefitJointMemoValidation(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, null, false, 0);
}

pub fn planAdaptiveForStorageRefitJointMemoValidationCritical(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, critical: CriticalTargetShape) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, critical, false, 0);
}

/// TRAIN-only primary probes terminate after scoring, geometry, and their
/// independently admitted candidate builder. Final replay/HHO allocations are
/// unreachable and must not reserve cache capacity.
pub fn planAdaptiveForStorageRefitJointMemoValidationReductionProbe(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, reduction_workspace_bytes: usize) !AdaptivePlan {
    var result = try planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, null, true, reduction_workspace_bytes);
    result.consumer_peak_bytes = @max(result.scoring_peak_bytes, result.signed_geometry_peak_bytes);
    if (try add(external_live_bytes, result.consumer_peak_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return result;
}

pub fn planAdaptiveForStorageRefitJointMemoValidationReductionCriticalProbe(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, reduction_workspace_bytes: usize, critical: CriticalTargetShape) !AdaptivePlan {
    var result = try planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, critical, true, reduction_workspace_bytes);
    result.consumer_peak_bytes = @max(result.scoring_peak_bytes, result.signed_geometry_peak_bytes);
    if (try add(external_live_bytes, result.consumer_peak_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return result;
}

pub fn planAdaptiveForStorageRefitJointMemoValidationReductionCritical(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, reduction_workspace_bytes: usize, critical: CriticalTargetShape) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, critical, false, reduction_workspace_bytes);
}

fn planAdaptiveForStorageRefitJointMemoValidationImpl(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, critical: ?CriticalTargetShape, probe_only: bool, reduction_workspace_bytes: usize) !AdaptivePlan {
    if (final_mass_strategy != .identity_chunk_coefficients_v18 and final_mass_strategy != .global_active_nnls_v16)
        return error.InvalidFitConfig;
    if (ranges.len == 0 or train_rows == 0 or heldout_rows == 0) return error.InvalidFitConfig;
    var max_rows: usize = 0;
    var max_selected: usize = 0;
    var max_residual: usize = 0;
    var max_core: usize = 0;
    var retained_selected: usize = 0;
    var retained_core_indices: usize = 0;
    if (critical) |shape| if (shape.total_rows.len != ranges.len or shape.residual_rows.len != ranges.len) return error.InvalidFitConfig;
    for (ranges, 0..) |range, ri| {
        if (range.rows == 0) return error.InvalidFitConfig;
        max_rows = @max(max_rows, range.rows);
        if (critical) |shape| {
            var core_count: ?usize = null;
            for (shape.total_rows[ri], shape.residual_rows[ri], 0..) |total, residual, level| {
                const divisor = ([3]usize{ 4, 3, 2 })[level];
                const nominal = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidAttentionShape;
                if (total < nominal or total >= range.rows or residual > total or (level != 0 and (total < shape.total_rows[ri][level - 1] or residual < shape.residual_rows[ri][level - 1]))) return error.InvalidFitConfig;
                const cores = total - residual;
                if (core_count != null and core_count.? != cores) return error.InvalidFitConfig;
                core_count = cores;
                retained_selected = try add(retained_selected, total);
                retained_core_indices = try add(retained_core_indices, cores);
                max_selected = @max(max_selected, total);
                max_residual = @max(max_residual, residual);
                max_core = @max(max_core, cores);
            }
        } else {
            max_selected = @max(max_selected, std.math.divCeil(usize, range.rows, 2) catch return error.InvalidAttentionShape);
            max_residual = max_selected;
            inline for (.{ 4, 3, 2 }) |divisor| retained_selected = try add(retained_selected, std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidAttentionShape);
        }
    }
    const score_table = try mul(ranges.len, @sizeOf(adaptive.RangeScores));
    const retained_geometry = try add(score_table, try mul(3, try add(try mul(ranges.len, @sizeOf(adaptive.Level) + @sizeOf(usize)), @sizeOf(adaptive.Geometry))));
    const scored_descriptors = try mul(try mul(ranges.len, 3), @sizeOf(joint.FrozenChunk) + @sizeOf(admission.Range) + (if (critical != null) @as(usize, 3) else @as(usize, 2)) * @sizeOf([]usize) + @sizeOf(joint.ChunkResult));
    const scored_payload = try add(try mul(retained_selected, try add(@sizeOf(usize) + 2 * @sizeOf(f64), try mul(original.dim, @sizeOf(f64)))), try mul(retained_core_indices, @sizeOf(usize)));
    const scored_table = try mul(ranges.len, scored_candidate_table_entry_bytes);
    const retained_scored = try add(scored_table, try add(scored_descriptors, scored_payload));
    const frozen = try add(@sizeOf(joint.FrozenChunk) + @sizeOf(admission.Range) + (if (critical != null) @as(usize, 3) else @as(usize, 2)) * @sizeOf([]usize), try add(try mul(max_selected, @sizeOf(usize) + @sizeOf(f64)), if (critical != null) try mul(max_core, @sizeOf(usize)) else 0));
    var local = cfg;
    local.target_rows = max_selected;
    const selection_peak = try selection.cachedWorkspaceBytes(max_rows, train_rows, local);
    const joint_peak = if (critical != null)
        (try joint.workspaceBytesShapeCoreAware(original, train_rows, 1, max_selected, max_selected, max_residual, cfg, true, .identity_chunk_coefficients_v18, false)).required
    else
        (try joint.workspaceBytesShape(original, train_rows, 1, max_selected, max_selected, cfg, true, .identity_chunk_coefficients_v18, false)).required;
    const fitted_bytes = try add(@sizeOf(joint.ChunkResult), try mul(try add(try mul(max_selected, original.dim), max_selected), @sizeOf(f64)));
    const score_vector_bytes = try mul(try add(train_rows, try mul(train_rows, original.dim)), @sizeOf(f64));
    // The range session is simultaneously live with candidate selection. It
    // is range-owned (not candidate scratch) and therefore charged once at
    // the largest N, including its typed descriptor.
    const range_gram_session_bytes = if (cfg.backend != null and cfg.backend.?.raw_range_policy == .range_shared_fp64_gram_original_kkt_v1) blk: {
        const workspace = cfg.backend.?.raw_range_workspace_fn orelse return error.UnsupportedOperation;
        break :blk try workspace(cfg.backend.?.context, max_rows);
    } else 0;
    _ = try admitRangeSessionBeforeAllocation(external_live_bytes, 0, range_gram_session_bytes, cfg.max_workspace_bytes);
    if (reduction_workspace_bytes != 0 and reduction_workspace_bytes < score_vector_bytes) return error.InvalidFitConfig;
    const fit_phase = try add(range_gram_session_bytes, try add(joint_fixed_memo_bytes, @max(try add(frozen, @max(selection_peak, joint_peak)), try add(frozen, fitted_bytes))));
    // Accelerated fitting completes and releases its fixed-original memo
    // before owned original reduction and streamed harm. The caller-provided
    // reduction authority is max(owned reduction, score vectors + exact harm
    // scratch), so these are three disjoint phases.
    const scoring_temporary = if (reduction_workspace_bytes == 0)
        try add(score_vector_bytes, try add(joint_fixed_memo_bytes, @max(try add(frozen, @max(selection_peak, joint_peak)), try add(try add(frozen, fitted_bytes), score_vector_bytes))))
    else
        @max(reduction_workspace_bytes, fit_phase);
    // Conservative pre-cache authority: all scored packets may be retained
    // while the largest current score attempt and its harm vectors are live.
    const scoring_peak = try add(try add(score_table, retained_scored), scoring_temporary);
    const signed_geometry_peak = switch (storage) {
        .dense => blk: {
            const signed_delta_table = try mul(try mul(ranges.len, 3), score_vector_bytes);
            const signed_score_descriptors = try mul(ranges.len, @sizeOf(adaptive.JointRangeScores));
            const signed_chooser_vectors = try mul(2, score_vector_bytes);
            break :blk try add(try add(retained_scored, retained_geometry), try add(signed_score_descriptors, try add(signed_delta_table, signed_chooser_vectors)));
        },
        .per_query_gram => blk: {
            const gram_peak = try gram_geometry.workspaceBytes(train_rows, original.dim, ranges.len, gram_tile_rows, gram_backend_scratch);
            const temporary_ranges = try mul(ranges.len, @sizeOf(gram_geometry.Range));
            break :blk try add(try add(retained_scored, retained_geometry), try add(temporary_ranges, gram_peak));
        },
    };
    if (probe_only) {
        const consumer = @max(scoring_peak, signed_geometry_peak);
        if (try add(external_live_bytes, consumer) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        return .{ .consumer_peak_bytes = consumer, .scoring_peak_bytes = scoring_peak, .retained_geometry_bytes = retained_geometry, .retained_scored_bytes = retained_scored, .signed_geometry_peak_bytes = signed_geometry_peak, .final_replay_peak_bytes = 0 };
    }
    // Identity targets only reduce final allocations, so the all-compacted
    // divisor ladder is a conservative authority for every chosen geometry.
    const final = try plan(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, final_mass_strategy, false);
    const branch_geometry_results = try mul(3, try mul(ranges.len, @sizeOf(usize)));
    var final_consumer = final.consumer_peak_bytes;
    if (critical) |shape| {
        var half_total: usize = 0;
        var legacy_half_total: usize = 0;
        var legacy_max_half: usize = 0;
        var max_half_total: usize = 0;
        var max_half_residual: usize = 0;
        var raised_all_levels: usize = 0;
        var core_indices_all_levels: usize = 0;
        for (ranges, shape.total_rows, shape.residual_rows) |range, totals, residuals| {
            const legacy = [3]usize{ std.math.divCeil(usize, range.rows, 4) catch return error.InvalidAttentionShape, std.math.divCeil(usize, range.rows, 3) catch return error.InvalidAttentionShape, std.math.divCeil(usize, range.rows, 2) catch return error.InvalidAttentionShape };
            for (totals, residuals, legacy) |total, residual, nominal| {
                raised_all_levels = try add(raised_all_levels, total - nominal);
                core_indices_all_levels = try add(core_indices_all_levels, total - residual);
            }
            half_total = try add(half_total, totals[2]);
            legacy_half_total = try add(legacy_half_total, legacy[2]);
            legacy_max_half = @max(legacy_max_half, legacy[2]);
            max_half_total = @max(max_half_total, totals[2]);
            max_half_residual = @max(max_half_residual, residuals[2]);
        }
        const critical_fit = (try joint.workspaceBytesShapeCoreAware(original, train_rows, ranges.len, half_total, max_half_total, max_half_residual, cfg, true, .identity_chunk_coefficients_v18, false)).required;
        const legacy_fit = (try joint.workspaceBytesShape(original, train_rows, ranges.len, legacy_half_total, legacy_max_half, cfg, true, .identity_chunk_coefficients_v18, false)).required;
        const retained_per_raised_row = try add(@sizeOf(usize) + 2 * @sizeOf(f64), try add(try mul(original.dim, 3 * @sizeOf(f64)), 2 * @sizeOf(usize)));
        var critical_delta = try mul(raised_all_levels, retained_per_raised_row);
        critical_delta = try add(critical_delta, try mul(core_indices_all_levels, @sizeOf(usize)));
        critical_delta = try add(critical_delta, critical_fit -| legacy_fit);
        final_consumer = try add(final_consumer, critical_delta);
    }
    if (validation_backend) |backend| {
        var scalar_cfg = cfg;
        scalar_cfg.backend = null;
        var accelerated_cfg = scalar_cfg;
        accelerated_cfg.backend = backend;
        const scalar_validation = try solver.validationBytes(original.rows, original.rows, original.dim, heldout_rows, scalar_cfg);
        const accelerated_validation = try solver.validationBytes(original.rows, original.rows, original.dim, heldout_rows, accelerated_cfg);
        if (accelerated_validation > scalar_validation) final_consumer = try add(final_consumer, accelerated_validation - scalar_validation);
    }
    if (observe_direct_train) {
        var validation_cfg = cfg;
        validation_cfg.backend = null;
        // Any signed geometry may preserve every range.  Original rows are
        // therefore the conservative compact-side validation width.
        const candidate_rows = original.rows;
        const heldout_validation = try solver.validationBytes(original.rows, candidate_rows, original.dim, heldout_rows, validation_cfg);
        const train_validation = try solver.validationBytes(original.rows, candidate_rows, original.dim, train_rows, validation_cfg);
        if (train_validation > heldout_validation) final_consumer = try add(final_consumer, train_validation - heldout_validation);
    }
    const final_replay_work = try add(branch_geometry_results, final_consumer);
    const final_replay_peak = try add(try add(retained_geometry, retained_scored), final_replay_work);
    const consumer = @max(@max(scoring_peak, signed_geometry_peak), final_replay_peak);
    if (try add(external_live_bytes, consumer) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return .{ .consumer_peak_bytes = consumer, .scoring_peak_bytes = scoring_peak, .retained_geometry_bytes = retained_geometry, .retained_scored_bytes = retained_scored, .signed_geometry_peak_bytes = signed_geometry_peak, .final_replay_peak_bytes = final_replay_peak };
}

/// Default-preserving wrapper for the optional cached-TRAIN reduction policy.
/// `reduction_workspace_bytes` includes the component's complete atomic output
/// staging and its caller-owned selected-ratio vector. It is simultaneous with
/// the existing scoring peak, never substituted for the scalar harm vectors.
pub fn planAdaptiveForStorageRefitJointMemoValidationReduction(allocator: std.mem.Allocator, original: solver.Head, train_rows: usize, heldout_rows: usize, ranges: []const admission.Range, cfg: solver.Config, branch_struct_bytes: usize, external_live_bytes: usize, observe_direct_train: bool, storage: GeometryStorage, gram_tile_rows: usize, gram_backend_scratch: usize, final_mass_strategy: joint.MassStrategy, joint_fixed_memo_bytes: usize, validation_backend: ?solver.Backend, reduction_workspace_bytes: usize) !AdaptivePlan {
    return planAdaptiveForStorageRefitJointMemoValidationImpl(allocator, original, train_rows, heldout_rows, ranges, cfg, branch_struct_bytes, external_live_bytes, observe_direct_train, storage, gram_tile_rows, gram_backend_scratch, final_mass_strategy, joint_fixed_memo_bytes, validation_backend, null, false, reduction_workspace_bytes);
}

pub fn plan(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train_rows: usize,
    heldout_rows: usize,
    ranges: []const admission.Range,
    cfg: solver.Config,
    branch_struct_bytes: usize,
    external_live_bytes: usize,
    mass_strategy: joint.MassStrategy,
    retain_training_diagnostics: bool,
) !Plan {
    if (ranges.len == 0 or train_rows == 0 or heldout_rows == 0) return error.InvalidFitConfig;
    if (try add(external_live_bytes, branch_struct_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var answer: Plan = .{ .consumer_peak_bytes = branch_struct_bytes, .selection_peak_bytes = 0, .joint_peak_bytes = 0, .compose_peak_bytes = 0, .validation_peak_bytes = 0 };
    var prior_alpha_bytes: usize = 0;
    for ([_]usize{ 4, 3, 2 }, 0..) |divisor, answer_index| {
        const history = try add(branch_struct_bytes, prior_alpha_bytes);
        var selected_total: usize = 0;
        var max_selected: usize = 0;
        for (ranges) |range| {
            const count = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidAttentionShape;
            selected_total = try add(selected_total, count);
            max_selected = @max(max_selected, count);
        }
        const planner_scratch = try add(try mul(max_selected, @sizeOf(usize) + @sizeOf(f64)), try mul(ranges.len, @sizeOf(joint.FrozenChunk)));
        if (try add(external_live_bytes, planner_scratch) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        const dummy_indices = try allocator.alloc(usize, max_selected);
        defer allocator.free(dummy_indices);
        const dummy_weights = try allocator.alloc(f64, max_selected);
        defer allocator.free(dummy_weights);
        for (dummy_indices, dummy_weights, 0..) |*index, *weight, i| {
            index.* = i;
            weight.* = 1;
        }
        const chunks = try allocator.alloc(joint.FrozenChunk, ranges.len);
        defer allocator.free(chunks);
        for (ranges, chunks) |range, *chunk| {
            const count = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidAttentionShape;
            chunk.* = .{ .start = range.start, .rows = range.rows, .selected_local = dummy_indices[0..count], .weights = dummy_weights[0..count] };
        }
        const frozen_base = try add(try mul(ranges.len, @sizeOf(joint.FrozenChunk)), try add(try mul(ranges.len, @sizeOf(admission.Range)), try mul(try mul(ranges.len, 2), @sizeOf([]usize))));
        var frozen_live = frozen_base;
        for (ranges, chunks) |range, chunk| {
            var local = cfg;
            local.target_rows = chunk.selected_local.len;
            const output = try mul(chunk.selected_local.len, @sizeOf(usize) + @sizeOf(f64));
            const select_workspace = try selection.cachedWorkspaceBytes(range.rows, train_rows, local);
            // During selection, previously frozen outputs remain live. The
            // current selector workspace already contains its returned output;
            // only the post-return duplicate overlaps it, never 2*output plus
            // the complete workspace.
            const peak = try add(history, try add(frozen_live, @max(select_workspace, try mul(output, 2))));
            answer.selection_peak_bytes = @max(answer.selection_peak_bytes, peak);
            frozen_live = try add(frozen_live, output);
        }
        const joint_workspace = (try joint.workspaceBytesStrategy(original, train_rows, chunks, cfg, true, mass_strategy, retain_training_diagnostics)).required;
        answer.joint_peak_bytes = @max(answer.joint_peak_bytes, try add(history, try add(frozen_live, joint_workspace)));
        const fitted_bytes = try add(try mul(chunks.len, @sizeOf(joint.ChunkResult)), try mul(try add(try mul(selected_total, original.dim), selected_total), @sizeOf(f64)));
        var candidate_rows = original.rows;
        for (ranges, chunks) |range, chunk| candidate_rows = try add(std.math.sub(usize, candidate_rows, range.rows) catch return error.InvalidAttentionShape, chunk.selected_local.len);
        const candidate_numeric = try mul(try add(try mul(try mul(candidate_rows, original.dim), 2), candidate_rows), @sizeOf(f64));
        const candidate_maps = try mul(try mul(candidate_rows, 2), @sizeOf(usize));
        const compose = try add(history, try add(try add(frozen_live, fitted_bytes), try add(try add(candidate_numeric, candidate_maps), try mul(max_selected, @sizeOf(usize)))));
        answer.compose_peak_bytes = @max(answer.compose_peak_bytes, compose);
        var held = cfg;
        held.backend = null;
        const validation = try solver.validationBytes(original.rows, candidate_rows, original.dim, heldout_rows, held);
        const alpha = try mul(chunks.len, @sizeOf(f64));
        const candidate_live = try add(try add(frozen_live, fitted_bytes), try add(candidate_numeric, candidate_maps));
        const validation_peak = try add(history, try add(candidate_live, validation));
        const branch_descriptor_bytes = branch_struct_bytes / 3;
        if (try mul(branch_descriptor_bytes, 3) != branch_struct_bytes) return error.InvalidAttentionShape;
        const accepted_duplicate = try mul(answer_index + 1, branch_descriptor_bytes);
        const accepted_peak = try add(history, try add(candidate_live, try add(alpha, accepted_duplicate)));
        answer.validation_peak_bytes = @max(answer.validation_peak_bytes, @max(validation_peak, accepted_peak));
        prior_alpha_bytes = try add(prior_alpha_bytes, alpha);
    }
    const elements = try mul(original.rows, original.dim);
    const fallback_numeric = try mul(try add(try mul(elements, 2), original.rows), @sizeOf(f64));
    const fallback_maps = try mul(try mul(original.rows, 2), @sizeOf(usize));
    const fallback_peak = try add(try add(branch_struct_bytes, prior_alpha_bytes), try add(fallback_numeric, fallback_maps));
    answer.compose_peak_bytes = @max(answer.compose_peak_bytes, fallback_peak);
    answer.consumer_peak_bytes = @max(@max(answer.selection_peak_bytes, answer.joint_peak_bytes), @max(answer.compose_peak_bytes, answer.validation_peak_bytes));
    if (try add(external_live_bytes, answer.consumer_peak_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return answer;
}

test "D23 streamed scoring observed authority fits exact combined cap and rejects cap minus one" {
    const q: usize = 12_288;
    const d: usize = 512;
    const vector = try mul(try add(q, try mul(q, d)), @sizeOf(f64));
    const new_scoring = try add(438_326_832 - 2 * vector, 64 * @sizeOf(f64));
    try std.testing.expectEqual(@as(usize, 337_467_440), new_scoring);
    const total = try add(294_168_820, try add(440_401_920, new_scoring));
    try std.testing.expectEqual(@as(usize, 1_072_038_180), total);
    try std.testing.expect(total <= 1024 * 1024 * 1024);
    try admitCombined(294_168_820 + 440_401_920, new_scoring, total);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitCombined(294_168_820 + 440_401_920, new_scoring, total - 1));
}

test "D23 initial noncritical planner uses production Metal workspace and fits observed cap" {
    if (!@import("build_options").enable_metal) return error.SkipZigTest;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var ranges: [23]admission.Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == ranges.len) 128 else 256 };
        start += range.rows;
    }
    try std.testing.expectEqual(@as(usize, 5888), start);
    const q: usize = 12_288;
    const d: usize = 512;
    const max_rows: usize = 256;
    const max_selected: usize = 128;
    const batch: usize = 64;
    const score_vectors = try mul(try add(q, try mul(q, d)), @sizeOf(f64));
    const original_scratch = try add(try mul(try add(try mul(batch, max_rows), try add(try mul(batch, d), max_rows)), @sizeOf(f64)), score_vectors);
    const harm_scratch = try add(score_vectors, try add(try mul(try add(try mul(batch, max_selected), try add(batch, try mul(batch, d))), @sizeOf(f64)), try mul(2 * max_selected, @sizeOf(f64))));
    const reduction = @max(original_scratch, harm_scratch);
    const external: usize = 294_168_820 + 440_401_920;
    const head: solver.Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 7936, .dim = d, .scale = 1 };
    var fitter = try Fitter.initWorker(std.testing.allocator, null, 1 << 30, 1 << 30);
    defer fitter.deinit();
    try fitter.setQueryBatchRows(64);
    var cfg: solver.Config = .{ .target_rows = 1, .ridge = 1e-8, .max_workspace_bytes = 1024 * 1024 * 1024, .backend = fitter.backend() };
    const planned = try planAdaptiveForStorageRefitJointMemoValidationReductionProbe(std.testing.allocator, head, q, 1024, &ranges, cfg, 3 * @sizeOf(usize), external, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, score_vectors, null, reduction);
    try admitCombined(external, planned.consumer_peak_bytes, cfg.max_workspace_bytes);
    cfg.max_workspace_bytes = try add(external, planned.consumer_peak_bytes);
    _ = try planAdaptiveForStorageRefitJointMemoValidationReductionProbe(std.testing.allocator, head, q, 1024, &ranges, cfg, 3 * @sizeOf(usize), external, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, score_vectors, null, reduction);
    cfg.max_workspace_bytes -= 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, planAdaptiveForStorageRefitJointMemoValidationReductionProbe(std.testing.allocator, head, q, 1024, &ranges, cfg, 3 * @sizeOf(usize), external, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, score_vectors, null, reduction));
    try std.testing.expectEqual(@as(u64, 0), fitter.dispatchCount());
    try std.testing.expectEqual(@as(u64, 0), fitter.attentionCallCount());
    try std.testing.expectEqual(@as(u64, 0), fitter.trainingCaptureCounters().allocations);
    try std.testing.expectEqual(@as(u64, 0), fitter.probabilityPageCounters().calls);
}

test "critical target shape separates full output from residual workspace at q12288" {
    const FinalBackend = struct {
        calls: usize = 0,
        fn workspace(raw: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return 4096;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {
            return error.TestUnexpectedResult;
        }
    };
    const range_count = 23;
    var ranges: [range_count]admission.Range = undefined;
    var totals: [range_count][3]usize = undefined;
    var residuals: [range_count][3]usize = undefined;
    var start: usize = 128;
    for (&ranges, &totals, &residuals, 0..) |*range, *total, *residual, index| {
        const rows = @min(@as(usize, 256), 5888 - start);
        range.* = .{ .start = start, .rows = rows };
        const nominal = [3]usize{
            std.math.divCeil(usize, rows, 4) catch unreachable,
            std.math.divCeil(usize, rows, 3) catch unreachable,
            std.math.divCeil(usize, rows, 2) catch unreachable,
        };
        if (index == 0) {
            total.* = .{ 96, 96, 128 }; // core=95 produces duplicate lower levels.
            residual.* = .{ 1, 1, 33 };
        } else {
            total.* = nominal;
            residual.* = nominal;
        }
        start += rows;
    }
    try std.testing.expectEqual(@as(usize, 5888), start);
    const head: solver.Head = .{ .keys = &.{}, .values = &.{}, .bias = null, .rows = 7936, .dim = 512, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 30 };
    const legacy = try planAdaptiveForStorageRefitJointMemoValidationReduction(std.testing.allocator, head, 12_288, 1024, &ranges, cfg, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, null, 0);
    const shape: CriticalTargetShape = .{ .total_rows = &totals, .residual_rows = &residuals };
    const critical = try planAdaptiveForStorageRefitJointMemoValidationReductionCritical(std.testing.allocator, head, 12_288, 1024, &ranges, cfg, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, null, 0, shape);
    var final_backend: FinalBackend = .{};
    const backend: solver.Backend = .{ .context = &final_backend, .workspace_fn = FinalBackend.workspace, .summary_fn = FinalBackend.summary };
    const probe = try planAdaptiveForStorageRefitJointMemoValidationReductionCriticalProbe(std.testing.allocator, head, 12_288, 1024, &ranges, cfg, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, backend, 0, shape);
    try std.testing.expectEqual(@as(usize, 0), final_backend.calls);
    _ = try planAdaptiveForStorageRefitJointMemoValidationReductionCritical(std.testing.allocator, head, 12_288, 1024, &ranges, cfg, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, backend, 0, shape);
    try std.testing.expect(final_backend.calls > 0);
    try std.testing.expect(critical.retained_scored_bytes > legacy.retained_scored_bytes);
    try std.testing.expectEqual(@max(probe.scoring_peak_bytes, probe.signed_geometry_peak_bytes), probe.consumer_peak_bytes);
    try std.testing.expectEqual(@as(usize, 0), probe.final_replay_peak_bytes);
    var probe_only_cap = cfg;
    probe_only_cap.max_workspace_bytes = probe.consumer_peak_bytes;
    _ = try planAdaptiveForStorageRefitJointMemoValidationReductionCriticalProbe(std.testing.allocator, head, 12_288, 1024, &ranges, probe_only_cap, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, null, 0, shape);
    probe_only_cap.max_workspace_bytes -= 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, planAdaptiveForStorageRefitJointMemoValidationReductionCriticalProbe(std.testing.allocator, head, 12_288, 1024, &ranges, probe_only_cap, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, null, 0, shape));
    try std.testing.expect(critical.consumer_peak_bytes >= legacy.consumer_peak_bytes);
    var capped = cfg;
    capped.max_workspace_bytes = critical.consumer_peak_bytes - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, planAdaptiveForStorageRefitJointMemoValidationReductionCritical(std.testing.allocator, head, 12_288, 1024, &ranges, capped, 3 * 256, 0, false, .per_query_gram, 64, 0, .identity_chunk_coefficients_v18, 0, null, 0, shape));
}

test "context adaptive planner jointly charges fixed memo and FP64 summary validation at exact cap" {
    const Probe = struct {
        workspace_calls: usize = 0,
        summary_calls: usize = 0,
        fn workspace(raw: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.workspace_calls += 1;
            return 4096;
        }
        fn summary(raw: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.summary_calls += 1;
            return error.TestUnexpectedResult;
        }
    };
    var probe: Probe = .{};
    const backend: solver.Backend = .{ .context = &probe, .workspace_fn = Probe.workspace, .summary_fn = Probe.summary };
    const keys = [_]f64{ 1, 0, 0, 1, 1, 1, -1, 1 };
    const values = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const original: solver.Head = .{ .keys = &keys, .values = &values, .rows = 4, .dim = 2, .scale = 1 };
    const ranges = [_]admission.Range{ .{ .start = 0, .rows = 2 }, .{ .start = 2, .rows = 2 } };
    const external: usize = 37;
    const memo: usize = 1234;
    var cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = std.math.maxInt(usize) };
    const none = try planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, null);
    const memo_only = try planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, memo, null);
    const backend_only = try planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, backend);
    const both = try planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, memo, backend);
    try std.testing.expect(probe.workspace_calls > 0);
    try std.testing.expectEqual(@as(usize, 0), probe.summary_calls);
    try std.testing.expectEqual(try add(none.scoring_peak_bytes, memo), memo_only.scoring_peak_bytes);
    try std.testing.expectEqual(try add(backend_only.scoring_peak_bytes, memo), both.scoring_peak_bytes);
    try std.testing.expectEqual(none.final_replay_peak_bytes, memo_only.final_replay_peak_bytes);
    var scalar_cfg = cfg;
    scalar_cfg.backend = null;
    var accelerated_cfg = scalar_cfg;
    accelerated_cfg.backend = backend;
    const scalar_validation = try solver.validationBytes(original.rows, original.rows, original.dim, 3, scalar_cfg);
    const accelerated_validation = try solver.validationBytes(original.rows, original.rows, original.dim, 3, accelerated_cfg);
    const validation_delta = accelerated_validation - scalar_validation;
    try std.testing.expectEqual(try add(none.final_replay_peak_bytes, validation_delta), backend_only.final_replay_peak_bytes);
    try std.testing.expectEqual(try add(memo_only.final_replay_peak_bytes, validation_delta), both.final_replay_peak_bytes);
    try std.testing.expectEqual(none.signed_geometry_peak_bytes, both.signed_geometry_peak_bytes);
    const expected_consumer = @max(@max(both.scoring_peak_bytes, both.signed_geometry_peak_bytes), both.final_replay_peak_bytes);
    try std.testing.expectEqual(expected_consumer, both.consumer_peak_bytes);
    const exact = try add(external, expected_consumer);
    cfg.max_workspace_bytes = exact;
    const admitted = try planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, memo, backend);
    try std.testing.expectEqual(expected_consumer, admitted.consumer_peak_bytes);
    try std.testing.expectEqual(@as(usize, 0), probe.summary_calls);
    cfg.max_workspace_bytes = exact - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, planAdaptiveForStorageRefitJointMemoValidation(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, memo, backend));
    try std.testing.expectEqual(@as(usize, 0), probe.summary_calls);
}

test "context adaptive planner charges cached TRAIN reduction workspace before dispatch" {
    const keys = [_]f64{ 1, 0, 0, 1, 1, 1, -1, 1 };
    const values = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const original: solver.Head = .{ .keys = &keys, .values = &values, .rows = 4, .dim = 2, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 4 }};
    const external: usize = 37;
    const reduction: usize = 4096;
    var cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = std.math.maxInt(usize) };
    const scalar = try planAdaptiveForStorageRefitJointMemoValidationReduction(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, null, 0);
    const accelerated = try planAdaptiveForStorageRefitJointMemoValidationReduction(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, null, reduction);
    // The original-reduction and candidate-fit phases do not overlap. This
    // fixture makes the old global addition strictly larger than the real peak.
    try std.testing.expect(accelerated.scoring_peak_bytes >= reduction);
    try std.testing.expect(accelerated.scoring_peak_bytes < try add(scalar.scoring_peak_bytes, reduction));
    try std.testing.expectEqual(scalar.signed_geometry_peak_bytes, accelerated.signed_geometry_peak_bytes);
    try std.testing.expectEqual(scalar.final_replay_peak_bytes, accelerated.final_replay_peak_bytes);
    try std.testing.expectEqual(@max(@max(accelerated.scoring_peak_bytes, accelerated.signed_geometry_peak_bytes), accelerated.final_replay_peak_bytes), accelerated.consumer_peak_bytes);
    const exact = try add(external, accelerated.consumer_peak_bytes);
    cfg.max_workspace_bytes = exact;
    _ = try planAdaptiveForStorageRefitJointMemoValidationReduction(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, null, reduction);
    cfg.max_workspace_bytes = exact - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, planAdaptiveForStorageRefitJointMemoValidationReduction(std.testing.allocator, original, 5, 3, &ranges, cfg, 3 * 64, external, false, .dense, 2, 0, .identity_chunk_coefficients_v18, 0, null, reduction));
}

test "range session owner exact cap admits and one byte under rejects before allocation" {
    const external: usize = 113;
    const existing: usize = 257;
    const payload: usize = 8_396_800;
    const descriptor: usize = 192;
    const owner = try add(payload, descriptor);
    const exact = try add(external, try add(existing, owner));
    try std.testing.expectEqual(exact, try admitRangeSessionBeforeAllocation(external, existing, owner, exact));
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitRangeSessionBeforeAllocation(external, existing, owner, exact - 1));
}

test "range session owner admission overflow fails before allocation" {
    try std.testing.expectError(error.InvalidAttentionShape, admitRangeSessionBeforeAllocation(std.math.maxInt(usize), 1, 1, std.math.maxInt(usize)));
}
