const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const checkpoint = @import("attention_match_checkpoint.zig");
const solver = @import("attention_match.zig");
const fixture_mod = @import("attention_match_rejected_fixture.zig");
const exact = @import("attention_match_exact_peak_residual.zig");
const mass_calibration = @import("attention_match_exact_core_mass_calibration.zig");
const fixed_support_refit = @import("attention_match_fixed_support_global_refit.zig");
const joint_mass_bias_refit = @import("attention_match_joint_mass_bias_refit.zig");
const accelerate_value_backend = @import("attention_match_accelerate_fp64_value_backend.zig");
const prior_cv = @import("attention_match_joint_multiplicity_prior_cv.zig");
const bounded_page_replay = @import("attention_match_joint_bounded_page_replay.zig");
const adaptive_geometry = @import("attention_match_joint_adaptive_geometry.zig");

const ResidualGroupCalibrationPolicy = enum { disabled, anchored_l2_v1, global_minimax_irls_p4_p8_p16_v1 };
const DiagnosticQueryBatchRows = enum { rows16, rows64 };

const Plan = struct {
    schema: []const u8,
    experiment_schema: []const u8,
    fixture_dir: []const u8,
    fixture_manifest_sha256: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    parent_fixture_manifest_sha256: []const u8,
    parent_replay_plan_sha256: []const u8,
    target_rows: usize,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    retention_divisor: usize,
    mandatory_training_top_k: usize,
    result_path: ?[]const u8 = null,
    risk_report_path: ?[]const u8 = null,
    risk_report_sha256: ?[]const u8 = null,
    geometry_targets: ?[]const usize = null,
    exact_core_source_keys: ?[]const []const usize = null,
    residual_group_mass_calibration: bool = false,
    residual_group_calibration_policy: ResidualGroupCalibrationPolicy = .disabled,
    diagnostic_phase_profiling: bool = false,
    diagnostic_query_batch_rows: DiagnosticQueryBatchRows = .rows16,
    deferred_intermediate_training_checks: bool = false,
};

const RiskReport = struct {
    schema: []const u8,
    authenticated: bool,
    donor_layer: u32,
    kv_head: usize,
    geometry_divisor: usize,
    heldout_query_values_read: bool,
    ranges: []const struct { target_budget: usize, exact_core_source_keys: []const usize },
};

const ConsumerTuple = struct { layer: u32, query_head: u32 };
const ShardRef = struct {
    fixture_dir: []const u8,
    fixture_manifest_sha256: []const u8,
    fixture_provenance_sha256: []const u8,
    consumers: []const ConsumerTuple,
};
const ShardedPlan = struct {
    schema: []const u8,
    experiment_schema: []const u8,
    result_path: []const u8,
    fixture_shards: []const ShardRef,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    donor_layer: u32,
    kv_head: usize,
    target_rows: usize,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    retention_divisor: usize,
    risk_report_path: []const u8,
    risk_report_sha256: []const u8,
    risk_rule: []const u8,
    geometry_targets: []const usize,
    exact_core_source_keys: []const []const usize,
    deferred_intermediate_training_checks: bool = false,
    diagnostic_phase_profiling: bool = false,
    diagnostic_query_batch_rows: DiagnosticQueryBatchRows = .rows16,
    joint_mass_bias_module_sha256: ?[]const u8 = null,
    joint_component_cap_mib: ?usize = null,
    joint_value_backend_accelerate: bool = false,
};

const PriorCvContext = struct {
    fold_ids: []const u8,
    variable_range_indices: []const u8,
    ranges: []const prior_cv.Range,
    result: ?prior_cv.Result = null,
};

fn selectPriorCvWeights(raw: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, rho: []const f64, q: usize, m: usize, warm: []const f64, cfg: solver.Config, external_live: usize) !joint_mass_bias_refit.SelectedWeights {
    const context: *PriorCvContext = @ptrCast(@alignCast(raw.?));
    if (context.result != null) return error.InvalidFitConfig;
    const live_scalar_count = try std.math.add(usize, try std.math.add(usize, a.len, rho.len), warm.len);
    var cv_external = try std.math.add(usize, external_live, try std.math.mul(usize, live_scalar_count, @sizeOf(f64)));
    cv_external = try std.math.add(usize, cv_external, context.fold_ids.len + context.variable_range_indices.len);
    cv_external = try std.math.add(usize, cv_external, try std.math.mul(usize, context.ranges.len, @sizeOf(prior_cv.Range)));
    cv_external = try std.math.add(usize, cv_external, try std.math.mul(usize, 64 * 7936, @sizeOf(f64)));
    cv_external = try std.math.add(usize, cv_external, 4908 + try std.math.mul(usize, m, @sizeOf(usize)));
    var result = try prior_cv.selectAndFit(allocator, a, rho, q, m, context.fold_ids, context.variable_range_indices, context.ranges, warm, cfg, cv_external);
    const weights = result.weights;
    result.weights = &.{};
    const sweeps = result.final_sweeps;
    context.result = result;
    return .{ .weights = weights, .sweeps = sweeps };
}
const GenericRiskReport = struct {
    schema: []const u8,
    authenticated: bool,
    donor_layer: u32,
    kv_head: usize,
    geometry_divisor: usize,
    original_source_rows: usize,
    original_source_positions_contiguous: bool,
    train_scope: []const u8,
    train_rows_total: usize,
    heldout_query_values_read: bool,
    risk_rule: []const u8,
    distinct_exact_core_source_keys: usize,
    ranges: []const struct { range_index: usize, start: usize, rows: usize, target_budget: usize, exact_core_count: usize, exact_core_source_keys: []const usize },
};

const FixedSupportRefitPlan = struct {
    schema: []const u8,
    experiment_schema: []const u8,
    result_path: []const u8,
    fixture_dir: []const u8,
    fixture_manifest_sha256: []const u8,
    fixture_provenance_sha256: []const u8,
    fixture_index_path: []const u8,
    fixture_index_sha256: []const u8,
    risk_report_path: []const u8,
    risk_report_sha256: []const u8,
    risk_rule: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    consumers: []const ConsumerTuple,
    donor_layer: u32,
    kv_head: usize,
    target_rows: usize,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    retention_divisor: usize,
    geometry_targets: []const usize,
    exact_core_source_keys: []const []const usize,
    expected_baseline_candidate_sha256: []const u8,
    expected_baseline_candidate_kvb_sha256: []const u8,
    source562_module_sha256: ?[]const u8 = null,
    joint_mass_bias_module_sha256: ?[]const u8 = null,
    joint_value_backend_accelerate: bool = false,
    joint_prior_cv: bool = false,
    joint_prior_cv_policy_sha256: ?[]const u8 = null,
    joint_prior_cv_module_sha256: ?[]const u8 = null,
    max_combined_bytes: usize,
    fitter_workspace_bytes: usize,
    query_batch_rows: usize,
};
const FixtureIndex = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    donor_layer: u32,
    kv_head: usize,
    source_rows: usize,
    head_dim: usize,
    candidate_rows: usize,
    config_sha256: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    geometry_targets: []const usize,
    ordered_unique_consumers: []const ConsumerTuple,
    fixture_shards: []const ShardRef,
};

const BorrowedCandidate = struct {
    head: solver.Head,
    positions: []const u64,
    coverage: []const u64,
};

fn borrowedCandidateDigest(candidate: BorrowedCandidate) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.d5h1-rounded-candidate.v1\x00");
    hash.update(std.mem.sliceAsBytes(candidate.head.keys));
    hash.update(std.mem.sliceAsBytes(candidate.head.values));
    hash.update(std.mem.sliceAsBytes(candidate.head.bias.?));
    hash.update(std.mem.sliceAsBytes(candidate.positions));
    hash.update(std.mem.sliceAsBytes(candidate.coverage));
    return hash.finalResult();
}

fn borrowedCandidateKvbDigest(candidate: BorrowedCandidate) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.d5h1-rounded-candidate-kvb.v1\x00");
    hash.update(std.mem.sliceAsBytes(candidate.head.keys));
    hash.update(std.mem.sliceAsBytes(candidate.head.values));
    hash.update(std.mem.sliceAsBytes(candidate.head.bias.?));
    return hash.finalResult();
}

fn bytesDigest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

fn validateBorrowedCandidate(allocator: std.mem.Allocator, original: solver.Head, candidate: BorrowedCandidate, heldout: []const f64) !Cell {
    const strict: solver.Config = .{ .target_rows = candidate.head.rows, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 };
    const quality = try solver.validationMetrics(allocator, original, candidate.head, heldout, strict);
    return .{
        .accelerated = false,
        .accepted = quality.max_relative_mass_error <= strict.max_relative_mass_error and quality.relative_output_rmse <= strict.max_relative_output_rmse,
        .elapsed_ns = 0,
        .accelerated_value_solves = 0,
        .max_nnls_sweeps = 0,
        .mass_error = quality.max_relative_mass_error,
        .output_rmse = quality.relative_output_rmse,
    };
}

fn exactRangeClosure(plan: FixedSupportRefitPlan, storage: *[23]fixed_support_refit.Range) ![]const fixed_support_refit.Range {
    if (plan.geometry_targets.len != storage.len or plan.exact_core_source_keys.len != storage.len) return error.InvalidRejectedHeadReplayPlan;
    var middle_total: usize = 0;
    for (storage, plan.geometry_targets, plan.exact_core_source_keys, 0..) |*range, target, cores, index| {
        const start = 128 + index * 256;
        const rows = @min(@as(usize, 256), 5888 - start);
        if (target == 0 or target > rows or cores.len >= target) return error.InvalidRejectedHeadReplayPlan;
        for (cores, 0..) |key, core_index| {
            if (key < start or key >= start + rows or std.mem.indexOfScalar(usize, cores[0..core_index], key) != null) return error.InvalidRejectedHeadReplayPlan;
        }
        range.* = .{ .start = start, .rows = rows, .target_budget = target, .exact_core_source_keys = cores };
        middle_total = try std.math.add(usize, middle_total, target);
    }
    if (plan.target_rows != try std.math.add(usize, 2176, middle_total)) return error.InvalidRejectedHeadReplayPlan;
    return storage;
}

fn jointRangeClosure(plan: FixedSupportRefitPlan, storage: *[23]joint_mass_bias_refit.Range) ![]const joint_mass_bias_refit.Range {
    if (plan.geometry_targets.len != storage.len or plan.exact_core_source_keys.len != storage.len) return error.InvalidRejectedHeadReplayPlan;
    var middle_total: usize = 0;
    for (storage, plan.geometry_targets, plan.exact_core_source_keys, 0..) |*range, target, cores, index| {
        const start = 128 + index * 256;
        const rows = @min(@as(usize, 256), 5888 - start);
        if (target == 0 or target > rows or cores.len >= target) return error.InvalidRejectedHeadReplayPlan;
        for (cores, 0..) |key, core_index| if (key < start or key >= start + rows or std.mem.indexOfScalar(usize, cores[0..core_index], key) != null) return error.InvalidRejectedHeadReplayPlan;
        range.* = .{ .start = start, .rows = rows, .target_budget = target, .exact_core_source_keys = cores };
        middle_total = try std.math.add(usize, middle_total, target);
    }
    if (plan.target_rows != try std.math.add(usize, 2176, middle_total)) return error.InvalidRejectedHeadReplayPlan;
    return storage;
}

fn baselineOwnedCandidateBytes(target_rows: usize, dim: usize) !usize {
    const scalars = try std.math.add(usize, try std.math.mul(usize, target_rows, try std.math.mul(usize, dim, 2)), try std.math.mul(usize, target_rows, 3));
    return std.math.mul(usize, scalars, @sizeOf(f64));
}

fn checkedFixedSupportAdmission(query_count: usize, original_rows: usize, dim: usize, target_rows: usize, maximum_target_budget: usize, plan_scratch_bytes: usize) !usize {
    var total: usize = fixture_mod.maximum_bytes;
    total = try std.math.add(usize, total, try baselineOwnedCandidateBytes(target_rows, dim));
    total = try std.math.add(usize, total, try fixed_support_refit.preCacheAdmissionBytes(query_count, original_rows, dim, target_rows, maximum_target_budget));
    total = try std.math.add(usize, total, plan_scratch_bytes);
    if (total > fixed_support_refit.combined_cap_bytes) return error.CompactionWorkspaceLimit;
    return total;
}

fn finalValidationLiveBytes(target_rows: usize, metadata_bytes: usize) !usize {
    const strict: solver.Config = .{ .target_rows = target_rows, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 };
    var total = try std.math.mul(usize, fixture_mod.maximum_bytes, 2);
    total = try std.math.add(usize, total, try baselineOwnedCandidateBytes(target_rows, 512));
    total = try std.math.add(usize, total, try fixed_support_refit.requiredOwnedOutputBytes(target_rows, 512));
    total = try std.math.add(usize, total, try solver.validationBytes(7936, target_rows, 512, 256, strict));
    total = try std.math.add(usize, total, metadata_bytes);
    return total;
}

fn checkedFinalValidationAdmission(target_rows: usize, metadata_bytes: usize) !usize {
    const total = try finalValidationLiveBytes(target_rows, metadata_bytes);
    if (total > fixed_support_refit.combined_cap_bytes) return error.CompactionWorkspaceLimit;
    return total;
}

const Cell = struct {
    accelerated: bool,
    accepted: bool,
    elapsed_ns: u64,
    accelerated_value_solves: u64,
    max_nnls_sweeps: usize,
    mass_error: f64,
    output_rmse: f64,
};

const OwnedCandidate = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    positions: []u64,
    coverage: []u64,
    accelerated: bool,
    max_nnls_sweeps: usize,
    elapsed_ns: u64,

    fn deinit(self: *OwnedCandidate) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.positions);
        self.allocator.free(self.coverage);
        self.* = undefined;
    }
    fn head(self: OwnedCandidate, original: solver.Head) solver.Head {
        return .{ .keys = self.keys, .values = self.values, .bias = self.bias, .rows = self.bias.len, .dim = original.dim, .scale = original.scale, .softcap = original.softcap };
    }
};

fn cloneCandidate(allocator: std.mem.Allocator, source: OwnedCandidate) !OwnedCandidate {
    const keys = try allocator.dupe(f64, source.keys);
    errdefer allocator.free(keys);
    const values = try allocator.dupe(f64, source.values);
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f64, source.bias);
    errdefer allocator.free(bias);
    const positions = try allocator.dupe(u64, source.positions);
    errdefer allocator.free(positions);
    const coverage = try allocator.dupe(u64, source.coverage);
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias, .positions = positions, .coverage = coverage, .accelerated = source.accelerated, .max_nnls_sweeps = source.max_nnls_sweeps, .elapsed_ns = source.elapsed_ns };
}

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidRejectedHeadReplayPlan;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidRejectedHeadReplayPlan;
    return out;
}

fn roundF16(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f16)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f16, @floatCast(value.*)));
    }
}

fn roundF32(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f32, @floatCast(value.*)));
    }
}

fn appendRows(dst_keys: []f64, dst_values: []f64, dst_bias: []f64, dst_positions: []u64, dst_coverage: []u64, at: *usize, source: solver.Head, positions: []const u64, coverage: []const u64) !void {
    if (positions.len != source.rows or coverage.len != source.rows or at.* + source.rows > dst_bias.len) return error.InvalidRejectedHeadReplay;
    const row = at.*;
    @memcpy(dst_keys[row * source.dim ..][0 .. source.rows * source.dim], source.keys);
    @memcpy(dst_values[row * source.dim ..][0 .. source.rows * source.dim], source.values);
    @memcpy(dst_bias[row..][0..source.rows], source.bias orelse return error.InvalidRejectedHeadReplay);
    @memcpy(dst_positions[row..][0..source.rows], positions);
    @memcpy(dst_coverage[row..][0..source.rows], coverage);
    at.* += source.rows;
}

fn buildCandidate(allocator: std.mem.Allocator, fixture: *const fixture_mod.Fixture, seeds: []const []const usize, geometry_targets: ?[]const usize, backend: ?solver.Backend, deferred_intermediate_training_checks: bool) !OwnedCandidate {
    const dim = fixture.geometry.head_dim;
    var middle_target_rows: usize = 0;
    if (geometry_targets) |targets| {
        if (targets.len != seeds.len) return error.InvalidRejectedHeadReplay;
        for (targets) |target| middle_target_rows = try std.math.add(usize, middle_target_rows, target);
    } else {
        const middle_rows = fixture.geometry.original_rows - fixture.geometry.prefix_rows - fixture.geometry.recent_rows;
        middle_target_rows = std.math.divCeil(usize, middle_rows, 2) catch return error.InvalidRejectedHeadReplay;
    }
    const target_rows = try std.math.add(usize, try std.math.add(usize, fixture.geometry.prefix_rows, middle_target_rows), fixture.geometry.recent_rows);
    if (target_rows != fixture.geometry.candidate_rows) return error.InvalidRejectedHeadReplay;
    const keys = try allocator.alloc(f64, target_rows * dim);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, target_rows * dim);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, target_rows);
    errdefer allocator.free(bias);
    const positions = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(positions);
    const coverage = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(coverage);
    const original = fixture.originalHead();
    const old_rows = original.rows - fixture.geometry.recent_rows;
    var out: usize = 0;
    const prefix: solver.Head = .{ .keys = original.keys[0 .. fixture.geometry.prefix_rows * dim], .values = original.values[0 .. fixture.geometry.prefix_rows * dim], .bias = original.bias.?[0..fixture.geometry.prefix_rows], .rows = fixture.geometry.prefix_rows, .dim = dim, .scale = original.scale, .softcap = original.softcap };
    try appendRows(keys, values, bias, positions, coverage, &out, prefix, fixture.original_positions[0..fixture.geometry.prefix_rows], fixture.original_coverage[0..fixture.geometry.prefix_rows]);

    var max_sweeps: usize = 0;
    var chunk_index: usize = 0;
    var start = fixture.geometry.prefix_rows;
    while (start < old_rows) : (chunk_index += 1) {
        if (chunk_index >= seeds.len) return error.InvalidRejectedHeadReplay;
        const rows = @min(fixture.geometry.chunk_rows, old_rows - start);
        const budget = if (geometry_targets) |targets| targets[chunk_index] else std.math.divCeil(usize, rows, 2) catch unreachable;
        if (budget == 0 or budget > rows or seeds[chunk_index].len >= budget) return error.InvalidRejectedHeadReplay;
        const source: solver.Head = .{ .keys = original.keys[start * dim ..][0 .. rows * dim], .values = original.values[start * dim ..][0 .. rows * dim], .bias = original.bias.?[start..][0..rows], .rows = rows, .dim = dim, .scale = original.scale, .softcap = original.softcap };
        if (geometry_targets != null and budget == rows) {
            try appendRows(keys, values, bias, positions, coverage, &out, source, fixture.original_positions[start..][0..rows], fixture.original_coverage[start..][0..rows]);
            start += rows;
            continue;
        }
        const cfg: solver.Config = .{ .target_rows = budget, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64), .backend = backend };
        const chunk_end = fixture.original_coverage[start + rows - 1];
        if (geometry_targets != null and seeds[chunk_index].len == 0) {
            var fitted = if (deferred_intermediate_training_checks)
                try solver.fitUnvalidatedDiagnostic(allocator, source, fixture.train, cfg)
            else
                try solver.fit(allocator, source, fixture.train, fixture.train, cfg);
            defer fitted.deinit();
            const fitted_positions = try allocator.alloc(u64, budget);
            defer allocator.free(fitted_positions);
            const fitted_coverage = try allocator.alloc(u64, budget);
            defer allocator.free(fitted_coverage);
            for (fitted.source_indices, fitted_positions, fitted_coverage) |source_index, *position, *coverage_end| {
                position.* = fixture.original_positions[start + source_index];
                coverage_end.* = chunk_end;
            }
            max_sweeps = @max(max_sweeps, fitted.metrics.nnls_sweeps);
            try appendRows(keys, values, bias, positions, coverage, &out, fitted.head(source), fitted_positions, fitted_coverage);
            start += rows;
            continue;
        }
        var fitted = if (deferred_intermediate_training_checks)
            try exact.fitUnvalidatedDiagnostic(allocator, source, fixture.original_positions[start..][0..rows], seeds[chunk_index], chunk_end, fixture.train, budget, cfg)
        else blk: {
            const intermediate_validation = if (geometry_targets != null) fixture.train else fixture.heldout;
            break :blk try exact.fitDiagnostic(allocator, source, fixture.original_positions[start..][0..rows], seeds[chunk_index], chunk_end, fixture.train, intermediate_validation, budget, cfg);
        };
        defer fitted.deinit();
        for (fitted.coverage_ends) |value| if (value != chunk_end) return error.InvalidRejectedHeadReplay;
        max_sweeps = @max(max_sweeps, fitted.fit_metrics.nnls_sweeps);
        try appendRows(keys, values, bias, positions, coverage, &out, fitted.head(source), fitted.positions, fitted.coverage_ends);
        start += rows;
    }
    if (chunk_index != seeds.len) return error.InvalidRejectedHeadReplay;
    const recent: solver.Head = .{ .keys = original.keys[old_rows * dim ..], .values = original.values[old_rows * dim ..], .bias = original.bias.?[old_rows..], .rows = fixture.geometry.recent_rows, .dim = dim, .scale = original.scale, .softcap = original.softcap };
    try appendRows(keys, values, bias, positions, coverage, &out, recent, fixture.original_positions[old_rows..], fixture.original_coverage[old_rows..]);
    if (out != target_rows) return error.InvalidRejectedHeadReplay;
    if (!std.mem.eql(u64, positions[0..fixture.geometry.prefix_rows], fixture.original_positions[0..fixture.geometry.prefix_rows]) or
        !std.mem.eql(u64, coverage[0..fixture.geometry.prefix_rows], fixture.original_coverage[0..fixture.geometry.prefix_rows]) or
        !std.mem.eql(u64, positions[target_rows - fixture.geometry.recent_rows ..], fixture.original_positions[old_rows..]) or
        !std.mem.eql(u64, coverage[target_rows - fixture.geometry.recent_rows ..], fixture.original_coverage[old_rows..]))
        return error.InvalidRejectedHeadReplay;
    for (positions, coverage, 0..) |position, coverage_end, index| {
        if (coverage_end <= position or coverage_end > fixture.geometry.logical_length or
            (index != 0 and (position <= positions[index - 1] or coverage_end < coverage[index - 1])))
            return error.InvalidRejectedHeadReplay;
    }
    try roundF16(keys);
    try roundF16(values);
    try roundF32(bias);
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias, .positions = positions, .coverage = coverage, .accelerated = backend != null, .max_nnls_sweeps = max_sweeps, .elapsed_ns = 0 };
}

fn validateCandidate(allocator: std.mem.Allocator, original: solver.Head, candidate: OwnedCandidate, heldout: []const f64) !Cell {
    const strict: solver.Config = .{ .target_rows = candidate.bias.len, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 };
    const quality = try solver.validationMetrics(allocator, original, candidate.head(original), heldout, strict);
    const accepted = quality.max_relative_mass_error <= strict.max_relative_mass_error and quality.relative_output_rmse <= strict.max_relative_output_rmse;
    return .{ .accelerated = candidate.accelerated, .accepted = accepted, .elapsed_ns = candidate.elapsed_ns, .accelerated_value_solves = 0, .max_nnls_sweeps = candidate.max_nnls_sweeps, .mass_error = quality.max_relative_mass_error, .output_rmse = quality.relative_output_rmse };
}

fn candidateDigest(candidate: OwnedCandidate) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.d5h1-rounded-candidate.v1\x00");
    hash.update(std.mem.sliceAsBytes(candidate.keys));
    hash.update(std.mem.sliceAsBytes(candidate.values));
    hash.update(std.mem.sliceAsBytes(candidate.bias));
    hash.update(std.mem.sliceAsBytes(candidate.positions));
    hash.update(std.mem.sliceAsBytes(candidate.coverage));
    return hash.finalResult();
}

fn candidateKvbDigest(candidate: OwnedCandidate) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.d5h1-rounded-candidate-kvb.v1\x00");
    hash.update(std.mem.sliceAsBytes(candidate.keys));
    hash.update(std.mem.sliceAsBytes(candidate.values));
    hash.update(std.mem.sliceAsBytes(candidate.bias));
    return hash.finalResult();
}

fn runReplay(plan_path: []const u8) !void {
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(Plan, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer parsed.deinit();
    const plan = parsed.value;
    const anchored_calibrated = std.mem.eql(u8, plan.experiment_schema, "antfly.d5h1-train-exact-core-residual-calibration-replay-plan.v2");
    const minimax_calibrated = std.mem.eql(u8, plan.experiment_schema, "antfly.d5h1-train-exact-core-residual-global-minimax-replay-plan.v3");
    const calibrated = anchored_calibrated or minimax_calibrated;
    const risk = calibrated or std.mem.eql(u8, plan.experiment_schema, "antfly.d5h1-train-exact-core-residual-replay-plan.v1");
    const legacy = std.mem.eql(u8, plan.experiment_schema, "antfly.source64-exact-peak-residual-replay-plan.v1");
    const policy_ok = if (anchored_calibrated) plan.residual_group_calibration_policy == .anchored_l2_v1 or plan.residual_group_calibration_policy == .disabled else if (minimax_calibrated) plan.residual_group_calibration_policy == .global_minimax_irls_p4_p8_p16_v1 else plan.residual_group_calibration_policy == .disabled;
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-guarded-diagnostic-test-plan.v3") or (!legacy and !risk) or plan.residual_group_mass_calibration != calibrated or !policy_ok or (plan.diagnostic_phase_profiling and (!risk or calibrated)) or (plan.diagnostic_query_batch_rows == .rows64 and !plan.diagnostic_phase_profiling) or (plan.deferred_intermediate_training_checks and (!risk or calibrated or !plan.diagnostic_phase_profiling)) or
        !std.fs.path.isAbsolute(plan.fixture_dir) or plan.prefix_rows != 128 or plan.recent_rows != 2048 or plan.retention_divisor != 2 or
        (legacy and (plan.target_rows != 5056 or plan.chunk_rows != 1024 or plan.mandatory_training_top_k != 2)) or
        (risk and (plan.target_rows != 4908 or plan.chunk_rows != 256 or plan.geometry_targets == null or plan.exact_core_source_keys == null or plan.risk_report_path == null or plan.risk_report_sha256 == null or plan.result_path == null or !std.fs.path.isAbsolute(plan.risk_report_path.?) or !std.fs.path.isAbsolute(plan.result_path.?))))
        return error.InvalidRejectedHeadReplayPlan;
    if (risk) {
        const risk_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan.risk_report_path.?, allocator, .limited(1024 * 1024));
        defer allocator.free(risk_bytes);
        var actual_risk_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(risk_bytes, &actual_risk_digest, .{});
        const expected_risk_digest = try digest(plan.risk_report_sha256.?);
        if (!std.mem.eql(u8, &actual_risk_digest, &expected_risk_digest)) return error.InvalidRejectedHeadReplayPlan;
        const risk_parsed = try std.json.parseFromSlice(RiskReport, allocator, risk_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
        defer risk_parsed.deinit();
        const report = risk_parsed.value;
        if (!std.mem.eql(u8, report.schema, "antfly.d5h1-train-risk-budget-audit.v1") or !report.authenticated or report.donor_layer != 5 or report.kv_head != 1 or report.geometry_divisor != 2 or report.heldout_query_values_read or report.ranges.len != plan.geometry_targets.?.len or report.ranges.len != plan.exact_core_source_keys.?.len) return error.InvalidRejectedHeadReplayPlan;
        for (report.ranges, plan.geometry_targets.?, plan.exact_core_source_keys.?) |row, target, cores| {
            if (row.target_budget != target or !std.mem.eql(usize, row.exact_core_source_keys, cores)) return error.InvalidRejectedHeadReplayPlan;
        }
    }
    const identity: checkpoint.Identity = .{ .weights = try digest(plan.identity.weights), .tokenizer = try digest(plan.identity.tokenizer), .template = try digest(plan.identity.template), .layout = try digest(plan.identity.layout) };
    _ = try digest(plan.parent_fixture_manifest_sha256);
    _ = try digest(plan.parent_replay_plan_sha256);
    const binding: fixture_mod.Binding = .{ .output_dir = plan.fixture_dir, .identity = identity, .config_sha256 = try digest(plan.config_sha256) };
    var fixture = if (risk) try fixture_mod.readTrainingOnly(allocator, io, plan.fixture_dir, binding) else try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
    defer fixture.deinit();
    if (fixture.geometry.original_rows != 7936 or fixture.geometry.candidate_rows != plan.target_rows or fixture.geometry.head_dim != 512 or
        fixture.geometry.prefix_rows != 128 or fixture.geometry.recent_rows != 2048 or fixture.geometry.chunk_rows != plan.chunk_rows or fixture.geometry.retention_divisor != 2)
        return error.InvalidRejectedHeadReplayPlan;
    var manifest_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_hash, .{});
    const expected_manifest_hash = try digest(plan.fixture_manifest_sha256);
    if (!std.mem.eql(u8, &manifest_hash, &expected_manifest_hash)) return error.InvalidRejectedHeadReplayPlan;

    const train_rows = fixture.train.len / fixture.geometry.head_dim;
    if (train_rows == 0 or train_rows * fixture.geometry.head_dim != fixture.train.len) return error.InvalidRejectedHeadReplay;
    const old_rows = fixture.geometry.original_rows - fixture.geometry.recent_rows;
    var seed_storage: [23][]usize = undefined;
    var seed_views: [23][]const usize = undefined;
    var seed_count: usize = 0;
    defer for (seed_storage[0..seed_count]) |seed| allocator.free(seed);
    var start = fixture.geometry.prefix_rows;
    while (start < old_rows) : (seed_count += 1) {
        if (seed_count >= seed_storage.len) return error.InvalidRejectedHeadReplay;
        const rows = @min(fixture.geometry.chunk_rows, old_rows - start);
        if (risk) {
            const targets = plan.geometry_targets.?;
            const cores = plan.exact_core_source_keys.?;
            if (targets.len != cores.len or seed_count >= targets.len or cores[seed_count].len >= targets[seed_count]) return error.InvalidRejectedHeadReplay;
            seed_storage[seed_count] = try allocator.alloc(usize, cores[seed_count].len);
            for (cores[seed_count], seed_storage[seed_count]) |global, *local| {
                if (global < start or global >= start + rows) return error.InvalidRejectedHeadReplay;
                local.* = global - start;
            }
            for (seed_storage[seed_count], 0..) |local, i| if (std.mem.indexOfScalar(usize, seed_storage[seed_count][0..i], local) != null) return error.InvalidRejectedHeadReplay;
            seed_views[seed_count] = seed_storage[seed_count];
            std.debug.print("attention_match_exact_train_core chunk_start={d} rows={d} exact_core={d} budget={d} residual_budget={d}\n", .{ start, rows, cores[seed_count].len, targets[seed_count], targets[seed_count] - cores[seed_count].len });
            start += rows;
            continue;
        }
        const source: solver.Head = .{ .keys = fixture.original_keys[start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim], .values = fixture.original_values[start * fixture.geometry.head_dim ..][0 .. rows * fixture.geometry.head_dim], .bias = fixture.original_bias[start..][0..rows], .rows = rows, .dim = fixture.geometry.head_dim, .scale = fixture.geometry.scale, .softcap = fixture.geometry.softcap };
        const probabilities = try allocator.alloc(f64, train_rows * rows);
        defer allocator.free(probabilities);
        try solver.attendBatch(allocator, source, fixture.train, probabilities, null, null, .{ .target_rows = 1 });
        const all = try allocator.alloc(usize, rows);
        defer allocator.free(all);
        const count = try solver.mandatoryTrainingTopKSeeds(probabilities, train_rows, rows, 2, all);
        const budget = std.math.divCeil(usize, rows, 2) catch unreachable;
        if (count == 0 or count >= budget) return error.InvalidRejectedHeadReplay;
        seed_storage[seed_count] = try allocator.dupe(usize, all[0..count]);
        seed_views[seed_count] = seed_storage[seed_count];
        std.debug.print("attention_match_exact_peak_reference chunk_start={d} rows={d} seeds={d} budget={d}\n", .{ start, rows, count, budget });
        start += rows;
    }
    const expected_seed_count: usize = if (risk) 23 else 6;
    if (seed_count != expected_seed_count) return error.InvalidRejectedHeadReplay;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    const query_batch_rows: usize = switch (plan.diagnostic_query_batch_rows) {
        .rows16 => 16,
        .rows64 => 64,
    };
    var baseline_profile: Fitter.PhaseCounters = .{};
    var exact_profile: Fitter.PhaseCounters = .{};
    var baseline_candidate: ?OwnedCandidate = null;
    defer if (baseline_candidate) |*candidate| candidate.deinit();
    var baseline_solves: u64 = 0;
    if (risk) {
        var baseline_fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
        var baseline_fitter_live = true;
        defer if (baseline_fitter_live) baseline_fitter.deinit();
        try baseline_fitter.setQueryBatchRows(query_batch_rows);
        if (baseline_fitter.queryBatchRows() != query_batch_rows) return error.InvalidRejectedHeadReplay;
        if (plan.diagnostic_phase_profiling) try baseline_fitter.setPhaseProfiling(true);
        var baseline_seeds: [23][]const usize = undefined;
        for (&baseline_seeds) |*seed| seed.* = &.{};
        const begin = platform.time.monotonicNs();
        baseline_candidate = try buildCandidate(allocator, &fixture, &baseline_seeds, plan.geometry_targets, baseline_fitter.backend(), plan.deferred_intermediate_training_checks);
        baseline_candidate.?.elapsed_ns = platform.time.monotonicNs() -| begin;
        if (baseline_candidate.?.elapsed_ns == 0) return error.InvalidRejectedHeadReplay;
        if (plan.diagnostic_phase_profiling) baseline_profile = baseline_fitter.phaseCounters();
        baseline_solves = baseline_fitter.deinitAndTakeValueSolves();
        baseline_fitter_live = false;
        if (baseline_solves == 0) return error.InvalidRejectedHeadReplay;
    }
    // A fresh fitter prevents the baseline from warming state or contaminating
    // the exact-core accelerated solve counter/timing.
    var fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    var live = true;
    defer if (live) fitter.deinit();
    try fitter.setQueryBatchRows(query_batch_rows);
    if (fitter.queryBatchRows() != query_batch_rows) return error.InvalidRejectedHeadReplay;
    if (plan.diagnostic_phase_profiling) try fitter.setPhaseProfiling(true);
    var candidates: [2]?OwnedCandidate = .{ null, null };
    defer for (&candidates) |*optional| if (optional.*) |*candidate| candidate.deinit();
    for ([_]?solver.Backend{ null, fitter.backend() }, 0..) |backend, index| {
        const begin = platform.time.monotonicNs();
        candidates[index] = try buildCandidate(allocator, &fixture, seed_views[0..seed_count], if (risk) plan.geometry_targets else null, backend, plan.deferred_intermediate_training_checks);
        candidates[index].?.elapsed_ns = platform.time.monotonicNs() -| begin;
        if (candidates[index].?.elapsed_ns == 0) return error.InvalidRejectedHeadReplay;
    }
    if (plan.diagnostic_phase_profiling) exact_profile = fitter.phaseCounters();
    const solves = fitter.deinitAndTakeValueSolves();
    live = false;
    if (candidates[0].?.accelerated or !candidates[1].?.accelerated or solves == 0) return error.InvalidRejectedHeadReplay;
    var calibrated_candidate: ?OwnedCandidate = null;
    defer if (calibrated_candidate) |*candidate| candidate.deinit();
    var calibration_result: ?mass_calibration.Result = null;
    defer if (calibration_result) |*result| result.deinit();
    var minimax_result: ?mass_calibration.MinimaxResult = null;
    defer if (minimax_result) |*result| result.deinit();
    var calibration_elapsed_ns: u64 = 0;
    if (calibrated) {
        const calibration_begin_ns = platform.time.monotonicNs();
        calibrated_candidate = try cloneCandidate(allocator, candidates[1].?);
        var ranges: [23]mass_calibration.Range = undefined;
        var range_start: usize = 128;
        for (&ranges, plan.exact_core_source_keys.?, 0..) |*range, exact_cores, index| {
            const rows: usize = if (index == 22) 128 else 256;
            range.* = .{ .start = range_start, .rows = rows, .exact_core_source_keys = exact_cores };
            range_start = try std.math.add(usize, range_start, rows);
        }
        if (range_start != 5888) return error.InvalidRejectedHeadReplay;
        const original_head = fixture.originalHead();
        const calibrated_head = calibrated_candidate.?.head(original_head);
        if (minimax_calibrated) {
            minimax_result = try mass_calibration.calibrateTrainGlobalMinimax(allocator, original_head, fixture.original_positions, calibrated_head, calibrated_candidate.?.positions, &ranges, fixture.train, calibrated_candidate.?.bias, .{ .target_rows = calibrated_candidate.?.bias.len, .mass_solver = .active_set, .max_workspace_bytes = 1024 * 1024 * 1024 });
            calibrated_candidate.?.max_nnls_sweeps = @max(calibrated_candidate.?.max_nnls_sweeps, minimax_result.?.nnls_sweeps);
        } else {
            calibration_result = try mass_calibration.calibrateTrain(allocator, original_head, fixture.original_positions, calibrated_head, calibrated_candidate.?.positions, &ranges, fixture.train, calibrated_candidate.?.bias, .{ .target_rows = calibrated_candidate.?.bias.len, .mass_solver = .active_set, .max_workspace_bytes = 1024 * 1024 * 1024 });
            calibrated_candidate.?.max_nnls_sweeps = @max(calibrated_candidate.?.max_nnls_sweeps, calibration_result.?.nnls_sweeps);
        }
        if (!std.mem.eql(f64, calibrated_candidate.?.keys, candidates[1].?.keys) or !std.mem.eql(f64, calibrated_candidate.?.values, candidates[1].?.values) or !std.mem.eql(u64, calibrated_candidate.?.positions, candidates[1].?.positions) or !std.mem.eql(u64, calibrated_candidate.?.coverage, candidates[1].?.coverage)) return error.InvalidRejectedHeadReplay;
        const groups = try mass_calibration.deriveCandidateGroups(allocator, fixture.original_positions, calibrated_candidate.?.positions, &ranges);
        defer allocator.free(groups);
        for (groups, calibrated_candidate.?.bias, candidates[1].?.bias) |group, after, before| if (group == mass_calibration.fixed_group and after != before) return error.InvalidRejectedHeadReplay;
        calibration_elapsed_ns = platform.time.monotonicNs() -| calibration_begin_ns;
        calibrated_candidate.?.elapsed_ns = try std.math.add(u64, calibrated_candidate.?.elapsed_ns, calibration_elapsed_ns);
    }
    const baseline_digest = if (baseline_candidate) |candidate| candidateDigest(candidate) else @as([32]u8, @splat(0));
    const candidate_digests = [2][32]u8{ candidateDigest(candidates[0].?), candidateDigest(candidates[1].?) };
    const calibrated_digest = if (calibrated_candidate) |candidate| candidateDigest(candidate) else @as([32]u8, @splat(0));
    const baseline_kvb_digest = if (baseline_candidate) |candidate| candidateKvbDigest(candidate) else @as([32]u8, @splat(0));
    const candidate_kvb_digests = [2][32]u8{ candidateKvbDigest(candidates[0].?), candidateKvbDigest(candidates[1].?) };
    const calibrated_kvb_digest = if (calibrated_candidate) |candidate| candidateKvbDigest(candidate) else @as([32]u8, @splat(0));
    var validation_fixture: ?fixture_mod.Fixture = null;
    defer if (validation_fixture) |*full| full.deinit();
    const validation_source: *const fixture_mod.Fixture = if (risk) blk: {
        validation_fixture = try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
        if (!std.mem.eql(u8, validation_fixture.?.manifest_bytes, fixture.manifest_bytes) or validation_fixture.?.heldout.len == 0 or fixture.heldout.len != 0) return error.InvalidRejectedHeadReplay;
        break :blk &validation_fixture.?;
    } else &fixture;
    var cells: [2]Cell = undefined;
    var baseline_result: ?Cell = null;
    if (baseline_candidate) |candidate| {
        baseline_result = try validateCandidate(allocator, validation_source.originalHead(), candidate, validation_source.heldout);
        baseline_result.?.accelerated_value_solves = baseline_solves;
    }
    for (&cells, &candidates) |*cell, candidate| cell.* = try validateCandidate(allocator, validation_source.originalHead(), candidate.?, validation_source.heldout);
    cells[1].accelerated_value_solves = solves;
    var calibrated_cell: ?Cell = null;
    if (calibrated_candidate) |candidate| {
        calibrated_cell = try validateCandidate(allocator, validation_source.originalHead(), candidate, validation_source.heldout);
        calibrated_cell.?.accelerated_value_solves = 0;
    }
    if (baseline_result) |baseline| std.debug.print("attention_match_exact_peak_residual_baseline accelerated=true accepted={} target_rows={d} elapsed_ns={d} accelerated_value_solves={d} mass_error={d} output_rmse={d} candidate_sha256={s} same_frozen_geometry=1 exact_core_retention=0 heldout_controls_fit=0 heldout_opened_after_candidate_freeze=1 diagnostic_only=1 quality_credit=0\n", .{ baseline.accepted, plan.target_rows, baseline.elapsed_ns, baseline.accelerated_value_solves, baseline.mass_error, baseline.output_rmse, std.fmt.bytesToHex(baseline_digest, .lower) });
    if (minimax_calibrated) {
        const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.d5h1-train-exact-core-residual-global-minimax-result.v5", .authenticated = true, .risk_report_sha256 = plan.risk_report_sha256.?, .fixture_manifest_sha256 = plan.fixture_manifest_sha256, .target_rows = plan.target_rows, .geometry_targets = plan.geometry_targets.?, .exact_core_source_keys = plan.exact_core_source_keys.?, .residual_group_calibration_policy = plan.residual_group_calibration_policy, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .minimax_accelerated_candidate_sha256 = std.fmt.bytesToHex(calibrated_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb_digest, .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[0], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[1], .lower), .minimax_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(calibrated_kvb_digest, .lower), .candidate_keys_values_positions_coverage_unchanged = true, .candidates_frozen_before_heldout_open = true, .calibration_group_count = 23, .calibration_alpha = minimax_result.?.alpha, .calibration_nnls_sweeps = minimax_result.?.nnls_sweeps, .calibration_backend = "cpu_scalar_fp64_global_minimax_irls", .calibration_value_solves = 0, .calibration_elapsed_ns = calibration_elapsed_ns, .train_mass_diagnostics = minimax_result.?.diagnostics, .baseline = baseline_result.?, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .minimax_accelerated = calibrated_cell.?, .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
        defer allocator.free(body);
        var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path.?, .{ .exclusive = true, .truncate = false });
        defer result_file.close(io);
        try result_file.writeStreamingAll(io, body);
        try result_file.writeStreamingAll(io, "\n");
        try result_file.sync(io);
    } else if (calibrated) {
        const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.d5h1-train-exact-core-residual-calibration-result.v4", .authenticated = true, .risk_report_sha256 = plan.risk_report_sha256.?, .fixture_manifest_sha256 = plan.fixture_manifest_sha256, .target_rows = plan.target_rows, .geometry_targets = plan.geometry_targets.?, .exact_core_source_keys = plan.exact_core_source_keys.?, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .calibrated_accelerated_candidate_sha256 = std.fmt.bytesToHex(calibrated_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb_digest, .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[0], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[1], .lower), .calibrated_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(calibrated_kvb_digest, .lower), .candidate_keys_values_positions_coverage_unchanged = true, .candidates_frozen_before_heldout_open = true, .calibration_group_count = 23, .calibration_alpha = calibration_result.?.alpha, .calibration_nnls_sweeps = calibration_result.?.nnls_sweeps, .calibration_backend = "cpu_scalar_fp64_anchored_nnls", .calibration_value_solves = 0, .calibration_elapsed_ns = calibration_elapsed_ns, .train_mass_diagnostics = calibration_result.?.diagnostics, .baseline = baseline_result.?, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .calibrated_accelerated = calibrated_cell.?, .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
        defer allocator.free(body);
        var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path.?, .{ .exclusive = true, .truncate = false });
        defer result_file.close(io);
        try result_file.writeStreamingAll(io, body);
        try result_file.writeStreamingAll(io, "\n");
        try result_file.sync(io);
    } else if (risk) {
        const body = if (plan.diagnostic_phase_profiling)
            try std.json.Stringify.valueAlloc(allocator, .{ .schema = if (plan.deferred_intermediate_training_checks) "antfly.d5h1-train-exact-core-residual-profile-result.v7" else "antfly.d5h1-train-exact-core-residual-profile-result.v6", .authenticated = true, .risk_report_sha256 = plan.risk_report_sha256.?, .fixture_manifest_sha256 = plan.fixture_manifest_sha256, .target_rows = plan.target_rows, .geometry_targets = plan.geometry_targets.?, .exact_core_source_keys = plan.exact_core_source_keys.?, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb_digest, .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[0], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[1], .lower), .baseline_and_exact_accelerators_fresh_isolated = true, .candidates_frozen_before_heldout_open = true, .phase_profile = .{ .query_batch_rows = query_batch_rows, .baseline = baseline_profile, .exact = exact_profile, .baseline_head_wall_ns = baseline_candidate.?.elapsed_ns, .exact_cpu_head_wall_ns = candidates[0].?.elapsed_ns, .exact_accelerated_head_wall_ns = candidates[1].?.elapsed_ns }, .baseline = baseline_result.?, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{})
        else
            try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.d5h1-train-exact-core-residual-result.v3", .authenticated = true, .risk_report_sha256 = plan.risk_report_sha256.?, .fixture_manifest_sha256 = plan.fixture_manifest_sha256, .target_rows = plan.target_rows, .geometry_targets = plan.geometry_targets.?, .exact_core_source_keys = plan.exact_core_source_keys.?, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb_digest, .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[0], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb_digests[1], .lower), .baseline_and_exact_accelerators_fresh_isolated = true, .candidates_frozen_before_heldout_open = true, .baseline = baseline_result.?, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
        defer allocator.free(body);
        var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path.?, .{ .exclusive = true, .truncate = false });
        defer result_file.close(io);
        try result_file.writeStreamingAll(io, body);
        try result_file.writeStreamingAll(io, "\n");
        try result_file.sync(io);
    }
    for (cells) |cell| std.debug.print("attention_match_exact_peak_residual accelerated={} accepted={} target_rows={d} risk_report_sha256={s} elapsed_ns={d} accelerated_value_solves={d} max_nnls_sweeps={d} mass_error={d} output_rmse={d} backend_null_fp64_authorization=1 train_only_core_freeze={} heldout_controls_fit=0 diagnostic_only=1 quality_credit=0 production_admission_claim=0\n", .{ cell.accelerated, cell.accepted, plan.target_rows, plan.risk_report_sha256 orelse "legacy", cell.elapsed_ns, cell.accelerated_value_solves, cell.max_nnls_sweeps, cell.mass_error, cell.output_rmse, risk });
}

fn sameOriginal(a: *const fixture_mod.Fixture, b: *const fixture_mod.Fixture) bool {
    return std.meta.eql(a.geometry, b.geometry) and std.mem.eql(f64, a.original_keys, b.original_keys) and
        std.mem.eql(f64, a.original_values, b.original_values) and std.mem.eql(f64, a.original_bias, b.original_bias) and
        std.mem.eql(u64, a.original_positions, b.original_positions) and std.mem.eql(u64, a.original_coverage, b.original_coverage);
}

fn validateShardRoles(fixture: *const fixture_mod.Fixture, expected: []const ConsumerTuple, seen: []u64, seen_count: *usize) !void {
    if (expected.len == 0 or expected.len > 4 or fixture.query_consumer_layers.len != expected.len or fixture.query_heads.len != expected.len or fixture.query_train_counts.len != expected.len or fixture.query_heldout_counts.len != expected.len) return error.InvalidRejectedHeadReplayPlan;
    for (expected, 0..) |tuple, i| {
        if (fixture.query_consumer_layers[i] != tuple.layer or fixture.query_heads[i] != tuple.query_head or fixture.query_train_counts[i] != 960 or fixture.query_heldout_counts[i] != 64) return error.InvalidRejectedHeadReplayPlan;
        const key = (@as(u64, tuple.layer) << 32) | tuple.query_head;
        if ((seen_count.* > 0 and key <= seen[seen_count.* - 1]) or std.mem.indexOfScalar(u64, seen[0..seen_count.*], key) != null or seen_count.* == seen.len) return error.InvalidRejectedHeadReplayPlan;
        seen[seen_count.*] = key;
        seen_count.* += 1;
    }
}

fn validateCompleteConsumerMatrix(donor_layer: u32, kv_head: usize, shard_count: usize, seen: []const u64) !void {
    const expected_layers = if (donor_layer == 23) [_]u32{ 23, 29, 35, 41 } else [_]u32{ donor_layer, 0, 0, 0 };
    const layer_count: usize = if (donor_layer == 23) 4 else 1;
    if (shard_count != layer_count or seen.len != layer_count * 4) return error.InvalidRejectedHeadReplayPlan;
    var index: usize = 0;
    for (expected_layers[0..layer_count]) |layer| {
        for (0..4) |offset| {
            const query_head = kv_head * 4 + offset;
            const expected = (@as(u64, layer) << 32) | @as(u64, @intCast(query_head));
            if (seen[index] != expected) return error.InvalidRejectedHeadReplayPlan;
            index += 1;
        }
    }
}

fn verifySidecar(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected: []const u8) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    const wanted = try digest(expected);
    if (!std.mem.eql(u8, &actual, &wanted)) return error.InvalidRejectedHeadReplayPlan;
}

fn shardedConstructionPolicyValid(schema: []const u8, deferred: bool, profiling: bool, batch: DiagnosticQueryBatchRows) bool {
    if (std.mem.eql(u8, schema, "antfly.all-head-train-exact-core-sharded-replay-plan.v2"))
        return !deferred and !profiling and batch == .rows16;
    if (std.mem.eql(u8, schema, "antfly.all-head-train-exact-core-sharded-replay-plan.v3"))
        return deferred and profiling;
    if (std.mem.eql(u8, schema, "antfly.all-head-joint-mass-bias-sharded-replay-plan.v4"))
        return deferred and profiling and batch == .rows64;
    if (std.mem.eql(u8, schema, "antfly.all-head-joint-mass-bias-sharded-replay-plan.v5"))
        return deferred and profiling and batch == .rows64;
    return false;
}

fn runShardedReplay(plan_path: []const u8) !void {
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const plan_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(plan_bytes);
    const parsed = try std.json.parseFromSlice(ShardedPlan, allocator, plan_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer parsed.deinit();
    const plan = parsed.value;
    const deferred_v3 = std.mem.eql(u8, plan.experiment_schema, "antfly.all-head-train-exact-core-sharded-replay-plan.v3");
    const joint_v4 = std.mem.eql(u8, plan.experiment_schema, "antfly.all-head-joint-mass-bias-sharded-replay-plan.v4");
    const joint_v5 = std.mem.eql(u8, plan.experiment_schema, "antfly.all-head-joint-mass-bias-sharded-replay-plan.v5");
    const joint_generic = joint_v4 or joint_v5;
    const policy_ok = shardedConstructionPolicyValid(plan.experiment_schema, plan.deferred_intermediate_training_checks, plan.diagnostic_phase_profiling, plan.diagnostic_query_batch_rows);
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-guarded-diagnostic-test-plan.v3") or !policy_ok or
        !std.fs.path.isAbsolute(plan.result_path) or !std.fs.path.isAbsolute(plan.risk_report_path) or plan.fixture_shards.len == 0 or plan.fixture_shards.len > 4 or
        (plan.donor_layer != 5 and plan.donor_layer != 11 and plan.donor_layer != 17 and plan.donor_layer != 23) or plan.kv_head > 1 or
        plan.prefix_rows != 128 or plan.recent_rows != 2048 or plan.chunk_rows != 256 or plan.retention_divisor != 2 or plan.target_rows < 4907 or plan.target_rows > 4910 or
        plan.geometry_targets.len != 23 or plan.exact_core_source_keys.len != 23) return error.InvalidRejectedHeadReplayPlan;
    if (joint_generic) {
        if (plan.joint_value_backend_accelerate != joint_v5 or plan.joint_mass_bias_module_sha256 == null or !std.mem.eql(u8, plan.joint_mass_bias_module_sha256.?, "678dc54e903cf083e1ab354e2085555b2b299e6bb581b800dff3a12c5cf2cd57") or plan.joint_component_cap_mib == null) return error.InvalidRejectedHeadReplayPlan;
        const expected_cap: usize = if (plan.donor_layer == 23) 768 else 512;
        if (plan.joint_component_cap_mib.? != expected_cap) return error.InvalidRejectedHeadReplayPlan;
    }
    const identity: checkpoint.Identity = .{ .weights = try digest(plan.identity.weights), .tokenizer = try digest(plan.identity.tokenizer), .template = try digest(plan.identity.template), .layout = try digest(plan.identity.layout) };
    const config_sha = try digest(plan.config_sha256);

    const risk_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan.risk_report_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(risk_bytes);
    var risk_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(risk_bytes, &risk_hash, .{});
    const expected_risk_hash = try digest(plan.risk_report_sha256);
    if (!std.mem.eql(u8, &risk_hash, &expected_risk_hash)) return error.InvalidRejectedHeadReplayPlan;
    const risk_parsed = try std.json.parseFromSlice(GenericRiskReport, allocator, risk_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer risk_parsed.deinit();
    const risk = risk_parsed.value;
    if (!std.mem.eql(u8, risk.schema, "antfly.train-risk-budget-audit.v2") or !risk.authenticated or plan.risk_rule.len == 0 or !std.mem.eql(u8, risk.risk_rule, plan.risk_rule) or risk.donor_layer != plan.donor_layer or risk.kv_head != plan.kv_head or risk.geometry_divisor != 2 or risk.original_source_rows != 7936 or !risk.original_source_positions_contiguous or
        !std.mem.eql(u8, risk.train_scope, "all960_primary256_historical448_future256_per_consumer") or risk.heldout_query_values_read or risk.ranges.len != 23) return error.InvalidRejectedHeadReplayPlan;
    var core_total: usize = 0;
    var middle_target_total: usize = 0;
    for (risk.ranges, plan.geometry_targets, plan.exact_core_source_keys, 0..) |row, target, cores, index| {
        const start = 128 + index * 256;
        const rows = @min(@as(usize, 256), 5888 - start);
        if (target == 0 or target > rows or row.range_index != index or row.start != start or row.rows != rows or row.target_budget != target or row.exact_core_count != cores.len or !std.mem.eql(usize, row.exact_core_source_keys, cores) or cores.len >= target) return error.InvalidRejectedHeadReplayPlan;
        for (cores, 0..) |key, i| if (key < start or key >= start + rows or std.mem.indexOfScalar(usize, cores[0..i], key) != null) return error.InvalidRejectedHeadReplayPlan;
        core_total = try std.math.add(usize, core_total, cores.len);
        middle_target_total = try std.math.add(usize, middle_target_total, target);
    }
    if (core_total > risk.distinct_exact_core_source_keys or plan.target_rows != try std.math.add(usize, 2176, middle_target_total)) return error.InvalidRejectedHeadReplayPlan;

    const max_train_rows: usize = 15360;
    const dim: usize = 512;
    const train_storage = try allocator.alloc(f64, max_train_rows * dim);
    defer allocator.free(train_storage);
    var train_elements: usize = 0;
    var first: ?fixture_mod.Fixture = null;
    defer if (first) |*fixture| fixture.deinit();
    var seen: [16]u64 = undefined;
    var seen_count: usize = 0;
    var phase_manifest_hashes: [4][32]u8 = undefined;
    for (plan.fixture_shards, 0..) |shard, shard_index| {
        if (!std.fs.path.isAbsolute(shard.fixture_dir)) return error.InvalidRejectedHeadReplayPlan;
        const provenance_path = try std.fs.path.join(allocator, &.{ shard.fixture_dir, "provenance.json" });
        defer allocator.free(provenance_path);
        try verifySidecar(allocator, io, provenance_path, shard.fixture_provenance_sha256);
        const binding: fixture_mod.Binding = .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config_sha };
        var fixture = try fixture_mod.readTrainingOnly(allocator, io, shard.fixture_dir, binding);
        var moved = false;
        defer if (!moved) fixture.deinit();
        std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &phase_manifest_hashes[shard_index], .{});
        const expected_manifest = try digest(shard.fixture_manifest_sha256);
        if (!std.mem.eql(u8, &phase_manifest_hashes[shard_index], &expected_manifest) or fixture.geometry.donor_layer != plan.donor_layer or fixture.geometry.kv_head != plan.kv_head or fixture.geometry.original_rows != 7936 or fixture.geometry.candidate_rows != plan.target_rows or fixture.geometry.head_dim != dim or fixture.geometry.prefix_rows != 128 or fixture.geometry.recent_rows != 2048 or fixture.geometry.chunk_rows != 256 or fixture.geometry.scale != 1.0 / @sqrt(@as(f64, 512)) or fixture.geometry.softcap != 0.0) return error.InvalidRejectedHeadReplayPlan;
        try validateShardRoles(&fixture, shard.consumers, &seen, &seen_count);
        for (shard.consumers) |tuple| {
            if (tuple.query_head / 4 != plan.kv_head or (plan.donor_layer == 23 and tuple.layer != 23 and tuple.layer != 29 and tuple.layer != 35 and tuple.layer != 41) or (plan.donor_layer != 23 and tuple.layer != plan.donor_layer)) return error.InvalidRejectedHeadReplayPlan;
        }
        if (first) |*reference| {
            if (!sameOriginal(reference, &fixture)) return error.InvalidRejectedHeadReplayPlan;
        } else {
            first = fixture;
            moved = true;
        }
        if (train_elements + fixture.train.len > train_storage.len) return error.CompactionWorkspaceLimit;
        @memcpy(train_storage[train_elements..][0..fixture.train.len], fixture.train);
        train_elements += fixture.train.len;
    }
    try validateCompleteConsumerMatrix(plan.donor_layer, plan.kv_head, plan.fixture_shards.len, seen[0..seen_count]);
    if (risk.train_rows_total != seen_count * 960 or train_elements != risk.train_rows_total * dim) return error.InvalidRejectedHeadReplayPlan;
    var combined = first.?;
    combined.train = train_storage[0..train_elements];

    var seed_storage: [23][]usize = undefined;
    var seed_views: [23][]const usize = undefined;
    var allocated_seed_count: usize = 0;
    defer for (seed_storage[0..allocated_seed_count]) |seed| allocator.free(seed);
    for (plan.exact_core_source_keys, 0..) |cores, index| {
        const start = 128 + index * 256;
        seed_storage[index] = try allocator.alloc(usize, cores.len);
        allocated_seed_count += 1;
        for (cores, seed_storage[index]) |key, *local| local.* = key - start;
        seed_views[index] = seed_storage[index];
    }
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    const query_batch_rows: usize = switch (plan.diagnostic_query_batch_rows) {
        .rows16 => 16,
        .rows64 => 64,
    };
    var baseline_profile: Fitter.PhaseCounters = .{};
    var exact_profile: Fitter.PhaseCounters = .{};
    var baseline_candidate: ?OwnedCandidate = null;
    defer if (baseline_candidate) |*candidate| candidate.deinit();
    var baseline_seeds: [23][]const usize = undefined;
    for (&baseline_seeds, 0..) |*seed, index| seed.* = if (joint_generic) seed_views[index] else &.{};
    var baseline_fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    var baseline_live = true;
    defer if (baseline_live) baseline_fitter.deinit();
    if (deferred_v3 or joint_generic) try baseline_fitter.setQueryBatchRows(query_batch_rows);
    if ((deferred_v3 or joint_generic) and baseline_fitter.queryBatchRows() != query_batch_rows) return error.InvalidRejectedHeadReplay;
    if (plan.diagnostic_phase_profiling) try baseline_fitter.setPhaseProfiling(true);
    var begin = platform.time.monotonicNs();
    baseline_candidate = try buildCandidate(allocator, &combined, &baseline_seeds, plan.geometry_targets, baseline_fitter.backend(), plan.deferred_intermediate_training_checks);
    baseline_candidate.?.elapsed_ns = platform.time.monotonicNs() -| begin;
    if (plan.diagnostic_phase_profiling) baseline_profile = baseline_fitter.phaseCounters();
    const baseline_solves = baseline_fitter.deinitAndTakeValueSolves();
    baseline_live = false;
    if (joint_generic) return finishShardedJoint(plan, joint_v5, allocator, io, identity, config_sha, &combined, &first.?, baseline_candidate.?, baseline_profile, baseline_solves, phase_manifest_hashes, seen[0..seen_count]);
    var fitter = try Fitter.init(allocator, null, 1024 * 1024 * 1024);
    var fitter_live = true;
    defer if (fitter_live) fitter.deinit();
    if (deferred_v3) try fitter.setQueryBatchRows(query_batch_rows);
    if (deferred_v3 and fitter.queryBatchRows() != query_batch_rows) return error.InvalidRejectedHeadReplay;
    if (plan.diagnostic_phase_profiling) try fitter.setPhaseProfiling(true);
    var candidates: [2]?OwnedCandidate = .{ null, null };
    defer for (&candidates) |*optional| if (optional.*) |*candidate| candidate.deinit();
    for ([_]?solver.Backend{ null, fitter.backend() }, 0..) |backend, index| {
        begin = platform.time.monotonicNs();
        candidates[index] = try buildCandidate(allocator, &combined, &seed_views, plan.geometry_targets, backend, plan.deferred_intermediate_training_checks);
        candidates[index].?.elapsed_ns = platform.time.monotonicNs() -| begin;
    }
    if (plan.diagnostic_phase_profiling) exact_profile = fitter.phaseCounters();
    const exact_solves = fitter.deinitAndTakeValueSolves();
    fitter_live = false;
    if (baseline_solves == 0 or exact_solves == 0) return error.InvalidRejectedHeadReplay;
    const baseline_digest = candidateDigest(baseline_candidate.?);
    const baseline_kvb = candidateKvbDigest(baseline_candidate.?);
    const candidate_digests = [2][32]u8{ candidateDigest(candidates[0].?), candidateDigest(candidates[1].?) };
    const candidate_kvb = [2][32]u8{ candidateKvbDigest(candidates[0].?), candidateKvbDigest(candidates[1].?) };

    const heldout_storage = try allocator.alloc(f64, seen_count * 64 * dim);
    defer allocator.free(heldout_storage);
    var heldout_elements: usize = 0;
    var validation_seen: [16]u64 = undefined;
    var validation_seen_count: usize = 0;
    for (plan.fixture_shards, 0..) |shard, shard_index| {
        const provenance_path = try std.fs.path.join(allocator, &.{ shard.fixture_dir, "provenance.json" });
        defer allocator.free(provenance_path);
        try verifySidecar(allocator, io, provenance_path, shard.fixture_provenance_sha256);
        const binding: fixture_mod.Binding = .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config_sha };
        var fixture = try fixture_mod.read(allocator, io, shard.fixture_dir, binding);
        defer fixture.deinit();
        var manifest_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_hash, .{});
        if (!std.mem.eql(u8, &manifest_hash, &phase_manifest_hashes[shard_index]) or !sameOriginal(&first.?, &fixture)) return error.InvalidRejectedHeadReplay;
        try validateShardRoles(&fixture, shard.consumers, &validation_seen, &validation_seen_count);
        @memcpy(heldout_storage[heldout_elements..][0..fixture.heldout.len], fixture.heldout);
        heldout_elements += fixture.heldout.len;
    }
    if (validation_seen_count != seen_count or !std.mem.eql(u64, seen[0..seen_count], validation_seen[0..validation_seen_count]) or heldout_elements != seen_count * 64 * dim) return error.InvalidRejectedHeadReplay;
    const heldout = heldout_storage[0..heldout_elements];
    const original = first.?.originalHead();
    var baseline = try validateCandidate(allocator, original, baseline_candidate.?, heldout);
    baseline.accelerated_value_solves = baseline_solves;
    var cells: [2]Cell = undefined;
    for (&cells, &candidates) |*cell, candidate| cell.* = try validateCandidate(allocator, original, candidate.?, heldout);
    cells[1].accelerated_value_solves = exact_solves;
    const body = if (deferred_v3)
        try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.all-head-train-exact-core-sharded-result.v2", .authenticated = true, .donor_layer = plan.donor_layer, .kv_head = plan.kv_head, .risk_report_sha256 = plan.risk_report_sha256, .risk_rule = plan.risk_rule, .shard_count = plan.fixture_shards.len, .consumer_count = seen_count, .train_rows = train_elements / dim, .heldout_rows = heldout_elements / dim, .target_rows = plan.target_rows, .construction_policy = "deferred_intermediate_training_checks_v1", .candidate_count = 3, .candidate_digest_count = 6, .query_batch_rows = query_batch_rows, .phase_profile = .{ .baseline = baseline_profile, .exact = exact_profile, .baseline_head_wall_ns = baseline_candidate.?.elapsed_ns, .exact_cpu_head_wall_ns = candidates[0].?.elapsed_ns, .exact_accelerated_head_wall_ns = candidates[1].?.elapsed_ns }, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb[1], .lower), .candidates_frozen_before_any_heldout_payload_open = true, .intermediate_training_checks_authorize_quality = false, .baseline = baseline, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{})
    else
        try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.all-head-train-exact-core-sharded-result.v1", .authenticated = true, .donor_layer = plan.donor_layer, .kv_head = plan.kv_head, .risk_report_sha256 = plan.risk_report_sha256, .risk_rule = plan.risk_rule, .shard_count = plan.fixture_shards.len, .consumer_count = seen_count, .train_rows = train_elements / dim, .heldout_rows = heldout_elements / dim, .target_rows = plan.target_rows, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb, .lower), .exact_core_cpu_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[0], .lower), .exact_core_cpu_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb[0], .lower), .exact_core_accelerated_candidate_sha256 = std.fmt.bytesToHex(candidate_digests[1], .lower), .exact_core_accelerated_candidate_kvb_sha256 = std.fmt.bytesToHex(candidate_kvb[1], .lower), .candidates_frozen_before_any_heldout_payload_open = true, .baseline = baseline, .exact_core_cpu = cells[0], .exact_core_accelerated = cells[1], .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
    defer allocator.free(body);
    var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer result_file.close(io);
    try result_file.writeStreamingAll(io, body);
    try result_file.writeStreamingAll(io, "\n");
    try result_file.sync(io);
}

fn finishShardedJoint(
    plan: ShardedPlan,
    accelerate_values: bool,
    allocator: std.mem.Allocator,
    io: std.Io,
    identity: checkpoint.Identity,
    config_sha: [32]u8,
    combined: *fixture_mod.Fixture,
    original_fixture: *fixture_mod.Fixture,
    baseline: OwnedCandidate,
    baseline_profile: @import("../../ops/metal_attention_match_fit.zig").Fitter.PhaseCounters,
    baseline_solves: u64,
    phase_manifest_hashes: [4][32]u8,
    train_seen: []const u64,
) !void {
    if (!std.mem.eql(u8, plan.risk_rule, "critical_probability_half_gate_0_025_v1") or baseline_solves == 0) return error.InvalidRejectedHeadReplayPlan;
    var ranges: [23]joint_mass_bias_refit.Range = undefined;
    for (&ranges, plan.geometry_targets, plan.exact_core_source_keys, 0..) |*range, target, cores, index| range.* = .{ .start = 128 + index * 256, .rows = @min(@as(usize, 256), 5888 - (128 + index * 256)), .target_budget = target, .exact_core_source_keys = cores };
    const component_cap: joint_mass_bias_refit.ComponentCap = if (plan.joint_component_cap_mib.? == 768) .mib768 else .mib512;
    const train_bytes = try std.math.mul(usize, 15360 * 512, @sizeOf(f64));
    const baseline_bytes = try std.math.add(usize, try std.math.mul(usize, baseline.keys.len + baseline.values.len + baseline.bias.len, @sizeOf(f64)), try std.math.mul(usize, baseline.positions.len + baseline.coverage.len, @sizeOf(u64)));
    var external_live = try std.math.add(usize, fixture_mod.maximum_bytes, train_bytes);
    external_live = try std.math.add(usize, external_live, baseline_bytes);
    external_live = try std.math.add(usize, external_live, 64 * 1024 * 1024);
    const immutable_k = bytesDigest(std.mem.sliceAsBytes(baseline.keys));
    const immutable_v = bytesDigest(std.mem.sliceAsBytes(baseline.values));
    const immutable_positions = bytesDigest(std.mem.sliceAsBytes(baseline.positions));
    const immutable_coverage = bytesDigest(std.mem.sliceAsBytes(baseline.coverage));
    var residual_rows: usize = 0;
    for (baseline.positions) |position| {
        const key = std.math.cast(usize, position) orelse return error.InvalidRejectedHeadReplayPlan;
        var fixed = key < 128 or key >= 5888;
        if (!fixed) for (ranges) |range| if (key >= range.start and key < range.start + range.rows) {
            fixed = range.target_budget == range.rows or std.mem.indexOfScalar(usize, range.exact_core_source_keys, key) != null;
            break;
        };
        if (!fixed) residual_rows += 1;
    }
    var fit_cfg: solver.Config = .{ .target_rows = plan.target_rows, .selection = .residual_omp, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = component_cap.bytes() };
    if (accelerate_values) {
        fit_cfg.value_backend = accelerate_value_backend.valueBackend();
        const q = combined.train.len / 512;
        const adapter_bytes = try fit_cfg.value_backend.?.workspace_fn(fit_cfg.value_backend.?.context, q, residual_rows, 1, fit_cfg.ridge);
        const charged_qr_scalars = try std.math.add(usize, try std.math.mul(usize, q + residual_rows, residual_rows), try std.math.add(usize, q + residual_rows, try std.math.add(usize, residual_rows, try std.math.add(usize, q + residual_rows, residual_rows))));
        const charged_qr_bytes = try std.math.add(usize, try std.math.mul(usize, charged_qr_scalars, @sizeOf(f64)), try joint_mass_bias_refit.valueBackendExtraBytes(q, residual_rows, 1, fit_cfg));
        if (adapter_bytes > charged_qr_bytes or adapter_bytes > component_cap.bytes()) return error.CompactionWorkspaceLimit;
    }
    var refit = try joint_mass_bias_refit.fitJointResidualMassBias(allocator, combined.originalHead(), combined.original_positions, combined.original_coverage, baseline.head(combined.originalHead()), baseline.positions, baseline.coverage, &ranges, combined.train, fit_cfg, external_live);
    defer refit.deinit();
    if (refit.residual_rows != residual_rows) return error.InvalidRejectedHeadReplay;
    if (!std.mem.eql(u8, &immutable_k, &bytesDigest(std.mem.sliceAsBytes(baseline.keys))) or !std.mem.eql(u8, &immutable_v, &bytesDigest(std.mem.sliceAsBytes(baseline.values))) or !std.mem.eql(u8, &immutable_positions, &bytesDigest(std.mem.sliceAsBytes(baseline.positions))) or !std.mem.eql(u8, &immutable_coverage, &bytesDigest(std.mem.sliceAsBytes(baseline.coverage)))) return error.InvalidRejectedHeadReplay;
    var joint = try cloneCandidate(allocator, baseline);
    defer joint.deinit();
    const replacement_bias = try allocator.dupe(f64, refit.bias);
    allocator.free(joint.bias);
    joint.bias = replacement_bias;
    const baseline_digest = candidateDigest(baseline);
    const baseline_kvb = candidateKvbDigest(baseline);
    const joint_digest = candidateDigest(joint);
    const joint_kvb = candidateKvbDigest(joint);

    const heldout_storage = try allocator.alloc(f64, train_seen.len * 64 * 512);
    defer allocator.free(heldout_storage);
    var final_live = try std.math.mul(usize, fixture_mod.maximum_bytes, 2);
    final_live = try std.math.add(usize, final_live, train_bytes);
    final_live = try std.math.add(usize, final_live, try std.math.mul(usize, baseline_bytes, 2));
    final_live = try std.math.add(usize, final_live, try std.math.mul(usize, train_seen.len * 64 * 512, @sizeOf(f64)));
    final_live = try std.math.add(usize, final_live, 64 * 1024 * 1024);
    final_live = try std.math.add(usize, final_live, try std.math.mul(usize, refit.bias.len + refit.weights.len, @sizeOf(f64)));
    final_live = try std.math.add(usize, final_live, try solver.validationBytes(7936, plan.target_rows, 512, train_seen.len * 64, .{ .target_rows = plan.target_rows, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 }));
    if (final_live > joint_mass_bias_refit.combined_cap_bytes) return error.CompactionWorkspaceLimit;
    var heldout_elements: usize = 0;
    var validation_seen: [16]u64 = undefined;
    var validation_seen_count: usize = 0;
    for (plan.fixture_shards, 0..) |shard, shard_index| {
        const provenance_path = try std.fs.path.join(allocator, &.{ shard.fixture_dir, "provenance.json" });
        defer allocator.free(provenance_path);
        try verifySidecar(allocator, io, provenance_path, shard.fixture_provenance_sha256);
        const binding: fixture_mod.Binding = .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config_sha };
        var fixture = try fixture_mod.read(allocator, io, shard.fixture_dir, binding);
        defer fixture.deinit();
        var manifest_hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_hash, .{});
        if (!std.mem.eql(u8, &manifest_hash, &phase_manifest_hashes[shard_index]) or !sameOriginal(original_fixture, &fixture)) return error.InvalidRejectedHeadReplay;
        try validateShardRoles(&fixture, shard.consumers, &validation_seen, &validation_seen_count);
        @memcpy(heldout_storage[heldout_elements..][0..fixture.heldout.len], fixture.heldout);
        heldout_elements += fixture.heldout.len;
    }
    if (validation_seen_count != train_seen.len or !std.mem.eql(u64, train_seen, validation_seen[0..validation_seen_count]) or heldout_elements != train_seen.len * 64 * 512) return error.InvalidRejectedHeadReplay;
    const original = original_fixture.originalHead();
    var baseline_cell = try validateCandidate(allocator, original, baseline, heldout_storage[0..heldout_elements]);
    baseline_cell.accelerated_value_solves = baseline_solves;
    const joint_cell = try validateCandidate(allocator, original, joint, heldout_storage[0..heldout_elements]);
    const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = if (accelerate_values) "antfly.all-head-joint-mass-bias-sharded-result.v2" else "antfly.all-head-joint-mass-bias-sharded-result.v1", .joint_value_backend = if (accelerate_values) accelerate_value_backend.policy else "native_fp64_householder", .authenticated = true, .donor_layer = plan.donor_layer, .kv_head = plan.kv_head, .risk_report_sha256 = plan.risk_report_sha256, .risk_rule = plan.risk_rule, .shard_count = plan.fixture_shards.len, .consumer_count = train_seen.len, .train_rows = combined.train.len / 512, .heldout_rows = heldout_elements / 512, .target_rows = plan.target_rows, .candidate_count = 2, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb, .lower), .joint_bias_candidate_sha256 = std.fmt.bytesToHex(joint_digest, .lower), .joint_bias_candidate_kvb_sha256 = std.fmt.bytesToHex(joint_kvb, .lower), .candidates_frozen_before_any_heldout_payload_open = true, .immutable_keys_values_positions_coverage_verified = true, .component_cap_mib = plan.joint_component_cap_mib.?, .combined_cap_bytes = joint_mass_bias_refit.combined_cap_bytes, .external_live_bytes = external_live, .fit_peak_bytes = refit.fit_peak_bytes, .final_validation_admission_bytes = final_live, .baseline_phase_profile = baseline_profile, .baseline = baseline_cell, .joint_mass_bias = joint_cell, .train_before = refit.train_before, .train_after = refit.train_after, .min_weight = @as(f64, 1e-12), .floored_weight_count = refit.floored_weight_count, .max_floor_perturbation = refit.max_floor_perturbation, .returned_weights_semantics = "raw_pre_storage_floor_kkt_solution", .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
    defer allocator.free(body);
    var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer result_file.close(io);
    try result_file.writeStreamingAll(io, body);
    try result_file.writeStreamingAll(io, "\n");
    try result_file.sync(io);
}

fn runJointWholeHeadMassBias(plan_path: []const u8) !void {
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const plan_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(plan_bytes);
    const parsed = try std.json.parseFromSlice(FixedSupportRefitPlan, allocator, plan_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const plan = parsed.value;
    const accelerate_v2 = std.mem.eql(u8, plan.experiment_schema, "antfly.joint-whole-head-mass-bias-single-head-plan.v2");
    const prior_v3 = std.mem.eql(u8, plan.experiment_schema, "antfly.joint-whole-head-mass-bias-single-head-plan.v3");
    if (!std.mem.eql(u8, plan.schema, "antfly.source61-guarded-diagnostic-test-plan.v3") or
        (!accelerate_v2 and !prior_v3 and !std.mem.eql(u8, plan.experiment_schema, "antfly.joint-whole-head-mass-bias-single-head-plan.v1")) or
        plan.joint_value_backend_accelerate != (accelerate_v2 or prior_v3) or plan.joint_prior_cv != prior_v3 or
        (prior_v3 and (plan.joint_prior_cv_policy_sha256 == null or !std.mem.eql(u8, plan.joint_prior_cv_policy_sha256.?, "de0343749d954f99c9c3c66f81d183b6c7379ea5aae56a66d3daa57523fa26ac") or plan.joint_prior_cv_module_sha256 == null or !std.mem.eql(u8, plan.joint_prior_cv_module_sha256.?, "160537bf00c4b1e9dcc6fb7e2bc7fb37e3dfefcbe7b749286230859834d3e87b"))) or
        (!prior_v3 and (plan.joint_prior_cv_policy_sha256 != null or plan.joint_prior_cv_module_sha256 != null)) or
        !std.fs.path.isAbsolute(plan.result_path) or !std.fs.path.isAbsolute(plan.fixture_dir) or
        !std.fs.path.isAbsolute(plan.fixture_index_path) or !std.fs.path.isAbsolute(plan.risk_report_path) or
        plan.donor_layer != 5 or plan.kv_head != 0 or plan.consumers.len != 4 or plan.prefix_rows != 128 or plan.recent_rows != 2048 or
        plan.chunk_rows != 256 or plan.retention_divisor != 2 or plan.query_batch_rows != 64 or
        plan.fitter_workspace_bytes != joint_mass_bias_refit.fit_workspace_bytes or
        plan.max_combined_bytes != joint_mass_bias_refit.combined_cap_bytes or
        plan.joint_mass_bias_module_sha256 == null or
        !std.mem.eql(u8, plan.joint_mass_bias_module_sha256.?, if (prior_v3) "d91887e28c5c51cddd950564f5c26e61b072b2c770bd1d6153a045d83cd06338" else "678dc54e903cf083e1ab354e2085555b2b299e6bb581b800dff3a12c5cf2cd57"))
        return error.InvalidRejectedHeadReplayPlan;

    var ranges_storage: [23]joint_mass_bias_refit.Range = undefined;
    const ranges = try jointRangeClosure(plan, &ranges_storage);
    const plan_scratch_bytes = try std.math.mul(usize, plan_bytes.len, 2);
    if (plan.target_rows != 4908 or !std.mem.eql(u8, plan.risk_rule, "critical_probability_half_gate_0_025_v1")) return error.InvalidRejectedHeadReplayPlan;

    const identity: checkpoint.Identity = .{ .weights = try digest(plan.identity.weights), .tokenizer = try digest(plan.identity.tokenizer), .template = try digest(plan.identity.template), .layout = try digest(plan.identity.layout) };
    const config_sha = try digest(plan.config_sha256);
    try verifySidecar(allocator, io, plan.fixture_index_path, plan.fixture_index_sha256);
    const index_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan.fixture_index_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(index_bytes);
    const index_parsed = try std.json.parseFromSlice(FixtureIndex, allocator, index_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer index_parsed.deinit();
    const fixture_index = index_parsed.value;
    if (!std.mem.eql(u8, fixture_index.schema, "antfly.all-head-fixture-shard-index.v1") or !fixture_index.diagnostic_only or fixture_index.quality_credit or
        fixture_index.donor_layer != plan.donor_layer or fixture_index.kv_head != plan.kv_head or fixture_index.source_rows != 7936 or fixture_index.head_dim != 512 or
        fixture_index.candidate_rows != plan.target_rows or !std.mem.eql(u8, fixture_index.config_sha256, plan.config_sha256) or
        !std.mem.eql(u8, fixture_index.identity.weights, plan.identity.weights) or !std.mem.eql(u8, fixture_index.identity.tokenizer, plan.identity.tokenizer) or
        !std.mem.eql(u8, fixture_index.identity.template, plan.identity.template) or !std.mem.eql(u8, fixture_index.identity.layout, plan.identity.layout) or
        !std.mem.eql(usize, fixture_index.geometry_targets, plan.geometry_targets) or fixture_index.fixture_shards.len != 1 or
        !std.mem.eql(u8, fixture_index.fixture_shards[0].fixture_dir, plan.fixture_dir) or
        !std.mem.eql(u8, fixture_index.fixture_shards[0].fixture_manifest_sha256, plan.fixture_manifest_sha256) or
        !std.mem.eql(u8, fixture_index.fixture_shards[0].fixture_provenance_sha256, plan.fixture_provenance_sha256) or
        fixture_index.ordered_unique_consumers.len != plan.consumers.len or fixture_index.fixture_shards[0].consumers.len != plan.consumers.len)
        return error.InvalidRejectedHeadReplayPlan;
    for (plan.consumers, fixture_index.ordered_unique_consumers, fixture_index.fixture_shards[0].consumers) |expected, ordered, shard_consumer| {
        if (expected.layer != ordered.layer or expected.query_head != ordered.query_head or expected.layer != shard_consumer.layer or expected.query_head != shard_consumer.query_head)
            return error.InvalidRejectedHeadReplayPlan;
    }
    const provenance_path = try std.fs.path.join(allocator, &.{ plan.fixture_dir, "provenance.json" });
    defer allocator.free(provenance_path);
    try verifySidecar(allocator, io, provenance_path, plan.fixture_provenance_sha256);

    const risk_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan.risk_report_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(risk_bytes);
    var risk_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(risk_bytes, &risk_hash, .{});
    if (!std.mem.eql(u8, &risk_hash, &(try digest(plan.risk_report_sha256)))) return error.InvalidRejectedHeadReplayPlan;
    const risk_parsed = try std.json.parseFromSlice(GenericRiskReport, allocator, risk_bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer risk_parsed.deinit();
    const risk = risk_parsed.value;
    if (!std.mem.eql(u8, risk.schema, "antfly.train-risk-budget-audit.v2") or !risk.authenticated or
        risk.donor_layer != plan.donor_layer or risk.kv_head != plan.kv_head or risk.geometry_divisor != 2 or
        risk.original_source_rows != 7936 or !risk.original_source_positions_contiguous or risk.train_rows_total != 3840 or
        !std.mem.eql(u8, risk.train_scope, "all960_primary256_historical448_future256_per_consumer") or
        risk.heldout_query_values_read or !std.mem.eql(u8, risk.risk_rule, plan.risk_rule) or risk.ranges.len != 23)
        return error.InvalidRejectedHeadReplayPlan;
    for (risk.ranges, ranges, 0..) |row, range, index| {
        if (row.range_index != index or row.start != range.start or row.rows != range.rows or row.target_budget != range.target_budget or
            row.exact_core_count != range.exact_core_source_keys.len or !std.mem.eql(usize, row.exact_core_source_keys, range.exact_core_source_keys))
            return error.InvalidRejectedHeadReplayPlan;
    }

    const binding: fixture_mod.Binding = .{ .output_dir = plan.fixture_dir, .identity = identity, .config_sha256 = config_sha };
    var fixture = try fixture_mod.readTrainingOnly(allocator, io, plan.fixture_dir, binding);
    defer fixture.deinit();
    var manifest_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_hash, .{});
    if (!std.mem.eql(u8, &manifest_hash, &(try digest(plan.fixture_manifest_sha256))) or fixture.geometry.donor_layer != plan.donor_layer or
        fixture.geometry.kv_head != plan.kv_head or fixture.geometry.original_rows != 7936 or fixture.geometry.candidate_rows != plan.target_rows or
        fixture.geometry.head_dim != 512 or fixture.geometry.prefix_rows != 128 or fixture.geometry.recent_rows != 2048 or
        fixture.geometry.chunk_rows != 256 or fixture.geometry.retention_divisor != 2 or fixture.geometry.scale != 0.044194173824159216 or
        fixture.geometry.softcap != 0.0 or fixture.train.len != 3840 * 512)
        return error.InvalidRejectedHeadReplayPlan;
    var seen: [4]u64 = undefined;
    var seen_count: usize = 0;
    try validateShardRoles(&fixture, plan.consumers, &seen, &seen_count);
    try validateCompleteConsumerMatrix(plan.donor_layer, plan.kv_head, 1, seen[0..seen_count]);
    for (fixture.query_train_counts) |count| if (count != 960) return error.InvalidRejectedHeadReplayPlan;

    var seed_storage: [23][]usize = undefined;
    var seed_views: [23][]const usize = undefined;
    var seed_count: usize = 0;
    defer for (seed_storage[0..seed_count]) |seed| allocator.free(seed);
    for (plan.exact_core_source_keys, 0..) |cores, index| {
        seed_storage[index] = try allocator.alloc(usize, cores.len);
        seed_count += 1;
        for (cores, seed_storage[index]) |key, *local| local.* = key - ranges[index].start;
        seed_views[index] = seed_storage[index];
    }

    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(allocator, null, joint_mass_bias_refit.fit_workspace_bytes);
    var fitter_live = true;
    defer if (fitter_live) fitter.deinit();
    try fitter.setQueryBatchRows(64);
    if (fitter.queryBatchRows() != 64) return error.InvalidRejectedHeadReplay;
    try fitter.setPhaseProfiling(true);
    const baseline_begin = platform.time.monotonicNs();
    var baseline = try buildCandidate(allocator, &fixture, &seed_views, plan.geometry_targets, fitter.backend(), true);
    defer baseline.deinit();
    baseline.elapsed_ns = platform.time.monotonicNs() -| baseline_begin;
    if (!baseline.accelerated) return error.InvalidRejectedHeadReplay;
    const baseline_profile = fitter.phaseCounters();
    const baseline_solves = fitter.deinitAndTakeValueSolves();
    fitter_live = false;
    if (baseline_solves == 0) return error.InvalidRejectedHeadReplay;
    const baseline_digest = candidateDigest(baseline);
    const baseline_kvb = candidateKvbDigest(baseline);
    if (!std.mem.eql(u8, &baseline_digest, &(try digest(plan.expected_baseline_candidate_sha256))) or
        !std.mem.eql(u8, &baseline_kvb, &(try digest(plan.expected_baseline_candidate_kvb_sha256))))
        return error.InvalidRejectedHeadReplay;

    const immutable_k_hash = bytesDigest(std.mem.sliceAsBytes(baseline.keys));
    const immutable_v_hash = bytesDigest(std.mem.sliceAsBytes(baseline.values));
    const immutable_positions_hash = bytesDigest(std.mem.sliceAsBytes(baseline.positions));
    const immutable_coverage_hash = bytesDigest(std.mem.sliceAsBytes(baseline.coverage));
    var external_live_bytes = try std.math.add(usize, fixture_mod.maximum_bytes, try baselineOwnedCandidateBytes(plan.target_rows, 512));
    external_live_bytes = try std.math.add(usize, external_live_bytes, @max(try std.math.mul(usize, plan_scratch_bytes, 4), 64 * 1024 * 1024));
    var cv_fold_ids: []u8 = &.{};
    var cv_variable_ranges: []u8 = &.{};
    var cv_ranges: []prior_cv.Range = &.{};
    defer if (cv_fold_ids.len != 0) allocator.free(cv_fold_ids);
    defer if (cv_variable_ranges.len != 0) allocator.free(cv_variable_ranges);
    defer if (cv_ranges.len != 0) allocator.free(cv_ranges);
    var cv_context: PriorCvContext = .{ .fold_ids = &.{}, .variable_range_indices = &.{}, .ranges = &.{} };
    defer if (cv_context.result) |*result| result.deinit();
    if (prior_v3) {
        cv_fold_ids = try allocator.alloc(u8, 3840);
        var row: usize = 0;
        for (fixture.query_train_counts, 0..) |count, consumer_ordinal| {
            if (count != 960 or consumer_ordinal >= 4) return error.InvalidRejectedHeadReplayPlan;
            @memset(cv_fold_ids[row..][0..count], @intCast(consumer_ordinal % prior_cv.fold_count));
            row += count;
        }
        if (row != cv_fold_ids.len) return error.InvalidRejectedHeadReplayPlan;
        cv_ranges = try allocator.alloc(prior_cv.Range, ranges.len);
        for (ranges, cv_ranges) |range, *out| out.* = .{ .rows = range.rows, .target_budget = range.target_budget, .exact_core_count = range.exact_core_source_keys.len };
        var residual_count: usize = 0;
        for (baseline.positions) |position| {
            const key = std.math.cast(usize, position) orelse return error.InvalidRejectedHeadReplayPlan;
            for (ranges) |range| if (key >= range.start and key < range.start + range.rows) {
                if (range.target_budget != range.rows and std.mem.indexOfScalar(usize, range.exact_core_source_keys, key) == null) residual_count += 1;
                break;
            };
        }
        cv_variable_ranges = try allocator.alloc(u8, residual_count);
        var variable: usize = 0;
        for (baseline.positions) |position| {
            const key = std.math.cast(usize, position) orelse return error.InvalidRejectedHeadReplayPlan;
            for (ranges, 0..) |range, range_index| if (key >= range.start and key < range.start + range.rows) {
                if (range.target_budget != range.rows and std.mem.indexOfScalar(usize, range.exact_core_source_keys, key) == null) {
                    cv_variable_ranges[variable] = @intCast(range_index);
                    variable += 1;
                }
                break;
            };
        }
        if (variable != residual_count or residual_count != 1254) return error.InvalidFitConfig;
        cv_context.fold_ids = cv_fold_ids;
        cv_context.variable_range_indices = cv_variable_ranges;
        cv_context.ranges = cv_ranges;
    }
    var fit_cfg: solver.Config = .{ .target_rows = plan.target_rows, .selection = .residual_omp, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = joint_mass_bias_refit.fit_workspace_bytes };
    if (accelerate_v2 or prior_v3) {
        fit_cfg.value_backend = accelerate_value_backend.valueBackend();
        const adapter_bytes = try fit_cfg.value_backend.?.workspace_fn(fit_cfg.value_backend.?.context, 3840, 1254, 1, fit_cfg.ridge);
        const charged_qr_scalars = try std.math.add(usize, try std.math.mul(usize, 3840 + 1254, 1254), try std.math.add(usize, 3840 + 1254, try std.math.add(usize, 1254, try std.math.add(usize, 3840 + 1254, 1254))));
        const charged_qr_bytes = try std.math.add(usize, try std.math.mul(usize, charged_qr_scalars, @sizeOf(f64)), try joint_mass_bias_refit.valueBackendExtraBytes(3840, 1254, 1, fit_cfg));
        if (adapter_bytes > charged_qr_bytes or adapter_bytes > joint_mass_bias_refit.fit_workspace_bytes) return error.CompactionWorkspaceLimit;
    }
    const refit_begin = platform.time.monotonicNs();
    var refit = try joint_mass_bias_refit.fitJointResidualMassBiasOptions(allocator, fixture.originalHead(), fixture.original_positions, fixture.original_coverage, baseline.head(fixture.originalHead()), baseline.positions, baseline.coverage, ranges, fixture.train, fit_cfg, external_live_bytes, .{ .selector = if (prior_v3) .{ .context = &cv_context, .select_fn = selectPriorCvWeights } else null });
    defer refit.deinit();
    if (refit.residual_rows != 1254) return error.InvalidFitConfig;
    const refit_elapsed_ns = platform.time.monotonicNs() -| refit_begin;
    const final_k_hash = bytesDigest(std.mem.sliceAsBytes(baseline.keys));
    const final_v_hash = bytesDigest(std.mem.sliceAsBytes(baseline.values));
    const final_positions_hash = bytesDigest(std.mem.sliceAsBytes(baseline.positions));
    const final_coverage_hash = bytesDigest(std.mem.sliceAsBytes(baseline.coverage));
    if (!std.mem.eql(u8, &immutable_k_hash, &final_k_hash) or !std.mem.eql(u8, &immutable_v_hash, &final_v_hash) or
        !std.mem.eql(u8, &immutable_positions_hash, &final_positions_hash) or
        !std.mem.eql(u8, &immutable_coverage_hash, &final_coverage_hash))
        return error.InvalidRejectedHeadReplay;
    const refit_view: BorrowedCandidate = .{ .head = .{ .keys = baseline.keys, .values = baseline.values, .bias = refit.bias, .rows = plan.target_rows, .dim = 512, .scale = fixture.geometry.scale, .softcap = fixture.geometry.softcap }, .positions = baseline.positions, .coverage = baseline.coverage };
    const refit_digest = borrowedCandidateDigest(refit_view);
    const refit_kvb = borrowedCandidateKvbDigest(refit_view);

    var parsed_input_bytes = try std.math.add(usize, plan_bytes.len, index_bytes.len);
    parsed_input_bytes = try std.math.add(usize, parsed_input_bytes, risk_bytes.len);
    parsed_input_bytes = try std.math.mul(usize, parsed_input_bytes, 4);
    parsed_input_bytes = try std.math.add(usize, parsed_input_bytes, try std.math.mul(usize, 7936, @sizeOf(usize)));
    const metadata_reserve_bytes = @max(parsed_input_bytes, 64 * 1024 * 1024);
    var final_validation_admission_bytes = try std.math.mul(usize, fixture_mod.maximum_bytes, 2);
    final_validation_admission_bytes = try std.math.add(usize, final_validation_admission_bytes, try baselineOwnedCandidateBytes(plan.target_rows, 512));
    final_validation_admission_bytes = try std.math.add(usize, final_validation_admission_bytes, try std.math.mul(usize, plan.target_rows + refit.residual_rows, @sizeOf(f64)));
    final_validation_admission_bytes = try std.math.add(usize, final_validation_admission_bytes, try solver.validationBytes(7936, plan.target_rows, 512, 256, .{ .target_rows = plan.target_rows, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05 }));
    final_validation_admission_bytes = try std.math.add(usize, final_validation_admission_bytes, metadata_reserve_bytes);
    if (final_validation_admission_bytes > joint_mass_bias_refit.combined_cap_bytes) return error.CompactionWorkspaceLimit;
    var full = try fixture_mod.read(allocator, io, plan.fixture_dir, binding);
    defer full.deinit();
    var full_manifest_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(full.manifest_bytes, &full_manifest_hash, .{});
    if (!std.mem.eql(u8, &full_manifest_hash, &manifest_hash) or !sameOriginal(&fixture, &full) or full.heldout.len != 4 * 64 * 512) return error.InvalidRejectedHeadReplay;
    var baseline_cell = try validateCandidate(allocator, fixture.originalHead(), baseline, full.heldout);
    baseline_cell.accelerated_value_solves = baseline_solves;
    const refit_cell = try validateBorrowedCandidate(allocator, fixture.originalHead(), refit_view, full.heldout);

    const body = if (prior_v3) blk: {
        const cv = &(cv_context.result orelse return error.InvalidRejectedHeadReplay);
        break :blk try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.joint-whole-head-mass-bias-single-head-result.v3", .joint_value_backend = accelerate_value_backend.policy, .joint_prior_cv_policy_sha256 = plan.joint_prior_cv_policy_sha256.?, .authenticated = true, .donor_layer = plan.donor_layer, .kv_head = plan.kv_head, .consumer_count = 4, .train_rows = 3840, .heldout_rows = 256, .target_rows = plan.target_rows, .risk_report_sha256 = plan.risk_report_sha256, .risk_rule = plan.risk_rule, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb, .lower), .joint_bias_candidate_sha256 = std.fmt.bytesToHex(refit_digest, .lower), .joint_bias_candidate_kvb_sha256 = std.fmt.bytesToHex(refit_kvb, .lower), .candidates_frozen_before_any_heldout_payload_open = true, .immutable_keys_values_positions_coverage_verified = true, .chosen_ratio = cv.chosen_ratio, .chosen_lambda = cv.chosen_lambda, .worst_fold_max_abs_residual = cv.worst_fold_max_abs_residual, .fold_metrics = cv.fold_metrics, .cv_fit_peak_bytes = cv.fit_peak_bytes, .returned_weights_semantics = "raw_pre_storage_floor_kkt_solution", .min_weight = @as(f64, 1e-12), .floored_weight_count = refit.floored_weight_count, .max_floor_perturbation = refit.max_floor_perturbation, .baseline = baseline_cell, .joint_mass_bias = refit_cell, .train_before = refit.train_before, .train_after = refit.train_after, .resource_proof = .{ .fit_stage_external_live_bytes = external_live_bytes, .joint_fit_peak_bytes = refit.fit_peak_bytes, .cv_fit_peak_bytes = cv.fit_peak_bytes, .final_validation_stage_admission_bytes = final_validation_admission_bytes, .combined_cap_bytes = joint_mass_bias_refit.combined_cap_bytes, .fixture_owner_count_final_stage = 2, .fixture_cap_bytes_each = fixture_mod.maximum_bytes, .metadata_reserve_bytes = metadata_reserve_bytes, .fit_workspace_bytes = joint_mass_bias_refit.fit_workspace_bytes, .validation_peak_bytes = refit.validation_peak_bytes }, .phase_profile = .{ .baseline = baseline_profile, .baseline_elapsed_ns = baseline.elapsed_ns, .refit_elapsed_ns = refit_elapsed_ns, .baseline_value_solves = baseline_solves, .refit_nnls_sweeps = refit.nnls_sweeps, .residual_rows = refit.residual_rows }, .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
    } else try std.json.Stringify.valueAlloc(allocator, .{ .schema = if (accelerate_v2) "antfly.joint-whole-head-mass-bias-single-head-result.v2" else "antfly.joint-whole-head-mass-bias-single-head-result.v1", .joint_value_backend = if (accelerate_v2) accelerate_value_backend.policy else "native_fp64_householder", .authenticated = true, .donor_layer = plan.donor_layer, .kv_head = plan.kv_head, .consumer_count = 4, .train_rows = 3840, .heldout_rows = 256, .target_rows = plan.target_rows, .risk_report_sha256 = plan.risk_report_sha256, .risk_rule = plan.risk_rule, .baseline_candidate_sha256 = std.fmt.bytesToHex(baseline_digest, .lower), .baseline_candidate_kvb_sha256 = std.fmt.bytesToHex(baseline_kvb, .lower), .joint_bias_candidate_sha256 = std.fmt.bytesToHex(refit_digest, .lower), .joint_bias_candidate_kvb_sha256 = std.fmt.bytesToHex(refit_kvb, .lower), .candidates_frozen_before_any_heldout_payload_open = true, .immutable_keys_values_positions_coverage_verified = true, .returned_weights_semantics = "raw_pre_storage_floor_kkt_solution", .min_weight = @as(f64, 1e-12), .floored_weight_count = refit.floored_weight_count, .max_floor_perturbation = refit.max_floor_perturbation, .baseline = baseline_cell, .joint_mass_bias = refit_cell, .train_before = refit.train_before, .train_after = refit.train_after, .resource_proof = .{ .fit_stage_external_live_bytes = external_live_bytes, .fit_peak_bytes = refit.fit_peak_bytes, .final_validation_stage_admission_bytes = final_validation_admission_bytes, .combined_cap_bytes = joint_mass_bias_refit.combined_cap_bytes, .fixture_owner_count_final_stage = 2, .fixture_cap_bytes_each = fixture_mod.maximum_bytes, .metadata_reserve_bytes = metadata_reserve_bytes, .fit_workspace_bytes = joint_mass_bias_refit.fit_workspace_bytes, .validation_peak_bytes = refit.validation_peak_bytes }, .phase_profile = .{ .baseline = baseline_profile, .baseline_elapsed_ns = baseline.elapsed_ns, .refit_elapsed_ns = refit_elapsed_ns, .baseline_value_solves = baseline_solves, .refit_nnls_sweeps = refit.nnls_sweeps, .residual_rows = refit.residual_rows }, .heldout_controls_fit = false, .final_gate_mass = 0.05, .final_gate_output = 0.05, .diagnostic_only = true, .quality_credit = false, .production_qualified = false }, .{});
    defer allocator.free(body);
    var result_file = try std.Io.Dir.createFileAbsolute(io, plan.result_path, .{ .exclusive = true, .truncate = false });
    defer result_file.close(io);
    try result_file.writeStreamingAll(io, body);
    try result_file.writeStreamingAll(io, "\n");
    try result_file.sync(io);
}

test "attention match joint whole head mass bias single head diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_JOINT_WHOLE_HEAD_MASS_BIAS_REPLAY_PLAN") orelse return error.SkipZigTest;
    try runJointWholeHeadMassBias(plan_path);
}

test "attention match exact peak residual captured head divisor2 CPU Metal diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_EXACT_PEAK_RESIDUAL_REPLAY_PLAN") orelse return error.SkipZigTest;
    try runReplay(plan_path);
}

test "attention match TRAIN exact peak residual retained D5H1 diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_D5H1_TRAIN_EXACT_CORE_REPLAY_PLAN") orelse return error.SkipZigTest;
    try runReplay(plan_path);
}

test "attention match TRAIN global minimax residual calibration D5H1 diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_D5H1_TRAIN_GLOBAL_MINIMAX_REPLAY_PLAN") orelse return error.SkipZigTest;
    try runReplay(plan_path);
}

test "attention match all head TRAIN exact core sharded retained diagnostic" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_ALL_HEAD_EXACT_CORE_SHARDED_REPLAY_PLAN") orelse return error.SkipZigTest;
    try runShardedReplay(plan_path);
}

test "fixed support final validation admission counts both fixture owners and rejects over cap" {
    const base = try finalValidationLiveBytes(4908, 0);
    try std.testing.expect(base > 2 * fixture_mod.maximum_bytes);
    const exact_metadata = fixed_support_refit.combined_cap_bytes - base;
    try std.testing.expectEqual(fixed_support_refit.combined_cap_bytes, try checkedFinalValidationAdmission(4908, exact_metadata));
    try std.testing.expectError(error.CompactionWorkspaceLimit, checkedFinalValidationAdmission(4908, exact_metadata + 1));
}

test "attention match sharded replay role closure rejects duplicate and wrong counts" {
    var fixture: fixture_mod.Fixture = undefined;
    const layers = [_]u32{ 23, 23, 23, 23 };
    const heads = [_]u32{ 0, 1, 2, 3 };
    var train_counts = [_]u32{ 960, 960, 960, 960 };
    const heldout_counts = [_]u32{ 64, 64, 64, 64 };
    fixture.query_consumer_layers = @constCast(&layers);
    fixture.query_heads = @constCast(&heads);
    fixture.query_train_counts = &train_counts;
    fixture.query_heldout_counts = @constCast(&heldout_counts);
    const tuples = [_]ConsumerTuple{ .{ .layer = 23, .query_head = 0 }, .{ .layer = 23, .query_head = 1 }, .{ .layer = 23, .query_head = 2 }, .{ .layer = 23, .query_head = 3 } };
    var seen: [16]u64 = undefined;
    var count: usize = 0;
    try validateShardRoles(&fixture, &tuples, &seen, &count);
    try std.testing.expectEqual(@as(usize, 4), count);
    try std.testing.expectError(error.InvalidRejectedHeadReplayPlan, validateShardRoles(&fixture, &tuples, &seen, &count));
    fixture.query_train_counts[2] = 959;
    count = 0;
    try std.testing.expectError(error.InvalidRejectedHeadReplayPlan, validateShardRoles(&fixture, &tuples, &seen, &count));
    try std.testing.expectError(error.InvalidRejectedHeadReplayPlan, validateCompleteConsumerMatrix(23, 0, 1, seen[0..4]));
    const layer5 = [_]u64{ (@as(u64, 5) << 32) | 0, (@as(u64, 5) << 32) | 1, (@as(u64, 5) << 32) | 2, (@as(u64, 5) << 32) | 3 };
    try validateCompleteConsumerMatrix(5, 0, 1, &layer5);
}

test "attention match sharded deferred construction policy is typed and opt in" {
    const v2 = "antfly.all-head-train-exact-core-sharded-replay-plan.v2";
    const v3 = "antfly.all-head-train-exact-core-sharded-replay-plan.v3";
    const v4 = "antfly.all-head-joint-mass-bias-sharded-replay-plan.v4";
    const v5 = "antfly.all-head-joint-mass-bias-sharded-replay-plan.v5";
    try std.testing.expect(shardedConstructionPolicyValid(v2, false, false, .rows16));
    try std.testing.expect(!shardedConstructionPolicyValid(v2, true, false, .rows16));
    try std.testing.expect(!shardedConstructionPolicyValid(v2, false, true, .rows16));
    try std.testing.expect(!shardedConstructionPolicyValid(v2, false, false, .rows64));
    try std.testing.expect(shardedConstructionPolicyValid(v3, true, true, .rows16));
    try std.testing.expect(shardedConstructionPolicyValid(v3, true, true, .rows64));
    try std.testing.expect(!shardedConstructionPolicyValid(v3, false, true, .rows64));
    try std.testing.expect(!shardedConstructionPolicyValid(v3, true, false, .rows64));
    try std.testing.expect(shardedConstructionPolicyValid(v4, true, true, .rows64));
    try std.testing.expect(!shardedConstructionPolicyValid(v4, true, true, .rows16));
    try std.testing.expect(!shardedConstructionPolicyValid(v4, false, true, .rows64));
    try std.testing.expect(shardedConstructionPolicyValid(v5, true, true, .rows64));
    try std.testing.expect(!shardedConstructionPolicyValid(v5, true, true, .rows16));
    try std.testing.expect(!shardedConstructionPolicyValid("foreign", true, true, .rows64));
}

test "replay QR admission includes the charged optimal workspace delta" {
    const cfg: solver.Config = .{ .target_rows = 4908, .max_workspace_bytes = joint_mass_bias_refit.fit_workspace_bytes, .value_backend = accelerate_value_backend.valueBackend() };
    const q: usize = 3840;
    const m: usize = 1254;
    const declared = try cfg.value_backend.?.workspace_fn(null, q, m, 1, cfg.ridge);
    const legacy = ((q + m) * m + 2 * (q + m) + 2 * m) * @sizeOf(f64);
    try std.testing.expect(declared > legacy);
    const charged = try std.math.add(usize, legacy, try joint_mass_bias_refit.valueBackendExtraBytes(q, m, 1, cfg));
    try std.testing.expect(charged >= declared);
    try std.testing.expect(charged <= cfg.max_workspace_bytes);
}

test "retained root505 scorer scalar versus Accelerate benchmark" {
    const index_path = platform.env.getenv("ANTFLY_METAL_ROOT505_SCORER_BENCHMARK_INDEX") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(index_path)) return error.InvalidRejectedHeadReplayPlan;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, index_path, allocator, .limited(2 * 1024 * 1024));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(FixtureIndex, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = true });
    defer parsed.deinit();
    const index = parsed.value;
    if (!std.mem.eql(u8, index.schema, "antfly.all-head-fixture-shard-index.v1") or index.donor_layer != 23 or index.kv_head > 1 or index.source_rows != 7936 or index.head_dim != 512 or index.fixture_shards.len != 4 or index.ordered_unique_consumers.len != 16) return error.InvalidRejectedHeadReplayPlan;
    const identity: checkpoint.Identity = .{ .weights = try digest(index.identity.weights), .tokenizer = try digest(index.identity.tokenizer), .template = try digest(index.identity.template), .layout = try digest(index.identity.layout) };
    const config_sha = try digest(index.config_sha256);
    const q: usize = 15360;
    const n: usize = 7936;
    const d: usize = 512;
    const train = try allocator.alloc(f64, q * d);
    defer allocator.free(train);
    var train_at: usize = 0;
    var first: ?fixture_mod.Fixture = null;
    defer if (first) |*owned| owned.deinit();
    for (index.fixture_shards, 0..) |shard, shard_index| {
        const provenance_path = try std.fs.path.join(allocator, &.{ shard.fixture_dir, "provenance.json" });
        defer allocator.free(provenance_path);
        try verifySidecar(allocator, io, provenance_path, shard.fixture_provenance_sha256);
        var fixture = if (shard_index == 0)
            try fixture_mod.read(allocator, io, shard.fixture_dir, .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config_sha })
        else
            try fixture_mod.readTrainingOnly(allocator, io, shard.fixture_dir, .{ .output_dir = shard.fixture_dir, .identity = identity, .config_sha256 = config_sha });
        var moved = false;
        defer if (!moved) fixture.deinit();
        var manifest_sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(fixture.manifest_bytes, &manifest_sha, .{});
        const expected_manifest_sha = try digest(shard.fixture_manifest_sha256);
        if (!std.mem.eql(u8, &manifest_sha, &expected_manifest_sha) or fixture.train.len != 3840 * d) return error.InvalidRejectedHeadReplayPlan;
        if (first) |*reference| {
            if (!sameOriginal(reference, &fixture)) return error.InvalidRejectedHeadReplayPlan;
        } else {
            first = fixture;
            moved = true;
        }
        @memcpy(train[train_at..][0..fixture.train.len], fixture.train);
        train_at += fixture.train.len;
    }
    if (train_at != train.len) return error.InvalidRejectedHeadReplayPlan;
    const fixture = &first.?;
    const target = fixture.candidate_positions.len;
    if (target != index.candidate_rows or fixture.candidate_bias.len != target or fixture.candidate_values.len != target * d) return error.InvalidRejectedHeadReplayPlan;
    const selected = try allocator.alloc(usize, target);
    defer allocator.free(selected);
    for (fixture.candidate_positions, selected) |position, *local| {
        local.* = std.mem.indexOfScalar(u64, fixture.original_positions, position) orelse return error.InvalidRejectedHeadReplayPlan;
    }
    var fitter = try @import("../../ops/metal_attention_match_fit.zig").Fitter.init(allocator, null, 2 * 1024 * 1024 * 1024);
    defer fitter.deinit();
    try fitter.setQueryBatchRows(64);
    var captured = try fitter.capturePageSource(allocator, fixture.originalHead(), train, 0, 2 * 1024 * 1024 * 1024, .{});
    defer captured.deinit();
    const rm0 = try allocator.alloc(f64, q);
    defer allocator.free(rm0);
    const rm1 = try allocator.alloc(f64, q);
    defer allocator.free(rm1);
    const cm0 = try allocator.alloc(f64, q);
    defer allocator.free(cm0);
    const cm1 = try allocator.alloc(f64, q);
    defer allocator.free(cm1);
    const rn0 = try allocator.alloc(f64, q * d);
    defer allocator.free(rn0);
    const rn1 = try allocator.alloc(f64, q * d);
    defer allocator.free(rn1);
    const cn0 = try allocator.alloc(f64, q * d);
    defer allocator.free(cn0);
    const cn1 = try allocator.alloc(f64, q * d);
    defer allocator.free(cn1);
    const page = try allocator.alloc(f64, 64 * n);
    defer allocator.free(page);
    const candidate_page = try allocator.alloc(f64, 64 * target);
    defer allocator.free(candidate_page);
    @memset(rm0, 0);
    @memset(cm0, 0);
    @memset(rn0, 0);
    @memset(cn0, 0);
    var scalar_ns: u64 = 0;
    var accelerate_ns: u64 = 0;
    var at: usize = 0;
    while (at < q) {
        const count = @min(@as(usize, 64), q - at);
        const p = page[0 .. count * n];
        try fitter.replayCapturedPageQueries(&captured, 0, n, at, count, p, .{});
        var begin = platform.time.monotonicNs();
        for (0..count) |lq| {
            for (p[lq * n ..][0..n], 0..) |probability, row| {
                rm0[at + lq] += probability;
                for (0..d) |di| rn0[(at + lq) * d + di] += probability * fixture.original_values[row * d + di];
            }
            for (selected, 0..) |row, si| {
                const probability = p[lq * n + row] * @exp(fixture.candidate_bias[si] - fixture.original_bias[row]);
                cm0[at + lq] += probability;
                for (0..d) |di| cn0[(at + lq) * d + di] += probability * fixture.candidate_values[si * d + di];
            }
        }
        scalar_ns +%= platform.time.monotonicNs() -| begin;
        begin = platform.time.monotonicNs();
        try bounded_page_replay.acceleratedTileReductions(p, fixture.original_values, selected, fixture.candidate_bias, fixture.original_bias, fixture.candidate_values, count, n, d, rm1[at..][0..count], rn1[at * d ..][0 .. count * d], cm1[at..][0..count], cn1[at * d ..][0 .. count * d], candidate_page[0 .. count * target]);
        accelerate_ns +%= platform.time.monotonicNs() -| begin;
        at += count;
    }
    try std.testing.expectEqualSlices(f64, rm0, rm1);
    try std.testing.expectEqualSlices(f64, cm0, cm1);
    for (rn0, rn1) |a, b| try std.testing.expectApproxEqRel(a, b, 2e-12);
    for (cn0, cn1) |a, b| try std.testing.expectApproxEqRel(a, b, 2e-12);
    const scalar_harm = try adaptive_geometry.measureRoundedCandidateHarm(rn0, rm0, rn0, cm0, cn0, q, d, 1e-12);
    const accelerated_harm = try adaptive_geometry.measureRoundedCandidateHarm(rn0, rm1, rn1, cm1, cn1, q, d, 1e-12);
    try std.testing.expectApproxEqRel(scalar_harm.max_relative_mass, accelerated_harm.max_relative_mass, 2e-12);
    try std.testing.expectApproxEqRel(scalar_harm.relative_output_rmse, accelerated_harm.relative_output_rmse, 2e-12);
    const scalar_accepted = scalar_harm.max_relative_mass <= 0.05 and scalar_harm.relative_output_rmse <= 0.05;
    const accelerated_accepted = accelerated_harm.max_relative_mass <= 0.05 and accelerated_harm.relative_output_rmse <= 0.05;
    try std.testing.expectEqual(scalar_accepted, accelerated_accepted);
    std.debug.print("root505_scorer_benchmark index={s} q={d} n={d} d={d} scalar_ns={d} accelerate_ns={d} speedup={d:.3} mass_harm={d:.12} output_harm={d:.12} accepted={}\n", .{ index_path, q, n, d, scalar_ns, accelerate_ns, @as(f64, @floatFromInt(scalar_ns)) / @as(f64, @floatFromInt(accelerate_ns)), accelerated_harm.max_relative_mass, accelerated_harm.relative_output_rmse, accelerated_accepted });
}
