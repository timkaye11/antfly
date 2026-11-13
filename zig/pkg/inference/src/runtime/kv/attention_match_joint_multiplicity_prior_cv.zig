// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Diagnostic-only TRAIN CV for a uniform-multiplicity prior on joint mass weights.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "joint_uniform_multiplicity_prior_train_consumer_class_cv_v1";
pub const ratio_grid = [_]f64{ 0, 1e-4, 1e-2, 1 };
pub const fold_count: usize = 4;
pub const tie_tolerance: f64 = 1e-12;
pub const component_cap_bytes: usize = 512 * 1024 * 1024;
pub const combined_cap_bytes: usize = 1024 * 1024 * 1024;

pub const Range = struct {
    rows: usize,
    target_budget: usize,
    exact_core_count: usize,
};

pub const FoldMetric = struct {
    ratio: f64,
    fold: u8,
    lambda: f64,
    max_abs_residual: f64,
    solver_sweeps: usize,
    failed: bool,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    weights: []f64,
    multiplicities: []f64,
    fold_metrics: []FoldMetric,
    chosen_ratio: f64,
    chosen_lambda: f64,
    worst_fold_max_abs_residual: f64,
    final_sweeps: usize,
    fit_peak_bytes: usize,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.weights);
        self.allocator.free(self.multiplicities);
        self.allocator.free(self.fold_metrics);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

/// Conservative simultaneous live set for one fold: source A/rho/fold ids are
/// caller-owned; this charges gathered fit A/rho, augmented A/rho, solver A,
/// full-active QR, vectors, and returned weights/multiplicities/metrics.
pub fn fitPeakBytes(q: usize, m: usize, metrics: usize, value_backend_bytes: usize) !usize {
    if (q == 0 or m == 0) return error.InvalidFitConfig;
    var scalars = try mul(try add(q, m), m); // gathered/augmented caller basis
    scalars = try add(scalars, try mul(try add(q, m), m)); // active solver basis
    scalars = try add(scalars, try mul(try add(q, try mul(2, m)), m)); // full-active QR
    scalars = try add(scalars, try mul(q, 4)); // rho, residual, norms scratch
    scalars = try add(scalars, try mul(m, 12)); // weights/prior/active/QR vectors
    var bytes = try mul(scalars, @sizeOf(f64));
    bytes = try add(bytes, try mul(metrics, @sizeOf(FoldMetric)));
    bytes = try add(bytes, value_backend_bytes);
    return bytes;
}

pub fn multiplicityForRange(range: Range) !f64 {
    if (range.rows == 0 or range.target_budget == 0 or range.target_budget > range.rows or range.exact_core_count > range.target_budget or range.exact_core_count > range.rows) return error.InvalidFitConfig;
    if (range.target_budget == range.rows) return 1;
    const residual_source = range.rows - range.exact_core_count;
    const residual_slots = range.target_budget - range.exact_core_count;
    if (residual_slots == 0) return error.InvalidFitConfig;
    const value = @as(f64, @floatFromInt(residual_source)) / @as(f64, @floatFromInt(residual_slots));
    if (!std.math.isFinite(value) or value < 1) return error.InvalidFitConfig;
    return value;
}

pub fn deriveMultiplicities(out: []f64, variable_range_indices: []const u8, ranges: []const Range) !void {
    if (out.len == 0 or out.len != variable_range_indices.len) return error.InvalidFitConfig;
    for (out, variable_range_indices) |*value, range_index| {
        if (range_index >= ranges.len) return error.InvalidFitConfig;
        if (ranges[range_index].target_budget == ranges[range_index].rows) return error.InvalidFitConfig;
        value.* = try multiplicityForRange(ranges[range_index]);
    }
}

fn meanColumnNormSq(a: []const f64, q: usize, m: usize, included: []const bool, multiplicities: []const f64) !f64 {
    var sum: f64 = 0;
    var fit_rows: usize = 0;
    for (included) |yes| if (yes) {
        fit_rows += 1;
    };
    if (fit_rows == 0) return error.InvalidFitConfig;
    for (0..m) |col| {
        var norm: f64 = 0;
        for (0..q) |row| if (included[row]) {
            const x = a[row * m + col] * multiplicities[col];
            norm += x * x;
        };
        sum += norm;
    }
    const mean = sum / @as(f64, @floatFromInt(m));
    if (!std.math.isFinite(mean) or mean < 0) return error.CompactionFitRejected;
    return mean;
}

const Solve = struct { weights: []f64, lambda: f64, sweeps: usize };

fn solveSubset(
    allocator: std.mem.Allocator,
    a: []const f64,
    rho: []const f64,
    q: usize,
    m: usize,
    included: []const bool,
    multiplicities: []const f64,
    baseline_weights: []const f64,
    ratio: f64,
    cfg: solver.Config,
) !Solve {
    var fit_rows: usize = 0;
    for (included) |yes| if (yes) {
        fit_rows += 1;
    };
    if (fit_rows == 0) return error.InvalidFitConfig;
    const lambda = ratio * try meanColumnNormSq(a, q, m, included, multiplicities);
    if (!std.math.isFinite(lambda) or lambda < 0) return error.CompactionFitRejected;
    const augmented_rows = try add(fit_rows, if (lambda == 0) 0 else m);
    const basis = try allocator.alloc(f64, try mul(augmented_rows, m));
    defer allocator.free(basis);
    const target = try allocator.alloc(f64, augmented_rows);
    defer allocator.free(target);
    var dst: usize = 0;
    for (0..q) |row| if (included[row]) {
        for (0..m) |col| basis[dst * m + col] = a[row * m + col] * (if (ratio == 0) 1 else multiplicities[col]);
        target[dst] = rho[row];
        dst += 1;
    };
    if (lambda != 0) {
        const root = @sqrt(lambda);
        for (0..m) |row| {
            @memset(basis[(dst + row) * m ..][0..m], 0);
            basis[(dst + row) * m + row] = root;
            target[dst + row] = root;
        }
    }
    const v = try allocator.alloc(f64, m);
    errdefer allocator.free(v);
    for (v, baseline_weights, multiplicities) |*value, warm, n| value.* = if (ratio == 0) warm else warm / n;
    const residual = try allocator.alloc(f64, augmented_rows);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, m);
    defer allocator.free(norms);
    const sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, basis, target, augmented_rows, m, v, residual, norms, cfg, true);
    for (v, multiplicities) |*weight, n| {
        if (ratio != 0) weight.* *= n;
        if (!std.math.isFinite(weight.*) or weight.* < 0) return error.CompactionFitRejected;
    }
    return .{ .weights = v, .lambda = lambda, .sweeps = sweeps };
}

fn heldFoldScore(a: []const f64, rho: []const f64, q: usize, m: usize, fold_ids: []const u8, fold: u8, weights: []const f64) !f64 {
    var count: usize = 0;
    var worst: f64 = 0;
    for (0..q) |row| if (fold_ids[row] == fold) {
        var predicted: f64 = 0;
        for (0..m) |col| predicted += a[row * m + col] * weights[col];
        worst = @max(worst, @abs(predicted - rho[row]));
        count += 1;
    };
    if (count == 0 or !std.math.isFinite(worst)) return error.InvalidFitConfig;
    return worst;
}

pub fn selectAndFit(
    allocator: std.mem.Allocator,
    original_basis: []const f64,
    rho: []const f64,
    q: usize,
    m: usize,
    fold_ids: []const u8,
    variable_range_indices: []const u8,
    ranges: []const Range,
    baseline_weights: []const f64,
    cfg: solver.Config,
    external_live_bytes: usize,
) !Result {
    if (q == 0 or m == 0 or original_basis.len != try mul(q, m) or rho.len != q or fold_ids.len != q or variable_range_indices.len != m or baseline_weights.len != m or cfg.max_workspace_bytes != component_cap_bytes) return error.InvalidFitConfig;
    for (original_basis) |x| if (!std.math.isFinite(x) or x < 0) return error.InvalidFitConfig;
    for (baseline_weights) |x| if (!std.math.isFinite(x) or x < 0) return error.InvalidFitConfig;
    for (rho) |x| if (!std.math.isFinite(x) or x < 0) return error.InvalidFitConfig;
    var fold_seen: [fold_count]bool = @splat(false);
    for (fold_ids) |fold| {
        if (fold >= fold_count) return error.InvalidFitConfig;
        fold_seen[fold] = true;
    }
    for (fold_seen) |seen| if (!seen) return error.InvalidFitConfig;
    const multiplicities = try allocator.alloc(f64, m);
    errdefer allocator.free(multiplicities);
    try deriveMultiplicities(multiplicities, variable_range_indices, ranges);
    const metric_count = ratio_grid.len * fold_count;
    const value_backend_bytes = if (cfg.value_backend) |backend| try backend.workspace_fn(backend.context, q + m, m, 1, cfg.ridge) else 0;
    const peak = try fitPeakBytes(q, m, metric_count, value_backend_bytes);
    if (peak > component_cap_bytes or external_live_bytes > combined_cap_bytes or peak > combined_cap_bytes - external_live_bytes) return error.CompactionWorkspaceLimit;
    const metrics = try allocator.alloc(FoldMetric, metric_count);
    errdefer allocator.free(metrics);
    var best_ratio: f64 = 0;
    var best_score = std.math.inf(f64);
    var have_best = false;
    var metric_index: usize = 0;
    for (ratio_grid) |ratio| {
        var ratio_score: f64 = 0;
        var ratio_valid = true;
        for (0..fold_count) |fold_usize| {
            const fold: u8 = @intCast(fold_usize);
            const included = try allocator.alloc(bool, q);
            defer allocator.free(included);
            for (included, fold_ids) |*yes, row_fold| yes.* = row_fold != fold;
            const solved = solveSubset(allocator, original_basis, rho, q, m, included, multiplicities, baseline_weights, ratio, cfg) catch |err| switch (err) {
                error.Cancelled, error.OutOfMemory => return err,
                else => {
                    metrics[metric_index] = .{ .ratio = ratio, .fold = fold, .lambda = 0, .max_abs_residual = 0, .solver_sweeps = 0, .failed = true };
                    metric_index += 1;
                    ratio_valid = false;
                    continue;
                },
            };
            defer allocator.free(solved.weights);
            const score = try heldFoldScore(original_basis, rho, q, m, fold_ids, fold, solved.weights);
            metrics[metric_index] = .{ .ratio = ratio, .fold = fold, .lambda = solved.lambda, .max_abs_residual = score, .solver_sweeps = solved.sweeps, .failed = false };
            metric_index += 1;
            ratio_score = @max(ratio_score, score);
        }
        if (ratio_valid and (!have_best or ratio_score < best_score - tie_tolerance or (@abs(ratio_score - best_score) <= tie_tolerance and ratio > best_ratio))) {
            have_best = true;
            best_ratio = ratio;
            best_score = ratio_score;
        }
    }
    if (!have_best) return error.CompactionFitRejected;
    const included = try allocator.alloc(bool, q);
    defer allocator.free(included);
    @memset(included, true);
    const final = try solveSubset(allocator, original_basis, rho, q, m, included, multiplicities, baseline_weights, best_ratio, cfg);
    return .{ .allocator = allocator, .weights = final.weights, .multiplicities = multiplicities, .fold_metrics = metrics, .chosen_ratio = best_ratio, .chosen_lambda = final.lambda, .worst_fold_max_abs_residual = best_score, .final_sweeps = final.sweeps, .fit_peak_bytes = peak };
}

test "uniform multiplicity derives compacted and identity priors" {
    try std.testing.expectApproxEqAbs(@as(f64, 2), try multiplicityForRange(.{ .rows = 256, .target_budget = 128, .exact_core_count = 0 }), 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1), try multiplicityForRange(.{ .rows = 256, .target_budget = 256, .exact_core_count = 7 }), 0);
    try std.testing.expectApproxEqAbs(@as(f64, 252.0 / 108.0), try multiplicityForRange(.{ .rows = 256, .target_budget = 112, .exact_core_count = 4 }), 0);
    try std.testing.expectError(error.InvalidFitConfig, multiplicityForRange(.{ .rows = 256, .target_budget = 4, .exact_core_count = 4 }));
}

test "CV tie chooses higher regularization deterministically" {
    // All rows have the same exactly representable optimum. Every ratio scores
    // zero, so the frozen tie rule must choose the largest ratio.
    const a = [_]f64{ 1, 1, 1, 1 };
    const rho = [_]f64{ 2, 2, 2, 2 };
    const folds = [_]u8{ 0, 1, 2, 3 };
    const range_ids = [_]u8{0};
    const ranges = [_]Range{.{ .rows = 2, .target_budget = 1, .exact_core_count = 0 }};
    var result = try selectAndFit(std.testing.allocator, &a, &rho, 4, 1, &folds, &range_ids, &ranges, &.{2}, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = component_cap_bytes }, 0);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 1), result.chosen_ratio);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.weights[0], 1e-12);
}

test "ratio zero preserves the direct original-basis warm NNLS solve" {
    const a = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const rho = [_]f64{ 2, 3, 5 };
    const included = [_]bool{ true, true, true };
    const multiplicities = [_]f64{ 2, 4 };
    const warm = [_]f64{ 0.5, 0.75 };
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = component_cap_bytes };
    const transformed = try solveSubset(std.testing.allocator, &a, &rho, 3, 2, &included, &multiplicities, &warm, 0, cfg);
    defer std.testing.allocator.free(transformed.weights);
    var direct = warm;
    var residual: [3]f64 = undefined;
    var norms: [2]f64 = undefined;
    _ = try solver.fitMassActiveTargetDiagnostic(std.testing.allocator, &a, &rho, 3, 2, &direct, &residual, &norms, cfg, true);
    try std.testing.expectApproxEqAbs(direct[0], transformed.weights[0], 1e-12);
    try std.testing.expectApproxEqAbs(direct[1], transformed.weights[1], 1e-12);
}

test "CV shape and admission failures are fail closed" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitPeakBytes(std.math.maxInt(usize), 2, 16, 0));
    try std.testing.expectError(error.InvalidFitConfig, deriveMultiplicities(&.{}, &.{}, &.{}));
}
