// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only raw OMP selection in full-original denominator units.
const std = @import("std");
const solver = @import("attention_match.zig");
const training_source = @import("attention_match_joint_training_source.zig");

pub const Selection = solver.MassSelectionResult;
pub const selection_policy = solver.mass_selection_policy;
pub const SelectionScore = solver.DiagnosticSelectionScore;
pub const SelectionPolicy = enum { raw, normalized, train_distinct_key_argmax_augmented_raw_v1, train_distinct_key_argmax_augmented_raw_x16_v1, train_peak_x16_centered_mass_ridge_1pct_v1, train_peak_support_centered_ridge_1pct_v1 };
pub const peak_augmented_max_query_rows: usize = 16384;

fn scorePolicy(score: SelectionScore) SelectionPolicy {
    return switch (score) {
        .raw => .raw,
        .normalized => .normalized,
    };
}

pub fn selectionPolicyDigest(score: SelectionScore) [32]u8 {
    return policyDigest(scorePolicy(score));
}

pub fn policyDigest(policy: SelectionPolicy) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.global-mass-selection-score.v1\x00");
    h.update(@tagName(policy));
    return h.finalResult();
}

fn checkedAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn checkedMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

fn selectFeaturesRuntimeScore(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config, score: SelectionScore) !Selection {
    try solver.validateConfigDiagnostic(cfg, rows);
    if (rho.len == 0 or probabilities.len != try checkedMul(rho.len, rows) or cfg.target_rows >= rows or cfg.selection != .residual_omp or cfg.mass_solver != .active_set) return error.InvalidFitConfig;
    const q = rho.len;
    const caller_bytes = try checkedAdd(try checkedMul(q, @sizeOf(f64)), try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    const nested_bytes = try solver.massSelectionWorkspaceBytes(rows, cfg.target_rows, q);
    if (try checkedAdd(caller_bytes, nested_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var nested = cfg;
    nested.max_workspace_bytes -= caller_bytes;
    return solver.selectMassTargetPolicyDiagnostic(allocator, probabilities, rho, rows, nested, score);
}

fn selectPeakSupportCenteredRidge(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config) !Selection {
    try solver.validateConfigDiagnostic(cfg, rows);
    if (rho.len == 0 or rho.len > peak_augmented_max_query_rows or probabilities.len != try checkedMul(rho.len, rows) or cfg.target_rows >= rows) return error.InvalidFitConfig;
    var target_progress = false;
    for (rho) |value| {
        if (!std.math.isFinite(value) or value < 0) return error.InvalidFitConfig;
        target_progress = target_progress or value > 0;
    }
    if (!target_progress) return error.InsufficientSelectionProgress;
    const rank_bytes = try checkedMul(rows, @sizeOf(usize) + @sizeOf(f64));
    const output_bytes = try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64));
    const required = try checkedAdd(try checkedAdd(rank_bytes, output_bytes), try solver.unitCenteredMassRidgeWorkspaceBytes(rho.len, cfg.target_rows, cfg));
    if (required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;

    const maxima = try allocator.alloc(f64, rows);
    defer allocator.free(maxima);
    const ranked = try allocator.alloc(usize, rows);
    defer allocator.free(ranked);
    var probability_progress = false;
    for (0..rows) |column| {
        if (column % 64 == 0) try cfg.control.check();
        var peak: f64 = 0;
        for (0..rho.len) |query| {
            if (query % 64 == 0) try cfg.control.check();
            const value = probabilities[query * rows + column];
            if (!std.math.isFinite(value) or value < 0 or value > 1) return error.NonFiniteAttention;
            peak = @max(peak, value);
            probability_progress = probability_progress or value > 0;
        }
        maxima[column] = peak;
        ranked[column] = column;
    }
    if (!probability_progress) return error.InsufficientSelectionProgress;
    std.mem.sort(usize, ranked, maxima, struct {
        fn less(scores: []f64, a: usize, b: usize) bool {
            return if (scores[a] == scores[b]) a < b else scores[a] > scores[b];
        }
    }.less);

    const indices = try allocator.dupe(usize, ranked[0..cfg.target_rows]);
    errdefer allocator.free(indices);
    std.mem.sort(usize, indices, {}, std.sort.asc(usize));
    const weights = try allocator.alloc(f64, cfg.target_rows);
    errdefer allocator.free(weights);
    @memset(weights, 1);
    var ridge_cfg = cfg;
    ridge_cfg.max_workspace_bytes -= rank_bytes + output_bytes;
    _ = try solver.refitMassWeightsUnitCenteredRidgeDiagnostic(allocator, probabilities, rho, rows, indices, weights, 0.01, ridge_cfg);
    return .{ .allocator = allocator, .indices = indices, .weights = weights, .query_rows_consumed = rho.len };
}

fn selectFeaturesRuntimePolicy(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config, policy: SelectionPolicy) !Selection {
    const multiplicity: usize = switch (policy) {
        .raw => return selectFeaturesRuntimeScore(allocator, probabilities, rho, rows, cfg, .raw),
        .normalized => return selectFeaturesRuntimeScore(allocator, probabilities, rho, rows, cfg, .normalized),
        .train_peak_support_centered_ridge_1pct_v1 => return selectPeakSupportCenteredRidge(allocator, probabilities, rho, rows, cfg),
        .train_distinct_key_argmax_augmented_raw_v1 => 1,
        .train_distinct_key_argmax_augmented_raw_x16_v1, .train_peak_x16_centered_mass_ridge_1pct_v1 => 16,
    };
    try solver.validateConfigDiagnostic(cfg, rows);
    if (rho.len == 0 or probabilities.len != try checkedMul(rho.len, rows)) return error.InvalidFitConfig;
    const maximum_q = try checkedAdd(rho.len, try checkedMul(multiplicity, @min(rows, rho.len)));
    const centered_ridge = policy == .train_peak_x16_centered_mass_ridge_1pct_v1;
    if ((policy == .train_distinct_key_argmax_augmented_raw_x16_v1 or centered_ridge) and maximum_q > peak_augmented_max_query_rows) return error.InvalidFitConfig;
    const maximum_retained = try checkedAdd(
        try checkedMul(try checkedAdd(try checkedMul(maximum_q, rows), maximum_q), @sizeOf(f64)),
        try checkedMul(rho.len, @sizeOf(bool)),
    );
    const maximum_caller = try checkedAdd(try checkedMul(maximum_q, @sizeOf(f64)), try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    const maximum_selection_bytes = try checkedAdd(maximum_retained, try checkedAdd(maximum_caller, try solver.massSelectionWorkspaceBytes(rows, cfg.target_rows, maximum_q)));
    const output_bytes = try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64));
    const maximum_ridge_bytes = if (centered_ridge) try checkedAdd(try checkedAdd(maximum_retained, output_bytes), try solver.unitCenteredMassRidgeWorkspaceBytes(maximum_q, cfg.target_rows, cfg)) else 0;
    if (@max(maximum_selection_bytes, if (centered_ridge) maximum_ridge_bytes else 0) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const marks = try allocator.alloc(bool, rho.len);
    defer allocator.free(marks);
    @memset(marks, false);
    for (0..rows) |column| {
        if (column % 64 == 0) try cfg.control.check();
        var best_query: usize = 0;
        var best = probabilities[column];
        if (!std.math.isFinite(best) or best < 0) return error.NonFiniteAttention;
        for (1..rho.len) |query| {
            const value = probabilities[query * rows + column];
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            if (value > best) {
                best = value;
                best_query = query;
            }
        }
        marks[best_query] = true;
    }
    var extra: usize = 0;
    for (marks) |marked| if (marked) {
        extra += 1;
    };
    const augmented_q = try checkedAdd(rho.len, try checkedMul(multiplicity, extra));
    const probability_count = try checkedMul(augmented_q, rows);
    const retained_bytes = try checkedAdd(try checkedMul(try checkedAdd(probability_count, augmented_q), @sizeOf(f64)), try checkedMul(marks.len, @sizeOf(bool)));
    const core_caller_bytes = try checkedAdd(try checkedMul(augmented_q, @sizeOf(f64)), try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    const core_nested_bytes = try solver.massSelectionWorkspaceBytes(rows, cfg.target_rows, augmented_q);
    const selection_bytes = try checkedAdd(retained_bytes, try checkedAdd(core_caller_bytes, core_nested_bytes));
    const ridge_bytes = if (centered_ridge) try checkedAdd(try checkedAdd(retained_bytes, output_bytes), try solver.unitCenteredMassRidgeWorkspaceBytes(augmented_q, cfg.target_rows, cfg)) else 0;
    if (@max(selection_bytes, if (centered_ridge) ridge_bytes else 0) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const augmented_probabilities = try allocator.alloc(f64, probability_count);
    defer allocator.free(augmented_probabilities);
    const augmented_rho = try allocator.alloc(f64, augmented_q);
    defer allocator.free(augmented_rho);
    @memcpy(augmented_probabilities[0..probabilities.len], probabilities);
    @memcpy(augmented_rho[0..rho.len], rho);
    var at = rho.len;
    for (marks, 0..) |marked, query| if (marked) {
        for (0..multiplicity) |_| {
            if ((at - rho.len) % 64 == 0) try cfg.control.check();
            @memcpy(augmented_probabilities[at * rows ..][0..rows], probabilities[query * rows ..][0..rows]);
            augmented_rho[at] = rho[query];
            at += 1;
        }
    };
    if (at != augmented_q) return error.InvalidAttentionShape;
    var nested = cfg;
    nested.max_workspace_bytes -= retained_bytes;
    var selected = try selectFeaturesRuntimeScore(allocator, augmented_probabilities, augmented_rho, rows, nested, .raw);
    errdefer selected.deinit();
    if (centered_ridge) {
        var ridge_cfg = cfg;
        ridge_cfg.max_workspace_bytes -= retained_bytes + output_bytes;
        _ = try solver.refitMassWeightsUnitCenteredRidgeDiagnostic(allocator, augmented_probabilities, augmented_rho, rows, selected.indices, selected.weights, 0.01, ridge_cfg);
    }
    return selected;
}

fn selectFeaturesRuntime(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config) !Selection {
    return selectFeaturesRuntimePolicy(allocator, probabilities, rho, rows, cfg, .raw);
}

pub fn selectFeatures(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global selection diagnostic entry is test-only");
    return selectFeaturesRuntime(allocator, probabilities, rho, rows, cfg);
}

pub fn selectFeaturesPolicy(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config, score: SelectionScore) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global selection diagnostic entry is test-only");
    return selectFeaturesRuntimePolicy(allocator, probabilities, rho, rows, cfg, scorePolicy(score));
}

pub fn selectFeaturesNamedPolicy(allocator: std.mem.Allocator, probabilities: []const f64, rho: []const f64, rows: usize, cfg: solver.Config, policy: SelectionPolicy) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global selection diagnostic entry is test-only");
    return selectFeaturesRuntimePolicy(allocator, probabilities, rho, rows, cfg, policy);
}

pub fn cachedWorkspaceBytes(chunk_rows: usize, query_count: usize, cfg: solver.Config) !usize {
    if (chunk_rows == 0 or query_count == 0 or cfg.target_rows >= chunk_rows) return error.InvalidFitConfig;
    const matrix_count = try checkedMul(query_count, chunk_rows);
    const crop_bytes = try checkedMul(try checkedAdd(matrix_count, query_count), @sizeOf(f64));
    const caller_bytes = try checkedAdd(try checkedMul(query_count, @sizeOf(f64)), try checkedMul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    return checkedAdd(crop_bytes, try checkedAdd(caller_bytes, try solver.massSelectionWorkspaceBytes(chunk_rows, cfg.target_rows, query_count)));
}

pub fn selectFromProbabilities(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, probabilities: []const f64, cfg: solver.Config) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global selection is diagnostic-test-only");
    try original.validate();
    try solver.validateConfigForTest(cfg, original.rows);
    if (chunk_rows == 0 or chunk_start > original.rows or chunk_rows > original.rows - chunk_start or probabilities.len == 0 or probabilities.len % original.rows != 0 or cfg.target_rows >= chunk_rows) return error.InvalidFitConfig;
    const q = probabilities.len / original.rows;
    if (try cachedWorkspaceBytes(chunk_rows, q, cfg) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const matrix = try allocator.alloc(f64, try checkedMul(q, chunk_rows));
    defer allocator.free(matrix);
    const rho = try allocator.alloc(f64, q);
    defer allocator.free(rho);
    for (0..q) |qi| {
        if (qi % 64 == 0) try cfg.control.check();
        const source = probabilities[qi * original.rows + chunk_start ..][0..chunk_rows];
        @memcpy(matrix[qi * chunk_rows ..][0..chunk_rows], source);
        var sum: f64 = 0;
        for (source) |value| {
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            sum += value;
        }
        if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
        rho[qi] = sum;
    }
    const crop_bytes = try checkedMul(try checkedAdd(try checkedMul(q, chunk_rows), q), @sizeOf(f64));
    var nested = cfg;
    nested.max_workspace_bytes -= crop_bytes;
    return selectFeatures(allocator, matrix, rho, chunk_rows, nested);
}

pub fn selectFromTrainingSourceRuntimePolicy(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, source: *const training_source.Source, cfg: solver.Config, score: SelectionScore) !Selection {
    return selectFromTrainingSourceRuntimeNamedPolicy(allocator, original, chunk_start, chunk_rows, source, cfg, scorePolicy(score));
}

pub fn selectFromTrainingSourceRuntimeNamedPolicy(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, source: *const training_source.Source, cfg: solver.Config, policy: SelectionPolicy) !Selection {
    try original.validate();
    try solver.validateConfigDiagnostic(cfg, original.rows);
    if (source.rows != original.rows or chunk_rows == 0 or chunk_start > original.rows or chunk_rows > original.rows - chunk_start or source.query_count == 0 or cfg.target_rows >= chunk_rows) return error.InvalidFitConfig;
    const q = source.query_count;
    if (try cachedWorkspaceBytes(chunk_rows, q, cfg) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const matrix = try allocator.alloc(f64, try checkedMul(q, chunk_rows));
    defer allocator.free(matrix);
    const rho = try allocator.alloc(f64, q);
    defer allocator.free(rho);
    for (0..q) |qi| {
        if (qi % 64 == 0) try cfg.control.check();
        const row = matrix[qi * chunk_rows ..][0..chunk_rows];
        try source.copyChunk(qi, chunk_start, row);
        var sum: f64 = 0;
        for (row) |value| {
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            sum += value;
        }
        if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
        rho[qi] = sum;
    }
    const crop_bytes = try checkedMul(try checkedAdd(try checkedMul(q, chunk_rows), q), @sizeOf(f64));
    var nested = cfg;
    nested.max_workspace_bytes -= crop_bytes;
    return selectFeaturesRuntimePolicy(allocator, matrix, rho, chunk_rows, nested, policy);
}

pub fn selectFromTrainingSourceRuntime(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, source: *const training_source.Source, cfg: solver.Config) !Selection {
    return selectFromTrainingSourceRuntimePolicy(allocator, original, chunk_start, chunk_rows, source, cfg, .raw);
}

pub fn selectFromTrainingSource(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, source: *const training_source.Source, cfg: solver.Config) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global cached selection diagnostic entry is test-only");
    return selectFromTrainingSourceRuntime(allocator, original, chunk_start, chunk_rows, source, cfg);
}

pub fn selectStreamingRuntimePolicy(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, train: []const f64, cfg: solver.Config, score: SelectionScore) !Selection {
    return selectStreamingRuntimeNamedPolicy(allocator, original, chunk_start, chunk_rows, train, cfg, scorePolicy(score));
}

pub fn selectStreamingRuntimeNamedPolicy(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, train: []const f64, cfg: solver.Config, policy: SelectionPolicy) !Selection {
    try original.validate();
    try solver.validateConfigDiagnostic(cfg, original.rows);
    if (chunk_rows == 0 or chunk_start > original.rows or chunk_rows > original.rows - chunk_start or train.len == 0 or train.len % original.dim != 0 or cfg.target_rows >= chunk_rows) return error.InvalidFitConfig;
    const q = train.len / original.dim;
    const matrix_count = try checkedMul(q, chunk_rows);
    const batch_count = try checkedMul(@as(usize, 64), original.rows);
    const retained_count = try checkedAdd(try checkedAdd(matrix_count, q), batch_count);
    const retained_bytes = try checkedMul(retained_count, @sizeOf(f64));
    const nested_bytes = try solver.massSelectionWorkspaceBytes(chunk_rows, cfg.target_rows, q);
    const attend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, original.rows, 1, @min(q, 64), original.dim) else 0;
    if (try checkedAdd(retained_bytes, @max(nested_bytes, attend_bytes)) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const matrix = try allocator.alloc(f64, matrix_count);
    defer allocator.free(matrix);
    const rho = try allocator.alloc(f64, q);
    defer allocator.free(rho);
    const batch = try allocator.alloc(f64, batch_count);
    defer allocator.free(batch);
    var offset: usize = 0;
    while (offset < q) {
        try cfg.control.check();
        const take = @min(@as(usize, 64), q - offset);
        try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], batch[0 .. take * original.rows], null, null, cfg);
        for (0..take) |local_query| {
            const source = batch[local_query * original.rows + chunk_start ..][0..chunk_rows];
            const destination = matrix[(offset + local_query) * chunk_rows ..][0..chunk_rows];
            @memcpy(destination, source);
            var sum: f64 = 0;
            for (source) |value| {
                if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                sum += value;
            }
            if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
            rho[offset + local_query] = sum;
        }
        offset += take;
    }
    var nested = cfg;
    nested.max_workspace_bytes -= retained_bytes;
    return selectFeaturesRuntimePolicy(allocator, matrix, rho, chunk_rows, nested, policy);
}

pub fn selectStreamingRuntime(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, train: []const f64, cfg: solver.Config) !Selection {
    return selectStreamingRuntimePolicy(allocator, original, chunk_start, chunk_rows, train, cfg, .raw);
}

pub fn selectStreaming(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, train: []const f64, cfg: solver.Config) !Selection {
    if (comptime !@import("builtin").is_test) @compileError("global streaming selection diagnostic entry is test-only");
    return selectStreamingRuntime(allocator, original, chunk_start, chunk_rows, train, cfg);
}

test "global denominator raw selection differs from local chunk normalization" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    // Query zero assigns only one percent of whole-head mass to this chunk;
    // local normalization nevertheless gives it the same selection weight.
    var global = try selectFeatures(std.testing.allocator, &.{ 0.0099, 0.0001, 0.45, 0.55 }, &.{ 0.01, 1 }, 2, cfg);
    defer global.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, global.indices);
    var local = try selectFeatures(std.testing.allocator, &.{ 0.99, 0.01, 0.45, 0.55 }, &.{ 1, 1 }, 2, cfg);
    defer local.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, local.indices);
}

test "typed score defaults to raw and normalized accepts arbitrary mass including zero range" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    const probabilities = [_]f64{ 0, 0, 0.9, 0.1, 0.2, 0.8 };
    const rho = [_]f64{ 0, 1, 1 };
    var established = try selectFeatures(std.testing.allocator, &probabilities, &rho, 2, cfg);
    defer established.deinit();
    var explicit_raw = try selectFeaturesPolicy(std.testing.allocator, &probabilities, &rho, 2, cfg, .raw);
    defer explicit_raw.deinit();
    try std.testing.expectEqualSlices(usize, established.indices, explicit_raw.indices);
    try std.testing.expectEqualSlices(f64, established.weights, explicit_raw.weights);
    var normalized = try selectFeaturesPolicy(std.testing.allocator, &probabilities, &rho, 2, cfg, .normalized);
    defer normalized.deinit();
    try std.testing.expectEqual(@as(usize, 1), normalized.indices.len);
    try std.testing.expect(std.math.isFinite(normalized.weights[0]) and normalized.weights[0] >= 0);
}

test "distinct TRAIN key argmax rows augment raw selection once with fail-closed controls" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    const probabilities = [_]f64{ 0.1, 0.2, 0.7, 0.9, 0.8, 0.1, 0.2, 0.1, 0.7 };
    const rho = [_]f64{ 1, 1, 1 };
    var raw = try selectFeatures(std.testing.allocator, &probabilities, &rho, 3, cfg);
    defer raw.deinit();
    var augmented = try selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cfg, .train_distinct_key_argmax_augmented_raw_v1);
    defer augmented.deinit();
    try std.testing.expectEqual(@as(usize, 3), raw.query_rows_consumed);
    try std.testing.expectEqual(@as(usize, 5), augmented.query_rows_consumed);
    var augmented_x16 = try selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cfg, .train_distinct_key_argmax_augmented_raw_x16_v1);
    defer augmented_x16.deinit();
    try std.testing.expectEqual(@as(usize, 35), augmented_x16.query_rows_consumed);
    // Independent one-column least-squares oracle: TRAIN queries 0 and 1
    // attain the column peaks, so their loss terms have weight 17.
    const selected_column = augmented_x16.indices[0];
    const p0 = probabilities[selected_column];
    const p1 = probabilities[3 + selected_column];
    const p2 = probabilities[6 + selected_column];
    const expected_weight = (17 * (p0 + p1) + p2) / (17 * (p0 * p0 + p1 * p1) + p2 * p2);
    try std.testing.expectApproxEqAbs(expected_weight, augmented_x16.weights[0], 1e-10);
    var centered = try selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cfg, .train_peak_x16_centered_mass_ridge_1pct_v1);
    defer centered.deinit();
    try std.testing.expectEqualSlices(usize, augmented_x16.indices, centered.indices);
    try std.testing.expectEqual(@as(usize, 35), centered.query_rows_consumed);
    const selected_energy = 17 * (p0 * p0 + p1 * p1) + p2 * p2;
    const lambda = @max(@as(f64, 1e-18), 0.01 * selected_energy);
    const expected_centered_weight = (17 * (p0 + p1) + p2 + lambda) / (selected_energy + lambda);
    try std.testing.expectApproxEqAbs(expected_centered_weight, centered.weights[0], 1e-10);
    try std.testing.expect(centered.weights[0] >= 0);
    var capped = cfg;
    capped.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, capped, .train_distinct_key_argmax_augmented_raw_v1));
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, capped, .train_peak_x16_centered_mass_ridge_1pct_v1));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled = cfg;
    cancelled.control.check_fn = Cancel.check;
    try std.testing.expectError(error.Cancelled, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cancelled, .train_distinct_key_argmax_augmented_raw_v1));
    try std.testing.expectError(error.Cancelled, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 3, cancelled, .train_peak_x16_centered_mass_ridge_1pct_v1));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, selectFeaturesNamedPolicy(failing.allocator(), &probabilities, &rho, 3, cfg, .train_distinct_key_argmax_augmented_raw_v1));
    var ridge_failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, selectFeaturesNamedPolicy(ridge_failing.allocator(), &probabilities, &rho, 3, cfg, .train_peak_x16_centered_mass_ridge_1pct_v1));
}

test "TRAIN peak support is stable exact and centered-ridge fail closed" {
    const probabilities = [_]f64{
        0.9, 0.2, 0.8, 0.1,
        0.1, 0.8, 0.3, 0.2,
        0.4, 0.1, 0.8, 0.7,
    };
    const rho = [_]f64{ 1, 0, 1 };
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var selected = try selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 4, cfg, .train_peak_support_centered_ridge_1pct_v1);
    defer selected.deinit();
    // Columns 1 and 2 tie at 0.8; lower source index wins the final slot.
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, selected.indices);
    try std.testing.expectEqual(@as(usize, 3), selected.query_rows_consumed);
    for (selected.weights) |weight| try std.testing.expect(std.math.isFinite(weight) and weight >= 0);

    const one_cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var one = try selectFeaturesNamedPolicy(std.testing.allocator, &.{ 0.5, 0.2, 0.5, 0.1 }, &.{ 1, 1 }, 2, one_cfg, .train_peak_support_centered_ridge_1pct_v1);
    defer one.deinit();
    const energy: f64 = 2 * 0.5 * 0.5;
    const lambda = 0.01 * energy;
    try std.testing.expectApproxEqAbs((1 + lambda) / (energy + lambda), one.weights[0], 1e-10);

    try std.testing.expectError(error.InsufficientSelectionProgress, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &.{ 0, 0, 0 }, 4, cfg, .train_peak_support_centered_ridge_1pct_v1));
    try std.testing.expectError(error.InsufficientSelectionProgress, selectFeaturesNamedPolicy(std.testing.allocator, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, &rho, 4, cfg, .train_peak_support_centered_ridge_1pct_v1));
    var capped = cfg;
    capped.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 4, capped, .train_peak_support_centered_ridge_1pct_v1));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled = cfg;
    cancelled.control.check_fn = Cancel.check;
    try std.testing.expectError(error.Cancelled, selectFeaturesNamedPolicy(std.testing.allocator, &probabilities, &rho, 4, cancelled, .train_peak_support_centered_ridge_1pct_v1));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, selectFeaturesNamedPolicy(failing.allocator(), &probabilities, &rho, 4, cfg, .train_peak_support_centered_ridge_1pct_v1));
}

test "global denominator selection rejects zero progress and fails closed on cap cancellation and allocation" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    try std.testing.expectError(error.InsufficientSelectionProgress, selectFeatures(std.testing.allocator, &.{ 0, 0, 0, 0 }, &.{ 0, 0 }, 2, cfg));
    var capped = cfg;
    capped.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFeatures(std.testing.allocator, &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, capped));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled = cfg;
    cancelled.control.check_fn = Cancel.check;
    try std.testing.expectError(error.Cancelled, selectFeatures(std.testing.allocator, &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, cancelled));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, selectFeatures(failing.allocator(), &.{ 0.1, 0.9, 0.9, 0.1 }, &.{ 1, 1 }, 2, cfg));
}

test "mass-only selection accepts a zero range crop with all full-head queries consumed" {
    const head: solver.Head = .{ .keys = &.{ 0, 1, 2, 3, 4 }, .values = &.{ 0, 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0, 0 }, .rows = 5, .dim = 1, .scale = 1 };
    // Query zero places all mass outside columns 1..3. Later rows are
    // informative; the zero crop remains a valid normalized full-head row.
    const probabilities = [_]f64{
        0, 0,   0,   0,   1,
        0, 0.8, 0.2, 0,   0,
        0, 0.1, 0.2, 0.7, 0,
    };
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var selected = try selectFromProbabilities(std.testing.allocator, head, 1, 3, &probabilities, cfg);
    defer selected.deinit();
    try std.testing.expectEqual(@as(usize, 3), selected.query_rows_consumed);
    try std.testing.expectEqual(@as(usize, 2), selected.indices.len);
    for (selected.weights) |weight| try std.testing.expect(std.math.isFinite(weight) and weight >= 0);
}

test "global denominator streaming matches monolithic across batch tail and fails before callback" {
    const q: usize = 65;
    const train = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(train);
    for (train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const head: solver.Head = .{ .keys = &.{ 0.2, -0.4, 0.7 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var streamed = try selectStreaming(std.testing.allocator, head, 0, 2, train, cfg);
    defer streamed.deinit();
    const full = try std.testing.allocator.alloc(f64, q * 3);
    defer std.testing.allocator.free(full);
    try solver.attendBatch(std.testing.allocator, head, train, full, null, null, cfg);
    const matrix = try std.testing.allocator.alloc(f64, q * 2);
    defer std.testing.allocator.free(matrix);
    const rho = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(rho);
    for (0..q) |i| {
        matrix[i * 2] = full[i * 3];
        matrix[i * 2 + 1] = full[i * 3 + 1];
        rho[i] = matrix[i * 2] + matrix[i * 2 + 1];
    }
    var monolithic = try selectFeatures(std.testing.allocator, matrix, rho, 2, cfg);
    defer monolithic.deinit();
    try std.testing.expectEqualSlices(usize, monolithic.indices, streamed.indices);
    const Blocked = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 1 << 20;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    Blocked.calls = 0;
    var capped = cfg;
    capped.backend = .{ .workspace_fn = Blocked.workspace, .attend_fn = Blocked.attend };
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectStreaming(std.testing.allocator, head, 0, 2, train, capped));
    try std.testing.expectEqual(@as(usize, 0), Blocked.calls);
    const Between = struct {
        var armed: bool = false;
        var calls: usize = 0;
        var processed: usize = 0;
        fn check(_: ?*anyopaque) !void {
            if (armed) return error.Cancelled;
        }
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, qs: []const f64, p: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            for (0..qs.len / h.dim) |i| _ = try solver.attend(h, qs[i * h.dim ..][0..h.dim], p[i * h.rows ..][0..h.rows], null);
            calls += 1;
            processed += qs.len / h.dim;
            armed = true;
        }
    };
    Between.armed = false;
    Between.calls = 0;
    Between.processed = 0;
    var cancelled = cfg;
    cancelled.control.check_fn = Between.check;
    cancelled.backend = .{ .workspace_fn = Between.workspace, .attend_fn = Between.attend };
    try std.testing.expectError(error.Cancelled, selectStreaming(std.testing.allocator, head, 0, 2, train, cancelled));
    try std.testing.expectEqual(@as(usize, 1), Between.calls);
    try std.testing.expectEqual(@as(usize, 64), Between.processed);
}

test "cached global selection q65 is exact and never calls backend" {
    const q: usize = 65;
    var train: [q]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const head: solver.Head = .{ .keys = &.{ 0.2, -0.4, 0.7 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var probabilities: [q * 3]f64 = undefined;
    try solver.attendBatch(std.testing.allocator, head, &train, &probabilities, null, null, cfg);
    var streamed = try selectStreaming(std.testing.allocator, head, 0, 2, &train, cfg);
    defer streamed.deinit();
    const Blocked = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 1 << 30;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    var cached_cfg = cfg;
    cached_cfg.backend = .{ .workspace_fn = Blocked.workspace, .attend_fn = Blocked.attend };
    var cached = try selectFromProbabilities(std.testing.allocator, head, 0, 2, &probabilities, cached_cfg);
    defer cached.deinit();
    try std.testing.expectEqual(@as(usize, 0), Blocked.calls);
    try std.testing.expectEqualSlices(usize, streamed.indices, cached.indices);
    try std.testing.expectEqualSlices(f64, streamed.weights, cached.weights);
    cached_cfg.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectFromProbabilities(std.testing.allocator, head, 0, 2, &probabilities, cached_cfg));
}

fn streamingAllocationFailure(allocator: std.mem.Allocator) !void {
    var train: [65]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const head: solver.Head = .{ .keys = &.{ 0.2, -0.4, 0.7 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var result = try selectStreaming(allocator, head, 0, 2, &train, cfg);
    result.deinit();
}

test "global denominator streaming unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, streamingAllocationFailure, .{});
}

test "global denominator selected rows refit without exact seeds end to end" {
    const global = @import("attention_match_global_residual_diagnostic.zig");
    const source: solver.Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 0.5, 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var selected = try selectStreaming(std.testing.allocator, source, 0, 2, &train, cfg);
    defer selected.deinit();
    try std.testing.expectEqualSlices(usize, &.{0}, selected.indices);
    var fitted = try global.fitOriginalContextChunk(std.testing.allocator, source, 0, 2, &.{}, selected.indices, &train, cfg);
    defer fitted.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 2), fitted.weights[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 3), fitted.values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 2)), fitted.bias[0], 1e-6);
    const keys = [_]f64{ source.keys[selected.indices[0]], source.keys[2] };
    const values = [_]f64{ fitted.values[0], source.values[2] };
    const biases = [_]f64{ fitted.bias[0], source.bias.?[2] };
    const compact: solver.Head = .{ .keys = &keys, .values = &values, .bias = &biases, .rows = 2, .dim = 1, .scale = 1 };
    const metrics = try solver.validationMetrics(std.testing.allocator, source, compact, &.{0.7}, .{ .target_rows = 2 });
    try std.testing.expect(metrics.max_relative_mass_error < 1e-6);
    try std.testing.expect(metrics.relative_output_rmse < 1e-6);
}
