// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only exact training-peak plus fitted-residual composition.
//! This module has no production engine integration or publication authority.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const Result = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    positions: []u64,
    coverage_ends: []u64,
    source_indices: []usize,
    fit_metrics: solver.Metrics,
    metrics: solver.Metrics,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.positions);
        self.allocator.free(self.coverage_ends);
        self.allocator.free(self.source_indices);
        self.* = undefined;
    }

    pub fn head(self: Result, source: solver.Head) solver.Head {
        return .{ .keys = self.keys, .values = self.values, .bias = self.bias, .rows = self.bias.len, .dim = source.dim, .scale = source.scale, .softcap = source.softcap };
    }
};

fn roundedF16(value: f64) !f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f16)) return error.NonFiniteAttention;
    return @floatCast(@as(f16, @floatCast(value)));
}

fn roundedF32(value: f64) !f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
    return @floatCast(@as(f32, @floatCast(value)));
}

/// Keeps `exact_indices` unchanged, fits only the residual rows, then validates
/// the rounded merged head against the original whole head. Every merged row
/// receives the chunk-end coverage boundary; no copied seed claims position+1.
fn fitDiagnosticImpl(
    allocator: std.mem.Allocator,
    source: solver.Head,
    positions: []const u64,
    exact_indices: []const usize,
    chunk_end: u64,
    train: []const f64,
    heldout: ?[]const f64,
    target_rows: usize,
    cfg: solver.Config,
) !Result {
    try source.validate();
    if (positions.len != source.rows or exact_indices.len == 0 or target_rows >= source.rows) return error.InvalidFitConfig;
    if (target_rows <= exact_indices.len) return error.CompactionFitRejected;
    for (exact_indices, 0..) |index, i| {
        if (index >= source.rows or std.mem.indexOfScalar(usize, exact_indices[0..i], index) != null) return error.InvalidFitConfig;
    }
    const residual_rows = source.rows - exact_indices.len;
    const residual_target = target_rows - exact_indices.len;
    if (residual_target == 0 or residual_target >= residual_rows) return error.CompactionFitRejected;

    const residual_keys = try allocator.alloc(f64, residual_rows * source.dim);
    defer allocator.free(residual_keys);
    const residual_values = try allocator.alloc(f64, residual_rows * source.dim);
    defer allocator.free(residual_values);
    const residual_bias = try allocator.alloc(f64, residual_rows);
    defer allocator.free(residual_bias);
    const residual_source_indices = try allocator.alloc(usize, residual_rows);
    defer allocator.free(residual_source_indices);
    var out: usize = 0;
    for (0..source.rows) |row| {
        if (std.mem.indexOfScalar(usize, exact_indices, row) != null) continue;
        @memcpy(residual_keys[out * source.dim ..][0..source.dim], source.keys[row * source.dim ..][0..source.dim]);
        @memcpy(residual_values[out * source.dim ..][0..source.dim], source.values[row * source.dim ..][0..source.dim]);
        residual_bias[out] = if (source.bias) |bias| bias[row] else 0;
        residual_source_indices[out] = row;
        out += 1;
    }
    const residual: solver.Head = .{ .keys = residual_keys, .values = residual_values, .bias = residual_bias, .rows = residual_rows, .dim = source.dim, .scale = source.scale, .softcap = source.softcap };
    var residual_cfg = cfg;
    residual_cfg.target_rows = residual_target;
    var fit_metrics: solver.Metrics = .{};
    residual_cfg.metrics = &fit_metrics;
    // The residual partition is only an intermediate. It must not recursively
    // reserve the already-copied peak rows or reject before exact-mass
    // composition. The merged original head receives the caller's strict gate.
    residual_cfg.mandatory_training_argmax = false;
    residual_cfg.mandatory_training_top_k = 1;
    residual_cfg.max_relative_mass_error = std.math.floatMax(f64);
    residual_cfg.max_relative_output_rmse = std.math.floatMax(f64);
    var fitted = if (heldout) |queries|
        try solver.fit(allocator, residual, train, queries, residual_cfg)
    else
        try solver.fitUnvalidatedDiagnostic(allocator, residual, train, residual_cfg);
    defer fitted.deinit();

    const keys = try allocator.alloc(f64, target_rows * source.dim);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, target_rows * source.dim);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, target_rows);
    errdefer allocator.free(bias);
    const result_positions = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(result_positions);
    const coverage = try allocator.alloc(u64, target_rows);
    errdefer allocator.free(coverage);
    const source_indices = try allocator.alloc(usize, target_rows);
    errdefer allocator.free(source_indices);
    @memcpy(source_indices[0..exact_indices.len], exact_indices);
    for (fitted.source_indices, exact_indices.len..) |index, destination| source_indices[destination] = residual_source_indices[index];
    std.mem.sort(usize, source_indices, {}, std.sort.asc(usize));

    for (source_indices, 0..) |source_index, destination| {
        const exact = std.mem.indexOfScalar(usize, exact_indices, source_index) != null;
        const fitted_index = if (exact) null else blk: {
            const residual_index = std.mem.indexOfScalar(usize, residual_source_indices, source_index) orelse return error.InvalidFitConfig;
            break :blk std.mem.indexOfScalar(usize, fitted.source_indices, residual_index) orelse return error.InvalidFitConfig;
        };
        if (exact) {
            @memcpy(keys[destination * source.dim ..][0..source.dim], source.keys[source_index * source.dim ..][0..source.dim]);
            @memcpy(values[destination * source.dim ..][0..source.dim], source.values[source_index * source.dim ..][0..source.dim]);
            bias[destination] = if (source.bias) |original| original[source_index] else 0;
        } else {
            const index = fitted_index.?;
            @memcpy(keys[destination * source.dim ..][0..source.dim], fitted.keys[index * source.dim ..][0..source.dim]);
            for (fitted.values[index * source.dim ..][0..source.dim], values[destination * source.dim ..][0..source.dim]) |value, *rounded| rounded.* = try roundedF16(value);
            bias[destination] = try roundedF32(fitted.bias[index]);
        }
        result_positions[destination] = positions[source_index];
        coverage[destination] = chunk_end;
    }
    var metrics: solver.Metrics = .{};
    if (heldout) |queries| {
        const compact: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = target_rows, .dim = source.dim, .scale = source.scale, .softcap = source.softcap };
        var validation_cfg = cfg;
        validation_cfg.backend = null;
        validation_cfg.metrics = null;
        metrics = try solver.validationMetrics(allocator, source, compact, queries, validation_cfg);
        if (metrics.max_relative_mass_error > validation_cfg.max_relative_mass_error or metrics.relative_output_rmse > validation_cfg.max_relative_output_rmse) return error.CompactionFitRejected;
    }
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias, .positions = result_positions, .coverage_ends = coverage, .source_indices = source_indices, .fit_metrics = fit_metrics, .metrics = metrics };
}

pub fn fitDiagnostic(allocator: std.mem.Allocator, source: solver.Head, positions: []const u64, exact_indices: []const usize, chunk_end: u64, train: []const f64, heldout: []const f64, target_rows: usize, cfg: solver.Config) !Result {
    return fitDiagnosticImpl(allocator, source, positions, exact_indices, chunk_end, train, heldout, target_rows, cfg);
}

/// Test-only exact-core merge constructor with no provisional validation or
/// authorization. The caller must perform final rounded whole-head validation.
pub fn fitUnvalidatedDiagnostic(allocator: std.mem.Allocator, source: solver.Head, positions: []const u64, exact_indices: []const usize, chunk_end: u64, train: []const f64, target_rows: usize, cfg: solver.Config) !Result {
    if (comptime !@import("builtin").is_test) @compileError("unvalidated exact-core constructor is diagnostic-test-only");
    return fitDiagnosticImpl(allocator, source, positions, exact_indices, chunk_end, train, null, target_rows, cfg);
}

test "exact peak plus residual fit composes mass output and chunk coverage" {
    const source: solver.Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    var result = try fitDiagnostic(std.testing.allocator, source, &.{ 10, 11, 12, 13 }, &.{0}, 14, &.{ 0, 1 }, &.{ -1, 2 }, 2, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.source_indices);
    try std.testing.expectEqualSlices(u64, &.{ 14, 14 }, result.coverage_ends);
    const rounded_residual_bias: f64 = @floatCast(@as(f32, @floatCast(@log(@as(f64, 3)))));
    const residual_mass = @exp(rounded_residual_bias);
    const expected_mass_error = @abs(@exp(@log(1 + residual_mass) - @log(@as(f64, 4))) - 1);
    const expected_output_error = @abs((2 + residual_mass * 6) / (1 + residual_mass) - 5) / 5;
    try std.testing.expectApproxEqAbs(expected_mass_error, result.metrics.max_relative_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(expected_output_error, result.metrics.relative_output_rmse, 1e-12);

    var unvalidated = try fitUnvalidatedDiagnostic(std.testing.allocator, source, &.{ 10, 11, 12, 13 }, &.{0}, 14, &.{ 0, 1 }, 2, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    defer unvalidated.deinit();
    try std.testing.expectEqualSlices(f64, result.keys, unvalidated.keys);
    try std.testing.expectEqualSlices(f64, result.values, unvalidated.values);
    try std.testing.expectEqualSlices(f64, result.bias, unvalidated.bias);
    try std.testing.expectEqualSlices(usize, result.source_indices, unvalidated.source_indices);
    try std.testing.expect(!unvalidated.metrics.quality_evaluated);

    try std.testing.expectError(error.CompactionFitRejected, fitDiagnostic(std.testing.allocator, source, &.{ 10, 11, 12, 13 }, &.{ 0, 1 }, 14, &.{ 0, 1 }, &.{ -1, 2 }, 2, .{ .target_rows = 2 }));
}
