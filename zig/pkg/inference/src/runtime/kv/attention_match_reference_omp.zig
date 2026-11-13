// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only reproduction of the authors' default OMP refit policy:
//! raw correlation, unconstrained zero-ridge least squares after every
//! selection, then an elementwise positive floor before the next residual.
//! This module has no publication authority and is reachable only from tests.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("attention_match.zig");

pub const reference_floor: f64 = 1e-12;

pub const Metrics = struct {
    selection_steps: usize = 0,
    mass_refits: usize = 0,
    negative_coefficients: usize = 0,
    floored_coefficients: usize = 0,
    mass_ridge: f64 = 0,
    value_ridge: f64 = 0,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    bias: []f64,
    source_indices: []usize,
    weights: []f64,
    metrics: Metrics,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.source_indices);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn finite(xs: []const f64) !void {
    for (xs) |x| if (!std.math.isFinite(x)) return error.NonFiniteAttention;
}

/// Conservative retained host allocation bound. Backend workspaces are added
/// by `fitShiftedForTest` using the exact call shapes.
pub fn workspaceBytes(n: usize, m: usize, q: usize, d: usize) !usize {
    var elements = try mul(q, n); // shifted features
    elements = try add(elements, try mul(q, d)); // original outputs
    elements = try add(elements, try mul(q, m)); // selected basis
    elements = try add(elements, try mul(q, m)); // normalized compact P
    elements = try add(elements, try mul(m, d)); // fitted values
    elements = try add(elements, try mul(q, 3)); // target/current/residual
    elements = try add(elements, try mul(m, 3)); // weights/bias/indices charge
    elements = try add(elements, n); // selected bitmap, conservatively f64
    const retained = try mul(elements, @sizeOf(f64));
    return add(retained, try solver.matrixWorkspaceBytes(n, m, q, d));
}

/// Exact-reference diagnostic from a real source head. `cfg.ridge` remains a
/// valid positive production Config value, but both diagnostic QR calls pass
/// explicit ridge zero and report that fact.
pub fn fitShiftedForTest(allocator: std.mem.Allocator, source: solver.Head, train: []const f64, cfg: solver.Config) !Result {
    if (comptime !builtin.is_test) @compileError("reference OMP is diagnostic-test-only");
    try source.validate();
    try solver.validateConfigForTest(cfg, source.rows);
    if (cfg.selection != .residual_omp or cfg.target_rows >= source.rows or cfg.min_weight != reference_floor or
        train.len == 0 or train.len % source.dim != 0) return error.InvalidFitConfig;
    const q = train.len / source.dim;
    const n = source.rows;
    const m = cfg.target_rows;
    const host = try workspaceBytes(n, m, q, source.dim);
    const backend_peak = if (cfg.backend) |backend| @max(
        try backend.workspace_fn(backend.context, n, 1, q, source.dim),
        try backend.workspace_fn(backend.context, m, m, q, source.dim),
    ) else 0;
    if (try add(host, backend_peak) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;

    const features = try allocator.alloc(f64, try mul(q, n));
    defer allocator.free(features);
    const targets = try allocator.alloc(f64, try mul(q, source.dim));
    defer allocator.free(targets);
    const mass_rhs = try allocator.alloc(f64, q);
    defer allocator.free(mass_rhs);
    try solver.attendBatch(allocator, source, train, features, targets, null, cfg);
    for (0..q) |row_index| {
        try cfg.control.check();
        const row = features[row_index * n ..][0..n];
        var maximum: f64 = 0;
        var sum: f64 = 0;
        for (row) |value| {
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            maximum = @max(maximum, value);
            sum += value;
        }
        if (!std.math.isFinite(sum) or maximum <= 0) return error.NonFiniteAttention;
        mass_rhs[row_index] = sum / maximum;
        for (row) |*value| value.* /= maximum;
    }
    return fitFeatures(allocator, features, targets, source.bias, n, source.dim, q, cfg, mass_rhs);
}

/// Matrix seam for deterministic algorithm regressions and captured frontends.
pub fn fitFeaturesForTest(allocator: std.mem.Allocator, features: []const f64, targets: []const f64, bias: ?[]const f64, rows: usize, dim: usize, query_rows: usize, cfg: solver.Config, mass_rhs: []const f64) !Result {
    if (comptime !builtin.is_test) @compileError("reference OMP is diagnostic-test-only");
    try solver.validateConfigForTest(cfg, rows);
    if (cfg.selection != .residual_omp or cfg.target_rows >= rows or cfg.min_weight != reference_floor or
        features.len != try mul(query_rows, rows) or targets.len != try mul(query_rows, dim) or mass_rhs.len != query_rows) return error.InvalidFitConfig;
    try finite(features);
    try finite(targets);
    try finite(mass_rhs);
    for (features) |x| if (x < 0) return error.InvalidFitConfig;
    for (mass_rhs) |x| if (x <= 0) return error.InvalidFitConfig;
    if (bias) |values| {
        if (values.len != rows) return error.InvalidFitConfig;
        try finite(values);
    }
    const host = try workspaceBytes(rows, cfg.target_rows, query_rows, dim);
    const backend_peak = if (cfg.backend) |backend| @max(
        try backend.workspace_fn(backend.context, rows, 1, query_rows, dim),
        try backend.workspace_fn(backend.context, cfg.target_rows, cfg.target_rows, query_rows, dim),
    ) else 0;
    if (try add(host, backend_peak) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return fitFeatures(allocator, features, targets, bias, rows, dim, query_rows, cfg, mass_rhs);
}

fn fitFeatures(allocator: std.mem.Allocator, features: []const f64, targets: []const f64, source_bias: ?[]const f64, n: usize, d: usize, q: usize, cfg: solver.Config, mass_rhs: []const f64) !Result {
    const m = cfg.target_rows;
    const indices = try allocator.alloc(usize, m);
    errdefer allocator.free(indices);
    const weights = try allocator.alloc(f64, m);
    errdefer allocator.free(weights);
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    const values = try allocator.alloc(f64, try mul(m, d));
    errdefer allocator.free(values);
    const selected = try allocator.alloc(bool, n);
    defer allocator.free(selected);
    @memset(selected, false);
    const current = try allocator.alloc(f64, q);
    defer allocator.free(current);
    @memset(current, 0);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const basis = try allocator.alloc(f64, try mul(q, m));
    defer allocator.free(basis);
    var metrics: Metrics = .{};

    for (0..m) |step| {
        try cfg.control.check();
        for (residual, mass_rhs, current) |*r, target, fitted| r.* = target - fitted;
        var best: ?usize = null;
        var best_score: f64 = -std.math.inf(f64);
        for (0..n) |column| {
            if (selected[column]) continue;
            var score: f64 = 0;
            for (0..q) |row| score += features[row * n + column] * residual[row];
            if (!std.math.isFinite(score)) return error.NonFiniteAttention;
            if (best == null or score > best_score) {
                best = column;
                best_score = score;
            }
        }
        const chosen = best orelse return error.InvalidFitConfig;
        indices[step] = chosen;
        selected[chosen] = true;
        const count = step + 1;
        for (0..q) |row| for (indices[0..count], 0..) |column, j| {
            basis[row * count + j] = features[row * n + column];
        };
        @memset(weights[0..count], 0);
        try solver.fitValuesWithBackendForTest(allocator, basis[0 .. q * count], mass_rhs, q, count, 1, 0, cfg, weights[0..count]);
        metrics.mass_refits += 1;
        for (weights[0..count]) |*weight| {
            if (weight.* < 0) metrics.negative_coefficients += 1;
            if (weight.* < reference_floor) {
                weight.* = reference_floor;
                metrics.floored_coefficients += 1;
            }
        }
        for (0..q) |row| {
            var fitted: f64 = 0;
            for (weights[0..count], 0..) |weight, j| fitted += basis[row * count + j] * weight;
            if (!std.math.isFinite(fitted)) return error.NonFiniteAttention;
            current[row] = fitted;
        }
        metrics.selection_steps += 1;
    }

    // Sorting is a representation canonicalization only. Move the associated
    // weights with their indices; do not refit after selection.
    for (1..m) |i| {
        const index = indices[i];
        const weight = weights[i];
        var j = i;
        while (j > 0 and indices[j - 1] > index) : (j -= 1) {
            indices[j] = indices[j - 1];
            weights[j] = weights[j - 1];
        }
        indices[j] = index;
        weights[j] = weight;
    }

    const fitted_probabilities = try allocator.alloc(f64, try mul(q, m));
    defer allocator.free(fitted_probabilities);
    for (0..q) |row| {
        var sum: f64 = 0;
        for (indices, weights, 0..) |column, weight, j| {
            const p = features[row * n + column] * weight;
            fitted_probabilities[row * m + j] = p;
            sum += p;
        }
        if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
        for (fitted_probabilities[row * m ..][0..m]) |*p| p.* /= sum;
    }
    @memset(values, 0);
    try solver.fitValuesWithBackendForTest(allocator, fitted_probabilities, targets, q, m, d, 0, cfg, values);
    for (indices, weights, bias) |column, weight, *out| {
        out.* = (if (source_bias) |source| source[column] else 0) + @log(weight);
    }
    try finite(values);
    try finite(bias);
    return .{ .allocator = allocator, .values = values, .bias = bias, .source_indices = indices, .weights = weights, .metrics = metrics };
}

test "reference per-step LS floor changes selected basis versus active NNLS fixture" {
    const original_a = [_]f64{
        5,  34, 13, 3,  55, 55, 8,
        21, 8,  21, 55, 2,  1,  13,
        1,  5,  1,  55, 2,  34, 5,
        5,  34, 1,  5,  55, 55, 3,
        21, 1,  1,  5,  21, 8,  21,
    };
    const original_z = [_]f64{ 173, 121, 103, 158, 78 };
    var a = original_a;
    var z = original_z;
    for (&a) |*value| value.* /= 55;
    for (&z) |*value| value.* /= 55;
    const y = [_]f64{ 0, 0, 0, 0, 0 };
    const cfg: solver.Config = .{ .target_rows = 4, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var reference = try fitFeaturesForTest(std.testing.allocator, &a, &y, null, 7, 1, 5, cfg, &z);
    defer reference.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 4, 5 }, reference.source_indices);
    try std.testing.expect(reference.metrics.negative_coefficients > 0);
    try std.testing.expect(reference.metrics.floored_coefficients > 0);

    var nnls = try solver.fitMatricesMassTargetPolicyForTest(std.testing.allocator, .{ .probabilities = &a, .targets = &y, .rows = 7, .dim = 1, .query_rows = 5 }, cfg, &z, .raw);
    defer nnls.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 4, 5 }, nnls.source_indices);
    try std.testing.expect(!std.mem.eql(usize, nnls.source_indices, reference.source_indices));
}

test "reference LS floor agrees with positive one-column solution" {
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var result = try fitFeaturesForTest(std.testing.allocator, &.{ 1, 2, 3, 1, 1, 1 }, &.{ 0, 0 }, null, 3, 1, 2, cfg, &.{ 6, 3 });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.metrics.negative_coefficients);
    try std.testing.expectEqual(@as(usize, 0), result.metrics.floored_coefficients);
    try std.testing.expect(result.weights[0] > reference_floor);
}

test "reference LS floor passes literal ridge zero to every backend mass and value solve" {
    const Recorder = struct {
        var calls: usize = 0;
        var nonzero_ridge_calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(_: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
            calls += 1;
            if (ridge != 0) nonzero_ridge_calls += 1;
            try solver.fitValuesReferenceForTest(allocator, a, y, q, m, d, ridge, control, out);
        }
    };
    Recorder.calls = 0;
    Recorder.nonzero_ridge_calls = 0;
    const backend: solver.Backend = .{ .workspace_fn = Recorder.workspace, .values_fn = Recorder.values };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 0.75, .backend = backend, .max_workspace_bytes = 1 << 20 };
    var result = try fitFeaturesForTest(std.testing.allocator, &.{ 1, 2, 3, 1, 1, 1 }, &.{ 0, 0 }, null, 3, 1, 2, cfg, &.{ 6, 3 });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), Recorder.calls);
    try std.testing.expectEqual(@as(usize, 0), Recorder.nonzero_ridge_calls);
    try std.testing.expectEqual(@as(f64, 0), result.metrics.mass_ridge);
    try std.testing.expectEqual(@as(f64, 0), result.metrics.value_ridge);
}

test "reference LS floor propagates rank failure and rejects nonfinite cap cancellation and OOM" {
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    const RankFailure = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, _: solver.Control, _: []f64) !void {
            return error.SingularValueFit;
        }
    };
    var rank = cfg;
    rank.backend = .{ .workspace_fn = RankFailure.workspace, .values_fn = RankFailure.values };
    try std.testing.expectError(error.SingularValueFit, fitFeaturesForTest(std.testing.allocator, &.{ 1, 2, 3, 3, 2, 1 }, &.{ 0, 0 }, null, 3, 1, 2, rank, &.{ 6, 6 }));
    var tiny = cfg;
    tiny.max_workspace_bytes = (try workspaceBytes(3, 2, 2, 1)) - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitFeaturesForTest(std.testing.allocator, &.{ 1, 2, 3, 3, 2, 1 }, &.{ 0, 0 }, null, 3, 1, 2, tiny, &.{ 6, 6 }));
    try std.testing.expectError(error.NonFiniteAttention, fitFeaturesForTest(std.testing.allocator, &.{ 1, std.math.nan(f64), 3, 3, 2, 1 }, &.{ 0, 0 }, null, 3, 1, 2, cfg, &.{ 6, 6 }));
    try std.testing.expectError(error.InvalidAttentionShape, workspaceBytes(std.math.maxInt(usize), 2, 2, 1));

    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var cancelled = cfg;
    cancelled.control.check_fn = Cancel.check;
    try std.testing.expectError(error.Cancelled, fitFeaturesForTest(std.testing.allocator, &.{ 1, 2, 3, 3, 2, 1 }, &.{ 0, 0 }, null, 3, 1, 2, cancelled, &.{ 6, 6 }));

    var storage: [64]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    try std.testing.expectError(error.OutOfMemory, fitFeaturesForTest(fixed.allocator(), &.{ 1, 2, 3, 3, 2, 1 }, &.{ 0, 0 }, null, 3, 1, 2, cfg, &.{ 6, 6 }));
}
