// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact cached raw selection and warm global refit for one cumulative chunk.
const std = @import("std");
const solver = @import("attention_match.zig");
const chunk_cache = @import("attention_match_chunk_training_cache.zig");
const global_selection = @import("attention_match_global_selection.zig");
const global_residual = @import("attention_match_global_residual.zig");

pub const Selection = global_selection.Selection;
pub const Result = global_residual.Result;
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn select(allocator: std.mem.Allocator, cache: *const chunk_cache.Cache, source: solver.Head, train: []const f64, cfg: solver.Config) !Selection {
    try cache.validate(source, train, cache.chunk_start, cache.chunk_rows, cfg);
    const view = cache.view();
    if (view.query_rows == 0 or view.probabilities.len != try mul(view.query_rows, view.chunk_rows) or view.rho.len != view.query_rows) return error.InvalidFitConfig;
    const target_bytes = try mul(view.query_rows, @sizeOf(f64));
    const output_bytes = try add(target_bytes, try mul(cfg.target_rows, @sizeOf(usize) + @sizeOf(f64)));
    const nested = try add(try solver.matrixWorkspaceBytes(view.chunk_rows, cfg.target_rows, view.query_rows, 1), try mul(cfg.target_rows, @sizeOf(f64)));
    if (try add(output_bytes, nested) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var fit_cfg = cfg;
    fit_cfg.retain_mass_weights = true;
    fit_cfg.max_workspace_bytes -= output_bytes;
    const target = try allocator.alloc(f64, view.query_rows);
    defer allocator.free(target);
    @memset(target, 0);
    var fit = try solver.fitMatricesMassTargetPolicyDiagnostic(allocator, .{ .probabilities = view.probabilities, .targets = target, .rows = view.chunk_rows, .dim = 1, .query_rows = view.query_rows }, fit_cfg, view.rho, .raw);
    defer fit.deinit();
    const indices = try allocator.dupe(usize, fit.source_indices);
    errdefer allocator.free(indices);
    return .{ .allocator = allocator, .indices = indices, .weights = try allocator.dupe(f64, fit.mass_weights.?), .query_rows_consumed = view.query_rows };
}

pub fn refitWarm(allocator: std.mem.Allocator, cache: *const chunk_cache.Cache, source: solver.Head, train: []const f64, selected: []const usize, initial_weights: []const f64, cfg: solver.Config) !Result {
    try source.validate();
    try cache.validate(source, train, cache.chunk_start, cache.chunk_rows, cfg);
    const view = cache.view();
    if (train.len != try mul(view.query_rows, view.dim) or source.dim != view.dim or selected.len == 0 or selected.len != initial_weights.len or view.original_outputs.len != train.len) return error.InvalidFitConfig;
    const q = view.query_rows;
    const m = selected.len;
    const d = view.dim;
    const qm = try mul(q, m);
    const qd = try mul(q, d);
    for (selected, 0..) |index, i| if (index >= view.chunk_rows or std.mem.indexOfScalar(usize, selected[0..i], index) != null) return error.InvalidFitConfig;
    for (initial_weights) |weight| if (!std.math.isFinite(weight) or weight < 0) return error.InvalidFitConfig;
    var retained = try add(try mul(qm, 2), qd);
    retained = try add(retained, try mul(q, 2));
    retained = try add(retained, try mul(m, 4));
    retained = try add(retained, try mul(m, d));
    const retained_bytes = try mul(retained, @sizeOf(f64));
    const cpu_nested = try solver.workspaceBytes(m, m, q, d);
    const backend_nested = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, m, m, q, d) else 0;
    const nested = @max(cpu_nested, backend_nested);
    if (try add(retained_bytes, nested) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const a = try allocator.alloc(f64, qm);
    defer allocator.free(a);
    const target = try allocator.alloc(f64, qd);
    defer allocator.free(target);
    for (0..q) |i| {
        @memcpy(target[i * d ..][0..d], view.original_outputs[i * d ..][0..d]);
        const row = view.probabilities[i * view.chunk_rows ..][0..view.chunk_rows];
        for (0..view.chunk_rows) |local| {
            for (target[i * d ..][0..d], source.values[(view.chunk_start + local) * d ..][0..d]) |*y, v| y.* -= row[local] * v;
        }
        for (selected, 0..) |local, j| a[i * m + j] = row[local];
    }
    const weights = try allocator.dupe(f64, initial_weights);
    errdefer allocator.free(weights);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, m);
    defer allocator.free(norms);
    const sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, a, view.rho, q, m, weights, residual, norms, cfg, true);
    for (weights) |*weight| weight.* = @max(weight.*, cfg.min_weight);
    const basis = try allocator.alloc(f64, qm);
    defer allocator.free(basis);
    for (0..q) |i| {
        var denominator: f64 = 1 - view.rho[i];
        for (0..m) |j| {
            basis[i * m + j] = a[i * m + j] * weights[j];
            denominator += basis[i * m + j];
        }
        for (target[i * d ..][0..d], view.original_outputs[i * d ..][0..d]) |*fixed, y| fixed.* = denominator * y - fixed.*;
    }
    const values = try allocator.alloc(f64, m * d);
    errdefer allocator.free(values);
    try solver.fitValuesWithBackendDiagnostic(allocator, basis, target, q, m, d, cfg.ridge, cfg, values);
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    for (selected, weights, bias) |local, w, *b| b.* = (if (source.bias) |source_bias| source_bias[view.chunk_start + local] else 0) + @log(@max(w, cfg.min_weight));
    var rho_min = std.math.inf(f64);
    var rho_max: f64 = 0;
    var rho_sum: f64 = 0;
    for (view.rho) |rho| {
        rho_min = @min(rho_min, rho);
        rho_max = @max(rho_max, rho);
        rho_sum += rho;
    }
    return .{ .allocator = allocator, .values = values, .bias = bias, .weights = weights, .rho_min = rho_min, .rho_max = rho_max, .rho_mean = rho_sum / @as(f64, @floatFromInt(q)), .nnls_sweeps = sweeps };
}

test "one chunk cache preserves raw selection and warm refit decisions" {
    const source: solver.Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const train = [_]f64{ -0.5, 0.25, 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 };
    var cache = try chunk_cache.Cache.init(std.testing.allocator, source, &train, 0, 3, cfg, 1 << 20);
    defer cache.deinit();
    var cached_selection = try select(std.testing.allocator, &cache, source, &train, cfg);
    defer cached_selection.deinit();
    var direct_selection = try global_selection.selectStreaming(std.testing.allocator, source, 0, 3, &train, cfg);
    defer direct_selection.deinit();
    try std.testing.expectEqualSlices(usize, direct_selection.indices, cached_selection.indices);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_selection.weights), std.mem.sliceAsBytes(cached_selection.weights));
    var cached_fit = try refitWarm(std.testing.allocator, &cache, source, &train, cached_selection.indices, cached_selection.weights, cfg);
    defer cached_fit.deinit();
    var direct_fit = try global_residual.fitOriginalContextChunkWarm(std.testing.allocator, source, 0, 3, direct_selection.indices, direct_selection.weights, &train, cfg);
    defer direct_fit.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_fit.weights), std.mem.sliceAsBytes(cached_fit.weights));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_fit.values), std.mem.sliceAsBytes(cached_fit.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(direct_fit.bias), std.mem.sliceAsBytes(cached_fit.bias));
}
