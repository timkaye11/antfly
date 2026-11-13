// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only fixed-support refit against the full original partition.
//! Selection, geometry, keys, positions, coverage and protected rows are immutable.
const std = @import("std");
const solver = @import("attention_match.zig");
const global = @import("attention_match_global_residual.zig");
const training_cache = @import("attention_match_global_training_cache.zig");

pub const fit_workspace_bytes: usize = 512 * 1024 * 1024;
pub const combined_cap_bytes: usize = training_cache.maximum_combined_bytes;

pub const Range = struct {
    start: usize,
    rows: usize,
    target_budget: usize,
    /// Global original-source row indices, not range-local indices.
    exact_core_source_keys: []const usize,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    bias: []f64,
    train_before: solver.Metrics,
    train_after: solver.Metrics,
    refitted_ranges: usize,
    refitted_rows: usize,
    nnls_sweeps: usize,
    cache_released_before_validation: bool,
    validation_peak_bytes: usize,
    owned_output_bytes: usize,
    maximum_index_scratch_bytes: usize,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}

/// The cache owns full-original probabilities and outputs. Reject unsupported
/// fanout before Cache.init can query or dispatch a backend.
pub fn requiredCacheBytes(query_count: usize, rows: usize, dim: usize) !usize {
    return mul(try add(try mul(query_count, rows), try mul(query_count, dim)), @sizeOf(f64));
}

pub fn requiredOwnedOutputBytes(candidate_rows: usize, dim: usize) !usize {
    return mul(try add(try mul(candidate_rows, dim), candidate_rows), @sizeOf(f64));
}

/// Four logical target-sized index vectors are live at the range peak. ArrayList
/// growth may retain spare capacity, so charge twice that exact logical bound.
pub fn maximumIndexScratchBytes(maximum_target_budget: usize) !usize {
    return mul(try mul(maximum_target_budget, 8), @sizeOf(usize));
}

pub fn preCacheAdmissionBytes(query_count: usize, original_rows: usize, dim: usize, candidate_rows: usize, maximum_target_budget: usize) !usize {
    var total = fit_workspace_bytes;
    total = try add(total, try requiredCacheBytes(query_count, original_rows, dim));
    total = try add(total, try requiredOwnedOutputBytes(candidate_rows, dim));
    total = try add(total, try maximumIndexScratchBytes(maximum_target_budget));
    if (total > combined_cap_bytes) return error.CompactionWorkspaceLimit;
    return total;
}

fn roundedF16(value: f64) !f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f16)) return error.NonFiniteAttention;
    return @floatCast(@as(f16, @floatCast(value)));
}

fn roundedF32(value: f64) !f64 {
    if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
    return @floatCast(@as(f32, @floatCast(value)));
}

fn validateImmutableInputs(
    original: solver.Head,
    original_positions: []const u64,
    candidate: solver.Head,
    candidate_positions: []const u64,
    candidate_coverage: []const u64,
    ranges: []const Range,
    train: []const f64,
) !void {
    try original.validate();
    try candidate.validate();
    if (original.bias == null or candidate.bias == null or original.dim != candidate.dim or
        original.scale != candidate.scale or original.softcap != candidate.softcap or
        original_positions.len != original.rows or candidate_positions.len != candidate.rows or
        candidate_coverage.len != candidate.rows or train.len == 0 or train.len % original.dim != 0)
        return error.InvalidFitConfig;
    for (original_positions, 0..) |position, index| {
        const expected = std.math.cast(u64, index) orelse return error.InvalidFitConfig;
        if (position != expected) return error.InvalidFitConfig;
    }
    for (candidate_positions, 0..) |position, index| {
        const source_key = std.math.cast(usize, position) orelse return error.InvalidFitConfig;
        if (source_key >= original.rows or (index != 0 and position <= candidate_positions[index - 1])) return error.InvalidFitConfig;
    }
    var previous_end: usize = 0;
    for (ranges) |range| {
        if (range.rows == 0 or range.start < previous_end or range.start > original.rows or
            range.rows > original.rows - range.start or range.target_budget == 0 or range.target_budget > range.rows)
            return error.InvalidFitConfig;
        previous_end = range.start + range.rows;
        for (range.exact_core_source_keys, 0..) |source_key, index| {
            if (source_key < range.start or source_key >= range.start + range.rows or
                std.mem.indexOfScalar(usize, range.exact_core_source_keys[0..index], source_key) != null)
                return error.InvalidFitConfig;
        }
    }
}

/// Returns owned replacement V/bias only. The caller continues to own and use
/// the immutable candidate K/positions/coverage arrays byte-for-byte.
pub fn refit(
    allocator: std.mem.Allocator,
    original: solver.Head,
    original_positions: []const u64,
    candidate: solver.Head,
    candidate_positions: []const u64,
    candidate_coverage: []const u64,
    ranges: []const Range,
    train: []const f64,
    input_cfg: solver.Config,
) !Result {
    try validateImmutableInputs(original, original_positions, candidate, candidate_positions, candidate_coverage, ranges, train);
    try input_cfg.control.check();
    if (input_cfg.max_workspace_bytes != fit_workspace_bytes) return error.InvalidFitConfig;
    const query_count = train.len / original.dim;
    var maximum_target_budget: usize = 0;
    for (ranges) |range| maximum_target_budget = @max(maximum_target_budget, range.target_budget);
    const owned_output_bytes = try requiredOwnedOutputBytes(candidate.rows, candidate.dim);
    const maximum_index_scratch_bytes = try maximumIndexScratchBytes(maximum_target_budget);
    _ = try preCacheAdmissionBytes(query_count, original.rows, original.dim, candidate.rows, maximum_target_budget);

    var cfg = input_cfg;
    cfg.max_workspace_bytes = fit_workspace_bytes;
    cfg.max_relative_mass_error = std.math.floatMax(f64);
    cfg.max_relative_output_rmse = std.math.floatMax(f64);
    var cache = try training_cache.Cache.init(allocator, original, train, cfg, combined_cap_bytes);
    var cache_live = true;
    defer if (cache_live) cache.deinit();

    const values = try allocator.dupe(f64, candidate.values);
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f64, candidate.bias.?);
    errdefer allocator.free(bias);
    var refitted_ranges: usize = 0;
    var refitted_rows: usize = 0;
    var nnls_sweeps: usize = 0;

    for (ranges) |range| {
        try cfg.control.check();
        var candidate_slots = std.ArrayList(usize).empty;
        defer candidate_slots.deinit(allocator);
        var source_keys = std.ArrayList(usize).empty;
        defer source_keys.deinit(allocator);
        for (candidate_positions, 0..) |position, slot| {
            const source_key = std.math.cast(usize, position) orelse return error.InvalidFitConfig;
            if (source_key < range.start or source_key >= range.start + range.rows) continue;
            try candidate_slots.append(allocator, slot);
            try source_keys.append(allocator, source_key);
        }
        if (candidate_slots.items.len != range.target_budget) return error.InvalidFitConfig;
        if (range.target_budget == range.rows) continue;

        var exact_local = try allocator.alloc(usize, range.exact_core_source_keys.len);
        defer allocator.free(exact_local);
        for (range.exact_core_source_keys, 0..) |source_key, index| {
            if (std.mem.indexOfScalar(usize, source_keys.items, source_key) == null) return error.InvalidFitConfig;
            exact_local[index] = source_key - range.start;
        }
        var residual_local = std.ArrayList(usize).empty;
        defer residual_local.deinit(allocator);
        var residual_slots = std.ArrayList(usize).empty;
        defer residual_slots.deinit(allocator);
        for (source_keys.items, candidate_slots.items) |source_key, slot| {
            if (std.mem.indexOfScalar(usize, range.exact_core_source_keys, source_key) != null) continue;
            try residual_local.append(allocator, source_key - range.start);
            try residual_slots.append(allocator, slot);
        }
        if (residual_local.items.len + exact_local.len != range.target_budget or residual_local.items.len == 0)
            return error.InvalidFitConfig;

        var fitted = try global.fitOriginalContextChunkCached(
            allocator,
            &cache,
            original,
            range.start,
            range.rows,
            exact_local,
            residual_local.items,
            train,
            cfg,
        );
        defer fitted.deinit();
        if (fitted.values.len != residual_slots.items.len * original.dim or fitted.bias.len != residual_slots.items.len)
            return error.InvalidFitConfig;
        for (residual_slots.items, 0..) |slot, fitted_index| {
            for (0..original.dim) |component| values[slot * original.dim + component] = try roundedF16(fitted.values[fitted_index * original.dim + component]);
            bias[slot] = try roundedF32(fitted.bias[fitted_index]);
        }
        refitted_ranges += 1;
        refitted_rows = try add(refitted_rows, residual_slots.items.len);
        nnls_sweeps = try add(nnls_sweeps, fitted.nnls_sweeps);
    }

    cache.deinit();
    cache_live = false;
    const before_cfg: solver.Config = .{ .target_rows = candidate.rows, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64), .control = cfg.control };
    if (owned_output_bytes != try mul(try add(values.len, bias.len), @sizeOf(f64))) return error.InvalidFitConfig;
    const validation_peak_bytes = try solver.validationBytes(original.rows, candidate.rows, original.dim, query_count, before_cfg);
    if (try add(owned_output_bytes, validation_peak_bytes) > combined_cap_bytes) return error.CompactionWorkspaceLimit;
    const train_before = try solver.validationMetrics(allocator, original, candidate, train, before_cfg);
    const after_head: solver.Head = .{ .keys = candidate.keys, .values = values, .bias = bias, .rows = candidate.rows, .dim = candidate.dim, .scale = candidate.scale, .softcap = candidate.softcap };
    const train_after = try solver.validationMetrics(allocator, original, after_head, train, before_cfg);
    return .{ .allocator = allocator, .values = values, .bias = bias, .train_before = train_before, .train_after = train_after, .refitted_ranges = refitted_ranges, .refitted_rows = refitted_rows, .nnls_sweeps = nnls_sweeps, .cache_released_before_validation = true, .validation_peak_bytes = validation_peak_bytes, .owned_output_bytes = owned_output_bytes, .maximum_index_scratch_bytes = maximum_index_scratch_bytes };
}

const TestFixture = struct {
    original: solver.Head,
    candidate: solver.Head,
    original_positions: [4]u64,
    candidate_positions: [3]u64,
    candidate_coverage: [3]u64,
};

fn fixture() TestFixture {
    return .{
        .original = .{ .keys = &.{ 0, 1, 2, 3 }, .values = &.{ 10, 20, 30, 40 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 },
        .candidate = .{ .keys = &.{ 0, 1, 3 }, .values = &.{ 10, 21, 40 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 },
        .original_positions = .{ 0, 1, 2, 3 },
        .candidate_positions = .{ 0, 1, 3 },
        .candidate_coverage = .{ 1, 3, 4 },
    };
}

test "fixed support global refit maps residual rows and preserves fixed bits" {
    const f = fixture();
    const keys_before = [3]f64{ f.candidate.keys[0], f.candidate.keys[1], f.candidate.keys[2] };
    const positions_before = f.candidate_positions;
    const coverage_before = f.candidate_coverage;
    const original_values = f.candidate.values[0];
    const original_bias = f.candidate.bias.?[0];
    var result = try refit(std.testing.allocator, f.original, &f.original_positions, f.candidate, &f.candidate_positions, &f.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{0} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes });
    defer result.deinit();
    try std.testing.expectEqual(original_values, result.values[0]);
    try std.testing.expectEqual(original_bias, result.bias[0]);
    try std.testing.expectEqual(f.candidate.values[2], result.values[2]);
    try std.testing.expectEqual(f.candidate.bias.?[2], result.bias[2]);
    try std.testing.expectEqual(@as(usize, 1), result.refitted_ranges);
    try std.testing.expectEqual(@as(usize, 1), result.refitted_rows);
    try std.testing.expect(result.cache_released_before_validation);
    try std.testing.expectEqualSlices(f64, &keys_before, f.candidate.keys);
    try std.testing.expectEqualSlices(u64, &positions_before, &f.candidate_positions);
    try std.testing.expectEqualSlices(u64, &coverage_before, &f.candidate_coverage);
}

test "fixed support global refit rejects a missing exact core without mutation" {
    const f = fixture();
    const values_before = [3]f64{ f.candidate.values[0], f.candidate.values[1], f.candidate.values[2] };
    const bias_before = [3]f64{ f.candidate.bias.?[0], f.candidate.bias.?[1], f.candidate.bias.?[2] };
    try std.testing.expectError(error.InvalidFitConfig, refit(std.testing.allocator, f.original, &f.original_positions, f.candidate, &f.candidate_positions, &f.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{2} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes }));
    try std.testing.expectEqualSlices(f64, &values_before, f.candidate.values);
    try std.testing.expectEqualSlices(f64, &bias_before, f.candidate.bias.?);
}

test "fixed support global refit rejects oversized full cache before backend dispatch" {
    _ = try preCacheAdmissionBytes(3_840, 7936, 512, 4908, 256);
    const q: usize = 15_360;
    try std.testing.expect((try requiredCacheBytes(q, 7936, 512)) > combined_cap_bytes - fit_workspace_bytes);
}

test "fixed support global refit admits cache alone but rejects simultaneous outputs and index scratch" {
    const rows = (combined_cap_bytes - fit_workspace_bytes) / @sizeOf(f64) - 64;
    const cache_bytes = try requiredCacheBytes(1, rows, 1);
    try std.testing.expect(cache_bytes + fit_workspace_bytes <= combined_cap_bytes);
    try std.testing.expectError(error.CompactionWorkspaceLimit, preCacheAdmissionBytes(1, rows, 1, 100, 100));
}

test "fixed support global refit rejects a smaller caller workspace before backend dispatch" {
    const Probe = struct {
        var workspace_calls: usize = 0;
        var attend_calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            workspace_calls += 1;
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            attend_calls += 1;
        }
    };
    Probe.workspace_calls = 0;
    Probe.attend_calls = 0;
    const f = fixture();
    try std.testing.expectError(error.InvalidFitConfig, refit(std.testing.allocator, f.original, &f.original_positions, f.candidate, &f.candidate_positions, &f.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{0} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes - 1, .backend = .{ .workspace_fn = Probe.workspace, .attend_fn = Probe.attend } }));
    try std.testing.expectEqual(@as(usize, 0), Probe.workspace_calls);
    try std.testing.expectEqual(@as(usize, 0), Probe.attend_calls);
}

test "fixed support global refit rejects a candidate position outside usize" {
    if (@bitSizeOf(usize) < @bitSizeOf(u64)) {
        var f = fixture();
        f.candidate_positions[2] = std.math.maxInt(u64);
        try std.testing.expectError(error.InvalidFitConfig, refit(std.testing.allocator, f.original, &f.original_positions, f.candidate, &f.candidate_positions, &f.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{0} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes }));
    }
}

test "fixed support global refit cancellation and allocation failure do not mutate input" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const f = fixture();
    try std.testing.expectError(error.Cancelled, refit(std.testing.allocator, f.original, &f.original_positions, f.candidate, &f.candidate_positions, &f.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{0} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes, .control = .{ .check_fn = Cancel.check } }));
    const Case = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const inner = fixture();
            var result = try refit(allocator, inner.original, &inner.original_positions, inner.candidate, &inner.candidate_positions, &inner.candidate_coverage, &.{.{ .start = 0, .rows = 3, .target_budget = 2, .exact_core_source_keys = &.{0} }}, &.{ 0, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = fit_workspace_bytes });
            defer result.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}
