// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only chronological chunk admission. Selection and coefficient
//! fitting use training queries; heldout queries are used only to admit a
//! rounded candidate against the complete, cumulatively composed head.
const std = @import("std");
const solver = @import("attention_match.zig");
const selection = @import("attention_match_global_selection_diagnostic.zig");
const global = @import("attention_match_global_residual_diagnostic.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const production_cumulative = @import("attention_match_cumulative_admission.zig");
pub const TrainingCacheCounters = production_cumulative.TrainingCacheCounters;

pub const ChunkResult = struct {
    start: usize,
    source_rows: usize,
    chosen_divisor: usize,
    candidate_rows: usize,
    attempts: usize,
    last_candidate_metrics_valid: bool,
    last_rejection_reason: []const u8,
    last_candidate_mass_error: f64,
    last_candidate_output_rmse: f64,
    cumulative_mass_error: f64,
    cumulative_output_rmse: f64,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    head: solver.Head,
    source_indices: []usize,
    coverage_source_indices: []usize,
    chunks: []ChunkResult,
    final_mass_error: f64,
    final_output_rmse: f64,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(@constCast(self.head.keys));
        self.allocator.free(@constCast(self.head.values));
        self.allocator.free(@constCast(self.head.bias.?));
        self.allocator.free(self.source_indices);
        self.allocator.free(self.coverage_source_indices);
        self.allocator.free(self.chunks);
        self.* = undefined;
    }
};

const OwnedChunk = struct {
    allocator: std.mem.Allocator,
    head: solver.Head,
    source_indices: []usize,
    fn deinit(self: *@This()) void {
        self.allocator.free(@constCast(self.head.keys));
        self.allocator.free(@constCast(self.head.values));
        self.allocator.free(@constCast(self.head.bias.?));
        self.allocator.free(self.source_indices);
        self.* = undefined;
    }
};

fn round16(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f16)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f16, @floatCast(value.*)));
    }
}

fn round32(values: []f64) !void {
    for (values) |*value| {
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f32, @floatCast(value.*)));
    }
}

fn fitChunk(allocator: std.mem.Allocator, original: solver.Head, range: admission.Range, train: []const f64, divisor: usize, cfg: solver.Config, warm_start: bool) !OwnedChunk {
    const budget = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig;
    if (budget == 0 or budget >= range.rows) return error.CompactionFitRejected;
    var local_cfg = cfg;
    local_cfg.target_rows = budget;
    local_cfg.selection = .residual_omp;
    local_cfg.mass_solver = .active_set;
    local_cfg.ridge = 1e-8;
    local_cfg.max_relative_mass_error = std.math.floatMax(f64);
    local_cfg.max_relative_output_rmse = std.math.floatMax(f64);
    var selected = try selection.selectStreaming(allocator, original, range.start, range.rows, train, local_cfg);
    defer selected.deinit();
    if (selected.indices.len != budget) return error.InvalidFitConfig;
    const selected_index_bytes = try std.math.mul(usize, selected.indices.len, @sizeOf(usize) + @sizeOf(f64));
    if (selected_index_bytes >= local_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var refit_cfg = local_cfg;
    refit_cfg.max_workspace_bytes -= selected_index_bytes;
    var fitted = if (warm_start)
        try global.fitOriginalContextChunkWarm(allocator, original, range.start, range.rows, selected.indices, selected.weights, train, refit_cfg)
    else
        try global.fitOriginalContextChunk(allocator, original, range.start, range.rows, &.{}, selected.indices, train, refit_cfg);
    defer fitted.deinit();
    const key_elements = try std.math.mul(usize, budget, original.dim);
    const keys = try allocator.alloc(f64, key_elements);
    errdefer allocator.free(keys);
    const values = try allocator.dupe(f64, fitted.values);
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f64, fitted.bias);
    errdefer allocator.free(bias);
    const source_indices = try allocator.alloc(usize, budget);
    errdefer allocator.free(source_indices);
    for (selected.indices, 0..) |source_index, destination| {
        if (source_index >= range.rows) return error.InvalidAttentionShape;
        @memcpy(keys[destination * original.dim ..][0..original.dim], original.keys[(range.start + source_index) * original.dim ..][0..original.dim]);
        source_indices[destination] = range.start + source_index;
    }
    // Checkpoint positions must be strictly increasing. Attention is
    // permutation-invariant, so sort the selected rows and their fitted
    // coefficients together by authoritative source position.
    for (1..budget) |index| {
        var cursor = index;
        while (cursor > 0 and source_indices[cursor] < source_indices[cursor - 1]) : (cursor -= 1) {
            std.mem.swap(usize, &source_indices[cursor], &source_indices[cursor - 1]);
            std.mem.swap(f64, &bias[cursor], &bias[cursor - 1]);
            for (0..original.dim) |column| {
                std.mem.swap(f64, &keys[cursor * original.dim + column], &keys[(cursor - 1) * original.dim + column]);
                std.mem.swap(f64, &values[cursor * original.dim + column], &values[(cursor - 1) * original.dim + column]);
            }
        }
    }
    try round16(keys);
    try round16(values);
    try round32(bias);
    return .{ .allocator = allocator, .head = .{ .keys = keys, .values = values, .bias = bias, .rows = budget, .dim = original.dim, .scale = original.scale, .softcap = original.softcap }, .source_indices = source_indices };
}

pub fn run(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize) !Result {
    return runWithWarmStart(allocator, original, train, heldout, ranges, cfg, max_cache_bytes, true);
}

pub fn runWithWarmStartForTest(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize, warm_start: bool) !Result {
    if (comptime !@import("builtin").is_test) @compileError("warm-start control is diagnostic-test-only");
    return runWithWarmStart(allocator, original, train, heldout, ranges, cfg, max_cache_bytes, warm_start);
}

fn runWithWarmStart(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize, warm_start: bool) !Result {
    if (comptime !@import("builtin").is_test) @compileError("chunk admission replay is diagnostic-test-only");
    try original.validate();
    try solver.validateConfigForTest(cfg, original.rows);
    if (original.bias == null or train.len == 0 or heldout.len == 0 or train.len % original.dim != 0 or heldout.len % original.dim != 0 or ranges.len == 0) return error.InvalidFitConfig;
    var heldout_cfg = cfg;
    heldout_cfg.backend = null;
    var cache = try admission.Cache.init(allocator, original, heldout, ranges, heldout_cfg, max_cache_bytes);
    defer cache.deinit();
    const choices = try allocator.alloc(?OwnedChunk, ranges.len);
    defer allocator.free(choices);
    @memset(choices, null);
    defer for (choices) |*choice| if (choice.*) |*owned| owned.deinit();
    const rows = try allocator.alloc(ChunkResult, ranges.len);
    errdefer allocator.free(rows);
    var live_owned_bytes = try std.math.add(usize, try std.math.mul(usize, choices.len, @sizeOf(?OwnedChunk)), try std.math.mul(usize, rows.len, @sizeOf(ChunkResult)));
    const divisors = [_]usize{ 4, 3, 2 };
    for (ranges, 0..) |range, chunk_index| {
        var reference = try cache.beginReference(chunk_index, live_owned_bytes);
        defer reference.deinit();
        var attempts: usize = 0;
        var last_metrics: admission.Metrics = .{};
        var last_metrics_valid = false;
        var last_rejection_reason: []const u8 = "none";
        var chosen_divisor: usize = 0;
        for (divisors) |divisor| {
            attempts += 1;
            const budget = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig;
            const candidate_elements = try std.math.add(usize, try std.math.mul(usize, try std.math.mul(usize, budget, original.dim), 2), budget);
            const candidate_bytes = try std.math.add(usize, try std.math.mul(usize, candidate_elements, @sizeOf(f64)), try std.math.mul(usize, budget, @sizeOf(usize)));
            const with_candidate = try std.math.add(usize, live_owned_bytes, candidate_bytes);
            var fit_cfg = cfg;
            // The returned candidate is allocated while selection/refit
            // temporaries and their result objects are still live.
            fit_cfg.max_workspace_bytes = @min(fit_cfg.max_workspace_bytes, try cache.remainingWorkspace(&reference, with_candidate));
            var candidate = fitChunk(allocator, original, range, train, divisor, fit_cfg, warm_start) catch |err| switch (err) {
                error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => {
                    last_rejection_reason = "numerical_candidate_rejected";
                    last_metrics = .{};
                    last_metrics_valid = false;
                    continue;
                },
                else => return err,
            };
            var keep = false;
            defer if (!keep) candidate.deinit();
            if (candidate.head.rows != budget) return error.InvalidAttentionShape;
            var trial = try cache.trial(&reference, candidate.head, with_candidate);
            defer trial.deinit();
            last_metrics = trial.metrics;
            last_metrics_valid = true;
            if (trial.metrics.max_relative_mass_error > cfg.max_relative_mass_error or trial.metrics.relative_output_rmse > cfg.max_relative_output_rmse) {
                last_rejection_reason = "heldout_gate";
                continue;
            }
            try cache.commit(&trial);
            choices[chunk_index] = candidate;
            live_owned_bytes = with_candidate;
            keep = true;
            chosen_divisor = divisor;
            last_rejection_reason = "none";
            break;
        }
        const cumulative = try cache.metrics();
        rows[chunk_index] = .{ .start = range.start, .source_rows = range.rows, .chosen_divisor = chosen_divisor, .candidate_rows = if (choices[chunk_index]) |choice| choice.head.rows else range.rows, .attempts = attempts, .last_candidate_metrics_valid = last_metrics_valid, .last_rejection_reason = last_rejection_reason, .last_candidate_mass_error = last_metrics.max_relative_mass_error, .last_candidate_output_rmse = last_metrics.relative_output_rmse, .cumulative_mass_error = cumulative.max_relative_mass_error, .cumulative_output_rmse = cumulative.relative_output_rmse };
    }
    var final_rows = original.rows;
    for (ranges, 0..) |range, index| if (choices[index]) |choice| {
        final_rows = std.math.sub(usize, final_rows, range.rows) catch return error.InvalidAttentionShape;
        final_rows = std.math.add(usize, final_rows, choice.head.rows) catch return error.InvalidAttentionShape;
    };
    const final_elements = try std.math.mul(usize, final_rows, original.dim);
    const final_numeric_bytes = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, final_elements, 2), final_rows), @sizeOf(f64));
    const final_owned_bytes = try std.math.add(usize, final_numeric_bytes, try std.math.mul(usize, try std.math.mul(usize, final_rows, 2), @sizeOf(usize)));
    const heldout_rows = heldout.len / original.dim;
    var validation_cfg = cfg;
    validation_cfg.backend = null;
    const validation_bytes = try solver.validationBytes(original.rows, final_rows, original.dim, heldout_rows, validation_cfg);
    if (try std.math.add(usize, try std.math.add(usize, try std.math.add(usize, cache.retained_bytes, live_owned_bytes), final_owned_bytes), validation_bytes) > max_cache_bytes) return error.CompactionWorkspaceLimit;
    const keys = try allocator.alloc(f64, final_elements);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, final_elements);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, final_rows);
    errdefer allocator.free(bias);
    const source_indices = try allocator.alloc(usize, final_rows);
    errdefer allocator.free(source_indices);
    const coverage_source_indices = try allocator.alloc(usize, final_rows);
    errdefer allocator.free(coverage_source_indices);
    var source_at: usize = 0;
    var destination_at: usize = 0;
    for (ranges, 0..) |range, index| {
        const prefix_rows = range.start - source_at;
        @memcpy(keys[destination_at * original.dim ..][0 .. prefix_rows * original.dim], original.keys[source_at * original.dim ..][0 .. prefix_rows * original.dim]);
        @memcpy(values[destination_at * original.dim ..][0 .. prefix_rows * original.dim], original.values[source_at * original.dim ..][0 .. prefix_rows * original.dim]);
        @memcpy(bias[destination_at..][0..prefix_rows], original.bias.?[source_at..][0..prefix_rows]);
        for (0..prefix_rows) |prefix_index| {
            source_indices[destination_at + prefix_index] = source_at + prefix_index;
            coverage_source_indices[destination_at + prefix_index] = source_at + prefix_index;
        }
        destination_at += prefix_rows;
        const replacement = if (choices[index]) |choice| choice.head else solver.Head{ .keys = original.keys[range.start * original.dim ..][0 .. range.rows * original.dim], .values = original.values[range.start * original.dim ..][0 .. range.rows * original.dim], .bias = original.bias.?[range.start..][0..range.rows], .rows = range.rows, .dim = original.dim, .scale = original.scale, .softcap = original.softcap };
        @memcpy(keys[destination_at * original.dim ..][0 .. replacement.rows * original.dim], replacement.keys);
        @memcpy(values[destination_at * original.dim ..][0 .. replacement.rows * original.dim], replacement.values);
        @memcpy(bias[destination_at..][0..replacement.rows], replacement.bias.?);
        if (choices[index]) |choice| {
            @memcpy(source_indices[destination_at..][0..replacement.rows], choice.source_indices);
            @memset(coverage_source_indices[destination_at..][0..replacement.rows], range.start + range.rows - 1);
        } else {
            for (0..replacement.rows) |row| {
                source_indices[destination_at + row] = range.start + row;
                coverage_source_indices[destination_at + row] = range.start + row;
            }
        }
        destination_at += replacement.rows;
        source_at = range.start + range.rows;
    }
    const tail_rows = original.rows - source_at;
    @memcpy(keys[destination_at * original.dim ..][0 .. tail_rows * original.dim], original.keys[source_at * original.dim ..][0 .. tail_rows * original.dim]);
    @memcpy(values[destination_at * original.dim ..][0 .. tail_rows * original.dim], original.values[source_at * original.dim ..][0 .. tail_rows * original.dim]);
    @memcpy(bias[destination_at..][0..tail_rows], original.bias.?[source_at..][0..tail_rows]);
    for (0..tail_rows) |index| {
        source_indices[destination_at + index] = source_at + index;
        coverage_source_indices[destination_at + index] = source_at + index;
    }
    destination_at += tail_rows;
    if (destination_at != final_rows) return error.InvalidAttentionShape;
    const composed: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = final_rows, .dim = original.dim, .scale = original.scale, .softcap = original.softcap };
    const metrics = try solver.validationMetrics(allocator, original, composed, heldout, .{ .target_rows = final_rows, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64), .control = cfg.control });
    const cached = try cache.metrics();
    if (@abs(metrics.max_relative_mass_error - cached.max_relative_mass_error) > 1e-10 or @abs(metrics.relative_output_rmse - cached.relative_output_rmse) > 1e-10) return error.InvalidAttentionShape;
    if (metrics.max_relative_mass_error > cfg.max_relative_mass_error or metrics.relative_output_rmse > cfg.max_relative_output_rmse) return error.CompactionFitRejected;
    return .{ .allocator = allocator, .head = composed, .source_indices = source_indices, .coverage_source_indices = coverage_source_indices, .chunks = rows, .final_mass_error = metrics.max_relative_mass_error, .final_output_rmse = metrics.relative_output_rmse };
}

fn belongs(batch: engine.ReferenceBatch, source: checkpoint.State, layout: engine.Layout, donor: usize, head: usize) bool {
    if (source.layer_donors[batch.consumer_layer] != donor) return false;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / source.donors[donor].heads.len;
    return batch.query_head / group == head;
}

pub fn runInput(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize) !Result {
    return runInputWithPolicy(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, false);
}

pub fn runInputWithPolicy(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool) !Result {
    return runInputWithPolicyObserved(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, reject_exact_active_fixed_point, null);
}

pub fn runInputWithPolicyObserved(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool, observer: ?solver.ActiveNnlsObserver) !Result {
    return runInputWithPolicyObservedWarm(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, reject_exact_active_fixed_point, observer, true);
}

pub fn runInputWithPolicyObservedWarm(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool, observer: ?solver.ActiveNnlsObserver, warm_start: bool) !Result {
    return runInputWithPolicyObservedWarmTrainingCache(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, reject_exact_active_fixed_point, observer, warm_start, false, null);
}

pub fn runInputWithPolicyObservedWarmTrainingCache(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool, observer: ?solver.ActiveNnlsObserver, warm_start: bool, training_cache: bool, counters: ?*TrainingCacheCounters) !Result {
    return runInputWithPolicyObservedWarmTrainingPageCache(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, reject_exact_active_fixed_point, observer, warm_start, training_cache, false, counters);
}

pub fn runInputWithPolicyObservedWarmTrainingPageCache(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool, observer: ?solver.ActiveNnlsObserver, warm_start: bool, training_cache: bool, page_cache: bool, counters: ?*TrainingCacheCounters) !Result {
    return runInputWithPolicyObservedWarmTrainingPageCacheGeometry(allocator, input, augmented, backend, donor, head_index, max_workspace_bytes, reject_exact_active_fixed_point, observer, warm_start, training_cache, page_cache, counters, null);
}

pub fn runInputWithPolicyObservedWarmTrainingPageCacheGeometry(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, max_workspace_bytes: usize, reject_exact_active_fixed_point: bool, observer: ?solver.ActiveNnlsObserver, warm_start: bool, training_cache: bool, page_cache: bool, counters: ?*TrainingCacheCounters, chunk_rows_override: ?usize) !Result {
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
    const fit = input.fit;
    const chunk_rows = chunk_rows_override orelse fit.chunk_rows;
    if (fit.prefix_rows >= source.rows or fit.recent_rows >= source.rows - fit.prefix_rows or chunk_rows == 0) return error.InvalidWholeFitInput;
    const old_end = source.rows - fit.recent_rows;
    const middle = old_end - fit.prefix_rows;
    const chunk_count = std.math.divCeil(usize, middle, chunk_rows) catch return error.InvalidWholeFitInput;
    const ranges = try allocator.alloc(admission.Range, chunk_count);
    defer allocator.free(ranges);
    var start = fit.prefix_rows;
    for (ranges) |*range| {
        const rows = @min(chunk_rows, old_end - start);
        range.* = .{ .start = start, .rows = rows };
        start += rows;
    }
    if (start != old_end) return error.InvalidWholeFitInput;
    var external_elements = try std.math.add(usize, try std.math.add(usize, keys.len, values.len), bias.len);
    external_elements = try std.math.add(usize, external_elements, train.len);
    external_elements = try std.math.add(usize, external_elements, heldout.len);
    external_elements = try std.math.add(usize, external_elements, try std.math.mul(usize, ranges.len, @sizeOf(admission.Range) / @sizeOf(f64)));
    const external_bytes = try std.math.mul(usize, external_elements, @sizeOf(f64));
    if (external_bytes >= max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const available = max_workspace_bytes - external_bytes;
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = fit.max_relative_mass_error, .max_relative_output_rmse = fit.max_relative_output_rmse, .max_sweeps = fit.max_sweeps, .nnls_tolerance = fit.nnls_tolerance, .min_weight = fit.min_weight, .max_workspace_bytes = available, .reject_exact_active_fixed_point = reject_exact_active_fixed_point, .active_nnls_observer = observer, .backend = backend };
    if (page_cache and !training_cache) return error.InvalidFitConfig;
    if (!training_cache) return runWithWarmStart(allocator, source, train, heldout, ranges, cfg, available, warm_start);
    if (!warm_start) return error.InvalidFitConfig;
    var cached = try production_cumulative.runDiagnosticPageCache(allocator, source, train, heldout, ranges, cfg, available, true, page_cache, counters);
    errdefer cached.deinit();
    const rows = try allocator.alloc(ChunkResult, cached.chunks.len);
    errdefer allocator.free(rows);
    for (cached.chunks, rows) |source_row, *destination| destination.* = .{ .start = source_row.start, .source_rows = source_row.source_rows, .chosen_divisor = source_row.chosen_divisor, .candidate_rows = source_row.candidate_rows, .attempts = source_row.attempts, .last_candidate_metrics_valid = source_row.last_candidate_metrics_valid, .last_rejection_reason = source_row.last_rejection_reason, .last_candidate_mass_error = source_row.last_candidate_mass_error, .last_candidate_output_rmse = source_row.last_candidate_output_rmse, .cumulative_mass_error = source_row.cumulative_mass_error, .cumulative_output_rmse = source_row.cumulative_output_rmse };
    allocator.free(cached.chunks);
    return .{ .allocator = allocator, .head = cached.head, .source_indices = cached.source_indices, .coverage_source_indices = cached.coverage_source_indices, .chunks = rows, .final_mass_error = cached.final_mass_error, .final_output_rmse = cached.final_output_rmse };
}

test "chunk admission replay policy is chronological and heldout-gated" {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var result = try run(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .max_relative_mass_error = 1, .max_relative_output_rmse = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.chunks.len);
    try std.testing.expect(result.chunks[0].chosen_divisor == 4 or result.chunks[0].chosen_divisor == 3 or result.chunks[0].chosen_divisor == 2 or result.chunks[0].chosen_divisor == 0);
    try std.testing.expect(result.final_mass_error <= 1 and result.final_output_rmse <= 1);
}

fn replayAllocationFailure(allocator: std.mem.Allocator) !void {
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var result = try run(allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .max_relative_mass_error = 1, .max_relative_output_rmse = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    result.deinit();
}

test "chunk admission replay maps unwind every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, replayAllocationFailure, .{});
}

test "chunk admission uses fit backend but forces heldout cache to CPU" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, masses: ?[]f64, control: solver.Control) !void {
            calls += 1;
            const q = queries.len / head.dim;
            for (0..q) |index| {
                const mass = try solver.attend(head, queries[index * head.dim ..][0..head.dim], probabilities[index * head.rows ..][0..head.rows], if (outputs) |out| out[index * head.dim ..][0..head.dim] else null);
                if (masses) |out| out[index] = mass;
            }
            _ = allocator;
            try control.check();
        }
    };
    Fake.calls = 0;
    const source: solver.Head = .{ .keys = &.{ 0, 0.5, 1, 1.5, 2, 2.5 }, .values = &.{ 1, 2, 3, 4, 5, 6 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const ranges = [_]admission.Range{.{ .start = 1, .rows = 4 }};
    var result = try run(std.testing.allocator, source, &.{ 0, 0.5, 1 }, &.{ 0.25, 0.75 }, &ranges, .{ .target_rows = 1, .max_relative_mass_error = 1, .max_relative_output_rmse = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } }, 1 << 20);
    defer result.deinit();
    try std.testing.expect(Fake.calls > 0);
}
