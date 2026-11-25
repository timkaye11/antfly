// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Production-neutral chronological chunk admission. Selection and coefficient
//! fitting use training queries; heldout queries are used only to admit a
//! rounded candidate against the complete, cumulatively composed head.
const std = @import("std");
const solver = @import("attention_match.zig");
const selection = @import("attention_match_global_selection.zig");
const global = @import("attention_match_global_residual.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const training_chunk_cache = @import("attention_match_chunk_training_cache.zig");
const training_page_cache = @import("attention_match_training_page_cache.zig");
const cached_refit = @import("attention_match_chunk_cached_refit.zig");

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

fn fitChunk(allocator: std.mem.Allocator, original: solver.Head, range: admission.Range, train: []const f64, divisor: usize, cfg: solver.Config, training: ?*const training_chunk_cache.Cache) !OwnedChunk {
    const budget = std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig;
    if (budget == 0 or budget >= range.rows) return error.CompactionFitRejected;
    var local_cfg = cfg;
    local_cfg.target_rows = budget;
    local_cfg.selection = .residual_omp;
    local_cfg.mass_solver = .active_set;
    local_cfg.ridge = 1e-8;
    local_cfg.max_relative_mass_error = std.math.floatMax(f64);
    local_cfg.max_relative_output_rmse = std.math.floatMax(f64);
    var selected = if (training) |stored| blk: {
        try stored.validate(original, train, range.start, range.rows, local_cfg);
        break :blk try cached_refit.select(allocator, stored, original, train, local_cfg);
    } else try selection.selectStreaming(allocator, original, range.start, range.rows, train, local_cfg);
    defer selected.deinit();
    if (selected.indices.len != budget) return error.InvalidFitConfig;
    const selected_index_bytes = try std.math.mul(usize, selected.indices.len, @sizeOf(usize) + @sizeOf(f64));
    if (selected_index_bytes >= local_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var refit_cfg = local_cfg;
    refit_cfg.max_workspace_bytes -= selected_index_bytes;
    var fitted = if (training) |stored|
        try cached_refit.refitWarm(allocator, stored, original, train, selected.indices, selected.weights, refit_cfg)
    else
        try global.fitOriginalContextChunkWarm(allocator, original, range.start, range.rows, selected.indices, selected.weights, train, refit_cfg);
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

pub const TrainingCacheCounters = struct {
    page_cache_enabled: bool = false,
    chunks_built: usize = 0,
    attention_batches: usize = 0,
    full_original_training_attend_calls: usize = 0,
    reuse_attempts: usize = 0,
    retained_bytes_peak: usize = 0,
    pages_built: usize = 0,
    page_children: usize = 0,
    page_hits: usize = 0,
    page_misses: usize = 0,
    page_fallbacks_before_gpu: usize = 0,
};

/// Default-off typed authority. The exact implementation retains the existing
/// 1024-row parent crop and immutable original outputs; it does not alter the
/// chronological range ladder or final uncached validation.
pub const PageCachePolicy = enum { disabled, exact_1024_v1 };

const AdmittedReference = struct {
    reference: admission.ChunkReference,
    page_retired: bool,
};

fn beginReferenceAccountingPage(cache: *admission.Cache, chunk_index: usize, live_owned_bytes: usize, page: *?training_page_cache.Cache, counters: ?*TrainingCacheCounters) !AdmittedReference {
    const reference_live_bytes = try std.math.add(usize, live_owned_bytes, if (page.*) |stored| stored.retained_bytes else 0);
    const reference = cache.beginReference(chunk_index, reference_live_bytes) catch |err| switch (err) {
        error.CompactionWorkspaceLimit => retry: {
            if (page.* == null) return err;
            if (counters) |counter| counter.page_fallbacks_before_gpu += 1;
            if (page.*) |*stored| stored.deinit();
            page.* = null;
            break :retry try cache.beginReference(chunk_index, live_owned_bytes);
        },
        else => return err,
    };
    return .{ .reference = reference, .page_retired = page.* == null and reference_live_bytes != live_owned_bytes };
}

pub fn run(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize) !Result {
    return runWithPageCachePolicy(allocator, original, train, heldout, ranges, cfg, max_cache_bytes, .disabled, null);
}

pub fn runWithPageCachePolicy(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize, policy: PageCachePolicy, counters: ?*TrainingCacheCounters) !Result {
    if (max_cache_bytes == 0 or max_cache_bytes > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    return runDiagnosticPageCache(allocator, original, train, heldout, ranges, cfg, max_cache_bytes, true, policy == .exact_1024_v1, counters);
}

/// Private diagnostic A/B seam. Production callers continue through `run`,
/// whose exact training cache is enabled after same-input decision validation.
pub fn runDiagnostic(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize, use_training_cache: bool, counters: ?*TrainingCacheCounters) !Result {
    return runDiagnosticPageCache(allocator, original, train, heldout, ranges, cfg, max_cache_bytes, use_training_cache, false, counters);
}

/// Private A/B seam. Page reuse is never selected by `run` or legacy callers.
pub fn runDiagnosticPageCache(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, heldout: []const f64, ranges: []const admission.Range, cfg: solver.Config, max_cache_bytes: usize, use_training_cache: bool, use_page_cache: bool, counters: ?*TrainingCacheCounters) !Result {
    if (counters) |counter| counter.page_cache_enabled = use_page_cache;
    try original.validate();
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
    var page: ?training_page_cache.Cache = null;
    defer if (page) |*stored| stored.deinit();
    const divisors = [_]usize{ 4, 3, 2 };
    for (ranges, 0..) |range, chunk_index| {
        const admitted = try beginReferenceAccountingPage(&cache, chunk_index, live_owned_bytes, &page, counters);
        const page_retired_before_reference = admitted.page_retired;
        var reference = admitted.reference;
        defer reference.deinit();
        var training: ?training_chunk_cache.Cache = null;
        defer if (training) |*stored| stored.deinit();
        if (use_training_cache) {
            const available = try cache.remainingWorkspace(&reference, live_owned_bytes);
            if (use_page_cache and !page_retired_before_reference) page_attempt: {
                // Start each page at the first chronological fit range rather
                // than absolute row zero; protected prefixes are never cached.
                const aligned_start = range.start;
                const aligned_rows = @min(training_page_cache.page_rows, original.rows - aligned_start);
                const contained = if (page) |stored| range.start >= stored.page_start and range.start + range.rows <= stored.page_start + stored.page_len else false;
                if (!contained) {
                    if (counters) |counter| counter.page_misses += 1;
                    if (page) |*stored| stored.deinit();
                    page = null;
                    const parent_required = try training_page_cache.Cache.requiredBytes(original.rows, train.len / original.dim, aligned_rows, original.dim, cfg);
                    const child_retained = try training_page_cache.Cache.childRetainedBytes(train.len / original.dim, range.rows);
                    if (try std.math.add(usize, parent_required, child_retained) > available) {
                        if (counters) |counter| counter.page_fallbacks_before_gpu += 1;
                        break :page_attempt;
                    }
                    page = try training_page_cache.Cache.init(allocator, original, train, aligned_start, aligned_rows, cfg, available - child_retained);
                    if (counters) |counter| {
                        counter.pages_built += 1;
                        counter.attention_batches += page.?.attention_batches;
                        counter.full_original_training_attend_calls += page.?.attention_batches;
                        counter.retained_bytes_peak = @max(counter.retained_bytes_peak, page.?.retained_bytes);
                    }
                } else if (counters) |counter| {
                    counter.page_hits += 1;
                }
                var child = page.?.child(allocator, original, train, range.start, range.rows, cfg, available) catch |err| switch (err) {
                    error.CompactionWorkspaceLimit => {
                        if (counters) |counter| counter.page_fallbacks_before_gpu += 1;
                        if (page) |*stored| stored.deinit();
                        page = null;
                        break :page_attempt;
                    },
                    else => return err,
                };
                errdefer child.deinit();
                training = try training_chunk_cache.Cache.fromPageChild(original, train, &child, cfg);
                if (counters) |counter| {
                    counter.page_children += 1;
                    counter.chunks_built += 1;
                    counter.retained_bytes_peak = @max(counter.retained_bytes_peak, try std.math.add(usize, page.?.retained_bytes, training.?.retained_bytes));
                }
            }
            if (training == null) training = try training_chunk_cache.Cache.init(allocator, original, train, range.start, range.rows, cfg, available);
            if (counters) |counter| {
                if (training.?.attention_batches != 0) {
                    counter.chunks_built += 1;
                    counter.attention_batches += training.?.attention_batches;
                    counter.full_original_training_attend_calls += training.?.attention_batches;
                    counter.retained_bytes_peak = @max(counter.retained_bytes_peak, training.?.retained_bytes);
                }
            }
        }
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
            const training_bytes = if (training) |stored| try std.math.add(usize, stored.retained_bytes, if (use_page_cache and page != null and !stored.owns_original_outputs) page.?.retained_bytes else 0) else 0;
            const with_candidate = try std.math.add(usize, try std.math.add(usize, live_owned_bytes, training_bytes), candidate_bytes);
            var fit_cfg = cfg;
            // The returned candidate is allocated while selection/refit
            // temporaries and their result objects are still live.
            fit_cfg.max_workspace_bytes = @min(fit_cfg.max_workspace_bytes, try cache.remainingWorkspace(&reference, with_candidate));
            if (training != null) {
                if (counters) |counter| counter.reuse_attempts += 1;
            } else if (counters) |counter| {
                const batches = std.math.divCeil(usize, train.len / original.dim, 64) catch return error.InvalidAttentionShape;
                counter.full_original_training_attend_calls = try std.math.add(usize, counter.full_original_training_attend_calls, try std.math.mul(usize, batches, 2));
            }
            var candidate = fitChunk(allocator, original, range, train, divisor, fit_cfg, if (training) |*stored| stored else null) catch |err| switch (err) {
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
            live_owned_bytes = std.math.sub(usize, with_candidate, training_bytes) catch return error.InvalidAttentionShape;
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

test "cumulative training page cache preserves decisions head and metrics" {
    const source: solver.Head = .{
        .keys = &.{ -2, -1, 0, 1, 2, 3 },
        .values = &.{ 1, 2, 4, 8, 16, 32 },
        .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6 },
        .rows = 6,
        .dim = 1,
        .scale = 0.75,
        .softcap = 1.5,
    };
    const ranges = [_]admission.Range{ .{ .start = 1, .rows = 2 }, .{ .start = 3, .rows = 2 } };
    const cfg: solver.Config = .{ .target_rows = 1, .max_relative_mass_error = 1, .max_relative_output_rmse = 1, .max_workspace_bytes = 1 << 20 };
    var ordinary_counters: TrainingCacheCounters = .{};
    var ordinary = try runDiagnosticPageCache(std.testing.allocator, source, &.{ -1, 0, 1 }, &.{ -0.5, 0.5 }, &ranges, cfg, 1 << 20, true, false, &ordinary_counters);
    defer ordinary.deinit();
    var page_counters: TrainingCacheCounters = .{};
    var paged = try runDiagnosticPageCache(std.testing.allocator, source, &.{ -1, 0, 1 }, &.{ -0.5, 0.5 }, &ranges, cfg, 1 << 20, true, true, &page_counters);
    defer paged.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.keys), std.mem.sliceAsBytes(paged.head.keys));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.values), std.mem.sliceAsBytes(paged.head.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.bias.?), std.mem.sliceAsBytes(paged.head.bias.?));
    try std.testing.expectEqualSlices(usize, ordinary.source_indices, paged.source_indices);
    try std.testing.expectEqualSlices(usize, ordinary.coverage_source_indices, paged.coverage_source_indices);
    try std.testing.expectEqual(ordinary.final_mass_error, paged.final_mass_error);
    try std.testing.expectEqual(ordinary.final_output_rmse, paged.final_output_rmse);
    try std.testing.expectEqual(@as(usize, 1), page_counters.pages_built);
    try std.testing.expectEqual(@as(usize, 2), page_counters.page_children);
    try std.testing.expect(page_counters.full_original_training_attend_calls < ordinary_counters.full_original_training_attend_calls);
}

test "typed production page cache matches disabled artifact and records hits misses peak" {
    const source: solver.Head = .{ .keys = &.{ -2, -1, 0, 1, 2, 3 }, .values = &.{ 1, 2, 4, 8, 16, 32 }, .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6 }, .rows = 6, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const ranges = [_]admission.Range{ .{ .start = 1, .rows = 2 }, .{ .start = 3, .rows = 2 } };
    const cfg: solver.Config = .{ .target_rows = 1, .max_relative_mass_error = 1, .max_relative_output_rmse = 1, .max_workspace_bytes = 1 << 20 };
    var disabled_counters: TrainingCacheCounters = .{};
    var disabled = try runWithPageCachePolicy(std.testing.allocator, source, &.{ -1, 0, 1 }, &.{ -0.5, 0.5 }, &ranges, cfg, 1 << 20, .disabled, &disabled_counters);
    defer disabled.deinit();
    var exact_counters: TrainingCacheCounters = .{};
    var exact = try runWithPageCachePolicy(std.testing.allocator, source, &.{ -1, 0, 1 }, &.{ -0.5, 0.5 }, &ranges, cfg, 1 << 20, .exact_1024_v1, &exact_counters);
    defer exact.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(disabled.head.keys), std.mem.sliceAsBytes(exact.head.keys));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(disabled.head.values), std.mem.sliceAsBytes(exact.head.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(disabled.head.bias.?), std.mem.sliceAsBytes(exact.head.bias.?));
    try std.testing.expectEqualSlices(usize, disabled.source_indices, exact.source_indices);
    try std.testing.expectEqualSlices(usize, disabled.coverage_source_indices, exact.coverage_source_indices);
    try std.testing.expectEqual(disabled.chunks.len, exact.chunks.len);
    for (disabled.chunks, exact.chunks) |before, after| {
        try std.testing.expectEqual(before.start, after.start);
        try std.testing.expectEqual(before.source_rows, after.source_rows);
        try std.testing.expectEqual(before.chosen_divisor, after.chosen_divisor);
        try std.testing.expectEqual(before.candidate_rows, after.candidate_rows);
        try std.testing.expectEqual(before.attempts, after.attempts);
        try std.testing.expectEqual(before.last_candidate_metrics_valid, after.last_candidate_metrics_valid);
        try std.testing.expectEqualStrings(before.last_rejection_reason, after.last_rejection_reason);
        try std.testing.expectEqual(@as(u64, @bitCast(before.last_candidate_mass_error)), @as(u64, @bitCast(after.last_candidate_mass_error)));
        try std.testing.expectEqual(@as(u64, @bitCast(before.last_candidate_output_rmse)), @as(u64, @bitCast(after.last_candidate_output_rmse)));
        try std.testing.expectEqual(@as(u64, @bitCast(before.cumulative_mass_error)), @as(u64, @bitCast(after.cumulative_mass_error)));
        try std.testing.expectEqual(@as(u64, @bitCast(before.cumulative_output_rmse)), @as(u64, @bitCast(after.cumulative_output_rmse)));
    }
    try std.testing.expectEqual(disabled.final_mass_error, exact.final_mass_error);
    try std.testing.expectEqual(disabled.final_output_rmse, exact.final_output_rmse);
    try std.testing.expectEqual(@as(usize, 1), exact_counters.page_misses);
    try std.testing.expectEqual(@as(usize, 1), exact_counters.page_hits);
    try std.testing.expect(exact_counters.retained_bytes_peak > 0);
    try std.testing.expectEqual(@as(usize, 0), exact_counters.page_fallbacks_before_gpu);
    try std.testing.expectError(error.CompactionWorkspaceLimit, runWithPageCachePolicy(std.testing.allocator, source, &.{ -1, 0, 1 }, &.{ -0.5, 0.5 }, &ranges, cfg, (1 << 20) + 1, .exact_1024_v1, null));
}

test "retained page over next-reference cap retires before attention and allocation failures unwind" {
    const Fixture = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const source: solver.Head = .{ .keys = &.{ -3, -2, -1, 0, 1, 2, 3, 4 }, .values = &.{ 1, 2, 4, 8, 16, 32, 64, 128 }, .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 }, .rows = 8, .dim = 1, .scale = 0.75, .softcap = 1.5 };
            const ranges = [_]admission.Range{ .{ .start = 1, .rows = 1 }, .{ .start = 2, .rows = 5 } };
            const heldout = [_]f64{ -0.75, 0.25, 0.75 };
            var train: [65]f64 = undefined;
            for (&train, 0..) |*value, index| value.* = @as(f64, @floatFromInt(index)) / 64 - 1;
            const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };
            var cache = try admission.Cache.init(allocator, source, &heldout, &ranges, cfg, 1 << 20);
            defer cache.deinit();
            var page: ?training_page_cache.Cache = try training_page_cache.Cache.init(allocator, source, &train, 1, 7, cfg, 1 << 20);
            defer if (page) |*stored| stored.deinit();
            const without_page = try std.math.add(usize, try std.math.add(usize, cache.retained_bytes, try admission.referenceRetainedBytes(heldout.len, source.dim)), try admission.attendScratchBytes(ranges[1].rows, source.dim));
            try std.testing.expect(without_page < without_page + page.?.retained_bytes);
            cache.max_bytes = without_page;
            var counters: TrainingCacheCounters = .{};
            var admitted = try beginReferenceAccountingPage(&cache, 1, 0, &page, &counters);
            defer admitted.reference.deinit();
            try std.testing.expect(admitted.page_retired);
            try std.testing.expect(page == null);
            try std.testing.expectEqual(@as(usize, 1), counters.page_fallbacks_before_gpu);
        }
    };
    try Fixture.run(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Fixture.run, .{});
}

test "retained page reference retry cancellation retires page and unwinds" {
    const Context = struct {
        checks: usize = 0,
        cancel_at: usize = std.math.maxInt(usize),
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == self.cancel_at) return error.Cancelled;
        }
    };
    var context: Context = .{};
    const control: solver.Control = .{ .context = &context, .check_fn = Context.check };
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .control = control };
    const source: solver.Head = .{ .keys = &.{ -3, -2, -1, 0, 1, 2, 3, 4 }, .values = &.{ 1, 2, 4, 8, 16, 32, 64, 128 }, .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 }, .rows = 8, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const ranges = [_]admission.Range{ .{ .start = 1, .rows = 1 }, .{ .start = 2, .rows = 5 } };
    const heldout = [_]f64{ -0.75, 0.25, 0.75 };
    var train: [65]f64 = undefined;
    for (&train, 0..) |*value, index| value.* = @as(f64, @floatFromInt(index)) / 64 - 1;
    var cache = try admission.Cache.init(std.testing.allocator, source, &heldout, &ranges, cfg, 1 << 20);
    defer cache.deinit();
    var page: ?training_page_cache.Cache = try training_page_cache.Cache.init(std.testing.allocator, source, &train, 1, 7, cfg, 1 << 20);
    defer if (page) |*stored| stored.deinit();
    const without_page = try std.math.add(usize, try std.math.add(usize, cache.retained_bytes, try admission.referenceRetainedBytes(heldout.len, source.dim)), try admission.attendScratchBytes(ranges[1].rows, source.dim));
    cache.max_bytes = without_page;
    context.checks = 0;
    context.cancel_at = 2;
    var counters: TrainingCacheCounters = .{};
    try std.testing.expectError(error.Cancelled, beginReferenceAccountingPage(&cache, 1, 0, &page, &counters));
    try std.testing.expect(page == null);
    try std.testing.expectEqual(@as(usize, 1), counters.page_fallbacks_before_gpu);
    try std.testing.expect(cache.active_reference_token == null and cache.active_trial_token == null);
}

test "cumulative page cache later-child cap fallback retires parent and preserves ample-cap result" {
    const source: solver.Head = .{ .keys = &.{ -3, -2, -1, 0, 1, 2, 3, 4 }, .values = &.{ 1, 2, 4, 8, 16, 32, 64, 128 }, .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8 }, .rows = 8, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const ranges = [_]admission.Range{ .{ .start = 1, .rows = 2 }, .{ .start = 3, .rows = 4 } };
    const train = [_]f64{ -1, -0.5, 0, 0.5, 1 };
    const heldout = [_]f64{ -0.75, 0.25, 0.75 };
    const cfg: solver.Config = .{ .target_rows = 1, .max_relative_mass_error = 10, .max_relative_output_rmse = 10, .max_workspace_bytes = 1 << 20 };
    var ordinary = try runDiagnosticPageCache(std.testing.allocator, source, &train, &heldout, &ranges, cfg, 1 << 20, true, false, null);
    defer ordinary.deinit();
    var paged = try runDiagnosticPageCache(std.testing.allocator, source, &train, &heldout, &ranges, cfg, 1 << 20, true, true, null);
    defer paged.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.keys), std.mem.sliceAsBytes(paged.head.keys));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.values), std.mem.sliceAsBytes(paged.head.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.head.bias.?), std.mem.sliceAsBytes(paged.head.bias.?));
    try std.testing.expectEqualSlices(usize, ordinary.source_indices, paged.source_indices);
    try std.testing.expectEqual(ordinary.final_mass_error, paged.final_mass_error);
    try std.testing.expectEqual(ordinary.final_output_rmse, paged.final_output_rmse);

    // Keep the ample-cap equality fixture above small. This separate geometry
    // deterministically admits page construction and child one, then rejects
    // the larger contained child while still admitting its ordinary cache.
    const fallback_ranges = [_]admission.Range{ .{ .start = 1, .rows = 1 }, .{ .start = 2, .rows = 5 } };
    var fallback_train: [512]f64 = undefined;
    for (&fallback_train, 0..) |*value, index| value.* = @as(f64, @floatFromInt(index)) / 64 - 1;
    const fallback_q = fallback_train.len / source.dim;
    const initial_live = try std.math.add(usize, try std.math.mul(usize, fallback_ranges.len, @sizeOf(?OwnedChunk)), try std.math.mul(usize, fallback_ranges.len, @sizeOf(ChunkResult)));
    const cumulative_retained = try admission.baseRetainedBytes(heldout.len / source.dim, source.dim, fallback_ranges.len);
    const reference_retained = try admission.referenceRetainedBytes(heldout.len / source.dim, source.dim);
    const page_rows = source.rows - fallback_ranges[0].start;
    const parent_required = try training_page_cache.Cache.requiredBytes(source.rows, fallback_q, page_rows, source.dim, cfg);
    const parent_retained = try training_page_cache.Cache.retainedBytes(fallback_q, page_rows, source.dim);
    const first_child = try training_page_cache.Cache.childRetainedBytes(fallback_q, fallback_ranges[0].rows);
    const later_child = try training_page_cache.Cache.childRetainedBytes(fallback_q, fallback_ranges[1].rows);
    const ordinary_later = try training_chunk_cache.Cache.requiredBytes(source.rows, fallback_q, fallback_ranges[1].rows, source.dim, cfg);
    const available = parent_required + first_child;
    try std.testing.expect(parent_retained + later_child > available);
    try std.testing.expect(ordinary_later <= available);
    const fallback_cap = cumulative_retained + reference_retained + initial_live + available;
    var counters: TrainingCacheCounters = .{};
    if (runDiagnosticPageCache(std.testing.allocator, source, &fallback_train, &heldout, &fallback_ranges, cfg, fallback_cap, true, true, &counters)) |result_value| {
        var result = result_value;
        result.deinit();
    } else |err| switch (err) {
        error.CompactionWorkspaceLimit, error.CompactionFitRejected => {},
        else => return err,
    }
    try std.testing.expectEqual(@as(usize, 1), counters.pages_built);
    try std.testing.expectEqual(@as(usize, 1), counters.page_children);
    try std.testing.expectEqual(@as(usize, 1), counters.page_fallbacks_before_gpu);

    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, h: solver.Head, training: []const f64, validation: []const f64, rs: []const admission.Range, c: solver.Config) !void {
            var result = try runDiagnosticPageCache(a, h, training, validation, rs, c, 1 << 20, true, true, null);
            result.deinit();
        }
    }.run, .{ source, &train, &heldout, &ranges, cfg });
}
