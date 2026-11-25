// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Isolated diagnostic seam: reduce training queries before invoking the
//! unchanged cumulative chunk-admission replay. Heldout queries remain full.
const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const replay = @import("attention_match_chunk_admission_replay.zig");
const portfolio = @import("attention_match_query_portfolio.zig");

pub const Result = struct {
    replayed: replay.Result,
    original_train_rows: usize,
    portfolio_rows: usize,
    heldout_rows: usize,
    original_train_sha256: [32]u8,
    sample_indices_sha256: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.replayed.deinit();
        self.* = undefined;
    }
};

fn belongs(batch: engine.ReferenceBatch, source: checkpoint.State, layout: engine.Layout, donor: usize, head: usize) !bool {
    if (batch.consumer_layer >= source.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len) return error.InvalidWholeFitInput;
    if (source.layer_donors[batch.consumer_layer] != donor) return false;
    const head_count = source.donors[donor].heads.len;
    if (head_count == 0 or layout.query_heads_by_layer[batch.consumer_layer] % head_count != 0) return error.InvalidWholeFitInput;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / head_count;
    if (group == 0 or batch.query_head >= layout.query_heads_by_layer[batch.consumer_layer]) return error.InvalidWholeFitInput;
    return batch.query_head / group == head;
}

pub fn runInput(allocator: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, donor: usize, head_index: usize, cap: usize, expected_original_train_sha256: [32]u8, expected_sample_indices_sha256: [32]u8, max_workspace_bytes: usize) !Result {
    if (comptime !@import("builtin").is_test) @compileError("query portfolio replay is diagnostic-test-only");
    const state = input.checkpoint_owned.state;
    if (donor >= state.donors.len or head_index >= state.donors[donor].heads.len or state.donors[donor].window != 0) return error.InvalidWholeFitInput;
    const source_head = state.donors[donor].heads[head_index];
    if (source_head.compressed.rows() != 0 or source_head.dim == 0 or source_head.positions.len != source_head.keys.len / source_head.dim) return error.UnsupportedKvCheckpoint;
    var train_count: usize = 0;
    var heldout_count: usize = 0;
    for (input.batches) |batch| if (try belongs(batch, state, input.layout, donor, head_index)) {
        train_count = try std.math.add(usize, train_count, batch.train.len);
        heldout_count = try std.math.add(usize, heldout_count, batch.heldout.len);
    };
    for (augmented) |batch| if (try belongs(batch, state, input.layout, donor, head_index)) {
        if (batch.heldout.len != 0) return error.InvalidWholeFitInput;
        train_count = try std.math.add(usize, train_count, batch.train.len);
    };
    if (train_count == 0 or heldout_count == 0 or train_count % source_head.dim != 0 or heldout_count % source_head.dim != 0) return error.MissingCompactionConsumerQueries;
    const full_train = try allocator.alloc(f64, train_count);
    defer allocator.free(full_train);
    const heldout = try allocator.alloc(f64, heldout_count);
    defer allocator.free(heldout);
    var train_at: usize = 0;
    var heldout_at: usize = 0;
    for (input.batches) |batch| if (try belongs(batch, state, input.layout, donor, head_index)) {
        for (full_train[train_at..][0..batch.train.len], batch.train) |*destination, value| destination.* = value;
        train_at += batch.train.len;
        for (heldout[heldout_at..][0..batch.heldout.len], batch.heldout) |*destination, value| destination.* = value;
        heldout_at += batch.heldout.len;
    };
    for (augmented) |batch| if (try belongs(batch, state, input.layout, donor, head_index)) {
        for (full_train[train_at..][0..batch.train.len], batch.train) |*destination, value| destination.* = value;
        train_at += batch.train.len;
    };
    if (train_at != train_count or heldout_at != heldout_count) return error.InvalidWholeFitInput;
    var selected = try portfolio.select(allocator, full_train, source_head.dim, cap, .{});
    defer selected.deinit();
    if (!std.mem.eql(u8, &selected.original_train_sha256, &expected_original_train_sha256) or !std.mem.eql(u8, &selected.sample_indices_sha256, &expected_sample_indices_sha256)) return error.InvalidWholeFitInput;

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
    if (fit.prefix_rows >= source.rows or fit.recent_rows >= source.rows - fit.prefix_rows or fit.chunk_rows == 0) return error.InvalidWholeFitInput;
    const old_end = source.rows - fit.recent_rows;
    const middle = old_end - fit.prefix_rows;
    const chunk_count = std.math.divCeil(usize, middle, fit.chunk_rows) catch return error.InvalidWholeFitInput;
    const ranges = try allocator.alloc(admission.Range, chunk_count);
    defer allocator.free(ranges);
    var start = fit.prefix_rows;
    for (ranges) |*range| {
        const rows = @min(fit.chunk_rows, old_end - start);
        range.* = .{ .start = start, .rows = rows };
        start += rows;
    }
    if (start != old_end) return error.InvalidWholeFitInput;
    var external_bytes: usize = 0;
    for (&[_]usize{ keys.len, values.len, bias.len, full_train.len, selected.queries.len, heldout.len }) |elements| external_bytes = try std.math.add(usize, external_bytes, try std.math.mul(usize, elements, @sizeOf(f64)));
    external_bytes = try std.math.add(usize, external_bytes, try std.math.mul(usize, selected.indices.len, @sizeOf(u64)));
    external_bytes = try std.math.add(usize, external_bytes, try std.math.mul(usize, ranges.len, @sizeOf(admission.Range)));
    if (external_bytes >= max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const available = max_workspace_bytes - external_bytes;
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = fit.max_relative_mass_error, .max_relative_output_rmse = fit.max_relative_output_rmse, .max_sweeps = fit.max_sweeps, .nnls_tolerance = fit.nnls_tolerance, .min_weight = fit.min_weight, .max_workspace_bytes = available, .backend = backend };
    var replayed = try replay.run(allocator, source, selected.queries, heldout, ranges, cfg, available);
    errdefer replayed.deinit();
    return .{ .replayed = replayed, .original_train_rows = selected.original_rows, .portfolio_rows = selected.selected_rows, .heldout_rows = heldout.len / source.dim, .original_train_sha256 = selected.original_train_sha256, .sample_indices_sha256 = selected.sample_indices_sha256 };
}
