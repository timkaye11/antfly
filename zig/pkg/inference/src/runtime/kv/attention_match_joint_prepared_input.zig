// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Owned preparation shared by diagnostic replay and the DEV native route.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const whole = @import("attention_match_whole_fit_input.zig");

pub const PreparedInput = struct {
    allocator: std.mem.Allocator,
    source: solver.Head,
    train: []f64,
    heldout: []f64,
    ranges: []admission.Range,
    cfg: solver.Config,
    external_live_bytes: usize,
    combined_cap: usize,

    pub fn deinit(self: *PreparedInput) void {
        self.allocator.free(@constCast(self.source.keys));
        self.allocator.free(@constCast(self.source.values));
        self.allocator.free(@constCast(self.source.bias.?));
        self.allocator.free(self.train);
        self.allocator.free(self.heldout);
        self.allocator.free(self.ranges);
        self.* = undefined;
    }
};

fn belongs(batch: engine.ReferenceBatch, state: checkpoint.State, layout: engine.Layout, donor: usize, head: usize) bool {
    if (state.layer_donors[batch.consumer_layer] != donor) return false;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / state.donors[donor].heads.len;
    return batch.query_head / group == head;
}

pub fn prepareInput(
    allocator: std.mem.Allocator,
    input: *const whole.OwnedInput,
    augmented: []const engine.ReferenceBatch,
    backend: solver.Backend,
    donor: usize,
    head_index: usize,
    branch_chunk_rows: usize,
    combined_cap: usize,
    control: solver.Control,
) !PreparedInput {
    try control.check();
    const state = input.checkpoint_owned.state;
    if (donor >= state.donors.len or head_index >= state.donors[donor].heads.len or
        state.donors[donor].window != 0 or branch_chunk_rows != 256 or combined_cap == 0)
        return error.InvalidWholeFitInput;
    const source_head = state.donors[donor].heads[head_index];
    if (source_head.dim == 0 or source_head.compressed.rows() != 0)
        return error.UnsupportedKvCheckpoint;
    const source_elements = std.math.mul(usize, source_head.positions.len, source_head.dim) catch return error.InvalidWholeFitInput;
    if (source_head.keys.len != source_elements or source_head.values.len != source_elements or source_head.bias.len != source_head.positions.len)
        return error.UnsupportedKvCheckpoint;

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
    if (train_count % source_head.dim != 0 or heldout_count % source_head.dim != 0) return error.InvalidWholeFitInput;

    const fit = input.fit;
    if (fit.prefix_rows >= source_head.positions.len or fit.recent_rows >= source_head.positions.len - fit.prefix_rows) return error.InvalidWholeFitInput;
    const old_end = source_head.positions.len - fit.recent_rows;
    const middle = old_end - fit.prefix_rows;
    const range_count = std.math.divCeil(usize, middle, branch_chunk_rows) catch return error.InvalidWholeFitInput;
    var elements = try std.math.add(usize, try std.math.mul(usize, source_elements, 2), source_head.positions.len);
    elements = try std.math.add(usize, elements, try std.math.add(usize, train_count, heldout_count));
    var external = try std.math.mul(usize, elements, @sizeOf(f64));
    external = try std.math.add(usize, external, try std.math.mul(usize, range_count, @sizeOf(admission.Range)));
    if (external > combined_cap) return error.CompactionWorkspaceLimit;

    const train = try allocator.alloc(f64, train_count);
    errdefer allocator.free(train);
    const heldout = try allocator.alloc(f64, heldout_count);
    errdefer allocator.free(heldout);
    var ti: usize = 0;
    var hi: usize = 0;
    for (input.batches) |batch| if (belongs(batch, state, input.layout, donor, head_index)) {
        try control.check();
        for (batch.train, 0..) |value, local| {
            if ((local & 4095) == 0) try control.check();
            train[ti] = value;
            ti += 1;
        }
        for (batch.heldout, 0..) |value, local| {
            if ((local & 4095) == 0) try control.check();
            heldout[hi] = value;
            hi += 1;
        }
    };
    for (augmented) |batch| if (belongs(batch, state, input.layout, donor, head_index)) {
        try control.check();
        for (batch.train, 0..) |value, local| {
            if ((local & 4095) == 0) try control.check();
            train[ti] = value;
            ti += 1;
        }
    };
    if (ti != train.len or hi != heldout.len) return error.InvalidWholeFitInput;

    const keys = try allocator.alloc(f64, source_head.keys.len);
    errdefer allocator.free(keys);
    for (source_head.keys, keys, 0..) |value, *out, i| {
        if ((i & 4095) == 0) try control.check();
        out.* = value;
    }
    const values = try allocator.alloc(f64, source_head.values.len);
    errdefer allocator.free(values);
    for (source_head.values, values, 0..) |value, *out, i| {
        if ((i & 4095) == 0) try control.check();
        out.* = value;
    }
    const bias = try allocator.alloc(f64, source_head.bias.len);
    errdefer allocator.free(bias);
    for (source_head.bias, bias, 0..) |value, *out, i| {
        if ((i & 4095) == 0) try control.check();
        out.* = value;
    }
    const source: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = source_head.positions.len, .dim = source_head.dim, .scale = input.layout.scales_by_donor[donor], .softcap = input.layout.softcaps_by_donor[donor] };
    try source.validate();
    if (train.len % source.dim != 0 or heldout.len % source.dim != 0) return error.InvalidWholeFitInput;

    const ranges = try allocator.alloc(admission.Range, range_count);
    errdefer allocator.free(ranges);
    var start = fit.prefix_rows;
    for (ranges) |*range| {
        try control.check();
        const rows = @min(branch_chunk_rows, old_end - start);
        range.* = .{ .start = start, .rows = rows };
        start = try std.math.add(usize, start, rows);
    }
    if (start != old_end) return error.InvalidWholeFitInput;
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_relative_mass_error = fit.max_relative_mass_error, .max_relative_output_rmse = fit.max_relative_output_rmse, .max_sweeps = fit.max_sweeps, .nnls_tolerance = fit.nnls_tolerance, .min_weight = fit.min_weight, .max_workspace_bytes = combined_cap - external, .backend = backend, .control = control };
    return .{ .allocator = allocator, .source = source, .train = train, .heldout = heldout, .ranges = ranges, .cfg = cfg, .external_live_bytes = external, .combined_cap = combined_cap };
}

const PreparedFixture = struct {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const positions = [_]u64{ 0, 1, 2, 3 };
    const coverage = [_]u64{ 1, 2, 3, 4 };
    const keys = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const values = [_]f32{ -1, -2, -3, -4, -5, -6, -7, -8 };
    const bias = [_]f32{ 0.5, -0.5, 1.5, -1.5 };
    const heads = [_]checkpoint.Head{.{ .dim = 2, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias }};
    const donors = [_]checkpoint.Donor{.{ .heads = &heads }};
    const layer_donors = [_]u32{0};
    const primary_train = [_]f32{ 10, 11, 12, 13 };
    const primary_heldout = [_]f32{ 20, 21 };
    const primary = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &primary_train, .heldout = &primary_heldout, .train_positions = &.{ 4, 5 }, .heldout_positions = &.{6} }};
    const augmented_train = [_]f32{ 30, 31 };
    const augmented = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &augmented_train, .heldout = &.{}, .train_positions = &.{7}, .heldout_positions = &.{} }};
    const query_heads = [_]u32{1};
    const scales = [_]f64{1};
    const softcaps = [_]f64{0};

    fn input() whole.OwnedInput {
        return .{
            .allocator = undefined,
            .checkpoint_owned = .{ .allocator = undefined, .state = .{ .identity = identity, .logical_length = 4, .layer_donors = &layer_donors, .donors = &donors } },
            .checkpoint_identity = identity,
            .checkpoint_payload_sha256 = @splat(5),
            .checkpoint_payload_bytes = 0,
            .batches = @constCast(&primary),
            .train = undefined,
            .heldout = undefined,
            .train_positions = undefined,
            .heldout_positions = undefined,
            .consumer_layers = undefined,
            .heads = undefined,
            .train_counts = undefined,
            .heldout_counts = undefined,
            .query_heads_by_layer = undefined,
            .scales_softcaps = undefined,
            .layout = .{ .query_heads_by_layer = &query_heads, .scales_by_donor = &scales, .softcaps_by_donor = &softcaps },
            .fit = .{ .prefix_rows = 0, .recent_rows = 1, .chunk_rows = 256, .retention_divisor = 4, .min_retention_divisor = 2, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_sweeps = 8, .nnls_tolerance = 1e-8, .min_weight = 1e-12, .selection = .residual_omp, .mass_solver = .active_set, .validation = .whole_head, .rejected_head_policy = .preserve_original, .storage_precision = .f16, .independent_fp64_publication = true },
            .method = .top1,
            .query_samples = 3,
            .source_has_metadata = true,
            .original_global_device_bytes = 0,
        };
    }
};

fn preparedFixture(allocator: std.mem.Allocator) !void {
    var input = PreparedFixture.input();
    var prepared = try prepareInput(allocator, &input, &PreparedFixture.augmented, undefined, 0, 0, 256, 4096, .{});
    defer prepared.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, prepared.source.keys);
    try std.testing.expectEqualSlices(f64, &.{ -1, -2, -3, -4, -5, -6, -7, -8 }, prepared.source.values);
    try std.testing.expectEqualSlices(f64, &.{ 0.5, -0.5, 1.5, -1.5 }, prepared.source.bias.?);
    try std.testing.expectEqualSlices(f64, &.{ 10, 11, 12, 13, 30, 31 }, prepared.train);
    try std.testing.expectEqualSlices(f64, &.{ 20, 21 }, prepared.heldout);
}

test "bounded prepared input widens f32 and keeps augmentation TRAIN-only" {
    try preparedFixture(std.testing.allocator);
}

test "bounded prepared input cap rejects before first allocation" {
    var input = PreparedFixture.input();
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.CompactionWorkspaceLimit, prepareInput(failing.allocator(), &input, &PreparedFixture.augmented, undefined, 0, 0, 256, 239, .{}));
    try std.testing.expectEqual(@as(usize, 0), failing.alloc_index);
}

test "bounded prepared input cancellation and allocation failures unwind" {
    const Cancel = struct {
        var checks: usize = 0;
        fn check(_: ?*anyopaque) !void {
            checks += 1;
            if (checks == 2) return error.Cancelled;
        }
    };
    var input = PreparedFixture.input();
    Cancel.checks = 0;
    try std.testing.expectError(error.Cancelled, prepareInput(std.testing.allocator, &input, &PreparedFixture.augmented, undefined, 0, 0, 256, 4096, .{ .check_fn = Cancel.check }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, preparedFixture, .{});
}
