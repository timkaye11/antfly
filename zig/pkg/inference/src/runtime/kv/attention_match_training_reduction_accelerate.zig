// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Default-off FP64 reductions over an authenticated cached TRAIN probability
//! source. This does not recompute QK/softmax and never replaces the scalar
//! harm or final-heldout oracle. DGEMM changes reduction order, so callers must
//! bind the policy separately and retain the unchanged 0.05 gates.
const builtin = @import("builtin");
const std = @import("std");
const solver = @import("attention_match.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");

extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

pub const policy = "cached_train_fp64_reduction_accelerate_v1";

pub const ProbabilitySource = struct {
    context: ?*anyopaque,
    query_count: usize,
    row_count: usize,
    copy_fn: *const fn (?*anyopaque, usize, usize, []f64) anyerror!void,
};

pub const Backend = struct {
    context: ?*anyopaque = null,
    gemm_fn: *const fn (?*anyopaque, usize, usize, usize, []const f64, []const f64, []f64) anyerror!void,
};

pub const Options = struct {
    backend: Backend,
    query_batch_rows: usize = 64,
    external_live_bytes: usize,
    max_workspace_bytes: usize,
    control: solver.Control = .{},
};

pub const Evidence = struct {
    accelerated: bool = false,
    tiles: usize = 0,
    backend_calls: usize = 0,
    copied_probability_bytes_peak: usize = 0,
    staged_output_bytes: usize = 0,
    ones_bytes: usize = 0,
    workspace_bytes: usize = 0,
};

pub const OwnedRange = struct {
    allocator: std.mem.Allocator,
    rho: []f64,
    y: []f64,
    evidence: Evidence,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.y);
        self.allocator.free(self.rho);
        self.* = undefined;
    }
};

pub const Harm = struct { max_relative_mass: f64, relative_output_rmse: f64, evidence: Evidence };

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn workspaceBytes(query_count: usize, columns: usize, dim: usize, batch_rows: usize) !usize {
    if (query_count == 0 or columns == 0 or dim == 0 or batch_rows == 0) return error.InvalidAttentionShape;
    const batch = @min(query_count, batch_rows);
    const staged_rho = try mul(query_count, @sizeOf(f64));
    const staged_y = try mul(try mul(query_count, dim), @sizeOf(f64));
    const page = try mul(try mul(batch, columns), @sizeOf(f64));
    const tile_y = try mul(try mul(batch, dim), @sizeOf(f64));
    const ones = try mul(columns, @sizeOf(f64));
    return try add(try add(staged_rho, staged_y), try add(try add(page, tile_y), ones));
}

fn finite(values: []const f64) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
}

/// Reduces one contiguous cached source range. `values` is columns x dim.
/// Destination slices remain unchanged on cap, allocation, backend, finite, or
/// cancellation failure.
pub fn reduceRange(
    allocator: std.mem.Allocator,
    source: ProbabilitySource,
    range_start: usize,
    columns: usize,
    values: []const f64,
    dim: usize,
    options: Options,
    rho: []f64,
    y: []f64,
    evidence: *Evidence,
) !void {
    evidence.* = .{};
    if (source.query_count == 0 or source.row_count == 0 or columns == 0 or dim == 0 or
        range_start > source.row_count or columns > source.row_count - range_start or
        values.len != try mul(columns, dim) or rho.len != source.query_count or
        y.len != try mul(source.query_count, dim) or options.query_batch_rows == 0)
        return error.InvalidAttentionShape;
    try finite(values);
    try options.control.check();
    const owned = try workspaceBytes(source.query_count, columns, dim, options.query_batch_rows);
    if (try add(options.external_live_bytes, owned) > options.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const batch = @min(source.query_count, options.query_batch_rows);
    const staged_rho = try allocator.alloc(f64, source.query_count);
    defer allocator.free(staged_rho);
    const staged_y = try allocator.alloc(f64, try mul(source.query_count, dim));
    defer allocator.free(staged_y);
    const page = try allocator.alloc(f64, try mul(batch, columns));
    defer allocator.free(page);
    const tile_y = try allocator.alloc(f64, try mul(batch, dim));
    defer allocator.free(tile_y);
    const ones = try allocator.alloc(f64, columns);
    defer allocator.free(ones);
    @memset(ones, 1);
    evidence.staged_output_bytes = try mul(try add(staged_rho.len, staged_y.len), @sizeOf(f64));
    evidence.copied_probability_bytes_peak = try mul(page.len, @sizeOf(f64));
    evidence.ones_bytes = try mul(ones.len, @sizeOf(f64));
    evidence.workspace_bytes = owned;

    var first: usize = 0;
    while (first < source.query_count) {
        try options.control.check();
        const count = @min(batch, source.query_count - first);
        const tile_page = page[0 .. count * columns];
        for (0..count) |local| {
            try options.control.check();
            try source.copy_fn(source.context, first + local, range_start, tile_page[local * columns ..][0..columns]);
        }
        try finite(tile_page);
        const tile_rho = staged_rho[first..][0..count];
        const tile_output = tile_y[0 .. count * dim];
        try options.backend.gemm_fn(options.backend.context, count, 1, columns, tile_page, ones, tile_rho);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try options.control.check();
        try options.backend.gemm_fn(options.backend.context, count, dim, columns, tile_page, values, tile_output);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try options.control.check();
        try finite(tile_rho);
        try finite(tile_output);
        @memcpy(staged_y[first * dim ..][0 .. count * dim], tile_output);
        evidence.tiles = try add(evidence.tiles, 1);
        first += count;
    }
    try options.control.check();
    @memcpy(rho, staged_rho);
    @memcpy(y, staged_y);
    evidence.accelerated = true;
}

/// Ownership-returning form used when the caller has no pre-existing output
/// buffers. This avoids retaining a second q+q*d destination beside the
/// reducer's atomic staging while preserving identical reduction order.
pub fn reduceRangeOwned(allocator: std.mem.Allocator, source: ProbabilitySource, range_start: usize, columns: usize, values: []const f64, dim: usize, options: Options) !OwnedRange {
    if (source.query_count == 0 or source.row_count == 0 or columns == 0 or dim == 0 or options.query_batch_rows == 0 or range_start > source.row_count or columns > source.row_count - range_start or values.len != try mul(columns, dim)) return error.InvalidAttentionShape;
    try finite(values);
    try options.control.check();
    const destination_count = try add(source.query_count, try mul(source.query_count, dim));
    const destination_bytes = try mul(destination_count, @sizeOf(f64));
    const full = try workspaceBytes(source.query_count, columns, dim, options.query_batch_rows);
    if (full < destination_bytes or try add(options.external_live_bytes, full) > options.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const rho = try allocator.alloc(f64, source.query_count);
    errdefer allocator.free(rho);
    const y = try allocator.alloc(f64, try mul(source.query_count, dim));
    errdefer allocator.free(y);
    const batch = @min(source.query_count, options.query_batch_rows);
    const page = try allocator.alloc(f64, try mul(batch, columns));
    defer allocator.free(page);
    const tile_y = try allocator.alloc(f64, try mul(batch, dim));
    defer allocator.free(tile_y);
    const ones = try allocator.alloc(f64, columns);
    defer allocator.free(ones);
    @memset(ones, 1);
    var evidence: Evidence = .{ .staged_output_bytes = destination_bytes, .copied_probability_bytes_peak = try mul(page.len, @sizeOf(f64)), .ones_bytes = try mul(ones.len, @sizeOf(f64)), .workspace_bytes = full };
    var first: usize = 0;
    while (first < source.query_count) {
        try options.control.check();
        const count = @min(batch, source.query_count - first);
        const tile_page = page[0 .. count * columns];
        for (0..count) |local| {
            try options.control.check();
            try source.copy_fn(source.context, first + local, range_start, tile_page[local * columns ..][0..columns]);
        }
        try finite(tile_page);
        try options.control.check();
        try options.backend.gemm_fn(options.backend.context, count, 1, columns, tile_page, ones, rho[first..][0..count]);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try options.control.check();
        try options.backend.gemm_fn(options.backend.context, count, dim, columns, tile_page, values, tile_y[0 .. count * dim]);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try finite(rho[first..][0..count]);
        try finite(tile_y[0 .. count * dim]);
        @memcpy(y[first * dim ..][0 .. count * dim], tile_y[0 .. count * dim]);
        evidence.tiles += 1;
        first += count;
    }
    try options.control.check();
    evidence.accelerated = true;
    return .{ .allocator = allocator, .rho = rho, .y = y, .evidence = evidence };
}

pub fn harmWorkspaceBytes(query_count: usize, columns: usize, dim: usize, batch_rows: usize) !usize {
    const batch = @min(query_count, batch_rows);
    return add(try mul(try add(try mul(batch, columns), try add(batch, try mul(batch, dim))), @sizeOf(f64)), try mul(columns, @sizeOf(f64)));
}

/// Builds the selected probability matrix in bounded query tiles, applying
/// each authenticated nonnegative ratio, then delegates to the same atomic
/// range reducer. `fitted_values` is selected_local.len x dim.
pub fn reduceSelected(
    allocator: std.mem.Allocator,
    source: ProbabilitySource,
    range_start: usize,
    range_rows: usize,
    selected_local: []const usize,
    ratios: []const f64,
    fitted_values: []const f64,
    dim: usize,
    options: Options,
    rho: []f64,
    y: []f64,
    evidence: *Evidence,
) !void {
    evidence.* = .{};
    if (selected_local.len == 0 or ratios.len != selected_local.len or fitted_values.len != try mul(selected_local.len, dim) or
        range_start > source.row_count or range_rows > source.row_count - range_start)
        return error.InvalidAttentionShape;
    for (selected_local, ratios) |local, ratio| if (local >= range_rows or !std.math.isFinite(ratio) or ratio < 0) return error.InvalidAttentionShape;
    const Adapter = struct {
        source: ProbabilitySource,
        range_start: usize,
        selected_local: []const usize,
        ratios: []const f64,
        fn copy(raw: ?*anyopaque, query: usize, _: usize, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (out.len != self.selected_local.len) return error.InvalidAttentionShape;
            var one: [1]f64 = undefined;
            for (self.selected_local, self.ratios, out) |local, ratio, *value| {
                try self.source.copy_fn(self.source.context, query, self.range_start + local, &one);
                value.* = one[0] * ratio;
                if (!std.math.isFinite(value.*) or value.* < 0) return error.NonFiniteAttention;
            }
        }
    };
    var adapter: Adapter = .{ .source = source, .range_start = range_start, .selected_local = selected_local, .ratios = ratios };
    return reduceRange(allocator, .{ .context = &adapter, .query_count = source.query_count, .row_count = selected_local.len, .copy_fn = Adapter.copy }, 0, selected_local.len, fitted_values, dim, options, rho, y, evidence);
}

/// Computes the exact rounded-harm scalars in chronological query order while
/// retaining only one query tile. No candidate q*d or atomic staging output is
/// materialized.
pub fn measureSelectedHarm(allocator: std.mem.Allocator, source: ProbabilitySource, range_start: usize, range_rows: usize, selected_local: []const usize, ratios: []const f64, fitted_values: []const f64, original_y: []const f64, range_rho: []const f64, range_y: []const f64, dim: usize, floor: f64, options: Options) !Harm {
    if (source.query_count == 0 or source.row_count == 0 or options.query_batch_rows == 0 or range_rows == 0 or range_start > source.row_count or range_rows > source.row_count - range_start or selected_local.len == 0 or ratios.len != selected_local.len or fitted_values.len != try mul(selected_local.len, dim) or original_y.len != try mul(source.query_count, dim) or range_rho.len != source.query_count or range_y.len != original_y.len or !std.math.isFinite(floor) or floor <= 0) return error.InvalidAttentionShape;
    try finite(fitted_values);
    try options.control.check();
    for (selected_local, ratios) |local, ratio| if (local >= range_rows or !std.math.isFinite(ratio) or ratio < 0) return error.InvalidAttentionShape;
    const batch = @min(source.query_count, options.query_batch_rows);
    const columns = selected_local.len;
    const owned = try harmWorkspaceBytes(source.query_count, columns, dim, options.query_batch_rows);
    if (try add(options.external_live_bytes, owned) > options.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const page = try allocator.alloc(f64, try mul(batch, columns));
    defer allocator.free(page);
    const tile_rho = try allocator.alloc(f64, batch);
    defer allocator.free(tile_rho);
    const tile_y = try allocator.alloc(f64, try mul(batch, dim));
    defer allocator.free(tile_y);
    const ones = try allocator.alloc(f64, columns);
    defer allocator.free(ones);
    @memset(ones, 1);
    var evidence: Evidence = .{ .workspace_bytes = owned, .copied_probability_bytes_peak = try mul(page.len, @sizeOf(f64)), .ones_bytes = try mul(ones.len, @sizeOf(f64)) };
    var max_mass: f64 = 0;
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    var first: usize = 0;
    while (first < source.query_count) {
        try options.control.check();
        const count = @min(batch, source.query_count - first);
        const tile_page = page[0 .. count * columns];
        for (0..count) |local_q| {
            try options.control.check();
            for (selected_local, ratios, 0..) |selected, ratio, column| {
                var one: [1]f64 = undefined;
                try source.copy_fn(source.context, first + local_q, range_start + selected, &one);
                tile_page[local_q * columns + column] = one[0] * ratio;
            }
        }
        try finite(tile_page);
        try options.control.check();
        try options.backend.gemm_fn(options.backend.context, count, 1, columns, tile_page, ones, tile_rho[0..count]);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try options.control.check();
        try options.backend.gemm_fn(options.backend.context, count, dim, columns, tile_page, fitted_values, tile_y[0 .. count * dim]);
        evidence.backend_calls = try add(evidence.backend_calls, 1);
        try finite(tile_rho[0..count]);
        try finite(tile_y[0 .. count * dim]);
        for (0..count) |local_q| {
            const qi = first + local_q;
            const delta = tile_rho[local_q] - range_rho[qi];
            const denominator = 1 + delta;
            if (!std.math.isFinite(denominator) or denominator <= 0) return error.NonFiniteAttention;
            max_mass = @max(max_mass, @abs(delta));
            for (0..dim) |di| {
                const at = qi * dim + di;
                const predicted = (original_y[at] - range_y[at] + tile_y[local_q * dim + di]) / denominator;
                const err = predicted - original_y[at];
                squared_error += err * err;
                squared_reference += original_y[at] * original_y[at];
            }
        }
        evidence.tiles += 1;
        first += count;
    }
    try options.control.check();
    evidence.accelerated = true;
    return .{ .max_relative_mass = max_mass, .relative_output_rmse = @sqrt(squared_error / @max(squared_reference, floor * floor)), .evidence = evidence };
}

fn accelerateGemm(_: ?*anyopaque, m: usize, n: usize, k: usize, a: []const f64, b: []const f64, c: []f64) !void {
    if (comptime builtin.os.tag != .macos) return error.UnsupportedOperation;
    if (m == 0 or n == 0 or k == 0 or a.len != try mul(m, k) or b.len != try mul(k, n) or c.len != try mul(m, n))
        return error.InvalidAttentionShape;
    const mm = std.math.cast(c_int, m) orelse return error.InvalidAttentionShape;
    const nn = std.math.cast(c_int, n) orelse return error.InvalidAttentionShape;
    const kk = std.math.cast(c_int, k) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(101, 111, 111, mm, nn, kk, 1, a.ptr, kk, b.ptr, nn, 0, c.ptr, nn);
}

pub fn accelerateBackend() ?Backend {
    return if (builtin.os.tag == .macos) .{ .gemm_fn = accelerateGemm } else null;
}

fn testGemm(_: ?*anyopaque, m: usize, n: usize, k: usize, a: []const f64, b: []const f64, c: []f64) !void {
    for (0..m) |i| for (0..n) |j| {
        var sum: f64 = 0;
        for (0..k) |x| sum += a[i * k + x] * b[x * n + j];
        c[i * n + j] = sum;
    };
}

const TestSource = struct {
    probabilities: []const f64,
    rows: usize,
    fn copy(raw: ?*anyopaque, query: usize, start: usize, out: []f64) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        if (start > self.rows or out.len > self.rows - start) return error.InvalidAttentionShape;
        @memcpy(out, self.probabilities[query * self.rows + start ..][0..out.len]);
    }
};

test "context cached TRAIN FP64 reductions match scalar range and selected oracles across a tail tile" {
    var source_data: TestSource = .{ .probabilities = &.{ 0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25, 0.4, 0.3, 0.2, 0.1 }, .rows = 4 };
    const source: ProbabilitySource = .{ .context = &source_data, .query_count = 3, .row_count = 4, .copy_fn = TestSource.copy };
    const values = [_]f64{ 1, 2, -1, 3, 4, -2, 0.5, 0.25 };
    var rho = [_]f64{ 99, 99, 99 };
    var y = [_]f64{ 99, 99, 99, 99, 99, 99 };
    var evidence: Evidence = .{};
    const needed = try workspaceBytes(3, 4, 2, 2);
    try reduceRange(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 17, .max_workspace_bytes = 17 + needed }, &rho, &y, &evidence);
    try std.testing.expectEqual(@as(usize, 2), evidence.tiles);
    try std.testing.expectEqual(@as(usize, 4), evidence.backend_calls);
    for (0..3) |q| {
        var want_rho: f64 = 0;
        var want_y = [_]f64{ 0, 0 };
        for (0..4) |r| {
            const p = source_data.probabilities[q * 4 + r];
            want_rho += p;
            for (0..2) |d| want_y[d] += p * values[r * 2 + d];
        }
        try std.testing.expectApproxEqAbs(want_rho, rho[q], 1e-15);
        for (0..2) |d| try std.testing.expectApproxEqAbs(want_y[d], y[q * 2 + d], 1e-15);
    }
    const selected = [_]usize{ 3, 1 };
    const ratios = [_]f64{ 1.5, 0.75 };
    const fitted = [_]f64{ 0.25, 2.5, -2, 1 };
    try reduceSelected(std.testing.allocator, source, 0, 4, &selected, &ratios, &fitted, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 0, .max_workspace_bytes = try workspaceBytes(3, 2, 2, 2) }, &rho, &y, &evidence);
    for (0..3) |q| {
        var want_rho: f64 = 0;
        var want_y = [_]f64{ 0, 0 };
        for (selected, ratios, 0..) |local, ratio, s| {
            const p = source_data.probabilities[q * 4 + local] * ratio;
            want_rho += p;
            for (0..2) |d| want_y[d] += p * fitted[s * 2 + d];
        }
        try std.testing.expectApproxEqAbs(want_rho, rho[q], 1e-15);
        for (0..2) |d| try std.testing.expectApproxEqAbs(want_y[d], y[q * 2 + d], 1e-15);
    }
}

test "owned original and streamed selected harm preserve the full-vector oracle" {
    var data: TestSource = .{ .probabilities = &.{ 0.1, 0.2, 0.3, 0.4, 0.25, 0.25, 0.25, 0.25, 0.4, 0.3, 0.2, 0.1 }, .rows = 4 };
    const source: ProbabilitySource = .{ .context = &data, .query_count = 3, .row_count = 4, .copy_fn = TestSource.copy };
    const values = [_]f64{ 1, 2, -1, 3, 4, -2, 0.5, 0.25 };
    const options: Options = .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 0, .max_workspace_bytes = 1 << 20 };
    var old_rho: [3]f64 = undefined;
    var old_y: [6]f64 = undefined;
    var old_evidence: Evidence = .{};
    try reduceRange(std.testing.allocator, source, 0, 4, &values, 2, options, &old_rho, &old_y, &old_evidence);
    var owned = try reduceRangeOwned(std.testing.allocator, source, 0, 4, &values, 2, options);
    defer owned.deinit();
    try std.testing.expectEqualSlices(f64, &old_rho, owned.rho);
    try std.testing.expectEqualSlices(f64, &old_y, owned.y);
    const selected = [_]usize{ 3, 1 };
    const ratios = [_]f64{ 1.5, 0.75 };
    const fitted = [_]f64{ 0.25, 2.5, -2, 1 };
    var candidate_rho: [3]f64 = undefined;
    var candidate_y: [6]f64 = undefined;
    try reduceSelected(std.testing.allocator, source, 0, 4, &selected, &ratios, &fitted, 2, options, &candidate_rho, &candidate_y, &old_evidence);
    const expected = try adaptive.measureRoundedCandidateHarm(&old_y, &old_rho, &old_y, &candidate_rho, &candidate_y, 3, 2, 1e-12);
    const streamed = try measureSelectedHarm(std.testing.allocator, source, 0, 4, &selected, &ratios, &fitted, &old_y, &old_rho, &old_y, 2, 1e-12, options);
    try std.testing.expectEqual(expected.max_relative_mass, streamed.max_relative_mass);
    try std.testing.expectEqual(expected.relative_output_rmse, streamed.relative_output_rmse);
    try std.testing.expect(try harmWorkspaceBytes(3, 2, 2, 2) < try workspaceBytes(3, 2, 2, 2));
    const exact = try workspaceBytes(3, 4, 2, 2);
    var exact_owned = try reduceRangeOwned(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 17, .max_workspace_bytes = 17 + exact });
    exact_owned.deinit();
    try std.testing.expectError(error.CompactionWorkspaceLimit, reduceRangeOwned(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 17, .max_workspace_bytes = 16 + exact }));
    try std.testing.expectError(error.CompactionWorkspaceLimit, reduceRangeOwned(std.testing.failing_allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 17, .max_workspace_bytes = 16 + exact }));
    try std.testing.expectError(error.InvalidAttentionShape, measureSelectedHarm(std.testing.allocator, source, 0, 4, &selected, &ratios, &fitted, &old_y, &old_rho, &old_y, 2, 1e-12, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 0, .external_live_bytes = 0, .max_workspace_bytes = 1 << 20 }));
    const cancelled = solver.Control{ .check_fn = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    }.check };
    try std.testing.expectError(error.Cancelled, reduceRangeOwned(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 2, .external_live_bytes = 0, .max_workspace_bytes = 1 << 20, .control = cancelled }));
    try std.testing.expectError(error.OutOfMemory, reduceRangeOwned(std.testing.failing_allocator, source, 0, 4, &values, 2, options));
}

test "context cached TRAIN FP64 full-query batching is equivalent and collapses dispatches" {
    var source_data: TestSource = .{ .probabilities = &.{
        0.1,  0.2,  0.3,  0.4,
        0.25, 0.25, 0.25, 0.25,
        0.4,  0.3,  0.2,  0.1,
    }, .rows = 4 };
    const source: ProbabilitySource = .{ .context = &source_data, .query_count = 3, .row_count = 4, .copy_fn = TestSource.copy };
    const values = [_]f64{ 1, 2, -1, 3, 4, -2, 0.5, 0.25 };
    var tiled_rho: [3]f64 = undefined;
    var tiled_y: [6]f64 = undefined;
    var full_rho: [3]f64 = undefined;
    var full_y: [6]f64 = undefined;
    var tiled: Evidence = .{};
    var full: Evidence = .{};
    try reduceRange(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 1, .external_live_bytes = 0, .max_workspace_bytes = try workspaceBytes(3, 4, 2, 1) }, &tiled_rho, &tiled_y, &tiled);
    try reduceRange(std.testing.allocator, source, 0, 4, &values, 2, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 3, .external_live_bytes = 0, .max_workspace_bytes = try workspaceBytes(3, 4, 2, 3) }, &full_rho, &full_y, &full);
    try std.testing.expectEqualSlices(f64, &tiled_rho, &full_rho);
    try std.testing.expectEqualSlices(f64, &tiled_y, &full_y);
    try std.testing.expectEqual(@as(usize, 6), tiled.backend_calls);
    try std.testing.expectEqual(@as(usize, 2), full.backend_calls);
    try std.testing.expectEqual(@as(usize, 1), full.tiles);
}

test "context cached TRAIN FP64 reductions cap and late cancellation preserve outputs" {
    var source_data: TestSource = .{ .probabilities = &.{ 0.5, 0.5, 0.25, 0.75 }, .rows = 2 };
    const source: ProbabilitySource = .{ .context = &source_data, .query_count = 2, .row_count = 2, .copy_fn = TestSource.copy };
    var rho = [_]f64{ 77, 77 };
    var y = [_]f64{ 88, 88 };
    var evidence: Evidence = .{};
    evidence.accelerated = true;
    try std.testing.expectError(error.InvalidAttentionShape, reduceSelected(std.testing.allocator, source, 0, 2, &.{}, &.{}, &.{}, 1, .{ .backend = .{ .gemm_fn = testGemm }, .external_live_bytes = 0, .max_workspace_bytes = 1024 }, &rho, &y, &evidence));
    try std.testing.expect(!evidence.accelerated);
    const needed = try workspaceBytes(2, 2, 1, 1);
    try std.testing.expectError(error.CompactionWorkspaceLimit, reduceRange(std.testing.allocator, source, 0, 2, &.{ 2, 4 }, 1, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 1, .external_live_bytes = 9, .max_workspace_bytes = 9 + needed - 1 }, &rho, &y, &evidence));
    try std.testing.expectEqualSlices(f64, &.{ 77, 77 }, &rho);
    try std.testing.expectEqualSlices(f64, &.{ 88, 88 }, &y);
    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 8) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, reduceRange(std.testing.allocator, source, 0, 2, &.{ 2, 4 }, 1, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 1, .external_live_bytes = 0, .max_workspace_bytes = needed, .control = .{ .context = &cancel, .check_fn = Cancel.check } }, &rho, &y, &evidence));
    try std.testing.expectEqualSlices(f64, &.{ 77, 77 }, &rho);
    try std.testing.expectEqualSlices(f64, &.{ 88, 88 }, &y);

    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, reduceRange(failing.allocator(), source, 0, 2, &.{ 2, 4 }, 1, .{ .backend = .{ .gemm_fn = testGemm }, .query_batch_rows = 1, .external_live_bytes = 0, .max_workspace_bytes = needed }, &rho, &y, &evidence));
    try std.testing.expectEqualSlices(f64, &.{ 77, 77 }, &rho);
    try std.testing.expectEqualSlices(f64, &.{ 88, 88 }, &y);
}
