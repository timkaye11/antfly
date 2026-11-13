//! TRAIN-only critical source-key extraction from an authenticated probability matrix.
//! Held-out values are intentionally absent from this API.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy_name = "critical_probability_half_gate_0_025_v1";
pub const policy_threshold: f64 = 0.025;

pub const Range = struct { start: usize, rows: usize };
pub const RangeKeys = struct { start: usize, rows: usize, key_begin: usize, key_count: usize };
pub const ProbabilityView = struct {
    values: []const f64,
    query_rows: usize,
    complete_query_rows: usize,
    source_rows: usize,
    source_key_space: enum { global_original },
    normalization: enum { full_original_softmax_post_softcap_bias },
    storage: enum { exact_f32_widened },
};

pub const Config = struct {
    threshold: f64 = policy_threshold,
    max_workspace_bytes: usize,
    control: solver.Control = .{},
};

/// `query_offsets[i]..query_offsets[i+1]` indexes the sorted TRAIN-query IDs
/// for `source_keys[i]`. `ranges` partitions `source_keys` in caller order.
pub const Result = struct {
    allocator: std.mem.Allocator,
    ranges: []RangeKeys,
    source_keys: []usize,
    query_offsets: []usize,
    critical_queries: []usize,
    query_rows: usize,
    complete_query_rows: usize,
    source_rows: usize,
    threshold: f64,
    peak_workspace_bytes: usize,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.ranges);
        self.allocator.free(self.source_keys);
        self.allocator.free(self.query_offsets);
        self.allocator.free(self.critical_queries);
        self.* = undefined;
    }

    pub fn keysForRange(self: @This(), index: usize) []const usize {
        const range = self.ranges[index];
        return self.source_keys[range.key_begin..][0..range.key_count];
    }

    pub fn queriesForKey(self: @This(), index: usize) []const usize {
        return self.critical_queries[self.query_offsets[index]..self.query_offsets[index + 1]];
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn bytes(count: usize, comptime T: type) !usize {
    return mul(count, @sizeOf(T));
}

/// Select every source key whose probability is >= threshold for any TRAIN
/// query. No argmax/top-1 key is forced. Ranges must be sorted and disjoint.
pub fn extract(
    allocator: std.mem.Allocator,
    view: ProbabilityView,
    ranges_in: []const Range,
    cfg: Config,
) !Result {
    const probabilities = view.values;
    const query_rows = view.query_rows;
    const source_rows = view.source_rows;
    if (query_rows == 0 or source_rows == 0 or ranges_in.len == 0 or
        view.complete_query_rows != query_rows or
        probabilities.len != try mul(query_rows, source_rows) or
        !std.math.isFinite(cfg.threshold) or cfg.threshold <= 0 or cfg.threshold > 1)
        return error.InvalidAttentionShape;

    var prior_end: usize = 0;
    for (ranges_in, 0..) |range, index| {
        const end = try add(range.start, range.rows);
        if (range.rows == 0 or end > source_rows or (index != 0 and range.start < prior_end)) return error.InvalidAttentionShape;
        prior_end = end;
    }

    const row_scratch = try mul(try bytes(source_rows, usize), 2);
    if (row_scratch > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();

    const key_slot = try allocator.alloc(usize, source_rows);
    defer allocator.free(key_slot);
    @memset(key_slot, std.math.maxInt(usize));
    const counts = try allocator.alloc(usize, source_rows);
    defer allocator.free(counts);
    @memset(counts, 0);
    for (ranges_in, 0..) |range, range_index| {
        for (range.start..range.start + range.rows) |source_key| key_slot[source_key] = range_index;
    }

    var pair_count: usize = 0;
    for (0..query_rows) |query| {
        try cfg.control.check();
        const row = probabilities[query * source_rows ..][0..source_rows];
        for (row, 0..) |probability, source_key| {
            if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
            if (key_slot[source_key] != std.math.maxInt(usize) and probability >= cfg.threshold) {
                counts[source_key] = try add(counts[source_key], 1);
                pair_count = try add(pair_count, 1);
            }
        }
    }

    var selected_count: usize = 0;
    for (counts) |count| if (count != 0) {
        selected_count = try add(selected_count, 1);
    };
    const offsets_count = try add(selected_count, 1);
    var peak = row_scratch;
    peak = try add(peak, try bytes(ranges_in.len, RangeKeys));
    peak = try add(peak, try bytes(selected_count, usize)); // source_keys
    peak = try add(peak, try bytes(offsets_count, usize)); // query_offsets
    peak = try add(peak, try bytes(pair_count, usize)); // critical_queries
    peak = try add(peak, try bytes(selected_count, usize)); // fill cursors
    if (peak > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();

    const ranges = try allocator.alloc(RangeKeys, ranges_in.len);
    errdefer allocator.free(ranges);
    const source_keys = try allocator.alloc(usize, selected_count);
    errdefer allocator.free(source_keys);
    const query_offsets = try allocator.alloc(usize, offsets_count);
    errdefer allocator.free(query_offsets);
    const critical_queries = try allocator.alloc(usize, pair_count);
    errdefer allocator.free(critical_queries);
    const cursors = try allocator.alloc(usize, selected_count);
    defer allocator.free(cursors);

    var key_at: usize = 0;
    var pair_at: usize = 0;
    for (ranges_in, ranges) |range, *out_range| {
        const begin = key_at;
        for (range.start..range.start + range.rows) |source_key| {
            if (counts[source_key] == 0) {
                key_slot[source_key] = std.math.maxInt(usize);
                continue;
            }
            source_keys[key_at] = source_key;
            query_offsets[key_at] = pair_at;
            cursors[key_at] = pair_at;
            key_slot[source_key] = key_at;
            pair_at = try add(pair_at, counts[source_key]);
            key_at += 1;
        }
        out_range.* = .{ .start = range.start, .rows = range.rows, .key_begin = begin, .key_count = key_at - begin };
    }
    query_offsets[selected_count] = pair_at;
    if (key_at != selected_count or pair_at != pair_count) return error.InvalidAttentionShape;

    for (0..query_rows) |query| {
        try cfg.control.check();
        const row = probabilities[query * source_rows ..][0..source_rows];
        for (row, 0..) |probability, source_key| {
            const slot = key_slot[source_key];
            if (slot != std.math.maxInt(usize) and probability >= cfg.threshold) {
                critical_queries[cursors[slot]] = query;
                cursors[slot] += 1;
            }
        }
    }
    for (cursors, 0..) |cursor, index| if (cursor != query_offsets[index + 1]) return error.InvalidAttentionShape;
    try cfg.control.check();

    return .{
        .allocator = allocator,
        .ranges = ranges,
        .source_keys = source_keys,
        .query_offsets = query_offsets,
        .critical_queries = critical_queries,
        .query_rows = query_rows,
        .complete_query_rows = view.complete_query_rows,
        .source_rows = source_rows,
        .threshold = cfg.threshold,
        .peak_workspace_bytes = peak,
    };
}

test "TRAIN critical keys use inclusive threshold without forced top1" {
    const p = [_]f64{
        0.024, 0.025, 0.026, 0.0, 0.024, 0.024,
        0.025, 0.024, 0.001, 0.4, 0.024, 0.024,
    };
    var result = try extract(std.testing.allocator, .{ .values = &p, .query_rows = 2, .complete_query_rows = 2, .source_rows = 6, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{ .{ .start = 0, .rows = 3 }, .{ .start = 4, .rows = 2 } }, .{ .max_workspace_bytes = 4096 });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, result.source_keys);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, result.keysForRange(0));
    try std.testing.expectEqual(@as(usize, 0), result.keysForRange(1).len);
    try std.testing.expectEqualSlices(usize, &.{1}, result.queriesForKey(0));
    try std.testing.expectEqualSlices(usize, &.{0}, result.queriesForKey(1));
    try std.testing.expectEqualSlices(usize, &.{0}, result.queriesForKey(2));
}

test "TRAIN critical query mapping spans multiple consumer row groups" {
    // Rows 0..2 and 2..4 model two caller-authenticated consumer groups.
    const p = [_]f64{
        0.03, 0.00, 0.00, 0.00,
        0.00, 0.04, 0.00, 0.00,
        0.03, 0.00, 0.05, 0.00,
        0.00, 0.04, 0.05, 0.00,
    };
    var result = try extract(std.testing.allocator, .{ .values = &p, .query_rows = 4, .complete_query_rows = 4, .source_rows = 4, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{ .{ .start = 0, .rows = 2 }, .{ .start = 2, .rows = 2 } }, .{ .max_workspace_bytes = 4096 });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, result.source_keys);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, result.queriesForKey(0));
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, result.queriesForKey(1));
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, result.queriesForKey(2));
}

test "TRAIN critical keys reject shape finite range and workspace errors" {
    const p = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    const view: ProbabilityView = .{ .values = &p, .query_rows = 2, .complete_query_rows = 2, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened };
    try std.testing.expectError(error.InvalidAttentionShape, extract(std.testing.allocator, view, &.{.{ .start = 1, .rows = 2 }}, .{ .max_workspace_bytes = 4096 }));
    var incomplete = view;
    incomplete.complete_query_rows = 1;
    try std.testing.expectError(error.InvalidAttentionShape, extract(std.testing.allocator, incomplete, &.{.{ .start = 0, .rows = 2 }}, .{ .max_workspace_bytes = 4096 }));
    var bad = p;
    bad[2] = std.math.nan(f64);
    var bad_view = view;
    bad_view.values = &bad;
    try std.testing.expectError(error.NonFiniteAttention, extract(std.testing.allocator, bad_view, &.{.{ .start = 0, .rows = 2 }}, .{ .max_workspace_bytes = 4096 }));
    try std.testing.expectError(error.CompactionWorkspaceLimit, extract(std.testing.allocator, view, &.{.{ .start = 0, .rows = 2 }}, .{ .max_workspace_bytes = 1 }));
}

test "TRAIN critical keys cancel after allocation without output" {
    const Cancel = struct {
        var checks: usize = 0;
        fn check(_: ?*anyopaque) !void {
            checks += 1;
            if (checks == 3) return error.Cancelled;
        }
    };
    Cancel.checks = 0;
    const p = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    try std.testing.expectError(error.Cancelled, extract(std.testing.allocator, .{ .values = &p, .query_rows = 2, .complete_query_rows = 2, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = 0, .rows = 2 }}, .{ .max_workspace_bytes = 4096, .control = .{ .check_fn = Cancel.check } }));
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    const p = [_]f64{ 0.5, 0.5, 0.5, 0.5 };
    var result = try extract(allocator, .{ .values = &p, .query_rows = 2, .complete_query_rows = 2, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = 0, .rows = 2 }}, .{ .max_workspace_bytes = 4096 });
    result.deinit();
}

test "TRAIN critical keys unwind every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
