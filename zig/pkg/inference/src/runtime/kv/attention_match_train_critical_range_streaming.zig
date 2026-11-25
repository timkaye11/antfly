const std = @import("std");
const solver = @import("attention_match.zig");

pub const ProbabilitySource = struct {
    context: ?*anyopaque,
    query_rows: usize,
    complete_query_rows: usize,
    global_start: usize,
    range_rows: usize,
    original_rows: usize,
    external_live_bytes: usize,
    source_key_space: enum { global_original_range },
    normalization: enum { full_original_softmax_post_softcap_bias },
    storage: enum { exact_f32_widened },
    copy_query_fn: *const fn (?*anyopaque, usize, []f64, solver.Control) anyerror!void,
};

pub const Config = struct { threshold: f64 = 0.025, combined_cap: usize, control: solver.Control = .{} };

pub const Result = struct {
    allocator: std.mem.Allocator,
    global_start: usize,
    range_rows: usize,
    original_rows: usize,
    query_rows: usize,
    local_keys: []usize,
    query_offsets: []usize,
    critical_queries: []usize,
    probability_sha256: [32]u8,
    peak_bytes: usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.critical_queries);
        self.allocator.free(self.query_offsets);
        self.allocator.free(self.local_keys);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn bytes(comptime T: type, n: usize) !usize {
    return std.math.mul(usize, @sizeOf(T), n) catch error.InvalidAttentionShape;
}

pub fn extract(a: std.mem.Allocator, source: ProbabilitySource, cfg: Config) !Result {
    if (source.query_rows == 0 or source.complete_query_rows != source.query_rows or source.range_rows == 0 or source.original_rows == 0 or
        source.global_start > source.original_rows or source.range_rows > source.original_rows - source.global_start or
        !std.math.isFinite(cfg.threshold) or cfg.threshold <= 0 or cfg.threshold > 1)
        return error.InvalidAttentionShape;
    var base = try add(source.external_live_bytes, try bytes(f64, source.range_rows));
    base = try add(base, try bytes(usize, source.range_rows));
    if (base > cfg.combined_cap) return error.CompactionWorkspaceLimit;
    const row = try a.alloc(f64, source.range_rows);
    defer a.free(row);
    const counts = try a.alloc(usize, source.range_rows);
    defer a.free(counts);
    @memset(counts, 0);
    var first_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var pairs: usize = 0;
    for (0..source.query_rows) |query| {
        try cfg.control.check();
        try source.copy_query_fn(source.context, query, row, cfg.control);
        first_hash.update(std.mem.sliceAsBytes(row));
        for (row, counts) |probability, *count| {
            if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
            if (probability >= cfg.threshold) {
                count.* = try add(count.*, 1);
                pairs = try add(pairs, 1);
            }
        }
    }
    var key_count: usize = 0;
    for (counts) |count| if (count != 0) {
        key_count = try add(key_count, 1);
    };
    var peak = base;
    peak = try add(peak, try bytes(usize, key_count));
    peak = try add(peak, try bytes(usize, try add(key_count, 1)));
    peak = try add(peak, try bytes(usize, pairs));
    peak = try add(peak, try bytes(usize, key_count));
    if (peak > cfg.combined_cap) return error.CompactionWorkspaceLimit;
    const keys = try a.alloc(usize, key_count);
    errdefer a.free(keys);
    const offsets = try a.alloc(usize, try add(key_count, 1));
    errdefer a.free(offsets);
    const queries = try a.alloc(usize, pairs);
    errdefer a.free(queries);
    const cursors = try a.alloc(usize, key_count);
    defer a.free(cursors);
    var key_index: usize = 0;
    var pair_index: usize = 0;
    for (counts, 0..) |count, local| if (count != 0) {
        keys[key_index] = local;
        offsets[key_index] = pair_index;
        cursors[key_index] = pair_index;
        pair_index = try add(pair_index, count);
        key_index += 1;
    };
    offsets[key_count] = pair_index;
    var second_hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..source.query_rows) |query| {
        try cfg.control.check();
        try source.copy_query_fn(source.context, query, row, cfg.control);
        second_hash.update(std.mem.sliceAsBytes(row));
        var sparse_index: usize = 0;
        for (row, counts) |probability, count| {
            if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
            if (count == 0) continue;
            if (probability >= cfg.threshold) {
                if (cursors[sparse_index] >= offsets[sparse_index + 1]) return error.TrainingSourceChanged;
                queries[cursors[sparse_index]] = query;
                cursors[sparse_index] += 1;
            }
            sparse_index += 1;
        }
    }
    const first = first_hash.finalResult();
    const second = second_hash.finalResult();
    if (!std.mem.eql(u8, &first, &second)) return error.TrainingSourceChanged;
    for (cursors, 0..) |cursor, i| if (cursor != offsets[i + 1]) return error.TrainingSourceChanged;
    return .{ .allocator = a, .global_start = source.global_start, .range_rows = source.range_rows, .original_rows = source.original_rows, .query_rows = source.query_rows, .local_keys = keys, .query_offsets = offsets, .critical_queries = queries, .probability_sha256 = first, .peak_bytes = peak };
}

test "range extraction preserves global authority while retaining only local keys" {
    const Fixture = struct {
        values: []const f64,
        rows: usize,
        fn copy(raw: ?*anyopaque, query: usize, out: []f64, control: solver.Control) !void {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            @memcpy(out, self.values[query * self.rows ..][0..self.rows]);
        }
    };
    var fixture: Fixture = .{ .values = &.{ 0.01, 0.025, 0.03, 0.0, 0.04, 0.01 }, .rows = 3 };
    var result = try extract(std.testing.allocator, .{ .context = &fixture, .query_rows = 2, .complete_query_rows = 2, .global_start = 256, .range_rows = 3, .original_rows = 7936, .external_live_bytes = 17, .source_key_space = .global_original_range, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened, .copy_query_fn = Fixture.copy }, .{ .combined_cap = 4096 });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, result.local_keys);
    try std.testing.expectEqual(@as(usize, 256), result.global_start);
    try std.testing.expectEqual(@as(usize, 7936), result.original_rows);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2, 3 }, result.query_offsets);
}

test "dense CSR extraction admits its exact peak and rejects one byte below it" {
    const Fixture = struct {
        fn copy(_: ?*anyopaque, _: usize, out: []f64, control: solver.Control) !void {
            try control.check();
            @memset(out, 0.025);
        }
    };
    const query_rows: usize = 3;
    const range_rows: usize = 4;
    const pairs = query_rows * range_rows;
    const exact_peak = range_rows * @sizeOf(f64) +
        range_rows * @sizeOf(usize) +
        range_rows * @sizeOf(usize) +
        (range_rows + 1) * @sizeOf(usize) +
        pairs * @sizeOf(usize) +
        range_rows * @sizeOf(usize);
    const source: ProbabilitySource = .{
        .context = null,
        .query_rows = query_rows,
        .complete_query_rows = query_rows,
        .global_start = 17,
        .range_rows = range_rows,
        .original_rows = 64,
        .external_live_bytes = 0,
        .source_key_space = .global_original_range,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
        .copy_query_fn = Fixture.copy,
    };
    try std.testing.expectError(error.CompactionWorkspaceLimit, extract(std.testing.allocator, source, .{ .combined_cap = exact_peak - 1 }));
    var result = try extract(std.testing.allocator, source, .{ .combined_cap = exact_peak });
    defer result.deinit();
    try std.testing.expectEqual(exact_peak, result.peak_bytes);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, result.local_keys);
    try std.testing.expectEqualSlices(usize, &.{ 0, 3, 6, 9, 12 }, result.query_offsets);
}

test "second pass pair growth is rejected before writing past the counted segment" {
    const Changing = struct {
        calls: usize = 0,
        fn copy(raw: ?*anyopaque, _: usize, out: []f64, control: solver.Control) !void {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            @memset(out, 0);
            // First pass has one pair. The second pass has two, so its second
            // write must hit the cursor/offset guard rather than the buffer.
            if (self.calls != 1) out[0] = 0.03;
            self.calls += 1;
        }
    };
    var changing: Changing = .{};
    try std.testing.expectError(error.TrainingSourceChanged, extract(std.testing.allocator, .{
        .context = &changing,
        .query_rows = 2,
        .complete_query_rows = 2,
        .global_start = 0,
        .range_rows = 2,
        .original_rows = 2,
        .external_live_bytes = 0,
        .source_key_space = .global_original_range,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
        .copy_query_fn = Changing.copy,
    }, .{ .combined_cap = 4096 }));
}

fn allocationFailure(a: std.mem.Allocator) !void {
    const Fixture = struct {
        fn copy(_: ?*anyopaque, _: usize, out: []f64, control: solver.Control) !void {
            try control.check();
            @memset(out, 0.03);
        }
    };
    var result = try extract(a, .{
        .context = null,
        .query_rows = 2,
        .complete_query_rows = 2,
        .global_start = 4,
        .range_rows = 3,
        .original_rows = 16,
        .external_live_bytes = 0,
        .source_key_space = .global_original_range,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
        .copy_query_fn = Fixture.copy,
    }, .{ .combined_cap = 4096 });
    result.deinit();
}

test "range extraction cancellation and allocation failure unwind every owner" {
    const Fixture = struct {
        fn copy(_: ?*anyopaque, _: usize, out: []f64, _: solver.Control) !void {
            @memset(out, 0.03);
        }
    };
    const Cancel = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 4) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, extract(std.testing.allocator, .{
        .context = null,
        .query_rows = 2,
        .complete_query_rows = 2,
        .global_start = 0,
        .range_rows = 3,
        .original_rows = 3,
        .external_live_bytes = 0,
        .source_key_space = .global_original_range,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
        .copy_query_fn = Fixture.copy,
    }, .{ .combined_cap = 4096, .control = .{ .context = &cancel, .check_fn = Cancel.check } }));
    try std.testing.expectEqual(@as(usize, 4), cancel.checks);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
