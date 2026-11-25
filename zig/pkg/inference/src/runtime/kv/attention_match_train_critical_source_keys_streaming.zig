//! Bounded TRAIN-only critical-key extraction. Held-out tensors are absent.
const std = @import("std");
const solver = @import("attention_match.zig");
const dense = @import("attention_match_train_critical_source_keys.zig");

pub const ProbabilitySource = struct {
    context: ?*anyopaque,
    query_rows: usize,
    complete_query_rows: usize,
    source_rows: usize,
    external_live_bytes: usize,
    source_key_space: enum { global_original },
    normalization: enum { full_original_softmax_post_softcap_bias },
    storage: enum { exact_f32_widened },
    copy_query_fn: *const fn (?*anyopaque, usize, []f64, solver.Control) anyerror!void,
};
pub const Config = struct { threshold: f64 = dense.policy_threshold, combined_cap: usize, control: solver.Control = .{} };
pub const Range = dense.Range;
pub const RangeKeys = dense.RangeKeys;
pub const Result = struct {
    inner: dense.Result,
    probability_sha256: [32]u8,
    combined_peak_bytes: usize,
    pub fn deinit(self: *@This()) void {
        self.inner.deinit();
        self.* = undefined;
    }
};
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn byteCount(comptime T: type, n: usize) !usize {
    return mul(@sizeOf(T), n);
}

pub fn extractStreaming(a: std.mem.Allocator, src: ProbabilitySource, ranges_in: []const Range, cfg: Config) !Result {
    if (src.query_rows == 0 or src.complete_query_rows != src.query_rows or src.source_rows == 0 or ranges_in.len == 0 or !std.math.isFinite(cfg.threshold) or cfg.threshold <= 0 or cfg.threshold > 1) return error.InvalidAttentionShape;
    var end: usize = 0;
    for (ranges_in, 0..) |r, i| {
        const next = try add(r.start, r.rows);
        if (r.rows == 0 or next > src.source_rows or (i != 0 and r.start < end)) return error.InvalidAttentionShape;
        end = next;
    }
    var base = try add(src.external_live_bytes, try byteCount(f64, src.source_rows));
    base = try add(base, try mul(try byteCount(usize, src.source_rows), 2));
    if (base > cfg.combined_cap) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const row = try a.alloc(f64, src.source_rows);
    defer a.free(row);
    const slot = try a.alloc(usize, src.source_rows);
    defer a.free(slot);
    @memset(slot, std.math.maxInt(usize));
    const counts = try a.alloc(usize, src.source_rows);
    defer a.free(counts);
    @memset(counts, 0);
    for (ranges_in, 0..) |r, ri| for (r.start..r.start + r.rows) |k| {
        slot[k] = ri;
    };
    var first = std.crypto.hash.sha2.Sha256.init(.{});
    var pairs: usize = 0;
    for (0..src.query_rows) |q| {
        try cfg.control.check();
        try src.copy_query_fn(src.context, q, row, cfg.control);
        first.update(std.mem.sliceAsBytes(row));
        for (row, 0..) |p, k| {
            if (!std.math.isFinite(p) or p < 0 or p > 1) return error.NonFiniteAttention;
            if (slot[k] != std.math.maxInt(usize) and p >= cfg.threshold) {
                counts[k] = try add(counts[k], 1);
                pairs = try add(pairs, 1);
            }
        }
    }
    var keys_n: usize = 0;
    for (counts) |n| if (n != 0) {
        keys_n = try add(keys_n, 1);
    };
    var peak = base;
    peak = try add(peak, try byteCount(RangeKeys, ranges_in.len));
    peak = try add(peak, try mul(try byteCount(usize, keys_n), 2));
    peak = try add(peak, try byteCount(usize, try add(keys_n, 1)));
    peak = try add(peak, try byteCount(usize, pairs));
    if (peak > cfg.combined_cap) return error.CompactionWorkspaceLimit;
    const ranges = try a.alloc(RangeKeys, ranges_in.len);
    errdefer a.free(ranges);
    const keys = try a.alloc(usize, keys_n);
    errdefer a.free(keys);
    const offsets = try a.alloc(usize, try add(keys_n, 1));
    errdefer a.free(offsets);
    const queries = try a.alloc(usize, pairs);
    errdefer a.free(queries);
    const cursors = try a.alloc(usize, keys_n);
    defer a.free(cursors);
    var ki: usize = 0;
    var pi: usize = 0;
    for (ranges_in, ranges) |r, *out| {
        const begin = ki;
        for (r.start..r.start + r.rows) |k| {
            if (counts[k] == 0) {
                slot[k] = std.math.maxInt(usize);
                continue;
            }
            keys[ki] = k;
            offsets[ki] = pi;
            cursors[ki] = pi;
            slot[k] = ki;
            pi = try add(pi, counts[k]);
            ki += 1;
        }
        out.* = .{ .start = r.start, .rows = r.rows, .key_begin = begin, .key_count = ki - begin };
    }
    offsets[keys_n] = pi;
    var second = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..src.query_rows) |q| {
        try cfg.control.check();
        try src.copy_query_fn(src.context, q, row, cfg.control);
        second.update(std.mem.sliceAsBytes(row));
        for (row, 0..) |p, k| {
            if (!std.math.isFinite(p) or p < 0 or p > 1) return error.NonFiniteAttention;
            const s = slot[k];
            if (s != std.math.maxInt(usize) and p >= cfg.threshold) {
                if (cursors[s] >= offsets[s + 1]) return error.TrainingSourceChanged;
                queries[cursors[s]] = q;
                cursors[s] += 1;
            }
        }
    }
    var h1: [32]u8 = undefined;
    var h2: [32]u8 = undefined;
    first.final(&h1);
    second.final(&h2);
    if (!std.mem.eql(u8, &h1, &h2)) return error.TrainingSourceChanged;
    for (cursors, 0..) |c, i| if (c != offsets[i + 1]) return error.TrainingSourceChanged;
    try cfg.control.check();
    return .{ .inner = .{ .allocator = a, .ranges = ranges, .source_keys = keys, .query_offsets = offsets, .critical_queries = queries, .query_rows = src.query_rows, .complete_query_rows = src.complete_query_rows, .source_rows = src.source_rows, .threshold = cfg.threshold, .peak_workspace_bytes = peak - src.external_live_bytes }, .probability_sha256 = h1, .combined_peak_bytes = peak };
}

const DenseContext = struct {
    values: []const f64,
    rows: usize,
    fn copy(raw: ?*anyopaque, q: usize, out: []f64, control: solver.Control) !void {
        try control.check();
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        @memcpy(out, self.values[q * self.rows ..][0..self.rows]);
    }
};
pub fn extractDense(a: std.mem.Allocator, view: dense.ProbabilityView, ranges: []const Range, cfg: Config) !Result {
    if (view.query_rows == 0 or view.source_rows == 0 or view.complete_query_rows != view.query_rows or
        view.values.len != try mul(view.query_rows, view.source_rows))
        return error.InvalidAttentionShape;
    var context = DenseContext{ .values = view.values, .rows = view.source_rows };
    return extractStreaming(a, .{ .context = &context, .query_rows = view.query_rows, .complete_query_rows = view.complete_query_rows, .source_rows = view.source_rows, .external_live_bytes = try byteCount(f64, view.values.len), .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened, .copy_query_fn = DenseContext.copy }, ranges, cfg);
}

test "dense wrapper rejects shape mismatch and overflow before allocation" {
    const short = [_]f64{0.5};
    const long = [_]f64{ 0.5, 0.5, 0.5 };
    try std.testing.expectError(error.InvalidAttentionShape, extractDense(std.testing.failing_allocator, .{ .values = &short, .query_rows = 1, .complete_query_rows = 1, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 }));
    try std.testing.expectError(error.InvalidAttentionShape, extractDense(std.testing.failing_allocator, .{ .values = &long, .query_rows = 1, .complete_query_rows = 1, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 }));
    try std.testing.expectError(error.InvalidAttentionShape, extractDense(std.testing.failing_allocator, .{ .values = &.{}, .query_rows = std.math.maxInt(usize), .complete_query_rows = std.math.maxInt(usize), .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 }));
}

test "streaming and dense wrappers agree and bind two passes" {
    const values = [_]f64{ 0.025, 0.0, 0.03, 0.024, 0.0, 0.04 };
    const view: dense.ProbabilityView = .{ .values = &values, .query_rows = 2, .complete_query_rows = 2, .source_rows = 3, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened };
    var result = try extractDense(std.testing.allocator, view, &.{.{ .start = 0, .rows = 3 }}, .{ .combined_cap = 4096 });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, result.inner.source_keys);
    var reference = try dense.extract(std.testing.allocator, view, &.{.{ .start = 0, .rows = 3 }}, .{ .max_workspace_bytes = 4096 });
    defer reference.deinit();
    try std.testing.expectEqualSlices(usize, reference.source_keys, result.inner.source_keys);
    try std.testing.expectEqualSlices(usize, reference.query_offsets, result.inner.query_offsets);
    try std.testing.expectEqualSlices(usize, reference.critical_queries, result.inner.critical_queries);
    var expected_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&values), &expected_hash, .{});
    try std.testing.expectEqual(expected_hash, result.probability_sha256);
}

test "streaming detects a changed probability source" {
    const C = struct {
        calls: usize = 0,
        fn copy(raw: ?*anyopaque, _: usize, out: []f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            @memset(out, if (self.calls == 0) 0.03 else 0.04);
            self.calls += 1;
        }
    };
    var c = C{};
    try std.testing.expectError(error.TrainingSourceChanged, extractStreaming(std.testing.allocator, .{ .context = &c, .query_rows = 1, .complete_query_rows = 1, .source_rows = 2, .external_live_bytes = 0, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened, .copy_query_fn = C.copy }, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 }));
}

fn allocationFailure(a: std.mem.Allocator) !void {
    const values = [_]f64{ 0.03, 0.04 };
    const view: dense.ProbabilityView = .{ .values = &values, .query_rows = 1, .complete_query_rows = 1, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened };
    var result = try extractDense(a, view, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 });
    result.deinit();
}

test "streaming critical keys cap cancellation and allocation unwind" {
    const values = [_]f64{ 0.03, 0.04 };
    const view: dense.ProbabilityView = .{ .values = &values, .query_rows = 1, .complete_query_rows = 1, .source_rows = 2, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened };
    try std.testing.expectError(error.CompactionWorkspaceLimit, extractDense(std.testing.allocator, view, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 1 }));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, extractDense(std.testing.allocator, view, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096, .control = .{ .check_fn = Cancel.check } }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "streaming rejects added second-pass critical pairs before writing beyond their segment" {
    const Changing = struct {
        calls: usize = 0,
        fn copy(raw: ?*anyopaque, _: usize, out: []f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            @memset(out, 0);
            if (self.calls != 1) out[0] = 0.03;
            self.calls += 1;
        }
    };
    var changing: Changing = .{};
    try std.testing.expectError(error.TrainingSourceChanged, extractStreaming(std.testing.allocator, .{
        .context = &changing,
        .query_rows = 2,
        .complete_query_rows = 2,
        .source_rows = 2,
        .external_live_bytes = 0,
        .source_key_space = .global_original,
        .normalization = .full_original_softmax_post_softcap_bias,
        .storage = .exact_f32_widened,
        .copy_query_fn = Changing.copy,
    }, &.{.{ .start = 0, .rows = 2 }}, .{ .combined_cap = 4096 }));
}
