const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_domain_cv_split_v1|domains=P256,H256,F256|cv=row_mod_8_eq_7_per_domain|fit=remaining|order=consumer_then_domain_then_row";
pub const rows_per_domain: usize = 256;
pub const domains_per_consumer: usize = 3;
pub const cv_stride: usize = 8;

pub const Result = struct {
    allocator: std.mem.Allocator,
    fit_queries: []f64,
    cv_queries: []f64,
    fit_original_rows: []u32,
    cv_original_rows: []u32,
    fit_rows_per_consumer: usize,
    cv_rows_per_consumer: usize,
    live_bytes: usize,
    digest: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.cv_original_rows);
        self.allocator.free(self.fit_original_rows);
        self.allocator.free(self.cv_queries);
        self.allocator.free(self.fit_queries);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}

pub fn workspaceBytes(query_rows: usize, dim: usize) !usize {
    const elements = try mul(query_rows, dim);
    return add(try mul(elements, @sizeOf(f64)), try mul(query_rows, @sizeOf(u32)));
}

pub fn split(allocator: std.mem.Allocator, queries: []const f64, dim: usize, consumer_count: usize, external_live_bytes: usize, combined_cap: usize, control: solver.Control) !Result {
    try control.check();
    if (dim == 0 or consumer_count == 0 or combined_cap == 0) return error.InvalidFitConfig;
    const rows_per_consumer = rows_per_domain * domains_per_consumer;
    const total_rows = try mul(consumer_count, rows_per_consumer);
    if (total_rows > std.math.maxInt(u32)) return error.InvalidAttentionShape;
    if (queries.len != try mul(total_rows, dim)) return error.InvalidAttentionShape;
    const cv_rows_per_consumer = rows_per_consumer / cv_stride;
    const fit_rows_per_consumer = rows_per_consumer - cv_rows_per_consumer;
    const cv_rows = try mul(consumer_count, cv_rows_per_consumer);
    const fit_rows = total_rows - cv_rows;
    const owned_bytes = try workspaceBytes(total_rows, dim);
    if (external_live_bytes > combined_cap or owned_bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;

    const fit_queries = try allocator.alloc(f64, try mul(fit_rows, dim));
    errdefer allocator.free(fit_queries);
    const cv_queries = try allocator.alloc(f64, try mul(cv_rows, dim));
    errdefer allocator.free(cv_queries);
    const fit_indices = try allocator.alloc(u32, fit_rows);
    errdefer allocator.free(fit_indices);
    const cv_indices = try allocator.alloc(u32, cv_rows);
    errdefer allocator.free(cv_indices);

    var fit_at: usize = 0;
    var cv_at: usize = 0;
    for (0..consumer_count) |consumer| for (0..domains_per_consumer) |domain| for (0..rows_per_domain) |row| {
        try control.check();
        const original_row = consumer * rows_per_consumer + domain * rows_per_domain + row;
        const is_cv = row % cv_stride == cv_stride - 1;
        const target_row = if (is_cv) cv_at else fit_at;
        const target = if (is_cv) cv_queries else fit_queries;
        @memcpy(target[target_row * dim ..][0..dim], queries[original_row * dim ..][0..dim]);
        if (is_cv) {
            cv_indices[cv_at] = @intCast(original_row);
            cv_at += 1;
        } else {
            fit_indices[fit_at] = @intCast(original_row);
            fit_at += 1;
        }
    };
    if (fit_at != fit_rows or cv_at != cv_rows) return error.InvalidAttentionShape;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(std.mem.asBytes(&dim));
    hash.update(std.mem.asBytes(&consumer_count));
    hash.update(std.mem.sliceAsBytes(fit_indices));
    hash.update(std.mem.sliceAsBytes(cv_indices));
    hash.update(std.mem.sliceAsBytes(fit_queries));
    hash.update(std.mem.sliceAsBytes(cv_queries));
    return .{ .allocator = allocator, .fit_queries = fit_queries, .cv_queries = cv_queries, .fit_original_rows = fit_indices, .cv_original_rows = cv_indices, .fit_rows_per_consumer = fit_rows_per_consumer, .cv_rows_per_consumer = cv_rows_per_consumer, .live_bytes = owned_bytes, .digest = hash.finalResult() };
}

test "P H F split is disjoint complete stable and ordered" {
    const consumers = 2;
    const dim = 1;
    var queries: [consumers * domains_per_consumer * rows_per_domain]f64 = undefined;
    for (&queries, 0..) |*value, row| value.* = @floatFromInt(row);
    var result = try split(std.testing.allocator, &queries, dim, consumers, 0, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 672), result.fit_rows_per_consumer);
    try std.testing.expectEqual(@as(usize, 96), result.cv_rows_per_consumer);
    try std.testing.expectEqual(@as(usize, consumers * 672), result.fit_original_rows.len);
    try std.testing.expectEqual(@as(usize, consumers * 96), result.cv_original_rows.len);
    var seen: [queries.len]bool = @splat(false);
    for (result.fit_original_rows) |row| {
        try std.testing.expect(!seen[row]);
        seen[row] = true;
    }
    for (result.cv_original_rows) |row| {
        try std.testing.expect(!seen[row]);
        seen[row] = true;
        try std.testing.expectEqual(@as(usize, 7), @as(usize, row) % rows_per_domain % cv_stride);
    }
    for (seen) |value| try std.testing.expect(value);
    try std.testing.expectEqual(@as(f64, 7), result.cv_queries[0]);
    try std.testing.expectEqual(@as(f64, 15), result.cv_queries[1]);
}

test "split rejects cap before allocation and unwinds allocation failures" {
    var queries: [domains_per_consumer * rows_per_domain]f64 = @splat(0);
    try std.testing.expectError(error.CompactionWorkspaceLimit, split(std.testing.failing_allocator, &queries, 1, 1, 0, 1, .{}));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, q: []const f64) !void {
            var result = try split(a, q, 1, 1, 0, 1 << 20, .{});
            result.deinit();
        }
    }.run, .{queries[0..]});
}

test "split cancellation is atomic" {
    const C = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 10) return error.Cancelled;
        }
    };
    var state: C = .{};
    var queries: [domains_per_consumer * rows_per_domain]f64 = @splat(0);
    try std.testing.expectError(error.Cancelled, split(std.testing.allocator, &queries, 1, 1, 0, 1 << 20, .{ .context = &state, .check_fn = C.check }));
}
