// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact one-chunk training cache for cumulative admission experiments.
const std = @import("std");
const solver = @import("attention_match.zig");
const page_cache = @import("attention_match_training_page_cache.zig");

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}

pub const View = struct {
    probabilities: []const f64,
    original_outputs: []const f64,
    rho: []const f64,
    query_rows: usize,
    chunk_start: usize,
    chunk_rows: usize,
    dim: usize,
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    source_keys: [*]const f64,
    source_values: [*]const f64,
    source_bias: ?[*]const f64,
    source_rows: usize,
    source_dim: usize,
    source_scale: f64,
    source_softcap: f64,
    train_ptr: [*]const f64,
    train_len: usize,
    chunk_start: usize,
    chunk_rows: usize,
    probabilities: []f64,
    original_outputs: []f64,
    rho: []f64,
    retained_bytes: usize,
    attention_batches: usize,
    backend: ?solver.Backend,
    owns_original_outputs: bool = true,

    fn sameBackend(a: ?solver.Backend, b: ?solver.Backend) bool {
        if (a == null or b == null) return a == null and b == null;
        const x = a.?;
        const y = b.?;
        return x.validation_batch == y.validation_batch and x.raw_range_policy == y.raw_range_policy and x.context == y.context and x.workspace_fn == y.workspace_fn and x.attend_fn == y.attend_fn and x.summary_fn == y.summary_fn and x.values_fn == y.values_fn and x.select_fn == y.select_fn and x.raw_select_fn == y.raw_select_fn and x.raw_range_workspace_fn == y.raw_range_workspace_fn and x.raw_range_begin_fn == y.raw_range_begin_fn and x.raw_range_end_fn == y.raw_range_end_fn;
    }

    pub fn retainedBytes(query_rows: usize, chunk_rows: usize, dim: usize) !usize {
        return mul(try add(try add(try mul(query_rows, chunk_rows), try mul(query_rows, dim)), query_rows), @sizeOf(f64));
    }

    pub fn constructionScratchBytes(source_rows: usize, dim: usize) !usize {
        _ = dim;
        return mul(try add(try mul(64, source_rows), 64), @sizeOf(f64));
    }

    pub fn requiredBytes(source_rows: usize, query_rows: usize, chunk_rows: usize, dim: usize, cfg: solver.Config) !usize {
        const backend = if (cfg.backend) |value| try value.workspace_fn(value.context, source_rows, 1, @min(query_rows, 64), dim) else 0;
        return add(try retainedBytes(query_rows, chunk_rows, dim), try add(try constructionScratchBytes(source_rows, dim), backend));
    }

    pub fn init(allocator: std.mem.Allocator, source: solver.Head, train: []const f64, chunk_start: usize, chunk_rows: usize, cfg: solver.Config, max_bytes: usize) !Cache {
        try source.validate();
        if (train.len == 0 or train.len % source.dim != 0 or chunk_rows == 0 or chunk_start > source.rows or chunk_rows > source.rows - chunk_start) return error.InvalidFitConfig;
        const q = train.len / source.dim;
        const retained = try retainedBytes(q, chunk_rows, source.dim);
        const required = try requiredBytes(source.rows, q, chunk_rows, source.dim, cfg);
        if (required > max_bytes or required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        const probabilities = try allocator.alloc(f64, try mul(q, chunk_rows));
        errdefer allocator.free(probabilities);
        const outputs = try allocator.alloc(f64, try mul(q, source.dim));
        errdefer allocator.free(outputs);
        const rho = try allocator.alloc(f64, q);
        errdefer allocator.free(rho);
        const batch = try allocator.alloc(f64, try mul(64, source.rows));
        defer allocator.free(batch);
        var offset: usize = 0;
        var batches: usize = 0;
        while (offset < q) {
            try cfg.control.check();
            const take = @min(@as(usize, 64), q - offset);
            try solver.attendBatch(allocator, source, train[offset * source.dim ..][0 .. take * source.dim], batch[0 .. take * source.rows], outputs[offset * source.dim ..][0 .. take * source.dim], null, cfg);
            for (0..take) |local| {
                const source_row = batch[local * source.rows + chunk_start ..][0..chunk_rows];
                const destination = probabilities[(offset + local) * chunk_rows ..][0..chunk_rows];
                @memcpy(destination, source_row);
                var sum: f64 = 0;
                for (source_row) |value| {
                    if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                    sum += value;
                }
                if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
                rho[offset + local] = sum;
            }
            batches += 1;
            offset += take;
        }
        return .{ .allocator = allocator, .source_keys = source.keys.ptr, .source_values = source.values.ptr, .source_bias = if (source.bias) |bias| bias.ptr else null, .source_rows = source.rows, .source_dim = source.dim, .source_scale = source.scale, .source_softcap = source.softcap, .train_ptr = train.ptr, .train_len = train.len, .chunk_start = chunk_start, .chunk_rows = chunk_rows, .probabilities = probabilities, .original_outputs = outputs, .rho = rho, .retained_bytes = retained, .attention_batches = batches, .backend = cfg.backend };
    }

    pub fn validate(self: *const Cache, source: solver.Head, train: []const f64, chunk_start: usize, chunk_rows: usize, cfg: solver.Config) !void {
        if (source.keys.ptr != self.source_keys or source.values.ptr != self.source_values or
            (if (source.bias) |bias| bias.ptr else null) != self.source_bias or source.rows != self.source_rows or source.dim != self.source_dim or
            source.scale != self.source_scale or source.softcap != self.source_softcap or train.ptr != self.train_ptr or train.len != self.train_len or
            chunk_start != self.chunk_start or chunk_rows != self.chunk_rows or !sameBackend(self.backend, cfg.backend)) return error.InvalidFitConfig;
    }

    /// Transfer a page child's contiguous P/rho storage while borrowing its
    /// immutable original Y. The page owner must outlive this cache.
    pub fn fromPageChild(source: solver.Head, train: []const f64, child: *page_cache.Child, cfg: solver.Config) !Cache {
        try child.owner.validate(source, train, cfg);
        if (child.originalOutputs().len != train.len or child.probabilities.len != try mul(train.len / source.dim, child.chunk_rows) or child.rho.len != train.len / source.dim)
            return error.InvalidFitConfig;
        const result: Cache = .{
            .allocator = child.allocator,
            .source_keys = source.keys.ptr,
            .source_values = source.values.ptr,
            .source_bias = if (source.bias) |bias| bias.ptr else null,
            .source_rows = source.rows,
            .source_dim = source.dim,
            .source_scale = source.scale,
            .source_softcap = source.softcap,
            .train_ptr = train.ptr,
            .train_len = train.len,
            .chunk_start = child.chunk_start,
            .chunk_rows = child.chunk_rows,
            .probabilities = child.probabilities,
            .original_outputs = @constCast(child.originalOutputs()),
            .rho = child.rho,
            .retained_bytes = child.retained_bytes,
            .attention_batches = 0,
            .backend = cfg.backend,
            .owns_original_outputs = false,
        };
        child.* = undefined;
        return result;
    }

    pub fn view(self: *const Cache) View {
        return .{ .probabilities = self.probabilities, .original_outputs = self.original_outputs, .rho = self.rho, .query_rows = self.rho.len, .chunk_start = self.chunk_start, .chunk_rows = self.chunk_rows, .dim = self.source_dim };
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.probabilities);
        if (self.owns_original_outputs) self.allocator.free(self.original_outputs);
        self.allocator.free(self.rho);
        self.* = undefined;
    }
};

test "one chunk training cache streams exact probabilities outputs and tail" {
    const q: usize = 65;
    const train = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(train);
    for (train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const source: solver.Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0.1, -0.2, 0.3 }, .rows = 3, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    var cache = try Cache.init(std.testing.allocator, source, train, 1, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 2), cache.attention_batches);
    const full = try std.testing.allocator.alloc(f64, q * source.rows);
    defer std.testing.allocator.free(full);
    const outputs = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(outputs);
    try solver.attendBatch(std.testing.allocator, source, train, full, outputs, null, .{ .target_rows = 1 });
    for (0..q) |i| {
        try std.testing.expectEqualSlices(f64, full[i * 3 + 1 ..][0..2], cache.probabilities[i * 2 ..][0..2]);
        try std.testing.expectEqual(outputs[i], cache.original_outputs[i]);
        try std.testing.expectEqual(full[i * 3 + 1] + full[i * 3 + 2], cache.rho[i]);
    }
}

test "one chunk training cache checks cap cancellation ownership and allocation unwind" {
    const source: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, Cache.init(std.testing.allocator, source, &.{ 0, 1 }, 0, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 }, 1));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, Cache.init(std.testing.allocator, source, &.{ 0, 1 }, 0, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }, 1 << 20));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator) !void {
            const h: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
            var cache = try Cache.init(a, h, &.{ 0, 1 }, 0, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
            cache.deinit();
        }
    }.run, .{});
    var cache = try Cache.init(std.testing.allocator, source, &.{ 0, 1 }, 0, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer cache.deinit();
    try std.testing.expectError(error.InvalidFitConfig, cache.validate(source, &.{ 0, 1 }, 0, 1, .{ .target_rows = 1 }));
}

test "one chunk training cache 32K E2 bound is below 512 MiB" {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, n: usize, m: usize, q: usize, d: usize) !usize {
            return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, n, q), try std.math.mul(usize, m, d)), @sizeOf(f64));
        }
    };
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace } };
    const retained = try Cache.retainedBytes(10240, 256, 512);
    const construction = try Cache.constructionScratchBytes(32512, 512);
    const backend = try Fake.workspace(null, 32512, 1, 64, 512);
    const required = try Cache.requiredBytes(32512, 10240, 256, 512, cfg);
    try std.testing.expectEqual(@as(usize, 62_996_480), retained);
    try std.testing.expectEqual(@as(usize, 16_646_656), construction);
    try std.testing.expectEqual(retained + construction + backend, required);
    try std.testing.expect(required < 512 << 20);
}

test "one chunk training cache binds backend tuple and cancels between backend batches" {
    const Fake = struct {
        var calls: usize = 0;
        var armed = false;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 4096;
        }
        fn check(_: ?*anyopaque) !void {
            if (armed) return error.Cancelled;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, queries: []const f64, p: []f64, y: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
            @memset(p, 0.5);
            if (y) |out| @memset(out, 1);
            _ = h;
            _ = queries;
            armed = true;
        }
        fn alternate(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {}
    };
    const source: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var train: [65]f64 = @splat(0);
    var marker: u8 = 0;
    var cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .context = &marker, .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .control = .{ .check_fn = Fake.check } };
    Fake.calls = 0;
    Fake.armed = false;
    try std.testing.expectError(error.Cancelled, Cache.init(std.testing.allocator, source, &train, 0, 2, cfg, 1 << 20));
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    cfg.control = .{};
    Fake.armed = false;
    var cache = try Cache.init(std.testing.allocator, source, &train, 0, 2, cfg, 1 << 20);
    defer cache.deinit();
    var changed = cfg;
    changed.backend.?.attend_fn = Fake.alternate;
    try std.testing.expectError(error.InvalidFitConfig, cache.validate(source, &train, 0, 2, changed));
    const required = try Cache.requiredBytes(2, 65, 2, 1, cfg);
    Fake.calls = 0;
    try std.testing.expectError(error.CompactionWorkspaceLimit, Cache.init(std.testing.allocator, source, &train, 0, 2, cfg, required - 1));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator) !void {
            const B = struct {
                fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
                    return 4096;
                }
                fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, p: []f64, y: ?[]f64, _: ?[]f64, _: solver.Control) !void {
                    @memset(p, 0.5);
                    if (y) |out| @memset(out, 1);
                }
            };
            const h: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
            var qs: [65]f64 = @splat(0);
            var allocation_marker: u8 = 0;
            var c = try Cache.init(a, h, &qs, 0, 2, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .context = &allocation_marker, .workspace_fn = B.workspace, .attend_fn = B.attend } }, 1 << 20);
            c.deinit();
        }
    }.run, .{});
}

test "page child remains caller-owned when transfer validation rejects" {
    const source: solver.Head = .{ .keys = &.{ 0, 1, 2 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const train = [_]f64{ -1, 0, 1 };
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, _: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
            @memset(probabilities, 1 / @as(f64, @floatFromInt(h.rows)));
            if (outputs) |out| @memset(out, 1);
        }
    };
    var marker: u8 = 0;
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .context = &marker, .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } };
    Fake.calls = 0;
    var parent = try page_cache.Cache.init(std.testing.allocator, source, &train, 0, 3, cfg, 1 << 20);
    defer parent.deinit();
    const calls_after_parent = Fake.calls;
    const child_bytes = try page_cache.Cache.childRetainedBytes(train.len, 2);
    try std.testing.expectError(error.CompactionWorkspaceLimit, parent.child(std.testing.allocator, source, &train, 0, 2, cfg, parent.retained_bytes + child_bytes - 1));
    try std.testing.expectEqual(calls_after_parent, Fake.calls);
    var child = try parent.child(std.testing.allocator, source, &train, 0, 2, cfg, parent.retained_bytes + child_bytes);
    defer child.deinit();
    var other_marker: u8 = 0;
    const mismatched: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .context = &other_marker, .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } };
    try std.testing.expectError(error.InvalidFitConfig, Cache.fromPageChild(source, &train, &child, mismatched));
    try std.testing.expectEqual(@as(usize, 6), child.probabilities.len);
    try std.testing.expectEqual(@as(usize, 3), child.rho.len);
}
