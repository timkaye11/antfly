// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const solver = @import("attention_match.zig");

pub const maximum_combined_bytes: usize = 1024 * 1024 * 1024;

fn requiredCacheBytes(query_count: usize, rows: usize, dim: usize) !usize {
    const probabilities = try std.math.mul(usize, query_count, rows);
    const outputs = try std.math.mul(usize, query_count, dim);
    return std.math.mul(usize, try std.math.add(usize, probabilities, outputs), @sizeOf(f64));
}

pub const Cache = struct {
    allocator: std.mem.Allocator,
    source_keys_ptr: [*]const f64,
    source_values_ptr: [*]const f64,
    source_bias_ptr: ?[*]const f64,
    train_ptr: [*]const f64,
    rows: usize,
    dim: usize,
    query_count: usize,
    scale: f64,
    softcap: f64,
    backend: ?solver.Backend,
    reserved_workspace_bytes: usize,
    probabilities: []f64,
    outputs: []f64,

    pub fn init(allocator: std.mem.Allocator, source: solver.Head, train: []const f64, cfg: solver.Config, combined_cap: usize) !Cache {
        if (comptime !@import("builtin").is_test) @compileError("global training cache is diagnostic-test-only");
        try source.validate();
        try solver.validateConfigForTest(cfg, source.rows);
        if (train.len == 0 or train.len % source.dim != 0 or combined_cap > maximum_combined_bytes or cfg.max_workspace_bytes >= combined_cap) return error.InvalidFitConfig;
        const q = train.len / source.dim;
        const probability_count = try std.math.mul(usize, q, source.rows);
        const output_count = try std.math.mul(usize, q, source.dim);
        const cache_bytes = try requiredCacheBytes(q, source.rows, source.dim);
        const cache_cap = combined_cap - cfg.max_workspace_bytes;
        if (cache_bytes > cache_cap) return error.CompactionWorkspaceLimit;
        if (cfg.backend) |backend| if (try backend.workspace_fn(backend.context, source.rows, 1, @min(q, 64), source.dim) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        const probabilities = try allocator.alloc(f64, probability_count);
        errdefer allocator.free(probabilities);
        const outputs = try allocator.alloc(f64, output_count);
        errdefer allocator.free(outputs);
        var offset: usize = 0;
        while (offset < q) {
            try cfg.control.check();
            const take = @min(@as(usize, 64), q - offset);
            try solver.attendBatch(allocator, source, train[offset * source.dim ..][0 .. take * source.dim], probabilities[offset * source.rows ..][0 .. take * source.rows], outputs[offset * source.dim ..][0 .. take * source.dim], null, cfg);
            offset += take;
        }
        return .{ .allocator = allocator, .source_keys_ptr = source.keys.ptr, .source_values_ptr = source.values.ptr, .source_bias_ptr = if (source.bias) |b| b.ptr else null, .train_ptr = train.ptr, .rows = source.rows, .dim = source.dim, .query_count = q, .scale = source.scale, .softcap = source.softcap, .backend = cfg.backend, .reserved_workspace_bytes = cfg.max_workspace_bytes, .probabilities = probabilities, .outputs = outputs };
    }

    pub fn validate(self: Cache, source: solver.Head, train: []const f64, cfg: solver.Config) !void {
        if (source.keys.ptr != self.source_keys_ptr or source.values.ptr != self.source_values_ptr or (if (source.bias) |b| b.ptr else null) != self.source_bias_ptr or train.ptr != self.train_ptr or train.len != self.query_count * self.dim or source.rows != self.rows or source.dim != self.dim or source.scale != self.scale or source.softcap != self.softcap or !backendEqual(self.backend, cfg.backend) or cfg.max_workspace_bytes > self.reserved_workspace_bytes) return error.InvalidFitConfig;
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.probabilities);
        self.allocator.free(self.outputs);
        self.* = undefined;
    }
};

fn backendEqual(a: ?solver.Backend, b: ?solver.Backend) bool {
    if (a == null or b == null) return a == null and b == null;
    const x = a.?;
    const y = b.?;
    return x.validation_batch == y.validation_batch and x.raw_range_policy == y.raw_range_policy and x.context == y.context and x.workspace_fn == y.workspace_fn and x.attend_fn == y.attend_fn and x.summary_fn == y.summary_fn and x.values_fn == y.values_fn and x.select_fn == y.select_fn and x.raw_select_fn == y.raw_select_fn and x.raw_range_workspace_fn == y.raw_range_workspace_fn and x.raw_range_begin_fn == y.raw_range_begin_fn and x.raw_range_end_fn == y.raw_range_end_fn;
}

test "global training cache checks combined cap before backend dispatch" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    Fake.calls = 0;
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, Cache.init(std.testing.allocator, head, &.{0}, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 1024 }, 1024 + 15));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
}

test "global training cache constructs backend probabilities once" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, control: solver.Control) !void {
            calls += 1;
            for (0..queries.len / head.dim) |i| _ = try solver.attend(head, queries[i * head.dim ..][0..head.dim], probabilities[i * head.rows ..][0..head.rows], if (outputs) |out| out[i * head.dim ..][0..head.dim] else null);
            try control.check();
        }
    };
    Fake.calls = 0;
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    var cache = try Cache.init(std.testing.allocator, head, &.{ 0, 1 }, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } }, maximum_combined_bytes);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    const Other = struct {
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {}
    };
    try std.testing.expectError(error.InvalidFitConfig, cache.validate(head, &.{ 0, 1 }, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Other.attend } }));
    try std.testing.expectError(error.InvalidFitConfig, cache.validate(head, &.{ 0, 1 }, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = cache.reserved_workspace_bytes + 1 }));
}

test "global training cache checks size overflow and allocation failure before dispatch" {
    try std.testing.expectError(error.Overflow, requiredCacheBytes(std.math.maxInt(usize), 2, 1));
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    Fake.calls = 0;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.OutOfMemory, Cache.init(failing.allocator(), head, &.{0}, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend } }, maximum_combined_bytes));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
}
