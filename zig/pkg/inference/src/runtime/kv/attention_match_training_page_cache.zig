// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only exact page cache for cumulative training attention.
//! A page retains a q-by-page crop of the original probabilities and one
//! immutable q-by-d original output matrix. Child chunks own only their
//! contiguous probability crop and rho; they alias the parent's outputs.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const page_rows: usize = 1024;

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

fn sameBackend(a: ?solver.Backend, b: ?solver.Backend) bool {
    if (a == null or b == null) return a == null and b == null;
    const x = a.?;
    const y = b.?;
    return x.validation_batch == y.validation_batch and x.raw_range_policy == y.raw_range_policy and x.context == y.context and
        x.workspace_fn == y.workspace_fn and x.attend_fn == y.attend_fn and
        x.summary_fn == y.summary_fn and x.values_fn == y.values_fn and
        x.select_fn == y.select_fn and x.raw_select_fn == y.raw_select_fn and
        x.raw_range_workspace_fn == y.raw_range_workspace_fn and x.raw_range_begin_fn == y.raw_range_begin_fn and x.raw_range_end_fn == y.raw_range_end_fn;
}

pub const Child = struct {
    allocator: std.mem.Allocator,
    owner: *const Cache,
    probabilities: []f64,
    rho: []f64,
    chunk_start: usize,
    chunk_rows: usize,
    retained_bytes: usize,

    pub fn originalOutputs(self: *const Child) []const f64 {
        return self.owner.original_outputs;
    }

    pub fn deinit(self: *Child) void {
        self.allocator.free(self.probabilities);
        self.allocator.free(self.rho);
        self.* = undefined;
    }
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
    page_start: usize,
    page_len: usize,
    probabilities: []f64,
    original_outputs: []f64,
    retained_bytes: usize,
    attention_batches: usize,
    backend: ?solver.Backend,

    pub fn retainedBytes(query_rows: usize, rows: usize, dim: usize) !usize {
        return mul(try add(try mul(query_rows, rows), try mul(query_rows, dim)), @sizeOf(f64));
    }

    pub fn childRetainedBytes(query_rows: usize, rows: usize) !usize {
        return mul(try add(try mul(query_rows, rows), query_rows), @sizeOf(f64));
    }

    pub fn constructionScratchBytes(source_rows: usize) !usize {
        return mul(try mul(64, source_rows), @sizeOf(f64));
    }

    pub fn requiredBytes(source_rows: usize, query_rows: usize, rows: usize, dim: usize, cfg: solver.Config) !usize {
        const backend = if (cfg.backend) |value| try value.workspace_fn(value.context, source_rows, 1, @min(query_rows, 64), dim) else 0;
        return add(try retainedBytes(query_rows, rows, dim), try add(try constructionScratchBytes(source_rows), backend));
    }

    pub fn init(allocator: std.mem.Allocator, source: solver.Head, train: []const f64, start: usize, rows: usize, cfg: solver.Config, max_bytes: usize) !Cache {
        try source.validate();
        if (train.len == 0 or train.len % source.dim != 0 or rows == 0 or rows > page_rows or start > source.rows or rows > source.rows - start)
            return error.InvalidFitConfig;
        const q = train.len / source.dim;
        const retained = try retainedBytes(q, rows, source.dim);
        const required = try requiredBytes(source.rows, q, rows, source.dim, cfg);
        if (required > max_bytes or required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        const probabilities = try allocator.alloc(f64, try mul(q, rows));
        errdefer allocator.free(probabilities);
        const outputs = try allocator.alloc(f64, try mul(q, source.dim));
        errdefer allocator.free(outputs);
        const batch = try allocator.alloc(f64, try mul(64, source.rows));
        defer allocator.free(batch);
        var offset: usize = 0;
        var batches: usize = 0;
        while (offset < q) {
            try cfg.control.check();
            const take = @min(@as(usize, 64), q - offset);
            try solver.attendBatch(allocator, source, train[offset * source.dim ..][0 .. take * source.dim], batch[0 .. take * source.rows], outputs[offset * source.dim ..][0 .. take * source.dim], null, cfg);
            for (0..take) |local| {
                const source_row = batch[local * source.rows + start ..][0..rows];
                const destination = probabilities[(offset + local) * rows ..][0..rows];
                @memcpy(destination, source_row);
                for (destination) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            }
            batches += 1;
            offset += take;
        }
        return .{
            .allocator = allocator,
            .source_keys = source.keys.ptr,
            .source_values = source.values.ptr,
            .source_bias = if (source.bias) |bias| bias.ptr else null,
            .source_rows = source.rows,
            .source_dim = source.dim,
            .source_scale = source.scale,
            .source_softcap = source.softcap,
            .train_ptr = train.ptr,
            .train_len = train.len,
            .page_start = start,
            .page_len = rows,
            .probabilities = probabilities,
            .original_outputs = outputs,
            .retained_bytes = retained,
            .attention_batches = batches,
            .backend = cfg.backend,
        };
    }

    pub fn validate(self: *const Cache, source: solver.Head, train: []const f64, cfg: solver.Config) !void {
        if (source.keys.ptr != self.source_keys or source.values.ptr != self.source_values or
            (if (source.bias) |bias| bias.ptr else null) != self.source_bias or
            source.rows != self.source_rows or source.dim != self.source_dim or
            source.scale != self.source_scale or source.softcap != self.source_softcap or
            train.ptr != self.train_ptr or train.len != self.train_len or !sameBackend(self.backend, cfg.backend))
            return error.InvalidFitConfig;
    }

    pub fn child(self: *const Cache, allocator: std.mem.Allocator, source: solver.Head, train: []const f64, chunk_start: usize, chunk_rows: usize, cfg: solver.Config, max_bytes: usize) !Child {
        try self.validate(source, train, cfg);
        if (chunk_rows == 0 or chunk_start < self.page_start or chunk_start > self.page_start + self.page_len or chunk_rows > self.page_start + self.page_len - chunk_start)
            return error.InvalidFitConfig;
        const q = train.len / source.dim;
        const retained = try childRetainedBytes(q, chunk_rows);
        if (try add(self.retained_bytes, retained) > max_bytes or try add(self.retained_bytes, retained) > cfg.max_workspace_bytes)
            return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        const probabilities = try allocator.alloc(f64, try mul(q, chunk_rows));
        errdefer allocator.free(probabilities);
        const rho = try allocator.alloc(f64, q);
        errdefer allocator.free(rho);
        const local_start = chunk_start - self.page_start;
        for (0..q) |qi| {
            if (qi % 64 == 0) try cfg.control.check();
            const source_row = self.probabilities[qi * self.page_len + local_start ..][0..chunk_rows];
            @memcpy(probabilities[qi * chunk_rows ..][0..chunk_rows], source_row);
            var sum: f64 = 0;
            for (source_row) |value| sum += value;
            if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
            rho[qi] = sum;
        }
        return .{ .allocator = allocator, .owner = self, .probabilities = probabilities, .rho = rho, .chunk_start = chunk_start, .chunk_rows = chunk_rows, .retained_bytes = retained };
    }

    pub fn deinit(self: *Cache) void {
        self.allocator.free(self.probabilities);
        self.allocator.free(self.original_outputs);
        self.* = undefined;
    }
};

test "training page cache crops exact child probabilities and aliases original outputs" {
    const q: usize = 65;
    var train: [q]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const source: solver.Head = .{ .keys = &.{ -2, -1, 0, 1, 2 }, .values = &.{ 1, 2, 4, 8, 16 }, .bias = &.{ 0.1, -0.2, 0.3, -0.4, 0.5 }, .rows = 5, .dim = 1, .scale = 0.75, .softcap = 1.5 };
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };
    var cache = try Cache.init(std.testing.allocator, source, &train, 1, 4, cfg, 1 << 20);
    defer cache.deinit();
    try std.testing.expectEqual(@as(usize, 2), cache.attention_batches);
    var child = try cache.child(std.testing.allocator, source, &train, 2, 2, cfg, 1 << 20);
    defer child.deinit();
    try std.testing.expectEqual(@intFromPtr(cache.original_outputs.ptr), @intFromPtr(child.originalOutputs().ptr));
    const full = try std.testing.allocator.alloc(f64, q * source.rows);
    defer std.testing.allocator.free(full);
    const outputs = try std.testing.allocator.alloc(f64, q);
    defer std.testing.allocator.free(outputs);
    try solver.attendBatch(std.testing.allocator, source, &train, full, outputs, null, cfg);
    for (0..q) |i| {
        try std.testing.expectEqualSlices(f64, full[i * source.rows + 2 ..][0..2], child.probabilities[i * 2 ..][0..2]);
        try std.testing.expectEqual(full[i * source.rows + 2] + full[i * source.rows + 3], child.rho[i]);
        try std.testing.expectEqual(outputs[i], child.originalOutputs()[i]);
    }
}

test "training page cache cap rejects before backend and cancellation stops after first batch" {
    const Fake = struct {
        var calls: usize = 0;
        var armed = false;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 4096;
        }
        fn check(_: ?*anyopaque) !void {
            if (armed) return error.Cancelled;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, _: []const f64, p: []f64, y: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
            @memset(p, 1 / @as(f64, @floatFromInt(h.rows)));
            if (y) |out| @memset(out, 1);
            armed = true;
        }
    };
    const source: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var train: [65]f64 = @splat(0);
    var marker: u8 = 0;
    var cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .context = &marker, .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .control = .{ .check_fn = Fake.check } };
    const required = try Cache.requiredBytes(2, 65, 2, 1, cfg);
    Fake.calls = 0;
    Fake.armed = false;
    try std.testing.expectError(error.CompactionWorkspaceLimit, Cache.init(std.testing.allocator, source, &train, 0, 2, cfg, required - 1));
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    Fake.calls = 0;
    Fake.armed = false;
    try std.testing.expectError(error.Cancelled, Cache.init(std.testing.allocator, source, &train, 0, 2, cfg, required));
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    cfg.control = .{};
    Fake.armed = false;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, h: solver.Head, qs: []const f64, c: solver.Config, bytes: usize) !void {
            var cache = try Cache.init(a, h, qs, 0, 2, c, bytes);
            cache.deinit();
        }
    }.run, .{ source, &train, cfg, required });
}

test "training page cache 32K E2 parent child and backend fit below 512 MiB" {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, n: usize, m: usize, q: usize, d: usize) !usize {
            return mul(try add(try mul(n, q), try mul(m, d)), @sizeOf(f64));
        }
    };
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace } };
    const parent = try Cache.retainedBytes(10240, page_rows, 512);
    const child = try Cache.childRetainedBytes(10240, 256);
    const construction = try Cache.constructionScratchBytes(32512);
    const backend = try Fake.workspace(null, 32512, 1, 64, 512);
    const required = try Cache.requiredBytes(32512, 10240, page_rows, 512, cfg);
    try std.testing.expectEqual(parent + construction + backend, required);
    try std.testing.expect(parent + child + construction + backend < 512 << 20);
}
