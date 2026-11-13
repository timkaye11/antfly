// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only full-original normalization for one residual chunk.
const std = @import("std");
const solver = @import("attention_match.zig");
const training_cache = @import("attention_match_global_training_cache.zig");

pub const Result = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    bias: []f64,
    weights: []f64,
    rho_min: f64,
    rho_max: f64,
    rho_mean: f64,
    nnls_sweeps: usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

pub fn fitOriginalContextChunk(
    allocator: std.mem.Allocator,
    original: solver.Head,
    chunk_start: usize,
    chunk_rows: usize,
    exact_local: []const usize,
    selected_residual_local: []const usize,
    train: []const f64,
    cfg: solver.Config,
) !Result {
    return fitOriginalContextChunkImpl(allocator, original, chunk_start, chunk_rows, exact_local, selected_residual_local, train, cfg, null, null);
}

pub fn fitOriginalContextChunkWarm(allocator: std.mem.Allocator, original: solver.Head, chunk_start: usize, chunk_rows: usize, selected_residual_local: []const usize, initial_weights: []const f64, train: []const f64, cfg: solver.Config) !Result {
    return fitOriginalContextChunkImpl(allocator, original, chunk_start, chunk_rows, &.{}, selected_residual_local, train, cfg, null, initial_weights);
}

test "global residual cached and uncached fits are bit identical" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, @log(@as(f64, 2)) };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    var uncached = try fitOriginalContextChunk(std.testing.allocator, head, 0, 3, &.{0}, &.{1}, &train, cfg);
    defer uncached.deinit();
    var cache = try training_cache.Cache.init(std.testing.allocator, head, &train, cfg, training_cache.maximum_combined_bytes);
    defer cache.deinit();
    var cached = try fitOriginalContextChunkCached(std.testing.allocator, &cache, head, 0, 3, &.{0}, &.{1}, &train, cfg);
    defer cached.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(uncached.weights), std.mem.sliceAsBytes(cached.weights));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(uncached.values), std.mem.sliceAsBytes(cached.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(uncached.bias), std.mem.sliceAsBytes(cached.bias));
}

pub fn fitOriginalContextChunkCached(allocator: std.mem.Allocator, cache: *const training_cache.Cache, original: solver.Head, chunk_start: usize, chunk_rows: usize, exact_local: []const usize, selected_residual_local: []const usize, train: []const f64, cfg: solver.Config) !Result {
    try cache.validate(original, train, cfg);
    return fitOriginalContextChunkImpl(allocator, original, chunk_start, chunk_rows, exact_local, selected_residual_local, train, cfg, cache, null);
}

fn fitOriginalContextChunkImpl(
    allocator: std.mem.Allocator,
    original: solver.Head,
    chunk_start: usize,
    chunk_rows: usize,
    exact_local: []const usize,
    selected_residual_local: []const usize,
    train: []const f64,
    cfg: solver.Config,
    cache: ?*const training_cache.Cache,
    initial_weights: ?[]const f64,
) !Result {
    if (comptime !@import("builtin").is_test) @compileError("global residual fit is diagnostic-test-only");
    try original.validate();
    try solver.validateConfigForTest(cfg, original.rows);
    if (chunk_rows == 0 or chunk_start > original.rows or chunk_rows > original.rows - chunk_start or
        selected_residual_local.len == 0 or train.len == 0 or train.len % original.dim != 0) return error.InvalidFitConfig;
    for (exact_local, 0..) |idx, i| if (idx >= chunk_rows or std.mem.indexOfScalar(usize, exact_local[0..i], idx) != null) return error.InvalidFitConfig;
    for (selected_residual_local, 0..) |idx, i| if (idx >= chunk_rows or std.mem.indexOfScalar(usize, selected_residual_local[0..i], idx) != null or std.mem.indexOfScalar(usize, exact_local, idx) != null) return error.InvalidFitConfig;
    const q = train.len / original.dim;
    const m = selected_residual_local.len;
    if (initial_weights) |warm| {
        if (exact_local.len != 0 or warm.len != m) return error.InvalidFitConfig;
        for (warm) |weight| if (!std.math.isFinite(weight) or weight < 0) return error.InvalidFitConfig;
    }
    const qm = try std.math.mul(usize, q, m);
    const qd = try std.math.mul(usize, q, original.dim);
    const batch_probabilities = if (cache == null) try std.math.mul(usize, 64, original.rows) else 0;
    const batch_outputs = if (cache == null) try std.math.mul(usize, 64, original.dim) else 0;
    var retained_elements = try std.math.mul(usize, qm, 2);
    retained_elements = try std.math.add(usize, retained_elements, try std.math.mul(usize, qd, 2));
    retained_elements = try std.math.add(usize, retained_elements, batch_probabilities);
    retained_elements = try std.math.add(usize, retained_elements, batch_outputs);
    retained_elements = try std.math.add(usize, retained_elements, try std.math.mul(usize, q, 2));
    retained_elements = try std.math.add(usize, retained_elements, try std.math.mul(usize, m, 4));
    retained_elements = try std.math.add(usize, retained_elements, try std.math.mul(usize, m, original.dim));
    const retained_bytes = try std.math.mul(usize, retained_elements, @sizeOf(f64));
    const cpu_nested = try solver.workspaceBytes(m, m, q, original.dim);
    const backend_peak = if (cfg.backend) |backend| @max(
        if (cache == null) try backend.workspace_fn(backend.context, original.rows, 1, @min(q, 64), original.dim) else 0,
        try backend.workspace_fn(backend.context, m, m, q, original.dim),
    ) else 0;
    const required_workspace = try std.math.add(usize, retained_bytes, @max(cpu_nested, backend_peak));
    if (required_workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const a = try allocator.alloc(f64, qm);
    defer allocator.free(a);
    const rho = try allocator.alloc(f64, q);
    defer allocator.free(rho);
    const target = try allocator.alloc(f64, qd);
    defer allocator.free(target);
    const original_y = try allocator.alloc(f64, qd);
    defer allocator.free(original_y);
    const probabilities = try allocator.alloc(f64, batch_probabilities);
    defer allocator.free(probabilities);
    const outputs = try allocator.alloc(f64, batch_outputs);
    defer allocator.free(outputs);
    var rho_min = std.math.inf(f64);
    var rho_max: f64 = 0;
    var rho_sum: f64 = 0;
    var offset: usize = 0;
    while (offset < q) {
        try cfg.control.check();
        const take = @min(@as(usize, 64), q - offset);
        const qs = train[offset * original.dim ..][0 .. take * original.dim];
        if (cache == null) try solver.attendBatch(allocator, original, qs, probabilities[0 .. take * original.rows], outputs[0 .. take * original.dim], null, cfg);
        for (0..take) |bi| {
            const absolute = offset + bi;
            const row = if (cache) |stored| stored.probabilities[absolute * original.rows ..][0..original.rows] else probabilities[bi * original.rows ..][0..original.rows];
            var residual_mass: f64 = 0;
            const out = target[absolute * original.dim ..][0..original.dim];
            const original_output = if (cache) |stored| stored.outputs[absolute * original.dim ..][0..original.dim] else outputs[bi * original.dim ..][0..original.dim];
            @memcpy(out, original_output);
            @memcpy(original_y[(offset + bi) * original.dim ..][0..original.dim], out);
            for (0..chunk_rows) |local| if (std.mem.indexOfScalar(usize, exact_local, local) == null) {
                const p = row[chunk_start + local];
                residual_mass += p;
                for (out, original.values[(chunk_start + local) * original.dim ..][0..original.dim]) |*y, v| y.* -= p * v;
            };
            rho[offset + bi] = residual_mass;
            if (!std.math.isFinite(residual_mass) or residual_mass < 0) return error.CompactionFitRejected;
            rho_min = @min(rho_min, residual_mass);
            rho_max = @max(rho_max, residual_mass);
            rho_sum += residual_mass;
            for (selected_residual_local, 0..) |local, j| a[(offset + bi) * m + j] = row[chunk_start + local];
        }
        offset += take;
    }
    const weights = try allocator.alloc(f64, m);
    errdefer allocator.free(weights);
    if (initial_weights) |warm| @memcpy(weights, warm) else @memset(weights, 0);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, m);
    defer allocator.free(norms);
    const sweeps = try solver.fitMassActiveTargetForTest(allocator, a, rho, q, m, weights, residual, norms, cfg, initial_weights != null);
    for (weights) |*weight| weight.* = @max(weight.*, cfg.min_weight);
    const basis = try allocator.alloc(f64, qm);
    defer allocator.free(basis);
    for (0..q) |i| {
        var d: f64 = 1 - rho[i];
        for (0..m) |j| {
            const w = weights[j];
            basis[i * m + j] = a[i * m + j] * w;
            d += basis[i * m + j];
        }
        for (target[i * original.dim ..][0..original.dim], original_y[i * original.dim ..][0..original.dim]) |*fixed, y| fixed.* = d * y - fixed.*;
    }
    const values = try allocator.alloc(f64, m * original.dim);
    errdefer allocator.free(values);
    try solver.fitValuesWithBackendForTest(allocator, basis, target, q, m, original.dim, cfg.ridge, cfg, values);
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    for (selected_residual_local, weights, bias) |local, w, *b| b.* = (if (original.bias) |source| source[chunk_start + local] else 0) + @log(@max(w, cfg.min_weight));
    return .{ .allocator = allocator, .values = values, .bias = bias, .weights = weights, .rho_min = rho_min, .rho_max = rho_max, .rho_mean = rho_sum / @as(f64, @floatFromInt(q)), .nnls_sweeps = sweeps };
}

test "global residual target uses D times output when D is not one" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    var result = try fitOriginalContextChunk(std.testing.allocator, head, 0, 3, &.{0}, &.{1}, &.{ 0, @log(@as(f64, 2)) }, cfg);
    defer result.deinit();
    const weight: f64 = 59.0 / 25.0;
    const b0: f64 = 59.0 / 75.0;
    const b1: f64 = 59.0 / 100.0;
    const c0: f64 = 114.0 / 25.0;
    const c1: f64 = 103.0 / 25.0;
    const expected_value = (b0 * c0 + b1 * c1) / (b0 * b0 + b1 * b1 + cfg.ridge);
    try std.testing.expectApproxEqAbs(weight, result.weights[0], 1e-10);
    try std.testing.expectApproxEqAbs(expected_value, result.values[0], 1e-9);
    try std.testing.expectApproxEqAbs(@log(weight), result.bias[0], 1e-10);
}

test "global residual accepts zero target and enforces workspace before allocation" {
    const head: solver.Head = .{ .keys = &.{ 1000, -1000 }, .values = &.{ 2, 8 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var result = try fitOriginalContextChunk(std.testing.allocator, head, 0, 2, &.{0}, &.{1}, &.{1}, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set });
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 0), result.rho_min);
    try std.testing.expectEqual(@as(f64, 0), result.rho_max);
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitOriginalContextChunk(std.testing.allocator, head, 0, 2, &.{0}, &.{1}, &.{1}, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 }));
}

test "global residual backend workspace uses full source and rejects before attend" {
    const Recording = struct {
        calls: usize = 0,
        args: [2][4]usize = undefined,
        attend_calls: usize = 0,

        fn workspace(raw: ?*anyopaque, n: usize, m: usize, q: usize, d: usize) !usize {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.calls >= self.args.len) return error.InvalidAttentionShape;
            self.args[self.calls] = .{ n, m, q, d };
            self.calls += 1;
            return 1024 * 1024;
        }

        fn attend(raw: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.attend_calls += 1;
            return error.TestUnexpectedResult;
        }
    };
    var recording: Recording = .{};
    const backend: solver.Backend = .{ .context = &recording, .workspace_fn = Recording.workspace, .attend_fn = Recording.attend };
    const head: solver.Head = .{
        .keys = &.{ 0, 0, 0, 0, 0, 0 },
        .values = &.{ 1, 2, 3, 4, 5, 6 },
        .rows = 6,
        .dim = 1,
        .scale = 1,
    };
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitOriginalContextChunk(
        std.testing.allocator,
        head,
        1,
        4,
        &.{0},
        &.{ 1, 2 },
        &.{ 0, 1, 2 },
        .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend, .max_workspace_bytes = 1024 },
    ));
    try std.testing.expectEqual(@as(usize, 2), recording.calls);
    try std.testing.expectEqual([4]usize{ 6, 1, 3, 1 }, recording.args[0]);
    try std.testing.expectEqual([4]usize{ 2, 2, 3, 1 }, recording.args[1]);
    try std.testing.expectEqual(@as(usize, 0), recording.attend_calls);
}

test "global residual fit unwinds every allocation failure" {
    const Case = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const head: solver.Head = .{
                .keys = &.{ 0, 0, 1 },
                .values = &.{ 2, 4, 8 },
                .bias = &.{ 0, 0, 0 },
                .rows = 3,
                .dim = 1,
                .scale = 1,
            };
            var result = try fitOriginalContextChunk(allocator, head, 0, 3, &.{0}, &.{1}, &.{ 0, @log(@as(f64, 2)) }, .{
                .target_rows = 1,
                .selection = .residual_omp,
                .mass_solver = .active_set,
            });
            defer result.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}

test "global residual cancellation unwinds owned scratch" {
    const Stop = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls >= 3) return error.Cancelled;
        }
    };
    var stop: Stop = .{};
    const head: solver.Head = .{
        .keys = &.{ 0, 0, 1 },
        .values = &.{ 2, 4, 8 },
        .bias = &.{ 0, 0, 0 },
        .rows = 3,
        .dim = 1,
        .scale = 1,
    };
    try std.testing.expectError(error.Cancelled, fitOriginalContextChunk(
        std.testing.allocator,
        head,
        0,
        3,
        &.{0},
        &.{1},
        &.{ 0, @log(@as(f64, 2)), 1 },
        .{
            .target_rows = 1,
            .selection = .residual_omp,
            .mass_solver = .active_set,
            .control = .{ .context = &stop, .check_fn = Stop.check },
        },
    ));
    try std.testing.expectEqual(@as(usize, 3), stop.calls);
}

test "global residual cached path preserves cancellation checks" {
    const Stop = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 8 }, .rows = 2, .dim = 1, .scale = 1 };
    const train = [_]f64{0};
    const base: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    var cache = try training_cache.Cache.init(std.testing.allocator, head, &train, base, training_cache.maximum_combined_bytes);
    defer cache.deinit();
    var cancelled = base;
    cancelled.control = .{ .check_fn = Stop.check };
    try std.testing.expectError(error.Cancelled, fitOriginalContextChunkCached(std.testing.allocator, &cache, head, 0, 2, &.{0}, &.{1}, &train, cancelled));
}
