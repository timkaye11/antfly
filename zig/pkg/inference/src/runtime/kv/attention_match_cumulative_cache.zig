// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Production-neutral cumulative whole-head admission for independently fitted chunks.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const Range = struct { start: usize, rows: usize };
pub const Metrics = struct { max_relative_mass_error: f64 = 0, relative_output_rmse: f64 = 0 };

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn baseRetainedBytes(query_rows: usize, dim: usize, range_count: usize) !usize {
    const qd = try mul(query_rows, dim);
    const elements = try add(try mul(query_rows, 2), try mul(qd, 2));
    return add(try mul(elements, @sizeOf(f64)), try mul(range_count, @sizeOf(Range)));
}

pub fn referenceRetainedBytes(query_rows: usize, dim: usize) !usize {
    return mul(try add(query_rows, try mul(query_rows, dim)), @sizeOf(f64));
}

pub fn attendScratchBytes(rows: usize, dim: usize) !usize {
    const elements = try add(try mul(64, rows), try add(try mul(64, dim), 64));
    return mul(elements, @sizeOf(f64));
}

pub const ChunkReference = struct {
    allocator: std.mem.Allocator,
    owner: *Cache,
    token: usize,
    chunk_index: usize,
    base_version: usize,
    mass: []f64,
    output: []f64,
    retained_bytes: usize,

    pub fn deinit(self: *@This()) void {
        std.debug.assert(self.owner.active_trial_token == null);
        if (self.owner.active_reference_token == self.token) self.owner.active_reference_token = null;
        self.allocator.free(self.mass);
        self.allocator.free(self.output);
        self.* = undefined;
    }
};

pub const Trial = struct {
    allocator: std.mem.Allocator,
    owner: *Cache,
    token: usize,
    chunk_index: usize,
    base_version: usize,
    delta_mass: []f64,
    delta_numerator: []f64,
    metrics: Metrics,
    committed: bool = false,
    pub fn deinit(self: *@This()) void {
        if (self.owner.active_trial_token == self.token) self.owner.active_trial_token = null;
        self.allocator.free(self.delta_mass);
        self.allocator.free(self.delta_numerator);
        self.* = undefined;
    }
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    original: solver.Head,
    queries: []const f64,
    ranges: []Range,
    original_mass: []f64,
    original_output: []f64,
    current_mass: []f64,
    current_numerator: []f64,
    control: solver.Control,
    retained_bytes: usize,
    max_bytes: usize,
    version: usize = 0,
    next_token: usize = 1,
    active_reference_token: ?usize = null,
    active_trial_token: ?usize = null,

    pub fn init(allocator: std.mem.Allocator, original: solver.Head, queries: []const f64, ranges: []const Range, cfg: solver.Config, max_bytes: usize) !Cache {
        try original.validate();
        if (cfg.backend != null or queries.len == 0 or queries.len % original.dim != 0 or ranges.len == 0) return error.InvalidFitConfig;
        var cursor: usize = 0;
        for (ranges) |r| {
            if (r.rows == 0 or r.start < cursor or r.start > original.rows or r.rows > original.rows - r.start) return error.InvalidFitConfig;
            cursor = r.start + r.rows;
        }
        const q = queries.len / original.dim;
        const qd = try mul(q, original.dim);
        const retained_bytes = try baseRetainedBytes(q, original.dim, ranges.len);
        const required = try add(retained_bytes, try attendScratchBytes(original.rows, original.dim));
        if (required > max_bytes or required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        const owned_ranges = try allocator.dupe(Range, ranges);
        errdefer allocator.free(owned_ranges);
        const original_mass = try allocator.alloc(f64, q);
        errdefer allocator.free(original_mass);
        const original_output = try allocator.alloc(f64, qd);
        errdefer allocator.free(original_output);
        const current_mass = try allocator.alloc(f64, q);
        errdefer allocator.free(current_mass);
        const current_numerator = try allocator.alloc(f64, qd);
        errdefer allocator.free(current_numerator);
        const probabilities = try allocator.alloc(f64, try mul(64, original.rows));
        defer allocator.free(probabilities);
        const outputs = try allocator.alloc(f64, try mul(64, original.dim));
        defer allocator.free(outputs);
        const masses = try allocator.alloc(f64, 64);
        defer allocator.free(masses);
        var offset: usize = 0;
        while (offset < q) {
            try cfg.control.check();
            const take = @min(@as(usize, 64), q - offset);
            const qs = queries[offset * original.dim ..][0 .. take * original.dim];
            try solver.attendBatch(allocator, original, qs, probabilities[0 .. take * original.rows], outputs[0 .. take * original.dim], masses[0..take], cfg);
            @memcpy(original_mass[offset..][0..take], masses[0..take]);
            @memcpy(original_output[offset * original.dim ..][0 .. take * original.dim], outputs[0 .. take * original.dim]);
            offset += take;
        }
        @memset(current_mass, 1);
        @memcpy(current_numerator, original_output);
        return .{ .allocator = allocator, .original = original, .queries = queries, .ranges = owned_ranges, .original_mass = original_mass, .original_output = original_output, .current_mass = current_mass, .current_numerator = current_numerator, .control = cfg.control, .retained_bytes = retained_bytes, .max_bytes = @min(max_bytes, cfg.max_workspace_bytes) };
    }

    pub fn deinit(self: *@This()) void {
        std.debug.assert(self.active_reference_token == null and self.active_trial_token == null);
        self.allocator.free(self.ranges);
        self.allocator.free(self.original_mass);
        self.allocator.free(self.original_output);
        self.allocator.free(self.current_mass);
        self.allocator.free(self.current_numerator);
        self.* = undefined;
    }

    fn validateReference(self: *const Cache, reference: *const ChunkReference) !void {
        if (reference.owner != self or self.active_reference_token != reference.token or reference.chunk_index >= self.ranges.len or reference.base_version != self.version or reference.mass.len != self.original_mass.len or reference.output.len != self.original_output.len) return error.StaleChunkReference;
    }

    pub fn beginReference(self: *Cache, chunk_index: usize, external_live_bytes: usize) !ChunkReference {
        if (chunk_index >= self.ranges.len) return error.InvalidFitConfig;
        if (self.active_reference_token != null or self.active_trial_token != null) return error.ConcurrentChunkReference;
        const token = self.next_token;
        const next_token = std.math.add(usize, token, 1) catch return error.InvalidAttentionShape;
        try self.control.check();
        const range = self.ranges[chunk_index];
        const q = self.original_mass.len;
        const d = self.original.dim;
        const qd = try mul(q, d);
        const reference_bytes = try referenceRetainedBytes(q, d);
        const peak = try add(try add(try add(self.retained_bytes, external_live_bytes), reference_bytes), try attendScratchBytes(range.rows, d));
        if (peak > self.max_bytes) return error.CompactionWorkspaceLimit;
        const mass = try self.allocator.alloc(f64, q);
        errdefer self.allocator.free(mass);
        const output = try self.allocator.alloc(f64, qd);
        errdefer self.allocator.free(output);
        const p = try self.allocator.alloc(f64, try mul(64, range.rows));
        defer self.allocator.free(p);
        const y = try self.allocator.alloc(f64, try mul(64, d));
        defer self.allocator.free(y);
        const z = try self.allocator.alloc(f64, 64);
        defer self.allocator.free(z);
        const chunk: solver.Head = .{ .keys = self.original.keys[range.start * d ..][0 .. range.rows * d], .values = self.original.values[range.start * d ..][0 .. range.rows * d], .bias = if (self.original.bias) |b| b[range.start..][0..range.rows] else null, .rows = range.rows, .dim = d, .scale = self.original.scale, .softcap = self.original.softcap };
        var offset: usize = 0;
        while (offset < q) {
            try self.control.check();
            const take = @min(@as(usize, 64), q - offset);
            const qs = self.queries[offset * d ..][0 .. take * d];
            try solver.attendBatch(self.allocator, chunk, qs, p[0 .. take * range.rows], y[0 .. take * d], z[0..take], .{ .target_rows = 1, .control = self.control });
            @memcpy(mass[offset..][0..take], z[0..take]);
            @memcpy(output[offset * d ..][0 .. take * d], y[0 .. take * d]);
            offset += take;
        }
        self.next_token = next_token;
        self.active_reference_token = token;
        return .{ .allocator = self.allocator, .owner = self, .token = token, .chunk_index = chunk_index, .base_version = self.version, .mass = mass, .output = output, .retained_bytes = reference_bytes };
    }

    pub fn remainingWorkspace(self: *const Cache, reference: *const ChunkReference, external_live_bytes: usize) !usize {
        try self.validateReference(reference);
        const live = try add(try add(self.retained_bytes, reference.retained_bytes), external_live_bytes);
        if (live > self.max_bytes) return error.CompactionWorkspaceLimit;
        return self.max_bytes - live;
    }

    pub fn trial(self: *Cache, reference: *const ChunkReference, candidate: solver.Head, external_live_bytes: usize) !Trial {
        try self.validateReference(reference);
        if (self.active_trial_token != null) return error.ConcurrentChunkTrial;
        const token = self.next_token;
        const next_token = std.math.add(usize, token, 1) catch return error.InvalidAttentionShape;
        if (candidate.dim != self.original.dim or candidate.rows == 0 or candidate.scale != self.original.scale or candidate.softcap != self.original.softcap) return error.InvalidFitConfig;
        try candidate.validate();
        try self.control.check();
        const q = self.original_mass.len;
        const d = self.original.dim;
        const qd = try mul(q, d);
        var trial_elements = try add(q, qd);
        trial_elements = try add(trial_elements, try mul(64, candidate.rows));
        trial_elements = try add(trial_elements, try mul(64, d));
        trial_elements = try add(trial_elements, 64);
        const trial_bytes = try mul(trial_elements, @sizeOf(f64));
        if (try add(try add(try add(self.retained_bytes, reference.retained_bytes), external_live_bytes), trial_bytes) > self.max_bytes) return error.CompactionWorkspaceLimit;
        const delta_mass = try self.allocator.alloc(f64, q);
        errdefer self.allocator.free(delta_mass);
        const delta_numerator = try self.allocator.alloc(f64, qd);
        errdefer self.allocator.free(delta_numerator);
        const p = try self.allocator.alloc(f64, try mul(64, candidate.rows));
        defer self.allocator.free(p);
        const y = try self.allocator.alloc(f64, try mul(64, d));
        defer self.allocator.free(y);
        const z = try self.allocator.alloc(f64, 64);
        defer self.allocator.free(z);
        var trial_metrics: Metrics = .{};
        var error_norm: f64 = 0;
        var target_norm: f64 = 0;
        var offset: usize = 0;
        while (offset < q) {
            try self.control.check();
            const take = @min(@as(usize, 64), q - offset);
            const qs = self.queries[offset * d ..][0 .. take * d];
            try solver.attendBatch(self.allocator, candidate, qs, p[0 .. take * candidate.rows], y[0 .. take * d], z[0..take], .{ .target_rows = 1, .control = self.control });
            for (0..take) |bi| {
                const qi = offset + bi;
                const rho = @exp(reference.mass[qi] - self.original_mass[qi]);
                const sigma = @exp(z[bi] - self.original_mass[qi]);
                const dm = sigma - rho;
                delta_mass[qi] = dm;
                const trial_d = self.current_mass[qi] + dm;
                if (!std.math.isFinite(trial_d) or trial_d <= 0) return error.NonFiniteAttention;
                trial_metrics.max_relative_mass_error = @max(trial_metrics.max_relative_mass_error, @abs(trial_d - 1));
                for (0..d) |j| {
                    const old = rho * reference.output[qi * d + j];
                    const fresh = sigma * y[bi * d + j];
                    const dn = fresh - old;
                    delta_numerator[qi * d + j] = dn;
                    const out = (self.current_numerator[qi * d + j] + dn) / trial_d;
                    error_norm = std.math.hypot(error_norm, out - self.original_output[qi * d + j]);
                    target_norm = std.math.hypot(target_norm, self.original_output[qi * d + j]);
                }
            }
            offset += take;
        }
        trial_metrics.relative_output_rmse = error_norm / @max(target_norm, 1e-12);
        if (!std.math.isFinite(trial_metrics.max_relative_mass_error) or !std.math.isFinite(trial_metrics.relative_output_rmse)) return error.NonFiniteAttention;
        self.next_token = next_token;
        self.active_trial_token = token;
        return .{ .allocator = self.allocator, .owner = self, .token = token, .chunk_index = reference.chunk_index, .base_version = reference.base_version, .delta_mass = delta_mass, .delta_numerator = delta_numerator, .metrics = trial_metrics };
    }

    pub fn commit(self: *Cache, proposal: *Trial) !void {
        if (proposal.delta_mass.len != self.current_mass.len or proposal.delta_numerator.len != self.current_numerator.len) return error.InvalidAttentionShape;
        if (proposal.owner != self or self.active_trial_token != proposal.token or proposal.committed or proposal.base_version != self.version or proposal.chunk_index >= self.ranges.len) return error.StaleChunkReference;
        try self.control.check();
        const next_version = std.math.add(usize, self.version, 1) catch return error.InvalidAttentionShape;
        for (self.current_mass, proposal.delta_mass) |*x, d| x.* += d;
        for (self.current_numerator, proposal.delta_numerator) |*x, d| x.* += d;
        self.version = next_version;
        proposal.committed = true;
    }

    pub fn metrics(self: *const Cache) !Metrics {
        var result: Metrics = .{};
        var error_norm: f64 = 0;
        var target_norm: f64 = 0;
        const d = self.original.dim;
        for (self.current_mass, 0..) |mass, query| {
            if (!std.math.isFinite(mass) or mass <= 0) return error.NonFiniteAttention;
            result.max_relative_mass_error = @max(result.max_relative_mass_error, @abs(mass - 1));
            for (0..d) |column| {
                const output = self.current_numerator[query * d + column] / mass;
                const target = self.original_output[query * d + column];
                error_norm = std.math.hypot(error_norm, output - target);
                target_norm = std.math.hypot(target_norm, target);
            }
        }
        result.relative_output_rmse = error_norm / @max(target_norm, 1e-12);
        if (!std.math.isFinite(result.max_relative_mass_error) or !std.math.isFinite(result.relative_output_rmse)) return error.NonFiniteAttention;
        return result;
    }
};

test "cumulative runtime chunk admission cached composition matches direct full head and rollback is ordered" {
    const original: solver.Head = .{ .keys = &.{ 0, 1, 2 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const queries = [_]f64{ 0.2, 0.8 };
    const ranges = [_]Range{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 2 } };
    var cache = try Cache.init(std.testing.allocator, original, &queries, &ranges, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer cache.deinit();
    var first_reference = try cache.beginReference(0, 0);
    const token_after_reference = cache.next_token;
    try std.testing.expectError(error.ConcurrentChunkReference, cache.beginReference(1, 0));
    try std.testing.expectEqual(token_after_reference, cache.next_token);
    var foreign_cache = try Cache.init(std.testing.allocator, original, &queries, &ranges, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer foreign_cache.deinit();
    const bad: solver.Head = .{ .keys = &.{4}, .values = &.{20}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    var rejected = try cache.trial(&first_reference, bad, 0);
    const token_after_trial = cache.next_token;
    try std.testing.expectError(error.ConcurrentChunkTrial, cache.trial(&first_reference, bad, 0));
    try std.testing.expectEqual(token_after_trial, cache.next_token);
    try std.testing.expect(rejected.metrics.max_relative_mass_error > 0.05);
    try std.testing.expectEqual(@as(f64, 1), cache.current_mass[0]);
    rejected.deinit();
    const first: solver.Head = .{ .keys = &.{0.01}, .values = &.{2.001}, .bias = &.{0.005}, .rows = 1, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.StaleChunkReference, foreign_cache.trial(&first_reference, first, 0));
    var first_trial = try cache.trial(&first_reference, first, 0);
    try cache.commit(&first_trial);
    try std.testing.expectError(error.StaleChunkReference, cache.commit(&first_trial));
    first_trial.deinit();
    try std.testing.expectError(error.StaleChunkReference, cache.trial(&first_reference, first, 0));
    first_reference.deinit();
    var second_reference = try cache.beginReference(1, 0);
    defer second_reference.deinit();
    const second: solver.Head = .{ .keys = &.{1.5}, .values = &.{6}, .bias = &.{@log(@as(f64, 2))}, .rows = 1, .dim = 1, .scale = 1 };
    var second_trial = try cache.trial(&second_reference, second, 0);
    defer second_trial.deinit();
    try cache.commit(&second_trial);
    const composed: solver.Head = .{ .keys = &.{ 0.01, 1.5 }, .values = &.{ 2.001, 6 }, .bias = &.{ 0.005, @log(@as(f64, 2)) }, .rows = 2, .dim = 1, .scale = 1 };
    const direct = try solver.validationMetrics(std.testing.allocator, original, composed, &queries, .{ .target_rows = 2 });
    const cached = try cache.metrics();
    try std.testing.expect(cached.max_relative_mass_error > 0);
    try std.testing.expect(cached.relative_output_rmse > 0);
    try std.testing.expectApproxEqAbs(direct.max_relative_mass_error, cached.max_relative_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(direct.relative_output_rmse, cached.relative_output_rmse, 1e-12);
}

fn allocationFailure(a: std.mem.Allocator) !void {
    const h: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    var c = try Cache.init(a, h, &.{0.5}, &.{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 1 } }, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer c.deinit();
    var reference = try c.beginReference(0, 0);
    defer reference.deinit();
    var t = try c.trial(&reference, .{ .keys = &.{0}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 }, 0);
    t.deinit();
}
test "cumulative runtime chunk admission enforces cap cancellation and allocation unwind" {
    const h: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, Cache.init(std.testing.allocator, h, &.{0.5}, &.{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 1 } }, .{ .target_rows = 1, .max_workspace_bytes = 1 }, 1));
    const C = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, Cache.init(std.testing.allocator, h, &.{0.5}, &.{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 1 } }, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = C.check } }, 1 << 20));
    var c = try Cache.init(std.testing.allocator, h, &.{0.5}, &.{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 1 } }, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer c.deinit();
    const reference_peak = try add(try add(c.retained_bytes, try referenceRetainedBytes(1, 1)), try attendScratchBytes(1, 1));
    const full_cap = c.max_bytes;
    c.max_bytes = reference_peak - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, c.beginReference(0, 0));
    try std.testing.expectEqual(@as(usize, 0), c.version);
    try std.testing.expectEqual(@as(f64, 1), c.current_mass[0]);
    c.max_bytes = full_cap;
    var reference = try c.beginReference(0, 0);
    defer reference.deinit();
    const saved = c.max_bytes;
    c.max_bytes = c.retained_bytes;
    try std.testing.expectError(error.CompactionWorkspaceLimit, c.trial(&reference, .{ .keys = &.{0}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 }, 0));
    c.max_bytes = saved;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "lazy cumulative reference is bit equal to direct eager chunk batches including tail" {
    const d = 2;
    const original: solver.Head = .{ .keys = &.{ 0, 1, 1, 0, 2, 2 }, .values = &.{ 2, 3, 4, 5, 6, 7 }, .bias = &.{ 0, 0.25, -0.5 }, .rows = 3, .dim = d, .scale = 0.5 };
    var queries: [65 * d]f64 = undefined;
    for (&queries, 0..) |*value, index| value.* = @as(f64, @floatFromInt(index % 17)) / 17;
    const ranges = [_]Range{.{ .start = 1, .rows = 2 }};
    var cache = try Cache.init(std.testing.allocator, original, &queries, &ranges, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer cache.deinit();
    var reference = try cache.beginReference(0, 0);
    defer reference.deinit();
    var expected_mass: [65]f64 = undefined;
    var expected_output: [65 * d]f64 = undefined;
    var p: [64 * 2]f64 = undefined;
    var y: [64 * d]f64 = undefined;
    var z: [64]f64 = undefined;
    const chunk: solver.Head = .{ .keys = original.keys[2..], .values = original.values[2..], .bias = original.bias.?[1..], .rows = 2, .dim = d, .scale = original.scale };
    var offset: usize = 0;
    while (offset < 65) {
        const take: usize = @min(@as(usize, 64), 65 - offset);
        try solver.attendBatch(std.testing.allocator, chunk, queries[offset * d ..][0 .. take * d], p[0 .. take * 2], y[0 .. take * d], z[0..take], .{ .target_rows = 1 });
        @memcpy(expected_mass[offset..][0..take], z[0..take]);
        @memcpy(expected_output[offset * d ..][0 .. take * d], y[0 .. take * d]);
        offset += take;
    }
    try std.testing.expectEqualSlices(f64, &expected_mass, reference.mass);
    try std.testing.expectEqualSlices(f64, &expected_output, reference.output);
    const candidate: solver.Head = .{ .keys = original.keys[2..], .values = original.values[2..], .bias = original.bias.?[1..], .rows = 2, .dim = d, .scale = original.scale };
    var lazy_trial = try cache.trial(&reference, candidate, 0);
    const lazy_metrics = lazy_trial.metrics;
    const lazy_delta_mass = try std.testing.allocator.dupe(f64, lazy_trial.delta_mass);
    defer std.testing.allocator.free(lazy_delta_mass);
    const lazy_delta_numerator = try std.testing.allocator.dupe(f64, lazy_trial.delta_numerator);
    defer std.testing.allocator.free(lazy_delta_numerator);
    lazy_trial.deinit();
    const eager_reference: ChunkReference = .{ .allocator = std.testing.allocator, .owner = &cache, .token = reference.token, .chunk_index = 0, .base_version = cache.version, .mass = &expected_mass, .output = &expected_output, .retained_bytes = try referenceRetainedBytes(65, d) };
    var eager_trial = try cache.trial(&eager_reference, candidate, 0);
    defer eager_trial.deinit();
    try std.testing.expectEqual(lazy_metrics, eager_trial.metrics);
    try std.testing.expectEqualSlices(f64, lazy_delta_mass, eager_trial.delta_mass);
    try std.testing.expectEqualSlices(f64, lazy_delta_numerator, eager_trial.delta_numerator);
}

test "lazy cumulative cache exact 32K retained and reference byte projections" {
    try std.testing.expectEqual(@as(usize, 21_014_384), try baseRetainedBytes(2560, 512, 119));
    try std.testing.expectEqual(@as(usize, 10_506_240), try referenceRetainedBytes(2560, 512));
    try std.testing.expectEqual(@as(usize, 8_406_896), try baseRetainedBytes(1024, 512, 119));
    try std.testing.expectEqual(@as(usize, 4_202_496), try referenceRetainedBytes(1024, 512));
    try std.testing.expectEqual(@as(usize, 16_908_800), try attendScratchBytes(32512, 512));
}

test "lazy cumulative reference cancellation leaves committed composite and version unchanged" {
    const Context = struct {
        checks: usize = 0,
        cancel_at: usize,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == self.cancel_at) return error.Cancelled;
        }
    };
    var context: Context = .{ .cancel_at = std.math.maxInt(usize) };
    const control: solver.Control = .{ .context = &context, .check_fn = Context.check };
    const original: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .rows = 2, .dim = 1, .scale = 1 };
    var cache = try Cache.init(std.testing.allocator, original, &.{0.5}, &.{.{ .start = 0, .rows = 1 }}, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .control = control }, 1 << 20);
    defer cache.deinit();
    context.cancel_at = context.checks + 2;
    try std.testing.expectEqual(@as(usize, 0), cache.version);
    try std.testing.expectEqual(@as(f64, 1), cache.current_mass[0]);
    try std.testing.expectError(error.Cancelled, cache.beginReference(0, 0));
    try std.testing.expectEqual(@as(usize, 0), cache.version);
    try std.testing.expectEqual(@as(usize, 1), cache.next_token);
    try std.testing.expect(cache.active_reference_token == null and cache.active_trial_token == null);
    try std.testing.expectEqual(@as(f64, 1), cache.current_mass[0]);
}
