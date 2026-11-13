// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only cumulative whole-head admission for independently fitted chunks.
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

pub const Trial = struct {
    allocator: std.mem.Allocator,
    delta_mass: []f64,
    delta_numerator: []f64,
    metrics: Metrics,
    pub fn deinit(self: *@This()) void {
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
    chunk_mass: []f64,
    chunk_output: []f64,
    current_mass: []f64,
    current_numerator: []f64,
    control: solver.Control,
    retained_bytes: usize,
    max_bytes: usize,

    pub fn init(allocator: std.mem.Allocator, original: solver.Head, queries: []const f64, ranges: []const Range, cfg: solver.Config, max_bytes: usize) !Cache {
        if (comptime !@import("builtin").is_test) @compileError("chunk admission is diagnostic-test-only");
        try original.validate();
        if (cfg.backend != null or queries.len == 0 or queries.len % original.dim != 0 or ranges.len == 0) return error.InvalidFitConfig;
        var cursor: usize = 0;
        for (ranges) |r| {
            if (r.rows == 0 or r.start < cursor or r.start > original.rows or r.rows > original.rows - r.start) return error.InvalidFitConfig;
            cursor = r.start + r.rows;
        }
        const q = queries.len / original.dim;
        const qd = try mul(q, original.dim);
        const qr = try mul(q, ranges.len);
        const rqd = try mul(ranges.len, qd);
        var retained = try add(try mul(q, 2), try mul(qd, 2));
        retained = try add(retained, qr);
        retained = try add(retained, rqd);
        const retained_bytes = try add(try mul(retained, @sizeOf(f64)), try mul(ranges.len, @sizeOf(Range)));
        const scratch = try add(try mul(64, original.rows), try add(try mul(64, original.dim), 64));
        const required = try add(retained_bytes, try mul(scratch, @sizeOf(f64)));
        if (required > max_bytes or required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        const owned_ranges = try allocator.dupe(Range, ranges);
        errdefer allocator.free(owned_ranges);
        const original_mass = try allocator.alloc(f64, q);
        errdefer allocator.free(original_mass);
        const original_output = try allocator.alloc(f64, qd);
        errdefer allocator.free(original_output);
        const chunk_mass = try allocator.alloc(f64, qr);
        errdefer allocator.free(chunk_mass);
        const chunk_output = try allocator.alloc(f64, rqd);
        errdefer allocator.free(chunk_output);
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
            for (ranges, 0..) |r, ri| {
                const chunk: solver.Head = .{ .keys = original.keys[r.start * original.dim ..][0 .. r.rows * original.dim], .values = original.values[r.start * original.dim ..][0 .. r.rows * original.dim], .bias = if (original.bias) |b| b[r.start..][0..r.rows] else null, .rows = r.rows, .dim = original.dim, .scale = original.scale, .softcap = original.softcap };
                try solver.attendBatch(allocator, chunk, qs, probabilities[0 .. take * r.rows], outputs[0 .. take * original.dim], masses[0..take], cfg);
                @memcpy(chunk_mass[ri * q + offset ..][0..take], masses[0..take]);
                @memcpy(chunk_output[ri * qd + offset * original.dim ..][0 .. take * original.dim], outputs[0 .. take * original.dim]);
            }
            offset += take;
        }
        @memset(current_mass, 1);
        @memcpy(current_numerator, original_output);
        return .{ .allocator = allocator, .original = original, .queries = queries, .ranges = owned_ranges, .original_mass = original_mass, .original_output = original_output, .chunk_mass = chunk_mass, .chunk_output = chunk_output, .current_mass = current_mass, .current_numerator = current_numerator, .control = cfg.control, .retained_bytes = retained_bytes, .max_bytes = @min(max_bytes, cfg.max_workspace_bytes) };
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.ranges);
        self.allocator.free(self.original_mass);
        self.allocator.free(self.original_output);
        self.allocator.free(self.chunk_mass);
        self.allocator.free(self.chunk_output);
        self.allocator.free(self.current_mass);
        self.allocator.free(self.current_numerator);
        self.* = undefined;
    }

    pub fn remainingWorkspace(self: *const Cache, external_live_bytes: usize) !usize {
        const live = try add(self.retained_bytes, external_live_bytes);
        if (live > self.max_bytes) return error.CompactionWorkspaceLimit;
        return self.max_bytes - live;
    }

    pub fn trial(self: *const Cache, chunk_index: usize, candidate: solver.Head, external_live_bytes: usize) !Trial {
        if (chunk_index >= self.ranges.len or candidate.dim != self.original.dim or candidate.rows == 0 or candidate.scale != self.original.scale or candidate.softcap != self.original.softcap) return error.InvalidFitConfig;
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
        if (try add(try add(self.retained_bytes, external_live_bytes), trial_bytes) > self.max_bytes) return error.CompactionWorkspaceLimit;
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
                const rho = @exp(self.chunk_mass[chunk_index * q + qi] - self.original_mass[qi]);
                const sigma = @exp(z[bi] - self.original_mass[qi]);
                const dm = sigma - rho;
                delta_mass[qi] = dm;
                const trial_d = self.current_mass[qi] + dm;
                if (!std.math.isFinite(trial_d) or trial_d <= 0) return error.NonFiniteAttention;
                trial_metrics.max_relative_mass_error = @max(trial_metrics.max_relative_mass_error, @abs(trial_d - 1));
                for (0..d) |j| {
                    const old = rho * self.chunk_output[chunk_index * qd + qi * d + j];
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
        return .{ .allocator = self.allocator, .delta_mass = delta_mass, .delta_numerator = delta_numerator, .metrics = trial_metrics };
    }

    pub fn commit(self: *Cache, proposal: *const Trial) !void {
        if (proposal.delta_mass.len != self.current_mass.len or proposal.delta_numerator.len != self.current_numerator.len) return error.InvalidAttentionShape;
        try self.control.check();
        for (self.current_mass, proposal.delta_mass) |*x, d| x.* += d;
        for (self.current_numerator, proposal.delta_numerator) |*x, d| x.* += d;
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

test "chunk admission cached composition matches direct full head and rollback is ordered" {
    const original: solver.Head = .{ .keys = &.{ 0, 1, 2 }, .values = &.{ 2, 4, 8 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const queries = [_]f64{ 0.2, 0.8 };
    const ranges = [_]Range{ .{ .start = 0, .rows = 1 }, .{ .start = 1, .rows = 2 } };
    var cache = try Cache.init(std.testing.allocator, original, &queries, &ranges, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }, 1 << 20);
    defer cache.deinit();
    const bad: solver.Head = .{ .keys = &.{4}, .values = &.{20}, .bias = &.{0}, .rows = 1, .dim = 1, .scale = 1 };
    var rejected = try cache.trial(0, bad, 0);
    defer rejected.deinit();
    try std.testing.expect(rejected.metrics.max_relative_mass_error > 0.05);
    try std.testing.expectEqual(@as(f64, 1), cache.current_mass[0]);
    const first: solver.Head = .{ .keys = &.{0.01}, .values = &.{2.001}, .bias = &.{0.005}, .rows = 1, .dim = 1, .scale = 1 };
    var first_trial = try cache.trial(0, first, 0);
    defer first_trial.deinit();
    try cache.commit(&first_trial);
    const second: solver.Head = .{ .keys = &.{1.5}, .values = &.{6}, .bias = &.{@log(@as(f64, 2))}, .rows = 1, .dim = 1, .scale = 1 };
    var second_trial = try cache.trial(1, second, 0);
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
    var t = try c.trial(0, .{ .keys = &.{0}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 }, 0);
    t.deinit();
}
test "chunk admission enforces cap cancellation and allocation unwind" {
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
    const saved = c.max_bytes;
    c.max_bytes = c.retained_bytes;
    try std.testing.expectError(error.CompactionWorkspaceLimit, c.trial(0, .{ .keys = &.{0}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 }, 0));
    c.max_bytes = saved;
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
