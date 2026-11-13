// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Exact per-head cache for the immutable ORIGINAL side of FP64 validation.
//! Candidate summaries never enter this adapter. Query roles have distinct
//! owners so TRAIN bytes cannot satisfy HHO requests (or vice versa).
const std = @import("std");
const platform = @import("antfly_platform");
const solver = @import("attention_match.zig");

pub const Policy = enum { disabled, exact_original_train_hho_v1 };
pub const Role = enum { train, heldout };

pub const Counters = struct {
    train_fills: usize = 0,
    train_hits: usize = 0,
    heldout_fills: usize = 0,
    heldout_hits: usize = 0,
    candidate_delegates: usize = 0,
    candidate_hits: usize = 0,
    retained_bytes_peak: usize = 0,
    cap_fallbacks_before_allocation: usize = 0,
    detailed_observer_bypasses: usize = 0,
    train_fill_ns: u64 = 0,
    train_hit_ns: u64 = 0,
    heldout_fill_ns: u64 = 0,
    heldout_hit_ns: u64 = 0,
    scalar_fill_ns: u64 = 0,
    accelerated_fill_ns: u64 = 0,
    train_init_ns: u64 = 0,
    heldout_init_ns: u64 = 0,
};

pub fn retainedBytes(query_rows: usize, dim: usize) !usize {
    const targets = try std.math.mul(usize, try std.math.mul(usize, query_rows, dim), @sizeOf(f64));
    const masses = try std.math.mul(usize, query_rows, @sizeOf(f64));
    const valid = try std.math.mul(usize, query_rows, @sizeOf(bool));
    return try std.math.add(usize, targets, try std.math.add(usize, masses, valid));
}

test "reference cache retained bytes are exact and overflow checked" {
    try std.testing.expectEqual(@as(usize, 2 * 3 * 8 + 2 * 8 + 2), try retainedBytes(2, 3));
    try std.testing.expectError(error.Overflow, retainedBytes(std.math.maxInt(usize), 2));
}

pub const Context = struct {
    allocator: std.mem.Allocator,
    role: Role,
    original: solver.Head,
    queries: []const f64,
    inner: ?solver.Backend,
    targets: []f64,
    masses: []f64,
    valid: []bool,
    counters: *Counters,

    pub fn init(allocator: std.mem.Allocator, role: Role, original: solver.Head, queries: []const f64, inner: solver.Backend, counters: *Counters) !Context {
        const started = platform.time.monotonicNs();
        if (inner.summary_fn == null or inner.validation_batch != .rows_256 or original.dim == 0 or queries.len == 0 or queries.len % original.dim != 0) return error.InvalidFitConfig;
        const rows = queries.len / original.dim;
        const new_retained_peak = try std.math.add(usize, counters.retained_bytes_peak, try retainedBytes(rows, original.dim));
        const targets = try allocator.alloc(f64, queries.len);
        errdefer allocator.free(targets);
        const masses = try allocator.alloc(f64, rows);
        errdefer allocator.free(masses);
        const valid = try allocator.alloc(bool, rows);
        @memset(valid, false);
        counters.retained_bytes_peak = new_retained_peak;
        switch (role) {
            .train => counters.train_init_ns +|= platform.time.monotonicNs() -| started,
            .heldout => counters.heldout_init_ns +|= platform.time.monotonicNs() -| started,
        }
        return .{ .allocator = allocator, .role = role, .original = original, .queries = queries, .inner = inner, .targets = targets, .masses = masses, .valid = valid, .counters = counters };
    }

    pub fn initScalar(allocator: std.mem.Allocator, original: solver.Head, queries: []const f64, counters: *Counters) !Context {
        const started = platform.time.monotonicNs();
        if (original.dim == 0 or queries.len == 0 or queries.len % original.dim != 0) return error.InvalidFitConfig;
        const rows = queries.len / original.dim;
        const new_retained_peak = try std.math.add(usize, counters.retained_bytes_peak, try retainedBytes(rows, original.dim));
        const targets = try allocator.alloc(f64, queries.len);
        errdefer allocator.free(targets);
        const masses = try allocator.alloc(f64, rows);
        errdefer allocator.free(masses);
        const valid = try allocator.alloc(bool, rows);
        @memset(valid, false);
        counters.retained_bytes_peak = new_retained_peak;
        counters.train_init_ns +|= platform.time.monotonicNs() -| started;
        return .{ .allocator = allocator, .role = .train, .original = original, .queries = queries, .inner = null, .targets = targets, .masses = masses, .valid = valid, .counters = counters };
    }

    pub fn deinit(self: *Context) void {
        self.allocator.free(self.valid);
        self.allocator.free(self.masses);
        self.allocator.free(self.targets);
        self.* = undefined;
    }

    fn sameOriginal(self: *const Context, head: solver.Head) bool {
        return head.rows == self.original.rows and head.dim == self.original.dim and head.scale == self.original.scale and head.softcap == self.original.softcap and
            head.keys.ptr == self.original.keys.ptr and head.keys.len == self.original.keys.len and
            head.values.ptr == self.original.values.ptr and head.values.len == self.original.values.len and
            ((head.bias == null and self.original.bias == null) or (head.bias != null and self.original.bias != null and head.bias.?.ptr == self.original.bias.?.ptr and head.bias.?.len == self.original.bias.?.len));
    }

    fn summary(raw: ?*anyopaque, allocator: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
        const started = platform.time.monotonicNs();
        const self: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        try control.check();
        if (!self.sameOriginal(head) or queries.len == 0 or queries.len % head.dim != 0 or targets.len != queries.len or masses.len != queries.len / head.dim) return error.InvalidFitConfig;
        const base = @intFromPtr(self.queries.ptr);
        const here = @intFromPtr(queries.ptr);
        if (here < base or (here - base) % @sizeOf(f64) != 0) return error.InvalidFitConfig;
        const scalar_offset = (here - base) / @sizeOf(f64);
        if (scalar_offset % head.dim != 0 or scalar_offset > self.queries.len or queries.len > self.queries.len - scalar_offset) return error.InvalidFitConfig;
        const row_offset = scalar_offset / head.dim;
        const rows = queries.len / head.dim;
        const validity = self.valid[row_offset..][0..rows];
        const all_valid = std.mem.allEqual(bool, validity, true);
        if (all_valid) {
            @memcpy(targets, self.targets[scalar_offset..][0..queries.len]);
            @memcpy(masses, self.masses[row_offset..][0..rows]);
            switch (self.role) {
                .train => {
                    self.counters.train_hits += 1;
                    self.counters.train_hit_ns +|= platform.time.monotonicNs() -| started;
                },
                .heldout => {
                    self.counters.heldout_hits += 1;
                    self.counters.heldout_hit_ns +|= platform.time.monotonicNs() -| started;
                },
            }
            return control.check();
        }
        if (!std.mem.allEqual(bool, validity, false)) return error.InvalidFitConfig;
        if (self.inner) |inner| {
            try inner.summary_fn.?(inner.context, allocator, head, queries, targets, masses, control);
            self.counters.accelerated_fill_ns +|= platform.time.monotonicNs() -| started;
        } else {
            const probabilities = try allocator.alloc(f64, try std.math.mul(usize, rows, head.rows));
            defer allocator.free(probabilities);
            try solver.attendBatch(allocator, head, queries, probabilities, targets, masses, .{ .target_rows = 1, .control = control });
            self.counters.scalar_fill_ns +|= platform.time.monotonicNs() -| started;
        }
        @memcpy(self.targets[scalar_offset..][0..queries.len], targets);
        @memcpy(self.masses[row_offset..][0..rows], masses);
        @memset(validity, true);
        switch (self.role) {
            .train => {
                self.counters.train_fills += 1;
                self.counters.train_fill_ns +|= platform.time.monotonicNs() -| started;
            },
            .heldout => {
                self.counters.heldout_fills += 1;
                self.counters.heldout_fill_ns +|= platform.time.monotonicNs() -| started;
            },
        }
    }

    fn workspace(raw: ?*anyopaque, rows: usize, candidates: usize, queries: usize, dim: usize) !usize {
        const self: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        if (self.inner) |inner| return inner.workspace_fn(inner.context, rows, candidates, queries, dim);
        return std.math.mul(usize, std.math.mul(usize, rows, queries) catch return error.InvalidAttentionShape, @sizeOf(f64)) catch error.InvalidAttentionShape;
    }

    fn candidateDelegate(raw: ?*anyopaque) void {
        const self: *Context = @ptrCast(@alignCast(raw orelse return));
        self.counters.candidate_delegates += 1;
    }

    pub fn referenceBackend(self: *Context) solver.ReferenceSummaryBackend {
        return .{ .backend = .{ .validation_batch = .rows_256, .context = self, .workspace_fn = workspace, .summary_fn = summary }, .observer_context = self, .candidate_delegate_fn = candidateDelegate };
    }
};

test "reference cache separates role source query shape and candidate delegation" {
    const Inner = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn summary(raw: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            @memcpy(targets, queries);
            @memset(masses, @floatFromInt(head.rows));
        }
    };
    var inner: Inner = .{};
    const backend: solver.Backend = .{ .validation_batch = .rows_256, .context = &inner, .workspace_fn = Inner.workspace, .summary_fn = Inner.summary };
    const values = [_]f64{ 1, 2 };
    const head: solver.Head = .{ .keys = &values, .values = &values, .rows = 1, .dim = 2, .scale = 1 };
    const queries = [_]f64{ 3, 4, 5, 6 };
    var counters: Counters = .{};
    var cache = try Context.init(std.testing.allocator, .train, head, &queries, backend, &counters);
    defer cache.deinit();
    var cfg: solver.Config = .{ .target_rows = 1, .backend = backend, .reference_summary_backend = cache.referenceBackend() };
    _ = try solver.validationMetrics(std.testing.allocator, head, head, &queries, cfg);
    _ = try solver.validationMetrics(std.testing.allocator, head, head, &queries, cfg);
    try std.testing.expectEqual(@as(usize, 1), counters.train_fills);
    try std.testing.expectEqual(@as(usize, 1), counters.train_hits);
    try std.testing.expectEqual(@as(usize, 2), counters.candidate_delegates);
    try std.testing.expectEqual(@as(usize, 0), counters.candidate_hits);
    cfg.reference_summary_backend = null;
    _ = try solver.validationMetrics(std.testing.allocator, head, head, &queries, cfg);
    try std.testing.expectEqual(@as(usize, 5), inner.calls);
}

test "reference cache unwinds allocation failures" {
    const values = [_]f64{1};
    const head: solver.Head = .{ .keys = &values, .values = &values, .rows = 1, .dim = 1, .scale = 1 };
    var counters: Counters = .{};
    const backend: solver.Backend = .{ .validation_batch = .rows_256, .workspace_fn = struct {
        fn f(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
    }.f, .summary_fn = struct {
        fn f(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {}
    }.f };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, h: solver.Head, b: solver.Backend, c: *Counters) !void {
            var x = try Context.init(a, .heldout, h, &.{1}, b, c);
            defer x.deinit();
        }
    }.run, .{ head, backend, &counters });
}

test "reference cache rejects foreign source query role shape and cancellation" {
    const Inner = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, queries: []const f64, targets: []f64, masses: []f64, control: solver.Control) !void {
            try control.check();
            @memcpy(targets, queries);
            @memset(masses, 0);
        }
    };
    const backend: solver.Backend = .{ .validation_batch = .rows_256, .workspace_fn = Inner.workspace, .summary_fn = Inner.summary };
    const payload = [_]f64{ 1, 2 };
    const source: solver.Head = .{ .keys = &payload, .values = &payload, .rows = 1, .dim = 2, .scale = 1 };
    const foreign: solver.Head = .{ .keys = &payload, .values = &payload, .rows = 1, .dim = 2, .scale = 2 };
    const queries = [_]f64{ 3, 4 };
    var counters: Counters = .{};
    var train = try Context.init(std.testing.allocator, .train, source, &queries, backend, &counters);
    defer train.deinit();
    var targets: [2]f64 = undefined;
    var masses: [1]f64 = undefined;
    try std.testing.expectError(error.InvalidFitConfig, Context.summary(&train, std.testing.allocator, foreign, &queries, &targets, &masses, .{}));
    // Distinct live storage is required: identical const arrays may share a
    // compiler-interned address and would not represent a foreign query owner.
    const copied = try std.testing.allocator.dupe(f64, &queries);
    defer std.testing.allocator.free(copied);
    try std.testing.expect(@intFromPtr(copied.ptr) != @intFromPtr(&queries));
    try std.testing.expectError(error.InvalidFitConfig, Context.summary(&train, std.testing.allocator, source, copied, &targets, &masses, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, Context.summary(&train, std.testing.allocator, source, &queries, &targets, &masses, .{ .check_fn = Cancel.check }));
    var heldout = try Context.init(std.testing.allocator, .heldout, source, &queries, backend, &counters);
    defer heldout.deinit();
    try Context.summary(&heldout, std.testing.allocator, source, &queries, &targets, &masses, .{});
    try std.testing.expectEqual(@as(usize, 0), counters.train_fills);
    try std.testing.expectEqual(@as(usize, 1), counters.heldout_fills);
}

test "scalar TRAIN cached and uncached validation metrics are bit identical" {
    const expectMetricsBits = struct {
        fn equal(expected: solver.Metrics, actual: solver.Metrics) !void {
            try std.testing.expectEqual(expected.quality_evaluated, actual.quality_evaluated);
            try std.testing.expectEqual(@as(u64, @bitCast(expected.max_relative_mass_error)), @as(u64, @bitCast(actual.max_relative_mass_error)));
            try std.testing.expectEqual(@as(u64, @bitCast(expected.relative_output_rmse)), @as(u64, @bitCast(actual.relative_output_rmse)));
            try std.testing.expectEqual(expected.nnls_sweeps, actual.nnls_sweeps);
            try std.testing.expectEqual(expected.selection_steps, actual.selection_steps);
            try std.testing.expectEqual(expected.selection_active_solves, actual.selection_active_solves);
        }
    }.equal;
    const source_keys = [_]f64{ 0.25, -0.5, 0.75, 0.125 };
    const source_values = [_]f64{ 1.5, -2, 0.25, 3 };
    const candidate_keys = [_]f64{ 0.25, -0.5 };
    const candidate_values = [_]f64{ 1.5, -2 };
    const source: solver.Head = .{ .keys = &source_keys, .values = &source_values, .bias = &.{ 0.1, -0.2 }, .rows = 2, .dim = 2, .scale = 0.75, .softcap = 4 };
    const candidate: solver.Head = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &.{0.1}, .rows = 1, .dim = 2, .scale = 0.75, .softcap = 4 };
    const queries = [_]f64{ 0.5, -1, 1.25, 0.375, -0.75, 2 };
    const uncached = try solver.validationMetrics(std.testing.allocator, source, candidate, &queries, .{ .target_rows = 1 });
    var counters: Counters = .{};
    var cache = try Context.initScalar(std.testing.allocator, source, &queries, &counters);
    defer cache.deinit();
    const cfg: solver.Config = .{ .target_rows = 1, .reference_summary_backend = cache.referenceBackend() };
    const first = try solver.validationMetrics(std.testing.allocator, source, candidate, &queries, cfg);
    const second = try solver.validationMetrics(std.testing.allocator, source, candidate, &queries, cfg);
    try expectMetricsBits(uncached, first);
    try expectMetricsBits(uncached, second);
    try std.testing.expectEqual(@as(usize, queries.len / source.dim), counters.train_fills);
    try std.testing.expectEqual(@as(usize, queries.len / source.dim), counters.train_hits);
    try std.testing.expectEqual(@as(usize, 2 * queries.len / source.dim), counters.candidate_delegates);
    try std.testing.expectEqual(@as(usize, 0), counters.candidate_hits);
}

test "bounded Accelerate TRAIN reference summary preserves scalar FP64 metrics" {
    const accelerated = @import("../../ops/metal_attention_match_validation.zig");
    const keys = [_]f64{ 0.25, -0.5, 0.75, 0.125, -0.375, 0.625 };
    const values = [_]f64{ 1.5, -2, 0.25, 3, -1.25, 0.75 };
    const source: solver.Head = .{ .keys = &keys, .values = &values, .bias = &.{ 0.1, -0.2, 0.05 }, .rows = 3, .dim = 2, .scale = 0.75, .softcap = 4 };
    const compact: solver.Head = .{ .keys = keys[0..4], .values = values[0..4], .bias = &.{ 0.1, -0.2 }, .rows = 2, .dim = 2, .scale = 0.75, .softcap = 4 };
    const queries = [_]f64{ 0.5, -1, 1.25, 0.375, -0.75, 2, 0.125, -0.25 };
    const scalar = try solver.validationMetrics(std.testing.allocator, source, compact, &queries, .{ .target_rows = 1 });
    var backend_context: accelerated.Context = .{};
    var counters: Counters = .{};
    var cache = try Context.init(std.testing.allocator, .train, source, &queries, accelerated.backend(&backend_context), &counters);
    defer cache.deinit();
    const observed = try solver.validationMetrics(std.testing.allocator, source, compact, &queries, .{ .target_rows = 1, .reference_summary_backend = cache.referenceBackend() });
    try std.testing.expectApproxEqRel(scalar.max_relative_mass_error, observed.max_relative_mass_error, 8 * std.math.floatEps(f64));
    try std.testing.expectApproxEqRel(scalar.relative_output_rmse, observed.relative_output_rmse, 8 * std.math.floatEps(f64));
    // Scalar candidate evaluation requests one original-reference row at a
    // time even when its reference provider uses bounded Accelerate kernels.
    try std.testing.expectEqual(queries.len / source.dim, counters.train_fills);
    try std.testing.expect(backend_context.successfulCalls() > 0);
}

test "detailed observer bypasses reference cache and preserves full observation bits" {
    const values = [_]f64{ 1, -0.5, 0.25, 2 };
    const source: solver.Head = .{ .keys = &values, .values = &values, .rows = 2, .dim = 2, .scale = 1 };
    const compact: solver.Head = .{ .keys = values[0..2], .values = values[0..2], .rows = 1, .dim = 2, .scale = 1 };
    const queries = [_]f64{ 0.5, -1, 2, 0.25 };
    const expected = try solver.validationMetricsDetailed(std.testing.allocator, source, compact, &queries, .{ .target_rows = 1 });
    var counters: Counters = .{};
    var cache = try Context.initScalar(std.testing.allocator, source, &queries, &counters);
    defer cache.deinit();
    const actual = try solver.validationMetricsDetailed(std.testing.allocator, source, compact, &queries, .{ .target_rows = 1, .reference_summary_backend = cache.referenceBackend() });
    try std.testing.expectEqualDeep(expected, actual);
    try std.testing.expectEqual(@as(usize, 0), counters.train_fills);
    try std.testing.expectEqual(@as(usize, 0), counters.train_hits);
    try std.testing.expectEqual(@as(usize, 0), counters.candidate_delegates);
}
