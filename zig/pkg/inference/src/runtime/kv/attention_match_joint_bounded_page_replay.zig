// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! Concrete default-off streaming-select + uncached joint-fit adapter.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");
const joint = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const bounded = @import("attention_match_joint_bounded_page_scoring.zig");
const tiled_bridge = @import("attention_match_joint_bounded_tiled_replay.zig");
const delta_kernel = @import("attention_match_gram_delta_accelerate.zig");
const accelerated_delta = @import("attention_match_bounded_page_delta_accelerate.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const streamed_gram = @import("attention_match_streamed_gram_selection.zig");
const critical_adapter = @import("attention_match_critical_core_adapter.zig");
const critical_range = @import("attention_match_train_critical_range_streaming.zig");
const metal = @import("../../ops/metal_attention_match_fit.zig");

extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

pub const policy = "bounded_stream_select_joint_fit_floatpair_score_v1";
pub const ScorePolicy = enum { scalar_reference, accelerate_fp64_tiled_v1 };
const Owned = struct { frozen: replay.FrozenOwned, fitted: joint.Result, range: admission.Range, target: usize };

const GeometryAuthority = struct { external: usize, transfer_peak: usize };

fn capAdd(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn capMul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn scaledProbability(probability: f64, fitted_bias: f64, original_bias: f64) !f64 {
    if (!std.math.isFinite(probability) or probability < 0 or !std.math.isFinite(fitted_bias) or !std.math.isFinite(original_bias)) return error.NonFiniteAttention;
    const ratio = @exp(fitted_bias - original_bias);
    const scaled = probability * ratio;
    if (!std.math.isFinite(ratio) or ratio < 0 or !std.math.isFinite(scaled)) return error.NonFiniteAttention;
    return scaled;
}

pub fn acceleratedTileReductions(page: []const f64, source_values: []const f64, selected_local: []const usize, fitted_bias: []const f64, original_bias: []const f64, fitted_values: []const f64, q: usize, rows: usize, d: usize, rm: []f64, rn: []f64, cm: []f64, cn: []f64, selected: []f64) !void {
    const target = selected_local.len;
    if (q == 0 or rows == 0 or target == 0 or d == 0 or page.len != q * rows or source_values.len != rows * d or fitted_bias.len != target or original_bias.len != rows or fitted_values.len != target * d or rm.len != q or rn.len != q * d or cm.len != q or cn.len != q * d or selected.len != q * target) return error.InvalidAttentionShape;
    if (comptime builtin.os.tag != .macos) return error.UnsupportedOperation;
    @memset(rm, 0);
    @memset(cm, 0);
    for (0..q) |qi| {
        for (page[qi * rows ..][0..rows]) |p| {
            if (!std.math.isFinite(p) or p < 0) return error.NonFiniteAttention;
            rm[qi] += p;
        }
        for (selected_local, 0..) |row, si| {
            if (row >= rows) return error.InvalidAttentionShape;
            const p = try scaledProbability(page[qi * rows + row], fitted_bias[si], original_bias[row]);
            selected[qi * target + si] = p;
            cm[qi] += p;
        }
    }
    const qi = std.math.cast(c_int, q) orelse return error.InvalidAttentionShape;
    const ri = std.math.cast(c_int, rows) orelse return error.InvalidAttentionShape;
    const ti = std.math.cast(c_int, target) orelse return error.InvalidAttentionShape;
    const di = std.math.cast(c_int, d) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(101, 111, 111, qi, di, ri, 1, page.ptr, ri, source_values.ptr, di, 0, rn.ptr, di);
    @"cblas_dgemm$NEWLAPACK"(101, 111, 111, qi, di, ti, 1, selected.ptr, ti, fitted_values.ptr, di, 0, cn.ptr, di);
    for (rn) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    for (cn) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
}

test "accelerated bounded score rejects nonfinite probability and ratio before BLAS" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try scaledProbability(0.25, @log(2.0), 0), 1e-15);
    try std.testing.expectError(error.NonFiniteAttention, scaledProbability(std.math.nan(f64), 0, 0));
    try std.testing.expectError(error.NonFiniteAttention, scaledProbability(-0.1, 0, 0));
    try std.testing.expectError(error.NonFiniteAttention, scaledProbability(1, std.math.inf(f64), 0));
    try std.testing.expectError(error.NonFiniteAttention, scaledProbability(std.math.floatMax(f64), @log(2.0), 0));
}

test "accelerated bounded score rejects empty candidate before BLAS" {
    var rm: [1]f64 = undefined;
    var rn: [1]f64 = undefined;
    var cm: [1]f64 = undefined;
    var cn: [1]f64 = undefined;
    var selected: [0]f64 = .{};
    try std.testing.expectError(error.InvalidAttentionShape, acceleratedTileReductions(&.{1}, &.{2}, &.{}, &.{}, &.{0}, &.{}, 1, 1, 1, &rm, &rn, &cm, &cn, &selected));
}

test "accelerated bounded score production tile reductions match scalar oracle" {
    if (comptime builtin.os.tag != .macos) return error.SkipZigTest;
    const q: usize = 3;
    const rows: usize = 4;
    const target: usize = 2;
    const d: usize = 3;
    const captured_f32 = [_]f32{ 0.10, 0.20, 0.30, 0.40, 0.25, 0.15, 0.35, 0.25, 0.05, 0.45, 0.10, 0.40 };
    var page: [captured_f32.len]f64 = undefined;
    for (captured_f32, &page) |narrow, *wide| wide.* = narrow;
    const source_values = [_]f64{
        1.0,  -2.0, 0.5, 0.25, 3.0,   -1.0,
        -0.5, 0.75, 2.0, 4.0,  -0.25, 1.5,
    };
    const selected_local = [_]usize{ 1, 3 };
    const fitted_bias = [_]f64{ @as(f64, @floatCast(@as(f32, 0.125))), @as(f64, @floatCast(@as(f32, -0.25))) };
    const original_bias = [_]f64{ 0.2, -0.1, 0.05, 0.3 };
    const fitted_values = [_]f64{
        @as(f64, @floatCast(@as(f16, 0.5))),  @as(f64, @floatCast(@as(f16, -1.25))), @as(f64, @floatCast(@as(f16, 2.0))),
        @as(f64, @floatCast(@as(f16, 3.25))), @as(f64, @floatCast(@as(f16, 0.75))),  @as(f64, @floatCast(@as(f16, -0.5))),
    };
    var scalar_rm: [q]f64 = @splat(0);
    var scalar_rn: [q * d]f64 = @splat(0);
    var scalar_cm: [q]f64 = @splat(0);
    var scalar_cn: [q * d]f64 = @splat(0);
    var selected: [q * target]f64 = undefined;
    for (0..q) |qi| {
        for (page[qi * rows ..][0..rows], 0..) |p, row| {
            scalar_rm[qi] += p;
            for (0..d) |di| scalar_rn[qi * d + di] += p * source_values[row * d + di];
        }
        for (selected_local, 0..) |row, si| {
            const p = try scaledProbability(page[qi * rows + row], fitted_bias[si], original_bias[row]);
            selected[qi * target + si] = p;
            scalar_cm[qi] += p;
            for (0..d) |di| scalar_cn[qi * d + di] += p * fitted_values[si * d + di];
        }
    }
    var accelerated_rm: [q]f64 = undefined;
    var accelerated_rn: [q * d]f64 = undefined;
    var accelerated_cm: [q]f64 = undefined;
    var accelerated_cn: [q * d]f64 = undefined;
    try acceleratedTileReductions(&page, &source_values, &selected_local, &fitted_bias, &original_bias, &fitted_values, q, rows, d, &accelerated_rm, &accelerated_rn, &accelerated_cm, &accelerated_cn, &selected);
    try std.testing.expectEqualSlices(f64, &scalar_rm, &accelerated_rm);
    try std.testing.expectEqualSlices(f64, &scalar_cm, &accelerated_cm);
    for (scalar_rn, accelerated_rn) |expected, actual| try std.testing.expectApproxEqRel(expected, actual, 2e-15);
    for (scalar_cn, accelerated_cn) |expected, actual| try std.testing.expectApproxEqRel(expected, actual, 2e-15);
    const original_y = [_]f64{ 1.5, -0.5, 2.25, 0.75, 1.0, -1.5, 2.5, 0.25, 0.5 };
    const scalar_harm = try adaptive.measureRoundedCandidateHarm(&original_y, &scalar_rm, &scalar_rn, &scalar_cm, &scalar_cn, q, d, 1e-12);
    const accelerated_harm = try adaptive.measureRoundedCandidateHarm(&original_y, &accelerated_rm, &accelerated_rn, &accelerated_cm, &accelerated_cn, q, d, 1e-12);
    try std.testing.expectApproxEqRel(scalar_harm.max_relative_mass, accelerated_harm.max_relative_mass, 2e-14);
    try std.testing.expectApproxEqRel(scalar_harm.relative_output_rmse, accelerated_harm.relative_output_rmse, 2e-14);
}

fn hashUsizeLe(hash: *std.crypto.hash.sha2.Sha256, value: usize) !void {
    const narrowed = std.math.cast(u64, value) orelse return error.InvalidAttentionShape;
    var encoded: [8]u8 = undefined;
    std.mem.writeInt(u64, &encoded, narrowed, .little);
    hash.update(&encoded);
}

fn geometryAuthorityBytes(range_count: usize, retained_artifact_bytes: usize, external_live_bytes: usize) !GeometryAuthority {
    const score_bytes = try capMul(range_count, @sizeOf(adaptive.RangeScores));
    const source_table_bytes = try capMul(range_count, @sizeOf([3]?bounded.Artifact));
    const scored_table_bytes = try capMul(range_count, @sizeOf([3]?replay.ScoredCandidate));
    const external = try capAdd(external_live_bytes, try capAdd(retained_artifact_bytes, try capAdd(score_bytes, scored_table_bytes)));
    return .{ .external = external, .transfer_peak = try capAdd(external, source_table_bytes) };
}

pub const Context = struct {
    fitter: *metal.Fitter,
    normalizer: *const metal.Fitter.CapturedPageSource,
    original: solver.Head,
    train: []const f64,
    cfg: solver.Config,
    tile_queries: usize,
    builder_transient_bound_bytes: usize,
    page_scratch: []f64,
    accelerated_scratch: []f64 = &.{},
    score_policy: ScorePolicy = .scalar_reference,
    accelerated_score_calls: usize = 0,
    accelerated_score_tiles: usize = 0,
    accelerated_score_dgemm_calls: usize = 0,
    accelerated_score_cancellation_checks: usize = 0,
    allocator: std.mem.Allocator,
    original_y: []f64,
    range_cache_reserved_bytes: usize,
    range_cache: ?training_source.Source = null,
    range_cache_start: usize = 0,
    range_cache_rows: usize = 0,
    cache_counters: training_source.Counters = .{},
    cache_batch_rows: usize,
    geometry_cache_requests: usize = 0,
    geometry_cache_hits: usize = 0,
    geometry_cache_loads: usize = 0,
    range_fill_context: ?*anyopaque = null,
    range_fill_fn: ?*const fn (?*anyopaque, admission.Range, usize, usize, []f64, solver.Control) anyerror!void = null,
    /// Optional accelerated geometry-delta authority. These values are owned
    /// by the bounded route and forwarded without substitution to the tiled
    /// accessor. The scalar default keeps every field empty.
    delta_backend: ?delta_kernel.Backend = null,
    require_accelerated_delta: bool = false,
    ratio_scratch: []f64 = &.{},
    delta_evidence: accelerated_delta.Evidence = .{},
    streamed_gram_selection: bool = false,
    streamed_gram_calls: usize = 0,
    streamed_gram_probability_values: usize = 0,
    streamed_gram_workspace_peak_bytes: usize = 0,
    streamed_gram_hasher: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),
    critical_core_policy: replay.CriticalCorePolicy = .disabled,
    critical_authority: ?critical_range.Result = null,
    critical_authority_start: usize = 0,
    critical_authority_rows: usize = 0,
    critical_authority_builds: usize = 0,
    critical_rows_peak: usize = 0,
    critical_rows_total: usize = 0,
    critical_authority_hasher: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),

    pub fn deinit(self: *Context) void {
        if (self.range_cache) |*source| source.deinit();
        self.range_cache = null;
        if (self.critical_authority) |*authority| authority.deinit();
        self.critical_authority = null;
    }

    pub fn clearCriticalAuthority(self: *Context) void {
        if (self.critical_authority) |*authority| authority.deinit();
        self.critical_authority = null;
        self.critical_authority_start = 0;
        self.critical_authority_rows = 0;
    }

    fn ensureCriticalAuthority(self: *Context, range: admission.Range, control: solver.Control) !?*critical_range.Result {
        if (self.critical_core_policy == .disabled) return null;
        if (self.critical_authority_start == range.start and self.critical_authority_rows == range.rows)
            if (self.critical_authority) |*authority| return authority;
        if (self.critical_authority) |*authority| authority.deinit();
        self.critical_authority = null;
        const cached = try self.ensureRangeCache(range, control);
        const Copy = struct {
            source: *const training_source.Source,
            range: admission.Range,
            fn one(raw: ?*anyopaque, query: usize, out: []f64, query_control: solver.Control) !void {
                const context: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
                try query_control.check();
                try context.source.copyChunk(query, context.range.start, out);
            }
        };
        var copy: Copy = .{ .source = cached, .range = range };
        const cache_bytes = try std.math.mul(usize, try std.math.mul(usize, cached.query_count, range.rows), @sizeOf(f32));
        self.critical_authority = try critical_range.extract(self.allocator, .{ .context = &copy, .query_rows = cached.query_count, .complete_query_rows = cached.query_count, .global_start = range.start, .range_rows = range.rows, .original_rows = self.original.rows, .external_live_bytes = cache_bytes, .source_key_space = .global_original_range, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened, .copy_query_fn = Copy.one }, .{ .combined_cap = self.builder_transient_bound_bytes, .control = control });
        self.critical_authority_start = range.start;
        self.critical_authority_rows = range.rows;
        self.critical_authority_builds = try std.math.add(usize, self.critical_authority_builds, 1);
        self.critical_rows_peak = @max(self.critical_rows_peak, self.critical_authority.?.local_keys.len);
        self.critical_rows_total = try std.math.add(usize, self.critical_rows_total, self.critical_authority.?.local_keys.len);
        self.critical_authority_hasher.update("antfly.bounded-critical-range-authority.v1\x00");
        try hashUsizeLe(&self.critical_authority_hasher, range.start);
        try hashUsizeLe(&self.critical_authority_hasher, range.rows);
        try hashUsizeLe(&self.critical_authority_hasher, self.critical_authority.?.local_keys.len);
        try hashUsizeLe(&self.critical_authority_hasher, cached.query_count);
        try hashUsizeLe(&self.critical_authority_hasher, self.original.rows);
        self.critical_authority_hasher.update(&self.critical_authority.?.probability_sha256);
        self.critical_authority_hasher.update(std.mem.sliceAsBytes(self.critical_authority.?.local_keys));
        if (self.critical_authority) |*authority| return authority;
        return error.InvalidFitConfig;
    }

    fn targetRows(raw: ?*anyopaque, range: admission.Range, divisor: usize, control: solver.Control) !usize {
        const self: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const authority = (try self.ensureCriticalAuthority(range, control)) orelse return std.math.divCeil(usize, range.rows, divisor) catch error.InvalidFitConfig;
        if (range.rows < 2 or authority.local_keys.len >= range.rows - 1) return error.CompactionFitRejected;
        return @max(try std.math.add(usize, authority.local_keys.len, 1), std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig);
    }

    fn criticalAuthorityOwnedBytes(authority: *const critical_range.Result) !usize {
        var bytes = try capMul(authority.local_keys.len, @sizeOf(usize));
        bytes = try capAdd(bytes, try capMul(authority.query_offsets.len, @sizeOf(usize)));
        return capAdd(bytes, try capMul(authority.critical_queries.len, @sizeOf(usize)));
    }

    fn ensureRangeCache(self: *Context, range: admission.Range, control: solver.Control) !*training_source.Source {
        if (self.range_cache_start == range.start and self.range_cache_rows == range.rows)
            if (self.range_cache) |*source| return source;
        if (self.range_cache) |*source| source.deinit();
        self.range_cache = null;
        const q = self.train.len / self.original.dim;
        const count = try std.math.mul(usize, q, range.rows);
        const bytes = try std.math.mul(usize, count, @sizeOf(f32));
        if (bytes > self.range_cache_reserved_bytes) return error.CompactionWorkspaceLimit;
        const probabilities = try self.allocator.alloc(f32, count);
        errdefer self.allocator.free(probabilities);
        var first: usize = 0;
        while (first < q) {
            try control.check();
            const take = @min(self.tile_queries, q - first);
            const page_len = try std.math.mul(usize, take, range.rows);
            if (page_len > self.page_scratch.len) return error.CompactionWorkspaceLimit;
            const page = self.page_scratch[0..page_len];
            if (self.range_fill_fn) |fill| {
                if (comptime !builtin.is_test) return error.InvalidFitConfig;
                try fill(self.range_fill_context, range, first, take, page, control);
            } else try self.fitter.replayCapturedPageQueries(self.normalizer, range.start, range.rows, first, take, page, control);
            for (page, probabilities[first * range.rows ..][0..page_len], 0..) |value, *out, i| {
                if ((i & 4095) == 0) try control.check();
                out.* = @floatCast(value);
            }
            first += take;
        }
        // The borrowed range cache does not own the consumer workspace, but
        // fitStrategy still requires explicit authority for its nested peak.
        // This is the same pre-admitted per-candidate transient bound used by
        // Builder.transient_bytes_fn; cfg.max_workspace_bytes remains the
        // stricter limit after frozen payloads are subtracted.
        var source = try training_source.initBorrowedRangeF32FromValidatedHead(self.allocator, self.original, self.train, self.cfg, range.start, range.rows, probabilities, self.original_y, self.builder_transient_bound_bytes);
        source.query_batch_rows = self.cache_batch_rows;
        self.range_cache = source;
        self.range_cache_start = range.start;
        self.range_cache_rows = range.rows;
        self.cache_counters.prepare_attempts += 1;
        self.cache_counters.builds += 1;
        self.cache_counters.cache_bytes = @max(self.cache_counters.cache_bytes, bytes);
        if (self.range_cache) |*ready| return ready;
        return error.InvalidFitConfig;
    }

    pub fn loadRangeForPostfit(self: *Context, range: admission.Range, control: solver.Control) !*const training_source.Source {
        return try self.ensureRangeCache(range, control);
    }

    pub fn releaseRangeForPostfit(self: *Context) void {
        if (self.range_cache) |*source| source.deinit();
        self.range_cache = null;
        self.range_cache_start = 0;
        self.range_cache_rows = 0;
    }

    fn copyBoundRangePage(raw: ?*anyopaque, start: usize, rows: usize, first: usize, count: usize, out: []f64, control: solver.Control) !void {
        const self: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        if (rows == 0 or first > self.train.len / self.original.dim or count > self.train.len / self.original.dim - first or
            out.len != try std.math.mul(usize, count, rows)) return error.InvalidAttentionShape;
        try control.check();
        self.geometry_cache_requests = try std.math.add(usize, self.geometry_cache_requests, 1);
        const was_ready = self.range_cache != null and self.range_cache_start == start and self.range_cache_rows == rows;
        const before_builds = self.cache_counters.builds;
        const cached = try self.ensureRangeCache(.{ .start = start, .rows = rows }, control);
        if (was_ready) self.geometry_cache_hits = try std.math.add(usize, self.geometry_cache_hits, 1);
        if (self.cache_counters.builds != before_builds) self.geometry_cache_loads = try std.math.add(usize, self.geometry_cache_loads, 1);
        for (0..count) |local_query| {
            try control.check();
            try cached.copyChunk(first + local_query, start, out[local_query * rows ..][0..rows]);
        }
    }

    fn retained(raw: ?*anyopaque, _: admission.Range, target: usize) !usize {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        var per = try std.math.add(usize, @sizeOf(usize) + @sizeOf(f64), try std.math.add(usize, @sizeOf(f64), try std.math.mul(usize, s.original.dim, @sizeOf(f64))));
        if (s.critical_core_policy != .disabled) per = try std.math.add(usize, per, @sizeOf(usize));
        return std.math.add(usize, @sizeOf(Owned) + @sizeOf(joint.FrozenChunk) + @sizeOf(joint.ChunkResult) + @sizeOf(admission.Range) + @sizeOf([]usize) + @sizeOf([]f64), try std.math.mul(usize, target, per));
    }
    fn transient(raw: ?*anyopaque, _: admission.Range, _: usize) !usize {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        return s.builder_transient_bound_bytes;
    }
    fn destroy(raw: ?*anyopaque, a: std.mem.Allocator) void {
        const o: *Owned = @ptrCast(@alignCast(raw orelse return));
        o.fitted.deinit();
        o.frozen.deinit();
        a.destroy(o);
    }
    fn build(raw: ?*anyopaque, a: std.mem.Allocator, range: admission.Range, target: usize, _: usize, control: solver.Control) !bounded.Artifact {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        if (target == 0 or target >= range.rows) return error.InvalidFitConfig;
        var cfg = s.cfg;
        cfg.control = control;
        cfg.max_workspace_bytes = s.builder_transient_bound_bytes;
        const cached = try s.ensureRangeCache(range, control);
        // Cached feature selection is scalar and never asks the combined Metal
        // ABI to admit all high-fanout TRAIN rows. Value fitting retains cfg's
        // explicit TSQR ValueBackend below.
        var frozen = if (s.critical_core_policy != .disabled) blk: {
            const authority = (try s.ensureCriticalAuthority(range, control)) orelse return error.InvalidFitConfig;
            const authority_bytes = try criticalAuthorityOwnedBytes(authority);
            if (authority_bytes >= cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            cfg.max_workspace_bytes -= authority_bytes;
            const Copy = struct {
                parent: *Context,
                range: admission.Range,
                fn one(raw_context: ?*anyopaque, query: usize, out: []f64, query_control: solver.Control) !void {
                    const self: *@This() = @ptrCast(@alignCast(raw_context orelse return error.InvalidCallbackContext));
                    return copyBoundRangePage(self.parent, self.range.start, self.range.rows, query, 1, out, query_control);
                }
            };
            var copy: Copy = .{ .parent = s, .range = range };
            var selected = critical_adapter.selectStreamedRange(a, .{ .context = &copy, .query_count = s.train.len / s.original.dim, .range = range, .copy_full_fn = Copy.one }, target, authority, .{ .target_rows = target, .numerical_policy = .normalized_probability_raw_v1, .query_batch_rows = s.tile_queries, .max_sweeps = cfg.max_sweeps, .nnls_tolerance = cfg.nnls_tolerance, .external_live_bytes = authority_bytes, .combined_cap = s.builder_transient_bound_bytes, .control = control }) catch |err| {
                if (err == error.MassFitDidNotConverge or err == error.InsufficientSelectionProgress) std.debug.print("attention_match_bounded_page_selection phase=critical_streamed_gram range_start={d} range_rows={d} target_rows={d} query_count={d} error={s} diagnostic_only=true\n", .{ range.start, range.rows, target, s.train.len / s.original.dim, @errorName(err) });
                return err;
            };
            defer selected.deinit();
            const chunks = try a.alloc(joint.FrozenChunk, 1);
            errdefer a.free(chunks);
            const ranges = try a.dupe(admission.Range, &.{range});
            errdefer a.free(ranges);
            const indices = try a.alloc([]usize, 1);
            errdefer a.free(indices);
            const weights = try a.alloc([]f64, 1);
            errdefer a.free(weights);
            indices[0] = try a.alloc(usize, selected.indices.len + selected.exact_core_local.len);
            errdefer a.free(indices[0]);
            @memcpy(indices[0][0..selected.indices.len], selected.indices);
            @memcpy(indices[0][selected.indices.len..], selected.exact_core_local);
            weights[0] = try a.dupe(f64, selected.weights);
            errdefer a.free(weights[0]);
            chunks[0] = .{ .start = range.start, .rows = range.rows, .selected_local = indices[0][0..selected.indices.len], .weights = weights[0], .exact_core_local = indices[0][selected.indices.len..] };
            s.streamed_gram_calls = try std.math.add(usize, s.streamed_gram_calls, 1);
            s.streamed_gram_probability_values = try std.math.add(usize, s.streamed_gram_probability_values, try std.math.mul(usize, s.train.len / s.original.dim, range.rows));
            s.streamed_gram_workspace_peak_bytes = @max(s.streamed_gram_workspace_peak_bytes, s.builder_transient_bound_bytes);
            s.streamed_gram_hasher.update(&authority.probability_sha256);
            s.streamed_gram_hasher.update(std.mem.sliceAsBytes(selected.indices));
            s.cache_counters.selection_hits += 1;
            break :blk replay.FrozenOwned{ .allocator = a, .chunks = chunks, .ranges = ranges, .indices = indices, .weights = weights };
        } else if (s.streamed_gram_selection) blk: {
            const Copy = struct {
                parent: *Context,
                range: admission.Range,
                fn one(raw_context: ?*anyopaque, query: usize, out: []f64, query_control: solver.Control) !void {
                    const self: *@This() = @ptrCast(@alignCast(raw_context orelse return error.InvalidCallbackContext));
                    return copyBoundRangePage(self.parent, self.range.start, self.range.rows, query, 1, out, query_control);
                }
            };
            var copy: Copy = .{ .parent = s, .range = range };
            var selected = streamed_gram.select(a, .{ .context = &copy, .query_count = s.train.len / s.original.dim, .row_count = range.rows, .copy_query_fn = Copy.one }, .{ .target_rows = target, .numerical_policy = .normalized_probability_raw_v1, .query_batch_rows = s.tile_queries, .max_sweeps = cfg.max_sweeps, .nnls_tolerance = cfg.nnls_tolerance, .external_live_bytes = 0, .combined_cap = s.builder_transient_bound_bytes, .control = control }) catch |err| {
                if (err == error.MassFitDidNotConverge or err == error.InsufficientSelectionProgress) std.debug.print("attention_match_bounded_page_selection phase=streamed_gram range_start={d} range_rows={d} target_rows={d} query_count={d} error={s} diagnostic_only=true\n", .{ range.start, range.rows, target, s.train.len / s.original.dim, @errorName(err) });
                return err;
            };
            errdefer selected.deinit();
            const chunks = try a.alloc(joint.FrozenChunk, 1);
            errdefer a.free(chunks);
            const ranges = try a.dupe(admission.Range, &.{range});
            errdefer a.free(ranges);
            const indices = try a.alloc([]usize, 1);
            errdefer a.free(indices);
            const weights = try a.alloc([]f64, 1);
            errdefer a.free(weights);
            indices[0] = selected.indices;
            weights[0] = selected.weights;
            chunks[0] = .{ .start = range.start, .rows = range.rows, .selected_local = selected.indices, .weights = selected.weights };
            s.streamed_gram_calls = try std.math.add(usize, s.streamed_gram_calls, 1);
            s.streamed_gram_probability_values = try std.math.add(usize, s.streamed_gram_probability_values, selected.evidence.probability_values_consumed);
            s.streamed_gram_workspace_peak_bytes = @max(s.streamed_gram_workspace_peak_bytes, selected.evidence.workspace_bytes);
            s.streamed_gram_hasher.update(&.{@backingInt(selected.evidence.numerical_policy)});
            s.streamed_gram_hasher.update(&selected.evidence.probability_sha256);
            s.streamed_gram_hasher.update(&selected.evidence.row_maxima_sha256);
            s.cache_counters.selection_hits += 1;
            break :blk replay.FrozenOwned{ .allocator = a, .chunks = chunks, .ranges = ranges, .indices = indices, .weights = weights };
        } else blk: {
            var selection_cfg = cfg;
            selection_cfg.backend = null;
            selection_cfg.value_backend = null;
            break :blk try replay.freezeBranch(a, s.original, s.train, &.{range}, 1, &.{target}, selection_cfg, cached, &s.cache_counters);
        };
        errdefer frozen.deinit();
        const frozen_bytes = try replay.frozenBytes(frozen);
        if (frozen_bytes >= cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        cfg.max_workspace_bytes -= frozen_bytes;
        var fixed: ?joint.FixedOriginalTerm = null;
        defer if (fixed) |*owned| owned.deinit();
        if (s.critical_core_policy != .disabled) {
            const fixed_bytes = try joint.fixedOriginalTermBytes(s.train.len / s.original.dim, s.original.dim);
            if (fixed_bytes >= cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
            fixed = try joint.prepareFixedOriginalTerm(a, s.original, s.train, frozen.chunks[0], cfg, cached, frozen_bytes, s.builder_transient_bound_bytes);
            cfg.max_workspace_bytes -= fixed_bytes;
        }
        var fitted = try joint.fitStrategyWithFixedOriginal(a, s.original, s.train, frozen.chunks, cfg, cached, &s.cache_counters, .identity_chunk_coefficients_v18, false, if (fixed) |*owned| owned else null);
        errdefer fitted.deinit();
        if (frozen.chunks.len != 1 or frozen.chunks[0].selected_local.len != target or fitted.chunks.len != 1 or
            fitted.chunks[0].selected_count != target or fitted.bias.len != target or
            fitted.values.len != try std.math.mul(usize, target, s.original.dim)) return error.InvalidAttentionShape;
        try replay.round16(fitted.values);
        try replay.round32(fitted.bias);
        const o = try a.create(Owned);
        o.* = .{ .frozen = frozen, .fitted = fitted, .range = range, .target = target };
        return .{ .context = o, .retained_bytes = try retained(s, range, target), .selected_rows = target, .destroy_fn = destroy };
    }
    fn validateY(raw: ?*anyopaque, y: []const f64, control: solver.Control) !void {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        if (y.len != s.normalizer.normalizer.outputs.len) return error.InvalidAttentionShape;
        for (y, s.normalizer.normalizer.outputs, 0..) |wide, narrow, i| {
            if ((i & 4095) == 0) try control.check();
            if (!std.math.isFinite(wide) or wide != @as(f64, narrow)) return error.InvalidFitConfig;
        }
    }
    fn scoreWorkspace(raw: ?*anyopaque, range: admission.Range, _: usize) !usize {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        if (s.tile_queries == 0 or s.page_scratch.len < try std.math.mul(usize, s.tile_queries, range.rows)) return error.CompactionWorkspaceLimit;
        var bytes = try std.math.add(usize, try std.math.mul(usize, s.page_scratch.len, @sizeOf(f64)), try s.fitter.capturedPageTransientBytes(s.normalizer, range.rows, s.tile_queries));
        if (s.score_policy == .accelerate_fp64_tiled_v1) {
            if (s.accelerated_scratch.len < try std.math.mul(usize, s.tile_queries, range.rows)) return error.CompactionWorkspaceLimit;
        }
        if (s.critical_authority) |authority| {
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, authority.local_keys.len, @sizeOf(usize)));
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, authority.query_offsets.len, @sizeOf(usize)));
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, authority.critical_queries.len, @sizeOf(usize)));
        }
        return bytes;
    }
    fn score(raw: ?*anyopaque, artifact: *const bounded.Artifact, range: admission.Range, _: usize, rm: []f64, rn: []f64, cm: []f64, cn: []f64, control: solver.Control) !void {
        const s: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const o: *const Owned = @ptrCast(@alignCast(artifact.context orelse return error.InvalidCallbackContext));
        if (o.range.start != range.start or o.range.rows != range.rows or artifact.selected_rows != o.target or
            o.frozen.chunks.len != 1 or o.fitted.chunks.len != 1 or s.original.bias == null) return error.InvalidAttentionShape;
        const d = s.original.dim;
        const qd = try std.math.mul(usize, rm.len, d);
        if (rn.len != qd or cm.len != rm.len or cn.len != rn.len or o.fitted.bias.len != o.target or
            o.fitted.values.len != try std.math.mul(usize, o.target, d)) return error.InvalidAttentionShape;
        const chunk = o.frozen.chunks[0];
        const cached = try s.ensureRangeCache(range, control);
        if (s.score_policy == .accelerate_fp64_tiled_v1) {
            const source_start = try std.math.mul(usize, range.start, d);
            const source_len = try std.math.mul(usize, range.rows, d);
            if (source_start > s.original.values.len or source_len > s.original.values.len - source_start) return error.InvalidAttentionShape;
            if (s.accelerated_scratch.len < try std.math.mul(usize, s.tile_queries, range.rows)) return error.CompactionWorkspaceLimit;
            var first: usize = 0;
            while (first < rm.len) {
                try control.check();
                s.accelerated_score_cancellation_checks = try std.math.add(usize, s.accelerated_score_cancellation_checks, 1);
                const count = @min(s.tile_queries, rm.len - first);
                const page_len = try std.math.mul(usize, count, range.rows);
                const selected_len = try std.math.mul(usize, count, o.target);
                if (page_len > s.page_scratch.len or selected_len > s.accelerated_scratch.len) return error.CompactionWorkspaceLimit;
                const page = s.page_scratch[0..page_len];
                const selected = s.accelerated_scratch[0..selected_len];
                for (0..count) |local_query| try cached.copyChunk(first + local_query, range.start, page[local_query * range.rows ..][0..range.rows]);
                const source_values = s.original.values[source_start..][0..source_len];
                const original_bias = s.original.bias.?[range.start..][0..range.rows];
                try acceleratedTileReductions(page, source_values, chunk.selected_local, o.fitted.bias, original_bias, o.fitted.values, count, range.rows, d, rm[first..][0..count], rn[first * d ..][0 .. count * d], cm[first..][0..count], cn[first * d ..][0 .. count * d], selected);
                s.accelerated_score_dgemm_calls = try std.math.add(usize, s.accelerated_score_dgemm_calls, 2);
                try control.check();
                s.accelerated_score_cancellation_checks = try std.math.add(usize, s.accelerated_score_cancellation_checks, 1);
                s.accelerated_score_tiles = try std.math.add(usize, s.accelerated_score_tiles, 1);
                first += count;
            }
            s.accelerated_score_calls = try std.math.add(usize, s.accelerated_score_calls, 1);
            return;
        }
        var first: usize = 0;
        while (first < rm.len) {
            try control.check();
            const count = @min(s.tile_queries, rm.len - first);
            const page_len = try std.math.mul(usize, count, range.rows);
            if (page_len > s.page_scratch.len) return error.CompactionWorkspaceLimit;
            const page = s.page_scratch[0..page_len];
            for (0..count) |local_query| try cached.copyChunk(first + local_query, range.start, page[local_query * range.rows ..][0..range.rows]);
            for (0..count) |lq| {
                const qi = first + lq;
                const page_base = try std.math.mul(usize, lq, range.rows);
                const query_base = try std.math.mul(usize, qi, d);
                for (page[page_base..][0..range.rows], 0..) |p, local| {
                    if (!std.math.isFinite(p) or p < 0) return error.NonFiniteAttention;
                    const global = try std.math.add(usize, range.start, local);
                    if (global >= s.original.rows) return error.InvalidAttentionShape;
                    rm[qi] += p;
                    const global_base = try std.math.mul(usize, global, d);
                    for (0..d) |di| rn[query_base + di] += p * s.original.values[global_base + di];
                }
                for (chunk.selected_local, 0..) |local, si| {
                    if (local >= range.rows or si >= o.target) return error.InvalidAttentionShape;
                    const global = try std.math.add(usize, range.start, local);
                    if (global >= s.original.rows) return error.InvalidAttentionShape;
                    const p = try scaledProbability(page[page_base + local], o.fitted.bias[si], s.original.bias.?[global]);
                    cm[qi] += p;
                    const selected_base = try std.math.mul(usize, si, d);
                    for (0..d) |di| cn[query_base + di] += p * o.fitted.values[selected_base + di];
                }
            }
            first += count;
        }
    }
    pub fn builder(s: *Context) bounded.Builder {
        return .{ .context = s, .retained_bytes_fn = retained, .transient_bytes_fn = transient, .build_fn = build, .core_aware_targets = s.critical_core_policy != .disabled, .target_rows_fn = if (s.critical_core_policy != .disabled) targetRows else null };
    }
    pub fn scorer(s: *Context) bounded.PageScorer {
        return .{ .context = s, .execution_policy = if (s.score_policy == .accelerate_fp64_tiled_v1) bounded.accelerate_execution_policy else bounded.scalar_execution_policy, .validate_original_y_fn = validateY, .workspace_fn = scoreWorkspace, .score_fn = score };
    }
};

test "borrowed range fit honors explicit nonzero consumer cap" {
    const head: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0.25, -0.5 };
    const outputs = [_]f64{ 3.5, 3.5 };
    const frozen = [_]joint.FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} }};
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };

    const too_small_probabilities = try std.testing.allocator.dupe(f32, &.{ 0.5, 0.5, 0.5, 0.5 });
    var too_small = try training_source.initBorrowedRangeF32(std.testing.allocator, head, &train, cfg, 0, 2, too_small_probabilities, @constCast(outputs[0..]), 1);
    defer too_small.deinit();
    try std.testing.expectError(error.CompactionWorkspaceLimit, joint.fitStrategy(std.testing.allocator, head, &train, &frozen, cfg, &too_small, null, .identity_chunk_coefficients_v18, false));

    const admitted_probabilities = try std.testing.allocator.dupe(f32, &.{ 0.5, 0.5, 0.5, 0.5 });
    var admitted = try training_source.initBorrowedRangeF32(std.testing.allocator, head, &train, cfg, 0, 2, admitted_probabilities, @constCast(outputs[0..]), cfg.max_workspace_bytes);
    defer admitted.deinit();
    var fitted = try joint.fitStrategy(std.testing.allocator, head, &train, &frozen, cfg, &admitted, null, .identity_chunk_coefficients_v18, false);
    defer fitted.deinit();
    try std.testing.expectEqual(@as(usize, 1), fitted.chunks.len);
}

/// Transfers a completely scored bounded plan into the existing replay-owned
/// candidate representation, then runs all three page-backed tiled geometry
/// branches. No probability matrix or Gram representation is materialized.
pub fn finishGeometry(
    allocator: std.mem.Allocator,
    plan: *bounded.Plan,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    tile_queries: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    context: *Context,
    control: solver.Control,
) !replay.AdaptivePlan {
    return finishGeometryWithBudgetPercent(allocator, plan, ranges, original_y, query_count, dim, tile_queries, 100, external_live_bytes, combined_cap, context, control);
}

pub fn finishGeometryWithBudgetPercent(
    allocator: std.mem.Allocator,
    plan: *bounded.Plan,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    tile_queries: usize,
    geometry_budget_percent: u8,
    external_live_bytes: usize,
    combined_cap: usize,
    context: *Context,
    control: solver.Control,
) !replay.AdaptivePlan {
    var plan_consumed = false;
    defer if (!plan_consumed) plan.deinit();
    if (plan.scores.len != ranges.len or plan.artifacts.len != ranges.len) return error.InvalidAttentionShape;
    // The scoring phase admitted these objects, but they remain live during
    // the separate transfer and geometry phases. Charge both descriptor
    // tables at the transfer peak, then retain the destination table, scores,
    // and every builder-authenticated packet in the geometry external bound.
    const authority = try geometryAuthorityBytes(ranges.len, plan.retained_artifact_bytes, external_live_bytes);
    if (authority.transfer_peak > combined_cap) return error.CompactionWorkspaceLimit;
    try control.check();
    const scored = try allocator.alloc([3]?replay.ScoredCandidate, ranges.len);
    errdefer allocator.free(scored);
    @memset(scored, .{ null, null, null });
    var moved: usize = 0;
    errdefer for (scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
    for (plan.artifacts, scored) |*source_levels, *dest_levels| for (source_levels, dest_levels) |*source, *dest| {
        const artifact = source.* orelse return error.InvalidAttentionShape;
        const owned: *Owned = @ptrCast(@alignCast(artifact.context orelse return error.InvalidCallbackContext));
        dest.* = .{ .frozen = owned.frozen, .fitted = owned.fitted };
        allocator.destroy(owned);
        source.* = null;
        moved += 1;
    };
    if (moved != ranges.len * 3) return error.InvalidAttentionShape;
    const scores = plan.scores;
    plan.allocator.free(plan.artifacts);
    plan.* = undefined;
    plan_consumed = true;
    var view: replay.AdaptivePlan = .{ .allocator = allocator, .scores = scores, .scored = scored, .branches = undefined, .geometry_policy = if (geometry_budget_percent == 100) tiled_bridge.policy else tiled_bridge.budget_percent_policy, .geometry_storage = .dense, .geometry_budget_percent = geometry_budget_percent };
    errdefer {
        for (view.scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
        allocator.free(view.scored);
        allocator.free(view.scores);
    }
    var page_context: tiled_bridge.PageContext = .{ .fitter = context.fitter, .normalizer = context.normalizer, .original = context.original, .train_queries = context.train, .scored = &view, .scratch = context.page_scratch, .maximum_range_rows = blk: {
        var n: usize = 0;
        for (ranges) |range| n = @max(n, range.rows);
        break :blk n;
    }, .cached_page_context = context, .copy_cached_page_fn = Context.copyBoundRangePage, .delta_backend = context.delta_backend, .require_accelerated_delta = context.require_accelerated_delta, .ratio_scratch = context.ratio_scratch };
    var geometry = try tiled_bridge.chooseFromScoredFloatPairWithBudgetPercent(allocator, ranges, original_y, query_count, dim, tile_queries, geometry_budget_percent, authority.external, combined_cap, &page_context, control);
    context.delta_evidence = page_context.delta_evidence;
    view.branches = geometry.branches;
    view.geometry_workspace_bound_bytes = geometry.workspace_bound_bytes;
    geometry = undefined;
    return view;
}

/// Rebuild only geometry from already-frozen scoring packets. Ownership of
/// `scores` and `scored` transfers to the returned plan (or is released on
/// error); scoring and value fitting are never repeated.
pub fn replanFrozenScoringWithBudgetPercent(
    allocator: std.mem.Allocator,
    scores: []adaptive.RangeScores,
    scored: [][3]?replay.ScoredCandidate,
    retained_artifact_bytes: usize,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    tile_queries: usize,
    geometry_budget_percent: u8,
    external_live_bytes: usize,
    combined_cap: usize,
    context: *Context,
    control: solver.Control,
) !replay.AdaptivePlan {
    var inputs_owned = true;
    errdefer if (inputs_owned) {
        for (scored) |*levels| for (levels) |*candidate| if (candidate.*) |*owned| owned.deinit();
        allocator.free(scored);
        allocator.free(scores);
    };
    const authority = try geometryAuthorityBytes(ranges.len, retained_artifact_bytes, external_live_bytes);
    var view: replay.AdaptivePlan = .{ .allocator = allocator, .scores = scores, .scored = scored, .branches = undefined, .geometry_policy = if (geometry_budget_percent == 100) tiled_bridge.policy else tiled_bridge.budget_percent_policy, .geometry_storage = .dense, .geometry_budget_percent = geometry_budget_percent };
    var page_context: tiled_bridge.PageContext = .{ .fitter = context.fitter, .normalizer = context.normalizer, .original = context.original, .train_queries = context.train, .scored = &view, .scratch = context.page_scratch, .maximum_range_rows = blk: {
        var n: usize = 0;
        for (ranges) |range| n = @max(n, range.rows);
        break :blk n;
    }, .cached_page_context = context, .copy_cached_page_fn = Context.copyBoundRangePage, .delta_backend = context.delta_backend, .require_accelerated_delta = context.require_accelerated_delta, .ratio_scratch = context.ratio_scratch };
    var geometry = try tiled_bridge.chooseFromScoredFloatPairWithBudgetPercent(allocator, ranges, original_y, query_count, dim, tile_queries, geometry_budget_percent, authority.external, combined_cap, &page_context, control);
    var geometry_live = true;
    errdefer if (geometry_live) geometry.deinit();
    context.delta_evidence.tiles = try capAdd(context.delta_evidence.tiles, page_context.delta_evidence.tiles);
    context.delta_evidence.backend_calls = try capAdd(context.delta_evidence.backend_calls, page_context.delta_evidence.backend_calls);
    context.delta_evidence.packed_bytes_peak = @max(context.delta_evidence.packed_bytes_peak, page_context.delta_evidence.packed_bytes_peak);
    context.delta_evidence.staged_bytes_peak = @max(context.delta_evidence.staged_bytes_peak, page_context.delta_evidence.staged_bytes_peak);
    context.delta_evidence.fallbacks = try capAdd(context.delta_evidence.fallbacks, page_context.delta_evidence.fallbacks);
    view.branches = geometry.branches;
    view.geometry_workspace_bound_bytes = geometry.workspace_bound_bytes;
    geometry_live = false;
    geometry = undefined;
    inputs_owned = false;
    return view;
}

test "fallback replan owns frozen scoring before authority overflow" {
    const allocator = std.testing.allocator;
    const scores = try allocator.alloc(adaptive.RangeScores, 1);
    const scored = try allocator.alloc([3]?replay.ScoredCandidate, 1);
    scored[0] = .{ null, null, null };
    var context: Context = undefined;
    try std.testing.expectError(error.CompactionWorkspaceLimit, replanFrozenScoringWithBudgetPercent(
        allocator,
        scores,
        scored,
        std.math.maxInt(usize),
        &.{.{ .start = 0, .rows = 1 }},
        &.{},
        1,
        1,
        1,
        95,
        1,
        std.math.maxInt(usize),
        &context,
        .{},
    ));
}

test "geometry transfer rejects cap and cancellation before destination allocation and unwinds source" {
    const Fake = struct {
        destroyed: *usize,

        fn destroy(raw: ?*anyopaque, allocator: std.mem.Allocator) void {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return));
            self.destroyed.* += 1;
            allocator.destroy(self);
        }

        fn plan(destroyed: *usize) !bounded.Plan {
            const scores = try std.testing.allocator.alloc(adaptive.RangeScores, 1);
            errdefer std.testing.allocator.free(scores);
            const artifacts = try std.testing.allocator.alloc([3]?bounded.Artifact, 1);
            errdefer std.testing.allocator.free(artifacts);
            artifacts[0] = .{ null, null, null };
            errdefer for (&artifacts[0]) |*entry| if (entry.*) |*owned| owned.deinit(std.testing.allocator);
            for (&artifacts[0]) |*entry| {
                const owner = try std.testing.allocator.create(@This());
                owner.* = .{ .destroyed = destroyed };
                entry.* = .{ .context = owner, .retained_bytes = 100, .selected_rows = 1, .destroy_fn = destroy };
            }
            return .{ .allocator = std.testing.allocator, .scores = scores, .artifacts = artifacts, .retained_artifact_bytes = 300, .workspace_bound_bytes = 0, .score_completions = 0, .score_query_rows = 0, .execution_policy = bounded.scalar_execution_policy };
        }
    };
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const ranges = [_]admission.Range{.{ .start = 0, .rows = 4 }};
    const authority = try geometryAuthorityBytes(1, 300, 17);
    var unused_context: Context = undefined;

    var destroyed: usize = 0;
    var cap_plan = try Fake.plan(&destroyed);
    var failing_cap = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.CompactionWorkspaceLimit, finishGeometry(failing_cap.allocator(), &cap_plan, &ranges, &.{}, 1, 1, 1, 17, authority.transfer_peak - 1, &unused_context, .{}));
    try std.testing.expectEqual(@as(usize, 3), destroyed);

    destroyed = 0;
    var cancel_plan = try Fake.plan(&destroyed);
    var failing_cancel = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.Cancelled, finishGeometry(failing_cancel.allocator(), &cancel_plan, &ranges, &.{}, 1, 1, 1, 17, authority.transfer_peak, &unused_context, .{ .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 3), destroyed);
}

test "range cache loader unwinds OOM and cancellation and replaces prior range" {
    const Fake = struct {
        calls: usize = 0,
        fn fill(raw: ?*anyopaque, range: admission.Range, first: usize, count: usize, out: []f64, control: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (out.len != count * range.rows) return error.InvalidAttentionShape;
            for (0..count) |local_query| for (0..range.rows) |local_row| {
                try control.check();
                out[local_query * range.rows + local_row] = @as(f64, @floatFromInt((first + local_query + 1) * 10 + range.start + local_row)) / 100;
            };
        }
    };
    const Cancel = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 3) return error.Cancelled;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 1, 2, 3, 4 }, .values = &.{ 5, 6, 7, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0.25, -0.5, 0.75 };
    var y = [_]f64{ 1, 2, 3 };
    var scratch: [4]f64 = undefined;
    var fake: Fake = .{};
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var context: Context = .{ .fitter = undefined, .normalizer = undefined, .original = head, .train = &train, .cfg = .{ .target_rows = 1 }, .tile_queries = 2, .builder_transient_bound_bytes = 1 << 20, .page_scratch = &scratch, .allocator = failing.allocator(), .original_y = &y, .range_cache_reserved_bytes = 3 * 2 * @sizeOf(f32), .cache_batch_rows = 16, .range_fill_context = &fake, .range_fill_fn = Fake.fill };
    defer context.deinit();
    try std.testing.expectError(error.OutOfMemory, context.ensureRangeCache(.{ .start = 0, .rows = 2 }, .{}));
    try std.testing.expect(context.range_cache == null);
    context.allocator = std.testing.allocator;
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, context.ensureRangeCache(.{ .start = 0, .rows = 2 }, .{ .context = &cancel, .check_fn = Cancel.check }));
    try std.testing.expect(context.range_cache == null);
    const first = try context.ensureRangeCache(.{ .start = 0, .rows = 2 }, .{});
    try std.testing.expectEqual(@as(f64, @floatCast(@as(f32, 0.1))), try first.probability(0, 0));
    _ = try context.ensureRangeCache(.{ .start = 2, .rows = 2 }, .{});
    try std.testing.expectEqual(@as(usize, 2), context.cache_counters.prepare_attempts);
    const provider_calls_after_scoring = fake.calls;
    var copied: [4]f64 = undefined;
    try Context.copyBoundRangePage(&context, 2, 2, 0, 2, &copied, .{});
    try std.testing.expectEqualSlices(f64, &.{
        @as(f64, @floatCast(@as(f32, 0.12))),
        @as(f64, @floatCast(@as(f32, 0.13))),
        @as(f64, @floatCast(@as(f32, 0.22))),
        @as(f64, @floatCast(@as(f32, 0.23))),
    }, &copied);
    // Same-range geometry access is an exact f32 widening and performs no
    // provider work. This is the central Source180 reuse invariant.
    try std.testing.expectEqual(provider_calls_after_scoring, fake.calls);
    try std.testing.expectEqual(@as(usize, 1), context.geometry_cache_requests);
    try std.testing.expectEqual(@as(usize, 1), context.geometry_cache_hits);
    try std.testing.expectEqual(@as(usize, 0), context.geometry_cache_loads);
    var cancelled_copy: Cancel = .{ .checks = 2 };
    try std.testing.expectError(error.Cancelled, Context.copyBoundRangePage(&context, 2, 2, 0, 1, copied[0..2], .{ .context = &cancelled_copy, .check_fn = Cancel.check }));
    try std.testing.expectEqual(@as(usize, 1), context.geometry_cache_requests);
    try std.testing.expectEqual(provider_calls_after_scoring, fake.calls);
    try Context.copyBoundRangePage(&context, 0, 2, 0, 1, copied[0..2], .{});
    try std.testing.expectEqual(@as(usize, 3), context.cache_counters.prepare_attempts);
    try std.testing.expectEqual(@as(usize, 3), context.cache_counters.builds);
    try std.testing.expectEqual(@as(usize, 2), context.geometry_cache_requests);
    try std.testing.expectEqual(@as(usize, 1), context.geometry_cache_hits);
    try std.testing.expectEqual(@as(usize, 1), context.geometry_cache_loads);
    try std.testing.expectEqual(@as(usize, 0), context.range_cache_start);
    try std.testing.expectError(error.InvalidAttentionShape, context.range_cache.?.probability(0, 2));

    // Model the exact cache-access nesting of three retained geometry
    // branches over two ranges and three candidate levels. Every range's first
    // level loads once; the next two levels are exact cache hits. This proves
    // legitimate geometry loads exceed R while final accounting remains
    // prepares = scoring-R + geometry-loads and requests = hits + loads.
    for (0..3) |_| for ([_]usize{ 0, 2 }) |range_start| for (0..3) |_| {
        try Context.copyBoundRangePage(&context, range_start, 2, 0, 1, copied[0..2], .{});
    };
    try std.testing.expect(context.geometry_cache_loads > 2);
    try std.testing.expectEqual(context.geometry_cache_requests, context.geometry_cache_hits + context.geometry_cache_loads);
    try std.testing.expectEqual(@as(usize, 2) + context.geometry_cache_loads, context.cache_counters.prepare_attempts);
    try std.testing.expectEqual(context.cache_counters.prepare_attempts, context.cache_counters.builds);
}
