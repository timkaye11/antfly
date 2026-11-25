const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const selection = @import("attention_match_global_selection_diagnostic.zig");
const production_selection = @import("attention_match_global_selection.zig");
const builtin = @import("builtin");
const streamed = @import("attention_match_streamed_gram_selection.zig");
const critical = @import("attention_match_train_critical_source_keys_streaming.zig");
const range_critical = @import("attention_match_train_critical_range_streaming.zig");
const calibrated = @import("attention_match_train_budget_calibrated_critical_threshold.zig");
const calibrated_v2 = @import("attention_match_train_budget_calibrated_critical_threshold_reserve16.zig");

pub const Policy = enum { disabled, train_probability_gte_0_025_no_top1_v1, train_probability_budget_calibrated_floor_0_0025_p91_v1, train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 };

pub fn policyDigest(policy: Policy) [32]u8 {
    @setEvalBranchQuota(10000);
    if (policy == .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2) return calibrated_v2.policyDigest();
    if (policy == .train_probability_budget_calibrated_floor_0_0025_p91_v1) return calibrated.policyDigest();
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(switch (policy) {
        .disabled => "disabled",
        .train_probability_gte_0_025_no_top1_v1 => "train_probability_gte_0_025_no_top1_v1|threshold=0.025|inclusive=true|top1=false|exact_core_original_k_v_bias=true|residual_fit_only=true",
        .train_probability_budget_calibrated_floor_0_0025_p91_v1 => unreachable,
        .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => unreachable,
    });
    return hash.finalResult();
}

pub fn authorityBytes(authority: *const critical.Result) !usize {
    var bytes = try std.math.mul(usize, authority.inner.ranges.len, @sizeOf(critical.RangeKeys));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, authority.inner.source_keys.len, @sizeOf(usize)));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, authority.inner.query_offsets.len, @sizeOf(usize)));
    return std.math.add(usize, bytes, try std.math.mul(usize, authority.inner.critical_queries.len, @sizeOf(usize)));
}

pub const OwnedSelection = struct {
    allocator: std.mem.Allocator,
    indices: []usize,
    weights: []f64,
    exact_core_local: []usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.exact_core_local);
        self.allocator.free(self.weights);
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn rangeCore(a: std.mem.Allocator, authority: *const critical.Result, range: admission.Range, target: usize) ![]usize {
    if (!std.math.isFinite(authority.inner.threshold) or authority.inner.threshold < calibrated.floor_probability or authority.inner.threshold > calibrated.ceiling_probability or authority.inner.complete_query_rows != authority.inner.query_rows) return error.InvalidFitConfig;
    for (authority.inner.ranges) |entry| if (entry.start == range.start and entry.rows == range.rows) {
        if (entry.key_begin > authority.inner.source_keys.len or entry.key_count > authority.inner.source_keys.len - entry.key_begin or entry.key_count > target) return error.CompactionFitRejected;
        const out = try a.alloc(usize, entry.key_count);
        errdefer a.free(out);
        for (authority.inner.source_keys[entry.key_begin..][0..entry.key_count], out, 0..) |global, *local, i| {
            if (global < range.start or global >= range.start + range.rows) return error.InvalidAttentionShape;
            local.* = global - range.start;
            if (i != 0 and out[i - 1] >= local.*) return error.InvalidAttentionShape;
        }
        return out;
    };
    return error.InvalidAttentionShape;
}

fn residualLocals(a: std.mem.Allocator, rows: usize, core: []const usize) ![]usize {
    const out = try a.alloc(usize, rows - core.len);
    errdefer a.free(out);
    var ci: usize = 0;
    var oi: usize = 0;
    for (0..rows) |local| {
        if (ci < core.len and core[ci] == local) {
            ci += 1;
        } else {
            out[oi] = local;
            oi += 1;
        }
    }
    if (ci != core.len or oi != out.len) return error.InvalidAttentionShape;
    return out;
}

fn compose(a: std.mem.Allocator, core: []usize, residual_map: []const usize, residual_indices: []const usize, residual_weights: []const f64, target: usize) !OwnedSelection {
    if (residual_indices.len != residual_weights.len or core.len + residual_indices.len != target) return error.InvalidAttentionShape;
    const indices = try a.alloc(usize, target);
    errdefer a.free(indices);
    const weights = try a.alloc(f64, target);
    errdefer a.free(weights);
    var ci: usize = 0;
    var ri: usize = 0;
    for (0..target) |out| {
        const residual_local = if (ri < residual_indices.len) residual_map[residual_indices[ri]] else std.math.maxInt(usize);
        const core_local = if (ci < core.len) core[ci] else std.math.maxInt(usize);
        if (core_local < residual_local) {
            indices[out] = core_local;
            weights[out] = 1;
            ci += 1;
        } else {
            if (residual_local == core_local or residual_local == std.math.maxInt(usize)) return error.InvalidAttentionShape;
            indices[out] = residual_local;
            weights[out] = residual_weights[ri];
            ri += 1;
        }
    }
    if (ci != core.len or ri != residual_indices.len) return error.InvalidAttentionShape;
    return .{ .allocator = a, .indices = indices, .weights = weights, .exact_core_local = core };
}

pub fn selectCached(a: std.mem.Allocator, original: solver.Head, train_source: *const training_source.Source, range: admission.Range, target: usize, cfg: solver.Config, authority: *const critical.Result, policy: selection.SelectionPolicy) !OwnedSelection {
    try original.validate();
    const core = try rangeCore(a, authority, range, target);
    errdefer a.free(core);
    const residual_target = target - core.len;
    if (residual_target == 0) return compose(a, core, &.{}, &.{}, &.{}, target);
    const map = try residualLocals(a, range.rows, core);
    defer a.free(map);
    if (residual_target >= map.len) return error.InvalidFitConfig;
    const q = train_source.query_count;
    const matrix_count = try std.math.mul(usize, q, map.len);
    var retained = try std.math.mul(usize, try std.math.add(usize, matrix_count, q), @sizeOf(f64));
    retained = try std.math.add(usize, retained, try std.math.mul(usize, try std.math.add(usize, try std.math.add(usize, core.len, map.len), range.rows), @sizeOf(usize)));
    if (retained > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const matrix = try a.alloc(f64, matrix_count);
    defer a.free(matrix);
    const rho = try a.alloc(f64, q);
    defer a.free(rho);
    const full = try a.alloc(f64, range.rows);
    defer a.free(full);
    for (0..q) |qi| {
        try cfg.control.check();
        try train_source.copyChunk(qi, range.start, full);
        var sum: f64 = 0;
        for (map, matrix[qi * map.len ..][0..map.len]) |local, *out| {
            out.* = full[local];
            sum += out.*;
        }
        if (!std.math.isFinite(sum) or sum < 0) return error.NonFiniteAttention;
        rho[qi] = sum;
    }
    var nested = cfg;
    nested.target_rows = residual_target;
    nested.max_workspace_bytes -= retained;
    var residual = if (policy == .raw)
        try production_selection.selectFeaturesRaw(a, matrix, rho, map.len, nested)
    else if (comptime builtin.is_test)
        try selection.selectFeaturesNamedPolicy(a, matrix, rho, map.len, nested, policy)
    else
        return error.InvalidFitConfig;
    defer residual.deinit();
    return compose(a, core, map, residual.indices, residual.weights, target);
}

pub const PageSource = struct {
    context: ?*anyopaque,
    query_count: usize,
    range: admission.Range,
    copy_full_fn: *const fn (?*anyopaque, usize, []f64, solver.Control) anyerror!void,
};

pub fn selectStreamed(a: std.mem.Allocator, source: PageSource, target: usize, authority: *const critical.Result, cfg: streamed.Config) !OwnedSelection {
    const core = try rangeCore(a, authority, source.range, target);
    errdefer a.free(core);
    const residual_target = target - core.len;
    if (residual_target == 0) return compose(a, core, &.{}, &.{}, &.{}, target);
    const map = try residualLocals(a, source.range.rows, core);
    defer a.free(map);
    if (residual_target >= map.len) return error.InvalidFitConfig;
    const Context = struct {
        source: PageSource,
        map: []const usize,
        full: []f64,
        fn copy(raw: ?*anyopaque, query: usize, out: []f64, control: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
            if (out.len != self.map.len) return error.InvalidAttentionShape;
            try self.source.copy_full_fn(self.source.context, query, self.full, control);
            for (self.map, out) |local, *value| value.* = self.full[local];
        }
    };
    const full = try a.alloc(f64, source.range.rows);
    defer a.free(full);
    var context: Context = .{ .source = source, .map = map, .full = full };
    var residual = try streamed.select(a, .{ .context = &context, .query_count = source.query_count, .row_count = map.len, .copy_query_fn = Context.copy }, .{
        .target_rows = residual_target,
        .numerical_policy = cfg.numerical_policy,
        .query_batch_rows = cfg.query_batch_rows,
        .max_sweeps = cfg.max_sweeps,
        .nnls_tolerance = cfg.nnls_tolerance,
        .external_live_bytes = try std.math.add(usize, cfg.external_live_bytes, try std.math.add(usize, try std.math.mul(usize, full.len, @sizeOf(f64)), try std.math.mul(usize, core.len + map.len, @sizeOf(usize)))),
        .combined_cap = cfg.combined_cap,
        .control = cfg.control,
    });
    defer residual.deinit();
    return compose(a, core, map, residual.indices, residual.weights, target);
}

pub fn selectStreamedRange(a: std.mem.Allocator, source: PageSource, target: usize, authority: *const range_critical.Result, cfg: streamed.Config) !OwnedSelection {
    if (authority.global_start != source.range.start or authority.range_rows != source.range.rows or authority.query_rows != source.query_count or authority.original_rows < source.range.start + source.range.rows or authority.local_keys.len > target) return error.InvalidFitConfig;
    const core = try a.dupe(usize, authority.local_keys);
    errdefer a.free(core);
    const residual_target = target - core.len;
    if (residual_target == 0) return compose(a, core, &.{}, &.{}, &.{}, target);
    const map = try residualLocals(a, source.range.rows, core);
    defer a.free(map);
    if (residual_target >= map.len) return error.InvalidFitConfig;
    const Context = struct {
        source: PageSource,
        map: []const usize,
        full: []f64,
        fn copy(raw: ?*anyopaque, query: usize, out: []f64, control: solver.Control) !void {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
            if (out.len != self.map.len) return error.InvalidAttentionShape;
            try self.source.copy_full_fn(self.source.context, query, self.full, control);
            for (self.map, out) |local, *value| value.* = self.full[local];
        }
    };
    const full = try a.alloc(f64, source.range.rows);
    defer a.free(full);
    var context: Context = .{ .source = source, .map = map, .full = full };
    var residual = try streamed.select(a, .{ .context = &context, .query_count = source.query_count, .row_count = map.len, .copy_query_fn = Context.copy }, .{
        .target_rows = residual_target,
        .numerical_policy = cfg.numerical_policy,
        .query_batch_rows = cfg.query_batch_rows,
        .max_sweeps = cfg.max_sweeps,
        .nnls_tolerance = cfg.nnls_tolerance,
        .external_live_bytes = try std.math.add(usize, cfg.external_live_bytes, try std.math.add(usize, try std.math.mul(usize, full.len, @sizeOf(f64)), try std.math.mul(usize, core.len + map.len, @sizeOf(usize)))),
        .combined_cap = cfg.combined_cap,
        .control = cfg.control,
    });
    defer residual.deinit();
    return compose(a, core, map, residual.indices, residual.weights, target);
}

test "cached critical adapter preserves sorted core and independently selected residual support" {
    const full = [_]f64{
        0.90, 0.01, 0.03, 0.02, 0.01, 0.03,
        0.85, 0.02, 0.04, 0.01, 0.02, 0.06,
    };
    const range: admission.Range = .{ .start = 1, .rows = 4 };
    var authority = try critical.extractDense(std.testing.allocator, .{ .values = &full, .query_rows = 2, .complete_query_rows = 2, .source_rows = 6, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = range.start, .rows = range.rows }}, .{ .combined_cap = 1 << 20 });
    defer authority.deinit();
    const head: solver.Head = .{ .keys = &.{ 0, 1, 2, 3, 4, 5 }, .values = &.{ 0, 1, 2, 3, 4, 5 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    const cached_probabilities = try std.testing.allocator.dupe(f32, &.{ 0.01, 0.03, 0.02, 0.01, 0.02, 0.04, 0.01, 0.02 });
    var outputs = [_]f64{ 0, 0 };
    const cfg: solver.Config = .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var source = try training_source.initBorrowedRangeF32(std.testing.allocator, head, &train, cfg, range.start, range.rows, cached_probabilities, &outputs, 1 << 20);
    defer source.deinit();
    var actual = try selectCached(std.testing.allocator, head, &source, range, 3, cfg, &authority, .raw);
    defer actual.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, actual.exact_core_local);
    for (actual.indices[1..], actual.indices[0 .. actual.indices.len - 1]) |value, prior| try std.testing.expect(prior < value);
    const core_slot = std.mem.indexOfScalar(usize, actual.indices, 1) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(f64, 1), actual.weights[core_slot]);
    const residual_matrix = [_]f64{ 0.01, 0.02, 0.01, 0.02, 0.01, 0.02 };
    var expected = try selection.selectFeaturesNamedPolicy(std.testing.allocator, &residual_matrix, &.{ 0.04, 0.05 }, 3, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 }, .raw);
    defer expected.deinit();
    const residual_map = [_]usize{ 0, 2, 3 };
    var ri: usize = 0;
    for (actual.indices, actual.weights) |local, weight| {
        if (local == 1) continue;
        try std.testing.expectEqual(residual_map[expected.indices[ri]], local);
        try std.testing.expectApproxEqAbs(expected.weights[ri], weight, 1e-7);
        ri += 1;
    }
    try std.testing.expectEqual(expected.indices.len, ri);
}

test "cached critical adapter core-only path publishes sorted unit cores" {
    const full = [_]f64{ 0.02, 0.03, 0.04, 0.91 };
    const range: admission.Range = .{ .start = 1, .rows = 2 };
    var authority = try critical.extractDense(std.testing.allocator, .{ .values = &full, .query_rows = 1, .complete_query_rows = 1, .source_rows = 4, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = range.start, .rows = range.rows }}, .{ .combined_cap = 1 << 20 });
    defer authority.deinit();
    const head: solver.Head = .{ .keys = &.{ 0, 1, 2, 3 }, .values = &.{ 0, 1, 2, 3 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const train = [_]f64{0};
    const probabilities = try std.testing.allocator.dupe(f32, &.{ 0.03, 0.04 });
    var outputs = [_]f64{0};
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var source = try training_source.initBorrowedRangeF32(std.testing.allocator, head, &train, cfg, range.start, range.rows, probabilities, &outputs, 1 << 20);
    defer source.deinit();
    var result = try selectCached(std.testing.allocator, head, &source, range, 2, cfg, &authority, .raw);
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.indices);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.exact_core_local);
    try std.testing.expectEqualSlices(f64, &.{ 1, 1 }, result.weights);
}

fn cachedAdapterAllocationFailure(a: std.mem.Allocator) !void {
    const full = [_]f64{ 0.90, 0.01, 0.03, 0.02, 0.01, 0.03, 0.85, 0.02, 0.04, 0.01, 0.02, 0.06 };
    const range: admission.Range = .{ .start = 1, .rows = 4 };
    var authority = try critical.extractDense(a, .{ .values = &full, .query_rows = 2, .complete_query_rows = 2, .source_rows = 6, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = range.start, .rows = range.rows }}, .{ .combined_cap = 1 << 20 });
    defer authority.deinit();
    const head: solver.Head = .{ .keys = &.{ 0, 1, 2, 3, 4, 5 }, .values = &.{ 0, 1, 2, 3, 4, 5 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    const probabilities = try a.dupe(f32, &.{ 0.01, 0.03, 0.02, 0.01, 0.02, 0.04, 0.01, 0.02 });
    var outputs = [_]f64{ 0, 0 };
    const cfg: solver.Config = .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    var source = try training_source.initBorrowedRangeF32(a, head, &train, cfg, range.start, range.rows, probabilities, &outputs, 1 << 20);
    defer source.deinit();
    var result = try selectCached(a, head, &source, range, 3, cfg, &authority, .raw);
    result.deinit();
}

test "cached critical adapter cap cancellation and allocation failures publish nothing" {
    const full = [_]f64{ 0.90, 0.01, 0.03, 0.02, 0.01, 0.03, 0.85, 0.02, 0.04, 0.01, 0.02, 0.06 };
    const range: admission.Range = .{ .start = 1, .rows = 4 };
    var authority = try critical.extractDense(std.testing.allocator, .{ .values = &full, .query_rows = 2, .complete_query_rows = 2, .source_rows = 6, .source_key_space = .global_original, .normalization = .full_original_softmax_post_softcap_bias, .storage = .exact_f32_widened }, &.{.{ .start = range.start, .rows = range.rows }}, .{ .combined_cap = 1 << 20 });
    defer authority.deinit();
    const head: solver.Head = .{ .keys = &.{ 0, 1, 2, 3, 4, 5 }, .values = &.{ 0, 1, 2, 3, 4, 5 }, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    const probabilities = try std.testing.allocator.dupe(f32, &.{ 0.01, 0.03, 0.02, 0.01, 0.02, 0.04, 0.01, 0.02 });
    var outputs = [_]f64{ 0, 0 };
    var source = try training_source.initBorrowedRangeF32(std.testing.allocator, head, &train, .{ .target_rows = 3 }, range.start, range.rows, probabilities, &outputs, 1 << 20);
    defer source.deinit();
    try std.testing.expectError(error.CompactionWorkspaceLimit, selectCached(std.testing.allocator, head, &source, range, 3, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 }, &authority, .raw));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, selectCached(std.testing.allocator, head, &source, range, 3, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }, &authority, .raw));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, cachedAdapterAllocationFailure, .{});
}
