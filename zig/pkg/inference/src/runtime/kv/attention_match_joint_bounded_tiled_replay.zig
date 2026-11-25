// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Default-off bounded TRAIN-geometry bridge for the FloatPair page provider.
//! This module intentionally begins after candidate scoring/value fitting. It
//! removes full-P and full-Gram materialization from geometry selection, but it
//! does not yet make candidate construction page-backed.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const tiled = @import("attention_match_joint_tiled_geometry.zig");
const delta_kernel = @import("attention_match_gram_delta_accelerate.zig");
const accelerated_delta = @import("attention_match_bounded_page_delta_accelerate.zig");
const metal = @import("../../ops/metal_attention_match_fit.zig");

pub const policy = "greedy_positive_joint_train_objective_per_row_tiled_floatpair_v1";
pub const budget_percent_policy = "greedy_positive_joint_train_objective_per_row_tiled_floatpair_divisor2_budget_percent_v1";

pub const Result = struct {
    branches: [3]adaptive.Geometry,
    workspace_bound_bytes: usize,

    pub fn deinit(self: *Result) void {
        for (&self.branches) |*branch| branch.deinit();
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn geometryPayloadBytes(range_count: usize) !usize {
    return mul(range_count, @sizeOf(adaptive.Level) + @sizeOf(usize));
}

fn unavailableGeometry(allocator: std.mem.Allocator, ranges: []const tiled.Range, divisor: usize, budget_percent: u8) !adaptive.Geometry {
    const levels = try allocator.alloc(adaptive.Level, ranges.len);
    errdefer allocator.free(levels);
    @memset(levels, .divisor4);
    const targets = try allocator.alloc(usize, ranges.len);
    errdefer allocator.free(targets);
    var eligible: usize = 0;
    var required: usize = 0;
    for (ranges, targets) |range, *target| {
        target.* = range.target_rows[0];
        eligible = try add(eligible, range.rows);
        required = try add(required, target.*);
    }
    const nominal_cap = std.math.divCeil(usize, eligible, divisor) catch return error.InvalidAttentionShape;
    const cap = if (budget_percent == 100) nominal_cap else try mul(nominal_cap, budget_percent) / 100;
    if (required <= cap) return error.InvalidFitConfig;
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("antfly.geometry-budget-infeasible.v1\x00");
    var encoded: [8]u8 = undefined;
    for ([_]usize{ divisor, budget_percent, cap, required }) |value| {
        std.mem.writeInt(u64, &encoded, std.math.cast(u64, value) orelse return error.InvalidAttentionShape, .little);
        h.update(&encoded);
    }
    for (targets) |target| {
        std.mem.writeInt(u64, &encoded, std.math.cast(u64, target) orelse return error.InvalidAttentionShape, .little);
        h.update(&encoded);
    }
    return .{ .allocator = allocator, .available = false, .levels = levels, .targets = targets, .middle_cap_rows = cap, .middle_rows = required, .preserved_ranges = 0, .fitted_ranges = ranges.len, .train_harm_before = 0, .train_harm_after = 0, .digest = h.finalResult() };
}

pub const PageContext = struct {
    pub const CopyCachedPageFn = *const fn (?*anyopaque, usize, usize, usize, usize, []f64, solver.Control) anyerror!void;

    fitter: *metal.Fitter,
    normalizer: *const metal.Fitter.CapturedPageSource,
    original: solver.Head,
    train_queries: []const f64,
    scored: *const replay.AdaptivePlan,
    scratch: []f64,
    maximum_range_rows: usize,
    /// Bound by bounded_page_replay.Context. This callback may only expose the
    /// current authenticated exact-f32 range Source; it is not a generic page
    /// injection seam.
    cached_page_context: ?*anyopaque = null,
    copy_cached_page_fn: ?CopyCachedPageFn = null,
    delta_backend: ?delta_kernel.Backend = null,
    require_accelerated_delta: bool = false,
    ratio_scratch: []f64 = &.{},
    ratio_range: usize = std.math.maxInt(usize),
    ratio_level: usize = std.math.maxInt(usize),
    delta_evidence: accelerated_delta.Evidence = .{},

    fn accessor(self: *PageContext) tiled.DeltaAccessor {
        return .{ .context = self, .fill_fn = fill };
    }

    fn workspaceBytes(self: *const PageContext, tile_rows: usize) !usize {
        if (tile_rows == 0 or self.maximum_range_rows == 0) return error.InvalidFitConfig;
        if (self.scratch.len < try mul(tile_rows, self.maximum_range_rows)) return error.CompactionWorkspaceLimit;
        const scratch_bytes = try mul(self.scratch.len, @sizeOf(f64));
        // Keep the already admitted provider transient in the bound even when
        // the bound exact-f32 cache serves geometry. This is conservative and
        // preserves the existing combined-cap authority.
        const provider = try self.fitter.capturedPageTransientBytes(self.normalizer, self.maximum_range_rows, tile_rows);
        const accelerated = if (self.require_accelerated_delta)
            try accelerated_delta.workspaceBytes(tile_rows, self.maximum_range_rows, self.original.dim)
        else
            0;
        return add(scratch_bytes, @max(provider, accelerated));
    }

    fn fill(raw: ?*anyopaque, range_index: usize, level_index: usize, first_query: usize, query_count: usize, mass: []f64, numerator: []f64, control: solver.Control) !void {
        const self: *PageContext = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
        const qd = try mul(query_count, self.original.dim);
        if (range_index >= self.scored.scores.len or level_index >= 3 or query_count == 0 or
            first_query > self.normalizer.query_count or query_count > self.normalizer.query_count - first_query or
            mass.len != query_count or numerator.len != qd) return error.InvalidAttentionShape;
        const score = self.scored.scores[range_index];
        if (score.rows > self.maximum_range_rows or score.start > self.original.rows or score.rows > self.original.rows - score.start) return error.InvalidAttentionShape;
        const elements = try mul(query_count, score.rows);
        if (elements > self.scratch.len) return error.CompactionWorkspaceLimit;
        const page = self.scratch[0..elements];
        if (self.copy_cached_page_fn) |copy| {
            try copy(self.cached_page_context, score.start, score.rows, first_query, query_count, page, control);
        } else {
            try self.fitter.replayCapturedPageQueries(self.normalizer, score.start, score.rows, first_query, query_count, page, control);
        }
        if (self.require_accelerated_delta) {
            const backend = self.delta_backend orelse return error.AcceleratedDeltaUnsupported;
            const owned = self.scored.scored[range_index][level_index] orelse return error.InvalidAttentionShape;
            if (owned.frozen.chunks.len != 1 or owned.fitted.chunks.len != 1) return error.InvalidAttentionShape;
            const chunk = owned.frozen.chunks[0];
            const original_bias = self.original.bias orelse return error.InvalidAttentionShape;
            if (chunk.selected_local.len == 0 or chunk.selected_local.len > self.ratio_scratch.len or
                chunk.selected_local.len != owned.fitted.bias.len or owned.fitted.values.len != try mul(chunk.selected_local.len, self.original.dim))
                return error.InvalidAttentionShape;
            const ratios = self.ratio_scratch[0..chunk.selected_local.len];
            if (self.ratio_range != range_index or self.ratio_level != level_index) {
                for (chunk.selected_local, owned.fitted.bias, ratios) |local, fitted_bias, *ratio| {
                    if (local >= score.rows) return error.InvalidAttentionShape;
                    const value = @exp(fitted_bias - original_bias[score.start + local]);
                    if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                    ratio.* = value;
                }
                self.ratio_range = range_index;
                self.ratio_level = level_index;
            }
            const local_workspace = try accelerated_delta.workspaceBytes(query_count, chunk.selected_local.len, self.original.dim);
            const value_start = try mul(score.start, self.original.dim);
            const value_count = try mul(score.rows, self.original.dim);
            try accelerated_delta.fill(allocatorForContext(self), page, self.original.values[value_start..][0..value_count], chunk.selected_local, ratios, owned.fitted.values, query_count, score.rows, self.original.dim, backend, 0, local_workspace, control, mass, numerator, &self.delta_evidence);
        } else try accumulatePageDelta(self.scored, range_index, level_index, self.original, query_count, page, control, mass, numerator);
    }
};

fn allocatorForContext(self: *PageContext) std.mem.Allocator {
    // Scored packets and page scratch share the bounded route allocator.
    // The context is created only by the owning bounded replay route.
    return self.scored.allocator;
}

fn accumulatePageDelta(plan: *const replay.AdaptivePlan, range_index: usize, level: usize, original: solver.Head, query_count: usize, probabilities: []const f64, control: solver.Control, mass: []f64, numerator: []f64) !void {
    const qd = try mul(query_count, original.dim);
    if (range_index >= plan.scores.len or level >= 3 or query_count == 0 or mass.len != query_count or numerator.len != qd) return error.InvalidAttentionShape;
    const score = plan.scores[range_index];
    if (probabilities.len != try mul(query_count, score.rows)) return error.InvalidAttentionShape;
    const owned = plan.scored[range_index][level] orelse return error.InvalidAttentionShape;
    if (owned.frozen.chunks.len != 1 or owned.fitted.chunks.len != 1) return error.InvalidAttentionShape;
    const chunk = owned.frozen.chunks[0];
    const original_bias = original.bias orelse return error.InvalidAttentionShape;
    if (chunk.selected_local.len != owned.fitted.bias.len or owned.fitted.values.len != try mul(chunk.selected_local.len, original.dim)) return error.InvalidAttentionShape;
    for (0..query_count) |qi| {
        try control.check();
        const page = probabilities[qi * score.rows ..][0..score.rows];
        for (page, 0..) |probability, local| {
            if (!std.math.isFinite(probability) or probability < 0) return error.NonFiniteAttention;
            mass[qi] -= probability;
            for (0..original.dim) |di| numerator[qi * original.dim + di] -= probability * original.values[(score.start + local) * original.dim + di];
        }
        for (chunk.selected_local, 0..) |local, selected_i| {
            if (local >= score.rows) return error.InvalidAttentionShape;
            const global = score.start + local;
            const ratio = @exp(owned.fitted.bias[selected_i] - original_bias[global]);
            if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
            const probability = page[local] * ratio;
            mass[qi] += probability;
            for (0..original.dim) |di| numerator[qi * original.dim + di] += probability * owned.fitted.values[selected_i * original.dim + di];
        }
    }
}

/// Exact geometry-only bridge. `external_live_bytes` must include the source
/// head, TRAIN queries, widened original Y, retained FloatPair normalizer,
/// scored packets, and every other caller-owned live allocation. The accessor
/// workspace includes caller page scratch plus provider transient/device state.
/// The full three-result peak is rejected before the first chooser allocation
/// or accessor dispatch.
pub fn chooseFromScoredFloatPair(
    allocator: std.mem.Allocator,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    tile_rows: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    context: *PageContext,
    control: solver.Control,
) !Result {
    return chooseFromScoredFloatPairWithBudgetPercent(allocator, ranges, original_y, query_count, dim, tile_rows, 100, external_live_bytes, combined_cap, context, control);
}

pub fn chooseFromScoredFloatPairWithBudgetPercent(
    allocator: std.mem.Allocator,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    tile_rows: usize,
    geometry_budget_percent: u8,
    external_live_bytes: usize,
    combined_cap: usize,
    context: *PageContext,
    control: solver.Control,
) !Result {
    try control.check();
    if (ranges.len == 0 or query_count == 0 or dim == 0 or tile_rows == 0 or
        geometry_budget_percent < 50 or geometry_budget_percent > 100 or
        context.scored.scores.len != ranges.len or context.scored.scored.len != ranges.len or
        context.original.dim != dim or context.normalizer.query_count != query_count or
        original_y.len != try mul(query_count, dim) or context.normalizer.normalizer.outputs.len != original_y.len or external_live_bytes > combined_cap)
        return error.InvalidFitConfig;
    for (original_y, context.normalizer.normalizer.outputs, 0..) |wide, narrow, i| {
        if ((i & 4095) == 0) try control.check();
        if (!std.math.isFinite(wide) or !std.math.isFinite(narrow) or wide != @as(f64, narrow)) return error.InvalidFitConfig;
    }

    const range_bytes = try mul(ranges.len, @sizeOf(tiled.Range));
    const accessor_workspace = try context.workspaceBytes(tile_rows);
    const chooser = try tiled.workspaceBytes(query_count, dim, ranges.len, tile_rows, accessor_workspace);
    const one_result = try geometryPayloadBytes(ranges.len);
    // chooser.owned_peak_bytes already includes one result; retain two earlier
    // results at the third invocation and retain the translated range table.
    const full_peak = try add(external_live_bytes, try add(range_bytes, try add(chooser.combined_peak_bytes, try mul(2, one_result))));
    if (full_peak > combined_cap) return error.CompactionWorkspaceLimit;

    const translated = try allocator.alloc(tiled.Range, ranges.len);
    defer allocator.free(translated);
    for (ranges, translated, context.scored.scores) |range, *out, score| {
        if (range.start != score.start or range.rows != score.rows) return error.InvalidAttentionShape;
        out.* = .{ .start = range.start, .rows = range.rows, .target_rows = .{
            score.target_rows[0], score.target_rows[1], score.target_rows[2], range.rows,
        } };
    }

    var made: usize = 0;
    var branches: [3]adaptive.Geometry = undefined;
    errdefer for (branches[0..made]) |*branch| branch.deinit();
    for ([_]usize{ 4, 3, 2 }, 0..) |divisor, i| {
        try control.check();
        const retained_results = try mul(i, one_result);
        branches[i] = tiled.chooseWithBudgetPercent(
            allocator,
            translated,
            original_y,
            query_count,
            dim,
            divisor,
            .{},
            if (divisor == 2) geometry_budget_percent else 100,
            tile_rows,
            accessor_workspace,
            try add(external_live_bytes, try add(range_bytes, retained_results)),
            combined_cap,
            context.accessor(),
            control,
        ) catch |err| switch (err) {
            error.GeometryBudgetInfeasible => try unavailableGeometry(allocator, translated, divisor, if (divisor == 2) geometry_budget_percent else 100),
            else => return err,
        };
        made += 1;
    }
    try control.check();
    return .{ .branches = branches, .workspace_bound_bytes = full_peak };
}

test "bounded tiled bridge peak includes three results before accessor use" {
    // The full provider/scored integration is exercised by the FloatPair and
    // tiled modules. This pure authority check protects the extra simultaneous
    // lifetime introduced by retaining three branch geometries.
    const ranges: usize = 119;
    const q: usize = 4096;
    const d: usize = 512;
    const tile: usize = 8;
    const accessor: usize = tile * (d + 256) * @sizeOf(f64);
    const plan = try tiled.workspaceBytes(q, d, ranges, tile, accessor);
    const one = try geometryPayloadBytes(ranges);
    const translated = try mul(ranges, @sizeOf(tiled.Range));
    const complete = try add(translated, try add(plan.combined_peak_bytes, try mul(2, one)));
    try std.testing.expect(complete > plan.combined_peak_bytes);
    try std.testing.expect(complete < 512 * 1024 * 1024);
}

test "bounded unavailable geometry retains minimum targets and nominal cap" {
    const ranges = [_]tiled.Range{
        .{ .start = 0, .rows = 8, .target_rows = .{ 3, 4, 5, 8 } },
        .{ .start = 8, .rows = 8, .target_rows = .{ 3, 4, 5, 8 } },
    };
    var geometry = try unavailableGeometry(std.testing.allocator, &ranges, 4, 100);
    defer geometry.deinit();
    try std.testing.expect(!geometry.available);
    try std.testing.expectEqual(@as(usize, 4), geometry.middle_cap_rows);
    try std.testing.expectEqual(@as(usize, 6), geometry.middle_rows);
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, geometry.targets);
    const zero = std.mem.zeroes([32]u8);
    try std.testing.expect(!std.mem.eql(u8, &geometry.digest, &zero));
}
