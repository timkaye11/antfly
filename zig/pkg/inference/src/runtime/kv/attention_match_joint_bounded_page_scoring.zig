// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Default-off ownership seam for bounded candidate selection, rounded joint
//! value fitting, and TRAIN-only page scoring.  The builder is responsible for
//! streaming selection and the joint fit; the scorer must obtain probability
//! pages from the bound FloatPair normalizer. Workspace callbacks are pure,
//! allocation-free, dispatch-free stable-bound queries. Neither execution
//! callback may retain its advertised transient workspace after returning.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const adaptive = @import("attention_match_joint_adaptive_geometry.zig");

pub const policy = "bounded_page_candidate_scoring_v1";
pub const scalar_execution_policy = "bounded_scalar_score_three_candidates_per_range_v1";
pub const accelerate_execution_policy = "bounded_accelerate_fp64_tiled_score_three_candidates_per_range_v1";
pub const execution_policy = scalar_execution_policy;

pub const Artifact = struct {
    context: ?*anyopaque,
    retained_bytes: usize,
    selected_rows: usize,
    destroy_fn: *const fn (?*anyopaque, std.mem.Allocator) void,

    pub fn deinit(self: *Artifact, allocator: std.mem.Allocator) void {
        self.destroy_fn(self.context, allocator);
        self.* = undefined;
    }
};

pub const Builder = struct {
    context: ?*anyopaque,
    retained_bytes_fn: *const fn (?*anyopaque, admission.Range, usize) anyerror!usize,
    transient_bytes_fn: *const fn (?*anyopaque, admission.Range, usize) anyerror!usize,
    build_fn: *const fn (?*anyopaque, std.mem.Allocator, admission.Range, usize, usize, solver.Control) anyerror!Artifact,
    /// Default false preserves established divisor targets byte-for-byte.
    core_aware_targets: bool = false,
    target_rows_fn: ?*const fn (?*anyopaque, admission.Range, usize, solver.Control) anyerror!usize = null,
};

fn targetRows(builder: Builder, range: admission.Range, divisor: usize, control: solver.Control) !usize {
    if (!builder.core_aware_targets) return std.math.divCeil(usize, range.rows, divisor) catch error.InvalidFitConfig;
    const callback = builder.target_rows_fn orelse return error.InvalidFitConfig;
    const target = try callback(builder.context, range, divisor, control);
    if (target == 0 or target >= range.rows) return error.CompactionFitRejected;
    return target;
}

pub const PageScorer = struct {
    context: ?*anyopaque,
    execution_policy: []const u8 = scalar_execution_policy,
    /// Synchronous, allocation-free and dispatch-free authority check. It must
    /// prove that Y is the exact widening of the bound normalizer output.
    validate_original_y_fn: *const fn (?*anyopaque, []const f64, solver.Control) anyerror!void,
    workspace_fn: *const fn (?*anyopaque, admission.Range, usize) anyerror!usize,
    score_fn: *const fn (
        ?*anyopaque,
        *const Artifact,
        admission.Range,
        usize,
        []f64,
        []f64,
        []f64,
        []f64,
        solver.Control,
    ) anyerror!void,
};

pub const Plan = struct {
    allocator: std.mem.Allocator,
    scores: []adaptive.RangeScores,
    artifacts: [][3]?Artifact,
    /// Sum of the builder-authenticated retained ownership for every scored
    /// packet. These packets remain live after build returns and throughout
    /// all three geometry choices.
    retained_artifact_bytes: usize,
    workspace_bound_bytes: usize,
    score_completions: usize,
    score_query_rows: usize,
    execution_policy: []const u8,

    pub fn deinit(self: *Plan) void {
        for (self.artifacts) |*levels| for (levels) |*artifact| if (artifact.*) |*owned| owned.deinit(self.allocator);
        self.allocator.free(self.artifacts);
        self.allocator.free(self.scores);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn vectorBytes(q: usize, dim: usize) !usize {
    const qd = try mul(q, dim);
    return mul(try add(try mul(2, q), try mul(2, qd)), @sizeOf(f64));
}

/// Builds and scores all d4/d3/d2 candidates without a full probability
/// matrix. `external_live_bytes` includes source/train queries, normalizer and
/// every caller-owned allocation. Callback workspace bounds must be stable for
/// the duration of this call. The full retained-artifact upper bound is
/// preflighted before the first callback or allocation.
pub fn build(
    allocator: std.mem.Allocator,
    ranges: []const admission.Range,
    original_y: []const f64,
    query_count: usize,
    dim: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    builder: Builder,
    scorer: PageScorer,
    control: solver.Control,
) !Plan {
    try control.check();
    if (ranges.len == 0 or query_count == 0 or dim == 0 or
        original_y.len != try mul(query_count, dim) or external_live_bytes > combined_cap) return error.InvalidFitConfig;
    for (original_y, 0..) |value, i| {
        if ((i & 4095) == 0) try control.check();
        if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    }
    try scorer.validate_original_y_fn(scorer.context, original_y, control);
    if (!std.mem.eql(u8, scorer.execution_policy, scalar_execution_policy) and
        !std.mem.eql(u8, scorer.execution_policy, accelerate_execution_policy)) return error.InvalidFitConfig;
    var previous_end: usize = 0;
    for (ranges, 0..) |range, i| {
        const end = try add(range.start, range.rows);
        if (range.rows == 0 or (i != 0 and range.start < previous_end)) return error.InvalidFitConfig;
        previous_end = end;
    }

    const table_bytes = try add(
        try mul(ranges.len, @sizeOf(adaptive.RangeScores)),
        try mul(ranges.len, @sizeOf([3]?Artifact)),
    );
    const vectors = try vectorBytes(query_count, dim);
    var retained_upper: usize = 0;
    var callback_peak: usize = 0;
    for (ranges) |range| {
        if (range.rows == 0) return error.InvalidFitConfig;
        for ([_]usize{ 4, 3, 2 }) |divisor| {
            try control.check();
            // Before any callback/dispatch, reserve the largest compacted
            // packet possible. Actual core-aware targets are obtained later.
            if (builder.core_aware_targets and range.rows < 2) return error.InvalidFitConfig;
            const target = if (builder.core_aware_targets) range.rows - 1 else std.math.divCeil(usize, range.rows, divisor) catch return error.InvalidFitConfig;
            const retained_bytes = try builder.retained_bytes_fn(builder.context, range, target);
            const build_workspace = try builder.transient_bytes_fn(builder.context, range, target);
            const score_workspace = try scorer.workspace_fn(scorer.context, range, target);
            callback_peak = @max(callback_peak, @max(build_workspace, score_workspace));
            retained_upper = try add(retained_upper, retained_bytes);
        }
    }
    const admitted = try add(external_live_bytes, try add(table_bytes, try add(vectors, try add(retained_upper, callback_peak))));
    if (admitted > combined_cap) return error.CompactionWorkspaceLimit;

    const scores = try allocator.alloc(adaptive.RangeScores, ranges.len);
    errdefer allocator.free(scores);
    const artifacts = try allocator.alloc([3]?Artifact, ranges.len);
    errdefer allocator.free(artifacts);
    @memset(artifacts, .{ null, null, null });
    errdefer for (artifacts) |*levels| for (levels) |*artifact| if (artifact.*) |*owned| owned.deinit(allocator);

    const qd = try mul(query_count, dim);
    const range_mass = try allocator.alloc(f64, query_count);
    defer allocator.free(range_mass);
    const range_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(range_numerator);
    const candidate_mass = try allocator.alloc(f64, query_count);
    defer allocator.free(candidate_mass);
    const candidate_numerator = try allocator.alloc(f64, qd);
    defer allocator.free(candidate_numerator);

    var actual_retained: usize = 0;
    var score_completions: usize = 0;
    var score_query_rows: usize = 0;
    for (ranges, scores, artifacts) |range, *score, *levels| {
        const targets = [4]usize{ try targetRows(builder, range, 4, control), try targetRows(builder, range, 3, control), try targetRows(builder, range, 2, control), range.rows };
        score.* = .{ .start = range.start, .rows = range.rows, .target_rows = targets, .mass_harm = .{ 0, 0, 0, 0 }, .output_harm = .{ 0, 0, 0, 0 } };
        for (0..3) |level| {
            try control.check();
            var artifact = try builder.build_fn(builder.context, allocator, range, targets[level], level, control);
            errdefer artifact.deinit(allocator);
            if (artifact.selected_rows != targets[level] or artifact.retained_bytes > try builder.retained_bytes_fn(builder.context, range, targets[level])) return error.InvalidFitConfig;
            @memset(range_mass, 0);
            @memset(range_numerator, 0);
            @memset(candidate_mass, 0);
            @memset(candidate_numerator, 0);
            try scorer.score_fn(scorer.context, &artifact, range, level, range_mass, range_numerator, candidate_mass, candidate_numerator, control);
            score_completions = try add(score_completions, 1);
            score_query_rows = try add(score_query_rows, query_count);
            const measured = try adaptive.measureRoundedCandidateHarm(
                original_y,
                range_mass,
                range_numerator,
                candidate_mass,
                candidate_numerator,
                query_count,
                dim,
                1e-12,
            );
            score.mass_harm[level] = measured.max_relative_mass;
            score.output_harm[level] = measured.relative_output_rmse;
            actual_retained = try add(actual_retained, artifact.retained_bytes);
            if (try add(external_live_bytes, try add(table_bytes, try add(vectors, actual_retained))) > combined_cap) return error.CompactionWorkspaceLimit;
            levels[level] = artifact;
        }
    }
    try control.check();
    return .{
        .allocator = allocator,
        .scores = scores,
        .artifacts = artifacts,
        .retained_artifact_bytes = actual_retained,
        .workspace_bound_bytes = admitted,
        .score_completions = score_completions,
        .score_query_rows = score_query_rows,
        .execution_policy = scorer.execution_policy,
    };
}

test "page scoring preflight rejects before callbacks" {
    const C = struct {
        calls: usize = 0,
        fn workspace(raw: ?*anyopaque, _: admission.Range, _: usize) !usize {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return 1 << 20;
        }
        fn buildFn(_: ?*anyopaque, _: std.mem.Allocator, _: admission.Range, _: usize, _: usize, _: solver.Control) !Artifact {
            return error.TestUnexpectedResult;
        }
        fn scoreFn(_: ?*anyopaque, _: *const Artifact, _: admission.Range, _: usize, _: []f64, _: []f64, _: []f64, _: []f64, _: solver.Control) !void {
            return error.TestUnexpectedResult;
        }
        fn validateY(_: ?*anyopaque, y: []const f64, _: solver.Control) !void {
            if (y.len != 16 * 8) return error.InvalidFitConfig;
        }
    };
    var c: C = .{};
    const original_y: [16 * 8]f64 = @splat(0);
    try std.testing.expectError(error.CompactionWorkspaceLimit, build(std.testing.allocator, &.{.{ .start = 0, .rows = 64 }}, &original_y, 16, 8, 0, 1024, .{ .context = &c, .retained_bytes_fn = C.workspace, .transient_bytes_fn = C.workspace, .build_fn = C.buildFn }, .{ .context = &c, .validate_original_y_fn = C.validateY, .workspace_fn = C.workspace, .score_fn = C.scoreFn }, .{}));
    try std.testing.expect(c.calls > 0);
}
