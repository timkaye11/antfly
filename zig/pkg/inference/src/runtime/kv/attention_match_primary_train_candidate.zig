const std = @import("std");
const platform = @import("antfly_platform");
const solver = @import("attention_match.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const joint = @import("attention_match_joint_chunk_mass_diagnostic.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const probe = @import("attention_match_train_critical_policy_probe.zig");

pub const policy = "primary_train_candidate_v1|branch=divisor2_at_primary_budget|heldout_parameter=absent|metric=independent_fp64_exact_f16_rounded_train_relative_mass_max|postfit=l2|ownership=atomic";

pub const MetricsDomain = enum { full_train, deterministic_train_cv };
pub const Timing = struct {
    freeze_ns: u64 = 0,
    fit_ns: u64 = 0,
    compose_ns: u64 = 0,
    postfit_ns: u64 = 0,
    validation_ns: u64 = 0,
    clock_context: ?*anyopaque = null,
    clock_fn: *const fn (?*anyopaque) u64 = monotonicClock,
};
fn monotonicClock(_: ?*anyopaque) u64 {
    return platform.time.monotonicNs();
}
const Timed = enum { freeze, fit, compose, postfit, validation };
fn start(t: ?*Timing) u64 {
    return if (t) |v| v.clock_fn(v.clock_context) else 0;
}
fn finish(t: ?*Timing, phase: Timed, began: u64) void {
    const v = t orelse return;
    const elapsed = v.clock_fn(v.clock_context) -| began;
    switch (phase) {
        .freeze => v.freeze_ns +|= elapsed,
        .fit => v.fit_ns +|= elapsed,
        .compose => v.compose_ns +|= elapsed,
        .postfit => v.postfit_ns +|= elapsed,
        .validation => v.validation_ns +|= elapsed,
    }
}
test "diagnostic timing classifies exact callback deltas" {
    const Clock = struct {
        value: u64 = 10,
        step: u64 = 3,
        fn now(raw: ?*anyopaque) u64 {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const out = self.value;
            self.value += self.step;
            return out;
        }
    };
    var clock: Clock = .{};
    var timing: Timing = .{ .clock_context = &clock, .clock_fn = Clock.now };
    var began = start(&timing);
    finish(&timing, .freeze, began);
    clock.step = 5;
    began = start(&timing);
    finish(&timing, .validation, began);
    try std.testing.expectEqual(@as(u64, 3), timing.freeze_ns);
    try std.testing.expectEqual(@as(u64, 5), timing.validation_ns);
    try std.testing.expectEqual(@as(u64, 0), timing.fit_ns);
}

pub const Owned = struct {
    candidate: replay.Candidate,
    variant: probe.Variant,
    critical_policy: replay.CriticalCorePolicy,
    critical_policy_sha256: [32]u8,
    candidate_sha256: [32]u8,
    source_authority_sha256: [32]u8,
    train_domain_sha256: [32]u8,
    train_metrics: solver.Metrics,
    postfit: replay.PostfitEvidence,
    primary_budget_percent: u8,
    candidate_live_bytes: usize,
    metrics_domain: MetricsDomain,
    fit_query_rows: usize,
    cv_query_rows: usize,
    probe_timing: Timing = .{},
    replay_phase_timing: replay.PhaseTiming = .{},

    pub fn deinit(self: *@This()) void {
        self.candidate.deinit();
        self.* = undefined;
    }

    pub fn probeCandidate(self: *const @This()) probe.Candidate {
        return .{
            .variant = self.variant,
            .variant_policy_sha256 = self.critical_policy_sha256,
            .source_authority_sha256 = self.source_authority_sha256,
            .train_domain_sha256 = self.train_domain_sha256,
            .candidate_sha256 = self.candidate_sha256,
            .primary_budget_percent = self.primary_budget_percent,
            .candidate_rows = self.candidate.head.rows,
            .candidate_live_bytes = self.candidate_live_bytes,
            .exact_rounded_train_max_error = self.train_metrics.max_relative_mass_error,
            .postfit_attempts = self.postfit.attempts,
            .postfit_completions = self.postfit.completions,
            .heldout_observations = 0,
            .frozen_before_heldout = true,
        };
    }
};

pub const Config = struct {
    critical_policy: replay.CriticalCorePolicy,
    primary_budget_percent: u8,
    postfit_workspace_bytes: usize,
    external_live_bytes: usize,
    combined_cap: usize,
    source_authority_sha256: [32]u8,
    train_domain_sha256: [32]u8,
    validation_backend: solver.Backend,
    selection_train_queries: []const f64 = &.{},
    selection_split_sha256: [32]u8 = @splat(0),
    timing: ?*Timing = null,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b);
}

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}

fn candidateBytes(candidate: replay.Candidate) !usize {
    const numeric = try mul(try add(try mul(candidate.head.rows, try mul(candidate.head.dim, 2)), candidate.head.rows), @sizeOf(f64));
    return add(numeric, try mul(try mul(candidate.head.rows, 2), @sizeOf(usize)));
}

fn digestCandidate(candidate: replay.Candidate) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(std.mem.sliceAsBytes(candidate.head.keys));
    hash.update(std.mem.sliceAsBytes(candidate.head.values));
    hash.update(std.mem.sliceAsBytes(candidate.head.bias.?));
    hash.update(std.mem.sliceAsBytes(candidate.source_indices));
    hash.update(std.mem.sliceAsBytes(candidate.coverage_source_indices));
    hash.update(std.mem.asBytes(&candidate.head.scale));
    hash.update(std.mem.asBytes(&candidate.head.softcap));
    return hash.finalResult();
}

fn policyVariant(value: replay.CriticalCorePolicy) !probe.Variant {
    return switch (value) {
        .train_probability_gte_0_025_no_top1_v1 => .fixed_probability_0_025_no_top1,
        .train_probability_budget_calibrated_floor_0_0025_p91_residual16_v2 => .calibrated_floor_0_0025_p91_residual16,
        else => error.InvalidFitConfig,
    };
}

fn boundTrainDomain(base: [32]u8, fit: []const f64, cv: []const f64, split: [32]u8) [32]u8 {
    if (cv.len == 0) return base;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.primary-train-probe.fit-cv-domain.v1\x00");
    hash.update(&base);
    hash.update(&split);
    hash.update(std.mem.sliceAsBytes(fit));
    hash.update(std.mem.sliceAsBytes(cv));
    return hash.finalResult();
}

fn validateTrainMetrics(metrics: solver.Metrics) !void {
    if (!std.math.isFinite(metrics.max_relative_mass_error) or !std.math.isFinite(metrics.relative_output_rmse))
        return error.NonFiniteAttention;
}

/// Builds only the divisor-2 candidate from an already frozen p91 TRAIN plan.
/// The type intentionally has no heldout argument or callback surface.
pub fn buildPrimaryTrainCandidate(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    ranges: []const admission.Range,
    adaptive_plan: *const replay.AdaptivePlan,
    fit_cfg: solver.Config,
    cached: *training_source.Source,
    cfg: Config,
) !Owned {
    return buildPrimaryTrainCandidateImpl(allocator, original, train, ranges, adaptive_plan, fit_cfg, cached, null, cfg);
}

/// Page-backed equivalent of buildPrimaryTrainCandidate. The hook must load
/// exact-f32 probabilities from the authenticated bounded provider and must
/// release its page before returning. No flat probability cache is accepted.
pub fn buildPrimaryTrainCandidatePageBacked(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    ranges: []const admission.Range,
    adaptive_plan: *const replay.AdaptivePlan,
    fit_cfg: solver.Config,
    postfit_hook: replay.CandidatePostfitHook,
    cfg: Config,
) !Owned {
    return buildPrimaryTrainCandidateImpl(allocator, original, train, ranges, adaptive_plan, fit_cfg, null, postfit_hook, cfg);
}

fn buildPrimaryTrainCandidateImpl(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    ranges: []const admission.Range,
    adaptive_plan: *const replay.AdaptivePlan,
    fit_cfg: solver.Config,
    cached: ?*training_source.Source,
    postfit_hook: ?replay.CandidatePostfitHook,
    cfg: Config,
) !Owned {
    try fit_cfg.control.check();
    if ((cached == null) == (postfit_hook == null)) return error.InvalidFitConfig;
    if (original.dim == 0 or train.len == 0 or train.len % original.dim != 0 or ranges.len == 0 or
        cfg.primary_budget_percent != 91 or cfg.combined_cap == 0 or
        cfg.external_live_bytes >= cfg.combined_cap or fit_cfg.max_workspace_bytes != cfg.combined_cap or
        cfg.postfit_workspace_bytes == 0 or std.mem.allEqual(u8, &cfg.source_authority_sha256, 0) or
        std.mem.allEqual(u8, &cfg.train_domain_sha256, 0) or cfg.validation_backend.summary_fn == null)
        return error.InvalidFitConfig;
    if ((cfg.selection_train_queries.len == 0) != std.mem.allEqual(u8, &cfg.selection_split_sha256, 0) or
        (cfg.selection_train_queries.len != 0 and cfg.selection_train_queries.len % original.dim != 0)) return error.InvalidFitConfig;
    const variant = try policyVariant(cfg.critical_policy);
    const branch_index: usize = 2;
    const geometry = &adaptive_plan.branches[branch_index];
    if (!geometry.available or geometry.targets.len != ranges.len)
        return error.InvalidFitConfig;
    const available = cfg.combined_cap - cfg.external_live_bytes;
    var postfit: replay.PostfitEvidence = undefined;
    var candidate = candidate: {
        const freeze_started = start(cfg.timing);
        var frozen = try replay.freezeFromScored(allocator, ranges, geometry.targets, adaptive_plan, available, fit_cfg.control);
        finish(cfg.timing, .freeze, freeze_started);
        defer frozen.deinit();
        const frozen_bytes = try replay.frozenBytes(frozen);
        if (frozen_bytes >= available) return error.CompactionWorkspaceLimit;
        const fit_started = start(cfg.timing);
        var fitted = try replay.fitFromScored(allocator, original, frozen, adaptive_plan, available - frozen_bytes, fit_cfg.control);
        finish(cfg.timing, .fit, fit_started);
        defer fitted.deinit();
        var selected: usize = 0;
        var candidate_rows = original.rows;
        var max_selected: usize = 0;
        for (frozen.ranges, frozen.chunks) |range, chunk| {
            selected = try add(selected, chunk.selected_local.len);
            max_selected = @max(max_selected, chunk.selected_local.len);
            candidate_rows = try add(std.math.sub(usize, candidate_rows, range.rows) catch return error.InvalidAttentionShape, chunk.selected_local.len);
        }
        const fitted_bytes = try add(try mul(fitted.chunks.len, @sizeOf(joint.ChunkResult)), try mul(try add(try mul(selected, original.dim), selected), @sizeOf(f64)));
        const candidate_numeric = try mul(try add(try mul(try mul(candidate_rows, original.dim), 2), candidate_rows), @sizeOf(f64));
        const candidate_maps = try mul(try mul(candidate_rows, 2), @sizeOf(usize));
        const compose_scratch = try mul(max_selected, @sizeOf(usize));
        const precompose_peak = try add(cfg.external_live_bytes, try add(frozen_bytes, try add(fitted_bytes, try add(candidate_numeric, try add(candidate_maps, compose_scratch)))));
        if (try add(precompose_peak, cfg.postfit_workspace_bytes) > cfg.combined_cap) return error.CompactionWorkspaceLimit;
        const compose_started = start(cfg.timing);
        var owned = try replay.compose(allocator, original, frozen.ranges, frozen, fitted);
        finish(cfg.timing, .compose, compose_started);
        errdefer owned.deinit();
        const owned_bytes = try candidateBytes(owned);
        const live_before_postfit = try add(cfg.external_live_bytes, try add(frozen_bytes, try add(fitted_bytes, owned_bytes)));
        if (try add(live_before_postfit, cfg.postfit_workspace_bytes) > cfg.combined_cap) return error.CompactionWorkspaceLimit;
        var post_cfg = fit_cfg;
        post_cfg.max_workspace_bytes = cfg.postfit_workspace_bytes;
        // Preserve the backend identity used to authenticate cached TRAIN data.
        const postfit_started = start(cfg.timing);
        postfit = if (postfit_hook) |hook|
            try hook.apply_fn(hook.context, allocator, original, &owned, ranges, frozen, post_cfg)
        else
            try replay.applyTrainGroupPostfit(allocator, original, &owned, ranges, frozen, cached orelse return error.InvalidFitConfig, post_cfg, cfg.postfit_workspace_bytes, .train_global_residual_range_mass_l2_exact_f32_cache_v1);
        finish(cfg.timing, .postfit, postfit_started);
        break :candidate owned;
    };
    errdefer candidate.deinit();
    const owned_bytes = try candidateBytes(candidate);
    const validation_queries = if (cfg.selection_train_queries.len == 0) train else cfg.selection_train_queries;
    var validation_cfg = fit_cfg;
    validation_cfg.backend = cfg.validation_backend;
    const validation_bytes = try solver.validationBytes(original.rows, candidate.head.rows, original.dim, validation_queries.len / original.dim, validation_cfg);
    if (try add(cfg.external_live_bytes, try add(owned_bytes, validation_bytes)) > cfg.combined_cap) return error.CompactionWorkspaceLimit;
    const validation_started = start(cfg.timing);
    const metrics = try solver.validationMetrics(allocator, original, candidate.head, validation_queries, validation_cfg);
    finish(cfg.timing, .validation, validation_started);
    try fit_cfg.control.check();
    try validateTrainMetrics(metrics);
    const domain_digest = boundTrainDomain(cfg.train_domain_sha256, train, cfg.selection_train_queries, cfg.selection_split_sha256);
    return .{
        .candidate = candidate,
        .variant = variant,
        .critical_policy = cfg.critical_policy,
        .critical_policy_sha256 = replay.criticalCorePolicyDigest(cfg.critical_policy),
        .candidate_sha256 = digestCandidate(candidate),
        .source_authority_sha256 = cfg.source_authority_sha256,
        .train_domain_sha256 = domain_digest,
        .train_metrics = metrics,
        .postfit = postfit,
        .primary_budget_percent = 91,
        .candidate_live_bytes = owned_bytes,
        .metrics_domain = if (cfg.selection_train_queries.len == 0) .full_train else .deterministic_train_cv,
        .fit_query_rows = train.len / original.dim,
        .cv_query_rows = cfg.selection_train_queries.len / original.dim,
        .probe_timing = if (cfg.timing) |timing| timing.* else .{},
    };
}

test "TRAIN probe metrics remain non-authorizing and require finite values" {
    const observed = solver.Metrics{
        .quality_evaluated = false,
        .max_relative_mass_error = 0.04,
        .relative_output_rmse = 0.01,
    };
    try validateTrainMetrics(observed);
    var nonfinite = observed;
    nonfinite.max_relative_mass_error = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteAttention, validateTrainMetrics(nonfinite));
    nonfinite = observed;
    nonfinite.relative_output_rmse = std.math.inf(f64);
    try std.testing.expectError(error.NonFiniteAttention, validateTrainMetrics(nonfinite));
}

test "primary TRAIN candidate body rejects invalid shape before allocation" {
    const Backend = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return error.TestUnexpectedResult;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {
            return error.TestUnexpectedResult;
        }
    };
    const invalid = solver.Head{
        .keys = &.{},
        .values = &.{},
        .rows = 0,
        .dim = 0,
        .scale = 1,
    };
    const fit_cfg = solver.Config{ .target_rows = 1, .max_workspace_bytes = 1024 * 1024 };
    const cfg = Config{
        .critical_policy = .train_probability_gte_0_025_no_top1_v1,
        .primary_budget_percent = 91,
        .postfit_workspace_bytes = 1,
        .external_live_bytes = 0,
        .combined_cap = 1024 * 1024,
        .source_authority_sha256 = @splat(1),
        .train_domain_sha256 = @splat(2),
        .validation_backend = .{ .workspace_fn = Backend.workspace, .summary_fn = Backend.summary },
    };
    // These objects have real addresses but must remain unread: the invalid
    // head is rejected before either plan/cache inspection or allocation.
    var plan: replay.AdaptivePlan = undefined;
    var cached: training_source.Source = undefined;
    try std.testing.expectError(error.InvalidFitConfig, buildPrimaryTrainCandidate(std.testing.failing_allocator, invalid, &.{}, &.{}, &plan, fit_cfg, &cached, cfg));
}

test "FIT CV domain binding is canonical and default path remains base digest" {
    const base = @as([32]u8, @splat(1));
    const split = @as([32]u8, @splat(2));
    try std.testing.expectEqualSlices(u8, &base, &boundTrainDomain(base, &.{1}, &.{}, @splat(0)));
    const a = boundTrainDomain(base, &.{ 1, 2 }, &.{3}, split);
    const b = boundTrainDomain(base, &.{ 1, 2 }, &.{3}, split);
    try std.testing.expectEqualSlices(u8, &a, &b);
    try std.testing.expect(!std.mem.eql(u8, &a, &base));
    try std.testing.expect(!std.mem.eql(u8, &a, &boundTrainDomain(base, &.{ 1, 2 }, &.{4}, split)));
}
