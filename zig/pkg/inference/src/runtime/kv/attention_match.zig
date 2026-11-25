// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! attention_match_v1 numerical reference. Inputs are actual post-RoPE queries
//! and keys, grouped by KV head across ALL consumers of a shared donor. This
//! module deliberately has no device dependencies. FP64 is the CUDA oracle,
//! not a qualified serving implementation. Bias is added AFTER logit softcap.
const std = @import("std");

pub const Head = struct {
    keys: []const f64,
    values: []const f64,
    /// Null means exact rows, each with unit attention mass.
    bias: ?[]const f64 = null,
    rows: usize,
    dim: usize,
    scale: f64,
    softcap: f64 = 0,

    pub fn validate(self: Head) !void {
        if (self.rows == 0 or self.dim == 0 or !std.math.isFinite(self.scale) or self.scale <= 0 or
            !std.math.isFinite(self.softcap) or self.softcap < 0) return error.InvalidAttentionShape;
        const count = try mul(self.rows, self.dim);
        if (self.keys.len != count or self.values.len != count) return error.InvalidAttentionShape;
        try finite(self.keys);
        try finite(self.values);
        if (self.bias) |b| {
            if (b.len != self.rows) return error.InvalidAttentionShape;
            try finite(b);
        }
    }
};

pub const Control = struct {
    context: ?*anyopaque = null,
    check_fn: ?*const fn (?*anyopaque) anyerror!void = null,

    pub fn check(self: Control) !void {
        if (self.check_fn) |check_fn| try check_fn(self.context);
    }
};

/// Phase-specific value solver. Unlike Backend, this contract is never used
/// for attention, validation, or selection workspace queries. `fit_fn` checks
/// cancellation and finiteness before atomically publishing `out`.
pub const ValueBackend = struct {
    context: ?*anyopaque = null,
    recompute_fixed_original_tiled: bool = false,
    workspace_fn: *const fn (?*anyopaque, usize, usize, usize, f64) anyerror!usize,
    fit_fn: *const fn (?*anyopaque, std.mem.Allocator, []const f64, []const f64, usize, usize, usize, f64, Control, []f64) anyerror!void,
};

/// Optional numerical acceleration. Acceptance, source ownership, storage
/// rounding and the final direct NNLS check remain in the native engine.
/// Every callback is synchronous and returns logical scratch ownership to its
/// backend. A backend may retain reusable
/// device capacity until deinit only while keeping it charged against the live
/// context ledger. GPU implementations admit new device memory against that
/// ledger. Host scratch beyond the reference bound must be declared.
pub const RawRangePolicy = enum { off, range_shared_fp64_gram_original_kkt_v1 };

pub const Backend = struct {
    pub const RawRangeSession = struct {
        context: *anyopaque,
        generation: u64,
    };
    pub const RawRangeIdentity = struct {
        range_start: usize,
        range_rows: usize,
        query_rows: usize,
        source_rows: usize,
        source_identity: [32]u8,
        train_identity: [32]u8,
        critical_authority_identity: [32]u8,
        policy_identity: [32]u8,
    };
    validation_batch: enum { rows_64, rows_256 } = .rows_64,
    raw_range_policy: RawRangePolicy = .off,
    context: ?*anyopaque = null,
    workspace_fn: *const fn (?*anyopaque, usize, usize, usize, usize) anyerror!usize,
    attend_fn: ?*const fn (?*anyopaque, std.mem.Allocator, Head, []const f64, []f64, ?[]f64, ?[]f64, Control) anyerror!void = null,
    /// Validation consumes only attention output and log mass. Backends may
    /// retain probabilities on device instead of downloading unused matrices.
    summary_fn: ?*const fn (?*anyopaque, std.mem.Allocator, Head, []const f64, []f64, []f64, Control) anyerror!void = null,
    values_fn: ?*const fn (?*anyopaque, std.mem.Allocator, []const f64, []const f64, usize, usize, usize, f64, Control, []f64) anyerror!void = null,
    select_fn: ?*const fn (?*anyopaque, std.mem.Allocator, []const f64, []const f64, usize, usize, []const usize, []usize, []f64, Config) anyerror!usize = null,
    /// Diagnostic-only arbitrary-RHS raw-correlation selector. Production
    /// normalized selection never consults this callback.
    raw_select_fn: ?*const fn (?*anyopaque, std.mem.Allocator, []const f64, []const f64, []const f64, usize, usize, []const usize, []usize, []f64, Config) anyerror!usize = null,
    /// Complete live range owner charge: FP64 payload plus implementation
    /// descriptor. It is queried before begin and used by replay planning.
    raw_range_workspace_fn: ?*const fn (?*anyopaque, usize) anyerror!usize = null,
    /// Range-scoped owner for raw selection state shared by the three divisor
    /// candidates. Begin is called only after admission; end is unconditional.
    raw_range_begin_fn: ?*const fn (?*anyopaque, std.mem.Allocator, RawRangeIdentity, usize) anyerror!RawRangeSession = null,
    raw_range_end_fn: ?*const fn (?*anyopaque, RawRangeSession) void = null,
};

/// Explicit immutable-reference validation hook. The solver invokes this only
/// for the source/original summary and reports every candidate delegation.
pub const ReferenceSummaryBackend = struct {
    backend: Backend,
    observer_context: ?*anyopaque = null,
    candidate_delegate_fn: ?*const fn (?*anyopaque) void = null,
};

pub const ActiveNnlsTraceEvent = struct {
    ordinal: usize = 0,
    phase: enum { outer, inner, drop },
    q: usize,
    m: usize,
    solves: usize,
    best: ?usize,
    active_count: usize,
    violation: f64,
    tolerance: f64,
    feasible: bool = false,
    alpha: f64 = 0,
    drop: ?usize = null,
    max_delta: f64 = 0,
    weights_sha256: [32]u8,
    active_sha256: [32]u8,
    trial_sha256: [32]u8 = @splat(0),
    fixed_point_enabled: bool,
};

pub const ActiveNnlsObserver = struct {
    context: ?*anyopaque,
    record_fn: *const fn (?*anyopaque, ActiveNnlsTraceEvent) void,

    fn record(self: ActiveNnlsObserver, event: ActiveNnlsTraceEvent) void {
        self.record_fn(self.context, event);
    }
};

pub const Config = struct {
    pub const WorkspaceAdmission = enum { standard_1g, resident_native_hybrid_2g_v1 };
    backend: ?Backend = null,
    reference_summary_backend: ?ReferenceSummaryBackend = null,
    /// Replaces only the dense Householder value workspace and value solve.
    /// Attention, selection, and validation continue to use `backend`.
    value_backend: ?ValueBackend = null,
    /// Reports held-out errors even when a candidate is rejected.
    metrics: ?*Metrics = null,
    target_rows: usize,
    selection: enum { rms, residual_omp } = .rms,
    mandatory_training_argmax: bool = false,
    mandatory_training_top_k: usize = 1,
    mass_solver: enum { coordinate, active_set } = .coordinate,
    max_workspace_bytes: usize = 512 * 1024 * 1024,
    workspace_admission: WorkspaceAdmission = .standard_1g,
    max_sweeps: usize = 4096,
    nnls_tolerance: f64 = 1e-8,
    ridge: f64 = 1e-8,
    /// Diagnostic-only early rejection of a deterministic active-set fixed
    /// point. Default false preserves the production Lawson-Hanson loop.
    reject_exact_active_fixed_point: bool = false,
    /// Test-only bounded diagnostics. A null observer is the production
    /// default and leaves allocation, workspace, and numerical behavior intact.
    active_nnls_observer: ?ActiveNnlsObserver = null,
    /// Explicit cumulative-selection seam. Retains final raw NNLS mass
    /// weights for an identical fixed-index refit; default callers allocate none.
    retain_mass_weights: bool = false,
    min_weight: f64 = 1e-12,
    max_relative_mass_error: f64 = 0.05,
    max_relative_output_rmse: f64 = 0.05,
    control: Control = .{},
    /// Non-owning token valid only inside prepareScoredBank's range scope.
    raw_range_session: ?Backend.RawRangeSession = null,
    /// Sealed fitter policy; default-off preserves the established path.
    raw_range_policy: RawRangePolicy = .off,

    pub fn admittedCombinedCap(self: Config) !usize {
        return switch (self.workspace_admission) {
            .standard_1g => 1024 * 1024 * 1024,
            .resident_native_hybrid_2g_v1 => if (self.max_workspace_bytes > 0 and self.max_workspace_bytes <= 2 * 1024 * 1024 * 1024) 2 * 1024 * 1024 * 1024 else error.InvalidFitConfig,
        };
    }

    fn validate(self: Config, rows: usize) !void {
        if (self.raw_range_policy == .off and self.raw_range_session != null) return error.InvalidFitConfig;
        if (self.target_rows == 0 or self.target_rows > rows or self.max_sweeps == 0) return error.InvalidFitConfig;
        if ((self.mandatory_training_argmax and (self.selection != .residual_omp or self.mandatory_training_top_k == 0 or self.mandatory_training_top_k > 2 or self.mandatory_training_top_k > rows)) or
            (!self.mandatory_training_argmax and self.mandatory_training_top_k != 1)) return error.InvalidFitConfig;
        for ([_]f64{ self.nnls_tolerance, self.ridge, self.min_weight }) |x| {
            if (!std.math.isFinite(x) or x <= 0) return error.InvalidFitConfig;
        }
        for ([_]f64{ self.max_relative_mass_error, self.max_relative_output_rmse }) |x| {
            if (!std.math.isFinite(x) or x < 0) return error.InvalidFitConfig;
        }
    }
};

pub const Metrics = struct {
    /// True only after held-out attention metrics were actually evaluated.
    /// Matrix-only device adapters populate solver counters but do not own a
    /// held-out acceptance decision.
    quality_evaluated: bool = false,
    max_relative_mass_error: f64 = 0,
    relative_output_rmse: f64 = 0,
    nnls_sweeps: usize = 0,
    selection_steps: usize = 0,
    selection_active_solves: usize = 0,
};

/// Bounded DEV observation from the same validation pass. Row indices are in
/// their respective source/compact heads; no query or probability payload is
/// retained.
pub const ValidationDetail = struct {
    worst_query_index: usize,
    worst_log_mass_ratio: f64,
    worst_relative_mass_error: f64,
    original_peak_row: usize,
    compact_peak_row: usize,
    original_peak_probability: f64,
    compact_peak_probability: f64,
    queries_over_0_05: usize,
    queries_over_0_5: usize,
    queries_over_0_9: usize,
    mean_reference_peak_probability: f64,
};

pub const ValidationObservation = struct { metrics: Metrics, detail: ValidationDetail };

pub const Result = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    source_indices: []usize,
    metrics: Metrics = .{},

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.source_indices);
        self.* = undefined;
    }

    pub fn head(self: Result, source: Head) Head {
        return .{ .keys = self.keys, .values = self.values, .bias = self.bias, .rows = self.bias.len, .dim = source.dim, .scale = source.scale, .softcap = source.softcap };
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn finite(xs: []const f64) !void {
    for (xs) |x| if (!std.math.isFinite(x)) return error.NonFiniteAttention;
}

/// Conservative bound on simultaneous scratch AND result allocations, excluding
/// caller-owned source/query buffers. Admission must also count those buffers and
/// the original device cache until replacement has committed.
pub fn workspaceBytes(n: usize, m: usize, q: usize, d: usize) !usize {
    var count = try mul(try mul(q, n), 2); // source probabilities and selection transpose
    count = try add(count, try mul(try mul(q, m), 5)); // basis, fitted probabilities, QR
    count = try add(count, try mul(try mul(q, d), 2)); // targets and QR RHS
    count = try add(count, try mul(try mul(m, m), 2)); // ridge and incremental QR rows
    count = try add(count, try mul(try mul(m, d), 4)); // QR RHS, result K/V, spare
    count = try add(count, try mul(try add(try add(n, m), try add(q, d)), 16));
    return mul(count, @sizeOf(f64));
}

pub fn fitWorkspaceBytes(n: usize, m: usize, q: usize, d: usize, heldout: usize, cfg: Config) !usize {
    const batch = validationBatchSize(heldout, cfg);
    const base = try workspaceBytes(n, m, @max(q, batch), d);
    const validation = try validationBytes(n, m, d, heldout, cfg);
    const extra = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, n, m, @max(q, batch), d) else 0;
    return add(try add(base, validation), extra);
}

pub fn valueWorkspaceBytes(q: usize, m: usize, d: usize, ridge: f64, cfg: Config) !usize {
    if (cfg.value_backend) |backend| return backend.workspace_fn(backend.context, q, m, d, ridge);
    return workspaceBytes(m, m, q, d);
}

fn validationBatchSize(queries: usize, cfg: Config) usize {
    return @min(queries, if (cfg.backend != null and (cfg.backend.?.attend_fn != null or cfg.backend.?.summary_fn != null)) (switch (cfg.backend.?.validation_batch) {
        .rows_64 => @as(usize, 64),
        .rows_256 => 256,
    }) else 1);
}

pub fn validationBytes(n: usize, m: usize, d: usize, queries: usize, cfg: Config) !usize {
    const batch = validationBatchSize(queries, cfg);
    const source_summary = cfg.reference_summary_backend != null or (cfg.backend != null and cfg.backend.?.summary_fn != null);
    const candidate_summary = cfg.backend != null and cfg.backend.?.summary_fn != null;
    const probability_rows = try add(if (source_summary) 0 else n, if (candidate_summary) 0 else m);
    const buffers = try mul(try mul(batch, try add(probability_rows, try add(try mul(d, 2), 2))), 8);
    const candidate_extra = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, m, 1, batch, d) else 0;
    const source_extra = if (cfg.reference_summary_backend) |reference|
        try reference.backend.workspace_fn(reference.backend.context, n, 1, batch, d)
    else if (cfg.backend) |backend|
        try backend.workspace_fn(backend.context, n, 1, batch, d)
    else
        0;
    const extra = @max(candidate_extra, source_extra);
    return add(buffers, extra);
}

test "validation workspace preserves larger source admission with reference hook off and on" {
    const BackendFixture = struct {
        fn workspace(_: ?*anyopaque, rows: usize, _: usize, _: usize, _: usize) !usize {
            return rows * 1000;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: Head, _: []const f64, _: []f64, _: []f64, _: Control) !void {}
    };
    const backend: Backend = .{ .validation_batch = .rows_256, .workspace_fn = BackendFixture.workspace, .summary_fn = BackendFixture.summary };
    const base: Config = .{ .target_rows = 1, .backend = backend };
    const without = try validationBytes(10, 3, 2, 4, base);
    var cached = base;
    cached.reference_summary_backend = .{ .backend = backend };
    const with = try validationBytes(10, 3, 2, 4, cached);
    const buffers = 4 * (2 * 2 + 2) * @sizeOf(f64);
    try std.testing.expectEqual(buffers + 10_000, without);
    try std.testing.expectEqual(without, with);
}

/// Caller-owned row-major buffers; never materializes all held-out queries in
/// one allocation. Control is checked before and after accelerated operations.
pub fn attendBatch(allocator: std.mem.Allocator, h: Head, queries: []const f64, probabilities: []f64, targets: ?[]f64, masses: ?[]f64, cfg: Config) !void {
    try cfg.control.check();
    if (h.dim == 0 or h.rows == 0 or queries.len == 0 or queries.len % h.dim != 0) return error.InvalidAttentionShape;
    const q = queries.len / h.dim;
    if (probabilities.len != try mul(q, h.rows)) return error.InvalidAttentionShape;
    if (targets) |out| if (out.len != queries.len) return error.InvalidAttentionShape;
    if (masses) |out| if (out.len != q) return error.InvalidAttentionShape;
    if (cfg.backend) |backend| if (backend.attend_fn) |call| {
        try call(backend.context, allocator, h, queries, probabilities, targets, masses, cfg.control);
        try finite(probabilities);
        if (targets) |out| try finite(out);
        if (masses) |out| try finite(out);
        try cfg.control.check();
        return;
    };
    for (0..q) |i| {
        try cfg.control.check();
        const mass = try attend(h, queries[i * h.dim ..][0..h.dim], probabilities[i * h.rows ..][0..h.rows], if (targets) |out| out[i * h.dim ..][0..h.dim] else null);
        if (masses) |out| out[i] = mass;
    }
}

pub fn attendSummary(allocator: std.mem.Allocator, head: Head, queries: []const f64, targets: []f64, masses: []f64, cfg: Config) !void {
    try cfg.control.check();
    if (head.dim == 0 or head.rows == 0 or queries.len == 0 or queries.len % head.dim != 0 or
        targets.len != queries.len or masses.len != queries.len / head.dim) return error.InvalidAttentionShape;
    const backend = cfg.backend orelse return error.UnsupportedAttentionSummary;
    const call = backend.summary_fn orelse return error.UnsupportedAttentionSummary;
    try call(backend.context, allocator, head, queries, targets, masses, cfg.control);
    try finite(targets);
    try finite(masses);
    try cfg.control.check();
}

pub fn validationMetrics(allocator: std.mem.Allocator, source: Head, compact: Head, queries: []const f64, cfg: Config) !Metrics {
    return (try validationMetricsInternal(allocator, source, compact, queries, cfg, false)).metrics;
}

/// DEV-only scalar detail. Reuses the ordinary backend-null validation
/// buffers and preserves its metric accumulation order and formulas.
pub fn validationMetricsDetailed(allocator: std.mem.Allocator, source: Head, compact: Head, queries: []const f64, cfg: Config) !ValidationObservation {
    if (cfg.backend != null) return error.InvalidFitConfig;
    var direct = cfg;
    direct.reference_summary_backend = null;
    return validationMetricsInternal(allocator, source, compact, queries, direct, true);
}

fn validationMetricsInternal(allocator: std.mem.Allocator, source: Head, compact: Head, queries: []const f64, cfg: Config, observe: bool) !ValidationObservation {
    if (source.dim == 0 or source.dim != compact.dim or queries.len == 0 or queries.len % source.dim != 0) return error.InvalidAttentionShape;
    const d = source.dim;
    const count = queries.len / d;
    const batch = validationBatchSize(count, cfg);
    const candidate_summary = cfg.backend != null and cfg.backend.?.summary_fn != null;
    const source_summary = candidate_summary or cfg.reference_summary_backend != null;
    const p = try allocator.alloc(f64, if (source_summary) 0 else try mul(batch, source.rows));
    defer allocator.free(p);
    const ph = try allocator.alloc(f64, if (candidate_summary) 0 else try mul(batch, compact.rows));
    defer allocator.free(ph);
    const y = try allocator.alloc(f64, try mul(batch, d));
    defer allocator.free(y);
    const yh = try allocator.alloc(f64, try mul(batch, d));
    defer allocator.free(yh);
    const mass = try allocator.alloc(f64, batch);
    defer allocator.free(mass);
    const fitted_mass = try allocator.alloc(f64, batch);
    defer allocator.free(fitted_mass);
    var result: Metrics = .{};
    var detail: ValidationDetail = .{ .worst_query_index = 0, .worst_log_mass_ratio = 0, .worst_relative_mass_error = 0, .original_peak_row = 0, .compact_peak_row = 0, .original_peak_probability = 0, .compact_peak_probability = 0, .queries_over_0_05 = 0, .queries_over_0_5 = 0, .queries_over_0_9 = 0, .mean_reference_peak_probability = 0 };
    var reference_peak_sum: f64 = 0;
    var error_norm: f64 = 0;
    var target_norm: f64 = 0;
    var offset: usize = 0;
    while (offset < count) {
        const take = @min(batch, count - offset);
        const qs = queries[offset * d ..][0 .. take * d];
        if (source_summary) {
            if (cfg.reference_summary_backend) |reference| {
                var reference_cfg = cfg;
                reference_cfg.backend = reference.backend;
                reference_cfg.reference_summary_backend = null;
                try attendSummary(allocator, source, qs, y[0 .. take * d], mass[0..take], reference_cfg);
            } else try attendSummary(allocator, source, qs, y[0 .. take * d], mass[0..take], cfg);
        } else try attendBatch(allocator, source, qs, p[0 .. take * source.rows], y[0 .. take * d], mass[0..take], cfg);
        if (candidate_summary) {
            try attendSummary(allocator, compact, qs, yh[0 .. take * d], fitted_mass[0..take], cfg);
        } else {
            try attendBatch(allocator, compact, qs, ph[0 .. take * compact.rows], yh[0 .. take * d], fitted_mass[0..take], cfg);
        }
        if (cfg.reference_summary_backend) |reference| if (reference.candidate_delegate_fn) |record| record(reference.observer_context);
        for (mass[0..take], fitted_mass[0..take], 0..) |a, b, bi| {
            const relative = @abs(@exp(b - a) - 1);
            if (!std.math.isFinite(relative)) return error.CompactionFitRejected;
            result.max_relative_mass_error = @max(result.max_relative_mass_error, relative);
            if (observe) {
                var source_peak: usize = 0;
                var compact_peak: usize = 0;
                for (p[bi * source.rows ..][0..source.rows], 0..) |value, row| {
                    if (value > p[bi * source.rows + source_peak]) source_peak = row;
                }
                for (ph[bi * compact.rows ..][0..compact.rows], 0..) |value, row| {
                    if (value > ph[bi * compact.rows + compact_peak]) compact_peak = row;
                }
                const source_probability = p[bi * source.rows + source_peak];
                const compact_probability = ph[bi * compact.rows + compact_peak];
                reference_peak_sum += source_probability;
                detail.queries_over_0_05 += @intFromBool(relative > 0.05);
                detail.queries_over_0_5 += @intFromBool(relative > 0.5);
                detail.queries_over_0_9 += @intFromBool(relative > 0.9);
                if (offset + bi == 0 or relative > detail.worst_relative_mass_error) detail = .{ .worst_query_index = offset + bi, .worst_log_mass_ratio = b - a, .worst_relative_mass_error = relative, .original_peak_row = source_peak, .compact_peak_row = compact_peak, .original_peak_probability = source_probability, .compact_peak_probability = compact_probability, .queries_over_0_05 = detail.queries_over_0_05, .queries_over_0_5 = detail.queries_over_0_5, .queries_over_0_9 = detail.queries_over_0_9, .mean_reference_peak_probability = 0 };
            }
        }
        for (y[0 .. take * d], yh[0 .. take * d]) |a, b| {
            error_norm = std.math.hypot(error_norm, a - b);
            target_norm = std.math.hypot(target_norm, a);
        }
        offset += take;
    }
    result.relative_output_rmse = error_norm / @max(target_norm, 1e-12);
    if (!std.math.isFinite(result.max_relative_mass_error) or !std.math.isFinite(result.relative_output_rmse))
        return error.CompactionFitRejected;
    result.quality_evaluated = true;
    if (observe) detail.mean_reference_peak_probability = reference_peak_sum / @as(f64, @floatFromInt(count));
    return .{ .metrics = result, .detail = detail };
}

test "context standalone validation metrics publish execution only after finite completion" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const source: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const compact: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const metrics = try validationMetrics(std.testing.allocator, source, compact, &.{ 0.25, 0.75 }, .{ .target_rows = 2 });
    try std.testing.expect(metrics.quality_evaluated);
    try std.testing.expectEqual(@as(f64, 0), metrics.max_relative_mass_error);
    try std.testing.expectEqual(@as(f64, 0), metrics.relative_output_rmse);
    // A completed evaluation can reject quality. Verify nonzero metrics with
    // a closed-form scalar reference, so an unevaluated zero result cannot
    // satisfy the execution flag contract.
    const one: Head = .{ .keys = &.{0}, .values = &.{3}, .bias = &.{@log(@as(f64, 2))}, .rows = 1, .dim = 1, .scale = 1 };
    const different = try validationMetrics(std.testing.allocator, source, one, &.{1}, .{ .target_rows = 1 });
    const expected_mass = (@exp(@as(f64, 1)) - 1) / (@exp(@as(f64, 1)) + 1);
    try std.testing.expect(different.quality_evaluated);
    try std.testing.expectApproxEqAbs(expected_mass, different.max_relative_mass_error, 1e-12);
    try std.testing.expectApproxEqAbs(expected_mass / (3 + expected_mass), different.relative_output_rmse, 1e-12);
    try std.testing.expectError(error.Cancelled, validationMetrics(std.testing.allocator, source, compact, &.{0.25}, .{
        .target_rows = 2,
        .control = .{ .check_fn = Cancel.check },
    }));
}

test "validation detail is scalar-only and metric-identical" {
    const source: Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 1, 2, 3 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1 };
    const compact: Head = .{ .keys = &.{ -1, 1 }, .values = &.{ 1, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const queries = [_]f64{ -0.5, 0.25, 1 };
    const cfg: Config = .{ .target_rows = 1 };
    const ordinary = try validationMetrics(std.testing.allocator, source, compact, &queries, cfg);
    const observed = try validationMetricsDetailed(std.testing.allocator, source, compact, &queries, cfg);
    try std.testing.expectEqual(ordinary, observed.metrics);
    try std.testing.expectEqual(ordinary.max_relative_mass_error, observed.detail.worst_relative_mass_error);
    try std.testing.expect(observed.detail.worst_query_index < queries.len);
    try std.testing.expect(observed.detail.original_peak_row < source.rows);
    try std.testing.expect(observed.detail.compact_peak_row < compact.rows);
    try std.testing.expect(observed.detail.queries_over_0_9 <= observed.detail.queries_over_0_5);
    try std.testing.expect(observed.detail.queries_over_0_5 <= observed.detail.queries_over_0_05);
    try std.testing.expect(observed.detail.mean_reference_peak_probability > 0 and observed.detail.mean_reference_peak_probability <= 1);
}

fn solveValues(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, cfg: Config, out: []f64) !void {
    try cfg.control.check();
    if (cfg.value_backend) |backend| {
        const required = try backend.workspace_fn(backend.context, q, m, d, ridge);
        if (required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
        try cfg.control.check();
        try backend.fit_fn(backend.context, allocator, a, y, q, m, d, ridge, cfg.control, out);
        return;
    }
    if (cfg.backend) |backend| if (backend.values_fn) |call| {
        try call(backend.context, allocator, a, y, q, m, d, ridge, cfg.control, out);
        try finite(out);
        try cfg.control.check();
        return;
    };
    try fitValues(allocator, a, y, q, m, d, ridge, cfg.control, out);
}

/// Four FP64 accumulation chains keep independent held-out attention off a
/// serial scalar dependency chain. Products and sums remain FP64; the native
/// softcap, bias, finite checks, and all acceptance thresholds are unchanged.
fn attentionDot(query: []const f64, keys: []const f64) f64 {
    var lanes: @Vector(4, f64) = @splat(0);
    var offset: usize = 0;
    while (offset + 4 <= query.len) : (offset += 4) {
        const q: @Vector(4, f64) = query[offset..][0..4].*;
        const k: @Vector(4, f64) = keys[offset..][0..4].*;
        lanes += q * k;
    }
    var result: f64 = @reduce(.Add, lanes);
    while (offset < query.len) : (offset += 1) result += query[offset] * keys[offset];
    return result;
}

fn logit(h: Head, query: []const f64, row: usize) !f64 {
    var x = attentionDot(query, h.keys[row * h.dim ..][0..h.dim]);
    x *= h.scale;
    if (!std.math.isFinite(x)) return error.NonFiniteAttention;
    if (h.softcap > 0) x = h.softcap * std.math.tanh(x / h.softcap);
    x += if (h.bias) |b| b[row] else 0;
    if (!std.math.isFinite(x)) return error.NonFiniteAttention;
    return x;
}

/// Returns log(sum(exp(logits))) and optionally the normalized attention output.
/// Callers must validate the head first; output and scratch cannot alias inputs.
pub fn attend(h: Head, query: []const f64, probabilities: []f64, output: ?[]f64) !f64 {
    if (query.len != h.dim or probabilities.len != h.rows or h.rows == 0) return error.InvalidAttentionShape;
    if (output) |y| if (y.len != h.dim) return error.InvalidAttentionShape;
    var maximum: f64 = -std.math.inf(f64);
    for (probabilities, 0..) |*p, i| {
        p.* = try logit(h, query, i);
        maximum = @max(maximum, p.*);
    }
    var sum: f64 = 0;
    for (probabilities) |*p| {
        p.* = @exp(p.* - maximum);
        sum += p.*;
    }
    for (probabilities) |*p| p.* /= sum;
    if (output) |y| {
        @memset(y, 0);
        for (probabilities, 0..) |p, i| {
            for (y, h.values[i * h.dim ..][0..h.dim]) |*v, original| v.* += p * original;
        }
        try finite(y);
    }
    const mass = maximum + @log(sum);
    if (!std.math.isFinite(mass)) return error.NonFiniteAttention;
    return mass;
}

/// Coordinate descent NNLS, with a projected-gradient KKT convergence check.
/// A is normalized by each source row's partition function, so the target is 1.
fn fitMass(a: []const f64, q: usize, m: usize, weights: []f64, residual: []f64, norms: []f64, cfg: Config) !usize {
    @memset(weights, 0);
    @memset(residual, 1);
    @memset(norms, 0);
    for (0..q) |i| for (0..m) |j| {
        norms[j] += a[i * m + j] * a[i * m + j];
    };
    for (0..cfg.max_sweeps) |sweep| {
        try cfg.control.check();
        for (0..m) |j| {
            if (norms[j] == 0) continue;
            var gradient: f64 = 0;
            for (0..q) |i| gradient += a[i * m + j] * residual[i];
            const next = @max(0, weights[j] + gradient / norms[j]);
            const delta = next - weights[j];
            if (!std.math.isFinite(next)) return error.NonFiniteAttention;
            weights[j] = next;
            for (0..q) |i| residual[i] -= delta * a[i * m + j];
        }
        var violation: f64 = 0;
        for (0..m) |j| {
            if (norms[j] == 0) continue;
            var gradient: f64 = 0;
            for (0..q) |i| gradient += a[i * m + j] * residual[i];
            const projected = if (weights[j] > 0) @abs(gradient) else @max(0, gradient);
            violation = @max(violation, projected / @sqrt(norms[j]));
        }
        if (violation <= cfg.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(q)))) return sweep + 1;
    }
    return error.MassFitDidNotConverge;
}

// Lawson-Hanson active-set solve using scaled Householder QR, with the same
// direct residual/KKT check as coordinate descent. No normal equations.
pub fn fitMassActive(allocator: std.mem.Allocator, a: []const f64, q: usize, m: usize, weights: []f64, residual: []f64, norms: []f64, cfg: Config, warm: bool) !usize {
    return fitMassActiveTarget(allocator, a, null, q, m, weights, residual, norms, cfg, warm);
}

pub fn fitMassActiveTargetForTest(allocator: std.mem.Allocator, a: []const f64, target: []const f64, q: usize, m: usize, weights: []f64, residual: []f64, norms: []f64, cfg: Config, warm: bool) !usize {
    if (comptime !@import("builtin").is_test) @compileError("target-RHS solve is diagnostic-test-only");
    if (target.len != q) return error.InvalidAttentionShape;
    return fitMassActiveTarget(allocator, a, target, q, m, weights, residual, norms, cfg, warm);
}

/// Optional diagnostic backend seam for arbitrary-RHS selectors. It does not
/// alter production selection policy; callers still undergo the final native
/// selected-basis NNLS/KKT solve in `solveMatrices`.
pub fn fitMassActiveTargetDiagnostic(allocator: std.mem.Allocator, a: []const f64, target: []const f64, q: usize, m: usize, weights: []f64, residual: []f64, norms: []f64, cfg: Config, warm: bool) !usize {
    if (target.len != q) return error.InvalidAttentionShape;
    return fitMassActiveTarget(allocator, a, target, q, m, weights, residual, norms, cfg, warm);
}

pub fn unitCenteredMassRidgeWorkspaceBytes(query_rows: usize, selected_rows: usize, cfg: Config) !usize {
    if (query_rows == 0 or selected_rows == 0) return error.InvalidFitConfig;
    const augmented_rows = try add(query_rows, selected_rows);
    var caller_count = try mul(augmented_rows, selected_rows);
    caller_count = try add(caller_count, try mul(augmented_rows, 2));
    caller_count = try add(caller_count, try mul(selected_rows, 2));
    const caller_bytes = try mul(caller_count, @sizeOf(f64));
    _ = cfg;
    const nested_bytes = try workspaceBytes(selected_rows, selected_rows, augmented_rows, 1);
    return add(caller_bytes, nested_bytes);
}

/// Fixed-support TRAIN-only mass refit with lambda * ||w-1||^2. Weights are
/// committed only after the unchanged active-set solve and direct KKT pass.
pub fn refitMassWeightsUnitCenteredRidgeDiagnostic(allocator: std.mem.Allocator, probabilities: []const f64, target: []const f64, rows: usize, selected: []const usize, weights: []f64, ratio: f64, cfg: Config) !usize {
    if (target.len == 0 or rows == 0 or probabilities.len != try mul(target.len, rows) or selected.len == 0 or selected.len != weights.len or !std.math.isFinite(ratio) or ratio <= 0) return error.InvalidFitConfig;
    try finite(target);
    for (target) |value| if (value < 0) return error.InvalidFitConfig;
    for (selected, 0..) |index, i| if (index >= rows or std.mem.indexOfScalar(usize, selected[0..i], index) != null) return error.InvalidFitConfig;
    const required = try unitCenteredMassRidgeWorkspaceBytes(target.len, selected.len, cfg);
    if (required > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const augmented_rows = try add(target.len, selected.len);
    const basis = try allocator.alloc(f64, try mul(augmented_rows, selected.len));
    defer allocator.free(basis);
    const rhs = try allocator.alloc(f64, augmented_rows);
    defer allocator.free(rhs);
    const residual = try allocator.alloc(f64, augmented_rows);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, selected.len);
    defer allocator.free(norms);
    const working = try allocator.dupe(f64, weights);
    defer allocator.free(working);
    var energy: f64 = 0;
    for (0..target.len) |query| {
        if (query % 64 == 0) try cfg.control.check();
        for (selected, 0..) |index, column| {
            const value = probabilities[query * rows + index];
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            basis[query * selected.len + column] = value;
            energy += value * value;
        }
    }
    const lambda = @max(@as(f64, 1e-18), ratio * energy / @as(f64, @floatFromInt(selected.len)));
    if (!std.math.isFinite(lambda)) return error.NonFiniteAttention;
    const root = @sqrt(lambda);
    @memcpy(rhs[0..target.len], target);
    @memset(basis[target.len * selected.len ..], 0);
    for (0..selected.len) |row| {
        basis[(target.len + row) * selected.len + row] = root;
        rhs[target.len + row] = root;
    }
    var nested = cfg;
    // This diagnostic's augmented identity rows may exceed the accelerated
    // selector's query cap. Keep the fixed-support solve on the native scalar
    // Householder/KKT path; selection itself still uses the configured backend.
    nested.backend = null;
    nested.value_backend = null;
    var caller_count = try mul(augmented_rows, selected.len);
    caller_count = try add(caller_count, try mul(augmented_rows, 2));
    caller_count = try add(caller_count, try mul(selected.len, 2));
    nested.max_workspace_bytes -= try mul(caller_count, @sizeOf(f64));
    const solves = try fitMassActiveTargetDiagnostic(allocator, basis, rhs, augmented_rows, selected.len, working, residual, norms, nested, true);
    try cfg.control.check();
    @memcpy(weights, working);
    return solves;
}

pub fn validateConfigForTest(cfg: Config, rows: usize) !void {
    if (comptime !@import("builtin").is_test) @compileError("config validation is diagnostic-test-only");
    try cfg.validate(rows);
}

pub fn validateConfigDiagnostic(cfg: Config, rows: usize) !void {
    try cfg.validate(rows);
}

test "resident workspace admission permits bounded child configs" {
    var cfg: Config = .{ .target_rows = 1, .max_workspace_bytes = 1, .workspace_admission = .resident_native_hybrid_2g_v1 };
    try std.testing.expectEqual(@as(usize, 2 * 1024 * 1024 * 1024), try cfg.admittedCombinedCap());
    cfg.max_workspace_bytes = 2 * 1024 * 1024 * 1024 + 1;
    try std.testing.expectError(error.InvalidFitConfig, cfg.admittedCombinedCap());
}

fn fitMassActiveTarget(allocator: std.mem.Allocator, a: []const f64, target: ?[]const f64, q: usize, m: usize, weights: []f64, residual: []f64, norms: []f64, cfg: Config, warm: bool) !usize {
    // Keep allocation lengths aligned to the subsequent f64 workspace. The
    // fixed-buffer allocator cannot reclaim alignment gaps on free; repeated
    // active-set solves must release every byte before the next selection step.
    const active_storage = try allocator.alloc(bool, std.mem.alignForward(usize, m, @alignOf(f64)));
    defer allocator.free(active_storage);
    const active = active_storage[0..m];
    const indices = try allocator.alloc(usize, m);
    defer allocator.free(indices);
    const matrix = try allocator.alloc(f64, try mul(q, m));
    defer allocator.free(matrix);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    const solution = try allocator.alloc(f64, m);
    defer allocator.free(solution);
    const trial = try allocator.alloc(f64, m);
    defer allocator.free(trial);
    if (target) |values| {
        if (values.len != q) return error.InvalidAttentionShape;
        @memcpy(rhs, values);
        try finite(rhs);
    } else @memset(rhs, 1);
    @memset(norms, 0);
    if (!warm) @memset(weights, 0);
    for (weights, active) |w, *flag| {
        if (!std.math.isFinite(w) or w < 0) return error.NonFiniteAttention;
        flag.* = w > 0;
    }
    for (0..q) |i| for (0..m) |j| {
        norms[j] += a[i * m + j] * a[i * m + j];
    };
    for (norms, weights, active) |*norm, *weight, *flag| {
        if (!std.math.isFinite(norm.*)) return error.NonFiniteAttention;
        norm.* = @sqrt(norm.*);
        if (norm.* == 0) {
            weight.* = 0;
            flag.* = false;
        }
    }
    const tolerance = cfg.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(q)));
    const trace = struct {
        fn digestF64(values: []const f64) [32]u8 {
            var result: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(values), &result, .{});
            return result;
        }
        fn digestActive(values: []const bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            for (values) |value| hash.update(&.{@intFromBool(value)});
            return hash.finalResult();
        }
        fn countActive(values: []const bool) usize {
            var count: usize = 0;
            for (values) |value| count += @intFromBool(value);
            return count;
        }
    };
    var solves: usize = 0;
    var residual_holds_outer_entry = false;
    defer if (residual_holds_outer_entry) {
        @memcpy(residual, rhs);
        for (0..q) |i| for (0..m) |j| {
            residual[i] -= a[i * m + j] * weights[j];
        };
    };
    while (true) {
        try cfg.control.check();
        residual_holds_outer_entry = false;
        @memcpy(residual, rhs);
        for (0..q) |i| for (0..m) |j| {
            residual[i] -= a[i * m + j] * weights[j];
        };
        try finite(residual);
        var violation: f64 = 0;
        var best: ?usize = null;
        var best_gradient: f64 = tolerance;
        for (0..m) |j| {
            if (norms[j] == 0) continue;
            var gradient: f64 = 0;
            for (0..q) |i| gradient += a[i * m + j] * residual[i];
            gradient /= norms[j];
            if (!std.math.isFinite(gradient)) return error.NonFiniteAttention;
            violation = @max(violation, if (weights[j] > 0) @abs(gradient) else @max(0, gradient));
            if (!active[j] and gradient > best_gradient) {
                best = j;
                best_gradient = gradient;
            }
        }
        if (cfg.active_nnls_observer) |observer| observer.record(.{
            .phase = .outer,
            .q = q,
            .m = m,
            .solves = solves,
            .best = best,
            .active_count = trace.countActive(active),
            .violation = violation,
            .tolerance = tolerance,
            .weights_sha256 = trace.digestF64(weights),
            .active_sha256 = trace.digestActive(active),
            .fixed_point_enabled = cfg.reject_exact_active_fixed_point,
        });
        if (violation <= tolerance) return @max(solves, 1);
        if (solves >= cfg.max_sweeps) {
            std.debug.print("attention_match_active_nnls phase=outer_exhausted q={d} m={d} solves={d} active_count={d} violation={d} tolerance={d} value_backend={} diagnostic_only=true\n", .{ q, m, solves, trace.countActive(active), violation, tolerance, cfg.value_backend != null });
            return error.MassFitDidNotConverge;
        }
        if (cfg.reject_exact_active_fixed_point and q >= m) {
            @memcpy(residual[0..m], weights);
            residual_holds_outer_entry = true;
        }
        if (best) |j| active[j] = true;
        while (true) {
            try cfg.control.check();
            if (solves >= cfg.max_sweeps) {
                std.debug.print("attention_match_active_nnls phase=inner_exhausted q={d} m={d} solves={d} active_count={d} violation={d} tolerance={d} value_backend={} diagnostic_only=true\n", .{ q, m, solves, trace.countActive(active), violation, tolerance, cfg.value_backend != null });
                return error.MassFitDidNotConverge;
            }
            var count: usize = 0;
            for (active, 0..) |flag, j| if (flag) {
                indices[count] = j;
                count += 1;
            };
            if (count == 0 or count > q) return error.SingularValueFit;
            for (0..q) |i| for (indices[0..count], 0..) |j, c| {
                matrix[i * count + c] = a[i * m + j] / norms[j];
            };
            // The active columns are added only for a positive projected
            // gradient, preserving independence to the KKT tolerance.
            try solveValues(allocator, matrix[0 .. q * count], rhs, q, count, 1, 0, cfg, solution[0..count]);
            solves += 1;
            @memset(trial, 0);
            var feasible = true;
            for (indices[0..count], solution[0..count]) |j, z| {
                trial[j] = z / norms[j];
                if (!std.math.isFinite(trial[j])) return error.NonFiniteAttention;
                if (trial[j] <= 0) feasible = false;
            }
            if (cfg.active_nnls_observer) |observer| {
                var max_delta: f64 = 0;
                for (weights, trial) |current, next| max_delta = @max(max_delta, @abs(next - current));
                observer.record(.{
                    .phase = .inner,
                    .q = q,
                    .m = m,
                    .solves = solves,
                    .best = best,
                    .active_count = trace.countActive(active),
                    .violation = violation,
                    .tolerance = tolerance,
                    .feasible = feasible,
                    .max_delta = max_delta,
                    .weights_sha256 = trace.digestF64(weights),
                    .active_sha256 = trace.digestActive(active),
                    .trial_sha256 = trace.digestF64(trial),
                    .fixed_point_enabled = cfg.reject_exact_active_fixed_point,
                });
            }
            if (feasible) {
                // The residual buffer is otherwise dead throughout the inner
                // Lawson-Hanson loop. When it contains the exact failed outer
                // entry weights, a feasible trial equal to that entry has
                // returned to the same deterministic state even after an
                // entering-variable drop. Restore observable state before the
                // diagnostic-only early rejection.
                if (residual_holds_outer_entry and std.mem.eql(u8, std.mem.sliceAsBytes(residual[0..m]), std.mem.sliceAsBytes(trial))) {
                    @memcpy(weights, residual[0..m]);
                    @memcpy(residual, rhs);
                    for (0..q) |i| for (0..m) |j| {
                        residual[i] -= a[i * m + j] * weights[j];
                    };
                    try finite(residual);
                    residual_holds_outer_entry = false;
                    std.debug.print("attention_match_active_nnls phase=exact_fixed_point q={d} m={d} solves={d} active_count={d} violation={d} tolerance={d} value_backend={} diagnostic_only=true\n", .{ q, m, solves, trace.countActive(active), violation, tolerance, cfg.value_backend != null });
                    return error.MassFitDidNotConverge;
                }
                @memcpy(weights, trial);
                break;
            }
            var alpha: f64 = 1;
            var drop: ?usize = null;
            for (indices[0..count]) |j| if (trial[j] <= 0) {
                const fraction = if (weights[j] == 0) 0 else weights[j] / (weights[j] - trial[j]);
                if (drop == null or fraction < alpha) {
                    alpha = fraction;
                    drop = j;
                }
            };
            if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return error.NonFiniteAttention;
            for (weights, 0..) |*w, j| {
                w.* += alpha * (trial[j] - w.*);
                if (w.* <= 0 or (drop != null and j == drop.?)) {
                    w.* = 0;
                    active[j] = false;
                }
            }
            if (cfg.active_nnls_observer) |observer| observer.record(.{
                .phase = .drop,
                .q = q,
                .m = m,
                .solves = solves,
                .best = best,
                .active_count = trace.countActive(active),
                .violation = violation,
                .tolerance = tolerance,
                .alpha = alpha,
                .drop = drop,
                .max_delta = blk: {
                    var max_delta: f64 = 0;
                    for (weights, trial) |current, next| max_delta = @max(max_delta, @abs(next - current));
                    break :blk max_delta;
                },
                .weights_sha256 = trace.digestF64(weights),
                .active_sha256 = trace.digestActive(active),
                .trial_sha256 = trace.digestF64(trial),
                .fixed_point_enabled = cfg.reject_exact_active_fixed_point,
            });
        }
    }
}

/// Householder QR of ridge-augmented [A; sqrt(lambda) I], avoiding normal
/// equations (which square the condition number). RHS columns are transformed
/// together, and one scratch vector is reused for each reflector.
/// Test-only access to the unchanged scalar Householder oracle. Production
/// accelerators cannot invoke this seam or replace the direct NNLS/KKT checks.
pub fn fitValuesReferenceForTest(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: Control, out: []f64) !void {
    if (comptime !@import("builtin").is_test) @compileError("reference value solve is test-only");
    try fitValues(allocator, a, y, q, m, d, ridge, control, out);
}

pub fn fitValuesWithBackendForTest(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, cfg: Config, out: []f64) !void {
    if (comptime !@import("builtin").is_test) @compileError("backend value solve is diagnostic-test-only");
    try solveValues(allocator, a, y, q, m, d, ridge, cfg, out);
}

pub fn fitValuesWithBackendDiagnostic(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, cfg: Config, out: []f64) !void {
    return solveValues(allocator, a, y, q, m, d, ridge, cfg, out);
}

fn fitValues(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: Control, out: []f64) !void {
    const rows = try add(q, m);
    const qr = try allocator.alloc(f64, try mul(rows, m));
    defer allocator.free(qr);
    const rhs = try allocator.alloc(f64, try mul(rows, d));
    defer allocator.free(rhs);
    const v = try allocator.alloc(f64, rows);
    defer allocator.free(v);
    const dots = try allocator.alloc(f64, @max(m, d));
    defer allocator.free(dots);
    @memset(qr, 0);
    @memset(rhs, 0);
    @memcpy(qr[0..a.len], a);
    @memcpy(rhs[0..y.len], y);
    for (0..m) |j| qr[(q + j) * m + j] = @sqrt(ridge);
    for (0..m) |j| {
        try control.check();
        var norm: f64 = 0;
        for (j..rows) |i| norm = std.math.hypot(norm, qr[i * m + j]);
        if (!std.math.isFinite(norm) or norm == 0) return error.SingularValueFit;
        const alpha = if (qr[j * m + j] >= 0) -norm else norm;
        for (j..rows) |i| v[i] = qr[i * m + j];
        v[j] -= alpha;
        var vn: f64 = 0;
        for (v[j..]) |x| vn = std.math.hypot(vn, x);
        for (v[j..]) |*x| x.* /= vn;
        @memset(dots[0..m], 0);
        for (j..rows) |i| for (j..m) |col| {
            dots[col] += v[i] * qr[i * m + col];
        };
        for (j..rows) |i| for (j..m) |col| {
            qr[i * m + col] -= 2 * v[i] * dots[col];
        };
        @memset(dots[0..d], 0);
        for (j..rows) |i| for (0..d) |col| {
            dots[col] += v[i] * rhs[i * d + col];
        };
        for (j..rows) |i| for (0..d) |col| {
            rhs[i * d + col] -= 2 * v[i] * dots[col];
        };
    }
    var j = m;
    while (j > 0) {
        j -= 1;
        for (0..d) |col| {
            var x = rhs[j * d + col];
            for (j + 1..m) |k| x -= qr[j * m + k] * out[k * d + col];
            out[j * d + col] = x / qr[j * m + j];
        }
    }
    try finite(out);
}

/// Device adapters may supply only derived attention matrices. The original
/// source K/V and captured queries remain device-resident. Rows of P must be
/// normalized probabilities for the immutable source; Y is its attention output.
pub const MatrixInput = struct {
    probabilities: []const f64,
    targets: []const f64,
    bias: ?[]const f64 = null,
    rows: usize,
    dim: usize,
    query_rows: usize,

    fn validate(self: MatrixInput) !void {
        return self.validateFeatures(true);
    }

    fn validateFeatures(self: MatrixInput, require_normalized: bool) !void {
        if (self.rows == 0 or self.dim == 0 or self.query_rows == 0 or
            self.probabilities.len != try mul(self.query_rows, self.rows) or
            self.targets.len != try mul(self.query_rows, self.dim)) return error.InvalidAttentionShape;
        try finite(self.targets);
        if (self.bias) |bias| {
            if (bias.len != self.rows) return error.InvalidAttentionShape;
            try finite(bias);
        }
        for (0..self.query_rows) |q| {
            var sum: f64 = 0;
            for (self.probabilities[q * self.rows ..][0..self.rows]) |p| {
                if (!std.math.isFinite(p) or p < 0 or p > 1) return error.InvalidAttentionProbabilities;
                sum += p;
            }
            // Allow normalization roundoff from an explicitly FP32 device
            // frontend; the native OMP/NNLS/QR solve itself remains FP64.
            if (require_normalized and @abs(sum - 1) > 1e-5) return error.InvalidAttentionProbabilities;
        }
    }
};

/// Unvalidated private staging. This object has no publication authority:
/// acceptance must evaluate rounded K/V and f32 bias against the full source.
pub const MatrixResult = struct {
    allocator: std.mem.Allocator,
    values: []f64,
    bias: []f64,
    source_indices: []usize,
    mass_weights: ?[]f64 = null,
    metrics: Metrics = .{},

    pub fn deinit(self: *MatrixResult) void {
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.allocator.free(self.source_indices);
        if (self.mass_weights) |weights| self.allocator.free(weights);
        self.* = undefined;
    }
};

/// Conservative simultaneous host scratch + private result bound. Caller-owned
/// P/Y, device buffers, and the live source must be admitted separately.
pub fn matrixWorkspaceBytes(n: usize, m: usize, q: usize, d: usize) !usize {
    return workspaceBytes(n, m, q, d);
}

pub fn fitMatrices(allocator: std.mem.Allocator, input: MatrixInput, cfg: Config) !MatrixResult {
    return fitMatricesTarget(allocator, input, cfg, null, .normalized);
}

pub const DiagnosticSelectionScore = enum { normalized, raw };

/// Diagnostic-only arbitrary mass RHS for reference-objective experiments.
/// Production callers retain the implicit all-ones objective through
/// `fitMatrices`; no Config or Backend ABI field exposes this seam.
pub fn fitMatricesMassTargetForTest(allocator: std.mem.Allocator, input: MatrixInput, cfg: Config, mass_rhs: []const f64) !MatrixResult {
    return fitMatricesMassTargetPolicyForTest(allocator, input, cfg, mass_rhs, .normalized);
}

pub fn fitMatricesMassTargetPolicyForTest(allocator: std.mem.Allocator, input: MatrixInput, cfg: Config, mass_rhs: []const f64, selection_score: DiagnosticSelectionScore) !MatrixResult {
    if (comptime !@import("builtin").is_test) @compileError("mass target is diagnostic-test-only");
    if (mass_rhs.len != input.query_rows or cfg.selection != .residual_omp or cfg.mass_solver != .active_set) return error.InvalidFitConfig;
    try finite(mass_rhs);
    for (mass_rhs) |value| if (value < 0) return error.InvalidFitConfig;
    var native_cfg = cfg;
    if (native_cfg.backend) |backend| {
        var native_backend = backend;
        native_backend.select_fn = null;
        if (selection_score != .raw) native_backend.raw_select_fn = null;
        native_cfg.backend = native_backend;
    }
    return fitMatricesTarget(allocator, input, native_cfg, mass_rhs, selection_score);
}

pub fn fitMatricesMassTargetPolicyDiagnostic(allocator: std.mem.Allocator, input: MatrixInput, cfg: Config, mass_rhs: []const f64, selection_score: DiagnosticSelectionScore) !MatrixResult {
    if (mass_rhs.len != input.query_rows or cfg.selection != .residual_omp or cfg.mass_solver != .active_set) return error.InvalidFitConfig;
    try finite(mass_rhs);
    for (mass_rhs) |value| if (value < 0) return error.InvalidFitConfig;
    var native_cfg = cfg;
    if (native_cfg.backend) |backend| {
        var native_backend = backend;
        native_backend.select_fn = null;
        if (selection_score != .raw) native_backend.raw_select_fn = null;
        native_cfg.backend = native_backend;
    }
    return fitMatricesTarget(allocator, input, native_cfg, mass_rhs, selection_score);
}

/// Selection-only raw/normalized OMP plus mass NNLS. This result has no values
/// or bias and therefore cannot be mistaken for a publishable fitted head.
pub const MassSelectionResult = struct {
    allocator: std.mem.Allocator,
    indices: []usize,
    weights: []f64,
    query_rows_consumed: usize,
    metrics: Metrics = .{},

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const mass_selection_policy = "normalized_probability_raw_omp_mass_only_active_nnls_v1";

/// Preserves the established conservative matrix admission bound, including
/// retained mass weights, while omitting value-result allocations at runtime.
pub fn massSelectionWorkspaceBytes(rows: usize, target_rows: usize, query_rows: usize) !usize {
    return add(try matrixWorkspaceBytes(rows, target_rows, query_rows, 1), try mul(target_rows, @sizeOf(f64)));
}

pub fn selectMassTargetPolicyDiagnostic(allocator: std.mem.Allocator, probabilities: []const f64, mass_rhs: []const f64, rows: usize, cfg: Config, selection_score: DiagnosticSelectionScore) !MassSelectionResult {
    if (cfg.metrics) |metrics| metrics.* = .{};
    try cfg.control.check();
    if (rows == 0 or mass_rhs.len == 0 or probabilities.len != try mul(mass_rhs.len, rows) or
        cfg.target_rows == 0 or cfg.target_rows >= rows or cfg.selection != .residual_omp or cfg.mass_solver != .active_set)
        return error.InvalidFitConfig;
    try cfg.validate(rows);
    try finite(mass_rhs);
    var has_progress = false;
    for (mass_rhs) |value| {
        if (value < 0) return error.InvalidFitConfig;
        has_progress = has_progress or value > 0;
    }
    if (!has_progress) return error.InsufficientSelectionProgress;
    for (probabilities) |value| if (!std.math.isFinite(value) or value < 0 or value > 1) return error.InvalidAttentionProbabilities;
    var native_cfg = cfg;
    if (native_cfg.backend) |backend| {
        var native_backend = backend;
        native_backend.select_fn = null;
        if (selection_score != .raw) native_backend.raw_select_fn = null;
        native_cfg.backend = native_backend;
    }
    const workspace = try massSelectionWorkspaceBytes(rows, cfg.target_rows, mass_rhs.len);
    if (workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const legacy_result_bytes = try mul(cfg.target_rows, 3 * @sizeOf(f64) + @sizeOf(usize));
    const scratch_bytes = std.math.sub(usize, workspace, legacy_result_bytes) catch return error.InvalidAttentionShape;
    const scratch_buffer = try allocator.alignedAlloc(u8, .of(f64), scratch_bytes);
    defer allocator.free(scratch_buffer);
    var fixed = std.heap.FixedBufferAllocator.init(scratch_buffer);
    const indices = try allocator.alloc(usize, cfg.target_rows);
    errdefer allocator.free(indices);
    const weights = try allocator.alloc(f64, cfg.target_rows);
    errdefer allocator.free(weights);
    var result: MassSelectionResult = .{ .allocator = allocator, .indices = indices, .weights = weights, .query_rows_consumed = mass_rhs.len };
    defer if (cfg.metrics) |metrics| {
        metrics.* = result.metrics;
    };
    try solveMatrices(&fixed, .{ .probabilities = probabilities, .targets = &.{}, .rows = rows, .dim = 1, .query_rows = mass_rhs.len }, native_cfg, mass_rhs, selection_score, &.{}, &.{}, indices, weights, &result.metrics, true);
    try cfg.control.check();
    return result;
}

test "mass-only selection preserves positive dense indices and weights" {
    const probabilities = [_]f64{ 0.7, 0.2, 0.1, 0.1, 0.65, 0.25, 0.2, 0.1, 0.7 };
    const rho = [_]f64{ 1, 1, 1 };
    const cfg: Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .retain_mass_weights = true, .max_workspace_bytes = 1 << 20 };
    var legacy = try fitMatricesMassTargetPolicyDiagnostic(std.testing.allocator, .{ .probabilities = &probabilities, .targets = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .query_rows = 3 }, cfg, &rho, .raw);
    defer legacy.deinit();
    var mass = try selectMassTargetPolicyDiagnostic(std.testing.allocator, &probabilities, &rho, 3, cfg, .raw);
    defer mass.deinit();
    try std.testing.expectEqualSlices(usize, legacy.source_indices, mass.indices);
    try std.testing.expectEqualSlices(f64, legacy.mass_weights.?, mass.weights);
}

/// Diagnostic-only reference objective. The existing attention frontend emits
/// normalized P and Y; each P row is then converted exactly as
/// A=P/max(P), z=sum(P)/max(P) before native OMP and NNLS.
pub fn fitShiftedForTest(allocator: std.mem.Allocator, source: Head, train: []const f64, cfg: Config) !MatrixResult {
    return fitShiftedSelectionForTest(allocator, source, train, cfg, .normalized);
}

/// Diagnostic-only normalized-attention control for the shifted objective.
/// It uses the same P/Y frontend, target row count, CPU selector, NNLS, and
/// value backend as `fitShiftedSelectionForTest`, but retains normalized P and
/// the ordinary all-ones mass RHS.
pub fn fitNormalizedSelectionForTest(allocator: std.mem.Allocator, source: Head, train: []const f64, cfg: Config, selection_score: DiagnosticSelectionScore) !MatrixResult {
    if (comptime !@import("builtin").is_test) @compileError("normalized selection control is diagnostic-test-only");
    try source.validate();
    try cfg.validate(source.rows);
    if (train.len == 0 or train.len % source.dim != 0) return error.InvalidAttentionShape;
    const q = train.len / source.dim;
    const probability_count = try mul(q, source.rows);
    const target_count = try mul(q, source.dim);
    const input_bytes = try mul(try add(try add(probability_count, target_count), q), @sizeOf(f64));
    const solver_bytes = try matrixWorkspaceBytes(source.rows, cfg.target_rows, q, source.dim);
    const attend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, source.rows, 1, q, source.dim) else 0;
    if (try add(input_bytes, @max(solver_bytes, attend_bytes)) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const probabilities = try allocator.alloc(f64, probability_count);
    defer allocator.free(probabilities);
    const targets = try allocator.alloc(f64, target_count);
    defer allocator.free(targets);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(rhs, 1);
    try attendBatch(allocator, source, train, probabilities, targets, null, cfg);
    return fitMatricesMassTargetPolicyForTest(allocator, .{ .probabilities = probabilities, .targets = targets, .bias = source.bias, .rows = source.rows, .dim = source.dim, .query_rows = q }, cfg, rhs, selection_score);
}

/// Diagnostic-only fixed-index IRLS coefficient refit. Selection is caller
/// owned and immutable; heldout queries never enter this function.
pub fn refitFixedIndicesIrlsMatricesForTest(allocator: std.mem.Allocator, input: MatrixInput, selected: []const usize, cfg: Config) !MatrixResult {
    if (comptime !@import("builtin").is_test) @compileError("IRLS refit is diagnostic-test-only");
    try input.validate();
    try cfg.validate(input.rows);
    const q = input.query_rows;
    const m = selected.len;
    const d = input.dim;
    if (m == 0 or m != cfg.target_rows) return error.InvalidFitConfig;
    for (selected, 0..) |idx, i| if (idx >= input.rows or std.mem.indexOfScalar(usize, selected[0..i], idx) != null) return error.InvalidSelectionResult;
    const qm = try mul(q, m);
    const retained_f64_bytes = try mul(try add(try add(try mul(qm, 3), try mul(q, 3)), try add(try mul(m, 5), try mul(m, d))), @sizeOf(f64));
    const retained_bytes = try add(retained_f64_bytes, try mul(m, @sizeOf(usize)));
    const matrix_bytes = try matrixWorkspaceBytes(m, m, q, d);
    const backend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, m, m, q, d) else 0;
    const nested_bytes = @max(matrix_bytes, backend_bytes);
    if (try add(retained_bytes, nested_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const basis = try allocator.alloc(f64, qm);
    defer allocator.free(basis);
    for (0..q) |i| {
        for (selected, 0..) |idx, j| basis[i * m + j] = input.probabilities[i * input.rows + idx];
    }
    const weights = try allocator.alloc(f64, m);
    defer allocator.free(weights);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, m);
    defer allocator.free(norms);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(rhs, 1);
    _ = try fitMassActiveTarget(allocator, basis, rhs, q, m, weights, residual, norms, cfg, false);
    const scaled = try allocator.alloc(f64, qm);
    defer allocator.free(scaled);
    const scaled_rhs = try allocator.alloc(f64, q);
    defer allocator.free(scaled_rhs);
    const previous = try allocator.alloc(f64, m);
    defer allocator.free(previous);
    const candidate = try allocator.alloc(f64, m);
    defer allocator.free(candidate);
    const powers = [_]u8{ 4, 8, 16 };
    for (powers) |power| {
        const exponent = @as(f64, @floatFromInt(power - 2)) / 2;
        for (0..8) |_| {
            try cfg.control.check();
            @memcpy(previous, weights);
            var previous_objective: f64 = 0;
            var max_scale: f64 = 0;
            for (0..q) |i| {
                var fitted: f64 = 0;
                for (0..m) |j| fitted += basis[i * m + j] * previous[j];
                const relative = @abs(fitted - rhs[i]) / @max(@abs(rhs[i]), 1e-12);
                previous_objective += std.math.pow(f64, relative, @floatFromInt(power));
                const scale = @min(1e6, std.math.pow(f64, @max(relative, 1e-6), exponent));
                if (!std.math.isFinite(scale) or scale <= 0) return error.NonFiniteAttention;
                max_scale = @max(max_scale, scale);
                scaled_rhs[i] = rhs[i] * scale;
                for (0..m) |j| scaled[i * m + j] = basis[i * m + j] * scale;
            }
            if (!std.math.isFinite(previous_objective) or !std.math.isFinite(max_scale) or max_scale <= 0) return error.NonFiniteAttention;
            for (scaled_rhs) |*value| value.* /= max_scale;
            for (scaled) |*value| value.* /= max_scale;
            @memcpy(candidate, previous);
            _ = try fitMassActiveTarget(allocator, scaled, scaled_rhs, q, m, candidate, residual, norms, cfg, true);
            var accepted = false;
            var alpha: f64 = 1;
            for (0..8) |_| {
                try cfg.control.check();
                for (weights, previous, candidate) |*out, old, proposed| out.* = old + alpha * (proposed - old);
                var objective: f64 = 0;
                for (0..q) |i| {
                    var fitted: f64 = 0;
                    for (0..m) |j| fitted += basis[i * m + j] * weights[j];
                    const relative = @abs(fitted - rhs[i]) / @max(@abs(rhs[i]), 1e-12);
                    objective += std.math.pow(f64, relative, @floatFromInt(power));
                }
                if (std.math.isFinite(objective) and objective <= previous_objective) {
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if (!accepted) {
                @memcpy(weights, previous);
                break;
            }
        }
    }
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    for (selected, weights, bias) |idx, weight, *out| {
        if (!std.math.isFinite(weight) or weight < 0) return error.NonFiniteAttention;
        out.* = (if (input.bias) |source_bias| source_bias[idx] else 0) + @log(@max(weight, cfg.min_weight));
    }
    const fitted = try allocator.alloc(f64, qm);
    defer allocator.free(fitted);
    for (0..q) |i| {
        var sum: f64 = 0;
        for (0..m) |j| {
            fitted[i * m + j] = basis[i * m + j] * @max(weights[j], cfg.min_weight);
            sum += fitted[i * m + j];
        }
        if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
        for (fitted[i * m ..][0..m]) |*value| value.* /= sum;
    }
    const values = try allocator.alloc(f64, try mul(m, d));
    errdefer allocator.free(values);
    try solveValues(allocator, fitted, input.targets, q, m, d, cfg.ridge, cfg, values);
    const indices = try allocator.dupe(usize, selected);
    return .{ .allocator = allocator, .values = values, .bias = bias, .source_indices = indices };
}

pub fn refitNormalizedFixedIndicesIrlsForTest(allocator: std.mem.Allocator, source: Head, train: []const f64, selected: []const usize, cfg: Config) !MatrixResult {
    if (comptime !@import("builtin").is_test) @compileError("IRLS refit is diagnostic-test-only");
    try source.validate();
    try cfg.validate(source.rows);
    if (train.len == 0 or train.len % source.dim != 0) return error.InvalidAttentionShape;
    const q = train.len / source.dim;
    const probability_count = try mul(q, source.rows);
    const target_count = try mul(q, source.dim);
    const frontend_bytes = try mul(try add(probability_count, target_count), @sizeOf(f64));
    const attend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, source.rows, 1, q, source.dim) else 0;
    if (try add(frontend_bytes, attend_bytes) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    var nested_cfg = cfg;
    nested_cfg.max_workspace_bytes -= frontend_bytes;
    const probabilities = try allocator.alloc(f64, probability_count);
    defer allocator.free(probabilities);
    const targets = try allocator.alloc(f64, target_count);
    defer allocator.free(targets);
    try attendBatch(allocator, source, train, probabilities, targets, null, cfg);
    return refitFixedIndicesIrlsMatricesForTest(allocator, .{ .probabilities = probabilities, .targets = targets, .bias = source.bias, .rows = source.rows, .dim = source.dim, .query_rows = q }, selected, nested_cfg);
}

pub fn fitShiftedSelectionForTest(allocator: std.mem.Allocator, source: Head, train: []const f64, cfg: Config, selection_score: DiagnosticSelectionScore) !MatrixResult {
    if (comptime !@import("builtin").is_test) @compileError("shifted fit is diagnostic-test-only");
    try source.validate();
    try cfg.validate(source.rows);
    if (train.len == 0 or train.len % source.dim != 0) return error.InvalidAttentionShape;
    const q = train.len / source.dim;
    const probability_count = try mul(q, source.rows);
    const target_count = try mul(q, source.dim);
    const input_bytes = try mul(try add(try add(probability_count, target_count), q), @sizeOf(f64));
    const solver_bytes = try matrixWorkspaceBytes(source.rows, cfg.target_rows, q, source.dim);
    const attend_bytes = if (cfg.backend) |backend| try backend.workspace_fn(backend.context, source.rows, 1, q, source.dim) else 0;
    if (try add(input_bytes, @max(solver_bytes, attend_bytes)) > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const probabilities = try allocator.alloc(f64, probability_count);
    defer allocator.free(probabilities);
    const targets = try allocator.alloc(f64, target_count);
    defer allocator.free(targets);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    try attendBatch(allocator, source, train, probabilities, targets, null, cfg);
    for (0..q) |i| {
        try cfg.control.check();
        const row = probabilities[i * source.rows ..][0..source.rows];
        var maximum: f64 = 0;
        var sum: f64 = 0;
        for (row) |value| {
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            maximum = @max(maximum, value);
            sum += value;
        }
        if (!std.math.isFinite(sum) or maximum <= 0) return error.NonFiniteAttention;
        rhs[i] = sum / maximum;
        if (!std.math.isFinite(rhs[i]) or rhs[i] <= 0) return error.NonFiniteAttention;
        for (row) |*value| value.* /= maximum;
    }
    return fitMatricesMassTargetPolicyForTest(allocator, .{ .probabilities = probabilities, .targets = targets, .bias = source.bias, .rows = source.rows, .dim = source.dim, .query_rows = q }, cfg, rhs, selection_score);
}

fn fitMatricesTarget(allocator: std.mem.Allocator, input: MatrixInput, cfg: Config, mass_rhs: ?[]const f64, selection_score: DiagnosticSelectionScore) !MatrixResult {
    if (cfg.metrics) |metrics| metrics.* = .{};
    try cfg.control.check();
    if (mass_rhs == null) try input.validate() else try input.validateFeatures(false);
    try cfg.validate(input.rows);
    // Keeping every row is an exact device copy, never a fitted substitute.
    if (cfg.target_rows == input.rows) return error.ExactAttentionCopyRequired;
    const m = cfg.target_rows;
    const retained_weight_bytes = if (cfg.retain_mass_weights) try mul(m, @sizeOf(f64)) else 0;
    const workspace = try add(try matrixWorkspaceBytes(input.rows, m, input.query_rows, input.dim), retained_weight_bytes);
    if (workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const result_bytes = try add(try add(try mul(try mul(m, input.dim), @sizeOf(f64)), try mul(m, @sizeOf(f64) + @sizeOf(usize))), retained_weight_bytes);
    const scratch_buffer = try allocator.alignedAlloc(u8, .of(f64), workspace - result_bytes);
    defer allocator.free(scratch_buffer);
    var fixed = std.heap.FixedBufferAllocator.init(scratch_buffer);
    const values = try allocator.alloc(f64, try mul(m, input.dim));
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    const indices = try allocator.alloc(usize, m);
    errdefer allocator.free(indices);
    const mass_weights = if (cfg.retain_mass_weights) try allocator.alloc(f64, m) else null;
    errdefer if (mass_weights) |weights| allocator.free(weights);
    var result = MatrixResult{ .allocator = allocator, .values = values, .bias = bias, .source_indices = indices, .mass_weights = mass_weights };
    defer if (cfg.metrics) |metrics| {
        metrics.* = result.metrics;
    };
    try solveMatrices(&fixed, input, cfg, mass_rhs, selection_score, values, bias, indices, mass_weights, &result.metrics, false);
    try cfg.control.check();
    return result;
}

fn solveMatrices(fixed: *std.heap.FixedBufferAllocator, input: MatrixInput, cfg: Config, mass_rhs: ?[]const f64, selection_score: DiagnosticSelectionScore, values: []f64, bias: []f64, indices: []usize, retained_mass_weights: ?[]f64, metrics: *Metrics, mass_only: bool) !void {
    const scratch = fixed.allocator();
    const n = input.rows;
    const m = cfg.target_rows;
    const q = input.query_rows;
    const d = input.dim;
    const probabilities = input.probabilities;
    const targets = input.targets;
    const importance = try scratch.alloc(f64, n);
    @memset(importance, 0);
    for (0..q) |i| {
        const ps = probabilities[i * n ..][0..n];
        for (importance, ps) |*score, p| score.* += p * p; // RMS ranking
    }
    const order = try scratch.alloc(usize, n);
    for (order, 0..) |*idx, i| idx.* = i;
    var mandatory_count: usize = 0;
    if (cfg.selection == .residual_omp and cfg.mandatory_training_argmax) {
        mandatory_count = try mandatoryTrainingTopKSeeds(probabilities, q, n, cfg.mandatory_training_top_k, order[0..m]);
    }
    if (cfg.selection == .rms) {
        std.mem.sort(usize, order, importance, struct {
            fn less(scores: []const f64, a: usize, b: usize) bool {
                return scores[a] > scores[b] or (scores[a] == scores[b] and a < b);
            }
        }.less);
        @memcpy(indices, order[0..m]);
    } else if (mass_rhs != null and selection_score == .raw and cfg.backend != null and cfg.backend.?.raw_select_fn != null) {
        const selection_mark = fixed.end_index;
        const selected_weights = try scratch.alloc(f64, m);
        const backend = cfg.backend.?;
        metrics.selection_active_solves = try backend.raw_select_fn.?(backend.context, scratch, probabilities, importance, mass_rhs.?, q, n, order[0..mandatory_count], indices, selected_weights, cfg);
        try cfg.control.check();
        for (indices, selected_weights, 0..) |idx, weight, i| {
            if (idx >= n or std.mem.indexOfScalar(usize, indices[0..i], idx) != null or !std.math.isFinite(weight) or weight < 0) return error.InvalidSelectionResult;
            if (i < mandatory_count and idx != order[i]) return error.InvalidSelectionResult;
            importance[idx] = weight;
        }
        metrics.selection_steps = m;
        fixed.end_index = selection_mark;
    } else if (cfg.backend != null and cfg.backend.?.select_fn != null) {
        const selection_mark = fixed.end_index;
        const selected_weights = try scratch.alloc(f64, m);
        const backend = cfg.backend.?;
        metrics.selection_active_solves = try backend.select_fn.?(backend.context, scratch, probabilities, importance, q, n, order[0..mandatory_count], indices, selected_weights, cfg);
        try cfg.control.check();
        // Validate accelerator output before using it for indexing. The final
        // native NNLS solve below verifies direct KKT convergence independently.
        for (indices, selected_weights, 0..) |idx, weight, i| {
            if (idx >= n or std.mem.indexOfScalar(usize, indices[0..i], idx) != null or !std.math.isFinite(weight) or weight < 0) return error.InvalidSelectionResult;
            if (i < mandatory_count and idx != order[i]) return error.InvalidSelectionResult;
            importance[idx] = weight;
        }
        metrics.selection_steps = m;
        fixed.end_index = selection_mark;
    } else {
        const selection_mark = fixed.end_index;
        {
            const columns = try scratch.alloc(f64, q * n);
            for (0..q) |i| for (0..n) |j| {
                columns[j * q + i] = probabilities[i * n + j];
            };
            const basis = try scratch.alloc(f64, q * m);
            const weights = try scratch.alloc(f64, m);
            const residual = try scratch.alloc(f64, q);
            const norms = try scratch.alloc(f64, m);
            const orth = try scratch.alloc(f64, q * m);
            const triangular = try scratch.alloc(f64, m * m);
            const projected = try scratch.alloc(f64, m);
            const trial = try scratch.alloc(f64, m);
            const column_norms = try scratch.alloc(f64, m);
            @memset(triangular, 0);
            var qr_valid = true;
            @memset(weights, 0);
            if (mass_rhs) |rhs| @memcpy(residual, rhs) else @memset(residual, 1);
            for (0..m) |selected| {
                try cfg.control.check();
                var best: ?usize = if (selected < mandatory_count) order[selected] else null;
                var best_score: f64 = -std.math.inf(f64);
                if (best == null) for (0..n) |column| {
                    if (std.mem.indexOfScalar(usize, indices[0..selected], column) != null) continue;
                    var dot: f64 = 0;
                    for (columns[column * q ..][0..q], residual) |x, r| dot += x * r;
                    const norm = importance[column];
                    const score = if (selection_score == .raw) dot else if (norm > 0) dot / @sqrt(norm) else 0;
                    if (best == null or score > best_score) {
                        best = column;
                        best_score = score;
                    }
                };
                indices[selected] = best orelse return error.InvalidFitConfig;
                const count = selected + 1;
                for (0..q) |i| for (indices[0..count], 0..) |idx, j| {
                    basis[i * count + j] = probabilities[i * n + idx];
                };
                weights[selected] = 0;
                column_norms[selected] = @sqrt(importance[indices[selected]]);
                if (qr_valid and column_norms[selected] > 0) {
                    const column = orth[selected * q ..][0..q];
                    for (column, 0..) |*v, i| v.* = probabilities[i * n + indices[selected]] / column_norms[selected];
                    // Twice-reorthogonalized MGS updates the QR without forming
                    // normal equations. Every accepted solution still passes
                    // the direct NNLS residual/KKT check below.
                    for (0..2) |_| for (0..selected) |j| {
                        const prior = orth[j * q ..][0..q];
                        var dot: f64 = 0;
                        for (column, prior) |v, u| dot += v * u;
                        triangular[j * m + selected] += dot;
                        for (column, prior) |*v, u| v.* -= dot * u;
                    };
                    var norm: f64 = 0;
                    for (column) |v| norm = std.math.hypot(norm, v);
                    qr_valid = std.math.isFinite(norm) and norm > 1e-12 and count <= q;
                    if (qr_valid) {
                        triangular[selected * m + selected] = norm;
                        projected[selected] = 0;
                        for (column, 0..) |*v, projected_index| {
                            v.* /= norm;
                            projected[selected] += v.* * (if (mass_rhs) |rhs| rhs[projected_index] else 1);
                        }
                    }
                } else qr_valid = false;
                metrics.selection_steps += 1;
                // Forced prefix columns do not need a partial NNLS solution to
                // choose the next forced column. Keep building their QR state,
                // then solve once the complete mandatory prefix is present.
                if (count < mandatory_count) continue;
                var feasible = qr_valid;
                if (qr_valid) {
                    var j = count;
                    while (j > 0) {
                        j -= 1;
                        var z = projected[j];
                        for (j + 1..count) |k| z -= triangular[j * m + k] * trial[k];
                        trial[j] = z / triangular[j * m + j];
                    }
                    for (trial[0..count], column_norms[0..count]) |*w, norm| {
                        w.* /= norm;
                        if (!std.math.isFinite(w.*) or w.* < 0) feasible = false;
                    }
                }
                if (feasible) {
                    if (mass_rhs) |rhs| @memcpy(residual, rhs) else @memset(residual, 1);
                    for (0..q) |i| for (0..count) |j| {
                        residual[i] -= basis[i * count + j] * trial[j];
                    };
                    var violation: f64 = 0;
                    @memset(norms[0..count], 0);
                    for (0..q) |i| for (0..count) |j| {
                        norms[j] += basis[i * count + j] * residual[i];
                    };
                    for (0..count) |j| {
                        const gradient = norms[j] / column_norms[j];
                        if (!std.math.isFinite(gradient)) {
                            feasible = false;
                            break;
                        }
                        violation = @max(violation, if (trial[j] > 0) @abs(gradient) else @max(0, gradient));
                    }
                    feasible = feasible and violation <= cfg.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(q)));
                }
                if (feasible) {
                    @memcpy(weights[0..count], trial[0..count]);
                } else metrics.selection_active_solves += try fitMassActiveTarget(scratch, basis[0 .. q * count], mass_rhs, q, count, weights[0..count], residual, norms[0..count], cfg, true);
            }
            for (indices, weights) |idx, w| importance[idx] = w;
        }
        fixed.end_index = selection_mark;
    }
    std.mem.sort(usize, indices, {}, std.sort.asc(usize));
    const basis = try scratch.alloc(f64, try mul(q, m));
    for (indices, 0..) |idx, j| {
        for (0..q) |i| basis[i * m + j] = probabilities[i * n + idx];
    }
    const weights = try scratch.alloc(f64, m);
    const residual = try scratch.alloc(f64, q);
    const norms = try scratch.alloc(f64, m);
    if (cfg.selection == .residual_omp) for (indices, weights) |idx, *w| {
        w.* = importance[idx];
    };
    metrics.nnls_sweeps = if (cfg.mass_solver == .active_set)
        try fitMassActiveTarget(scratch, basis, mass_rhs, q, m, weights, residual, norms, cfg, cfg.selection == .residual_omp)
    else
        try fitMass(basis, q, m, weights, residual, norms, cfg);
    if (retained_mass_weights) |destination| @memcpy(destination, weights);
    if (mass_only) return;
    for (indices, bias, weights) |idx, *b, w| b.* = (if (input.bias) |bs| bs[idx] else @as(f64, 0)) + @log(@max(w, cfg.min_weight));
    try finite(bias);
    const fitted = try scratch.alloc(f64, try mul(q, m));
    // Reweight the original normalized probabilities. This is algebraically
    // identical to softmax over the selected keys plus their new log mass,
    // including any bias already represented by the source. No source KV or
    // query readback is needed by a device adapter.
    for (0..q) |i| {
        var sum: f64 = 0;
        for (indices, 0..) |idx, j| {
            const weight = @max(weights[j], cfg.min_weight);
            fitted[i * m + j] = probabilities[i * n + idx] * weight;
            sum += fitted[i * m + j];
        }
        if (!std.math.isFinite(sum) or sum <= 0) return error.NonFiniteAttention;
        for (fitted[i * m ..][0..m]) |*p| p.* /= sum;
    }
    try solveValues(scratch, fitted, targets, q, m, d, cfg.ridge, cfg, values);
}

/// Returns the sorted union of the lowest-index probability argmax from every
/// training row. A target that cannot retain the complete union is rejected;
/// callers must retry with a larger target rather than truncate coverage.
pub fn mandatoryTrainingArgmaxSeeds(probabilities: []const f64, query_rows: usize, source_rows: usize, out: []usize) !usize {
    return mandatoryTrainingTopKSeeds(probabilities, query_rows, source_rows, 1, out);
}

/// Returns the sorted union of each training row's top-k probability columns.
/// Per-row ordering is value descending with stable lower-index ties.
pub fn mandatoryTrainingTopKSeeds(probabilities: []const f64, query_rows: usize, source_rows: usize, top_k: usize, out: []usize) !usize {
    if (query_rows == 0 or source_rows == 0 or probabilities.len != try std.math.mul(usize, query_rows, source_rows)) return error.InvalidAttentionShape;
    if (top_k == 0 or top_k > 2 or top_k > source_rows) return error.InvalidFitConfig;
    var count: usize = 0;
    for (0..query_rows) |i| {
        const row = probabilities[i * source_rows ..][0..source_rows];
        var previous: ?usize = null;
        for (0..top_k) |_| {
            var best: ?usize = null;
            for (row, 0..) |value, column| {
                if (previous) |prior| {
                    if (value > row[prior] or (value == row[prior] and column <= prior)) continue;
                }
                if (best == null or value > row[best.?] or (value == row[best.?] and column < best.?)) best = column;
            }
            const selected = best orelse return error.InvalidAttentionProbabilities;
            previous = selected;
            if (std.mem.indexOfScalar(usize, out[0..count], selected) == null) {
                if (count == out.len) return error.CompactionFitRejected;
                out[count] = selected;
                count += 1;
            }
        }
    }
    std.mem.sort(usize, out[0..count], {}, std.sort.asc(usize));
    return count;
}

test "attention_match mandatory training top k preserves second peaks stable ties and overflow" {
    const p = [_]f64{ 0.6, 0.3, 0.1, 0, 0.6, 0, 0.1, 0.3 };
    var top1: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 1), try mandatoryTrainingArgmaxSeeds(&p, 2, 4, &top1));
    try std.testing.expectEqualSlices(usize, &.{0}, top1[0..1]);
    var top2: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 3), try mandatoryTrainingTopKSeeds(&p, 2, 4, 2, &top2));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, top2[0..3]);

    var ties: [3]usize = undefined;
    try std.testing.expectEqual(@as(usize, 2), try mandatoryTrainingTopKSeeds(&.{ 0.4, 0.4, 0.2 }, 1, 3, 2, &ties));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, ties[0..2]);
    try std.testing.expectError(error.CompactionFitRejected, mandatoryTrainingTopKSeeds(&p, 2, 4, 2, top2[0..2]));
    try std.testing.expectError(error.InvalidFitConfig, mandatoryTrainingTopKSeeds(&p, 2, 4, 0, &top2));
    try std.testing.expectError(error.InvalidFitConfig, mandatoryTrainingTopKSeeds(&p, 2, 4, 3, &top2));

    const input: MatrixInput = .{ .probabilities = &.{ 0.6, 0.3, 0.1, 0, 0.6, 0, 0.1, 0.3 }, .targets = &.{ 1, 2 }, .rows = 4, .dim = 1, .query_rows = 2 };
    try std.testing.expectError(error.InvalidFitConfig, fitMatrices(std.testing.allocator, input, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_top_k = 2 }));
    try std.testing.expectError(error.InvalidFitConfig, fitMatrices(std.testing.allocator, input, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true, .mandatory_training_top_k = 3 }));
}

/// Fit a single donor KV head. Held-out queries are mandatory, and callers must
/// collect independent queries from every consumer/GQA head. No source mutation
/// occurs, including on OOM, poor fit, or convergence failure.
fn fitImpl(allocator: std.mem.Allocator, source: Head, train: []const f64, heldout: ?[]const f64, cfg: Config) !Result {
    if (cfg.metrics) |metrics| metrics.* = .{};
    try cfg.control.check();
    try source.validate();
    try cfg.validate(source.rows);
    if (train.len == 0 or train.len % source.dim != 0 or (heldout != null and (heldout.?.len == 0 or heldout.?.len % source.dim != 0)))
        return error.InvalidAttentionShape;
    try finite(train);
    if (heldout) |queries| try finite(queries);
    const n = source.rows;
    const m = cfg.target_rows;
    const d = source.dim;
    const q = train.len / d;
    const result_bytes = try add(try mul(try mul(m, d), 2 * @sizeOf(f64)), try mul(m, @sizeOf(f64) + @sizeOf(usize)));
    // The diagnostic no-intermediate-validation path retains the same TRAIN-sized
    // workspace authority as the prior `heldout=train` prototype.
    const validation_rows = if (heldout) |queries| queries.len / d else q;
    const workspace = if (m == n) result_bytes else try fitWorkspaceBytes(n, m, q, d, validation_rows, cfg);
    if (workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const scratch_buffer = try allocator.alignedAlloc(u8, .of(f64), workspace - result_bytes);
    defer allocator.free(scratch_buffer);
    var fixed = std.heap.FixedBufferAllocator.init(scratch_buffer);
    const scratch = fixed.allocator();
    const keys = try allocator.alloc(f64, try mul(m, d));
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, try mul(m, d));
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, m);
    errdefer allocator.free(bias);
    const indices = try allocator.alloc(usize, m);
    errdefer allocator.free(indices);
    var result: Result = .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias, .source_indices = indices };
    defer {
        if (cfg.metrics) |metrics| metrics.* = result.metrics;
    }
    if (m == n) {
        @memcpy(keys, source.keys);
        @memcpy(values, source.values);
        for (bias, indices, 0..) |*b, *idx, i| {
            b.* = if (source.bias) |bs| bs[i] else 0;
            idx.* = i;
        }
        return result;
    }
    const probabilities = try scratch.alloc(f64, try mul(q, n));
    const targets = try scratch.alloc(f64, try mul(q, d));
    try attendBatch(scratch, source, train, probabilities, targets, null, cfg);
    try solveMatrices(&fixed, .{ .probabilities = probabilities, .targets = targets, .bias = source.bias, .rows = n, .dim = d, .query_rows = q }, cfg, null, .normalized, values, bias, indices, null, &result.metrics, false);
    for (indices, 0..) |idx, j| @memcpy(keys[j * d ..][0..d], source.keys[idx * d ..][0..d]);
    const compact = result.head(source);
    if (heldout) |queries| {
        const validation = try validationMetrics(scratch, source, compact, queries, cfg);
        result.metrics.max_relative_mass_error = validation.max_relative_mass_error;
        result.metrics.relative_output_rmse = validation.relative_output_rmse;
        result.metrics.quality_evaluated = true;
        if (!std.math.isFinite(result.metrics.max_relative_mass_error) or
            !std.math.isFinite(result.metrics.relative_output_rmse) or
            result.metrics.max_relative_mass_error > cfg.max_relative_mass_error or
            result.metrics.relative_output_rmse > cfg.max_relative_output_rmse) return error.CompactionFitRejected;
    }
    return result;
}

pub fn fit(allocator: std.mem.Allocator, source: Head, train: []const f64, heldout: []const f64, cfg: Config) !Result {
    return fitImpl(allocator, source, train, heldout, cfg);
}

/// Test-only constructor seam. It performs the identical TRAIN attention,
/// selection, NNLS/KKT, value fit, and source-index copy, but grants no quality
/// authorization and never accepts a heldout query slice.
pub fn fitUnvalidatedDiagnostic(allocator: std.mem.Allocator, source: Head, train: []const f64, cfg: Config) !Result {
    if (comptime !@import("builtin").is_test) @compileError("unvalidated attention match constructor is diagnostic-test-only");
    return fitImpl(allocator, source, train, null, cfg);
}

fn unvalidatedAllocationFixture(allocator: std.mem.Allocator) !void {
    const source: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 0, 1 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var result = try fitUnvalidatedDiagnostic(allocator, source, &.{ -1, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    result.deinit();
}

test "unvalidated diagnostic constructor retains checks ownership and no quality grant" {
    const source: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 0, 1 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.InvalidAttentionShape, fitUnvalidatedDiagnostic(std.testing.allocator, source, &.{}, .{ .target_rows = 1 }));
    try std.testing.expectError(error.NonFiniteAttention, fitUnvalidatedDiagnostic(std.testing.allocator, source, &.{std.math.nan(f64)}, .{ .target_rows = 1 }));
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, fitUnvalidatedDiagnostic(std.testing.allocator, source, &.{ -1, 1 }, .{ .target_rows = 1, .control = .{ .check_fn = Cancel.check } }));
    var result = try fitUnvalidatedDiagnostic(std.testing.allocator, source, &.{ -1, 1 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expect(!result.metrics.quality_evaluated);
    result.bias[0] += 10;
    const final = try validationMetrics(std.testing.allocator, source, result.head(source), &.{0}, .{ .target_rows = 1 });
    try std.testing.expect(final.max_relative_mass_error > 0.05 or final.relative_output_rmse > 0.05);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, unvalidatedAllocationFixture, .{});
}

test "attention_match matrix adapter matches native source fit without source KV" {
    const h: Head = .{ .keys = &.{ -1, 0, 0.5, 1 }, .values = &.{ 3, 2, 1, 4 }, .bias = &.{ 0.1, -0.2, 0.3, 0 }, .rows = 4, .dim = 1, .scale = 0.5, .softcap = 0.7 };
    const queries = [_]f64{ -2, -1, 0, 1, 2, 3 };
    var p: [queries.len * 4]f64 = undefined;
    var y: [queries.len]f64 = undefined;
    const cfg: Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 10, .max_relative_output_rmse = 10 };
    try attendBatch(std.testing.allocator, h, &queries, &p, &y, null, cfg);
    var matrix = try fitMatrices(std.testing.allocator, .{ .probabilities = &p, .targets = &y, .bias = h.bias, .rows = h.rows, .dim = h.dim, .query_rows = queries.len }, cfg);
    defer matrix.deinit();
    var native = try fit(std.testing.allocator, h, &queries, &.{ -0.5, 0.5 }, cfg);
    defer native.deinit();
    try std.testing.expectEqualSlices(usize, native.source_indices, matrix.source_indices);
    try std.testing.expectEqualSlices(f64, native.bias, matrix.bias);
    try std.testing.expectEqualSlices(f64, native.values, matrix.values);
    try std.testing.expect(!matrix.metrics.quality_evaluated);
    try std.testing.expect(native.metrics.quality_evaluated);
}

test "attention_match matrix input rejects invalid probabilities and enforces scratch admission" {
    const input: MatrixInput = .{ .probabilities = &.{ 0.5, 0.5, 0.25, 0.75 }, .targets = &.{ 1, 2 }, .rows = 2, .dim = 1, .query_rows = 2 };
    var cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    cfg.max_workspace_bytes = (try matrixWorkspaceBytes(2, 1, 2, 1)) - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitMatrices(std.testing.failing_allocator, input, cfg));
    var invalid = input;
    invalid.probabilities = &.{ 0.5, 0.4, 0.25, 0.75 };
    try std.testing.expectError(error.InvalidAttentionProbabilities, fitMatrices(std.testing.failing_allocator, invalid, cfg));
}

test "attention_match matrix fitting frees every failed allocation" {
    const Case = struct {
        fn run(a: std.mem.Allocator) !void {
            var r = try fitMatrices(a, .{ .probabilities = &.{ 0.5, 0.5, 0.5, 0.5 }, .targets = &.{ 3, 3 }, .rows = 2, .dim = 1, .query_rows = 2 }, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set });
            defer r.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, Case.run, .{});
}

test "attention_match identity preserves fitted bias and values exactly" {
    const h: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 3, 4 }, .bias = &.{ 2, -1 }, .rows = 2, .dim = 1, .scale = 1 };
    var r = try fit(std.testing.allocator, h, &.{0}, &.{1}, .{ .target_rows = 2 });
    defer r.deinit();
    try std.testing.expectEqualSlices(f64, h.keys, r.keys);
    try std.testing.expectEqualSlices(f64, h.values, r.values);
    try std.testing.expectEqualSlices(f64, h.bias.?, r.bias);
}

test "attention_match target RHS active set preserves nonunit objective" {
    const matrix = [_]f64{ 1, 0, 0, 1, 1, 1 };
    const target = [_]f64{ 0.25, 0.5, 0.75 };
    var weights = [_]f64{ 0, 0 };
    var residual: [3]f64 = undefined;
    var norms: [2]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &matrix, &target, 3, 2, &weights, &residual, &norms, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set }, false);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), weights[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), weights[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), residual[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), residual[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0), residual[2], 1e-10);
}

test "attention_match mandatory training argmax covers peaks ordinary mass OMP omits" {
    const input: MatrixInput = .{ .probabilities = &.{ 0.2, 0.8, 0, 0.2, 0, 0.8 }, .targets = &.{ 1, 2 }, .rows = 3, .dim = 1, .query_rows = 2 };
    const base: Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set };
    var ordinary = try fitMatrices(std.testing.allocator, input, base);
    defer ordinary.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, ordinary.source_indices);

    var covered_cfg = base;
    covered_cfg.mandatory_training_argmax = true;
    var covered = try fitMatrices(std.testing.allocator, input, covered_cfg);
    defer covered.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 2 }, covered.source_indices);
    try std.testing.expect(covered.metrics.nnls_sweeps > 0);

    covered_cfg.target_rows = 1;
    try std.testing.expectError(error.CompactionFitRejected, fitMatrices(std.testing.allocator, input, covered_cfg));
    covered_cfg.selection = .rms;
    try std.testing.expectError(error.InvalidFitConfig, fitMatrices(std.testing.allocator, input, covered_cfg));
}

test "attention_match duplicate keys preserve mass alongside a future exact token" {
    const h: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .rows = 4, .dim = 1, .scale = 1 };
    var r = try fit(std.testing.allocator, h, &.{ 0, 1 }, &.{ -2, 3 }, .{ .target_rows = 1 });
    defer r.deinit();
    try std.testing.expectApproxEqAbs(@log(@as(f64, 4)), r.bias[0], 1e-10);
    const full: Head = .{ .keys = &.{ 0, 0, 0, 0, 1 }, .values = &.{ 2, 4, 6, 8, 10 }, .rows = 5, .dim = 1, .scale = 1 };
    const compact: Head = .{ .keys = &.{ 0, 1 }, .values = &.{ r.values[0], 10 }, .bias = &.{ r.bias[0], 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var full_p: [5]f64 = undefined;
    var compact_p: [2]f64 = undefined;
    var full_y: [1]f64 = undefined;
    var compact_y: [1]f64 = undefined;
    const a = try attend(full, &.{0.7}, &full_p, &full_y);
    const b = try attend(compact, &.{0.7}, &compact_p, &compact_y);
    try std.testing.expectApproxEqAbs(a, b, 1e-10);
    try std.testing.expectApproxEqAbs(full_y[0], compact_y[0], 1e-7);
}

test "attention_match repeated compaction incorporates existing extreme biases" {
    const h: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 2, 2, 2 }, .bias = &.{ 1000, 1000, 1000, 1000 }, .rows = 4, .dim = 1, .scale = 1 };
    var first = try fit(std.testing.allocator, h, &.{ 0, 1 }, &.{2}, .{ .target_rows = 2 });
    defer first.deinit();
    var second = try fit(std.testing.allocator, first.head(h), &.{ 0, 1 }, &.{2}, .{ .target_rows = 1 });
    defer second.deinit();
    try std.testing.expectApproxEqAbs(1000 + @log(@as(f64, 4)), second.bias[0], 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 2), second.values[0], 1e-7);
}

test "attention_match rejects unseen-query mass loss and invalid inputs" {
    const h: Head = .{ .keys = &.{ -1, 1 }, .values = &.{ 1, 1 }, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.CompactionFitRejected, fit(std.testing.allocator, h, &.{0}, &.{10}, .{ .target_rows = 1 }));
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, h, &.{0}, &.{1}, .{ .target_rows = 1, .max_workspace_bytes = 1 }));
    try std.testing.expectError(error.NonFiniteAttention, fit(std.testing.allocator, h, &.{std.math.nan(f64)}, &.{1}, .{ .target_rows = 1 }));
    try std.testing.expectError(error.InvalidAttentionShape, workspaceBytes(std.math.maxInt(usize), 2, 2, 2));
}

test "attention_match rejected fit retains diagnostics and resets them on early failure" {
    const h: Head = .{ .keys = &.{ -1, 1 }, .values = &.{ 1, 1 }, .rows = 2, .dim = 1, .scale = 1 };
    var metrics: Metrics = .{};
    try std.testing.expectError(error.CompactionFitRejected, fit(std.testing.allocator, h, &.{0}, &.{10}, .{ .target_rows = 1, .metrics = &metrics }));
    try std.testing.expect(metrics.max_relative_mass_error > 0.99);
    try std.testing.expect(metrics.quality_evaluated);
    try std.testing.expect(metrics.nnls_sweeps > 0);
    try std.testing.expect(metrics.relative_output_rmse < 0.05);
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, h, &.{0}, &.{10}, .{ .target_rows = 1, .max_workspace_bytes = 1, .metrics = &metrics }));
    try std.testing.expectEqualDeep(Metrics{}, metrics);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    const h: Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 1, 2, 3 }, .rows = 3, .dim = 1, .scale = 1 };
    var r = try fit(allocator, h, &.{ 0, 1 }, &.{2}, .{ .target_rows = 1 });
    defer r.deinit();
}
test "attention_match allocation failures leave source intact and leak no memory" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "attention_match QR recovers multiple value columns without normal equations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var result: [4]f64 = undefined;
    // A * [[2, -1], [4, 3]] = Y. A has full column rank.
    try fitValues(arena.allocator(), &.{ 1, 0, 0, 1, 0.5, 0.5 }, &.{ 2, -1, 4, 3, 3, 1 }, 3, 2, 2, 1e-12, .{}, &result);
    for (result, [_]f64{ 2, -1, 4, 3 }) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-10);
    // Duplicate columns are rank deficient without ridge but must stay finite.
    try fitValues(arena.allocator(), &.{ 0.5, 0.5, 0.5, 0.5 }, &.{ 3, 1, 3, 1 }, 2, 2, 2, 1e-8, .{}, &result);
    try std.testing.expectApproxEqAbs(@as(f64, 3), (result[0] + result[2]) / 2, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 1), (result[1] + result[3]) / 2, 1e-7);
}

test "attention_match cancellation interrupts fitting and unwinds allocations" {
    const Check = struct {
        calls: usize = 0,
        fn check(ctx: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            if (self.calls >= 4) return error.Cancelled;
        }
    };
    var check: Check = .{};
    const h: Head = .{ .keys = &.{ 0, 0, 0 }, .values = &.{ 1, 2, 3 }, .rows = 3, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.Cancelled, fit(std.testing.allocator, h, &.{ 0, 1 }, &.{2}, .{
        .target_rows = 1,
        .control = .{ .context = &check, .check_fn = Check.check },
    }));
    try std.testing.expectEqual(@as(usize, 4), check.calls);
}

test "attention_match biases are added after softcap" {
    const h: Head = .{ .keys = &.{10}, .values = &.{3}, .bias = &.{1000}, .rows = 1, .dim = 1, .scale = 1, .softcap = 2 };
    var p: [1]f64 = undefined;
    const mass = try attend(h, &.{1}, &p, null);
    try std.testing.expectApproxEqAbs(1000 + 2 * std.math.tanh(@as(f64, 5)), mass, 1e-10);
}

test "attention_match NNLS satisfies nonnegative optimum and rejects unconverged fit" {
    var weights: [2]f64 = undefined;
    var residual: [2]f64 = undefined;
    var norms: [2]f64 = undefined;
    const matrix = [_]f64{ 1, 2, 2, 3 };
    // Unconstrained LS gives [-1, 1]; NNLS gives [0, 5/13]. Clipping
    // unconstrained LS to zero would NOT solve this constrained problem.
    _ = try fitMass(&matrix, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2 });
    try std.testing.expectApproxEqAbs(@as(f64, 0), weights[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0 / 13.0), weights[1], 1e-7);
    try std.testing.expectError(error.MassFitDidNotConverge, fitMass(&matrix, 2, 2, &weights, &residual, &norms, .{
        .target_rows = 2,
        .max_sweeps = 1,
    }));
}

test "attention_match active NNLS enforces boundary optimum and ignores duplicate zero columns" {
    var w: [3]f64 = undefined;
    var r: [2]f64 = undefined;
    var norms: [3]f64 = undefined;
    const a = [_]f64{ 0.1, 0.2, 0, 0.5, 0.9, 0 };
    _ = try fitMassActive(std.testing.allocator, &a, 2, 3, &w, &r, &norms, .{ .target_rows = 1 }, false);
    try std.testing.expectEqual(@as(f64, 0), w[0]);
    try std.testing.expectApproxEqAbs(1.1 / 0.85, w[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0), w[2]);
    try std.testing.expect(a[0] * r[0] + a[3] * r[1] <= 0);
    const duplicate = [_]f64{ 0.5, 0.5, 0, 0.5, 0.5, 0 };
    _ = try fitMassActive(std.testing.allocator, &duplicate, 2, 3, &w, &r, &norms, .{ .target_rows = 1 }, false);
    try std.testing.expectApproxEqAbs(2, w[0] + w[1], 1e-12);
    try std.testing.expectEqual(@as(f64, 0), w[2]);
}

test "attention_match active NNLS bounds work and rejects nonfinite warm state" {
    var w: [2]f64 = undefined;
    var r: [2]f64 = undefined;
    var norms: [2]f64 = undefined;
    const a = [_]f64{ 1, 0, 0, 1 };
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActive(std.testing.allocator, &a, 2, 2, &w, &r, &norms, .{ .target_rows = 2, .max_sweeps = 1 }, false));
    _ = try fitMassActive(std.testing.allocator, &a, 2, 2, &w, &r, &norms, .{ .target_rows = 2 }, true);
    try std.testing.expectApproxEqAbs(1, w[0], 1e-12);
    try std.testing.expectApproxEqAbs(1, w[1], 1e-12);
    w[0] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteAttention, fitMassActive(std.testing.allocator, &a, 2, 2, &w, &r, &norms, .{ .target_rows = 2 }, true));
}

test "attention_match diagnostic exact active fixed point rejects without repeated scalar solves" {
    const Fake = struct {
        calls: usize = 0,

        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }

        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, m: usize, d: usize, _: f64, control: Control, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            try std.testing.expectEqual(@as(usize, 1), m);
            try std.testing.expectEqual(@as(usize, 1), d);
            self.calls += 1;
            out[0] = 1;
        }
    };
    const a = [_]f64{1};
    const target = [_]f64{2};
    var weights = [_]f64{1};
    var residual: [1]f64 = undefined;
    var norms: [1]f64 = undefined;
    var fake: Fake = .{};
    const backend: Backend = .{ .context = &fake, .workspace_fn = Fake.workspace, .values_fn = Fake.values };

    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &a, &target, 1, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .max_sweeps = 4,
        .backend = backend,
    }, true));
    try std.testing.expectEqual(@as(usize, 4), fake.calls);

    fake.calls = 0;
    weights[0] = 1;
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &a, &target, 1, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .max_sweeps = 4,
        .reject_exact_active_fixed_point = true,
        .backend = backend,
    }, true));
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
}

test "attention_match diagnostic exact active fixed point preserves convergence and does not fire after a drop" {
    const Fake = struct {
        calls: usize = 0,
        drop_first: bool = false,

        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }

        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, m: usize, _: usize, _: f64, control: Control, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            self.calls += 1;
            if (self.drop_first and self.calls == 1) {
                try std.testing.expectEqual(@as(usize, 2), m);
                out[0] = 1;
                out[1] = -1;
            } else {
                @memset(out[0..m], 1);
            }
        }
    };
    var fake: Fake = .{};
    const backend: Backend = .{ .context = &fake, .workspace_fn = Fake.workspace, .values_fn = Fake.values };
    var weights = [_]f64{0};
    var residual: [1]f64 = undefined;
    var norms: [1]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &.{1}, &.{1}, 1, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .reject_exact_active_fixed_point = true,
        .backend = backend,
    }, false);
    try std.testing.expectEqual(@as(f64, 1), weights[0]);
    try std.testing.expectEqual(@as(usize, 1), fake.calls);

    fake = .{ .drop_first = true };
    var drop_weights = [_]f64{ 1, 1 };
    var drop_residual: [2]f64 = undefined;
    var drop_norms: [2]f64 = undefined;
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 2 }, 2, 2, &drop_weights, &drop_residual, &drop_norms, .{
        .target_rows = 2,
        .mass_solver = .active_set,
        .max_sweeps = 2,
        .reject_exact_active_fixed_point = true,
        .backend = backend,
    }, true));
    try std.testing.expectEqual(@as(usize, 2), fake.calls);
}

test "attention_match diagnostic exact active fixed point honors cancellation" {
    const Cancel = struct {
        calls: usize = 0,
        fn check(ctx: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            if (self.calls == 2) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    var weights = [_]f64{1};
    var residual: [1]f64 = undefined;
    var norms: [1]f64 = undefined;
    try std.testing.expectError(error.Cancelled, fitMassActiveTargetForTest(std.testing.allocator, &.{1}, &.{2}, 1, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .reject_exact_active_fixed_point = true,
        .control = .{ .context = &cancel, .check_fn = Cancel.check },
    }, true));
    try std.testing.expectEqual(@as(usize, 2), cancel.calls);
}

test "attention_match diagnostic outer-entry fixed point catches an entering-variable drop cycle" {
    const Fake = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, m: usize, _: usize, _: f64, control: Control, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            self.calls += 1;
            if (m == 2) {
                out[0] = 1;
                out[1] = -1;
            } else out[0] = 1;
        }
    };
    var fake: Fake = .{};
    const backend: Backend = .{ .context = &fake, .workspace_fn = Fake.workspace, .values_fn = Fake.values };
    var weights = [_]f64{ 1, 0 };
    var residual: [2]f64 = undefined;
    var norms: [2]f64 = undefined;
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 1 }, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .max_sweeps = 6, .backend = backend }, true));
    try std.testing.expectEqual(@as(usize, 6), fake.calls);
    fake.calls = 0;
    weights = .{ 1, 0 };
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 1 }, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .max_sweeps = 6, .reject_exact_active_fixed_point = true, .backend = backend }, true));
    try std.testing.expectEqual(@as(usize, 2), fake.calls);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]f64{ 1, 0 }), std.mem.sliceAsBytes(&weights));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]f64{ 1, 1 }), std.mem.sliceAsBytes(&residual));
}

test "attention_match diagnostic outer-entry fixed point permits changed post-drop state and convergence" {
    const Fake = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, m: usize, _: usize, _: f64, control: Control, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            self.calls += 1;
            switch (self.calls) {
                1 => {
                    out[0] = 1;
                    out[1] = -1;
                },
                2 => out[0] = 1.5,
                3 => {
                    out[0] = 1.5;
                    out[1] = 1;
                },
                else => return error.UnexpectedValueSolve,
            }
            _ = m;
        }
    };
    var fake: Fake = .{};
    const backend: Backend = .{ .context = &fake, .workspace_fn = Fake.workspace, .values_fn = Fake.values };
    var weights = [_]f64{ 1, 0 };
    var residual: [2]f64 = undefined;
    var norms: [2]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 1.5, 1 }, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .backend = backend }, true);
    try std.testing.expectEqual(@as(usize, 3), fake.calls);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]f64{ 1.5, 1 }), std.mem.sliceAsBytes(&weights));
}

test "attention_match diagnostic outer-entry fixed point is inactive when queries are fewer than columns" {
    const Fake = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, control: Control, out: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            self.calls += 1;
            out[0] = 1;
        }
    };
    var fake: Fake = .{};
    const backend: Backend = .{ .context = &fake, .workspace_fn = Fake.workspace, .values_fn = Fake.values };
    var weights = [_]f64{ 1, 0 };
    var residual: [1]f64 = undefined;
    var norms: [2]f64 = undefined;
    try std.testing.expectError(error.MassFitDidNotConverge, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0 }, &.{2}, 1, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .max_sweeps = 3, .reject_exact_active_fixed_point = true, .backend = backend }, true));
    try std.testing.expectEqual(@as(usize, 3), fake.calls);
}

test "attention_match diagnostic outer-entry fixed point restores residual on cancellation and backend failure" {
    const Context = struct {
        calls: usize = 0,
        cancel_at: usize = 0,
        fail_oom: bool = false,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn check(ctx: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.calls += 1;
            if (self.cancel_at != 0 and self.calls == self.cancel_at) return error.Cancelled;
        }
        fn values(ctx: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, control: Control, _: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            try control.check();
            if (self.fail_oom) return error.OutOfMemory;
        }
    };
    var context: Context = .{ .cancel_at = 3 };
    const backend: Backend = .{ .context = &context, .workspace_fn = Context.workspace, .values_fn = Context.values };
    var weights = [_]f64{ 1, 0 };
    var residual: [2]f64 = undefined;
    var norms: [2]f64 = undefined;
    try std.testing.expectError(error.Cancelled, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 1 }, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .backend = backend, .control = .{ .context = &context, .check_fn = Context.check } }, true));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]f64{ 1, 1 }), std.mem.sliceAsBytes(&residual));
    context = .{ .fail_oom = true };
    weights = .{ 1, 0 };
    try std.testing.expectError(error.OutOfMemory, fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 2, 1 }, 2, 2, &weights, &residual, &norms, .{ .target_rows = 2, .mass_solver = .active_set, .reject_exact_active_fixed_point = true, .backend = backend }, true));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]f64{ 1, 1 }), std.mem.sliceAsBytes(&residual));
}

test "attention_match active NNLS observer is explicit and allocation free" {
    const Observer = struct {
        calls: usize = 0,
        saw_outer: bool = false,
        saw_inner: bool = false,
        fn record(raw: ?*anyopaque, event: ActiveNnlsTraceEvent) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            self.saw_outer = self.saw_outer or event.phase == .outer;
            self.saw_inner = self.saw_inner or event.phase == .inner;
            std.debug.assert(event.q == 2 and event.m == 1 and event.active_count <= 1);
        }
    };
    var observer: Observer = .{};
    var weights = [_]f64{0};
    var residual: [2]f64 = undefined;
    var norms: [1]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &.{ 1, 1 }, &.{ 2, 2 }, 2, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .active_nnls_observer = .{ .context = &observer, .record_fn = Observer.record },
    }, false);
    try std.testing.expect(observer.calls >= 2);
    try std.testing.expect(observer.saw_outer and observer.saw_inner);
    try std.testing.expectEqual(@as(f64, 2), weights[0]);
}

fn exactFixedPointAllocationFailure(allocator: std.mem.Allocator) !void {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, control: Control, out: []f64) !void {
            try control.check();
            out[0] = 1;
        }
    };
    var weights = [_]f64{1};
    var residual: [1]f64 = undefined;
    var norms: [1]f64 = undefined;
    const result = fitMassActiveTargetForTest(allocator, &.{1}, &.{2}, 1, 1, &weights, &residual, &norms, .{
        .target_rows = 1,
        .mass_solver = .active_set,
        .max_sweeps = 2,
        .reject_exact_active_fixed_point = true,
        .backend = .{ .workspace_fn = Fake.workspace, .values_fn = Fake.values },
    }, true);
    if (result) |_| return error.ExpectedFixedPointRejection else |err| switch (err) {
        error.MassFitDidNotConverge => {},
        else => return err,
    }
}

test "attention_match diagnostic exact active fixed point unwinds allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exactFixedPointAllocationFailure, .{});
}

test "attention_match diagnostic mass RHS controls OMP selection and coefficients" {
    const input: MatrixInput = .{
        .probabilities = &.{ 0.9, 0.1, 0.1, 0.9 },
        .targets = &.{ 3, 7 },
        .rows = 2,
        .dim = 1,
        .query_rows = 2,
    };
    const cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    var ordinary = try fitMatrices(std.testing.allocator, input, cfg);
    defer ordinary.deinit();
    var ones = try fitMatricesMassTargetForTest(std.testing.allocator, input, cfg, &.{ 1, 1 });
    defer ones.deinit();
    try std.testing.expectEqualSlices(usize, ordinary.source_indices, ones.source_indices);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.values), std.mem.sliceAsBytes(ones.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ordinary.bias), std.mem.sliceAsBytes(ones.bias));

    var targeted = try fitMatricesMassTargetForTest(std.testing.allocator, input, cfg, &.{ 0.2, 2 });
    defer targeted.deinit();
    try std.testing.expectEqual(@as(usize, 1), targeted.source_indices[0]);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 91.0 / 41.0)), targeted.bias[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0 / (2.0 + cfg.ridge)), targeted.values[0], 1e-12);
    try std.testing.expectError(error.InvalidFitConfig, fitMatricesMassTargetForTest(std.testing.allocator, input, cfg, &.{1}));
    try std.testing.expectError(error.InvalidFitConfig, fitMatricesMassTargetForTest(std.testing.allocator, input, cfg, &.{ -1, 1 }));
    try std.testing.expectError(error.NonFiniteAttention, fitMatricesMassTargetForTest(std.testing.allocator, input, cfg, &.{ std.math.nan(f64), 1 }));
}

test "attention_match shifted diagnostic reconstructs max shifted mass objective" {
    const source: Head = .{
        .keys = &.{ 0, @log(@as(f64, 2)), @log(@as(f64, 4)) },
        .values = &.{ 3, 7, 11 },
        .rows = 3,
        .dim = 1,
        .scale = 1,
    };
    const cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    // A = [[1,1,1],[.25,.5,1]], z = [3,1.75]. Index 1 has a
    // unique maximum normalized OMP correlation; its NNLS weight is 31/10.
    var result = try fitShiftedForTest(std.testing.allocator, source, &.{ 0, 1 }, cfg);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 1), result.source_indices[0]);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 31.0 / 10.0)), result.bias[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, (110.0 / 7.0) / (2.0 + cfg.ridge)), result.values[0], 1e-12);
    // Normalizing each row instead changes the least-squares weighting.
    var normalized = try fitMatrices(std.testing.allocator, .{
        .probabilities = &.{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0 },
        .targets = &.{ 7, 61.0 / 7.0 },
        .rows = 3,
        .dim = 1,
        .query_rows = 2,
    }, cfg);
    defer normalized.deinit();
    try std.testing.expectEqual(@as(usize, 1), normalized.source_indices[0]);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 273.0 / 85.0)), normalized.bias[0], 1e-12);

    const input_bytes = (6 + 2 + 2) * @sizeOf(f64);
    const required = input_bytes + try matrixWorkspaceBytes(3, 1, 2, 1);
    var capped = cfg;
    capped.max_workspace_bytes = required - 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitShiftedForTest(std.testing.allocator, source, &.{ 0, 1 }, capped));
}

test "attention_match diagnostic mass RHS bypasses backend selection and preserves cancellation allocation failures" {
    const Fake = struct {
        var selection_calls: usize = 0;
        var value_calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn select(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: []const usize, _: []usize, _: []f64, _: Config) !usize {
            selection_calls += 1;
            return error.TestUnexpectedResult;
        }
        fn values(_: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: Control, out: []f64) !void {
            value_calls += 1;
            try fitValuesReferenceForTest(allocator, a, y, q, m, d, ridge, control, out);
        }
    };
    Fake.selection_calls = 0;
    Fake.value_calls = 0;
    const input: MatrixInput = .{ .probabilities = &.{ 0.9, 0.1, 0.1, 0.9 }, .targets = &.{ 3, 7 }, .rows = 2, .dim = 1, .query_rows = 2 };
    const backend: Backend = .{ .workspace_fn = Fake.workspace, .values_fn = Fake.values, .select_fn = Fake.select };
    var result = try fitMatricesMassTargetForTest(std.testing.allocator, input, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend }, &.{ 0.2, 2 });
    result.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.selection_calls);
    try std.testing.expectEqual(@as(usize, 1), Fake.value_calls);

    const Stop = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, fitMatricesMassTargetForTest(std.testing.allocator, input, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .control = .{ .check_fn = Stop.check } }, &.{ 0.2, 2 }));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, fitMatricesMassTargetForTest(failing.allocator(), input, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set }, &.{ 0.2, 2 }));
}

test "attention_match diagnostic raw and normalized OMP scores select independently" {
    const input: MatrixInput = .{ .probabilities = &.{ 1, 0.01, 1, 0.9 }, .targets = &.{ 3, 7 }, .rows = 2, .dim = 1, .query_rows = 2 };
    const cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set };
    var normalized = try fitMatricesMassTargetPolicyForTest(std.testing.allocator, input, cfg, &.{ 0, 1 }, .normalized);
    defer normalized.deinit();
    var raw = try fitMatricesMassTargetPolicyForTest(std.testing.allocator, input, cfg, &.{ 0, 1 }, .raw);
    defer raw.deinit();
    try std.testing.expectEqual(@as(usize, 1), normalized.source_indices[0]);
    try std.testing.expectEqual(@as(usize, 0), raw.source_indices[0]);
}

test "attention_match fixed-index IRLS reduces training Linf and preserves cancellation caps" {
    const input: MatrixInput = .{ .probabilities = &.{ 0.1, 0.9, 0.1, 0.9, 1, 0 }, .targets = &.{ 2, 2, 2 }, .rows = 2, .dim = 1, .query_rows = 3 };
    const cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1024 * 1024 };
    var weights: [1]f64 = undefined;
    var residual: [3]f64 = undefined;
    var norms: [1]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &.{ 0.1, 0.1, 1 }, &.{ 1, 1, 1 }, 3, 1, &weights, &residual, &norms, cfg, false);
    var baseline_max: f64 = 0;
    for ([_]f64{ 0.1, 0.1, 1 }) |value| baseline_max = @max(baseline_max, @abs(value * weights[0] - 1));
    var result = try refitFixedIndicesIrlsMatricesForTest(std.testing.allocator, input, &.{0}, cfg);
    defer result.deinit();
    const final_weight = @exp(result.bias[0]);
    var final_max: f64 = 0;
    for ([_]f64{ 0.1, 0.1, 1 }) |value| final_max = @max(final_max, @abs(value * final_weight - 1));
    try std.testing.expectApproxEqAbs(@as(f64, 20.0 / 17.0), weights[0], 1e-10);
    try std.testing.expect(final_max < 0.85);
    try std.testing.expect(final_max < baseline_max);
    try std.testing.expectEqualSlices(usize, &.{0}, result.source_indices);
    var capped = cfg;
    capped.max_workspace_bytes = 1;
    try std.testing.expectError(error.CompactionWorkspaceLimit, refitFixedIndicesIrlsMatricesForTest(std.testing.allocator, input, &.{0}, capped));
    const Stop = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var stopped = cfg;
    stopped.control = .{ .check_fn = Stop.check };
    try std.testing.expectError(error.Cancelled, refitFixedIndicesIrlsMatricesForTest(std.testing.allocator, input, &.{0}, stopped));
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, refitFixedIndicesIrlsMatricesForTest(failing.allocator(), input, &.{0}, cfg));

    const sparse: MatrixInput = .{ .probabilities = &.{ 1, 0, 1, 0 }, .targets = &.{ 2, 2 }, .rows = 2, .dim = 1, .query_rows = 2 };
    var sparse_cfg = cfg;
    sparse_cfg.target_rows = 2;
    var sparse_result = try refitFixedIndicesIrlsMatricesForTest(std.testing.allocator, sparse, &.{ 0, 1 }, sparse_cfg);
    defer sparse_result.deinit();
    try std.testing.expectApproxEqAbs(@log(sparse_cfg.min_weight), sparse_result.bias[1], 1e-12);

    const small_residual: MatrixInput = .{ .probabilities = &.{ 0.92, 0.08, 0.92, 0.08, 1, 0 }, .targets = &.{ 2, 2, 2 }, .rows = 2, .dim = 1, .query_rows = 3 };
    var small_weights: [1]f64 = undefined;
    var small_residuals: [3]f64 = undefined;
    var small_norms: [1]f64 = undefined;
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &.{ 0.92, 0.92, 1 }, &.{ 1, 1, 1 }, 3, 1, &small_weights, &small_residuals, &small_norms, cfg, false);
    var small_result = try refitFixedIndicesIrlsMatricesForTest(std.testing.allocator, small_residual, &.{0}, cfg);
    defer small_result.deinit();
    const minimax = 2.0 / 1.92;
    try std.testing.expect(@abs(@exp(small_result.bias[0]) - minimax) < @abs(small_weights[0] - minimax));
}

test "attention_match normalized shifted by unit raw selection forms analytic 2x2" {
    const source: Head = .{ .keys = &.{ 0, @log(2.0), @log(4.0) }, .values = &.{ 3, 7, 11 }, .bias = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .scale = 1, .softcap = 0 };
    const cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8 };
    var normalized_unit = try fitNormalizedSelectionForTest(std.testing.allocator, source, &.{ 0, 1 }, cfg, .normalized);
    defer normalized_unit.deinit();
    var normalized_raw = try fitNormalizedSelectionForTest(std.testing.allocator, source, &.{ 0, 1 }, cfg, .raw);
    defer normalized_raw.deinit();
    var shifted_unit = try fitShiftedSelectionForTest(std.testing.allocator, source, &.{ 0, 1 }, cfg, .normalized);
    defer shifted_unit.deinit();
    var shifted_raw = try fitShiftedSelectionForTest(std.testing.allocator, source, &.{ 0, 1 }, cfg, .raw);
    defer shifted_raw.deinit();
    try std.testing.expectEqual(@as(usize, 1), normalized_unit.source_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), normalized_raw.source_indices[0]);
    try std.testing.expectEqual(@as(usize, 1), shifted_unit.source_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), shifted_raw.source_indices[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 273.0 / 85.0), @exp(normalized_unit.bias[0]), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 399.0 / 193.0), @exp(normalized_raw.bias[0]), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 31.0 / 10.0), @exp(shifted_unit.bias[0]), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0 / 8.0), @exp(shifted_raw.bias[0]), 1e-10);
    const expected_value = (110.0 / 7.0) / (2.0 + cfg.ridge);
    try std.testing.expectApproxEqAbs(expected_value, normalized_unit.values[0], 1e-10);
    try std.testing.expectApproxEqAbs(expected_value, normalized_raw.values[0], 1e-10);
    try std.testing.expectApproxEqAbs(expected_value, shifted_unit.values[0], 1e-10);
    try std.testing.expectApproxEqAbs(expected_value, shifted_raw.values[0], 1e-10);
}

fn activeAllocationFailure(allocator: std.mem.Allocator) !void {
    const h: Head = .{ .keys = &.{ -1, 0, 0.2, 1 }, .values = &.{ 2, 4, 6, 8 }, .rows = 4, .dim = 1, .scale = 1 };
    var result = try fit(allocator, h, &.{ -0.5, 0, 0.5, 1 }, &.{0.25}, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = 1, .max_relative_output_rmse = 1 });
    defer result.deinit();
}
test "attention_match residual selection active solve unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, activeAllocationFailure, .{});
}

test "attention_match residual QR handles dependent columns and more retained rows than queries" {
    const h: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 2, 2, 2 }, .rows = 4, .dim = 1, .scale = 1 };
    var r = try fit(std.testing.allocator, h, &.{ 0, 1 }, &.{ -1, 2 }, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set });
    defer r.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2 }, r.source_indices);
    try std.testing.expectEqual(@as(usize, 3), r.metrics.selection_steps);
    try std.testing.expect(r.metrics.selection_active_solves > 0);
    try std.testing.expect(r.metrics.max_relative_mass_error < 1e-8);
    try std.testing.expect(r.metrics.relative_output_rmse < 1e-8);
}

test "attention_match repeated active solves return fixed workspace to its mark" {
    var memory: [4096]u8 align(8) = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&memory);
    var w: [3]f64 = undefined;
    var r: [2]f64 = undefined;
    var norms: [3]f64 = undefined;
    const a = [_]f64{ 0.1, 0.2, 0, 0.5, 0.9, 0 };
    for (0..32) |_| {
        _ = try fitMassActive(fixed.allocator(), &a, 2, 3, &w, &r, &norms, .{ .target_rows = 1 }, false);
        try std.testing.expectEqual(@as(usize, 0), fixed.end_index);
    }
}

const TestBackend = struct {
    calls: usize = 0,
    largest_batch: usize = 0,
    extra: usize = 0,
    invalid_indices: bool = false,
    nonfinite: bool = false,
    fn workspace(raw: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        return self.extra;
    }
    fn attention(raw: ?*anyopaque, allocator: std.mem.Allocator, h: Head, queries: []const f64, p: []f64, y: ?[]f64, masses: ?[]f64, control: Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        self.largest_batch = @max(self.largest_batch, queries.len / h.dim);
        try attendBatch(allocator, h, queries, p, y, masses, .{ .target_rows = 1, .control = control });
        if (self.nonfinite) p[0] = std.math.nan(f64);
    }
    fn values(_: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: Control, out: []f64) !void {
        try fitValues(allocator, a, y, q, m, d, ridge, control, out);
    }
    fn select(raw: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, _: Config) !usize {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        for (indices, weights, 0..) |*idx, *weight, i| {
            idx.* = if (self.invalid_indices) n else if (i < mandatory.len) mandatory[i] else i;
            weight.* = 0; // Native final NNLS must repair this unfit warm state.
        }
        return 0;
    }
    fn backend(self: *@This()) Backend {
        return .{ .context = self, .workspace_fn = workspace, .attend_fn = attention, .values_fn = values, .select_fn = select };
    }
};

test "attention_match backend receives sorted mandatory training argmax seeds" {
    var backend: TestBackend = .{};
    const input: MatrixInput = .{ .probabilities = &.{ 0, 0.8, 0.2, 0, 0, 0.2, 0, 0.8 }, .targets = &.{ 1, 2 }, .rows = 4, .dim = 1, .query_rows = 2 };
    var result = try fitMatrices(std.testing.allocator, input, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true, .backend = backend.backend() });
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, result.source_indices);
}

fn backendAllocationFailure(allocator: std.mem.Allocator) !void {
    var backend: TestBackend = .{};
    const source: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .rows = 4, .dim = 1, .scale = 1 };
    const heldout: [131]f64 = @splat(1);
    var result = try fit(allocator, source, &.{0}, &heldout, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend.backend() });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), result.values[0], 1e-6);
    try std.testing.expectEqual(@as(usize, 64), backend.largest_batch);
    try std.testing.expectEqual(@as(usize, 7), backend.calls);
}

test "attention_match backend bounds heldout batches independently of training count and unwinds failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, backendAllocationFailure, .{});
}

test "attention_match backend cannot bypass shape finite or workspace gates" {
    var backend: TestBackend = .{ .invalid_indices = true };
    const source: Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 2, 4 }, .rows = 2, .dim = 1, .scale = 1 };
    var cfg: Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend.backend() };
    try std.testing.expectError(error.InvalidSelectionResult, fit(std.testing.allocator, source, &.{0}, &.{1}, cfg));
    backend.invalid_indices = false;
    backend.nonfinite = true;
    try std.testing.expectError(error.NonFiniteAttention, fit(std.testing.allocator, source, &.{0}, &.{1}, cfg));
    backend.nonfinite = false;
    backend.extra = cfg.max_workspace_bytes;
    backend.calls = 0;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, source, &.{0}, &.{1}, cfg));
    try std.testing.expectEqual(@as(usize, 0), backend.calls);
    cfg.max_workspace_bytes = 0;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, source, &.{0}, &.{1}, cfg));
}

fn backend256AllocationFailure(allocator: std.mem.Allocator) !void {
    var backend: TestBackend = .{};
    var implementation = backend.backend();
    implementation.validation_batch = .rows_256;
    const source: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .rows = 4, .dim = 1, .scale = 1 };
    const heldout: [515]f64 = @splat(1);
    var result = try fit(allocator, source, &.{0}, &heldout, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .backend = implementation });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 5), result.values[0], 1e-6);
    try std.testing.expectEqual(@as(usize, 256), backend.largest_batch);
    try std.testing.expectEqual(@as(usize, 7), backend.calls);
}
test "attention_match bounded 256-row validation handles tails and unwinds allocations" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, backend256AllocationFailure, .{});
}

const SummaryTestBackend = struct {
    fn summary(raw: ?*anyopaque, allocator: std.mem.Allocator, h: Head, queries: []const f64, y: []f64, masses: []f64, control: Control) !void {
        const state: *TestBackend = @ptrCast(@alignCast(raw.?));
        const p = try allocator.alloc(f64, queries.len / h.dim * h.rows);
        defer allocator.free(p);
        try TestBackend.attention(raw, allocator, h, queries, p, y, masses, control);
        if (state.nonfinite) masses[0] = std.math.nan(f64);
    }
};
fn summaryAllocationFailure(allocator: std.mem.Allocator) !void {
    var backend: TestBackend = .{};
    var implementation = backend.backend();
    implementation.validation_batch = .rows_256;
    implementation.summary_fn = SummaryTestBackend.summary;
    const source: Head = .{ .keys = &.{ 0, 0, 0, 0 }, .values = &.{ 2, 4, 6, 8 }, .rows = 4, .dim = 1, .scale = 1 };
    const compact: Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 5, 5 }, .bias = &.{ @log(@as(f64, 2)), @log(@as(f64, 2)) }, .rows = 2, .dim = 1, .scale = 1 };
    const queries: [515]f64 = @splat(1);
    const cfg: Config = .{ .target_rows = 2, .backend = implementation };
    const actual = try validationMetrics(allocator, source, compact, &queries, cfg);
    try std.testing.expectEqual(@as(usize, 256), backend.largest_batch);
    try std.testing.expectEqual(@as(usize, 6), backend.calls);
    var reference_cfg = cfg;
    reference_cfg.backend.?.summary_fn = null;
    const expected = try validationMetrics(allocator, source, compact, &queries, reference_cfg);
    try std.testing.expectEqual(expected.max_relative_mass_error, actual.max_relative_mass_error);
    try std.testing.expectEqual(expected.relative_output_rmse, actual.relative_output_rmse);
    try std.testing.expectEqual(@as(usize, 256 * 6 * 8), try validationBytes(4, 2, 1, queries.len, reference_cfg) - try validationBytes(4, 2, 1, queries.len, cfg));
}
test "attention_match output-only validation preserves metrics and unwinds allocations" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, summaryAllocationFailure, .{});
}

test "attention_match output-only validation rejects nonfinite accelerated summaries" {
    var backend: TestBackend = .{ .nonfinite = true };
    var implementation = backend.backend();
    implementation.summary_fn = SummaryTestBackend.summary;
    const h: Head = .{ .keys = &.{0}, .values = &.{2}, .rows = 1, .dim = 1, .scale = 1 };
    var targets: [1]f64 = undefined;
    var masses: [1]f64 = undefined;
    try std.testing.expectError(error.NonFiniteAttention, attendSummary(std.testing.allocator, h, &.{1}, &targets, &masses, .{ .target_rows = 1, .backend = implementation }));
}

test "attention_match FP64 vector dot agrees with independent quad precision across tails" {
    var queries: [1024]f64 = undefined;
    var keys: [1024]f64 = undefined;
    for ([_]usize{ 1, 2, 3, 4, 7, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024 }) |dim| {
        var quad: f128 = 0;
        for (queries[0..dim], keys[0..dim], 0..) |*q, *k, i| {
            q.* = @as(f64, @floatFromInt(@as(i64, @intCast((i * 31 + i / 7) % 127)) - 63)) / 73.123456712345;
            k.* = @as(f64, @floatFromInt(@as(i64, @intCast((i * 23 + i / 11) % 127)) - 63)) / 71.123456712345;
            quad += @as(f128, q.*) * @as(f128, k.*);
        }
        const expected: f64 = @floatCast(quad);
        try std.testing.expectApproxEqAbs(expected, attentionDot(queries[0..dim], keys[0..dim]), 2e-12);
    }
}

test "attention_match FP64 vector dot preserves attention probabilities mass and poison rejection" {
    const queries = [_]f64{ 0.1371239123, -0.25349123, 0.342349123, 0.412349123, -0.582349123, 0.612349123, 0.72349123 };
    var keys = [_]f64{ 0.81349123, -0.912349123, 0.1212349123, 0.232349123, 0.34349123, -0.4512349123, 0.512349123, 0, 0, 0, 0, 0, 0, 0 };
    var quad: f128 = 0;
    for (queries, keys[0..7]) |q, k| quad += @as(f128, q) * @as(f128, k);
    const score: f64 = @floatCast(quad);
    const expected_mass = @log(1 + @exp(score));
    const first = @exp(score - expected_mass);
    const values = [_]f64{ 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75, -0.75 };
    const head: Head = .{ .keys = &keys, .values = &values, .rows = 2, .dim = 7, .scale = 1 };
    var probabilities: [2]f64 = undefined;
    var output: [7]f64 = undefined;
    const mass = try attend(head, &queries, &probabilities, &output);
    try std.testing.expectApproxEqAbs(expected_mass, mass, 2e-12);
    try std.testing.expectApproxEqAbs(first, probabilities[0], 2e-12);
    try std.testing.expectApproxEqAbs(1 - first, probabilities[1], 2e-12);
    for (output) |value| try std.testing.expectApproxEqAbs(first * 0.125 - (1 - first) * 0.75, value, 2e-12);
    for (0..7) |index| {
        const original = keys[index];
        keys[index] = std.math.nan(f64);
        try std.testing.expectError(error.NonFiniteAttention, attend(head, &queries, &probabilities, &output));
        keys[index] = std.math.inf(f64);
        try std.testing.expectError(error.NonFiniteAttention, attend(head, &queries, &probabilities, &output));
        keys[index] = original;
    }
}

test "attention_match retained raw weights warm identical fixed basis without solve" {
    const probabilities = [_]f64{ 0.45, 0.20, 0.35, 0.35, 0.60, 0.05, 0.40, 0.40, 0.20 };
    const rhs = [_]f64{ 1, 1, 1 };
    var fitted = try fitMatricesMassTargetPolicyForTest(std.testing.allocator, .{ .probabilities = &probabilities, .targets = &.{ 0, 0, 0 }, .rows = 3, .dim = 1, .query_rows = 3 }, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .retain_mass_weights = true }, &rhs, .raw);
    defer fitted.deinit();
    try std.testing.expect(fitted.mass_weights != null);
    var basis: [6]f64 = undefined;
    for (0..3) |q| {
        for (fitted.source_indices, 0..) |index, j| basis[q * 2 + j] = probabilities[q * 3 + index];
    }
    const weights = try std.testing.allocator.dupe(f64, fitted.mass_weights.?);
    defer std.testing.allocator.free(weights);
    var residual: [3]f64 = undefined;
    var norms: [2]f64 = undefined;
    const Recorder = struct {
        calls: usize = 0,
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn values(raw: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, _: Control, _: []f64) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return error.UnexpectedValueSolve;
        }
    };
    var recorder: Recorder = .{};
    _ = try fitMassActiveTargetForTest(std.testing.allocator, &basis, &rhs, 3, 2, weights, &residual, &norms, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .backend = .{ .context = &recorder, .workspace_fn = Recorder.workspace, .values_fn = Recorder.values } }, true);
    try std.testing.expectEqual(@as(usize, 0), recorder.calls);
    var bad = [_]f64{ 0, 0 };
    const repaired = try fitMassActiveTargetForTest(std.testing.allocator, &basis, &rhs, 3, 2, &bad, &residual, &norms, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set }, true);
    try std.testing.expect(repaired > 0);
    for (bad) |weight| try std.testing.expect(std.math.isFinite(weight) and weight >= 0);
    for (0..2) |j| {
        var gradient: f64 = 0;
        for (0..3) |q| gradient += basis[q * 2 + j] * residual[q];
        try std.testing.expect(@abs(gradient) <= 1e-8 * @sqrt(@as(f64, 3)) + 1e-12);
    }
}
