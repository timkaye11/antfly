// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const solver = @import("attention_match.zig");
const reference_omp = @import("attention_match_reference_omp.zig");
const exact = @import("attention_match_exact_peak_residual.zig");
const global = @import("attention_match_global_residual_diagnostic.zig");
const global_selection = @import("attention_match_global_selection_diagnostic.zig");
const training_cache = @import("attention_match_global_training_cache.zig");
const adaptive = @import("attention_match_adaptive_chunk_budget.zig");
const whole = @import("attention_match_whole_fit_input.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const builtin = @import("builtin");
const build_options = @import("build_options");
const platform = @import("antfly_platform");
const MassObjective = enum { residual_relative, global_original, normalized_ordinary, official_shifted };
const BudgetPolicy = enum { uniform, seed_adaptive };
const TrainingCachePolicy = enum { disabled, per_head_full_original };
const SelectionNormalization = enum { normalized, raw };
const MandatoryBasis = enum { disabled, training_top1_free };
const CoefficientPolicy = enum { baseline_nnls, fixed_indices_irls_p4_p8_p16, ref_per_step_ls_floor, global_denominator_omp };
const RetentionPolicy = enum { divisor2_only, ladder_4_3_2 };
const legacy_workspace_bytes: usize = 512 * 1024 * 1024;
const augmented_workspace_bytes: usize = 1024 * 1024 * 1024;

fn coefficientPolicyAllowed(schema: []const u8, policy: CoefficientPolicy) bool {
    const v7 = std.mem.eql(u8, schema, "antfly.attention-match-whole-exact-peak-replay-plan.v7");
    const v9 = std.mem.eql(u8, schema, "antfly.attention-match-whole-exact-peak-replay-plan.v9");
    const v11 = std.mem.eql(u8, schema, "antfly.attention-match-whole-exact-peak-replay-plan.v11");
    return if (v7) policy == .fixed_indices_irls_p4_p8_p16 else if (v9) policy == .ref_per_step_ls_floor else if (v11) policy == .global_denominator_omp else policy == .baseline_nnls;
}

test "whole replay IRLS policy is isolated from legacy plan schemas" {
    for ([_][]const u8{
        "antfly.attention-match-whole-exact-peak-replay-plan.v1",
        "antfly.attention-match-whole-exact-peak-replay-plan.v2",
        "antfly.attention-match-whole-exact-peak-replay-plan.v3",
        "antfly.attention-match-whole-exact-peak-replay-plan.v4",
        "antfly.attention-match-whole-exact-peak-replay-plan.v5",
        "antfly.attention-match-whole-exact-peak-replay-plan.v6",
    }) |schema| {
        try std.testing.expect(coefficientPolicyAllowed(schema, .baseline_nnls));
        try std.testing.expect(!coefficientPolicyAllowed(schema, .fixed_indices_irls_p4_p8_p16));
    }
    const v7 = "antfly.attention-match-whole-exact-peak-replay-plan.v7";
    try std.testing.expect(!coefficientPolicyAllowed(v7, .baseline_nnls));
    try std.testing.expect(coefficientPolicyAllowed(v7, .fixed_indices_irls_p4_p8_p16));
    const v9 = "antfly.attention-match-whole-exact-peak-replay-plan.v9";
    try std.testing.expect(!coefficientPolicyAllowed(v9, .baseline_nnls));
    try std.testing.expect(coefficientPolicyAllowed(v9, .ref_per_step_ls_floor));
}

fn augmentedWorkspaceFloor() !usize {
    const q: usize = 6544 + 16 * 320;
    const n: usize = 1024;
    const m: usize = 512;
    const d: usize = 512;
    const matrix = try solver.matrixWorkspaceBytes(n, m, q, d);
    const probabilities = try std.math.mul(usize, try std.math.mul(usize, q, n), @sizeOf(f64));
    const outputs = try std.math.mul(usize, try std.math.mul(usize, q, d), @sizeOf(f64));
    const rhs = try std.math.mul(usize, q, @sizeOf(f64));
    return std.math.add(usize, matrix, try std.math.add(usize, probabilities, try std.math.add(usize, outputs, rhs)));
}

pub const HeadResult = struct { donor: usize, head: usize, accepted: bool, divisor: usize, rows_before: usize, rows_after: usize, attempts: usize, rejection_reason: []const u8, rejection_chunk_start: usize, rejection_seed_count: usize, rejection_budget: usize, mass_error: f64, output_rmse: f64, train_metrics_valid: bool, parent_train_mass_error: f64, parent_train_output_rmse: f64, combined_train_mass_error: f64, combined_train_output_rmse: f64, elapsed_ns: u64 };
const LadderHeadResult = struct { result: HeadResult, source_bytes: usize, candidate_bytes: usize };
fn headStorageBytes(head: checkpoint.Head, staged: bool) !usize {
    const exact_rows = std.math.sub(usize, head.positions.len, head.compressed.rows()) catch return error.InvalidKvCheckpoint;
    const elements = std.math.mul(usize, exact_rows, head.dim) catch return error.InvalidKvCheckpoint;
    const exact_bytes = std.math.mul(usize, elements, 4) catch return error.InvalidKvCheckpoint;
    const kv = std.math.add(usize, exact_bytes, head.compressed.bytes()) catch return error.InvalidKvCheckpoint;
    const metadata = if (staged or head.compacted_until != 0) std.math.mul(usize, head.positions.len, 20) catch return error.InvalidKvCheckpoint else 0;
    return std.math.add(usize, kv, metadata) catch return error.InvalidKvCheckpoint;
}
fn ladderHeads(a: std.mem.Allocator, source: checkpoint.State, candidate: checkpoint.State, heads: []const HeadResult) ![]LadderHeadResult {
    const out = try a.alloc(LadderHeadResult, heads.len);
    errdefer a.free(out);
    for (out, heads) |*dst, row| dst.* = .{ .result = row, .source_bytes = try headStorageBytes(source.donors[row.donor].heads[row.head], false), .candidate_bytes = try headStorageBytes(candidate.donors[row.donor].heads[row.head], true) };
    return out;
}
const LegacyHeadResult = struct { donor: usize, head: usize, accepted: bool, divisor: usize, rows_before: usize, rows_after: usize, attempts: usize, rejection_reason: []const u8, rejection_chunk_start: usize, rejection_seed_count: usize, rejection_budget: usize, mass_error: f64, output_rmse: f64, elapsed_ns: u64 };
fn legacyHeads(a: std.mem.Allocator, heads: []const HeadResult) ![]LegacyHeadResult {
    const out = try a.alloc(LegacyHeadResult, heads.len);
    for (out, heads) |*dst, src| dst.* = .{ .donor = src.donor, .head = src.head, .accepted = src.accepted, .divisor = src.divisor, .rows_before = src.rows_before, .rows_after = src.rows_after, .attempts = src.attempts, .rejection_reason = src.rejection_reason, .rejection_chunk_start = src.rejection_chunk_start, .rejection_seed_count = src.rejection_seed_count, .rejection_budget = src.rejection_budget, .mass_error = src.mass_error, .output_rmse = src.output_rmse, .elapsed_ns = src.elapsed_ns };
    return out;
}
pub const Report = struct {
    candidate: checkpoint.Owned,
    heads: []HeadResult,
    source_global_bytes: usize,
    candidate_global_bytes: usize,
    savings_ratio: f64,
    accelerated_value_solves: u64,
    reference_chunks: usize,
    reference_mass_refits: usize,
    reference_negative_coefficients: usize,
    reference_floored_coefficients: usize,
    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        self.candidate.deinit();
        allocator.free(self.heads);
        self.* = undefined;
    }
};
fn belongs(batch: engine.ReferenceBatch, source: checkpoint.State, layout: engine.Layout, donor: usize, head: usize) bool {
    if (source.layer_donors[batch.consumer_layer] != donor) return false;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / source.donors[donor].heads.len;
    return batch.query_head / group == head;
}
fn toF64(a: std.mem.Allocator, x: []const f32) ![]f64 {
    const r = try a.alloc(f64, x.len);
    for (r, x) |*d, s| d.* = s;
    return r;
}
const Queries = struct {
    a: std.mem.Allocator,
    train: []f64,
    heldout: []f64,
    parent_train_len: usize,
    fn deinit(s: *@This()) void {
        s.a.free(s.train);
        s.a.free(s.heldout);
    }
};
fn queries(a: std.mem.Allocator, state: checkpoint.State, batches: []const engine.ReferenceBatch, augmented: []const engine.ReferenceBatch, layout: engine.Layout, donor: usize, head: usize) !Queries {
    var nt: usize = 0;
    var nh: usize = 0;
    for (batches) |b| if (belongs(b, state, layout, donor, head)) {
        nt += b.train.len;
        nh += b.heldout.len;
    };
    const parent_train_len = nt;
    for (augmented) |b| if (belongs(b, state, layout, donor, head)) {
        if (b.heldout.len != 0) return error.InvalidWholeFitInput;
        nt += b.train.len;
    };
    if (nt == 0 or nh == 0) return error.MissingCompactionConsumerQueries;
    const t = try a.alloc(f64, nt);
    errdefer a.free(t);
    const h = try a.alloc(f64, nh);
    errdefer a.free(h);
    var ti: usize = 0;
    var hi: usize = 0;
    for (batches) |b| if (belongs(b, state, layout, donor, head)) {
        for (b.train) |v| {
            t[ti] = v;
            ti += 1;
        }
        for (b.heldout) |v| {
            h[hi] = v;
            hi += 1;
        }
    };
    for (augmented) |b| if (belongs(b, state, layout, donor, head)) for (b.train) |v| {
        t[ti] = v;
        ti += 1;
    };
    if (ti != nt or hi != nh) return error.InvalidWholeFitInput;
    return .{ .a = a, .train = t, .heldout = h, .parent_train_len = parent_train_len };
}
fn round16(v: []f64) !void {
    for (v) |*x| {
        if (!std.math.isFinite(x.*) or @abs(x.*) > std.math.floatMax(f16)) return error.NonFiniteAttention;
        x.* = @floatCast(@as(f16, @floatCast(x.*)));
    }
}
fn round32(v: []f64) !void {
    for (v) |*x| {
        if (!std.math.isFinite(x.*) or @abs(x.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        x.* = @floatCast(@as(f32, @floatCast(x.*)));
    }
}
fn append(k: []f64, v: []f64, b: []f64, p: []u64, c: []u64, at: *usize, h: solver.Head, hp: []const u64, hc: []const u64) !void {
    const d = h.dim;
    @memcpy(k[(at.*) * d ..][0 .. h.rows * d], h.keys);
    @memcpy(v[(at.*) * d ..][0 .. h.rows * d], h.values);
    @memcpy(b[(at.*)..][0..h.rows], h.bias.?);
    @memcpy(p[(at.*)..][0..h.rows], hp);
    @memcpy(c[(at.*)..][0..h.rows], hc);
    at.* += h.rows;
}

const Attempt = struct { head: checkpoint.Head, mass: f64, rmse: f64 };
const ReferenceCounters = struct { chunks: usize = 0, mass_refits: usize = 0, negative_coefficients: usize = 0, floored_coefficients: usize = 0 };
const Rejection = struct { reason: []const u8 = "none", chunk_start: usize = 0, seed_count: usize = 0, budget: usize = 0, mass: f64 = 0, rmse: f64 = 0, train_metrics_valid: bool = false, parent_train_mass: f64 = 0, parent_train_rmse: f64 = 0, combined_train_mass: f64 = 0, combined_train_rmse: f64 = 0 };
fn attempt(a: std.mem.Allocator, state: checkpoint.State, original: checkpoint.Head, source: solver.Head, cache: ?*const training_cache.Cache, q: Queries, top_k: usize, divisor: usize, fit: whole.FitConfig, backend: solver.Backend, scale: f64, softcap: f64, mass_objective: MassObjective, budget_policy: BudgetPolicy, selection_normalization: SelectionNormalization, mandatory_basis: MandatoryBasis, coefficient_policy: CoefficientPolicy, collect_train_metrics: bool, reference_counters: *ReferenceCounters, rejection: *Rejection) !?Attempt {
    if (original.compressed.rows() != 0) return error.UnsupportedKvCheckpoint;
    const options: engine.Options = .{ .prefix_rows = fit.prefix_rows, .recent_rows = fit.recent_rows, .chunk_rows = fit.chunk_rows, .retention_divisor = divisor, .min_retention_divisor = divisor, .validation = .whole_head, .storage_precision = .f16, .fit = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_relative_mass_error = fit.max_relative_mass_error, .max_relative_output_rmse = fit.max_relative_output_rmse, .max_sweeps = fit.max_sweeps, .nnls_tolerance = fit.nnls_tolerance, .min_weight = fit.min_weight, .max_workspace_bytes = if (collect_train_metrics) augmented_workspace_bytes else legacy_workspace_bytes, .backend = backend } };
    const plan = (try engine.headPlan(state.logical_length, original, options)) orelse {
        rejection.reason = "not_eligible";
        return null;
    };
    const dim = original.dim;
    const ok = source.keys;
    const ov = source.values;
    const ob = source.bias.?;
    const middle_rows = plan.old_rows - plan.prefix_rows;
    const chunk_count = std.math.divCeil(usize, middle_rows, fit.chunk_rows) catch unreachable;
    const chunk_source_rows = try a.alloc(usize, chunk_count);
    defer a.free(chunk_source_rows);
    const chunk_seed_counts = try a.alloc(usize, chunk_count);
    defer a.free(chunk_seed_counts);
    const chunk_targets = try a.alloc(usize, chunk_count);
    defer a.free(chunk_targets);
    const cached_seeds = try a.alloc(usize, try std.math.mul(usize, chunk_count, fit.chunk_rows));
    defer a.free(cached_seeds);
    if (budget_policy == .seed_adaptive) {
        var pre_start = plan.prefix_rows;
        for (0..chunk_count) |chunk_index| {
            const rows = @min(fit.chunk_rows, plan.old_rows - pre_start);
            const chunk: solver.Head = .{ .keys = ok[pre_start * dim ..][0 .. rows * dim], .values = ov[pre_start * dim ..][0 .. rows * dim], .bias = ob[pre_start..][0..rows], .rows = rows, .dim = dim, .scale = scale, .softcap = softcap };
            const tr = q.train.len / dim;
            const probs = try a.alloc(f64, try std.math.mul(usize, tr, rows));
            defer a.free(probs);
            try solver.attendBatch(a, chunk, q.train, probs, null, null, .{ .target_rows = 1 });
            const seeds = cached_seeds[chunk_index * fit.chunk_rows ..][0..rows];
            chunk_source_rows[chunk_index] = rows;
            chunk_seed_counts[chunk_index] = try solver.mandatoryTrainingTopKSeeds(probs, tr, rows, top_k, seeds);
            chunk_targets[chunk_index] = std.math.divCeil(usize, rows, divisor) catch unreachable;
            pre_start += rows;
        }
        _ = adaptive.adaptiveChunkTargets(chunk_source_rows, chunk_seed_counts, divisor, chunk_targets) catch |err| switch (err) {
            error.CompactionFitRejected => {
                rejection.reason = "adaptive_budget_rejected";
                return null;
            },
            else => return err,
        };
    }
    const keys = try a.alloc(f64, plan.target_rows * dim);
    errdefer a.free(keys);
    const vals = try a.alloc(f64, plan.target_rows * dim);
    errdefer a.free(vals);
    const bias = try a.alloc(f64, plan.target_rows);
    errdefer a.free(bias);
    const pos = try a.alloc(u64, plan.target_rows);
    errdefer a.free(pos);
    const cov = try a.alloc(u64, plan.target_rows);
    errdefer a.free(cov);
    var out: usize = 0;
    const prefix: solver.Head = .{ .keys = ok[0 .. plan.prefix_rows * dim], .values = ov[0 .. plan.prefix_rows * dim], .bias = ob[0..plan.prefix_rows], .rows = plan.prefix_rows, .dim = dim, .scale = scale, .softcap = softcap };
    try append(keys, vals, bias, pos, cov, &out, prefix, original.positions[0..plan.prefix_rows], original.coverage_ends[0..plan.prefix_rows]);
    var start = plan.prefix_rows;
    var chunk_index: usize = 0;
    while (start < plan.old_rows) {
        const rows = @min(fit.chunk_rows, plan.old_rows - start);
        const budget = if (budget_policy == .seed_adaptive) chunk_targets[chunk_index] else std.math.divCeil(usize, rows, divisor) catch unreachable;
        const chunk: solver.Head = .{ .keys = ok[start * dim ..][0 .. rows * dim], .values = ov[start * dim ..][0 .. rows * dim], .bias = ob[start..][0..rows], .rows = rows, .dim = dim, .scale = scale, .softcap = softcap };
        if (coefficient_policy == .global_denominator_omp) {
            if (divisor != 2 or mass_objective != .global_original or budget_policy != .uniform or cache != null or mandatory_basis != .disabled) return error.InvalidFitConfig;
            var cfg = options.fit;
            cfg.target_rows = budget;
            cfg.ridge = 1e-8;
            cfg.max_relative_mass_error = std.math.floatMax(f64);
            cfg.max_relative_output_rmse = std.math.floatMax(f64);
            var selection = try global_selection.selectStreaming(a, source, start, rows, q.train, cfg);
            defer selection.deinit();
            var refit = try global.fitOriginalContextChunk(a, source, start, rows, &.{}, selection.indices, q.train, cfg);
            defer refit.deinit();
            const selected_keys = try a.alloc(f64, budget * dim);
            defer a.free(selected_keys);
            const selected_positions = try a.alloc(u64, budget);
            defer a.free(selected_positions);
            const selected_coverage = try a.alloc(u64, budget);
            defer a.free(selected_coverage);
            for (selection.indices, 0..) |source_index, destination| {
                @memcpy(selected_keys[destination * dim ..][0..dim], chunk.keys[source_index * dim ..][0..dim]);
                selected_positions[destination] = original.positions[start + source_index];
                selected_coverage[destination] = original.coverage_ends[start + rows - 1];
            }
            const selected: solver.Head = .{ .keys = selected_keys, .values = refit.values, .bias = refit.bias, .rows = budget, .dim = dim, .scale = scale, .softcap = softcap };
            try append(keys, vals, bias, pos, cov, &out, selected, selected_positions, selected_coverage);
            start += rows;
            chunk_index += 1;
            continue;
        }
        if (mass_objective == .official_shifted or mass_objective == .normalized_ordinary) {
            var shifted_cfg = options.fit;
            shifted_cfg.target_rows = budget;
            shifted_cfg.ridge = 1e-8;
            shifted_cfg.max_relative_mass_error = std.math.floatMax(f64);
            shifted_cfg.max_relative_output_rmse = std.math.floatMax(f64);
            shifted_cfg.mandatory_training_argmax = mandatory_basis == .training_top1_free;
            shifted_cfg.mandatory_training_top_k = 1;
            if (mandatory_basis == .training_top1_free) {
                try shifted_cfg.control.check();
                const train_rows = q.train.len / dim;
                const probability_count = try std.math.mul(usize, train_rows, rows);
                const probability_bytes = try std.math.mul(usize, probability_count, @sizeOf(f64));
                const seed_bytes = try std.math.mul(usize, rows, @sizeOf(usize));
                const probe_bytes = try std.math.add(usize, probability_bytes, seed_bytes);
                if (probe_bytes > shifted_cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
                // Backend-null FP64 attendBatch has no hidden allocation beyond P/seeds.
                const probe_cfg: solver.Config = .{ .target_rows = 1, .control = shifted_cfg.control };
                const probabilities = try a.alloc(f64, probability_count);
                defer a.free(probabilities);
                try solver.attendBatch(a, chunk, q.train, probabilities, null, null, probe_cfg);
                const seeds = try a.alloc(usize, rows);
                defer a.free(seeds);
                const seed_count = try solver.mandatoryTrainingArgmaxSeeds(probabilities, train_rows, rows, seeds);
                std.debug.print("attention_match_mandatory_free_seed chunk_start={d} count={d} budget={d} overflow={} diagnostic_only=1 quality_credit=0\n", .{ start, seed_count, budget, seed_count > budget });
                if (seed_count > budget) {
                    rejection.* = .{ .reason = "mandatory_seed_budget", .chunk_start = start, .seed_count = seed_count, .budget = budget };
                    a.free(keys);
                    a.free(vals);
                    a.free(bias);
                    a.free(pos);
                    a.free(cov);
                    return null;
                }
            }
            var reference: ?reference_omp.Result = null;
            defer if (reference) |*owned| owned.deinit();
            var shifted: ?solver.MatrixResult = null;
            defer if (shifted) |*owned| owned.deinit();
            if (coefficient_policy == .ref_per_step_ls_floor) {
                if (mass_objective != .official_shifted or selection_normalization != .raw or mandatory_basis != .disabled or divisor != 2) return error.InvalidFitConfig;
                shifted_cfg.min_weight = reference_omp.reference_floor;
                reference = reference_omp.fitShiftedForTest(a, chunk, q.train, shifted_cfg) catch |err| switch (err) {
                    error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => {
                        rejection.* = .{ .reason = "reference_fit_rejected", .chunk_start = start, .budget = budget };
                        a.free(keys);
                        a.free(vals);
                        a.free(bias);
                        a.free(pos);
                        a.free(cov);
                        return null;
                    },
                    else => return err,
                };
                const metrics = reference.?.metrics;
                if (metrics.mass_ridge != 0 or metrics.value_ridge != 0) return error.InvalidFitConfig;
                reference_counters.chunks += 1;
                reference_counters.mass_refits += metrics.mass_refits;
                reference_counters.negative_coefficients += metrics.negative_coefficients;
                reference_counters.floored_coefficients += metrics.floored_coefficients;
                std.debug.print("attention_match_reference_omp_chunk start={d} rows={d} budget={d} mass_refits={d} negative_coefficients={d} floored_coefficients={d} mass_ridge=0 value_ridge=0 diagnostic_only=1 quality_credit=0\n", .{ start, rows, budget, metrics.mass_refits, metrics.negative_coefficients, metrics.floored_coefficients });
            } else {
                shifted = (if (mass_objective == .official_shifted)
                    solver.fitShiftedSelectionForTest(a, chunk, q.train, shifted_cfg, if (selection_normalization == .raw) .raw else .normalized)
                else
                    solver.fitNormalizedSelectionForTest(a, chunk, q.train, shifted_cfg, if (selection_normalization == .raw) .raw else .normalized)) catch |err| switch (err) {
                    error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => {
                        rejection.* = .{ .reason = "official_shifted_fit_rejected", .chunk_start = start, .budget = budget };
                        a.free(keys);
                        a.free(vals);
                        a.free(bias);
                        a.free(pos);
                        a.free(cov);
                        return null;
                    },
                    else => return err,
                };
            }
            var irls: ?solver.MatrixResult = null;
            defer if (irls) |*owned| owned.deinit();
            if (coefficient_policy == .fixed_indices_irls_p4_p8_p16) {
                if (mass_objective != .normalized_ordinary or selection_normalization != .raw or mandatory_basis != .disabled) return error.InvalidFitConfig;
                irls = try solver.refitNormalizedFixedIndicesIrlsForTest(a, chunk, q.train, shifted.?.source_indices, shifted_cfg);
            }
            const selected_indices = if (reference) |*owned| owned.source_indices else if (irls) |*owned| owned.source_indices else shifted.?.source_indices;
            const selected_values = if (reference) |*owned| owned.values else if (irls) |*owned| owned.values else shifted.?.values;
            const selected_bias = if (reference) |*owned| owned.bias else if (irls) |*owned| owned.bias else shifted.?.bias;
            const selected_keys = try a.alloc(f64, budget * dim);
            defer a.free(selected_keys);
            const selected_positions = try a.alloc(u64, budget);
            defer a.free(selected_positions);
            const selected_coverage = try a.alloc(u64, budget);
            defer a.free(selected_coverage);
            for (selected_indices, 0..) |source_index, destination| {
                @memcpy(selected_keys[destination * dim ..][0..dim], chunk.keys[source_index * dim ..][0..dim]);
                selected_positions[destination] = original.positions[start + source_index];
                selected_coverage[destination] = original.coverage_ends[start + rows - 1];
            }
            const selected: solver.Head = .{ .keys = selected_keys, .values = selected_values, .bias = selected_bias, .rows = budget, .dim = dim, .scale = scale, .softcap = softcap };
            try append(keys, vals, bias, pos, cov, &out, selected, selected_positions, selected_coverage);
            start += rows;
            chunk_index += 1;
            continue;
        }
        const tr = q.train.len / dim;
        const probs = try a.alloc(f64, if (budget_policy == .uniform) try std.math.mul(usize, tr, rows) else 0);
        defer a.free(probs);
        if (budget_policy == .uniform) try solver.attendBatch(a, chunk, q.train, probs, null, null, .{ .target_rows = 1 });
        const owned_seeds = try a.alloc(usize, if (budget_policy == .uniform) rows else 0);
        defer a.free(owned_seeds);
        const all = if (budget_policy == .uniform) owned_seeds else cached_seeds[chunk_index * fit.chunk_rows ..][0..rows];
        const count = if (budget_policy == .uniform) try solver.mandatoryTrainingTopKSeeds(probs, tr, rows, top_k, all) else chunk_seed_counts[chunk_index];
        if (count >= budget) {
            rejection.* = .{ .reason = "seed_budget", .chunk_start = start, .seed_count = count, .budget = budget };
            a.free(keys);
            a.free(vals);
            a.free(bias);
            a.free(pos);
            a.free(cov);
            return null;
        }
        var cfg = options.fit;
        cfg.target_rows = budget;
        cfg.max_relative_mass_error = std.math.floatMax(f64);
        cfg.max_relative_output_rmse = std.math.floatMax(f64);
        var fitted = exact.fitDiagnostic(a, chunk, original.positions[start..][0..rows], all[0..count], original.coverage_ends[start + rows - 1], q.train, q.heldout, budget, cfg) catch |err| switch (err) {
            error.CompactionFitRejected, error.MassFitDidNotConverge => {
                rejection.* = .{ .reason = "fit_rejected", .chunk_start = start, .seed_count = count, .budget = budget };
                a.free(keys);
                a.free(vals);
                a.free(bias);
                a.free(pos);
                a.free(cov);
                return null;
            },
            else => return err,
        };
        defer fitted.deinit();
        if (mass_objective == .global_original) {
            const selected = try a.alloc(usize, fitted.source_indices.len - count);
            defer a.free(selected);
            var selected_count: usize = 0;
            for (fitted.source_indices) |index| if (std.mem.indexOfScalar(usize, all[0..count], index) == null) {
                selected[selected_count] = index;
                selected_count += 1;
            };
            if (selected_count != selected.len) return error.InvalidWholeFitInput;
            var refit = (if (cache) |stored| global.fitOriginalContextChunkCached(a, stored, source, start, rows, all[0..count], selected, q.train, cfg) else global.fitOriginalContextChunk(a, source, start, rows, all[0..count], selected, q.train, cfg)) catch |err| switch (err) {
                error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit => {
                    rejection.* = .{ .reason = "global_refit_rejected", .chunk_start = start, .seed_count = count, .budget = budget };
                    a.free(keys);
                    a.free(vals);
                    a.free(bias);
                    a.free(pos);
                    a.free(cov);
                    return null;
                },
                else => return err,
            };
            defer refit.deinit();
            for (fitted.source_indices, 0..) |source_index, destination| if (std.mem.indexOfScalar(usize, all[0..count], source_index) == null) {
                const refit_index = std.mem.indexOfScalar(usize, selected, source_index) orelse return error.InvalidWholeFitInput;
                @memcpy(fitted.values[destination * dim ..][0..dim], refit.values[refit_index * dim ..][0..dim]);
                fitted.bias[destination] = refit.bias[refit_index];
            };
        }
        try append(keys, vals, bias, pos, cov, &out, fitted.head(chunk), fitted.positions, fitted.coverage_ends);
        start += rows;
        chunk_index += 1;
    }
    const recent = original.positions.len - plan.old_rows;
    const tail: solver.Head = .{ .keys = ok[plan.old_rows * dim ..], .values = ov[plan.old_rows * dim ..], .bias = ob[plan.old_rows..], .rows = recent, .dim = dim, .scale = scale, .softcap = softcap };
    try append(keys, vals, bias, pos, cov, &out, tail, original.positions[plan.old_rows..], original.coverage_ends[plan.old_rows..]);
    if (out != plan.target_rows) return error.InvalidKvCheckpoint;
    try round16(keys);
    try round16(vals);
    try round32(bias);
    const compact: solver.Head = .{ .keys = keys, .values = vals, .bias = bias, .rows = out, .dim = dim, .scale = scale, .softcap = softcap };
    if (collect_train_metrics) {
        const parent_train = try solver.validationMetrics(a, source, compact, q.train[0..q.parent_train_len], .{ .target_rows = out, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64) });
        const combined_train = try solver.validationMetrics(a, source, compact, q.train, .{ .target_rows = out, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64) });
        rejection.parent_train_mass = parent_train.max_relative_mass_error;
        rejection.parent_train_rmse = parent_train.relative_output_rmse;
        rejection.combined_train_mass = combined_train.max_relative_mass_error;
        rejection.combined_train_rmse = combined_train.relative_output_rmse;
        rejection.train_metrics_valid = true;
    }
    const metrics = try solver.validationMetrics(a, source, compact, q.heldout, .{ .target_rows = out, .max_relative_mass_error = fit.max_relative_mass_error, .max_relative_output_rmse = fit.max_relative_output_rmse });
    if (metrics.max_relative_mass_error > fit.max_relative_mass_error or metrics.relative_output_rmse > fit.max_relative_output_rmse) {
        rejection.reason = "heldout_gate";
        rejection.mass = metrics.max_relative_mass_error;
        rejection.rmse = metrics.relative_output_rmse;
        a.free(keys);
        a.free(vals);
        a.free(bias);
        a.free(pos);
        a.free(cov);
        return null;
    }
    const k32 = try a.alloc(f32, keys.len);
    errdefer a.free(k32);
    const v32 = try a.alloc(f32, vals.len);
    errdefer a.free(v32);
    const b32 = try a.alloc(f32, bias.len);
    errdefer a.free(b32);
    for (k32, keys) |*d, s| d.* = @floatCast(s);
    for (v32, vals) |*d, s| d.* = @floatCast(s);
    for (b32, bias) |*d, s| d.* = @floatCast(s);
    a.free(keys);
    a.free(vals);
    a.free(bias);
    return .{ .head = .{ .dim = dim, .compacted_until = original.coverage_ends[plan.old_rows - 1], .positions = pos, .coverage_ends = cov, .keys = k32, .values = v32, .bias = b32 }, .mass = metrics.max_relative_mass_error, .rmse = metrics.relative_output_rmse };
}

pub fn run(a: std.mem.Allocator, input: *const whole.OwnedInput, augmented: []const engine.ReferenceBatch, backend: solver.Backend, limits: checkpoint.Limits, mass_objective: MassObjective, budget_policy: BudgetPolicy, cache_policy: TrainingCachePolicy, selection_normalization: SelectionNormalization, mandatory_basis: MandatoryBasis, coefficient_policy: CoefficientPolicy, retention_policy: RetentionPolicy, diagnostic_donor: ?usize, diagnostic_head: ?usize) !Report {
    const state = input.checkpoint_owned.state;
    const layout = input.layout;
    var candidate = try checkpoint.cloneMetadata(a, state);
    errdefer candidate.deinit();
    var rows = std.ArrayList(HeadResult).empty;
    var reference_counters: ReferenceCounters = .{};
    defer rows.deinit(a);
    for (state.donors, 0..) |donor, di| for (donor.heads, 0..) |head, hi| {
        if (donor.window != 0 or (diagnostic_donor != null and (diagnostic_donor.? != di or diagnostic_head.? != hi))) {
            @constCast(candidate.state.donors[di].heads)[hi] = try checkpoint.cloneHead(a, head);
            continue;
        }
        var q = try queries(a, state, input.batches, augmented, layout, di, hi);
        defer q.deinit();
        const begun = @import("antfly_platform").time.monotonicNs();
        if (cache_policy == .per_head_full_original) std.debug.print("attention_match_whole_cache_head_begin donor={d} head={d}\n", .{ di, hi });
        const source_keys = try toF64(a, head.keys);
        defer a.free(source_keys);
        const source_values = try toF64(a, head.values);
        defer a.free(source_values);
        const source_bias = try toF64(a, head.bias);
        defer a.free(source_bias);
        const source: solver.Head = .{ .keys = source_keys, .values = source_values, .bias = source_bias, .rows = head.positions.len, .dim = head.dim, .scale = layout.scales_by_donor[di], .softcap = layout.softcaps_by_donor[di] };
        var cache_storage: training_cache.Cache = undefined;
        var cache_live = false;
        defer if (cache_live) cache_storage.deinit();
        if (cache_policy == .per_head_full_original) {
            if (mass_objective != .global_original) return error.InvalidWholeFitInput;
            const cache_cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .backend = backend };
            cache_storage = try training_cache.Cache.init(a, source, q.train, cache_cfg, training_cache.maximum_combined_bytes);
            cache_live = true;
        }
        var replacement: ?Attempt = null;
        var used: usize = 0;
        var attempts: usize = 0;
        var rejection: Rejection = .{};
        const divisors = [_]usize{ 4, 3, 2 };
        const divisor_start: usize = if (coefficient_policy == .global_denominator_omp) 2 else if (retention_policy == .ladder_4_3_2) 0 else if (mass_objective == .official_shifted or mass_objective == .normalized_ordinary) 2 else 0;
        for (divisors[divisor_start..]) |divisor| {
            attempts += 1;
            replacement = attempt(a, state, head, source, if (cache_live) &cache_storage else null, q, if (input.method == .top1) 1 else 2, divisor, input.fit, backend, layout.scales_by_donor[di], layout.softcaps_by_donor[di], mass_objective, budget_policy, selection_normalization, mandatory_basis, coefficient_policy, augmented.len != 0, &reference_counters, &rejection) catch |err| switch (err) {
                error.CompactionFitRejected, error.MassFitDidNotConverge, error.SingularValueFit, error.NonFiniteAttention => blk: {
                    std.debug.print("attention_match_whole_candidate_numerical_reject donor={d} head={d} divisor={d} error={s}\n", .{ di, hi, divisor, @errorName(err) });
                    rejection.reason = "numerical_candidate_rejected";
                    break :blk null;
                },
                else => return err,
            };
            if (replacement != null) {
                used = divisor;
                break;
            }
        }
        const accepted = replacement != null;
        const mass = if (replacement) |result| result.mass else rejection.mass;
        const rmse = if (replacement) |result| result.rmse else rejection.rmse;
        @constCast(candidate.state.donors[di].heads)[hi] = if (replacement) |result| result.head else try checkpoint.cloneHead(a, head);
        const head_elapsed = @import("antfly_platform").time.monotonicNs() -| begun;
        try rows.append(a, .{ .donor = di, .head = hi, .accepted = accepted, .divisor = used, .rows_before = head.positions.len, .rows_after = candidate.state.donors[di].heads[hi].positions.len, .attempts = attempts, .rejection_reason = if (accepted) "none" else rejection.reason, .rejection_chunk_start = rejection.chunk_start, .rejection_seed_count = rejection.seed_count, .rejection_budget = rejection.budget, .mass_error = mass, .output_rmse = rmse, .train_metrics_valid = rejection.train_metrics_valid, .parent_train_mass_error = rejection.parent_train_mass, .parent_train_output_rmse = rejection.parent_train_rmse, .combined_train_mass_error = rejection.combined_train_mass, .combined_train_output_rmse = rejection.combined_train_rmse, .elapsed_ns = head_elapsed });
        if (cache_policy == .per_head_full_original) std.debug.print("attention_match_whole_cache_head_end donor={d} head={d} elapsed_ns={d}\n", .{ di, hi, head_elapsed });
    };
    try candidate.state.validateWithControl(limits, .{});
    const sb = try engine.globalStorageBytes(state, .f16, input.source_has_metadata);
    if (sb != input.original_global_device_bytes) return error.InvalidWholeFitInput;
    const cb = try engine.globalStorageBytes(candidate.state, .f16, true);
    return .{ .candidate = candidate, .heads = try rows.toOwnedSlice(a), .source_global_bytes = sb, .candidate_global_bytes = cb, .savings_ratio = if (cb == 0) 1 else @as(f64, @floatFromInt(sb)) / @as(f64, @floatFromInt(cb)), .accelerated_value_solves = 0, .reference_chunks = reference_counters.chunks, .reference_mass_refits = reference_counters.mass_refits, .reference_negative_coefficients = reference_counters.negative_coefficients, .reference_floored_coefficients = reference_counters.floored_coefficients };
}

const ReplayPlan = struct {
    schema: []const u8,
    input_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    input_manifest_sha256: []const u8,
    method: []const u8,
    query_samples: u16,
    result_path: []const u8,
    mass_objective: MassObjective = .residual_relative,
    budget_policy: BudgetPolicy = .uniform,
    training_cache: TrainingCachePolicy = .disabled,
    diagnostic_donor: ?usize = null,
    diagnostic_head: ?usize = null,
    ridge: f64 = 1e-8,
    selection_normalization: SelectionNormalization = .normalized,
    selection_comparison: bool = false,
    augmentation_dir: ?[]const u8 = null,
    augmentation_manifest_sha256: ?[]const u8 = null,
    parent_study_tokens_sha256: ?[]const u8 = null,
    study_tokens_sha256: ?[]const u8 = null,
    fixed_source65_manifest_sha256: ?[]const u8 = null,
    max_workspace_bytes: usize = legacy_workspace_bytes,
    mandatory_basis: MandatoryBasis = .disabled,
    coefficient_policy: CoefficientPolicy = .baseline_nnls,
    retention_policy: RetentionPolicy = .divisor2_only,
};

fn allHeadBaselinePolicyShape(p: ReplayPlan) bool {
    return p.mass_objective == .normalized_ordinary and p.budget_policy == .uniform and p.training_cache == .disabled and p.selection_normalization == .raw and !p.selection_comparison and p.mandatory_basis == .disabled and p.coefficient_policy == .baseline_nnls and p.retention_policy == .divisor2_only and p.diagnostic_donor == null and p.diagnostic_head == null and p.ridge == 1e-8;
}

fn allHeadLadderPolicyShape(p: ReplayPlan) bool {
    return p.mass_objective == .normalized_ordinary and p.budget_policy == .uniform and p.training_cache == .disabled and p.selection_normalization == .raw and !p.selection_comparison and p.mandatory_basis == .disabled and p.coefficient_policy == .baseline_nnls and p.retention_policy == .ladder_4_3_2 and p.diagnostic_donor == null and p.diagnostic_head == null and p.ridge == 1e-8 and p.query_samples == 512 and std.mem.eql(u8, p.method, "top1") and p.max_workspace_bytes == augmented_workspace_bytes;
}

fn referencePolicyShape(p: ReplayPlan) bool {
    return p.mass_objective == .official_shifted and p.budget_policy == .uniform and p.training_cache == .disabled and p.selection_normalization == .raw and !p.selection_comparison and p.mandatory_basis == .disabled and p.coefficient_policy == .ref_per_step_ls_floor and p.diagnostic_donor == 5 and p.diagnostic_head == 1 and p.ridge == 0 and p.query_samples == 512 and std.mem.eql(u8, p.method, "top1") and p.max_workspace_bytes == augmented_workspace_bytes;
}

fn globalSelectionPolicyShape(p: ReplayPlan) bool {
    return p.mass_objective == .global_original and p.budget_policy == .uniform and p.training_cache == .disabled and p.selection_normalization == .raw and !p.selection_comparison and p.mandatory_basis == .disabled and p.coefficient_policy == .global_denominator_omp and p.retention_policy == .divisor2_only and p.diagnostic_donor == 5 and p.diagnostic_head == 1 and p.ridge == 1e-8 and p.query_samples == 512 and std.mem.eql(u8, p.method, "top1") and p.max_workspace_bytes == augmented_workspace_bytes;
}

test "whole replay reference OMP v9 policy is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const base: ReplayPlan = .{ .schema = "antfly.attention-match-whole-exact-peak-replay-plan.v9", .input_dir = "/input", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .method = "top1", .query_samples = 512, .result_path = "/result", .mass_objective = .official_shifted, .selection_normalization = .raw, .diagnostic_donor = 5, .diagnostic_head = 1, .ridge = 0, .max_workspace_bytes = augmented_workspace_bytes, .coefficient_policy = .ref_per_step_ls_floor };
    try std.testing.expect(referencePolicyShape(base));
    var wrong = base;
    wrong.ridge = 1e-8;
    try std.testing.expect(!referencePolicyShape(wrong));
    wrong = base;
    wrong.diagnostic_head = 0;
    try std.testing.expect(!referencePolicyShape(wrong));
}

test "whole replay global denominator selection v11 policy is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const base: ReplayPlan = .{ .schema = "antfly.attention-match-whole-exact-peak-replay-plan.v11", .input_dir = "/input", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .method = "top1", .query_samples = 512, .result_path = "/result", .mass_objective = .global_original, .selection_normalization = .raw, .diagnostic_donor = 5, .diagnostic_head = 1, .ridge = 1e-8, .max_workspace_bytes = augmented_workspace_bytes, .coefficient_policy = .global_denominator_omp };
    try std.testing.expect(globalSelectionPolicyShape(base));
    var wrong = base;
    wrong.retention_policy = .ladder_4_3_2;
    try std.testing.expect(!globalSelectionPolicyShape(wrong));
    wrong = base;
    wrong.training_cache = .per_head_full_original;
    try std.testing.expect(!globalSelectionPolicyShape(wrong));
    wrong = base;
    wrong.diagnostic_head = 0;
    try std.testing.expect(!globalSelectionPolicyShape(wrong));
}

test "whole replay all-head normalized raw baseline policy is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const base: ReplayPlan = .{
        .schema = "antfly.attention-match-whole-exact-peak-replay-plan.v8",
        .input_dir = "/input",
        .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest },
        .config_sha256 = digest,
        .source_freeze_sha256 = digest,
        .binary_sha256 = digest,
        .exporter_sha256 = digest,
        .input_manifest_sha256 = digest,
        .method = "top1",
        .query_samples = 512,
        .result_path = "/result",
        .mass_objective = .normalized_ordinary,
        .selection_normalization = .raw,
        .max_workspace_bytes = augmented_workspace_bytes,
    };
    try std.testing.expect(allHeadBaselinePolicyShape(base));
    var scoped = base;
    scoped.diagnostic_donor = 5;
    scoped.diagnostic_head = 1;
    try std.testing.expect(!allHeadBaselinePolicyShape(scoped));
    var irls = base;
    irls.coefficient_policy = .fixed_indices_irls_p4_p8_p16;
    try std.testing.expect(!allHeadBaselinePolicyShape(irls));
}

test "whole replay all-head retention ladder policy is exact" {
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    const base: ReplayPlan = .{ .schema = "antfly.attention-match-whole-exact-peak-replay-plan.v10", .input_dir = "/input", .identity = .{ .weights = digest, .tokenizer = digest, .template = digest, .layout = digest }, .config_sha256 = digest, .source_freeze_sha256 = digest, .binary_sha256 = digest, .exporter_sha256 = digest, .input_manifest_sha256 = digest, .method = "top1", .query_samples = 512, .result_path = "/result", .mass_objective = .normalized_ordinary, .selection_normalization = .raw, .max_workspace_bytes = augmented_workspace_bytes, .retention_policy = .ladder_4_3_2 };
    try std.testing.expect(allHeadLadderPolicyShape(base));
    var fixed = base;
    fixed.retention_policy = .divisor2_only;
    try std.testing.expect(!allHeadLadderPolicyShape(fixed));
    var scoped = base;
    scoped.diagnostic_donor = 5;
    scoped.diagnostic_head = 1;
    try std.testing.expect(!allHeadLadderPolicyShape(scoped));
}
fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidWholeFitInput;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidWholeFitInput;
    return out;
}

test "attention match whole cache exact peak residual Metal replay" {
    if (comptime builtin.os.tag != .macos or !build_options.enable_metal) return error.SkipZigTest;
    const plan_path = platform.env.getenv("ANTFLY_METAL_WHOLE_CACHE_EXACT_PEAK_REPLAY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidWholeFitInput;
    const a = std.testing.allocator;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, a, .limited(1024 * 1024));
    defer a.free(bytes);
    const parsed = try std.json.parseFromSlice(ReplayPlan, a, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const p = parsed.value;
    const legacy_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v1");
    const policy_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v2");
    const cache_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v3");
    const raw_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v4");
    const augmented_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v5");
    const mandatory_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v6");
    const irls_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v7");
    const allhead_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v8");
    const reference_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v9");
    const ladder_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v10");
    const global_selection_plan = std.mem.eql(u8, p.schema, "antfly.attention-match-whole-exact-peak-replay-plan.v11");
    const augmentation_plan = augmented_plan or mandatory_plan or irls_plan or allhead_plan or reference_plan or ladder_plan or global_selection_plan;
    if ((augmentation_plan and p.max_workspace_bytes != augmented_workspace_bytes) or (!augmentation_plan and p.max_workspace_bytes != legacy_workspace_bytes)) return error.InvalidWholeFitInput;
    if ((!legacy_plan and !policy_plan and !cache_plan and !raw_plan and !augmented_plan and !mandatory_plan and !irls_plan and !allhead_plan and !reference_plan and !ladder_plan and !global_selection_plan) or !std.fs.path.isAbsolute(p.input_dir) or !std.fs.path.isAbsolute(p.result_path) or (p.query_samples != 320 and p.query_samples != 512) or (!std.mem.eql(u8, p.method, "top1") and !std.mem.eql(u8, p.method, "top2")) or ((legacy_plan or policy_plan) and p.training_cache != .disabled) or ((legacy_plan or policy_plan or cache_plan) and (p.selection_normalization != .normalized or p.selection_comparison)) or (legacy_plan and (p.mass_objective != .residual_relative or p.budget_policy != .uniform)) or (p.training_cache == .per_head_full_original and p.mass_objective != .global_original) or (!reference_plan and p.mass_objective == .official_shifted and (p.budget_policy != .uniform or p.training_cache != .disabled or p.diagnostic_donor == null or p.diagnostic_head == null or p.ridge != 1e-8)) or ((!mandatory_plan and !irls_plan and !allhead_plan and !reference_plan and !ladder_plan and !global_selection_plan and p.mass_objective != .official_shifted) and (p.diagnostic_donor != null or p.diagnostic_head != null)) or ((raw_plan or augmented_plan) and (p.mass_objective != .official_shifted or p.selection_normalization != .raw or !p.selection_comparison)) or (mandatory_plan and (p.mass_objective != .normalized_ordinary or p.selection_normalization != .raw or p.mandatory_basis != .training_top1_free or p.diagnostic_donor == null or p.diagnostic_head == null or p.ridge != 1e-8)) or (!mandatory_plan and p.mandatory_basis != .disabled) or (irls_plan and (p.mass_objective != .normalized_ordinary or p.selection_normalization != .raw or p.budget_policy != .uniform or p.mandatory_basis != .disabled or p.diagnostic_donor == null or p.diagnostic_head == null or p.ridge != 1e-8)) or (allhead_plan and !allHeadBaselinePolicyShape(p)) or (reference_plan and !referencePolicyShape(p)) or (ladder_plan and !allHeadLadderPolicyShape(p)) or (global_selection_plan and !globalSelectionPolicyShape(p)) or (!ladder_plan and !global_selection_plan and p.retention_policy != .divisor2_only) or !coefficientPolicyAllowed(p.schema, p.coefficient_policy) or (augmentation_plan != (p.augmentation_dir != null and p.augmentation_manifest_sha256 != null and p.parent_study_tokens_sha256 != null and p.study_tokens_sha256 != null)) or (augmentation_plan and !std.fs.path.isAbsolute(p.augmentation_dir.?))) return error.InvalidWholeFitInput;
    if (augmentation_plan and p.fixed_source65_manifest_sha256 == null) return error.InvalidWholeFitInput;
    const manifest_path = try std.fs.path.join(a, &.{ p.input_dir, "manifest.json" });
    defer a.free(manifest_path);
    const manifest = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, a, .limited(whole.manifest_reservation));
    defer a.free(manifest);
    var manifest_digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(manifest, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &(try parseDigest(p.input_manifest_sha256)))) return error.InvalidWholeFitInput;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(p.identity.weights), .tokenizer = try parseDigest(p.identity.tokenizer), .template = try parseDigest(p.identity.template), .layout = try parseDigest(p.identity.layout) };
    const binding: whole.Binding = .{ .output_dir = p.input_dir, .identity = identity, .config_sha256 = try parseDigest(p.config_sha256), .source_freeze_sha256 = try parseDigest(p.source_freeze_sha256), .binary_sha256 = try parseDigest(p.binary_sha256), .exporter_sha256 = try parseDigest(p.exporter_sha256), .limits = checkpoint.serving_limits };
    var input = try whole.read(a, io, binding);
    defer input.deinit();
    var augmented: ?augmentation.Owned = null;
    defer if (augmented) |*owned| owned.deinit();
    if (augmentation_plan) {
        const augmentation_manifest_path = try std.fs.path.join(a, &.{ p.augmentation_dir.?, "manifest.json" });
        defer a.free(augmentation_manifest_path);
        const augmentation_manifest = try std.Io.Dir.cwd().readFileAlloc(io, augmentation_manifest_path, a, .limited(64 * 1024));
        defer a.free(augmentation_manifest);
        var augmentation_digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(augmentation_manifest, &augmentation_digest, .{});
        if (!std.mem.eql(u8, &augmentation_digest, &(try parseDigest(p.augmentation_manifest_sha256.?)))) return error.InvalidWholeFitInput;
        augmented = try augmentation.read(a, io, .{ .output_dir = p.augmentation_dir.?, .parent_manifest_sha256 = manifest_digest, .identity = identity, .source_freeze_sha256 = try parseDigest(p.source_freeze_sha256), .binary_sha256 = try parseDigest(p.binary_sha256), .exporter_sha256 = try parseDigest(p.exporter_sha256), .parent_study_tokens_sha256 = try parseDigest(p.parent_study_tokens_sha256.?), .study_tokens_sha256 = try parseDigest(p.study_tokens_sha256.?) });
    }
    if (input.query_samples != p.query_samples or (std.mem.eql(u8, p.method, "top1") and input.method != .top1) or (std.mem.eql(u8, p.method, "top2") and input.method != .top2)) return error.InvalidWholeFitInput;
    const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;
    var fitter = try Fitter.init(a, null, 1024 * 1024 * 1024);
    var live = true;
    defer if (live) fitter.deinit();
    const started = platform.time.monotonicNs();
    const augmented_batches: []const engine.ReferenceBatch = if (augmented) |*owned| owned.batches else &.{};
    var normalized_unit: ?Report = if (raw_plan or (augmented_plan and !reference_plan and !global_selection_plan)) try run(a, &input, augmented_batches, fitter.backend(), checkpoint.serving_limits, .normalized_ordinary, .uniform, .disabled, .normalized, .disabled, .baseline_nnls, .divisor2_only, p.diagnostic_donor, p.diagnostic_head) else null;
    defer if (normalized_unit) |*control| control.deinit(a);
    var normalized_raw: ?Report = if (raw_plan or (augmentation_plan and !allhead_plan and !reference_plan and !ladder_plan and !global_selection_plan)) try run(a, &input, augmented_batches, fitter.backend(), checkpoint.serving_limits, .normalized_ordinary, .uniform, .disabled, .raw, .disabled, .baseline_nnls, .divisor2_only, p.diagnostic_donor, p.diagnostic_head) else null;
    defer if (normalized_raw) |*control| control.deinit(a);
    var shifted_unit: ?Report = if (raw_plan or (augmented_plan and !reference_plan and !global_selection_plan)) try run(a, &input, augmented_batches, fitter.backend(), checkpoint.serving_limits, .official_shifted, .uniform, .disabled, .normalized, .disabled, .baseline_nnls, .divisor2_only, p.diagnostic_donor, p.diagnostic_head) else null;
    defer if (shifted_unit) |*control| control.deinit(a);
    var report = try run(a, &input, augmented_batches, fitter.backend(), checkpoint.serving_limits, p.mass_objective, p.budget_policy, p.training_cache, p.selection_normalization, p.mandatory_basis, p.coefficient_policy, p.retention_policy, p.diagnostic_donor, p.diagnostic_head);
    defer report.deinit(a);
    const elapsed = platform.time.monotonicNs() -| started;
    report.accelerated_value_solves = fitter.deinitAndTakeValueSolves();
    live = false;
    if (elapsed == 0 or report.accelerated_value_solves == 0) return error.InvalidWholeFitInput;
    var global_heads: usize = 0;
    for (input.checkpoint_owned.state.donors) |donor| {
        if (donor.window == 0) global_heads += donor.heads.len;
    }
    if (report.heads.len != (if (allhead_plan or ladder_plan) global_heads else if (global_selection_plan or p.mass_objective == .official_shifted or p.mass_objective == .normalized_ordinary) 1 else global_heads)) return error.InvalidWholeFitInput;
    if ((raw_plan or (augmented_plan and !global_selection_plan)) and (normalized_unit.?.heads.len != 1 or normalized_raw.?.heads.len != 1 or shifted_unit.?.heads.len != 1)) return error.InvalidWholeFitInput;
    const legacy_report: []LegacyHeadResult = if (!augmented_plan) try legacyHeads(a, report.heads) else &.{};
    defer if (!augmented_plan) a.free(legacy_report);
    const legacy_normalized_unit: []LegacyHeadResult = if (raw_plan) try legacyHeads(a, normalized_unit.?.heads) else &.{};
    defer if (raw_plan) a.free(legacy_normalized_unit);
    const legacy_normalized_raw: []LegacyHeadResult = if (raw_plan) try legacyHeads(a, normalized_raw.?.heads) else &.{};
    defer if (raw_plan) a.free(legacy_normalized_raw);
    const legacy_shifted_unit: []LegacyHeadResult = if (raw_plan) try legacyHeads(a, shifted_unit.?.heads) else &.{};
    defer if (raw_plan) a.free(legacy_shifted_unit);
    for (report.heads) |head| std.debug.print("attention_match_whole_exact_peak_head method={s} donor={d} head={d} accepted={} divisor={d} attempts={d} rows_before={d} rows_after={d} mass_error={d} output_rmse={d} elapsed_ns={d}\n", .{ p.method, head.donor, head.head, head.accepted, head.divisor, head.attempts, head.rows_before, head.rows_after, head.mass_error, head.output_rmse, head.elapsed_ns });
    std.debug.print("attention_match_whole_exact_peak_summary method={s} mass_objective={s} budget_policy={s} training_cache={s} query_samples={d} global_heads={d} source_has_metadata={} source_global_bytes={d} candidate_global_bytes={d} savings_ratio={d} meets_1_5={} accelerated_value_solves={d} elapsed_ns={d} diagnostic_only=1 quality_credit=0 publication_claim=0 admission_claim=0\n", .{ p.method, @tagName(p.mass_objective), @tagName(p.budget_policy), @tagName(p.training_cache), p.query_samples, global_heads, input.source_has_metadata, report.source_global_bytes, report.candidate_global_bytes, report.savings_ratio, report.savings_ratio >= 1.5, report.accelerated_value_solves, elapsed });
    var accepted_count: usize = 0;
    for (report.heads) |head| if (head.accepted) {
        accepted_count += 1;
    };
    const ladder_rows: []LadderHeadResult = if (ladder_plan) try ladderHeads(a, input.checkpoint_owned.state, report.candidate.state, report.heads) else &.{};
    defer if (ladder_plan) a.free(ladder_rows);
    if (ladder_plan) {
        var source_sum: usize = 0;
        var candidate_sum: usize = 0;
        for (ladder_rows) |row| {
            source_sum = std.math.add(usize, source_sum, row.source_bytes) catch return error.InvalidWholeFitInput;
            candidate_sum = std.math.add(usize, candidate_sum, row.candidate_bytes) catch return error.InvalidWholeFitInput;
        }
        if (source_sum != report.source_global_bytes or candidate_sum != report.candidate_global_bytes) return error.InvalidWholeFitInput;
    }
    const result = if (global_selection_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v11", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "global_original", .selection_objective = "global_denominator_raw_omp_active_nnls", .coefficient_policy = "global_denominator_omp", .budget_policy = "uniform", .retention_policy = "divisor2_only", .retention_divisor = 2, .storage_precision = "f16", .ridge = p.ridge, .mass_gate = @as(f64, 0.05), .output_gate = @as(f64, 0.05), .diagnostic_donor = p.diagnostic_donor.?, .diagnostic_head = p.diagnostic_head.?, .query_samples = p.query_samples, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .heldout_selected = false, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false, .heads = report.heads }, .{})
    else if (ladder_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v10", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "normalized_ordinary", .selection_normalization = "raw", .coefficient_policy = "baseline_nnls", .budget_policy = "uniform", .retention_policy = "ladder_4_3_2", .retention_ladder = .{ 4, 3, 2 }, .rejected_head_policy = "preserve_original", .storage_precision = "f16", .ridge = p.ridge, .mass_gate = @as(f64, 0.05), .output_gate = @as(f64, 0.05), .query_samples = p.query_samples, .global_heads = global_heads, .accepted_count = accepted_count, .preserved_count = global_heads - accepted_count, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false, .heads = ladder_rows }, .{})
    else if (reference_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v9", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "official_shifted", .selection_normalization = "raw", .coefficient_policy = "ref_per_step_ls_floor", .budget_policy = "uniform", .retention_divisor = 2, .storage_precision = "f16", .ridge = @as(f64, 0), .mass_ridge = @as(f64, 0), .value_ridge = @as(f64, 0), .coefficient_floor = reference_omp.reference_floor, .mandatory_basis = "disabled", .diagnostic_donor = p.diagnostic_donor.?, .diagnostic_head = p.diagnostic_head.?, .query_samples = p.query_samples, .reference_chunks = report.reference_chunks, .reference_mass_refits = report.reference_mass_refits, .reference_negative_coefficients = report.reference_negative_coefficients, .reference_floored_coefficients = report.reference_floored_coefficients, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false, .heads = report.heads }, .{})
    else if (allhead_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v8", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "normalized_ordinary", .selection_normalization = "raw", .coefficient_policy = "baseline_nnls", .budget_policy = "uniform", .storage_precision = "f16", .ridge = p.ridge, .query_samples = p.query_samples, .global_heads = global_heads, .accepted_count = accepted_count, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .publication_claim = false, .admission_claim = false, .heads = report.heads }, .{})
    else if (irls_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v7", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "normalized_ordinary", .selection_normalization = "raw", .coefficient_policy = "fixed_indices_irls_p4_p8_p16", .selection_fixed = true, .heldout_selected = false, .powers = .{ 4, 8, 16 }, .inner_iterations_per_power = 8, .line_search_steps = 8, .line_search_damping = 0.5, .irls_row_scale_cap = 1e6, .irls_row_scale_normalization = "max_to_one", .ridge = p.ridge, .query_samples = p.query_samples, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .cells = .{ .baseline_nnls = normalized_raw.?.heads, .final_irls = report.heads } }, .{})
    else if (mandatory_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v6", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .method = p.method, .mass_objective = "normalized_ordinary", .selection_normalization = "raw", .mandatory_basis = "training_top1_free", .coefficients = "active_set_nnls_free", .ridge = p.ridge, .query_samples = p.query_samples, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .heldout_selected = false, .cells = .{ .mandatory_off = normalized_raw.?.heads, .mandatory_on = report.heads } }, .{})
    else if (augmented_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v5", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .augmentation_manifest_sha256 = p.augmentation_manifest_sha256.?, .fixed_source65_manifest_sha256 = p.fixed_source65_manifest_sha256.?, .max_workspace_bytes = p.max_workspace_bytes, .training_augmentation = .{ .study_pass = 1, .samples_per_stratum = 320, .training_only = true, .fixed_parent_heldout = true }, .method = p.method, .mass_objective = @tagName(p.mass_objective), .budget_policy = @tagName(p.budget_policy), .training_cache = @tagName(p.training_cache), .selection_normalization = @tagName(p.selection_normalization), .selection_comparison = "objective_normalized_shifted_x_score_unit_raw", .ridge = p.ridge, .query_samples = p.query_samples, .global_heads = global_heads, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .known_regression = "augmentation is training-only and reuses the old heldout authorization set", .official_differences = .{ .nnls = "active_set_nnls_vs_reference_unclipped_ls_then_floor", .ridge = p.ridge }, .cells = .{ .normalized_unit = normalized_unit.?.heads, .normalized_raw = normalized_raw.?.heads, .shifted_unit = shifted_unit.?.heads, .shifted_raw = report.heads }, .heads = report.heads }, .{})
    else if (raw_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v4", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .method = p.method, .mass_objective = @tagName(p.mass_objective), .budget_policy = @tagName(p.budget_policy), .training_cache = @tagName(p.training_cache), .selection_normalization = @tagName(p.selection_normalization), .selection_comparison = "objective_normalized_shifted_x_score_unit_raw", .ridge = p.ridge, .query_samples = p.query_samples, .global_heads = global_heads, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .official_differences = .{ .nnls = "active_set_nnls_vs_reference_unclipped_ls_then_floor", .ridge = p.ridge }, .cells = .{ .normalized_unit = legacy_normalized_unit, .normalized_raw = legacy_normalized_raw, .shifted_unit = legacy_shifted_unit, .shifted_raw = legacy_report }, .heads = legacy_report }, .{})
    else if (cache_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v3", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .method = p.method, .mass_objective = @tagName(p.mass_objective), .budget_policy = @tagName(p.budget_policy), .training_cache = @tagName(p.training_cache), .ridge = p.ridge, .query_samples = p.query_samples, .global_heads = global_heads, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .heads = legacy_report }, .{})
    else if (policy_plan)
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v2", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .method = p.method, .mass_objective = @tagName(p.mass_objective), .budget_policy = @tagName(p.budget_policy), .query_samples = p.query_samples, .global_heads = global_heads, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .heads = legacy_report }, .{})
    else
        try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-exact-peak-replay-result.v1", .authenticated = true, .input_manifest_sha256 = std.fmt.bytesToHex(manifest_digest, .lower), .method = p.method, .query_samples = p.query_samples, .global_heads = global_heads, .source_has_metadata = input.source_has_metadata, .source_global_bytes = report.source_global_bytes, .candidate_global_bytes = report.candidate_global_bytes, .savings_ratio = report.savings_ratio, .meets_1_5 = report.savings_ratio >= 1.5, .accelerated_value_solves = report.accelerated_value_solves, .elapsed_ns = elapsed, .diagnostic_only = true, .quality_credit = false, .publication_claim = false, .admission_claim = false, .heads = legacy_report }, .{});
    defer a.free(result);
    var result_file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
    defer result_file.close(io);
    try result_file.writeStreamingAll(io, result);
    try result_file.writeStreamingAll(io, "\n");
    try result_file.sync(io);
}

test "whole replay legacy head serialization excludes augmentation metrics" {
    const a = std.testing.allocator;
    const head: HeadResult = .{ .donor = 1, .head = 2, .accepted = false, .divisor = 2, .rows_before = 8, .rows_after = 8, .attempts = 1, .rejection_reason = "heldout_gate", .rejection_chunk_start = 0, .rejection_seed_count = 0, .rejection_budget = 0, .mass_error = 1, .output_rmse = 1, .train_metrics_valid = true, .parent_train_mass_error = 2, .parent_train_output_rmse = 3, .combined_train_mass_error = 4, .combined_train_output_rmse = 5, .elapsed_ns = 6 };
    const legacy = try legacyHeads(a, &.{head});
    defer a.free(legacy);
    const json = try std.json.Stringify.valueAlloc(a, legacy, .{});
    defer a.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "train_metrics") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "parent_train") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "combined_train") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "elapsed_ns") != null);
}

test "whole replay augmented workspace covers actual E4 query geometry" {
    const floor = try augmentedWorkspaceFloor();
    try std.testing.expect(floor > legacy_workspace_bytes);
    try std.testing.expect(floor < augmented_workspace_bytes);
    try std.testing.expectError(error.Overflow, std.math.add(usize, std.math.maxInt(usize), floor));
}

test "whole replay mandatory free basis equality fits nonunit NNLS weights" {
    const probabilities = [_]f64{ 0.45, 0.20, 0.35, 0.35, 0.60, 0.05, 0.40, 0.40, 0.20 };
    const targets = [_]f64{ 1, 1, 1 };
    var result = try solver.fitMatricesMassTargetPolicyForTest(std.testing.allocator, .{ .probabilities = &probabilities, .targets = &targets, .rows = 3, .dim = 1, .query_rows = 3 }, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .mandatory_training_argmax = true, .mandatory_training_top_k = 1 }, &.{ 1, 1, 1 }, .raw);
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.source_indices);
    try std.testing.expectApproxEqAbs(@as(f64, 2), @exp(result.bias[0]), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), @exp(result.bias[1]), 1e-10);
    try std.testing.expect(@exp(result.bias[0]) != 1 and @exp(result.bias[1]) != 1);
    var too_small: [1]usize = undefined;
    try std.testing.expectError(error.CompactionFitRejected, solver.mandatoryTrainingArgmaxSeeds(&probabilities, 3, 3, &too_small));
}
