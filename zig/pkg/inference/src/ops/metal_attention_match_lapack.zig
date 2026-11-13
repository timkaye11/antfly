// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! FP64 Householder least squares with the reference solver's scratch bound.
const std = @import("std");
const solver = @import("../runtime/kv/attention_match.zig");
const builtin = @import("builtin");

// LP64 CBLAS_NEW_LAPACK declarations from the macOS 13.3+ SDK.
extern "c" fn @"cblas_dgemv$NEWLAPACK"(c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;
extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;
extern "c" fn @"cblas_ddot$NEWLAPACK"(c_int, [*]const f64, c_int, [*]const f64, c_int) f64;
extern "c" fn @"cblas_daxpy$NEWLAPACK"(c_int, f64, [*]const f64, c_int, [*]f64, c_int) void;
extern "c" fn @"cblas_dnrm2$NEWLAPACK"(c_int, [*]const f64, c_int) f64;

pub const GramAdvisoryCounters = struct {
    calls: u64 = 0,
    gram_precomputes: u64 = 0,
    unsupported_fallbacks: u64 = 0,
    allocation_fallbacks: u64 = 0,
    nonfinite_fallbacks: u64 = 0,
    candidate_attempts: u64 = 0,
    candidate_kkt_ready: u64 = 0,
    candidate_incomplete: u64 = 0,
    native_original_p_kkt_checks: u64 = 0,
    advisory_bytes_peak: usize = 0,
    preflight_peak_bytes: usize = 0,
};

fn advisoryPeakBytes(q: usize, n: usize, m: usize) !usize {
    const addBytes = struct {
        fn one(total: *usize, count: usize, size: usize) !void {
            total.* = try std.math.add(usize, total.*, try std.math.mul(usize, count, size));
        }
    }.one;
    var total: usize = 0;
    // Full Gram and correlation, followed by every live selectImpl buffer.
    try addBytes(&total, try std.math.add(usize, try std.math.mul(usize, n, n), n), @sizeOf(f64));
    try addBytes(&total, if (q > 1) n else 0, @sizeOf(f64));
    try addBytes(&total, std.mem.alignForward(usize, n, @alignOf(f64)), @sizeOf(bool));
    try addBytes(&total, try std.math.mul(usize, q, m), 2 * @sizeOf(f64)); // basis + orth
    try addBytes(&total, q, @sizeOf(f64)); // residual
    try addBytes(&total, try std.math.add(usize, try std.math.mul(usize, m, m), 4 * m), @sizeOf(f64));
    // Worst native original-P active-set state plus its nested scalar
    // Householder backend. The allocator remains authoritative for alignment.
    try addBytes(&total, std.mem.alignForward(usize, m, @alignOf(f64)), @sizeOf(bool));
    try addBytes(&total, m, @sizeOf(usize));
    try addBytes(&total, try std.math.add(usize, try std.math.mul(usize, q, m), q + 2 * m), @sizeOf(f64));
    try addBytes(&total, try std.math.add(usize, try std.math.mul(usize, try std.math.add(usize, q, m), m), try std.math.add(usize, try std.math.add(usize, q, m), try std.math.add(usize, m, try std.math.add(usize, q, @max(m, 1))))), @sizeOf(f64));
    return total;
}

/// FP64 residual OMP with two-pass reorthogonalization. Acceleration changes
/// dot products and matrix/vector evaluation only. Both active-set fallback
/// and final mass acceptance use the common native direct NNLS/KKT solver.
/// P is already validated by fitMatrices. Scratch stays below the native
/// selector bound: the transposed Q-by-N copy is replaced by N scores/flags;
/// the single-query case needs only flags. No source KV or held data enters.
pub fn select(_: ?*anyopaque, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
    return selectImpl(allocator, p, importance, null, q, n, mandatory, indices, weights, cfg, true, null, null, null);
}

/// Diagnostic raw-correlation OMP with an explicit nonnegative mass RHS.
/// It shares the normalized selector's Householder and active-set policy;
/// only residual initialization and column scoring differ.
pub fn selectRawTarget(_: ?*anyopaque, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
    return selectImpl(allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg, false, null, null, null);
}

pub const PreparedRawGram = struct {
    allocator: std.mem.Allocator,
    storage: []f64,
    q: usize,
    n: usize,

    pub fn gram(self: *const PreparedRawGram) []const f64 {
        return self.storage[0 .. self.n * self.n];
    }
    pub fn correlation(self: *const PreparedRawGram) []const f64 {
        return self.storage[self.n * self.n ..];
    }
    pub fn deinit(self: *PreparedRawGram) void {
        self.allocator.free(self.storage);
        self.* = undefined;
    }
};

pub fn rawGramPayloadBytes(n: usize) !usize {
    if (n == 0 or n > 1024) return error.UnsupportedOperation;
    return std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, n, n), n), @sizeOf(f64));
}

/// Allocates once from the range owner. No candidate scratch escapes.
pub fn prepareRawGram(allocator: std.mem.Allocator, p: []const f64, rhs: []const f64, q: usize, n: usize, control: solver.Control) !PreparedRawGram {
    if (q == 0 or n == 0 or n > 1024 or p.len != try std.math.mul(usize, q, n) or rhs.len != q) return error.InvalidAttentionShape;
    try control.check();
    const count = try std.math.add(usize, try std.math.mul(usize, n, n), n);
    const storage = try allocator.alloc(f64, count);
    errdefer allocator.free(storage);
    const nq = std.math.cast(c_int, q) orelse return error.InvalidAttentionShape;
    const nn = std.math.cast(c_int, n) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(101, 112, 111, nn, nn, nq, 1, p.ptr, nn, p.ptr, nn, 0, storage.ptr, nn);
    @"cblas_dgemv$NEWLAPACK"(101, 112, nq, nn, 1, p.ptr, nn, rhs.ptr, 1, 0, storage[n * n ..].ptr, 1);
    for (storage) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    try control.check();
    return .{ .allocator = allocator, .storage = storage, .q = q, .n = n };
}

/// Consumes a range-owned Gram but keeps original-P NNLS/KKT as authority.
pub fn selectRawTargetPreparedGram(counters: *GramAdvisoryCounters, prepared: *const PreparedRawGram, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
    // The range token binds the authenticated immutable source/TRAIN/critical
    // authority. The fitter establishes q/n on first consume and requires the
    // same design width thereafter, avoiding three full P/RHS hash rescans.
    if (prepared.q != q or prepared.n != n) return error.RawRangeIdentityMismatch;
    counters.calls +|= 1;
    const result = try selectImpl(allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg, false, prepared.gram(), prepared.correlation(), null);
    counters.candidate_attempts +|= 1;
    counters.native_original_p_kkt_checks +|= 1;
    return result;
}

/// Private diagnostic proposal. The Gram matrix is advisory only: selection
/// scores still use original P, and every proposed weight vector is passed to
/// the unchanged original-P native NNLS/KKT solver. A bad, singular, or
/// insufficiently converged proposal therefore falls back to its Householder
/// path. Production remains on selectRawTarget until this seam is explicitly
/// selected and qualified.
pub fn selectRawTargetGramAdvisory(raw: ?*anyopaque, allocator: std.mem.Allocator, p: []const f64, importance: []const f64, rhs: []const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config) !usize {
    const counters: *GramAdvisoryCounters = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
    counters.calls +|= 1;
    const m = indices.len;
    if (n > 1024) {
        counters.unsupported_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    }
    if (q == 0 or n == 0 or m == 0 or m > n or p.len != try std.math.mul(usize, q, n) or rhs.len != q) return error.InvalidAttentionShape;
    try cfg.control.check();
    const peak_bytes = advisoryPeakBytes(q, n, m) catch {
        counters.unsupported_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    };
    counters.preflight_peak_bytes = @max(counters.preflight_peak_bytes, peak_bytes);
    if (peak_bytes > cfg.max_workspace_bytes) {
        counters.unsupported_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    }
    const gram_len = try std.math.mul(usize, n, n);
    const advisory_len = try std.math.add(usize, gram_len, n);
    const advisory_bytes = try std.math.mul(usize, advisory_len, @sizeOf(f64));
    counters.advisory_bytes_peak = @max(counters.advisory_bytes_peak, advisory_bytes);
    if (advisory_bytes > cfg.max_workspace_bytes) {
        counters.unsupported_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    }
    const advisory = allocator.alloc(f64, advisory_len) catch |err| switch (err) {
        error.OutOfMemory => {
            counters.allocation_fallbacks +|= 1;
            return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
        },
    };
    defer allocator.free(advisory);
    const gram = advisory[0..gram_len];
    const correlation = advisory[gram_len..];
    const nq = std.math.cast(c_int, q) orelse return error.InvalidAttentionShape;
    const nn = std.math.cast(c_int, n) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(101, 112, 111, nn, nn, nq, 1, p.ptr, nn, p.ptr, nn, 0, gram.ptr, nn);
    @"cblas_dgemv$NEWLAPACK"(101, 112, nq, nn, 1, p.ptr, nn, rhs.ptr, 1, 0, correlation.ptr, 1);
    for (gram) |value| if (!std.math.isFinite(value)) {
        counters.nonfinite_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    };
    for (correlation) |value| if (!std.math.isFinite(value)) {
        counters.nonfinite_fallbacks +|= 1;
        return selectRawTarget(null, allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg);
    };
    counters.gram_precomputes +|= 1;
    return selectImpl(allocator, p, importance, rhs, q, n, mandatory, indices, weights, cfg, false, gram, correlation, counters);
}

fn warmGramCandidate(gram: []const f64, correlation: []const f64, q: usize, n: usize, selected: []const usize, weights: []f64, cfg: solver.Config) !bool {
    @memset(weights, 0);
    const tolerance = cfg.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(q)));
    var sweep: usize = 0;
    while (sweep < cfg.max_sweeps) : (sweep += 1) {
        try cfg.control.check();
        for (selected, 0..) |column, j| {
            const diagonal = gram[column * n + column];
            if (!std.math.isFinite(diagonal) or diagonal < 0) return false;
            if (diagonal == 0) continue;
            var gradient = correlation[column];
            for (selected, weights) |other, weight| gradient -= gram[column * n + other] * weight;
            if (!std.math.isFinite(gradient)) return false;
            const next = @max(0, weights[j] + gradient / diagonal);
            if (!std.math.isFinite(next)) return false;
            weights[j] = next;
        }
        var violation: f64 = 0;
        for (selected, weights) |column, weight| {
            const norm = @sqrt(gram[column * n + column]);
            if (!std.math.isFinite(norm) or norm == 0) continue;
            var gradient = correlation[column];
            for (selected, weights) |other, other_weight| gradient -= gram[column * n + other] * other_weight;
            gradient /= norm;
            if (!std.math.isFinite(gradient)) return false;
            violation = @max(violation, if (weight > 0) @abs(gradient) else @max(0, gradient));
        }
        if (violation <= tolerance) return true;
    }
    return false;
}

fn selectImpl(allocator: std.mem.Allocator, p: []const f64, importance: []const f64, target: ?[]const f64, q: usize, n: usize, mandatory: []const usize, indices: []usize, weights: []f64, cfg: solver.Config, normalize_score: bool, gram: ?[]const f64, correlation: ?[]const f64, gram_counters: ?*GramAdvisoryCounters) !usize {
    if (comptime !supported) return error.UnsupportedMetalFp64Selection;
    const m = indices.len;
    if (q == 0 or n == 0 or m == 0 or m > n or weights.len != m or importance.len != n or
        p.len != try std.math.mul(usize, q, n) or mandatory.len > m or (target != null and target.?.len != q)) return error.InvalidAttentionShape;
    if (target) |rhs| for (rhs) |value| if (!std.math.isFinite(value) or value < 0) return error.InvalidFitConfig;
    for (mandatory, 0..) |index, i| if (index >= n or std.mem.indexOfScalar(usize, mandatory[0..i], index) != null) return error.InvalidSelectionResult;
    const nq = std.math.cast(c_int, q) orelse return error.InvalidAttentionShape;
    const nn = std.math.cast(c_int, n) orelse return error.InvalidAttentionShape;
    _ = std.math.cast(c_int, m) orelse return error.InvalidAttentionShape;
    try cfg.control.check();
    for (importance) |norm| if (!std.math.isFinite(norm) or norm < 0) return error.NonFiniteAttention;
    const scores = try allocator.alloc(f64, if (q > 1) n else 0);
    defer allocator.free(scores);
    const used_storage = try allocator.alloc(bool, std.mem.alignForward(usize, n, @alignOf(f64)));
    defer allocator.free(used_storage);
    const used = used_storage[0..n];
    const basis = try allocator.alloc(f64, try std.math.mul(usize, q, m));
    defer allocator.free(basis);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, m);
    defer allocator.free(norms);
    const orth = try allocator.alloc(f64, try std.math.mul(usize, q, m));
    defer allocator.free(orth);
    const triangular = try allocator.alloc(f64, try std.math.mul(usize, m, m));
    defer allocator.free(triangular);
    const projected = try allocator.alloc(f64, m);
    defer allocator.free(projected);
    const trial = try allocator.alloc(f64, m);
    defer allocator.free(trial);
    const column_norms = try allocator.alloc(f64, m);
    defer allocator.free(column_norms);
    @memset(used, false);
    @memset(triangular, 0);
    @memset(weights, 0);
    if (target) |rhs| @memcpy(residual, rhs) else @memset(residual, 1);
    var qr_valid = true;
    var active_solves: usize = 0;
    for (0..m) |selected| {
        try cfg.control.check();
        if (q > 1) {
            if (gram != null and correlation != null) {
                // Exact Gram-domain P^T(rhs-Pw). The original P remains below
                // for basis formation and the authoritative native KKT solve.
                for (scores, 0..) |*score, column| {
                    score.* = correlation.?[column];
                    for (indices[0..selected], weights[0..selected]) |chosen, weight| score.* -= gram.?[column * n + chosen] * weight;
                }
            } else @"cblas_dgemv$NEWLAPACK"(101, 112, nq, nn, 1, p.ptr, nn, residual.ptr, 1, 0, scores.ptr, 1);
        }
        var best: ?usize = if (selected < mandatory.len) mandatory[selected] else null;
        var best_score: f64 = -std.math.inf(f64);
        if (best == null) for (0..n) |column| {
            const dot = if (q == 1) p[column] * residual[0] else scores[column];
            const norm = importance[column];
            if (used[column]) continue;
            const score = if (!normalize_score) dot else if (norm > 0) dot / @sqrt(norm) else 0;
            if (!std.math.isFinite(score)) return error.NonFiniteAttention;
            if (best == null or score > best_score) {
                best = column;
                best_score = score;
            }
        };
        indices[selected] = best orelse return error.InvalidFitConfig;
        used[indices[selected]] = true;
        const count = selected + 1;
        const nc: c_int = @intCast(count);
        for (0..q) |i| for (indices[0..count], 0..) |index, j| {
            basis[i * count + j] = p[i * n + index];
        };
        weights[selected] = 0;
        column_norms[selected] = @sqrt(importance[indices[selected]]);
        if (qr_valid and column_norms[selected] > 0) {
            const column = orth[selected * q ..][0..q];
            for (column, 0..) |*value, i| value.* = p[i * n + indices[selected]] / column_norms[selected];
            for (0..2) |_| for (0..selected) |j| {
                const prior = orth[j * q ..][0..q];
                const dot = @"cblas_ddot$NEWLAPACK"(nq, column.ptr, 1, prior.ptr, 1);
                triangular[j * m + selected] += dot;
                @"cblas_daxpy$NEWLAPACK"(nq, -dot, prior.ptr, 1, column.ptr, 1);
            };
            const norm = @"cblas_dnrm2$NEWLAPACK"(nq, column.ptr, 1);
            qr_valid = std.math.isFinite(norm) and norm > 1e-12 and count <= q;
            if (qr_valid) {
                triangular[selected * m + selected] = norm;
                projected[selected] = 0;
                for (column, 0..) |*value, projected_index| {
                    value.* /= norm;
                    projected[selected] += value.* * (if (target) |rhs| rhs[projected_index] else 1);
                }
            }
        } else qr_valid = false;
        if (count < mandatory.len) continue;
        var feasible = qr_valid;
        if (qr_valid) {
            var j = count;
            while (j > 0) {
                j -= 1;
                var value = projected[j];
                for (j + 1..count) |k| value -= triangular[j * m + k] * trial[k];
                trial[j] = value / triangular[j * m + j];
            }
            for (trial[0..count], column_norms[0..count]) |*weight, norm| {
                weight.* /= norm;
                if (!std.math.isFinite(weight.*) or weight.* < 0) feasible = false;
            }
        }
        if (feasible) {
            if (target) |rhs| @memcpy(residual, rhs) else @memset(residual, 1);
            @"cblas_dgemv$NEWLAPACK"(101, 111, nq, nc, -1, basis.ptr, nc, trial.ptr, 1, 1, residual.ptr, 1);
            @"cblas_dgemv$NEWLAPACK"(101, 112, nq, nc, 1, basis.ptr, nc, residual.ptr, 1, 0, norms.ptr, 1);
            var violation: f64 = 0;
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
        if (feasible) @memcpy(weights[0..count], trial[0..count]) else active_solves += if (target) |rhs| blk: {
            if (gram != null and correlation != null) {
                const counters = gram_counters;
                if (counters) |value| value.candidate_attempts +|= 1;
                if (try warmGramCandidate(gram.?, correlation.?, q, n, indices[0..count], weights[0..count], cfg)) {
                    if (counters) |value| value.candidate_kkt_ready +|= 1;
                } else {
                    if (counters) |value| value.candidate_incomplete +|= 1;
                }
                if (counters) |value| value.native_original_p_kkt_checks +|= 1;
            }
            break :blk try solver.fitMassActiveTargetDiagnostic(allocator, basis[0 .. q * count], rhs, q, count, weights[0..count], residual, norms[0..count], cfg, true);
        } else try solver.fitMassActive(allocator, basis[0 .. q * count], q, count, weights[0..count], residual, norms[0..count], cfg, true);
    }
    return active_solves;
}

pub const supported = if (builtin.os.tag == .macos)
    builtin.os.version_range.semver.min.order(.{ .major = 13, .minor = 3, .patch = 0 }) != .lt
else
    false;

test "range prepared Gram is shared without changing original-P KKT authority" {
    if (!supported) return error.SkipZigTest;
    const p = [_]f64{ 1, 0, 0.2, 0, 1, 0.8, 0.5, 0.5, 0.5 };
    const rhs = [_]f64{ 1, 1, 1 };
    const importance = [_]f64{ 1.25, 1.25, 0.93 };
    var prepared = try prepareRawGram(std.testing.allocator, &p, &rhs, 3, 3, .{});
    defer prepared.deinit();
    var counters: GramAdvisoryCounters = .{};
    var expected_i: [2]usize = undefined;
    var expected_w: [2]f64 = undefined;
    const expected_solves = try selectRawTarget(null, std.testing.allocator, &p, &importance, &rhs, 3, 3, &.{0}, &expected_i, &expected_w, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    var actual_i: [2]usize = undefined;
    var actual_w: [2]f64 = undefined;
    const actual_solves = try selectRawTargetPreparedGram(&counters, &prepared, std.testing.allocator, &p, &importance, &rhs, 3, 3, &.{0}, &actual_i, &actual_w, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    try std.testing.expectEqual(expected_i, actual_i);
    for (expected_w, actual_w) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-10);
    try std.testing.expectEqual(expected_solves, actual_solves);
    try std.testing.expect(counters.native_original_p_kkt_checks > 0);
}

test "prepared Gram design width mismatch fails closed" {
    if (!supported) return error.SkipZigTest;
    const p = [_]f64{ 1, 0, 0, 1 };
    const rhs = [_]f64{ 1, 1 };
    var prepared = try prepareRawGram(std.testing.allocator, &p, &rhs, 2, 2, .{});
    defer prepared.deinit();
    var counters: GramAdvisoryCounters = .{};
    var indices: [1]usize = undefined;
    var weights: [1]f64 = undefined;
    try std.testing.expectError(error.RawRangeIdentityMismatch, selectRawTargetPreparedGram(&counters, &prepared, std.testing.allocator, &.{ 1, 1 }, &.{1}, &rhs, 2, 1, &.{}, &indices, &weights, .{ .target_rows = 1 }));
}

test "N1024 range Gram payload is exactly 8396800 bytes" {
    try std.testing.expectEqual(@as(usize, 8_396_800), try rawGramPayloadBytes(1024));
    try std.testing.expectError(error.UnsupportedOperation, rawGramPayloadBytes(1025));
}

test "one prepared range Gram serves three candidates with three original P checks" {
    if (!supported) return error.SkipZigTest;
    const p = [_]f64{ 1, 0, 0, 0, 0, 1, 0, 0, 0.5, 0.5, 1, 0, 0.25, 0.25, 0.25, 1 };
    const rhs = [_]f64{ 1, 1, 1, 1 };
    const importance = [_]f64{ 1.3125, 1.3125, 1.0625, 1 };
    var prepared = try prepareRawGram(std.testing.allocator, &p, &rhs, 4, 4, .{});
    defer prepared.deinit();
    var counters: GramAdvisoryCounters = .{};
    var indices1: [1]usize = undefined;
    var w1: [1]f64 = undefined;
    _ = try selectRawTargetPreparedGram(&counters, &prepared, std.testing.allocator, &p, &importance, &rhs, 4, 4, &.{}, &indices1, &w1, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set });
    var indices2: [2]usize = undefined;
    var w2: [2]f64 = undefined;
    _ = try selectRawTargetPreparedGram(&counters, &prepared, std.testing.allocator, &p, &importance, &rhs, 4, 4, &.{}, &indices2, &w2, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set });
    var indices3: [3]usize = undefined;
    var w3: [3]f64 = undefined;
    _ = try selectRawTargetPreparedGram(&counters, &prepared, std.testing.allocator, &p, &importance, &rhs, 4, 4, &.{}, &indices3, &w3, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set });
    try std.testing.expectEqual(@as(u64, 3), counters.calls);
    try std.testing.expectEqual(@as(u64, 3), counters.native_original_p_kkt_checks);
}

test "range Gram build allocation failure and cancellation publish no owner" {
    if (!supported) return error.SkipZigTest;
    const p = [_]f64{ 1, 0, 0, 1 };
    const rhs = [_]f64{ 1, 1 };
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, prepareRawGram(failing.allocator(), &p, &rhs, 2, 2, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, prepareRawGram(std.testing.allocator, &p, &rhs, 2, 2, .{ .check_fn = Cancel.check }));
}

// SDK lapack_version.h binds the LP64 API to these macOS 13.3+ symbols.
extern "c" fn @"dgelqf$NEWLAPACK"(*const c_int, *const c_int, [*]f64, *const c_int, [*]f64, [*]f64, *const c_int, *c_int) void;
extern "c" fn @"dormlq$NEWLAPACK"(*const u8, *const u8, *const c_int, *const c_int, *const c_int, [*]f64, *const c_int, [*]const f64, [*]f64, *const c_int, [*]f64, *const c_int, *c_int) void;
extern "c" fn cblas_dtrsm(c_int, c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]f64, c_int) void;

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub const WorkspaceMetadata = struct {
    lq_optimal_elements: usize,
    apply_optimal_elements: usize,
    declared_cap_elements: usize,
    chosen_elements: usize,
    blas_backsolve: bool,
};

pub fn chooseWorkspaceElements(minimum: usize, declared_cap: usize, lq_optimal: f64, apply_optimal: f64) !usize {
    if (minimum == 0 or declared_cap < minimum or !std.math.isFinite(lq_optimal) or !std.math.isFinite(apply_optimal) or lq_optimal < 1 or apply_optimal < 1) return error.AttentionMatchLapackFailure;
    const chosen: usize = @intFromFloat(@min(@as(f64, @floatFromInt(declared_cap)), @max(lq_optimal, apply_optimal)));
    if (chosen < minimum) return error.AttentionMatchLapackFailure;
    return chosen;
}

pub fn values(context: ?*anyopaque, allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, out: []f64) !void {
    _ = context;
    try valuesImpl(allocator, a, y, q, m, d, ridge, control, null, false, null, out);
}

/// Diagnostic-only larger-workspace seam. The ordinary `values` entrypoint and
/// its legacy cap remain unchanged.
pub fn valuesWithWorkspaceLimit(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, work_cap_elements: usize, blas_backsolve: bool, metadata: *WorkspaceMetadata, out: []f64) !void {
    try valuesImpl(allocator, a, y, q, m, d, ridge, control, work_cap_elements, blas_backsolve, metadata, out);
}

fn valuesImpl(allocator: std.mem.Allocator, a: []const f64, y: []const f64, q: usize, m: usize, d: usize, ridge: f64, control: solver.Control, work_cap_elements: ?usize, blas_backsolve: bool, metadata: ?*WorkspaceMetadata, out: []f64) !void {
    if (comptime !supported) return error.UnsupportedMetalFp64Values;
    try control.check();
    if (q == 0 or m == 0 or d == 0 or !std.math.isFinite(ridge) or ridge < 0 or
        a.len != try mul(q, m) or y.len != try mul(q, d) or out.len != try mul(m, d))
        return error.InvalidAttentionShape;
    for (a) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    for (y) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    const rows = try std.math.add(usize, q, m);
    const nr = std.math.cast(c_int, rows) orelse return error.InvalidAttentionShape;
    const nm = std.math.cast(c_int, m) orelse return error.InvalidAttentionShape;
    const nd = std.math.cast(c_int, d) orelse return error.InvalidAttentionShape;
    const qr = try allocator.alloc(f64, try mul(rows, m));
    defer allocator.free(qr);
    const rhs = try allocator.alloc(f64, try mul(rows, d));
    defer allocator.free(rhs);
    const tau = try allocator.alloc(f64, m);
    defer allocator.free(tau);
    @memset(qr, 0);
    @memset(rhs, 0);
    @memcpy(qr[0..a.len], a);
    @memcpy(rhs[0..y.len], y);
    for (0..m) |j| qr[(q + j) * m + j] = @sqrt(ridge);
    const side: u8 = 'R';
    const trans: u8 = 'T';
    const query: c_int = -1;
    var info: c_int = 0;
    var lq_query: f64 = 0;
    var apply_query: f64 = 0;
    @"dgelqf$NEWLAPACK"(&nm, &nr, qr.ptr, &nm, tau.ptr, @ptrCast(&lq_query), &query, &info);
    if (info != 0) return error.AttentionMatchLapackFailure;
    @"dormlq$NEWLAPACK"(&side, &trans, &nd, &nr, &nm, qr.ptr, &nm, tau.ptr, rhs.ptr, &nd, @ptrCast(&apply_query), &query, &info);
    if (info != 0 or !std.math.isFinite(lq_query) or !std.math.isFinite(apply_query) or
        lq_query < 1 or apply_query < 1) return error.AttentionMatchLapackFailure;
    // Reference QR uses rows + max(m,d) elements for v and dots. Preserve
    // that exact bound across tau + work, even if LAPACK prefers more work.
    const legacy_cap = try std.math.add(usize, q, @max(m, d));
    const cap = work_cap_elements orelse legacy_cap;
    if (cap < @max(m, d)) return error.CompactionWorkspaceLimit;
    const count = try chooseWorkspaceElements(@max(m, d), cap, lq_query, apply_query);
    const lwork = std.math.cast(c_int, count) orelse return error.InvalidAttentionShape;
    const work = try allocator.alloc(f64, count);
    defer allocator.free(work);
    if (metadata) |record| record.* = .{ .lq_optimal_elements = @intFromFloat(@ceil(lq_query)), .apply_optimal_elements = @intFromFloat(@ceil(apply_query)), .declared_cap_elements = cap, .chosen_elements = count, .blas_backsolve = blas_backsolve };
    try control.check();
    // Row-major A is column-major A^T. Its LQ factorization represents A's
    // Householder QR without an extra transpose or normal equations.
    @"dgelqf$NEWLAPACK"(&nm, &nr, qr.ptr, &nm, tau.ptr, work.ptr, &lwork, &info);
    if (info != 0) return error.AttentionMatchLapackFailure;
    try control.check();
    // Y^T Q^T is the row-major representation of QY, the transformed RHS.
    @"dormlq$NEWLAPACK"(&side, &trans, &nd, &nr, &nm, qr.ptr, &nm, tau.ptr, rhs.ptr, &nd, work.ptr, &lwork, &info);
    if (info != 0) return error.AttentionMatchLapackFailure;
    try control.check();
    // Validate rank before either backsolve. The diagnostic BLAS path uses the
    // same row-major upper-triangular factor and transformed RHS as the scalar
    // reference; it changes dispatch only, never QR or the fitted objective.
    for (0..m) |j| {
        const diagonal = qr[j * m + j];
        if (!std.math.isFinite(diagonal) or diagonal == 0) return error.SingularValueFit;
    }
    if (blas_backsolve) {
        try control.check();
        cblas_dtrsm(101, 141, 121, 111, 131, nm, nd, 1, qr.ptr, nm, rhs.ptr, nd);
        try control.check();
        @memcpy(out, rhs[0 .. m * d]);
        for (out) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
        return;
    }
    var j = m;
    while (j > 0) {
        j -= 1;
        try control.check();
        const diagonal = qr[j * m + j];
        for (0..d) |col| {
            var value = rhs[j * d + col];
            for (j + 1..m) |k| value -= qr[j * m + k] * out[k * d + col];
            out[j * d + col] = value / diagonal;
            if (!std.math.isFinite(out[j * d + col])) return error.NonFiniteAttention;
        }
    }
}

test "bounded LAPACK workspace chooses queried optimum within declared cap" {
    try std.testing.expectEqual(@as(usize, 40), try chooseWorkspaceElements(8, 64, 40, 32));
    try std.testing.expectEqual(@as(usize, 64), try chooseWorkspaceElements(8, 64, 96, 32));
    try std.testing.expectError(error.AttentionMatchLapackFailure, chooseWorkspaceElements(8, 7, 40, 32));
    try std.testing.expectError(error.AttentionMatchLapackFailure, chooseWorkspaceElements(8, 64, std.math.nan(f64), 32));
}

test "private Gram advisory produces KKT warm states for full-rank and duplicate columns" {
    var weights = [_]f64{ 0, 0 };
    try std.testing.expect(try warmGramCandidate(
        &.{ 1, 0, 0, 1 },
        &.{ 2, 3 },
        2,
        2,
        &.{ 0, 1 },
        &weights,
        .{ .target_rows = 2 },
    ));
    try std.testing.expectApproxEqAbs(@as(f64, 2), weights[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), weights[1], 1e-12);

    try std.testing.expect(try warmGramCandidate(
        &.{ 1, 1, 1, 1 },
        &.{ 1, 1 },
        2,
        2,
        &.{ 0, 1 },
        &weights,
        .{ .target_rows = 2 },
    ));
    try std.testing.expectApproxEqAbs(@as(f64, 1), weights[0] + weights[1], 1e-12);
}

test "private Gram advisory rejects nonfinite candidate and honors cancellation" {
    var weights = [_]f64{0};
    try std.testing.expect(!(try warmGramCandidate(
        &.{std.math.nan(f64)},
        &.{1},
        1,
        1,
        &.{0},
        &weights,
        .{ .target_rows = 1 },
    )));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, warmGramCandidate(
        &.{1},
        &.{1},
        1,
        1,
        &.{0},
        &weights,
        .{ .target_rows = 1, .control = .{ .check_fn = Cancel.check } },
    ));
}

test "private Gram advisory allocation and cap fallback preserve native selector" {
    if (comptime !supported) return error.SkipZigTest;
    var p: [128]f64 = undefined;
    var importance: [64]f64 = undefined;
    for (0..64) |j| {
        p[j] = @as(f64, @floatFromInt(j + 1)) / 64;
        p[64 + j] = @as(f64, @floatFromInt(64 - j)) / 64;
        importance[j] = p[j] * p[j] + p[64 + j] * p[64 + j];
    }
    var expected_index: [1]usize = undefined;
    var expected_weight: [1]f64 = undefined;
    _ = try selectRawTarget(null, std.testing.allocator, &p, &importance, &.{ 1, 1 }, 2, 64, &.{}, &expected_index, &expected_weight, .{ .target_rows = 1 });

    var buffer: [8192]u8 align(@alignOf(f64)) = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&buffer);
    var counters: GramAdvisoryCounters = .{};
    var actual_index: [1]usize = undefined;
    var actual_weight: [1]f64 = undefined;
    _ = try selectRawTargetGramAdvisory(&counters, fixed.allocator(), &p, &importance, &.{ 1, 1 }, 2, 64, &.{}, &actual_index, &actual_weight, .{ .target_rows = 1 });
    try std.testing.expectEqual(@as(u64, 1), counters.allocation_fallbacks);
    try std.testing.expectEqualSlices(usize, &expected_index, &actual_index);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_weight), std.mem.sliceAsBytes(&actual_weight));

    counters = .{};
    _ = try selectRawTargetGramAdvisory(&counters, std.testing.allocator, &p, &importance, &.{ 1, 1 }, 2, 64, &.{}, &actual_index, &actual_weight, .{ .target_rows = 1, .max_workspace_bytes = 1 });
    try std.testing.expectEqual(@as(u64, 1), counters.unsupported_fallbacks);
    try std.testing.expectEqualSlices(usize, &expected_index, &actual_index);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&expected_weight), std.mem.sliceAsBytes(&actual_weight));
}

test "private Gram advisory ill-conditioned proposal always reaches original-P native KKT" {
    if (comptime !supported) return error.SkipZigTest;
    const p = [_]f64{
        1, 1 + 1e-12, 0,
        0, 1e-12,     1,
        1, 1,         1,
    };
    const importance = [_]f64{ 2, (1 + 1e-12) * (1 + 1e-12) + 1, 2 };
    var indices: [2]usize = undefined;
    var weights: [2]f64 = undefined;
    var counters: GramAdvisoryCounters = .{};
    _ = try selectRawTargetGramAdvisory(&counters, std.testing.allocator, &p, &importance, &.{ 1, 0.5, 1 }, 3, 3, &.{ 0, 1 }, &indices, &weights, .{ .target_rows = 2, .max_sweeps = 128 });
    try std.testing.expect(counters.gram_precomputes == 1);
    try std.testing.expect(counters.candidate_attempts > 0);
    try std.testing.expectEqual(counters.candidate_attempts, counters.native_original_p_kkt_checks);
    for (weights) |weight| try std.testing.expect(std.math.isFinite(weight) and weight >= 0);
}
