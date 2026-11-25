// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Training-only diagnostic: jointly rescale frozen per-chunk bases, then
//! refit values with those weights fixed. Heldout data is intentionally absent.
const std = @import("std");
const solver = @import("attention_match.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const anchored = @import("attention_match_joint_chunk_mass_anchored.zig");

pub const MassStrategy = enum { global_active_nnls_v16, anchored_active_nnls_v17, identity_chunk_coefficients_v18 };

pub const FrozenChunk = struct {
    start: usize,
    rows: usize,
    selected_local: []const usize,
    weights: []const f64,
    /// Borrowed local source-row indices. `selected_local` remains the full
    /// output order; exact cores must occur in it with weight exactly one.
    exact_core_local: []const usize = &.{},
};

pub const ChunkResult = struct {
    start: usize,
    rows: usize,
    value_offset: usize,
    selected_count: usize,
    alpha: f64,
};

pub const FixedOriginalTerm = struct {
    allocator: std.mem.Allocator,
    start: usize,
    rows: usize,
    query_count: usize,
    dim: usize,
    original_values_ptr: [*]const f64,
    train_ptr: [*]const f64,
    source_digest: [32]u8,
    exact_core_digest: [32]u8,
    rho: []f64,
    residual: []f64,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.residual);
        self.allocator.free(self.rho);
        self.* = undefined;
    }
};

fn exactCoreDigest(core: []const usize) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-exact-core-local.v1\x00");
    hash.update(std.mem.sliceAsBytes(core));
    return hash.finalResult();
}

fn isExactCore(chunk: FrozenChunk, local: usize) bool {
    return std.mem.indexOfScalar(usize, chunk.exact_core_local, local) != null;
}

pub const Result = struct {
    allocator: std.mem.Allocator,
    chunks: []ChunkResult,
    values: []f64,
    bias: []f64,
    nnls_sweeps: usize,
    training_diagnostics: ?anchored.Diagnostics = null,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.chunks);
        self.allocator.free(self.values);
        self.allocator.free(self.bias);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn fixedOriginalTermBytes(query_count: usize, dim: usize) !usize {
    if (query_count == 0 or dim == 0) return error.InvalidAttentionShape;
    return mul(try add(query_count, try mul(query_count, dim)), @sizeOf(f64));
}

pub fn prepareFixedOriginalTerm(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, chunk: FrozenChunk, cfg: solver.Config, cached: *training_source.Source, external_live_bytes: usize, combined_cap: usize) !FixedOriginalTerm {
    try original.validate();
    if (train.len == 0 or train.len % original.dim != 0 or chunk.rows == 0 or chunk.start > original.rows or chunk.rows > original.rows - chunk.start) return error.InvalidFitConfig;
    try cached.validate(original, train, cfg);
    for (chunk.exact_core_local, 0..) |index, i| {
        const selected_index = std.mem.indexOfScalar(usize, chunk.selected_local, index) orelse return error.InvalidFitConfig;
        if (index >= chunk.rows or std.mem.indexOfScalar(usize, chunk.exact_core_local[0..i], index) != null or selected_index >= chunk.weights.len or chunk.weights[selected_index] != 1) return error.InvalidFitConfig;
    }
    try cfg.control.check();
    const q = train.len / original.dim;
    const bytes = try fixedOriginalTermBytes(q, original.dim);
    if (external_live_bytes > combined_cap or bytes > combined_cap - external_live_bytes) return error.CompactionWorkspaceLimit;
    const rho = try allocator.alloc(f64, q);
    errdefer allocator.free(rho);
    const residual = try allocator.alloc(f64, try mul(q, original.dim));
    errdefer allocator.free(residual);
    for (0..q) |qi| {
        try cfg.control.check();
        var mass: f64 = 0;
        const out = residual[qi * original.dim ..][0..original.dim];
        @memcpy(out, cached.outputs[qi * original.dim ..][0..original.dim]);
        for (0..chunk.rows) |local| {
            if (isExactCore(chunk, local)) continue;
            const p = try cached.probability(qi, chunk.start + local);
            mass += p;
            for (out, original.values[(chunk.start + local) * original.dim ..][0..original.dim]) |*v, source_value| v.* -= p * source_value;
        }
        if (!std.math.isFinite(mass)) return error.NonFiniteAttention;
        rho[qi] = mass;
    }
    try cfg.control.check();
    return .{ .allocator = allocator, .start = chunk.start, .rows = chunk.rows, .query_count = q, .dim = original.dim, .original_values_ptr = original.values.ptr, .train_ptr = train.ptr, .source_digest = cached.digest, .exact_core_digest = exactCoreDigest(chunk.exact_core_local), .rho = rho, .residual = residual };
}

fn validateChunks(original: solver.Head, chunks: []const FrozenChunk) !usize {
    if (chunks.len == 0) return error.InvalidFitConfig;
    var prior_end: usize = 0;
    var selected_total: usize = 0;
    for (chunks) |chunk| {
        if (chunk.rows == 0 or chunk.start < prior_end or chunk.start > original.rows or chunk.rows > original.rows - chunk.start or
            chunk.selected_local.len == 0 or chunk.selected_local.len != chunk.weights.len) return error.InvalidFitConfig;
        for (chunk.selected_local, chunk.weights, 0..) |index, weight, i| {
            if (index >= chunk.rows or !std.math.isFinite(weight) or weight < 0 or
                std.mem.indexOfScalar(usize, chunk.selected_local[0..i], index) != null) return error.InvalidFitConfig;
        }
        for (chunk.exact_core_local, 0..) |index, i| {
            const selected_index = std.mem.indexOfScalar(usize, chunk.selected_local, index) orelse return error.InvalidFitConfig;
            if (index >= chunk.rows or std.mem.indexOfScalar(usize, chunk.exact_core_local[0..i], index) != null or chunk.weights[selected_index] != 1) return error.InvalidFitConfig;
        }
        prior_end = chunk.start + chunk.rows;
        selected_total = try add(selected_total, chunk.selected_local.len);
    }
    return selected_total;
}

pub const Workspace = struct { required: usize, phase1_bytes: usize, phase2_bytes: usize };

pub fn workspaceBytesShape(original: solver.Head, train_rows: usize, chunk_count: usize, selected_total: usize, max_selected: usize, cfg: solver.Config, cached: bool, strategy: MassStrategy, retain_training_diagnostics: bool) !Workspace {
    if (train_rows == 0 or chunk_count == 0 or selected_total == 0 or max_selected == 0) return error.InvalidAttentionShape;
    const q = train_rows;
    const k = chunk_count;
    const qd = try mul(q, original.dim);
    const qk = try mul(q, k);
    const qm = try mul(q, max_selected);
    const batch_p = try mul(@min(q, 64), original.rows);
    const batch_y = try mul(@min(q, 64), original.dim);
    const output_values = try mul(selected_total, original.dim);
    var phase1_f64 = if (!cached) try add(try add(qd, qk), try add(batch_p, batch_y)) else qk;
    phase1_f64 = try add(phase1_f64, try add(try mul(q, 2), try mul(k, 2)));
    if (strategy == .anchored_active_nnls_v17) {
        phase1_f64 = try add(phase1_f64, try mul(k, 2));
        if (retain_training_diagnostics) phase1_f64 = try add(phase1_f64, qk);
    }
    const phase1_bytes = try mul(phase1_f64, @sizeOf(f64));
    var phase2_f64 = try add(phase1_f64, try add(qm, qd));
    phase2_f64 = try add(phase2_f64, try add(output_values, selected_total));
    var phase2_bytes = try mul(phase2_f64, @sizeOf(f64));
    phase2_bytes = try add(phase2_bytes, try mul(k, @sizeOf(ChunkResult)));
    const mass_nested = switch (strategy) {
        .anchored_active_nnls_v17 => try anchored.workspaceBytes(q, k, cfg),
        .global_active_nnls_v16 => try solver.workspaceBytes(k, k, q, 1),
        .identity_chunk_coefficients_v18 => 0,
    };
    const value_nested = try solver.valueWorkspaceBytes(q, max_selected, original.dim, cfg.ridge, cfg);
    const backend_peak = if (cfg.backend) |backend| blk: {
        const attention_peak = if (cached) 0 else try backend.workspace_fn(backend.context, original.rows, 1, @min(q, 64), original.dim);
        // Preserve the combined legacy Backend ABI. An explicit ValueBackend
        // owns this phase and prevents a value-shaped Metal workspace query.
        const legacy_value_peak = if (cfg.value_backend == null and backend.values_fn != null)
            try backend.workspace_fn(backend.context, max_selected, max_selected, q, original.dim)
        else
            0;
        break :blk @max(attention_peak, legacy_value_peak);
    } else 0;
    return .{ .required = @max(try add(phase1_bytes, @max(mass_nested, backend_peak)), try add(phase2_bytes, @max(value_nested, backend_peak))), .phase1_bytes = phase1_bytes, .phase2_bytes = phase2_bytes };
}

/// Core-aware identity-v18 workspace: output ownership uses the full selected
/// cardinality, while basis/QR/backend work uses only non-core residual rows.
pub fn workspaceBytesShapeCoreAware(original: solver.Head, train_rows: usize, chunk_count: usize, selected_total: usize, max_total_selected: usize, max_residual_selected: usize, cfg: solver.Config, cached: bool, strategy: MassStrategy, retain_training_diagnostics: bool) !Workspace {
    if (train_rows == 0 or chunk_count == 0 or selected_total == 0 or max_total_selected == 0 or max_residual_selected > max_total_selected or strategy != .identity_chunk_coefficients_v18 or retain_training_diagnostics) return error.InvalidAttentionShape;
    const q = train_rows;
    const k = chunk_count;
    const qd = try mul(q, original.dim);
    const qk = try mul(q, k);
    const qr = try mul(q, max_residual_selected);
    const batch_p = try mul(@min(q, 64), original.rows);
    const batch_y = try mul(@min(q, 64), original.dim);
    const output_values = try mul(selected_total, original.dim);
    var phase1_f64 = if (!cached) try add(try add(qd, qk), try add(batch_p, batch_y)) else qk;
    phase1_f64 = try add(phase1_f64, try add(try mul(q, 2), try mul(k, 2)));
    const phase1_bytes = try mul(phase1_f64, @sizeOf(f64));
    var phase2_f64 = try add(phase1_f64, try add(qr, qd));
    phase2_f64 = try add(phase2_f64, try add(output_values, selected_total));
    // Residual solver output is scattered around exact core rows. The map is
    // full selected length; the temporary value matrix is residual-only.
    phase2_f64 = try add(phase2_f64, try mul(max_residual_selected, original.dim));
    var phase2_bytes = try mul(phase2_f64, @sizeOf(f64));
    phase2_bytes = try add(phase2_bytes, try mul(k, @sizeOf(ChunkResult)));
    phase2_bytes = try add(phase2_bytes, try mul(max_total_selected, @sizeOf(usize)));
    const value_nested = if (max_residual_selected == 0) 0 else try solver.valueWorkspaceBytes(q, max_residual_selected, original.dim, cfg.ridge, cfg);
    const backend_peak = if (cfg.backend) |backend| blk: {
        const attention_peak = if (cached) 0 else try backend.workspace_fn(backend.context, original.rows, 1, @min(q, 64), original.dim);
        const legacy_value_peak = if (max_residual_selected != 0 and cfg.value_backend == null and backend.values_fn != null)
            try backend.workspace_fn(backend.context, max_residual_selected, max_residual_selected, q, original.dim)
        else
            0;
        break :blk @max(attention_peak, legacy_value_peak);
    } else 0;
    return .{ .required = @max(try add(phase1_bytes, backend_peak), try add(phase2_bytes, @max(value_nested, backend_peak))), .phase1_bytes = phase1_bytes, .phase2_bytes = phase2_bytes };
}

pub fn workspaceBytes(original: solver.Head, train_rows: usize, chunks: []const FrozenChunk, cfg: solver.Config, cached: bool) !Workspace {
    return workspaceBytesStrategy(original, train_rows, chunks, cfg, cached, .global_active_nnls_v16, false);
}

pub fn workspaceBytesStrategy(original: solver.Head, train_rows: usize, chunks: []const FrozenChunk, cfg: solver.Config, cached: bool, strategy: MassStrategy, retain_training_diagnostics: bool) !Workspace {
    const selected_total = try validateChunks(original, chunks);
    if (strategy != .identity_chunk_coefficients_v18) for (chunks) |chunk| if (chunk.exact_core_local.len != 0) return error.InvalidFitConfig;
    var max_m: usize = 0;
    for (chunks) |chunk| max_m = @max(max_m, chunk.selected_local.len);
    var has_core = false;
    var max_residual_m: usize = 0;
    for (chunks) |chunk| {
        has_core = has_core or chunk.exact_core_local.len != 0;
        max_residual_m = @max(max_residual_m, chunk.selected_local.len - chunk.exact_core_local.len);
    }
    if (!has_core) return workspaceBytesShape(original, train_rows, chunks.len, selected_total, max_m, cfg, cached, strategy, retain_training_diagnostics);
    return workspaceBytesShapeCoreAware(original, train_rows, chunks.len, selected_total, max_m, max_residual_m, cfg, cached, strategy, retain_training_diagnostics);
}

pub fn fit(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    chunks: []const FrozenChunk,
    cfg: solver.Config,
) !Result {
    return fitWithTrainingSource(allocator, original, train, chunks, cfg, null, null);
}

pub fn fitWithTrainingSource(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    chunks: []const FrozenChunk,
    cfg: solver.Config,
    source: ?*training_source.Source,
    counters: ?*training_source.Counters,
) !Result {
    return fitStrategy(allocator, original, train, chunks, cfg, source, counters, .global_active_nnls_v16, false);
}

pub fn fitStrategy(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    chunks: []const FrozenChunk,
    cfg: solver.Config,
    source: ?*training_source.Source,
    counters: ?*training_source.Counters,
    strategy: MassStrategy,
    retain_training_diagnostics: bool,
) !Result {
    return fitStrategyWithFixedOriginal(allocator, original, train, chunks, cfg, source, counters, strategy, retain_training_diagnostics, null);
}

pub fn fitStrategyWithFixedOriginal(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    chunks: []const FrozenChunk,
    cfg: solver.Config,
    source: ?*training_source.Source,
    counters: ?*training_source.Counters,
    strategy: MassStrategy,
    retain_training_diagnostics: bool,
    fixed_original: ?*const FixedOriginalTerm,
) !Result {
    try original.validate();
    try solver.validateConfigDiagnostic(cfg, original.rows);
    if (train.len == 0 or train.len % original.dim != 0 or original.bias == null) return error.InvalidFitConfig;
    if (source) |cached| {
        try cached.validate(original, train, cfg);
    }
    const selected_total = try validateChunks(original, chunks);
    const q = train.len / original.dim;
    const k = chunks.len;
    var max_m: usize = 0;
    for (chunks) |chunk| max_m = @max(max_m, chunk.selected_local.len);
    const qd = try mul(q, original.dim);
    if (fixed_original) |fixed| {
        const cached = source orelse return error.InvalidFitConfig;
        if (strategy != .identity_chunk_coefficients_v18 or chunks.len != 1 or fixed.start != chunks[0].start or fixed.rows != chunks[0].rows or
            fixed.query_count != q or fixed.dim != original.dim or fixed.original_values_ptr != original.values.ptr or fixed.train_ptr != train.ptr or
            !std.mem.eql(u8, &fixed.source_digest, &cached.digest) or !std.mem.eql(u8, &fixed.exact_core_digest, &exactCoreDigest(chunks[0].exact_core_local)) or fixed.rho.len != q or fixed.residual.len != qd) return error.InvalidFitConfig;
    }
    const qk = try mul(q, k);
    const batch_p = try mul(@min(q, 64), original.rows);
    const batch_y = try mul(@min(q, 64), original.dim);
    const output_values = try mul(selected_total, original.dim);
    const workspace = try workspaceBytesStrategy(original, q, chunks, cfg, source != null, strategy, retain_training_diagnostics);
    const phase1_bytes = workspace.phase1_bytes;
    const phase2_bytes = workspace.phase2_bytes;
    if (workspace.required > cfg.max_workspace_bytes or (source != null and workspace.required > source.?.consumer_peak_bytes)) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    if (source) |cached| cached.noteFitHit(counters);

    const original_y_owned = if (source == null) try allocator.alloc(f64, qd) else null;
    defer if (original_y_owned) |owned| allocator.free(owned);
    const original_y: []const f64 = if (source) |cached| cached.outputs else original_y_owned.?;
    const c = try allocator.alloc(f64, qk);
    defer allocator.free(c);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(c, 0);
    @memset(rhs, 0);
    const diag_a = if (strategy == .anchored_active_nnls_v17) try allocator.alloc(f64, k) else null;
    defer if (diag_a) |values| allocator.free(values);
    const diag_b = if (strategy == .anchored_active_nnls_v17) try allocator.alloc(f64, k) else null;
    defer if (diag_b) |values| allocator.free(values);
    const rho_diagnostic = if (strategy == .anchored_active_nnls_v17 and retain_training_diagnostics) try allocator.alloc(f64, qk) else null;
    defer if (rho_diagnostic) |values| allocator.free(values);
    if (diag_a) |values| @memset(values, 0);
    if (diag_b) |values| @memset(values, 0);
    const probabilities = if (source == null) try allocator.alloc(f64, batch_p) else null;
    defer if (probabilities) |owned| allocator.free(owned);
    const outputs = if (source == null) try allocator.alloc(f64, batch_y) else null;
    defer if (outputs) |owned| allocator.free(owned);
    var offset: usize = 0;
    while (offset < q) {
        try cfg.control.check();
        const take = @min(@as(usize, 64), q - offset);
        if (source == null) {
            try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], probabilities.?[0 .. take * original.rows], outputs.?[0 .. take * original.dim], null, cfg);
            @memcpy(original_y_owned.?[offset * original.dim ..][0 .. take * original.dim], outputs.?[0 .. take * original.dim]);
        }
        for (0..take) |bi| {
            const qi = offset + bi;
            for (chunks, 0..) |chunk, ci| {
                var rho: f64 = 0;
                var candidate: f64 = 0;
                for (0..chunk.rows) |local| rho += if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local];
                for (chunk.selected_local, chunk.weights) |local, weight| candidate += (if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local]) * @max(weight, cfg.min_weight);
                if (!std.math.isFinite(rho) or !std.math.isFinite(candidate) or rho < 0 or candidate < 0) return error.NonFiniteAttention;
                rhs[qi] += rho;
                c[qi * k + ci] = candidate;
                if (diag_a) |values| values[ci] += candidate * candidate;
                if (diag_b) |values| values[ci] += candidate * rho;
                if (rho_diagnostic) |values| values[qi * k + ci] = rho;
            }
        }
        offset += take;
    }
    const alpha = try allocator.alloc(f64, k);
    defer allocator.free(alpha);
    @memset(alpha, 0);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, k);
    defer allocator.free(norms);
    var mass_cfg = cfg;
    mass_cfg.max_workspace_bytes -= phase1_bytes;
    // Enforce alpha >= min_weight in the optimization itself rather than
    // clamping an unconstrained solution afterwards: alpha=min+beta, beta>=0.
    if (strategy == .global_active_nnls_v16 and cfg.min_weight > 0) for (0..q) |qi| {
        var row_sum: f64 = 0;
        for (c[qi * k ..][0..k]) |value| row_sum += value;
        rhs[qi] -= cfg.min_weight * row_sum;
        if (!std.math.isFinite(rhs[qi])) return error.NonFiniteAttention;
    };
    var training_diagnostics: ?anchored.Diagnostics = null;
    const sweeps = switch (strategy) {
        .global_active_nnls_v16 => block: {
            const value = try solver.fitMassActiveTargetDiagnostic(allocator, c, rhs, q, k, alpha, residual, norms, mass_cfg, false);
            for (alpha) |*entry| entry.* += cfg.min_weight;
            break :block value;
        },
        .anchored_active_nnls_v17 => block: {
            var solved = try anchored.fit(allocator, c, rhs, diag_a.?, diag_b.?, rho_diagnostic, mass_cfg);
            defer solved.deinit();
            @memcpy(alpha, solved.alpha);
            if (retain_training_diagnostics) training_diagnostics = solved.diagnostics;
            break :block solved.nnls_sweeps;
        },
        .identity_chunk_coefficients_v18 => block: {
            @memset(alpha, 1);
            break :block 0;
        },
    };

    const results = try allocator.alloc(ChunkResult, k);
    errdefer allocator.free(results);
    const values = try allocator.alloc(f64, output_values);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, selected_total);
    errdefer allocator.free(bias);
    var value_cursor: usize = 0;
    var bias_cursor: usize = 0;
    for (chunks, alpha, 0..) |chunk, chunk_alpha, ci| {
        const m = chunk.selected_local.len;
        if (chunk.exact_core_local.len == 0) {
            const basis = try allocator.alloc(f64, try mul(q, m));
            defer allocator.free(basis);
            const target = try allocator.alloc(f64, qd);
            defer allocator.free(target);
            offset = 0;
            while (offset < q) {
                try cfg.control.check();
                const take = @min(@as(usize, 64), q - offset);
                if (source == null) try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], probabilities.?[0 .. take * original.rows], null, null, cfg);
                for (0..take) |bi| {
                    const qi = offset + bi;
                    var rho: f64 = if (fixed_original) |fixed| fixed.rho[qi] else 0;
                    const out = target[qi * original.dim ..][0..original.dim];
                    if (fixed_original) |fixed| @memcpy(out, fixed.residual[qi * original.dim ..][0..original.dim]) else {
                        @memcpy(out, original_y[qi * original.dim ..][0..original.dim]);
                        for (0..chunk.rows) |local| {
                            const p = if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local];
                            rho += p;
                            for (out, original.values[(chunk.start + local) * original.dim ..][0..original.dim]) |*v, source_value| v.* -= p * source_value;
                        }
                    }
                    var d = 1 - rho;
                    for (chunk.selected_local, chunk.weights, 0..) |local, weight, j| {
                        const p = if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local];
                        const b = p * @max(weight, cfg.min_weight) * chunk_alpha;
                        basis[qi * m + j] = b;
                        d += b;
                    }
                    for (out, original_y[qi * original.dim ..][0..original.dim]) |*fixed, y| fixed.* = d * y - fixed.*;
                }
                offset += take;
            }
            var value_cfg = cfg;
            value_cfg.max_workspace_bytes -= phase2_bytes;
            const chunk_values = values[value_cursor..][0 .. m * original.dim];
            try solver.fitValuesWithBackendDiagnostic(allocator, basis, target, q, m, original.dim, cfg.ridge, value_cfg, chunk_values);
            for (chunk.selected_local, chunk.weights, bias[bias_cursor..][0..m]) |local, weight, *b| b.* = original.bias.?[chunk.start + local] + @log(@max(weight, cfg.min_weight) * chunk_alpha);
            results[ci] = .{ .start = chunk.start, .rows = chunk.rows, .value_offset = value_cursor, .selected_count = m, .alpha = chunk_alpha };
            value_cursor += m * original.dim;
            bias_cursor += m;
            continue;
        }
        const residual_indices = try allocator.alloc(usize, m);
        defer allocator.free(residual_indices);
        var residual_m: usize = 0;
        for (chunk.selected_local, 0..) |local, selected_index| if (!isExactCore(chunk, local)) {
            residual_indices[residual_m] = selected_index;
            residual_m += 1;
        };
        const basis = try allocator.alloc(f64, try mul(q, residual_m));
        defer allocator.free(basis);
        const target = try allocator.alloc(f64, qd);
        defer allocator.free(target);
        offset = 0;
        while (offset < q) {
            try cfg.control.check();
            const take = @min(@as(usize, 64), q - offset);
            if (source == null)
                try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], probabilities.?[0 .. take * original.rows], null, null, cfg);
            for (0..take) |bi| {
                const qi = offset + bi;
                var rho: f64 = if (fixed_original) |fixed| fixed.rho[qi] else 0;
                const out = target[qi * original.dim ..][0..original.dim];
                if (fixed_original) |fixed| @memcpy(out, fixed.residual[qi * original.dim ..][0..original.dim]) else {
                    @memcpy(out, original_y[qi * original.dim ..][0..original.dim]);
                    for (0..chunk.rows) |local| {
                        if (isExactCore(chunk, local)) continue;
                        const p = if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local];
                        rho += p;
                        for (out, original.values[(chunk.start + local) * original.dim ..][0..original.dim]) |*v, source_value| v.* -= p * source_value;
                    }
                }
                var d = 1 - rho;
                for (residual_indices[0..residual_m], 0..) |selected_index, j| {
                    const local = chunk.selected_local[selected_index];
                    const weight = chunk.weights[selected_index];
                    const p = if (source) |cached| try cached.probability(qi, chunk.start + local) else probabilities.?[bi * original.rows + chunk.start + local];
                    const b = p * @max(weight, cfg.min_weight) * chunk_alpha;
                    basis[qi * residual_m + j] = b;
                    d += b;
                }
                for (out, original_y[qi * original.dim ..][0..original.dim]) |*fixed, y| fixed.* = d * y - fixed.*;
            }
            offset += take;
        }
        var value_cfg = cfg;
        value_cfg.max_workspace_bytes -= phase2_bytes;
        const chunk_values = values[value_cursor..][0 .. m * original.dim];
        const residual_values = try allocator.alloc(f64, try mul(residual_m, original.dim));
        defer allocator.free(residual_values);
        if (residual_m != 0) try solver.fitValuesWithBackendDiagnostic(allocator, basis, target, q, residual_m, original.dim, cfg.ridge, value_cfg, residual_values);
        var residual_index: usize = 0;
        for (chunk.selected_local, chunk.weights, 0..) |local, weight, selected_index| {
            const out = chunk_values[selected_index * original.dim ..][0..original.dim];
            if (isExactCore(chunk, local)) {
                @memcpy(out, original.values[(chunk.start + local) * original.dim ..][0..original.dim]);
                bias[bias_cursor + selected_index] = original.bias.?[chunk.start + local];
            } else {
                @memcpy(out, residual_values[residual_index * original.dim ..][0..original.dim]);
                bias[bias_cursor + selected_index] = original.bias.?[chunk.start + local] + @log(@max(weight, cfg.min_weight) * chunk_alpha);
                residual_index += 1;
            }
        }
        if (residual_index != residual_m) return error.InvalidAttentionShape;
        results[ci] = .{ .start = chunk.start, .rows = chunk.rows, .value_offset = value_cursor, .selected_count = m, .alpha = chunk_alpha };
        value_cursor += m * original.dim;
        bias_cursor += m;
    }
    return .{ .allocator = allocator, .chunks = results, .values = values, .bias = bias, .nnls_sweeps = sweeps, .training_diagnostics = training_diagnostics };
}

test "v18 identity chunk coefficients bypass mass solve and preserve raw warm weights" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{
        .{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{2} },
        .{ .start = 2, .rows = 2, .selected_local = &.{1}, .weights = &.{3} },
    };
    var result = try fitStrategy(std.testing.allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 }, null, null, .identity_chunk_coefficients_v18, false);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.nnls_sweeps);
    for (result.chunks) |chunk| try std.testing.expectEqual(@as(f64, 1), chunk.alpha);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 2)), result.bias[0], 0);
    try std.testing.expectApproxEqAbs(@log(@as(f64, 3)), result.bias[1], 0);
}

test "v18 exact core preserves original value and bias while residual basis remains fitted" {
    const head: solver.Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 2, 5, 11 }, .bias = &.{ 0.25, -0.5, 0.75 }, .rows = 3, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 3, .selected_local = &.{ 0, 2 }, .weights = &.{ 1, 2 }, .exact_core_local = &.{0} }};
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 };
    var result = try fitStrategy(std.testing.allocator, head, &.{ -1, 1 }, &chunks, cfg, null, null, .identity_chunk_coefficients_v18, false);
    defer result.deinit();
    try std.testing.expectEqual(@as(f64, 2), result.values[0]);
    try std.testing.expectEqual(@as(f64, 0.25), result.bias[0]);
    try std.testing.expect(std.math.isFinite(result.values[1]));
    try std.testing.expect(std.math.isFinite(result.bias[1]));
    var candidate_probabilities: [4]f64 = undefined;
    var candidate_outputs: [2]f64 = undefined;
    const candidate: solver.Head = .{ .keys = &.{ -1, 1 }, .values = result.values, .bias = result.bias, .rows = 2, .dim = 1, .scale = 1 };
    try solver.attendBatch(std.testing.allocator, candidate, &.{ -1, 1 }, &candidate_probabilities, &candidate_outputs, null, cfg);
    try std.testing.expect(candidate_outputs[0] != candidate_outputs[1]);
    try std.testing.expect(candidate_probabilities[0] != candidate_probabilities[2]);
}

test "v18 exact core residual value matches independent one-column ridge objective and memo parity" {
    const head: solver.Head = .{ .keys = &.{ -2, -1, 1, 2 }, .values = &.{ 1, 3, 7, 13 }, .bias = &.{ 0.1, 0.2, -0.3, 0.4 }, .rows = 4, .dim = 1, .scale = 1 };
    const train = [_]f64{ -1, 1 };
    const chunks = [_]FrozenChunk{.{ .start = 1, .rows = 2, .selected_local = &.{ 0, 1 }, .weights = &.{ 1, 2 }, .exact_core_local = &.{0} }};
    const cfg: solver.Config = .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 0.125, .max_workspace_bytes = 1 << 20 };
    var numerator: f64 = 0;
    var denominator: f64 = cfg.ridge;
    for (train) |query| {
        var logits: [4]f64 = undefined;
        var maximum = -std.math.inf(f64);
        for (&logits, head.keys, head.bias.?) |*logit, key, bias_value| {
            logit.* = query * key + bias_value;
            maximum = @max(maximum, logit.*);
        }
        var probabilities: [4]f64 = undefined;
        var z: f64 = 0;
        for (&probabilities, logits) |*probability, logit| {
            probability.* = @exp(logit - maximum);
            z += probability.*;
        }
        var original_y: f64 = 0;
        for (&probabilities, head.values) |*probability, value| {
            probability.* /= z;
            original_y += probability.* * value;
        }
        const noncore_probability = probabilities[2];
        const fixed_original = original_y - noncore_probability * head.values[2];
        const basis = noncore_probability * 2;
        const d = 1 - noncore_probability + basis;
        const target = d * original_y - fixed_original;
        numerator += basis * target;
        denominator += basis * basis;
    }
    const expected_residual_value = numerator / denominator;
    var direct = try fitStrategy(std.testing.allocator, head, &train, &chunks, cfg, null, null, .identity_chunk_coefficients_v18, false);
    defer direct.deinit();
    try std.testing.expectEqual(@as(f64, 3), direct.values[0]);
    try std.testing.expectEqual(@as(f64, 0.2), direct.bias[0]);
    try std.testing.expectApproxEqAbs(expected_residual_value, direct.values[1], 2e-12);
    try std.testing.expectApproxEqAbs(-0.3 + @log(@as(f64, 2)), direct.bias[1], 2e-15);

    var counters: training_source.Counters = .{};
    const cached_workspace = (try workspaceBytesStrategy(head, train.len, &chunks, cfg, true, .identity_chunk_coefficients_v18, false)).required;
    const prepared = try training_source.prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = cached_workspace }, &counters);
    var source = prepared.ready;
    defer source.deinit();
    var memo = try prepareFixedOriginalTerm(std.testing.allocator, head, &train, chunks[0], cfg, &source, 0, 1 << 20);
    defer memo.deinit();
    var cached = try fitStrategyWithFixedOriginal(std.testing.allocator, head, &train, &chunks, cfg, &source, &counters, .identity_chunk_coefficients_v18, false, &memo);
    defer cached.deinit();
    try std.testing.expectEqualSlices(f64, direct.values, cached.values);
    try std.testing.expectEqualSlices(f64, direct.bias, cached.bias);
    var recomputed = try fitStrategyWithFixedOriginal(std.testing.allocator, head, &train, &chunks, cfg, &source, &counters, .identity_chunk_coefficients_v18, false, null);
    defer recomputed.deinit();
    try std.testing.expectEqualSlices(f64, cached.values, recomputed.values);
    try std.testing.expectEqualSlices(f64, cached.bias, recomputed.bias);
}

test "v18 core-only chunk copies source and never dispatches value fit" {
    const Counting = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: f64) !usize {
            return 0;
        }
        fn fit(_: ?*anyopaque, _: std.mem.Allocator, _: []const f64, _: []const f64, _: usize, _: usize, _: usize, _: f64, _: solver.Control, _: []f64) !void {
            calls += 1;
            return error.TestUnexpectedResult;
        }
    };
    Counting.calls = 0;
    const head: solver.Head = .{ .keys = &.{ -1, 1 }, .values = &.{ 2, 9 }, .bias = &.{ 0.25, -0.5 }, .rows = 2, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 1, .selected_local = &.{0}, .weights = &.{1}, .exact_core_local = &.{0} }};
    var result = try fitStrategy(std.testing.allocator, head, &.{ -1, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20, .value_backend = .{ .workspace_fn = Counting.workspace, .fit_fn = Counting.fit } }, null, null, .identity_chunk_coefficients_v18, false);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), Counting.calls);
    try std.testing.expectEqual(@as(f64, 2), result.values[0]);
    try std.testing.expectEqual(@as(f64, 0.25), result.bias[0]);
}

test "exact cores are identity-v18 only unique selected rows with unit weight" {
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 };
    const bad_weight = [_]FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{2}, .exact_core_local = &.{0} }};
    try std.testing.expectError(error.InvalidFitConfig, fitStrategy(std.testing.allocator, head, &.{ 0, 1 }, &bad_weight, cfg, null, null, .identity_chunk_coefficients_v18, false));
    const duplicate = [_]FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1}, .exact_core_local = &.{ 0, 0 } }};
    try std.testing.expectError(error.InvalidFitConfig, fitStrategy(std.testing.allocator, head, &.{ 0, 1 }, &duplicate, cfg, null, null, .identity_chunk_coefficients_v18, false));
    const valid = [_]FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1}, .exact_core_local = &.{0} }};
    try std.testing.expectError(error.InvalidFitConfig, fitStrategy(std.testing.allocator, head, &.{ 0, 1 }, &valid, cfg, null, null, .global_active_nnls_v16, false));
}

fn exactCoreAllocationFailure(allocator: std.mem.Allocator) !void {
    const head: solver.Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 2, 5, 11 }, .bias = &.{ 0.25, -0.5, 0.75 }, .rows = 3, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 3, .selected_local = &.{ 0, 2 }, .weights = &.{ 1, 2 }, .exact_core_local = &.{0} }};
    var result = try fitStrategy(allocator, head, &.{ -1, 1 }, &chunks, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 }, null, null, .identity_chunk_coefficients_v18, false);
    result.deinit();
}

test "exact core fit rejects its added workspace and unwinds allocation failures" {
    const head: solver.Head = .{ .keys = &.{ -1, 0, 1 }, .values = &.{ 2, 5, 11 }, .bias = &.{ 0.25, -0.5, 0.75 }, .rows = 3, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 3, .selected_local = &.{ 0, 2 }, .weights = &.{ 1, 2 }, .exact_core_local = &.{0} }};
    const required = (try workspaceBytesStrategy(head, 2, &chunks, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 }, false, .identity_chunk_coefficients_v18, false)).required;
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitStrategy(std.testing.allocator, head, &.{ -1, 1 }, &chunks, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = required - 1 }, null, null, .identity_chunk_coefficients_v18, false));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exactCoreAllocationFailure, .{});
}

test "core-aware workspace separates full output rows from residual fit rows" {
    const shape: solver.Head = .{ .keys = &.{}, .values = &.{}, .bias = &.{}, .rows = 256, .dim = 512, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 256, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 30 };
    const legacy = try workspaceBytesShape(shape, 768, 1, 256, 256, cfg, true, .identity_chunk_coefficients_v18, false);
    const core_aware = try workspaceBytesShapeCoreAware(shape, 768, 1, 256, 256, 128, cfg, true, .identity_chunk_coefficients_v18, false);
    try std.testing.expect(core_aware.required < legacy.required);
    const core_only = try workspaceBytesShapeCoreAware(shape, 768, 1, 256, 256, 0, cfg, true, .identity_chunk_coefficients_v18, false);
    try std.testing.expect(core_only.required < core_aware.required);
    try std.testing.expectError(error.InvalidAttentionShape, workspaceBytesShapeCoreAware(shape, 768, 1, 256, 128, 129, cfg, true, .identity_chunk_coefficients_v18, false));
}

test "empty-core workspace strategy remains byte-identical to legacy shape" {
    const shape: solver.Head = .{ .keys = &.{}, .values = &.{}, .bias = &.{}, .rows = 256, .dim = 512, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 128, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 30 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 256, .selected_local = &.{ 0, 1 }, .weights = &.{ 1, 1 } }};
    const expected = try workspaceBytesShape(shape, 768, 1, 2, 2, cfg, true, .identity_chunk_coefficients_v18, false);
    const actual = try workspaceBytesStrategy(shape, 768, &chunks, cfg, true, .identity_chunk_coefficients_v18, false);
    try std.testing.expectEqual(expected.required, actual.required);
    try std.testing.expectEqual(expected.phase1_bytes, actual.phase1_bytes);
    try std.testing.expectEqual(expected.phase2_bytes, actual.phase2_bytes);
}

fn v18AllocationFailure(allocator: std.mem.Allocator) !void {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} }};
    var result = try fitStrategy(allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 }, null, null, .identity_chunk_coefficients_v18, false);
    result.deinit();
}
test "v18 identity coefficient fit unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, v18AllocationFailure, .{});
}

test "joint chunk mass correction balances chunks then refits fixed-weight values" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{
        .{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} },
        .{ .start = 2, .rows = 2, .selected_local = &.{1}, .weights = &.{1} },
    };
    var result = try fit(std.testing.allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.chunks.len);
    // Each chunk has two identical keys but retains one basis row. Joint mass
    // correction must double both retained masses, then the fixed-weight local
    // value solve recovers each chunk's mean value.
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.chunks[0].alpha, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 2), result.chunks[1].alpha, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, 3), result.values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 7), result.values[1], 1e-6);
}

test "anchored v17 is explicit and retains training-only diagnostics" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{
        .{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} },
        .{ .start = 2, .rows = 2, .selected_local = &.{1}, .weights = &.{1} },
    };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 };
    var result = try fitStrategy(std.testing.allocator, head, &.{ 0, 1 }, &chunks, cfg, null, null, .anchored_active_nnls_v17, true);
    defer result.deinit();
    try std.testing.expect(result.training_diagnostics != null);
    try std.testing.expect(result.training_diagnostics.?.per_chunk_max_relative_before != null);
    try std.testing.expect(result.training_diagnostics.?.per_chunk_max_relative_after != null);
}

test "joint flat training source preserves exact fit bytes and records reuse" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    const chunks = [_]FrozenChunk{
        .{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} },
        .{ .start = 2, .rows = 2, .selected_local = &.{1}, .weights = &.{1} },
    };
    const cfg: solver.Config = .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .max_workspace_bytes = 1 << 20 };
    var legacy = try fit(std.testing.allocator, head, &train, &chunks, cfg);
    defer legacy.deinit();
    var counters: training_source.Counters = .{};
    const prepared = try training_source.prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = (try workspaceBytes(head, train.len, &chunks, cfg, true)).required }, &counters);
    var source = prepared.ready;
    defer source.deinit();
    var cached = try fitWithTrainingSource(std.testing.allocator, head, &train, &chunks, cfg, &source, &counters);
    defer cached.deinit();
    try std.testing.expectEqual(@as(usize, 1), counters.builds);
    try std.testing.expectEqual(@as(usize, 1), counters.fit_hits);
    try std.testing.expectEqual(@as(usize, 0), counters.misses);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(legacy.chunks), std.mem.sliceAsBytes(cached.chunks));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(legacy.values), std.mem.sliceAsBytes(cached.values));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(legacy.bias), std.mem.sliceAsBytes(cached.bias));
}

test "joint chunk lower bound permits negative shifted RHS with beta at zero" {
    const head: solver.Head = .{ .keys = &.{ 0, 0, 1, 1 }, .values = &.{ 2, 4, 6, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{
        .{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} },
        .{ .start = 2, .rows = 2, .selected_local = &.{1}, .weights = &.{1} },
    };
    var result = try fit(std.testing.allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .ridge = 1e-8, .min_weight = 3, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
    try std.testing.expectApproxEqAbs(@as(f64, 3), result.chunks[0].alpha, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 3), result.chunks[1].alpha, 1e-12);
}

test "joint chunk mass correction checks cap cancellation and allocation failures" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const chunks = [_]FrozenChunk{.{ .start = 0, .rows = 2, .selected_local = &.{0}, .weights = &.{1} }};
    try std.testing.expectError(error.CompactionWorkspaceLimit, fit(std.testing.allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 }));
    try std.testing.expectError(error.Cancelled, fit(std.testing.allocator, head, &.{ 0, 1 }, &chunks, .{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }));
    const AllocationProbe = struct {
        fn run(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, frozen: []const FrozenChunk, config: solver.Config) !void {
            var result = try fit(allocator, original, train, frozen, config);
            defer result.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationProbe.run, .{ head, &[_]f64{ 0, 1 }, &chunks, solver.Config{ .target_rows = 1, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 } });
}

test "fixed original term cap allocation and cancellation unwind" {
    const head: solver.Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 1, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    var probabilities = [_]f64{ 0.75, 0.25, 0.25, 0.75 };
    var outputs = [_]f64{ 1.5, 2.5 };
    var source: training_source.Source = .{ .allocator = std.testing.allocator, .keys_ptr = head.keys.ptr, .values_ptr = head.values.ptr, .bias_ptr = head.bias.?.ptr, .train_ptr = train[0..].ptr, .rows = 2, .dim = 1, .query_count = 2, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 16, .cache_bytes = 6 * @sizeOf(f64), .consumer_peak_bytes = 1 << 20, .digest = @splat(7), .probabilities = &probabilities, .outputs = &outputs };
    const chunk: FrozenChunk = .{ .start = 0, .rows = 2, .selected_local = &.{}, .weights = &.{} };
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, prepareFixedOriginalTerm(std.testing.failing_allocator, head, &train, chunk, cfg, &source, 0, 31));
    try std.testing.expectError(error.OutOfMemory, prepareFixedOriginalTerm(std.testing.failing_allocator, head, &train, chunk, cfg, &source, 0, 32));
    const Cancel = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 3) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    var cancel_cfg = cfg;
    cancel_cfg.control = .{ .context = &cancel, .check_fn = Cancel.check };
    try std.testing.expectError(error.Cancelled, prepareFixedOriginalTerm(std.testing.allocator, head, &train, chunk, cancel_cfg, &source, 0, 1 << 20));
}
