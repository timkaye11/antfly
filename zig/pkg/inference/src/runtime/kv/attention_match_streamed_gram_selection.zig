// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Default-off prototype for bounded raw residual-OMP selection from an
//! authenticated probability page source. This is a distinct numerical policy:
//! it accumulates normal equations in query order and does not claim the QR
//! reduction order of the legacy dense selector.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const NumericalPolicy = enum(u8) {
    normalized_probability_raw_v1,
    shifted_row_max_raw_v1,
};
pub const normalized_policy = "streamed_normalized_probability_f64_gram_raw_omp_lawson_hanson_v3";
pub const shifted_policy = "streamed_row_max_scaled_f64_gram_raw_omp_lawson_hanson_v5";
/// The bounded E2 route preserves the actual flat-v18 P/rho objective. The
/// shifted frontend remains an explicitly separate diagnostic policy.
pub const policy = normalized_policy;
pub const maximum_rows: usize = 256;
pub const maximum_target_rows: usize = 128;
pub const maximum_queries: usize = 65_536;

pub const ProbabilitySource = struct {
    context: ?*anyopaque,
    query_count: usize,
    row_count: usize,
    copy_query_fn: *const fn (?*anyopaque, usize, []f64, solver.Control) anyerror!void,
};

pub const Config = struct {
    target_rows: usize,
    numerical_policy: NumericalPolicy = .normalized_probability_raw_v1,
    query_batch_rows: usize = 64,
    max_sweeps: usize = 128,
    nnls_tolerance: f64 = 1e-8,
    external_live_bytes: usize,
    retained_scored_bytes: usize = 0,
    combined_cap: usize,
    control: solver.Control = .{},
};

pub const Evidence = struct {
    numerical_policy: NumericalPolicy,
    query_count: usize,
    row_count: usize,
    probability_values_consumed: usize,
    query_tiles: usize,
    workspace_bytes: usize,
    probability_sha256: [32]u8,
    row_maxima_sha256: [32]u8,
    maximum_probability: f64,
    maximum_frontend_scale: f64,
    nnls_sweeps: usize,
    maximum_selected_set_kkt_violation: f64,
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    indices: []usize,
    weights: []f64,
    evidence: Evidence,

    pub fn deinit(self: *Result) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const FullRouteBudget = struct {
    route_external_bytes: usize,
    retained_scored_bytes: usize,
    selector_workspace_bytes: usize,
    value_phase_bytes: usize,
    geometry_phase_bytes: usize,

    pub fn peak(self: FullRouteBudget) !usize {
        const retained = try add(self.route_external_bytes, self.retained_scored_bytes);
        const selection = try add(retained, self.selector_workspace_bytes);
        const value = try add(retained, self.value_phase_bytes);
        const geometry = try add(retained, self.geometry_phase_bytes);
        return @max(selection, @max(value, geometry));
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

pub fn workspaceBytes(query_count: usize, rows: usize, target_rows: usize, query_batch_rows: usize) !usize {
    if (query_count == 0 or query_count > maximum_queries or rows < 2 or rows > maximum_rows or
        target_rows == 0 or target_rows > maximum_target_rows or target_rows >= rows or query_batch_rows == 0)
        return error.InvalidFitConfig;
    const batch = @min(query_count, query_batch_rows);
    var bytes = try mul(try mul(batch, rows), @sizeOf(f64)); // probability page
    bytes = try add(bytes, try mul(try mul(rows, rows), @sizeOf(f64))); // full Gram
    bytes = try add(bytes, try mul(try mul(rows, 2), @sizeOf(f64))); // h and diagonal
    bytes = try add(bytes, try mul(try mul(target_rows, 3), @sizeOf(f64))); // weights, trial, gradients
    bytes = try add(bytes, try mul(try mul(target_rows, target_rows), @sizeOf(f64))); // selected Gram Cholesky
    bytes = try add(bytes, try mul(try mul(target_rows, 2), @sizeOf(usize))); // selected and passive working sets
    // Owned result is allocated before scratch is released.
    bytes = try add(bytes, try mul(target_rows, @sizeOf(usize) + @sizeOf(f64)));
    return bytes;
}

pub fn admissionBytes(config: Config, source: ProbabilitySource) !usize {
    if (!std.math.isFinite(config.nnls_tolerance) or config.nnls_tolerance <= 0 or config.max_sweeps == 0 or
        source.query_count == 0 or source.row_count == 0)
        return error.InvalidFitConfig;
    const owned = try workspaceBytes(source.query_count, source.row_count, config.target_rows, config.query_batch_rows);
    const retained = try add(config.external_live_bytes, config.retained_scored_bytes);
    return try add(retained, owned);
}

fn validate(config: Config, source: ProbabilitySource) !usize {
    const admitted = try admissionBytes(config, source);
    if (admitted > config.combined_cap) return error.CompactionWorkspaceLimit;
    return admitted - config.external_live_bytes - config.retained_scored_bytes;
}

fn selectedContains(selected: []const usize, value: usize) bool {
    return std.mem.indexOfScalar(usize, selected, value) != null;
}

fn hasNormalizedProgress(raw_score: f64, diagonal: f64, tolerance: f64) !bool {
    if (!std.math.isFinite(raw_score) or !std.math.isFinite(diagonal) or !std.math.isFinite(tolerance) or tolerance <= 0 or diagonal < 0) return error.NonFiniteAttention;
    if (diagonal == 0) return false;
    const score = raw_score / @sqrt(diagonal);
    if (!std.math.isFinite(score)) return error.NonFiniteAttention;
    return score > tolerance;
}

fn appendCholeskyPivot(gram: []const f64, rows: usize, selected: []const usize, factor: []f64, stride: usize) !void {
    const at = selected.len - 1;
    const candidate = selected[at];
    var squared_projection: f64 = 0;
    for (0..at) |j| {
        var value = gram[candidate * rows + selected[j]];
        for (0..j) |k| value -= factor[at * stride + k] * factor[j * stride + k];
        const diagonal = factor[j * stride + j];
        if (!std.math.isFinite(diagonal) or diagonal <= 0) return error.RankDeficientSelection;
        value /= diagonal;
        if (!std.math.isFinite(value)) return error.NonFiniteAttention;
        factor[at * stride + j] = value;
        squared_projection += value * value;
    }
    const diagonal_gram = gram[candidate * rows + candidate];
    const pivot = diagonal_gram - squared_projection;
    const threshold = 64 * std.math.floatEps(f64) * @max(diagonal_gram, 1);
    if (!std.math.isFinite(pivot) or pivot <= threshold) return error.RankDeficientSelection;
    factor[at * stride + at] = @sqrt(pivot);
}

fn solveSelectedGram(
    gram: []const f64,
    correlation: []const f64,
    rows: usize,
    query_count: usize,
    selected: []const usize,
    weights: []f64,
    gradients: []f64,
    factor: []f64,
    passive: []usize,
    config: Config,
) !struct { sweeps: usize, violation: f64 } {
    if (weights.len != selected.len or gradients.len != selected.len or passive.len < selected.len or factor.len < selected.len * selected.len) return error.InvalidAttentionShape;
    for (weights) |weight| if (!std.math.isFinite(weight) or weight < 0) return error.NonFiniteAttention;
    const tolerance = config.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(query_count)));
    var passive_len: usize = 0;
    for (weights, 0..) |weight, index| if (weight > 0) {
        passive[passive_len] = index;
        passive_len += 1;
    };
    var solves: usize = 0;
    while (true) {
        try config.control.check();
        var violation: f64 = 0;
        var entering: ?usize = null;
        var entering_gradient: f64 = -std.math.inf(f64);
        for (selected, 0..) |column, index| {
            const norm = @sqrt(gram[column * rows + column]);
            var gradient = correlation[column];
            for (selected, weights) |other, weight| gradient -= gram[column * rows + other] * weight;
            gradients[index] = if (norm > 0) gradient / norm else gradient;
            if (!std.math.isFinite(gradients[index])) return error.NonFiniteAttention;
            const passive_member = std.mem.indexOfScalar(usize, passive[0..passive_len], index) != null;
            violation = @max(violation, if (passive_member) @abs(gradients[index]) else @max(0, gradients[index]));
            if (!passive_member and gradients[index] > entering_gradient) {
                entering = index;
                entering_gradient = gradients[index];
            }
        }
        if (violation <= tolerance) return .{ .sweeps = solves, .violation = violation };
        if (solves == config.max_sweeps) break;
        const add_index = entering orelse return error.MassFitDidNotConverge;
        if (entering_gradient <= tolerance) return error.MassFitDidNotConverge;
        passive[passive_len] = add_index;
        passive_len += 1;
        while (true) {
            try config.control.check();
            if (solves == config.max_sweeps) return error.MassFitDidNotConverge;
            @memset(factor[0 .. selected.len * selected.len], 0);
            for (0..passive_len) |i| {
                const row = selected[passive[i]];
                var projection: f64 = 0;
                for (0..i) |j| {
                    var value = gram[row * rows + selected[passive[j]]];
                    for (0..j) |k| value -= factor[i * selected.len + k] * factor[j * selected.len + k];
                    value /= factor[j * selected.len + j];
                    if (!std.math.isFinite(value)) return error.NonFiniteAttention;
                    factor[i * selected.len + j] = value;
                    projection += value * value;
                }
                const diagonal_gram = gram[row * rows + row];
                const pivot = diagonal_gram - projection;
                if (!std.math.isFinite(pivot) or pivot <= 64 * std.math.floatEps(f64) * @max(diagonal_gram, 1)) return error.RankDeficientSelection;
                factor[i * selected.len + i] = @sqrt(pivot);
            }
            @memset(gradients, 0);
            for (0..passive_len) |i| {
                var rhs = correlation[selected[passive[i]]];
                for (0..i) |j| rhs -= factor[i * selected.len + j] * gradients[passive[j]];
                gradients[passive[i]] = rhs / factor[i * selected.len + i];
            }
            var reverse = passive_len;
            while (reverse > 0) {
                reverse -= 1;
                const index = passive[reverse];
                var rhs = gradients[index];
                for (reverse + 1..passive_len) |j| rhs -= factor[j * selected.len + reverse] * gradients[passive[j]];
                gradients[index] = rhs / factor[reverse * selected.len + reverse];
                if (!std.math.isFinite(gradients[index])) return error.NonFiniteAttention;
            }
            solves += 1;
            var alpha: f64 = 1;
            var all_positive = true;
            for (passive[0..passive_len]) |index| if (gradients[index] <= 0) {
                all_positive = false;
                const denominator = weights[index] - gradients[index];
                if (!std.math.isFinite(denominator) or denominator < 0) return error.NonFiniteAttention;
                if (denominator == 0) {
                    // A newly activated coordinate can have x=z=0 after the
                    // exact passive solve. Lawson-Hanson removes it at alpha0.
                    alpha = 0;
                } else alpha = @min(alpha, weights[index] / denominator);
            };
            if (all_positive) {
                @memcpy(weights, gradients);
                break;
            }
            if (!std.math.isFinite(alpha) or alpha < 0 or alpha > 1) return error.NonFiniteAttention;
            for (weights, gradients) |*weight, candidate| {
                weight.* += alpha * (candidate - weight.*);
                if (!std.math.isFinite(weight.*)) return error.NonFiniteAttention;
                if (weight.* <= config.nnls_tolerance) weight.* = 0;
            }
            var cursor: usize = 0;
            while (cursor < passive_len) {
                if (weights[passive[cursor]] == 0) {
                    passive_len -= 1;
                    std.mem.copyForwards(usize, passive[cursor..passive_len], passive[cursor + 1 .. passive_len + 1]);
                } else cursor += 1;
            }
        }
    }
    var final_violation: f64 = 0;
    for (selected, weights) |column, weight| {
        const norm = @sqrt(gram[column * rows + column]);
        var gradient = correlation[column];
        for (selected, weights) |other, other_weight| gradient -= gram[column * rows + other] * other_weight;
        if (!std.math.isFinite(gradient) or !std.math.isFinite(norm)) return error.NonFiniteAttention;
        const normalized = if (norm > 0) gradient / norm else gradient;
        final_violation = @max(final_violation, if (weight > 0) @abs(normalized) else @max(0, normalized));
    }
    std.debug.print("attention_match_streamed_gram_nnls phase=exhausted query_count={d} rows={d} selected_count={d} sweeps={d} final_violation={d} tolerance={d} diagnostic_only=true\n", .{ query_count, rows, selected.len, config.max_sweeps, final_violation, tolerance });
    return error.MassFitDidNotConverge;
}

fn validateNnlsWeights(weights: []const f64) !void {
    var positive = false;
    for (weights) |weight| {
        if (!std.math.isFinite(weight) or weight < 0) return error.NonFiniteAttention;
        positive = positive or weight > 0;
    }
    if (!positive) return error.InsufficientSelectionProgress;
}

fn sortSelectedWeights(selected: []usize, weights: []f64) !void {
    if (selected.len != weights.len) return error.InvalidAttentionShape;
    for (1..selected.len) |at| {
        var cursor = at;
        while (cursor > 0 and selected[cursor] < selected[cursor - 1]) : (cursor -= 1) {
            std.mem.swap(usize, &selected[cursor], &selected[cursor - 1]);
            std.mem.swap(f64, &weights[cursor], &weights[cursor - 1]);
        }
    }
}

pub fn select(allocator: std.mem.Allocator, source: ProbabilitySource, config: Config) !Result {
    const owned = try validate(config, source);
    try config.control.check();
    const rows = source.row_count;
    const batch = @min(source.query_count, config.query_batch_rows);
    const page = try allocator.alloc(f64, try mul(batch, rows));
    defer allocator.free(page);
    const gram = try allocator.alloc(f64, try mul(rows, rows));
    defer allocator.free(gram);
    const correlation = try allocator.alloc(f64, rows);
    defer allocator.free(correlation);
    const diagonal = try allocator.alloc(f64, rows);
    defer allocator.free(diagonal);
    const selected = try allocator.alloc(usize, config.target_rows);
    defer allocator.free(selected);
    const weights = try allocator.alloc(f64, config.target_rows);
    defer allocator.free(weights);
    const trial = try allocator.alloc(f64, config.target_rows);
    defer allocator.free(trial);
    const gradients = try allocator.alloc(f64, config.target_rows);
    defer allocator.free(gradients);
    const cholesky = try allocator.alloc(f64, try mul(config.target_rows, config.target_rows));
    defer allocator.free(cholesky);
    const passive = try allocator.alloc(usize, config.target_rows);
    defer allocator.free(passive);
    @memset(gram, 0);
    @memset(correlation, 0);
    @memset(diagonal, 0);
    @memset(weights, 0);
    @memset(trial, 0);
    @memset(cholesky, 0);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var maxima_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var hash_bytes: [8]u8 = undefined;
    var consumed: usize = 0;
    var tiles: usize = 0;
    var maximum_probability: f64 = 0;
    var maximum_frontend_scale: f64 = 0;
    var first: usize = 0;
    while (first < source.query_count) {
        try config.control.check();
        const count = @min(batch, source.query_count - first);
        for (0..count) |local| {
            const row = page[local * rows ..][0..rows];
            try source.copy_query_fn(source.context, first + local, row, config.control);
            var rho: f64 = 0;
            var maximum: f64 = 0;
            for (row) |value| {
                if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                rho += value;
                maximum = @max(maximum, value);
                std.mem.writeInt(u64, &hash_bytes, @bitCast(value), .little);
                hash.update(&hash_bytes);
            }
            if (!std.math.isFinite(rho) or rho < 0) return error.NonFiniteAttention;
            const frontend_scale: f64 = switch (config.numerical_policy) {
                .normalized_probability_raw_v1 => 1,
                .shifted_row_max_raw_v1 => if (maximum > 0) 1 / maximum else return error.NonFiniteAttention,
            };
            if (!std.math.isFinite(frontend_scale) or frontend_scale <= 0) return error.NonFiniteAttention;
            maximum_probability = @max(maximum_probability, maximum);
            maximum_frontend_scale = @max(maximum_frontend_scale, frontend_scale);
            std.mem.writeInt(u64, &hash_bytes, @bitCast(maximum), .little);
            maxima_hash.update(&hash_bytes);
            const scaled_rho = rho * frontend_scale;
            if (!std.math.isFinite(scaled_rho) or
                (config.numerical_policy == .shifted_row_max_raw_v1 and scaled_rho <= 0))
                return error.NonFiniteAttention;
            for (row, 0..) |left, i| {
                const scaled_left = left * frontend_scale;
                correlation[i] += scaled_left * scaled_rho;
                for (row, 0..) |right, j| gram[i * rows + j] += scaled_left * (right * frontend_scale);
            }
            consumed = try add(consumed, rows);
        }
        tiles = try add(tiles, 1);
        first += count;
    }
    if (consumed != try mul(source.query_count, rows)) return error.InvalidAttentionShape;
    for (0..rows) |i| {
        diagonal[i] = gram[i * rows + i];
        if (!std.math.isFinite(diagonal[i]) or diagonal[i] < 0 or !std.math.isFinite(correlation[i])) return error.NonFiniteAttention;
    }
    var total_sweeps: usize = 0;
    var maximum_violation: f64 = 0;
    for (0..config.target_rows) |at| {
        try config.control.check();
        var best: ?usize = null;
        var best_score: f64 = -std.math.inf(f64);
        for (0..rows) |candidate| {
            if (selectedContains(selected[0..at], candidate)) continue;
            var score = correlation[candidate];
            for (selected[0..at], weights[0..at]) |chosen, weight| score -= gram[candidate * rows + chosen] * weight;
            if (!std.math.isFinite(score)) return error.NonFiniteAttention;
            if (best == null or score > best_score) {
                best = candidate;
                best_score = score;
            }
        }
        const progress_tolerance = config.nnls_tolerance * @sqrt(@as(f64, @floatFromInt(source.query_count)));
        const chosen = best orelse return error.InsufficientSelectionProgress;
        if (!try hasNormalizedProgress(best_score, diagonal[chosen], progress_tolerance)) {
            std.debug.print("attention_match_streamed_gram_selection phase=no_progress query_count={d} rows={d} target_rows={d} selected_count={d} chosen={d} raw_score={d} diagonal={d} tolerance={d} diagnostic_only=true\n", .{ source.query_count, rows, config.target_rows, at, chosen, best_score, diagonal[chosen], progress_tolerance });
            return error.InsufficientSelectionProgress;
        }
        selected[at] = chosen;
        // solveSelectedGram reuses this buffer for a packed passive factor.
        // Rebuild the full selected-prefix rank certificate from scratch on
        // every OMP append before handing the storage back to the NNLS solve.
        @memset(cholesky, 0);
        for (1..at + 2) |prefix|
            try appendCholeskyPivot(gram, rows, selected[0..prefix], cholesky, config.target_rows);
        // The selected set grows by one column. Preserve the exact converged
        // solution for the common prefix and warm only the new coordinate at
        // zero. Restarting all coordinates discarded useful KKT progress and
        // made the fixed 128-sweep bound depend on selected-set size.
        @memcpy(trial[0..at], weights[0..at]);
        trial[at] = 0;
        const solved = try solveSelectedGram(gram, correlation, rows, source.query_count, selected[0 .. at + 1], trial[0 .. at + 1], gradients[0 .. at + 1], cholesky, passive, config);
        @memcpy(weights[0 .. at + 1], trial[0 .. at + 1]);
        total_sweeps = try add(total_sweeps, solved.sweeps);
        maximum_violation = @max(maximum_violation, solved.violation);
    }
    // Canonical output order matches the legacy selector. Re-solve in that
    // order so weights remain paired with indices.
    try sortSelectedWeights(selected, weights);
    @memcpy(trial, weights);
    const final = try solveSelectedGram(gram, correlation, rows, source.query_count, selected, trial, gradients, cholesky, passive, config);
    total_sweeps = try add(total_sweeps, final.sweeps);
    maximum_violation = @max(maximum_violation, final.violation);
    // NNLS permits inactive selected columns. The joint value fitter already
    // applies cfg.min_weight when materializing their finite log-bias; retain
    // the exact nonnegative solution and the fixed selected-row budget here.
    try validateNnlsWeights(trial);
    const out_indices = try allocator.dupe(usize, selected);
    errdefer allocator.free(out_indices);
    const out_weights = try allocator.dupe(f64, trial);
    errdefer allocator.free(out_weights);
    try config.control.check();
    return .{ .allocator = allocator, .indices = out_indices, .weights = out_weights, .evidence = .{
        .numerical_policy = config.numerical_policy,
        .query_count = source.query_count,
        .row_count = rows,
        .probability_values_consumed = consumed,
        .query_tiles = tiles,
        .workspace_bytes = owned,
        .probability_sha256 = hash.finalResult(),
        .row_maxima_sha256 = maxima_hash.finalResult(),
        .maximum_probability = maximum_probability,
        .maximum_frontend_scale = maximum_frontend_scale,
        .nnls_sweeps = total_sweeps,
        .maximum_selected_set_kkt_violation = maximum_violation,
    } };
}

const FixtureSource = struct {
    probabilities: []const f64,
    rows: usize,
    calls: usize = 0,
    fn copy(raw: ?*anyopaque, query: usize, out: []f64, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        try control.check();
        if (out.len != self.rows or query >= self.probabilities.len / self.rows) return error.InvalidAttentionShape;
        @memcpy(out, self.probabilities[query * self.rows ..][0..self.rows]);
        self.calls += 1;
    }
    fn source(self: *@This()) ProbabilitySource {
        return .{ .context = self, .query_count = self.probabilities.len / self.rows, .row_count = self.rows, .copy_query_fn = copy };
    }
};

test "selected Gram Lawson Hanson solves correlated basis within sealed passive bound" {
    const gram = [_]f64{ 1, 0.999, 0.999, 1 };
    const correlation = [_]f64{ 1, 1 };
    const selected = [_]usize{ 0, 1 };
    var gradients: [2]f64 = undefined;
    var factor: [4]f64 = undefined;
    var passive: [2]usize = undefined;
    const cfg: Config = .{
        .target_rows = 2,
        .max_sweeps = 2,
        .nnls_tolerance = 1e-8,
        .external_live_bytes = 0,
        .combined_cap = 1 << 20,
    };
    var warm: [2]f64 = @splat(0);
    const solved = try solveSelectedGram(&gram, &correlation, 2, 1, &selected, &warm, &gradients, &factor, &passive, cfg);
    try std.testing.expectEqual(@as(usize, 2), solved.sweeps);
    try std.testing.expect(solved.violation <= cfg.nnls_tolerance);
    try std.testing.expect(warm[0] > 0 and warm[1] > 0);
}

test "streamed selector exercises bounded active set on three correlated columns" {
    const p = [_]f64{
        1.000000, 0.999000, 0.998000, 0.12,
        0.998000, 1.000000, 0.999000, 0.11,
        0.999000, 0.998000, 1.000000, 0.13,
        1.000000, 0.998500, 0.999500, 0.10,
        0.998500, 1.000000, 0.998500, 0.14,
        0.999500, 0.998500, 1.000000, 0.09,
    };
    var fixture: FixtureSource = .{ .probabilities = &p, .rows = 4 };
    var gram: [16]f64 = @splat(0);
    var correlation: [4]f64 = @splat(0);
    for (0..6) |query| {
        const row = p[query * 4 ..][0..4];
        var rho: f64 = 0;
        for (row) |value| rho += value;
        for (row, 0..) |left, i| {
            correlation[i] += left * rho;
            for (row, 0..) |right, j| gram[i * 4 + j] += left * right;
        }
    }
    // This is the former cold cyclic-coordinate solve on the actual OMP order.
    // It remains far outside the unchanged KKT tolerance after all 128 sweeps.
    const legacy_selected = [_]usize{ 0, 3, 1 };
    var legacy_weights: [3]f64 = @splat(0);
    for (0..128) |_| for (legacy_selected, 0..) |column, j| {
        var gradient = correlation[column];
        for (legacy_selected, legacy_weights) |other, weight| gradient -= gram[column * 4 + other] * weight;
        legacy_weights[j] = @max(0, legacy_weights[j] + gradient / gram[column * 4 + column]);
    };
    var legacy_violation: f64 = 0;
    for (legacy_selected, legacy_weights) |column, weight| {
        var gradient = correlation[column];
        for (legacy_selected, legacy_weights) |other, other_weight| gradient -= gram[column * 4 + other] * other_weight;
        const normalized = gradient / @sqrt(gram[column * 4 + column]);
        legacy_violation = @max(legacy_violation, if (weight > 0) @abs(normalized) else @max(0, normalized));
    }
    const cfg: Config = .{ .target_rows = 3, .query_batch_rows = 2, .max_sweeps = 128, .nnls_tolerance = 1e-10, .external_live_bytes = 0, .combined_cap = 2 * 1024 * 1024 };
    try std.testing.expect(legacy_violation > cfg.nnls_tolerance * @sqrt(@as(f64, 6)));
    var result = try select(std.testing.allocator, fixture.source(), cfg);
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 6), fixture.calls);
    try std.testing.expectEqual(@as(usize, 3), result.indices.len);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, result.indices);
    try std.testing.expect(result.evidence.nnls_sweeps <= cfg.max_sweeps * (cfg.target_rows + 1));
    try std.testing.expect(result.evidence.maximum_selected_set_kkt_violation <= cfg.nnls_tolerance * @sqrt(@as(f64, 6)));
    for (result.weights) |weight| try std.testing.expect(std.math.isFinite(weight) and weight > 0);
}

test "streamed Gram selector preserves valid zero NNLS coefficients" {
    var probabilities: [20]f64 = undefined;
    probabilities[0] = 0.5;
    probabilities[1] = 0.125;
    @memset(probabilities[2..10], 0.046875);
    probabilities[10] = 0.75;
    probabilities[11] = 0.125;
    @memset(probabilities[12..20], 0.015625);
    var fixture: FixtureSource = .{ .probabilities = &probabilities, .rows = 10 };
    const cfg: Config = .{ .target_rows = 2, .external_live_bytes = 0, .combined_cap = 1 << 20 };
    var result = try select(std.testing.allocator, fixture.source(), cfg);
    defer result.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, result.indices);
    try std.testing.expectApproxEqAbs(@as(f64, 0), result.weights[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 8), result.weights[1], 1e-12);
    try std.testing.expect(result.evidence.maximum_selected_set_kkt_violation <= cfg.nnls_tolerance * @sqrt(@as(f64, 2)));
    try std.testing.expectEqual(@as(usize, 2), fixture.calls);
    for (0..2) |query| {
        var mass: f64 = 0;
        for (result.indices, result.weights) |index, weight| mass += probabilities[query * 10 + index] * weight;
        try std.testing.expectApproxEqAbs(@as(f64, 1), mass, 1e-12);
    }
}

test "selected NNLS publication accepts zero and rejects negative nonfinite and empty mass" {
    try validateNnlsWeights(&.{ 0, 1 });
    try std.testing.expectError(error.NonFiniteAttention, validateNnlsWeights(&.{ -1e-12, 1 }));
    try std.testing.expectError(error.NonFiniteAttention, validateNnlsWeights(&.{ std.math.nan(f64), 1 }));
    try std.testing.expectError(error.NonFiniteAttention, validateNnlsWeights(&.{ std.math.inf(f64), 1 }));
    try std.testing.expectError(error.InsufficientSelectionProgress, validateNnlsWeights(&.{ 0, 0 }));
}

test "canonical selected order keeps warm weights paired" {
    var selected = [_]usize{ 9, 2, 7, 1 };
    var weights = [_]f64{ 90, 20, 70, 10 };
    try sortSelectedWeights(&selected, &weights);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 7, 9 }, &selected);
    try std.testing.expectEqualSlices(f64, &.{ 10, 20, 70, 90 }, &weights);
}

test "streamed Gram raw selector agrees with small dense oracle" {
    const dense_selection = @import("attention_match_global_selection_diagnostic.zig");
    const p = [_]f64{
        0.00, 0.00, 0.00, 0.00,
        0.70, 0.20, 0.10, 0.00,
        0.10, 0.65, 0.20, 0.05,
        0.05, 0.10, 0.75, 0.10,
        0.20, 0.05, 0.10, 0.65,
        0.55, 0.15, 0.20, 0.10,
    };
    var fixture: FixtureSource = .{ .probabilities = &p, .rows = 4 };
    const cfg: Config = .{ .target_rows = 2, .query_batch_rows = 2, .external_live_bytes = 17, .combined_cap = 2 * 1024 * 1024 };
    var actual = try select(std.testing.allocator, fixture.source(), cfg);
    defer actual.deinit();
    const head: solver.Head = .{ .keys = &.{ 1, 2, 3, 4 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    // This is the actual legacy-flat frontend: full-head normalized P is
    // cropped unchanged and rho is the crop sum. `.raw` describes the OMP
    // column score, not a per-query P/max(P) transform.
    var expected = try dense_selection.selectFromProbabilities(std.testing.allocator, head, 0, 4, &p, .{ .target_rows = 2, .selection = .residual_omp, .mass_solver = .active_set, .max_workspace_bytes = 2 * 1024 * 1024 });
    defer expected.deinit();
    try std.testing.expectEqualSlices(usize, expected.indices, actual.indices);
    for (expected.weights, actual.weights) |want, got| try std.testing.expectApproxEqRel(want, got, 2e-7);
    try std.testing.expectEqual(@as(usize, p.len), actual.evidence.probability_values_consumed);
    try std.testing.expectEqual(@as(usize, 3), actual.evidence.query_tiles);
    try std.testing.expectEqual(NumericalPolicy.normalized_probability_raw_v1, actual.evidence.numerical_policy);
}

test "streamed row-max frontend is invariant to tiny global probability scale" {
    const p = [_]f64{ 0.7, 0.2, 0.1, 0.1, 0.65, 0.25, 0.2, 0.1, 0.7 };
    var tiny: [p.len]f64 = undefined;
    for (p, &tiny) |value, *out| out.* = value * 1e-8;
    var ordinary_source: FixtureSource = .{ .probabilities = &p, .rows = 3 };
    var tiny_source: FixtureSource = .{ .probabilities = &tiny, .rows = 3 };
    const cfg: Config = .{ .target_rows = 2, .numerical_policy = .shifted_row_max_raw_v1, .query_batch_rows = 2, .max_sweeps = 512, .external_live_bytes = 0, .combined_cap = 1 << 20 };
    var ordinary = try select(std.testing.allocator, ordinary_source.source(), cfg);
    defer ordinary.deinit();
    var scaled = try select(std.testing.allocator, tiny_source.source(), cfg);
    defer scaled.deinit();
    try std.testing.expectEqualSlices(usize, ordinary.indices, scaled.indices);
    for (ordinary.weights, scaled.weights) |want, got| try std.testing.expectApproxEqRel(want, got, 2e-12);
    try std.testing.expect(!std.mem.eql(u8, &ordinary.evidence.probability_sha256, &scaled.evidence.probability_sha256));
    try std.testing.expect(!std.mem.eql(u8, &ordinary.evidence.row_maxima_sha256, &scaled.evidence.row_maxima_sha256));
    try std.testing.expectApproxEqRel(ordinary.evidence.maximum_frontend_scale * 1e8, scaled.evidence.maximum_frontend_scale, 2e-15);
    try std.testing.expectEqual(NumericalPolicy.shifted_row_max_raw_v1, ordinary.evidence.numerical_policy);
}

test "normalized probability policy keeps absolute scale and fails tiny progress closed" {
    const p = [_]f64{ 0.7e-8, 0.2e-8, 0.1e-8, 0.1e-8, 0.65e-8, 0.25e-8 };
    var tiny: FixtureSource = .{ .probabilities = &p, .rows = 3 };
    try std.testing.expectError(error.InsufficientSelectionProgress, select(std.testing.allocator, tiny.source(), .{ .target_rows = 2, .numerical_policy = .normalized_probability_raw_v1, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
}

test "normalized progress admits a valid tiny third raw correlation and matches dense native selection" {
    const p = [_]f64{
        0.8, 0.2, 2e-4, 0,
        0.7, 0.3, 0,    0,
        0.6, 0.4, 0,    0,
    };
    // The public selector requires target_rows < rows; the fourth zero column
    // is a control. The first three columns are the requested q3/rows3 case.
    var source: FixtureSource = .{ .probabilities = &p, .rows = 4 };
    const cfg: Config = .{ .target_rows = 3, .query_batch_rows = 2, .max_sweeps = 512, .external_live_bytes = 0, .combined_cap = 2 * 1024 * 1024 };
    var actual = try select(std.testing.allocator, source.source(), cfg);
    defer actual.deinit();
    const dense_selection = @import("attention_match_global_selection_diagnostic.zig");
    const head: solver.Head = .{ .keys = &.{ 1, 2, 3, 4 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    var expected = try dense_selection.selectFromProbabilities(std.testing.allocator, head, 0, 4, &p, .{ .target_rows = 3, .selection = .residual_omp, .mass_solver = .active_set, .max_sweeps = 512, .max_workspace_bytes = 2 * 1024 * 1024 });
    defer expected.deinit();
    try std.testing.expectEqualSlices(usize, expected.indices, actual.indices);
    var actual_prediction: [3]f64 = @splat(0);
    var expected_prediction: [3]f64 = @splat(0);
    for (0..3) |query| {
        for (actual.indices, actual.weights) |index, weight| {
            try std.testing.expect(std.math.isFinite(weight) and weight >= 0);
            actual_prediction[query] += p[query * 4 + index] * weight;
        }
        for (expected.indices, expected.weights) |index, weight| {
            try std.testing.expect(std.math.isFinite(weight) and weight >= 0);
            expected_prediction[query] += p[query * 4 + index] * weight;
        }
    }
    const tolerance = cfg.nnls_tolerance * @sqrt(@as(f64, 3));
    var maximum_prediction_delta: f64 = 0;
    for (actual_prediction, expected_prediction) |actual_mass, expected_mass| {
        maximum_prediction_delta = @max(maximum_prediction_delta, @abs(actual_mass - expected_mass));
    }
    // Both solutions satisfy normalized KKT in the same full-rank, square
    // selected basis B. With e=B^T r, r=B^-T e, so their prediction
    // difference is bounded by 2*tol*||B^-T||_inf (triangle inequality).
    var normalized_basis: [9]f64 = undefined;
    for (0..3) |selected_column| {
        const source_column = actual.indices[selected_column];
        var norm_sq: f64 = 0;
        for (0..3) |query| norm_sq += p[query * 4 + source_column] * p[query * 4 + source_column];
        const norm = @sqrt(norm_sq);
        try std.testing.expect(std.math.isFinite(norm) and norm > 0);
        for (0..3) |query| normalized_basis[query * 3 + selected_column] = p[query * 4 + source_column] / norm;
    }
    const a = normalized_basis[0];
    const b = normalized_basis[1];
    const c = normalized_basis[2];
    const d = normalized_basis[3];
    const e = normalized_basis[4];
    const f = normalized_basis[5];
    const g = normalized_basis[6];
    const h = normalized_basis[7];
    const i = normalized_basis[8];
    const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    try std.testing.expect(std.math.isFinite(determinant) and determinant != 0);
    const inverse = [9]f64{
        (e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant,
        (f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant,
        (d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant,
    };
    const inverse_transpose_inf_norm = @max(
        @abs(inverse[0]) + @abs(inverse[3]) + @abs(inverse[6]),
        @max(@abs(inverse[1]) + @abs(inverse[4]) + @abs(inverse[7]), @abs(inverse[2]) + @abs(inverse[5]) + @abs(inverse[8])),
    );
    const prediction_roundoff = 64.0 * std.math.floatEps(f64) * inverse_transpose_inf_norm;
    const conditioned_prediction_bound = 2.0 * tolerance * inverse_transpose_inf_norm + prediction_roundoff;
    try std.testing.expectApproxEqAbs(@as(f64, 0), maximum_prediction_delta, conditioned_prediction_bound);
    for (0..4) |column| {
        var diagonal: f64 = 0;
        var actual_gradient: f64 = 0;
        var expected_gradient: f64 = 0;
        for (0..3) |query| {
            const probability = p[query * 4 + column];
            var rho: f64 = 0;
            for (0..4) |source_column| rho += p[query * 4 + source_column];
            diagonal += probability * probability;
            actual_gradient += probability * (rho - actual_prediction[query]);
            expected_gradient += probability * (rho - expected_prediction[query]);
        }
        if (diagonal > 0) {
            const actual_selected = std.mem.indexOfScalar(usize, actual.indices, column);
            const expected_selected = std.mem.indexOfScalar(usize, expected.indices, column);
            const actual_violation = if (actual_selected != null and actual.weights[actual_selected.?] > 0) @abs(actual_gradient) else @max(0, actual_gradient);
            const expected_violation = if (expected_selected != null and expected.weights[expected_selected.?] > 0) @abs(expected_gradient) else @max(0, expected_gradient);
            try std.testing.expect(actual_violation / @sqrt(diagonal) <= tolerance);
            try std.testing.expect(expected_violation / @sqrt(diagonal) <= tolerance);
        }
    }
    // Direct last-step units regression: for epsilon=2e-4 the residual
    // correlation is epsilon^2/6, below tol*sqrt(3), while division by the
    // epsilon column norm exposes real progress.
    const epsilon = 2e-4;
    const raw_residual_correlation = epsilon * epsilon / 6.0;
    try std.testing.expect(raw_residual_correlation < tolerance);
    try std.testing.expect(try hasNormalizedProgress(raw_residual_correlation, epsilon * epsilon, tolerance));
}

test "progress normalization does not change raw candidate ranking" {
    const p = [_]f64{
        0.9, 0.1, 0,
        0,   0.1, 0.9,
    };
    var source: FixtureSource = .{ .probabilities = &p, .rows = 3 };
    var result = try select(std.testing.allocator, source.source(), .{ .target_rows = 1, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    defer result.deinit();
    // Raw scores are .9, .2, .9, so stable raw ranking chooses index zero.
    // Norm-ranking would choose the smaller-norm middle column instead.
    try std.testing.expectEqualSlices(usize, &.{0}, result.indices);
}

test "streamed Gram selector consumes tail tile and consistently rejects zero and invalid inputs" {
    var fixture: FixtureSource = .{ .probabilities = &.{
        0.8, 0.2, 0, 0.7, 0.3, 0, 0.6, 0.4, 0, 0.5, 0.5, 0, 0.4, 0.6, 0,
    }, .rows = 3 };
    var result = try select(std.testing.allocator, fixture.source(), .{ .target_rows = 1, .query_batch_rows = 2, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 5), fixture.calls);
    try std.testing.expectEqual(@as(usize, 3), result.evidence.query_tiles);
    var negative: FixtureSource = .{ .probabilities = &.{ 0.5, -0.5, 0.25, 0.75 }, .rows = 2 };
    try std.testing.expectError(error.NonFiniteAttention, select(std.testing.allocator, negative.source(), .{ .target_rows = 1, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
    var nonfinite: FixtureSource = .{ .probabilities = &.{ 0.5, std.math.nan(f64), 0.25, 0.75 }, .rows = 2 };
    try std.testing.expectError(error.NonFiniteAttention, select(std.testing.allocator, nonfinite.source(), .{ .target_rows = 1, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
    var zero: FixtureSource = .{ .probabilities = &.{ 0, 0, 0, 0 }, .rows = 2 };
    try std.testing.expectError(error.InsufficientSelectionProgress, select(std.testing.allocator, zero.source(), .{ .target_rows = 1, .numerical_policy = .normalized_probability_raw_v1, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
    try std.testing.expectError(error.NonFiniteAttention, select(std.testing.allocator, zero.source(), .{ .target_rows = 1, .numerical_policy = .shifted_row_max_raw_v1, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
}

test "streamed duplicate positive columns fail rank and progress checks" {
    const duplicate_gram = [_]f64{ 2, 2, 2, 2 };
    var factor = [_]f64{ 0, 0, 0, 0 };
    try appendCholeskyPivot(&duplicate_gram, 2, &.{0}, &factor, 2);
    try std.testing.expectError(error.RankDeficientSelection, appendCholeskyPivot(&duplicate_gram, 2, &.{ 0, 1 }, &factor, 2));
    var duplicate: FixtureSource = .{ .probabilities = &.{ 0.5, 0.5, 0, 0, 0.25, 0.25, 0.5, 0 }, .rows = 4 };
    try std.testing.expectError(error.InsufficientSelectionProgress, select(std.testing.allocator, duplicate.source(), .{ .target_rows = 3, .external_live_bytes = 0, .combined_cap = 1 << 20 }));
}

fn allocationFailureFixture(allocator: std.mem.Allocator) !void {
    var fixture: FixtureSource = .{ .probabilities = &.{ 0.8, 0.2, 0.2, 0.8 }, .rows = 2 };
    var result = try select(allocator, fixture.source(), .{ .target_rows = 1, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    result.deinit();
}

test "streamed Gram selector cap cancel and allocation failures publish nothing" {
    var fixture: FixtureSource = .{ .probabilities = &.{ 0.8, 0.2, 0.2, 0.8 }, .rows = 2 };
    const needed = try workspaceBytes(2, 2, 1, 64);
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.allocator, fixture.source(), .{ .target_rows = 1, .external_live_bytes = 9, .retained_scored_bytes = 77, .combined_cap = 9 + 77 + needed - 1 }));
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 5) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, fixture.source(), .{ .target_rows = 1, .query_batch_rows = 1, .external_live_bytes = 0, .combined_cap = 1 << 20, .control = .{ .context = &cancel, .check_fn = Cancel.check } }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureFixture, .{});
}

test "streamed Gram selector is deterministic on an ill-conditioned fixture" {
    var fixture: FixtureSource = .{ .probabilities = &.{
        0.5000000001, 0.5, 0, 0.4999999999, 0.5, 0, 0.25, 0.2500000001, 0.4999999999,
    }, .rows = 3 };
    var first = try select(std.testing.allocator, fixture.source(), .{ .target_rows = 2, .query_batch_rows = 2, .max_sweeps = 512, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    defer first.deinit();
    fixture.calls = 0;
    var second = try select(std.testing.allocator, fixture.source(), .{ .target_rows = 2, .query_batch_rows = 2, .max_sweeps = 512, .external_live_bytes = 0, .combined_cap = 1 << 20 });
    defer second.deinit();
    try std.testing.expectEqualSlices(usize, first.indices, second.indices);
    try std.testing.expectEqualSlices(f64, first.weights, second.weights);
    try std.testing.expectEqual(first.evidence.probability_sha256, second.evidence.probability_sha256);
}

test "streamed ill-conditioned Gram publishes at exact two-passive-solve budget" {
    var fixture: FixtureSource = .{ .probabilities = &.{
        1.0,   0.999, 0.001,
        0.999, 1.0,   0.001,
        0.5,   0.499, 0.501,
        0.499, 0.5,   0.501,
    }, .rows = 3 };
    var result = try select(std.testing.allocator, fixture.source(), .{
        .target_rows = 2,
        .query_batch_rows = 2,
        .max_sweeps = 2,
        .nnls_tolerance = 1e-16,
        .external_live_bytes = 0,
        .combined_cap = 1 << 20,
    });
    defer result.deinit();
    try std.testing.expect(result.evidence.maximum_selected_set_kkt_violation <= 1e-16 * 2);
    for (result.weights) |weight| try std.testing.expect(std.math.isFinite(weight) and weight > 0);
}

test "streamed d14 selector workspace remains below two MiB" {
    const bytes = try workspaceBytes(30_720, 256, 128, 64);
    try std.testing.expect(bytes < 2 * 1024 * 1024);
    const route: FullRouteBudget = .{
        .route_external_bytes = 501_482_992,
        .retained_scored_bytes = 50 * 1024 * 1024,
        .selector_workspace_bytes = bytes,
        .value_phase_bytes = 160 * 1024 * 1024,
        .geometry_phase_bytes = 252 * 1024 * 1024,
    };
    try std.testing.expect(try route.peak() < 1 << 30);
    const selection_dominates: FullRouteBudget = .{ .route_external_bytes = 100, .retained_scored_bytes = 70, .selector_workspace_bytes = 90, .value_phase_bytes = 1, .geometry_phase_bytes = 1 };
    try std.testing.expectEqual(@as(usize, 260), try selection_dominates.peak());
}
