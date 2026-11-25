//! TRAIN-only residual-group mass calibration. Held-out data is not accepted.
const std = @import("std");
const solver = @import("attention_match.zig");
const anchored = @import("attention_match_joint_chunk_mass_anchored.zig");

pub const fixed_group: u8 = 0xff;
pub const Diagnostics = struct { max_abs_before: f64, max_abs_after: f64, signed_min_before: f64, signed_max_before: f64, signed_min_after: f64, signed_max_after: f64 };
pub const Result = struct {
    allocator: std.mem.Allocator,
    alpha: []f64,
    diagnostics: Diagnostics,
    nnls_sweeps: usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.alpha);
        self.* = undefined;
    }
};
pub const Range = struct { start: usize, rows: usize, exact_core_source_keys: []const usize };
pub const MinimaxDiagnostics = struct {
    max_abs_before: f64,
    mean_abs_before: f64,
    max_abs_after: f64,
    mean_abs_after: f64,
    chosen_power: u8,
    chosen_iteration: usize,
    accepted_iterations: usize,
};
pub const MinimaxResult = struct {
    allocator: std.mem.Allocator,
    alpha: []f64,
    diagnostics: MinimaxDiagnostics,
    nnls_sweeps: usize,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.alpha);
        self.* = undefined;
    }
};
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}

/// Authenticates the Root479 layout and derives a caller-owned row group map.
/// Prefix, recent, and exact-core rows remain fixed; only other middle rows
/// receive their range index. Candidate positions must be unique and sorted.
pub fn deriveCandidateGroups(allocator: std.mem.Allocator, original_positions: []const u64, candidate_positions: []const u64, ranges: []const Range) ![]u8 {
    if (original_positions.len != 7936 or ranges.len != 23 or candidate_positions.len == 0) return error.InvalidAttentionShape;
    for (original_positions, 0..) |position, index| if (position != @as(u64, @intCast(index))) return error.InvalidAttentionShape;
    var expected_start: usize = 128;
    for (ranges, 0..) |range, group| {
        const range_end = try add(range.start, range.rows);
        if (range.start != expected_start or range.rows == 0 or range_end > 5888 or group >= fixed_group) return error.InvalidAttentionShape;
        expected_start = try add(expected_start, range.rows);
        for (range.exact_core_source_keys, 0..) |key, index| if (key < range.start or key >= range_end or (index != 0 and key <= range.exact_core_source_keys[index - 1])) return error.InvalidAttentionShape;
    }
    if (expected_start != 5888) return error.InvalidAttentionShape;
    const groups = try allocator.alloc(u8, candidate_positions.len);
    errdefer allocator.free(groups);
    for (candidate_positions, groups, 0..) |position_u64, *group, index| {
        const position = std.math.cast(usize, position_u64) orelse return error.InvalidAttentionShape;
        if (position >= original_positions.len or (index != 0 and position_u64 <= candidate_positions[index - 1])) return error.InvalidAttentionShape;
        group.* = fixed_group;
        if (position < 128 or position >= 5888) continue;
        for (ranges, 0..) |range, range_index| if (position >= range.start and position < range.start + range.rows) {
            if (std.mem.indexOfScalar(usize, range.exact_core_source_keys, position) == null) group.* = @intCast(range_index);
            break;
        };
    }
    return groups;
}

/// Includes all retained arrays owned by this layer and the nested anchored solve.
pub fn workspaceBytes(q: usize, original_rows: usize, candidate_rows: usize, groups: usize, cfg: solver.Config) !usize {
    if (q == 0 or original_rows == 0 or candidate_rows == 0 or groups == 0) return error.InvalidAttentionShape;
    var n = try mul(try mul(q, groups), 2 * @sizeOf(f64)); // C and rho
    n = try add(n, try mul(q, 2 * @sizeOf(f64))); // fixed mass and rhs
    n = try add(n, try mul(groups, 3 * @sizeOf(f64))); // diagonals and returned alpha
    n = try add(n, try mul(candidate_rows, @sizeOf(f64))); // atomic staged bias
    return add(n, try anchored.workspaceBytes(q, groups, cfg));
}

/// `original_group` and `candidate_group` use fixed_group for prefix/recent/exact-core
/// rows, otherwise 0..groups-1. Probabilities are normalized per head; log masses
/// convert candidate probabilities to candidate unnormalized mass / original Z.
// Synthetic-only convenience path. Production callers must use calibrateTrain,
// whose admission includes caller probability scratch and bounded tiles.
fn fitAndApply(
    allocator: std.mem.Allocator,
    original_p: []const f64,
    candidate_p: []const f64,
    original_log_z: []const f64,
    candidate_log_z: []const f64,
    original_group: []const u8,
    candidate_group: []const u8,
    groups: usize,
    candidate_bias: []f64,
    cfg: solver.Config,
) !Result {
    const q = original_log_z.len;
    if (q == 0 or groups == 0 or groups > 254 or candidate_log_z.len != q or original_group.len == 0 or candidate_group.len != candidate_bias.len or original_p.len != try mul(q, original_group.len) or candidate_p.len != try mul(q, candidate_group.len)) return error.InvalidAttentionShape;
    const admitted = try workspaceBytes(q, original_group.len, candidate_group.len, groups, cfg);
    if (admitted > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    for (original_group) |g| if (g != fixed_group and g >= groups) return error.InvalidAttentionShape;
    for (candidate_group) |g| if (g != fixed_group and g >= groups) return error.InvalidAttentionShape;
    const c = try allocator.alloc(f64, try mul(q, groups));
    defer allocator.free(c);
    @memset(c, 0);
    const rho = try allocator.alloc(f64, c.len);
    defer allocator.free(rho);
    @memset(rho, 0);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(rhs, 0);
    const da = try allocator.alloc(f64, groups);
    defer allocator.free(da);
    @memset(da, 0);
    const db = try allocator.alloc(f64, groups);
    defer allocator.free(db);
    @memset(db, 0);
    var before: Diagnostics = .{ .max_abs_before = 0, .max_abs_after = 0, .signed_min_before = std.math.inf(f64), .signed_max_before = -std.math.inf(f64), .signed_min_after = 0, .signed_max_after = 0 };
    for (0..q) |qi| {
        try cfg.control.check();
        const ratio = std.math.exp(candidate_log_z[qi] - original_log_z[qi]);
        if (!std.math.isFinite(ratio)) return error.NonFiniteAttention;
        var fixed_mass: f64 = 0;
        for (candidate_group, 0..) |g, ri| {
            const v = candidate_p[qi * candidate_group.len + ri] * ratio;
            if (!std.math.isFinite(v) or v < 0) return error.NonFiniteAttention;
            if (g == fixed_group) fixed_mass += v else c[qi * groups + g] += v;
        }
        for (original_group, 0..) |g, ri| {
            const v = original_p[qi * original_group.len + ri];
            if (!std.math.isFinite(v) or v < 0) return error.NonFiniteAttention;
            if (g != fixed_group) {
                rho[qi * groups + g] += v;
                rhs[qi] += v;
            }
        }
        var total = fixed_mass;
        for (0..groups) |g| {
            const cv = c[qi * groups + g];
            const rv = rho[qi * groups + g];
            total += cv;
            da[g] += cv * cv;
            db[g] += cv * rv;
        }
        const signed = total - 1;
        before.max_abs_before = @max(before.max_abs_before, @abs(signed));
        before.signed_min_before = @min(before.signed_min_before, signed);
        before.signed_max_before = @max(before.signed_max_before, signed);
    }
    var solved = try anchored.fit(allocator, c, rhs, da, db, rho, cfg);
    defer solved.deinit();
    const alpha = try allocator.dupe(f64, solved.alpha);
    errdefer allocator.free(alpha);
    for (candidate_bias, candidate_group) |*bias, g| if (g != fixed_group) {
        if (!(alpha[g] > 0) or !std.math.isFinite(alpha[g])) return error.NonFiniteAttention;
        bias.* += @log(alpha[g]);
        if (!std.math.isFinite(bias.*) or @abs(bias.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        bias.* = @floatCast(@as(f32, @floatCast(bias.*)));
    };
    before.signed_min_after = std.math.inf(f64);
    before.signed_max_after = -std.math.inf(f64);
    for (0..q) |qi| {
        var total: f64 = 0;
        const ratio = std.math.exp(candidate_log_z[qi] - original_log_z[qi]);
        for (candidate_group, 0..) |g, ri| {
            const v = candidate_p[qi * candidate_group.len + ri] * ratio;
            total += if (g == fixed_group) v else v * alpha[g];
        }
        const signed = total - 1;
        before.max_abs_after = @max(before.max_abs_after, @abs(signed));
        before.signed_min_after = @min(before.signed_min_after, signed);
        before.signed_max_after = @max(before.signed_max_after, signed);
    }
    try cfg.control.check();
    return .{ .allocator = allocator, .alpha = alpha, .diagnostics = before, .nnls_sweeps = solved.nnls_sweeps };
}

/// Bounded diagnostic entry point. The caller accumulates `c`, `rho`, and
/// `fixed_mass` in bounded query tiles, so neither full attention matrix is
/// retained. `c` and `rho` are query-major q-by-groups.
pub fn fitFeaturesAndApply(
    allocator: std.mem.Allocator,
    c: []const f64,
    rho: []const f64,
    fixed_mass: []const f64,
    candidate_group: []const u8,
    groups: usize,
    candidate_bias: []f64,
    cfg: solver.Config,
) !Result {
    const q = fixed_mass.len;
    if (q == 0 or groups == 0 or groups > 254 or c.len != try mul(q, groups) or rho.len != c.len or candidate_group.len != candidate_bias.len) return error.InvalidAttentionShape;
    var retained = try mul(c.len, 2 * @sizeOf(f64));
    retained = try add(retained, try mul(q, @sizeOf(f64)));
    retained = try add(retained, try mul(groups, 3 * @sizeOf(f64)));
    retained = try add(retained, try mul(candidate_bias.len, @sizeOf(f64)));
    retained = try add(retained, try anchored.workspaceBytes(q, groups, cfg));
    if (retained > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    const da = try allocator.alloc(f64, groups);
    defer allocator.free(da);
    @memset(da, 0);
    const db = try allocator.alloc(f64, groups);
    defer allocator.free(db);
    @memset(db, 0);
    var diagnostics: Diagnostics = .{ .max_abs_before = 0, .max_abs_after = 0, .signed_min_before = std.math.inf(f64), .signed_max_before = -std.math.inf(f64), .signed_min_after = std.math.inf(f64), .signed_max_after = -std.math.inf(f64) };
    for (0..q) |qi| {
        try cfg.control.check();
        if (!std.math.isFinite(fixed_mass[qi]) or fixed_mass[qi] < 0) return error.NonFiniteAttention;
        rhs[qi] = 0;
        var total = fixed_mass[qi];
        for (0..groups) |group| {
            const cv = c[qi * groups + group];
            const rv = rho[qi * groups + group];
            if (!std.math.isFinite(cv) or cv < 0 or !std.math.isFinite(rv) or rv < 0) return error.NonFiniteAttention;
            rhs[qi] += rv;
            total += cv;
            da[group] += cv * cv;
            db[group] += cv * rv;
        }
        const signed = total - 1;
        diagnostics.max_abs_before = @max(diagnostics.max_abs_before, @abs(signed));
        diagnostics.signed_min_before = @min(diagnostics.signed_min_before, signed);
        diagnostics.signed_max_before = @max(diagnostics.signed_max_before, signed);
    }
    var solved = try anchored.fit(allocator, c, rhs, da, db, rho, cfg);
    defer solved.deinit();
    const alpha = try allocator.dupe(f64, solved.alpha);
    errdefer allocator.free(alpha);
    const staged_bias = try allocator.dupe(f64, candidate_bias);
    defer allocator.free(staged_bias);
    for (staged_bias, candidate_group) |*bias, group| if (group != fixed_group) {
        if (group >= groups or !(alpha[group] > 0) or !std.math.isFinite(alpha[group])) return error.NonFiniteAttention;
        bias.* += @log(alpha[group]);
        if (!std.math.isFinite(bias.*) or @abs(bias.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        bias.* = @floatCast(@as(f32, @floatCast(bias.*)));
    };
    try cfg.control.check();
    @memcpy(candidate_bias, staged_bias);
    for (0..q) |qi| {
        var total = fixed_mass[qi];
        for (alpha, 0..) |value, group| total += c[qi * groups + group] * value;
        const signed = total - 1;
        diagnostics.max_abs_after = @max(diagnostics.max_abs_after, @abs(signed));
        diagnostics.signed_min_after = @min(diagnostics.signed_min_after, signed);
        diagnostics.signed_max_after = @max(diagnostics.signed_max_after, signed);
    }
    return .{ .allocator = allocator, .alpha = alpha, .diagnostics = diagnostics, .nnls_sweeps = solved.nnls_sweeps };
}

/// End-to-end bounded TRAIN-only path. Attention is evaluated in at most 32
/// query rows per tile using the scalar FP64 path and identical Head semantics.
pub fn calibrateTrain(
    allocator: std.mem.Allocator,
    original: solver.Head,
    original_positions: []const u64,
    candidate: solver.Head,
    candidate_positions: []const u64,
    ranges: []const Range,
    train: []const f64,
    candidate_bias: []f64,
    cfg: solver.Config,
) !Result {
    if (original.rows != original_positions.len or candidate.rows != candidate_positions.len or candidate.bias == null or candidate.bias.?.ptr != candidate_bias.ptr or candidate_bias.len != candidate.rows or original.dim != candidate.dim or train.len == 0 or train.len % original.dim != 0) return error.InvalidAttentionShape;
    const q = train.len / original.dim;
    const groups = ranges.len;
    const tile_rows: usize = 32;
    var admitted = try workspaceBytes(q, original.rows, candidate.rows, groups, cfg);
    admitted = try add(admitted, try mul(candidate.rows, @sizeOf(u8)));
    admitted = try add(admitted, try mul(try mul(tile_rows, try add(original.rows, candidate.rows)), @sizeOf(f64)));
    admitted = try add(admitted, try mul(tile_rows, 2 * @sizeOf(f64)));
    if (admitted > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    const candidate_group = try deriveCandidateGroups(allocator, original_positions, candidate_positions, ranges);
    defer allocator.free(candidate_group);
    const c = try allocator.alloc(f64, try mul(q, groups));
    defer allocator.free(c);
    @memset(c, 0);
    const rho = try allocator.alloc(f64, c.len);
    defer allocator.free(rho);
    @memset(rho, 0);
    const fixed_mass = try allocator.alloc(f64, q);
    defer allocator.free(fixed_mass);
    @memset(fixed_mass, 0);
    const op = try allocator.alloc(f64, try mul(tile_rows, original.rows));
    defer allocator.free(op);
    const cp = try allocator.alloc(f64, try mul(tile_rows, candidate.rows));
    defer allocator.free(cp);
    const oz = try allocator.alloc(f64, tile_rows);
    defer allocator.free(oz);
    const cz = try allocator.alloc(f64, tile_rows);
    defer allocator.free(cz);
    var offset: usize = 0;
    while (offset < q) : (offset += tile_rows) {
        try cfg.control.check();
        const take = @min(tile_rows, q - offset);
        const queries = train[offset * original.dim ..][0 .. take * original.dim];
        try solver.attendBatch(allocator, original, queries, op[0 .. take * original.rows], null, oz[0..take], .{ .target_rows = 1, .control = cfg.control });
        try solver.attendBatch(allocator, candidate, queries, cp[0 .. take * candidate.rows], null, cz[0..take], .{ .target_rows = 1, .control = cfg.control });
        for (0..take) |local_q| {
            const qi = offset + local_q;
            const ratio = std.math.exp(cz[local_q] - oz[local_q]);
            if (!std.math.isFinite(ratio)) return error.NonFiniteAttention;
            for (candidate_group, 0..) |group, row| {
                const value = cp[local_q * candidate.rows + row] * ratio;
                if (group == fixed_group) fixed_mass[qi] += value else c[qi * groups + group] += value;
            }
            for (original_positions, 0..) |position_u64, row| {
                const position = std.math.cast(usize, position_u64) orelse return error.InvalidAttentionShape;
                if (position < 128 or position >= 5888) continue;
                for (ranges, 0..) |range, group| if (position >= range.start and position < range.start + range.rows) {
                    if (std.mem.indexOfScalar(usize, range.exact_core_source_keys, position) == null) rho[qi * groups + group] += op[local_q * original.rows + row];
                    break;
                };
            }
        }
    }
    var result = try fitFeaturesAndApply(allocator, c, rho, fixed_mass, candidate_group, groups, candidate_bias, cfg);
    errdefer result.deinit();
    // Re-evaluate the rounded bias bytes. These are actual native FP64 TRAIN
    // log-mass diagnostics, not the pre-round linear prediction.
    result.diagnostics.max_abs_after = 0;
    result.diagnostics.signed_min_after = std.math.inf(f64);
    result.diagnostics.signed_max_after = -std.math.inf(f64);
    offset = 0;
    while (offset < q) : (offset += tile_rows) {
        try cfg.control.check();
        const take = @min(tile_rows, q - offset);
        const queries = train[offset * original.dim ..][0 .. take * original.dim];
        try solver.attendBatch(allocator, original, queries, op[0 .. take * original.rows], null, oz[0..take], .{ .target_rows = 1, .control = cfg.control });
        try solver.attendBatch(allocator, candidate, queries, cp[0 .. take * candidate.rows], null, cz[0..take], .{ .target_rows = 1, .control = cfg.control });
        for (oz[0..take], cz[0..take]) |original_z, candidate_z| {
            const signed = std.math.exp(candidate_z - original_z) - 1;
            if (!std.math.isFinite(signed)) return error.NonFiniteAttention;
            result.diagnostics.max_abs_after = @max(result.diagnostics.max_abs_after, @abs(signed));
            result.diagnostics.signed_min_after = @min(result.diagnostics.signed_min_after, signed);
            result.diagnostics.signed_max_after = @max(result.diagnostics.signed_max_after, signed);
        }
    }
    return result;
}

fn absoluteStats(c: []const f64, fixed_mass: []const f64, alpha: []const f64) !struct { max: f64, mean: f64, norm: f64 } {
    const q = fixed_mass.len;
    const groups = alpha.len;
    if (q == 0 or groups == 0 or c.len != try mul(q, groups)) return error.InvalidAttentionShape;
    var maximum: f64 = 0;
    var total: f64 = 0;
    var norm: f64 = 0;
    for (alpha) |value| norm += (value - 1) * (value - 1);
    for (0..q) |qi| {
        var fitted = fixed_mass[qi];
        for (alpha, 0..) |value, group| fitted += c[qi * groups + group] * value;
        const residual = @abs(fitted - 1);
        if (!std.math.isFinite(residual)) return error.NonFiniteAttention;
        maximum = @max(maximum, residual);
        total += residual;
    }
    return .{ .max = maximum, .mean = total / @as(f64, @floatFromInt(q)), .norm = norm };
}

fn lpObjective(c: []const f64, rhs: []const f64, alpha: []const f64, power: u8) !f64 {
    const groups = alpha.len;
    var objective: f64 = 0;
    for (rhs, 0..) |target, qi| {
        var fitted: f64 = 0;
        for (alpha, 0..) |value, group| fitted += c[qi * groups + group] * value;
        objective += std.math.pow(f64, @abs(fitted - target), @floatFromInt(power));
    }
    if (!std.math.isFinite(objective)) return error.NonFiniteAttention;
    return objective;
}

fn irlsRowScale(residual: f64, power: u8) !f64 {
    if (!std.math.isFinite(residual) or power < 2) return error.NonFiniteAttention;
    const exponent = @as(f64, @floatFromInt(power - 2)) / 2;
    const scale = @min(1e6, std.math.pow(f64, @max(@abs(residual), 1e-6), exponent));
    if (!std.math.isFinite(scale) or scale <= 0) return error.NonFiniteAttention;
    return scale;
}

/// TRAIN-only global original-unit minimax calibration. No per-range anchor is
/// present: every row minimizes C*alpha + fixed_mass - 1. Candidate choice is
/// frozen to alpha=1, unanchored L2, and accepted p={4,8,16} IRLS iterates.
pub fn calibrateTrainGlobalMinimax(
    allocator: std.mem.Allocator,
    original: solver.Head,
    original_positions: []const u64,
    candidate: solver.Head,
    candidate_positions: []const u64,
    ranges: []const Range,
    train: []const f64,
    candidate_bias: []f64,
    cfg: solver.Config,
) !MinimaxResult {
    if (original.rows != original_positions.len or candidate.rows != candidate_positions.len or candidate.bias == null or candidate.bias.?.ptr != candidate_bias.ptr or candidate_bias.len != candidate.rows or original.dim != candidate.dim or train.len == 0 or train.len % original.dim != 0) return error.InvalidAttentionShape;
    if (!std.math.isFinite(cfg.min_weight) or cfg.min_weight <= 0 or cfg.min_weight > 1) return error.InvalidFitConfig;
    const q = train.len / original.dim;
    const groups = ranges.len;
    const tile_rows: usize = 32;
    const qg = try mul(q, groups);
    var admitted = try mul(qg, 2 * @sizeOf(f64)); // C and scaled C
    admitted = try add(admitted, try mul(q, 4 * @sizeOf(f64))); // fixed, rhs, scaled rhs, residual
    admitted = try add(admitted, try mul(groups, 6 * @sizeOf(f64))); // current/proposed/previous/best/norms/returned alpha
    admitted = try add(admitted, try mul(candidate.rows, @sizeOf(u8) + @sizeOf(f64))); // groups and staged bias
    admitted = try add(admitted, try mul(try mul(tile_rows, try add(original.rows, candidate.rows)), @sizeOf(f64)));
    admitted = try add(admitted, try mul(tile_rows, 2 * @sizeOf(f64)));
    admitted = try add(admitted, try solver.workspaceBytes(groups, groups, q, 1));
    if (admitted > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const candidate_group = try deriveCandidateGroups(allocator, original_positions, candidate_positions, ranges);
    defer allocator.free(candidate_group);
    const c = try allocator.alloc(f64, qg);
    defer allocator.free(c);
    @memset(c, 0);
    const fixed_mass = try allocator.alloc(f64, q);
    defer allocator.free(fixed_mass);
    @memset(fixed_mass, 0);
    const op = try allocator.alloc(f64, try mul(tile_rows, original.rows));
    defer allocator.free(op);
    const cp = try allocator.alloc(f64, try mul(tile_rows, candidate.rows));
    defer allocator.free(cp);
    const oz = try allocator.alloc(f64, tile_rows);
    defer allocator.free(oz);
    const cz = try allocator.alloc(f64, tile_rows);
    defer allocator.free(cz);
    var offset: usize = 0;
    while (offset < q) : (offset += tile_rows) {
        try cfg.control.check();
        const take = @min(tile_rows, q - offset);
        const queries = train[offset * original.dim ..][0 .. take * original.dim];
        try solver.attendBatch(allocator, original, queries, op[0 .. take * original.rows], null, oz[0..take], .{ .target_rows = 1, .control = cfg.control });
        try solver.attendBatch(allocator, candidate, queries, cp[0 .. take * candidate.rows], null, cz[0..take], .{ .target_rows = 1, .control = cfg.control });
        for (0..take) |local_q| {
            const qi = offset + local_q;
            const ratio = std.math.exp(cz[local_q] - oz[local_q]);
            if (!std.math.isFinite(ratio)) return error.NonFiniteAttention;
            for (candidate_group, 0..) |group, row| {
                const value = cp[local_q * candidate.rows + row] * ratio;
                if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                if (group == fixed_group) fixed_mass[qi] += value else c[qi * groups + group] += value;
            }
        }
    }
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    for (fixed_mass, rhs) |fixed, *target| {
        target.* = 1 - fixed;
        if (!std.math.isFinite(target.*) or target.* < 0) return error.NonFiniteAttention;
    }
    const current = try allocator.alloc(f64, groups);
    defer allocator.free(current);
    @memset(current, 1);
    const proposed = try allocator.alloc(f64, groups);
    defer allocator.free(proposed);
    const previous = try allocator.alloc(f64, groups);
    defer allocator.free(previous);
    const best = try allocator.alloc(f64, groups);
    defer allocator.free(best);
    @memset(best, 1);
    const scaled = try allocator.alloc(f64, qg);
    defer allocator.free(scaled);
    const scaled_rhs = try allocator.alloc(f64, q);
    defer allocator.free(scaled_rhs);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, groups);
    defer allocator.free(norms);
    const before = try absoluteStats(c, fixed_mass, current);
    var best_stats = before;
    var best_power: u8 = 0;
    var best_iteration: usize = 0;
    var accepted_iterations: usize = 0;
    var sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, c, rhs, q, groups, current, residual, norms, cfg, false);
    for (current) |*value| {
        if (!std.math.isFinite(value.*) or value.* < 0) return error.NonFiniteAttention;
        value.* = @max(value.*, cfg.min_weight);
    }
    var stats = try absoluteStats(c, fixed_mass, current);
    if (stats.max < best_stats.max or (stats.max == best_stats.max and stats.norm < best_stats.norm)) {
        @memcpy(best, current);
        best_stats = stats;
        best_power = 2;
    }
    for ([_]u8{ 4, 8, 16 }) |power| {
        for (0..8) |iteration| {
            try cfg.control.check();
            @memcpy(previous, current);
            const old_objective = try lpObjective(c, rhs, previous, power);
            var max_scale: f64 = 0;
            for (rhs, 0..) |target, qi| {
                var fitted: f64 = 0;
                for (previous, 0..) |value, group| fitted += c[qi * groups + group] * value;
                const scale = try irlsRowScale(fitted - target, power);
                max_scale = @max(max_scale, scale);
                scaled_rhs[qi] = target * scale;
                for (0..groups) |group| scaled[qi * groups + group] = c[qi * groups + group] * scale;
            }
            for (scaled_rhs) |*value| value.* /= max_scale;
            for (scaled) |*value| value.* /= max_scale;
            @memcpy(proposed, previous);
            sweeps = @max(sweeps, try solver.fitMassActiveTargetDiagnostic(allocator, scaled, scaled_rhs, q, groups, proposed, residual, norms, cfg, true));
            for (proposed) |*value| {
                if (!std.math.isFinite(value.*) or value.* < 0) return error.NonFiniteAttention;
                value.* = @max(value.*, cfg.min_weight);
            }
            var accepted = false;
            var step: f64 = 1;
            for (0..8) |_| {
                try cfg.control.check();
                for (current, previous, proposed) |*out, old, next| out.* = @max(cfg.min_weight, old + step * (next - old));
                if (try lpObjective(c, rhs, current, power) <= old_objective) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) {
                @memcpy(current, previous);
                break;
            }
            accepted_iterations += 1;
            stats = try absoluteStats(c, fixed_mass, current);
            if (stats.max < best_stats.max or (stats.max == best_stats.max and stats.norm < best_stats.norm)) {
                @memcpy(best, current);
                best_stats = stats;
                best_power = power;
                best_iteration = iteration + 1;
            }
        }
    }
    const alpha = try allocator.dupe(f64, best);
    errdefer allocator.free(alpha);
    const staged_bias = try allocator.dupe(f64, candidate_bias);
    defer allocator.free(staged_bias);
    for (staged_bias, candidate_group) |*bias, group| if (group != fixed_group) {
        bias.* += @log(alpha[group]);
        if (!std.math.isFinite(bias.*) or @abs(bias.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        bias.* = @floatCast(@as(f32, @floatCast(bias.*)));
    };
    try cfg.control.check();
    @memcpy(candidate_bias, staged_bias);
    // Native rounded-bias TRAIN evaluation is authoritative for returned max/mean.
    var native_max: f64 = 0;
    var native_mean: f64 = 0;
    offset = 0;
    while (offset < q) : (offset += tile_rows) {
        try cfg.control.check();
        const take = @min(tile_rows, q - offset);
        const queries = train[offset * original.dim ..][0 .. take * original.dim];
        try solver.attendBatch(allocator, original, queries, op[0 .. take * original.rows], null, oz[0..take], .{ .target_rows = 1, .control = cfg.control });
        try solver.attendBatch(allocator, candidate, queries, cp[0 .. take * candidate.rows], null, cz[0..take], .{ .target_rows = 1, .control = cfg.control });
        for (oz[0..take], cz[0..take]) |original_z, candidate_z| {
            const value = @abs(std.math.exp(candidate_z - original_z) - 1);
            if (!std.math.isFinite(value)) return error.NonFiniteAttention;
            native_max = @max(native_max, value);
            native_mean += value;
        }
    }
    native_mean /= @as(f64, @floatFromInt(q));
    return .{ .allocator = allocator, .alpha = alpha, .diagnostics = .{ .max_abs_before = before.max, .mean_abs_before = before.mean, .max_abs_after = native_max, .mean_abs_after = native_mean, .chosen_power = best_power, .chosen_iteration = best_iteration, .accepted_iterations = accepted_iterations }, .nnls_sweeps = sweeps };
}

test "calibration preserves fixed bias and scales only residual groups" {
    var bias = [_]f64{ 1, 2, 3 };
    var r = try fitAndApply(std.testing.allocator, &.{ 0.5, 0.5, 0.25, 0.75 }, &.{ 0.75, 0.125, 0.125, 0.5, 0.25, 0.25 }, &.{ 0, 0 }, &.{ @log(2.0 / 3.0), @log(0.5) }, &.{ fixed_group, 0 }, &.{ fixed_group, 0, 0 }, 1, &bias, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer r.deinit();
    try std.testing.expectEqual(@as(f64, 1), bias[0]);
    try std.testing.expectApproxEqAbs(@as(f64, 3), r.alpha[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@as(f32, @floatCast(2 + @log(3.0))))), bias[1], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@as(f32, @floatCast(3 + @log(3.0))))), bias[2], 1e-12);
    try std.testing.expect(r.diagnostics.max_abs_after <= r.diagnostics.max_abs_before + 1e-8);
}
test "calibration rejects workspace and malformed groups before mutation" {
    var bias = [_]f64{0};
    try std.testing.expectError(error.CompactionWorkspaceLimit, fitAndApply(std.testing.allocator, &.{1}, &.{1}, &.{0}, &.{0}, &.{0}, &.{0}, 1, &bias, .{ .target_rows = 1, .max_workspace_bytes = 1 }));
    try std.testing.expectError(error.InvalidAttentionShape, fitAndApply(std.testing.allocator, &.{1}, &.{1}, &.{0}, &.{0}, &.{1}, &.{0}, 1, &bias, .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 }));
    try std.testing.expectEqual(@as(f64, 0), bias[0]);
}

test "position grouping rejects duplicate candidate and noncontiguous authority" {
    var original: [7936]u64 = undefined;
    for (&original, 0..) |*value, index| value.* = index;
    var ranges: [23]Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, index| {
        const rows: usize = if (index == 22) 128 else 256;
        range.* = .{ .start = start, .rows = rows, .exact_core_source_keys = &.{} };
        start += rows;
    }
    try std.testing.expectError(error.InvalidAttentionShape, deriveCandidateGroups(std.testing.allocator, &original, &.{ 128, 128 }, &ranges));
    original[1] = 9;
    try std.testing.expectError(error.InvalidAttentionShape, deriveCandidateGroups(std.testing.allocator, &original, &.{128}, &ranges));
}

test "feature calibration observes cancellation before bias mutation" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var bias = [_]f64{0};
    try std.testing.expectError(error.Cancelled, fitFeaturesAndApply(std.testing.allocator, &.{1}, &.{1}, &.{0}, &.{0}, 1, &bias, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20, .control = .{ .check_fn = Cancel.check } }));
    try std.testing.expectEqual(@as(f64, 0), bias[0]);
}

fn allocationFailureFixture(allocator: std.mem.Allocator) !void {
    var bias = [_]f64{ 0, 0 };
    var result = try fitFeaturesAndApply(allocator, &.{ 0.4, 0.3 }, &.{ 0.5, 0.5 }, &.{ 0, 0 }, &.{ 0, 0 }, 1, &bias, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 1 << 20 });
    defer result.deinit();
}

test "feature calibration allocation failure sweep is leak free" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureFixture, .{});
}

test "global minimax absolute objective distinguishes max error and norm tie" {
    const c = [_]f64{ 1, 0, 0, 1 };
    const fixed = [_]f64{ 0, 0 };
    const a = try absoluteStats(&c, &fixed, &.{ 0.5, 1.5 });
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), a.max, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), a.mean, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), a.norm, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.125), try lpObjective(&c, &.{ 1, 1 }, &.{ 0.5, 1.5 }, 4), 1e-12);
}

test "global minimax IRLS floors residual magnitude before powers" {
    try std.testing.expectApproxEqAbs(@as(f64, 1e-6), try irlsRowScale(0, 4), 1e-20);
    try std.testing.expectApproxEqAbs(@as(f64, 1e-18), try irlsRowScale(0, 8), 1e-30);
    try std.testing.expectApproxEqAbs(@as(f64, 1e-42), try irlsRowScale(0, 16), 1e-54);
    try std.testing.expectEqual(@as(f64, 1e6), try irlsRowScale(1e3, 16));
}

fn minimaxAllocationFixture(allocator: std.mem.Allocator) !void {
    const keys = try allocator.alloc(f64, 7936);
    defer allocator.free(keys);
    @memset(keys, 0);
    const values = try allocator.alloc(f64, 7936);
    defer allocator.free(values);
    @memset(values, 0);
    const bias = try allocator.alloc(f64, 7936);
    defer allocator.free(bias);
    @memset(bias, 0);
    const positions = try allocator.alloc(u64, 7936);
    defer allocator.free(positions);
    for (positions, 0..) |*position, index| position.* = index;
    var candidate_bias = [_]f64{0};
    var ranges: [23]Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, index| {
        const rows: usize = if (index == 22) 128 else 256;
        range.* = .{ .start = start, .rows = rows, .exact_core_source_keys = &.{} };
        start += rows;
    }
    const original: solver.Head = .{ .keys = keys, .values = values, .bias = bias, .rows = 7936, .dim = 1, .scale = 1 };
    const candidate: solver.Head = .{ .keys = &.{0}, .values = &.{0}, .bias = &candidate_bias, .rows = 1, .dim = 1, .scale = 1 };
    var result = try calibrateTrainGlobalMinimax(allocator, original, positions, candidate, &.{128}, &ranges, &.{0}, &candidate_bias, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 8 << 20 });
    result.deinit();
}

test "global minimax calibration cancellation and allocation unwind" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var positions: [7936]u64 = undefined;
    for (&positions, 0..) |*position, index| position.* = index;
    var zeros: [7936]f64 = @splat(0);
    var candidate_bias = [_]f64{0};
    var ranges: [23]Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, index| {
        const rows: usize = if (index == 22) 128 else 256;
        range.* = .{ .start = start, .rows = rows, .exact_core_source_keys = &.{} };
        start += rows;
    }
    const original: solver.Head = .{ .keys = &zeros, .values = &zeros, .bias = &zeros, .rows = 7936, .dim = 1, .scale = 1 };
    const candidate: solver.Head = .{ .keys = &.{0}, .values = &.{0}, .bias = &candidate_bias, .rows = 1, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.Cancelled, calibrateTrainGlobalMinimax(std.testing.allocator, original, &positions, candidate, &.{128}, &ranges, &.{0}, &candidate_bias, .{ .target_rows = 1, .mass_solver = .active_set, .max_workspace_bytes = 8 << 20, .control = .{ .check_fn = Cancel.check } }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, minimaxAllocationFixture, .{});
}
