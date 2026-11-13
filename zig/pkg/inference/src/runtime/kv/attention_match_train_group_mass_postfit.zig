//! TRAIN-only global residual-range mass L2 postfit.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("attention_match.zig");
const training_source = @import("attention_match_joint_training_source.zig");
const lawson = @import("attention_match_train_group_mass_lawson.zig");

pub const policy = "train_global_residual_range_mass_l2_exact_f32_cache_v1";
pub const policy_digest: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};
pub const combined_cap_bytes: usize = 1024 * 1024 * 1024;
pub const Range = struct { start: usize, rows: usize, target_rows: usize };
pub const LawsonOutcome = enum { disabled, not_needed, improved, no_improvement, proposal_rejected };
pub const Diagnostics = struct {
    max_abs_before: f64,
    max_abs_after: f64,
    nnls_sweeps: usize,
    floored_groups: usize,
    workspace_bytes: usize,
    lawson_attempted_iterations: usize = 0,
    lawson_accepted_improvements: usize = 0,
    lawson_nnls_sweeps: usize = 0,
    lawson_outcome: LawsonOutcome = .disabled,
};
pub const RangePageLoader = struct {
    context: ?*anyopaque,
    train: []const f64,
    query_rows: usize,
    original_rows: usize,
    load_fn: *const fn (?*anyopaque, Range, solver.Control) anyerror!*const training_source.Source,
    release_fn: *const fn (?*anyopaque) void,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub fn workspaceBytes(q: usize, original_rows: usize, candidate_rows: usize, groups: usize) !usize {
    if (q == 0 or original_rows == 0 or candidate_rows == 0 or groups == 0) return error.InvalidFitConfig;
    var bytes = try mul(try mul(q, groups), @sizeOf(f64));
    bytes = try add(bytes, try mul(q, 3 * @sizeOf(f64)));
    bytes = try add(bytes, try mul(groups, 4 * @sizeOf(f64)));
    bytes = try add(bytes, try mul(candidate_rows, @sizeOf(f64)));
    // Core mask, original residual keys, candidate residual slots/groups/ratios.
    bytes = try add(bytes, original_rows);
    bytes = try add(bytes, try mul(original_rows, @sizeOf(usize)));
    bytes = try add(bytes, try mul(candidate_rows, 2 * @sizeOf(usize) + @sizeOf(f64)));
    return add(bytes, try solver.workspaceBytes(groups, groups, q, 1));
}

pub fn workspaceBytesLawson(q: usize, original_rows: usize, candidate_rows: usize, groups: usize) !usize {
    return add(try workspaceBytes(q, original_rows, candidate_rows, groups), try lawson.workspaceBytes(q, groups));
}
fn groupFor(key: usize, ranges: []const Range) ?usize {
    for (ranges, 0..) |range, group| if (key >= range.start and key < range.start + range.rows) return group;
    return null;
}

fn applyBiasAtomic(allocator: std.mem.Allocator, bias: []f64, source_indices: []const usize, ranges: []const Range, core_mask: []const bool, alpha: []const f64, control: solver.Control) !void {
    if (bias.len != source_indices.len or alpha.len != ranges.len) return error.InvalidFitConfig;
    const staged = try allocator.dupe(f64, bias);
    defer allocator.free(staged);
    for (staged, source_indices) |*value, key| {
        const group = groupFor(key, ranges) orelse continue;
        if (key >= core_mask.len) return error.InvalidFitConfig;
        if (ranges[group].target_rows == ranges[group].rows or core_mask[key]) continue;
        value.* += @log(alpha[group]);
        if (!std.math.isFinite(value.*) or @abs(value.*) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        value.* = @floatCast(@as(f32, @floatCast(value.*)));
    }
    try control.check();
    @memcpy(bias, staged);
}

fn applyInternal(allocator: std.mem.Allocator, original: solver.Head, candidate: solver.Head, candidate_source_indices: []const usize, ranges: []const Range, exact_core_source_indices: []const usize, cached: *const training_source.Source, cfg: solver.Config, external_live_bytes: usize, use_lawson: bool, force_lawson_rejection_for_test: bool) !Diagnostics {
    if (force_lawson_rejection_for_test and (!builtin.is_test or !use_lawson)) return error.InvalidFitConfig;
    try original.validate();
    try candidate.validate();
    try cached.validate(original, cached.train_ptr[0..try mul(cached.query_count, cached.dim)], cfg);
    if (cached.probabilities_f32.len == 0 or cached.probability_row_start != 0 or (cached.probability_rows != 0 and cached.probability_rows != original.rows) or original.dim != candidate.dim or candidate.rows != candidate_source_indices.len or original.bias == null or candidate.bias == null or ranges.len == 0 or cfg.min_weight != 1e-12) return error.InvalidFitConfig;
    var expected_start: usize = ranges[0].start;
    if (expected_start >= original.rows) return error.InvalidFitConfig;
    for (ranges) |range| {
        if (range.start != expected_start or range.rows == 0 or range.target_rows == 0 or range.target_rows > range.rows) return error.InvalidFitConfig;
        expected_start = try add(expected_start, range.rows);
    }
    if (expected_start > original.rows) return error.InvalidFitConfig;
    const middle_start = ranges[0].start;
    const middle_end = expected_start;
    for (exact_core_source_indices, 0..) |key, i| if (key < middle_start or key >= middle_end or (i != 0 and exact_core_source_indices[i - 1] >= key)) return error.InvalidFitConfig;
    const q = cached.query_count;
    const groups = ranges.len;
    var active_count: usize = 0;
    for (ranges) |range| active_count += @intFromBool(range.target_rows != range.rows);
    if (active_count == 0) return .{ .max_abs_before = 0, .max_abs_after = 0, .nnls_sweeps = 0, .floored_groups = 0, .workspace_bytes = 0 };
    const workspace = if (use_lawson) try workspaceBytesLawson(q, original.rows, candidate.rows, groups) else try workspaceBytes(q, original.rows, candidate.rows, groups);
    const admitted_cap = try cfg.admittedCombinedCap();
    if (external_live_bytes > admitted_cap or workspace > admitted_cap - external_live_bytes or workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const core_mask = try allocator.alloc(bool, original.rows);
    defer allocator.free(core_mask);
    @memset(core_mask, false);
    for (exact_core_source_indices) |key| core_mask[key] = true;
    const active_groups = try allocator.alloc(usize, active_count);
    defer allocator.free(active_groups);
    var ai: usize = 0;
    for (ranges, 0..) |range, group| if (range.target_rows != range.rows) {
        active_groups[ai] = group;
        ai += 1;
    };
    const source_residual = try allocator.alloc(usize, original.rows);
    defer allocator.free(source_residual);
    var source_residual_count: usize = 0;
    for (ranges) |range| {
        if (range.target_rows == range.rows) continue;
        for (range.start..range.start + range.rows) |key| if (!core_mask[key]) {
            source_residual[source_residual_count] = key;
            source_residual_count += 1;
        };
    }
    const candidate_slots = try allocator.alloc(usize, candidate.rows);
    defer allocator.free(candidate_slots);
    const candidate_active_groups = try allocator.alloc(usize, candidate.rows);
    defer allocator.free(candidate_active_groups);
    const candidate_ratios = try allocator.alloc(f64, candidate.rows);
    defer allocator.free(candidate_ratios);
    var candidate_residual_count: usize = 0;
    for (candidate_source_indices, 0..) |key, slot| {
        if (key >= original.rows or (slot != 0 and candidate_source_indices[slot - 1] >= key)) return error.InvalidFitConfig;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(candidate.keys[slot * candidate.dim ..][0..candidate.dim]), std.mem.sliceAsBytes(original.keys[key * original.dim ..][0..original.dim]))) return error.InvalidFitConfig;
        const group = groupFor(key, ranges);
        const fixed = group == null or ranges[group.?].target_rows == ranges[group.?].rows or core_mask[key];
        if (fixed) {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(candidate.values[slot * candidate.dim ..][0..candidate.dim]), std.mem.sliceAsBytes(original.values[key * original.dim ..][0..original.dim])) or @as(u64, @bitCast(candidate.bias.?[slot])) != @as(u64, @bitCast(original.bias.?[key]))) return error.InvalidFitConfig;
            continue;
        }
        const active = std.mem.indexOfScalar(usize, active_groups, group.?) orelse return error.InvalidFitConfig;
        const ratio = std.math.exp(candidate.bias.?[slot] - original.bias.?[key]);
        if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
        candidate_slots[candidate_residual_count] = slot;
        candidate_active_groups[candidate_residual_count] = active;
        candidate_ratios[candidate_residual_count] = ratio;
        candidate_residual_count += 1;
    }
    const c = try allocator.alloc(f64, try mul(q, active_count));
    defer allocator.free(c);
    @memset(c, 0);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(rhs, 0);
    for (0..q) |qi| {
        if ((qi & 31) == 0) try cfg.control.check();
        for (source_residual[0..source_residual_count]) |key| rhs[qi] += try cached.probability(qi, key);
        if (!std.math.isFinite(rhs[qi]) or rhs[qi] < 0) return error.NonFiniteAttention;
        for (candidate_slots[0..candidate_residual_count], candidate_active_groups[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, active, ratio| {
            const key = candidate_source_indices[slot];
            const value = (try cached.probability(qi, key)) * ratio;
            if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
            c[qi * active_count + active] += value;
        }
    }
    const active_alpha = try allocator.alloc(f64, active_count);
    defer allocator.free(active_alpha);
    @memset(active_alpha, 1);
    const alpha = try allocator.alloc(f64, groups);
    defer allocator.free(alpha);
    @memset(alpha, 1);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, groups);
    defer allocator.free(norms);
    var max_before: f64 = 0;
    for (rhs, 0..) |target, qi| {
        var total: f64 = 0;
        for (0..active_count) |active| total += c[qi * active_count + active];
        max_before = @max(max_before, @abs(total - target));
    }
    const sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, c, rhs, q, active_count, active_alpha, residual, norms[0..active_count], cfg, true);
    var floored: usize = 0;
    for (active_alpha) |*value| {
        if (!std.math.isFinite(value.*) or value.* < 0) return error.NonFiniteAttention;
        if (value.* < cfg.min_weight) {
            value.* = cfg.min_weight;
            floored += 1;
        }
    }
    var lawson_attempted: usize = 0;
    var lawson_accepted: usize = 0;
    var lawson_sweeps: usize = 0;
    var lawson_best_exact: ?f64 = null;
    var lawson_outcome: LawsonOutcome = .disabled;
    if (use_lawson) {
        const Exact = struct {
            cached: *const training_source.Source,
            original_bias: []const f64,
            candidate_bias: []const f64,
            candidate_source_indices: []const usize,
            candidate_slots: []const usize,
            candidate_active_groups: []const usize,
            rounded_ratios: []f64,
            rhs: []const f64,

            fn evaluate(raw: ?*anyopaque, proposed: []const f64, control: solver.Control) !f64 {
                const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
                if (self.rounded_ratios.len != self.candidate_slots.len or self.candidate_slots.len != self.candidate_active_groups.len) return error.InvalidFitConfig;
                for (self.candidate_slots, self.candidate_active_groups, self.rounded_ratios) |slot, active, *ratio| {
                    if (active >= proposed.len) return error.InvalidFitConfig;
                    const key = self.candidate_source_indices[slot];
                    const adjusted = self.candidate_bias[slot] + @log(proposed[active]);
                    if (!std.math.isFinite(adjusted) or @abs(adjusted) > std.math.floatMax(f32)) return error.NonFiniteAttention;
                    const rounded: f64 = @floatCast(@as(f32, @floatCast(adjusted)));
                    ratio.* = std.math.exp(rounded - self.original_bias[key]);
                    if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
                }
                var maximum: f64 = 0;
                for (self.rhs, 0..) |target, qi| {
                    if ((qi & 31) == 0) try control.check();
                    var total: f64 = 0;
                    for (self.candidate_slots, self.rounded_ratios) |slot, ratio| {
                        const key = self.candidate_source_indices[slot];
                        total += (try self.cached.probability(qi, key)) * ratio;
                    }
                    maximum = @max(maximum, @abs(total - target));
                }
                return maximum;
            }
        };
        var exact: Exact = .{ .cached = cached, .original_bias = original.bias.?, .candidate_bias = candidate.bias.?, .candidate_source_indices = candidate_source_indices, .candidate_slots = candidate_slots[0..candidate_residual_count], .candidate_active_groups = candidate_active_groups[0..candidate_residual_count], .rounded_ratios = candidate_ratios[0..candidate_residual_count], .rhs = rhs };
        var refined_optional: ?lawson.Result = (if (force_lawson_rejection_for_test) error.LawsonProposalRejected else lawson.refine(allocator, c, rhs, q, active_count, active_alpha, cfg, .{ .context = &exact, .evaluate_fn = Exact.evaluate }, .{})) catch |err| switch (err) {
            error.LawsonProposalRejected => rejected: {
                lawson_outcome = .proposal_rejected;
                lawson_best_exact = try Exact.evaluate(&exact, active_alpha, cfg.control);
                break :rejected null;
            },
            else => return err,
        };
        if (refined_optional) |*refined| {
            defer refined.deinit();
            if (refined.alpha.len != active_alpha.len or refined.best_exact_max > refined.baseline_exact_max) return error.InvalidFitConfig;
            @memcpy(active_alpha, refined.alpha);
            lawson_attempted = refined.attempted_iterations;
            lawson_accepted = refined.accepted_improvements;
            lawson_sweeps = refined.nnls_sweeps;
            lawson_best_exact = refined.best_exact_max;
            lawson_outcome = if (refined.attempted_iterations == 0) .not_needed else if (refined.accepted_improvements != 0) .improved else .no_improvement;
        }
    }
    for (active_alpha, active_groups) |value, group| alpha[group] = value;
    // Compute evidence from the exact f32 bytes that atomic publication will
    // install, and reject before publication on every fallible numeric check.
    for (candidate_slots[0..candidate_residual_count], candidate_active_groups[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, active, *ratio| {
        const key = candidate_source_indices[slot];
        const adjusted = candidate.bias.?[slot] + @log(active_alpha[active]);
        if (!std.math.isFinite(adjusted) or @abs(adjusted) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        const rounded: f64 = @floatCast(@as(f32, @floatCast(adjusted)));
        ratio.* = std.math.exp(rounded - original.bias.?[key]);
        if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
    }
    var max_after: f64 = 0;
    for (rhs, 0..) |target, qi| {
        var total: f64 = 0;
        for (candidate_slots[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, ratio| total += (try cached.probability(qi, candidate_source_indices[slot])) * ratio;
        max_after = @max(max_after, @abs(total - target));
    }
    if (lawson_best_exact) |expected| if (max_after != expected) return error.InvalidFitConfig;
    try applyBiasAtomic(allocator, @constCast(candidate.bias.?), candidate_source_indices, ranges, core_mask, alpha, cfg.control);
    return .{ .max_abs_before = max_before, .max_abs_after = max_after, .nnls_sweeps = sweeps, .floored_groups = floored, .workspace_bytes = workspace, .lawson_attempted_iterations = lawson_attempted, .lawson_accepted_improvements = lawson_accepted, .lawson_nnls_sweeps = lawson_sweeps, .lawson_outcome = lawson_outcome };
}

pub fn apply(allocator: std.mem.Allocator, original: solver.Head, candidate: solver.Head, candidate_source_indices: []const usize, ranges: []const Range, exact_core_source_indices: []const usize, cached: *const training_source.Source, cfg: solver.Config, external_live_bytes: usize) !Diagnostics {
    return applyInternal(allocator, original, candidate, candidate_source_indices, ranges, exact_core_source_indices, cached, cfg, external_live_bytes, false, false);
}

pub fn applyLawson(allocator: std.mem.Allocator, original: solver.Head, candidate: solver.Head, candidate_source_indices: []const usize, ranges: []const Range, exact_core_source_indices: []const usize, cached: *const training_source.Source, cfg: solver.Config, external_live_bytes: usize) !Diagnostics {
    return applyInternal(allocator, original, candidate, candidate_source_indices, ranges, exact_core_source_indices, cached, cfg, external_live_bytes, true, false);
}

/// Bounded equivalent of `apply`. Each active original range is loaded and
/// released before the next range. No q-by-original probability matrix is
/// retained; heldout is absent from this API.
pub fn applyBounded(allocator: std.mem.Allocator, original: solver.Head, candidate: solver.Head, candidate_source_indices: []const usize, ranges: []const Range, exact_core_source_indices: []const usize, loader: RangePageLoader, cfg: solver.Config, external_live_bytes: usize) !Diagnostics {
    try original.validate();
    try candidate.validate();
    if (loader.query_rows == 0 or loader.train.len != try mul(loader.query_rows, original.dim) or loader.original_rows != original.rows or original.dim != candidate.dim or candidate.rows != candidate_source_indices.len or original.bias == null or candidate.bias == null or ranges.len == 0 or cfg.min_weight != 1e-12) return error.InvalidFitConfig;
    var expected_start: usize = ranges[0].start;
    if (expected_start >= original.rows) return error.InvalidFitConfig;
    var active_count: usize = 0;
    for (ranges) |range| {
        if (range.start != expected_start or range.rows == 0 or range.target_rows == 0 or range.target_rows > range.rows) return error.InvalidFitConfig;
        expected_start = try add(expected_start, range.rows);
        active_count += @intFromBool(range.target_rows != range.rows);
    }
    if (expected_start > original.rows) return error.InvalidFitConfig;
    for (exact_core_source_indices, 0..) |key, i| if (key < ranges[0].start or key >= expected_start or (i != 0 and exact_core_source_indices[i - 1] >= key)) return error.InvalidFitConfig;
    if (active_count == 0) return .{ .max_abs_before = 0, .max_abs_after = 0, .nnls_sweeps = 0, .floored_groups = 0, .workspace_bytes = 0 };
    const q = loader.query_rows;
    const workspace = try workspaceBytes(q, original.rows, candidate.rows, ranges.len);
    const admitted_cap = try cfg.admittedCombinedCap();
    if (external_live_bytes > admitted_cap or workspace > admitted_cap - external_live_bytes or workspace > cfg.max_workspace_bytes) return error.CompactionWorkspaceLimit;
    try cfg.control.check();
    const core_mask = try allocator.alloc(bool, original.rows);
    defer allocator.free(core_mask);
    @memset(core_mask, false);
    for (exact_core_source_indices) |key| core_mask[key] = true;
    const active_groups = try allocator.alloc(usize, active_count);
    defer allocator.free(active_groups);
    var ai: usize = 0;
    for (ranges, 0..) |range, group| if (range.target_rows != range.rows) {
        active_groups[ai] = group;
        ai += 1;
    };
    const candidate_slots = try allocator.alloc(usize, candidate.rows);
    defer allocator.free(candidate_slots);
    const candidate_active_groups = try allocator.alloc(usize, candidate.rows);
    defer allocator.free(candidate_active_groups);
    const candidate_ratios = try allocator.alloc(f64, candidate.rows);
    defer allocator.free(candidate_ratios);
    var candidate_residual_count: usize = 0;
    for (candidate_source_indices, 0..) |key, slot| {
        if (key >= original.rows or (slot != 0 and candidate_source_indices[slot - 1] >= key)) return error.InvalidFitConfig;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(candidate.keys[slot * candidate.dim ..][0..candidate.dim]), std.mem.sliceAsBytes(original.keys[key * original.dim ..][0..original.dim]))) return error.InvalidFitConfig;
        const group = groupFor(key, ranges);
        const fixed = group == null or ranges[group.?].target_rows == ranges[group.?].rows or core_mask[key];
        if (fixed) {
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(candidate.values[slot * candidate.dim ..][0..candidate.dim]), std.mem.sliceAsBytes(original.values[key * original.dim ..][0..original.dim])) or @as(u64, @bitCast(candidate.bias.?[slot])) != @as(u64, @bitCast(original.bias.?[key]))) return error.InvalidFitConfig;
            continue;
        }
        const active = std.mem.indexOfScalar(usize, active_groups, group.?) orelse return error.InvalidFitConfig;
        const ratio = std.math.exp(candidate.bias.?[slot] - original.bias.?[key]);
        if (!std.math.isFinite(ratio) or ratio < 0) return error.NonFiniteAttention;
        candidate_slots[candidate_residual_count] = slot;
        candidate_active_groups[candidate_residual_count] = active;
        candidate_ratios[candidate_residual_count] = ratio;
        candidate_residual_count += 1;
    }
    const c = try allocator.alloc(f64, try mul(q, active_count));
    defer allocator.free(c);
    @memset(c, 0);
    const rhs = try allocator.alloc(f64, q);
    defer allocator.free(rhs);
    @memset(rhs, 0);

    // First page pass freezes the TRAIN objective.
    for (ranges, 0..) |range, group| {
        if (range.target_rows == range.rows) continue;
        const page = try loader.load_fn(loader.context, range, cfg.control);
        defer loader.release_fn(loader.context);
        try page.validate(original, loader.train, cfg);
        if (page.query_count != q or page.probability_row_start != range.start or page.probability_rows != range.rows or page.probabilities.len != 0 or page.probabilities_f32.len != try mul(q, range.rows)) return error.InvalidFitConfig;
        const active = std.mem.indexOfScalar(usize, active_groups, group) orelse return error.InvalidFitConfig;
        for (0..q) |qi| {
            if ((qi & 31) == 0) try cfg.control.check();
            for (0..range.rows) |local| {
                if (!core_mask[range.start + local]) rhs[qi] += try page.probability(qi, range.start + local);
            }
            for (candidate_slots[0..candidate_residual_count], candidate_active_groups[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, candidate_active, ratio| {
                if (candidate_active != active) continue;
                const value = (try page.probability(qi, candidate_source_indices[slot])) * ratio;
                if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
                c[qi * active_count + active] += value;
            }
        }
    }
    for (rhs) |value| if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
    const active_alpha = try allocator.alloc(f64, active_count);
    defer allocator.free(active_alpha);
    @memset(active_alpha, 1);
    const alpha = try allocator.alloc(f64, ranges.len);
    defer allocator.free(alpha);
    @memset(alpha, 1);
    const residual = try allocator.alloc(f64, q);
    defer allocator.free(residual);
    const norms = try allocator.alloc(f64, ranges.len);
    defer allocator.free(norms);
    var max_before: f64 = 0;
    for (rhs, 0..) |target, qi| {
        var total: f64 = 0;
        for (0..active_count) |active| total += c[qi * active_count + active];
        max_before = @max(max_before, @abs(total - target));
    }
    const sweeps = try solver.fitMassActiveTargetDiagnostic(allocator, c, rhs, q, active_count, active_alpha, residual, norms[0..active_count], cfg, true);
    var floored: usize = 0;
    for (active_alpha) |*value| {
        if (!std.math.isFinite(value.*) or value.* < 0) return error.NonFiniteAttention;
        if (value.* < cfg.min_weight) {
            value.* = cfg.min_weight;
            floored += 1;
        }
    }
    for (active_alpha, active_groups) |value, group| alpha[group] = value;
    for (candidate_slots[0..candidate_residual_count], candidate_active_groups[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, active, *ratio| {
        const key = candidate_source_indices[slot];
        const adjusted = candidate.bias.?[slot] + @log(active_alpha[active]);
        if (!std.math.isFinite(adjusted) or @abs(adjusted) > std.math.floatMax(f32)) return error.NonFiniteAttention;
        const rounded: f64 = @floatCast(@as(f32, @floatCast(adjusted)));
        ratio.* = std.math.exp(rounded - original.bias.?[key]);
        if (!std.math.isFinite(ratio.*) or ratio.* < 0) return error.NonFiniteAttention;
    }
    const rounded_totals = try allocator.alloc(f64, q);
    defer allocator.free(rounded_totals);
    @memset(rounded_totals, 0);
    // Second page pass validates the exact rounded bytes before publication.
    for (ranges, 0..) |range, group| {
        if (range.target_rows == range.rows) continue;
        const page = try loader.load_fn(loader.context, range, cfg.control);
        defer loader.release_fn(loader.context);
        try page.validate(original, loader.train, cfg);
        if (page.query_count != q or page.probability_row_start != range.start or page.probability_rows != range.rows or page.probabilities.len != 0 or page.probabilities_f32.len != try mul(q, range.rows)) return error.InvalidFitConfig;
        const active = std.mem.indexOfScalar(usize, active_groups, group) orelse return error.InvalidFitConfig;
        for (0..q) |qi| for (candidate_slots[0..candidate_residual_count], candidate_active_groups[0..candidate_residual_count], candidate_ratios[0..candidate_residual_count]) |slot, candidate_active, ratio| {
            if (candidate_active == active) rounded_totals[qi] += (try page.probability(qi, candidate_source_indices[slot])) * ratio;
        };
    }
    var max_after: f64 = 0;
    for (rhs, rounded_totals) |target, total| max_after = @max(max_after, @abs(total - target));
    try applyBiasAtomic(allocator, @constCast(candidate.bias.?), candidate_source_indices, ranges, core_mask, alpha, cfg.control);
    return .{ .max_abs_before = max_before, .max_abs_after = max_after, .nnls_sweeps = sweeps, .floored_groups = floored, .workspace_bytes = workspace };
}

test "group postfit workspace rejects overflow" {
    try std.testing.expectError(error.CompactionWorkspaceLimit, workspaceBytes(std.math.maxInt(usize), 1, 1, 23));
    try std.testing.expectError(error.CompactionWorkspaceLimit, workspaceBytesLawson(std.math.maxInt(usize), 1, 1, 23));
    const baseline_peak: usize = 119601520 + 440401920 + 384242192;
    const postfit_peak = try workspaceBytes(12288, 7936, 7936, 23);
    const metadata = 7936 * @sizeOf(usize) + 23 * @sizeOf(Range);
    try std.testing.expect(baseline_peak + postfit_peak + metadata <= combined_cap_bytes);
    const lawson_peak = try workspaceBytesLawson(12288, 7936, 7936, 23);
    try std.testing.expect(baseline_peak + lawson_peak + metadata <= combined_cap_bytes);
}

test "bias publication is atomic on cancellation and allocation failure" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    const ranges = [_]Range{.{ .start = 2, .rows = 2, .target_rows = 1 }};
    const source = [_]usize{ 0, 2, 3, 4 };
    var bias = [_]f64{ 0, 0, 0, 0 };
    const before = bias;
    const core_mask = [_]bool{ false, false, false, true, false };
    try std.testing.expectError(error.Cancelled, applyBiasAtomic(std.testing.allocator, &bias, &source, &ranges, &core_mask, &.{2}, .{ .check_fn = Cancel.check }));
    try std.testing.expectEqualSlices(f64, &before, &bias);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, applyBiasAtomic(failing.allocator(), &bias, &source, &ranges, &core_mask, &.{2}, .{}));
    try std.testing.expectEqualSlices(f64, &before, &bias);
}

test "two-query group L2 recovers rounded mass and preserves fixed rows" {
    const original_keys = [_]f64{ 0, 1, 2, 3, 4, 5 };
    const original_values = [_]f64{ 10, 11, 12, 13, 14, 15 };
    const original_bias = [_]f64{ 0, 0, 0, 0, 0, 0 };
    const original: solver.Head = .{ .keys = &original_keys, .values = &original_values, .bias = &original_bias, .rows = 6, .dim = 1, .scale = 1 };
    const source_indices = [_]usize{ 0, 1, 2, 4, 5 };
    const candidate_keys = [_]f64{ 0, 1, 2, 4, 5 };
    const candidate_values = [_]f64{ 10, 11, 12, 14, 15 };
    var candidate_bias = [_]f64{ 0, 0, 0, 0, 0 };
    const candidate: solver.Head = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &candidate_bias, .rows = 5, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    var probabilities = [_]f32{ 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2 };
    var outputs = [_]f64{ 0, 0 };
    var cached: training_source.Source = .{ .allocator = std.testing.allocator, .keys_ptr = original.keys.ptr, .values_ptr = original.values.ptr, .bias_ptr = original.bias.?.ptr, .train_ptr = &train, .rows = 6, .probability_rows = 6, .dim = 1, .query_count = 2, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 1, .cache_bytes = probabilities.len * @sizeOf(f32), .consumer_peak_bytes = 1, .digest = @splat(0), .probabilities_f32 = &probabilities, .outputs = &outputs, .owns_outputs = false };
    const fixed_before = candidate_bias;
    const result = try apply(std.testing.allocator, original, candidate, &source_indices, &.{.{ .start = 2, .rows = 2, .target_rows = 1 }}, &.{}, &cached, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, 0);
    try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@as(f32, @floatCast(@log(@as(f64, 2)))))), candidate_bias[2], 1e-7);
    try std.testing.expect(result.max_abs_after < 1e-6);
    try std.testing.expectEqual(fixed_before[0], candidate_bias[0]);
    try std.testing.expectEqual(fixed_before[1], candidate_bias[1]);
    try std.testing.expectEqual(fixed_before[3], candidate_bias[3]);
    try std.testing.expectEqual(fixed_before[4], candidate_bias[4]);

    var rejected_bias = [_]f64{ 0, 0, 0, 0, 0 };
    const rejected_candidate: solver.Head = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &rejected_bias, .rows = 5, .dim = 1, .scale = 1 };
    const rejected = try applyInternal(std.testing.allocator, original, rejected_candidate, &source_indices, &.{.{ .start = 2, .rows = 2, .target_rows = 1 }}, &.{}, &cached, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, 0, true, true);
    try std.testing.expectEqual(LawsonOutcome.proposal_rejected, rejected.lawson_outcome);
    try std.testing.expectEqual(result.max_abs_after, rejected.max_abs_after);
    try std.testing.expectEqualSlices(f64, &candidate_bias, &rejected_bias);
}

test "bounded loader releases every page and matches analytic group L2" {
    const original_keys = [_]f64{ 0, 1, 2, 3, 4, 5 };
    const original_values = [_]f64{ 10, 11, 12, 13, 14, 15 };
    const original_bias = [_]f64{ 0, 0, 0, 0, 0, 0 };
    const original: solver.Head = .{ .keys = &original_keys, .values = &original_values, .bias = &original_bias, .rows = 6, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0, 1 };
    const Context = struct {
        original: solver.Head,
        train: []const f64,
        full: [12]f32,
        page: [4]f32 = undefined,
        outputs: [2]f64 = .{ 0, 0 },
        source: training_source.Source = undefined,
        active: bool = false,
        loads: usize = 0,
        releases: usize = 0,
        fn load(raw: ?*anyopaque, range: Range, control: solver.Control) !*const training_source.Source {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
            try control.check();
            if (self.active or range.rows != 2) return error.InvalidFitConfig;
            for (0..2) |qi| {
                for (0..range.rows) |local| self.page[qi * range.rows + local] = self.full[qi * self.original.rows + range.start + local];
            }
            self.source = .{ .allocator = std.testing.allocator, .keys_ptr = self.original.keys.ptr, .values_ptr = self.original.values.ptr, .bias_ptr = self.original.bias.?.ptr, .train_ptr = self.train.ptr, .rows = self.original.rows, .probability_row_start = range.start, .probability_rows = range.rows, .dim = self.original.dim, .query_count = 2, .scale = self.original.scale, .softcap = self.original.softcap, .backend = null, .query_batch_rows = 1, .cache_bytes = self.page.len * @sizeOf(f32), .consumer_peak_bytes = 1, .digest = @splat(0), .probabilities_f32 = &self.page, .outputs = &self.outputs, .owns_outputs = false };
            self.active = true;
            self.loads += 1;
            return &self.source;
        }
        fn release(raw: ?*anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(raw orelse return));
            std.debug.assert(self.active);
            self.active = false;
            self.releases += 1;
        }
    };
    var context: Context = .{ .original = original, .train = &train, .full = .{ 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2 } };
    const source_indices = [_]usize{ 0, 1, 2, 4, 5 };
    const candidate_keys = [_]f64{ 0, 1, 2, 4, 5 };
    const candidate_values = [_]f64{ 10, 11, 12, 14, 15 };
    var candidate_bias = [_]f64{ 0, 0, 0, 0, 0 };
    const candidate: solver.Head = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &candidate_bias, .rows = 5, .dim = 1, .scale = 1 };
    const diagnostics = try applyBounded(std.testing.allocator, original, candidate, &source_indices, &.{.{ .start = 2, .rows = 2, .target_rows = 1 }}, &.{}, .{ .context = &context, .train = &train, .query_rows = 2, .original_rows = 6, .load_fn = Context.load, .release_fn = Context.release }, .{ .target_rows = 1, .mass_solver = .active_set, .min_weight = 1e-12, .max_workspace_bytes = 1 << 20 }, 0);
    try std.testing.expect(!context.active);
    try std.testing.expectEqual(@as(usize, 2), context.loads);
    try std.testing.expectEqual(context.loads, context.releases);
    try std.testing.expect(diagnostics.max_abs_after < 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, @floatCast(@as(f32, @floatCast(@log(@as(f64, 2)))))), candidate_bias[2], 1e-7);
}
