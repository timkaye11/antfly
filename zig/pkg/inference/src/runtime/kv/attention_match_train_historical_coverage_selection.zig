//! TRAIN-only H512 -> H256 selector: weighted critical-key coverage, then Q diversity.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_historical_weighted_critical_coverage_then_normalized_q_kcenter_v1;threshold=0.025;baseline=primary+future;role_exclusion=logical_heldout_position;stable_ties=chronological;serialization=chronological";
pub const policy_digest: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};
pub const max_rows = 512;
pub const max_dim = 512;
pub const max_selected = 256;
pub const max_critical_keys_per_query = 40;

pub const Input = struct {
    queries: []const f32,
    positions: []const u64,
    heldout_positions: []const u64,
    dim: usize,
    selected_count: usize,
    source_key_weights: []const f64,
    baseline_covered_keys: []const u32,
    candidate_offsets: []const u32,
    candidate_keys: []const u32,
};
pub const Evidence = struct {
    eligible_rows: usize,
    excluded_rows: usize,
    baseline_covered_key_count: usize,
    newly_covered_key_count: usize,
    coverage_selected_rows: usize,
    kcenter_selected_rows: usize,
    baseline_covered_weight: f64,
    final_covered_weight: f64,
    selection_digest: [32]u8,
};
pub const Result = struct {
    allocator: std.mem.Allocator,
    indices: []u32,
    coverage_indices: []u32,
    evidence: Evidence,
    pub fn deinit(self: *Result) void {
        self.allocator.free(self.coverage_indices);
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
pub fn workspaceBytes(rows: usize, dim: usize, source_rows: usize, selected_count: usize) !usize {
    var bytes = try mul(try mul(rows, dim), @sizeOf(f64));
    bytes = std.math.add(usize, bytes, try mul(rows, @sizeOf(u32) + @sizeOf(f64) + 2 * @sizeOf(bool))) catch return error.CompactionWorkspaceLimit;
    bytes = std.math.add(usize, bytes, source_rows) catch return error.CompactionWorkspaceLimit;
    return std.math.add(usize, bytes, try mul(selected_count, 2 * @sizeOf(u32))) catch error.CompactionWorkspaceLimit;
}
fn excluded(position: u64, heldout: []const u64) bool {
    for (heldout) |value| if (value == position) return true;
    return false;
}
fn keysFor(input: Input, row: usize) ![]const u32 {
    const begin: usize = input.candidate_offsets[row];
    const end: usize = input.candidate_offsets[row + 1];
    if (end < begin or end > input.candidate_keys.len or end - begin > max_critical_keys_per_query) return error.InvalidTrainingAugmentation;
    return input.candidate_keys[begin..end];
}
fn addCenter(normalized: []const f64, chosen: []bool, nearest: []f64, dim: usize, center: usize) !void {
    if (chosen[center]) return;
    chosen[center] = true;
    for (0..chosen.len) |candidate| if (!chosen[candidate]) {
        var similarity: f64 = 0;
        for (0..dim) |column| similarity += normalized[candidate * dim + column] * normalized[center * dim + column];
        if (!std.math.isFinite(similarity)) return error.InvalidTrainingAugmentation;
        nearest[candidate] = @max(nearest[candidate], similarity);
    };
    nearest[center] = std.math.inf(f64);
}

pub fn select(allocator: std.mem.Allocator, input: Input, cap: usize, control: solver.Control) !Result {
    const rows = input.positions.len;
    if (rows < 2 or rows > max_rows or input.dim == 0 or input.dim > max_dim or input.selected_count == 0 or input.selected_count > max_selected or input.selected_count > rows or input.queries.len != try mul(rows, input.dim) or input.source_key_weights.len == 0 or input.candidate_offsets.len != rows + 1 or input.candidate_offsets[0] != 0 or input.candidate_offsets[rows] != input.candidate_keys.len) return error.InvalidTrainingAugmentation;
    if (try workspaceBytes(rows, input.dim, input.source_key_weights.len, input.selected_count) > cap) return error.CompactionWorkspaceLimit;
    for (input.positions[1..], input.positions[0 .. rows - 1]) |next, previous| if (next <= previous) return error.InvalidTrainingAugmentation;
    if (input.heldout_positions.len > 1) {
        for (input.heldout_positions[1..], input.heldout_positions[0 .. input.heldout_positions.len - 1]) |next, previous| if (next <= previous) return error.InvalidTrainingAugmentation;
    }
    for (input.source_key_weights) |weight| if (!std.math.isFinite(weight) or weight < 0 or weight > 1) return error.InvalidTrainingAugmentation;
    for (input.baseline_covered_keys, 0..) |key, i| if (@as(usize, key) >= input.source_key_weights.len or (i > 0 and input.baseline_covered_keys[i - 1] >= key)) return error.InvalidTrainingAugmentation;
    for (0..rows) |row| for (try keysFor(input, row), 0..) |key, i| if (@as(usize, key) >= input.source_key_weights.len or input.source_key_weights[key] <= 0 or (i > 0 and (try keysFor(input, row))[i - 1] >= key)) return error.InvalidTrainingAugmentation;
    try control.check();

    const eligible = try allocator.alloc(u32, rows);
    defer allocator.free(eligible);
    var eligible_len: usize = 0;
    for (input.positions, 0..) |position, row| if (!excluded(position, input.heldout_positions)) {
        eligible[eligible_len] = @intCast(row);
        eligible_len += 1;
    };
    if (eligible_len < input.selected_count) return error.InsufficientSelectionProgress;
    const erows = eligible[0..eligible_len];
    const normalized = try allocator.alloc(f64, try mul(eligible_len, input.dim));
    defer allocator.free(normalized);
    for (erows, 0..) |raw, ei| {
        if ((ei & 31) == 0) try control.check();
        const row: usize = raw;
        var norm2: f64 = 0;
        for (input.queries[row * input.dim ..][0..input.dim]) |value| {
            if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
            norm2 += @as(f64, value) * value;
        }
        if (!std.math.isFinite(norm2) or norm2 <= 0) return error.InsufficientSelectionProgress;
        const inv = 1 / @sqrt(norm2);
        for (input.queries[row * input.dim ..][0..input.dim], normalized[ei * input.dim ..][0..input.dim]) |value, *out| out.* = @as(f64, value) * inv;
    }
    const covered = try allocator.alloc(bool, input.source_key_weights.len);
    defer allocator.free(covered);
    @memset(covered, false);
    var baseline_weight: f64 = 0;
    for (input.baseline_covered_keys) |raw| {
        covered[raw] = true;
        baseline_weight += input.source_key_weights[raw];
    }
    var final_weight = baseline_weight;
    var newly_covered: usize = 0;
    const chosen = try allocator.alloc(bool, eligible_len);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const output = try allocator.alloc(u32, input.selected_count);
    errdefer allocator.free(output);
    var output_len: usize = 0;
    var coverage_count: usize = 0;
    while (output_len < output.len) {
        try control.check();
        var winner: ?usize = null;
        var best_gain: f64 = 0;
        for (erows, 0..) |row, candidate| if (!chosen[candidate]) {
            var gain: f64 = 0;
            for (try keysFor(input, row)) |key| {
                if (!covered[key]) gain += input.source_key_weights[key];
            }
            if (gain > best_gain) {
                winner = candidate;
                best_gain = gain;
            }
        };
        if (winner == null or !(best_gain > 0)) break;
        const center = winner.?;
        chosen[center] = true;
        output[output_len] = erows[center];
        output_len += 1;
        coverage_count += 1;
        for (try keysFor(input, erows[center])) |key| if (!covered[key]) {
            covered[key] = true;
            final_weight += input.source_key_weights[key];
            newly_covered += 1;
        };
    }
    const coverage_indices = try allocator.dupe(u32, output[0..coverage_count]);
    errdefer allocator.free(coverage_indices);
    std.mem.sort(u32, coverage_indices, {}, std.sort.asc(u32));
    const nearest = try allocator.alloc(f64, eligible_len);
    defer allocator.free(nearest);
    @memset(nearest, -std.math.inf(f64));
    // Coverage winners seed k-center.
    for (0..eligible_len) |center| if (chosen[center]) {
        chosen[center] = false;
        try addCenter(normalized, chosen, nearest, input.dim, center);
    };
    if (output_len < output.len and !chosen[0]) {
        output[output_len] = erows[0];
        output_len += 1;
        try addCenter(normalized, chosen, nearest, input.dim, 0);
    }
    if (output_len < output.len and !chosen[eligible_len - 1]) {
        output[output_len] = erows[eligible_len - 1];
        output_len += 1;
        try addCenter(normalized, chosen, nearest, input.dim, eligible_len - 1);
    }
    while (output_len < output.len) {
        try control.check();
        var winner: ?usize = null;
        var score = std.math.inf(f64);
        for (nearest, 0..) |value, candidate| if (!chosen[candidate] and (winner == null or value < score)) {
            winner = candidate;
            score = value;
        };
        const center = winner orelse return error.InsufficientSelectionProgress;
        if (!std.math.isFinite(score)) return error.InsufficientSelectionProgress;
        output[output_len] = erows[center];
        output_len += 1;
        try addCenter(normalized, chosen, nearest, input.dim, center);
    }
    for (1..output.len) |i| {
        const value = output[i];
        var j = i;
        while (j > 0 and output[j - 1] > value) : (j -= 1) output[j] = output[j - 1];
        output[j] = value;
    }
    try control.check();
    var digest: [32]u8 = undefined;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(std.mem.sliceAsBytes(output));
    hash.final(&digest);
    return .{ .allocator = allocator, .indices = output, .coverage_indices = coverage_indices, .evidence = .{ .eligible_rows = eligible_len, .excluded_rows = rows - eligible_len, .baseline_covered_key_count = input.baseline_covered_keys.len, .newly_covered_key_count = newly_covered, .coverage_selected_rows = coverage_count, .kcenter_selected_rows = output.len - coverage_count, .baseline_covered_weight = baseline_weight, .final_covered_weight = final_weight, .selection_digest = digest } };
}

test "coverage precedes diversity and heldout positions are role-only exclusions" {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1, 0.7, 0.7 };
    var result = try select(std.testing.allocator, .{ .queries = &q, .positions = &.{ 10, 11, 12, 13, 14 }, .heldout_positions = &.{11}, .dim = 2, .selected_count = 3, .source_key_weights = &.{ 0.4, 0.3, 0.2, 0.1 }, .baseline_covered_keys = &.{0}, .candidate_offsets = &.{ 0, 1, 2, 3, 3, 3 }, .candidate_keys = &.{ 1, 2, 3 } }, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 4 }, result.indices);
    try std.testing.expectEqual(@as(usize, 2), result.evidence.coverage_selected_rows);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, result.coverage_indices);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), result.evidence.final_covered_weight, 1e-12);
}

test "empty heldout role set is valid" {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1 };
    var result = try select(std.testing.allocator, .{ .queries = &q, .positions = &.{ 1, 2, 3, 4 }, .heldout_positions = &.{}, .dim = 2, .selected_count = 2, .source_key_weights = &.{0.5}, .baseline_covered_keys = &.{}, .candidate_offsets = &.{ 0, 1, 1, 1, 1 }, .candidate_keys = &.{0} }, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.indices.len);
    try std.testing.expectEqual(@as(usize, 0), result.evidence.excluded_rows);
}

fn allocationCase(allocator: std.mem.Allocator) !void {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1 };
    var result = try select(allocator, .{ .queries = &q, .positions = &.{ 1, 2, 3, 4 }, .heldout_positions = &.{}, .dim = 2, .selected_count = 2, .source_key_weights = &.{0.5}, .baseline_covered_keys = &.{}, .candidate_offsets = &.{ 0, 1, 1, 1, 1 }, .candidate_keys = &.{0} }, 1 << 20, .{});
    result.deinit();
}
test "cancellation and allocation failure publish no partial selection" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, .{ .queries = &.{ 1, 0, 0, 1 }, .positions = &.{ 1, 2 }, .heldout_positions = &.{}, .dim = 2, .selected_count = 1, .source_key_weights = &.{0.5}, .baseline_covered_keys = &.{}, .candidate_offsets = &.{ 0, 1, 1 }, .candidate_keys = &.{0} }, 1 << 20, .{ .check_fn = Cancel.check }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationCase, .{});
}
