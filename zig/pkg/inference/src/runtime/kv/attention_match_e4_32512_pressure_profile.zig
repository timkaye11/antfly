const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const pressure = @import("attention_match_train_pressure_exact_head_reservation.zig");
const shape_authority = @import("attention_match_e4_32512_shape_authority.zig");
const physical_budget = @import("attention_match_physical_budget.zig");

pub const policy = "e4_32512_train_pressure_exact_head_v1|source32512|middle30336|ranges119|last128|p91|fallback95|max3|one_exact|page_train_cache|heldout_values_forbidden";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const Binding = struct {
    shape: shape_authority.Binding,
    expected_shape_sha256: [32]u8,
};

pub fn ranges() [shape_authority.range_count]admission.Range {
    var out: [shape_authority.range_count]admission.Range = undefined;
    var start: usize = shape_authority.prefix_rows;
    for (&out, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == out.len) 128 else shape_authority.chunk_rows };
        start += range.rows;
    }
    return out;
}

fn validateBinding(binding: Binding) !shape_authority.Validated {
    if (!std.crypto.timing_safe.eql([32]u8, binding.shape.expected_digest, binding.expected_shape_sha256))
        return error.UnsupportedShape;
    return shape_authority.validate(binding.shape);
}

fn identityLess(_: void, a: pressure.HeadIdentity, b: pressure.HeadIdentity) bool {
    return a.donor_layer < b.donor_layer or (a.donor_layer == b.donor_layer and a.head_index < b.head_index);
}

pub fn select(allocator: std.mem.Allocator, inputs: []const pressure.HeadInput, binding: Binding, maximum_workspace_bytes: usize, control: solver.Control) !pressure.Result {
    const validated = try validateBinding(binding);
    if (inputs.len != shape_authority.eligible_heads) return error.InvalidFitConfig;
    const topology = (try physical_budget.deriveUniformWithFallbackSearchIfFeasible(
        shape_authority.source_rows,
        shape_authority.dim,
        shape_authority.eligible_heads,
        shape_authority.prefix_rows + shape_authority.recent_rows,
        shape_authority.maximum_fallbacks,
    )) orelse return error.PhysicalBudgetUnsupported;
    if (topology.fallback_percent != shape_authority.fallback_percent or
        topology.maximum_fallbacks != shape_authority.maximum_fallbacks)
        return error.PhysicalBudgetUnsupported;
    // The authenticated 32K route declares p91.  The generic helper returns
    // the largest feasible percentage, so derive the declared plan explicitly
    // and independently retain the unchanged page-rounded 1.5x physical gate.
    const middle_target_rows = std.math.divCeil(usize, validated.middle_rows, 2) catch return error.InvalidFitConfig;
    const primary_rows = shape_authority.prefix_rows + shape_authority.recent_rows +
        (try std.math.mul(usize, middle_target_rows, shape_authority.primary_percent)) / 100;
    const exact_cost = try physical_budget.candidateHeadBytes(shape_authority.source_rows, shape_authority.dim);
    const fallback_cost = try physical_budget.candidateHeadBytes(topology.fallback_rows, shape_authority.dim);
    const primary_cost = try physical_budget.candidateHeadBytes(primary_rows, shape_authority.dim);
    var worst_candidate_bytes = try std.math.add(usize, exact_cost, try std.math.mul(usize, shape_authority.maximum_fallbacks, fallback_cost));
    worst_candidate_bytes = try std.math.add(usize, worst_candidate_bytes, try std.math.mul(usize, shape_authority.eligible_heads - 1 - shape_authority.maximum_fallbacks, primary_cost));
    if (try std.math.mul(usize, worst_candidate_bytes, 3) > try std.math.mul(usize, topology.ordinary_bytes, 2))
        return error.PhysicalBudgetUnsupported;
    var identities: [shape_authority.eligible_heads]pressure.HeadIdentity = undefined;
    for (inputs, &identities) |input, *identity| identity.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index };
    std.mem.sort(pressure.HeadIdentity, &identities, {}, identityLess);
    const range_values = ranges();
    var result = try pressure.selectForShape(allocator, inputs, .{
        .source_rows = shape_authority.source_rows,
        .prefix_rows = shape_authority.prefix_rows,
        .middle_rows = validated.middle_rows,
        .recent_rows = shape_authority.recent_rows,
        .chunk_rows = shape_authority.chunk_rows,
        .ranges = &range_values,
        .eligible_head_identities = &identities,
        .primary_percent = shape_authority.primary_percent,
        .planned_exact_heads = 1,
        .physical_plan = .{
            .flat_geometry = true,
            .primary_percent = shape_authority.primary_percent,
            .planned_exact_heads = 1,
            .source_physical_bytes = topology.ordinary_bytes,
            .maximum_candidate_physical_bytes = worst_candidate_bytes,
            .minimum_global_savings_ratio = 1.5,
        },
    }, maximum_workspace_bytes, control);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&policy_sha256);
    hash.update(&validated.digest);
    hash.update(&result.evidence_sha256);
    var frame: [24]u8 = undefined;
    std.mem.writeInt(u64, frame[0..8], @intCast(topology.ordinary_bytes), .little);
    std.mem.writeInt(u64, frame[8..16], @intCast(worst_candidate_bytes), .little);
    std.mem.writeInt(u64, frame[16..24], @bitCast(@as(f64, 1.5)), .little);
    hash.update(&frame);
    result.policy_sha256 = policy_sha256;
    result.evidence_sha256 = hash.finalResult();
    return result;
}

fn testBinding(seed: u8) Binding {
    var shape: shape_authority.Binding = .{
        .source_freeze_sha256 = @splat(seed),
        .configuration_sha256 = @splat(seed + 1),
        .pressure_policy_sha256 = policy_sha256,
        .packed_source_policy_sha256 = @splat(seed + 2),
        .expected_digest = @splat(0),
    };
    shape.expected_digest = shape_authority.digest(shape) catch unreachable;
    return .{ .shape = shape, .expected_shape_sha256 = shape.expected_digest };
}

test "E4 32512 pressure ranges cover middle with ragged final128" {
    const values = ranges();
    try std.testing.expectEqual(@as(usize, 119), values.len);
    try std.testing.expectEqual(@as(usize, 128), values[0].start);
    try std.testing.expectEqual(@as(usize, 128), values[118].rows);
    try std.testing.expectEqual(@as(usize, 30_336), values[118].start);
    var total: usize = 0;
    for (values) |range| total += range.rows;
    try std.testing.expectEqual(@as(usize, 30_336), total);
    try std.testing.expectEqual(@as(usize, 30_464), values[118].start + values[118].rows);
}

test "E4 32512 pressure binds distinct policy and exact physical gate" {
    const maxima: [shape_authority.source_rows]f64 = @splat(0.01);
    var inputs: [shape_authority.eligible_heads]pressure.HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    var result = try select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 8), result.heads.len);
    try std.testing.expectEqualSlices(u8, &policy_sha256, &result.policy_sha256);
    try std.testing.expect(!std.mem.eql(u8, &pressure.shape_policy_sha256, &result.policy_sha256));
}

test "E4 32512 pressure rejects stale shape cap and cancellation" {
    const maxima: [shape_authority.source_rows]f64 = @splat(0);
    var inputs: [shape_authority.eligible_heads]pressure.HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    var stale = testBinding(1);
    stale.expected_shape_sha256[0] +%= 1;
    try std.testing.expectError(error.UnsupportedShape, select(std.testing.allocator, &inputs, stale, 1 << 20, .{}));
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, &inputs, testBinding(1), 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{ .check_fn = Cancel.check }));
}
