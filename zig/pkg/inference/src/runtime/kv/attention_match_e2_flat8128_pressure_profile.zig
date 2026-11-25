//! Default-off E2/8128 TRAIN-pressure profile. It never reads heldout values.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const pressure = @import("attention_match_train_pressure_exact_head_reservation.zig");
const shape_authority = @import("attention_match_e2_flat_shape_authority.zig");
const topology = @import("attention_match_topology_physical_proposal.zig");

pub const schema = "antfly.context.resident-native-hybrid-e2-8128.pressure-profile.v1";
pub const policy = "e2_flat8128_train_pressure_exact_head_v3|heads=4:0,9:0,14:0|consumer_batches=56|per_owner=8,8,40|source8128|ranges24|last64|primary60|fallbacks0|census=2adaptive+1exact|heldout=forbidden|activation=disabled";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};
pub const Binding = struct { shape: shape_authority.Shape, expected_shape_sha256: [32]u8 };
pub const Census = struct {
    adaptive_heads: u8 = 2,
    planned_exact_heads: u8 = 1,
    fallback_heads: u8 = 0,
    eligible_heads: u8 = 3,
    exact: shape_authority.HeadIdentity,
};

pub fn ranges() [24]admission.Range {
    var out: [24]admission.Range = undefined;
    var start: usize = 128;
    for (&out, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == out.len) 64 else 256 };
        start += range.rows;
    }
    return out;
}
fn validateBinding(binding: Binding) !struct { validated: shape_authority.Validated, physical: topology.Evidence } {
    const validated = try shape_authority.validate(binding.shape);
    if (!std.crypto.timing_safe.eql([32]u8, validated.digest, binding.expected_shape_sha256)) return error.UnsupportedShape;
    const heads = [_]topology.HeadIdentity{ .{ .donor_layer = 4, .kv_head = 0 }, .{ .donor_layer = 9, .kv_head = 0 }, .{ .donor_layer = 14, .kv_head = 0 } };
    const physical = try topology.propose(.{ .authority_sha256 = binding.shape.authority_sha256, .topology = .e2_global, .source_rows = 8128, .dim = 512, .fixed_rows = 2176, .heads = &heads });
    if (physical.primary_percent != 60 or physical.maximum_fallbacks != 0 or physical.planned_exact_heads != 1 or physical.primary_rows != 3961 or physical.ordinary_bytes != 49_938_432 or physical.worst_candidate_bytes != 33_226_752) return error.PhysicalBudgetUnsupported;
    return .{ .validated = validated, .physical = physical };
}

pub fn select(allocator: std.mem.Allocator, inputs: []const pressure.HeadInput, binding: Binding, maximum_workspace_bytes: usize, control: solver.Control) !pressure.Result {
    const bound = try validateBinding(binding);
    if (inputs.len != 3) return error.InvalidFitConfig;
    const ids = [_]pressure.HeadIdentity{ .{ .donor_layer = 4, .head_index = 0 }, .{ .donor_layer = 9, .head_index = 0 }, .{ .donor_layer = 14, .head_index = 0 } };
    const range_values = ranges();
    var result = try pressure.selectForShape(allocator, inputs, .{
        .source_rows = 8128,
        .prefix_rows = 128,
        .middle_rows = 5952,
        .recent_rows = 2048,
        .chunk_rows = 256,
        .ranges = &range_values,
        .eligible_head_identities = &ids,
        .primary_percent = 60,
        .planned_exact_heads = 1,
        .physical_plan = .{ .flat_geometry = true, .primary_percent = 60, .planned_exact_heads = 1, .source_physical_bytes = bound.physical.ordinary_bytes, .maximum_candidate_physical_bytes = bound.physical.worst_candidate_bytes, .minimum_global_savings_ratio = 1.5 },
    }, maximum_workspace_bytes, control);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&policy_sha256);
    hash.update(&bound.validated.digest);
    hash.update(&bound.physical.evidence_sha256);
    hash.update(&result.evidence_sha256);
    result.policy_sha256 = policy_sha256;
    result.evidence_sha256 = hash.finalResult();
    return result;
}
pub fn census(result: pressure.Result) !Census {
    for (shape_authority.canonical_heads) |identity| {
        if (identity.donor_layer == result.chosen_donor_layer and identity.head_index == result.chosen_head_index)
            return .{ .exact = identity };
    }
    return error.InvalidFitConfig;
}
fn testBinding(byte: u8) Binding {
    const shape = shape_authority.Shape{ .authority_sha256 = @splat(byte) };
    const v = shape_authority.validate(shape) catch unreachable;
    return .{ .shape = shape, .expected_shape_sha256 = v.digest };
}

test "E2 profile binds 60 percent zero fallback physical tuple and canonical pressure ordering" {
    const low: [8128]f64 = @splat(0.001);
    var high: [8128]f64 = @splat(0.001);
    @memset(high[128..384], 0.025);
    const inputs = [_]pressure.HeadInput{ .{ .donor_layer = 14, .head_index = 0, .maximum_probability = &low }, .{ .donor_layer = 4, .head_index = 0, .maximum_probability = &low }, .{ .donor_layer = 9, .head_index = 0, .maximum_probability = &high } };
    var result = try select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 9), result.chosen_donor_layer);
    try std.testing.expectEqual(@as(usize, 0), result.chosen_head_index);
    const actual = try census(result);
    try std.testing.expectEqual(@as(u8, 2), actual.adaptive_heads);
    try std.testing.expectEqual(@as(u8, 1), actual.planned_exact_heads);
    try std.testing.expectEqual(@as(u8, 0), actual.fallback_heads);
    try std.testing.expectEqual(@as(u8, 3), actual.eligible_heads);
    try std.testing.expectEqualSlices(u8, &policy_sha256, &result.policy_sha256);
    try std.testing.expect(!std.mem.eql(u8, &pressure.shape_policy_sha256, &result.policy_sha256));
}
test "E2 profile rejects authority identity cap allocation and cancellation failures" {
    const maxima: [8128]f64 = @splat(0.001);
    var inputs = [_]pressure.HeadInput{ .{ .donor_layer = 4, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 9, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 14, .head_index = 0, .maximum_probability = &maxima } };
    var wrong = testBinding(1);
    wrong.expected_shape_sha256 = @splat(9);
    try std.testing.expectError(error.UnsupportedShape, select(std.testing.allocator, &inputs, wrong, 1 << 20, .{}));
    inputs[2].donor_layer = 15;
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{}));
    inputs[2].donor_layer = 14;
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.allocator, &inputs, testBinding(1), 1, .{}));
    try std.testing.expectError(error.OutOfMemory, select(std.testing.failing_allocator, &inputs, testBinding(1), 1 << 20, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{ .check_fn = Cancel.check }));
}
