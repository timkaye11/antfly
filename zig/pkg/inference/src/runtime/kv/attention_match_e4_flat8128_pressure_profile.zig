const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const pressure = @import("attention_match_train_pressure_exact_head_reservation.zig");
const flat_shape = @import("attention_match_flat_shape_authority.zig");
const physical_budget = @import("attention_match_physical_budget.zig");

pub const policy = "e4_flat8128_train_pressure_exact_head_v1|source8128|middle5952|ranges24|last64|p91|fallback95|max3|one_exact|heldout_values_forbidden";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const Binding = struct {
    shape: flat_shape.Shape,
    expected_shape_sha256: [32]u8,
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

fn validateBinding(binding: Binding) !flat_shape.Validated {
    const validated = try flat_shape.validate(binding.shape);
    if (validated.shape.source_rows != 8128 or validated.range_count != 24 or validated.shape.primary_percent != 91 or validated.shape.fallback_percent != 95 or validated.shape.maximum_fallbacks != 3 or !std.crypto.timing_safe.eql([32]u8, validated.digest, binding.expected_shape_sha256)) return error.UnsupportedShape;
    return validated;
}

fn identityLess(_: void, a: pressure.HeadIdentity, b: pressure.HeadIdentity) bool {
    return a.donor_layer < b.donor_layer or (a.donor_layer == b.donor_layer and a.head_index < b.head_index);
}

pub fn select(allocator: std.mem.Allocator, inputs: []const pressure.HeadInput, binding: Binding, maximum_workspace_bytes: usize, control: solver.Control) !pressure.Result {
    const validated = try validateBinding(binding);
    if (inputs.len != 8) return error.InvalidFitConfig;
    const fixed: usize = 128 + 2048;
    const half_middle = std.math.divCeil(usize, 8128 - fixed, 2) catch return error.InvalidFitConfig;
    const primary_rows = fixed + half_middle * 91 / 100;
    const fallback_rows = fixed + half_middle * 95 / 100;
    const source_physical_bytes = try std.math.mul(usize, 8, try physical_budget.ordinaryHeadBytes(8128, 512));
    var maximum_candidate_physical_bytes = try physical_budget.candidateHeadBytes(8128, 512);
    maximum_candidate_physical_bytes = try std.math.add(usize, maximum_candidate_physical_bytes, try std.math.mul(usize, 3, try physical_budget.candidateHeadBytes(fallback_rows, 512)));
    maximum_candidate_physical_bytes = try std.math.add(usize, maximum_candidate_physical_bytes, try std.math.mul(usize, 4, try physical_budget.candidateHeadBytes(primary_rows, 512)));
    if (primary_rows != 4884 or fallback_rows != 5003 or maximum_candidate_physical_bytes * 3 > source_physical_bytes * 2) return error.PhysicalBudgetUnsupported;
    var identities: [8]pressure.HeadIdentity = undefined;
    for (inputs, &identities) |input, *identity| identity.* = .{ .donor_layer = input.donor_layer, .head_index = input.head_index };
    std.mem.sort(pressure.HeadIdentity, &identities, {}, identityLess);
    const range_values = ranges();
    var result = try pressure.selectForShape(allocator, inputs, .{
        .source_rows = validated.shape.source_rows,
        .prefix_rows = validated.shape.prefix_rows,
        .middle_rows = validated.shape.source_rows - validated.shape.prefix_rows - validated.shape.recent_rows,
        .recent_rows = validated.shape.recent_rows,
        .chunk_rows = validated.shape.chunk_rows,
        .ranges = &range_values,
        .eligible_head_identities = &identities,
        .primary_percent = validated.shape.primary_percent,
        .planned_exact_heads = 1,
        .physical_plan = .{ .flat_geometry = true, .primary_percent = 91, .planned_exact_heads = 1, .source_physical_bytes = source_physical_bytes, .maximum_candidate_physical_bytes = maximum_candidate_physical_bytes, .minimum_global_savings_ratio = 1.5 },
    }, maximum_workspace_bytes, control);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&policy_sha256);
    hash.update(&validated.digest);
    hash.update(&result.evidence_sha256);
    var frame: [24]u8 = undefined;
    std.mem.writeInt(u64, frame[0..8], source_physical_bytes, .little);
    std.mem.writeInt(u64, frame[8..16], maximum_candidate_physical_bytes, .little);
    std.mem.writeInt(u64, frame[16..24], @bitCast(@as(f64, 1.5)), .little);
    hash.update(&frame);
    result.policy_sha256 = policy_sha256;
    result.evidence_sha256 = hash.finalResult();
    return result;
}

fn testBinding(authority: u8) Binding {
    var shape = flat_shape.Shape.sealed7936(@splat(authority));
    shape.source_rows = 8128;
    const validated = flat_shape.validate(shape) catch unreachable;
    return .{ .shape = shape, .expected_shape_sha256 = validated.digest };
}

test "E4 flat8128 ranges cover exactly middle5952 with final64 boundary" {
    const r = ranges();
    try std.testing.expectEqual(@as(usize, 24), r.len);
    try std.testing.expectEqual(@as(usize, 128), r[0].start);
    try std.testing.expectEqual(@as(usize, 256), r[22].rows);
    try std.testing.expectEqual(@as(usize, 6016), r[23].start);
    try std.testing.expectEqual(@as(usize, 64), r[23].rows);
    try std.testing.expectEqual(@as(usize, 6080), r[23].start + r[23].rows);
    var middle: usize = 0;
    for (r) |range| middle += range.rows;
    try std.testing.expectEqual(@as(usize, 5952), middle);
}

test "E4 flat8128 profile selects one pressure head and binds distinct policy" {
    const maxima: [8128]f64 = @splat(0.01);
    var inputs: [8]pressure.HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    var result = try select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 8), result.heads.len);
    try std.testing.expectEqual(@as(u32, 5), result.chosen_donor_layer);
    try std.testing.expectEqualSlices(u8, &policy_sha256, &result.policy_sha256);
    try std.testing.expect(!std.mem.eql(u8, &pressure.policy_sha256, &result.policy_sha256));
    for (result.heads) |head| try std.testing.expectEqual(@as(usize, 2708), head.middle_cap_rows);
}

test "E4 flat8128 final64 mass pressure participates in exact-head choice" {
    const low: [8128]f64 = @splat(0);
    var final_chunk_pressure: [8128]f64 = @splat(0);
    @memset(final_chunk_pressure[6016..6080], 0.025);
    var inputs: [8]pressure.HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = if (i == 6) &final_chunk_pressure else &low };
    var result = try select(std.testing.allocator, &inputs, testBinding(1), 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(u32, 11), result.chosen_donor_layer);
    var found = false;
    for (result.heads) |head| if (head.donor_layer == 11) {
        try std.testing.expect(head.critical_rows >= 64);
        found = true;
    };
    try std.testing.expect(found);
}

test "E4 flat8128 rejects legacy shape wrong authority cap and cancellation" {
    const maxima: [8128]f64 = @splat(0);
    var inputs: [8]pressure.HeadInput = undefined;
    for (&inputs, 0..) |*input, i| input.* = .{ .donor_layer = @intCast(5 + i), .head_index = 0, .maximum_probability = &maxima };
    var legacy = testBinding(1);
    legacy.shape.source_rows = 7936;
    try std.testing.expectError(error.UnsupportedShape, select(std.testing.allocator, &inputs, legacy, 1 << 20, .{}));
    var wrong = testBinding(1);
    wrong.expected_shape_sha256 = @splat(9);
    try std.testing.expectError(error.UnsupportedShape, select(std.testing.allocator, &inputs, wrong, 1 << 20, .{}));
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, &inputs, testBinding(1), 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.failing_allocator, &inputs, testBinding(1), 1 << 20, .{ .check_fn = Cancel.check }));
}

test "legacy 7936 policy digest is frozen" {
    const expected = "53098b9174b8c0a616522452df2db7f3dc4aeb814a444f4f70b07fc26a4a72a4";
    const actual = std.fmt.bytesToHex(pressure.policy_sha256, .lower);
    try std.testing.expectEqualStrings(expected, &actual);
}
