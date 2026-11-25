//! Default-off E2/8128 TRAIN-only exact-head pressure derivation.
//! This deliberately does not alter the E4 `[8]` historical-coverage evidence.
const std = @import("std");
const solver = @import("attention_match.zig");
const admission = @import("attention_match_cumulative_cache.zig");
const pressure = @import("attention_match_train_pressure_exact_head_reservation.zig");

pub const policy = "e2_historical_pressure_derivation_v1|heads=4:0,9:0,14:0|source8128|prefix128|middle5952|recent2048|ranges24|primary60|planned_exact1|adaptive_rows_cap3961|fallbacks0|heldout=forbidden";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};
pub const formal_policy = "e2_historical_pressure_derivation_v2|source_rows=7936,32512|geometry=prefix128,recent2048,chunk256|primary60|planned_exact1|fallbacks0|physical=captured_inventory_and_actual_candidate";
pub const head_identities = [_]pressure.HeadIdentity{ .{ .donor_layer = 4, .head_index = 0 }, .{ .donor_layer = 9, .head_index = 0 }, .{ .donor_layer = 14, .head_index = 0 } };
pub const source_rows: usize = 8128;
pub const maximum_adaptive_rows: usize = 3961;
pub const exact_head_bytes: usize = 16_809_984;
pub const adaptive_head_bytes: usize = 8_208_384;
pub const source_physical_bytes: usize = 49_938_432;
pub const maximum_candidate_physical_bytes: usize = 33_226_752;

pub const Evidence = struct {
    policy_sha256: [32]u8,
    evidence_sha256: [32]u8,
    chosen_exact_donor_layer: u32,
    chosen_exact_head_index: usize,
    pressure_infeasible_count: usize,
    exact_head_pressure: [3]pressure.HeadEvidence,
};
pub const Census = struct { adaptive_heads: usize, exact_heads: usize, fallback_heads: usize, adaptive_rows: [2]usize, adaptive_physical_bytes: [2]usize, exact_physical_bytes: usize };
fn putU64(hash: *std.crypto.hash.sha2.Sha256, value: usize) !void {
    const cast = std.math.cast(u64, value) orelse return error.InvalidFitConfig;
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, cast, .little);
    hash.update(&bytes);
}
fn putU32(hash: *std.crypto.hash.sha2.Sha256, value: u32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    hash.update(&bytes);
}
pub fn ranges() [24]admission.Range {
    var out: [24]admission.Range = undefined;
    var at: usize = 128;
    for (&out, 0..) |*r, i| {
        r.* = .{ .start = at, .rows = if (i + 1 == out.len) 64 else 256 };
        at += r.rows;
    }
    return out;
}
fn shape(range_values: []const admission.Range) pressure.Shape {
    return .{ .source_rows = source_rows, .prefix_rows = 128, .middle_rows = 5952, .recent_rows = 2048, .chunk_rows = 256, .ranges = range_values, .eligible_head_identities = &head_identities, .primary_percent = 60, .planned_exact_heads = 1, .physical_plan = .{ .flat_geometry = true, .primary_percent = 60, .planned_exact_heads = 1, .source_physical_bytes = source_physical_bytes, .maximum_candidate_physical_bytes = maximum_candidate_physical_bytes, .minimum_global_savings_ratio = 1.5 } };
}

fn shapeForRows(source_row_count: usize, range_values: []const admission.Range, source_bytes: usize, candidate_bytes: usize) pressure.Shape {
    return .{ .source_rows = source_row_count, .prefix_rows = 128, .middle_rows = source_row_count - 128 - 2048, .recent_rows = 2048, .chunk_rows = 256, .ranges = range_values, .eligible_head_identities = &head_identities, .primary_percent = 60, .planned_exact_heads = 1, .physical_plan = .{ .flat_geometry = source_row_count <= 8_128, .primary_percent = 60, .planned_exact_heads = 1, .source_physical_bytes = source_bytes, .maximum_candidate_physical_bytes = candidate_bytes, .minimum_global_savings_ratio = 1.5 } };
}

pub fn deriveForRows(allocator: std.mem.Allocator, inputs: []const pressure.HeadInput, source_row_count: usize, captured_source_physical_bytes: usize, shape_authority_sha256: [32]u8, maximum_workspace_bytes: usize, control: solver.Control) !Evidence {
    if (source_row_count == 8_128) return derive(allocator, inputs, shape_authority_sha256, maximum_workspace_bytes, control);
    if (source_row_count != 7_936 and source_row_count != 32_512) return error.InvalidFitConfig;
    const range_count = std.math.divCeil(usize, source_row_count - 128 - 2048, 256) catch return error.InvalidFitConfig;
    const owned_ranges = try allocator.alloc(admission.Range, range_count);
    defer allocator.free(owned_ranges);
    var at: usize = 128;
    for (owned_ranges, 0..) |*range, index| {
        const remaining = source_row_count - 2048 - at;
        range.* = .{ .start = at, .rows = @min(@as(usize, 256), remaining) };
        if (range.rows == 0 or (index + 1 != owned_ranges.len and range.rows != 256)) return error.InvalidFitConfig;
        at += range.rows;
    }
    if (captured_source_physical_bytes == 0) return error.InvalidFitConfig;
    const twice_source = std.math.mul(usize, captured_source_physical_bytes, 2) catch return error.InvalidFitConfig;
    const candidate_bytes = twice_source / 3;
    var selected = try pressure.selectForShape(allocator, inputs, shapeForRows(source_row_count, owned_ranges, captured_source_physical_bytes, candidate_bytes), maximum_workspace_bytes, control);
    defer selected.deinit();
    if (selected.heads.len != 3) return error.InvalidFitConfig;
    var heads: [3]pressure.HeadEvidence = undefined;
    @memcpy(&heads, selected.heads);
    var policy_hash = std.crypto.hash.sha2.Sha256.init(.{});
    policy_hash.update(formal_policy);
    try putU64(&policy_hash, source_row_count);
    policy_hash.update(&shape_authority_sha256);
    policy_hash.update(&selected.policy_sha256);
    const formal_policy_sha256 = policy_hash.finalResult();
    var evidence_hash = std.crypto.hash.sha2.Sha256.init(.{});
    evidence_hash.update(&formal_policy_sha256);
    evidence_hash.update(&selected.evidence_sha256);
    for (heads) |head| {
        putU32(&evidence_hash, head.donor_layer);
        try putU64(&evidence_hash, head.head_index);
        try putU64(&evidence_hash, head.critical_rows);
        evidence_hash.update(&head.calibrated_policy_sha256);
    }
    return .{ .policy_sha256 = formal_policy_sha256, .evidence_sha256 = evidence_hash.finalResult(), .chosen_exact_donor_layer = selected.chosen_donor_layer, .chosen_exact_head_index = selected.chosen_head_index, .pressure_infeasible_count = selected.pressure_infeasible_count, .exact_head_pressure = heads };
}

pub fn derive(allocator: std.mem.Allocator, inputs: []const pressure.HeadInput, shape_authority_sha256: [32]u8, maximum_workspace_bytes: usize, control: solver.Control) !Evidence {
    const zero: [32]u8 = @splat(0);
    try control.check();
    if (inputs.len != 3 or std.mem.eql(u8, &shape_authority_sha256, &zero)) return error.InvalidFitConfig;
    const rs = ranges();
    var selected = try pressure.selectForShape(allocator, inputs, shape(&rs), maximum_workspace_bytes, control);
    defer selected.deinit();
    if (selected.heads.len != 3) return error.InvalidFitConfig;
    var heads: [3]pressure.HeadEvidence = undefined;
    @memcpy(&heads, selected.heads);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&policy_sha256);
    hash.update(&shape_authority_sha256);
    hash.update(&selected.policy_sha256);
    hash.update(&selected.evidence_sha256);
    for (heads) |head| {
        putU32(&hash, head.donor_layer);
        try putU64(&hash, head.head_index);
        var threshold: [8]u8 = undefined;
        std.mem.writeInt(u64, &threshold, @bitCast(head.threshold), .little);
        hash.update(&threshold);
        try putU64(&hash, head.critical_rows);
        try putU64(&hash, head.minimum_middle_rows);
        try putU64(&hash, head.middle_cap_rows);
        hash.update(&.{@intFromBool(head.feasible)});
        hash.update(&head.calibrated_policy_sha256);
    }
    return .{ .policy_sha256 = policy_sha256, .evidence_sha256 = hash.finalResult(), .chosen_exact_donor_layer = selected.chosen_donor_layer, .chosen_exact_head_index = selected.chosen_head_index, .pressure_infeasible_count = selected.pressure_infeasible_count, .exact_head_pressure = heads };
}
pub fn validateCensus(c: Census) !void {
    if (c.adaptive_heads != 2 or c.exact_heads != 1 or c.fallback_heads != 0 or c.exact_physical_bytes != exact_head_bytes) return error.PhysicalBudgetUnsupported;
    var total = c.exact_physical_bytes;
    for (c.adaptive_rows, c.adaptive_physical_bytes) |rows, bytes| {
        if (rows > maximum_adaptive_rows or bytes > adaptive_head_bytes) return error.PhysicalBudgetUnsupported;
        total = std.math.add(usize, total, bytes) catch return error.PhysicalBudgetUnsupported;
    }
    if (total > maximum_candidate_physical_bytes or total * 3 > source_physical_bytes * 2) return error.PhysicalBudgetUnsupported;
}

pub fn validateFormalCensus(c: Census, source_row_count: usize, captured_source_physical_bytes: usize) !void {
    if ((source_row_count != 7_936 and source_row_count != 32_512) or c.adaptive_heads != 2 or c.exact_heads != 1 or c.fallback_heads != 0 or captured_source_physical_bytes == 0)
        return error.PhysicalBudgetUnsupported;
    const per_head_cap = captured_source_physical_bytes / 3;
    var total = c.exact_physical_bytes;
    if (c.exact_physical_bytes == 0 or c.exact_physical_bytes > per_head_cap) return error.PhysicalBudgetUnsupported;
    for (c.adaptive_rows, c.adaptive_physical_bytes) |rows, bytes| {
        if (rows == 0 or rows > source_row_count or bytes == 0 or bytes > per_head_cap) return error.PhysicalBudgetUnsupported;
        total = std.math.add(usize, total, bytes) catch return error.PhysicalBudgetUnsupported;
    }
    const candidate_scaled = std.math.mul(usize, total, 3) catch return error.PhysicalBudgetUnsupported;
    const source_scaled = std.math.mul(usize, captured_source_physical_bytes, 2) catch return error.PhysicalBudgetUnsupported;
    if (candidate_scaled > source_scaled) return error.PhysicalBudgetUnsupported;
}
test "E2 derivation binds cardinality order authority and physical census" {
    const low: [8128]f64 = @splat(0.001);
    var high: [8128]f64 = @splat(0.001);
    @memset(high[128..384], 0.025);
    const inputs = [_]pressure.HeadInput{ .{ .donor_layer = 14, .head_index = 0, .maximum_probability = &low }, .{ .donor_layer = 4, .head_index = 0, .maximum_probability = &low }, .{ .donor_layer = 9, .head_index = 0, .maximum_probability = &high } };
    const e = try derive(std.testing.allocator, &inputs, @splat(1), 1 << 20, .{});
    try std.testing.expectEqual(@as(u32, 9), e.chosen_exact_donor_layer);
    try std.testing.expectEqual(@as(usize, 3), e.exact_head_pressure.len);
    try std.testing.expectEqual(@as(u32, 4), e.exact_head_pressure[0].donor_layer);
    try validateCensus(.{ .adaptive_heads = 2, .exact_heads = 1, .fallback_heads = 0, .adaptive_rows = .{ maximum_adaptive_rows, maximum_adaptive_rows }, .adaptive_physical_bytes = .{ adaptive_head_bytes, adaptive_head_bytes }, .exact_physical_bytes = exact_head_bytes });
    var bad: Census = .{ .adaptive_heads = 2, .exact_heads = 1, .fallback_heads = 0, .adaptive_rows = .{ 3962, 3000 }, .adaptive_physical_bytes = .{ adaptive_head_bytes, adaptive_head_bytes }, .exact_physical_bytes = exact_head_bytes };
    try std.testing.expectError(error.PhysicalBudgetUnsupported, validateCensus(bad));
    bad.adaptive_rows[0] = 3961;
    bad.fallback_heads = 1;
    try std.testing.expectError(error.PhysicalBudgetUnsupported, validateCensus(bad));
}
test "E2 derivation rejects authority cap cancellation and allocation failures" {
    const maxima: [8128]f64 = @splat(0.001);
    const inputs = [_]pressure.HeadInput{ .{ .donor_layer = 4, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 9, .head_index = 0, .maximum_probability = &maxima }, .{ .donor_layer = 14, .head_index = 0, .maximum_probability = &maxima } };
    try std.testing.expectError(error.InvalidFitConfig, derive(std.testing.allocator, &inputs, @splat(0), 1 << 20, .{}));
    try std.testing.expectError(error.CompactionWorkspaceLimit, derive(std.testing.allocator, &inputs, @splat(1), 1, .{}));
    try std.testing.expectError(error.OutOfMemory, derive(std.testing.failing_allocator, &inputs, @splat(1), 1 << 20, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, derive(std.testing.allocator, &inputs, @splat(1), 1 << 20, .{ .check_fn = Cancel.check }));
}

test "formal E2 physical census uses captured source bytes and actual candidate bytes" {
    const mib = 1024 * 1024;
    try validateFormalCensus(.{ .adaptive_heads = 2, .exact_heads = 1, .fallback_heads = 0, .adaptive_rows = .{ 12_000, 12_100 }, .adaptive_physical_bytes = .{ 80 * mib, 80 * mib }, .exact_physical_bytes = 96 * mib }, 32_512, 384 * mib);
    try std.testing.expectError(error.PhysicalBudgetUnsupported, validateFormalCensus(.{ .adaptive_heads = 2, .exact_heads = 1, .fallback_heads = 0, .adaptive_rows = .{ 12_000, 12_100 }, .adaptive_physical_bytes = .{ 80 * mib, 80 * mib }, .exact_physical_bytes = 96 * mib }, 8_128, 384 * mib));
    try std.testing.expectError(error.PhysicalBudgetUnsupported, validateFormalCensus(.{ .adaptive_heads = 2, .exact_heads = 1, .fallback_heads = 0, .adaptive_rows = .{ 12_000, 12_100 }, .adaptive_physical_bytes = .{ 160 * mib, 80 * mib }, .exact_physical_bytes = 96 * mib }, 32_512, 384 * mib));
}
