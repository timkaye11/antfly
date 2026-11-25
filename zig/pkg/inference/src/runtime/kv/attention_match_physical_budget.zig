const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");

pub const policy = "private_precodec_f16_page_budget_primary_then95_max3_v2";
pub const page_bytes: usize = 16 * 1024;
pub const metadata_bytes_per_row: usize = 20;

pub const Plan = struct {
    primary_percent: u8,
    fallback_percent: u8 = 95,
    maximum_fallbacks: u8 = 3,
    primary_rows: usize,
    fallback_rows: usize,
    ordinary_bytes: usize,
    worst_candidate_bytes: usize,
};

pub fn eligibleFreshF16(is_f16: bool, source_has_metadata: bool, candidate_count: usize) bool {
    return is_f16 and !source_has_metadata and candidate_count > 0 and candidate_count <= 96;
}

/// All staged private encodings share the unchanged pre-codec F16
/// geometry-savings gate. Their codec quality and post-codec physical gates
/// remain independent.
pub fn eligibleFreshPhysicalEncoding(encoding: checkpoint.Encoding, source_has_metadata: bool, candidate_count: usize) bool {
    return (encoding == .f16 or encoding == .fp8 or encoding == .rotated4) and
        !source_has_metadata and candidate_count > 0 and candidate_count <= 96;
}

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b);
}
fn rounded(bytes: usize) !usize {
    const pages = std.math.divCeil(usize, bytes, page_bytes) catch return error.InvalidFitConfig;
    return mul(pages, page_bytes);
}
pub fn candidateHeadBytes(rows: usize, dim: usize) !usize {
    if (rows == 0 or dim == 0) return error.InvalidFitConfig;
    const plane = try rounded(try mul(try mul(rows, dim), 2));
    return add(try mul(plane, 2), try rounded(try mul(rows, metadata_bytes_per_row)));
}
pub fn ordinaryHeadBytes(rows: usize, dim: usize) !usize {
    if (rows == 0 or dim == 0) return error.InvalidFitConfig;
    // Two ordinary F16 planes: K and V, two bytes per scalar.
    return mul(try mul(rows, dim), 4);
}

pub fn freshContiguousHead(head: checkpoint.Head) bool {
    if (head.compacted_until != 0 or head.fp8.rows != 0 or head.compressed.spans.len != 0 or head.compressed.keys.len != 0 or head.compressed.values.len != 0 or
        head.positions.len == 0 or head.positions.len != head.coverage_ends.len or head.positions.len != head.bias.len)
        return false;
    for (head.positions, head.coverage_ends, head.bias, 0..) |position, coverage, bias, i| {
        if (position != i or coverage != i + 1 or bias != 0) return false;
    }
    return true;
}
fn rowsAt(rows: usize, fixed: usize, percent: u8) !usize {
    if (fixed >= rows or percent < 50 or percent > 95) return error.InvalidFitConfig;
    const middle = std.math.divCeil(usize, rows - fixed, 2) catch return error.InvalidFitConfig;
    return add(fixed, try mul(middle, percent) / 100);
}
pub fn deriveUniform(rows: usize, dim: usize, heads: usize, fixed: usize, maximum_fallbacks: u8) !Plan {
    if (heads < 2 or maximum_fallbacks >= heads or maximum_fallbacks > 3) return error.InvalidFitConfig;
    const ordinary = try mul(heads, try ordinaryHeadBytes(rows, dim));
    const fallback_rows = try rowsAt(rows, fixed, 95);
    const fallback_cost = try candidateHeadBytes(fallback_rows, dim);
    const exact_cost = try candidateHeadBytes(rows, dim);
    var percent: u8 = 95;
    while (true) : (percent -= 1) {
        const primary_rows = try rowsAt(rows, fixed, percent);
        const primary_heads = heads - 1 - maximum_fallbacks;
        var candidate = try add(exact_cost, try mul(maximum_fallbacks, fallback_cost));
        candidate = try add(candidate, try mul(primary_heads, try candidateHeadBytes(primary_rows, dim)));
        if (try mul(candidate, 3) <= try mul(ordinary, 2)) return .{ .primary_percent = percent, .maximum_fallbacks = maximum_fallbacks, .primary_rows = primary_rows, .fallback_rows = fallback_rows, .ordinary_bytes = ordinary, .worst_candidate_bytes = candidate };
        if (percent == 50) return error.CompactionPhysicalBudgetUnsupported;
    }
}

/// Returns no derived private budget when the predeclared fallback reserve
/// cannot fit this head geometry. The caller must retain its established p95
/// route; arithmetic overflow and other unexpected failures still propagate.
pub fn deriveUniformIfFeasible(rows: usize, dim: usize, heads: usize, fixed: usize, maximum_fallbacks: u8) !?Plan {
    if (heads < 2 or maximum_fallbacks > 3 or fixed >= rows or dim == 0) return error.InvalidFitConfig;
    if (maximum_fallbacks >= heads) return null;
    return deriveUniform(rows, dim, heads, fixed, maximum_fallbacks) catch |err| switch (err) {
        error.CompactionPhysicalBudgetUnsupported => null,
        else => return err,
    };
}

/// Opt-in topology-aware proposal. Existing generation continues to call
/// deriveUniformIfFeasible and therefore retains its exact fixed-count behavior.
/// A future separately authenticated policy may request the largest fallback
/// reserve, up to its declared maximum, that preserves one exact head and the
/// unchanged page-rounded global 1.5x gate. Every nonzero fallback remains p95.
pub fn deriveUniformWithFallbackSearchIfFeasible(rows: usize, dim: usize, heads: usize, fixed: usize, maximum_fallbacks: u8) !?Plan {
    if (heads < 2 or maximum_fallbacks > 3 or fixed >= rows or dim == 0) return error.InvalidFitConfig;
    var fallbacks: u8 = @intCast(@min(@as(usize, maximum_fallbacks), heads - 1));
    while (true) : (fallbacks -= 1) {
        const plan = deriveUniform(rows, dim, heads, fixed, fallbacks) catch |err| switch (err) {
            error.CompactionPhysicalBudgetUnsupported => null,
            else => return err,
        };
        if (plan) |value| return value;
        if (fallbacks == 0) return null;
    }
}

test "context Case1 derives p91 with three p95 reserves" {
    const p = try deriveUniform(7936, 512, 8, 2176, 3);
    try std.testing.expectEqual(@as(u8, 91), p.primary_percent);
    try std.testing.expectEqual(@as(usize, 4796), p.primary_rows);
    try std.testing.expectEqual(@as(usize, 4912), p.fallback_rows);
    try std.testing.expectEqual(@as(usize, 130023424), p.ordinary_bytes);
    try std.testing.expectEqual(@as(usize, 86605824), p.worst_candidate_bytes);
    try std.testing.expect(p.worst_candidate_bytes * 3 <= p.ordinary_bytes * 2);
}

test "context physical budget rejects unsupported shapes and overflows" {
    try std.testing.expectError(error.InvalidFitConfig, deriveUniform(1, 512, 8, 1, 3));
    try std.testing.expectError(error.InvalidFitConfig, deriveUniform(7936, 512, 3, 2176, 3));
    try std.testing.expectError(error.Overflow, ordinaryHeadBytes(std.math.maxInt(usize), 512));
}

test "context optional physical budget preserves E2B three-head p95 route" {
    // Gemma4 E2B has three global donors and one global KV head per donor.
    // Reserving one exact head plus three p95 fallbacks is impossible, so the
    // derived physical tuple must remain absent rather than abort preparation.
    try std.testing.expectEqual(@as(?Plan, null), try deriveUniformIfFeasible(7936, 512, 3, 2176, 3));
    try std.testing.expect((try deriveUniformIfFeasible(7936, 512, 8, 2176, 3)) != null);
    try std.testing.expectError(error.Overflow, deriveUniformIfFeasible(std.math.maxInt(usize), 512, 8, 1, 3));
    try std.testing.expectError(error.InvalidFitConfig, deriveUniformIfFeasible(7936, 512, 1, 2176, 0));
    try std.testing.expectError(error.InvalidFitConfig, deriveUniformIfFeasible(7936, 512, 8, 2176, 4));
    try std.testing.expectError(error.InvalidFitConfig, deriveUniformIfFeasible(7936, 512, 3, 7936, 3));
    try std.testing.expectError(error.InvalidFitConfig, deriveUniformIfFeasible(7936, 0, 3, 2176, 3));
}

test "context opt-in fallback search proposes E2 while default remains absent" {
    try std.testing.expectEqual(@as(?Plan, null), try deriveUniformIfFeasible(7936, 512, 3, 2176, 3));
    const e2 = (try deriveUniformWithFallbackSearchIfFeasible(7936, 512, 3, 2176, 3)).?;
    try std.testing.expectEqual(@as(u8, 0), e2.maximum_fallbacks);
    try std.testing.expectEqual(@as(u8, 59), e2.primary_percent);
    try std.testing.expectEqual(@as(u8, 95), e2.fallback_percent);
    try std.testing.expectEqual(@as(usize, 3875), e2.primary_rows);
    try std.testing.expect(e2.worst_candidate_bytes * 3 <= e2.ordinary_bytes * 2);

    const existing_e4 = (try deriveUniformIfFeasible(7936, 512, 8, 2176, 3)).?;
    const proposed_e4 = (try deriveUniformWithFallbackSearchIfFeasible(7936, 512, 8, 2176, 3)).?;
    try std.testing.expectEqualDeep(existing_e4, proposed_e4);
}

test "context opt-in fallback search is route-neutral at 32K" {
    const p = (try deriveUniformWithFallbackSearchIfFeasible(32512, 512, 8, 2176, 3)).?;
    try std.testing.expectEqual(@as(u8, 3), p.maximum_fallbacks);
    try std.testing.expectEqual(@as(u8, 95), p.fallback_percent);
    try std.testing.expect(p.primary_rows <= p.fallback_rows);
    try std.testing.expect(p.fallback_rows < 32512);
    try std.testing.expect(p.worst_candidate_bytes * 3 <= p.ordinary_bytes * 2);
    const exact = try candidateHeadBytes(32512, 512);
    const fallback = try candidateHeadBytes(p.fallback_rows, 512);
    const primary = try candidateHeadBytes(p.primary_rows, 512);
    try std.testing.expectEqual(exact + 3 * fallback + 4 * primary, p.worst_candidate_bytes);
}

test "context activation preserves bounded codecs and metadata cycles" {
    try std.testing.expect(eligibleFreshF16(true, false, 69));
    try std.testing.expect(!eligibleFreshF16(true, false, 99));
    try std.testing.expect(!eligibleFreshF16(false, false, 69)); // fp8/rotated4
    try std.testing.expect(!eligibleFreshF16(true, true, 69));
    try std.testing.expect(eligibleFreshPhysicalEncoding(.f16, false, 69));
    try std.testing.expect(eligibleFreshPhysicalEncoding(.fp8, false, 69));
    try std.testing.expect(eligibleFreshPhysicalEncoding(.rotated4, false, 69));
    try std.testing.expect(!eligibleFreshPhysicalEncoding(.f32, false, 69));
    try std.testing.expect(!eligibleFreshPhysicalEncoding(.fp8, false, 0));
    try std.testing.expect(!eligibleFreshPhysicalEncoding(.fp8, false, 99));
    try std.testing.expect(!eligibleFreshPhysicalEncoding(.fp8, true, 69));
}

test "context fresh authority rejects prior fitted and storage annotations" {
    const positions = [_]u64{ 0, 1, 2 };
    const coverage = [_]u64{ 1, 2, 3 };
    const bias = [_]f32{ 0, 0, 0 };
    const fresh: checkpoint.Head = .{ .positions = &positions, .coverage_ends = &coverage, .bias = &bias };
    try std.testing.expect(freshContiguousHead(fresh));
    var changed = fresh;
    changed.compacted_until = 1;
    try std.testing.expect(!freshContiguousHead(changed));
    changed = fresh;
    changed.fp8.rows = 1;
    try std.testing.expect(!freshContiguousHead(changed));
    const nonzero = [_]f32{ 0, 0.5, 0 };
    changed = fresh;
    changed.bias = &nonzero;
    try std.testing.expect(!freshContiguousHead(changed));
    const gapped = [_]u64{ 0, 2, 3 };
    changed = fresh;
    changed.positions = &gapped;
    try std.testing.expect(!freshContiguousHead(changed));
}
