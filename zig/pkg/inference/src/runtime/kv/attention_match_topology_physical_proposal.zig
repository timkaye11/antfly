//! Default-off topology-aware physical-budget proposal.
//!
//! This module derives evidence only. It neither changes generation options nor
//! authorizes publication, held-out evaluation, fallback attempts, or a profile.
const std = @import("std");
const budget = @import("attention_match_physical_budget.zig");

pub const policy = "topology_aware_one_exact_largest_feasible_p95_fallback_v1|global_gate=1.5|fallback_max=3|publication=disabled";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const Topology = enum(u8) { e2_global = 1, e4_global = 2 };
pub const HeadIdentity = struct { donor_layer: u32, kv_head: u32 };
pub const Request = struct {
    authority_sha256: [32]u8,
    topology: Topology,
    source_rows: usize,
    dim: usize,
    fixed_rows: usize,
    heads: []const HeadIdentity,
    planned_exact_heads: u8 = 1,
    maximum_fallbacks: u8 = 3,
};
pub const Evidence = struct {
    policy_sha256: [32]u8,
    authority_sha256: [32]u8,
    topology: Topology,
    source_rows: usize,
    dim: usize,
    fixed_rows: usize,
    head_count: usize,
    planned_exact_heads: u8,
    primary_percent: u8,
    fallback_percent: u8,
    maximum_fallbacks: u8,
    primary_rows: usize,
    fallback_rows: usize,
    ordinary_bytes: usize,
    worst_candidate_bytes: usize,
    evidence_sha256: [32]u8,
    diagnostic_only: bool = true,
    qualification_credit: bool = false,
};

fn putU64(hash: *std.crypto.hash.sha2.Sha256, value: usize) !void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, std.math.cast(u64, value) orelse return error.Overflow, .little);
    hash.update(&bytes);
}
fn putU32(hash: *std.crypto.hash.sha2.Sha256, value: u32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    hash.update(&bytes);
}
fn nonzero(bytes: [32]u8) bool {
    return !std.mem.allEqual(u8, &bytes, 0);
}
fn less(a: HeadIdentity, b: HeadIdentity) bool {
    return a.donor_layer < b.donor_layer or (a.donor_layer == b.donor_layer and a.kv_head < b.kv_head);
}

pub fn propose(request: Request) !Evidence {
    if (!nonzero(request.authority_sha256) or request.planned_exact_heads != 1 or
        request.maximum_fallbacks > 3 or request.heads.len < 2 or request.fixed_rows >= request.source_rows or request.dim == 0)
        return error.InvalidFitConfig;
    const expected_heads: usize = switch (request.topology) {
        .e2_global => 3,
        .e4_global => 8,
    };
    if (request.heads.len != expected_heads) return error.UnsupportedTopology;
    for (request.heads, 0..) |identity, i| {
        if (i != 0 and !less(request.heads[i - 1], identity)) return error.InvalidFitConfig;
    }
    const plan = (try budget.deriveUniformWithFallbackSearchIfFeasible(request.source_rows, request.dim, request.heads.len, request.fixed_rows, request.maximum_fallbacks)) orelse
        return error.CompactionPhysicalBudgetUnsupported;
    if (plan.maximum_fallbacks > request.maximum_fallbacks or plan.maximum_fallbacks > 3 or
        try std.math.mul(usize, plan.worst_candidate_bytes, 3) > try std.math.mul(usize, plan.ordinary_bytes, 2))
        return error.CompactionPhysicalBudgetUnsupported;

    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.topology-aware-physical-proposal.v1\n");
    hash.update(&policy_sha256);
    hash.update(&request.authority_sha256);
    hash.update(&.{@intFromEnum(request.topology)});
    try putU64(&hash, request.source_rows);
    try putU64(&hash, request.dim);
    try putU64(&hash, request.fixed_rows);
    try putU64(&hash, request.heads.len);
    for (request.heads) |identity| {
        putU32(&hash, identity.donor_layer);
        putU32(&hash, identity.kv_head);
    }
    hash.update(&.{ request.planned_exact_heads, request.maximum_fallbacks, plan.primary_percent, plan.fallback_percent, plan.maximum_fallbacks });
    try putU64(&hash, plan.primary_rows);
    try putU64(&hash, plan.fallback_rows);
    try putU64(&hash, plan.ordinary_bytes);
    try putU64(&hash, plan.worst_candidate_bytes);
    return .{ .policy_sha256 = policy_sha256, .authority_sha256 = request.authority_sha256, .topology = request.topology, .source_rows = request.source_rows, .dim = request.dim, .fixed_rows = request.fixed_rows, .head_count = request.heads.len, .planned_exact_heads = 1, .primary_percent = plan.primary_percent, .fallback_percent = plan.fallback_percent, .maximum_fallbacks = plan.maximum_fallbacks, .primary_rows = plan.primary_rows, .fallback_rows = plan.fallback_rows, .ordinary_bytes = plan.ordinary_bytes, .worst_candidate_bytes = plan.worst_candidate_bytes, .evidence_sha256 = hash.finalResult() };
}

// Current E2 profile authority has three global donors and one KV head each.
const e2_heads = [_]HeadIdentity{ .{ .donor_layer = 4, .kv_head = 0 }, .{ .donor_layer = 9, .kv_head = 0 }, .{ .donor_layer = 14, .kv_head = 0 } };
const e4_heads = [_]HeadIdentity{ .{ .donor_layer = 5, .kv_head = 0 }, .{ .donor_layer = 5, .kv_head = 1 }, .{ .donor_layer = 11, .kv_head = 0 }, .{ .donor_layer = 11, .kv_head = 1 }, .{ .donor_layer = 17, .kv_head = 0 }, .{ .donor_layer = 17, .kv_head = 1 }, .{ .donor_layer = 23, .kv_head = 0 }, .{ .donor_layer = 23, .kv_head = 1 } };
fn authority(byte: u8) [32]u8 {
    return @splat(byte);
}

test "E2 topology derives one exact and the largest feasible fallback reserve" {
    const evidence = try propose(.{ .authority_sha256 = authority(1), .topology = .e2_global, .source_rows = 7936, .dim = 512, .fixed_rows = 2176, .heads = &e2_heads });
    try std.testing.expectEqual(@as(u8, 0), evidence.maximum_fallbacks);
    try std.testing.expectEqual(@as(u8, 59), evidence.primary_percent);
    try std.testing.expect(evidence.worst_candidate_bytes * 3 <= evidence.ordinary_bytes * 2);
    try std.testing.expect(evidence.diagnostic_only and !evidence.qualification_credit);
}

test "E4 32K topology preserves three p95 fallbacks and binds authority" {
    const a = try propose(.{ .authority_sha256 = authority(2), .topology = .e4_global, .source_rows = 32512, .dim = 512, .fixed_rows = 2176, .heads = &e4_heads });
    const b = try propose(.{ .authority_sha256 = authority(3), .topology = .e4_global, .source_rows = 32512, .dim = 512, .fixed_rows = 2176, .heads = &e4_heads });
    try std.testing.expectEqual(@as(u8, 3), a.maximum_fallbacks);
    try std.testing.expectEqual(@as(u8, 95), a.fallback_percent);
    try std.testing.expect(!std.mem.eql(u8, &a.evidence_sha256, &b.evidence_sha256));
}

test "proposal rejects unordered, wrong-cardinality, and zero authority shapes" {
    var reversed = e2_heads;
    std.mem.swap(HeadIdentity, &reversed[0], &reversed[1]);
    try std.testing.expectError(error.InvalidFitConfig, propose(.{ .authority_sha256 = authority(1), .topology = .e2_global, .source_rows = 7936, .dim = 512, .fixed_rows = 2176, .heads = &reversed }));
    try std.testing.expectError(error.UnsupportedTopology, propose(.{ .authority_sha256 = authority(1), .topology = .e2_global, .source_rows = 7936, .dim = 512, .fixed_rows = 2176, .heads = e2_heads[0..2] }));
    try std.testing.expectError(error.InvalidFitConfig, propose(.{ .authority_sha256 = authority(0), .topology = .e4_global, .source_rows = 32512, .dim = 512, .fixed_rows = 2176, .heads = &e4_heads }));
}
