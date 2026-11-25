const std = @import("std");
const platform = @import("antfly_platform");
const fixture_mod = @import("attention_match_rejected_fixture.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const solver = @import("attention_match.zig");

pub const DomainMetrics = struct {
    queries: usize,
    max_relative_mass_error: f64,
    relative_output_rmse: f64,
};

pub const Report = struct {
    schema: []const u8 = "antfly.attention-match-v18-rejected-domain-replay.v2",
    recent: DomainMetrics,
    future_primary: DomainMetrics,
    combined_train: DomainMetrics,
    heldout: DomainMetrics,
    source_middle_rows: usize,
    source_exact_middle_rows: usize,
    source_middle_causal_coverage_bound_union_tokens: usize,
    candidate_middle_rows: usize,
    candidate_exact_middle_rows: usize,
    candidate_middle_causal_coverage_bound_union_tokens: usize,
    diagnostic_only: bool = true,
    qualification_credit: bool = false,
};

fn metrics(allocator: std.mem.Allocator, original: solver.Head, candidate: solver.Head, queries: []const f64) !DomainMetrics {
    if (queries.len == 0 or queries.len % original.dim != 0) return error.InvalidRejectedHeadFixture;
    const cfg: solver.Config = .{ .target_rows = 1, .max_relative_mass_error = std.math.floatMax(f64), .max_relative_output_rmse = std.math.floatMax(f64), .max_workspace_bytes = 1024 * 1024 * 1024 };
    const observed = try solver.validationMetrics(allocator, original, candidate, queries, cfg);
    if (!observed.quality_evaluated or !std.math.isFinite(observed.max_relative_mass_error) or !std.math.isFinite(observed.relative_output_rmse)) return error.InvalidRejectedHeadFixture;
    return .{ .queries = queries.len / original.dim, .max_relative_mass_error = observed.max_relative_mass_error, .relative_output_rmse = observed.relative_output_rmse };
}

fn causalCoverageBoundUnion(positions: []const u64, coverage_ends: []const u64, prefix_rows: u64, middle_end: u64, logical_length: u64) !usize {
    if (positions.len != coverage_ends.len or prefix_rows > middle_end or middle_end > logical_length) return error.InvalidRejectedHeadFixture;
    var union_tokens: usize = 0;
    var union_begin: ?u64 = null;
    var union_end: u64 = 0;
    for (positions, coverage_ends, 0..) |position, coverage_end, row| {
        if (coverage_end <= position or coverage_end > logical_length or
            (row > 0 and (position <= positions[row - 1] or coverage_end < coverage_ends[row - 1])))
            return error.InvalidRejectedHeadFixture;
        const begin = @max(position, prefix_rows);
        const end = @min(coverage_end, middle_end);
        if (begin >= end) continue;
        if (union_begin == null) {
            union_begin = begin;
            union_end = end;
        } else if (begin <= union_end) {
            union_end = @max(union_end, end);
        } else {
            union_tokens = try std.math.add(usize, union_tokens, @intCast(union_end - union_begin.?));
            union_begin = begin;
            union_end = end;
        }
    }
    if (union_begin) |begin| union_tokens = try std.math.add(usize, union_tokens, @intCast(union_end - begin));
    return union_tokens;
}

fn validateDomainPositions(train_positions: []const u64, heldout_positions: []const u64, batch_count: usize) !void {
    if (train_positions.len != batch_count * 128 or heldout_positions.len != batch_count * 64) return error.InvalidRejectedHeadFixture;
    for (0..batch_count) |batch| {
        for (0..64) |row| {
            if (train_positions[batch * 128 + row] != 8064 + row or
                train_positions[batch * 128 + 64 + row] != 8132 + 8 * row or
                heldout_positions[batch * 64 + row] != 8128 + 8 * row)
                return error.InvalidRejectedHeadFixture;
        }
    }
}

pub fn analyze(allocator: std.mem.Allocator, fixture: *const fixture_mod.Fixture) !Report {
    const d = fixture.geometry.head_dim;
    if (fixture.geometry.logical_length != 8128 or fixture.geometry.prefix_rows != 128 or fixture.geometry.recent_rows != 2048 or fixture.geometry.retention_divisor != 2 or d != 512 or fixture.query_train_counts.len == 0 or fixture.query_train_counts.len != fixture.query_heldout_counts.len) return error.InvalidRejectedHeadFixture;
    var train_rows: usize = 0;
    var heldout_rows: usize = 0;
    for (fixture.query_train_counts, fixture.query_heldout_counts) |nt, nh| {
        if (nt != 128 or nh != 64) return error.InvalidRejectedHeadFixture;
        train_rows = try std.math.add(usize, train_rows, nt);
        heldout_rows = try std.math.add(usize, heldout_rows, nh);
    }
    if (fixture.train.len != train_rows * d or fixture.heldout.len != heldout_rows * d) return error.InvalidRejectedHeadFixture;
    try validateDomainPositions(fixture.train_positions, fixture.heldout_positions, fixture.query_train_counts.len);
    const half_rows = train_rows / 2;
    const recent = try allocator.alloc(f64, half_rows * d);
    defer allocator.free(recent);
    const future = try allocator.alloc(f64, half_rows * d);
    defer allocator.free(future);
    var input_row: usize = 0;
    var recent_row: usize = 0;
    var future_row: usize = 0;
    for (fixture.query_train_counts) |_| {
        for (0..64) |row| {
            @memcpy(recent[recent_row * d ..][0..d], fixture.train[(input_row + row) * d ..][0..d]);
            recent_row += 1;
        }
        for (0..64) |row| {
            @memcpy(future[future_row * d ..][0..d], fixture.train[(input_row + 64 + row) * d ..][0..d]);
            future_row += 1;
        }
        input_row += 128;
    }
    const original = fixture.originalHead();
    const candidate = fixture.candidateHead();
    const middle_end: u64 = fixture.geometry.logical_length - fixture.geometry.recent_rows;
    const source_middle_causal_coverage_bound_union_tokens = try causalCoverageBoundUnion(fixture.original_positions, fixture.original_coverage, fixture.geometry.prefix_rows, middle_end, fixture.geometry.logical_length);
    var source_middle_rows: usize = 0;
    var source_exact_middle_rows: usize = 0;
    for (fixture.original_positions, fixture.original_coverage) |position, coverage_end| {
        if (position >= fixture.geometry.prefix_rows and position < middle_end) {
            source_middle_rows += 1;
            source_exact_middle_rows += @intFromBool(coverage_end == position + 1);
        }
    }
    const candidate_middle_causal_coverage_bound_union_tokens = try causalCoverageBoundUnion(fixture.candidate_positions, fixture.candidate_coverage, fixture.geometry.prefix_rows, middle_end, fixture.geometry.logical_length);
    var candidate_middle_rows: usize = 0;
    var candidate_exact_middle_rows: usize = 0;
    for (fixture.candidate_positions, fixture.candidate_coverage) |position, coverage_end| {
        if (position >= fixture.geometry.prefix_rows and position < middle_end) {
            candidate_middle_rows += 1;
            candidate_exact_middle_rows += @intFromBool(coverage_end == position + 1);
        }
    }
    return .{
        .recent = try metrics(allocator, original, candidate, recent),
        .future_primary = try metrics(allocator, original, candidate, future),
        .combined_train = try metrics(allocator, original, candidate, fixture.train),
        .heldout = try metrics(allocator, original, candidate, fixture.heldout),
        .source_middle_rows = source_middle_rows,
        .source_exact_middle_rows = source_exact_middle_rows,
        .source_middle_causal_coverage_bound_union_tokens = source_middle_causal_coverage_bound_union_tokens,
        .candidate_middle_rows = candidate_middle_rows,
        .candidate_exact_middle_rows = candidate_exact_middle_rows,
        .candidate_middle_causal_coverage_bound_union_tokens = candidate_middle_causal_coverage_bound_union_tokens,
    };
}

const Plan = struct {
    schema: []const u8,
    fixture_dir: []const u8,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    result_path: []const u8,
};

fn parseDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidRejectedHeadReplayPlan;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidRejectedHeadReplayPlan;
    return out;
}

test "attention match V18 rejected domain replay" {
    const plan_path = platform.env.getenv("ANTFLY_ATTENTION_MATCH_V18_REJECTED_DOMAIN_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidRejectedHeadReplayPlan;
    const io = std.testing.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, std.testing.allocator, .limited(64 * 1024));
    defer std.testing.allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(Plan, std.testing.allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    defer parsed.deinit();
    const p = parsed.value;
    if (!std.mem.eql(u8, p.schema, "antfly.attention-match-v18-rejected-domain-plan.v1") or !std.fs.path.isAbsolute(p.fixture_dir) or !std.fs.path.isAbsolute(p.result_path)) return error.InvalidRejectedHeadReplayPlan;
    const identity: checkpoint.Identity = .{ .weights = try parseDigest(p.identity.weights), .tokenizer = try parseDigest(p.identity.tokenizer), .template = try parseDigest(p.identity.template), .layout = try parseDigest(p.identity.layout) };
    const binding: fixture_mod.Binding = .{ .output_dir = p.fixture_dir, .identity = identity, .config_sha256 = try parseDigest(p.config_sha256) };
    var fixture = try fixture_mod.read(std.testing.allocator, io, p.fixture_dir, binding);
    defer fixture.deinit();
    const report = try analyze(std.testing.allocator, &fixture);
    const body = try std.json.Stringify.valueAlloc(std.testing.allocator, report, .{});
    defer std.testing.allocator.free(body);
    var file = try std.Io.Dir.createFileAbsolute(io, p.result_path, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
    std.debug.print("attention_match_v18_rejected_domain recent_mass={d} recent_rmse={d} future_mass={d} future_rmse={d} combined_mass={d} combined_rmse={d} hho_mass={d} hho_rmse={d} candidate_middle_rows={d} candidate_exact_middle_rows={d} candidate_middle_causal_coverage_bound_union_tokens={d} diagnostic_only=true quality_credit=false\n", .{ report.recent.max_relative_mass_error, report.recent.relative_output_rmse, report.future_primary.max_relative_mass_error, report.future_primary.relative_output_rmse, report.combined_train.max_relative_mass_error, report.combined_train.relative_output_rmse, report.heldout.max_relative_mass_error, report.heldout.relative_output_rmse, report.candidate_middle_rows, report.candidate_exact_middle_rows, report.candidate_middle_causal_coverage_bound_union_tokens });
}

test "domain replay accepts equal coverage ends and unions overlapping causal bounds" {
    const positions = [_]u64{ 128, 129, 132 };
    const equal_and_overlapping_ends = [_]u64{ 132, 132, 136 };
    try std.testing.expectEqual(@as(usize, 8), try causalCoverageBoundUnion(&positions, &equal_and_overlapping_ends, 128, 256, 512));
    const decreasing_ends = [_]u64{ 132, 131, 136 };
    try std.testing.expectError(error.InvalidRejectedHeadFixture, causalCoverageBoundUnion(&positions, &decreasing_ends, 128, 256, 512));
}

test "domain replay rejects mutated E positions before metrics" {
    var positions: [128]u64 = undefined;
    for (positions[0..64], 0..) |*position, row| position.* = 8064 + row;
    for (positions[64..], 0..) |*position, row| position.* = 8132 + 8 * row;
    var heldout: [64]u64 = undefined;
    for (&heldout, 0..) |*position, row| position.* = 8128 + 8 * row;
    try validateDomainPositions(&positions, &heldout, 1);
    positions[64] -= 1;
    try std.testing.expectError(error.InvalidRejectedHeadFixture, validateDomainPositions(&positions, &heldout, 1));
    positions[64] += 1;
    heldout[1] += 1;
    try std.testing.expectError(error.InvalidRejectedHeadFixture, validateDomainPositions(&positions, &heldout, 1));
}
