const std = @import("std");
const solver = @import("attention_match.zig");

pub const policy = "train_challenge_hybrid_v1|mandatory=source918_coverage_winners|remaining=error_weighted_normalized_q_diversity|ties=lowest_dense_index|serialize=chronological|hho_values=forbidden";

pub const Result = struct {
    allocator: std.mem.Allocator,
    indices: []u32,
    mandatory_count: usize,
    challenge_count: usize,
    digest: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

pub fn workspaceBytes(eligible_rows: usize, selected_count: usize) !usize {
    return std.math.add(usize, try mul(eligible_rows, @sizeOf(f64) + @sizeOf(bool)), try mul(selected_count, @sizeOf(u32))) catch error.CompactionWorkspaceLimit;
}

/// `eligible_originals` maps normalized/error rows to Dense512 row indices.
/// `mandatory` is the sorted Source918 coverage-winner authority for this
/// consumer. HHO values are not part of this API.
pub fn select(allocator: std.mem.Allocator, eligible_originals: []const usize, normalized: []const f64, errors: []const f64, dim: usize, mandatory: []const u32, selected_count: usize, cap: usize, control: solver.Control) !Result {
    if (dim == 0 or eligible_originals.len == 0 or errors.len != eligible_originals.len or
        normalized.len != try mul(eligible_originals.len, dim) or selected_count == 0 or
        selected_count > eligible_originals.len or mandatory.len > selected_count)
        return error.InvalidFitConfig;
    if (try workspaceBytes(eligible_originals.len, selected_count) > cap) return error.CompactionWorkspaceLimit;
    for (eligible_originals, 0..) |raw, i| if (i > 0 and eligible_originals[i - 1] >= raw) return error.InvalidFitConfig;
    for (mandatory, 0..) |raw, i| if (i > 0 and mandatory[i - 1] >= raw) return error.InvalidFitConfig;
    var maximum_error: f64 = 0;
    for (errors) |value| {
        if (!std.math.isFinite(value) or value < 0) return error.NonFiniteAttention;
        maximum_error = @max(maximum_error, value);
    }
    for (normalized) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    try control.check();

    const nearest = try allocator.alloc(f64, eligible_originals.len);
    defer allocator.free(nearest);
    @memset(nearest, 0);
    const chosen = try allocator.alloc(bool, eligible_originals.len);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const output = try allocator.alloc(u32, selected_count);
    errdefer allocator.free(output);
    var output_len: usize = 0;

    // Coverage rows are immutable members, and seed diversity for only the
    // flexible remainder. Reject any authority row excluded by the role mask.
    for (mandatory) |raw| {
        try control.check();
        const at = std.sort.binarySearch(usize, eligible_originals, @as(usize, raw), struct {
            fn order(key: usize, value: usize) std.math.Order {
                return std.math.order(key, value);
            }
        }.order) orelse return error.InvalidFitConfig;
        chosen[at] = true;
        output[output_len] = raw;
        output_len += 1;
    }
    for (0..chosen.len) |center| if (chosen[center]) {
        for (0..chosen.len) |row| if (!chosen[row]) {
            if ((row & 31) == 0) try control.check();
            var cosine: f64 = 0;
            for (0..dim) |di| cosine += normalized[center * dim + di] * normalized[row * dim + di];
            if (!std.math.isFinite(cosine)) return error.NonFiniteAttention;
            nearest[row] = @max(nearest[row], std.math.clamp(cosine, 0, 1));
        };
    };

    const denom = @max(maximum_error, std.math.floatEps(f64));
    while (output_len < output.len) {
        try control.check();
        var best: ?usize = null;
        var best_score: f64 = -1;
        for (errors, 0..) |value, row| if (!chosen[row]) {
            const score = (value / denom) * (1 - nearest[row]);
            if (best == null or score > best_score or (score == best_score and eligible_originals[row] < eligible_originals[best.?])) {
                best = row;
                best_score = score;
            }
        };
        const picked = best orelse return error.InsufficientSelectionProgress;
        chosen[picked] = true;
        output[output_len] = @intCast(eligible_originals[picked]);
        output_len += 1;
        for (0..chosen.len) |row| if (!chosen[row]) {
            if ((row & 31) == 0) try control.check();
            var cosine: f64 = 0;
            for (0..dim) |di| cosine += normalized[picked * dim + di] * normalized[row * dim + di];
            if (!std.math.isFinite(cosine)) return error.NonFiniteAttention;
            nearest[row] = @max(nearest[row], std.math.clamp(cosine, 0, 1));
        };
    }
    std.mem.sort(u32, output, {}, std.sort.asc(u32));
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    var frame: [32]u8 = undefined;
    std.mem.writeInt(u64, frame[0..8], @intCast(dim), .little);
    std.mem.writeInt(u64, frame[8..16], @intCast(eligible_originals.len), .little);
    std.mem.writeInt(u64, frame[16..24], @intCast(mandatory.len), .little);
    std.mem.writeInt(u64, frame[24..32], @intCast(selected_count), .little);
    hash.update(&frame);
    for (eligible_originals) |value| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @intCast(value), .little);
        hash.update(&bytes);
    }
    for (mandatory) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        hash.update(&bytes);
    }
    for (output) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        hash.update(&bytes);
    }
    for (errors) |value| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        hash.update(&bytes);
    }
    for (normalized) |value| {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        hash.update(&bytes);
    }
    return .{ .allocator = allocator, .indices = output, .mandatory_count = mandatory.len, .challenge_count = selected_count - mandatory.len, .digest = hash.finalResult() };
}

test "coverage winners are immutable and only remaining slots use TRAIN challenge" {
    const originals = [_]usize{ 0, 1, 2, 3, 4 };
    // Row4 is the highest challenge, but mandatory rows1/3 must survive.
    const normalized = [_]f64{ 1, 0, 0, 1, -1, 0, 0, -1, 0.7, 0.7 };
    var result = try select(std.testing.allocator, &originals, &normalized, &.{ 0.1, 0.01, 0.2, 0.01, 1.0 }, 2, &.{ 1, 3 }, 3, 1 << 20, .{});
    defer result.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 4 }, result.indices);
    try std.testing.expectEqual(@as(usize, 2), result.mandatory_count);
    try std.testing.expectEqual(@as(usize, 1), result.challenge_count);
}

test "mandatory authority exclusion cap cancellation and OOM are atomic" {
    const originals = [_]usize{ 0, 2, 3 };
    const normalized = [_]f64{ 1, 0, 0, 1, -1, 0 };
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &originals, &normalized, &.{ 1, 1, 1 }, 2, &.{1}, 2, 1 << 20, .{}));
    const need = try workspaceBytes(3, 2);
    try std.testing.expectError(error.CompactionWorkspaceLimit, select(std.testing.failing_allocator, &originals, &normalized, &.{ 1, 1, 1 }, 2, &.{2}, 2, need - 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &originals, &normalized, &.{ 1, 1, 1 }, 2, &.{2}, 2, need, .{ .context = null, .check_fn = Cancel.check }));
}
