const std = @import("std");
const solver = @import("attention_match.zig");
pub const policy = "historical_anchor_authority_v2|floor128|allcoverage|remaining=farthest_original_h256|hho_values=absent";
pub const mandatory_floor: usize = 128;
pub const Result = struct {
    allocator: std.mem.Allocator,
    indices: []u32,
    digest: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
pub fn workspaceBytes(original_count: usize, mandatory_count: usize) !usize {
    return std.math.add(usize, try mul(original_count, @sizeOf(f64) + @sizeOf(bool)), try mul(mandatory_count, @sizeOf(u32))) catch error.CompactionWorkspaceLimit;
}
fn find(originals: []const usize, raw: u32) ?usize {
    for (originals, 0..) |value, i| if (value == raw) return i;
    return null;
}
pub fn buildEligible(allocator: std.mem.Allocator, eligible_originals: []const usize, normalized: []const f64, dim: usize, original_h256: []const u32, coverage: []const u32, query_authority: [32]u8, cap: usize, control: solver.Control) !Result {
    if (dim == 0 or eligible_originals.len == 0 or normalized.len != try mul(eligible_originals.len, dim) or original_h256.len != 256 or coverage.len > original_h256.len) return error.InvalidFitConfig;
    for (original_h256, 0..) |raw, i| if ((i > 0 and original_h256[i - 1] >= raw) or find(eligible_originals, raw) == null) return error.InvalidFitConfig;
    for (coverage, 0..) |raw, i| if ((i > 0 and coverage[i - 1] >= raw) or std.sort.binarySearch(u32, original_h256, raw, struct {
        fn order(key: u32, value: u32) std.math.Order {
            return std.math.order(key, value);
        }
    }.order) == null) return error.InvalidFitConfig;
    const count = @max(coverage.len, mandatory_floor);
    if (try workspaceBytes(original_h256.len, count) > cap) return error.CompactionWorkspaceLimit;
    const chosen = try allocator.alloc(bool, original_h256.len);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const nearest = try allocator.alloc(f64, original_h256.len);
    defer allocator.free(nearest);
    @memset(nearest, -std.math.inf(f64));
    // `original_h256` is fixed at 256 rows by the validated policy. Resolve its
    // sparse source indices once; the selection loop must not repeatedly scan
    // `eligible_originals` for the same raw row.
    var eligible_index: [256]usize = undefined;
    for (original_h256, 0..) |raw, i| eligible_index[i] = find(eligible_originals, raw).?;
    const out = try allocator.alloc(u32, count);
    errdefer allocator.free(out);
    var out_len: usize = 0;
    for (coverage) |raw| {
        try control.check();
        const at = std.sort.binarySearch(u32, original_h256, raw, struct {
            fn order(key: u32, value: u32) std.math.Order {
                return std.math.order(key, value);
            }
        }.order).?;
        chosen[at] = true;
        out[out_len] = raw;
        out_len += 1;
    }
    // Match the reference algorithm's center order exactly. Coverage is marked
    // first, then visited in original_h256 order. Every later winner is greater
    // than all previously visited center indices only in visitation time, not
    // necessarily raw value, so update immediately after it is selected.
    if (out_len != 0 and out_len < out.len) for (original_h256, 0..) |_, candidate| if (!chosen[candidate]) {
        var maximum = -std.math.inf(f64);
        for (original_h256, 0..) |_, center| if (chosen[center]) {
            if ((center & 31) == 0) try control.check();
            var cosine: f64 = 0;
            for (0..dim) |di| cosine += normalized[eligible_index[candidate] * dim + di] * normalized[eligible_index[center] * dim + di];
            maximum = @max(maximum, cosine);
        };
        nearest[candidate] = maximum;
    };
    while (out_len < out.len) {
        try control.check();
        var winner: ?usize = null;
        var best = std.math.inf(f64);
        for (original_h256, 0..) |raw, candidate| if (!chosen[candidate]) {
            const score: f64 = if (out_len == 0) 0 else nearest[candidate];
            if (winner == null or score < best or (score == best and raw < original_h256[winner.?])) {
                winner = candidate;
                best = score;
            }
        };
        const at = winner orelse return error.InsufficientSelectionProgress;
        chosen[at] = true;
        out[out_len] = original_h256[at];
        out_len += 1;
        if (out_len < out.len) for (original_h256, 0..) |_, candidate| if (!chosen[candidate]) {
            if ((candidate & 31) == 0) try control.check();
            var cosine: f64 = 0;
            for (0..dim) |di| cosine += normalized[eligible_index[candidate] * dim + di] * normalized[eligible_index[at] * dim + di];
            nearest[candidate] = @max(nearest[candidate], cosine);
        };
    }
    std.mem.sort(u32, out, {}, std.sort.asc(u32));
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&query_authority);
    var frame: [24]u8 = undefined;
    std.mem.writeInt(u64, frame[0..8], @intCast(original_h256.len), .little);
    std.mem.writeInt(u64, frame[8..16], @intCast(coverage.len), .little);
    std.mem.writeInt(u64, frame[16..24], @intCast(out.len), .little);
    hash.update(&frame);
    for (original_h256) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        hash.update(&bytes);
    }
    for (coverage) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        hash.update(&bytes);
    }
    for (out) |value| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, value, .little);
        hash.update(&bytes);
    }
    return .{ .allocator = allocator, .indices = out, .digest = hash.finalResult() };
}

test "coverage above floor is never truncated and coverage below floor gets original anchors" {
    var originals: [256]u32 = undefined;
    var eligible: [256]usize = undefined;
    var normalized: [256]f64 = undefined;
    for (&originals, &eligible, &normalized, 0..) |*raw, *ei, *q, i| {
        raw.* = @intCast(i);
        ei.* = i;
        q.* = if ((i & 1) == 0) 1 else -1;
    }
    var coverage129: [129]u32 = undefined;
    for (&coverage129, 0..) |*raw, i| raw.* = @intCast(i);
    var above = try buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &coverage129, @splat(1), 1 << 20, .{});
    defer above.deinit();
    try std.testing.expectEqualSlices(u32, &coverage129, above.indices);
    var below = try buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &.{ 3, 9, 17 }, @splat(2), 1 << 20, .{});
    defer below.deinit();
    try std.testing.expectEqual(@as(usize, 128), below.indices.len);
    for ([_]u32{ 3, 9, 17 }) |value| try std.testing.expect(std.mem.indexOfScalar(u32, below.indices, value) != null);
}

const Reference = struct {
    fn select(allocator: std.mem.Allocator, normalized: []const f64, dim: usize, coverage: []const u32) ![]u32 {
        const n: usize = 256;
        const count = @max(coverage.len, mandatory_floor);
        if (coverage.len == count) return allocator.dupe(u32, coverage);
        const gram = try allocator.alloc(f64, n * n);
        defer allocator.free(gram);
        for (0..n) |i| for (0..n) |j| {
            var dot: f64 = 0;
            for (0..dim) |di| dot += normalized[i * dim + di] * normalized[j * dim + di];
            gram[i * n + j] = dot;
        };
        const chosen = try allocator.alloc(bool, n);
        defer allocator.free(chosen);
        @memset(chosen, false);
        const out = try allocator.alloc(u32, count);
        errdefer allocator.free(out);
        for (coverage, 0..) |raw, i| {
            chosen[raw] = true;
            out[i] = raw;
        }
        var out_len = coverage.len;
        while (out_len < count) {
            var winner: ?usize = null;
            var best = std.math.inf(f64);
            for (0..n) |candidate| if (!chosen[candidate]) {
                var maximum = -std.math.inf(f64);
                for (0..n) |center| {
                    if (chosen[center]) maximum = @max(maximum, gram[candidate * n + center]);
                }
                const score: f64 = if (out_len == 0) 0 else maximum;
                if (winner == null or score < best or (score == best and candidate < winner.?)) {
                    winner = candidate;
                    best = score;
                }
            };
            const at = winner.?;
            chosen[at] = true;
            out[out_len] = @intCast(at);
            out_len += 1;
        }
        std.mem.sort(u32, out, {}, std.sort.asc(u32));
        return out;
    }

    fn digest(originals: []const u32, coverage: []const u32, out: []const u32, authority: [32]u8) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(policy);
        hash.update(&authority);
        var frame: [24]u8 = undefined;
        std.mem.writeInt(u64, frame[0..8], @intCast(originals.len), .little);
        std.mem.writeInt(u64, frame[8..16], @intCast(coverage.len), .little);
        std.mem.writeInt(u64, frame[16..24], @intCast(out.len), .little);
        hash.update(&frame);
        for ([_][]const u32{ originals, coverage, out }) |values| for (values) |value| {
            var bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &bytes, value, .little);
            hash.update(&bytes);
        };
        return hash.finalResult();
    }
};

test "incremental selector is bit exact to full-center reference at 256 by 512" {
    const allocator = std.testing.allocator;
    var originals: [256]u32 = undefined;
    var eligible: [256]usize = undefined;
    for (&originals, &eligible, 0..) |*raw, *index, i| {
        raw.* = @intCast(i);
        index.* = i;
    }
    const normalized = try allocator.alloc(f64, 256 * 512);
    defer allocator.free(normalized);
    // Exact powers of two, negative cosines, and repeated rows exercise ties
    // without making the reference depend on a transcendental implementation.
    for (normalized, 0..) |*value, i| {
        const row = i / 512;
        const column = i % 512;
        const signed: i32 = @intCast((row * 17 + column * 13) % 31);
        value.* = @as(f64, @floatFromInt(signed - 15)) / 16.0;
    }
    var coverage127: [127]u32 = undefined;
    var coverage128: [128]u32 = undefined;
    var coverage256: [256]u32 = undefined;
    for (&coverage256, 0..) |*value, i| value.* = @intCast(i);
    @memcpy(&coverage127, coverage256[0..127]);
    @memcpy(&coverage128, coverage256[0..128]);
    const authority: [32]u8 = @splat(0xa5);
    const coverages = [_][]const u32{ &.{}, &coverage127, &coverage128, &coverage256 };
    for (coverages) |coverage| {
        const expected = try Reference.select(allocator, normalized, 512, coverage);
        defer allocator.free(expected);
        var actual = try buildEligible(allocator, &eligible, normalized, 512, &originals, coverage, authority, 1 << 20, .{});
        defer actual.deinit();
        try std.testing.expectEqualSlices(u32, expected, actual.indices);
        try std.testing.expectEqual(Reference.digest(&originals, coverage, expected, authority), actual.digest);
    }
}

test "incremental selector maps gapped raw ids through permuted eligible rows" {
    const allocator = std.testing.allocator;
    const dim: usize = 8;
    var originals: [256]u32 = undefined;
    var eligible: [512]usize = undefined;
    var normalized: [512 * dim]f64 = undefined;
    for (&eligible, 0..) |*raw, row| {
        const permuted = (row * 73) % 512;
        raw.* = permuted;
        for (0..dim) |column| {
            const signed: i32 = @intCast((permuted * 11 + column * 7) % 19);
            normalized[row * dim + column] = @as(f64, @floatFromInt(signed - 9)) / 8.0;
        }
    }
    var gathered: [256 * dim]f64 = undefined;
    for (&originals, 0..) |*raw, i| {
        raw.* = @intCast(i * 2);
        const eligible_row = find(&eligible, raw.*).?;
        @memcpy(gathered[i * dim ..][0..dim], normalized[eligible_row * dim ..][0..dim]);
    }
    const coverage_raw = [_]u32{ 6, 42, 100 };
    const coverage_dense = [_]u32{ 3, 21, 50 };
    const authority: [32]u8 = @splat(0x3c);
    const expected_dense = try Reference.select(allocator, &gathered, dim, &coverage_dense);
    defer allocator.free(expected_dense);
    const expected_raw = try allocator.alloc(u32, expected_dense.len);
    defer allocator.free(expected_raw);
    for (expected_dense, expected_raw) |dense_index, *raw| raw.* = originals[dense_index];
    var actual = try buildEligible(allocator, &eligible, &normalized, dim, &originals, &coverage_raw, authority, 1 << 20, .{});
    defer actual.deinit();
    try std.testing.expectEqualSlices(u32, expected_raw, actual.indices);
    try std.testing.expectEqual(Reference.digest(&originals, &coverage_raw, expected_raw, authority), actual.digest);
}

test "incremental selector preserves admission and atomic cancellation" {
    var originals: [256]u32 = undefined;
    var eligible: [256]usize = undefined;
    var normalized: [256]f64 = @splat(-1);
    for (&originals, &eligible, 0..) |*raw, *index, i| {
        raw.* = @intCast(i);
        index.* = i;
    }
    try std.testing.expectError(error.CompactionWorkspaceLimit, buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &.{}, @splat(0), 1, .{}));
    for (0..3) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        try std.testing.expectError(error.OutOfMemory, buildEligible(failing.allocator(), &eligible, &normalized, 1, &originals, &.{}, @splat(0), 1 << 20, .{}));
    }
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &.{}, @splat(0), 1 << 20, .{ .check_fn = Cancel.check }));
    const CancelDuringUpdate = struct {
        checks: usize = 0,
        fn check(raw: ?*anyopaque) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks == 260) return error.Cancelled;
        }
    };
    var during_update: CancelDuringUpdate = .{};
    try std.testing.expectError(error.Cancelled, buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &.{0}, @splat(0), 1 << 20, .{ .context = &during_update, .check_fn = CancelDuringUpdate.check }));
    try std.testing.expectEqual(@as(usize, 260), during_update.checks);
    normalized[17] = std.math.nan(f64);
    const all = coverageAll();
    var nonfinite = try buildEligible(std.testing.allocator, &eligible, &normalized, 1, &originals, &all, @splat(1), 1 << 20, .{});
    defer nonfinite.deinit();
    try std.testing.expectEqual(@as(usize, 256), nonfinite.indices.len);
}

fn coverageAll() [256]u32 {
    var values: [256]u32 = undefined;
    for (&values, 0..) |*value, i| value.* = @intCast(i);
    return values;
}
