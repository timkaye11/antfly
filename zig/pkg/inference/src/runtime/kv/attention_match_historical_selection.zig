// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Development-only TRAIN selector. It never observes heldout query values;
//! heldout logical positions are used solely as role exclusions.
const std = @import("std");
const platform = @import("antfly_platform");
const attention_match = @import("attention_match.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
pub const policy = "antfly.historical-q-kcenter.normalized-cosine.v1;role-exclusion=same-consumer-logical-position;seeds=first-last-eligible-chronological;choice=farthest-nearest-tie-lowest-original-index;serialization=chronological";
pub const policy_sha256: [32]u8 = .{ 0x67, 0x15, 0x4a, 0x73, 0x6a, 0x68, 0xd9, 0x2f, 0xb3, 0xcc, 0x4a, 0xce, 0x87, 0x5e, 0x59, 0x34, 0xa7, 0xa3, 0x3b, 0xa6, 0x34, 0xc8, 0x24, 0x63, 0x1d, 0x5c, 0xdf, 0xa6, 0xa6, 0x69, 0x1a, 0xa3 };
pub const max_rows: usize = 512;
pub const max_dim: usize = 512;
pub const max_selected: usize = 256;
pub const maximum_workspace_bytes: usize = max_rows * max_dim * @sizeOf(f64) + max_rows * (@sizeOf(u32) + @sizeOf(f64) + @sizeOf(bool)) + max_selected * @sizeOf(u32);

pub fn policyDigest() [32]u8 {
    return policy_sha256;
}

test "selector policy digest is frozen Root408 policy" {
    try std.testing.expectEqual(policy_sha256, policyDigest());
    try std.testing.expectEqual(@as(usize, 2_104_832), maximum_workspace_bytes);
}

pub const OwnedSelection = struct {
    allocator: std.mem.Allocator,
    indices: []u32,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.* = undefined;
    }
};

fn excluded(position: u64, heldout_positions: []const u64) bool {
    for (heldout_positions) |heldout| if (position == heldout) return true;
    return false;
}

/// Selects chronological original row indices. Queries are `[rows, dim]`.
/// Allocation is bounded by `max_workspace_bytes`; output is published only
/// after cancellation, finiteness, shape, and degeneracy checks complete.
pub fn select(
    allocator: std.mem.Allocator,
    queries: []const f32,
    positions: []const u64,
    heldout_positions: []const u64,
    dim: usize,
    selected_count: usize,
    control: attention_match.Control,
) !OwnedSelection {
    const rows = positions.len;
    if (rows < 2 or rows > max_rows or dim == 0 or dim > max_dim or selected_count < 2 or selected_count > max_selected or selected_count > rows or queries.len != try std.math.mul(usize, rows, dim)) return error.InvalidTrainingAugmentation;
    for (positions[1..], positions[0 .. positions.len - 1]) |next, prior| if (next <= prior) return error.InvalidTrainingAugmentation;
    try control.check();

    const eligible = try allocator.alloc(u32, rows);
    defer allocator.free(eligible);
    var eligible_len: usize = 0;
    for (positions, 0..) |position, row| if (!excluded(position, heldout_positions)) {
        eligible[eligible_len] = @intCast(row);
        eligible_len += 1;
    };
    if (eligible_len < selected_count) return error.InsufficientSelectionProgress;
    const eligible_rows = eligible[0..eligible_len];

    const normalized = try allocator.alloc(f64, try std.math.mul(usize, eligible_len, dim));
    defer allocator.free(normalized);
    for (eligible_rows, 0..) |original, eligible_index| {
        try control.check();
        var norm2: f64 = 0;
        for (queries[@as(usize, original) * dim ..][0..dim]) |raw| {
            const value: f64 = raw;
            if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
            norm2 += value * value;
        }
        if (!std.math.isFinite(norm2) or norm2 <= 0) return error.InsufficientSelectionProgress;
        const inverse = 1.0 / @sqrt(norm2);
        for (queries[@as(usize, original) * dim ..][0..dim], normalized[eligible_index * dim ..][0..dim]) |raw, *dst| dst.* = @as(f64, raw) * inverse;
    }

    const nearest = try allocator.alloc(f64, eligible_len);
    defer allocator.free(nearest);
    const chosen = try allocator.alloc(bool, eligible_len);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const output = try allocator.alloc(u32, selected_count);
    errdefer allocator.free(output);

    output[0] = eligible_rows[0];
    output[1] = eligible_rows[eligible_len - 1];
    chosen[0] = true;
    chosen[eligible_len - 1] = true;
    for (0..eligible_len) |candidate| {
        var first: f64 = 0;
        var last: f64 = 0;
        for (0..dim) |column| {
            const value = normalized[candidate * dim + column];
            first += value * normalized[column];
            last += value * normalized[(eligible_len - 1) * dim + column];
        }
        nearest[candidate] = @max(first, last);
    }
    nearest[0] = std.math.inf(f64);
    nearest[eligible_len - 1] = std.math.inf(f64);

    var output_len: usize = 2;
    while (output_len < selected_count) : (output_len += 1) {
        try control.check();
        var winner: ?usize = null;
        var score = std.math.inf(f64);
        for (nearest, 0..) |value, candidate| if (!chosen[candidate] and (winner == null or value < score)) {
            winner = candidate;
            score = value;
        };
        const center = winner orelse return error.InsufficientSelectionProgress;
        if (!std.math.isFinite(score)) return error.InsufficientSelectionProgress;
        chosen[center] = true;
        output[output_len] = eligible_rows[center];
        for (0..eligible_len) |candidate| if (!chosen[candidate]) {
            var similarity: f64 = 0;
            for (0..dim) |column| similarity += normalized[candidate * dim + column] * normalized[center * dim + column];
            if (!std.math.isFinite(similarity)) return error.InsufficientSelectionProgress;
            nearest[candidate] = @max(nearest[candidate], similarity);
        };
        nearest[center] = std.math.inf(f64);
    }
    // Chronological serialization without an allocator or comparator context.
    for (1..output.len) |i| {
        const value = output[i];
        var j = i;
        while (j > 0 and output[j - 1] > value) : (j -= 1) output[j] = output[j - 1];
        output[j] = value;
    }
    try control.check();
    return .{
        .allocator = allocator,
        .indices = output,
    };
}

test "selector excludes heldout roles and serializes chronological indices" {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1, 0.7, 0.7 };
    var result = try select(std.testing.allocator, &q, &.{ 10, 11, 12, 13, 14 }, &.{12}, 2, 3, .{});
    defer result.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 4 }, result.indices);
}

test "selector tie chooses lowest original index" {
    const q = [_]f32{ 1, 0, 0, 1, 0, -1, -1, 0 };
    var result = try select(std.testing.allocator, &q, &.{ 1, 2, 3, 4 }, &.{}, 2, 3, .{});
    defer result.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 3 }, result.indices);
}

test "selector rejects nonfinite and zero norm without publishing output" {
    try std.testing.expectError(error.InvalidTrainingAugmentation, select(std.testing.allocator, &.{ 1, 0, std.math.nan(f32), 1 }, &.{ 1, 2 }, &.{}, 2, 2, .{}));
    try std.testing.expectError(error.InsufficientSelectionProgress, select(std.testing.allocator, &.{ 1, 0, 0, 0 }, &.{ 1, 2 }, &.{}, 2, 2, .{}));
}

const CancelAfter = struct {
    remaining: usize,
    fn check(raw: ?*anyopaque) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        if (self.remaining == 0) return error.Cancelled;
        self.remaining -= 1;
    }
};

test "selector cancellation during normalization frees workspace" {
    var cancel: CancelAfter = .{ .remaining = 2 };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &.{ 1, 0, 0, 1 }, &.{ 1, 2 }, &.{}, 2, 2, .{ .context = &cancel, .check_fn = CancelAfter.check }));
}

test "selector cancellation during selection frees output and scratch" {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1 };
    var cancel: CancelAfter = .{ .remaining = 5 }; // initial + four normalization checks; next selection check cancels.
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &q, &.{ 1, 2, 3, 4 }, &.{}, 2, 3, .{ .context = &cancel, .check_fn = CancelAfter.check }));
}

fn allocationFailureCase(allocator: std.mem.Allocator) !void {
    const q = [_]f32{ 1, 0, 0, 1, -1, 0, 0, -1 };
    var result = try select(allocator, &q, &.{ 1, 2, 3, 4 }, &.{}, 2, 3, .{});
    result.deinit();
}

test "selector unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailureCase, .{});
}

const ParityPlan = struct {
    dense_queries_path: []const u8,
    dense_queries_sha256: []const u8,
    dense_positions_path: []const u8,
    dense_positions_sha256: []const u8,
    heldout_positions_path: []const u8,
    heldout_positions_sha256: []const u8,
    expected_indices_path: []const u8,
    expected_indices_sha256: []const u8,
};

fn decodeDigest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidTrainingAugmentation;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidTrainingAugmentation;
    return out;
}

fn readTypedPinned(comptime T: type, allocator: std.mem.Allocator, io: std.Io, path: []const u8, digest_text: []const u8, count: usize) ![]T {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidTrainingAugmentation;
    const out = try allocator.alloc(T, count);
    errdefer allocator.free(out);
    var file = try std.Io.Dir.openFileAbsolute(io, path, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidTrainingAugmentation;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.InvalidTrainingAugmentation;
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &(try decodeDigest(digest_text)))) return error.InvalidTrainingAugmentation;
    return out;
}

test "compiled selector matches caller-pinned Root414 all-batch fixture" {
    const plan_path = platform.env.getenv("ANTFLY_HISTORICAL_SELECTION_PARITY_PLAN") orelse return error.SkipZigTest;
    if (!std.fs.path.isAbsolute(plan_path)) return error.InvalidTrainingAugmentation;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const plan_bytes = try std.Io.Dir.cwd().readFileAlloc(io, plan_path, allocator, .limited(16 * 1024));
    defer allocator.free(plan_bytes);
    const parsed = try std.json.parseFromSlice(ParityPlan, allocator, plan_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const p = parsed.value;
    const queries = try readTypedPinned(f32, allocator, io, p.dense_queries_path, p.dense_queries_sha256, 56 * 512 * 512);
    defer allocator.free(queries);
    const positions = try readTypedPinned(u64, allocator, io, p.dense_positions_path, p.dense_positions_sha256, 56 * 512);
    defer allocator.free(positions);
    const heldout = try readTypedPinned(u64, allocator, io, p.heldout_positions_path, p.heldout_positions_sha256, 56 * 64);
    defer allocator.free(heldout);
    const expected = try readTypedPinned(u32, allocator, io, p.expected_indices_path, p.expected_indices_sha256, 56 * 256);
    defer allocator.free(expected);
    var selected_hash = Sha256.init(.{});
    for (0..56) |batch| {
        var actual = try select(allocator, queries[batch * 512 * 512 ..][0 .. 512 * 512], positions[batch * 512 ..][0..512], heldout[batch * 64 ..][0..64], 512, 256, .{});
        defer actual.deinit();
        try std.testing.expectEqualSlices(u32, expected[batch * 256 ..][0..256], actual.indices);
        selected_hash.update(std.mem.sliceAsBytes(actual.indices));
    }
    try std.testing.expectEqual(try decodeDigest(p.expected_indices_sha256), selected_hash.finalResult());
}
