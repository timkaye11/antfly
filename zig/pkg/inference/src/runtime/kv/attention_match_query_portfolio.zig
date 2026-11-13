// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic-only deterministic training-query portfolio selection.
//! Heldout queries are deliberately absent from this API.
const std = @import("std");

pub const maximum_rows: usize = 512;

pub const Portfolio = struct {
    allocator: std.mem.Allocator,
    indices: []u64,
    queries: []f64,
    original_rows: usize,
    selected_rows: usize,
    original_train_sha256: [32]u8,
    sample_indices_sha256: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.indices);
        self.allocator.free(self.queries);
        self.* = undefined;
    }
};

fn hashF64(values: []const f64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [8]u8 = undefined;
    for (values) |value| {
        std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
        hash.update(&bytes);
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

fn hashIndices(indices: []const u64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [8]u8 = undefined;
    for (indices) |index| {
        std.mem.writeInt(u64, &bytes, index, .little);
        hash.update(&bytes);
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

pub fn select(allocator: std.mem.Allocator, training: []const f64, dim: usize, cap: usize, control: @import("attention_match.zig").Control) !Portfolio {
    if (comptime !@import("builtin").is_test) @compileError("query portfolio is diagnostic-test-only");
    if (dim == 0 or training.len == 0 or training.len % dim != 0 or cap == 0 or cap > maximum_rows) return error.InvalidFitConfig;
    const original_rows = training.len / dim;
    if (cap > original_rows) return error.InvalidFitConfig;
    try control.check();
    const indices = try allocator.alloc(u64, cap);
    errdefer allocator.free(indices);
    const query_elements = try std.math.mul(usize, cap, dim);
    const queries = try allocator.alloc(f64, query_elements);
    errdefer allocator.free(queries);
    for (0..cap) |portfolio_row| {
        try control.check();
        // Center of each equal-width stratum, computed without floating point.
        const numerator = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, portfolio_row, 2), 1), original_rows);
        const denominator = try std.math.mul(usize, cap, 2);
        const source_row = numerator / denominator;
        if (source_row >= original_rows or (portfolio_row > 0 and source_row <= indices[portfolio_row - 1])) return error.InvalidFitConfig;
        indices[portfolio_row] = @intCast(source_row);
        @memcpy(queries[portfolio_row * dim ..][0..dim], training[source_row * dim ..][0..dim]);
    }
    try control.check();
    return .{ .allocator = allocator, .indices = indices, .queries = queries, .original_rows = original_rows, .selected_rows = cap, .original_train_sha256 = hashF64(training), .sample_indices_sha256 = hashIndices(indices) };
}

test "query portfolio evenly covers training rows deterministically" {
    var training: [20]f64 = undefined;
    for (&training, 0..) |*value, index| value.* = @floatFromInt(index);
    var first = try select(std.testing.allocator, &training, 2, 4, .{});
    defer first.deinit();
    var second = try select(std.testing.allocator, &training, 2, 4, .{});
    defer second.deinit();
    try std.testing.expectEqualSlices(u64, &.{ 1, 3, 6, 8 }, first.indices);
    try std.testing.expectEqualSlices(u64, first.indices, second.indices);
    try std.testing.expectEqualSlices(f64, first.queries, second.queries);
    try std.testing.expectEqual(first.original_train_sha256, second.original_train_sha256);
    try std.testing.expectEqual(first.sample_indices_sha256, second.sample_indices_sha256);
    try std.testing.expectEqual(@as(f64, 2), first.queries[0]);
    try std.testing.expectEqual(@as(f64, 17), first.queries[first.queries.len - 1]);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    var result = try select(allocator, &.{ 0, 1, 2, 3 }, 1, 2, .{});
    result.deinit();
}

test "query portfolio rejects invalid caps cancellation and unwinds allocations" {
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &.{ 0, 1 }, 1, 0, .{}));
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &.{ 0, 1 }, 1, 3, .{}));
    try std.testing.expectError(error.InvalidFitConfig, select(std.testing.allocator, &.{ 0, 1 }, 1, maximum_rows + 1, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &.{ 0, 1 }, 1, 1, .{ .check_fn = Cancel.check }));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
