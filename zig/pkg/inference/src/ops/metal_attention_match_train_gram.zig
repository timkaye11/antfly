// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bounded FP64 per-query Gram backend for the diagnostic v18 geometry path.
const std = @import("std");
const builtin = @import("builtin");
const solver = @import("../runtime/kv/attention_match.zig");
const gram = @import("../runtime/kv/attention_match_joint_gram_geometry.zig");

// LP64 CBLAS_NEW_LAPACK declaration from the macOS 13.3+ SDK. The inference
// Metal build already links Accelerate for metal_attention_match_lapack.zig.
extern "c" fn @"cblas_dgemm$NEWLAPACK"(c_int, c_int, c_int, c_int, c_int, c_int, f64, [*]const f64, c_int, [*]const f64, c_int, f64, [*]f64, c_int) void;

const row_major: c_int = 101;
const no_transpose: c_int = 111;
const transpose: c_int = 112;
const supported = if (builtin.os.tag == .macos)
    builtin.os.version_range.semver.min.order(.{ .major = 13, .minor = 3, .patch = 0 }) != .lt
else
    false;

pub const Context = struct {
    /// One context belongs to one serialized replay worker. It is deliberately
    /// non-atomic and must not be shared by concurrent Gram builds.
    successful_calls: usize = 0,

    pub fn successfulCalls(self: *const @This()) usize {
        return self.successful_calls;
    }
};

fn checkedWorkspace(k: usize, dim: usize) !usize {
    if (k == 0 or dim == 0) return error.InvalidAttentionShape;
    _ = std.math.cast(c_int, k) orelse return error.InvalidAttentionShape;
    _ = std.math.cast(c_int, dim) orelse return error.InvalidAttentionShape;
    const elements = try std.math.mul(usize, k, k);
    const bytes = try std.math.mul(usize, elements, @sizeOf(f64));
    return try std.math.add(usize, bytes, @alignOf(f64) - 1);
}

fn workspace(_: ?*anyopaque, k: usize, dim: usize) !usize {
    return checkedWorkspace(k, dim);
}

fn compute(raw: ?*anyopaque, w: []const f64, k: usize, dim: usize, packed_output: []f64, scratch: []u8, control: solver.Control) !void {
    if (comptime !supported) return error.UnsupportedOperation;
    const context: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidCallbackContext));
    const required = try checkedWorkspace(k, dim);
    const w_len = try std.math.mul(usize, k, dim);
    const packed_len = try std.math.divExact(usize, try std.math.mul(usize, k, try std.math.add(usize, k, 1)), 2);
    if (w.len != w_len or packed_output.len != packed_len) return error.InvalidAttentionShape;
    if (scratch.len < required) return error.CompactionWorkspaceLimit;
    try control.check();
    for (w) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;

    const dense_len = try std.math.mul(usize, k, k);
    const dense_bytes_len = try std.math.mul(usize, dense_len, @sizeOf(f64));
    const address = @intFromPtr(scratch.ptr);
    const aligned_address = std.mem.alignForward(usize, address, @alignOf(f64));
    const skip = aligned_address - address;
    const end = try std.math.add(usize, skip, dense_bytes_len);
    if (end > scratch.len) return error.CompactionWorkspaceLimit;
    const dense_bytes = scratch[skip..end];
    const aligned_bytes: []align(@alignOf(f64)) u8 = @alignCast(dense_bytes);
    const dense = std.mem.bytesAsSlice(f64, aligned_bytes);

    const kk = std.math.cast(c_int, k) orelse return error.InvalidAttentionShape;
    const dd = std.math.cast(c_int, dim) orelse return error.InvalidAttentionShape;
    @"cblas_dgemm$NEWLAPACK"(row_major, no_transpose, transpose, kk, kk, dd, 1, w.ptr, dd, w.ptr, dd, 0, dense.ptr, kk);
    try control.check();
    for (dense) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttention;

    var at: usize = 0;
    for (0..k) |row| for (0..row + 1) |column| {
        packed_output[at] = dense[row * k + column];
        at += 1;
    };
    std.debug.assert(at == packed_output.len);
    context.successful_calls +|= 1;
}

pub fn backend(context: *Context) gram.GramBackend {
    return .{ .context = context, .workspace_fn = workspace, .compute_fn = compute };
}

test "Accelerate Gram backend uses unaligned bounded workspace and matches direct dots" {
    if (comptime !supported) return error.SkipZigTest;
    var context: Context = .{};
    const actual = backend(&context);
    const w = [_]f64{
        1,  2,   3,
        -2, 0.5, 4,
        0,  -1,  2,
    };
    var packed_output = [_]f64{ 0, 0, 0, 0, 0, 0 };
    var storage: [9 * @sizeOf(f64) + @alignOf(f64)]u8 align(@alignOf(f64)) = undefined;
    const scratch = storage[1..];
    try actual.compute_fn(actual.context, &w, 3, 3, &packed_output, scratch, .{});
    var expected: [6]f64 = undefined;
    var at: usize = 0;
    for (0..3) |row| for (0..row + 1) |column| {
        var dot: f64 = 0;
        for (0..3) |di| dot += w[row * 3 + di] * w[column * 3 + di];
        expected[at] = dot;
        at += 1;
    };
    try std.testing.expectEqualSlices(f64, &expected, &packed_output);
    try std.testing.expectEqual(@as(usize, 1), context.successful_calls);
}

test "Accelerate Gram backend fails closed before completion" {
    if (comptime !supported) return error.SkipZigTest;
    const Stop = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var context: Context = .{};
    const actual = backend(&context);
    const w = [_]f64{ 1, 2, 3, 4 };
    var packed_output = [_]f64{ 0, 0, 0 };
    var storage: [4 * @sizeOf(f64) + @alignOf(f64) - 1]u8 = undefined;
    try std.testing.expectError(error.Cancelled, actual.compute_fn(actual.context, &w, 2, 2, &packed_output, &storage, .{ .check_fn = Stop.check }));
    try std.testing.expectEqualSlices(f64, &.{ 0, 0, 0 }, &packed_output);
    try std.testing.expectEqual(@as(usize, 0), context.successful_calls);
    try std.testing.expectError(error.CompactionWorkspaceLimit, actual.compute_fn(actual.context, &w, 2, 2, &packed_output, storage[0 .. storage.len - 1], .{}));
    try std.testing.expectError(error.InvalidAttentionShape, actual.workspace_fn(actual.context, 0, 2));
    var nonfinite = w;
    nonfinite[0] = std.math.nan(f64);
    try std.testing.expectError(error.NonFiniteAttention, actual.compute_fn(actual.context, &nonfinite, 2, 2, &packed_output, &storage, .{}));
    try std.testing.expectEqual(@as(usize, 0), context.successful_calls);
}
