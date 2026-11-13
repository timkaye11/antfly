// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shared ownership-only plumbing for a TRAIN-planned exact attention head.
//! Policy selection remains with the caller; this module only clones the
//! selected source head or delegates every other head to the adaptive path.
const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");

pub const Context = struct {
    physical: adaptive.PhysicalBudgetContext = .{},
    enabled: bool,
    donor_layer: u32,
    head_index: usize,
    adaptive_fit_attempts: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    planned_exact_heads: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    pub fn callback(raw: ?*anyopaque, allocator: std.mem.Allocator, source: engine.SourceMetadata, head: checkpoint.Head, donor: usize, head_index: usize, batches: []const engine.ReferenceBatch, layout: engine.Layout, packed_plan: [4]usize, options: *const engine.Options, resident: usize) anyerror!checkpoint.Head {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidFitConfig));
        if (self.enabled and try source.donorLayer(donor) == self.donor_layer and head_index == self.head_index) {
            const cloned = try cloneHead(allocator, head, options.fit.control);
            _ = self.planned_exact_heads.fetchAdd(1, .acq_rel);
            return cloned;
        }
        _ = self.adaptive_fit_attempts.fetchAdd(1, .acq_rel);
        return adaptive.engineCallback(&self.physical, allocator, source, head, donor, head_index, batches, layout, packed_plan, options, resident);
    }
};

pub fn cloneHead(allocator: std.mem.Allocator, source: checkpoint.Head, control: solver.Control) !checkpoint.Head {
    try control.check();
    return checkpoint.cloneHead(allocator, source);
}

pub fn freeHead(allocator: std.mem.Allocator, head: checkpoint.Head) void {
    var compressed = head.compressed;
    compressed.deinit(allocator);
    allocator.free(head.bias);
    allocator.free(head.values);
    allocator.free(head.keys);
    allocator.free(head.coverage_ends);
    allocator.free(head.positions);
}

fn sameBytes(a: []const f32, b: []const f32) bool {
    return std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

fn sameHead(a: checkpoint.Head, b: checkpoint.Head) bool {
    return a.dim == b.dim and a.compacted_until == b.compacted_until and
        std.mem.eql(u64, a.positions, b.positions) and std.mem.eql(u64, a.coverage_ends, b.coverage_ends) and
        sameBytes(a.keys, b.keys) and sameBytes(a.values, b.values) and sameBytes(a.bias, b.bias);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    const source: checkpoint.Head = .{ .dim = 1, .compacted_until = 2, .positions = &.{ 0, 1 }, .coverage_ends = &.{ 1, 2 }, .keys = &.{ @as(f32, @bitCast(@as(u32, 0x80000000))), 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 } };
    const cloned = try cloneHead(allocator, source, .{});
    defer freeHead(allocator, cloned);
    try std.testing.expect(sameHead(source, cloned));
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(cloned.keys[0])));
    @constCast(cloned.keys)[0] = 0;
    try std.testing.expect(!sameHead(source, cloned));
}

test "planned exact clone preserves signed zero and is atomic under allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "planned exact clone checks cancellation before allocation" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    const source: checkpoint.Head = .{ .dim = 1, .compacted_until = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &.{1}, .values = &.{2}, .bias = &.{0} };
    try std.testing.expectError(error.Cancelled, cloneHead(std.testing.failing_allocator, source, .{ .check_fn = Cancel.check }));
}

test "planned exact callback counts only a successful clone" {
    const source_head: checkpoint.Head = .{ .dim = 1, .compacted_until = 1, .positions = &.{0}, .coverage_ends = &.{1}, .keys = &.{1}, .values = &.{2}, .bias = &.{0} };
    const donors = [_]checkpoint.Donor{.{ .layer = 0, .heads = &.{source_head} }};
    const source: checkpoint.State = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 1, .tokens = &.{0}, .logits = &.{0}, .layer_donors = &.{0}, .donors = &donors };
    const metadata = try engine.sourceMetadataFromState(source, .{}, .{});
    var context: Context = .{ .enabled = true, .donor_layer = 0, .head_index = 0 };
    var options: engine.Options = .{};
    const cloned = try Context.callback(&context, std.testing.allocator, metadata, source_head, 0, 0, &.{}, undefined, .{ 1, 0, 1, 1 }, &options, 0);
    defer freeHead(std.testing.allocator, cloned);
    try std.testing.expectEqual(@as(usize, 1), context.planned_exact_heads.load(.acquire));

    context.planned_exact_heads.store(0, .release);
    try std.testing.expectError(error.OutOfMemory, Context.callback(&context, std.testing.failing_allocator, metadata, source_head, 0, 0, &.{}, undefined, .{ 1, 0, 1, 1 }, &options, 0));
    try std.testing.expectEqual(@as(usize, 0), context.planned_exact_heads.load(.acquire));

    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    options.fit.control = .{ .check_fn = Cancel.check };
    try std.testing.expectError(error.Cancelled, Context.callback(&context, std.testing.allocator, metadata, source_head, 0, 0, &.{}, undefined, .{ 1, 0, 1, 1 }, &options, 0));
    try std.testing.expectEqual(@as(usize, 0), context.planned_exact_heads.load(.acquire));
}
