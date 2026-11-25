// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const solver = @import("attention_match.zig");
const storage = @import("../../ops/metal_attention_match_storage.zig");

pub const Context = struct {
    allocator: std.mem.Allocator,
    state: *storage.DeviceState,
    donors: []const checkpoint.StreamingDonor,
    maximum_head_download_bytes: usize,
    active: ?storage.DownloadedHead = null,

    pub fn provider(self: *@This()) checkpoint.StreamingHeadProvider {
        return .{ .context = self, .open_fn = open, .verify_empty_fn = verifyEmpty };
    }
    pub fn deinit(self: *@This()) void {
        if (self.active != null) @panic("device checkpoint provider destroyed with live head");
        self.* = undefined;
    }
    fn open(raw: ?*anyopaque, donor: usize, head_index: usize, control: solver.Control) !checkpoint.StreamingHeadLease {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidKvCheckpoint));
        if (self.active != null or donor >= self.donors.len or head_index >= self.donors[donor].head_count) return error.InvalidKvCheckpoint;
        self.active = try self.state.downloadHead(self.allocator, self.donors[donor].layer, head_index, self.donors[donor].window, self.maximum_head_download_bytes, control);
        return .{ .context = self, .close_fn = close, .donor = donor, .head_index = head_index, .head = self.active.?.head() };
    }
    fn close(raw: ?*anyopaque, _: *checkpoint.StreamingHeadLease) void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse @panic("missing device checkpoint provider")));
        if (self.active) |*active| active.deinit() else @panic("device checkpoint provider close without open");
        self.active = null;
    }
    fn verifyEmpty(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidKvCheckpoint));
        if (self.active != null) return error.IncompleteContextAllocationScope;
        try self.state.synchronize();
    }
};

test "candidate builder device provider is byte exact for every production codec" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const packed_storage = checkpoint.packed_storage;
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const bias = [_]f32{ 0, 0 };
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const Compare = struct {
        fn run(head: checkpoint.Head, encoding: checkpoint.Encoding, expected_codec_rows: usize) !void {
            var stage: []const u8 = "source_validate";
            var completed = false;
            defer if (!completed) std.debug.print("device_checkpoint_codec_failure encoding={s} stage={s} compacted_until={d} fp8_rows={d} packed_rows={d}\n", .{ @tagName(encoding), stage, head.compacted_until, head.fp8.rows, head.compressed.rows() });
            try checkpoint.validateHeadForSource(head, 2, 0, checkpoint.serving_limits, .{});
            stage = "runner_init";
            var runner = try storage.Runner.init(32 * 1024 * 1024, null);
            defer runner.deinit();
            const backend = runner.backend(std.testing.allocator, .{});
            const descriptors = [_]storage.CandidateHeadDescriptor{.{ .layer = 0, .head = 0, .window = 0 }};
            stage = "builder_init";
            var builder = try storage.CandidateBuilder.init(backend, 2, &descriptors, @splat(5), @splat(6), 16 * 1024 * 1024, .{});
            defer builder.deinit();
            stage = "builder_upload";
            try builder.uploadHead(0, head, .{});
            stage = "builder_finish";
            var device = try builder.finish(@splat(5), @splat(6), .{});
            defer device.deinit();
            const donors = [_]checkpoint.StreamingDonor{.{ .layer = 0, .window = 0, .head_count = 1 }};
            const metadata: checkpoint.StreamingMetadata = .{ .identity = identity, .logical_length = 2, .tokens = &.{ 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = &donors };
            var provider_context: Context = .{ .allocator = std.testing.allocator, .state = &device, .donors = &donors, .maximum_head_download_bytes = 4 * 1024 * 1024 };
            defer provider_context.deinit();
            const expected_state: checkpoint.State = .{ .identity = identity, .logical_length = 2, .tokens = metadata.tokens, .logits = metadata.logits, .layer_donors = metadata.layer_donors, .donors = &.{.{ .layer = 0, .window = 0, .heads = &.{head} }} };
            stage = "expected_size";
            const expected_size = try checkpoint.encodedSize(expected_state, checkpoint.serving_limits, encoding, .{});
            stage = "provider_size";
            const actual_size = try checkpoint.encodedSizeFromProvider(metadata, provider_context.provider(), checkpoint.serving_limits, encoding, .{});
            try std.testing.expectEqual(expected_size, actual_size);
            const Sink = struct {
                list: std.ArrayList(u8) = .empty,
                pub fn writeAll(self: *@This(), bytes: []const u8) !void {
                    try self.list.appendSlice(std.testing.allocator, bytes);
                }
                fn deinit(self: *@This()) void {
                    self.list.deinit(std.testing.allocator);
                }
            };
            var expected: Sink = .{};
            defer expected.deinit();
            var actual: Sink = .{};
            defer actual.deinit();
            stage = "expected_export";
            const expected_digest = try checkpoint.exportEncoded(expected_state, checkpoint.serving_limits, encoding, &expected, .{});
            stage = "provider_export";
            const actual_digest = try checkpoint.exportEncodedFromProvider(metadata, provider_context.provider(), checkpoint.serving_limits, encoding, &actual, .{});
            try std.testing.expectEqualDeep(expected_digest, actual_digest);
            try std.testing.expectEqualSlices(u8, expected.list.items, actual.list.items);
            stage = "final_download";
            var downloaded = try device.downloadHead(std.testing.allocator, 0, 0, 0, 4 * 1024 * 1024, .{});
            defer downloaded.deinit();
            stage = "final_download_validate";
            try checkpoint.validateHeadForSource(downloaded.head(), 2, 0, checkpoint.serving_limits, .{});
            const observed_codec_rows = if (encoding == .fp8) downloaded.head().fp8.rows else if (encoding == .rotated4) downloaded.head().compressed.rows() else 0;
            try std.testing.expectEqual(expected_codec_rows, observed_codec_rows);
            completed = true;
        }
    };
    const exact_keys: [2 * 64]f32 = @splat(0.25);
    const exact_values: [2 * 64]f32 = @splat(0.5);
    const exact: checkpoint.Head = .{ .dim = 64, .compacted_until = 2, .positions = &positions, .coverage_ends = &coverage, .keys = &exact_keys, .values = &exact_values, .bias = &bias };
    try Compare.run(exact, .f16, 0);
    var fp8 = exact;
    fp8.fp8 = .{ .first_row = 0, .rows = 2, .key_scale = 0.25, .value_scale = 0.5 };
    try Compare.run(fp8, .fp8, 2);
    var encoded = try (packed_storage.Packed{}).append(std.testing.allocator, 0, 64, exact_keys[0..64], exact_values[0..64], .{});
    defer encoded.deinit(std.testing.allocator);
    var rotated_keys = exact_keys;
    var rotated_values = exact_values;
    try encoded.decodeRow(64, 0, rotated_keys[0..64], rotated_values[0..64]);
    var rotated = exact;
    rotated.keys = &rotated_keys;
    rotated.values = &rotated_values;
    rotated.compressed = encoded;
    try Compare.run(rotated, .rotated4, 1);
}
