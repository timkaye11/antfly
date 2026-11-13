const std = @import("std");
const engine = @import("attention_match_engine.zig");

pub const schema = "antfly.attention-match-recent64-future-primary-hybrid.v1";
pub const batch_count: usize = 56;
pub const head_dim: usize = 512;
pub const rows_per_role: usize = 64;

pub const Mode = enum { fixed_recent32_future32, full_recent64_future64, full_recent64_future64_phase_balanced };

pub const Owned = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    recent_authority_sha256: [32]u8,
    future_primary_authority_sha256: [32]u8,
    digest: [32]u8,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

fn rows(mode: Mode) usize {
    return switch (mode) {
        .fixed_recent32_future32 => 64,
        .full_recent64_future64, .full_recent64_future64_phase_balanced => 128,
    };
}

fn take(mode: Mode) usize {
    return switch (mode) {
        .fixed_recent32_future32 => 32,
        .full_recent64_future64, .full_recent64_future64_phase_balanced => 64,
    };
}

fn validateBatch(recent: engine.ReferenceBatch, future: engine.ReferenceBatch, mode: Mode) !void {
    if (recent.consumer_layer != future.consumer_layer or recent.query_head != future.query_head or
        recent.train.len != rows_per_role * head_dim or recent.train_positions.len != rows_per_role or
        future.train.len != rows_per_role * head_dim or future.train_positions.len != rows_per_role or
        recent.heldout.len != 0 or recent.heldout_positions.len != 0 or
        future.heldout.len != 0 or future.heldout_positions.len != 0)
        return error.InvalidRecentFuturePrimaryHybrid;
    for (recent.train_positions, 0..) |position, index| if (position != 8064 + index)
        return error.InvalidRecentFuturePrimaryHybrid;
    for (future.train_positions, 0..) |position, index| if (position != switch (mode) {
        .full_recent64_future64_phase_balanced => 8129 + 8 * index + index % 7,
        else => 8132 + 8 * index,
    })
        return error.InvalidRecentFuturePrimaryHybrid;
    for (recent.train_positions) |a| for (future.train_positions) |b| if (a == b)
        return error.InvalidRecentFuturePrimaryHybrid;
}

pub fn merge(
    allocator: std.mem.Allocator,
    recent: []const engine.ReferenceBatch,
    recent_authority_sha256: [32]u8,
    future: []const engine.ReferenceBatch,
    future_primary_authority_sha256: [32]u8,
    mode: Mode,
) !Owned {
    if (recent.len != batch_count or future.len != batch_count or
        std.mem.allEqual(u8, &recent_authority_sha256, 0) or
        std.mem.allEqual(u8, &future_primary_authority_sha256, 0))
        return error.InvalidRecentFuturePrimaryHybrid;
    const role_rows = take(mode);
    const output_rows = rows(mode);
    const values = try allocator.alloc(f32, batch_count * output_rows * head_dim);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, batch_count * output_rows);
    errdefer allocator.free(positions);
    const batches = try allocator.alloc(engine.ReferenceBatch, batch_count);
    errdefer allocator.free(batches);
    for (recent, future, batches, 0..) |r, f, *out, index| {
        try validateBatch(r, f, mode);
        const value_out = values[index * output_rows * head_dim ..][0 .. output_rows * head_dim];
        const position_out = positions[index * output_rows ..][0..output_rows];
        // The fixed ablation uses the most recent 32 causal source rows and
        // the first 32 independently teacher-forced future-primary rows.
        const recent_start = rows_per_role - role_rows;
        @memcpy(value_out[0 .. role_rows * head_dim], r.train[recent_start * head_dim ..][0 .. role_rows * head_dim]);
        @memcpy(position_out[0..role_rows], r.train_positions[recent_start..][0..role_rows]);
        if (mode == .fixed_recent32_future32) {
            // Cover the full 512-token teacher-forced continuation rather than
            // concentrating the ablation in its first half.
            for (0..role_rows) |row| {
                const source_row = 2 * row;
                @memcpy(value_out[(role_rows + row) * head_dim ..][0..head_dim], f.train[source_row * head_dim ..][0..head_dim]);
                position_out[role_rows + row] = f.train_positions[source_row];
            }
        } else {
            @memcpy(value_out[role_rows * head_dim ..], f.train[0 .. role_rows * head_dim]);
            @memcpy(position_out[role_rows..], f.train_positions[0..role_rows]);
        }
        out.* = .{
            .consumer_layer = r.consumer_layer,
            .query_head = r.query_head,
            .train = value_out,
            .train_positions = position_out,
            .heldout = &.{},
            .heldout_positions = &.{},
            .independent_train_split = role_rows,
        };
    }
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(schema ++ "\x00");
    hash.update(@tagName(mode));
    hash.update(&recent_authority_sha256);
    hash.update(&future_primary_authority_sha256);
    hash.update(std.mem.sliceAsBytes(values));
    hash.update(std.mem.sliceAsBytes(positions));
    return .{
        .allocator = allocator,
        .batches = batches,
        .values = values,
        .positions = positions,
        .recent_authority_sha256 = recent_authority_sha256,
        .future_primary_authority_sha256 = future_primary_authority_sha256,
        .digest = hash.finalResult(),
    };
}

test "hybrid fixed split preserves roles positions and canonical order" {
    var recent_values: [batch_count][rows_per_role * head_dim]f32 = undefined;
    var future_values: [batch_count][rows_per_role * head_dim]f32 = undefined;
    var recent_positions: [batch_count][rows_per_role]u64 = undefined;
    var future_positions: [batch_count][rows_per_role]u64 = undefined;
    var recent_batches: [batch_count]engine.ReferenceBatch = undefined;
    var future_batches: [batch_count]engine.ReferenceBatch = undefined;
    for (0..batch_count) |batch| {
        for (&recent_values[batch], 0..) |*value, index| value.* = @floatFromInt(index);
        for (&future_values[batch], 0..) |*value, index| value.* = @floatFromInt(1_000_000 + index);
        for (&recent_positions[batch], 0..) |*position, index| position.* = 8064 + index;
        for (&future_positions[batch], 0..) |*position, index| position.* = 8132 + 8 * index;
        recent_batches[batch] = .{ .consumer_layer = @intCast(5 + 6 * (batch / 8)), .query_head = @intCast(batch % 8), .train = &recent_values[batch], .train_positions = &recent_positions[batch], .heldout = &.{}, .heldout_positions = &.{} };
        future_batches[batch] = .{ .consumer_layer = recent_batches[batch].consumer_layer, .query_head = recent_batches[batch].query_head, .train = &future_values[batch], .train_positions = &future_positions[batch], .heldout = &.{}, .heldout_positions = &.{} };
    }
    var owned = try merge(std.testing.allocator, &recent_batches, @splat(1), &future_batches, @splat(2), .fixed_recent32_future32);
    defer owned.deinit();
    try std.testing.expectEqual(@as(usize, 32), owned.batches[0].independent_train_split.?);
    try std.testing.expectEqual(@as(u64, 8096), owned.batches[0].train_positions[0]);
    try std.testing.expectEqual(@as(u64, 8132), owned.batches[0].train_positions[32]);
    try std.testing.expectEqual(@as(f32, 32 * head_dim), owned.batches[0].train[0]);
    try std.testing.expectEqual(@as(f32, 1_000_000), owned.batches[0].train[32 * head_dim]);
    try std.testing.expectEqual(@as(u64, 8628), owned.batches[0].train_positions[63]);
}

test "hybrid rejects HHO offset and cross-role collisions" {
    var rv: [rows_per_role * head_dim]f32 = @splat(0);
    var fv: [rows_per_role * head_dim]f32 = @splat(0);
    var rp: [rows_per_role]u64 = undefined;
    var fp: [rows_per_role]u64 = undefined;
    for (&rp, 0..) |*p, i| p.* = 8064 + i;
    for (&fp, 0..) |*p, i| p.* = 8128 + 8 * i;
    const r: engine.ReferenceBatch = .{ .consumer_layer = 5, .query_head = 0, .train = &rv, .train_positions = &rp, .heldout = &.{}, .heldout_positions = &.{} };
    const f: engine.ReferenceBatch = .{ .consumer_layer = 5, .query_head = 0, .train = &fv, .train_positions = &fp, .heldout = &.{}, .heldout_positions = &.{} };
    try std.testing.expectError(error.InvalidRecentFuturePrimaryHybrid, validateBatch(r, f, .full_recent64_future64));
}
