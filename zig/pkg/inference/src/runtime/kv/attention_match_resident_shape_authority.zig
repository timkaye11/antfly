const std = @import("std");
const flat_shape = @import("attention_match_flat_shape_authority.zig");
const wide_shape = @import("attention_match_e4_32512_shape_authority.zig");

/// A closed resident-source shape domain. The legacy flat arm delegates all
/// validation and digest construction to flat_shape so its authority bytes are
/// unchanged. The 32K arm is independently authenticated by the wide authority.
pub const Binding = union(enum) {
    flat: flat_shape.Shape,
    e4_32512: wide_shape.Binding,
};

pub const Validated = struct {
    binding: Binding,
    source_rows: usize,
    dim: usize,
    batch_count: usize,
    eligible_heads: usize,
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    primary_rows: usize,
    dense_historical_rows: usize,
    selected_historical_rows: usize,
    future_rows: usize,
    heldout_rows: usize,
    primary_percent: u8,
    fallback_percent: u8,
    maximum_fallbacks: u8,
    range_count: usize,
    candidate_count: usize,
    digest: [32]u8,
    requires_bounded_page: bool,
};

pub fn defaultFlat(source_freeze_sha256: [32]u8) Binding {
    return .{ .flat = flat_shape.Shape.sealed7936(source_freeze_sha256) };
}

pub fn validate(binding: Binding) !Validated {
    return switch (binding) {
        .flat => |shape| blk: {
            const value = try flat_shape.validate(shape);
            break :blk .{
                .binding = binding,
                .source_rows = value.shape.source_rows,
                .dim = value.shape.dim,
                .batch_count = value.shape.batch_count,
                .eligible_heads = value.shape.eligible_heads,
                .prefix_rows = value.shape.prefix_rows,
                .recent_rows = value.shape.recent_rows,
                .chunk_rows = value.shape.chunk_rows,
                .primary_rows = value.shape.primary_rows,
                .dense_historical_rows = value.shape.dense_historical_rows,
                .selected_historical_rows = value.shape.selected_historical_rows,
                .future_rows = value.shape.future_rows,
                .heldout_rows = value.shape.heldout_rows,
                .primary_percent = value.shape.primary_percent,
                .fallback_percent = value.shape.fallback_percent,
                .maximum_fallbacks = value.shape.maximum_fallbacks,
                .range_count = value.range_count,
                .candidate_count = value.candidate_count,
                .digest = value.digest,
                .requires_bounded_page = false,
            };
        },
        .e4_32512 => |shape| blk: {
            const value = try wide_shape.validate(shape);
            break :blk .{
                .binding = binding,
                .source_rows = wide_shape.source_rows,
                .dim = wide_shape.dim,
                .batch_count = wide_shape.batch_count,
                .eligible_heads = wide_shape.eligible_heads,
                .prefix_rows = wide_shape.prefix_rows,
                .recent_rows = wide_shape.recent_rows,
                .chunk_rows = wide_shape.chunk_rows,
                .primary_rows = wide_shape.primary_rows,
                .dense_historical_rows = wide_shape.dense_historical_rows,
                .selected_historical_rows = wide_shape.selected_historical_rows,
                .future_rows = wide_shape.future_rows,
                .heldout_rows = wide_shape.heldout_rows,
                .primary_percent = wide_shape.primary_percent,
                .fallback_percent = wide_shape.fallback_percent,
                .maximum_fallbacks = wide_shape.maximum_fallbacks,
                .range_count = value.range_count,
                .candidate_count = value.candidate_count,
                .digest = value.digest,
                .requires_bounded_page = true,
            };
        },
    };
}

pub fn explicitBindingValid(value: Validated, source_freeze_sha256: [32]u8, expected_digest: [32]u8) bool {
    if (std.mem.allEqual(u8, &expected_digest, 0) or
        !std.crypto.timing_safe.eql([32]u8, value.digest, expected_digest)) return false;
    return switch (value.binding) {
        .flat => |shape| std.crypto.timing_safe.eql([32]u8, shape.authority_sha256, source_freeze_sha256),
        .e4_32512 => |shape| std.crypto.timing_safe.eql([32]u8, shape.source_freeze_sha256, source_freeze_sha256) and
            std.crypto.timing_safe.eql([32]u8, shape.expected_digest, expected_digest),
    };
}

fn wideBinding(seed: u8) wide_shape.Binding {
    var value: wide_shape.Binding = .{
        .source_freeze_sha256 = @splat(seed),
        .configuration_sha256 = @splat(seed + 1),
        .pressure_policy_sha256 = @splat(seed + 2),
        .packed_source_policy_sha256 = @splat(seed + 3),
        .expected_digest = @splat(0),
    };
    value.expected_digest = wide_shape.digest(value) catch unreachable;
    return value;
}

test "resident shape preserves legacy flat digest bytes and 7936 8128 caps" {
    inline for (.{ @as(usize, 7936), @as(usize, 8128) }) |rows| {
        var shape = flat_shape.Shape.sealed7936(@splat(7));
        shape.source_rows = rows;
        const legacy = try flat_shape.validate(shape);
        const resident = try validate(.{ .flat = shape });
        try std.testing.expectEqualSlices(u8, &legacy.digest, &resident.digest);
        try std.testing.expectEqual(legacy.range_count, resident.range_count);
        try std.testing.expectEqual(legacy.candidate_count, resident.candidate_count);
        try std.testing.expect(!resident.requires_bounded_page);
        try std.testing.expect(explicitBindingValid(resident, @splat(7), legacy.digest));
    }
}

test "resident shape admits only authenticated 32512 bounded authority" {
    const binding = wideBinding(11);
    const value = try validate(.{ .e4_32512 = binding });
    try std.testing.expectEqual(@as(usize, 32_512), value.source_rows);
    try std.testing.expectEqual(@as(usize, 119), value.range_count);
    try std.testing.expectEqual(@as(usize, 357), value.candidate_count);
    try std.testing.expect(value.requires_bounded_page);
    try std.testing.expect(explicitBindingValid(value, @splat(11), binding.expected_digest));
    try std.testing.expect(!explicitBindingValid(value, @splat(12), binding.expected_digest));
    var stale = binding;
    stale.expected_digest[0] +%= 1;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, validate(.{ .e4_32512 = stale }));
}

test "resident shape does not widen flat arm to 32512 or accept zero explicit digest" {
    var flat = flat_shape.Shape.sealed7936(@splat(1));
    flat.source_rows = 32_512;
    try std.testing.expectError(error.UnsupportedShape, validate(.{ .flat = flat }));
    const value = try validate(defaultFlat(@splat(1)));
    try std.testing.expect(!explicitBindingValid(value, @splat(1), @splat(0)));
}
