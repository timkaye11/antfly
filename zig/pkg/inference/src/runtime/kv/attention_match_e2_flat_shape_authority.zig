//! Default-off E2/8128 flat shape authority. No live activation.
const std = @import("std");

pub const schema = "antfly.context.resident-native-hybrid-e2-8128.shape-authority.v1";
pub const policy = "e2_flat_shape_authority_v4|source8128|dim512|consumer_batches=56|per_owner=8,8,40|heads=4:0,9:0,14:0|census=2adaptive+1exact|prefix128|recent2048|chunk256|P256|H512|F256|HHO64|primary60|fallbacks0|activation=disabled";
pub const formal_fit_policy = "e2_flat_shape_authority_v5|source_rows=7936,32512|dim512|consumer_batches=56|per_owner=8,8,40|heads=4:0,9:0,14:0|census=2adaptive+1exact|prefix128|recent2048|chunk256|P256|H512|F256|HHO64|primary60|fallbacks0|activation=guarded_formal_fit";
pub const HeadIdentity = struct { donor_layer: u32, head_index: usize };
pub const canonical_heads = [_]HeadIdentity{ .{ .donor_layer = 4, .head_index = 0 }, .{ .donor_layer = 9, .head_index = 0 }, .{ .donor_layer = 14, .head_index = 0 } };
pub const canonical_owner_consumer_counts = [_]u8{ 8, 8, 40 };

pub const Shape = struct {
    source_rows: usize = 8128,
    dim: usize = 512,
    batch_count: usize = 56,
    eligible_heads: usize = 3,
    prefix_rows: usize = 128,
    recent_rows: usize = 2048,
    chunk_rows: usize = 256,
    primary_percent: u8 = 60,
    maximum_fallbacks: u8 = 0,
    planned_exact_heads: u8 = 1,
    primary_rows: usize = 256,
    dense_historical_rows: usize = 512,
    future_rows: usize = 256,
    heldout_rows: usize = 64,
    authority_sha256: [32]u8,
};
pub const Validated = struct { shape: Shape, range_count: usize, candidate_count: usize, fixed_rows: usize, digest: [32]u8 };

fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}
pub fn validate(shape: Shape) !Validated {
    if ((shape.source_rows != 7_936 and shape.source_rows != 8_128 and shape.source_rows != 32_512) or shape.dim != 512 or shape.batch_count != 56 or shape.eligible_heads != 3 or
        shape.prefix_rows != 128 or shape.recent_rows != 2048 or shape.chunk_rows != 256 or
        shape.primary_percent != 60 or shape.maximum_fallbacks != 0 or shape.planned_exact_heads != 1 or
        shape.primary_rows != 256 or shape.dense_historical_rows != 512 or shape.future_rows != 256 or shape.heldout_rows != 64 or !nonzero(shape.authority_sha256))
        return error.UnsupportedShape;
    const fixed = try std.math.add(usize, shape.prefix_rows, shape.recent_rows);
    const middle = shape.source_rows - fixed;
    const ranges = std.math.divCeil(usize, middle, shape.chunk_rows) catch return error.InvalidFitConfig;
    const candidates = try std.math.mul(usize, ranges, 3);
    const expected_ranges: usize = if (shape.source_rows == 7_936) 23 else if (shape.source_rows == 8_128) 24 else 119;
    if (ranges != expected_ranges or candidates != expected_ranges * 3) return error.UnsupportedShape;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(if (shape.source_rows == 8_128) policy else formal_fit_policy);
    hash.update(&shape.authority_sha256);
    inline for (@typeInfo(Shape).@"struct".field_names, @typeInfo(Shape).@"struct".field_types) |name, T| if (T == usize or T == u8) {
        var frame: [8]u8 = undefined;
        std.mem.writeInt(u64, &frame, @intCast(@field(shape, name)), .little);
        hash.update(&frame);
    };
    for (canonical_heads) |identity| {
        hash.update(std.mem.asBytes(&identity.donor_layer));
        hash.update(std.mem.asBytes(&identity.head_index));
    }
    return .{ .shape = shape, .range_count = ranges, .candidate_count = candidates, .fixed_rows = fixed, .digest = hash.finalResult() };
}

test "E2 authority seals real model identities and 8128 flat geometry" {
    const v = try validate(.{ .authority_sha256 = @splat(1) });
    try std.testing.expectEqual(@as(usize, 24), v.range_count);
    try std.testing.expectEqual(@as(usize, 72), v.candidate_count);
    try std.testing.expectEqual(@as(usize, 56), v.shape.batch_count);
    try std.testing.expectEqualDeep([_]HeadIdentity{ .{ .donor_layer = 4, .head_index = 0 }, .{ .donor_layer = 9, .head_index = 0 }, .{ .donor_layer = 14, .head_index = 0 } }, canonical_heads);
}
test "E2 authority rejects E4 policy and unauthenticated shape" {
    var wrong = Shape{ .authority_sha256 = @splat(1) };
    wrong.eligible_heads = 8;
    try std.testing.expectError(error.UnsupportedShape, validate(wrong));
    try std.testing.expectError(error.UnsupportedShape, validate(.{ .authority_sha256 = @splat(0) }));
    wrong = .{ .authority_sha256 = @splat(1) };
    wrong.batch_count = 40;
    try std.testing.expectError(error.UnsupportedShape, validate(wrong));
}
test "E2 formal fit shapes have distinct authenticated range geometry" {
    const short = try validate(.{ .source_rows = 7_936, .authority_sha256 = @splat(1) });
    const long = try validate(.{ .source_rows = 32_512, .authority_sha256 = @splat(1) });
    try std.testing.expectEqual(@as(usize, 23), short.range_count);
    try std.testing.expectEqual(@as(usize, 69), short.candidate_count);
    try std.testing.expectEqual(@as(usize, 119), long.range_count);
    try std.testing.expectEqual(@as(usize, 357), long.candidate_count);
    try std.testing.expect(!std.mem.eql(u8, &short.digest, &long.digest));
}
