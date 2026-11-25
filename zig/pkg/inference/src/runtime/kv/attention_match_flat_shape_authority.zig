const std = @import("std");

pub const policy = "flat_shape_authority_v1|E4_only|source=7936_or8128|dim512|consumers56|heads8|P256_Hdense512_Hselected256_F256_HHO64|prefix128_recent2048_chunk256|p91_p95_max3";

pub const Shape = struct {
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
    authority_sha256: [32]u8,

    pub fn sealed7936(authority_sha256: [32]u8) Shape {
        return .{ .source_rows = 7936, .dim = 512, .batch_count = 56, .eligible_heads = 8, .prefix_rows = 128, .recent_rows = 2048, .chunk_rows = 256, .primary_rows = 256, .dense_historical_rows = 512, .selected_historical_rows = 256, .future_rows = 256, .heldout_rows = 64, .primary_percent = 91, .fallback_percent = 95, .maximum_fallbacks = 3, .authority_sha256 = authority_sha256 };
    }
};

pub const Validated = struct {
    shape: Shape,
    range_count: usize,
    candidate_count: usize,
    fixed_rows: usize,
    query_rows_per_consumer: usize,
    digest: [32]u8,
};

fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}
pub fn validate(shape: Shape) !Validated {
    if ((shape.source_rows != 7936 and shape.source_rows != 8128) or shape.dim != 512 or shape.batch_count != 56 or shape.eligible_heads != 8 or shape.prefix_rows != 128 or shape.recent_rows != 2048 or shape.chunk_rows != 256 or shape.primary_rows != 256 or shape.dense_historical_rows != 512 or shape.selected_historical_rows != 256 or shape.future_rows != 256 or shape.heldout_rows != 64 or shape.primary_percent != 91 or shape.fallback_percent != 95 or shape.maximum_fallbacks != 3 or !nonzero(shape.authority_sha256)) return error.UnsupportedShape;
    const fixed = std.math.add(usize, shape.prefix_rows, shape.recent_rows) catch return error.InvalidFitConfig;
    if (fixed >= shape.source_rows) return error.InvalidFitConfig;
    const middle = shape.source_rows - fixed;
    const ranges = std.math.divCeil(usize, middle, shape.chunk_rows) catch return error.InvalidFitConfig;
    const candidates = std.math.mul(usize, ranges, 3) catch return error.InvalidFitConfig;
    if (candidates > 96) return error.UnsupportedShape;
    const query_rows = std.math.add(usize, shape.primary_rows, std.math.add(usize, shape.selected_historical_rows, shape.future_rows) catch return error.InvalidFitConfig) catch return error.InvalidFitConfig;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&shape.authority_sha256);
    inline for (@typeInfo(Shape).@"struct".field_names, @typeInfo(Shape).@"struct".field_types) |name, FieldType| if (FieldType == usize or FieldType == u8) {
        var bytes: [8]u8 = undefined;
        std.mem.writeInt(u64, &bytes, @intCast(@field(shape, name)), .little);
        hash.update(&bytes);
    };
    return .{ .shape = shape, .range_count = ranges, .candidate_count = candidates, .fixed_rows = fixed, .query_rows_per_consumer = query_rows, .digest = hash.finalResult() };
}

test "sealed development and formal 8128 flat shapes validate without E2 or bounded widening" {
    const dev = try validate(Shape.sealed7936(@splat(1)));
    try std.testing.expectEqual(@as(usize, 23), dev.range_count);
    var formal = Shape.sealed7936(@splat(2));
    formal.source_rows = 8128;
    const v = try validate(formal);
    try std.testing.expectEqual(@as(usize, 24), v.range_count);
    try std.testing.expectEqual(@as(usize, 72), v.candidate_count);
    formal.eligible_heads = 3;
    try std.testing.expectError(error.UnsupportedShape, validate(formal));
    formal.eligible_heads = 8;
    formal.source_rows = 32512;
    try std.testing.expectError(error.UnsupportedShape, validate(formal));
}
