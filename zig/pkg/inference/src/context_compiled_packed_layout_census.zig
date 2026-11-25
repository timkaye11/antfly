const std = @import("std");

pub const Layer = struct { layer: u32, owner: u32 };
pub const Census = struct { attention_layers: u32, owner_count: u32, layout_sha256: [32]u8, owner_projection_sha256: [32]u8 };

pub fn derive(layers: []const Layer, layout_sha256: [32]u8) !Census {
    if (layers.len == 0 or layers.len > std.math.maxInt(u32) or std.mem.allEqual(u8, &layout_sha256, 0)) return error.InvalidCompiledPackedLayoutCensus;
    var owners: u32 = 0;
    var projection = std.crypto.hash.sha2.Sha256.init(.{});
    projection.update("antfly.compiled-packed-owner-projection.v1\x00");
    for (layers, 0..) |entry, i| {
        if (entry.layer != i or entry.owner >= layers.len or entry.owner > entry.layer) return error.InvalidCompiledPackedLayoutCensus;
        if (layers[entry.owner].owner != entry.owner) return error.InvalidCompiledPackedLayoutCensus;
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(u32, encoded[0..4], entry.layer, .little);
        std.mem.writeInt(u32, encoded[4..8], entry.owner, .little);
        projection.update(&encoded);
        if (entry.owner == entry.layer) owners += 1;
    }
    if (owners == 0) return error.InvalidCompiledPackedLayoutCensus;
    return .{ .attention_layers = @intCast(layers.len), .owner_count = owners, .layout_sha256 = layout_sha256, .owner_projection_sha256 = projection.finalResult() };
}

test "compiled packed census derives roots and direct sharing" {
    const layers = [_]Layer{ .{ .layer = 0, .owner = 0 }, .{ .layer = 1, .owner = 0 }, .{ .layer = 2, .owner = 2 } };
    const census = try derive(&layers, @splat(1));
    try std.testing.expectEqual(@as(u32, 3), census.attention_layers);
    try std.testing.expectEqual(@as(u32, 2), census.owner_count);
    try std.testing.expect(!std.mem.allEqual(u8, &census.owner_projection_sha256, 0));
}

test "compiled packed census rejects malformed and foreign layout" {
    var layers = [_]Layer{ .{ .layer = 0, .owner = 0 }, .{ .layer = 2, .owner = 0 } };
    try std.testing.expectError(error.InvalidCompiledPackedLayoutCensus, derive(&layers, @splat(1)));
    layers[1].layer = 1;
    layers[1].owner = 2;
    try std.testing.expectError(error.InvalidCompiledPackedLayoutCensus, derive(&layers, @splat(1)));
    layers[1].owner = 0;
    try std.testing.expectError(error.InvalidCompiledPackedLayoutCensus, derive(&layers, @splat(0)));
    const indirect = [_]Layer{ .{ .layer = 0, .owner = 0 }, .{ .layer = 1, .owner = 0 }, .{ .layer = 2, .owner = 1 } };
    try std.testing.expectError(error.InvalidCompiledPackedLayoutCensus, derive(&indirect, @splat(1)));
}
