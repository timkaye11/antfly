// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");

pub const Owned = struct {
    allocator: std.mem.Allocator,
    metadata: engine.SourceMetadata,
    inventory_sha256: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.metadata.donors);
        self.* = undefined;
    }
};

pub fn build(
    allocator: std.mem.Allocator,
    identity: checkpoint.Identity,
    tokens: []const u32,
    logits: []const f32,
    layer_donors: []const u32,
    inventory: metal_storage.InventoryOwned,
    expected_inventory_sha256: [32]u8,
    complete_state_sha256: [32]u8,
) !Owned {
    if (tokens.len != inventory.logical_length or logits.len == 0 or layer_donors.len == 0 or inventory.entries.len == 0 or
        inventory.physical_bytes == 0 or inventory.global_physical_bytes == 0 or
        !std.mem.eql(u8, &inventory.digest, &expected_inventory_sha256) or std.mem.allEqual(u8, &complete_state_sha256, 0))
        return error.InvalidTrainingAugmentationBinding;
    var donor_count: usize = 0;
    var previous_layer: ?u32 = null;
    for (inventory.entries, 0..) |entry, index| {
        if (entry.rows == 0 or entry.dim == 0 or entry.physical_bytes == 0 or entry.capacity < entry.rows or entry.start >= entry.capacity)
            return error.InvalidTrainingAugmentationBinding;
        if (previous_layer == null or previous_layer.? != entry.layer) {
            if (previous_layer != null and previous_layer.? >= entry.layer) return error.InvalidTrainingAugmentationBinding;
            if (entry.head != 0) return error.InvalidTrainingAugmentationBinding;
            donor_count += 1;
            previous_layer = entry.layer;
        } else if (entry.head != inventory.entries[index - 1].head + 1) return error.InvalidTrainingAugmentationBinding;
    }
    const donors = try allocator.alloc(engine.SourceDonorMetadata, donor_count);
    errdefer allocator.free(donors);
    var donor_at: usize = 0;
    var entry_at: usize = 0;
    while (entry_at < inventory.entries.len) {
        const first = inventory.entries[entry_at];
        var end = entry_at + 1;
        var physical = first.physical_bytes;
        while (end < inventory.entries.len and inventory.entries[end].layer == first.layer) : (end += 1) {
            const other = inventory.entries[end];
            if (other.window != first.window or other.dim != first.dim or other.rows != first.rows)
                return error.InvalidTrainingAugmentationBinding;
            physical = std.math.add(usize, physical, other.physical_bytes) catch return error.CompactionWorkspaceLimit;
        }
        donors[donor_at] = .{ .layer = first.layer, .window = first.window, .head_count = end - entry_at, .head_dim = first.dim, .rows = first.rows, .physical_bytes = physical };
        donor_at += 1;
        entry_at = end;
    }
    for (layer_donors, 0..) |donor, layer| if (donor >= donors.len or donors[donor].layer > layer)
        return error.InvalidTrainingAugmentationBinding;
    var metadata: engine.SourceMetadata = .{ .identity = identity, .logical_length = inventory.logical_length, .tokens = tokens, .logits = logits, .layer_donors = layer_donors, .donors = donors, .structure_sha256 = undefined, .state_sha256 = complete_state_sha256, .source_census_physical_bytes = inventory.physical_bytes, .source_global_storage_bytes = inventory.global_physical_bytes };
    metadata.structure_sha256 = engine.sourceMetadataStructureDigest(metadata);
    return .{ .allocator = allocator, .metadata = metadata, .inventory_sha256 = inventory.digest };
}

test "installed inventory builds global and local structural authority" {
    var entries = [_]metal_storage.InventoryEntry{
        .{ .layer = 0, .head = 0, .window = 0, .dim = 2, .rows = 8, .capacity = 8, .start = 0, .encoding = .exact, .physical_bytes = 4096 },
        .{ .layer = 0, .head = 1, .window = 0, .dim = 2, .rows = 8, .capacity = 8, .start = 0, .encoding = .exact, .physical_bytes = 4096 },
        .{ .layer = 1, .head = 0, .window = 4, .dim = 2, .rows = 4, .capacity = 4, .start = 1, .encoding = .exact, .physical_bytes = 4096 },
    };
    const inventory: metal_storage.InventoryOwned = .{ .allocator = std.testing.allocator, .logical_length = 8, .entries = &entries, .physical_bytes = 12288, .global_physical_bytes = 8192, .digest = @splat(7) };
    var owned = try build(std.testing.allocator, .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, &.{ 0, 0, 0, 0, 0, 0, 0, 0 }, &.{0}, &.{ 0, 1 }, inventory, inventory.digest, @splat(9));
    defer owned.deinit();
    try std.testing.expectEqual(@as(usize, 2), owned.metadata.donors.len);
    try std.testing.expectEqual(@as(u32, 4), owned.metadata.donors[1].window);
    try std.testing.expectEqual(@as(usize, 12288), owned.metadata.source_census_physical_bytes);
    entries[1].rows = 7;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, build(std.testing.allocator, owned.metadata.identity, owned.metadata.tokens, owned.metadata.logits, owned.metadata.layer_donors, inventory, inventory.digest, @splat(9)));
}
