const std = @import("std");
const engine = @import("attention_match_engine.zig");
const recent = @import("attention_match_prefill_recent64_capture.zig");

pub const schema = "antfly.recent64-raw-train-fixture.v1";
pub const values_name = "values.f32";
pub const positions_name = "positions.u64";
pub const manifest_name = "manifest.json";
pub const serialization_scratch_bytes: usize = 8 * 1024 * 1024;
pub const value_count = recent.batch_count * recent.horizon * recent.head_dim;
pub const position_count = recent.batch_count * recent.horizon;
pub const canonical_consumers = [_]u32{ 5, 11, 17, 23, 29, 35, 41 };
pub const canonical_donors = [_]u32{ 5, 11, 17, 23, 23, 23, 23 };

pub const FileRef = struct { name: []const u8, bytes: usize, sha256: [32]u8 };
pub const Batch = struct { donor_layer: u32, consumer_layer: u32, query_head: u32, value_offset: usize, position_offset: usize };
pub const Manifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    qualification_credit: bool,
    serving_activation: bool,
    source_checkpoint_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    execution_sha256: [32]u8,
    profile_sha256: [32]u8,
    development_policy_sha256: [32]u8,
    identity: recent.Identity,
    training_authority_sha256: [32]u8,
    portfolio_sha256: [32]u8,
    logical_tokens: usize,
    horizon: usize,
    head_dim: usize,
    consumer_count: usize,
    query_heads: usize,
    batch_count: usize,
    endianness: []const u8,
    values: FileRef,
    positions: FileRef,
    batches: [recent.batch_count]Batch,
};

pub const Provenance = struct {
    source_checkpoint_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    execution_sha256: [32]u8,
    profile_sha256: [32]u8,
    development_policy_sha256: [32]u8,
    identity: recent.Identity,
};

fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

pub fn buildManifest(training: recent.Training, topology: [recent.consumer_count]recent.Consumer, provenance: Provenance) !Manifest {
    if (training.batches.len != recent.batch_count or training.values.len != value_count or training.positions.len != position_count)
        return error.InvalidRecent64CapturedFixture;
    var batches: [recent.batch_count]Batch = undefined;
    for (training.batches, 0..) |batch, index| {
        const consumer_index = index / recent.query_heads;
        if (batch.consumer_layer != topology[consumer_index].consumer_layer or batch.query_head != index % recent.query_heads or
            batch.train.len != recent.horizon * recent.head_dim or batch.train_positions.len != recent.horizon or
            batch.heldout.len != 0 or batch.heldout_positions.len != 0)
            return error.InvalidRecent64CapturedFixture;
        for (batch.train_positions, 0..) |position, row| if (position != 8064 + row) return error.InvalidRecent64CapturedFixture;
        batches[index] = .{
            .donor_layer = @intCast(topology[consumer_index].donor_layer),
            .consumer_layer = batch.consumer_layer,
            .query_head = batch.query_head,
            .value_offset = index * recent.horizon * recent.head_dim,
            .position_offset = index * recent.horizon,
        };
    }
    const manifest: Manifest = .{
        .schema = schema,
        .diagnostic_only = true,
        .qualification_credit = false,
        .serving_activation = false,
        .source_checkpoint_sha256 = provenance.source_checkpoint_sha256,
        .source_freeze_sha256 = provenance.source_freeze_sha256,
        .binary_sha256 = provenance.binary_sha256,
        .implementation_semantic_sha256 = provenance.implementation_semantic_sha256,
        .execution_sha256 = provenance.execution_sha256,
        .profile_sha256 = provenance.profile_sha256,
        .development_policy_sha256 = provenance.development_policy_sha256,
        .identity = provenance.identity,
        .training_authority_sha256 = training.authority_sha256,
        .portfolio_sha256 = training.portfolio_sha256,
        .logical_tokens = 8128,
        .horizon = recent.horizon,
        .head_dim = recent.head_dim,
        .consumer_count = recent.consumer_count,
        .query_heads = recent.query_heads,
        .batch_count = recent.batch_count,
        .endianness = "little",
        .values = .{ .name = values_name, .bytes = training.values.len * @sizeOf(f32), .sha256 = digest(std.mem.sliceAsBytes(training.values)) },
        .positions = .{ .name = positions_name, .bytes = training.positions.len * @sizeOf(u64), .sha256 = digest(std.mem.sliceAsBytes(training.positions)) },
        .batches = batches,
    };
    try validateManifest(manifest);
    return manifest;
}

pub fn validateManifest(manifest: Manifest) !void {
    if (!std.mem.eql(u8, manifest.schema, schema) or !manifest.diagnostic_only or manifest.qualification_credit or manifest.serving_activation or
        manifest.logical_tokens != 8128 or manifest.horizon != recent.horizon or manifest.head_dim != recent.head_dim or
        manifest.consumer_count != recent.consumer_count or manifest.query_heads != recent.query_heads or manifest.batch_count != recent.batch_count or
        !std.mem.eql(u8, manifest.endianness, "little") or !std.mem.eql(u8, manifest.values.name, values_name) or
        !std.mem.eql(u8, manifest.positions.name, positions_name) or manifest.values.bytes != value_count * @sizeOf(f32) or
        manifest.positions.bytes != position_count * @sizeOf(u64)) return error.InvalidRecent64CapturedFixture;
    inline for (.{ manifest.source_checkpoint_sha256, manifest.source_freeze_sha256, manifest.binary_sha256, manifest.implementation_semantic_sha256, manifest.execution_sha256, manifest.profile_sha256, manifest.development_policy_sha256, manifest.identity.weights, manifest.identity.tokenizer, manifest.identity.template, manifest.identity.layout, manifest.training_authority_sha256, manifest.portfolio_sha256 }) |hash| {
        if (std.mem.allEqual(u8, &hash, 0)) return error.InvalidRecent64CapturedFixture;
    }
    for (manifest.batches, 0..) |batch, index| {
        const consumer_index = index / recent.query_heads;
        if (batch.query_head != index % recent.query_heads or batch.value_offset != index * recent.horizon * recent.head_dim or
            batch.position_offset != index * recent.horizon or batch.consumer_layer != canonical_consumers[consumer_index] or
            batch.donor_layer != canonical_donors[consumer_index] or (index > 0 and index % recent.query_heads != 0 and
            (batch.donor_layer != manifest.batches[index - 1].donor_layer or batch.consumer_layer != manifest.batches[index - 1].consumer_layer)))
            return error.InvalidRecent64CapturedFixture;
    }
}

pub fn reconstruct(allocator: std.mem.Allocator, manifest: Manifest, value_bytes: []const u8, position_bytes: []const u8) !recent.Training {
    try validateManifest(manifest);
    if (value_bytes.len != manifest.values.bytes or position_bytes.len != manifest.positions.bytes or
        !std.mem.eql(u8, &digest(value_bytes), &manifest.values.sha256) or !std.mem.eql(u8, &digest(position_bytes), &manifest.positions.sha256))
        return error.Recent64CapturedFixtureChanged;
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    @memcpy(std.mem.sliceAsBytes(values), value_bytes);
    @memcpy(std.mem.sliceAsBytes(positions), position_bytes);
    const batches = try allocator.alloc(engine.ReferenceBatch, recent.batch_count);
    errdefer allocator.free(batches);
    for (manifest.batches, batches) |batch, *out| {
        const train = values[batch.value_offset..][0 .. recent.horizon * recent.head_dim];
        const train_positions = positions[batch.position_offset..][0..recent.horizon];
        for (train_positions, 0..) |position, row| if (position != 8064 + row) return error.InvalidRecent64CapturedFixture;
        out.* = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = train, .train_positions = train_positions, .heldout = &.{}, .heldout_positions = &.{} };
    }
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .authority_sha256 = manifest.training_authority_sha256, .portfolio_sha256 = manifest.portfolio_sha256 };
}

test "captured fixture rejects topology position and payload mutations" {
    const one: [32]u8 = @splat(1);
    var batches: [recent.batch_count]Batch = undefined;
    for (&batches, 0..) |*batch, index| batch.* = .{ .donor_layer = canonical_donors[index / recent.query_heads], .consumer_layer = canonical_consumers[index / recent.query_heads], .query_head = @intCast(index % recent.query_heads), .value_offset = index * recent.horizon * recent.head_dim, .position_offset = index * recent.horizon };
    var manifest: Manifest = .{
        .schema = schema,
        .diagnostic_only = true,
        .qualification_credit = false,
        .serving_activation = false,
        .source_checkpoint_sha256 = one,
        .source_freeze_sha256 = one,
        .binary_sha256 = one,
        .implementation_semantic_sha256 = one,
        .execution_sha256 = one,
        .profile_sha256 = one,
        .development_policy_sha256 = one,
        .identity = .{ .weights = one, .tokenizer = one, .template = one, .layout = one },
        .training_authority_sha256 = one,
        .portfolio_sha256 = one,
        .logical_tokens = 8128,
        .horizon = recent.horizon,
        .head_dim = recent.head_dim,
        .consumer_count = recent.consumer_count,
        .query_heads = recent.query_heads,
        .batch_count = recent.batch_count,
        .endianness = "little",
        .values = .{ .name = values_name, .bytes = value_count * @sizeOf(f32), .sha256 = one },
        .positions = .{ .name = positions_name, .bytes = position_count * @sizeOf(u64), .sha256 = one },
        .batches = batches,
    };
    try validateManifest(manifest);
    manifest.batches[9].query_head = 7;
    try std.testing.expectError(error.InvalidRecent64CapturedFixture, validateManifest(manifest));
    manifest.batches[9].query_head = 1;
    manifest.positions.bytes -= 1;
    try std.testing.expectError(error.InvalidRecent64CapturedFixture, validateManifest(manifest));
}
