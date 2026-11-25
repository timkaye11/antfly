const std = @import("std");
const solver = @import("attention_match.zig");
const engine = @import("attention_match_engine.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const merge_mod = @import("attention_match_native_coverage_training_merge.zig");
const rewrite_mod = @import("attention_match_e2_production768_train_challenge_rewrite.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const flat_shape = @import("attention_match_e2_flat_shape_authority.zig");
const shared_coverage = @import("attention_match_e2_production768_historical_coverage.zig");
const probability_accelerate = @import("attention_match_accelerate_fp64_probability_scan.zig");
const e2_flat8128_pressure = @import("attention_match_e2_flat8128_pressure_profile.zig");
const e2_pressure_derivation = @import("attention_match_e2_historical_pressure_derivation.zig");
const formal_inventory = @import("attention_match_e2_formal_owner_inventory.zig");

pub const policy = "e2_native_resident_hybrid_bridge_v3|consumer_batches56|per_owner8,8,40|heads3|adaptive2|exact1|primary60|fallback0|max_adaptive_rows3961|shared_production768_math|chronological_capture_tiles64|retained_exact_f32_cache|cpu_fp64_validation|activation=disabled";
pub const implementation_semantic_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy ++ "|" ++ shared_coverage.policy ++ "|" ++ shared_coverage.accelerated_policy ++ "|" ++ probability_accelerate.policy ++ "|" ++ e2_flat8128_pressure.policy ++ "|" ++ rewrite_mod.policy, &out, .{});
    break :blk out;
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}

fn validateE2ConsumerTuples(batches: []const engine.ReferenceBatch) !void {
    if (batches.len != 56) return error.NativeCoverageShapeUnsupported;
    for (batches, 0..) |batch, index| {
        if (batch.consumer_layer != @as(u32, @intCast(4 + (index / 8) * 5)) or batch.query_head != @as(u32, @intCast(index % 8)))
            return error.NativeCoverageShapeUnsupported;
    }
}

test "E2 bridge admits all 56 ordered consumer tuples and rejects aggregate-40 geometry" {
    var batches: [56]engine.ReferenceBatch = undefined;
    for (&batches, 0..) |*batch, index| batch.* = .{
        .consumer_layer = @intCast(4 + (index / 8) * 5),
        .query_head = @intCast(index % 8),
        .train = &.{},
        .heldout = &.{},
        .train_positions = &.{},
        .heldout_positions = &.{},
    };
    try validateE2ConsumerTuples(&batches);
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, validateE2ConsumerTuples(batches[0..40]));
    batches[55].consumer_layer = 29;
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, validateE2ConsumerTuples(&batches));
}
fn hashU64(hash: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hash.update(&bytes);
}
fn hashU32(hash: *std.crypto.hash.sha2.Sha256, value: u32) void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    hash.update(&bytes);
}

fn validateCanonicalViews(source: augmentation.OwnedTraining) !void {
    if (source.batches.len == 0) return error.InvalidTrainingAugmentationBinding;
    var value_at: usize = 0;
    var position_at: usize = 0;
    for (source.batches) |batch| {
        if (batch.train.len == 0 or batch.train_positions.len == 0 or batch.train.len % batch.train_positions.len != 0 or batch.heldout.len != 0 or batch.heldout_positions.len != 0) return error.InvalidTrainingAugmentationBinding;
        const value_end = add(value_at, batch.train.len) catch return error.InvalidTrainingAugmentationBinding;
        const position_end = add(position_at, batch.train_positions.len) catch return error.InvalidTrainingAugmentationBinding;
        if (value_end > source.values.len or position_end > source.positions.len or batch.train.ptr != source.values[value_at..].ptr or batch.train_positions.ptr != source.positions[position_at..].ptr) return error.InvalidTrainingAugmentationBinding;
        value_at = value_end;
        position_at = position_end;
    }
    if (value_at != source.values.len or position_at != source.positions.len) return error.InvalidTrainingAugmentationBinding;
}

fn denseDigest(source: augmentation.OwnedTraining) ![32]u8 {
    try validateCanonicalViews(source);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-resident-dense-owner.v2\x00");
    hashU64(&hash, @intCast(source.batches.len));
    hashU64(&hash, @intCast(source.values.len));
    for (source.values) |value| hashU32(&hash, @bitCast(value));
    hashU64(&hash, @intCast(source.positions.len));
    for (source.positions) |position| hashU64(&hash, position);
    for (source.batches, 0..) |batch, i| {
        hashU64(&hash, @intCast(i));
        hashU32(&hash, batch.consumer_layer);
        hashU32(&hash, batch.query_head);
        hashU64(&hash, @intCast(batch.train_positions.len));
        hashU64(&hash, @intCast(batch.train.len / batch.train_positions.len));
    }
    return hash.finalResult();
}

pub fn historicalBatchDigest(batches: []const rewrite_mod.TrainingBatchView) ![32]u8 {
    if (batches.len == 0) return error.InvalidTrainingAugmentationBinding;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.production768-derived-historical-domain.v1\x00");
    for (batches) |batch| {
        if (batch.train.len == 0 or batch.train_positions.len == 0 or batch.train.len % batch.train_positions.len != 0) return error.InvalidTrainingAugmentationBinding;
        for (batch.train) |value| hashU32(&hash, @bitCast(value));
    }
    for (batches) |batch| for (batch.train_positions) |position| hashU64(&hash, position);
    return hash.finalResult();
}

/// Cross-binds the Source918-compatible byte-domain digest to the ordered
/// consumer tuples without changing the Source918 parity field's framing.
pub fn historicalTupleAuthority(batches: []const rewrite_mod.TrainingBatchView, source918_digest: [32]u8) ![32]u8 {
    if (batches.len == 0 or !nonzero(source918_digest)) return error.InvalidTrainingAugmentationBinding;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.native-resident-historical-tuple-authority.v1\x00");
    hash.update(&source918_digest);
    hashU64(&hash, @intCast(batches.len));
    for (batches, 0..) |batch, batch_index| {
        if (batch.train_positions.len == 0 or batch.train.len % batch.train_positions.len != 0) return error.InvalidTrainingAugmentationBinding;
        hashU64(&hash, @intCast(batch_index));
        hashU32(&hash, batch.consumer_layer);
        hashU32(&hash, batch.query_head);
        hashU64(&hash, @intCast(batch.train_positions.len));
        hashU64(&hash, @intCast(batch.train.len / batch.train_positions.len));
    }
    return hash.finalResult();
}

pub const DenseOwner = struct {
    source: *const augmentation.OwnedTraining,
    payload_sha256: [32]u8,
    maximum_clone_bytes: usize,
    source_length: usize,
    repeat_length: usize,

    pub fn init(source: *const augmentation.OwnedTraining, source_length: usize, repeat_length: usize) !@This() {
        if (source_length == 0 or repeat_length == 0) return error.InvalidTrainingAugmentationBinding;
        const digest = try denseDigest(source.*);
        var bytes = try add(try mul(source.values.len, @sizeOf(f32)), try mul(source.positions.len, @sizeOf(u64)));
        bytes = try add(bytes, try mul(source.batches.len, @sizeOf(engine.ReferenceBatch) + 4 * @sizeOf(u32)));
        bytes = try add(bytes, digest.len);
        return .{ .source = source, .payload_sha256 = digest, .maximum_clone_bytes = bytes, .source_length = source_length, .repeat_length = repeat_length };
    }

    pub fn load(raw: ?*anyopaque, allocator: std.mem.Allocator, control: solver.Control) !augmentation.Owned {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        try control.check();
        const actual = try denseDigest(self.source.*);
        if (!std.mem.eql(u8, &actual, &self.payload_sha256)) return error.InvalidTrainingAugmentationBinding;
        const manifest = try allocator.dupe(u8, &self.payload_sha256);
        errdefer allocator.free(manifest);
        const training = try allocator.dupe(f32, self.source.values);
        errdefer allocator.free(training);
        try control.check();
        const positions = try allocator.dupe(u64, self.source.positions);
        errdefer allocator.free(positions);
        const batches = try allocator.alloc(engine.ReferenceBatch, self.source.batches.len);
        errdefer allocator.free(batches);
        const layers = try allocator.alloc(u32, batches.len);
        errdefer allocator.free(layers);
        const heads = try allocator.alloc(u32, batches.len);
        errdefer allocator.free(heads);
        const counts = try allocator.alloc(u32, batches.len);
        errdefer allocator.free(counts);
        const dimensions = try allocator.alloc(u32, batches.len);
        errdefer allocator.free(dimensions);
        var value_at: usize = 0;
        var position_at: usize = 0;
        for (self.source.batches, batches, layers, heads, counts, dimensions) |source_batch, *out, *layer, *head, *count, *dim| {
            try control.check();
            const batch_dim = source_batch.train.len / source_batch.train_positions.len;
            const value_end = try add(value_at, source_batch.train.len);
            const position_end = try add(position_at, source_batch.train_positions.len);
            layer.* = source_batch.consumer_layer;
            head.* = source_batch.query_head;
            count.* = @intCast(source_batch.train_positions.len);
            dim.* = @intCast(batch_dim);
            out.* = .{ .consumer_layer = layer.*, .query_head = head.*, .train = training[value_at..value_end], .heldout = &.{}, .train_positions = positions[position_at..position_end], .heldout_positions = &.{} };
            value_at = value_end;
            position_at = position_end;
        }
        return .{ .allocator = allocator, .manifest_bytes = manifest, .batches = batches, .training = training, .positions = positions, .consumer_layers = layers, .query_heads = heads, .counts = counts, .dimensions = dimensions, .source_length = self.source_length, .repeat_length = self.repeat_length };
    }
};

pub fn denseCloneBytes(source: *const augmentation.OwnedTraining, source_length: usize, repeat_length: usize) !usize {
    return (try DenseOwner.init(source, source_length, repeat_length)).maximum_clone_bytes;
}

pub const PrepareBinding = struct {
    expected_source_rows: usize = 8128,
    formal_owner_inventory_sha256: [32]u8 = @splat(0),
    formal_owner_inventory: ?formal_inventory.Artifact = null,
    /// Fresh inventory read from this request's live ordinary Metal source.
    /// Formal rows require it; diagnostic 8128 must leave it null.
    current_raw_inventory: ?formal_inventory.RawInventory = null,
    current_hardware_sha256: [32]u8 = @splat(0),
    source_length: usize,
    repeat_length: usize,
    source_freeze_sha256: [32]u8,
    config_authority_sha256: [32]u8,
    expected_shape_sha256: [32]u8,
    collector_required_bytes: usize,
    collector_manifest_sha256: [32]u8,
    external_live_bytes: usize,
    actual_native_device_bytes: usize,
    combined_cap: usize,
    diagnostic_query_batch_rows: u16 = 64,
};

pub const FullTrainingOwned = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    heldout_values: []f32,
    heldout_positions: []u64,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.heldout_positions);
        self.allocator.free(self.heldout_values);
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
    pub fn liveBytes(self: @This()) !usize {
        var bytes = try mul(self.batches.len, @sizeOf(engine.ReferenceBatch));
        bytes = try add(bytes, try mul(self.values.len + self.heldout_values.len, @sizeOf(f32)));
        return add(bytes, try mul(self.positions.len + self.heldout_positions.len, @sizeOf(u64)));
    }
};

pub const Ledger = struct {
    retained_host_source_bytes: usize,
    native_device_bytes: usize,
    dense_owner_bytes: usize,
    metadata_bytes: usize,
    plan_bytes: usize,
    merged_bytes: usize,
    reference_batch_bytes: usize,
    head_evidence_bytes: usize,
    view_bytes: usize,
    coverage_csr_bytes: usize,
    rewrite_bytes: usize,
    final_bytes: usize,
    peak_postfit_workspace_bytes: usize,
};

pub const EarlyRetainedPlan = struct {
    external_live_bytes: usize,
    dense_clone_bytes: usize,
    metadata_bytes: usize,
    required_bytes: usize,

    pub fn admit(self: @This(), cap: usize) !void {
        if (self.required_bytes > cap) return error.CompactionWorkspaceLimit;
    }
};

fn earlyRetainedPlan(external_live_bytes: usize, dense_clone_bytes: usize, batch_count: usize, position_count: usize) !EarlyRetainedPlan {
    if (dense_clone_bytes == 0 or batch_count != 56 or position_count != 56 * 512)
        return error.NativeCoverageShapeUnsupported;
    const metadata_arrays = try mul(try mul(batch_count, 4), @sizeOf(u32));
    const metadata_positions = try mul(position_count, @sizeOf(u64));
    const metadata_bytes = try add(metadata_arrays, metadata_positions);
    return .{ .external_live_bytes = external_live_bytes, .dense_clone_bytes = dense_clone_bytes, .metadata_bytes = metadata_bytes, .required_bytes = try add(external_live_bytes, try add(dense_clone_bytes, metadata_bytes)) };
}

fn mergedLiveBytes(merged: merge_mod.Owned) !usize {
    var bytes = try mul(merged.batches.len, @sizeOf(engine.ReferenceBatch));
    bytes = try add(bytes, try mul(merged.values.len + merged.heldout_values.len, @sizeOf(f32)));
    bytes = try add(bytes, try mul(merged.positions.len + merged.heldout_positions.len, @sizeOf(u64)));
    bytes = try add(bytes, try mul(merged.coverage_offsets.len + merged.coverage_indices.len, @sizeOf(u32)));
    bytes = try add(bytes, try mul(merged.heads.len, @sizeOf(merge_mod.HeadEvidence)));
    for (merged.heads) |head| bytes = try add(bytes, try mul(head.maximum_probability.len, @sizeOf(f64)));
    return bytes;
}

fn rewriteLiveBytes(rewritten: rewrite_mod.Owned) !usize {
    var bytes = try mul(rewritten.batches.len, @sizeOf(engine.ReferenceBatch));
    bytes = try add(bytes, try mul(rewritten.values.len, @sizeOf(f32)));
    return add(bytes, try mul(rewritten.positions.len, @sizeOf(u64)));
}

fn assembleInitialMix(
    allocator: std.mem.Allocator,
    primary: []const engine.ReferenceBatch,
    historical: shared_coverage.Owned,
    future: []const engine.ReferenceBatch,
    combined_cap: usize,
    external: usize,
) !merge_mod.Owned {
    if (primary.len == 0 or primary.len != historical.batches.len or primary.len != future.len) return error.NativeCoverageShapeUnsupported;
    var value_count: usize = 0;
    var position_count: usize = 0;
    var heldout_value_count: usize = 0;
    var heldout_position_count: usize = 0;
    for (primary, historical.batches, future) |p, h, f| {
        if (p.consumer_layer != h.consumer_layer or p.consumer_layer != f.consumer_layer or p.query_head != h.query_head or p.query_head != f.query_head or
            p.train_positions.len != 256 or h.train_positions.len != 256 or f.train_positions.len != 256 or p.heldout_positions.len != 64)
            return error.NativeCoverageShapeUnsupported;
        const dim = p.train.len / 256;
        if (dim != 512 or h.train.len != 256 * dim or f.train.len != 256 * dim or p.heldout.len != 64 * dim) return error.NativeCoverageShapeUnsupported;
        value_count = try add(value_count, try add(p.train.len, try add(h.train.len, f.train.len)));
        position_count = try add(position_count, 768);
        heldout_value_count = try add(heldout_value_count, p.heldout.len);
        heldout_position_count = try add(heldout_position_count, 64);
    }
    var planned = try mul(primary.len, @sizeOf(engine.ReferenceBatch));
    planned = try add(planned, try mul(try add(value_count, heldout_value_count), @sizeOf(f32)));
    planned = try add(planned, try mul(try add(position_count, heldout_position_count), @sizeOf(u64)));
    planned = try add(planned, try mul(historical.coverage_offsets.len + historical.coverage_indices.len, @sizeOf(u32)));
    if (external > combined_cap or planned > combined_cap - external) return error.CompactionWorkspaceLimit;
    const batches = try allocator.alloc(engine.ReferenceBatch, primary.len);
    errdefer allocator.free(batches);
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    const heldout_values = try allocator.alloc(f32, heldout_value_count);
    errdefer allocator.free(heldout_values);
    const heldout_positions = try allocator.alloc(u64, heldout_position_count);
    errdefer allocator.free(heldout_positions);
    const coverage_offsets = try allocator.dupe(u32, historical.coverage_offsets);
    errdefer allocator.free(coverage_offsets);
    const coverage_indices = try allocator.dupe(u32, historical.coverage_indices);
    errdefer allocator.free(coverage_indices);
    const heads = try allocator.alloc(merge_mod.HeadEvidence, 0);
    errdefer allocator.free(heads);
    var value_at: usize = 0;
    var position_at: usize = 0;
    var heldout_value_at: usize = 0;
    var heldout_position_at: usize = 0;
    for (primary, historical.batches, future, batches) |p, h, f, *out| {
        const dim = p.train.len / 256;
        const value_end = try add(value_at, try mul(768, dim));
        @memcpy(values[value_at..][0 .. 256 * dim], p.train);
        @memcpy(values[value_at + 256 * dim ..][0 .. 256 * dim], h.train);
        @memcpy(values[value_at + 512 * dim ..][0 .. 256 * dim], f.train);
        const position_end = try add(position_at, 768);
        @memcpy(positions[position_at..][0..256], p.train_positions);
        @memcpy(positions[position_at + 256 ..][0..256], h.train_positions);
        @memcpy(positions[position_at + 512 ..][0..256], f.train_positions);
        const heldout_value_end = try add(heldout_value_at, p.heldout.len);
        @memcpy(heldout_values[heldout_value_at..heldout_value_end], p.heldout);
        const heldout_position_end = try add(heldout_position_at, 64);
        @memcpy(heldout_positions[heldout_position_at..heldout_position_end], p.heldout_positions);
        out.* = .{ .consumer_layer = p.consumer_layer, .query_head = p.query_head, .train = values[value_at..value_end], .heldout = heldout_values[heldout_value_at..heldout_value_end], .train_positions = positions[position_at..position_end], .heldout_positions = heldout_positions[heldout_position_at..heldout_position_end], .independent_train_split = 256, .independent_train_second_split = 512 };
        value_at = value_end;
        position_at = position_end;
        heldout_value_at = heldout_value_end;
        heldout_position_at = heldout_position_end;
    }
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .heldout_values = heldout_values, .heldout_positions = heldout_positions, .coverage_offsets = coverage_offsets, .coverage_indices = coverage_indices, .coverage_authority_sha256 = historical.evidence.coverage_authority_sha256, .heads = heads, .digest = historical.evidence.selection_digest };
}

fn mapsToHead(state: checkpoint.State, layout: engine.Layout, batch: engine.ReferenceBatch, donor_index: usize, head_index: usize) !bool {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len or state.layer_donors[batch.consumer_layer] != donor_index or donor_index >= state.donors.len) return false;
    const head_count = state.donors[donor_index].heads.len;
    const query_heads = layout.query_heads_by_layer[batch.consumer_layer];
    if (head_count == 0 or query_heads % head_count != 0) return error.NativeCoverageShapeUnsupported;
    const group = query_heads / head_count;
    return group != 0 and batch.query_head / group == head_index;
}

pub const Result = struct {
    final_training: FullTrainingOwned,
    planned_exact_donor_layer: u32,
    planned_exact_head_index: usize,
    initial_historical_sha256: [32]u8,
    historical_tuple_authority_sha256: [32]u8,
    coverage_authority_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    config_authority_sha256: [32]u8,
    collector_required_bytes: usize,
    collector_manifest_sha256: [32]u8,
    merge_digest: [32]u8,
    rewrite_evidence: rewrite_mod.Evidence,
    candidate_physical_bytes: usize,
    source_physical_bytes: usize,
    ledger: Ledger,
    pub fn deinit(self: *@This()) void {
        self.final_training.deinit();
        self.* = undefined;
    }
};

fn assembleFinal(allocator: std.mem.Allocator, merged: merge_mod.Owned, rewritten: rewrite_mod.Owned, combined_cap: usize, external: usize) !FullTrainingOwned {
    if (merged.batches.len != rewritten.batches.len) return error.NativeCoverageShapeUnsupported;
    var expected_values: usize = 0;
    var expected_positions: usize = 0;
    var expected_heldout_values: usize = 0;
    var expected_heldout_positions: usize = 0;
    for (merged.batches, rewritten.batches) |source, rewritten_batch| {
        if (source.consumer_layer != rewritten_batch.consumer_layer or source.query_head != rewritten_batch.query_head or
            source.train_positions.len != 768 or source.train.len == 0 or source.train.len % 768 != 0 or
            (source.independent_train_split orelse return error.NativeCoverageShapeUnsupported) != 256 or
            (source.independent_train_second_split orelse return error.NativeCoverageShapeUnsupported) != 512 or
            rewritten_batch.train_positions.len != 256 or source.heldout_positions.len != 64)
            return error.NativeCoverageShapeUnsupported;
        const dimension = source.train.len / 768;
        if (rewritten_batch.train.len != try mul(256, dimension) or source.heldout.len != try mul(64, dimension))
            return error.NativeCoverageShapeUnsupported;
        expected_values = try add(expected_values, source.train.len);
        expected_positions = try add(expected_positions, 768);
        expected_heldout_values = try add(expected_heldout_values, source.heldout.len);
        expected_heldout_positions = try add(expected_heldout_positions, 64);
    }
    if (expected_values != merged.values.len or expected_positions != merged.positions.len or
        expected_heldout_values != merged.heldout_values.len or expected_heldout_positions != merged.heldout_positions.len)
        return error.NativeCoverageShapeUnsupported;
    var planned_bytes = try mul(merged.batches.len, @sizeOf(engine.ReferenceBatch));
    planned_bytes = try add(planned_bytes, try mul(try add(expected_values, expected_heldout_values), @sizeOf(f32)));
    planned_bytes = try add(planned_bytes, try mul(try add(expected_positions, expected_heldout_positions), @sizeOf(u64)));
    if (external > combined_cap or planned_bytes > combined_cap - external) return error.CompactionWorkspaceLimit;
    const values = try allocator.alloc(f32, merged.values.len);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, merged.positions.len);
    errdefer allocator.free(positions);
    const heldout_values = try allocator.dupe(f32, merged.heldout_values);
    errdefer allocator.free(heldout_values);
    const heldout_positions = try allocator.dupe(u64, merged.heldout_positions);
    errdefer allocator.free(heldout_positions);
    const batches = try allocator.alloc(engine.ReferenceBatch, merged.batches.len);
    errdefer allocator.free(batches);
    var value_at: usize = 0;
    var position_at: usize = 0;
    var heldout_value_at: usize = 0;
    var heldout_position_at: usize = 0;
    for (merged.batches, rewritten.batches, batches) |source, replacement, *out| {
        if (source.train_positions.len != 768 or (source.independent_train_split orelse return error.NativeCoverageShapeUnsupported) != 256 or (source.independent_train_second_split orelse return error.NativeCoverageShapeUnsupported) != 512 or replacement.train_positions.len != 256 or source.heldout_positions.len != 64) return error.NativeCoverageShapeUnsupported;
        const dim = source.train.len / 768;
        if (dim == 0 or replacement.train.len != 256 * dim or source.heldout.len != 64 * dim) return error.NativeCoverageShapeUnsupported;
        const value_end = try add(value_at, source.train.len);
        const position_end = try add(position_at, 768);
        @memcpy(values[value_at..value_end], source.train);
        @memcpy(positions[position_at..position_end], source.train_positions);
        @memcpy(values[value_at + 256 * dim ..][0 .. 256 * dim], replacement.train);
        @memcpy(positions[position_at + 256 ..][0..256], replacement.train_positions);
        const heldout_value_end = try add(heldout_value_at, source.heldout.len);
        const heldout_position_end = try add(heldout_position_at, 64);
        out.* = .{ .consumer_layer = source.consumer_layer, .query_head = source.query_head, .train = values[value_at..value_end], .heldout = heldout_values[heldout_value_at..heldout_value_end], .train_positions = positions[position_at..position_end], .heldout_positions = heldout_positions[heldout_position_at..heldout_position_end], .independent_train_split = 256, .independent_train_second_split = 512 };
        value_at = value_end;
        position_at = position_end;
        heldout_value_at = heldout_value_end;
        heldout_position_at = heldout_position_end;
    }
    const out: FullTrainingOwned = .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .heldout_values = heldout_values, .heldout_positions = heldout_positions };
    std.debug.assert(try out.liveBytes() == planned_bytes);
    return out;
}

pub fn prepare(allocator: std.mem.Allocator, state: checkpoint.State, layout: engine.Layout, primary: []const engine.ReferenceBatch, dense_source: *const augmentation.OwnedTraining, future: []const engine.ReferenceBatch, fit: solver.Config, cpu_fp64_backend: solver.Backend, binding: PrepareBinding) !Result {
    if (primary.len != 56 or dense_source.batches.len != primary.len or future.len != primary.len or
        (binding.expected_source_rows != 7_936 and binding.expected_source_rows != 8_128 and binding.expected_source_rows != 32_512) or
        ((binding.expected_source_rows == 8_128) != std.mem.allEqual(u8, &binding.formal_owner_inventory_sha256, 0)) or
        ((binding.expected_source_rows == 8_128) != (binding.formal_owner_inventory == null)) or
        ((binding.expected_source_rows == 8_128) != (binding.current_raw_inventory == null)) or
        binding.combined_cap != fit.max_workspace_bytes or binding.combined_cap != 2 * 1024 * 1024 * 1024 or !nonzero(binding.source_freeze_sha256) or !nonzero(binding.config_authority_sha256) or !nonzero(binding.expected_shape_sha256) or !nonzero(binding.collector_manifest_sha256) or binding.collector_required_bytes == 0 or cpu_fp64_backend.summary_fn == null) return error.NativeCoverageShapeUnsupported;
    if (binding.expected_source_rows != 8_128) {
        const inventory = binding.formal_owner_inventory orelse return error.NativeCoverageShapeUnsupported;
        if (!std.mem.eql(u8, &inventory.context.hardware_sha256, &binding.current_hardware_sha256))
            return error.NativeCoverageShapeUnsupported;
        const final_configuration = formal_inventory.finalConfigurationDigest(inventory);
        if (!std.mem.eql(u8, &inventory.digest, &binding.formal_owner_inventory_sha256) or
            !std.mem.eql(u8, &final_configuration, &binding.config_authority_sha256)) return error.NativeCoverageShapeUnsupported;
        const conversion_receipt = formal_inventory.verifyState(inventory, state, binding.current_raw_inventory.?, .{
            .source_freeze_sha256 = binding.source_freeze_sha256,
            .base_configuration_sha256 = inventory.context.base_configuration_sha256,
            .shape_sha256 = binding.expected_shape_sha256,
            .binary_sha256 = inventory.context.binary_sha256,
            .capture_binary_sha256 = inventory.context.capture_binary_sha256,
            .hardware_sha256 = inventory.context.hardware_sha256,
            .owner_identity = inventory.context.owner_identity,
            .owner_logical_bytes = inventory.context.owner_logical_bytes,
            .owner_physical_bytes = inventory.context.owner_physical_bytes,
        }, binding.expected_source_rows) catch return error.NativeCoverageShapeUnsupported;
        if (std.mem.allEqual(u8, &conversion_receipt, 0)) return error.NativeCoverageShapeUnsupported;
    }
    try validateE2ConsumerTuples(primary);
    try validateE2ConsumerTuples(future);
    try validateE2ConsumerTuples(dense_source.batches);
    if (binding.diagnostic_query_batch_rows != 64 and binding.diagnostic_query_batch_rows != 256) return error.NativeCoverageShapeUnsupported;
    var owner = try DenseOwner.init(dense_source, binding.source_length, binding.repeat_length);
    const early_plan = try earlyRetainedPlan(binding.external_live_bytes, owner.maximum_clone_bytes, dense_source.batches.len, dense_source.positions.len);
    try early_plan.admit(binding.combined_cap);
    var dense = try DenseOwner.load(&owner, allocator, fit.control);
    var dense_live = true;
    defer if (dense_live) dense.deinit();
    var metadata = try rewrite_mod.freezeDenseMetadata(allocator, &dense, owner.maximum_clone_bytes);
    defer metadata.deinit();
    var peak_postfit: usize = 0;
    var planned_head_count: usize = 0;
    var canonical_seen: [flat_shape.canonical_heads.len]bool = @splat(false);
    var shape_source_rows: usize = 0;
    for (state.donors, 0..) |donor, donor_index| if (donor.window == 0) for (donor.heads, 0..) |head, head_index| {
        var canonical_index: ?usize = null;
        for (flat_shape.canonical_heads, 0..) |identity, i| {
            if (identity.donor_layer == donor.layer and identity.head_index == head_index) canonical_index = i;
        }
        const identity_index = canonical_index orelse return error.NativeCoverageShapeUnsupported;
        if (canonical_seen[identity_index] or donor.heads.len != 1 or head.dim != 512) return error.NativeCoverageShapeUnsupported;
        canonical_seen[identity_index] = true;
        const plan = (try engine.headPlan(state.logical_length, head, .{ .prefix_rows = 128, .recent_rows = 2048, .chunk_rows = 256, .retention_divisor = 2 })) orelse return error.NativeCoverageShapeUnsupported;
        planned_head_count += 1;
        if (shape_source_rows == 0) shape_source_rows = head.positions.len;
        var consumers: usize = 0;
        for (primary) |batch| {
            consumers += @intFromBool(try mapsToHead(state, layout, batch, donor_index, head_index));
        }
        const expected_consumers: usize = switch (donor.layer) {
            4, 9 => 8,
            14 => 40,
            else => return error.NativeCoverageShapeUnsupported,
        };
        if (consumers != expected_consumers) return error.NativeCoverageShapeUnsupported;
        const ranges = std.math.divCeil(usize, plan.old_rows - plan.prefix_rows, 256) catch return error.InvalidFitConfig;
        peak_postfit = @max(peak_postfit, try replay.postfitWorkspaceReserve(768 * consumers, head.positions.len, head.positions.len, ranges));
    };
    if (planned_head_count != 3 or shape_source_rows != binding.expected_source_rows or !std.mem.allEqual(bool, &canonical_seen, true)) return error.NativeCoverageShapeUnsupported;
    var retained = try add(binding.external_live_bytes, try add(owner.maximum_clone_bytes, try metadata.liveBytes()));
    if (retained >= binding.combined_cap) return error.CompactionWorkspaceLimit;
    const shape: flat_shape.Shape = .{ .source_rows = binding.expected_source_rows, .authority_sha256 = binding.source_freeze_sha256 };
    const validated_shape = try flat_shape.validate(shape);
    if (!std.crypto.timing_safe.eql([32]u8, validated_shape.digest, binding.expected_shape_sha256)) return error.InvalidTrainingAugmentationBinding;
    var probability_context: probability_accelerate.Context = .{};
    var historical = try shared_coverage.deriveWithProbabilityBackend(allocator, state, layout, primary, future, &dense, binding.combined_cap - retained, fit.control, .{ .backend = probability_accelerate.backend(&probability_context), .gather_eligible_historical = true, .e2_shape_authority_sha256 = validated_shape.digest, .e2_source_physical_bytes = if (binding.formal_owner_inventory) |inventory| @intCast(inventory.raw_total_physical_bytes) else 0 });
    var historical_live = true;
    defer if (historical_live) historical.deinit();
    const historical_live_bytes = try historical.liveBytes();
    const coverage_evidence = historical.evidence;
    var merged = try assembleInitialMix(allocator, primary, historical, future, binding.combined_cap, try add(retained, historical_live_bytes));
    defer merged.deinit();
    historical.deinit();
    historical_live = false;
    dense.deinit();
    dense_live = false;
    const merged_bytes = try mergedLiveBytes(merged);
    retained = try add(retained, merged_bytes);
    const view_count = try mul(primary.len, 3);
    const view_bytes = try mul(view_count, @sizeOf(rewrite_mod.TrainingBatchView));
    if (retained > binding.combined_cap or view_bytes > binding.combined_cap - retained) return error.CompactionWorkspaceLimit;
    const views = try allocator.alloc(rewrite_mod.TrainingBatchView, view_count);
    defer allocator.free(views);
    retained = try add(retained, view_bytes);
    const p_views = views[0..primary.len];
    const h_views = views[primary.len .. 2 * primary.len];
    const f_views = views[2 * primary.len ..];
    for (merged.batches, p_views, h_views, f_views) |batch, *p, *h, *f| {
        const dim = batch.train.len / batch.train_positions.len;
        if ((batch.independent_train_split orelse return error.NativeCoverageShapeUnsupported) != 256 or (batch.independent_train_second_split orelse return error.NativeCoverageShapeUnsupported) != 512 or batch.train_positions.len != 768 or dim != 512) return error.NativeCoverageShapeUnsupported;
        p.* = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[0 .. 256 * dim], .train_positions = batch.train_positions[0..256], .heldout_positions = batch.heldout_positions };
        h.* = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[256 * dim .. 512 * dim], .train_positions = batch.train_positions[256..512], .heldout_positions = &.{} };
        f.* = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .train = batch.train[512 * dim ..], .train_positions = batch.train_positions[512..], .heldout_positions = &.{} };
    }
    const initial_historical_sha256 = try historicalBatchDigest(h_views);
    if (!std.crypto.timing_safe.eql([32]u8, initial_historical_sha256, coverage_evidence.derived_historical_sha256)) return error.InvalidTrainingAugmentationBinding;
    const historical_tuple_authority_sha256 = try historicalTupleAuthority(h_views, initial_historical_sha256);
    const exact_layer = coverage_evidence.chosen_exact_donor_layer;
    const exact_head = coverage_evidence.chosen_exact_head_index;
    const dense_loader: rewrite_mod.DenseLoader = .{ .context = &owner, .load_fn = DenseOwner.load, .maximum_live_bytes = owner.maximum_clone_bytes };
    var resident_fit = fit;
    resident_fit.workspace_admission = .resident_native_hybrid_2g_v1;
    var rewritten = try rewrite_mod.rewrite(allocator, state, layout, p_views, h_views, f_views, metadata, dense_loader, null, .{ .fit = resident_fit, .backend = cpu_fp64_backend, .combined_cap = binding.combined_cap, .external_live_bytes = retained, .postfit_workspace_bytes = peak_postfit, .source_freeze_sha256 = binding.source_freeze_sha256, .dense_manifest_sha256 = metadata.manifest_sha256, .initial_historical_sha256 = initial_historical_sha256, .skip_donor_layer = exact_layer, .skip_head_index = exact_head, .hybrid_coverage_anchor = true, .coverage_offsets = merged.coverage_offsets, .coverage_indices = merged.coverage_indices, .coverage_authority_sha256 = merged.coverage_authority_sha256, .shape = shape, .expected_shape_digest = binding.expected_shape_sha256, .private_performance_policy = if (binding.diagnostic_query_batch_rows == 256) .diagnostic_capture_gram_batch256_v1 else .owned_capture_summary_incremental_reduction_critical025_v1, .expected_query_batch_rows = binding.diagnostic_query_batch_rows, .maximum_adaptive_candidate_rows = if (binding.expected_source_rows == 8_128) 3_961 else binding.expected_source_rows });
    defer rewritten.deinit();
    if (rewritten.evidence.pilot_head_count != 2 or rewritten.evidence.skipped_exact_head_count != 1 or rewritten.evidence.rewritten_batch_count + rewritten.evidence.skipped_exact_batch_count != 56) return error.NativeCoverageShapeUnsupported;
    var exact_physical_bytes = e2_pressure_derivation.exact_head_bytes;
    if (binding.formal_owner_inventory) |inventory| {
        var found = false;
        for (inventory.raw_entries) |entry| if (entry.layer == exact_layer) {
            if (found) return error.PhysicalBudgetUnsupported;
            exact_physical_bytes = @intCast(entry.installed_physical_bytes);
            found = true;
        };
        if (!found) return error.PhysicalBudgetUnsupported;
    }
    const census: e2_pressure_derivation.Census = .{ .adaptive_heads = rewritten.evidence.pilot_head_count, .exact_heads = rewritten.evidence.skipped_exact_head_count, .fallback_heads = 0, .adaptive_rows = rewritten.evidence.adaptive_candidate_rows, .adaptive_physical_bytes = rewritten.evidence.adaptive_candidate_physical_bytes, .exact_physical_bytes = exact_physical_bytes };
    if (binding.expected_source_rows == 8_128)
        try e2_pressure_derivation.validateCensus(census)
    else
        try e2_pressure_derivation.validateFormalCensus(census, binding.expected_source_rows, @intCast(binding.formal_owner_inventory.?.raw_total_physical_bytes));
    const candidate_physical_bytes = try add(census.exact_physical_bytes, try add(rewritten.evidence.adaptive_candidate_physical_bytes[0], rewritten.evidence.adaptive_candidate_physical_bytes[1]));
    const rewrite_bytes = try rewriteLiveBytes(rewritten);
    const final_external = try add(retained, rewrite_bytes);
    var final = try assembleFinal(allocator, merged, rewritten, binding.combined_cap, final_external);
    errdefer final.deinit();
    const head_evidence_bytes = @sizeOf(@TypeOf(coverage_evidence.exact_head_pressure));
    return .{ .final_training = final, .planned_exact_donor_layer = exact_layer, .planned_exact_head_index = exact_head, .initial_historical_sha256 = initial_historical_sha256, .historical_tuple_authority_sha256 = historical_tuple_authority_sha256, .coverage_authority_sha256 = merged.coverage_authority_sha256, .source_freeze_sha256 = binding.source_freeze_sha256, .config_authority_sha256 = binding.config_authority_sha256, .collector_required_bytes = binding.collector_required_bytes, .collector_manifest_sha256 = binding.collector_manifest_sha256, .merge_digest = merged.digest, .rewrite_evidence = rewritten.evidence, .candidate_physical_bytes = candidate_physical_bytes, .source_physical_bytes = if (binding.formal_owner_inventory) |inventory| @intCast(inventory.raw_total_physical_bytes) else e2_pressure_derivation.source_physical_bytes, .ledger = .{ .retained_host_source_bytes = binding.external_live_bytes, .native_device_bytes = binding.actual_native_device_bytes, .dense_owner_bytes = owner.maximum_clone_bytes, .metadata_bytes = try metadata.liveBytes(), .plan_bytes = 0, .merged_bytes = merged_bytes, .reference_batch_bytes = try mul(try add(try add(merged.batches.len, rewritten.batches.len), final.batches.len), @sizeOf(engine.ReferenceBatch)), .head_evidence_bytes = head_evidence_bytes, .view_bytes = view_bytes, .coverage_csr_bytes = try mul(merged.coverage_offsets.len + merged.coverage_indices.len, @sizeOf(u32)), .rewrite_bytes = rewrite_bytes, .final_bytes = try final.liveBytes(), .peak_postfit_workspace_bytes = peak_postfit } };
}

test "historical digest matches Source918 framing and signed zero is distinct" {
    const a: rewrite_mod.TrainingBatchView = .{ .consumer_layer = 5, .query_head = 20, .train = &.{ -0.0, 1.25 }, .train_positions = &.{7}, .heldout_positions = &.{} };
    const b: rewrite_mod.TrainingBatchView = .{ .consumer_layer = 6, .query_head = 21, .train = &.{ 2, 3 }, .train_positions = &.{9}, .heldout_positions = &.{} };
    const actual = try historicalBatchDigest(&.{ a, b });
    var source918 = std.crypto.hash.sha2.Sha256.init(.{});
    source918.update("antfly.production768-derived-historical-domain.v1\x00");
    const values = [_]f32{ -0.0, 1.25, 2, 3 };
    const positions = [_]u64{ 7, 9 };
    source918.update(std.mem.sliceAsBytes(&values));
    source918.update(std.mem.sliceAsBytes(&positions));
    const expected = source918.finalResult();
    try std.testing.expectEqualSlices(u8, &expected, &actual);
}

test "dense owner rejects same-size backing and view mutation and propagates cancellation OOM" {
    const allocator = std.testing.allocator;
    const values = try allocator.dupe(f32, &.{ -0.0, 1 });
    defer allocator.free(values);
    const positions = try allocator.dupe(u64, &.{ 7, 9 });
    defer allocator.free(positions);
    const batches = try allocator.dupe(engine.ReferenceBatch, &.{.{ .consumer_layer = 3, .query_head = 2, .train = values, .heldout = &.{}, .train_positions = positions, .heldout_positions = &.{} }});
    defer allocator.free(batches);
    var source: augmentation.OwnedTraining = .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions };
    var owner = try DenseOwner.init(&source, 11, 5);
    var valid = try DenseOwner.load(&owner, allocator, .{});
    try std.testing.expectEqual(@as(u32, 0x80000000), @as(u32, @bitCast(valid.training[0])));
    valid.deinit();
    values[1] = 2;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, DenseOwner.load(&owner, allocator, .{}));
    values[1] = 1;
    batches[0].train = values[1..];
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, DenseOwner.load(&owner, allocator, .{}));
    batches[0].train = values;
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, DenseOwner.load(&owner, allocator, .{ .check_fn = Cancel.check }));
    try std.testing.expectError(error.OutOfMemory, DenseOwner.load(&owner, std.testing.failing_allocator, .{}));
}

test "full owner assembles P rewritten-H F and unchanged HHO with exact splits" {
    var merged_values: [768]f32 = undefined;
    var merged_positions: [768]u64 = undefined;
    for (&merged_values, 0..) |*value, i| value.* = @floatFromInt(i);
    for (&merged_positions, 0..) |*position, i| position.* = i;
    var heldout_values: [64]f32 = @splat(9);
    var heldout_positions: [64]u64 = undefined;
    for (&heldout_positions, 0..) |*position, i| position.* = 1000 + i;
    var merged_batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 20, .train = &merged_values, .heldout = &heldout_values, .train_positions = &merged_positions, .heldout_positions = &heldout_positions, .independent_train_split = 256, .independent_train_second_split = 512 }};
    var rewritten_values: [256]f32 = @splat(-7);
    var rewritten_positions: [256]u64 = undefined;
    for (&rewritten_positions, 0..) |*position, i| position.* = 2000 + i;
    var rewritten_batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 20, .train = &rewritten_values, .heldout = &.{}, .train_positions = &rewritten_positions, .heldout_positions = &.{} }};
    const merged: merge_mod.Owned = .{ .allocator = undefined, .batches = &merged_batches, .values = &merged_values, .positions = &merged_positions, .heldout_values = &heldout_values, .heldout_positions = &heldout_positions, .coverage_offsets = &.{}, .coverage_indices = &.{}, .coverage_authority_sha256 = @splat(1), .heads = &.{}, .digest = @splat(2) };
    const rewritten: rewrite_mod.Owned = .{ .allocator = undefined, .batches = &rewritten_batches, .values = &rewritten_values, .positions = &rewritten_positions, .evidence = undefined };
    var full = try assembleFinal(std.testing.allocator, merged, rewritten, 1 << 20, 0);
    defer full.deinit();
    try std.testing.expectEqual(@as(f32, 255), full.batches[0].train[255]);
    try std.testing.expectEqual(@as(f32, -7), full.batches[0].train[256]);
    try std.testing.expectEqual(@as(f32, -7), full.batches[0].train[511]);
    try std.testing.expectEqual(@as(f32, 512), full.batches[0].train[512]);
    try std.testing.expectEqual(@as(u64, 2000), full.batches[0].train_positions[256]);
    try std.testing.expectEqual(@as(f32, 9), full.batches[0].heldout[63]);
    try std.testing.expectEqual(@as(?usize, 256), full.batches[0].independent_train_split);
    try std.testing.expectEqual(@as(?usize, 512), full.batches[0].independent_train_second_split);
    try std.testing.expectError(error.CompactionWorkspaceLimit, assembleFinal(std.testing.failing_allocator, merged, rewritten, 1, 0));
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(a: std.mem.Allocator, source: merge_mod.Owned, replacement: rewrite_mod.Owned) !void {
            var owner = try assembleFinal(a, source, replacement, 1 << 20, 0);
            defer owner.deinit();
            try std.testing.expectEqual(@as(f32, -7), owner.batches[0].train[256]);
            try std.testing.expectEqual(@as(f32, 9), owner.batches[0].heldout[63]);
        }
    }.run, .{ merged, rewritten });
}

test "historical tuple authority rejects tuple reassignment without corrupting Source918 parity" {
    const a: rewrite_mod.TrainingBatchView = .{ .consumer_layer = 5, .query_head = 20, .train = &.{ 1, 2 }, .train_positions = &.{7}, .heldout_positions = &.{} };
    const reassigned: rewrite_mod.TrainingBatchView = .{ .consumer_layer = 6, .query_head = 21, .train = a.train, .train_positions = a.train_positions, .heldout_positions = &.{} };
    const source918_a = try historicalBatchDigest(&.{a});
    const source918_reassigned = try historicalBatchDigest(&.{reassigned});
    try std.testing.expectEqualSlices(u8, &source918_a, &source918_reassigned);
    const authority_a = try historicalTupleAuthority(&.{a}, source918_a);
    const authority_reassigned = try historicalTupleAuthority(&.{reassigned}, source918_reassigned);
    try std.testing.expect(!std.mem.eql(u8, &authority_a, &authority_reassigned));
}

test "bridge prepare public body is semantically compiled" {
    const dense: augmentation.OwnedTraining = .{ .allocator = undefined, .batches = &.{}, .values = &.{}, .positions = &.{} };
    const state: checkpoint.State = .{ .identity = undefined, .logical_length = 0, .tokens = &.{}, .logits = &.{}, .layer_donors = &.{}, .donors = &.{} };
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{}, .scales_by_donor = &.{}, .softcaps_by_donor = &.{} };
    const fit: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 2 * 1024 * 1024 * 1024 };
    const backend: solver.Backend = undefined;
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, prepare(std.testing.allocator, state, layout, &.{}, &dense, &.{}, fit, backend, .{
        .source_length = 1,
        .repeat_length = 1,
        .source_freeze_sha256 = @splat(1),
        .config_authority_sha256 = @splat(2),
        .expected_shape_sha256 = @splat(3),
        .collector_required_bytes = 1,
        .collector_manifest_sha256 = @splat(4),
        .external_live_bytes = 0,
        .actual_native_device_bytes = 0,
        .combined_cap = 2 * 1024 * 1024 * 1024,
    }));
}

test "E2 guarded trial preadmits real Dense512 clone and metadata at exact cap" {
    const batch_count: usize = 56;
    const rows_per_batch: usize = 512;
    const dim: usize = 512;
    const position_count = batch_count * rows_per_batch;
    const value_count = position_count * dim;
    var dense_clone = try mul(value_count, @sizeOf(f32));
    dense_clone = try add(dense_clone, try mul(position_count, @sizeOf(u64)));
    dense_clone = try add(dense_clone, try mul(batch_count, @sizeOf(engine.ReferenceBatch) + 4 * @sizeOf(u32)));
    dense_clone = try add(dense_clone, 32);
    const external: usize = 320 * 1024 * 1024;
    const plan = try earlyRetainedPlan(external, dense_clone, batch_count, position_count);
    try plan.admit(plan.required_bytes);
    try std.testing.expectError(error.CompactionWorkspaceLimit, plan.admit(plan.required_bytes - 1));
    try std.testing.expect(plan.required_bytes < 2 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(try add(external, try add(dense_clone, batch_count * 4 * @sizeOf(u32) + position_count * @sizeOf(u64))), plan.required_bytes);
}
