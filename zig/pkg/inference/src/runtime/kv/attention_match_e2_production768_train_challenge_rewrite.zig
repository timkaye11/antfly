const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");
const adaptive = @import("attention_match_joint_adaptive_quality.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const challenge = @import("attention_match_train_challenge_historical_selection.zig");
const phase_loader = @import("attention_match_checkpoint_phase_loader.zig");
const flat_shape = @import("attention_match_e2_flat_shape_authority.zig");
const physical_budget = @import("attention_match_physical_budget.zig");

pub const policy = "e2_production768_train_challenge_historical_rewrite_v2|consumer_batches56|per_owner8,8,40|heads3|adaptive2|exact1|max_adaptive_rows3961|one_p60_divisor2_fixed025_l2_pilot_per_shared_head|pilot_train=P256_Hinitial256_F256_all_consumers|capture_tiles64_chronological|challenge=eligible_denseH512_per_consumer|planned_exact=unchanged|hho_values=absent";

fn expectedConsumerBatches(donor_layer: u32) !usize {
    return switch (donor_layer) {
        4, 9 => 8,
        14 => 40,
        else => error.NativeCoverageShapeUnsupported,
    };
}

pub const Config = struct {
    fit: solver.Config,
    backend: solver.Backend,
    combined_cap: usize,
    external_live_bytes: usize,
    postfit_workspace_bytes: usize,
    source_freeze_sha256: [32]u8,
    dense_manifest_sha256: [32]u8,
    initial_historical_sha256: [32]u8,
    skip_donor_layer: u32,
    skip_head_index: usize,
    hybrid_coverage_anchor: bool = false,
    coverage_offsets: []const u32 = &.{},
    coverage_indices: []const u32 = &.{},
    coverage_authority_sha256: [32]u8 = @splat(0),
    shape: ?flat_shape.Shape = null,
    expected_shape_digest: [32]u8 = @splat(0),
    private_performance_policy: engine.Options.PrivatePerformancePolicy = .owned_capture_summary_incremental_reduction_critical025_v1,
    expected_query_batch_rows: u16 = 64,
    maximum_adaptive_candidate_rows: usize = 3961,
};

/// Deliberately cannot carry heldout query values. Role positions are retained
/// solely so Dense512 rows can be excluded before their values are observed.
pub const TrainingBatchView = struct {
    consumer_layer: u32,
    query_head: u32,
    train: []const f32,
    train_positions: []const u64,
    heldout_positions: []const u64,
};

pub const DenseMetadataOwned = struct {
    allocator: std.mem.Allocator,
    manifest_sha256: [32]u8,
    consumer_layers: []u32,
    query_heads: []u32,
    counts: []u32,
    dimensions: []u32,
    positions: []u64,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.dimensions);
        self.allocator.free(self.counts);
        self.allocator.free(self.query_heads);
        self.allocator.free(self.consumer_layers);
        self.* = undefined;
    }

    pub fn liveBytes(self: @This()) !usize {
        const bytes = try mul(self.consumer_layers.len + self.query_heads.len + self.counts.len + self.dimensions.len, @sizeOf(u32));
        return add(bytes, try mul(self.positions.len, @sizeOf(u64)));
    }
};

pub const DenseLoader = struct {
    context: ?*anyopaque,
    load_fn: *const fn (?*anyopaque, std.mem.Allocator, solver.Control) anyerror!augmentation.Owned,
    /// Conservative maximum live bytes of the authenticated reader, including
    /// its manifest allocation/transient. This is admitted globally before the
    /// callback may allocate anything.
    maximum_live_bytes: usize,

    pub fn load(self: @This(), allocator: std.mem.Allocator, control: solver.Control) !augmentation.Owned {
        try control.check();
        return self.load_fn(self.context, allocator, control);
    }
};

pub fn freezeDenseMetadata(allocator: std.mem.Allocator, dense: *const augmentation.Owned, maximum_bytes: usize) !DenseMetadataOwned {
    if (dense.batches.len == 0 or maximum_bytes == 0 or
        dense.consumer_layers.len != dense.batches.len or dense.query_heads.len != dense.batches.len or
        dense.counts.len != dense.batches.len or dense.dimensions.len != dense.batches.len or dense.positions.len == 0)
        return error.InvalidTrainingAugmentation;
    var expected_positions: usize = 0;
    for (dense.counts) |count| expected_positions = try add(expected_positions, @intCast(count));
    if (expected_positions != dense.positions.len) return error.InvalidTrainingAugmentation;
    var required = try mul(dense.consumer_layers.len + dense.query_heads.len + dense.counts.len + dense.dimensions.len, @sizeOf(u32));
    required = try add(required, try mul(dense.positions.len, @sizeOf(u64)));
    if (required > maximum_bytes) return error.CompactionWorkspaceLimit;
    const consumer_layers = try allocator.dupe(u32, dense.consumer_layers);
    errdefer allocator.free(consumer_layers);
    const query_heads = try allocator.dupe(u32, dense.query_heads);
    errdefer allocator.free(query_heads);
    const counts = try allocator.dupe(u32, dense.counts);
    errdefer allocator.free(counts);
    const dimensions = try allocator.dupe(u32, dense.dimensions);
    errdefer allocator.free(dimensions);
    const positions = try allocator.dupe(u64, dense.positions);
    errdefer allocator.free(positions);
    var manifest_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &manifest_sha256, .{});
    return .{ .allocator = allocator, .manifest_sha256 = manifest_sha256, .consumer_layers = consumer_layers, .query_heads = query_heads, .counts = counts, .dimensions = dimensions, .positions = positions };
}

fn validateReload(metadata: DenseMetadataOwned, dense: *const augmentation.Owned) !void {
    var manifest_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(dense.manifest_bytes, &manifest_sha256, .{});
    if (!std.mem.eql(u8, &manifest_sha256, &metadata.manifest_sha256) or
        !std.mem.eql(u32, dense.consumer_layers, metadata.consumer_layers) or !std.mem.eql(u32, dense.query_heads, metadata.query_heads) or
        !std.mem.eql(u32, dense.counts, metadata.counts) or !std.mem.eql(u32, dense.dimensions, metadata.dimensions) or
        !std.mem.eql(u64, dense.positions, metadata.positions)) return error.InvalidTrainingAugmentationBinding;
}

fn denseLiveBytes(dense: augmentation.Owned) !usize {
    var bytes = dense.manifest_bytes.len;
    bytes = try add(bytes, try mul(dense.batches.len, @sizeOf(engine.ReferenceBatch)));
    bytes = try add(bytes, try mul(dense.training.len, @sizeOf(f32)));
    bytes = try add(bytes, try mul(dense.positions.len, @sizeOf(u64)));
    bytes = try add(bytes, try mul(dense.consumer_layers.len + dense.query_heads.len + dense.counts.len + dense.dimensions.len, @sizeOf(u32)));
    return bytes;
}

fn admitDenseReload(current_live: usize, loader_maximum_live: usize, combined_cap: usize) !void {
    if (loader_maximum_live == 0 or current_live > combined_cap or loader_maximum_live > combined_cap - current_live)
        return error.CompactionWorkspaceLimit;
}

pub const Evidence = struct {
    total_batch_count: usize,
    pilot_head_count: usize,
    skipped_exact_head_count: usize,
    skipped_exact_batch_count: usize,
    rewritten_batch_count: usize,
    maximum_train_challenge_error: f64,
    mean_train_challenge_error: f64,
    selection_digest: [32]u8,
    frozen_before_heldout: bool = true,
    heldout_value_observations: usize = 0,
    hybrid_mandatory_count: usize = 0,
    hybrid_coverage_count: usize = 0,
    hybrid_anchor_count: usize = 0,
    hybrid_challenge_count: usize = 0,
    coverage_authority_sha256: [32]u8 = @splat(0),
    hybrid_authority_sha256: [32]u8 = @splat(0),
    shape_authority_sha256: [32]u8 = @splat(0),
    maximum_adaptive_candidate_rows: usize,
    adaptive_candidate_rows: [2]usize,
    adaptive_candidate_physical_bytes: [2]usize,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    evidence: Evidence,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

const Mapped = struct { donor: usize, donor_layer: u32, head_index: usize, head: checkpoint.Head, scale: f64, softcap: f64 };

pub const ReleaseFn = *const fn (?*anyopaque) anyerror!void;
pub const PhaseSource = struct {
    authority: phase_loader.AuthorityOwned,
    loader: phase_loader.Loader,
    release_context: ?*anyopaque,
    release_fn: ReleaseFn,
    external_without_checkpoint: usize,
    prerelease_external_live_bytes: usize,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn expectedTrainingCacheBytes(query_rows: usize, source_rows: usize, dim: usize) !usize {
    // exact_f32 probability cache plus the retained FP64 target matrix. The
    // runtime planner may conservatively admit more; this field localizes the
    // dominant concrete owner without pretending to be its full bound.
    const probabilities = try mul(try mul(query_rows, source_rows), @sizeOf(f32));
    const targets = try mul(try mul(query_rows, dim), @sizeOf(f64));
    return add(probabilities, targets);
}
fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}

fn explicitShapeBindingValid(shape: flat_shape.Validated, source_freeze_sha256: [32]u8, expected_shape_digest: [32]u8) bool {
    return std.mem.eql(u8, &shape.shape.authority_sha256, &source_freeze_sha256) and nonzero(expected_shape_digest) and std.mem.eql(u8, &expected_shape_digest, &shape.digest);
}

fn coverageAuthority(primary: []const TrainingBatchView, offsets: []const u32, indices: []const u32) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e2-production768-historical-coverage-csr-authority.v1\x00");
    var bytes8: [8]u8 = undefined;
    var bytes4: [4]u8 = undefined;
    std.mem.writeInt(u64, &bytes8, @intCast(primary.len), .little);
    hash.update(&bytes8);
    for (primary, 0..) |batch, batch_index| {
        const begin: usize = offsets[batch_index];
        const end: usize = offsets[batch_index + 1];
        if (end < begin or end > indices.len) return error.InvalidTrainingAugmentationBinding;
        var frame: [16]u8 = undefined;
        std.mem.writeInt(u32, frame[0..4], batch.consumer_layer, .little);
        std.mem.writeInt(u32, frame[4..8], batch.query_head, .little);
        std.mem.writeInt(u32, frame[8..12], @intCast(batch_index), .little);
        std.mem.writeInt(u32, frame[12..16], @intCast(end - begin), .little);
        hash.update(&frame);
        for (indices[begin..end], 0..) |value, i| {
            if (value >= 512 or (i > 0 and indices[begin + i - 1] >= value)) return error.InvalidTrainingAugmentationBinding;
            std.mem.writeInt(u32, &bytes4, value, .little);
            hash.update(&bytes4);
        }
    }
    for (offsets) |value| {
        std.mem.writeInt(u32, &bytes4, value, .little);
        hash.update(&bytes4);
    }
    return hash.finalResult();
}

test "coverage authority matches producer framing and row crossbind is bit exact" {
    const batch: TrainingBatchView = .{ .consumer_layer = 5, .query_head = 20, .train = &.{}, .train_positions = &.{}, .heldout_positions = &.{} };
    const actual = try coverageAuthority(&.{batch}, &.{ 0, 2 }, &.{ 3, 9 });
    try std.testing.expectEqualSlices(u8, &.{ 0xc2, 0x09, 0x9a, 0x9b, 0x5a, 0x81, 0xd4, 0xc9, 0x6b, 0x8e, 0xab, 0xcf, 0xfb, 0x9f, 0x65, 0x1e, 0xbb, 0x45, 0xed, 0x0b, 0x9b, 0xdb, 0x55, 0xd9, 0x64, 0x31, 0x88, 0x66, 0xd8, 0x13, 0x23, 0x40 }, &actual);
    const positive = [_]f32{0.0};
    const negative = [_]f32{-0.0};
    try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(&positive), std.mem.sliceAsBytes(&negative)));
}

test "explicit shape binds the same source freeze and expected digest" {
    const validated = try flat_shape.validate(.{ .authority_sha256 = @splat(7) });
    try std.testing.expect(explicitShapeBindingValid(validated, @splat(7), validated.digest));
    try std.testing.expect(!explicitShapeBindingValid(validated, @splat(8), validated.digest));
    try std.testing.expect(!explicitShapeBindingValid(validated, @splat(7), @splat(9)));
}

fn sameTuple(a: anytype, b: anytype) bool {
    return a.consumer_layer == b.consumer_layer and a.query_head == b.query_head;
}

fn mappedHead(state: checkpoint.State, layout: engine.Layout, batch: anytype) !Mapped {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len) return error.InvalidTrainingAugmentation;
    const donor = state.layer_donors[batch.consumer_layer];
    if (donor >= state.donors.len or donor >= layout.scales_by_donor.len or donor >= layout.softcaps_by_donor.len) return error.InvalidTrainingAugmentation;
    const heads = state.donors[donor].heads;
    if (heads.len == 0 or layout.query_heads_by_layer[batch.consumer_layer] % heads.len != 0) return error.InvalidTrainingAugmentation;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / heads.len;
    if (group == 0) return error.InvalidTrainingAugmentation;
    const head_index = batch.query_head / group;
    if (head_index >= heads.len) return error.InvalidTrainingAugmentation;
    return .{ .donor = donor, .donor_layer = state.donors[donor].layer, .head_index = head_index, .head = heads[head_index], .scale = layout.scales_by_donor[donor], .softcap = layout.softcaps_by_donor[donor] };
}

fn mappedFrozen(authority: phase_loader.AuthorityOwned, batch: anytype) !Mapped {
    const entry = try authority.entry(batch.consumer_layer, batch.query_head);
    return .{ .donor = entry.donor_index, .donor_layer = entry.donor_layer, .head_index = entry.head_index, .head = undefined, .scale = entry.scale, .softcap = entry.softcap };
}

const WideHead = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    fn deinit(self: *@This()) void {
        self.allocator.free(self.bias);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.* = undefined;
    }
    fn head(self: *@This(), source: checkpoint.Head, scale: f64, softcap: f64) solver.Head {
        return .{ .keys = self.keys, .values = self.values, .bias = self.bias, .rows = source.positions.len, .dim = source.dim, .scale = scale, .softcap = softcap };
    }
};

fn widen(allocator: std.mem.Allocator, source: checkpoint.Head) !WideHead {
    const keys = try allocator.alloc(f64, source.keys.len);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, source.values.len);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, source.bias.len);
    errdefer allocator.free(bias);
    for (keys, source.keys) |*out, value| out.* = value;
    for (values, source.values) |*out, value| out.* = value;
    for (bias, source.bias) |*out, value| out.* = value;
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias };
}

fn appendTrain(out: []f64, at: *usize, source: []const f32) void {
    for (source) |value| {
        out[at.*] = value;
        at.* += 1;
    }
}

pub fn rewrite(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    primary: []const TrainingBatchView,
    initial_historical: []const TrainingBatchView,
    future: []const TrainingBatchView,
    dense_metadata: DenseMetadataOwned,
    dense_loader: DenseLoader,
    phase_source: ?PhaseSource,
    config: Config,
) !Owned {
    try config.fit.control.check();
    const shape = try flat_shape.validate(config.shape orelse .{ .authority_sha256 = config.source_freeze_sha256 });
    if (config.shape != null) {
        if (!explicitShapeBindingValid(shape, config.source_freeze_sha256, config.expected_shape_digest)) return error.InvalidTrainingAugmentationBinding;
    } else if (nonzero(config.expected_shape_digest)) return error.InvalidTrainingAugmentationBinding;
    if (primary.len == 0 or primary.len != initial_historical.len or primary.len != future.len or primary.len != dense_metadata.counts.len or
        dense_metadata.consumer_layers.len != primary.len or dense_metadata.query_heads.len != primary.len or
        dense_metadata.dimensions.len != primary.len or
        config.combined_cap == 0 or config.backend.summary_fn == null or !nonzero(config.source_freeze_sha256) or
        !nonzero(config.dense_manifest_sha256) or !nonzero(config.initial_historical_sha256)) return error.InvalidFitConfig;
    if (!std.mem.eql(u8, &dense_metadata.manifest_sha256, &config.dense_manifest_sha256))
        return error.InvalidTrainingAugmentationBinding;
    if (config.hybrid_coverage_anchor) {
        if (config.coverage_offsets.len != primary.len + 1 or config.coverage_offsets[0] != 0 or config.coverage_offsets[primary.len] != config.coverage_indices.len or !nonzero(config.coverage_authority_sha256)) return error.InvalidTrainingAugmentationBinding;
        for (config.coverage_offsets[1..], config.coverage_offsets[0..primary.len]) |next, previous| if (next < previous) return error.InvalidTrainingAugmentationBinding;
        const actual_coverage_authority = try coverageAuthority(primary, config.coverage_offsets, config.coverage_indices);
        if (!std.mem.eql(u8, &actual_coverage_authority, &config.coverage_authority_sha256)) return error.InvalidTrainingAugmentationBinding;
    } else if (config.coverage_offsets.len != 0 or config.coverage_indices.len != 0 or nonzero(config.coverage_authority_sha256)) return error.InvalidTrainingAugmentationBinding;
    const dim = shape.shape.dim;
    // Close tuple authority before any output or pilot allocation. Positional
    // zipping is legal only after all four authenticated producers agree.
    for (primary, initial_historical, future, dense_metadata.consumer_layers, dense_metadata.query_heads, dense_metadata.counts, dense_metadata.dimensions) |p, h, f, consumer_layer, query_head, count, dimension| {
        if (!sameTuple(p, h) or !sameTuple(p, f) or p.consumer_layer != consumer_layer or p.query_head != query_head or
            p.train.len != 256 * dim or h.train.len != 256 * dim or f.train.len != 256 * dim or
            p.train_positions.len != 256 or h.train_positions.len != 256 or f.train_positions.len != 256 or
            p.heldout_positions.len != 64 or h.heldout_positions.len != 0 or f.heldout_positions.len != 0 or count != 512 or dimension != dim)
            return error.InvalidTrainingAugmentation;
    }
    if (dense_metadata.positions.len != try mul(primary.len, shape.shape.dense_historical_rows)) return error.InvalidTrainingAugmentation;
    if (primary.len != shape.shape.batch_count) return error.InvalidTrainingAugmentation;
    for (primary, 0..) |batch, i| for (primary[0..i]) |prior| if (sameTuple(batch, prior)) return error.InvalidTrainingAugmentation;
    var identities: [3][2]usize = undefined;
    var identity_count: usize = 0;
    for (primary) |batch| {
        const mapped = try mappedHead(state, layout, batch);
        if (phase_source) |phase| {
            const frozen = try phase.authority.entry(batch.consumer_layer, batch.query_head);
            if (frozen.donor_index != mapped.donor or frozen.donor_layer != mapped.donor_layer or frozen.head_index != mapped.head_index or
                frozen.rows != mapped.head.positions.len or frozen.dim != @as(usize, mapped.head.dim) or frozen.scale != mapped.scale or frozen.softcap != mapped.softcap)
                return error.InvalidTrainingAugmentationBinding;
        }
        var seen = false;
        for (identities[0..identity_count]) |identity| if (identity[0] == mapped.donor and identity[1] == mapped.head_index) {
            seen = true;
            break;
        };
        if (seen) continue;
        if (identity_count == identities.len) return error.InvalidTrainingAugmentation;
        identities[identity_count] = .{ mapped.donor, mapped.head_index };
        identity_count += 1;
        var consumers: usize = 0;
        for (primary) |other_batch| {
            const other = try mappedHead(state, layout, other_batch);
            consumers += @intFromBool(other.donor == mapped.donor and other.head_index == mapped.head_index);
        }
        if (consumers == 0) return error.InvalidTrainingAugmentation;
        if (consumers != try expectedConsumerBatches(mapped.donor_layer)) return error.NativeCoverageShapeUnsupported;
    }
    if (identity_count != identities.len) return error.InvalidTrainingAugmentation;
    const value_count = try mul(try mul(primary.len, 256), dim);
    const position_count = try mul(primary.len, 256);
    var output_live = try mul(primary.len, @sizeOf(engine.ReferenceBatch));
    output_live = try add(output_live, try mul(value_count, @sizeOf(f32)));
    output_live = try add(output_live, try mul(position_count, @sizeOf(u64)));
    output_live = try add(output_live, try mul(primary.len, @sizeOf(bool)));
    const preallocation_external = if (phase_source) |phase| phase.prerelease_external_live_bytes else config.external_live_bytes;
    if (preallocation_external > config.combined_cap or output_live > config.combined_cap - preallocation_external) return error.CompactionWorkspaceLimit;

    const batches = try allocator.alloc(engine.ReferenceBatch, primary.len);
    errdefer allocator.free(batches);
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    var completed = try allocator.alloc(bool, primary.len);
    defer allocator.free(completed);
    @memset(completed, false);
    var aggregate = std.crypto.hash.sha2.Sha256.init(.{});
    aggregate.update(policy);
    aggregate.update(&config.source_freeze_sha256);
    aggregate.update(&config.dense_manifest_sha256);
    aggregate.update(&config.initial_historical_sha256);
    aggregate.update(std.mem.asBytes(&config.skip_donor_layer));
    aggregate.update(std.mem.asBytes(&config.skip_head_index));
    aggregate.update(std.mem.asBytes(&primary.len));
    if (config.shape != null) aggregate.update(&shape.digest);
    var pilot_heads: usize = 0;
    var maximum_candidate_rows: usize = 0;
    var adaptive_candidate_rows: [2]usize = @splat(0);
    var adaptive_physical_bytes: [2]usize = @splat(0);
    var skipped: usize = 0;
    var skipped_batches: usize = 0;
    var rewritten: usize = 0;
    var max_error: f64 = 0;
    var error_sum: f64 = 0;
    var hybrid_mandatory_count: usize = 0;
    var hybrid_coverage_count: usize = 0;
    var hybrid_anchor_count: usize = 0;
    var hybrid_challenge_count: usize = 0;
    var hybrid_authority = std.crypto.hash.sha2.Sha256.init(.{});
    hybrid_authority.update("antfly.e2-production768-hybrid-authority.v1\x00");
    hybrid_authority.update(&config.coverage_authority_sha256);

    if (phase_source) |phase| try phase.release_fn(phase.release_context);

    for (primary, 0..) |seed_batch, seed_index| {
        if (completed[seed_index]) continue;
        const mapped = if (phase_source) |phase| try mappedFrozen(phase.authority, seed_batch) else try mappedHead(state, layout, seed_batch);
        const skip = mapped.donor_layer == config.skip_donor_layer and mapped.head_index == config.skip_head_index;
        var matching: usize = 0;
        for (primary) |batch| {
            const other = if (phase_source) |phase| try mappedFrozen(phase.authority, batch) else try mappedHead(state, layout, batch);
            matching += @intFromBool(other.donor == mapped.donor and other.head_index == mapped.head_index);
        }
        if (matching == 0) return error.InvalidTrainingAugmentation;
        if (matching != try expectedConsumerBatches(mapped.donor_layer)) return error.NativeCoverageShapeUnsupported;

        if (skip) {
            skipped += 1;
            for (primary, initial_historical, 0..) |p, h, batch_index| {
                const other = if (phase_source) |phase| try mappedFrozen(phase.authority, p) else try mappedHead(state, layout, p);
                if (other.donor != mapped.donor or other.head_index != mapped.head_index) continue;
                if (h.train.len != 256 * dim or h.train_positions.len != 256) return error.InvalidTrainingAugmentation;
                const value_start = batch_index * 256 * dim;
                const position_start = batch_index * 256;
                @memcpy(values[value_start..][0..h.train.len], h.train);
                @memcpy(positions[position_start..][0..256], h.train_positions);
                batches[batch_index] = .{ .consumer_layer = h.consumer_layer, .query_head = h.query_head, .train = values[value_start..][0..h.train.len], .heldout = &.{}, .train_positions = positions[position_start..][0..256], .heldout_positions = &.{} };
                completed[batch_index] = true;
                skipped_batches += 1;
            }
            continue;
        }

        var phased_head: ?phase_loader.HeadOwned = if (phase_source) |phase| try phase_loader.loadHead(allocator, phase.authority, phase.loader, seed_batch.consumer_layer, seed_batch.query_head, try add(phase.external_without_checkpoint, output_live), config.combined_cap, config.fit.control) else null;
        defer if (phased_head) |*owned| owned.deinit();
        const source_head = if (phased_head) |*owned| owned.head() else mapped.head;
        const source_head_live = if (phased_head) |owned| try owned.liveBytes() else 0;
        const train_values = try mul(try mul(matching, 768), dim);
        const train_bytes = try mul(train_values, @sizeOf(f64));
        const train_phase_live = try add(config.external_live_bytes, try add(output_live, try add(source_head_live, train_bytes)));
        if (train_phase_live > config.combined_cap) return error.CompactionWorkspaceLimit;
        const train = try allocator.alloc(f64, train_values);
        defer allocator.free(train);
        var train_at: usize = 0;
        for (primary, initial_historical, future) |p, h, f| {
            const other = if (phase_source) |phase| try mappedFrozen(phase.authority, p) else try mappedHead(state, layout, p);
            if (other.donor != mapped.donor or other.head_index != mapped.head_index) continue;
            if (p.train.len != 256 * dim or h.train.len != 256 * dim or f.train.len != 256 * dim) return error.InvalidTrainingAugmentation;
            appendTrain(train, &train_at, p.train);
            appendTrain(train, &train_at, h.train);
            appendTrain(train, &train_at, f.train);
        }
        if (train_at != train.len) return error.InvalidTrainingAugmentation;
        if (source_head.positions.len != shape.shape.source_rows) return error.InvalidTrainingAugmentationBinding;
        const plan_options: engine.Options = .{ .prefix_rows = shape.shape.prefix_rows, .recent_rows = shape.shape.recent_rows, .chunk_rows = shape.shape.chunk_rows, .retention_divisor = 2 };
        const plan = (try engine.headPlan(shape.shape.source_rows, source_head, plan_options)) orelse return error.InvalidFitConfig;
        if (train_bytes != try mul(train.len, @sizeOf(f64))) return error.InvalidTrainingAugmentation;
        const wide_bytes = try mul(try add(source_head.keys.len, try add(source_head.values.len, source_head.bias.len)), @sizeOf(f64));
        // The widened source is needed only by the challenge selector. Keep it
        // out of the cache-backed pilot phase, whose source normalization is
        // independently owned and admitted inside runHeadWithPrimaryProbe.
        const pilot_external = try add(config.external_live_bytes, try add(source_head_live, try add(output_live, train_bytes)));
        const query_rows = train.len / dim;
        if (query_rows != matching * 768) return error.InvalidTrainingAugmentation;
        const range_count = shape.range_count;
        if (range_count != (std.math.divCeil(usize, plan.old_rows - plan.prefix_rows, shape.shape.chunk_rows) catch return error.InvalidFitConfig)) return error.InvalidFitConfig;
        const actual_postfit_workspace = try replay.postfitWorkspaceReserve(query_rows, source_head.positions.len, source_head.positions.len, range_count);
        if (actual_postfit_workspace > config.postfit_workspace_bytes) return error.CompactionWorkspaceLimit;
        const expected_cache_bytes = try expectedTrainingCacheBytes(query_rows, source_head.positions.len, source_head.dim);
        const expected_candidate_bytes = try mul(source_head.positions.len, try add(try mul(source_head.dim, 2 * @sizeOf(f64)), 2 * @sizeOf(usize) + @sizeOf(f64)));
        std.debug.print("production768_train_challenge_pilot phase=begin donor_layer={d} head={d} consumer_count={d} query_rows={d} source_rows={d} dim={d} external_live_bytes={d} output_live_bytes={d} train_bytes={d} deferred_wide_head_bytes={d} pilot_external_live_bytes={d} expected_cache_bytes={d} expected_candidate_upper_bytes={d} postfit_workspace_bytes={d} configured_postfit_ceiling={d} combined_cap={d} fit_backend_present={} fit_backend_summary={} validation_backend_summary={} diagnostic_only=true\n", .{ mapped.donor_layer, mapped.head_index, matching, query_rows, source_head.positions.len, source_head.dim, config.external_live_bytes, output_live, train_bytes, wide_bytes, pilot_external, expected_cache_bytes, expected_candidate_bytes, actual_postfit_workspace, config.postfit_workspace_bytes, config.combined_cap, config.fit.backend != null, if (config.fit.backend) |backend| backend.summary_fn != null else false, config.backend.summary_fn != null });
        if ((config.private_performance_policy == .diagnostic_capture_gram_batch256_v1) != (config.expected_query_batch_rows == 256) or
            (config.expected_query_batch_rows != 64 and config.expected_query_batch_rows != 256))
            return error.InvalidFitConfig;
        var pilot = adaptive.runHeadWithPrimaryProbe(allocator, source_head, plan, train, mapped.scale, mapped.softcap, config.fit, pilot_external, config.combined_cap, .accepted_f32_gram_reuse_v1, .flat_gram_only, 0, 0, false, false, .accelerate_fp64_summary_v1, .disabled, .incremental_gram_v1, config.private_performance_policy, .train_global_residual_range_mass_l2_exact_f32_cache_v1, false, true, .{ .heldout_query_rows = shape.shape.heldout_rows * matching, .critical_policy = .train_probability_gte_0_025_no_top1_v1, .postfit_workspace_bytes = actual_postfit_workspace, .validation_backend = config.backend }) catch |err| {
            std.debug.print("production768_train_challenge_pilot phase=failed donor_layer={d} head={d} consumer_count={d} query_rows={d} error={s} pilot_external_live_bytes={d} deferred_wide_head_bytes={d} expected_cache_bytes={d} expected_candidate_upper_bytes={d} postfit_workspace_bytes={d} combined_cap={d} diagnostic_only=true\n", .{ mapped.donor_layer, mapped.head_index, matching, query_rows, @errorName(err), pilot_external, wide_bytes, expected_cache_bytes, expected_candidate_bytes, actual_postfit_workspace, config.combined_cap });
            return err;
        };
        defer pilot.deinit();
        if (pilot.candidate.head.rows > config.maximum_adaptive_candidate_rows) return error.CompactionPhysicalBudgetUnsupported;
        if (pilot_heads >= adaptive_physical_bytes.len) return error.NativeCoverageShapeUnsupported;
        maximum_candidate_rows = @max(maximum_candidate_rows, pilot.candidate.head.rows);
        adaptive_candidate_rows[pilot_heads] = pilot.candidate.head.rows;
        adaptive_physical_bytes[pilot_heads] = try physical_budget.candidateHeadBytes(pilot.candidate.head.rows, pilot.candidate.head.dim);
        std.debug.print("production768_train_challenge_pilot phase=complete donor_layer={d} head={d} consumer_count={d} query_rows={d} candidate_rows={d} candidate_live_bytes={d} train_mass_error={d} diagnostic_only=true\n", .{ mapped.donor_layer, mapped.head_index, matching, query_rows, pilot.candidate.head.rows, pilot.candidate_live_bytes, pilot.train_metrics.max_relative_mass_error });
        const reload_current_live = try add(pilot_external, pilot.candidate_live_bytes);
        try admitDenseReload(reload_current_live, dense_loader.maximum_live_bytes, config.combined_cap);
        var reloaded_dense = try dense_loader.load(allocator, config.fit.control);
        defer reloaded_dense.deinit();
        try validateReload(dense_metadata, &reloaded_dense);
        const reloaded_dense_bytes = try denseLiveBytes(reloaded_dense);
        if (reloaded_dense_bytes > dense_loader.maximum_live_bytes) return error.CompactionWorkspaceLimit;
        const selector_retained = try add(pilot_external, try add(pilot.candidate_live_bytes, try add(wide_bytes, reloaded_dense_bytes)));
        if (selector_retained > config.combined_cap) return error.CompactionWorkspaceLimit;
        var wide = try widen(allocator, source_head);
        defer wide.deinit();
        const original = wide.head(source_head, mapped.scale, mapped.softcap);
        pilot_heads += 1;
        for (primary, reloaded_dense.batches, reloaded_dense.counts, 0..) |p, dense_batch, count, batch_index| {
            const other = if (phase_source) |phase| try mappedFrozen(phase.authority, p) else try mappedHead(state, layout, p);
            if (other.donor != mapped.donor or other.head_index != mapped.head_index) continue;
            if (count != 512 or dense_batch.train.len != 512 * dim or dense_batch.train_positions.len != 512 or p.heldout_positions.len != 64) return error.InvalidTrainingAugmentation;
            const selector_external = try add(config.external_live_bytes, try add(source_head_live, try add(output_live, try add(train_bytes, try add(wide_bytes, try add(pilot.candidate_live_bytes, reloaded_dense_bytes))))));
            var selected = if (config.hybrid_coverage_anchor) blk: {
                var original_h256: [256]u32 = undefined;
                for (initial_historical[batch_index].train_positions, 0..) |position, hi| {
                    const dense_index = std.sort.binarySearch(u64, dense_batch.train_positions, position, struct {
                        fn order(key: u64, value: u64) std.math.Order {
                            return std.math.order(key, value);
                        }
                    }.order) orelse return error.InvalidTrainingAugmentationBinding;
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(initial_historical[batch_index].train[hi * dim ..][0..dim]), std.mem.sliceAsBytes(dense_batch.train[dense_index * dim ..][0..dim]))) return error.InvalidTrainingAugmentationBinding;
                    original_h256[hi] = @intCast(dense_index);
                }
                const coverage = config.coverage_indices[config.coverage_offsets[batch_index]..config.coverage_offsets[batch_index + 1]];
                break :blk try challenge.selectHybrid(allocator, original, pilot.candidate.head, dense_batch.train, dense_batch.train_positions, p.heldout_positions, 256, &original_h256, coverage, config.coverage_authority_sha256, config.backend, selector_external, config.combined_cap, config.fit.control);
            } else try challenge.select(allocator, original, pilot.candidate.head, dense_batch.train, dense_batch.train_positions, p.heldout_positions, 256, config.backend, selector_external, config.combined_cap, config.fit.control);
            defer selected.deinit();
            hybrid_mandatory_count = try add(hybrid_mandatory_count, selected.hybrid_mandatory_count);
            hybrid_coverage_count = try add(hybrid_coverage_count, selected.hybrid_coverage_count);
            hybrid_anchor_count = try add(hybrid_anchor_count, selected.hybrid_anchor_count);
            hybrid_challenge_count = try add(hybrid_challenge_count, selected.hybrid_challenge_count);
            if (config.hybrid_coverage_anchor) {
                hybrid_authority.update(&selected.hybrid_authority_sha256);
            }
            const value_start = batch_index * 256 * dim;
            const position_start = batch_index * 256;
            for (selected.indices, 0..) |index_u32, row| {
                const index: usize = index_u32;
                @memcpy(values[value_start + row * dim ..][0..dim], dense_batch.train[index * dim ..][0..dim]);
                positions[position_start + row] = dense_batch.train_positions[index];
            }
            batches[batch_index] = .{ .consumer_layer = p.consumer_layer, .query_head = p.query_head, .train = values[value_start..][0 .. 256 * dim], .heldout = &.{}, .train_positions = positions[position_start..][0..256], .heldout_positions = &.{} };
            aggregate.update(&pilot.candidate_sha256);
            aggregate.update(&selected.selection_digest);
            max_error = @max(max_error, selected.maximum_error);
            error_sum += selected.mean_error;
            rewritten += 1;
            completed[batch_index] = true;
        }
    }
    for (completed) |done| if (!done) return error.InvalidTrainingAugmentation;
    if (pilot_heads + skipped != 3 or pilot_heads != 2 or skipped != 1 or rewritten + skipped_batches != primary.len) return error.InvalidTrainingAugmentation;
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .evidence = .{ .total_batch_count = primary.len, .pilot_head_count = pilot_heads, .skipped_exact_head_count = skipped, .skipped_exact_batch_count = skipped_batches, .rewritten_batch_count = rewritten, .maximum_train_challenge_error = max_error, .mean_train_challenge_error = if (rewritten == 0) 0 else error_sum / @as(f64, @floatFromInt(rewritten)), .selection_digest = aggregate.finalResult(), .hybrid_mandatory_count = hybrid_mandatory_count, .hybrid_coverage_count = hybrid_coverage_count, .hybrid_anchor_count = hybrid_anchor_count, .hybrid_challenge_count = hybrid_challenge_count, .coverage_authority_sha256 = config.coverage_authority_sha256, .hybrid_authority_sha256 = if (config.hybrid_coverage_anchor) hybrid_authority.finalResult() else @splat(0), .shape_authority_sha256 = if (config.shape != null) shape.digest else @splat(0), .maximum_adaptive_candidate_rows = maximum_candidate_rows, .adaptive_candidate_rows = adaptive_candidate_rows, .adaptive_candidate_physical_bytes = adaptive_physical_bytes } };
}

test "E2 owner census binds 56 batches and owner14 remains one chronological tiled pilot" {
    try std.testing.expectEqual(@as(usize, 8), try expectedConsumerBatches(4));
    try std.testing.expectEqual(@as(usize, 8), try expectedConsumerBatches(9));
    try std.testing.expectEqual(@as(usize, 40), try expectedConsumerBatches(14));
    try std.testing.expectEqual(@as(usize, 56), try std.math.add(usize, try expectedConsumerBatches(4), try std.math.add(usize, try expectedConsumerBatches(9), try expectedConsumerBatches(14))));
    try std.testing.expectEqual(@as(usize, 30_720), (try expectedConsumerBatches(14)) * 768);
    try std.testing.expectEqual(@as(usize, 480), 30_720 / 64);
    const cache_bytes = try expectedTrainingCacheBytes(30_720, 8128, 512);
    try std.testing.expectEqual(@as(usize, 1_124_597_760), cache_bytes);
    const train_bytes = 30_720 * 512 * @sizeOf(f64);
    try std.testing.expectEqual(@as(usize, 125_829_120), train_bytes);
    try std.testing.expect(cache_bytes + train_bytes < 2 * 1024 * 1024 * 1024);
    try std.testing.expectError(error.NativeCoverageShapeUnsupported, expectedConsumerBatches(19));
}

fn matchingBatchCount(batches: []const TrainingBatchView, state: checkpoint.State, layout: engine.Layout, donor_layer: u32, head_index: usize) usize {
    var count: usize = 0;
    for (batches) |batch| {
        const mapped = mappedHead(state, layout, batch) catch continue;
        count += @intFromBool(state.donors[mapped.donor].layer == donor_layer and mapped.head_index == head_index);
    }
    return count;
}

test "consumer tuple substitution is rejected before payload use" {
    const a: TrainingBatchView = .{ .consumer_layer = 5, .query_head = 20, .train = &.{}, .train_positions = &.{}, .heldout_positions = &.{} };
    const same = a;
    const wrong_consumer: TrainingBatchView = .{ .consumer_layer = 6, .query_head = 20, .train = &.{}, .train_positions = &.{}, .heldout_positions = &.{} };
    const wrong_head: TrainingBatchView = .{ .consumer_layer = 5, .query_head = 21, .train = &.{}, .train_positions = &.{}, .heldout_positions = &.{} };
    try std.testing.expect(sameTuple(a, same));
    try std.testing.expect(!sameTuple(a, wrong_consumer));
    try std.testing.expect(!sameTuple(a, wrong_head));
}

test "rewrite invalid shape rejects before allocation and instantiates body" {
    const Backend = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn summary(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {}
    };
    const backend: solver.Backend = .{ .workspace_fn = Backend.workspace, .summary_fn = Backend.summary };
    const metadata: DenseMetadataOwned = .{ .allocator = undefined, .manifest_sha256 = @splat(0), .consumer_layers = &.{}, .query_heads = &.{}, .counts = &.{}, .dimensions = &.{}, .positions = &.{} };
    try std.testing.expectError(error.InvalidFitConfig, rewrite(std.testing.failing_allocator, undefined, undefined, &.{}, &.{}, &.{}, metadata, undefined, null, .{
        .fit = .{ .target_rows = 1 },
        .backend = backend,
        .combined_cap = 1,
        .external_live_bytes = 0,
        .postfit_workspace_bytes = 1,
        .source_freeze_sha256 = @splat(1),
        .dense_manifest_sha256 = @splat(2),
        .initial_historical_sha256 = @splat(3),
        .skip_donor_layer = 5,
        .skip_head_index = 0,
    }));
}

test "D23 pilot cache and postfit use actual query rows" {
    try std.testing.expectEqual(@as(usize, 440_401_920), try expectedTrainingCacheBytes(12_288, 7_936, 512));
    const actual = try replay.postfitWorkspaceReserve(12_288, 7_936, 7_936, 23);
    const stale = try replay.postfitWorkspaceReserve(15_360, 7_936, 7_936, 23);
    try std.testing.expect(actual < stale);
}

test "Dense reload authority rejects changed positions" {
    var manifest = [_]u8{'m'};
    var consumer = [_]u32{5};
    var heads = [_]u32{20};
    var counts = [_]u32{512};
    var dimensions = [_]u32{512};
    var positions = [_]u64{10};
    var empty_batches: [0]engine.ReferenceBatch = .{};
    var empty_training: [0]f32 = .{};
    var manifest_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&manifest, &manifest_sha256, .{});
    const metadata: DenseMetadataOwned = .{ .allocator = undefined, .manifest_sha256 = manifest_sha256, .consumer_layers = &consumer, .query_heads = &heads, .counts = &counts, .dimensions = &dimensions, .positions = &positions };
    var dense: augmentation.Owned = .{ .allocator = undefined, .manifest_bytes = &manifest, .batches = &empty_batches, .training = &empty_training, .positions = &positions, .consumer_layers = &consumer, .query_heads = &heads, .counts = &counts, .dimensions = &dimensions, .source_length = 1, .repeat_length = 1 };
    try validateReload(metadata, &dense);
    var changed_positions = [_]u64{11};
    dense.positions = &changed_positions;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, validateReload(metadata, &dense));
}

test "Dense loader cancellation occurs before callback" {
    const Counter = struct {
        calls: usize = 0,
        fn cancel(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
        fn load(raw: ?*anyopaque, _: std.mem.Allocator, _: solver.Control) !augmentation.Owned {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            return error.InvalidTrainingAugmentation;
        }
    };
    var counter: Counter = .{};
    const loader: DenseLoader = .{ .context = &counter, .load_fn = Counter.load, .maximum_live_bytes = 1 };
    try std.testing.expectError(error.Cancelled, loader.load(std.testing.failing_allocator, .{ .check_fn = Counter.cancel }));
    try std.testing.expectEqual(@as(usize, 0), counter.calls);
}

fn freezeMetadataAllocationFixture(allocator: std.mem.Allocator) !void {
    var manifest = [_]u8{'m'};
    var batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 20, .train = &.{}, .heldout = &.{}, .train_positions = &.{}, .heldout_positions = &.{} }};
    var empty_training: [0]f32 = .{};
    var positions = [_]u64{10};
    var consumer = [_]u32{5};
    var heads = [_]u32{20};
    var counts = [_]u32{1};
    var dimensions = [_]u32{512};
    const dense: augmentation.Owned = .{ .allocator = undefined, .manifest_bytes = &manifest, .batches = &batches, .training = &empty_training, .positions = &positions, .consumer_layers = &consumer, .query_heads = &heads, .counts = &counts, .dimensions = &dimensions, .source_length = 1, .repeat_length = 1 };
    var metadata = try freezeDenseMetadata(allocator, &dense, 1024);
    defer metadata.deinit();
}

test "Dense metadata freeze unwinds every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, freezeMetadataAllocationFixture, .{});
}

test "Dense reload is globally admitted before loader allocation" {
    try admitDenseReload(700, 300, 1000);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitDenseReload(701, 300, 1000));
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitDenseReload(0, 0, 1000));
}

test "Dense metadata rejects descriptor and count mismatch before allocation" {
    var manifest = [_]u8{'m'};
    var batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 5, .query_head = 20, .train = &.{}, .heldout = &.{}, .train_positions = &.{}, .heldout_positions = &.{} }};
    var empty_training: [0]f32 = .{};
    var positions = [_]u64{10};
    var consumer = [_]u32{5};
    var heads = [_]u32{20};
    var bad_counts = [_]u32{2};
    var dimensions = [_]u32{512};
    const dense: augmentation.Owned = .{ .allocator = undefined, .manifest_bytes = &manifest, .batches = &batches, .training = &empty_training, .positions = &positions, .consumer_layers = &consumer, .query_heads = &heads, .counts = &bad_counts, .dimensions = &dimensions, .source_length = 1, .repeat_length = 1 };
    try std.testing.expectError(error.InvalidTrainingAugmentation, freezeDenseMetadata(std.testing.failing_allocator, &dense, 1024));
}
