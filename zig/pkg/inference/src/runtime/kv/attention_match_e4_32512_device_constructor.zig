// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const bridge = @import("attention_match_native_resident_hybrid_bridge.zig");
const solver = @import("attention_match.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");
const metal_compute = @import("../../ops/metal_compute.zig");
const storage_runtime = @import("storage_runtime.zig");
const tracker_mod = @import("host_allocation_tracker.zig");
const metadata_mod = @import("attention_match_e4_32512_source_metadata.zig");
const provider_mod = @import("attention_match_e4_32512_metal_head_provider.zig");
const packed_mod = @import("attention_match_e4_32512_packed_head_source.zig");
const mixed_mod = @import("attention_match_e4_32512_streaming_source.zig");
const source_checkpoint_provider = @import("attention_match_streaming_source_checkpoint_provider.zig");
const planned_exact = @import("attention_match_planned_exact_context.zig");
const builtin = @import("builtin");
const block = @import("block.zig");

/// Test-only receipt populated after the real installed inventory, global
/// provider, mixed state digest and transactional candidate builder exist.
pub const InstalledProviderProbe = struct {
    inventory_entries: usize = 0,
    global_mappings: usize = 0,
    donor_count: usize = 0,
    candidate_descriptors: usize = 0,
    revision: u64 = 0,
    structure_sha256: [32]u8 = @splat(0),
    state_sha256: [32]u8 = @splat(0),
};

pub const Binding = struct {
    bridge_binding: bridge.PrepareBinding,
    engine_options: engine.Options,
    candidate_device_max_bytes: usize,
    retained_query_bytes: usize,
    maximum_current_head_bytes: usize,
    source_export_context: ?*anyopaque = null,
    source_export_write_fn: ?*const fn (*anyopaque, []const u8) anyerror!void = null,
    source_export_sha256_out: ?*[32]u8 = null,
    final_training_context: ?*anyopaque = null,
    final_training_fn: ?*const fn (*anyopaque, []const engine.ReferenceBatch, engine.Layout) anyerror!void = null,

    fn validate(self: @This(), source: *const engine.StreamingHeadSource) !void {
        if ((self.source_export_context == null) != (self.source_export_write_fn == null) or
            (self.source_export_context == null) != (self.source_export_sha256_out == null) or
            (self.final_training_context == null) != (self.final_training_fn == null))
            return error.InvalidTrainingAugmentationBinding;
        if (self.bridge_binding.shape == null or switch (self.bridge_binding.shape.?) {
            .e4_32512 => false,
            else => true,
        } or !self.engine_options.adaptive_v18_bounded_tsqr_value_backend or
            self.engine_options.adaptive_v18_geometry_route != .bounded_page_over_96 or
            self.candidate_device_max_bytes == 0 or self.maximum_current_head_bytes == 0 or
            !std.mem.eql(u8, &source.metadata.structure_sha256, &source.expected_structure_sha256) or
            !std.mem.eql(u8, &source.metadata.state_sha256, &source.expected_state_sha256))
            return error.InvalidTrainingAugmentationBinding;
    }
};

pub const Result = struct {
    allocator: std.mem.Allocator,
    device: metal_storage.DeviceState,
    metadata: checkpoint.StreamingMetadata,
    maximum_head_download_bytes: usize,
    source_census_physical_bytes: usize,
    source_global_storage_bytes: usize,
    evidence: engine.StreamingDeviceEvidence,
    receipt: Receipt,
    taken: bool = false,

    pub fn deinit(self: *@This()) void {
        if (self.taken) {
            self.* = undefined;
            return;
        }
        self.device.deinit();
        self.allocator.free(self.metadata.donors);
        self.allocator.free(self.metadata.layer_donors);
        self.allocator.free(self.metadata.logits);
        self.allocator.free(self.metadata.tokens);
        self.* = undefined;
    }

    pub const Transfer = struct {
        allocator: std.mem.Allocator,
        device: metal_storage.DeviceState,
        metadata: checkpoint.StreamingMetadata,
        maximum_head_download_bytes: usize,
        source_census_physical_bytes: usize,
        source_global_storage_bytes: usize,
        evidence: engine.StreamingDeviceEvidence,
        receipt: Receipt,
    };

    /// Transfers every owned payload field exactly once to the generation
    /// Result union. The caller becomes responsible for device and metadata.
    pub fn take(self: *@This()) Transfer {
        std.debug.assert(!self.taken);
        const result: Transfer = .{ .allocator = self.allocator, .device = self.device, .metadata = self.metadata, .maximum_head_download_bytes = self.maximum_head_download_bytes, .source_census_physical_bytes = self.source_census_physical_bytes, .source_global_storage_bytes = self.source_global_storage_bytes, .evidence = self.evidence, .receipt = self.receipt };
        self.taken = true;
        return result;
    }
};

pub const Receipt = struct {
    source_freeze_sha256: [32]u8,
    config_authority_sha256: [32]u8,
    coverage_authority_sha256: [32]u8,
    merge_digest: [32]u8,
    collector_manifest_sha256: [32]u8,
    collector_required_bytes: usize,
    planned_exact_donor_layer: u32,
    planned_exact_head_index: usize,
    train_mass_only_calls: usize,
    train_mass_only_gemm_calls: usize,
};

const UploadContext = struct {
    builder: *metal_storage.CandidateBuilder,

    fn upload(raw: ?*anyopaque, index: usize, head: checkpoint.Head, control: solver.Control) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        return self.builder.uploadHead(index, head, control);
    }
};

fn cloneMetadata(allocator: std.mem.Allocator, source: engine.SourceMetadata) !checkpoint.StreamingMetadata {
    const tokens = try allocator.dupe(u32, source.tokens);
    errdefer allocator.free(tokens);
    const logits = try allocator.dupe(f32, source.logits);
    errdefer allocator.free(logits);
    const layer_donors = try allocator.dupe(u32, source.layer_donors);
    errdefer allocator.free(layer_donors);
    const donors = try allocator.alloc(checkpoint.StreamingDonor, source.donors.len);
    errdefer allocator.free(donors);
    for (donors, source.donors) |*out, donor| out.* = .{ .layer = donor.layer, .window = donor.window, .head_count = donor.head_count };
    return .{ .identity = source.identity, .logical_length = source.logical_length, .tokens = tokens, .logits = logits, .layer_donors = layer_donors, .donors = donors };
}

fn freeMetadata(allocator: std.mem.Allocator, metadata: checkpoint.StreamingMetadata) void {
    allocator.free(metadata.donors);
    allocator.free(metadata.layer_donors);
    allocator.free(metadata.logits);
    allocator.free(metadata.tokens);
}

/// Executes the authenticated streamed bridge and then streams every fitted
/// head into an unpublished device candidate. CandidateBuilder is transactional:
/// any bridge, fit, validation, cancellation, or upload error leaves the
/// canonical installed source untouched and builder.deinit releases staging.
pub fn prepare(
    allocator: std.mem.Allocator,
    source: *engine.StreamingHeadSource,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    dense: *const augmentation.OwnedTraining,
    future: []const engine.ReferenceBatch,
    fit: solver.Config,
    fp64_validation: solver.Backend,
    builder: *metal_storage.CandidateBuilder,
    binding: Binding,
) !Result {
    try binding.validate(source);
    defer builder.deinit();
    var resident = try bridge.prepareFromHeadSource(allocator, source, layout, primary, dense, future, fit, fp64_validation, binding.bridge_binding);
    errdefer resident.deinit();
    if (binding.final_training_fn) |observe|
        try observe(binding.final_training_context.?, resident.final_training.batches, layout);
    var planned_context: planned_exact.Context = .{
        .enabled = true,
        .donor_layer = resident.planned_exact_donor_layer,
        .head_index = resident.planned_exact_head_index,
    };
    var engine_options = binding.engine_options;
    engine_options.adaptive_v18_fn = planned_exact.Context.callback;
    engine_options.adaptive_v18_context = &planned_context;
    var upload: UploadContext = .{ .builder = builder };
    const evidence = try engine.prepareFromHeadSourceIntoDevice(source, resident.final_training.batches, layout, engine_options, .{
        .allocator = allocator,
        .context = &upload,
        .upload_fn = UploadContext.upload,
        .candidate_device_max_bytes = binding.candidate_device_max_bytes,
        .retained_query_bytes = binding.retained_query_bytes,
        .maximum_current_head_bytes = binding.maximum_current_head_bytes,
    });
    const planned_exact_heads = planned_context.planned_exact_heads.load(.acquire);
    const adaptive_fit_attempts = planned_context.adaptive_fit_attempts.load(.acquire);
    if (planned_exact_heads != 1 or adaptive_fit_attempts != 7) return error.InvalidTrainingAugmentationBinding;
    if (engine_options.stats) |stats| {
        if (stats.v18_planned_exact_heads != 0) return error.InvalidTrainingAugmentationBinding;
        stats.v18_planned_exact_heads = planned_exact_heads;
    }
    var device = try builder.finish(source.expected_structure_sha256, source.expected_state_sha256, engine_options.fit.control);
    errdefer device.deinit();
    try source.verify_no_live_fn(source.context);
    const receipt: Receipt = .{
        .source_freeze_sha256 = resident.source_freeze_sha256,
        .config_authority_sha256 = resident.config_authority_sha256,
        .coverage_authority_sha256 = resident.coverage_authority_sha256,
        .merge_digest = resident.merge_digest,
        .collector_manifest_sha256 = resident.collector_manifest_sha256,
        .collector_required_bytes = resident.collector_required_bytes,
        .planned_exact_donor_layer = resident.planned_exact_donor_layer,
        .planned_exact_head_index = resident.planned_exact_head_index,
        .train_mass_only_calls = resident.train_challenge_mass_only_calls,
        .train_mass_only_gemm_calls = resident.train_challenge_mass_only_gemm_calls,
    };
    resident.deinit();
    const metadata = try cloneMetadata(allocator, source.metadata);
    return .{
        .allocator = allocator,
        .device = device,
        .metadata = metadata,
        .maximum_head_download_bytes = binding.maximum_current_head_bytes,
        .source_census_physical_bytes = source.metadata.source_census_physical_bytes,
        .source_global_storage_bytes = source.metadata.source_global_storage_bytes,
        .evidence = evidence,
        .receipt = receipt,
    };
}

pub const InstalledBinding = struct {
    source_kind: provider_mod.SourceKind = .installed_candidate,
    compute: *metal_compute.MetalCompute,
    storage: *storage_runtime.KvStorageRuntime,
    sequence: storage_runtime.SequenceId,
    tracker: *tracker_mod.HostAllocationTracker,
    identity: checkpoint.Identity,
    tokens: []const u32,
    logits: []const f32,
    layer_donors: []const u32,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    owner_identity: u64,
    maximum_head_download_bytes: usize,
    candidate_device_max_bytes: usize,
    constructor: Binding,
    test_installed_provider_probe: ?*InstalledProviderProbe = null,
    raw_logical_blocks: []const block.KvBlockId = &.{},
    raw_owner_layers: []const u32 = &.{},
};

/// Concrete production provider construction. Installed structural inventory,
/// global packed census, and exact local payload digests are all observed from
/// the same immutable revision. At most one downloaded/cloned head is live.
pub fn prepareInstalled(
    allocator: std.mem.Allocator,
    installed: InstalledBinding,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    dense: *const augmentation.OwnedTraining,
    future: []const engine.ReferenceBatch,
    fit: solver.Config,
    fp64_validation: solver.Backend,
) !Result {
    if (installed.tokens.len != 32512 or installed.owner_identity != installed.tracker.owner_identity or
        installed.maximum_head_download_bytes == 0 or installed.candidate_device_max_bytes == 0 or
        std.mem.allEqual(u8, &installed.token_sha256, 0) or std.mem.allEqual(u8, &installed.binary_sha256, 0) or
        (!builtin.is_test and installed.test_installed_provider_probe != null))
        return error.InvalidTrainingAugmentationBinding;
    try installed.tracker.verifyBoundedPrepare(installed.owner_identity);
    const revision = try installed.compute.attentionMatchRevision(installed.storage);
    const control = installed.constructor.engine_options.fit.control;
    const head_source: provider_mod.Source = switch (installed.source_kind) {
        .installed_candidate => .{ .installed_candidate = .{ .compute = installed.compute, .storage = installed.storage, .sequence = installed.sequence, .revision = revision } },
        .raw_prefill => .{ .raw_prefill = .{ .compute = installed.compute, .storage = installed.storage, .sequence = installed.sequence, .revision = revision, .logical_tokens = @intCast(installed.tokens.len), .logical_blocks = installed.raw_logical_blocks } },
    };
    var inventory = switch (installed.source_kind) {
        .installed_candidate => try installed.compute.attentionMatchStructuralInventory(installed.storage, installed.sequence, allocator, revision, control),
        .raw_prefill => try installed.compute.attentionMatchRawStructuralInventory(installed.storage, allocator, .{ .context_instance_id = installed.owner_identity, .sequence_id = installed.sequence, .logical_tokens = @intCast(installed.tokens.len), .position_offset = 0, .logical_blocks = installed.raw_logical_blocks, .owner_layers = installed.raw_owner_layers }, .{ .token_sha256 = installed.token_sha256, .shape_sha256 = installed.shape_authority_sha256, .binary_sha256 = installed.binary_sha256, .source_freeze_sha256 = installed.source_freeze_sha256 }, control),
    };
    defer inventory.deinit();
    var metadata = try metadata_mod.build(allocator, installed.identity, installed.tokens, installed.logits, installed.layer_donors, inventory, inventory.digest, @splat(1));
    defer metadata.deinit();

    const mappings = try allocator.alloc(provider_mod.Mapping, primary.len);
    defer allocator.free(mappings);
    var mapping_count: usize = 0;
    for (primary) |batch| {
        if (batch.consumer_layer >= metadata.metadata.layer_donors.len) return error.InvalidTrainingAugmentationBinding;
        const donor_index: usize = metadata.metadata.layer_donors[batch.consumer_layer];
        if (donor_index >= metadata.metadata.donors.len) return error.InvalidTrainingAugmentationBinding;
        const donor = metadata.metadata.donors[donor_index];
        if (donor.window != 0) continue;
        const query_heads = layout.query_heads_by_layer[batch.consumer_layer];
        if (query_heads == 0 or query_heads % donor.head_count != 0) return error.InvalidTrainingAugmentationBinding;
        const group = query_heads / donor.head_count;
        const head_index: usize = @as(usize, batch.query_head) / group;
        mappings[mapping_count] = .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .donor_index = donor_index, .donor_layer = donor.layer, .head_index = head_index, .scale = layout.scales_by_donor[donor_index], .softcap = layout.softcaps_by_donor[donor_index] };
        mapping_count += 1;
    }
    if (mapping_count == 0) return error.InvalidTrainingAugmentationBinding;
    var authority = try provider_mod.freezeAuthorityFromSource(allocator, head_source, .{
        .identity = installed.identity,
        .token_sha256 = installed.token_sha256,
        .binary_sha256 = installed.binary_sha256,
        .source_freeze_sha256 = installed.source_freeze_sha256,
        .shape_authority_sha256 = installed.shape_authority_sha256,
        .owner_identity = installed.owner_identity,
        .maximum_head_download_bytes = installed.maximum_head_download_bytes,
    }, mappings[0..mapping_count], control);
    defer authority.deinit();
    var provider = try provider_mod.Context.initSource(allocator, head_source, &authority);
    defer provider.deinit();
    var packed_source = try packed_mod.Source.init(&authority, provider.loader(), installed.tracker);
    defer packed_source.verifyNoLiveHead() catch @panic("packed head leaked");
    const complete_state = try mixed_mod.completeStateDigestFromSource(allocator, head_source, metadata.metadata, authority.state_sha256, metadata.inventory_sha256, installed.maximum_head_download_bytes, control);
    metadata.metadata.state_sha256 = complete_state;
    var mixed = mixed_mod.Context{ .allocator = allocator, .compute = installed.compute, .storage = installed.storage, .sequence = installed.sequence, .revision = revision, .metadata = &metadata.metadata, .packed_source = &packed_source, .maximum_head_download_bytes = installed.maximum_head_download_bytes, .head_source = head_source };
    defer mixed.deinit();
    var source = mixed.source(metadata.metadata.structure_sha256, complete_state, installed.constructor.bridge_binding.external_live_bytes);
    if (installed.constructor.source_export_write_fn) |write_fn| {
        const Sink = struct {
            context: *anyopaque,
            write_fn: *const fn (*anyopaque, []const u8) anyerror!void,
            pub fn writeAll(self: *@This(), bytes: []const u8) !void {
                return self.write_fn(self.context, bytes);
            }
        };
        const source_metadata = try cloneMetadata(allocator, source.metadata);
        defer freeMetadata(allocator, source_metadata);
        var adapter: source_checkpoint_provider.Context = .{ .source = &source };
        defer adapter.deinit();
        var sink: Sink = .{ .context = installed.constructor.source_export_context.?, .write_fn = write_fn };
        const seal = try checkpoint.exportEncodedFromProvider(source_metadata, adapter.provider(), checkpoint.serving_limits, .f16, &sink, control);
        installed.constructor.source_export_sha256_out.?.* = seal.file_sha256;
    }

    var descriptor_count: usize = 0;
    for (metadata.metadata.donors) |donor| descriptor_count = try std.math.add(usize, descriptor_count, donor.head_count);
    const descriptors = try allocator.alloc(metal_storage.CandidateHeadDescriptor, descriptor_count);
    defer allocator.free(descriptors);
    var descriptor_at: usize = 0;
    for (metadata.metadata.donors) |donor| for (0..donor.head_count) |head_index| {
        descriptors[descriptor_at] = .{ .layer = donor.layer, .head = @intCast(head_index), .window = donor.window };
        descriptor_at += 1;
    };
    const backend = try installed.compute.attentionMatchStorageBackend(installed.storage, allocator, checkpoint.serving_limits, installed.candidate_device_max_bytes);
    var builder = try metal_storage.CandidateBuilder.init(backend, metadata.metadata.logical_length, descriptors, metadata.metadata.structure_sha256, complete_state, installed.candidate_device_max_bytes, control);
    if (installed.test_installed_provider_probe) |probe| {
        probe.* = .{
            .inventory_entries = inventory.entries.len,
            .global_mappings = mapping_count,
            .donor_count = metadata.metadata.donors.len,
            .candidate_descriptors = descriptor_count,
            .revision = revision,
            .structure_sha256 = metadata.metadata.structure_sha256,
            .state_sha256 = complete_state,
        };
        builder.deinit();
        try source.verify_no_live_fn(source.context);
        return error.InstalledProviderFixtureComplete;
    }
    return prepare(allocator, &source, layout, primary, dense, future, fit, fp64_validation, &builder, installed.constructor);
}

test "constructor rejects unsealed source before provider open" {
    var opens: usize = 0;
    const Probe = struct {
        fn open(raw: ?*anyopaque, _: usize, _: usize, _: solver.Control) !engine.StreamingHeadLease {
            const count: *usize = @ptrCast(@alignCast(raw orelse return error.TestUnexpectedResult));
            count.* += 1;
            return error.TestUnexpectedResult;
        }
        fn verify(_: ?*anyopaque) !void {}
    };
    var source: engine.StreamingHeadSource = .{
        .metadata = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 32512, .tokens = &.{}, .logits = &.{}, .layer_donors = &.{}, .structure_sha256 = @splat(5), .state_sha256 = @splat(6), .source_census_physical_bytes = 1 },
        .expected_structure_sha256 = @splat(7),
        .expected_state_sha256 = @splat(6),
        .context = &opens,
        .open_fn = Probe.open,
        .verify_no_live_fn = Probe.verify,
        .external_live_bytes = 1,
    };
    const invalid: Binding = .{ .bridge_binding = undefined, .engine_options = .{}, .candidate_device_max_bytes = 1, .retained_query_bytes = 0, .maximum_current_head_bytes = 1 };
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, invalid.validate(&source));
    try std.testing.expectEqual(@as(usize, 0), opens);
}

test "device result metadata owns request slices before request teardown" {
    var tokens = [_]u32{ 1, 2, 3 };
    var logits = [_]f32{ 4, 5 };
    var layers = [_]u32{0};
    var donors = [_]engine.SourceDonorMetadata{.{ .layer = 0, .window = 0, .head_count = 1, .head_dim = 2, .rows = 3, .physical_bytes = 4096 }};
    const source: engine.SourceMetadata = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 3, .tokens = &tokens, .logits = &logits, .layer_donors = &layers, .donors = &donors, .structure_sha256 = @splat(5), .state_sha256 = @splat(6), .source_census_physical_bytes = 4096 };
    const owned = try cloneMetadata(std.testing.allocator, source);
    defer freeMetadata(std.testing.allocator, owned);
    tokens[0] = 99;
    logits[0] = 99;
    layers[0] = 99;
    donors[0].head_count = 99;
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, owned.tokens);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, owned.logits);
    try std.testing.expectEqual(@as(u32, 0), owned.layer_donors[0]);
    try std.testing.expectEqual(@as(usize, 1), owned.donors[0].head_count);
    try std.testing.expectError(error.OutOfMemory, cloneMetadata(std.testing.failing_allocator, source));
}
