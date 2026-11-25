// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Concrete Metal provider for the authenticated E4 32512 packed head source.
//! It downloads and releases one global source head at a time.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const packed_source = @import("attention_match_e4_32512_packed_head_source.zig");
const storage_runtime = @import("storage_runtime.zig");
const solver = @import("attention_match.zig");
const metal_compute = @import("../../ops/metal_compute.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");
const block = @import("block.zig");

pub const SourceKind = enum(u8) { raw_prefill, installed_candidate };

pub const Source = union(SourceKind) {
    raw_prefill: struct {
        compute: *metal_compute.MetalCompute,
        storage: *storage_runtime.KvStorageRuntime,
        sequence: storage_runtime.SequenceId,
        revision: u64,
        logical_tokens: u32,
        logical_blocks: []const block.KvBlockId,
    },
    installed_candidate: struct {
        compute: *metal_compute.MetalCompute,
        storage: *storage_runtime.KvStorageRuntime,
        sequence: storage_runtime.SequenceId,
        revision: u64,
    },

    pub fn kind(self: @This()) SourceKind {
        return std.meta.activeTag(self);
    }
    pub fn revision(self: @This()) u64 {
        return switch (self) {
            inline else => |value| value.revision,
        };
    }

    pub fn download(self: @This(), allocator: std.mem.Allocator, layer: u32, head: usize, window: u32, max_bytes: usize, control: solver.Control) !metal_storage.DownloadedHead {
        return switch (self) {
            .installed_candidate => |value| value.compute.attentionMatchDownloadHead(value.storage, value.sequence, allocator, layer, head, window, value.revision, max_bytes, control),
            .raw_prefill => |value| value.compute.attentionMatchDownloadRawHead(value.storage, allocator, .{ .sequence_id = value.sequence, .source_revision = value.revision, .logical_tokens = value.logical_tokens, .logical_blocks = value.logical_blocks, .donor_layer = layer, .head_index = @intCast(head), .window = window }, max_bytes, control),
        };
    }
};

pub const Mapping = struct {
    consumer_layer: u32,
    query_head: u32,
    donor_index: usize,
    donor_layer: u32,
    head_index: usize,
    scale: f64,
    softcap: f64,
};

pub const ProducerBinding = struct {
    identity: checkpoint.Identity,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    owner_identity: u64,
    maximum_head_download_bytes: usize,
};

/// Build the census from the actual installed DeviceState. Each head is
/// downloaded, hashed, and released before the next mapping is visited.
pub fn freezeAuthorityFromMetal(
    allocator: std.mem.Allocator,
    compute: *metal_compute.MetalCompute,
    storage: *storage_runtime.KvStorageRuntime,
    sequence: storage_runtime.SequenceId,
    binding: ProducerBinding,
    mappings: []const Mapping,
    control: solver.Control,
) !packed_source.AuthorityOwned {
    return freezeAuthorityFromSource(allocator, .{ .installed_candidate = .{ .compute = compute, .storage = storage, .sequence = sequence, .revision = try compute.attentionMatchRevision(storage) } }, binding, mappings, control);
}

pub fn freezeAuthorityFromSource(
    allocator: std.mem.Allocator,
    source: Source,
    binding: ProducerBinding,
    mappings: []const Mapping,
    control: solver.Control,
) !packed_source.AuthorityOwned {
    if (mappings.len == 0 or binding.maximum_head_download_bytes == 0) return error.InvalidTrainingAugmentationBinding;
    const revision = source.revision();
    if (revision == 0) return error.InvalidTrainingAugmentationBinding;
    const entries = try allocator.alloc(packed_source.Entry, mappings.len);
    defer allocator.free(entries);
    var observed_maximum: usize = 0;
    var unique_count: usize = 0;
    for (mappings, entries, 0..) |mapping, *entry, index| {
        try control.check();
        var shared: ?packed_source.Entry = null;
        for (mappings[0..index], entries[0..index]) |prior_mapping, prior_entry| {
            if (prior_mapping.donor_index != mapping.donor_index or prior_mapping.head_index != mapping.head_index) continue;
            if (prior_mapping.donor_layer != mapping.donor_layer or
                @as(u64, @bitCast(prior_mapping.scale)) != @as(u64, @bitCast(mapping.scale)) or
                @as(u64, @bitCast(prior_mapping.softcap)) != @as(u64, @bitCast(mapping.softcap)))
                return error.InvalidTrainingAugmentationBinding;
            shared = prior_entry;
            break;
        }
        if (shared) |prior| {
            entry.* = prior;
            entry.consumer_layer = mapping.consumer_layer;
            entry.query_head = mapping.query_head;
            continue;
        }
        // The 32512 authority is global/all-history. Local-window heads have a
        // different row geometry and are rejected by DeviceState validation.
        entry.* = blk: {
            var downloaded = try source.download(allocator, mapping.donor_layer, mapping.head_index, 0, binding.maximum_head_download_bytes, control);
            defer downloaded.deinit();
            observed_maximum = @max(observed_maximum, try downloaded.liveBytes());
            break :blk try packed_source.censusEntry(mapping.consumer_layer, mapping.query_head, mapping.donor_index, mapping.donor_layer, mapping.head_index, mapping.scale, mapping.softcap, revision, downloaded.head());
        };
        unique_count += 1;
    }
    if (observed_maximum == 0 or observed_maximum > binding.maximum_head_download_bytes)
        return error.AttentionMatchSourceChanged;
    var state_hash = std.crypto.hash.sha2.Sha256.init(.{});
    state_hash.update("antfly.e4-32512-metal-unique-head-state.v1");
    var source_kind_frame: [4]u8 = undefined;
    std.mem.writeInt(u32, &source_kind_frame, @backingInt(source.kind()), .little);
    state_hash.update(&source_kind_frame);
    var field_frame: [8]u8 = undefined;
    std.mem.writeInt(u64, &field_frame, revision, .little);
    state_hash.update(&field_frame);
    std.mem.writeInt(u64, &field_frame, @intCast(unique_count), .little);
    state_hash.update(&field_frame);
    for (entries, 0..) |entry, index| {
        var first = true;
        for (entries[0..index]) |prior| if (prior.donor_index == entry.donor_index and prior.head_index == entry.head_index) {
            first = false;
            break;
        };
        if (!first) continue;
        inline for (.{ entry.donor_index, entry.donor_layer, entry.head_index, entry.rows, entry.dim }) |field| {
            std.mem.writeInt(u64, &field_frame, @intCast(field), .little);
            state_hash.update(&field_frame);
        }
        state_hash.update(&entry.payload_sha256);
    }
    const state_sha256 = state_hash.finalResult();
    return packed_source.freezeAuthority(allocator, binding.identity, binding.token_sha256, binding.binary_sha256, binding.source_freeze_sha256, binding.shape_authority_sha256, state_sha256, binding.owner_identity, binding.maximum_head_download_bytes, entries);
}

pub const Context = struct {
    allocator: std.mem.Allocator,
    compute: *metal_compute.MetalCompute,
    storage: *storage_runtime.KvStorageRuntime,
    sequence: storage_runtime.SequenceId,
    authority: *const packed_source.AuthorityOwned,
    downloaded: ?metal_storage.DownloadedHead = null,
    source: ?Source = null,

    pub fn init(allocator: std.mem.Allocator, compute: *metal_compute.MetalCompute, storage: *storage_runtime.KvStorageRuntime, sequence: storage_runtime.SequenceId, authority: *const packed_source.AuthorityOwned) !Context {
        if (authority.entries.len == 0 or authority.maximum_loader_live_bytes == 0 or
            try compute.attentionMatchRevision(storage) != authority.entries[0].source_revision)
            return error.InvalidTrainingAugmentationBinding;
        return .{ .allocator = allocator, .compute = compute, .storage = storage, .sequence = sequence, .authority = authority, .source = .{ .installed_candidate = .{ .compute = compute, .storage = storage, .sequence = sequence, .revision = authority.entries[0].source_revision } } };
    }

    pub fn initSource(allocator: std.mem.Allocator, source: Source, authority: *const packed_source.AuthorityOwned) !Context {
        if (authority.entries.len == 0 or authority.maximum_loader_live_bytes == 0 or source.revision() != authority.entries[0].source_revision)
            return error.InvalidTrainingAugmentationBinding;
        return switch (source) {
            .raw_prefill => |value| .{ .allocator = allocator, .compute = value.compute, .storage = value.storage, .sequence = value.sequence, .authority = authority, .source = source },
            .installed_candidate => |value| .{ .allocator = allocator, .compute = value.compute, .storage = value.storage, .sequence = value.sequence, .authority = authority, .source = source },
        };
    }

    pub fn loader(self: *Context) packed_source.Loader {
        return .{ .context = self, .open_fn = open, .close_fn = close, .maximum_loader_live_bytes = self.authority.maximum_loader_live_bytes, .head_allocation_copies = 2 };
    }

    pub fn deinit(self: *Context) void {
        if (self.downloaded != null) @panic("Metal packed head provider destroyed with a borrowed head");
        self.* = undefined;
    }

    fn open(raw: ?*anyopaque, entry: packed_source.Entry, control: solver.Control) !packed_source.BorrowedHead {
        const self: *Context = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        if (self.downloaded != null) return error.PackedHeadAlreadyLive;
        var downloaded = try (self.source orelse return error.InvalidTrainingAugmentationBinding).download(self.allocator, entry.donor_layer, entry.head_index, 0, self.authority.maximum_loader_live_bytes, control);
        errdefer downloaded.deinit();
        const live = try downloaded.liveBytes();
        if (live > self.authority.maximum_loader_live_bytes) return error.CompactionWorkspaceLimit;
        self.downloaded = downloaded;
        return .{ .entry = entry, .identity = self.authority.identity, .token_sha256 = self.authority.token_sha256, .binary_sha256 = self.authority.binary_sha256, .source_freeze_sha256 = self.authority.source_freeze_sha256, .shape_authority_sha256 = self.authority.shape_authority_sha256, .state_sha256 = self.authority.state_sha256, .census_sha256 = self.authority.census_sha256, .head = self.downloaded.?.head(), .loader_live_bytes = live };
    }

    fn close(raw: ?*anyopaque, _: *packed_source.BorrowedHead) void {
        const self: *Context = @ptrCast(@alignCast(raw orelse @panic("missing Metal packed head provider")));
        if (self.downloaded) |*downloaded| downloaded.deinit() else @panic("Metal packed head provider close without open");
        self.downloaded = null;
    }
};
