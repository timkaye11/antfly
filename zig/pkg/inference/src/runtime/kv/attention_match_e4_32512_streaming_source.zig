// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const engine = @import("attention_match_engine.zig");
const packed_source = @import("attention_match_e4_32512_packed_head_source.zig");
const storage_runtime = @import("storage_runtime.zig");
const solver = @import("attention_match.zig");
const metal_compute = @import("../../ops/metal_compute.zig");
const metal_storage = @import("../../ops/metal_attention_match_storage.zig");
const provider = @import("attention_match_e4_32512_metal_head_provider.zig");

const Active = union(enum) { packed_head: packed_source.LoadedHead, local: metal_storage.DownloadedHead };

pub const Context = struct {
    allocator: std.mem.Allocator,
    compute: *metal_compute.MetalCompute,
    storage: *storage_runtime.KvStorageRuntime,
    sequence: storage_runtime.SequenceId,
    revision: u64,
    metadata: *const engine.SourceMetadata,
    packed_source: *packed_source.Source,
    maximum_head_download_bytes: usize,
    head_source: provider.Source,
    active: ?Active = null,

    pub fn source(self: *@This(), expected_structure_sha256: [32]u8, expected_state_sha256: [32]u8, external_live_bytes: usize) engine.StreamingHeadSource {
        return .{ .metadata = self.metadata.*, .expected_structure_sha256 = expected_structure_sha256, .expected_state_sha256 = expected_state_sha256, .context = self, .open_fn = open, .verify_no_live_fn = verifyNoLive, .external_live_bytes = external_live_bytes };
    }

    pub fn deinit(self: *@This()) void {
        if (self.active != null) @panic("streaming source destroyed with live head");
        self.* = undefined;
    }

    fn open(raw: ?*anyopaque, donor: usize, head_index: usize, control: solver.Control) !engine.StreamingHeadLease {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        if (self.active != null or donor >= self.metadata.donors.len or head_index >= self.metadata.donors[donor].head_count)
            return error.InvalidTrainingAugmentationBinding;
        const descriptor = self.metadata.donors[donor];
        if (descriptor.window == 0) {
            var match: ?packed_source.Entry = null;
            for (self.packed_source.authority.entries) |entry| if (entry.donor_index == donor and entry.head_index == head_index) {
                if (match != null and (match.?.consumer_layer != entry.consumer_layer or match.?.query_head != entry.query_head)) continue;
                match = entry;
            };
            const entry = match orelse return error.InvalidTrainingAugmentationBinding;
            self.active = .{ .packed_head = try self.packed_source.loadHead(entry.consumer_layer, entry.query_head, 0, control) };
        } else {
            self.active = .{ .local = try self.head_source.download(self.allocator, descriptor.layer, head_index, descriptor.window, self.maximum_head_download_bytes, control) };
        }
        const head = switch (self.active.?) {
            .packed_head => |*value| value.head(),
            .local => |*value| value.head(),
        };
        return .{ .context = self, .close_fn = close, .donor_index = donor, .head_index = head_index, .head = head };
    }

    fn close(raw: ?*anyopaque, _: *engine.StreamingHeadLease) void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse @panic("missing streaming source")));
        if (self.active) |*active| switch (active.*) {
            .packed_head => |*value| value.deinit(),
            .local => |*value| value.deinit(),
        } else @panic("streaming source close without live head");
        self.active = null;
    }

    fn verifyNoLive(raw: ?*anyopaque) !void {
        const self: *@This() = @ptrCast(@alignCast(raw orelse return error.InvalidTrainingAugmentationBinding));
        if (self.active != null) return error.IncompleteContextAllocationScope;
        try self.packed_source.verifyNoLiveHead();
        if (try self.compute.attentionMatchRevision(self.storage) != self.revision) return error.AttentionMatchSourceChanged;
    }
};

/// Binds packed global payload authority and every exact local-SWA payload to
/// the installed structural inventory. Downloads are sequential and released
/// immediately; this never creates a whole checkpoint snapshot.
pub fn completeStateDigest(
    allocator: std.mem.Allocator,
    compute: *metal_compute.MetalCompute,
    storage: *storage_runtime.KvStorageRuntime,
    sequence: storage_runtime.SequenceId,
    revision: u64,
    metadata: engine.SourceMetadata,
    packed_state_sha256: [32]u8,
    inventory_sha256: [32]u8,
    maximum_head_download_bytes: usize,
    control: solver.Control,
) ![32]u8 {
    return completeStateDigestFromSource(allocator, .{ .installed_candidate = .{ .compute = compute, .storage = storage, .sequence = sequence, .revision = revision } }, metadata, packed_state_sha256, inventory_sha256, maximum_head_download_bytes, control);
}

pub fn completeStateDigestFromSource(
    allocator: std.mem.Allocator,
    source: provider.Source,
    metadata: engine.SourceMetadata,
    packed_state_sha256: [32]u8,
    inventory_sha256: [32]u8,
    maximum_head_download_bytes: usize,
    control: solver.Control,
) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e4-32512-complete-global-local-state.v1\x00");
    hash.update(&metadata.structure_sha256);
    hash.update(&packed_state_sha256);
    hash.update(&inventory_sha256);
    for (metadata.donors, 0..) |donor, donor_index| if (donor.window != 0) for (0..donor.head_count) |head_index| {
        var downloaded = try source.download(allocator, donor.layer, head_index, donor.window, maximum_head_download_bytes, control);
        defer downloaded.deinit();
        const entry = try packed_source.censusEntry(@intCast(donor.layer), @intCast(head_index), donor_index, donor.layer, head_index, 1, 0, source.revision(), downloaded.head());
        hash.update(std.mem.asBytes(&donor_index));
        hash.update(std.mem.asBytes(&head_index));
        hash.update(&entry.payload_sha256);
    };
    try control.check();
    return hash.finalResult();
}
