// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Device-resident KV storage for the Metal backend (Phase 6 / option B).
//!
//! `MetalKvStorage` implements the `KvStorageRuntime.DeviceWriteHook` so that
//! `writeLayerKvSuffix` on device-resident k/v tensors goes straight to the
//! Metal encode kernel — no host materialization, no host KvPool shadow on the
//! hot path. Gather still works by downloading the encoded span bytes from
//! the device slot buffers and decoding them on the host using the existing
//! turboquant routines.
//!
//! **Current scope**:
//! - KV dtypes supported: polar4, turbo3 (compressed encode), f32 (raw
//!   device→device blit), f16 (device f32→half convert kernel), and int8
//!   per-head (device quantize with threadgroup tree reduction over
//!   head_dim). int4, fp8, bf16 still fall through to the host path.
//! - Full-history paged KV uses one device page pool per layer. Physical page
//!   offsets are the storage runtime's reference-counted block ids, so cached
//!   prefixes remain resident after the request that populated them exits.
//!   Sliding-window rings remain sequence-owned because their cyclic physical
//!   offsets intentionally alias across logical pages. Capacity-exhaustion
//!   returns `error.DeviceWriteFallback` so callers downgrade gracefully.
//! - Caller is responsible for keeping the host KvPool in sync if any path
//!   still reads from it. In the full Phase 6 rollout `ensurePagedKvSuffixWritten`
//!   calls the device path first and skips the host write when the hook
//!   accepts; gather-driven paths download from the slot buffer instead.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const block = @import("../runtime/kv/block.zig");
const storage_runtime = @import("../runtime/kv/storage_runtime.zig");
const selection = @import("../runtime/kv/selection.zig");
const pool_mod = @import("../runtime/kv/pool.zig");
const turboquant = @import("../runtime/kv/turboquant.zig");
const metal_runtime = @import("metal_runtime.zig");
const owner_alloc = @import("../runtime/kv/host_allocation_tracker.zig");
const context_teacher_forced = @import("context_teacher_forced.zig");
const attention_match_storage = @import("../ops/metal_attention_match_storage.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
const AttentionMatchControl = @import("../runtime/kv/attention_match.zig").Control;

fn fingerprintU32(hash: *std.crypto.hash.sha2.Sha256, value: usize) !void {
    const narrowed = std.math.cast(u32, value) orelse return error.QualificationFingerprintOverflow;
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, narrowed, .little);
    hash.update(&bytes);
}

fn fingerprintQualificationCoverage(
    hash: *std.crypto.hash.sha2.Sha256,
    logical_start: usize,
    logical_end: usize,
    page_size: usize,
) !void {
    if (page_size == 0 or logical_start >= logical_end) return error.InvalidQualificationFingerprint;
    const first_page = logical_start / page_size;
    const last_page = (logical_end - 1) / page_size;
    try fingerprintU32(hash, last_page - first_page + 1);
    var logical_page = first_page;
    while (logical_page <= last_page) : (logical_page += 1) {
        const page_first = @max(logical_start, logical_page * page_size);
        const page_end = @min(logical_end, (logical_page + 1) * page_size);
        try fingerprintU32(hash, logical_page);
        try fingerprintU32(hash, page_first);
        try fingerprintU32(hash, page_end - page_first);
    }
}

const QualificationVisibleRange = struct { start: usize, count: usize };

fn validateQualificationOwnerLayers(owner_layers: []const u32, expected_count: usize) !void {
    if (owner_layers.len == 0 or owner_layers.len != expected_count)
        return error.InvalidQualificationFingerprint;
    var previous: ?u32 = null;
    for (owner_layers) |layer| {
        if (previous) |value| if (layer <= value)
            return error.InvalidQualificationFingerprint;
        previous = layer;
    }
}

fn qualificationVisibleRange(binding: SlotBinding, logical_tokens: usize, page_size: usize) !QualificationVisibleRange {
    if (logical_tokens == 0 or page_size == 0) return error.InvalidQualificationFingerprint;
    const count = if (binding.ring_page_count > 0) blk: {
        if (!binding.sequence_owned or binding.sliding_window_tokens == 0) return error.InvalidQualificationFingerprint;
        const ring_capacity = std.math.mul(usize, binding.ring_page_count, page_size) catch
            return error.QualificationFingerprintOverflow;
        if (binding.sliding_window_tokens > ring_capacity) return error.InvalidQualificationFingerprint;
        break :blk @min(logical_tokens, binding.sliding_window_tokens);
    } else blk: {
        if (binding.sliding_window_tokens != 0) return error.InvalidQualificationFingerprint;
        break :blk logical_tokens;
    };
    return .{ .start = logical_tokens - count, .count = count };
}

fn fingerprintQualificationLayer(
    hash: *std.crypto.hash.sha2.Sha256,
    layer: usize,
    binding: SlotBinding,
    logical_start: usize,
    visible_tokens: usize,
    logical_tokens: usize,
    page_size: usize,
    key_row_bytes: usize,
    value_row_stride: usize,
    value_element_bytes: usize,
) !void {
    // Deliberately omit slot id, physical pages, buffer capacities, and pool
    // capacity. Independent sibling contexts may choose different allocation
    // geometry while implementing identical logical context semantics.
    try fingerprintU32(hash, layer);
    try fingerprintU32(hash, if (binding.sequence_owned) 1 else 0);
    try fingerprintU32(hash, binding.ring_page_count);
    try fingerprintU32(hash, binding.sliding_window_tokens);
    try fingerprintU32(hash, binding.written_tokens);
    try fingerprintU32(hash, binding.position_offset);
    try fingerprintU32(hash, logical_start);
    try fingerprintU32(hash, visible_tokens);
    try fingerprintU32(hash, key_row_bytes);
    try fingerprintU32(hash, value_row_stride);
    try fingerprintU32(hash, value_element_bytes);
    try fingerprintQualificationCoverage(hash, logical_start, logical_tokens, page_size);
}

/// Matches the `format` values accepted by
/// `termite_metal_decode_runtime_update_attention_span_from_f32_key_device_slot`.
/// Compressed formats run the polar4/turbo3 encode pipeline; `raw_f32` skips
/// the pipeline and copies device→device; `f16` dispatches the f32→half
/// converter kernel for both keys and values; `int8_per_head` quantizes keys
/// into a (f32_scale + int8[head_dim]) per-head layout via the threadgroup
/// reduction kernel while values stay raw f32 (parallel to polar4/turbo3).
/// Other KV dtypes (int4, bf16) still fall through to the host write path.
pub const KeyFormat = enum(u32) {
    polar4 = 0,
    turbo3 = 1,
    raw_f32 = 2,
    f16 = 3,
    int8_per_head = 4,

    pub fn fromKvDType(dtype: pool_mod.KvDType) ?KeyFormat {
        return switch (dtype) {
            .polar4 => .polar4,
            .turbo3 => .turbo3,
            .f32 => .raw_f32,
            .f16 => .f16,
            .int8 => .int8_per_head,
            else => null,
        };
    }

    pub fn isCompressed(self: KeyFormat) bool {
        return switch (self) {
            .polar4, .turbo3, .int8_per_head => true,
            .raw_f32, .f16 => false,
        };
    }
};

/// Retained for external callers that imported the old name.
pub const CompressedKeyFormat = KeyFormat;

/// Key for the (sequence, layer) → slot mapping. Layer index fits in u32 —
/// even 80-layer models (Llama-3 405B) stay far below u32.
const SlotKey = struct {
    sequence_id: storage_runtime.SequenceId,
    layer_index: u32,
};

/// Per-sequence view into a backing slot. Full-history bindings for the same
/// layer share a slot but keep sequence-local layout and position metadata.
/// Ring bindings own a dedicated slot and are reclaimed with the sequence.
const SlotBinding = struct {
    slot: usize,
    ring_page_count: usize = 0,
    /// Exact logical attention window. Ring capacity is page rounded and also
    /// includes admitted in-flight rows, so it is not a visibility bound.
    sliding_window_tokens: usize = 0,
    sequence_owned: bool = false,
    compactable_global: bool = false,
    logical_contiguous: bool = false,
    physical_base_tokens: usize = 0,
    written_tokens: usize = 0,
    position_offset: usize = 0,
    selected_positions: ?[]u32 = null,

    fn ownsSlot(self: SlotBinding) bool {
        return self.sequence_owned;
    }

    fn covers(self: SlotBinding, token_count: usize) bool {
        return self.written_tokens >= token_count;
    }

    fn coversBeforePendingSuffix(self: SlotBinding, token_count: usize, pending_suffix_token_count: usize) bool {
        if (pending_suffix_token_count > token_count) return false;
        return self.covers(token_count - pending_suffix_token_count);
    }

    fn commitWrite(self: *SlotBinding, total_token_count: usize, position_offset: usize) bool {
        if (total_token_count < self.written_tokens or position_offset < self.position_offset) return false;
        self.written_tokens = total_token_count;
        self.position_offset = position_offset;
        return true;
    }

    fn truncateTo(self: *SlotBinding, retained_token_count: usize) void {
        self.written_tokens = @min(self.written_tokens, retained_token_count);
    }
};

const ExactExportSourceSnapshot = struct {
    slot: usize,
    written_tokens: usize,
    position_offset: usize,
    key_handle: *anyopaque,
    value_handle: *anyopaque,
    key_capacity: usize,
    value_capacity: usize,
    key_row_bytes: usize,
    value_row_stride: usize,
    info_tokens: usize,
    info_position_offset: usize,

    fn matches(self: @This(), binding: SlotBinding, info: MetalKvStorage.SlotInfo) bool {
        return binding.slot == self.slot and binding.selected_positions == null and binding.physical_base_tokens == 0 and
            binding.ring_page_count == 0 and binding.logical_contiguous and binding.written_tokens == self.written_tokens and
            binding.position_offset == self.position_offset and info.encoded_key_handle != null and info.v_handle != null and
            info.encoded_key_handle.? == self.key_handle and info.v_handle.? == self.value_handle and
            info.encoded_key_capacity == self.key_capacity and info.v_capacity == self.value_capacity and
            info.key_row_bytes == self.key_row_bytes and info.v_row_stride == self.value_row_stride and
            info.tokens == self.info_tokens and info.position_offset == self.info_position_offset;
    }
};

pub const AttentionMatchBinding = struct {
    diagnostic_only: bool = false,
    sequence: storage_runtime.SequenceId,
    device: attention_match_storage.DeviceState,
    control: AttentionMatchControl,
    invalid: bool = false,
    attention_calls: usize = 0,
    donor_appends: usize = 0,
    rollback_tokens: usize = 0,

    pub fn deinit(self: *AttentionMatchBinding) void {
        self.device.runner.cancelDeferredFrame();
        std.log.info("{s} source_tokens={d} logical_tokens={d} attention_calls={d} donor_appends={d} allocated_device_bytes={d} invalid={d} rollback_tokens={d}", .{
            if (self.diagnostic_only) @as([]const u8, "attention_match_shadow_serving") else "attention_match_serving",
            self.device.initial_logical_length,
            self.device.logical_length,
            self.attention_calls,
            self.donor_appends,
            self.device.allocated_bytes,
            @intFromBool(self.invalid),
            self.rollback_tokens,
        });
        self.device.deinit();
    }
};

pub const AttentionMatchLocalCapacityPolicy = enum { legacy_spare512_v1, exact_window_causal_union_v1 };
const AttentionMatchLocalCapacityState = struct {
    policy: AttentionMatchLocalCapacityPolicy = .legacy_spare512_v1,
    locked: bool = false,
    fn configure(self: *@This(), requested: AttentionMatchLocalCapacityPolicy) !void {
        if (self.locked and self.policy != requested) return error.AttentionMatchLocalCapacityPolicyChanged;
        self.policy = requested;
    }
    fn lock(self: *@This()) attention_match_storage.RunnerOptions {
        self.locked = true;
        return .{ .exact_local_window_capacity = self.policy == .exact_window_causal_union_v1 };
    }
};

pub const MetalKvStorage = struct {
    allocator: std.mem.Allocator,
    runtime: *metal_runtime.RawMetalDecodeRuntime,
    format: KeyFormat,
    num_kv_heads: u32,
    head_dim: u32,
    page_size_tokens: u16,
    /// (seq, layer) → sequence-local view of a device slot.
    slot_map: std.AutoHashMapUnmanaged(SlotKey, SlotBinding) = .empty,
    /// Stable layer → slot mapping for full-history paged KV. These slots are
    /// never reset at a request boundary: retained block ids, rather than a
    /// transient sequence id, own the bytes stored at each physical page.
    global_layer_slots: std.AutoHashMapUnmanaged(u32, usize) = .empty,
    /// Physical slots are leased by the shared raw runtime rather than handed
    /// out independently by each hook. A prompt-cache hook can persist across
    /// requests while request-local hooks use the same runtime, so local bump
    /// allocators would alias and corrupt retained KV pages.
    leased_slots: [metal_runtime.attention_span_slot_capacity]bool = @as([metal_runtime.attention_span_slot_capacity]bool, @splat(false)),
    slot_ring_page_count: [metal_runtime.attention_span_slot_capacity]usize = @as([metal_runtime.attention_span_slot_capacity]usize, @splat(0)),
    slot_ring_policy_initialized: [metal_runtime.attention_span_slot_capacity]bool = @as([metal_runtime.attention_span_slot_capacity]bool, @splat(false)),
    slot_buffer_capacity_tokens: [metal_runtime.attention_span_slot_capacity]usize = @as([metal_runtime.attention_span_slot_capacity]usize, @splat(0)),
    allocation_peak_bytes: u64 = 0,
    /// Non-null only for a request-exclusive hook. Shared prompt-cache hooks
    /// leave this null and disclose their retained residency separately.
    owner_allocation_tracker: ?*owner_alloc.HostAllocationTracker = null,
    slot_accounted_live_bytes: [metal_runtime.attention_span_slot_capacity]usize = @splat(0),
    last_owner_transition: if (builtin.is_test) struct { slot: usize = 0, prior: usize = 0, newly_allocated: usize = 0, final: usize = 0 } else void = if (builtin.is_test) .{} else {},
    cyclic_page_table_cache: storage_runtime.CyclicPageTableCache = .{},
    attention_match_runner: ?attention_match_storage.Runner = null,
    attention_match_local_capacity: AttentionMatchLocalCapacityState = .{},
    attention_match_binding: ?*AttentionMatchBinding = null,
    attention_match_revision: u64 = 0,

    /// Allocate a MetalKvStorage keyed to `runtime`. The storage does not own
    /// the runtime — callers are responsible for its lifetime. `dtype` must
    /// resolve to a compressed key format; other formats are unsupported by
    /// this fast path (caller falls back to the host write path).
    pub fn create(
        allocator: std.mem.Allocator,
        runtime: *metal_runtime.RawMetalDecodeRuntime,
        dtype: pool_mod.KvDType,
        num_kv_heads: u32,
        head_dim: u32,
        page_size_tokens: u16,
    ) !*MetalKvStorage {
        return createTracked(allocator, runtime, dtype, num_kv_heads, head_dim, page_size_tokens, null);
    }

    pub fn createTracked(
        allocator: std.mem.Allocator,
        runtime: *metal_runtime.RawMetalDecodeRuntime,
        dtype: pool_mod.KvDType,
        num_kv_heads: u32,
        head_dim: u32,
        page_size_tokens: u16,
        owner_allocation_tracker: ?*owner_alloc.HostAllocationTracker,
    ) !*MetalKvStorage {
        const format = KeyFormat.fromKvDType(dtype) orelse return error.DeviceWriteFormatUnsupported;
        const self = try allocator.create(MetalKvStorage);
        self.* = .{
            .allocator = allocator,
            .runtime = runtime,
            .format = format,
            .num_kv_heads = num_kv_heads,
            .head_dim = head_dim,
            .page_size_tokens = page_size_tokens,
            .owner_allocation_tracker = owner_allocation_tracker,
        };
        return self;
    }

    fn ownerBindingEmpty(self: *const MetalKvStorage) bool {
        if (metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime)) return false;
        if (self.slot_map.count() != 0 or self.global_layer_slots.count() != 0 or
            self.attention_match_binding != null or self.attention_match_runner != null or
            self.cyclic_page_table_cache.offsets.items.len != 0)
            return false;
        for (self.leased_slots) |leased| if (leased) return false;
        for (self.slot_accounted_live_bytes) |bytes| if (bytes != 0) return false;
        for (self.slot_ring_page_count) |pages| if (pages != 0) return false;
        for (self.slot_ring_policy_initialized) |initialized| if (initialized) return false;
        for (self.slot_buffer_capacity_tokens) |capacity| if (capacity != 0) return false;
        return true;
    }

    fn prepareRequestOwnerDetach(self: *MetalKvStorage) !void {
        if (metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime) or
            self.slot_map.count() != 0 or self.attention_match_binding != null)
            return error.ContextFrameOwnershipMismatch;
        self.global_layer_slots.clearRetainingCapacity();
        for (self.leased_slots, 0..) |leased, slot| if (leased) try self.discardPrivateSlotRequired(slot);
        self.cyclic_page_table_cache.deinit(self.allocator);
        if (self.attention_match_runner) |*runner| runner.deinit();
        self.attention_match_runner = null;
        if (!self.ownerBindingEmpty()) return error.ContextFrameOwnershipMismatch;
    }

    pub fn bindRequestOwnerTracker(self: *MetalKvStorage, tracker: *owner_alloc.HostAllocationTracker) !void {
        if (self.owner_allocation_tracker != null or !self.ownerBindingEmpty())
            return error.ContextFrameOwnershipMismatch;
        self.owner_allocation_tracker = tracker;
    }

    pub fn unbindRequestOwnerTracker(self: *MetalKvStorage, tracker: *owner_alloc.HostAllocationTracker) !void {
        if (self.owner_allocation_tracker != tracker)
            return error.ContextFrameOwnershipMismatch;
        try self.prepareRequestOwnerDetach();
        self.owner_allocation_tracker = null;
    }

    fn allocateSlot(self: *MetalKvStorage) !usize {
        var slot: usize = undefined;
        if (metal_runtime.termite_metal_decode_runtime_acquire_paged_kv_slot(self.runtime, &slot) != 0)
            return error.DeviceWriteFallback;
        if (slot >= self.leased_slots.len) {
            _ = metal_runtime.termite_metal_decode_runtime_release_paged_kv_slot(self.runtime, slot);
            return error.DeviceWriteFallback;
        }
        self.leased_slots[slot] = true;
        errdefer self.reclaimSlot(slot);
        self.slot_ring_page_count[slot] = 0;
        self.slot_ring_policy_initialized[slot] = false;
        self.slot_buffer_capacity_tokens[slot] = 0;
        // A released raw-runtime slot may retain model-owned cached buffers.
        // Once leased, their actual allocatedSize is request-exclusive for
        // this hook's lifetime and belongs in the request ledger.
        try self.reconcileSlotAccounting(slot, 0);
        return slot;
    }

    fn reclaimSlot(self: *MetalKvStorage, slot: usize) void {
        if (slot >= self.leased_slots.len or !self.leased_slots[slot]) return;
        if (metal_runtime.termite_metal_decode_runtime_release_paged_kv_slot(self.runtime, slot) != 0)
            @panic("request-exclusive Metal KV slot released before backend drain");
        self.reconcileSlotAccounting(slot, 0) catch
            @panic("request-exclusive Metal KV release accounting failed");
        self.leased_slots[slot] = false;
        self.slot_ring_page_count[slot] = 0;
        self.slot_ring_policy_initialized[slot] = false;
        self.slot_buffer_capacity_tokens[slot] = 0;
    }

    fn discardPrivateSlot(self: *MetalKvStorage, slot: usize) void {
        if (slot >= self.leased_slots.len or !self.leased_slots[slot]) return;
        if (metal_runtime.termite_metal_decode_runtime_discard_paged_kv_slot(self.runtime, slot) != 0) {
            self.reclaimSlot(slot);
            return;
        }
        self.reconcileSlotAccounting(slot, 0) catch
            @panic("request-exclusive Metal KV discard accounting failed");
        self.leased_slots[slot] = false;
        self.slot_ring_page_count[slot] = 0;
        self.slot_ring_policy_initialized[slot] = false;
        self.slot_buffer_capacity_tokens[slot] = 0;
    }

    fn discardPrivateSlotRequired(self: *MetalKvStorage, slot: usize) !void {
        if (slot >= self.leased_slots.len or !self.leased_slots[slot]) return error.UnsafeExactCloneSourceDiscard;
        if (metal_runtime.termite_metal_decode_runtime_discard_paged_kv_slot(self.runtime, slot) != 0)
            return error.UnsafeExactCloneSourceDiscard;
        try self.reconcileSlotAccounting(slot, 0);
        self.leased_slots[slot] = false;
        self.slot_ring_page_count[slot] = 0;
        self.slot_ring_policy_initialized[slot] = false;
        self.slot_buffer_capacity_tokens[slot] = 0;
    }

    fn slotAllocationInfo(self: *MetalKvStorage, slot: usize) !metal_runtime.PagedKvSlotAllocationInfo {
        var info: metal_runtime.PagedKvSlotAllocationInfo = undefined;
        if (metal_runtime.termite_metal_decode_runtime_attention_span_slot_allocation_info(self.runtime, slot, &info) != 0)
            return error.DeviceAllocationSnapshotUnavailable;
        return info;
    }

    fn slotLiveBytes(info: metal_runtime.PagedKvSlotAllocationInfo) !usize {
        const installed: usize = @intCast(info.installed_live_bytes);
        const active: usize = @intCast(info.active_frame_retained_bytes);
        const submitted: usize = @intCast(info.submitted_frame_retained_bytes);
        return try std.math.add(usize, try std.math.add(usize, installed, active), submitted);
    }

    fn slotTransitionInfo(self: *MetalKvStorage, slot: usize) !metal_runtime.PagedKvSlotTransitionInfo {
        var info: metal_runtime.PagedKvSlotTransitionInfo = undefined;
        if (metal_runtime.termite_metal_decode_runtime_attention_span_slot_transition_info(self.runtime, slot, &info) != 0)
            return error.DeviceAllocationSnapshotUnavailable;
        return info;
    }

    fn reconcileSlotAccounting(self: *MetalKvStorage, slot: usize, newly_allocated_bytes: usize) !void {
        const tracker = self.owner_allocation_tracker orelse return;
        const prior = self.slot_accounted_live_bytes[slot];
        const info = try self.slotAllocationInfo(slot);
        const final = if (info.leased == 0) 0 else try slotLiveBytes(info);
        if (comptime builtin.is_test) self.last_owner_transition = .{ .slot = slot, .prior = prior, .newly_allocated = newly_allocated_bytes, .final = final };
        try tracker.recordDeviceTransition(.private_device, prior, newly_allocated_bytes, final);
        self.slot_accounted_live_bytes[slot] = final;
    }

    /// Bind a sequence/layer to either the layer-global full-history page pool
    /// or a dedicated cyclic SWA ring. `share_global` must be true before
    /// global sharing is allowed; legacy contiguous writes remain
    /// sequence-owned.
    fn acquireBinding(
        self: *MetalKvStorage,
        key: SlotKey,
        ring_page_count: usize,
        sliding_window_tokens: usize,
        share_global: bool,
        compactable_global: bool,
    ) !*SlotBinding {
        if (self.slot_map.getPtr(key)) |binding| {
            if (binding.ring_page_count != ring_page_count) return error.DeviceWriteFallback;
            if (compactable_global and !binding.compactable_global) return error.DeviceWriteFallback;
            if (binding.sliding_window_tokens != sliding_window_tokens) return error.DeviceWriteFallback;
            return binding;
        }

        const owns_slot = ring_page_count > 0 or !share_global;
        var global_created = false;
        var owned_created = false;
        var slot: usize = undefined;
        errdefer {
            if (global_created) _ = self.global_layer_slots.remove(key.layer_index);
            if (global_created or owned_created) self.reclaimSlot(slot);
        }
        if (!owns_slot) {
            if (self.global_layer_slots.get(key.layer_index)) |existing| {
                slot = existing;
            } else {
                slot = try self.allocateSlot();
                global_created = true;
                try self.global_layer_slots.put(self.allocator, key.layer_index, slot);
                self.slot_ring_page_count[slot] = 0;
                self.slot_ring_policy_initialized[slot] = true;
            }
        } else {
            slot = try self.allocateSlot();
            owned_created = true;
            _ = try self.configureSlotRingPolicy(slot, ring_page_count);
        }

        try self.slot_map.put(self.allocator, key, .{
            .slot = slot,
            .ring_page_count = ring_page_count,
            .sliding_window_tokens = sliding_window_tokens,
            .sequence_owned = owns_slot,
            .compactable_global = compactable_global,
        });
        return self.slot_map.getPtr(key).?;
    }

    fn flagValueEnabled(value: []const u8) bool {
        return value.len != 0 and
            !std.mem.eql(u8, value, "0") and
            !std.ascii.eqlIgnoreCase(value, "false") and
            !std.ascii.eqlIgnoreCase(value, "no") and
            !std.ascii.eqlIgnoreCase(value, "off");
    }

    fn envFlagEnabled(name: [*:0]const u8) bool {
        const value = std.c.getenv(name) orelse return false;
        return flagValueEnabled(std.mem.span(value));
    }

    pub fn splitSwaKvRingEnabled() bool {
        return !envFlagEnabled("TERMITE_METAL_DISABLE_SPLIT_SWA_KV_RING") and
            !envFlagEnabled("TERMITE_METAL_DISABLE_PAGED_SLOT_ATTENTION");
    }

    fn requestedRingPageCount(
        page_size_tokens: u16,
        sliding_window: usize,
        max_inflight_tokens: usize,
        allow_swa_ring: bool,
    ) !usize {
        if (!allow_swa_ring or sliding_window == 0 or max_inflight_tokens == 0 or page_size_tokens == 0) return 0;
        // Production requests opt in through the typed KV policy. Keep only a
        // hard rollback flag here.
        if (envFlagEnabled("TERMITE_METAL_DISABLE_SPLIT_SWA_KV_RING")) return 0;
        if (envFlagEnabled("TERMITE_METAL_DISABLE_PAGED_SLOT_ATTENTION")) return 0;
        return storage_runtime.swaRingPageCount(page_size_tokens, sliding_window, max_inflight_tokens, allow_swa_ring);
    }

    fn configureSlotRingPolicy(self: *MetalKvStorage, slot: usize, requested_page_count: usize) !usize {
        if (!self.slot_ring_policy_initialized[slot]) {
            self.slot_ring_page_count[slot] = requested_page_count;
            self.slot_ring_policy_initialized[slot] = true;
        } else if (self.slot_ring_page_count[slot] != requested_page_count) {
            return error.DeviceWriteFallback;
        }
        return self.slot_ring_page_count[slot];
    }

    fn pageTokenOffsets(
        self: *MetalKvStorage,
        logical_blocks: []const u32,
        needed_blocks: usize,
        page_size_tokens: usize,
        ring_page_count: usize,
    ) !storage_runtime.PageTokenOffsets {
        if (ring_page_count > 0) {
            return self.cyclic_page_table_cache.get(
                self.allocator,
                page_size_tokens,
                ring_page_count,
                needed_blocks,
                !envFlagEnabled("TERMITE_METAL_DISABLE_CYCLIC_PAGE_TABLE_CACHE"),
            );
        }
        if (logical_blocks.len < needed_blocks) return error.KvCapacityTooSmall;
        const offsets = try self.allocator.alloc(u32, needed_blocks);
        errdefer self.allocator.free(offsets);
        for (offsets, logical_blocks[0..needed_blocks]) |*offset, block_id| {
            const token_offset = std.math.mul(usize, @as(usize, block_id), page_size_tokens) catch return error.KvCapacityTooSmall;
            if (token_offset > std.math.maxInt(u32)) return error.KvCapacityTooSmall;
            offset.* = @intCast(token_offset);
        }
        return .{ .owned = offsets };
    }

    fn identityPageTokenOffsets(
        self: *MetalKvStorage,
        needed_blocks: usize,
        page_size_tokens: usize,
    ) !storage_runtime.PageTokenOffsets {
        if (needed_blocks == 0 or page_size_tokens == 0) return error.KvCapacityTooSmall;
        const offsets = try self.allocator.alloc(u32, needed_blocks);
        errdefer self.allocator.free(offsets);
        for (offsets, 0..) |*offset, block_index| {
            const token_offset = std.math.mul(usize, block_index, page_size_tokens) catch return error.KvCapacityTooSmall;
            offset.* = std.math.cast(u32, token_offset) orelse return error.KvCapacityTooSmall;
        }
        return .{ .owned = offsets };
    }

    fn reserveSlotCapacity(
        self: *MetalKvStorage,
        slot: usize,
        required_tokens: usize,
        page_size_tokens: usize,
        ring_page_count: usize,
        key_row_bytes: usize,
        v_row_stride: usize,
    ) !void {
        const current = self.slot_buffer_capacity_tokens[slot];
        if (required_tokens <= current) return;
        const target = if (ring_page_count > 0 or envFlagEnabled("TERMITE_METAL_DISABLE_GEOMETRIC_KV_GROWTH"))
            required_tokens
        else
            try storage_runtime.geometricKvTokenCapacity(current, required_tokens, page_size_tokens);
        // Reconcile generations retired since the previous frame before
        // recording this transition. This has no allocation overlap.
        try self.reconcileSlotAccounting(slot, 0);
        const transition_before = try self.slotTransitionInfo(slot);
        const rc = metal_runtime.termite_metal_decode_runtime_reserve_attention_span_slot_buffers(
            self.runtime,
            slot,
            @backingInt(self.format),
            target,
            key_row_bytes,
            v_row_stride,
        );
        const transition_after = try self.slotTransitionInfo(slot);
        if (transition_after.operation_nonce != transition_before.operation_nonce +% 1)
            return error.DeviceAllocationSnapshotUnavailable;
        const newly_allocated = std.math.cast(usize, transition_after.newly_allocated_bytes) orelse
            return error.DeviceAllocationReceiptOverflow;
        // The C receipt survives every failure return, including a new
        // allocation that is released before this after-call snapshot.
        try self.reconcileSlotAccounting(slot, newly_allocated);
        if (transition_after.overflow != 0)
            return error.DeviceAllocationReceiptOverflow;
        if (rc != 0) return error.DeviceWriteFallback;
        self.slot_buffer_capacity_tokens[slot] = target;
        _ = self.allocationSnapshot();
    }

    fn allocationSnapshotOp(ctx: *anyopaque) storage_runtime.DeviceAllocationSnapshot {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        return self.allocationSnapshot();
    }

    fn allocationSnapshot(self: *MetalKvStorage) storage_runtime.DeviceAllocationSnapshot {
        var snapshot = storage_runtime.DeviceAllocationSnapshot{ .available = true };
        for (self.leased_slots, 0..) |leased, slot| {
            if (!leased) continue;
            var info: metal_runtime.PagedKvSlotAllocationInfo = undefined;
            if (metal_runtime.termite_metal_decode_runtime_attention_span_slot_allocation_info(self.runtime, slot, &info) != 0 or info.leased == 0)
                return .{};
            snapshot.kv_bytes +|= info.installed_live_bytes;
            snapshot.live_bytes +|= info.installed_live_bytes;
            snapshot.live_bytes +|= info.active_frame_retained_bytes;
            snapshot.live_bytes +|= info.submitted_frame_retained_bytes;
            snapshot.readback_bytes +|= info.lease_readback_bytes;
        }
        self.allocation_peak_bytes = @max(self.allocation_peak_bytes, snapshot.live_bytes);
        snapshot.peak_bytes = self.allocation_peak_bytes;
        return snapshot;
    }

    fn requiredTokenCapacity(block_offsets: []const u32, page_size_tokens: usize) !usize {
        var required: usize = 0;
        for (block_offsets) |offset| {
            const end = std.math.add(usize, @as(usize, offset), page_size_tokens) catch return error.KvCapacityTooSmall;
            required = @max(required, end);
        }
        return required;
    }

    /// Remove every binding for `sequence_id`. Full-history page-pool slots
    /// remain live because retained physical block ids may still reference
    /// their bytes; only sequence-owned ring/legacy slots are reset/reclaimed.
    fn releaseSequenceSlots(self: *MetalKvStorage, sequence_id: storage_runtime.SequenceId) void {
        var keys: [metal_runtime.attention_span_slot_capacity]SlotKey = undefined;
        while (true) {
            var it = self.slot_map.iterator();
            var key_count: usize = 0;
            while (it.next()) |entry| {
                if (entry.key_ptr.sequence_id != sequence_id) continue;
                keys[key_count] = entry.key_ptr.*;
                key_count += 1;
                if (key_count == keys.len) break;
            }
            if (key_count == 0) return;
            for (keys[0..key_count]) |key| {
                if (self.slot_map.fetchRemove(key)) |removed| {
                    if (removed.value.selected_positions) |positions| self.allocator.free(positions);
                    if (removed.value.selected_positions != null) {
                        self.discardPrivateSlot(removed.value.slot);
                    } else if (removed.value.ownsSlot()) {
                        self.reclaimSlot(removed.value.slot);
                    }
                }
            }
        }
    }

    fn rowLayout(
        self: *const MetalKvStorage,
        num_kv_heads: u32,
        head_dim: u32,
    ) struct {
        token_values: usize,
        key_row_bytes: usize,
        base_key_row_bytes: usize,
        v_row_stride: usize,
    } {
        const token_values: usize = @as(usize, num_kv_heads) * @as(usize, head_dim);
        const int8_row_bytes: usize = @as(usize, num_kv_heads) * (@as(usize, head_dim) + @sizeOf(f32));
        const key_row_bytes: usize = switch (self.format) {
            .polar4 => turboquant.polar4KeyBytes(num_kv_heads, head_dim),
            .turbo3 => turboquant.turbo3KeyBytes(num_kv_heads, head_dim) + turboquant.turbo3ResidualBytes(num_kv_heads, head_dim),
            .raw_f32 => token_values * @sizeOf(f32),
            .f16 => token_values * @sizeOf(u16),
            .int8_per_head => int8_row_bytes,
        };
        const base_key_row_bytes: usize = switch (self.format) {
            .polar4 => turboquant.polar4KeyBytes(num_kv_heads, head_dim),
            .turbo3 => turboquant.turbo3KeyBytes(num_kv_heads, head_dim),
            .raw_f32 => token_values * @sizeOf(f32),
            .f16 => token_values * @sizeOf(u16),
            .int8_per_head => int8_row_bytes,
        };
        return .{
            .token_values = token_values,
            .key_row_bytes = key_row_bytes,
            .base_key_row_bytes = base_key_row_bytes,
            .v_row_stride = token_values,
        };
    }

    pub fn deviceWriteHook(self: *MetalKvStorage) storage_runtime.DeviceWriteHook {
        return .{
            .ctx = @ptrCast(self),
            .vtable = &hook_vtable,
            .owner_accounting = if (self.owner_allocation_tracker != null)
                .exact_private_device
            else
                .unavailable,
        };
    }

    pub fn fromDeviceWriteHook(hook: storage_runtime.DeviceWriteHook) !*MetalKvStorage {
        if (hook.vtable != &hook_vtable) return error.UnsupportedAttentionMatchRoute;
        return @ptrCast(@alignCast(hook.ctx));
    }

    pub fn configureAttentionMatchLocalCapacityPolicy(self: *MetalKvStorage, requested: AttentionMatchLocalCapacityPolicy) !void {
        try self.attention_match_local_capacity.configure(requested);
    }

    pub fn attentionMatchBackend(
        self: *MetalKvStorage,
        allocator: std.mem.Allocator,
        limits: @import("../runtime/kv/attention_match_checkpoint.zig").Limits,
        max_device_bytes: usize,
    ) !attention_match_storage.Backend {
        if (self.format != .f16) return error.UnsupportedMetalKvCheckpointEncoding;
        const tracker = self.owner_allocation_tracker orelse return error.ContextPhysicalAccountingUnavailable;
        if (self.attention_match_runner == null) {
            self.attention_match_runner = try attention_match_storage.Runner.initWithOptions(
                max_device_bytes,
                tracker,
                self.attention_match_local_capacity.lock(),
            );
        } else self.attention_match_runner.?.raiseMaxDeviceBytes(max_device_bytes);
        return self.attention_match_runner.?.backend(allocator, limits);
    }

    pub fn attentionMatchRunner(self: *MetalKvStorage) ?*attention_match_storage.Runner {
        if (self.attention_match_runner) |*runner| return runner;
        return null;
    }

    pub fn attentionMatchExactLocalWindowCapacity(self: *const MetalKvStorage) bool {
        return if (self.attention_match_runner) |runner| runner.exact_local_window_capacity else false;
    }

    pub fn retireExactKvForAttentionMatch(self: *MetalKvStorage) void {
        var binding_it = self.slot_map.valueIterator();
        while (binding_it.next()) |binding| {
            if (binding.selected_positions) |positions| self.allocator.free(positions);
        }
        self.slot_map.clearRetainingCapacity();
        self.global_layer_slots.clearRetainingCapacity();
        for (&self.leased_slots, 0..) |*leased, slot| {
            if (!leased.*) continue;
            self.reclaimSlot(slot);
        }
    }

    pub fn validateExactAttentionMatchSource(
        self: *MetalKvStorage,
        sequence: storage_runtime.SequenceId,
        expected_length: usize,
        candidate: *attention_match_storage.DeviceState,
    ) !void {
        if (self.format != .f16 or self.attention_match_binding != null or candidate.heads.len == 0)
            return error.UnsupportedAttentionMatchRoute;
        var donors: usize = 0;
        for (candidate.heads) |head| {
            if (head.logical_length != @as(u64, @intCast(expected_length))) return error.InvalidKvCheckpoint;
            if (head.head != 0) continue;
            donors += 1;
            const heads = try candidate.layer(head.layer);
            const binding = self.slot_map.get(.{ .sequence_id = sequence, .layer_index = head.layer }) orelse
                return error.StaleKvCompaction;
            if (binding.selected_positions != null or binding.ring_page_count != 0 or
                !binding.logical_contiguous or binding.position_offset != 0 or
                binding.physical_base_tokens != 0 or binding.written_tokens != expected_length)
                return error.StaleKvCompaction;
            try validateExactAttentionMatchGeometry(self.num_kv_heads, self.head_dim, heads, try self.slotInfo(binding.slot), expected_length);
        }
        if (donors != self.slot_map.count()) return error.UnsupportedAttentionMatchRoute;
    }

    fn validateExactAttentionMatchGeometry(pool_heads: u32, maximum_dim: u32, heads: []const attention_match_storage.Head, info: SlotInfo, expected_length: usize) !void {
        // Pool dimensions are allocation maxima. Gemma4 local/global donors
        // have different widths; the encoded source slot proves the actual
        // width. Keep the existing fixed-head-count route restriction so a
        // different head grouping cannot alias the same combined row width.
        if (heads.len == 0 or heads.len != pool_heads or heads[0].dim == 0 or heads[0].dim > maximum_dim)
            return error.InvalidKvCheckpoint;
        const dim = heads[0].dim;
        for (heads, 0..) |head, index| {
            if (head.dim != dim or head.head != index or head.layer != heads[0].layer or
                head.window != heads[0].window or head.logical_length != expected_length)
                return error.InvalidKvCheckpoint;
        }
        const width = try std.math.mul(usize, heads.len, dim);
        const row_bytes = try std.math.mul(usize, width, @sizeOf(f16));
        const bytes = try std.math.mul(usize, expected_length, row_bytes);
        if (info.tokens != expected_length or info.position_offset != 0 or
            info.key_row_bytes != row_bytes or info.v_row_stride != width or
            info.encoded_key_handle == null or info.v_handle == null or
            info.encoded_key_capacity < bytes or info.v_capacity < bytes)
            return error.InvalidKvCheckpoint;
    }

    fn cloneAllocationBytes(request: storage_runtime.device_clone.Request) !usize {
        const common = try request.allocationBytes(@sizeOf(f16));
        var metal_bytes: usize = 0;
        for (request.donors) |donor| {
            const width = try std.math.mul(usize, donor.heads, donor.dim);
            const row_bytes = try std.math.mul(usize, width, @sizeOf(f16));
            const capacity_rows: usize = if (donor.capacity_rows == 0) request.capacity else donor.capacity_rows;
            const buffer_bytes = try std.math.mul(usize, capacity_rows, row_bytes);
            // Metal private buffers may round their backing allocation to a
            // 16 KiB VM page. Bound K and V independently before allocating.
            const rounded = try std.math.add(usize, buffer_bytes, 16 * 1024 - 1);
            const per_buffer = rounded & ~@as(usize, 16 * 1024 - 1);
            metal_bytes = try std.math.add(usize, metal_bytes, try std.math.mul(usize, per_buffer, 2));
        }
        const allocation_bound = @max(common, metal_bytes);
        if (allocation_bound > request.max_bytes) return error.DeviceKvCloneBudgetExceeded;
        return allocation_bound;
    }

    fn cloneExactPrefix(
        ctx: *anyopaque,
        source_hook: storage_runtime.DeviceWriteHook,
        request: storage_runtime.device_clone.Request,
    ) anyerror!storage_runtime.device_clone.Result {
        const destination: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (source_hook.vtable != &hook_vtable or source_hook.ctx == ctx)
            return error.InvalidDeviceKvClone;
        const source: *MetalKvStorage = @ptrCast(@alignCast(source_hook.ctx));
        if (source.runtime != destination.runtime or source.format != .f16 or destination.format != .f16 or
            source.attention_match_binding != null or destination.attention_match_binding != null or
            source.page_size_tokens != request.page_size or destination.page_size_tokens != request.page_size or
            source.slot_map.count() != request.donors.len or destination.slot_map.count() != 0 or
            destination.global_layer_slots.count() != 0 or metal_runtime.hasActiveFrame(destination.runtime) or
            metal_runtime.hasSubmittedFrame(destination.runtime))
            return error.DeviceKvCloneUnsupported;

        const allocation_bound = try cloneAllocationBytes(request);
        const admitted_host_staging_bytes = try request.hostStagingBytes(@sizeOf(f16));
        try request.check();

        const Staged = struct {
            key: SlotKey,
            slot: usize,
            layout: @TypeOf(destination.rowLayout(1, 1)),
            capacity_rows: usize,
            ring_page_count: usize,
            sliding_window_tokens: usize,
            written_tokens: usize,
            position_offset: usize,
        };
        var staged: [256]Staged = undefined;

        // Validate the complete immutable source before reserving destination
        // map storage or leasing a single Metal slot.
        for (request.donors, 0..) |donor, index| {
            const key = SlotKey{ .sequence_id = request.source_sequence, .layer_index = donor.layer };
            const binding = source.slot_map.get(key) orelse return error.InvalidDeviceKvClone;
            const layout = source.rowLayout(donor.heads, donor.dim);
            const capacity_rows: usize = if (donor.capacity_rows == 0) request.capacity else donor.capacity_rows;
            const expected_ring_rows = try std.math.mul(usize, binding.ring_page_count, request.page_size);
            if (binding.selected_positions != null or binding.physical_base_tokens != 0) {
                std.log.err("device_kv_clone_layout_rejection group=binding_selection_base source_sequence={d} destination_sequence={d} layer={d} heads={d} dim={d} selected_positions_present={} physical_base_tokens={d}", .{ request.source_sequence, request.destination_sequence, donor.layer, donor.heads, donor.dim, binding.selected_positions != null, binding.physical_base_tokens });
                return error.UnsupportedDeviceKvCloneLayout;
            }
            if (binding.written_tokens != request.tokens or binding.position_offset != 0) {
                std.log.err("device_kv_clone_layout_rejection group=binding_token_offset source_sequence={d} destination_sequence={d} layer={d} written_tokens={d} request_tokens={d} position_offset={d}", .{ request.source_sequence, request.destination_sequence, donor.layer, binding.written_tokens, request.tokens, binding.position_offset });
                return error.UnsupportedDeviceKvCloneLayout;
            }
            if (binding.ring_page_count == 0 and (donor.host_staged_ring or !binding.logical_contiguous or capacity_rows < request.tokens)) {
                std.log.err("device_kv_clone_layout_rejection group=binding_contiguous source_sequence={d} destination_sequence={d} layer={d} ring_page_count={d} host_staged_ring={} logical_contiguous={} capacity_rows={d} request_tokens={d} page_size={d}", .{ request.source_sequence, request.destination_sequence, donor.layer, binding.ring_page_count, donor.host_staged_ring, binding.logical_contiguous, capacity_rows, request.tokens, request.page_size });
                return error.UnsupportedDeviceKvCloneLayout;
            }
            if (binding.ring_page_count != 0 and (!donor.host_staged_ring or binding.logical_contiguous or capacity_rows != expected_ring_rows)) {
                std.log.err("device_kv_clone_layout_rejection group=binding_ring source_sequence={d} destination_sequence={d} layer={d} ring_page_count={d} host_staged_ring={} logical_contiguous={} capacity_rows={d} expected_ring_rows={d} page_size={d}", .{ request.source_sequence, request.destination_sequence, donor.layer, binding.ring_page_count, donor.host_staged_ring, binding.logical_contiguous, capacity_rows, expected_ring_rows, request.page_size });
                return error.UnsupportedDeviceKvCloneLayout;
            }
            const info = try source.slotInfo(binding.slot);
            const key_bytes = try std.math.mul(usize, request.tokens, layout.key_row_bytes);
            const value_bytes = try std.math.mul(
                usize,
                try std.math.mul(usize, request.tokens, layout.v_row_stride),
                @sizeOf(f16),
            );
            const required_key_capacity = @min(key_bytes, capacity_rows * layout.key_row_bytes);
            const required_value_capacity = @min(value_bytes, capacity_rows * layout.v_row_stride * @sizeOf(f16));
            if (info.tokens != request.tokens or info.position_offset != 0 or
                info.key_row_bytes != layout.key_row_bytes or info.v_row_stride != layout.v_row_stride or
                info.encoded_key_handle == null or info.v_handle == null or
                info.encoded_key_capacity < required_key_capacity or info.v_capacity < required_value_capacity)
            {
                std.log.err("device_kv_clone_layout_rejection group=slot_info source_sequence={d} destination_sequence={d} layer={d} info_tokens={d} request_tokens={d} info_position_offset={d} info_key_row_bytes={d} expected_key_row_bytes={d} info_v_row_stride={d} expected_v_row_stride={d} encoded_key_handle_present={} v_handle_present={} encoded_key_capacity={d} required_key_capacity={d} v_capacity={d} required_value_capacity={d}", .{ request.source_sequence, request.destination_sequence, donor.layer, info.tokens, request.tokens, info.position_offset, info.key_row_bytes, layout.key_row_bytes, info.v_row_stride, layout.v_row_stride, info.encoded_key_handle != null, info.v_handle != null, info.encoded_key_capacity, required_key_capacity, info.v_capacity, required_value_capacity });
                return error.UnsupportedDeviceKvCloneLayout;
            }
            staged[index] = .{
                .key = .{ .sequence_id = request.destination_sequence, .layer_index = donor.layer },
                .slot = binding.slot,
                .layout = layout,
                .capacity_rows = capacity_rows,
                .ring_page_count = binding.ring_page_count,
                .sliding_window_tokens = binding.sliding_window_tokens,
                .written_tokens = binding.written_tokens,
                .position_offset = binding.position_offset,
            };
        }
        try destination.slot_map.ensureUnusedCapacity(destination.allocator, @intCast(request.donors.len));

        var staged_count: usize = 0;
        errdefer for (staged[0..staged_count]) |entry| destination.discardPrivateSlot(entry.slot);
        var copied_bytes: usize = 0;
        var peak_host_staging_bytes: usize = 0;
        for (staged[0..request.donors.len]) |*entry| {
            try request.check();
            const source_slot = entry.slot;
            var destination_slot = try destination.allocateSlot();
            errdefer destination.discardPrivateSlot(destination_slot);

            const requested_buffer_bytes = try std.math.mul(usize, entry.capacity_rows, entry.layout.key_row_bytes);
            const rounded = try std.math.add(usize, requested_buffer_bytes, 16 * 1024 - 1);
            const per_donor_bound = try std.math.mul(usize, rounded & ~@as(usize, 16 * 1024 - 1), 2);
            var allocation = try destination.slotAllocationInfo(destination_slot);
            if (try slotLiveBytes(allocation) > per_donor_bound) {
                // A released slot may retain a larger previous tenant's
                // backing. Drop it so this owner's admitted bound stays exact.
                destination.discardPrivateSlot(destination_slot);
                destination_slot = try destination.allocateSlot();
                allocation = try destination.slotAllocationInfo(destination_slot);
                if (allocation.installed_live_bytes != 0) return error.DeviceKvCloneBudgetExceeded;
            }
            try destination.reserveSlotCapacity(
                destination_slot,
                entry.capacity_rows,
                request.page_size,
                entry.ring_page_count,
                entry.layout.key_row_bytes,
                entry.layout.v_row_stride,
            );
            allocation = try destination.slotAllocationInfo(destination_slot);
            if (try slotLiveBytes(allocation) > per_donor_bound)
                return error.DeviceKvCloneBudgetExceeded;

            if (entry.ring_page_count == 0) {
                const clone_rc = metal_runtime.termite_metal_decode_runtime_clone_attention_span_slot_prefix(
                    destination.runtime,
                    source_slot,
                    destination_slot,
                    @backingInt(destination.format),
                    request.tokens,
                    entry.layout.key_row_bytes,
                    entry.layout.v_row_stride,
                    0,
                );
                if (clone_rc != 0) return error.DeviceKvCloneUnsupported;
            } else {
                // Preserve the physical ring and its logical-page mapping.
                // Only the bounded local ring is staged through host memory;
                // global full-history donors remain device-to-device clones.
                const source_info = try source.slotInfo(source_slot);
                const destination_info = try destination.slotInfo(destination_slot);
                const key_bytes = try std.math.mul(usize, entry.capacity_rows, entry.layout.key_row_bytes);
                const value_row_bytes = try std.math.mul(usize, entry.layout.v_row_stride, @sizeOf(f16));
                const value_bytes = try std.math.mul(usize, entry.capacity_rows, value_row_bytes);
                peak_host_staging_bytes = @max(peak_host_staging_bytes, try std.math.add(usize, key_bytes, value_bytes));
                if (peak_host_staging_bytes > admitted_host_staging_bytes)
                    return error.DeviceKvCloneHostStagingBudgetExceeded;
                const key_staging = try destination.allocator.alloc(u8, key_bytes);
                defer destination.allocator.free(key_staging);
                const value_staging = try destination.allocator.alloc(u8, value_bytes);
                defer destination.allocator.free(value_staging);
                try request.check();
                if (metal_runtime.termite_metal_buffer_download(destination.runtime, source_info.encoded_key_handle.?, 0, key_staging.ptr, key_bytes) != 0 or
                    metal_runtime.termite_metal_buffer_download(destination.runtime, source_info.v_handle.?, 0, value_staging.ptr, value_bytes) != 0 or
                    metal_runtime.termite_metal_buffer_upload(destination.runtime, destination_info.encoded_key_handle.?, 0, key_staging.ptr, key_bytes) != 0 or
                    metal_runtime.termite_metal_buffer_upload(destination.runtime, destination_info.v_handle.?, 0, value_staging.ptr, value_bytes) != 0)
                    return error.DeviceKvCloneUnsupported;
                if (metal_runtime.termite_metal_decode_runtime_publish_attention_span_slot_ring(
                    destination.runtime,
                    destination_slot,
                    entry.written_tokens,
                    entry.capacity_rows,
                    entry.layout.key_row_bytes,
                    entry.layout.v_row_stride,
                    entry.position_offset,
                ) != 0) return error.DeviceKvCloneUnsupported;
            }
            allocation = try destination.slotAllocationInfo(destination_slot);
            if (try slotLiveBytes(allocation) > per_donor_bound)
                return error.DeviceKvCloneBudgetExceeded;
            entry.slot = destination_slot;
            destination.slot_ring_page_count[destination_slot] = entry.ring_page_count;
            destination.slot_ring_policy_initialized[destination_slot] = true;
            copied_bytes = try std.math.add(usize, copied_bytes, try std.math.mul(
                usize,
                if (entry.ring_page_count == 0) request.tokens else entry.capacity_rows,
                try std.math.mul(usize, entry.layout.key_row_bytes, 2),
            ));
            staged_count += 1;
        }
        try request.check();

        // Map capacity was reserved before any device mutation. Publication is
        // now infallible and transfers ownership of every staged slot together.
        for (staged[0..staged_count]) |entry| {
            destination.slot_map.putAssumeCapacityNoClobber(entry.key, .{
                .slot = entry.slot,
                .sequence_owned = true,
                .logical_contiguous = entry.ring_page_count == 0,
                .ring_page_count = entry.ring_page_count,
                .sliding_window_tokens = entry.sliding_window_tokens,
                .physical_base_tokens = 0,
                .written_tokens = entry.written_tokens,
                .position_offset = entry.position_offset,
            });
        }
        staged_count = 0;
        destination.attention_match_revision +|= 1;
        return .{
            .allocation_bound_bytes = allocation_bound,
            .copied_bytes = copied_bytes,
            .peak_host_staging_bytes = peak_host_staging_bytes,
            .donors = request.donors.len,
        };
    }

    fn exportExactLayerRanges(ctx: *anyopaque, request: storage_runtime.ExactLayerRangeExportRequest) anyerror!storage_runtime.ExactLayerRangeExportResult {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.format != .f16 or self.attention_match_binding != null or metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime)) return error.DeviceKvExportUnsupported;
        const admitted = try request.validate();
        try request.check();
        var copies: [256]metal_runtime.ExactKvRangeCopy = undefined;
        var observed: [256]ExactExportSourceSnapshot = undefined;
        for (request.ranges, 0..) |range, i| {
            const binding = self.slot_map.get(.{ .sequence_id = request.source_sequence, .layer_index = range.layer_index }) orelse {
                std.log.err("exact_range_export_source_rejected group=missing_binding sequence={d} layer={d} expected_tokens={d} expected_offset={d}", .{ request.source_sequence, range.layer_index, request.expected_written_tokens, request.expected_position_offset });
                return error.InvalidExactLayerRangeExportSource;
            };
            if (binding.selected_positions != null or binding.physical_base_tokens != 0 or binding.ring_page_count != 0 or !binding.logical_contiguous or binding.written_tokens != request.expected_written_tokens or binding.position_offset != request.expected_position_offset) {
                std.log.err("exact_range_export_source_rejected group=binding sequence={d} layer={d} slot={d} selected_positions={} physical_base={d} ring_pages={d} contiguous={} written={d} expected_tokens={d} offset={d} expected_offset={d}", .{ request.source_sequence, range.layer_index, binding.slot, binding.selected_positions != null, binding.physical_base_tokens, binding.ring_page_count, binding.logical_contiguous, binding.written_tokens, request.expected_written_tokens, binding.position_offset, request.expected_position_offset });
                return error.InvalidExactLayerRangeExportSource;
            }
            const layout = self.rowLayout(range.num_kv_heads, range.head_dim);
            const info = try self.slotInfo(binding.slot);
            const bytes = try std.math.mul(usize, range.rows, layout.key_row_bytes);
            const source_offset = try std.math.mul(usize, range.source_start, layout.key_row_bytes);
            if (info.tokens != request.expected_written_tokens or info.position_offset != request.expected_position_offset or
                info.key_row_bytes != layout.key_row_bytes or info.v_row_stride != layout.v_row_stride or
                info.encoded_key_handle == null or info.v_handle == null or source_offset > info.encoded_key_capacity or
                bytes > info.encoded_key_capacity - source_offset or source_offset > info.v_capacity or
                bytes > info.v_capacity - source_offset)
            {
                std.log.err("exact_range_export_source_rejected group=slot sequence={d} layer={d} slot={d} info_tokens={d} expected_tokens={d} info_offset={d} expected_offset={d} key_row={d} expected_key_row={d} value_stride={d} expected_value_stride={d} key_handle={} value_handle={} source_offset={d} bytes={d} key_capacity={d} value_capacity={d}", .{ request.source_sequence, range.layer_index, binding.slot, info.tokens, request.expected_written_tokens, info.position_offset, request.expected_position_offset, info.key_row_bytes, layout.key_row_bytes, info.v_row_stride, layout.v_row_stride, info.encoded_key_handle != null, info.v_handle != null, source_offset, bytes, info.encoded_key_capacity, info.v_capacity });
                return error.InvalidExactLayerRangeExportSource;
            }
            observed[i] = .{
                .slot = binding.slot,
                .written_tokens = binding.written_tokens,
                .position_offset = binding.position_offset,
                .key_handle = info.encoded_key_handle.?,
                .value_handle = info.v_handle.?,
                .key_capacity = info.encoded_key_capacity,
                .value_capacity = info.v_capacity,
                .key_row_bytes = info.key_row_bytes,
                .value_row_stride = info.v_row_stride,
                .info_tokens = info.tokens,
                .info_position_offset = info.position_offset,
            };
            copies[i] = .{ .src_k = info.encoded_key_handle.?, .src_v = info.v_handle.?, .src_k_offset = source_offset, .src_v_offset = source_offset, .dst_k = range.destination_k.handle, .dst_v = range.destination_v.handle, .dst_k_offset = range.destination_k.byte_offset, .dst_v_offset = range.destination_v.byte_offset, .bytes = bytes, .rows = @intCast(range.rows), .source_start = @intCast(range.source_start), .heads = range.num_kv_heads, .dim = range.head_dim, .destination_head_capacity_rows = @intCast(range.destination_head_capacity_rows), .destination_ring_capacity_rows = @intCast(range.destination_ring_capacity_rows), .destination_layout = @backingInt(range.destination_layout) };
        }
        for (request.ranges) |range| for (copies[0..request.ranges.len]) |source| if (range.destination_k.handle == source.src_k.? or range.destination_k.handle == source.src_v.? or range.destination_v.handle == source.src_k.? or range.destination_v.handle == source.src_v.?) return error.AliasedExactLayerRangeExport;
        try request.check();
        if (metal_runtime.termite_metal_exact_kv_range_copy_batch(self.runtime, copies[0..request.ranges.len].ptr, request.ranges.len) != 0) return error.MetalExactLayerRangeExportFailed;
        try request.check();
        for (request.ranges, 0..) |range, i| {
            const binding = self.slot_map.get(.{ .sequence_id = request.source_sequence, .layer_index = range.layer_index }) orelse return error.ExactLayerRangeExportSourceChanged;
            const info = try self.slotInfo(binding.slot);
            if (!observed[i].matches(binding, info))
                return error.ExactLayerRangeExportSourceChanged;
        }
        // The exact range batch consumes source/destination Metal handles and
        // offsets directly. No host allocation or materialization occurs in
        // this exporter before the fenced completion receipt.
        return .{ .copied_bytes = admitted, .layers = request.ranges.len, .fenced = true, .peak_host_staging_bytes = 0 };
    }

    test "exact export source snapshot rejects binding and buffer mutation" {
        var key: u8 = 0;
        var value: u8 = 0;
        var other: u8 = 0;
        var binding: SlotBinding = .{ .slot = 3, .logical_contiguous = true, .written_tokens = 8 };
        var info: SlotInfo = .{ .encoded_key_handle = &key, .encoded_key_capacity = 64, .v_handle = &value, .v_capacity = 64, .tokens = 8, .key_row_bytes = 8, .v_row_stride = 4 };
        const snapshot: ExactExportSourceSnapshot = .{
            .slot = 3,
            .written_tokens = 8,
            .position_offset = 0,
            .key_handle = &key,
            .value_handle = &value,
            .key_capacity = 64,
            .value_capacity = 64,
            .key_row_bytes = 8,
            .value_row_stride = 4,
            .info_tokens = 8,
            .info_position_offset = 0,
        };
        try std.testing.expect(snapshot.matches(binding, info));
        binding.ring_page_count = 1;
        try std.testing.expect(!snapshot.matches(binding, info));
        binding.ring_page_count = 0;
        info.encoded_key_handle = &other;
        try std.testing.expect(!snapshot.matches(binding, info));
        info.encoded_key_handle = &key;
        info.tokens = 7;
        try std.testing.expect(!snapshot.matches(binding, info));
    }

    fn traceKvGather() bool {
        // Cached: this runs per KV write/read in the decode inner loop.
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |value| return value;
        const enabled = blk: {
            const value = std.c.getenv("TERMITE_METAL_TRACE_KV_GATHER") orelse break :blk false;
            const span = std.mem.span(value);
            break :blk span.len > 0 and !std.mem.eql(u8, span, "0");
        };
        S.cached = enabled;
        return enabled;
    }

    fn writeLayerKvSuffix(
        ctx: *anyopaque,
        write: storage_runtime.KvSuffixWrite,
        k: storage_runtime.DeviceKvRef,
        v: storage_runtime.DeviceKvRef,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.attention_match_binding != null) return error.UnsupportedAttentionMatchRoute;
        const num_kv_heads: u32 = write.num_kv_heads;
        const head_dim: u32 = write.head_dim;

        const layout = self.rowLayout(num_kv_heads, head_dim);
        const key_row_bytes = layout.key_row_bytes;
        const base_key_row_bytes = layout.base_key_row_bytes;
        const v_row_stride = layout.v_row_stride;

        const expected_elems = write.suffix_token_count * v_row_stride;
        const expected_bytes = expected_elems * @sizeOf(f32);
        if (k.byte_len < expected_bytes or v.byte_len < expected_bytes) return error.InvalidKvShape;

        const requested_ring_pages = try requestedRingPageCount(
            write.page_size_tokens,
            write.sliding_window,
            write.max_inflight_tokens,
            write.allow_swa_ring,
        );
        const binding = try self.acquireBinding(
            .{
                .sequence_id = write.sequence_id,
                .layer_index = @intCast(write.layer_index),
            },
            requested_ring_pages,
            if (requested_ring_pages > 0) write.sliding_window else 0,
            write.logical_blocks != null and !write.sequence_owned_global,
            write.sequence_owned_global,
        );
        if (binding.compactable_global and write.total_token_count > try selection.defaults.transientRows())
            return error.SelectedKvCapacityExceeded;
        if (binding.selected_positions) |old_positions| {
            if (requested_ring_pages != 0) return error.SelectedKvRequiresPagedAttention;
            if (write.suffix_token_count == 0) return error.SelectedKvWriteFailed;
            if (write.total_token_count < write.suffix_token_count or
                write.total_token_count - write.suffix_token_count != binding.written_tokens) return error.StaleKvSelection;
            const compact_total = std.math.add(usize, old_positions.len, write.suffix_token_count) catch return error.KvCapacityTooSmall;
            if (compact_total > try selection.defaults.transientRows()) return error.SelectedKvCapacityExceeded;
            const positions = try self.allocator.alloc(u32, compact_total);
            errdefer self.allocator.free(positions);
            @memcpy(positions[0..old_positions.len], old_positions);
            const suffix_abs_start = std.math.add(
                usize,
                write.position_offset,
                write.total_token_count - write.suffix_token_count,
            ) catch return error.KvCapacityTooSmall;
            for (positions[old_positions.len..], 0..) |*position, index| {
                const absolute = std.math.add(usize, suffix_abs_start, index) catch return error.KvCapacityTooSmall;
                position.* = std.math.cast(u32, absolute) orelse return error.KvCapacityTooSmall;
                if (position.* <= positions[old_positions.len - 1]) return error.InvalidKvPositions;
            }
            const compact_page_size: usize = self.page_size_tokens;
            if (compact_page_size == 0) return error.SelectedKvWriteFailed;
            const block_count = std.math.divCeil(usize, compact_total, compact_page_size) catch return error.KvCapacityTooSmall;
            const block_offsets = try self.allocator.alloc(u32, block_count);
            defer self.allocator.free(block_offsets);
            for (block_offsets, 0..) |*offset, block_index| {
                const token_offset = std.math.mul(usize, block_index, compact_page_size) catch return error.KvCapacityTooSmall;
                offset.* = std.math.cast(u32, token_offset) orelse return error.KvCapacityTooSmall;
            }
            try self.reserveSlotCapacity(
                binding.slot,
                compact_total,
                compact_page_size,
                0,
                key_row_bytes,
                v_row_stride,
            );
            const selected_rc = metal_runtime.termite_metal_decode_runtime_update_attention_paged_from_f32_key_device_slot(
                self.runtime,
                binding.slot,
                @backingInt(self.format),
                k.handle,
                k.byte_offset,
                v.handle,
                v.byte_offset,
                compact_total,
                write.suffix_token_count,
                num_kv_heads,
                head_dim,
                key_row_bytes,
                base_key_row_bytes,
                v_row_stride,
                positions[0],
                block_offsets.ptr,
                block_offsets.len,
                compact_page_size,
            );
            if (selected_rc != 0) return error.SelectedKvWriteFailed;
            binding.selected_positions = positions;
            binding.written_tokens = write.total_token_count;
            binding.position_offset = positions[0];
            self.allocator.free(old_positions);
            self.attention_match_revision +|= 1;
            return;
        }
        const slot = binding.slot;
        const ring_page_count = binding.ring_page_count;
        if (traceKvGather()) std.debug.print(
            "kv-write: seq={d} layer={d} suffix={d} total={d} slot={d} ring_pages={d}\n",
            .{ write.sequence_id, write.layer_index, write.suffix_token_count, write.total_token_count, slot, ring_page_count },
        );

        const rc = if (write.logical_blocks) |logical_blocks| paged: {
            if (logical_blocks.len == 0 or write.page_size_tokens == 0) break :paged -9999;
            const needed_blocks = std.math.divCeil(usize, write.total_token_count, write.page_size_tokens) catch break :paged -9999;
            if (logical_blocks.len < needed_blocks) break :paged -9999;
            const block_offsets_result = if (binding.compactable_global)
                self.identityPageTokenOffsets(needed_blocks, write.page_size_tokens) catch break :paged -9999
            else
                self.pageTokenOffsets(
                    logical_blocks,
                    needed_blocks,
                    write.page_size_tokens,
                    ring_page_count,
                ) catch break :paged -9999;
            defer block_offsets_result.deinit(self.allocator);
            const block_offsets = block_offsets_result.values();
            const required_tokens = if (ring_page_count > 0)
                std.math.mul(usize, ring_page_count, write.page_size_tokens) catch break :paged -9999
            else
                requiredTokenCapacity(block_offsets, write.page_size_tokens) catch break :paged -9999;
            self.reserveSlotCapacity(
                slot,
                required_tokens,
                write.page_size_tokens,
                ring_page_count,
                key_row_bytes,
                v_row_stride,
            ) catch break :paged -9999;
            binding.physical_base_tokens = if (ring_page_count > 0) 0 else block_offsets[0];
            binding.logical_contiguous = ring_page_count == 0;
            if (ring_page_count == 0) {
                for (block_offsets, 0..) |offset, block_idx| {
                    const expected = binding.physical_base_tokens + block_idx * @as(usize, write.page_size_tokens);
                    if (offset != expected) {
                        binding.logical_contiguous = false;
                        break;
                    }
                }
            }
            break :paged metal_runtime.termite_metal_decode_runtime_update_attention_paged_from_f32_key_device_slot(
                self.runtime,
                slot,
                @backingInt(self.format),
                k.handle,
                k.byte_offset,
                v.handle,
                v.byte_offset,
                write.total_token_count,
                write.suffix_token_count,
                num_kv_heads,
                head_dim,
                key_row_bytes,
                base_key_row_bytes,
                v_row_stride,
                write.position_offset,
                block_offsets.ptr,
                block_offsets.len,
                write.page_size_tokens,
            );
        } else blk: {
            if (ring_page_count > 0) break :blk -9999;
            binding.logical_contiguous = true;
            binding.physical_base_tokens = 0;
            self.reserveSlotCapacity(
                slot,
                write.total_token_count,
                write.page_size_tokens,
                ring_page_count,
                key_row_bytes,
                v_row_stride,
            ) catch break :blk -9999;
            break :blk metal_runtime.termite_metal_decode_runtime_update_attention_span_from_f32_key_device_slot(
                self.runtime,
                slot,
                @backingInt(self.format),
                k.handle,
                k.byte_offset,
                v.handle,
                v.byte_offset,
                write.total_token_count,
                num_kv_heads,
                head_dim,
                key_row_bytes,
                base_key_row_bytes,
                v_row_stride,
                write.position_offset,
            );
        };
        if (rc != 0) return error.DeviceWriteFallback;
        binding.written_tokens = write.total_token_count;
        binding.position_offset = write.position_offset;
        self.attention_match_revision +|= 1;
    }

    fn reserveLayerKvDevice(
        ctx: *anyopaque,
        reserve: storage_runtime.DeviceKvLayerReserve,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.attention_match_binding != null) return error.UnsupportedAttentionMatchRoute;
        if (reserve.token_capacity == 0) return;
        const layout = self.rowLayout(reserve.num_kv_heads, reserve.head_dim);
        if (layout.key_row_bytes == 0 or layout.v_row_stride == 0) return error.InvalidKvShape;

        const requested_ring_pages = try requestedRingPageCount(
            reserve.page_size_tokens,
            reserve.sliding_window,
            reserve.max_inflight_tokens,
            reserve.allow_swa_ring,
        );
        const binding = try self.acquireBinding(
            .{
                .sequence_id = reserve.sequence_id,
                .layer_index = @intCast(reserve.layer_index),
            },
            requested_ring_pages,
            if (requested_ring_pages > 0) reserve.sliding_window else 0,
            reserve.logical_blocks != null and !reserve.sequence_owned_global,
            reserve.sequence_owned_global,
        );
        // Selected donors grow by compact physical row count in the write
        // path; the caller's logical token capacity is not a valid buffer size.
        if (binding.selected_positions != null) return;
        const slot = binding.slot;
        const ring_page_count = binding.ring_page_count;
        if (traceKvGather()) std.debug.print(
            "kv-reserve: seq={d} layer={d} capacity={d} slot={d} ring_pages={d}\n",
            .{ reserve.sequence_id, reserve.layer_index, reserve.token_capacity, slot, ring_page_count },
        );
        if (ring_page_count > 0 and reserve.logical_blocks == null) return error.DeviceWriteFallback;

        const token_capacity = if (binding.compactable_global) blk: {
            binding.logical_contiguous = true;
            binding.physical_base_tokens = 0;
            break :blk reserve.token_capacity;
        } else if (ring_page_count > 0)
            std.math.mul(usize, ring_page_count, reserve.page_size_tokens) catch return error.KvCapacityTooSmall
        else if (reserve.logical_blocks) |logical_blocks| blk: {
            if (logical_blocks.len == 0 or reserve.page_size_tokens == 0) break :blk reserve.token_capacity;
            const needed_blocks = std.math.divCeil(usize, reserve.token_capacity, reserve.page_size_tokens) catch break :blk reserve.token_capacity;
            if (logical_blocks.len < needed_blocks) break :blk reserve.token_capacity;
            var capacity_tokens: usize = 0;
            binding.physical_base_tokens = std.math.mul(
                usize,
                @as(usize, logical_blocks[0]),
                reserve.page_size_tokens,
            ) catch return error.KvCapacityTooSmall;
            binding.logical_contiguous = true;
            for (logical_blocks[0..needed_blocks], 0..) |block_id, block_idx| {
                const block_start = std.math.mul(
                    usize,
                    @as(usize, block_id),
                    reserve.page_size_tokens,
                ) catch return error.KvCapacityTooSmall;
                const expected_delta = std.math.mul(
                    usize,
                    block_idx,
                    @as(usize, reserve.page_size_tokens),
                ) catch return error.KvCapacityTooSmall;
                const expected = std.math.add(
                    usize,
                    binding.physical_base_tokens,
                    expected_delta,
                ) catch return error.KvCapacityTooSmall;
                if (block_start != expected) binding.logical_contiguous = false;
                const block_end = std.math.add(
                    usize,
                    block_start,
                    reserve.page_size_tokens,
                ) catch return error.KvCapacityTooSmall;
                if (block_end > capacity_tokens) capacity_tokens = block_end;
            }
            break :blk capacity_tokens;
        } else blk: {
            binding.logical_contiguous = true;
            binding.physical_base_tokens = 0;
            break :blk reserve.token_capacity;
        };
        if (ring_page_count > 0) {
            binding.logical_contiguous = false;
            binding.physical_base_tokens = 0;
        }
        binding.position_offset = reserve.position_offset;

        try self.reserveSlotCapacity(
            slot,
            token_capacity,
            reserve.page_size_tokens,
            ring_page_count,
            layout.key_row_bytes,
            layout.v_row_stride,
        );
    }

    fn commitLayerKvDeviceWrite(
        ctx: *anyopaque,
        commit: storage_runtime.DeviceKvLayerWriteCommit,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.attention_match_binding != null) return error.UnsupportedAttentionMatchRoute;
        const binding = self.slot_map.getPtr(.{
            .sequence_id = commit.sequence_id,
            .layer_index = @intCast(commit.layer_index),
        }) orelse return error.DeviceWriteFallback;
        if (binding.selected_positions != null) return error.DeviceWriteFallback;
        if (!binding.commitWrite(commit.total_token_count, commit.position_offset)) return error.DeviceWriteFallback;
        self.attention_match_revision +|= 1;
    }

    fn repackSelectedGlobalKv(
        ctx: *anyopaque,
        repack: storage_runtime.SelectedGlobalKvRepack,
    ) anyerror!storage_runtime.DevicePagedKvLayer {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (repack.key_positions.len == 0 or repack.key_positions.len != repack.source_token_offsets.len)
            return error.InvalidKvShape;
        if (repack.num_kv_heads != self.num_kv_heads or repack.head_dim != self.head_dim)
            return error.InvalidKvShape;
        for (repack.key_positions[1..], repack.key_positions[0 .. repack.key_positions.len - 1]) |position, previous| {
            if (previous >= position) return error.InvalidKvPositions;
        }
        const key = SlotKey{
            .sequence_id = repack.sequence_id,
            .layer_index = std.math.cast(u32, repack.layer_index) orelse return error.InvalidKvShape,
        };
        const binding = self.slot_map.getPtr(key) orelse return error.DeviceReadFallback;
        if (binding.slot != repack.expected_source_slot or binding.written_tokens != repack.expected_written_tokens)
            return error.StaleKvSelection;
        if (binding.ring_page_count != 0 or !binding.compactable_global or !binding.sequence_owned)
            return error.DeviceWriteFallback;
        const source_physical_rows = if (binding.selected_positions) |positions| positions.len else binding.written_tokens;
        try validateSelectedSourceRows(repack.source_token_offsets, source_physical_rows);

        const layout = self.rowLayout(repack.num_kv_heads, repack.head_dim);
        const positions = try self.allocator.dupe(u32, repack.key_positions);
        errdefer self.allocator.free(positions);
        const destination_slot = try self.allocateSlot();
        errdefer self.discardPrivateSlot(destination_slot);
        const destination_capacity = try selection.defaults.transientRows();
        if (positions.len > destination_capacity) return error.SelectedKvCapacityExceeded;
        try self.reserveSlotCapacity(
            destination_slot,
            destination_capacity,
            self.page_size_tokens,
            0,
            layout.key_row_bytes,
            layout.v_row_stride,
        );
        self.slot_ring_page_count[destination_slot] = 0;
        self.slot_ring_policy_initialized[destination_slot] = true;

        const compact_rc = metal_runtime.termite_metal_decode_runtime_compact_attention_span_slot(
            self.runtime,
            binding.slot,
            destination_slot,
            @backingInt(self.format),
            repack.source_token_offsets.ptr,
            repack.source_token_offsets.len,
            layout.key_row_bytes,
            layout.v_row_stride,
            positions[0],
        );
        if (compact_rc != 0) return error.DeviceWriteFallback;

        // No allocation or fallible operation follows this point. Publish the
        // fully copied staging slot, then release only a sequence-owned source;
        // shared global page pools remain available to other bindings/cache
        // owners.
        const source_slot = binding.slot;
        const source_owned = binding.sequence_owned;
        const old_positions = binding.selected_positions;
        binding.* = .{
            .slot = destination_slot,
            .sequence_owned = true,
            .compactable_global = true,
            .logical_contiguous = true,
            .physical_base_tokens = 0,
            .written_tokens = repack.expected_written_tokens,
            .position_offset = positions[0],
            .selected_positions = positions,
        };
        if (old_positions) |old| self.allocator.free(old);
        if (source_owned) self.discardPrivateSlot(source_slot);

        return .{
            .runtime = @ptrCast(self.runtime),
            .slot = destination_slot,
            .format = @backingInt(self.format),
            .token_count = positions.len,
            .key_row_bytes = layout.key_row_bytes,
            .base_key_row_bytes = layout.base_key_row_bytes,
            .v_row_stride = layout.v_row_stride,
            .page_size_tokens = self.page_size_tokens,
            .position_offset = positions[0],
            .key_positions = positions,
        };
    }

    fn scoreSelectedAttention(ctx: *anyopaque, request: storage_runtime.SelectedAttentionScore) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        const layer_index = std.math.cast(u32, request.layer_index) orelse return error.InvalidKvShape;
        const binding = self.slot_map.get(.{ .sequence_id = request.sequence_id, .layer_index = layer_index }) orelse return error.DeviceReadFallback;
        if (request.key_positions.len == 0 or request.scores_out.len != request.key_positions.len) return error.InvalidKvShape;
        const query_head_values = std.math.mul(usize, request.query_positions.len, request.num_heads) catch return error.InvalidKvShape;
        const query_values = std.math.mul(usize, query_head_values, request.head_dim) catch return error.InvalidKvShape;
        if (request.query_positions.len == 0 or request.queries.len != query_values) return error.InvalidKvShape;
        if (request.num_heads == 0 or request.num_heads > 256 or request.num_kv_heads == 0 or request.head_dim == 0 or
            request.num_kv_heads > std.math.maxInt(u32) or request.head_dim > std.math.maxInt(u32) or
            request.num_heads % request.num_kv_heads != 0) return error.InvalidKvShape;
        if (request.query_positions.len > selection.defaults.query_horizon or
            request.key_positions.len > try selection.defaults.transientRows()) return error.InvalidKvShape;
        for (request.query_positions[1..], request.query_positions[0 .. request.query_positions.len - 1]) |position, previous| {
            if (position <= previous) return error.InvalidKvShape;
        }
        for (request.key_positions[1..], request.key_positions[0 .. request.key_positions.len - 1]) |position, previous| {
            if (position <= previous) return error.InvalidKvShape;
        }
        if (binding.selected_positions) |positions| {
            if (!std.mem.eql(u32, positions, request.key_positions)) return error.StaleKvSelection;
        } else if (request.key_positions.len != binding.written_tokens) return error.StaleKvSelection;
        // The scorer consumes compact physical rows from zero. A paged/ring
        // binding with a nonzero physical base needs the generic attention
        // route; silently treating its logical row zero as physical row zero
        // would score another sequence's keys.
        if (!binding.logical_contiguous or binding.physical_base_tokens != 0) return error.DeviceReadFallback;
        const layout = self.rowLayout(@intCast(request.num_kv_heads), @intCast(request.head_dim));
        const rc = metal_runtime.termite_metal_decode_runtime_score_selected_attention(
            self.runtime,
            binding.slot,
            @backingInt(self.format),
            request.queries.ptr,
            request.query_positions.ptr,
            request.query_positions.len,
            request.key_positions.ptr,
            request.key_positions.len,
            request.num_heads,
            request.num_kv_heads,
            request.head_dim,
            layout.key_row_bytes,
            layout.base_key_row_bytes,
            request.score_scale,
            request.softcap,
            request.scores_out.ptr,
            request.check_ctx,
            request.check_fn,
        );
        if (rc == -12) return error.Cancelled;
        if (rc != 0) return error.DeviceReadFallback;
        var score_sum: f64 = 0;
        for (request.scores_out) |score| {
            if (!std.math.isFinite(score) or score < 0) return error.InvalidSelectionScore;
            score_sum += score;
        }
        // Every captured query has at least sink/recent causal keys, so a
        // normalized consumer distribution must sum to one. A shortfall also
        // surfaces any nonfinite encoded key rejected by pass one.
        if (@abs(score_sum - 1.0) > 2e-3) return error.InvalidSelectionScore;
    }

    fn validateSelectedSourceRows(source_rows: []const u32, committed_physical_rows: usize) !void {
        if (source_rows.len == 0 or committed_physical_rows == 0) return error.InvalidKvShape;
        var previous: ?u32 = null;
        for (source_rows) |row| {
            if (row >= committed_physical_rows) return error.InvalidKvShape;
            if (previous) |prior| if (row <= prior) return error.InvalidKvShape;
            previous = row;
        }
    }

    fn hookDeinit(ctx: *anyopaque, _: std.mem.Allocator) void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        // The hook owns its allocator-backed state. The storage runtime may
        // have a different allocator; destruction must use the one that
        // created the hook and its maps.
        const allocator = self.allocator;
        drainForOwnerSnapshot(self) catch
            @panic("request-exclusive Metal KV drain failed during teardown");
        if (self.attention_match_binding) |binding| {
            binding.deinit();
            allocator.destroy(binding);
            self.attention_match_binding = null;
        }
        var binding_it = self.slot_map.valueIterator();
        while (binding_it.next()) |binding| {
            if (binding.selected_positions) |positions| allocator.free(positions);
            if (binding.selected_positions != null and binding.sequence_owned) self.discardPrivateSlot(binding.slot);
        }
        for (&self.leased_slots, 0..) |*leased, slot| {
            if (!leased.*) continue;
            self.reclaimSlot(slot);
        }
        self.slot_map.deinit(allocator);
        self.global_layer_slots.deinit(allocator);
        self.cyclic_page_table_cache.deinit(allocator);
        if (self.attention_match_runner) |*runner| runner.deinit();
        allocator.destroy(self);
    }

    fn drainForOwnerSnapshot(ctx: *anyopaque) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        const runtime = self.runtime;
        if (metal_runtime.hasActiveFrame(runtime)) try metal_runtime.cancelFrame(runtime);
        if (metal_runtime.hasSubmittedFrame(runtime)) try metal_runtime.waitFrame(runtime);
        for (self.leased_slots, 0..) |leased, slot| {
            if (!leased) continue;
            try self.reconcileSlotAccounting(slot, 0);
        }
    }

    fn releaseSequenceOp(ctx: *anyopaque, sequence_id: storage_runtime.SequenceId) void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.attention_match_binding) |binding| if (binding.sequence == sequence_id) {
            binding.deinit();
            self.allocator.destroy(binding);
            self.attention_match_binding = null;
        };
        self.releaseSequenceSlots(sequence_id);
    }

    fn discardExactCloneSourceOp(ctx: *anyopaque, sequence_id: storage_runtime.SequenceId) anyerror!storage_runtime.ExactCloneSourceDiscard {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (sequence_id == 0 or self.attention_match_binding != null or
            metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime))
            return error.UnsafeExactCloneSourceDiscard;
        var referenced: [metal_runtime.attention_span_slot_capacity]bool = @splat(false);
        var keys: [metal_runtime.attention_span_slot_capacity]SlotKey = undefined;
        var key_count: usize = 0;
        var it = self.slot_map.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.sequence_id != sequence_id or entry.value_ptr.selected_positions != null or
                entry.value_ptr.slot >= referenced.len or key_count == keys.len)
                return error.UnsafeExactCloneSourceDiscard;
            keys[key_count] = entry.key_ptr.*;
            key_count += 1;
            referenced[entry.value_ptr.slot] = true;
        }
        if (key_count == 0) return error.UnsafeExactCloneSourceDiscard;
        var globals = self.global_layer_slots.iterator();
        while (globals.next()) |entry| {
            if (entry.value_ptr.* >= referenced.len or !referenced[entry.value_ptr.*])
                return error.UnsafeExactCloneSourceDiscard;
        }
        var slot_count: usize = 0;
        for (self.leased_slots, 0..) |leased, slot| {
            if (leased != referenced[slot]) return error.UnsafeExactCloneSourceDiscard;
            if (leased) slot_count += 1;
        }
        if (slot_count == 0) return error.UnsafeExactCloneSourceDiscard;
        const before = self.allocationSnapshot();
        for (keys[0..key_count]) |key| {
            const removed = self.slot_map.fetchRemove(key) orelse return error.UnsafeExactCloneSourceDiscard;
            if (removed.value.selected_positions) |positions| self.allocator.free(positions);
        }
        self.global_layer_slots.clearRetainingCapacity();
        for (referenced, 0..) |selected, slot| if (selected) try self.discardPrivateSlotRequired(slot);
        const after = self.allocationSnapshot();
        if (after.live_bytes != 0 or before.live_bytes < after.live_bytes)
            return error.UnsafeExactCloneSourceDiscard;
        return .{
            .slots = slot_count,
            .live_bytes_before = before.live_bytes,
            .live_bytes_after = after.live_bytes,
            .physically_released_bytes = before.live_bytes - after.live_bytes,
        };
    }

    /// The storage/runtime ABI can stage and atomically publish compacted K/V
    /// rows with an explicit absolute-position table. Production admission is
    /// a separate pipeline qualification decision; this bit only describes
    /// the implemented backend capability.
    fn supportsSelectedGlobalKv(_: *const anyopaque) bool {
        return true;
    }

    fn truncateSequenceOp(
        ctx: *anyopaque,
        sequence_id: storage_runtime.SequenceId,
        retained_token_count: usize,
    ) void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        self.attention_match_revision +|= 1;
        if (self.attention_match_binding) |binding| if (binding.sequence == sequence_id) {
            if (binding.invalid) return;
            const previous = binding.device.logical_length;
            binding.device.truncate(@intCast(retained_token_count), .{}) catch {
                binding.invalid = true;
                return;
            };
            binding.rollback_tokens +|= @intCast(previous - @as(u64, @intCast(retained_token_count)));
            return;
        };
        var it = self.slot_map.iterator();
        while (it.next()) |entry| {
            if (entry.key_ptr.sequence_id != sequence_id) continue;
            entry.value_ptr.truncateTo(retained_token_count);
        }
    }

    fn gatherLayerKv(
        ctx: *anyopaque,
        gather: storage_runtime.KvLayerGather,
        k_out: []f32,
        v_out: []f32,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        const num_kv_heads = gather.num_kv_heads;
        const head_dim = gather.head_dim;
        const token_width: usize = @as(usize, num_kv_heads) * @as(usize, head_dim);
        const expected_elems = gather.token_count * token_width;
        if (k_out.len < expected_elems or v_out.len < expected_elems) return error.InvalidKvShape;

        if (traceKvGather()) std.debug.print("kv-read: seq={d} layer={d} tokens={d}\n", .{ gather.sequence_id, gather.layer_index, gather.token_count });
        const binding = self.slot_map.get(.{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        }) orelse return error.DeviceReadFallback;
        if (binding.selected_positions != null) return error.SelectedKvRequiresPagedAttention;
        const slot = binding.slot;
        if (binding.ring_page_count > 0) return error.RingKvRequiresPagedAttention;
        if (!binding.logical_contiguous) return error.DeviceReadFallback;
        if (binding.written_tokens < gather.token_count) return error.DeviceReadFallback;
        const info = try self.slotInfo(slot);

        const k_handle = info.encoded_key_handle orelse return error.DeviceReadFallback;
        const v_handle = info.v_handle orelse return error.DeviceReadFallback;

        const key_row_bytes = info.key_row_bytes;
        const v_row_stride = info.v_row_stride;
        const expect_v_stride = token_width;
        if (v_row_stride != expect_v_stride) return error.DeviceReadFallback;

        // For f16 format values are stored as halfs (2 bytes each), for all
        // other formats values are stored as plain f32.
        const v_element_bytes: usize = if (self.format == .f16) @sizeOf(u16) else @sizeOf(f32);
        const key_bytes = gather.token_count * key_row_bytes;
        const v_byte_count = gather.token_count * v_row_stride * v_element_bytes;
        const physical_base = binding.physical_base_tokens;
        const key_byte_offset = physical_base * key_row_bytes;
        const v_byte_offset = physical_base * v_row_stride * v_element_bytes;
        if (key_byte_offset + key_bytes > info.encoded_key_capacity or v_byte_offset + v_byte_count > info.v_capacity) return error.DeviceReadFallback;

        const k_staging = try self.allocator.alloc(u8, key_bytes);
        defer self.allocator.free(k_staging);
        const v_staging = try self.allocator.alloc(u8, v_byte_count);
        defer self.allocator.free(v_staging);

        if (metal_runtime.termite_metal_buffer_download(self.runtime, k_handle, key_byte_offset, @ptrCast(k_staging.ptr), key_bytes) != 0) {
            return error.MetalDeviceReadFailed;
        }
        if (metal_runtime.termite_metal_buffer_download(self.runtime, v_handle, v_byte_offset, @ptrCast(v_staging.ptr), v_byte_count) != 0) {
            return error.MetalDeviceReadFailed;
        }

        for (0..gather.token_count) |tok_idx| {
            const row_bytes = k_staging[tok_idx * key_row_bytes ..][0..key_row_bytes];
            const row_dst = k_out[tok_idx * token_width ..][0..token_width];
            switch (self.format) {
                .polar4 => try turboquant.decodePolar4Key(row_bytes, row_dst, num_kv_heads, head_dim),
                .turbo3 => {
                    const base_bytes = turboquant.turbo3KeyBytes(num_kv_heads, head_dim);
                    try turboquant.decodeTurbo3KeyWithResidual(row_bytes[0..base_bytes], row_bytes[base_bytes..], row_dst, num_kv_heads, head_dim);
                },
                .raw_f32 => {
                    const src_f32: [*]const f32 = @ptrCast(@alignCast(row_bytes.ptr));
                    @memcpy(row_dst, src_f32[0..token_width]);
                },
                .f16 => {
                    const src_f16: [*]const f16 = @ptrCast(@alignCast(row_bytes.ptr));
                    for (0..token_width) |i| row_dst[i] = @floatCast(src_f16[i]);
                },
                .int8_per_head => pool_mod.dequantizeInt8PerHeadToF32(row_bytes, row_dst, num_kv_heads, head_dim),
            }
            const v_row_src = v_staging[tok_idx * v_row_stride * v_element_bytes ..][0 .. v_row_stride * v_element_bytes];
            const v_row_dst = v_out[tok_idx * token_width ..][0..token_width];
            switch (self.format) {
                .f16 => {
                    const src_f16: [*]const f16 = @ptrCast(@alignCast(v_row_src.ptr));
                    for (0..token_width) |i| v_row_dst[i] = @floatCast(src_f16[i]);
                },
                .int8_per_head => {
                    const src_f32: [*]const f32 = @ptrCast(@alignCast(v_row_src.ptr));
                    @memcpy(v_row_dst, src_f32[0..token_width]);
                },
                else => {
                    const src_f32: [*]const f32 = @ptrCast(@alignCast(v_row_src.ptr));
                    @memcpy(v_row_dst, src_f32[0..token_width]);
                },
            }
        }
    }

    fn readExactAttentionMatchKv(
        ctx: *anyopaque,
        range: storage_runtime.attention_match_snapshot.Range,
        keys: []f32,
        values: []f32,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.attention_match_binding != null) return error.UnsupportedAttentionMatchRoute;
        if (self.format != .f16 or range.count == 0 or range.heads == 0 or range.dim == 0 or
            range.start > range.logical_length or range.count > range.logical_length - range.start)
            return error.UnsupportedKvSnapshotLayout;
        const width = std.math.mul(usize, range.heads, range.dim) catch return error.InvalidKvCheckpoint;
        const elements = std.math.mul(usize, range.count, width) catch return error.InvalidKvCheckpoint;
        if (keys.len != elements or values.len != elements or
            elements > storage_runtime.attention_match_snapshot.chunk_elements)
            return error.InvalidKvCheckpoint;
        const binding = self.slot_map.get(.{
            .sequence_id = range.sequence,
            .layer_index = range.layer,
        }) orelse return error.DeviceReadUnsupported;
        if (binding.selected_positions != null or binding.position_offset != 0 or
            binding.written_tokens != range.logical_length or
            (binding.ring_page_count == 0 and !binding.logical_contiguous) or
            (binding.ring_page_count != 0 and binding.logical_contiguous))
            return error.UnsupportedKvSnapshotLayout;
        const info = try self.slotInfo(binding.slot);
        const row_bytes = std.math.mul(usize, width, @sizeOf(f16)) catch return error.InvalidKvCheckpoint;
        if (info.key_row_bytes != row_bytes or info.v_row_stride != width or
            (binding.ring_page_count == 0 and info.tokens < range.logical_length))
            return error.UnsupportedKvSnapshotLayout;
        const byte_count = std.math.mul(usize, range.count, row_bytes) catch return error.InvalidKvCheckpoint;
        const encoded_keys = std.mem.sliceAsBytes(keys)[0..byte_count];
        const encoded_values = std.mem.sliceAsBytes(values)[0..byte_count];
        if (binding.ring_page_count == 0) {
            const first_row = std.math.add(usize, binding.physical_base_tokens, range.start) catch return error.InvalidKvCheckpoint;
            const byte_offset = std.math.mul(usize, first_row, row_bytes) catch return error.InvalidKvCheckpoint;
            if (byte_offset > info.encoded_key_capacity or byte_count > info.encoded_key_capacity - byte_offset or
                byte_offset > info.v_capacity or byte_count > info.v_capacity - byte_offset) return error.InvalidKvCheckpoint;
            if (metal_runtime.termite_metal_buffer_download(self.runtime, info.encoded_key_handle.?, byte_offset, @ptrCast(encoded_keys.ptr), byte_count) != 0 or
                metal_runtime.termite_metal_buffer_download(self.runtime, info.v_handle.?, byte_offset, @ptrCast(encoded_values.ptr), byte_count) != 0)
                return error.MetalDeviceReadFailed;
        } else {
            const visible = try qualificationVisibleRange(binding, range.logical_length, self.page_size_tokens);
            if (range.start < visible.start) return error.UnsupportedKvSnapshotLayout;
            var logical = range.start;
            var destination_row: usize = 0;
            while (destination_row < range.count) {
                const page_remaining = self.page_size_tokens - logical % self.page_size_tokens;
                const rows = @min(page_remaining, range.count - destination_row);
                const logical_page = logical / self.page_size_tokens;
                const physical_page = storage_runtime.physicalPageForLogical(logical_page, binding.ring_page_count);
                const physical_row = physical_page * self.page_size_tokens + logical % self.page_size_tokens;
                const source_offset = try std.math.mul(usize, physical_row, row_bytes);
                const segment_bytes = try std.math.mul(usize, rows, row_bytes);
                const destination_offset = try std.math.mul(usize, destination_row, row_bytes);
                if (source_offset > info.encoded_key_capacity or segment_bytes > info.encoded_key_capacity - source_offset or
                    source_offset > info.v_capacity or segment_bytes > info.v_capacity - source_offset) return error.InvalidKvCheckpoint;
                if (metal_runtime.termite_metal_buffer_download(self.runtime, info.encoded_key_handle.?, source_offset, encoded_keys[destination_offset..].ptr, segment_bytes) != 0 or
                    metal_runtime.termite_metal_buffer_download(self.runtime, info.v_handle.?, source_offset, encoded_values[destination_offset..].ptr, segment_bytes) != 0)
                    return error.MetalDeviceReadFailed;
                logical += rows;
                destination_row += rows;
            }
        }
        const readback_bytes = std.math.mul(usize, byte_count, 2) catch return error.InvalidKvCheckpoint;
        if (metal_runtime.termite_metal_decode_runtime_attention_span_slot_note_readback(
            self.runtime,
            binding.slot,
            readback_bytes,
        ) != 0) return error.DeviceAllocationSnapshotUnavailable;
        // Expand backwards so the caller's bounded f32 output is also the
        // only staging allocation needed for exact f16 readback.
        for ([_][]f32{ keys, values }) |output| {
            const half: [*]const f16 = @ptrCast(@alignCast(output.ptr));
            var index = output.len;
            while (index != 0) {
                index -= 1;
                output[index] = @floatCast(half[index]);
            }
        }
    }

    fn gatherLayerKeys(
        ctx: *anyopaque,
        gather: storage_runtime.KvLayerGather,
        k_out: []f32,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        const token_width: usize = @as(usize, gather.num_kv_heads) * @as(usize, gather.head_dim);
        if (k_out.len < gather.token_count * token_width) return error.InvalidKvShape;
        const binding = self.slot_map.get(.{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        }) orelse return error.DeviceReadFallback;
        if (binding.ring_page_count > 0 or !binding.logical_contiguous or binding.written_tokens < gather.token_count)
            return error.DeviceReadFallback;
        const info = try self.slotInfo(binding.slot);
        const handle = info.encoded_key_handle orelse return error.DeviceReadFallback;
        const byte_count = gather.token_count * info.key_row_bytes;
        const byte_offset = binding.physical_base_tokens * info.key_row_bytes;
        if (byte_offset + byte_count > info.encoded_key_capacity) return error.DeviceReadFallback;
        const encoded = try self.allocator.alloc(u8, byte_count);
        defer self.allocator.free(encoded);
        if (metal_runtime.termite_metal_buffer_download(self.runtime, handle, byte_offset, @ptrCast(encoded.ptr), byte_count) != 0)
            return error.MetalDeviceReadFailed;
        for (0..gather.token_count) |row| {
            const src = encoded[row * info.key_row_bytes ..][0..info.key_row_bytes];
            const dst = k_out[row * token_width ..][0..token_width];
            switch (self.format) {
                .polar4 => try turboquant.decodePolar4Key(src, dst, gather.num_kv_heads, gather.head_dim),
                .turbo3 => {
                    const base_bytes = turboquant.turbo3KeyBytes(gather.num_kv_heads, gather.head_dim);
                    try turboquant.decodeTurbo3KeyWithResidual(src[0..base_bytes], src[base_bytes..], dst, gather.num_kv_heads, gather.head_dim);
                },
                .raw_f32 => {
                    const values: [*]const f32 = @ptrCast(@alignCast(src.ptr));
                    @memcpy(dst, values[0..token_width]);
                },
                .f16 => {
                    const values: [*]const f16 = @ptrCast(@alignCast(src.ptr));
                    for (0..token_width) |i| dst[i] = @floatCast(values[i]);
                },
                .int8_per_head => pool_mod.dequantizeInt8PerHeadToF32(src, dst, gather.num_kv_heads, gather.head_dim),
            }
        }
    }

    fn gatherLayerKvDevice(
        ctx: *anyopaque,
        gather: storage_runtime.DeviceKvLayerGather,
    ) anyerror!storage_runtime.DeviceKvLayer {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.format != .raw_f32) return error.DeviceReadFallback;
        const token_width: usize = @as(usize, gather.num_kv_heads) * @as(usize, gather.head_dim);
        const byte_len = gather.token_count * token_width * @sizeOf(f32);

        if (traceKvGather()) std.debug.print("kv-read: seq={d} layer={d} tokens={d}\n", .{ gather.sequence_id, gather.layer_index, gather.token_count });
        const binding = self.slot_map.get(.{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        }) orelse return error.DeviceReadFallback;
        // DeviceKvLayer has no absolute-position channel. Returning a compact
        // span would silently reinterpret selected rows as contiguous history.
        if (binding.selected_positions != null) return error.SelectedKvRequiresPagedAttention;
        const slot = binding.slot;
        if (binding.ring_page_count > 0) return error.RingKvRequiresPagedAttention;
        if (!binding.logical_contiguous) return error.DeviceReadFallback;
        if (binding.written_tokens < gather.token_count) return error.DeviceReadFallback;
        const info = try self.slotInfo(slot);
        if (info.key_row_bytes != token_width * @sizeOf(f32)) return error.DeviceReadFallback;
        if (info.v_row_stride != token_width) return error.DeviceReadFallback;

        const k_handle = info.encoded_key_handle orelse return error.DeviceReadFallback;
        const v_handle = info.v_handle orelse return error.DeviceReadFallback;
        const byte_offset = binding.physical_base_tokens * token_width * @sizeOf(f32);
        if (byte_offset + byte_len > info.encoded_key_capacity or byte_offset + byte_len > info.v_capacity) return error.DeviceReadFallback;

        return .{
            .runtime = @ptrCast(self.runtime),
            .k = .{
                .handle = k_handle,
                .byte_offset = byte_offset,
                .byte_len = byte_len,
            },
            .v = .{
                .handle = v_handle,
                .byte_offset = byte_offset,
                .byte_len = byte_len,
            },
            .token_count = if (binding.selected_positions) |positions| positions.len else gather.token_count,
            .row_width = token_width,
            .position_offset = info.position_offset,
            .value_element_bytes = @sizeOf(f32),
        };
    }

    fn pagedLayerKvDevice(
        ctx: *anyopaque,
        gather: storage_runtime.DeviceKvLayerGather,
    ) anyerror!storage_runtime.DevicePagedKvLayer {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        const num_kv_heads = gather.num_kv_heads;
        const head_dim = gather.head_dim;
        const token_width: usize = @as(usize, num_kv_heads) * @as(usize, head_dim);
        const int8_row_bytes: usize = @as(usize, num_kv_heads) * (@as(usize, head_dim) + @sizeOf(f32));
        const key_row_bytes: usize = switch (self.format) {
            .polar4 => turboquant.polar4KeyBytes(num_kv_heads, head_dim),
            .turbo3 => turboquant.turbo3KeyBytes(num_kv_heads, head_dim) + turboquant.turbo3ResidualBytes(num_kv_heads, head_dim),
            .raw_f32 => token_width * @sizeOf(f32),
            .f16 => token_width * @sizeOf(u16),
            .int8_per_head => int8_row_bytes,
        };
        const base_key_row_bytes: usize = switch (self.format) {
            .polar4 => turboquant.polar4KeyBytes(num_kv_heads, head_dim),
            .turbo3 => turboquant.turbo3KeyBytes(num_kv_heads, head_dim),
            .raw_f32 => token_width * @sizeOf(f32),
            .f16 => token_width * @sizeOf(u16),
            .int8_per_head => int8_row_bytes,
        };
        const key = SlotKey{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        };
        const active_frame = metal_runtime.hasActiveFrame(self.runtime);
        const binding = self.slot_map.get(key) orelse return error.DeviceReadFallback;
        const slot = binding.slot;
        const ring_page_count = binding.ring_page_count;
        const info_opt = self.slotInfo(slot) catch |err| blk: {
            if (!active_frame) return err;
            break :blk null;
        };
        // A fused operation may expose a reserved slot before it encodes the
        // current suffix, but only while an ordered frame is active. Every
        // token before that explicitly declared suffix must already be
        // committed; otherwise the slot contains unreadable capacity.
        if (gather.pending_suffix_token_count != 0 and !active_frame) return error.DeviceReadFallback;
        if (!binding.coversBeforePendingSuffix(gather.token_count, gather.pending_suffix_token_count)) {
            return error.DeviceReadFallback;
        }
        if (info_opt) |info| {
            if (info.key_row_bytes != 0 and info.key_row_bytes != key_row_bytes) return error.DeviceReadFallback;
            if (info.v_row_stride != 0 and info.v_row_stride != token_width) return error.DeviceReadFallback;
        }
        return .{
            .runtime = @ptrCast(self.runtime),
            .slot = slot,
            .format = @backingInt(self.format),
            .token_count = if (binding.selected_positions) |positions| positions.len else gather.token_count,
            .key_row_bytes = key_row_bytes,
            .base_key_row_bytes = base_key_row_bytes,
            .v_row_stride = token_width,
            .page_size_tokens = self.page_size_tokens,
            .ring_page_count = ring_page_count,
            .position_offset = binding.position_offset,
            .key_positions = binding.selected_positions,
        };
    }

    /// Fetch a slot's current state + device buffer handles so a caller can
    /// download the encoded bytes and decode them on the host. Used by the
    /// gather path — the read logic itself lives in the caller rather than
    /// this module so that the pool's existing decode routines can be reused.
    pub fn slotInfo(self: *const MetalKvStorage, slot: usize) !SlotInfo {
        var info: SlotInfo = .{};
        const rc = metal_runtime.termite_metal_decode_runtime_attention_span_slot_info(
            self.runtime,
            slot,
            &info.encoded_key_handle,
            &info.encoded_key_capacity,
            &info.v_handle,
            &info.v_capacity,
            &info.tokens,
            &info.key_row_bytes,
            &info.v_row_stride,
            &info.position_offset,
        );
        if (rc != 0) return error.InvalidSlot;
        return info;
    }

    /// Return the physical slot backing a sequence/layer view. Intended for
    /// diagnostics and invariant tests; callers must not treat slot identity
    /// as stable beyond the owning hook's lifetime.
    pub fn boundSlot(self: *const MetalKvStorage, sequence_id: storage_runtime.SequenceId, layer_index: usize) ?usize {
        const binding = self.slot_map.get(.{
            .sequence_id = sequence_id,
            .layer_index = std.math.cast(u32, layer_index) orelse return null,
        }) orelse return null;
        return binding.slot;
    }

    pub const QualificationFingerprintRequest = struct {
        context_instance_id: u64,
        sequence_id: storage_runtime.SequenceId,
        logical_tokens: u32,
        position_offset: u32,
        logical_blocks: []const block.KvBlockId,
        /// Actual KV owners in canonical loaded-layout order. Pool
        /// `num_layers_packed` is a physical stride/capacity and includes
        /// shared consumers that never own a storage binding.
        owner_layers: []const u32,
    };

    pub const QualificationInventoryEntry = struct {
        layer: u32,
        window: u32,
        rows: u32,
        capacity_rows: u32,
        start_row: u32,
        kv_heads: u32,
        head_dim: u32,
        key_row_bytes: u64,
        value_row_bytes: u64,
        key_capacity_bytes: u64,
        value_capacity_bytes: u64,
        installed_physical_bytes: u64,
    };

    pub const QualificationInventoryOwned = struct {
        allocator: std.mem.Allocator,
        context_instance_id: u64,
        sequence_id: storage_runtime.SequenceId,
        source_revision: u64,
        logical_tokens: u32,
        entries: []QualificationInventoryEntry,
        installed_physical_bytes: u64,
        descriptor_sha256: [32]u8,

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.entries);
            self.* = undefined;
        }
    };

    pub const RawInventorySeal = struct {
        token_sha256: [32]u8,
        shape_sha256: [32]u8,
        binary_sha256: [32]u8,
        source_freeze_sha256: [32]u8,
        pub fn validate(self: @This()) !void {
            inline for (.{ self.token_sha256, self.shape_sha256, self.binary_sha256, self.source_freeze_sha256 }) |digest|
                if (std.mem.allEqual(u8, &digest, 0)) return error.UnqualifiedCompactionArtifact;
        }
    };

    pub const RawHeadAuthority = struct {
        sequence_id: storage_runtime.SequenceId,
        source_revision: u64,
        logical_tokens: u32,
        logical_blocks: []const block.KvBlockId,
        donor_layer: u32,
        head_index: u32,
        window: u32,
    };

    /// Bounded one-head readback from the genuine ordinary pre-fit source.
    /// The admission covers this returned head plus the mandatory downstream
    /// clone, so OpenFn cannot create an unaccounted second live copy.
    pub fn downloadRawExactHead(
        self: *MetalKvStorage,
        allocator: std.mem.Allocator,
        authority: RawHeadAuthority,
        max_two_copy_host_bytes: usize,
        control: AttentionMatchControl,
    ) !attention_match_storage.DownloadedHead {
        try control.check();
        if (self.format != .f16 or self.attention_match_binding != null or authority.source_revision == 0 or
            authority.source_revision != self.attention_match_revision or authority.logical_tokens == 0 or
            authority.head_index >= self.num_kv_heads or max_two_copy_host_bytes == 0)
            return error.AttentionMatchSourceChanged;
        if (metal_runtime.hasActiveFrame(self.runtime)) return error.AttentionMatchSourceChanged;
        if (metal_runtime.hasSubmittedFrame(self.runtime)) try metal_runtime.waitFrame(self.runtime);
        if (metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime)) return error.AttentionMatchSourceChanged;
        const binding = self.slot_map.get(.{ .sequence_id = authority.sequence_id, .layer_index = authority.donor_layer }) orelse
            return error.AttentionMatchSourceChanged;
        if (binding.written_tokens != authority.logical_tokens or binding.sliding_window_tokens != authority.window)
            return error.AttentionMatchSourceChanged;
        const info = try self.slotInfo(binding.slot);
        const visible = try qualificationVisibleRange(binding, authority.logical_tokens, self.page_size_tokens);
        const rows = visible.count;
        const dim: usize = self.head_dim;
        if (dim == 0 or dim > 512) return error.AttentionMatchSourceChanged;
        const vector_count = try std.math.mul(usize, rows, dim);
        const metadata_bytes = try std.math.mul(usize, rows, 2 * @sizeOf(u64) + @sizeOf(f32));
        const vector_bytes = try std.math.mul(usize, vector_count, 2 * @sizeOf(f32));
        const one_copy_bytes = try std.math.add(usize, metadata_bytes, vector_bytes);
        const two_copy_bytes = try std.math.mul(usize, one_copy_bytes, 2);
        if (two_copy_bytes > max_two_copy_host_bytes) return error.CompactionWorkspaceLimit;
        const positions = try allocator.alloc(u64, rows);
        errdefer allocator.free(positions);
        const coverage = try allocator.alloc(u64, rows);
        errdefer allocator.free(coverage);
        const bias = try allocator.alloc(f32, rows);
        errdefer allocator.free(bias);
        const keys = try allocator.alloc(f32, vector_count);
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, vector_count);
        errdefer allocator.free(values);
        const spans = try allocator.alloc(checkpoint.packed_storage.Span, 0);
        errdefer allocator.free(spans);
        const compressed_keys = try allocator.alloc(u8, 0);
        errdefer allocator.free(compressed_keys);
        const compressed_values = try allocator.alloc(u8, 0);
        errdefer allocator.free(compressed_values);
        const source_row_values = try std.math.mul(usize, self.num_kv_heads, dim);
        const source_row_bytes = try std.math.mul(usize, source_row_values, @sizeOf(f16));
        if (info.key_row_bytes != source_row_bytes or info.v_row_stride != source_row_values or
            info.encoded_key_handle == null or info.v_handle == null)
            return error.AttentionMatchSourceChanged;
        var raw_key: [512]f16 = undefined;
        var raw_value: [512]f16 = undefined;
        for (0..rows) |row| {
            if (row % 64 == 0) try control.check();
            const logical_row = visible.start + row;
            const physical_row = try qualificationPhysicalRow(binding, authority.logical_blocks, self.page_size_tokens, logical_row);
            const base = try std.math.add(usize, try std.math.mul(usize, physical_row, source_row_bytes), try std.math.mul(usize, authority.head_index, dim * @sizeOf(f16)));
            const head_bytes = try std.math.mul(usize, dim, @sizeOf(f16));
            if (base > info.encoded_key_capacity or head_bytes > info.encoded_key_capacity - base or
                base > info.v_capacity or head_bytes > info.v_capacity - base)
                return error.AttentionMatchSourceChanged;
            if (metal_runtime.termite_metal_buffer_download(self.runtime, info.encoded_key_handle.?, base, @ptrCast(raw_key[0..dim].ptr), head_bytes) != 0 or
                metal_runtime.termite_metal_buffer_download(self.runtime, info.v_handle.?, base, @ptrCast(raw_value[0..dim].ptr), head_bytes) != 0)
                return error.DeviceReadFallback;
            positions[row] = logical_row;
            coverage[row] = logical_row + 1;
            bias[row] = 0;
            for (keys[row * dim ..][0..dim], values[row * dim ..][0..dim], raw_key[0..dim], raw_value[0..dim]) |*key, *value, source_key, source_value| {
                key.* = @floatCast(source_key);
                value.* = @floatCast(source_value);
            }
        }
        try control.check();
        if (self.attention_match_revision != authority.source_revision) return error.AttentionMatchSourceChanged;
        return .{ .allocator = allocator, .value = .{ .dim = @intCast(dim), .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias, .compressed = .{ .spans = spans, .keys = compressed_keys, .values = compressed_values } } };
    }

    /// Metadata-only inventory of the genuine pre-fit Metal KV source. This
    /// does not require or accept an installed attention-match candidate.
    pub fn qualificationInventoryDeviceHook(
        hook: storage_runtime.DeviceWriteHook,
        allocator: std.mem.Allocator,
        request: QualificationFingerprintRequest,
    ) !QualificationInventoryOwned {
        if (!builtin.is_test) return error.QualificationOnly;
        return inventoryDeviceHook(hook, allocator, request, null);
    }

    /// Production raw-prefill inventory seam. The caller must carry the
    /// authenticated token/shape/binary/freeze tuple; the storage independently
    /// validates every observed sequence, revision, geometry and byte count.
    pub fn rawStructuralInventoryDeviceHook(
        hook: storage_runtime.DeviceWriteHook,
        allocator: std.mem.Allocator,
        request: QualificationFingerprintRequest,
        seal: RawInventorySeal,
    ) !QualificationInventoryOwned {
        try seal.validate();
        return inventoryDeviceHook(hook, allocator, request, seal);
    }

    fn inventoryDeviceHook(
        hook: storage_runtime.DeviceWriteHook,
        allocator: std.mem.Allocator,
        request: QualificationFingerprintRequest,
        seal: ?RawInventorySeal,
    ) !QualificationInventoryOwned {
        if (hook.vtable != &hook_vtable) return error.OrdinaryExactMetalKvRequired;
        const self: *MetalKvStorage = @ptrCast(@alignCast(hook.ctx));
        if (request.context_instance_id == 0 or request.logical_tokens == 0 or request.position_offset != 0 or
            self.format != .f16 or metal_runtime.hasActiveFrame(self.runtime))
            return error.InvalidQualificationFingerprint;
        if (metal_runtime.hasSubmittedFrame(self.runtime)) try metal_runtime.waitFrame(self.runtime);
        if (metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime) or self.attention_match_revision == 0)
            return error.InvalidQualificationFingerprint;
        try validateQualificationOwnerLayers(request.owner_layers, self.global_layer_slots.count() + @as(usize, self.countSequenceOwnedLayers(request.sequence_id)));
        const entries = try allocator.alloc(QualificationInventoryEntry, request.owner_layers.len);
        errdefer allocator.free(entries);
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("antfly.ordinary-exact-context.structural-inventory.v1\x00");
        if (seal) |authority| {
            hash.update("raw-prefill\x00");
            hash.update(&authority.token_sha256);
            hash.update(&authority.shape_sha256);
            hash.update(&authority.binary_sha256);
            hash.update(&authority.source_freeze_sha256);
        } else hash.update("qualification-only\x00");
        try fingerprintU32(&hash, request.logical_tokens);
        try fingerprintU32(&hash, self.num_kv_heads);
        try fingerprintU32(&hash, self.head_dim);
        var physical: u64 = 0;
        for (request.owner_layers, entries) |layer, *entry| {
            const binding = self.slot_map.get(.{ .sequence_id = request.sequence_id, .layer_index = layer }) orelse return error.InvalidQualificationFingerprint;
            if (binding.written_tokens != request.logical_tokens) return error.InvalidQualificationFingerprint;
            const info = try self.slotInfo(binding.slot);
            const allocation = try self.slotAllocationInfo(binding.slot);
            if (allocation.leased == 0 or allocation.installed_live_bytes == 0 or
                allocation.active_frame_retained_bytes != 0 or allocation.submitted_frame_retained_bytes != 0)
                return error.InvalidQualificationFingerprint;
            const visible = try qualificationVisibleRange(binding, request.logical_tokens, self.page_size_tokens);
            const value_row_bytes = try std.math.mul(usize, info.v_row_stride, @sizeOf(f16));
            entry.* = .{
                .layer = layer,
                .window = @intCast(binding.sliding_window_tokens),
                .rows = @intCast(visible.count),
                .capacity_rows = @intCast(self.slot_buffer_capacity_tokens[binding.slot]),
                .start_row = @intCast(if (binding.ring_page_count == 0) 0 else visible.start % self.slot_buffer_capacity_tokens[binding.slot]),
                .kv_heads = self.num_kv_heads,
                .head_dim = self.head_dim,
                .key_row_bytes = @intCast(info.key_row_bytes),
                .value_row_bytes = @intCast(value_row_bytes),
                .key_capacity_bytes = @intCast(info.encoded_key_capacity),
                .value_capacity_bytes = @intCast(info.v_capacity),
                .installed_physical_bytes = allocation.installed_live_bytes,
            };
            physical = std.math.add(u64, physical, entry.installed_physical_bytes) catch return error.QualificationFingerprintOverflow;
            inline for (.{ entry.layer, entry.window, entry.rows, entry.capacity_rows, entry.start_row, entry.kv_heads, entry.head_dim }) |value| try fingerprintU32(&hash, value);
            inline for (.{ entry.key_row_bytes, entry.value_row_bytes, entry.key_capacity_bytes, entry.value_capacity_bytes, entry.installed_physical_bytes }) |value| {
                var frame: [8]u8 = undefined;
                std.mem.writeInt(u64, &frame, value, .little);
                hash.update(&frame);
            }
        }
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return .{ .allocator = allocator, .context_instance_id = request.context_instance_id, .sequence_id = request.sequence_id, .source_revision = self.attention_match_revision, .logical_tokens = request.logical_tokens, .entries = entries, .installed_physical_bytes = physical, .descriptor_sha256 = digest };
    }

    /// Qualification-only raw fingerprint of an ordinary exact Metal KV
    /// context. Serving never calls this method: it deliberately drains and
    /// downloads every logically visible K/V byte into bounded page staging.
    /// Reserved capacity and overwritten SWA rows are excluded. Payload order
    /// is layer, logical token, K row, V row.
    pub fn qualificationFingerprintDeviceHook(
        hook: storage_runtime.DeviceWriteHook,
        allocator: std.mem.Allocator,
        request: QualificationFingerprintRequest,
    ) !context_teacher_forced.QualityFingerprint {
        if (!builtin.is_test) return error.QualificationOnly;
        if (hook.vtable != &hook_vtable) return error.OrdinaryExactMetalKvRequired;
        const self: *MetalKvStorage = @ptrCast(@alignCast(hook.ctx));
        return self.qualificationFingerprint(allocator, request);
    }

    fn qualificationPhysicalRow(
        binding: SlotBinding,
        logical_blocks: []const block.KvBlockId,
        page_size: usize,
        logical_token: usize,
    ) !usize {
        if (page_size == 0) return error.InvalidQualificationFingerprint;
        const logical_page = logical_token / page_size;
        const in_page = logical_token % page_size;
        const physical_page = if (binding.ring_page_count > 0)
            storage_runtime.physicalPageForLogical(logical_page, binding.ring_page_count)
        else blk: {
            if (logical_page >= logical_blocks.len) return error.InvalidQualificationFingerprint;
            break :blk @as(usize, logical_blocks[logical_page]);
        };
        return std.math.add(
            usize,
            try std.math.mul(usize, physical_page, page_size),
            in_page,
        ) catch return error.QualificationFingerprintOverflow;
    }

    fn qualificationFingerprint(
        self: *MetalKvStorage,
        allocator: std.mem.Allocator,
        request: QualificationFingerprintRequest,
    ) !context_teacher_forced.QualityFingerprint {
        const owner_count = request.owner_layers.len;
        if (self.format != .f16 or request.context_instance_id == 0 or request.logical_tokens == 0 or
            request.position_offset != 0)
            return error.InvalidQualificationFingerprint;
        try validateQualificationOwnerLayers(
            request.owner_layers,
            self.global_layer_slots.count() + @as(usize, self.countSequenceOwnedLayers(request.sequence_id)),
        );
        if (metal_runtime.hasActiveFrame(self.runtime)) return error.QualificationFingerprintFrameActive;
        if (metal_runtime.hasSubmittedFrame(self.runtime)) try metal_runtime.waitFrame(self.runtime);
        if (metal_runtime.hasActiveFrame(self.runtime) or metal_runtime.hasSubmittedFrame(self.runtime))
            return error.QualificationFingerprintFrameActive;

        const logical_tokens: usize = request.logical_tokens;
        const page_size: usize = self.page_size_tokens;
        if (page_size == 0) return error.InvalidQualificationFingerprint;
        const needed_blocks = std.math.divCeil(usize, logical_tokens, page_size) catch
            return error.QualificationFingerprintOverflow;
        if (request.logical_blocks.len < needed_blocks) return error.InvalidQualificationFingerprint;

        var descriptor = std.crypto.hash.sha2.Sha256.init(.{});
        descriptor.update("antfly.ordinary-exact-context.descriptor.v1\x00");
        try fingerprintU32(&descriptor, @backingInt(self.format));
        try fingerprintU32(&descriptor, self.num_kv_heads);
        try fingerprintU32(&descriptor, self.head_dim);
        try fingerprintU32(&descriptor, page_size);
        try fingerprintU32(&descriptor, owner_count);
        try fingerprintU32(&descriptor, logical_tokens);
        try fingerprintU32(&descriptor, request.position_offset);

        for (request.owner_layers) |layer_u32| {
            const layer: usize = @intCast(layer_u32);
            const binding = self.slot_map.get(.{
                .sequence_id = request.sequence_id,
                .layer_index = layer_u32,
            }) orelse return error.InvalidQualificationFingerprint;
            if (binding.written_tokens != logical_tokens) return error.InvalidQualificationFingerprint;
            const info = try self.slotInfo(binding.slot);
            const v_element_bytes: usize = if (self.format == .f16) @sizeOf(u16) else @sizeOf(f32);
            const v_row_bytes = std.math.mul(usize, info.v_row_stride, v_element_bytes) catch
                return error.QualificationFingerprintOverflow;
            if (info.key_row_bytes == 0 or v_row_bytes == 0 or info.encoded_key_handle == null or info.v_handle == null)
                return error.InvalidQualificationFingerprint;
            if (info.tokens != binding.written_tokens or info.position_offset != binding.position_offset)
                return error.InvalidQualificationFingerprint;
            const visible = try qualificationVisibleRange(binding, logical_tokens, page_size);
            const visible_tokens = visible.count;
            const logical_start = visible.start;
            try fingerprintQualificationLayer(
                &descriptor,
                layer,
                binding,
                logical_start,
                visible_tokens,
                logical_tokens,
                page_size,
                info.key_row_bytes,
                info.v_row_stride,
                v_element_bytes,
            );
        }
        var descriptor_sha256: [32]u8 = undefined;
        descriptor.final(&descriptor_sha256);

        var payload = std.crypto.hash.sha2.Sha256.init(.{});
        payload.update("antfly.ordinary-exact-context.payload.v1\x00");
        var readback_bytes: u64 = 0;
        for (request.owner_layers) |layer_u32| {
            const binding = self.slot_map.get(.{
                .sequence_id = request.sequence_id,
                .layer_index = layer_u32,
            }).?;
            const info = try self.slotInfo(binding.slot);
            const key_handle = info.encoded_key_handle.?;
            const v_handle = info.v_handle.?;
            const v_element_bytes: usize = if (self.format == .f16) @sizeOf(u16) else @sizeOf(f32);
            const v_row_bytes = try std.math.mul(usize, info.v_row_stride, v_element_bytes);
            const visible = try qualificationVisibleRange(binding, logical_tokens, page_size);
            var logical_token = visible.start;
            while (logical_token < logical_tokens) {
                const page_end = @min(logical_tokens, (logical_token / page_size + 1) * page_size);
                const row_count = page_end - logical_token;
                const physical_row = try qualificationPhysicalRow(binding, request.logical_blocks, page_size, logical_token);
                const key_bytes = try std.math.mul(usize, row_count, info.key_row_bytes);
                const value_bytes = try std.math.mul(usize, row_count, v_row_bytes);
                const key_offset = try std.math.mul(usize, physical_row, info.key_row_bytes);
                const value_offset = try std.math.mul(usize, physical_row, v_row_bytes);
                if (key_offset > info.encoded_key_capacity or key_bytes > info.encoded_key_capacity - key_offset or
                    value_offset > info.v_capacity or value_bytes > info.v_capacity - value_offset)
                    return error.InvalidQualificationFingerprint;
                const keys = try allocator.alloc(u8, key_bytes);
                defer allocator.free(keys);
                const values = try allocator.alloc(u8, value_bytes);
                defer allocator.free(values);
                if (metal_runtime.termite_metal_buffer_download(self.runtime, key_handle, key_offset, @ptrCast(keys.ptr), keys.len) != 0 or
                    metal_runtime.termite_metal_buffer_download(self.runtime, v_handle, value_offset, @ptrCast(values.ptr), values.len) != 0)
                    return error.QualificationFingerprintReadbackFailed;
                var row: usize = 0;
                while (row < row_count) : (row += 1) {
                    payload.update(keys[row * info.key_row_bytes ..][0..info.key_row_bytes]);
                    payload.update(values[row * v_row_bytes ..][0..v_row_bytes]);
                }
                const segment_bytes = std.math.add(usize, key_bytes, value_bytes) catch return error.QualificationFingerprintOverflow;
                readback_bytes = std.math.add(u64, readback_bytes, std.math.cast(u64, segment_bytes) orelse
                    return error.QualificationFingerprintOverflow) catch return error.QualificationFingerprintOverflow;
                logical_token = page_end;
            }
        }
        payload.update(&descriptor_sha256);
        var payload_sha256: [32]u8 = undefined;
        payload.final(&payload_sha256);
        if (readback_bytes == 0) return error.QualificationFingerprintOverflow;
        return .{
            .context_instance_id = request.context_instance_id,
            .publication_epoch = 1,
            .logical_tokens = request.logical_tokens,
            .payload_sha256 = payload_sha256,
            .descriptor_sha256 = descriptor_sha256,
            .diagnostic_kv_readback_bytes = readback_bytes,
        };
    }

    fn countSequenceOwnedLayers(self: *const MetalKvStorage, sequence_id: storage_runtime.SequenceId) u32 {
        var count: u32 = 0;
        var iterator = self.slot_map.iterator();
        while (iterator.next()) |entry| {
            if (entry.key_ptr.sequence_id == sequence_id and entry.value_ptr.sequence_owned) count +|= 1;
        }
        return count;
    }

    pub const SlotInfo = struct {
        encoded_key_handle: ?*anyopaque = null,
        encoded_key_capacity: usize = 0,
        v_handle: ?*anyopaque = null,
        v_capacity: usize = 0,
        tokens: usize = 0,
        key_row_bytes: usize = 0,
        v_row_stride: usize = 0,
        position_offset: usize = 0,
    };
};

test "attention_match exact source validates actual local and global slot widths" {
    var heads = [_]attention_match_storage.Head{
        .{ .layer = 0, .head = 0, .dim = 256, .logical_length = 3 },
        .{ .layer = 0, .head = 1, .dim = 256, .logical_length = 3 },
    };
    var info: MetalKvStorage.SlotInfo = .{
        .encoded_key_handle = @ptrFromInt(0x1000),
        .v_handle = @ptrFromInt(0x2000),
        .tokens = 3,
        .key_row_bytes = 1024,
        .v_row_stride = 512,
        .encoded_key_capacity = 3072,
        .v_capacity = 3072,
    };
    try MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3);
    for (&heads) |*head| head.dim = 512;
    // A wider candidate must not reinterpret the smaller source slot.
    try std.testing.expectError(error.InvalidKvCheckpoint, MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3));
    info.key_row_bytes = 2048;
    info.v_row_stride = 1024;
    info.encoded_key_capacity = 6144;
    info.v_capacity = 6144;
    try MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3);
    heads[0].dim = 256;
    try std.testing.expectError(error.InvalidKvCheckpoint, MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3));
    heads[0].dim = 512;
    info.v_capacity -= 1;
    try std.testing.expectError(error.InvalidKvCheckpoint, MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3));
    info.v_capacity += 1;
    info.tokens -= 1;
    try std.testing.expectError(error.InvalidKvCheckpoint, MetalKvStorage.validateExactAttentionMatchGeometry(2, 512, &heads, info, 3));
}

test "context attention match local capacity policy locks on first owner allocation" {
    var legacy: AttentionMatchLocalCapacityState = .{};
    try std.testing.expect(!legacy.lock().exact_local_window_capacity);
    try legacy.configure(.legacy_spare512_v1);
    try std.testing.expectError(error.AttentionMatchLocalCapacityPolicyChanged, legacy.configure(.exact_window_causal_union_v1));
    var exact: AttentionMatchLocalCapacityState = .{};
    try exact.configure(.exact_window_causal_union_v1);
    try std.testing.expect(exact.lock().exact_local_window_capacity);
    try exact.configure(.exact_window_causal_union_v1);
    try std.testing.expectError(error.AttentionMatchLocalCapacityPolicyChanged, exact.configure(.legacy_spare512_v1));
}

test "Metal KV rollout flags recognize common false values" {
    try std.testing.expect(!MetalKvStorage.flagValueEnabled(""));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("0"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("false"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("NO"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("Off"));
    try std.testing.expect(MetalKvStorage.flagValueEnabled("1"));
    try std.testing.expect(MetalKvStorage.flagValueEnabled("true"));
}

test "Metal KV binding only covers successfully encoded tokens" {
    var binding = SlotBinding{
        .slot = 0,
        .written_tokens = 32,
    };
    try std.testing.expect(binding.covers(32));
    try std.testing.expect(!binding.covers(33));
    try std.testing.expect(binding.coversBeforePendingSuffix(40, 8));
    try std.testing.expect(!binding.coversBeforePendingSuffix(40, 7));
    try std.testing.expect(!binding.coversBeforePendingSuffix(32, 33));
    try std.testing.expect(binding.commitWrite(40, 4));
    try std.testing.expectEqual(@as(usize, 40), binding.written_tokens);
    try std.testing.expectEqual(@as(usize, 4), binding.position_offset);
    // Expert-checkpoint replay rewrites the same token position after a miss.
    // Equal watermarks are intentionally idempotent and must remain valid.
    try std.testing.expect(binding.commitWrite(40, 4));
    try std.testing.expectEqual(@as(usize, 40), binding.written_tokens);
    try std.testing.expectEqual(@as(usize, 4), binding.position_offset);
    try std.testing.expect(!binding.commitWrite(39, 4));
    try std.testing.expect(!binding.commitWrite(40, 3));
    binding.truncateTo(32);
    try std.testing.expectEqual(@as(usize, 32), binding.written_tokens);
    try std.testing.expectEqual(@as(usize, 4), binding.position_offset);
    try std.testing.expect(binding.commitWrite(33, 4));
}

test "context ordinary exact fingerprint maps logical pages canonically" {
    const blocks = [_]block.KvBlockId{ 7, 3, 11, 2 };
    const full = SlotBinding{ .slot = 1 };
    try std.testing.expectEqual(@as(usize, 28), try MetalKvStorage.qualificationPhysicalRow(full, &blocks, 4, 0));
    try std.testing.expectEqual(@as(usize, 13), try MetalKvStorage.qualificationPhysicalRow(full, &blocks, 4, 5));
    try std.testing.expectEqual(@as(usize, 9), try MetalKvStorage.qualificationPhysicalRow(full, &blocks, 4, 13));

    const ring = SlotBinding{ .slot = 2, .ring_page_count = 3, .sequence_owned = true };
    try std.testing.expectEqual(@as(usize, 0), try MetalKvStorage.qualificationPhysicalRow(ring, &blocks, 4, 0));
    try std.testing.expectEqual(@as(usize, 9), try MetalKvStorage.qualificationPhysicalRow(ring, &blocks, 4, 9));
    try std.testing.expectEqual(@as(usize, 1), try MetalKvStorage.qualificationPhysicalRow(ring, &blocks, 4, 13));
    try std.testing.expectError(error.InvalidQualificationFingerprint, MetalKvStorage.qualificationPhysicalRow(full, &.{}, 4, 0));
    try std.testing.expectError(error.InvalidQualificationFingerprint, MetalKvStorage.qualificationPhysicalRow(full, &blocks, 0, 0));
}

test "context ordinary exact fingerprint uses actual owners rather than pool layer capacity" {
    // The pool reserves a stride for every model layer. Shared-tail consumers
    // do not own bindings, so the canonical list contains only the loaded
    // layout's non-shared owner prefix.
    var owners: [24]u32 = undefined;
    for (&owners, 0..) |*owner, layer| owner.* = @intCast(layer);
    try validateQualificationOwnerLayers(&owners, owners.len);
    try std.testing.expectError(
        error.InvalidQualificationFingerprint,
        validateQualificationOwnerLayers(&owners, 35),
    );
    var duplicate = owners;
    duplicate[23] = 22;
    try std.testing.expectError(
        error.InvalidQualificationFingerprint,
        validateQualificationOwnerLayers(&duplicate, duplicate.len),
    );
    var reordered = owners;
    std.mem.swap(u32, &reordered[7], &reordered[8]);
    try std.testing.expectError(
        error.InvalidQualificationFingerprint,
        validateQualificationOwnerLayers(&reordered, reordered.len),
    );
}

test "context ordinary exact fingerprint excludes allocator page identity" {
    const binding = SlotBinding{ .slot = 1, .written_tokens = 13 };
    const sibling_binding = SlotBinding{ .slot = 9, .written_tokens = 13 };
    const blocks_a = [_]block.KvBlockId{ 7, 3, 11, 2 };
    const blocks_b = [_]block.KvBlockId{ 17, 13, 21, 12 };
    try std.testing.expect(try MetalKvStorage.qualificationPhysicalRow(binding, &blocks_a, 4, 5) !=
        try MetalKvStorage.qualificationPhysicalRow(binding, &blocks_b, 4, 5));

    var a = std.crypto.hash.sha2.Sha256.init(.{});
    var b = std.crypto.hash.sha2.Sha256.init(.{});
    try fingerprintQualificationLayer(&a, 0, binding, 0, 13, 13, 4, 32, 16, 2);
    try fingerprintQualificationLayer(&b, 0, sibling_binding, 0, 13, 13, 4, 32, 16, 2);
    var digest_a: [32]u8 = undefined;
    var digest_b: [32]u8 = undefined;
    a.final(&digest_a);
    b.final(&digest_b);
    try std.testing.expectEqualSlices(u8, &digest_a, &digest_b);

    var changed = std.crypto.hash.sha2.Sha256.init(.{});
    try fingerprintQualificationLayer(&changed, 0, binding, 1, 12, 13, 4, 32, 16, 2);
    var changed_digest: [32]u8 = undefined;
    changed.final(&changed_digest);
    try std.testing.expect(!std.mem.eql(u8, &digest_a, &changed_digest));
}

test "context ordinary exact fingerprint excludes SWA ring slack" {
    const binding = SlotBinding{
        .slot = 3,
        .ring_page_count = 2,
        .sliding_window_tokens = 5,
        .sequence_owned = true,
    };
    const visible = try qualificationVisibleRange(binding, 11, 4);
    try std.testing.expectEqual(@as(usize, 6), visible.start);
    try std.testing.expectEqual(@as(usize, 5), visible.count);

    const blocks = [_]block.KvBlockId{ 0, 1, 2 };
    const base = [_]u8{ 10, 11, 12, 90, 91, 92, 16, 17 };
    var poisoned_slack = base;
    poisoned_slack[3] = 1;
    poisoned_slack[4] = 2;
    poisoned_slack[5] = 3;
    var changed_visible = base;
    changed_visible[0] ^= 0xff; // Logical token 8 maps to physical row zero.

    var base_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var slack_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var changed_hash = std.crypto.hash.sha2.Sha256.init(.{});
    var token = visible.start;
    while (token < visible.start + visible.count) : (token += 1) {
        const physical = try MetalKvStorage.qualificationPhysicalRow(binding, &blocks, 4, token);
        base_hash.update(base[physical..][0..1]);
        slack_hash.update(poisoned_slack[physical..][0..1]);
        changed_hash.update(changed_visible[physical..][0..1]);
    }
    var base_digest: [32]u8 = undefined;
    var slack_digest: [32]u8 = undefined;
    var changed_digest: [32]u8 = undefined;
    base_hash.final(&base_digest);
    slack_hash.final(&slack_digest);
    changed_hash.final(&changed_digest);
    try std.testing.expectEqualSlices(u8, &base_digest, &slack_digest);
    try std.testing.expect(!std.mem.eql(u8, &base_digest, &changed_digest));
}

const hook_vtable: storage_runtime.DeviceWriteHook.VTable = .{
    .readExactAttentionMatchKv = MetalKvStorage.readExactAttentionMatchKv,
    .cloneExactPrefix = MetalKvStorage.cloneExactPrefix,
    .exportExactLayerRanges = MetalKvStorage.exportExactLayerRanges,
    .discardExactCloneSource = MetalKvStorage.discardExactCloneSourceOp,
    .writeLayerKvSuffix = MetalKvStorage.writeLayerKvSuffix,
    .gatherLayerKv = MetalKvStorage.gatherLayerKv,
    .gatherLayerKeys = MetalKvStorage.gatherLayerKeys,
    .gatherLayerKvDevice = MetalKvStorage.gatherLayerKvDevice,
    .pagedLayerKvDevice = MetalKvStorage.pagedLayerKvDevice,
    .reserveLayerKvDevice = MetalKvStorage.reserveLayerKvDevice,
    .commitLayerKvDeviceWrite = MetalKvStorage.commitLayerKvDeviceWrite,
    .repackSelectedGlobalKv = MetalKvStorage.repackSelectedGlobalKv,
    .scoreSelectedAttention = MetalKvStorage.scoreSelectedAttention,
    .truncateSequence = MetalKvStorage.truncateSequenceOp,
    .releaseSequence = MetalKvStorage.releaseSequenceOp,
    .supportsSelectedGlobalKv = MetalKvStorage.supportsSelectedGlobalKv,
    .allocationSnapshot = MetalKvStorage.allocationSnapshotOp,
    .drainForOwnerSnapshot = MetalKvStorage.drainForOwnerSnapshot,
    .deinit = MetalKvStorage.hookDeinit,
};

test "Metal KV hook teardown uses its allocation owner instead of the request allocator" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    const Bounded = @import("../runtime/bounded_allocator.zig").BoundedAllocator;
    var owner = Bounded{ .backing = std.testing.allocator, .limit = 128 * 1024 };
    var request = Bounded{ .backing = std.testing.allocator, .limit = 128 * 1024 };
    const runtime: *metal_runtime.RawMetalDecodeRuntime = @ptrFromInt(@alignOf(usize));
    const storage = try MetalKvStorage.create(owner.allocator(), runtime, .f16, 2, 256, 16);
    storage.deviceWriteHook().deinit(request.allocator());
    try std.testing.expectEqual(@as(usize, 0), owner.live);
    try std.testing.expectEqual(@as(usize, 0), request.live);
}

test "MetalKvStorage.create rejects unsupported dtype" {
    if (!build_options.enable_metal) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    // RawMetalDecodeRuntime is `opaque {}` so it has no known alignment at
    // comptime; use a word-aligned pointer value since `create` returns before
    // ever dereferencing when the dtype check fails.
    const fake_runtime: *metal_runtime.RawMetalDecodeRuntime = @ptrFromInt(@alignOf(usize));
    // int4 and fp8 still have no device kernel — fall through to host.
    try std.testing.expectError(
        error.DeviceWriteFormatUnsupported,
        MetalKvStorage.create(allocator, fake_runtime, .int4, 8, 128, 256),
    );
    try std.testing.expectError(
        error.DeviceWriteFormatUnsupported,
        MetalKvStorage.create(allocator, fake_runtime, .fp8, 8, 128, 256),
    );
}

test "request owner detach rejects live binding and permits fresh owner reuse" {
    const allocator = std.testing.allocator;
    const metal = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(metal);
    if (metal_runtime.termite_metal_decode_runtime_ready(metal) == 0) return error.SkipZigTest;

    const storage = try MetalKvStorage.create(allocator, metal, .f16, 1, 2, 4);
    const hook = storage.deviceWriteHook();
    defer hook.deinit(allocator);
    var first = owner_alloc.HostAllocationTracker.init(null, 0);
    var second = owner_alloc.HostAllocationTracker.init(null, 0);
    try storage.bindRequestOwnerTracker(&first);
    _ = try storage.acquireBinding(.{ .sequence_id = 17, .layer_index = 0 }, 1, 4, false, false);
    try std.testing.expectError(error.ContextFrameOwnershipMismatch, storage.unbindRequestOwnerTracker(&first));
    storage.releaseSequenceSlots(17);
    try storage.unbindRequestOwnerTracker(&first);
    try std.testing.expect(storage.owner_allocation_tracker == null);
    try storage.bindRequestOwnerTracker(&second);
    try storage.unbindRequestOwnerTracker(&second);
}

test "context request owner empty proof is hook local while foreign hook owns live device KV" {
    if (!build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const runtime = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(runtime);
    if (metal_runtime.termite_metal_decode_runtime_ready(runtime) == 0) return error.SkipZigTest;

    const dim: usize = 256;
    const foreign = try MetalKvStorage.create(allocator, runtime, .f16, 1, dim, 4);
    defer foreign.deviceWriteHook().deinit(allocator);
    const local = try MetalKvStorage.create(allocator, runtime, .f16, 1, dim, 4);
    defer local.deviceWriteHook().deinit(allocator);

    const rows: usize = 65;
    const bytes = rows * dim * @sizeOf(f32);
    const k_handle = metal_runtime.termite_metal_buffer_alloc(runtime, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(k_handle);
    const v_handle = metal_runtime.termite_metal_buffer_alloc(runtime, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(v_handle);
    const values = try allocator.alloc(f32, rows * dim);
    defer allocator.free(values);
    for (values, 0..) |*value, i| value.* = @floatFromInt(i);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, k_handle, 0, values.ptr, bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, v_handle, 0, values.ptr, bytes));

    const foreign_hook = foreign.deviceWriteHook();
    try foreign_hook.writeLayerKvSuffix(.{ .sequence_id = 91, .layer_index = 0, .total_token_count = 17, .suffix_token_count = 17, .position_offset = 0, .num_kv_heads = 1, .head_dim = dim }, .{ .handle = k_handle, .byte_offset = 0, .byte_len = bytes }, .{ .handle = v_handle, .byte_offset = 0, .byte_len = bytes });
    // The native span writer derives suffix_start from its installed token
    // count and consumes suffix rows starting at the caller's device offset.
    const suffix_offset = 17 * dim * @sizeOf(f32);
    try foreign_hook.writeLayerKvSuffix(.{ .sequence_id = 91, .layer_index = 0, .total_token_count = rows, .suffix_token_count = rows - 17, .position_offset = 0, .num_kv_heads = 1, .head_dim = dim }, .{ .handle = k_handle, .byte_offset = suffix_offset, .byte_len = bytes - suffix_offset }, .{ .handle = v_handle, .byte_offset = suffix_offset, .byte_len = bytes - suffix_offset });
    const foreign_before = foreign_hook.allocationSnapshot();
    const foreign_slot = foreign.slot_map.get(.{ .sequence_id = 91, .layer_index = 0 }).?.slot;
    const foreign_info = try foreign.slotAllocationInfo(foreign_slot);
    const foreign_transition = try foreign.slotTransitionInfo(foreign_slot);
    std.debug.print("owner-kv-growth stage=foreign hook_live={d} hook_kv={d} slot_key={d} slot_value={d} slot_installed={d} transition_nonce={d} transition_new={d}\n", .{ foreign_before.live_bytes, foreign_before.kv_bytes, foreign_info.key_allocated_bytes, foreign_info.value_allocated_bytes, foreign_info.installed_live_bytes, foreign_transition.operation_nonce, foreign_transition.newly_allocated_bytes });
    try std.testing.expect(foreign_before.live_bytes > 0);

    const local_hook = local.deviceWriteHook();
    var first = owner_alloc.HostAllocationTracker.init(null, 0);
    var second = owner_alloc.HostAllocationTracker.init(null, 0);
    try local.bindRequestOwnerTracker(&first);
    try local_hook.writeLayerKvSuffix(.{ .sequence_id = 92, .layer_index = 0, .total_token_count = 17, .suffix_token_count = 17, .position_offset = 0, .num_kv_heads = 1, .head_dim = dim }, .{ .handle = k_handle, .byte_offset = 0, .byte_len = bytes }, .{ .handle = v_handle, .byte_offset = 0, .byte_len = bytes });
    const first_17 = first.snapshot();
    const local_17 = local_hook.allocationSnapshot();
    const local_slot = local.slot_map.get(.{ .sequence_id = 92, .layer_index = 0 }).?.slot;
    const slot_17 = try local.slotAllocationInfo(local_slot);
    const transition_17 = try local.slotTransitionInfo(local_slot);
    std.debug.print("owner-kv-growth stage=17 hook_live={d} hook_kv={d} owner_live={d} owner_peak={d} slot_key={d} slot_value={d} slot_installed={d} slot_active={d} slot_submitted={d} transition_nonce={d} transition_new={d}\n", .{ local_17.live_bytes, local_17.kv_bytes, first_17.private_device_live_bytes, first_17.private_device_peak_bytes, slot_17.key_allocated_bytes, slot_17.value_allocated_bytes, slot_17.installed_live_bytes, slot_17.active_frame_retained_bytes, slot_17.submitted_frame_retained_bytes, transition_17.operation_nonce, transition_17.newly_allocated_bytes });
    try std.testing.expect(local_17.kv_bytes > 0);
    try std.testing.expectEqual(local_17.live_bytes, first_17.private_device_live_bytes);
    try local_hook.writeLayerKvSuffix(.{ .sequence_id = 92, .layer_index = 0, .total_token_count = rows, .suffix_token_count = rows - 17, .position_offset = 0, .num_kv_heads = 1, .head_dim = dim }, .{ .handle = k_handle, .byte_offset = suffix_offset, .byte_len = bytes - suffix_offset }, .{ .handle = v_handle, .byte_offset = suffix_offset, .byte_len = bytes - suffix_offset });
    const first_65 = first.snapshot();
    const local_65 = local_hook.allocationSnapshot();
    try std.testing.expect(local_65.kv_bytes > local_17.kv_bytes);
    try std.testing.expectEqual(local_65.live_bytes, first_65.private_device_live_bytes);
    const growth = local.last_owner_transition;
    const slot_65 = try local.slotAllocationInfo(local_slot);
    const transition_65 = try local.slotTransitionInfo(local_slot);
    std.debug.print("owner-kv-growth stage=65 hook_live={d} hook_kv={d} owner_live={d} owner_peak={d} slot_key={d} slot_value={d} slot_installed={d} slot_active={d} slot_submitted={d} native_nonce={d} native_new={d} reconciled_prior={d} reconciled_new={d} reconciled_final={d}\n", .{ local_65.live_bytes, local_65.kv_bytes, first_65.private_device_live_bytes, first_65.private_device_peak_bytes, slot_65.key_allocated_bytes, slot_65.value_allocated_bytes, slot_65.installed_live_bytes, slot_65.active_frame_retained_bytes, slot_65.submitted_frame_retained_bytes, transition_65.operation_nonce, transition_65.newly_allocated_bytes, growth.prior, growth.newly_allocated, growth.final });
    const expected_overlap = first_17.private_device_live_bytes + growth.newly_allocated;
    try std.testing.expectEqual(@max(first_17.private_device_peak_bytes, expected_overlap), first_65.private_device_peak_bytes);
    try std.testing.expect(first_65.private_device_live_bytes > first_17.private_device_live_bytes);

    const gathered_k = try allocator.alloc(f32, rows * dim);
    defer allocator.free(gathered_k);
    const gathered_v = try allocator.alloc(f32, rows * dim);
    defer allocator.free(gathered_v);
    try local_hook.gatherLayerKv(.{ .sequence_id = 92, .layer_index = 0, .token_count = rows, .num_kv_heads = 1, .head_dim = dim }, gathered_k, gathered_v);
    for (values, gathered_k, gathered_v) |expected, actual_k, actual_v| {
        try std.testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(expected)))), actual_k);
        try std.testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(expected)))), actual_v);
    }
    var logical_blocks: [17]block.KvBlockId = undefined;
    for (&logical_blocks, 0..) |*id, index| id.* = @intCast(index);
    const one_copy_bytes = rows * (2 * @sizeOf(u64) + @sizeOf(f32) + 2 * dim * @sizeOf(f32));
    const raw_authority: MetalKvStorage.RawHeadAuthority = .{ .sequence_id = 92, .source_revision = local.attention_match_revision, .logical_tokens = rows, .logical_blocks = &logical_blocks, .donor_layer = 0, .head_index = 0, .window = 0 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, local.downloadRawExactHead(allocator, raw_authority, 2 * one_copy_bytes - 1, .{}));
    var stale = raw_authority;
    stale.source_revision += 1;
    try std.testing.expectError(error.AttentionMatchSourceChanged, local.downloadRawExactHead(allocator, stale, 2 * one_copy_bytes, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, local.downloadRawExactHead(allocator, raw_authority, 2 * one_copy_bytes, .{ .check_fn = Cancel.check }));
    var downloaded = try local.downloadRawExactHead(allocator, raw_authority, 2 * one_copy_bytes, .{});
    defer downloaded.deinit();
    try std.testing.expectEqual(rows, downloaded.value.positions.len);
    try std.testing.expectEqual(@as(u64, 0), downloaded.value.positions[0]);
    try std.testing.expectEqual(@as(u64, rows), downloaded.value.coverage_ends[rows - 1]);
    try std.testing.expectEqualSlices(f32, gathered_k, downloaded.value.keys);
    try std.testing.expectEqualSlices(f32, gathered_v, downloaded.value.values);
    try std.testing.expectError(error.ContextFrameOwnershipMismatch, local.unbindRequestOwnerTracker(&first));
    local.releaseSequenceSlots(92);
    const first_released = first.snapshot();
    std.debug.print("owner-kv-growth stage=released owner_live={d} owner_peak={d} foreign_live={d} foreign_kv={d}\n", .{ first_released.private_device_live_bytes, first_released.private_device_peak_bytes, foreign_hook.allocationSnapshot().live_bytes, foreign_hook.allocationSnapshot().kv_bytes });
    try std.testing.expectEqual(@as(u64, 0), first_released.private_device_live_bytes);
    try local.unbindRequestOwnerTracker(&first);

    try local.bindRequestOwnerTracker(&second);
    try local_hook.writeLayerKvSuffix(.{ .sequence_id = 93, .layer_index = 0, .total_token_count = 17, .suffix_token_count = 17, .position_offset = 0, .num_kv_heads = 1, .head_dim = dim }, .{ .handle = k_handle, .byte_offset = 0, .byte_len = bytes }, .{ .handle = v_handle, .byte_offset = 0, .byte_len = bytes });
    try std.testing.expect(second.snapshot().private_device_live_bytes > 0);
    local.releaseSequenceSlots(93);
    try std.testing.expectEqual(@as(u64, 0), second.snapshot().private_device_live_bytes);
    try local.unbindRequestOwnerTracker(&second);
    const foreign_after = foreign_hook.allocationSnapshot();
    std.debug.print("owner-kv-growth stage=foreign-after hook_live={d} hook_kv={d} before_live={d} before_kv={d}\n", .{ foreign_after.live_bytes, foreign_after.kv_bytes, foreign_before.live_bytes, foreign_before.kv_bytes });
    try std.testing.expectEqual(foreign_before.live_bytes, foreign_after.live_bytes);
    try std.testing.expectEqual(foreign_before.kv_bytes, foreign_after.kv_bytes);
}

test "compiled owner drain reconciles active and submitted retired KV generations before committed snapshot" {
    if (!build_options.enable_metal or !metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const compiled_runtime = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(compiled_runtime);
    if (metal_runtime.termite_metal_decode_runtime_ready(compiled_runtime) == 0) return error.SkipZigTest;
    const outer_runtime = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(outer_runtime);
    if (metal_runtime.termite_metal_decode_runtime_ready(outer_runtime) == 0) return error.SkipZigTest;

    const compiled = try MetalKvStorage.create(allocator, compiled_runtime, .f16, 1, 256, 8);
    const compiled_hook = compiled.deviceWriteHook();
    defer compiled_hook.deinit(allocator);
    const outer = try MetalKvStorage.create(allocator, outer_runtime, .f16, 1, 256, 8);
    const outer_hook = outer.deviceWriteHook();
    defer outer_hook.deinit(allocator);
    var tracker = owner_alloc.HostAllocationTracker.init(null, 0);
    try compiled.bindRequestOwnerTracker(&tracker);
    const binding = try compiled.acquireBinding(.{ .sequence_id = 501, .layer_index = 0 }, 1, 128, false, false);
    const row_bytes = 256 * @sizeOf(f16);

    try compiled.reserveSlotCapacity(binding.slot, 16, 16, 0, row_bytes, row_bytes);
    const installed_initial = (try compiled.slotAllocationInfo(binding.slot)).installed_live_bytes;
    try std.testing.expect(installed_initial > 0);

    try metal_runtime.beginFrame(compiled_runtime);
    try compiled.reserveSlotCapacity(binding.slot, 32, 16, 0, row_bytes, row_bytes);
    const active = try compiled.slotAllocationInfo(binding.slot);
    try std.testing.expect(active.active_frame_retained_bytes > 0);
    const active_accounted = tracker.snapshot();
    try std.testing.expectEqual(@as(u64, @intCast(try MetalKvStorage.slotLiveBytes(active))), active_accounted.private_device_live_bytes);
    try outer_hook.drainForOwnerSnapshot();
    try std.testing.expect(metal_runtime.hasActiveFrame(compiled_runtime));
    try std.testing.expectEqual(active_accounted.private_device_live_bytes, tracker.snapshot().private_device_live_bytes);

    try compiled_hook.drainForOwnerSnapshot();
    try std.testing.expect(!metal_runtime.hasActiveFrame(compiled_runtime));
    const after_active = try compiled.slotAllocationInfo(binding.slot);
    try std.testing.expectEqual(@as(u64, 0), after_active.active_frame_retained_bytes);
    try std.testing.expectEqual(after_active.installed_live_bytes, tracker.snapshot().private_device_live_bytes);

    try metal_runtime.beginFrame(compiled_runtime);
    try compiled.reserveSlotCapacity(binding.slot, 64, 16, 0, row_bytes, row_bytes);
    try metal_runtime.submitFrame(compiled_runtime);
    const submitted = try compiled.slotAllocationInfo(binding.slot);
    try std.testing.expect(submitted.submitted_frame_retained_bytes > 0);
    const submitted_accounted = tracker.snapshot();
    try std.testing.expectEqual(@as(u64, @intCast(try MetalKvStorage.slotLiveBytes(submitted))), submitted_accounted.private_device_live_bytes);
    try outer_hook.drainForOwnerSnapshot();
    try std.testing.expect(metal_runtime.hasSubmittedFrame(compiled_runtime));
    try std.testing.expectEqual(submitted_accounted.private_device_live_bytes, tracker.snapshot().private_device_live_bytes);

    try compiled_hook.drainForOwnerSnapshot();
    try std.testing.expect(!metal_runtime.hasSubmittedFrame(compiled_runtime));
    const committed = try compiled.slotAllocationInfo(binding.slot);
    try std.testing.expectEqual(@as(u64, 0), committed.submitted_frame_retained_bytes);
    try std.testing.expectEqual(committed.installed_live_bytes, tracker.snapshot().private_device_live_bytes);
    try std.testing.expect(tracker.snapshot().private_device_peak_bytes > committed.installed_live_bytes);

    compiled.releaseSequenceSlots(501);
    try compiled.unbindRequestOwnerTracker(&tracker);
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().private_device_live_bytes);
}

test "KeyFormat.fromKvDType covers supported dtypes" {
    try std.testing.expectEqual(KeyFormat.polar4, KeyFormat.fromKvDType(.polar4).?);
    try std.testing.expectEqual(KeyFormat.turbo3, KeyFormat.fromKvDType(.turbo3).?);
    try std.testing.expectEqual(KeyFormat.raw_f32, KeyFormat.fromKvDType(.f32).?);
    try std.testing.expectEqual(KeyFormat.f16, KeyFormat.fromKvDType(.f16).?);
    try std.testing.expectEqual(KeyFormat.int8_per_head, KeyFormat.fromKvDType(.int8).?);
    try std.testing.expect(KeyFormat.fromKvDType(.int4) == null);
    try std.testing.expect(KeyFormat.fromKvDType(.fp8) == null);
    try std.testing.expect(KeyFormat.polar4.isCompressed());
    try std.testing.expect(KeyFormat.turbo3.isCompressed());
    try std.testing.expect(!KeyFormat.raw_f32.isCompressed());
    try std.testing.expect(!KeyFormat.f16.isCompressed());
    try std.testing.expect(KeyFormat.int8_per_head.isCompressed());
}

test "exact layer export mixes head-major and wrapped ring bitwise on one Metal hook" {
    if (!build_options.enable_metal or builtin.os.tag != .macos) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const metal = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(metal);
    if (metal_runtime.termite_metal_decode_runtime_ready(metal) == 0) return error.SkipZigTest;

    const source = try MetalKvStorage.create(allocator, metal, .f16, 2, 512, 16);
    const hook = source.deviceWriteHook();
    defer hook.deinit(allocator);
    const rows: usize = 17;
    const heads: usize = 2;
    const dim: usize = 512;
    const elements = rows * heads * dim;
    const input_k = try allocator.alloc(f32, elements);
    defer allocator.free(input_k);
    const input_v = try allocator.alloc(f32, elements);
    defer allocator.free(input_v);
    for (input_k, input_v, 0..) |*k, *v, i| {
        const row = i / (heads * dim);
        const head = (i / dim) % heads;
        const lane = i % dim;
        k.* = @floatFromInt(1 + row * 4 + head * 2 + lane % 2);
        v.* = -k.*;
    }
    const input_bytes = elements * @sizeOf(f32);
    const input_k_handle = metal_runtime.termite_metal_buffer_alloc(metal, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(input_k_handle);
    const input_v_handle = metal_runtime.termite_metal_buffer_alloc(metal, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(input_v_handle);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, input_k_handle, 0, @ptrCast(input_k.ptr), input_bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, input_v_handle, 0, @ptrCast(input_v.ptr), input_bytes));
    const blocks = [_]u32{ 0, 1 };
    for ([_]u32{ 1, 4 }) |layer| try hook.writeLayerKvSuffix(.{
        .sequence_id = 77,
        .layer_index = layer,
        .total_token_count = rows,
        .suffix_token_count = rows,
        .position_offset = 0,
        .num_kv_heads = heads,
        .head_dim = dim,
        .logical_blocks = &blocks,
        .page_size_tokens = 16,
    }, .{ .handle = input_k_handle, .byte_offset = 0, .byte_len = input_bytes }, .{
        .handle = input_v_handle,
        .byte_offset = 0,
        .byte_len = input_bytes,
    });

    const global_capacity: usize = 20;
    const global_bytes = global_capacity * heads * dim * @sizeOf(f16);
    const ring_capacity: usize = 8;
    const ring_bytes = ring_capacity * heads * dim * @sizeOf(f16);
    const global_k = metal_runtime.termite_metal_buffer_alloc(metal, global_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(global_k);
    const global_v = metal_runtime.termite_metal_buffer_alloc(metal, global_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(global_v);
    const ring_k = metal_runtime.termite_metal_buffer_alloc(metal, ring_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(ring_k);
    const ring_v = metal_runtime.termite_metal_buffer_alloc(metal, ring_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(ring_v);
    const sentinel = try allocator.alloc(u16, global_bytes / @sizeOf(u16));
    defer allocator.free(sentinel);
    @memset(sentinel, 0x7bcd);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, global_k, 0, @ptrCast(sentinel.ptr), global_bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, global_v, 0, @ptrCast(sentinel.ptr), global_bytes));
    const ranges = [_]storage_runtime.ExactLayerRangeExport{
        .{ .layer_index = 1, .source_start = 0, .rows = rows, .num_kv_heads = heads, .head_dim = dim, .destination_layout = .head_major, .destination_head_capacity_rows = global_capacity, .destination_k = .{ .handle = global_k, .byte_offset = 0, .byte_len = global_bytes }, .destination_v = .{ .handle = global_v, .byte_offset = 0, .byte_len = global_bytes } },
        .{ .layer_index = 4, .source_start = rows - ring_capacity, .rows = ring_capacity, .num_kv_heads = heads, .head_dim = dim, .destination_layout = .token_major_ring, .destination_ring_capacity_rows = ring_capacity, .destination_k = .{ .handle = ring_k, .byte_offset = 0, .byte_len = ring_bytes }, .destination_v = .{ .handle = ring_v, .byte_offset = 0, .byte_len = ring_bytes } },
    };
    try std.testing.expectError(error.ExactLayerRangeExportBudgetExceeded, hook.exportExactLayerRanges(.{
        .source_sequence = 77,
        .expected_written_tokens = rows,
        .ranges = &ranges,
        .max_bytes = 1,
    }));
    var aliased = ranges;
    aliased[1].destination_k = aliased[0].destination_k;
    try std.testing.expectError(error.InvalidExactLayerRangeExport, hook.exportExactLayerRanges(.{
        .source_sequence = 77,
        .expected_written_tokens = rows,
        .ranges = &aliased,
        .max_bytes = std.math.maxInt(usize),
    }));
    const receipt = try hook.exportExactLayerRanges(.{ .source_sequence = 77, .expected_written_tokens = rows, .ranges = &ranges, .max_bytes = 2 * (rows + ring_capacity) * heads * dim * @sizeOf(f16) });
    try std.testing.expectEqual(@as(?usize, 0), receipt.peak_host_staging_bytes);
    try std.testing.expect(receipt.fenced);
    try std.testing.expectEqual(@as(usize, 2), receipt.layers);
    const observed_global = try allocator.alloc(u16, global_bytes / 2);
    defer allocator.free(observed_global);
    const observed_ring = try allocator.alloc(u16, ring_bytes / 2);
    defer allocator.free(observed_ring);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_download(metal, global_k, 0, @ptrCast(observed_global.ptr), global_bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_download(metal, ring_k, 0, @ptrCast(observed_ring.ptr), ring_bytes));
    for (0..rows) |row| for (0..heads) |head| for (0..dim) |lane| {
        const source_index = (row * heads + head) * dim + lane;
        const global_index = (head * global_capacity + row) * dim + lane;
        try std.testing.expectEqual(@as(u16, @bitCast(@as(f16, @floatCast(input_k[source_index])))), observed_global[global_index]);
    };
    for (0..heads) |head| for (rows..global_capacity) |row| for (0..dim) |lane|
        try std.testing.expectEqual(@as(u16, 0x7bcd), observed_global[(head * global_capacity + row) * dim + lane]);
    for (rows - ring_capacity..rows) |row| for (0..heads) |head| for (0..dim) |lane| {
        const source_index = (row * heads + head) * dim + lane;
        const ring_index = ((row % ring_capacity) * heads + head) * dim + lane;
        try std.testing.expectEqual(@as(u16, @bitCast(@as(f16, @floatCast(input_k[source_index])))), observed_ring[ring_index]);
    };
}

test "attention_match Metal exact prefix clone is transactional owner independent and records readback" {
    if (!build_options.enable_metal or builtin.os.tag != .macos) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const metal = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(metal);
    if (metal_runtime.termite_metal_decode_runtime_ready(metal) == 0) return error.SkipZigTest;

    const source = try MetalKvStorage.create(allocator, metal, .f16, 1, 2, 4);
    const source_hook = source.deviceWriteHook();
    defer source_hook.deinit(allocator);
    const destination = try MetalKvStorage.create(allocator, metal, .f16, 1, 2, 4);
    const destination_hook = destination.deviceWriteHook();
    defer destination_hook.deinit(allocator);

    const tokens: usize = 3;
    const width: usize = 2;
    const blocks = [_]u32{0};
    const layer_one_keys = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const layer_one_values = [_]f32{ -1, -2, -3, -4, -5, -6 };
    const layer_four_keys = [_]f32{ 7, 8, 9, 10, 11, 12 };
    const layer_four_values = [_]f32{ -7, -8, -9, -10, -11, -12 };
    const bytes = tokens * width * @sizeOf(f32);
    const key_handle = metal_runtime.termite_metal_buffer_alloc(metal, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(key_handle);
    const value_handle = metal_runtime.termite_metal_buffer_alloc(metal, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(value_handle);
    const fixtures = [_]struct { layer: u32, keys: []const f32, values: []const f32 }{
        .{ .layer = 1, .keys = &layer_one_keys, .values = &layer_one_values },
        .{ .layer = 4, .keys = &layer_four_keys, .values = &layer_four_values },
    };
    for (fixtures) |fixture| {
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(
            metal,
            key_handle,
            0,
            @ptrCast(fixture.keys.ptr),
            bytes,
        ));
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(
            metal,
            value_handle,
            0,
            @ptrCast(fixture.values.ptr),
            bytes,
        ));
        try source_hook.writeLayerKvSuffix(.{
            .sequence_id = 11,
            .layer_index = fixture.layer,
            .total_token_count = tokens,
            .suffix_token_count = tokens,
            .position_offset = 0,
            .num_kv_heads = 1,
            .head_dim = 2,
            .logical_blocks = &blocks,
            .page_size_tokens = 4,
        }, .{ .handle = key_handle, .byte_offset = 0, .byte_len = bytes }, .{
            .handle = value_handle,
            .byte_offset = 0,
            .byte_len = bytes,
        });
    }

    const request: storage_runtime.device_clone.Request = .{
        .source_sequence = 11,
        .destination_sequence = 22,
        .tokens = tokens,
        .capacity = 8,
        .page_size = 4,
        .source_blocks = &blocks,
        .destination_blocks = &blocks,
        .donors = &.{
            .{ .layer = 1, .heads = 1, .dim = 2 },
            .{ .layer = 4, .heads = 1, .dim = 2 },
        },
        .max_bytes = 1024 * 1024,
    };
    var oom_owner = owner_alloc.HostAllocationTracker.init(null, 0);
    const oom_destination = try MetalKvStorage.createTracked(
        oom_owner.allocator(),
        metal,
        .f16,
        1,
        2,
        4,
        &oom_owner,
    );
    const oom_hook = oom_destination.deviceWriteHook();
    defer oom_hook.deinit(allocator);
    try oom_owner.armNextPositiveAllocationFailure();
    try std.testing.expectError(
        error.OutOfMemory,
        oom_hook.cloneExactPrefix(source_hook, request),
    );
    try std.testing.expectEqual(@as(usize, 0), oom_destination.slot_map.count());
    try std.testing.expectEqual(@as(u64, 0), oom_hook.allocationSnapshot().live_bytes);

    var over_budget = request;
    over_budget.max_bytes = 1;
    try std.testing.expectError(
        error.DeviceKvCloneBudgetExceeded,
        destination_hook.cloneExactPrefix(source_hook, over_budget),
    );
    try std.testing.expectEqual(@as(usize, 0), destination.slot_map.count());

    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            // The hook wrapper, preflight, and first donor pass. Cancel before
            // the second donor and require the first staged slot to roll back.
            if (self.calls == 4) return error.CompactionCancelled;
        }
    };
    var cancel: Cancel = .{};
    var cancelled = request;
    cancelled.check_ctx = &cancel;
    cancelled.check_fn = Cancel.check;
    try std.testing.expectError(
        error.CompactionCancelled,
        destination_hook.cloneExactPrefix(source_hook, cancelled),
    );
    try std.testing.expectEqual(@as(usize, 0), destination.slot_map.count());
    try std.testing.expectEqual(@as(u64, 0), destination_hook.allocationSnapshot().live_bytes);

    const result = try destination_hook.cloneExactPrefix(source_hook, request);
    try std.testing.expectEqual(@as(usize, 2), result.donors);
    try std.testing.expectEqual(@as(usize, tokens * width * @sizeOf(f16) * 2 * 2), result.copied_bytes);
    try std.testing.expect(result.allocation_bound_bytes <= request.max_bytes);

    var source_k: [tokens * width]f32 = undefined;
    var source_v: [tokens * width]f32 = undefined;
    var destination_k: [tokens * width]f32 = undefined;
    var destination_v: [tokens * width]f32 = undefined;
    try source_hook.gatherLayerKv(.{
        .sequence_id = 11,
        .layer_index = 1,
        .token_count = tokens,
        .num_kv_heads = 1,
        .head_dim = 2,
    }, &source_k, &source_v);
    try destination_hook.gatherLayerKv(.{
        .sequence_id = 22,
        .layer_index = 1,
        .token_count = tokens,
        .num_kv_heads = 1,
        .head_dim = 2,
    }, &destination_k, &destination_v);
    try std.testing.expectEqualSlices(f32, &source_k, &destination_k);
    try std.testing.expectEqualSlices(f32, &source_v, &destination_v);

    const suffix_keys = [_]f32{ 13, 14 };
    const suffix_values = [_]f32{ -13, -14 };
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(
        metal,
        key_handle,
        0,
        @ptrCast(&suffix_keys),
        suffix_keys.len * @sizeOf(f32),
    ));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(
        metal,
        value_handle,
        0,
        @ptrCast(&suffix_values),
        suffix_values.len * @sizeOf(f32),
    ));
    try destination_hook.writeLayerKvSuffix(.{
        .sequence_id = 22,
        .layer_index = 1,
        .total_token_count = 4,
        .suffix_token_count = 1,
        .position_offset = 0,
        .num_kv_heads = 1,
        .head_dim = 2,
        .logical_blocks = &blocks,
        .page_size_tokens = 4,
    }, .{ .handle = key_handle, .byte_offset = 0, .byte_len = suffix_keys.len * @sizeOf(f32) }, .{
        .handle = value_handle,
        .byte_offset = 0,
        .byte_len = suffix_values.len * @sizeOf(f32),
    });
    var destination_appended_k: [4 * width]f32 = undefined;
    var destination_appended_v: [4 * width]f32 = undefined;
    try destination_hook.gatherLayerKv(.{
        .sequence_id = 22,
        .layer_index = 1,
        .token_count = 4,
        .num_kv_heads = 1,
        .head_dim = 2,
    }, &destination_appended_k, &destination_appended_v);
    try std.testing.expectEqualSlices(f32, &source_k, destination_appended_k[0 .. tokens * width]);
    try std.testing.expectApproxEqAbs(@as(f32, 13), destination_appended_k[6], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 14), destination_appended_k[7], 1e-3);

    const readback_before = source_hook.allocationSnapshot().readback_bytes;
    var readback_k: [2 * width]f32 = undefined;
    var readback_v: [2 * width]f32 = undefined;
    const reader = try source_hook.attentionMatchReader();
    try reader.read_fn(reader.context, .{
        .sequence = 11,
        .layer = 1,
        .logical_length = tokens,
        .start = 1,
        .count = 2,
        .heads = 1,
        .dim = 2,
    }, &readback_k, &readback_v);
    const readback_after = source_hook.allocationSnapshot().readback_bytes;
    try std.testing.expectEqual(
        @as(u64, 2 * width * @sizeOf(f16) * 2),
        readback_after - readback_before,
    );
    for (readback_k, layer_one_keys[width..]) |actual, expected|
        try std.testing.expectApproxEqAbs(expected, actual, 1e-3);

    destination_hook.releaseSequence(22);
    try std.testing.expectEqual(@as(usize, 0), destination.slot_map.count());
    var source_after_release_k: [tokens * width]f32 = undefined;
    var source_after_release_v: [tokens * width]f32 = undefined;
    try source_hook.gatherLayerKv(.{
        .sequence_id = 11,
        .layer_index = 1,
        .token_count = tokens,
        .num_kv_heads = 1,
        .head_dim = 2,
    }, &source_after_release_k, &source_after_release_v);
    try std.testing.expectEqualSlices(f32, &source_k, &source_after_release_k);
    try std.testing.expectEqualSlices(f32, &source_v, &source_after_release_v);
}

test "attention_match Metal exact prefix clone preserves split SWA geometry and chronology" {
    if (!build_options.enable_metal or builtin.os.tag != .macos) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const metal = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(metal);
    if (metal_runtime.termite_metal_decode_runtime_ready(metal) == 0) return error.SkipZigTest;

    const source = try MetalKvStorage.create(allocator, metal, .f16, 1, 2, 4);
    const source_hook = source.deviceWriteHook();
    defer source_hook.deinit(allocator);
    const destination = try MetalKvStorage.create(allocator, metal, .f16, 1, 2, 4);
    const destination_hook = destination.deviceWriteHook();
    defer destination_hook.deinit(allocator);

    const tokens: usize = 12;
    const width: usize = 2;
    // Same LIFO recycle shape proven by KvManager: the first three logical
    // pages remain uniquely owned but their physical IDs are reversed.
    const blocks = [_]u32{ 2, 1, 0, 3 };
    var keys: [tokens * width]f32 = undefined;
    var values: [tokens * width]f32 = undefined;
    for (&keys, &values, 0..) |*key, *value, index| {
        key.* = @floatFromInt(index + 1);
        value.* = -key.*;
    }
    const bytes = keys.len * @sizeOf(f32);
    const key_handle = metal_runtime.termite_metal_buffer_alloc(metal, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(key_handle);
    const value_handle = metal_runtime.termite_metal_buffer_alloc(metal, bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(value_handle);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, key_handle, 0, @ptrCast(&keys), bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, value_handle, 0, @ptrCast(&values), bytes));

    try source_hook.writeLayerKvSuffix(.{
        .sequence_id = 31,
        .layer_index = 0,
        .total_token_count = tokens,
        .suffix_token_count = tokens,
        .position_offset = 0,
        .num_kv_heads = 1,
        .head_dim = 2,
        .logical_blocks = &blocks,
        .page_size_tokens = 4,
        .sequence_owned_global = true,
    }, .{ .handle = key_handle, .byte_offset = 0, .byte_len = bytes }, .{ .handle = value_handle, .byte_offset = 0, .byte_len = bytes });
    const chunk_tokens: usize = 4;
    const chunk_bytes = chunk_tokens * width * @sizeOf(f32);
    for (0..3) |chunk_index| {
        const chunk_end = (chunk_index + 1) * chunk_tokens;
        const chunk_offset = chunk_index * chunk_bytes;
        try source_hook.writeLayerKvSuffix(.{
            .sequence_id = 31,
            .layer_index = 1,
            .total_token_count = chunk_end,
            .suffix_token_count = chunk_tokens,
            .position_offset = 0,
            .num_kv_heads = 1,
            .head_dim = 2,
            .logical_blocks = &blocks,
            .page_size_tokens = 4,
            .sliding_window = 5,
            .max_inflight_tokens = 1,
            .allow_swa_ring = true,
        }, .{ .handle = key_handle, .byte_offset = chunk_offset, .byte_len = chunk_bytes }, .{ .handle = value_handle, .byte_offset = chunk_offset, .byte_len = chunk_bytes });
    }

    const request: storage_runtime.device_clone.Request = .{
        .source_sequence = 31,
        .destination_sequence = 32,
        .tokens = tokens,
        .capacity = 16,
        .page_size = 4,
        .source_blocks = &blocks,
        .destination_blocks = &blocks,
        .logical_block_policy = .metal_sequence_slot_owned,
        .donors = &.{
            .{ .layer = 0, .heads = 1, .dim = 2, .capacity_rows = 16 },
            .{ .layer = 1, .heads = 1, .dim = 2, .capacity_rows = 8, .host_staged_ring = true },
        },
        .max_bytes = 1024 * 1024,
        .max_host_staging_bytes = 64,
    };
    var host_capped = request;
    host_capped.max_host_staging_bytes -= 1;
    try std.testing.expectError(
        error.DeviceKvCloneHostStagingBudgetExceeded,
        destination_hook.cloneExactPrefix(source_hook, host_capped),
    );
    try std.testing.expectEqual(@as(usize, 0), destination.slot_map.count());
    try std.testing.expectEqual(@as(u64, 0), destination_hook.allocationSnapshot().live_bytes);

    const result = try destination_hook.cloneExactPrefix(source_hook, request);
    try std.testing.expectEqual(@as(usize, 2), result.donors);
    try std.testing.expectEqual(@as(usize, (tokens + 8) * width * @sizeOf(f16) * 2), result.copied_bytes);

    const reader = try destination_hook.attentionMatchReader();
    var global_k: [tokens * width]f32 = undefined;
    var global_v: [tokens * width]f32 = undefined;
    try reader.read_fn(reader.context, .{
        .sequence = 32,
        .layer = 0,
        .logical_length = tokens,
        .start = 0,
        .count = tokens,
        .heads = 1,
        .dim = 2,
    }, &global_k, &global_v);
    for (global_k, keys) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    for (global_v, values) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-3);

    const local_start = tokens - 5;
    var local_k: [5 * width]f32 = undefined;
    var local_v: [5 * width]f32 = undefined;
    try reader.read_fn(reader.context, .{
        .sequence = 32,
        .layer = 1,
        .logical_length = tokens,
        .start = local_start,
        .count = 5,
        .heads = 1,
        .dim = 2,
    }, &local_k, &local_v);
    for (local_k, keys[local_start * width ..]) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-3);
    for (local_v, values[local_start * width ..]) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 1e-3);

    const source_reader = try source_hook.attentionMatchReader();
    var source_local_k: [5 * width]f32 = undefined;
    var source_local_v: [5 * width]f32 = undefined;
    try source_reader.read_fn(source_reader.context, .{
        .sequence = 31,
        .layer = 1,
        .logical_length = tokens,
        .start = local_start,
        .count = 5,
        .heads = 1,
        .dim = 2,
    }, &source_local_k, &source_local_v);
    try std.testing.expectEqualSlices(f32, &local_k, &source_local_k);
    try std.testing.expectEqualSlices(f32, &local_v, &source_local_v);
    try source_hook.writeLayerKvSuffix(.{
        .sequence_id = 99,
        .layer_index = 0,
        .total_token_count = 1,
        .suffix_token_count = 1,
        .position_offset = 0,
        .num_kv_heads = 1,
        .head_dim = 2,
        .logical_blocks = &blocks,
        .page_size_tokens = 4,
        .sequence_owned_global = true,
    }, .{ .handle = key_handle, .byte_offset = 0, .byte_len = width * @sizeOf(f32) }, .{ .handle = value_handle, .byte_offset = 0, .byte_len = width * @sizeOf(f32) });
    try std.testing.expectError(error.UnsafeExactCloneSourceDiscard, source_hook.discardExactCloneSource(31));
    var other_k: [width]f32 = undefined;
    var other_v: [width]f32 = undefined;
    try source_hook.gatherLayerKv(.{ .sequence_id = 99, .layer_index = 0, .token_count = 1, .num_kv_heads = 1, .head_dim = 2 }, &other_k, &other_v);
    try std.testing.expectApproxEqAbs(keys[0], other_k[0], 1e-3);
    source_hook.releaseSequence(99);
    const discarded = try source_hook.discardExactCloneSource(31);
    try std.testing.expectEqual(@as(usize, 2), discarded.slots);
    try std.testing.expect(discarded.physically_released_bytes > 0);
    try std.testing.expectEqual(@as(u64, 0), discarded.live_bytes_after);
    try std.testing.expectEqual(@as(usize, 0), source.slot_map.count());
    var destination_after_discard_k: [5 * width]f32 = undefined;
    var destination_after_discard_v: [5 * width]f32 = undefined;
    try reader.read_fn(reader.context, .{
        .sequence = 32,
        .layer = 1,
        .logical_length = tokens,
        .start = local_start,
        .count = 5,
        .heads = 1,
        .dim = 2,
    }, &destination_after_discard_k, &destination_after_discard_v);
    try std.testing.expectEqualSlices(f32, &local_k, &destination_after_discard_k);
    try std.testing.expectEqualSlices(f32, &local_v, &destination_after_discard_v);
    destination_hook.releaseSequence(32);
    try std.testing.expectEqual(@as(usize, 0), destination.slot_map.count());
}

test "selected Metal repack validates committed physical source rows" {
    try MetalKvStorage.validateSelectedSourceRows(&.{ 0, 3, 7 }, 8);
    try std.testing.expectError(error.InvalidKvShape, MetalKvStorage.validateSelectedSourceRows(&.{}, 8));
    try std.testing.expectError(error.InvalidKvShape, MetalKvStorage.validateSelectedSourceRows(&.{ 0, 8 }, 8));
    try std.testing.expectError(error.InvalidKvShape, MetalKvStorage.validateSelectedSourceRows(&.{ 2, 2 }, 8));
    try std.testing.expectError(error.InvalidKvShape, MetalKvStorage.validateSelectedSourceRows(&.{ 3, 2 }, 8));
}

fn selectionTestNotCancelled(_: ?*anyopaque) callconv(.c) c_int {
    return 0;
}

test "selected Metal GPU scorer matches decoded-key oracle before and after repack" {
    if (!build_options.enable_metal or builtin.os.tag != .macos) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const metal = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(metal);
    if (metal_runtime.termite_metal_decode_runtime_ready(metal) == 0) return error.SkipZigTest;

    const rows: usize = 4;
    const kv_heads: u32 = 1;
    const query_heads: u32 = 2;
    const head_dim: u32 = 128;
    const width: usize = @as(usize, kv_heads) * @as(usize, head_dim);
    var keys: [rows * width]f32 = undefined;
    var values: [rows * width]f32 = undefined;
    for (&keys, &values, 0..) |*key, *value, i| {
        key.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 31.0;
        value.* = @as(f32, @floatFromInt(i % 17)) / 19.0;
    }
    var queries: [2 * query_heads * head_dim]f32 = undefined;
    for (&queries, 0..) |*query, i| query.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 27.0;
    const query_positions = [_]u32{ 2, 3 };
    const original_positions = [_]u32{ 0, 1, 2, 3 };

    const input_bytes = keys.len * @sizeOf(f32);
    const k_handle = metal_runtime.termite_metal_buffer_alloc(metal, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(k_handle);
    const v_handle = metal_runtime.termite_metal_buffer_alloc(metal, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(v_handle);
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, k_handle, 0, &keys, input_bytes));
    try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(metal, v_handle, 0, &values, input_bytes));

    inline for ([_]pool_mod.KvDType{ .polar4, .turbo3, .f32, .f16, .int8 }) |dtype| {
        const owned = try MetalKvStorage.create(allocator, metal, dtype, kv_heads, head_dim, 4);
        defer owned.deviceWriteHook().deinit(allocator);
        const binding = try owned.acquireBinding(.{ .sequence_id = 91, .layer_index = 0 }, 0, 0, false, true);
        // This fixture seeds the native slot ABI directly instead of going
        // through reserveLayerKvDevice, so publish the same compact-contiguous
        // layout metadata that the production reserve path would establish.
        binding.logical_contiguous = true;
        binding.physical_base_tokens = 0;
        const layout = owned.rowLayout(kv_heads, head_dim);
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_decode_runtime_update_attention_span_from_f32_key_device_slot(
            metal,
            binding.slot,
            @backingInt(owned.format),
            k_handle,
            0,
            v_handle,
            0,
            rows,
            kv_heads,
            head_dim,
            layout.key_row_bytes,
            layout.base_key_row_bytes,
            layout.v_row_stride,
            0,
        ));
        binding.written_tokens = rows;

        // This fixture writes the physical prefix directly, bypassing the
        // normal writer that establishes its logical row mapping.
        binding.logical_contiguous = true;

        var decoded_keys: [rows * width]f32 = undefined;
        var decoded_values: [rows * width]f32 = undefined;
        try MetalKvStorage.gatherLayerKv(@ptrCast(owned), .{
            .sequence_id = 91,
            .layer_index = 0,
            .token_count = rows,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
        }, &decoded_keys, &decoded_values);
        var gpu_scores: [rows]f32 = undefined;
        try MetalKvStorage.scoreSelectedAttention(@ptrCast(owned), .{
            .sequence_id = 91,
            .layer_index = 0,
            .queries = &queries,
            .query_positions = &query_positions,
            .key_positions = &original_positions,
            .num_heads = query_heads,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .score_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .scores_out = &gpu_scores,
            .check_ctx = null,
            .check_fn = selectionTestNotCancelled,
        });
        const oracle = try selection.scoreConsumerNormalizedAttention(allocator, .{
            .queries = &queries,
            .query_positions = &.{ 2, 3 },
            .keys = &decoded_keys,
            .key_positions = &.{ 0, 1, 2, 3 },
            .num_query_heads = query_heads,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .score_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        });
        defer allocator.free(oracle);
        for (gpu_scores, oracle) |actual, expected| try std.testing.expectApproxEqAbs(expected, @as(f64, actual), 3e-4);

        const selected_rows = [_]u32{ 0, 2, 3 };
        const selected_positions = [_]u32{ 0, 2, 3 };
        const compact = try MetalKvStorage.repackSelectedGlobalKv(@ptrCast(owned), .{
            .sequence_id = 91,
            .layer_index = 0,
            .expected_source_slot = binding.slot,
            .expected_written_tokens = rows,
            .source_token_offsets = &selected_rows,
            .key_positions = &selected_positions,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
        });
        try std.testing.expectEqual(@as(usize, 3), compact.token_count);
        var compact_scores: [3]f32 = undefined;
        try MetalKvStorage.scoreSelectedAttention(@ptrCast(owned), .{
            .sequence_id = 91,
            .layer_index = 0,
            .queries = &queries,
            .query_positions = &query_positions,
            .key_positions = &selected_positions,
            .num_heads = query_heads,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .score_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .scores_out = &compact_scores,
            .check_ctx = null,
            .check_fn = selectionTestNotCancelled,
        });
        var compact_keys: [3 * width]f32 = undefined;
        try MetalKvStorage.gatherLayerKeys(@ptrCast(owned), .{
            .sequence_id = 91,
            .layer_index = 0,
            .token_count = 3,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
        }, &compact_keys);
        const compact_oracle = try selection.scoreConsumerNormalizedAttention(allocator, .{
            .queries = &queries,
            .query_positions = &.{ 2, 3 },
            .keys = &compact_keys,
            .key_positions = &.{ 0, 2, 3 },
            .num_query_heads = query_heads,
            .num_kv_heads = kv_heads,
            .head_dim = head_dim,
            .score_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
        });
        defer allocator.free(compact_oracle);
        for (compact_scores, compact_oracle) |actual, expected| try std.testing.expectApproxEqAbs(expected, @as(f64, actual), 3e-4);

        // Exercise the selected-position attention ABI itself. Query position
        // 2 may attend absolute positions 0 and 2, but must reject position 3
        // even though it is compact physical row 2. This detects a kernel that
        // accidentally substitutes compact row indices for absolute positions.
        const output_values = @as(usize, query_heads) * @as(usize, head_dim);
        const query_head_count: usize = query_heads;
        const head_width: usize = head_dim;
        const output_bytes = output_values * @sizeOf(f32);
        const q_offset_values: usize = 0;
        const q_device = metal_runtime.termite_metal_buffer_alloc(metal, output_bytes, 0) orelse return error.SkipZigTest;
        defer metal_runtime.termite_metal_buffer_release(q_device);
        const out_device = metal_runtime.termite_metal_buffer_alloc(metal, output_bytes, 0) orelse return error.SkipZigTest;
        defer metal_runtime.termite_metal_buffer_release(out_device);
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(
            metal,
            q_device,
            0,
            @ptrCast(queries[q_offset_values..].ptr),
            output_bytes,
        ));
        const compact_blocks = [_]u32{0};
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_decode_runtime_attention_paged_slot_device_selected(
            metal,
            compact.slot,
            compact.format,
            q_device,
            0,
            &compact_blocks,
            compact_blocks.len,
            &selected_positions,
            selected_positions.len,
            compact.page_size_tokens,
            1,
            selected_positions.len,
            query_heads,
            kv_heads,
            head_dim,
            compact.key_row_bytes,
            compact.base_key_row_bytes,
            2,
            0,
            0,
            null,
            0,
            out_device,
            0,
        ));
        var gpu_attention: [output_values]f32 = undefined;
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_download(metal, out_device, 0, &gpu_attention, output_bytes));
        var cpu_attention: [output_values]f32 = @splat(0);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        for (0..query_head_count) |head| {
            var logits: [3]f32 = undefined;
            var best = -std.math.inf(f32);
            for (selected_rows[0..2], 0..) |source_row, retained_row| {
                var dot: f32 = 0;
                for (0..head_width) |d| {
                    dot += queries[q_offset_values + head * head_width + d] * decoded_keys[@as(usize, source_row) * width + d];
                }
                logits[retained_row] = dot * scale;
                best = @max(best, logits[retained_row]);
            }
            var denom: f32 = 0;
            for (logits[0..2]) |logit| denom += @exp(logit - best);
            for (selected_rows[0..2], 0..) |source_row, retained_row| {
                const probability = @exp(logits[retained_row] - best) / denom;
                for (0..head_width) |d| {
                    cpu_attention[head * head_width + d] += probability * decoded_values[@as(usize, source_row) * width + d];
                }
            }
        }
        for (gpu_attention, cpu_attention) |actual, expected| try std.testing.expectApproxEqAbs(expected, actual, 8e-4);
    }
}

test "Metal KV sequence release removes more bindings than one slot batch" {
    const allocator = std.testing.allocator;
    var storage: MetalKvStorage = undefined;
    storage.allocator = allocator;
    storage.slot_map = .empty;
    defer storage.slot_map.deinit(allocator);

    const target_sequence: storage_runtime.SequenceId = 7;
    for (0..metal_runtime.attention_span_slot_capacity + 1) |layer_index| {
        try storage.slot_map.put(allocator, .{
            .sequence_id = target_sequence,
            .layer_index = @intCast(layer_index),
        }, .{
            .slot = 0,
            .sequence_owned = false,
        });
    }
    try storage.slot_map.put(allocator, .{
        .sequence_id = 8,
        .layer_index = 0,
    }, .{
        .slot = 0,
        .sequence_owned = false,
    });

    storage.releaseSequenceSlots(target_sequence);
    try std.testing.expectEqual(@as(usize, 1), storage.slot_map.count());
    try std.testing.expect(storage.slot_map.contains(.{ .sequence_id = 8, .layer_index = 0 }));
}

test "Metal paged KV hooks sharing a runtime lease disjoint slots" {
    if (!build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const runtime = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(runtime);
    if (metal_runtime.termite_metal_decode_runtime_ready(runtime) == 0) return error.SkipZigTest;

    const cached = try MetalKvStorage.create(allocator, runtime, .f16, 1, 2, 2);
    defer cached.deviceWriteHook().deinit(allocator);
    const cached_binding = try cached.acquireBinding(.{ .sequence_id = 1, .layer_index = 0 }, 0, 0, true, false);
    const cached_slot = cached_binding.slot;

    const request_local = try MetalKvStorage.create(allocator, runtime, .f32, 1, 2, 2);
    const local_binding = try request_local.acquireBinding(.{ .sequence_id = 1, .layer_index = 0 }, 0, 0, true, false);
    const local_slot = local_binding.slot;
    try std.testing.expect(cached_slot >= metal_runtime.paged_kv_slot_base);
    try std.testing.expect(local_slot >= metal_runtime.paged_kv_slot_base);
    try std.testing.expect(cached_slot != local_slot);

    // Returning a request-local hook publishes only its own lease. The cache
    // still owns its original slot, while its next layer can reuse the slot
    // that the completed request returned to the shared runtime.
    request_local.deviceWriteHook().deinit(allocator);
    const cached_layer_1 = try cached.acquireBinding(.{ .sequence_id = 1, .layer_index = 1 }, 0, 0, true, false);
    try std.testing.expectEqual(local_slot, cached_layer_1.slot);
    try std.testing.expect(cached_layer_1.slot != cached_slot);
}

test "Metal paged KV keeps layer-global pages across source sequence release" {
    if (!build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const runtime = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(runtime);
    if (metal_runtime.termite_metal_decode_runtime_ready(runtime) == 0) return error.SkipZigTest;

    var storage = try storage_runtime.KvStorageRuntime.init(allocator, .{
        .backend = .metal,
        .dtype = .f32,
        .page_size_tokens = 2,
        .num_layers_packed = 2,
        .num_kv_heads = 1,
        .head_dim = 2,
    });
    defer storage.deinit();
    const metal_storage = try MetalKvStorage.create(allocator, runtime, .f32, 1, 2, 2);
    storage.setDeviceWriteHook(metal_storage.deviceWriteHook());

    const input_bytes = 4 * @sizeOf(f32);
    const k_handle = metal_runtime.termite_metal_buffer_alloc(runtime, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(k_handle);
    const v_handle = metal_runtime.termite_metal_buffer_alloc(runtime, input_bytes, 0) orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_buffer_release(v_handle);

    const source_id = try storage.attachSequence(storage.poolId());
    try storage.appendTokens(source_id, 2);
    const source_layers = [_]struct { k: [4]f32, v: [4]f32 }{
        .{ .k = .{ 1, 2, 3, 4 }, .v = .{ 11, 12, 13, 14 } },
        .{ .k = .{ 21, 22, 23, 24 }, .v = .{ 31, 32, 33, 34 } },
    };
    var original_slots: [2]usize = undefined;
    for (source_layers, 0..) |layer, layer_index| {
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, k_handle, 0, &layer.k, input_bytes));
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, v_handle, 0, &layer.v, input_bytes));
        try storage.writeLayerKvSuffixDevice(.{
            .sequence_id = source_id,
            .layer_index = layer_index,
            .total_token_count = 2,
            .suffix_token_count = 2,
            .position_offset = 0,
            .num_kv_heads = 1,
            .head_dim = 2,
        }, .{ .handle = k_handle, .byte_offset = 0, .byte_len = input_bytes }, .{ .handle = v_handle, .byte_offset = 0, .byte_len = input_bytes });
        original_slots[layer_index] = metal_storage.slot_map.get(.{
            .sequence_id = source_id,
            .layer_index = @intCast(layer_index),
        }).?.slot;
        try std.testing.expectEqual(original_slots[layer_index], metal_storage.global_layer_slots.get(@intCast(layer_index)).?);
    }
    try std.testing.expect(original_slots[0] != original_slots[1]);

    var retained: std.ArrayListUnmanaged(block.KvBlockId) = .empty;
    defer retained.deinit(allocator);
    try storage.retainSequencePrefixBlocks(source_id, 2, &retained);
    try storage.releaseSequence(source_id);
    try std.testing.expect(metal_storage.slot_map.get(.{ .sequence_id = source_id, .layer_index = 0 }) == null);
    try std.testing.expectEqual(original_slots[0], metal_storage.global_layer_slots.get(0).?);
    try std.testing.expectEqual(original_slots[1], metal_storage.global_layer_slots.get(1).?);

    const derived_id = try storage.attachSequenceWithRetainedBlocks(storage.poolId(), retained.items, 2);
    storage.releaseRetainedBlocks(retained.items);
    try storage.appendTokens(derived_id, 1);
    const suffix_layers = [_]struct { k: [2]f32, v: [2]f32 }{
        .{ .k = .{ 5, 6 }, .v = .{ 15, 16 } },
        .{ .k = .{ 25, 26 }, .v = .{ 35, 36 } },
    };
    for (suffix_layers, 0..) |layer, layer_index| {
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, k_handle, 0, &layer.k, layer.k.len * @sizeOf(f32)));
        try std.testing.expectEqual(@as(c_int, 0), metal_runtime.termite_metal_buffer_upload(runtime, v_handle, 0, &layer.v, layer.v.len * @sizeOf(f32)));
        try storage.writeLayerKvSuffixDevice(.{
            .sequence_id = derived_id,
            .layer_index = layer_index,
            .total_token_count = 3,
            .suffix_token_count = 1,
            .position_offset = 0,
            .num_kv_heads = 1,
            .head_dim = 2,
        }, .{ .handle = k_handle, .byte_offset = 0, .byte_len = layer.k.len * @sizeOf(f32) }, .{ .handle = v_handle, .byte_offset = 0, .byte_len = layer.v.len * @sizeOf(f32) });
        try std.testing.expectEqual(original_slots[layer_index], metal_storage.slot_map.get(.{
            .sequence_id = derived_id,
            .layer_index = @intCast(layer_index),
        }).?.slot);
        const gathered = try storage.gatherLayerKv(allocator, derived_id, layer_index, 3);
        defer allocator.free(gathered.k);
        defer allocator.free(gathered.v);
        const expected_k = source_layers[layer_index].k ++ layer.k;
        const expected_v = source_layers[layer_index].v ++ layer.v;
        try std.testing.expectEqualSlices(f32, &expected_k, gathered.k);
        try std.testing.expectEqualSlices(f32, &expected_v, gathered.v);
    }
    try storage.releaseSequence(derived_id);
}
