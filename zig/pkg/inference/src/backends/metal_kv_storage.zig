// Copyright 2026 Antfly, Inc.
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
//! - Multi-sequence via a (sequence_id, layer_index) → slot map. Slots are
//!   reclaimed on `releaseSequence`. Capacity-exhaustion returns
//!   `error.DeviceWriteFallback` so callers downgrade gracefully. Kernel
//!   encode failures also return `DeviceWriteFallback`; the hook contract is
//!   best-effort, not a fatal runtime boundary.
//! - Caller is responsible for keeping the host KvPool in sync if any path
//!   still reads from it. In the full Phase 6 rollout `ensurePagedKvSuffixWritten`
//!   calls the device path first and skips the host write when the hook
//!   accepts; gather-driven paths download from the slot buffer instead.

const std = @import("std");
const build_options = @import("build_options");
const storage_runtime = @import("../runtime/kv/storage_runtime.zig");
const pool_mod = @import("../runtime/kv/pool.zig");
const turboquant = @import("../runtime/kv/turboquant.zig");
const budget_ledger = @import("../runtime/moe/budget_ledger.zig");
const metal_runtime = @import("metal_runtime.zig");

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

/// Session-level slot policy threaded in from the backend that provisions
/// the device-write hook. The KV storage itself has no view of the session,
/// so the compact contract's decisions arrive here by value/pointer.
pub const SlotPolicy = struct {
    /// Grow slot buffers to exactly the required token count instead of the
    /// geometric 1.5x schedule. Set for compact sessions, where slack
    /// capacity is charged against a hard phys-footprint ceiling. The
    /// `TERMITE_METAL_DISABLE_GEOMETRIC_KV_GROWTH` env override forces
    /// exact fit regardless of this flag.
    prefer_exact_kv_growth: bool = false,
    /// Compact residency ledger for observability-only KV accounting.
    /// Owned by the session's CompactRuntimeState, which outlives this
    /// storage; null outside the compact profile.
    ledger: ?*budget_ledger.ResidentBudgetLedger = null,
};

pub const MetalKvStorage = struct {
    allocator: std.mem.Allocator,
    runtime: *metal_runtime.RawMetalDecodeRuntime,
    format: KeyFormat,
    num_kv_heads: u32,
    head_dim: u32,
    page_size_tokens: u16,
    policy: SlotPolicy = .{},
    /// (seq, layer) → slot. Inserted on first write for the pair; removed and
    /// the slot returned to `free_slots` when the sequence is released.
    slot_map: std.AutoHashMapUnmanaged(SlotKey, usize) = .empty,
    /// Slot indices reclaimed from released sequences, consumed LIFO before
    /// bumping `next_slot`. Reclaimed slots have their backing MTLBuffers
    /// preserved for capacity reuse but their slot metadata reset so the C
    /// kernel's incremental-append check starts clean for the next tenant.
    free_slots: std.ArrayListUnmanaged(usize) = .empty,
    /// Highest slot index handed out so far + 1. Grows until it hits the
    /// runtime's attention_span_slot_capacity; after that only `free_slots`
    /// entries can be acquired and exhaustion signals fallback.
    next_slot: usize = 0,
    /// True when logical KV rows for this slot map to a single contiguous
    /// physical token range. Raw f32 device gathers can then expose a borrowed
    /// view at `slot_physical_base_tokens[slot]`; non-contiguous tables must
    /// use the paged attention operator or a gathered-span fallback.
    slot_logical_contiguous: [metal_runtime.attention_span_slot_capacity]bool = [_]bool{false} ** metal_runtime.attention_span_slot_capacity,
    slot_physical_base_tokens: [metal_runtime.attention_span_slot_capacity]usize = [_]usize{0} ** metal_runtime.attention_span_slot_capacity,
    slot_ring_page_count: [metal_runtime.attention_span_slot_capacity]usize = [_]usize{0} ** metal_runtime.attention_span_slot_capacity,
    slot_ring_policy_initialized: [metal_runtime.attention_span_slot_capacity]bool = [_]bool{false} ** metal_runtime.attention_span_slot_capacity,
    slot_buffer_capacity_tokens: [metal_runtime.attention_span_slot_capacity]usize = [_]usize{0} ** metal_runtime.attention_span_slot_capacity,
    /// Ledger-charged device bytes per slot. Mirrors the C-side slot buffer
    /// capacities, which only ever grow: sequence release returns a slot to
    /// the free list but keeps its backing MTLBuffers, so the charge must
    /// survive slot reuse and only positive deltas are ever charged.
    slot_charged_bytes: [metal_runtime.attention_span_slot_capacity]u64 = [_]u64{0} ** metal_runtime.attention_span_slot_capacity,
    charged_total_bytes: u64 = 0,
    cyclic_page_table_cache: storage_runtime.CyclicPageTableCache = .{},

    /// Allocate a MetalKvStorage keyed to `runtime`. The storage does not own
    /// the runtime — callers are responsible for its lifetime. `dtype` must
    /// resolve to a compressed key format; other formats are unsupported by
    /// this fast path (caller falls back to the host write path). `policy`
    /// carries the session-level knobs (compact exact-fit growth and the
    /// residency ledger); pass `.{}` outside the compact profile.
    pub fn create(
        allocator: std.mem.Allocator,
        runtime: *metal_runtime.RawMetalDecodeRuntime,
        dtype: pool_mod.KvDType,
        num_kv_heads: u32,
        head_dim: u32,
        page_size_tokens: u16,
        policy: SlotPolicy,
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
            .policy = policy,
        };
        return self;
    }

    /// Acquire a slot for the given (sequence, layer) pair. Reuses the slot
    /// already bound to this pair if one exists, otherwise pulls from the
    /// free list, otherwise bumps `next_slot` up to the capacity ceiling.
    /// Returns `error.DeviceWriteFallback` when exhausted so the caller can
    /// downgrade to the host path for this write.
    fn acquireSlot(self: *MetalKvStorage, key: SlotKey) !usize {
        if (self.slot_map.get(key)) |slot| return slot;
        const slot: usize = if (self.free_slots.pop()) |reused| blk: {
            _ = metal_runtime.termite_metal_decode_runtime_reset_attention_span_slot(self.runtime, reused);
            break :blk reused;
        } else if (self.next_slot < metal_runtime.attention_span_slot_capacity) blk: {
            const s = self.next_slot;
            self.next_slot += 1;
            break :blk s;
        } else {
            return error.DeviceWriteFallback;
        };
        self.slot_logical_contiguous[slot] = false;
        self.slot_physical_base_tokens[slot] = 0;
        self.slot_ring_page_count[slot] = 0;
        self.slot_ring_policy_initialized[slot] = false;
        self.slot_buffer_capacity_tokens[slot] = 0;
        try self.slot_map.put(self.allocator, key, slot);
        return slot;
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

    fn requestedRingPageCount(
        page_size_tokens: u16,
        sliding_window: usize,
        max_inflight_tokens: usize,
        allow_swa_ring: bool,
    ) !usize {
        if (!allow_swa_ring or sliding_window == 0 or max_inflight_tokens == 0 or page_size_tokens == 0) return 0;
        // Production requests opt in through the typed KV policy. Keep only a
        // hard rollback flag here; the legacy ENABLE flag remains harmless.
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

    /// Pure capacity decision for one slot reservation. Ring slots are fixed
    /// at their ring span and never grow; exact fit is chosen by the session
    /// policy (compact contract) or the env override; everything else takes
    /// the geometric 1.5x growth schedule to amortize copy-on-grow.
    fn slotCapacityTargetTokens(
        current: usize,
        required_tokens: usize,
        page_size_tokens: usize,
        ring_page_count: usize,
        prefer_exact: bool,
    ) !usize {
        if (ring_page_count > 0 or prefer_exact) return required_tokens;
        return storage_runtime.geometricKvTokenCapacity(current, required_tokens, page_size_tokens);
    }

    /// Device bytes the C runtime allocates for a slot reservation. Mirrors
    /// `termite_metal_decode_runtime_reserve_attention_span_slot_buffers`:
    /// one encoded-key buffer of `tokens * key_row_bytes` plus one value
    /// buffer of `tokens * v_row_stride * element_bytes` (f16 values are
    /// halfs, every other format stores f32 values).
    fn slotReservedBytes(
        format: KeyFormat,
        token_capacity: usize,
        key_row_bytes: usize,
        v_row_stride: usize,
    ) u64 {
        const v_element_bytes: u64 = if (format == .f16) @sizeOf(u16) else @sizeOf(f32);
        const tokens: u64 = @intCast(token_capacity);
        const key_bytes = tokens *| @as(u64, @intCast(key_row_bytes));
        const v_bytes = tokens *| @as(u64, @intCast(v_row_stride)) *| v_element_bytes;
        return key_bytes +| v_bytes;
    }

    /// Observability-only ledger accounting for slot growth. The C-side slot
    /// buffers only ever grow, and released sequences keep their buffers for
    /// reuse, so `slot_charged_bytes` tracks the high water per slot and only
    /// positive deltas are charged. The whole balance is returned on
    /// `hookDeinit`; buffers that outlive this storage in the runtime's slot
    /// pool are then observed only by the profile's phys-footprint sampler.
    fn chargeSlotReservation(self: *MetalKvStorage, slot: usize, target_tokens: usize, key_row_bytes: usize, v_row_stride: usize) void {
        const ledger = self.policy.ledger orelse return;
        const reserved = slotReservedBytes(self.format, target_tokens, key_row_bytes, v_row_stride);
        const charged = self.slot_charged_bytes[slot];
        if (reserved <= charged) return;
        const delta = reserved - charged;
        ledger.noteCommittedObserved(.kv_cache, delta);
        self.slot_charged_bytes[slot] = reserved;
        self.charged_total_bytes +|= delta;
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
        const prefer_exact = self.policy.prefer_exact_kv_growth or
            envFlagEnabled("TERMITE_METAL_DISABLE_GEOMETRIC_KV_GROWTH");
        const target = try slotCapacityTargetTokens(
            current,
            required_tokens,
            page_size_tokens,
            ring_page_count,
            prefer_exact,
        );
        const rc = metal_runtime.termite_metal_decode_runtime_reserve_attention_span_slot_buffers(
            self.runtime,
            slot,
            @intFromEnum(self.format),
            target,
            key_row_bytes,
            v_row_stride,
        );
        if (rc != 0) return error.DeviceWriteFallback;
        self.slot_buffer_capacity_tokens[slot] = target;
        self.chargeSlotReservation(slot, target, key_row_bytes, v_row_stride);
    }

    fn requiredTokenCapacity(block_offsets: []const u32, page_size_tokens: usize) !usize {
        var required: usize = 0;
        for (block_offsets) |offset| {
            const end = std.math.add(usize, @as(usize, offset), page_size_tokens) catch return error.KvCapacityTooSmall;
            required = @max(required, end);
        }
        return required;
    }

    /// Release every slot bound to `sequence_id` back to the free pool and
    /// reset their slot metadata so the next tenant re-encodes from scratch.
    /// Safe to call for a sequence that holds no slots.
    fn releaseSequenceSlots(self: *MetalKvStorage, sequence_id: storage_runtime.SequenceId) void {
        var it = self.slot_map.iterator();
        var to_release: std.ArrayListUnmanaged(SlotKey) = .empty;
        defer to_release.deinit(self.allocator);
        while (it.next()) |entry| {
            if (entry.key_ptr.sequence_id != sequence_id) continue;
            to_release.append(self.allocator, entry.key_ptr.*) catch {
                // On allocation failure, reset the slot in place — we won't
                // be able to reuse it, but we also won't double-bind it.
                const slot = entry.value_ptr.*;
                _ = metal_runtime.termite_metal_decode_runtime_reset_attention_span_slot(self.runtime, slot);
                self.slot_buffer_capacity_tokens[slot] = 0;
                continue;
            };
        }
        for (to_release.items) |key| {
            if (self.slot_map.fetchRemove(key)) |removed| {
                _ = metal_runtime.termite_metal_decode_runtime_reset_attention_span_slot(self.runtime, removed.value);
                self.slot_buffer_capacity_tokens[removed.value] = 0;
                self.free_slots.append(self.allocator, removed.value) catch {
                    // If we can't record the reuse slot, the slot leaks for
                    // this session — it still gets cleared on reset_state.
                };
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
        };
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
        const num_kv_heads: u32 = write.num_kv_heads;
        const head_dim: u32 = write.head_dim;

        const layout = self.rowLayout(num_kv_heads, head_dim);
        const key_row_bytes = layout.key_row_bytes;
        const base_key_row_bytes = layout.base_key_row_bytes;
        const v_row_stride = layout.v_row_stride;

        const expected_elems = write.suffix_token_count * v_row_stride;
        const expected_bytes = expected_elems * @sizeOf(f32);
        if (k.byte_len < expected_bytes or v.byte_len < expected_bytes) return error.InvalidKvShape;

        const slot = try self.acquireSlot(.{
            .sequence_id = write.sequence_id,
            .layer_index = @intCast(write.layer_index),
        });
        const requested_ring_pages = try requestedRingPageCount(
            write.page_size_tokens,
            write.sliding_window,
            write.max_inflight_tokens,
            write.allow_swa_ring,
        );
        const ring_page_count = try self.configureSlotRingPolicy(slot, requested_ring_pages);
        if (traceKvGather()) std.debug.print(
            "kv-write: seq={d} layer={d} suffix={d} total={d} slot={d} ring_pages={d}\n",
            .{ write.sequence_id, write.layer_index, write.suffix_token_count, write.total_token_count, slot, ring_page_count },
        );

        const rc = if (write.logical_blocks) |logical_blocks| paged: {
            if (logical_blocks.len == 0 or write.page_size_tokens == 0) break :paged -9999;
            const needed_blocks = std.math.divCeil(usize, write.total_token_count, write.page_size_tokens) catch break :paged -9999;
            if (logical_blocks.len < needed_blocks) break :paged -9999;
            const block_offsets_result = self.pageTokenOffsets(
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
            self.slot_physical_base_tokens[slot] = if (ring_page_count > 0) 0 else block_offsets[0];
            self.slot_logical_contiguous[slot] = ring_page_count == 0;
            if (ring_page_count == 0) {
                for (block_offsets, 0..) |offset, block_idx| {
                    const expected = self.slot_physical_base_tokens[slot] + block_idx * @as(usize, write.page_size_tokens);
                    if (offset != expected) {
                        self.slot_logical_contiguous[slot] = false;
                        break;
                    }
                }
            }
            break :paged metal_runtime.termite_metal_decode_runtime_update_attention_paged_from_f32_key_device_slot(
                self.runtime,
                slot,
                @intFromEnum(self.format),
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
            self.slot_logical_contiguous[slot] = true;
            self.slot_physical_base_tokens[slot] = 0;
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
                @intFromEnum(self.format),
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
    }

    fn reserveLayerKvDevice(
        ctx: *anyopaque,
        reserve: storage_runtime.DeviceKvLayerReserve,
    ) anyerror!void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (reserve.token_capacity == 0) return;
        const layout = self.rowLayout(reserve.num_kv_heads, reserve.head_dim);
        if (layout.key_row_bytes == 0 or layout.v_row_stride == 0) return error.InvalidKvShape;

        const slot = try self.acquireSlot(.{
            .sequence_id = reserve.sequence_id,
            .layer_index = @intCast(reserve.layer_index),
        });
        const requested_ring_pages = try requestedRingPageCount(
            reserve.page_size_tokens,
            reserve.sliding_window,
            reserve.max_inflight_tokens,
            reserve.allow_swa_ring,
        );
        const ring_page_count = try self.configureSlotRingPolicy(slot, requested_ring_pages);
        if (traceKvGather()) std.debug.print(
            "kv-reserve: seq={d} layer={d} capacity={d} slot={d} ring_pages={d}\n",
            .{ reserve.sequence_id, reserve.layer_index, reserve.token_capacity, slot, ring_page_count },
        );
        if (ring_page_count > 0 and reserve.logical_blocks == null) return error.DeviceWriteFallback;

        const token_capacity = if (ring_page_count > 0)
            std.math.mul(usize, ring_page_count, reserve.page_size_tokens) catch return error.KvCapacityTooSmall
        else if (reserve.logical_blocks) |logical_blocks| blk: {
            if (logical_blocks.len == 0 or reserve.page_size_tokens == 0) break :blk reserve.token_capacity;
            const needed_blocks = std.math.divCeil(usize, reserve.token_capacity, reserve.page_size_tokens) catch break :blk reserve.token_capacity;
            if (logical_blocks.len < needed_blocks) break :blk reserve.token_capacity;
            var capacity_tokens: usize = 0;
            self.slot_physical_base_tokens[slot] = std.math.mul(
                usize,
                @as(usize, logical_blocks[0]),
                reserve.page_size_tokens,
            ) catch return error.KvCapacityTooSmall;
            self.slot_logical_contiguous[slot] = true;
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
                    self.slot_physical_base_tokens[slot],
                    expected_delta,
                ) catch return error.KvCapacityTooSmall;
                if (block_start != expected) self.slot_logical_contiguous[slot] = false;
                const block_end = std.math.add(
                    usize,
                    block_start,
                    reserve.page_size_tokens,
                ) catch return error.KvCapacityTooSmall;
                if (block_end > capacity_tokens) capacity_tokens = block_end;
            }
            break :blk capacity_tokens;
        } else blk: {
            self.slot_logical_contiguous[slot] = true;
            self.slot_physical_base_tokens[slot] = 0;
            break :blk reserve.token_capacity;
        };
        if (ring_page_count > 0) {
            self.slot_logical_contiguous[slot] = false;
            self.slot_physical_base_tokens[slot] = 0;
        }

        try self.reserveSlotCapacity(
            slot,
            token_capacity,
            reserve.page_size_tokens,
            ring_page_count,
            layout.key_row_bytes,
            layout.v_row_stride,
        );
    }

    fn hookDeinit(ctx: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.policy.ledger) |ledger| {
            ledger.release(.kv_cache, self.charged_total_bytes);
        }
        self.slot_map.deinit(allocator);
        self.free_slots.deinit(allocator);
        self.cyclic_page_table_cache.deinit(allocator);
        allocator.destroy(self);
    }

    fn releaseSequenceOp(ctx: *anyopaque, sequence_id: storage_runtime.SequenceId) void {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        self.releaseSequenceSlots(sequence_id);
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
        const slot = self.slot_map.get(.{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        }) orelse return error.DeviceReadFallback;
        if (self.slot_ring_page_count[slot] > 0) return error.RingKvRequiresPagedAttention;
        if (!self.slot_logical_contiguous[slot]) return error.DeviceReadFallback;
        const info = try self.slotInfo(slot);
        if (info.tokens != 0 and info.tokens < gather.token_count) return error.DeviceReadFallback;

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
        const physical_base = self.slot_physical_base_tokens[slot];
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
                    try turboquant.decodeTurbo3Key(row_bytes[0..base_bytes], row_dst, num_kv_heads, head_dim);
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

    fn gatherLayerKvDevice(
        ctx: *anyopaque,
        gather: storage_runtime.DeviceKvLayerGather,
    ) anyerror!storage_runtime.DeviceKvLayer {
        const self: *MetalKvStorage = @ptrCast(@alignCast(ctx));
        if (self.format != .raw_f32) return error.DeviceReadFallback;
        const token_width: usize = @as(usize, gather.num_kv_heads) * @as(usize, gather.head_dim);
        const byte_len = gather.token_count * token_width * @sizeOf(f32);

        if (traceKvGather()) std.debug.print("kv-read: seq={d} layer={d} tokens={d}\n", .{ gather.sequence_id, gather.layer_index, gather.token_count });
        const slot = self.slot_map.get(.{
            .sequence_id = gather.sequence_id,
            .layer_index = @intCast(gather.layer_index),
        }) orelse return error.DeviceReadFallback;
        if (self.slot_ring_page_count[slot] > 0) return error.RingKvRequiresPagedAttention;
        if (!self.slot_logical_contiguous[slot]) return error.DeviceReadFallback;
        const info = try self.slotInfo(slot);
        if (info.tokens < gather.token_count) return error.DeviceReadFallback;
        if (info.key_row_bytes != token_width * @sizeOf(f32)) return error.DeviceReadFallback;
        if (info.v_row_stride != token_width) return error.DeviceReadFallback;

        const k_handle = info.encoded_key_handle orelse return error.DeviceReadFallback;
        const v_handle = info.v_handle orelse return error.DeviceReadFallback;
        const byte_offset = self.slot_physical_base_tokens[slot] * token_width * @sizeOf(f32);
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
            .token_count = gather.token_count,
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
        const slot = self.slot_map.get(key) orelse return error.DeviceReadFallback;
        const ring_page_count = self.slot_ring_page_count[slot];
        const info_opt = self.slotInfo(slot) catch |err| blk: {
            if (!active_frame) return err;
            break :blk null;
        };
        const position_offset = if (info_opt) |info| blk: {
            if (ring_page_count == 0 and !active_frame and info.tokens < gather.token_count) return error.DeviceReadFallback;
            if (ring_page_count == 0 and info.tokens != 0 and info.tokens < gather.token_count and !active_frame) return error.DeviceReadFallback;
            if (info.key_row_bytes != 0 and info.key_row_bytes != key_row_bytes) return error.DeviceReadFallback;
            if (info.v_row_stride != 0 and info.v_row_stride != token_width) return error.DeviceReadFallback;
            break :blk info.position_offset;
        } else 0;
        return .{
            .runtime = @ptrCast(self.runtime),
            .slot = slot,
            .format = @intFromEnum(self.format),
            .token_count = gather.token_count,
            .key_row_bytes = key_row_bytes,
            .base_key_row_bytes = base_key_row_bytes,
            .v_row_stride = token_width,
            .page_size_tokens = self.page_size_tokens,
            .ring_page_count = ring_page_count,
            .position_offset = position_offset,
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

test "Metal KV rollout flags recognize common false values" {
    try std.testing.expect(!MetalKvStorage.flagValueEnabled(""));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("0"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("false"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("NO"));
    try std.testing.expect(!MetalKvStorage.flagValueEnabled("Off"));
    try std.testing.expect(MetalKvStorage.flagValueEnabled("1"));
    try std.testing.expect(MetalKvStorage.flagValueEnabled("true"));
}

const hook_vtable: storage_runtime.DeviceWriteHook.VTable = .{
    .writeLayerKvSuffix = MetalKvStorage.writeLayerKvSuffix,
    .gatherLayerKv = MetalKvStorage.gatherLayerKv,
    .gatherLayerKvDevice = MetalKvStorage.gatherLayerKvDevice,
    .pagedLayerKvDevice = MetalKvStorage.pagedLayerKvDevice,
    .reserveLayerKvDevice = MetalKvStorage.reserveLayerKvDevice,
    .releaseSequence = MetalKvStorage.releaseSequenceOp,
    .deinit = MetalKvStorage.hookDeinit,
};

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
        MetalKvStorage.create(allocator, fake_runtime, .int4, 8, 128, 256, .{}),
    );
    try std.testing.expectError(
        error.DeviceWriteFormatUnsupported,
        MetalKvStorage.create(allocator, fake_runtime, .fp8, 8, 128, 256, .{}),
    );
}

test "Metal KV slot capacity growth is exact for rings and compact sessions" {
    // Ring slots are fixed at their ring span regardless of policy.
    try std.testing.expectEqual(
        @as(usize, 640),
        try MetalKvStorage.slotCapacityTargetTokens(0, 640, 16, 40, false),
    );
    try std.testing.expectEqual(
        @as(usize, 640),
        try MetalKvStorage.slotCapacityTargetTokens(512, 640, 16, 40, true),
    );
    // Exact-fit policy (compact contract or env override) never overshoots.
    try std.testing.expectEqual(
        @as(usize, 2488),
        try MetalKvStorage.slotCapacityTargetTokens(2432, 2488, 16, 0, true),
    );
    // Default policy keeps the geometric 1.5x schedule from storage_runtime.
    try std.testing.expectEqual(
        @as(usize, 100),
        try MetalKvStorage.slotCapacityTargetTokens(0, 100, 16, 0, false),
    );
    try std.testing.expectEqual(
        @as(usize, 160),
        try MetalKvStorage.slotCapacityTargetTokens(100, 101, 16, 0, false),
    );
}

test "Metal KV slot reservation bytes mirror the C-side buffer math" {
    // f16: keys are halfs (key_row_bytes already reflects that) and values
    // are halfs. One token of 8 heads x 128 dims = 1024 values.
    try std.testing.expectEqual(
        @as(u64, 1024 * 2 + 1024 * 2),
        MetalKvStorage.slotReservedBytes(.f16, 1, 1024 * 2, 1024),
    );
    // raw f32 stores 4-byte keys and values.
    try std.testing.expectEqual(
        @as(u64, 2 * (1024 * 4 + 1024 * 4)),
        MetalKvStorage.slotReservedBytes(.raw_f32, 2, 1024 * 4, 1024),
    );
    // Compressed key formats keep f32 values.
    try std.testing.expectEqual(
        @as(u64, 3 * (40 + 1024 * 4)),
        MetalKvStorage.slotReservedBytes(.polar4, 3, 40, 1024),
    );
}

test "Metal KV ledger charges only positive high-water deltas" {
    if (!build_options.enable_metal) return error.SkipZigTest;

    var ledger = budget_ledger.ResidentBudgetLedger.init(std.math.maxInt(u64), 0);
    const fake_runtime: *metal_runtime.RawMetalDecodeRuntime = @ptrFromInt(@alignOf(usize));
    var storage = MetalKvStorage{
        .allocator = std.testing.allocator,
        .runtime = fake_runtime,
        .format = .f16,
        .num_kv_heads = 8,
        .head_dim = 128,
        .page_size_tokens = 16,
        .policy = .{ .prefer_exact_kv_growth = true, .ledger = &ledger },
    };

    const kv_index = @intFromEnum(budget_ledger.BudgetCategory.kv_cache);
    storage.chargeSlotReservation(3, 128, 2048, 1024);
    const first = MetalKvStorage.slotReservedBytes(.f16, 128, 2048, 1024);
    try std.testing.expectEqual(first, ledger.snapshot().committed_by_category[kv_index]);

    // Growth charges the delta only.
    storage.chargeSlotReservation(3, 256, 2048, 1024);
    const second = MetalKvStorage.slotReservedBytes(.f16, 256, 2048, 1024);
    try std.testing.expectEqual(second, ledger.snapshot().committed_by_category[kv_index]);
    try std.testing.expectEqual(second, storage.charged_total_bytes);

    // A reused slot re-reserved below its high water charges nothing: the
    // C-side buffer never shrank.
    storage.chargeSlotReservation(3, 64, 2048, 1024);
    try std.testing.expectEqual(second, ledger.snapshot().committed_by_category[kv_index]);

    // Deinit-time release returns the whole balance.
    ledger.release(.kv_cache, storage.charged_total_bytes);
    try std.testing.expectEqual(@as(u64, 0), ledger.snapshot().committed_by_category[kv_index]);
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
