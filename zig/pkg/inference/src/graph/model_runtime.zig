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

const std = @import("std");
const cache_mod = @import("cache.zig");
const activations = @import("../backends/activations.zig");
const InferenceExecutionControl = @import("../execution_control.zig").InferenceExecutionControl;
const Interruption = @import("../execution_control.zig").Interruption;
const UninterruptibleGuard = @import("../execution_control.zig").UninterruptibleGuard;
const ops = @import("../ops/ops.zig");
const contracts = @import("backend_contracts.zig");
const recent_prefill_query_capture = @import("recent_prefill_query_capture.zig");
const kv_storage = @import("../runtime/kv/storage_runtime.zig");
const owner_alloc = @import("../runtime/kv/host_allocation_tracker.zig");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");

/// Ephemeral authority used to bind a newly-created mutable compiled runtime
/// to one request allocation scope.  Cached model executors must never retain
/// this value; the backend consumes it into its request-local storage tracker.
pub const ContextOwnerBinding = struct {
    owner: *owner_alloc.RequestAllocationOwner,
    witness: owner_alloc.RequestAllocationOwner.ContextScopeWitness,

    pub fn validate(self: ContextOwnerBinding) !void {
        try self.owner.validateContextScopeWitness(self.witness);
    }
};

test "compiled runtime owner binding rejects stale identity before backend dispatch" {
    var owner = try owner_alloc.RequestAllocationOwner.init(null);
    defer _ = owner.release();
    const witness = try owner.beginContextScope(7);
    var calls: usize = 0;
    const Probe = struct {
        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: PrefillRequest) !ModelOutput {
            return error.UnsupportedShape;
        }
        fn deinit(_: *anyopaque) void {}
        fn bind(raw: *anyopaque, _: ContextOwnerBinding) !void {
            const count: *usize = @ptrCast(@alignCast(raw));
            count.* += 1;
        }
        const vtable = ModelRuntime.VTable{
            .prefill = prefill,
            .deinit = deinit,
            .bind_context_owner = bind,
        };
    };
    var runtime = ModelRuntime{ .ptr = &calls, .vtable = &Probe.vtable };
    try runtime.bindContextOwner(.{ .owner = &owner, .witness = witness });
    try std.testing.expectEqual(@as(usize, 1), calls);
    // A continuation under the same live scope remains valid. The production
    // backend's tracker check makes this bind idempotent.
    try runtime.bindContextOwner(.{ .owner = &owner, .witness = witness });
    try std.testing.expectEqual(@as(usize, 2), calls);
    var stale = witness;
    stale.load_generation +%= 1;
    try std.testing.expectError(error.StaleContextOwnerScope, runtime.bindContextOwner(.{ .owner = &owner, .witness = stale }));
    try std.testing.expectEqual(@as(usize, 2), calls);
}

pub fn sameOptionalModelIdentity(a: ?checkpoint.Identity, b: ?checkpoint.Identity) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.?.eql(b.?);
}

test "compiled runtime identity scope covers none some and full mismatch" {
    const a = checkpoint.Identity{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    var b = a;
    b.layout[0] ^= 1;
    try std.testing.expect(sameOptionalModelIdentity(null, null));
    try std.testing.expect(!sameOptionalModelIdentity(null, a));
    try std.testing.expect(!sameOptionalModelIdentity(a, null));
    try std.testing.expect(sameOptionalModelIdentity(a, a));
    try std.testing.expect(!sameOptionalModelIdentity(a, b));
}

pub const RecentPrefillQueryCaptureDescriptor = recent_prefill_query_capture.Descriptor;
pub const RecentPrefillQueryNativePlanRequest = recent_prefill_query_capture.NativePlanRequest;
pub const RecentPrefillQueryNativePlan = recent_prefill_query_capture.NativePlan;
pub const RecentPrefillQueryNativePlanBatch = recent_prefill_query_capture.NativePlanBatch;
pub const RecentPrefillQueryNativeBatchTransport = recent_prefill_query_capture.NativeBatchTransport;
pub const RecentPrefillQueryCaptureEncodeRequest = recent_prefill_query_capture.EncodeRequest;
pub const RecentPrefillQueryCaptureCompletion = recent_prefill_query_capture.Completion;

/// Backend-owned model runtime state.
///
/// This is intentionally separate from PartitionExecutor. A PartitionExecutor
/// fills node values inside a host-managed graph execution. A ModelRuntime is
/// the mutable per-session state for a compiled model that owns the whole
/// model execution path, including backend-side cache/KV state as backends grow
/// that capability.
pub const ModelRuntime = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// Immutable authority scope copied from the executor at construction.
    model_identity: ?checkpoint.Identity = null,

    pub const VTable = struct {
        capabilities: ?*const fn (ctx: *anyopaque) RuntimeCapabilities = null,
        prepare: ?*const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: PrepareRequest,
        ) anyerror!bool = null,
        prefill: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: PrefillRequest,
        ) anyerror!ModelOutput,
        decode: ?*const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: DecodeRequest,
        ) anyerror!ModelOutput = null,
        decode_sample: ?*const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: SampledDecodeRequest,
        ) anyerror!SampledDecodeOutput = null,
        /// Returns true only when `decode_sample` can execute this exact
        /// request without falling back to a second model forward. Callers
        /// must check this before submitting a decode frame.
        supports_sample_decode_request: ?*const fn (
            ctx: *anyopaque,
            request: SampledDecodeRequest,
        ) bool = null,
        decode_greedy: ?*const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: DecodeRequest,
        ) anyerror!GreedyDecodeOutput = null,
        verify_greedy_transaction: ?*const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            request: GreedyVerifyRequest,
        ) anyerror!GreedyVerifyOutput = null,
        deinit: *const fn (ctx: *anyopaque) void,
        reset: ?*const fn (ctx: *anyopaque) anyerror!void = null,
        finish_prefill_for_decode: ?*const fn (ctx: *anyopaque, request: PrepareRequest) anyerror!bool = null,
        exact_paged_kv_authority: ?*const fn (ctx: *anyopaque) anyerror!ExactPagedKvSourceAuthority = null,
        export_exact_paged_kv_ranges: ?*const fn (ctx: *anyopaque, request: ExactPagedKvRangeExportRequest) anyerror!kv_storage.ExactLayerRangeExportResult = null,
        handoff_exact_paged_kv: ?*const fn (ctx: *anyopaque, request: ExactPagedKvHandoffRequest) anyerror!ExactPagedKvHandoffReceipt = null,
        discard_exact_paged_kv_source: ?*const fn (ctx: *anyopaque, request: ExactPagedKvSourceDiscardRequest) anyerror!ExactPagedKvSourceDiscardReceipt = null,
        bind_context_owner: ?*const fn (ctx: *anyopaque, binding: ContextOwnerBinding) anyerror!void = null,
        /// Drain backend-owned request storage and reconcile its owner ledger
        /// before a committed owner snapshot is observed.
        drain_context_owner_snapshot: ?*const fn (ctx: *anyopaque, binding: ContextOwnerBinding) anyerror!void = null,
        context_owner_binding_state: ?*const fn (ctx: *anyopaque, binding: ContextOwnerBinding) ContextOwnerBindingState = null,
        debug_timing_stats: ?*const fn (ctx: *anyopaque) RuntimeDebugTimingStats = null,
        reset_debug_timing_stats: ?*const fn (ctx: *anyopaque) void = null,
        print_debug_timing: ?*const fn (ctx: *anyopaque) void = null,
    };

    pub const ContextOwnerBindingState = enum { unbound, exact, other };

    pub fn contextOwnerBindingState(self: *const ModelRuntime, binding: ContextOwnerBinding) !ContextOwnerBindingState {
        try binding.validate();
        const state_fn = self.vtable.context_owner_binding_state orelse return error.ContextPhysicalAccountingUnavailable;
        return state_fn(self.ptr, binding);
    }

    pub fn capabilities(self: *const ModelRuntime) RuntimeCapabilities {
        const caps_fn = self.vtable.capabilities orelse return .{
            .supports_decode = self.vtable.decode != null,
            .supports_sample_decode = self.vtable.decode_sample != null,
            .supports_greedy_decode = self.vtable.decode_greedy != null,
        };
        var caps = caps_fn(self.ptr);
        caps.supports_decode = caps.supports_decode and self.vtable.decode != null;
        caps.supports_sample_decode = caps.supports_sample_decode and self.vtable.decode_sample != null;
        caps.supports_greedy_decode = caps.supports_greedy_decode and self.vtable.decode_greedy != null;
        caps.supports_greedy_verify_transaction = caps.supports_greedy_verify_transaction and
            self.vtable.verify_greedy_transaction != null;
        return caps;
    }

    pub fn reset(self: *ModelRuntime) !void {
        if (self.vtable.reset) |reset_fn| return reset_fn(self.ptr);
    }

    pub fn bindContextOwner(self: *ModelRuntime, binding: ContextOwnerBinding) !void {
        try binding.validate();
        const bind_fn = self.vtable.bind_context_owner orelse
            return error.ContextPhysicalAccountingUnavailable;
        try bind_fn(self.ptr, binding);
    }

    pub fn drainContextOwnerSnapshot(self: *ModelRuntime, binding: ContextOwnerBinding) !void {
        try binding.validate();
        const drain_fn = self.vtable.drain_context_owner_snapshot orelse
            return error.ContextOwnerDrainUnsupported;
        try drain_fn(self.ptr, binding);
    }

    pub fn finishPrefillForDecode(self: *ModelRuntime, request: PrepareRequest) !bool {
        const finish_fn = self.vtable.finish_prefill_for_decode orelse {
            try request.check();
            return false;
        };
        // The backend hook owns both request checks so it can arm cleanup
        // before the first cancellation point and retain that cleanup through
        // the post-transition check.
        return finish_fn(self.ptr, request);
    }

    /// Copies the completed exact prefill cache into a fresh request-owned
    /// paged destination. The backend must leave the source unchanged. The
    /// caller owns destination rollback if this operation fails.
    pub fn handoffExactPagedKv(self: *ModelRuntime, request: ExactPagedKvHandoffRequest) !ExactPagedKvHandoffReceipt {
        try request.validate();
        const handoff_fn = self.vtable.handoff_exact_paged_kv orelse return error.UnsupportedExactPagedKvHandoff;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        const receipt = try handoff_fn(self.ptr, request);
        try request.validateReceipt(receipt);
        try request.check();
        return receipt;
    }

    pub fn exactPagedKvAuthority(self: *ModelRuntime) !ExactPagedKvSourceAuthority {
        const authority_fn = self.vtable.exact_paged_kv_authority orelse return error.UnsupportedExactPagedKvHandoff;
        const authority = try authority_fn(self.ptr);
        try authority.validate();
        return authority;
    }

    pub fn exportExactPagedKvRanges(self: *ModelRuntime, request: ExactPagedKvRangeExportRequest) !kv_storage.ExactLayerRangeExportResult {
        try request.validate();
        const expected = try (kv_storage.ExactLayerRangeExportRequest{
            .source_sequence = request.authority.source_sequence,
            .expected_written_tokens = request.authority.logical_tokens,
            .ranges = request.ranges,
            .max_bytes = request.max_copy_bytes,
            .check_ctx = request.check_ctx,
            .check_fn = request.check_fn,
        }).validate();
        const export_fn = self.vtable.export_exact_paged_kv_ranges orelse return error.UnsupportedExactPagedKvRangeExport;
        const result = try export_fn(self.ptr, request);
        if (!result.fenced or result.layers != request.ranges.len or result.copied_bytes != expected)
            return error.InvalidExactPagedKvRangeExportReceipt;
        try request.check();
        return result;
    }

    pub fn exactPagedKvRangeExporter(self: *ModelRuntime) kv_storage.ExactLayerRangeExporter {
        return .{ .ctx = self, .export_fn = exactPagedKvRangeExportThunk };
    }

    fn exactPagedKvRangeExportThunk(raw: *anyopaque, authority: kv_storage.ExactLayerRangeSourceAuthority, ranges: []const kv_storage.ExactLayerRangeExport, max_bytes: usize, check_ctx: ?*anyopaque, check_fn: ?*const fn (?*anyopaque) anyerror!void) !kv_storage.ExactLayerRangeExportResult {
        const self: *ModelRuntime = @ptrCast(@alignCast(raw));
        return self.exportExactPagedKvRanges(.{ .authority = authority, .ranges = ranges, .max_copy_bytes = max_bytes, .check_ctx = check_ctx, .check_fn = check_fn });
    }

    pub fn discardExactPagedKvSource(self: *ModelRuntime, request: ExactPagedKvSourceDiscardRequest) !ExactPagedKvSourceDiscardReceipt {
        try request.check();
        const discard_fn = self.vtable.discard_exact_paged_kv_source orelse return error.UnsupportedExactPagedKvHandoff;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        const receipt = try discard_fn(self.ptr, request);
        if (!std.mem.eql(u8, &receipt.authority_sha256, &request.authority_sha256) or
            receipt.source_revision != request.source_revision or receipt.source_sequence != request.source_sequence or
            receipt.slots == 0 or receipt.physically_released_bytes == 0 or receipt.live_bytes_after != 0 or
            receipt.live_bytes_before < receipt.live_bytes_after or
            receipt.live_bytes_before - receipt.live_bytes_after != receipt.physically_released_bytes)
            return error.InvalidExactPagedKvSourceDiscardReceipt;
        try request.check();
        return receipt;
    }

    pub fn prepare(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: PrepareRequest,
    ) !bool {
        try request.check();
        const prepare_fn = self.vtable.prepare orelse return false;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        const prepared = try prepare_fn(self.ptr, allocator, request);
        try request.check();
        return prepared;
    }

    pub fn debugTimingStats(self: *const ModelRuntime) RuntimeDebugTimingStats {
        if (self.vtable.debug_timing_stats) |stats_fn| {
            return stats_fn(self.ptr);
        }
        return .{};
    }

    pub fn resetDebugTimingStats(self: *ModelRuntime) void {
        if (self.vtable.reset_debug_timing_stats) |reset_fn| {
            reset_fn(self.ptr);
        }
    }

    pub fn printDebugTiming(self: *const ModelRuntime) void {
        if (self.vtable.print_debug_timing) |print_fn| {
            print_fn(self.ptr);
        }
    }

    pub fn prefill(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: PrefillRequest,
    ) !ModelOutput {
        try request.check();
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        var output = try self.vtable.prefill(self.ptr, allocator, request);
        errdefer output.deinit(allocator);
        try request.check();
        return output;
    }

    pub fn decode(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: DecodeRequest,
    ) !ModelOutput {
        try request.check();
        const decode_fn = self.vtable.decode orelse return error.UnsupportedDecode;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        var output = try decode_fn(self.ptr, allocator, request);
        errdefer output.deinit(allocator);
        try request.check();
        return output;
    }

    pub fn decodeGreedy(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: DecodeRequest,
    ) !GreedyDecodeOutput {
        try request.check();
        const decode_greedy_fn = self.vtable.decode_greedy orelse return error.UnsupportedGreedyDecode;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        var output = try decode_greedy_fn(self.ptr, allocator, request);
        errdefer output.deinit();
        try request.check();
        return output;
    }

    pub fn decodeSample(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: SampledDecodeRequest,
    ) !SampledDecodeOutput {
        try request.decode.check();
        const decode_sample_fn = self.vtable.decode_sample orelse return error.UnsupportedSampleDecode;
        var hard_cancellation = if (request.decode.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        const output = try decode_sample_fn(self.ptr, allocator, request);
        try request.decode.check();
        return output;
    }

    pub fn verifyGreedyTransaction(
        self: *ModelRuntime,
        allocator: std.mem.Allocator,
        request: GreedyVerifyRequest,
    ) !GreedyVerifyOutput {
        try request.check();
        const verify_fn = self.vtable.verify_greedy_transaction orelse
            return error.UnsupportedGreedyVerifyTransaction;
        var hard_cancellation = if (request.execution_control) |control|
            try control.enterUninterruptible(self.capabilities().interruption)
        else
            UninterruptibleGuard{};
        defer hard_cancellation.deinit();
        const output = try verify_fn(self.ptr, allocator, request);
        try request.check();
        return output;
    }

    pub fn supportsSampleDecodeRequest(
        self: *const ModelRuntime,
        request: SampledDecodeRequest,
    ) bool {
        if (self.vtable.decode_sample == null) return false;
        const supports_fn = self.vtable.supports_sample_decode_request orelse return false;
        return supports_fn(self.ptr, request);
    }

    pub fn deinit(self: *ModelRuntime) void {
        self.vtable.deinit(self.ptr);
        self.* = undefined;
    }
};

test "ModelRuntime owner snapshot drain is typed and fails closed when unsupported or unbound" {
    const Probe = struct {
        bound: bool = false,
        drains: usize = 0,

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: PrefillRequest) !ModelOutput {
            return error.TestUnexpectedPrefill;
        }
        fn deinit(_: *anyopaque) void {}
        fn drain(raw: *anyopaque, _: ContextOwnerBinding) !void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            if (!self.bound) return error.ContextPhysicalAccountingUnavailable;
            self.drains += 1;
        }
    };
    var probe: Probe = .{};
    var unsupported: ModelRuntime = .{ .ptr = &probe, .vtable = &.{ .prefill = Probe.prefill, .deinit = Probe.deinit } };
    var owner = try owner_alloc.RequestAllocationOwner.init(null);
    defer _ = owner.release();
    const witness = try owner.beginContextScope(1);
    const binding: ContextOwnerBinding = .{ .owner = &owner, .witness = witness };
    try std.testing.expectError(error.ContextOwnerDrainUnsupported, unsupported.drainContextOwnerSnapshot(binding));
    var runtime: ModelRuntime = .{ .ptr = &probe, .vtable = &.{ .prefill = Probe.prefill, .deinit = Probe.deinit, .drain_context_owner_snapshot = Probe.drain } };
    try std.testing.expectError(error.ContextPhysicalAccountingUnavailable, runtime.drainContextOwnerSnapshot(binding));
    probe.bound = true;
    try runtime.drainContextOwnerSnapshot(binding);
    try std.testing.expectEqual(@as(usize, 1), probe.drains);
    try owner.invalidateContextScope(witness);
    try std.testing.expectError(error.InvalidContextOwnerScope, runtime.drainContextOwnerSnapshot(binding));
    try std.testing.expectEqual(@as(usize, 1), probe.drains);
}

pub const ExactPagedKvDonor = struct {
    layer: u32,
    heads: u32,
    dim: u32,
    capacity_rows: u32 = 0,
    host_staged_ring: bool = false,
};

pub const ExactPagedKvSourceAuthority = struct {
    model_weights_sha256: [32]u8,
    model_identity_layout_sha256: [32]u8,
    context_layout_sha256: [32]u8,
    token_stream_sha256: [32]u8,
    source_revision: u64,
    source_sequence: u32,
    logical_tokens: usize,

    pub fn validate(self: ExactPagedKvSourceAuthority) !void {
        if (std.mem.allEqual(u8, &self.model_weights_sha256, 0) or
            std.mem.allEqual(u8, &self.model_identity_layout_sha256, 0) or
            std.mem.allEqual(u8, &self.context_layout_sha256, 0) or
            std.mem.allEqual(u8, &self.token_stream_sha256, 0) or
            self.source_revision == 0 or self.source_sequence == 0 or self.logical_tokens == 0)
            return error.InvalidExactPagedKvHandoffAuthority;
    }
};

pub const ExactPagedKvRangeExportRequest = struct {
    authority: kv_storage.ExactLayerRangeSourceAuthority,
    ranges: []const kv_storage.ExactLayerRangeExport,
    max_copy_bytes: usize,
    check_ctx: ?*anyopaque = null,
    check_fn: ?*const fn (?*anyopaque) anyerror!void = null,
    pub fn check(self: @This()) !void {
        if (self.check_fn) |f| try f(self.check_ctx);
    }
    pub fn validate(self: @This()) !void {
        try self.check();
        try self.authority.validate();
        if (self.ranges.len == 0 or self.ranges.len > 256 or self.max_copy_bytes == 0)
            return error.InvalidExactPagedKvRangeExport;
        _ = try (kv_storage.ExactLayerRangeExportRequest{
            .source_sequence = self.authority.source_sequence,
            .expected_written_tokens = self.authority.logical_tokens,
            .ranges = self.ranges,
            .max_bytes = self.max_copy_bytes,
        }).validate();
    }
};

/// Authority for the one-way compiled-prefill handoff. The two digests are
/// supplied by the caller's sealed model/prompt binding; the backend verifies
/// its layout digest and its independently accumulated token-stream digest.
pub const ExactPagedKvHandoffRequest = struct {
    authority_sha256: [32]u8,
    model_layout_sha256: [32]u8,
    token_stream_sha256: [32]u8,
    source_revision: u64,
    destination_owner_id: u64,
    logical_tokens: usize,
    capacity_tokens: usize,
    page_size_tokens: u16,
    max_copy_bytes: usize,
    max_host_staging_bytes: usize,
    full_history_exact: bool = false,
    donors: []const ExactPagedKvDonor,
    destination: ops.KvCacheView,
    execution_control: ?InferenceExecutionControl = null,

    pub fn check(self: ExactPagedKvHandoffRequest) !void {
        if (self.execution_control) |control| try control.check();
    }

    pub fn validate(self: ExactPagedKvHandoffRequest) !void {
        try self.check();
        if (std.mem.allEqual(u8, &self.authority_sha256, 0) or
            std.mem.allEqual(u8, &self.model_layout_sha256, 0) or
            std.mem.allEqual(u8, &self.token_stream_sha256, 0) or
            self.source_revision == 0 or self.destination_owner_id == 0 or
            self.logical_tokens == 0 or self.capacity_tokens < self.logical_tokens or
            self.page_size_tokens == 0 or self.capacity_tokens % self.page_size_tokens != 0 or
            self.max_copy_bytes == 0 or self.max_host_staging_bytes == 0 or self.donors.len == 0 or self.donors.len > 256 or
            self.destination.kv_storage == null or self.destination.logical_blocks == null or
            self.destination.position_offset != 0)
            return error.InvalidExactPagedKvHandoff;
        for (self.donors, 0..) |donor, i| {
            if (self.full_history_exact and
                (donor.host_staged_ring or donor.capacity_rows != self.capacity_tokens))
                return error.InvalidExactPagedKvHandoff;
            if (donor.heads == 0 or donor.dim == 0 or (donor.host_staged_ring and donor.capacity_rows == 0) or
                (donor.capacity_rows != 0 and (donor.capacity_rows > self.capacity_tokens or donor.capacity_rows % self.page_size_tokens != 0)) or
                (i > 0 and donor.layer <= self.donors[i - 1].layer))
                return error.InvalidExactPagedKvHandoff;
        }
    }

    pub fn validateReceipt(self: ExactPagedKvHandoffRequest, receipt: ExactPagedKvHandoffReceipt) !void {
        if (!std.mem.eql(u8, &receipt.authority_sha256, &self.authority_sha256) or
            !std.mem.eql(u8, &receipt.model_layout_sha256, &self.model_layout_sha256) or
            !std.mem.eql(u8, &receipt.token_stream_sha256, &self.token_stream_sha256) or
            receipt.source_revision != self.source_revision or
            receipt.destination_owner_id != self.destination_owner_id or
            receipt.destination_storage_address != @intFromPtr(self.destination.kv_storage.?) or
            receipt.logical_tokens != self.logical_tokens or
            receipt.donors != self.donors.len or receipt.copied_bytes == 0 or
            receipt.source_unchanged != true or receipt.destination_complete != true or
            receipt.fenced != true)
            return error.InvalidExactPagedKvHandoffReceipt;
    }
};

pub const ExactPagedKvHandoffReceipt = struct {
    authority_sha256: [32]u8,
    model_layout_sha256: [32]u8,
    token_stream_sha256: [32]u8,
    source_revision: u64,
    destination_owner_id: u64,
    destination_storage_address: usize,
    logical_tokens: usize,
    donors: usize,
    allocation_bound_bytes: usize,
    copied_bytes: usize,
    peak_host_staging_bytes: usize,
    source_unchanged: bool,
    destination_complete: bool,
    fenced: bool,
};

pub const ExactPagedKvSourceDiscardRequest = struct {
    authority_sha256: [32]u8,
    source_revision: u64,
    source_sequence: u32,
    token_stream_sha256: [32]u8,
    logical_tokens: usize,
    expected_materialized_tokens: usize,
    execution_control: ?InferenceExecutionControl = null,

    pub fn check(self: @This()) !void {
        if (std.mem.allEqual(u8, &self.authority_sha256, 0) or self.source_revision == 0 or
            self.source_sequence == 0 or std.mem.allEqual(u8, &self.token_stream_sha256, 0) or self.logical_tokens == 0 or
            self.expected_materialized_tokens < self.logical_tokens)
            return error.InvalidExactPagedKvSourceDiscard;
        if (self.execution_control) |control| try control.check();
    }
};

pub const ExactPagedKvSourceDiscardReceipt = struct {
    authority_sha256: [32]u8,
    source_revision: u64,
    source_sequence: u32,
    slots: usize,
    live_bytes_before: u64,
    live_bytes_after: u64,
    physically_released_bytes: u64,
};

pub fn exactPagedKvTokenStreamDigest(tokens: []const i64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.compiled-prefill-token-stream.v1\x00");
    var count_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &count_bytes, @intCast(tokens.len), .little);
    hash.update(&count_bytes);
    for (tokens) |token| {
        var token_bytes: [8]u8 = undefined;
        std.mem.writeInt(i64, &token_bytes, token, .little);
        hash.update(&token_bytes);
    }
    var result: [32]u8 = undefined;
    hash.final(&result);
    return result;
}

test "exact paged KV handoff binds authority tokens owner and completion" {
    var fake_storage: @import("../runtime/kv/storage_runtime.zig").KvStorageRuntime = undefined;
    const blocks = [_]u32{0};
    const tokens = [_]i64{ 7, 11, 13 };
    const digest = exactPagedKvTokenStreamDigest(&tokens);
    var request = ExactPagedKvHandoffRequest{
        .authority_sha256 = @splat(1),
        .model_layout_sha256 = @splat(2),
        .token_stream_sha256 = digest,
        .source_revision = 4,
        .destination_owner_id = 9,
        .logical_tokens = 3,
        .capacity_tokens = 16,
        .page_size_tokens = 16,
        .max_copy_bytes = 4096,
        .max_host_staging_bytes = 4096,
        .donors = &.{.{ .layer = 5, .heads = 2, .dim = 8 }},
        .destination = .{ .sequence_id = 1, .pool_id = 0, .logical_block_count = 1, .tail_tokens = 3, .logical_blocks = &blocks, .kv_storage = &fake_storage },
    };
    try request.validate();
    request.full_history_exact = true;
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 16 }};
    try request.validate();
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 16, .host_staged_ring = true }};
    try std.testing.expectError(error.InvalidExactPagedKvHandoff, request.validate());
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 0 }};
    try std.testing.expectError(error.InvalidExactPagedKvHandoff, request.validate());
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 16 }};
    request.full_history_exact = false;
    const receipt = ExactPagedKvHandoffReceipt{
        .authority_sha256 = request.authority_sha256,
        .model_layout_sha256 = request.model_layout_sha256,
        .token_stream_sha256 = request.token_stream_sha256,
        .source_revision = request.source_revision,
        .destination_owner_id = request.destination_owner_id,
        .destination_storage_address = @intFromPtr(request.destination.kv_storage.?),
        .logical_tokens = request.logical_tokens,
        .donors = 1,
        .allocation_bound_bytes = 1024,
        .copied_bytes = 512,
        .peak_host_staging_bytes = 0,
        .source_unchanged = true,
        .destination_complete = true,
        .fenced = true,
    };
    const Harness = struct {
        called: bool = false,

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: PrefillRequest) !ModelOutput {
            return error.UnexpectedPrefill;
        }

        fn deinit(_: *anyopaque) void {}

        fn handoff(raw: *anyopaque, candidate: ExactPagedKvHandoffRequest) !ExactPagedKvHandoffReceipt {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.called = true;
            return .{
                .authority_sha256 = candidate.authority_sha256,
                .model_layout_sha256 = candidate.model_layout_sha256,
                .token_stream_sha256 = candidate.token_stream_sha256,
                .source_revision = candidate.source_revision,
                .destination_owner_id = candidate.destination_owner_id,
                .destination_storage_address = @intFromPtr(candidate.destination.kv_storage.?),
                .logical_tokens = candidate.logical_tokens,
                .donors = candidate.donors.len,
                .allocation_bound_bytes = 1024,
                .copied_bytes = 512,
                .peak_host_staging_bytes = 0,
                .source_unchanged = true,
                .destination_complete = true,
                .fenced = true,
            };
        }
    };
    var harness: Harness = .{};
    var runtime: ModelRuntime = .{ .ptr = &harness, .vtable = &.{
        .prefill = Harness.prefill,
        .deinit = Harness.deinit,
        .handoff_exact_paged_kv = Harness.handoff,
    } };
    request.full_history_exact = true;
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 15 }};
    try std.testing.expectError(error.InvalidExactPagedKvHandoff, runtime.handoffExactPagedKv(request));
    try std.testing.expect(!harness.called);
    request.donors = &.{.{ .layer = 5, .heads = 2, .dim = 8, .capacity_rows = 16 }};
    const actual = try runtime.handoffExactPagedKv(request);
    try std.testing.expect(harness.called);
    try std.testing.expectEqual(receipt, actual);
    try request.validateReceipt(receipt);
    request.destination_owner_id += 1;
    try std.testing.expectError(error.InvalidExactPagedKvHandoffReceipt, request.validateReceipt(receipt));
    request.destination_owner_id -= 1;
    var partial = receipt;
    partial.destination_complete = false;
    try std.testing.expectError(error.InvalidExactPagedKvHandoffReceipt, request.validateReceipt(partial));
}

test "exact paged KV range export callback rejects authority mutation before backend" {
    const digest: [32]u8 = @splat(3);
    var k: u8 = 0;
    var v: u8 = 0;
    const ranges = [_]kv_storage.ExactLayerRangeExport{.{ .layer_index = 4, .source_start = 0, .rows = 1, .num_kv_heads = 1, .head_dim = 1, .destination_k = .{ .handle = &k, .byte_offset = 0, .byte_len = 2 }, .destination_v = .{ .handle = &v, .byte_offset = 0, .byte_len = 2 } }};
    const Harness = struct {
        calls: usize = 0,
        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: PrefillRequest) !ModelOutput {
            return error.UnexpectedPrefill;
        }
        fn deinit(_: *anyopaque) void {}
        fn exportCallback(raw: *anyopaque, request: ExactPagedKvRangeExportRequest) !kv_storage.ExactLayerRangeExportResult {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            try request.validate();
            return .{ .copied_bytes = 4, .layers = request.ranges.len, .fenced = true };
        }
    };
    var harness: Harness = .{};
    var runtime: ModelRuntime = .{ .ptr = &harness, .vtable = &.{ .prefill = Harness.prefill, .deinit = Harness.deinit, .export_exact_paged_kv_ranges = Harness.exportCallback } };
    var authority = kv_storage.ExactLayerRangeSourceAuthority{ .model_weights_sha256 = digest, .model_identity_layout_sha256 = digest, .context_layout_sha256 = digest, .token_stream_sha256 = digest, .source_revision = 2, .source_sequence = 7, .logical_tokens = 1 };
    const exporter = runtime.exactPagedKvRangeExporter();
    const result = try exporter.exportRanges(authority, &ranges, 4, null, null);
    try std.testing.expect(result.fenced);
    try std.testing.expectEqual(@as(usize, 1), harness.calls);
    authority.token_stream_sha256 = @splat(0);
    try std.testing.expectError(error.InvalidExactLayerRangeExportAuthority, exporter.exportRanges(authority, &ranges, 4, null, null));
    try std.testing.expectEqual(@as(usize, 1), harness.calls);
}

test "exact paged KV source discard validates physical release receipt" {
    const Harness = struct {
        called: bool = false,

        fn prefill(_: *anyopaque, _: std.mem.Allocator, _: PrefillRequest) !ModelOutput {
            return error.UnexpectedPrefill;
        }

        fn deinit(_: *anyopaque) void {}

        fn discard(raw: *anyopaque, request: ExactPagedKvSourceDiscardRequest) !ExactPagedKvSourceDiscardReceipt {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.called = true;
            return .{
                .authority_sha256 = request.authority_sha256,
                .source_revision = request.source_revision,
                .source_sequence = 7,
                .slots = 34,
                .live_bytes_before = 8192,
                .live_bytes_after = 0,
                .physically_released_bytes = 8192,
            };
        }
    };
    var harness: Harness = .{};
    var runtime: ModelRuntime = .{ .ptr = &harness, .vtable = &.{
        .prefill = Harness.prefill,
        .deinit = Harness.deinit,
        .discard_exact_paged_kv_source = Harness.discard,
    } };
    var request: ExactPagedKvSourceDiscardRequest = .{
        .authority_sha256 = @splat(9),
        .source_revision = 3,
        .source_sequence = 7,
        .token_stream_sha256 = @splat(8),
        .logical_tokens = 16,
        .expected_materialized_tokens = 18,
    };
    const receipt = try runtime.discardExactPagedKvSource(request);
    try std.testing.expect(harness.called);
    try std.testing.expectEqual(@as(u64, 8192), receipt.physically_released_bytes);
    try std.testing.expectEqual(@as(u64, 0), receipt.live_bytes_after);
    request.source_sequence = 8;
    try std.testing.expectError(error.InvalidExactPagedKvSourceDiscardReceipt, runtime.discardExactPagedKvSource(request));
}

pub const RuntimeStateOwnership = enum {
    /// Runtime execution still materializes graph inputs from host state.
    /// This can use ModelRuntime orchestration, but the backend does not own
    /// persistent model/KV state yet.
    host_assisted_inputs,
    /// Runtime owns persistent KV/cache tensors, but they remain host-side
    /// tensors fed back into each backend call.
    runtime_owned_host_cache,
    /// Runtime execution owns persistent backend-side model/KV/cache state.
    backend_owned,
};

pub const RuntimeCapabilities = struct {
    supports_decode: bool = false,
    supports_sample_decode: bool = false,
    supports_greedy_decode: bool = false,
    supports_greedy_verify_transaction: bool = false,
    state_ownership: RuntimeStateOwnership = .host_assisted_inputs,
    interruption: Interruption = .cooperative,
};

pub const RuntimeDebugTimingStats = struct {
    backend: ops.BackendDebugTimingSnapshot = .{},
    decoder_runtime_ready: bool = false,
    decoder_runtime_absolute_embeddings_prepared: bool = false,
    runtime_prepare_calls: u64 = 0,
    runtime_prepare_nanos: u128 = 0,
    runtime_prepare_family_nanos: u128 = 0,
    runtime_prepare_greedy_nanos: u128 = 0,
    runtime_prepare_fast_hits: u64 = 0,
    prefill_calls: u64 = 0,
    prefill_prepare_nanos: u128 = 0,
    prefill_direct_last_logits_nanos: u128 = 0,
    prefill_direct_family_nanos: u128 = 0,
    prefill_direct_family_project_nanos: u128 = 0,
    prefill_direct_family_span_prep_nanos: u128 = 0,
    prefill_direct_family_quant_attn_nanos: u128 = 0,
    prefill_direct_family_block_apply_nanos: u128 = 0,
    prefill_direct_family_frame_wait_nanos: u128 = 0,
    prefill_direct_family_frame_gpu_nanos: u128 = 0,
    prefill_fallback_logits_nanos: u128 = 0,
    decode_begin_step_nanos: u128 = 0,
    decode_sample_calls: u64 = 0,
    decode_sample_direct_nanos: u128 = 0,
    decode_sample_fallback_nanos: u128 = 0,
    decode_greedy_calls: u64 = 0,
    decode_greedy_direct_nanos: u128 = 0,
    decode_greedy_fallback_nanos: u128 = 0,
    decode_greedy_device_token_handoff_attempts: u64 = 0,
    decode_greedy_device_token_handoff_hits: u64 = 0,
    decode_greedy_device_token_handoff_fallbacks: u64 = 0,
    decode_greedy_device_token_seeds: u64 = 0,
    ensure_prepared_calls: u64 = 0,
    ensure_prepared_nanos: u128 = 0,
    ensure_prepared_sync_nanos: u128 = 0,
    ensure_prepared_family_nanos: u128 = 0,
    ensure_prepared_greedy_nanos: u128 = 0,
    ensure_prepared_fast_hits: u64 = 0,
};

pub const SamplingConfig = struct {
    temperature: f32 = 0,
    top_p: f32 = 0,
    top_k: i32 = 0,
    min_p: f32 = 0,
    repetition_penalty: f32 = 1.0,
    frequency_penalty: f32 = 0,
    presence_penalty: f32 = 0,

    pub fn isPureGreedy(self: SamplingConfig) bool {
        return self.temperature <= 0 and
            self.repetition_penalty == 1.0 and
            self.frequency_penalty == 0 and
            self.presence_penalty == 0;
    }
};

pub const ModelOutput = struct {
    pub const PreparedTailNorm = enum {
        layer,
        rms,
    };

    pub const PreparedTail = struct {
        final_hidden: contracts.CT,
        backend: ops.ComputeBackend,
        norm: PreparedTailNorm,
        norm_slot: usize,
        lm_head_slot: usize,
        hidden_size: usize,
        vocab_size: usize,
        eps: f32,
        final_logit_softcap: f32 = 0.0,
        /// Diagnostic-only request used by the fail-closed LM-head repack
        /// quality campaign. Production outputs leave this false and retain
        /// checkpoint-format full logits.
        use_transformed_lm_head: bool = false,
        greedy_token_id: ?i64 = null,

        fn logits(self: *PreparedTail) !?contracts.CT {
            return switch (self.norm) {
                .layer => self.backend.decoderRuntimeApplyLayerNormLinear(&.{
                    .input = self.final_hidden,
                    .norm_slot = self.norm_slot,
                    .linear_slot = self.lm_head_slot,
                    .hidden_size = self.hidden_size,
                    .eps = self.eps,
                    .out_dim = self.vocab_size,
                }),
                .rms => self.backend.decoderRuntimeApplyRmsNormLinear(&.{
                    .input = self.final_hidden,
                    .norm_slot = self.norm_slot,
                    .linear_slot = self.lm_head_slot,
                    .hidden_size = self.hidden_size,
                    .eps = self.eps,
                    .out_dim = self.vocab_size,
                    .use_transformed_lm_head = self.use_transformed_lm_head,
                }),
            };
        }

        fn greedyTokenSuppress(self: *PreparedTail, suppress_token_ids: []const i32) !?i64 {
            // The cached token was selected under the policy active when the
            // tail was produced. With no explicit request override that is
            // the unchanged legacy policy. A nonempty request list has no
            // retained origin key, so it must recompute even when the cached
            // id itself is not newly suppressed.
            if (suppress_token_ids.len == 0) {
                if (self.greedy_token_id) |token_id| return token_id;
            }
            // Final logit softcap is monotonic, so it changes sampling
            // probabilities but not the greedy argmax ordering.
            const token = switch (self.norm) {
                .layer => if (suppress_token_ids.len == 0) try self.backend.decoderRuntimeApplyLayerNormLinearArgmax(&.{
                    .input = self.final_hidden,
                    .norm_slot = self.norm_slot,
                    .linear_slot = self.lm_head_slot,
                    .hidden_size = self.hidden_size,
                    .eps = self.eps,
                    .out_dim = self.vocab_size,
                }) else null,
                .rms => try self.backend.decoderRuntimeApplyRmsNormLinearArgmax(&.{
                    .input = self.final_hidden,
                    .norm_slot = self.norm_slot,
                    .linear_slot = self.lm_head_slot,
                    .hidden_size = self.hidden_size,
                    .eps = self.eps,
                    .out_dim = self.vocab_size,
                    .suppress_token_ids = suppress_token_ids,
                }),
            };
            if (token) |token_id| {
                const selected: i64 = @intCast(token_id);
                if (!tokenIdSuppressed(selected, suppress_token_ids)) return selected;
            }
            return null;
        }

        pub fn deinit(self: *PreparedTail) void {
            self.backend.free(self.final_hidden);
            self.* = undefined;
        }
    };

    /// Last-step logits owned by the caller.
    logits: ?[]f32 = null,
    device_logits: ?contracts.CT = null,
    device_logits_backend: ?ops.ComputeBackend = null,
    prepared_tail: ?PreparedTail = null,
    final_logit_softcap: f32 = 0.0,

    pub fn hostLogits(self: *ModelOutput, allocator: std.mem.Allocator) ![]const f32 {
        if (self.logits) |logits| return logits;
        if (self.prepared_tail) |*tail| {
            const device_logits = (try tail.logits()) orelse return error.InvalidModelOutput;
            const backend = tail.backend;
            const logits = try backend.toFloat32(device_logits, allocator);
            backend.free(device_logits);
            applyFinalLogitSoftcapInPlace(logits, tail.final_logit_softcap);
            tail.deinit();
            self.prepared_tail = null;
            self.logits = logits;
            return logits;
        }
        if (self.device_logits) |device_logits| {
            const backend = self.device_logits_backend orelse return error.InvalidModelOutput;
            const logits = try backend.toFloat32(device_logits, allocator);
            backend.free(device_logits);
            self.device_logits = null;
            self.device_logits_backend = null;
            applyFinalLogitSoftcapInPlace(logits, self.final_logit_softcap);
            self.logits = logits;
            return logits;
        }
        return error.InvalidModelOutput;
    }

    pub fn takeHostLogits(self: *ModelOutput, allocator: std.mem.Allocator) ![]f32 {
        _ = try self.hostLogits(allocator);
        const logits = self.logits orelse return error.InvalidModelOutput;
        self.logits = null;
        return logits;
    }

    pub fn greedyToken(self: *ModelOutput, allocator: std.mem.Allocator, vocab_size: usize) !i64 {
        return self.greedyTokenSuppress(allocator, vocab_size, &.{});
    }

    pub fn greedyTokenSuppress(self: *ModelOutput, allocator: std.mem.Allocator, vocab_size: usize, suppress_token_ids: []const i32) !i64 {
        if (vocab_size == 0) return error.InvalidModelOutput;
        if (self.prepared_tail) |*tail| {
            if (tail.vocab_size == vocab_size) {
                if (try tail.greedyTokenSuppress(suppress_token_ids)) |token_id| return token_id;
            }
        }
        if (self.device_logits) |device_logits| {
            const backend = self.device_logits_backend orelse return error.InvalidModelOutput;
            if (suppress_token_ids.len == 0) {
                if (try backend.argmaxLastRow(device_logits, 1, vocab_size)) |token_id| return @intCast(token_id);
            } else if (try backend.argmaxRowsSuppress(device_logits, 0, 1, vocab_size, suppress_token_ids, allocator)) |token_ids| {
                defer allocator.free(token_ids);
                if (token_ids.len != 1) return error.InvalidModelOutput;
                const token_id: i64 = @intCast(token_ids[0]);
                if (!tokenIdSuppressed(token_id, suppress_token_ids)) return token_id;
            }
        }

        const logits = try self.hostLogits(allocator);
        if (logits.len < vocab_size) return error.InvalidModelOutput;
        var best_idx: ?usize = null;
        for (logits[0..vocab_size], 0..) |value, idx| {
            if (tokenIdSuppressed(@intCast(idx), suppress_token_ids)) continue;
            if (best_idx == null or value > logits[best_idx.?]) best_idx = idx;
        }
        return @intCast(best_idx orelse return error.InvalidModelOutput);
    }

    pub fn deinit(self: *ModelOutput, allocator: std.mem.Allocator) void {
        if (self.logits) |logits| allocator.free(logits);
        if (self.device_logits) |device_logits| {
            if (self.device_logits_backend) |backend| backend.free(device_logits);
        }
        if (self.prepared_tail) |*tail| tail.deinit();
        self.* = undefined;
    }
};

fn tokenIdSuppressed(token_id: i64, suppress_token_ids: []const i32) bool {
    if (token_id < 0 or token_id > std.math.maxInt(i32)) return false;
    const needle: i32 = @intCast(token_id);
    for (suppress_token_ids) |candidate| if (candidate == needle) return true;
    return false;
}

fn applyFinalLogitSoftcapInPlace(logits: []f32, softcap: f32) void {
    if (softcap <= 0.0) return;
    for (logits) |*value| {
        value.* = std.math.tanh(value.* / softcap) * softcap;
    }
}

pub const PrepareRequest = struct {
    execution_control: ?InferenceExecutionControl = null,
    e2b_f16_mirror_admission: ?ops.DecoderRuntimeE2bF16MirrorAdmission = null,
    /// Best-effort prompt/KV length hint. Backends may use this to prebuild
    /// shape-specific resources without mutating logical generation state.
    kv_tokens_hint: usize = 0,
    /// Request-local transactional verification cannot share the greedy
    /// decoder's speculative one-token lookahead: that lookahead advances KV
    /// bookkeeping beyond the logical pending-token boundary.
    disable_pipelined_decode: bool = false,

    pub fn check(self: PrepareRequest) !void {
        if (self.execution_control) |control| try control.check();
    }
};

pub const PrefillRequest = struct {
    execution_control: ?InferenceExecutionControl = null,
    input_ids: []const i64,
    seq_len: usize,
    query_seq_len: usize,
    attention_mode: cache_mod.AttentionMode,
    prepare_after_fresh_reset: ?PrepareRequest = null,
    /// Largest KV suffix this request may write at once. Device runtimes use
    /// this with `allow_swa_ring` to size sliding-window rings before the first
    /// prefill write.
    max_inflight_tokens: usize = 0,
    allow_swa_ring: bool = false,
    force_host_logits: bool = false,
    prefer_greedy_token: bool = false,
    /// Borrowed for this synchronous call only; runtimes must not retain it.
    suppress_token_ids: []const i32 = &.{},
    /// Borrowed for this synchronous compiled-prefill call only. This is a
    /// development capture seam, not a selection/repack policy switch.
    recent_query_capture: ?RecentPrefillQueryCaptureDescriptor = null,

    pub fn check(self: PrefillRequest) !void {
        if (self.execution_control) |control| try control.check();
        if (self.recent_query_capture) |capture| try capture.validate();
    }
};

test "recent prefill query capture is default-off and validates causal geometry" {
    const ordinary = PrefillRequest{
        .input_ids = &.{1},
        .seq_len = 1,
        .query_seq_len = 1,
        .attention_mode = .paged_prefill,
    };
    try ordinary.check();
    try std.testing.expect(ordinary.recent_query_capture == null);

    const Callbacks = struct {
        fn accepts(_: *anyopaque, _: u32, _: u32) bool {
            return true;
        }
        fn lastEncodedFrame(_: *anyopaque) u64 {
            return 0;
        }
        fn encode(_: *anyopaque, _: RecentPrefillQueryCaptureEncodeRequest) anyerror!void {}
        fn complete(_: *anyopaque, _: RecentPrefillQueryCaptureCompletion) anyerror!void {}
    };
    const context: *anyopaque = @ptrFromInt(16);
    const valid = RecentPrefillQueryCaptureDescriptor{
        .context = context,
        .authority_epoch = 9,
        .accepts_fn = Callbacks.accepts,
        .last_encoded_frame_fn = Callbacks.lastEncodedFrame,
        .encode_fn = Callbacks.encode,
        .complete_fn = Callbacks.complete,
    };
    try valid.validate();

    var invalid = valid;
    invalid.fitted_middle_end_exclusive = 8128 - 64 + 1;
    try std.testing.expectError(error.InvalidRecentPrefillQueryCaptureDescriptor, invalid.validate());
}

pub const DecodeRequest = struct {
    execution_control: ?InferenceExecutionControl = null,
    token_id: i64,
    input_token_tensor: ?ops.CT = null,
    position: usize,
    attention_mode: cache_mod.AttentionMode = .paged_decode,
    /// Borrowed for this synchronous call only; runtimes must not retain it.
    suppress_token_ids: []const i32 = &.{},
    /// Whether a pipelined runtime may encode the frame following the token
    /// returned by this request. Callers with a known terminal output budget
    /// set this false so the current pending frame is consumed without
    /// speculatively advancing KV or submitting an unused continuation.
    allow_pipeline_rearm: bool = true,

    pub fn check(self: DecodeRequest) !void {
        if (self.execution_control) |control| try control.check();
    }
};

test "decode request preserves pipeline rearm unless caller marks a terminal token" {
    const ordinary = DecodeRequest{ .token_id = 7, .position = 11 };
    try std.testing.expect(ordinary.allow_pipeline_rearm);
    const terminal = DecodeRequest{
        .token_id = 7,
        .position = 11,
        .allow_pipeline_rearm = false,
    };
    try std.testing.expect(!terminal.allow_pipeline_rearm);
}

/// One model-authoritative prompt-lookup verification transaction. The backend
/// materializes the pending token and proposal in a single q_len=2 forward,
/// commits matched proposal rows, and rolls back every rejected reserved row
/// before returning. Correction/bonus tokens are logical outputs and remain
/// pending for the next transaction or ordinary decode.
pub const GreedyVerifyRequest = struct {
    execution_control: ?InferenceExecutionControl = null,
    pending_token_id: i64,
    proposal_token_id: i64,
    position: usize,
    attention_mode: cache_mod.AttentionMode = .paged_prefill,
    suppress_token_ids: []const i32 = &.{},
    eos_token_ids: []const i32 = &.{},
    ignore_eos: bool = false,
    accept_bonus: bool = true,

    pub fn check(self: GreedyVerifyRequest) !void {
        if (self.execution_control) |control| try control.check();
        if (self.pending_token_id < 0 or self.proposal_token_id < 0)
            return error.InvalidModelOutput;
        if (self.attention_mode != .paged_prefill) return error.UnsupportedAttentionMode;
    }
};

pub const GreedyVerifyOutput = struct {
    accepted_token_ids: [2]i64 = .{ 0, 0 },
    accepted_count: u8 = 0,
    matched_proposals: u8 = 0,
    correction_added: bool = false,
    bonus_added: bool = false,
    hit_eos: bool = false,
    materialized_rows: u8 = 1,
    rolled_back_rows: u8 = 0,

    pub fn accepted(self: *const GreedyVerifyOutput) []const i64 {
        return self.accepted_token_ids[0..self.accepted_count];
    }
};

pub const SampledDecodeRequest = struct {
    decode: DecodeRequest,
    sampling: SamplingConfig,
    token_history: []const i64,
};

pub const GreedyDecodeOutput = struct {
    token_id: i64,
    /// Optional backend-owned token tensor for the returned token. Callers that
    /// pass it back as DecodeRequest.input_token_tensor own it and must either
    /// transfer it again or call deinit.
    token_tensor: ?ops.CT = null,
    token_tensor_backend: ?*const ops.ComputeBackend = null,

    pub fn deinit(self: *GreedyDecodeOutput) void {
        if (self.token_tensor) |tensor| {
            if (self.token_tensor_backend) |backend| backend.free(tensor);
        }
        self.* = undefined;
    }
};

pub const SampledDecodeOutput = struct {
    token_id: i64,
};

pub const ForwardRequest = union(enum) {
    prefill: PrefillRequest,
    decode: DecodeRequest,
};

const SamplingPenaltyState = struct {
    counts: std.AutoHashMapUnmanaged(u32, u32) = .empty,

    pub fn deinit(self: *SamplingPenaltyState, allocator: std.mem.Allocator) void {
        self.counts.deinit(allocator);
        self.* = .{};
    }

    fn seedFromHistory(self: *SamplingPenaltyState, allocator: std.mem.Allocator, token_history: []const i64) !void {
        for (token_history) |token_id| try self.noteToken(allocator, token_id);
    }

    fn noteToken(self: *SamplingPenaltyState, allocator: std.mem.Allocator, token_id: i64) !void {
        if (token_id < 0) return;
        const entry = try self.counts.getOrPut(allocator, @intCast(token_id));
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }

    fn isEmpty(self: *const SamplingPenaltyState) bool {
        return self.counts.count() == 0;
    }
};

pub fn sampleTokenFromLogits(
    allocator: std.mem.Allocator,
    logits: []const f32,
    config: SamplingConfig,
    token_history: []const i64,
) usize {
    var penalty_state = SamplingPenaltyState{};
    defer penalty_state.deinit(allocator);
    penalty_state.seedFromHistory(allocator, token_history) catch {};
    return sample(@constCast(logits), config, &penalty_state, allocator);
}

fn sample(logits: []const f32, config: SamplingConfig, penalty_state: *const SamplingPenaltyState, allocator: std.mem.Allocator) usize {
    const has_penalties = config.repetition_penalty != 1.0 or
        config.frequency_penalty != 0 or
        config.presence_penalty != 0;
    if (config.temperature <= 0 and !has_penalties) {
        return activations.argmax(logits);
    }

    const working = allocator.alloc(f32, logits.len) catch return activations.argmax(logits);
    defer allocator.free(working);
    @memcpy(working, logits);

    if (has_penalties and !penalty_state.isEmpty()) {
        applyRepetitionPenalties(working, penalty_state, config);
    }
    if (config.temperature <= 0) {
        return activations.argmax(working);
    }

    const inv_temp = 1.0 / config.temperature;
    for (working) |*v| v.* *= inv_temp;
    activations.softmax(working, working.len);

    if (config.top_k > 0 and @as(usize, @intCast(config.top_k)) < working.len) {
        activations.topK(working, @intCast(config.top_k), allocator);
    }
    if (config.top_p > 0 and config.top_p < 1.0) {
        activations.topP(working, config.top_p, allocator);
    }
    if (config.min_p > 0 and config.min_p < 1.0) {
        applyMinP(working, config.min_p);
    }
    return activations.sampleFromProbs(working);
}

fn applyRepetitionPenalties(logits: []f32, penalty_state: *const SamplingPenaltyState, config: SamplingConfig) void {
    var it = penalty_state.counts.iterator();
    while (it.next()) |entry| {
        const token_id = entry.key_ptr.*;
        const count = entry.value_ptr.*;
        if (token_id >= logits.len) continue;
        if (config.repetition_penalty != 1.0) {
            const logit = logits[token_id];
            if (logit > 0) {
                logits[token_id] = logit / config.repetition_penalty;
            } else {
                logits[token_id] = logit * config.repetition_penalty;
            }
        }
        if (config.frequency_penalty != 0) {
            logits[token_id] -= config.frequency_penalty * @as(f32, @floatFromInt(count));
        }
        if (config.presence_penalty != 0) {
            logits[token_id] -= config.presence_penalty;
        }
    }
}

fn applyMinP(probs: []f32, min_p: f32) void {
    if (probs.len == 0) return;
    var max_prob: f32 = 0;
    for (probs) |p| max_prob = @max(max_prob, p);
    const threshold = max_prob * min_p;
    for (probs) |*p| {
        if (p.* < threshold) p.* = 0;
    }
}

/// Type-erased whole-model executor.
///
/// Implementations own immutable compiled model/session resources. They create
/// ModelRuntime values for per-request mutable state. Runtime values then
/// execute prefill/decode so cache/KV ownership has one place to live.
pub const ModelExecutor = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// Immutable loaded-generation identity. Never upgraded on a live cache.
    model_identity: ?checkpoint.Identity = null,

    pub const VTable = struct {
        create_runtime: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator) anyerror!ModelRuntime,
        deinit: *const fn (ctx: *anyopaque) void,
    };

    pub fn createRuntime(self: *const ModelExecutor, allocator: std.mem.Allocator) !ModelRuntime {
        var runtime = try self.vtable.create_runtime(self.ptr, allocator);
        if (runtime.model_identity != null and !sameOptionalModelIdentity(runtime.model_identity, self.model_identity)) {
            runtime.deinit();
            return error.CompiledModelIdentityMismatch;
        }
        runtime.model_identity = self.model_identity;
        return runtime;
    }

    pub fn deinit(self: *ModelExecutor) void {
        self.vtable.deinit(self.ptr);
        self.* = undefined;
    }
};

const MockModel = struct {
    runtime_resets: usize = 0,
    runtime_prepares: usize = 0,
    runtime_finishes: usize = 0,
    last_prepare_hint: usize = 0,
    runtime_deinits: usize = 0,
    executor_deinits: usize = 0,
    saw_prefill_control: bool = false,
    verify_calls: usize = 0,
    fail_verify_rollback: bool = false,

    const runtime_vtable = ModelRuntime.VTable{
        .capabilities = capabilities,
        .prepare = prepare,
        .finish_prefill_for_decode = finishPrefillForDecode,
        .prefill = prefill,
        .decode = decode,
        .decode_sample = decodeSample,
        .decode_greedy = decodeGreedy,
        .verify_greedy_transaction = verifyGreedyTransaction,
        .deinit = runtimeDeinit,
        .reset = runtimeReset,
    };

    fn capabilities(_: *anyopaque) RuntimeCapabilities {
        return .{
            .supports_decode = true,
            .supports_sample_decode = true,
            .supports_greedy_decode = true,
            .supports_greedy_verify_transaction = true,
        };
    }

    const executor_vtable = ModelExecutor.VTable{
        .create_runtime = createRuntime,
        .deinit = executorDeinit,
    };

    fn runtimeDeinit(ctx: *anyopaque) void {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.runtime_deinits += 1;
    }

    fn runtimeReset(ctx: *anyopaque) !void {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.runtime_resets += 1;
    }

    fn prepare(
        ctx: *anyopaque,
        _: std.mem.Allocator,
        request: PrepareRequest,
    ) !bool {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.runtime_prepares += 1;
        self.last_prepare_hint = request.kv_tokens_hint;
        return true;
    }

    fn finishPrefillForDecode(ctx: *anyopaque, request: PrepareRequest) !bool {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.runtime_finishes += 1;
        self.last_prepare_hint = request.kv_tokens_hint;
        return true;
    }

    fn createRuntime(ctx: *anyopaque, _: std.mem.Allocator) !ModelRuntime {
        return .{ .ptr = ctx, .vtable = &runtime_vtable };
    }

    fn prefill(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        request: PrefillRequest,
    ) !ModelOutput {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.saw_prefill_control = request.execution_control != null;
        const logits = try allocator.alloc(f32, 2);
        logits[0] = @floatFromInt(request.input_ids.len);
        logits[1] = @floatFromInt(request.seq_len + request.query_seq_len);
        return .{ .logits = logits };
    }

    fn decode(
        _: *anyopaque,
        allocator: std.mem.Allocator,
        request: DecodeRequest,
    ) !ModelOutput {
        const logits = try allocator.alloc(f32, 1);
        logits[0] = @floatFromInt(request.token_id + @as(i64, @intCast(request.position)));
        return .{ .logits = logits };
    }

    fn decodeGreedy(
        _: *anyopaque,
        _: std.mem.Allocator,
        request: DecodeRequest,
    ) !GreedyDecodeOutput {
        return .{ .token_id = request.token_id + @as(i64, @intCast(request.position)) };
    }

    fn decodeSample(
        _: *anyopaque,
        allocator: std.mem.Allocator,
        request: SampledDecodeRequest,
    ) !SampledDecodeOutput {
        const logits = [_]f32{
            @floatFromInt(request.decode.token_id),
            @floatFromInt(request.decode.position + request.token_history.len),
        };
        return .{ .token_id = @intCast(sampleTokenFromLogits(allocator, logits[0..], request.sampling, request.token_history)) };
    }

    fn verifyGreedyTransaction(
        ctx: *anyopaque,
        _: std.mem.Allocator,
        request: GreedyVerifyRequest,
    ) !GreedyVerifyOutput {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.verify_calls += 1;
        if (self.fail_verify_rollback) return error.GreedyVerifyRollbackFailed;
        return .{
            .accepted_token_ids = .{ request.proposal_token_id, request.proposal_token_id + 1 },
            .accepted_count = 2,
            .matched_proposals = 1,
            .bonus_added = true,
            .materialized_rows = 2,
        };
    }

    fn executorDeinit(ctx: *anyopaque) void {
        const self: *MockModel = @ptrCast(@alignCast(ctx));
        self.executor_deinits += 1;
    }
};

test "ModelExecutor and ModelRuntime dispatch through vtables" {
    const allocator = std.testing.allocator;

    var mock = MockModel{};
    var executor = ModelExecutor{ .ptr = &mock, .vtable = &MockModel.executor_vtable };
    var runtime = try executor.createRuntime(allocator);
    const caps = runtime.capabilities();
    try std.testing.expect(caps.supports_decode);
    try std.testing.expect(caps.supports_sample_decode);
    try std.testing.expect(caps.supports_greedy_decode);
    try std.testing.expect(caps.supports_greedy_verify_transaction);
    try std.testing.expectEqual(RuntimeStateOwnership.host_assisted_inputs, caps.state_ownership);

    try runtime.reset();
    try std.testing.expectEqual(@as(usize, 1), mock.runtime_resets);

    try std.testing.expect(try runtime.prepare(allocator, .{ .kv_tokens_hint = 17 }));
    try std.testing.expectEqual(@as(usize, 1), mock.runtime_prepares);
    try std.testing.expectEqual(@as(usize, 17), mock.last_prepare_hint);

    try std.testing.expect(try runtime.finishPrefillForDecode(.{ .kv_tokens_hint = 19 }));
    try std.testing.expectEqual(@as(usize, 1), mock.runtime_finishes);
    try std.testing.expectEqual(@as(usize, 19), mock.last_prepare_hint);

    var output = try runtime.prefill(allocator, .{
        .input_ids = &.{ 1, 2, 3 },
        .seq_len = 3,
        .query_seq_len = 3,
        .attention_mode = .paged_prefill,
    });
    defer output.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 3, 6 }, try output.hostLogits(allocator));

    var decoded = try runtime.decode(allocator, .{
        .token_id = 4,
        .position = 5,
    });
    defer decoded.deinit(allocator);
    try std.testing.expectEqualSlices(f32, &.{9}, try decoded.hostLogits(allocator));
    try std.testing.expectEqual(@as(i64, 0), try decoded.greedyToken(allocator, 1));

    const greedy = try runtime.decodeGreedy(allocator, .{
        .token_id = 4,
        .position = 5,
    });
    try std.testing.expectEqual(@as(i64, 9), greedy.token_id);

    const sampled = try runtime.decodeSample(allocator, .{
        .decode = .{
            .token_id = 4,
            .position = 5,
        },
        .sampling = .{},
        .token_history = &.{ 1, 2, 3 },
    });
    try std.testing.expectEqual(@as(i64, 1), sampled.token_id);

    const verified = try runtime.verifyGreedyTransaction(allocator, .{
        .pending_token_id = 9,
        .proposal_token_id = 10,
        .position = 3,
    });
    try std.testing.expectEqualSlices(i64, &.{ 10, 11 }, verified.accepted());
    try std.testing.expectEqual(@as(usize, 1), mock.verify_calls);
    mock.fail_verify_rollback = true;
    try std.testing.expectError(error.GreedyVerifyRollbackFailed, runtime.verifyGreedyTransaction(allocator, .{
        .pending_token_id = 11,
        .proposal_token_id = 12,
        .position = 5,
    }));

    runtime.deinit();
    executor.deinit();
    try std.testing.expectEqual(@as(usize, 1), mock.runtime_deinits);
    try std.testing.expectEqual(@as(usize, 1), mock.executor_deinits);
}

test "ModelRuntime propagates control and releases output when the exit check cancels" {
    const CancelAfterCall = struct {
        checks: usize = 0,

        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.checks += 1;
            if (self.checks >= 2) return error.Cancelled;
        }
    };

    var state = CancelAfterCall{};
    var mock = MockModel{};
    var runtime = ModelRuntime{ .ptr = &mock, .vtable = &MockModel.runtime_vtable };
    try std.testing.expectError(error.Cancelled, runtime.prefill(std.testing.allocator, .{
        .execution_control = .{ .ptr = &state, .check_fn = CancelAfterCall.check },
        .input_ids = &.{ 1, 2, 3 },
        .seq_len = 3,
        .query_seq_len = 3,
        .attention_mode = .paged_prefill,
    }));
    try std.testing.expect(mock.saw_prefill_control);
    try std.testing.expectEqual(@as(usize, 2), state.checks);
}

test "ModelExecutor reports unsupported decode when backend has no decode path" {
    const allocator = std.testing.allocator;

    const NoDecode = struct {
        const runtime_vtable = ModelRuntime.VTable{
            .prefill = MockModel.prefill,
            .deinit = MockModel.runtimeDeinit,
        };

        const executor_vtable = ModelExecutor.VTable{
            .create_runtime = createRuntime,
            .deinit = MockModel.executorDeinit,
        };

        fn createRuntime(ctx: *anyopaque, _: std.mem.Allocator) !ModelRuntime {
            return .{ .ptr = ctx, .vtable = &runtime_vtable };
        }
    };

    var mock = MockModel{};
    var executor = ModelExecutor{ .ptr = &mock, .vtable = &NoDecode.executor_vtable };
    defer executor.deinit();
    var runtime = try executor.createRuntime(allocator);
    defer runtime.deinit();
    const caps = runtime.capabilities();
    try std.testing.expect(!caps.supports_decode);
    try std.testing.expect(!caps.supports_sample_decode);
    try std.testing.expect(!caps.supports_greedy_decode);
    try std.testing.expect(!try runtime.finishPrefillForDecode(.{ .kv_tokens_hint = 23 }));

    try std.testing.expectError(error.UnsupportedDecode, runtime.decode(allocator, .{
        .token_id = 1,
        .position = 0,
    }));
    try std.testing.expectError(error.UnsupportedGreedyDecode, runtime.decodeGreedy(allocator, .{
        .token_id = 1,
        .position = 0,
    }));
    try std.testing.expectError(error.UnsupportedSampleDecode, runtime.decodeSample(allocator, .{
        .decode = .{
            .token_id = 1,
            .position = 0,
        },
        .sampling = .{},
        .token_history = &.{},
    }));
}

test "ModelOutput host greedy suppression skips ranked EOS and rejects all tokens" {
    const allocator = std.testing.allocator;
    const values = [_]f32{ 9, 6, 8, 7 };
    const logits = try allocator.dupe(f32, &values);
    var output: ModelOutput = .{ .logits = logits };
    defer output.deinit(allocator);
    try std.testing.expectEqual(@as(i64, 1), try output.greedyTokenSuppress(allocator, values.len, &.{ 0, 2, 3 }));
    try std.testing.expectError(error.InvalidModelOutput, output.greedyTokenSuppress(allocator, values.len, &.{ 0, 1, 2, 3 }));
    try std.testing.expectEqual(@as(i64, 0), try output.greedyToken(allocator, values.len));
}

test "ModelOutput prepared and device greedy forward exact request suppressions" {
    const Probe = struct {
        expected: []const i32,
        selected: usize,
        prepared_calls: usize = 0,
        device_calls: usize = 0,
        frees: usize = 0,

        fn prepared(raw: *anyopaque, request: *const ops.DecoderRuntimeApplyRmsNormLinearArgmaxRequest) !?usize {
            const self: *@This() = @ptrCast(@alignCast(raw));
            try std.testing.expectEqualSlices(i32, self.expected, request.suppress_token_ids);
            self.prepared_calls += 1;
            return self.selected;
        }
        fn device(raw: *anyopaque, _: ops.CT, _: usize, _: usize, _: usize, ids: []const i32, allocator: std.mem.Allocator) !?[]u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            try std.testing.expectEqualSlices(i32, self.expected, ids);
            self.device_calls += 1;
            return try allocator.dupe(u32, &.{1});
        }
        fn free(raw: *anyopaque, _: ops.CT) void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.frees += 1;
        }
    };
    const request_ids = [_]i32{ 0, 50, 106, 212 };
    var probe = Probe{ .expected = &request_ids, .selected = 1 };
    var table: ops.ComputeBackend.VTable = undefined;
    table.decoderRuntimeApplyRmsNormLinearArgmax = Probe.prepared;
    table.argmaxRowsSuppress = Probe.device;
    table.freeTensor = Probe.free;
    const backend: ops.ComputeBackend = .{ .ptr = &probe, .vtable = &table };
    const fake_tensor: ops.CT = @ptrFromInt(1);

    var prepared: ModelOutput = .{
        .prepared_tail = .{
            .final_hidden = fake_tensor,
            .backend = backend,
            .norm = .rms,
            .norm_slot = 1,
            .lm_head_slot = 2,
            .hidden_size = 4,
            .vocab_size = 256,
            .eps = 1e-6,
            // Allowed by both explicit masks. Returning this would prove the old
            // value-only cache check incorrectly reused policy-A output for A/B.
            .greedy_token_id = 9,
        },
    };
    defer prepared.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 1), try prepared.greedyTokenSuppress(std.testing.allocator, 256, &request_ids));
    try std.testing.expectEqual(@as(usize, 1), probe.prepared_calls);
    const changed_request_ids = [_]i32{ 0, 7, 50, 106 };
    probe.expected = &changed_request_ids;
    probe.selected = 2;
    try std.testing.expectEqual(@as(i64, 2), try prepared.greedyTokenSuppress(std.testing.allocator, 256, &changed_request_ids));
    try std.testing.expectEqual(@as(usize, 2), probe.prepared_calls);

    var cached: ModelOutput = .{ .prepared_tail = .{
        .final_hidden = fake_tensor,
        .backend = backend,
        .norm = .rms,
        .norm_slot = 1,
        .lm_head_slot = 2,
        .hidden_size = 4,
        .vocab_size = 256,
        .eps = 1e-6,
        .greedy_token_id = 1,
    } };
    defer cached.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 1), try cached.greedyTokenSuppress(std.testing.allocator, 256, &.{}));
    try std.testing.expectEqual(@as(usize, 2), probe.prepared_calls);

    probe.expected = &request_ids;
    var device: ModelOutput = .{ .device_logits = fake_tensor, .device_logits_backend = backend };
    defer device.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(i64, 1), try device.greedyTokenSuppress(std.testing.allocator, 256, &request_ids));
    try std.testing.expectEqual(@as(usize, 1), probe.device_calls);
}
