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

//! Experimental text-only eager Qwen4Exp executor. One sequence, FP32 state,
//! synchronous Metal numerical execution and bounded SSD expert/ngram caches.
const std = @import("std");
const options = @import("build_options");
const model = @import("../models/qwen4exp.zig");
const store_mod = @import("../models/tensor_store.zig");
const gguf = @import("../gguf/root.zig");
const control = @import("../util/local_control.zig");
const c_file = @import("../util/c_file.zig");
const gpu = @import("../backends/qwen4exp_kernels.zig");
const mr = @import("../backends/metal_runtime.zig");
pub const pager = @import("../runtime/moe/qwen4exp_pager.zig");
const memory = @import("../runtime/tier/memory.zig");
const profile = @import("../runtime/machine_profile.zig");
const MiB = 1024 * 1024;

pub const PlanRequest = struct {
    context: usize,
    /// Zero derives the largest expert cache the envelope admits.
    cache_bytes: usize = 0,
    /// Zero derives the process envelope from the finished plan.
    cap_bytes: usize = 0,
};

pub const Plan = struct {
    cfg: model.Config,
    slots: [48]u16,
    context: usize,
    amounts: memory.AdmissionAmounts,
    host: profile.Host,
    envelope: profile.Envelope,
    /// Bytes of expert slots actually planned, never the requested budget.
    cache_bytes: usize,
    cache_source: profile.Source,
    pub fn inspect(store: store_mod.TensorStore, host: profile.Host, request: PlanRequest) !Plan {
        if (request.context == 0 or request.context > profile.max_context) return error.UnsupportedQwen4ExpContext;
        const file = store.ggufFile() orelse return error.NotGguf;
        const cfg = try model.Config.fromGguf(file);
        const state = try cfg.stateBytes(request.context);
        const inventory = try model.Inventory.inspect(file, cfg);
        try model.Inventory.validateRequired(file, cfg);
        var weights: usize = 0;
        var largest: usize = 0;
        for (file.tensors) |tensor| {
            if (isPagedWeight(tensor.name)) continue;
            const bytes = try rounded(@intCast(gguf.tensor_types.byteLen(tensor.tensor_type, tensor.dimensions).?));
            weights = try std.math.add(usize, weights, bytes);
            largest = @max(largest, bytes);
        }
        weights = try std.math.add(usize, weights, try rounded(embedding_row_bytes));
        var amounts: memory.AdmissionAmounts = .{
            .backend_weight_bytes = weights,
            .backend_kv_bytes = state + 48 * 4 * std.heap.pageSize(),
            // Includes all scratch pages, Metal runtime overhead, metadata and tokenizer.
            .backend_scratch_bytes = 64 * MiB,
            .host_scratch_bytes = 64 * MiB + 64 * MiB + 768 * MiB + pager.read_executor_bytes,
        };
        // Each layer's slots are one page-rounded allocation.
        const rounding = 48 * std.heap.pageSize();
        const cache_source: profile.Source = if (request.cache_bytes != 0) .explicit else .derived;
        const budget = if (request.cache_bytes != 0) request.cache_bytes else blk: {
            const fixed_backend = amounts.backendTotalBytes() + rounding;
            const available = profile.cacheBudget(host, request.cap_bytes, amounts.hostTotalBytes() + fixed_backend, fixed_backend) catch |err| return profileError(err);
            break :blk @min(available, try inventory.cacheBytes(cfg.experts));
        };
        const slots = inventory.cacheSlots(budget, cfg.top_k, cfg.experts) catch |err| return switch (err) {
            // A derived budget below one route means this host cannot hold the model.
            error.InvalidQwen4ExpCacheBudget => if (cache_source == .derived) error.Qwen4ExpMemoryBudgetExceeded else err,
            else => err,
        };
        var cache_bytes: usize = 0;
        for (slots, inventory.per_expert_bytes) |count, stride| {
            const bytes = try rounded(@as(usize, count) * stride);
            cache_bytes += @as(usize, count) * stride;
            largest = @max(largest, bytes);
            amounts.backend_weight_bytes = try std.math.add(usize, amounts.backend_weight_bytes, bytes);
        }
        // An explicit cap must hold the admission reserve too: a plan the
        // controller can never admit is rejected here, where a dry run sees it.
        const resolved = profile.envelope(host, request.cap_bytes, amounts.hostTotalBytes() + amounts.backendTotalBytes(), profile.admission_reserve_bytes) catch |err| return profileError(err);
        profile.checkBackend(host, amounts.backendTotalBytes(), largest) catch |err| return profileError(err);
        return .{
            .cfg = cfg,
            .slots = slots,
            .context = request.context,
            .amounts = amounts,
            .host = host,
            .envelope = resolved,
            .cache_bytes = cache_bytes,
            .cache_source = cache_source,
        };
    }
    fn profileError(err: profile.Error) anyerror {
        return switch (err) {
            error.MemoryBudgetExceeded => error.Qwen4ExpMemoryBudgetExceeded,
            error.MetalWorkingSetExceeded => error.Qwen4ExpMetalWorkingSetExceeded,
            error.MetalBufferTooLarge => error.Qwen4ExpMetalBufferTooLarge,
            else => err,
        };
    }
    pub fn total(self: Plan) usize {
        return self.amounts.hostTotalBytes() + self.amounts.backendTotalBytes();
    }
    pub fn acquire(self: Plan, controller: *memory.AdmissionController) !memory.AdmissionLease {
        if (comptime !options.enable_metal) return error.MetalNotEnabled;
        if (self.host.metal_working_set_bytes == 0) return error.UnifiedMetalDeviceRequired;
        const cap = self.envelope.cap_bytes;
        // Derived or explicit, the envelope is this process's whole bound. A
        // derived one already excludes the node headroom, so the controller
        // must not reserve that headroom a second time inside it.
        controller.configureProcessMemoryLimit(cap, .explicit);
        return controller.tryAcquire(.gpu, .{ .host_limit_bytes = cap, .backend_limit_bytes = cap, .combined_limit_bytes = cap }, self.amounts, true);
    }
};
const embedding_row_bytes = 2560 / 32 * 34;
fn isPagedWeight(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "_exps.weight") or
        std.mem.eql(u8, name, "per_layer_token_embd.weight") or
        std.mem.eql(u8, name, "token_embd.weight");
}
fn rounded(bytes: usize) !usize {
    const page = std.heap.pageSize();
    return (try std.math.add(usize, bytes, page - 1)) & ~(page - 1);
}

const Weight = struct { buffer: gpu.Buffer, format: u32, input: usize, output: usize };
const State = struct {
    parts: [4]?gpu.Buffer = @splat(null),
    fn deinit(self: *State) void {
        for (&self.parts) |*p| if (p.*) |*buffer| buffer.deinit();
    }
    fn ref(self: *State, index: usize) gpu.Ref {
        return self.parts[index].?.ref();
    }
};

pub const Decoder = struct {
    allocator: std.mem.Allocator,
    plan: Plan,
    kernels: gpu.Kernels,
    weights: std.StringHashMapUnmanaged(Weight) = .{},
    layers: [48]pager.Layer = undefined,
    layer_handles: [48]*anyopaque = undefined,
    layer_count: usize = 0,
    states: [48]State = @splat(.{}),
    scratch: [24]gpu.Buffer = undefined,
    scratch_count: usize = 0,
    ngrams: ?pager.NgramCache = null,
    ngram_bytes: ?gpu.Buffer = null,
    /// Borrowed from the store, like every other weight range.
    embedding_reader: ?*const c_file.RangeReader = null,
    embedding_offset: u64 = 0,
    embedding_read_retries: u64 = 0,
    embedding_bytes: ?gpu.Buffer = null,
    ple_history: ?gpu.Buffer = null,
    logits: ?gpu.Buffer = null,
    staging: []u8 = &.{},
    ngram_state: model.NgramState,
    position: usize = 0,
    /// Terminal: the GPU or model state can no longer be trusted.
    failed: bool = false,
    /// Sequence state is partially advanced but the loaded weights, expert
    /// cache and GPU pipeline are intact; reset() restores an empty sequence.
    needs_reset: bool = false,
    controller: ?*memory.AdmissionController = null,
    read_io: ?std.Io = null,
    route_trace: ?[]u16 = null,
    prefill_chunk: usize = 128,
    prefill_storage: ?gpu.Buffer = null,
    batch_linear: bool = false,
    prefill_tile: ?gpu.Buffer = null,
    prefill_result: ?gpu.Buffer = null,
    const prefill_widths = [_]usize{ 10240, 2560, 4, 2560, 20, 25600, 2560, 2560, 10, 2560, 4, 10240, 6144, 48, 48, 6144 };

    /// Caller holds the admission lease from Plan.acquire until after deinit.
    pub fn create(allocator: std.mem.Allocator, store: store_mod.TensorStore, plan: Plan) !*Decoder {
        const self = try allocator.create(Decoder);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .plan = plan, .kernels = try gpu.Kernels.init(), .ngram_state = model.NgramState.init(plan.cfg.eos) };
        errdefer self.deinitContents();
        self.kernels.cancelled = &control.cancelled;
        try control.check();
        // Every range read below shares the descriptors the artifact was verified through.
        try store_mod.SplitGgufStore.bindReaders(store);
        if (std.c.getenv("ANTFLY_QWEN4EXP_ROUTE_TRACE") != null) self.route_trace = try allocator.alloc(u16, plan.context * 48 * 10);
        if (std.c.getenv("ANTFLY_QWEN4EXP_PREFILL_CHUNK")) |value| {
            self.prefill_chunk = try std.fmt.parseInt(usize, std.mem.span(value), 10);
            if (self.prefill_chunk == 1 or self.prefill_chunk > 128) return error.InvalidQwen4ExpPrefillChunk;
        }
        self.batch_linear = std.c.getenv("ANTFLY_QWEN4EXP_BATCH_LINEAR") != null;
        if (self.prefill_chunk != 0) {
            var floats: usize = 0;
            for (prefill_widths[0..if (self.batch_linear) prefill_widths.len else 9]) |width| floats += width * self.prefill_chunk;
            // At most 34.83 MiB, inside the existing 64 MiB scratch reservation.
            self.prefill_storage = try gpu.Buffer.init(self.kernels.runtime, floats * 4);
        }
        if (std.c.getenv("ANTFLY_QWEN4EXP_PREFILL_MPS") != null) {
            self.prefill_tile = try gpu.Buffer.init(self.kernels.runtime, 512 * 6144 * 4);
            self.prefill_result = try gpu.Buffer.init(self.kernels.runtime, 32 * 512 * 4);
        }
        self.staging = try allocator.alloc(u8, 64 * MiB);
        self.ngrams = try pager.NgramCache.init(allocator, store, 64 * MiB);
        self.ngram_bytes = try gpu.Buffer.init(self.kernels.runtime, 16 * 90);
        {
            var range = (try store.describeTensorRange(allocator, "token_embd.weight")) orelse return error.MissingTensorRange;
            defer range.deinit(allocator);
            self.embedding_reader = try store_mod.SplitGgufStore.tensorReader(store, "token_embd.weight");
            const size = self.embedding_reader.?.size;
            if (range.byte_offset > size or range.byte_len > size - range.byte_offset) return error.RegionOutOfBounds;
            self.embedding_offset = range.byte_offset;
            self.embedding_bytes = try gpu.Buffer.init(self.kernels.runtime, embedding_row_bytes);
        }
        self.ple_history = try self.zeroBuffer(9 * 10240);
        self.logits = try gpu.Buffer.init(self.kernels.runtime, 248320 * 4);
        for (&self.scratch) |*buffer| {
            buffer.* = try gpu.Buffer.init(self.kernels.runtime, @max(25600, plan.context * 24) * 4);
            self.scratch_count += 1;
        }
        for (0..48) |layer| {
            try control.check();
            self.layers[layer] = try pager.Layer.init(allocator, store, layer, plan.slots[layer]);
            self.layers[layer].cancelled = &control.cancelled;
            self.layer_handles[layer] = gpu.termite_metal_qwen4exp_alias(self.kernels.runtime, self.layers[layer].bytes.ptr, self.layers[layer].bytes.len) orelse {
                self.layers[layer].deinit();
                return error.MetalBufferAllocationFailed;
            };
            self.layer_count += 1;
            const sizes: [4]usize = if (plan.cfg.isLinear(layer)) .{ 48 * 128 * 128, 3 * 10240, 0, 0 } else .{ plan.context * 512, plan.context * 512, plan.context * 128, ((plan.context + 3) / 4) * 128 };
            for (sizes, 0..) |size, i| if (size > 0) {
                self.states[layer].parts[i] = try self.zeroBuffer(size);
            };
        }
        for (store.ggufFile().?.tensors) |tensor| {
            try control.check();
            if (isPagedWeight(tensor.name)) continue;
            var range = (try store.describeTensorRange(allocator, tensor.name)) orelse return error.MissingTensorRange;
            defer range.deinit(allocator);
            var buffer = try gpu.Buffer.init(self.kernels.runtime, range.byte_len);
            errdefer buffer.deinit();
            const reader = try store_mod.SplitGgufStore.tensorReader(store, tensor.name);
            try reader.read(range.byte_offset, buffer.bytes[0..range.byte_len]);
            const name = try allocator.dupe(u8, tensor.name);
            errdefer allocator.free(name);
            try self.weights.put(allocator, name, .{ .buffer = buffer, .format = tensor.tensor_type.raw(), .input = @intCast(tensor.dimensions[0]), .output = if (tensor.dimensions.len > 1) @intCast(tensor.dimensions[1]) else 1 });
        }
        if (std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_RESIDENCY") == null) {
            var handles: std.ArrayList(*anyopaque) = .empty;
            defer handles.deinit(allocator);
            try handles.append(allocator, self.kernels.dummy.handle);
            var weights = self.weights.valueIterator();
            while (weights.next()) |weight| try handles.append(allocator, weight.buffer.handle);
            try handles.appendSlice(allocator, &self.layer_handles);
            for (self.states) |state| for (state.parts) |part| {
                if (part) |buffer| try handles.append(allocator, buffer.handle);
            };
            for (self.scratch) |buffer| try handles.append(allocator, buffer.handle);
            for ([_]?gpu.Buffer{ self.ngram_bytes, self.embedding_bytes, self.ple_history, self.logits }) |part| {
                if (part) |buffer| try handles.append(allocator, buffer.handle);
            }
            for ([_]?gpu.Buffer{ self.prefill_storage, self.prefill_tile, self.prefill_result }) |part| if (part) |buffer| try handles.append(allocator, buffer.handle);
            try self.kernels.makeResident(handles.items, plan.amounts.backendTotalBytes());
        }
        return self;
    }
    fn zeroBuffer(self: *Decoder, n: usize) !gpu.Buffer {
        var buffer = try gpu.Buffer.init(self.kernels.runtime, n * 4);
        errdefer buffer.deinit();
        try self.kernels.run(.{ .op = 0, .n = @intCast(n) }, &.{}, buffer.ref());
        return buffer;
    }
    fn deinitContents(self: *Decoder) void {
        // All kernels are synchronous. Drop runtime-owned weight aliases first.
        self.kernels.deinit();
        var it = self.weights.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.buffer.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.weights.deinit(self.allocator);
        for (0..self.layer_count) |layer| {
            gpu.termite_metal_buffer_release(self.layer_handles[layer]);
            self.layers[layer].deinit();
        }
        for (&self.states) |*state| state.deinit();
        for (self.scratch[0..self.scratch_count]) |*buffer| buffer.deinit();
        if (self.ngrams) |*cache| cache.deinit();
        if (self.ngram_bytes) |*buffer| buffer.deinit();
        if (self.embedding_bytes) |*buffer| buffer.deinit();
        if (self.ple_history) |*buffer| buffer.deinit();
        if (self.logits) |*buffer| buffer.deinit();
        self.allocator.free(self.staging);
        if (self.route_trace) |trace| self.allocator.free(trace);
        if (self.prefill_storage) |*buffer| buffer.deinit();
        if (self.prefill_tile) |*buffer| buffer.deinit();
        if (self.prefill_result) |*buffer| buffer.deinit();
    }
    pub fn deinit(self: *Decoder) void {
        self.deinitContents();
        self.allocator.destroy(self);
    }
    pub fn overlapEnabled(self: *const Decoder) bool {
        return self.read_io != null and self.kernels.batching_enabled and
            @import("antfly_platform").env.getenvBoolDefault("ANTFLY_QWEN4EXP_IO_OVERLAP", true) and
            std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_IO_OVERLAP") == null;
    }
    pub fn pagingStats(self: *const Decoder) pager.Stats {
        var result: pager.Stats = .{};
        for (self.layers) |layer| {
            result.hits +|= layer.stats.hits;
            result.misses +|= layer.stats.misses;
            result.bytes_read +|= layer.stats.bytes_read;
            result.read_failures +|= layer.stats.read_failures;
            result.read_retries +|= layer.stats.read_retries;
        }
        return result;
    }
    pub fn expertCacheBytes(self: *const Decoder) usize {
        var bytes: usize = 0;
        for (self.layers) |layer| bytes += layer.slotCount() * layer.stride;
        return bytes;
    }
    /// Boundary check: process cancellation and live memory pressure.
    pub fn check(self: *Decoder) !void {
        try control.check();
        if (self.controller) |controller| if (controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
    }
    /// Between the layers of a multi-token chunk a request cancel or memory
    /// pressure may stop the work, at the price of the sequence. A single
    /// token is never interrupted for them: it completes, and its owner
    /// checks at the token boundary with the sequence still intact.
    fn checkInsideChunk(self: *Decoder) !void {
        try control.checkRequest();
        try self.check();
    }
    pub const recoverable = control.recoverableExecutionError;
    fn noteFailure(self: *Decoder, err: anyerror, mutated: bool) void {
        if (!recoverable(err)) {
            self.failed = true;
        } else if (mutated) self.needs_reset = true;
    }
    fn ready(self: *const Decoder) !void {
        if (self.failed) return error.Qwen4ExpStateInvalid;
        if (self.needs_reset) return error.Qwen4ExpStateNeedsReset;
    }
    fn readEmbeddingRow(self: *Decoder, token: usize) !void {
        try pager.readRetrying(self.embedding_reader.?, self.embedding_offset + @as(u64, token) * embedding_row_bytes, self.embedding_bytes.?.bytes[0..embedding_row_bytes], &control.cancelled, null, &self.embedding_read_retries);
    }
    fn s(self: *Decoder, index: usize) gpu.Ref {
        return self.scratch[index].ref();
    }
    fn w(self: *Decoder, layer: ?usize, suffix: []const u8) !*Weight {
        var name: [128]u8 = undefined;
        return self.weights.getPtr(if (layer) |l| try std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ l, suffix }) else suffix) orelse error.MissingQwen4ExpTensor;
    }
    fn wr(self: *Decoder, layer: ?usize, suffix: []const u8) !gpu.Ref {
        return (try self.w(layer, suffix)).buffer.ref();
    }
    fn mm(self: *Decoder, layer: ?usize, name: []const u8, input: gpu.Ref, out: gpu.Ref) !void {
        const weight = try self.w(layer, name);
        try self.kernels.matvec(weight.format, weight.buffer.logical_bytes, weight.input, weight.output, weight.buffer.ref(), input, out);
    }
    fn run(self: *Decoder, params: gpu.Params, inputs: []const gpu.Ref, out: gpu.Ref) !void {
        try self.kernels.run(params, inputs, out);
    }
    fn norm(self: *Decoder, input: gpu.Ref, weight: gpu.Ref, out: gpu.Ref, n: usize, dim: usize, mode: u32) !void {
        if (dim == 0 or n % dim != 0) return error.InvalidQwen4ExpNormShape;
        try self.run(.{ .op = 2, .n = @intCast(n / dim * 32), .dim = @intCast(dim), .mode = mode, .epsilon = self.plan.cfg.epsilon }, &.{ input, weight }, out);
    }
    fn hc(self: *Decoder, layer: ?usize, kind: []const u8, inject: bool) !void {
        var names: [4][96]u8 = undefined;
        const prefix = if (layer == null) "output_hc" else kind;
        const norm_name = try std.fmt.bufPrint(&names[0], "{s}_norm.weight", .{prefix});
        const down = try std.fmt.bufPrint(&names[1], "{s}_down.weight", .{prefix});
        const up = try std.fmt.bufPrint(&names[2], "{s}_up.weight", .{prefix});
        try self.norm(self.s(0), try self.wr(layer, norm_name), self.s(1), 10240, 2560, 0);
        try self.mm(layer, down, self.s(1), self.s(2));
        try self.run(.{ .op = 3, .n = 320, .scale = 0.25 }, &.{self.s(2)}, self.s(3));
        try self.mm(layer, up, self.s(3), self.s(2));
        try self.run(.{ .op = 4, .n = 2560, .dim = 2560 }, &.{ self.s(1), self.s(2) }, self.s(4));
        if (inject) try self.mm(layer, try std.fmt.bufPrint(&names[3], "{s}_inject.weight", .{prefix}), self.s(1), self.s(5));
    }
    fn combine(self: *Decoder, branch: gpu.Ref) !void {
        try self.run(.{ .op = 5, .n = 10240, .dim = 2560 }, &.{ self.s(0), branch, self.s(5) }, self.s(0));
    }
    fn ple(self: *Decoder, token: u32) !void {
        const cfg = self.plan.cfg;
        const rows = try self.ngram_state.next(token, cfg.eos, cfg.multipliers, cfg.head_offsets, cfg.head_vocab_sizes);
        try self.ngrams.?.readRows(&rows, self.ngram_bytes.?.bytes[0..1440]);
        try self.run(.{ .op = 8, .n = 2560 }, &.{self.ngram_bytes.?.ref()}, self.s(4));
        try self.capture("ple_embd", 1, self.s(4), 2560);
        try self.mm(1, "ple_key.weight", self.s(4), self.s(7));
        try self.mm(1, "ple_value.weight", self.s(4), self.s(9));
        try self.norm(self.s(7), try self.wr(1, "ple_norm_key.weight"), self.s(8), 10240, 2560, 0);
        try self.norm(self.s(0), try self.wr(1, "ple_norm_query.weight"), self.s(10), 10240, 2560, 0);
        try self.run(.{ .op = 17, .n = 10240, .dim = 2560 }, &.{ self.s(8), self.s(10), self.s(9) }, self.s(11));
        try self.capture("ple_gated", 1, self.s(11), 10240);
        try self.norm(self.s(11), try self.wr(1, "ple_norm_conv.weight"), self.s(12), 10240, 2560, 0);
        try self.run(.{ .op = 9, .n = 10240, .stride = 3 }, &.{ self.s(12), try self.wr(1, "ple_conv1d.weight"), self.ple_history.?.ref() }, self.s(13));
        try self.capture("ple_conv", 1, self.s(13), 10240);
        try self.run(.{ .op = 16, .n = 10240 }, &.{ self.s(11), self.s(13) }, self.s(14));
        try self.run(.{ .op = 16, .n = 10240 }, &.{ self.s(0), self.s(14) }, self.s(0));
    }
    fn linear(self: *Decoder, l: usize) !void {
        try self.mm(l, "attn_qkv.weight", self.s(4), self.s(7));
        try self.mm(l, "attn_gate.weight", self.s(4), self.s(10));
        try self.mm(l, "ssm_beta.weight", self.s(4), self.s(11));
        try self.mm(l, "ssm_alpha.weight", self.s(4), self.s(12));
        try self.linearState(l);
        try self.mm(l, "ssm_out.weight", self.s(15), self.s(6));
    }
    fn linearState(self: *Decoder, l: usize) !void {
        try self.run(.{ .op = 9, .n = 10240, .stride = 1 }, &.{ self.s(7), try self.wr(l, "ssm_conv1d.weight"), self.states[l].ref(1) }, self.s(8));
        try self.run(.{ .op = 1, .n = 10240, .dim = 10240 }, &.{self.s(8)}, self.s(9));
        try self.norm(self.s(8), self.s(8), self.s(9), 4096, 128, 2);
        try self.run(.{ .op = 10, .n = 6144 }, &.{ self.s(9), self.s(11), self.s(12), try self.wr(l, "ssm_a"), try self.wr(l, "ssm_dt.bias"), self.states[l].ref(0) }, self.s(13));
        try self.norm(self.s(13), try self.wr(l, "ssm_norm.weight"), self.s(14), 6144, 128, 1);
        try self.run(.{ .op = 11, .n = 6144 }, &.{ self.s(14), self.s(10) }, self.s(15));
    }
    fn rope(self: *Decoder, input: gpu.Ref, out: gpu.Ref, n: usize, dim: usize, pos: usize) !void {
        try self.run(.{ .op = 19, .n = @intCast(n), .dim = @intCast(dim), .position = @intCast(pos), .theta = self.plan.cfg.theta }, &.{input}, out);
    }
    fn attention(self: *Decoder, l: usize) !void {
        const pos: u32 = @intCast(self.position);
        try self.mm(l, "attn_q.weight", self.s(4), self.s(7));
        try self.run(.{ .op = 18, .n = 6144 }, &.{ self.s(7), self.s(10) }, self.s(8));
        try self.norm(self.s(8), try self.wr(l, "attn_q_norm.weight"), self.s(9), 6144, 256, 1);
        try self.rope(self.s(9), self.s(7), 6144, 256, self.position);
        try self.mm(l, "attn_k.weight", self.s(4), self.s(8));
        try self.norm(self.s(8), try self.wr(l, "attn_k_norm.weight"), self.s(9), 512, 256, 1);
        try self.rope(self.s(9), self.s(8), 512, 256, self.position);
        try self.run(.{ .op = 20, .n = 512, .dim = 512, .position = pos }, &.{self.s(8)}, self.states[l].ref(0));
        try self.mm(l, "attn_v.weight", self.s(4), self.s(8));
        try self.run(.{ .op = 20, .n = 512, .dim = 512, .position = pos }, &.{self.s(8)}, self.states[l].ref(1));
        try self.mm(l, "indexer.q_proj.weight", self.s(4), self.s(8));
        try self.norm(self.s(8), try self.wr(l, "indexer.q_norm.weight"), self.s(9), 512, 128, 1);
        try self.rope(self.s(9), self.s(11), 512, 128, self.position);
        try self.mm(l, "indexer.k_proj.weight", self.s(4), self.s(8));
        try self.run(.{ .op = 20, .n = 128, .dim = 128, .position = pos }, &.{self.s(8)}, self.states[l].ref(2));
        if ((pos + 1) % 4 == 0) {
            try self.run(.{ .op = 21, .n = 128, .position = pos }, &.{self.states[l].ref(2)}, self.s(8));
            try self.norm(self.s(8), try self.wr(l, "indexer.k_norm.weight"), self.s(9), 128, 128, 1);
            try self.rope(self.s(9), self.s(8), 128, 128, self.position - 3);
            try self.run(.{ .op = 20, .n = 128, .dim = 128, .position = pos / 4 }, &.{self.s(8)}, self.states[l].ref(3));
        }
        // Selection keeps index_budget tokens' worth of compressed blocks. Up
        // to that many blocks it keeps every one, so the scoring pass and the
        // single-thread top-k are skipped and attention reads no mask (mode 1).
        // The mask would also exclude blocks whose index score is NaN, but a
        // NaN there already means non-finite logits.
        const blocks = (pos + 1) / self.plan.cfg.compress_ratio;
        const prune = blocks > self.plan.cfg.index_budget / self.plan.cfg.compress_ratio;
        if (prune) {
            try self.run(.{ .op = 22, .n = @intCast(blocks) }, &.{ self.s(11), self.states[l].ref(3) }, self.s(12));
            try self.run(.{ .op = 23, .n = 1, .position = pos }, &.{self.s(12)}, self.s(13));
        }
        try self.run(.{ .op = 24, .n = 24 * (pos + 1), .position = pos, .mode = @intFromBool(!prune) }, &.{ self.s(7), self.states[l].ref(0), self.s(13) }, self.s(14));
        try self.run(.{ .op = 25, .n = 24, .position = pos }, &.{self.s(14)}, self.s(15));
        try self.run(.{ .op = 26, .n = 6144, .position = pos }, &.{ self.s(15), self.states[l].ref(1), self.s(10) }, self.s(16));
        try self.mm(l, "attn_output.weight", self.s(16), self.s(6));
    }
    fn computeRouter(self: *Decoder, l: usize) !void {
        // Continue the attention batch through routing; only the CPU expert
        // selection needs a completion boundary.
        errdefer self.kernels.cancelBatch();
        try self.mm(l, "ffn_gate_shexp.weight", self.s(4), self.s(7));
        try self.mm(l, "ffn_up_shexp.weight", self.s(4), self.s(8));
        try self.run(.{ .op = 12, .n = 640 }, &.{ self.s(7), self.s(8) }, self.s(9));
        try self.mm(l, "ffn_down_shexp.weight", self.s(9), self.s(10));
        try self.mm(l, "ffn_gate_inp_shexp.weight", self.s(4), self.s(11));
        try self.run(.{ .op = 15, .n = 2560 }, &.{ self.s(10), self.s(11) }, self.s(19));
        if (l == 0) try self.capture("shared", l, self.s(19), 2560);
        try self.mm(l, "ffn_gate_inp.weight", self.s(4), self.s(12));
        try self.run(.{ .op = 13, .n = 32 }, &.{self.s(12)}, self.s(18));
        try self.kernels.endProfiledBatch(.attention); // Includes shared expert/router.
        try self.capture("route", l, self.s(18), 20);
    }
    fn moe(self: *Decoder, l: usize) !void {
        try self.computeRouter(l);
        errdefer self.kernels.cancelBatch();
        const ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(self.scratch[18].bytes[0..40])));
        if (self.route_trace) |trace| {
            for (ids, 0..) |id, route| trace[(self.position * 48 + l) * 10 + route] = std.math.cast(u16, id) orelse return error.Qwen4ExpInvalidRoute;
        }
        const overlap = self.overlapEnabled();
        if (overlap) {
            var pending: pager.Layer.Pending = .{};
            try pending.begin(&self.layers[l], ids, self.staging, self.read_io);
            defer pending.abort();
            // Drain all encoded/submitted GPU readers before dropping ownership.
            errdefer self.kernels.cancelBatch();
            const grouped = std.c.getenv("ANTFLY_QWEN4EXP_GROUPED_EXPERTS") != null;
            var resident_mask: u32 = 0;
            const slots = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(self.scratch[21].bytes[0..40])));
            for (0..10) |route| {
                slots[route] = pending.route.route_slots[route];
                if (!pending.route.repaired[route]) resident_mask |= @as(u32, 1) << @intCast(route);
            }
            if (pending.route.hit_count > 0) {
                try self.kernels.beginBatch();
                if (grouped) try self.groupedExperts(l, resident_mask) else for (0..10) |route| {
                    if (pending.route.repaired[route]) continue;
                    var sources: [3][]const u8 = undefined;
                    for (0..3) |projection| sources[projection] = try pending.residentProjection(route, projection);
                    try self.expert(l, route, sources, try self.s(20).slice(route * 2560 * 4, 2560 * 4));
                }
                try self.kernels.submitBatch();
            }
            const paging_start = if (self.kernels.profile.enabled) gpu.Profile.now() else 0;
            var lease = try pending.finish();
            self.kernels.profile.record(29, paging_start);
            defer lease.releaseAfterCompletion();
            errdefer self.kernels.cancelBatch();
            try self.kernels.waitBatch(.experts);
            try self.kernels.beginBatch();
            if (grouped) try self.groupedExperts(l, 1023 ^ resident_mask) else for (0..10) |route| {
                if (!lease.route.repaired[route]) continue;
                var sources: [3][]const u8 = undefined;
                for (0..3) |projection| sources[projection] = try lease.projection(route, projection);
                try self.expert(l, route, sources, try self.s(20).slice(route * 2560 * 4, 2560 * 4));
            }
            // Read completion never changes floating point reduction order.
            for (0..10) |route| try self.run(.{ .op = 14, .n = 2560, .index = @intCast(10 + route) }, &.{ try self.s(20).slice(route * 2560 * 4, 2560 * 4), self.s(18) }, self.s(19));
            try self.combine(self.s(19));
            try self.kernels.endProfiledBatch(.experts);
        } else {
            const paging_start = if (self.kernels.profile.enabled) gpu.Profile.now() else 0;
            var lease = try self.layers[l].acquireWithIo(ids, self.staging, self.read_io);
            self.kernels.profile.record(29, paging_start);
            defer lease.releaseAfterCompletion();
            try self.kernels.beginBatch();
            errdefer self.kernels.cancelBatch();
            for (0..10) |route| {
                var sources: [3][]const u8 = undefined;
                for (0..3) |projection| sources[projection] = try lease.projection(route, projection);
                try self.expert(l, route, sources, self.s(10));
                try self.run(.{ .op = 14, .n = 2560, .index = @intCast(10 + route) }, &.{ self.s(10), self.s(18) }, self.s(19));
            }
            try self.combine(self.s(19));
            try self.kernels.endProfiledBatch(.experts);
        }
    }
    fn groupedExperts(self: *Decoder, l: usize, mask: u32) !void {
        if (mask == 0) return;
        const layer = &self.layers[l];
        var offset: usize = 0;
        for (layer.projections, 0..) |p, projection| {
            try self.run(.{ .op = 28, .n = @intCast(10 * p.output * 32), .dim = @intCast(p.input), .groups = @intCast(p.output), .stride = @intCast(layer.stride), .position = @intCast(offset), .mode = p.format.raw(), .index = mask, .scale = if (projection == 2) 1 else 0 }, &.{ .{ .handle = self.layer_handles[l], .bytes = layer.bytes.len }, self.s(if (projection == 2) 9 else 4), self.s(21) }, self.s(if (projection == 0) 7 else if (projection == 1) 8 else 20));
            if (projection == 1) try self.run(.{ .op = 29, .n = 6400, .dim = 640, .index = mask }, &.{ self.s(7), self.s(8) }, self.s(9));
            offset += p.stride;
        }
    }
    fn expert(self: *Decoder, l: usize, route: usize, sources: [3][]const u8, output: gpu.Ref) !void {
        for (sources, 0..) |source, projection| {
            const layer = &self.layers[l];
            const p = layer.projections[projection];
            const offset = @intFromPtr(source.ptr) - @intFromPtr(layer.bytes.ptr);
            const weight = gpu.Ref{ .handle = self.layer_handles[l], .offset = offset, .bytes = source.len };
            const destination = if (projection == 2) output else self.s(if (projection == 0) 7 else 8);
            try self.kernels.matvec(p.format.raw(), source.len, p.input, p.output, weight, if (projection == 2) self.s(9) else self.s(4), destination);
            if (l == 0 and route == 0) try self.capture(if (projection == 0) "expert_gate" else if (projection == 1) "expert_up" else "expert_down", l, destination, p.output);
            if (projection == 1) try self.run(.{ .op = 12, .n = 640 }, &.{ self.s(7), self.s(8) }, self.s(9));
        }
    }
    fn capture(self: *Decoder, name: []const u8, layer: usize, ref: gpu.Ref, n: usize) !void {
        const dir = std.c.getenv("ANTFLY_QWEN4EXP_CAPTURE_DIR") orelse return;
        var filename: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&filename, "{s}/{s}-{d}.f32", .{ std.mem.span(dir), name, layer });
        for (self.scratch[0..self.scratch_count]) |buffer| {
            if (buffer.handle != ref.handle) continue;
            try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{ .sub_path = path, .data = buffer.bytes[ref.offset..][0 .. n * 4] });
            return;
        }
    }
    fn prefillBytes(self: *Decoder, part: usize) []u8 {
        var offset: usize = 0;
        for (prefill_widths[0..part]) |width| offset += width * self.prefill_chunk * 4;
        return self.prefill_storage.?.bytes[offset..][0 .. prefill_widths[part] * self.prefill_chunk * 4];
    }
    fn prefillRef(self: *Decoder, part: usize, offset: usize, bytes: usize) !gpu.Ref {
        const data = self.prefillBytes(part);
        if (offset > data.len or bytes > data.len - offset) return error.MetalBufferRange;
        return .{ .handle = self.prefill_storage.?.handle, .offset = @intFromPtr(data.ptr) - @intFromPtr(self.prefill_storage.?.bytes.ptr) + offset, .bytes = bytes };
    }
    fn prefillMatmul(self: *Decoder, format: u32, weight: gpu.Ref, input_dim: usize, output_dim: usize, rows: usize, input: gpu.Ref, output: gpu.Ref) !void {
        if (format == 0 or format == 30) {
            for (0..rows) |r| try self.kernels.matvec(format, weight.bytes, input_dim, output_dim, weight, try input.slice(r * input_dim * 4, input_dim * 4), try output.slice(r * output_dim * 4, output_dim * 4));
        } else if (self.prefill_tile) |tile| {
            try self.kernels.matmulTiled(format, weight, input_dim, output_dim, rows, input, output, tile.ref(), self.prefill_result.?.ref());
        } else try self.kernels.matmul(format, weight, input_dim, output_dim, rows, input, output);
    }
    fn batchProjection(self: *Decoder, l: usize, name: []const u8, rows: usize, input_part: usize, output_part: usize) !void {
        const weight = try self.w(l, name);
        try self.kernels.beginBatch();
        var at: usize = 0;
        while (at < rows) {
            const count = @min(32, rows - at);
            try self.prefillMatmul(weight.format, weight.buffer.ref(), weight.input, weight.output, count, try self.prefillRef(input_part, at * weight.input * 4, count * weight.input * 4), try self.prefillRef(output_part, at * weight.output * 4, count * weight.output * 4));
            at += count;
        }
        try self.kernels.endProfiledBatch(.attention);
    }
    fn prefillLinear(self: *Decoder, l: usize, tokens: []const i32, base: usize) !void {
        for (tokens, 0..) |token, t| {
            self.position = base + t;
            @memcpy(self.scratch[0].bytes[0..40960], self.prefillBytes(0)[t * 40960 ..][0..40960]);
            try self.kernels.beginBatch();
            if (l == 1) try self.ple(@intCast(token));
            try self.hc(l, "hc_attn", true);
            try self.kernels.endProfiledBatch(.attention);
            for ([_]usize{ 0, 4, 5 }, [_]usize{ 0, 9, 10 }) |scratch, part| {
                const bytes = prefill_widths[part] * 4;
                @memcpy(self.prefillBytes(part)[t * bytes ..][0..bytes], self.scratch[scratch].bytes[0..bytes]);
            }
        }
        for ([_][]const u8{ "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight" }, 11..) |name, part| try self.batchProjection(l, name, tokens.len, 9, part);
        for (0..tokens.len) |t| {
            for ([_]usize{ 7, 10, 11, 12 }, 11..) |scratch, part| {
                const bytes = prefill_widths[part] * 4;
                @memcpy(self.scratch[scratch].bytes[0..bytes], self.prefillBytes(part)[t * bytes ..][0..bytes]);
            }
            try self.kernels.beginBatch();
            try self.linearState(l);
            try self.kernels.endProfiledBatch(.attention);
            @memcpy(self.prefillBytes(15)[t * 6144 * 4 ..][0 .. 6144 * 4], self.scratch[15].bytes[0 .. 6144 * 4]);
        }
        try self.batchProjection(l, "ssm_out.weight", tokens.len, 15, 9);
    }
    /// Layer-major bounded prefill. Attention/recurrent updates retain causal
    /// token order; routed experts are loaded once per union and use GEMMs.
    /// Only CPU copies/gathers occur on the host, never model arithmetic.
    /// `output` asks for the prediction after the chunk. Callers pass false
    /// for every chunk but a prompt's last, whose result alone is used; the
    /// output head is then skipped and 0 returned.
    pub fn prefillChunk(self: *Decoder, tokens: []const i32, output: bool) !u32 {
        try self.ready();
        if (tokens.len == 0 or tokens.len > self.prefill_chunk or self.position + tokens.len > self.plan.context) return error.Qwen4ExpContextOrTokenOutOfRange;
        for (tokens) |token| if (token < 0 or token >= 248320) return error.Qwen4ExpContextOrTokenOutOfRange;
        var mutated = false;
        return self.prefillChunkUnguarded(tokens, output or std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PREFILL_HEAD_SKIP") != null, &mutated) catch |err| {
            self.kernels.cancelBatch();
            self.noteFailure(err, mutated);
            return err;
        };
    }
    fn prefillChunkUnguarded(self: *Decoder, tokens: []const i32, output: bool, mutated: *bool) !u32 {
        // Until the first layer runs, only scratch is written: a pressure
        // blip, a cancellation or an unreadable embedding row leaves the
        // sequence exactly where the previous call left it.
        try self.check();
        const base = self.position;
        for (tokens, 0..) |token, t| {
            try self.readEmbeddingRow(@intCast(token));
            try self.run(.{ .op = 7, .n = 10240, .dim = 2560, .index = 0 }, &.{self.embedding_bytes.?.ref()}, try self.prefillRef(0, t * 10240 * 4, 10240 * 4));
        }
        mutated.* = true;
        for (0..48) |l| {
            try self.checkInsideChunk();
            const batched = self.batch_linear and self.plan.cfg.isLinear(l);
            if (batched) try self.prefillLinear(l, tokens, base);
            for (tokens, 0..) |token, t| {
                self.position = base + t;
                @memcpy(self.scratch[0].bytes[0 .. 10240 * 4], self.prefillBytes(0)[t * 10240 * 4 ..][0 .. 10240 * 4]);
                try self.kernels.beginBatch();
                if (batched) {
                    @memcpy(self.scratch[5].bytes[0..16], self.prefillBytes(10)[t * 16 ..][0..16]);
                    @memcpy(self.scratch[6].bytes[0..10240], self.prefillBytes(9)[t * 10240 ..][0..10240]);
                } else {
                    if (l == 1) try self.ple(@intCast(token));
                    try self.hc(l, "hc_attn", true);
                    if (self.plan.cfg.isLinear(l)) try self.linear(l) else try self.attention(l);
                }
                try self.combine(self.s(6));
                try self.hc(l, "hc_ffn", true);
                try self.computeRouter(l);
                const route_ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(self.scratch[18].bytes[0..40])));
                for (route_ids, 0..) |id, r| {
                    if (id >= 512 or std.mem.indexOfScalar(u32, route_ids[0..r], id) != null) return error.Qwen4ExpInvalidRoute;
                }
                for ([_]usize{ 0, 4, 5, 19, 18 }, 0..) |scratch, part| {
                    const bytes = prefill_widths[part] * 4;
                    @memcpy(self.prefillBytes(part)[t * bytes ..][0..bytes], self.scratch[scratch].bytes[0..bytes]);
                }
                if (self.route_trace) |trace| {
                    const ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(self.scratch[18].bytes[0..40])));
                    for (ids, 0..) |id, r| trace[(self.position * 48 + l) * 10 + r] = std.math.cast(u16, id) orelse return error.Qwen4ExpInvalidRoute;
                }
            }
            var used: [512]bool = @splat(false);
            for (0..tokens.len) |t| {
                const ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(self.prefillBytes(4)[t * 80 ..][0..40])));
                for (ids) |id| used[id] = true;
            }
            var experts: [512]u32 = undefined;
            var expert_count: usize = 0;
            for (used, 0..) |present, id| if (present) {
                experts[expert_count] = @intCast(id);
                expert_count += 1;
            };
            var expert_start: usize = 0;
            while (expert_start < expert_count) {
                const expert_end = @min(expert_count, expert_start + 10);
                var lease = try self.layers[l].acquireWithIo(experts[expert_start..expert_end], self.staging, self.read_io);
                defer lease.releaseAfterCompletion();
                errdefer self.kernels.cancelBatch();
                // Each expert owns an immutable index slice until the entire
                // group completes. GPU gathers/scatters allow scratch reuse
                // without a CPU completion boundary for every expert.
                try self.kernels.beginBatch();
                for (experts[expert_start..expert_end], 0..) |expert_id, lease_index| {
                    const index_offset = lease_index * self.prefill_chunk * 4;
                    const locations = std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(self.prefillBytes(8)[index_offset..][0 .. self.prefill_chunk * 4])));
                    var count: usize = 0;
                    for (0..tokens.len) |t| {
                        const ids = std.mem.bytesAsSlice(u32, @as([]align(4) const u8, @alignCast(self.prefillBytes(4)[t * 80 ..][0..40])));
                        for (ids, 0..) |id, r| if (id == expert_id) {
                            locations[count] = @intCast(t * 10 + r);
                            count += 1;
                        };
                    }
                    if (count == 0) continue;
                    const indices = try self.prefillRef(8, index_offset, count * 4);
                    try self.run(.{ .op = 30, .n = @intCast(count * 2560), .dim = 2560 }, &.{ try self.prefillRef(1, 0, tokens.len * 2560 * 4), indices }, try self.prefillRef(6, 0, count * 2560 * 4));
                    var batch_start: usize = 0;
                    while (batch_start < count) {
                        const batch_rows: usize = @min(32, count - batch_start);
                        for (self.layers[l].projections, 0..) |projection, p| {
                            const source = try lease.projection(lease_index, p);
                            const weight = gpu.Ref{ .handle = self.layer_handles[l], .offset = @intFromPtr(source.ptr) - @intFromPtr(self.layers[l].bytes.ptr), .bytes = source.len };
                            try self.prefillMatmul(projection.format.raw(), weight, projection.input, projection.output, batch_rows, if (p == 2) self.s(9) else try self.prefillRef(6, batch_start * 2560 * 4, batch_rows * 2560 * 4), if (p == 2) try self.prefillRef(7, batch_start * 2560 * 4, batch_rows * 2560 * 4) else self.s(if (p == 0) 7 else 8));
                            if (p == 1) try self.run(.{ .op = 12, .n = @intCast(batch_rows * 640) }, &.{ self.s(7), self.s(8) }, self.s(9));
                        }
                        batch_start += batch_rows;
                    }
                    try self.run(.{ .op = 31, .n = @intCast(count * 2560), .dim = 2560 }, &.{ try self.prefillRef(7, 0, count * 2560 * 4), indices }, try self.prefillRef(5, 0, tokens.len * 25600 * 4));
                }
                try self.kernels.endProfiledBatch(.experts);
                expert_start = expert_end;
            }
            for (0..tokens.len) |t| {
                for ([_]usize{ 0, 4, 5, 19, 18 }, 0..) |scratch, part| {
                    const bytes = prefill_widths[part] * 4;
                    @memcpy(self.scratch[scratch].bytes[0..bytes], self.prefillBytes(part)[t * bytes ..][0..bytes]);
                }
                try self.kernels.beginBatch();
                for (0..10) |r| try self.run(.{ .op = 14, .n = 2560, .index = @intCast(10 + r) }, &.{ try self.prefillRef(5, (t * 10 + r) * 2560 * 4, 2560 * 4), self.s(18) }, self.s(19));
                try self.combine(self.s(19));
                try self.kernels.endProfiledBatch(.experts);
                @memcpy(self.prefillBytes(0)[t * 10240 * 4 ..][0 .. 10240 * 4], self.scratch[0].bytes[0 .. 10240 * 4]);
            }
        }
        if (!output) {
            self.position = base + tokens.len;
            return 0;
        }
        try self.kernels.beginBatch();
        try self.hc(null, "", false);
        try self.mm(null, "output.weight", self.s(4), self.logits.?.ref());
        try self.run(.{ .op = 27, .n = 32, .dim = 248320 }, &.{self.logits.?.ref()}, self.s(23));
        try self.kernels.endProfiledBatch(.output);
        const result = std.mem.readInt(u32, self.scratch[23].bytes[0..4], .little);
        if (result >= 248320) return error.Qwen4ExpNonFiniteLogits;
        self.position = base + tokens.len;
        return result;
    }
    /// Start an empty sequence. This also clears a recoverable failure that
    /// stopped inside a token; the expert cache and loaded weights are kept.
    /// A terminal failure still requires destroying the decoder.
    pub fn reset(self: *Decoder) !void {
        if (self.failed) return error.Qwen4ExpStateInvalid;
        try self.kernels.endBatch();
        for (self.states) |state| for (state.parts) |part| {
            if (part) |b| @memset(b.bytes, 0);
        };
        if (self.ple_history) |b| @memset(b.bytes, 0);
        self.ngram_state = model.NgramState.init(self.plan.cfg.eos);
        self.position = 0;
        self.needs_reset = false;
    }
    pub fn step(self: *Decoder, token: u32) !u32 {
        return self.stepWithOutput(token, true);
    }
    pub fn prefillToken(self: *Decoder, token: u32, last: bool) !u32 {
        return self.stepWithOutput(token, last or std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PREFILL_HEAD_SKIP") != null);
    }
    fn stepWithOutput(self: *Decoder, token: u32, output: bool) !u32 {
        try self.ready();
        if (self.position >= self.plan.context or token >= 248320) return error.Qwen4ExpContextOrTokenOutOfRange;
        var mutated = false;
        return self.stepUnguarded(token, output, &mutated) catch |err| {
            self.kernels.cancelBatch();
            self.noteFailure(err, mutated);
            return err;
        };
    }
    fn stepUnguarded(self: *Decoder, token: u32, output: bool, mutated: *bool) !u32 {
        // See prefillChunkUnguarded: nothing before the first layer touches the sequence.
        try self.check();
        // Only one 2720-byte Q8 row is used. Keeping the entire 644 MiB table
        // resident wastes expert-cache capacity and Metal residency work.
        try self.readEmbeddingRow(token);
        try self.run(.{ .op = 7, .n = 10240, .dim = 2560, .index = 0 }, &.{self.embedding_bytes.?.ref()}, self.s(0));
        try self.capture("embed", 0, self.s(0), 2560);
        mutated.* = true;
        for (0..48) |l| {
            try control.check();
            try self.kernels.beginBatch();
            if (l == 1) try self.ple(token);
            try self.hc(l, "hc_attn", true);
            try self.capture("hc_attn_mixed", l, self.s(4), 2560);
            if (self.plan.cfg.isLinear(l)) try self.linear(l) else try self.attention(l);
            try self.capture("attn_out", l, self.s(6), 2560);
            try self.combine(self.s(6));
            try self.hc(l, "hc_ffn", true);
            try self.capture("hc_ffn_mixed", l, self.s(4), 2560);
            try self.moe(l);
            try self.capture("moe_out", l, self.s(19), 2560);
            try self.capture("l_last", l, self.s(0), 10240);
        }
        if (!output) {
            self.position += 1;
            return 0;
        }
        try self.kernels.beginBatch();
        try self.hc(null, "", false);
        try self.mm(null, "output.weight", self.s(4), self.logits.?.ref());
        try self.run(.{ .op = 27, .n = 32, .dim = 248320 }, &.{self.logits.?.ref()}, self.s(23));
        try self.kernels.endProfiledBatch(.output);
        const result = std.mem.readInt(u32, self.scratch[23].bytes[0..4], .little);
        if (result >= 248320) return error.Qwen4ExpNonFiniteLogits;
        self.position += 1;
        return result;
    }
};

test "Qwen4Exp installed decoder performs native Metal generation" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    const path = std.c.getenv("ANTFLY_QWEN4EXP_RUN_GGUF") orelse return error.SkipZigTest;
    const store = try store_mod.SplitGgufStore.open(std.testing.allocator, std.mem.span(path));
    defer store.deinit();
    const plan = try Plan.inspect(store, try profile.Host.detect(), .{ .context = 16, .cache_bytes = 2 * 1024 * MiB, .cap_bytes = 10 * 1024 * MiB });
    var controller: memory.AdmissionController = .{};
    var lease = try plan.acquire(&controller);
    defer lease.release();
    var executor = std.Io.Threaded.init(std.testing.allocator, .{ .async_limit = .limited(pager.read_lane_count), .concurrent_limit = .limited(pager.read_lane_count), .stack_size = pager.read_stack_bytes });
    defer executor.deinit();
    const decoder = try Decoder.create(std.testing.allocator, store, plan);
    defer decoder.deinit();
    decoder.read_io = executor.io();
    // Teacher-forced sequences permit independent per-position parity,
    // including EOS reset and recurrence, without depending on greedy drift.
    const tokens = if (std.c.getenv("ANTFLY_QWEN4EXP_TOKENS")) |value| std.mem.span(value) else "9707";
    var ids = std.mem.splitScalar(u8, tokens, ',');
    var logits: std.ArrayList(u8) = .empty;
    defer logits.deinit(std.testing.allocator);
    while (ids.next()) |id| {
        const token = try decoder.step(try std.fmt.parseInt(u32, id, 10));
        std.debug.print("Qwen4Exp native Metal position={d} next token={d} admitted_bytes={d}\n", .{ decoder.position - 1, token, plan.total() });
        try std.testing.expect(token < 248320);
        for (decoder.logits.?.floats()) |value| try std.testing.expect(std.math.isFinite(value));
        try logits.appendSlice(std.testing.allocator, decoder.logits.?.bytes[0 .. 248320 * 4]);
    }
    std.debug.print("Qwen4Exp memory={any}\n", .{@import("antfly_platform").process_memory.pressureSnapshot()}); // raw smoke only; not a quality assertion
    if (std.c.getenv("ANTFLY_QWEN4EXP_LOGITS")) |output| try std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = std.mem.span(output), .data = logits.items });
}

test "Qwen4Exp installed chunked prefill preserves complete recurrent state and decode continuation" {
    if (comptime !options.enable_metal) return error.SkipZigTest;
    const path = std.c.getenv("ANTFLY_QWEN4EXP_PREFILL_CHECK") orelse return error.SkipZigTest;
    // Enable the bounded buffer through the same construction path as the CLI.
    if (std.c.getenv("ANTFLY_QWEN4EXP_PREFILL_CHUNK") == null) return error.SkipZigTest;
    const a = std.testing.allocator;
    const store = try store_mod.SplitGgufStore.open(a, std.mem.span(path));
    defer store.deinit();
    const test_cache_mb = if (std.c.getenv("ANTFLY_QWEN4EXP_TEST_CACHE_MB")) |value| try std.fmt.parseInt(usize, std.mem.span(value), 10) else 2048;
    const plan = try Plan.inspect(store, try profile.Host.detect(), .{ .context = 32, .cache_bytes = test_cache_mb * MiB, .cap_bytes = 9 * 1024 * MiB });
    var controller: memory.AdmissionController = .{};
    var admission = try plan.acquire(&controller);
    defer admission.release();
    var executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(pager.read_lane_count), .concurrent_limit = .limited(pager.read_lane_count), .stack_size = pager.read_stack_bytes });
    defer executor.deinit();
    const tokens = [_]i32{ 9419, 11, 353, 9707, 16451, 13, 9419, 11, 353, 248044, 9419, 11, 353, 9707, 16451, 13 };
    var expected: std.ArrayList(f32) = .empty;
    defer expected.deinit(a);
    const check = struct {
        fn append(decoder: *Decoder, values: *std.ArrayList(f32), allocator: std.mem.Allocator) !void {
            for (decoder.states) |state| for (state.parts) |part| {
                if (part) |buffer| try values.appendSlice(allocator, buffer.floats());
            };
            try values.appendSlice(allocator, decoder.ple_history.?.floats());
        }
        fn compare(actual: []const f32, reference: []const f32) !void {
            try std.testing.expectEqual(reference.len, actual.len);
            var difference: f64 = 0;
            var norm: f64 = 0;
            for (actual, reference) |x, y| {
                try std.testing.expect(std.math.isFinite(x));
                difference += (@as(f64, x) - y) * (@as(f64, x) - y);
                norm += @as(f64, y) * y;
            }
            const relative = @sqrt(difference / @max(norm, 1e-30));
            std.debug.print("Qwen4Exp chunk parity elements={d} relative_l2={e}\n", .{ actual.len, relative });
            try std.testing.expect(relative < 0.0002);
        }
    };
    var ngram_expected: model.NgramState = undefined;
    var continuation: u32 = undefined;
    {
        const decoder = try Decoder.create(a, store, plan);
        defer decoder.deinit();
        decoder.read_io = executor.io();
        for (tokens, 0..) |token, t| {
            _ = try decoder.step(@intCast(token));
            if (t == 7 or t == 11 or t == 15) try expected.appendSlice(a, decoder.logits.?.floats());
        }
        try check.append(decoder, &expected, a);
        ngram_expected = decoder.ngram_state;
        continuation = try decoder.step(9419);
        try expected.appendSlice(a, decoder.logits.?.floats());
    }
    {
        const decoder = try Decoder.create(a, store, plan);
        defer decoder.deinit();
        decoder.read_io = executor.io();
        for ([_]usize{ 0, 8, 12 }, [_]usize{ 8, 12, 16 }, 0..) |start, end, i| {
            _ = try decoder.prefillChunk(tokens[start..end], true);
            try check.compare(decoder.logits.?.floats(), expected.items[i * 248320 ..][0..248320]);
        }
        try std.testing.expectEqualDeep(ngram_expected, decoder.ngram_state);
        var actual: std.ArrayList(f32) = .empty;
        defer actual.deinit(a);
        try check.append(decoder, &actual, a);
        try check.compare(actual.items, expected.items[3 * 248320 ..][0..actual.items.len]);
        try std.testing.expectEqual(continuation, try decoder.step(9419));
        try check.compare(decoder.logits.?.floats(), expected.items[expected.items.len - 248320 ..]);
        try std.testing.expectEqual(@as(usize, 17), decoder.position);
    }
}

test "Qwen4Exp tiled prefill buffers fit the admitted scratch budget" {
    var total: usize = 24 * profile.max_context * 24 * 4 + 512 * 6144 * 4 + 32 * 512 * 4;
    for (Decoder.prefill_widths) |width| total += width * 128 * 4;
    try std.testing.expect(total < 64 * MiB);
}

test "Qwen4Exp decoder keeps the loaded model for cancellation pressure and transient reads only" {
    for ([_]anyerror{ error.RequestCancelled, error.ResourceTemporarilyUnavailable, error.InputOutput, error.ReadTimedOut, error.ReadFailed }) |err| try std.testing.expect(Decoder.recoverable(err));
    // The artifact or the GPU can no longer be trusted after these.
    for ([_]anyerror{ error.Cancelled, error.IncompleteRead, error.NotOpenForReading, error.NoDevice, error.MetalFrameSyncFailed, error.Qwen4ExpNonFiniteLogits, error.Qwen4ExpInvalidRoute, error.OutOfMemory }) |err| try std.testing.expect(!Decoder.recoverable(err));
}
