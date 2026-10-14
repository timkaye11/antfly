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

//! Dense Qwen3.8-27B, two explicit SSD layer buffers, bounded FP32 state.
const std = @import("std");
const model = @import("../models/qwen35_streamed.zig");
const gpu = @import("../backends/qwen35_streamed_kernels.zig");
const pager = @import("../runtime/dense_weight_pager.zig");
const memory = @import("../runtime/tier/memory.zig");
const profile = @import("../runtime/machine_profile.zig");
const control = @import("../util/local_control.zig");
const gguf = @import("../gguf/root.zig");
const MiB = model.MiB;
// Include page rounding and driver temporaries beyond the explicit buffers.
fn scratchBytes(context: usize, chunk: usize) usize {
    const activations = chunk * (4 * 12288 + 6144 + 3 * 5120 + 2 * 17408) * 4;
    const sequential = 17 * @max(17408, context * 24) * 4;
    const explicit = activations + sequential + gpu.tile_bytes + gpu.tile_result_bytes + gpu.quant_grid_bytes + 248320 * 4;
    return @max(256 * MiB, std.mem.alignForward(usize, explicit, MiB) + 64 * MiB);
}

pub const checkpoint_bytes = 48 * (48 * 128 * 128 + 3 * 10240) * 4;
pub const PlanRequest = struct {
    context: usize,
    chunk: usize,
    /// Reserve the recurrent-state checkpoint that prompt lookup restores from.
    lookup: bool = false,
    /// Zero derives the process envelope from the finished plan.
    cap_bytes: usize = 0,
};
pub const Plan = struct {
    lookup: bool = false,
    context: usize,
    chunk: usize,
    cap: usize,
    cap_source: profile.Source,
    host: profile.Host,
    amounts: memory.AdmissionAmounts,
    pub fn inspect(source: *const model.Source, host: profile.Host, request: PlanRequest) !Plan {
        if (request.chunk == 0 or request.chunk > gpu.max_chunk) return error.UnsupportedQwen35Budget;
        if (request.context == 0 or request.context > profile.max_context) return error.UnsupportedQwen35Context;
        const head = try model.byteLen(try source.tensor("output.weight"));
        const state = try model.stateBytes(request.context);
        const amounts: memory.AdmissionAmounts = .{
            .backend_weight_bytes = head + 2 * model.layer_buffer_bytes + MiB,
            .backend_kv_bytes = state + 2 * MiB,
            .backend_scratch_bytes = scratchBytes(request.context, request.chunk),
            .host_scratch_bytes = 768 * MiB + @as(usize, if (request.lookup) checkpoint_bytes else 0),
        };
        // Explicit and derived envelopes alike keep the admission reserve
        // beyond the plan; a larger explicit cap only loosens the bound.
        const resolved = profile.envelope(host, request.cap_bytes, amounts.hostTotalBytes() + amounts.backendTotalBytes(), profile.admission_reserve_bytes) catch |err| return profileError(err);
        profile.checkBackend(host, amounts.backendTotalBytes(), @max(head, model.layer_buffer_bytes)) catch |err| return profileError(err);
        return .{ .lookup = request.lookup, .context = request.context, .chunk = request.chunk, .cap = resolved.cap_bytes, .cap_source = resolved.source, .host = host, .amounts = amounts };
    }
    fn profileError(err: profile.Error) anyerror {
        return switch (err) {
            error.MemoryBudgetExceeded => error.Qwen35MemoryBudgetExceeded,
            error.MetalWorkingSetExceeded => error.Qwen35MetalWorkingSetExceeded,
            error.MetalBufferTooLarge => error.Qwen35MetalBufferTooLarge,
            else => err,
        };
    }
    pub fn total(self: Plan) usize {
        return self.amounts.hostTotalBytes() + self.amounts.backendTotalBytes();
    }
    pub fn acquire(self: Plan, controller: *memory.AdmissionController) !memory.AdmissionLease {
        if (comptime !@import("build_options").enable_metal) return error.MetalNotEnabled;
        if (self.host.metal_working_set_bytes == 0) return error.UnifiedMetalDeviceRequired;
        controller.configureProcessMemoryLimit(self.cap, .explicit);
        return controller.tryAcquire(.gpu, .{ .host_limit_bytes = self.cap, .backend_limit_bytes = self.cap, .combined_limit_bytes = self.cap }, self.amounts, true);
    }
};
const State = struct {
    first: ?gpu.Buffer = null,
    second: ?gpu.Buffer = null,
    fn deinit(self: *State) void {
        if (self.first) |*b| b.deinit();
        if (self.second) |*b| b.deinit();
    }
};
pub const Decoder = struct {
    allocator: std.mem.Allocator,
    source: *const model.Source,
    plan: Plan,
    controller: *memory.AdmissionController,
    cancelled: *const std.atomic.Value(bool),
    kernels: gpu.Kernels,
    buffers: [2]?gpu.Buffer = @splat(null),
    slots: [2]pager.Slot = undefined,
    slot_count: usize = 0,
    states: [64]State = @splat(.{}),
    scratch: [17]?gpu.Buffer = @splat(null),
    projections: [4]?gpu.Buffer = @splat(null),
    attn_values: ?gpu.Buffer = null,
    hidden: ?gpu.Buffer = null,
    normed: ?gpu.Buffer = null,
    branch: ?gpu.Buffer = null,
    gate: ?gpu.Buffer = null,
    up: ?gpu.Buffer = null,
    head: ?gpu.Buffer = null,
    output_norm: ?gpu.Buffer = null,
    logits: ?gpu.Buffer = null,
    position: usize = 0,
    current_layer: usize = 0,
    /// Terminal: the GPU or artifact can no longer be trusted.
    failed: bool = false,
    /// Sequence state is partially advanced but the head, slots and GPU
    /// pipeline are intact; reset() restores an empty sequence.
    needs_reset: bool = false,
    /// The call in progress spans several tokens (see checkInside).
    interruptible: bool = false,
    io: ?std.Io = null,
    embedding_bytes_read: u64 = 0,
    read_wait_ns: u64 = 0,
    checkpoint: ?[]u8 = null,
    predict_rows: ?[]u32 = null,
    lookup_stats: struct { attempts: usize = 0, proposed: usize = 0, accepted: usize = 0, replays: usize = 0 } = .{},
    pub fn create(a: std.mem.Allocator, source: *const model.Source, plan: Plan, controller: *memory.AdmissionController, cancelled: *const std.atomic.Value(bool)) !*Decoder {
        if (!source.verified) return error.Qwen35ArtifactNotVerified;
        const d = try a.create(Decoder);
        errdefer a.destroy(d);
        d.* = .{ .allocator = a, .source = source, .plan = plan, .controller = controller, .cancelled = cancelled, .kernels = try gpu.Kernels.init() };
        errdefer d.deinitContents();
        d.kernels.cancelled = cancelled;
        if (plan.lookup) d.checkpoint = try a.alloc(u8, checkpoint_bytes);
        for (&d.buffers, 0..) |*b, i| {
            b.* = try gpu.Buffer.init(d.kernels.core.runtime, model.layer_buffer_bytes);
            d.slots[i] = .{ .bytes = b.*.?.bytes[0..model.layer_buffer_bytes] };
            d.slot_count += 1;
        }
        for (&d.scratch) |*b| b.* = try d.zero(@max(17408, plan.context * 24));
        for (&d.projections) |*b| b.* = try d.zero(plan.chunk * 12288);
        d.attn_values = try d.zero(plan.chunk * 6144);
        d.hidden = try d.zero(plan.chunk * 5120);
        d.normed = try d.zero(plan.chunk * 5120);
        d.branch = try d.zero(plan.chunk * 5120);
        d.gate = try d.zero(plan.chunk * 17408);
        d.up = try d.zero(plan.chunk * 17408);
        d.logits = try d.zero(248320);
        for (&d.states, 0..) |*state, l| {
            state.first = try d.zero(if (model.isLinear(l)) 48 * 128 * 128 else plan.context * 1024);
            state.second = try d.zero(if (model.isLinear(l)) 3 * 10240 else plan.context * 1024);
        }
        d.head = try d.loadGlobal("output.weight");
        d.output_norm = try d.loadGlobal("output_norm.weight");
        var handles: std.ArrayList(*anyopaque) = .empty;
        defer handles.deinit(a);
        try handles.append(a, d.kernels.core.dummy.handle);
        try handles.append(a, d.kernels.tile.handle);
        try handles.append(a, d.kernels.tile_result.handle);
        for (d.buffers) |b| try handles.append(a, b.?.handle);
        for (d.scratch) |b| try handles.append(a, b.?.handle);
        for (d.projections) |b| try handles.append(a, b.?.handle);
        try handles.append(a, d.attn_values.?.handle);
        for (d.states) |state| {
            try handles.append(a, state.first.?.handle);
            try handles.append(a, state.second.?.handle);
        }
        for ([_]?gpu.Buffer{ d.hidden, d.normed, d.branch, d.gate, d.up, d.head, d.output_norm, d.logits }) |b| try handles.append(a, b.?.handle);
        try d.kernels.core.makeResident(handles.items, plan.amounts.backendTotalBytes());
        try d.check();
        return d;
    }
    fn zero(self: *Decoder, n: usize) !gpu.Buffer {
        var b = try gpu.Buffer.init(self.kernels.core.runtime, n * 4);
        errdefer b.deinit();
        // Shared allocations are host-addressable; initialization precedes all GPU use.
        @memset(b.bytes, 0);
        return b;
    }
    fn loadGlobal(self: *Decoder, name: []const u8) !gpu.Buffer {
        const t = try self.source.tensor(name);
        var b = try gpu.Buffer.init(self.kernels.core.runtime, try model.byteLen(t));
        errdefer b.deinit();
        var at: usize = 0;
        while (at < b.logical_bytes) {
            try self.check();
            const end = @min(at + 4 * MiB, b.logical_bytes);
            try self.source.reader.read(t.data_offset + at, b.bytes[at..end]);
            at = end;
        }
        return b;
    }
    /// Boundary check: process cancellation, live memory pressure and the cap.
    pub fn check(self: *Decoder) !void {
        try self.checkInsideToken();
        if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
    }
    /// A single token is never interrupted for a request cancel or a pressure
    /// blip: it completes, and its owner checks at the token boundary with
    /// the sequence still intact. Only process cancellation and a breached
    /// cap stop it.
    fn checkInsideToken(self: *Decoder) !void {
        if (self.cancelled.load(.acquire)) return error.Cancelled;
        const p = @import("antfly_platform").process_memory.pressureSnapshot();
        if (p.available and p.footprint_bytes > self.plan.cap) return error.Qwen35MemoryBudgetExceeded;
    }
    /// Inside a multi-token chunk a request cancel or memory pressure may
    /// also stop the work, at the price of the sequence.
    fn checkInside(self: *Decoder) !void {
        if (!self.interruptible) return self.checkInsideToken();
        try control.checkRequest();
        try self.check();
    }
    /// Submitted batches are drained and both slots emptied before a
    /// recoverable failure surfaces.
    pub const recoverable = control.recoverableExecutionError;
    fn noteFailure(self: *Decoder, err: anyerror, mutated: bool) void {
        if (!recoverable(err)) {
            self.failed = true;
        } else if (mutated) self.needs_reset = true;
    }
    fn ready(self: *const Decoder) !void {
        if (self.failed) return error.Qwen35StateInvalid;
        if (self.needs_reset) return error.Qwen35StateNeedsReset;
    }
    fn deinitContents(self: *Decoder) void {
        if (self.checkpoint) |bytes| self.allocator.free(bytes);
        self.kernels.core.cancelBatch();
        for (self.slots[0..self.slot_count]) |*slot| slot.drain();
        self.kernels.deinit();
        for (&self.buffers) |*b| if (b.*) |*buf| buf.deinit();
        for (&self.scratch) |*b| if (b.*) |*buf| buf.deinit();
        for (&self.states) |*state| state.deinit();
        for (&self.projections) |*b| if (b.*) |*buf| buf.deinit();
        if (self.attn_values) |*b| b.deinit();
        inline for (.{ "hidden", "normed", "branch", "gate", "up", "head", "output_norm", "logits" }) |field| if (@field(self, field)) |*b| b.deinit();
    }
    pub fn deinit(self: *Decoder) void {
        self.deinitContents();
        self.allocator.destroy(self);
    }
    pub fn bytesRead(self: *const Decoder) u64 {
        return self.slots[0].total_bytes_read + self.slots[1].total_bytes_read;
    }
    fn s(self: *Decoder, i: usize) gpu.Ref {
        return self.scratch[i].?.ref();
    }
    fn part(b: *const gpu.Buffer, row: usize, width: usize) !gpu.Ref {
        return b.ref().slice(row * width * 4, width * 4);
    }
    fn weight(self: *Decoder, l: usize, name: []const u8) !gpu.Ref {
        const t = try self.source.layerTensor(l, name);
        const slot = &self.slots[l % 2];
        if (slot.state != .gpu or slot.layer != l or t.data_offset < slot.span.offset) return error.InvalidStreamedSlotState;
        return self.buffers[l % 2].?.ref().slice(@intCast(t.data_offset - slot.span.offset), try model.byteLen(t));
    }
    /// Encodes only; cancellation and the memory cap are checked per layer
    /// and every eight positions, not per projection.
    fn mm(self: *Decoder, l: usize, name: []const u8, rows: usize, input: gpu.Ref, out: gpu.Ref) !void {
        const t = try self.source.layerTensor(l, name);
        const w = try self.weight(l, name);
        const raw = self.buffers[l % 2].?.bytes[w.offset..][0..w.bytes];
        try self.kernels.matmul(t.tensor_type, raw, w, @intCast(t.dimensions[0]), @intCast(t.dimensions[1]), rows, input, out);
    }
    fn run(self: *Decoder, p: gpu.Params, ins: []const gpu.Ref, out: gpu.Ref) !void {
        try self.kernels.run(p, ins, out);
    }
    fn norm(self: *Decoder, x: gpu.Ref, w: gpu.Ref, out: gpu.Ref, n: usize, dim: usize, mode: u32) !void {
        try self.run(.{ .op = 2, .n = @intCast(n / dim * 32), .dim = @intCast(dim), .mode = mode, .epsilon = 1e-6 }, &.{ x, w }, out);
    }
    fn linear(self: *Decoder, l: usize, row: usize) !void {
        const mixed = try part(&self.projections[0].?, row, 10240);
        const gate = try part(&self.projections[1].?, row, 6144);
        const beta = try part(&self.projections[2].?, row, 48);
        const alpha = try part(&self.projections[3].?, row, 48);
        try self.run(.{ .op = 9, .n = 10240, .stride = 1 }, &.{ mixed, try self.weight(l, "ssm_conv1d.weight"), self.states[l].second.?.ref() }, self.s(4));
        try self.run(.{ .op = 1, .n = 10240, .dim = 10240 }, &.{self.s(4)}, self.s(5));
        try self.norm(self.s(4), self.s(4), self.s(5), 4096, 128, 2);
        try self.run(.{ .op = 10, .n = 6144 }, &.{ self.s(5), beta, alpha, try self.weight(l, "ssm_a"), try self.weight(l, "ssm_dt.bias"), self.states[l].first.?.ref() }, self.s(6));
        try self.norm(self.s(6), try self.weight(l, "ssm_norm.weight"), self.s(7), 6144, 128, 1);
        try self.run(.{ .op = 11, .n = 6144 }, &.{ self.s(7), gate }, try part(&self.attn_values.?, row, 6144));
    }
    fn attention(self: *Decoder, l: usize, row: usize, pos: usize) !void {
        const p: u32 = @intCast(pos);
        try self.run(.{ .op = 18, .n = 6144 }, &.{ try part(&self.projections[0].?, row, 12288), self.s(1) }, self.s(2));
        try self.norm(self.s(2), try self.weight(l, "attn_q_norm.weight"), self.s(3), 6144, 256, 1);
        try self.run(.{ .op = 19, .n = 6144, .dim = 256, .position = p }, &.{self.s(3)}, self.s(4));
        try self.norm(try part(&self.projections[1].?, row, 1024), try self.weight(l, "attn_k_norm.weight"), self.s(6), 1024, 256, 1);
        try self.run(.{ .op = 19, .n = 1024, .dim = 256, .position = p }, &.{self.s(6)}, self.s(5));
        try self.run(.{ .op = 20, .n = 1024, .dim = 1024, .position = p }, &.{self.s(5)}, self.states[l].first.?.ref());
        try self.run(.{ .op = 20, .n = 1024, .dim = 1024, .position = p }, &.{try part(&self.projections[2].?, row, 1024)}, self.states[l].second.?.ref());
        try self.run(.{ .op = 24, .n = 24 * (p + 1), .position = p }, &.{ self.s(4), self.states[l].first.?.ref() }, self.s(6));
        try self.run(.{ .op = 25, .n = 24, .position = p }, &.{self.s(6)}, self.s(7));
        try self.run(.{ .op = 26, .n = 6144, .position = p }, &.{ self.s(7), self.states[l].second.?.ref(), self.s(1) }, try part(&self.attn_values.?, row, 6144));
    }
    fn capture(self: *Decoder, name: []const u8, bytes: []const u8) !void {
        if (std.c.getenv("ANTFLY_QWEN35_CAPTURE_LOGITS_ONLY") != null and !std.mem.startsWith(u8, name, "logits-")) return;
        if (std.c.getenv("ANTFLY_QWEN35_CAPTURE_DIR")) |raw| {
            const path = try std.fs.path.join(self.allocator, &.{ std.mem.span(raw), name });
            defer self.allocator.free(path);
            try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{ .sub_path = path, .data = bytes });
        }
    }
    pub const Advance = struct { next: u32, accepted: [4]i32 = @splat(0), count: usize = 0 };
    /// The last history token is pending; all earlier tokens are already consumed.
    /// Attention writes after the restored position are unreachable. Only recurrent
    /// and convolution state needs a checkpoint; a mismatch replays accepted input.
    pub fn advancePromptLookup(self: *Decoder, history: []const i32, limit: usize, eos: []const u32) !Advance {
        if (!self.plan.lookup or history.len == 0 or limit > 4 or self.position >= self.plan.context) return error.InvalidPromptLookup;
        var inputs: [5]i32 = undefined;
        inputs[0] = history[history.len - 1];
        var count = draftLookup(history, inputs[1..][0..@min(@min(limit, self.plan.chunk - 1), self.plan.context - self.position - 1)]);
        for (inputs[1..][0..count], 0..) |id, i| if (std.mem.indexOfScalar(u32, eos, @intCast(id)) != null) {
            count = i;
            break;
        };
        if (count == 0) return .{ .next = try self.prefill(inputs[0..1]) };
        return self.verifyDraft(inputs[0 .. count + 1]);
    }
    pub fn verifyDraft(self: *Decoder, inputs: []const i32) !Advance {
        try self.ready();
        if (!self.plan.lookup or self.position > self.plan.context or inputs.len < 2 or inputs.len > 5 or inputs.len > self.plan.chunk or inputs.len > self.plan.context - self.position) return error.InvalidPromptLookup;
        const count = inputs.len - 1;
        for (inputs) |token| if (token < 0 or token >= 248320) return error.InvalidQwen35Token;
        const old_position = self.position;
        var offset: usize = 0;
        for (self.states, 0..) |state, l| if (model.isLinear(l)) {
            for ([_]?gpu.Buffer{ state.first, state.second }) |buffer| {
                const b = buffer.?;
                @memcpy(self.checkpoint.?[offset..][0..b.logical_bytes], b.bytes[0..b.logical_bytes]);
                offset += b.logical_bytes;
            }
        };
        std.debug.assert(offset == checkpoint_bytes);
        var predictions: [5]u32 = undefined;
        self.predict_rows = predictions[0 .. count + 1];
        defer self.predict_rows = null;
        _ = try self.prefill(inputs[0 .. count + 1]);
        self.predict_rows = null;
        var accepted: usize = 0;
        while (accepted < count and predictions[accepted] == @as(u32, @intCast(inputs[accepted + 1]))) : (accepted += 1) {}
        self.lookup_stats.attempts += 1;
        self.lookup_stats.proposed += count;
        self.lookup_stats.accepted += accepted;
        var next = predictions[accepted];
        if (accepted < count) {
            offset = 0;
            for (self.states, 0..) |state, l| if (model.isLinear(l)) {
                for ([_]?gpu.Buffer{ state.first, state.second }) |buffer| {
                    const b = buffer.?;
                    @memcpy(b.bytes[0..b.logical_bytes], self.checkpoint.?[offset..][0..b.logical_bytes]);
                    offset += b.logical_bytes;
                }
            };
            self.position = old_position;
            next = try self.prefill(inputs[0 .. accepted + 1]);
            self.lookup_stats.replays += 1;
        }
        var result: Advance = .{ .next = next, .count = accepted };
        @memcpy(result.accepted[0..accepted], inputs[1..][0..accepted]);
        return result;
    }
    /// Start an empty sequence. This also clears a recoverable failure that
    /// stopped inside a chunk; a terminal failure still requires a new decoder.
    pub fn reset(self: *Decoder) !void {
        if (self.failed) return error.Qwen35StateInvalid;
        try self.kernels.endBatch();
        for (&self.slots) |*slot| slot.drain();
        for (self.states) |state| {
            if (state.first) |b| @memset(b.bytes, 0);
            if (state.second) |b| @memset(b.bytes, 0);
        }
        self.position = 0;
        self.current_layer = 0;
        self.needs_reset = false;
    }
    pub fn prefill(self: *Decoder, tokens: []const i32) !u32 {
        try self.ready();
        if (tokens.len == 0 or tokens.len > self.plan.chunk or tokens.len > self.plan.context - self.position) return error.UnsupportedQwen35Context;
        for (tokens) |t| if (t < 0 or t >= 248320) return error.InvalidQwen35Token;
        var mutated = false;
        return self.prefillUnguarded(tokens, &mutated) catch |err| {
            self.noteFailure(err, mutated);
            self.kernels.core.cancelBatch();
            for (&self.slots) |*slot| slot.drain();
            return err;
        };
    }
    fn prefillUnguarded(self: *Decoder, tokens: []const i32, mutated: *bool) !u32 {
        // Until the first layer executes, only scratch and a layer slot are
        // written: a pressure blip, a cancellation or an unreadable range
        // leaves the sequence exactly where the previous call left it.
        self.interruptible = tokens.len > 1;
        try self.check();
        const embedding = try self.source.tensor("token_embd.weight");
        const row_bytes = (try model.byteLen(embedding)) / 248320;
        var encoded: [8192]u8 = undefined;
        if (row_bytes > encoded.len) return error.InvalidQwen35Embedding;
        for (tokens, 0..) |t, i| {
            try self.source.reader.read(embedding.data_offset + @as(u64, @intCast(t)) * row_bytes, encoded[0..row_bytes]);
            try gguf.quant_codec.dequantizeRow(embedding.tensor_type, encoded[0..row_bytes], 5120, 0, self.hidden.?.floats()[i * 5120 ..][0..5120]);
            self.embedding_bytes_read += row_bytes;
        }
        try self.slots[0].begin(&self.source.reader, 0, self.source.layers[0], self.io, self.cancelled);
        for (0..64) |l| {
            self.current_layer = l;
            try self.checkInside();
            var slot = &self.slots[l % 2];
            const read_tick = if (self.kernels.profile_enabled) gpu.Profile.now() else 0;
            try slot.finish();
            if (self.kernels.profile_enabled) self.read_wait_ns += gpu.Profile.now() - read_tick;
            var lease = try slot.acquire();
            if (l + 1 < 64) try self.slots[(l + 1) % 2].begin(&self.source.reader, l + 1, self.source.layers[l + 1], self.io, self.cancelled);
            try self.kernels.core.beginBatch();
            try self.norm(self.hidden.?.ref(), try self.weight(l, "attn_norm.weight"), self.normed.?.ref(), tokens.len * 5120, 5120, 1);
            if (model.isLinear(l)) {
                try self.mm(l, "attn_qkv.weight", tokens.len, self.normed.?.ref(), self.projections[0].?.ref());
                try self.mm(l, "attn_gate.weight", tokens.len, self.normed.?.ref(), self.projections[1].?.ref());
                try self.mm(l, "ssm_beta.weight", tokens.len, self.normed.?.ref(), self.projections[2].?.ref());
                try self.mm(l, "ssm_alpha.weight", tokens.len, self.normed.?.ref(), self.projections[3].?.ref());
            } else {
                try self.mm(l, "attn_q.weight", tokens.len, self.normed.?.ref(), self.projections[0].?.ref());
                try self.mm(l, "attn_k.weight", tokens.len, self.normed.?.ref(), self.projections[1].?.ref());
                try self.mm(l, "attn_v.weight", tokens.len, self.normed.?.ref(), self.projections[2].?.ref());
            }
            // Recurrent and KV updates are encoded from here on.
            mutated.* = true;
            for (0..tokens.len) |i| {
                if (model.isLinear(l)) try self.linear(l, i) else try self.attention(l, i, self.position + i);
                // Bound command queue size while retaining one layer load per chunk.
                if ((i + 1) % 8 == 0) {
                    try self.kernels.endBatch();
                    try self.checkInside();
                    try self.kernels.core.beginBatch();
                }
            }
            try self.mm(l, if (model.isLinear(l)) "ssm_out.weight" else "attn_output.weight", tokens.len, self.attn_values.?.ref(), self.branch.?.ref());
            try self.run(.{ .op = 16, .n = @intCast(tokens.len * 5120) }, &.{ self.hidden.?.ref(), self.branch.?.ref() }, self.hidden.?.ref());
            try self.norm(self.hidden.?.ref(), try self.weight(l, "post_attention_norm.weight"), self.normed.?.ref(), tokens.len * 5120, 5120, 1);
            try self.mm(l, "ffn_gate.weight", tokens.len, self.normed.?.ref(), self.gate.?.ref());
            try self.mm(l, "ffn_up.weight", tokens.len, self.normed.?.ref(), self.up.?.ref());
            try self.run(.{ .op = 12, .n = @intCast(tokens.len * 17408) }, &.{ self.gate.?.ref(), self.up.?.ref() }, self.gate.?.ref());
            try self.mm(l, "ffn_down.weight", tokens.len, self.gate.?.ref(), self.branch.?.ref());
            try self.run(.{ .op = 16, .n = @intCast(tokens.len * 5120) }, &.{ self.hidden.?.ref(), self.branch.?.ref() }, self.hidden.?.ref());
            try self.kernels.endBatch();
            try lease.releaseAfterCompletion();
            var name: [64]u8 = undefined;
            try self.capture(try std.fmt.bufPrint(&name, "layer-{d:0>2}-pos-{d}.f32", .{ l, self.position + tokens.len }), self.hidden.?.bytes[0 .. tokens.len * 5120 * 4]);
            if (std.c.getenv("ANTFLY_QWEN35_CAPTURE_STATE") != null) {
                try self.capture(try std.fmt.bufPrint(&name, "state-{d:0>2}-pos-{d}.f32", .{ l, self.position + tokens.len }), self.states[l].first.?.bytes[0..self.states[l].first.?.logical_bytes]);
                try self.capture(try std.fmt.bufPrint(&name, "aux-{d:0>2}-pos-{d}.f32", .{ l, self.position + tokens.len }), self.states[l].second.?.bytes[0..self.states[l].second.?.logical_bytes]);
            }
        }
        const head_rows = if (self.predict_rows != null) tokens.len else 1;
        for (0..head_rows) |row| {
            try self.kernels.core.beginBatch();
            try self.norm(try part(&self.hidden.?, if (self.predict_rows != null) row else tokens.len - 1, 5120), self.output_norm.?.ref(), self.s(0), 5120, 5120, 1);
            const t = try self.source.tensor("output.weight");
            try self.kernels.matmul(t.tensor_type, self.head.?.bytes[0..self.head.?.logical_bytes], self.head.?.ref(), 5120, 248320, 1, self.s(0), self.logits.?.ref());
            try self.run(.{ .op = 27, .n = 32, .dim = 248320 }, &.{self.logits.?.ref()}, self.s(1));
            try self.kernels.endBatch();
            const prediction = std.mem.readInt(u32, self.scratch[1].?.bytes[0..4], .little);
            if (prediction >= 248320) return error.Qwen35NonFiniteLogits;
            if (self.predict_rows) |predictions| predictions[row] = prediction;
        }
        var capture_name: [64]u8 = undefined;
        try self.capture(try std.fmt.bufPrint(&capture_name, "logits-pos-{d}.f32", .{self.position + tokens.len}), self.logits.?.bytes[0 .. 248320 * 4]);
        const token = std.mem.readInt(u32, self.scratch[1].?.bytes[0..4], .little);
        if (token >= 248320) return error.Qwen35NonFiniteLogits;
        self.position += tokens.len;
        // The call is complete and consistent. Pressure is the next call's
        // boundary check, so this result still reaches the caller.
        try self.checkInsideToken();
        return token;
    }
};

test "Qwen35 largest prefill scratch includes every activation and staging buffer" {
    try std.testing.expect(scratchBytes(4096, gpu.max_chunk) >= 519 * MiB);
    try std.testing.expect(scratchBytes(4096, gpu.max_chunk) <= 528 * MiB);
    try std.testing.expectEqual(256 * MiB, scratchBytes(4096, 128));
}

/// Search only prior occurrences; never read invented continuation beyond history.
pub fn draftLookup(history: []const i32, output: []i32) usize {
    if (history.len < 3 or output.len == 0) return 0;
    var width = @min(4, history.len - 1);
    while (width >= 2) : (width -= 1) {
        var best: usize = 0;
        var end = history.len - 1;
        while (end >= width) : (end -= 1) {
            if (std.mem.eql(i32, history[end - width .. end], history[history.len - width ..])) {
                const count = @min(output.len, history.len - end);
                if (count > best) {
                    @memcpy(output[0..count], history[end..][0..count]);
                    best = count;
                }
                if (best == output.len) return best;
            }
        }
        if (best > 0) return best;
    }
    return 0;
}
test "Qwen35 prompt lookup uses prior matching suffixes and bounded continuations" {
    var out: [4]i32 = undefined;
    try std.testing.expectEqual(@as(usize, 2), draftLookup(&.{ 1, 2, 3, 4, 1, 2 }, out[0..2]));
    try std.testing.expectEqualSlices(i32, &.{ 3, 4 }, out[0..2]);
    try std.testing.expectEqual(@as(usize, 0), draftLookup(&.{ 1, 2, 3 }, &out));
    try std.testing.expectEqual(@as(usize, 1), draftLookup(&.{ 8, 8, 8, 8 }, &out));
}

test "Qwen35 installed draft verification restores rejected recurrence and preserves accepted continuation" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const path = std.c.getenv("ANTFLY_QWEN35_VERIFY_DRAFT") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    var source = try model.Source.open(a, std.mem.span(path));
    defer source.deinit();
    const plan = try Plan.inspect(&source, try profile.Host.detect(), .{ .context = 32, .chunk = 512, .lookup = true, .cap_bytes = 4096 * MiB });
    var controller: memory.AdmissionController = .{};
    var admission = try plan.acquire(&controller);
    defer admission.release();
    var cancelled: std.atomic.Value(bool) = .init(false);
    try source.verify(std.mem.span(path), .full, null, &cancelled);
    var executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(1), .concurrent_limit = .limited(1), .stack_size = 256 * 1024 });
    defer executor.deinit();
    const decoder = try Decoder.create(a, &source, plan, &controller, &cancelled);
    defer decoder.deinit();
    decoder.io = executor.io();
    const prompt = [_]i32{ 760, 6511, 314, 9338, 369 };
    const paris = try decoder.prefill(&prompt);
    const period = try decoder.prefill(&.{@intCast(paris)});
    const reference = try a.dupe(f32, decoder.logits.?.floats());
    defer a.free(reference);
    try decoder.reset();
    try std.testing.expectEqual(paris, try decoder.prefill(&prompt));
    const rejected = try decoder.verifyDraft(&.{ @intCast(paris), @intCast((period + 1) % 248320) });
    try std.testing.expectEqual(@as(usize, 0), rejected.count);
    try std.testing.expectEqual(period, rejected.next);
    try std.testing.expectEqual(@as(usize, 6), decoder.position);
    for (reference, decoder.logits.?.floats()) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);
    const after_period = try decoder.prefill(&.{@intCast(period)});
    @memcpy(reference, decoder.logits.?.floats());
    try decoder.reset();
    _ = try decoder.prefill(&prompt);
    const accepted = try decoder.verifyDraft(&.{ @intCast(paris), @intCast(period) });
    try std.testing.expectEqual(@as(usize, 1), accepted.count);
    try std.testing.expectEqual(after_period, accepted.next);
    try std.testing.expectEqual(@as(usize, 7), decoder.position);
    for (reference, decoder.logits.?.floats()) |want, got| try std.testing.expectApproxEqAbs(want, got, 0.001);
    try std.testing.expectEqual(@as(usize, 1), decoder.lookup_stats.replays);
    try std.testing.expectEqual(@as(usize, 1), decoder.lookup_stats.accepted);
}

test "Qwen35 streamed plan derives its envelope and accepts caps above the old 4 GiB limit" {
    // Only the head size comes from the artifact; a synthetic source keeps
    // this a CPU-only contract test.
    var dimensions = [_]u64{ 5120, 248320 };
    var tensors = [_]gguf.format.TensorInfo{.{ .name = "output.weight", .dimensions = &dimensions, .tensor_type = .{ .known = .Q6_K }, .offset = 0, .data_offset = 0 }};
    var source: model.Source = undefined;
    source.file.tensors = &tensors;
    source.index = .empty;
    defer source.index.deinit(std.testing.allocator);
    try source.index.put(std.testing.allocator, "output.weight", 0);
    const air: profile.Host = .{ .physical_bytes = 16 * 1024 * MiB, .metal_working_set_bytes = 11453251584 };
    const derived = try Plan.inspect(&source, air, .{ .context = 4096, .chunk = 512 });
    try std.testing.expectEqual(profile.Source.derived, derived.cap_source);
    try std.testing.expectEqual(derived.total() + profile.derived_headroom_bytes, derived.cap);
    try std.testing.expect(derived.cap <= 4096 * MiB);
    const lookup = try Plan.inspect(&source, air, .{ .context = 4096, .chunk = 512, .lookup = true });
    try std.testing.expectEqual(derived.cap + checkpoint_bytes, lookup.cap);
    // The historical explicit cap still admits; a tighter one still fails closed.
    const explicit = try Plan.inspect(&source, air, .{ .context = 4096, .chunk = 512, .cap_bytes = 4096 * MiB });
    try std.testing.expectEqual(@as(usize, 4096 * MiB), explicit.cap);
    try std.testing.expectEqual(profile.Source.explicit, explicit.cap_source);
    try std.testing.expectError(error.Qwen35MemoryBudgetExceeded, Plan.inspect(&source, air, .{ .context = 4096, .chunk = 512, .cap_bytes = 2048 * MiB }));
    // A larger host may raise the cap; nothing clamps or rejects it at 4 GiB.
    const studio: profile.Host = .{ .physical_bytes = 48 * 1024 * MiB, .metal_working_set_bytes = 36 * 1024 * MiB };
    try std.testing.expectEqual(@as(usize, 20 * 1024 * MiB), (try Plan.inspect(&source, studio, .{ .context = 4096, .chunk = 1024, .cap_bytes = 20 * 1024 * MiB })).cap);
    try std.testing.expectError(error.ProcessCapExceedsPhysicalMemory, Plan.inspect(&source, air, .{ .context = 4096, .chunk = 512, .cap_bytes = 20 * 1024 * MiB }));
    try std.testing.expectError(error.UnsupportedQwen35Context, Plan.inspect(&source, air, .{ .context = profile.max_context + 1, .chunk = 512 }));
    try std.testing.expectError(error.UnsupportedQwen35Budget, Plan.inspect(&source, air, .{ .context = 4096, .chunk = gpu.max_chunk + 1 }));
}

test "Qwen35 decoder keeps the loaded model for cancellation pressure and transient reads only" {
    for ([_]anyerror{ error.RequestCancelled, error.ResourceTemporarilyUnavailable, error.InputOutput, error.ReadTimedOut, error.ReadFailed }) |err| try std.testing.expect(Decoder.recoverable(err));
    for ([_]anyerror{ error.Cancelled, error.IncompleteRead, error.NotOpenForReading, error.NoDevice, error.Qwen35MemoryBudgetExceeded, error.Qwen35NonFiniteLogits, error.StaleStreamedLease, error.OutOfMemory }) |err| try std.testing.expect(!Decoder.recoverable(err));
}
