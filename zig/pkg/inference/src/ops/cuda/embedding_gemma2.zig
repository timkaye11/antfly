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
const artifact = @import("artifact.zig");
const driver = @import("driver.zig");
const Context = @import("context.zig").CudaContext;
const Buffer = @import("buffer.zig").DeviceBuffer;

pub const Attention = struct {
    batch: u32,
    seq: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    /// Zero selects global attention. A nonzero value is an inclusive,
    /// symmetric radius, so radius=512 permits abs(query-key) <= 512.
    local_radius: u32,

    pub fn validate(self: Attention) !void {
        if (self.batch == 0 or self.seq == 0 or self.seq > 8192) return error.InvalidEmbeddingGemma2Shape;
        if (self.q_heads != 4 or (self.kv_heads != 1 and self.kv_heads != 2)) return error.InvalidEmbeddingGemma2Shape;
        if ((self.kv_heads == 1 and self.head_dim != 512) or (self.kv_heads == 2 and self.head_dim != 256)) return error.InvalidEmbeddingGemma2Shape;
        if (self.q_heads % self.kv_heads != 0) return error.InvalidEmbeddingGemma2Shape;
    }
};

pub const Stats = struct {
    launches: u64 = 0,
    cublaslt_matmuls: u64 = 0,
    attention_launches: u64 = 0,
    attention_scalar_fallbacks: u64 = 0,
    scalar_matmul_fallbacks: u64 = 0,
    output_bytes: u64 = 0,
};

pub const Module = struct {
    module: driver.CUmodule,
    lookup: driver.CUfunction,
    audio_attention: driver.CUfunction,
    audio_clamp: driver.CUfunction,
    audio_glu_rows: driver.CUfunction,
    audio_depthwise_conv: driver.CUfunction,
    rms_norm: driver.CUfunction,
    rope: driver.CUfunction,
    attention: driver.CUfunction,
    local_flash_attention: driver.CUfunction,
    attention_softmax: driver.CUfunction,
    matmul: driver.CUfunction,
    gelu_mul: driver.CUfunction,
    bf16_to_f32: driver.CUfunction,
    scale: driver.CUfunction,
    scale_device: driver.CUfunction,
    gelu_ple: driver.CUfunction,
    residual: driver.CUfunction,
    mean_l2: driver.CUfunction,
    stats_: Stats = .{},

    pub fn init(ctx: *Context) !Module {
        // BF16 is a hard model contract. Refuse pre-Ampere devices rather
        // than silently executing an FP16 or host fallback.
        if (ctx.info.compute_major < 8) return error.CudaBf16Unsupported;
        try ctx.makeCurrent();
        var m: driver.CUmodule = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleLoadDataEx(&m, artifact.image.ptr, 0, null, null));
        errdefer _ = ctx.driver.fns.cuModuleUnload(m);
        return .{
            .module = m,
            .lookup = try function(ctx, m, "termite_embedding_gemma2_lookup_bf16"),
            .audio_attention = try function(ctx, m, "termite_gemma4_audio_local_attention_f32"),
            .audio_clamp = try function(ctx, m, "termite_gemma4_audio_clamp_f32"),
            .audio_glu_rows = try function(ctx, m, "termite_gemma4_audio_glu_rows_f32"),
            .audio_depthwise_conv = try function(ctx, m, "termite_gemma4_audio_depthwise_causal_conv1d_f32"),
            .rms_norm = try function(ctx, m, "termite_embedding_gemma2_rms_norm_bf16"),
            .rope = try function(ctx, m, "termite_embedding_gemma2_rope_bf16"),
            .attention = try function(ctx, m, "termite_embedding_gemma2_attention_bf16"),
            .local_flash_attention = try function(ctx, m, "termite_embedding_gemma2_attention_local_flash_bf16"),
            .attention_softmax = try function(ctx, m, "termite_embedding_gemma2_attention_softmax_bf16"),
            .matmul = try function(ctx, m, "termite_embedding_gemma2_matmul_bf16"),
            .gelu_mul = try function(ctx, m, "termite_embedding_gemma2_gelu_mul_bf16"),
            .bf16_to_f32 = try function(ctx, m, "termite_embedding_gemma2_bf16_to_f32"),
            .scale = try function(ctx, m, "termite_embedding_gemma2_scale_bf16"),
            .scale_device = try function(ctx, m, "termite_embedding_gemma2_scale_device_bf16"),
            .gelu_ple = try function(ctx, m, "termite_embedding_gemma2_gelu_ple_bf16"),
            .residual = try function(ctx, m, "termite_embedding_gemma2_residual_bf16"),
            .mean_l2 = try function(ctx, m, "termite_embedding_gemma2_mean_l2_f32"),
        };
    }

    pub fn deinit(self: *Module, ctx: *Context) void {
        ctx.makeCurrent() catch {};
        if (self.module != null) _ = ctx.driver.fns.cuModuleUnload(self.module);
        self.* = undefined;
    }

    pub fn stats(self: *const Module) Stats {
        return self.stats_;
    }

    pub fn noteCublasLtMatmul(self: *Module) void {
        self.stats_.cublaslt_matmuls += 1;
    }
    pub fn noteAttentionFallback(self: *Module) void {
        self.stats_.attention_scalar_fallbacks += 1;
    }
    pub fn noteAttentionLaunch(self: *Module) void {
        self.stats_.attention_launches += 1;
    }

    pub fn launchAudioAttention(self: *Module, ctx: *Context, out: Buffer, q: Buffer, k: Buffer, v: Buffer, rel: Buffer, scales: Buffer, valid: Buffer, p: @import("../ops.zig").Gemma4AudioLocalAttentionParams) !void {
        if (p.rows == 0 or p.rows > std.math.maxInt(i32) or p.hidden == 0 or p.heads == 0 or p.heads > 65535 or p.head_dim == 0 or
            p.chunk == 0 or p.chunk > std.math.maxInt(i32) or p.context == 0 or p.context_left < 2 or p.context_left > p.context or p.context > 24 or
            !std.math.isFinite(p.k_scale) or !std.math.isFinite(p.logit_cap) or p.logit_cap <= 0 or !std.math.isFinite(p.invalid_value))
            return error.InvalidEmbeddingGemma2Shape;
        const derived_hidden = std.math.mul(usize, p.heads, p.head_dim) catch return error.InvalidEmbeddingGemma2Shape;
        if (derived_hidden != p.hidden) return error.InvalidEmbeddingGemma2Shape;
        const count = try std.math.mul(usize, p.rows, p.hidden);
        const matrix_bytes = try std.math.mul(usize, count, @sizeOf(f32));
        const relative_count = try std.math.mul(usize, p.context_left, p.hidden);
        const relative_bytes = try std.math.mul(usize, relative_count, @sizeOf(f32));
        const scale_bytes = try std.math.mul(usize, p.head_dim, @sizeOf(f32));
        const valid_bytes = try std.math.mul(usize, p.rows, @sizeOf(f32));
        try require(out, matrix_bytes);
        try require(q, matrix_bytes);
        try require(k, matrix_bytes);
        try require(v, matrix_bytes);
        try require(rel, relative_bytes);
        try require(scales, scale_bytes);
        try require(valid, valid_bytes);
        var op = out.ptr;
        var qp = q.ptr;
        var kp = k.ptr;
        var vp = v.ptr;
        var rp = rel.ptr;
        var sp = scales.ptr;
        var mp = valid.ptr;
        if (p.rows > std.math.maxInt(u32) or p.hidden > std.math.maxInt(u32) or p.head_dim > std.math.maxInt(u32) or p.chunk > std.math.maxInt(u32)) return error.InvalidEmbeddingGemma2Shape;
        var rows: u32 = @intCast(p.rows);
        var hidden: u32 = @intCast(p.hidden);
        var heads: u32 = @intCast(p.heads);
        var hd: u32 = @intCast(p.head_dim);
        var chunk: u32 = @intCast(p.chunk);
        var left: u32 = @intCast(p.context_left);
        var context: u32 = @intCast(p.context);
        var ks = p.k_scale;
        var cap = p.logit_cap;
        var invalid = p.invalid_value;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&qp), @ptrCast(&kp), @ptrCast(&vp), @ptrCast(&rp), @ptrCast(&sp), @ptrCast(&mp), @ptrCast(&rows), @ptrCast(&hidden), @ptrCast(&heads), @ptrCast(&hd), @ptrCast(&chunk), @ptrCast(&left), @ptrCast(&context), @ptrCast(&ks), @ptrCast(&cap), @ptrCast(&invalid) };
        try launch(ctx, self.audio_attention, &args, .{ rows, heads, 1 }, .{ 32, 1, 1 });
        self.stats_.launches += 1;
        self.stats_.attention_launches += 1;
    }

    pub fn launchAudioClamp(self: *Module, ctx: *Context, out: Buffer, input: Buffer, count: u32, min_value: ?f32, max_value: ?f32) !void {
        if (count == 0 or (min_value == null and max_value == null)) return error.InvalidEmbeddingGemma2Shape;
        if (min_value) |value| if (!std.math.isFinite(value)) return error.InvalidEmbeddingGemma2Shape;
        if (max_value) |value| if (!std.math.isFinite(value)) return error.InvalidEmbeddingGemma2Shape;
        if (min_value != null and max_value != null and min_value.? > max_value.?) return error.InvalidEmbeddingGemma2Shape;
        const bytes = try std.math.mul(usize, count, @sizeOf(f32));
        try require(out, bytes);
        try require(input, bytes);
        var op = out.ptr;
        var ip = input.ptr;
        var n = count;
        var lower = min_value orelse 0;
        var upper = max_value orelse 0;
        var has_lower: u32 = @intFromBool(min_value != null);
        var has_upper: u32 = @intFromBool(max_value != null);
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&n), @ptrCast(&lower), @ptrCast(&upper), @ptrCast(&has_lower), @ptrCast(&has_upper) };
        try self.launch1d(ctx, self.audio_clamp, &args, count);
    }

    pub fn launchAudioGluRows(self: *Module, ctx: *Context, out: Buffer, input: Buffer, rows: u32, dim: u32) !void {
        if (rows == 0 or dim == 0) return error.InvalidEmbeddingGemma2Shape;
        const count = try std.math.mul(usize, rows, dim);
        if (count > std.math.maxInt(u32)) return error.InvalidEmbeddingGemma2Shape;
        try require(out, try std.math.mul(usize, count, @sizeOf(f32)));
        try require(input, try std.math.mul(usize, count, 2 * @sizeOf(f32)));
        var op = out.ptr;
        var ip = input.ptr;
        var r = rows;
        var d = dim;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&r), @ptrCast(&d) };
        try self.launch1d(ctx, self.audio_glu_rows, &args, count);
    }

    pub fn launchAudioDepthwiseConv(self: *Module, ctx: *Context, out: Buffer, input: Buffer, weight: Buffer, rows: u32, dim: u32, kernel_size: u32) !void {
        if (rows == 0 or dim == 0 or kernel_size == 0) return error.InvalidEmbeddingGemma2Shape;
        const count = try std.math.mul(usize, rows, dim);
        if (count > std.math.maxInt(u32)) return error.InvalidEmbeddingGemma2Shape;
        try require(out, try std.math.mul(usize, count, @sizeOf(f32)));
        try require(input, try std.math.mul(usize, count, @sizeOf(f32)));
        const weight_count = try std.math.mul(usize, kernel_size, dim);
        try require(weight, try std.math.mul(usize, weight_count, @sizeOf(f32)));
        var op = out.ptr;
        var ip = input.ptr;
        var wp = weight.ptr;
        var r = rows;
        var d = dim;
        var k = kernel_size;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&wp), @ptrCast(&r), @ptrCast(&d), @ptrCast(&k) };
        try self.launch1d(ctx, self.audio_depthwise_conv, &args, count);
    }

    pub fn launchLookup(self: *Module, ctx: *Context, out: Buffer, table: Buffer, ids: Buffer, rows: u32, vocab: u32, hidden: u32, scale: f32) !void {
        const count = try std.math.mul(usize, rows, hidden);
        try require(out, count * 2);
        try require(table, @as(usize, vocab) * hidden * 2);
        try require(ids, @as(usize, rows) * 8);
        var op = out.ptr;
        var tp = table.ptr;
        var ip = ids.ptr;
        var r = rows;
        var v = vocab;
        var h = hidden;
        var s = scale;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&tp), @ptrCast(&ip), @ptrCast(&r), @ptrCast(&v), @ptrCast(&h), @ptrCast(&s) };
        try self.launch1d(ctx, self.lookup, &args, count);
    }

    pub fn launchRmsNorm(self: *Module, ctx: *Context, out: Buffer, input: Buffer, weight: ?Buffer, rows: u32, width: u32, eps: f32) !void {
        const bytes = @as(usize, rows) * width * 2;
        try require(out, bytes);
        try require(input, bytes);
        if (weight) |wbuf| try require(wbuf, @as(usize, width) * 2);
        var op = out.ptr;
        var ip = input.ptr;
        var wp = if (weight) |wbuf| wbuf.ptr else 0;
        var r = rows;
        var w = width;
        var e = eps;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&wp), @ptrCast(&r), @ptrCast(&w), @ptrCast(&e) };
        try launch(ctx, self.rms_norm, &args, .{ rows, 1, 1 }, .{ 256, 1, 1 });
        self.stats_.launches += 1;
    }

    pub fn launchRope(self: *Module, ctx: *Context, q: Buffer, k: Buffer, mask: Buffer, shape: Attention, theta: f32, rope_scale: f32) !void {
        try shape.validate();
        try require(q, @as(usize, shape.batch) * shape.seq * shape.q_heads * shape.head_dim * 2);
        try require(k, @as(usize, shape.batch) * shape.seq * shape.kv_heads * shape.head_dim * 2);
        if (mask.ptr != 0) try require(mask, @as(usize, shape.batch) * shape.seq * 8);
        var qp = q.ptr;
        var kp = k.ptr;
        var mp = mask.ptr;
        var b = shape.batch;
        var s = shape.seq;
        var qh = shape.q_heads;
        var kh = shape.kv_heads;
        var d = shape.head_dim;
        var t = theta;
        var rs = rope_scale;
        var args = [_]?*anyopaque{ @ptrCast(&qp), @ptrCast(&kp), @ptrCast(&mp), @ptrCast(&b), @ptrCast(&s), @ptrCast(&qh), @ptrCast(&kh), @ptrCast(&d), @ptrCast(&t), @ptrCast(&rs) };
        const token_rows = try std.math.mul(u32, shape.batch, shape.seq);
        try launch(ctx, self.rope, &args, .{ token_rows, (shape.head_dim / 2 + 127) / 128, shape.q_heads }, .{ 128, 1, 1 });
        self.stats_.launches += 1;
    }

    pub fn launchAttention(self: *Module, ctx: *Context, out: Buffer, q: Buffer, k: Buffer, v: Buffer, mask: Buffer, shape: Attention) !void {
        try shape.validate();
        const qb = @as(usize, shape.batch) * shape.seq * shape.q_heads * shape.head_dim * 2;
        const kvb = @as(usize, shape.batch) * shape.seq * shape.kv_heads * shape.head_dim * 2;
        try require(out, qb);
        try require(q, qb);
        try require(k, kvb);
        try require(v, kvb);
        if (mask.ptr != 0) try require(mask, @as(usize, shape.batch) * shape.seq * 8);
        var op = out.ptr;
        var qp = q.ptr;
        var kp = k.ptr;
        var vp = v.ptr;
        var mp = mask.ptr;
        var b = shape.batch;
        var s = shape.seq;
        var qh = shape.q_heads;
        var kh = shape.kv_heads;
        var d = shape.head_dim;
        var radius = shape.local_radius;
        var query_offset: u32 = 0;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&qp), @ptrCast(&kp), @ptrCast(&vp), @ptrCast(&mp), @ptrCast(&b), @ptrCast(&s), @ptrCast(&qh), @ptrCast(&kh), @ptrCast(&d), @ptrCast(&radius), @ptrCast(&query_offset) };
        try launch(ctx, self.attention, &args, .{ shape.seq, shape.q_heads, shape.batch }, .{ 32, 1, 1 });
        self.stats_.launches += 1;
        self.stats_.attention_launches += 1;
    }

    pub fn launchLocalFlashAttention(self: *Module, ctx: *Context, out: Buffer, q: Buffer, k: Buffer, v: Buffer, mask: Buffer, shape: Attention) !void {
        try shape.validate();
        if (shape.local_radius != 512 or shape.kv_heads != 2 or shape.head_dim != 256) return error.InvalidEmbeddingGemma2Shape;
        const qb = try std.math.mul(usize, @as(usize, shape.batch) * shape.seq * shape.q_heads, shape.head_dim * 2);
        const kvb = try std.math.mul(usize, @as(usize, shape.batch) * shape.seq * shape.kv_heads, shape.head_dim * 2);
        try require(out, qb);
        try require(q, qb);
        try require(k, kvb);
        try require(v, kvb);
        if (mask.ptr != 0) try require(mask, @as(usize, shape.batch) * shape.seq * 8);
        const full_tiles = shape.seq / 64;
        if (full_tiles != 0) {
            var op = out.ptr;
            var qp = q.ptr;
            var kp = k.ptr;
            var vp = v.ptr;
            var mp = mask.ptr;
            var b = shape.batch;
            var s = shape.seq;
            var qh = shape.q_heads;
            var kh = shape.kv_heads;
            var d = shape.head_dim;
            var radius = shape.local_radius;
            var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&qp), @ptrCast(&kp), @ptrCast(&vp), @ptrCast(&mp), @ptrCast(&b), @ptrCast(&s), @ptrCast(&qh), @ptrCast(&kh), @ptrCast(&d), @ptrCast(&radius) };
            const shared = @as(u32, 16) * shape.head_dim * 2 + 4 * 64 * 16 * 4 + 64 * 16 * 2 + 3 * 64 * 4;
            try launchShared(ctx, self.local_flash_attention, &args, .{ shape.q_heads, full_tiles, shape.batch }, .{ 512, 1, 1 }, shared);
            self.stats_.launches += 1;
        }
        const tail = shape.seq % 64;
        if (tail != 0) {
            var op = out.ptr;
            var qp = q.ptr;
            var kp = k.ptr;
            var vp = v.ptr;
            var mp = mask.ptr;
            var b = shape.batch;
            var s = shape.seq;
            var qh = shape.q_heads;
            var kh = shape.kv_heads;
            var d = shape.head_dim;
            var radius = shape.local_radius;
            var query_offset = shape.seq - tail;
            var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&qp), @ptrCast(&kp), @ptrCast(&vp), @ptrCast(&mp), @ptrCast(&b), @ptrCast(&s), @ptrCast(&qh), @ptrCast(&kh), @ptrCast(&d), @ptrCast(&radius), @ptrCast(&query_offset) };
            try launch(ctx, self.attention, &args, .{ tail, shape.q_heads, shape.batch }, .{ 32, 1, 1 });
            self.stats_.launches += 1;
        }
        self.stats_.attention_launches += 1;
    }

    pub fn launchAttentionSoftmax(self: *Module, ctx: *Context, probs: Buffer, scores: Buffer, mask: Buffer, sequence: u32, query_base: u32, rows: u32, batch_index: u32, heads: u32, tile_capacity: u32) !void {
        if (sequence == 0 or sequence > 8192 or tile_capacity == 0 or tile_capacity > 2048 or rows == 0 or rows > tile_capacity or query_base > sequence or rows > sequence - query_base or heads == 0 or heads > 64) return error.InvalidEmbeddingGemma2Shape;
        const elements = try std.math.mul(usize, try std.math.mul(usize, heads, tile_capacity), sequence);
        try require(scores, try std.math.mul(usize, elements, 4));
        try require(probs, try std.math.mul(usize, elements, 2));
        if (mask.ptr != 0) try require(mask, (@as(usize, batch_index) + 1) * sequence * 8);
        var pp = probs.ptr;
        var sp = scores.ptr;
        var mp = mask.ptr;
        var s = sequence;
        var qb = query_base;
        var r = rows;
        var bi = batch_index;
        var tc = tile_capacity;
        var args = [_]?*anyopaque{ @ptrCast(&pp), @ptrCast(&sp), @ptrCast(&mp), @ptrCast(&s), @ptrCast(&qb), @ptrCast(&r), @ptrCast(&bi), @ptrCast(&tc) };
        try launchShared(ctx, self.attention_softmax, &args, .{ rows * heads, 1, 1 }, .{ 256, 1, 1 }, 256 * 4);
        self.stats_.launches += 1;
    }

    pub fn launchMatmul(self: *Module, ctx: *Context, out: Buffer, input: Buffer, weight: Buffer, rows: u32, in_dim: u32, out_dim: u32) !void {
        try require(out, @as(usize, rows) * out_dim * 2);
        try require(input, @as(usize, rows) * in_dim * 2);
        try require(weight, @as(usize, out_dim) * in_dim * 2);
        var op = out.ptr;
        var ip = input.ptr;
        var wp = weight.ptr;
        var r = rows;
        var k = in_dim;
        var n = out_dim;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&wp), @ptrCast(&r), @ptrCast(&k), @ptrCast(&n) };
        try launch(ctx, self.matmul, &args, .{ rows, (out_dim + 127) / 128, 1 }, .{ 128, 1, 1 });
        self.stats_.launches += 1;
        self.stats_.scalar_matmul_fallbacks += 1;
    }

    pub fn launchGeluPle(self: *Module, ctx: *Context, out: Buffer, gate: Buffer, ple: Buffer, count: u32) !void {
        const bytes = @as(usize, count) * 2;
        try require(out, bytes);
        try require(gate, bytes);
        try require(ple, bytes);
        var op = out.ptr;
        var gp = gate.ptr;
        var pp = ple.ptr;
        var n = count;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&gp), @ptrCast(&pp), @ptrCast(&n) };
        try self.launch1d(ctx, self.gelu_ple, &args, count);
    }

    pub fn launchGeluMul(self: *Module, ctx: *Context, out: Buffer, gate: Buffer, up: Buffer, count: u32) !void {
        const bytes = @as(usize, count) * 2;
        try require(out, bytes);
        try require(gate, bytes);
        try require(up, bytes);
        var op = out.ptr;
        var gp = gate.ptr;
        var uptr = up.ptr;
        var n = count;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&gp), @ptrCast(&uptr), @ptrCast(&n) };
        try self.launch1d(ctx, self.gelu_mul, &args, count);
    }

    pub fn launchBf16ToF32(self: *Module, ctx: *Context, out: Buffer, input: Buffer, count: u32) !void {
        try require(out, @as(usize, count) * @sizeOf(f32));
        try require(input, @as(usize, count) * @sizeOf(u16));
        var op = out.ptr;
        var ip = input.ptr;
        var n = count;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&n) };
        try self.launch1d(ctx, self.bf16_to_f32, &args, count);
    }
    pub fn launchScale(self: *Module, ctx: *Context, out: Buffer, input: Buffer, count: u32, scale_value: f32) !void {
        const bytes = @as(usize, count) * 2;
        try require(out, bytes);
        try require(input, bytes);
        var op = out.ptr;
        var ip = input.ptr;
        var n = count;
        var s = scale_value;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&n), @ptrCast(&s) };
        try self.launch1d(ctx, self.scale, &args, count);
    }
    pub fn launchScaleDevice(self: *Module, ctx: *Context, out: Buffer, input: Buffer, scale_value: Buffer, count: u32) !void {
        const bytes = @as(usize, count) * 2;
        try require(out, bytes);
        try require(input, bytes);
        try require(scale_value, 2);
        var op = out.ptr;
        var ip = input.ptr;
        var sp = scale_value.ptr;
        var n = count;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&ip), @ptrCast(&sp), @ptrCast(&n) };
        try self.launch1d(ctx, self.scale_device, &args, count);
    }
    pub fn launchResidual(self: *Module, ctx: *Context, out: Buffer, residual: Buffer, branch: Buffer, ple: ?Buffer, count: u32, branch_scale: f32, ple_scale: f32) !void {
        const bytes = @as(usize, count) * 2;
        try require(out, bytes);
        try require(residual, bytes);
        try require(branch, bytes);
        if (ple) |p| try require(p, bytes);
        var op = out.ptr;
        var rp = residual.ptr;
        var bp = branch.ptr;
        var pp = if (ple) |p| p.ptr else 0;
        var n = count;
        var bs = branch_scale;
        var ps = ple_scale;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&rp), @ptrCast(&bp), @ptrCast(&pp), @ptrCast(&n), @ptrCast(&bs), @ptrCast(&ps) };
        try self.launch1d(ctx, self.residual, &args, count);
    }
    pub fn launchMeanL2(self: *Module, ctx: *Context, out: Buffer, projected: Buffer, mask: Buffer, batch: u32, seq: u32, width: u32) !void {
        try require(out, @as(usize, batch) * width * 4);
        try require(projected, @as(usize, batch) * seq * width * 2);
        if (mask.ptr != 0) try require(mask, @as(usize, batch) * seq * 8);
        var op = out.ptr;
        var pp = projected.ptr;
        var mp = mask.ptr;
        var b = batch;
        var s = seq;
        var w = width;
        var args = [_]?*anyopaque{ @ptrCast(&op), @ptrCast(&pp), @ptrCast(&mp), @ptrCast(&b), @ptrCast(&s), @ptrCast(&w) };
        try launch(ctx, self.mean_l2, &args, .{ batch, 1, 1 }, .{ 256, 1, 1 });
        self.stats_.launches += 1;
        self.stats_.output_bytes += @as(u64, batch) * width * 4;
    }
    fn launch1d(self: *Module, ctx: *Context, f: driver.CUfunction, args: []?*anyopaque, n: usize) !void {
        if (n == 0) return;
        try launch(ctx, f, args, .{ @intCast((n + 255) / 256), 1, 1 }, .{ 256, 1, 1 });
        self.stats_.launches += 1;
    }
};

fn function(ctx: *Context, m: driver.CUmodule, name: [:0]const u8) !driver.CUfunction {
    var f: driver.CUfunction = null;
    try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&f, m, name.ptr));
    return f;
}
fn require(b: Buffer, n: usize) !void {
    if ((n != 0 and b.ptr == 0) or b.len < n) return error.InvalidCudaState;
}
fn launch(ctx: *Context, f: driver.CUfunction, args: []?*anyopaque, grid: [3]u32, block: [3]u32) !void {
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, ctx.stream, args.ptr, null));
    ctx.noteKernelLaunch();
}
fn launchShared(ctx: *Context, f: driver.CUfunction, args: []?*anyopaque, grid: [3]u32, block: [3]u32, shared: u32) !void {
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], shared, ctx.stream, args.ptr, null));
    ctx.noteKernelLaunch();
}

test "EmbeddingGemma 2 attention shape contract" {
    try (Attention{ .batch = 1, .seq = 8192, .q_heads = 4, .kv_heads = 2, .head_dim = 256, .local_radius = 512 }).validate();
    try (Attention{ .batch = 2, .seq = 17, .q_heads = 4, .kv_heads = 1, .head_dim = 512, .local_radius = 0 }).validate();
    try std.testing.expectError(error.InvalidEmbeddingGemma2Shape, (Attention{ .batch = 1, .seq = 8193, .q_heads = 4, .kv_heads = 2, .head_dim = 256, .local_radius = 512 }).validate());
}
