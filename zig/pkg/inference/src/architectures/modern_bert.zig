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

// ModernBERT encoder architecture using abstract ComputeBackend ops.
//
// ModernBERT (Warner et al., 2024) is a modernised BERT-family encoder with:
//   - Pre-norm (LayerNorm before each sub-layer, not after)
//   - RoPE positional encoding applied per-layer (no absolute position embeddings)
//   - GeGLU feed-forward networks
//   - Alternating global (full) and local (sliding-window) self-attention
//
// Weight naming follows the HuggingFace ModernBERT safetensors convention:
//   model.embeddings.tok_embeddings.weight
//   model.embeddings.norm.{weight,bias}
//   model.layers.N.attn_norm.{weight,bias}
//   model.layers.N.attn.{query_proj,key_proj,value_proj}.{weight,bias}
//   model.layers.N.attn.Wo.{weight,bias}
//   model.layers.N.mlp_norm.{weight,bias}
//   model.layers.N.mlp.Wi.weight          [2*intermediate_size, hidden_size]
//   model.layers.N.mlp.Wo.weight          [hidden_size, intermediate_size]
//   model.final_norm.{weight,bias}
//
// Single implementation works with any ComputeBackend (native, etc).

const std = @import("std");
const ops = @import("../ops/ops.zig");
const CT = ops.CT;
const ComputeBackend = ops.ComputeBackend;

const tensor_probe_hash_offset: u64 = 14695981039346656037;
const tensor_probe_hash_prime: u64 = 1099511628211;
const tensor_probe_top_abs_count: usize = 8;

fn getenvBool(comptime name: [*:0]const u8) bool {
    const raw = std.c.getenv(name) orelse return false;
    if (raw[0] == 0) return false;
    return !(raw[0] == '0' and raw[1] == 0);
}

fn layerTimingEnabled() bool {
    return getenvBool("ANTFLY_MODERN_BERT_LAYER_TIMING");
}

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

fn elapsedNs(start_ns: u64) u64 {
    const end_ns = monotonicNowNs();
    return if (end_ns > start_ns) end_ns - start_ns else 0;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms));
}

fn printModernBertTiming(layer_idx: usize, stage: []const u8, ns: u64) void {
    if (layer_idx == std.math.maxInt(usize)) {
        std.debug.print("modernbert_timing layer=none stage={s} ms={d:.3}\n", .{ stage, nsToMs(ns) });
    } else {
        std.debug.print("modernbert_timing layer={d} stage={s} ms={d:.3}\n", .{ layer_idx, stage, nsToMs(ns) });
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub const Config = struct {
    vocab_size: u32 = 50368,
    hidden_size: u32 = 768,
    num_hidden_layers: u32 = 22,
    num_attention_heads: u32 = 12,
    /// GeGLU inner dimension.  Wi projects hidden → 2*intermediate_size, then
    /// we split the output, apply GELU to the gate half, multiply, and project
    /// the resulting [total, intermediate_size] down via Wo.
    intermediate_size: u32 = 1152,
    max_position_embeddings: u32 = 8192,
    /// RoPE theta for global (full) attention layers.
    global_rope_theta: f32 = 160000.0,
    /// RoPE theta for local (sliding-window) attention layers.
    local_rope_theta: f32 = 10000.0,
    /// Layers whose index is divisible by this value use full attention.
    /// All other layers use sliding-window (local) attention.
    global_attn_every_n_layers: u32 = 3,
    /// Full sliding-window width: each query attends ±(local_attention_window/2) tokens.
    local_attention_window: u32 = 128,
    layer_norm_eps: f32 = 1e-5,
    use_geglu: bool = true,
    /// LoRA rank for targeted encoder projections.  0 = LoRA disabled.
    /// When non-zero the encoder tries to load lora_a/lora_b weight tensors
    /// from the active WeightStore and uses linearLoRA for Q/K/V/O and MLP Wo.
    lora_rank: u32 = 0,
    /// LoRA scaling alpha.  The effective scale applied to the LoRA delta is
    /// alpha / rank.  Defaults to rank (i.e., scale = 1.0) when 0 is passed.
    lora_alpha: f32 = 0.0,
    /// NEFTune embedding noise alpha.  When non-zero, uniform noise is added
    /// after token gather and before embedding layer norm during training.
    neftune_alpha: f32 = 0.0,
    neftune_seed: u64 = 0,
    /// RoPE pairing convention. `true` (default) rotates interleaved adjacent
    /// pairs (2j, 2j+1) - the convention the gopeft-trained fused chunker
    /// weights expect. `false` uses HF ModernBERT's `rotate_half` split-half
    /// pairing (j, j+head_dim/2), which the stock nomic modernbert-embed-base
    /// checkpoint was trained with. Serving the frozen nomic weights with the
    /// interleaved convention scrambles attention and roughly halves retrieval
    /// quality, so the native embedder session sets this to `false`.
    rope_interleaved: bool = true,
    /// Layer-0 pre-attention norm behavior. HF ModernBERT sets the first
    /// layer's `attn_norm` to `nn.Identity()` because `embeddings.norm`
    /// already applied a (learned) LayerNorm; re-normalizing at layer 0 would
    /// discard that learned affine. `false` (default) keeps the historical
    /// LayerNorm(ones,zeros) the gopeft-trained fused chunker expects; `true`
    /// makes layer 0 a true identity, matching stock HF / nomic weights.
    attn_norm0_identity: bool = false,
};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Run the full ModernBERT encoder forward pass.
/// Returns an owned f32 slice of shape [batch * seq_len * hidden_size].
/// Caller must free the returned slice with `allocator.free`.
pub fn forward(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    /// 1 = real token, 0 = padding; flat shape [batch * seq_len].
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
) ![]f32 {
    const result_ct = try forwardCT(cb, allocator, config, input_ids, attention_mask, batch, seq_len);
    defer cb.free(result_ct);
    return cb.toFloat32(result_ct, allocator);
}

/// Run the full ModernBERT encoder forward pass and return a CT.
/// Caller owns the returned tensor and must free it with `cb.free`.
pub fn forwardCT(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    /// 1 = real token, 0 = padding; flat shape [batch * seq_len].
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
) !CT {
    const trace_timing = layerTimingEnabled();
    var stage_start_ns = if (trace_timing) monotonicNowNs() else 0;

    // 1. Token embeddings + embedding LayerNorm.
    //    ModernBERT has no absolute position embeddings; RoPE is applied in each
    //    attention layer instead.
    var hidden = try embeddingsBlock(cb, allocator, config, input_ids, batch * seq_len, seq_len, null);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "embeddings", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    var local_window_bias: ?CT = null;
    defer if (local_window_bias) |bias| cb.free(bias);

    // 2. Encoder layers
    for (0..config.num_hidden_layers) |layer_idx| {
        const new_hidden = try encoderLayer(
            cb,
            allocator,
            config,
            hidden,
            attention_mask,
            batch,
            seq_len,
            layer_idx,
            &local_window_bias,
        );
        cb.free(hidden);
        hidden = new_hidden;
    }
    if (trace_timing) stage_start_ns = monotonicNowNs();

    // 3. Final layer norm
    var name_buf: [128]u8 = undefined;
    const fn_w = try cb.getWeight(std.fmt.bufPrint(&name_buf, "model.final_norm.weight", .{}) catch return error.NameTooLong);
    defer cb.free(fn_w);
    const fn_b = try getWeightOrZeroBias(cb, allocator, std.fmt.bufPrint(&name_buf, "model.final_norm.bias", .{}) catch return error.NameTooLong, @intCast(config.hidden_size));
    defer cb.free(fn_b);

    const normed_final = try cb.layerNorm(hidden, fn_w, fn_b, @intCast(config.hidden_size), config.layer_norm_eps);
    cb.free(hidden);
    if (trace_timing) printModernBertTiming(std.math.maxInt(usize), "final_norm", elapsedNs(stage_start_ns));
    return normed_final;
}

// ---------------------------------------------------------------------------
// Embeddings block
// ---------------------------------------------------------------------------

/// Public wrapper around the embeddings block (token lookup + NEFTune noise +
/// embedding LayerNorm) for callers that run the encoder layers through a
/// compiled segment session. Behaves exactly like the eager forward's
/// embedding stage, including the embedding captures when `captures` is set.
/// Caller owns the returned tensor and must free it with `cb.free`.
pub fn embeddingsForwardCT(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    total: usize,
    seq_len: usize,
    captures: ?*ActivationBuffer,
) !CT {
    return embeddingsBlock(cb, allocator, config, input_ids, total, seq_len, captures);
}

fn embeddingsBlock(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    total: usize,
    seq_len: usize,
    captures: ?*ActivationBuffer,
) !CT {
    const H = config.hidden_size;
    const H_usize: usize = @intCast(H);

    // Word / token embeddings
    const tok_emb_w = try cb.getWeight("model.embeddings.tok_embeddings.weight");
    defer cb.free(tok_emb_w);
    var embedded = try cb.embeddingLookup(tok_emb_w, input_ids, total, H);
    errdefer cb.free(embedded);

    if (config.neftune_alpha > 0.0 and total > 0) {
        const noise_scale = neftuneNoiseScale(config.neftune_alpha, seq_len, H_usize);
        if (try cb.addNeftuneNoise(embedded, config.neftune_seed, noise_scale)) |noised| {
            cb.free(embedded);
            embedded = noised;
        } else {
            const noise = try allocator.alloc(f32, total * H_usize);
            defer allocator.free(noise);

            var prng = std.Random.DefaultPrng.init(config.neftune_seed);
            const rng = prng.random();
            for (noise) |*v| {
                v.* = (rng.float(f32) * 2.0 - 1.0) * noise_scale;
            }

            const noise_shape = [_]i32{ @as(i32, @intCast(total)), @as(i32, @intCast(H)) };
            const noise_ct = try cb.fromFloat32Shape(noise, &noise_shape);
            defer cb.free(noise_ct);

            const noised = try cb.add(embedded, noise_ct);
            cb.free(embedded);
            embedded = noised;
        }
    }

    // Embedding-level LayerNorm (replaces post-sum norm from classic BERT)
    const ln_w = try cb.getWeight("model.embeddings.norm.weight");
    defer cb.free(ln_w);
    const ln_b = try getWeightOrZeroBias(cb, allocator, "model.embeddings.norm.bias", H);
    defer cb.free(ln_b);

    const normed = try cb.layerNorm(embedded, ln_w, ln_b, H, 1e-5);
    errdefer cb.free(normed);
    if (captures) |capture_buf| {
        const embedded_f32 = try cb.toFloat32(embedded, allocator);
        var owns_embedded = true;
        errdefer if (owns_embedded) allocator.free(embedded_f32);
        const normed_f32 = try cb.toFloat32(normed, allocator);
        var owns_normed = true;
        errdefer if (owns_normed) allocator.free(normed_f32);
        try capture_buf.setEmbeddingOwned(embedded_f32, normed_f32, total, H_usize);
        owns_embedded = false;
        owns_normed = false;
    }
    cb.free(embedded);
    return normed;
}

inline fn neftuneNoiseScale(alpha: f32, seq_len: usize, hidden_size: usize) f32 {
    if (alpha <= 0.0 or seq_len == 0 or hidden_size == 0) return 0.0;
    const denom = @as(f32, @floatFromInt(seq_len)) * @as(f32, @floatFromInt(hidden_size));
    return alpha / @sqrt(denom);
}

// ---------------------------------------------------------------------------
// Single encoder layer
// ---------------------------------------------------------------------------

fn hasLocalAttentionLayers(config: Config) bool {
    if (config.global_attn_every_n_layers == 0) return false;
    for (0..config.num_hidden_layers) |layer_idx| {
        if ((layer_idx % @as(usize, @intCast(config.global_attn_every_n_layers))) != 0) return true;
    }
    return false;
}

fn modernBertAttention(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    Q: CT,
    K: CT,
    V: CT,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    is_global: bool,
    local_window_bias_cache: ?*?CT,
) !CT {
    if (!is_global) {
        const half: usize = @intCast(config.local_attention_window / 2);
        if (try cb.scaledDotProductAttentionLocal(Q, K, V, attention_mask, batch, seq_len, num_heads, head_dim, half)) |attn_out| {
            return attn_out;
        }
    }

    var owned_window_bias: ?CT = null;
    defer if (owned_window_bias) |wb| cb.free(wb);
    const window_bias: ?CT = if (!is_global) blk: {
        if (local_window_bias_cache) |cache| {
            if (cache.* == null) {
                const half: usize = @intCast(config.local_attention_window / 2);
                cache.* = try buildSlidingWindowBias(cb, allocator, seq_len, num_heads, half);
            }
            break :blk cache.*.?;
        }
        const half: usize = @intCast(config.local_attention_window / 2);
        owned_window_bias = try buildSlidingWindowBias(cb, allocator, seq_len, num_heads, half);
        break :blk owned_window_bias.?;
    } else null;

    return cb.scaledDotProductAttention(
        Q,
        K,
        V,
        attention_mask,
        window_bias,
        batch,
        seq_len,
        num_heads,
        head_dim,
    );
}

fn encoderLayer(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden: CT,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    layer_idx: usize,
    local_window_bias_cache: ?*?CT,
) !CT {
    const trace_timing = layerTimingEnabled();
    const layer_start_ns = if (trace_timing) monotonicNowNs() else 0;
    var stage_start_ns = layer_start_ns;
    const H: usize = @intCast(config.hidden_size);
    const num_heads: usize = @intCast(config.num_attention_heads);
    const head_dim = H / num_heads;
    const intermediate: usize = @intCast(config.intermediate_size);
    const total = batch * seq_len;

    // Layers 0, 3, 6, … use full (global) attention; all others are local.
    const is_global = (layer_idx % @as(usize, @intCast(config.global_attn_every_n_layers))) == 0;
    const rope_theta = if (is_global) config.global_rope_theta else config.local_rope_theta;

    var name_buf: [256]u8 = undefined;

    // -----------------------------------------------------------------------
    // Self-attention sub-layer  (pre-norm)
    // -----------------------------------------------------------------------

    const normed_attn = try preAttentionNorm(cb, allocator, config, hidden, layer_idx, total, H, &name_buf);
    defer cb.free(normed_attn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "pre_attn_norm", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Q projection — use linearLoRA if LoRA is enabled in the config.
    const q_w = try getLayerWeight(cb, layer_idx, "attn.query_proj.weight", &name_buf);
    defer cb.free(q_w);
    const q_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.query_proj.bias", H, &name_buf);
    defer cb.free(q_b);
    const Q_raw = try linearWithLoRA(cb, allocator, normed_attn, q_w, q_b, layer_idx, "query_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(Q_raw);

    // K projection — use linearLoRA if LoRA is enabled in the config.
    const k_w = try getLayerWeight(cb, layer_idx, "attn.key_proj.weight", &name_buf);
    defer cb.free(k_w);
    const k_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.key_proj.bias", H, &name_buf);
    defer cb.free(k_b);
    const K_raw = try linearWithLoRA(cb, allocator, normed_attn, k_w, k_b, layer_idx, "key_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(K_raw);

    // V projection — use linearLoRA if LoRA is enabled in the config.
    const v_w = try getLayerWeight(cb, layer_idx, "attn.value_proj.weight", &name_buf);
    defer cb.free(v_w);
    const v_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.value_proj.bias", H, &name_buf);
    defer cb.free(v_b);
    const V = try linearWithLoRA(cb, allocator, normed_attn, v_w, v_b, layer_idx, "value_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(V);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "qkv_project", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Apply RoPE to Q and K.
    // RoPE pairing is config.rope_interleaved: interleaved (2j,2j+1) for the
    // gopeft-trained fused chunker, or HF rotate_half (j, j+head_dim/2) for the
    // stock nomic modernbert-embed-base checkpoint.
    // rope_dim == head_dim: the full head dimension is rotated.
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(K);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "rope", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    // Bidirectional scaled dot-product attention (encoder, no causal mask).
    // The padding mask (attention_mask) is consumed by the backend: positions
    // where mask[b*seq_len + ki] == 0 are set to -inf before softmax.
    const attn_out = try modernBertAttention(
        cb,
        allocator,
        config,
        Q,
        K,
        V,
        attention_mask,
        batch,
        seq_len,
        num_heads,
        head_dim,
        is_global,
        local_window_bias_cache,
    );
    defer cb.free(attn_out);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "attention", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Output projection
    const out_w = try getLayerWeight(cb, layer_idx, "attn.Wo.weight", &name_buf);
    defer cb.free(out_w);
    const out_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.Wo.bias", H, &name_buf);
    defer cb.free(out_b);
    const attn_proj = try linearWithLoRA(cb, allocator, attn_out, out_w, out_b, layer_idx, "Wo", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(attn_proj);

    // Residual: add the projected attention output to the *original* (pre-norm)
    // hidden state — pre-norm residual pattern.
    const hidden_after_attn = try cb.add(attn_proj, hidden);
    defer cb.free(hidden_after_attn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "attn_output_residual", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // -----------------------------------------------------------------------
    // FFN sub-layer  (pre-norm, GeGLU)
    // -----------------------------------------------------------------------

    // Pre-FFN LayerNorm
    const mlp_ln_w = try getLayerWeight(cb, layer_idx, "mlp_norm.weight", &name_buf);
    defer cb.free(mlp_ln_w);
    const mlp_ln_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "mlp_norm.bias", H, &name_buf);
    defer cb.free(mlp_ln_b);
    const normed_ffn = try cb.layerNorm(hidden_after_attn, mlp_ln_w, mlp_ln_b, H, config.layer_norm_eps);
    defer cb.free(normed_ffn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "ffn_norm", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // GeGLU feed-forward (Wi and Wo both have no bias in ModernBERT's MLP)
    const Wi_w = try getLayerWeight(cb, layer_idx, "mlp.Wi.weight", &name_buf);
    defer cb.free(Wi_w);
    const Wo_w = try getLayerWeight(cb, layer_idx, "mlp.Wo.weight", &name_buf);
    defer cb.free(Wo_w);

    const ffn_out = try geGluFfn(cb, allocator, normed_ffn, Wi_w, Wo_w, layer_idx, config.lora_rank, config.lora_alpha, total, H, intermediate, null);
    defer cb.free(ffn_out);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "geglu", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Residual: add FFN output to post-attention hidden state
    const layer_out = try cb.add(ffn_out, hidden_after_attn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "ffn_residual", elapsedNs(stage_start_ns));
        printModernBertTiming(layer_idx, "total", elapsedNs(layer_start_ns));
    }
    return layer_out;
}

// ---------------------------------------------------------------------------
// GeGLU feed-forward network
// ---------------------------------------------------------------------------
//
// Architecture (matches gopeft / HuggingFace ModernBERT):
//
//   gated  = input @ Wi^T        [total, 2*intermediate]   (no bias)
//   gate   = gated[..., :intermediate]                     first half
//   value  = gated[..., intermediate:]                     second half
//   act    = GELU(gate) * value  [total, intermediate]
//   output = act @ Wo^T          [total, hidden]           (no bias)
//
// Keep the gate/value split on the active backend.  Metal wires slice,
// exact GELU, and multiply to resident device ops, which avoids a per-layer
// download/re-upload of the large [total, 2*intermediate] projection.

fn geGluFfn(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    input: CT,
    Wi_w: CT,
    Wo_w: CT,
    layer_idx: usize,
    lora_rank: u32,
    lora_alpha: f32,
    total: usize,
    hidden_size: usize,
    intermediate_size: usize,
    captures: ?*ActivationBuffer,
) !CT {
    // Project to 2*intermediate.  Wi is [2*intermediate, hidden] (row-major,
    // transposed by the linear op) so the output is [total, 2*intermediate].
    const gated_ct = try cb.linearNoBias(input, Wi_w, total, hidden_size, 2 * intermediate_size);
    defer cb.free(gated_ct);

    const gate_ct = try cb.sliceLastDim(gated_ct, 0, intermediate_size);
    defer cb.free(gate_ct);
    const value_ct = try cb.sliceLastDim(gated_ct, intermediate_size, 2 * intermediate_size);
    defer cb.free(value_ct);

    var gate_gelu_capture_ct: ?CT = null;
    defer if (gate_gelu_capture_ct) |ct| cb.free(ct);
    if (captures) |capture_buf| {
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            const gated = try cb.toFloat32(gated_ct, allocator);
            defer allocator.free(gated);
            if (gated.len != total * intermediate_size * 2) return error.UnexpectedOutputShape;
            try capture_buf.addLayerState(@intCast(layer_idx), "wi_output", gated);

            const gate = try cb.toFloat32(gate_ct, allocator);
            defer allocator.free(gate);
            if (gate.len != total * intermediate_size) return error.UnexpectedOutputShape;
            try capture_buf.addLayerState(@intCast(layer_idx), "gate_input", gate);

            const value = try cb.toFloat32(value_ct, allocator);
            defer allocator.free(value);
            if (value.len != total * intermediate_size) return error.UnexpectedOutputShape;
            try capture_buf.addLayerState(@intCast(layer_idx), "gate_value", value);

            gate_gelu_capture_ct = try cb.gelu(gate_ct);
            const gelu = try cb.toFloat32(gate_gelu_capture_ct.?, allocator);
            defer allocator.free(gelu);
            if (gelu.len != total * intermediate_size) return error.UnexpectedOutputShape;
            try capture_buf.addLayerState(@intCast(layer_idx), "gelu_input", gelu);
        }
    }

    const activated_ct = (try cb.activationMultiply(gate_ct, value_ct, .gelu, total, intermediate_size)) orelse blk: {
        const gate_gelu_ct = try cb.gelu(gate_ct);
        defer cb.free(gate_gelu_ct);
        break :blk try cb.multiply(gate_gelu_ct, value_ct);
    };
    defer cb.free(activated_ct);

    if (captures) |capture_buf| {
        const activated = try cb.toFloat32(activated_ct, allocator);
        defer allocator.free(activated);
        if (activated.len != total * intermediate_size) return error.UnexpectedOutputShape;
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            try capture_buf.addLayerState(@intCast(layer_idx), "wo_input", activated);
        }
        try capture_buf.add(@intCast(layer_idx), "wo", activated, intermediate_size, hidden_size, total);
        if (capture_buf.capture_projection_decomposition_layer != null and capture_buf.capture_projection_decomposition_layer.? == layer_idx) {
            try captureProjectionDecomposition(
                cb,
                allocator,
                capture_buf,
                @intCast(layer_idx),
                "wo",
                activated_ct,
                Wo_w,
                null,
                "mlp",
                "Wo",
                lora_rank,
                lora_alpha,
                total,
                intermediate_size,
                hidden_size,
            );
        }
    }

    // Wo is [hidden, intermediate] so the output is [total, hidden].
    const ffn_out_ct = try linearWithScopedLoRA(cb, allocator, activated_ct, Wo_w, null, layer_idx, "mlp", "Wo", lora_rank, lora_alpha, total, intermediate_size, hidden_size);
    errdefer cb.free(ffn_out_ct);
    if (captures) |capture_buf| {
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            const ffn_out = try cb.toFloat32(ffn_out_ct, allocator);
            defer allocator.free(ffn_out);
            if (ffn_out.len != total * hidden_size) return error.UnexpectedOutputShape;
            try capture_buf.addLayerState(@intCast(layer_idx), "ffn_out", ffn_out);
        }
    }
    return ffn_out_ct;
}

// ---------------------------------------------------------------------------
// Sliding-window additive attention bias  (local attention layers)
// ---------------------------------------------------------------------------
//
// Returns a CT of flat length [num_heads * seq_len * seq_len] where element
// [h, qi, ki] is:
//   0.0  when |qi - ki| <= window_half  (ki is inside the sliding window)
//   -inf when |qi - ki| >  window_half  (ki is outside the sliding window)
//
// All heads share an identical mask.  The BLAS sdpaOp selects the shared
// head-indexed form when len == num_heads * seq_len * seq_len.

fn buildSlidingWindowBias(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    seq_len: usize,
    num_heads: usize,
    window_half: usize,
) !CT {
    const n = num_heads * seq_len * seq_len;
    const data = try allocator.alloc(f32, n);
    defer allocator.free(data);

    for (0..num_heads) |h| {
        const head_base = h * seq_len * seq_len;
        for (0..seq_len) |qi| {
            for (0..seq_len) |ki| {
                const diff: usize = if (qi >= ki) qi - ki else ki - qi;
                data[head_base + qi * seq_len + ki] =
                    if (diff > window_half) -std.math.inf(f32) else 0.0;
            }
        }
    }

    return cb.fromFloat32(data);
}

// ---------------------------------------------------------------------------
// Weight-name helpers
// ---------------------------------------------------------------------------

/// Build "model.layers.{layer}.{suffix}" and look up the weight tensor.
fn getLayerWeight(
    cb: *const ComputeBackend,
    layer: usize,
    suffix: []const u8,
    buf: *[256]u8,
) !CT {
    const name = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, suffix }) catch return error.NameTooLong;
    return cb.getWeight(name);
}

fn getWeightOrZeroBias(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    name: []const u8,
    dim: usize,
) !CT {
    return cb.getWeight(name) catch |err| switch (err) {
        error.MissingWeight => try zeroBiasTensor(cb, allocator, dim),
        else => err,
    };
}

fn getLayerWeightOrZeroBias(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    layer: usize,
    suffix: []const u8,
    dim: usize,
    buf: *[256]u8,
) !CT {
    const name = std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ layer, suffix }) catch return error.NameTooLong;
    return getWeightOrZeroBias(cb, allocator, name, dim);
}

fn zeroBiasTensor(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    dim: usize,
) !CT {
    const zeros = try allocator.alloc(f32, dim);
    defer allocator.free(zeros);
    @memset(zeros, 0.0);
    return cb.fromFloat32Shape(zeros, &.{@as(i32, @intCast(dim))});
}

fn oneWeightTensor(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    dim: usize,
) !CT {
    const ones = try allocator.alloc(f32, dim);
    defer allocator.free(ones);
    @memset(ones, 1.0);
    return cb.fromFloat32Shape(ones, &.{@as(i32, @intCast(dim))});
}

fn cloneHiddenForNormSkip(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    hidden: CT,
    total: usize,
    hidden_size: usize,
) !CT {
    const shape = [_]i32{ @intCast(total), @intCast(hidden_size) };
    if (try cb.cloneTensorShape(hidden, &shape)) |cloned| return cloned;
    const host = try cb.toFloat32(hidden, allocator);
    defer allocator.free(host);
    return cb.fromFloat32Shape(host, &shape);
}

fn preAttentionNorm(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden: CT,
    layer_idx: usize,
    total: usize,
    hidden_size: usize,
    name_buf: *[256]u8,
) !CT {
    if (layer_idx == 0) {
        if (config.attn_norm0_identity) {
            // HF ModernBERT layer 0 uses nn.Identity: embeddings.norm already
            // normalized, so pass the hidden state through unchanged (an owned
            // copy, since the caller frees the returned tensor).
            return cloneHiddenForNormSkip(cb, allocator, hidden, total, hidden_size);
        }
        const attn_ln_w = try oneWeightTensor(cb, allocator, hidden_size);
        defer cb.free(attn_ln_w);
        const attn_ln_b = try zeroBiasTensor(cb, allocator, hidden_size);
        defer cb.free(attn_ln_b);
        return cb.layerNorm(hidden, attn_ln_w, attn_ln_b, hidden_size, config.layer_norm_eps);
    }
    const attn_ln_w = try getLayerWeight(cb, layer_idx, "attn_norm.weight", name_buf);
    defer cb.free(attn_ln_w);
    const attn_ln_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn_norm.bias", hidden_size, name_buf);
    defer cb.free(attn_ln_b);
    return cb.layerNorm(hidden, attn_ln_w, attn_ln_b, hidden_size, config.layer_norm_eps);
}

/// Run a linear projection for a LoRA-targeted attention module.
///
/// When `lora_rank > 0` and the backend vtable has `linearLoRA`, this function
/// tries to load the LoRA A/B tensors from the WeightStore.  If both are found
/// it calls `cb.linearLoRA`; otherwise it falls back to plain `cb.linear`.
///
/// Weight keys:  "model.layers.{layer}.attn.{proj_name}.lora_{a,b}"
fn linearFallback(
    cb: *const ComputeBackend,
    input: CT,
    base_w: CT,
    base_b: ?CT,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !CT {
    if (base_b) |bias| {
        return cb.linear(input, base_w, bias, rows, in_dim, out_dim);
    }
    return cb.linearNoBias(input, base_w, rows, in_dim, out_dim);
}

fn linearWithScopedLoRA(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    input: CT,
    base_w: CT,
    base_b: ?CT,
    layer: usize,
    scope_name: []const u8,
    proj_name: []const u8,
    lora_rank: u32,
    lora_alpha: f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !CT {
    _ = allocator;
    if (lora_rank > 0 and cb.vtable.linearLoRA != null) {
        var key_a_buf: [128]u8 = undefined;
        var key_b_buf: [128]u8 = undefined;
        const key_a = std.fmt.bufPrint(&key_a_buf, "model.layers.{d}.{s}.{s}.lora_a", .{ layer, scope_name, proj_name }) catch
            return linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim);
        const key_b = std.fmt.bufPrint(&key_b_buf, "model.layers.{d}.{s}.{s}.lora_b", .{ layer, scope_name, proj_name }) catch
            return linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim);

        const lora_a = cb.getWeight(key_a) catch |err| switch (err) {
            error.MissingWeight => return linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim),
            else => return err,
        };
        defer cb.free(lora_a);

        const lora_b = cb.getWeight(key_b) catch |err| switch (err) {
            error.MissingWeight => return linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim),
            else => return err,
        };
        defer cb.free(lora_b);

        const rank: usize = @intCast(lora_rank);
        // Effective alpha: if caller passed 0.0, use rank so that scale = alpha/rank = 1.0.
        const effective_alpha: f32 = if (lora_alpha == 0.0) @floatFromInt(lora_rank) else lora_alpha;
        if (base_b == null) {
            return linearNoBiasWithLoRADelta(cb, input, base_w, lora_a, lora_b, effective_alpha, rank, rows, in_dim, out_dim);
        }
        return cb.linearLoRA(input, base_w, base_b.?, lora_a, lora_b, effective_alpha, rank, rows, in_dim, out_dim);
    }
    return linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim);
}

fn linearNoBiasWithLoRADelta(
    cb: *const ComputeBackend,
    input: CT,
    base_w: CT,
    lora_a: CT,
    lora_b: CT,
    alpha: f32,
    rank: usize,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !CT {
    const base = try cb.linearNoBias(input, base_w, rows, in_dim, out_dim);
    errdefer cb.free(base);
    if (rank == 0 or alpha == 0.0) return base;

    const a_proj = try cb.linearNoBias(input, lora_a, rows, in_dim, rank);
    defer cb.free(a_proj);
    const b_proj = try cb.linearNoBias(a_proj, lora_b, rows, rank, out_dim);
    defer cb.free(b_proj);

    const scale = alpha / @as(f32, @floatFromInt(rank));
    if (scale == 0.0) return base;

    const scaled_delta = if (scale == 1.0) b_proj else scaled_blk: {
        const scale_shape = [_]i32{1};
        const scale_tensor = try cb.fromFloat32Shape(&[_]f32{scale}, &scale_shape);
        defer cb.free(scale_tensor);
        break :scaled_blk try cb.multiply(b_proj, scale_tensor);
    };
    defer if (scaled_delta != b_proj) cb.free(scaled_delta);

    const out = try cb.add(base, scaled_delta);
    cb.free(base);
    return out;
}

fn tensorProbeStatsFromCT(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    tensor: CT,
) !TensorProbeStats {
    const values = try cb.toFloat32(tensor, allocator);
    defer allocator.free(values);
    return computeTensorProbeStats(values);
}

fn optionalTensorProbeStatsFromCT(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    tensor: ?CT,
) !TensorProbeStats {
    if (tensor) |t| return tensorProbeStatsFromCT(cb, allocator, t);
    return .{};
}

fn captureProjectionDecomposition(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    capture_buf: *ActivationBuffer,
    layer_idx: u32,
    name: []const u8,
    input: CT,
    base_w: CT,
    base_b: ?CT,
    scope_name: []const u8,
    proj_name: []const u8,
    lora_rank: u32,
    lora_alpha: f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    const base = try linearFallback(cb, input, base_w, base_b, rows, in_dim, out_dim);
    defer cb.free(base);

    const input_values = try cb.toFloat32(input, allocator);
    defer allocator.free(input_values);
    const base_values = try cb.toFloat32(base, allocator);
    defer allocator.free(base_values);
    const weight_values = try cb.toFloat32(base_w, allocator);
    defer allocator.free(weight_values);

    var bias_values_opt: ?[]f32 = null;
    defer if (bias_values_opt) |bias_values| allocator.free(bias_values);
    if (base_b) |bias_tensor| bias_values_opt = try cb.toFloat32(bias_tensor, allocator);

    const input_stats = tensorProbeStatsFromScratch(input_values);
    const base_stats = tensorProbeStatsFromScratch(base_values);
    const weight_stats = tensorProbeStatsFromScratch(weight_values);
    const bias_stats = if (bias_values_opt) |bias_values| tensorProbeStatsFromScratch(bias_values) else TensorProbeStats{};
    const base_reference_error = computeLinearReferenceErrorSample(base_values, input_values, weight_values, bias_values_opt, rows, in_dim, out_dim);

    var lora_a_stats = TensorProbeStats{};
    var lora_b_stats = TensorProbeStats{};
    var delta_stats = computeZeroTensorProbeStats(rows * out_dim);
    var output_stats = base_stats;
    var lora_a_weight_stats = TensorProbeStats{};
    var lora_b_weight_stats = TensorProbeStats{};
    var lora_a_reference_error = TensorProbeStats{};
    var lora_b_reference_error = TensorProbeStats{};
    var delta_reference_error = TensorProbeStats{};
    var output_reference_error = base_reference_error;
    var actual_rank: usize = 0;
    var scale: f32 = 0.0;

    if (lora_rank > 0 and cb.vtable.linearLoRA != null) lora_blk: {
        var key_a_buf: [128]u8 = undefined;
        var key_b_buf: [128]u8 = undefined;
        const key_a = std.fmt.bufPrint(&key_a_buf, "model.layers.{d}.{s}.{s}.lora_a", .{ layer_idx, scope_name, proj_name }) catch break :lora_blk;
        const key_b = std.fmt.bufPrint(&key_b_buf, "model.layers.{d}.{s}.{s}.lora_b", .{ layer_idx, scope_name, proj_name }) catch break :lora_blk;

        const lora_a = cb.getWeight(key_a) catch |err| switch (err) {
            error.MissingWeight => break :lora_blk,
            else => return err,
        };
        defer cb.free(lora_a);
        const lora_b = cb.getWeight(key_b) catch |err| switch (err) {
            error.MissingWeight => break :lora_blk,
            else => return err,
        };
        defer cb.free(lora_b);

        actual_rank = @intCast(lora_rank);
        const effective_alpha: f32 = if (lora_alpha == 0.0) @floatFromInt(lora_rank) else lora_alpha;
        scale = if (actual_rank == 0) 0.0 else effective_alpha / @as(f32, @floatFromInt(actual_rank));
        const lora_a_values = try cb.toFloat32(lora_a, allocator);
        defer allocator.free(lora_a_values);
        const lora_b_values = try cb.toFloat32(lora_b, allocator);
        defer allocator.free(lora_b_values);
        lora_a_weight_stats = tensorProbeStatsFromScratch(lora_a_values);
        lora_b_weight_stats = tensorProbeStatsFromScratch(lora_b_values);

        if (actual_rank == 0 or effective_alpha == 0.0) break :lora_blk;

        const a_proj = try cb.linearNoBias(input, lora_a, rows, in_dim, actual_rank);
        defer cb.free(a_proj);
        const b_proj = try cb.linearNoBias(a_proj, lora_b, rows, actual_rank, out_dim);
        defer cb.free(b_proj);

        const scaled_delta = if (scale == 1.0) b_proj else scaled_blk: {
            const scale_shape = [_]i32{1};
            const scale_tensor = try cb.fromFloat32Shape(&[_]f32{scale}, &scale_shape);
            defer cb.free(scale_tensor);
            break :scaled_blk try cb.multiply(b_proj, scale_tensor);
        };
        defer if (scaled_delta != b_proj) cb.free(scaled_delta);

        const output = try cb.add(base, scaled_delta);
        defer cb.free(output);

        const a_proj_values = try cb.toFloat32(a_proj, allocator);
        defer allocator.free(a_proj_values);
        const b_proj_values = try cb.toFloat32(b_proj, allocator);
        defer allocator.free(b_proj_values);
        const scaled_delta_values = try cb.toFloat32(scaled_delta, allocator);
        defer allocator.free(scaled_delta_values);
        const output_values = try cb.toFloat32(output, allocator);
        defer allocator.free(output_values);

        lora_a_stats = tensorProbeStatsFromScratch(a_proj_values);
        lora_b_stats = tensorProbeStatsFromScratch(b_proj_values);
        delta_stats = tensorProbeStatsFromScratch(scaled_delta_values);
        output_stats = tensorProbeStatsFromScratch(output_values);
        lora_a_reference_error = computeLinearReferenceErrorSample(a_proj_values, input_values, lora_a_values, null, rows, in_dim, actual_rank);
        lora_b_reference_error = computeLinearReferenceErrorSample(b_proj_values, a_proj_values, lora_b_values, null, rows, actual_rank, out_dim);
        delta_reference_error = computeScaleReferenceErrorSample(scaled_delta_values, b_proj_values, scale);
        output_reference_error = computeAddReferenceErrorSample(output_values, base_values, scaled_delta_values);
    }

    try capture_buf.addProjectionDecomposition(.{
        .layer_idx = layer_idx,
        .name = name,
        .input = input_stats,
        .base = base_stats,
        .lora_a = lora_a_stats,
        .lora_b = lora_b_stats,
        .delta = delta_stats,
        .output = output_stats,
        .weight = weight_stats,
        .bias = bias_stats,
        .lora_a_weight = lora_a_weight_stats,
        .lora_b_weight = lora_b_weight_stats,
        .base_reference_error = base_reference_error,
        .lora_a_reference_error = lora_a_reference_error,
        .lora_b_reference_error = lora_b_reference_error,
        .delta_reference_error = delta_reference_error,
        .output_reference_error = output_reference_error,
        .scale = scale,
        .rank = actual_rank,
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .has_bias = base_b != null,
    });
}

fn linearWithLoRA(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    input: CT,
    base_w: CT,
    base_b: CT,
    layer: usize,
    proj_name: []const u8,
    lora_rank: u32,
    lora_alpha: f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !CT {
    return linearWithScopedLoRA(cb, allocator, input, base_w, base_b, layer, "attn", proj_name, lora_rank, lora_alpha, rows, in_dim, out_dim);
}

// ---------------------------------------------------------------------------
// Activation capture types
// ---------------------------------------------------------------------------

/// One captured linear-layer input from the encoder forward pass.
pub const ActivationCapture = struct {
    layer_idx: u32,
    /// Attention module name such as "query_proj", "key_proj", "value_proj", or "out_proj".
    module_name: []const u8,
    /// Owned flat buffer: [total * in_features] in row-major order.
    /// total = batch * seq_len
    input: []f32,
    owns_input: bool = true,
    in_features: usize,
    out_features: usize,
    total: usize, // batch * seq_len

    pub fn deinit(self: *ActivationCapture, allocator: std.mem.Allocator) void {
        if (self.owns_input) allocator.free(self.input);
        self.* = undefined;
    }
};

pub const AttentionInternalCapture = struct {
    layer_idx: u32,
    name: []const u8,
    values: []f32,

    pub fn deinit(self: *AttentionInternalCapture, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        self.* = undefined;
    }
};

pub const LayerStateCapture = struct {
    layer_idx: u32,
    name: []const u8,
    values: []f32,

    pub fn deinit(self: *LayerStateCapture, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        self.* = undefined;
    }
};

pub const TensorProbeStats = struct {
    elems: u64 = 0,
    mean: f32 = 0,
    rms: f32 = 0,
    max_abs: f32 = 0,
    max_abs_index: u64 = 0,
    max_abs_value: f32 = 0,
    hash: u64 = tensor_probe_hash_offset,
    sample_len: u8 = 0,
    sample: [16]f32 = [_]f32{0} ** 16,
    top_abs_len: u8 = 0,
    top_abs_indices: [tensor_probe_top_abs_count]u64 = [_]u64{0} ** tensor_probe_top_abs_count,
    top_abs_values: [tensor_probe_top_abs_count]f32 = [_]f32{0} ** tensor_probe_top_abs_count,
};

fn hashProbeF32(hash: *u64, value: f32) void {
    var bits: u32 = @bitCast(value);
    for (0..4) |_| {
        hash.* ^= bits & 0xff;
        hash.* *%= tensor_probe_hash_prime;
        bits >>= 8;
    }
}

fn addTensorProbeTopAbs(stats: *TensorProbeStats, index: usize, value: f32) void {
    const ax = @abs(value);
    if (ax > stats.max_abs) {
        stats.max_abs = ax;
        stats.max_abs_index = @intCast(index);
        stats.max_abs_value = value;
    }

    var insert_at: usize = undefined;
    if (stats.top_abs_len < stats.top_abs_values.len) {
        insert_at = stats.top_abs_len;
        stats.top_abs_len += 1;
    } else if (ax > @abs(stats.top_abs_values[stats.top_abs_values.len - 1])) {
        insert_at = stats.top_abs_values.len - 1;
    } else {
        return;
    }

    while (insert_at > 0 and ax > @abs(stats.top_abs_values[insert_at - 1])) : (insert_at -= 1) {
        stats.top_abs_values[insert_at] = stats.top_abs_values[insert_at - 1];
        stats.top_abs_indices[insert_at] = stats.top_abs_indices[insert_at - 1];
    }
    stats.top_abs_values[insert_at] = value;
    stats.top_abs_indices[insert_at] = @intCast(index);
}

pub fn computeTensorProbeStats(values: []const f32) TensorProbeStats {
    var out = TensorProbeStats{
        .elems = @intCast(values.len),
    };
    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var finite_count: u64 = 0;
    for (values, 0..) |value, i| {
        if (i < out.sample.len) {
            out.sample[i] = value;
            out.sample_len = @intCast(i + 1);
        }
        hashProbeF32(&out.hash, value);
        if (!std.math.isFinite(value)) continue;
        const v: f64 = @floatCast(value);
        sum += v;
        sum_sq += v * v;
        addTensorProbeTopAbs(&out, i, value);
        finite_count += 1;
    }
    if (finite_count > 0) {
        const denom = @as(f64, @floatFromInt(finite_count));
        out.mean = @floatCast(sum / denom);
        out.rms = @floatCast(@sqrt(sum_sq / denom));
    }
    return out;
}

pub fn computeZeroTensorProbeStats(elems: usize) TensorProbeStats {
    var out = TensorProbeStats{
        .elems = @intCast(elems),
        .sample_len = @intCast(@min(elems, 16)),
    };
    for (0..elems) |_| hashProbeF32(&out.hash, 0);
    return out;
}

fn tensorProbeStatsFromScratch(values: []const f32) TensorProbeStats {
    return computeTensorProbeStats(values);
}

fn computeLinearReferenceErrorSample(
    actual: []const f32,
    input: []const f32,
    weight: []const f32,
    bias: ?[]const f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) TensorProbeStats {
    if (input.len != rows * in_dim or weight.len != out_dim * in_dim or actual.len < rows * out_dim) return .{};
    if (bias) |b| {
        if (b.len < out_dim) return .{};
    }
    var errors: [16]f32 = [_]f32{0} ** 16;
    const count = @min(errors.len, actual.len);
    for (0..count) |flat_idx| {
        const row = flat_idx / out_dim;
        const col = flat_idx % out_dim;
        var ref: f32 = if (bias) |b| b[col] else 0.0;
        const input_row = input[row * in_dim ..][0..in_dim];
        const weight_row = weight[col * in_dim ..][0..in_dim];
        for (input_row, weight_row) |x, w| {
            ref += x * w;
        }
        errors[flat_idx] = actual[flat_idx] - ref;
    }
    return tensorProbeStatsFromScratch(errors[0..count]);
}

fn computeScaleReferenceErrorSample(
    actual: []const f32,
    input: []const f32,
    scale: f32,
) TensorProbeStats {
    var errors: [16]f32 = [_]f32{0} ** 16;
    const count = @min(@min(errors.len, actual.len), input.len);
    for (0..count) |i| {
        errors[i] = actual[i] - input[i] * scale;
    }
    return tensorProbeStatsFromScratch(errors[0..count]);
}

fn computeAddReferenceErrorSample(
    actual: []const f32,
    lhs: []const f32,
    rhs: []const f32,
) TensorProbeStats {
    var errors: [16]f32 = [_]f32{0} ** 16;
    const count = @min(@min(@min(errors.len, actual.len), lhs.len), rhs.len);
    for (0..count) |i| {
        errors[i] = actual[i] - (lhs[i] + rhs[i]);
    }
    return tensorProbeStatsFromScratch(errors[0..count]);
}

pub const ProjectionDecompositionCapture = struct {
    layer_idx: u32,
    name: []const u8,
    input: TensorProbeStats = .{},
    base: TensorProbeStats = .{},
    lora_a: TensorProbeStats = .{},
    lora_b: TensorProbeStats = .{},
    delta: TensorProbeStats = .{},
    output: TensorProbeStats = .{},
    weight: TensorProbeStats = .{},
    bias: TensorProbeStats = .{},
    lora_a_weight: TensorProbeStats = .{},
    lora_b_weight: TensorProbeStats = .{},
    base_reference_error: TensorProbeStats = .{},
    lora_a_reference_error: TensorProbeStats = .{},
    lora_b_reference_error: TensorProbeStats = .{},
    delta_reference_error: TensorProbeStats = .{},
    output_reference_error: TensorProbeStats = .{},
    scale: f32 = 0,
    rank: usize = 0,
    rows: usize = 0,
    in_dim: usize = 0,
    out_dim: usize = 0,
    has_bias: bool = false,
};

/// Buffer of ActivationCapture records from one forward pass.
pub const ActivationBuffer = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayListUnmanaged(ActivationCapture),
    capture_attention_internal_layer: ?u32 = null,
    capture_projection_decomposition_layer: ?u32 = null,
    capture_layer_state_layer: ?u32 = null,
    attention_internals: std.ArrayListUnmanaged(AttentionInternalCapture) = .empty,
    layer_states: std.ArrayListUnmanaged(LayerStateCapture) = .empty,
    projection_decompositions: std.ArrayListUnmanaged(ProjectionDecompositionCapture) = .empty,
    embedding_lookup: ?[]f32 = null,
    embedding_output: ?[]f32 = null,
    embedding_total: usize = 0,
    embedding_hidden: usize = 0,
    final_norm_input: ?[]f32 = null,
    final_norm_weight: ?[]f32 = null,
    final_norm_total: usize = 0,
    final_norm_hidden: usize = 0,
    layer_inputs: std.ArrayListUnmanaged(LayerInputCapture) = .empty,

    pub fn init(allocator: std.mem.Allocator) ActivationBuffer {
        return .{ .allocator = allocator, .items = .empty };
    }

    pub fn deinit(self: *ActivationBuffer) void {
        for (self.items.items) |*cap| cap.deinit(self.allocator);
        self.items.deinit(self.allocator);
        for (self.attention_internals.items) |*cap| cap.deinit(self.allocator);
        self.attention_internals.deinit(self.allocator);
        for (self.layer_states.items) |*cap| cap.deinit(self.allocator);
        self.layer_states.deinit(self.allocator);
        self.projection_decompositions.deinit(self.allocator);
        if (self.embedding_lookup) |buf| self.allocator.free(buf);
        if (self.embedding_output) |buf| self.allocator.free(buf);
        if (self.final_norm_input) |buf| self.allocator.free(buf);
        if (self.final_norm_weight) |buf| self.allocator.free(buf);
        for (self.layer_inputs.items) |*cap| cap.deinit(self.allocator);
        self.layer_inputs.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn add(
        self: *ActivationBuffer,
        layer_idx: u32,
        module_name: []const u8,
        input_f32: []const f32,
        in_features: usize,
        out_features: usize,
        total: usize,
    ) !void {
        const owned = try self.allocator.dupe(f32, input_f32);
        errdefer self.allocator.free(owned);
        try self.items.append(self.allocator, .{
            .layer_idx = layer_idx,
            .module_name = module_name,
            .input = owned,
            .owns_input = true,
            .in_features = in_features,
            .out_features = out_features,
            .total = total,
        });
    }

    pub fn addAttentionInternal(
        self: *ActivationBuffer,
        layer_idx: u32,
        name: []const u8,
        values: []const f32,
    ) !void {
        const owned = try self.allocator.dupe(f32, values);
        errdefer self.allocator.free(owned);
        try self.attention_internals.append(self.allocator, .{
            .layer_idx = layer_idx,
            .name = name,
            .values = owned,
        });
    }

    pub fn addLayerState(
        self: *ActivationBuffer,
        layer_idx: u32,
        name: []const u8,
        values: []const f32,
    ) !void {
        const owned = try self.allocator.dupe(f32, values);
        errdefer self.allocator.free(owned);
        try self.layer_states.append(self.allocator, .{
            .layer_idx = layer_idx,
            .name = name,
            .values = owned,
        });
    }

    pub fn addProjectionDecomposition(
        self: *ActivationBuffer,
        capture: ProjectionDecompositionCapture,
    ) !void {
        try self.projection_decompositions.append(self.allocator, capture);
    }

    pub fn addAlias(
        self: *ActivationBuffer,
        layer_idx: u32,
        module_name: []const u8,
        input_f32: []f32,
        in_features: usize,
        out_features: usize,
        total: usize,
    ) !void {
        try self.items.append(self.allocator, .{
            .layer_idx = layer_idx,
            .module_name = module_name,
            .input = input_f32,
            .owns_input = false,
            .in_features = in_features,
            .out_features = out_features,
            .total = total,
        });
    }

    pub fn setEmbeddingOwned(
        self: *ActivationBuffer,
        lookup: []f32,
        output: []f32,
        total: usize,
        hidden: usize,
    ) !void {
        if (lookup.len != total * hidden or output.len != total * hidden) return error.InvalidEmbeddingCaptureShape;
        if (self.embedding_lookup) |old| self.allocator.free(old);
        if (self.embedding_output) |old| self.allocator.free(old);
        self.embedding_lookup = lookup;
        self.embedding_output = output;
        self.embedding_total = total;
        self.embedding_hidden = hidden;
    }

    pub fn setFinalNormOwned(
        self: *ActivationBuffer,
        input: []f32,
        weight: []f32,
        total: usize,
        hidden: usize,
    ) !void {
        if (input.len != total * hidden or weight.len != hidden) return error.InvalidFinalNormCaptureShape;
        if (self.final_norm_input) |old| self.allocator.free(old);
        if (self.final_norm_weight) |old| self.allocator.free(old);
        self.final_norm_input = input;
        self.final_norm_weight = weight;
        self.final_norm_total = total;
        self.final_norm_hidden = hidden;
    }

    pub fn addLayerInput(
        self: *ActivationBuffer,
        layer_idx: u32,
        input_f32: []const f32,
        total: usize,
        hidden: usize,
    ) !void {
        if (input_f32.len != total * hidden) return error.InvalidLayerInputCaptureShape;
        const owned = try self.allocator.dupe(f32, input_f32);
        errdefer self.allocator.free(owned);
        try self.layer_inputs.append(self.allocator, .{
            .layer_idx = layer_idx,
            .input = owned,
            .total = total,
            .hidden = hidden,
        });
    }

    pub fn findLayerInput(self: *const ActivationBuffer, layer_idx: u32) ?*const LayerInputCapture {
        for (self.layer_inputs.items) |*cap| {
            if (cap.layer_idx == layer_idx) return cap;
        }
        return null;
    }

    pub fn findLayerState(self: *const ActivationBuffer, layer_idx: u32, name: []const u8) ?[]const f32 {
        for (self.layer_states.items) |*cap| {
            if (cap.layer_idx == layer_idx and std.mem.eql(u8, cap.name, name)) return cap.values;
        }
        return null;
    }
};

pub const LayerInputCapture = struct {
    layer_idx: u32,
    input: []f32,
    total: usize,
    hidden: usize,

    fn deinit(self: *LayerInputCapture, allocator: std.mem.Allocator) void {
        allocator.free(self.input);
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Activation-capturing forward pass
// ---------------------------------------------------------------------------

/// Like `forward` but also captures the inputs to attention LoRA projections
/// in each layer into `captures`.  The returned f32 slice is owned by the caller.
pub fn forwardCapturingActivations(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    active_layers: ?[]const bool,
    capture_layer_inputs: bool,
    capture_all_layer_inputs: bool,
    captures: *ActivationBuffer,
) ![]f32 {
    const trace_timing = layerTimingEnabled();
    const total_start_ns = if (trace_timing) monotonicNowNs() else 0;
    var stage_start_ns = total_start_ns;
    const result_ct = try forwardCapturingActivationsCT(
        cb,
        allocator,
        config,
        input_ids,
        attention_mask,
        batch,
        seq_len,
        active_layers,
        capture_layer_inputs,
        capture_all_layer_inputs,
        captures,
    );
    defer cb.free(result_ct);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_forward_ct", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    const out = try cb.toFloat32(result_ct, allocator);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_output_to_float32", elapsedNs(stage_start_ns));
        printModernBertTiming(std.math.maxInt(usize), "capture_forward_total", elapsedNs(total_start_ns));
    }
    return out;
}

fn shouldCaptureLoRALayer(active_layers: ?[]const bool, layer_idx: usize) bool {
    const layers = active_layers orelse return true;
    return layer_idx < layers.len and layers[layer_idx];
}

fn forwardCapturingActivationsCT(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    active_layers: ?[]const bool,
    capture_layer_inputs: bool,
    capture_all_layer_inputs: bool,
    captures: *ActivationBuffer,
) !CT {
    const total_tokens = batch * seq_len;
    const H: usize = @intCast(config.hidden_size);
    const trace_timing = layerTimingEnabled();
    var stage_start_ns = if (trace_timing) monotonicNowNs() else 0;

    // Collect normed_attn CTs from all layers without downloading them yet.
    // This lets us batch-evaluate all 22 tensors in one GPU sync (one Metal
    // command buffer submission on device backends) instead of one per layer.
    var normed_attn_cts = std.ArrayListUnmanaged(CT).empty;
    defer {
        for (normed_attn_cts.items) |ct| cb.free(ct);
        normed_attn_cts.deinit(allocator);
    }
    var attn_out_cts = std.ArrayListUnmanaged(CT).empty;
    defer {
        for (attn_out_cts.items) |ct| cb.free(ct);
        attn_out_cts.deinit(allocator);
    }
    var captured_layer_indices = std.ArrayListUnmanaged(u32).empty;
    defer captured_layer_indices.deinit(allocator);
    var layer_input_cts = std.ArrayListUnmanaged(CT).empty;
    defer {
        for (layer_input_cts.items) |ct| cb.free(ct);
        layer_input_cts.deinit(allocator);
    }
    var captured_layer_input_indices = std.ArrayListUnmanaged(u32).empty;
    defer captured_layer_input_indices.deinit(allocator);

    var hidden = try embeddingsBlock(cb, allocator, config, input_ids, total_tokens, seq_len, captures);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_embeddings", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    // Free hidden on any error path; the happy path frees it explicitly below.
    errdefer cb.free(hidden);
    var local_window_bias: ?CT = null;
    defer if (local_window_bias) |bias| cb.free(bias);

    for (0..config.num_hidden_layers) |layer_idx| {
        const capture_layer = shouldCaptureLoRALayer(active_layers, layer_idx);
        const capture_layer_input = capture_layer_inputs and (capture_all_layer_inputs or capture_layer);
        if (capture_layer_input) {
            if (try cb.cloneTensorShape(hidden, &.{ @intCast(total_tokens), @intCast(H) })) |layer_input_ct| {
                layer_input_cts.append(allocator, layer_input_ct) catch |err| {
                    cb.free(layer_input_ct);
                    return err;
                };
                captured_layer_input_indices.append(allocator, @intCast(layer_idx)) catch |err| {
                    _ = layer_input_cts.pop();
                    cb.free(layer_input_ct);
                    return err;
                };
            } else {
                const layer_input = try cb.toFloat32(hidden, allocator);
                defer allocator.free(layer_input);
                try captures.addLayerInput(@intCast(layer_idx), layer_input, total_tokens, H);
            }
        }
        const layer_result = try encoderLayerWithNormedAttn(
            cb,
            allocator,
            config,
            hidden,
            attention_mask,
            batch,
            seq_len,
            layer_idx,
            &local_window_bias,
            if (capture_layer) captures else null,
        );
        cb.free(hidden);
        hidden = layer_result.hidden;
        if (!capture_layer) {
            cb.free(layer_result.normed_attn);
            cb.free(layer_result.attn_out);
            continue;
        }
        // Transfer ownership of normed_attn to the list.  On append failure,
        // free it immediately before propagating the error.
        normed_attn_cts.append(allocator, layer_result.normed_attn) catch |err| {
            cb.free(layer_result.normed_attn);
            cb.free(layer_result.attn_out);
            return err;
        };
        attn_out_cts.append(allocator, layer_result.attn_out) catch |err| {
            cb.free(layer_result.attn_out);
            return err;
        };
        captured_layer_indices.append(allocator, @intCast(layer_idx)) catch |err| {
            _ = normed_attn_cts.pop();
            _ = attn_out_cts.pop();
            cb.free(layer_result.normed_attn);
            cb.free(layer_result.attn_out);
            return err;
        };
    }
    if (trace_timing) stage_start_ns = monotonicNowNs();

    // Batch-download all normed_attn tensors — single GPU sync on Metal.
    const batch_results = try cb.toFloat32Batch(normed_attn_cts.items, allocator);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_normed_attn_download", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    defer {
        for (batch_results) |r| allocator.free(r);
        allocator.free(batch_results);
    }
    const attn_out_results = try cb.toFloat32Batch(attn_out_cts.items, allocator);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_attn_out_download", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    defer {
        for (attn_out_results) |r| allocator.free(r);
        allocator.free(attn_out_results);
    }

    // Populate captures from the downloaded data.
    for (captured_layer_indices.items, 0..) |captured_layer_idx, capture_idx| {
        const normed_f32 = batch_results[capture_idx];
        try captures.add(captured_layer_idx, "query_proj", normed_f32, H, H, total_tokens);
        const shared_normed = captures.items.items[captures.items.items.len - 1].input;
        try captures.addAlias(captured_layer_idx, "key_proj", shared_normed, H, H, total_tokens);
        try captures.addAlias(captured_layer_idx, "value_proj", shared_normed, H, H, total_tokens);
        try captures.add(captured_layer_idx, "out_proj", attn_out_results[capture_idx], H, H, total_tokens);
    }

    if (layer_input_cts.items.len > 0) {
        const layer_input_results = try cb.toFloat32Batch(layer_input_cts.items, allocator);
        if (trace_timing) {
            printModernBertTiming(std.math.maxInt(usize), "capture_layer_input_download", elapsedNs(stage_start_ns));
            stage_start_ns = monotonicNowNs();
        }
        defer {
            for (layer_input_results) |r| allocator.free(r);
            allocator.free(layer_input_results);
        }
        for (captured_layer_input_indices.items, 0..) |captured_layer_idx, capture_idx| {
            try captures.addLayerInput(captured_layer_idx, layer_input_results[capture_idx], total_tokens, H);
        }
    }

    // Final layer norm (same as forwardCT)
    var name_buf: [128]u8 = undefined;
    const fn_w = try cb.getWeight(std.fmt.bufPrint(&name_buf, "model.final_norm.weight", .{}) catch return error.NameTooLong);
    defer cb.free(fn_w);
    const fn_b = try getWeightOrZeroBias(cb, allocator, std.fmt.bufPrint(&name_buf, "model.final_norm.bias", .{}) catch return error.NameTooLong, @intCast(config.hidden_size));
    defer cb.free(fn_b);
    const normed_final = try cb.layerNorm(hidden, fn_w, fn_b, @intCast(config.hidden_size), config.layer_norm_eps);
    errdefer cb.free(normed_final);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_final_norm", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    const final_norm_input = try cb.toFloat32(hidden, allocator);
    var owns_final_norm_input = true;
    errdefer if (owns_final_norm_input) allocator.free(final_norm_input);
    const final_norm_weight = try cb.toFloat32(fn_w, allocator);
    var owns_final_norm_weight = true;
    errdefer if (owns_final_norm_weight) allocator.free(final_norm_weight);
    if (trace_timing) {
        printModernBertTiming(std.math.maxInt(usize), "capture_final_norm_download", elapsedNs(stage_start_ns));
    }
    try captures.setFinalNormOwned(final_norm_input, final_norm_weight, total_tokens, H);
    owns_final_norm_input = false;
    owns_final_norm_weight = false;
    cb.free(hidden);
    return normed_final;
}

// ---------------------------------------------------------------------------
// Encoder layer variant that returns the pre-attention normed hidden state
// as an owned CT alongside the layer output.  Used by the batched activation
// capture path so we can defer all GPU→CPU downloads to a single eval call.
// ---------------------------------------------------------------------------

const LayerWithNormedAttn = struct {
    /// The updated hidden state for the next encoder layer.  Caller owns it.
    hidden: CT,
    /// The pre-attention LayerNorm output (normed_attn) for this layer.
    /// Caller owns it; NOT freed inside this function.
    normed_attn: CT,
    /// The attention output before the output projection. Caller owns it.
    attn_out: CT,
};

/// Like encoderLayer, but returns normed_attn as a second CT instead of
/// immediately freeing it.  The rest of the layer runs normally so that the
/// returned hidden state is correct.  The caller is responsible for freeing
/// both returned CTs.
fn encoderLayerWithNormedAttn(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden: CT,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    layer_idx: usize,
    local_window_bias_cache: ?*?CT,
    captures: ?*ActivationBuffer,
) !LayerWithNormedAttn {
    const trace_timing = layerTimingEnabled();
    const layer_start_ns = if (trace_timing) monotonicNowNs() else 0;
    var stage_start_ns = layer_start_ns;
    const H: usize = @intCast(config.hidden_size);
    const num_heads: usize = @intCast(config.num_attention_heads);
    const head_dim = H / num_heads;
    const intermediate: usize = @intCast(config.intermediate_size);
    const total = batch * seq_len;

    const is_global = (layer_idx % @as(usize, @intCast(config.global_attn_every_n_layers))) == 0;
    const rope_theta = if (is_global) config.global_rope_theta else config.local_rope_theta;

    var name_buf: [256]u8 = undefined;

    // -----------------------------------------------------------------------
    // Self-attention sub-layer  (pre-norm)
    // -----------------------------------------------------------------------

    // Pre-attention LayerNorm — NOT deferred; ownership returned to caller.
    const normed_attn = try preAttentionNorm(cb, allocator, config, hidden, layer_idx, total, H, &name_buf);
    errdefer cb.free(normed_attn);
    // NOTE: no `defer cb.free(normed_attn)` here — returned to caller.
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_pre_attn_norm", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Q projection — use linearLoRA if LoRA is enabled in the config.
    const q_w = try getLayerWeight(cb, layer_idx, "attn.query_proj.weight", &name_buf);
    defer cb.free(q_w);
    const q_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.query_proj.bias", H, &name_buf);
    defer cb.free(q_b);
    const Q_raw = try linearWithLoRA(cb, allocator, normed_attn, q_w, q_b, layer_idx, "query_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(Q_raw);

    // K projection — use linearLoRA if LoRA is enabled in the config.
    const k_w = try getLayerWeight(cb, layer_idx, "attn.key_proj.weight", &name_buf);
    defer cb.free(k_w);
    const k_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.key_proj.bias", H, &name_buf);
    defer cb.free(k_b);
    const K_raw = try linearWithLoRA(cb, allocator, normed_attn, k_w, k_b, layer_idx, "key_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(K_raw);

    // V projection — use linearLoRA if LoRA is enabled in the config.
    const v_w = try getLayerWeight(cb, layer_idx, "attn.value_proj.weight", &name_buf);
    defer cb.free(v_w);
    const v_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.value_proj.bias", H, &name_buf);
    defer cb.free(v_b);
    const V = try linearWithLoRA(cb, allocator, normed_attn, v_w, v_b, layer_idx, "value_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(V);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_qkv_project", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    const should_capture_projection_decomposition = if (captures) |capture_buf|
        capture_buf.capture_projection_decomposition_layer != null and capture_buf.capture_projection_decomposition_layer.? == layer_idx
    else
        false;
    if (should_capture_projection_decomposition) {
        const capture_buf = captures.?;
        try captureProjectionDecomposition(cb, allocator, capture_buf, @intCast(layer_idx), "query_proj", normed_attn, q_w, q_b, "attn", "query_proj", config.lora_rank, config.lora_alpha, total, H, H);
        try captureProjectionDecomposition(cb, allocator, capture_buf, @intCast(layer_idx), "key_proj", normed_attn, k_w, k_b, "attn", "key_proj", config.lora_rank, config.lora_alpha, total, H, H);
        try captureProjectionDecomposition(cb, allocator, capture_buf, @intCast(layer_idx), "value_proj", normed_attn, v_w, v_b, "attn", "value_proj", config.lora_rank, config.lora_alpha, total, H, H);
    }

    const should_capture_attention_internals = if (captures) |capture_buf|
        capture_buf.capture_attention_internal_layer != null and capture_buf.capture_attention_internal_layer.? == layer_idx
    else
        false;
    if (should_capture_attention_internals) {
        const capture_buf = captures.?;
        const q_raw_f32 = try cb.toFloat32(Q_raw, allocator);
        defer allocator.free(q_raw_f32);
        try capture_buf.addAttentionInternal(@intCast(layer_idx), "q_raw", q_raw_f32);
        const k_raw_f32 = try cb.toFloat32(K_raw, allocator);
        defer allocator.free(k_raw_f32);
        try capture_buf.addAttentionInternal(@intCast(layer_idx), "k_raw", k_raw_f32);
        const v_raw_f32 = try cb.toFloat32(V, allocator);
        defer allocator.free(v_raw_f32);
        try capture_buf.addAttentionInternal(@intCast(layer_idx), "v_raw", v_raw_f32);
    }

    // RoPE
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(K);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_rope", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    if (should_capture_attention_internals) {
        const capture_buf = captures.?;
        const q_rope_f32 = try cb.toFloat32(Q, allocator);
        defer allocator.free(q_rope_f32);
        try capture_buf.addAttentionInternal(@intCast(layer_idx), "q_rope", q_rope_f32);
        const k_rope_f32 = try cb.toFloat32(K, allocator);
        defer allocator.free(k_rope_f32);
        try capture_buf.addAttentionInternal(@intCast(layer_idx), "k_rope", k_rope_f32);
    }

    // Bidirectional scaled dot-product attention.
    const attn_out = try modernBertAttention(
        cb,
        allocator,
        config,
        Q,
        K,
        V,
        attention_mask,
        batch,
        seq_len,
        num_heads,
        head_dim,
        is_global,
        local_window_bias_cache,
    );
    errdefer cb.free(attn_out);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_attention", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    // Output projection
    const out_w = try getLayerWeight(cb, layer_idx, "attn.Wo.weight", &name_buf);
    defer cb.free(out_w);
    const out_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.Wo.bias", H, &name_buf);
    defer cb.free(out_b);
    const attn_proj = try linearWithLoRA(cb, allocator, attn_out, out_w, out_b, layer_idx, "Wo", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(attn_proj);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_attn_output_project", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    if (should_capture_projection_decomposition) {
        try captureProjectionDecomposition(cb, allocator, captures.?, @intCast(layer_idx), "out_proj", attn_out, out_w, out_b, "attn", "Wo", config.lora_rank, config.lora_alpha, total, H, H);
    }

    // Residual
    const hidden_after_attn = try cb.add(attn_proj, hidden);
    defer cb.free(hidden_after_attn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_attn_residual", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    if (captures) |capture_buf| {
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            const hidden_after_attn_f32 = try cb.toFloat32(hidden_after_attn, allocator);
            defer allocator.free(hidden_after_attn_f32);
            try capture_buf.addLayerState(@intCast(layer_idx), "hidden_after_attn", hidden_after_attn_f32);
        }
    }

    // -----------------------------------------------------------------------
    // FFN sub-layer  (pre-norm, GeGLU)
    // -----------------------------------------------------------------------

    const mlp_ln_w = try getLayerWeight(cb, layer_idx, "mlp_norm.weight", &name_buf);
    defer cb.free(mlp_ln_w);
    const mlp_ln_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "mlp_norm.bias", H, &name_buf);
    defer cb.free(mlp_ln_b);
    const normed_ffn = try cb.layerNorm(hidden_after_attn, mlp_ln_w, mlp_ln_b, H, config.layer_norm_eps);
    defer cb.free(normed_ffn);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_ffn_norm", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }
    if (captures) |capture_buf| {
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            const normed_ffn_f32 = try cb.toFloat32(normed_ffn, allocator);
            defer allocator.free(normed_ffn_f32);
            try capture_buf.addLayerState(@intCast(layer_idx), "mlp_norm_output", normed_ffn_f32);
        }
    }

    const Wi_w = try getLayerWeight(cb, layer_idx, "mlp.Wi.weight", &name_buf);
    defer cb.free(Wi_w);
    const Wo_w = try getLayerWeight(cb, layer_idx, "mlp.Wo.weight", &name_buf);
    defer cb.free(Wo_w);

    const ffn_out = try geGluFfn(cb, allocator, normed_ffn, Wi_w, Wo_w, layer_idx, config.lora_rank, config.lora_alpha, total, H, intermediate, captures);
    defer cb.free(ffn_out);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_geglu", elapsedNs(stage_start_ns));
        stage_start_ns = monotonicNowNs();
    }

    const layer_output = try cb.add(ffn_out, hidden_after_attn);
    errdefer cb.free(layer_output);
    if (trace_timing) {
        printModernBertTiming(layer_idx, "capture_ffn_residual", elapsedNs(stage_start_ns));
        printModernBertTiming(layer_idx, "capture_total", elapsedNs(layer_start_ns));
    }
    if (captures) |capture_buf| {
        if (capture_buf.capture_layer_state_layer != null and capture_buf.capture_layer_state_layer.? == layer_idx) {
            const layer_output_f32 = try cb.toFloat32(layer_output, allocator);
            defer allocator.free(layer_output_f32);
            try capture_buf.addLayerState(@intCast(layer_idx), "layer_output", layer_output_f32);
        }
    }

    return .{
        .hidden = layer_output,
        .normed_attn = normed_attn,
        .attn_out = attn_out,
    };
}

fn encoderLayerCapturing(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden: CT,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    layer_idx: usize,
    captures: *ActivationBuffer,
) !CT {
    const H: usize = @intCast(config.hidden_size);
    const num_heads: usize = @intCast(config.num_attention_heads);
    const head_dim = H / num_heads;
    const intermediate: usize = @intCast(config.intermediate_size);
    const total = batch * seq_len;

    // Layers 0, 3, 6, … use full (global) attention; all others are local.
    const is_global = (layer_idx % @as(usize, @intCast(config.global_attn_every_n_layers))) == 0;
    const rope_theta = if (is_global) config.global_rope_theta else config.local_rope_theta;

    var name_buf: [256]u8 = undefined;

    // -----------------------------------------------------------------------
    // Self-attention sub-layer  (pre-norm)
    // -----------------------------------------------------------------------

    const normed_attn = try preAttentionNorm(cb, allocator, config, hidden, layer_idx, total, H, &name_buf);
    defer cb.free(normed_attn);

    // Capture normed_attn as the shared input to Q/K/V projections.
    const normed_attn_f32 = try cb.toFloat32(normed_attn, allocator);
    defer allocator.free(normed_attn_f32);
    try captures.add(@intCast(layer_idx), "query_proj", normed_attn_f32, H, H, total);
    const shared_normed = captures.items.items[captures.items.items.len - 1].input;
    try captures.addAlias(@intCast(layer_idx), "key_proj", shared_normed, H, H, total);
    try captures.addAlias(@intCast(layer_idx), "value_proj", shared_normed, H, H, total);

    // Q projection — use linearLoRA if LoRA is enabled in the config.
    const q_w = try getLayerWeight(cb, layer_idx, "attn.query_proj.weight", &name_buf);
    defer cb.free(q_w);
    const q_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.query_proj.bias", H, &name_buf);
    defer cb.free(q_b);
    const Q_raw = try linearWithLoRA(cb, allocator, normed_attn, q_w, q_b, layer_idx, "query_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(Q_raw);

    // K projection — use linearLoRA if LoRA is enabled in the config.
    const k_w = try getLayerWeight(cb, layer_idx, "attn.key_proj.weight", &name_buf);
    defer cb.free(k_w);
    const k_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.key_proj.bias", H, &name_buf);
    defer cb.free(k_b);
    const K_raw = try linearWithLoRA(cb, allocator, normed_attn, k_w, k_b, layer_idx, "key_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(K_raw);

    // V projection — use linearLoRA if LoRA is enabled in the config.
    const v_w = try getLayerWeight(cb, layer_idx, "attn.value_proj.weight", &name_buf);
    defer cb.free(v_w);
    const v_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.value_proj.bias", H, &name_buf);
    defer cb.free(v_b);
    const V = try linearWithLoRA(cb, allocator, normed_attn, v_w, v_b, layer_idx, "value_proj", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(V);

    // Apply RoPE to Q and K.
    // RoPE pairing is config.rope_interleaved: interleaved (2j,2j+1) for the
    // gopeft-trained fused chunker, or HF rotate_half (j, j+head_dim/2) for the
    // stock nomic modernbert-embed-base checkpoint.
    // rope_dim == head_dim: the full head dimension is rotated.
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, config.rope_interleaved);
    defer cb.free(K);

    // Bidirectional scaled dot-product attention (encoder, no causal mask).
    // The padding mask (attention_mask) is consumed by the backend: positions
    // where mask[b*seq_len + ki] == 0 are set to -inf before softmax.
    const attn_out = try modernBertAttention(
        cb,
        allocator,
        config,
        Q,
        K,
        V,
        attention_mask,
        batch,
        seq_len,
        num_heads,
        head_dim,
        is_global,
        null,
    );
    defer cb.free(attn_out);

    const attn_out_f32 = try cb.toFloat32(attn_out, allocator);
    defer allocator.free(attn_out_f32);
    try captures.add(@intCast(layer_idx), "out_proj", attn_out_f32, H, H, total);

    // Output projection
    const out_w = try getLayerWeight(cb, layer_idx, "attn.Wo.weight", &name_buf);
    defer cb.free(out_w);
    const out_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.Wo.bias", H, &name_buf);
    defer cb.free(out_b);
    const attn_proj = try linearWithLoRA(cb, allocator, attn_out, out_w, out_b, layer_idx, "Wo", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(attn_proj);

    // Residual: add the projected attention output to the *original* (pre-norm)
    // hidden state — pre-norm residual pattern.
    const hidden_after_attn = try cb.add(attn_proj, hidden);
    defer cb.free(hidden_after_attn);

    // -----------------------------------------------------------------------
    // FFN sub-layer  (pre-norm, GeGLU)
    // -----------------------------------------------------------------------

    // Pre-FFN LayerNorm
    const mlp_ln_w = try getLayerWeight(cb, layer_idx, "mlp_norm.weight", &name_buf);
    defer cb.free(mlp_ln_w);
    const mlp_ln_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "mlp_norm.bias", H, &name_buf);
    defer cb.free(mlp_ln_b);
    const normed_ffn = try cb.layerNorm(hidden_after_attn, mlp_ln_w, mlp_ln_b, H, config.layer_norm_eps);
    defer cb.free(normed_ffn);

    // GeGLU feed-forward (Wi and Wo both have no bias in ModernBERT's MLP)
    const Wi_w = try getLayerWeight(cb, layer_idx, "mlp.Wi.weight", &name_buf);
    defer cb.free(Wi_w);
    const Wo_w = try getLayerWeight(cb, layer_idx, "mlp.Wo.weight", &name_buf);
    defer cb.free(Wo_w);

    const ffn_out = try geGluFfn(cb, allocator, normed_ffn, Wi_w, Wo_w, layer_idx, config.lora_rank, config.lora_alpha, total, H, intermediate, captures);
    defer cb.free(ffn_out);

    // Residual: add FFN output to post-attention hidden state
    return cb.add(ffn_out, hidden_after_attn);
}

test "ModernBERT NEFTune scale uses per-sequence length like Go" {
    const alpha: f32 = 5.0;
    const seq_len: usize = 384;
    const hidden: usize = 768;
    const expected = alpha / @sqrt(@as(f32, @floatFromInt(seq_len * hidden)));

    try std.testing.expectApproxEqAbs(expected, neftuneNoiseScale(alpha, seq_len, hidden), 1e-8);

    const batch: usize = 8;
    const wrong_batch_scaled = alpha / @sqrt(@as(f32, @floatFromInt(batch * seq_len * hidden)));
    try std.testing.expect(neftuneNoiseScale(alpha, seq_len, hidden) > wrong_batch_scaled);
}
