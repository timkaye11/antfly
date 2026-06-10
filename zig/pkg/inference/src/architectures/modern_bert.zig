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
// Single implementation works with any ComputeBackend (BLAS, MLX, etc).

const std = @import("std");
const ops = @import("../ops/ops.zig");
const CT = ops.CT;
const ComputeBackend = ops.ComputeBackend;

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
    // 1. Token embeddings + embedding LayerNorm.
    //    ModernBERT has no absolute position embeddings; RoPE is applied in each
    //    attention layer instead.
    var hidden = try embeddingsBlock(cb, allocator, config, input_ids, batch * seq_len, seq_len);
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

    // 3. Final layer norm
    var name_buf: [128]u8 = undefined;
    const fn_w = try cb.getWeight(std.fmt.bufPrint(&name_buf, "model.final_norm.weight", .{}) catch return error.NameTooLong);
    defer cb.free(fn_w);
    const fn_b = try getWeightOrZeroBias(cb, allocator, std.fmt.bufPrint(&name_buf, "model.final_norm.bias", .{}) catch return error.NameTooLong, @intCast(config.hidden_size));
    defer cb.free(fn_b);

    const normed_final = try cb.layerNorm(hidden, fn_w, fn_b, @intCast(config.hidden_size), config.layer_norm_eps);
    cb.free(hidden);
    return normed_final;
}

// ---------------------------------------------------------------------------
// Embeddings block
// ---------------------------------------------------------------------------

fn embeddingsBlock(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    total: usize,
    seq_len: usize,
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
    // consecutive_pairs=true: ModernBERT uses interleaved rotation pairs
    // (matching gopeft fused_chunker_embedder.go convention).
    // rope_dim == head_dim: the full head dimension is rotated.
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
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
        local_window_bias_cache,
    );
    defer cb.free(attn_out);

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

    const ffn_out = try geGluFfn(cb, allocator, normed_ffn, Wi_w, Wo_w, layer_idx, config.lora_rank, config.lora_alpha, total, H, intermediate, null);
    defer cb.free(ffn_out);

    // Residual: add FFN output to post-attention hidden state
    return cb.add(ffn_out, hidden_after_attn);
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
// GELU, and multiply to resident device ops, which avoids a per-layer
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

    const activated_ct = (try cb.activationMultiply(gate_ct, value_ct, .gelu_new, total, intermediate_size)) orelse blk: {
        const gate_gelu_ct = try cb.geluNew(gate_ct);
        defer cb.free(gate_gelu_ct);
        break :blk try cb.multiply(gate_gelu_ct, value_ct);
    };
    defer cb.free(activated_ct);

    if (captures) |capture_buf| {
        const activated = try cb.toFloat32(activated_ct, allocator);
        defer allocator.free(activated);
        if (activated.len != total * intermediate_size) return error.UnexpectedOutputShape;
        try capture_buf.add(@intCast(layer_idx), "wo", activated, intermediate_size, hidden_size, total);
    }

    // Wo is [hidden, intermediate] so the output is [total, hidden].
    return linearWithScopedLoRA(cb, allocator, activated_ct, Wo_w, null, layer_idx, "mlp", "Wo", lora_rank, lora_alpha, total, intermediate_size, hidden_size);
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
        return cloneHiddenForNormSkip(cb, allocator, hidden, total, hidden_size);
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

/// Buffer of ActivationCapture records from one forward pass.
pub const ActivationBuffer = struct {
    allocator: std.mem.Allocator,
    items: std.ArrayListUnmanaged(ActivationCapture),
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
    return cb.toFloat32(result_ct, allocator);
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

    // Collect normed_attn CTs from all layers without downloading them yet.
    // This lets us batch-evaluate all 22 tensors in one GPU sync instead of
    // one per layer.
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

    var hidden = try embeddingsBlock(cb, allocator, config, input_ids, total_tokens, seq_len);
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

    // Batch-download captured tensors to keep GPU syncs amortized.
    const batch_results = try cb.toFloat32Batch(normed_attn_cts.items, allocator);
    defer {
        for (batch_results) |r| allocator.free(r);
        allocator.free(batch_results);
    }
    const attn_out_results = try cb.toFloat32Batch(attn_out_cts.items, allocator);
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
    const final_norm_input = try cb.toFloat32(hidden, allocator);
    var owns_final_norm_input = true;
    errdefer if (owns_final_norm_input) allocator.free(final_norm_input);
    const final_norm_weight = try cb.toFloat32(fn_w, allocator);
    var owns_final_norm_weight = true;
    errdefer if (owns_final_norm_weight) allocator.free(final_norm_weight);
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

    // RoPE
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
    defer cb.free(K);

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

    // Output projection
    const out_w = try getLayerWeight(cb, layer_idx, "attn.Wo.weight", &name_buf);
    defer cb.free(out_w);
    const out_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "attn.Wo.bias", H, &name_buf);
    defer cb.free(out_b);
    const attn_proj = try linearWithLoRA(cb, allocator, attn_out, out_w, out_b, layer_idx, "Wo", config.lora_rank, config.lora_alpha, total, H, H);
    defer cb.free(attn_proj);

    // Residual
    const hidden_after_attn = try cb.add(attn_proj, hidden);
    defer cb.free(hidden_after_attn);

    // -----------------------------------------------------------------------
    // FFN sub-layer  (pre-norm, GeGLU)
    // -----------------------------------------------------------------------

    const mlp_ln_w = try getLayerWeight(cb, layer_idx, "mlp_norm.weight", &name_buf);
    defer cb.free(mlp_ln_w);
    const mlp_ln_b = try getLayerWeightOrZeroBias(cb, allocator, layer_idx, "mlp_norm.bias", H, &name_buf);
    defer cb.free(mlp_ln_b);
    const normed_ffn = try cb.layerNorm(hidden_after_attn, mlp_ln_w, mlp_ln_b, H, config.layer_norm_eps);
    defer cb.free(normed_ffn);

    const Wi_w = try getLayerWeight(cb, layer_idx, "mlp.Wi.weight", &name_buf);
    defer cb.free(Wi_w);
    const Wo_w = try getLayerWeight(cb, layer_idx, "mlp.Wo.weight", &name_buf);
    defer cb.free(Wo_w);

    const ffn_out = try geGluFfn(cb, allocator, normed_ffn, Wi_w, Wo_w, layer_idx, config.lora_rank, config.lora_alpha, total, H, intermediate, captures);
    defer cb.free(ffn_out);

    return .{
        .hidden = try cb.add(ffn_out, hidden_after_attn),
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
    // consecutive_pairs=true: ModernBERT uses interleaved rotation pairs
    // (matching gopeft fused_chunker_embedder.go convention).
    // rope_dim == head_dim: the full head dimension is rotated.
    const Q = try cb.rope(Q_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
    defer cb.free(Q);
    const K = try cb.rope(K_raw, seq_len, head_dim, head_dim, rope_theta, 1.0, 0, true);
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
