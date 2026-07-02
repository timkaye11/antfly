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

// ModernBERT encoder forward pass as an `ml.graph.Graph`, built with the
// high-level `Builder` API. This is the graph-building sibling of
// `src/architectures/modern_bert.zig`, whose ComputeBackend-based eager
// forward serves as the reference implementation.
//
// Why this exists
// ---------------
// The "level-3" training harness (`real_autodiff_trainer.zig`) requires
// models to be expressed as a computation graph so that
// `ml.graph.autodiff.gradient` can differentiate every op and
// `ml.graph.lora.injectLoRA` can splice LoRA adapters onto the linear
// projections by parameter-name substring. Once this port exists,
// `fused_chunker` — which is built on ModernBERT — can graduate from
// "head-only autodiff" to full-encoder multi-layer LoRA training.
//
// MVP simplifications (TODO)
// --------------------------
// 1. **Windowed attention is caller-provided.** ModernBERT's real
//    architecture alternates global (full) attention layers with local
//    (sliding-window) attention layers. This graph consumes caller-built
//    additive attention biases; segmented callers may provide one bias/RoPE
//    placeholder per layer so mixed local/global segments remain faithful.
// 2. **No LoRA bookkeeping.** `buildForwardGraph` emits plain parameter
//    nodes; LoRA is injected after construction by
//    `ml.graph.lora.injectLoRA`, which matches on substrings like
//    `"query_proj"` / `"value_proj"`.
//
// Parameter naming
// ----------------
// Names mirror `src/architectures/modern_bert.zig` and the HF checkpoint:
//
//   model.embeddings.tok_embeddings.weight
//   model.embeddings.norm.{weight,bias}
//   model.layers.{i}.attn_norm.{weight,bias}
//   model.layers.{i}.attn.query_proj.{weight,bias}
//   model.layers.{i}.attn.key_proj.{weight,bias}
//   model.layers.{i}.attn.value_proj.{weight,bias}
//   model.layers.{i}.attn.Wo.{weight,bias}
//   model.layers.{i}.mlp_norm.{weight,bias}
//   model.layers.{i}.mlp.Wi.weight
//   model.layers.{i}.mlp.Wo.weight
//   model.final_norm.{weight,bias}
//
// Input convention
// ----------------
// Following the new graph-port convention, `buildForwardGraph` does NOT
// create its own input placeholders. The caller builds `input_ids`,
// `attn_bias`, `rope_cos`, and `rope_sin` as parameter nodes (or constant
// nodes) ahead of time and threads them in. The returned struct echoes
// those NodeIds back for convenience.

const std = @import("std");
const ml = @import("ml");

const Graph = ml.graph.Graph;
const Builder = ml.graph.Builder;
const NodeId = ml.graph.NodeId;
const Shape = ml.graph.Shape;
const null_node = ml.graph.null_node;

// ── Public Config ───────────────────────────────────────────────────────────

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    max_position_embeddings: u32 = 8192,
    layer_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 160000.0,
    local_rope_theta: f32 = 10000.0,
    local_attention_window: u32 = 128,
    /// Global attention every N layers. Segment callers use this to choose
    /// the per-layer RoPE table and local-attention mask.
    global_attn_every_n_layers: u32 = 3,
};

/// Result of graph construction. All NodeIds live in the `Graph` that was
/// passed to `buildForwardGraph` via its `Builder`.
pub const ModernBertGraph = struct {
    /// Input placeholder: `[batch * seq_len]` i64 token IDs — echoed from
    /// the caller-provided argument.
    input_ids_node: NodeId,
    /// Input placeholder: `[batch * num_heads, seq_len, seq_len]` f32 additive
    /// attention bias. Caller builds it (0 for valid positions, -1e9 for
    /// masked / padded). Passed in so this function does not mint a new
    /// placeholder; mirrors bert_graph / qwen2_graph convention.
    attn_bias_node: NodeId,
    /// RoPE cos table, `[seq_len, head_dim]` f32. Caller-precomputed for
    /// `config.rope_theta`.
    rope_cos_node: NodeId,
    /// RoPE sin table, `[seq_len, head_dim]` f32.
    rope_sin_node: NodeId,
    /// Output: `[batch * seq_len, hidden_size]` — final encoder hidden
    /// state after the model-level LayerNorm (`model.final_norm`).
    output_node: NodeId,
};

/// Result of constructing a differentiable encoder segment. The synthetic
/// scalar loss is `sum(segment_output * upstream_grad)`, so autodiff wrt
/// `hidden_in_node` yields the VJP needed by the preceding segment and
/// autodiff wrt injected LoRA parameters yields exact segment-local adapter
/// gradients.
pub const ModernBertSegmentGraph = struct {
    hidden_in_node: NodeId,
    upstream_grad_node: NodeId,
    attn_bias_node: NodeId,
    rope_cos_node: NodeId,
    rope_sin_node: NodeId,
    output_node: NodeId,
    loss_node: NodeId,
};

pub const SegmentLayerInputs = struct {
    attn_bias_node: NodeId,
    rope_cos_node: NodeId,
    rope_sin_node: NodeId,
};

/// Per-layer intermediate nodes exposed by the forward-only segment builder.
/// These are the same tensors the eager capture path downloads for LoRA
/// gradient accumulation: the pre-attention LayerNorm output (Q/K/V input),
/// the merged attention output (Wo input), the GeGLU activation (mlp.Wo
/// input), and the layer output (next layer's input).
pub const SegmentForwardLayerNodes = struct {
    attn_normed_node: NodeId,
    attn_merged_node: NodeId,
    mlp_activated_node: NodeId,
    hidden_out_node: NodeId,
};

pub const LayerBackwardSubstage = enum {
    mlp_wo,
    mlp_gelu_input,
    mlp_gate_value,
    mlp_gate_input,
    mlp_wi_output,
    mlp_norm_output,
    mlp_hidden_after_attn,
    attn_out_proj,
    attention_core,
    attention_core_post_rope,
    attention_scores_raw,
    attention_scores_masked,
    attention_probs,
    qkv_proj,
    qkv_proj_split,
    attn_norm_hidden_in,
};

pub const ModernBertLayerBackwardSubstageGraph = struct {
    stage: LayerBackwardSubstage,
    primary_input_node: NodeId = null_node,
    q_raw_node: NodeId = null_node,
    k_raw_node: NodeId = null_node,
    v_raw_node: NodeId = null_node,
    hidden_in_aux_node: NodeId = null_node,
    hidden_after_attn_aux_node: NodeId = null_node,
    gate_input_aux_node: NodeId = null_node,
    gate_value_aux_node: NodeId = null_node,
    upstream_grad_node: NodeId,
    attn_bias_node: NodeId = null_node,
    rope_cos_node: NodeId = null_node,
    rope_sin_node: NodeId = null_node,
    output_node: NodeId,
    loss_node: NodeId,
};

// ── Entry point ─────────────────────────────────────────────────────────────

/// Construct the ModernBERT encoder forward graph.
///
/// Weight parameters are emitted as `bld.parameter` nodes using the HF
/// ModernBERT checkpoint naming convention (see file header for the one
/// intentional divergence around `mlp.Wi`). No weights are bound — the
/// caller is responsible for populating the parameter store with tensor
/// data before execution.
///
/// `input_ids`, `attn_bias`, `rope_cos`, `rope_sin` must already be valid
/// NodeIds in the graph (caller-provided placeholders or constants). The
/// `batch` and `seq_len` arguments are baked into shape metadata; rebuild
/// the graph if either changes.
pub fn buildForwardGraph(
    bld: *Builder,
    config: Config,
    input_ids: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    batch: u32,
    seq_len: u32,
) !ModernBertGraph {
    const H: u32 = config.hidden_size;
    const total: u32 = batch * seq_len;
    const num_heads: u32 = config.num_attention_heads;
    const head_dim: u32 = config.head_dim;

    if (num_heads * head_dim != H) return error.InvalidHeadDim;

    // ── Embeddings: token lookup + embedding LayerNorm ─────────────────
    //
    // ModernBERT has no absolute position embeddings — RoPE is applied
    // inside each attention layer instead. So the embedding block is just
    // `tok_embeddings → LayerNorm(weight, bias)`, where both tensors are
    // under `model.embeddings.*`.
    var hidden = try embeddings(bld, config, input_ids, total, H);

    // ── Encoder layers ─────────────────────────────────────────────────
    var layer: u32 = 0;
    while (layer < config.num_hidden_layers) : (layer += 1) {
        const layer_nodes = try encoderLayer(
            bld,
            config,
            hidden,
            attn_bias,
            rope_cos,
            rope_sin,
            config.rope_theta,
            batch,
            seq_len,
            layer,
            num_heads,
            head_dim,
        );
        hidden = layer_nodes.hidden_out_node;
    }

    // ── Final LayerNorm ────────────────────────────────────────────────
    //
    // `model.final_norm` is a LayerNorm over the hidden dimension. The
    // pre-norm residual pattern inside each layer means the unnormalized
    // residual stream needs one last normalization before exiting the
    // encoder — otherwise downstream heads see raw pre-norm activations.
    const final_norm_w = try bld.parameter(
        "model.final_norm.weight",
        Shape.init(.f32, &.{@intCast(H)}),
    );
    const final_norm_b = try bld.parameter(
        "model.final_norm.bias",
        Shape.init(.f32, &.{@intCast(H)}),
    );
    const output = try bld.layerNorm(hidden, final_norm_w, final_norm_b, H, config.layer_norm_eps);

    return .{
        .input_ids_node = input_ids,
        .attn_bias_node = attn_bias,
        .rope_cos_node = rope_cos,
        .rope_sin_node = rope_sin,
        .output_node = output,
    };
}

/// Construct a differentiable encoder segment over layers `[start_layer, end_layer)`.
///
/// The caller supplies the hidden input instead of token IDs, making this the
/// graph equivalent of Go's segmented synthetic-loss backward path. For the
/// first production slice we use it for the last global layer, but the shape
/// and naming contracts are already segment-oriented.
pub fn buildEncoderSegmentGraph(
    bld: *Builder,
    config: Config,
    hidden_in: NodeId,
    upstream_grad: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
) !ModernBertSegmentGraph {
    const single_input = SegmentLayerInputs{
        .attn_bias_node = attn_bias,
        .rope_cos_node = rope_cos,
        .rope_sin_node = rope_sin,
    };
    return buildEncoderSegmentGraphInternal(
        bld,
        config,
        hidden_in,
        upstream_grad,
        &.{single_input},
        true,
        batch,
        seq_len,
        start_layer,
        end_layer,
    );
}

/// Construct a differentiable encoder segment over layers `[start_layer, end_layer)`,
/// using one attention-bias/RoPE input tuple per layer in the segment.
pub fn buildEncoderSegmentGraphWithLayerInputs(
    bld: *Builder,
    config: Config,
    hidden_in: NodeId,
    upstream_grad: NodeId,
    layer_inputs: []const SegmentLayerInputs,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
) !ModernBertSegmentGraph {
    return buildEncoderSegmentGraphInternal(
        bld,
        config,
        hidden_in,
        upstream_grad,
        layer_inputs,
        false,
        batch,
        seq_len,
        start_layer,
        end_layer,
    );
}

/// Construct a forward-only encoder segment over layers `[start_layer, end_layer)`
/// using one attention-bias/RoPE input tuple per layer. Reuses the exact same
/// `encoderLayer` body as the segment VJP builder so a compiled forward matches
/// the forward recompute already performed inside the VJP graphs.
///
/// `layer_nodes_out` (len == end_layer - start_layer) receives the per-layer
/// capture nodes; the caller decides which of them to mark as graph outputs.
/// Returns the segment output hidden node.
pub fn buildEncoderSegmentForwardGraph(
    bld: *Builder,
    config: Config,
    hidden_in: NodeId,
    layer_inputs: []const SegmentLayerInputs,
    layer_nodes_out: []SegmentForwardLayerNodes,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
) !NodeId {
    if (start_layer >= end_layer or end_layer > config.num_hidden_layers) return error.InvalidLayerRange;
    const segment_len = end_layer - start_layer;
    if (layer_inputs.len != @as(usize, @intCast(segment_len))) return error.InvalidSegmentLayerInputCount;
    if (layer_nodes_out.len != @as(usize, @intCast(segment_len))) return error.InvalidSegmentLayerInputCount;

    const H: u32 = config.hidden_size;
    const num_heads: u32 = config.num_attention_heads;
    const head_dim: u32 = config.head_dim;
    if (num_heads * head_dim != H) return error.InvalidHeadDim;

    var hidden = hidden_in;
    var layer = start_layer;
    while (layer < end_layer) : (layer += 1) {
        const offset: usize = @intCast(layer - start_layer);
        const layer_input = layer_inputs[offset];
        const layer_nodes = try encoderLayer(
            bld,
            config,
            hidden,
            layer_input.attn_bias_node,
            layer_input.rope_cos_node,
            layer_input.rope_sin_node,
            ropeThetaForLayer(config, layer),
            batch,
            seq_len,
            layer,
            num_heads,
            head_dim,
        );
        layer_nodes_out[offset] = layer_nodes;
        hidden = layer_nodes.hidden_out_node;
    }
    return hidden;
}

fn buildEncoderSegmentGraphInternal(
    bld: *Builder,
    config: Config,
    hidden_in: NodeId,
    upstream_grad: NodeId,
    layer_inputs: []const SegmentLayerInputs,
    repeat_single_input: bool,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
) !ModernBertSegmentGraph {
    if (start_layer >= end_layer or end_layer > config.num_hidden_layers) return error.InvalidLayerRange;
    const segment_len = end_layer - start_layer;
    if (layer_inputs.len == 0) return error.MissingSegmentLayerInputs;
    if (!repeat_single_input and layer_inputs.len != @as(usize, @intCast(segment_len))) return error.InvalidSegmentLayerInputCount;

    const H: u32 = config.hidden_size;
    const num_heads: u32 = config.num_attention_heads;
    const head_dim: u32 = config.head_dim;
    if (num_heads * head_dim != H) return error.InvalidHeadDim;

    var hidden = hidden_in;
    var layer = start_layer;
    while (layer < end_layer) : (layer += 1) {
        const input_offset: usize = if (repeat_single_input) 0 else @intCast(layer - start_layer);
        const layer_input = layer_inputs[input_offset];
        const rope_theta = ropeThetaForLayer(config, layer);
        const layer_nodes = try encoderLayer(
            bld,
            config,
            hidden,
            layer_input.attn_bias_node,
            layer_input.rope_cos_node,
            layer_input.rope_sin_node,
            rope_theta,
            batch,
            seq_len,
            layer,
            num_heads,
            head_dim,
        );
        hidden = layer_nodes.hidden_out_node;
    }

    const weighted = try bld.elemMultiply(hidden, upstream_grad);
    const loss = try bld.reduceSum(weighted, &.{ 0, 1 });
    try bld.graph.markOutput(loss);

    return .{
        .hidden_in_node = hidden_in,
        .upstream_grad_node = upstream_grad,
        .attn_bias_node = layer_inputs[0].attn_bias_node,
        .rope_cos_node = layer_inputs[0].rope_cos_node,
        .rope_sin_node = layer_inputs[0].rope_sin_node,
        .output_node = hidden,
        .loss_node = loss,
    };
}

pub fn isGlobalAttentionLayer(config: Config, layer: u32) bool {
    return config.global_attn_every_n_layers == 0 or
        layer % config.global_attn_every_n_layers == 0;
}

pub fn ropeThetaForLayer(config: Config, layer: u32) f32 {
    return if (isGlobalAttentionLayer(config, layer)) config.rope_theta else config.local_rope_theta;
}

pub fn buildEncoderLayerBackwardSubstageGraph(
    bld: *Builder,
    config: Config,
    stage: LayerBackwardSubstage,
    batch: u32,
    seq_len: u32,
    layer: u32,
) !ModernBertLayerBackwardSubstageGraph {
    if (layer >= config.num_hidden_layers) return error.InvalidLayerRange;
    const H: u32 = config.hidden_size;
    const I: u32 = config.intermediate_size;
    const total: u32 = batch * seq_len;
    const total_i: i64 = @intCast(total);
    const hidden_i: i64 = @intCast(H);
    const intermediate_i: i64 = @intCast(I);
    const doubled_intermediate_i: i64 = @intCast(2 * I);
    const head_dim_i: i64 = @intCast(config.head_dim);
    const bh_i: i64 = @intCast(batch * config.num_attention_heads);
    const seq_i: i64 = @intCast(seq_len);
    const num_heads = config.num_attention_heads;
    const head_dim = config.head_dim;
    if (num_heads * head_dim != H) return error.InvalidHeadDim;

    const upstream_grad = try bld.parameter(
        "__substage_upstream_grad",
        Shape.init(.f32, &.{ total_i, hidden_i }),
    );

    var out = ModernBertLayerBackwardSubstageGraph{
        .stage = stage,
        .upstream_grad_node = upstream_grad,
        .output_node = null_node,
        .loss_node = null_node,
    };

    const output = switch (stage) {
        .mlp_wo => blk: {
            const activated = try bld.parameter(
                "__substage_mlp_wo_input",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = activated;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromMlpActivated(
                bld,
                config,
                activated,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_gelu_input => blk: {
            const gate_gelu = try bld.parameter(
                "__substage_gelu_input",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const gate_value = try bld.parameter(
                "__substage_gate_value_aux",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = gate_gelu;
            out.gate_value_aux_node = gate_value;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromGeluOutput(
                bld,
                config,
                gate_gelu,
                gate_value,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_gate_value => blk: {
            const gate_value = try bld.parameter(
                "__substage_gate_value",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const gate_input = try bld.parameter(
                "__substage_gate_input_aux",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = gate_value;
            out.gate_input_aux_node = gate_input;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromGateInputs(
                bld,
                config,
                gate_input,
                gate_value,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_gate_input => blk: {
            const gate_input = try bld.parameter(
                "__substage_gate_input",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const gate_value = try bld.parameter(
                "__substage_gate_value_aux",
                Shape.init(.f32, &.{ total_i, intermediate_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = gate_input;
            out.gate_value_aux_node = gate_value;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromGateInputs(
                bld,
                config,
                gate_input,
                gate_value,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_wi_output => blk: {
            const gated = try bld.parameter(
                "__substage_wi_output",
                Shape.init(.f32, &.{ total_i, doubled_intermediate_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = gated;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromWiOutput(
                bld,
                config,
                gated,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_norm_output => blk: {
            const mlp_normed = try bld.parameter(
                "__substage_mlp_norm_output",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = mlp_normed;
            out.hidden_after_attn_aux_node = hidden_after_attn;
            break :blk try encoderLayerTailFromMlpNormed(
                bld,
                config,
                mlp_normed,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .mlp_hidden_after_attn => blk: {
            const hidden_after_attn = try bld.parameter(
                "__substage_hidden_after_attn",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = hidden_after_attn;
            break :blk try encoderLayerTailFromHiddenAfterAttn(
                bld,
                config,
                hidden_after_attn,
                total,
                layer,
            );
        },
        .attn_out_proj => blk: {
            const attn_merged = try bld.parameter(
                "__substage_attn_merged",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = attn_merged;
            out.hidden_in_aux_node = hidden_in_aux;
            break :blk try encoderLayerTailFromAttnMerged(
                bld,
                config,
                attn_merged,
                hidden_in_aux,
                total,
                layer,
            );
        },
        .attention_core => blk: {
            const q_raw = try bld.parameter(
                "__substage_q_raw",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const k_raw = try bld.parameter(
                "__substage_k_raw",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const v_raw = try bld.parameter(
                "__substage_v_raw",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos = try bld.parameter(
                "__substage_rope_cos",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin = try bld.parameter(
                "__substage_rope_sin",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            out.q_raw_node = q_raw;
            out.k_raw_node = k_raw;
            out.v_raw_node = v_raw;
            out.hidden_in_aux_node = hidden_in_aux;
            out.attn_bias_node = attn_bias;
            out.rope_cos_node = rope_cos;
            out.rope_sin_node = rope_sin;
            break :blk try encoderLayerTailFromQKVRaw(
                bld,
                config,
                q_raw,
                k_raw,
                v_raw,
                hidden_in_aux,
                attn_bias,
                rope_cos,
                rope_sin,
                ropeThetaForLayer(config, layer),
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .attention_core_post_rope => blk: {
            const q_rope = try bld.parameter(
                "__substage_q_rope",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const k_rope = try bld.parameter(
                "__substage_k_rope",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const v_raw = try bld.parameter(
                "__substage_v_attention_input",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            out.q_raw_node = q_rope;
            out.k_raw_node = k_rope;
            out.v_raw_node = v_raw;
            out.hidden_in_aux_node = hidden_in_aux;
            out.attn_bias_node = attn_bias;
            break :blk try encoderLayerTailFromQKVRopedFlat(
                bld,
                config,
                q_rope,
                k_rope,
                v_raw,
                hidden_in_aux,
                attn_bias,
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .attention_scores_raw => blk: {
            const scores_raw = try bld.parameter(
                "__substage_scores_raw",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const v_raw = try bld.parameter(
                "__substage_v_attention_input",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            out.primary_input_node = scores_raw;
            out.v_raw_node = v_raw;
            out.hidden_in_aux_node = hidden_in_aux;
            out.attn_bias_node = attn_bias;
            break :blk try encoderLayerTailFromAttentionScoresRaw(
                bld,
                config,
                scores_raw,
                v_raw,
                hidden_in_aux,
                attn_bias,
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .attention_scores_masked => blk: {
            const scores_masked = try bld.parameter(
                "__substage_scores_masked",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const v_raw = try bld.parameter(
                "__substage_v_attention_input",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = scores_masked;
            out.v_raw_node = v_raw;
            out.hidden_in_aux_node = hidden_in_aux;
            break :blk try encoderLayerTailFromAttentionScoresMasked(
                bld,
                config,
                scores_masked,
                v_raw,
                hidden_in_aux,
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .attention_probs => blk: {
            const probs = try bld.parameter(
                "__substage_attention_probs",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const v_raw = try bld.parameter(
                "__substage_v_attention_input",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            out.primary_input_node = probs;
            out.v_raw_node = v_raw;
            out.hidden_in_aux_node = hidden_in_aux;
            break :blk try encoderLayerTailFromAttentionProbs(
                bld,
                config,
                probs,
                v_raw,
                hidden_in_aux,
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .qkv_proj => blk: {
            const attn_normed = try bld.parameter(
                "__substage_attn_normed",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos = try bld.parameter(
                "__substage_rope_cos",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin = try bld.parameter(
                "__substage_rope_sin",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            out.primary_input_node = attn_normed;
            out.hidden_in_aux_node = hidden_in_aux;
            out.attn_bias_node = attn_bias;
            out.rope_cos_node = rope_cos;
            out.rope_sin_node = rope_sin;
            break :blk try encoderLayerTailFromAttnNormed(
                bld,
                config,
                attn_normed,
                hidden_in_aux,
                attn_bias,
                rope_cos,
                rope_sin,
                ropeThetaForLayer(config, layer),
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .qkv_proj_split => blk: {
            const q_attn_normed = try bld.parameter(
                "__substage_q_attn_normed",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const k_attn_normed = try bld.parameter(
                "__substage_k_attn_normed",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const v_attn_normed = try bld.parameter(
                "__substage_v_attn_normed",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const hidden_in_aux = try bld.parameter(
                "__substage_hidden_in_aux",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos = try bld.parameter(
                "__substage_rope_cos",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin = try bld.parameter(
                "__substage_rope_sin",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            out.q_raw_node = q_attn_normed;
            out.k_raw_node = k_attn_normed;
            out.v_raw_node = v_attn_normed;
            out.hidden_in_aux_node = hidden_in_aux;
            out.attn_bias_node = attn_bias;
            out.rope_cos_node = rope_cos;
            out.rope_sin_node = rope_sin;
            break :blk try encoderLayerTailFromAttnNormedSplit(
                bld,
                config,
                q_attn_normed,
                k_attn_normed,
                v_attn_normed,
                hidden_in_aux,
                attn_bias,
                rope_cos,
                rope_sin,
                ropeThetaForLayer(config, layer),
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
        },
        .attn_norm_hidden_in => blk: {
            const hidden_in = try bld.parameter(
                "__substage_hidden_in",
                Shape.init(.f32, &.{ total_i, hidden_i }),
            );
            const attn_bias = try bld.parameter(
                "__substage_attn_bias",
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos = try bld.parameter(
                "__substage_rope_cos",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin = try bld.parameter(
                "__substage_rope_sin",
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            out.primary_input_node = hidden_in;
            out.attn_bias_node = attn_bias;
            out.rope_cos_node = rope_cos;
            out.rope_sin_node = rope_sin;
            const layer_nodes = try encoderLayer(
                bld,
                config,
                hidden_in,
                attn_bias,
                rope_cos,
                rope_sin,
                ropeThetaForLayer(config, layer),
                batch,
                seq_len,
                layer,
                num_heads,
                head_dim,
            );
            break :blk layer_nodes.hidden_out_node;
        },
    };

    const loss = try syntheticUpstreamLoss(bld, output, upstream_grad);
    try bld.graph.markOutput(loss);
    out.output_node = output;
    out.loss_node = loss;
    return out;
}

// ── Embeddings block ────────────────────────────────────────────────────────

fn embeddings(
    bld: *Builder,
    config: Config,
    input_ids: NodeId,
    total: u32,
    H: u32,
) !NodeId {
    // Token embeddings: `[vocab_size, hidden]` lookup indexed by flat token ids.
    const tok_emb_param = try bld.parameter(
        "model.embeddings.tok_embeddings.weight",
        Shape.init(.f32, &.{ @intCast(config.vocab_size), @intCast(H) }),
    );
    const tok_emb = try bld.embeddingLookup(tok_emb_param, input_ids, total, H);

    // Embedding-level LayerNorm (ModernBERT replaces BERT's post-sum norm
    // with a pre-encoder norm).
    const ln_w = try bld.parameter(
        "model.embeddings.norm.weight",
        Shape.init(.f32, &.{@intCast(H)}),
    );
    const ln_b = try bld.parameter(
        "model.embeddings.norm.bias",
        Shape.init(.f32, &.{@intCast(H)}),
    );
    return bld.layerNorm(tok_emb, ln_w, ln_b, H, config.layer_norm_eps);
}

// ── Single encoder layer ────────────────────────────────────────────────────
//
// Pre-norm structure:
//
//   x1 = LayerNorm(x)                        (attn_norm)
//   x2 = x + Attention(x1)                   (residual)
//   x3 = LayerNorm(x2)                       (mlp_norm)
//   out = x2 + GeGLU_MLP(x3)                 (residual)
//
// Both residual adds are on the unnormalized stream — the normed tensor
// is fed into the sub-layer, but the residual flows around it.

fn encoderLayer(
    bld: *Builder,
    config: Config,
    hidden_in: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    rope_theta: f32,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !SegmentForwardLayerNodes {
    const H: u32 = config.hidden_size;
    const total: u32 = batch * seq_len;

    // ── Pre-attention LayerNorm ────────────────────────────────────────
    // Layer 0 has no checkpoint attn_norm tensors; the eager path still
    // applies layer norm with gamma=1 and beta=0, so the graph VJP must do the
    // same to keep segment-local gradients isomorphic to the forward path.
    const attn_normed = if (layer == 0) blk: {
        const attn_ln_w = try constantVector(bld, H, 1.0);
        const attn_ln_b = try constantVector(bld, H, 0.0);
        break :blk try bld.layerNorm(hidden_in, attn_ln_w, attn_ln_b, H, config.layer_norm_eps);
    } else blk: {
        const attn_ln_w = try layerParam(bld, layer, "attn_norm.weight", .{H});
        const attn_ln_b = try layerParam(bld, layer, "attn_norm.bias", .{H});
        break :blk try bld.layerNorm(hidden_in, attn_ln_w, attn_ln_b, H, config.layer_norm_eps);
    };

    // ── Q / K / V projections (with bias — HF ModernBERT keeps bias on attn) ─
    //
    // Names match `src/architectures/modern_bert.zig` so both the eager
    // and graph code paths can share a weight loader.
    const q_w = try layerParam(bld, layer, "attn.query_proj.weight", .{ H, H });
    const q_b = try layerParam(bld, layer, "attn.query_proj.bias", .{H});
    const Q_flat = try bld.linear(attn_normed, q_w, q_b, total, H, H);

    const k_w = try layerParam(bld, layer, "attn.key_proj.weight", .{ H, H });
    const k_b = try layerParam(bld, layer, "attn.key_proj.bias", .{H});
    const K_flat = try bld.linear(attn_normed, k_w, k_b, total, H, H);

    const v_w = try layerParam(bld, layer, "attn.value_proj.weight", .{ H, H });
    const v_b = try layerParam(bld, layer, "attn.value_proj.bias", .{H});
    const V_flat = try bld.linear(attn_normed, v_w, v_b, total, H, H);

    // ── Reshape Q/K/V into per-head layout `[B*H, S, D]` ────────────────
    //
    // Start from `[B*S, H*D]`, reshape to `[B, S, H, D]`, transpose to
    // `[B, H, S, D]`, then flatten leading axes to `[B*H, S, D]`. This is
    // the rank-3 layout matmul3D expects.
    const q_bhsd = try splitHeads(bld, Q_flat, batch, seq_len, num_heads, head_dim);
    const k_bhsd = try splitHeads(bld, K_flat, batch, seq_len, num_heads, head_dim);
    const v_bhsd = try splitHeads(bld, V_flat, batch, seq_len, num_heads, head_dim);

    // ── Apply RoPE to Q and K (bidirectional — same cos/sin tables) ─────
    //
    // ModernBERT rotates the full head dimension (`rope_dim == head_dim`).
    // RoPE does not depend on attention direction, so the same `rope_cos`
    // / `rope_sin` tables that would be used by a causal decoder apply
    // here. V is not rotated.
    const q_roped = try bld.ropeWithOptions(
        q_bhsd,
        rope_cos,
        rope_sin,
        seq_len,
        head_dim,
        head_dim,
        rope_theta,
        true,
    );
    const k_roped = try bld.ropeWithOptions(
        k_bhsd,
        rope_cos,
        rope_sin,
        seq_len,
        head_dim,
        head_dim,
        rope_theta,
        true,
    );

    // ── Manual masked scaled dot-product attention ─────────────────────
    //
    // Mirrors bert_graph.zig's decomposition so the autodiff VJPs line up.
    // scores = Q @ K^T / sqrt(d), shape `[B*H, S, S]`.
    const k_t = try bld.transpose(k_roped, &.{ 0, 2, 1 });
    const scores_raw = try bld.matmul3D(q_roped, k_t);

    const inv_sqrt_d = try bld.scalarConst(
        .f32,
        1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    );
    const scores_scaled = try bld.mul(scores_raw, inv_sqrt_d);

    // Additive attention bias. The caller is expected to have shaped this
    // as `[B*H, S, S]` (or broadcastable to that) with 0 on valid positions
    // and -1e9 on masked ones. Bidirectional attention: NO causal mask is
    // applied — the bias is pure padding mask.
    const scores_masked = try bld.add(scores_scaled, attn_bias);

    // Softmax along the last axis (keys).
    const probs = try bld.softmax(scores_masked);

    // attn_out = probs @ V, shape `[B*H, S, D]`.
    const attn_bhsd = try bld.matmul3D(probs, v_bhsd);

    // ── Merge heads back to `[B*S, H*D]` for the output projection ─────
    const attn_merged = try mergeHeads(bld, attn_bhsd, batch, seq_len, num_heads, head_dim);

    // ── Output projection (with bias) ──────────────────────────────────
    const o_w = try layerParam(bld, layer, "attn.Wo.weight", .{ H, H });
    const o_b = try layerParam(bld, layer, "attn.Wo.bias", .{H});
    const attn_proj = try bld.linear(attn_merged, o_w, o_b, total, H, H);

    // Pre-norm residual: add projected attention output back onto the
    // *unnormalized* hidden state.
    const hidden_after_attn = try bld.add(attn_proj, hidden_in);

    // ── Pre-MLP LayerNorm ──────────────────────────────────────────────
    const mlp_ln_w = try layerParam(bld, layer, "mlp_norm.weight", .{H});
    const mlp_ln_b = try layerParam(bld, layer, "mlp_norm.bias", .{H});
    const mlp_normed = try bld.layerNorm(
        hidden_after_attn,
        mlp_ln_w,
        mlp_ln_b,
        H,
        config.layer_norm_eps,
    );

    // ── GeGLU MLP ──────────────────────────────────────────────────────
    //
    // GeGLU = GELU-gated linear unit:
    //
    //   gated  = mlp_normed @ Wi^T            [total, 2*intermediate]
    //   gate   = gated[:, :intermediate]      [total, intermediate]
    //   up     = gated[:, intermediate:]      [total, intermediate]
    //   act    = GELU(gate) * up              [total, intermediate]
    //   out    = act @ Wo^T                   [total, hidden]
    //
    // All three linears are bias-free in HF ModernBERT's MLP. No
    // `Builder.geGluActivation` wrapper exists yet — we compose it out of
    // primitives. swigluActivation() is the closest analogue but uses SiLU
    // instead of GELU, so it's not quite what we want.
    const I: u32 = config.intermediate_size;

    const wi_w = try layerParam(bld, layer, "mlp.Wi.weight", .{ 2 * I, H });
    const gated = try bld.linearNoBias(mlp_normed, wi_w, total, H, 2 * I);
    const gate_linear = try bld.sliceLastDim(gated, 0, @intCast(I));
    const up_linear = try bld.sliceLastDim(gated, @intCast(I), @intCast(2 * I));

    const gate_gelu = try bld.geluExact(gate_linear);
    const activated = try bld.elemMultiply(gate_gelu, up_linear);

    const down_w = try layerParam(bld, layer, "mlp.Wo.weight", .{ H, I });
    const mlp_out = try bld.linearNoBias(activated, down_w, total, I, H);

    // Pre-norm residual on the post-attention stream.
    const hidden_out = try bld.add(mlp_out, hidden_after_attn);
    return .{
        .attn_normed_node = attn_normed,
        .attn_merged_node = attn_merged,
        .mlp_activated_node = activated,
        .hidden_out_node = hidden_out,
    };
}

fn encoderLayerTailFromMlpActivated(
    bld: *Builder,
    config: Config,
    activated: NodeId,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const H: u32 = config.hidden_size;
    const I: u32 = config.intermediate_size;
    const down_w = try layerParam(bld, layer, "mlp.Wo.weight", .{ H, I });
    const mlp_out = try bld.linearNoBias(activated, down_w, total, I, H);
    return bld.add(mlp_out, hidden_after_attn);
}

fn encoderLayerTailFromGeluOutput(
    bld: *Builder,
    config: Config,
    gate_gelu: NodeId,
    gate_value: NodeId,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const activated = try bld.elemMultiply(gate_gelu, gate_value);
    return encoderLayerTailFromMlpActivated(bld, config, activated, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromGateInputs(
    bld: *Builder,
    config: Config,
    gate_input: NodeId,
    gate_value: NodeId,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const gate_gelu = try bld.geluExact(gate_input);
    return encoderLayerTailFromGeluOutput(bld, config, gate_gelu, gate_value, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromWiOutput(
    bld: *Builder,
    config: Config,
    gated: NodeId,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const I: u32 = config.intermediate_size;
    const gate_linear = try bld.sliceLastDim(gated, 0, @intCast(I));
    const up_linear = try bld.sliceLastDim(gated, @intCast(I), @intCast(2 * I));
    return encoderLayerTailFromGateInputs(bld, config, gate_linear, up_linear, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromMlpNormed(
    bld: *Builder,
    config: Config,
    mlp_normed: NodeId,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const H: u32 = config.hidden_size;
    const I: u32 = config.intermediate_size;
    const wi_w = try layerParam(bld, layer, "mlp.Wi.weight", .{ 2 * I, H });
    const gated = try bld.linearNoBias(mlp_normed, wi_w, total, H, 2 * I);
    return encoderLayerTailFromWiOutput(bld, config, gated, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromHiddenAfterAttn(
    bld: *Builder,
    config: Config,
    hidden_after_attn: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const H: u32 = config.hidden_size;
    const mlp_ln_w = try layerParam(bld, layer, "mlp_norm.weight", .{H});
    const mlp_ln_b = try layerParam(bld, layer, "mlp_norm.bias", .{H});
    const mlp_normed = try bld.layerNorm(
        hidden_after_attn,
        mlp_ln_w,
        mlp_ln_b,
        H,
        config.layer_norm_eps,
    );
    return encoderLayerTailFromMlpNormed(bld, config, mlp_normed, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromAttnMerged(
    bld: *Builder,
    config: Config,
    attn_merged: NodeId,
    hidden_in: NodeId,
    total: u32,
    layer: u32,
) !NodeId {
    const H: u32 = config.hidden_size;
    const o_w = try layerParam(bld, layer, "attn.Wo.weight", .{ H, H });
    const o_b = try layerParam(bld, layer, "attn.Wo.bias", .{H});
    const attn_proj = try bld.linear(attn_merged, o_w, o_b, total, H, H);
    const hidden_after_attn = try bld.add(attn_proj, hidden_in);
    return encoderLayerTailFromHiddenAfterAttn(bld, config, hidden_after_attn, total, layer);
}

fn encoderLayerTailFromQKVRaw(
    bld: *Builder,
    config: Config,
    q_raw: NodeId,
    k_raw: NodeId,
    v_raw: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    rope_theta: f32,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const q_bhsd = try splitHeads(bld, q_raw, batch, seq_len, num_heads, head_dim);
    const k_bhsd = try splitHeads(bld, k_raw, batch, seq_len, num_heads, head_dim);
    const v_bhsd = try splitHeads(bld, v_raw, batch, seq_len, num_heads, head_dim);
    const q_roped = try bld.ropeWithOptions(
        q_bhsd,
        rope_cos,
        rope_sin,
        seq_len,
        head_dim,
        head_dim,
        rope_theta,
        true,
    );
    const k_roped = try bld.ropeWithOptions(
        k_bhsd,
        rope_cos,
        rope_sin,
        seq_len,
        head_dim,
        head_dim,
        rope_theta,
        true,
    );
    return encoderLayerTailFromPostRopeAttention(
        bld,
        config,
        q_roped,
        k_roped,
        v_bhsd,
        hidden_in,
        attn_bias,
        batch,
        seq_len,
        layer,
        num_heads,
        head_dim,
    );
}

fn encoderLayerTailFromQKVRopedFlat(
    bld: *Builder,
    config: Config,
    q_rope: NodeId,
    k_rope: NodeId,
    v_raw: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const q_bhsd = try splitHeads(bld, q_rope, batch, seq_len, num_heads, head_dim);
    const k_bhsd = try splitHeads(bld, k_rope, batch, seq_len, num_heads, head_dim);
    const v_bhsd = try splitHeads(bld, v_raw, batch, seq_len, num_heads, head_dim);
    return encoderLayerTailFromPostRopeAttention(
        bld,
        config,
        q_bhsd,
        k_bhsd,
        v_bhsd,
        hidden_in,
        attn_bias,
        batch,
        seq_len,
        layer,
        num_heads,
        head_dim,
    );
}

fn encoderLayerTailFromPostRopeAttention(
    bld: *Builder,
    config: Config,
    q_roped: NodeId,
    k_roped: NodeId,
    v_bhsd: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const k_t = try bld.transpose(k_roped, &.{ 0, 2, 1 });
    const scores_raw = try bld.matmul3D(q_roped, k_t);
    const inv_sqrt_d = try bld.scalarConst(
        .f32,
        1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    );
    const scores_scaled = try bld.mul(scores_raw, inv_sqrt_d);
    const scores_masked = try bld.add(scores_scaled, attn_bias);
    const probs = try bld.softmax(scores_masked);
    return encoderLayerTailFromAttentionProbsBhsd(bld, config, probs, v_bhsd, hidden_in, batch, seq_len, layer, num_heads, head_dim);
}

fn encoderLayerTailFromAttentionScoresRaw(
    bld: *Builder,
    config: Config,
    scores_raw: NodeId,
    v_raw: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const inv_sqrt_d = try bld.scalarConst(
        .f32,
        1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
    );
    const scores_scaled = try bld.mul(scores_raw, inv_sqrt_d);
    const scores_masked = try bld.add(scores_scaled, attn_bias);
    return encoderLayerTailFromAttentionScoresMasked(bld, config, scores_masked, v_raw, hidden_in, batch, seq_len, layer, num_heads, head_dim);
}

fn encoderLayerTailFromAttentionScoresMasked(
    bld: *Builder,
    config: Config,
    scores_masked: NodeId,
    v_raw: NodeId,
    hidden_in: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const probs = try bld.softmax(scores_masked);
    return encoderLayerTailFromAttentionProbs(bld, config, probs, v_raw, hidden_in, batch, seq_len, layer, num_heads, head_dim);
}

fn encoderLayerTailFromAttentionProbs(
    bld: *Builder,
    config: Config,
    probs: NodeId,
    v_raw: NodeId,
    hidden_in: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const v_bhsd = try splitHeads(bld, v_raw, batch, seq_len, num_heads, head_dim);
    return encoderLayerTailFromAttentionProbsBhsd(bld, config, probs, v_bhsd, hidden_in, batch, seq_len, layer, num_heads, head_dim);
}

fn encoderLayerTailFromAttentionProbsBhsd(
    bld: *Builder,
    config: Config,
    probs: NodeId,
    v_bhsd: NodeId,
    hidden_in: NodeId,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const attn_bhsd = try bld.matmul3D(probs, v_bhsd);
    const attn_merged = try mergeHeads(bld, attn_bhsd, batch, seq_len, num_heads, head_dim);
    return encoderLayerTailFromAttnMerged(bld, config, attn_merged, hidden_in, batch * seq_len, layer);
}

fn encoderLayerTailFromAttnNormed(
    bld: *Builder,
    config: Config,
    attn_normed: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    rope_theta: f32,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    return encoderLayerTailFromAttnNormedSplit(
        bld,
        config,
        attn_normed,
        attn_normed,
        attn_normed,
        hidden_in,
        attn_bias,
        rope_cos,
        rope_sin,
        rope_theta,
        batch,
        seq_len,
        layer,
        num_heads,
        head_dim,
    );
}

fn encoderLayerTailFromAttnNormedSplit(
    bld: *Builder,
    config: Config,
    q_attn_normed: NodeId,
    k_attn_normed: NodeId,
    v_attn_normed: NodeId,
    hidden_in: NodeId,
    attn_bias: NodeId,
    rope_cos: NodeId,
    rope_sin: NodeId,
    rope_theta: f32,
    batch: u32,
    seq_len: u32,
    layer: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const H: u32 = config.hidden_size;
    const total = batch * seq_len;
    const q_w = try layerParam(bld, layer, "attn.query_proj.weight", .{ H, H });
    const q_b = try layerParam(bld, layer, "attn.query_proj.bias", .{H});
    const q_raw = try bld.linear(q_attn_normed, q_w, q_b, total, H, H);
    const k_w = try layerParam(bld, layer, "attn.key_proj.weight", .{ H, H });
    const k_b = try layerParam(bld, layer, "attn.key_proj.bias", .{H});
    const k_raw = try bld.linear(k_attn_normed, k_w, k_b, total, H, H);
    const v_w = try layerParam(bld, layer, "attn.value_proj.weight", .{ H, H });
    const v_b = try layerParam(bld, layer, "attn.value_proj.bias", .{H});
    const v_raw = try bld.linear(v_attn_normed, v_w, v_b, total, H, H);
    return encoderLayerTailFromQKVRaw(
        bld,
        config,
        q_raw,
        k_raw,
        v_raw,
        hidden_in,
        attn_bias,
        rope_cos,
        rope_sin,
        rope_theta,
        batch,
        seq_len,
        layer,
        num_heads,
        head_dim,
    );
}

fn syntheticUpstreamLoss(bld: *Builder, output: NodeId, upstream_grad: NodeId) !NodeId {
    const weighted = try bld.elemMultiply(output, upstream_grad);
    return bld.reduceSum(weighted, &.{ 0, 1 });
}

// ── Head-split / merge helpers ──────────────────────────────────────────────

/// `[B*S, H*D] -> [B*H, S, D]`
///
/// Reshape to `[B, S, H, D]`, transpose `(0, 2, 1, 3)`, flatten batch/head.
fn splitHeads(
    bld: *Builder,
    flat: NodeId,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const bsnh = try bld.reshape(
        flat,
        Shape.init(.f32, &.{
            @intCast(batch),
            @intCast(seq_len),
            @intCast(num_heads),
            @intCast(head_dim),
        }),
    );
    const bnsh = try bld.transpose(bsnh, &.{ 0, 2, 1, 3 });
    return bld.reshape(
        bnsh,
        Shape.init(.f32, &.{
            @intCast(batch * num_heads),
            @intCast(seq_len),
            @intCast(head_dim),
        }),
    );
}

/// `[B*H, S, D] -> [B*S, H*D]` (inverse of splitHeads).
fn mergeHeads(
    bld: *Builder,
    bhsd: NodeId,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) !NodeId {
    const bnsh = try bld.reshape(
        bhsd,
        Shape.init(.f32, &.{
            @intCast(batch),
            @intCast(num_heads),
            @intCast(seq_len),
            @intCast(head_dim),
        }),
    );
    const bsnh = try bld.transpose(bnsh, &.{ 0, 2, 1, 3 });
    return bld.reshape(
        bsnh,
        Shape.init(.f32, &.{
            @intCast(batch * seq_len),
            @intCast(num_heads * head_dim),
        }),
    );
}

// ── Parameter helper ────────────────────────────────────────────────────────

/// Emit a `bld.parameter` node named `model.layers.{layer}.{suffix}` with a
/// 1- or 2-D f32 shape described by `dims` (a tuple / anonymous struct of
/// `u32` values). Mirrors `bert_graph.zig`'s helper one-for-one so the two
/// files can drift in parallel without surprise.
fn layerParam(bld: *Builder, layer: u32, suffix: []const u8, dims: anytype) !NodeId {
    var name_buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(
        &name_buf,
        "model.layers.{d}.{s}",
        .{ layer, suffix },
    );
    const d = dims;
    const shape = switch (@typeInfo(@TypeOf(d)).@"struct".fields.len) {
        1 => Shape.init(.f32, &.{@intCast(d[0])}),
        2 => Shape.init(.f32, &.{ @intCast(d[0]), @intCast(d[1]) }),
        else => @compileError("layerParam: only 1-D or 2-D dims supported"),
    };
    return bld.parameter(name, shape);
}

fn constantVector(bld: *Builder, len: u32, value: f32) !NodeId {
    const allocator = bld.graph.allocator;
    const data = try allocator.alloc(f32, len);
    defer allocator.free(data);
    @memset(data, value);
    return bld.tensorConst(data, Shape.init(.f32, &.{@intCast(len)}));
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "buildForwardGraph constructs ModernBERT graph with correct output shape" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    const config = Config{
        .vocab_size = 100,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .head_dim = 8,
        .intermediate_size = 64,
        .max_position_embeddings = 64,
    };

    const batch: u32 = 1;
    const seq_len: u32 = 8;
    const total: i64 = @intCast(batch * seq_len);
    const head_dim_i: i64 = @intCast(config.head_dim);
    const bh_i: i64 = @intCast(batch * config.num_attention_heads);
    const seq_i: i64 = @intCast(seq_len);

    // Caller-provided input placeholders — per the new convention.
    const input_ids = try bld.parameter(
        "__input_ids",
        Shape.init(.i64, &.{total}),
    );
    const attn_bias = try bld.parameter(
        "__attn_bias",
        Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
    );
    const rope_cos = try bld.parameter(
        "__rope_cos",
        Shape.init(.f32, &.{ seq_i, head_dim_i }),
    );
    const rope_sin = try bld.parameter(
        "__rope_sin",
        Shape.init(.f32, &.{ seq_i, head_dim_i }),
    );

    const mb_graph = try buildForwardGraph(
        &bld,
        config,
        input_ids,
        attn_bias,
        rope_cos,
        rope_sin,
        batch,
        seq_len,
    );

    // Output shape should be [batch*seq, hidden] = [8, 32].
    const out_shape = g.node(mb_graph.output_node).output_shape;
    try std.testing.expectEqual(@as(i64, 8), out_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 32), out_shape.dim(1));

    // Sanity check on emitted parameter count. Per layer we emit:
    //   - attn_norm (w, b)                                2
    //   - Q/K/V proj (w, b) × 3                           6
    //   - Wo (w, b)                                       2
    //   - mlp_norm (w, b)                                 2
    //   - Wi (w), Wo (w)                                  2
    //                                                    = 14
    // Plus embeddings: tok_embeddings.weight + embeddings.norm.{w,b} = 3
    // Plus final_norm.{w,b} = 2
    // Plus 4 caller-provided input placeholders.
    // Total = 3 + 2 + 2*14 + 4 = 37.
    try std.testing.expect(g.parameters.items.len >= 35);

    var saw_wi = false;
    var saw_split_gate = false;
    var saw_split_up = false;
    for (g.parameters.items) |param_id| {
        const name = g.parameterName(g.node(param_id));
        if (std.mem.eql(u8, name, "model.layers.0.mlp.Wi.weight")) saw_wi = true;
        if (std.mem.eql(u8, name, "model.layers.0.mlp.gate_proj.weight")) saw_split_gate = true;
        if (std.mem.eql(u8, name, "model.layers.0.mlp.up_proj.weight")) saw_split_up = true;
    }
    try std.testing.expect(saw_wi);
    try std.testing.expect(!saw_split_gate);
    try std.testing.expect(!saw_split_up);
}

test "buildEncoderSegmentGraph emits synthetic scalar loss" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    const config = Config{
        .vocab_size = 100,
        .hidden_size = 32,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .head_dim = 8,
        .intermediate_size = 64,
        .max_position_embeddings = 64,
    };

    const batch: u32 = 1;
    const seq_len: u32 = 8;
    const total: i64 = @intCast(batch * seq_len);
    const hidden_i: i64 = @intCast(config.hidden_size);
    const head_dim_i: i64 = @intCast(config.head_dim);
    const bh_i: i64 = @intCast(batch * config.num_attention_heads);
    const seq_i: i64 = @intCast(seq_len);

    const hidden_in = try bld.parameter(
        "__segment_hidden_in",
        Shape.init(.f32, &.{ total, hidden_i }),
    );
    const upstream_grad = try bld.parameter(
        "__segment_upstream_grad",
        Shape.init(.f32, &.{ total, hidden_i }),
    );
    const attn_bias = try bld.parameter(
        "__segment_attn_bias",
        Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
    );
    const rope_cos = try bld.parameter(
        "__rope_cos",
        Shape.init(.f32, &.{ seq_i, head_dim_i }),
    );
    const rope_sin = try bld.parameter(
        "__rope_sin",
        Shape.init(.f32, &.{ seq_i, head_dim_i }),
    );

    const seg = try buildEncoderSegmentGraph(
        &bld,
        config,
        hidden_in,
        upstream_grad,
        attn_bias,
        rope_cos,
        rope_sin,
        batch,
        seq_len,
        1,
        2,
    );

    const out_shape = g.node(seg.output_node).output_shape;
    try std.testing.expectEqual(@as(i64, 8), out_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 32), out_shape.dim(1));
    try std.testing.expectEqual(@as(u8, 0), g.node(seg.loss_node).output_shape.rank());
}
