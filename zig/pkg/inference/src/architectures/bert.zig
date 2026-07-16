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

// BERT encoder architecture using abstract ComputeBackend ops.
//
// Single implementation works with any ComputeBackend implementation.
// The compute backend handles all hardware-specific execution.

const std = @import("std");
const platform = @import("antfly_platform");
const ops = @import("../ops/ops.zig");
const CT = ops.CT;
const ComputeBackend = ops.ComputeBackend;
const bert_config = @import("../models/bert.zig");

pub const Config = bert_config.Config;

const BertLinearSlotKind = enum(usize) {
    q,
    k,
    v,
    attention_output,
    intermediate,
    output,
};

const BertLayerNormSlotKind = enum(usize) {
    attention_output,
    output,
};

const bert_linear_specs = [_]struct {
    kind: BertLinearSlotKind,
    weight: []const u8,
    bias: []const u8,
    input_intermediate: bool = false,
    output_intermediate: bool = false,
}{
    .{ .kind = .q, .weight = "attention.self.query.weight", .bias = "attention.self.query.bias" },
    .{ .kind = .k, .weight = "attention.self.key.weight", .bias = "attention.self.key.bias" },
    .{ .kind = .v, .weight = "attention.self.value.weight", .bias = "attention.self.value.bias" },
    .{ .kind = .attention_output, .weight = "attention.output.dense.weight", .bias = "attention.output.dense.bias" },
    .{ .kind = .intermediate, .weight = "intermediate.dense.weight", .bias = "intermediate.dense.bias", .output_intermediate = true },
    .{ .kind = .output, .weight = "output.dense.weight", .bias = "output.dense.bias", .input_intermediate = true },
};

const bert_layer_norm_specs = [_]struct {
    kind: BertLayerNormSlotKind,
    weight: []const u8,
    bias: []const u8,
}{
    .{ .kind = .attention_output, .weight = "attention.output.LayerNorm.weight", .bias = "attention.output.LayerNorm.bias" },
    .{ .kind = .output, .weight = "output.LayerNorm.weight", .bias = "output.LayerNorm.bias" },
};

fn metalEncoderFrameEnabled() bool {
    if (@import("builtin").target.cpu.arch.isWasm()) return false;
    return !platform.env.getenvBool("TERMITE_METAL_DISABLE_BERT_ENCODER_FRAME");
}

fn bertLinearSlot(layer: usize, kind: BertLinearSlotKind) usize {
    return layer * bert_linear_specs.len + @intFromEnum(kind);
}

fn bertLayerNormSlot(layer: usize, kind: BertLayerNormSlotKind) usize {
    return layer * bert_layer_norm_specs.len + @intFromEnum(kind);
}

fn bertEmbeddingLayerNormSlot(layer_count: usize) usize {
    return layer_count * bert_layer_norm_specs.len;
}

fn preplanMetalEncoder(cb: *const ComputeBackend, allocator: std.mem.Allocator, config: Config) !bool {
    if (cb.kind() != .metal or !metalEncoderFrameEnabled() or cb.decoderRuntimeHasActiveFrame()) return false;

    const layer_count: usize = @intCast(config.num_hidden_layers);
    const hidden: usize = @intCast(config.hidden_size);
    const intermediate: usize = @intCast(config.intermediate_size);
    const heads: usize = @intCast(config.num_attention_heads);
    if (layer_count == 0 or hidden == 0 or intermediate == 0 or heads == 0 or hidden % heads != 0) return false;

    for (0..layer_count) |layer| {
        for (bert_linear_specs) |spec| {
            var weight_name: [256]u8 = undefined;
            var bias_name: [256]u8 = undefined;
            const weight = try getLayerWeight(cb, allocator, layer, spec.weight, &weight_name);
            defer cb.free(weight);
            const bias = try getLayerWeight(cb, allocator, layer, spec.bias, &bias_name);
            defer cb.free(bias);
            if (!(try cb.decoderRuntimePrepareLinear(&.{
                .slot = bertLinearSlot(layer, spec.kind),
                .weight = weight,
                .bias = bias,
                .in_dim = if (spec.input_intermediate) intermediate else hidden,
                .out_dim = if (spec.output_intermediate) intermediate else hidden,
                .retain_dense_fallback = true,
            }))) return false;
        }

        for (bert_layer_norm_specs) |spec| {
            var weight_name: [256]u8 = undefined;
            var bias_name: [256]u8 = undefined;
            const weight = try getLayerWeight(cb, allocator, layer, spec.weight, &weight_name);
            defer cb.free(weight);
            const bias = try getLayerWeight(cb, allocator, layer, spec.bias, &bias_name);
            defer cb.free(bias);
            if (!(try cb.decoderRuntimePrepareLayerNorm(&.{
                .slot = bertLayerNormSlot(layer, spec.kind),
                .weight = weight,
                .bias = bias,
                .hidden_size = hidden,
            }))) return false;
        }
    }

    const embedding_norm_weight = try cb.getWeight("embeddings.LayerNorm.weight");
    defer cb.free(embedding_norm_weight);
    const embedding_norm_bias = try cb.getWeight("embeddings.LayerNorm.bias");
    defer cb.free(embedding_norm_bias);
    return cb.decoderRuntimePrepareLayerNorm(&.{
        .slot = bertEmbeddingLayerNormSlot(layer_count),
        .weight = embedding_norm_weight,
        .bias = embedding_norm_bias,
        .hidden_size = hidden,
    });
}

/// Run the full BERT encoder forward pass.
/// Returns an owned f32 slice: [batch * seq_len * hidden_size].
pub fn forward(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    attention_mask: []const i64,
    token_type_ids: ?[]const i64,
    batch: usize,
    seq_len: usize,
) ![]f32 {
    const hidden = try forwardCt(cb, allocator, config, input_ids, attention_mask, token_type_ids, batch, seq_len);
    defer cb.free(hidden);
    return cb.toFloat32(hidden, allocator);
}

/// Run the encoder while keeping intermediate tensors on the backend.
pub fn forwardCt(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    attention_mask: []const i64,
    token_type_ids: ?[]const i64,
    batch: usize,
    seq_len: usize,
) !CT {
    const H = config.hidden_size;
    const total = batch * seq_len;
    // ponytail: prepare on first use; move this to session warmup if cold-request latency matters.
    const resident_slots = try preplanMetalEncoder(cb, allocator, config);

    var encoder_frame_active = false;
    if (resident_slots and !cb.decoderRuntimeHasActiveFrame()) {
        encoder_frame_active = try cb.decoderRuntimeBeginFrame();
    }
    errdefer if (encoder_frame_active) cb.decoderRuntimeCancelFrame() catch {};

    // 1. Embeddings: word + position + token_type + LayerNorm
    var hidden = try embeddings(
        cb,
        allocator,
        config,
        input_ids,
        token_type_ids,
        total,
        seq_len,
        H,
        if (resident_slots) bertEmbeddingLayerNormSlot(@intCast(config.num_hidden_layers)) else null,
    );
    errdefer cb.free(hidden);

    // 2. Encoder layers
    for (0..config.num_hidden_layers) |layer| {
        const new_hidden = try encoderLayer(cb, allocator, config, hidden, attention_mask, batch, seq_len, layer, resident_slots);
        cb.free(hidden);
        hidden = new_hidden;
    }

    if (encoder_frame_active) {
        try cb.decoderRuntimeSubmitAndWaitFrame();
        encoder_frame_active = false;
    }
    return hidden;
}

pub fn forwardUntilLayer(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    attention_mask: []const i64,
    token_type_ids: ?[]const i64,
    batch: usize,
    seq_len: usize,
    stop_layer_exclusive: usize,
) ![]f32 {
    const H = config.hidden_size;
    const total = batch * seq_len;
    const clamped_stop = @min(stop_layer_exclusive, config.num_hidden_layers);

    var hidden = try embeddings(cb, allocator, config, input_ids, token_type_ids, total, seq_len, H, null);
    for (0..clamped_stop) |layer| {
        const new_hidden = try encoderLayer(cb, allocator, config, hidden, attention_mask, batch, seq_len, layer, false);
        cb.free(hidden);
        hidden = new_hidden;
    }

    const result = try cb.toFloat32(hidden, allocator);
    cb.free(hidden);
    return result;
}

pub fn forwardFromHidden(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden_in: []const f32,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    start_layer: usize,
) ![]f32 {
    return forwardFromHiddenRange(cb, allocator, config, hidden_in, attention_mask, batch, seq_len, start_layer, config.num_hidden_layers);
}

pub fn forwardFromHiddenRange(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden_in: []const f32,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    start_layer: usize,
    end_layer_exclusive: usize,
) ![]f32 {
    const H = config.hidden_size;
    const total = batch * seq_len;
    if (hidden_in.len != total * H) return error.ShapeMismatch;

    const shape = [_]i32{ @intCast(total), @intCast(H) };
    var hidden = try cb.fromFloat32Shape(hidden_in, &shape);
    const clamped_start = @min(start_layer, config.num_hidden_layers);
    const clamped_end = @max(clamped_start, @min(end_layer_exclusive, config.num_hidden_layers));
    for (clamped_start..clamped_end) |layer| {
        const new_hidden = try encoderLayer(cb, allocator, config, hidden, attention_mask, batch, seq_len, layer, false);
        cb.free(hidden);
        hidden = new_hidden;
    }

    const result = try cb.toFloat32(hidden, allocator);
    cb.free(hidden);
    return result;
}

fn embeddings(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    input_ids: []const i64,
    token_type_ids: ?[]const i64,
    total: usize,
    seq_len: usize,
    H: u32,
    layer_norm_slot: ?usize,
) !CT {
    // Word embeddings
    const word_emb = try cb.getWeight("embeddings.word_embeddings.weight");
    defer cb.free(word_emb);
    var result = try cb.embeddingLookup(word_emb, input_ids, total, H);

    // Position embeddings
    const pos_emb = try cb.getWeight("embeddings.position_embeddings.weight");
    defer cb.free(pos_emb);
    // Build position IDs: [0, 1, ..., seq_len-1] repeated for each batch item
    const pos_ids = try allocator.alloc(i64, total);
    defer allocator.free(pos_ids);
    for (0..total) |i| pos_ids[i] = @intCast(i % seq_len);
    const pos_lookup = try cb.embeddingLookup(pos_emb, pos_ids, total, H);
    defer cb.free(pos_lookup);

    const with_pos = try cb.add(result, pos_lookup);
    cb.free(result);
    result = with_pos;

    // Token type embeddings (optional, not present in distilbert)
    if (config.model_type != .distilbert) {
        if (cb.getWeight("embeddings.token_type_embeddings.weight")) |tt_emb| {
            defer cb.free(tt_emb);
            // Token type IDs: use provided or default to all zeros
            const tt_ids = try allocator.alloc(i64, total);
            defer allocator.free(tt_ids);
            if (token_type_ids) |tids| {
                @memcpy(tt_ids, tids[0..total]);
            } else {
                @memset(tt_ids, 0);
            }
            const tt_lookup = try cb.embeddingLookup(tt_emb, tt_ids, total, H);
            defer cb.free(tt_lookup);
            const with_tt = try cb.add(result, tt_lookup);
            cb.free(result);
            result = with_tt;
        } else |_| {}
    }

    // LayerNorm
    const ln_w = try cb.getWeight("embeddings.LayerNorm.weight");
    defer cb.free(ln_w);
    const ln_b = try cb.getWeight("embeddings.LayerNorm.bias");
    defer cb.free(ln_b);
    const normed = if (layer_norm_slot) |slot|
        (try cb.decoderRuntimeApplyLayerNorm(&.{
            .slot = slot,
            .input = result,
            .hidden_size = H,
            .eps = config.layer_norm_eps,
        })) orelse try cb.layerNorm(result, ln_w, ln_b, H, config.layer_norm_eps)
    else
        try cb.layerNorm(result, ln_w, ln_b, H, config.layer_norm_eps);
    cb.free(result);

    return normed;
}

fn encoderLayer(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    config: Config,
    hidden: CT,
    attention_mask: []const i64,
    batch: usize,
    seq_len: usize,
    layer: usize,
    resident_slots: bool,
) !CT {
    const hidden_dim: usize = @intCast(config.hidden_size);
    const num_heads: usize = @intCast(config.num_attention_heads);
    const head_dim = hidden_dim / num_heads;
    const intermediate_dim: usize = @intCast(config.intermediate_size);
    const total = batch * seq_len;

    var name_buf: [256]u8 = undefined;

    const tp_world_size = tensorParallelWorldSize(cb);
    const use_tp = tp_world_size > 1;
    if (use_tp and (hidden_dim % tp_world_size != 0 or num_heads % tp_world_size != 0 or intermediate_dim % tp_world_size != 0)) {
        return error.InvalidTensorParallelShape;
    }
    const local_num_heads = if (use_tp) num_heads / tp_world_size else num_heads;

    const Q = try layerLinearWithSlot(
        cb,
        allocator,
        layer,
        hidden,
        "attention.self.query.weight",
        "attention.self.query.bias",
        total,
        hidden_dim,
        hidden_dim,
        if (resident_slots) bertLinearSlot(layer, .q) else null,
        &name_buf,
    );
    defer {
        cb.free(Q);
    }

    const K = try layerLinearWithSlot(
        cb,
        allocator,
        layer,
        hidden,
        "attention.self.key.weight",
        "attention.self.key.bias",
        total,
        hidden_dim,
        hidden_dim,
        if (resident_slots) bertLinearSlot(layer, .k) else null,
        &name_buf,
    );
    defer {
        cb.free(K);
    }

    const V = try layerLinearWithSlot(
        cb,
        allocator,
        layer,
        hidden,
        "attention.self.value.weight",
        "attention.self.value.bias",
        total,
        hidden_dim,
        hidden_dim,
        if (resident_slots) bertLinearSlot(layer, .v) else null,
        &name_buf,
    );
    defer {
        cb.free(V);
    }
    const attn_out = try cb.scaledDotProductAttention(Q, K, V, attention_mask, null, batch, seq_len, local_num_heads, head_dim);
    defer {
        cb.free(attn_out);
    }

    const attn_normed = blk: {
        if (resident_slots) {
            if (try cb.decoderRuntimeApplyLinearLayerNorm(&.{
                .linear_slot = bertLinearSlot(layer, .attention_output),
                .layer_norm_slot = bertLayerNormSlot(layer, .attention_output),
                .input = attn_out,
                .residual = hidden,
                .in_dim = hidden_dim,
                .hidden_size = hidden_dim,
                .eps = config.layer_norm_eps,
            })) |output| break :blk output;
        }

        const attn_proj = try layerLinearWithSlot(
            cb,
            allocator,
            layer,
            attn_out,
            "attention.output.dense.weight",
            "attention.output.dense.bias",
            total,
            hidden_dim,
            hidden_dim,
            if (resident_slots) bertLinearSlot(layer, .attention_output) else null,
            &name_buf,
        );
        defer cb.free(attn_proj);
        const attn_res = try cb.add(attn_proj, hidden);
        defer cb.free(attn_res);
        break :blk try layerNormForLayerWithSlot(
            cb,
            allocator,
            layer,
            attn_res,
            "attention.output.LayerNorm.weight",
            "attention.output.LayerNorm.bias",
            hidden_dim,
            config.layer_norm_eps,
            if (resident_slots) bertLayerNormSlot(layer, .attention_output) else null,
            &name_buf,
        );
    };
    defer cb.free(attn_normed);

    if (resident_slots) {
        if (try cb.decoderRuntimeApplyFfnLayerNorm(&.{
            .first_linear_slot = bertLinearSlot(layer, .intermediate),
            .second_linear_slot = bertLinearSlot(layer, .output),
            .layer_norm_slot = bertLayerNormSlot(layer, .output),
            .input = attn_normed,
            .residual = attn_normed,
            .hidden_size = hidden_dim,
            .intermediate_size = intermediate_dim,
            .eps = config.layer_norm_eps,
            .activation = .gelu,
        })) |layer_out| {
            return layer_out;
        }
    }

    const ffn_res = blk: {
        if (resident_slots) {
            if (try cb.runDenseFfnResidual(&.{
                .first_linear_slot = bertLinearSlot(layer, .intermediate),
                .second_linear_slot = bertLinearSlot(layer, .output),
                .input = attn_normed,
                .residual = attn_normed,
                .hidden_size = hidden_dim,
                .intermediate_size = intermediate_dim,
                .activation = .gelu,
            })) |output| break :blk output;
        }

        const ffn_inter = try layerLinearWithSlot(
            cb,
            allocator,
            layer,
            attn_normed,
            "intermediate.dense.weight",
            "intermediate.dense.bias",
            total,
            hidden_dim,
            intermediate_dim,
            if (resident_slots) bertLinearSlot(layer, .intermediate) else null,
            &name_buf,
        );
        defer cb.free(ffn_inter);
        const ffn_gelu = try cb.gelu(ffn_inter);
        defer cb.free(ffn_gelu);
        const ffn_out = try layerLinearWithSlot(
            cb,
            allocator,
            layer,
            ffn_gelu,
            "output.dense.weight",
            "output.dense.bias",
            total,
            intermediate_dim,
            hidden_dim,
            if (resident_slots) bertLinearSlot(layer, .output) else null,
            &name_buf,
        );
        defer cb.free(ffn_out);
        break :blk try cb.add(ffn_out, attn_normed);
    };
    defer cb.free(ffn_res);

    const layer_out = try layerNormForLayerWithSlot(
        cb,
        allocator,
        layer,
        ffn_res,
        "output.LayerNorm.weight",
        "output.LayerNorm.bias",
        hidden_dim,
        config.layer_norm_eps,
        if (resident_slots) bertLayerNormSlot(layer, .output) else null,
        &name_buf,
    );
    return layer_out;
}

/// Build a layer weight name like "encoder.layer.N.suffix" and look it up.
fn getLayerWeight(cb: *const ComputeBackend, _: std.mem.Allocator, layer: usize, suffix: []const u8, buf: *[256]u8) !CT {
    const name = std.fmt.bufPrint(buf, "encoder.layer.{d}.{s}", .{ layer, suffix }) catch return error.NameTooLong;
    return cb.getWeight(name);
}

fn tensorParallelWorldSize(cb: *const ComputeBackend) usize {
    _ = cb;
    return 1;
}

fn layerLinearWithSlot(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    layer: usize,
    input: CT,
    weight_suffix: []const u8,
    bias_suffix: []const u8,
    rows: usize,
    input_dim: usize,
    output_dim: usize,
    slot: ?usize,
    name_buf: *[256]u8,
) !CT {
    if (slot) |prepared_slot| {
        if (try cb.decoderRuntimeApplyLinear(&.{
            .slot = prepared_slot,
            .input = input,
            .in_dim = input_dim,
            .out_dim = output_dim,
        })) |output| return output;
    }
    const weight = try getLayerWeight(cb, allocator, layer, weight_suffix, name_buf);
    defer cb.free(weight);
    const bias = try getLayerWeight(cb, allocator, layer, bias_suffix, name_buf);
    defer cb.free(bias);
    return cb.linear(input, weight, bias, rows, input_dim, output_dim);
}

fn layerNormForLayerWithSlot(
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    layer: usize,
    input: CT,
    weight_suffix: []const u8,
    bias_suffix: []const u8,
    hidden_size: usize,
    eps: f32,
    slot: ?usize,
    name_buf: *[256]u8,
) !CT {
    if (slot) |prepared_slot| {
        if (try cb.decoderRuntimeApplyLayerNorm(&.{
            .slot = prepared_slot,
            .input = input,
            .hidden_size = hidden_size,
            .eps = eps,
        })) |output| return output;
    }
    const weight = try getLayerWeight(cb, allocator, layer, weight_suffix, name_buf);
    defer cb.free(weight);
    const bias = try getLayerWeight(cb, allocator, layer, bias_suffix, name_buf);
    defer cb.free(bias);
    return cb.layerNorm(input, weight, bias, hidden_size, eps);
}

test "BERT resident slot layout is stable and non-overlapping" {
    try std.testing.expectEqual(@as(usize, 143), bertLinearSlot(23, .output));
    try std.testing.expectEqual(@as(usize, 47), bertLayerNormSlot(23, .output));
    try std.testing.expectEqual(@as(usize, 48), bertEmbeddingLayerNormSlot(24));
}
