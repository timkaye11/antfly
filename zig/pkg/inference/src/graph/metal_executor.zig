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

const build_options = @import("build_options");
const std = @import("std");
const platform = @import("antfly_platform");
const a4b_feature_flags = @import("../util/a4b_feature_flags.zig");
const backends = @import("../backends/backends.zig");
const decoder_bitnet_runtime = @import("../backends/decoder_bitnet_runtime.zig");
const decoder_gated_runtime = @import("../backends/decoder_gated_runtime.zig");
const decoder_rms_runtime = @import("../backends/decoder_rms_runtime.zig");
const decoder_tail_runtime = @import("../backends/decoder_tail_runtime.zig");
const metal_runtime = @import("../backends/metal_runtime.zig");
const cache_mod = @import("cache.zig");
const session_factory = @import("../architectures/session_factory.zig");
const gemma4_runtime = @import("../architectures/gemma4_runtime.zig");
const generation = @import("../pipelines/generation.zig");
const debug_timing = @import("../debug_timing.zig");
const gpt_arch = @import("../architectures/gpt.zig");
const gpt_mod = @import("../models/gpt.zig");
const ops = @import("../ops/ops.zig");
const runtime = @import("../runtime/root.zig");
const contracts = @import("backend_contracts.zig");
const model_runtime = @import("model_runtime.zig");
const quant_matmul = @import("quant_matmul.zig");

const c_std = std.c;

pub const TimingStats = model_runtime.RuntimeDebugTimingStats;

var timing_stats = TimingStats{};

pub fn resetTimingStats() void {
    timing_stats = .{};
}

pub fn getTimingStats() TimingStats {
    return timing_stats;
}

fn printRuntimeDebugTimingStats(metal_stats: model_runtime.RuntimeDebugTimingStats, has_runtime: bool) void {
    std.debug.print(
        "metal_executor_ms: runtime_prepare_calls={d} runtime_prepare={d} runtime_prepare_family={d} runtime_prepare_greedy={d} runtime_prepare_fast_hits={d} prefill_calls={d} prefill_prepare={d} prefill_direct_last={d} prefill_direct_family={d} prefill_family_project={d} prefill_family_span_prep={d} prefill_family_quant_attn={d} prefill_family_block_apply={d} prefill_family_frame_wait={d} prefill_family_frame_gpu={d} prefill_fallback={d} decode_begin={d} sample_calls={d} sample_direct={d} sample_fallback={d}\n",
        .{
            metal_stats.runtime_prepare_calls,
            @divTrunc(metal_stats.runtime_prepare_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.runtime_prepare_family_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.runtime_prepare_greedy_nanos, std.time.ns_per_ms),
            metal_stats.runtime_prepare_fast_hits,
            metal_stats.prefill_calls,
            @divTrunc(metal_stats.prefill_prepare_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_last_logits_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_project_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_span_prep_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_quant_attn_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_block_apply_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_frame_wait_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_direct_family_frame_gpu_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.prefill_fallback_logits_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.decode_begin_step_nanos, std.time.ns_per_ms),
            metal_stats.decode_sample_calls,
            @divTrunc(metal_stats.decode_sample_direct_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.decode_sample_fallback_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "metal_executor_ms2: greedy_calls={d} greedy_direct={d} greedy_fallback={d} greedy_token_handoff_attempts={d} greedy_token_handoff_hits={d} greedy_token_handoff_fallbacks={d} greedy_token_seeds={d} ensure_prepared_calls={d} ensure_prepared={d} ensure_sync={d} ensure_family={d} ensure_greedy={d} ensure_fast_hits={d}\n",
        .{
            metal_stats.decode_greedy_calls,
            @divTrunc(metal_stats.decode_greedy_direct_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.decode_greedy_fallback_nanos, std.time.ns_per_ms),
            metal_stats.decode_greedy_device_token_handoff_attempts,
            metal_stats.decode_greedy_device_token_handoff_hits,
            metal_stats.decode_greedy_device_token_handoff_fallbacks,
            metal_stats.decode_greedy_device_token_seeds,
            metal_stats.ensure_prepared_calls,
            @divTrunc(metal_stats.ensure_prepared_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.ensure_prepared_sync_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.ensure_prepared_family_nanos, std.time.ns_per_ms),
            @divTrunc(metal_stats.ensure_prepared_greedy_nanos, std.time.ns_per_ms),
            metal_stats.ensure_prepared_fast_hits,
        },
    );
    if (false and has_runtime) {
        debug_timing.printBackendTimingDetails(
            .metal,
            metal_stats.backend,
            metal_stats.decoder_runtime_ready,
            metal_stats.decoder_runtime_absolute_embeddings_prepared,
        );
    }
    const decoder_rms_stats = decoder_rms_runtime.getTimingStats();
    std.debug.print(
        "decoder_rms_prepare_ms: embed_calls={d} embed_lookup={d} embed_gather={d} embed_scale={d} rms_calls={d} rms={d} linear_calls={d} linear={d} linear_quantized_calls={d} linear_quantized={d} linear_dense_calls={d} linear_dense={d}\n",
        .{
            decoder_rms_stats.embed_calls,
            @divTrunc(decoder_rms_stats.embed_lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_rms_stats.embed_gather_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_rms_stats.embed_scale_nanos, std.time.ns_per_ms),
            decoder_rms_stats.rms_norm_calls,
            @divTrunc(decoder_rms_stats.rms_norm_nanos, std.time.ns_per_ms),
            decoder_rms_stats.linear_calls,
            @divTrunc(decoder_rms_stats.linear_nanos, std.time.ns_per_ms),
            decoder_rms_stats.linear_quantized_calls,
            @divTrunc(decoder_rms_stats.linear_quantized_nanos, std.time.ns_per_ms),
            decoder_rms_stats.linear_dense_calls,
            @divTrunc(decoder_rms_stats.linear_dense_nanos, std.time.ns_per_ms),
        },
    );
    const decoder_gated_stats = decoder_gated_runtime.getTimingStats();
    const prefill_named_block_nanos =
        decoder_gated_stats.prefill_block_attention_qkv_nanos +
        decoder_gated_stats.prefill_block_attention_head_norm_nanos +
        decoder_gated_stats.prefill_block_attention_apply_nanos +
        decoder_gated_stats.prefill_block_attention_project_nanos +
        decoder_gated_stats.prefill_block_attention_fused_residual_nanos +
        decoder_gated_stats.prefill_block_ffn_norm_nanos +
        decoder_gated_stats.prefill_block_ffn_fused_nanos +
        decoder_gated_stats.prefill_block_ple_nanos +
        decoder_gated_stats.prefill_block_output_scale_nanos +
        decoder_gated_stats.prefill_frame_begin_nanos +
        decoder_gated_stats.prefill_frame_layer_spec_nanos +
        decoder_gated_stats.prefill_frame_plan_nanos +
        decoder_gated_stats.prefill_frame_execute_nanos;
    const prefill_block_unattributed_nanos = if (decoder_gated_stats.prefill_block_nanos > prefill_named_block_nanos)
        decoder_gated_stats.prefill_block_nanos - prefill_named_block_nanos
    else
        0;
    std.debug.print(
        "decoder_gated_prepare_ms: calls={d} greedy={d} lookup={d} norm_prep={d} linear_prep={d} final_lookup={d} final_prep={d} greedy_fail={d} attn_pre_fail={d} attn_post_fail={d} ffn_pre_fail={d} ffn_post_fail={d} q_fail={d} k_fail={d} v_fail={d} out_fail={d} gate_fail={d} up_fail={d} down_fail={d} final_fail={d} out_shape_rank={d} out_shape0={d} out_shape1={d}\n",
        .{
            decoder_gated_stats.prepare_calls,
            @divTrunc(decoder_gated_stats.prepare_greedy_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.norm_prep_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.linear_prep_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.final_lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.final_prep_nanos, std.time.ns_per_ms),
            decoder_gated_stats.prepare_greedy_failures,
            decoder_gated_stats.prepare_attn_pre_norm_failures,
            decoder_gated_stats.prepare_attn_post_norm_failures,
            decoder_gated_stats.prepare_ffn_pre_norm_failures,
            decoder_gated_stats.prepare_ffn_post_norm_failures,
            decoder_gated_stats.prepare_attn_q_failures,
            decoder_gated_stats.prepare_attn_k_failures,
            decoder_gated_stats.prepare_attn_v_failures,
            decoder_gated_stats.prepare_attn_out_failures,
            decoder_gated_stats.prepare_mlp_gate_failures,
            decoder_gated_stats.prepare_mlp_up_failures,
            decoder_gated_stats.prepare_mlp_down_failures,
            decoder_gated_stats.prepare_final_norm_failures,
            decoder_gated_stats.prepare_attn_out_first_rank,
            decoder_gated_stats.prepare_attn_out_first_dim0,
            decoder_gated_stats.prepare_attn_out_first_dim1,
        },
    );
    std.debug.print(
        "decoder_gated_prefill_ms: calls={d} embed={d} ple_prepare={d} block={d} tail={d} compare={d} sync={d} last_slice={d}\n",
        .{
            decoder_gated_stats.prefill_calls,
            @divTrunc(decoder_gated_stats.prefill_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_prepare_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_tail_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_compare_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_sync_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_last_hidden_slice_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_preplan_frame_ms: layer_specs={d} plan={d}\n",
        .{
            @divTrunc(decoder_gated_stats.preplan_frame_layer_spec_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.preplan_frame_plan_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_prefill_ple_ms: lookup={d} embedding={d} model_proj={d} norm={d} combine={d} fallback={d}\n",
        .{
            @divTrunc(decoder_gated_stats.prefill_ple_lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_embedding_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_model_proj_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_combine_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_ple_fallback_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_prefill_frame_ms: begin={d} layer_specs={d} plan={d} execute={d} finish={d}\n",
        .{
            @divTrunc(decoder_gated_stats.prefill_frame_begin_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_frame_layer_spec_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_frame_plan_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_frame_execute_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_frame_finish_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_prefill_breakdown_ms: named_block={d} unattributed_block={d} qkv={d} head_norm={d} attn_apply={d} attn_project={d} attn_fused_residual={d} ffn_norm={d} ffn_fused={d} ple={d} output_scale={d}\n",
        .{
            @divTrunc(prefill_named_block_nanos, std.time.ns_per_ms),
            @divTrunc(prefill_block_unattributed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_qkv_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_head_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_apply_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_project_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_fused_residual_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ffn_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ffn_fused_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ple_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_output_scale_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_prefill_ops: tokens={d} layers={d} attn_norm={d} q_linear={d} qkv={d} qkv_fused={d} qkv_split={d} kv_pair={d} head_norm={d} kv_seed={d} attn_apply={d} attn_out_linear={d} attn_post_norm={d} attn_residual_add={d}\n",
        .{
            decoder_gated_stats.prefill_query_tokens,
            decoder_gated_stats.prefill_layers,
            decoder_gated_stats.prefill_attn_norm_ops,
            decoder_gated_stats.prefill_q_linear_ops,
            decoder_gated_stats.prefill_qkv_ops,
            decoder_gated_stats.prefill_qkv_fused_ops,
            decoder_gated_stats.prefill_qkv_split_ops,
            decoder_gated_stats.prefill_kv_pair_ops,
            decoder_gated_stats.prefill_head_norm_ops,
            decoder_gated_stats.prefill_kv_seed_ops,
            decoder_gated_stats.prefill_attention_apply_ops,
            decoder_gated_stats.prefill_attn_out_linear_ops,
            decoder_gated_stats.prefill_attn_post_norm_ops,
            decoder_gated_stats.prefill_attn_residual_add_ops,
        },
    );
    std.debug.print(
        "decoder_gated_prefill_ffn_ops: ffn_norm={d} ffn_fused={d} ffn_split={d} ffn_pair={d} ffn_activation={d} ffn_multiply={d} ffn_down_linear={d} ffn_post_norm={d} ffn_residual_add={d} ple={d} ple_direct={d} ple_fallback={d} output_scale={d} tail_logits={d}\n",
        .{
            decoder_gated_stats.prefill_ffn_norm_ops,
            decoder_gated_stats.prefill_ffn_fused_ops,
            decoder_gated_stats.prefill_ffn_split_ops,
            decoder_gated_stats.prefill_ffn_pair_ops,
            decoder_gated_stats.prefill_ffn_activation_ops,
            decoder_gated_stats.prefill_ffn_multiply_ops,
            decoder_gated_stats.prefill_ffn_down_linear_ops,
            decoder_gated_stats.prefill_ffn_post_norm_ops,
            decoder_gated_stats.prefill_ffn_residual_add_ops,
            decoder_gated_stats.prefill_ple_ops,
            decoder_gated_stats.prefill_ple_direct_hits,
            decoder_gated_stats.prefill_ple_fallbacks,
            decoder_gated_stats.prefill_output_scale_ops,
            decoder_gated_stats.prefill_tail_logits_ops,
        },
    );
    std.debug.print(
        "decoder_gated_block_ms: prefill_attn={d} prefill_attn_project={d} prefill_attn_apply={d} prefill_ffn={d} prefill_ple={d} prefill_scale={d} greedy_attn={d} greedy_attn_project={d} greedy_attn_apply={d} greedy_ffn={d} greedy_ple={d} greedy_scale={d} sampled_attn={d} sampled_attn_project={d} sampled_attn_apply={d} sampled_ffn={d} sampled_ple={d} sampled_scale={d}\n",
        .{
            @divTrunc(decoder_gated_stats.prefill_block_attention_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_project_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_apply_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ffn_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ple_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_output_scale_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_project_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_apply_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_ffn_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_ple_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_output_scale_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_project_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_apply_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_ffn_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_ple_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_output_scale_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_attn_ms: prefill_qkv={d} prefill_head_norm={d} prefill_fused_residual={d} greedy_qkv={d} greedy_head_norm={d} greedy_fused_residual={d} sampled_qkv={d} sampled_head_norm={d} sampled_fused_residual={d}\n",
        .{
            @divTrunc(decoder_gated_stats.prefill_block_attention_qkv_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_head_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_attention_fused_residual_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_qkv_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_head_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_attention_fused_residual_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_qkv_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_head_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_attention_fused_residual_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_ffn_ms: prefill_norm={d} prefill_fused={d} greedy_norm={d} greedy_fused={d} sampled_norm={d} sampled_fused={d}\n",
        .{
            @divTrunc(decoder_gated_stats.prefill_block_ffn_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.prefill_block_ffn_fused_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_ffn_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_ffn_fused_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_ffn_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_ffn_fused_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_gated_qkv: gemma_fused_hits={d} gemma_fused_fallbacks={d} gemma_fused_attn_residual_hits={d} gemma_fused_attn_residual_fallbacks={d} gemma_fused_ffn_hits={d} gemma_fused_ffn_fallbacks={d}\n",
        .{
            decoder_gated_stats.gemma_fused_qkv_hits,
            decoder_gated_stats.gemma_fused_qkv_fallbacks,
            decoder_gated_stats.gemma_fused_attn_residual_hits,
            decoder_gated_stats.gemma_fused_attn_residual_fallbacks,
            decoder_gated_stats.gemma_fused_ffn_hits,
            decoder_gated_stats.gemma_fused_ffn_fallbacks,
        },
    );
    std.debug.print(
        "decoder_gated_decode_ms: greedy_calls={d} greedy_layer_specs={d} greedy_embed={d} greedy_block={d} greedy_tail={d} sampled_calls={d} sampled_embed={d} sampled_block={d} sampled_tail={d}\n",
        .{
            decoder_gated_stats.greedy_calls,
            @divTrunc(decoder_gated_stats.greedy_layer_spec_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.greedy_tail_nanos, std.time.ns_per_ms),
            decoder_gated_stats.sampled_calls,
            @divTrunc(decoder_gated_stats.sampled_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_gated_stats.sampled_tail_nanos, std.time.ns_per_ms),
        },
    );
    const decoder_bitnet_stats = decoder_bitnet_runtime.getTimingStats();
    std.debug.print(
        "decoder_bitnet_prepare_ms: calls={d} greedy={d} lookup={d} norm_prep={d} linear_prep={d} final_lookup={d} final_prep={d}\n",
        .{
            decoder_bitnet_stats.prepare_calls,
            @divTrunc(decoder_bitnet_stats.prepare_greedy_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.norm_prep_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.linear_prep_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.final_lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.final_prep_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_bitnet_prefill_ms: calls={d} embed={d} block={d} tail={d}\n",
        .{
            decoder_bitnet_stats.prefill_calls,
            @divTrunc(decoder_bitnet_stats.prefill_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.prefill_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.prefill_tail_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "decoder_bitnet_decode_ms: greedy_calls={d} greedy_embed={d} greedy_block={d} greedy_tail={d} sampled_calls={d} sampled_embed={d} sampled_block={d} sampled_tail={d}\n",
        .{
            decoder_bitnet_stats.greedy_calls,
            @divTrunc(decoder_bitnet_stats.greedy_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.greedy_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.greedy_tail_nanos, std.time.ns_per_ms),
            decoder_bitnet_stats.sampled_calls,
            @divTrunc(decoder_bitnet_stats.sampled_embed_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.sampled_block_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_bitnet_stats.sampled_tail_nanos, std.time.ns_per_ms),
        },
    );
    const decoder_tail_stats = decoder_tail_runtime.getTimingStats();
    std.debug.print(
        "decoder_tail_ms: logits_calls={d} logits_norm={d} logits_lm_head={d} logits_linear={d} sampled_calls={d} sampled_logits={d} sampled_device={d} sampled_host={d}\n",
        .{
            decoder_tail_stats.logits_calls,
            @divTrunc(decoder_tail_stats.logits_norm_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_tail_stats.logits_lm_head_lookup_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_tail_stats.logits_linear_nanos, std.time.ns_per_ms),
            decoder_tail_stats.sampled_calls,
            @divTrunc(decoder_tail_stats.sampled_logits_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_tail_stats.sampled_device_sample_nanos, std.time.ns_per_ms),
            @divTrunc(decoder_tail_stats.sampled_host_sample_nanos, std.time.ns_per_ms),
        },
    );
    const provider_stats = metal_stats.backend.provider;
    std.debug.print(
        "metal_active_decode_ms: project={d} span_prep={d} encode={d} apply={d} replace_span={d} attention_span={d} attention_prefix={d} gated_ffn_residual={d} command_wait={d} gpu={d} quant_attention={d} quant_ffn={d}\n",
        .{
            @divTrunc(provider_stats.compressed_block_project_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_span_prep_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_encode_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_apply_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_replace_span_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_attention_span_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_attention_prefix_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_gated_ffn_residual_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_command_wait_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_gpu_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_quantized_attention_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.compressed_block_quantized_ffn_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "metal_active_decode_kernels: layers={d} attention_f32={d} quant_linear={d} quant_attention_linear={d} quant_ffn_down_linear={d} quant_ple_linear={d} quant_gate_up_pair={d} rms_norm={d} rms_norm_add={d} layer_norm={d} add={d} head_norm_rope_fused={d} blit_copies={d}\n",
        .{
            provider_stats.active_decode_layers,
            provider_stats.active_decode_attention_f32_kernels,
            provider_stats.active_decode_quant_linear_kernels,
            provider_stats.active_decode_quant_attention_linear_kernels,
            provider_stats.active_decode_quant_ffn_down_linear_kernels,
            provider_stats.active_decode_quant_ple_linear_kernels,
            provider_stats.active_decode_quant_gate_up_pair_kernels,
            provider_stats.active_decode_rms_norm_kernels,
            provider_stats.active_decode_rms_norm_add_kernels,
            provider_stats.active_decode_layer_norm_kernels,
            provider_stats.active_decode_add_kernels,
            provider_stats.active_decode_head_norm_rope_fused_kernels,
            provider_stats.active_decode_blit_copies,
        },
    );
    std.debug.print(
        "metal_active_decode_ops: layer_input_direct={d}/{d} attn_norm={d} q_linear={d} qkv={d} head_norm={d} rope={d} head_norm_rope_fused={d} ple={d} fused_argmax={d} split_argmax={d} nucleus_samples={d} frame={d}/{d} disabled={d} scratch_failures={d} fallbacks={d} batch_fallbacks={d} initial_tensor_fallbacks={d} layer_fallbacks={d} tail_fallbacks={d}\n",
        .{
            provider_stats.active_decode_layer_input_direct_hits,
            provider_stats.active_decode_layer_input_direct_attempts,
            provider_stats.active_decode_attn_norm_ops,
            provider_stats.active_decode_q_linear_ops,
            provider_stats.active_decode_qkv_ops,
            provider_stats.active_decode_head_norm_ops,
            provider_stats.active_decode_rope_ops,
            provider_stats.active_decode_head_norm_rope_fused_ops,
            provider_stats.active_decode_ple_ops,
            provider_stats.active_decode_final_fused_argmax_ops,
            provider_stats.active_decode_final_split_argmax_ops,
            provider_stats.metal_runtime_full_vocab_nucleus_sample_calls,
            provider_stats.active_decode_frame_successes,
            provider_stats.active_decode_frame_attempts,
            provider_stats.active_decode_frame_disabled,
            provider_stats.active_decode_frame_scratch_failures,
            provider_stats.active_decode_frame_fallbacks,
            provider_stats.active_decode_frame_batch_fallbacks,
            provider_stats.active_decode_frame_initial_tensor_fallbacks,
            provider_stats.active_decode_frame_layer_fallbacks,
            provider_stats.active_decode_frame_tail_fallbacks,
        },
    );
    std.debug.print(
        "metal_memory: tensor_device_live_mb={d} tensor_device_peak_mb={d} tensor_device_created={d} tensor_device_released={d} host_mirror_live_mb={d} host_mirror_peak_mb={d} host_mirror_download_mb={d} host_mirror_allocs={d} host_mirror_frees={d} to_host_device_calls={d} quant_slots={d} quant_raw_ref_mb={d} quant_raw_owned_mb={d} quant_prepared_mb={d} quant_runtime_slots={d} quant_runtime_mb={d} dense_host_mb={d} norm_host_mb={d} gathered_entries={d} gathered_device_mb={d} gathered_encoded_host_mb={d}\n",
        .{
            provider_stats.metal_tensor_device_owned_live_bytes / (1024 * 1024),
            provider_stats.metal_tensor_device_owned_peak_live_bytes / (1024 * 1024),
            provider_stats.metal_tensor_device_owned_buffers_created,
            provider_stats.metal_tensor_device_owned_buffers_released,
            provider_stats.metal_tensor_host_mirror_live_bytes / (1024 * 1024),
            provider_stats.metal_tensor_host_mirror_peak_live_bytes / (1024 * 1024),
            provider_stats.metal_tensor_host_mirror_download_bytes / (1024 * 1024),
            provider_stats.metal_tensor_host_mirror_allocations,
            provider_stats.metal_tensor_host_mirror_frees,
            provider_stats.metal_tensor_to_host_device_calls,
            provider_stats.metal_provider_quantized_slots,
            provider_stats.metal_provider_quantized_raw_bytes / (1024 * 1024),
            provider_stats.metal_provider_quantized_raw_owned_bytes / (1024 * 1024),
            provider_stats.metal_provider_quantized_prepared_bytes / (1024 * 1024),
            provider_stats.metal_provider_quantized_runtime_prepared_slots,
            provider_stats.metal_provider_quantized_runtime_prepared_bytes / (1024 * 1024),
            provider_stats.metal_provider_dense_slot_host_bytes / (1024 * 1024),
            provider_stats.metal_provider_norm_slot_host_bytes / (1024 * 1024),
            provider_stats.metal_provider_gathered_span_entries,
            provider_stats.metal_provider_gathered_span_device_bytes / (1024 * 1024),
            provider_stats.metal_provider_gathered_span_encoded_host_bytes / (1024 * 1024),
        },
    );
    std.debug.print(
        "metal_quant_runtime_prepare: private_slots={d} private_mb={d} private_ms={d} mapped_slots={d} mapped_mb={d} mapped_ms={d} mapped_attempts={d} mapped_fallbacks={d} mapped_failures={d}\n",
        .{
            provider_stats.metal_provider_quantized_runtime_private_slots,
            provider_stats.metal_provider_quantized_runtime_private_bytes / (1024 * 1024),
            @divTrunc(provider_stats.metal_provider_quantized_runtime_private_nanos, std.time.ns_per_ms),
            provider_stats.metal_provider_quantized_runtime_mapped_slots,
            provider_stats.metal_provider_quantized_runtime_mapped_bytes / (1024 * 1024),
            @divTrunc(provider_stats.metal_provider_quantized_runtime_mapped_nanos, std.time.ns_per_ms),
            provider_stats.metal_provider_quantized_runtime_mapped_attempts,
            provider_stats.metal_provider_quantized_runtime_mapped_fallbacks,
            provider_stats.metal_provider_quantized_runtime_mapped_failures,
        },
    );
    std.debug.print(
        "metal_runtime_residency_memory: private_mb={d} shared_mb={d} managed_mb={d} embedding_logical_mb={d}\n",
        .{
            provider_stats.metal_runtime_private_bytes / (1024 * 1024),
            provider_stats.metal_runtime_shared_bytes / (1024 * 1024),
            provider_stats.metal_runtime_managed_bytes / (1024 * 1024),
            provider_stats.metal_runtime_embedding_logical_bytes / (1024 * 1024),
        },
    );
    std.debug.print(
        "metal_runtime_memory: buffers={d} total_mb={d} embedding_mb={d} norm_mb={d} dense_linear_mb={d} dense_linear_buffers={d} dense_largest_slot={d} dense_largest_mb={d} dense_largest_in={d} dense_largest_out={d} dense_weight_mb={d} dense_f32_mb={d} dense_bf16_mb={d} dense_f16_mb={d} dense_f32_slots={d} dense_bf16_slots={d} dense_f16_slots={d} quant_linear_mb={d} scratch_mb={d} scratch_pool_mb={d} scratch_pool_slots={d} scratch_pool_in_use={d} scratch_pool_pending={d} reuse_pool_mb={d} reuse_pool_slots={d} reuse_pool_peak_slots={d} reuse_allocs={d} reuse_hits={d} attention_span_mb={d} hidden_state_mb={d} frame_retained_mb={d}\n",
        .{
            provider_stats.metal_runtime_buffer_count,
            provider_stats.metal_runtime_total_bytes / (1024 * 1024),
            provider_stats.metal_runtime_embedding_bytes / (1024 * 1024),
            provider_stats.metal_runtime_norm_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_buffer_count,
            provider_stats.metal_runtime_dense_linear_largest_slot,
            provider_stats.metal_runtime_dense_linear_largest_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_largest_in_dim,
            provider_stats.metal_runtime_dense_linear_largest_out_dim,
            provider_stats.metal_runtime_dense_linear_weight_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_f32_weight_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_bf16_weight_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_f16_weight_bytes / (1024 * 1024),
            provider_stats.metal_runtime_dense_linear_f32_slots,
            provider_stats.metal_runtime_dense_linear_bf16_slots,
            provider_stats.metal_runtime_dense_linear_f16_slots,
            provider_stats.metal_runtime_quant_linear_bytes / (1024 * 1024),
            provider_stats.metal_runtime_scratch_bytes / (1024 * 1024),
            provider_stats.metal_runtime_scratch_pool_bytes / (1024 * 1024),
            provider_stats.metal_runtime_scratch_pool_slots,
            provider_stats.metal_runtime_scratch_pool_in_use_slots,
            provider_stats.metal_runtime_scratch_pool_pending_slots,
            provider_stats.metal_runtime_reuse_pool_bytes / (1024 * 1024),
            provider_stats.metal_runtime_reuse_pool_slots,
            provider_stats.metal_runtime_reuse_pool_peak_slots,
            provider_stats.metal_runtime_reuse_alloc_count,
            provider_stats.metal_runtime_reuse_hit_count,
            provider_stats.metal_runtime_attention_span_bytes / (1024 * 1024),
            provider_stats.metal_runtime_hidden_state_bytes / (1024 * 1024),
            provider_stats.metal_runtime_frame_retained_bytes / (1024 * 1024),
        },
    );
    std.debug.print(
        "metal_runtime_memory_plan: graph_plan_mb={d} graph_plan_slots={d} graph_plan_active={d} graph_plan_count={d} graph_plan_allocs={d} graph_plan_reuses={d}\n",
        .{
            provider_stats.metal_runtime_graph_plan_bytes / (1024 * 1024),
            provider_stats.metal_runtime_graph_plan_slots,
            provider_stats.metal_runtime_graph_plan_active,
            provider_stats.metal_runtime_graph_plan_count,
            provider_stats.metal_runtime_graph_plan_allocations,
            provider_stats.metal_runtime_graph_plan_reuses,
        },
    );
    std.debug.print(
        "metal_runtime_encoders: compute={d} blit={d} last_frame_compute={d} last_frame_blit={d} planned_scopes={d} planned_barriers={d}\n",
        .{
            provider_stats.metal_runtime_compute_encoder_count,
            provider_stats.metal_runtime_blit_encoder_count,
            provider_stats.metal_runtime_last_frame_compute_encoder_count,
            provider_stats.metal_runtime_last_frame_blit_encoder_count,
            provider_stats.metal_runtime_last_frame_planned_compute_scope_count,
            provider_stats.metal_runtime_last_frame_planned_barrier_count,
        },
    );
    std.debug.print(
        "metal_runtime_blit_sources: upload={d} copy={d} slice={d} attention_span={d} ffn_copy={d} embedding={d} other={d}\n",
        .{
            provider_stats.metal_runtime_last_frame_blit_buffer_upload_count,
            provider_stats.metal_runtime_last_frame_blit_buffer_copy_count,
            provider_stats.metal_runtime_last_frame_blit_buffer_slice_count,
            provider_stats.metal_runtime_last_frame_blit_attention_span_count,
            provider_stats.metal_runtime_last_frame_blit_ffn_copy_count,
            provider_stats.metal_runtime_last_frame_blit_embedding_count,
            provider_stats.metal_runtime_last_frame_blit_other_count,
        },
    );
    std.debug.print(
        "metal_runtime_compute_sources: quant_linear={d} quant_qkv={d} quant_pair_act={d} attention={d} rms_norm={d} head_rope={d} ffn={d} ple={d} tail={d} embedding={d} dense_linear={d} layer={d} other={d}\n",
        .{
            provider_stats.metal_runtime_last_frame_compute_quant_linear_count,
            provider_stats.metal_runtime_last_frame_compute_quant_qkv_count,
            provider_stats.metal_runtime_last_frame_compute_quant_pair_act_count,
            provider_stats.metal_runtime_last_frame_compute_attention_count,
            provider_stats.metal_runtime_last_frame_compute_rms_norm_count,
            provider_stats.metal_runtime_last_frame_compute_head_rope_count,
            provider_stats.metal_runtime_last_frame_compute_ffn_count,
            provider_stats.metal_runtime_last_frame_compute_ple_count,
            provider_stats.metal_runtime_last_frame_compute_tail_count,
            provider_stats.metal_runtime_last_frame_compute_embedding_count,
            provider_stats.metal_runtime_last_frame_compute_dense_linear_count,
            provider_stats.metal_runtime_last_frame_compute_layer_count,
            provider_stats.metal_runtime_last_frame_compute_other_count,
        },
    );
    std.debug.print(
        "metal_runtime_compute_regions: attention={d} attention_project={d} ffn_norm={d} ffn={d} ple={d} tail={d} embedding={d} layer={d} other={d}\n",
        .{
            provider_stats.metal_runtime_last_frame_compute_region_attention_count,
            provider_stats.metal_runtime_last_frame_compute_region_attention_project_count,
            provider_stats.metal_runtime_last_frame_compute_region_ffn_norm_count,
            provider_stats.metal_runtime_last_frame_compute_region_ffn_count,
            provider_stats.metal_runtime_last_frame_compute_region_ple_count,
            provider_stats.metal_runtime_last_frame_compute_region_tail_count,
            provider_stats.metal_runtime_last_frame_compute_region_embedding_count,
            provider_stats.metal_runtime_last_frame_compute_region_layer_count,
            provider_stats.metal_runtime_last_frame_compute_region_other_count,
        },
    );
    const command_ops = provider_stats.metal_runtime_last_frame_planned_command_op_kind_counts;
    const command_operators = provider_stats.metal_runtime_last_frame_planned_command_operator_counts;
    const command_dispatch = provider_stats.metal_runtime_last_frame_planned_command_quant_dispatch_counts;
    std.debug.print(
        "metal_runtime_command_ops: total={d} attention_pre_norm={d} qkv_linear={d} q_head_norm_rope={d} k_head_norm_rope={d} v_norm={d} kv_seed={d} attention={d} attention_output_linear={d} attention_post_norm_residual={d} ffn_pre_norm_scale={d} ffn_gate_up_activation={d} ffn_down_linear={d} ffn_post_norm_residual={d} ple_gate_activation={d} ple_projection={d} ple_post_norm_residual={d} tail_final_norm={d} tail_lm_head={d} tail_argmax={d}\n",
        .{
            provider_stats.metal_runtime_last_frame_planned_command_op_count,
            command_ops[1],
            command_ops[2],
            command_ops[3],
            command_ops[4],
            command_ops[5],
            command_ops[6],
            command_ops[7],
            command_ops[8],
            command_ops[9],
            command_ops[10],
            command_ops[11],
            command_ops[12],
            command_ops[13],
            command_ops[14],
            command_ops[15],
            command_ops[16],
            command_ops[17],
            command_ops[18],
            command_ops[19],
        },
    );
    std.debug.print(
        "metal_runtime_command_operators: fallback={d} mul_mv={d} mul_mv_ext={d} mul_mm={d} get_rows={d} set_rows={d} cpy_q_to_f32={d} cpy_f32_to_q={d} attention_flash={d} attention_paged={d} attention_quantized_kv={d} dispatch_scalar={d} dispatch_mmv={d} dispatch_small_batch={d} dispatch_mm={d}\n",
        .{
            command_operators[0],
            command_operators[1],
            command_operators[2],
            command_operators[3],
            command_operators[4],
            command_operators[5],
            command_operators[6],
            command_operators[7],
            command_operators[8],
            command_operators[9],
            command_operators[10],
            command_dispatch[0],
            command_dispatch[1],
            command_dispatch[2],
            command_dispatch[3],
        },
    );
    std.debug.print(
        "metal_prefill_frame_executor_ms: layer_runtime={d} setup={d} block={d} staged={d} full_frames={d} windows={d} layer_contracts={d} tail_contracts={d}\n",
        .{
            @divTrunc(provider_stats.prefill_frame_executor_layer_runtime_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.prefill_frame_executor_layer_setup_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.prefill_frame_executor_layer_block_nanos, std.time.ns_per_ms),
            @divTrunc(provider_stats.prefill_frame_executor_layer_staged_nanos, std.time.ns_per_ms),
            provider_stats.prefill_frame_contract_full_frames,
            provider_stats.prefill_frame_contract_windows,
            provider_stats.prefill_frame_executor_layer_contracts,
            provider_stats.prefill_frame_executor_tail_contracts,
        },
    );
    std.debug.print(
        "metal_q8_0_dispatch: scalar={d} mmv={d} small_batch={d} mm={d} rows_1={d} rows_2_8={d} rows_9_64={d} rows_65_plus={d} pair_act_mm_out_f16={d} linear_mm_in_f16={d} pair_act_rms_mmv_out_f16={d} linear_mmv_in_f16={d}\n",
        .{
            provider_stats.metal_runtime_q8_0_linear_dispatch_scalar,
            provider_stats.metal_runtime_q8_0_linear_dispatch_mmv,
            provider_stats.metal_runtime_q8_0_linear_dispatch_small_batch,
            provider_stats.metal_runtime_q8_0_linear_dispatch_mm,
            provider_stats.metal_runtime_q8_0_linear_rows_1,
            provider_stats.metal_runtime_q8_0_linear_rows_2_8,
            provider_stats.metal_runtime_q8_0_linear_rows_9_64,
            provider_stats.metal_runtime_q8_0_linear_rows_65_plus,
            provider_stats.metal_runtime_q8_0_pair_activation_mm_f16_output,
            provider_stats.metal_runtime_q8_0_linear_mm_f16_input,
            provider_stats.metal_runtime_q8_0_pair_activation_rms_scale_mmv_f16_output,
            provider_stats.metal_runtime_q8_0_linear_mmv_f16_input,
        },
    );
    std.debug.print(
        "metal_attention_dispatch: paged_1x={d} decode_gqa_split={d} decode_gqa_split_fallback={d} generated_decode_1x={d} generated_flash_prefill={d} generated_flash_prefill_hd512={d} prefill_direct_kv={d} prefill_paged_kv={d} generated_rms_norm={d}\n",
        .{
            provider_stats.metal_runtime_paged_attention_1x_calls,
            provider_stats.metal_runtime_decode_gqa_split_calls,
            provider_stats.metal_runtime_decode_gqa_split_fallback_calls,
            provider_stats.metal_runtime_generated_attention_decode_1x_calls,
            provider_stats.metal_runtime_generated_attention_flash_prefill_calls,
            provider_stats.metal_runtime_generated_attention_flash_prefill_hd512_calls,
            provider_stats.metal_runtime_attention_prefill_direct_kv_calls,
            provider_stats.metal_runtime_attention_prefill_paged_kv_calls,
            provider_stats.metal_runtime_generated_rms_norm_calls,
        },
    );
    std.debug.print(
        "metal_prepared_frame: fast_path={d} fallback={d}\n",
        .{
            provider_stats.metal_runtime_prepared_frame_fast_path_calls,
            provider_stats.metal_runtime_prepared_frame_fallback_calls,
        },
    );
    std.debug.print(
        "metal_q4_0_dispatch: linear_reduce={d} linear_reduce_rows={d}/{d}/{d}/{d} linear_reduce_in_f16={d} linear_reduce_out_f16={d} linear_reduce_in_f16_out_f16={d} linear_reduce_sumsq={d} pair_act_reduce={d} pair_act_reduce_out_f16={d} pair_act_rms_scale_reduce_out_f16={d} activation_rhs_reduce={d} activation_rhs_reduce_out_f16={d} rms_norm_add_sumsq={d} pair_reduce={d} pair={d}\n",
        .{
            provider_stats.metal_runtime_q4_0_linear_reduce,
            provider_stats.metal_runtime_q4_0_linear_reduce_rows_1,
            provider_stats.metal_runtime_q4_0_linear_reduce_rows_2_8,
            provider_stats.metal_runtime_q4_0_linear_reduce_rows_9_64,
            provider_stats.metal_runtime_q4_0_linear_reduce_rows_65_plus,
            provider_stats.metal_runtime_q4_0_linear_reduce_f16_input,
            provider_stats.metal_runtime_q4_0_linear_reduce_f16_output,
            provider_stats.metal_runtime_q4_0_linear_reduce_f16_input_f16_output,
            provider_stats.metal_runtime_q4_0_linear_reduce_sumsq,
            provider_stats.metal_runtime_q4_0_pair_activation_reduce,
            provider_stats.metal_runtime_q4_0_pair_activation_reduce_f16_output,
            provider_stats.metal_runtime_q4_0_pair_activation_rms_scale_reduce_f16_output,
            provider_stats.metal_runtime_q4_0_activation_rhs_reduce,
            provider_stats.metal_runtime_q4_0_activation_rhs_reduce_f16_output,
            provider_stats.metal_runtime_rms_norm_add_sumsq,
            provider_stats.metal_runtime_q4_0_pair_reduce,
            provider_stats.metal_runtime_q4_0_pair,
        },
    );
    std.debug.print(
        "metal_q4_0_encode_us: linear_reduce={d} pair_reduce={d} pair_act_reduce={d} activation_rhs_reduce={d}\n",
        .{
            @divTrunc(provider_stats.metal_runtime_q4_0_linear_reduce_encode_nanos, 1000),
            @divTrunc(provider_stats.metal_runtime_q4_0_pair_reduce_encode_nanos, 1000),
            @divTrunc(provider_stats.metal_runtime_q4_0_pair_activation_reduce_encode_nanos, 1000),
            @divTrunc(provider_stats.metal_runtime_q4_0_activation_rhs_reduce_encode_nanos, 1000),
        },
    );
    std.debug.print(
        "metal_q4_0_policy: mmv_nr4_nsg2={d} mmv_nr8_nsg2={d} mmv_nr4_nsg4={d} mmv_nr8_nsg4={d} mmv_variant_fallbacks={d} mm_sg_aligned={d} mm_sg_aligned_tail={d} mm_sg_unrolled={d}\n",
        .{
            provider_stats.metal_runtime_q4_0_mmv_nr4_nsg2_dispatches,
            provider_stats.metal_runtime_q4_0_mmv_nr8_nsg2_dispatches,
            provider_stats.metal_runtime_q4_0_mmv_nr4_nsg4_dispatches,
            provider_stats.metal_runtime_q4_0_mmv_nr8_nsg4_dispatches,
            provider_stats.metal_runtime_q4_0_mmv_variant_fallbacks,
            provider_stats.metal_runtime_q4_0_mm_sg_aligned_dispatches,
            provider_stats.metal_runtime_q4_0_mm_sg_aligned_tail_dispatches,
            provider_stats.metal_runtime_q4_0_mm_sg_unrolled_dispatches,
        },
    );
    std.debug.print(
        "metal_q4_0_pair_activation_policy: mmv_nr4_nsg2={d} mmv_nr8_nsg2={d} mmv_nr4_nsg4={d} mmv_nr8_nsg4={d} mmv_variant_fallbacks={d} mm_m32_n64_aligned={d} mm_m32_n64_tail={d} mm_m32_n32_aligned={d} mm_m32_n32_tail={d} mm_variant_fallbacks={d}\n",
        .{
            provider_stats.metal_runtime_q4_0_pair_activation_mmv_nr4_nsg2_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mmv_nr8_nsg2_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mmv_nr4_nsg4_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mmv_nr8_nsg4_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mmv_variant_fallbacks,
            provider_stats.metal_runtime_q4_0_pair_activation_mm_m32_n64_aligned_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mm_m32_n64_tail_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mm_m32_n32_aligned_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mm_m32_n32_tail_dispatches,
            provider_stats.metal_runtime_q4_0_pair_activation_mm_variant_fallbacks,
        },
    );
    std.debug.print(
        "metal_q4_q6_k_dispatch: q4_linear_reduce={d} q4_linear_reduce_rows={d}/{d}/{d}/{d} q4_pair_reduce={d} q4_pair_act_reduce={d} q4_pair_act_reduce_out_f16={d} q4_activation_rhs_reduce={d} q6_linear_reduce={d} q6_linear_reduce_rows={d}/{d}/{d}/{d} q6_linear_reduce_in_f16={d} q6_high_row_mm_matrix={d}\n",
        .{
            provider_stats.metal_runtime_q4_k_linear_reduce,
            provider_stats.metal_runtime_q4_k_linear_reduce_rows_1,
            provider_stats.metal_runtime_q4_k_linear_reduce_rows_2_8,
            provider_stats.metal_runtime_q4_k_linear_reduce_rows_9_64,
            provider_stats.metal_runtime_q4_k_linear_reduce_rows_65_plus,
            provider_stats.metal_runtime_q4_k_pair_reduce,
            provider_stats.metal_runtime_q4_k_pair_activation_reduce,
            provider_stats.metal_runtime_q4_k_pair_activation_reduce_f16_output,
            provider_stats.metal_runtime_q4_k_activation_rhs_reduce,
            provider_stats.metal_runtime_q6_k_linear_reduce,
            provider_stats.metal_runtime_q6_k_linear_reduce_rows_1,
            provider_stats.metal_runtime_q6_k_linear_reduce_rows_2_8,
            provider_stats.metal_runtime_q6_k_linear_reduce_rows_9_64,
            provider_stats.metal_runtime_q6_k_linear_reduce_rows_65_plus,
            provider_stats.metal_runtime_q6_k_linear_reduce_f16_input,
            provider_stats.metal_runtime_q6_k_high_row_mm_matrix_dispatches,
        },
    );
    // Counter names and their order come from the shared table so this line
    // stays byte-identical to the historical hand-written format string.
    std.debug.print("metal_generated_quant_dispatch:", .{});
    for (quant_matmul.generated_quant_counter_names) |counter| {
        std.debug.print(" {s}={d}", .{
            counter.name,
            quant_matmul.generatedQuantDispatchCount(&provider_stats.metal_runtime_antfly_generated_dispatch_counts, counter.format, counter.epilogue),
        });
    }
    std.debug.print("\n", .{});
    std.debug.print(
        "metal_q4_0_ple_dispatch: activation_rhs_reduce_out_f16={d} linear_reduce_in_f16={d}\n",
        .{
            provider_stats.metal_runtime_q4_0_ple_activation_rhs_reduce_f16_output,
            provider_stats.metal_runtime_q4_0_ple_linear_reduce_f16_input,
        },
    );
    const q8_family_dispatch = provider_stats.metal_runtime_q8_0_linear_family_dispatch_counts;
    std.debug.print(
        "metal_q8_0_dispatch_families: none={d}/{d}/{d}/{d} pair_act={d}/{d}/{d}/{d} pair_act_rms={d}/{d}/{d}/{d} act_rhs={d}/{d}/{d}/{d} pair={d}/{d}/{d}/{d} qkv={d}/{d}/{d}/{d}\n",
        .{
            q8_family_dispatch[0][0],
            q8_family_dispatch[0][1],
            q8_family_dispatch[0][2],
            q8_family_dispatch[0][3],
            q8_family_dispatch[1][0],
            q8_family_dispatch[1][1],
            q8_family_dispatch[1][2],
            q8_family_dispatch[1][3],
            q8_family_dispatch[2][0],
            q8_family_dispatch[2][1],
            q8_family_dispatch[2][2],
            q8_family_dispatch[2][3],
            q8_family_dispatch[3][0],
            q8_family_dispatch[3][1],
            q8_family_dispatch[3][2],
            q8_family_dispatch[3][3],
            q8_family_dispatch[4][0],
            q8_family_dispatch[4][1],
            q8_family_dispatch[4][2],
            q8_family_dispatch[4][3],
            q8_family_dispatch[5][0],
            q8_family_dispatch[5][1],
            q8_family_dispatch[5][2],
            q8_family_dispatch[5][3],
        },
    );
    std.debug.print(
        "metal_q8_0_dispatch_families_ext: rms_scale={d}/{d}/{d}/{d} attention_out={d}/{d}/{d}/{d} ffn_down={d}/{d}/{d}/{d} ple_projection={d}/{d}/{d}/{d} tail={d}/{d}/{d}/{d} ple_gate={d}/{d}/{d}/{d}\n",
        .{
            q8_family_dispatch[6][0],
            q8_family_dispatch[6][1],
            q8_family_dispatch[6][2],
            q8_family_dispatch[6][3],
            q8_family_dispatch[7][0],
            q8_family_dispatch[7][1],
            q8_family_dispatch[7][2],
            q8_family_dispatch[7][3],
            q8_family_dispatch[8][0],
            q8_family_dispatch[8][1],
            q8_family_dispatch[8][2],
            q8_family_dispatch[8][3],
            q8_family_dispatch[9][0],
            q8_family_dispatch[9][1],
            q8_family_dispatch[9][2],
            q8_family_dispatch[9][3],
            q8_family_dispatch[10][0],
            q8_family_dispatch[10][1],
            q8_family_dispatch[10][2],
            q8_family_dispatch[10][3],
            q8_family_dispatch[11][0],
            q8_family_dispatch[11][1],
            q8_family_dispatch[11][2],
            q8_family_dispatch[11][3],
        },
    );
    const gpt_stats = gpt_arch.getDebugTimingStats();
    std.debug.print(
        "gpt_timing_ms: attention={d} attn_norm={d} attn_qkv={d} attn_core={d} attn_rope={d} attn_gqa={d} attn_out_proj={d} ffn={d} moe_router_weight_fetch={d} moe_router_proj={d} moe_route_select={d} moe_router_download={d} moe_expert_scale_download={d} moe_expert_weight_fetch={d} moe_input_download={d} moe_prepare_layer={d} moe_append_route={d} moe_finalize_layer={d} moe_prefetch_hint={d}\n",
        .{
            @divTrunc(gpt_stats.attention_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_norm_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_qkv_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_core_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_rope_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_gqa_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.attention_out_proj_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.ffn_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_router_weight_fetch_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_router_proj_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_route_select_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_router_download_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_expert_scale_download_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_expert_weight_fetch_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_input_download_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_prepare_layer_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_append_route_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_finalize_layer_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_prefetch_hint_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "gpt_moe_timing_ms: grouped_attempts={d} grouped_successes={d} moe_grouped={d} moe_fallback={d} moe_grouped_input_copy={d} moe_grouped_input_upload={d} moe_grouped_ops={d} moe_grouped_sync_w1={d} moe_grouped_sync_w3={d} moe_grouped_sync_gate={d} moe_grouped_sync_w2={d} moe_grouped_sync_ops={d} moe_grouped_output_download={d} moe_grouped_scatter={d} moe_grouped_sync_scatter={d} moe_grouped_cleanup={d}\n",
        .{
            gpt_stats.moe_grouped_attempts,
            gpt_stats.moe_grouped_successes,
            @divTrunc(gpt_stats.moe_grouped_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_fallback_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_input_copy_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_input_upload_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_ops_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_w1_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_w3_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_gate_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_w2_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_ops_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_output_download_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_scatter_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_sync_scatter_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.moe_grouped_cleanup_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "gpt_overhead_ms: eval={d} eval_count={d} shared_expert_ffn={d} norm={d}\n",
        .{
            @divTrunc(gpt_stats.eval_nanos, std.time.ns_per_ms),
            gpt_stats.eval_count,
            @divTrunc(gpt_stats.shared_expert_ffn_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.norm_nanos, std.time.ns_per_ms),
        },
    );
    std.debug.print(
        "gpt_block_counts: dense_attempts={d} dense_successes={d} gated_attempts={d} gated_successes={d} gated_input_attempts={d} gated_input_successes={d} gated_input_prefill={d} gated_input_decode={d} gated_qkv_attempts={d} gated_qkv_successes={d}\n",
        .{
            gpt_stats.dense_block_attempts,
            gpt_stats.dense_block_successes,
            gpt_stats.gated_block_attempts,
            gpt_stats.gated_block_successes,
            gpt_stats.gated_block_input_attempts,
            gpt_stats.gated_block_input_successes,
            @divTrunc(gpt_stats.gated_block_input_prefill_nanos, std.time.ns_per_ms),
            @divTrunc(gpt_stats.gated_block_input_decode_nanos, std.time.ns_per_ms),
            gpt_stats.gated_block_qkv_attempts,
            gpt_stats.gated_block_qkv_successes,
        },
    );
}

pub fn printRuntimeDebugTiming(runtime_opt: ?*const model_runtime.ModelRuntime) void {
    const metal_stats = if (runtime_opt) |runtime_model|
        runtime_model.debugTimingStats()
    else
        model_runtime.RuntimeDebugTimingStats{};
    printRuntimeDebugTimingStats(metal_stats, runtime_opt != null);
}

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

/// Metal-backed whole-model executor.
///
/// The first slice keeps prefill on the existing Metal runtime but routes
/// qLen=1 supported decoder-family decode through the raw whole-token
/// Metal-owned input/override path that was previously only reachable from
/// `generation.zig`.
fn disablePagedKvDebug() bool {
    const value = c_std.getenv("TERMITE_DISABLE_PAGED_KV") orelse return false;
    const slice = std.mem.span(value);
    return std.mem.eql(u8, slice, "1") or
        std.ascii.eqlIgnoreCase(slice, "true") or
        std.ascii.eqlIgnoreCase(slice, "yes") or
        std.ascii.eqlIgnoreCase(slice, "on");
}

const ExecutorContext = struct {
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType = null,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
    model_identity: ?runtime.kv.attention_match_checkpoint.Identity,
};

const DecoderRuntimeSpanState = struct {
    kv_tokens: usize,
    kv_position_offset: usize,
};

fn executorUsesLazyDeviceKvReserve(config: gpt_mod.Config) bool {
    if (config.family != .gemma or !config.hasPle()) return false;
    const mixed_width = config.hasGlobalAttentionLayers() and
        (config.maxKvHeads() != config.effectiveKVHeads() or
            config.maxHeadDim() != config.headDim());
    return mixed_width or config.num_kv_shared_layers > 0;
}

fn executorContractKvCacheView(kv_view: generation.KvView) contracts.KvCacheView {
    return .{
        .sequence_id = kv_view.sequence_id,
        .pool_id = kv_view.pool_id,
        .logical_block_count = kv_view.logical_block_count,
        .tail_tokens = kv_view.tail_tokens,
        .position_offset = kv_view.position_offset,
        .max_inflight_tokens = kv_view.max_inflight_tokens,
        .allow_swa_ring = kv_view.allow_swa_ring,
        .logical_blocks = kv_view.logical_blocks,
        .kv_storage = kv_view.kv_storage,
    };
}

fn executorDecodeKvCacheView(kv_view: generation.KvView) gpt_arch.DecodeContext.KvCacheView {
    return .{
        .sequence_id = kv_view.sequence_id,
        .pool_id = kv_view.pool_id,
        .logical_block_count = kv_view.logical_block_count,
        .tail_tokens = kv_view.tail_tokens,
        .position_offset = kv_view.position_offset,
        .max_inflight_tokens = kv_view.max_inflight_tokens,
        .allow_swa_ring = kv_view.allow_swa_ring,
        .logical_blocks = kv_view.logical_blocks,
        .kv_storage = kv_view.kv_storage,
    };
}

const ExecutorKvMetadataLayer = struct {
    allocator: std.mem.Allocator,
    pool_id: runtime.kv.block.KvPoolId,
    lazy_device_reserve: bool,
    sequence_id: ?runtime.kv.manager.SequenceId = null,
    total_token_count: usize = 0,
    position_offset: usize = 0,
    tail_tokens: u16 = 0,
    kv_view: ?generation.KvView = null,
    compacted: bool = false,
    max_inflight_tokens: usize = 0,
    allow_swa_ring: bool = false,
    logical_blocks: std.ArrayListUnmanaged(runtime.kv.block.KvBlockId) = .empty,

    fn init(allocator: std.mem.Allocator, pool_id: runtime.kv.block.KvPoolId, lazy_device_reserve: bool) ExecutorKvMetadataLayer {
        return .{ .allocator = allocator, .pool_id = pool_id, .lazy_device_reserve = lazy_device_reserve };
    }

    pub fn deinit(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) void {
        self.releaseOwnedSequence(storage) catch {};
        self.logical_blocks.deinit(self.allocator);
        self.* = undefined;
    }

    fn reset(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
        try self.releaseOwnedSequence(storage);
        self.total_token_count = 0;
        self.position_offset = 0;
        self.tail_tokens = 0;
        self.kv_view = null;
        self.compacted = false;
        self.max_inflight_tokens = 0;
        self.allow_swa_ring = false;
        self.logical_blocks.clearRetainingCapacity();
    }

    fn configureKvPolicy(self: *ExecutorKvMetadataLayer, max_inflight_tokens: usize, allow_swa_ring: bool) !void {
        const enabled = allow_swa_ring and max_inflight_tokens > 0;
        const normalized_max = if (enabled) max_inflight_tokens else 0;
        if (self.sequence_id != null or self.total_token_count != 0) {
            if (self.max_inflight_tokens != normalized_max or self.allow_swa_ring != enabled) {
                return error.InvalidPagedKvState;
            }
            return;
        }
        self.max_inflight_tokens = normalized_max;
        self.allow_swa_ring = enabled;
    }

    fn tokenCount(self: *const ExecutorKvMetadataLayer, storage: *const runtime.kv.storage_runtime.KvStorageRuntime) usize {
        if (self.logical_blocks.items.len == 0) return 0;
        const pool = storage.getPool(self.pool_id) orelse return 0;
        const page_size = pool.config.page_size_tokens;
        return (self.logical_blocks.items.len - 1) * page_size + self.tail_tokens;
    }

    fn updateView(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) void {
        const sequence_id = self.sequence_id orelse {
            self.kv_view = null;
            return;
        };
        const retained_tokens = self.tokenCount(storage);
        if (retained_tokens == 0) {
            self.kv_view = null;
            return;
        }
        self.kv_view = .{
            .sequence_id = sequence_id,
            .pool_id = self.pool_id,
            .logical_block_count = self.logical_blocks.items.len,
            .tail_tokens = self.tail_tokens,
            .token_count = retained_tokens,
            .position_offset = self.position_offset,
            .max_inflight_tokens = self.max_inflight_tokens,
            .allow_swa_ring = self.allow_swa_ring,
            .logical_blocks = self.logical_blocks.items,
            .kv_storage = storage,
        };
    }

    fn syncSequenceState(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
        const sequence_id = self.sequence_id orelse return;
        const seq_state = try storage.sequenceMut(sequence_id);
        seq_state.compacted = self.compacted;
        seq_state.block_table.shared_prefix_blocks = 0;
        try seq_state.block_table.blocks.resize(self.allocator, self.logical_blocks.items.len);
        @memcpy(seq_state.block_table.blocks.items, self.logical_blocks.items);
        seq_state.block_table.tail_tokens = self.tail_tokens;
        self.updateView(storage);
        try self.reserveDeviceCapacity(storage);
    }

    fn reserveDeviceCapacity(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
        // The generic eager reserve does not know whether a layer is local or
        // global. Let the first layer write establish that slot's ring geometry
        // when split SWA/global caching is enabled.
        if (self.lazy_device_reserve or self.allow_swa_ring or disableDeviceKvReserveRequested()) return;
        const sequence_id = self.sequence_id orelse return;
        const pool = storage.getPool(self.pool_id) orelse return error.InvalidPoolId;
        const retained_tokens = self.tokenCount(storage);
        if (retained_tokens == 0) return;
        for (0..pool.config.num_layers_packed) |layer_index| {
            storage.reserveLayerKvDeviceCapacity(
                sequence_id,
                layer_index,
                retained_tokens,
                self.position_offset,
                pool.config.num_kv_heads,
                pool.config.head_dim,
                0,
                0,
                false,
            ) catch |err| switch (err) {
                error.DeviceWriteUnsupported,
                error.DeviceWriteFormatUnsupported,
                error.DeviceWriteFallback,
                => return,
                else => return err,
            };
        }
    }

    fn ensureAttached(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
        if (self.sequence_id != null) return;
        self.sequence_id = try storage.attachSequence(self.pool_id);
        try self.syncSequenceState(storage);
    }

    fn releaseOwnedSequence(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !void {
        const sequence_id = self.sequence_id orelse return;
        try storage.releaseSequence(sequence_id);
        self.sequence_id = null;
        self.total_token_count = 0;
        self.position_offset = 0;
        self.tail_tokens = 0;
        self.compacted = false;
        self.max_inflight_tokens = 0;
        self.allow_swa_ring = false;
        self.logical_blocks.clearRetainingCapacity();
        self.kv_view = null;
    }

    fn reserveTailBlock(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !runtime.kv.block.KvBlockId {
        const pool = storage.getPoolMut(self.pool_id) orelse return error.InvalidPoolId;
        const page_size = pool.config.page_size_tokens;
        if (self.logical_blocks.items.len > 0 and self.tail_tokens < page_size) {
            return self.logical_blocks.items[self.logical_blocks.items.len - 1];
        }
        const id = try pool.acquire(self.allocator);
        try self.logical_blocks.append(self.allocator, id);
        self.tail_tokens = 0;
        return id;
    }

    fn appendTokensNoTrim(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime, count: usize) !void {
        try self.ensureAttached(storage);
        const pool = storage.getPoolMut(self.pool_id) orelse return error.InvalidPoolId;
        const page_size = pool.config.page_size_tokens;
        var remaining: usize = count;
        while (remaining > 0) {
            _ = try self.reserveTailBlock(storage);
            const space = page_size - self.tail_tokens;
            const consumed = @min(space, remaining);
            self.tail_tokens += @intCast(consumed);
            remaining -= consumed;
        }
    }

    fn fullBlockCount(self: *const ExecutorKvMetadataLayer, storage: *const runtime.kv.storage_runtime.KvStorageRuntime) usize {
        if (self.logical_blocks.items.len == 0) return 0;
        const pool = storage.getPool(self.pool_id) orelse return 0;
        const page_size = pool.config.page_size_tokens;
        if (self.tail_tokens == page_size) return self.logical_blocks.items.len;
        return self.logical_blocks.items.len - 1;
    }

    fn trimToSlidingWindow(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime) !usize {
        if (self.compacted) return 0;
        const pool = storage.getPoolMut(self.pool_id) orelse return error.InvalidPoolId;
        const keep_tokens = pool.config.sliding_window_size orelse return 0;
        const current_tokens = self.tokenCount(storage);
        if (keep_tokens >= current_tokens) return 0;
        const page_size = pool.config.page_size_tokens;
        const excess_tokens = current_tokens - keep_tokens;
        const droppable_blocks = @min(excess_tokens / page_size, self.fullBlockCount(storage));
        if (droppable_blocks == 0) return 0;
        for (self.logical_blocks.items[0..droppable_blocks]) |block_id| {
            _ = try pool.releaseRef(self.allocator, block_id);
        }
        std.mem.copyForwards(runtime.kv.block.KvBlockId, self.logical_blocks.items[0 .. self.logical_blocks.items.len - droppable_blocks], self.logical_blocks.items[droppable_blocks..]);
        self.logical_blocks.items.len -= droppable_blocks;
        return droppable_blocks * page_size;
    }

    fn notePrefill(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime, token_count: usize) !generation.KvMutationResult {
        const before_view = self.kv_view;
        const before_sequence_id = self.sequence_id;
        self.total_token_count = token_count;
        self.position_offset = 0;
        self.compacted = false;
        try self.appendTokensNoTrim(storage, token_count);
        try self.syncSequenceState(storage);
        return .{
            .token_count = self.total_token_count,
            .kv_view = self.kv_view,
            .compacted = self.compacted,
            .delta = .{
                .sequence_replaced = before_sequence_id != self.sequence_id,
                .logical_block_count_before = if (before_view) |view| view.logical_block_count else 0,
                .logical_block_count_after = if (self.kv_view) |view| view.logical_block_count else 0,
                .position_offset_before = if (before_view) |view| view.position_offset else 0,
                .position_offset_after = self.position_offset,
            },
        };
    }

    fn appendPrefillChunk(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime, token_count: usize) !generation.KvMutationResult {
        const before_view = self.kv_view;
        const before_sequence_id = self.sequence_id;
        self.total_token_count += token_count;
        self.position_offset = 0;
        self.compacted = false;
        try self.appendTokensNoTrim(storage, token_count);
        try self.syncSequenceState(storage);
        return .{
            .token_count = self.total_token_count,
            .kv_view = self.kv_view,
            .compacted = self.compacted,
            .delta = .{
                .sequence_replaced = before_sequence_id != self.sequence_id,
                .logical_block_count_before = if (before_view) |view| view.logical_block_count else 0,
                .logical_block_count_after = if (self.kv_view) |view| view.logical_block_count else 0,
                .position_offset_before = if (before_view) |view| view.position_offset else 0,
                .position_offset_after = self.position_offset,
            },
        };
    }

    fn appendGeneratedTokens(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime, count: usize) !generation.KvMutationResult {
        const before_view = self.kv_view;
        const before_sequence_id = self.sequence_id;
        self.total_token_count += count;
        try self.appendTokensNoTrim(storage, count);
        const retained_tokens_before_trim = self.tokenCount(storage);
        _ = try self.trimToSlidingWindow(storage);
        const retained_tokens = self.tokenCount(storage);
        self.position_offset = self.total_token_count - retained_tokens;
        try self.syncSequenceState(storage);
        return .{
            .token_count = self.total_token_count,
            .kv_view = self.kv_view,
            .compacted = self.compacted,
            .delta = .{
                .sequence_replaced = before_sequence_id != self.sequence_id,
                .logical_block_count_before = if (before_view) |view| view.logical_block_count else 0,
                .logical_block_count_after = if (self.kv_view) |view| view.logical_block_count else 0,
                .position_offset_before = if (before_view) |view| view.position_offset else 0,
                .position_offset_after = self.position_offset,
                .retained_tokens = retained_tokens_before_trim,
            },
        };
    }

    fn truncateTokens(self: *ExecutorKvMetadataLayer, storage: *runtime.kv.storage_runtime.KvStorageRuntime, count: usize) !generation.KvMutationResult {
        const before_view = self.kv_view;
        const before_sequence_id = self.sequence_id;
        if (count > self.total_token_count) return error.TruncateBeyondStart;
        self.total_token_count -= count;
        const pool = storage.getPoolMut(self.pool_id) orelse return error.InvalidPoolId;
        const page_size = pool.config.page_size_tokens;
        const current = self.tokenCount(storage);
        if (count >= current) {
            const dropped = try self.allocator.dupe(runtime.kv.block.KvBlockId, self.logical_blocks.items);
            defer self.allocator.free(dropped);
            self.logical_blocks.clearRetainingCapacity();
            self.tail_tokens = 0;
            for (dropped) |block_id| _ = try pool.releaseRef(self.allocator, block_id);
        } else if (count > 0) {
            const old_len = self.logical_blocks.items.len;
            const target = current - count;
            const needed_blocks = (target + page_size - 1) / page_size;
            const excess_blocks = if (old_len > needed_blocks) old_len - needed_blocks else 0;
            if (excess_blocks > 0) {
                const dropped = try self.allocator.dupe(runtime.kv.block.KvBlockId, self.logical_blocks.items[old_len - excess_blocks .. old_len]);
                defer self.allocator.free(dropped);
                self.logical_blocks.items.len -= excess_blocks;
                for (dropped) |block_id| _ = try pool.releaseRef(self.allocator, block_id);
            }
            const rem: u16 = @intCast(target % page_size);
            self.tail_tokens = if (rem == 0) page_size else rem;
        }
        const retained_tokens = self.tokenCount(storage);
        self.position_offset = self.total_token_count - retained_tokens;
        try self.syncSequenceState(storage);
        return .{
            .token_count = self.total_token_count,
            .kv_view = self.kv_view,
            .compacted = self.compacted,
            .delta = .{
                .sequence_replaced = before_sequence_id != self.sequence_id,
                .logical_block_count_before = if (before_view) |view| view.logical_block_count else 0,
                .logical_block_count_after = if (self.kv_view) |view| view.logical_block_count else 0,
                .position_offset_before = if (before_view) |view| view.position_offset else 0,
                .position_offset_after = self.position_offset,
            },
        };
    }
};

const PipelinedDecodeMode = enum {
    greedy,
    sampled_nucleus,
};

const PipelinedSuppressKind = enum {
    model,
    request,
};

fn pipelinedSuppressPolicyMatches(
    installed_kind: ?PipelinedSuppressKind,
    installed_ids: []const i32,
    desired_kind: PipelinedSuppressKind,
    desired_ids: []const i32,
) bool {
    return installed_kind == desired_kind and std.mem.eql(i32, installed_ids, desired_ids);
}

test "pipelined suppression policy compares kind length and contents not address" {
    var ids = [_]i32{ 1, 7, 11, 50, 106 };
    const owned = try std.testing.allocator.dupe(i32, &ids);
    defer std.testing.allocator.free(owned);
    try std.testing.expect(pipelinedSuppressPolicyMatches(.request, owned, .request, &ids));
    ids[1] = 9;
    try std.testing.expect(!pipelinedSuppressPolicyMatches(.request, owned, .request, &ids));
    try std.testing.expect(!pipelinedSuppressPolicyMatches(.request, owned, .model, owned));
    try std.testing.expect(!pipelinedSuppressPolicyMatches(.request, owned, .request, owned[0..4]));
}

fn validateSampledPipelineContinuation(
    armed: bool,
    active_mode: ?PipelinedDecodeMode,
    next_position: usize,
    requested_position: usize,
) !void {
    if (!armed) return;
    if (active_mode != .sampled_nucleus) return error.PipelinedDecodeModeChanged;
    if (next_position != requested_position) return error.PipelinedDecodePositionChanged;
}

fn sampledPipelineConfigMatches(
    active: ?model_runtime.SamplingConfig,
    requested: model_runtime.SamplingConfig,
) bool {
    return active != null and std.meta.eql(active.?, requested);
}

fn submitSampledPipelineToken(cb: *const ops.ComputeBackend, token: i64, vocab_size: usize) !void {
    // Frame A has completed, but B still depends on its device token. An
    // invalid draw (including the nucleus UINT32_MAX sentinel) must cancel B
    // rather than submit it with an invalid embedding index.
    errdefer decoder_gated_runtime.pipelinedCancelPending(cb);
    if (token < 0 or @as(u64, @intCast(token)) >= vocab_size) return error.InvalidModelOutput;
    if ((try decoder_gated_runtime.decoderRuntimePipelinedControl(cb, .submit_pending)) == null) {
        return error.ResidentSamplingFrameFailed;
    }
}

const E2bMirrorAdmissionHandoff = struct {
    pending: ?ops.DecoderRuntimeE2bF16MirrorAdmission = null,
    active: ?ops.DecoderRuntimeE2bF16MirrorAdmission = null,
    active_sequence_end: ?usize = null,

    fn stage(self: *@This(), value: ?ops.DecoderRuntimeE2bF16MirrorAdmission) void {
        self.pending = value;
        self.active = null;
        self.active_sequence_end = null;
    }

    fn forPrefill(self: *@This(), prior_tokens: usize, seq_len: usize, query_seq_len: usize) ?ops.DecoderRuntimeE2bF16MirrorAdmission {
        if (query_seq_len == 0 or seq_len < query_seq_len or seq_len - query_seq_len != prior_tokens) {
            self.clear();
            return null;
        }
        if (prior_tokens == 0) {
            self.active = self.pending;
            self.pending = null;
            self.active_sequence_end = if (self.active != null) seq_len else null;
            return self.active;
        }
        // Append chunks may reuse only the active contract that authorized the
        // exact prior prefix. A newly pending contract is never consumed as an
        // append: it belongs to a future fresh sequence.
        if (self.pending != null or self.active == null or self.active_sequence_end != prior_tokens) {
            self.clear();
            return null;
        }
        self.active_sequence_end = seq_len;
        return self.active;
    }

    fn clear(self: *@This()) void {
        self.pending = null;
        self.active = null;
        self.active_sequence_end = null;
    }
};

test "E2B mirror admission handoff consumes once reuses chunks and clears" {
    const first: ops.DecoderRuntimeE2bF16MirrorAdmission = .{
        .cohort = .e2b_full35_gate_up_down,
        .model_layer_count = 35,
        .backend_cap_bytes = 7 * 1024 * 1024 * 1024,
        .nonlinear_weights_complete = true,
        .adapters_active = false,
        .declared_backend_kv_bytes = 1,
        .declared_backend_scratch_bytes = 1,
        .max_context_tokens = 2048,
        .max_prefill_rows = 512,
        .layer_count = 35,
    };
    var handoff: E2bMirrorAdmissionHandoff = .{};
    handoff.stage(first);
    try std.testing.expect(handoff.pending != null and handoff.active == null);
    try std.testing.expectEqual(first.backend_cap_bytes, handoff.forPrefill(0, 512, 512).?.backend_cap_bytes);
    try std.testing.expect(handoff.pending == null and handoff.active != null);
    try std.testing.expectEqual(first.backend_cap_bytes, handoff.forPrefill(512, 1024, 512).?.backend_cap_bytes);
    // Wrong prior prefix cannot reuse request authority.
    try std.testing.expect(handoff.forPrefill(511, 1023, 512) == null);
    // A fresh sequence without a new prepare cannot inherit prior active state.
    handoff.active = first;
    handoff.active_sequence_end = 1024;
    try std.testing.expect(handoff.forPrefill(0, 512, 512) == null);
    var replacement = first;
    replacement.backend_cap_bytes -= 1;
    handoff.stage(replacement);
    try std.testing.expectEqual(replacement.backend_cap_bytes, handoff.forPrefill(0, 512, 512).?.backend_cap_bytes);
    handoff.stage(null);
    try std.testing.expect(handoff.forPrefill(0, 512, 512) == null);
    handoff.stage(first);
    handoff.clear();
    try std.testing.expect(handoff.forPrefill(0, 512, 512) == null);
}

const RuntimeContext = struct {
    allocator: std.mem.Allocator,
    cb: ops.ComputeBackend,
    gpt_config: gpt_mod.Config,
    model_identity: ?runtime.kv.attention_match_checkpoint.Identity,
    use_decoder_runtime_executor: bool,
    kv_storage: runtime.kv.storage_runtime.KvStorageRuntime,
    pool_id: runtime.kv.block.KvPoolId,
    moe_runtime: runtime.moe.runtime.MoeRuntime,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
    kv_metadata: ExecutorKvMetadataLayer,
    mirrored_token_count: usize,
    mirrored_kv_view: ?generation.KvView,
    mirrored_kv_compacted: bool,
    pipelined_decode_armed: bool = false,
    pipelined_decode_mode: ?PipelinedDecodeMode = null,
    pipelined_sampling: ?model_runtime.SamplingConfig = null,
    pipelined_next_position: usize = 0,
    pipelined_kv_advanced_for: ?usize = null,
    pipelined_suppress_kind: ?PipelinedSuppressKind = null,
    pipelined_suppress_ids: ?[]i32 = null,
    request_disables_pipelined_decode: bool = false,
    raw_span_state: ?DecoderRuntimeSpanState,
    greedy_device_token_tensor: ?ops.CT,
    greedy_device_token_id: ?i64,
    handoff_tokens: std.ArrayListUnmanaged(i64) = .empty,
    handoff_revision: u64 = 1,
    e2b_mirror_admission_handoff: E2bMirrorAdmissionHandoff = .{},
    /// Request tracker installed into mutable KV storage.  The cached model
    /// executor never sees or retains this pointer.
    bound_context_owner: ?model_runtime.ContextOwnerBinding = null,

    fn init(
        allocator: std.mem.Allocator,
        session: backends.Session,
        gpt_config: gpt_mod.Config,
        kv_dtype_override: ?runtime.kv.pool.KvDType,
        shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
        model_identity: ?runtime.kv.attention_match_checkpoint.Identity,
    ) !*RuntimeContext {
        // The runtime outlives the request that creates it and is cached on
        // the loaded model, so it must not hold the per-request Metal lease.
        const cb = try session_factory.getComputeBackendBorrowingSharedProvider(session, allocator);
        errdefer {
            var cb_mut = cb;
            cb_mut.deinit();
        }

        const session_backend = session.backend();
        const kv_dtype = kv_dtype_override orelse session_factory.recommendedKvDTypeForSession(session, .metal);
        const needs_cpu_kv_bridge = switch (session_backend) {
            .metal => false,
            else => false,
        };
        const sliding_window_size = executorKvSlidingWindowSize(gpt_config);

        var kv_storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(allocator, .{
            .backend = .metal,
            .dtype = kv_dtype,
            .page_size_tokens = 16,
            .num_layers_packed = @intCast(gpt_config.num_hidden_layers),
            .num_kv_heads = gpt_config.maxKvHeads(),
            .head_dim = gpt_config.maxHeadDim(),
            .sliding_window_size = sliding_window_size,
            .store_cpu_bytes = needs_cpu_kv_bridge,
        });
        errdefer kv_storage.deinit();
        // Phase 6 wiring: ask the Metal compute backend to install its device
        // write hook if the KV dtype matches the fast path. When unsupported
        // (e.g. f32/int8 KVs, or no decoder runtime), this is a no-op and
        // writes continue through the host path.
        cb.provisionKvDeviceWriteHook(&kv_storage) catch {};

        const pool_id = kv_storage.poolId();

        const ctx = try allocator.create(RuntimeContext);
        ctx.* = .{
            .allocator = allocator,
            .cb = cb,
            .gpt_config = gpt_config,
            .model_identity = model_identity,
            .use_decoder_runtime_executor = session.backend() == .metal,
            .kv_storage = kv_storage,
            .pool_id = pool_id,
            .moe_runtime = runtime.moe.runtime.MoeRuntime.init(allocator, shared_moe_cache),
            .shared_moe_cache = shared_moe_cache,
            .kv_metadata = ExecutorKvMetadataLayer.init(allocator, pool_id, executorUsesLazyDeviceKvReserve(gpt_config)),
            .mirrored_token_count = 0,
            .mirrored_kv_view = null,
            .mirrored_kv_compacted = false,
            .raw_span_state = null,
            .greedy_device_token_tensor = null,
            .greedy_device_token_id = null,
            .bound_context_owner = null,
        };
        return ctx;
    }

    fn resetState(self: *RuntimeContext) !void {
        try self.invalidateDecoderRuntimeSpanState();
        self.moe_runtime.deinit();
        self.moe_runtime = runtime.moe.runtime.MoeRuntime.init(self.allocator, self.shared_moe_cache);
        try self.kv_metadata.reset(&self.kv_storage);
        self.mirrored_token_count = 0;
        self.mirrored_kv_view = null;
        self.mirrored_kv_compacted = false;
        self.handoff_tokens.clearRetainingCapacity();
        self.handoff_revision +%= 1;
        if (self.handoff_revision == 0) self.handoff_revision = 1;
    }

    pub fn deinit(self: *RuntimeContext) void {
        // Sampled and greedy pipelining intentionally leave the next frame
        // submitted after returning a token. The runtime borrows the shared
        // Metal provider, so its ComputeBackend teardown does not own or drain
        // that frame. Complete it before releasing KV storage and ending the
        // borrowed request that the in-flight command buffer depends on.
        self.clearPipelinedRequestState();
        self.e2b_mirror_admission_handoff.clear();
        self.cleanupBoundedPleRequestAfterError() catch |err| {
            std.log.err("required bounded PLE request cleanup failed during runtime teardown: {s}", .{@errorName(err)});
            @panic("unresolved bounded PLE request cleanup during runtime teardown");
        };
        if (self.bound_context_owner) |binding| {
            const tracker = binding.owner.tracker;
            // Release every sequence/slot before detaching the request tracker.
            // A cached runtime must never outlive the owner whose ledger it uses.
            self.resetState() catch |err| {
                std.log.err("compiled runtime reset before owner detach failed: {s}", .{@errorName(err)});
                @panic("compiled runtime retained request-owned storage during teardown");
            };
            self.kv_storage.unbindRequestOwnerTracker(tracker) catch |err| {
                std.log.err("compiled runtime owner detach failed: {s}", .{@errorName(err)});
                @panic("compiled runtime retained request owner during teardown");
            };
            binding.owner.detachCompiledStorage(binding.witness, tracker) catch |err| {
                std.log.err("compiled runtime owner detach receipt failed: {s}", .{@errorName(err)});
                @panic("compiled runtime owner detach was not acknowledged");
            };
            self.bound_context_owner = null;
        }
        self.clearGreedyDeviceToken();
        self.handoff_tokens.deinit(self.allocator);
        self.moe_runtime.deinit();
        self.kv_metadata.deinit(&self.kv_storage);
        self.kv_storage.deinit();
        self.cb.deinit();
        self.allocator.destroy(self);
    }

    fn invalidateDecoderRuntimeSpanState(self: *RuntimeContext) !void {
        self.clearGreedyDeviceToken();
        try self.cb.decoderRuntimeResetState();
        self.raw_span_state = null;
    }

    fn greedyDeviceTokenHandoffEnabled(self: *const RuntimeContext) bool {
        return self.cb.kind() == .cuda and envFlag("ANTFLY_INFERENCE_CUDA_GREEDY_DEVICE_TOKEN_HANDOFF");
    }

    fn clearGreedyDeviceToken(self: *RuntimeContext) void {
        if (self.greedy_device_token_tensor) |token| {
            self.cb.free(token);
        }
        self.greedy_device_token_tensor = null;
        self.greedy_device_token_id = null;
    }

    fn clearE2bMirrorAdmissionAfterError(self: *RuntimeContext) void {
        self.e2b_mirror_admission_handoff.clear();
        self.cleanupBoundedPleRequestAfterError() catch |err| {
            std.log.err("required bounded PLE request cleanup failed after runtime error: {s}", .{@errorName(err)});
            @panic("unresolved bounded PLE request cleanup after runtime error");
        };
    }

    fn cleanupBoundedPleRequestAfterError(self: *RuntimeContext) !void {
        try cleanupBoundedPleRequest(&self.cb);
    }

    fn storeGreedyDeviceTokenResult(self: *RuntimeContext, result: gpt_arch.GreedyDeviceTokenResult) void {
        const token_tensor = result.token_tensor;
        self.clearGreedyDeviceToken();
        if (token_tensor) |token| {
            self.greedy_device_token_tensor = token;
            self.greedy_device_token_id = @intCast(result.token_id);
        }
    }

    fn seedGreedyDeviceToken(self: *RuntimeContext, token_id: i64) !void {
        if (!self.greedyDeviceTokenHandoffEnabled()) return;
        if (token_id < 0 or token_id > std.math.maxInt(i32)) {
            self.clearGreedyDeviceToken();
            return;
        }
        const data = [_]i32{@intCast(token_id)};
        const shape = [_]i32{1};
        const token = (try self.cb.fromInt32Shape(data[0..], shape[0..])) orelse {
            self.clearGreedyDeviceToken();
            return;
        };
        self.clearGreedyDeviceToken();
        self.greedy_device_token_tensor = token;
        self.greedy_device_token_id = token_id;
        timing_stats.decode_greedy_device_token_seeds += 1;
    }

    fn forwardGreedyDeviceTokenHandoff(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        request: model_runtime.DecodeRequest,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
    ) !?i64 {
        var token_tensor = request.input_token_tensor;
        var using_stored = false;
        if (token_tensor == null and self.greedyDeviceTokenHandoffEnabled()) {
            if (self.greedy_device_token_tensor) |stored| {
                if (self.greedy_device_token_id != null and self.greedy_device_token_id.? == request.token_id) {
                    token_tensor = stored;
                    using_stored = true;
                }
            }
        }
        const input_token = token_tensor orelse return null;

        timing_stats.decode_greedy_device_token_handoff_attempts += 1;
        const maybe_result = gpt_arch.forwardGreedyLastTokenFromTokenTensor(
            &self.cb,
            allocator,
            self.gpt_config,
            input_token,
            1,
            seq_len,
            decode_context,
        ) catch |err| switch (err) {
            error.CudaGraphReplayRequired => null,
            else => return err,
        };
        const result = maybe_result orelse {
            timing_stats.decode_greedy_device_token_handoff_fallbacks += 1;
            if (using_stored) self.clearGreedyDeviceToken();
            return null;
        };
        timing_stats.decode_greedy_device_token_handoff_hits += 1;
        const token_id: i64 = @intCast(result.token_id);
        self.storeGreedyDeviceTokenResult(result);
        self.noteDecoderRuntimeStateFromCurrentView();
        return token_id;
    }

    fn applyKvMutationResult(self: *RuntimeContext, result: generation.KvMutationResult) !void {
        _ = result;
        self.mirrored_token_count = self.kv_metadata.total_token_count;
        self.mirrored_kv_view = self.kv_metadata.kv_view;
        self.mirrored_kv_compacted = self.kv_metadata.compacted;
    }

    fn decoderRuntimeConfiguredLayerCount(self: *const RuntimeContext) usize {
        return self.gpt_config.num_hidden_layers;
    }

    fn decoderRuntimeExecutorEnabled(self: *const RuntimeContext) bool {
        return self.use_decoder_runtime_executor and !envFlag("TERMITE_METAL_DISABLE_DECODER_RUNTIME_EXECUTOR");
    }

    fn notePrefill(self: *RuntimeContext, token_count: usize) !void {
        try self.invalidateDecoderRuntimeSpanState();
        const result = try self.kv_metadata.notePrefill(&self.kv_storage, token_count);
        try self.applyKvMutationResult(result);
    }

    fn appendPrefillChunk(self: *RuntimeContext, token_count: usize) !void {
        try self.invalidateDecoderRuntimeSpanState();
        const result = try self.kv_metadata.appendPrefillChunk(&self.kv_storage, token_count);
        try self.applyKvMutationResult(result);
    }

    fn appendGeneratedToken(self: *RuntimeContext) !usize {
        const result = try self.kv_metadata.appendGeneratedTokens(&self.kv_storage, 1);
        try self.applyKvMutationResult(result);
        return self.currentTokenCount();
    }

    fn appendGeneratedTokens(self: *RuntimeContext, count: usize) !usize {
        const result = try self.kv_metadata.appendGeneratedTokens(&self.kv_storage, count);
        try self.applyKvMutationResult(result);
        return self.currentTokenCount();
    }

    fn truncateGeneratedTokens(self: *RuntimeContext, count: usize) !void {
        try self.invalidateDecoderRuntimeSpanState();
        const result = try self.kv_metadata.truncateTokens(&self.kv_storage, count);
        try self.applyKvMutationResult(result);
    }

    fn compactKvCache(self: *RuntimeContext, config: runtime.kv.compaction.CompactionConfig) !void {
        try config.validate();
        if (self.currentTokenCount() > 512) return error.InvalidCompactionShape;
        try self.invalidateDecoderRuntimeSpanState();
        const sequence_id = self.kv_metadata.sequence_id orelse return;
        const kv_view = self.kv_metadata.kv_view orelse return error.InvalidPagedKvState;
        const ops_kv_view = self.kvCacheView(kv_view);
        var compacted = blk: {
            if (try self.cb.gatherPagedKvLayerCache(self.allocator, ops_kv_view, kv_view.token_count, 0)) |first_gathered| {
                const pool = self.kv_storage.getPool(self.pool_id) orelse return error.InvalidPoolId;
                const num_layers = pool.config.num_layers_packed;
                const gathered_layers = try self.allocator.alloc(runtime.kv.compaction.GatheredLayerKv, num_layers);
                defer self.allocator.free(gathered_layers);
                gathered_layers[0] = .{ .k = first_gathered.k, .v = first_gathered.v };
                var gathered_count: usize = 1;
                defer {
                    for (gathered_layers[0..gathered_count]) |layer_data| {
                        self.allocator.free(layer_data.k);
                        self.allocator.free(layer_data.v);
                    }
                }
                for (1..num_layers) |layer| {
                    const gathered = (try self.cb.gatherPagedKvLayerCache(
                        self.allocator,
                        ops_kv_view,
                        kv_view.token_count,
                        layer,
                    )) orelse return error.InvalidPagedKvState;
                    gathered_layers[layer] = .{ .k = gathered.k, .v = gathered.v };
                    gathered_count += 1;
                }
                break :blk try runtime.kv.compaction.compactGatheredSequence(
                    self.allocator,
                    gathered_layers,
                    kv_view.token_count,
                    pool.config.num_kv_heads,
                    pool.config.head_dim,
                    config,
                );
            }
            break :blk try runtime.kv.compaction.compactStorageSequence(
                self.allocator,
                &self.kv_storage,
                sequence_id,
                config,
            );
        };
        defer compacted.deinit();

        var replacement = ExecutorKvMetadataLayer.init(self.allocator, self.pool_id, self.kv_metadata.lazy_device_reserve);
        replacement.sequence_id = try self.kv_storage.attachSequence(self.pool_id);
        errdefer replacement.releaseOwnedSequence(&self.kv_storage) catch {};
        replacement.total_token_count = self.kv_metadata.total_token_count;
        replacement.position_offset = replacement.total_token_count - compacted.retained_count;
        replacement.compacted = true;
        try replacement.appendTokensNoTrim(&self.kv_storage, compacted.retained_count);
        try replacement.syncSequenceState(&self.kv_storage);
        replacement.updateView(&self.kv_storage);
        const pool = self.kv_storage.getPool(self.pool_id) orelse return error.InvalidPoolId;
        if (replacement.kv_view) |replacement_view| {
            const replacement_ops_view = self.kvCacheView(replacement_view);
            if (try self.cb.seedPagedKvLayerCache(
                replacement_ops_view,
                replacement_view.token_count,
                0,
                compacted.k_per_layer[0],
                compacted.v_per_layer[0],
            )) {
                for (1..pool.config.num_layers_packed) |layer| {
                    _ = try self.cb.seedPagedKvLayerCache(
                        replacement_ops_view,
                        replacement_view.token_count,
                        layer,
                        compacted.k_per_layer[layer],
                        compacted.v_per_layer[layer],
                    );
                }
            } else {
                for (0..pool.config.num_layers_packed) |layer| {
                    try self.kv_storage.writeFullLayerKv(
                        replacement.sequence_id.?,
                        layer,
                        compacted.retained_count,
                        compacted.k_per_layer[layer],
                        compacted.v_per_layer[layer],
                    );
                }
            }
        } else {
            for (0..pool.config.num_layers_packed) |layer| {
                try self.kv_storage.writeFullLayerKv(
                    replacement.sequence_id.?,
                    layer,
                    compacted.retained_count,
                    compacted.k_per_layer[layer],
                    compacted.v_per_layer[layer],
                );
            }
        }

        const before_view = self.kv_metadata.kv_view;
        const before_sequence_id = self.kv_metadata.sequence_id;
        try self.kv_metadata.releaseOwnedSequence(&self.kv_storage);
        self.kv_metadata.logical_blocks.deinit(self.allocator);
        self.kv_metadata = replacement;
        try self.applyKvMutationResult(.{
            .token_count = self.kv_metadata.total_token_count,
            .kv_view = self.kv_metadata.kv_view,
            .compacted = self.kv_metadata.compacted,
            .delta = .{
                .sequence_replaced = before_sequence_id != self.kv_metadata.sequence_id,
                .logical_block_count_before = if (before_view) |view| view.logical_block_count else 0,
                .logical_block_count_after = if (self.kv_metadata.kv_view) |view| view.logical_block_count else 0,
                .position_offset_before = if (before_view) |view| view.position_offset else 0,
                .position_offset_after = self.kv_metadata.position_offset,
                .retained_tokens = compacted.retained_count,
            },
        });
    }

    fn currentTokenCount(self: *const RuntimeContext) usize {
        return self.kv_metadata.total_token_count;
    }

    fn kvView(self: *const RuntimeContext) ?generation.KvView {
        return self.kv_metadata.kv_view;
    }

    fn kvCacheView(_: *const RuntimeContext, kv_view: generation.KvView) contracts.KvCacheView {
        return executorContractKvCacheView(kv_view);
    }

    fn configureKvPolicy(self: *RuntimeContext, max_inflight_tokens: usize, allow_swa_ring: bool) !void {
        try self.kv_metadata.configureKvPolicy(
            max_inflight_tokens,
            allow_swa_ring and self.gpt_config.supportsSplitSwaGlobalKvRing(),
        );
    }

    fn validateDecodePosition(self: *const RuntimeContext, position: usize) !void {
        if (self.currentTokenCount() != position) return error.InvalidDecodePosition;
    }

    fn syncDecoderRuntimeStateForView(self: *RuntimeContext, kv_view: generation.KvView) !void {
        const prior = self.raw_span_state orelse return;
        const prior_end = prior.kv_position_offset + prior.kv_tokens;
        const current_end = kv_view.position_offset + kv_view.token_count;
        const regressed = kv_view.position_offset < prior.kv_position_offset or current_end < prior_end;
        if (regressed) {
            try self.invalidateDecoderRuntimeSpanState();
        }
    }

    fn noteDecoderRuntimeState(self: *RuntimeContext, kv_view: generation.KvView) void {
        self.raw_span_state = .{
            .kv_tokens = kv_view.token_count,
            .kv_position_offset = kv_view.position_offset,
        };
    }

    fn noteDecoderRuntimeStateFromCurrentView(self: *RuntimeContext) void {
        if (self.kvView()) |kv_view| self.noteDecoderRuntimeState(kv_view);
    }

    fn applyDecoderRuntimeSpanHints(self: *RuntimeContext, decode_context: *gpt_arch.DecodeContext) void {
        const prior = self.raw_span_state orelse return;
        decode_context.decoder_runtime_resident_kv_sequence_len = prior.kv_tokens;
        decode_context.decoder_runtime_resident_kv_position_offset = prior.kv_position_offset;
    }

    fn makeDecodeContext(
        self: *RuntimeContext,
        seq_len: usize,
        query_seq_len: usize,
        attention_mode: cache_mod.AttentionMode,
    ) gpt_arch.DecodeContext {
        var decode_context: gpt_arch.DecodeContext = undefined;
        if (disablePagedKvDebug()) {
            decode_context = .{
                .attention_mode = .full_recompute,
                .total_sequence_len = seq_len,
                .query_sequence_len = query_seq_len,
                .kv_sequence_len = seq_len,
                .kv_position_offset = 0,
                .moe_runtime = &self.moe_runtime,
            };
        } else {
            const kv_view = self.kvView();
            const resolved_attention_mode: gpt_arch.DecodeContext.AttentionMode = switch (attention_mode) {
                .full_recompute => .full_recompute,
                .paged_prefill => .paged_prefill,
                .paged_decode => .paged_decode,
            };
            decode_context = .{
                .attention_mode = if (kv_view != null) resolved_attention_mode else .full_recompute,
                .total_sequence_len = seq_len,
                .query_sequence_len = query_seq_len,
                .kv_sequence_len = if (kv_view) |view| view.token_count else seq_len,
                .kv_position_offset = if (kv_view) |view| view.position_offset else 0,
                .kv_storage = &self.kv_storage,
                .moe_runtime = &self.moe_runtime,
                .kv_cache = if (kv_view) |view|
                    executorDecodeKvCacheView(view)
                else
                    null,
            };
        }
        self.applyDecoderRuntimeSpanHints(&decode_context);
        return decode_context;
    }

    fn preparePrefill(
        self: *RuntimeContext,
        seq_len: usize,
        query_seq_len: usize,
        attention_mode: cache_mod.AttentionMode,
        recent_query_capture: ?model_runtime.RecentPrefillQueryCaptureDescriptor,
    ) !gpt_arch.DecodeContext {
        if (self.currentTokenCount() == 0) {
            if (seq_len != query_seq_len) return error.UnsupportedShape;
            try self.notePrefill(query_seq_len);
        } else {
            const expected_prior = seq_len - query_seq_len;
            if (self.currentTokenCount() != expected_prior) return error.InvalidPrefillSequence;
            try self.appendPrefillChunk(query_seq_len);
        }
        var decode_context = self.makeDecodeContext(seq_len, query_seq_len, attention_mode);
        decode_context.recent_prefill_query_capture = recent_query_capture;
        return decode_context;
    }

    fn beginDecodeStep(self: *RuntimeContext, position: usize, attention_mode: cache_mod.AttentionMode) !struct {
        seq_len: usize,
        decode_context: gpt_arch.DecodeContext,
    } {
        try self.validateDecodePosition(position);
        const seq_len = try self.appendGeneratedToken();
        return .{
            .seq_len = seq_len,
            .decode_context = self.makeDecodeContext(seq_len, 1, attention_mode),
        };
    }

    fn ensureDecoderRuntimePrepared(self: *RuntimeContext) !bool {
        timing_stats.ensure_prepared_calls += 1;
        const started_at = monotonicNowNs();
        defer timing_stats.ensure_prepared_nanos += @intCast(monotonicNowNs() - started_at);
        const kv_view = self.kvView() orelse {
            if (decoderRuntimePrepareTraceRequested()) {
                std.debug.print("prepare-trace: no kv view available before runtime prepare\n", .{});
            }
            return false;
        };
        const sync_started_at = monotonicNowNs();
        try self.syncDecoderRuntimeStateForView(kv_view);
        timing_stats.ensure_prepared_sync_nanos += @intCast(monotonicNowNs() - sync_started_at);
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        const prepare_started_at = monotonicNowNs();
        const prepare = try self.cb.decoderRuntimePrepareOrReuseFamily(
            self.allocator,
            self.gpt_config,
            kv_view.token_count,
            configured_layer_count,
        );
        if (decoderRuntimePrepareTraceRequested()) {
            std.debug.print(
                "prepare-trace: kv_tokens={d} configured_layers={d} prepared={} fast_hit={} used_greedy={} runtime_ready={}\n",
                .{
                    kv_view.token_count,
                    configured_layer_count,
                    prepare.prepared,
                    prepare.fast_hit,
                    prepare.used_greedy,
                    self.cb.decoderRuntimeReady(),
                },
            );
        }
        if (prepare.fast_hit) {
            timing_stats.ensure_prepared_fast_hits += 1;
        } else if (prepare.used_greedy) {
            timing_stats.ensure_prepared_greedy_nanos += @intCast(monotonicNowNs() - prepare_started_at);
        } else {
            timing_stats.ensure_prepared_family_nanos += @intCast(monotonicNowNs() - prepare_started_at);
        }
        return prepare.prepared;
    }

    fn forwardDecoderRuntimeLastLogits(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        token_id: i64,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
    ) !?model_runtime.ModelOutput {
        if (!(try self.ensureDecoderRuntimePrepared())) return null;
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        const output = if (try metal_runtime.forwardLastLogitsTensorFamily(
            &self.cb,
            allocator,
            self.gpt_config,
            configured_layer_count,
            token_id,
            seq_len,
            decode_context,
        )) |logits| model_runtime.ModelOutput{
            .device_logits = logits,
            .device_logits_backend = self.cb,
            .final_logit_softcap = self.gpt_config.final_logit_softcapping,
        } else null;
        if (output != null) {
            self.noteDecoderRuntimeStateFromCurrentView();
        }
        return output;
    }

    fn forwardDecoderRuntimeBackendOwnedFullLogits(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        token_id: i64,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
    ) !?model_runtime.ModelOutput {
        if (!(try self.ensureDecoderRuntimePrepared())) return null;
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        return switch (try metal_runtime.forwardBackendOwnedFullLogitsFamily(
            &self.cb,
            allocator,
            self.gpt_config,
            configured_layer_count,
            token_id,
            seq_len,
            decode_context,
        )) {
            .unsupported => null,
            .completed => |host| blk: {
                // `host` was allocated by the request allocator above. Its
                // ownership transfers directly to ModelOutput.deinit.
                self.noteDecoderRuntimeStateFromCurrentView();
                break :blk modelOutputFromOwnedHostLogits(host);
            },
        };
    }

    fn drainPipelinedDecode(self: *RuntimeContext) void {
        if (!self.pipelined_decode_armed) return;
        _ = metal_runtime.decoderRuntimePipelinedAwaitToken(&self.cb) catch null;
        self.pipelined_decode_armed = false;
        self.pipelined_decode_mode = null;
        self.pipelined_sampling = null;
        self.pipelined_kv_advanced_for = null;
        self.clearPipelinedSuppressPolicy();
    }

    fn clearPipelinedSuppressPolicy(self: *RuntimeContext) void {
        if (self.pipelined_suppress_ids) |ids| self.allocator.free(ids);
        self.pipelined_suppress_ids = null;
        self.pipelined_suppress_kind = null;
    }

    fn clearCompletedPipelinedFrame(self: *RuntimeContext, clear_policy: bool) void {
        self.pipelined_decode_armed = false;
        self.pipelined_decode_mode = null;
        self.pipelined_sampling = null;
        self.pipelined_next_position = 0;
        self.pipelined_kv_advanced_for = null;
        if (clear_policy) self.clearPipelinedSuppressPolicy();
    }

    fn clearPipelinedRequestState(self: *RuntimeContext) void {
        self.drainPipelinedDecode();
        self.clearCompletedPipelinedFrame(true);
    }

    fn pipelinedSuppressSlice(self: *const RuntimeContext) []const i32 {
        return self.pipelined_suppress_ids orelse &.{};
    }

    fn installPipelinedSuppressPolicy(self: *RuntimeContext, request_ids: []const i32) !void {
        const kind: PipelinedSuppressKind = if (request_ids.len == 0) .model else .request;
        const ids = if (request_ids.len == 0)
            self.gpt_config.suppress_token_ids[0..self.gpt_config.suppress_token_ids_len]
        else
            request_ids;
        if (pipelinedSuppressPolicyMatches(
            self.pipelined_suppress_kind,
            self.pipelinedSuppressSlice(),
            kind,
            ids,
        )) return;

        // A prior failed continuation may have already advanced KV while
        // retaining the old policy. Changing policy at that boundary cannot
        // safely replay or discard the speculative append.
        if (self.pipelined_kv_advanced_for != null)
            return error.PipelinedSuppressPolicyChangedAfterKvAdvance;

        // Allocate before draining: allocation failure cannot alter the old
        // in-flight frame or advance the new request's KV position.
        const owned = try self.allocator.dupe(i32, ids);
        if (self.pipelined_decode_armed)
            self.drainPipelinedDecode()
        else
            self.clearPipelinedSuppressPolicy();
        self.pipelined_suppress_ids = owned;
        self.pipelined_suppress_kind = kind;
    }

    fn forwardGreedyTokenForPipelinedPolicy(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        configured_layer_count: usize,
        token_id: i64,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
    ) !?i64 {
        return switch (self.pipelined_suppress_kind orelse return error.MissingPipelinedSuppressPolicy) {
            .model => metal_runtime.forwardGreedyTokenFamily(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                token_id,
                seq_len,
                decode_context,
            ),
            .request => (try metal_runtime.forwardGreedyTokenSuppressFamily(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                token_id,
                seq_len,
                decode_context,
                self.pipelinedSuppressSlice(),
            )) orelse return error.CommittedPipelinedDecodeDeclined,
        };
    }

    fn decodeGreedyPipelined(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        request: model_runtime.DecodeRequest,
    ) !?i64 {
        if (request.attention_mode != .paged_decode) return null;
        if (self.gpt_config.family != .gemma) return null;
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        if (self.pipelined_decode_armed and self.pipelined_decode_mode != .greedy) {
            self.drainPipelinedDecode();
            return error.PipelinedDecodeModeChanged;
        }
        if (self.pipelined_decode_armed and self.pipelined_next_position != request.position) {
            self.drainPipelinedDecode();
        }
        // Install only after every mismatch drain. The owned slice remains
        // alive through arm, continuation, and final await.
        try self.installPipelinedSuppressPolicy(request.suppress_token_ids);
        errdefer self.drainPipelinedDecode();
        if (!self.pipelined_decode_armed) {
            var arm_seq_len: usize = 0;
            var arm_context: gpt_arch.DecodeContext = undefined;
            if (self.pipelined_kv_advanced_for) |advanced_position| {
                if (advanced_position != request.position) return error.InvalidPagedKvState;
                self.pipelined_kv_advanced_for = null;
                arm_seq_len = request.position + 1;
                arm_context = self.makeDecodeContext(arm_seq_len, 1, request.attention_mode);
            } else {
                const step = try self.beginDecodeStep(request.position, request.attention_mode);
                arm_seq_len = step.seq_len;
                arm_context = step.decode_context;
            }
            if (!(try self.ensureDecoderRuntimePrepared())) {
                // KV already advanced; run the synchronous forward on this state.
                return try self.forwardGreedyTokenForPipelinedPolicy(
                    allocator,
                    configured_layer_count,
                    request.token_id,
                    arm_seq_len,
                    &arm_context,
                );
            }
            const armed = switch (self.pipelined_suppress_kind.?) {
                .model => metal_runtime.forwardGreedyTokenPipelinedArmFamily(
                    &self.cb,
                    allocator,
                    self.gpt_config,
                    configured_layer_count,
                    request.token_id,
                    arm_seq_len,
                    &arm_context,
                ) catch false,
                .request => try metal_runtime.forwardGreedyTokenPipelinedArmSuppressFamily(
                    &self.cb,
                    allocator,
                    self.gpt_config,
                    configured_layer_count,
                    request.token_id,
                    arm_seq_len,
                    &arm_context,
                    self.pipelinedSuppressSlice(),
                ),
            };
            if (!armed) {
                return try self.forwardGreedyTokenForPipelinedPolicy(
                    allocator,
                    configured_layer_count,
                    request.token_id,
                    arm_seq_len,
                    &arm_context,
                );
            }
            self.pipelined_decode_armed = true;
            self.pipelined_decode_mode = .greedy;
            self.pipelined_sampling = null;
            self.pipelined_next_position = request.position;
        }
        if (!request.allow_pipeline_rearm) {
            // The caller knows this is the last requested output token. The
            // current pending frame is authoritative and must complete, but
            // encoding position+1 would only produce an unused continuation.
            // Await first: no logical/KV state is changed until the backend
            // has fenced and returned a valid result.
            const token = (metal_runtime.decoderRuntimePipelinedAwaitToken(&self.cb) catch |err| {
                // await_only is the single wait owner. A wait failure can
                // leave an active frame or pending seed cleanup, so cancel it
                // without calling drainPipelinedDecode and awaiting twice.
                metal_runtime.decoderRuntimePipelinedCancelPending(&self.cb);
                self.clearCompletedPipelinedFrame(false);
                self.noteDecoderRuntimeStateFromCurrentView();
                return err;
            }) orelse {
                metal_runtime.decoderRuntimePipelinedCancelPending(&self.cb);
                self.clearCompletedPipelinedFrame(false);
                self.noteDecoderRuntimeStateFromCurrentView();
                return error.InvalidModelOutput;
            };
            self.clearCompletedPipelinedFrame(false);
            self.noteDecoderRuntimeStateFromCurrentView();
            if (token < 0) return error.InvalidModelOutput;
            return token;
        }
        // Pending frame produces the token at position request.position + 1.
        // Speculatively advance KV bookkeeping and encode the next frame with
        // the device-resident token id, then wait the pending frame.
        const next_step = try self.beginDecodeStep(request.position + 1, request.attention_mode);
        if (!(try self.ensureDecoderRuntimePrepared())) {
            self.pipelined_kv_advanced_for = request.position + 1;
            const token = (try metal_runtime.decoderRuntimePipelinedAwaitToken(&self.cb)) orelse return error.InvalidModelOutput;
            self.pipelined_decode_armed = false;
            self.pipelined_decode_mode = null;
            self.pipelined_sampling = null;
            if (token < 0) return error.InvalidModelOutput;
            self.noteDecoderRuntimeStateFromCurrentView();
            return token;
        }
        const stepped = switch (self.pipelined_suppress_kind.?) {
            .model => try metal_runtime.forwardGreedyTokenPipelinedStepFamily(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                next_step.seq_len,
                &next_step.decode_context,
            ),
            .request => try metal_runtime.forwardGreedyTokenPipelinedStepSuppressFamily(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                next_step.seq_len,
                &next_step.decode_context,
                self.pipelinedSuppressSlice(),
            ),
        };
        if (stepped) |token| {
            if (token < 0) return error.InvalidModelOutput;
            if (metal_runtime.decoderRuntimePipelinedSubmitPending(&self.cb)) {
                self.pipelined_decode_armed = true;
                self.pipelined_decode_mode = .greedy;
                self.pipelined_sampling = null;
                self.pipelined_next_position = request.position + 1;
            } else {
                metal_runtime.decoderRuntimePipelinedCancelPending(&self.cb);
                self.pipelined_decode_armed = false;
                self.pipelined_decode_mode = null;
                self.pipelined_sampling = null;
                self.pipelined_kv_advanced_for = request.position + 1;
            }
            self.noteDecoderRuntimeStateFromCurrentView();
            return token;
        }
        // Step failed after speculative append: drain pending frame for its token.
        self.pipelined_kv_advanced_for = request.position + 1;
        const token = (try metal_runtime.decoderRuntimePipelinedAwaitToken(&self.cb)) orelse return error.InvalidModelOutput;
        self.pipelined_decode_armed = false;
        self.pipelined_decode_mode = null;
        self.pipelined_sampling = null;
        if (token < 0) return error.InvalidModelOutput;
        self.noteDecoderRuntimeStateFromCurrentView();
        return token;
    }

    fn decodeSamplePipelined(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        request: model_runtime.SampledDecodeRequest,
    ) !?i64 {
        if (request.decode.attention_mode != .paged_decode) {
            if (self.pipelined_decode_armed) return error.PipelinedSamplingContractChanged;
            return null;
        }
        if (self.gpt_config.family != .gemma) {
            if (self.pipelined_decode_armed) return error.PipelinedSamplingContractChanged;
            return null;
        }
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();

        try validateSampledPipelineContinuation(
            self.pipelined_decode_armed,
            self.pipelined_decode_mode,
            self.pipelined_next_position,
            request.decode.position,
        );
        if (self.pipelined_decode_armed) {
            if (!sampledPipelineConfigMatches(self.pipelined_sampling, request.sampling)) {
                return error.PipelinedSamplingContractChanged;
            }
            if (!decoder_gated_runtime.residentSampledTokenPipelineContinuationSupported(
                self.gpt_config,
                configured_layer_count,
                request.sampling,
            )) return error.PipelinedSamplingContractChanged;
        } else {
            // runtimeSupportsSampleDecodeRequest completed the backend
            // capability check and fixed-capacity scratch reservation before
            // decodeSample entered this method.
            if (!(try self.ensureDecoderRuntimePrepared())) return null;
        }
        if (!self.pipelined_decode_armed) {
            if (self.pipelined_kv_advanced_for != null) return error.InvalidPagedKvState;
            const arm_step = try self.beginDecodeStep(request.decode.position, request.decode.attention_mode);
            const armed = try decoder_gated_runtime.forwardSampledTokenPipelinedArm(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                request.decode.token_id,
                arm_step.seq_len,
                &arm_step.decode_context,
                request.sampling,
                request.token_history,
            );
            if (!armed) return error.ResidentSamplingPreflightChanged;
            self.pipelined_decode_armed = true;
            self.pipelined_decode_mode = .sampled_nucleus;
            self.pipelined_sampling = request.sampling;
            self.pipelined_next_position = request.decode.position;
        }
        errdefer self.drainPipelinedDecode();

        // Encode B while submitted A executes, then wait/read A before B is
        // submitted. The single sampler scratch set is therefore never used
        // by two executing command buffers at once.
        const next_step = try self.beginDecodeStep(request.decode.position + 1, request.decode.attention_mode);
        const token = (try decoder_gated_runtime.forwardSampledTokenPipelinedStep(
            &self.cb,
            allocator,
            self.gpt_config,
            configured_layer_count,
            next_step.seq_len,
            &next_step.decode_context,
            request.sampling,
            request.token_history,
        )) orelse return error.ResidentSamplingFrameFailed;
        try submitSampledPipelineToken(&self.cb, token, self.gpt_config.vocab_size);
        self.pipelined_decode_armed = true;
        self.pipelined_decode_mode = .sampled_nucleus;
        self.pipelined_sampling = request.sampling;
        self.pipelined_next_position = request.decode.position + 1;
        self.noteDecoderRuntimeStateFromCurrentView();
        return token;
    }

    fn forwardDecoderRuntimeGreedyToken(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        token_id: i64,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
        suppress_token_ids: []const i32,
    ) !?i64 {
        if (!(try self.ensureDecoderRuntimePrepared())) return null;
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        const token = if (suppress_token_ids.len == 0)
            try metal_runtime.forwardGreedyTokenFamily(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                token_id,
                seq_len,
                decode_context,
            )
        else switch (self.gpt_config.family) {
            .llama, .mistral, .qwen2, .gemma => try decoder_gated_runtime.forwardGreedyTokenSuppress(
                &self.cb,
                allocator,
                self.gpt_config,
                configured_layer_count,
                token_id,
                seq_len,
                decode_context,
                suppress_token_ids,
            ),
            else => null,
        };
        if (token != null) {
            self.noteDecoderRuntimeStateFromCurrentView();
        }
        return token;
    }

    fn forwardDecoderRuntimeSampledToken(
        self: *RuntimeContext,
        allocator: std.mem.Allocator,
        token_id: i64,
        seq_len: usize,
        decode_context: *const gpt_arch.DecodeContext,
        sampling: model_runtime.SamplingConfig,
        token_history: []const i64,
    ) !?i64 {
        if (!(try self.ensureDecoderRuntimePrepared())) return null;
        const configured_layer_count = self.decoderRuntimeConfiguredLayerCount();
        const token = try metal_runtime.forwardSampledTokenFamily(
            &self.cb,
            allocator,
            self.gpt_config,
            configured_layer_count,
            token_id,
            seq_len,
            decode_context,
            sampling,
            token_history,
        );
        if (token != null) {
            self.noteDecoderRuntimeStateFromCurrentView();
        }
        return token;
    }
};

fn executorKvSlidingWindowSize(config: gpt_mod.Config) ?u32 {
    return config.kvPoolSlidingWindowSize(false);
}

const runtime_vtable = model_runtime.ModelRuntime.VTable{
    .capabilities = runtimeCapabilities,
    .prepare = runtimePrepare,
    .prefill = runtimePrefill,
    .decode = runtimeDecode,
    .decode_sample = runtimeDecodeSample,
    .supports_sample_decode_request = runtimeSupportsSampleDecodeRequest,
    .decode_greedy = runtimeDecodeGreedy,
    .verify_greedy_transaction = runtimeVerifyGreedyTransaction,
    .deinit = runtimeDeinit,
    .reset = runtimeReset,
    .finish_prefill_for_decode = runtimeFinishPrefillForDecode,
    .exact_paged_kv_authority = runtimeExactPagedKvAuthority,
    .export_exact_paged_kv_ranges = runtimeExportExactPagedKvRanges,
    .handoff_exact_paged_kv = runtimeHandoffExactPagedKv,
    .discard_exact_paged_kv_source = runtimeDiscardExactPagedKvSource,
    .bind_context_owner = runtimeBindContextOwner,
    .drain_context_owner_snapshot = runtimeDrainContextOwnerSnapshot,
    .context_owner_binding_state = runtimeContextOwnerBindingState,
    .debug_timing_stats = runtimeDebugTimingStats,
    .reset_debug_timing_stats = runtimeResetDebugTimingStats,
    .print_debug_timing = runtimePrintDebugTiming,
};

fn runtimeBindContextOwner(ctx: *anyopaque, binding: model_runtime.ContextOwnerBinding) !void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try binding.validate();
    const tracker = binding.owner.tracker;
    if (runtime_ctx.bound_context_owner) |bound| {
        if (bound.owner.tracker != tracker) return error.ContextFrameOwnershipMismatch;
        return;
    }
    try bindCompiledOwnerStorage(&runtime_ctx.kv_storage, binding);
    runtime_ctx.bound_context_owner = binding;
}

fn runtimeDrainContextOwnerSnapshot(ctx: *anyopaque, binding: model_runtime.ContextOwnerBinding) !void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try binding.validate();
    const bound = runtime_ctx.bound_context_owner orelse return error.ContextPhysicalAccountingUnavailable;
    if (bound.owner != binding.owner or bound.owner.tracker != binding.owner.tracker or
        bound.witness.owner_identity != binding.witness.owner_identity or
        bound.witness.load_generation != binding.witness.load_generation)
        return error.ContextFrameOwnershipMismatch;
    try runtime_ctx.kv_storage.drainDeviceWritesForOwnerSnapshot();
    runtime_ctx.clearPipelinedRequestState();
}

fn runtimeContextOwnerBindingState(ctx: *anyopaque, binding: model_runtime.ContextOwnerBinding) model_runtime.ModelRuntime.ContextOwnerBindingState {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const bound = runtime_ctx.bound_context_owner orelse return .unbound;
    if (bound.owner == binding.owner and bound.owner.tracker == binding.owner.tracker and
        bound.witness.owner_identity == binding.witness.owner_identity and
        bound.witness.load_generation == binding.witness.load_generation) return .exact;
    return .other;
}

fn bindCompiledOwnerStorage(storage: *runtime.kv.storage_runtime.KvStorageRuntime, binding: model_runtime.ContextOwnerBinding) !void {
    const tracker = binding.owner.tracker;
    try storage.bindRequestOwnerTracker(tracker);
    errdefer storage.unbindRequestOwnerTracker(tracker) catch {};
    try binding.owner.bindCompiledStorage(binding.witness, tracker);
}

test "compiled owner receipt failure rolls back the real Metal storage hook" {
    const allocator = std.testing.allocator;
    const raw = metal_runtime.termite_metal_decode_runtime_create() orelse return error.SkipZigTest;
    defer metal_runtime.termite_metal_decode_runtime_destroy(raw);
    if (metal_runtime.termite_metal_decode_runtime_ready(raw) == 0) return error.SkipZigTest;

    var storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(allocator, .{
        .backend = .metal,
        .dtype = .f16,
        .page_size_tokens = 4,
        .num_layers_packed = 1,
        .num_kv_heads = 1,
        .head_dim = 2,
    });
    defer storage.deinit();
    const metal_storage = try backends.metal_kv_storage.MetalKvStorage.create(allocator, raw, .f16, 1, 2, 4);
    storage.setDeviceWriteHook(metal_storage.deviceWriteHook());

    var owner = try runtime.kv.host_allocation_tracker.RequestAllocationOwner.init(null);
    const witness = try owner.beginContextScope(41);
    try owner.requireCompiledStorage(witness);
    // Force the receipt stage to fail only after the production helper has
    // installed the tracker into both storage layers.
    try owner.bindCompiledStorage(witness, owner.tracker);
    try std.testing.expectError(error.ContextFrameOwnershipMismatch, bindCompiledOwnerStorage(&storage, .{ .owner = &owner, .witness = witness }));
    try std.testing.expect(storage.ownerAllocationTracker() == null);
    try std.testing.expect(metal_storage.owner_allocation_tracker == null);
    try owner.detachCompiledStorage(witness, owner.tracker);
    _ = owner.release();
}

const executor_vtable = model_runtime.ModelExecutor.VTable{
    .create_runtime = createRuntime,
    .deinit = executorDeinit,
};

fn envFlag(name: [:0]const u8) bool {
    const value = c_std.getenv(name) orelse return false;
    const slice = std.mem.span(value);
    return slice.len > 0 and !std.mem.eql(u8, slice, "0");
}

fn cleanupBoundedPleRequest(backend: anytype) !void {
    // The backend's captured immutable policy is authoritative. Environment
    // flags may change after prepare and cannot safely select this route.
    if (try backend.decoderRuntimeResetBoundedPleRequest()) return;
    // False means this backend never captured handoff ownership. Preserve the
    // legacy v43 fence and mirror purge in that case.
    try backend.decoderRuntimeResetState();
    // The boolean reports whether a mirror had been prepared. False after a
    // successful purge/disarm is normal; only a thrown error is a failure.
    _ = try backend.decoderRuntimeFinalizeE2bF16Mirrors(null);
}

fn cleanupBoundedPleRequestAtPublicReset(backend: anytype) !void {
    // False means no captured phase. Public reset already performed the
    // ordinary revoke/fence and must retain legacy warm mirrors, as in v43.
    _ = try backend.decoderRuntimeResetBoundedPleRequest();
}

test "bounded PLE cleanup follows captured ownership and preserves legacy purge" {
    const Mock = struct {
        resets: usize = 0,
        finalizes: usize = 0,
        phase_resets: usize = 0,
        phase_result: bool = true,
        phase_error: bool = false,
        fn decoderRuntimeResetState(self: *@This()) !void {
            self.resets += 1;
        }
        fn decoderRuntimeFinalizeE2bF16Mirrors(self: *@This(), _: anytype) !bool {
            self.finalizes += 1;
            return false;
        }
        fn decoderRuntimeResetBoundedPleRequest(self: *@This()) !bool {
            self.phase_resets += 1;
            if (self.phase_error) return error.InjectedFenceFailure;
            return self.phase_result;
        }
    };
    var mock = Mock{};
    mock.phase_result = false;
    try cleanupBoundedPleRequest(&mock);
    try std.testing.expectEqual(@as(usize, 1), mock.resets);
    try std.testing.expectEqual(@as(usize, 1), mock.finalizes);
    try std.testing.expectEqual(@as(usize, 1), mock.phase_resets);
    mock.phase_result = true;
    try cleanupBoundedPleRequest(&mock);
    try std.testing.expectEqual(@as(usize, 1), mock.resets);
    try std.testing.expectEqual(@as(usize, 1), mock.finalizes);
    try std.testing.expectEqual(@as(usize, 2), mock.phase_resets);
    mock.phase_error = true;
    try std.testing.expectError(error.InjectedFenceFailure, cleanupBoundedPleRequest(&mock));
    try std.testing.expectEqual(@as(usize, 1), mock.resets);
    try std.testing.expectEqual(@as(usize, 1), mock.finalizes);
}

test "public reset retains legacy mirrors while error cleanup purges them" {
    const Mock = struct {
        resets: usize = 0,
        finalizes: usize = 0,
        phase_resets: usize = 0,
        fn decoderRuntimeResetBoundedPleRequest(self: *@This()) !bool {
            self.phase_resets += 1;
            return false;
        }
        fn decoderRuntimeResetState(self: *@This()) !void {
            self.resets += 1;
        }
        fn decoderRuntimeFinalizeE2bF16Mirrors(self: *@This(), _: anytype) !bool {
            self.finalizes += 1;
            return false;
        }
    };
    var mock = Mock{};
    try cleanupBoundedPleRequestAtPublicReset(&mock);
    try std.testing.expectEqual(@as(usize, 1), mock.phase_resets);
    try std.testing.expectEqual(@as(usize, 0), mock.resets);
    try std.testing.expectEqual(@as(usize, 0), mock.finalizes);
    try cleanupBoundedPleRequest(&mock);
    try std.testing.expectEqual(@as(usize, 2), mock.phase_resets);
    try std.testing.expectEqual(@as(usize, 1), mock.resets);
    try std.testing.expectEqual(@as(usize, 1), mock.finalizes);
}

fn pipelinedDecodeFrameEnabledForFlags(
    enable_requested: bool,
    disable_requested: bool,
    a4b_high_memory_requested: bool,
    qualified_a4b: bool,
    prepared_a4b: bool,
) bool {
    if (disable_requested) return false;
    if (!qualified_a4b) return enable_requested;
    return prepared_a4b and (enable_requested or a4b_high_memory_requested);
}

fn pipelinedDecodeFrameEnabled(gpt_config: gpt_mod.Config) bool {
    const qualified_a4b = gemma4_runtime.isQualifiedA4bArchitecture(gpt_config);
    if (!qualified_a4b) return metal_runtime.pipelinedDecodeFrameEnabled();
    return pipelinedDecodeFrameEnabledForFlags(
        platform.env.getenvBool("TERMITE_METAL_ENABLE_PIPELINED_DECODE_FRAME"),
        platform.env.getenvBool("TERMITE_METAL_DISABLE_PIPELINED_DECODE_FRAME"),
        a4b_feature_flags.highMemoryFastPathEnabled(),
        qualified_a4b,
        qualified_a4b and gemma4_runtime.supportsPreparedA4bRuntimeConfig(gpt_config),
    );
}

fn decoderRuntimePrefillAfterPrepareRequested() bool {
    return envFlag("TERMITE_METAL_PREFILL_DIRECT_AFTER_PREPARE");
}

fn decoderRuntimePrefillTraceRequested() bool {
    return envFlag("TERMITE_METAL_PREFILL_TRACE");
}

fn decoderRuntimePrepareTraceRequested() bool {
    return envFlag("TERMITE_METAL_PREPARE_TRACE");
}

fn disableDeviceKvReserveRequested() bool {
    return envFlag("TERMITE_METAL_DISABLE_DEVICE_KV_RESERVE");
}

pub fn supportsSession(session: backends.Session) bool {
    const gpt_config = session_factory.getGptConfig(session) orelse return false;
    if (!session.backend().usesGpuHostedSession()) return false;
    return metal_runtime.supportsDecoderRuntimeConfig(gpt_config);
}

fn prewarmEmbeddingWeight(cb: *const ops.ComputeBackend, gpt_config: gpt_mod.Config) !void {
    const embed_w = try gpt_arch.getEmbeddingWeight(cb, gpt_config);
    defer cb.free(embed_w);
    const ids = [_]i64{0};
    const embedded = try cb.embeddingLookup(embed_w, ids[0..], ids.len, gpt_config.hidden_size);
    defer cb.free(embedded);
}

fn qualifiedE2bMirrorWholeModelConfig(config: gpt_mod.Config) bool {
    if (config.family != .gemma or config.usesMoe() or config.gemma4_mtp_assistant or !config.hasPle()) return false;
    const e2 =
        !config.usesMoe() and
        config.hidden_size == 1536 and
        config.num_hidden_layers == 35 and
        config.intermediate_size == 6144 and
        config.shared_layer_intermediate_size == 12288 and
        config.num_attention_heads == 8 and
        config.headDim() == 256 and
        config.global_head_dim == 512 and
        config.sliding_window == 512;
    const e4 = config.hidden_size == 2560 and
        config.num_hidden_layers == 42 and
        config.num_kv_shared_layers == 18 and
        config.intermediate_size == 10240 and
        config.num_attention_heads == 8 and
        config.num_key_value_heads == 2 and
        config.headDim() == 256 and
        config.global_head_dim == 512 and
        config.sliding_window == 512;
    return e2 or e4;
}

fn e2bMirrorNonlinearBoundarySatisfied(
    qualified_config: bool,
    family_prepared: bool,
    embedding_prepared: bool,
    prefill_plan_prepared: bool,
    ple_embedding_prepared: bool,
) bool {
    return qualified_config and family_prepared and embedding_prepared and
        prefill_plan_prepared and ple_embedding_prepared;
}

test "E2B mirror nonlinear completion boundary is fail closed" {
    try std.testing.expect(e2bMirrorNonlinearBoundarySatisfied(true, true, true, true, true));
    try std.testing.expect(!e2bMirrorNonlinearBoundarySatisfied(false, true, true, true, true));
    try std.testing.expect(!e2bMirrorNonlinearBoundarySatisfied(true, false, true, true, true));
    try std.testing.expect(!e2bMirrorNonlinearBoundarySatisfied(true, true, false, true, true));
    try std.testing.expect(!e2bMirrorNonlinearBoundarySatisfied(true, true, true, false, true));
    try std.testing.expect(!e2bMirrorNonlinearBoundarySatisfied(true, true, true, true, false));
}

fn a4bMappedLayer0PrewarmEnabled(gpt_config: gpt_mod.Config) bool {
    return gemma4_runtime.isQualifiedA4bArchitecture(gpt_config) and
        gemma4_runtime.getenvBool("TERMITE_METAL_ENABLE_A4B_LAYER0_DEVICE_MAPPED_MOE") and
        gemma4_runtime.getenvBool("TERMITE_METAL_ENABLE_A4B_LAYER0_DEVICE_MAPPED_PREWARM");
}

fn prewarmA4bMappedLayer0(cb: *const ops.ComputeBackend, gpt_config: gpt_mod.Config) !void {
    if (!a4bMappedLayer0PrewarmEnabled(gpt_config)) return;
    const w1 = try gpt_arch.getModelWeight(
        cb,
        gpt_config,
        "model.layers.0.block_sparse_moe.packed.w1.weight",
    );
    defer cb.free(w1);
    const w3 = try gpt_arch.getModelWeight(
        cb,
        gpt_config,
        "model.layers.0.block_sparse_moe.packed.w3.weight",
    );
    defer cb.free(w3);
    const w2 = try gpt_arch.getModelWeight(
        cb,
        gpt_config,
        "model.layers.0.block_sparse_moe.packed.w2.weight",
    );
    defer cb.free(w2);
    const result = (try cb.prewarmA4bMappedLayer0(&.{
        .layer_index = 0,
        .w1 = w1,
        .w3 = w3,
        .w2 = w2,
    })) orelse return error.A4bMetalMappedLayer0PrewarmUnavailable;
    std.log.info(
        "A4B Metal layer-0 mapped prewarm complete: logical_bytes={d} page_touches={d}",
        .{ result.logical_bytes, result.page_touches },
    );
}

pub fn prewarmSharedDecoderRuntime(
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
) !bool {
    return prewarmSharedDecoderRuntimeWithControl(allocator, session, gpt_config, null);
}

pub fn prewarmSharedDecoderRuntimeWithControl(
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    control: ?@import("../execution_control.zig").InferenceExecutionControl,
) !bool {
    if (control) |active| try active.update(.preparing_weights, 0, 1);
    if (!supportsSession(session)) return false;
    if (gemma4_runtime.shouldSkipSharedDecoderPrewarm(gpt_config)) return false;

    var managed = try session_factory.getComputeBackendWithControl(session, allocator, control);
    defer managed.deinit();
    const cb = &managed.backend;

    // A4B uses the graph fallback rather than the prepared dense-family
    // decoder, so its bounded mapped-page warmup is the complete shared
    // pre-publication prepare operation when explicitly enabled.
    if (a4bMappedLayer0PrewarmEnabled(gpt_config)) {
        try prewarmA4bMappedLayer0(cb, gpt_config);
        if (control) |active| try active.update(.preparing_weights, 1, 1);
        return true;
    }
    // Multi-row prefill and the prepared A4B decoder share the provider's
    // fixed-capacity weight-slot directory. Install A4B only after canonical
    // prefill has published its dynamic slots.
    if (gemma4_runtime.supportsPreparedA4bRuntimeConfig(gpt_config)) return false;

    const configured_layer_count = gpt_config.num_hidden_layers;
    const prepare = try cb.decoderRuntimePrepareOrReuseFamily(
        allocator,
        gpt_config,
        0,
        configured_layer_count,
    );
    if (prepare.prepared) {
        try cb.checkExecutionControl();
        try prewarmEmbeddingWeight(cb, gpt_config);
        try cb.checkExecutionControl();
        try prewarmA4bMappedLayer0(cb, gpt_config);
    }
    if (control) |active| try active.update(.preparing_weights, 1, 1);
    return prepare.prepared;
}

test "Metal prewarm checks request lifetime before session inspection" {
    try std.testing.expectError(error.Timeout, prewarmSharedDecoderRuntimeWithControl(
        std.testing.allocator,
        undefined,
        .{},
        .{ .deadline_ns = 0 },
    ));
}

pub fn createModelExecutor(
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
) !model_runtime.ModelExecutor {
    return createModelExecutorBound(allocator, session, gpt_config, kv_dtype, shared_moe_cache, null);
}

pub fn createModelExecutorBound(
    allocator: std.mem.Allocator,
    session: backends.Session,
    gpt_config: gpt_mod.Config,
    kv_dtype: ?runtime.kv.pool.KvDType,
    shared_moe_cache: ?*runtime.moe.shared.SharedExpertCache,
    model_identity: ?runtime.kv.attention_match_checkpoint.Identity,
) !model_runtime.ModelExecutor {
    if (!supportsSession(session)) return error.UnsupportedCompileBackend;
    const ctx = try allocator.create(ExecutorContext);
    ctx.* = .{
        .allocator = allocator,
        .session = session,
        .gpt_config = gpt_config,
        .kv_dtype = kv_dtype,
        .shared_moe_cache = shared_moe_cache,
        .model_identity = model_identity,
    };
    return .{ .ptr = ctx, .vtable = &executor_vtable, .model_identity = model_identity };
}

fn createRuntime(ctx: *anyopaque, allocator: std.mem.Allocator) !model_runtime.ModelRuntime {
    const exec_ctx: *ExecutorContext = @ptrCast(@alignCast(ctx));
    const runtime_ctx = try RuntimeContext.init(
        allocator,
        exec_ctx.session,
        exec_ctx.gpt_config,
        exec_ctx.kv_dtype,
        exec_ctx.shared_moe_cache,
        exec_ctx.model_identity,
    );
    return .{ .ptr = runtime_ctx, .vtable = &runtime_vtable, .model_identity = exec_ctx.model_identity };
}

fn executorDeinit(ctx: *anyopaque) void {
    const exec_ctx: *ExecutorContext = @ptrCast(@alignCast(ctx));
    exec_ctx.allocator.destroy(exec_ctx);
}

fn runtimeCapabilities(ctx: *anyopaque) model_runtime.RuntimeCapabilities {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    // Decode context assembly and hot-path token bookkeeping are executor-owned
    // now, but KV mutation/storage still depends on the host-backed block
    // manager and paged cache metadata. Keep this at runtime_owned_host_cache
    // until that metadata and mutation policy move fully behind the backend.
    return .{
        .supports_decode = true,
        .supports_sample_decode = true,
        .supports_greedy_decode = true,
        .supports_greedy_verify_transaction = gemma4_runtime.supportsPreparedDenseRuntimeConfig(runtime_ctx.gpt_config) and
            !gemma4_runtime.isQualifiedA4bArchitecture(runtime_ctx.gpt_config) and
            runtime_ctx.decoderRuntimeExecutorEnabled(),
        .state_ownership = .runtime_owned_host_cache,
        .interruption = .uninterruptible,
    };
}

fn verifyTokenIsEos(token_id: u32, eos_token_ids: []const i32) bool {
    for (eos_token_ids) |eos_token_id| {
        if (eos_token_id >= 0 and token_id == @as(u32, @intCast(eos_token_id))) return true;
    }
    return false;
}

fn resolveGreedyVerifyChoices(
    request: model_runtime.GreedyVerifyRequest,
    target_choices: []const u32,
) !model_runtime.GreedyVerifyOutput {
    if (target_choices.len < 2) return error.InvalidTensorShape;
    var result = model_runtime.GreedyVerifyOutput{};
    const proposal: u32 = @intCast(request.proposal_token_id);
    const proposal_matches = target_choices[0] == proposal;
    if (proposal_matches and (request.ignore_eos or
        !verifyTokenIsEos(target_choices[0], request.eos_token_ids)))
    {
        result.matched_proposals = 1;
        result.accepted_token_ids[0] = request.proposal_token_id;
        result.accepted_count = 1;
        if (request.accept_bonus) {
            if (!request.ignore_eos and verifyTokenIsEos(target_choices[1], request.eos_token_ids)) {
                result.hit_eos = true;
            } else {
                result.accepted_token_ids[1] = @intCast(target_choices[1]);
                result.accepted_count = 2;
                result.bonus_added = true;
            }
        }
    } else if (proposal_matches) {
        result.hit_eos = true;
    } else if (!request.ignore_eos and verifyTokenIsEos(target_choices[0], request.eos_token_ids)) {
        result.hit_eos = true;
    } else {
        result.accepted_token_ids[0] = @intCast(target_choices[0]);
        result.accepted_count = 1;
        result.correction_added = true;
    }
    const rollback_rows: usize = 1 - @as(usize, result.matched_proposals);
    result.materialized_rows = 1 + result.matched_proposals;
    result.rolled_back_rows = @intCast(rollback_rows);
    return result;
}

fn validateGreedyVerifyRequestForModel(
    request: model_runtime.GreedyVerifyRequest,
    vocab_size: usize,
) !usize {
    try request.check();
    const pending: usize = std.math.cast(usize, request.pending_token_id) orelse
        return error.InvalidModelOutput;
    const proposal: usize = std.math.cast(usize, request.proposal_token_id) orelse
        return error.InvalidModelOutput;
    if (pending >= vocab_size or proposal >= vocab_size) return error.InvalidModelOutput;
    return std.math.add(usize, request.position, 2) catch error.InvalidDecodePosition;
}

fn restoreGreedyVerifyTransaction(
    runtime_ctx: *RuntimeContext,
    original_position: usize,
) !void {
    const current = runtime_ctx.currentTokenCount();
    if (current == original_position) return;
    if (current > original_position) {
        runtime_ctx.truncateGeneratedTokens(current - original_position) catch {
            runtime_ctx.resetState() catch {};
            return error.GreedyVerifyRollbackFailed;
        };
        if (runtime_ctx.currentTokenCount() == original_position) return;
    }
    // A regressed or otherwise inconsistent cursor cannot be repaired by
    // suffix truncation. Reset the request-local runtime so no later decode can
    // observe poisoned KV metadata, then surface the transaction failure.
    runtime_ctx.resetState() catch {};
    return error.GreedyVerifyRollbackFailed;
}

fn runtimeVerifyGreedyTransaction(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.GreedyVerifyRequest,
) !model_runtime.GreedyVerifyOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const original_token_count = runtime_ctx.currentTokenCount();
    return runtimeVerifyGreedyTransactionImpl(runtime_ctx, allocator, request) catch |err| {
        restoreGreedyVerifyTransaction(runtime_ctx, original_token_count) catch |rollback_err| return rollback_err;
        return err;
    };
}

fn runtimeVerifyGreedyTransactionImpl(
    runtime_ctx: *RuntimeContext,
    allocator: std.mem.Allocator,
    request: model_runtime.GreedyVerifyRequest,
) !model_runtime.GreedyVerifyOutput {
    const verify_seq = try validateGreedyVerifyRequestForModel(request, runtime_ctx.gpt_config.vocab_size);
    runtime_ctx.drainPipelinedDecode();
    runtime_ctx.clearGreedyDeviceToken();
    if (!gemma4_runtime.supportsPreparedDenseRuntimeConfig(runtime_ctx.gpt_config) or
        gemma4_runtime.isQualifiedA4bArchitecture(runtime_ctx.gpt_config) or
        !runtime_ctx.decoderRuntimeExecutorEnabled() or
        request.attention_mode != .paged_prefill)
        return error.UnsupportedGreedyVerifyTransaction;
    if (runtime_ctx.currentTokenCount() != request.position)
        return error.InvalidDecodePosition;
    // Finish every fallible preflight before reserving KV. Once rows are
    // reserved, all errors must roll the complete suffix back locally.
    if (!(try runtime_ctx.ensureDecoderRuntimePrepared()))
        return error.UnsupportedGreedyVerifyTransaction;

    const verify_rows: usize = 2;
    const input_ids = [_]i64{ request.pending_token_id, request.proposal_token_id };
    const decode_context = try runtime_ctx.preparePrefill(verify_seq, verify_rows, request.attention_mode, null);

    var hidden_rows = (try decoder_gated_runtime.forwardFinalHiddenRowsPreparedArgmax(
        &runtime_ctx.cb,
        allocator,
        runtime_ctx.gpt_config,
        runtime_ctx.decoderRuntimeConfiguredLayerCount(),
        &input_ids,
        verify_seq,
        &decode_context,
        request.suppress_token_ids,
    )) orelse return error.UnsupportedGreedyVerifyTransaction;
    defer hidden_rows.deinit(&runtime_ctx.cb);
    if (hidden_rows.rows != verify_rows) return error.InvalidTensorShape;

    var choices = hidden_rows.prepared_tail_choices;
    hidden_rows.prepared_tail_choices = null;
    hidden_rows.choices_allocator = null;
    defer if (choices) |owned| allocator.free(owned);
    if (choices == null) {
        choices = try decoder_gated_runtime.forwardPreparedLmHeadArgmaxRows(
            &runtime_ctx.cb,
            runtime_ctx.gpt_config,
            runtime_ctx.decoderRuntimeConfiguredLayerCount(),
            hidden_rows.final_hidden,
            verify_rows,
            request.suppress_token_ids,
            allocator,
        );
    }
    const target_choices = choices orelse return error.UnsupportedGreedyVerifyTransaction;
    if (target_choices.len < verify_rows) return error.InvalidTensorShape;
    for (target_choices[0..verify_rows]) |choice| {
        if (@as(usize, choice) >= runtime_ctx.gpt_config.vocab_size) return error.InvalidModelOutput;
    }

    const result = try resolveGreedyVerifyChoices(request, target_choices);
    const rollback_rows: usize = result.rolled_back_rows;
    if (rollback_rows != 0) try runtime_ctx.truncateGeneratedTokens(rollback_rows);
    runtime_ctx.noteDecoderRuntimeStateFromCurrentView();
    return result;
}

test "transactional greedy verification resolves match correction budget and EOS" {
    const base = model_runtime.GreedyVerifyRequest{
        .pending_token_id = 10,
        .proposal_token_id = 20,
        .position = 7,
        .eos_token_ids = &.{2},
    };
    const full = try resolveGreedyVerifyChoices(base, &.{ 20, 30 });
    try std.testing.expectEqualSlices(i64, &.{ 20, 30 }, full.accepted());
    try std.testing.expectEqual(@as(u8, 1), full.matched_proposals);
    try std.testing.expectEqual(@as(u8, 0), full.rolled_back_rows);
    try std.testing.expect(full.bonus_added);

    const correction = try resolveGreedyVerifyChoices(base, &.{ 21, 30 });
    try std.testing.expectEqualSlices(i64, &.{21}, correction.accepted());
    try std.testing.expect(correction.correction_added);
    try std.testing.expectEqual(@as(u8, 1), correction.rolled_back_rows);

    var budget_one = base;
    budget_one.accept_bonus = false;
    const bounded = try resolveGreedyVerifyChoices(budget_one, &.{ 20, 30 });
    try std.testing.expectEqualSlices(i64, &.{20}, bounded.accepted());
    try std.testing.expect(!bounded.bonus_added);

    const eos_bonus = try resolveGreedyVerifyChoices(base, &.{ 20, 2 });
    try std.testing.expectEqualSlices(i64, &.{20}, eos_bonus.accepted());
    try std.testing.expect(eos_bonus.hit_eos);

    const eos_correction = try resolveGreedyVerifyChoices(base, &.{ 2, 30 });
    try std.testing.expectEqual(@as(usize, 0), eos_correction.accepted().len);
    try std.testing.expect(eos_correction.hit_eos);
    try std.testing.expectEqual(@as(u8, 1), eos_correction.rolled_back_rows);

    var ignore_eos = base;
    ignore_eos.ignore_eos = true;
    const ignored = try resolveGreedyVerifyChoices(ignore_eos, &.{ 20, 2 });
    try std.testing.expectEqualSlices(i64, &.{ 20, 2 }, ignored.accepted());
    try std.testing.expect(!ignored.hit_eos);
}

test "transactional greedy verification rejects token bounds and position overflow before mutation" {
    const valid = model_runtime.GreedyVerifyRequest{
        .pending_token_id = 1,
        .proposal_token_id = 2,
        .position = 7,
    };
    try std.testing.expectEqual(@as(usize, 9), try validateGreedyVerifyRequestForModel(valid, 3));
    var negative = valid;
    negative.pending_token_id = -1;
    try std.testing.expectError(error.InvalidModelOutput, validateGreedyVerifyRequestForModel(negative, 3));
    var out_of_vocab = valid;
    out_of_vocab.proposal_token_id = 3;
    try std.testing.expectError(error.InvalidModelOutput, validateGreedyVerifyRequestForModel(out_of_vocab, 3));
    var overflow = valid;
    overflow.position = std.math.maxInt(usize);
    try std.testing.expectError(error.InvalidDecodePosition, validateGreedyVerifyRequestForModel(overflow, 3));
}

test "runtime pipeline owns request suppressions and rejects unsafe policy replacement" {
    var runtime_ctx: RuntimeContext = undefined;
    runtime_ctx.allocator = std.testing.allocator;
    runtime_ctx.pipelined_decode_armed = false;
    runtime_ctx.pipelined_decode_mode = null;
    runtime_ctx.pipelined_sampling = null;
    runtime_ctx.pipelined_kv_advanced_for = null;
    runtime_ctx.pipelined_suppress_kind = null;
    runtime_ctx.pipelined_suppress_ids = null;
    runtime_ctx.gpt_config = .{ .family = .gemma, .num_hidden_layers = 1 };
    runtime_ctx.gpt_config.suppress_token_ids = @splat(-1);
    runtime_ctx.gpt_config.suppress_token_ids[0] = 7;
    runtime_ctx.gpt_config.suppress_token_ids[1] = 11;
    runtime_ctx.gpt_config.suppress_token_ids_len = 2;
    defer runtime_ctx.clearPipelinedRequestState();

    var caller_ids = [_]i32{ 1, 50, 106 };
    try runtime_ctx.installPipelinedSuppressPolicy(&caller_ids);
    try std.testing.expectEqual(PipelinedSuppressKind.request, runtime_ctx.pipelined_suppress_kind.?);
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());
    caller_ids[1] = 9;
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());

    // Same address and length with changed contents must replace the owned A
    // policy rather than treating pointer identity as a cache key.
    try runtime_ctx.installPipelinedSuppressPolicy(&caller_ids);
    try std.testing.expectEqualSlices(i32, &.{ 1, 9, 106 }, runtime_ctx.pipelinedSuppressSlice());

    // Empty request override restores an owned snapshot of model policy.
    try runtime_ctx.installPipelinedSuppressPolicy(&.{});
    try std.testing.expectEqual(PipelinedSuppressKind.model, runtime_ctx.pipelined_suppress_kind.?);
    try std.testing.expectEqualSlices(i32, &.{ 7, 11 }, runtime_ctx.pipelinedSuppressSlice());

    // Once speculative KV bookkeeping has advanced, a policy change fails
    // before replacing or freeing the authoritative model policy.
    runtime_ctx.pipelined_kv_advanced_for = 42;
    try std.testing.expectError(
        error.PipelinedSuppressPolicyChangedAfterKvAdvance,
        runtime_ctx.installPipelinedSuppressPolicy(&caller_ids),
    );
    try std.testing.expectEqual(PipelinedSuppressKind.model, runtime_ctx.pipelined_suppress_kind.?);
    try std.testing.expectEqualSlices(i32, &.{ 7, 11 }, runtime_ctx.pipelinedSuppressSlice());
    // A committed speculative position cannot decline into the outer
    // synchronous path, which would append KV again at a different position.
    try std.testing.expectError(error.InvalidPagedKvState, runtime_ctx.decodeGreedyPipelined(
        std.testing.allocator,
        .{ .token_id = 0, .position = 43 },
    ));
    try std.testing.expectEqual(@as(?usize, 42), runtime_ctx.pipelined_kv_advanced_for);
    runtime_ctx.pipelined_kv_advanced_for = null;

    // Allocation failure occurs before drain/publication and preserves A.
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    runtime_ctx.allocator = failing.allocator();
    try std.testing.expectError(error.OutOfMemory, runtime_ctx.installPipelinedSuppressPolicy(&caller_ids));
    runtime_ctx.allocator = std.testing.allocator;
    try std.testing.expectEqual(PipelinedSuppressKind.model, runtime_ctx.pipelined_suppress_kind.?);
    try std.testing.expectEqualSlices(i32, &.{ 7, 11 }, runtime_ctx.pipelinedSuppressSlice());
}

test "completed terminal pipeline frame clears continuation and retains owned policy until request reset" {
    var runtime_ctx: RuntimeContext = undefined;
    runtime_ctx.allocator = std.testing.allocator;
    runtime_ctx.pipelined_decode_armed = true;
    runtime_ctx.pipelined_decode_mode = .greedy;
    runtime_ctx.pipelined_sampling = null;
    runtime_ctx.pipelined_next_position = 42;
    runtime_ctx.pipelined_kv_advanced_for = null;
    runtime_ctx.pipelined_suppress_kind = .request;
    runtime_ctx.pipelined_suppress_ids = try std.testing.allocator.dupe(i32, &.{ 1, 50, 106 });

    defer runtime_ctx.clearPipelinedSuppressPolicy();
    runtime_ctx.clearCompletedPipelinedFrame(false);
    try std.testing.expect(!runtime_ctx.pipelined_decode_armed);
    try std.testing.expectEqual(@as(?PipelinedDecodeMode, null), runtime_ctx.pipelined_decode_mode);
    try std.testing.expectEqual(@as(usize, 0), runtime_ctx.pipelined_next_position);
    try std.testing.expectEqual(@as(?usize, null), runtime_ctx.pipelined_kv_advanced_for);
    try std.testing.expectEqual(PipelinedSuppressKind.request, runtime_ctx.pipelined_suppress_kind.?);
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());
}

test "nonterminal completed-frame transition can retain exact owned request policy" {
    var runtime_ctx: RuntimeContext = undefined;
    runtime_ctx.allocator = std.testing.allocator;
    runtime_ctx.pipelined_decode_armed = true;
    runtime_ctx.pipelined_decode_mode = .greedy;
    runtime_ctx.pipelined_sampling = null;
    runtime_ctx.pipelined_next_position = 42;
    runtime_ctx.pipelined_kv_advanced_for = null;
    runtime_ctx.pipelined_suppress_kind = .request;
    runtime_ctx.pipelined_suppress_ids = try std.testing.allocator.dupe(i32, &.{ 1, 50, 106 });
    defer runtime_ctx.clearPipelinedSuppressPolicy();

    runtime_ctx.clearCompletedPipelinedFrame(false);
    try std.testing.expect(!runtime_ctx.pipelined_decode_armed);
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());
}

test "prearmed terminal greedy request awaits exactly once without step or submit" {
    const Probe = struct {
        awaits: usize = 0,
        steps: usize = 0,
        submits: usize = 0,
        cancels: usize = 0,
        fail_await: bool = false,

        fn decode(ctx: *anyopaque, request: *const ops.DecoderRuntimeDecodeRequest) !bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            switch (request.phase) {
                .await_only => {
                    self.awaits += 1;
                    if (self.fail_await) return error.InjectedAwaitFailure;
                    request.output_token_ids[0] = 123;
                    return true;
                },
                .step => self.steps += 1,
                .submit_pending => self.submits += 1,
                .cancel_pending => {
                    self.cancels += 1;
                    return true;
                },
                else => return error.UnexpectedDecodePhase,
            }
            return true;
        }
    };
    var probe = Probe{};
    var vtable: ops.ComputeBackend.VTable = undefined;
    vtable.decoderRuntimeDecode = Probe.decode;

    var runtime_ctx: RuntimeContext = undefined;
    runtime_ctx.allocator = std.testing.allocator;
    runtime_ctx.cb = .{ .ptr = &probe, .vtable = &vtable };
    runtime_ctx.gpt_config = .{ .family = .gemma, .num_hidden_layers = 42 };
    runtime_ctx.kv_metadata.kv_view = null;
    runtime_ctx.pipelined_decode_armed = true;
    runtime_ctx.pipelined_decode_mode = .greedy;
    runtime_ctx.pipelined_sampling = null;
    runtime_ctx.pipelined_next_position = 2047 + 298;
    runtime_ctx.pipelined_kv_advanced_for = null;
    runtime_ctx.pipelined_suppress_kind = .request;
    runtime_ctx.pipelined_suppress_ids = try std.testing.allocator.dupe(i32, &.{ 1, 50, 106 });
    defer runtime_ctx.clearPipelinedSuppressPolicy();

    const token = try runtime_ctx.decodeGreedyPipelined(std.testing.allocator, .{
        .token_id = 17,
        .position = 2047 + 298,
        .suppress_token_ids = &.{ 1, 50, 106 },
        .allow_pipeline_rearm = false,
    });
    try std.testing.expectEqual(@as(?i64, 123), token);
    try std.testing.expectEqual(@as(usize, 1), probe.awaits);
    try std.testing.expectEqual(@as(usize, 0), probe.steps);
    try std.testing.expectEqual(@as(usize, 0), probe.submits);
    try std.testing.expect(!runtime_ctx.pipelined_decode_armed);
    try std.testing.expectEqual(@as(?usize, null), runtime_ctx.pipelined_kv_advanced_for);
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());

    runtime_ctx.pipelined_decode_armed = true;
    runtime_ctx.pipelined_decode_mode = .greedy;
    runtime_ctx.pipelined_next_position = 2047 + 299;
    probe.fail_await = true;
    try std.testing.expectError(error.InjectedAwaitFailure, runtime_ctx.decodeGreedyPipelined(
        std.testing.allocator,
        .{
            .token_id = 123,
            .position = 2047 + 299,
            .suppress_token_ids = &.{ 1, 50, 106 },
            .allow_pipeline_rearm = false,
        },
    ));
    try std.testing.expectEqual(@as(usize, 2), probe.awaits);
    try std.testing.expectEqual(@as(usize, 1), probe.cancels);
    try std.testing.expectEqual(@as(usize, 0), probe.steps);
    try std.testing.expectEqual(@as(usize, 0), probe.submits);
    try std.testing.expect(!runtime_ctx.pipelined_decode_armed);
    try std.testing.expectEqualSlices(i32, &.{ 1, 50, 106 }, runtime_ctx.pipelinedSuppressSlice());
}

fn runtimePrepare(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.PrepareRequest,
) !bool {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    // Every explicit prepare starts a new request contract. Do not purge a
    // valid retained physical cache yet; any prepare failure takes the fenced
    // reset+purge path below.
    runtime_ctx.e2b_mirror_admission_handoff.clear();
    errdefer runtime_ctx.clearE2bMirrorAdmissionAfterError();
    timing_stats.runtime_prepare_calls += 1;
    const started_at = monotonicNowNs();
    defer timing_stats.runtime_prepare_nanos += @intCast(monotonicNowNs() - started_at);

    // Set this before any prepared-runtime work or decode frame can be
    // submitted. A cached runtime is reused across requests, so every prepare
    // explicitly replaces the prior request's policy.
    runtime_ctx.request_disables_pipelined_decode = request.disable_pipelined_decode;
    if (request.disable_pipelined_decode) runtime_ctx.drainPipelinedDecode();

    // Operational rollback must dominate every prepared-runtime branch,
    // including A4B's mapped-layer warmup, because that warmup dispatches GPU
    // work of its own.
    if (!runtime_ctx.decoderRuntimeExecutorEnabled()) return false;

    // Qualified A4B currently executes through the graph fallback, not the
    // prepared dense-family decoder. Treat its explicitly requested bounded
    // page warmup as a complete prepare operation before probing that family.
    if (a4bMappedLayer0PrewarmEnabled(runtime_ctx.gpt_config)) {
        try prewarmA4bMappedLayer0(&runtime_ctx.cb, runtime_ctx.gpt_config);
        return true;
    }
    if (gemma4_runtime.supportsPreparedA4bRuntimeConfig(runtime_ctx.gpt_config)) return false;

    const configured_layer_count = runtime_ctx.decoderRuntimeConfiguredLayerCount();
    const prepare_started_at = monotonicNowNs();
    const prepare = try runtime_ctx.cb.decoderRuntimePrepareOrReuseFamily(
        allocator,
        runtime_ctx.gpt_config,
        request.kv_tokens_hint,
        configured_layer_count,
    );
    if (prepare.fast_hit) {
        timing_stats.runtime_prepare_fast_hits += 1;
    } else if (prepare.used_greedy) {
        timing_stats.runtime_prepare_greedy_nanos += @intCast(monotonicNowNs() - prepare_started_at);
    } else {
        timing_stats.runtime_prepare_family_nanos += @intCast(monotonicNowNs() - prepare_started_at);
    }
    if (prepare.prepared) {
        const embedding_prepared = blk: {
            prewarmEmbeddingWeight(&runtime_ctx.cb, runtime_ctx.gpt_config) catch break :blk false;
            break :blk true;
        };
        try prewarmA4bMappedLayer0(&runtime_ctx.cb, runtime_ctx.gpt_config);
        const prefill_plan_prepared = decoder_gated_runtime.preplanPrefillFrame(
            &runtime_ctx.cb,
            allocator,
            runtime_ctx.gpt_config,
            configured_layer_count,
            request.kv_tokens_hint,
        ) catch false;
        const ple_embedding_prepared = decoder_gated_runtime.prewarmPleTokenEmbedding(
            &runtime_ctx.cb,
            runtime_ctx.gpt_config,
        ) catch |err| blk: {
            if (runtime_ctx.cb.boundedPleQ6Active() or
                (envFlag("TERMITE_METAL_ENABLE_BOUNDED_PLE_Q6_ROWS") and
                    !envFlag("TERMITE_METAL_DISABLE_BOUNDED_PLE_Q6_ROWS"))) return err;
            break :blk false;
        };
        var finalized_admission = request.e2b_f16_mirror_admission;
        if (finalized_admission) |*value| {
            value.nonlinear_weights_complete = e2bMirrorNonlinearBoundarySatisfied(
                qualifiedE2bMirrorWholeModelConfig(runtime_ctx.gpt_config),
                prepare.prepared,
                embedding_prepared,
                prefill_plan_prepared,
                ple_embedding_prepared,
            );
        }
        if (envFlag("ANTFLY_TRACE_E2B_F16_MIRROR_PREPARE")) std.debug.print(
            "metal_e2b_mirror_boundary: qualified_config={} family={} embedding={} preplan={} ple={} admission={} nonlinear_complete={}\n",
            .{
                qualifiedE2bMirrorWholeModelConfig(runtime_ctx.gpt_config),
                prepare.prepared,
                embedding_prepared,
                prefill_plan_prepared,
                ple_embedding_prepared,
                finalized_admission != null,
                if (finalized_admission) |value| value.nonlinear_weights_complete else false,
            },
        );
        // Carry typed request facts across notePrefill/appendPrefillChunk's
        // backend reset. Physical materialization and per-request budget
        // revalidation occur after that reset and before the first GPU frame.
        runtime_ctx.e2b_mirror_admission_handoff.stage(finalized_admission);
        if (finalized_admission == null)
            _ = try runtime_ctx.cb.decoderRuntimeFinalizeE2bF16Mirrors(null);
    } else {
        // Null/failed requests explicitly disarm a cache admitted by an older request.
        runtime_ctx.e2b_mirror_admission_handoff.clear();
        _ = try runtime_ctx.cb.decoderRuntimeFinalizeE2bF16Mirrors(null);
    }
    return prepare.prepared;
}

fn runtimeReset(ctx: *anyopaque) !void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    runtime_ctx.clearPipelinedRequestState();
    runtime_ctx.e2b_mirror_admission_handoff.clear();
    // Internal reset revokes span ownership without destroying current-request
    // phase state. Public reset then performs the explicit fenced request
    // cleanup before a new typed prepare can establish authority.
    try runtime_ctx.resetState();
    try cleanupBoundedPleRequestAtPublicReset(&runtime_ctx.cb);
}

fn runtimeFinishPrefillForDecode(ctx: *anyopaque, request: model_runtime.PrepareRequest) !bool {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    errdefer runtime_ctx.clearE2bMirrorAdmissionAfterError();
    try request.check();
    const admission = request.e2b_f16_mirror_admission orelse {
        runtime_ctx.clearE2bMirrorAdmissionAfterError();
        return false;
    };
    const finished = try runtime_ctx.cb.decoderRuntimeFinishBoundedPlePrefill(&admission);
    if (!finished) {
        runtime_ctx.clearE2bMirrorAdmissionAfterError();
        return false;
    }
    try request.check();
    return true;
}

fn runtimeExactPagedKvAuthority(ctx: *anyopaque) !model_runtime.ExactPagedKvSourceAuthority {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const identity = runtime_ctx.model_identity orelse return error.UnsupportedKvIdentity;
    if (runtime_ctx.currentTokenCount() == 0 or runtime_ctx.currentTokenCount() != runtime_ctx.handoff_tokens.items.len)
        return error.InvalidExactPagedKvHandoffState;
    const context_layout_sha256 = try runtime.context_policy.layoutDigest(try runtime.context_policy.Layout.init(runtime_ctx.gpt_config));
    return .{
        .model_weights_sha256 = identity.weights,
        .model_identity_layout_sha256 = identity.layout,
        .context_layout_sha256 = context_layout_sha256,
        .token_stream_sha256 = model_runtime.exactPagedKvTokenStreamDigest(runtime_ctx.handoff_tokens.items),
        .source_revision = runtime_ctx.handoff_revision,
        .source_sequence = (runtime_ctx.kv_metadata.kv_view orelse return error.InvalidExactPagedKvHandoffState).sequence_id,
        .logical_tokens = runtime_ctx.currentTokenCount(),
    };
}

fn exactPagedAuthorityMatchesExport(observed: model_runtime.ExactPagedKvSourceAuthority, expected: runtime.kv.storage_runtime.ExactLayerRangeSourceAuthority) bool {
    return std.mem.eql(u8, &observed.model_weights_sha256, &expected.model_weights_sha256) and
        std.mem.eql(u8, &observed.model_identity_layout_sha256, &expected.model_identity_layout_sha256) and
        std.mem.eql(u8, &observed.context_layout_sha256, &expected.context_layout_sha256) and
        std.mem.eql(u8, &observed.token_stream_sha256, &expected.token_stream_sha256) and
        observed.source_revision == expected.source_revision and observed.source_sequence == expected.source_sequence and
        observed.logical_tokens == expected.logical_tokens;
}

test "compiled range export authority binds every source field" {
    const digest: [32]u8 = @splat(4);
    const observed = model_runtime.ExactPagedKvSourceAuthority{ .model_weights_sha256 = digest, .model_identity_layout_sha256 = digest, .context_layout_sha256 = digest, .token_stream_sha256 = digest, .source_revision = 2, .source_sequence = 7, .logical_tokens = 8 };
    var expected = runtime.kv.storage_runtime.ExactLayerRangeSourceAuthority{ .model_weights_sha256 = digest, .model_identity_layout_sha256 = digest, .context_layout_sha256 = digest, .token_stream_sha256 = digest, .source_revision = 2, .source_sequence = 7, .logical_tokens = 8 };
    try std.testing.expect(exactPagedAuthorityMatchesExport(observed, expected));
    expected.source_revision += 1;
    try std.testing.expect(!exactPagedAuthorityMatchesExport(observed, expected));
    expected.source_revision -= 1;
    expected.token_stream_sha256[0] ^= 1;
    try std.testing.expect(!exactPagedAuthorityMatchesExport(observed, expected));
    expected.token_stream_sha256[0] ^= 1;
    expected.model_weights_sha256[0] ^= 1;
    try std.testing.expect(!exactPagedAuthorityMatchesExport(observed, expected));
}

fn runtimeExportExactPagedKvRanges(ctx: *anyopaque, request: model_runtime.ExactPagedKvRangeExportRequest) !runtime.kv.storage_runtime.ExactLayerRangeExportResult {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.validate();
    const before = try runtimeExactPagedKvAuthority(ctx);
    if (!exactPagedAuthorityMatchesExport(before, request.authority)) return error.InvalidExactPagedKvRangeExportAuthority;
    const source_hook = runtime_ctx.kv_storage.device_write_hook orelse return error.UnsupportedExactPagedKvRangeExport;
    const result = try source_hook.exportExactLayerRanges(.{
        .source_sequence = request.authority.source_sequence,
        .expected_written_tokens = request.authority.logical_tokens,
        .expected_position_offset = 0,
        .ranges = request.ranges,
        .max_bytes = request.max_copy_bytes,
        .check_ctx = request.check_ctx,
        .check_fn = request.check_fn,
    });
    try request.check();
    const after = try runtimeExactPagedKvAuthority(ctx);
    if (!exactPagedAuthorityMatchesExport(after, request.authority)) return error.ExactPagedKvRangeExportSourceChanged;
    return result;
}

fn validateObservedFullHistoryExact(full: bool, source_ring: bool, destination_ring: bool, capacity: usize, logical: usize) !void {
    if (full and (source_ring or destination_ring or capacity < logical)) return error.InvalidExactPagedKvHandoffState;
}

test "compiled handoff observes both full-history views before clone" {
    try validateObservedFullHistoryExact(true, false, false, 32512, 32512);
    try std.testing.expectError(error.InvalidExactPagedKvHandoffState, validateObservedFullHistoryExact(true, true, false, 32768, 32512));
    try std.testing.expectError(error.InvalidExactPagedKvHandoffState, validateObservedFullHistoryExact(true, false, true, 32768, 32512));
    try std.testing.expectError(error.InvalidExactPagedKvHandoffState, validateObservedFullHistoryExact(true, false, false, 32511, 32512));
    try validateObservedFullHistoryExact(false, true, true, 1, 32512);
}

fn runtimeHandoffExactPagedKv(ctx: *anyopaque, request: model_runtime.ExactPagedKvHandoffRequest) !model_runtime.ExactPagedKvHandoffReceipt {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.validate();
    if (runtime_ctx.gpt_config.family != .gemma or !runtime_ctx.gpt_config.usesGemma4Channels() or
        !runtime_ctx.gpt_config.supportsSplitSwaGlobalKvRing()) return error.UnsupportedExactPagedKvHandoff;
    const layout_sha256 = (runtime_ctx.model_identity orelse return error.UnsupportedKvIdentity).layout;
    const token_stream_sha256 = model_runtime.exactPagedKvTokenStreamDigest(runtime_ctx.handoff_tokens.items);
    if (!std.mem.eql(u8, &layout_sha256, &request.model_layout_sha256) or !std.mem.eql(u8, &token_stream_sha256, &request.token_stream_sha256) or
        runtime_ctx.handoff_revision != request.source_revision or runtime_ctx.currentTokenCount() != request.logical_tokens or
        runtime_ctx.handoff_tokens.items.len != request.logical_tokens) return error.InvalidExactPagedKvHandoffAuthority;
    const source_view = runtime_ctx.kv_metadata.kv_view orelse return error.InvalidExactPagedKvHandoffState;
    if (source_view.position_offset != 0 or source_view.token_count != request.logical_tokens) return error.InvalidExactPagedKvHandoffState;
    try validateObservedFullHistoryExact(request.full_history_exact, source_view.allow_swa_ring, request.destination.allow_swa_ring, request.capacity_tokens, request.logical_tokens);
    const source_blocks = source_view.logical_blocks orelse return error.InvalidExactPagedKvHandoffState;
    const destination_storage = request.destination.kv_storage orelse return error.InvalidExactPagedKvHandoff;
    const destination_blocks = request.destination.logical_blocks orelse return error.InvalidExactPagedKvHandoff;
    if (destination_storage.tokenCount(request.destination.sequence_id) != request.logical_tokens or request.destination.position_offset != 0)
        return error.InvalidExactPagedKvHandoffDestination;
    const source_hook = runtime_ctx.kv_storage.device_write_hook orelse return error.UnsupportedExactPagedKvHandoff;
    const destination_hook = destination_storage.device_write_hook orelse return error.UnsupportedExactPagedKvHandoff;
    // The destination is unpublished and request-owned. Any failure revokes
    // backend cache state; the caller's state/owner defer releases its blocks.
    errdefer destination_hook.releaseSequence(request.destination.sequence_id);
    const device_clone = @import("../runtime/kv/device_clone.zig");
    var donors: [256]device_clone.Donor = undefined;
    for (request.donors, 0..) |donor, i| donors[i] = .{ .layer = donor.layer, .heads = donor.heads, .dim = donor.dim, .capacity_rows = donor.capacity_rows, .host_staged_ring = donor.host_staged_ring };
    try request.check();
    try runtime_ctx.cb.decoderRuntimeFlushActiveFrame();
    try runtime_ctx.cb.decoderRuntimeWaitSubmittedFrame();
    const revision_before = runtime_ctx.handoff_revision;
    const result = try destination_hook.cloneExactPrefix(source_hook, .{
        .source_sequence = source_view.sequence_id,
        .destination_sequence = request.destination.sequence_id,
        .tokens = request.logical_tokens,
        .capacity = request.capacity_tokens,
        .page_size = request.page_size_tokens,
        .source_blocks = source_blocks,
        .destination_blocks = destination_blocks,
        .logical_block_policy = .metal_sequence_slot_owned,
        .donors = donors[0..request.donors.len],
        .max_bytes = request.max_copy_bytes,
        .max_host_staging_bytes = request.max_host_staging_bytes,
        .check_ctx = @ptrCast(@constCast(&request)),
        .check_fn = struct {
            fn check(raw: ?*anyopaque) anyerror!void {
                const typed: *const model_runtime.ExactPagedKvHandoffRequest = @ptrCast(@alignCast(raw orelse return error.InvalidExactPagedKvHandoff));
                try typed.check();
            }
        }.check,
    });
    try request.check();
    if (runtime_ctx.handoff_revision != revision_before or runtime_ctx.currentTokenCount() != request.logical_tokens)
        return error.ExactPagedKvHandoffSourceChanged;
    return .{
        .authority_sha256 = request.authority_sha256,
        .model_layout_sha256 = request.model_layout_sha256,
        .token_stream_sha256 = request.token_stream_sha256,
        .source_revision = request.source_revision,
        .destination_owner_id = request.destination_owner_id,
        .destination_storage_address = @intFromPtr(destination_storage),
        .logical_tokens = request.logical_tokens,
        .donors = result.donors,
        .allocation_bound_bytes = result.allocation_bound_bytes,
        .copied_bytes = result.copied_bytes,
        .peak_host_staging_bytes = result.peak_host_staging_bytes,
        .source_unchanged = true,
        .destination_complete = true,
        .fenced = true,
    };
}

fn runtimeDiscardExactPagedKvSource(ctx: *anyopaque, request: model_runtime.ExactPagedKvSourceDiscardRequest) !model_runtime.ExactPagedKvSourceDiscardReceipt {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.check();
    const source_view = runtime_ctx.kv_metadata.kv_view orelse return error.StaleExactPagedKvSource;
    const token_stream_sha256 = model_runtime.exactPagedKvTokenStreamDigest(runtime_ctx.handoff_tokens.items);
    if (runtime_ctx.handoff_revision != request.source_revision or source_view.sequence_id != request.source_sequence or
        runtime_ctx.handoff_tokens.items.len != request.logical_tokens or runtime_ctx.currentTokenCount() != request.expected_materialized_tokens or
        !std.mem.eql(u8, &token_stream_sha256, &request.token_stream_sha256))
        return error.StaleExactPagedKvSource;
    try runtime_ctx.cb.decoderRuntimeFlushActiveFrame();
    try runtime_ctx.cb.decoderRuntimeWaitSubmittedFrame();
    const sequence_id = source_view.sequence_id;
    const hook = runtime_ctx.kv_storage.device_write_hook orelse return error.UnsupportedExactPagedKvHandoff;
    const discarded = try hook.discardExactCloneSource(sequence_id);
    try runtime_ctx.resetState();
    return .{
        .authority_sha256 = request.authority_sha256,
        .source_revision = request.source_revision,
        .source_sequence = sequence_id,
        .slots = discarded.slots,
        .live_bytes_before = discarded.live_bytes_before,
        .live_bytes_after = discarded.live_bytes_after,
        .physically_released_bytes = discarded.physically_released_bytes,
    };
}

fn runtimeDebugTimingStats(ctx: *anyopaque) model_runtime.RuntimeDebugTimingStats {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    var stats = getTimingStats();
    stats.backend = runtime_ctx.cb.debugTimingSnapshot();
    stats.decoder_runtime_ready = runtime_ctx.cb.decoderRuntimeReady();
    stats.decoder_runtime_absolute_embeddings_prepared = runtime_ctx.cb.decoderRuntimeAbsoluteEmbeddingsPrepared();
    return stats;
}

fn runtimeResetDebugTimingStats(ctx: *anyopaque) void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    resetTimingStats();
    runtime_ctx.cb.resetDebugTimingStats();
    decoder_rms_runtime.resetTimingStats();
    decoder_gated_runtime.resetTimingStats();
    decoder_bitnet_runtime.resetTimingStats();
    decoder_tail_runtime.resetTimingStats();
}

fn runtimePrintDebugTiming(ctx: *anyopaque) void {
    printRuntimeDebugTimingStats(runtimeDebugTimingStats(ctx), true);
}

fn runtimeDeinit(ctx: *anyopaque) void {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    runtime_ctx.deinit();
}

fn runtimePrefill(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.PrefillRequest,
) !model_runtime.ModelOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    const recent_capture_terminal = if (request.recent_query_capture) |capture|
        try capture.validateChunk(request.seq_len, request.query_seq_len)
    else
        false;
    var recent_capture_completed = false;
    errdefer if (recent_capture_terminal and !recent_capture_completed) if (request.recent_query_capture) |capture| {
        capture.complete_fn(capture.context, .{
            .authority_epoch = capture.authority_epoch,
            .submitted = false,
            .completed = false,
            .cancelled = true,
            .frame_submission_epoch = 0,
            .completed_frame_watermark = 0,
        }) catch {};
    };
    errdefer runtime_ctx.clearE2bMirrorAdmissionAfterError();
    try request.check();
    runtime_ctx.clearPipelinedRequestState();
    if (request.input_ids.len == 0 or request.query_seq_len == 0) return error.EmptyPrompt;
    if (request.input_ids.len != request.query_seq_len) return error.UnsupportedShape;
    if (request.query_seq_len > request.seq_len) return error.UnsupportedShape;
    try runtime_ctx.configureKvPolicy(request.max_inflight_tokens, request.allow_swa_ring);

    timing_stats.prefill_calls += 1;
    const prepare_started_at = monotonicNowNs();
    const prior_token_count = runtime_ctx.currentTokenCount();
    if (runtime_ctx.handoff_tokens.items.len != prior_token_count)
        return error.InvalidExactPagedKvHandoffState;
    const prior_handoff_len = runtime_ctx.handoff_tokens.items.len;
    const prior_handoff_revision = runtime_ctx.handoff_revision;
    try runtime_ctx.handoff_tokens.appendSlice(runtime_ctx.allocator, request.input_ids);
    runtime_ctx.handoff_revision +%= 1;
    if (runtime_ctx.handoff_revision == 0) runtime_ctx.handoff_revision = 1;
    errdefer {
        runtime_ctx.handoff_tokens.shrinkRetainingCapacity(prior_handoff_len);
        runtime_ctx.handoff_revision = prior_handoff_revision;
    }
    const decode_context = try runtime_ctx.preparePrefill(
        request.seq_len,
        request.query_seq_len,
        request.attention_mode,
        request.recent_query_capture,
    );
    try request.check();
    timing_stats.prefill_prepare_nanos += @intCast(monotonicNowNs() - prepare_started_at);
    // The qualified A4B fast path owns qLen=1 decode only. Let canonical
    // prefill establish its dynamic packed-weight slots first, then install
    // the fixed decode plan on the first generated token. Preparing in the
    // opposite order leaves too little shared slot-directory capacity for
    // the canonical multi-row MoE prefill.
    const defer_a4b_decode_prepare = gemma4_runtime.supportsPreparedA4bRuntimeConfig(runtime_ctx.gpt_config);
    const decoder_runtime_ready = if (request.force_host_logits)
        false
    else if (defer_a4b_decode_prepare)
        runtime_ctx.decoderRuntimeExecutorEnabled()
    else if (decoderRuntimePrefillAfterPrepareRequested())
        runtime_ctx.decoderRuntimeExecutorEnabled() and (try runtime_ctx.ensureDecoderRuntimePrepared())
    else blk: {
        _ = runtime_ctx.ensureDecoderRuntimePrepared() catch false;
        break :blk runtime_ctx.decoderRuntimeExecutorEnabled();
    };
    if (request.recent_query_capture != null and !decoder_runtime_ready)
        return error.RecentPrefillQueryCaptureUnavailable;
    if (runtime_ctx.e2b_mirror_admission_handoff.forPrefill(prior_token_count, request.seq_len, request.query_seq_len)) |admission| {
        var request_admission = admission;
        if (!(try runtime_ctx.cb.decoderRuntimeFinalizeE2bF16Mirrors(&request_admission)))
            runtime_ctx.e2b_mirror_admission_handoff.clear();
    }
    if (decoderRuntimePrefillTraceRequested()) {
        std.debug.print("prefill-trace: runtimePrefill qlen={d} seq={d} ready={}\n", .{ request.query_seq_len, request.seq_len, decoder_runtime_ready });
    }
    if (decoder_runtime_ready and request.query_seq_len == 1) {
        try request.check();
        const token_id = request.input_ids[request.input_ids.len - 1];
        const direct_started_at = monotonicNowNs();
        if (try runtime_ctx.forwardDecoderRuntimeLastLogits(allocator, token_id, request.seq_len, &decode_context)) |output| {
            timing_stats.prefill_direct_last_logits_nanos += @intCast(monotonicNowNs() - direct_started_at);
            return output;
        }
        timing_stats.prefill_direct_last_logits_nanos += @intCast(monotonicNowNs() - direct_started_at);
    }
    if (decoder_runtime_ready) {
        try request.check();
        if (decoderRuntimePrefillTraceRequested()) {
            std.debug.print("prefill-trace: runtimePrefill entering direct family path\n", .{});
        }
        const configured_layer_count = runtime_ctx.decoderRuntimeConfiguredLayerCount();
        const timing_before = runtime_ctx.cb.directFamilyTimingSnapshot();
        const direct_started_at = monotonicNowNs();
        const maybe_tail: ?metal_runtime.PrefillPreparedTail = if (request.suppress_token_ids.len == 0)
            try metal_runtime.forwardPrefillLastPreparedTailFamily(
                &runtime_ctx.cb,
                allocator,
                runtime_ctx.gpt_config,
                configured_layer_count,
                request.input_ids,
                request.seq_len,
                &decode_context,
                prefillMayCacheGreedyToken(request.prefer_greedy_token, request.suppress_token_ids),
            )
        else if (try decoder_gated_runtime.forwardPrefillLastPreparedTailSuppress(
            &runtime_ctx.cb,
            allocator,
            runtime_ctx.gpt_config,
            configured_layer_count,
            request.input_ids,
            request.seq_len,
            &decode_context,
            prefillMayCacheGreedyToken(request.prefer_greedy_token, request.suppress_token_ids),
            request.suppress_token_ids,
        )) |tail| .{
            .final_hidden = tail.final_hidden,
            .final_norm_slot = tail.final_norm_slot,
            .final_lm_head_slot = tail.final_lm_head_slot,
            .hidden_size = tail.hidden_size,
            .vocab_size = tail.vocab_size,
            .norm_eps = tail.norm_eps,
            .greedy_token_id = tail.greedy_token_id,
        } else null;
        if (maybe_tail) |tail| {
            // The prepared tail is not transferred into ModelOutput until the
            // capture completion certificate has been validated. Release its
            // backend tensor if completion or any later bookkeeping fails.
            errdefer runtime_ctx.cb.free(tail.final_hidden);
            const timing_after = runtime_ctx.cb.directFamilyTimingSnapshot();
            const delta = timing_after.delta(timing_before);
            timing_stats.prefill_direct_family_nanos += @intCast(monotonicNowNs() - direct_started_at);
            timing_stats.prefill_direct_family_project_nanos += delta.project_nanos;
            timing_stats.prefill_direct_family_span_prep_nanos += delta.span_prep_nanos;
            timing_stats.prefill_direct_family_quant_attn_nanos += delta.quant_attn_nanos;
            timing_stats.prefill_direct_family_block_apply_nanos += delta.block_apply_nanos;
            timing_stats.prefill_direct_family_frame_wait_nanos += delta.frame_wait_nanos;
            timing_stats.prefill_direct_family_frame_gpu_nanos += delta.frame_gpu_nanos;
            runtime_ctx.noteDecoderRuntimeStateFromCurrentView();
            if (recent_capture_terminal) if (request.recent_query_capture) |capture| {
                const copied_nonce = capture.lastEncodedFrame();
                const completed_nonce = runtime_ctx.cb.decoderRuntimeLastCompletedRecentQueryCaptureFrameNonce(capture.context);
                std.debug.print(
                    "recent-prefill-capture terminal seq={d} rows={d} copied_nonce={d} completed_capture_nonce={d}\n",
                    .{ request.seq_len, request.query_seq_len, copied_nonce, completed_nonce },
                );
                if (copied_nonce == 0 or completed_nonce != copied_nonce) return error.IncompleteRecentQueryCapture;
                try capture.complete_fn(capture.context, .{
                    .authority_epoch = capture.authority_epoch,
                    .submitted = true,
                    .completed = true,
                    .cancelled = false,
                    .frame_submission_epoch = copied_nonce,
                    .completed_frame_watermark = completed_nonce,
                });
                recent_capture_completed = true;
            };
            return .{ .prepared_tail = .{
                .final_hidden = tail.final_hidden,
                .backend = runtime_ctx.cb,
                .norm = .rms,
                .norm_slot = tail.final_norm_slot,
                .lm_head_slot = tail.final_lm_head_slot,
                .hidden_size = tail.hidden_size,
                .vocab_size = tail.vocab_size,
                .eps = tail.norm_eps,
                .final_logit_softcap = runtime_ctx.gpt_config.final_logit_softcapping,
                .use_transformed_lm_head = decoder_tail_runtime.repackQualityRawLogitsEnabled(),
                .greedy_token_id = tail.greedy_token_id,
            } };
        }
        if (recent_capture_terminal) return error.RecentPrefillQueryCaptureUnavailable;
        const timing_after = runtime_ctx.cb.directFamilyTimingSnapshot();
        const delta = timing_after.delta(timing_before);
        timing_stats.prefill_direct_family_nanos += @intCast(monotonicNowNs() - direct_started_at);
        timing_stats.prefill_direct_family_project_nanos += delta.project_nanos;
        timing_stats.prefill_direct_family_span_prep_nanos += delta.span_prep_nanos;
        timing_stats.prefill_direct_family_quant_attn_nanos += delta.quant_attn_nanos;
        timing_stats.prefill_direct_family_block_apply_nanos += delta.block_apply_nanos;
    }
    try request.check();
    const fallback_started_at = monotonicNowNs();
    const logits = try forwardLastLogits(
        runtime_ctx,
        allocator,
        request.input_ids,
        request.seq_len,
        request.query_seq_len,
        &decode_context,
    );
    timing_stats.prefill_fallback_logits_nanos += @intCast(monotonicNowNs() - fallback_started_at);
    return .{
        .logits = logits,
    };
}

fn runtimeDecode(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.DecodeRequest,
) !model_runtime.ModelOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.check();
    runtime_ctx.drainPipelinedDecode();
    runtime_ctx.clearGreedyDeviceToken();
    const step = try runtime_ctx.beginDecodeStep(request.position, request.attention_mode);
    try request.check();

    if (runtime_ctx.decoderRuntimeExecutorEnabled()) {
        if (try runtime_ctx.forwardDecoderRuntimeBackendOwnedFullLogits(allocator, request.token_id, step.seq_len, &step.decode_context)) |output| {
            return output;
        }
        if (try runtime_ctx.forwardDecoderRuntimeLastLogits(allocator, request.token_id, step.seq_len, &step.decode_context)) |output| {
            return output;
        }
    }

    const input_ids = [_]i64{request.token_id};
    return .{
        .logits = try forwardLastLogits(runtime_ctx, allocator, input_ids[0..], step.seq_len, 1, &step.decode_context),
    };
}

fn modelOutputFromOwnedHostLogits(host: []f32) model_runtime.ModelOutput {
    return .{ .logits = host };
}

test "prepared full logits transfer request allocator ownership to model output" {
    const host = try std.testing.allocator.alloc(f32, 150);
    var output = modelOutputFromOwnedHostLogits(host);
    try std.testing.expect(output.logits.?.ptr == host.ptr);
    try std.testing.expectEqual(host.len, output.logits.?.len);
    output.deinit(std.testing.allocator);
}

fn runtimeDecodeSample(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.SampledDecodeRequest,
) !model_runtime.SampledDecodeOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.decode.check();
    if (request.sampling.isPureGreedy()) {
        const greedy = try runtimeDecodeGreedy(ctx, allocator, request.decode);
        return .{ .token_id = greedy.token_id };
    }
    const committed_direct = runtimeSupportsSampleDecodeRequest(ctx, request);

    timing_stats.decode_sample_calls += 1;
    runtime_ctx.clearGreedyDeviceToken();
    if (committed_direct and
        (runtime_ctx.pipelined_decode_armed or pipelinedDecodeFrameEnabled(runtime_ctx.gpt_config)))
    {
        const pipelined_started_at = monotonicNowNs();
        if (try runtime_ctx.decodeSamplePipelined(allocator, request)) |token_id| {
            timing_stats.decode_sample_direct_nanos += @intCast(monotonicNowNs() - pipelined_started_at);
            return .{ .token_id = token_id };
        }
    }
    runtime_ctx.drainPipelinedDecode();
    const begin_started_at = monotonicNowNs();
    const step = try runtime_ctx.beginDecodeStep(request.decode.position, request.decode.attention_mode);
    try request.decode.check();
    timing_stats.decode_begin_step_nanos += @intCast(monotonicNowNs() - begin_started_at);
    if (committed_direct) {
        const direct_started_at = monotonicNowNs();
        const configured_layer_count = runtime_ctx.decoderRuntimeConfiguredLayerCount();
        const token_id = (try decoder_gated_runtime.forwardResidentSampledToken(
            &runtime_ctx.cb,
            allocator,
            runtime_ctx.gpt_config,
            configured_layer_count,
            request.decode.token_id,
            step.seq_len,
            &step.decode_context,
            request.sampling,
            request.token_history,
        )) orelse return error.ResidentSamplingPreflightChanged;
        runtime_ctx.noteDecoderRuntimeStateFromCurrentView();
        timing_stats.decode_sample_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
        return .{ .token_id = token_id };
    }
    if (runtime_ctx.decoderRuntimeExecutorEnabled()) {
        const direct_started_at = monotonicNowNs();
        if (try runtime_ctx.forwardDecoderRuntimeSampledToken(
            allocator,
            request.decode.token_id,
            step.seq_len,
            &step.decode_context,
            request.sampling,
            request.token_history,
        )) |token_id| {
            timing_stats.decode_sample_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
            return .{ .token_id = token_id };
        }
        timing_stats.decode_sample_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
    }

    var output: model_runtime.ModelOutput = undefined;
    const fallback_started_at = monotonicNowNs();
    if (runtime_ctx.decoderRuntimeExecutorEnabled()) {
        if (try runtime_ctx.forwardDecoderRuntimeLastLogits(allocator, request.decode.token_id, step.seq_len, &step.decode_context)) |raw_output| {
            output = raw_output;
        } else {
            const input_ids = [_]i64{request.decode.token_id};
            output = .{
                .logits = try forwardLastLogits(runtime_ctx, allocator, input_ids[0..], step.seq_len, 1, &step.decode_context),
            };
        }
    } else {
        const input_ids = [_]i64{request.decode.token_id};
        output = .{
            .logits = try forwardLastLogits(runtime_ctx, allocator, input_ids[0..], step.seq_len, 1, &step.decode_context),
        };
    }
    timing_stats.decode_sample_fallback_nanos += @intCast(monotonicNowNs() - fallback_started_at);
    defer output.deinit(allocator);
    if (output.final_logit_softcap <= 0.0) {
        if (output.device_logits) |device_logits| {
            const backend = output.device_logits_backend orelse return error.InvalidModelOutput;
            if (try backend.sampleLastRow(&.{
                .tensor = device_logits,
                .rows = 1,
                .dim = runtime_ctx.gpt_config.vocab_size,
                .temperature = request.sampling.temperature,
                .top_k = @intCast(@max(request.sampling.top_k, 0)),
                .top_p = request.sampling.top_p,
                .min_p = request.sampling.min_p,
                .repetition_penalty = request.sampling.repetition_penalty,
                .frequency_penalty = request.sampling.frequency_penalty,
                .presence_penalty = request.sampling.presence_penalty,
                .token_history = request.token_history,
            })) |token_id| {
                return .{ .token_id = token_id };
            }
        }
    }
    return .{
        .token_id = @intCast(model_runtime.sampleTokenFromLogits(
            allocator,
            try output.hostLogits(allocator),
            request.sampling,
            request.token_history,
        )),
    };
}

fn runtimeSupportsSampleDecodeRequest(
    ctx: *anyopaque,
    request: model_runtime.SampledDecodeRequest,
) bool {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    // Once a sampled frame is submitted, the caller must enter decodeSample
    // so continuation mismatches become fatal errors. Returning false here
    // would let the graph dispatcher replay the same KV position elsewhere.
    if (runtime_ctx.pipelined_decode_armed) return true;
    if (request.decode.attention_mode != .paged_decode) return false;
    if (!runtime_ctx.decoderRuntimeExecutorEnabled()) return false;
    return switch (runtime_ctx.gpt_config.family) {
        .llama, .mistral, .qwen2, .gemma => decoder_gated_runtime.residentSampledTokenSupported(
            &runtime_ctx.cb,
            runtime_ctx.gpt_config,
            runtime_ctx.decoderRuntimeConfiguredLayerCount(),
            request.sampling,
            request.token_history,
        ),
        else => false,
    };
}

fn runtimeDecodeGreedy(
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    request: model_runtime.DecodeRequest,
) !model_runtime.GreedyDecodeOutput {
    const runtime_ctx: *RuntimeContext = @ptrCast(@alignCast(ctx));
    try request.check();
    timing_stats.decode_greedy_calls += 1;
    if (request.suppress_token_ids.len != 0) runtime_ctx.clearGreedyDeviceToken();
    if (!runtime_ctx.request_disables_pipelined_decode and
        pipelinedDecodeFrameEnabled(runtime_ctx.gpt_config))
    {
        const pipelined_started_at = monotonicNowNs();
        if (try runtime_ctx.decodeGreedyPipelined(allocator, request)) |token_id| {
            timing_stats.decode_greedy_direct_nanos += @intCast(monotonicNowNs() - pipelined_started_at);
            return .{ .token_id = token_id };
        }
    }
    const begin_started_at = monotonicNowNs();
    const step = try runtime_ctx.beginDecodeStep(request.position, request.attention_mode);
    try request.check();
    timing_stats.decode_begin_step_nanos += @intCast(monotonicNowNs() - begin_started_at);

    const direct_started_at = monotonicNowNs();
    if (request.suppress_token_ids.len == 0 and (request.input_token_tensor != null or runtime_ctx.greedyDeviceTokenHandoffEnabled())) {
        if (try runtime_ctx.forwardGreedyDeviceTokenHandoff(allocator, request, step.seq_len, &step.decode_context)) |token_id| {
            timing_stats.decode_greedy_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
            return .{ .token_id = token_id };
        }
    }
    if (try runtime_ctx.forwardDecoderRuntimeGreedyToken(allocator, request.token_id, step.seq_len, &step.decode_context, request.suppress_token_ids)) |token_id| {
        if (request.suppress_token_ids.len == 0) try runtime_ctx.seedGreedyDeviceToken(token_id);
        timing_stats.decode_greedy_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
        return .{ .token_id = token_id };
    }
    if (try runtime_ctx.forwardDecoderRuntimeLastLogits(allocator, request.token_id, step.seq_len, &step.decode_context)) |output| {
        var owned_output = output;
        defer owned_output.deinit(allocator);
        const token_id = try owned_output.greedyTokenSuppress(allocator, runtime_ctx.gpt_config.vocab_size, request.suppress_token_ids);
        if (request.suppress_token_ids.len == 0) try runtime_ctx.seedGreedyDeviceToken(token_id);
        timing_stats.decode_greedy_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);
        return .{ .token_id = token_id };
    }
    timing_stats.decode_greedy_direct_nanos += @intCast(monotonicNowNs() - direct_started_at);

    const input_ids = [_]i64{request.token_id};
    const fallback_started_at = monotonicNowNs();
    const token_id: i64 = if (request.suppress_token_ids.len == 0)
        @intCast(try gpt_arch.forwardGreedyLastToken(
            &runtime_ctx.cb,
            allocator,
            runtime_ctx.gpt_config,
            input_ids[0..],
            1,
            step.seq_len,
            &step.decode_context,
        ))
    else blk: {
        var output: model_runtime.ModelOutput = .{
            .logits = try forwardLastLogits(runtime_ctx, allocator, input_ids[0..], step.seq_len, 1, &step.decode_context),
        };
        defer output.deinit(allocator);
        break :blk try output.greedyTokenSuppress(allocator, runtime_ctx.gpt_config.vocab_size, request.suppress_token_ids);
    };
    timing_stats.decode_greedy_fallback_nanos += @intCast(monotonicNowNs() - fallback_started_at);
    if (request.suppress_token_ids.len == 0) try runtime_ctx.seedGreedyDeviceToken(@intCast(token_id));
    return .{
        .token_id = @intCast(token_id),
    };
}

fn prefillMayCacheGreedyToken(prefer_greedy_token: bool, suppress_token_ids: []const i32) bool {
    return prefer_greedy_token and suppress_token_ids.len == 0;
}

fn forwardLastLogits(
    runtime_ctx: *RuntimeContext,
    allocator: std.mem.Allocator,
    input_ids: []const i64,
    seq_len: usize,
    query_seq_len: usize,
    decode_context: *const gpt_arch.DecodeContext,
) ![]f32 {
    const logits = try gpt_arch.forward(
        &runtime_ctx.cb,
        allocator,
        runtime_ctx.gpt_config,
        input_ids,
        1,
        seq_len,
        decode_context,
    );
    defer allocator.free(logits);
    const vocab_size: usize = @intCast(runtime_ctx.gpt_config.vocab_size);
    const last_pos_offset = (query_seq_len - 1) * vocab_size;
    return allocator.dupe(f32, logits[last_pos_offset..][0..vocab_size]);
}

test "metal executor preserves mixed Gemma global KV history" {
    var config = gpt_mod.Config{
        .family = .gemma,
        .num_hidden_layers = 6,
        .position_encoding = .rope,
        .sliding_window = 512,
        .sliding_window_pattern = 6,
    };

    try std.testing.expectEqual(@as(?u32, null), executorKvSlidingWindowSize(config));

    config.num_hidden_layers = 5;
    try std.testing.expectEqual(@as(?u32, 512), executorKvSlidingWindowSize(config));
}

test "metal executor limits lazy device KV reserve to mixed-geometry Gemma" {
    var config = gpt_mod.Config{
        .family = .gemma,
        .num_hidden_layers = 6,
        .num_attention_heads = 8,
        .num_key_value_heads = 2,
        .attention_head_dim = 256,
        .position_encoding = .rope,
        .sliding_window = 512,
        .sliding_window_pattern = 6,
        .ple_hidden_size = 256,
    };

    try std.testing.expect(!executorUsesLazyDeviceKvReserve(config));

    config.global_head_dim = 512;
    try std.testing.expect(executorUsesLazyDeviceKvReserve(config));

    config.ple_hidden_size = 0;
    try std.testing.expect(!executorUsesLazyDeviceKvReserve(config));

    config.ple_hidden_size = 256;
    config.global_head_dim = 0;
    config.num_kv_shared_layers = 1;
    try std.testing.expect(executorUsesLazyDeviceKvReserve(config));

    config.family = .llama;
    try std.testing.expect(!executorUsesLazyDeviceKvReserve(config));
}

test "metal executor preserves split KV policy through its views and reset" {
    const allocator = std.testing.allocator;
    var storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(allocator, .{
        .backend = .metal,
        .dtype = .f16,
        .page_size_tokens = 16,
        .num_layers_packed = 2,
        .num_kv_heads = 2,
        .head_dim = 256,
    });
    defer storage.deinit();

    var metadata = ExecutorKvMetadataLayer.init(allocator, storage.poolId(), true);
    defer metadata.deinit(&storage);
    try metadata.configureKvPolicy(128, true);
    _ = try metadata.notePrefill(&storage, 32);
    try metadata.configureKvPolicy(128, true);
    try std.testing.expectError(error.InvalidPagedKvState, metadata.configureKvPolicy(64, true));

    const view = metadata.kv_view.?;
    try std.testing.expectEqual(@as(usize, 128), view.max_inflight_tokens);
    try std.testing.expect(view.allow_swa_ring);

    const contract_view = executorContractKvCacheView(view);
    try std.testing.expectEqual(@as(usize, 128), contract_view.max_inflight_tokens);
    try std.testing.expect(contract_view.allow_swa_ring);
    const decode_view = executorDecodeKvCacheView(view);
    try std.testing.expectEqual(@as(usize, 128), decode_view.max_inflight_tokens);
    try std.testing.expect(decode_view.allow_swa_ring);

    try metadata.reset(&storage);
    try std.testing.expectEqual(@as(usize, 0), metadata.max_inflight_tokens);
    try std.testing.expect(!metadata.allow_swa_ring);
    try std.testing.expect(metadata.kv_view == null);
}

test "metal executor only enables split KV policy for mixed local and global Gemma" {
    var config = gpt_mod.Config{
        .family = .gemma,
        .num_hidden_layers = 6,
        .position_encoding = .rope,
        .sliding_window = 512,
        .sliding_window_pattern = 6,
        .ple_hidden_size = 256,
    };
    try std.testing.expect(config.supportsSplitSwaGlobalKvRing());

    config.ple_hidden_size = 0;
    try std.testing.expect(!config.supportsSplitSwaGlobalKvRing());
    config.ple_hidden_size = 256;

    config.sliding_window = 0;
    try std.testing.expect(!config.supportsSplitSwaGlobalKvRing());
    config.sliding_window = 512;

    config.sliding_window_pattern = 7;
    try std.testing.expect(!config.supportsSplitSwaGlobalKvRing());
    config.sliding_window_pattern = 6;

    config.family = .llama;
    try std.testing.expect(!config.supportsSplitSwaGlobalKvRing());
}

test "metal executor pipelined decode frames delegate to the shared policy" {
    try std.testing.expect(metal_runtime.pipelinedDecodeFrameEnabledForFlags(false, false, true));
}

test "request suppressions retain guarded prefill cache policy" {
    try std.testing.expect(prefillMayCacheGreedyToken(true, &.{}));
    try std.testing.expect(!prefillMayCacheGreedyToken(true, &.{ 1, 50, 106 }));
    try std.testing.expect(!prefillMayCacheGreedyToken(false, &.{}));
}

test "metal executor keeps speculative decode frames opt in" {
    try std.testing.expect(!pipelinedDecodeFrameEnabledForFlags(false, false, false, true, true));
    try std.testing.expect(pipelinedDecodeFrameEnabledForFlags(true, false, false, false, false));
    try std.testing.expect(pipelinedDecodeFrameEnabledForFlags(false, false, true, true, true));
    try std.testing.expect(!pipelinedDecodeFrameEnabledForFlags(false, false, true, true, false));
    try std.testing.expect(!pipelinedDecodeFrameEnabledForFlags(false, false, true, false, false));
    try std.testing.expect(!pipelinedDecodeFrameEnabledForFlags(false, true, true, true, true));
    try std.testing.expect(!pipelinedDecodeFrameEnabledForFlags(true, true, false, false, false));
}

test "sampled pipeline rejects mode and position changes after arm" {
    try validateSampledPipelineContinuation(false, null, 0, 99);
    try validateSampledPipelineContinuation(true, .sampled_nucleus, 42, 42);
    try std.testing.expectError(
        error.PipelinedDecodeModeChanged,
        validateSampledPipelineContinuation(true, .greedy, 42, 42),
    );
    try std.testing.expectError(
        error.PipelinedDecodePositionChanged,
        validateSampledPipelineContinuation(true, .sampled_nucleus, 42, 43),
    );

    const matched = model_runtime.SamplingConfig{ .temperature = 0.8, .top_p = 0.95 };
    try std.testing.expect(sampledPipelineConfigMatches(matched, matched));
    try std.testing.expect(!sampledPipelineConfigMatches(null, matched));
    var changed = matched;
    changed.top_p = 0.9;
    try std.testing.expect(!sampledPipelineConfigMatches(matched, changed));
}

test "sampled pipeline cancels invalid draws before submitting the next frame" {
    const Probe = struct {
        submits: usize = 0,
        cancels: usize = 0,
        reject_submit: bool = false,
        fail_submit: bool = false,

        fn decode(ctx: *anyopaque, request: *const ops.DecoderRuntimeDecodeRequest) !bool {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            switch (request.phase) {
                .submit_pending => {
                    self.submits += 1;
                    if (self.fail_submit) return error.InjectedSubmitFailure;
                    return !self.reject_submit;
                },
                .cancel_pending => {
                    self.cancels += 1;
                    return true;
                },
                else => return error.UnexpectedDecodePhase,
            }
        }
    };
    var probe = Probe{};
    // The control path touches only this callback, never a tensor operation.
    var vtable: ops.ComputeBackend.VTable = undefined;
    vtable.decoderRuntimeDecode = Probe.decode;
    const cb = ops.ComputeBackend{ .ptr = &probe, .vtable = &vtable };
    const vocab_size: usize = 262144;
    for ([_]i64{ -1, vocab_size, std.math.maxInt(u32) }) |invalid| {
        try std.testing.expectError(error.InvalidModelOutput, submitSampledPipelineToken(&cb, invalid, vocab_size));
    }
    try std.testing.expectEqual(@as(usize, 0), probe.submits);
    try std.testing.expectEqual(@as(usize, 3), probe.cancels);

    try submitSampledPipelineToken(&cb, vocab_size - 1, vocab_size);
    try std.testing.expectEqual(@as(usize, 1), probe.submits);
    try std.testing.expectEqual(@as(usize, 3), probe.cancels);
    probe.reject_submit = true;
    try std.testing.expectError(error.ResidentSamplingFrameFailed, submitSampledPipelineToken(&cb, 7, vocab_size));
    probe.fail_submit = true;
    try std.testing.expectError(error.InjectedSubmitFailure, submitSampledPipelineToken(&cb, 7, vocab_size));
    try std.testing.expectEqual(@as(usize, 3), probe.submits);
    try std.testing.expectEqual(@as(usize, 5), probe.cancels);
}
