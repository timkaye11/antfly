#!/usr/bin/env bash

# Shared Gemma 4 QAT CUDA tuning profile. Source this file, then call
# gemma4_qat_cuda_tuning_env with the graph replay KV capacity. The resulting
# GEMMA4_QAT_CUDA_ENV array is suitable for `env "${...[@]}" command ...`.
gemma4_qat_cuda_tuning_env() {
  local capacity="${1:-${ANTFLY_CAPTURE_FORCE_KV_CAPACITY:-544}}"
  local decode_graph_replay="${antfly_decode_graph_replay:-${ANTFLY_DECODE_GRAPH_REPLAY:-required}}"
  local q4_rows="${antfly_q4_0_q8_1_prefill_rows:-${ANTFLY_Q4_0_Q8_1_PREFILL_ROWS:-1}}"
  local gate_up_q8="${ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE:-${antfly_q4_0_gate_up_activation_q8_1_precompute:-${ANTFLY_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE:-0}}}"
  local gqa_fast="${antfly_gqa_prefill_fast:-${ANTFLY_GQA_PREFILL_FAST:-1}}"
  local gqa_tiled="${antfly_gqa_prefill_tiled:-${ANTFLY_GQA_PREFILL_TILED:-0}}"
  local gqa_mma="${antfly_gqa_prefill_mma:-${ANTFLY_GQA_PREFILL_MMA:-0}}"
  local tile4_w8_min="${antfly_q4_0_linear_q8_1_tile4_w8_min_in_dim:-${ANTFLY_Q4_0_LINEAR_Q8_1_TILE4_W8_MIN_IN_DIM:-2048}}"
  local rows8_c4="${antfly_q4_0_linear_q8_1_rows8_c4:-${ANTFLY_Q4_0_LINEAR_Q8_1_ROWS8_C4:-1}}"
  local pair_rows8_c2="${antfly_q4_0_pair_activation_q8_1_rows8_c2:-${ANTFLY_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS8_C2:-1}}"
  local pair_rows16_c1="${antfly_q4_0_pair_activation_q8_1_rows16_c1:-${ANTFLY_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS16_C1:-1}}"
  local gemma_prewarm="${antfly_cuda_gemma_prefill_prewarm:-${ANTFLY_CUDA_GEMMA_PREFILL_PREWARM:-1}}"
  local prefill_first="${antfly_cuda_prefill_first_token:-${ANTFLY_CUDA_PREFILL_FIRST_TOKEN:-1}}"
  local prefill_coalesce="${antfly_cuda_prefill_first_token_coalesce_tokens:-${ANTFLY_CUDA_PREFILL_FIRST_TOKEN_COALESCE_TOKENS:-2048}}"
  local prefill_profile="${antfly_cuda_profile_prefill_ops:-${ANTFLY_CUDA_PROFILE_PREFILL_OPS:-0}}"
  local decode_profile="${antfly_cuda_profile_decode:-${ANTFLY_CUDA_PROFILE_DECODE:-0}}"
  local rms_bf16="${antfly_rms_norm_bf16_mirror:-${ANTFLY_INFERENCE_CUDA_RMS_NORM_BF16_MIRROR:-0}}"
  local resident_bf16="${ANTFLY_INFERENCE_CUDA_DEQUANTIZE_Q4_0_MATRIX_WEIGHTS_BF16:-${antfly_bf16_resident_weights:-0}}"
  local hybrid_bf16="${antfly_hybrid_bf16_prefill:-${ANTFLY_INFERENCE_CUDA_Q4_0_WEIGHTS_BF16_PREFILL:-0}}"
  local ple_bf16="${antfly_ple_model_proj_bf16:-${ANTFLY_INFERENCE_CUDA_PLE_MODEL_PROJ_BF16:-$resident_bf16}}"
  # Generated decode attention is a dev candidate: its split-KV reduction has
  # long-output parity drift, so production stays on the handwritten route
  # unless an explicit candidate run opts in.
  local generated_attention="${antfly_generated_attention_decode:-${ANTFLY_GENERATED_ATTENTION_DECODE:-${ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE:-0}}}"
  # The paged score-prework route is exact and faster on the E2B L4 corpus, but
  # stays default-off pending multi-model and long-context promotion evidence.
  local generated_attention_score_prework="${antfly_generated_attention_score_prework:-${ANTFLY_GENERATED_ATTENTION_SCORE_PREWORK:-${ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK:-0}}}"
  local turboquant_split_attention="${antfly_turboquant_split_attention:-${ANTFLY_TURBOQUANT_SPLIT_ATTENTION:-${ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION:-0}}}"
  local turboquant_split_attention_chunk="${antfly_turboquant_split_attention_chunk:-${ANTFLY_TURBOQUANT_SPLIT_ATTENTION_CHUNK:-${ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION_CHUNK:-12}}}"
  local q4_0_q8_1_lm_head_argmax="${antfly_q4_0_q8_1_lm_head_argmax:-${ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX:-1}}"
  local generated_q6_k_q8_1_lm_head_argmax="${antfly_generated_q6_k_q8_1_lm_head_argmax:-${ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX:-0}}"
  local generated_e2b_ffn="${ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN:-${antfly_generated_q4_0_e2b_ffn:-0}}"
  local generated_e2b_ffn_exact="${ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT:-${antfly_generated_q4_0_e2b_ffn_exact:-0}}"
  local linear_q8_1_dp4a="${ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A:-1}"
  local pair_activation_q8_1_dp4a="${ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A:-1}"
  local gated_down_q8_1_dp4a="${ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A:-1}"
  local async_i32_download_staging="${ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING:-1}"
  local greedy_pending_token_readback="${ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK:-1}"
  local temp_slot_period="${ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD:-${antfly_cuda_temp_slot_period:-${ANTFLY_CUDA_TEMP_SLOT_PERIOD:-853}}}"
  local temp_slot_skip="${ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP:-${antfly_cuda_temp_slot_skip:-${ANTFLY_CUDA_TEMP_SLOT_SKIP:-2500}}}"

  GEMMA4_QAT_CUDA_ENV=(
    ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING="$async_i32_download_staging"
    ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION="$turboquant_split_attention"
    ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION_CHUNK="$turboquant_split_attention_chunk"
    ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_PRECOMPUTE=1
    ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE="$gate_up_q8"
    ANTFLY_INFERENCE_CUDA_Q4_0_Q8_1_PREFILL_ROWS="$q4_rows"
    ANTFLY_INFERENCE_CUDA_GQA_PREFILL_FAST="$gqa_fast"
    ANTFLY_INFERENCE_CUDA_GQA_PREFILL_TILED="$gqa_tiled"
    ANTFLY_INFERENCE_CUDA_GQA_PREFILL_MMA="$gqa_mma"
    ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_TILE4_W8=1
    ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_TILE4_W8_MIN_IN_DIM="$tile4_w8_min"
    ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_ROWS8_C4="$rows8_c4"
    ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS8_C2="$pair_rows8_c2"
    ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS16_C1="$pair_rows16_c1"
    ANTFLY_INFERENCE_CUDA_GEMMA_PREFILL_PREWARM="$gemma_prewarm"
    ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN="$prefill_first"
    ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN_COALESCE_TOKENS="$prefill_coalesce"
    ANTFLY_INFERENCE_CUDA_PROFILE_PREFILL_OPS="$prefill_profile"
    ANTFLY_INFERENCE_CUDA_PROFILE_DECODE="$decode_profile"
    ANTFLY_INFERENCE_CUDA_RMS_NORM_BF16_MIRROR="$rms_bf16"
    ANTFLY_INFERENCE_CUDA_DEQUANTIZE_Q4_0_MATRIX_WEIGHTS_BF16="$resident_bf16"
    ANTFLY_INFERENCE_CUDA_Q4_0_WEIGHTS_BF16_PREFILL="$hybrid_bf16"
    ANTFLY_INFERENCE_CUDA_PLE_MODEL_PROJ_BF16="$ple_bf16"
    ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_TILE4_W10_E4B_DOWN=1
    ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_TILE4_W8=1
    ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK="$greedy_pending_token_readback"
    ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_Q8_1_DP4A="$linear_q8_1_dp4a"
    ANTFLY_INFERENCE_CUDA_Q4_0_QKV_Q8_1_DP4A=1
    ANTFLY_INFERENCE_CUDA_Q4_0_QKV_Q8_1_TILE8=1
    ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_Q8_1_DP4A=1
    ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_ACTIVATION_Q8_1_DP4A="$pair_activation_q8_1_dp4a"
    ANTFLY_INFERENCE_CUDA_Q4_0_ACTIVATION_SLICE_Q8_1_DP4A=1
    ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_Q8_1_DP4A="$gated_down_q8_1_dp4a"
    ANTFLY_INFERENCE_CUDA_Q6_K_LM_HEAD_Q8_1=1
    ANTFLY_INFERENCE_CUDA_Q6_K_LM_HEAD_Q8_1_TILE8_EXACT_THREADS=1
    ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY="$decode_graph_replay"
    ANTFLY_INFERENCE_CUDA_CAPTURE_FINAL_HIDDEN=1
    ANTFLY_INFERENCE_CUDA_CAPTURE_UPDATE_EXEC=1
    ANTFLY_INFERENCE_CUDA_CAPTURE_DEVICE_SCALARS=1
    ANTFLY_INFERENCE_CUDA_CAPTURE_PERSISTENT_REPLAY=1
    ANTFLY_INFERENCE_CUDA_CAPTURE_GREEDY_TOKEN=1
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD="$temp_slot_period"
    ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP="$temp_slot_skip"
    ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY="$capacity"
    ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE="$generated_attention"
    ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK="$generated_attention_score_prework"
    ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX="$q4_0_q8_1_lm_head_argmax"
    ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX="$generated_q6_k_q8_1_lm_head_argmax"
    ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN="$generated_e2b_ffn"
    ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT="$generated_e2b_ffn_exact"
  )

  case "$decode_graph_replay" in
    off|0|false)
      GEMMA4_QAT_CUDA_ENV+=(
        ANTFLY_INFERENCE_CUDA_CAPTURE_FINAL_HIDDEN=0
        ANTFLY_INFERENCE_CUDA_CAPTURE_UPDATE_EXEC=0
        ANTFLY_INFERENCE_CUDA_CAPTURE_DEVICE_SCALARS=0
        ANTFLY_INFERENCE_CUDA_CAPTURE_PERSISTENT_REPLAY=0
        ANTFLY_INFERENCE_CUDA_CAPTURE_GREEDY_TOKEN=0
        ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY=0
      )
      ;;
  esac
}
