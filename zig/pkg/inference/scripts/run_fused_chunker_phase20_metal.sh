#!/usr/bin/env bash
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd -- "$script_dir/.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  elif [[ -x "$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig" ]]; then
    zig_bin="$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

train_data="${ANTFLY_FUSED_CHUNKER_TRAIN_DATA:-/Users/tim/Documents/af/gopeft/data/fused_train.jsonl}"
val_data="${ANTFLY_FUSED_CHUNKER_VAL_DATA:-/Users/tim/Documents/af/gopeft/data/fused_val.jsonl}"
# Go Phase 20 (best F1 0.786) trained from answerdotai modernbert-base, not
# the nomic embed variant; replication runs must start from the same weights.
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-base}"
output_dir="${ANTFLY_FUSED_CHUNKER_OUTPUT:-/private/tmp/zig-fused-phase20-metal}"
epochs="${ANTFLY_FUSED_CHUNKER_EPOCHS:-5}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
split="${ANTFLY_FUSED_CHUNKER_SPLIT:-train}"
val_split="${ANTFLY_FUSED_CHUNKER_VAL_SPLIT:-val}"
num_layers="${ANTFLY_FUSED_CHUNKER_NUM_LAYERS:-22}"
learning_rate="${ANTFLY_FUSED_CHUNKER_LR:-0.00002}"
boundary_head_lr_multiplier="${ANTFLY_FUSED_CHUNKER_BOUNDARY_HEAD_LR_MULTIPLIER:-1.0}"
warmup_steps="${ANTFLY_FUSED_CHUNKER_WARMUP_STEPS:-200}"
lr_total_steps="${ANTFLY_FUSED_CHUNKER_LR_TOTAL_STEPS:-0}"
weight_decay="${ANTFLY_FUSED_CHUNKER_WEIGHT_DECAY:-0.01}"
max_grad_norm="${ANTFLY_FUSED_CHUNKER_MAX_GRAD_NORM:-1}"
lora_rank="${ANTFLY_FUSED_CHUNKER_LORA_RANK:-16}"
lora_alpha="${ANTFLY_FUSED_CHUNKER_LORA_ALPHA:-32}"
lambda_chunk="${ANTFLY_FUSED_CHUNKER_LAMBDA_CHUNK:-1.0}"
lambda_embed="${ANTFLY_FUSED_CHUNKER_LAMBDA_EMBED:-0.3}"
boundary_focus_epochs="${ANTFLY_FUSED_CHUNKER_BOUNDARY_FOCUS_EPOCHS:-3}"
boundary_focus_lambda_embed="${ANTFLY_FUSED_CHUNKER_BOUNDARY_FOCUS_LAMBDA_EMBED:-0.1}"
boundary_dropout="${ANTFLY_FUSED_CHUNKER_BOUNDARY_DROPOUT:-0.1}"
# Go Phase 20 best-boundary recipe uses NEFTune alpha 5.0 with encoder
# dropout disabled and boundary-head dropout at 0.1.
neftune_alpha="${ANTFLY_FUSED_CHUNKER_NEFTUNE_ALPHA:-5.0}"
# Phase 20's recorded best run attributes its no-plateau behavior to applied
# NEFTune. Keep this enabled for production/quality runs; set
# ANTFLY_FUSED_CHUNKER_ENCODER_NEFTUNE=0 only for deterministic parity probes.
encoder_neftune="${ANTFLY_FUSED_CHUNKER_ENCODER_NEFTUNE:-1}"
mrl_dims="${ANTFLY_FUSED_CHUNKER_MRL_DIMS:-768,256,128}"
# Go Phase 20 used pos_weight=1.0 on the fused semantic data; the auto
# estimate (~0.45% positive rate) produces a huge weight that drives the model
# to predict ~39% positives. Set ANTFLY_FUSED_CHUNKER_POS_WEIGHT=auto to
# estimate a balanced weight from tokenized train labels instead.
pos_weight="${ANTFLY_FUSED_CHUNKER_POS_WEIGHT:-1.0}"
pos_weight_auto_max_examples="${ANTFLY_FUSED_CHUNKER_POS_WEIGHT_AUTO_MAX_EXAMPLES:-2048}"
boundary_rank_loss_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_RANK_LOSS_WEIGHT:-0.0}"
boundary_rank_loss_margin="${ANTFLY_FUSED_CHUNKER_BOUNDARY_RANK_LOSS_MARGIN:-1.0}"
boundary_rank_loss_top_k="${ANTFLY_FUSED_CHUNKER_BOUNDARY_RANK_LOSS_TOP_K:-1}"
boundary_same_token_rank_loss_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_SAME_TOKEN_RANK_LOSS_WEIGHT:-0.0}"
boundary_same_token_rank_loss_top_k="${ANTFLY_FUSED_CHUNKER_BOUNDARY_SAME_TOKEN_RANK_LOSS_TOP_K:-4}"
boundary_same_token_negative_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_SAME_TOKEN_NEGATIVE_WEIGHT:-1.0}"
boundary_same_token_negative_top_k="${ANTFLY_FUSED_CHUNKER_BOUNDARY_SAME_TOKEN_NEGATIVE_TOP_K:-0}"
boundary_candidate_rank_loss_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_CANDIDATE_RANK_LOSS_WEIGHT:-0.0}"
boundary_candidate_rank_loss_top_k="${ANTFLY_FUSED_CHUNKER_BOUNDARY_CANDIDATE_RANK_LOSS_TOP_K:-8}"
boundary_candidate_negative_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_CANDIDATE_NEGATIVE_WEIGHT:-1.0}"
boundary_gold_count_rank_loss_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_GOLD_COUNT_RANK_LOSS_WEIGHT:-0.0}"
boundary_gold_count_rank_loss_margin="${ANTFLY_FUSED_CHUNKER_BOUNDARY_GOLD_COUNT_RANK_LOSS_MARGIN:-1.0}"
boundary_gold_count_rank_loss_negative_multiplier="${ANTFLY_FUSED_CHUNKER_BOUNDARY_GOLD_COUNT_RANK_LOSS_NEGATIVE_MULTIPLIER:-1}"
boundary_local_window_loss_weight="${ANTFLY_FUSED_CHUNKER_BOUNDARY_LOCAL_WINDOW_LOSS_WEIGHT:-0.0}"
boundary_local_window_radius="${ANTFLY_FUSED_CHUNKER_BOUNDARY_LOCAL_WINDOW_RADIUS:-12}"
boundary_feature_mode="${ANTFLY_FUSED_CHUNKER_BOUNDARY_FEATURE_MODE:-token}"
boundary_loss_type="${ANTFLY_FUSED_CHUNKER_LOSS_TYPE:-ce}"
boundary_focal_gamma="${ANTFLY_FUSED_CHUNKER_FOCAL_GAMMA:-2.0}"
boundary_focal_alpha="${ANTFLY_FUSED_CHUNKER_FOCAL_ALPHA:-0.75}"
log_every="${ANTFLY_FUSED_CHUNKER_LOG_EVERY:-100}"
eval_every="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY:-1}"
checkpoint_every_steps="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVERY_STEPS:-0}"
eval_every_steps="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-0}"
step_eval_max_examples="${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-0}"
step_train_eval_max_examples="${ANTFLY_FUSED_CHUNKER_STEP_TRAIN_EVAL_MAX_EXAMPLES:-0}"
boundary_alignment_dump_dir="${ANTFLY_FUSED_CHUNKER_BOUNDARY_ALIGNMENT_DUMP_DIR:-}"
boundary_alignment_dump_top_k="${ANTFLY_FUSED_CHUNKER_BOUNDARY_ALIGNMENT_DUMP_TOP_K:-32}"
checkpoint_roundtrip_eval="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_ROUNDTRIP_EVAL:-0}"
checkpoint_roundtrip_max_examples="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_ROUNDTRIP_MAX_EXAMPLES:-0}"
max_steps="${ANTFLY_FUSED_CHUNKER_MAX_STEPS:-0}"
save_optimizer_state="${ANTFLY_FUSED_CHUNKER_SAVE_OPTIMIZER_STATE:-1}"
resume_from="${ANTFLY_FUSED_CHUNKER_RESUME_FROM:-}"
encoder_vjp="${ANTFLY_FUSED_CHUNKER_ENCODER_VJP:-full}"
segment_vjp_execution="${ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION:-${ANTFLY_FUSED_CHUNKER_ENCODER_VJP_BACKEND:-mpsgraph_required}}"
mpsgraph_smoke="${ANTFLY_FUSED_CHUNKER_MPSGRAPH_SMOKE:-1}"
layers_per_segment="${ANTFLY_FUSED_CHUNKER_LAYERS_PER_SEGMENT:-1}"
lora_train_top_k="${ANTFLY_FUSED_CHUNKER_LORA_TRAIN_TOP_K:-0}"
lora_start_epoch="${ANTFLY_FUSED_CHUNKER_LORA_START_EPOCH:-0}"
enable_splade="${ANTFLY_FUSED_CHUNKER_SPLADE:-0}"
lambda_splade="${ANTFLY_FUSED_CHUNKER_LAMBDA_SPLADE:-0.15}"
lambda_flops="${ANTFLY_FUSED_CHUNKER_LAMBDA_FLOPS:-3e-5}"
splade_focus_epoch="${ANTFLY_FUSED_CHUNKER_SPLADE_FOCUS_EPOCH:-4}"
hash_model_artifacts="${ANTFLY_FUSED_CHUNKER_HASH_MODEL_ARTIFACTS:-1}"
deterministic="${ANTFLY_FUSED_CHUNKER_DETERMINISTIC:-0}"
go_epoch_shuffle="${ANTFLY_FUSED_CHUNKER_GO_EPOCH_SHUFFLE:-1}"
mixed_precision="${ANTFLY_FUSED_CHUNKER_MIXED_PRECISION:-0}"
memory_sample_every="${ANTFLY_FUSED_CHUNKER_MEMORY_SAMPLE_EVERY:-1}"
memory_warn_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_WARN_RSS_GB:-0}"
memory_abort_rss_gb="${ANTFLY_FUSED_CHUNKER_MEMORY_ABORT_RSS_GB:-0}"
debug_update_step="${ANTFLY_FUSED_CHUNKER_DEBUG_UPDATE_STEP:-0}"
debug_update_step_exit="${ANTFLY_FUSED_CHUNKER_DEBUG_UPDATE_STEP_EXIT:-0}"
debug_boundary_step="${ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_STEP:-0}"
debug_boundary_step_exit="${ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_STEP_EXIT:-0}"
debug_boundary_head_overfit_steps="${ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_HEAD_OVERFIT_STEPS:-0}"
debug_boundary_head_overfit_lr="${ANTFLY_FUSED_CHUNKER_DEBUG_BOUNDARY_HEAD_OVERFIT_LR:-0.001}"
debug_step_json="${ANTFLY_FUSED_CHUNKER_DEBUG_STEP_JSON:-}"
debug_batch_offset="${ANTFLY_FUSED_CHUNKER_DEBUG_BATCH_OFFSET:-}"
debug_encoder_probe_layer="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_PROBE_LAYER:-0}"
debug_encoder_layer_inputs_only="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_LAYER_INPUTS_ONLY:-0}"
debug_encoder_replay_input="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_REPLAY_INPUT:-}"
debug_encoder_replay_upstream="${ANTFLY_FUSED_CHUNKER_DEBUG_ENCODER_REPLAY_UPSTREAM:-}"
debug_layer_backward_decomp="${ANTFLY_FUSED_CHUNKER_DEBUG_LAYER_BACKWARD_DECOMP:-0}"
debug_qkv_split_vjp="${ANTFLY_FUSED_CHUNKER_QKV_SPLIT_VJP:-0}"
dense_device_forward="${TERMITE_METAL_DISABLE_DEVICE_DENSE_LINEAR_FORWARD:-1}"
# Opt-in step-time fast paths (all default OFF; validated baseline behavior is
# unchanged until they pass the GPU validation gate):
#   compiled_segment_forward — P1: run the LoRA encoder forward through
#     compiled MPSGraph segment sessions instead of the eager capture forward.
#   vjp_feed_cache — P2: cache rope tables / frozen base-weight feed buffers
#     per VJP session and share the two attention-bias variants per step.
#   compiled_boundary_head — P4: run the boundary-head forward/backward as
#     cached MPSGraph executables (CE math and dropout RNG stay on CPU).
#   compiled_forward_check — first-step eager-vs-compiled activation parity
#     gate for P1 (errors loudly on divergence above the tolerance).
compiled_segment_forward="${ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD:-0}"
vjp_feed_cache="${ANTFLY_FUSED_CHUNKER_VJP_FEED_CACHE:-0}"
compiled_boundary_head="${ANTFLY_FUSED_CHUNKER_COMPILED_BOUNDARY_HEAD:-0}"
compiled_forward_check="${ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK:-0}"

sha256_or_skip() {
  local path="$1"
  case "$hash_model_artifacts" in
    1|true|TRUE|yes|YES) ;;
    *) echo "disabled"; return 0 ;;
  esac
  if command -v shasum >/dev/null 2>&1; then
    local line
    line="$(shasum -a 256 "$path")"
    echo "${line%% *}"
  else
    echo "unavailable"
  fi
}

if [[ ! -f "$train_data" ]]; then
  echo "missing training data at $train_data" >&2
  exit 1
fi

if [[ ! -f "$val_data" ]]; then
  echo "missing validation data at $val_data" >&2
  exit 1
fi

if [[ ! -f "$model_dir/model.safetensors" ]]; then
  echo "missing ModernBERT weights at $model_dir/model.safetensors" >&2
  exit 1
fi

if [[ ! -f "$model_dir/tokenizer.json" ]]; then
  echo "missing ModernBERT tokenizer at $model_dir/tokenizer.json" >&2
  exit 1
fi

model_sha256="$(sha256_or_skip "$model_dir/model.safetensors")"
tokenizer_sha256="$(sha256_or_skip "$model_dir/tokenizer.json")"

cd "$pkg_root"
export ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION="$segment_vjp_execution"
export TERMITE_MPSGRAPH_SMOKE="$mpsgraph_smoke"
export TERMITE_METAL_DISABLE_DEVICE_DENSE_LINEAR_FORWARD="$dense_device_forward"
export ANTFLY_FUSED_CHUNKER_VJP_FEED_CACHE="$vjp_feed_cache"
export ANTFLY_FUSED_CHUNKER_COMPILED_BOUNDARY_HEAD="$compiled_boundary_head"
export ANTFLY_FUSED_CHUNKER_COMPILED_FORWARD_CHECK="$compiled_forward_check"
case "$vjp_feed_cache" in
  1|true|TRUE|yes|YES)
    # The feed-cache host buffers stay alive across the synchronous MPSGraph
    # execute call, so the no-copy NSData input views are safe to enable.
    export TERMITE_MPSGRAPH_EXECUTE_NOCOPY_INPUTS="${TERMITE_MPSGRAPH_EXECUTE_NOCOPY_INPUTS:-1}"
    ;;
esac

echo "training Zig fused chunker Phase-20 Metal/MPSGraph parity run"
echo "Phase 20 best-boundary parity contract"
echo "  source=gopeft Phase 20 best F1=0.786 boundary run"
echo "  scope=boundary+dense only; SPLADE disabled unless explicitly requested"
echo "  epochs=$epochs batch_size=$batch_size max_seq_len=$max_seq_len max_chunks=$max_chunks"
echo "  lr=$learning_rate boundary_head_lr_multiplier=$boundary_head_lr_multiplier warmup_steps=$warmup_steps lr_total_steps=$lr_total_steps weight_decay=$weight_decay max_grad_norm=$max_grad_norm"
echo "  lora_rank=$lora_rank lora_alpha=$lora_alpha targets=go(query_proj,value_proj,key_proj,Wo) zig(query_proj,key_proj,value_proj,out_proj,wo)"
echo "  lambda_chunk=$lambda_chunk lambda_embed=$lambda_embed boundary_focus_epochs=$boundary_focus_epochs boundary_focus_lambda_embed=$boundary_focus_lambda_embed"
echo "  boundary_dropout=$boundary_dropout neftune_alpha=$neftune_alpha encoder_neftune=$encoder_neftune mrl_dims=$mrl_dims loss_type=$boundary_loss_type pos_weight=$pos_weight boundary_rank_loss_weight=$boundary_rank_loss_weight boundary_rank_loss_margin=$boundary_rank_loss_margin boundary_rank_loss_top_k=$boundary_rank_loss_top_k boundary_same_token_rank_loss_weight=$boundary_same_token_rank_loss_weight boundary_same_token_rank_loss_top_k=$boundary_same_token_rank_loss_top_k boundary_same_token_negative_weight=$boundary_same_token_negative_weight boundary_same_token_negative_top_k=$boundary_same_token_negative_top_k boundary_candidate_rank_loss_weight=$boundary_candidate_rank_loss_weight boundary_candidate_rank_loss_top_k=$boundary_candidate_rank_loss_top_k boundary_candidate_negative_weight=$boundary_candidate_negative_weight boundary_gold_count_rank_loss_weight=$boundary_gold_count_rank_loss_weight boundary_gold_count_rank_loss_margin=$boundary_gold_count_rank_loss_margin boundary_gold_count_rank_loss_negative_multiplier=$boundary_gold_count_rank_loss_negative_multiplier boundary_local_window_loss_weight=$boundary_local_window_loss_weight boundary_local_window_radius=$boundary_local_window_radius boundary_feature_mode=$boundary_feature_mode"
echo "  note=pos_weight=auto estimates token-level neg/positive balance before training"
echo "  train_data=$train_data"
echo "  val_data=$val_data"
echo "  split=$split"
echo "  val_split=$val_split"
echo "  model_dir=$model_dir"
echo "  tokenizer_path=$model_dir/tokenizer.json"
echo "  model_sha256=$model_sha256"
echo "  tokenizer_sha256=$tokenizer_sha256"
echo "  output_dir=$output_dir"
echo "  zig=$zig_bin"
echo "  pos_weight=$pos_weight"
echo "  boundary_head_lr_multiplier=$boundary_head_lr_multiplier"
echo "  pos_weight_auto_max_examples=$pos_weight_auto_max_examples"
echo "  boundary_rank_loss_weight=$boundary_rank_loss_weight"
echo "  boundary_rank_loss_margin=$boundary_rank_loss_margin"
echo "  boundary_rank_loss_top_k=$boundary_rank_loss_top_k"
echo "  boundary_same_token_rank_loss_weight=$boundary_same_token_rank_loss_weight"
echo "  boundary_same_token_rank_loss_top_k=$boundary_same_token_rank_loss_top_k"
echo "  boundary_same_token_negative_weight=$boundary_same_token_negative_weight"
echo "  boundary_same_token_negative_top_k=$boundary_same_token_negative_top_k"
echo "  boundary_candidate_rank_loss_weight=$boundary_candidate_rank_loss_weight"
echo "  boundary_candidate_rank_loss_top_k=$boundary_candidate_rank_loss_top_k"
echo "  boundary_candidate_negative_weight=$boundary_candidate_negative_weight"
echo "  boundary_gold_count_rank_loss_weight=$boundary_gold_count_rank_loss_weight"
echo "  boundary_gold_count_rank_loss_margin=$boundary_gold_count_rank_loss_margin"
echo "  boundary_gold_count_rank_loss_negative_multiplier=$boundary_gold_count_rank_loss_negative_multiplier"
echo "  boundary_local_window_loss_weight=$boundary_local_window_loss_weight"
echo "  boundary_local_window_radius=$boundary_local_window_radius"
echo "  boundary_feature_mode=$boundary_feature_mode"
echo "  boundary_loss_type=$boundary_loss_type"
echo "  focal_gamma=$boundary_focal_gamma"
echo "  focal_alpha=$boundary_focal_alpha"
echo "  neftune_alpha=$neftune_alpha"
echo "  encoder_neftune=$encoder_neftune"
echo "  encoder_vjp=$encoder_vjp"
echo "  segment_vjp_execution=$segment_vjp_execution"
echo "  mpsgraph_smoke=$mpsgraph_smoke"
echo "  layers_per_segment=$layers_per_segment"
echo "  lora_start_epoch=$lora_start_epoch"
echo "  lora_train_top_k=$lora_train_top_k"
echo "  lr_total_steps=$lr_total_steps"
echo "  checkpoint_every_steps=$checkpoint_every_steps"
echo "  eval_every_steps=$eval_every_steps"
echo "  step_eval_max_examples=$step_eval_max_examples"
echo "  step_train_eval_max_examples=$step_train_eval_max_examples"
echo "  boundary_alignment_dump_dir=$boundary_alignment_dump_dir"
echo "  boundary_alignment_dump_top_k=$boundary_alignment_dump_top_k"
echo "  checkpoint_roundtrip_eval=$checkpoint_roundtrip_eval"
echo "  checkpoint_roundtrip_max_examples=$checkpoint_roundtrip_max_examples"
echo "  max_steps=$max_steps"
echo "  save_optimizer_state=$save_optimizer_state"
echo "  resume_from=$resume_from"
echo "  splade=$enable_splade"
echo "  deterministic=$deterministic"
echo "  go_epoch_shuffle=$go_epoch_shuffle"
echo "  mixed_precision=$mixed_precision"
echo "  disable_device_dense_linear_forward=$dense_device_forward"
echo "  compiled_segment_forward=$compiled_segment_forward"
echo "  vjp_feed_cache=$vjp_feed_cache"
echo "  compiled_boundary_head=$compiled_boundary_head"
echo "  compiled_forward_check=$compiled_forward_check"
echo "  memory_sample_every=$memory_sample_every"
echo "  memory_warn_rss_gb=$memory_warn_rss_gb"
echo "  memory_abort_rss_gb=$memory_abort_rss_gb"
echo "  debug_update_step=$debug_update_step"
echo "  debug_update_step_exit=$debug_update_step_exit"
echo "  debug_boundary_step=$debug_boundary_step"
echo "  debug_boundary_step_exit=$debug_boundary_step_exit"
echo "  debug_boundary_head_overfit_steps=$debug_boundary_head_overfit_steps"
echo "  debug_boundary_head_overfit_lr=$debug_boundary_head_overfit_lr"
echo "  debug_step_json=$debug_step_json"
echo "  debug_batch_offset=$debug_batch_offset"
echo "  debug_encoder_probe_layer=$debug_encoder_probe_layer"
echo "  debug_encoder_layer_inputs_only=$debug_encoder_layer_inputs_only"
echo "  debug_layer_backward_decomp=$debug_layer_backward_decomp"

extra_args=()
if [[ "$lr_total_steps" != "0" ]]; then
  extra_args+=(--lr-total-steps "$lr_total_steps")
fi
if [[ "$checkpoint_every_steps" != "0" ]]; then
  extra_args+=(--checkpoint-every-steps "$checkpoint_every_steps")
fi
if [[ "$eval_every_steps" != "0" ]]; then
  extra_args+=(--eval-every-steps "$eval_every_steps")
fi
if [[ "$step_eval_max_examples" != "0" ]]; then
  extra_args+=(--step-eval-max-examples "$step_eval_max_examples")
fi
if [[ "$step_train_eval_max_examples" != "0" ]]; then
  extra_args+=(--step-train-eval-max-examples "$step_train_eval_max_examples")
fi
if [[ "$boundary_local_window_loss_weight" != "0" && "$boundary_local_window_loss_weight" != "0.0" ]]; then
  extra_args+=(--boundary-local-window-loss-weight "$boundary_local_window_loss_weight")
fi
if [[ "$boundary_local_window_radius" != "12" ]]; then
  extra_args+=(--boundary-local-window-radius "$boundary_local_window_radius")
fi
if [[ "$boundary_gold_count_rank_loss_weight" != "0" && "$boundary_gold_count_rank_loss_weight" != "0.0" ]]; then
  extra_args+=(--boundary-gold-count-rank-loss-weight "$boundary_gold_count_rank_loss_weight")
fi
if [[ "$boundary_gold_count_rank_loss_margin" != "1" && "$boundary_gold_count_rank_loss_margin" != "1.0" ]]; then
  extra_args+=(--boundary-gold-count-rank-loss-margin "$boundary_gold_count_rank_loss_margin")
fi
if [[ "$boundary_gold_count_rank_loss_negative_multiplier" != "1" ]]; then
  extra_args+=(--boundary-gold-count-rank-loss-negative-multiplier "$boundary_gold_count_rank_loss_negative_multiplier")
fi
if [[ "$boundary_feature_mode" != "token" ]]; then
  extra_args+=(--boundary-feature-mode "$boundary_feature_mode")
fi
if [[ -n "$boundary_alignment_dump_dir" ]]; then
  extra_args+=(--boundary-alignment-dump-dir "$boundary_alignment_dump_dir" --boundary-alignment-dump-top-k "$boundary_alignment_dump_top_k")
fi
case "$checkpoint_roundtrip_eval" in
  1|true|TRUE|yes|YES)
    extra_args+=(--checkpoint-roundtrip-eval)
    if [[ "$checkpoint_roundtrip_max_examples" != "0" ]]; then
      extra_args+=(--checkpoint-roundtrip-max-examples "$checkpoint_roundtrip_max_examples")
    fi
    ;;
esac
if [[ "$max_steps" != "0" ]]; then
  extra_args+=(--max-steps "$max_steps")
fi
if [[ -n "$resume_from" ]]; then
  extra_args+=(--resume-from "$resume_from")
fi
case "$save_optimizer_state" in
  1|true|TRUE|yes|YES)
    extra_args+=(--save-optimizer-state)
    ;;
esac
case "$deterministic" in
  1|true|TRUE|yes|YES)
    extra_args+=(--deterministic)
    ;;
esac
case "$go_epoch_shuffle" in
  1|true|TRUE|yes|YES)
    extra_args+=(--go-epoch-shuffle)
    ;;
esac
case "$mixed_precision" in
  1|true|TRUE|yes|YES)
    extra_args+=(--mixed-precision)
    ;;
esac
if [[ "$memory_sample_every" != "1" ]]; then
  extra_args+=(--memory-sample-every "$memory_sample_every")
fi
if [[ "$memory_warn_rss_gb" != "0" ]]; then
  extra_args+=(--memory-warn-rss-gb "$memory_warn_rss_gb")
fi
if [[ "$memory_abort_rss_gb" != "0" ]]; then
  extra_args+=(--memory-abort-rss-gb "$memory_abort_rss_gb")
fi
if [[ "$debug_update_step" != "0" ]]; then
  extra_args+=(--debug-update-step "$debug_update_step")
fi
case "$debug_update_step_exit" in
  1|true|TRUE|yes|YES)
    extra_args+=(--debug-update-step-exit)
    ;;
esac
if [[ "$debug_boundary_step" != "0" ]]; then
  extra_args+=(--debug-boundary-step "$debug_boundary_step")
fi
case "$debug_boundary_step_exit" in
  1|true|TRUE|yes|YES)
    extra_args+=(--debug-boundary-step-exit)
    ;;
esac
if [[ "$debug_boundary_head_overfit_steps" != "0" ]]; then
  extra_args+=(--debug-boundary-head-overfit-steps "$debug_boundary_head_overfit_steps" --debug-boundary-head-overfit-lr "$debug_boundary_head_overfit_lr")
fi
if [[ -n "$debug_step_json" ]]; then
  extra_args+=(--debug-step-json "$debug_step_json")
fi
if [[ -n "$debug_batch_offset" ]]; then
  extra_args+=(--debug-batch-offset "$debug_batch_offset")
fi
if [[ "$debug_encoder_probe_layer" != "0" ]]; then
  extra_args+=(--debug-encoder-probe-layer "$debug_encoder_probe_layer")
fi
case "$debug_encoder_layer_inputs_only" in
  1|true|TRUE|yes|YES)
    extra_args+=(--debug-encoder-layer-inputs-only)
    ;;
esac
if [[ -n "$debug_encoder_replay_input" ]]; then
  extra_args+=(--debug-encoder-replay-input "$debug_encoder_replay_input")
fi
if [[ -n "$debug_encoder_replay_upstream" ]]; then
  extra_args+=(--debug-encoder-replay-upstream "$debug_encoder_replay_upstream")
fi
case "$debug_layer_backward_decomp" in
  1|true|TRUE|yes|YES)
    extra_args+=(--debug-layer-backward-decomp)
    ;;
esac
case "$debug_qkv_split_vjp" in
  1|true|TRUE|yes|YES)
    extra_args+=(--debug-qkv-split-vjp)
    ;;
esac
case "$compiled_segment_forward" in
  1|true|TRUE|yes|YES)
    extra_args+=(--compiled-segment-forward)
    ;;
esac
case "$encoder_neftune" in
  1|true|TRUE|yes|YES)
    extra_args+=(--encoder-neftune)
    ;;
  *)
    extra_args+=(--disable-encoder-neftune)
    ;;
esac
case "$enable_splade" in
  1|true|TRUE|yes|YES)
    extra_args+=(--splade --lambda-splade "$lambda_splade" --lambda-flops "$lambda_flops" --splade-focus-epoch "$splade_focus_epoch")
    ;;
esac

if ((${#extra_args[@]} > 0)); then
  set -- "${extra_args[@]}" "$@"
fi

exec "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true train-fused-chunker -- \
  --data "$train_data" \
  --val-data "$val_data" \
  --output "$output_dir" \
  --model-dir "$model_dir" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --backend metal \
  --lora-rank "$lora_rank" \
  --lora-alpha "$lora_alpha" \
  --num-layers "$num_layers" \
  --max-seq-len "$max_seq_len" \
  --max-chunks "$max_chunks" \
  --split "$split" \
  --val-split "$val_split" \
  --learning-rate "$learning_rate" \
  --boundary-head-lr-multiplier "$boundary_head_lr_multiplier" \
  --warmup-steps "$warmup_steps" \
  --weight-decay "$weight_decay" \
  --max-grad-norm "$max_grad_norm" \
  --lambda-chunk "$lambda_chunk" \
  --lambda-embed "$lambda_embed" \
  --boundary-focus-epochs "$boundary_focus_epochs" \
  --boundary-focus-lambda-embed "$boundary_focus_lambda_embed" \
  --boundary-dropout "$boundary_dropout" \
  --neftune-alpha "$neftune_alpha" \
  --mrl \
  --mrl-dims "$mrl_dims" \
  --pos-weight "$pos_weight" \
  --boundary-pos-weight-auto-max-examples "$pos_weight_auto_max_examples" \
  --boundary-rank-loss-weight "$boundary_rank_loss_weight" \
  --boundary-rank-loss-margin "$boundary_rank_loss_margin" \
  --boundary-rank-loss-top-k "$boundary_rank_loss_top_k" \
  --boundary-same-token-rank-loss-weight "$boundary_same_token_rank_loss_weight" \
  --boundary-same-token-rank-loss-top-k "$boundary_same_token_rank_loss_top_k" \
  --boundary-same-token-negative-weight "$boundary_same_token_negative_weight" \
  --boundary-same-token-negative-top-k "$boundary_same_token_negative_top_k" \
  --boundary-candidate-rank-loss-weight "$boundary_candidate_rank_loss_weight" \
  --boundary-candidate-rank-loss-top-k "$boundary_candidate_rank_loss_top_k" \
  --boundary-candidate-negative-weight "$boundary_candidate_negative_weight" \
  --loss-type "$boundary_loss_type" \
  --focal-gamma "$boundary_focal_gamma" \
  --focal-alpha "$boundary_focal_alpha" \
  --checkpoint-every 1 \
  --log-every "$log_every" \
  --eval-every "$eval_every" \
  --lora-start-epoch "$lora_start_epoch" \
  --lora-train-top-k "$lora_train_top_k" \
  --encoder-vjp "$encoder_vjp" \
  --layers-per-segment "$layers_per_segment" \
  "$@"
