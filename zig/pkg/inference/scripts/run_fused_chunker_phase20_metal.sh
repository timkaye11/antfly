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
model_dir="${ANTFLY_FUSED_CHUNKER_MODEL_DIR:-$HOME/.cache/modernbert-embed-base}"
output_dir="${ANTFLY_FUSED_CHUNKER_OUTPUT:-/private/tmp/zig-fused-phase20-metal}"
epochs="${ANTFLY_FUSED_CHUNKER_EPOCHS:-5}"
batch_size="${ANTFLY_FUSED_CHUNKER_BATCH_SIZE:-8}"
max_seq_len="${ANTFLY_FUSED_CHUNKER_MAX_SEQ_LEN:-384}"
max_chunks="${ANTFLY_FUSED_CHUNKER_MAX_CHUNKS:-32}"
num_layers="${ANTFLY_FUSED_CHUNKER_NUM_LAYERS:-22}"
learning_rate="${ANTFLY_FUSED_CHUNKER_LR:-0.00002}"
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
mrl_dims="${ANTFLY_FUSED_CHUNKER_MRL_DIMS:-768,256,128}"
# The Go CLI default is 5.0, but the Phase 20 best F1=0.786 run explicitly
# used plain CE with pos_weight=1.0.
pos_weight="${ANTFLY_FUSED_CHUNKER_POS_WEIGHT:-1.0}"
boundary_loss_type="${ANTFLY_FUSED_CHUNKER_LOSS_TYPE:-ce}"
boundary_focal_gamma="${ANTFLY_FUSED_CHUNKER_FOCAL_GAMMA:-2.0}"
boundary_focal_alpha="${ANTFLY_FUSED_CHUNKER_FOCAL_ALPHA:-0.75}"
log_every="${ANTFLY_FUSED_CHUNKER_LOG_EVERY:-100}"
eval_every="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY:-1}"
checkpoint_every_steps="${ANTFLY_FUSED_CHUNKER_CHECKPOINT_EVERY_STEPS:-0}"
eval_every_steps="${ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS:-0}"
step_eval_max_examples="${ANTFLY_FUSED_CHUNKER_STEP_EVAL_MAX_EXAMPLES:-0}"
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

echo "training Zig fused chunker Phase-20 Metal/MPSGraph parity run"
echo "Phase 20 best-boundary parity contract"
echo "  source=gopeft Phase 20 best F1=0.786 boundary run"
echo "  scope=boundary+dense only; SPLADE disabled unless explicitly requested"
echo "  epochs=$epochs batch_size=$batch_size max_seq_len=$max_seq_len max_chunks=$max_chunks"
echo "  lr=$learning_rate warmup_steps=$warmup_steps lr_total_steps=$lr_total_steps weight_decay=$weight_decay max_grad_norm=$max_grad_norm"
echo "  lora_rank=$lora_rank lora_alpha=$lora_alpha targets=go(query_proj,value_proj,key_proj,Wo) zig(query_proj,key_proj,value_proj,out_proj,wo)"
echo "  lambda_chunk=$lambda_chunk lambda_embed=$lambda_embed boundary_focus_epochs=$boundary_focus_epochs boundary_focus_lambda_embed=$boundary_focus_lambda_embed"
echo "  boundary_dropout=$boundary_dropout neftune_alpha=$neftune_alpha mrl_dims=$mrl_dims loss_type=$boundary_loss_type pos_weight=$pos_weight"
echo "  note=Go CLI default pos_weight is 5.0; Phase 20 best run uses 1.0"
echo "  train_data=$train_data"
echo "  val_data=$val_data"
echo "  model_dir=$model_dir"
echo "  tokenizer_path=$model_dir/tokenizer.json"
echo "  model_sha256=$model_sha256"
echo "  tokenizer_sha256=$tokenizer_sha256"
echo "  output_dir=$output_dir"
echo "  zig=$zig_bin"
echo "  pos_weight=$pos_weight"
echo "  boundary_loss_type=$boundary_loss_type"
echo "  focal_gamma=$boundary_focal_gamma"
echo "  focal_alpha=$boundary_focal_alpha"
echo "  neftune_alpha=$neftune_alpha"
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
echo "  max_steps=$max_steps"
echo "  save_optimizer_state=$save_optimizer_state"
echo "  resume_from=$resume_from"
echo "  splade=$enable_splade"

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
case "$enable_splade" in
  1|true|TRUE|yes|YES)
    extra_args+=(--splade --lambda-splade "$lambda_splade" --lambda-flops "$lambda_flops" --splade-focus-epoch "$splade_focus_epoch")
    ;;
esac

if ((${#extra_args[@]} > 0)); then
  set -- "${extra_args[@]}" "$@"
fi

exec "$zig_bin" build -Doptimize=ReleaseFast -Dmetal=true -Dmlx=false train-fused-chunker -- \
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
  --learning-rate "$learning_rate" \
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
