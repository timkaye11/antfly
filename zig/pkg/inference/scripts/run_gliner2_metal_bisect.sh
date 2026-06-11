#!/bin/zsh
# Phase 0.1 guard bisect (Metal_Gliner_Next_steps.md §0.1).
# Runs the production repro with each logicalStridesOrContiguous null
# source disabled in turn (TERMITE_BISECT_GUARD_*_OFF), plus baseline and
# all-off (= old stride semantics) sanity runs. Interpreter mode.
# Logs to /tmp/bisect_<label>.log; prints a loss summary at the end.
set -e
cd "$(dirname "$0")/.."

COMMON="--model-dir /private/tmp/termite-models/gliner2 \
  --train-data /tmp/gliner2_metal_diag.jsonl --epochs 1 --batch-size 2 --max-examples 2 \
  --seq-len 64 --learning-rate 0.001 --weight-decay 0.0 --backend metal \
  --objective gliner2-total-loss --max-span-width 4 --span-loss bce \
  --span-loss-reduction sum --span-positive-weight 1.0 --span-negative-weight 1.0 \
  --span-hard-negative-weight 1.0 --span-negative-mask-rate 0.0 \
  --lora-rank 4 --lora-alpha 8.0 --lora-dropout 0.0 \
  --lora-targets encoder,span_rep,classifier,count_embed,count_pred --seed 42 \
  --lora-only-trainables --deterministic \
  --initial-adapter-checkpoint /private/tmp/gliner2-metal-probe/python/initial_adapter/adapter_weights.safetensors"

run_one() {
  local label="$1"
  echo "=== bisect run: $label ==="
  TERMITE_COMPILED_TRAIN_TRACE=1 zig build -Dmetal=true -Doptimize=ReleaseFast \
    train-gliner2-autodiff -- ${=COMMON} --out-dir "/tmp/repro-bisect-$label" \
    > "/tmp/bisect_$label.log" 2>&1 || echo "RUN FAILED (exit $?)"
  grep -E "loss=|train_step_ms" "/tmp/bisect_$label.log" | tail -2
}

run_one baseline
( export TERMITE_BISECT_GUARD_INDEX_MAP_OFF=1; run_one index_map_off )
( export TERMITE_BISECT_GUARD_RANK_OFF=1; run_one rank_off )
( export TERMITE_BISECT_GUARD_DIMS_OFF=1; run_one dims_off )
( export TERMITE_BISECT_GUARD_INDEX_MAP_OFF=1 TERMITE_BISECT_GUARD_RANK_OFF=1 TERMITE_BISECT_GUARD_DIMS_OFF=1
  run_one all_off )

echo ""
echo "=== SUMMARY (expect baseline=0.000000, all_off=19.230522) ==="
for label in baseline index_map_off rank_off dims_off all_off; do
  loss=$(grep -Eo "loss=[0-9.]+" "/tmp/bisect_$label.log" | tail -1)
  echo "$label: $loss"
done
