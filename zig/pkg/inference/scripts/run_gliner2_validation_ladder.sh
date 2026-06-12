#!/bin/bash
# GLiNER2 Metal training validation ladder (plan Phase A2/A3).
# Runs the correctness ladder + perf snapshot, every training run wrapped in
# the memory-pressure watchdog. Logs land under output/validation/.
#
# Usage: scripts/run_gliner2_validation_ladder.sh [--with-rel-gather]
#   --with-rel-gather  additionally run the ladder with
#                      TERMITE_DEBERTA_REL_SCORE_GATHER=1 (Phase D A/B)
set -u
cd "$(dirname "$0")/.." || exit 1
WATCHDOG=scripts/gliner2_memory_watchdog.sh
MODEL=/private/tmp/termite-models/gliner2
HARD=/tmp/gliner2_hard.jsonl
REALISTIC=/tmp/gliner2_realistic.jsonl
OUT=output/validation
mkdir -p "$OUT"
BIN_ARGS_COMMON="--model-dir $MODEL --learning-rate 1e-3 --objective gliner2-total-loss \
  --lora-rank 4 --lora-alpha 8 --lora-dropout 0 --max-span-width 4 --lora-only-trainables \
  --span-loss-reduction sum --seed 42"

REL_GATHER=0
[ "${1:-}" = "--with-rel-gather" ] && REL_GATHER=1

run_point() { # name env_str extra_args...
  local name="$1"; shift
  local envs="$1"; shift
  echo "=== $name ==="
  rm -rf "$OUT/$name"
  # shellcheck disable=SC2086
  env $envs "$WATCHDOG" "$OUT/$name.log" -- \
    zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- \
    $BIN_ARGS_COMMON --out-dir "$OUT/$name" "$@"
  local rc=$?
  if [ $rc -eq 42 ] || [ $rc -eq 43 ]; then
    echo "!!! $name killed by memory watchdog (rc=$rc)"
  elif [ $rc -ne 0 ]; then
    echo "!!! $name FAILED rc=$rc"
  fi
  grep -E "effective seq_len|estimated peak|loss|parity|WATCHDOG" "$OUT/$name.log" | tail -25
  return $rc
}

ladder() { # prefix env_prefix
  local p="$1" e="$2"

  # A2.1 hard fixture b2 s64, graph executor + in-Zig parity gate
  run_point "${p}hard_b2s64_graphexec" \
    "$e TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK=1" \
    --train-data "$HARD" --epochs 4 --batch-size 2 --max-examples 4 --seq-len 64 --backend metal

  # A2.1b same config, metal interpreter (no graph executor)
  run_point "${p}hard_b2s64_interp" "$e" \
    --train-data "$HARD" --epochs 4 --batch-size 2 --max-examples 4 --seq-len 64 --backend metal

  # A2.4 native regression
  run_point "${p}hard_b2s64_native" "$e" \
    --train-data "$HARD" --epochs 4 --batch-size 2 --max-examples 4 --seq-len 64 --backend native

  # A2.2 realistic fixture b1 s128: native vs interpreter vs graph-exec
  run_point "${p}real_b1s128_native" "$e" \
    --train-data "$REALISTIC" --epochs 1 --batch-size 1 --max-examples 1 --seq-len 128 --backend native
  run_point "${p}real_b1s128_interp" "$e" \
    --train-data "$REALISTIC" --epochs 1 --batch-size 1 --max-examples 1 --seq-len 128 --backend metal
  run_point "${p}real_b1s128_graphexec" "$e TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1" \
    --train-data "$REALISTIC" --epochs 1 --batch-size 1 --max-examples 1 --seq-len 128 --backend metal

  # A2.3 the crash config: b1 s256 must clamp via fit-to-data and survive
  run_point "${p}real_b1s256_graphexec" "$e TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1" \
    --train-data "$REALISTIC" --epochs 1 --batch-size 1 --max-examples 1 --seq-len 256 --backend metal

  # A3 perf snapshot: warm steps with loop profile
  run_point "${p}perf_hard_b2s64" "$e TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 TERMITE_METAL_PARTITION_LOOP_PROFILE=1" \
    --train-data "$HARD" --epochs 10 --batch-size 2 --max-examples 4 --seq-len 64 --backend metal
  run_point "${p}perf_real_b1s128" "$e TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 TERMITE_METAL_PARTITION_LOOP_PROFILE=1" \
    --train-data "$REALISTIC" --epochs 10 --batch-size 1 --max-examples 4 --seq-len 128 --backend metal
}

ladder "" "TERMITE_DEBERTA_REL_SCORE_GATHER=0"
if [ "$REL_GATHER" = "1" ]; then
  ladder "relgather_" "TERMITE_DEBERTA_REL_SCORE_GATHER=1"
fi

echo "=== ladder complete; logs in $OUT ==="
