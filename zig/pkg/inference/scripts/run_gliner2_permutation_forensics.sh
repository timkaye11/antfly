#!/bin/zsh
# Phase 0.3 permutation forensics (Metal_Gliner_Next_steps.md §0.3).
#
# Usage:
#   scripts/run_gliner2_permutation_forensics.sh discover
#       Run the Metal repro with TERMITE_DUMP_GRAPH_NODES=1395-1415 and
#       print the node-1405 mul's input node ids.
#   scripts/run_gliner2_permutation_forensics.sh dump <operand_node_id>
#       Dump that node's full buffer on: native (truth), metal-new
#       (current semantics), metal-old (all bisect guards off = old
#       semantics), then run the numpy permutation analysis.
set -e
cd "$(dirname "$0")/.."

OUT=/tmp/forensics
mkdir -p $OUT

COMMON_BASE="--model-dir /private/tmp/termite-models/gliner2 \
  --train-data /tmp/gliner2_metal_diag.jsonl --epochs 1 --batch-size 2 --max-examples 2 \
  --seq-len 64 --learning-rate 0.001 --weight-decay 0.0 \
  --objective gliner2-total-loss --max-span-width 4 --span-loss bce \
  --span-loss-reduction sum --span-positive-weight 1.0 --span-negative-weight 1.0 \
  --span-hard-negative-weight 1.0 --span-negative-mask-rate 0.0 \
  --lora-rank 4 --lora-alpha 8.0 --lora-dropout 0.0 \
  --lora-targets encoder,span_rep,classifier,count_embed,count_pred --seed 42 \
  --lora-only-trainables --deterministic \
  --initial-adapter-checkpoint /private/tmp/gliner2-metal-probe/python/initial_adapter/adapter_weights.safetensors"

run_train() {
  # $1 = backend, $2 = out-dir label; extra env via caller export
  local backend="$1" label="$2"
  TERMITE_COMPILED_TRAIN_TRACE=1 zig build -Dmetal=true -Doptimize=ReleaseFast \
    train-gliner2-autodiff -- ${=COMMON_BASE} --backend "$backend" \
    --out-dir "/tmp/repro-forensics-$label" > "$OUT/run_$label.log" 2>&1 || true
}

case "$1" in
  discover)
    ( export TERMITE_DUMP_GRAPH_NODES="1395-1415"
      run_train metal discover )
    echo "--- nodes 1395-1415 (look for the {1536,64,64} mul and its inputs) ---"
    grep -E "node=(139[5-9]|140[0-9]|141[0-5])" "$OUT/run_discover.log" | head -40
    ;;
  dump)
    node_id="$2"
    [[ -n "$node_id" ]] || { echo "usage: $0 dump <operand_node_id>"; exit 1 }
    echo "=== native (truth) ==="
    ( export TERMITE_GRAPH_NODE_DUMP="$node_id:$OUT/native_a.bin"
      run_train native native )
    grep -E "node-dump|loss=" "$OUT/run_native.log" | tail -3
    echo "=== metal new semantics ==="
    ( export TERMITE_GRAPH_NODE_DUMP="$node_id:$OUT/metal_new_a.bin"
      run_train metal metal_new )
    grep -E "node-dump|loss=" "$OUT/run_metal_new.log" | tail -3
    echo "=== metal old semantics (all bisect guards off) ==="
    ( export TERMITE_GRAPH_NODE_DUMP="$node_id:$OUT/metal_old_a.bin"
      export TERMITE_BISECT_GUARD_INDEX_MAP_OFF=1 TERMITE_BISECT_GUARD_RANK_OFF=1 TERMITE_BISECT_GUARD_DIMS_OFF=1
      run_train metal metal_old )
    grep -E "node-dump|loss=" "$OUT/run_metal_old.log" | tail -3
    echo "=== permutation analysis: truth vs metal_new ==="
    python3 scripts/gliner2_permutation_forensics.py \
      --truth $OUT/native_a.bin --wrong $OUT/metal_new_a.bin --also $OUT/metal_old_a.bin
    echo "=== permutation analysis: truth vs metal_old ==="
    python3 scripts/gliner2_permutation_forensics.py \
      --truth $OUT/native_a.bin --wrong $OUT/metal_old_a.bin
    ;;
  *)
    echo "usage: $0 discover | dump <operand_node_id>"
    exit 1
    ;;
esac
