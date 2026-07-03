# GPU Validation Runbook — Late SPLADE Head-Only Stage (Go Phase 32 port)

Validates the staged-SPLADE machinery in `train-fused-chunker`:
`--splade-training-mode head-only`, `--restore-best-for-splade-stage`,
`--reset-optimizer-on-splade-stage`, `--separate-splade-adapter`, and the
forward-only encoder fast path used during the head-only stage.

**Do not run while another GPU training job is active.** The mini-run below
takes roughly 10–20 minutes on an M-series GPU (2 × 60 steps at 22 layers,
batch 8, seq 384, plus two 128-sample evals).

CPU-side verification already done (no GPU required, re-runnable any time):

```bash
cd zig/pkg/inference
zig build -Dskip-openapi=true test-fused-chunker-splade-stage   # stage-logic unit tests
zig build -Dskip-openapi=true test-finetune                     # full finetune suite
zig build -Dskip-openapi=true -Dmetal=true train-fused-chunker -- --help
bash -n scripts/run_fused_chunker_phase20_metal.sh
```

## 0. Prepare the bounded 2-stage dataset

480 train samples → exactly 60 steps/epoch at batch 8. Epoch 1 trains
boundary+dense (60 steps), epoch 2 is the late SPLADE head-only stage
(60 steps) because `splade_focus_epoch=1` (0-indexed).

```bash
mkdir -p /private/tmp/fused_splade_stage
head -480 /private/tmp/fused_race/fused_train_4k.jsonl > /private/tmp/fused_splade_stage/train_480.jsonl
head -128 /private/tmp/fused_race/fused_val_512.jsonl  > /private/tmp/fused_splade_stage/val_128.jsonl
wc -l /private/tmp/fused_splade_stage/*.jsonl   # expect 480 / 128
```

## 1. Run the 2-stage mini-run (Metal)

```bash
cd zig/pkg/inference
ANTFLY_FUSED_CHUNKER_TRAIN_DATA=/private/tmp/fused_splade_stage/train_480.jsonl \
ANTFLY_FUSED_CHUNKER_VAL_DATA=/private/tmp/fused_splade_stage/val_128.jsonl \
ANTFLY_FUSED_CHUNKER_OUTPUT=/private/tmp/fused_splade_stage/run1 \
ANTFLY_FUSED_CHUNKER_EPOCHS=2 \
ANTFLY_FUSED_CHUNKER_WARMUP_STEPS=10 \
ANTFLY_FUSED_CHUNKER_LOG_EVERY=10 \
ANTFLY_FUSED_CHUNKER_EVAL_EVERY=1 \
ANTFLY_FUSED_CHUNKER_SPLADE=1 \
ANTFLY_FUSED_CHUNKER_SPLADE_FOCUS_EPOCH=1 \
ANTFLY_FUSED_CHUNKER_SPLADE_TRAINING_MODE=head-only \
ANTFLY_FUSED_CHUNKER_SEPARATE_SPLADE_ADAPTER=1 \
ANTFLY_FUSED_CHUNKER_RESTORE_BEST_FOR_SPLADE_STAGE=1 \
ANTFLY_FUSED_CHUNKER_RESET_OPTIMIZER_ON_SPLADE_STAGE=1 \
ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1 \
bash scripts/run_fused_chunker_phase20_metal.sh 2>&1 | tee /private/tmp/fused_splade_stage/run1.log
```

Notes:
- `ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1` also enables the
  capture-free compiled eval forward that the head-only stage reuses; drop it
  to validate the eager fallback instead.
- λ schedule (Go parity): with focus=1 and epochs=2, the single SPLADE epoch
  runs at λ_splade = 0.15 × 0.5 = 0.075 (first-active-epoch ramp-in) and
  λ_flops = 0 (quadratic ramp starts at 0). This is expected in the mini-run;
  production runs have multiple SPLADE epochs so FLOPS ramps in.

## 2. Assertions

### 2a. Stage transition fired

```bash
grep -E "entering SPLADE head-only stage|restoring best boundary checkpoint|resetting optimizer state for SPLADE stage|splade_stage epoch" /private/tmp/fused_splade_stage/run1.log
```

Expect all of, in order:
- `splade_stage_config mode=head-only focus_epoch=1 separate_adapter=true restore_best=true reset_optimizer=true ...`
- `entering SPLADE head-only stage at epoch 2/2`
- `restoring best boundary checkpoint before SPLADE stage: .../best_model.safetensors (epoch=1 ...)`
- `resetting optimizer state for SPLADE stage`
- `splade_stage epoch=2/2 head_only=true boundary_updates=frozen dense_updates=frozen lora_updates=frozen splade_updates=active`

### 2b. Boundary F1 identical before/after the SPLADE stage (frozen paths)

```bash
grep "^validation epoch" /private/tmp/fused_splade_stage/run1.log
```

The `validation epoch 1/2` and `validation epoch 2/2` lines must be
**identical** in every metric (`f1`, `precision`, `recall`, `tp/fp/fn`,
`best_f1`, `ap`, probability stats). The head-only stage never calls the
boundary/dense/LoRA update paths, and stage entry restores the best checkpoint
(saved at the end of epoch 1), so any difference is a freeze violation — fail
the validation and investigate.

Quick check:

```bash
a=$(grep "^validation epoch 1/2" /private/tmp/fused_splade_stage/run1.log | sed 's/^validation epoch 1\/2//; s/eval_ms.*//')
b=$(grep "^validation epoch 2/2" /private/tmp/fused_splade_stage/run1.log | sed 's/^validation epoch 2\/2//; s/eval_ms.*//')
[ "$a" = "$b" ] && echo "PASS: boundary metrics frozen" || { echo "FAIL: boundary metrics drifted"; exit 1; }
```

### 2c. SPLADE loss decreasing

```bash
grep "splade_loss:" /private/tmp/fused_splade_stage/run1.log | head -3
grep "splade_loss:" /private/tmp/fused_splade_stage/run1.log | tail -3
```

Expect ~60 `splade_loss: ... stage: head-only` lines (epoch 2 only) and a
clearly decreasing trend from the first to the last few steps. Also expect
`lambda_splade: 0.0750` on every line (single active epoch → ramp-in weight).

### 2d. Sparsity stats reasonable

Each SPLADE line prints `sparsity_pct` (fraction of exactly-zero activations
across valid chunk vectors, Go parity metric). Freshly initialized W gives
roughly ~50% zeros (ReLU of symmetric init); with λ_flops=0 in this mini-run
it should stay in a sane band, not collapse:

- PASS: `sparsity_pct` within [30, 99.9] on the final steps and not 100
  (100% = dead head; near-0% with FLOPS active would mean no sparsification).
- Production runs (multiple SPLADE epochs, FLOPS ramp) should trend toward
  Go's ~96%.

### 2e. Step throughput reflects the forward-only fast path

```bash
grep "^epoch 1/2 done" /private/tmp/fused_splade_stage/run1.log
grep "^epoch 2/2 done" /private/tmp/fused_splade_stage/run1.log
```

Epoch 2 `avg_step_ms` should be substantially lower than epoch 1 (no boundary
VJP, no LoRA backward, no hard-negative forward; `train` and `lora` columns
should be ~0 while `splade` is nonzero).

### 2f. Checkpoint contains SPLADE head tensors and export picks them up

```bash
python3 - <<'EOF'
import json, struct
p = "/private/tmp/fused_splade_stage/run1/checkpoint_final.safetensors"
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
key = "fused_chunker_embedder/splade_head/proj/weight"
assert key in hdr, f"missing {key}"
assert hdr[key]["shape"] == [50368, 768], hdr[key]["shape"]
print("PASS: checkpoint carries", key, hdr[key]["shape"])
EOF

zig build -Dskip-openapi=true export-fused-chunker-model -- \
  --checkpoint /private/tmp/fused_splade_stage/run1/checkpoint_final.safetensors \
  --model-dir "$HOME/.cache/modernbert-base" \
  --out /private/tmp/fused_splade_stage/export1 \
  | tee /private/tmp/fused_splade_stage/export1.log
grep '"splade":true\|"splade": true' /private/tmp/fused_splade_stage/export1.log \
  && echo "PASS: export tool picked up SPLADE head"
```

(`best_model.safetensors` also carries the tensor whenever a best checkpoint
is re-saved while `--splade` is on; the legacy standalone
`splade_w_final.safetensors` is still written for older tooling.)

### 2g. Optional joint-mode control (regression sanity)

Re-run step 1 with `ANTFLY_FUSED_CHUNKER_SPLADE_TRAINING_MODE=joint` into
`.../run_joint`. Expect: no `entering SPLADE head-only stage` line, `stage:
joint` on SPLADE lines, and `validation epoch 2/2` metrics that DIFFER from
epoch 1/2 (boundary keeps training) — demonstrating the head-only freeze is
what pins the metrics in 2b.

### 2h. Optional compiled/eager parity for the head-only forward

Run step 1 twice (same seed) with `ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD`
set to 1 and 0. Epoch-2 `validation epoch 2/2` metrics must match run-to-run
per path; splade_loss trajectories should agree within cross-kernel noise
(compiled vs eager kernels differ by ~2e-3 max-rel on features at depth 22).

## 3. Cleanup

```bash
rm -rf /private/tmp/fused_splade_stage
```
