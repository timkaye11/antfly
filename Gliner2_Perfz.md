# Beat Python CPU: kill the host-encode overhead in fused-attention training

## Context

Branch `gliner2_finetuning_parity`. The fused disentangled-attention work is **done and committed**:
forward (flash4) + backward (4 custom Metal kernels) run as single GPU kernels with no `[bh,S,S]`
materialization, behind `TERMITE_DEBERTA_FUSED_ATTENTION` (default-off), parity-clean (in-Zig
graph-vs-direct rel ~1e-7). At b1 seq128 a step dropped from **2648ms → 408ms (6.5×)** and the GPU
attention whale is gone (frame GPU ~14ms). The step is now **host-CPU-encode-bound**: the loop
profile shows `execution_ms ~270` (the per-node loop) vs `submit_frame_ms ~18` (one GPU submit).
Python CPU is ~167ms, so we're ~2.4× away.

**Investigation (3 explore agents + code reads) localized the ~270ms:**
- **Verified the attention kernels DO batch into one frame** — `termite_metal_decode_runtime_begin_frame`
  (metal_kernels.m:35038-35076) sets `active_frame_cb`; `termite_metal_decode_runtime_command_buffer`
  (:6035-6039) returns the active frame CB with `frame_owned = false` so dispatches reuse the frame
  (one submit). So "separate per-layer submits" is NOT the problem.
- **Prime suspect — per-op mask host-reads (~80-120ms).** Each fused op (×24 = 12 fwd + 12 bwd)
  derives the key mask via `attentionMaskFromBias` (metal_partition_executor.zig:5763-5784; call
  sites :5697, :9119, :9146) / `disentangledMaskFromBias` (interpreter.zig:1750-1773; call sites
  :2107, :2132; ALSO the custom VJP at autodiff.zig:1304) → `cb.toFloat32(attn_bias)`, which copies
  the Private `[bh,S,S]` bias device→host (`toFloat32Op` metal_compute.zig:4294-4365 →
  `toHostSlice` → frame flush + `termite_metal_buffer_download`, metal_tensor.zig:524-573). So each
  read forces a GPU sync MID-FRAME, ×24. And it's **redundant**: the `__gliner2_attn_bias`
  parameter is the SAME node for all 12 layers, so 23 of 24 reads are wasted. **Compounding it:**
  the Metal backend then converts each `[]i64` mask to f32 and uploads a FRESH device buffer per op
  (metal_compute.zig:8256-8262) before the kernel reads it as `device const float *mask
  [[buffer(5)]]` (metal_kernels.m:3874) — 24 redundant small uploads on top of the 24 reads.
- **Secondary — per-dispatch hazard scan (~50-80ms)**: `termite_metal_decode_runtime_prepare_planned_compute_accesses`
  (metal_kernels.m:4879-4918) is O(ranges × active_ranges) per kernel dispatch (conflict/overflow →
  flush + memory barrier); the backward runs 5 dispatches/op × 12 = 60, each declaring up to 9 ranges.
- **Per-node bookkeeping (~14-70ms)**: `cloneOutputIfAliasedInputWouldBeFreed` / `freeExpiredInputs`
  / `classifyMetalExecutionKind` recomputed for ~4747 nodes every step. Mostly static —
  `classifyMetalExecutionKind` (:501-529) and the `freeExpiredInputs` decision (:11121-11176) are
  pure functions of op type + `last_use[]`. **EXCEPTION:** `cloneOutputIfAliasedInputWouldBeFreed`
  (interpreter.zig:961-992) detects aliasing by runtime pointer comparison (`ct == output_ct`), so
  it is per-step data-dependent and canNOT be fully precomputed — only its static precondition
  (`canKeepAliasedOutput(op)` + `last_use` check) can.
- `partition_view` + `runtime_region_plan` are ALREADY cached when the graph is static
  (metal_partition_executor.zig :580-587; reuse path :695-710 via `plan.matches()`, keyed on
  first/last node id + lengths + value_count — NOTE: no shapes in the key).
  `analyzeRuntimeFrameEligibility` (:2938)'s compiled "frame" is inference-only and blocks training
  graphs (`.missing_model_metadata`, :3009) — NOT reusable.

**Decisions (user-confirmed):** commit to beating Python now (mask fix AND the plan-replay
rearchitecture). Each lever is a PURE perf change that must keep the in-Zig graph-vs-direct parity
green (rel ~1e-7) and the loss trajectory unchanged; run the strict Python gate once at the end.
All behind the existing flag (default-off). Measure the s128 wall delta after EACH lever — the
agent estimates conflict, so let measurement, not guesses, drive the order (12-false-fix discipline).

## Lever 1 — Eliminate the per-op mask host-reads (clearest, do first)

Get the padding mask onto the device ONCE per step and feed it to every fused attention op,
instead of each op reading the `[bh,S,S]` bias to host (and re-uploading a fresh mask buffer).

- **Primary: bind the mask the trainer already has.** The trainer holds the `[B,S]` attention mask
  host-side and currently DISCARDS it (`_ = attention_mask;` gliner2_real_autodiff.zig:194) — then
  builds the bias FROM it (`BertPlaceholderPrep.buildAttnBias`, :255) and binds only
  `__gliner2_attn_bias` (:214-221, bound in `bindArchInputs` :241-267). So: add a
  `__gliner2_attn_mask` `[batch, seq]` f32 graph parameter, bind it from the same host data in
  `bindArchInputs`, and pass its NodeId into `buildForwardGraphInternal`
  (deberta_graph.zig:101-127) alongside `attn_bias`, shared across all 12 `encoderLayer` calls
  (:165). NO in-graph derivation needed (avoids slicing `[bh,S,S]`; note `greater_than` doesn't
  exist as a prim anyway — only `less_than`/`where_select`).
- **Thread it into BOTH ops.** Forward: add the mask node as an input of
  `fused_disentangled_attention` (today ins = qkv_packed, qr_kr_packed, attn_bias[2]; bias stays —
  the kernel still consumes it). Backward: the custom VJP (autodiff.zig:1295-1310, NOT
  `vjp_alternate`-lowered — lower.zig:319-320) must thread the mask node into
  `fused_disentangled_attention_backward`'s inputs too (today ins[0..3] = qkv, qr_kr, bias, dOut),
  since it currently re-derives the mask itself (autodiff.zig:1304). Mask params get no gradient
  (parameters are leaves; comparison ops are non-differentiable anyway) — semantically correct.
- **cb contract (`src/ops/ops.zig`):** change `disentangledRelativeAttention`/`...Backward`
  (ops.zig:1199/:1206) `mask: []const i64` → `CT` (device mask tensor). Update the 4 backends:
  metal passes the CT's buffer straight to the kernel — it already takes a float `[B,S]` mask at
  `buffer(5)` (metal_kernels.m:3874), so this kills BOTH the per-op `toFloat32` host read AND the
  per-op f32-convert+upload (metal_compute.zig:8256-8262); native/mlx/wasm read the CT host-side
  into the `[]i64` they use today. Update the interpreter + executor arms (interpreter.zig:2107,
  :2132; metal_partition_executor.zig:9119, :9146, and the DebertaAttentionPattern path :5697) to
  pass the mask CT (drop `attentionMaskFromBias`/`disentangledMaskFromBias` on the hot path).
- If the CT-signature change proves too invasive, the lower-risk fallback is a per-step cache of the
  derived `[]i64` mask in `MetalCompute` keyed by the bias node id (first op derives, rest reuse) —
  kills the 24 host reads (1 instead of 24), no graph/signature change. NOTE: the fallback still
  leaves the 24 per-op f32-convert+device-uploads; cache the uploaded device mask buffer per step
  too if measurement says it matters.
- **Gate:** in-Zig parity green + identical loss trajectory; measure s128 wall (expect ~80-120ms off).

## Lever 2 — Compiled per-step plan-replay (the big rearchitecture)

The static training graph re-derives per-node execution decisions and re-encodes every step. Build
a `CompiledStepPlan` once (keyed by the static node_ids/shapes, alongside the existing
`runtime_region_plan` cache) and replay it as a branch-light loop.

- Precompute per node what IS static: execution kind (`classifyMetalExecutionKind`, :501-529 —
  function of op type + storage class), `OperatorPlan`/region assignment, the `freeExpiredInputs`
  list (:11121-11176 — pure function of `last_use[]` + `plan.needsAttentionInputAfterNode()`), and
  the static PREcondition of the alias-clone check (`canKeepAliasedOutput(op)` + `last_use`).
- What is NOT precomputable: the alias-clone decision itself
  (`cloneOutputIfAliasedInputWouldBeFreed`, interpreter.zig:961-992) compares runtime buffer
  pointers (`ct == output_ct`), which vary step-to-step. Replay path: skip the call entirely for
  nodes whose static precondition is false (the vast majority), and keep the cheap pointer check at
  runtime only for the statically-flagged candidates.
- **Files:** `src/graph/metal_partition_executor.zig` — add the `CompiledStepPlan` to the persistent
  `MetalPartitionExecutor` struct (:580-587, beside `partition_view`/`runtime_region_plan`),
  populate it when the region plan is first built/cached (`buildRuntimeRegionPlan` :2378, reuse
  path :695-710), and add a replay path in the node loop (`while (node_pos < node_ids.len)` at
  :973, through ~:1259) that skips per-step re-derivation on cache hit.
- This is a NEW training-step plan cache — do NOT route through `analyzeRuntimeFrameEligibility`
  (inference-only, blocked for training).
- **Gate:** parity green; measure (this targets ALL ops' per-step planning overhead, not just
  attention).

## Lever 3 — Trim per-dispatch hazard-scan + dispatch count (cleanup)

After Levers 1-2, attack the residual:
- Reduce the backward's 5 dispatches (K1-K4, K3×2) — fuse `bwd_dv`/`bwd_dq_dk`/`bwd_dqr_dkr` where
  the gather structure allows, or cut declared `prepare_planned_compute_accesses` ranges.
- Profile `prepare_planned_compute_accesses` cost; if O(active_ranges) dominates, bound the active
  set or skip hazard tracking for the two private score-scratch buffers (provably single-writer→
  single-reader per dispatch).
- **Gate:** parity green; measure.

## Critical files

- `src/training/gliner2_real_autodiff.zig` — bind `__gliner2_attn_mask` `[B,S]` param (:194, :241-267)
- `src/architectures/deberta_graph.zig` — thread the mask NodeId through `buildForwardGraphInternal` → `encoderLayer` → fused op inputs
- `src/graph/autodiff.zig` — custom VJP (:1295-1310) threads mask into the backward node's inputs
- `src/ops/ops.zig` — attention-op `mask` param `[]i64` → `CT` (4-backend vtable, :1199/:1206)
- `src/ops/metal_compute.zig` / `native_compute.zig` / `mlx_compute.zig` / `wasm_compute.zig` — mask-CT handling
- `src/graph/interpreter.zig` + `src/graph/metal_partition_executor.zig` — fused-op arms pass the mask CT; `CompiledStepPlan` + replay
- `src/backends/metal_kernels.m` — (Lever 3) backward dispatch/hazard tuning
- Reuse: `partition_view`/`runtime_region_plan` caching, the watchdog + ladder scripts, the in-Zig parity check

## Verification (after each lever)

```bash
cd zig/pkg/inference
# parity (must stay green, rel ~1e-7) + loss trajectory unchanged
TERMITE_DEBERTA_FUSED_ATTENTION=1 TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 \
TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK=1 scripts/gliner2_memory_watchdog.sh output/p.log -- \
  zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- \
  --model-dir /private/tmp/termite-models/gliner2 --train-data /tmp/gliner2_hard.jsonl \
  --out-dir output/p --epochs 4 --batch-size 2 --max-examples 4 --seq-len 64 --backend metal \
  --objective gliner2-total-loss --lora-rank 4 --lora-alpha 8 --lora-dropout 0 --max-span-width 4 \
  --lora-only-trainables --span-loss-reduction sum --span-positive-weight 1 --span-negative-weight 1 \
  --span-hard-negative-weight 1 --seed 42 --learning-rate 1e-3
# wall/gpu measurement at s128 (the bottleneck regime)
TERMITE_DEBERTA_FUSED_ATTENTION=1 TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 ... --seq-len 128 \
  --epochs 12 ... ; # read trainer_total_ms + metal_frame_gpu_ms from training_metrics.jsonl
```

Final: strict Python metal gate (entity smoke, `--disable-python-model-dropout`, 5e-4 roundtrip tol).
**Success = s128 step < 167ms (beats Python CPU), parity green, default-off flag flippable on.**

## Key risks

1. Mask-CT signature change touches 4 backends → keep the `[]i64` fallback path; or use the
   per-step mask cache (no signature change) if it proves cleaner. Threading a new input through
   the custom VJP changes the backward node's input arity — keep the non-fused (default-off) path
   byte-identical.
2. Plan-replay cache invalidation: must rebuild on any shape change (seq-len, batch). CAUTION: the
   existing `runtime_region_plan` key (`plan.matches()`: first/last node id + lengths +
   value_count) does NOT include shapes — fine for region patterns, but a `CompiledStepPlan` that
   bakes in shape-derived decisions needs a shape (or graph-generation) component in its key. A
   stale replay = silent wrong execution → guard with the in-Zig parity check on every run.
3. Do NOT precompute the alias-clone decision (runtime pointer-dependent, see Lever 2) — a baked-in
   wrong clone/no-clone is a use-after-free or silent corruption.
4. A lever doesn't move the needle → measurement reorders priorities (don't sink effort into
   plan-replay if Lever 1 alone gets under Python).
