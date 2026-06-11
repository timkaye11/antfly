# Gemma 4 / GLiNER2 Next Steps

Last updated: 2026-06-10

## Goal

Get the Zig GLiNER2 LoRA fine-tuning path to production-grade parity with upstream Python GLiNER2, then make the Metal path fast enough to be comparable to Python training speed.

Companion docs:

- `Metal_Gliner2_Claude.md`: current evidence dossier.
- `Metal_Gliner_Next_steps.md`: focused Metal correctness and performance plan for branch `gliner2_finetuning_parity`.

Current performance baseline from `Metal_Gliner_Next_steps.md`: Python CPU is roughly `0.17-0.26s/step` at batch 2; Metal graph executor is roughly `3.0s/step` warm today, with wrong numerics.

The target is not just "the command runs." The target is:

- Same preprocessing semantics as upstream Python GLiNER2.
- Same loss semantics for `total_loss = classification_loss + structure_loss + count_loss`.
- Same LoRA target resolution, adapter format, optimizer state progression, gradient clipping, and scheduler behavior.
- Same trained-adapter behavior when loaded back through the Python GLiNER2/PEFT stack.
- Metal correctness for both interpreter and graph executor paths.
- Metal performance with real graph residency, no silent interpreter fallback, no host trainable transfer loop, and stable benchmark gates.

## Current State

`Metal_Gliner2_Claude.md` says the native Zig path is in good shape:

- Native Zig vs Python parity is CI-gated on the small entity fixture.
- One-step loss parity is at very tight tolerance.
- Multi-step optimizer state parity and adapter round-trip checks exist.
- Partial final batches and Adam zero-grad step-count drift were already fixed.

The current blocker is Metal correctness:

- Expected first-step GLiNER2 total loss: `19.230522`.
- Metal interpreter and Metal graph executor currently produce `0.000000`.
- The first known divergence is in the relative-attention region around reduce node `1418` over multiply node `1405`.
- `Metal_Gliner_Next_steps.md` narrows node `1405` to `deberta_graph.zig` `positionToContent`: `rel_tiled * kc_flat`, with the suspect operand being the kc broadcast `{24,1,64,64} -> {24,64,64,64}`.
- Existing evidence points to either a true element-order/permutation bug, a materialization stride bug, or an axis-convention mismatch between multiply operands.
- The next diagnostic should not be another order-blind checksum. The sharper sequence is: guard bisect, structural broadcast invariant, native-vs-Metal raw buffer forensics, and only then a Python-side hook if those disagree.

The codebase already has useful parity surfaces:

- `zig/pkg/inference/scripts/compare_gliner2_lora_python_zig.py`
- `zig/e2e/inference/test_gliner2_lora_parity.py`
- `zig/pkg/inference/scripts/run_gliner2_metal_train_parity.sh`
- `zig/pkg/inference/src/finetune/run_gliner2_production_readiness.zig`
- `zig/pkg/inference/src/finetune/gliner2_run_validation.zig`

But the current working tree also has a large amount of diagnostic-only Metal code in:

- `zig/pkg/inference/src/ops/metal_compute.zig`
- `zig/pkg/inference/src/backends/metal_tensor.zig`
- `zig/pkg/inference/src/graph/interpreter.zig`
- `zig/pkg/inference/src/graph/training.zig`
- `zig/pkg/inference/src/graph/metal_partition_executor.zig`

That code is valuable for finishing the bug hunt, but it cannot be the final production shape unless it is moved behind stable debug APIs, tested, or removed.

## Important Architecture Call

Gemma 4 is not a drop-in GLiNER2 backbone replacement in this codebase. GLiNER2 upstream is built around an encoder model, currently DeBERTa-style, with schema-conditioned span, classification, and count heads. A Gemma-style decoder/backbone swap would be a separate model architecture project, not a parity step.

For this parity effort, keep the scope on upstream GLiNER2 semantics first. Treat Gemma 4 only as a future teacher model, distillation source, or separate extraction model, not as the immediate GLiNER2 parity target.

## What Is Missing

### 1. Metal correctness is not done

The remaining Metal bug is still in the forward graph before the loss. The exact-zero loss likely means the wrong logits/masks reach the loss, not that the model is learning perfectly.

Needed:

- Guard bisection of `logicalStridesOrContiguous`, structural broadcast invariants, and native-vs-Metal raw buffer forensics at the first divergent multiply.
- A CPU reference for the exact production tensors at the failing op, using the same logical metadata as the Metal operands.
- A regression test that fails on the current production tensor ordering/pairing, not a modeled mini-chain that already passed.
- Acceptance on both Metal paths:
  - Metal interpreter.
  - `TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1` graph executor.

### 2. The parity envelope is too narrow

Native parity is strong, but much of it is scoped to controlled fixtures and pinned Python behavior.

Upstream Python GLiNER2 supports:

- Entities.
- Classifications.
- JSON structures.
- Relations.
- `classification_loss`, `structure_loss`, `count_loss`.
- Stochastic schema sampling and label/field/entity dropping.
- LoRA through PEFT target resolution.
- Scheduler choices: linear, cosine, cosine restarts, constant.
- Warmup.
- Evaluation.
- Save-best and checkpoint retention.
- Early stopping.
- FP16/BF16 where supported.

The local compare script already notes the risk: non-entity task annotations are detected, but some comparisons are still scoped or warning-only. The production parity goal should make those gaps explicit gates.

### 3. Metal parity is not integrated as the main Python-vs-Zig gate

`compare_gliner2_lora_python_zig.py` supports `--zig-backend metal` and has Metal readiness checks, but the canonical CI test currently uses `--zig-backend native`. The Metal script has useful smoke/readiness gates, but it is not yet the same apples-to-apples Python parity gate.

Needed:

- A strict Metal parity mode that compares Python vs Zig Metal for:
  - Preprocessing.
  - Component losses.
  - Step losses.
  - Adapter round-trip.
  - Optimizer state where practical.
  - Nonzero Metal graph dispatches when graph executor is requested.
- A failure if the Metal run silently falls back to interpreter-only or host-heavy execution.

### 4. Production trainer behavior is only partially aligned

Zig `train-gliner2-autodiff` has CLI support for eval strategy, eval steps, save-best, gradient accumulation, and compiled-required, but there are still gaps relative to upstream Python:

- No real held-out eval loop wired through the main CLI; current comments say eval/save-best uses average training loss over the window.
- Scheduler support is effectively constant LR in the parity path.
- Warmup/cosine parity is not a first-class GLiNER2 gate.
- Early stopping is not surfaced for GLiNER2 training.
- Mixed precision parity is not defined.
- Production checkpoint semantics need to match PEFT-native adapter expectations, not only the legacy adapter shape.

### 5. Performance has not started because correctness is blocking it

The notes say executor warm step is roughly `3.0s/step` while Python CPU is about `0.17-0.26s/step`, with a theoretical target below `50ms/step`.

Likely performance blockers:

- Per-step graph rebuild or replan.
- Host materialization around graph outputs and debug reads.
- Interpreter fallbacks inside the training graph.
- Incomplete DeBERTa encoder frame residency.
- Dot/general and relative-attention paths not consistently using the best resident Metal kernels.
- Device optimizer correctness is present enough to report `optimizer_backend=metal`, but it still needs throughput validation under nontrivial multi-step workloads.

## Recommended Plan

### Phase 0: Freeze the evidence and avoid scope drift

Do this before the next bug-fix attempt.

1. Capture the exact current failing command and output into a small artifact under `/private/tmp` or `output/`.
2. Record:
   - Git SHA and dirty diff stat.
   - Command line.
   - First-step loss.
   - Graph executor metrics.
   - Relevant env vars.
3. Keep the known-good native parity command as the baseline.
4. Do not change preprocessing, optimizer, or loss code while debugging Metal tensor ordering unless the pairing probe names that layer.

Acceptance:

- Repro command deterministically gives Metal loss `0.000000`.
- Native path still gives the Python-matching loss.
- The failing op/node IDs remain stable enough to instrument.

### Phase 1: Finish Metal correctness

Primary task: resolve the relative-attention multiply pairing/order defect. `Metal_Gliner_Next_steps.md` improves the order of operations here: use the cheapest decisive experiments before adding broader pairing probes.

Steps:

1. Run the `logicalStridesOrContiguous` guard bisect.
   - The old/new semantics flip changed three null-return sources at once: index-map, rank mismatch, and dim mismatch.
   - Build three temporary variants, each disabling exactly one source.
   - Run the production repro for each variant.
   - If one variant flips loss `0.000000 -> 19.230522`, that names the defective fallback path.
   - If none flips alone, test pairs to find the interaction.
2. Add the structural-invariant probe to `TERMITE_DUAL_READ_MUL`.
   - For the kc broadcast operand, `a[b,qi,ki,d]` must be constant as `qi` changes.
   - Check coordinates such as `(0,0,1,0)` vs `(0,5,1,0)`, plus single-axis variations.
   - If qi-constancy breaks only under new semantics, focus on reshape/broadcast fallback stride pairing.
   - If it breaks under both semantics, focus upstream on the device transpose around node `1328`.
3. Dump the full operand-a buffer three ways:
   - Native CPU backend as truth. This backend already has strict Python parity and runs the same graph node.
   - Metal-old semantics.
   - Metal-new semantics.
   - Use a small numpy forensics script to recover the `truth -> wrong` permutation over factorizations `{24,64,64,64}`, `{2,12,64,64,64}`, and `{1536,64,64}`.
   - Treat the permutation structure as the diagnosis; it should name the wrong stride composition directly and determine whether the previous "ground truth" checksum was a probe artifact.
4. Only if the three probes disagree, add a Python-side hook at the p2c/c2p score-level tensor.
   - HF DeBERTa does not necessarily materialize the `[bh,S,S,D]` intermediate, so compare score-level `[24,64,64]` tensors with scale matching.
   - Also verify whether Zig's simplified attention contract in `deberta_graph.zig` is semantically intended to match that score.
5. If still needed, add a temporary pairing-level probe at multiply node `1405`.
   - Print logical shape, physical shape, view strides, logical offset, and device byte offset.
   - Print `a[i]`, `b[i]`, and product for strategic indices around row, head, and batch boundaries.
6. Decide the actual bug class:
   - `a` is misordered.
   - `b` is misordered.
   - both are internally correct but paired with incompatible axis conventions.
   - the previous "ground truth" checksum was produced by a bad view read.
   - Zig's relative-attention graph contract is wrong despite native parity appearing to pass through the tested surface.
7. Fix only the named bug.
8. Add a regression test using the production-shape tensor path, not a simplified modeled chain.

Acceptance:

- Metal interpreter first-step loss matches `19.230522` within `1e-3`.
- Metal graph executor first-step loss matches `19.230522` within `1e-3`.
- Four-step acceptance from the campaign note matches:
  - `19.230522`
  - `13.608498`
  - `14.701510`
  - `14.222713`
- `grad_norm` is nonzero.
- No exact-zero loss on a supervised batch.
- Deterministic repeat count is at least three for any number used to make the fix decision.
- Diagnostics that change graph output liveness use the existing keep-output path, then final verification runs without diagnostic graph-shape changes.

Do not pursue these while Phase 1 is open:

- Split semantics by context. Executor mode is also wrong and reaches the same prim ops through fallbacks; this would mask the bug.
- Precompute the relative-bias chain. It is weight-dependent because the encoder is a LoRA target and gradients must flow through it.
- More modeled-chain fixes without production-shape validation.

### Phase 1A: Run the cost waterfall in parallel

This can run independently of the correctness fix as long as results are treated as diagnostic, not production performance claims.

Run one discriminating profile at batch 2 and batch 16, and re-time Python at both batch sizes:

```bash
cd zig/pkg/inference
TERMITE_METAL_PARTITION_LOOP_PROFILE=1 \
TERMITE_METAL_PARTITION_OP_STATS=1 \
TERMITE_METAL_TRACE_FRAME_LIFECYCLE=1 \
TERMITE_COMPILED_TRAIN_TRACE=1 \
TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 \
zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- ${=COMMON}
```

The waterfall should split the current warm `2.9-3.4s/step` into:

- GPU execution time.
- Host bookkeeping over the roughly 9,958-node graph walk.
- Region dispatch overhead across roughly 218 region dispatches.
- Mid-frame flush drains caused by host reads.

Competing hypotheses:

- H1: host round-trips cause mid-frame submit/wait/begin drains.
- H2: region dispatch scheduler overhead dominates.
- H3: per-node host bookkeeping dominates.

The output of this phase decides the order of the performance work. Do not optimize blindly before this run.

### Phase 2: Promote Metal into the Python parity gate

Once Phase 1 passes, extend the existing compare harness instead of creating a parallel one-off script.

Steps:

1. Make this command a first-class strict gate:

   ```bash
   cd zig/pkg/inference
   python3 scripts/compare_gliner2_lora_python_zig.py \
     --strict \
     --deterministic \
     --zig-backend metal \
     --zig-training-graph-executor \
     --zig-objective gliner2-total-loss \
     --steps 1 \
     --batch-size 2 \
     --seq-len 64 \
     --max-span-width 4 \
     --lora-rank 4 \
     --lora-alpha 8 \
     --span-loss-reduction sum \
     --span-positive-weight 1 \
     --span-negative-weight 1 \
     --span-hard-negative-weight 1 \
     --span-negative-mask-rate 0 \
     --seed 42
   ```

2. Ensure `--strict` fails if:
   - Metal backend was requested but manifest backend is not Metal.
   - Graph executor was requested but command/planned dispatches are zero.
   - `graph_executor_fallback_reason` is non-empty.
   - Interpreter fallback count exceeds an explicit threshold.
   - Device-resident transfer count is nonzero for trainables.
   - Component loss, step loss, or adapter round-trip fails tolerance.
3. Add a pytest sibling to the native parity test that is skipped unless Metal and the model bundle are present.

Acceptance:

- Native parity test remains green.
- Metal parity test is green on a Metal machine.
- The report JSON clearly states Python elapsed, Zig elapsed, graph executor dispatches, fallbacks, host outputs, and optimizer backend.

### Phase 3: Expand parity beyond entity-only fixtures

The current all-task fixture is the right seed, but the gate must graduate from warning coverage to required behavior.

Steps:

1. Build separate fixtures for:
   - Entity extraction.
   - Classification-only.
   - JSON structure extraction.
   - Relation extraction.
   - Mixed all-task batches.
   - Multi-instance structures with `count > 1`.
   - Empty/negative examples.
   - Partial final batches.
2. For each fixture, compare:
   - Preprocessed token IDs and attention mask.
   - Text word indices.
   - Schema special indices.
   - Task type order.
   - Positive target counts.
   - Component losses.
   - Total loss.
3. Keep stochastic sampling disabled for deterministic gates.
4. Add a second "sampling parity" mode only after deterministic parity passes:
   - Seed Python and Zig the same way.
   - Either replicate Python's random choices exactly or define that Zig production uses deterministic/no-sampling mode.

Acceptance:

- Required components are no longer warning-only for all-task fixtures.
- `classification_loss`, `structure_loss`, and `count_loss` all match Python where applicable.
- Relation and JSON structure examples pass component and total loss parity.

### Phase 4: Align production trainer semantics

After core losses are correct, close the trainer behavior gap.

Steps:

1. Add scheduler parity:
   - constant
   - linear warmup/decay
   - cosine
   - cosine restarts if required
2. Add real eval-data support to `train-gliner2-autodiff`:
   - `--eval-data`
   - eval batch size
   - `--eval-strategy epoch|steps|none`
   - `--eval-steps`
3. Wire save-best to eval loss, not the training-window average.
4. Add early stopping if the production workflow needs upstream feature parity.
5. Verify PEFT-native adapter compatibility:
   - Zig-written adapter loads into Python.
   - Python-written adapter loads into Zig.
   - Config contains enough metadata for downstream PEFT consumers.
6. Define the mixed-precision policy:
   - Native parity stays fp32.
   - Metal production can use fp32 first.
   - FP16/BF16 only become production gates after fp32 parity is stable.

Acceptance:

- A production training run can use train/eval files and save `best` and `final` artifacts with clear manifest metadata.
- Scheduler state, optimizer step counts, and checkpoint load/resume behavior are testable.

### Phase 5: Performance work

Start production performance claims only after the Metal correctness gate is stable. The Phase 1A waterfall can run earlier, but it is only a diagnostic input until correctness passes.

Primary target sequence:

1. Get below `1.0s/step` at batch 2 by removing host round-trips and obvious fallback paths.
2. Make a credible comparable-performance claim at batch 16: `step_time / 16 <= Python_time / 16`, with Python re-timed at batch 16.
3. Get below `100ms/step` at batch 2 through plan replay and fused attention.
4. Then chase the theoretical `<50ms/step` region if the graph shape and hardware make it realistic.
5. Compare against Python CPU and Python GPU separately. Do not mix those baselines.

Work items:

1. Remove correctness diagnostics from measured builds.
2. Ensure graph plans are cached/reused across steps.
3. Eliminate interpreter fallbacks in hot training regions.
4. Reduce host outputs to required final outputs only.
5. Keep LoRA trainables and Adam state resident on Metal.
6. Implement device `scatter_add`.
   - Current evidence says `executeRuntimeScatterAdd` does CPU download/loop/re-upload despite capability claims.
   - Treat rank-2/axis-0/f32 as the first production kernel.
   - Verify the four production losses after landing it.
7. Implement device `convert_dtype`.
   - This should be a small cast kernel and removes another host round-trip class.
   - Verify the four production losses after landing it.
8. After Phase 1 correctness is fixed, implement device index-map broadcast for the `{1,12,...} -> {2,...}` host-repeat path.
   - This touches the buggy chain, so do it only after the correctness fix is locked.
9. Audit `hostFallbackReshape` and `DotGeneral` survivors based on op stats.
10. Profile DeBERTa encoder frame residency:
   - encoder layer plan attempts/successes/reuses
   - relative QK pair fallbacks
   - FFN fused fallbacks
   - attention GEMM/legacy/flash calls
11. Add a plan-replay loop for the warm path.
   - Precompute alias/free decisions into cached runtime region plans.
   - Avoid a per-node branch-heavy host walk over the whole graph each step.
12. Route DeBERTa attention through fused attention where the graph shape and capability gates allow it.
13. Shrink plan-slot memory only if the waterfall shows allocation churn or memory pressure.
14. Benchmark with warmup and measured iterations, not a single cold compile step.

Strategic non-goal:

- Do not chase the current "runtime frame" eligibility machinery as the training solution. `Metal_Gliner_Next_steps.md` identifies it as inference/decode-shaped. The likely end-state is a compiled training step: per-shape cached plan replayed as one event-ordered command buffer, not a Gemma decode frame retrofit.

Acceptance:

- ReleaseFast benchmark report includes:
  - mean, median, p95 step time
  - first-step compile/build time separated from warm step time
  - graph dispatch count
  - fallback count
  - host output count
  - Metal GPU/wait time
  - supervised tokens/sec
- Performance gates fail on silent fallback.
- Batch-16 per-example time is no worse than the re-timed Python batch-16 baseline before claiming comparable performance.
- Final performance gates include both correctness and timing; no benchmark is valid if the four production losses drift.

### Phase 6: Production hardening

Before merge/release:

1. Strip or quarantine diagnostic-only code:
   - `TERMITE_TRACE_BUF_BIRTH`
   - `TERMITE_DUAL_READ_MUL`
   - `TERMITE_DUAL_READ_REDUCE`
   - order checksum helpers
   - raw-device debug helpers unless promoted to stable debug API
   - graph abs/zero traces unless already general-purpose
2. Keep useful diagnostics behind documented env vars with tests.
3. Add a short `GLiNER2_METAL_PARITY.md` or update existing docs with:
   - required model bundle
   - parity venv
   - canonical native gate
   - canonical Metal gate
   - benchmark command
   - expected acceptance numbers
4. Run formatting and focused tests:

   ```bash
   cd zig/pkg/inference
   zig fmt --check src/finetune src/graph src/ops src/backends
   zig build test-finetune -Dmetal=false --summary failures
   zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- --help
   scripts/run_gliner2_metal_train_parity.sh --suite
   python3 scripts/compare_gliner2_lora_python_zig.py --strict --zig-backend native ...
   python3 scripts/compare_gliner2_lora_python_zig.py --strict --zig-backend metal --zig-training-graph-executor ...
   ```

Acceptance:

- No diagnostic print spam in normal runs.
- CI has native parity.
- Metal parity is opt-in but strict and documented.
- Production readiness gate does not use `--allow-flat-loss` for the final production claim.
- Propose a commit series after Phase 1 closes: fixes plus tests first, durable tooling second, diagnostic cleanup last.

## Timeline And Done Criteria

The more operational timeline from `Metal_Gliner_Next_steps.md` is the better execution target:

| Week | Deliverable |
|---|---|
| W1 | Metal correctness closed on both paths, cost waterfall captured, device `scatter_add` and `convert_dtype` landed if waterfall confirms round-trips matter |
| W2 | Host round-trip phase complete, batch-16 comparable-performance claim, strict Metal gate green on a Metal CI or opt-in machine |
| W3-W4 | Below `100ms/step` at batch 2 if plan-replay and fused-attention work lands, plus cleanup and commit series |

Done means:

- `compare_gliner2_lora_python_zig.py --strict --zig-backend metal` exits 0.
- The Metal run has nonzero graph dispatches when graph executor is requested.
- Native and Metal gates both pass.
- The perf gate enforces step time at or below Python per-example time at batch 16.
- The production readiness claim does not depend on `--allow-flat-loss`.

## Risks In The Current Approach

1. Checksums can lie for order bugs.
   Abs-sums and sampled values are useful but insufficient. Guard bisection, structural invariants, native raw-buffer truth, and pairing-aware probes are required.

2. Mode changes can hide the bug.
   Parity diagnostics can alter graph liveness and fusion. Keep production-shape outputs alive with the existing keep-output env when needed, but verify final behavior without diagnostic graph changes.

3. Mini-chain unit tests have been overtrusted.
   The production bug survived several modeled-chain fixes. New regression coverage should preserve the production metadata and shape path.

4. "Metal backend" does not automatically mean "Metal training."
   The final gates must check optimizer backend, dispatch counts, fallback reasons, trainable residency, and host outputs.

5. Python parity has two levels.
   Deterministic, sampling-disabled parity is the first level. Full upstream production behavior includes stochastic schema/data augmentation and broader trainer features.

6. Performance numbers are not meaningful until correctness is fixed.
   Keep performance measurements separate from diagnostic builds and first-step compile costs.

7. The simplified Zig attention contract may itself be the issue.
   If native-vs-Metal forensics show both Metal semantics are wrong relative to native truth, the fix target moves into `deberta_graph.zig`, not Metal view materialization.

8. The waterfall may disprove the expected bottleneck.
   If region scheduler overhead dominates instead of host round-trips, re-rank Phase 5 toward region merging and plan replay before individual host-round-trip kernels.

## Immediate Next Action

Run the guard-bisect and structural-invariant probes before another broad fix. The pairing-level probe remains useful, but it should come after the cheaper decisive checks from `Metal_Gliner_Next_steps.md`.

```bash
cd /Users/timkaye/Documents/af/antfly/zig/pkg/inference
TERMITE_COMPILED_TRAIN_TRACE=1 \
TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=0 \
zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- \
  --model-dir /private/tmp/termite-models/gliner2 \
  --train-data /tmp/gliner2_metal_diag.jsonl \
  --epochs 1 \
  --batch-size 2 \
  --max-examples 2 \
  --seq-len 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0 \
  --backend metal \
  --objective gliner2-total-loss \
  --max-span-width 4 \
  --span-loss bce \
  --span-loss-reduction sum \
  --span-positive-weight 1.0 \
  --span-negative-weight 1.0 \
  --span-hard-negative-weight 1.0 \
  --span-negative-mask-rate 0.0 \
  --lora-rank 4 \
  --lora-alpha 8.0 \
  --lora-dropout 0.0 \
  --lora-targets encoder,span_rep,classifier,count_embed,count_pred \
  --seed 42 \
  --lora-only-trainables \
  --deterministic \
  --initial-adapter-checkpoint /private/tmp/gliner2-metal-probe/python/initial_adapter/adapter_weights.safetensors \
  --out-dir /tmp/gliner2-metal-guard-bisect
```

Use that command shape for each guard variant and for the invariant probe. Only after the guard/invariant/native-dump evidence names the actual defect should the code be changed.

## Source Notes

Local files reviewed:

- `Metal_Gliner2_Claude.md`
- `Metal_Gliner_Next_steps.md`
- `zig/pkg/inference/scripts/compare_gliner2_lora_python_zig.py`
- `zig/e2e/inference/test_gliner2_lora_parity.py`
- `zig/pkg/inference/scripts/run_gliner2_metal_train_parity.sh`
- `zig/pkg/inference/src/finetune/train/train_gliner2_autodiff.zig`
- `zig/pkg/inference/src/finetune/gliner2_data.zig`
- `zig/pkg/inference/src/finetune/run_gliner2_production_readiness.zig`
- `zig/pkg/inference/src/finetune/gliner2_run_validation.zig`
- `zig/pkg/inference/src/ops/metal_compute.zig`
- `zig/pkg/inference/src/backends/metal_tensor.zig`

Upstream Python GLiNER2 references:

- https://github.com/fastino-ai/GLiNER2
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/training/trainer.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/model.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/training/lora.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/processor.py
