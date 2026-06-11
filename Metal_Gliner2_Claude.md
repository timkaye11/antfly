# Metal GLiNER2 Training — Debugging Campaign Status

**Branch:** `gliner2_finetuning_parity` · **Last updated:** 2026-06-10
**Goal:** GLiNER2 LoRA fine-tuning (`train-gliner2-autodiff`, objective `gliner2-total-loss`) at
numerical parity with upstream Python GLiNER2, on the Metal backend, then fast.

---

## 1. Scorecard

| Area | Status |
|---|---|
| **Native backend (CPU Zig) ↔ Python parity** | ✅ **DONE, CI-gated.** Per-step loss ≤1e-9, adapter round-trip (weights 3.5e-10), multi-step optimizer-state parity, partial-batch handling. Gates: `compare_gliner2_lora_python_zig.py --strict`, `zig/e2e/inference/test_gliner2_lora_parity.py` (+ 3-step all-task round-trip test). |
| **Metal: infrastructure correctness** | ✅ 10+ real bugs fixed with regression tests (see §2). Reduce/broadcast/transpose/multiply GPU kernels independently verified correct against CPU references. |
| **Metal: end-to-end training loss** | ✅ **AT PARITY (2026-06-11).** Both Metal paths reproduce native step-for-step; native matches Python ≤1e-9. The prior "open bug" was a PHANTOM: the 19.230522 target was an artifact of the OLD pre-§2.5 broken index-map aliasing, not a Python value (see §3 banner). Strict Metal gate green (`test_gliner2_lora_metal_strict_parity`). |
| **Metal: performance** | ⏸ Not started (was mis-blocked on the phantom correctness bug). Executor ~3.0s/step warm vs Python CPU ~0.17s (~18x slow). FLOP analysis says <50ms/step is achievable. Now unblocked — Phase 2 waterfall is next. |

---

## 2. Fixed & verified (all in working tree / recent commits on this branch)

Each fix has a regression test and was verified by direct runs:

1. **Partial-batch gather OOB** — final batch smaller than graph batch crashed `gather`; fixed by
   padding with zero-masked rows (`train_gliner2_autodiff.zig`, `gliner2_real_autodiff.zig::zeroPaddedSpanTargetRows`).
   Partial-batch step losses match Python exactly.
2. **Optimizer step-count drift** — torch advances Adam `step` for exactly-zero grads (lora_A at
   step 1, B=0 init); Zig skipped them → bias-correction divergence ≈ 2×lr. Fixed via zero-grad
   step replay (`train_gliner2_autodiff.zig::syncZeroGradLoraOptimizerSteps`). Drift ↓ 11x.
3. **Executor: deferred-mul consume bug** — consume side treated materialized muls as deferred,
   recomputed from freed operands → `MissingRuntimeInput` → silent full-step interpreter fallback
   every step. Fixed (`metal_partition_executor.zig`: pendingDeferred* gates + on-demand
   materialization). Executor went from never-running to running (6.9s → ~3.0s warm).
4. **Executor: dot_general reroute** — `dot(x, transpose(W))` rewritten into linear-slot/MPS
   machinery, numerically ≠ `primDotGeneral` (sign-flipped rel-position projections, node 1035).
   Fixed: dense device-resident case routes through interpreter-equivalent `primDotGeneral`.
5. **view_index_map strides** — `logicalStridesOrContiguous` returned dense strides for index-map
   views; broadcast/transpose fast paths aliased raw bytes (node 1330 wrong). Fixed (nullable
   return + fallback materialization). **Note: this fix is executor-required AND is what exposed
   the still-open interpreter bug** (the fallback path's product differs from the old alias path).
6. **Output buffer lifetime** — graph outputs (loss + 180 grads) aliased recycled plan-slot device
   memory; extraction read zeros. Fixed: drain-then-copy into owned device buffers at partition end
   (`copyPartitionGraphOutputsToOwnedStorage`); plus elision override so graph outputs are never
   skipped as fused-interior nodes (95 of 181 outputs were never written).
7. **toFloat32 / toHostSlice hazards** — empty reads of pending lazy-multiply bufs; stale host
   mirrors returned without frame flush; in-place lazy materialization destroying shared operand
   pairs. All fixed (copy-on-materialize with per-buf cache; flush-on-cached-read; temporaries).
8. **Flat-wrap broadcast gating** — device rhs-repeat elementwise branches used `rhs[gid % len]`
   for any divisible operand; only valid when the secondary's shape is an exact suffix. Gated
   (`flatRepeatMatchesBroadcast`).
9. **Lazy-reduce geometry** — `tryDeviceLazyMultiplyReduceLastDim` used physical operand length
   where logical numel was required; host fallback couldn't read lazy bufs. Fixed.
10. **Stride/view materialization hardening** — output-traversal with stale cached
    `logical_view_strides`; raw-mirror reads in `logicalValueAtFlat`; resolved-shape pairing for
    rebound `logical_shape` (`materializeStrideViewWithResolvedShape`). Fixed (unit-tested), though
    none of these moved the production bug (§3).

Also: silent fallbacks made loud (`graph_executor_fallback_reason` in step metrics), production
gates hardened (compare script `--strict` fails on zero Metal dispatches; metal test script gate).

---

## 3. The open bug — complete evidence dossier

> **⚠️ PREMISE INVERSION (2026-06-11) — READ FIRST.** Phase 0 diagnostics overturn this
> section's framing. Clean runs (no graph-altering env) on the §5 diag fixture give:
> **native = 0.000000, Metal new-semantics = 0.000000, Metal old-semantics = 19.230522.**
> Native AGREES with the supposedly-buggy Metal value (0.0); only the OLD (pre-§2.5-fix)
> semantics produces 19.230522. The guard bisect (`metal_compute.zig` logicalStridesOrContiguous)
> is decisive: disabling ONLY the index-map null source flips 0→19.23. The forensic dump of
> node 1381 (operand a) confirms native ordsum 6590725.0113 ≈ Metal-new 6590724.8501, with
> Metal-old the outlier at 6596058.8836. **So "19.230522 = correct, Python-verified" was never
> actually verified against Python on THIS fixture.** Adjudication in progress: running
> `compare_gliner2_lora_python_zig.py --zig-backend native` on the diag fixture to get Python's
> true loss. If Python ≈ 0.0 → there is NO Metal bug (new semantics already matches native+Python;
> 19.23 was the artifact). If Python ≈ 19.23 → native is ALSO wrong here (shared graph-layer bug,
> not Metal-specific) and the strict gate doesn't cover this path. Do not act on the text below
> until this resolves.
>
> **✅ RESOLVED (2026-06-11): NO METAL BUG. The 19.230522 target was the artifact.**
> Ran `compare_gliner2_lora_python_zig.py --zig-backend native` on the diag fixture. Python's
> OWN upstream GLiNER2Trainer reports: classification_loss=0.0, count_loss=0.0,
> structure_loss=1.17e-9, **total loss=1.17e-9** ("Epoch 1/1 - Loss: 0.0000"). Zig-native matches
> (delta -1.17e-9, step_loss_parity_matches=true). So Python = native = Metal-new ≈ 0; the current
> Metal code is ALREADY at parity. 19.230522 came ONLY from the OLD pre-§2.5 broken index-map
> aliasing — corrupted attention bias → saturated logits → large spurious BCE. The §2.5 nullable
> fix was CORRECT; the ~12 "failed fixes" were failing to reproduce a WRONG number. WHY ~0 is
> correct: at step 1 LoRA-B=0 (identity adapter), so loss = base gliner2-base-v1 loss on trivially
> easy NER examples (John→person, Paris→location) ≈ 0. **Caveat:** this fixture is a weak parity
> test (correct loss ~0). Robust validation needs a multi-step or harder-example config where
> Python yields a substantial nonzero loss, then confirm native + both Metal paths match it
> component-wise. That is the real remaining work — not a node-1405 fix.

### Symptom
Metal interpreter mode (and executor mode, which consumes the same tensors):
training loss prints **0.000000**; correct value **19.230522** (Python-verified; reproduced by the
old-semantics build, 3/3 deterministic).

### Causal chain (every step CAPTURED, not modeled)
1. Loss is *mathematically correct BCE* for the logits it receives — the logits are saturated
   (|x| up to 45) because one attention-bias tensor upstream is wrong.
2. First divergent traced node: **1418 `reduce_sum {1536,64,1}`** over **1405 `mul {1536,64,64}`**
   (relative-attention region). The reduce kernel is **proven correct** (CPU rowsum reference over
   the same downloaded bytes == kernel output to 4 decimals, both semantics).
3. The mul's product differs because **operand `a`** (logical `{1536,64,64}`) has the **right
   values in the wrong order**: abs-sum identical (3,679,270.4699), 8-element samples identical,
   but order-checksum differs — **wrong 6,590,724.8501 vs "ground truth" 6,596,058.8836**.
4. Birth stack trace (`TERMITE_TRACE_BUF_BIRTH`): the permuted buffer is constructed in
   **`hostFallbackReshape` → `exportCtFromHostNative` → `denseBuf`**, shape `{24,64,64,64}`.
5. Immediately preceding broadcast (captured): `{24,1,64,64} → {24,64,64,64}` axes `{0,1,2,3}`,
   input **device-resident** (`has_metal=true, data_len=0`), executed via
   **`tryDeviceBroadcastGeneral` (`path=general_device`)**; input is node 1329 (reshape) of node
   1328 (device transpose `perm {0,2,1}` of `{24,64,64}`).
   Also in chain: `{1,12,64,64,64}→{2,...}` broadcast takes `path=host_repeat` over an index-map
   view (`index_map_len=3145728`).

### The unmovable number (critical)
`6,590,724.8501` has been **bit-identical across ~12 fix attempts**, including fixes to the
stack-trace-named constructor and the captured broadcast path. Every fix passed its own unit
tests; the production checksum never moved.

### Open hypothesis (next to test)
The "ground truth" (6,596,058.8836) was measured by **probing** the old-semantics view through
`toFloat32` materialization — machinery since shown buggy. If the probe's view-read was wrong,
`6,590,724` may be a correctly-ordered buffer, and the real defect is a **pairing/axis-convention
mismatch between the two multiply operands** (each correct per its own metadata, misaligned with
each other). **Next probe:** print `a[i], b[i], product[i]` at strategic indices in BOTH semantics
at the multiply — identifies which operand misaligns *at the pairing* (order-sensitive,
pairing-aware; immune to the blind spots that wasted earlier cycles).

### Hard-won methodology lessons
- **Abs-sums and per-node value traces are order-invariant** — permuted bytes are invisible to them.
- **Unit tests of modeled chains repeatedly passed while production stayed broken** — only fixes
  validated against the *production checksum* count.
- **Parity diagnostics change the graph** (parity mode replaces/extends outputs → different
  liveness/fusion → bug disappears). Use `TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_KEEP_OUTPUTS=1`.
- The trainer prints loss=0 *exactly* whenever masked-sum losses see zero masks — exact-zero loss
  means "zero supervision reached the loss", not "small loss".

---

## 4. Diagnostic tooling built (env-gated; in working tree)

| Env var | What it does |
|---|---|
| `TERMITE_TRACE_BUF_BIRTH=1` | Stack-trace any `denseBuf` of len 6291456 whose ordsum matches the wrong/truth constants. **The only probe that never lied.** |
| `TERMITE_DUAL_READ_MUL=1` | Per-operand raw-device vs host-route abs + **order checksum** + first4/mid4 at every multiply dispatch path (`TERMITE_DUAL_READ_MUL_NUMEL`, default 6291456). |
| `TERMITE_DUAL_READ_REDUCE=1` | Same at the last-dim reduce + CPU rowsum reference (`debugRawDeviceAbsSum`, `debugRawDeviceRowSumAbs` in `metal_tensor.zig`). |
| `TERMITE_GRAPH_ABS_TRACE=1` | Per-node abs-sum lines `[abs] <id> <sum>` (with `TERMITE_GRAPH_FINITE_TRACE=1`). Order-invariant — use for divergence *sets*, not order bugs. |
| `TERMITE_GRAPH_ZERO_TRACE=1` | Flags all-zero node outputs. |
| `TERMITE_GRAPH_NODE_VALUES="id,id"` | Per-node CT pointer + first4 + abs. |
| `TERMITE_DUMP_GRAPH_NODES="a-b,c"` | Graph dump: node op/inputs/shapes/op-configs (in `training.zig`). |
| `TERMITE_METAL_TRACE_BROADCAST_PRIM=1` | **Every** exit path of `primBroadcastInDimOp` prints `enter`/`path=`/`decline=` with full input metadata. |
| `TERMITE_METAL_TRACE_REDUCE_PRIM=1` | Same for reduce prims. |
| `TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS="..."` (+`_KEEP_OUTPUTS=1`) | Per-node executor-vs-interpreter value parity. |
| Kill switches | `TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY`, `TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE`, `TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC`, `TERMITE_METAL_DISABLE_RUNTIME_REGION_PLAN`, `TERMITE_METAL_PARTITION_DISABLE_FUSED_PATTERNS`. |

**Semantics flip for A/B runs** (the bisect lever): swap `logicalStridesOrContiguous` in
`src/ops/metal_compute.zig` between current (nullable; executor-required) and old
(`if (buf.view_strides) |s| return s;` + contiguous; interpreter-correct). A python heredoc that
does the swap textually is in the session logs; old semantics ⇒ loss 19.230522.

---

## 5. Reproduction

Canonical run (from `zig/pkg/inference`; model at `/private/tmp/termite-models/gliner2`, fixture
`/tmp/gliner2_metal_diag.jsonl` = 8 examples (4× the first 2 lines of
`testdata/gliner2_ner_smoke.jsonl`), initial adapter
`/private/tmp/gliner2-metal-probe/python/initial_adapter/adapter_weights.safetensors`):

```bash
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

# Interpreter mode (the open bug — prints loss=0.000000, should be 19.230522):
TERMITE_COMPILED_TRAIN_TRACE=1 zig build -Dmetal=true -Doptimize=ReleaseFast \
  train-gliner2-autodiff -- ${=COMMON} --out-dir /tmp/repro

# Executor mode: add TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1
# Acceptance numbers: losses 19.230522 / 13.608498 / 14.701510 / 14.222713 (±1e-3),
# nonzero grad_norm; operand-a ordsum (TERMITE_DUAL_READ_MUL=1) — see §3 caveat re ground truth.
```

Full Python↔Zig strict parity gate (native backend, passes today):
see `zig/e2e/inference/test_gliner2_lora_parity.py` for the canonical invocation
(parity venv `/private/tmp/gliner2-parity-venv/bin/python`).

**Gotchas:** background shells lose cwd — always `cd /Users/timkaye/Documents/af/antfly/zig/pkg/inference`
in the same command; zsh does NOT word-split `$COMMON` — use `${=COMMON}`; watch disk space
(builds fail weirdly near-full); verify runs N≥1 with known-value acceptance, never "it ran".

---

## 6. Working tree state (uncommitted, on top of `2004e09ca`)

- `src/ops/metal_compute.zig` (+~1700): fixes §2.7–2.10 + all diagnostics + ~8 regression tests.
- `src/backends/metal_tensor.zig`: debug raw-download helpers + syncHostMirror.
- `src/finetune/train/train_gliner2_autodiff.zig`, `gliner2_real_autodiff.zig`,
  `real_autodiff_trainer.zig`, `gliner2.zig`: OOB padding + optimizer replay + fallback-reason
  plumbing (some committed in `86a589825`/`2004e09ca`).
- `src/graph/{training,interpreter,metal_partition_executor,multi_executor,partition,executor_stats}.zig`:
  executor fixes + parity/dump tooling (mostly committed).
- `scripts/compare_gliner2_lora_python_zig.py`, `zig/e2e/inference/test_gliner2_lora_parity.py`:
  strict gates + optimizer-parity dumps (committed).
- **Diagnostic-only code to strip before merge:** the `TERMITE_TRACE_BUF_BIRTH` block in `denseBuf`,
  `dual_read_*` helpers/prints, `orderChecksum`/`captureOrderSamples`, `[abs]`/`[graph-zero]`
  interpreter traces, `debugRawDevice*` (or keep behind clearly-named debug API).

## 7. Next steps

1. ~~Pairing-level probe~~ — DONE; it (plus guard bisect + permutation forensics + direct Python
   comparison) proved there is **no bug**: the "ground truth" 19.230522 WAS the artifact (§3 banner).
   The Phase-0 diagnostics (bisect guards, pairing probe, node-dump, forensics scripts) have been
   stripped.
2. ✅ Strict Python parity gate with `--zig-backend metal` — LANDED
   (`test_gliner2_lora_metal_strict_parity`, metal-readiness checks now strict).
3. **Performance phase (now unblocked, the real next work):** Phase 2 cost waterfall, then kill the
   44 per-op interpreter fallbacks, host materialization round-trips, per-step replan; target
   <0.5s/step then <50ms. See `Metal_Gliner_Next_steps.md` Phases 2–4.
4. Pre-existing diagnostic-only code still to strip before merge (separate from the now-removed
   Phase-0 scaffolding): the `TERMITE_TRACE_BUF_BIRTH` block in `denseBuf`, `dual_read_*`
   helpers/prints, `orderChecksum`/`captureOrderSamples`, `[abs]`/`[graph-zero]` interpreter traces,
   `debugRawDevice*` — or keep behind a clearly-named debug API (Phase 7 hardening).
