# Metal GLiNER2 Fine-tuning — Next Steps Plan (v2, merged)

> **Production batch-32 correction 2026-06-14 — in progress.**
> The current batch-32 OOM class is not a Python/Zig numerical mismatch; it is
> the Metal training executor's single-frame lifetime model retaining thousands
> of dead private intermediates until frame completion. The corrective path is
> now explicit: in-frame private-buffer reuse is default-ON in the Metal runtime
> (escape hatch: `TERMITE_METAL_BUFFER_REUSE=0`), per-step JSONL telemetry reports
> Metal live/runtime/frame-retained/reuse counters, and the stable regression
> entrypoint is `scripts/run_gliner2_metal_train_tests.sh batch32`. That gate
> forwards true `--batch-size 32` into `gliner2-production-readiness`, defaults
> to `seq_len=128`/32 train examples, enables reuse stats, and requires allocator
> reuse hits so a future accidental disablement fails loudly. Frame chunking is
> diagnostic/fallback work only until it shows measured peak-memory wins.

> **✅ PRODUCTION PUSH 2026-06-11 — OOM FIXED + restructure shipped (default-on).**
> The machine-OOM (Metal at realistic seq≥128 exhausting 16GB) was the
> **`[bh,S,S,D]` disentangled-attention materialization**. Replaced C2P/P2C with
> the HF contraction-over-`num_rel` + Toeplitz score-gather form
> (`deberta_graph.zig`: `relScoreGatherEnabled`/`buildRelScoreIndices`/
> `contentToPositionGather`/`positionToContentGather`), now **default-ON**
> (escape: `TERMITE_DEBERTA_REL_SCORE_GATHER=0`). Validated: native+metal entity
> strict Python gates **PASS** with it on (all 7 metal-readiness checks green);
> native==metal, graph==direct exact, legacy==restructured 8-step trajectory
> bit-identical; metal now runs at s128 **and** s256 (the original crash config)
> where the legacy path OOMed; ~28% faster GPU at b2s64. Also landed:
> **fit-to-data effective seq-len** (pads to actual tokens, not fixed `--seq-len`;
> loss-neutral — seq40==seq64 bit-identical; escape:
> `TERMITE_GLINER2_DISABLE_FIT_SEQ_LEN=1`), **memory pre-flight rails**
> (`--allow-large-memory` + `estimateTrainingPeakBytes`, refuses configs >60% RAM),
> a **memory-pressure watchdog** (`scripts/gliner2_memory_watchdog.sh`) and
> **validation ladder** (`scripts/run_gliner2_validation_ladder.sh`), plus the
> device-side `lazy_multiply` materialization in `metal_compute.zig`. Known
> remaining gap (out of scope here): 6 **multi-schema** full-task parity fixtures
> (cls/json/rel/alltask/multicount/negative) still FAIL on `structure_loss` — the
> documented Phase 5 envelope-expansion work, pre-existing (native backend,
> rel-gather off, fit-to-data isolated out). Wall-clock at b1 s128 (~2.1s GPU) is
> still slower than Python CPU (~167ms) — the known kernel-efficiency gap.


> **⚠️ STATUS UPDATE 2026-06-11 — Phase 0 is CLOSED; its premise was wrong.**
> There is **no node-1405 correctness bug**. The Phase 0 target loss `19.230522` was an
> **artifact of the OLD pre-§2.5 broken index-map-aliasing semantics**, not a Python value.
> Verified directly: upstream Python GLiNER2Trainer computes the correct loss, native Zig matches
> it to ≤1e-9, and **Metal (interpreter AND graph executor) reproduces native step-for-step on
> every fixture tested** — degenerate (~0) and real-signal (loss 450 entity, ~11 all-task). Where
> native matches Python, Metal matches Python; where native diverges (fixture/config parity
> characteristics), Metal diverges identically. The §2.5 nullable-return fix was CORRECT. The ~12
> "failed fixes" were failing to reproduce a WRONG number. See `Metal_Gliner2_Claude.md` §3 banner
> for the full evidence. **Phase 0 ladder below (0.0–0.4) is retained for historical context only;
> do not execute it.** Phase 1 (strict Metal gate) has LANDED. Remaining real work: Phases 2–7
> (performance, parity-envelope expansion, trainer semantics, hardening). Determinism note for any
> Python comparison: always pass `--disable-python-model-dropout`, else Python loss varies
> run-to-run and fakes "Metal divergence."

**Goal:** GLiNER2 LoRA fine-tuning on Metal at **numerical parity with Python** and
**comparable-then-better performance** (Python CPU ≈0.17–0.26 s/step @ batch 2; Metal executor
today ≈3.0 s/step warm — numerics now CONFIRMED CORRECT).
**Companion docs:** `Metal_Gliner2_Claude.md` (evidence dossier — read first),
`Gemma_4_Gliner_next_steps.md` (independent review; merged into this v2).
**Branch:** `gliner2_finetuning_parity`. **Plan v2: 2026-06-10. Phase 0 closed 2026-06-11.**

**Scope guardrail (from the independent review, adopted):** Gemma 4 is *not* a GLiNER2 backbone
replacement in this codebase — GLiNER2 is encoder(DeBERTa)+schema-conditioned heads. Treat Gemma
only as a future teacher/distillation source. This effort stays on upstream GLiNER2 semantics.

---

## Phase 0 — Close the correctness bug (BLOCKING; est. 2–4 days)

One bug remains: a value-preserving element-order (or operand-pairing) defect at the
`positionToContent` multiply (`deberta_graph.zig:349,392`; node 1405 = `rel_tiled * kc_flat`; the
suspect operand is the **kc-broadcast** `{24,1,64,64}→{24,64,64,64}`). Twelve fixes failed because
probes were order-blind or validated against modeled inputs.

### 0.0 Evidence freeze (1 hour; adopted from review)
Before any further fix attempt, snapshot to `output/metal-bug-evidence/`: git SHA + `diff --stat`,
exact failing command + output (Metal loss 0.000000, both paths), known-good native command +
output, graph-executor metrics, env vars. Rule: do **not** touch preprocessing/optimizer/loss code
while debugging tensor ordering unless the probes name that layer.

### 0.1 Guard bisect (½ day) — cheapest decisive experiment, never run
`logicalStridesOrContiguous` (`metal_compute.zig:1128`) has **three independent null sources**:
index-map (:1130), rank mismatch (:1134), dim mismatch (:1136–1139). The old/new flip toggled all
three at once. Three one-line builds, each disabling ONE source; run the production repro each
time. Whichever single guard flips loss 0 → 19.230522 names the defective fallback path.

### 0.2 Pairing + structural-invariant probe (½ day, same builds)
Combines our invariant with the review's pairing probe at node 1405, printing for BOTH operands:
logical shape, physical len, view strides/offsets, device byte offset, and `a[i]`, `b[i]`,
`(a*b)[i]` at: 0..8, first-row boundary, 64/4096/batch/head boundaries, deterministic randoms —
**plus the qi-constancy invariant**: operand `a` is a qi-axis broadcast ⇒ `a[b,qi,ki,d]` must be
constant in qi (`a[(0,0,1,0)]==a[(0,5,1,0)]`, varying one coordinate at a time). Run under both
semantics. Outcome map: qi-constancy broken under new semantics only → materialization stride
pairing; broken in both → upstream device transpose (node 1328); both operands internally
consistent but products differ → axis-convention pairing mismatch.

### 0.3 Permutation forensics vs NATIVE ground truth (1–2 days) — settles everything
Key insight: the **native CPU backend is CI-trusted at 1e-9 vs Python and runs the same graph** —
node 1405 is the same node; it IS the "CPU reference indexed by logical metadata" the review asks
for, already built and trusted. Dump the full 6,291,456-float operand-a buffer three ways: native
(truth), Metal-old, Metal-new (25 MB raw dumps). A numpy script recovers the permutation
`truth → wrong` over factorizations `{24,64,64,64}` / `{2,12,64,64,64}` / `{1536,64,64}` (values
near-unique). **The permutation's structure IS the diagnosis** — and it simultaneously resolves
whether the prior "ground truth" checksum was a probe artifact.

### 0.4 Fallback: Python-side hook (only if 0.1–0.3 disagree)
Hook at the p2c/c2p **score** level (`[24,64,64]`, Zig node ≈1430) via the harness's debug-hook
pattern (`compare_gliner2_lora_python_zig.py` ~:1314); HF DeBERTa never materializes the
`[bh,S,S,D]` intermediate. Also verify the simplified Zig attention's intended contract
(`tileRelEmbAcrossBatch` :407 / `positionToContent` :349).

### Rejected strategies (evaluated, code-grounded)
Split semantics by context (executor reaches the same prims via its 44 fallbacks; executor is also
wrong); precompute the rel chain (weight-dependent, LoRA-trainable); more unit-tested single-path
fixes without production validation (12 precedents).

### Fix + regression rule (adopted from review)
Fix only the bug the probes name. The regression test must preserve **production metadata and
shape path** (real tensors/views), not a modeled mini-chain — mini-chains passed 12 times while
production stayed broken.

### Acceptance (non-negotiable)
- Production repro prints **19.230522 / 13.608498 / 14.701510 / 14.222713** (±1e-3), nonzero
  grad_norm, **no exact-zero loss on a supervised batch**, on **BOTH** Metal paths (interpreter
  AND `TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1`).
- Deterministic ×3. Diagnostics that alter graph shape use `_PARITY_KEEP_OUTPUTS=1`; final
  verification runs WITHOUT diagnostic graph changes.

---

## Phase 1 — Promote Metal into the Python parity gate ✅ LANDED 2026-06-11

**Done.** The metal-readiness signals are now strict failures (not warnings) in
`scripts/compare_gliner2_lora_python_zig.py`, gated on `--zig-backend metal`:
`metal_manifest_backend_is_metal`, `metal_optimizer_backend_is_metal`,
`metal_device_resident_transfers_zero`, `metal_finite_step_loss`, and (graph-executor-gated)
`metal_graph_executor_dispatches_nonzero`, `metal_graph_executor_fallback_reasons_empty`,
`metal_interpreter_fallbacks_within_threshold`. New `--metal-max-interpreter-fallbacks` arg
(default 64; the step uses 44). Pytest sibling `test_gliner2_lora_metal_strict_parity`
(`zig/e2e/inference/test_gliner2_lora_parity.py`) mirrors the proven native config with
`--zig-backend metal --zig-training-graph-executor`, skips off macOS, and PASSES end-to-end (155s).
Also fixed a pre-existing `REPO_ROOT` path bug that was silently SKIPPING the whole parity suite
(native gate now runs and passes too). The canonical command and spec below are retained for
reference.

Extend the existing harness — no parallel one-off scripts. Canonical strict Metal gate:

```bash
cd zig/pkg/inference
python3 scripts/compare_gliner2_lora_python_zig.py \
  --strict --deterministic --zig-backend metal --zig-training-graph-executor \
  --zig-objective gliner2-total-loss --steps 1 --batch-size 2 --seq-len 64 \
  --max-span-width 4 --lora-rank 4 --lora-alpha 8 --span-loss-reduction sum \
  --span-positive-weight 1 --span-negative-weight 1 --span-hard-negative-weight 1 \
  --span-negative-mask-rate 0 --seed 42
```

`--strict` must FAIL if (merged gate spec): manifest backend ≠ Metal when requested; graph
executor requested but command+planned dispatches are zero; `graph_executor_fallback_reason`
non-empty; interpreter-fallback count exceeds an explicit threshold; device-resident transfer
count nonzero for trainables; component-loss / step-loss / preprocessing / adapter-round-trip
tolerance failures. Report JSON states Python elapsed, Zig elapsed, dispatches, fallbacks, host
outputs, optimizer backend. Add a pytest sibling to the native parity test (skipped unless Metal +
model bundle present). **This green = the "Metal at parity" claim.** Native gate stays green.

---

## Phase 2 — Performance: cost waterfall (½ day; **[indep] — may start during Phase 0**)

One discriminating run (instrumentation exists), batch 2 AND 16, ReleaseFast, warm steps measured
separately from first-step compile; re-time Python at both batch sizes (CPU; if a Python-GPU/MPS
baseline is ever added, report it separately — never mix baselines):

```
TERMITE_METAL_PARTITION_LOOP_PROFILE=1 TERMITE_METAL_PARTITION_OP_STATS=1 \
TERMITE_METAL_TRACE_FRAME_LIFECYCLE=1 TERMITE_COMPILED_TRAIN_TRACE=1 \
TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 zig build ... ${=COMMON}
```

- Loop profile (`metal_partition_executor.zig:1724`): GPU execution vs host bookkeeping
  (`stats_ns`/`alias_clone_ns`/`free_expired_ns`/`graph_plan_ns` over 9,958 nodes).
- Op stats rank the 218 region dispatches; frame lifecycle (`metal_kernels.m:4723`, flush :34923)
  counts mid-frame flushes (the step already uses ONE command buffer per partition — begin :864,
  submit+wait :1278 — so cost hides in host-read-forced drains).
- Hypotheses to discriminate: (H1) mid-frame flush drains; (H2) per-dispatch scheduler overhead
  (218 × 10–15 ms); (H3) per-node host bookkeeping. GPU FLOP time ≪50 ms is NOT the problem.
  **Do not optimize before this run.**

**MEASURED (2026-06-10, batch 2, warm, stats env ON):** `submit_frame_ms ≈ 2,846–2,878 (87%)` ·
`stats_ms ≈ 314 (10%; may be instrumentation self-cost — clean baseline pending)` ·
host encode ≈ 44–52 · boundary outputs ≈ 30–60 · graph_plan/alias/free < 2 ·
**fallbacks (scatter_add 32 + convert_dtype 12) = 0.3 ms TOTAL — perf-irrelevant.**
Interpretation: ONE submitted command buffer with ~5,099 GPU micro-commands ≈ 0.5 ms each of
GPU-timeline dispatch overhead. **Verdict: H2' (GPU-side per-command overhead inside the single
buffer). Re-ranking below applied.**

---

## Phase 3 — Performance: fewer/larger GPU commands → **<1 s/step @ batch 2** (RE-RANKED per waterfall)

**The waterfall overturned the original ranking**: host round-trips cost 0.3 ms; the step is 87%
GPU command-buffer wall (~5,099 micro-commands × ~0.5 ms dispatch overhead). New order:

| # | Item | Why | Where | Est. |
|---|---|---|---|---|
| 3.1′ | **Region/kernel fusion — cut GPU command count** (was Phase 4) | 87% of step = per-command GPU overhead in one buffer | bigger fused regions; route attention via `fused_sdpa` (`metal_capabilities.zig:323`); fused QKV (:7881); merge elementwise chains | 3–5d |
| 3.2′ | **stats/bookkeeping elimination on warm path** | 314 ms/step (verify vs clean baseline w/o stats env) | precompute into cached `RuntimeRegionPlan`; skip stats when disabled | 1d |
| 3.3′ | **Batch-16 amortization measurement** | command count ~batch-independent ⇒ 8× per-example amortization | measure; may deliver the ≤Python-per-example claim early | 0.5d |
| 3.4′ | (Deprioritized) device scatter_add/convert_dtype/index-map broadcast | residency hygiene, NOT speed (0.3 ms) | as originally specced | when convenient |

After EACH item: four production losses bit-stable + step-time delta recorded.
Then **batch-16 milestone (≈free, 1 day):** fixed overhead amortizes 8× —
**acceptance: step_time/16 ≤ Python_time/16 at batch 16** (the honest first "comparable
performance" claim). Review's interim target adopted: **<0.5 s/step** on the canonical small gate
before chasing further.

---

## Phase 4 — Performance: **<100 ms/step @ batch 2** (1–2 weeks)

Prereq: Phase 3 done; frame-lifecycle shows one submit per partition per step.
1. **Plan-replay loop** (~3d): precompute alias/free/stats decisions
   (`cloneOutputIfAliasedInputWouldBeFreed` :1224, `freeExpiredInputs` :1237, stats :1177-1217 —
   currently 9,958×/step on host) into the cached `RuntimeRegionPlan`; warm path becomes a
   branch-light replay.
2. **Fused attention** (~3d): route through `fused_sdpa` (`metal_capabilities.zig:323`); fused QKV
   patterns exist (:7881).
3. Memory: shrink the ~1 GB plan reservation (26 slots, :52) only if profiling shows churn.

**Strategic call (supersedes the review's frame-residency item):** do NOT chase runtime-frame
eligibility — `analyzeRuntimeFrameEligibility` (:2929-3000) encodes the gemma-decode inference
state machine and hard-fails training graphs (`missing_model_metadata`). DO profile encoder-layer
plan attempts/reuses and per-region kernel choice (the review's residency counters) — but the
architecture target is a **compiled training step** (per-shape cached plan replayed as one
event-ordered command buffer); plan-replay is the incremental road; a first-class "training frame"
is the post-Phase-4 follow-on.

**Benchmark protocol (adopted from review):** warmup + measured iterations; report mean/median/p95
step time, first-step compile separated, dispatch/fallback/host-output counts, Metal GPU vs wait
time, supervised tokens/sec; perf gates fail on silent fallback
(`benchmark_gliner2_lora_perf.py` median thresholds already exist — wire them).

---

## Phase 5 — Parity envelope expansion (parallel track, native-first; adopted from review)

Today's parity is entity-fixture-narrow; several comparisons are warning-only. Graduate them:
1. Fixtures: entity-only; classification-only; JSON structure; relations; mixed all-task;
   multi-instance structures (`count > 1`); empty/negative examples; partial final batches.
2. Per fixture, gate (not warn): token IDs/attention mask, text word indices, schema special
   indices, task-type order, positive target counts, `classification_loss`/`structure_loss`/
   `count_loss`, total loss.
3. Stochastic sampling stays disabled for deterministic gates; add a "sampling parity" mode only
   after deterministic parity passes (seed-replicate Python's draws, or declare Zig production
   deterministic/no-sampling).
This track runs on the **native** backend, fully parallel to Phases 2–4; re-run green fixtures on
Metal once Phase 1 lands.

---

## Phase 6 — Production trainer semantics (adopted from review; after Phase 1)

Close the trainer-behavior gap vs upstream:
1. **Schedulers**: constant / linear warmup-decay / cosine (/restarts if needed) — parity path is
   effectively constant-LR today.
2. **Real held-out eval**: `--eval-data`, eval batch size, `--eval-strategy epoch|steps`,
   `--eval-steps`; **wire save-best to eval loss** (today it uses training-window average).
3. Early stopping (if production workflow needs it).
4. **PEFT compat both directions**: Zig-written adapter loads in Python AND Python-written adapter
   loads in Zig; config metadata sufficient for downstream PEFT consumers.
5. **Mixed-precision policy**: native parity stays fp32; Metal production fp32 first; FP16/BF16
   become gates only after fp32 parity is stable.
Acceptance: production run uses train+eval files, saves `best`+`final` with manifest metadata;
scheduler state, optimizer step counts, checkpoint load/resume are testable.

---

## Phase 7 — Production hardening (before merge)

1. Strip or quarantine diagnostic-only code (dossier §6 list: `TERMITE_TRACE_BUF_BIRTH` block in
   `denseBuf`, `dual_read_*`, `orderChecksum`, `[abs]`/`[graph-zero]` traces); keep durable
   diagnostics behind documented env vars with tests. No print spam in normal runs.
2. Commit series for user approval: fixes+tests first, then tooling. (~1,800 uncommitted lines
   exist today.)
3. Docs: update `Metal_Gliner2_Claude.md` (or a `GLiNER2_METAL_PARITY.md`) with model bundle,
   parity venv, canonical native + Metal gates, benchmark command, acceptance numbers.
4. Final checklist:
   `zig fmt --check src/finetune src/graph src/ops src/backends`;
   `zig build test-finetune -Dmetal=false --summary failures`;
   `zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- --help`;
   `scripts/run_gliner2_metal_train_parity.sh --suite`;
   strict native gate; strict Metal gate.
   The production-readiness claim must NOT rely on `--allow-flat-loss`.

---

## Sequencing & timeline

| Week | Deliverables |
|---|---|
| W1 | 0.0–0.3 close the bug (both Metal paths print the four losses) · Phase 2 waterfall + 3.1/3.2 land in parallel |
| W2 | Phase 1 strict Metal gate green · Phase 3 complete (<1 s, then <0.5 s/step b2) · **batch-16 ≤ Python per-example claim** · Phase 5 fixtures started (parallel) |
| W3–4 | Phase 4 (<100 ms/step b2) · Phase 5 gates required-not-warning · Phase 6 trainer semantics · Phase 7 hardening + commit series |

**DONE means:** strict Metal parity gate green in CI (opt-in on Metal machines) with nonzero
dispatches and no fallback reasons · perf gate enforces ≤ Python per-example @ batch 16 with
fallback/host-output ceilings · parity envelope gates required for all-task fixtures · trainer
semantics (sched/eval/save-best/PEFT-both-ways) tested.

## Top risks (merged)

1. **Phase 0 reveals both Metal variants wrong vs native truth** (the old-semantics "ground truth"
   was a probe artifact) → 0.3 catches it; fix target moves into the deberta_graph chain; native
   gates bound the blast radius.
2. **Waterfall says dispatch latency dominates** (H2) → re-rank Phase 3 toward region merging;
   plan-replay moves up.
3. **False-fix loop #13** → acceptance rule + production-shape regression tests + 0.3's property
   that diagnosis and fix are the same artifact.
4. **"Metal backend" ≠ "Metal training"** → Phase 1's gate checks optimizer backend, dispatches,
   fallback reasons, residency, host outputs.
5. **Two levels of Python parity** → deterministic/no-sampling parity first (Phases 0–1);
   stochastic/upstream-production behavior is Phase 5/6 scope, gated separately.
