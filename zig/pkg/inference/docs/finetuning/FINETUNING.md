# Fine-Tuning in antfly-inference-zig

antfly-inference-zig supports training and fine-tuning through reverse-mode automatic differentiation on the graph IR. Features are implemented in Zig and run on CPU (BLAS), Apple Silicon (Metal/MLX), or supported NVIDIA GPUs through the CUDA backend. CUDA is only a runtime/build dependency when that backend is selected; the native and Metal paths do not require it.

The same-Mac harness now has separate fresh-process MLX-LM and Antfly runners
with a shared locked optimizer, workload, adapter, synchronization, memory,
and precision-evidence contract. Both runners have completed the same
diagnostic-only E2B sequence-128/rank-16/accumulation-1 cell. After the guarded
64-row simdgroup BF16 pass and the current eight-row sparse-loss vocabulary
projection batching, Antfly measured `0.470017 s` mean step time and
`1,808,025,952` bytes peak physical footprint; MLX-LM measured `0.263396 s`
and `11,497,430,816` bytes. Thus MLX is `1.784x` faster in that bounded cell
while using `6.359x` the process footprint. On the exact sequence-512 cell,
Antfly measured `1.466469 s` and `6,023,989,768` bytes versus MLX-LM's
`0.990706 s` and `15,999,361,256` bytes: MLX is `1.480x` faster while using
`2.656x` the footprint. The vocabulary batching is byte-identical to the
one-row rollback after one and five eight-target updates and after one
fifteen-target update. Both frameworks'
samples incurred paging, came from diagnostic producers, and do not form the
locked alternating multi-repeat matrix. The simdgroup path also introduces
small mixed-precision update drift and is not exact numerical parity. The MLX
runner still marks comparison-critical dtype surfaces `not_asserted`, so the
full release comparator rejects these samples. Treat the ratios as optimization
guidance, not release evidence.

---

## Unified Recipes

The recipe runner gives post-training one stable entry point:

```sh
antfly inference finetune run recipe.json
```

Compatible Gemma4 text DPO/GRPO jobs can retain one admitted model and CUDA or
Metal backend across the complete sequence:

```sh
antfly inference finetune run-suite \
  --report /runs/preference-suite.json \
  dpo-recipe.json grpo-recipe.json
```

`run-suite` requires at least two optimizer-backed text preference recipes
with the same canonical model path and backend. Each recipe must select a
distinct artifact root and trained-adapter directory. Before its first write,
the suite canonicalizes every input and output path and rejects cross-job
roots, adapters, manifests, reports, or suite-report collisions, including an
output that contains another job's model, dataset, recipe, projector, or
initial adapter. A job's artifact root may contain that same job's recipe and
dataset as strict descendants; concrete output files and adapter directories
must still be disjoint from them.
Without `--report`, the suite report is placed beside the first job's artifact
root rather than inside that job's namespace. The model/backend is
admitted exactly once; every job still gets fresh graph caches, LoRA tensors,
gradients, and optimizer state. Multimodal preference recipes remain
fail-closed here until their projector-backed trainers consume the same shared
ownership boundary. The v2 suite report records planned/completed runs, one
model admission, reuse hits, model-admission duration, total duration, and a
separate wall duration for every job. Each DPO/GRPO report records its session
run index, and text preference reports persist the exact token/sampling input
contract used by the trainer.

It follows the same direction as TRL's trainer taxonomy and Training Hub's algorithm-first routing layer: users choose an algorithm, keep common fields stable, and let antfly inference route to the existing family-specific prepare, train/eval, and materialize tools.

### Schema

All recipes use the same top-level sections:

```json
{
  "recipe": "lora-sft",
  "model": {
    "path": "/models/gemma4",
    "family": "gemma4"
  },
  "dataset": {
    "train_path": "/data/train.jsonl",
    "eval_path": "/data/eval.jsonl",
    "train_split": "train",
    "eval_split": "eval",
    "eval_cache_path": "/runs/gemma4/prepared_eval_inputs.json",
    "max_examples": 128,
    "eval_max_examples": 64,
    "max_seq_len": 512
  },
  "adapter": {
    "path": "/runs/gemma4/adapter-bootstrap",
    "rank": 16,
    "alpha": 32,
    "target_preset": "text-all-linear",
    "scaling": "standard",
    "init_lora_weights": "default",
    "use_dora": false
  },
  "optimizer": {
    "learning_rate": 0.0002,
    "epochs": 2,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0
  },
  "eval": {
    "max_examples": 64
  },
  "artifacts": {
    "root": "/runs/gemma4",
    "manifest_path": "/runs/gemma4/recipe_run_manifest.json",
    "prepared_path": "/runs/gemma4/prepared_inputs.json",
    "trained_adapter_dir": "/runs/gemma4/adapter-trained"
  },
  "backend": "native"
}
```

Supported recipe names across all families are `sft`, `lora-sft`, `qlora-sft`,
`dpo`, `grpo`, `reranker`, and `vlm-retrieval`. Gemma4 supervised training
currently admits only `lora-sft`; its full SFT and QLoRA spellings fail closed.
Recipe parsing is strict: unknown fields are errors rather than silently
defaulting a misspelled option.

Gemma4 `dpo` and `grpo` are optimizer-backed LoRA routes over BF16
SafeTensors bases. `backend = "metal"` or `backend = "cuda"` selects the
matching strict compiled-device trainer and never falls back to native
execution. Same-base Gemma4 DPO
precomputes frozen reference logprobs through the exact policy graph with LoRA
zeroed. Gemma4 GRPO uses the same compiled policy graph with immutable
zero-valued LoRA device bindings for its frozen reference. Initial
sampling/rescore and zero-adapter policy/reference parity are fail-closed.
Packed-GGUF QLoRA for preference optimization remains a typed, fail-closed
boundary pending end-to-end parity and quality evidence.

For a bounded real-checkpoint CUDA qualification, use
`scripts/run_gemma4_cuda_preference_smoke.py`. The runner accepts mounted model
directories, inspects config/index/SafeTensors headers before starting CUDA,
uses the memory-bounded `peft-qv` adapter preset, captures peak process GPU
memory, and requires the generated DPO/GRPO reports to prove strict compiled
execution with no fallback, host gradients, undeclared transfers, or readback
beyond one scalar loss per graph step. It never downloads or converts a model.
When both objectives are selected, the runner uses `run-suite` by default so
each model/repetition has one process and one model admission. Pass
`--isolated-processes` only for a cold-start control measurement.
The `gemma4-cuda-training-tests` CI lane runs strict CUDA unit, graph, build,
and device-smoke gates whenever repository variable `ANTFLY_CUDA_RUNNER`
names an NVIDIA-enabled runner. Optional `ANTFLY_GEMMA4_E2B_MODEL_DIR` and
`ANTFLY_GEMMA4_E4B_MODEL_DIR` repository variables enable the corresponding
bounded real-checkpoint suites without downloading model weights in CI.
Without `--grpo-target`, the qualification runner uses the deterministic
`ranked-first` reward: the first sampled rank receives `1` and every other
rank receives `0`. This proves optimizer behavior without assuming that E2B,
E4B, and future checkpoints share a top token. Pass an explicit target to use
the semantic `prefix-match` reward instead:

```sh
scripts/run_gemma4_cuda_preference_smoke.py \
  --model e2b=/models/gemma-4-E2B-it \
  --grpo-target e2b=Paris \
  --repetitions 3 \
  --out /runs/gemma4-cuda-e2b
```

Add `--matched-benchmark` for the locked Unsloth-comparison gate. It selects
rank-16/alpha-32 Q/V LoRA, sequence cap 128, and exactly 25 optimizer updates
per objective (`1` cold, `1` first, `3` warmup, `20` measured). DPO and GRPO
reuse one admitted model and one immutable initial Antfly adapter per model,
but retain isolated trainers and output adapters. The runner fails unless both
reports contain the exact input contract, fixed timing protocol, same-base
initial logprob parity, strict CUDA evidence, and positive per-job timings
separate from model admission:

```sh
scripts/run_gemma4_cuda_preference_smoke.py \
  --model e2b=/models/gemma-4-E2B-it-bf16 \
  --matched-benchmark \
  --out /runs/gemma4-cuda-e2b-matched
```

The matched GRPO recipe uses `rendered-text-grpo` deliberately: its plain-text
fixture is already the final decoder prompt, so Antfly and TRL both feed the
same five-token decoder sequence instead of Antfly adding a chat wrapper. The
report gate validates that actual model input. The gate also locks the reward
mode to deterministic `ranked-first`, producing `[1, 0]` for the two ranked
completions without depending on checkpoint-specific decoded tokens.

GRPO remains an algorithm-contract comparison rather than exact stochastic
trajectory parity: Antfly records deterministic ranked top-k sampling, while
the pinned TRL/Unsloth control samples stochastically from top-k. Use the
persisted initial adapter and completion IDs for a later replay-based numerical
parity lane; do not label the performance gate itself cross-framework numeric
parity.

Pass both `--model e2b=...` and `--model e4b=...` (with one
`--grpo-target` per model) for the two-model gate. Use `--preflight-only` to
verify artifact suitability without creating an output directory or touching
the GPU. A direct `.gguf` or GGUF-only directory is rejected before execution;
strict Metal and CUDA also reject unsupported rank-2 F16 stored weights from
SafeTensors headers before allocating a device session.

### LoRA Defaults

Recipe-level LoRA defaults are intentionally PEFT-like:

- Supported rank-16 LoRA routes bootstrap at `rank = 16`, `alpha = 32` unless a
  family-specific contract says otherwise.
- GRPO adapter-training routes default to `rank = 8`, `alpha = 32`; raise rank for larger policy tasks after an eval sweep justifies the extra adapter capacity.
- Scaling is standard LoRA `alpha / rank`. Recipe `adapter.scaling` currently accepts only `standard` aliases; rank-stabilized scaling is not enabled in the graph trainer path.
- Gemma4 defaults to `target_preset = "text-all-linear"`, which resolves exact
  text attention, MLP, and PLE linear paths. `peft-qv` is the smaller baseline.
  Providing both a preset and explicit target modules is an error.
- GLiNER2 LoRA defaults match the upstream Python GLiNER2 LoRA surface:
  `rank = 16`, `alpha = 32`, `lora_dropout = 0`, and target groups
  `encoder,span_rep,classifier,count_embed,count_pred`. The encoder group
  expands to query/key/value plus dense encoder projections.
- Qwen/ColQwen optimizer-backed routes default to their all-linear target module lists. They also accept `target_preset = "all-linear"`, `attention-only`, or `mlp-only`; `moe-experts` is rejected until expert-aware rank and routing policy is wired through those bootstraps.
- Gemma4 recipes currently admit standard LoRA initialization only. DoRA,
  PiSSA, LoftQ, EVA, and LoRA-GA fail closed until their graph, adjusted-base,
  or initializer-stat artifact contracts are wired end to end.

For learning-rate selection, do not copy full-finetune LRs directly. Start LoRA sweeps around `1e-4`, `3e-4`, and `1e-3` with the real target metric, then keep the smallest rank/target set that passes. Smaller micro-batches plus gradient accumulation are usually a better first move than shrinking rank below the defaults.

Use `--dry-run` to print the routed tool plan without launching training:

```sh
antfly inference finetune run recipe.json --dry-run
```

For a no-download recipe-layer verifier, use:

```sh
antfly inference finetune smoke-fast
```

`smoke-fast` runs quick dry-runs across every family adapter fixture, executes synthetic no-download GLiNER2, Qwen2, and Gemma4 recipe cases plus the fast scalar DPO/GRPO recipes, verifies the normalized run artifacts reach `status = "succeeded"`, and writes a suite summary at `/tmp/antfly-inference-finetune-smoke-fast/fast_smoke_summary.json` by default.

### GLiNER2 Entity-Training Checks and Release Readiness

The native command below is a scoped resident-Metal entity-training and adapter
reload check. It is not a production-readiness verdict: its JSON always reports
`production_ready = false` and lists the full-task, cross-runtime, and trainer
lifecycle blockers that it does not cover. Use the strict release wrapper later
in this section for the authoritative fail-closed readiness assessment.

```sh
zig build -Dmetal=true gliner2-entity-training-readiness -- \
  /private/tmp/antfly-inference-models/gliner2 \
  /private/tmp/gliner2-conll2003-train-200.jsonl \
  /private/tmp/gliner2-conll2003-eval-200.jsonl \
  /private/tmp/antfly-inference-gliner2-metal-prod-gate \
  person,organization,location \
  --entity-metal-readiness \
  --semantic-golden "Microsoft opened an office in London" Microsoft organization 0.03 \
  --quality-eval \
  --quality-max-examples 25 \
  --quality-nms-overlap 0.0 \
  --quality-top-k-per-label 1 \
  --quality-max-predictions-per-example 3 \
  --quality-best-span-per-label-start \
  --min-entity-f1 0.15
```

`--entity-metal-readiness` is the scoped resident Metal preset: backend
`metal`, compiled required, 200 examples/steps, batch size 1, sequence length
32, Metal optimizer, zero trainable host/device transfers, nonzero resident
trainable bytes, finite/decreasing loss, and `avg_step_wall_ms <= 3000`.
It also runs a 25-example shaped entity-quality eval and requires `f1 >= 0.15`.
This is useful infrastructure and entity-path evidence, but it is neither a
full held-out quality evaluation nor a release gate. The command defaults to
`--objective gliner2-total-loss`, sum-reduced BCE,
`--span-positive-weight 1`, `--span-negative-weight 1`, and the upstream 0.5
negative-mask rate. Use
`--span-loss mse` only to reproduce older calibration runs. For total-loss
runs, step and epoch metrics report `schema_slot_positive_counts`, whose
positions are record-local schema slots. Manifests additionally report raw
`entity_label_positive_counts`, aligned to `entity_labels`, so entity coverage
remains auditable across heterogeneous records. Regular span-start runs keep
reporting `entity_label_positive_counts` in step, epoch, and manifest output.
`--span-hard-negative-weight` adds extra loss weight to negative span labels
whose candidate span overlaps a gold entity, which is useful when diagnostics
show partial-span or wrong-label predictions near true entities.
For long runs, loss-decrease validation uses a first-window vs last-window
trend instead of comparing only the first and final individual step. Semantic
goldens pass when the expected entity appears among decoded predictions above
the requested score threshold; the reported `top_entity` remains available for
calibration debugging.
Semantic adapter reload is required by default. Use repeatable
`--semantic-golden TEXT EXPECT_TEXT EXPECT_LABEL MIN_SCORE` entries for a
stronger release gate; the older single-golden `--eval-text`,
`--expect-label`, and `--min-score` form remains supported.
Optional `--semantic-nms-overlap`, `--semantic-max-predictions`,
`--semantic-top-k-per-label`, `--semantic-best-span-per-label-start`, and
`--semantic-label-thresholds label=FLOAT[,label=FLOAT...]` flags expose
stricter single-text decode shaping for calibration experiments.
`--quality-eval` runs saved-adapter dataset evaluation and supports
`--min-entity-precision`, `--min-entity-recall`, and `--min-entity-f1`.
The evaluator reports exact-match metrics for all decoded entities above
`--quality-min-prediction-score` / `--min-prediction-score`. Span calibration
uses same-label overlap NMS by default; use
`--quality-label-thresholds label=FLOAT[,label=FLOAT...]` for per-label score
floors. Use `--quality-nms-overlap FLOAT`, `--quality-no-nms`, and
`--quality-sweep-thresholds CSV` to inspect decoding calibration; threshold
sweeps report both a global `best_threshold` and
`best_per_label_thresholds` candidates.
`--quality-top-k-per-label`,
`--quality-max-predictions-per-example`, and
`--quality-best-span-per-label-start` provide stricter decode shaping before
raising model-quality thresholds.
When quality eval is enabled, the scoped entity gate also writes
`quality_summary.json` and `quality_thresholds.csv` in the run directory and
includes both paths in the final gate JSON. The summary sidecar contains
per-label metrics, threshold-sweep calibration candidates, a reusable
`recommended_label_thresholds_csv` string, and a bounded FP/FN diagnostic sample
controlled by `--quality-diagnostic-limit`. The CSV file can be fed back into
`--semantic-label-thresholds` or `--quality-label-thresholds` for calibrated
decode checks.

Example 200-step resident Metal entity-check output from this rollout:
`avg_step_wall_ms=2792.75`, `max_device_trainable_transfer_count=0`,
`max_device_trainable_bytes=9486400`, and 25-example shaped quality
`precision=0.1733`, `recall=0.2766`, `f1=0.2131` at threshold `0.03`.
A post-run threshold sweep selected threshold `0.15` with `f1=0.2241`.
Initial per-label thresholds are available as a calibration control, but the
current adapter still needs model-side quality work: a stricter candidate
`person=0.30,organization=0.15,location=0.25` reduced predictions from 75 to
67 and raised organization precision, but aggregate F1 fell to `0.1930`.

For custom thresholds, pass the explicit options instead of the preset:

```sh
zig build -Dmetal=true gliner2-entity-training-readiness -- \
  /models/gliner2 \
  /data/gliner2_train.jsonl \
  /data/gliner2_eval.jsonl \
  /runs/gliner2-prod-gate \
  person,organization,location \
  --eval-text "Alice joined Acme in Paris" \
  --expect-label organization \
  --min-score 0.05 \
  --materialized-dir /runs/gliner2-prod-gate/materialized
```

The scoped entity command rejects duplicate/cycled examples and train/eval text overlap before
it performs dataset readiness checks, full autodiff training, artifact
validation, LoRA bundle inspection, fixed-text semantic eval against the
reloaded adapter, and optional materialization. Use `--dry-run
--skip-semantic-eval` to verify build wiring without local model/data files.

The authoritative Zig/Metal release-readiness gate additionally requires
deterministic multi-step full-task loss/update parity, an exact same-artifact
adapter round-trip, repeated production-shape performance, and a current
five-seed Python/Zig convergence summary. Independently trained tensor equality
is diagnostic only: cancellation-sensitive near-zero gradients can have
different signs across frameworks even when the objective agrees. Deployment
quality is instead gated statistically on held-out metrics. Python is the
pinned oracle and timing baseline, not a production runtime dependency:

```sh
scripts/run_gliner2_lora_production_readiness.sh \
  --model-dir /models/gliner2 \
  --python-model /models/gliner2 \
  --release-adapter-dir /runs/gliner2-release-adapter \
  --train-data /data/gliner2_full_task_train.jsonl \
  --eval-data /data/gliner2_full_task_eval.jsonl \
  --python-bin /usr/local/bin/python3.12 \
  --upstream-source /src/GLiNER2 \
  --convergence-summary /runs/gliner2-convergence-summary.json \
  --heldout-min entities.micro_f1=0.80 \
  --heldout-min entities.exact_match=0.80 \
  --heldout-min classifications.micro_f1=0.80 \
  --heldout-min classifications.exact_match=0.80 \
  --heldout-min json_structures.micro_f1=0.80 \
  --heldout-min json_structures.exact_match=0.80 \
  --heldout-min relations.micro_f1=0.80 \
  --heldout-min relations.exact_match=0.80 \
  --heldout-min count.accuracy=0.80
```

Both files must be representative and content-disjoint. Checked-in smoke rows,
cycled records, and empty task-family placeholders are rejected. The production
gate requires at least `compare_steps * 32` unique training rows, at least 32
evaluation rows, and non-empty entity, classification, JSON, and relation
annotations inside the exact bounded training slice used by both runtimes.
The separately supplied release adapter must contain its Zig training manifest;
the gate verifies that its SHA-256 training-data and base-model fingerprints
match these exact inputs. The base-model fingerprint hashes a fixed, ordered
list of files in `--model-dir`. Most are required and a missing one aborts the
run with `RequiredFingerprintFileMissing`; `spm.model` is optional because the
stock `fastino/gliner2-base-v1` snapshot does not ship it and nothing reads it
for tokenization. An optional file is hashed when present and hashed as an
explicit "absent" marker when it is not, so present and absent snapshots never
share a fingerprint, and a snapshot that does ship the file keeps the digest it
had before the entry became optional. Release provenance also requires `--max-examples 0`,
`--max-steps 0`, exact completion of every requested epoch and microbatch or a
mathematically valid recorded early stop,
no drop-last training records, a batch/accumulation shape that does not trigger
upstream's trailing-batch floor behavior, and clean base-model initialization
without an untracked initial adapter checkpoint.
After parity passes, the script loads that fully trained standard PEFT adapter
into the pinned clean upstream checkout and streams the eval file through upstream's raw
decoder on CPU. It reports micro exact-atom F1 and task-level exact match
for entities, classifications, JSON structures, and head/tail relations, plus
repeatable JSON/relation count accuracy. All nine deployment-specific floors
are required; missing floors, missing positive task coverage, unsupported
relation fields, decode errors, or scores below a floor fail closed.
The same supplied adapter is then scored across the complete held-out set
through Antfly's native evaluator, using each record's structured schema. The
native gate decodes entities, single- and multi-label classifications, JSON
structures, ordered head/tail relations, and structure/relation counts. It
requires the same nine caller-supplied floors and positive task coverage;
missing coverage, unsupported relation fields, decode errors, or scores below
a floor fail closed. The report keeps the number of entity-bearing records
separate from the total held-out record count.
`production_ready` becomes true only when deterministic parity/performance,
the pinned-oracle and Zig-native quality gates, five-seed convergence, and
cross-report model/train/eval fingerprints all pass. Missing, stale, or
inconsistent evidence remains an explicit blocker in the generated report.
Training batching and partial accumulation match the pinned upstream trainer:
datasets no larger than the requested batch use one reduced batch, larger
datasets drop their final partial batch, and a partial accumulation flush keeps
the configured accumulation divisor. Reported `loss` and `final_avg_loss` use
that same upstream-scaled objective; per-microbatch `raw_loss` is retained for
diagnostics. Manifest `micro_batch_steps`/legacy `total_steps` count forward-
backward batches, while `optimizer_steps` is the upstream global-step count and
is what `--min-steps` release thresholds enforce.
Upstream's public relation decoder returns only `head` and `tail`; held-out
records with additional relation fields are rejected rather than partially
scored. Exact atoms use the versioned
`unicode_nfc_collapsed_whitespace_casefold/v1` contract. Python applies the
complete normalization; Zig admits the proven Unicode 15 NFC-inert,
single-scalar-casefold subset and fails closed outside it.
Exact cross-runtime result parity is currently an augmentation-free,
deterministic-core contract. The harness disables Python encoder/count-
transformer dropout, LoRA dropout, negative-span masking, SamplingConfig data
augmentation, training-example shuffle, and task/field shuffling, and pins
example and schema ordering. That is a
stricter subset than stock upstream `deterministic=True` and is used only for
trace-level loss, gradient, optimizer-state, and update comparisons. The
five-seed outcome gate uses a separate stock-stochastic contract: Python must
run with `deterministic=False`, Fastino's exact `SamplingConfig` defaults,
training-mode schema conditioning, model dropout enabled, LoRA dropout `0.0`,
shuffled training data, and the default `0.5` negative-span mask. Zig records
its actual asymmetric policy in every release manifest: SamplingConfig and
model dropout disabled, deterministic eval-form schema conditioning, epoch
shuffle enabled, LoRA dropout `0.0`, and negative-span masking at `0.5`. The
two sides use independent RNG streams and do not claim identical augmentations
or dropout masks; parity there means held-out result distributions meet the
paired quality/deficit gates. The convergence materializer verifies both
policies and rejects deterministic, weakened, or legacy reports.
Deterministic schema conditioning includes entity and JSON descriptions,
classification prompts and few-shot examples, and classification label
descriptions. Their prompt layout and token IDs match the pinned upstream
processor. For deterministic trace runs only, the harness pins upstream's stochastic `example_mode` selection
(`_process_entities`/`_process_classifications`/`_process_json_structures`)
and `_transform_schema` description/example shuffles to eval-mode semantics —
entities/JSON emit every present `[DESCRIPTION]` segment, classifications emit
`[DESCRIPTION]` plus `[EXAMPLE]`/`[OUTPUT]` — which is exactly the
deterministic form the Zig data pipeline always emits.
`testdata/gliner2/described_smoke.jsonl` exercises that surface in the full-task parity
gates. Stock-stochastic outcome studies leave upstream training-mode
conditioning untouched and record that fact in their evidence.

Parity review notes (vs the frozen fastino-ai/GLiNER2 oracle
`8f3fc399bcc5a00749a62a1565e5c6529f04b574`):

- The oracle environment is frozen as well as its source: Python 3.12 / Unicode
  15 plus every direct numerical/runtime dependency in
  `scripts/requirements-gliner2-oracle.txt`. Generated trainers, held-out
  evaluation, repeated performance summaries, convergence evidence, and the
  final readiness report all reject a missing or mismatched package version.

- LR schedule with `--grad-accum N > 1` now matches upstream: the schedule
  horizon, the warmup, *and* the optimizer-step target all use upstream's
  floor-based `len(train_loader) // grad_accum` (clamped >= 1) per epoch.
  Upstream's `_flush_gradients` advances `global_step` and the scheduler like
  any other step and both training loops break on `global_step >= max_steps`,
  so upstream stops at that horizon and completes fewer than `--epochs` data
  passes whenever `len(train_loader) % grad_accum != 0`. The trainer prints a
  warning when that happens. Running the extra end-of-epoch flushes instead
  would not be equivalent even at LR 0: an AdamW step still advances the
  moments and the per-parameter step counter, and it consumes data upstream
  never sees.
- `GlinerAutodiffConfig` struct defaults are now the upstream-parity values
  (`span_start_positive_weight=1.0`, `span_start_loss_reduction=.sum`); the
  previous 32.0/`.mean` defaults survived only for callers that bypassed the
  CLI.
- Adapter eval tools default `--max-span-width` from the adapter manifest's
  trained value (trainer default 8 for manifests predating the field) instead
  of a fixed 4, so evaluation cannot silently use a different candidate-span
  grid than training.
- `parseConfig` fails closed on DeBERTa `share_att_key=false`; the training
  graph reuses the content Q/K projections for the relative-position
  projections and would silently mis-project such checkpoints.
- The compare harness defaults to `--zig-objective gliner2-total-loss` so
  component-loss parity runs by default, and the multi-step e2e gate now
  exercises decoupled AdamW weight decay (`--weight-decay 0.01`) end to end.

Same-day end-to-end run of every gate (bundle rebuilt from the HF cache)
surfaced and fixed three further defects the gates had never executed:

- Optimizer step gating now mirrors PyTorch's grad-presence semantics on both
  host and device optimizer paths:
  upstream leaves `grad=None` for head modules whose task family is absent
  from a batch (no Adam state advance, no decoupled decay), while the Zig
  graph always produced zero-valued grads and stepped everything. The trainer
  gains conditional optimizer families
  (`registerConditionalOptimizerFamily`/`markOptimizerFamilyPresent`,
  presence ORed across the grad-accum window, reset per optimizer step); the
  gliner2-total-loss CLI registers `classifier.`, `count_pred.`,
  `span_rep.`, and `count_embed.` and marks presence from each micro-batch's
  task ids. The multi-step gate's per-parameter Adam step counts now match
  exactly. Presence is family-level (a structure task with gold count 0 marks
  its family), and absent device families have their accumulated gradients
  zeroed and skip their Adam step/decay.
- `--require-full-task-parity` slice-coverage checks required every task
  family in the compared slice unconditionally, which no single-family
  fixture could satisfy; they now require only the families present in the
  source fixture, and the multicount fixture moved a multi-instance
  relations record into the compared slice.
- The dedicated multi-step optimizer gate requires
  `trained_adapter_parity_ok`; one-step independently trained output deltas
  remain diagnostic because tiny cross-framework weight differences are
  amplified by the near-zero fixture. Same-artifact round-trip parity remains
  exact in every gate.
- `--deterministic` now also pins training data order on both sides (the
  Zig CLI skips its epoch shuffle; the harness passes the Python loader
  `--no-train-shuffle`), so deterministic comparisons cannot silently train
  on permuted batches.

Measured on the deterministic 3-step config (batch 2, seq 64, rank 4):
Python torch-CPU ≈ 0.2 s/step, Zig native ≈ 15 s/step (the correctness
reference, not a performance target), Zig Metal ≈ 0.55 s/step; the
performance target remains the Metal backend at production shapes
(`run_gliner2_lora_perf_gate.sh`, warm-step median ≤ 1.0× Python at
batch 32/seq 128). Historical short probes are diagnostic only; the production
wrapper now requires exactly five paired independent seeds, all deployment
floors, at most 0.02 mean Zig metric deficit, and at most 0.05 deficit in any
paired run.

The production batch-32 profile uses 16-sample structure-loss chunks and
reports the graph executor's device-owned peak-live metric separately from
process RSS. There is no generic release ceiling: operators can set an
explicit device-memory limit for their deployment, and that limit remains
fail-closed on representative release data.

Accepted, deliberately not "fixed":

- FFN GELU uses an erf approximation (Abramowitz–Stegun 7.1.26, ~1.5e-7
  max per-element error) rather than libm erf. Deterministic summed losses use
  a combined `1e-4` absolute and `5e-6` relative gate so the threshold scales
  with production-batch loss sums; replacing the approximation would desync
  the matching Metal kernel implementations and perturb unrelated golden
  baselines.
- Bitwise trace parity with stock upstream RNG streams (torch shuffle order,
  dropout masks, negative-span masking draws) is out of scope by design; the
  deterministic-core contract above is the parity claim.
- Preprocess parity asserts the aggregate `structure_positive_count`, not
  per-cell placement. The Zig side already dumps the full per-cell grid
  (`span_labels_all`, layout `[sample][span][entity]`, cell value =
  multiplicity); a positions-multiset comparison
  (`flat = start_word * max_span_width + (end - start)`) is the intended
  tightening once the Python dump's span serialization (single `[s,e]` vs
  list-of-subspans) is pinned against a live run.
- Python release-contract tests and the native/Metal per-parameter gradient
  gate are operator-run release checks. Real-model performance and five-seed
  quality likewise remain artifact-driven release jobs; ordinary CI does not
  provide that evidence.
The native trainer supports recoverable optimizer checkpoints at complete
epoch boundaries with `--checkpoint-every-epochs`, and exact-run resume with
`--resume-checkpoint`. Checkpoints include trainable weights, Adam moments,
step counters, and a run fingerprint; resume also restores deterministic data
order, held-out selection state, and the checkpoint-bound metrics prefix. A
resumed release retains its source as a content-addressed checkpoint and
records that file's exact digest and restored counters in the manifest.
`--checkpoint-keep-last` bounds retained periodic state files to three by
default. Metal checkpoint/save/resume requires `--compiled-required` so an
exact run cannot switch between compiled and interpreter execution across
processes.

Pass a content-disjoint single JSONL file with `--eval-data` for in-process
epoch loss, `--eval-every-epochs` for cadence, and `--eval-batch-size` for the
upstream-equivalent logical evaluation batch (default 8, with no drop-last).
`--early-stopping-patience` enables strict lower-loss stopping and
`--early-stopping-threshold` sets the required decrease. Evaluation only runs
after complete epochs, disables LoRA dropout, never advances optimizer or
training RNG state, and preserves upstream's equal weighting of logical batch
losses (including its summed full-task loss). Crash-safe, epoch-addressed
`checkpoints/best-epoch-N.safetensors` files preserve the recoverable best state
for every retained resume boundary; final PEFT and task-head artifacts are
exported from the selected epoch. Resume rebuilds best loss/epoch and patience
from the exact metrics prefix, removes abandoned future bests, and fails closed
if the selected best checkpoint does not match it.
Training text splitting is pinned to Python 3.12's Unicode 15.0 behavior for
`str.lower()`, `\w`, `\s`, URL/email/mention branches, punctuation, emoji,
and UTF-8 byte boundaries. This is intentionally versioned rather than a claim
of parity with every Python Unicode release. U+0130 (`İ`) still fails closed:
Python expands it to two code points, so its lowered offsets cannot map
one-to-one onto original UTF-8 supervision/decode boundaries. Unicode schema
conditioning keeps its existing ASCII/lower-Latin-1 restriction. Native
encoding also verifies the exact Fastino `Strip`/precompiled-charsmap/space-
replacement normalizer fingerprint and rejects any text, prefix, or schema
fragment that it cannot prove the unimplemented normalizer would leave
unchanged; unknown normalizers are rejected at tokenizer initialization. The
conservative admitted set includes NFKC-stable assigned letters, numbers,
punctuation, and symbols such as `Москва`, `東京`, and simple emoji. Compatibility
forms such as `①`, `Ａ`, and `ﬁ`; Unicode marks, separators, and controls; scalars
that can participate as the second half of canonical composition; U+2581 and
U+FFFD; leading, trailing, or repeated ASCII spaces; and emoji sequences
containing marks or joiners fail closed. Regenerate and exhaustively verify the
scalar/category/context tables with
`python3.12 scripts/generate_gliner2_unicode_tables.py --write` and verify with
`python3.12 scripts/generate_gliner2_unicode_tables.py --check`.
With the pinned model available, add
`--tokenizer-json <model_dir>/tokenizer.json` to verify the normalizer shape,
charsmap fingerprint, and every admitted scalar against the real tokenizer.

Full-task training uses per-sample contextual schema slots. Different samples
can reuse the same slot for different fields and inactive tail slots are masked;
manifest entity-label coverage remains training-derived. The admitted schema,
structure-instance, and task limits cover both training and held-out records,
so evaluation can use wider contextual axes without changing trainable shapes.
The trainer still fails closed above 256 schema slots and reports the offending
per-record width before graph construction.
Each shuffled train or eval batch selects bounded local graph axes: sequence
length uses eight-token buckets, while schema slots, structure instances, and
task count use next-power-of-two buckets capped by the admitted dataset limits.
Text-word routing and span grids likewise use the batch's actual Fastino-style
word count rounded to a bounded power of two, with a floor large enough to pack
every schema-task row. They no longer allocate `seq_len * max_span_width` rows
for short texts. This is the `batch-local-v2` manifest policy; release
validation intentionally rejects pre-v2 shape-policy artifacts.
A deterministic LRU cache retains at most `--graph-cache-capacity` signatures
(default 2, maximum 8), shares trainables, Adam state, counters, and RNG across
them, and releases inactive runtime inputs. Cache policy and build/hit/reuse/
eviction/residency metrics are recorded in the training fingerprint, metrics,
and manifest. Sequence-length preflight runs the exact tokenizer/prompt encoder
with bounded scratch space and does not build span grids or labels. The
conservative admission estimate still budgets the union maxima for safe cache
construction and accounts for full-task structure-instance scores, schema-task
classification/count heads, count-transformer attention, and packed targets
with saturating shape arithmetic.
Step, epoch, and manifest schema-slot counts are computed from the final active
loss masks after negative masking; raw entity-label counts remain separate.
Total-loss adapter evaluation requires explicit `--entity-types`, because a
dataset-wide raw-label union is not a valid contextual-slot schema.

### Adapter Matrix

| Recipe | Family | Current route |
|--------|--------|---------------|
| `lora-sft` | `gemma4` | prepare train → prepare disjoint eval → bootstrap exact adapter inventory → native, strict-Metal, or strict-CUDA train/eval with `--eval-prepared`; full `sft` and `qlora-sft` are typed errors |
| `lora-sft`, `qlora-sft` | `gliner2` | `train-gliner2-autodiff` (real full-encoder autodiff training; the cached probe-surrogate bundle route was removed) |
| `lora-sft`, `qlora-sft` | `layoutlmv3` | `bootstrap-layoutlmv3-lora` → `train-eval-layoutlmv3-lora-sequence` or `train-eval-layoutlmv3-lora-token` → optional `materialize-layoutlmv3-checkpoint` |
| `sft` | supported non-Gemma LoRA families | same route as the family `lora-sft` adapter while full-weight SFT backends are still family-specific |
| `dpo` | scalar preference fixtures | direct internal `preference_loss.zig` adapter over `dataset.format = "scalar-logprobs"` JSONL |
| `dpo` | decoder models with local weights | direct internal `preference_harness.zig` adapter over `dataset.format = "text-preference"` or `"rendered-text-preference"` JSONL with `model.path` and optional `model.reference_path`; Gemma4 has native, strict-Metal, and strict-CUDA optimizer-backed LoRA paths, exact same-graph base-reference caching, and chosen/rejected gradient accumulation kept in one preference unit; Qwen2, ColQwen2, and Qwen3.5 text recipes also have optimizer-backed adapter-training paths |
| `grpo` | scalar/token fixtures | direct internal `grpo.zig` adapter over `dataset.format = "token-logprobs"` JSONL |
| `grpo` | decoder models with local weights | direct internal adapter over `dataset.format = "text-grpo"` or `"rendered-text-grpo"` JSONL with deterministic ranked sampling, decoded-text and token-level reward modes, and optional `model.reference_path`; Gemma4 text training has native, strict-Metal, and strict-CUDA optimizer-backed LoRA paths, keeps each completion group inside one accumulation unit, and rejects runs with no reward-derived advantage or no policy gradient; Gemma4 multimodal GRPO remains native/Metal-only while its CUDA projector backward path is unqualified; Qwen2, ColQwen2, and Qwen3.5 text recipes also have optimizer-backed paths |
| `reranker` | `reranker` | `prepare-reranker-pooled-cache` → `train-eval-reranker-head-cached` → optional `materialize-reranker-head` |
| `lora-sft`, `qlora-sft` | `reranker` | `bootstrap-reranker-lora` → `prepare-reranker-top-layer-cache` → `train-eval-reranker-lora-top-layer-cached-surrogate` → optional `materialize-reranker-lora` |
| `vlm-retrieval` | `colqwen2` | `prepare-colqwen2-inputs` → `bootstrap-colqwen2-lora` → `train-eval-colqwen2-lora-bundle` |
| `sft`, `lora-sft`, `qlora-sft`, `dpo`, `grpo` | `qwen3_5`, Chandra OCR text-only | direct Qwen autodiff trainer route for text JSONL recipes; still requires real-weight CPU and MLX/Metal smokes before production readiness |

The runner first looks for a peer tool executable next to the current Antfly inference executable. If it is not installed, it falls back to the existing `zig build <tool> -- ...` build step from the package root, preserving today's build-step workflow.

### Run Artifacts

Non-dry runs write a normalized manifest at `artifacts.manifest_path` or `<artifacts.root>/recipe_run_manifest.json`.

The manifest schema version is `antfly_inference_finetune_recipe_run/v1` and records:

- the original parsed recipe
- artifact root
- expanded step names and argv
- overall run status: `planned`, `running`, `succeeded`, or `failed`
- per-step status, exit code, stdout byte count, and stderr byte count

The runner writes the manifest before execution starts, updates it before each step, and writes a final success or failure state.

Every non-dry run also writes:

- `<artifacts.root>/training_config.json` with the normalized recipe, expanded step plan, dataset fingerprints, backend build metadata, and optimizer summary
- `<artifacts.root>/training_report.json` with the normalized final status, per-step execution records, dataset fingerprints, backend build metadata, optimizer summary, and final artifact checksums

Direct DPO and GRPO adapters also write `artifacts.report_path`, or `<artifacts.root>/dpo_report.json` / `<artifacts.root>/grpo_report.json` when no explicit report path is provided.
Optimizer-backed Gemma4 reports include the actual policy backend plus optimizer
and micro-batch step counts plus an exact post-training LoRA tensor-movement
summary. A zero-learning-rate, non-finite, or no-parameter-movement run fails
before adapter publication. The same rendered objective report is stored as
`antfly_preference_run_report.json` inside the immutable trained-adapter bundle
before the run-level report is replaced, so a late report-path failure cannot
leave an evidence-free adapter. GRPO reports also include the validated reward
mode, reward mean and standard deviation, frozen-reference mode and scoring
time, initial ranked token IDs and
policy/reference logprobs, and sampling/rescore parity, so a mechanically
healthy but reward-degenerate or reference-drifted run is visible.
Gemma4 DPO/GRPO reports additionally include `device_execution`, a cumulative
per-trainer evidence record covering scoring and backward graph calls. It
records compiled-session builds, dispatch/partition counts, fallback and host
output counts, declared versus observed runtime uploads, CUDA transfer and
kernel totals, and peak resident bytes. The real-checkpoint CUDA smoke treats
that record as a release gate rather than informational telemetry.
Gemma4 DPO reports additionally include `policy_scoring_mode`,
`training_microbatch_mode`, `metal_buffer_reuse_mode`, `reference_mode`,
reference-precompute time, completion-fenced Metal workspace statistics, and
initial zero-adapter policy/reference logprob parity. Publication fails if that
initial parity drifts beyond the admitted tolerance. Metal DPO enables the
completion-fenced workspace by default and sizes it to 9/16 of the device's
`recommendedMaxWorkingSetSize`, rounded down to MiB. Set
`ANTFLY_GEMMA4_DPO_COMPLETION_FENCED_CACHE=0` for the fail-safe rollback. The
generic Metal runtime remains opt-in; `TERMITE_METAL_COMPLETION_CACHE_MAX_MB`,
`TERMITE_METAL_COMPLETION_CACHE_MAX_BUFFER_MB`, and
`TERMITE_METAL_COMPLETION_CACHE_MAX_SLOTS` are diagnostic overrides rather
than recipe-level tuning requirements.

The exact qualified E2B topology also defaults to planned-encoder-fenced
same-frame private-buffer reuse. Set
`ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE=0` to restore the qualified rollback.
E4B and unknown Gemma4 shapes remain fail-closed unless that variable is
explicitly set to `1` for research. The report records the effective policy so
benchmark and release evidence cannot silently mix the two modes.

The scalar DPO input format remains JSONL with precomputed logprob rows:

```json
{"policy_chosen_logp":-1.0,"policy_rejected_logp":-2.0,"ref_chosen_logp":-1.2,"ref_rejected_logp":-1.8}
```

Model-backed DPO also accepts text preference rows:

```json
{"prompt":"Answer with one word: yes or no?","chosen":"yes","rejected":"no"}
```

Use `dataset.format = "text-preference"` to treat `prompt` as user content and
apply the model chat template when one exists, or
`dataset.format = "rendered-text-preference"` when `prompt` is already the
final rendered decoder prompt. `model.reference_path` is optional; when
omitted, the runner reuses `model.path` as the reference model. The current
optimizer-backed Gemma4 DPO route admits only that same-base reference. An
explicit different Gemma4 reference directory fails closed instead of silently
scoring through a graph with different semantics.

For a fixed-protocol performance probe, set
`ANTFLY_GEMMA4_DPO_BENCHMARK=1`. The recipe must produce exactly 25 optimizer
updates: one cold update, one first steady update, three warmups, and 20
measured updates. The DPO report then includes every duration and loss plus the
measured median and mean. This is a diagnostic contract, not a general training
setting.

The qualified E2B sequence-128 one-token fast path coalesces a shared-prompt
chosen/rejected pair into one sparse policy score, one sparse frozen-reference
score, and one weighted backward microbatch. CUDA selects this qualified path
without an ambient feature flag. Multi-token and non-shared-prompt pairs retain
the general sequence path.

For real multi-token DPO parity, use
`scripts/materialize_gemma4_dpo_hf_parity.py` to create a provenance-bound
Antfly JSONL plus an exact-token MLX case. Its `--bucket-occurrence 1` mode
creates a deterministic disjoint holdout from the same pinned source. Run the
matched MLX training side with `scripts/run_gemma4_dpo_mlx_benchmark.py`, then
score the Antfly and MLX adapters under one runtime with
`scripts/evaluate_gemma4_dpo_adapters_mlx.py`. General multi-token Metal DPO
executes each chosen/rejected sequence once, detaches the first adapter-gradient
set, combines both branches on device with the DPO coefficients, and performs
one optimizer flush. Benchmark samples and their machine-specific measurements
belong in external evidence bundles, not this production runbook.

The current direct GRPO input is JSONL with token-level rows:

```json
{"prompt_idx":0,"tokens":[10,11],"old_logps":[-0.4,-0.6],"ref_logps":[-0.5,-0.7],"new_logps":[-0.35,-0.65],"reward":1.0}
```

Model-backed GRPO also accepts text prompt rows:

```json
{"prompt":"Answer with one word: yes or no?","target":"yes"}
```

Use `dataset.format = "text-grpo"` to treat `prompt` as user content and apply the model chat template when one exists, or `dataset.format = "rendered-text-grpo"` when `prompt` is already the final decoder prompt. The current model-backed GRPO route supports:

- `grpo.group_size` from 2 through 8; larger deterministic-ranked groups are
  rejected instead of repeating top-eight candidates
- `grpo.max_completion_tokens`
- `grpo.reward_mode = "exact-match"`, `"exact-match-ci"`, `"prefix-match"`, `"token-exact-match"`, `"token-prefix-match"`, or `"ranked-first"`; the last mode is intended for deterministic infrastructure and performance qualification
- optional `model.reference_path`

`exact-match-ci` is trimmed ASCII case-insensitive equality. Token reward modes
tokenize the trimmed `target` and compare token IDs directly, which is useful
for format and control-token objectives that intentionally disappear during
normal decoding. Empty targets fail closed. Optimizer-backed Gemma4 GRPO also
requires at least one nonzero reward-derived advantage and a nonzero policy
gradient before it can publish an adapter; a KL-only update is not accepted as
a successful GRPO run. An explicit `model.reference_path` different from the
policy base fails closed; the compiled zero-LoRA reference is valid only for
the same base graph and weights.

Optimizer-backed Gemma4 DPO/GRPO recipes are admitted fail-closed. Constant
learning rate, AdamW weight decay, epochs, preference-unit gradient
accumulation, and max-gradient norm are supported. Scheduler/warmup,
`max_steps`, micro-batch sizing, schedule-free optimization, LLRD,
checkpoint/resume, evaluation, and runtime overrides are rejected until the
preference runners implement those semantics. DPO requires a finite positive
`preference.beta`; GRPO validates finite clip, KL, and advantage parameters.

For the fixed E2B performance diagnostic, set
`ANTFLY_GEMMA4_GRPO_BENCHMARK=1`. The recipe must produce exactly 25 complete
groups: one cold group, one first steady group, three warmups, and 20 measured
groups. The GRPO report records the complete loss/reward/parity telemetry for
each group plus measured median and mean duration. The checked-in workload is
`testdata/gemma4_grpo_e2b_seq128_benchmark.json`, with the corresponding
single-row CLI dataset in
`testdata/gemma4_grpo_e2b_seq128_benchmark.jsonl`. The pinned MLX-LM side is
`scripts/run_gemma4_grpo_mlx_benchmark.py`.

The optimized path reuses one shared ranked-prompt forward, sampling logprobs,
a bounded exact reference cache, a coalesced one-token group update, and a
sparse selected-row projection. The benchmark qualifies command execution,
repeatability, and bounded-workload performance; broad GRPO quality remains a
separate held-out evaluation gate. Store measured samples and machine-specific
comparisons in external evidence bundles.

For Gemma4 multimodal GRPO, add `model.projector_path` and use prompt rows with media placeholders plus `image_paths` / `audio_paths`. The current optimizer-backed multimodal route reuses the Gemma projector-backed autodiff path and requires the reference path to stay on the same base model directory.

### Remaining Task List

Completed:

1. Added a common recipe schema and `antfly inference finetune run <recipe.json>`.
2. Added adapter routing for Gemma4 LoRA, GLiNER2 LoRA, LayoutLMv3 LoRA, reranker head, reranker LoRA, and ColQwen2 VLM retrieval.
3. Split train/eval dataset and cache fields where existing tools require separate train/eval inputs.
4. Added dry-run expansion tests for every supported adapter family plus SFT, DPO, and GRPO recipes.
5. Added example recipe files under `testdata/`.
6. Added a normalized recipe-run manifest with status and expanded step records.
7. Promoted `sft`, `dpo`, and `grpo` from reserved schema values to runnable recipes.
8. Added direct internal DPO and GRPO adapters over normalized logprob fixture formats.
9. Added normalized `training_config.json` and `training_report.json` run artifacts.
10. Replaced shell-out execution for reranker head recipes with direct internal prepare, train/eval, and materialize adapters.
11. Replaced shell-out execution for Gemma4 recipes with direct internal prepare, bootstrap, and train/eval adapters.
12. Replaced shell-out execution for GLiNER2 recipes with direct internal bootstrap, cache prepare, train/eval, and materialize adapters.
13. Replaced shell-out execution for LayoutLMv3 recipes with direct internal bootstrap, train/eval, and materialize adapters.
14. Replaced shell-out execution for reranker LoRA recipes with direct internal bootstrap, top-layer cache prepare, surrogate train/eval, and materialize adapters.
15. Replaced shell-out execution for ColQwen2 recipes with direct internal prepare, bootstrap, and train/eval adapters.
16. Extended normalized recipe reports with dataset fingerprints, backend build metadata, optimizer summaries, and artifact checksums.
17. Added a first model-backed DPO route for decoder models using `preference_harness.zig`, real sequence logprobs, and optional explicit reference model paths.
18. Added a first model-backed GRPO route for decoder models using `preference_harness.zig`, deterministic decoder sampling, exact-match rewards, and optional explicit reference model paths.
19. Added `antfly inference finetune smoke-fast` for fast no-download recipe-layer verification across family dry-runs and scalar preference executes.
20. Added one synthetic no-download GLiNER2 direct-family execute case to `smoke-fast`, covering bootstrap, cache prepare, train/eval, and normalized artifact finalization through the unified recipe runner.
21. Updated the fine-tuning docs so `antfly inference finetune run` is the primary public entrypoint and family build-step commands are documented as backend reference.
22. Added an initial optimizer-backed Gemma4 LoRA DPO path for `dataset.format = "text-preference"`, using live autodiff policy logprobs plus `preference_loss` gradients to train adapters and emit a trained adapter bundle.
23. Added an initial optimizer-backed Gemma4 LoRA GRPO path for `dataset.format = "text-grpo"`, using live autodiff sampling plus token-logprob gradients to train adapters.
24. Added `prefix-match` as a second text reward mode for model-backed GRPO and covered the new dry-run route in `smoke-fast`.
25. Broadened the optimizer-backed Gemma4 LoRA DPO and GRPO routes to also accept `rendered-text-preference` and `rendered-text-grpo`, using token-based prepared examples for the rendered DPO path.
26. Tightened targeted Gemma autodiff coverage for token-logprob gradient projection across prompt/completion boundaries.
27. Broadened optimizer-backed Gemma4 GRPO to a multimodal route using `model.projector_path`, media-aware prompt preparation, and a frozen multimodal reference trainer for KL scoring.
28. Added `exact-match-ci` as a trimmed ASCII case-insensitive GRPO text reward mode and covered it with a `smoke-fast` dry-run recipe.
29. Broadened optimizer-backed DPO beyond Gemma4 by adding a Qwen2 text route that reuses the unified token-preference recipe flow and emits standard adapter artifacts.
30. Broadened optimizer-backed GRPO beyond Gemma4 by adding a Qwen2 text route that reuses the unified prompt-sampling recipe flow and is covered by `smoke-fast` dry-run recipes.
31. Broadened optimizer-backed text GRPO and DPO family routing to include ColQwen2 text-only recipes via the existing Qwen2-backed decoder trainer path.
32. Added execute-path verification for optimizer-backed Qwen2 DPO and GRPO in `smoke-fast`.
33. Added execute-path verification for optimizer-backed Gemma4 GRPO in `smoke-fast`; the native backend now preserves unshaped vector gather semantics for the current decoder graph.
34. Removed the external local tokenizer-bundle dependency from the synthetic decoder smoke assets by generating tiny fallback HF tokenizer files when needed.
35. Added Qwen3.5/Chandra fine-tune readiness gating so unified recipes no longer infer those models as Qwen2 or route adapter training through the Qwen2 autodiff graph.
36. Added the first Qwen3.5 training graph slice: full-attention text layers now build with gated `q_proj`, Qwen3.5 `1 + weight` RMSNorm, and partial-RoPE metadata, while linear-attention layers fail explicitly.
37. Added Qwen3.5 linear-attention graph IR and routed text SFT/DPO/GRPO adapter recipes through the Qwen autodiff trainer.
38. Made Gemma4 DPO and GRPO honor the requested native or strict-Metal recipe backend for both policy training and reference scoring, with pair/group-safe accumulation and optimizer-step evidence in reports.
39. Added deterministic synthetic Gemma4 DPO/GRPO execute cases, token-level GRPO rewards, owned reward-target storage, reward-distribution telemetry, and a fail-closed no-advantage/no-policy-gradient publication gate.
40. Qualified one-step BF16 LoRA DPO and GRPO on real Gemma4 E2B and E4B artifacts through the Metal recipe route, including changed adapter payloads and non-degenerate GRPO rewards.
41. Added exact compiled-graph frozen-reference caching and zero-adapter parity gates for Gemma4 DPO, resident device-weight policy scoring, preference-unit gradient scaling, and a matched 25-update E2B MLX benchmark contract.
42. Replaced the eager Gemma4 text-GRPO reference with immutable zero-LoRA bindings on the compiled policy graph, added fail-closed sampling/reference parity and a 25-group benchmark protocol, and qualified the matched E2B Metal/MLX diagnostic above.
43. Optimized the locked one-token GRPO path with shared ranked sampling, on-policy logprob reuse, an exact bounded reference cache, coalesced sparse-row gradients, and alignment-safe Metal projection; then made in-frame private-buffer reuse barrier-safe and qualified five byte-identical 25-group trajectories at `0.773689 s` median-of-medians and `1.871 GB` peak physical footprint.
44. Optimized identical-prompt one-token Gemma4 DPO by sharing policy/reference projection and coalescing opposing gradients into one sparse weighted row. Five final E2B runs produced identical trajectories and adapters at `0.760557 s/update` median-of-medians (`2.427x` faster than the prior Antfly pair path); at that checkpoint DPO fail-safely disabled the unresolved Metal reuse pool and recorded that policy in the report.
45. Routed general multi-token Gemma4 DPO policy scoring through cached pruned loss-only compiled sessions. Two byte-identical 25-update UltraFeedback runs measured a conservative `6.55229 s/update` median (`6.05%` below the eager-score fused path), retained exact initial base-reference parity, and matched all MLX preference decisions on training and disjoint holdout rows.
46. Replaced redundant general multi-token policy scoring/backward with exact detached chosen/rejected device gradients and one optimizer flush, coalesced qualified rank-16 LoRA-A MPS dots, and added conservative exact-order add3 fusion. The exact final binary's 25-update sequence-512 run measured `5.05956 s/update` median and `14.744 GB` peak physical footprint, retained deterministic adapter bytes, and matched every MLX preference decision on both the five-row training set and disjoint holdout.
47. Replaced the dominant decomposed Gemma4 attention VJP with a compact saved-value-aware path using five batched MPS contractions plus fused pack, softmax-VJP, and compact-KV reduction kernels. The exact E2B topology is now default-on with a kill switch: the final same-binary sequence-512 DPO gate improved median/mean by `9.75%/10.50%`, used `14.622 GB` peak footprint, reproduced three byte-identical 25-step trajectories, and retained full preference-decision agreement with MLX. E4B remains fail-closed on the decomposed graph because its superficially faster compact 25-step run diverged to update cosine `0.807816` and relative L2 `0.661817`.
48. Repaired ahead-of-order fusion liveness, scoped the in-frame and completion-fenced Metal reuse policies as one drained-boundary transaction, and promoted planned-encoder-fenced reuse only for the exact E2B topology. Three alternating fixed-25 candidates, a same-hot-path rollback, and the final no-override run had identical loss arrays and adapter bytes. The final binary measured `2.53610 s/update` median and `8.759 GB` peak footprint: `37.98%` faster and `40.10%` lower-memory than rollback, only `23.13%` behind MLX median time, and `54.25%` below MLX peak memory. E4B reproduced its rollback adapter with in-frame reuse disabled.
49. Traced the remaining E2B backward graph and implemented a bit-exact frozen-RMSNorm residual-add fusion with protected-output lifetime guards. All `138` expected chosen/rejected chains fused, and one-step plus fixed-25 adapters stayed byte-identical, but the measured gain was only `0.08%/0.12%` median/mean while lifetime footprint rose `4.90%`. The route is therefore research-only and default-off until its output write is integrated with a proven buffer-plan or last-use residual allocation.
50. Added the strict CUDA compiled-device lane for BF16 Gemma4 text SFT, DPO, and GRPO, including device-resident gradients/AdamW, fused frozen-weight linear cross-entropy, mixed-BF16 backward primitives, transfer evidence, fail-closed artifact admission, and tiny real optimizer-step gates for both preference objectives.
51. Added a same-model CUDA preference suite with one admitted model, isolated DPO/GRPO trainers, immutable shared initial adapters, and v2 admission/per-job timing telemetry. The locked rank-16, 25-update real-weight gate passed on both E2B and E4B with zero graph fallback, host gradients, or initial policy/reference error. E2B DPO/GRPO medians were `0.329475 s` / `0.886613 s` at `9,764 MiB` peak; E4B medians were `0.619986 s` / `1.183090 s` at `15,304 MiB` peak. Against the pinned Unsloth controls, Antfly DPO latency was `3.83%` / `56.39%` higher while GRPO latency was `21.92%` / `13.99%` lower. This is a single-run performance/algorithm-contract gate, not broad quality or stochastic-trajectory parity.

Remaining:

1. Integrate the traced RMSNorm-backward residual add with buffer-plan ownership or a proven last-use in-place residual write, including explicit same-encoder hazard fencing and alias-escape tests; then profile and coarsen the larger sequence-512 projection/normalization regions. Keep the compact attention VJP and in-frame reuse E2B-only until numerically stable E4B implementations pass multi-step trajectory and shared-oracle gates.
2. Extend the shared GRPO sampler to multi-token groups by batching active divergent prefixes per decode step, with exact prompt/prefix-keyed reference caching and parity tests.
3. Rework E4B DPO attention under a bounded-memory, stable-reduction design, then repeat the fixed-25 performance/trajectory and shared-MLX quality gates; run the corresponding matched E4B GRPO campaign only after that DPO lane is stable.
4. Add real-weight one-step smoke coverage for Qwen3.5 text SFT/DPO/GRPO on CPU and MLX/Metal.
5. Add execute-path verification for the broader Qwen-family text-decoder routes, including ColQwen2 if we keep that path.
6. Add Chandra multimodal training data preparation with dynamic image-token expansion before enabling multimodal fine-tune recipes.
7. Add pluggable/custom reward functions when tasks need semantics beyond the built-in decoded-text and token matching modes.
8. Repeat the fixed-protocol BF16 E2B/E4B CUDA campaigns across multiple processes with trajectory hashes and a shared external quality oracle; profile and close the remaining `56.39%` E4B DPO latency gap without regressing E2B or GRPO.

---

## Architecture

```
Forward:  trace → optimize → cache → interpreter.execute(fused ops)
Backward: autodiff.gradient() → optional checkpoint rewrite → interpreter.execute(primitive ops)
Update:   extract grad f32 → flat optimizer state update → upload updated weights
```

The key insight: fused ops (`linear`, `rms_norm`, `attention`) are efficient for inference but opaque to differentiation. Each fused op carries a `vjp_alternate` pointer to its primitive decomposition. `autodiff.gradient()` uses `lower.zig` to expand these, then applies VJP rules on the primitive graph.

---

## Autodiff

`lib/ml/src/graph/autodiff.zig`

Reverse-mode AD with ~25 VJP rules covering all primitive ops. Given a scalar loss node and a list of parameter nodes, returns gradient node IDs in the lowered graph.

Flow:
1. `lower.lower()` expands fused ops via `vjp_alternate` into primitives
2. Walk backward from the loss node, applying VJP rules at each primitive
3. Accumulate adjoints when a node has multiple consumers
4. Return `GradientResult { grad_graph, param_grad_ids, loss_id }`

### VJP Rules

- **Elementwise** (add, mul, exp, log, sqrt, tanh, ...): standard calculus
- **dot_general**: `dA = dY @ B^T`, `dB = A^T @ dY` (with batched 3D support)
- **reduce_sum**: broadcast gradient back to original shape
- **transpose/reshape**: inverse permutation/reshape
- **gather/scatter**: scatter_add for gather grad, gather for scatter grad
- **softmax**: `dX = softmax * (dY - sum(dY * softmax))`

`lib/ml/src/graph/grad_check.zig` provides finite-difference gradient verification for validating VJP implementations.

---

## Primitive Op Backend

The `ComputeBackend` vtable has optional methods for all primitive ops (defaulting to `null` so backends that don't support training compile unchanged). The BLAS backend (`src/ops/blas_compute.zig`) implements all of them:

- **Elementwise**: subtract, divide, negate, sqrt, rsqrt, exp, log, sin, cos, tanh, abs, erf, less_than, where_select
- **Shape-aware**: reduce_sum/max/mean (with axes), reshape, transpose (with permutation), broadcast_in_dim, slice, concat, pad
- **Structured**: dot_general (batched matmul via BLAS sgemm), gather, scatter_add
- **Fused primitives**: softmax, log_softmax (with last-dim-size parameter)

The interpreter (`src/graph/interpreter.zig`) dispatches primitive ops to these vtable methods, extracting shape info from graph nodes for shape-aware ops.

---

## Loss Functions

Built as compositions in the builder API (`lib/ml/src/graph/builder.zig`):

- **softmax / log_softmax**: emitted as fused ops (`fused_softmax`, `fused_log_softmax`) with `vjp_alternate` decompositions
- **Cross-entropy loss**: `-reduceSum(target * logSoftmax(logits)) / batch_size`
- **MSE loss**: `reduceMean((pred - target)^2)`

No special loss op nodes — losses compose from existing primitives and get their gradients automatically through autodiff.

---

## Training Step

`src/graph/training.zig`

```zig
pub fn trainStep(allocator, graph, loss_node, cb, options) !TrainStepResult
```

1. Run `autodiff.gradient()` to produce the combined forward+backward graph
2. Mark loss and gradient nodes as outputs
3. Execute the combined graph through the interpreter
4. Extract loss as f32 scalar and gradients as f32 slices keyed by parameter name

`TrainStepResult` fields:
- `loss`
- `gradients`
- `profile`: autodiff / checkpoint / execute / extract / total timing
- `checkpoint_summary`: optional savings analysis for checkpointed runs

---

## Optimizers

`lib/ml/src/graph/optimizers.zig` — pure f32 math, no backend dependency.

| Optimizer | Features |
|-----------|----------|
| SGD | Optional momentum |
| Adam | Bias-corrected first/second moments (beta1, beta2, epsilon) |
| AdamW | Decoupled weight decay |

Learning rate schedules: `constant`, `cosine` (with min_lr), `warmup_cosine` (linear warmup then cosine decay).

Gradient clipping via `clipGradients()` with configurable L2 norm threshold.

Per-parameter state (momentum `m`, second moment `v`) stored in `OptimizerState`.

The Adam/AdamW hot path also supports `stepSlices()` — a fused SIMD path for contiguous `f32` buffers used by the training loop.

---

## Training Loop

`src/graph/training_loop.zig`

### TrainingWeightStore

The inference `WeightStore` is read-only. `TrainingWeightStore` wraps it with mutable f32 copies for trainable parameters:

- `materializeTrainable(name)`: copies a weight from the base store to a mutable f32 buffer
- `setWeight(name, data)`: updates a trainable weight
- Frozen weights delegate reads to the base store

### FlatTrainingState

The training loop builds an internal flat execution layout over trainable parameters and optimizer state:

- deterministic per-parameter layout metadata
- one contiguous parameter buffer
- one contiguous gradient buffer
- one contiguous first-moment buffer
- one contiguous second-moment buffer

This removes name-hash lookups and fragmented iteration from the optimizer hot path.

### TrainingLoop

```
for each step:
    1. refresh flat state from named trainable weights + optimizer state
    2. build runtime inputs from contiguous parameter spans
    3. training.trainStep() → loss + gradients + timing profile
    4. copy gradients into contiguous gradient spans
    5. clipGradients()
    6. optimizer.stepSlices() on each contiguous trainable span
    7. sync updated params/moments back to the named stores
```

Per-step metrics include: runtime-input build time, optimizer time, total step time, flat-state size counters, and the underlying `TrainStepProfile`.

### Checkpointing (Binary)

Binary checkpoint format: header + named parameter blobs + optimizer state (`m`, `v` vectors per parameter + step counter). `saveCheckpoint(path)` / `loadCheckpoint(path)`.

---

## LoRA

`lib/ml/src/graph/lora.zig`

```
output = frozen_linear(x) + scale * (x @ A^T @ B^T)
```

- A is random-initialized, B is zero-initialized (so initial output equals the frozen output)
- `injectLoRA(graph, config)` finds `fused_linear_no_bias` nodes matching target patterns (e.g., `"q_proj"`, `"v_proj"`) and injects A/B parameter nodes + matmul + add
- Gradients flow through the injected ops automatically
- `mergeLoRA()` folds trained adapters back into base weights for inference

---

## Activation Checkpointing

`lib/ml/src/graph/checkpoint.zig`

Trades compute for memory by recomputing activations instead of storing them:

- Identifies checkpoint boundaries in the forward subgraph (every N layers or at attention outputs)
- For backward nodes referencing non-checkpoint activations, inserts recomputation chains from the preceding checkpoint
- Exposed through `TrainingConfig.checkpoint_config`
- Optional summary reporting: total forward activations, checkpointed activations, recomputable activations, savings ratio

Checkpointing is opt-in. In the current synthetic graph benchmark it increases step time while process RSS stays roughly flat — useful as a configurable memory lever.

---

## Distributed Training

`src/graph/distributed_training.zig`

Data-parallel training across multiple devices:

1. Each device runs forward+backward on its data shard
2. Gradients are averaged via `collective_ops.allReduceSum`
3. Each device applies `optimizer.step` with the averaged gradients

Threads through the same checkpoint configuration and checkpoint-analysis reporting as the single-device path.

---

## Training Features by Model Family

The fused-chunker embedder (`src/finetune/fused_chunker_train.zig`) is the reference implementation; other model families (LayoutLMv3, Reranker, ColQwen2, GLiNER2, Gemma4) progressively share these features.

| Feature | Fused Chunker | LayoutLMv3 Seq | LayoutLMv3 Token | Reranker LoRA | ColQwen2 | GLiNER2 LoRA | Gemma4 LoRA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| AdamW optimizer | yes | yes | yes | yes | yes | yes | yes |
| Layer-wise LR decay (LLRD) | yes | yes | yes | yes | yes | yes | yes |
| Global gradient norm clipping | yes | yes | yes | yes | yes | yes | yes |
| Gradient accumulation | yes | yes | yes | — | yes | yes | yes |
| Schedule-Free AdamW | yes | — | — | yes | yes | — | yes |
| DDP (MLX allReduce) | — | — | — | — | yes | — | yes |
| PJRT fast path | — | — | — | — | yes | — | yes |
| Cross-Batch Memory (XBM) | yes | — | — | — | — | — | — |
| NEFTune noise | yes | — | — | — | — | — | — |
| SPLADE sparse embeddings | yes | — | — | — | — | — | — |
| Matryoshka Repr. Learning (MRL) | yes | — | — | — | — | — | — |
| Mixed precision (bf16) | yes | — | — | — | — | — | — |
| LoRA+ | yes | — | — | — | — | — | — |
| Checkpoint resume | yes | — | — | — | — | — | — |
| Optimizer state save/load | yes | — | — | — | — | — | — |

### AdamW Optimizer

Default hyperparameters:

| Parameter | Default |
|---|---|
| β1 (first moment decay) | 0.9 |
| β2 (second moment decay) | 0.999 |
| ε (numerical stability) | 1e-8 |
| weight decay | 0.01 |

All model families use these defaults. The fused-chunker routes through `lib/ml` `optimizers.Optimizer`; LayoutLMv3, ColQwen2, and reranker families use an inline `applyAdamWInPlace` helper with the same constants.

Note: optimizer moment buffers (`m`, `v`) must be allocated once per run and carried across all epochs and steps — resetting them inside the per-epoch loop causes the adaptive learning rate to restart from scratch.

### Layer-wise Learning Rate Decay (LLRD)

LLRD assigns a lower learning rate to shallower (earlier) encoder layers, preventing catastrophic forgetting in lower layers while allowing task-specific top layers to adapt more quickly.

**Formula:** for layer index `i` (0 = shallowest), with `N` total layers and decay factor `d`:

```
lr_i = base_lr × d^(depth_from_top)
```

where `depth_from_top = N - 1 - i`. At `d = 1.0` all layers receive `base_lr` (disabled). At `d = 0.9`, each shallower layer is multiplied by another factor of 0.9.

**CLI flag:** `--llrd-decay <float>` (default: `1.0` = disabled)

### Global Gradient Norm Clipping

The global L2 norm is computed across all trainable parameter gradients simultaneously, then all gradients are scaled uniformly if the global norm exceeds the threshold. Joint clipping preserves gradient direction between layers; per-tensor clipping does not.

**CLI flag:** `--max-grad-norm <float>` (default: `1.0`; `0` to disable)

### Gradient Accumulation

Sums gradients across multiple forward/backward passes before applying a single optimizer step, simulating a larger effective batch size. With `--grad-accum N`, the optimizer step is deferred until `N` batches have been processed; gradients are normalized by `N` before the step.

**CLI flag:** `--grad-accum <int>` (default: `1` = disabled)

### Schedule-Free AdamW (Defazio 2024)

Eliminates the need for an explicit LR schedule by maintaining two parameter vectors:

- **z**: the "base iterate", updated by the standard gradient step
- **x**: the "Polyak average" of z, used as actual model weights for inference

During training: `x = (1 - c) * x + c * z`, where `c = (1 - β1) * lr`.

**CLI flag:** `--schedule-free` (boolean)

### Cross-Batch Memory (XBM) — fused-chunker only

A ring buffer of chunk embeddings from recent training batches. During InfoNCE contrastive loss computation, stored embeddings are concatenated with the current batch to expand the effective negative set. The buffer stores up to `--xbm-capacity` embedding vectors as a flat circular array. A monotonically increasing doc-ID offset prevents false-negative collisions across batches.

**CLI flag:** `--xbm-capacity <int>` (default: `0` = disabled)

### NEFTune Noise — fused-chunker only

Adds uniform random noise to encoder hidden states during the forward pass, scaled as `alpha / sqrt(seq_len * hidden_size)` so that noise magnitude is independent of sequence length and model width.

**CLI flag:** `--neftune-alpha <float>` (default: `0.0` = disabled)

### SPLADE Sparse Embeddings — fused-chunker only

Adds a vocabulary-space sparse vector head alongside the dense chunk embedding:

```
v[vocab] = max_over_tokens( log(1 + relu(hidden[t] @ W^T)) )
```

Uses a FLOPS regularization term to keep sparse vectors truly sparse.

**CLI flags:**
- `--splade` — enable the SPLADE head
- `--lambda-splade <float>` — SPLADE contrastive loss weight (default: `0.15`)
- `--lambda-flops <float>` — FLOPS sparsity regularization weight (default: `3e-5`)
- `--splade-focus-epoch <int>` — epoch at which SPLADE loss activates (default: `4`)

### Matryoshka Representation Learning (MRL) — fused-chunker only

Trains the model to produce useful embeddings at multiple truncated dimensions simultaneously. At inference time, embedding size can be traded against retrieval quality by truncating to any of the trained sizes.

**CLI flags:**
- `--mrl` — enable MRL
- `--mrl-dims <string>` — comma-separated list of embedding dimensions (default: `"768,256,128"`)

### Mixed Precision (bf16) — fused-chunker only

Enables bfloat16 computation for the MLX backend. Weights and activations are stored and multiplied in bf16, with gradient accumulation in f32.

**CLI flag:** `--mixed-precision` (boolean; MLX backend only)

### LoRA+ — fused-chunker only

Applies a higher learning rate to the LoRA B matrix than to the A matrix, since B is initialized to zero and must learn a larger signal in early training.

**Formula:** B-matrix LR = `R * base_lr`; A-matrix LR = `base_lr`. At `R = 1.0` this reduces to standard LoRA.

**CLI flag:** `--lora-plus-ratio <float>` (default: `1.0`)

---

## SafeTensors Checkpoint Format

```
[8 bytes: header_size as u64 little-endian]
[header_size bytes: UTF-8 JSON]
[tensor data: concatenated f32 values, little-endian]
```

The JSON header contains per-tensor metadata: `dtype`, `shape`, and `data_offsets` relative to the start of the data section. Compatible with the Python `safetensors` library.

`FusedTrainer.loadCheckpoint` first attempts to parse as SafeTensors; if that fails it falls back to the legacy binary format.

**Optimizer state persistence:** When `--save-optimizer-state` is passed, AdamW first and second moment buffers (`adam_m_*`, `adam_v_*`) and the step counter (`adam_step`) are saved as a separate SafeTensors file. For Schedule-Free AdamW, `z` and `v` buffers are saved as `sf_z_*` and `sf_v_*`.

---

## Benchmarking

`src/bench/training.zig`

Two focused measurements:

- **Optimizer microbenchmark**: scalar AdamW reference vs. fused/SIMD AdamW slice path
- **Graph benchmark**: synthetic train graph with configurable depth/width/batch, optional GELU activations, checkpoint interval sweep (`off`, `2`, `4`, `8`), average timing and peak resident-memory reporting

```sh
zigup run master build bench-training
zigup run master build bench-training -- --mode both
zigup run master build bench-training -- --mode graph --graph-activation gelu --checkpoint-sweep
```

Current results:
- The fused/SIMD AdamW path improves the isolated optimizer kernel by ~1.25x–1.30x
- End-to-end graph time is dominated by graph execution rather than optimizer work
- Checkpointing is structurally correct: recompute cost is more visible than memory savings in synthetic benchmarks

---

## Family Command Reference

Use `antfly inference finetune run <recipe.json>` for normal post-training work.

The family-specific `zig build <tool> -- ...` commands below are still useful
for implementation work, debugging, and narrow backend verification, but they
are backend-facing reference surfaces now, not the primary public workflow.

---

## LayoutLMv3 PEFT Surface

This section documents the LayoutLMv3 backend commands used by the recipe
runner.

Preferred user path:

```sh
antfly inference finetune run recipe_layoutlmv3_lora_token.json
antfly inference finetune run recipe_layoutlmv3_lora_token.json --dry-run
```

### Build Steps

| Step | Purpose |
|------|---------|
| `bootstrap-layoutlmv3-lora` | Initialize LoRA bundle from base model |
| `inspect-layoutlmv3-lora-bundle` | Inspect a bootstrapped or trained bundle |
| `materialize-layoutlmv3-checkpoint` | Merge LoRA adapters into base weights |
| `train-layoutlmv3-lora-one-step` | Bounded single-step LoRA training |
| `train-eval-layoutlmv3-lora-sequence` | Bounded sequence classification train/eval |
| `train-eval-layoutlmv3-lora-token` | Bounded token classification train/eval |
| `run-layoutlmv3-lora-smoke-workflow` | Full bootstrap→train→inspect→materialize chain |
| `test-layoutlmv3-finetune` | Unit test |

Local verification:

```bash
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-local-cache \
zigup run master build test-layoutlmv3-finetune -Dblas=false -Donnx=false -Dmlx=false
```

### Expected Inputs

Base model directory: must contain `config.json` and `model.safetensors`.

Adapter directory after bootstrap or training:
- `adapter_model.safetensors`
- `adapter_config.json`
- After sequence training: `layoutdoc_sequence_head.safetensors`, `sequence_head_config.json`
- After token training: `layoutdoc_token_head.safetensors`, `token_head_config.json`

Dataset JSONL fields:
- `document_id`, `page_id`, `image_path`, `tokens` (required)
- `label` (required for sequence examples)
- `token_labels` (required for token examples; count must match token count)
- Optional: `runtime_token_weights`, `teacher_token_hidden`, `teacher_token_probs`

Each token: `{ "text": "...", "bbox": [x0, y0, x1, y1] }`. Bbox values must be within `0..1000`.

### Core Commands

Bootstrap a LoRA bundle:

```bash
zigup run master build bootstrap-layoutlmv3-lora -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/bootstrap_dir \
  8 \
  16
```

Inspect a bundle:

```bash
zigup run master build inspect-layoutlmv3-lora-bundle -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/adapter_dir \
  /tmp/layoutlmv3_lora_inspect.json
```

Materialize a merged checkpoint:

```bash
zigup run master build materialize-layoutlmv3-checkpoint -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/trained_adapter_dir \
  sequence \
  /path/to/materialized_dir \
  /tmp/layoutlmv3_materialize_report.json
```

Run bounded sequence training:

```bash
zigup run master build train-eval-layoutlmv3-lora-sequence -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/bootstrap_dir \
  /path/to/train.jsonl \
  /path/to/val.jsonl \
  /path/to/sequence_out \
  128 \
  0.001 \
  64 \
  4 \
  @layoutlmv3_sequence_top3
```

Run bounded token training:

```bash
zigup run master build train-eval-layoutlmv3-lora-token -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/bootstrap_dir \
  /path/to/train.jsonl \
  /path/to/val.jsonl \
  /path/to/token_out \
  128 \
  0.001 \
  64 \
  4 \
  @layoutlmv3_token_top3
```

### Smoke Workflow

The smoke workflow chains bootstrap, train, inspect, and materialize in one command:

```bash
zigup run master build run-layoutlmv3-lora-smoke-workflow -Dblas=false -Donnx=false -Dmlx=false -- \
  /path/to/layoutlmv3_base \
  /path/to/train.jsonl \
  /path/to/val.jsonl \
  sequence \
  /path/to/output_root \
  8 \
  16 \
  32 \
  0.001 \
  16 \
  2 \
  @layoutlmv3_sequence_top3
```

For token classification, replace `sequence` with `token` and use `@layoutlmv3_token_top3`.

Smoke workflow writes:
- `bootstrap/`, `trained/`, `materialized/`
- `smoke_workflow_report.json`
- `training_config.json`, `training_report.json`, `run_status.json`

Report contents: dataset stats, bootstrap summary, initial adapter inspection, train/eval summary, merged materialization summary.

### Layer Scope Presets

| Preset | Use |
|--------|-----|
| `@layoutlmv3_token_top1` | Single top-layer token scope |
| `@layoutlmv3_token_top3` | Top-3 layers, token |
| `@layoutlmv3_sequence_top3` | Top-3 layers, sequence |

### Artifact File Contracts

| Artifact | File |
|----------|------|
| Base checkpoint | `model.safetensors` |
| LoRA adapter | `adapter_model.safetensors` |
| Sequence head | `sequence_head.safetensors` + `sequence_head_config.json` |
| Token head | `token_head.safetensors` + `token_head_config.json` |
| Merged bundle | `model.safetensors` + `config.json` + tokenizer files |

### Runbook for Larger Machines

1. Confirm the base bundle contains `config.json` and `model.safetensors`.
2. Confirm the train/val JSONL data loads cleanly and matches the expected task.
3. Run `test-layoutlmv3-finetune` once on that machine.
4. Run `run-layoutlmv3-lora-smoke-workflow` with small example counts first.
5. Inspect `smoke_workflow_report.json`, `bootstrap/adapter_config.json`, and the trained head config.
6. If the smoke run is healthy, increase the bounded example counts and epochs.

### Limitations

- No full-backbone LayoutLMv3 fine-tuning
- No distributed training or mixed precision on this path
- Task heads are bounded Antfly inference-owned implementations

---

## CLI Reference

### Fused Chunker (`src/train_fused_chunker.zig`)

```
usage: train-fused-chunker --data <path> --output <dir> [options]

  --data <path>             JSONL data path (file or directory)
  --output <dir>            Output directory for checkpoints
  --model-dir <dir>         Model directory (tokenizer + encoder weights)
  --epochs <n>              Number of epochs (default: 10)
  --batch-size <n>          Batch size (default: 16)
  --lr <f>                  Learning rate (default: 1e-4)
  --hidden-size <n>         Encoder hidden size (default: 768)
  --max-seq-len <n>         Max token sequence length (default: 384)
  --checkpoint-every <n>    Save checkpoint every N epochs (0=disabled)
  --split <name>            Dataset split name filter (default: "train")
  --seed <n>                Random seed (default: 42)
  --lora-rank <n>           LoRA rank (default: 0 = disabled)
  --intermediate-size <n>   ModernBERT intermediate_size (default: 1152)
  --backend native|mlx|auto   Compute backend (default: auto)
  --grad-accum <n>          Gradient accumulation steps (default: 1)
  --schedule-free           Use Schedule-Free AdamW
  --neftune-alpha <f>       NEFTune noise magnitude (default: 0.0=disabled)
  --xbm-capacity <n>        Cross-Batch Memory capacity (default: 0=disabled)
  --llrd-decay <f>          Layer-wise LR decay (default: 1.0=disabled)
  --lora-plus-ratio <f>     LoRA+ B/A LR ratio (default: 1.0=disabled)
  --length-bucketing        Enable length bucketing
  --bucket-size <n>         Bucket window size (default: 256)
  --mixed-precision         Enable bf16 mixed precision (MLX only)
  --splade                  Enable SPLADE sparse embedding head
  --lambda-splade <f>       SPLADE contrastive loss weight (default: 0.15)
  --lambda-flops <f>        SPLADE FLOPS regularization weight (default: 3e-5)
  --splade-focus-epoch <n>  Epoch when SPLADE activates (default: 4)
  --mrl                     Enable Matryoshka Representation Learning
  --mrl-dims <s>            Comma-separated MRL dims (default: "768,256,128")
  --resume-from <path>      Resume training from a checkpoint file
  --save-optimizer-state    Save Adam optimizer state alongside each checkpoint
```

### LayoutLMv3 Sequence and Token

These are backend implementation commands. Prefer a `layoutlmv3` recipe unless
you are debugging this family surface directly.

```
usage: train-eval-layoutlmv3-lora-sequence <base_model_dir> <adapter_model_dir>
    <train_jsonl_or_dir> <val_jsonl_or_dir> <out_dir>
    [max_train_examples]   default: 128
    [learning_rate]        default: 0.001
    [max_val_examples]     default: 64
    [epochs]               default: 4
    [layer_name|@layoutlmv3_token_top1|@layoutlmv3_token_top3|@layoutlmv3_sequence_top3]

Flags:
  --max-grad-norm <f>    Gradient norm clipping threshold (default: 1.0)
  --llrd-decay <f>       Layer-wise LR decay factor (default: 1.0=disabled)
  --grad-accum <n>       Gradient accumulation steps (default: 1)
```

Token classification uses identical positional and flag interface (`train-eval-layoutlmv3-lora-token`).

### Reranker Surrogate (`src/train_eval_reranker_lora_surrogate.zig`)

This is a backend command behind the `lora-sft` / `qlora-sft` reranker recipe.

```
usage: train-eval-reranker-lora-surrogate <model-dir> <adapter-dir>
    <head-dir-or-file> <train-jsonl-or-dir> <eval-jsonl-or-dir> <out-dir>
    [train-split] [eval-split]

Flags:
  --backend auto|native|mlx   Compute backend (default: auto)
  --max-examples <n>        Max training examples (default: 128)
  --epochs <n>              Number of epochs (default: 1)
  --learning-rate <f>       Learning rate (default: 0.001)
  --layer-name <name>       Scope to a specific layer name
  --max-grad-norm <f>       Gradient norm clipping threshold (default: 1.0)
  --schedule-free           Enable Schedule-Free AdamW
```

Note: gradient accumulation is defined in `SurrogateTrainOptions` but the surrogate CLI does not yet expose `--grad-accum`.

### ColQwen2 LoRA Bundle

This is the backend train/eval command behind `vlm-retrieval` recipes.

```
usage: train-eval-colqwen2-lora-bundle <base_model_dir> <adapter_model_dir>
    <prepared_inputs_json> <out_dir> [options]

Flags:
  --lr, --learning-rate <f>     Learning rate (default: 0.001)
  --max-examples <n>            Max examples per epoch (default: 32)
  --epochs <n>                  Number of epochs (default: 1)
  --layer-name, --layer <str>   Scope to layer name or @colqwen2_focus_top3
  --max-grad-norm <f>           Gradient norm clipping threshold (default: 1.0, 0=disabled)
  --grad-accum <n>              Gradient accumulation steps (default: 1)
  --llrd-decay <f>              Layer-wise LR decay factor (default: 1.0=disabled)
  --schedule-free               Enable Schedule-Free AdamW (default: false)
```

### GLiNER2 LoRA (Real Autodiff)

GLiNER2 LoRA fine-tuning runs through `train-gliner2-autodiff`: full
DeBERTa-v3 encoder forward + autodiff with the upstream GLiNER2 total loss
(`structure + classification + count`) via `--objective gliner2-total-loss`.
The earlier cached probe-surrogate bundle route
(`train-eval-gliner2-lora-bundle`, MSE against deterministic probe targets)
was removed — it was a smoke fixture, not real GLiNER2 training. Python↔Zig
loss parity is gated by
`scripts/compare_gliner2_lora_python_zig.py --strict` (see
`zig/e2e/inference/test_gliner2_lora_parity.py`).

Adapters are saved in PEFT-compatible format
(`adapter_model.safetensors` + `adapter_config.json`).

### Gemma4 LoRA

Gemma4 E2B/E4B LoRA is an experimental, single-device, text-only lane. It is
not production-ready. The authoritative current-state, validation, and release
gate document is [GEMMA4.md](GEMMA4.md); this section is only the command and
data-schema index.

The recipe planner and convenience workflows now use the mandatory four-step
held-out-eval contract:

```sh
antfly inference finetune run recipe_gemma4_lora.json
antfly inference finetune run recipe_gemma4_lora.json --dry-run
```

The supported-intent flow has four steps: prepare training inputs, prepare a
separate evaluation artifact, bootstrap the adapter, then train/evaluate with
`--eval-prepared`. `--trainer auto` is an autodiff alias; it never falls back to
surrogate training. The legacy explicit surrogate spelling returns a typed
unsupported error and is not a runnable public mode.

Public prepare/train paths reject multimodal/projector inputs before backend
work. Historical multimodal tools remain diagnostic only and are not an
accepted training lane.

New adapter directories contain a public-schema `adapter_config.json`
(`task_type = "CAUSAL_LM"`) and the Antfly-only
`antfly_finetune_manifest.json` sidecar. The Safetensors payload currently uses
Antfly's weight-qualified tensor keys, so do not describe the raw training
bundle as directly interchangeable. Export a standard `peft-qv` or
`text-all-linear` artifact through the public immutable conversion boundary:

```sh
antfly inference finetune adapter export gemma4-peft \
  --model <base_model_dir> --adapter <trained_adapter_dir> \
  --out <new_peft_adapter_dir>
```

The export preserves F32 payload bytes, emits stock PEFT tensor keys, and adds
a hash-bound `antfly_peft_export.json` sidecar. A pinned structural smoke test
loads, saves, and reloads the result with PEFT at exact tensor and logit
equality. Real E2B/E4B numerical interoperability remains a release gate.

#### Chat Dataset Contract

Gemma text finetuning now accepts a chat-native dataset schema, `gemma_chat/v1`, in addition to the legacy flat row formats.

Each `gemma_chat/v1` JSONL row may include:
- `schema`: must be `gemma_chat/v1`
- `id`: optional row identifier
- `split`: optional split name
- `messages`: ordered turns with `role` in `system|user|assistant|tool`
- `tools`: optional tool specifications for provenance/documentation
- `metadata`: optional fields such as `policy_version`, `source`, and
  `group_id` (or the accepted `group` alias)

Assistant turns may include `tool_calls`, and tool turns may include `tool_call_id` plus `name`. On the Gemma rendering path:
- assistant text and assistant tool-call blocks are supervised
- tool responses are injected as context and masked from labels
- system text is merged into the first user turn, matching Gemma prompt conventions

This gives Gemma a single conversation contract for:
- plain SFT chat
- tool-calling traces
- multi-turn assistant/tool handoffs

**Step 1 — prepare text dataset:**
```
usage: prepare-gemma4-text-dataset <dataset-path> <split|-> <out_csv_path> <out_summary_path> [max_examples]
```

Accepted row shapes:
- `gemma_chat/v1` rows with `messages`
- legacy instruction rows with `prompt` or `instruction`, optional `input`, and `response|completion|output`
- legacy completion rows with `text`

Legacy rows are coerced internally into the chat contract before tokenization, so existing datasets continue to work.

**Step 1b — prepare multimodal dataset:**
```
usage: prepare-gemma4-multimodal-dataset <dataset-path> <split|-> <out_csv_path> <out_summary_path> [max_examples]
```

The multimodal preparation path now accepts the same `messages` contract for text turns and can extract image paths from:
- top-level `image_path|image|image_file|file_name`
- top-level `images`
- `messages[].content[]` image parts in `gemma_chat/v1` rows

The current multimodal converter still materializes a flat CSV (`image,prompt,response`) artifact, so it shares the conversation schema at ingestion time while keeping the existing output contract.

**Prepare training and evaluation inputs separately:**
```
usage: prepare-gemma4-lora-inputs <model_dir> <dataset_path> <split|-> <out_summary_json> [options]

  --max-examples N    Maximum number of examples to prepare (default: 0 = all)
  --max-seq-len N     Maximum sequence length in tokens (default: 512)
  --dataset-revision R Immutable source revision (default: resolved split digest)
```

Run this command once for the training source and once for a genuinely disjoint
evaluation source/split. The training command requires the second artifact via
`--eval-prepared`. Sequence admission is currently bounded to
`1..min(model context, 2048)` before backend construction.

#### Prepared Inputs v6

`prepare-gemma4-lora-inputs` emits `gemma4_prepared/v6`. It uses causal
generation tokenization: the rendered chat owns one literal BOS, no implicit
EOS is appended, and every assistant turn is supervised even when the
tokenizer does not expose offsets. Each prepared example records:
- legacy prompt/response token views for format compatibility
- full rendered `input_ids`
- `labels` with non-assistant and tool-response tokens masked to the ignore label
- `num_input_tokens`
- `num_supervised_tokens`
- `turn_count`
- `has_tool_calls`
- `has_tool_messages`
- optional `policy_version`
- source id and group id
- canonical source-record and rendered-chat SHA-256 digests
- content digests for every referenced image or audio payload

The summary also records:
- base-artifact, tokenizer-asset, and chat-template identity digests
- source dataset path/content digest, split, and immutable revision
- a schema-aware canonical prepared-example digest
- `max_input_tokens`
- `max_supervised_tokens`
- `examples_with_tool_calls`
- `examples_with_tool_messages`
- `examples_with_multiturn`

The loader recomputes sequence, supervision, modality, and aggregate counters;
validates vocabulary bounds; and compares train/eval identities to the selected
model and adapter. It rejects prepared-token, canonical source-row, and group
overlap. Prepared and standalone eval JSON are file-synced and atomically
published without replacement. They remain whole-buffer artifacts with a 128
MiB load ceiling, so streaming immutable shards are still required for
production-scale datasets. `gemma4_prepared/v4` and `/v5` are read for
migration/inspection compatibility but are rejected by the production trainer;
only v6 supplies the causal-tokenization contract used by release evidence.

**Optional — materialize teacher targets:**
```
usage: materialize-gemma4-teacher-targets <base_model_dir> <prepared_inputs_json> <out_summary_json> [options]

  --top-k N              Teacher tokens per row (default: 8)
  --temperature F        Temperature applied before top-k softmax (default: 1.0)
  --max-examples N       Maximum examples to materialize (default: 0 = all)
  --backend native|mlx   Teacher inference backend (default: native)
```

This tool runs the full Gemma4 teacher model over prepared inputs and writes sparse row-major `teacher_top_k_token_ids` and `teacher_top_k_probs` into the output prepared-input JSON. For multimodal prepared inputs, pass `--gguf-projector <projector.gguf>` unless the prepared summary records a valid projector path. The autodiff trainer consumes those soft targets when present, which is the first distillation path for recursive LoRA compression.
Teacher probabilities are produced after applying `--temperature`, and the trainer applies the standard distillation `T^2` loss scale from each example's `teacher_temperature`.

This utility does not yet bind the supplied teacher model to the prepared
artifact's model provenance. Treat teacher-target materialization as
experimental until the teacher/base digest is validated and persisted.

**Optional — materialize compressed recursive base:**
```text
usage: materialize-gemma4-recursive-base <base_model_dir> <recursive_adapter_dir> <out_dir> [options]
```

The build step is `zig build materialize-gemma4-recursive-base -- <base_model_dir> <recursive_adapter_dir> <out_dir>`. It writes a compressed recursive base `model.safetensors` containing non-layer tensors plus only the physical shared layer block recorded in the recursive adapter metadata. The output also includes copied HF support files and `recursive_lora_base_config.json` with tensor counts, byte sizes, and compression ratio. The copied `config.json` keeps the original logical layer count; the recursive adapter metadata remains the runtime contract for mapping logical layers to physical tensors.

For the recursive compression path, the bounded smoke workflow is:

```text
usage: run-gemma4-recursive-lora-smoke-workflow <base_model_dir> <output_root> [options]
```

The corresponding developer build step is `zig build run-gemma4-recursive-lora-smoke-workflow -- <base_model_dir> <output_root> ...`. Its current implementation prepares a distinct evaluation dataset/artifact and passes it to the typed training operation. It remains experimental until the integrated tests and pinned real-model recursive campaign pass.
Successful real runs write `<output_root>/recursive_smoke_results.json` with adapter sizes, before/after loss, teacher coverage, elapsed time, and supervised-token throughput.
For Gemma4 E2B, use `--recursive-shared-block-size 5`; the text stack has 35 layers and the local/full attention pattern repeats every five layers. The current recursive smoke defaults to attention-only targets (`q_proj,k_proj,v_proj,o_proj`) because E2B MLP weights become double-wide after layer 15.

For baseline-vs-recursive comparison sweeps:

```text
usage: run-gemma4-recursive-lora-sweep <base_model_dir> <output_root> [options]
```

The build step is `zig build run-gemma4-recursive-lora-sweep -- <base_model_dir> <output_root> ...`. It runs normal LoRA baselines plus recursive variants across rank, shared-block-size, and teacher-temperature grids, then writes `<output_root>/recursive_lora_sweep_comparison.json`.

To turn a completed sweep into a pass/fail recommendation:

```text
usage: analyze-gemma4-recursive-lora-sweep <comparison_json> <out_dir> [options]
```

The build step is `zig build analyze-gemma4-recursive-lora-sweep -- <comparison_json> <out_dir> ...`. It writes `recursive_lora_sweep_decision.json` and `.md` using explicit loss, adapter-size, compressed-base-size, teacher-coverage, and throughput thresholds.

**Step 4 — train/eval (canonical public interface):**
```
antfly inference finetune train gemma4-lora \
  --model <base_model_dir> --adapter <adapter_model_dir> \
  --train-prepared <prepared_train.json> --eval-prepared <prepared_eval.json> \
  --out <out_dir> --backend native|metal|cuda [options]

Flags:
  --trainer auto|autodiff       Trainer implementation (default: autodiff; auto is an autodiff alias)
  --lr, --learning-rate <f>     Learning rate (default: 0.001)
  --max-examples <n>            Max examples per epoch (default: 32)
  --eval-max-examples <n>       Max examples for before/after evaluation
  --epochs <n>                  Number of epochs (default: 1)
  --layer-name, --layer <str>   Rejected by the autodiff trainer until scoped semantics are implemented
  --max-grad-norm <f>           Gradient norm clipping threshold (default: 1.0, 0=disabled)
  --grad-accum <n>              Gradient accumulation steps (default: 1)
  --activation-checkpoint-interval <n>  Recompute every N layer boundaries
  --llrd-decay <f>              Rejected by autodiff unless left at 1.0
  --schedule-free               Rejected by autodiff
  --backend native|metal|cuda   Required; there is no automatic backend fallback
```

Trainer mode behavior:
- `auto` is retained only as an alias for `autodiff`; it never falls back to a
  successful surrogate run
- `autodiff` supports the Gemma4 text graph contract, including PLE,
  sliding/full attention, per-layer GQA/RoPE, and shared-KV donor reuse
- compiled Metal/CUDA evaluation uses a loss-only graph and does not allocate
  gradients or Adam moments
- public multimodal/projector training is rejected before backend work
- MoE and non-RoPE Gemma configs remain unsupported
- `autodiff` uses token-level next-token cross-entropy over the prepared `labels` mask, including assistant tool-call output while masking tool responses
- `autodiff` reuses the incoming Gemma adapter bundle as initialization and
  writes the trained adapters back out with PEFT config semantics plus an
  Antfly provenance/key-format sidecar

Default Gemma4 selection: exact `text-all-linear` paths. Use `peft-qv` for the
smaller Q/V baseline. Exact paths and the preset are persisted in the Antfly
sidecar; standard PEFT fields remain in `adapter_config.json`.

Outputs:
- `<out_dir>/adapter_model.safetensors` (Antfly weight-qualified keys)
- `<out_dir>/adapter_config.json` with `task_type = "CAUSAL_LM"`
- `<out_dir>/antfly_finetune_manifest.json` with model provenance, target
  policy, and tensor-key format
- `<out_dir>/train_eval_report.json` — `before`, per-epoch `epoch_history`, `after` metrics for the selected trainer
- `<out_dir>/training_config.json` and `<out_dir>/training_report.json`
- `<out_dir>/run_manifest.json` — closed completion ledger with the size and
  SHA-256 digest of every other regular root payload

For stock PEFT deployment, export the completed directory to a new path with
`adapter export gemma4-peft`. The destination contains
`adapter_model.safetensors`, `adapter_config.json`, and
`antfly_peft_export.json`; the source training artifact is unchanged.

`<out_dir>` must not exist. The adapter and every report are written to a
sibling staging directory and published together with one no-replace rename.
Gemma4 Metal/CUDA train and eval select the strict training executor themselves;
no executor environment opt-in is required by the public CLI. Explicit
diagnostic disable/parity flags remain fail-closed. Every step rejects native/unsupported
partitions, interpreter/runtime fallback, true host outputs, undeclared or
graph-execution uploads, explicit runtime-input transfers,
gather/reduce/cache promotions, or non-resident gradients before optimizer
mutation. Stored BF16
Safetensors use the dedicated device-only frozen-linear input-gradient kernel;
autodiff prunes frozen base-weight gradients that cannot reach a requested
LoRA parameter. The default eight-row sparse vocabulary-loss chunk additionally
selects an exact-arithmetic M8/N32/K64 Metal input-gradient tile only for its
large aligned BF16 shape; other shapes retain the generic route. Diagnostic
operators can force the generic route with
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SMALL_ROWS=1`. Packed Q4_0, Q4_K, and
Q6_K frozen-linear input gradients now
pass direct runtime and graph-executor tests without host dequantization, but
whole-model GGUF training and the Gemma4 `qlora-sft` recipe remain fail-closed
until a real optimizer/memory/parity/quality campaign passes. Stored F16 also
remains unsupported. See
[GEMMA4.md](GEMMA4.md) for the authoritative release gates.

Standalone loss-only evaluation is available as:

```sh
antfly inference finetune eval gemma4-lora \
  --model <base_model_dir> --adapter <adapter_dir> \
  --prepared <prepared_eval.json> --out <eval.json> --backend native|metal|cuda
```

It validates the same model/adapter/prepared contracts and atomically publishes
an immutable report without allocating gradients or Adam moments.

Activation checkpointing is recomputation only. The low-level trainer can now
save/inspect/restore weights, Adam state, incomplete accumulation, counters,
and caller-owned epoch/cursor/RNG progress, including resident Metal slot
restoration. The Gemma4 CLI exposes `--checkpoint-path`,
`--checkpoint-every-epochs`, and `--resume`; recipes expose one mutable
epoch-boundary state through `checkpoint.every_epochs` and
`checkpoint.resume_path` while rejecting `keep_last`. Resume binds the exact
run fingerprint and must publish to a new immutable output directory. It has
not yet passed a real interrupted E2B/E4B Metal trajectory.

For the pinned HF/PEFT correctness oracle, same-Mac MLX-LM performance harness,
and Unsloth's separate CUDA-only comparison role, see
[GEMMA4_ORACLE.md](GEMMA4_ORACLE.md).

---

## Run Contract

The following machine-readable workflow artifacts are emitted by the unified
recipe runner and by the lower-level family smoke-workflow entrypoints.

Contract versions: `run_status/v1`, `training_config/v1`, `training_report/v1`

Applies to:
- `antfly inference finetune run <recipe.json>`
- `antfly inference finetune smoke-fast`
- `run-layoutlmv3-lora-smoke-workflow`
- `run-gliner2-boundary-task-head-smoke-workflow`
- `train-eval-layoutlmv3-lora-sequence`
- `train-eval-layoutlmv3-lora-token`
- `train-eval-colqwen2-lora-bundle`
- `train-eval-gliner2-top-layer-boundary-task-head`

### `run_status.json`

Written by smoke workflows.

Required top-level fields: `contract_version`, `status`, `task`, `out_dir`, `resume_from`, `actions`, `derived`, `artifacts`

Required `derived` fields: `outcome_code`, `alerts`, `metric_summary`

Required `artifacts` fields: `report`, `best`, `latest`, `final`

Semantics:
- `status` is one of `running`, `failed`, or `completed`
- `artifacts.report` points to `training_report.json`
- `best` and `latest` point at the workflow's trained artifact dir
- `final` points at the materialized output dir
- `alerts` is currently an empty list placeholder

### `training_config.json`

Written by train/eval entrypoints and smoke workflows.

Required fields: `contract_version`, `artifact_family_version`, `task`, `inputs`, `training` (or workflow-specific config object)

Optional but recommended: `backend_policy`, `distributed`, `run_plan`, `output_root`

Semantics:
- `inputs` records the user-facing source paths
- `training` records bounded hyperparameters and layer selection
- `backend_policy.selected` is the backend actually requested/used by the CLI
- `backend_policy.preferred` is the current antfly inference default for that entrypoint

### `training_report.json`

Written by train/eval entrypoints and smoke workflows.

Required fields: `contract_version`, `artifact_family_version`, `task`, `report` (or `summary`)

Optional but recommended: `backend_policy`, `distributed`

The `report`/`summary` object contains the workflow-specific bounded metrics payload. The contract intentionally does not normalize all model-family metrics into a single schema.

### Compatibility Policy

Stable: every required field listed above.

Current non-goals: `run_actions.json`, deferred quantize/shard promotion, full `shared_cache` and `loader_shared_cache` parity with `gopeft-zig`, family-wide metric normalization across all trainers.

---

## Key Files

| File | Purpose |
|------|---------|
| `lib/ml/src/graph/autodiff.zig` | Reverse-mode AD, ~25 VJP rules |
| `lib/ml/src/graph/lower.zig` | Fused → primitive lowering via vjp_alternate |
| `lib/ml/src/graph/grad_check.zig` | Finite-difference gradient verification |
| `lib/ml/src/graph/optimizers.zig` | SGD/Adam/AdamW + LR schedules |
| `lib/ml/src/graph/lora.zig` | LoRA adapter injection and merging |
| `lib/ml/src/graph/checkpoint.zig` | Activation checkpointing pass |
| `src/graph/training.zig` | Training step orchestration |
| `src/graph/training_loop.zig` | TrainingWeightStore + TrainingLoop + checkpoints |
| `src/graph/distributed_training.zig` | Data-parallel distributed training |
| `src/bench/training.zig` | Native optimizer / checkpoint benchmark |
| `src/finetune/fused_chunker_train.zig` | Reference fused-chunker training implementation |
| `src/ops/blas_compute.zig` | BLAS primitive op implementations |
| `src/graph/interpreter.zig` | Primitive op dispatch |
```

---

## Validation & Testing Plan

### Unit Tests (266 tests, all pass)

24 standalone files under `src/finetune/` exercise individual modules:
LoRA math, preference losses, optimizers, quantization, sequence packing,
chat templates, training guards, Hypura memory management, etc.

### E2E Integration Tests (6 tests)

Each validates the full level-3 pipeline: graph → LoRA injection → autodiff → execution → loss → optimizer step. Uses tiny configs with random weights; asserts loss decreases over 5 training steps.

| Test | Architecture | Params | Head |
|------|-------------|--------|------|
| `test_bert_e2e.zig` | BERT (2L, 4H, H=64) | 37 | MSE |
| `test_qwen2_e2e.zig` | Qwen2 (2L, 4H/2KV, H=32) | 26 | pooled MSE |
| `test_deberta_e2e.zig` | DeBERTa-v3 (2L, 4H, H=64) | 38 | MSE |
| `test_gliner2_e2e.zig` | DeBERTa + NER head (5 classes) | 40 | token CE |
| `test_fused_chunker_e2e.zig` | ModernBERT + boundary MLP | 39 | 2-class CE |
| `test_layoutlmv3_e2e.zig` | LayoutLMv3 + token cls (5 classes) | 43 | token CE |

### Real-World Validation Plan

**Phase 1 — BERT reranker (smallest model, fastest iteration)**
- Model: `bert-base-uncased` (110M params)
- Data: MS MARCO reranking subset (~1K pairs)
- Goal: loss converges, saved LoRA adapters produce correct scores at inference
- Metric: MRR@10 on eval split
- Compare: wall-clock speed vs HF PEFT on same hardware

**Phase 2 — DeBERTa + GLiNER2 (NER)**
- Model: `deberta-v3-base` (184M params)
- Data: CoNLL-2003 or custom NER dataset
- Goal: F1 on entity spans matches or exceeds HF PEFT baseline
- Validates: disentangled attention (C2C + C2P + P2C) gradients on real data

**Phase 3 — Qwen2 decoder (large vocab, causal)**
- Model: `Qwen2-0.5B` or `Qwen2-1.5B`
- Data: small instruction-tuning dataset (Alpaca subset)
- Goal: perplexity on held-out set decreases; generated text is coherent
- Validates: GQA fan-out, RoPE, SwiGLU, chunked CE on 150K+ vocab

**Phase 4 — LayoutLMv3 (document AI)**
- Model: `layoutlmv3-base` (133M params)
- Data: FUNSD or custom document field-extraction dataset
- Goal: token-level F1 on field labels
- Validates: 2D bbox embeddings, attention masking on padded layouts

**Phase 5 — ColQwen2 multimodal retrieval**
- Model: ColQwen2-VL (text-only MVP, then with vision tower)
- Data: document retrieval pairs with relevance scores
- Goal: late-interaction retrieval scores correlate with ground truth

### Known issues likely to surface during validation
- Weight name mismatches between graph builders and real HF checkpoints
- Tokenizer integration (CLI driver uses placeholder char-level; needs HF tokenizer)
- Memory pressure on larger models (good test for Hypura stack)
- `head_dim` edge cases on non-standard model sizes (e.g., Qwen2-7B has head_dim=128)

---
