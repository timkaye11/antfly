# Inference scripts

Model-specific diagnostics and release tooling live in family directories;
shared inference, code-generation, and kernel utilities remain at this level.
The model documentation owns the detailed commands and acceptance criteria.

## Model families

- `gemma4/` contains Gemma4 Metal/CUDA benchmarks, qualification gates,
  speculative decoding checks, LoRA workflows, tests, and fixtures.
- `gliner2/` contains GLiNER2 training, parity, release-readiness, hardware
  qualification, tests, and fixtures.
- `mxbai/`, `florence2/`, `clipclap/`, and `layoutdoc/` contain the focused
  verification scripts for those model families.

## Gemma4 Metal interrupted/resumed acceptance

`gemma4/qualify_gemma4_metal_resume.py` runs a real strict-Metal Gemma4 LoRA job
uninterrupted, kills an identical job immediately after a durable epoch-boundary
checkpoint, resumes it in a fresh output directory, and publishes a PASS only
when the final adapter is byte-identical and the post-resume optimizer trajectory
matches with zero native/interpreter fallback. The output root must not already
exist.

```bash
python3 zig/pkg/inference/scripts/gemma4/qualify_gemma4_metal_resume.py \
  --binary zig/zig-out/bin/antfly \
  --model /path/to/gemma4-model \
  --adapter /path/to/seed-adapter \
  --train-prepared /path/to/train-prepared.json \
  --eval-prepared /path/to/eval-prepared.json \
  --output-dir /path/to/new/qualification-root
```

`--model` may also be one pinned decoder `.gguf` file. That research lane must
be admitted explicitly with `--experimental-gguf-qlora`; the harness then
binds `ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA=1` into the reported contract,
hashes and snapshots the GGUF as an immutable input, and accepts the absence of
standalone tokenizer sidecars only when both final adapter bundles omit them.

`gemma4/qualify_gemma4_preference_resume.py` is the DPO/GRPO counterpart. It runs one
uninterrupted job, sends SIGTERM to a second process only after its
content-addressed sidecar and checkpoint are durable, retains and later
re-verifies that exact interruption boundary, and resumes the checkpoint in a
third immutable root. The default targets an epoch boundary;
`--interrupt-after-examples` qualifies an eager DPO/GRPO mid-epoch boundary,
including a one-epoch trajectory. A PASS requires byte-identical final
checkpoint and adapter trees, exact training and discrete evaluation traces,
and only the documented narrow terminal Metal GRPO KL tolerances.
`--expected-final-adapter-sha256` additionally binds recovery to a predeclared
accepted adapter digest. Use
`--direct-gguf-training` for the explicitly experimental canonical E2B GGUF
lane. Use `--incremental-kv`, `--incremental-kv-clone-prompt-tail`, and
`--incremental-kv-shadow-exact` for the SafeTensors paged sampler. Direct GGUF
plus incremental KV is intentionally rejected because that composition has not
preserved the canonical token trajectory.

Both preference qualifiers load
`../src/finetune/gemma4_preference_environment.policy`, remove every inherited
`TERMITE_*` and `ANTFLY_GEMMA4_*` override, install only its strict Metal
bindings, and record the policy SHA-256 plus sanitizer contract. The product
binary embeds the same policy and rejects unreviewed semantic/debug controls
during recipe planning, before any run manifest or report can be created.

## Gemma4 numerical-oracle export

`gemma4/export_gemma4_lora_zig_oracle.py` drives the typed Gemma4 trainer's private
oracle-capture mode and publishes a common `gemma4_oracle_trace/v1` directory
for native or Metal. It requires a clean source checkout, revision-matching
release binary, locked local model, prepared-v6 row and source dataset, and a
provenance-bound Antfly adapter. The wrapper validates the raw gradients,
checkpoint weights and Adam moments, deterministic logit probes, strict Metal
execution counters, and all immutable manifests before writing `COMPLETE.json`.
Its minimal exact packaging dependencies are in
`gemma4/requirements-gemma4-zig-oracle.txt` and are lock-validated independently of
the CUDA-only HF environment. It never downloads assets or replaces an
existing output. See
`../docs/finetuning/GEMMA4_ORACLE.md` for the exact commands and release matrix.

`gemma4/qualify_gemma4_preference_quality_campaign.py` runs a minimum three-seed DPO
or GRPO quality gate through the public recipe CLI. Long-horizon admission is
based on actual training units rather than repeated passes over a tiny slice:
at least 40 DPO pairs or 512 GRPO prompt groups are required. Each
typed seed now produces a fresh adapter through the public Gemma4 bootstrap,
binds the initialization seed in a v3 Antfly manifest, and selects a
deterministic permutation of the same immutable training-row multiset. GRPO
then applies an independently derived deterministic Fisher-Yates prompt
permutation at every epoch; reports attest the order algorithm and preserve
original dataset indices for sampling, rewards, traces, and frozen-reference
caching. Epoch-boundary resume reconstructs the same epoch order directly from
the typed seed, logical epoch, and dataset size. The
campaign rejects duplicate initial adapter checkpoints, requires every run to
account for the complete logical horizon, change its adapter, publish a distinct
final adapter digest, and pass held-out floors. GRPO additionally requires at
least 25% of logical groups to contain reward variation and reach the optimizer
and permits at most 1% to be safely rejected by the train-time KL budget.
Overrides may only make either bound stricter; zero-variance or KL-rejected
groups cannot make a nominally long but nearly update-free run look qualified.
KL-controller admissions must match the realized optimizer groups rather than
the skipped logical groups. Each run evaluates its fresh initialized adapter
before training. DPO requires held-out accuracy, loss, and margin improvement.
GRPO requires mean-reward improvement while its first/greedy-completion reward
may only tie or improve. Its stochastic positive-reward group rate must remain
within one net evaluation group of baseline and above the absolute floor; the
one-group production cap may only be tightened. Native recipe admission derives
that bound in `f64`, and the outer qualifier reconstructs exact integer counts
from the reported `f32` rates, so non-power-of-two holdouts such as 254 groups
cannot fail or pass because of rate rounding. The qualifier recomputes all three
metrics from the digest-bound baseline/final reward traces. Every seed must have
more prompt-level reward wins than losses. After all seeds pass their individual
gates, the qualifier averages each prompt's group reward delta across training
seeds and applies one common-random-number exact sign test at `p <= 0.05` across
prompts. Completions in a group and repeated evaluation of a prompt under
different training seeds are not treated as independent samples. This keeps the
held-out prompt as the experimental unit without requiring an independent
significance result from every seed. The p-value ceiling cannot be relaxed.
Campaign schema v7 records the per-seed and multi-seed paired counts, exact
positive-group counts, and trace identities alongside both evaluation reports.
Its output root must not already exist. The campaign fails fast at the first
seed that misses an individual floor or directionality requirement, preserves
that run's immutable artifacts and a `status = "fail"` campaign report, and does
not run later seeds. The aggregate significance gate runs only after at least
three seeds complete. A failed predeclared floor remains negative evidence; a
methodological contract revision requires a new schema, recipe identity, output
root, and full rerun rather than reclassifying the old result.

`materialize-gemma4-lora` merges a validated standard LoRA/DoRA adapter into a
monolithic deployment Safetensors file. The writer accepts monolithic or
sharded Safetensors bases, streams untouched tensor bytes exactly, materializes
only one adapted weight at a time, re-encodes it in the source F32/F16/BF16
dtype, validates the staged inventory, metadata, finite merged values, and
untouched payloads, then publishes the output directory without replacement.
GGUF materialization remains unsupported.

`gemma4/materialize_gemma4_grpo_boolq.py` converts pinned Google BoolQ Parquet train
and validation files into balanced, disjoint `text-grpo` JSONL artifacts. It
rejects prompt truncation, binds the full dataset revision and file hashes,
opens Gemma4's `final` response channel, verifies that the `yes`/`no` target is
exactly one token, admits an explicit 1-32-token rollout budget, and records the
exact tokenizer and materializer identities for a replayable real-data GRPO
campaign. For a fresh evaluation slice, repeat `--exclude-eval-manifest` with
prior v1/v2 materialization manifests; v2 excludes their exact source IDs and
binds each manifest digest. Prefer exact exclusions over a per-label offset
when changing the token budget, because admission changes can otherwise make
offset slices overlap. Both MLX parity runners admit historical v1 and current
v2 manifests; v2 selection/exclusion evidence and every manifest's semantic
SHA-256 are revalidated before the referenced JSONL or model is opened.

`gemma4/run_gemma4_grpo_boolq_mlx_parity.py` consumes that manifest plus a completed
Antfly acceptance root and runs an offline, provenance-locked MLX comparison.
It separates exact Antfly trace replay from native MLX rollout, checks
candidate/reward behavior independently from optimizer-level numerical drift,
and requires the batched scorer's policy-first/frozen-reference execution
order when the acceptance advertises that mode.

`gemma4/run_gemma4_grpo_boolq_mlx_multitoken.py` is the stricter multi-token E2B/E4B
campaign runner. It accepts the qualified GRPO v4/v2 reports and the newer
v5/v3 reports carrying incremental-KV telemetry, requires a complete
KL-control trace, replays Antfly's exact completions in one MLX lane, and runs
an independent deterministic MLX rollout in another. Every divergent candidate
is sampled, scored, and differentiated at physical batch size one, then
token-normalized gradients are accumulated, clipped once, and applied as one
group update. A `1e-4` sampling/rescore gate and raw-K3 `train_max_kl` budget
fail before accepting a result. Its output always keeps broad GRPO performance
parity and long-horizon quality parity outside the claim boundary.

`gemma4/validate_gemma4_grpo_incremental_kv_parity.py` compares one completed
incremental-KV campaign with its qualified full-prefix baseline. It requires
exact train/eval reward-trace, KL-control-trace, and final-adapter hashes plus
one canonical prompt prefill per group, nonzero paged reuse, device-resident
ranked selection, exact completion rescoring, and zero host-logit fallbacks.
The output path must not already exist.

The segmented prompt-tail clone lane uses the same validator and exact artifact
gate. `ANTFLY_GEMMA4_GRPO_BATCH_MULTI_TOKEN_BACKWARD=1` is a separate,
default-off research control; optionally cap it with
`ANTFLY_GEMMA4_GRPO_MULTI_TOKEN_BACKWARD_MAX_BATCH`. Reports identify the
physical batch size and physical micro-batches per group. Do not use this lane
for release evidence: bounded E2B/E4B probes reduced backward time but failed
the byte-identical adapter and peak-memory promotion gates. See
`../docs/finetuning/GEMMA4.md` for the retained artifacts and the
segment-aware/ragged-attention follow-up.

## GLiNER2 fine-tuning

Primary entry points:

- `gliner2/run_gliner2_lora_production_readiness.sh` runs the complete release
  gate.
- `gliner2/run_gliner2_lora_perf_gate.sh` runs repeated accelerator/performance
  parity.
- `gliner2/compare_gliner2_lora_python_zig.py` runs one native, Metal, or CUDA parity
  comparison against the pinned Python oracle.
- `gliner2/qualify_gliner2_cuda_hardware.py` records a CUDA hardware lane, and
  `gliner2/summarize_gliner2_cuda_hardware.py` verifies the
  multi-architecture matrix.
- `gliner2/prepare_gliner2_smoke_diagnostic.py` materializes explicitly non-release
  synthetic data from `../testdata/gliner2/`.

Supporting modules are grouped by role in their names:

- `evaluate_*`, `validate_*`, and `finalize_*` enforce release contracts.
- `summarize_*` materializes retained convergence or hardware evidence.
- `gliner2_*` contains shared data and contract helpers.
- `test_*gliner2*` and the adjacent named tests cover these script contracts.

See `../docs/finetuning/GLINER2.md` for the authoritative workflow. The
generated Unicode tables must be checked with
`python3.12 gliner2/generate_gliner2_unicode_tables.py --check`.

## Other model families

Prefer the matching documentation before running hardware gates; many scripts
intentionally require local model artifacts or a specific accelerator. Shared
CUDA artifact generation, paired benchmarking, Metal debugging, and OpenAPI or
descriptor regeneration remain in this directory because multiple model
families consume them.
