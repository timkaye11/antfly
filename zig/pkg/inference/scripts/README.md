# Inference scripts

Most files here are model-specific diagnostics or release tooling, not CI entry
points. The model documentation owns the detailed commands and acceptance
criteria. This index identifies the small set of scripts an operator normally
starts with.

## Gemma4 Metal interrupted/resumed acceptance

`qualify_gemma4_metal_resume.py` runs a real strict-Metal Gemma4 LoRA job
uninterrupted, kills an identical job immediately after a durable epoch-boundary
checkpoint, resumes it in a fresh output directory, and publishes a PASS only
when the final adapter is byte-identical and the post-resume optimizer trajectory
matches with zero native/interpreter fallback. The output root must not already
exist.

```bash
python3 zig/pkg/inference/scripts/qualify_gemma4_metal_resume.py \
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

`qualify_gemma4_preference_resume.py` is the DPO/GRPO counterpart. It runs one
uninterrupted job, sends SIGTERM to a second process only after its
content-addressed epoch sidecar and checkpoint are durable, and resumes that
checkpoint in a third immutable root. A PASS requires byte-identical final
checkpoint and adapter trees, exact training and discrete evaluation traces,
and only the documented narrow terminal Metal GRPO KL tolerances. Use
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

`qualify_gemma4_preference_quality_campaign.py` runs a minimum three-seed,
minimum four-epoch DPO or GRPO quality gate through the public recipe CLI. Each
typed seed now produces a fresh adapter through the public Gemma4 bootstrap,
binds the initialization seed in a v3 Antfly manifest, and selects a
deterministic permutation of the same immutable training-row multiset. The
campaign rejects duplicate initial adapter checkpoints, requires every run to
complete the exact optimizer horizon, change its adapter, publish a distinct
final adapter digest, and pass held-out floors. Each run evaluates its fresh
initialized adapter before training and also requires strict held-out accuracy,
loss, and margin improvement for DPO or reward improvements for GRPO. Baseline
and final evaluation reports are digest-bound into the campaign report. Its
output root must not already exist. The campaign fails fast at the first seed
that misses a floor, preserves that run's immutable artifacts and a
`status = "fail"` campaign report, and does not run later seeds. A failed
predeclared floor is negative evidence; do not weaken it after observing the
result.

`materialize-gemma4-lora` merges a validated standard LoRA/DoRA adapter into a
monolithic deployment Safetensors file. The writer accepts monolithic or
sharded Safetensors bases, streams untouched tensor bytes exactly, materializes
only one adapted weight at a time, re-encodes it in the source F32/F16/BF16
dtype, validates the staged inventory, metadata, finite merged values, and
untouched payloads, then publishes the output directory without replacement.
GGUF materialization remains unsupported.

`materialize_gemma4_grpo_boolq.py` converts pinned Google BoolQ Parquet train
and validation files into balanced, disjoint `text-grpo` JSONL artifacts. It
rejects prompt truncation, binds the full dataset revision and file hashes,
opens Gemma4's `final` response channel, verifies that the `yes`/`no` target is
exactly one token, admits an explicit 1-32-token rollout budget, and records the
exact tokenizer and materializer identities for a replayable real-data GRPO
campaign.

`run_gemma4_grpo_boolq_mlx_parity.py` consumes that manifest plus a completed
Antfly acceptance root and runs an offline, provenance-locked MLX comparison.
It separates exact Antfly trace replay from native MLX rollout, checks
candidate/reward behavior independently from optimizer-level numerical drift,
and requires the batched scorer's policy-first/frozen-reference execution
order when the acceptance advertises that mode.

`run_gemma4_grpo_boolq_mlx_multitoken.py` is the stricter multi-token E2B/E4B
campaign runner. It accepts the qualified GRPO v4/v2 reports and the newer
v5/v3 reports carrying incremental-KV telemetry, requires a complete
KL-control trace, replays Antfly's exact completions in one MLX lane, and runs
an independent deterministic MLX rollout in another. Every divergent candidate
is sampled, scored, and differentiated at physical batch size one, then
token-normalized gradients are accumulated, clipped once, and applied as one
group update. A `1e-4` sampling/rescore gate and raw-K3 `train_max_kl` budget
fail before accepting a result. Its output always keeps broad GRPO performance
parity and long-horizon quality parity outside the claim boundary.

`validate_gemma4_grpo_incremental_kv_parity.py` compares one completed
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

- `run_gliner2_lora_production_readiness.sh` runs the complete release gate.
- `run_gliner2_lora_perf_gate.sh` runs repeated accelerator/performance parity.
- `compare_gliner2_lora_python_zig.py` runs one native, Metal, or CUDA parity
  comparison against the pinned Python oracle.
- `qualify_gliner2_cuda_hardware.py` records a CUDA hardware lane, and
  `summarize_gliner2_cuda_hardware.py` verifies the multi-architecture matrix.
- `prepare_gliner2_smoke_diagnostic.py` materializes explicitly non-release
  synthetic data from `../testdata/gliner2/`.

Supporting modules are grouped by role in their names:

- `evaluate_*`, `validate_*`, and `finalize_*` enforce release contracts.
- `summarize_*` materializes retained convergence or hardware evidence.
- `gliner2_*` contains shared data and contract helpers.
- `test_*gliner2*` and the adjacent named tests cover these script contracts.

See `../docs/finetuning/GLINER2.md` for the authoritative workflow. The
generated Unicode tables must be checked with
`python3.12 generate_gliner2_unicode_tables.py --check`.

## Other model families

Gemma4, Florence2, ClipClap, reranker, and quant-kernel scripts use the model or
backend name in the filename. Prefer the matching documentation before running
hardware gates; many scripts intentionally require local model artifacts or a
specific accelerator.
