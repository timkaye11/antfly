# Gemma4 E2B/E4B LoRA Fine-Tuning

> Status: production-candidate for the explicitly qualified single-device,
> text-only BF16 LoRA-SFT and DPO lanes. E2B GRPO passes a bounded
> three-seed/eight-epoch absolute-floor gate; E4B GRPO misses its predeclared
> top-rank floor and remains quality-blocked. Exact incremental KV reuse is
> experimental and default-off because the first serial implementation
> regresses throughput. Canonical direct-GGUF E2B SFT/DPO/GRPO remains an
> experimental research surface, and public QLoRA remains fail-closed. This is
> not a blanket production-ready claim: independent-initialization and
> baseline-relative quality, required CI, deployment materialization, and GGUF
> task-parity gates remain.

The current Zig path implements the text causal-LM graph, sparse loss, LoRA
optimizer, evaluation, and artifact pipeline intended for Gemma4 E2B and E4B
models. It prepares Gemma4 chat data, bootstraps a strict LoRA adapter
inventory, requires a distinct evaluation artifact, and selects the native or
Metal backend. Device-only BF16 frozen linears and embedding gathers, rank-4
attention VJPs, strict optimizer steps, and bounded real E2B/E4B Metal smokes
have passed the focused checks described below. The final-source DPO
ReleaseFast binary completed matched real UltraFeedback E2B and E4B 25-update
profiles with fixed and pair-safe length schedules. Final-binary E2B and E4B
BF16 Metal jobs also passed real process-kill/resume gates with byte-identical
adapters and exact post-boundary training/discrete trajectories. Terminal GRPO reward and
completion behavior is exact; two independent Metal KL floats use narrow,
fail-closed absolute tolerances after identical-checkpoint replay proved small
fresh-process GPU evaluation variation. The standalone report's derived total
loss tracks the weighted-KL tolerance while its policy-gradient loss remains
exact. Three-seed/eight-epoch absolute-floor campaigns pass for E2B/E4B DPO and
E2B GRPO; E4B GRPO completes healthy optimizer work but fails held-out top-rank
quality. These results establish bounded data-order robustness, not independent
initialization, baseline-relative improvement, multi-task convergence,
repeated distribution-level performance, or required-CI acceptance. This
remains an implementation and qualification contract rather than a blanket
readiness claim.

Distributed training, multimodal/projector training, public GGUF QLoRA recipes,
and Gemma4 MoE models are outside the supported path described here. Q4_0,
Q4_K, and Q6_K packed frozen-linear input-gradient kernels now exist. An
explicitly gated canonical direct-GGUF E2B command lane has passed optimizer
and interrupted/resumed Metal gates for SFT, DPO, and GRPO, but remains
experimental pending memory, cross-framework parity, and task-quality
campaigns. Direct GGUF plus incremental-KV GRPO is rejected after exact token
divergence. MLX-LM is a same-Mac performance reference, not an Antfly training
backend.

## Implemented Scope

- `gemma_chat/v1` supervised fine-tuning data, including system, multi-turn,
  tool-call, and tool-result messages.
- The same Gemma4 turn/channel wire format used by inference. Loss spans cover
  assistant final-channel payloads and the turn terminator, not the role or
  thought-channel prompt.
- Dense E2B/E4B graph contracts including PLE, sliding/full attention,
  per-layer GQA, shared KV layers, tied embeddings, Gemma4 scaling, and
  softcapping.
- Real LoRA optimizer steps and before/after evaluation on an explicitly chosen
  `native` or `metal` backend. Autodiff requires a separately prepared
  evaluation artifact and rejects exact prepared-example overlap.
- Adapter bootstrap from monolithic or sharded Hugging Face Safetensors and
  GGUF tensor metadata, with exact target paths persisted in the adapter
  contract. Target discovery is graph-aware: checkpoint-present K/V weights
  from shared-KV tail layers, omitted-V layers, and out-of-range layers are
  excluded, and an explicit request for one of those inert tensors fails
  closed. Sharded bootstrap validates the complete index and shard set;
  production-scale merged-model materialization is not implemented.
- Memory-bounded sparse causal targets. The graph never owns a dense
  `[sequence, vocabulary]` target. Strict-Metal hard-label training uses the
  frozen tied-head fused linear cross-entropy primitive, which returns a
  device-reduced scalar and its hidden-state gradient without materializing a
  global logits tensor. General signed per-token targets retain a bounded
  sparse projection fallback.
- Production preparation emits `gemma4_prepared/v6`. It binds tokenized
  examples to the selected base artifact, tokenizer assets, chat-template,
  source dataset/split/revision, canonical source row, rendered chat, group,
  and media-content identities. Adapter bootstrap records the model identities
  and a closed, exact A/B target inventory.
- `adapter_config.json` stays inside the public PEFT LoRA schema. Antfly-only
  provenance, exact target policy, recursive metadata, and the internal tensor
  key format live in the strict `antfly_finetune_manifest.json` sidecar. The
  sidecar also binds the adapter checkpoint byte size and SHA-256 digest.
- `adapter export gemma4-peft` validates a standard preset adapter against the
  exact base provenance, preserves every F32 tensor payload byte, translates
  only Antfly's internal tensor names into stock PEFT names, and atomically
  publishes `adapter_model.safetensors`, `adapter_config.json`, and a
  hash-bound `antfly_peft_export.json` sidecar. A pinned local structural
  smoke loads that export through PEFT, saves and reloads it, and requires
  exact adapter tensors and logits; real E2B/E4B interoperability is still a
  separate gate.
- The public `antfly inference finetune` path exposes typed Gemma4 prepare,
  bootstrap, train, standalone eval, adapter-validation, and PEFT-export
  operations. Named flags are canonical; positional forms are a one-release
  compatibility bridge on older operations.

## Text Training Flow

An installed Antfly binary is the product interface. From
`zig/pkg/inference`, `zig build finetune -- ...` forwards the same arguments to
the same dispatcher. A dataset row can be as small as:

```json
{"schema":"gemma_chat/v1","messages":[{"role":"user","content":"Reply briefly."},{"role":"assistant","content":"Okay."}]}
```

Use the explicit four-step flow so training data, held-out data, and adapter
inventory are auditable. `TRAIN_DATA` and `EVAL_DATA` must be genuinely
disjoint datasets or splits; evaluation is not a prefix replay of training.

```sh
BASE=/path/to/gemma-4-E2B-it-bf16
TRAIN_DATA=/path/to/train.jsonl
EVAL_DATA=/path/to/eval.jsonl
RUN=/path/to/gemma4-lora-run
STATE=/path/to/gemma4-lora-state

mkdir -p "$RUN" "$STATE"

antfly inference finetune dataset prepare gemma4-lora \
  --model "$BASE" --dataset "$TRAIN_DATA" --split train \
  --out "$RUN/prepared_train.json" --dataset-revision TRAIN_REVISION \
  --max-examples 1000 --max-seq-len 512

antfly inference finetune dataset prepare gemma4-lora \
  --model "$BASE" --dataset "$EVAL_DATA" --split eval \
  --out "$RUN/prepared_eval.json" --dataset-revision EVAL_REVISION \
  --max-examples 128 --max-seq-len 512

antfly inference finetune adapter bootstrap gemma4 \
  --model "$BASE" --out "$RUN/adapter_seed" \
  --rank 16 --alpha 32 --target-preset peft-qv

antfly inference finetune train gemma4-lora \
  --model "$BASE" --adapter "$RUN/adapter_seed" \
  --train-prepared "$RUN/prepared_train.json" \
  --eval-prepared "$RUN/prepared_eval.json" \
  --out "$RUN/train_native" --backend native \
  --lr 0.0003 --max-examples 1000 --eval-max-examples 128 \
  --epochs 1 --max-grad-norm 1.0 --grad-accum 1

antfly inference finetune eval gemma4-lora \
  --model "$BASE" --adapter "$RUN/train_native" \
  --prepared "$RUN/prepared_eval.json" \
  --out "$RUN/eval.json" --backend native --max-examples 128

antfly inference finetune adapter validate gemma4 \
  --model "$BASE" --adapter "$RUN/train_native"

antfly inference finetune adapter export gemma4-peft \
  --model "$BASE" --adapter "$RUN/train_native" \
  --out "$RUN/train_native_peft"
```

The backend flag and `--eval-prepared` are required; there is no implicit
backend fallback or training-data eval fallback.
`--trainer auto` is only a compatibility alias for `autodiff` and never falls
back to surrogate training. The final training directory, adapter bootstrap
directory, standalone eval report, and prepared JSON paths must not already
exist. Prepared and eval JSON use a synced sibling temporary followed by an
atomic no-replace rename; directory artifacts use a sibling staging directory
and a no-replace publish.

The admitted Metal training path uses rank-2 BF16 Safetensors weights. Autodiff
cancels the linear VJP's redundant double transpose, and Metal computes
`dX = dY @ W` directly from the persistent BF16 forward-weight slot with f32
accumulation. It neither creates a transposed weight nor materializes a
full-model f32 copy. Dedicated packed Q4_0, Q4_K, and Q6_K kernels implement
the same frozen-linear input gradient and have crossed runtime and graph-
executor tests, but are not yet a public GGUF training contract. Direct-GGUF
E2B training uses that substrate only behind explicit experimental admission
and selects a graph-visible decomposed vocabulary loss; the builder-level
fused wrapper over the quantized tied head failed exact repeatability and is
therefore not used.
Native BF16 embedding tables also use a device-index gather, and autodiff
propagates q/k/v gradients through Gemma4's rank-4 two-batch-axis attention
contractions.
Rank-2 F16 Safetensors remain outside the documented production contract;
packed GGUF bases remain rejected unless the explicit experimental admission
is present. Gemma4 Metal train and eval select the strict training executor
for the lifetime of the operation, so the public CLI has no hidden executor
environment prerequisite. Explicit executor-disable and parity diagnostics,
native partitions, unsupported operations,
interpreter/runtime fallbacks, host-materialized graph outputs, undeclared or
graph-execution uploads, gather/reduce/cache promotions, explicit runtime-input
transfers, and non-device gradients fail the step before gradient accumulation
or optimizer mutation. Compiled Metal evaluation uses a loss-only graph with
resident weights and no gradient or Adam-state allocation.

For epoch-boundary recovery, add a checkpoint outside every immutable input
and outside the final output directory:

```sh
antfly inference finetune train gemma4-lora \
  --model "$BASE" --adapter "$RUN/adapter_seed" \
  --train-prepared "$RUN/prepared_train.json" \
  --eval-prepared "$RUN/prepared_eval.json" \
  --out "$RUN/train_metal" --backend metal \
  --epochs 4 --seed 42 \
  --checkpoint-path "$STATE/gemma4-trainer.safetensors" \
  --checkpoint-every-epochs 1

# After interruption, repeat the exact bound run in a new --out directory.
# Keep the same checkpoint path and add --resume.
```

`--resume` requires `--checkpoint-path`. The command fingerprints the complete
run contract and rejects a checkpoint from different model, adapter,
train/eval data, optimizer, seed, or schedule settings. Checkpoints are one
mutable, atomically replaced recovery file; recipe `keep_last` generations are
not implemented. Resume is text-only and currently starts at an epoch
boundary. A checkpoint at `epoch == --epochs` is accepted so a crash after the
last state save can republish the final immutable bundle without retraining.

The 2026-08-19 final-binary qualification killed each training process after a
durable epoch-1 checkpoint and resumed epoch 2 in a fresh immutable directory.
E2B published byte-identical adapter
`sha256:ab17f813...618484`; E4B published
`sha256:e338eb86...7341d`. Their post-boundary loss and gradient histories are
exact, with no native/interpreter fallback. Reports are retained at
`/private/tmp/antfly-gemma4-e2b-resume-acceptance-20260819-v3` and
`/private/tmp/antfly-gemma4-e4b-resume-acceptance-20260819-v2`. Use
`scripts/qualify_gemma4_metal_resume.py` to reproduce the gate; a direct GGUF
requires the explicit `--experimental-gguf-qlora` admission flag.

`--activation-checkpoint-interval N` enables graph activation recomputation at
layer boundaries. It is a memory-control mechanism and is unrelated to durable
training checkpoint/resume.

## Data, Provenance, and Sequence Admission

The production preparation command emits `gemma4_prepared/v6`. The summary
records:

- a digest of the selected model artifact set, including a Safetensors index
  and every referenced shard when present;
- tokenizer-asset and chat-template identity digests;
- the resolved source dataset path, content digest, split, and immutable
  revision (the resolved split digest is the default revision);
- a schema-aware digest of all prepared examples; and
- recomputed maxima and aggregate counters used for sequence and modality
  admission.

Each v6 example records a stable source id and group id, a canonical source-row
digest, a rendered-chat digest, and a content digest for every referenced image
or audio payload in addition to token ids and labels. Training recomputes model,
adapter, prepared-example, vocabulary, sequence, supervision, and aggregate
contracts before backend work. Train/eval admission rejects exact token/media
identity overlap, canonical source-record overlap, and group overlap.

The loader retains `gemma4_prepared/v4` and `/v5` compatibility so existing
artifacts can be inspected and migrated. V6 additionally guarantees causal
generation tokenization: the rendered transcript owns exactly one literal BOS,
no implicit EOS is appended, and all assistant turns are recovered for
supervision even without tokenizer offsets. V4/v5 artifacts are not accepted
for training or release evidence. New production campaigns must use v6 and pin
a source revision rather than relying on a mutable pathname.

The final training directory has a completion manifest that enumerates, sizes,
and SHA-256 hashes every regular root payload; nested directories and symlinks
are rejected. Input identities and the exact run contract are bound separately
by the run fingerprint. Teacher-target materialization still does not bind the
teacher model digest to every generated distribution, so that optional lane is
not release-admissible.

Sequence length is admitted before graph/backend construction and is currently
bounded to `1..min(model_max_position_embeddings, 2048)`. The 2048 limit is a
temporary safety ceiling, not a claim that every admitted E2B/E4B sequence fits
the available device memory. Prepared JSON is still a single whole-buffer
artifact with a 128 MiB load ceiling; streaming/sharded prepared data is an open
production requirement.

The recipe and pilot/recursive convenience workflows now plan separate train
and eval preparation and pass the eval artifact into training. The explicit
four-step path remains the clearest acceptance interface because every input
path and revision is visible at invocation time.

## Adapter Artifact Contract

Every newly bootstrapped or trained adapter directory contains:

- `adapter_model.safetensors`: the LoRA A/B payload;
- `adapter_config.json`: only standard PEFT constructor fields such as
  `peft_type`, `task_type`, `r`, `lora_alpha`, and `target_modules`; and
- `antfly_finetune_manifest.json`: Antfly's strict model/tokenizer/template
  digests, exact target preset and inventory, initializer, recursive metadata,
  tensor-key-format declaration, and the adapter checkpoint byte size and
  SHA-256 digest.

Inspection rejects sidecar/config disagreement and the closed-inventory check
requires exactly one A/B pair per configured target, exact model-resolved
preset coverage, F32 finite payloads, base-compatible shapes, and a matching
checkpoint hash and size. Unsupported PEFT math such as DoRA, RSLoRA, nonzero
dropout, bias tuning, fan-in/fan-out layout, or modules-to-save fails before
backend construction; missing or inference-only PEFT configs are not accepted
as trainable adapters. The native training artifact stores weight-qualified
tensor keys (`*.weight.lora_A/B.weight`), while stock PEFT uses a different
wrapper/key layout. Stock PEFT must not load the native artifact directly. The
named export command produces the stock layout in a separate immutable
directory, retains the source and destination checkpoint digests, and binds
the copied PEFT config to the base/tokenizer/template provenance. A
dependency-pinned tiny-model `PeftModel.from_pretrained` save/reload smoke is
the structural gate. Pinned real E2B/E4B PEFT load, fixed-logit/generation
comparison, and reverse import remain release gates rather than implied
capabilities.

## Target Presets

Pass the preset explicitly even though `text-all-linear` is the bootstrap
default. This keeps experiments and acceptance artifacts self-describing.

| Preset | Exact selection | Intended use |
| --- | --- | --- |
| `peft-qv` | Available `q_proj` and `v_proj` tensors | Small PEFT-compatible baseline with the lowest adapter and optimizer footprint |
| `text-all-linear` | Available Q/K/V/O, gate/up/down, and Gemma4 PLE input-gate/projection tensors | Higher-capacity text tuning after the Q/V baseline is correct and memory-safe |

Target discovery understands both Hugging Face Safetensors names and GGUF
names. E2B and E4B do not expose identical inventories because their layer and
shared-tail layouts differ; bootstrap records the resolved tensor paths rather
than relying on a fixed count. Missing or conflicting selections fail instead
of silently training a partial adapter. Artifact selection is deterministic:
monolithic Safetensors, then a Safetensors index, then GGUF.

## Fail-Closed Contract

The supported training lane deliberately rejects ambiguous or degraded runs:

- Autodiff requires `--backend native|metal`; omission returns
  `MissingBackend` before model artifacts are opened.
- GGUF autodiff is rejected before prepared inputs or output artifacts are
  opened unless `ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA=1` is set. Rank-2 BF16
  Safetensors are admitted through the dedicated device-only input-gradient
  path. The packed Q4_0, Q4_K, and Q6_K input-gradient kernels are component-
  and executor-tested; the experimental direct-GGUF lane has no host-dequant
  fallback and automatically disables the nondeterministic fused quantized-
  head loss wrapper.
- Metal requires the training graph executor explicitly enabled. Every eval
  and train step records executor partitions/dispatches and rejects native or
  unsupported partitions, diagnostic direct execution, interpreter/runtime
  fallback, true host outputs, undeclared uploads, graph-execution uploads,
  explicit runtime-input transfers, gather/reduce/resident-cache promotions,
  or non-resident gradients before mutation.
- `auto` resolves to real autodiff. The legacy `--trainer surrogate` spelling
  is accepted only far enough to return a typed unsupported error; public
  commands and production recipes cannot run surrogate training.
- Adapter target patterns are strict. Missing, empty, unknown, or conflicting
  target selections are errors.
- Unsupported graph/model configurations and malformed Gemma4 tool-call wire
  data return typed errors rather than being rewritten approximately.
- The production-intent lane is text-only. The lower-level command still
  contains experimental multimodal internals, but public train/eval parsing and
  typed admission reject projector flags and prepared media before backend or
  output creation.
- Layer-scoped autodiff, layer-wise LR decay, and schedule-free autodiff are
  rejected until their semantics are implemented and tested.
- DoRA training and PiSSA/LoftQ initialization are rejected until the graph and
  adjusted-base artifact semantics are implemented. Generic save/materialize
  also rejects recursive adapters that cannot be represented as one merged
  base.
- Gemma4 recipes admit only `lora-sft`. Full `sft`, `qlora-sft`, and declared
  optimizer/eval/runtime/algorithm/artifact options that the command cannot
  honor fail with typed errors rather than being silently ignored. Recipes map
  `checkpoint.every_epochs` and `checkpoint.resume_path` to the typed command;
  `keep_last` remains rejected because the v1 contract owns one mutable state
  file.
  Unknown JSON fields are errors, and bootstrap/trained output directories must
  be normalized, disjoint leaves. `model.name` is retained as report metadata;
  it does not select the checkpoint.
- DPO and GRPO recipes require both `execution.mode` and `dataset.format`.
  `train` admits only model-backed text preference formats plus explicit
  adapter intent; `score` admits only precomputed logprob fixture formats and
  rejects adapter, optimizer, checkpoint, runtime, and trainer fields that it
  would otherwise ignore. Neither path defaults to the other, so a UI or API
  caller cannot receive a successful scoring report while believing it trained
  an adapter.
- Gemma4 preference training applies the same-base reference, supported
  backend/trainer, adapter, optimizer, checkpoint, and disjoint output-path
  checks during planning. A separate held-out JSONL and task-specific DPO or
  GRPO thresholds are mandatory. Token-identical train/eval prompts fail
  closed; the post-update evaluator writes its report before publication and a
  failed threshold leaves the trained adapter unpublished. Task report v3
  binds execution mode, format, and the passing evaluation summary, while the
  normalized run report fingerprints both datasets, evaluation evidence, and
  bootstrap/trained adapter trees. Publication also requires a nonempty,
  finite LoRA parameter set whose digest changed from initialization.
- Gemma4 text GRPO accepts typed weighted built-in rewards plus pinned generic
  verifier and `model-command` executables. External calls use no shell or
  inherited environment, have bounded timeouts and output ranges, recheck the
  executable and model-input SHA-256 identities for every call, require
  model/tokenizer/template/calibration response attestation, and retain
  versioned request/response/evidence or structured failure traces for both
  training and held-out evaluation. The lower-level multimodal GRPO code
  remains research-only until it owns the same evaluator and reward-evidence
  contract.

An accepted training epoch must report finite loss, nonzero supervised tokens,
and `optimizer_steps > 0`. Before/after evaluation intentionally performs no
updates, so evaluation records may correctly report zero optimizer steps; that
is not evidence that training succeeded.

## Artifact Publication and Materialization

The current source stages and no-replace-publishes bootstrap adapters, generic
adapter saves, the bounded legacy merge path, recursive compressed-base output,
and the primary autodiff adapter-plus-report directory. Prepared and standalone
eval JSON are written to an exclusive sibling temporary, file-synced, renamed
without replacement, and followed by a parent-directory sync. Adapter
publication validates a closed inventory: each configured exact target has one
A/B pair, unconfigured tensors are rejected, and DoRA metadata and magnitude
tensors must agree.

Directory publication recursively syncs staged regular files and directories,
then uses a no-replace rename and parent-directory sync. Mutable recovery
checkpoints intentionally use atomic replacement of one named state file and
are a different contract from immutable final artifacts. The legacy
`materialize-gemma4-lora --eval` spelling is rejected until a typed evaluator
can target the staged artifact and complete before publication. Power-loss
failure injection and stale-staging recovery remain open.

Full-model materialization is deliberately fail-closed while it uses the
legacy whole-model F32 implementation. It accepts only a monolithic
Safetensors checkpoint of at most 64 MiB and rejects GGUF, sharded
Safetensors, and larger files. This prevents an E2B/E4B OOM but also means
production-scale E2B/E4B materialization is unavailable until a streaming,
dtype-preserving, sharded writer is implemented.

## Checkpoint and Resume Status

`RealAutodiffTrainer` now has low-level save, inspect, and restore APIs for a
fully resumable state. The checkpoint includes trainable values, Adam moments
and per-slot step counts, an incomplete gradient-accumulation window, trainer
seed and counters, conditional optimizer-family presence, run/metrics-prefix
fingerprints, and caller-owned epoch/example/order/PRNG progress. Restore
validates names, shapes, counters, accumulation configuration, seed, optional
fingerprints, and optimizer-family state before mutation, then restores Metal
optimizer slots into their existing device allocations.

The public Gemma4 train command exposes `--seed`, `--checkpoint-path`,
`--checkpoint-every-epochs`, and `--resume`; the recipe maps epoch cadence and
resume path to the same typed operation. The text loop restores its epoch,
example cursor, order, and PRNG state, admits a completed final checkpoint for
publication recovery, and rejects partial/mismatched state. The current public
contract is epoch-boundary recovery with one mutable checkpoint. Final-binary
E2B and E4B BF16 Metal qualifications now pass real process-kill/resume with
byte-identical final adapters and exact post-boundary loss/gradient
trajectories. The explicitly admitted official E2B Q4_0 GGUF lane passes the
same gate with its decomposed loss graph. Retained checkpoint generations and
mid-epoch CLI scheduling remain open; activation checkpointing is
recomputation and does not close those gates.

## Reference Oracles and Performance Targets

The reference strategy deliberately assigns one job to each implementation:

| Reference | Role | Required evidence |
| --- | --- | --- |
| Hugging Face Transformers + PEFT | Primary correctness and artifact-semantics oracle | Exact Antfly `input_ids`/`labels`; pinned local model and package revisions; eager causal loss; loss, logit probes, per-target gradients, Adam state, updates, and normalized adapter inventory |
| MLX-LM | Same-Mac performance and memory reference only | Identical pinned model/case/protocol/hardware; fresh processes; explicit device synchronization; alternating framework order; at least five repetitions and the locked sequence/accumulation matrix |
| Unsloth | Separate NVIDIA recipe, convergence, and quality cross-check | Pinned CUDA hardware/software and datasets; never mixed into the Apple-Metal throughput gate |

Unsloth is not the Metal equivalent of the GLiNER2 Fastino oracle: it is
CUDA-oriented and cannot provide a defensible same-Mac performance comparison.
HF/PEFT owns numerical correctness because it exposes the standard adapter and
optimizer semantics; MLX-LM owns the local Apple-Silicon performance target.
No oracle result is valid unless model files, revisions, environment versions,
prepared source, target set, rank/alpha, optimizer hyperparameters, and protocol
match the checked-in lock. The harness runs offline and fails on drift.

See [GEMMA4_ORACLE.md](GEMMA4_ORACLE.md) for commands, schemas, tolerance
profiles, and the release matrix. The checked-in harness is scaffolding until
pinned real E2B and E4B traces and benchmark campaigns have been archived.

## BF16 Correctness and Q4 Deployment Lanes

Keep precision concerns separated until quantized training has its own parity
and quality evidence.

| Lane | Base artifact | Purpose | Current release posture |
| --- | --- | --- | --- |
| BF16 correctness | Unquantized BF16 Safetensors | Establish native forward, loss, gradient, optimizer-update, and adapter-save correctness, then compare the stored-weight Metal kernel | Final-binary E2B and E4B Metal jobs pass process-kill/resume with byte-identical adapters. Three-seed/eight-epoch E2B/E4B DPO and E2B GRPO pass bounded absolute quality floors; E4B GRPO fails the predeclared top-rank floor, so baseline-relative quality, independent initialization, distribution-level performance, and required CI remain open |
| Q4 deployment / QLoRA substrate | QAT Q4 GGUF used by serving | Prove target-name compatibility, adapter loading/application, memory bounds, and post-training generation quality on the deployed base | Packed Q4_0/Q4_K/Q6_K `dX` kernels pass without host dequantization, and explicitly gated canonical E2B Q4_0 SFT/DPO/GRPO lanes pass optimizer and process-kill/resume gates. Public `qlora-sft`, E4B GGUF, memory, parity, and task-quality gates remain fail-closed; direct-GGUF plus incremental-KV GRPO is separately rejected |

The loaders may accept a Q4 GGUF and bootstrap adapter targets from its tensor
headers. Kernel availability and target discovery alone are not proof that
QLoRA training is numerically correct or production-ready. Promote direct
quantized-base training only after a real model completes strict no-host
optimizer, memory, resume, artifact, parity, and task-quality gates against the
BF16 reference.

## Validation Status

### Current source verification

The current branch adds prepared-input v6 causal-tokenization plus
source/group/media provenance, mandatory separate eval,
vocabulary/sequence/aggregate admission, PEFT-schema config plus Antfly
sidecar, typed public Gemma4 operations, four-step recipes and workflows,
immutable file publication, loss-only compiled Metal evaluation, strict
promotion/upload telemetry, exact-resume substrate, and packed Q4_0/Q4_K/Q6_K
frozen-linear input gradients.

The 2026-08-19 integrated source snapshot passed these local gates:

- final required-device ReleaseFast `test-gemma4-finetune`: 263 selected, 261
  passed, two optional real-model tests skipped, zero failed. This includes the
  strict tiny BF16 CLI optimizer step, packed-format tile-boundary numerical
  checks, Q4_0 graph-executor dispatch, direct-GGUF loss repeatability, and the
  real subprocess `model-command` success/failure/attestation contract;
- the earlier non-Metal ReleaseFast `test-gemma4-finetune`: 180 passed, eight
  Metal-only tests skipped, zero failed, confirming that the training API and
  unsupported-backend stubs remained portable at that checkpoint;
- current-source ReleaseFast inference-edition `antfly`: all ten build steps
  passed and `antfly inference finetune --help` reached the unified dispatcher;
- 125 offline Gemma4 oracle, publication, PEFT-key, MLX-build-attestation,
  benchmark-runner, and campaign-contract tests, all passing; and
- the checked-in oracle lock validation at
  `sha256:c848acb5fa38abda012f52c31cc122927b26775896d4a460e58cc9336cf27383`.

One bounded real `google/gemma-4-E2B-it` BF16 run completed through the public
Metal CLI with prepared-v6 disjoint inputs, rank-4 Q/V LoRA, one optimizer
step, one Metal optimizer update, finite gradient norm `2.42837`, and zero
recorded graph/interpreter fallback. Held-out loss moved from `6.82213` before
the step to `6.20983` after it. The immutable output and run manifest are under
`/private/tmp/antfly-gemma4-e2b-real-smoke-20260811-1555/` on the qualification
host. The trained adapter also exported through the public stock-key PEFT
boundary and completed a full E2B stock-PEFT load/forward smoke.

The SDK 26.2 MLX reference lane is now executable. On 2026-08-12, a separate
diagnostic-only pinned MLX 0.31.2 / MLX-LM 0.31.3 E2B run completed the exact
sequence-128, accumulation-1, rank-16/alpha-32 `peft-qv` workload. Twenty
synchronized measured optimizer steps had median/mean latency
`0.263281/0.263396 s` (about `485.96` input tokens/s); allocator and process
physical-footprint peaks were `10.014` and `10.708 GiB`, with zero swap. The
measured window did incur 1,097,728 bytes of page-ins and 589,824 bytes of
page-outs, so it would fail the release lane's zero-paging threshold. The run
loaded the SDK-required attested `libjaccl.dylib` and
closed all BF16-base/F32-LoRA/F32-gradient/F32-AdamW precision inventories.
This artifact is explicitly not release evidence: the Antfly checkout was
dirty, release memory thresholds were not enforced, and there is not yet an
alternating five-repeat campaign. It establishes that the MLX reference runs
end to end.

The matching Antfly diagnostic lane now also completes end to end. Its current
default-on Metal route uses simdgroup matrix multiply for both BF16
frozen-linear forward and input-gradient products when rows and both matrix
dimensions are at least 128. Exact 64-row-compatible shapes use a 64-row by
64-column by 32-K tile with eight simdgroups and 256 threads, doubling weight
reuse relative to the preceding 32-row tile. Devices that cannot admit the
256-thread pipeline, irregular matrix dimensions, and non-64-multiple row
counts retain the 32-row path. Both paths convert BF16 weights and F32
activations or output gradients into transient FP16 threadgroup tiles, perform
half 8x8 simdgroup MMA, and accumulate into F32. The routes have independent
emergency rollback controls:

- `TERMITE_METAL_DISABLE_BF16_SIMDGROUP_MM=1` disables the forward route; and
- `TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_MM=1` disables the
  input-gradient route.

`TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64=1` and
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64=1` disable only the 64-row
forward and backward specializations, respectively, and fall back to the
proven 32-row simdgroup route.

The older 16x32 tiled routes remain available behind their existing rollback
controls. Disabling both simdgroup routes with the final binary restores the
previous promoted loss `7.611953259`, gradient norm `3.143428326`, and
approximately `1.370 s` synchronized frame.

On the identical sequence-128, accumulation-1, rank-16/alpha-32 `peft-qv`
workload, the final guarded 64-row binary's strict 20-step diagnostic measured
`0.568763/0.568748 s` median/mean (`225.06` input tokens/s). This is `1.235x`
faster than the preceding 32-row simdgroup binary (`0.702349 s`), `2.495x`
faster than the 16x32-tiled binary (`1.418835 s`), and `3.815x` faster than the
original `2.169801 s` baseline. Process peak physical footprint stayed
effectively flat at `1,774,864,856` bytes. The measured window recorded 16,384
bytes of page-ins, zero page-outs, and zero swap; the nonzero page-in count still
fails the release lane's zero-paging gate. The exact executable SHA-256 is
`18ceaa0a0025b3e728118584488c7772d1bc4f8a658cc9fda923da4e688c2c6f`.
The diagnostic artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-simdgroup-m64-final-diagnostic-v1.json`
with SHA-256
`06cf51a0128119d263326c0313fbc13be866189befcc97d5fc58efe662c7683f`.

This route deliberately trades exact BF16/F32 arithmetic identity for the
simdgroup's FP16 tile inputs. Against the preceding tiled route, one-step loss
moved from `7.611953259` to `7.611551762` and gradient norm from `3.143428326`
to `3.143764734`. Across all 100 adapter tensors, the resulting one-step adapter
had maximum absolute delta `0.000399718`, relative L2 delta `0.001254`, and
cosine `0.999999214`; the update vector itself had relative L2 delta `4.09%`
and cosine `0.999165`. This is small model-state drift, but it is not exact
numerical parity. The 64-row scheduling change itself adds no further drift:
at both sequence lengths 128 and 512, its one-step adapter checkpoint was byte
identical to the 32-row rollback. A five-update sequence-128 run also produced
an identical final adapter (`SHA-256
95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`),
identical loss/gradient history, held-out loss `4.839618`, one Metal optimizer
update, 50 runtime LoRA regions, and zero fallback per step.

MLX-LM remains `2.159x` faster on the sequence-128 cell, down from `2.667x`
before the 64-row pass and `5.387x` before the first simdgroup pass, while its
process peak is `6.479x` larger. On the exact 512-token prepared workload, the
final Antfly binary measured `1.566196/1.565846 s` median/mean (`326.98` input
tokens/s), a `1.306x` improvement over the 32-row mean of `2.045645 s`, with a
`5,987,846,664`-byte process peak. Matched MLX-LM measured `0.990706 s`
(`516.80` input tokens/s) and a `15,999,361,256`-byte peak. MLX is therefore
`1.581x` faster at 512 tokens but uses `2.672x` the process footprint. The final
Antfly sample recorded 3,784,704 bytes of page-ins, 114,688 bytes of page-outs,
and zero swap; it still fails the zero-paging gate. The Antfly and MLX artifacts
are respectively
`/private/tmp/antfly-gemma4-zig-e2b-seq512-simdgroup-m64-final-diagnostic-v1.json`
(SHA-256
`a23a5eefe5ecc48bdd53a63dd732d51730920f21d66be9132bc3640e398359ff`)
and `/private/tmp/antfly-gemma4-mlx-e2b-seq512-diagnostic-v1.json` (SHA-256
`d7d6a7518193963e7bfaf83ffaf289dd8af176f49791e16a1231dd80aa75513c`).

The sparse causal-loss graph now batches up to eight supervised rows through
each tied vocabulary projection. The former one-row graph reread Gemma4 E2B's
approximately 805 MiB BF16 embedding table once per target token. Eight rows
match the existing Metal backward-input tile height, reduce those frozen-weight
scans from eight to one on the locked workload, and add only an 8 MiB F32
logits tensor. `TERMITE_GEMMA4_DISABLE_BATCHED_SPARSE_LOSS=1` restores the
one-row graph. `TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=<1..64>` supplies a
bounded diagnostic override; invalid values retain the default of eight.

In synchronized no-frame profiling, batching reduced the eight
`1x262144 * 262144x1536` vocabulary input-gradient products from `102.590 ms`
to one `8x262144 * 262144x1536` product at `22.836 ms`. The matching forward
projection fell from `29.239 ms` across eight products to `20.182 ms` in one.
Total training `dot_general` time fell from `596.780 ms` to `515.718 ms`, and
the compiled command count fell from `3,852` to `3,449`, with no fallback.

The final same-binary sequence-128 diagnostic measured
`0.470076/0.470017 s` median/mean (`272.33` input tokens/s), a further `1.210x`
improvement over the 64-row result and `4.616x` over the original baseline.
Peak physical footprint was `1,808,025,952` bytes, 33,161,096 bytes above the
one-row M64 sample. MLX-LM is now `1.784x` faster while using `6.359x` the
process footprint. The sample recorded 3,047,424 bytes of page-ins, no
page-outs, and no swap. Its artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`63d16c86dd9aaeb89699e1b38632a4a6d97edab0ae47e9e4b4c09037e3623052`.

At sequence 512, the same binary measured `1.466413/1.466469 s`
median/mean (`349.14` input tokens/s), a further `1.068x` improvement over M64.
Peak physical footprint was `6,023,989,768` bytes. MLX-LM is `1.480x` faster
while using `2.656x` the process footprint. This sample recorded 8,404,992
bytes of page-ins, 16,384 bytes of page-outs, and no swap. Its artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`1035fd111fc79e5d2103d02ba074f11531cacf5d03f1f65127e91eeee1c243e2`.
The diagnostic executable SHA-256 is
`c265e78692dd29da4bfa198e0e0e00e50412d769d7519dd2abf2ceca2cc28d9a`.

The graph change adds no observed optimizer-state drift. For the locked
eight-target example, one- and five-update adapters are byte-identical to the
one-row M64 graph; the five-update artifact remains SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`
with identical loss and gradient history and held-out loss `4.839618`. A
separate fifteen-target update differed by one F32 loss ULP
(`6.822133064` versus `6.822132587`), had an identical gradient norm, and
produced the same byte-identical adapter (SHA-256
`629a459767d7d826cc047b6cb6584e47021b1c046fa1e727da88ec19d20e71a0`).

The batched graph exposed one remaining fixed vocabulary cost: the frozen BF16
input-gradient product `8x262144 * 262144x1536 -> 8x1536`. Metal now admits a
narrow exact-arithmetic M8/N32/K64 specialization for that product. It uses 128
threads, F32 inputs/accumulators/output, the precise Metal library, and the same
ascending K accumulation order as the generic tile. Admission requires exactly
eight rows, `in_dim >= 128`, `out_dim >= 65536`, `in_dim % 32 == 0`, and
`out_dim % 64 == 0`; all other shapes retain the generic path.
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SMALL_ROWS=1` is the same-binary rollback.

Two repeated no-frame profiles reduced this kernel from `23.7375 ms` mean to
`20.2905 ms`, a `14.5%` local improvement and `3.447 ms` saved. The trimmed
unified binary then passed the locked same-binary 20-step diagnostic at both
sequence lengths. At 128 tokens, enabled measured `0.466987/0.466848 s`
median/mean (`274.18` input tokens/s) against `0.470053/0.470101 s`
(`272.28` tokens/s) with only this route disabled: `3.253 ms` saved, or
`0.697%`. At 512 tokens, enabled measured `1.464261/1.464651 s`
(`349.57` tokens/s) against `1.471423/1.472216 s` (`347.78` tokens/s):
`7.564 ms` saved, or `0.516%`. The enabled process peaks were
`1,808,107,968` and `6,024,006,080` bytes respectively. Against the retained
MLX-LM cells, MLX remains `1.774x` faster at sequence 128 and `1.478x` faster
at sequence 512, while using `6.356x` and `2.656x` the process footprint.

The sequence-128 enabled/rollback artifacts are
`/private/tmp/antfly-gemma4-zig-e2b-seq128-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`6cb67bd0cf13f5c3f7560fb453072d94ab8359c095e11b1c31c3090558d11cd5`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq128-vocab-backward-n32k64-disabled-diagnostic-v1.json`
(SHA-256
`352a76aa177edcc27d029affe7dbf5d66a5fa6fe2a9d32381d6b97c7d24d86e3`).
The sequence-512 pair is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`2d9fe10409584f3c0f39b85c3854e2f6888a6994cdc363a3c36148ee2dada57f`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq512-vocab-backward-n32k64-disabled-diagnostic-v1.json`
(SHA-256
`fafd5fd7069df1317483919e8d9f6dfec16b83f4895015ea603365d92d2253d3`).
All four bind executable SHA-256
`55796f5424996465b5b0a3aab16fe40bfbdd1a1780d5aa76acba68a543b3ebd7`.

The specialization is training-state exact. The trimmed binary reproduced
one-step loss `7.611551761627197`, gradient norm `3.1437647342681885`, and
adapter SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.
After five updates it reproduced adapter SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`
and the accepted loss/gradient history, with zero fallback. Current-source
required-device ReleaseFast `test-gemma4-finetune` passed 201 tests with two
optional skips, and inference-edition `antfly-main-test` passed its unified CLI
test with all ten build steps successful. The sequence-512 measurements still
incurred page-ins and page-outs, and each A/B arm is one diagnostic sample, so
this result remains optimization evidence rather than a release-campaign PASS.

### Integrated hardening and final v8 diagnostic (2026-08-12)

The final integrated pass hardened two paths that only appeared under a full
multi-step run. Gemma4's fused RMSNorm VJP is now selected only for Metal, so
native execution remains an independent decomposed oracle. The Metal VJP
requires its activation and output gradient to already be device resident. A
frozen norm weight is served from a prepared runtime slot, while a trainable
norm weight must be directly device resident; neither case may silently upload
a host operand during backward. Recreated graph constants now carry a stable
content identity through device residency and output cloning. The runtime keys
frozen RMSNorm slots by that identity instead of a transient device address,
preventing semantically identical constants from consuming all 512 dynamic
slots across a 25-step benchmark.

The packed-weight path also stopped unconditionally locking the prefetch queue.
Synthetic stores and intentionally synchronous callers do not initialize that
queue, so the old lock could sleep forever before inspecting an already
available Q4_0 tensor. Both affected load paths now lock only when
`prefetch_initialized` is true. The isolated packed-Q4 graph regression passed
in 300 ms, and the final real-device ReleaseFast aggregate completed with 201
tests passed, two optional skips, and zero failures. The seven offline Python
Gemma4 modules completed 125 tests with zero failures; all changed Zig sources
passed `zig fmt --check`, and `git diff --check` was clean.

The exact inference-edition v8 CLI has executable SHA-256
`d8e4819493d1d24b9aaba9548e87907aba9a8f3dea949374d5fb7ddf44ed4400`.
It completed every cold, warmup, first-steady, and measured optimizer window in
both strict diagnostic cells. The comparison uses the retained, matching MLX
cell rather than a new unmatched run:

| Sequence | Antfly median / mean | MLX median / mean | MLX speed advantage | Antfly / MLX peak footprint |
| --- | ---: | ---: | ---: | ---: |
| 128 | `0.467937 / 0.468015 s` | `0.263110 / 0.263169 s` | `1.778x` | `1,809,451,384 / 11,492,007,952` bytes |
| 512 | `1.464869 / 1.465439 s` | `0.992220 / 0.990706 s` | `1.479x` | `6,025,595,400 / 15,999,361,256` bytes |

MLX therefore used `6.351x` Antfly's process footprint at sequence 128 and
`2.655x` at sequence 512. The Antfly sequence-128 measured window recorded
720,896 bytes of page-ins, no page-outs, and zero swap. Sequence 512 recorded
360,448 bytes of page-ins, 294,912 bytes of page-outs, and zero swap, so it
still fails the release lane's zero-paging gate. These are diagnostic samples
from an intentionally dirty checkout and are never admissible as release
evidence. The artifacts are:

- `/private/tmp/antfly-gemma4-zig-e2b-seq128-integrated-v8-final-v1.json`
  (`sha256:c81cf70ad66d5fdedf6d990327366959abd494c4927b8f5057929c8a6f7eb845`);
- `/private/tmp/antfly-gemma4-zig-e2b-seq512-integrated-v8-final-v1.json`
  (`sha256:9a4108ce17118088b6dbda69b71982953b3a8972d6c85f90acdbe7c6ec4b426d`);
- `/private/tmp/antfly-gemma4-mlx-e2b-seq128-diagnostic-v1.json`; and
- `/private/tmp/antfly-gemma4-mlx-e2b-seq512-diagnostic-v1.json`.

This pass also added guarded, separately attributable Gemma4 gate/up kernels.
The forward kernel reuses one activation tile across both BF16 projections;
the backward kernel keeps two independent F32 accumulation streams and writes
their sum directly. The most recent focused backward microbenchmark printed
`6.482 ms` for two qualified products plus the ordinary add and `6.074 ms` for
the fused route. Both routes remain opt-in and require exactly 64-compatible
rows. Their forward and backward counters were zero in both final E2E cells
because the real graph presents 128 rows. Telemetry now proves that zero-hit
fact instead of allowing an enabled-but-unused optimization to receive credit.
The next MLP kernel must target the observed 128-row graph shape and earn an
end-to-end win before promotion.

### Rejected MLP storage and seq128 fusion candidates (2026-08-13)

The follow-up pass first tested F16 mirrors of the frozen BF16 MLP weights so
MPSGraph could own the dense products. A full forward-only mirror reduced the
locked sequence-128 mean from the `0.468884 s` control to `0.458693 s`
(`2.17%`) but raised process peak physical footprint from `1,809,778,992` to
`4,957,474,680` bytes and failed the one-step adapter parity gate. A
backward-only mirror reached `0.441966 s` (`5.74%` faster) at a
`4,956,721,016`-byte peak. Its aggregate adapter comparison looked close, but
43 of 100 target tensors failed the required per-tensor numerical gate; the
worst relative L2 delta was `0.0519005` with cosine `0.998653`. Both mirror
routes and their model-sized duplicate storage were removed. The diagnostic
artifacts are:

- `/private/tmp/antfly-gemma4-zig-e2b-seq128-f16-mps-mlp-v15-v1.json`
  (`sha256:0e31d90f82c45d684246e5de517227f6fa373de98fdd7f57813d469a0510b881`);
  and
- `/private/tmp/antfly-gemma4-zig-e2b-seq128-f16-mps-backward-v16-v1.json`
  (`sha256:4da3b394dbe4a247166cecfca049719314d9093f4d94e113b1fd22a36bfe2089`).

The graph dump was then corrected to skip the smaller 2,933-node loss-only
evaluation graph and inspect the 6,693-node training/autodiff graph. The new
debug-only `TERMITE_DUMP_GRAPH_MIN_NODES` threshold composes with
`TERMITE_DUMP_GRAPH_NODES` for that purpose. It showed the production layer-34
gate/up input gradients as nodes `3367` and `3368`, both `[128,1536]`, followed
by add node `3369`. This explained why the rows-64 experimental matcher had
correctly reported zero calls.

Two bit-exact rows-128 simdgroup-M64 fusion variants were evaluated rather
than inferred from the local kernel timing. Both produced the control adapter
SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`,
executed exactly 35 fused calls per optimizer step, and reduced graph commands
from 3,449 to 3,414. Neither reduced the 1,253 compute encoders, however, and
both lost the strict 20-step end-to-end gate:

| seq128 route | Mean / median | GPU mean | Versus `0.467500 s` control |
| --- | ---: | ---: | ---: |
| 32 KiB two-product threadgroup tile | `0.575830 / 0.576104 s` | `0.523477 s` | `23.17%` slower |
| 16 KiB tile plus exact destination add | `0.513459 / 0.513238 s` | `0.460887 s` | `9.83%` slower |

The 16 KiB kernel was locally bit-exact and measured `6.571 ms` versus
`7.618 ms` for two isolated products plus add, demonstrating why a
microbenchmark is insufficient here: combining the two products reduced GPU
scheduling parallelism across the full training frame. The rows-128 matcher,
kernel, and runtime admission were removed. The original rows-64 experiment
remains opt-in and production seq128 training remains on the faster independent
M64 products plus add. The rejected end-to-end artifacts are
`/private/tmp/antfly-gemma4-zig-e2b-seq128-bf16-gateup-pairsum-v20-v1.json`
(`sha256:58748405b58b5e60866df1ff036f4f5c573c4b8d4442c36f4f0e1e5af8d93d2e`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq128-bf16-gateup-pairsum-v21-v1.json`
(`sha256:846f6638bf21e61e982b17942cf87dca908e196e8971b4858e08d216856dd499`).

### Qualified coalesced BF16 M64 backward loader (2026-08-13)

Same-binary rollback attribution identified the standalone frozen-linear input
gradient as the next useful target. Disabling only the forward M64 route raised
the sequence-128 mean from `0.467500 s` to `0.492410 s` (`5.33%`), while
disabling only backward M64 raised it to `0.573880 s` (`22.76%`). The result
also explained the failed pair fusion above: the independent backward products
are valuable, but their per-product transposed-weight loader still had room to
improve.

The legacy M64 backward kernel assigned one output column to four adjacent
threads, making the global BF16 weight reads stride by `in_dim`. The promoted
kernel instead assigns each thread an eight-column contiguous weight segment
and swizzles that segment into the exact existing threadgroup tile. Its tile
shape, output-gradient loader, three simdgroup barriers per K fragment, MMA
order, F32 accumulators, and stores are unchanged. This follows the relevant
design lesson in the pinned MLX 0.31.2 wheel's Apple Steel GEMM sources: use a
transpose-aware block loader, while retaining the synchronization required by
the shared-memory MMA implementation. It does not embed or call MLX.

The real-Metal regression uses the production Gemma 4 E2B dimensions
`rows=128`, `in_dim=1536`, and `out_dim=6144` and requires bit-identical output
against the legacy M64 kernel. It passed. Independent strict CLI one-step runs
also produced byte-identical adapters at SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.
Two 20-step sequence-128 A/B pairs, with the execution order reversed in the
second pair, measured:

| Pair | Legacy mean | Coalesced mean | Legacy GPU mean | Coalesced GPU mean |
| --- | ---: | ---: | ---: | ---: |
| control then candidate | `0.468076 s` | `0.445565 s` | `0.420842 s` | `0.397391 s` |
| candidate then control | `0.468155 s` | `0.445322 s` | `0.420647 s` | `0.397410 s` |

Across those two pairs, mean wall time improved `4.84%` and mean Metal frame
time improved `5.55%`. The route is therefore default-on;
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_COALESCED=1` restores the
qualified legacy loader without disabling the broader M64 specialization.

The final default-on binary then passed strict paired cells at both locked
sequence lengths:

| Sequence | Default median / mean | Rollback median / mean | Default / rollback GPU mean | Wall improvement | Default / MLX mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.444689 / 0.444468 s` | `0.468757 / 0.468896 s` | `0.397388 / 0.420521 s` | `5.21%` | `1.689x` |
| 512 | `1.363282 / 1.366533 s` | `1.467746 / 1.467627 s` | `1.259479 / 1.363107 s` | `6.89%` | `1.379x` |

Peak physical footprint was effectively unchanged: `1,809,992,080` bytes at
sequence 128 and `6,025,562,536` bytes at sequence 512. The retained MLX cells
use `11,492,007,952` and `15,999,361,256` bytes respectively, so MLX remains
faster while using `6.35x` and `2.65x` Antfly's process footprint. The final
binary SHA-256 is
`fd3a023742ca50fbe0bcc2bffcd0f896ed6858f0271512d3446aab50e472f318`.
The default and rollback artifacts are:

- sequence 128:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-coalesced-default-v33-v1.json`
  (`sha256:ddc98ccb6b47aa3c9d370266511745d2e3dab1be8b159aa714b432458a3f533a`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-coalesced-rollback-v34-v1.json`
  (`sha256:22f715eb11fc74c39ebedc05d6575a711e2936084dc17572b9008590815c86dd`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-coalesced-default-v35-v1.json`
  (`sha256:bd6f8140d2871323d373e07a3cafd38b4b5c5de8cbded0db24006d8a62f27fee`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-coalesced-rollback-v36-v1.json`
  (`sha256:2c25e65395a785d0168b32351c650aa0dc618381008179cf5b29f866cb05eb3f`).

These remain diagnostic artifacts from a dirty checkout. The sequence-128
default cell recorded 16,384 bytes of page-ins, and the sequence-512 cell
recorded page-ins and page-outs, so neither is a release-campaign PASS.

### Qualified packed BF16 M64 backward loads (2026-08-13)

The coalesced loader above still issued eight scalar BF16 loads per thread.
Replacing them with ordinary `ushort4` loads looked faster in microbenchmarks,
but a strict one-step CLI run rejected that implementation: the adapter changed
from the qualified
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`
to
`d7da901c07ce5d6d182e020865043b33ad85fb317e3065ae2a9d3d915c359c85`,
and the mean gradient norm changed from `3.143764734` to `2.776347160`.

The fault was an alignment-contract violation rather than an MMA or scheduling
error. Persistent Safetensors BF16 weights are borrowed with
`newBufferWithBytesNoCopy`; they are guaranteed to be dtype-aligned, not
eight- or sixteen-byte aligned. The real E2B checkpoint's projection payloads
start at address modulo eight equal to two. An ordinary Metal `ushort4 *`
therefore asserted alignment the storage does not provide. The promoted kernel
uses `packed_ushort4` for BF16 weights and `packed_float4` for gradient views,
then retains the existing vector conversion, threadgroup layout, barriers, MMA
order, F32 accumulation, and stores.

The real-Metal regression now covers every one of the 11 E2B text projection
geometries that can reach the 128-row M64 route, a nonzero F32 gradient-buffer
view offset, and a borrowed BF16 weight base deliberately offset by two bytes.
It is bit-identical to the scalar-coalesced route. Final user-facing CLI runs
through `antfly inference finetune train gemma4-lora` also produced
byte-identical adapters for the default packed and scalar rollback paths, both
at the qualified adapter SHA-256 above. Training loss (`7.611551762`), gradient
norm (`3.143764734`), and post-update evaluation loss (`6.111435890`) matched
exactly.

The final same-binary diagnostic pairs measured:

| Sequence | Packed median / mean | Scalar rollback median / mean | Packed / rollback GPU mean | Wall improvement | Packed / MLX mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.428906 / 0.428800 s` | `0.446268 / 0.446523 s` | `0.380754 / 0.397433 s` | `3.97%` | `1.628x` |
| 512 | `1.287999 / 1.288199 s` | `1.367928 / 1.368350 s` | `1.182072 / 1.260595 s` | `5.86%` | `1.300x` |

The packed route is default-on.
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_PACKED=1` restores the
qualified scalar-coalesced loader without disabling the broader M64 route.
Peak physical footprint remained effectively unchanged at `1,809,811,856`
bytes for sequence 128 and `6,025,775,528` bytes for sequence 512. The retained
MLX 0.31.2 cells measured `0.263396 s` and `0.990706 s` with peaks of
`11,497,430,816` and `15,999,361,256` bytes, so MLX is still faster while using
`6.35x` and `2.66x` Antfly's process footprint. The root CLI binary SHA-256 is
`8fd482628ddbcb3a2b0190980cd1a940cc41215cc7d66aed6220d49822756545`.

The final default and rollback artifacts are:

- sequence 128:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-default-final-v62-v1.json`
  (`sha256:60465e62d32f02c2079bd99464bb35a76a0d2fea7caad34718a343d39997810a`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-rollback-final-v63-v1.json`
  (`sha256:a1e935426585825c0de8dfe9d72c5331f3cad3bd66c5e70e58587131b40d3c78`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-packed-default-final-v64-v1.json`
  (`sha256:cbbaf6e39cb0fe0add1f8874f33d9bb4c6dee0133a160c663a71030fd74f9260`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-packed-rollback-final-v65-v1.json`
  (`sha256:a7ea29475b87df77adba3f25649d38d74c823f3566cd241b422dea17a40241f3`).

These are bounded diagnostic artifacts from a dirty checkout. The default
sequence-128 cell recorded 65,536 bytes of page-ins, and the sequence-512 cell
recorded page-ins and page-outs, so this kernel promotion is qualified but the
overall branch still does not have a locked release-campaign PASS.

### Qualified packed BF16 M64 forward loads (2026-08-13)

The packed-backward default was reprofiled before changing another kernel. A
diagnostic no-frame run captured the complete 6,693-node training graph and
then failed closed at the expected strict post-step evaluation boundary; its
absolute per-command times are synchronization-distorted and are used only for
ranking. The training graph executed 1,057 dot/GEMM commands in `468.977 ms`
of `961.819 ms` total execution. The largest individual family was the
40-call `128x1536 * 1536x12288` projection at `62.627 ms`; the corresponding
reverse projection and the 6,144-wide MLP projections were also among the top
shapes. Dense dispatch tracing confirmed that eligible forward projections
still used the scalar-load `bf16_simdgroup_m64` kernel while input gradients
used the packed sibling. The retained profile is
`/private/tmp/antfly-gemma4-packed-default-profile-v68.log`
(`sha256:bee9039ef26a98b771a6e18df23b08064f786df3a330071cbbfd24c4d5768331`).

The promoted forward sibling replaces each eight-value scalar BF16 and F32
global-load sequence with two `packed_ushort4` and `packed_float4` loads. It
does not change the bias seed, threadgroup indices, simdgroup loads, barriers,
MMA order, F32 accumulators, or output stores. Packed types are required here:
the persistent Safetensors view is allowed to be only two-byte aligned, and a
compiled F32 activation view is allowed to begin at a four-byte offset.

The real-Metal regression compares the packed route bit-for-bit with the
retained scalar kernel for all 11 E2B projection geometries at 128 rows. It
also pairs a borrowed BF16 base deliberately offset by two bytes with an F32
device view offset by four bytes and requires both outputs to remain
device-resident. A strict user-facing CLI step selected the packed forward
kernel for every eligible projection, completed one Metal optimizer update
with zero graph/interpreter fallback, and matched the scalar rollback exactly:
before/train/after loss was
`6.822133064 / 7.611551762 / 6.111435890`, mean gradient norm was
`3.143764734`, and both adapter files had SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.

Two sequence-128 pairs reversed execution order, and the cleaned final binary
repeated the packed-then-scalar order. A sequence-512 pair exercised the
longer-context pressure regime on the same kernel build:

| Sequence / order | Packed median / mean | Scalar median / mean | Packed / scalar GPU mean | Wall improvement |
| --- | ---: | ---: | ---: | ---: |
| 128, packed then scalar | `0.409764 / 0.409210 s` | `0.428644 / 0.429399 s` | `0.363127 / 0.380234 s` | `4.70%` |
| 128, scalar then packed | `0.409909 / 0.409623 s` | `0.427586 / 0.427572 s` | `0.362807 / 0.380513 s` | `4.20%` |
| 128, cleaned final binary | `0.409577 / 0.409291 s` | `0.428065 / 0.427702 s` | `0.363286 / 0.380291 s` | `4.30%` |
| 512, packed then scalar | `1.215124 / 1.209975 s` | `1.285686 / 1.286470 s` | `1.112794 / 1.181448 s` | `5.95%` |

Across all three sequence-128 pairs, wall mean improved `4.40%` and GPU-frame
mean improved `4.54%`. Peak physical footprint remained effectively unchanged:
the largest packed sequence-128 observation was `1,809,893,800` bytes and the
sequence-512 observation was `6,025,628,048` bytes. Against the retained MLX
0.31.2 means, the remaining wall-time gaps are now `1.554x` at sequence 128 and
`1.221x` at sequence 512. This improves on the immediately preceding
packed-backward default by `4.53%` and `6.07%`, respectively.

The packed forward route is default-on.
`TERMITE_METAL_DISABLE_BF16_FORWARD_SIMDGROUP_M64_PACKED=1` restores the exact
scalar M64 loader without disabling the broader M64 specialization. The
attested inference-edition benchmark binary SHA-256 is
`a11b433d319aba536db70f508874d609c43bf5c97eb89d72b17ebf84cd93f546`.
After removing the temporary no-frame profiling relaxation, the cleaned
attested ReleaseFast inference binary SHA-256 is
`2bf227a22a918351315e938543858d455daef4a6fe94a04a151acadeb1f4ec0c`.
That binary repeated the strict one-step loss, gradient, update, fallback, and
adapter-SHA result above. The real-device ReleaseFast `test-gemma4-finetune`
aggregate also exited successfully with the all-shape packed-forward regression
included.
The diagnostic artifacts are:

- sequence 128, packed then scalar:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-candidate-v71-v1.json`
  (`sha256:b5f6ac1e2738f351f09fa504c8f8c2c6cae792075b8a329edd45deac4788175e`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-scalar-v72-v1.json`
  (`sha256:73dbe10127b3c2d15bf8462b70efc12e6a582bde139baea81c9c739a2868ac8e`);
- sequence 128, scalar then packed:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-scalar-v73-v1.json`
  (`sha256:85d971f50ed634b2dd76956927e691ac2127f263b153d8aaf4cb4462b0c223d6`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-candidate-v74-v1.json`
  (`sha256:9935222ee07f3dfedb3b9ca4c68be0324e0e3cac583a851cd4d0c613c043583e`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-forward-packed-candidate-v75-v1.json`
  (`sha256:a99b1d97dd08e28a49d8e84b8350c7e2c7d596f993342a8cb06e8225a261c505`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-forward-packed-scalar-v76-v1.json`
  (`sha256:de73e784986e1c8fc112ad6dedacab07922ac4806ea7d49cff33bbbbe437eaec`);
  and
- sequence 128, cleaned final binary:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-final-v78-v1.json`
  (`sha256:e040e9470507e73e17b82f4b5f401c3faedd28dadb29a19d951bbceef2a42647`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-scalar-final-v79-v1.json`
  (`sha256:83d179970ee5e5eb6b3b12d0308e5e94325f023388d2e2378ac0710fa445f58e`).

### Rejected zero-bias seed specialization and packed-route audit (2026-08-13)

A follow-up review confirmed that the promoted packed forward kernel changes
only its alignment-safe global loads; the bias seed, shared-memory layout,
barriers, simdgroup MMA order, accumulators, and stores remain identical to the
scalar M64 control. The regression now also observes a runtime
`bf16_forward_simdgroup_m64_packed_calls` counter. With the packed rollback set,
the counter must not move; with the packed route admitted, every tested
projection must increment it exactly once. This closes the prior gap where
bitwise output equality alone did not independently prove route execution.

Gemma 4's frozen dense projections are represented by positive-zero bias
buffers, so a bounded candidate initialized the packed kernel's simdgroup
accumulators directly to zero and skipped the 1,024-value threadgroup bias
tile plus its two barriers. Admission required a slot-preparation scan proving
every bias value was bitwise positive zero, and
`TERMITE_METAL_DISABLE_BF16_FORWARD_SIMDGROUP_M64_ZERO_BIAS=1` restored the
ordinary packed bias-tile path. The same binary
(`sha256:0916896d1ca025389a897fa3a69c9dde9a148128b08101fe792ece59f247cf3e`)
ran default, rollback, rollback, default at both sequence lengths:

| Sequence | Zero-bias wall mean | Rollback wall mean | Zero-bias GPU mean | Rollback GPU mean | Wall effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.409631 s` | `0.410623 s` | `0.363361 s` | `0.363319 s` | `0.24%` faster |
| 512 | `1.216293 s` | `1.215180 s` | `1.114177 s` | `1.113450 s` | `0.09%` slower |

The sequence-128 wall result was not corroborated by GPU time, and both
sequence-512 measures regressed. The candidate was therefore removed rather
than promoted. The retained change is only the packed-route counter and its
rollback/admission assertions. The post-revert required-device ReleaseFast
Gemma 4 aggregate passed 204 tests with two optional skips and zero failures.

The final reviewed inference-edition binary is
`/private/tmp/antfly-gemma4-packed-reviewed-root-v92/bin/antfly`
(`sha256:98acf9abd3820917832e1352324ab880809d44e1dcaf84553163f0c4950c322b`).
Its 20-step sequence-128 confirmation measured `0.410223 s` wall mean,
`0.410580 s` wall median, and `0.363887 s` synchronized Metal-frame mean. It
retained the same workload and initial-adapter semantic digests, used no
diagnostic overrides, and reported 3,449 command dispatches, 1,253 compute
encoders, and 1,650 planned barriers. The retained artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-reviewed-final-v93.json`
(`sha256:16e787ab12c36ea317a145d35d6a5cff5222663b67f69f97302a02b6de62951a`).

The rejected A/B artifacts are:

- sequence 128 default A / rollback A / rollback B / default B:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-default-v84-a.json`
  (`sha256:472d2ca16e8ea0aa680d50a2fddfd1370c5fac5c08cf85191d2ff475d0c00bb9`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-rollback-v85-a.json`
  (`sha256:10af4d858f4478ac81aa22999002fb445a67aa2a0f7b119c8e1e67a84bd77e6f`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-rollback-v86-b.json`
  (`sha256:9e898b69a55d58432db13e649445333a65720103bbbfac9dedf2fa5298913847`),
  and `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-default-v87-b.json`
  (`sha256:8c44a9b1b2c9afb65d65ef50ec5e0d94c6c06cd5cc8515089b9dadd52585f37f`);
- sequence 512 default A / rollback A / rollback B / default B:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-default-v88-a.json`
  (`sha256:c5a55ba5b901b70a6b332d5cc57ea17fe742feab2b201653aada83d0086cf998`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-rollback-v89-a.json`
  (`sha256:c6d43f567cbff8ca2316673642a66efd01d894122d53fccdc5c4d1838e20a004`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-rollback-v90-b.json`
  (`sha256:0b2fb829d6f1a96f41b3cddfebee2da19afb5aab57bdd88ab5527dc1839bbb41`),
  and `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-default-v91-b.json`
  (`sha256:e70c67ebeff3eafcb9d94fb1d578e04d9de4dc4057edd1d320cb6d7b92ae1d04`).

These are still bounded diagnostics from a dirty checkout. Every cell recorded
page-ins, and both sequence-512 cells recorded page-outs; the packed sequence-
512 cell also observed a `-21%` memory-pressure availability delta. The kernel
promotion is qualified, but none of these artifacts is a release-campaign PASS.

The current `TERMITE_ENABLE_FUSED_LINEAR_CROSS_ENTROPY=1` experiment is also
not a cut cross-entropy implementation: it remains default-off and still
materializes the complete `[rows, vocabulary]` logits and gradient-logits
tensors. It must not be used to claim Unsloth-style or Apple-style memory
behavior. The production design is a new forward/backward op pair that saves
only per-row log-sum-exp, valid-count, and mean-loss state, tiles vocabulary in
both directions, and accumulates `d_hidden` and optional `d_weight` without a
global logits tensor. On the current 8-row by 262,144-vocabulary cell, the
existing route owns about 8 MiB of forward logits and about 16 MiB of
logits/gradient working storage during backward. A 64 KiB vocabulary tile plus
the small saved state would cut those CE intermediates by more than 99%.

That design follows the useful part of the industry architecture without
copying CUDA performance claims onto Metal. The pinned
[MLX-LM trainer](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py)
still obtains model logits before its ordinary cross-entropy call. Apple's
[Cut Cross-Entropy implementation](https://github.com/apple/ml-cross-entropy)
and [research paper](https://arxiv.org/abs/2411.09009) establish the
logits-free vocabulary-tiled algorithm, while
[Unsloth's CCE integration](https://github.com/unslothai/cut-cross-entropy)
builds on that work. Antfly should implement the algorithm as first-class Zig
graph and Metal operations, qualify BF16 first, and then add Q4_0, Q4_K, and
Q6_K frozen-output-weight readers. It should not hide it behind the current
misnamed experimental switch.

With both M64 directions using qualified alignment-safe packed loaders, the
normal synchronized frame is now reprofiled. Its 3,449 command dispatches,
1,253 compute encoders, and 1,650 planned barriers make graph/dispatch fusion
the next performance target; another tile-seed micro-optimization is unlikely
to close the MLX gap. The next pass should first attribute and coalesce the
1,057 dot/GEMM, 1,084 elementwise, 399 transpose, and 309 activation-backward
dispatches without reviving the already rejected independent-product fusions.
True cut cross-entropy remains the parallel memory and longer-context target.
Promotion also requires a run-scoped, manifest-bound numerical-kernel policy
fingerprint so environment changes cannot alter approximate kernel admission
mid-run. Direct GGUF QLoRA, E4B, multi-seed convergence, interrupted resume
equivalence, and a clean alternating five-pair release campaign remain open
gates.

These ratios are bounded diagnostic A/B evidence from a dirty Antfly checkout,
not the locked alternating multi-repeat performance campaign, a zero-paging
PASS, or a numerical-parity claim.

Two other candidates were rejected rather than left on by default. MPS BF16
matrix multiplication hit Apple's mixed-dtype assertion because MPS admits the
current F32-output contract only for an F16 right-hand matrix. A shared-row
BF16 reduction remained numerically exact at 128 rows but regressed the training
frame by about `5.2x` because it collapsed row parallelism.

### Measured Metal optimization pass (2026-08-11)

The same one-example, sequence-64, rank-4 Q/V E2B case was used to optimize
the training graph without changing its model, prepared inputs, seed, or
hyperparameters. A custom RMSNorm VJP now replaces the decomposed primitive
backward. For LoRA, norm weights are frozen, so the VJP also returns only
`d_input`; the Metal kernel skips the dead `d_weight` reduction and its private
inverse-RMS buffer. The full `d_input + d_weight` path remains covered for
callers that train norm weights.

| Training-step measure | Pre-fusion | Current RMSNorm VJP | Change |
| --- | ---: | ---: | ---: |
| Logical graph commands | 9,434 | 4,362 | -53.8% |
| Elementwise commands | 6,630 | 1,558 | -76.5% |
| Reduce commands | 649 | 168 | -74.1% |
| Interpreter fallbacks | 0 | 0 | unchanged |
| Warm peak physical footprint | 1,119,488,640 bytes | 1,011,255,864 bytes | -9.7% |

The input-only specialization removes another 239 training graph nodes and
executions (`8,346 -> 8,107` and `6,476 -> 6,237`). In a same-minute profiled
A/B, Metal frame submission moved from `5,779.021 ms` to `5,748.113 ms`
(-30.9 ms, about 0.54%). Whole-process warm time was tied within noise
(`18.25 s` full-gradient control versus `18.24 s` input-only), so this is a
dispatch/memory improvement, not evidence of a material end-to-end speedup.
Loss before/train/after remained `6.822132587 / 9.059597015 / 6.209827900`,
mean gradient norm remained `2.428374052`, and strict Metal fallback counters
remained zero. Native finite differences and native-versus-Metal tests over
hidden widths 3 through 768 bound the worst observed RMSNorm-backward absolute
delta at `7.1525574e-7`.

Two LoRA-backward candidates were not promoted. The full three-gradient region
needs a recomputed checkpoint activation that is unavailable when the region
is scheduled, so canonical direct-B matching caused strict interpreter
fallback and was removed. The safer low-rank `d_after_a + dB` region executed
50 times with zero fallback and reduced commands from 4,362 to 4,262, but its
paired warm time was neutral/slightly worse (`18.07 s` versus `18.01 s`); it
therefore remains opt-in. The dominant measured training work is now the 1,184
dot/GEMM commands and the attention/linear backward path.

The existing raw-linear training-region route was also evaluated rather than
enabled by default. It matched 297 regions with zero fallback and produced a
byte-identical adapter, but alternating warm runs regressed from `17.69 s` and
`17.68 s` without the route to `17.91 s` and `18.19 s` with it. The route stays
disabled for training; region-hit count alone is not a performance result.

Attention batched-dot VJPs now choose operand order and contracting axes that
emit `dQ`, `dK`, `dP`, and `dV` in their final physical layouts. The Metal
batched-dot kernel accepts either matrix axis as the contraction, eliminating
the input/output transpose materializations around those contractions. On the
same E2B case this reduced training commands from 4,362 to 4,224 (-3.2%),
transpose commands from 537 to 399 (-25.7%), graph nodes from 8,107 to 7,969,
and executed nodes from 6,237 to 6,099. A profiled Metal frame moved from
`5,804.462 ms` to `5,778.212 ms` (-0.45%). Three alternating warm whole-process
runs measured `17.93/18.03/18.08 s` before and `17.98/18.00/18.05 s` after
(medians `18.03 s` and `18.00 s`), which is effectively tied. The adapter was
byte-identical (`sha256:df62cf44593f4305e6496470afcd5f5a0e7cbe8174095cf0a6248bf6bfe94168`),
loss and gradient metrics were unchanged, and fallback and host-output counts
remained zero. This is a real graph/dispatch reduction, not a material
end-to-end speedup claim.

Attention-sized batched dot products now use one cached
`MPSMatrixMultiplication` across every batch matrix, including all four
left/right contracting-axis layouts. Large 2D projections already used the
same Apple MPS primitive, so this targets the previously scalar-per-output
batched path rather than duplicating that optimized projection route. The
default admission gate is deliberately bounded to `batch_count >= 2`,
`m >= 128`, `n >= 8`, and `k >= 32`; sequence-length-64 alternating runs were
tied at `17.88 s` median with and without MPS. At sequence length 256, the
profiled training frame improved from `5,603.455 ms` to `5,369.956 ms`
(-4.17%), while peak RSS changed by only 49,152 bytes. Three alternating
whole-process runs improved from `16.95 s` to `16.81 s` median (-0.83%). The
scalar and MPS paths produced byte-identical adapters
(`sha256:7cfa386de0e2d76ec0f7c01a40bed375fdbc83e556611568d853b5059b6cc9fd`),
identical training loss and gradient norm, one optimizer update, and zero
fallback or host-output events. `TERMITE_METAL_DISABLE_DOT_GENERAL_BATCHED_MPS`
provides the paired control, while
`TERMITE_METAL_REQUIRE_DOT_GENERAL_BATCHED_MPS` makes route admission a hard
test assertion. This is a bounded one-example optimization result, not MLX-LM
parity evidence.

These are Antfly old-versus-new measurements, not an MLX-LM comparison. The
locked alternating multi-repeat same-Mac campaign described above is still
required before claiming MLX performance parity.

This is useful execution evidence, not an oracle or quality result: it is one
example and one update, the before/after rows differ in supervised-token count,
and no HF/native/Metal gradient or update trace comparison was produced.
Broad multi-step convergence, deterministic overfit, bounded peak memory,
repeated adapter reload/generation, and required CI remain open. Exact
epoch-boundary process-kill/resume has now passed for bounded real E2B and E4B
Metal jobs; those narrow gates are no longer roadmap-only claims.

Local results are not substitutes for required CI:

```sh
zig build test-gemma4-finetune
zig build test-gemma-graph
python3 scripts/compare_gemma4_lora_hf_zig.py validate-lock
python3 -m unittest discover -v -s scripts -p 'test_*gemma4*py'
```

The macOS job is a synthetic real-GPU gate, not an E2B/E4B scale gate. It must
run successfully in CI and be made required in repository branch protection;
workflow source alone is not evidence that either has happened. The Python
contract tests validate the lock, schemas, deterministic fixtures, producer
roles, evidence ledgers, numerical checks, runner coordination, and benchmark
pairing on synthetic data; they do not execute the locked HF or MLX-LM model
campaigns.

The official E2B QAT Q4_0 GGUF (`sha256:fa401b55...dec6634`) now passes the
explicit direct-GGUF Metal lane for one real UltraChat row at sequence length
64, rank 2, Q/V targets. Two fresh production-default processes produced exact
gradient fingerprints and the same adapter
`sha256:881e74dd...27016e`. The complete two-epoch process-kill/resume gate then
produced adapter `sha256:e38d0dd8...d99d961`, with uninterrupted and resumed
epoch-2 loss `3.6382982731`, gradient norm `0.3051027656`, one Metal optimizer
step, and zero fallback. The qualification report is
`/private/tmp/antfly-gemma4-e2b-gguf-qlora-resume-acceptance-20260819-v3/qualification_report.json`.

The important negative result is retained: wrapping the quantized tied-head
projection, CE, and input-gradient projection inside one builder-level fused
loss node was nondeterministic even after internal encoder fences. The exact
decomposed graph was repeatable, so direct GGUF selects
`linear_cross_entropy_mode = "decomposed-gguf"` automatically and binds that
mode into the checkpoint fingerprint. This closes optimizer and resume
correctness for the exercised E2B shape; it does not yet close peak-memory,
MLX/HF numerical parity, multi-seed task quality, E4B GGUF, or deployment
generation gates. The public `qlora-sft` recipe therefore remains a typed
error.

### Qualified gate/up backward graph fusion (2026-08-13)

Gemma 4 LoRA product training now recognizes the same-layer gate/up
backward-input pair
`d_gate @ W_gate + d_up @ W_up` as one prepared runtime region. The Metal
entry point consumes both cached BF16 weight slots, produces the sum directly,
and keeps the non-anchor dot elided after the fused add has been satisfied.
This removes one dot dispatch and one add dispatch per transformer layer. The
runtime-region plan owns the match, so steady-state execution does not rescan
the graph.

The fusion is enabled by default only for the qualified 64-, 128-, and
512-row lanes. `TERMITE_METAL_DISABLE_GEMMA4_BF16_GATE_UP_BACKWARD_INPUT_SUM=1`
is the same-binary production rollback. Explicit
`TERMITE_METAL_ENABLE_GEMMA4_BF16_GATE_UP_BACKWARD_INPUT_SUM=1` retains an
experimental surface for other kernel-supported 64-row multiples. Matcher
misses and runtime misses retain the two-dot-plus-add path.

A CPU sample caught an initially hidden host regression: generic Gemma
residency telemetry recursively classified the fused backward add 35 times per
step. The fusion now records its statically proven `residual_add` category
directly. Planned-region time on the profiled sequence-128 step fell from about
10.25 ms to 6.57 ms, slightly below the 6.88 ms rollback profile.

On the locked diagnostic E2B Q/V rank-16 workload, the final default-on binary
and its disable override produced the sequence-128 cells below. The sequence-512
cells used the immediately preceding attested qualification binary with the
fusion explicitly enabled; its region, prepared-slot kernel, liveness, and
telemetry paths are identical, and only the subsequent default-admission policy
changed.

| Cell | Fused mean | Rollback mean | Wall improvement | Structural evidence |
| --- | ---: | ---: | ---: | --- |
| seq128, accumulation 1, repetition 1 | 413.357 ms | 414.585 ms | 1.227 ms (0.30%) | 35 fusions; 3,344 -> 3,274 commands |
| seq128, accumulation 1, repetition 2 | 414.189 ms | 415.637 ms | 1.448 ms (0.35%) | reverse order; same counts |
| seq128, accumulation 4 | 1,687.872 ms | 1,695.919 ms | 8.048 ms (0.47%) | 140 fusions; four plan-cache hits |
| seq512, accumulation 1, repetition 1 | 1,204.384 ms | 1,210.696 ms | 6.311 ms (0.52%) | 35 fusions; 3.87 ms GPU reduction |
| seq512, accumulation 1, repetition 2 | 1,204.381 ms | 1,215.415 ms | 11.034 ms (0.91%) | reverse order; 3.62 ms GPU reduction |

The expanded Metal parity test covers both Gemma FFN widths at rows 128 and
512. Rows 64 are bit-exact. At rows 512, the worst observed maximum absolute
delta was `4.005e-5` and worst relative L2 was `2.80e-6`. A full one-step
adapter comparison against rollback had identical loss
(`7.611551761627197`), gradient-norm delta `0.000263`, state maximum absolute
delta `0.0019914`, relative L2 `0.0032884`, and cosine `0.9999946`. Those state
metrics pass the checked-in native/Metal BF16 profile (`0.02`, `0.005`, and
`0.9999`). The packed accumulation order is not byte-identical to rollback, so
promotion relies on the numerical oracle rather than an adapter SHA claim.

The locked seq2048 diagnostic exceeded its 1,800-second subprocess watchdog
while still actively executing and published no sample. It is therefore not a
pass or a regression. Rows 2048 remain outside the default-qualified set; the
legacy path remains active there until the long-sequence harness can measure a
complete paired cell within a practical watchdog. These local artifacts come
from a dirty diagnostic checkout and are performance/qualification evidence,
not release evidence or an MLX parity claim.

### Matched E2B DPO optimization benchmark (2026-08-17)

The locked one-token DPO case now recognizes the exact structural condition in
which chosen and rejected responses share one prompt and each contain one
completion token. Policy and frozen-reference scoring project the final prompt
row once for both candidates. Their opposing logprob gradients are then encoded
as two weighted targets on that same causal row, so one compiled backward
microbatch replaces the previous chosen and rejected sequence microbatches.
General multi-token or non-shared-prompt data remains on the existing sequence
path.

Five fresh-process runs of the final strict-Metal binary produced identical
25-update loss trajectories and byte-identical trained adapters. The table uses
the median of the five per-run measured medians for the optimized Antfly cell;
the prior Antfly and pinned MLX-LM cells retain their matched 20-update measured
medians.

| Implementation | Measured seconds / update | Peak process footprint | Result |
| --- | ---: | ---: | --- |
| Prior Antfly pair path | `1.846122` | `3,659,354,480` bytes | baseline |
| Optimized Antfly paired-row path | `0.760557` | `3,656,355,992` bytes | `2.427x` faster than prior Antfly |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `0.520192` | `12,782,264,168` bytes | Antfly is `1.462x` duration / `46.21%` slower |

Antfly uses `71.40%` less comparable peak process footprint than MLX-LM, and
MLX-LM uses `3.496x` Antfly's footprint. All five Antfly adapters have SHA-256
`fd2060d835e2c35a052dec593f02091b8df5b7aaa61cd0c891bddccb72dfd430`;
all 100 adapter tensors changed. Adapter movement was L2 `1.907393` versus
MLX-LM's `1.824274`. Across cold, first, warmup, and measured updates, the
optimized trajectory differs from the prior Antfly route by mean absolute
`0.000434` and maximum absolute `0.002761`; final loss differs by
`1.78e-6`. Against MLX-LM those values are `0.009541`, `0.040490`, and
`2.93e-5` respectively.

An important correctness guard remains explicit. Reuse-enabled repetitions
occasionally exposed a stale compiled update even after eager scoring was
isolated. Gemma4 DPO therefore disables the Metal in-frame private-buffer pool
for the complete preference run while retaining compiled execution and all
qualified fused kernels. The report records
`metal_buffer_reuse_mode = "disabled-for-dpo-run"`. This restores exact
five-run determinism without giving back the speedup, but it postpones the
lower-memory `~2.57 GB` experimental result until cross-scope alias dependencies
can be proven globally.

Initial same-graph policy/reference error is exactly zero. The Antfly-to-MLX
base reference chosen/rejected logprob deltas remain `0.213154` and `0.037575`,
with a `0.175579` preference-margin delta, so this is matched training-behavior
evidence rather than exact cross-framework forward-logit parity. It is also a
bounded E2B sequence-128 one-token diagnostic, not multi-token, E4B, or release
campaign qualification. The retained comparison artifact is
`/private/tmp/antfly-gemma4-e2b-dpo-final-mlx-comparison-20260817-v1.json`.

### Real UltraFeedback multi-token DPO parity (2026-08-17)

The general sequence path now also has a provenance-locked real-data result.
`scripts/materialize_gemma4_dpo_hf_parity.py` consumes
`HuggingFaceH4/ultrafeedback_binarized` `test_prefs` at revision
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`. The source Parquet SHA-256 is
`e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389`.
The materializer verifies the exact local Gemma4 tokenizer contract, forbids
truncation, requires distinct chosen/rejected token IDs and a score margin, and
selects one example from each of five total-token buckets through sequence 512.
A second-admitted-per-bucket mode produced a disjoint five-example holdout.

The public recipe CLI trained the five-example set for five epochs through 25
complete DPO updates from the same rank-16, alpha-32 Q/V seed adapter. The
pinned MLX `0.31.2` / MLX-LM `0.31.3` runner consumed the exact materialized
token IDs and update order. Both adapters were then loaded and scored by
`scripts/evaluate_gemma4_dpo_adapters_mlx.py` under one MLX oracle, eliminating
base-runtime differences from the result comparison.

| Shared MLX evaluation | Compiled-score Antfly adapter | MLX-trained adapter |
| --- | ---: | ---: |
| Training preference accuracy | `1.0` | `1.0` |
| Training mean DPO loss | `0.00163453` | `0.000813062` |
| Training mean reward margin | `12.01240` | `12.88266` |
| Disjoint holdout preference accuracy | `0.6` | `0.6` |
| Disjoint holdout mean DPO loss | `0.652640` | `0.664016` |
| Disjoint holdout mean reward margin | `0.429506` | `0.634040` |

Holdout preference-decision agreement is `1.0` across all five rows. The
Antfly/MLX holdout mean-loss ratio is `0.98287`, with absolute accuracy delta
zero. This is a bounded behavioral-parity pass on real, disjoint preferences;
five training and five holdout examples are not a broad quality or convergence
campaign.

The original general sequence path materialized large vocabulary-shaped work
for both scoring and the signed DPO backward objective. The production path now
uses four reusable supervised-row buckets (`64`, `128`, `256`, and `512`), a
compact uniform `[row, token, scale, ...]` target contract, and frozen tied-head
fused linear cross-entropy. Policy and frozen-reference scoring return a single
device-reduced sequence logprob; backward returns the exact signed summed-logp
gradient without owning a global logits tensor. Chosen and rejected backward
passes remain separate batch-1 microbatches because a batch-2 experiment
regressed both throughput and memory. The live policy scorer now executes those
two loss-only forwards through the trainer's cached pruned compiled session.
Frozen-reference precomputation keeps explicit zero-LoRA bindings on its
separate route, so scoring never swaps or mutates optimizer-owned weights.

The exact final ReleaseFast binary SHA-256 is
`ca5cb5b3bbbbf7816a88379d950bfcde57e9b6ac5505010788d6e24b167ce257`.
Its fresh fixed-work result is:

| Implementation | Measured median / mean | Reference precompute | Lifetime peak physical footprint |
| --- | ---: | ---: | ---: |
| Prior Antfly sequence path | `41.4782 / 45.5115 s` | `23.9595 s` | `32,555,638,368` bytes |
| Intermediate Antfly fused path | `6.97423 / 7.48183 s` | `13.1064 s` | `14,775,393,072` bytes |
| Final Antfly compiled-score path | `6.55229 / 6.97276 s` | `11.5299 s` | `15,034,801,184` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.05975 / 2.06250 s` | `5.31961 s` | `19,168,812,736` bytes |

The compiled scorer reduces the intermediate fused path's median by `6.05%`
and mean by `6.80%`. Relative to the original sequence path, the retained path
is `6.330x` faster by median and `6.527x` faster by mean. Against pinned
MLX-LM, Antfly is now `3.181x` the median duration and `3.381x` the mean
duration, while using `21.57%` less comparable peak process footprint. The
conservative footprint is the larger of two repetitions; it is `1.76%` above
the intermediate fused run and `53.82%` below the original path. MLX reference
precompute remains `2.167x` faster.

Two independent compiled-score repetitions published byte-identical adapter
payload SHA-256
`e49063871260b65baf09bdbf277ce210afe195db4d739d7cc5bde9b2f7acf337`
and identical 25-loss trajectories. Their medians were `6.552286 s` and
`6.459342 s`; the table retains the slower result. Against the intermediate
fused adapter, the final update has cosine `0.996205`, norm ratio `1.101293`,
relative L2 difference `0.136449`, and `97.445%` nonzero sign agreement. The
compiled reduction order therefore is not a byte-equivalent multi-step route,
even though the first update is byte-identical. The shared MLX oracle above is
the acceptance evidence: training and holdout decisions remain identical, and
the final Antfly holdout mean loss is closer to MLX than the intermediate
adapter's (`0.652640` versus `0.612447`, with MLX at `0.664016`).

The next high-value target is the compiled backward graph, not another scoring
special case. A sequence-512 frame trace attributes the chosen and rejected
backward passes to `1,311` and `1,316` Metal compute commands. Each carries
`1,336` planned scopes and `1,060` planned barriers, for `1.784 s` and
`1.129 s` of GPU time in the traced cold update. MLX compiles scoring, loss,
backward, and optimizer as one step; Antfly still submits the two signed
sequence backwards independently. The next pass should coarsen safe graph
regions and remove redundant cross-region barriers while retaining batch-1
memory behavior, then reprofile before attempting another shape-specific
kernel.

Retained artifacts:

- training case semantic SHA-256
  `9f788b5cb6090c66f257ec6d35cea02c8e0d5170a8e0b0f1d9076e4879af2ade`;
- holdout case semantic SHA-256
  `afea7cdb02f4b4ed838a905872d82d5ad678cd31d60b911e5196529ba1c5c5c1`;
- prior Antfly report and adapter under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-run-v1`;
- exact final optimized Antfly report and adapter under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-compiled-policy-final-v1`;
- deterministic repetition under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-compiled-policy-repeat-v1`;
- matched MLX report
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/mlx-run-v3-cross-eval.json`; and
- optimized training and disjoint shared-oracle results
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/compiled-policy-{training,heldout}-mlx-cross-eval-v1.json`.

### Exact detached-gradient DPO and graph coalescing follow-up (2026-08-17)

The general multi-token Metal route no longer needs a score-only policy forward
followed by a second backward execution for each side of the preference pair.
For logical gradient accumulation one and a non-recursive adapter, chosen and
rejected each execute exactly once with a coefficient of one, producing the raw
summed sequence logprob and its gradient. The chosen device gradients are
detached before the rejected branch runs; both device sets are then combined
with the exact host DPO coefficients and flushed through AdamW once. Reports
identify this route as
`policy_scoring_mode = "backward-loss-reuse-device-detached"` and
`training_microbatch_mode = "chosen-rejected-raw-gradients-device-combined"`.
`ANTFLY_GEMMA4_DPO_DETACHED_GRADIENTS=0` restores the proven two-backward
rollback. Recursive LoRA, explicit pair-graph experiments, and logical gradient
accumulation above one conservatively retain their older route.

The one-step detached/rollback adapter comparison measured gradient cosine
`0.9997975674`, norm ratio `0.9999953928`, relative L2 delta `0.020121`, and
`99.99048%` sign agreement. Independent detached repetitions were byte-identical
at adapter SHA-256
`37c5a0672306a4a81c8aec3c6bdcd75adc486ab9d0c02adc122859e71843b959`.
The complete 25-update final adapter is SHA-256
`5bf72497c5c57208d0903a4878ae2d9490d6a58407d8756ff7fa31acd6a58ae5`;
the retained run before the final graph-fusion pass has the same bytes and the
same 25-loss trajectory.

Two command-level changes survived same-binary rollback gates:

- The prepared static execution plan now groups qualified rank-16 LoRA-A MPS
  dots by exact parameter family and shape. Fifteen Q/V pairs per backward
  branch coalesce, reducing branch compute-command counts from `1,311/1,316`
  to `1,296/1,301`. Pair-backward GPU time improved `0.54%` in the controlled
  A/B. `TERMITE_METAL_DISABLE_GROUPED_LORA_A_R16=1` is the rollback.
- A precise-library add3 kernel fuses 18 single-use, same-shape F32 add chains
  per branch while preserving `(a+b)+c` versus `c+(a+b)` order. It rejects
  broadcasts, multi-use producers, two deferred children, and chains deeper
  than one level. Planned scopes fell from `1,321` to `1,303` and barriers from
  `1,060` to `1,042` per branch. Across two alternating pairs, pair-backward GPU
  time averaged `2,906.220 ms` versus `2,913.438 ms` with
  `TERMITE_METAL_DISABLE_ADD3_FUSION=1`, a small but repeatable `0.248%` win.

The custom batched-MPS rollback and a packed K=16 expansion GEMM were rejected.
Disabling batched MPS reduced encoder count but raised chosen/rejected GPU time
to roughly `2.704/2.052 s`; the fallback kernels are compute-bound. The packed
`512x16 * 16x1536` candidate passed exact output parity but was slower in both
same-binary pairs, so its kernel, admission, and tests were removed. These
results make dispatch-count reduction alone an insufficient promotion rule.

The final fixed-work result from the production-default ReleaseFast path is:

| Implementation | Measured median / mean | Reference precompute | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Prior compiled-score Antfly path | `6.55229 / 6.97276 s` | `11.5299 s` | `15,034,801,184` bytes |
| Final detached-gradient Antfly path | `5.05956 / 5.24470 s` | `11.5553 s` | `14,744,034,000` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.05975 / 2.06250 s` | `5.31961 s` | `19,146,055,360` bytes |

This table is retained as the detached-gradient attribution checkpoint; the
compact-attention and allocator-lifetime sections below supersede it for the
current shipping result.

The exact hardened CLI for the final Antfly row is SHA-256
`371634ba358c528100b2752f27f2779e8a9f186927e2548bc29d04fd0c2a186a`.
Detached gradients reduce the prior Antfly median by `22.78%` and mean by
`24.78%`. Antfly remains `2.456x` the MLX median duration and `2.543x` the MLX
mean duration, while its peak physical footprint is `22.99%` lower. The final
Antfly report records loss `0.1946493`, mean reward margin `4.299646`, accuracy
`0.84`, exact initial same-base policy/reference logprobs, 25 optimizer steps,
and 50 physical branch microbatches.

The new adapter also passed the shared-runtime behavioral gate rather than
inheriting the older adapter's result. Under the pinned MLX oracle, Antfly and
MLX both reached training accuracy `1.0` and holdout accuracy `0.6`, with
preference-decision agreement `1.0` on all five training and five disjoint
holdout rows. Final Antfly/MLX mean DPO loss was
`0.00187977/0.000813062` on training and `0.612271/0.664016` on holdout. The
bounded holdout mean-loss ratio is `0.922073`; this remains five-row diagnostic
evidence, not a broad convergence claim.

Evidence:

- final report and adapter:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-detached-final-full-v2`;
- training shared-oracle result:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/detached-final-training-mlx-cross-eval-v1.json`;
- disjoint holdout shared-oracle result:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/detached-final-heldout-mlx-cross-eval-v1.json`; and
- matched MLX training report:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/mlx-run-v3-cross-eval.json`.

The remaining gap was not in policy score reuse or a single uncovered rank-16
GEMM. Both backward branches still owned about 1,300 compute commands and
roughly `1.78/1.12 s` of GPU work. A retained trace attributed enough of that
work to the decomposed attention VJP to justify the compact-GQA campaign below.

### Compact GQA attention VJP and E2B-only promotion (2026-08-18)

The retained production trace showed `140` masked-softmax calls, `70`
softmax-backward calls, and the surrounding attention contractions in one DPO
update. The accepted implementation packs token-major Q/K/V/dO once into a
reused head-major private buffer, evaluates QK-transpose, dO-V-transpose,
dScore-transpose-Q, dScore-K, and P-transpose-dO as five batched MPS matrix
multiplications, applies causal/sliding softmax VJP in one Metal kernel, then
unpacks dQ and reduces the expanded per-query-head dK/dV back into compact
shared-KV tensors. The workspace reuses Q/K/V regions only after their last
encoded read. Sequence lengths below `128` retain the scalar compact route.

The final shipping policy is deliberately architecture-specific. The compact
VJP is default-on only for the exact qualified E2B topology: hidden size
`1536`, `35` layers, `8Q/1KV`, local/global head dimensions `256/512`,
intermediate size `6144`, sliding window/pattern `512/5`, `20` shared-KV tail
layers, and PLE width `256`. `TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION=1`
restores the decomposed production graph. The explicit
`TERMITE_METAL_ENABLE_GEMMA_GQA_ATTENTION_FUSION=1` variable remains a research
override for non-qualified shapes, and
`TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_MPS_VJP=1` selects the slower scalar
compact diagnostic route. `TERMITE_METAL_TRACE_GEMMA_GQA_ATTENTION_FUSION=1`
reports the selected compact kernel geometry.

The final same-binary, fixed-25 E2B comparison used the real fingerprinted
UltraFeedback slice, rank-16/alpha-32 Q/V LoRA, sequence length `512`, and the
locked cold/first/three-warmup/20-measured protocol:

| Implementation | Measured median / mean | Reference precompute | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Final binary, decomposed rollback | `4.286381 / 4.422427 s` | `8.692538 s` | `15,206,603,736` bytes |
| Final binary, production E2B default | `3.868544 / 3.958108 s` | `12.087822 s` | `14,622,085,632` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.059747 / 2.062500 s` | `5.319610 s` | `19,146,055,360` bytes |

The production default won `18/20` paired measured updates and reduced median
and mean update time by `9.75%` and `10.50%`, respectively. It saved
`584,518,104` bytes (`3.84%`) versus rollback. Relative to pinned MLX, the
conservative final-binary result is `1.878x/1.919x` the median/mean duration
while using `23.63%` less peak process footprint. Two earlier optimized runs,
before the memory-pressure-heavy E4B campaign, measured `3.696854` and
`3.696983 s` medians; all three runs produced the same complete loss trajectory
and byte-identical adapter payload
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`.
This table isolates the compact-attention change before the allocator-lifetime
promotion below.

Against the decomposed 25-step adapter, the accepted E2B update has cosine
`0.99998331`, norm ratio `1.00004211`, relative L2 difference `0.00577777`, and
`99.9234%` sign agreement. Initial policy/reference logprobs remain exact. In
the shared pinned MLX evaluator, the default adapter and MLX both reached
training accuracy `1.0` and disjoint holdout accuracy `0.6`, with decision
agreement `1.0` on all ten rows. Their training mean DPO losses were
`0.00191398/0.000813062`; holdout losses were `0.631058/0.664016`. This is a
bounded real-data behavior gate, not a broad convergence claim.

E4B was tested and explicitly rejected from the default. A sequence-512
one-step run exercised both `8Q/2KV` local `head_dim=256` and global
`head_dim=512` MPS routes. Its update remained close to rollback (cosine
`0.999815`, norm ratio `0.999998`, relative L2 `0.01924`), but the matched
25-step trajectory diverged materially:

| E4B implementation | Median / mean | Final loss / reward / accuracy | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Decomposed rollback | `32.5795 / 39.7838 s` | `0.11109 / 182.904 / 0.88` | `21,505,631,064` bytes |
| Compact MPS research override | `31.9462 / 36.8085 s` | `0.24120 / 3.351 / 0.80` | `21,170,132,992` bytes |

The final E4B update cosine fell to `0.807816`, relative L2 difference rose to
`0.661817`, and sign agreement fell to `82.00%`. The small speed and memory wins
therefore do not satisfy the quality/parity gate. The final no-override E4B CLI
smoke emitted no compact trace and reproduced the decomposed one-step adapter
byte-for-byte. E4B needs a numerically tighter compact reduction or a different
bounded-memory attention strategy before reconsideration.

### Planned-encoder DPO buffer reuse and E2B promotion (2026-08-18)

The remaining coarse DPO guard disabled all same-frame private-buffer reuse.
The lifetime audit found two independent contracts. First, the executor must
not free a pre-materialized parameter or constant after an ahead-of-order fused
consumer while its own graph position is still pending. Second, a private
buffer may enter the same-frame pool only while a barrier-capable planned
encoder owns its last use; reuse forces an encoder barrier. Releases outside
that scope are quarantined until command-buffer completion. Focused tests cover
ahead-of-order graph liveness, same-encoder reuse fences, unscoped quarantine,
completion publication, live escaped tensors, and cancelled frames.

The scoped policy now configures in-frame and completion-fenced reuse as one
drained-boundary transaction and restores both on every exit path. It defaults
on only for the exact qualified E2B topology. Set
`ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE=0` for the production rollback. Setting
it to `1` remains a research override for other shapes; E4B and unknown
topologies otherwise stay fail-closed.

Three alternating reuse-enabled fixed-25 runs produced byte-identical adapters
and exactly identical complete loss arrays around a same-binary reuse-disabled
run. Their medians were `2.494348`, `2.540593`, and `2.545429 s/update`. The
final no-override shipping-binary run measured `2.536096 / 2.825541 s`
median/mean and `8,758,956,424` bytes peak footprint:

| E2B buffer policy | Measured median / mean | Peak physical footprint |
| --- | ---: | ---: |
| Completion-fenced cache; in-frame reuse disabled | `4.089158 / 4.145905 s` | `14,621,594,016` bytes |
| Planned-encoder in-frame reuse plus completion-fenced cache | `2.536096 / 2.825541 s` | `8,758,956,424` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.059747 / 2.062500 s` | `19,146,055,360` bytes |

Against the same-hot-path rollback, the promoted default is `37.98%` faster by
median, `31.85%` faster by mean, and uses `40.10%` less peak memory. It is now
`1.231x` the MLX median duration and `1.370x` the MLX mean duration while using
`54.25%` less peak process footprint. All candidate, rollback, and final-default
runs produced adapter SHA-256
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`
with exact initial policy/reference parity and identical final loss, reward,
and accuracy. A final E4B one-step smoke reported in-frame reuse disabled and
reproduced its decomposed adapter SHA-256
`75ff9a99fc02b74c797a0da4f74aaf63082fe97aba53fa15e7e0a3dcf6ac62b5`.

The final ReleaseFast CLI is SHA-256
`b3af4a952966e354346745b05cb00bcf3c4111e3a1c9d8fb339389fd8e94b4f6`.
The focused Metal gate selected `244` tests: `242` passed and two optional
real-artifact tests skipped. The portable non-Metal gate selected `229` tests:
`212` passed and `17` Metal-dependent tests skipped. The direct native
comparison covers scalar sequence `8` and the MPS admission boundary at
sequence `128`, including full causal and sliding-window forward/backward
paths. A final fixed-25 CLI run on that exact binary selected the E2B
planned-encoder reuse default and reproduced the qualified adapter payload.

Evidence:

- final production-default E2B report and adapter:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-reuse-default-final-v1`;
- three reuse-enabled qualification runs:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-in-frame-reuse-v{1,2,3}`;
- same-hot-path reuse-disabled qualification run:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-reuse-rollback-same-binary-v1`;
- final E4B fail-closed policy smoke:
  `/private/tmp/antfly-gemma4-e4b-dpo-one-step-reuse-policy-final-v1`;
- pre-reuse compact-GQA production report:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-gqa-default-final-v7`;
- E2B training and disjoint shared-MLX evaluation:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/gqa-mps-v4-{training,heldout}-mlx-cross-eval.json`;
- E4B compact and rollback fixed-25 reports:
  `/private/tmp/antfly-gemma4-e4b-dpo-real-fixed25-{gqa-mps,baseline}-20260818-v1`; and
- E4B final update comparison:
  `/private/tmp/antfly-gemma4-e4b-gqa-mps-vs-baseline-fixed25-20260818-v1.json`.

### Production in-place RMSNorm-backward residual-add fusion (2026-08-18)

A node-level sequence-512 E2B trace narrowed the next normalization candidate
to `69` adjacent `fused_rms_norm_backward -> reshape -> add` chains in each
chosen/rejected backward, or `138` opportunities per DPO optimizer update. The
initial candidate extended the frozen-weight RMSNorm-backward Metal kernel with
an exact-operand-order residual add, rejected graph outputs and protected
scalar tails, and retained the ordinary path for trainable norm weights. Its
standalone output allocation was bit-exact but failed the production memory
gate: the fixed-25 footprint rose by `429,539,304` bytes (`4.90%`) for a
noise-level `0.08%/0.12%` median/mean improvement, so that implementation was
kept default-off.

The production follow-up writes into the residual's existing Metal range. It
requires the residual to be an internal add, its exact final graph use to be
the matched add, and its storage to be executor-owned. Runtime alias checking
compares exact byte ranges rather than rejecting disjoint retained views of a
shared allocation. Overlapping live aliases, graph outputs, protected values,
borrowed runtime inputs, trainable norm weights, shape changes, and unavailable
planned-encoder barriers all fail closed. The write emits an explicit
buffer-scope barrier before reusing the range. Chained matches may consume a
preceding fusion's already-materialized add output even though that producer
is marked skipped for traversal bookkeeping.

The in-place one-step trace executed all `138` chosen/rejected chains without
a decline, warning, or fallback. Its adapter remained byte-identical to the
prior final profile at SHA-256
`99c41810b809f6ee7ed3f96fc5a884a00d9e5d0296a8040bbdb03a73998c7732`.
The same-binary fixed-25 in-place and rollback runs also had identical complete
loss arrays and the canonical final adapter SHA-256
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`:

| RMS backward path | Measured median / mean | Reference precompute | Lifetime peak footprint | Completion-cache peak | Metal allocation requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-place final-use residual | `2.518822 / 2.817389 s` | `8.773315 s` | `8,556,564,872` bytes | `5,244,553,216` bytes | `138,760` |
| Same-binary rollback | `2.502889 / 2.813252 s` | `8.796792 s` | `8,760,152,432` bytes | `5,421,238,272` bytes | `145,660` |

Throughput is neutral: the in-place route was `0.64%` slower by median and
`0.15%` slower by mean, while total process wall time was `0.22%` faster. The
memory result is material in the controlled run: lifetime peak fell `203,587,560`
bytes (`2.32%`), completion-cache peak fell `176,685,056` bytes (`3.26%`), and
allocation requests fell by exactly `6,900` (`138` fusions times `50`
microbatches). The route is therefore production-default as a memory and
allocator-pressure optimization, not claimed as a throughput win. Automatic
enablement is restricted to the qualified E2B hidden width (`1536`); E4B
remains default-off pending a multi-step same-binary qualification.
`TERMITE_METAL_DISABLE_RMS_NORM_BACKWARD_RESIDUAL_ADD_FUSION=1` is the rollback
switch and takes precedence; the legacy enable variable remains accepted and
can be set false for an equivalent process-local opt-out or true to force an
explicit research run on an otherwise unqualified width.

The final no-enable-override ReleaseFast CLI is SHA-256
`625a1fde0109587eecf9cd79415e6c306839eab86d53aa654dd3536f0b69480c`.
Its traced E2B one-step smoke executed all `138` in-place fusions with zero
decline, warning, or fallback markers and reproduced the canonical one-step
adapter. The final portable matrix selected `230` tests (`212` passed and `18`
Metal/optional tests skipped); the required-device matrix selected `238`
tests (`236` passed and two optional real-artifact tests skipped).

Evidence:

- in-place one-step fusion trace:
  `/private/tmp/antfly-gemma4-e2b-rms-residual-inplace-one-step-v6-20260818.log`;
- byte-exact one-step run:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-one-step-rms-residual-inplace-v6`;
- timed fixed-25 in-place candidate:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-rms-residual-inplace-candidate-v3`;
- timed same-binary rollback:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-rms-residual-inplace-rollback-v2`;
- qualified A/B ReleaseFast binary SHA-256:
  `53e6bb1203ec07e7a0d475d7e73982dbd07bbfe98b93d37c85f42b641838ad5b`;
- final default-on smoke trace:
  `/private/tmp/antfly-gemma4-e2b-rms-residual-inplace-default-one-step-v7-20260818.log`;
- final default-on one-step artifacts:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-one-step-rms-residual-inplace-v7-default`.

### Matched E2B GRPO benchmark (2026-08-17)

Gemma4 text GRPO now scores its frozen same-base reference through the exact
compiled policy graph using reusable zero-valued LoRA device bindings. The
first real E2B probe selected ranked tokens `[7001, 711]`, produced prefix-match
rewards `[1, 0]`, and reported zero sampling/rescore and policy/reference error.
The previous eager-reference result had a false nonzero initial KL term; the
corrected cold loss, policy-gradient loss, and KL loss are all exactly zero
while the reward-derived gradient still updates every adapter tensor.

The locked rank-16 Q/V, sequence-128 diagnostic uses one rendered prompt, two
ranked one-token completions, AdamW at `1e-4`, KL coefficient `0.04`, and 25
complete groups split as cold 1, first 1, warmup 3, and measured 20. The
algorithmically matched MLX-LM runner uses the pinned MLX `0.31.2` and MLX-LM
`0.31.3` source revisions. It caches the immutable base prompt distribution
once and evaluates each completion at batch 1. A batch-2 MLX rescore changed
the second initial logprob by `0.124493`, so it was rejected rather than hidden
inside a nominally faster but semantically different baseline.

The optimized Antfly route now runs one shared ranked prompt forward, reuses
the sampled on-policy logprobs, caches exact frozen-reference rows, coalesces a
one-token completion group into one weighted gradient row, and projects only
the selected causal row. Five independent production-default runs produced
the same complete 25-point loss trajectory and byte-identical adapter payload
`a7c98538931e1fb20457343c491c287282b9f83d32b183aa4484cf9c3408178b`.

| Metric | Antfly Zig/Metal | Pinned MLX-LM |
| --- | ---: | ---: |
| Measured median per complete group | 0.773689 s | 0.629479 s |
| Measured mean per complete group | 0.777806 s | 0.629428 s |
| Peak process physical footprint | 1,871,007,776 bytes | 11,767,013,080 bytes |
| Adapter delta L2 | 0.782024 | 0.715478 |

The Antfly value is the median of five run medians; the five medians ranged
from `0.772952 s` to `0.774567 s`. It is `1.229x` the MLX-LM duration, or
`22.91%` slower. This is a `4.772x` speedup over the first matched Antfly
baseline (`3.692312 s`) without changing the locked workload. Antfly uses
`84.10%` less comparable peak process footprint, or MLX-LM uses `6.289x` as
much. The exact reference cache held two rows and served 48 hits; total
reference-scoring time fell from `22.722172 s` to `0.836700 s` for the full
25-group run.

The original in-frame private-buffer pool could recycle a Metal buffer under a
new logical tensor identity without a provable encoder dependency. Repeated
runs exposed exact stale losses from earlier updates. The production allocator
now reuses such a buffer only when an active planned encoder can emit a
buffer-scope barrier; otherwise the buffer remains quarantined until the frame
ends and the allocation is fresh. The eager sampling path also closes and
synchronizes its Metal frame at the logits readback boundary. This combination
preserved the low-memory result and passed five repeated trajectories; merely
staging gradients or adding a readback wait did not.

Cross-framework numerical parity is still not exact. The optimized Antfly and
MLX-LM initial second-token logprobs are `-8.051079` and `-7.940530`, an
absolute delta of `0.110549`; the ranked-margin delta is `0.110499`. Their
25-step loss trajectories have MAE `0.287661`, maximum absolute delta
`0.656068`, and final delta `0.327514`. Antfly's adapter-update L2 is `9.30%`
larger. This qualifies the command path, deterministic Antfly trajectory, and
bounded performance diagnostic; it does not establish broad GRPO quality or
cross-framework result parity.

The next optimization target is general multi-token GRPO: batch active
divergent prefixes by decode step, cache immutable reference rows by exact
prompt/prefix identity, and profile the remaining compiled backward path at
realistic sequence lengths before adding more specialized kernels.

Evidence:

- optimized compact comparison:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-mlx-comparison-v1.json`
- original pre-optimization comparison:
  `/private/tmp/antfly-gemma4-e2b-grpo-mlx-comparison-v1.json`
- Antfly five-run reports:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-20260817-v{1,2,3,4,5}/grpo_report.json`
- Antfly timed run:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-20260817-v1.time.txt`
- MLX-LM report:
  `/private/tmp/antfly-gemma4-e2b-grpo-mlx-repo-run-v2.json`
- rebuilt one-step cross-framework-logprob probe:
  `/private/tmp/antfly-gemma4-e2b-grpo-logprob-probe-20260817-v2/grpo_report.json`

The final five-run Antfly result is bound to binary
`sha256:932ab0d54dc0fa9826fc1c624485568a63827a6a13d51ab28018e530f64190e7`.
The working tree was intentionally dirty; no commit or push was performed.

### Real BoolQ GRPO acceptance, MLX parity, and stability boundary (2026-08-20)

`scripts/materialize_gemma4_grpo_boolq.py` pins `google/boolq` at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. It verifies the local tokenizer,
forbids rendered-prompt truncation, requires one-token `yes`/`no` targets, and
materializes balanced, source-disjoint 64-row train and validation artifacts.
Their JSONL SHA-256 digests are respectively
`36e2a2759413914466e7670583794e803b81b25d6f487faac1aad971f54fc3d7`
and `03ccbae22059e529e4abcd54977367ac8b64fd9f47feee06bf53ace20d9af7cc`.

ReleaseFast binary
`sha256:9b81aea87ff1e83e30ead95e3247f034d4e302ada63918f34ef720efff6f26a1`
completed the bounded cell through the public
`antfly inference finetune run <recipe.json>` command with no `TERMITE_*`
overrides. The cell uses the first eight pinned train rows, group size eight,
one epoch, learning rate `1e-7`, and all 64 held-out rows. It performed eight
optimizer updates, started with exact sampling/rescore and policy/reference
parity, published a changed adapter, and reported training mean reward
`0.328125` with mean KL `3.59379e-7`. Held-out mean reward was `0.36328125`,
top-ranked mean reward `0.703125`, positive-reward group rate `1.0`, and mean
KL `5.83758e-7`. Those values pass the unchanged floors `0.125`, `0.55`,
`0.75`, and maximum KL `1.0`.

The final run is byte-identical to the accepted 2026-08-19 trajectory. Its
adapter, train reward trace, and evaluation reward trace SHA-256 values are
respectively
`bd49818094089e7203e9844d528909e6c8f725c14e09d122e947990dfd5321b3`,
`990f206f2f58ae0034aa369bc8fa8b08837ac3ed087e8aef0b22a4aacc3b3d27`,
and `71685d49a5e039300567feb83d8fc3e2b74a7e4e8a7f9ebc483f327f509731fb`.
The held-out policy digest is
`sha256:42036e6ed505932978d5794db244917de810440b6eb8099a2dfe26d9a4836cd6`.
The complete final evidence root is
`/private/tmp/antfly-gemma4-e2b-boolq-grpo-direct-gather-acceptance-20260820-v13`.

The earlier intermittent Metal trajectory drift was traced to BF16 mmap
embedding-row staging across multiple runtimes in one optimizer-backed GRPO
process. GRPO now owns a nested process-scoped suppression guard and keeps
ordered planned encoders through the complete trainer lifetime. The remaining
fifth-evaluation-call failure was a separate ABA cache bug: compiled sparse
scoring sent ephemeral `[128,1536]` hidden activations through the generic
embedding-table cache, whose device-handle key could be recycled after the
activation was freed. Tensor-id embedding lookup now selects ordinary dense
device rows with the direct axis-0 Metal gather, bypassing the cache and a
`786,432`-byte activation copy per scorer call. The legacy device-table
preparation fallback also refuses pointer-only hits. A 12-lifetime real-Metal
regression passed without host fallback, and a cache-traced five-prompt run was
byte-identical to the canonical evaluation prefix while exposing only the two
stable mmap-backed model tables. The required-device ReleaseFast focused gate
selected 266 tests, passed 264, and skipped two optional local-model fixtures.

The final public run took `70.81 s`, with `569,163,776` bytes maximum RSS and
`3,465,826,216` bytes peak physical footprint. This is `4.378x` faster than
the previous exact Antfly acceptance and closes the prior `7.411x` MLX gap.
A fresh pinned MLX 0.31.2 / MLX-LM 0.31.3 comparison took `43.0698 s` and
peaked at `11,298,022,880` physical bytes. MLX is therefore `1.644x` faster in
this full bounded campaign, while Antfly uses `69.324%` less peak physical
memory. The comparator again classifies the result as **behavioral parity with
numerical drift**: all behavioral checks pass, including exact baseline mean
reward, `0.980469` baseline candidate recall, and `0.984375` top-1 agreement.
Native MLX held-out mean/top-ranked reward is `0.357422` / `0.6875`, within the
locked bounds. Numerical parity does not pass: adapter-delta cosine is
`0.572289`, relative L2 is `0.014554`, maximum absolute difference is
`1.58776e-6`, and held-out KL differs by `5.94119e-4`. The comparison artifact
is `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-parity-20260820-direct-gather-v3.json`.

This pass is deliberately bounded. Two larger, preserved campaigns exposed a
sharp stability boundary rather than being relabeled as successes: 192 updates
at `1e-5` and 64 updates at `1e-6` both failed the held-out gate, left the
trained adapter unpublished, and reported mean KL `1.62371584e9`. Their roots
are `/private/tmp/antfly-gemma4-e2b-boolq-grpo-acceptance-20260819-v5` and
`/private/tmp/antfly-gemma4-e2b-boolq-grpo-acceptance-20260819-v7`.
That historical acceptance predated the train-time KL budget and adaptive
controller qualified below. Those controls close the missing fail-safe, but a
multi-seed update-count sweep is still required: the bounded pass proves the
real dataset, optimizer, evaluator, reward trace, and publication contract,
not long-horizon convergence. That acceptance was one-token only; the bounded
multi-token E2B/E4B execution and MLX boundaries are qualified separately
below.

### Real E2B/E4B multi-token GRPO sparse-row qualification (2026-08-20)

The production Metal multi-token route now projects only the active causal row at
each ranked decode step and only the completion predictor rows during policy
and frozen-reference rescoring. Sampling and rescoring use the same projection
geometry under one default-on switch; this preserves the fail-closed on-policy
sampling/rescore check. Set
`ANTFLY_GEMMA4_GRPO_SPARSE_MULTI_TOKEN=0` to restore full sequence-vocabulary
projection for both phases. The optimization applies only when
`max_completion_tokens > 1` on Metal; native execution and the
already-qualified coalesced one-token route are unchanged.

This is deliberately not whole-transformer candidate batching. A first
candidate flattened the completion group into a padded transformer batch, but
real E2B Metal checks rejected it: mixed sparse rescoring differed from legacy
sampling by up to `0.002059937`, and padding projected rows to the legacy
sequence width still increased the maximum error to `0.010498047`. The
experimental batch switch remains default-off and retains the strict parity
failure. The promoted path keeps each divergent transformer prefix at batch
one while removing the dominant unused vocabulary rows.

Real pinned BoolQ cells used one train prompt, one source-disjoint evaluation
prompt, group size three, sequence length 128, rank-16 LoRA, learning rate
`1e-7`, and prefix-match reward. Both the train and held-out groups exercised
variable completion lengths `[4, 4, 3]` in the four-token cells.

| Workload | Legacy wall | Sparse wall | Maximum RSS, legacy -> sparse | Peak physical, legacy -> sparse |
| --- | ---: | ---: | ---: | ---: |
| E2B, two-token cap | `22.71 s` | `19.29 s` | `1,189,740,544 -> 568,836,096` bytes | `6,056,481,112 -> 5,517,381,448` bytes |
| E2B, four-token cap | `20.69 s` | `19.59 s` | `1,222,656,000 -> 573,947,904` bytes | `6,034,477,280 -> 5,524,983,624` bytes |
| E4B, four-token cap, median of two runs per mode | `37.14 s` | `34.52 s` | promotion-build same-binary `1,222,197,248 -> 632,438,784` bytes | promotion-build same-binary `7,693,342,384 -> 7,201,461,624` bytes |

The bounded E4B timing is noisy: the pre-promotion pair favored sparse
`31.70 s` to `40.91 s`, while the promotion-build same-binary
default/rollback pair was `37.33 s` to `33.36 s`. The two-run median is a
modest `7.1%` sparse win, not a stable throughput distribution. The memory
result was consistent and is the
stronger promotion signal: promotion-build RSS fell `48.3%` and peak physical
footprint fell `6.4%` versus rollback. E2B improved wall time by `15.1%` at the
two-token cap and `5.3%` at the four-token cap while cutting RSS by more than
half.

Every E2B and E4B cell retained exact-zero sampling/rescore and initial
policy/reference error within its selected geometry. Legacy and sparse modes
produced byte-identical train/evaluation reward traces and byte-identical
trained adapters. The E4B adapter SHA-256 is
`837ee12fad644025600aef1a39e5008d4426aa2281892e3baefd55878e6bb120`;
its train and evaluation trace SHA-256 values are respectively
`cfbc93de6eef294239e417f0d538f937ab12b71f6ba1e8fe4e3869e313277a4a`
and `e7d6d2d21af7db58821dd6fdbfb7502c4731eb54775ecda9f74cf013edcbffe7`.
The two projection geometries are not bit-identical internally: E4B diagnostic
first-token logprobs differ by at most `0.000261069`, but selected tokens,
rewards, loss, optimizer counts, adapter bytes, and replay traces are exact.

The final no-override ReleaseFast Metal CLI is SHA-256
`d866bce982a94e5dc7c5e3c614a928001534b6083c2e3b6daf7a79d549c7ed70`.
Its exact-source E4B run took `37.85 s`, used `625,639,424` bytes maximum
RSS and `7,194,482,016` bytes peak physical footprint, selected the qualified
sparse modes, and reproduced the accepted adapter and trace digests above.
Its focused Gemma4 gate selected `267` tests, passed `234`, and skipped `33`
optional hardware or local-fixture cases, with no failures. Final E4B evidence
is under
`/private/tmp/antfly-gemma4-e4b-boolq-grpo-four-token-final-20260820-v3`;
the promotion-build same-binary rollback is under
`/private/tmp/antfly-gemma4-e4b-boolq-grpo-four-token-rollback-20260820-v2`.
This closes the bounded multi-token projection boundary, not long-horizon
quality or MLX parity. The next performance target is true incremental KV
reuse and prompt/prefix-aware active-candidate scheduling, so transformer work
is shared rather than merely reducing LM-head projection rows.

### Matched multi-token E2B/E4B GRPO with train-time KL control (2026-08-20)

Gemma4 GRPO now applies a raw mean-token K3 budget before every optimizer
mutation. `grpo.train_max_kl` defaults to `0.1`; a non-finite, negative, or
larger observation writes a `budget-exceeded` record and aborts without
admitting that group. Setting `adaptive_kl = true` additionally requires a
positive `target_kl < train_max_kl` and `kl_horizon >= 1`. The bounded
proportional controller uses the current beta for the current group and updates
the coefficient for the next group, clamped to `min_kl_coef` / `max_kl_coef`
(defaults `0.001` / `1.0`). GRPO v4 train reports and v2 evaluation reports
carry raw `mean_kl`; the atomic `grpo_kl_control_trace.jsonl` binds every
observation, decision, optimizer-step count, and before/after coefficient.
The same admission path covers text and multimodal Gemma4 GRPO.

The matched campaign used pinned `google/boolq` revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The materializer selected 64
balanced train and 64 balanced, source-disjoint validation rows, forbade prompt
truncation at sequence length 128, required one-token `yes`/`no` targets, and
bound a four-token rollout cap. The materialization manifest SHA-256 is
`1b476974011cec1e725f2097b6d8581c9f0884f2e32e559763122681405ec7e3`;
train/eval JSONL SHA-256 values are
`01b35bec10abba5d540a29c0c4b0600c44bf54305d18bea0a7ea58d19b464057`
and `00b2c0fd7cacb639e2912a50e835b3a1d29c3d4dcad5b2fb13be83d582724f1e`.
Both E2B and E4B used the first eight train rows, 16 held-out rows, group size
four, rank-16/alpha-32 Q/V LoRA, learning rate `1e-7`, beta `0.04`, target KL
`0.01`, horizon `100`, and hard raw-KL budget `0.1` through the public
`antfly inference finetune run <recipe.json>` surface. The ReleaseFast binary
SHA-256 is
`e9bb7af988686e5b2730fc9c13af6b4bb67166a17e0ec4b4d0fb847453b3c57e`.

The pinned MLX comparator uses official MLX `0.31.2`, MLX-LM `0.31.3` at
commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, and runner SHA-256
`1a87f81e860204ae468d49d4c787a2c8a5b1f507671ade10321af6c141cf8062`.
It runs an exact Antfly completion/reward replay and an independent ranked MLX
rollout from the same seed adapter. An initial implementation batched divergent
MLX candidates and produced up to `0.359` sampling/rescore logprob error, so
those timings were rejected. The accepted comparator keeps sampling, policy
scoring, reference scoring, and differentiation at physical batch size one;
it token-normalizes and accumulates all four gradients, clips once, and makes
one optimizer update. All accepted E2B/E4B train lanes had exactly zero
sampling/rescore and differentiable-rescore error.

| Model | Antfly / MLX train seconds | Antfly / MLX eval seconds | Train / eval time ratio | Antfly / MLX peak physical | Antfly memory reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B | `100.231608 / 27.909717` | `113.877259 / 42.069351` | `3.591x / 2.707x` | `5,539,827,648 / 14,524,769,952` bytes | `61.86%` |
| E4B | `168.024695 / 50.238484` | `134.783294 / 66.170529` | `3.345x / 2.037x` | `7,201,281,352 / 20,408,988,776` bytes | `64.72%` |

Antfly admitted all eight groups. E2B's maximum raw train KL was
`4.68729e-5` and beta decreased to `0.03936446`; native MLX reached
`0.0315538` and beta increased to `0.04032064`. Held-out mean/top/positive-group
reward was exactly `0.34375 / 0.5625 / 1.0` in both implementations. E4B's
maximum was `1.12171e-5` with final Antfly beta `0.03936446`; native MLX
reached `0.01630496` with final beta `0.03952224`. E4B held-out mean and
positive-group reward matched at `0.234375 / 0.9375`; top-rank reward was weak
in both implementations (`0.0625` Antfly, `0.0` MLX). A first E4B run
correctly failed and withheld its adapter because a provisional absolute
top-rank floor of `0.25` was unsupported. The preserved v2 run kept nonzero
mean/positive reward and KL gates but treated top-rank reward as a parity
measurement; it reproduced the failed run's train, evaluation, and KL traces
byte-for-byte.

These are behavioral, not numerical-update, passes. Exact-trace adapter-delta
cosine was only `0.533616` for E2B and `0.598704` for E4B, despite relative L2
differences of `0.0316425` and `0.0241640`. Both comparison artifacts therefore
classify the result as `bounded-campaign-with-measured-drift` and explicitly set
`broad_grpo_performance_parity = false` and
`long_horizon_quality_parity = false`. Closing that boundary requires a
predeclared longer horizon, multiple seeds and tasks, baseline-relative quality
gates, and repeated same-Mac timing distributions; a single deterministic
BoolQ campaign is not sufficient.

Evidence:

- E2B Antfly root:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-adaptive-campaign-20260820-v1`
  (adapter SHA-256
  `d8c40074fa0a3e3d9784276c8ad19eea415377c8456f1904d05821eb6a59627c`);
- E2B MLX comparison:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-multitoken-adaptive-campaign-20260820-v1.json`
  (SHA-256
  `901166b34870576c2f5d6e5b9242107456bffb2160c826344b06a228eb951d27`);
- preserved fail-closed E4B quality-gate root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-adaptive-campaign-20260820-v1`;
- accepted E4B Antfly root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-adaptive-campaign-20260820-v2`
  (adapter SHA-256
  `6cd296df2eddd03fa87dfcc31d2666215844b08868b931a7d6c8cfa512999cc3`);
  and
- E4B MLX comparison:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-multitoken-adaptive-campaign-20260820-v1.json`
  (SHA-256
  `392d4603791e0ccf7bc2277040aa4cf9f1edf263b99e399267db6294a4fa3658`).

### Experimental incremental KV reuse and exact E2B/E4B gate (2026-08-20)

The sparse multi-token Metal sampler now has an opt-in paged-decode lane. Set
`ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV=1` to prefill each prompt once, share only
complete 16-token F32 KV pages with every candidate, replay each prompt tail,
and decode one token at a time with the live Q/V LoRA adapter. Ranked token
selection remains device-resident. Setting
`ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV_SHADOW_EXACT=1` additionally runs the
qualified full-prefix sampler for the first group and rejects any selected-token
or F32 logprob-bit drift.

Paged decode and the qualified fixed-shape graph use different reduction
orders. The incremental route therefore canonicalizes each sampled
completion's policy logprobs with one qualified sparse full-sequence rescore.
This preserves exact optimizer and artifact behavior while still proving that
token selection uses the shared KV cache. GRPO v5 train and v3 held-out reports
record canonical/tail prefill counts, decode forwards, exact rescoring,
device-resident ranked selections, host fallbacks, shared/reused prompt tokens,
page size, and cache dtype.

`validate_gemma4_grpo_incremental_kv_parity.py` compared each opt-in campaign
with its full-prefix baseline. Both gates passed exact hashes for the training
reward trace, held-out reward trace, KL-control trace, and final adapter:

| Model | Train / eval canonical prefills | Train / eval decode forwards | Train / eval exact rescoring | Host fallback | Final adapter SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| E2B | `8 / 16` | `88 / 176` | `32 / 64` | `0` | `d8c40074fa0a3e3d9784276c8ad19eea415377c8456f1904d05821eb6a59627c` |
| E4B | `8 / 16` | `69 / 123` | `32 / 64` | `0` | `6cd296df2eddd03fa87dfcc31d2666215844b08868b931a7d6c8cfa512999cc3` |

The exact gate is a correctness success but the first implementation is a
performance rejection. Fresh matched MLX `0.31.2` / MLX-LM `0.31.3` campaigns
used the same pinned BoolQ rows, adapters, eight train groups, 16 held-out
groups, group size four, and four-token cap:

| Model | Full-prefix Antfly train / eval | Incremental Antfly train / eval | Incremental slowdown | Fresh MLX train / eval | Incremental Antfly / MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B | `100.231608 / 113.877259 s` | `157.262167 / 215.813487 s` | `1.569x / 1.895x` | `27.769290 / 38.714746 s` | `5.663x / 5.574x` |
| E4B | `168.024695 / 134.783294 s` | `256.107200 / 278.369770 s` | `1.524x / 2.065x` | `51.297557 / 62.883978 s` | `4.993x / 4.427x` |

Sampling alone regressed `2.260x / 2.228x` for E2B train/eval and
`2.525x / 2.584x` for E4B. MLX timing moved only by single-digit percentages
versus the prior campaign, so the gap is not reference noise. The dominant
boundary is serial eager one-token decoder dispatch with live LoRA plus one exact full-sequence
rescore per completion. Incremental KV reuse remains default-off. Promotion
requires prompt/prefix-bucketed active-candidate batching or a LoRA-aware
compiled paged decoder, removal of canonical rescoring only after logprobs are
bit-exact, the same exact E2B/E4B artifact gate, and a measured win over the
full-prefix rollback. The previously rejected padded whole-transformer batch
is not an acceptable shortcut.

Evidence:

- E2B incremental root:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-incremental-kv-20260820-v6`;
- E2B fresh MLX comparison:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-incremental-reprofile-20260820-v1.json`
  (SHA-256
  `302608afceea524917a1cd8acf1cfeaa1d9cb4bc9c7706c663064ab25eb2857d`);
- E4B incremental root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-incremental-kv-20260820-v1`; and
- E4B fresh MLX comparison:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-incremental-reprofile-20260820-v1.json`
  (SHA-256
  `d2dc9636a5f7f345d4de191fd051d67e3a8786090f463679210b15f606edb570`).

### Segmented prompt prefill and Unsloth transfer audit (2026-08-20)

The incremental sampler now copies each candidate's non-page-aligned prompt
tail from one canonical prefill instead of replaying that tail independently.
This is an exact optimization: the E2B and E4B campaigns reproduced the
accepted train reward trace, held-out reward trace, KL-control trace, and final
adapter byte-for-byte. Relative to the first active-candidate implementation,
train/eval sampling improved from `82.286618 / 158.290720 s` to
`74.673257 / 140.767838 s` on E2B (`9.3% / 11.1%`) and from
`124.513679 / 210.227072 s` to `104.816143 / 182.845191 s` on E4B
(`15.8% / 13.0%`). Peak physical footprint was `5.956 GB` and `8.063 GB`.

The same-binary full-prefix sampler is still faster: segmented incremental
sampling is `1.656x / 1.692x` slower on E2B train/eval and
`1.846x / 2.041x` slower on E4B. Fresh pinned MLX native rollouts took
`28.940351 / 42.193658 s` for E2B train/eval and
`51.768443 / 66.379926 s` for E4B. Comparing complete accounted Antfly train
and evaluation loops, not just the optimized sampling phase, leaves gaps of
approximately `4.50x / 4.06x` on E2B and `4.20x / 3.45x` on E4B. Antfly used
about `58.9%` and `59.8%` less peak physical memory than MLX, respectively.

Unsloth's [padding-free and explicit sequence-packing
work](https://unsloth.ai/docs/new/3x-faster-training-packing) identifies the
next high-value direction, but the existing Antfly `sequence_packing.zig`
utility is not wired into Gemma training. It already emits reset `position_ids`,
`doc_ids`, `cu_seqlens`, and a block-diagonal causal mask; however,
`GemmaAutodiffCtx.buildForward` currently ignores its `attention_mask`, and
`gemma_graph.zig` creates one fixed causal/sliding mask and global RoPE
positions. Concatenating examples now would therefore allow cross-example
attention and apply incorrect positions. It is not an admissible optimization.

The production implementation should use a segment-aware/ragged GQA graph:

1. start with length-bucketed independent batch rows to remove right-padding
   work without changing attention semantics;
2. add `cu_seqlens`-driven causal/sliding attention and per-segment RoPE reset,
   without materializing a quadratic block mask;
3. map sparse supervised rows through the packed layout for LoRA-SFT, preserve
   independent chosen/rejected units for DPO, and preserve independent
   completions plus shared-prompt accounting for GRPO;
4. retain sample order, optimizer boundaries, and stable reductions so the
   existing E2B/E4B trace and adapter-hash gates remain meaningful; and
5. promote only after both models beat the unpacked rollback with bounded peak
   physical memory.

An opt-in intermediate experiment batches independent GRPO completion rows for
backward only. Set
`ANTFLY_GEMMA4_GRPO_BATCH_MULTI_TOKEN_BACKWARD=1`; cap the physical batch with
`ANTFLY_GEMMA4_GRPO_MULTI_TOKEN_BACKWARD_MAX_BATCH` (default `4`). GRPO v5
reports record the physical batch size and physical micro-batches per group.
This lane is deliberately default-off:

| Model/lane | Backward/update | Gain vs serial | Peak physical | Exact adapter gate |
| --- | ---: | ---: | ---: | --- |
| E2B serial | `4.827423 s` | - | `5.611 GB` | canonical |
| E2B batch 2 | `3.067223 s` | `36.5%` | `8.587 GB` | fail, max abs `1.993e-7` |
| E2B batch 4 | `2.164943 s` | `55.2%` | `14.029 GB` | fail, max abs `1.996e-7` |
| E4B serial | `10.930324 s` | - | accepted campaign `8.063 GB` | canonical |
| E4B batch 2 | `7.817791 s` | `28.5%` | `12.076 GB` | fail, max abs `1.986e-7` |

The E2B one-group wall times were effectively flat (`32.42-32.55 s`), so the
faster backward did not yet improve end-to-end latency. The numerical drift is
small and expected from a changed floating-point reduction order, but it is
still outside the exact promotion contract. The lane remains useful for a
future predeclared tolerance-based quality campaign; it is not a production
default.

Unsloth's [long-context activation
offload](https://unsloth.ai/blog/long-context) is also not a current
sequence-128 Metal target. Apple Silicon uses unified physical memory, so
moving saved activations to CPU address space does not free a separate VRAM
pool and may add copies or synchronization. Reconsider it only for profiled
long-context runs (at least sequence 512) that fail a physical-memory gate.
Likewise, the [Unsloth GRPO memory
optimization](https://www.unsloth.ai/blog/grpo) that avoids retaining full
generation-by-token-by-vocabulary logits is largely already represented here
by sparse completion-row rescoring and fused/cut cross-entropy; current
profiles point to decoder dispatch, canonical rescoring, and backward rather
than retained full-vocab logits.

Evidence:

- exact E2B segmented campaign:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-kv-segmented-canonical-campaign-20260820-v1`;
- exact E4B segmented campaign:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-kv-segmented-canonical-campaign-20260820-v1`;
- fresh pinned E2B/E4B MLX comparisons:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-segmented-canonical-reprofile-20260820-v1.json`
  and
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-segmented-canonical-reprofile-20260820-v1.json`;
- E2B serial, batch-2, and batch-4 probes:
  `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-serial-control-20260820-v1`,
  `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-batch2-probe-20260820-v1`,
  and `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-probe-20260820-v3`;
  and
- E4B batch-2 probe:
  `/private/tmp/antfly-gemma4-e4b-grpo-group-backward-batch2-probe-20260820-v2`.

### Opt-in independent-row length buckets (2026-08-21)

Text LoRA-SFT can now avoid executing every example at the prepared artifact's
maximum sequence length. The default remains the established fixed-shape path.
Enable the new policy explicitly:

```sh
antfly inference finetune train gemma4-lora \
  --model "$BASE" --adapter "$ADAPTER" \
  --train-prepared "$TRAIN_PREPARED" --eval-prepared "$EVAL_PREPARED" \
  --out "$OUT" --backend metal \
  --sequence-length-bucket-quantum 16 \
  --sequence-length-bucket-min 32 \
  --graph-cache-capacity 4
```

Recipes expose the same policy under `runtime`:

```json
{
  "runtime": {
    "sequence_length_bucket_quantum": 16,
    "sequence_length_bucket_min": 32,
    "graph_cache_capacity": 4
  }
}
```

Each example remains an independent causal row and keeps its original sample,
optimizer, attention, and RoPE order. The scheduler only rounds its logical
token count upward to a bounded graph shape; it never concatenates examples or
truncates logical tokens. A deterministic bounded graph cache retains the
resulting signatures. Run fingerprints bind the bucket policy and effective
cache capacity. The unchanged fixed policy retains the prior v2 fingerprint
domain so compatible fixed-shape checkpoints are not invalidated by this
feature.

Reports record logical, scheduled, and fixed-baseline rows; avoided padding;
bucketed-example and min/max-shape counts; and graph cache builds, hits, active
reuses, evictions, and peak residency. `epoch_history` additionally records
`antfly.gemma4.epoch-timing/v1` monotonic wall time and example, logical-token,
scheduled-token, supervised-token, and optimizer-update rates, plus the cache
delta for that epoch. `phase_timing` uses
`antfly.gemma4.phase-timing/v1` to separate initialization/restore, initial
evaluation, total epoch training, adapter save, and final evaluation. The
timers rely on the trainer step's existing completion contract and add no GPU
synchronization. Bucket and cache flags are rejected by the locked benchmark
contract. A cache capacity without a bucket quantum is also rejected. Public
multimodal training fails closed when bucketing is requested.

The first real-data systems campaign used the pinned 64-train/64-eval
`google/boolq` materialization at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`, with four train rows and two eval
rows at prepared maximum 512. The source's `target` field was mechanically
renamed to the SFT loader's `response` field. Because the materialization's
prompts were already rendered for the GRPO campaign and were rendered again by
the SFT loader, this proves real-data execution and performance behavior, not a
canonical BoolQ quality-training result.

The four train rows contained `493` logical tokens. Quantum-16 scheduling used
`544` transformer rows per epoch instead of `2048`, avoiding `1504` rows
(`73.44%`). Both models built three signatures, reused them without eviction,
made eight strict-Metal optimizer updates, and reported zero fallback steps.

| Model | Fixed wall / peak physical | Bucketed wall / peak physical | Update parity from common input |
| --- | ---: | ---: | --- |
| E2B | `22.08 s / 6.732 GB`; reverse-order repeat `19.64 s / 6.732 GB` | `20.46 s / 2.113 GB`; repeat `32.66 s / 2.110 GB` | cosine `0.9999804`; relative L2 `0.0062656`; magnitude delta `0.0201%` |
| E4B | `72.49 s / 11.400 GB` | `42.12 s / 3.472 GB` | cosine `0.9999438`; relative L2 `0.0106013`; magnitude delta `0.00665%` |

The E4B pair improved end-to-end wall time by `41.9%` and peak physical memory
by `69.5%`. E2B peak physical memory improved by about `68.6%`, but its wall
time is inconclusive: the repeat suffered hundreds of thousands of macOS page
faults and reversed the first pair's small apparent gain. Do not use these
full-process E2B timings as a throughput claim. Darwin maximum RSS remained
large for both policies because it includes the mapped base artifact; peak
physical footprint is the relevant working-set measurement.

The fixed and bucketed E2B runs were each byte-reproducible across two fresh
processes (adapter SHA-256
`d962d9055a1c3d8e59808080616b3cb8ba00d0219590c530025f134ef9c14339`
and `3e27b97b335140bbccab51a841e3e88006d564d82b91052941f9b3c7d7eaf370`,
respectively). Fixed versus bucketed updates are not
bit-exact because the changed BF16/Metal shapes change floating-point reduction
order. The measured update cosines and magnitudes support a production-safe
opt-in, not a silent default change or exact-trajectory claim.

The larger steady-state campaign used 16 train rows, four disjoint evaluation
rows, three epochs, 48 strict-Metal optimizer updates, learning rate `1e-4`,
and seed `42`. The 20 rows span 18 distinct logical lengths but only four Q16
shapes (`96`, `112`, `128`, and `144`); every completion has seven supervised
tokens. Epochs two and three are the steady-state gate. Both bucketed arms
reported `reuse_only = true`, four resident signatures, zero builds, and zero
evictions in those epochs. Each policy avoided fallback for all updates.

| Model | Fixed steady epoch / logical tok/s | Q16 steady epoch / logical tok/s | Through-final-eval wall | Peak physical footprint | Update parity from common input |
| --- | ---: | ---: | ---: | ---: | --- |
| E2B | `18.796 s / 102.15` | `19.384 s / 99.05` | fixed `68.688 s`; Q16 `83.386 s` | fixed `6.732 GB`; Q16 `2.142 GB` | cosine `0.9999901`; relative L2 `0.0044516`; max abs `5.7765e-4` |
| E4B | `72.287 s / 26.56` | `37.114 s / 51.73` | fixed `257.486 s`; Q16 `140.876 s` | fixed `11.379 GB`; Q16 `3.525 GB` | cosine `0.9999829`; relative L2 `0.0058628`; max abs `7.5147e-4` |

Q16 removes `75.20%` of scheduled training rows in both pairs. On E4B that
becomes a `1.948x` steady-state speedup, a `1.828x` through-final-eval speedup,
and a `69.02%` footprint reduction. On E2B it saves `68.18%` of peak physical
footprint but is `3.13%` slower per steady epoch and `21.40%` slower through
final evaluation. The E2B result is now conclusive rather than page-fault
noise: at these short shapes, per-example launch/optimizer cost and lower GPU
occupancy outweigh avoided padded rows. Fixed shape therefore remains the
cross-model default. Q16 is the qualified E4B throughput/memory policy for
similarly heterogeneous short rows, but should stay explicit until broader
length distributions establish an automatic selection threshold.

The held-out results also agree: fixed/Q16 loss is
`0.2253238`/`0.2253264` for E2B and `0.00365768`/`0.00365325` for E4B. Shape
dependent BF16 reductions prevent byte identity, so the cosine/relative-L2
gate remains the correct numerical contract.

The final instrumented ReleaseFast Metal binary is 29,632,896 bytes with
SHA-256
`8654d156e2bc868757dd90362fab515aa757378ca7066149734f64f63f1e6e45`.
It produced all four steady-state roots above. The full focused ReleaseSafe
Metal gate selected 275 tests: 273 passed and two optional real-artifact tests
skipped.

Steady-state evidence:

- consolidated campaign artifact:
  `/private/tmp/antfly-gemma4-boolq-sft-steady16-length-bucket-campaign-20260821-v1.json`
  (SHA-256
  `bbc6ac638fbeca623521766cd29428c29f9e664b9c80fe9482c8dfbe214118d9`);
- prepared E2B train/eval artifacts:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-20260821-train-prepared.json`
  and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-20260821-eval-prepared.json`;
- prepared E4B train/eval artifacts use the corresponding `e4b` paths;
- E2B fixed and Q16 roots:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-fixed-20260821-v1` and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-bucket16-20260821-v1`;
  and
- E4B fixed and Q16 roots use the corresponding `e4b` paths.

Initial evidence:

- transformed real-data inputs:
  `/private/tmp/antfly-gemma4-boolq-sft-length-buckets-20260821-train.jsonl`
  and
  `/private/tmp/antfly-gemma4-boolq-sft-length-buckets-20260821-eval.jsonl`;
- E2B fixed and bucketed roots:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-fixed-20260821-v1` and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-bucket16-20260821-v1`; reverse-order
  repeats use the corresponding `v2` roots;
- E4B fixed root and adapter SHA-256:
  `/private/tmp/antfly-gemma4-e4b-boolq-sft-fixed-20260821-v1`,
  `932c45051efafc91b852a648a1d083921a006f042e77f98333419ba45b8c1e7e`;
  and
- E4B bucketed root and adapter SHA-256:
  `/private/tmp/antfly-gemma4-e4b-boolq-sft-bucket16-20260821-v1`,
  `5eb178e1c9fba0c3834879eabc8e2fe49dbdde7a634b7a7a81e0d44948fe6faf`.

The independent-row policy remains SFT-specific, but DPO now has a separate
pair-safe scheduler. One rounded shape is computed from the maximum logical
chosen/rejected row; reference precompute, both policy branches, backward, and
held-out evaluation share that sequence length and the pair's maximum
weighted-target bucket. Recipe/report provenance binds the quantum, minimum,
cache capacity, scheduled rows, graph signatures, and phase snapshots. The
bucketed route defaults to the qualified four-graph cache; eight is an explicit
upper bound, not the implicit policy. Every distinct bucket graph signature is
checked for zero-adapter policy/reference parity before optimizer mutation.
Fixed padding remains the default and rollback. GRPO still requires its own
group-safe policy that preserves completion-group boundaries, shared-prompt
accounting, ranked selection, and canonical rescore semantics. True
padding-free packing remains the longer-term target: `cu_seqlens`-driven
causal/sliding attention, per-segment RoPE reset, and packed sparse-target row
mapping without a quadratic block mask.

### Pair-safe DPO buckets and matched MLX reprofile (2026-08-21)

The final campaign used `HuggingFaceH4/ultrafeedback_binarized` at revision
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` (source parquet SHA-256
`e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389`).
Both frameworks trained the same five pairs for five epochs: 25 optimizer
updates, sequence maximum 512, beta `0.1`, AdamW learning rate `1e-4`, and
rank-16/alpha-32 Q/V adapters. Q128/min256 scheduled pair shapes
`[384, 384, 512, 256, 256]` instead of five fixed 512-row shapes, reducing
executed branch rows from `5120` to `3584` (`30%`). Antfly used cache capacity
four, which held all four training signatures after one bootstrap eviction.

Each performance row below is one fresh process. Times are the median and mean
of the 20 measured updates after the locked cold/first/three-warmup sequence;
memory is macOS peak physical footprint. This is a same-Mac matched reprofile,
not yet the five-process alternating distribution gate.

| Model / policy | Antfly median / mean | MLX-LM median / mean | Antfly / MLX median | Antfly peak | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B fixed 512 | `2.4951 / 2.8161 s` | `2.1114 / 2.1124 s` | `1.182x` | `8.572 GB` | `18.945 GB` |
| E2B Q128/min256 | `1.8727 / 2.0616 s` | `1.5354 / 1.4590 s` | `1.220x` | `8.258 GB` | `18.841 GB` |
| E4B fixed 512 | `32.9120 / 38.0635 s` | `37.3854 / 38.4071 s` | `0.880x` | `21.573 GB` | `26.390 GB` |
| E4B Q128/min256 | `26.6671 / 34.3067 s` | `15.3322 / 16.3493 s` | `1.739x` | `26.629 GB` | `26.366 GB` |

Within Antfly, Q128 improved median time by `1.332x` on E2B and `1.234x` on
E4B. It reduced E2B peak memory by `3.66%`, but increased E4B peak by `23.44%`
because four specialized sequence executables remain resident. Fixed E4B is
the strongest current result: Antfly is `11.97%` faster by median, effectively
tied by mean, and uses `18.26%` less peak memory than MLX-LM. The optimized
E4B boundary is not parity: MLX-LM Q128 is `1.739x/2.098x` faster by
median/mean with effectively equal memory. MLX converts the same row reduction
into a `2.438x` median speedup while Antfly retains substantial per-shape and
per-branch overhead.

Quality was scored under one pinned MLX oracle on source rows disjoint from
training. E2B Q128 Antfly and MLX agreed on all five decisions and both reached
`0.60` accuracy. The expanded E4B gate used 25 unique rows with zero training
overlap: Antfly/MLX Q128 agreement was `22/25` (`0.88`) with `0.60`/`0.64`
accuracy; fixed agreement was also `0.88`. Fixed-versus-Q128 agreement within
each framework was `0.96`. This passes a bounded behavioral gate but fails
exact decision parity and does not establish broad, multi-seed, or long-horizon
quality parity. Shape-dependent BF16 reductions also make fixed and bucketed
adapter bytes intentionally non-identical. Fixed scheduling therefore remains
the default and rollback; Q128/min256 remains an explicit throughput policy.

The tested ReleaseFast CLI SHA-256 is
`63bd3647bfb8c6c1c9cddc4bb1a9f091b291e6abde7aba271ccdd1b8843c5bf2`.
The bounded diagnostic comparison used MLX `0.31.2` at
`68cf2fddd8de5edd8ab3d926391772b2e2cedad8` and MLX-LM `0.31.3` at
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`; its recovered SDK 26.2 Metal
library was hash-bound in the campaign. That archived runner predates the v3
closed-native-runtime contract: it did not bind the loaded Python extension,
both runtime dylibs, package inventory, and native-build receipt as one
postflight-rechecked bundle. Its measurements remain useful same-machine
diagnostics, but are not promotable cross-framework release evidence. New DPO
MLX training and shared-oracle evaluation runs require the same strict
`--mlx-build-attestation` used by the LoRA oracle runner. Consolidated evidence is at
`/private/tmp/antfly-gemma4-dpo-length-buckets-20260821-v1/campaign.json`
(SHA-256
`541ea5ab5cbf66d16618cf987f9f1e56e89269d53d03db8e031e486326dfc321`).
The expanded E4B quality summary is beside it as
`e4b-holdout-25-summary.json` (SHA-256
`8b6cd22bb905e8db532c7106c45d174fed44708ed29f86b2b7b590d321e94b73`).

The next performance target is a memory-safe compiled whole-pair DPO objective
or segment-aware packed chosen/rejected graph. It must preserve pair-shared
attention/RoPE/target semantics and optimizer order while amortizing the two
branch passes and short-shape launch overhead. More bucket-quantum sweeps are
unlikely to close the optimized E4B gap: the schedule already removes 30% of
rows, while Antfly's four graph signatures and branch execution dominate the
remaining difference.

The post-campaign production review emits DPO report v5. It replaced quadratic
graph-signature and train/eval-overlap scans with hash-indexed linear passes,
made cache capacity four the bucketed default, added the pre-update parity gate
for every distinct bucket signature, and made DPO MLX training report v3 and
shared-oracle evaluation v2 verify and postflight-recheck the full model/input
surface, installed package versions, clean source revisions, loaded native
runtime, and strict build attestation.
These changes do not alter the explicit cache-four Antfly training schedule
used for the measured campaign; its binary hash remains the immutable Antfly
performance provenance above.

The production-review source snapshot was rebuilt as the shipping
`antfly` CLI (77,181,608 bytes, SHA-256
`b93433368ab36a53eb7b607be453e2f981a9d9fe15ef650da7c2741a5a87ca05`).
Its real-device ReleaseSafe gate selected 276 tests: 274 passed and the two
optional fixture-dependent tests skipped. Fresh CLI acceptances then trained
one epoch over the same five pinned UltraFeedback pairs on both E2B and E4B.
Each emitted report v5, executed five optimizer updates, admitted the implicit
four-graph cache, checked all four observed graph signatures with exactly zero
initial policy/reference error, published a changed adapter, and passed a
five-row held-out evaluation with zero prompt overlap. Independent adapter
validation found 50 E2B and 66 E4B Q/V target modules at rank 16/alpha 32. The
immutable roots are
`/private/tmp/antfly-gemma4-production-review-e2b-dpo-v5-20260821-v1` and
`/private/tmp/antfly-gemma4-production-review-e4b-dpo-v5-20260821-v1`.
These are mechanics and artifact acceptances; their deliberately permissive
smoke thresholds do not replace the longer campaign's bounded quality result.

## Preference Recovery and Terminal Evaluation Boundary (2026-08-21)

DPO and GRPO report v6 close the epoch-boundary recovery contract that report
v5 left implicit. A preference checkpoint is admitted only with a matching
`antfly_gemma4_preference_checkpoint_state/v1` content-addressed sidecar. The
sidecar restores exact DPO/GRPO aggregates; GRPO additionally restores reward
trace state, raw and weighted KL totals, adaptive-KL controller state, and the
initial diagnostic prefix. Publication orders sidecar-before-checkpoint, so a
visible checkpoint cannot name unpublished aggregate state.

The preference run fingerprint is v5. Besides model, adapter, train/eval data,
optimizer, reward, and graph/scheduling controls, it hashes the resolved
`antfly_gemma4_metal_numerical_policy/v2` contract: sparse-loss and CCE tile
geometry, BF16 kernels, eager LoRA/dot/quant routes, and graph-executor fusion
routes. A change to any covered route therefore rejects resume before optimizer
mutation. Unknown `TERMITE_METAL_*`, `TERMITE_GEMMA4_*`, or Gemma preference
environment controls fail closed until they are explicitly reviewed and added
to this attestation boundary. Admitted boolean values are canonical `0`/`1`;
presence-only kill switches accept only `1`. The fingerprint also binds DPO
activation-checkpoint interval/recursion and the exact GRPO backward batch size.
Checkpointed GRPO rejects custom reward trace/exchange paths rather than sharing
mutable files across artifact roots. The main report records the same policy
and the checkpoint sidecar path/digest.

The product and both preference qualifiers now share
`src/finetune/gemma4_preference_environment.policy` as the single typed source
for environment scope, admission, sanitization, and strict bindings. The Zig
binary embeds it; Python qualification reads it directly and records its
SHA-256. Qualification strips every inherited `TERMITE_*` and
`ANTFLY_GEMMA4_*` override before installing the strict executor bindings.
Product planning rejects unreviewed semantic/debug controls and explicitly
denies graph-output ownership/elision, output host-mirror resync, and paged-KV
kill switches before printing a plan or publishing any run artifact.

Held-out preference evaluation no longer inherits the training allocation
history. The source trainer drains Metal, synchronizes the final trainables to
host, retires compiled graphs and device optimizer slots, and becomes
terminal-only. A separately initialized backend receives the exact host
snapshot and runs evaluation with in-frame and completion-cache private-buffer
reuse disabled. Reports attest the policy string
`terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled`;
any attempted optimizer step after that boundary fails.

`scripts/qualify_gemma4_preference_resume.py` schema v2 runs the same pinned
recipe uninterrupted, kills a second process only after an epoch-1 checkpoint
and sidecar are durable, then resumes into a new immutable output root. It
requires byte-identical adapters, exact final sidecars, training metrics,
completion/reward/KL-control traces, discrete evaluation behavior, and Metal
policy. It also opens the standalone evaluation reports and all three GRPO
trace artifacts, verifies their paths and content digests, and compares their
semantic contents. It rejects symlink leaves and output/input overlap, validates
the complete adapter tree plus manifest, preserves every immutable seed
companion, and requires a changed tensor payload. DPO terminal metrics are
exact. Repeated E2B evaluation-only replays
from the same byte-identical checkpoint showed fresh-process variation only in
GRPO's terminal Metal KL reductions, so the qualifier removes no other field
and permits only:

- `evaluation.kl_loss`: absolute delta at most `1e-6`;
- `evaluation.mean_kl`: absolute delta at most `1e-5`.

The standalone evaluation report's derived `loss` uses a `1e-6` bound because
it includes weighted KL; `pg_loss` remains exact. All observed values and
bounds are emitted in the qualification report. A value outside a bound, any
adapter/checkpoint/sidecar/trace/discrete drift, a missing artifact, or a
missing numerical-policy attestation still fails closed.

The 2026-08-21 recovery-boundary source snapshot was rebuilt as a
29,795,824-byte ReleaseFast Metal
binary with SHA-256
`b6b8cd957cee34058ffd0cae3a1a9ca0794ba8ad7562647253775b1bb0540304`.
The real-device focused gate selected 279 tests: 277 passed and the two optional
fixture-dependent tests skipped. All four same-binary v5-fingerprint/v2-policy
SIGTERM/resume campaigns then passed:

| Task / model | Recovery comparison | Adapter SHA-256 | Held-out result |
| --- | --- | --- | --- |
| DPO E2B | exact checkpoint, sidecar, adapter tree, metrics, and evaluation | `3e690f8bff46af295028328d46fcdaee6f484f04cf57d82ce2feafd5005555bd` | loss `0.68912619`, accuracy `0.60`, passed |
| DPO E4B | exact checkpoint, sidecar, adapter tree, metrics, and evaluation | `657bbd8f89b738cdd6453e147955eb099da53e6b25ad3778fabb2c324071df78` | loss `0.64424115`, accuracy `0.80`, passed |
| GRPO E2B | exact training/discrete trajectory; terminal `kl_loss` / `mean_kl` deltas `1.13e-7 / 2.81e-6` | `43e6d637a316b38247737c5e3879843e5576b95ed1c81655850f91962666a787` | mean/top-rank reward `0.34375 / 0.5625`, passed |
| GRPO E4B | exact training/discrete trajectory; terminal `kl_loss` / `mean_kl` deltas `9.59e-9 / 2.40e-7` | `0efbf1ad76f9e145382c18aaa24813d081220e8ceb373988bf33b0967e92a8c7` | mean/top-rank reward `0.234375 / 0.0625`, passed |

The reports for that recovery snapshot are:

- `/private/tmp/antfly-gemma4-e2b-dpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-dpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e2b-grpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-grpo-preference-resume-acceptance-20260821-v11/qualification_report.json`.

The pre-v5/pre-policy-v2 reports are retained at:

- `/private/tmp/antfly-gemma4-e2b-grpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-grpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e2b-dpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-dpo-preference-resume-acceptance-20260821-v4/qualification_report.json`.

The first E4B GRPO qualifier invocation failed safely because an older seed
directory had a tensor checkpoint but no `adapter_config.json`. The accepted
campaign uses
`/private/tmp/antfly-gemma4-dpo-length-buckets-20260821-v1/e4b-seed-qv-r16-a32`,
whose adapter tensor SHA-256 is identical
(`44c328721be136ce6795d7870109ca704a04b7094e8fc9553947620e6a30ddc5`)
and whose complete
rank-16/alpha-32 Q/V contract names the pinned E4B base. The rejected root was
preserved rather than repaired in place.

The v11 artifacts close the recovery and terminal-evaluation boundary for the
exercised short E2B/E4B DPO/GRPO campaigns. They do not establish broad
long-horizon quality or distribution-level performance parity. The subsequent
checkpoint contract admits incremental-KV GRPO only at a drained,
zero-live-sequence epoch boundary and restores cumulative telemetry while
rebuilding transient pages. Canonical full-prefix direct-GGUF E2B DPO/GRPO is
an explicitly gated research lane; combining that Q4_0 base with incremental
KV remains fail-closed because the paged decoder changed the canonical token
trajectory.

## Final Preference PR Qualification (2026-08-22)

The audited source was rebuilt once as a 29,831,008-byte ReleaseFast Metal
binary with SHA-256
`6e0dde73f07fd7bbbb7ef5d953979580c52fd9d7b91a1d9fe323b012965d3402`.
All qualification below uses that exact binary. The post-documentation local
mirror of the required PR workflow ran the same 280-test finetuning root in
Debug, ReleaseSafe, and ReleaseFast; each passed 278, skipped only the two
optional fixture-dependent tests, and failed none. The Debug and ReleaseSafe
Gemma graph roots passed `7/7`, the filtered ownership/lifecycle gate passed
`4/4`, the pinned Python 3.12 suite passed `593/593`, and deterministic fixture
regeneration passed. Hosted required CI remains pending until the branch is
pushed; documentation changes do not alter the qualified executable.

The long-horizon harness used seeds `17`, `42`, and `991`, eight epochs, strict
Metal, and a fixed initialized adapter. Each seed receives a deterministic
permutation of the same immutable row multiset. DPO runs use five pinned
UltraFeedback pairs per epoch and a disjoint five-pair holdout; GRPO runs use
eight pinned BoolQ prompts per epoch, group size four, a four-token completion
cap, and 16 disjoint held-out groups. The training and held-out dataset
SHA-256 values are respectively
`45d12cce115dfd0b9ec40b1e0c98d7b805f56adee559467b940762e9b8240f2f` /
`29a34e7d08e045fb50eebcf441a29c02b909d50eb33d112027141bc9dfb0a1de`
for UltraFeedback and
`01b35bec10abba5d540a29c0c4b0600c44bf54305d18bea0a7ea58d19b464057` /
`00b2c0fd7cacb639e2912a50e835b3a1d29c3d4dcad5b2fb13be83d582724f1e`
for BoolQ.

| Campaign | Result | Optimizer horizon | Held-out evidence |
| --- | --- | ---: | --- |
| E2B DPO | **PASS**, three distinct adapter digests | `3 x 40` updates | accuracy `0.60-0.80` (mean `0.6667`), loss `0.5888-0.7921`, positive margin `0.2648-0.8901` |
| E4B DPO | **PASS**, three distinct adapter digests | `3 x 40` updates | accuracy `0.60-0.80` (mean `0.7333`), loss `0.5939-0.7355`, positive margin `0.2986-0.4979` |
| E2B GRPO | **PASS**, three distinct adapter digests and zero host-logit fallback | `3 x 64` updates | mean/top-rank reward `0.34375 / 0.5625`, positive-group rate `1.0`, KL loss `2.73e-6-3.43e-6` |
| E4B GRPO | **FAIL** at seed 17; seeds 42/991 not run after fail-fast | `64` updates completed | mean reward `0.234375` and positive-group rate `0.9375` pass, KL loss `1.16e-6` passes, but top-rank reward `0.0625` misses the predeclared `0.125` floor |

The passing campaign reports and SHA-256 values are:

- `campaigns/e2b-dpo/campaign_report.json`:
  `e2424c14e4c690ad5c2740369ea27454a7c37b0e99a7c9d92a87d513f8aa6247`;
- `campaigns/e4b-dpo/campaign_report.json`:
  `c5ef75f4fac2b82c357aaa540fc195260cf944a9a9bbfff59bb98909ca893c09`;
- `campaigns/e2b-grpo/campaign_report.json`:
  `7f3de930dbd833b6af2de172e35268bfa63b35c72264d21c85fd143dff241532`.

The preserved E4B GRPO failure report is
`campaigns/e4b-grpo/campaign_report.json` at SHA-256
`84a8f27b2b40df2fdc037edc44b72dcc2b79a88bde87869007252b5c4d207668`;
its completed seed-17 adapter is
`39ed6a3aa48c59511324fe420c15cd05dde8826f8234754471b51c703f315684`.
All paths above are relative to the immutable evidence root
`/private/tmp/antfly-gemma4-prready-20260822-v1`. The E4B run completed cleanly
and changed the adapter, but its training reward repeated exactly across all
eight epochs and only one of 16 held-out top-ranked candidates was correct.
The campaign floor was not weakened after observing this result. E4B GRPO
therefore remains a quality blocker, not a runtime or numerical-policy failure.

The same binary also reran the experimental direct-GGUF E2B recovery boundary
against official Q4_0 GGUF SHA-256
`fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`.
Canonical full-prefix DPO passed exact interruption/resume with adapter
`ead771802f218710253e09f993da8a510bed40b44d5305af4dda397124fdec5f`
and qualification-report SHA-256
`5e9683ed0a3c819676e3ced992eba6b2387b4c5e5865745f7f74d97cd7543546`.
Canonical full-prefix GRPO passed exact training/discrete recovery plus the
documented bounded terminal Metal KL comparison, with adapter
`9ce1b03dc532740961f189b589930efbc46c2fbd7c9108ece0e4262d901893b3`
and report SHA-256
`6d612a6867517a04aa6543336ae97cc891a173e85172ca0cf07612fa92207eab`.
The negative direct-GGUF/incremental-KV dry run still exits nonzero with
`DirectGgufGrpoIncrementalKvNotQualified`; exact shadowing previously exposed
the divergent token prefix `3771 236761 7993 236743` versus canonical
`3771 236761 108 16907`.

SafeTensors E2B and E4B incremental-KV GRPO separately passed the two-epoch
SIGTERM/resume gate with active-candidate batching, prompt-tail cloning, exact
full-prefix shadowing, and zero host-logit fallbacks. Both runs reproduced
their uninterrupted adapters and terminal KL values exactly:

| Model | Adapter SHA-256 | Qualification-report SHA-256 | Held-out mean / top-rank reward |
| --- | --- | --- | ---: |
| E2B | `43e6d637a316b38247737c5e3879843e5576b95ed1c81655850f91962666a787` | `57b9f701cf0ae3374af37f5469e637fff47c77fa4911482e36036470823514ff` | `0.34375 / 0.5625` |
| E4B | `0efbf1ad76f9e145382c18aaa24813d081220e8ceb373988bf33b0967e92a8c7` | `eec8f3936d8eaadfa36935d4480653b671bca005b3e80854cdc2e5d314b238d5` | `0.234375 / 0.0625` |

These reports live under `boundaries/e2b-incremental-resume` and
`boundaries/e4b-incremental-resume` in the same evidence root. They prove
frame-drained epoch-boundary recovery and telemetry continuity; the E4B row
does not override the separate long-horizon top-rank quality failure.

The final-source same-Mac DPO reprofile uses pinned MLX `0.31.2`, MLX-LM
`0.31.3` at revision
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, clean MLX revision
`68cf2fddd8de5edd8ab3d926391772b2e2cedad8`, and native-build attestation
SHA-256
`96a04c5e5176788310f373863054821ee35e90ec0e609cec35df854831b78f09`.
Every cell uses one cold update, one first update, three warmups, and 20
measured updates over the identical five-pair order. Peak physical footprint
covers the complete process lifetime. Q128 means pair-safe 128-row buckets
with a 256-row minimum.

| Model / schedule | Antfly median / mean | MLX median / mean | Antfly / MLX peak physical bytes | Median result |
| --- | ---: | ---: | ---: | ---: |
| E2B fixed | `2.498859 / 2.810538 s` | `2.047034 / 2.046491 s` | `13,012,883,360 / 18,910,715,824` | MLX `1.221x` faster; Antfly `31.19%` lower memory |
| E2B Q128 | `1.861362 / 2.049095 s` | `1.530776 / 1.453914 s` | `11,442,083,456 / 18,957,165,112` | MLX `1.216x` faster; Antfly `39.64%` lower memory |
| E4B fixed | `29.944145 / 32.246341 s` | `24.459206 / 25.994102 s` | `21,924,705,720 / 26,414,804,728` | MLX `1.224x` faster; Antfly `17.00%` lower memory |
| E4B Q128 | `22.326479 / 26.784846 s` | `11.221655 / 10.058360 s` | `26,676,814,464 / 26,362,818,560` | MLX `1.990x` faster; Antfly `1.19%` higher memory |

Q128 improves Antfly median update time by `1.342x` on E2B and `1.341x` on
E4B. It reduces E2B peak memory by `12.07%`, but increases E4B peak memory by
`21.67%`. The fresh fixed-E4B cell no longer reproduces the earlier Antfly
timing lead; MLX is faster in all four final-source cells. Common-oracle MLX
evaluation of every Antfly adapter still prefers all five chosen responses.
These are single-process diagnostics, not an alternating repeated-process
distribution.

The Antfly wrapper / task-report and MLX report SHA-256 pairs are:

- E2B fixed:
  `66a86326d03569d2f336664c8631cb96c74bbb6836cedd97bdfe35cd5ea8fc2b` /
  `333f9e7e58d557b9e4cbc5a7dc27887d88fda06aede3e07a9b24ed82dd36599d`,
  MLX `b8bee961ad4f33baec0329c9fa2312285b29c21339f69b50afa07f9b0321c3cc`;
- E2B Q128:
  `35199634cf49b3ffc3369adafd196b8554c377d9c3e391e88e2f657e36e59c9f` /
  `83dec9dcbdd654afd5c550f030b267dc1fc148df3f92b20c738823f920ee63f1`,
  MLX `3379ae4a5b5e43622e5e18478783cb7b1ae8918284da3645e27c59c64e2bdd48`;
- E4B fixed:
  `a9b405ff63b9fea4c0ba09d3ecd6bc68b90e5acbadab6d12d4b1d341d27e6095` /
  `cc281bcb8e8888a2d60f3f04c167cbe4c01b0f72acb9aee60032a4553b49c364`,
  MLX `c1135b3edfc6cf891960ba2eb583528a69b5cefa4ba719af0eccd005220fa533`;
- E4B Q128:
  `4d977802c0c0e70a288a6f97a695532570a619fd72a7a7c7d6d45766272568b8` /
  `bba65f42c9b4b4b81e28cb07f9a9ca64b676bca08982e24ccee8fc94e9ebb19c`,
  MLX `043283a3e02eb3713c24a62fe13b2660e5f0b4565efe52e4cb3ad11554e88d36`.

The first E2B fixed wrapper attempt lost only its final process-exit footprint
sample after training succeeded; the corrected sampler tolerates that bounded
post-exit race. The interrupted E4B Q128 root has no wrapper report and remains
excluded. Both incomplete roots were preserved, and accepted reruns used new
immutable `v2` roots.

These campaigns establish bounded RNG/data-order robustness, not independent
initialization: all three seeds start from the same adapter. They use small,
fixed row sets and absolute floors, not baseline-relative improvement,
multi-task convergence, or a five-process performance distribution. The
historical E2B high-learning-rate GRPO collapse remains valid negative
evidence. None of these results enables public `qlora-sft` or broad E4B GRPO
quality claims.

## Preference P1 Hardening (2026-08-23)

The final production review closed two public-boundary defects without
expanding the qualified training surface. First, mutable output paths are no
longer compared only after lexical normalization. The shared fine-tuning path
guard canonicalizes the deepest existing ancestor and retains any missing
suffix, so an output routed through a symlink into an immutable model, dataset,
adapter, checkpoint, reward input, or planned report fails before creation.
Second, product admission and Python qualification use the shared typed
environment policy described above. The four previously unqualified
correctness switches fail closed, and the qualifiers cannot inherit them.

The exact-source PR gate selected 284 Gemma4 fine-tuning tests in each of
Debug, ReleaseSafe, and ReleaseFast: 282 passed and only the two optional
fixture-dependent tests skipped. The focused graph gate passed 7/7 in Debug
and ReleaseSafe, the lifecycle smoke passed 4/4, the shared policy and path
suites passed 3/3 each, and the deterministic 16-row oracle fixture regenerated
and checked byte-for-byte. The final arm64 ReleaseFast binary is
`/private/tmp/antfly-gemma4-p1-fixed-release-20260823-v3/bin/antfly-inference`
at SHA-256
`0b97bb30396f14e15fc126ca156f0da79b7a8b6a98c9366d21a2754c7fbe2b97`;
its embedded policy source is SHA-256
`62333c528648a43a4711f7ab4a04e0fc70876f54b37549443de8469d8ec88dd5`.
Public probes against that binary rejected the symlinked output as
`PreferenceArtifactInputConflict` and each of
`TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY`,
`TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE`,
`TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC`, and
`TERMITE_DISABLE_PAGED_KV` as
`UnattestedGemma4PreferenceEnvironmentOverride`; neither probe created an
output manifest, report, adapter, or model subdirectory. These fixes do not
change the preserved E4B GRPO quality failure or enable direct-GGUF plus
incremental-KV GRPO.

## Production Roadmap and Release Gates

1. **Keep one green product contract.** Compile the typed CLI, four-step
   recipe/workflows, v6 admission, PEFT sidecar, checkpoint/resume, and strict
   Metal path together; require focused Debug, ReleaseSafe, and shipping-mode
   macOS gates from the final clean branch. Snapshot the numerical-kernel
   admission policy once per run, bind its fingerprint into the run manifest
   and telemetry, and reject policy drift before optimizer mutation.
2. **Expand the oracle and real-data preference loops.** The bounded E2B/E4B
   UltraFeedback DPO and BoolQ GRPO comparisons are archived with disjoint
   evaluation and matched MLX evidence. Three-seed/eight-epoch absolute-floor
   campaigns now pass for E2B/E4B DPO and E2B GRPO; E4B GRPO fails its
   predeclared top-rank floor. Next run pinned HF/PEFT one-, two-, and
   eight-step traces against Zig native and Metal for both target presets,
   prove semantic adapter equality after explicit key translation, diagnose
   E4B GRPO under-learning, and add independent initialization plus
   baseline-relative multi-task release thresholds.
3. **Finish optional-lane provenance.** Bind teacher targets to their
   teacher/base identity and extend the closed run ledger when optional
   artifacts are admitted.
4. **Complete artifact transactions.** Add a typed staged-artifact evaluator,
   stale-staging recovery, and power-loss failure injection, and replace
   whole-buffer prepared JSON with immutable streaming shards.
5. **Extend the qualified durable-resume surface.** Real E2B and E4B BF16 Metal
   SFT/DPO/GRPO interrupted-and-resumed training trajectories now match
   uninterrupted execution exactly, with byte-identical adapters; terminal
   GRPO KL floats carry the separately attested narrow GPU-evaluation bounds
   above. Checkpointed incremental-KV GRPO also passes E2B/E4B epoch-boundary
   recovery with cumulative telemetry and rebuilt transient pages. Experimental
   canonical direct-GGUF E2B SFT, DPO, and GRPO have recovery passes. Add
   retained generations and mid-epoch scheduling only if operational evidence
   justifies the extra state surface.
6. **Finish production-shape compute.** The gate/up backward-input sum is now a
   qualified default runtime region at rows 64, 128, and 512, and frozen-head
   fused linear cross-entropy now owns the strict-Metal hard-label and uniform
   DPO sequence objectives without global logits. Independent-row SFT length
   buckets and pair-safe DPO buckets are qualified opt-ins with fixed-shape
   rollback, bounded graph caching, and real E2B/E4B evidence. Next build a
   memory-safe whole-pair or segment-aware packed DPO graph to close the E4B
   Q128 gap, finish group-safe GRPO scheduling, and coalesce the forward
   gate/up projection and saved-value path without violating autodiff
   lifetimes. Extend the fused loss reader to packed Q4_0, Q4_K, and Q6_K only
   when direct QLoRA admission is ready. Prove native parity and peak-memory
   bounds at intended E2B/E4B sequence lengths before promotion.
7. **Extend the same-Mac performance baseline.** The realistic fixed/Q128
   E2B/E4B DPO matrix and four-token GRPO diagnostics now establish bounded
   single-process baselines. The final-source refresh finds MLX faster in all
   four DPO cells; Antfly retains a material memory advantage except at E4B
   Q128, where footprint is effectively tied and slightly worse. Repeat the
   DPO matrix as an alternating five-process distribution on the pinned host
   and report dispersion before making a stable performance claim.
8. **Real E2B acceptance.** Pinned BF16 DPO/GRPO now pass bounded
   three-seed/eight-epoch absolute quality floors and exact epoch-boundary
   recovery. Add independent initialization, baseline-relative held-out gates,
   larger disjoint datasets, adapter-reload generation, and the remaining
   native-versus-Metal/HF per-target parity cells before a broad claim.
9. **Real E4B acceptance.** Pinned BF16 DPO passes the bounded multi-seed gate
   and incremental GRPO recovery is exact, but GRPO misses its predeclared
   top-rank quality floor. Diagnose that failure, then repeat the broader E2B
   gates with sequences and target presets that exercise shared KV, PLE, and
   the larger adapter/optimizer footprint without fallback or unbounded growth.
10. **Deployment and materialization.** Implement a streaming, dtype-preserving,
   sharded writer; then apply accepted BF16-trained adapters to pinned E2B/E4B
   QAT Q4 serving artifacts and require fingerprint, exact-token, quality,
   memory, and repeated-generation gates.
11. **Finish optional direct Q4/QLoRA qualification.** The pinned official E2B
    Q4_0 inventory now passes strict optimizer and process-kill/resume gates for
    canonical SFT/DPO/GRPO without host dequantization. Direct-GGUF GRPO plus
    incremental KV remains rejected after exact token divergence. Next prove
    peak memory, MLX/HF multi-step parity, multi-seed task quality, adapter
    reload generation, and E4B GGUF. Until those artifacts pass, the public
    `qlora-sft` recipe remains unsupported.
12. **Scale model-based GRPO rewards deliberately.** The typed weighted-rule,
    pinned external-verifier, and pinned `model-command` contracts now provide
    bounded failures, artifact/tokenizer/template/calibration identities,
    response attestation, subprocess integration coverage, structured failure
    traces, and replayable train/eval evidence. Keep built-in exact,
    case-insensitive, prefix, and token-match modes scoped to verifiable tasks.
    Add true request batching only with an explicit resource and ordering
    contract; the current implementation correctly enforces
    `max_batch_size = 1`.

Finally, run the macOS real-GPU workflow, make it a required branch-protection
check, and archive its artifacts. Keep separate opt-in gates for pinned real
E2B/E4B models because the synthetic runner job is not a production-scale test.

Do not call the path production-ready until every gate above has a reproducible
artifact, pinned model/dataset provenance, and an explicit pass threshold.
