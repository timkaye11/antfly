# Gemma4 E2B/E4B LoRA Fine-Tuning

> Status: production-candidate only for the previously attested exact-source,
> single-device text-only BF16 LoRA-SFT and sigmoid-DPO lanes. The current
> algorithm-hardening source changes the GRPO sampling distribution from ranked
> selection to resume-stable seeded categorical sampling and expands DPO to
> cDPO, IPO, and SimPO with report schema v7. A fresh real E2B Metal diagnostic
> previously passed the v7 three-seed, prompt-paired GRPO quality contract. Its separate
> release holdout remains sealed for a fresh zero-swap host, so this result does
> not yet promote GRPO. A September 15 E4B diagnostic passes unchanged absolute
> and baseline-relative quality gates after 64 training groups and 32 heldout
> groups. The full multi-seed v7-scale E4B campaign remains unqualified.
> Exact incremental KV reuse remains experimental and slower than rollback.
> Canonical direct-GGUF SFT/DPO/GRPO remains a research surface, and public
> QLoRA remains fail-closed. Required hosted CI, a fresh-host zero-paging
> attestation, full E4B memory/performance and quality qualification,
> repeated performance, and GGUF task-parity gates remain open.

The September 15–16 review remediation changes training rendering, F16 backward,
attention score scaling, reference snapshots, and resume identities. Earlier
quality and performance results do not qualify these changes. See
[the remediation ledger](GEMMA4_REVIEW_REMEDIATION.md) for current checks and
remaining release gates.

The September 15 E4B diagnostic improves mean reward from 0.236328125 to 0.23828125
and top-ranked reward from 0.28125 to 0.40625, with KL loss 3.82473e-5. Its
trained adapter is published after all configured acceptance checks pass.
The capture has no sampled swap growth, but the host already has swap in use;
this is bounded diagnostic evidence. Current comparisons use pinned MLX;
HF/PEFT numerical qualification is outside this work's scope.
The matched E4B MLX replay completes all 61 updates with 0.0113% adapter-update
relative L2 error across 686 tensors. All 1024 training and 512 heldout
completion tokens match; heldout reward metrics are identical. The bounded
frozen-PLE cache makes this numerical evidence, not a performance qualification.
The matched-size E2B replay has 0.0308% update-vector relative L2 error across
552 tensors and the same exact completion-token agreement. Mean reward stays
at 0.759765625 in both frameworks, so the unchanged improvement gate prevents
publication. These one-seed, one-token-horizon captures do not replace the
multi-seed quality campaign or qualify older source revisions.

All performance and quality artifacts below remain valid historical evidence
for the exact source and binaries they name. They do not attest the current
algorithm-hardening diff unless a section explicitly says so.

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
quality. Those archived results establish bounded data-order robustness only.
The current qualifier creates a distinct deterministically seeded adapter per
run and requires held-out improvement over that run's initialized adapter. E2B
DPO passes that stronger gate. The current E2B GRPO all-linear candidate also
passes the fresh schema-v7 diagnostic described below: all three seeds improve
mean and top-completion reward, remain within the exact positive-group
noninferiority allowance, and pass the aggregate prompt-paired sign test.
The separate final holdout has not been opened because the diagnostic host was
paging. E4B has not been rerun under the stronger contract. Multi-task
convergence, repeated distribution-level performance, and required-CI
acceptance also remain open. This remains an implementation and qualification
contract rather than a blanket readiness claim.

Distributed training, multimodal/projector training, public GGUF QLoRA recipes,
and Gemma4 MoE models are outside the supported path described here. Q4_0,
Q4_K, and Q6_K packed frozen-linear input-gradient kernels now exist. An
explicitly gated canonical direct-GGUF E2B command lane has passed optimizer
and interrupted/resumed Metal gates for SFT, DPO, and GRPO, but remains
experimental pending memory, cross-framework parity, and task-quality
campaigns. Direct GGUF plus incremental-KV GRPO is rejected after exact token
divergence. MLX-LM is a same-Mac performance reference, not an Antfly training
backend.

## Renderer and resume policy

Prepared Gemma text uses the canonical non-thinking serving layout. Channel
annotations and thought content are stripped from every assistant turn,
including the target. Thinking-mode/reasoning-trace training is not supported;
datasets requiring preserved reasoning traces need a separate renderer. A pending
`<|tool_response>` delimiter remains in the rendered prompt but is excluded
from assistant labels, as are completed tool observations. Empty Gemma input
contains `<bos>` and has no label spans.

Renderer provenance deliberately hashes the entire renderer source. This is a
conservative cache-invalidation policy: even a comment edit requires prepared
inputs to be refreshed. It avoids claiming semantic equivalence without a
complete renderer behavior oracle.

Preference checkpoints now use fingerprint domain v6, which binds the
numerical environment as well as the model, input data, initialized adapter,
optimizer, and evaluation acceptance contract. Pre-v6 checkpoints require a
new run. Evaluation identity remains bound intentionally: resuming must not
silently change the data or thresholds that authorize adapter publication.
The AdamW default remains 0.01; this review does not change it.

## Implemented Scope

- `gemma_chat/v1` supervised fine-tuning data, including system, multi-turn,
  tool-call, and tool-result messages.
- The canonical Gemma4 text/tool wire format used by inference, checked against
  its Jinja template. System messages occupy their own turn; tool observations
  remain inside the assistant turn. Completion-only loss includes assistant
  content, calls, and terminators, and excludes role headers and tool results.
  Historical channel annotations are stripped as in the default serving template.
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
  closed. Sharded bootstrap validates the complete index and shard set. Merged
  deployment export accepts monolithic or sharded Safetensors and streams
  untouched tensors byte-for-byte while materializing one adapted weight at a
  time in its original F32/F16/BF16 dtype.
- Memory-bounded sparse causal targets. The graph never owns a dense
  `[sequence, vocabulary]` target. Strict-Metal hard-label training uses the
  frozen tied-head fused linear cross-entropy primitive, which returns a
  device-reduced scalar and its hidden-state gradient without materializing a
  global logits tensor when the BF16 head and at-most-512-row CCE route is
  available. Other heads or shapes may materialize full logits and emit a warning;
  `TERMITE_METAL_REQUIRE_LINEAR_CCE=1` rejects that fallback. General signed
  per-token targets retain a bounded sparse projection fallback.
- Production preparation emits `gemma4_prepared/v6`. It binds tokenized
  examples to the selected base artifact, tokenizer assets, chat-template,
  source dataset/split/revision, canonical source row, rendered chat, group,
  and media-content identities. Adapter bootstrap records the model identities
  and a closed, exact A/B target inventory.
- `adapter_config.json` stays inside the public PEFT LoRA schema. Antfly-only
  provenance, exact target policy, recursive metadata, and the internal tensor
  key format live in the strict `antfly_finetune_manifest.json` sidecar. The
  sidecar also binds the adapter checkpoint byte size and SHA-256 digest. V3
  manifests additionally bind the deterministic initialization seed, which is
  preserved by generic bundle save and trained-adapter publication.
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

Gemma SFT, DPO, and GRPO expose `optimizer.weight_decay` (SFT CLI:
`--weight-decay`). The historical default remains **0.01**, applied to every
trainable LoRA tensor. Set **0** explicitly when comparing against a no-decay
reference. The effective value is recorded and bound into checkpoint identity.
Beta1, beta2, and epsilon remain 0.9, 0.999, and 1e-8.

Training environment overrides are bound into checkpoint identity on native
and Metal. SFT records the assignments and their digest; preference reports
record the assignments alongside resolved numerical flags. See
[environment controls](GEMMA4_ENVIRONMENT.md). SFT run identity v3 and the new
preference environment identity reject pre-remediation checkpoints. Prepared
inputs also need regeneration because the renderer digest now binds its source.

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
exact, with no native/interpreter fallback. Historical report basenames are
`antfly-gemma4-e2b-resume-acceptance-20260819-v3` and
`antfly-gemma4-e4b-resume-acceptance-20260819-v2`; their original temporary
locations are not durable qualification evidence. Use
`scripts/gemma4/qualify_gemma4_metal_resume.py` to reproduce the gate; a direct GGUF
requires the internal `ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA=1` environment
admission. The public QLoRA recipe remains rejected.

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
adapter saves, the streaming merged-model path, recursive compressed-base output,
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

Full-model materialization now accepts monolithic or sharded Safetensors input,
sorts one deterministic output inventory, copies every untouched payload
byte-for-byte, and merges only one LoRA/DoRA target at a time. Adapted weights
are accumulated in F32 and encoded back to the source F32, F16, or BF16 dtype.
Before no-replace publication, the staged checkpoint is reopened and checked
for exact inventory, shapes, dtypes, byte lengths, finite merged values, and
unchanged non-target payloads. Synthetic sharded BF16 coverage proves the
streaming and dtype-preservation contract; a pinned full E2B/E4B export,
reload/generation comparison, disk-footprint measurement, and serving-quality
gate remain required deployment evidence. GGUF merged export remains
unsupported.

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
same gate with its decomposed loss graph.

Preference training additionally owns a durable mid-epoch checkpoint cadence.
`checkpoint.every_examples` counts optimizer examples (DPO pairs or GRPO
prompt groups) inside each epoch and writes the same atomic trainer
checkpoint plus content-addressed sidecar immediately before the example at
each cadence boundary, so the durable state always describes exactly the
completed prefix regardless of any in-body skip path. The sidecar schema is
now `antfly_gemma4_preference_checkpoint_state/v2` with an explicit
`examples_into_epoch` cursor; v1 sidecars remain loadable with an implied
zero cursor. Resume recomputes the deterministic per-epoch order (on-disk
pair order for DPO, seeded Fisher-Yates prompt order for GRPO), skips the
consumed prefix of the restored epoch, and replays the identical trajectory
because GRPO sampling seeds derive from the run seed, epoch, and original
prompt index. The cadence requires `every_epochs`, is preference-only, and
is fail-closed for the SFT lane, compiled GRPO sampling, and incremental-KV
GRPO. Retained checkpoint generations and mid-epoch SFT CLI scheduling
remain open; activation checkpointing is recomputation and does not close
those gates.

## Reference Oracles and Performance Targets

The reference strategy deliberately assigns one job to each implementation:

| Reference | Role | Required evidence |
| --- | --- | --- |
| Hugging Face Transformers + PEFT | Adapter interchange check; HF numerical qualification deferred | Exact Antfly `input_ids`/`labels`; pinned local model and package revisions; eager causal loss; loss, logit probes, per-target gradients, Adam state, updates, and normalized adapter inventory |
| MLX-LM | Current same-Mac numerical, performance, and memory reference | Identical pinned model/case/protocol/hardware; fresh processes; explicit device synchronization; alternating framework order; at least five repetitions and the locked sequence/accumulation matrix |
| Unsloth | Separate NVIDIA recipe, convergence, and quality cross-check | Pinned CUDA hardware/software and datasets; never mixed into the Apple-Metal throughput gate |

Unsloth is not the Metal equivalent of the GLiNER2 Fastino oracle: it is
CUDA-oriented and cannot provide a defensible same-Mac performance comparison.
The current campaign is MLX-first for numerical trajectories and local
Apple-Silicon performance. Stock PEFT loading checks adapter interoperability;
HF numerical qualification and CUDA work are deferred.
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
| BF16 correctness | Unquantized BF16 Safetensors | Establish native forward, loss, gradient, optimizer-update, and adapter-save correctness, then compare the stored-weight Metal kernel | Final-binary E2B and E4B Metal jobs pass process-kill/resume with byte-identical adapters. Three-seed/eight-epoch E2B/E4B DPO and E2B GRPO pass historical bounded absolute quality floors; E4B GRPO fails the predeclared top-rank floor. Independent-initialization, baseline-relative E2B DPO passes, while restart-safe compiled E2B GRPO fails its current seed-17 learning-rate sweep and E4B has not been rerun under the stronger contract. Full HF parity, distribution-level performance, and required CI remain open |
| Q4 deployment / QLoRA substrate | QAT Q4 GGUF used by serving | Prove target-name compatibility, adapter loading/application, memory bounds, and post-training generation quality on the deployed base | Packed Q4_0/Q4_K/Q6_K `dX` kernels pass without host dequantization, and explicitly gated canonical E2B Q4_0 SFT/DPO/GRPO lanes pass optimizer and process-kill/resume gates. Public `qlora-sft`, E4B GGUF, memory, parity, and task-quality gates remain fail-closed; direct-GGUF plus incremental-KV GRPO is separately rejected |

The loaders may accept a Q4 GGUF and bootstrap adapter targets from its tensor
headers. Kernel availability and target discovery alone are not proof that
QLoRA training is numerically correct or production-ready. Promote direct
quantized-base training only after a real model completes strict no-host
optimizer, memory, resume, artifact, parity, and task-quality gates against the
BF16 reference.

## Current verification and historical evidence

Current checks, pending gates, and the MLX-first performance sequence are tracked
in [the remediation ledger](GEMMA4_REVIEW_REMEDIATION.md). Older measurements,
failed experiments, and dated qualification runs are preserved separately in
[the historical evidence index](GEMMA4_HISTORY.md); they do not qualify current source.

## GRPO scoring policy

GRPO scores and differentiates `log_softmax(final_logits / temperature)` for
old, current, and frozen-reference policies. The temperature division follows
Gemma's final logit softcap. Both materialized and bounded fused loss heads use
this objective. Reports identify it as
`temperature-scaled-full-vocabulary/v1`, and checkpoints bind the objective
version and temperature. Captures made with raw-logit scoring must be refreshed.

Top-k and top-p restrict rollout sampling; they do not renormalize this training
objective. This matches the separation in the
[TRL scoring implementation](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py).
A truncated rollout distribution is not the full-support policy distribution:
do not describe filtered sampling, or the greedy evaluation anchor, as an exact
on-policy estimator. Use top-k 0 and top-p 1 for full-support sampling. MLX
comparison runners apply the same temperature and reject captures missing the
scoring-policy identity.

For matched single-token GRPO diagnostics, the MLX runner accepts
`--completion-execution coalesced-single-token`: one shared causal predictor
supplies the group loss and a single backward pass. Multi-token use fails
closed. The existing sequential mode remains available for numerical controls.
`coalesced-single-token-eager` uses the same group loss without whole-step
compilation, making compilation's memory and numerical effects measurable.
`--mlx-cache-limit-mib` bounds only MLX's reusable free-buffer cache and is
recorded in the capture contract; omitting it preserves the runtime default.
Neither option changes the acceptance thresholds or promotes diagnostic runs.

For memory-bounded numerical replay only, `--compact-frozen-ple` retains exact
frozen PLE embedding rows for the attested prompts and recorded completions.
It is restricted to aligned-F32, single-token categorical trace replay; an
uncaptured input token fails closed. The trainable per-layer projection stays
in the normal graph. Captures record the admitted row IDs and are explicitly
ineligible for performance qualification.
