# Gemma4 E2B/E4B LoRA Fine-Tuning

> Status: experimental, single-device, text-only, and not production-ready.
> The focused CPU and synthetic strict-Metal source gates are green locally.
> The dedicated macOS GPU workflow still must pass in CI, and a bounded
> real-model first-step run remains required.

The current Zig path implements the text causal-LM graph, sparse loss, LoRA
optimizer, evaluation, and artifact pipeline intended for Gemma4 E2B and E4B
models. It prepares Gemma4 chat data, bootstraps a strict LoRA adapter
inventory, requires a distinct evaluation artifact, and selects the native or
Metal backend. Device-only BF16 frozen linears and embedding gathers, rank-4
attention VJPs, and a strict synthetic optimizer step have focused test
coverage. Real-model first-step acceptance and a successful required Metal CI
run remain open, so this is an implementation contract rather than a readiness
claim.

MLX, distributed training, multimodal/projector training, and Gemma4 MoE models
are outside the supported path described here.

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
  contract. Sharded bootstrap validates the complete index and shard set;
  production-scale merged-model materialization is not implemented.
- Memory-bounded sparse causal targets. The current loss graph projects a small
  number of supervised rows at a time instead of owning a full
  `[sequence, vocabulary]` activation.
- Prepared-input schema `gemma4_prepared/v4` binds tokenized examples to the
  selected base artifact, tokenizer assets, and chat-template identity. Adapter
  bootstrap records the same identities and a closed, exact A/B target
  inventory.
- Sparse teacher targets carry their temperature and are accepted only when
  their persisted base/tokenizer/template provenance matches the prepared
  student artifact. Multimodal targets additionally bind the projector digest.

## Text Training Flow

Run commands from `zig/pkg/inference`. A dataset row can be as small as:

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

mkdir -p "$RUN"

zig build prepare-gemma4-lora-inputs -- \
  "$BASE" "$TRAIN_DATA" train "$RUN/prepared_train.json" \
  --max-examples 1000 --max-seq-len 512

zig build prepare-gemma4-lora-inputs -- \
  "$BASE" "$EVAL_DATA" eval "$RUN/prepared_eval.json" \
  --max-examples 128 --max-seq-len 512

zig build bootstrap-gemma4-lora -- \
  "$BASE" "$RUN/adapter_seed" \
  --rank 16 --alpha 32 --target-preset peft-qv

zig build train-eval-gemma4-lora-bundle -- \
  "$BASE" "$RUN/adapter_seed" "$RUN/prepared_train.json" "$RUN/train_native" \
  --eval-prepared "$RUN/prepared_eval.json" \
  --trainer autodiff --backend native \
  --lr 0.0003 --max-examples 1000 --eval-max-examples 128 \
  --epochs 1 --max-grad-norm 1.0 --grad-accum 1
```

The backend flag and `--eval-prepared` are required; there is no implicit
backend fallback or training-data eval fallback.
`--trainer auto` is only a compatibility alias for `autodiff` and never falls
back to surrogate training. The final primary training directory and adapter
bootstrap directory must not already exist and are published with no-replace
semantics. Prepared JSON files do not yet have the same immutable publication
contract; use fresh paths.

The Metal path admits rank-2 BF16 Safetensors weights. Autodiff cancels the
linear VJP's redundant double transpose, and Metal computes `dX = dY @ W`
directly from the persistent BF16 forward-weight slot with f32 accumulation.
It neither creates a transposed weight nor materializes a full-model f32 copy.
Native BF16 embedding tables also use a device-index gather, and autodiff
propagates q/k/v gradients through Gemma4's rank-4 two-batch-axis attention
contractions.
Rank-2 F16 Safetensors and packed GGUF bases remain rejected before backend
construction. Metal also requires `TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1`.
Parity diagnostics, native partitions, unsupported operations,
interpreter/runtime fallbacks, host-materialized graph outputs, explicit
runtime-input transfers, and non-device gradients fail the step before gradient
accumulation or optimizer mutation. Additional promotion telemetry is not yet
fully gated, as detailed below. Compiled Metal evaluation uses a loss-only graph
with resident weights and no gradient or Adam-state allocation.

`--activation-checkpoint-interval N` enables graph activation recomputation at
layer boundaries. It is a memory-control mechanism, not durable training
checkpoint/resume.

## Data, Provenance, and Sequence Admission

Newly prepared artifacts use `gemma4_prepared/v4`. The current source records:

- a digest of the selected model artifact set, including a Safetensors index
  and its shards when present;
- tokenizer-asset and chat-template identity digests;
- a canonical digest of the prepared token/label examples; and
- the prepared artifact family and schema versions.

Training recomputes those identities against the selected base, validates the
adapter's recorded base/tokenizer/chat identities, and requires a separate eval
artifact. Every prepared example carries a canonical source identity derived
from its row id, source metadata, untruncated messages/tools, and ordered media
contents. Train/eval overlap checks use that identity, so changing truncation or
renaming an unchanged media file cannot bypass the held-out gate. The loader
also recomputes every declared aggregate and derives text-versus-multimodal
routing directly from the examples. Dataset and split provenance are not yet
persisted, so release-quality reports still require an external audit of the
declared source split.

The provenance chain is not yet complete outside the primary training
admission. Generic adapter save/load/materialize does not consistently preserve
or validate these identities, and the final report does not yet hash every
input and published payload. Teacher-target materialization is intentionally
same-base only: it fingerprints and validates the teacher before graph
construction, persists the teacher identity, and exposes it in the final
training report. Broader cross-model distillation needs an explicit future
schema. The remaining gaps are release blockers, not optional metadata
improvements.

Sequence length is admitted before graph/backend construction and is currently
bounded to `1..min(model_max_position_embeddings, 2048)`. The 2048 limit is a
temporary safety ceiling, not a claim that every admitted E2B/E4B sequence fits
the available device memory. Prepared JSON is still a single whole-buffer
artifact with a 128 MiB load ceiling; streaming/sharded prepared data is an open
production requirement.

The pilot and recursive convenience workflows prepare a separate held-out
artifact and pass it through the mandatory `--eval-prepared` contract. Generated
pilots create deterministic non-overlapping train/eval datasets. With a custom
dataset, use an `eval` split or pass `--eval-dataset` and `--eval-split`.

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
- GGUF autodiff and Metal rank-2 F16 Safetensors are rejected before prepared
  inputs or output artifacts are opened. Rank-2 BF16 Safetensors are admitted
  through the dedicated device-only input-gradient path. A host-dequant
  fallback is intentionally not provided.
- Metal requires the training graph executor explicitly enabled. Every eval
  and train step records executor partitions/dispatches and rejects native or
  unsupported partitions, diagnostic direct execution, interpreter/runtime
  fallback, true host outputs, explicit runtime-input transfers, or
  non-resident gradients before mutation. Gather/reduce/resident-cache
  promotion counters are recorded but are not all strict-gated yet, so the
  current report is not a complete proof of zero host-to-device promotion.
- `auto` resolves to real autodiff. Surrogate behavior requires the explicit
  diagnostic-only `--trainer surrogate` spelling; production recipes reject
  that spelling.
- Adapter target patterns are strict. Missing, empty, unknown, or conflicting
  target selections are errors.
- Unsupported graph/model configurations and malformed Gemma4 tool-call wire
  data return typed errors rather than being rewritten approximately.
- The production-intent lane is text-only. The lower-level command still
  contains an experimental multimodal route whose projector/embedding host
  round trips occur outside strict-step telemetry; do not treat that route as
  admitted until it fails closed or receives a separate end-to-end contract.
- Layer-scoped autodiff, layer-wise LR decay, and schedule-free autodiff are
  rejected until their semantics are implemented and tested.
- DoRA training and PiSSA/LoftQ initialization are rejected until the graph and
  adjusted-base artifact semantics are implemented. Generic save/materialize
  also rejects recursive adapters that cannot be represented as one merged
  base.
- Gemma4 recipes admit only `lora-sft`. Full `sft`, `qlora-sft`, and declared
  optimizer/eval/checkpoint/runtime/algorithm/artifact options that the command
  cannot honor fail with typed errors rather than being silently ignored.
  Unknown JSON fields are errors, and bootstrap/trained output directories must
  be normalized, disjoint leaves. `model.name` is retained as report metadata;
  it does not select the checkpoint.

An accepted training epoch must report finite loss, nonzero supervised tokens,
and `optimizer_steps > 0`. Before/after evaluation intentionally performs no
updates, so evaluation records may correctly report zero optimizer steps; that
is not evidence that training succeeded.

## Artifact Publication and Materialization

The current source stages and no-replace-publishes bootstrap adapters, generic
adapter saves, the bounded legacy merge path, recursive compressed-base output,
and the primary autodiff adapter-plus-report directory. Adapter publication
validates a closed inventory: each configured exact target has one A/B pair,
unconfigured tensors are rejected, and DoRA metadata and magnitude tensors must
agree.

These guarantees are visibility-atomic, not yet power-loss durable. The shared
publisher does not fsync staged files, the staging directory, or its parent.
Prepared JSON is still written directly. The public trainer bundle saver uses
the shared no-replace publisher, but the `materialize-gemma4-lora --eval`
wrapper still publishes before its post-merge evaluation succeeds. That wrapper
must not be used as a durable production transaction yet.

Full-model materialization is deliberately fail-closed while it uses the
legacy whole-model F32 implementation. It accepts only a monolithic
Safetensors checkpoint of at most 64 MiB and rejects GGUF, sharded
Safetensors, and larger files. This prevents an E2B/E4B OOM but also means
production-scale E2B/E4B materialization is unavailable until a streaming,
dtype-preserving, sharded writer is implemented.

## BF16 Correctness and Q4 Deployment Lanes

Keep precision concerns separated until quantized training has its own parity
and quality evidence.

| Lane | Base artifact | Purpose | Current release posture |
| --- | --- | --- | --- |
| BF16 correctness | Unquantized BF16 Safetensors | Establish native forward, loss, gradient, optimizer-update, and adapter-save correctness, then compare the stored-weight Metal kernel | The BF16 frozen-linear kernel and admission gate are component-tested; full E2B/E4B first-step acceptance is open |
| Q4 deployment | QAT Q4 GGUF used by serving | Prove target-name compatibility, adapter loading/application, memory bounds, and post-training generation quality on the deployed base | Deployment validation only; direct Q4/QLoRA training is not yet an accepted correctness lane |

The loaders may accept a Q4 GGUF and bootstrap adapter targets from its tensor
headers. That capability alone is not proof that QLoRA training is numerically
correct or production-ready. Promote direct quantized-base training only after
it matches the BF16 reference within declared loss, gradient, update, and task
quality tolerances.

## Validation Status

### Previously passing focused evidence

Before the newest provenance, held-out-eval, publication, and resume hardening,
focused local tests had passed for:

- Gemma4 forward-graph construction, PLE naming, attention/activation scaling,
  shared KV, omitted-V contracts, and E2B/E4B target inventory rules;
- sparse causal targets and supervised-row LM-head projection;
- selected-artifact GGUF rejection, Metal BF16 admission, F16 rejection, and
  adapter rank/alpha/initializer/DoRA admission;
- BF16 frozen-linear backward-input parity, lazy-parameter compiled dispatch,
  WRT-aware frozen-`dW` pruning, device BF16 embedding gather, and rank-4
  attention VJPs;
- exact Gemma4 chat/channel spans and production tool-parser round trips; and
- a deterministic one-layer BF16 strict-Metal CLI step with one Metal optimizer
  update, a nonzero saved LoRA B tensor, zero then-known fallback counters, and
  immutable whole-run publication.

### Current integrated verification

The current branch adds prepared-input v4 provenance, mandatory separate eval,
sequence admission, closed adapter inventories, teacher-target provenance,
shared no-replace publication, loss-only compiled Metal evaluation, extra
strict-executor telemetry, and in-place restoration of resident Metal optimizer
slots. The included macOS `macos-15-xlarge` workflow runs the synthetic Gemma4
Debug, ReleaseSafe, and ReleaseFast gates with missing Metal treated as a
failure.

The current integrated source passes the following Linux ReleaseSafe gates with
Metal, CUDA, ONNX, and PJRT disabled:

```sh
zig build test-gemma4-finetune
zig build test-gemma-graph
```

`test-gemma4-finetune` selected 165 tests: 161 passed and four platform or
optional-fixture tests skipped. `test-gemma-graph` passed all six tests. Neither
result exercises a Metal device.

On a local Apple M4, the strict-Metal `test-gemma4-finetune` gate passed in
Debug, ReleaseSafe, and ReleaseFast: 173 tests selected, 171 passed, and two
artifact-dependent tests skipped in each mode. The synthetic CLI check performed
one real Metal optimizer step with a finite nonzero gradient and zero true host
outputs. The Debug and ReleaseSafe Metal graph gates passed all six tests, and
the focused Debug lifecycle gate passed all four matching tests.

The macOS job is a synthetic real-GPU gate, not an E2B/E4B scale gate. It must
run successfully in CI and be made required in repository branch protection;
workflow source alone is not evidence that either has happened.

A bounded live E2B QAT Q4 GGUF run (one example, sequence length 32, rank 2,
Q/V targets) passed dataset generation, exact 50-module bootstrap, tokenizer
preparation, and Metal graph admission. It then failed before the first
optimizer update when autodiff requested a transpose of a quantized base
weight (`UnsupportedTensorType`). The trainer now rejects GGUF autodiff earlier
with `GgufAutodiffBackwardNotYetSupported`; this is the expected open
QLoRA-backward gate, not a successful Q4 training result. The same workflow
passed its Metal dry run. No full dense BF16 E2B or E4B artifact was available
for the real-model correctness lane. The BF16 stored-weight component gate is
implemented and the strict synthetic CLI step passed in the earlier tested
snapshot, but neither is a substitute for that real-model acceptance run. The
locally available BF16 Gemma4 assistant checkpoint is an MTP draft that requires
target-model KV
donation and is not a standalone causal-LM acceptance artifact.

## Production Roadmap and Release Gates

1. **Finish dataset and artifact identity.** Persist dataset/split provenance,
   preserve provenance in generic adapter saves, and record
   training/eval/config/adapter payload digests in the immutable run report.
2. **Make the supported boundary honest.** Reject multimodal examples before
   publication/backend work until that lane is separately accepted, and gate
   every recorded strict-Metal promotion counter.
3. **Complete artifact transactions.** Make prepared data immutable and
   streaming/sharded, evaluate materialized output before publication, and add
   file/directory/parent fsync plus stale staging recovery and failure
   injection.
4. **Implement durable Gemma4 checkpoint/resume.** The low-level Metal restore
   now overwrites resident optimizer slots, but the CLI/recipe still need
   atomic adapter/optimizer/step/accumulation/epoch/cursor/RNG checkpoints. A
   real Metal interrupted-and-resumed trajectory must match uninterrupted
   training within declared tolerances.
5. **Bound production-shape compute.** Add autodiff-capable fused/chunked Metal
   attention and fused sparse LM-head cross-entropy. Prove native parity and
   peak-memory bounds at the intended E2B/E4B sequence lengths.
6. **Real E2B acceptance.** On a pinned BF16 artifact and disjoint, fingerprinted
   dataset, pass native-versus-Metal first-step loss/per-target-gradient/update
   parity, deterministic overfit, bounded multi-epoch training, resume, adapter
   reload, and held-out quality thresholds.
7. **Real E4B acceptance.** Repeat the E2B gates at E4B scale with sequences and
   target presets that exercise shared KV, PLE, and the larger
   adapter/optimizer footprint without fallback or unbounded growth.
8. **Deployment and materialization.** Implement a streaming, dtype-preserving,
   sharded writer; then apply accepted BF16-trained adapters to pinned E2B/E4B
   QAT Q4 serving artifacts and require fingerprint, exact-token, quality,
   memory, and repeated-generation gates.
9. **Optional direct Q4/QLoRA training.** Only after the BF16 lane is accepted,
    add packed-weight backward-input kernels and prove forward, gradient,
    optimizer, memory, and task-quality parity without host dequantization.
    Until then, direct training on GGUF remains unsupported.

Finally, run the macOS real-GPU workflow, make it a required branch-protection
check, and archive its artifacts. Keep separate opt-in gates for pinned real
E2B/E4B models because the synthetic runner job is not a production-scale test.

Do not call the path production-ready until every gate above has a reproducible
artifact, pinned model/dataset provenance, and an explicit pass threshold.
