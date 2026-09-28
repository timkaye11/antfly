# Gemma4 E2B/E4B LoRA Fine-Tuning

> Status: production qualification remains open for the current branch.
> Text-only BF16 LoRA SFT and sigmoid DPO are production candidates; GRPO,
> cDPO, IPO, SimPO, and direct-GGUF training require separate qualification.
> Passing local tests does not promote a training lane.

Current qualification work targets E2B. E4B qualification is deferred at the
user's request after its answer evaluator repeatedly exceeded the host memory
guard. Retained E4B diagnostics do not establish production support, and E4B
is no longer a prerequisite for the remaining E2B campaigns. All E2B quality,
recovery, oracle, performance, sealed acceptance and CI gates remain required.

The supported architecture scope is the dense Gemma4 E2B/E4B text decoder on
one device. The implementation includes chat preparation, adapter bootstrap,
SFT and preference training, held-out evaluation, checkpoint recovery, and
adapter export. Evaluation data must be distinct from training data. Native
and Metal backends are selected explicitly, with strict device execution for
the qualified Metal path.

The September 15 Metal/MLX GRPO comparison is a retained historical
diagnostic: E2B failed the reward-improvement gate in both backends, while E4B
passed its configured single-seed gate. Both used a one-token completion
horizon. These results do not establish multi-seed, longer-horizon quality.
Later renderer, numerical, resume, and checkpoint changes need evidence tied
to their own source and binary identities.

The September 22 hardened binary passes a bounded E2B GRPO development
campaign: three seeds, 64 training prompts for eight epochs, and 256 evaluation
prompts. Mean reward improves from 0.688151 to 0.691895; the prompt-level
paired test passes at p=0.006330. This one-token campaign does not replace the
full-size E2B/E4B campaign or the reserved holdout.

The completed full-size E2B DPO development campaign uses 1,960 distinct training
pairs for one epoch, three seeds (17, 42, 991), and the same 256 evaluation
pairs. Learning rate remains 1e-6 and the bounded candidate's quality thresholds
are unchanged. Input checks confirm distinct tokenized prompts, no train/eval
overlap and no truncation. All three seeds pass: mean held-out DPO loss improves
from 0.69314611 to 0.53642686, and preference accuracy reaches 75.78125% for
each seed. This is preference-margin accuracy, not answer accuracy. Full-size
midpoint recovery remains pending; existing host paging excludes clean-host
performance claims. Evidence is `full-e2b-dpo/independent-campaign-review.json`
under `.benchmark-assets/gemma4-sft-production-20260923/`.

The same binary passes the bounded sigmoid DPO development campaign at learning
rate 1e-6: three seeds, 64 training pairs for eight epochs, and 256 evaluation
pairs. Mean held-out DPO loss improves from 0.693146 to 0.634064. The initial
1e-5 candidate failed the held-out loss gate and its adapter was withheld;
the passing candidate changed only learning rate, with unchanged acceptance
thresholds. These results qualify the recorded development workload only.

September 23 recovery validation reproduces the accepted seed-42 DPO adapter
after interruption at update 224 of 512. The resumed adapter and full trainer
checkpoint are byte-identical to uninterrupted execution, and terminal metrics
match exactly. This covers the accepted development trajectory; it does not
replace the remaining E2B multi-token, oracle, or clean-host release gates.

The September 23 source-ordered SFT candidate fails development answer quality
after one epoch over 1,960 distinct training examples: greedy complete-answer
accuracy in pinned MLX falls from 75.00% to 63.28%. The balanced training set
ends with 342 consecutive `no` answers. Deterministically shuffling the same
records raises accuracy to 78.125%, with exact saved-adapter reload, but the
paired improvement is not significant (19 wins, 11 losses, p=0.10024421).
Lower learning rate and eight-example gradient accumulation also fail the
unchanged quality gates. The all-linear trial stops at the resource guard's
swap-growth limit before producing a quality result or publishing its output.
These are development-selection results on 256
prompts; the reserved holdout remains unopened. No SFT recipe is qualified by
these results; lower training loss alone does not establish answer quality.

A separate read-only evaluation of the all-linear staging adapter reaches
80.078125% complete-answer accuracy (23 wins, 10 losses, p=0.01754102), passing
the development answer gate. The CLI now releases training resources before
loading the independent final-evaluation model. The full rerun reproduced the
same adapter but still exceeded the swap-growth guard during final evaluation.
Standalone native evaluation passed all 256 examples with loss 0.24053803 and
zero fallback, while reproducing the memory growth investigated below.
Metal buffer and encoder lifetime corrections remove unpooled references.
Further allocation tracing identified a partially executed LoRA Q/K/V region
whose fallback overwrote an owned Q output. Checking every branch's inputs
before execution preserves exact loss on all 256 examples and reduces peak
physical memory from 8.67 GB to 2.69 GB, with no sampled swap growth. The fixed
binary completes all 1,960 updates, both 256-example evaluations, publication,
and independent 256-example reload. The published adapter matches the previous
training trajectory byte for byte; final and reload loss are both exactly
0.2405380331704805. The full command peaks at 4.23 GB physical memory with no
sampled swap growth and zero strict Metal violations. This closes the measured
E2B memory/publication regression.

All three E2B SFT development seeds now pass the frozen full-size recipe:
1,960 updates per seed, publication, exact reload and 256-prompt answer quality.
Complete-answer accuracy is 80.078125%, 80.078125% and 79.296875% for seeds
42, 17 and 991, versus 75% baseline. Each paired test passes p <= 0.05;
the aggregate test averages seed effects within each prompt and passes at
p=0.04807088 (28 wins, 16 losses, 212 ties). Seed 991's first attempt was stopped
for host paging; its isolated retry used identical inputs and resource limits.
The accepted seed-42 recipe also passes final-checkpoint publication recovery:
SIGTERM after all 1,960 updates, followed by resume, reproduces the complete
checkpoint and adapter byte for byte and final evaluation loss exactly.
Peak sampled physical memory is 4.25 GB with no swap growth; existing swap and
pageouts keep this functional result separate from zero-paging qualification.
E4B seed 42 has completed all 1,960 updates and both 256-example evaluations,
reducing evaluation loss from 2.80485258 to 0.12501888 with zero strict Metal
violations. Peak physical memory is 5.96 GB. Independent reload reproduces the
final loss exactly. MLX answer evaluation was stopped before producing a
result when host swap grew by 479.88 MiB. A later isolated base-load probe
completed without swap growth, but one fresh full evaluation attempt hit the
same guard. E4B answer quality needs more host memory headroom; no result or
quality pass was produced. This loss result does not establish task quality.
E4B qualification is now deferred. E2B multi-token recovery, GRPO quality and
sealed acceptance remain open, alongside the other release gates below.

The accepted E2B seed-42 adapter also passes real-model materialization and
independent verification of all 2,011 exported tensors. All 276 LoRA targets
meet the fixed FP32 arithmetic and BF16 rounding bounds; the other 1,735
tensors are byte-identical to the base checkpoint. The verifier resolves
Antfly's per-layer projection names to their HF counterparts without changing
the arithmetic bounds. Merged E2B generation also passes all 128 exact-token
comparisons against pinned MLX across raw and public-chat requests with F32
and F16 KV caches. All 132 HTTP comparisons also pass: streaming and
non-streaming text, finish reasons, token counts, and repeated prompts agree
on the resident server. Sampled peak process-group memory is 0.93 GB, and the
owned server shuts down and is reaped. This qualifies the recorded E2B export
and serving integration workload; E4B and full release gates remain open.

Release still requires hosted CI for the submitted revision, fresh real-model
quality/parity and interruption/recovery checks, a separate sealed holdout,
and clean-host memory/performance qualification. GRPO additionally needs the
full-size multi-seed quality campaign. A host with existing swap can produce
diagnostic evidence, but cannot establish the zero-paging release requirement.

See [the remediation ledger](GEMMA4_REVIEW_REMEDIATION.md) for current fixes,
validation, and outstanding gates, [the oracle contract](GEMMA4_ORACLE.md) for
qualification requirements, and [the historical evidence index](GEMMA4_HISTORY.md)
for older results. Historical metrics qualify only their recorded source.

Distributed training, multimodal/projector training, public `qlora-sft`, and
Gemma4 MoE models are outside this supported path. Packed Q4_0/Q4_K/Q6_K
frozen-linear gradient kernels exist, but direct-GGUF SFT/DPO/GRPO remains
experimental. Direct-GGUF plus incremental-KV GRPO is rejected after exact
token divergence. MLX is a comparison backend, not an Antfly training backend.

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

SFT consumes prepared examples in file order; `--seed` does not shuffle them.
Audit class/order clustering before training, even if the full dataset is
balanced. For a reproducible fixed shuffle of a `gemma_chat/v1` training split
with unique string `id` fields, run this from the repository root:

```bash
python3 zig/pkg/inference/scripts/gemma4/shuffle_gemma4_sft_dataset.py \
  --dataset train.jsonl --output train-ordered-42 --seed 42
```

Prepare `train-ordered-42/train.jsonl` with the public dataset command, then use
that prepared artifact for training. The tool preserves every record, sorts by
a hash of seed and ID without consulting labels, and writes a hash-bound
manifest. It refuses an existing output directory. This is one fixed order,
reused across epochs; prepare evaluation and sealed splits independently.
Measure task answers as well as total loss: assistant terminators and trailing
formatting tokens contribute to SFT loss and can dominate its improvement.

For the locked rank-16 yes/no development workload, the reusable answer check
uses a pinned MLX environment and its runtime/source archive attestation:

```bash
"$MLX_PYTHON" zig/pkg/inference/scripts/gemma4/evaluate_gemma4_sft_answers_mlx.py \
  --model /models/gemma-4-E2B-it --model-id gemma-4-E2B-it \
  --prepared-eval eval.prepared.json --expected-examples 256 \
  --initial-adapter adapter-initial --trained-adapter adapter-trained \
  --mlx-attestation mlx-preflight.json --output answer-quality.json
```

It rechecks the pinned model, MLX wheel, and MLX-LM source archive; loads exact
adapter tensors; and evaluates the prepared prompt IDs without retokenization.
Acceptance requires improved greedy answers ending immediately with end-of-turn,
an exact paired improvement test at p ≤ 0.05, and no forced-choice regression.
Exit status 1 retains a failing quality report. This narrow task diagnostic
does not replace native evaluation/reload, cross-backend parity, multi-seed
quality, or sealed acceptance. Use development data while choosing a recipe.

For multi-token extractive QA, `prepare_gemma4_squad_sft.py` selects development
examples from the official SQuAD 1.1 training JSON, separating articles and
passages between training and evaluation. It leaves the official development
set reserved. Shuffle the selected training records by seed before native
preparation; verify zero truncation on the resulting prepared files.
`evaluate_gemma4_squad_mlx.py` accepts those prepared prompts, the original
evaluation JSONL, reference answers, and selection manifest. It verifies both
raw-file and native-domain source hashes, generates complete greedy answers,
and compares cached versus uncached generation on eight prompts per adapter.
`score_gemma4_squad_sft.py` measures exact match and token F1 and groups paired
significance by article. Three-seed aggregation requires every seed to pass
and averages changes within each article. The full E2B development campaign
passes for seeds 17/42/991 with 1,960 updates each and 256 evaluation questions
from 88 articles. Mean F1 improves from 78.1798% to 83.7765% and mean exact match
from 54.2969% to 69.0104%; all answers complete and the aggregate article test
passes at p = 0.006966938. Independent rescoring, exact native reloads and all
48 cached/uncached checks pass. These results do not establish sealed acceptance
or clean-host performance; multi-token recovery remains pending. See
`multi-token-squad/e2b/independent-campaign-review.json` in the current evidence
root.

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
{"schema":"gemma_chat/v1","id":"example-1","messages":[{"role":"user","content":"Reply briefly."},{"role":"assistant","content":"Okay."}]}
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

The v3 recovery qualifier also accepts `--epochs 1 --interrupt-after-epoch 1`
to test interruption after the final checkpoint is durable and before output
publication. It requires byte-identical final trainer checkpoints and adapters,
exact terminal evaluation metrics, and zero evaluation fallbacks. Pass
`--expected-final-adapter-sha256 <64-hex-digest>` to require the uninterrupted
control to reproduce an already evaluated adapter before attempting recovery.
This final-boundary case proves publication recovery; recovery between training
epochs remains a separate trajectory check.

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

A qualification-only reverse converter now accepts an explicit Antfly origin
and translates a stock checkpoint back to its internal tensor inventory. A
nonzero CPU fixture passes native validation with every tensor payload and
the origin initialization seed preserved; 21 checks cover malformed inputs,
provenance mismatches, and exclusive publication. Plain stock checkpoints
without export sidecars retain explicitly caller-asserted training lineage.
This helper is not a public import command or real-model numerical evidence.
Its evidence is `peft-import-validation.json` under
`.benchmark-assets/gemma4-sft-production-20260923/`.

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
  failed threshold leaves the trained adapter unpublished. Task reports
  bind execution mode, format, and the passing evaluation summary, while the
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

Sigmoid DPO reports preference accuracy: the fraction of pairs whose
policy-versus-reference chosen/rejected log-probability margin is strictly
positive. An unchanged policy ties the reference and reports zero. This metric
does not measure the base model's answer accuracy on the task.

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
contract is epoch-boundary recovery with one mutable checkpoint. Historical
E2B/E4B BF16 Metal and explicitly admitted E2B Q4_0 GGUF runs passed real
process-kill/resume with byte-identical adapters. Current-source recovery
evidence and its scale are recorded in the remediation ledger.

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
| BF16 correctness | Unquantized BF16 Safetensors | Verify forward, loss, gradients, optimizer updates, and adapter save/reload against the pinned reference | Text SFT and sigmoid DPO are production candidates. Current-source quality, parity, recovery, clean-host memory/performance, and hosted CI remain release gates. GRPO remains experimental; see the remediation ledger for bounded diagnostics. |
| Q4 deployment / QLoRA substrate | QAT Q4 GGUF used by serving | Verify target names, adapter application, memory bounds, and post-training task quality on the deployed base | Packed input-gradient kernels and historical E2B recovery diagnostics are implemented. Direct-GGUF training remains experimental pending its own parity, memory, and quality qualification. Public `qlora-sft` and direct-GGUF plus incremental-KV GRPO remain rejected. |

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
