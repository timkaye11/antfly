# Gemma 4 fine-tuning

Antfly supports optimizer-backed Gemma 4 E2B and E4B LoRA training through
the native autodiff runtime. Text SFT, DPO, and GRPO use the same strict
artifact, backend-residency, and publication contracts.

## Supported scope

- BF16 Hugging Face Safetensors bases for training
- native CUDA and Metal execution
- text SFT, DPO, and GRPO
- image/audio projector-backed GRPO where the recipe supplies a compatible
  projector
- PEFT-compatible LoRA adapter export
- immutable final adapter publication and atomic report publication
- bounded checkpoint/resume for the SFT command

GGUF remains an inference and deployment format. Packed GGUF training fails
closed until quantized-base backward and optimizer semantics have a separate
production qualification.

## Text training flow

The typed CLI separates data preparation, adapter bootstrap, training, and
validation. Output paths are immutable and must not overlap input artifacts.

```sh
antfly inference finetune dataset prepare gemma4-lora \
  --model /models/gemma4 \
  --dataset train.jsonl --split train \
  --out prepared-train.json --dataset-revision TRAIN_REVISION

antfly inference finetune dataset prepare gemma4-lora \
  --model /models/gemma4 \
  --dataset eval.jsonl --split eval \
  --out prepared-eval.json --dataset-revision EVAL_REVISION

antfly inference finetune adapter bootstrap gemma4 \
  --model /models/gemma4 --out adapter-bootstrap \
  --rank 16 --alpha 32 --target-preset text-all-linear

antfly inference finetune train gemma4-lora \
  --model /models/gemma4 --adapter adapter-bootstrap \
  --train-prepared prepared-train.json \
  --eval-prepared prepared-eval.json \
  --out trained-adapter --trainer autodiff --backend cuda

antfly inference finetune adapter validate gemma4 \
  --model /models/gemma4 --adapter trained-adapter

antfly inference finetune adapter export gemma4-peft \
  --model /models/gemma4 --adapter trained-adapter \
  --out trained-adapter-peft
```

Use `--backend metal` on Apple Silicon. Device-backed training is strict:
unsupported operations, runtime promotions, unexpected host transfers, or
host optimizer fallback fail the run.

## Dataset contract

Chat rows use `messages` with `system`, `user`, `assistant`, and `tool` roles.
Only assistant content contributes to the causal loss. Prepared inputs bind
the tokenizer, chat template, raw source bytes, split, sequence limits,
vocabulary, and provenance groups. Train and evaluation inputs must be
separate and cross-split duplicates are rejected.

Multimodal rows may include image/audio placeholders and their corresponding
media paths. The projector must match the selected base model and requested
modalities.

## DPO and GRPO recipes

Run preference jobs through the unified recipe command:

```sh
antfly inference finetune run dpo-recipe.json
antfly inference finetune run grpo-recipe.json

# Reuse one admitted model/backend session across multiple jobs.
antfly inference finetune run-suite \
  --report preference-suite.json dpo-recipe.json grpo-recipe.json
```

DPO supports `text-preference` and `rendered-text-preference` rows. GRPO
supports `text-grpo` and `rendered-text-grpo` rows with group sizes from 2
through 8. The `ranked-first` reward mode is deterministic and intended for
infrastructure qualification; production reward modes should encode the
actual task objective.

Preference training requires finite hyperparameters, a positive DPO beta,
nonzero optimizer steps, and measurable LoRA tensor movement. GRPO additionally
requires reward variation and a nonzero policy-gradient signal; a KL-only
update is not accepted as successful training.

The CUDA shared-prompt, one-token DPO route uses the qualified coalesced sparse
objective by default. General multi-token Metal DPO uses detached
adapter-gradient branches and combines them on device before one optimizer
flush. These are implementation policies, not ambient environment settings.

## Adapter and report artifacts

A successful output bundle contains:

- `adapter_model.safetensors`
- `adapter_config.json`
- `antfly_adapter_manifest.json`
- tokenizer/chat-template sidecars when present
- `antfly_preference_run_report.json` for DPO/GRPO

The embedded preference report is published in the adapter transaction before
the external run report is replaced. Readers therefore never observe a final
adapter without its execution evidence. Existing final paths are never
silently replaced.

Preference reports include optimizer and microbatch counts, backend execution
evidence, fallback/transfer counters, sequence-bucket telemetry, reference
parity, and exact trainable-tensor movement. Suite reports separately record
model-admission time and per-job execution time so session reuse is auditable.

## Production controls

Only qualified rollback and resource controls remain environment-configurable:

- `ANTFLY_GEMMA4_DPO_COMPLETION_FENCED_CACHE=0` disables the Metal completion
  workspace cache.
- `ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE=0` disables qualified E2B Metal
  planned-encoder buffer reuse. Unknown topologies remain fail-closed.
- `TERMITE_METAL_COMPLETION_CACHE_MAX_MB`,
  `TERMITE_METAL_COMPLETION_CACHE_MAX_BUFFER_MB`, and
  `TERMITE_METAL_COMPLETION_CACHE_MAX_SLOTS` bound the Metal completion cache.

Benchmark harness flags are set by the checked-in qualification scripts and
are not training configuration.

## Validation

The focused release gate is:

```sh
TERMITE_REQUIRE_CUDA_TESTS=1 zig build test-gemma4-finetune -j1 \
  -Dmetal=false -Dcuda=true -Donnx=false -Dpjrt=false \
  -Doptimize=ReleaseSafe --summary all
```

It covers BF16 CUDA optimizer execution, DPO/GRPO objective gradients,
fail-closed recipe admission, adapter publication, report telemetry, and core
Gemma 4 graph/projector contracts. Real-model qualification is provided by
`scripts/run_gemma4_cuda_preference_smoke.py`; framework comparison tooling is
documented in [GEMMA4_ORACLE.md](GEMMA4_ORACLE.md).

## Current limitations

- training requires BF16 Safetensors; GGUF QLoRA is not admitted
- DPO/GRPO references must resolve to the same base model artifact
- preference checkpoint/resume and scheduler/warmup options remain rejected
  until their exact semantics are implemented
- multimodal SFT/DPO and broad multimodal GRPO quality qualification remain
  separate release gates
- production deployments should independently capacity-test E2B/E4B sequence
  lengths, adapter target sets, and multimodal workloads
