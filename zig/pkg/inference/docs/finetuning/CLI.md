# Fine-Tuning CLI

> Current state: `antfly inference finetune` is the public dispatcher. The
> explicitly qualified Gemma4 text-only BF16 LoRA-SFT and DPO lanes have typed
> data, adapter, optimizer, held-out-evaluation, and epoch-boundary recovery
> contracts. E2B GRPO passes a bounded three-seed/eight-epoch absolute-floor
> gate; E4B GRPO misses its predeclared top-rank floor and remains
> quality-blocked. Real E2B and E4B Metal process-kill/resume campaigns pass
> with byte-identical final adapters. This is a production-candidate boundary,
> not a blanket readiness claim: required CI, independent-initialization and
> baseline-relative quality, deployment materialization, and public QLoRA
> gates remain open.

The final preference qualification is tied to the ReleaseFast Metal
binary SHA-256
`6e0dde73f07fd7bbbb7ef5d953979580c52fd9d7b91a1d9fe323b012965d3402`.
Its quality campaigns, direct-GGUF and incremental-KV recovery boundaries,
matched MLX profiles, and preserved E4B GRPO failure are recorded in
[GEMMA4.md](GEMMA4.md).

## Goal

Antfly inference fine-tuning should have one durable user-facing CLI and one typed
programmatic workflow API. The build graph should build and run those entry
points, not define the product surface.

The durable shape is:

- `antfly inference finetune ...` is the public interface.
- `src/finetune/cli/` only parses arguments and dispatches.
- `src/finetune/workflows/` composes typed operations directly.
- `zig build` exposes a small number of developer entry points.
- Tests call typed functions or fixed test roots, not subprocess chains.

There is no legacy-compatibility requirement for the existing many-step
`zig build <finetune-tool>` surface. The refactor should optimize for the
long-term shape, not alias preservation.

## Remaining Refactor Work

Some older family paths still mix several concerns:

- many standalone build steps for tools, train/eval runners, eval commands, and
  workflow runners
- workflow runners that shell back out to `zig build`
- generated pilot datasets living near fixed test fixtures
- per-command argument parsing embedded directly in train/tool implementation
  files
- build-step names acting as the CLI contract

This makes it hard to answer basic questions:

- Which commands are public?
- Which commands are internal workflow pieces?
- Which datasets are canonical fixtures versus generated smoke data?
- How should a test run the same logic as a CLI without spawning a command?
- Where should a new model family's fine-tuning pipeline be added?

The remaining issue is not naming. Older build tools, testdata generation, and
workflow code still share parts of one surface. New Gemma4 work should use the
typed operations rather than add another subprocess/build-step API.

## Target CLI

The public surface should be hierarchical and task-oriented:

```sh
antfly inference finetune dataset inspect <family> ...
antfly inference finetune dataset generate <generator> ...
antfly inference finetune dataset prepare <family> ...

antfly inference finetune adapter bootstrap <family> ...
antfly inference finetune adapter inspect <family> ...
antfly inference finetune adapter materialize <family> ...
antfly inference finetune adapter compose ...

antfly inference finetune train <family-or-task> ...
antfly inference finetune eval <family-or-task> ...

antfly inference finetune workflow <workflow-name> ...
```

The unified dispatcher is now the only `antfly inference finetune` entry from the
top-level CLI. It routes recipe execution through `antfly inference finetune run` and
`antfly inference finetune smoke-fast`, routes hierarchical commands through
`src/finetune/cli/root.zig`, and still accepts legacy tool names as
compatibility wrappers:

```sh
antfly inference finetune run /tmp/recipe.json
antfly inference finetune dataset prepare colqwen2 /models/colqwen2 /data /tmp/examples.jsonl /tmp/prepared.json
antfly inference finetune prepare-colqwen2-inputs /models/colqwen2 /data /tmp/examples.jsonl /tmp/prepared.json
```

Concrete examples:

```sh
antfly inference finetune dataset generate gemma4-pilot /tmp/pilot.jsonl --count 1000 --split train
antfly inference finetune dataset prepare gemma4-lora \
  --model /models/gemma4 --dataset /data/train.jsonl --split train \
  --out /tmp/prepared-train.json --dataset-revision TRAIN_REVISION \
  --max-examples 1000 --max-seq-len 512
antfly inference finetune dataset prepare gemma4-lora \
  --model /models/gemma4 --dataset /data/eval.jsonl --split eval \
  --out /tmp/prepared-eval.json --dataset-revision EVAL_REVISION \
  --max-examples 128 --max-seq-len 512

antfly inference finetune adapter bootstrap gemma4 \
  --model /models/gemma4 --out /tmp/adapter \
  --rank 16 --alpha 32 --target-preset peft-qv
antfly inference finetune train gemma4-lora \
  --model /models/gemma4 --adapter /tmp/adapter \
  --train-prepared /tmp/prepared-train.json \
  --eval-prepared /tmp/prepared-eval.json \
  --out /tmp/out --trainer autodiff --backend metal
antfly inference finetune eval gemma4-lora \
  --model /models/gemma4 --adapter /tmp/out \
  --prepared /tmp/prepared-eval.json --out /tmp/eval.json --backend metal
antfly inference finetune adapter validate gemma4 \
  --model /models/gemma4 --adapter /tmp/out
antfly inference finetune adapter export gemma4-peft \
  --model /models/gemma4 --adapter /tmp/out --out /tmp/out-peft
antfly inference finetune adapter materialize gemma4 /models/gemma4 /tmp/out /tmp/merged

antfly inference finetune workflow gemma4-pilot text /models/gemma4 /tmp/pilot-run --count 1000 --backend metal
antfly inference finetune workflow gemma4-recursive-lora-smoke /models/gemma4 /tmp/recursive-smoke --count 16
antfly inference finetune workflow gliner2-entity-cleanup-smoke /models/gliner2 /tmp/adapter train.jsonl eval.jsonl /tmp/out
```

### Gemma4 command contract

- Named flags are canonical for prepare, bootstrap, train, eval, adapter
  validation, and PEFT export. The accepted positional spellings on older
  operations normalize into the same typed operation and exist for one release
  only; PEFT export is named-only. Do not put positional forms in new automation.
- Prepare emits an immutable `gemma4_prepared/v6` file. Training requires a
  separately prepared eval file and rejects token/source/group overlap.
- Text LoRA-SFT can opt into independent-row scheduling with
  `--sequence-length-bucket-quantum`, optional
  `--sequence-length-bucket-min`, and `--graph-cache-capacity` (maximum 8).
  Gemma4 DPO recipes accept the corresponding `runtime` fields under a distinct
  pair-safe policy: chosen and rejected branches share the rounded maximum
  sequence length and weighted-target shape for reference scoring and every
  policy update. Bucketed DPO retains four graphs by default; an explicit
  capacity may raise that bound to eight. Fixed-shape scheduling remains the
  default. GRPO and
  multimodal paths do not inherit either policy. The SFT training report
  separates initialization, evaluation, epoch, and adapter-save wall time;
  each epoch also reports monotonic throughput and graph-cache deltas. DPO
  reports record logical/scheduled/fixed rows and graph-cache phase snapshots.
- `native` and `metal` are the only Gemma4 training/eval backends. MLX-LM is a
  benchmark reference, not a value accepted by `--backend`.
- Gemma4 Metal train and eval select the strict training executor themselves;
  no hidden executor environment opt-in is required. Explicit diagnostic
  disable/parity flags still fail closed. Any native/unsupported partition,
  fallback, undeclared upload, runtime promotion, host output, or host gradient
  fails before optimizer mutation.
- Standalone `eval gemma4-lora` is loss-only and atomically publishes one
  immutable report. It does not allocate gradients or Adam state.
- `adapter_config.json` contains standard PEFT constructor fields;
  `antfly_finetune_manifest.json` contains Antfly model provenance, exact target
  policy, recursive metadata, internal tensor-key format, and the adapter
  checkpoint size/SHA-256. The training artifact intentionally retains Antfly's
  internal tensor keys. `adapter export gemma4-peft` validates it against the
  exact base, translates only tensor names, preserves payload bytes, and
  atomically publishes a stock-key PEFT directory plus
  `antfly_peft_export.json` provenance.
- Text LoRA training supports one atomically replaced epoch-boundary recovery
  file through `--checkpoint-path`, `--checkpoint-every-epochs`, and `--resume`.
  `--resume` must repeat the exact run contract in a new immutable `--out`
  directory. Recipes support `checkpoint.every_epochs` and
  `checkpoint.resume_path`; `keep_last` remains rejected. Optimizer-backed DPO
  and GRPO additionally publish a content-addressed aggregate sidecar before
  the checkpoint becomes visible. Their run fingerprint binds model, adapter,
  train/eval data, optimizer/reward contract, and the resolved Metal numerical
  policy. The v5 fingerprint and v2 policy include sparse-loss/CCE geometry,
  eager LoRA/dot routes, and graph-executor fusions; unrecognized Gemma/Metal
  training environment controls fail closed. The product embeds the same
  `gemma4_preference_environment.policy` that the recovery and quality
  qualifiers read and hash. Qualification removes every inherited
  `TERMITE_*` and `ANTFLY_GEMMA4_*` override before installing its strict
  bindings; graph-output ownership/elision, host-mirror resync, and paged-KV
  kill switches are explicitly denied. Recipe planning performs this check
  before printing a plan or publishing a manifest.
- Gemma4 mutable recipe paths are canonicalized through their deepest existing
  filesystem ancestor before overlap checks. An output rooted below a symlink
  into the model, reference model, datasets, seed adapter, checkpoint, reward
  artifacts, or planned reports is rejected before any output is created.
  Lexically distinct aliases do not bypass the immutable-input boundary.
- Checkpointed incremental-KV GRPO
  is admitted only at a completed epoch boundary: the sidecar stores cumulative
  sampler telemetry, while checkpoint publication drains the decoder frame and
  requires zero live logical or device KV sequences. Transient KV pages are
  deliberately rebuilt after resume.
- Packed Q4_0/Q4_K/Q6_K frozen-linear input gradients are implemented and
  executor-tested on Metal. Explicitly gated canonical full-prefix direct-GGUF
  E2B SFT, DPO, and GRPO research lanes have real optimizer and
  process-kill/resume evidence, but public `qlora-sft` remains fail-closed
  pending memory, cross-framework, reload, and task-quality qualification.
  Direct-GGUF GRPO combined with incremental KV is separately rejected because
  its Q4_0 paged decoder did not preserve the canonical token trajectory.

The authoritative support boundary and release gates live in
[GEMMA4.md](GEMMA4.md). Oracle and benchmark commands live in
[GEMMA4_ORACLE.md](GEMMA4_ORACLE.md).

## Production Support Matrix

Use this matrix as the PR gate for declaring the unified CLI production ready.

| Family / Task | Dataset | Adapter | Train/Eval | Materialize | Required Backend Lane |
| --- | --- | --- | --- | --- | --- |
| Gemma4 text LoRA | prepared v6 train/eval with causal multi-turn labels plus source and group identity | bootstrap/inspect/validate plus immutable stock-key PEFT export with hash-bound provenance | typed BF16 supervised autodiff + standalone loss-only eval + epoch-boundary recovery; direct QLoRA fail-closed | PEFT adapter export; bounded diagnostic merge + recursive base; production-scale streaming merge pending | native correctness and strict Metal; HF/PEFT oracle + same-Mac MLX-LM reference |
| Gemma4 text DPO / GRPO LoRA | text or rendered-text preference/prompt JSONL; GRPO supports decoded-text and token reward targets | bootstrap/inspect/validate plus immutable stock-key PEFT export | optimizer-backed live-logprob DPO/GRPO with pair/group-safe accumulation; compiled-graph zero-LoRA references and parity gates; content-addressed epoch recovery; fresh-backend terminal eval; GRPO hard/adaptive KL control; strict no-update and no-reward-advantage gates; packed-GGUF preference QLoRA fail-closed | PEFT adapter export | strict Metal E2B/E4B real process-kill/resume; byte-exact adapters/training/discrete traces, with explicitly bounded terminal GRPO KL floats; three-seed/eight-epoch absolute floors pass for E2B/E4B DPO and E2B GRPO, while E4B GRPO misses its top-rank floor; baseline-relative quality, independent initialization, repeated performance distributions, and required CI remain open |
| Gemma4 multimodal LoRA | historical diagnostic preparation only | bootstrap/inspect | public train/eval rejects projector/media | diagnostic only | unsupported production lane |
| ColQwen2 / Qwen2VL | multimodal prepared inputs | bootstrap/inspect | LoRA train/eval bundle | LoRA merge | native/BLAS CPU smoke |
| Qwen3.5 / Chandra OCR text-only | text SFT/DPO/GRPO JSONL; dynamic image preparation pending | bootstrap/inspect | Qwen autodiff trainer for text SFT/DPO/GRPO | adapter save; merged materialization pending | native/BLAS CPU smoke required, MLX/Metal smoke pending |
| GLiNER2 | dataset inspect + boundary caches | bootstrap/inspect | LoRA, autodiff, boundary heads | LoRA merge | native/BLAS CPU smoke |
| LayoutLMv3 | document token/sequence data | bootstrap/inspect | token and sequence train/eval | checkpoint materialize | native/BLAS CPU smoke |
| Reranker | dataset inspect + pooled/top-layer caches | bootstrap/inspect | head and LoRA surrogate paths | head and LoRA materialize | native/BLAS CPU smoke |
| Fused chunker | dataset fixtures | n/a | train/eval roots | checkpoint output | native/BLAS CPU smoke |

Optional lanes should prove MLX-LM reference performance, Metal, PJRT, ONNX, and quantized export where
the model family actually supports them. Unsupported combinations must fail
with explicit errors rather than falling back silently.

## PR Readiness Checklist

- `antfly inference finetune` reaches the unified dispatcher from `src/main.zig`.
- `antfly inference finetune run` and `antfly inference finetune smoke-fast` keep the recipe
  engine stable.
- Each command has a unique canonical tuple
  `<domain, action, subject>` and a unique legacy alias.
- Legacy aliases remain wrappers, not a second product surface.
- `zig build finetune -- <args>` runs the same binary and dispatcher as the
  installed CLI.
- `zig build test-finetune` and the root antfly inference test cover the dispatcher,
  recipe plan manifests, and synthetic smoke fixtures.
- CPU-only smoke paths are mandatory in CI; accelerator paths are separate
  opt-in lanes with clear skip behavior.

The CLI names should describe the user's intent first and the implementation
second. For example, `adapter materialize gemma4` is clearer than a top-level
`materialize-gemma4-lora` command because it sits next to `bootstrap`,
`inspect`, and `compose`.

## Build Steps

The build graph should stop being the fine-tuning command namespace.

Keep a small developer surface:

```sh
zig build finetune -- <args passed to antfly inference finetune>
zig build test-finetune
```

Optional narrowly scoped test steps are acceptable when they are real test
roots:

```sh
zig build test-finetune-data
zig build test-finetune-gemma4
zig build test-finetune-gliner2
```

Avoid adding build steps for every tool, trainer, materializer, cache preparer,
or workflow. Those belong under `antfly inference finetune`.

## Programmatic Contract

Every operation should expose a typed API. The CLI should be a thin wrapper
around that API.

Use this shape for tools, trainers, evaluators, and workflows:

```zig
pub const Options = struct {
    // command-specific inputs
};

pub const Result = struct {
    // stable summary suitable for JSON reports and tests
};

pub fn run(ctx: RunContext, opts: Options) !Result {
    // implementation
}
```

The `main()` for a command should do only four things:

1. Parse CLI arguments into `Options`.
2. Build a `RunContext`.
3. Call `run(ctx, opts)`.
4. Render output, status, and artifacts.

Workflow code should call these `run()` functions directly. It should not spawn
`zig build`, `termite`, or another subprocess for in-repo operations.

## Run Context

Use one shared context for all fine-tuning operations:

```zig
pub const RunContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.fs.Dir,
    artifact_writer: ArtifactWriter,
    backend_policy: BackendPolicy,
    logger: ?Logger = null,
};
```

The exact fields can follow existing Antfly inference conventions, but the important
properties are:

- paths resolve consistently
- output writing is centralized
- backend selection is explicit
- tests can provide temporary directories and deterministic contexts
- workflow steps can share context without reconstructing process state

## Source Layout

Organize by responsibility first, then by model family:

```text
src/finetune/
  core/
    run_context.zig
    contracts.zig
    artifact_writer.zig
    training_budget.zig
    training_guards.zig

  data/
    jsonl_resolve.zig
    streaming_dataset.zig
    document_data.zig
    gemma_chat_data.zig
    generators/

  adapters/
    lora.zig
    peft.zig
    lora_adapter_set.zig
    safetensors_checkpoint.zig
    compose.zig

  trainers/
    real_autodiff_trainer.zig
    graph_bridge.zig
    graph_input_binder.zig
    graph_weight_bridge.zig
    optimizers_ext.zig

  families/
    gemma4/
      data.zig
      adapter.zig
      prepare.zig
      train.zig
      eval.zig
      recursive_lora.zig
    gliner2/
    reranker/
    colqwen2/
    layoutlmv3/
    fused_chunker/

  workflows/
    gemma4_pilot.zig
    gemma4_recursive_lora_smoke.zig
    gemma4_recursive_lora_sweep.zig
    gliner2_entity_cleanup_smoke.zig
    gliner2_boundary_task_head_smoke.zig
    layoutlmv3_lora_smoke.zig

  cli/
    root.zig
    dataset.zig
    adapter.zig
    train.zig
    eval.zig
    workflow.zig

  test/
```

The families do not need identical internals, but they should expose consistent
operation names where possible:

- `prepare`
- `bootstrapAdapter`
- `inspectAdapter`
- `materializeAdapter`
- `train`
- `eval`

## Workflow Rules

A workflow is a typed composition of operations. For example, the Gemma4 pilot
workflow should directly call:

1. dataset generator or dataset loader
2. adapter bootstrap
3. input preparation
4. optional teacher target materialization
5. train/eval
6. artifact validation and summary writing

It should not construct command arrays like:

```zig
.{ "zig", "build", "prepare-gemma4-lora-inputs", "--", ... }
```

Subprocesses should be reserved for external tools outside this Zig package.

## Artifact Contract

Keep the existing idea of explicit run artifacts, but make it a first-class
fine-tuning contract rather than ad hoc JSON per command.

Every train or workflow run should write:

```text
<out_dir>/
  run_status.json
  training_config.json
  training_report.json
```

When applicable:

```text
<out_dir>/
  prepared.json
  prepared.teacher.json
  adapter_model.safetensors
  adapter_config.json
  antfly_finetune_manifest.json
  eval_report.json
  workflow_report.json
```

The reports should include:

- command or workflow name
- artifact contract version
- model family
- selected backend
- input paths and fingerprints where useful
- max examples, epochs, learning rate, and trainer mode
- before/after metrics
- output artifact paths
- validation failures or alerts

CLI output can be human-readable, but persisted reports should be stable JSON.

## Testdata Policy

Separate fixed fixtures from generated smoke data.

Use fixed fixtures under:

```text
pkg/inference/testdata/finetune/
  gemma4/
    smoke_train.jsonl
    smoke_eval.jsonl
  entity_cleanup/
    smoke_train.jsonl
    smoke_eval.jsonl
  reranker/
  gliner2/
```

Use deterministic generators for pilot data, but do not treat their outputs as
canonical checked-in fixtures unless a test specifically needs a frozen sample.

Generator commands belong under:

```sh
antfly inference finetune dataset generate ...
```

Tests should prefer fixed small fixtures. Real-model pilots should be explicit
integration workflows, not normal unit test steps.

## Testing Strategy

Use three tiers:

1. Unit tests for parsers, data loaders, adapter metadata, loss helpers, and
   artifact contracts.
2. Bounded integration tests using tiny fixed fixtures and synthetic or minimal
   model data.
3. Real-model workflows that are manually or CI-gated by model availability and
   backend support.

Tests should call typed `run()` functions where possible. CLI tests should be
limited to parser and dispatch behavior.

## Migration Status and Next Steps

The root dispatcher is wired, `zig build finetune -- ...` reaches that same
binary, and the production-intent Gemma4 prepare/bootstrap/train/eval/validate
path uses typed operations. The Gemma4 pilot and recursive workflows now carry
separate eval inputs and call the typed prepare/train operations for the
supported text lane.

Next steps, in order:

1. Put the current green focused Debug and real-Metal gates in required CI and
   archive the exact source, binary SHA-256, and qualification reports.
2. Extend the completed final-source E2B/E4B fixed/Q128 MLX-LM matrix into an
   alternating multi-process distribution rather than relying on one timing
   sample. Keep the completed three-seed/eight-epoch DPO and E2B GRPO
   absolute-floor campaigns as bounded evidence; diagnose E4B GRPO's failed
   top-rank floor, then add independent initializations and baseline-relative
   quality gates.
3. Remove the one-release Gemma4 positional bridge after the documented
   compatibility window.
4. Extend the completed E2B/E4B epoch-boundary recovery gate only when retained
   generations or mid-epoch scheduling justify the additional resumable state.
5. Extend the experimental canonical direct-GGUF E2B SFT/DPO/GRPO evidence to
   E4B with strict memory, parity, reload, and quality campaigns before enabling
   public `qlora-sft`; keep direct-GGUF plus incremental-KV GRPO rejected until
   the quantized paged decoder reproduces the canonical token trajectory.
6. Finish direct typed conversion of the remaining legacy Gemma4 materialize
   and recursive helpers, then remove redundant per-tool build steps.
7. Convert the remaining GLiNER2, reranker, ColQwen2, LayoutLMv3, and fused
   chunker subprocess adapters without changing their validated semantics.
8. Collapse `build/finetune/tools.zig` and `build/finetune/workflows.zig` once
   the CLI owns every public operation, retaining only focused test roots.

## Non-Goals

- Preserve every existing `zig build <finetune-tool>` name indefinitely.
- Keep subprocess-based workflow composition.
- Make generated pilot datasets the default unit-test fixture source.
- Force every model family into identical internals.
- Hide backend-specific behavior behind vague command names.

## Design Principle

Fine-tuning should have one public command tree and many typed internal
operations. The command tree should express user intent. The typed operations
should express reusable implementation boundaries. The build graph should only
assemble and run those entry points.
