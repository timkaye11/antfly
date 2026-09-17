# Gemma4 LoRA external oracle and performance contract

This directory defines the evidence boundary for comparing Antfly's Gemma4
LoRA trainer with independent implementations. It does not make the trainer
production-ready by itself. A release claim requires real E2B and E4B model
campaigns, successful trace comparison, checkpoint/resume evidence, and the
same-Mac performance matrix described below.

The reference stack is intentionally composite:

- Hugging Face Transformers plus PEFT is the numerical correctness oracle.
  It receives the exact `input_ids` and `labels` emitted by Antfly; it does not
  re-tokenize examples inside the numerical comparison.
- MLX-LM is the Apple Silicon performance reference. It is not the numerical
  correctness oracle because changing frameworks and kernels at the same time
  makes a discrepancy harder to localize.
- Unsloth is a useful optional NVIDIA throughput comparison, but it is not a
  same-Mac Metal baseline and is therefore outside the release gate here.

This split follows the primary implementation surfaces documented by
[Google's Gemma tuning guide](https://ai.google.dev/gemma/docs/tune),
[Transformers' Gemma4 documentation](https://huggingface.co/docs/transformers/model_doc/gemma4),
[PEFT's checkpoint-format guide](https://huggingface.co/docs/peft/main/en/developer_guides/checkpoint),
and [MLX-LM's LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

## Frozen inputs and artifact roles

`scripts/gemma4/gemma4_oracle.lock.json` is the single checked-in lock. It pins full
model revisions and file digests for `google/gemma-4-E2B-it` and
`google/gemma-4-E4B-it`, exact Python packages, upstream source commits,
target policies, numerical tolerance profiles, and the performance protocol.
Oracle commands fail if required files, dependency versions, revisions, or
hardware identities differ. Model loading is local-only and the tools do not
download models.

The release trajectory is also fixed to Zig's AdamW defaults: seed 42,
learning rate 0.001, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01,
gradient clipping at 1.0, and checkpoints after 1, 2, and 8 steps. The lock
contains exact per-suffix target counts for both models, so a q-only adapter
cannot be mislabeled as `peft-qv` and a partial all-linear adapter cannot enter
the release comparison.
The external HF trace lane is eager CUDA BF16 with FP32 LoRA parameters and
optimizer state. The exporter rejects CPU and non-BF16 execution; separate
diagnostic tooling would be required for those profiles.

The other artifacts have distinct roles:

- `gemma4_oracle_lock.schema.json`: machine-readable lock shape.
- `gemma4_oracle_publication.schema.json`: closed `COMPLETE.json` evidence
  ledger shape.
- `gemma4_oracle_trace.schema.json`: common HF/Zig numerical trace. It binds
  model identity, the exact prepared example, targets, logits, loss trajectory,
  raw gradients, optimizer moments, parameter updates, and adapter semantics.
  Release traces store external F32 Safetensors and bind each declared dtype to
  the actual tensor header; inline F64 tensors are synthetic-test-only.
- `gemma4_mlx_benchmark.schema.json`: one synchronized fresh-process benchmark
  observation. It binds the prepared-v6 artifact, source row, rendered chat,
  ordered input/label/mask workload digest, and complete optimizer-step token
  totals. The aggregator rejects mixed machines, workloads, or protocols and
  requires exact paired Antfly/MLX token counts.
- `gemma4_benchmark_campaign.schema.json`: the closed serial campaign ledger.
  It binds orchestrator-owned run identities and launch order to exact runner
  argv, non-overlapping wall-clock intervals, and the final byte size and
  SHA-256 of every sample admitted to a release comparison.
- `testdata/gemma4_oracle/contract_cases.json`: deterministic chat/tool case
  families. The generator expands these into group-disjoint train/eval/test
  JSONL without runtime randomness.
- `requirements-gemma4-oracle.txt` and
  `requirements-gemma4-mlx-reference.txt`: exact framework package manifests,
  checked against the JSON lock before a framework is imported.
- `requirements-gemma4-zig-oracle.txt`: the minimal NumPy/Safetensors subset
  used by the Zig trace packager and macOS contract tests. Lock validation
  checks its exact versions against the HF oracle package set without pulling
  the CUDA-only framework stack onto the Metal runner.

Only `gemma4_prepared/v6` is release-admissible. V6 binds causal generation
tokenization with one literal BOS, no implicit EOS, and all assistant turns.
The loader recomputes the
`gemma4_prepared_examples/v3` digest, requires per-record source/group/name and
record/rendered-chat digests, and re-hashes the immutable raw dataset split
using `gemma4_dataset_source/v1`. V4/v5 artifacts are deliberately rejected by
this oracle even though Antfly may retain them for migration and inspection.

## Adapter key boundary

Antfly and stock PEFT do not currently serialize identical tensor names:

| Artifact | Example LoRA A key |
| --- | --- |
| Antfly | `model.layers.N.self_attn.q_proj.weight.lora_A.weight` |
| Stock PEFT | `base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight` |
| Comparator identity | `model.layers.N.self_attn.q_proj:lora_A` |

An Antfly artifact must include `antfly_finetune_manifest.json` with schema
`antfly_gemma4_finetune/v2` or `antfly_gemma4_finetune/v3` and
`tensor_key_format: antfly_gemma4_adapter_keys/v1`. The sidecar owns the target
preset and base-model/tokenizer/chat-template provenance; `adapter_config.json`
remains a stock PEFT configuration. It also binds the exact
`adapter_model.safetensors` byte size and SHA-256 digest; validation recomputes
both before accepting the artifact. V3 additionally requires the deterministic
adapter-initialization seed, carries it into oracle inspection provenance, and
is the required form for independent-initialization quality campaigns. A stock
PEFT artifact has no sidecar, so
the caller must supply `--target-preset` explicitly. An artifact produced by
`antfly inference finetune adapter export gemma4-peft` remains a stock-key
PEFT checkpoint, but also carries `antfly_peft_export.json`. The oracle
validates that export sidecar's source and destination key formats, base-model
provenance, target preset, tensor count, file sizes, and file hashes before it
accepts the artifact.

The public export command is the immutable Antfly-to-PEFT product boundary;
it never modifies the source artifact. A pinned, local structural smoke test
loads that output directly with `PeftModel.from_pretrained`, saves it with
stock PEFT, reloads it, and requires exact tensor and logit equality:

```sh
python3 zig/pkg/inference/scripts/gemma4/verify_gemma4_peft_export_roundtrip.py \
  --antfly /path/to/releasefast/antfly \
  --work-dir /new/evidence/gemma4-peft-roundtrip
```

The smoke model deliberately has Gemma-compatible projection names but is not
a Gemma4 release model. Passing it proves the serialized key, configuration,
and PEFT load/save boundary; it does **not** prove real E2B/E4B fixed-logit or
generation parity. The HF oracle can still materialize a temporary one-way
`antfly_to_stock_peft_translation/v1` checkpoint for old internal artifacts.
Real-model Antfly-to-PEFT comparison and reverse PEFT-to-Antfly import remain
separate release gates.

## Offline contract checks

From the repository root, these commands use only the Python standard library
and do not fetch dependencies or models:

```sh
python3 zig/pkg/inference/scripts/gemma4/compare_gemma4_lora_hf_zig.py validate-lock
python3 -m unittest discover -v \
  -s zig/pkg/inference/scripts -p 'test_*gemma4*py'
```

Generate a deterministic contract dataset into a new directory, then verify
that regeneration is byte-identical:

```sh
fixture_dir="$(mktemp -d)"
python3 zig/pkg/inference/scripts/gemma4/generate_gemma4_oracle_fixture.py \
  --output "$fixture_dir/gemma4-contract.jsonl"
python3 zig/pkg/inference/scripts/gemma4/generate_gemma4_oracle_fixture.py \
  --output "$fixture_dir/gemma4-contract.jsonl" --check
```

Use `--train-count 1024 --eval-count 256 --test-count 256` for the production
contract campaign. Character-boundary cases are not described as token
boundaries; exact tokenizer parity is established by comparing the resulting
v6 prepared arrays.

## Real-model correctness campaign

First validate local assets. `--source-dataset` may relocate the immutable raw
split, but its digest must equal the value recorded by the v6 prepared file.

```sh
python3 zig/pkg/inference/scripts/gemma4/compare_gemma4_lora_hf_zig.py \
  validate-model --model-key gemma-4-E2B-it --model-dir /models/gemma-4-E2B-it
python3 zig/pkg/inference/scripts/gemma4/compare_gemma4_lora_hf_zig.py \
  validate-prepared --prepared /evidence/prepared-v6.json \
  --source-dataset /datasets/gemma4-contract.jsonl --example-index 0
```

In the exact pinned HF environment, export a reference trace. The target
preset is inferred from a valid Antfly sidecar; it is mandatory on the command
line for a stock PEFT seed adapter.

```sh
python3 zig/pkg/inference/scripts/gemma4/export_gemma4_lora_hf_oracle.py \
  --model-key gemma-4-E2B-it \
  --model-dir /models/gemma-4-E2B-it \
  --transformers-source /src/transformers-v5.5.2 \
  --peft-source /src/peft-v0.19.1 \
  --adapter /adapters/seed \
  --target-preset peft-qv \
  --prepared /evidence/prepared-v6.json \
  --source-dataset /datasets/gemma4-contract.jsonl \
  --example-index 0 --steps 1 \
  --seed 42 --learning-rate 0.001 --weight-decay 0.01 \
  --output-dir /evidence/hf-e2b-qv-step1
```

Both source directories are mandatory, must be clean at the commits in the
oracle lock, and must be the code actually imported by Python. Exact package
versions alone do not justify recording an upstream source revision.

Export the paired Zig trace from a clean Antfly checkout with the exact
release-built executable. The wrapper verifies the executable hash and embedded
source revision, re-hashes the locked local model, prepared-v6 artifact, raw
source split, and provenance-bound Antfly adapter, sanitizes all inherited
`TERMITE_*` and `ANTFLY_GEMMA4_*` settings plus the experimental GGUF admission
switch, and refuses to replace an output. `--metal-device` is an auditable
device label and is required for the Metal producer. Use
`--backend native` without that flag for the independent CPU implementation.

```sh
python3 zig/pkg/inference/scripts/gemma4/export_gemma4_lora_zig_oracle.py \
  --antfly /path/to/releasefast/antfly \
  --source-root /src/clean-antfly \
  --model-key gemma-4-E2B-it \
  --model-dir /models/gemma-4-E2B-it \
  --adapter /adapters/seed \
  --prepared /evidence/prepared-v6.json \
  --source-dataset /datasets/gemma4-contract.jsonl \
  --example-index 0 --steps 1 \
  --backend metal --metal-device "Apple M-series" \
  --output-dir /evidence/zig-metal-e2b-qv-step1
```

The product CLI's paired `--oracle-request` / `--oracle-capture-out` flags are
the private producer boundary, not the release workflow. They accept only the
locked 1-, 2-, or 8-step AdamW trajectory, repeat exactly one selected prepared
row, skip evaluation, and publish a separate closed capture containing the
final pre-update supervised-position logit probes, final raw F32 gradients,
post-update weights and Adam moments, and strict execution counters. A
read-only observer downloads only the final Metal raw-gradient surface without
rerouting any step away from the normal direct-device AdamW path. The wrapper
independently validates the
request echo, stable probe token projection, complete A/B inventory, checkpoint
counters and zero accumulation cursor, candidate/checkpoint weight equality,
all file hashes, and fallback-free Metal evidence. It then publishes the common
trace, candidate adapter, raw producer capture, child logs, and a final closed
`COMPLETE.json` ledger.

Once both traces have been produced, run the comparison:

```sh
python3 zig/pkg/inference/scripts/gemma4/compare_gemma4_lora_hf_zig.py \
  compare --reference /evidence/hf-e2b-qv-step1/trace.json \
  --candidate /evidence/zig-metal-e2b-qv-step1/trace.json \
  --profile hf-zig-bf16
python3 zig/pkg/inference/scripts/gemma4/compare_gemma4_lora_hf_zig.py \
  compare-adapters --reference /evidence/hf-e2b-qv-step1/reference_adapter \
  --model-key gemma-4-E2B-it \
  --reference-target-preset peft-qv \
  --candidate /evidence/zig-metal-e2b-qv-step1/candidate_adapter \
  --profile hf-zig-bf16
```

Run the 1-, 2-, and 8-step trajectories for both models and both target
presets. Also prove exact checkpoint/resume behavior with the `resume` profile.
Profiles have fixed producer roles: `hf-zig-bf16` accepts HF/PEFT as reference
and Zig Metal as candidate; `native-metal-bf16` accepts Zig native and Zig
Metal from the same clean commit and Mac; `resume` compares the same Zig
backend/build/hardware before and after recovery. Synthetic producers are
accepted only by `tiny-f32` tests and cannot create release evidence.
The checked-in BF16 tolerances are proposed gates, not empirical proof; they
must be calibrated from repeated real campaigns and tightened when evidence
supports it. Do not widen a threshold to make a single failure pass.

Every release evidence directory must end with a sorted, closed
`COMPLETE.json` ledger containing the SHA-256 and byte size of every other
regular file. Validators reject symlinks, nested uncommitted files, missing or
mutated payloads, and a trace or adapter directory outside a committed
publication. Numerical comparison recomputes the global raw-gradient norm and
checks per-surface max-absolute error in addition to relative L2 and cosine
similarity, so neither a false reported norm nor a localized outlier can be
diluted into a passing aggregate.

## Same-Mac MLX-LM performance campaign

The repository includes two fail-closed framework runners:

- `scripts/gemma4/run_gemma4_lora_mlx_benchmark.py` runs the pinned MLX-LM source in a
  fresh child process; and
- `scripts/gemma4/run_antfly_gemma4_lora_benchmark.py` runs a purpose-built Antfly
  executable through the typed Gemma4 benchmark-telemetry contract.

Each process consumes the same prepared-v6 row and byte-equivalent canonical
F32 LoRA A/B tensors, verifies the complete target inventory, synchronizes the
device at every optimizer boundary, and publishes one
`gemma4_mlx_benchmark.schema.json` sample with no-replace semantics.
The runner does not use MLX-LM's generic `--mask-prompt` data path, which treats
only the final chat message as the completion; it consumes Antfly's exact v6
labels so every assistant turn and masked interval is identical across engines.
MLX emits sample schema v4 plus a sibling `<sample>.precision.json` artifact conforming
to `gemma4_mlx_precision_evidence.schema.json`; the sample references the
artifact by SHA-256, and the artifact binds the finalized sample, exact runner
source, and native build receipt. Both runners also embed a
`antfly_gemma4_benchmark_producer_source/v1` manifest. That manifest binds the
runner entrypoint and its complete local Python/lock/requirements dependency
closure, including the campaign verifier and MLX build attester, to exact file
digests at one Git commit/tree. The producing checkout must be clean across the
whole repository, every listed file must be tracked and byte-equal to that
commit, and the entrypoint digest must agree with its inventory. The precision
artifact carries the identical MLX producer manifest rather than a weaker
runner-only hash. The precision artifact is durably published
before the sample, so a failed publication can leave only unreferenced
evidence, never a valid sample with missing initial evidence. The fixed
process trajectory is one cold compile-and-update window, one first-steady
window, three warmup windows, and twenty measured windows. Parent-owned 10 ms
Darwin physical-footprint sampling and two-way phase barriers keep Python
sampling and `vm_stat`/`memory_pressure` collection outside timed device work.
Paging, swap, control-protocol drift, partial accumulation windows, and missing
strict-Metal evidence fail the sample.

Antfly product version and benchmark source provenance are independent build
identities. Build benchmark binaries from the clean source revision with both
`-Dantfly-version=<product-version>` and
`-Dbenchmark-source-revision=<40-hex-HEAD>`. The runner queries the embedded
product version, binds it separately from the clean checkout's commit, and the
typed trainer admits telemetry only when both embedded values match. The
benchmark source revision defaults to `dev`, so ordinary development and
release binaries retain their normal product version and fail closed only if
someone attempts to use them for benchmark evidence without an explicit commit
attestation. The Antfly wrapper itself must reside at the admitted path inside
the exact `--source-root` checkout; an unrelated clean source tree is rejected,
and its producer-manifest revision must equal the executable's embedded
benchmark revision. MLX evidence separately binds the clean MLX and MLX-LM source
revisions, the imported Python roots, and a closed native inventory containing
the loaded core extension, `libjaccl.dylib`, `libmlx.dylib`, and
`mlx.metallib`, plus its build attestation. System frameworks reported only as
logical paths from the sealed dyld shared cache are not dereferenced; all three
required loaded Mach-O images still must resolve to the exact attested files.

The locked E2B/E4B checkpoints retain K/V tensors for the configured shared-KV
tail even though neither MLX-LM nor the current Hugging Face topology uses
them. The MLX runner does not use generic `strict=False` loading. It admits
only the exact config-derived unused shared-KV surplus, rejects every missing
or additional name, requires BF16 storage for every used checkpoint tensor,
then performs a strict shape-checked load of the complete used inventory.

Benchmark semantics v3 verifies BF16 base-model storage, F32 LoRA-parameter
storage, the exact logical F32 raw/accumulated/clipped gradient-tree inventory,
the exact materialized post-cold AdamW `m`/`v` names, shapes, and F32 storage,
and both the F32 loss-tensor and loss-reduction-input dtypes. These observations
do not prove activation or dispatched matmul-accumulator precision. Therefore
`activation_dtype` and `matmul_accumulator_dtype` remain the only
`not_asserted` fields. Sample v4 is diagnostic only, and the full-matrix/release
comparison fails closed while either field remains unasserted; these samples
cannot support a release performance claim.

The full matrix has 24 cells: E2B/E4B, `peft-qv`/`text-all-linear`, sequence
length 128/512/2048, and gradient accumulation 1/4 at microbatch 1, rank 16,
alpha 32. Every cell requires at least five paired fresh processes and
alternating framework order. Both
frameworks must report byte-equivalent Mac hardware identity and the same
locked model artifact, prepared-v6/workload identities, complete optimizer-step
input-token total, and supervised-token total.

Identity scopes are deliberate. `implementation.command_sha256` describes one
exact process invocation and must be unique across every sample in the
campaign; it is not an implementation identifier. Within one paired matrix
cell, case, semantic-contract, and protocol bytes form one invariant execution
contract across both frameworks and every repetition. Across the entire
campaign, Antfly must retain one producer manifest, product/build revision, and
executable digest; MLX-LM must retain one producer manifest, MLX/MLX-LM source
identity, native artifact inventory, build receipt, and Python executable
content identity. Absolute checkout, receipt, loaded-extension, and interpreter
paths are excluded from campaign identity only where their content digests and
relative runtime roles are already bound. A command digest collision, paired
execution-contract change, or campaign-wide implementation drift fails closed.

The smallest diagnostic cell is E2B, `peft-qv`, sequence length 128,
gradient accumulation 1, microbatch 1, rank 16, and alpha 32. Direct runner
invocation can exercise that cell, but self-supplied campaign metadata is not
release evidence. During development, `--diagnostic-only` additionally permits
an intentionally dirty Antfly producer checkout, does not enforce release
memory thresholds, and publishes only
`antfly_gemma4_mlx_diagnostic_sample/v1`. The payload sets
`release_eligible=false` and cannot validate as benchmark sample v4; never feed
it into the paired comparison. A production campaign starts from an immutable plan whose
command arrays omit `--campaign-id`, `--run-id`, `--repetition`,
`--sequence-index`, and `--output`; those fields belong exclusively to the
campaign orchestrator. A one-cell plan has this shape (the full release plan
contains all 24 locked matrix cells):

```json
{
  "schema_version": "antfly_gemma4_lora_benchmark_campaign_plan/v1",
  "repetitions": 5,
  "cells": [
    {
      "cell_id": "e2b-peft-qv-s128-ga1",
      "commands": {
        "antfly-zig-metal": [
          "/usr/bin/python3",
          "/src/antfly/zig/pkg/inference/scripts/gemma4/run_antfly_gemma4_lora_benchmark.py",
          "--lock", "/src/antfly/zig/pkg/inference/scripts/gemma4/gemma4_oracle.lock.json",
          "--antfly", "/build/antfly",
          "--source-root", "/src/antfly",
          "--model-key", "gemma-4-E2B-it",
          "--model-dir", "/models/gemma-4-E2B-it",
          "--adapter-dir", "/adapters/e2b-qv-seed",
          "--train-prepared", "/evidence/train-v6.json",
          "--eval-prepared", "/evidence/eval-v6.json",
          "--train-source-dataset", "/datasets/train.jsonl",
          "--eval-source-dataset", "/datasets/eval.jsonl",
          "--example-index", "0",
          "--target-preset", "peft-qv",
          "--sequence-length", "128",
          "--grad-accum", "1",
          "--metal-device", "Apple M4"
        ],
        "mlx-lm": [
          "/envs/gemma4-mlx/bin/python3.12",
          "/src/antfly/zig/pkg/inference/scripts/gemma4/run_gemma4_lora_mlx_benchmark.py",
          "--model-key", "gemma-4-E2B-it",
          "--model-dir", "/models/gemma-4-E2B-it",
          "--prepared", "/evidence/train-v6.json",
          "--source-dataset", "/datasets/train.jsonl",
          "--adapter", "/adapters/e2b-qv-seed",
          "--example-index", "0",
          "--target-preset", "peft-qv",
          "--sequence-length", "128",
          "--grad-accum", "1",
          "--mlx-source", "/src/mlx",
          "--mlx-lm-source", "/src/mlx-lm",
          "--mlx-build-attestation",
          "/src/mlx/build/antfly-gemma4-mlx/antfly-native-build.json"
        ]
      }
    }
  ]
}
```

Run the plan once into a new evidence directory:

```sh
python3 zig/pkg/inference/scripts/gemma4/run_gemma4_lora_benchmark_campaign.py \
  --plan /evidence/gemma4-release-plan.json \
  --output-dir /evidence/gemma4-release-campaign
```

For each cell, repetition zero runs Antfly then MLX; repetition one reverses
that order, and so on. Runs are launched serially. Only after every child has
successfully published its sample does the orchestrator atomically create the
no-replace `COMPLETE.json` marker. The marker records exact argv plus its
digest, start/completion timestamps, and the final sample digest and size.

The MLX build pipeline, not the benchmark process, must create the strict
attestation inside a Git-ignored build-output directory of the otherwise clean
pinned MLX checkout. Its exact fields are:

```json
{
  "schema_version": "antfly_mlx_native_build_attestation/v1",
  "source_revision": "<locked 40-hex MLX revision>",
  "source_clean": true,
  "native_artifact_inventory_sha256": "sha256:<closed inventory digest>",
  "build_command_sha256": "sha256:<exact build receipt digest>",
  "precision_policy_sha256": "<digest pinned by gemma4_oracle.lock.json>"
}
```

The runner verifies this bundle before importing `mlx.core`, confirms the loaded
extension's closed inventory, holds every receipt, log, tool, dependency-tree,
and native-file identity stable through the sample, then re-hashes the complete
bundle at postflight. It also disables Python bytecode writes before importing
either clean source tree. Create the bundle with the strict local
dependency/tool manifest; never substitute an ad hoc digest:

```sh
python3 zig/pkg/inference/scripts/gemma4/build_and_attest_gemma4_mlx.py build \
  --mlx-source /src/mlx \
  --build-output /src/mlx/build/antfly-gemma4-mlx \
  --inputs /evidence/mlx-native-build-inputs.json
python3 zig/pkg/inference/scripts/gemma4/build_and_attest_gemma4_mlx.py verify \
  --mlx-source /src/mlx \
  --build-output /src/mlx/build/antfly-gemma4-mlx \
  --inputs /evidence/mlx-native-build-inputs.json
```

The exact `antfly_mlx_native_build_inputs/v1` manifest pins a new build's job
count, deployment target, selected SDK version, canonical Xcode developer
directory, native `clang`/`clang++`/CMake/Ninja/Python executables, and local
source trees for metal-cpp 26, nlohmann/json 3.11.3, fmt 12.1.0, and nanobind
2.12.0. Use `file-digest` for each executable and `tree-digest` for each source
tree. Paths consumed through MLX's `CMAKE_ARGS` must not contain whitespace.
The strict shape is:

```json
{
  "schema_version": "antfly_mlx_native_build_inputs/v1",
  "jobs": 8,
  "macos_deployment_target": "14.0",
  "macos_sdk_version": "26.2",
  "metal_toolchain_identifier": "com.apple.dt.toolchain.Metal.32023.864",
  "developer_dir": "/Applications/Xcode.app/Contents/Developer",
  "user_home": "/Users/benchmark",
  "dependencies": {
    "fmt": {"path": "/deps/fmt-12.1.0", "tree_sha256": "sha256:<64-hex>"},
    "json": {"path": "/deps/json-3.11.3", "tree_sha256": "sha256:<64-hex>"},
    "metal_cpp": {"path": "/deps/metal-cpp-26", "tree_sha256": "sha256:<64-hex>"},
    "nanobind": {"path": "/deps/nanobind-2.12.0", "tree_sha256": "sha256:<64-hex>"}
  },
  "tools": {
    "clang": {"path": "/xcode/usr/bin/clang", "sha256": "sha256:<64-hex>"},
    "clangxx": {"path": "/xcode/usr/bin/clang++", "sha256": "sha256:<64-hex>"},
    "cmake": {"path": "/tools/bin/cmake", "sha256": "sha256:<64-hex>"},
    "ninja": {"path": "/tools/bin/ninja", "sha256": "sha256:<64-hex>"},
    "python": {"path": "/python/bin/python3.12", "sha256": "sha256:<64-hex>"}
  }
}
```

All paths must be canonical and non-symlinked; `clang` and `clangxx` must equal
the tools selected by the pinned developer directory's `xcrun`, and `python`
must be the interpreter running the command. The Git-ignored build output must
not already exist.

The builder sets `FETCHCONTENT_FULLY_DISCONNECTED=ON`, rejects a shadowed bare
build command, records the selected `xcrun` Metal/C/C++/link tools and child
Python setuptools/build-command identities, disables generated Python stubs, and safely
removes only the known ignored `include/`, `share/`, and `lib/cmake/` install
copies before closing the checkout over the four runtime artifacts. It publishes the fsynced
receipt and attestation atomically without replacement; a failed build retains
its log but never publishes an attestation. Archive and remove that failed
ignored output before retrying in a new directory, because checkout closure
rejects residue from an earlier attempt.

The current v2 closed inventory admits exactly `libjaccl.dylib`,
`mlx.metallib`, the Python extension, and `libmlx.dylib`. Oracle lock v3 requires
SDK 26.2 or newer, where pinned MLX 0.31.2 enables JACCL. The builder explicitly
selects the installed Apple Metal toolchain through the pinned `TOOLCHAINS`
identifier, preserves the real account home so `xcrun` can resolve that
component, disables the unused GGUF dependency surface, and forces C++ driver
link semantics for JACCL. It rejects an absent or different Metal component
rather than silently using another toolchain.

This is not yet a release-grade supply-chain root. FetchContent and pip are
configured not to fetch, but the command does not enforce OS-level network
isolation and upstream MLX invokes shell commands; run it in a deny-network CI
worker. Dependency/tool digests are operator-provided rather than allowlisted by
the checked-in oracle lock, the selected Xcode tools are hashed but the whole SDK
tree is not, and evidence is unsigned and still path-based rather than an
immutable content-addressed snapshot. Preserve the manifest and dependency
trees with the evidence, use one pinned Mac/Xcode image, and treat the resulting
samples as diagnostic until checked-in policy pins plus signed DSSE/SLSA-style
provenance and race-resistant staging exist.

```sh
python3 zig/pkg/inference/scripts/gemma4/bench_gemma4_lora_mlx_zig.py \
  validate /evidence/sample.json
python3 zig/pkg/inference/scripts/gemma4/bench_gemma4_lora_mlx_zig.py \
  compare \
  --campaign-manifest /evidence/gemma4-release-campaign/COMPLETE.json \
  /evidence/gemma4-release-campaign/samples/[0-9][0-9][0-9][0-9].json
```

The full comparison verifies that the manifest contains exactly the supplied
samples, every file still matches its final SHA-256 and byte size, recorded
argv owns the same campaign/run/repetition/sequence labels as the sample, run
intervals do not overlap, pairs are adjacent, and framework-first order
alternates by repetition. A missing, incomplete, relabeled, regrouped, or
mutated ledger fails closed. `--allow-partial` remains diagnostic and may be
used without a manifest; a partial campaign cannot support a release or broad
performance claim.

Two diagnostic-only MLX-LM E2B `peft-qv`, rank-16/alpha-32, sequence-128,
accumulation-1 samples completed on the 24 GB M4 Pro Mac mini on 2026-08-12.
Pinned MLX 0.31.2 and MLX-LM 0.31.3 loaded all four attested native artifacts,
materialized 540 BF16 base tensors, trained 100 F32 LoRA tensors with 200 F32
AdamW moment tensors, and completed twenty synchronized measured optimizer
steps per run. The final current-source rerun measured `0.263310/0.263396 s`
median/mean (`485.96` input tokens/s); its mean differed from the prior run by
less than 0.09%. Allocator/process peaks were `10.014/10.708 GiB`, swap was
zero, and the measured window incurred 1,097,728 bytes of page-ins plus 589,824
bytes of page-outs, so it would fail the release lane's zero-paging threshold.
The artifact is
`/private/tmp/antfly-gemma4-mlx-e2b-seq128-diagnostic-v2.json` with SHA-256
`eeb9b5abc07ab589b6cb58e4d1379a073098d0135a2ba02a4ad0b8092efda4e8`.
It is intentionally non-release evidence because the MLX producer checkout was
not admitted to a closed release campaign and the measured window incurred
paging.

A matching diagnostic-only Antfly E2B sample completed the same workload after
promoting the guarded 64-row simdgroup BF16 frozen-linear forward and
input-gradient products. Its twenty measured steps had
`0.568763/0.568748 s` median/mean (`225.06` input tokens/s), versus
`0.702321/0.702349 s` for the preceding 32-row simdgroup binary,
`1.418818/1.418835 s` for the 16x32-tiled binary, and
`2.169686/2.169801 s` for the original baseline. Peak physical footprint was
`1,774,864,856` bytes. The final artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-simdgroup-m64-final-diagnostic-v1.json`
with SHA-256
`06cf51a0128119d263326c0313fbc13be866189befcc97d5fc58efe662c7683f`.
It incurred 16,384 bytes of page-ins, and its source checkout was dirty, so it
is not release-admissible.

Within the sequence-128 diagnostic cell, the 64-row route is `1.235x` faster
than the preceding 32-row route, `2.495x` faster than the tiled route, and
`3.815x` faster than the original baseline. MLX-LM is still `2.159x` faster
than optimized Antfly, while MLX's process peak is `6.479x` larger. The initial
simdgroup input conversion introduces bounded
mixed-precision drift: the one-step adapter has relative L2 delta `0.001254`
and cosine `0.999999214` against the tiled route, while the update-vector
relative L2 delta is `4.09%` with cosine `0.999165`. The 64-row route is byte
identical to its 32-row rollback after both one and five updates. Its two
specializations can be disabled independently with
`TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64=1` and
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64=1`.

A genuine sequence-512, accumulation-1 diagnostic cell now covers both
frameworks. The final Antfly binary measured `1.566196/1.565846 s` median/mean
(`326.98` input tokens/s) with a `5,987,846,664`-byte peak, a `1.306x`
improvement over its preceding 32-row result. MLX-LM measured
`0.992220/0.990706 s` (`516.80` input tokens/s) with a
`15,999,361,256`-byte peak. Thus MLX is `1.581x` faster while using `2.672x`
the process footprint. The Antfly artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-simdgroup-m64-final-diagnostic-v1.json`
at SHA-256
`a23a5eefe5ecc48bdd53a63dd732d51730920f21d66be9132bc3640e398359ff`;
the MLX artifact is
`/private/tmp/antfly-gemma4-mlx-e2b-seq512-diagnostic-v1.json` at SHA-256
`d7d6a7518193963e7bfaf83ffaf289dd8af176f49791e16a1231dd80aa75513c`.
Both samples incurred page activity, so neither clears the strict memory gate.

The next diagnostic pass batches up to eight supervised sparse-loss rows per
tied vocabulary projection. This changes the locked eight-target workload from
eight complete BF16 embedding-table scans to one while bounding the additional
F32 logits workspace to 8 MiB. The final candidate binary has SHA-256
`c265e78692dd29da4bfa198e0e0e00e50412d769d7519dd2abf2ceca2cc28d9a`.
`TERMITE_GEMMA4_DISABLE_BATCHED_SPARSE_LOSS=1` is the one-row rollback, and
`TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=<1..64>` is a bounded diagnostic
override.

The sequence-128 diagnostic measured `0.470076/0.470017 s` median/mean
(`272.33` input tokens/s) with a `1,808,025,952`-byte process peak. This is a
further `1.210x` improvement over the M64 graph; MLX-LM is now `1.784x` faster
while using `6.359x` the process footprint. The artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`63d16c86dd9aaeb89699e1b38632a4a6d97edab0ae47e9e4b4c09037e3623052`.
It incurred 3,047,424 bytes of page-ins, so it remains diagnostic-only.

The sequence-512 diagnostic measured `1.466413/1.466469 s` median/mean
(`349.14` input tokens/s) with a `6,023,989,768`-byte peak. This is a further
`1.068x` improvement over M64; MLX-LM is `1.480x` faster while using `2.656x`
the footprint. The artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`1035fd111fc79e5d2103d02ba074f11531cacf5d03f1f65127e91eeee1c243e2`.
It incurred 8,404,992 bytes of page-ins and 16,384 bytes of page-outs, so it
also fails the zero-paging gate.

The optimization is training-state clean. One- and five-update eight-target
adapters are byte-identical to the one-row M64 graph, including final SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`.
A separate fifteen-target update differed by one F32 loss ULP, retained the
same gradient norm, and produced the same adapter SHA-256
`629a459767d7d826cc047b6cb6584e47021b1c046fa1e727da88ec19d20e71a0`.

The next bounded optimization specializes the batched vocabulary
input-gradient shape `8x262144 * 262144x1536 -> 8x1536`. The exact-arithmetic
Metal M8/N32/K64 route uses F32 accumulation in the generic path's K order and
is admitted only for eight-row BF16 frozen weights with large, aligned output
dimensions. `TERMITE_METAL_DISABLE_BF16_BACKWARD_SMALL_ROWS=1` provides the
same-binary rollback. Repeated no-frame profiling reduced that product from
`23.7375 ms` to `20.2905 ms` mean (`14.5%`, or `3.447 ms`).

The locked 20-step sequence-128 diagnostic measured `0.466987/0.466848 s`
median/mean (`274.18` input tokens/s) enabled versus
`0.470053/0.470101 s` (`272.28` tokens/s) disabled, a `0.697%` end-to-end
speedup. The sequence-512 pair measured `1.464261/1.464651 s` (`349.57`
tokens/s) versus `1.471423/1.472216 s` (`347.78` tokens/s), a `0.516%`
speedup. Both pairs bind executable SHA-256
`55796f5424996465b5b0a3aab16fe40bfbdd1a1780d5aa76acba68a543b3ebd7`.
The enabled artifacts are
`/private/tmp/antfly-gemma4-zig-e2b-seq128-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`6cb67bd0cf13f5c3f7560fb453072d94ab8359c095e11b1c31c3090558d11cd5`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq512-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`2d9fe10409584f3c0f39b85c3854e2f6888a6994cdc363a3c36148ee2dada57f`);
their rollback partners are retained beside them with `disabled` in the file
name.

One-step loss, gradient norm, and adapter bytes remain exact, including adapter
SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.
The five-update adapter remains SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`
with the accepted loss/gradient history and zero fallback. Required-device
ReleaseFast `test-gemma4-finetune` passed 196 tests with two optional skips,
and inference-edition `antfly-main-test` passed. MLX-LM remains `1.774x` faster
at sequence 128 and `1.478x` faster at sequence 512; the sequence-512 Antfly
sample still paged, and the A/B has one sample per arm, so it is not release
evidence.

These are useful same-host, exact-workload diagnostic ratios. They are not the
locked alternating five-repeat campaign, do not cover sequence length 2048 or
accumulation 4, and do not satisfy the zero-paging release threshold. The HF
oracle remains blocked until its exact CUDA/BF16 environment and pinned source
commits are available. E4B still needs an explicit no-swap capacity preflight.

## Remaining production gates

The standard-library tests prove contract parsing, deterministic fixture
generation, v6 provenance and causal-tokenization binding, closed evidence
publication, producer-role
admission, actual dtype and raw-gradient-norm binding, key
normalization/translation rules, max-absolute/relative/cosine comparison on
synthetic tensors, and fail-closed workload-paired benchmark aggregation. They
do not prove:

- E2B or E4B can complete the locked HF/Zig native/Zig Metal comparison
  sequence (the producer has a required-device tiny-BF16 integration gate, but
  no locked real-model Zig trace has been promoted yet);
- HF/Zig logits, loss, gradients, optimizer moments, and updates meet the
  proposed tolerances at steps 1, 2, and 8;
- the locked numerical-oracle checkpoint/resume profile is exact on real
  artifacts (the preference qualifier separately has real E2B/E4B recovery
  evidence, including the subset-scale E2B mid-epoch result);
- the one-way Antfly-to-PEFT translator matches fixed logits or generations on
  the locked E2B/E4B oracle, or that a translated PEFT adapter round-trips back
  into Antfly (a non-oracle full E2B PEFT load/forward smoke has passed);
- the MLX-LM and Antfly runners complete release-admissible synchronized
  real-model samples for the full same-Mac matrix (one diagnostic E2B cell has
  completed in both runners, but no closed alternating campaign has); or
- throughput and peak memory satisfy the locked release thresholds.

Those gates require the pinned model snapshots and target hardware. Preserve
the raw evidence directories; publication is no-replace and uses a final
`COMPLETE.json` marker so interrupted runs cannot masquerade as complete.
