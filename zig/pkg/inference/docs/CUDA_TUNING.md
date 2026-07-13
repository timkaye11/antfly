# CUDA Tuning Profile

Gemma 4 QAT CUDA measurements must use the shared profile in
`scripts/gemma4_qat_cuda_tuning.sh`. The CLI benchmark and server wrapper both
source it so production route settings stay aligned.

## Production Defaults

The tuned profile currently uses these conservative defaults:

- Q4_0 x Q8_1 generated LM-head argmax: on.
- Generated Q6_K x Q8_1 LM-head argmax: off.
- Generated decode attention: off.
- Generated paged score-prework attention: off.
- Generated E2B FFN pair and down kernels: off.
- Continuous generation batching: off.
- Request-scoped persistent CUDA graph replay: required.

Run the tuned server with:

```sh
zig/pkg/inference/scripts/with_gemma4_qat_cuda_tuning.sh \
  zig/pkg/inference/zig-out/bin/antfly-inference run \
  --host 127.0.0.1 --port 8080 --models-dir .models --config server.json
```

With its default graph profile, the wrapper sets
`ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET=1` and disables continuous
batching so graph state cannot cross active requests. The default graph KV
capacity is 544 tokens. For longer requests, set a capacity covering prompt
and output tokens, rounded up to a KV page:

```sh
ANTFLY_CAPTURE_FORCE_KV_CAPACITY=1056 \
  zig/pkg/inference/scripts/with_gemma4_qat_cuda_tuning.sh \
  zig/pkg/inference/zig-out/bin/antfly-inference run \
  --host 127.0.0.1 --port 8080 --models-dir .models --config server.json
```

The pinned temporary schedule uses a measured 853 allocations per E2B decode
iteration. `ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD` (or
`ANTFLY_CUDA_TEMP_SLOT_PERIOD`) can override it for a different model/runtime
topology; do not reuse the E2B value without an allocation trace. Request reset
clears prior slot ABI mappings before rewinding the sequence and fails closed if
any slot is still live.

Continuous batching and request-scoped graph replay are mutually exclusive.
Set both server controls explicitly when running the batching profile:

```sh
ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING=0 \
ANTFLY_SERVER_DECODE_GRAPH_REPLAY=off \
  zig/pkg/inference/scripts/with_gemma4_qat_cuda_tuning.sh \
  zig/pkg/inference/zig-out/bin/antfly-inference run \
  --host 127.0.0.1 --port 8080 --models-dir .models --config server.json
```

Graph-off mode forces the pinned slot period, skip, and request reset to zero.
The wrapper rejects batching with any other graph mode, and the server retains
the model-wide lock for graph, speculative/MTP, and multimodal requests.
`generation_batching.mode: auto` also remains serialized until the CUDA release
throughput gate is promoted; use `mode: on` only for an explicit row-2 run.

`scripts/benchmark_gemma4_cuda_batching.py` uses distinguishable equal-length
prompts, alternating row order, and a staggered mixed-length/page-growth probe.
Scheduled CI enforces that correctness corpus and a regression floor. Release
runs retain the `1.5x` C2 aggregate-throughput promotion threshold.

## Model-Neutral Kernel Catalog

The checked-in AOT catalog selects kernels by semantic operation signature,
runtime shape, and target, not by model name or layer index. Inspect a GGUF from
the inference package directory:

```sh
cd zig/pkg/inference
ZIG=../../../.tools/zig-x86_64-linux-0.16.0/zig
MODEL=../../../path/to/model.gguf

$ZIG build quant-kernel-codegen -- --inspect-model "$MODEL"
$ZIG build quant-kernel-codegen -- --check-model "$MODEL"
```

Both model commands are read-only and emit
`antfly.quant_kernel.model_inventory.v1` JSON. `--inspect-model` always reports
the inventory. `--check-model` also exits nonzero when `coverage.complete` is
false, meaning a required quantized operation or attention topology is not
represented by the compiler's semantic catalog. Backend route fields are
reported separately.

Exact AOT status is intentionally a second dimension. `aot_resolutions` records
target-specific `aot_resolved`, `aot_missing`, or `semantic_unsupported` status,
the kernel and source fingerprints, and `production_enabled`. An exact AOT miss
is an optimization gap and does not fail the semantic model check. The
`aot_targets` summary counts exact hits and misses; `all_signatures_resolved`
means every emitted signature probe resolved, not that every possible runtime
route is promoted.

The checked-in exact catalog currently targets CUDA `sm_89` only. Its scope is
selected Q4_0 and Q4_K row-1 matmuls, Q4_0 x Q8_1 fused FFN shapes, Q6_K x
Q8_1 K=2560 and K=3840 tile-argmax candidates, and Gemma 4 decode-attention
topologies. In the merged unified generated-artifact registry, 21 CUDA entries
are non-promoted; no single Q4_K route represents the remaining candidate set.
Default-on generated Q4 runtime routes are therefore enabled only on SM89.
Other compute capabilities use the handwritten fallback even when their code
objects are present in the fatbin. Development experiments can bypass this
promotion guard with
`ANTFLY_INFERENCE_CUDA_ALLOW_UNPROMOTED_GENERATED_KERNELS=1`; results from that
mode are not production evidence.
Production-enabled entries may route without a candidate opt-in. The 1536-wide
FFN, Q4_K, Q6_K argmax, and generated-attention entries remain candidates; the
2560 x 10240 FFN pair/down entries are production-enabled. Q4_K candidates are
not runtime-wired. The Q6_K argmax stage-1 candidates are runtime-wired behind
a disabled opt-in and must pass device parity and target-specific performance
gates before promotion. The dense split-KV attention route remains dev-only
because its reordered partial-softmax merge does not satisfy the long-output
parity contract. The paged score-prework route instead parallelizes only QK
scores and retains the production chronological softmax recurrence.

Generated attention remains opt-in through
`ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE=1`; the standard server and
tuned CLI profile leave that gate off. Its default split-KV threshold is 512,
and `ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SPLIT_KV_MIN_TOKENS` can select
another threshold for an explicit experiment. Set
`ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SPLIT_KV_SPLITS` to `2`, `4`, or `8`
to select a dev-only reduction schedule; it defaults to `8`. Each split count
uses a distinct CUDA graph replay key, and none is production-enabled.

The paged TurboQuant candidate has a separate default-off gate:
`ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK=1`. It is eligible
only for batch-1, single-row decode with head dimension 256 or 512, Polar4 or
F16 keys, F32 values, no mask or bias, and a physical KV capacity at most 4096
tokens. An explicit score-prework opt-in takes precedence over the handwritten
split-attention experiment; an ineligible call falls back to the established
fast decode kernel. Its fixed 128-chunk/4096-score topology is CUDA graph
stable. The gate and physical KV capacity are startup-static, so this route
shares the normal decode replay identity; it cannot change topology within an
eligible request. The shared Gemma 4 tuning profile and release gate keep it
disabled pending multi-model and long-context promotion evidence.

For raw numerical diagnosis, use the differential harness. It bypasses runtime
routing and graph replay, launches the handwritten fast decode kernel alongside
generated serial and split-KV kernels directly, and reports bitwise mismatches,
maximum absolute/relative error, and maximum F32 ULP distance. The default
corpus covers head dimensions 256 and 512 with random, near-tie softmax, and
cancellation-heavy inputs:

```sh
cd zig/pkg/inference
../../../.tools/zig-x86_64-linux-0.16.0/zig build \
  quant-kernel-cuda-attention-diff -Dcuda=true -Dmetal=false \
  -Dcuda-artifacts=sm89 -Doptimize=ReleaseFast -- --kv-len 1024
```

Use `--head-dim`, `--heads`, `--kv-heads`, `--pattern`, and
`--split-count 2|4|8|all` to isolate a dense topology and reduction schedule.
`--query-position` and `--sliding-window` isolate individual split boundaries.
`--max-abs`, `--max-ulp`, and `--require-bitwise` turn the report into a
failure gate; `--json` emits `antfly.cuda_attention_diff.v2` for automation.

For the production paged-KV ABI, use the score-prework differential. The build
step compiles both generated candidates with the pinned CUDA 13.2 toolkit,
reverses the physical page table, checks Polar4 and F16 keys, and fails on any
bit mismatch against `termite_gqa_attention_decode_turboquant_fast_f32`:

```sh
cd zig/pkg/inference
../../../.tools/zig-x86_64-linux-0.16.0/zig build \
  quant-kernel-cuda-paged-attention-diff -Dcuda=true -Dmetal=false \
  -Dcuda-artifacts=sm89 -Doptimize=ReleaseFast -- \
  --head-dim all --kv-len 2048 --pattern all --key-format all \
  --iterations 100
```

The paged harness supports fixed/reversed page layouts, visible-prefix and
sliding-window boundaries, both supported key formats, and the same adversarial
input patterns. `--json` emits `antfly.cuda_paged_attention_diff.v1`. On SM89
L4, every HD256/HD512, Polar4/F16, and random/near-tie/cancellation case was
bitwise exact at KV lengths 256, 1024, and 2048. The generated launch measured
1.43x to 1.83x faster than the handwritten paged baseline across that matrix.

Freeze full-model parity, graph replay, route counters, and paired throughput
with the model-neutral candidate validator:

```sh
cd zig/pkg/inference
python3 scripts/validate_gemma4_cuda_candidate.py \
  --kernel-id cuda.attention.gqa.decode.score_prework \
  --cache-dtype polar4 \
  --prompt Ants \
  --prompt "Write one sentence about the night sky." \
  --prompt "List the first five prime numbers." \
  --lengths 256 1024 --repeats 1 --capture-kv-capacity 2048 \
  --output-dir /tmp/antfly-score-prework-final
```

The validator locks F32 values, disables the handwritten split experiment and
dense generated attention in both arms, alternates execution order, compares
every token ID, requires persistent graph replay, and requires the dedicated
`launch_attention_gqa_decode_score_prework` counter. On the E2B QAT model and
SM89 L4, the three-prompt corpus measured 98.918 versus 82.183 tok/s median at
256 outputs (1.1999x) and 67.422 versus 44.483 tok/s at 1024 outputs (1.5157x).
All six pairs had exact token IDs, persistent replay, zero fallback, and no
graph discard or capacity skip. The F32-cache production control remained
healthy at 91.103 tok/s with 251 persistent replays. These results justify
retaining the candidate, but not enabling it globally before broader
model/context coverage.

The generated serial decode path now matches the production decode-scalar
contract exactly: it passed three prompts at 64 and 256 output tokens with a
0.9995x median paired throughput ratio, and three prompts at 1024 output
tokens with zero graph-capacity skips. Split-KV is substantially faster in the
current short-output corpus (about 1.28x at 256 tokens with a 128-token
threshold), and those runs had exact token IDs. It is still experimental: the
same split-128 candidate changes one token at position 915 for one prompt in a
three-prompt 1024-token corpus, despite a roughly 2x throughput gain. Do not
enable split-KV in a production profile or promote the generated attention
catalog entries until a long-output, multi-model parity corpus passes.

The split-count sweep on the SM89 L4 keeps split-8 as the short-context winner:
at 256 output tokens it measured 115.8 tok/s versus 89.9 tok/s for the
baseline (1.288x median) with exact tokens on the three-prompt corpus. Split-2
and split-4 both changed the prime-number prompt at token 232. At 1024 outputs,
split-8 measured 103.3 tok/s versus 51.4 tok/s (2.01x median) but retained the
sky-prompt change at token 915; raising its threshold to 512 reduced throughput
to 1.54x without moving that divergence. These are development measurements,
not production tuning defaults.

The raw harness isolates the source of dense split-KV drift: generated serial
matches the handwritten fast kernel bitwise, and each split partition's local
recurrence is also bitwise. The first mismatch occurs when stage 2 merges a
multi-token partition summary into the preceding online-softmax state. The
paged score-prework route avoids that merge by parallelizing only score
calculation, then applying the canonical chronological softmax/value recurrence.

Inventory `coverage.complete`, `cuda_route_complete`, and
`metal_route_complete` describe semantic/runtime-route availability, not exact
AOT promotion. The legacy `*_catalog_complete` fields are compatibility aliases
for route completeness. Exact target summaries split required and optional
signatures and expose `all_required_signatures_resolved` and
`all_optional_signatures_resolved`. `present_quantized_operation_count` counts
model tensors; `quantized_operation_count` counts active decode operations after
shared-KV and omitted-V rules. Every numeric fingerprint also has a canonical
16-digit `*_fingerprint_hex` field for lossless JSON consumers.

Enable catalog FFN candidates with the model-neutral gate:

```sh
ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_CATALOG_FFN_CANDIDATES=1
```

`ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN` is the legacy alias. It supplies
the default only when the catalog gate is unset, so the catalog gate takes
precedence. Keep candidates disabled in production until parity and repeated
full-model performance gates pass.

With JSON timing enabled, use
`cuda.generated_kernel_catalog_resolve_attempts`,
`cuda.generated_kernel_catalog_resolve_misses`,
`cuda.generated_kernel_catalog_hits`, and
`cuda.generated_kernel_catalog_fallbacks`. Resolve misses identify unsupported
shapes or targets; fallbacks identify a resolved kernel that failed to launch.
The top-level
`quant_kernel_plan` object reports `planned`, `handwritten_production`,
`generated_production`, `generated_candidates`, `unsupported_routes`,
`fast_path_misses`, and granular fallback reasons. The E2B-named pair/down hit
and fallback counters remain compatibility telemetry.

The generated Q6_K x Q8_1 LM-head stage-1 candidates are guarded by:

```sh
ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX=1
```

They are deliberately limited to greedy, unsuppressed `rows=1`,
`out_dim=262144` LM heads with `in_dim=2560` or `3840`. The route retains the
existing Q8_1 quantizer, partial workspace, shared argmax reducer, token
handoff, and graph-replay behavior; only stage 1 is generated. The standard
tuning profile leaves this gate at `0`. The candidate validator requires an
explicit `--model` because the default E2B QAT fixture has a Q4_0 LM head and
cannot exercise this Q6_K route. Use a GGUF whose output tensor is Q6_K, then
validate exact tokens, replay, and route telemetry before comparing throughput:

```sh
zig/pkg/inference/scripts/validate_gemma4_cuda_candidate.py \
  --candidate q6-k-q8-1-lm-head-argmax \
  --model .models/google/gemma-4-12B-it-q4_k/gemma-4-12B-it-Q4_K_M.gguf \
  --config-label l4-sm89-q6-lm-head \
  --lengths 64 128 256 512 --repeats 5 \
  --min-candidate-ratio 1.00 --max-cv 0.02 \
  --output-dir /tmp/antfly-q6-lm-argmax-validation
```

The current generated body specializes Q6 sub-layout unpacking while preserving
the four dependent DP4A operations and the complete floating-point reduction
order. In the canonical SM89 artifact it uses 40 registers with no local memory,
versus 96 registers for the handwritten Q6 stage-1 baselines. The isolated
complete chain is bitwise-equal at every stage-1 partial value and index, and
measured 1.129x at K=2560 and 1.085x at K=3840 on the L4. This has not cleared
the production gate: a three-repeat Gemma 4 12B Q4_K_M graph-replay run at 64
and 256 tokens had exact token IDs, required route hits, and zero fallbacks, but
a 0.9992x median paired ratio and 0.9847x worst pair. Keep the gate off until a
larger end-to-end improvement is demonstrated.

Regenerate owned sources only after changing compiler or catalog inputs, then
check the result:

```sh
$ZIG build quant-kernel-codegen -- --write
$ZIG build quant-kernel-codegen -- --check
$ZIG build quant-kernel-codegen -- --check-metal  # Requires the Metal toolchain.
```

## Paired Comparison

This is the exact 256-token Antfly/llama.cpp collection command for the current
profile. The 255/256 request values are intentional benchmark accounting.

```sh
(cd zig/pkg/inference && \
  ../../../.tools/zig-x86_64-linux-0.16.0/zig build \
    -Dcuda=true -Dmetal=false -Dcuda-artifacts=sm89 -Doptimize=ReleaseFast)

OUT_DIR=/tmp/antfly-gemma4-l4-256 \
WARMUPS=1 REPEATS=3 PROMPT='Here is a sentence about ants:' \
ANTFLY_TOKENS=255 LLAMA_TOKENS=256 \
ANTFLY_CACHE_DTYPE=f32 LLAMA_CACHE_TYPE_K=f32 LLAMA_CACHE_TYPE_V=f32 \
ANTFLY_DECODE_GRAPH_REPLAY=required REQUIRE_GRAPH_REPLAY=1 \
ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE=0 \
ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX=1 \
ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN=0 \
REQUIRE_GENERATED_ATTENTION=0 REQUIRE_LM_HEAD_ARGMAX=1 \
REQUIRE_GENERATED_E2B_FFN=0 \
MIN_LLAMA_THROUGHPUT_RATIO=0 MIN_COMPARABLE_THROUGHPUT_RATIO=0 \
MIN_ANTFLY_TOK_S=0 MAX_ANTFLY_TOK_S_CV=0.02 \
  zig/pkg/inference/scripts/gemma4_qat_llamacpp_pair_benchmark.sh
```

On an NVIDIA L4, the verified ReleaseFast result was 90.138 Antfly decode
tok/s with CV 0.0006, versus 134.34 llama.cpp eval tok/s. The raw throughput
ratio was 0.671. The comparable ratio, using llama.cpp evaluation plus
sampling time, was 0.728.

## Output Matrix

Collect the current production matrix without asserting future throughput
targets:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_matrix.py \
  --lengths 64 128 256 512 --target-length 256 \
  --min-antfly-tok-s 0 --min-comparable-ratio 0 --max-cv 1 \
  --no-require-generated-attention --collect-only
```

The matrix still evaluates healthy persistent graph replay by default;
`--collect-only` records a failed check without exiting nonzero. Generated
attention is not required because the production profile keeps it off.

The following are aspirational gates, not current achieved values:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_matrix.py \
  --lengths 64 128 256 512 --target-length 256 \
  --min-antfly-tok-s 120 --min-comparable-ratio 0.95 --max-cv 0.02 \
  --no-require-generated-attention
```

Use `--require-generated-attention` only for an explicit attention candidate
run after exact-token parity has passed.

## Candidate Validation

The candidate validator checks every baseline/candidate pair for exact token
IDs, persistent graph replay without discards or capacity skips, zero baseline
route hits, required candidate route hits, and zero forbidden fallback counts.
With `--repeats`, it alternates baseline-first and candidate-first execution.

`passed` includes the configured performance gates. Defaults remain permissive
for compatibility, so promotion runs must set `--min-candidate-ratio` and
`--max-cv` explicitly. The ratio gate applies to the worst individual pair;
the CV gate applies to both routes within each prompt/output-length case.

Run the repeated LM-head regression gate with:

```sh
zig/pkg/inference/scripts/validate_gemma4_cuda_candidate.py \
  --candidate q4-0-q8-1-lm-head-argmax \
  --lengths 64 128 256 512 --repeats 5 \
  --min-candidate-ratio 1.01 --max-cv 0.02 \
  --output-dir /tmp/antfly-lm-argmax-validation
```

The verified full LM-head matrix covered three prompts and four lengths. All
12/12 cases had exact tokens and zero candidate fallback counts. Median paired
improvement was 2.52%; the worst case improved 1.59%. This evidence supports
the current profile default of LM-head argmax on.

Generated attention remains off by default. The earlier three-prompt divergence
was traced to candidate routing through the non-device-scalar warm-up path,
whose production kernel uses a different reduction contract. Generated
attention is now restricted to the matching decode-scalar path. The serial
candidate has exact token parity across the current three-prompt 64/256 corpus
and is throughput-neutral; split-KV remains opt-in because its long-output
parity gate still fails despite its material throughput advantage.

The corrected isolated E2B FFN chain measured 1.23x at width 6144 and 1.31x at
width 12288, but the five-repeat full-model gate improved only 0.21% at the
median and diverged deterministically on two of three prompts. The isolated
harness synchronizes after each chain, while production replays a CUDA graph,
so launch savings do not translate directly. These kernels remain dev-only.
Require repeated full-model evidence before changing the profile default:

```sh
zig/pkg/inference/scripts/validate_gemma4_cuda_candidate.py \
  --candidate q4-0-q8-1-e2b-ffn \
  --lengths 64 128 256 512 --repeats 5 \
  --min-candidate-ratio 1.00 --max-cv 0.02 \
  --output-dir /tmp/antfly-e2b-ffn-validation
```

The exact F32 E2B FFN candidate is a separate default-off route. It preserves
the production F32 pair-activation boundary and reduction topology instead of
the Q8-intermediate contract above. Its validator pins both arms to raw-Q4_0
F32 pair activation and the handwritten tile4 down path: it disables the Q8
pair/down and Q8 precompute routes, catalog E2B FFN candidates, generated Q4_0
MMV, alternate F32 tile shapes, and Q4_0 dequant-on-upload. The resolved locks
are recorded in `candidate_summary.json`. It then requires exact token IDs,
persistent replay, nonzero exact pair and down hits only in the candidate, and
zero exact-route fallbacks:

The semantic AOT catalog records these as optional, production-disabled F32
entries for model inventory and validation. Runtime dispatch remains the
separate `ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT` direct gate;
cataloging the artifacts does not enable that route.

```sh
zig/pkg/inference/scripts/validate_gemma4_cuda_candidate.py \
  --candidate q4-0-e2b-ffn-exact \
  --lengths 64 256 --repeats 3 --capture-kv-capacity 2048 \
  --min-candidate-ratio 1.00 --max-cv 0.02 \
  --output-dir /tmp/antfly-e2b-ffn-exact-validation
```

## L4 Release Evidence

`.github/workflows/cuda-gemma4-l4.yml` is the GPU-only evidence lane. It runs
nightly and on manual dispatch; pull requests remain CPU-only and run the
generator plus benchmark/validator contract tests in `Zig Tests`.

Configure the self-hosted runner with repository variables rather than adding a
hard-coded label to the workflow:

- `ANTFLY_CUDA_L4_RUNNER_LABELS_JSON`: JSON runner-label array, for example
  `["self-hosted","linux","x64","gpu-l4"]`.
- `ANTFLY_CUDA_E2B_MODEL` and `ANTFLY_CUDA_GEMMA12B_Q4_MODEL`: absolute GGUF
  paths on that runner. Workflow inputs override these for a manual run.
- `ANTFLY_CUDA_LLAMA_CPP_BIN`: absolute `llama-completion` path; it defaults to
  `/tmp/llama.cpp/build/bin/llama-completion` for the existing benchmark host.
- `ANTFLY_CUDA_VISIBLE_DEVICES`: optional CUDA device index, defaulting to `0`.

The gate refuses a device other than a single NVIDIA L4 / SM89, checks generated
sources and checked-in CUDA artifacts, builds ReleaseFast, and runs
`cuda-info --smoke`. It then records the fixed 255/256-token E2B f32 paired
benchmark with persistent graph replay and candidate attention, Q6 LM-head,
Q8-intermediate E2B FFN, and exact-F32 E2B FFN routes disabled. It also runs
the 12B Q4_K_M f32 replay case twice and
requires identical token IDs, healthy replay, and zero disabled-candidate
counters. The production E2B profile keeps the faster Q8_1 DP4A linear and
pair routes ahead of the generated Q4_0 MMV/MM/pair routes and requires zero
fallback from every generated Q4_0 route. Generated-route hit and promotion
evidence remains a separate exact-token kernel-candidate gate because the
fixed release prompt does not exercise every generated row bucket.

Use manual `gate=release` to preflight the commit intended for release. Tag-based
publication calls the same workflow at the tag SHA and cannot publish assets
until it passes. Its
`release_scope` is `target_only`: CUDA MTP is not executed or certified. That mode
requires the E2B comparable llama.cpp ratio to be at least `0.80` and a token
throughput CV no higher than `0.02`; nightly collection keeps the same
correctness/replay contract without asserting the future throughput target.
Each run uploads `release_summary.json`, `release_provenance.json`, paired raw
logs/timing, and 12B timing evidence.

CUDA MTP diagnostics run only in `gate=nightly`. They are experimental,
optional, and non-gating, and they make no production-readiness or
llama.cpp-superiority claim. Strict CUDA MTP certification is follow-up work.

For an equivalent local evidence run after a ReleaseFast CUDA build:

```sh
python3 zig/pkg/inference/scripts/gemma4_cuda_l4_release_gate.py \
  --binary zig/pkg/inference/zig-out/bin/antfly-inference \
  --llama-cpp-bin /path/to/llama-completion \
  --e2b-model /path/to/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  --gemma12b-q4-model /path/to/gemma-4-12B-it-Q4_K_M.gguf \
  --output-dir /tmp/antfly-gemma4-l4-release \
  --enforce-performance --min-comparable-ratio 0.80 --verify-artifacts
```

## Server Batching

Measure warmed HTTP server throughput and repeatability with:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_server.py \
  --tokens 256 --warmups 1 --repeats 5
```

Continuous batching remains off. Before enabling it, run the row-2 production
gate for dense and compressed KV storage:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_batching.py \
  --tokens 256 --cache-dtypes f32 polar4 \
  --concurrency 1 2 4 --min-c2-speedup 1.5
```

The gate requires exact response equality, positive scheduler row-2 counters,
at least 1.5x C2 aggregate throughput versus batching-off C1, and bounded C1
latency. Concurrency 4 and above remains diagnostic under the current two-row
validated envelope.
