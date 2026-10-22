# Native KV compaction

Antfly's numerical KV compaction engine fits a smaller attention cache and
exports a portable binary checkpoint tied to the same model revision. It uses
native Zig/CUDA and the existing grammar engine. Structured text memory is a
separate context-management feature.

The current scope is Gemma 4 E4B Q4 on NVIDIA L4 with f16 KV and eager CUDA
execution. Broader answer quality and complete production qualification remain
open. Numerical validation alone does not establish answer quality.

## Preparation and restore

1. Ingest the exact prompt and preserve its tokens and next-token logits.
2. Clone the live KV prefix into an isolated CUDA owner. Replay its last 512
   tokens in 256-token chunks, collecting 320 post-RoPE queries per global
   consumer: 256 training and 64 held out. Capture every shared consumer's
   query heads; never use cached keys as proxy queries.
3. Preserve local sliding-window donors, the first 128 global positions and
   the most recent 2,048 global positions exactly. Fit older global history in
   1,024-row chunks, targeting one retained row per four eligible rows. Shared
   donors are fitted once for their consumers.
4. Use residual key selection, nonnegative attention-mass fitting and
   regularized value fitting. Store mass adjustments as attention-logit biases.
   Validate complete heads after rounding to serving precision, requiring
   relative mass error and normalized output RMSE at most 0.05.
5. On numerical rejection, retry the affected head from its immutable source at
   3:1 and then 2:1 eligible-row retention. The error limits stay unchanged; use
   the first accepted fit and preserve the original head if all attempts fail.
   OOM, cancellation and invalid inputs never trigger a retention retry. Accept the complete
   candidate only if actual global device bytes, including metadata, improve
   by at least 1.5x. Otherwise export the original exact state with the explicit
   outcome `unchanged_insufficient_savings`. Cancellation, OOM and invalid
   inputs remain errors.
6. Export wire-version-3 state by default (legacy V2 remains available): model/tokenizer/template/layout identities,
   tokens, logical positions, donor mappings, exact and fitted KV, coverage,
   biases and continuation logits. Resume validates the complete transfer and
   exact rendered-token prefix, uploads independently owned state, and evaluates
   only appended input. An unchanged prompt uses saved next-token logits.

The V3 disk codec stores exact f16 K/V by default, retaining f32 biases/logits
and full integer metadata. Explicit `X-Antfly-KV-Encoding: fp8` requests try
E4M3FN with separate power-of-two key/value scales for each fitted historical
head range. The fitter tests dequantized values against the original whole head
on held-out consumer queries, with the same 5% mass/output limits. Only accepted
ranges replace the candidate values; failures retain the validated f16 fit.
Protected prefix/recent rows, local windows and insufficient-savings fallbacks
remain exact. This storage optimization does not change resident GPU precision
or global-device savings. It is experimental until broader answer-quality checks
include quantized fresh-question continuations. V1/V2 decoding remains supported;
`f32` export selects V2 for older clients. Decoding budgets expanded host state
independently of compressed wire bytes and publishes only after checksum and
semantic validation.

The checkpoint stays immutable and supports a fresh process with matching
identity. Physical compaction preserves logical positions; it does not enlarge
the 32,768-token context window. Appended prompt input is bounded to 8,192 tokens,
and the full prompt plus reserved output must fit the window.

The fitting planner and HTTP admission use the same geometry-derived allocation
bound: original state, worst-case candidate, task metadata and the largest
workspace for each admitted fit worker, including the largest 2:1 retry.
Up to four workers fit concurrently; the planner reduces this to three at
32K on the designated model. Fitting has a 3,840 MiB host ceiling,
with 128 MiB additional preparation allowance. Whole-head validation is limited
to one worker at a time. The reservation is recomputed from geometry, including per-head codec metadata;
this allocation bound is separate from measured RSS.

During upload, admission reserves bounded host memory before acquiring GPU
execution resources. Transfer ownership holds that lease through destruction.
Candidate publication checks owner revision, logical invariants, completed
device work and cancellation before replacing live state. Terminal responses
follow resource release. Ordinary generation can proceed while a partial
checkpoint upload is held; only one native checkpoint operation is admitted.

## Code and public interface

| Responsibility | Location under `zig/pkg/inference/src/` |
| --- | --- |
| Numerical reference and bounded fitting | `runtime/kv/attention_match.zig`, `attention_match_engine.zig` |
| Query collection, snapshot and identity | `runtime/kv/attention_match_queries.zig`, `attention_match_snapshot.zig`, `attention_match_identity.zig` |
| Portable state and transactional ownership | `runtime/kv/attention_match_checkpoint.zig`, `attention_match_resume.zig`, `attention_match_transaction.zig` |
| CUDA fitting, storage and fitted attention | `ops/cuda/attention_match_fit.zig`, `attention_match_storage.zig`, `attention_match_decode.zig` |
| Model integration and admission | `pipelines/generation.zig`, `server/server.zig`, `server/native_kv_wire.zig` |

Public endpoints are `POST /ai/v1/chat/context/checkpoint` and
`POST /ai/v1/chat/context/resume`. The Python SDK provides streaming file
helpers; JSON and SSE continuations use the same checkpoint contract.
See [API usage and supported configurations](../../../zig/pkg/inference/OPENAI.md#portable-native-kv-checkpoints-under-qualification)
and [SDK usage](../../../py/packages/sdk/README.md#native-kv-checkpoints-under-qualification).
The legacy compaction/selected-KV paths are separate and do not qualify this engine.

## Recorded evidence, 2026-10-20

The pre-merge standalone ReleaseFast candidate
`9ee66204fb1b4df5557f68bbac3ce66c6039e941104cf9a57b8c84febd751deb`
passed public 8K/32K export, SDK repeat preparation, byte-identical re-export,
fresh-process restore, wrong-prefix/checksum rejection and recovery, and
context-limit rejection.

| Public-path observation | 8K | 32K |
| --- | ---: | ---: |
| Logical prefix tokens | 8,189 | 32,752 |
| Actual global-KV savings | 1.686994x | 2.085466x |
| Validated / preserved global heads | 6 / 2 | 6 / 2 |
| Checkpoint bytes | 201,728,532 | 555,662,840 |
| Warm saved-prefix resume | 2.411 s | 6.265 s |
| Complete initial prepare/export request | 102.611 s | 403.763 s |
| Fresh-process restore, including model load/identity hashing | 70.082 s | 74.347 s |
| Peak exporter listener + worker RSS | 5.422 GiB | 7.541 GiB |
| Peak exporter GPU memory | 11.658 GiB | 12.424 GiB |

These are individual observations, not preparation or general serving
percentiles. Global-KV savings do not describe total GPU memory or checkpoint
file compression. The 8K checkpoint plus a 1,437-token suffix resumes in
9.214 s versus 40.746 s for full replay of the same conversation.
A separate saved-prefix completion campaign with five warmups and 100 samples
measures 2.385 s p50, 2.478 s p95 and 2.731 s maximum, for a five-token answer.

The same candidate passed cancellation during upload, prefill, fitting and
export; physical restore OOM/recovery; SSE abort/recovery; and terminal cleanup.
Ordinary generation completed during a held partial upload, competing native
operations failed promptly, and reset released the guard in 11.797 ms.
This qualifies bounded request overlap, not simultaneous GPU model execution.

After merging main at `e0fa87c69f`, 130 native CPU tests passed, with one
CUDA-only skip covered by the 18/18 L4 CUDA checks. Go SDK tests passed;
Python SDK had 38 passes, TypeScript 493 passes/one skip plus typecheck,
OpenAPI 10 passes, and isolated grammar/tokenizer 37 passes.
These are recorded checks, not a substitute for final-candidate qualification.

The post-merge focused inference server
`6a43b29c70a0ad1eee752d223dd1117311dbbce41398e3088ddfe17538607d8d`
reproduces the 8K reference checkpoint byte for byte and restores its correct
answer. Its digest is
`5b09989a64944c79c14990b9d7bea167dc45509323527452ce038825b7a63c8a`.
The complete standalone campaign has not been repeated on the merged candidate.

## Checkpoint disk encoding

V3 FP16 was round-tripped against both saved public reference checkpoints. All
K/V values, biases, positions, coverage, tokens and continuation logits matched
exactly, including their floating-point representations.

| Reference | V2 FP32 bytes | V3 FP16 bytes | File reduction |
| --- | ---: | ---: | ---: |
| 8K | 201,728,532 | 101,995,796 | 49.44% |
| 32K | 555,662,840 | 279,871,736 | 49.63% |

The CUDA inference-server candidate
`adbc7eecff99a76147c586d4f7afad82232d99daf2ece537b60155b00cc3498c`
also restored both V2 and V3 FP16 references at 8K and 32K, producing identical
answers and attesting saved-prefix reuse. Default HTTP and Python SDK exports
produced identical V3 FP16 files. These checks used the inference server, not
the complete standalone deployment.

The FP8 readiness follow-up replaces scalar searches and power calls with
constant-time IEEE exponent/mantissa conversion. An independent arithmetic
oracle checks every FP16 bit pattern at all 17 supported scales, plus both
neighbors of every FP32 rounding midpoint. The existing 8K FP8 file remains
byte-identical after decode/re-export. Core codec, checkpoint, solver and engine
checks passed (70 tests), as did the Python SDK checks (46 tests). SDK metadata
validation requires consistent accepted/fallback FP8 head counts before atomic
file replacement.

The optimized codec passed CUDA inference-server public roundtrips at 8K and
32K. Each accepted five FP8 heads, retained FP16 for one rejected attempt and
preserved two uncompacted global heads. Protected rows and local windows remain
FP16. These are disk savings beyond the lossless FP16 encoding, not additional
device-memory savings:

| Workload | FP16 file | FP8 file | Additional disk saving | Export/write phase | Warm restore |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8,189 tokens | 97.27 MiB | 89.93 MiB | 7.55% | 3.07–3.10 s | p50 1.826 s; p95 1.858 s; max 1.961 s |
| 32,752 tokens | 266.91 MiB | 229.58 MiB | 13.98% | 9.79 s | 4.76–4.99 s; roundtrip observations only |

The 8K percentiles use five warmups and 100 measured SSE restores, all with
identical continuation and token accounting. They describe portable restore,
not full replay or llama.cpp parity. The 8K final candidate is
`595d485ee23c9d9c1d71d00e793f006819ff2cdeaa12510a4360353aa6300194`.
Its complete first preparation took 102.61 s including ingestion and cold
identity work; fitting took 8.79–9.17 s separately. Export/write previously
took about 10.4 s before the codec optimization. The 1,437-token appended
continuation took 8.62 s. Public and SDK exports retained the same file digest
`39bc695d64ab02d00d1bd92d2a7764dbc0755b1ab4e187dc9361329f822c3273`.

Both sizes passed fresh-process restore, streaming, checksum/prefix rejection
and recovery. The 32K run also passed context-limit rejection and recovery,
with 2.0855x global-device savings and peak host/GPU memory of 7.34/12.46 GiB.
It used the preceding optimized candidate
`3f50aaca791b674546b559e737d52759a74cb35c0b1fa1bd5cdf8f91c5ec410d`;
the final native-code change adds physical CUDA OOM logging without changing
the codec or allocation error contract.

The final candidate passed physical GPU OOM/recovery, overlapping uploads and
streaming disconnect/recovery. The OOM test requires a fresh physical driver
failure log, so a quota rejection cannot falsely pass; after external pressure
was released, the same worker restored without retry. Streaming disconnect
released ownership in 11 ms. Partial-upload, prefill, fitting and export
cancellation passed on the preceding optimized candidate. These checks do not
qualify OOM during live cache replacement, sustained/repeated-cycle operation,
or the complete standalone deployment.

Local evidence is under `.cache/fp8-readiness/`: `final-8k.json`,
`public-32k.json`, `restore-100.json`, `oom-fixed.json`, `concurrency.json`,
`sse-abort.json`, and `qualify_native_cancellation.json`. Earlier codec evidence
remains under `.cache/kv-codec/`. The historical performance table above still
describes V2.

For FP8 quality runs, pass `--encoding fp8` to
`qualify_native_checkpoint_quality.py`. Each case compares full KV, a separately
prepared FP16 checkpoint, and FP8 on the same fresh question. Reports distinguish
model baseline failures, compaction regressions, additional FP8 regressions,
and actual quantization coverage by category. Exact and FP16 fallbacks cannot
count as FP8 coverage; an improvement on another case cannot hide a quantized
case's regression. The frozen suite and existing quality thresholds are unchanged.
The public roundtrip, preparation sampling, and sustained runners also accept
`--encoding fp8`; cancellation uses the encoding of its qualified reference.

FP8 remains experimental. A final-candidate pilot of four frozen archive-only
cases (`v2-fact-04`, `v2-code-05`, `v2-instruction-06`, `v2-tool_call-07`) completed
without infrastructure errors, but all four used `unchanged_insufficient_savings`
with zero FP8 heads. Full KV, separately prepared FP16 and FP8-requested restores
all scored 3/4 with identical answers; `v2-code-05` returned 15 instead of 29 in
every path. This is fallback evidence, **not FP8 answer-quality qualification**.
See `quality-pilot-8k.json`. The full 64-case paired suite has not passed.
The separate `fp8_noninferiority` report requires the full sample counts,
critical-case count, actual FP8 coverage in every category and no new failures
versus FP16; it does not relax the whole-system release gate or forgive missing
quantization coverage. Improving archive-only fitting coverage remains necessary
before those quality gates, final sustained testing and standalone qualification
can close. The 5% numerical error and 1.5x public savings thresholds remain intact.

## Answer quality

`prepare_native_quality_suite.py` generates the frozen V2 suite: 64 synthetic
cases, 16 per category, crossing four task families with archive positions at
10%, 25%, 45% and 60%. Sixteen cases are designated critical before execution.
The archive is checkpointed before the final question; restore appends the
question and must attest saved-prefix reuse and equal full-prompt token counts.
The suite digest remains
`11b587964826b3e84a90881363b845964e5d857294ec148d2bf799a12dd8a14e`.

All 16 critical cases were exercised in two diagnostic runs:

| Category | Full-KV correct | Restored correct | Actually compacted |
| --- | ---: | ---: | ---: |
| Multi-step facts | 4/4 | 4/4 | 0/4 |
| Instructions | 3/4 | 3/4 | 0/4 |
| Code reasoning | 2/4 | 2/4 | 0/4 |
| Typed tool calls | 4/4 | 4/4 | 0/4 |

All used `unchanged_insufficient_savings`. Equal 13/16 scores establish no added
failures on exact fallback; they do **not** establish quality under compaction.
The full 64-case campaign has not passed. The failures are `v2-code-01`
(filtered sum 17 instead of 25), `v2-code-03` (JSON count 16 instead of 19),
and `v2-instruction-03` (`ready.` instead of `ready`, without credential exposure).
The code answer keys agree with Python execution. On identical native tokens,
llama.cpp b11669 reproduces the first arithmetic failure at short and long
context; that is a scoped diagnostic, not broad runtime parity.

Scoring rejects malformed/duplicate-key/nonfinite JSON, wrong argument types,
extra fields/calls, truncated answers and changed/reordered requests. It
separates baseline failures, regressions and improvements. Unchanged fallbacks
cannot conceal failures on the actually compacted subset; interrupted runs
retain completed pairs without passing full-suite gates. Older evidence remains
readable, but new runs require an explicit frozen suite path.

A private fit-only diagnostic on the 8,253-token code archive found:

| Calibration replay | Validated / preserved heads | Candidate savings | Query collection |
| --- | ---: | ---: | ---: |
| Last 512 tokens | 3 / 5 | 1.248935x | 0.941 s |
| Full prefix | 2 / 6 | 1.148928x | 16.840 s |

Neither passes the public 1.5x gate. No candidate was installed or production
threshold changed. Including the final question before checkpointing produces
1.513153x savings on that code prompt, but does not validate unseen questions.

## Reproducing checks

Run from the repository worktree root. Keep generated suites, model data,
checkpoints, logs and build outputs out of Git, under `.cache/`, `.models/` and
`zig/zig-out/`. Use fresh evidence paths and run hardware campaigns serially on
an idle GPU after compilation finishes.

```sh
py/packages/sdk/.venv/bin/python -m unittest discover \
  -s zig/pkg/inference/scripts/gemma4 -p 'test_*quality*.py'
bash zig/pkg/inference/scripts/regen-attention-match-fit.sh --check
bash zig/pkg/inference/scripts/regen-attention-match-decode.sh --check

python3 zig/pkg/inference/scripts/gemma4/prepare_native_quality_suite.py \
  --output .cache/native-compaction/native-quality-suite-v2.jsonl
py/packages/sdk/.venv/bin/python \
  zig/pkg/inference/scripts/gemma4/qualify_native_checkpoint_quality.py \
  --server zig/zig-out/bin/antfly --server-kind standalone --models-dir .models \
  --suite .cache/native-compaction/native-quality-suite-v2.jsonl \
  --output zig/zig-out/qualification/context-cuda/quality.json \
  --checkpoint .cache/native-compaction/quality-current.kv
```

Retained entry points under `zig/pkg/inference/scripts/gemma4/`:

| Purpose | Runner |
| --- | --- |
| Public export, SDK roundtrip and fresh-process restore | `qualify_native_checkpoint_http.py` |
| Prepare-phase sampling; terminal ownership and resume sampling | `qualify_native_prepare_sampling.py`; `qualify_completion.py` |
| Cancellation, physical OOM, upload overlap, SSE abort | `qualify_native_cancellation.py`, `qualify_native_checkpoint_oom.py`, `qualify_native_concurrency.py`, `qualify_native_sse_abort.py` |
| Sustained native requests | `qualify_native_stress.py` |
| Matched exact-token replay measurements and comparison | `qualify_matched_antfly.py`, `qualify_matched_llama.py`, `compare_matched_latency.py` |
| Private query, fit, restore and repeated-cycle diagnostics | `qualify_attention_match_queries.py` |
| Structured-memory/window API checks | `qualify_context_management.py`, `stress_context_cuda.py` |

Use each runner's `--help` for its required reference/checkpoint inputs.
Private diagnostics do not qualify the public API. Kernel-level CUDA oracles
are retained for the production fitter, fitted attention, repeat attention and
staged Q4 projection (`qualify_attention_match_fit.{cu,py}`,
`qualify_attention_match_decode.cu`, `qualify_repeat_attention{,_kernel}.cu`,
`qualify_q4_bf16_staged.cu`). The repeat `_kernel` file supplies the independent
chronological trace oracle; it is not a spare prototype. CUDA regeneration
scripts and shipped artifacts remain required.

## Release gates and next work

| Area | Required evidence |
| --- | --- |
| Quality | At most 2 percentage points aggregate drop and 5 per category versus full KV; all designated critical cases pass; actual compaction in every category with the same limits on the compacted subset |
| Compression | Nominal 4x eligible-old-row retention and at least 2x actual global-KV savings on the designated 32K workload; public candidate minimum remains 1.5x |
| Memory | At most 8 GiB combined listener/worker RSS and 18 GiB GPU memory, including temporary fitting/replacement allocations |
| Latency | Matched full-replay p50 at most 1.25x and p95 at most 1.5x llama.cpp; at least five warmups and 100 samples per designated workload |
| Reliability | 1,000 mixed requests, 100 compactions and 20 successive real-model compaction cycles; no corrupt state, leaked admission or unbounded growth |

Bounded per-head retention retries are implemented. Captured archive-only
inputs now validate seven of eight heads (six FP8, one FP16 fallback), compared
with three before retries, under the unchanged numerical gates. This numerical
diagnostic does not qualify answer quality or public installation. Its engine
checks passed 74 tests, including retry cancellation, allocation failures,
parallel determinism and exact host-budget boundaries. Public quality and
standalone qualification must use the successor binary.
Keep baseline reasoning/format failures separate from compaction regressions.
Freeze the successor before the complete 64-case paired quality suite and matrix,
then broaden histories and 32K coverage. Final-candidate sustained testing,
live-replacement failure checks, matched latency and merged standalone
requalification remain open. Historical 10 s/30 s compaction targets were
accepted as nonblocking by the user; continue recording phase timings separately
from ingestion. Automatic mid-generation compaction and additional models or
backends need separate qualification.

Evidence lives locally under `zig/zig-out/qualification/context-cuda/`:
`public-checkpoint-deferred-upload-admission-{8k,32k,cancellation,oom,concurrency,sse-abort,completion}.json`,
`quality-v2-critical-summary.json` (with both source fingerprints),
`quality-v2-merged-reference-diagnostic.json`, `quality-v2-code-token-capture.json`,
`quality-v2-llama-code-diagnostic.json`, and `quality-v2-calibration-diagnostic.json`.
The first quality run stopped intentionally after eight complete cases; the
second finished with `passed=false` and no infrastructure error. Historical
experiment chronology is available in Git rather than duplicated here.
