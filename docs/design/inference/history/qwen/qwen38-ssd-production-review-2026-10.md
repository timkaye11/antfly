# Qwen3.8 SSD inference: competitive and production review

Review date: 2026-10-11. Starting checkout: `732aad0bbc`, branch
`qwen38-flash-next-metal`, initially clean. User-confirmed target: reliable
local CLI/chat on this 16 GiB M4 Mac. HTTP serving, concurrent users,
CUDA, vision and new model families are outside this release target.

**Retain the Zig/Metal architecture and the experimental designation.** The
implementation has substantial numerical and memory work behind it, but is
not yet a production-qualified local release. Prioritize correctness and
lifecycle gaps, then reduce SSD bytes per useful token.

## Current implementation and evidence

The old session note that 27B did not run is superseded by the current code and
checked-in October 9–10 records. Both families now have separate runtimes.

| Property | Qwen3.8-27B | Qwen3.8-Flash-Next |
| --- | --- | --- |
| Architecture | Dense Qwen3.5-style trunk, 64 executed layers; MTP excluded | Qwen4Exp, 48 layers, 512 experts/layer, top 10, shared expert, n-gram embeddings |
| Weight policy | Resident output head, two 288 MiB layer buffers, embedding row reads | Resident ordinary/shared weights, bounded per-layer expert caches, n-gram row cache |
| Limits | 4 GiB process cap, 4096 context, default prefill chunk 512, maximum 1024 | At most 10 GiB and Metal's recommended working set, 4096 context, chunked prefill |
| Chat helper defaults | 4 GiB cap | 1536 MiB expert cache, 9 GiB cap |
| Persistent session | Exact consumed-prefix reuse, reset, JSONL events | Same single-owner protocol |
| Dominant recorded cost | 14,344,308,736 layer bytes read per ordinary decode token | Missed expert reads; effectiveness depends on route locality |

The native Flash CLI still defaults to a 2048 MiB expert cache. Start with the
helper's more conservative 1536 MiB setting on this host; those configurations
are not equivalent qualification.

Live inspection confirmed 16 GiB RAM and `/Volumes/Extreme` as an external USB
APFS SSD. About 1.6 GiB of swap was already occupied. Occupancy alone does not
measure a run's swap activity; time-window counter deltas are required. All
five GGUF sizes and the tokenizer, tokenizer-config and generation-config
hashes for both models matched their manifests. Full GGUF hashes were not
re-read in the initial inspection. The release-hardening follow-up below now
performs full runtime verification on both models.

Evidence in `zig/pkg/inference/scripts/`:

- `qwen35_streamed/qualification-m4.json`: independent NumPy FP32 equations
  with pinned ggml codecs, short and 128-token numerical comparisons, and a
  4096-token functional/memory test. The latter is not an independent
  full-context numerical comparison.
- `qwen_local/qualification-m4.json`: session, prefill-chunk, route-replay and
  prompt-lookup experiments.
- `qwen_local/gpu-dequant-m4.json`: matched CPU/GPU comparisons with exact
  source and binary identities. These are historical records, not fresh
  measurements of this review's binaries.

The historical 27B comparison eliminated about 2.91 seconds of CPU matrix
expansion per decode token but changed throughput only from roughly 0.0602 to
0.0605 tokens/s. Exposed read wait increased as CPU work stopped hiding I/O.
This supports prioritizing I/O reduction. A repeated-`x` lookup experiment
accepted 6/6 proposals and improved speed, but that favorable input does not
establish ordinary chat acceleration.

## Competitive research

Primary sources inspected on the review date. Project measurements are
author-reported, not reproduced here. Different hardware, model, precision,
memory residency and cache state prevent direct throughput comparisons.

| Source | Mechanism | Implication for Antfly |
| --- | --- | --- |
| [Apple: LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) | Reduce bytes transferred and favor larger contiguous reads; windowing and row/column bundling exploit activation sparsity | Adopt the cost-model discipline. Streaming dense 27B does not give it sparse routing; learned sparsity needs quality qualification. |
| [llama.cpp merged TENSOR_READ_LAZY](https://github.com/ggml-org/llama.cpp/pull/27794) and [Qwen PLE offload experiment](https://github.com/ggml-org/llama.cpp/discussions/27864) | Lazy access to large embedding tables; explicit n-gram row reads in the experiment | Strong n-gram baseline. Keeping most weights resident while paging PLE is not equivalent to paging Flash experts on 16 GiB. |
| [llama.cpp expert paging RFC/PoC](https://github.com/ggml-org/llama.cpp/discussions/23324) | Per-layer Metal pools, `pread`, route remapping, `MTLSharedEvent` synchronization | Closely matches our ownership problem. Study event-based overlap if profiles show exposed synchronization. This source is a PoC discussion, not a universal released offload backend. |
| [ANEMLL Flash MLX](https://github.com/Anemll/anemll-flash-mlx) | Packed expert sidecars, native workers, stable slot banks, temporal prefetch, resident/oracle comparisons | Best methodological match: measure locality, all-hit compute, miss cost and oracle prefetch before adopting a complex policy. Published M5 Max/128 GB and smaller-MoE results are not our baseline. |
| [FlashMoE](https://arxiv.org/abs/2601.17063) | Learned replacement combines recency and frequency | Compare against our existing frequency-aware decaying policy on held-out routes with equal byte budgets; a win against LRU elsewhere is insufficient. |
| [SpecExec](https://github.com/yandex-research/specexec), [Sequoia](https://arxiv.org/abs/2402.12374) | Verify several proposals per target pass to amortize offload | Promising for dense 27B. Hybrid recurrent state needs correct checkpoint/restore and replay; tree verification is not a drop-in KV-cache algorithm. |
| [FlexGen](https://arxiv.org/abs/2303.06865) | Throughput-oriented offload scheduling and batching | Supports layer-major prefill. Large offline batches optimize a different objective from one user's chat latency. |
| [MLX LM](https://github.com/ml-explore/mlx-lm) | Unified-memory inference and large-model memory wiring | Useful resident baseline when a compatible artifact fits. Raising the wired limit cannot safely make larger-than-RAM weights fit. |
| [MLX-Flash](https://github.com/szibis/mlx-flash) | Broad cache, offload and speculative feature collection | Treat feature tables as project claims. Its README also identifies unfinished cached LayerSkip and missing pretrained heads. Pruning and KV eviction need separate quality gates. |

Fair comparisons must pin revisions, exact artifacts, rendered token IDs,
precision, context, output length, memory budget and storage placement. A
competitor that cannot safely run the exact artifact is unsupported in that
configuration, not a zero-throughput sample. Compare different quantizations
separately with quality measurements.

## Optimization order

Use decimal bytes consistently:

`I/O service seconds per token = bytes read per token / sustained bytes per second`

With perfect overlap, the lower bound is approximately the larger of compute
and I/O service time; dependent misses and synchronization add unhidden time.
Application read ranges are not physical USB traffic. GPU execution overlaps
host completion waits, so those counters must not be added.

For illustration, 14.344 GB/token on a hypothetical sustained 1 GB/s path costs
14.344 seconds of I/O service. Kernel tuning cannot eliminate those bytes.
Measure the actual SSD before using this as a prediction. Persistent chat
saves reload and prefix work but not each dense decode weight pass.

A fresh read-only `qwen4exp/storage_probe.py` run read 96 MiB per case with
Darwin `F_NOCACHE`, varying 512 KiB, 1 MiB and 3 MiB ranges. Three workers
measured 0.951–0.974 GB/s; six measured 0.867–0.969 GB/s; one measured
0.412–0.804 GB/s. Swapin and swapout deltas were both zero. This short probe
does not establish sustained SSD or inference throughput; OS/device caches
were not purged and each case used different pseudorandom offsets. It supports
testing three versus six read lanes with matched real routes, not changing
the default based on this sweep. Raw record:
`/tmp/qwen-production-storage-probe.json`.

1. **27B resident-layer cache:** explicitly account for encoded layers, Metal
   aliases and runtime overhead. Pin a bounded subset; ordinary LRU on a full
   cyclic layer scan can thrash. Retain host headroom and fail admission when
   it does not fit. No implicit process-cap increase.
2. **Flash trace-driven allocation:** maximize bytes saved per cache byte,
   preserving route pins and transactional publication. Compare existing
   replacement, per-layer variants and an oracle on held-out traces. Existing
   global-LRU replay predicted worse decode reads, so do not switch blindly.
3. **Exact 27B multi-token verification:** measure accepted tokens per total
   target pass, including rejection replay, draft work and checkpoint memory.
   Keep lookup opt-in until ordinary workloads and lookup-enabled sessions pass.
4. **I/O layout and queue depth:** replay real range sizes/offsets with bounded
   workers. Consider projection coalescing or a versioned packed sidecar only
   when fewer calls outweigh additional bytes/storage. Preserve original GGUFs.
5. **Metal synchronization/prefill:** optimize exposed costs from profiles.
   Keep completion before slot reuse. Existing Flash MPS/batched-linear
   prototypes remain opt-in because recorded runs did not establish a benefit.

## Local release gates

| Priority | Finding at the starting checkout | Required evidence/status |
| --- | --- | --- |
| P0 | Flash sidecars are not verified before parsing | Fixed here: bounded size and SHA-256 verification, with corrupted/oversized/empty-file regression coverage. |
| P0 | Flash inventory/geometry checks do not verify all shard contents; its normal CLI does not enforce full hashes | Implemented full, bounded, cancellable SHA-256 verification of all four ordered shards before weight loading. Receipts and session readiness report verified identity. Runtime pins are tested against the artifact manifest. |
| P0 | Flash lacks 27B-equivalent cooperative SIGINT and staged failure receipts | Implemented process-scoped SIGINT/SIGTERM cancellation, checks during initialization/expert reads/Metal dispatch, staged failure receipts, and SIGPIPE handling. The supervised lifecycle campaign requires cooperative shutdown within 15 seconds. |
| P0 | Shared benchmark accepts missing receipts, checks only swapouts, and does not compare repeated greedy outputs | Fixed here for `qwen_local/benchmark.py`: strict receipts, times, token counts, memory, both swap counters and repeat checks; retain failure/timeout rows and reject output-directory reuse. Older experiment harnesses retain their narrower checks. |
| P0 | Focused Qwen storage/lifecycle and evidence tests absent from inspected workflow | Added Python and CPU-only Zig invocations to `.github/workflows/zig-tests.yml`; local validation only. CPU-only tests do not certify Metal. |
| P0 | Historical parity/4K functional checks are not a final-binary release matrix | Refresh independent full-vocabulary logits and state/continuation at short, chunk-boundary, QSA-boundary and near-limit contexts, with fixed tolerances and recorded identities. |
| P0 | No final-binary long-session/fault campaign demonstrated here | Bounded session and lifecycle campaigns now pass on both final binaries, plus pager fault fixtures. Extended endurance and clean-memory qualification remain open; eight repeat cycles cannot establish absence of slow memory growth. |
| P1 | Short diagnostics and uncontrolled caches do not establish sustained latency or answer quality | Fixed chat/code/document corpus, longer outputs, quality checks, repeated AB/BA runs, TTFT and p50/p95 inter-token latency. Declare an actual local-chat latency target before claiming it is met. |

Use fault injection or disposable fixtures for SSD loss; do not unplug the
active model volume. Blocking kernel reads can outlive cooperative cancellation,
so measure the bound and provide a supervisor timeout where necessary.

Useful foundations confirmed in code: dense slots reject reuse before GPU
completion and stale leases; expert replacement stages and rolls back failed
reads; execution errors poison decoder state; session reuse requires an exact
consumed-token prefix. These do not replace the release fault/soak campaign.

## Changes and validation

Added `util/verified_sidecar.zig`; the 27B wrapper preserves its existing error
names and Flash now applies the same checks. Flash's `sidecars_hash_verified`
receipt field deliberately does not claim GGUF verification.

Added strict shared-benchmark checks and adversarial tests for absent/corrupt
receipts, nonfinite/invalid timing, inconsistent token counts, missing memory
measurements, cap violations, swapins alone, counter resets, changed greedy
outputs and timeout reporting. Passing means usable benchmark evidence, not
independent numerical parity or production qualification.

Added lightweight Python and CPU-only Zig CI invocations. Initial review validation (before the release-hardening follow-up):

| Check | Result |
| --- | --- |
| Bounded ReleaseFast Metal build, both CLIs | Passed, 22/22 build steps including the initial test pass |
| Final focused Metal suites | 27B: 24 passed, 1 installed-model skip; Flash: 40 passed, 4 installed-model skips |
| Final CPU-only focused suites | 27B: 20 passed, 5 skipped; Flash: 27 passed, 16 skipped |
| New Python evidence tests | 6 passed, including adversarial subcases and harness failure reporting |
| Workflow validation-scope tests | 11 passed |
| Historical 27B receipts checked against new validator | 10 accepted; compatibility check only, not rerun inference |
| Changed source formatting, Ruff, license headers, diff whitespace | Passed |

Both CPU test roots explicitly import sidecar tests, so those tests cannot
disappear when Metal execution is compiled out. Logs are under
`/tmp/qwen-production-{metal-final,cpu,evidence}-tests.log`.
## Release-hardening follow-up

The runtime now verifies the complete Flash artifact, with a 4 MiB hashing
buffer and uncached reads. It verifies shard count, size, ordered SHA-256 pins,
and final descriptor size; cancellation is checked between reads. The installed
artifact must remain unchanged during execution. This is startup integrity
verification, not protection against concurrent replacement of the install.
Persistent sessions amortize the complete startup read.

Both runners install scoped SIGINT/SIGTERM handlers and ignore SIGPIPE so output
errors take the normal cleanup path. An idle session polls stdin in 100 ms
intervals. Flash checks cancellation during initialization and layer execution;
pager failures drain workers and roll back unpublished slots, while decoder
execution failures poison state. Disposable pager tests cover cancellation,
actual file truncation, and an unavailable descriptor without modifying the SSD.
27B additionally replaces misleading success receipts after streaming failures
and returns failure when a requested timing receipt cannot be written.

`qualify_lifecycle.py` supervises cancellation, idle termination, oversized input,
and both session and one-shot broken pipes. `qualify_sessions.py` adds repeated
reset/generate cycles, Unicode, malformed JSON, unknown fields, wrong types,
context overflow, bounded footprint variation and both swap counters. These
bounded checks do not substitute for a long-duration endurance run or a real
storage-driver hang test.

Fresh evidence is in
[`release-hardening-m4.json`](../../../../../zig/pkg/inference/scripts/qwen_local/release-hardening-m4.json),
with exact binary/source identities, reference commands and paths to raw receipts.
Completed campaign results:

| Final-build check | Result |
| --- | --- |
| Focused Metal suites | 27B 26 passed / 1 skipped; Flash 42 passed / 4 skipped |
| CPU-only suites | 27B 22 passed / 5 skipped; Flash 29 passed / 16 skipped |
| Python evidence and manifest tests | 7 passed |
| Workflow validation-scope tests | 11 passed |
| 27B lifecycle | All 8 cases passed; longest measured cooperative cancellation/session-fault exit 0.176 s |
| 27B raw/chat sessions | All 42 functional cases passed, including 8 reset/generate cycles in each mode; peak 2.654 GiB; post-warmup footprint range 16 KiB raw / 0 chat |
| 27B independent full-vocabulary parity | Short and 128-token prefill plus continuation passed; maximum relative L2 0.000005319 and all argmax IDs matched |
| 27B clean-memory gate | Failed: session window observed 4,030 swapins and 18,936 swapouts; numerical comparison window observed 703 swapins and no swapouts |
| Flash lifecycle | All 8 cases passed; longest measured cooperative cancellation/session-fault exit 0.251 s |
| Flash raw/chat sessions | All 42 functional cases passed, including 8 reset/generate cycles in each mode; peak 6.687 GiB under the 9 GiB cap; post-warmup footprint range 112 KiB raw / 416 KiB chat |
| Flash independent full-vocabulary parity | Continuation logits after a five-token prompt and one teacher-forced token passed; relative L2 0.0000007442 and matching argmax |
| Flash chunked state/continuation | Installed native sequential vs. chunks 8/4/4 passed across 16 tokens and token 17 continuation; 29,964,288 state values compared, maximum logit relative L2 0.0000008919. Suite: 43 passed / 3 skipped, including the 42 focused tests above. This is not an independent reference comparison. |
| Flash artifact corruption | Same-size mutation of a disposable first-shard copy rejected with `ArtifactHashMismatch` at `artifact_verification`; installed model files unchanged |
| Flash clean-memory gate | Failed: session window observed 12,157 swapins and no swapouts; numerical comparison window observed 4,868 swapins and no swapouts |

The additional installed-state test window observed 65,477 swapins and 360,340
swapouts, so its numerical success provides no clean-memory or timing claim.
Small footprint increases across Flash's eight repeat cycles are below the
bounded-test limit but cannot establish the absence of a slow leak. Extended
session endurance remains a release gate.

System-wide swap counters do not identify the responsible process. Functional
success and numerical agreement remain distinct from clean-memory performance
qualification. The final-binary independent near-limit numerical matrix remains
a release gate. The user explicitly chose bounded checks in this pass and
retained the multi-hour full 4K reference comparison as a later release gate;
short parity, synthetic QSA boundaries and historical 4K functional runs must
not be relabeled as independent full-model 4K parity.

No commit, push, release promotion or CI dispatch occurred. Keep both runners
experimental until the remaining P0 gates pass on the exact release binaries
on this 16 GiB host.

## PR preparation review

Reviewed the complete branch, including shared tokenizer/template, Metal dispatch,
storage and arena changes. The branch and all local hardening changes apply
cleanly to current main `b9feda47fb` in an isolated checkout. Both CLIs build
there; focused suites pass 68 Metal tests (5 skipped) and 51 CPU tests
(21 skipped), tokenizer tests pass 88 (2 skipped), and template suites pass 135.
This integration check does not replace the recorded full-model hardware campaign
or merge main into the user's branch.

Review fixed a qualification-harness timeout gap: a full child stdin pipe could
block the oversized-request check, and separate read/exit waits could exceed the
advertised bound. Nonblocking writes, reads and exit now share one 15-second
deadline; cleanup reaps a stalled child before closing its potentially full pipe.
Eight Python tests pass, including real OS-pipe timeout/closed-reader regression
coverage. The original hardware identities remain in the evidence file, with
the later harness-only changes recorded separately. Release gates remain open.
