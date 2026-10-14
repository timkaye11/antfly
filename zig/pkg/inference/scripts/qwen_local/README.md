# Local Qwen sessions and performance experiments

Run commands from the repository root. These are experimental single-sequence,
greedy, text-only Metal runners with a 4096-token context limit. Keep inference
and builds serial on the 16 GiB machine.

## Persistent terminal chat

```sh
MODELS=/Volumes/Extreme/antfly/models/unsloth  # wherever the models were copied
QWEN_27B_GGUF=$MODELS/Qwen3.8-27B-GGUF-UD-Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
QWEN_FLASH_FIRST_SHARD=$MODELS/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

python3 zig/pkg/inference/scripts/qwen_local/chat.py \
  --family 27b --binary zig/zig-out/bin/antfly-qwen35-streamed \
  --model "$QWEN_27B_GGUF"

python3 zig/pkg/inference/scripts/qwen_local/chat.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD"
```

The client streams text, keeps the model loaded between turns, and supports
`/reset` and `/quit`. Ctrl-C while a reply is streaming stops that reply and
keeps the session; at the prompt it exits. It disables thinking. The process
cap and the Flash expert cache come from the runner's machine profile unless
`--combined-budget-mb` or `--cache-mb` is given (see
[Machine profile and memory budgets](#machine-profile-and-memory-budgets)).
The client defaults to `--verify stamp`; see
[Artifact verification](#artifact-verification). Admission and live pressure
checks remain active. Other applications can still prevent admission or cause
swap; process caps are not system-wide reservations.

Sessions disable thinking; `--enable-thinking` is rejected because this protocol
does not yet retain separate reasoning messages.

The native protocol (`qwen-local-jsonl-v2`) uses positional prompt `-` with
`--session`. It accepts one JSON object per stdin line:
`{"prompt":"Hello","max_tokens":16}`, `{"reset":true}`, `{"cancel":true}`, or
`{"quit":true}`. Stdout contains `ready`, `delta`, `done`, `error`, and `reset`
JSON events; diagnostics go to stderr. `ready` reports the verification mode
and outcome, the context size and `process_cap_bytes`. `done` includes consumed
position, exact reused/prefilled token counts, timings, read bytes, token IDs,
and process memory. Concatenated `delta` text equals the final text: a delta
holds back only a UTF-8 sequence that may still complete, and invalid bytes
arrive as U+FFFD. There is no HTTP endpoint or concurrent request support.

Every request line except `cancel` gets exactly one terminal event. A `cancel`
stops the request that is executing, even when other requests were queued
before it: that request ends with `done` and `finish_reason: "cancelled"`,
carrying whatever text was already produced, and the queued requests then run
in order. A cancel with nothing to stop is ignored without a reply. A decode token always
completes, so a cancelled reply stops at the next token boundary with the
conversation state intact and the following turn reuses the prefix. A cancel
during a multi-token prefill chunk is honoured between layers; the sequence
state is then cleared (`state_reset: true`) and the next turn prefills its
whole prompt again. The loaded weights and the Flash expert cache are kept
either way.

`error` events carry `fatal`, `state_reset` and `retryable`. Argument errors
are nonfatal and leave everything untouched. Memory pressure is checked at
token boundaries and stops the turn with the consumed prefix intact. Pressure
between the layers of a multi-token chunk, or a transient read error that
survives the pager's bounded retries, clears the sequence state but keeps the
model (`fatal: false, state_reset: true`). A truncated or revoked artifact, any
Metal failure, an output write failure and SIGINT/SIGTERM remain fatal.

Chat requests append user turns and render the artifact template over history.
With `--raw-prompt`, every request supplies the entire desired raw prefix.
Reuse requires every previously consumed token to match. A mismatch resets
recurrent and KV state and recomputes the prompt. EOS and a final unconsumed
prediction are handled separately from the consumed prefix. `/reset` clears
conversation state while retaining the loaded model and Flash expert cache.

## Serial benchmark and qualification

```sh
python3 zig/pkg/inference/scripts/qwen_local/benchmark.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD" \
  --rounds 2 --trace --output /tmp/qwen-flash-baseline

python3 zig/pkg/inference/scripts/qwen_local/qualify_sessions.py \
  --family 27b --binary zig/zig-out/bin/antfly-qwen35-streamed \
  --model "$QWEN_27B_GGUF" --output /tmp/qwen-session-check
```

Without `--cache-mb` and `--combined-budget-mb` every harness lets the runner
derive its budgets. To reproduce the 16 GiB M4 records below, pass the caps
they were measured with: `--cache-mb 1536 --combined-budget-mb 9216` for Flash
and `--combined-budget-mb 4096` for 27B.

The shared benchmark includes chat, code, and document prompts. It records
binary identity, host, commands, receipts, output, and system swapin/swapout
deltas. Use a new output directory for every run. It rejects missing or invalid
receipts, invalid timings/counts, unavailable memory measurements, cap violations,
any swap-counter change, and changed token IDs across repeated greedy runs.
It also rejects a receipt whose enforced process cap or allocated expert cache
differs from what was requested. Failures and timeouts retain a partial summary
with `benchmark_checks_passed=false`. Rebuild both runners before using the
hardened harness. Flash also reports `sidecars_hash_verified=true`. The
benchmark defaults to `--verify stamp`, so only the first process on a host
hashes the artifact; `--verify full` hashes in every process. The chat client
allows 600 seconds for verification and initialization. Keep installed
artifacts unchanged during a run.

`qualify_sessions.py` checks raw and templated chat, exact prefix reuse, reset
parity, invalid requests, a cancelled reply followed by an unchanged greedy
turn, and eight fixed-workload reset/generate cycles per mode. It requires
identical greedy IDs, matching streamed/final text, the cap the runner reports
it enforces, at most 128 MiB footprint variation after warmup, and zero
swap-counter changes. These bounded cycles are a regression soak, not an
hours-long endurance test. Both qualification harnesses always run with
`--verify full`.

```sh
python3 zig/pkg/inference/scripts/qwen_local/qualify_lifecycle.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD" --output /tmp/qwen-flash-lifecycle
```

The lifecycle campaign separately exercises SIGINT during artifact verification,
prefill and decode, SIGTERM while waiting for stdin, oversized requests and a
closed output pipe, plus a deliberately unwritable timing destination. It
requires cancellation and session-fault exits within 15 seconds and retains
each receipt/diagnostic. Session-fault input writes, response reads and process
exit share one deadline; a full stdin pipe cannot block the supervisor. Startup
and complete one-shot fault cases have a
600-second deadline. A supervisor terminates, then kills, an unresponsive
child; that case fails qualification. Functional results and swap cleanliness are
reported separately. Runtime cancellation checks drain submitted Metal commands
and read workers before releasing their buffers. A blocked operating-system
read still depends on the storage driver's completion behavior.

The first generated token comes from prefill. Effective decode throughput counts
subsequent output tokens; executed decode steps separately include any final EOS
check. OS and SSD cache state are uncontrolled. Short outputs measure latency
and regression behavior, not answer quality or sustained throughput. The stricter
checks apply to this shared benchmark, not retroactively to older experiment
harnesses or historical JSON records.

See the [competitive research and local production review](../../../../../docs/design/inference/history/qwen/qwen38-ssd-production-review-2026-10.md)
for optimization priorities and the remaining release gates.

## Machine profile and memory budgets

`src/runtime/machine_profile.zig` is the one place that turns a host into
limits. It reads physical memory and Metal's recommended working set and
derives, unless overridden:

- **Process cap.** The finished memory plan plus 768 MiB (the admission
  controller's 512 MiB emergency reserve and 256 MiB for what the process
  already holds when it asks for admission). No derived cap exceeds physical
  memory minus the controller's node headroom: 10 GiB on a 16 GiB host, 36 GiB
  on 48 GiB.
- **Flash expert cache.** The largest whole-slot allocation that fits that
  envelope and Metal's working set, up to every expert (512 slots per layer).
  Slot indices are 16-bit; the old 128-slot ceiling is gone.
- **Context.** `max_context` (4096) is the kernels' qualified limit. State
  buffers and token histories are sized from the admitted context.

`--combined-budget-mb` and `--moe-cache-mb` are explicit overrides. They are
validated, never clamped: a cap above physical memory, a plan that does not fit
its cap together with the 512 MiB admission reserve, a cache budget below one
route per layer or above full residency, and GPU-visible bytes above Metal's
working set are all errors. An explicit cap may
exceed the derived machine cap. Receipts and the session `ready` event report
what was enforced: `process_cap_bytes`, `process_cap_source`,
`expert_cache_bytes`, `expert_cache_source`, `expert_slot_counts` and a
`machine_profile` object with the host facts.

`--dry-run` prints the plan either runner would admit, without verifying or
loading weights:

```sh
zig/zig-out/bin/antfly-qwen4exp "$QWEN_FLASH_FIRST_SHARD" - \
  --experimental-qwen4exp --backend metal --speculation-policy off --dry-run
```

On the 16 GiB M4 this derives a 3.71 GB expert cache (24–25 slots per layer,
4.8% of experts) under a 10.0 GiB envelope for Flash, and a 3.99 GiB envelope
for 27B. A derived Flash plan needs nearly all of that envelope to be free at
start; with other applications open, pass smaller explicit budgets. The 27B
runner still streams through two layer buffers at any cap: a larger cap only
loosens its bound until the resident mode lands.

## Artifact verification

`--verify full` (the runner default) hashes every pinned byte before loading
weights. Flash now hashes, and afterwards reads every weight range, through one
set of descriptors per shard, each checked to name the same file its metadata
was parsed from; a path replaced after parsing is rejected.

`--verify stamp` accepts an earlier full pass instead of hashing again. After
any complete full pass the runner writes a stamp under
`~/.antfly/inference/verification/` (override with
`ANTFLY_QWEN_VERIFY_STAMP_DIR`) recording, per file: absolute path, size,
pinned SHA-256, device, inode, modification time and a sampled content
fingerprint. A stamp is accepted only while all of those still hold for the
open descriptors and the pinned digests are unchanged; otherwise the runner
falls back to a full pass and rewrites it. Receipts report `verification_mode`,
`hash_verified` (hashed by this process) and `stamp_verified`. A stamp guards
against replacement and accidental change. It is not tamper evidence, it is
host-local, and copying a model to another machine invalidates it.

## Route traces and the cache-size sweep

```sh
python3 zig/pkg/inference/scripts/qwen4exp/record_routes.py \
  --binary zig/zig-out/bin/antfly-qwen4exp --model "$QWEN_FLASH_FIRST_SHARD" \
  --output /tmp/qwen-routes

python3 zig/pkg/inference/scripts/qwen4exp/cache_replay.py \
  --manifest /tmp/qwen-routes/manifest.json --output /tmp/qwen-routes/sweep.json
```

`record_routes.py` runs chat, code and document prompts (640 output tokens
each) and a five-turn chat session with `ANTFLY_QWEN4EXP_ROUTE_TRACE` set.
Routes do not depend on the cache size used while recording. The sweep replays
them at 1.6–36 GB under the runtime's policy (`lfu_decay_slots`), per-layer and
global LRU, expert-keyed LFU that remembers evicted experts (`lfu_ghost`), the
same with slots allocated from the other workloads' miss curves
(`lfu_ghost_weighted`), and per-layer and global furthest-next-use bounds
(`belady_*`). It prints predicted decode megabytes read per token at 30 GB for
the whole decode and for its second half, which approximates a warm cache.
These are application read ranges from an offline model, not throughput. Where
a workload has a receipt, the sweep first replays the runtime's policy at the
recorded slot counts and reports whether it reproduces the measured hits and
misses (`model_check`). The 16 GiB M4 recording and sweep are summarised in
[`../qwen4exp/route-replay-m4.json`](../qwen4exp/route-replay-m4.json) and
discussed in the handoff plan.

## Opt-in experiments

* Flash: `ANTFLY_QWEN4EXP_BATCH_LINEAR=1` batches the linear-attention input
  and output projections during prefill while preserving sequential recurrence.
  `ANTFLY_QWEN4EXP_PREFILL_MPS=1` uses bounded FP32 dequantization tiles and MPS
  for quantized matrices with at least eight rows; smaller batches keep the
  packed path. Benchmark flags are `--batch-linear --prefill-mps`.
* 27B: `--prefill-chunk-size 1024` permits a larger bounded chunk. The default
  remains 512. Scratch admission includes the larger buffers.
* 27B: `--prompt-lookup 4 --speculation-policy force` enables a prompt-lookup
  prototype (1–4 proposals). It verifies proposals with the target model and
  checkpoints recurrent state; rejection restores and replays the accepted
  prefix. No draft model is loaded. Repetitive inputs may benefit; rejection
  adds work. For lookup receipts, `decode_steps` counts decoder advance calls;
  effective decode throughput uses `(generated_tokens - 1) / decode_seconds`.
  Replays add target passes within an advance, visible in `lookup_stats` and
  read-byte totals. The default is off. The checkpoint consumes approximately 150 MiB
  extra host memory and must fit admission. Combining the largest chunk and
  lookup may exceed the 4 GiB envelope.

Replay Flash routes with `../qwen4exp/cache_replay.py TRACE RECEIPT`. Replay
predicts I/O for per-layer and global LRU, and re-runs the runtime's own
replacement policy as a check on the model; it is not measured throughput.
Current measurements and decisions are recorded in `qualification-m4.json`
beside this file. For the hit-rate-versus-cache-size question, see
[Route traces and the cache-size sweep](#route-traces-and-the-cache-size-sweep).

## M4 observations (2026-10-09)

Final chat checks passed for both models: second-turn reuse of 20 consumed
tokens, matching streamed/final text, nonfatal invalid requests, and reset
parity, with zero new system swapouts. The tokenizer fix also passed the full
Hugging Face tokenizer test target. Flash's native `--stream` output matched
reference decoding in all three benchmark cases.

The Flash batch-linear + MPS prototype preserved generated tokens, but prefill
was 30.92/31.24/189.42 seconds on chat/code/document versus 29.30/30.09/182.53
seconds in the control. It remains opt-in; these runs do not establish a speed
benefit. Full-state/logit parity passed against sequential execution, including
an eight-row MPS chunk and subsequent decode. Per-layer expert caching remains
unchanged because global LRU replay predicted more decode reads.

On the synthetic repeated-`x` prompt, 27B lookup generated the same eight tokens
in 70.08 seconds end-to-end versus 153.59 seconds. Decode alone fell from
117.60 to 34.32 seconds (0.060 to 0.204 effective tokens/s). It accepted 6/6
proposals, reduced decode layer reads from 100.41 to 28.69 GB, and peaked at
2.83 GiB versus 2.68 GiB. This deliberately favorable workload is not evidence
of a general chat speedup. Forced acceptance and rejection tests separately
verified full-vocabulary continuation and recurrent-state restoration.

These are single-run local observations with uncontrolled OS/device caches.
The 2 GiB Flash-cache timing run and one 27B chunk-control run were excluded
because system swapouts increased. The retained benchmarks use a 1536 MiB
Flash cache and report their own swapout counters.

A 1,024-token document prompt took 48.28 seconds to prefill with 512-token
chunks and 43.28 seconds with one 1024-token chunk. Both the final prefill logits
and the following decode logits were bit-identical across all 248,320 entries.
Layer reads halved from 28.69 to 14.34 GB; peak footprint rose from 2.68 to
2.88 GiB. Both retained runs had zero swapouts. The default remains 512;
1024 is an optional modest speed/memory tradeoff, not a new 4096-token timing
qualification. Lookup was qualified through the one-shot CLI; session checks
used lookup off.

## 27B GPU weight dequantization

The streamed 27B runner now decodes IQ3_S and IQ4_XS directly in Metal for
small matrix-vector batches. Batched prefill uses GPU-decoded FP32 tiles for
these formats and Q3_K, followed by MPS GEMM. The existing 512-row tile is
reused in GPU command order, eliminating CPU tile expansion and the associated
per-tile completion waits. Layer-buffer release, CPU output access, and bounded
recurrent-command batches retain their completion barriers. The only extra
logical allocation is the 2 KiB IQ3_S codebook (page-rounded by the allocator).
Input embedding rows still use bounded CPU decoding; `cpu_dequant_bytes` and
`cpu_weight_dequantization` describe matrix weights, not embeddings.

`ANTFLY_QWEN35_CPU_DEQUANT=1` restores the previous matrix fallback for A/B
comparison. `ANTFLY_QWEN35_PROFILE=1` records separate `prefill_profile` and
`decode_profile` counters: CPU matrix dequantization, host GPU-completion wait,
GPU execution, completed batches, and exposed layer-read wait. GPU execution
and host wait overlap and must not be added together. Read wait is the time
blocked joining prefetched reads, not total disk-service time. Timing runs
should omit profiling and captures; both options remain off by default.

```sh
python3 zig/pkg/inference/scripts/qwen_local/qualify_gpu_dequant.py \
  --binary zig/zig-out/bin/antfly-qwen35-streamed --model "$QWEN_27B_GGUF" \
  --order cpu gpu --profile --capture --output /tmp/qwen-dequant-parity

python3 zig/pkg/inference/scripts/qwen_local/qualify_gpu_dequant.py \
  --binary zig/zig-out/bin/antfly-qwen35-streamed --model "$QWEN_27B_GGUF" \
  --order cpu gpu gpu cpu --output /tmp/qwen-dequant-timing
```

The parity run requires matching prompt/output IDs, finite 248,320-entry
logit vectors, matching argmax, and relative L2 error below 0.001. The harness
also requires zero matrix CPU-dequantization bytes on the GPU path and no new
system swapouts. Independent ggml codec fixtures are covered separately by
the Metal tests, including nonzero buffer offsets, actual model widths,
multiple staging-tile reuse, partial final tiles, and batches through 1,024 rows.

### M4 GPU dequantization observations (2026-10-10)

The new GPU path is enabled by default. Unprofiled CPU/GPU/GPU/CPU runs on the
same 33-token chat prompt (four output tokens, three decode steps) measured
16.59–16.63 seconds per decoded token for the CPU control and 16.52–16.53 for
the GPU path: 0.0602 versus 0.0605 tokens/s. Prefill remained approximately
16.7–16.8 seconds. This roughly 0.5% decode difference does not establish a
material throughput gain. All four runs matched token IDs and layer-read
bytes, had zero new system swapouts, and peaked at 2.65–2.68 GiB footprint.

A separate profiled pair explains the result. Per decode token, CPU matrix
expansion fell from 2.91 seconds to zero, completed batches from 3,080 to 65,
and host GPU-completion waits from 2.81 to 1.66 seconds. Exposed layer-read
wait rose from 10.86 to 14.80 seconds as CPU work stopped hiding that I/O.
GPU execution was 1.71 versus 1.61 seconds (overlapping host waits). Weight
streaming still determines throughput. All four 248,320-entry logit captures
in this pair were bit-identical. The focused Metal suite passed 24 tests,
including installed-model forced draft acceptance and rejection.

Flash's fresh diagnostic profile completed, but system swapouts increased
by 57,756 pages, so it is excluded from clean performance conclusions.
Flash's router boundary is required for CPU expert selection; its current
resident-expert work already overlaps missing-expert reads. No Flash runtime
change is included in this pass. See [gpu-dequant-m4.json](gpu-dequant-m4.json)
for binary/source identities, receipts, counter definitions, and limits.

The 356-token document prompt benefited more: the captured CPU/GPU pair took
20.01/17.27 seconds to prefill, with bit-identical full-vocabulary logits both
after prefill and after the following decode step. A reverse-order GPU/CPU
repeat without captures or profiling measured 19.03/21.91 seconds—about 13%
less prefill time on GPU. These are two short local pairs with uncontrolled
OS/device caches, not a general long-context speed guarantee. Decode remained
16.57–16.68 seconds/token. All four document runs had zero new system swapouts.

## Handoff to another Mac

Written for the move from the 16 GiB M4 to a 48 GB Mac Studio; the plan and its
phases are in
[`qwen38-studio-handoff-plan-2026-10.md`](../../../../../docs/design/inference/history/qwen/qwen38-studio-handoff-plan-2026-10.md).
Use Apple Silicon with macOS 15+, Zig 0.17.0 and the Metal build. These runners
are experimental local text CLIs, not HTTP serving or CUDA implementations.

**Copy.** Keep the exact GGUFs and tokenizer sidecars from
`../qwen35_streamed/artifact.json` and `../qwen4exp/artifact.json`. The 27B
GGUF is about 16.46 GB; Flash needs all four shards, about 111.33 GB in total.
Flash's first shard is a small index and is not sufficient by itself. Preserve
its `UD-Q4_K_XL/` subdirectory, with tokenizer/config sidecars in its parent.
The 27B sidecars may be beside the GGUF or one directory above it. Storage
placement decides Flash throughput: prefer internal NVMe or Thunderbolt over
USB, and record which was used.

**Rebuild and verify on the destination.** Do not transfer build caches or
binaries.

```sh
# Repository root; set these to the destination's model directories.
export QWEN_27B_DIR=/path/to/Qwen3.8-27B-GGUF-UD-Q4_K_M
export QWEN_FLASH_DIR=/path/to/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL
export QWEN_27B_GGUF="$QWEN_27B_DIR/Qwen3.8-27B-UD-Q4_K_M.gguf"
export QWEN_FLASH_FIRST_SHARD="$QWEN_FLASH_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"

(cd zig && python3 tools/run_bounded_zig_build.py \
  build inference-test-qwen35-streamed inference-test-qwen4exp \
  inference-qwen35-streamed inference-qwen4exp -j1 -Dmetal=true -Doptimize=ReleaseFast)

python3 zig/pkg/inference/scripts/qwen_local/verify_install.py \
  --flash-dir "$QWEN_FLASH_DIR" --27b-dir "$QWEN_27B_DIR" \
  --flash-binary zig/zig-out/bin/antfly-qwen4exp \
  --27b-binary zig/zig-out/bin/antfly-qwen35-streamed \
  --output /tmp/qwen-install.json
```

`verify_install.py` checks every manifest file's layout, size and SHA-256 and
records chip, memory, macOS, the bus each model sits on, Metal's recommended
working set and the plan each runner derives there. Keep its report with the
host's evidence. On a 16 GiB host add `ANTFLY_ZIG_MAX_RSS=8589934592` to the
build.

**Pinned in source; a different artifact needs a code change and
requalification:**

- GGUF sizes and digests: `src/models/qwen4exp.zig` (`artifact_shards`,
  `artifact_revision`) and `src/models/qwen35_streamed.zig` (`artifact_bytes`,
  `artifact_sha256`). `test_evidence.py` checks the Flash pins against the
  manifest.
- Sidecar sizes and digests for `tokenizer.json`, `tokenizer_config.json` and
  `generation_config.json`, shared by both models, in
  `src/qwen_local_sidecars.zig`.
- Vocabulary size 248320 and the exact tensor inventories and geometries
  (`Inventory.validateRequired`, `models/qwen35_streamed.zig` `inspect`).
- The Qwen pre-tokenizer Split pattern is matched by exact string in
  `zig/lib/tokenizer/src/hf_tokenizer.zig`; another pattern silently falls back
  to the GPT-2 scanner. The digest pin above is what protects these two models.

**Not portable; regenerate on the destination:**

- Verification stamps (`~/.antfly/inference/verification/`): keyed by path,
  device and inode, so the first run on a new host always hashes in full.
- Every `*-m4.json` evidence file here and under `../qwen35_streamed/`: those
  are measurements of the 16 GiB M4 and its binaries. Do not treat additional
  memory as a measured speedup.
- The llama.cpp reference build used for parity (it lived under `/tmp`).
  Rebuild it at `llama_cpp_revision` from `../qwen4exp/artifact.json`, with the
  `init_mappings` change described in `../../tests/README.md`.
- Raw receipts, traces and logs referenced by path from the evidence files.

**Bring-up order.** Reproduce the old host first, then widen:

1. Focused suites (built above), then `qualify_sessions.py` and
   `qualify_lifecycle.py` for both families at the old caps
   (`--cache-mb 1536 --combined-budget-mb 9216` for Flash,
   `--combined-budget-mb 4096` for 27B), then the parity checks in
   `../../tests/README.md`. The token IDs must match the M4 records.
2. `storage_probe.py` on each candidate volume.
3. The same harnesses with no budget flags, so the runner derives them;
   compare `--dry-run` output with the report from `verify_install.py`.
4. `record_routes.py` and the cache-size sweep at the derived cache.

Admission or live pressure can reject a run below any cap. Stop a cache sweep
if swap counters increase, and do not remove resource checks to force a run.
The experiment harness requires NumPy and Hugging Face `tokenizers`; the chat,
benchmark, session, lifecycle and install harnesses use only the standard
library. Installed-model gates in the Zig suites are opt-in:
`ANTFLY_QWEN35_VERIFY_DRAFT="$QWEN_27B_GGUF"` with
`inference-test-qwen35-streamed`, and
`ANTFLY_QWEN4EXP_PREFILL_CHECK="$QWEN_FLASH_FIRST_SHARD"`,
`ANTFLY_QWEN4EXP_PREFILL_CHUNK=128`, `ANTFLY_QWEN4EXP_TEST_CACHE_MB=1536`
with `inference-test-qwen4exp`. Unset these variables after testing. The normal
focused suites explicitly skip installed-model gates when artifacts are not
selected; skipped tests are not hardware qualification.
