# Local Qwen sessions and performance experiments

Run commands from the repository root. These are experimental single-sequence,
greedy, text-only Metal runners with a 4096-token context limit. Keep inference
and builds serial on the 16 GiB machine.

## Persistent terminal chat

```sh
QWEN_27B_GGUF=/Volumes/Extreme/antfly/models/unsloth/Qwen3.8-27B-GGUF-UD-Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
QWEN_FLASH_FIRST_SHARD=/Volumes/Extreme/antfly/models/unsloth/Qwen3.8-Flash-Next-GGUF-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

python3 zig/pkg/inference/scripts/qwen_local/chat.py \
  --family 27b --binary zig/zig-out/bin/antfly-qwen35-streamed \
  --model "$QWEN_27B_GGUF"

python3 zig/pkg/inference/scripts/qwen_local/chat.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD"
```

The client streams text, keeps the model loaded between turns, and supports
`/reset` and `/quit`. It disables thinking. The Flash client uses a 1536 MiB
expert cache and a 9 GiB process cap; the 27B runner uses its 4 GiB cap.
Admission and live pressure checks remain active. Other applications can still
prevent admission or cause swap; process caps are not system-wide reservations.

Sessions disable thinking; `--enable-thinking` is rejected because this protocol
does not yet retain separate reasoning messages.

The native protocol uses positional prompt `-` with `--session`. It accepts
one JSON object per stdin line: `{"prompt":"Hello","max_tokens":16}`,
`{"reset":true}`, or `{"quit":true}`. Stdout contains `ready`, `delta`,
`done`, `error`, and `reset` JSON events; diagnostics go to stderr. `done`
includes consumed position, exact reused/prefilled token counts, timings,
read bytes, token IDs, and process memory. Argument errors are nonfatal;
execution errors invalidate the session and terminate it. There is no HTTP
endpoint or concurrent request support.

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
  --model "$QWEN_FLASH_FIRST_SHARD" --cache-mb 1536 \
  --rounds 2 --trace --output /tmp/qwen-flash-baseline

python3 zig/pkg/inference/scripts/qwen_local/qualify_sessions.py \
  --family 27b --binary zig/zig-out/bin/antfly-qwen35-streamed \
  --model "$QWEN_27B_GGUF" --output /tmp/qwen-session-check
```

The shared benchmark includes chat, code, and document prompts. It records
binary identity, host, commands, receipts, output, and system swapin/swapout
deltas. Use a new output directory for every run. It rejects missing or invalid
receipts, invalid timings/counts, unavailable memory measurements, cap violations,
any swap-counter change, and changed token IDs across repeated greedy runs.
Failures and timeouts retain a partial summary with `benchmark_checks_passed=false`.
Rebuild both runners before using the hardened harness. It requires runtime
`hash_verified=true`; Flash also reports `sidecars_hash_verified=true`. Both
runners hash their entire pinned GGUF artifacts before loading model weights.
Flash verifies all four shards, including its metadata-only first shard. This
adds a complete SSD read at startup; the chat client allows 600 seconds for
verification and initialization. Keep installed artifacts unchanged during a run.

`qualify_sessions.py` checks raw and templated chat, exact prefix reuse, reset
parity, invalid requests, and eight fixed-workload reset/generate cycles per mode.
It requires identical greedy IDs, matching streamed/final text, a measured memory
cap, at most 128 MiB footprint variation after warmup, and zero swap-counter
changes. These bounded cycles are a regression soak, not an hours-long endurance
test.

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
predicts I/O for per-layer and global LRU; the runtime policy differs, so this
is not measured throughput. Current measurements and decisions are recorded
in `qualification-m4.json` beside this file.

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

## Moving to a 24 GiB Mac mini

Use Apple Silicon with macOS 15+, Zig 0.17.0 and the Metal build. These runners
are experimental local text CLIs, not HTTP serving or CUDA implementations.
Rebuild on the destination. Record its chip, RAM, OS, model volume, executable
hashes and system swapout deltas; the M4/16 GiB records above remain historical
measurements tied to their own binaries. Do not transfer build caches or treat
additional RAM as a measured speedup.

Keep the exact GGUFs and tokenizer sidecars from
`../qwen35_streamed/artifact.json` and `../qwen4exp/artifact.json`. The 27B GGUF
is about 16.46 GB; Flash needs all four shards, about 111.33 GB in total.
Flash's first shard is a small index and is not sufficient by itself. Preserve
its `UD-Q4_K_XL/` subdirectory, with tokenizer/config sidecars in its parent.
The 27B sidecars may be beside the GGUF or one directory above it. Verify file
sizes and SHA-256 after copying. Storage placement must be recorded: moving
from an external SSD to internal storage changes the performance comparison.

The helper tools now accept `--combined-budget-mb`. Defaults remain 4096 MiB
for 27B and 9216 MiB for Flash; the benchmark's default expert cache is 1536
MiB, matching the chat client. Current runtime limits still apply:

- 27B: maximum 4096 MiB process cap, two streamed layer buffers, context up to
  4096, prefill chunks up to 1024. More RAM does not automatically cache layers.
- Flash: maximum 10240 MiB process cap, also bounded by Metal's recommended
  working-set size; per-layer expert cache currently supports at most 128 slots
  out of 512 experts. Larger cache flags do not enable full model residency.
- Admission or live pressure can reject a run even below these caps. Stop a
  cache sweep if swapouts increase. Do not remove resource checks to force it.

Start with the focused tests and existing baseline settings, running models
and builds serially. Use fresh output directories for each check:

```sh
# Repository root; set these to the destination's exact installed files.
export QWEN_27B_GGUF=/path/to/Qwen3.8-27B-UD-Q4_K_M.gguf
export QWEN_FLASH_FIRST_SHARD=/path/to/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

(cd zig && ANTFLY_ZIG_MAX_RSS=8589934592 python3 tools/run_bounded_zig_build.py \
  build inference-test-qwen35-streamed inference-test-qwen4exp \
  inference-qwen35-streamed inference-qwen4exp -j1 -Dmetal=true -Doptimize=ReleaseFast)

python3 zig/pkg/inference/scripts/qwen_local/qualify_sessions.py \
  --family 27b --binary zig/zig-out/bin/antfly-qwen35-streamed \
  --model "$QWEN_27B_GGUF" --output /tmp/qwen-mini-session27
python3 zig/pkg/inference/scripts/qwen_local/qualify_sessions.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD" --output /tmp/qwen-mini-sessionflash
```

Then benchmark Flash at 1536, 2048, 3072 and 3584 MiB expert cache, using a
10240 MiB process cap for the sweep. Keep prompts, output lengths and other
options fixed; repeat the 1536 MiB baseline at the end to expose cache/order
effects. Each invocation below runs two rounds of chat/code/document prompts;
replace CACHE with one size at a time and stop on failure or swap growth:

```sh
python3 zig/pkg/inference/scripts/qwen_local/benchmark.py \
  --family flash --binary zig/zig-out/bin/antfly-qwen4exp \
  --model "$QWEN_FLASH_FIRST_SHARD" --cache-mb CACHE --combined-budget-mb 10240 \
  --rounds 2 --tokens 8 --trace --output /tmp/qwen-mini-flash-CACHE
```

For 27B, run `qualify_gpu_dequant.py --profile --capture --order cpu gpu`
first, then an unprofiled comparison and `qualify_experiments.py --mode chunks`.
The experiment harness requires NumPy and Hugging Face `tokenizers`; the chat,
benchmark and GPU-dequantization harnesses use only Python's standard library.
Installed-model gates are opt-in: `ANTFLY_QWEN35_VERIFY_DRAFT="$QWEN_27B_GGUF"`
with `inference-test-qwen35-streamed`, and
`ANTFLY_QWEN4EXP_PREFILL_CHECK="$QWEN_FLASH_FIRST_SHARD"`,
`ANTFLY_QWEN4EXP_PREFILL_CHUNK=128`, `ANTFLY_QWEN4EXP_TEST_CACHE_MB=1536`
with `inference-test-qwen4exp`. Unset these variables after testing. The normal
focused suites explicitly skip installed-model gates when artifacts are not
selected; skipped tests are not hardware qualification.

Next implementation work, after destination measurements: evaluate a bounded
27B layer cache with an explicit larger admission plan, and use Flash route
traces to choose cache allocation. Keep Flash's batch-linear/MPS experiment
and 27B prompt lookup opt-in. Do not assume the 16 GiB timings, full-context
parity, lookup-enabled sessions, larger memory plans, or production serving
have been qualified on the new machine.
