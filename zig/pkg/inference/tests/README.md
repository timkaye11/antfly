# Inference cancellation E2E

Run from `zig/` with `zig build inference-test-cancellation-e2e`, or from
`zig/pkg/inference/` with `zig build test-cancellation-e2e`. The unfiltered
`inference-test` / package `test` aggregate also runs this suite on native Linux
and macOS. It needs Python 3, but no downloaded model, GPU, or Python packages.

The fixture runs production HTTP routes, tokenization, request/weighted/run
admission, execution control, the hard-cancellation watchdog, and the process
supervisor. Only the model session is synthetic. Its control routes exist only
in this test executable, which is not installed or linked into the server.

The embedding cases first verify a successful embedding, arm a blocking model call,
waits until the real HTTP request owns admission and scratch memory, and then
aborts its TCP client with a reset (RST):

- A cooperative session must observe cancellation, unwind its permits, and
  continue serving in the same worker.
- An uninterruptible session never checks cancellation or requests a restart;
  the production watchdog must terminate it and the supervisor must replace it.

Both paths must return to baseline resource accounting and serve the same
embedding successfully afterward. All network waits are bounded, and cleanup
terminates the fixture process group even on failure.

By default, an orderly TCP half-close (FIN) is not cancellation: an HTTP/1
client may still be waiting to read the response. The embedding tests use RST
to establish response abandonment, matching the default transport contract.

The HTTP reranker and linked reranker cases abort their requests with a TCP
reset. Their synthetic GPU batch returns only
after the test releases it. The worker must remain alive while the batch is
in progress, then observe cancellation and release admission before the next
rerank request.

The embedding case exercises the same safe native-call grace through the
shared inference control. A cancelled request releases at the end of its
current call without replacing the worker; a genuinely stuck call still
restarts the worker after the bounded grace.

This covers transport-abort propagation through model execution. It does not
exercise real GPU drivers, cold model loading, or application-deadline wiring;
those need separate coverage.

## Experimental Qwen4Exp SSD decoder

The opt-in text path supports the exact 48-layer, 512-expert Qwen3.8-Flash-Next
GGUF geometry recorded in `../scripts/qwen4exp/artifact.json`. It uses Metal
numerical kernels, a bounded cache of encoded expert projections, and uncached
range reads for misses, ngram rows, and the single input embedding row.
A route lease pins all selected experts
until GPU completion. Failed reads preserve the previous cache contents.
This is one-sequence eager greedy generation, with a 4096-token maximum;
HTTP serving, vision, MTP, sequence batching and other geometries are unsupported.
Metal commands for each attention branch and shared/router computation run in
one batch, followed by a batch for the selected experts and residual update.
The final output projection has its own batch. CPU route reads
and expert lease release wait for GPU completion. This changes submission
scheduling without changing kernel math.
Bounded Metal allocations share an explicit residency set on macOS 15 or later;
no full expert tensor or model mmap is added. The CLI owns six I/O executor
lanes with 256 KiB stacks, reserved before construction. Projection reads write
disjoint staging regions and all lanes join before publishing any cache change.
The router and final greedy selection use one SIMD group with deterministic
lowest-ID tie breaks and nonfinite-value rejection.
RMS/L2 normalization also uses one SIMD group per normalization group. Its
parallel reduction changes FP32 summation order, so qualification compares
logits against the pinned reference tolerances rather than requiring identical
floating-point bytes from the earlier scalar reduction.

From `zig/`, build the focused CLI and run it using the first split shard:

```sh
ANTFLY_ZIG_MAX_RSS=8589934592 python3 tools/run_bounded_zig_build.py build inference-qwen4exp -j1 -Dmetal=true
zig-out/bin/antfly-qwen4exp "$QWEN_GGUF" 'Hello' \
  --experimental-qwen4exp --backend metal --mode eager \
  --speculation-policy off --context-size 4096 --moe-cache-mb 2048 \
  --combined-budget-mb 10240 --max-tokens 8 --disable-thinking \
  --print-timing --print-token-ids --json-timing /tmp/qwen4exp-run.json
```

The same arguments are accepted by `antfly inference generate`. The focused
binary shares that parser and implementation. The tokenizer sidecars must be
beside the shard directory or beside the shard itself. Use `--raw-prompt` for
tokenizer-only input without the artifact's chat template.

The 10 GiB process cap is also clamped to Metal's recommended working set.
Explicit budgets retain live pressure checks. Default expert storage is 2 GiB,
ngram storage 64 MiB, and I/O staging 64 MiB; tokenizer parsing and caches have
a separate 512 MiB allocation ceiling included in admission. These budgets do
not promise admission while other applications consume the machine's memory.
For a larger expert cache, use `--moe-cache-mb 3584` with the same
`--combined-budget-mb 10240`. Admission still preserves the emergency reserve;
a 4096 MiB expert cache does not fit that envelope with context 4096.
Receipts report process footprint (the relevant macOS counter), cache hits,
misses, bytes read and load/prefill/decode times. Decode throughput is
`decode_steps * 1000 / decode_ms`; the first generated token comes from prefill.
`metal_command_batching` records the submission mode used for that run.

Set `ANTFLY_QWEN4EXP_DISABLE_BATCHING=1` to compare synchronous submissions.
`ANTFLY_QWEN4EXP_DISABLE_RESIDENCY=1` opts out of the explicit residency set,
and `ANTFLY_QWEN4EXP_DISABLE_PARALLEL_READS=1` uses one read lane for comparison.
Debug captures also disable command batching so every capture sees completed
data. `ANTFLY_QWEN4EXP_PROFILE=1` adds host wall-time counters to the receipt:
indices 0..27 are shader opcodes, 28 is quantized matvec, 29 is expert paging,
and 30 is batch submit/wait. Within batches, opcode counters measure encoding
time; GPU completion time is charged to index 30. Compare identical prompts,
context/cache budgets and output lengths, with only one model process running.
`batch_wait_ns` and `batch_gpu_ns` split completion waits and actual GPU time
into attention plus shared/router, an unused slot, routed experts plus the
residual update, and final output, in that order.

Performance experiments also support:

- `ANTFLY_QWEN4EXP_IO_OVERLAP=1`: opt into required-read overlap.
  `ANTFLY_QWEN4EXP_DISABLE_IO_OVERLAP=1` forces the baseline scheduler.
  The overlapping scheduler computes immutable resident experts while required
  reads fill disjoint staging. It joins reads before publishing slots and drains
  GPU readers before releasing leases on both success and failure. Contributions
  are accumulated in original router order, independent of read completion.
- `ANTFLY_QWEN4EXP_DISABLE_PREFILL_HEAD_SKIP=1`: compute the otherwise-unused
  vocabulary head at every prompt position for a matched control.
- `ANTFLY_QWEN4EXP_GROUPED_EXPERTS=1`: experimental indexed mixed-format expert
  kernels. These are opt-in: the first full-model pilot was slower than existing
  matvec kernels. This setting requires batching and read overlap.
- `ANTFLY_QWEN4EXP_PREFILL_CHUNK=128`: opt-in layer-major prefill (2..128 tokens).
  Recurrent and sparse-attention updates stay in causal order. Each layer's expert
  union is acquired in groups of ten and evaluated in matrix batches of at most
  32 rows, preserving one load per expert per chunk.
  Dense attention/HC projections remain token-wise. Chunk storage is at most
  22.52 MiB inside the existing 64 MiB backend scratch reservation. Prefix caching
  and MTP are not implemented. The pinned GGUF has no MTP tensors.
- `ANTFLY_QWEN4EXP_ROUTE_TRACE=/tmp/routes.u16`: diagnostic little-endian u16
  `[position,48,10]` route trace. `scripts/qwen4exp/cache_replay.py` compares the
  existing measured policy with per-layer and global byte-budget LRU. It replays
  layer-major prefill acquisitions when enabled; predicted read savings are not
  runtime speed measurements.

Receipts include build mode, actual feature switches, per-phase expert I/O,
per-decode-step timings and time until the first token is ready. That last metric
is not streamed delivery latency: the CLI currently prints the completed text.
Profile indices 31/32 time host encoding of experimental grouped projection and
activation. With overlap, GPU durations, paging waits and host encoding overlap;
do not sum them as independent elapsed-time components.

Build an optimized binary with `-Doptimize=ReleaseFast`, then compare immutable
copies of baseline/candidate binaries with the serial benchmark harness:

```sh
python3 pkg/inference/scripts/qwen4exp/benchmark.py \
  --binary baseline=/tmp/qwen-baseline --binary candidate=/tmp/qwen-candidate \
  --model "$QWEN_GGUF" \
  --cases pkg/inference/scripts/qwen4exp/benchmark_cases.json \
  --out /tmp/qwen-comparison --rounds 4 --cache-mb 2048
```

The harness alternates A/B and B/A, preserves every receipt/output and hashes the
binaries and cases. Each sample has fresh application caches; OS cache state is
uncontrolled. It records system swap counters and marks affected observations.
Use `--profile` only for a separate diagnostic series. Use `--variant-env
candidate:ANTFLY_QWEN4EXP_PREFILL_CHUNK=128` to enable a candidate option. The
bundled cases cover chat, code, JSON and prose; actual input lengths are recorded,
not padded or represented as long-context qualification. Custom case files can
supply longer prompts within the 4096-token total sequence cap.

`scripts/qwen4exp/storage_probe.py` compares bounded parallel range reads without
writing the model or purging global caches. F_NOCACHE requests no caching; existing
OS/device caches are not invalidated, and the report does not claim physical SSD
traffic. Prefer distinct ranges and enough transferred bytes over tiny repeated
reads. The first 384 MiB-per-case SanDisk probe measured approximately 0.92–0.98
GB/s at 3–6 readers for 512 KiB–3 MiB ranges. Larger contiguous records alone did
not show a large bandwidth gain in that probe.

Run the focused synthetic suite with `inference-test-qwen4exp -Dmetal=true`
(or `-Dmetal=false` for storage/CLI tests). Installed-artifact tests are opt-in:

```sh
ANTFLY_QWEN4EXP_GGUF="$QWEN_GGUF" ANTFLY_ZIG_MAX_RSS=8589934592 \
  python3 tools/run_bounded_zig_build.py build inference-test-qwen4exp -j1 -Dmetal=true
ANTFLY_QWEN4EXP_RUN_GGUF="$QWEN_GGUF" \
ANTFLY_QWEN4EXP_TOKENS=9419,11,353,9707,16451,13,9419,11,353,248044,9419,11 \
ANTFLY_QWEN4EXP_LOGITS=/tmp/qwen-native.f32 ANTFLY_ZIG_MAX_RSS=8589934592 \
  python3 tools/run_bounded_zig_build.py build inference-test-qwen4exp -j1 \
  -Dmetal=true -Dtest-filter='installed decoder'
```

`scripts/qwen4exp/reference.cpp` is a validation-only harness for the pinned
llama.cpp revision in `artifact.json`. Build that checkout with CMake, CPU
enabled, tests/examples/server disabled. In `src/llama-model.cpp`, change
`ml.init_mappings(true, ...)` to `ml.init_mappings(false, ...)` to prevent
whole-model prefetch on this host. Compile the harness against `include`,
`ggml/include`, and the resulting `build/bin` libraries (`lllama`, `lggml`,
`lggml-base`, and macOS Accelerate). For example, with `LLAMA_SRC` pointing to
the pinned checkout:

```sh
clang++ -O2 -std=c++17 pkg/inference/scripts/qwen4exp/reference.cpp \
  -I"$LLAMA_SRC/include" -I"$LLAMA_SRC/ggml/include" \
  -L"$LLAMA_SRC/build/bin" -Wl,-rpath,"$LLAMA_SRC/build/bin" \
  -lllama -lggml -lggml-base -framework Accelerate -o /tmp/qwen-reference
/tmp/qwen-reference "$QWEN_GGUF" /tmp/qwen-reference.f32 \
  9419 11 353 9707 16451 13 9419 11 353 248044 9419 11
python3 pkg/inference/scripts/qwen4exp/compare_logits.py \
  /tmp/qwen-native.f32 /tmp/qwen-reference.f32 \
  --tokens 9419,11,353,9707,16451,13,9419,11,353,248044,9419,11 \
  --output /tmp/qwen-parity.json
```

The reference uses the pinned architecture graph, but replaces quantized/BF16
matvec results with upstream dequantization plus FP32 BLAS in 64-row tiles.
Stock CPU ggml quantizes activations; Antfly Metal keeps FP32 activations.
Flash attention is disabled in the reference because its graph casts K/V to
FP16 even with FP32 cache storage; both compared paths use FP32 attention.
Therefore this is explicitly an adapted precision reference, not a claim of
bitwise parity with stock llama.cpp. The comparator requires equal top-1 IDs
and relative L2 error <=0.002 at every supplied position. Run native and
reference model processes sequentially on a 16 GiB host. The synthetic QSA
boundary test exercises selection through position 4095; it does not establish
full-model generation correctness across a 4096-token sequence.
