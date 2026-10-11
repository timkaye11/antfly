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
ANTFLY_ZIG_MAX_RSS=8589934592 python3 tools/run_bounded_zig_build.py build inference-qwen4exp -j1 -Dmetal=true -Doptimize=ReleaseFast
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

- Required expert reads overlap resident-expert computation by default.
  `ANTFLY_QWEN4EXP_IO_OVERLAP=0` disables overlap;
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
- `ANTFLY_QWEN4EXP_PREFILL_CHUNK=128`: default layer-major prefill (2..128 tokens).
  Set it to `0` for the token-wise control.
  Recurrent and sparse-attention updates stay in causal order. Each layer's expert
  union is acquired in groups of ten and evaluated in matrix batches of at most
  32 rows, preserving one load per expert per chunk. GPU gathers and scatters
  retain original token/route positions, so each group of ten experts shares
  one completion boundary. Q4_K, Q5_K and Q5_1 batches use the existing
  row-indexed SIMD reduction kernels through a Qwen-only dispatch option.
  Dense attention/HC projections remain token-wise. Chunk storage is at most
  22.53 MiB inside the existing 64 MiB backend scratch reservation. Prefix caching
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
activation; 33/34 time prefill gather/scatter encoding. With overlap, GPU durations,
paging waits and host encoding overlap;
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

### Experimental Qwen3.8-27B layer streaming (Metal)

Qwen3.8-27B is a dense `qwen35` model. Every ordinary decoder layer runs for
every token; there is no expert router. The opt-in CLI keeps the encoded output
head and FP32 recurrent/KV state resident, reads embedding rows on demand, and
alternates two 288 MiB layer buffers using uncached `pread`. Only the 64 trunk
layers execute; the artifact's extra MTP layer is excluded. A slot cannot be
reused until its GPU work completes. Read failures and cancellation invalidate
state and drain outstanding work.

The exact UD-Q4_K_M GGUF and tokenizer sidecars are SHA-256 pinned in
`../scripts/qwen35_streamed/artifact.json`. Startup verifies the entire GGUF in
bounded reads. `--dry-run` only validates the inventory and memory plan and
explicitly reports `hash_verified=false`. It does not initialize Metal or load
weights. Supported scope is local text, eager Metal on unified memory, one
sequence, greedy sampling, and at most 4096 total prompt/output tokens. HTTP,
vision, speculation, sampling, and MoE cache flags are rejected.

```sh
cd zig
ANTFLY_ZIG_MAX_RSS=8589934592 python3 tools/run_bounded_zig_build.py build \
  inference-qwen35-streamed inference-test-qwen35-streamed \
  -j1 -Dmetal=true -Doptimize=ReleaseFast

MODEL=/Volumes/Extreme/antfly/models/unsloth/Qwen3.8-27B-GGUF-UD-Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
./zig-out/bin/antfly-qwen35-streamed "$MODEL" 'Explain why the sky is blue.' \
  --experimental-qwen35-streamed --backend metal --speculation-policy off \
  --context-size 4096 --prefill-chunk-size 512 --combined-budget-mb 4096 \
  --max-tokens 64 --stream --print-timing --json-timing /tmp/qwen27b.json
```

Add `--dry-run` to inspect the memory plan. `--raw-prompt` bypasses the pinned
chat template; thinking defaults off. Model weights must remain on the SSD.
There is no requirement to create a second model copy or convert its mixed
quantization. The same opt-in flags are accepted by the general native generate
entry point.

At context 4096 and the default 512-token chunk, the conservative admission
plan is 3,478,392,832 bytes within a 4 GiB cap, retaining at least 512 MiB spare. This includes a 512 MiB bounded
vocabulary allocator, both encoded layer buffers, a 995 MiB encoded output head,
FP32 recurrent/conv and KV caches, activation buffers, the 34 MiB dequant tile,
a 2 MiB compact tile-result buffer, a page-rounded 2 KiB IQ3 codebook, and
runtime allowances. Peak physical footprint and live host pressure are checked
during execution. IQ3_S and IQ4_XS now use fused packed Metal matvec for small
batches. At least eight prompt rows use 512-output-row GPU-dequantized FP32
tiles and MPS GEMM with compact-result scatter, including Q3_K, Q8_0, Q4_K,
Q5_K, Q6_K, and IQ4_NL. Tile reuse follows GPU command order without a CPU
completion wait per tile. Layer-buffer reuse still requires completion.
Decode and small prompts retain packed GPU operations, including the dedicated
wide-row IQ4_NL kernel. The generic single-row IQ4_NL kernel under-dispatches wide matrices
in this checkout. Prefill defaults to 512 tokens per layer load and permits up to 1024; packed small-batch projections dispatch at most 32 rows. Decode reads
14,344,308,736 trunk-weight bytes per token. Receipt byte counts describe completed application read ranges, not
physical SSD traffic.

Qualification tools (run models serially, without concurrent builds):

```sh
# Independent FP32 equation oracle, using codecs from the pinned upstream ggml.
python3 pkg/inference/scripts/qwen35_streamed/reference.py "$MODEL" \
  --ggml-library /path/to/pinned/llama.cpp/build/bin/libggml-base.dylib \
  --ids 760,6511,314,9338,369 --steps 2 --output /tmp/qwen27b-reference \
  --capture /tmp/qwen27b-reference/layers

# Real GPU runs: chunk1/chunk512, IO overlap, chat, cancellation, budget rejection,
# and, with --long, a 4094-token prompt plus two generated tokens.
python3 pkg/inference/scripts/qwen35_streamed/qualify.py "$MODEL" \
  --binary zig-out/bin/antfly-qwen35-streamed --output /tmp/qwen27b-qualification \
  --reference /tmp/qwen27b-reference --chunk-parity --long
```

For the full 128-token numerical check, run the oracle with 128 comma-separated
copies of token `830` and `--steps 1`, using the same directory for `--output`
and `--capture`. Pass that directory to `qualify.py --wide-reference DIR`.
This compares all 64 layer outputs and final logits for the raw prompt
`" x"` repeated 128 times.

The oracle uses explicit range reads and 128-row dequant tiles, without mmap or
whole FP32 matrices. It independently implements the upstream equations; it is
not a full llama.cpp execution or an original BF16 model comparison. The fixture
provenance records the exact upstream revision, source tensors, and byte hashes.
`make_fixtures.py` regenerates the small real-weight codec fixtures with that
library. Tests exercise all nine artifact formats, batches through 1024 rows, and repeated tile reuse with a short final tile.

`ANTFLY_QWEN35_CAPTURE_DIR` writes layer outputs and logits at each
processed position for numerical comparison; create the directory first.
`ANTFLY_QWEN35_DISABLE_IO_OVERLAP=1` serializes layer reads for parity checks.
SIGINT requests cancellation. JSON timing receipts include token IDs, finish
reason, process peak footprint, per-token times, layer-read bytes, and CPU
matrix-weight-dequantization time/bytes (excluding input embeddings).
`ANTFLY_QWEN35_CPU_DEQUANT=1` restores the old matrix fallback for comparisons;
`ANTFLY_QWEN35_PROFILE=1` adds per-phase GPU execution, host completion wait,
completed batches, and exposed prefetch read waits. GPU execution overlaps
host waits; do not add the counters together. The current qualification and
reproducible harness are described in
[`qwen_local/README.md`](../scripts/qwen_local/README.md).
Argument errors before model execution are
reported on stderr. Failure receipts during model setup identify the stage.

The original 128-token-chunk [M4 qualification record](../scripts/qwen35_streamed/qualification-m4.json)
pins the tested dirty-tree source and executable hashes. On the 16 GiB Apple M4,
the 4094-token prompt plus two-token generation completed with a peak physical
footprint of 2,693,204,656 bytes (2.51 GiB) and zero system swapout growth.
Prefill took 1289.06 seconds (21.48 minutes); the subsequent decode step took
16.85 seconds (0.059 tokens/second). These are local measurements, not a
throughput guarantee. The separate 128-token prompt took 33.73 seconds to
prefill. Its 64 layer outputs had a maximum relative L2 error of 0.0000412
against the bounded CPU oracle; the maximum logit difference was 0.0001792.
The 4K run qualifies functional execution and memory, not full-context
reference parity. Chat/EOS, cancellation, insufficient-budget rejection,
context rejection, chunk-size parity, and synchronous/overlapped I/O checks
passed. The focused suite passed 17 tests; Flash Next regression tests passed
35 with four artifact-dependent skips. Formatting and license checks passed.

The follow-up [prefill optimization record](../scripts/qwen35_streamed/prefill-optimization-m4.json)
pins the earlier 512-token-chunk implementation and executable separately from
that original qualification. Bounded FP32 matrix tiles replace repeated packed
vector work during prefill; supported formats decode those tiles on the GPU.
The same M4 and exact SSD artifact produced these serial measurements:

| Raw prompt tokens | Original prefill | Optimized prefill | Speedup |
| --- | ---: | ---: | ---: |
| 128 | 33.73 s | 17.13 s | 1.97x |
| 512 | 146.97 s | 24.02 s | 6.12x |
| 4094 | 1289.06 s | 269.48 s | 4.78x |

The 4094-token run plus two generated tokens completed at 2,880,113,448 bytes
(2.68 GiB) peak footprint with zero system swapout growth. Prefill read
114,754,469,888 trunk-weight bytes rather than 459,017,879,552. The continued
decode step took 16.72 seconds. Load/hash verification added 20.92 seconds and
is excluded from prefill timing. These are single-trial synthetic prompts
(repeated token 830), with uncontrolled OS/device caches, not a general
throughput guarantee. The 128/512-token numerical runs capture layer outputs;
the full-context timing does not.

The updated focused suite passed 18 tests, including all packed artifact formats,
512-row batches, repeated staging-tile reuse, and partial final tiles. The
128-token independent reference comparison covers all 64 layers (maximum
relative L2 error 0.0000425) and logits (relative L2 error 0.00000926). The
512-token prefill and next decode match the original 128-chunk baseline with
logit relative L2 errors below 0.00000435. The full 4K run remains functional
and memory evidence, not a full-context independent numerical comparison.
`qualify.py --chunk-parity` adds a 512-token prompt at chunk sizes 128 and 512,
comparing logits before and after one continued decode step.

Final-build chat/EOS, cancellation during prefill, insufficient-budget rejection,
and rejection of chunks larger than 512 also passed, with zero system swapout
growth. Formatting, lint, license headers, and diff checks passed.

### Persistent local sessions and follow-up experiments

See [`../scripts/qwen_local/README.md`](../scripts/qwen_local/README.md) for
the streaming terminal client, JSONL session protocol, shared realistic prompt
benchmark, opt-in Flash projection batching/MPS, 1024-token 27B chunks, and
27B target-verified prompt lookup. These additions have separate qualification
receipts; earlier measurements above retain their original binary identities.
