# Antfly inference NVIDIA Inference Plan

## Goal

Support inference on NVIDIA GPUs while preserving Antfly inference's current portability
model:

- A normal Antfly inference build does not require the CUDA toolkit.
- A normal Antfly inference container does not ship CUDA runtime libraries, cuBLAS,
  cuDNN, TensorRT, ONNX Runtime, or XLA.
- The native CUDA path uses only the NVIDIA driver ABI at runtime:
  `libcuda.so.1`, loaded dynamically.
- XLA/PJRT remains an optional compiled-graph path for dense/static graph
  execution, not the primary GGUF quantized runtime.
- CPU/native fallback remains available for unsupported devices, tensor
  formats, and operators.

The production target is GGUF decoder inference on GKE-class NVIDIA nodes,
starting with L4 and T4. A100 and H100 should be supported by the same portable
kernel artifacts, then optimized when profiling justifies architecture-specific
paths.

## Research Basis

The current design is based on:

- NVIDIA CUDA Driver API documentation: the driver API lives in the driver
  `cuda` dynamic library and exposes `cu*` entry points; this matches a
  `dlopen("libcuda.so.1")` runtime contract.
- NVIDIA CUDA compatibility documentation: PTX embedded for a lower virtual
  compute capability can be JIT-compiled by the driver for later GPUs, but
  older PTX will not automatically exploit newer architecture features.
- OpenXLA PJRT documentation: PJRT is a uniform device API with device-specific
  plugin implementations. Antfly inference already exposes an `xla` backend choice that
  maps to PJRT when `enable_pjrt` is compiled.
- OpenXLA XLA:GPU documentation: XLA lowers StableHLO graphs through GPU
  compilation pipelines that can emit GPU kernels, including PTX-oriented code.
- OpenXLA StableHLO quantization documentation: StableHLO quantization is
  uniform per-tensor/per-axis quantization; GGUF's block-packed formats such as
  `Q4_0`, `Q8_0`, and K-quants are not naturally represented as one StableHLO
  quantized `dot_general`.
- XLA custom-call documentation: XLA FFI can call host functions that receive a
  CUDA stream and launch CUDA kernels, but that API is still experimental. It is
  useful as an escape hatch, not the core dependency-free CUDA backend.
- GGUF/ggml practice: common GGUF files carry mixed tensor formats including
  F16/BF16/F32 plus block quantized legacy, K-quant, I-quant, and newer formats.

Reference links:

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/driver-api.html
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- https://docs.nvidia.com/cuda/cuda-driver-api/
- https://openxla.org/xla/pjrt
- https://openxla.org/xla/gpu_architecture
- https://openxla.org/stablehlo/quantization
- https://openxla.org/stablehlo/spec
- https://openxla.org/xla/custom_call
- https://huggingface.co/docs/hub/en/gguf

## Strategic Decision

Use two lanes, with different jobs:

| Lane | Role | Dependencies | Best Fit | Not Best Fit |
|---|---|---|---|---|
| Native CUDA driver backend | Primary GGUF inference path | `libcuda.so.1` only at runtime | Quantized GGUF linears, resident weights, decoder fast paths | Full compiler optimization, arbitrary model graphs |
| XLA/PJRT | Optional compiled-graph path | PJRT plugin and XLA artifacts | Dense/static graph inference, safetensors/ONNX-style graph exports, correctness comparison | Direct block-packed GGUF quant matmul without custom calls |

Do not make XLA a prerequisite for CUDA. The native CUDA path should be useful
when the only NVIDIA component visible in the container is the driver mounted by
the node.

Do not require cuBLAS for the first production path. Optional cuBLASLt dispatch
is allowed for dense F16/BF16 matmul when the libraries are present, but the
driver-only kernels still need to preserve correctness and coverage for
dependency-light deployments and GGUF packed-weight formats.

## Existing Codebase Fit

Relevant local state:

- `pkg/inference/src/native_backend_choice.zig` already has `xla`, mapped to
  `BackendKind.pjrt` for compiled partitions.
- `pkg/inference/src/graph/compiled_pjrt.zig`,
  `pkg/inference/src/graph/pjrt_compiler.zig`, and
  `pkg/inference/src/graph/pjrt_executor.zig` already define the PJRT lane.
- `pkg/inference/src/graph/quant_matmul.zig` already provides the shared
  quantized matmul vocabulary: dispatch buckets, row buckets, packed format
  descriptors, and operator support.
- `pkg/inference/src/gguf/tensor_types.zig` and
  `pkg/inference/src/gguf/quant_codec.zig` are the canonical GGUF type and
  dequantization references.
- `pkg/inference/src/ops/native_compute.zig`,
  `pkg/inference/src/ops/metal_compute.zig`, and
  `pkg/inference/src/ops/wasm_compute.zig` already contain quantized matmul
  behavior and fallback patterns.
- `pkg/inference/src/graph/backend_contracts.zig` does not yet include
  `cuda` as a graph `BackendKind` or `cuda_buffer` storage class.
- `pkg/inference/src/backends/backends.zig` does not yet expose CUDA as a direct
  session backend.
- `build.zig` does not yet have `-Dcuda` or CUDA artifact options.

The CUDA work should integrate through these existing contracts instead of
creating another quant selector or model-specific backend path.

## CUDA Runtime Contract

CUDA support is optional and probe-based:

- On startup, try `dlopen("libcuda.so.1")`.
- Resolve only the CUDA Driver API symbols Antfly inference uses.
- Call `cuInit`, enumerate devices, and select one device.
- Prefer retaining the device primary context so Antfly inference composes with other
  driver users in the same process. Backend-owned contexts are acceptable for
  isolated smoke tests, but not the production default.
- Use one default stream per CUDA backend instance at first. Add extra streams
  only after there is measured overlap to exploit.
- If any probe step fails, mark CUDA unavailable and keep the existing fallback
  chain.
- Keep all driver handles behind Zig-owned `CudaDriver` and `CudaContext`
  tables.
- All device allocations, streams, modules, functions, and events are owned by
  the CUDA backend and released by backend teardown.

Initial symbol set:

- `cuInit`
- `cuDriverGetVersion`
- `cuDeviceGetCount`
- `cuDeviceGet`
- `cuDeviceGetName`
- `cuDeviceComputeCapability`
- `cuDevicePrimaryCtxRetain`
- `cuDevicePrimaryCtxRelease`
- `cuCtxSetCurrent`
- `cuStreamCreate`
- `cuStreamSynchronize`
- `cuStreamDestroy`
- `cuMemAlloc`
- `cuMemFree`
- `cuMemcpyHtoDAsync`
- `cuMemcpyDtoHAsync`
- `cuMemcpyDtoDAsync`
- `cuModuleLoadDataEx`
- `cuModuleUnload`
- `cuModuleGetFunction`
- `cuLaunchKernel`
- `cuGetErrorName`
- `cuGetErrorString`

Add events, graph launch, stream-ordered allocation, virtual memory, and
multi-GPU APIs only after the single-device inference path is correct.

## GPU Compatibility

Compatibility floor:

| GPU | Compute capability | Role |
|---|---:|---|
| T4 | `sm_75` | Cheapest compatibility floor |
| A100 | `sm_80` | Existing high-throughput accelerator |
| L4 | `sm_89` | Preferred GKE cost/performance target |
| H100 | `sm_90` | High-end validation target |

Checked-in CUDA artifacts are generated with the pinned CUDA `13.2` toolkit.
The portable artifact is PTX ISA `9.2` for `compute_75` / `.target sm_75`, so
the NVIDIA driver can JIT it on later devices. The default artifact is a fatbin
with cubins for the current validation targets and a `compute_75` PTX fallback:

- `sm_75` baseline cubin for T4 startup latency.
- `sm_80` cubin for A100.
- `sm_89` cubin for L4.
- `sm_90` cubin for H100.
- `sm_100`, `sm_110`, and `sm_120` cubins for Blackwell-generation targets.

Keep the `compute_75` PTX path even when fatbins are used by default.
Architecture-specific cubins must never be the only checked-in artifact.

## Kernel Artifact Policy

Normal builds must not invoke `nvcc`, `ptxas`, `clang --cuda`, or network
downloads.

Use this layout:

| Path | Purpose |
|---|---|
| `pkg/inference/src/ops/cuda/driver.zig` | Driver API dynamic loader |
| `pkg/inference/src/ops/cuda/context.zig` | Device/context/stream lifecycle |
| `pkg/inference/src/ops/cuda/buffer.zig` | Device memory and host copies |
| `pkg/inference/src/ops/cuda/kernels.zig` | Embedded PTX/fatbin module loading and diagnostics |
| `pkg/inference/src/ops/cuda/quant.zig` | GGUF format descriptors for CUDA |
| `pkg/inference/src/ops/cuda/cuda_compute.zig` | `ComputeBackend` implementation |
| `pkg/inference/src/ops/cuda/kernels/*.cu` | Developer kernel sources |
| `pkg/inference/src/ops/cuda/artifacts/*.ptx` | Checked-in portable PTX |
| `pkg/inference/src/ops/cuda/artifacts/*.fatbin` | Checked-in multi-arch fatbins |

Build flags:

- `-Dcuda=true`: compile CUDA backend Zig code and embed checked-in artifacts.
- `-Dcuda=false`: default until the backend is mature.
- `-Dcuda-artifacts=portable`: embed the checked-in portable PTX only.
- `-Dcuda-artifacts=fatbin`: embed the checked-in multi-arch fatbin. This is
  the default.
- `-Dcuda-libs=auto`: use optional CUDA library acceleration when available.
- `-Dcuda-libs=required`: require CUDA libraries such as cuBLASLt to load.
- `-Dcuda-libs=off`: do not load optional CUDA libraries.

Use `scripts/regen-cuda-artifacts.sh --check` to verify CUDA artifacts and
`scripts/regen-cuda-artifacts.sh --write` to update them. The script requires
CUDA `13.2`, verifies portable PTX ISA `9.2` and `.target sm_75`, verifies
fatbin cubins for the supported SM targets when `cuobjdump` is available, and
checks that required CUDA symbols are present before updating checked-in
artifacts. CUDA-enabled CI may verify checked-in artifact freshness, but normal
CI should not need CUDA.

## GLiNER2 CUDA Q4 Span Kernels

CUDA GLiNER2 span-head weights use resident `Q4_K` kernels by default when the
checked-in CUDA module exposes the required GLiNER span primitives. This avoids
upload-time dequantization for `span_rep.span_rep_layer.*.weight` tensors and
keeps the span head on packed weights.

Runtime overrides:

- `TERMITE_CUDA_DISABLE_GLINER_SPAN_Q4_KERNELS=1`: use the fp32-upload span
  path instead of the resident GLiNER span `Q4_K` kernels.
- `TERMITE_CUDA_ENABLE_GLINER_SPAN_Q4_KERNELS=0`: legacy opt-out alias for the
  same behavior.
- `TERMITE_CUDA_DEQUANTIZE_QUANT_WEIGHTS=1`: force upload-time dequantization
  for quantized weights.

## Gemma4 And TurboQuant KV Status

Gemma4 CUDA defaults remain `f32` KV for production correctness. The optional
TurboQuant cache formats (`polar4`, `turbo3`) are available through
`--cache-dtype`, and CUDA now has an opt-in fully paged device path: compressed
keys are scored directly on device, values are stored as int8-per-head rows, and
attention resolves logical tokens through the CUDA block table instead of a
contiguous span assumption.

Current CUDA behavior:

- Default Gemma4 CUDA runs through the existing f32 device KV read/write and GQA
  attention path.
- `--cache-dtype polar4` stores device K rows in packed 4-bit Polar4 format and
  scores Q against compressed K directly.
- `--cache-dtype turbo3` stores device K rows as packed 3-bit keys plus the
  deterministic residual sketch and scores Q against both pieces directly.
- Both TurboQuant dtypes store CUDA V rows with the same int8-per-head value
  codec used by host KV storage, then dequantize V inside the decode attention
  kernel.
- `ANTFLY_CUDA_DISABLE_TURBOQUANT_COMPRESSED_V=1` keeps compressed K but forces
  f32 V storage for A/B testing.
- Unsupported shapes, missing CUDA symbols, stale artifacts, or
  `ANTFLY_CUDA_DISABLE_TURBOQUANT_KV=1` fall back to the existing non-compressed
  path.

Gemma4 CUDA decode graph replay is controlled by
`ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY=off|auto|required`. The E4B QAT
production gate uses `required` with stable temp reuse, delayed capture, device
decode scalars, and persistent replay. The gate defaults are:
`ANTFLY_INFERENCE_CUDA_TEMP_STABLE_REUSE=1`,
`ANTFLY_INFERENCE_CUDA_CAPTURE_ALLOW_UNPINNED=1`,
`ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ=10000`,
`ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD=0`, and
`ANTFLY_INFERENCE_CUDA_CAPTURE_FORCE_KV_CAPACITY=1024`.
For f32 KV decode, the production Gemma4 path also uses the scalar-replay-aware
fast GQA kernel when the shape is `batch=1`, `q_seq_len=1`, no mask/bias, and a
32-aligned head dimension up to 512. Set `ANTFLY_CUDA_DISABLE_FAST_GQA_DECODE=1`
to force the legacy scalar replay kernel. JSON timing exposes
`launch_attention_gqa_decode_fast` and
`launch_attention_gqa_decode_fast_fallbacks`; the E4B QAT gate requires fast
hits to be non-zero and fast fallbacks to remain zero.

Tiny decode scalar uploads and downloads use pinned host staging by default.
Set `ANTFLY_INFERENCE_CUDA_DISABLE_PINNED_SCALAR_UPLOADS=1` or
`ANTFLY_INFERENCE_CUDA_DISABLE_PINNED_SCALAR_DOWNLOADS=1` to restore the older
host transfer paths for A/B testing.

Greedy CUDA generation also carries the previous token as a backend tensor when
the native pipeline can prove pure greedy decode, no grammar, CUDA backend, and
resident decode KV. This now supports Gemma4 PLE/QAT token embeddings. Set
`ANTFLY_INFERENCE_CUDA_GREEDY_DEVICE_TOKEN_HANDOFF=0` to disable that
token-tensor handoff for A/B testing. JSON timing exposes
`generation_decoder_runtime.device_token_handoff_attempts`, `*_hits`,
`*_fallbacks`, and `*_seeds`; the E4B QAT 512-token validation requires
handoff hits to match post-seed decode steps. The same gate also requires the
raw i32 token export path for greedy token reads, reported as
`cuda_generate.to_float32_calls=0` and `cuda_generate.to_float32_bytes=0`, so
single-token extraction does not fall back through the float-conversion
download path.

Pure-greedy non-streaming CUDA decode can delay host-visible token readback by
one decode step with `ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK=1`.
The path keeps the argmax token as a backend tensor, enqueues an async pinned
4-byte i32 download, speculatively reserves the next KV slot, launches the
following decode, and then resolves the prior token id from the completed event.
EOS rollback truncates the speculative slot and synchronizes the discarded
lookahead tensor before stop. The E4B QAT production gate enables this path and
requires `cuda.download_syncs<=4` for each 512-token target run.

The deeper Gemma4 gated-runtime token-tensor decode experiment is behind
`ANTFLY_INFERENCE_CUDA_GATED_TOKEN_TENSOR_DECODE=1` and is off by default. It
tries to keep the token id tensor inside the Gemma4 PLE/gated runtime instead of
only handing the token tensor to the shared GPT decode path. On the current L4
diagnostic this regressed throughput by losing replay/handoff efficiency
(`/tmp/gemma4-e4b-qat-gated-tensor-16.json` reported
`decode_tok_per_s=11.503`, `graph_capture_persistent_replays=0`, and zero
device-token handoff hits), so it is retained only as an opt-in implementation
probe. The default path was revalidated after gating this off:
`/tmp/gemma4-e4b-qat-gated-default-16.json` reported
`decode_tok_per_s=18.265`, `device_token_handoff_hits=15`,
`device_token_handoff_fallbacks=0`, and
`graph_capture_persistent_replays=4`.

The earlier 2026-06-25 512-token E4B QAT default-path L4 validation
(`/tmp/gemma4-e4b-qat-gated-default-512.json`) reported
`decode_tok_per_s=15.019`, `device_token_handoff_hits=511`,
`device_token_handoff_fallbacks=0`, `device_token_handoff_seeds=1`,
`graph_capture_persistent_replays=500`, `launches_per_token=25.867`, and zero
fast-GQA or Q4_0 fused-kernel fallbacks. This cleared the then-current numeric
gate in that run, but previous repeats landed just under 15 tok/s, so it
motivated the later Q4_0 kernel and token-export work below.

Q4_0 tile8 decode kernels are compiled and smoke-tested. The global
`ANTFLY_INFERENCE_CUDA_Q4_0_DECODE_TILE8=1` switch still enables all tile8
variants for A/B testing, but the E4B QAT L4 production gate now defaults the
tile8 family off. The isolated 128-token matrix showed QKV tile8 and gate/up-pair
tile8 slower than tile4, and later 512-token compressed-KV gates were faster
with Q4_0 gated-down precompute on the tile4 path. Per-kernel controls are
available as
`ANTFLY_INFERENCE_CUDA_Q4_0_QKV_TILE8=1`,
`ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_TILE8=1`,
`ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8=1`, and
`ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_TILE8=1`.
`ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE16=1` is compiled only as a wider
tile experiment. The 128-token L4 run
`/tmp/gemma4-e4b-qat-tile-split-down16-128.json` reported
`decode_tok_per_s=16.028`, slower than the tile4 baseline and gated-down tile8,
so it is not part of the production gate.

The Q4_0 decode kernels also precompute the packed-nibble offset and high-nibble
flag once per input lane before the inner column loop. This keeps the original
per-lane scale load behavior but removes repeated lane branch/address work in
the hot Q4_0 tile, embedding, QKV, pair, gated-down, and argmax kernels. The
first warp-broadcast scale experiment regressed and was removed
(`/tmp/gemma4-e4b-qat-warp-scale-128.json` reported
`decode_tok_per_s=14.735`). The branch-hoisted candidate fed the current
production path: `/tmp/gemma4-e4b-qat-q4nibble-128.json` reported
`decode_tok_per_s=19.997`; after applying the same helper to Q4_0 embedding
lookup, `/tmp/gemma4-e4b-qat-q4nibble-embed-128.json` reported
`decode_tok_per_s=19.913`. The 512-token repeats
`/tmp/gemma4-e4b-qat-q4nibble-512.json` and
`/tmp/gemma4-e4b-qat-q4nibble-512-r2.json` were historical tile8 A/B runs and
reported `17.376` and `17.227` tok/s with
`graph_capture_persistent_replays=500`, device-token handoff hits for all
post-seed decode steps, and zero Q4_0 fused-kernel fallbacks. Later 512-token
production gates defaulted tile8 off after the tile4 gated-down precompute path
proved faster.

CUDA greedy token extraction now exports the 1x i32 token tensor directly
instead of converting it through `toFloat32`. The 128-token check
`/tmp/gemma4-e4b-qat-i32-export-128.json` reported
`decode_tok_per_s=19.953` with `cuda_generate.to_float32_calls=0` and
`cuda_generate.to_float32_bytes=0`. PLE combine now uses the CUDA
`addWeightedScalars` fusion, replacing the previous add-multiply-plus-scale
sequence with one weighted combine launch, and the hot device-token PLE path
fuses the Q6_K per-layer token embedding lookup into that weighted combine. The
current target-vs-Q4_K production gate at
`/tmp/gemma4-q4-target-vs-q4k-new-defaults-gate-20260625-r1` ran one 512-token E4B Q4_K
baseline and one 512-token E4B QAT Q4_0 target pass with delayed token readback,
PLE fusion, scaled embedding lookup, and fused PLE embedding-combine enabled.
The Q4_K baseline reported `decode_tok_per_s=16.676`; QAT reported
`30.125` tok/s, a `1.806x` QAT-over-Q4_K ratio against the gate's default
`MIN_E4B_QAT_OVER_Q4K_RATIO=1.25` floor. The QAT run kept
`graph_capture_persistent_replays=500`, `launches_per_token=21.787`,
`launch_embedding=513`, `launch_scalar=0`, `add_mul_scalar_fused=512`,
`linear_activation_slice_fused_q4_0=504`, `qkv_fused_q4_0=288`,
`linear_pair_fused_q4_0=504`, `gated_down_fused_q4_0=504`,
`gated_down_fused_q4_0_precompute=462`, `gated_down_fused_q4_0_tile8=0`,
`device_token_handoff_hits=511`, Q4_0 fused fallbacks at zero,
`graph_capture_capacity_skips=0`, and `download_syncs=2`.
The default E4B QAT production gate is now `E4B_QAT_REPEATS=2`,
`MIN_E4B_QAT_TOK_S=24.0`, `MIN_E4B_QAT_RUN_TOK_S=24.0`,
`E4B_QAT_PENDING_TOKEN_READBACK=1`, `E4B_QAT_MAX_DOWNLOAD_SYNCS=4`,
`E4B_QAT_REQUIRE_PLE_FUSION=1`, `E4B_QAT_Q4_0_GATED_DOWN_TILE8=0`,
`E4B_QAT_REQUIRE_GATED_DOWN_TILE8=0`,
`E4B_QAT_Q4_0_PLE_GATE_FUSION=1`, `E4B_QAT_PLE_RMS_EMBED_FUSION=0`,
`E4B_QAT_MAX_LAUNCHES_PER_TOKEN=22.5`, and, when an `E4B_Q4K_BASELINE` is
present, `MIN_E4B_QAT_OVER_Q4K_RATIO=1.25` on the L4 gate.

The long-context replay gate is opt-in so the normal 512-token production gate
stays fast:

```bash
RUN_BUILD=0 RUN_SMOKE=0 RUN_MICROBENCH=0 RUN_DEFAULT_POLICY=0 \
RUN_TARGET_ONLY=0 RUN_E4B_Q4K_BASELINE=off RUN_E4B_QAT=off \
RUN_E4B_QAT_LONG=required E4B_QAT_LONG_MIN_TOKENS=900 \
RUN_12B_MTP=0 RUN_E2B_MTP=0 RUN_TIMEOUT=720s \
scripts/gemma4_cuda_production_gate.sh
```

The top-level gate also has an opt-in QAT resident soak check for serving
stability under concurrent HTTP load. It reuses the preloaded QAT server after
the warm pass, sends concurrent requests, gates aggregate throughput, and
requires per-request throughput, p95 latency, and graph replay coverage across
the warm plus soak tokens:

```bash
RUN_BUILD=0 RUN_SMOKE=0 RUN_MICROBENCH=0 RUN_DEFAULT_POLICY=0 \
RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=required \
RUN_E4B_QAT_RESIDENT_SOAK=required RUN_E4B_Q4K_RESIDENT_BASELINE=off \
RUN_12B_MTP=0 RUN_E2B_MTP=0 \
scripts/gemma4_cuda_production_gate.sh
```

For overload behavior, the top-level gate can also start the resident server
with bounded weighted request capacity and require excess concurrent generation
requests to fail quickly with HTTP 503 instead of sitting in a long queue:

```bash
RUN_BUILD=0 RUN_SMOKE=0 RUN_MICROBENCH=0 RUN_DEFAULT_POLICY=0 \
RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=required \
RUN_E4B_QAT_RESIDENT_BACKPRESSURE=required \
E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS=6 \
RUN_E4B_Q4K_RESIDENT_BASELINE=off RUN_12B_MTP=0 RUN_E2B_MTP=0 \
scripts/gemma4_cuda_production_gate.sh
```

The gate passes this through to `antfly-inference run` as
`--max-concurrent-requests 6`; operators can use the same flag directly for
per-replica admission control. The top-level and package-local gates both fail
before starting a resident server if soak/backpressure is enabled without
`RUN_E4B_QAT_RESIDENT`, or if backpressure is enabled without
`E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS`, so CI catches admission-control
misconfiguration without spending a model warmup.
The resident server also exports admission-control telemetry from `/metrics`:
`antfly_inference_request_queue_capacity`,
`antfly_inference_request_queue_depth`,
`antfly_inference_request_queue_available`,
`antfly_inference_request_queue_active_requests`,
`antfly_inference_request_queue_rejections_total`, and
`antfly_inference_request_queue_rejected_units_total`. The metrics-backed
top-level run at
`/tmp/gemma4-cuda-top-resident-backpressure-metrics-20260625-r1` reported two
accepted and two rejected requests, `queue_rejections=2`,
`queue_rejected_units=6`, queue depth `0`, queue active requests `0`, and queue
available `6` after the accepted requests completed.

The current integrated run at
`/tmp/gemma4-cuda-production-gate-20260625-integrated` requested 1024 tokens,
forced graph replay KV capacity to 2048, required at least 900 generated tokens,
and generated 936 tokens before EOS. It reported `decode_tok_per_s=15.844`,
`graph_capture_persistent_replays=926`, `graph_capture_capacity_skips=0`,
`launches_per_token=14.238`, `launch_embedding=939` (the long gate allows
`tokens+3` for EOS/lookahead cleanup), `launch_scalar=0`,
`add_mul_scalar_fused=938`, `device_token_handoff_hits=937`,
`device_token_handoff_fallbacks=0`, `download_syncs=1`, and zero Q4_0/GQA
fallback counters.

The preloaded resident serving gate is opt-in in the package-local production
gate. It starts `antfly-inference run` with `--preload-model
generator:cuda:<E4B_QAT>`, keeps the model resident, then measures warm HTTP
generation requests through `/ai/v1/generate`:

```bash
RUN_MTP=off RUN_RESIDENT=off RUN_E4B_QAT=off RUN_E4B_QAT_LONG=off \
RUN_E4B_QAT_RESIDENT=required RUN_E4B_Q4K_BASELINE=off \
RUN_E4B_Q4K_RESIDENT_BASELINE=required \
E4B_Q4K_BASELINE_MODEL=/home/timkaye/tim/antfly/.models/google/gemma-4-E4B-it-q4_k \
E4B_QAT_RESIDENT_TOKENS=512 E4B_QAT_RESIDENT_WARM_REPEATS=2 \
E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY=required \
E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY=required \
MIN_E4B_QAT_RESIDENT_WARM_TOK_S=15.0 \
MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO=1.05 \
zig/pkg/inference/scripts/gemma4_cuda_production_gate.sh --mtp-only
```

The top-level production gate exposes the same opt-in serving check:

```bash
RUN_BUILD=0 RUN_SMOKE=0 RUN_MICROBENCH=0 RUN_DEFAULT_POLICY=0 \
RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=required \
RUN_E4B_Q4K_RESIDENT_BASELINE=required RUN_12B_MTP=0 RUN_E2B_MTP=0 \
E4B_Q4K_BASELINE=/home/timkaye/tim/antfly/.models/google/gemma-4-E4B-it-q4_k \
E4B_QAT_RESIDENT_TOKENS=512 E4B_QAT_RESIDENT_WARM_REPEATS=2 \
E4B_QAT_RESIDENT_DECODE_GRAPH_REPLAY=required \
E4B_Q4K_RESIDENT_DECODE_GRAPH_REPLAY=required \
MIN_E4B_QAT_RESIDENT_WARM_TOK_S=15.0 \
MIN_E4B_QAT_RESIDENT_OVER_Q4K_RATIO=1.05 \
scripts/gemma4_cuda_production_gate.sh
```

The current forced-replay resident run at
`/tmp/gemma4-cuda-qat-resident-replay-20260625-r3` preloaded and warmed the E4B
QAT CUDA model, then completed two 128-token warm HTTP requests at `17.013` and
`17.043` tok/s E2E (`avg=17.028`). The gate also asserted serving graph replay
from the server log and reported `replays=246` with a `floor=42`. The serving
path now mirrors CLI generation by attaching `KvStorageRuntime` to the decode
state so CUDA can use device KV; CUDA graph slots are invalidated when a new
per-request KV device hook is provisioned so a graph captured for one request
cannot replay against freed KV buffers in the next request.
The package-local QAT-vs-Q4_K serving ratio gate at
`/tmp/gemma4-cuda-pkg-resident-q4k-ratio-20260625-r1` reported QAT resident
warm requests at `17.049` and `17.030` tok/s E2E (`avg=17.040`) and Q4_K
resident warm requests at `12.861` and `12.831` tok/s E2E (`avg=12.846`), for a
serving ratio of `1.326x` against the `1.05x` floor.
The package-local 512-token serving gate at
`/tmp/gemma4-cuda-pkg-resident-q4k-ratio-512-20260625-r1` reported QAT
resident warm requests at `16.211` and `16.061` tok/s E2E (`avg=16.136`) and
Q4_K resident warm requests at `12.711` and `12.827` tok/s E2E (`avg=12.769`),
for a serving ratio of `1.264x` against the `1.05x` floor with
`graph_replays=1014` for both resident paths. The resident gate now defaults to
512 generated tokens, requires CUDA graph replay, and uses a 15.0 tok/s warm
floor; set `E4B_QAT_RESIDENT_TOKENS=128` only for shorter smoke checks.
The top-level resident replay-only result at
`/tmp/gemma4-cuda-top-resident-replay-20260625-r2` reported `17.111` and
`17.134` tok/s E2E (`avg=17.123`) with `graph_replays=246`. The serving
QAT-vs-Q4_K ratio gate at
`/tmp/gemma4-cuda-top-resident-q4k-ratio-20260625-r3` reported QAT resident
warm requests at `17.000` and `17.013` tok/s E2E (`avg=17.007`) and Q4_K
resident warm requests at `12.796` and `12.785` tok/s E2E (`avg=12.790`), for a
serving ratio of `1.330x` against the `1.05x` floor. The top-level
`readiness.json` includes the resident, Q4_K baseline, and ratio step outcomes
for downstream automation. New runs also emit `cuda_environment.json` and copy
that object into `readiness.json` under `environment.cuda_smoke`; compare
throughput artifacts together with `device_name`, `compute_capability`,
`driver_version`, `artifacts`, and the CUDA smoke/capability map before
promoting thresholds across GPUs. The package-local gate emits the same
`cuda_environment.json` and `PASS cuda_environment` summary line by default
(`RUN_CUDA_ENV=off` disables it; `RUN_CUDA_ENV=required` makes missing CUDA
environment metadata fatal). Both gates also emit
`e4b_qat_production_summary.json`, which gathers target QAT/Q4_K ratios,
long-context QAT throughput, resident warm QAT/Q4_K ratios, soak latency and
aggregate throughput, backpressure/queue metrics, graph replay counts, and the
CUDA environment into a single artifact for CI dashboards and cross-GPU
threshold review. The summary includes a `verdict` section; when a gate phase
is enabled, the gate passes the same thresholds into the summary so missing
evidence or sub-threshold QAT/Q4_K ratios are reported as structured
`verdict.failures`, not just as free-form log output.
For industry-style throughput targets, keep base decode and speculative decode
separate. Public `40+ tok/s` Gemma 4 claims are usually measured on different
hardware/runtime stacks, and the `120 tok/s` 12B QAT result used MTP/speculative
decoding rather than one-token-at-a-time decode. The top-level gate can now run
an E4B QAT MTP matrix with the official assistant path:

```sh
RUN_E4B_QAT_MTP=required \
E4B_QAT_ASSISTANT_Q8=.models/google/gemma-4-E4B-it-assistant \
E4B_QAT_MTP_TOKENS=512 \
E4B_QAT_MTP_SPEC_KS="2 4 6" \
E4B_QAT_MTP_PROMPT_FILTER="ants_chat code_chat" \
RUN_12B_MTP=0 RUN_E2B_MTP=0 \
scripts/gemma4_cuda_production_gate.sh
```

The matrix writes `mtp_e4b_qat/summary.tsv`; `e4b_qat_production_summary.json`
also includes it under `mtp.mtp_e4b_qat` with `best` and `best_active` rows.
Use that artifact to track acceptance, active/disabled policy decisions,
decode tok/s, and speedup versus target-only rows before treating MTP as a
provider-competitive path.
The top-level gate also has an opt-in compressed-KV QAT check:

```sh
RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=off \
RUN_E4B_QAT_COMPRESSED_KV=required RUN_12B_MTP=0 RUN_E2B_MTP=0 \
RUN_E4B_QAT_MTP=off \
scripts/gemma4_cuda_production_gate.sh
```

This forces the requested `--cache-dtype` path with
`E4B_QAT_COMPRESSED_KV_DTYPE=polar4` and
`E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS=0`, then validates 512 generated
tokens, decode tok/s, persistent graph replays, download syncs, page-boundary
capacity skips, and compressed-V read/write counters. On 2026-06-25 the
artifact `/tmp/gemma4-q4-compressed-kv-new-defaults-gate-20260625-r1` reported
`38.823 tok/s`, `500` persistent replays, `501` graph replays, `2` download
syncs, `0` capacity skips, `462` fast compressed-GQA launches, `504`
compressed-V reads, `288` compressed-V writes, `24` paged block-table uploads,
`504` fused PLE gate projections, `462` Q4_0 gated-down precompute hits, zero
Q4_0 gated-down tile8 hits, and `22.303` launches/token. The attention-read
path now elides the block-table lookup when the paged KV allocation is
identity-contiguous while retaining the normal block-table path for
non-contiguous pages. The final validation artifact
`/tmp/gemma4-q4-compressed-kv-identity-attn-only-gate-20260625-r1` reported
`38.956 tok/s`, `504` identity attention reads, `24` paged block-table uploads,
`500` persistent graph replays, `2` download syncs, and `0` capacity skips;
earlier attention-only repeats landed at `39.354` and `39.219` tok/s. This
closes the page-boundary graph-break blocker, enables the paged/block-table
Polar4 compressed attention fast path, defaults the row-1 Q4_0 gate/up pair
kernel to the 4-warp tile4 variant, defaults Q4_0 gated-down to precompute
`activation(gate) * up` once before the down projection, and leaves the fused
PLE RMS/embed construction opt-in via
`ANTFLY_INFERENCE_CUDA_PLE_RMS_EMBED_FUSION=1` because it lowers launch count
but has not beaten the default path reliably in gate runs. The combined gate/up
activation kernel also remains opt-in via
`ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_PRECOMPUTE=1` because it
regressed to `37.185 tok/s` on the same gate. Write-side block-table elision was
tested and backed out after regressing to `38.850 tok/s`. Treat the repeatable
near-39.5 tok/s result as the focused production target; the full-release
blocking floor is set to `36.0 tok/s` to avoid flaking on mixed-gate
run-to-run variance, with
`40 tok/s` kept as a stretch/provider-reference target rather than a hard
blocker. The industry-grade focused gate at
`/tmp/gemma4-e4b-qat-industry-compressed-gate-20260625-r1` passed with
`39.102 tok/s`, `500` persistent graph replays, `2` download syncs, `0`
capacity skips, `504` compressed-V reads, `288` compressed-V writes, `24` paged
block-table uploads, `504` identity attention reads, `462` fast GQA launches,
and `0` compressed-KV write fallbacks.
For a fixed industry-style floor without a provider baseline, enable the
competitive-floor verdict explicitly:

```sh
RUN_E4B_QAT_COMPETITIVE_FLOOR=required \
RUN_E4B_QAT_COMPRESSED_KV=required \
E4B_QAT_COMPETITIVE_FLOORS="compressed_kv_decode_tok_s=36.0" \
scripts/gemma4_cuda_production_gate.sh
```

When `RUN_E4B_QAT_COMPRESSED_KV=required`, the gate and
`e4b_qat_production_summary.json` now require more than speed: at least the
configured token count, persistent graph replays, bounded download syncs, zero
capacity skips, compressed V reads/writes, paged block-table uploads, identity
attention fast-path hits for the fresh contiguous benchmark, fast GQA launches,
and zero compressed-KV write fallbacks. This keeps a provider-readiness artifact
from passing if the benchmark silently falls back to f32 KV, non-paged attention,
or host-heavy decode.

Supported floor metrics are the same local metrics used for provider
comparison: `compressed_kv_decode_tok_s`, `target_decode_tok_s`,
`long_decode_tok_s`, `resident_e2e_tok_s`, `soak_aggregate_tok_s`, and
`backpressure_accepted_e2e_tok_s`. The fixed local floor should target
`compressed_kv_decode_tok_s` because that is the paged Polar4 KV production
speed path; the f32 target metric remains useful for QAT/Q4_K apples-to-apples
kernel comparisons. This gate is intentionally separate from the QAT/Q4_K ratio
check, so CI can report "faster than Q4_K", "near-40 compressed KV floor", and
"provider baseline ratio" as distinct readiness claims.
Provider comparisons are optional but enforceable: set
`RUN_E4B_QAT_PROVIDER_COMPARISON=required`,
`E4B_QAT_PROVIDER_BASELINE_JSON=/path/to/provider-baselines.json` or
`E4B_QAT_PROVIDER_BASELINE_INLINE='{"baselines":[...]}'`, and
`MIN_E4B_QAT_PROVIDER_RATIO=1.0` to require local QAT throughput to meet or
beat external/provider baselines. Each baseline entry can name a `provider`, a
`metric` (`compressed_kv_decode_tok_s`, `target_decode_tok_s`,
`long_decode_tok_s`, `resident_e2e_tok_s`, `soak_aggregate_tok_s`, or
`backpressure_accepted_e2e_tok_s`), and a `tok_s` value; per-entry `min_ratio`
overrides the global ratio. These comparisons are recorded under
`provider_comparison` and fail the summary verdict on missing metrics, load
errors, or local/provider ratios below the floor.
`E4B_QAT_REQUIRE_PROVIDER_METADATA=1` is the default for provider comparisons;
release baselines must also provide `model`, `hardware`, `tokens`, `workload`,
`measured_at`, and `source_url` so the provider claim is auditable and matched
to the same workload shape.
Use `zig/pkg/inference/scripts/gemma4_qat_provider_baselines.example.json` as
the template for real external measurements. Before making provider comparison
required in CI, validate the baseline file independently:

```sh
python3 zig/pkg/inference/scripts/gemma4_qat_production_summary.py \
  --validate-provider-baselines-only \
  --provider-baseline /path/to/provider-baselines.json \
  --output /tmp/gemma4-provider-baseline-validation.json
```

This standalone mode checks that the file loads, uses supported metrics, and
contains the required provenance fields. The full production gate still performs
the local/provider throughput ratio check using the metrics from that gate run.
For OpenAI-compatible providers, collect a matching non-streaming 512-token
baseline with the provider benchmark helper instead of hand-authoring the JSON:

```sh
PROVIDER_API_KEY=... \
python3 zig/pkg/inference/scripts/gemma4_qat_provider_benchmark.py \
  --base-url https://provider.example/v1 \
  --api-key-env PROVIDER_API_KEY \
  --provider provider-name \
  --model google/gemma-4-E4B-it-qat-q4_0-gguf \
  --hardware "provider GPU or instance class" \
  --source-url https://provider.example/run-or-dashboard \
  --tokens 512 \
  --min-completion-tokens 512 \
  --repeats 2 \
  --warmup 1 \
  --baseline-stats avg,median,min \
  --output /tmp/gemma4-provider-baselines.json \
  --rows-tsv /tmp/gemma4-provider-baselines.tsv
```

The helper writes `baselines[]` in the same schema consumed by
`E4B_QAT_PROVIDER_BASELINE_JSON` and records per-request rows for audit. For
production comparisons, keep `avg,median,min` so the summary emits separate,
labeled provider comparison checks from one measured provider run; use
`--baseline-stat avg` only for narrow experiments.
For providers that publish or optimize for streaming throughput, collect a
streamed decode baseline and compare it to Antfly's local decode-rate metric:

```sh
PROVIDER_API_KEY=... \
python3 zig/pkg/inference/scripts/gemma4_qat_provider_benchmark.py \
  --base-url https://provider.example/v1 \
  --api-key-env PROVIDER_API_KEY \
  --provider provider-name \
  --model google/gemma-4-E4B-it-qat-q4_0-gguf \
  --hardware "provider GPU or instance class" \
  --source-url https://provider.example/run-or-dashboard \
  --metric target_decode_tok_s \
  --stream \
  --rate-source stream_decode \
  --tokens 512 \
  --min-completion-tokens 512 \
  --baseline-stats avg,median,min \
  --output /tmp/gemma4-provider-stream-baselines.json
```

Streaming mode requests OpenAI-compatible `stream_options.include_usage=true`
by default so completion-token counts remain provider-reported; use
`--no-stream-include-usage --allow-token-fallback` only for providers that lack
usage-bearing stream chunks and only when the prompt is known to complete at the
requested token limit.
The production gates can also collect that provider baseline inline before
writing `e4b_qat_production_summary.json`:

```sh
PROVIDER_API_KEY=... \
RUN_E4B_QAT_PROVIDER_BENCHMARK=required \
RUN_E4B_QAT_PROVIDER_COMPARISON=required \
E4B_QAT_PROVIDER_BASE_URL=https://provider.example/v1 \
E4B_QAT_PROVIDER_API_KEY_ENV=PROVIDER_API_KEY \
E4B_QAT_PROVIDER_NAME=provider-name \
E4B_QAT_PROVIDER_HARDWARE="provider GPU or instance class" \
E4B_QAT_PROVIDER_SOURCE_URL=https://provider.example/run-or-dashboard \
E4B_QAT_PROVIDER_BASELINE_STATS=avg,median,min \
scripts/gemma4_cuda_production_gate.sh
```

For streamed provider comparisons, add
`E4B_QAT_PROVIDER_STREAM=1`,
`E4B_QAT_PROVIDER_RATE_SOURCE=stream_decode`, and usually
`E4B_QAT_PROVIDER_METRIC=target_decode_tok_s`.

This writes `e4b_qat_provider_baselines.json`,
`e4b_qat_provider_baselines.tsv`, `e4b_qat_provider_benchmark.log`, and
`e4b_qat_provider_baseline_validation.json` in the gate output directory, then
feeds the generated JSON into the provider comparison verdict. The top-level
gate also copies these paths plus the parsed validation result into
`readiness.json` under `provider_benchmark`. `RUN_E4B_QAT_PROVIDER_BENCHMARK=auto`
skips cleanly when `E4B_QAT_PROVIDER_BASE_URL` is unset.
For merge/release readiness, run the CUDA release wrapper:

```sh
scripts/ci/gemma4-qat-cuda-release-gate.sh
```

The wrapper requires long-context decode, Polar4 compressed-KV decode, resident
warm, resident soak, resident backpressure, resident Q4_K comparison, and the
E4B QAT MTP safety matrix. It also re-checks
`readiness.json` and `e4b_qat_production_summary.json` for the exact
compressed-KV fast-path counters used by the release gate. The matching GitHub
Actions workflow is `Gemma4 QAT CUDA Release Gate`; it is manual because it
requires a GPU runner, local model artifacts, and optional provider credentials.
Use `provider_mode=baseline` with `provider_baseline_json` to validate a
pre-collected provider artifact, or `provider_mode=benchmark` with provider
inputs plus `PROVIDER_API_KEY` to collect a fresh OpenAI-compatible provider
baseline. Do not make an external provider claim from the local gate alone.
The local release-wrapper validation at
`/tmp/gemma4-e4b-qat-release-wrapper-gate-20260625-r2` passed `readiness=ok`
and the wrapper post-check with compressed-KV `36.726` tok/s, `500` persistent
graph replays, `504` compressed-V reads, `288` compressed-V writes, `24` paged
block-table uploads, `504` identity attention reads, `462` fast GQA launches,
and zero compressed-KV write fallbacks. The same run reported long-context QAT
`26.181` tok/s over 936 generated tokens, resident QAT warm `27.093` tok/s
E2E, resident Q4_K warm `17.818` tok/s E2E, resident ratio `1.521x`, soak
aggregate `29.613` tok/s, soak p95 `17390.3` ms, backpressure `2` accepted and
`2` rejected requests with max reject latency `6.7` ms, no unsafe resident graph
markers, and zero active MTP candidates.
The current required-phase top-level run at
`/tmp/gemma4-current-local-required-20260625-r3` required CUDA smoke, target
QAT/Q4_K, long-context QAT, resident QAT, resident Q4_K, soak, and
backpressure phases after merging latest `main`. It passed `readiness=ok` with
`verdict=ok`, zero verdict failures, L4 `sm_89` metadata, target QAT
`avg=16.982` tok/s, target Q4_K `avg=11.596` tok/s, target ratio `1.464x`,
long-context QAT `15.847` tok/s over 936 generated tokens, resident QAT
`avg=16.108` tok/s E2E, resident Q4_K `avg=12.758` tok/s E2E, resident ratio
`1.263x`, soak aggregate `16.896` tok/s, soak p95 `15476.5` ms, and
backpressure with one accepted and three HTTP 503 rejected requests in at most
`8.1` ms.
The top-level 512-token serving gate at
`/tmp/gemma4-cuda-top-resident-q4k-ratio-512-20260625-r1` reported QAT
resident warm requests at `16.563` and `16.384` tok/s E2E (`avg=16.474`) and
Q4_K resident warm requests at `12.745` and `12.815` tok/s E2E (`avg=12.780`),
for a serving ratio of `1.289x` against the `1.05x` floor with
`graph_replays=1014` for both resident paths.
After tightening the defaults, the top-level default-settings run at
`/tmp/gemma4-cuda-top-resident-defaults-512-20260625-r1` omitted the explicit
token, replay, and warm-floor overrides and still ran 512-token resident
requests. It reported QAT `avg=16.215` tok/s E2E, Q4_K `avg=12.768` tok/s E2E,
`ratio=1.270x`, and `graph_replays=1014`. The package-local default-settings
run at `/tmp/gemma4-cuda-pkg-resident-defaults-512-20260625-r1` likewise used
512-token resident requests by default and reported QAT `avg=16.179` tok/s E2E,
Q4_K `avg=12.762` tok/s E2E, `ratio=1.268x`, and `graph_replays=1014`.
Package-local serving coverage now also includes the same post-warm soak and
CLI-backed overload probes as the top-level gate. The combined package-local
run at `/tmp/gemma4-cuda-pkg-resident-soak-backpressure-20260625-r1` used
`--max-concurrent-requests 6`, reported QAT warm `avg=16.505` tok/s E2E,
completed the six-request concurrency-two soak at aggregate `16.966` tok/s with
minimum request `8.444` tok/s E2E and `p95=30317.3` ms, then accepted two and
rejected two overload requests with HTTP 503 in at most `3.5` ms. The run kept
graph replay coverage at `2544` against a `2496` soak floor and `3054` against
a `1504` backpressure floor. The metrics-backed package-local backpressure run
at `/tmp/gemma4-cuda-pkg-resident-backpressure-metrics-20260625-r1` reported
QAT warm `avg=16.158` tok/s E2E, accepted two and rejected two requests,
`queue_rejections=2`, `queue_rejected_units=6`, and post-run queue depth `0`.
The first top-level soak run at
`/tmp/gemma4-cuda-top-resident-soak-20260625-r1` completed the default two
512-token warm requests at QAT `avg=16.303` tok/s E2E, then ran six 256-token
HTTP requests with concurrency two against the same preloaded server. The soak
step reported aggregate `16.933` tok/s, per-request `avg=9.875` tok/s E2E
(queue time included), `graph_replays=2544`, and `graph_floor=2496`, with no
unsafe graph-capture markers in the server log.
After adding latency distribution gates, the stricter run at
`/tmp/gemma4-cuda-top-resident-soak-latency-20260625-r1` completed the same
profile with QAT warm `avg=16.336` tok/s E2E, soak aggregate `16.934` tok/s,
minimum request `8.430` tok/s E2E against the 8.0 tok/s floor, `p50=30146.0`
ms, `p95=30366.4` ms against the 35000 ms ceiling, `p99=30366.4` ms,
`graph_replays=2544`, and `graph_floor=2496`.
The longer 12-request soak at
`/tmp/gemma4-cuda-top-resident-soak-12req-20260625-r1` kept the same
concurrency-two, 256-token request shape and passed the same latency gates. It
reported QAT warm `avg=16.154` tok/s E2E, soak aggregate `16.972` tok/s,
minimum request `8.477` tok/s E2E, `p50=30176.4` ms, `p95=30200.7` ms,
`p99=30200.7` ms, `graph_replays=4074`, and `graph_floor=3984`.
The unconstrained concurrency-four probe at
`/tmp/gemma4-cuda-top-resident-soak-c4-20260625-r1` kept aggregate throughput
stable at `16.962` tok/s, but failed the per-request latency gate because queued
requests stretched to `p95=105720.2` ms and individual request rates fell as low
as `2.421` tok/s E2E. That is a production backpressure signal, not a decode
kernel regression. The CLI-backed capacity gate at
`/tmp/gemma4-cuda-top-resident-backpressure-cli-20260625-r2` started
`antfly-inference run` with `--max-concurrent-requests 6`; with four concurrent
256-token requests it accepted two, rejected two with HTTP 503 in at most
`5.2` ms, completed the accepted requests at `avg=12.854` tok/s E2E, and kept
graph replay coverage at `1524` against a `1504` floor.

Status checked on 2026-06-25 on an NVIDIA L4 (`sm_89`) with CUDA Toolkit 13.2
and driver R580:

- `zig build -Dcuda=true`, `regen-cuda-artifacts.sh --check --all`,
  `antfly-inference cuda-info --smoke`, and `zig build test -Dcuda=true` pass.
- After the resident default tightening, `/home/timkaye/tim/antfly/.tools/zig-x86_64-linux-0.16.0/zig build -Dcuda=true`
  and `/home/timkaye/tim/antfly/.tools/zig-x86_64-linux-0.16.0/zig build test -Dcuda=true`
  passed from `zig/pkg/inference`; the test run selected 1836 tests with 1750
  passed and 86 skipped.
- `polar4` is a production-candidate opt-in compressed-K/compressed-V path. It
  stays fully resident on CUDA with zero host attention fallback in the E2B and
  12B Q4 checks below.
- `turbo3` is functional and resident, but remains experimental. It is slower
  than `polar4` on the current L4 decode workloads and can change output quality
  more aggressively.
- `polar4` is not the CUDA default yet. Deterministic 12B Q4 f32/polar4 output
  matched in the 32-token raw check, but E2B f32/polar4 output diverged. Promote
  only after an explicit quality/parity acceptance gate, not just the runtime
  gate.

Measured L4 results from `/tmp/antfly-cuda-turboquant-prod`:

| Workload | Cache | Tokens | Load | Warm TTFT | Cold TTFT | Decode tok/s | CUDA KV status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E2B Korean summary | f32 | 128 | 8.56s | 0.30s | 8.86s | 17.11 | 4480/4480 device KV successes |
| E2B Korean summary | polar4 | 128 | 8.43s | 0.30s | 8.73s | 16.70 | 1920 compressed-V writes, 4480 reads |
| 12B Q4 Korean summary | f32 | 40 | 17.01s | 2.01s | 19.02s | 8.67 | 1968/1968 device KV successes |
| 12B Q4 Korean summary | polar4 | 30 | 17.10s | 1.98s | 19.08s | 8.66 | 1488 compressed-V writes, 1488 reads |
| 12B Q4 raw repeat 1 | polar4 | 32 | 17.09s | 0.63s | 17.71s | 9.16 | zero fallback |
| 12B Q4 raw repeat 2 | polar4 | 32 | 17.02s | 0.63s | 17.65s | 9.06 | zero fallback |
| 12B Q4 raw repeat 3 | polar4 | 32 | 16.82s | 0.63s | 17.45s | 9.04 | zero fallback |

The current performance win is memory residency and lower metadata overhead, not
higher tok/s. The block-table upload cache reduced E2B 16-token `polar4`
block-table uploads to 30, and the longer 128-token E2B `polar4` stress run used
135 uploads while completing 4480 device-KV reads with zero fallback.

User-facing E2B CUDA smoke from the repository root:

```sh
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  "Give a one sentence summary of Korean history." \
  --backend cuda \
  --max-tokens 128 \
  --print-timing \
  --print-token-count
```

If running from `zig/pkg/inference/zig-out/bin`, pass an absolute model path.
The model loader treats the first argument as a model path relative to the
current working directory, so `./antfly-inference generate .models/...` from the
binary directory will fail with `NoTokenizerFound`.

```sh
./antfly-inference generate \
  /home/timkaye/tim/antfly/.models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  "Give a one sentence summary of Korean history." \
  --backend cuda \
  --max-tokens 128 \
  --print-timing \
  --print-token-count
```

Optional `polar4` E2B smoke:

```sh
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  "Give a one sentence summary of Korean history." \
  --backend cuda \
  --cache-dtype polar4 \
  --max-tokens 128 \
  --print-timing \
  --print-token-count
```

Validation ladder for compressed KV:

```sh
zig build -Dcuda=true

zig/pkg/inference/scripts/regen-cuda-artifacts.sh --check --all

zig/pkg/inference/zig-out/bin/antfly-inference cuda-info --smoke

zig/pkg/inference/scripts/validate_cuda_turboquant_gemma4.sh --quick

zig/pkg/inference/zig-out/bin/antfly-inference generate \
  /path/to/gemma4-12b-target \
  "Write one sentence about ants." \
  --backend cuda \
  --cache-dtype polar4 \
  --max-tokens 16 \
  --temperature 0 \
  --print-token-ids \
  --print-timing

zig/pkg/inference/zig-out/bin/antfly-inference generate \
  /path/to/gemma4-12b-target \
  "Write one sentence about ants." \
  --backend cuda \
  --cache-dtype turbo3 \
  --max-tokens 16 \
  --temperature 0 \
  --print-token-ids \
  --print-timing
```

## Inference Surface

The first CUDA execution surface should be:

```text
C[M, N] = A[M, K] @ B_quant[N, K]^T
```

where:

- `A` is dense f32 initially; add f16 input once f32 correctness is locked.
- `B_quant` is raw GGUF-packed weight storage.
- `C` is f32 initially; add f16 output only after tolerances and downstream ops
  are explicit.
- `M = 1` decode is the first performance target.
- `M = 2..8` small-batch decode/prompt is the second target.
- `M >= 9` prefill is the third target.

Route every CUDA quantized linear through `graph/quant_matmul.zig`:

- Use `quant_matmul.plan(...)` for row bucket and preferred operator.
- Add CUDA-local capability checks that turn unsupported preferred operators
  into fallback.
- Record counters with the same operator names as Metal/WebGPU/native:
  `mul_mv`, `mul_mv_ext`, `mul_mm`, and `fallback`.
- Do not add public per-format APIs such as `cudaQ4KMatmul`; keep one internal
  descriptor-driven dispatch.

The current CUDA backend also exposes the common dense/model primitives needed
by ClipClap, GLiNER2, and DeBERTa reranker sessions:

- dense f32 linear/bias, dense f16/bf16 weight paths, activation,
  normalization, embedding, concat, convolution, and attention helpers
- optional cuBLASLt f16/bf16 matmul dispatch for eligible dense weights
- GGUF `Q8_0`, `Q4_0`, and `Q4_K` linear kernels
- GLiNER-oriented DeBERTa attention/head helper kernels

Required common kernels are loaded eagerly when the CUDA module is loaded. If a
stale artifact bundle is missing the selected model family's required symbols,
session creation must fail with `CudaKernelUnavailable` instead of silently
falling back to an incomplete GPU path.

CUDA session creation now uses explicit capability profiles:

| Profile | Model families | Required capability group |
|---|---|---|
| `clipclap` | CLIP, CLAP, ClipCLAP embedding paths | dense linears, bias/activation fusions, embedding lookup, layer/RMS norm, concat, conv2d, SDPA |
| `deberta_reranker` | DeBERTa cross-encoder rerankers | `clipclap` primitives plus take-rows, DeBERTa attention, split-last-dim |
| `gliner2` | GLiNER2 recognition | `deberta_reranker` primitives plus GLiNER word embeddings and label GRU combine |

`antfly inference cuda-info --smoke` prints the loaded artifact's profile
capability booleans before running kernel smokes. Production validation should
require the relevant profile to be `true` before running real model fixtures.
## Quantization Priorities

The minimum useful GGUF set is:

1. `Q8_0`: simplest correctness anchor; useful for activation-like data.
2. `Q4_0`: common legacy 4-bit format and simple 32-value blocks.
3. `Q4_K`: common modern GGUF target and the first K-quant proof.

Then broaden:

4. `Q5_K`, `Q6_K`, `Q8_K`.
5. `Q4_1`, `Q5_0`, `Q5_1`, `Q8_1`.
6. `Q2_K`, `Q3_K`, `IQ4_NL`, `IQ4_XS`, `I2_S`, `Q1_0` as target models
   require them.
7. `MXFP4`, `NVFP4`, `TQ1_0`, `TQ2_0`, and other newer formats only after
   their CPU references and model demand are clear.

Every CUDA-supported format needs:

- byte-size agreement with `gguf/tensor_types.zig`
- row-dequant parity with `gguf/quant_codec.zig`
- synthetic matrix parity against CPU dense reference
- real GGUF smoke counters proving the CUDA kernel executed
- fallback behavior for unsupported row shapes and packed expert variants

## Quantized GGUF Limitations

The important constraint is not "CUDA cannot run unquantized GGUF"; it can.
The issue is where performance and memory come from:

- Dense F16/BF16/F32 GGUF weights need dense GEMM. Optional cuBLASLt dispatch
  handles eligible F16/BF16 cases, while driver-only dense kernels remain a
  correctness fallback and are not expected to beat vendor libraries.
- Large unquantized models require much more VRAM than Q4/Q5/Q6 GGUF files, so
  the useful GKE target set is narrower unless we add robust CPU/GPU layer
  offload.
- StableHLO quantized types do not directly encode GGUF block layouts, scales,
  mins, lookup tables, and mixed per-tensor formats. A naive XLA route would
  either dequantize weights to dense buffers or require custom calls.
- Dequantizing all weights to f16/f32 on GPU discards GGUF's main memory
  advantage and can exceed VRAM.
- Custom calls can let XLA invoke our kernels, but then XLA becomes an
  orchestration layer around the same CUDA kernels and brings an experimental
  ABI plus plugin/runtime dependencies.

Therefore:

- Native CUDA should optimize GGUF packed-weight inference.
- XLA/PJRT should optimize dense/static graph inference and serve as a
  validation/packaging path.
- Do not block native CUDA on solving arbitrary unquantized LLM performance.

## Kernel Strategy

### Correctness Kernels

Start simple:

- One CTA computes one or a small group of output elements.
- Load GGUF-packed blocks from global memory.
- Decode in registers or shared memory.
- Accumulate in f32.
- Write f32 output.

These kernels establish memory ownership, module loading, launches, and parity.
They are allowed to be slower than ggml.

### Decoder Kernels

Then implement ggml-shaped decode kernels:

- `mul_mv` for `M = 1`.
- One block or warp group per output row, depending on format and `K`.
- Coalesced reads of packed weight blocks.
- Shared input vector cache when it improves reuse.
- Per-format dot helpers under one kernel family.
- Optional Q8 activation packing only after profiling shows it helps.

### Small Batch And Prefill

Add:

- `mul_mv_ext` for `M = 2..8`.
- `mul_mm` for prompt/prefill.
- Shared temporary activation layout for large `M` if it beats direct dense
  f32/f16 loads.
- Batched QKV and gate/up paired linears once single linear kernels are stable.

### Architecture-Specific Fast Paths

Only after generic kernels work:

- DP4A-style integer dot paths for T4 and later.
- Tensor-core-assisted paths where the quant format can be profitably repacked.
- `sm_80`, `sm_89`, and `sm_90` cubins selected at runtime.

Architecture-specific kernels are optional accelerators. They must fall back to
portable `compute_75` PTX.

## Integration Phases

### Phase 0: Build And Backend Plumbing

- Add `-Dcuda` and `-Dcuda-artifacts`.
- Add `cuda` to graph/backend contracts:
  - `BackendKind.cuda`
  - `TensorStorageClass.cuda_buffer`
  - partition/runtime parsing for `"cuda"`
- Add `cuda` to session backend ordering, CLI choices, and explicit
  `--backend cuda` validation.
- Keep default `auto` order unchanged until CUDA passes real smoke tests.

### Phase 1: Capability Probe

- Implement `CudaDriver` dynamic loader.
- Add an internal smoke probe that prints driver version, selected
  device, compute capability, total memory if available, and artifact mode.
- Test no-CUDA machines: probe returns unavailable without crashing.
- Test CUDA machines: probe succeeds without CUDA toolkit in the container.

### Phase 2: Buffers And Kernel Launch

- Implement device allocation, free, H2D/D2H/D2D copies, stream sync, and module
  loading.
- Embed one tiny PTX kernel such as fill or vector add.
- Capture CUDA JIT info/error logs during module loading so PTX problems are
  visible on the first NVIDIA-box run.
- Add skipped/fallback-safe tests for host copy, kernel launch, and output
  parity.

### Phase 3: Dense Linear Correctness

- Implement basic f32 `linearNoBias` and `linear` for dense weights.
- Return CUDA tensors from `fromFloat32Shape` and copy back through
  `toFloat32`.
- Route only explicit `--backend cuda` to this path.
- Compare small/medium shapes against native CPU.

### Phase 4: Quantized Linear MVP

- Implement CUDA tensor storage for host-packed GGUF weight bytes.
- Implement `Q8_0` and `Q4_0` `mul_mv`.
- Route through `quant_matmul.plan(...)`.
- Add counters for planned operator, actual operator, format, row bucket, and
  fallback reason.
- Add synthetic tests and one real GGUF smoke where CUDA quantized matmul is
  observed.

### Phase 5: First Production GGUF Path

- Implement `Q4_K` `mul_mv`.
- Keep quantized weights resident on device across tokens.
- Add prepared linear slots for decoder runtime requests:
  - QKV
  - output projection
  - FFN gate/up/down
  - LM head
- Add CPU fallback per unsupported format/operator, not per whole model when
  possible.
- Validate fixed-token generation on L4 and T4.

### Phase 6: Broader Format And Operator Coverage

- Add `Q5_K`, `Q6_K`, `Q8_K`.
- Add `mul_mv_ext` for small batches.
- Add `mul_mm` for prefill.
- Add RMSNorm, RoPE, softmax, and attention only after quantized linears are
  stable and measured.

### Phase 7: XLA/PJRT NVIDIA Lane

- Keep `--backend xla` as PJRT, with CUDA GPU plugin supplied externally.
- Document required environment variables:
  - `ANTFLY_INFERENCE_XLA_PLUGIN`
  - `ANTFLY_INFERENCE_PJRT_PLUGIN`
  - `PJRT_PLUGIN_PATH`
  - `PJRT_PLUGIN`
- Use XLA first for dense/static graph models and compiled artifact workflows.
- Do not use XLA as the default GGUF path unless a model is exported to dense
  graph artifacts and fits memory.
- Revisit XLA custom calls only after native CUDA kernels exist and there is a
  concrete need to run them inside compiled graph partitions.

### Phase 8: GKE Container Validation

- Build one Linux image with CUDA backend compiled in and no CUDA runtime
  libraries included.
- Deploy on GKE L4 first.
- Validate that `libcuda.so.1` comes from the NVIDIA driver mount.
- Run:
  - no-CUDA startup fallback
  - CUDA smoke probe
  - dense linear parity smoke
  - `Q8_0`, `Q4_0`, `Q4_K` synthetic parity
  - real GGUF generation with CUDA counters
- Repeat on T4, then A100/H100.

## XLA/PJRT Capability Plan

Use PJRT for:

- whole-model or partitioned dense graphs
- HLO/executable artifact packaging already present in `native_compile.zig`
- correctness comparison for dense paths
- eventual graph-level scheduling around native CUDA results if custom calls
  become worth the dependency

Do not use PJRT for:

- loading raw GGUF packed weights directly into XLA quantized tensors
- the first quantized decoder runtime
- dependency-free CUDA deployment

Concrete PJRT work:

- Make `-Dpjrt=true` a real configurable build option if it is intended for
  NVIDIA deployments; today `build.zig` hardcodes `enable_pjrt` false.
- Add NVIDIA-specific docs for `ANTFLY_INFERENCE_XLA_PLUGIN`/`PJRT_PLUGIN_PATH`.
- Add a dense graph smoke on a CUDA PJRT plugin once available.
- Add a clear error when `--backend xla` is requested without a plugin.
- Keep PJRT artifacts and native CUDA artifacts separate in manifests.

## Testing Matrix

Most tests should run without NVIDIA hardware:

- no-CUDA dynamic loader test
- CUDA symbol table construction test with a fake loader where possible
- GGUF tensor type and byte-size tests
- quant row-dequant tests against `quant_codec.zig`
- quant matmul planner tests in `graph/quant_matmul.zig`
- CUDA artifact presence/currentness test that does not execute GPU code

CUDA-present tests:

- capability probe
- vector fill/add launch
- dense f32 linear parity
- `Q8_0`, `Q4_0`, `Q4_K` matmul parity
- fallback-on-unsupported-format test
- real GGUF generation with fixed prompt/settings and CUDA counters
- GKE L4 container smoke

Correctness rules:

- Dense f32 matmul: tight absolute/relative tolerance.
- Quantized matmul: compare against CPU dequantized or native quant reference
  with explicit per-format tolerance.
- Generation smoke: stable token IDs for fixed seed/settings where sampling is
  deterministic.
- No-CUDA startup behavior: byte-for-byte same CLI behavior where practical,
  except debug/probe logs.

## Telemetry And Debugging

Add counters early:

- device name and compute capability
- selected artifact kind: PTX or cubin/fatbin
- loaded capability profiles: `clipclap`, `gliner2`
- planned quant operator
- actual CUDA operator
- fallback reason
- per-format kernel counts
- H2D/D2H bytes during generation
- resident weight bytes
- peak device bytes

Expose these in existing smoke/generate timing output so acceptance tests can
prove GPU execution instead of just proving successful text generation.

Fallback defaults are intentionally strict:

- Planned quant matmul fallback is rejected unless
  `ANTFLY_CUDA_ALLOW_PLANNED_FALLBACK=1` is set.
- RoPE/GQA host fallback remains debug-only behind
  `ANTFLY_CUDA_ALLOW_HOST_ATTENTION_FALLBACK=1`.
- Any production smoke that enables either flag must report fallback counters
  and should not be used as a release gate unless the fallback is the behavior
  being tested.

## Acceptance Criteria

CUDA is minimally useful when:

- `termite` starts on machines without CUDA and behaves as before.
- The same binary starts on a GKE L4 node and reports CUDA availability when
  requested.
- The container image contains no CUDA runtime, cuBLAS, cuDNN, TensorRT, ONNX
  Runtime, or XLA libraries for the native CUDA path.
- `Q8_0`, `Q4_0`, and `Q4_K` GGUF linears run on the GPU.
- A real GGUF generation smoke shows CUDA quantized matmul counters.
- CPU fallback remains available for unsupported formats and devices.
- XLA/PJRT remains independently usable for dense compiled graph inference when
  a PJRT plugin is supplied.

## Open Questions

- Which GGUF model defines first acceptance: small deterministic fixture,
  a common 7B Q4_K model, or both?
- Should release builds ship only portable PTX at first, or PTX plus L4/T4
  cubins once CI can generate them?
- How aggressive should per-op fallback be before the cost of CPU/GPU transfers
  makes whole-layer fallback preferable?
- Should CUDA direct sessions load before or after Metal in `auto` when
  running on multi-platform developer machines?
- When should CUDA graph launch be introduced for decoder token loops?
