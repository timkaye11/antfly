# CUDA Tuning Profiles

Gemma 4 QAT CUDA measurements must use the shared tuning profile in
`scripts/gemma4_qat_cuda_tuning.sh`. The paired CLI benchmark and tuned server
wrapper both source this file so kernel-route settings cannot drift.

Run the tuned, batching-off server with:

```sh
zig/pkg/inference/scripts/with_gemma4_qat_cuda_tuning.sh \
  zig/pkg/inference/zig-out/bin/antfly-inference run \
  --host 127.0.0.1 --port 8080 --models-dir .models --config server.json
```

The wrapper forces continuous batching off. It enables the validated Q8
projection and FFN routes, generated attention, prefill preparation, pinned
scalar transfers, and request-scoped CUDA graph replay used by the paired CLI
benchmark.

CLI benchmark samples run in fresh processes. A server retains its CUDA
allocator across requests, so the wrapper also enables
`ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET=1`. This synchronizes prior
work and resets the pinned temporary-allocation sequence whenever request-owned
KV storage is provisioned. Do not combine this mode with continuous batching.

The default graph KV capacity is 544 tokens. For longer requests, set a capacity
covering prompt and output tokens, rounded up to a KV page:

```sh
ANTFLY_CAPTURE_FORCE_KV_CAPACITY=1056 \
  zig/pkg/inference/scripts/with_gemma4_qat_cuda_tuning.sh \
  zig/pkg/inference/zig-out/bin/antfly-inference run \
  --host 127.0.0.1 --port 8080 --models-dir .models --config server.json
```

Run the paired Antfly/llama.cpp comparison with:

```sh
WARMUPS=2 REPEATS=5 ANTFLY_CACHE_DTYPE=f32 \
LLAMA_CACHE_TYPE_K=f32 LLAMA_CACHE_TYPE_V=f32 \
  zig/pkg/inference/scripts/gemma4_qat_llamacpp_pair_benchmark.sh
```

Measure warmed HTTP server throughput and repeatability with:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_server.py \
  --tokens 256 --warmups 1 --repeats 5
```
