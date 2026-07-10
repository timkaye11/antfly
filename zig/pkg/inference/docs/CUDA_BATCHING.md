# CUDA Generation Batching

CUDA generation batching is available through the server configuration:

```json
{
  "generation_batching": {
    "mode": "on",
    "max_step_items": 2,
    "max_step_query_tokens": 512,
    "max_decode_wait_us": 1000
  }
}
```

The current production envelope batches two decode rows with matching sequence
lengths and KV positions. Prefill steps stay singleton. When more than two
requests are active, the scheduler preserves fairness but uses the optimized
singleton CUDA route.

The row-two path is memory-safe across repeated paged-KV growth and is
deterministic within a batch. It is not enabled by default yet: long generation
can choose a different greedy token stream than singleton execution because the
row kernels have different floating-point reduction order, and the optimized
singleton decoder runtime is currently faster.

For optimal singleton service, use the shared tuned server wrapper documented
in `CUDA_TUNING.md`. It deliberately forces batching off because its
request-scoped graph replay profile is not compatible with interleaved batched
requests.

Run the rollout gate with:

```sh
zig/pkg/inference/scripts/benchmark_gemma4_cuda_batching.py \
  --tokens 256 --concurrency 1 2 4 --max-step-items 2
```

The gate records latency, aggregate throughput, response fingerprints, and
scheduler metrics in `/tmp/antfly-gemma4-cuda-batching` by default. Do not make
batching the default or widen the row cap until the exact-response, concurrency
4 throughput, and singleton p95 latency checks pass.
