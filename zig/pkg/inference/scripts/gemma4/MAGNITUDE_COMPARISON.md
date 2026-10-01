# Gemma4 E2B QAT comparison

`benchmark_magnitude_gemma4.py` reproduces the five supplied Magnitude prompts,
including the 4,031-token reference-notes prompt. Each fresh server receives a
64-token warm-up before the timed requests. The harness saves exact requests,
complete responses, usage, stream events, model/binary hashes, configuration,
memory observations, and partial receipts on failure.

The sampling contract is temperature 0.8, top-p 0.95, top-k 0, reasoning disabled,
and prompt caching disabled. Magnitude's sampling parameters were inferred from
its source defaults; its effective runtime configuration and RNG seed were not
captured. This is a workload comparison, not seeded output parity.

## Run

Build from `zig/`:

```sh
zig build inference-bench-server -Dmetal=true -Doptimize=ReleaseFast -j1 \
  --prefix /private/tmp/antfly-gemma4-candidate
```

Place the checkpoint at
`MODELS/generators/unsloth/gemma4-e2b-qat/model.gguf`. Verify the tensor types:
the supplied Magnitude checkpoint is named `UD-Q4_K_XL` but contains Q4_0 and
F32 tensors. Its tied embedding is the checkpoint LM head.

From the repository root, run with a new output directory:

```sh
python3 zig/pkg/inference/scripts/gemma4/benchmark_magnitude_gemma4.py \
  --binary /private/tmp/antfly-gemma4-candidate/bin/antfly-inference-bench-server \
  --models-dir /path/to/MODELS \
  --output-dir /private/tmp/antfly-gemma4-measurement \
  --env ANTFLY_INFERENCE_METAL_RESIDENT_NUCLEUS=1 \
  --env TERMITE_METAL_ENABLE_E2B_FLASH_PREFILL_HD256=1 \
  --env TERMITE_METAL_ENABLE_E2B_Q4_0_MM_SG_ALIGNED=1 \
  --env TERMITE_METAL_ENABLE_E2B_Q4_0_PAIR_ACTIVATION_MM=1
```

The E2B routes and resident compiled sampler are opt-in pending performance
qualification. They reuse the native quantized kernels and exact checkpoint
logits. The sampler retains the full-vocabulary nucleus and cutoff ties; it does
not truncate to an approximate top-k. Unsupported configurations decline before
submission; errors after submission propagate without replaying the token.

On the local 16 GiB M4 Air, explicit benchmark budgets were necessary:

```sh
--server-arg=--process-memory-budget-mb --server-arg=6144 \
--server-arg=--combined-budget-mb --server-arg=5632 \
--server-arg=--kv-budget-mb --server-arg=512 \
--server-arg=--scratch-budget-mb --server-arg=512
```

These are benchmark overrides, not production policy changes. The harness
monitors the owned process tree with a 10 GiB physical-footprint limit. Samples
do not establish instantaneous peak usage or total system/GPU residency.

## Qualification

Use `--temperature 0` for greedy output-hash controls. For repeated uncached
long prompts, use `--case long-input-512 --rounds 20`. Alternate fresh baseline
and candidate servers for A/B comparisons, with the same checkpoint, budgets,
and workload. Run `--diagnostic` separately: its timings are not performance
receipts. Do not compile or run another GPU workload during timed requests.

Client decode throughput is `(completion_tokens - 1) / (last_content_time -
first_content_time)`. TTFT starts before the HTTP request. Overall throughput
includes prefill. Compare these with Magnitude's client streaming measurements;
its server-reported generation rate uses a different boundary.

Validate the reused E2B aligned and ragged native Q4_0 routes before measuring
serving performance:

```sh
# From zig/pkg/inference/
zig build bench-metal-gemma4-e2b-prefill-routes \
  -Dmetal=true -Doptimize=ReleaseFast -j1
```

Resident sampling can be disabled with
`ANTFLY_INFERENCE_METAL_RESIDENT_NUCLEUS=0`. Each E2B prefill flag has a matching
`TERMITE_METAL_DISABLE_E2B_*` rollback. The sampler's parallel radix and monotonic
softcap-max stages have independent rollbacks:
`TERMITE_METAL_DISABLE_PARALLEL_NUCLEUS_RADIX=1` and
`TERMITE_METAL_DISABLE_NUCLEUS_MONOTONIC_MAX=1`.
