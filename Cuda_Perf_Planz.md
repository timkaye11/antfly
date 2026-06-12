# Gemma4 CUDA Performance Plan

Last updated: 2026-06-12

## Current State

Gemma4 CUDA correctness is now restored for the local formats we have been
testing.

Confirmed outputs:

- Q8_0 GGUF CUDA:
  - Path: `.models/google/gemma-4-12B-it-q8_0`
  - Prompt: `Write one sentence about ants.`
  - Output: `Ants are highly social insects that live in complex colonies and work together to build intricate underground structures.`
  - 20 tokens, generation time about 7.7s after load.
- Q4_K GGUF CUDA:
  - Path: `.models/google/gemma-4-12B-it-q4_k`
  - Prompt: `Write one sentence about ants.`
  - Output: `Ants are highly social insects that work together in complex colonies to gather food and build intricate underground structures.`
  - 21 tokens, generation time about 8.2s after load.
- Full BF16 safetensors CUDA:
  - Path: `.models/google/gemma-4-12B-it`
  - First token matches Q8/Q4: token id `14054`, text `Ant`.
  - One-token streamed control takes about 123s.
  - 32-token streamed control timed out at 300s.

Root cause of the prior bad-token failure:

- CUDA `reshape2d` is a non-owning view.
- Several Gemma4 norm helpers returned a reshaped view of `normed_flat` while
  also scheduling `defer cb.free(normed_flat)`.
- With CUDA's async stream and temp-buffer cache, returned tensors could point
  at freed/reusable device storage.
- The fix added an owned CUDA shape clone path and switched returned Gemma4 norm
  reshape results to `reshape2dOwned`.

Older assumptions that Q8_0/Q4_K CUDA generation returns garbage are obsolete.
The next problem is performance plus a real quality gate.

## Goals

Primary targets on the L4, batch size 1:

- Add a small eval gate before performance work changes more numerics.
- Improve Q8_0 and Q4_K resident decode throughput.
- Improve BF16 streamed first-token and short-control throughput enough to keep
  the full model usable as a correctness reference.

Longer-term performance targets from `Gemma4_NextSteps_Cl.md` remain:

- Q8_0: 15-20 tok/s.
- Q4_K or Q4-mixed: 25-30 tok/s.

Current rough measured throughput is about 2.5-2.7 tok/s for Q8_0/Q4_K on the
short smoke prompt, so there is still substantial headroom.

## Implementation Status - 2026-06-12

Implemented from Phase 0:

- `generate --print-timing` now prints `decode_tok_per_s`.
- `generate --json-timing PATH` writes stable JSON with:
  - model/backend, token count, finish reason, decode tok/s
  - outer timing and inner generation timing
  - CUDA launches, syncs, transfer bytes, temp-buffer stats
  - QKV/Q4_K/BF16/cuBLASLt/dense-stream/prefetch/device-KV counters
- Q8_0 smoke with `--max-tokens 2` passed on CUDA:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - inner prefill: `1831ms`
  - inner decode: `289ms`
  - `decode_tok_per_s=6.920`
  - JSON validated with `jq`.

Implemented from Phase 1:

- `compare --quality-eval` is available.
- Added flags:
  - `--prompt-file PATH`
  - `--max-prompts N`
  - `--json-out PATH`
- Added prompt fixture:
  - `zig/pkg/inference/testdata/gemma4_quality_prompts.txt`
- Q8_0 one-prompt BF16-reference CUDA quality smoke passed:
  - native top-1: `14054`
  - reference top-1: `14054`
  - top-k overlap: `7/8`
  - top-1 pct: `100%`
- Q4_K one-prompt BF16-reference CUDA quality smoke passed:
  - native top-1: `14054`
  - reference top-1: `14054`
  - top-k overlap: `6/8`
  - top-1 pct: `100%`

Validation:

- ReleaseFast CUDA build passed:
  - `../../../.tools/zig-x86_64-linux-0.16.0/zig build -Dcuda=true -Doptimize=ReleaseFast --global-cache-dir /tmp/zig-global-cache`
- `git diff --check` passed for the touched files.

Still pending from the original Phase 0/1 acceptance:

- Extend `bench-cuda` with Q8_0 Gemma4 decode shapes and explicit Q4_K tile4
  vs tile8 row-1 comparison.
- Run the full 12-prompt Q8_0 and Q4_K quality gates. The one-prompt smoke is
  healthy, but the full run is currently too expensive because each prompt
  reloads/evaluates the slow streamed BF16 reference.

## Important Findings

- Q8_0 still uses `termite_linear_q8_0_f32`, a one-thread-per-output scalar
  kernel. This likely makes Q8 bandwidth much worse than the L4 should allow.
- Q4_K already has a tiled kernel family. Q4_K not being much faster than Q8_0
  means format-independent overhead also matters.
- Q4_K fast decode is gated by `ANTFLY_CUDA_ENABLE_Q4K_DECODE_FAST=1`. The
  previously observed `decode_fallbacks` count is not enough evidence by itself
  because the fast path defaults off.
- Device greedy argmax already exists, but Gemma4 disables it because
  `final_logit_softcapping > 0`. Softcap is monotonic, so pure greedy argmax
  can use device argmax if tie-breaking matches host behavior.
- CUDA `releaseDeviceBuffer` currently synchronizes on uncached frees to protect
  against the use-after-free class that caused bad tokens. Do not remove this
  safety without replacing it with deferred/event-safe reclamation.
- BF16 CUDA uses cuBLASLt successfully, but streamed dense/FFN weight reads and
  uploads dominate the 123s one-token control.

## Phase 0 - Baseline And Measurement

Implementation:

- Add `decode_tok_per_s` to `--print-timing`.
- Add `--json-timing PATH` for stable before/after comparisons.
- Include these fields in JSON timing:
  - load time, prefill time, decode time, generated token count, decode tok/s
  - CUDA kernel launches and sync counts
  - upload/download sync counts and bytes
  - Q4_K fast hits/fallbacks
  - Q8/Q4/fused QKV counters
  - BF16 cuBLASLt counters
  - dense stream read/upload times and resident bytes
  - temp-buffer cache hits/misses/releases/evictions
  - device KV success/failure counters
- Extend `bench-cuda` with Q8_0 real decode shapes:
  - `3840 -> 4096`
  - `3840 -> 2048`
  - `3840 -> 15360`
  - `15360 -> 3840`
  - `3840 -> 262144`
- Keep Q4_K microbench cases and add a tile4 vs tile8 row-1 comparison.

Baseline commands:

```bash
cd /home/timkaye/tim/antfly/zig/pkg/inference

../../../.tools/zig-x86_64-linux-0.16.0/zig build \
  -Dcuda=true \
  -Doptimize=ReleaseFast \
  --global-cache-dir /tmp/zig-global-cache

zig-out/bin/antfly-inference generate \
  /home/timkaye/tim/antfly/.models/google/gemma-4-12B-it-q8_0 \
  "Write one sentence about ants." \
  --backend cuda \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 64 \
  --temperature 0 \
  --print-token-ids \
  --print-timing

zig-out/bin/antfly-inference generate \
  /home/timkaye/tim/antfly/.models/google/gemma-4-12B-it-q4_k \
  "Write one sentence about ants." \
  --backend cuda \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 64 \
  --temperature 0 \
  --print-token-ids \
  --print-timing

zig-out/bin/antfly-inference cuda-info --smoke
zig-out/bin/antfly-inference bench-cuda
```

Acceptance:

- Baseline numbers are recorded for Q8_0, Q4_K, and BF16 first-token control.
- `cuda-info --smoke` passes.
- Smoke prompt token ids remain coherent.

## Phase 1 - Quality Gate Smoke Eval

Do this before numerics-changing performance work.

Implementation:

- Add `compare --quality-eval`.
- Add flags:
  - `--prompt-file PATH`
  - `--max-prompts N`
  - `--json-out PATH`
- Add a checked-in small prompt file with 12 short prompts covering:
  - simple factual response
  - one-sentence instruction
  - short reasoning
  - list formatting
  - code-adjacent text
  - creative sentence
  - raw prompt and chat-template prompt coverage
- Run BF16 and candidate sequentially under CUDA budgets to avoid loading both
  models at once.
- Initial metrics:
  - top-1 agreement
  - top-k overlap
  - reference top token rank in candidate logits
  - candidate top token rank in reference logits
  - empty/special top-token failures
  - per-prompt timing

Initial smoke gates:

- Q8_0:
  - no empty/special top-1 tokens
  - top-1 agreement >= 90%
- Q4_K:
  - no empty/special top-1 tokens
  - top-1 agreement >= 75%

Notes:

- Do not default to BF16 native/CPU teacher logits for this 12B model.
- Use BF16 CUDA sequential first-token/logit eval first.
- Add full multi-token teacher-forced KL only after BF16 streaming is fast
  enough to make that practical.

Acceptance:

- Q8_0 and Q4_K pass the 12-prompt smoke gate.
- JSON output is deterministic enough for before/after comparisons.

## Phase 2 - BF16 Dense Streaming Throughput

BF16 production serving is out of scope on this card, but BF16 first-token and
short-control throughput are in scope because BF16 is the correctness reference.

Implementation:

- Run BF16 one-token control with:
  - `ANTFLY_INFERENCE_CUDA_DENSE_STREAM_BUDGET_MB=2048`
  - `ANTFLY_INFERENCE_CUDA_DENSE_STREAM_BUDGET_MB=8192`
  - `ANTFLY_INFERENCE_CUDA_DENSE_STREAM_BUDGET_MB=12000`
- Compare:
  - total first-token time
  - dense stream hits/misses/evictions
  - read ms
  - H2D ms
  - resident MB
  - prefetch ready hits and inflight steals
- If larger budgets help, derive the default dense stream budget from available
  backend budget after KV/scratch reservations instead of using a fixed 2 GB
  default.
- Tune dense prefetch only after budget tuning is measured.

Acceptance:

- BF16 one-token time improves materially from about 123s, or counters prove
  dense stream cache size is not the limiting factor.
- BF16 first token remains `14054` on the ant prompt.

## Phase 3 - Safe Sync And Logits Transfer Reduction

Implementation:

- Relax device greedy argmax for Gemma4 only for pure greedy decode:
  - `temperature == 0`
  - no sampling path
  - no grammar path requiring host logits
  - no caller requiring full logits
- Keep host logits route for sampling and diagnostics.
- Verify device argmax tie-breaking matches host argmax.
- Use monotonicity of final logit softcap to allow argmax before softcap.
- Replace sync-on-free only with a safe alternative:
  - deferred-free list drained at natural sync points, or
  - event-based reclamation
- Do not simply delete the synchronization added for correctness.
- Check device KV counters. If device KV success is not 100%, fix the specific
  `device_kv_fail_*` cause before further decode optimization.

Acceptance:

- Greedy token ids remain identical for Q8_0 and Q4_K smoke prompts.
- Full-logit D2H disappears for pure greedy decode.
- Sync counts per token drop without reintroducing activation trace corruption.
- `compare --activation-trace` spot check still passes.

## Phase 4 - Quantized Matmul Work

Q4_K:

- Run Q4_K with `ANTFLY_CUDA_ENABLE_Q4K_DECODE_FAST=1`.
- Compare tile8 vs tile4 in `bench-cuda` and end-to-end generation.
- If tile8 is correct and faster, make it the default for row-1 decode.
- Fix counters so flag-off is not reported as a fast-path fallback.

Q8_0:

- Add tiled Q8_0 row-1 matvec kernels modeled on the existing Q4_K tile family.
- Wire launchers through `kernels.zig`.
- Dispatch from the `.Q8_0` branch in `cuda_compute.zig`.
- Keep scalar Q8_0 as a fallback.
- Cover the major Gemma4 shapes:
  - Q/K/V/O projections
  - FFN gate/up/down
  - tied LM-head

Acceptance:

- Q4_K fast path has zero unexpected kernel fallbacks when enabled.
- Q8_0 tiled microbench shows a clear GB/s improvement over scalar Q8_0.
- Q8_0 and Q4_K pass Phase 1 smoke eval after changes.

## Phase 5 - Launch Reduction And Fusions

Implementation:

- Add Q8_0 fused QKV, mirroring the existing Q4_K fused QKV path.
- Add CUDA `linearNoBiasPair` for gate+up, using the existing vtable hook.
- Verify existing fusions engage through counters:
  - head-norm/RoPE fusion
  - QKV fusion
  - RMSNorm/residual/scale fusion where applicable
- Re-measure kernel launches per token after each fusion.

Acceptance:

- Kernel launches per token drop materially.
- Greedy token ids remain unchanged for numerics-neutral fusions.
- Phase 1 smoke eval still passes.

## Phase 6 - Later High-Complexity Work

Only start if Phases 1-5 do not reach target speed.

Options:

- CUDA graph capture of steady-state decode:
  - add graph APIs to `driver.zig`
  - use a dedicated decode arena with stable buffer addresses
  - keep env kill-switch
- Int8 activation path:
  - Q8_1 activation quantization
  - `__dp4a` matvec for Q8_0 x Q8_1 and Q4_K x Q8_1
  - requires Phase 1 eval gate because numerics change

Deferred:

- Flash-style prefill attention.
- MTP/speculative decoding tuning.
- BF16 production serving on the L4.

## Critical Files

- `zig/pkg/inference/src/native_generate.zig`
- `zig/pkg/inference/src/cli/compare_generate.zig`
- `zig/pkg/inference/src/bench/cuda_microbench.zig`
- `zig/pkg/inference/src/architectures/gpt.zig`
- `zig/pkg/inference/src/ops/cuda/cuda_compute.zig`
- `zig/pkg/inference/src/ops/cuda/kernels.zig`
- `zig/pkg/inference/src/ops/cuda/driver.zig`
- `zig/pkg/inference/src/ops/cuda/context.zig`
- `zig/pkg/inference/src/ops/cuda/artifacts/inference_cuda_kernels.cu`
- `zig/pkg/inference/src/ops/cuda/artifacts/inference_cuda_kernels.ptx`

Any `.cu` change must regenerate PTX through the repo's CUDA artifact script
and then pass `cuda-info --smoke` on the L4.

## Expected Outcome

First implementation pass:

- Reliable quality smoke gate for Q8_0 and Q4_K.
- Better BF16 first-token control time or a clear measurement proving the next
  BF16 bottleneck.
- Q4_K fast decode path validated and enabled if it wins.
- Q8_0 tiled matvec baseline in place.

Longer-term:

- Q8_0 approaches 15-20 tok/s after tiled Q8_0, reduced logits transfer, safe
  sync reduction, and launch fusions.
- Q4_K approaches 25-30 tok/s after fast decode, reduced sync/logits transfer,
  and launch fusions.
