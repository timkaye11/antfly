# Gemma4 CUDA Performance Plan

Last updated: 2026-06-14

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

Current sustained measured throughput on the ant smoke after the parallel
RMSNorm, RMSNorm+residual fusion, and 512-wide GQA decode work is about:

- Q8_0: `6.57 tok/s`.
- Q4_K: `5.14 tok/s`.

Very short two-token latency smokes can report about 8.6-10 tok/s, but the
20-token runs are the better steady-state planning baseline.

## Implementation Status - 2026-06-13

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

Implemented from Phase 2:

- CUDA dense-stream budget now keeps the explicit
  `ANTFLY_INFERENCE_CUDA_DENSE_STREAM_BUDGET_MB` override, but otherwise derives
  the default from the active backend run budget after KV/scratch reservations.
- The derived default is clamped to a conservative 2-12 GB range. With the
  current `--backend-budget-mb 19000` control, the no-env BF16 run reaches about
  `11936 MB` dense-stream resident bytes instead of the old 2 GB default.
- `compare --quality-eval` no longer reloads both models for every prompt:
  - candidate prompts run under one candidate model load
  - the candidate model is released
  - reference prompts run under one BF16 reference model load
  - this keeps peak memory sequential while preserving CUDA session caches across
    prompts where possible
- Q8_0 two-prompt quality smoke after this change passed:
  - prompt count: `2`
  - top-1 pct: `100%`
  - first prompt top-1: `14054`
  - second prompt top-1: `11634`

Implemented from Phase 3/4 first pass:

- Gemma-family final-logit softcap now permits the pure-greedy device argmax
  fast path while keeping non-Gemma capped logits conservative.
- Q4_K row-1 decode fast path now defaults on. The fallback counter now reports
  unexpected tile8 kernel fallback, not "fast path disabled by env".
- Added a Q8_0 row-1 tiled matvec kernel:
  - CUDA source: `termite_linear_q8_0_f32_tile4`
  - Zig launcher: `launchLinearQ8_0Tile4F32`
  - `.Q8_0` dispatch uses tile4 for `rows == 1` and keeps scalar Q8_0 fallback
- Regenerated the embedded CUDA PTX artifact.
- `compare --quality-eval` now reports per-prompt candidate/reference elapsed
  milliseconds in both console output and JSON.

Implemented from Phase 5A-C first pass:

- `generate --json-timing` and `--print-timing` now expose:
  - `launches_per_token`
  - `syncs_per_token`
  - Q8/Q4 linear-pair fusion counters
- Added Q8_0 fused QKV:
  - CUDA source: `termite_linear_q8_0_qkv_nobias_f32_tile4`
  - Zig launcher: `launchLinearQ8_0QkvNoBiasTile4F32`
  - Q8_0 QKV dispatch now records `qkv_fused_q8`
- Added quantized no-bias pair fusion for Gemma gate/up:
  - CUDA source: `termite_linear_q8_0_pair_nobias_f32_tile4`
  - CUDA source: `termite_linear_q4_k_pair_nobias_f32_tile4`
  - Zig launchers:
    - `launchLinearQ8_0PairNoBiasTile4F32`
    - `launchLinearQ4KPairNoBiasTile4F32`
  - CUDA now implements the `linearNoBiasPair` vtable hook and falls back to
    two `linearNoBias` calls for unsupported formats or missing optional kernels

Implemented from Phase 5D-F first pass:

- Phase 5E full-logit transfer cleanup is implemented for pure-greedy CUDA
  generation:
  - prefill can now return a CUDA greedy token instead of cached host logits
  - the final scheduler-owned prefill chunk falls through to that path when it
    is safe
  - Gemma4 `suppress_tokens` are handled by a new device masked-argmax path
    rather than forcing host logits
- Added a small masked argmax backend hook:
  - ops hook: `argmaxLastRowSuppressTensor`
  - CUDA source: `termite_argmax_last_row_suppress_f32`
  - Zig launcher: `launchArgmaxLastRowSuppressF32`
  - regenerated embedded CUDA PTX
- Phase 5F sync-safe deferred reclamation infrastructure is implemented:
  - `ANTFLY_INFERENCE_CUDA_DEFER_FREE` defaults on
  - `ANTFLY_INFERENCE_CUDA_DEFER_FREE_BUDGET_MB` controls pending-free pressure
  - pending uncached frees drain after existing stream synchronizations, on
    allocation pressure, and during CUDA context teardown
  - timing JSON/print output now reports deferred-free counters
- Follow-up measurement implementation:
  - `generate --print-timing` now reports `cuda_generate_counts` and
    `cuda_generate_rates`
  - `generate --json-timing` now writes a compact `cuda_generate` object
  - these generation-scoped counters exclude model-load uploads/syncs and are
    the preferred launch/sync signal for decode throughput work
- Added CUDA smoke coverage for the suppress-token masked argmax kernel.
- Phase 5D residual/RMSNorm/scale fusion was audited, but not forced:
  - CUDA already had an RMSNorm+residual+scale kernel wired in the backend
  - the current Gemma4 graph does not present that exact operand pattern at the
    local call sites
  - a correct Phase 5D speedup needs a graph/helper change for Gemma4 post-norm
    plus residual/output-scale epilogues, not just flipping the existing kernel

Implemented from Phase 5E follow-up:

- Added fused quantized LM-head greedy argmax for CUDA Q8_0 and Q4_K:
  - ops hook: `linearNoBiasArgmaxLastRowSuppressTensor`
  - CUDA sources:
    - `termite_linear_q8_0_argmax_stage1_tile4`
    - `termite_linear_q4_k_argmax_stage1_tile4`
    - `termite_argmax_reduce_pairs_f32`
  - Zig launchers:
    - `launchLinearQ8_0ArgmaxTile4F32`
    - `launchLinearQ4KArgmaxTile4F32`
  - the existing unsuppressed tensor argmax path now tries the fused quantized
    route before falling back to materialized logits
  - Gemma4 suppress-token greedy generation now tries the fused quantized route
    before falling back to device logits plus masked argmax
  - timing JSON/print output now reports `lm_head_argmax_fused_q8`,
    `lm_head_argmax_fused_q4`, and `lm_head_argmax_fallbacks`

Implemented from Phase 5G/H epilogue probe:

- PLE gating now uses the existing backend `activationMultiply` hook, with the
  previous activation-then-multiply sequence kept as fallback.
- Fixed the PLE output-scale fusion call order so `addMultiplyScalarTensor`
  computes `normed * scale + hidden`, matching the unfused fallback.
- CUDA `addMultiplyScalarTensor` now defaults on, and timing output/JSON report:
  - `activation_multiply_fused`
  - `add_mul_scalar_fused`
- Measured result on the current Gemma4 Q8_0/Q4_K ant smokes:
  - `activation_multiply_fused` counts existing FFN activation-product fusion
  - `add_mul_scalar_fused=0`
  - PLE/scaled-residual fusion is not active on these runs
  - launch counts and throughput are unchanged

Implemented from Phase 5H active norm-kernel work:

- Replaced scalar-per-row CUDA RMSNorm kernels with 256-thread per-row block
  reductions:
  - `termite_rms_norm_f32`
  - `termite_rms_norm_add_mul_scalar_f32`
  - `termite_rms_norm_bare_f32`
- Kept the same exported CUDA symbols and Zig launch wrappers; only the launch
  block width changed from one thread to `f32_tiled_threads`.
- Added CUDA smoke coverage for `termite_rms_norm_bare_f32`, which Gemma4 uses
  heavily for V norm.
- Regenerated embedded CUDA PTX.

Implemented from Phase 5H head-norm/RoPE probe:

- Replaced the active fused head-norm/RoPE kernel's per-element RMS recompute
  with a block-per-head/chunk reduction:
  - CUDA source: `termite_rms_norm_heads_rope_f32`
  - Zig launcher now launches one 256-thread block per head/chunk instead of a
    flat element grid
- Regenerated embedded CUDA PTX.
- Added finer CUDA norm launch counters in timing output/JSON:
  - `launch_norm_layer`
  - `launch_norm_add_layer`
  - `launch_norm_rms`
  - `launch_norm_rms_add`
  - `launch_norm_rms_add_mul_scalar`
  - `launch_norm_rms_bare`
  - `launch_norm_head_rope`

Implemented from Phase 5H RMSNorm+residual epilogue fusion:

- Added a CUDA fused weighted RMSNorm+residual kernel:
  - CUDA source: `termite_rms_norm_add_f32`
  - Zig launcher: `launchRmsNormAddF32`
  - ops hook: `rmsNormAddTensor`
- Routed Gemma post-attention and post-FFN post-norm residuals through the new
  hook when activation tracing/tensor dumps are off.
- Kept diagnostics conservative: activation trace, GPT tensor dumps, and gated
  layer dumps preserve the old intermediate `attn-post` / `ffn-post` tensors.
- Added CUDA smoke coverage for the fused kernel.
- Timing output/JSON now report:
  - `launch_norm_rms_add`
  - `rms_norm_add_fused`

Implemented from the attention-path probe:

- CUDA Gemma4 attention is custom grouped-query causal attention, not
  FlashAttention.
- Timing output/JSON now report:
  - `launch_attention_gqa_decode`
  - `launch_attention_gqa_scalar`
- Initial two-token Q8_0/Q4_K counters showed `gqa_decode=80` and
  `gqa_scalar=16`: the 40 sliding-attention layers used the decode kernel, but
  the 8 full-attention layers fell back to the scalar GQA kernel because
  Gemma4 uses `global_head_dim=512` for full attention.
- Extended `termite_gqa_attention_decode_f32` and its launcher to support
  `head_dim <= 512`, using 512 CUDA threads for the full-attention heads.
- Regenerated embedded CUDA PTX.

Implemented from the bench-cuda Gemma4 shape pass:

- `bench-cuda` now accepts `--gemma4-shapes`.
- The new mode adds synthetic row-1 Gemma4 decode matvec coverage for:
  - Q projection: `3840 -> 4096`
  - KV projection: `3840 -> 2048`
  - attention output projection: `4096 -> 3840`
  - full-attention output projection: `8192 -> 3840`
  - FFN gate/up: `3840 -> 15360`
  - FFN down: `15360 -> 3840`
  - tied LM head: `3840 -> 262144`
- The table reports Q8_0 scalar vs tile4 and Q4_K scalar vs tile4 vs tile8.

Validation:

- ReleaseFast CUDA build passed:
  - `../../../.tools/zig-x86_64-linux-0.16.0/zig build -Dcuda=true -Doptimize=ReleaseFast --global-cache-dir /tmp/zig-global-cache`
- `git diff --check` passed for the touched files.
- Post-Phase-2 Q8_0 CUDA generate smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=6.897`
- Post-Phase-2 no-env BF16 one-token control passed:
  - output token: `14054`
  - output text: `Ant`
  - dense-stream resident: about `11936 MB`
  - prefill remains about `123.9s`
- Post-Phase-3/4 Q8_0 two-token CUDA smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - inner prefill: `1874ms`
  - inner decode: `204ms`
  - `decode_tok_per_s=9.804`
  - `launch_argmax=1`
- Post-Phase-3/4 Q4_K two-token CUDA smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - inner prefill: `3016ms`
  - inner decode: `232ms`
  - `decode_tok_per_s=8.621`
  - `q4k_decode_fast_hits=209`
  - `q4k_decode_fast_fallbacks=0`
- Q8_0 sustained smoke, `--max-tokens 32`, stopped naturally after 20 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.745`
  - inner prefill: `1831ms`
  - inner decode: `4215ms`
  - launches: `19184`, about `959` launches per emitted token
  - QKV fused count: `0`
  - QKV fallback unsupported count: `840`
  - downloads: one full-logit download plus 20 token-id downloads
- Q4_K sustained smoke, `--max-tokens 32`, stopped naturally after 21 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.074`
  - inner prefill: `3012ms`
  - inner decode: `5155ms`
  - launches: `18333`, about `873` launches per emitted token
  - Q4 fused QKV count: `880`
  - Q4_K fast decode hits: `4389`
  - Q4_K fast decode fallbacks: `0`
- Post-Phase-5 Q8_0 two-token smoke passed:
  - output tokens: `14054 236751`
  - `decode_tok_per_s=10.050`
  - launches per token: `828.500`
  - Q8 fused QKV hits: `80`
  - Q8 linear-pair hits: `96`
  - QKV fallback unsupported: `0`
  - linear-pair fallbacks: `0`
- Post-Phase-5 Q4_K two-token smoke passed:
  - output tokens: `14054 236751`
  - `decode_tok_per_s=8.696`
  - launches per token: `828.500`
  - Q4 fused QKV hits: `80`
  - Q4 linear-pair hits: `96`
  - Q4_K fast decode fallbacks: `0`
  - linear-pair fallbacks: `0`
- Post-Phase-5 Q8_0 sustained smoke, `--max-tokens 32`, stopped naturally after
  20 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.766`
  - inner prefill: `2435ms`
  - inner decode: `4196ms`
  - launches: `16496`, `824.800` launches per emitted token
  - syncs: `687`, `34.350` syncs per emitted token
  - Q8 fused QKV hits: `840`
  - Q8 linear-pair hits: `1008`
  - QKV fallback unsupported: `0`
  - linear-pair fallbacks: `0`
- Post-Phase-5 Q4_K sustained smoke, `--max-tokens 32`, stopped naturally after
  21 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.077`
  - inner prefill: `3398ms`
  - inner decode: `5151ms`
  - launches: `17277`, `822.714` launches per emitted token
  - syncs: `688`, `32.762` syncs per emitted token
  - Q4 fused QKV hits: `880`
  - Q4 linear-pair hits: `1056`
  - Q4_K fast decode fallbacks: `0`
  - linear-pair fallbacks: `0`
- Post-Phase-5D-F Q8_0 one-token debug smoke:
  - `executePrefill` reports `prefill_greedy=true`
  - output token: `14054`
  - D2H dropped from one `19922944` byte full-logit download to `1x4`
  - `cuda_transfer_breakdown`: `to_f32_calls=1`, `to_f32_bytes=4`
- Post-Phase-5D-F Q8_0 sustained smoke, `--max-tokens 32`, stopped naturally
  after 20 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.764`
  - inner prefill: `2320ms`
  - inner decode: `4198ms`
  - total launches: `16497`, `824.850` launches per emitted token
  - total syncs: `687`, `34.350` syncs per emitted token
  - generation-scoped syncs: `21`, `1.050` syncs per emitted token
  - downloads: `21x4`, `d2h_bytes=84`
  - Q8 fused QKV hits: `840`
  - Q8 linear-pair hits: `1008`
  - deferred-free counters stayed zero because `temp_evictions=0`
- Post-Phase-5D-F Q4_K sustained smoke, `--max-tokens 32`, stopped naturally
  after 21 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.082`
  - inner prefill: `3231ms`
  - inner decode: `5144ms`
  - total launches: `17278`, `822.762` launches per emitted token
  - total syncs: `688`, `32.762` syncs per emitted token
  - generation-scoped syncs: `22`, `1.048` syncs per emitted token
  - downloads: `22x4`, `d2h_bytes=88`
  - Q4 fused QKV hits: `880`
  - Q4 linear-pair hits: `1056`
  - Q4_K fast decode fallbacks: `0`
  - deferred-free counters stayed zero because `temp_evictions=0`
- Phase 5F pressure validation, Q8_0 two-token smoke with
  `ANTFLY_INFERENCE_CUDA_TEMP_CACHE_MB=0`:
  - deferred-free enabled:
    - output tokens: `14054 236751`
    - generation-scoped syncs: `2`, `1.000` syncs/token
    - generation temp evictions: `2108`
    - generation deferred frees queued: `2108`
    - deferred-free drains: `2`
  - deferred-free disabled with `ANTFLY_INFERENCE_CUDA_DEFER_FREE=0`:
    - output tokens: `14054 236751`
    - generation-scoped syncs: `2110`, `1055.000` syncs/token
    - generation temp evictions: `2108`
    - deferred frees queued: `0`
  - conclusion: deferred-free is a real safety/performance win under temp-cache
    pressure, even though resident Q8_0/Q4_K default runs do not exercise it.
- Q8_0 self-reference CUDA quality-eval timing smoke passed:
  - prompt count: `1`
  - top-1 pct: `100%`
  - native/reference top-1: `14054`
  - native/reference elapsed: `1816ms` / `1819ms`
- Post-fused-LM-head Q8_0 two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=9.709`
  - generation-scoped `lm_head_argmax_fused_q8=2`
  - generation-scoped `lm_head_argmax_fallbacks=0`
- Post-fused-LM-head Q4_K two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=8.368`
  - generation-scoped `lm_head_argmax_fused_q4=2`
  - generation-scoped `lm_head_argmax_fallbacks=0`
- Post-fused-LM-head Q8_0 sustained smoke, `--max-tokens 32`, stopped
  naturally after 20 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.761`
  - generation-scoped launches: `16497`, `824.850` launches per emitted token
  - generation-scoped syncs: `21`, `1.050` syncs per emitted token
  - downloads: `21x4`, `d2h_bytes=84`
  - uploads from suppress-token staging: `h2d_bytes=480`
  - generation-scoped `lm_head_argmax_fused_q8=21`
  - generation-scoped `lm_head_argmax_fallbacks=0`
- Post-fused-LM-head Q4_K sustained smoke, `--max-tokens 32`, stopped
  naturally after 21 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.108`
  - generation-scoped launches: `17278`, `822.762` launches per emitted token
  - generation-scoped syncs: `22`, `1.048` syncs per emitted token
  - downloads: `22x4`, `d2h_bytes=88`
  - uploads from suppress-token staging: `h2d_bytes=496`
  - generation-scoped `lm_head_argmax_fused_q4=22`
  - generation-scoped `lm_head_argmax_fallbacks=0`
- Post-epilogue-probe Q8_0 sustained smoke, `--max-tokens 32`, stopped
  naturally after 20 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.776`
  - generation-scoped launches: `16497`, `824.850` launches per emitted token
  - generation-scoped `activation_multiply_fused=1008`
  - generation-scoped `add_mul_scalar_fused=0`
- Post-epilogue-probe Q4_K sustained smoke, `--max-tokens 32`, stopped
  naturally after 21 tokens:
  - output remains coherent
  - `decode_tok_per_s=4.086`
  - generation-scoped launches: `17278`, `822.762` launches per emitted token
  - generation-scoped `activation_multiply_fused=1056`
  - generation-scoped `add_mul_scalar_fused=0`
- Post-parallel-RMSNorm Q8_0 sustained smoke, `--max-tokens 32`, stopped
  naturally after 20 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q8_0 ant output
  - `decode_tok_per_s=6.289`
  - inner prefill: `2078ms`
  - inner decode: `3180ms`
  - generation-scoped launches unchanged: `16497`, `824.850` launches/token
  - generation-scoped `launch_norm=7077`
  - downloads remain token ids only: `d2h_bytes=84`
- Post-parallel-RMSNorm Q4_K sustained smoke, `--max-tokens 32`, stopped
  naturally after 21 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q4_K ant output
  - `decode_tok_per_s=5.035`
  - inner prefill: `2906ms`
  - inner decode: `4171ms`
  - generation-scoped launches unchanged: `17278`, `822.762` launches/token
  - generation-scoped `launch_norm=7414`
  - downloads remain token ids only: `d2h_bytes=88`
- Post-head-norm/RoPE-parallel Q8_0 sustained smoke, `--max-tokens 32`,
  stopped naturally after 20 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q8_0 ant output
  - `decode_tok_per_s=6.311`
  - inner prefill: `2083ms`
  - inner decode: `3169ms`
  - generation-scoped launches unchanged: `16497`, `824.850` launches/token
  - `cuda_head_norm_rope_counts: fused_hits=1920 fused_fallbacks=0`
- Post-head-norm/RoPE-parallel Q4_K sustained smoke, `--max-tokens 32`,
  stopped naturally after 21 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q4_K ant output
  - `decode_tok_per_s=5.010`
  - inner prefill: `2922ms`
  - inner decode: `4192ms`
  - generation-scoped launches unchanged: `17278`, `822.762` launches/token
  - `cuda_head_norm_rope_counts: fused_hits=2016 fused_fallbacks=0`
- Post-norm-counter Q8_0 two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=12.903`
  - generation-scoped norm launches: `674`
  - norm breakdown:
    - `launch_norm_rms=482`
    - `launch_norm_rms_bare=96`
    - `launch_norm_head_rope=96`
    - layer/add-layer/RMSNorm-add-mul-scalar are all `0`
- Post-RMSNorm+residual-fusion Q8_0 two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=13.245`
  - generation-scoped launches dropped from `829/token` to `733/token`
  - elementwise launches dropped from `288` to `96`
  - `rms_norm_add_fused=192`
  - norm breakdown moved weighted RMSNorm residual epilogues into
    `launch_norm_rms_add=192`
- Post-RMSNorm+residual-fusion Q8_0 sustained smoke, `--max-tokens 32`,
  stopped naturally after 20 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q8_0 ant output
  - `decode_tok_per_s=6.329`
  - inner prefill: `2073ms`
  - inner decode: `3160ms`
  - generation-scoped launches: `14481`, `724.050` launches/token
  - `launch_elementwise=1008`
  - `launch_norm_rms_add=2016`
  - `rms_norm_add_fused=2016`
- Post-RMSNorm+residual-fusion Q4_K sustained smoke, `--max-tokens 32`,
  stopped naturally after 21 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q4_K ant output
  - `decode_tok_per_s=4.987`
  - inner prefill: `2913ms`
  - inner decode: `4211ms`
  - generation-scoped launches: `15166`, `722.190` launches/token
  - `launch_elementwise=1056`
  - `launch_norm_rms_add=2112`
  - `rms_norm_add_fused=2112`
- Pre-512-GQA two-token attention-counter smokes passed:
  - Q8_0 output tokens: `14054 236751`
  - Q4_K output tokens: `14054 236751`
  - both reported `gqa_decode=80`, `gqa_scalar=16`, matching the 40
    sliding-attention / 8 full-attention layer split
- Post-512-GQA Q8_0 two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=13.514`
  - `cuda_generate_attention_launch_breakdown: gqa_decode=96 gqa_scalar=0`
- Post-512-GQA Q4_K two-token smoke passed:
  - output tokens: `14054 236751`
  - output text: `Ants`
  - `decode_tok_per_s=10.638`
  - `cuda_generate_attention_launch_breakdown: gqa_decode=96 gqa_scalar=0`
- Post-512-GQA Q8_0 sustained smoke, `--max-tokens 32`, stopped naturally
  after 20 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q8_0 ant output
  - `decode_tok_per_s=6.572`
  - inner prefill: `2078ms`
  - inner decode: `3043ms`
  - generation-scoped launches: `14481`, `724.050` launches/token
  - `cuda_generate_attention_launch_breakdown: gqa_decode=1008 gqa_scalar=0`
- Post-512-GQA Q4_K sustained smoke, `--max-tokens 32`, stopped naturally
  after 21 tokens:
  - output remains coherent
  - token ids unchanged from the previous coherent Q4_K ant output
  - `decode_tok_per_s=5.138`
  - inner prefill: `2897ms`
  - inner decode: `4087ms`
  - generation-scoped launches: `15166`, `722.190` launches/token
  - `cuda_generate_attention_launch_breakdown: gqa_decode=1056 gqa_scalar=0`
- `bench-cuda --warmup-iters 1 --measure-iters 3 --gemma4-shapes` passed on
  the L4. Initial Gemma4 synthetic row-1 timings:
  - Q8_0 tile4 wins on Q/KV/attention/FFN shapes, but is neutral/slightly
    slower on the materialized LM-head shape: `10.55ms` scalar vs `10.77ms`
    tile4.
  - Q4_K tile4 beats scalar and tile8 on all measured Gemma4 shapes.
  - Q4_K LM head: `53.94ms` scalar, `13.79ms` tile4, `14.46ms` tile8.
  - Q4_K FFN down: `5.27ms` scalar, `0.613ms` tile4, `0.674ms` tile8.

Still pending from the original Phase 0/1 acceptance:

- Run the full 12-prompt Q8_0 and Q4_K quality gates. The one-prompt smoke is
  healthy, and the two-prompt Q8_0 smoke is healthy, but the full run is still
  expensive because streamed BF16 reads/uploads dominate each prompt.

Phase 2 measurement summary:

| Dense budget | Generate ms | Prefill ms | Dense misses | Evictions | Read ms | H2D ms | Resident MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 MB | 123862 | 123855 | 344 | 299 | 114208 | 4908 | 1968 |
| 8192 MB | 123850 | 123850 | 344 | 202 | 114278 | 4879 | 8133 |
| 12000 MB | 123852 | 123851 | 344 | 142 | 114250 | 4855 | 11936 |
| auto | 123900 | 123898 | 344 | 142 | 114165 | 4925 | 11936 |

Conclusion:

- Larger dense-stream budgets reduce evictions and are useful for possible
  reuse, but they do not materially improve BF16 first-token time.
- First-token time is dominated by reading about 9.7 GB and uploading about
  20.8 GB of streamed dense weights. The next Phase 2 throughput work should
  target read/upload volume and overlap, not only cache size.

## Important Findings

- Q8_0 now has row-1 tiled matvec, fused QKV, and gate/up pair fusion. The
  sustained Q8_0 run shows `qkv_fallback_unsupported=0`, `qkv_fused_q8=840`,
  and `linear_pair_fused_q8=1008`.
- Q4_K fast decode is healthy, fused QKV is healthy, and gate/up pair fusion is
  healthy. The sustained smoke shows zero unexpected QKV/linear-pair/fast-decode
  fallback.
- `bench-cuda --gemma4-shapes` now covers the row-1 Gemma4 decode matvec
  shapes. On the initial L4 run, Q4_K tile4 is consistently faster than tile8
  for these shapes, so there is no current evidence to switch resident decode
  back to tile8. Q8_0 tile4 is the right default for most Gemma4 decode matvecs,
  but a materialized LM-head matvec is not faster than scalar in this synthetic
  microbench; generation uses the fused LM-head argmax path instead.
- Q8_0 launch count dropped from about `959` launches/token to `825`
  launches/token, but throughput only moved from `4.745` to `4.766 tok/s`.
  Q4_K launch count dropped from about `873` launches/token to `823`
  launches/token, while throughput stayed about `4.07 tok/s`.
- The remaining shared bottleneck is now clearly not only QKV/gate-up launch
  overhead. The hot buckets are still `norm`, `elementwise`, `scalar`, and
  synchronization.
- Device greedy argmax is active for Gemma pure-greedy decode, including the
  prefill first token and Gemma4 suppress-token masking. Sustained Q8_0/Q4_K
  runs now show only 4-byte token-id downloads.
- Quantized LM-head greedy argmax is now fused for Q8_0 and Q4_K, including
  Gemma4 suppress-token masking. The implementation avoids materializing the
  full device logits tensor, but it still uses a safe two-stage reduction, so
  it does not reduce launch count. Sustained throughput is effectively neutral:
  Q8_0 stayed about `4.76 tok/s`; Q4_K measured about `4.11 tok/s`.
- PLE activation-product and add-multiply-scalar fusion plumbing is now safer
  and observable, but the current Gemma4 Q8_0/Q4_K ant smokes do not exercise
  the add-multiply-scalar path. `activation_multiply_fused` counts the existing
  FFN path and launch counts are unchanged.
- Parallelizing the active plain RMSNorm kernels was the first major
  post-correctness throughput win without reducing launch count: Q8_0 moved
  from about `4.78` to `6.29 tok/s`, and Q4_K moved from about `4.09` to
  `5.04 tok/s`. This proves remaining kernel body work still matters, not just
  launch count.
- Parallelizing the active fused head-norm/RoPE kernel was correct but
  throughput-neutral on the sustained ant smoke: Q8_0 measured `6.311 tok/s`
  and Q4_K measured `5.010 tok/s`, with unchanged token ids and zero
  head-norm/RoPE fallbacks. The next pass should use the new norm breakdown
  counters before spending more effort on this specific kernel.
- The first norm breakdown smoke shows weighted RMSNorm launches dominate the
  remaining norm bucket (`482/674` norm launches on a Q8_0 two-token smoke).
  Bare V norms and fused head-norm/RoPE are each `96/674`. The next high-value
  path is graph-level residual/RMSNorm fusion or reducing weighted RMSNorm
  launch count, not more head-norm/RoPE kernel-body tuning.
- RMSNorm+residual fusion now removes the two main post-norm residual add
  launches per Gemma layer. Sustained launch counts dropped from about
  `824-823/token` to about `724-722/token`, and elementwise launches dropped by
  about two-thirds. Throughput moved only slightly: Q8_0 `6.311 -> 6.329 tok/s`
  and Q4_K `5.010 -> 4.987 tok/s`. This is useful launch cleanup, but not the
  route to `20+ tok/s` by itself.
- CUDA attention is not FlashAttention. The active path is custom GQA decode
  over the device KV cache. After adding 512-wide head support, all Gemma4
  resident Q8_0/Q4_K attention calls hit `termite_gqa_attention_decode_f32`;
  the scalar GQA fallback is now zero on both two-token and sustained ant
  smokes. The throughput effect is positive but modest: Q8_0 `6.329 -> 6.572
  tok/s`, Q4_K `4.987 -> 5.138 tok/s`.
- Sync-safe deferred reclamation is implemented, but the resident Q8_0/Q4_K
  smokes do not exercise it because the temp cache absorbs all temporary frees.
  Under `ANTFLY_INFERENCE_CUDA_TEMP_CACHE_MB=0`, it reduces generation syncs
  from `2110` to `2` on a two-token Q8_0 smoke.
- Generation-scoped stats show the real steady-state sync rate is about
  `1.05 syncs/token`; the old `32-34 syncs/token` figure was mostly load/setup
  synchronization. The remaining throughput problem is launch count and kernel
  work, not runtime stream synchronization. The next high-impact work remains
  an active Gemma4 norm/residual path: either CUDA `runAttentionResidual` /
  `runGatedDecoderBlock` support comparable to Metal/MLX, or direct graph-level
  Gemma4 post-norm/residual fusion. More LM-head or PLE cleanup will not move
  the current Q8_0/Q4_K ant benchmark.
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

Status:

- Implemented behind `bench-cuda --gemma4-shapes`.
- Added the listed shapes plus attention output projection shapes.
- Initial Q4_K measurements favor tile4 over tile8 for Gemma4 row-1 decode.

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

Status:

- Gemma pure-greedy device argmax is enabled and validated on Q8_0/Q4_K smoke
  prompts.
- Per-token downloads are now 4-byte token-id downloads, but one full-logit
  download remains in `generate`.
- Sync counts are still high: about 687-688 stream syncs on the sustained
  Q8_0/Q4_K smokes.

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
- Full-logit D2H disappears for pure greedy decode, including the remaining
  one-time full-logit download.
- Sync counts per token drop without reintroducing activation trace corruption.
- `compare --activation-trace` spot check still passes.

## Phase 4 - Quantized Matmul Work

Status:

- Q4_K row-1 tile8 fast decode now defaults on and validates with zero
  unexpected fallback on the sustained smoke.
- Q8_0 row-1 tile4 matvec is implemented and enabled with scalar fallback.
- Q8_0/Q4_K correctness smokes pass after the changes.
- `bench-cuda --gemma4-shapes` coverage for Q8_0/Q4_K Gemma4 row-1 decode
  shapes is implemented.

Q4_K:

- Run Q4_K with `ANTFLY_CUDA_ENABLE_Q4K_DECODE_FAST=1`.
- Compare tile8 vs tile4 in `bench-cuda` and end-to-end generation.
- If tile8 is correct and faster, make it the default for row-1 decode.
- Fix counters so flag-off is not reported as a fast-path fallback.

Current result:

- The new Gemma4 microbench favors tile4 over tile8 on every measured row-1
  shape, including LM head and FFN down.
- Keep tile4 as the preferred row-1 decode kernel unless a later real-weight
  or end-to-end profile contradicts the synthetic benchmark.

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
- Q8_0 tiled microbench shows a clear improvement over scalar Q8_0 on the
  active non-LM Gemma4 decode matvec shapes.
- Q8_0 and Q4_K pass Phase 1 smoke eval after changes.

## Phase 5 - Launch Reduction And Fusions

20+ tok/s likely requires reducing decode from roughly 870-960 launches/token
to a few hundred launches/token, while also improving quantized matvec
efficiency. Individual matvec speedups are useful, but not enough by themselves
while every layer still performs many small kernels.

Status:

- Phase 5A launch/sync-per-token measurement is implemented.
- Phase 5B Q8_0 fused QKV is implemented and validated.
- Phase 5C Q8_0/Q4_K gate-up `linearNoBiasPair` fusion is implemented and
  validated.
- Phase 5E full-logit transfer cleanup is implemented and validated:
  sustained Q8_0/Q4_K generate runs now show only 4-byte token-id downloads.
- Phase 5F deferred-free infrastructure is implemented and env-gated, but it
  does not change the resident Q8_0/Q4_K smokes because they have zero temp
  evictions.
- Phase 5D was audited. The existing CUDA RMSNorm+residual+scale kernel is not
  currently exercised by the Gemma4 graph; useful fusion now needs graph-level
  Gemma4 post-norm/residual/output-scale epilogue work.
- Phase 5E follow-up fused quantized LM-head argmax is implemented and
  validated for Q8_0/Q4_K. It removes full device-logit materialization at the
  LM-head selection point, but the safe implementation still uses two kernel
  launches, so it is not a major throughput lever by itself.
- Phase 5A-G reduced launch count and eliminated host-logit downloads. The
  parallel RMSNorm work then raised sustained throughput to about `5.0-6.3
  tok/s`; RMSNorm+residual and 512-wide GQA decode then raised it to about
  `5.1-6.6 tok/s`. Continue with broader block-level fusion and CUDA graph
  capture; the head-norm/RoPE probe shows not every active norm kernel is worth
  deeper optimization.

Implementation:

Phase 5A - Measurement harness:

- Add or extend `bench-cuda` cases for Gemma4 decode shapes:
  - QKV projections
  - output projection
  - FFN gate/up/down
  - tied LM head
- Add a concise launch-per-token summary to timing JSON or docs so sustained
  runs can be compared without manually dividing launch counts.
- Keep Q8_0 and Q4_K sustained ant smokes as the primary before/after checks.

Phase 5B - Q8_0 fused QKV:

- Mirror the existing Q4_K fused QKV path with Q8_0 weights.
- Dispatch through the same QKV hook that currently records
  `qkv_fallback_unsupported` for Q8_0.
- Expected direct effect: convert three Q/K/V matvec launches into one per
  layer/token for Q8_0, removing roughly 80-100 launches/token on Gemma4.
- Acceptance:
  - Q8_0 `qkv_fallback_unsupported` drops to zero for supported shapes.
  - Q8_0 `launch_linear_qkv` becomes nonzero.
  - Q8_0 token ids remain unchanged.

Phase 5C - Gate/up pair fusion:

- Add CUDA `linearNoBiasPair` coverage for quantized row-1 decode, starting with
  Q4_K and then Q8_0.
- Fuse FFN gate and up projections that share the same input.
- Optionally add a second-stage fused activation/product kernel if the existing
  graph cannot consume the pair output directly.
- Expected direct effect: remove one large FFN matvec launch per layer/token and
  reduce temp-buffer churn.
- Acceptance:
  - Pair-fusion counter shows hits on Gemma4 MLP layers.
  - Q8_0/Q4_K sustained token ids remain unchanged.

Phase 5D - Residual/RMSNorm/scale fusion:

- Audit the Gemma4 decode graph for repeated patterns:
  - residual add followed by RMSNorm
  - RMSNorm followed by scalar multiply
  - residual add plus post-attention/pre-FFN norm
- Add targeted fused CUDA kernels for the highest-frequency patterns.
- Keep the existing head-norm/RoPE fusion, which is already hitting.
- Current audit result:
  - existing CUDA `rmsNormAddMultiplyScalarTensor` is wired but not naturally
    reached by the Gemma4 decode call sites
  - do not just enable it globally; add a Gemma4 graph/helper path that passes
    post-norm input, residual, and output-scale together
- Expected direct effect: reduce the `norm`, `elementwise`, and `scalar`
  launch buckets, which currently account for hundreds of launches/token.
- Acceptance:
  - Launch buckets fall materially on both Q8_0 and Q4_K.
  - Activation trace spot checks remain clean.

Phase 5E - Remaining logits transfer cleanup:

- Find why pure-greedy `generate` still downloads one full logits tensor.
- Keep full logits for compare/diagnostics/sampling, but route normal greedy
  generation through the device-token path end to end.
- Status:
  - implemented for CUDA pure greedy, including Gemma4 suppress-token masking
  - Q8_0 sustained: `cuda_download_top_sizes: 21x4`
  - Q4_K sustained: `cuda_download_top_sizes: 22x4`
- Acceptance:
  - Sustained generate runs show only 4-byte token-id downloads.
  - Token ids/text remain unchanged.

Phase 5F - Sync-safe deferred reclamation:

- Replace per-free synchronization with event/deferred reclamation for uncached
  CUDA buffers.
- Drain the deferred list at natural sync points and on context teardown.
- Keep an env kill-switch until activation traces and sustained generation are
  stable.
- Status:
  - implemented with `ANTFLY_INFERENCE_CUDA_DEFER_FREE`
  - resident Q8_0/Q4_K smokes show `deferred_free_queued=0` because
    `temp_evictions=0`
  - next validation should use a constrained temp-cache run to force uncached
    frees and confirm sync reductions under pressure
- Acceptance:
  - Sync counts drop without reviving the freed-view corruption bug.
  - Q8_0/Q4_K quality eval still passes.

Phase 5G - Quantized LM-head argmax materialization cleanup:

- Avoid materializing the full `[1, vocab]` device logits tensor for greedy
  quantized LM-head selection.
- Status:
  - implemented for Q8_0 and Q4_K with a two-stage tile-winner plus reducer
  - Gemma4 suppress-token masking is handled inside the fused quantized path
  - Q8_0 sustained: `lm_head_argmax_fused_q8=21`, fallback `0`
  - Q4_K sustained: `lm_head_argmax_fused_q4=22`, fallback `0`
- Finding:
  - launch count is unchanged because the safe implementation still needs a
    final reducer kernel
  - throughput impact is neutral; keep this for memory/temp-pressure cleanup,
    but do not treat it as the route to `20+ tok/s`
- Acceptance:
  - fused hit count equals emitted token count for Q8_0/Q4_K greedy smokes
  - fallback count remains zero
  - token ids/text remain unchanged

Phase 5H - Active Gemma4 norm/residual fusion:

- CUDA currently does not implement the high-level `runAttentionResidual` hook
  used by Metal/MLX paths.
- Status:
  - plain RMSNorm kernels are now parallel and produced a real throughput win
  - fused head-norm/RoPE is now parallel but throughput-neutral on the ant
    smoke
  - weighted RMSNorm+residual epilogues now fuse through `rmsNormAddTensor`,
    reducing launches but producing only small/noisy sustained throughput gains
  - the remaining high-impact work likely needs either broader block-level graph
    fusion or attention/GQA kernel tuning, not just one more local epilogue
- Implementing CUDA `runAttentionResidual` for Gemma4 paged decode should fuse
  attention output projection, optional post-linear norm, and residual handling
  inside the active layer path instead of relying on separate graph-level calls.
- If the full hook is too broad, add a narrower Gemma4 helper around the active
  post-attention and post-FFN norm/residual sequence.
- Acceptance:
  - active Q8_0/Q4_K ant smokes show reduced `launch_norm`,
    `launch_elementwise`, or `launch_scalar`
  - token ids/text remain unchanged
  - `cuda_generate.launches_per_token` drops materially

Phase 5I - Active GQA attention coverage:

- Status:
  - CUDA Gemma4 uses custom GQA decode, not FlashAttention.
  - 256-wide sliding-attention heads and 512-wide full-attention heads now both
    use `termite_gqa_attention_decode_f32`.
  - Q8_0/Q4_K sustained smokes show `gqa_scalar=0` and no host attention
    fallback.
- Next possible attention work:
  - optimize the decode kernel body itself with better reductions/tile reuse
  - separate 512-wide full-attention tuning if profiling shows it dominates
  - defer Flash-style prefill attention until prefill becomes a priority
- Acceptance:
  - `cuda_generate_attention_launch_breakdown` reports scalar fallback zero on
    Q8_0 and Q4_K resident smokes.
  - token ids/text remain unchanged.

Phase 5J - Re-evaluate CUDA graph capture:

- Only after the above fusions reduce dynamic allocation and stabilize buffer
  addresses, prototype graph capture for steady-state single-token decode.
- This is likely necessary for the final jump toward 20+ tok/s if launch
  overhead remains dominant after kernel fusion.

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
