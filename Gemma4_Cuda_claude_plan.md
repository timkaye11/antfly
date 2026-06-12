# Gemma4 CUDA Production-Readiness Plan

Last revised: 2026-06-10

## Context

Branch `gemma4_gpu_stuff` targets running Gemma4 12B-it on CUDA, especially on an NVIDIA L4 with about 23 GB visible VRAM. The production target for L4-class hardware is a resident quantized model, not full BF16.

Current state from `Gemma_Cuda_status.md` and code review:

- **BF16 safetensors on CUDA is the correctness control.** It produces plausible first tokens, but it streams/offloads because the 12B BF16 weights do not fit on the L4.
- **Q8_0 and Q4_K GGUF CUDA generation still produces bad first tokens.**
- **CUDA math matches the quantized artifacts tightly.** Existing primitive and real-tensor parity checks point away from a simple CUDA kernel mismatch.
- **The quantized artifacts drift from HF BF16 early.** Q8_0 drift is moderate after layer 0; Q4_K drift is much larger.
- **The `layer0dense` Q8_0 artifact has been exported.** It reduces layer0 drift, but still needs generation/snapshot runs.
- **Disk is tight.** Current free space is about 7.1 GB, so external/reference GGUF experiments and dense GGUF export require cleanup first.

Important runtime constraint: the CUDA linear path currently supports quantized `.Q8_0`, `.Q4_0`, and `.Q4_K` matmuls. The exporter can write and the CPU path can dequantize other K formats such as Q5_K/Q6_K, but a production CUDA profile must not emit Q5_K/Q6_K tensors unless we also add CUDA kernels or force those tensors through dense/lazy-dequant paths.

## Implementation Guardrails

- Do not start Q4_K work until Q8_0 has a named root cause or a known-good Q8 profile.
- Do not start CUDA performance work until a resident or hybrid Q8 profile produces coherent deterministic output.
- Do not delete the layer0-dense artifact until its CUDA and CPU/native diagnostics have been captured.
- Capture all Phase 0 outputs into timestamped logs under `/tmp` or another scratch path before freeing model artifacts.
- Prefer diagnostics that reuse existing artifacts before exporting another 12 GB file.
- Treat exact first-token equality as a bonus signal, not a required pass/fail condition.

## Key Findings

### 1. Q8_0 needs a stricter diagnostic split before any quantizer work

Q8_0 should normally be good enough for first-token quality. In this repo:

- Q8_0 CUDA parity against CPU for the Q8 artifact is tight.
- Projection-only lazy dequant still produced a bad first token.
- Layer0-dense Q8 reduces layer0 drift but does not make it zero.
- `session_factory.zig` currently has a dirty diagnostic change that can also lazy-dequant quantized embeddings via `ANTFLY_INFERENCE_CUDA_LAZY_GGUF_QUANT_EMBEDDING=1` or `TERMITE_CUDA_DEQUANTIZE_QUANT_EMBEDDING=1`.

That means the next Q8 work should isolate artifact/runtime/export-layout/profile sensitivity, especially `token_embd.weight` and the tied LM head, before changing Q4_K.

### 2. The Q4_K quantizer has real quality problems

`zig/pkg/inference/src/gguf/quant_codec.zig` `quantizeQ4_KBlock` quantizes payload values using the original floating sub-scales/mins after it has rounded those scales/mins into packed 6-bit metadata. Dequantization reconstructs using the rounded metadata, so the encoded payload is optimized for slightly different values than the decoder will use.

The Q4_K implementation is also a single-pass min/max quantizer. ggml's reference path uses an iterative scale/min search (`make_qkx2_quants`-style) that materially improves 4-bit quality.

These Q4_K fixes are valid, but they should not be treated as the first Q8_0 fix. Q8_0 has a separate failure mode that must be understood first.

### 3. llama.cpp's mixed profiles are a useful reference, but not directly CUDA-compatible here

llama.cpp treats tied embeddings/output, attention-v-like tensors, and selected FFN-down tensors specially for mixed K quantization. For `Q4_K_M`, that can include Q6_K upgrades for sensitive tensors.

In this codebase, a Gemma4 profile must be CUDA-aware:

- Prefer Q8_0 or dense/BF16 for sensitive tensors when base format is Q4_K.
- Do not emit Q5_K/Q6_K for tensors that must run through CUDA matmul until CUDA support exists.
- Keep the profile as named exporter behavior, not a hand-maintained `--quantize-exclude` string.

### 4. CUDA performance work is important, but should wait until correctness is pinned

Confirmed performance gaps:

| Gap | Current state | Impact |
|---|---|---|
| Q8_0 linear is scalar, one thread per output | `termite_linear_q8_0_f32` | Q8_0 decode hot path is slow |
| Q8_0 has no tiled/fused variants | Q4_K has multiple tile/fusion paths | Q8 misses QKV and bias/activation fusion |
| Device argmax exists but routing is limited | eager decode still downloads logits | avoid 262k-vocab download for plain greedy |
| Device KV path can fall back to host | hook is f32-only and needs counter-driven diagnosis | host round-trips per token |
| Many kernel launches/syncs | single-stream eager path | launch overhead at batch-1 decode |
| Prefill attention is not flash-style | scalar dense/GQA kernels | long-prompt latency |

## Phase 0 - Q8_0 Correctness Bisection

Goal: identify whether Q8 failure is owned by runtime, exporter/layout, quantization profile, or embedding/output sensitivity. No kernel rewrites in this phase.

### Immediate Diagnostic Runbook

Run these in order before changing code:

1. **Record baseline state.**
   - `df -h /home/timkaye/tim/antfly`
   - `ls -lh .models/google/*.gguf`
   - `git status --short`

2. **Run layer0-dense Q8 on CUDA.**
   - Use `.models/google/gemma-4-12B-it-q8_0-layer0dense/`.
   - Use both status-doc prompts.
   - Enable first-token trace/top-k logging.

3. **Run normal Q8 and layer0-dense Q8 on native CPU.**
   - If native CPU has the same bad token behavior, keep focusing on artifact/export/profile.
   - If native CPU is good and CUDA is bad, reopen runtime investigation.

4. **Run lazy-dequant Q8 diagnostics.**
   - Projection lazy-dequant only, as already tested, for reproducibility.
   - Projection plus embedding lazy-dequant using `ANTFLY_INFERENCE_CUDA_LAZY_GGUF_QUANT_EMBEDDING=1`.
   - Compare first-token top-k and special-token behavior, not only the chosen token.

5. **Only after those logs are captured, free disk for external/dense GGUF experiments.**

### Phase 0 Implementation Tasks

1. **Upgrade `cuda_info.zig` diagnostics.**
   - Extend `--gemma4-cross-layer0` to print baseline magnitudes and relative error, not just absolute `max_abs`/`mean_abs`.
   - Add a dense BF16 GGUF-vs-HF layer0 path to isolate export/name/layout drift from quantization drift.
   - Add a weight-space RMSE sweep over all tensors, grouped by tensor category.
   - Add first-token top-k comparison: BF16 rank of Q8 top token, top-k overlap, top logit margins, and special-token flags.

2. **Cross-validate with llama.cpp after freeing disk.**
   - Run a llama.cpp-produced Q8_0 GGUF of the same checkpoint through our engine on CPU and CUDA.
   - Run our Q8_0 GGUF through llama.cpp.
   - The local sidecar directories are symlink wrappers: `model.gguf` points to a sibling `.gguf`, while config/tokenizer files point to the BF16 model dir. For an external GGUF, create the same wrapper layout and point `model.gguf` at the external file.

3. **Dense GGUF end-to-end test only if ambiguity remains.**
   - Export `--format none` when disk allows.
   - This validates GGUF export/name-mapping/load independent of quantization.

Exit criteria:

- We know whether the Q8_0 failure is due to runtime, exporter/layout, quantization/profile, or embedding/output sensitivity.
- We have at least one named tensor category or stage explaining the failure well enough to implement a targeted fix.

### Phase 0 Decision Table

| Result | Interpretation | Next action |
|---|---|---|
| CPU/native Q8 is bad and CUDA Q8 matches it | Artifact/export/profile problem | Continue with exporter/profile diagnostics |
| CPU/native Q8 is good but CUDA Q8 is bad | Runtime problem despite parity tests | Add targeted CUDA/runtime parity around the divergent stage |
| Layer0-dense fixes early snapshots and improves first token | Early layer quantization sensitivity | Profile early dense/Q8 exceptions |
| Layer0-dense improves layer0 drift but first token stays bad | Later-layer accumulation or embedding/output sensitivity | Run embedding lazy-dequant and full-layer/tensor RMSE sweep |
| Projection plus embedding lazy-dequant fixes Q8 | Embedding/tied-head quantization is a key lever | Codify embedding/output exception in Q8 profile |
| llama.cpp Q8 works in our runtime but our Q8 fails in llama.cpp | Our exporter/quantizer/profile is bad | Fix exporter/profile before kernels |
| llama.cpp Q8 fails in our runtime but works in llama.cpp | Our GGUF loader/runtime path is bad | Focus on metadata/layout/runtime compatibility |

## Phase 1 - Make Q8_0 Healthy On CUDA

Goal: produce a Q8-based Gemma4 artifact that gives coherent deterministic output and fits L4-class GPUs.

1. **Codify a Gemma4 Q8 CUDA profile in `native_export_gguf.zig`.**
   - Start from evidence gathered in Phase 0.
   - If embedding lazy-dequant fixes Q8, keep `token_embd.weight`/tied output dense BF16 in the first profile.
   - If layer0-dense materially fixes Q8 behavior, keep layer0 dense in the first profile and then shrink the exception set later.
   - If neither helps, do not invent a profile; continue Phase 0 bisection with llama.cpp/dense GGUF.
   - Keep the profile CUDA-compatible. Do not emit unsupported CUDA quant types.

2. **Re-export using the named profile.**
   - Avoid manual include/exclude strings except for temporary experiments.
   - Verify the sidecar wrapper points at the intended GGUF.

3. **Validate CPU and CUDA behavior.**
   - BF16 top-k comparison, not just exact token equality.
   - First token should not be special/blank/garbage.
   - Deterministic temperature-0 outputs should be coherent on the two status-doc prompts.
   - Relative drift should be bounded enough to explain generation behavior.

4. **Only then treat Q8_0 as the production correctness baseline.**

Exit criteria:

- Q8 profile runs on CPU and CUDA with matching behavior.
- Outputs are coherent at temperature 0.
- CUDA primitive parity remains green.
- Artifact fits within the L4 budget with KV/scratch headroom.
- The profile's exception list is documented with evidence from Phase 0, not guesswork.

## Phase 2 - Fix Q4_K Quality

Goal: make Q4_K viable without breaking CUDA compatibility.

1. **Fix Q4_K payload quantization against reconstructed scale/min values.**
   - After packing rounded 6-bit `scs`/`mins`, quantize payload nibbles using `d * scs[sub]` and `dmin * mins[sub]`, matching what dequantization will reconstruct.

2. **Port an iterative `make_qkx2_quants`-style scale/min search.**
   - Keep this local and well-tested.
   - Add bounded-error tests rather than relying only on generation quality.

3. **Add golden/reference tests.**
   - Fixed input blocks for Q4_K and Q8_0.
   - Round-trip RMSE checks.
   - If using a ggml/llama.cpp reference fixture, store only small deterministic vectors, not large artifacts.

4. **Add a Gemma4 Q4 CUDA-compatible profile.**
   - Base tensors: Q4_K where CUDA Q4_K kernels support the tensor shape.
   - Sensitive tensors: Q8_0 or dense/BF16, not Q5_K/Q6_K unless those CUDA kernels are added.
   - Candidate sensitive categories: tied embedding/output, `attn_v`, selected `ffn_down`, and any stage identified by Phase 0/1 diagnostics.
   - If a llama.cpp-like Q4_K_M comparison is desired, treat it as a reference artifact, not as the exact export recipe for CUDA.

5. **Re-export and validate.**
   - Compare against the healthy Q8 profile and BF16 control.
   - Require coherent text, not exact BF16 token equality.

Exit criteria:

- Q4_K no longer has severe layer0 drift.
- CUDA and CPU agree on Q4 behavior.
- Q4 output is coherent under deterministic generation.

## Phase 3 - CUDA Decode Performance

Goal: improve decode speed after a known-good resident Q8 profile exists.

Ordered by expected wall-clock return:

1. **Add a benchmark harness first.**
   - Tokens/sec.
   - Per-phase timing.
   - CUDA counters: linear kernel counts, syncs, KV device successes/failures, host fallback counts.
   - Use the same prompt/model/profile before and after each kernel change.
   - Record model profile, prompt, token count, budgets, GPU name, and driver version in the benchmark output.

2. **Tiled/fused Q8_0 matmul family.**
   - Port the existing Q4_K tiled/fused approach to Q8_0.
   - Include QKV no-bias fusion and bias/activation/residual variants that the graph actually uses.
   - Include the LM-head path used by the selected Q8 profile.

3. **Device-side greedy routing.**
   - Reuse existing argmax kernels where safe.
   - Plain greedy can avoid applying Gemma final-logit softcap before argmax because the softcap is monotonic.
   - Fall back to host logits when grammar, suppression, repetition penalties, or sampling require full mutable logits, unless equivalent device kernels are added.

4. **KV cache fallback elimination.**
   - First measure `device_kv_fail_*` counters.
   - Keep the f32 device path healthy.
   - Extend dtype support only if the selected profile or future KV policy requires it.

5. **Reduce syncs and launches.**
   - Audit `ctx.synchronize()` and download syncs.
   - Make token-id download the only required per-token host transfer for plain greedy.
   - Consider CUDA graph capture for the steady-state decode path after kernels stabilize.

6. **Int8 dot-product path.**
   - Add activation quantization to Q8_1-style blocks and use `__dp4a` where it is a clear win.
   - Do this after tiled/fused Q8_0 because it changes numerical behavior and dispatch more deeply.

7. **Prefill attention optimization.**
   - Treat flash-style/paged attention as a later pass after decode is healthy.

Every `.cu` change requires regenerating `inference_cuda_kernels.ptx` and adding or extending `cuda-info --smoke` parity coverage.

## Phase 4 - Productize

1. Document the known-good Gemma4 CUDA profiles and L4 budget recipes.
2. Promote useful diagnostic env gates to documented flags.
3. Add CI smoke coverage:
   - quantize small fixture,
   - load GGUF,
   - generate/assert sane deterministic token/top-k behavior,
   - CUDA smoke for new kernels when CUDA is available.
4. Commit the branch's diagnostic/runtime changes only after they are either productized or intentionally removed.

## Immediate Artifact Policy

Keep until captured:

- `.models/google/gemma-4-12B-it-q8_0-layer0dense.gguf`
- `.models/google/gemma-4-12B-it-q8_0-layer0dense/`
- `.models/google/mmproj-q8_0-layer0dense.gguf`

Potential cleanup after Phase 0 logs are captured:

- Q4_K artifacts, because Q4_K work is explicitly deferred until Q8 is healthy.
- Projector variants not needed for text-only generation diagnostics.

Do not remove the BF16 safetensors model; it is the correctness control.

## Files To Modify By Phase

- **P0 diagnostics:** `zig/pkg/inference/src/cuda_info.zig`; optionally generation debug output only if needed.
- **P1 Q8 profile:** `zig/pkg/inference/src/native_export_gguf.zig`, exporter tests, sidecar/profile docs.
- **P2 Q4_K quality:** `zig/pkg/inference/src/gguf/quant_codec.zig`, quantizer tests, `native_export_gguf.zig` for Q4 profile.
- **P3 performance:** `zig/pkg/inference/src/ops/cuda/artifacts/inference_cuda_kernels.cu`, regenerated `.ptx`, `kernels.zig`, `cuda_compute.zig`, `src/pipelines/generation.zig`, and KV runtime files only if counters show they are on the hot path.
- **P4 productization:** docs and CI/smoke tests.

## Verification

Correctness:

- BF16 CUDA control remains plausible.
- CPU and CUDA agree for each quantized artifact.
- `cuda-info --smoke` passes.
- `--gemma4-cross-layer0` reports both absolute and relative drift.
- First-token top-k comparison shows Q8/Q4 top candidates are in the BF16 neighborhood.
- Deterministic generation is coherent and does not choose blank/special garbage tokens.

Performance:

- Record tokens/sec before each CUDA kernel change.
- Require temperature-0 output equivalence against the pre-change healthy profile unless the change intentionally alters numerical behavior.
- Track host transfers and sync counts, especially logits download and KV fallback counters.

## Risks And Constraints

- **Disk:** about 7.1 GB free currently blocks external Q8 GGUF and dense GGUF experiments until cleanup.
- **BF16 speed on L4:** not achievable as a production target because the full BF16 model exceeds usable VRAM.
- **Q8 root cause is still unproven:** do not skip Phase 0.
- **Exact token equality is too strict:** use top-k/rank/drift/coherence criteria.
- **Q5_K/Q6_K CUDA gap:** llama.cpp-style mixed profiles must be adapted to formats this CUDA runtime can actually execute.
