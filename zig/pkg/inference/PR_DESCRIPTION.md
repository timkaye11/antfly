# Metal quant-kernel compiler → generated-kernel framework, prefill flash attention, and speculative-decode fixes

## Summary

This branch grows the Metal quant kernel **registry** into a descriptor-driven, evidence-gated **kernel compiler/generator**, lands a real prefill-attention performance win, and fixes long-standing speculative/MTP decode bugs. Three arcs, each independently validated with on-device gates:

1. **Quant kernel compiler → generator (industry-grade arc).** Kernels are now *rendered* from descriptors + schedules instead of stored as frozen text; schedules are autotunable; three op-kinds (small-batch matmul, microkernel, attention) are generated, single-sourced, and evidence-gated.
2. **Prefill flash attention.** The simdgroup-MMA flash prefill kernel gains a contiguous direct-device K/V load and flips default-on: **+12.7% prefill at 1k tokens**, bit-identical output.
3. **Speculative/MTP decode fixes.** A use-after-free that crashed `k=2` and the bonus-token path (one root cause) is fixed; verify argmax is batched (one sync instead of k+1 frame teardowns). Spec decode goes **8.4 → 12.7 tok/s** with two previously-crashing configurations now working and bit-identical.

## Arc 1 — Quant kernel compiler

**Generator (Phases 0–5).** `KernelSchedule` (threads/cols/reduction) + `metal_production_schedules` become the single source for dispatch, launch-shape table (codegen-owned region in `metal_kernels.m`), and kernel bodies. A descriptor-driven renderer (`quant_kernel_metal_renderer.zig`) builds each kernel from a canonical skeleton + per-format `FormatDecoder` dequant fragments over a shared `antfly_qk_*` vocabulary. A `--sweep` autotuner benchmarks schedule variants on-device (directional filter; decode-runtime A/B is the promotion arbiter).

**Promoted route wins** (all decode-runtime-validated, bit-exact, 22-case production-regression gate): q2_k/none ~2×, q4_k/bias ~1.7×, q6_k/none ~2.5×, q6_k/bias ~2×, q8_0/none ~1.35× re-tunes; q4_k/none ~3.4×, q5_k/none ~3-4.5×, q8_k/none ~2.4× candidate promotions. 11 promoted Metal routes total.

**Full single-sourcing.** The 55 frozen `metal_rt_body_*`/helper constants are deleted; both the `metal_kernels.m` runtime region and the checked-in `.metal` files render from the same renderer (comptime, so source fingerprints/evidence stay comptime-derived). Re-tuning a route = edit the schedule table + `--write`; `--check` byte-verifies everything.

**Op-kind framework.** `OpKind { small_batch_matmul, microkernel, attention }` on `GeneratedArtifact`; non-matmul routes live in separate artifact lists so the matmul machinery stays byte-identical. Landed routes: **RMSNorm microkernel** (first non-matmul route, on-device conformance exact), **decode-1x paged attention** (generated body byte-identical to hand-written; model tokens bit-identical), and **flash prefill attention as a tunable route** (key_chunk/skip_rescale schedule knobs, sweep-for-attention wiring, KC=32 baseline byte-identical). All opt-in behind env gates; hand-written kernels remain the production defaults until promotion.

**CUDA.** 5 promoted generated Q4_0 routes (mmv/mm/pair/pair_activation_q8/gated_down) with sequential benchmark evidence on L4; e2e −4.5% total on Gemma4 E2B QAT, bit-identical output. CUDA source remains hand-written templates (renderer is Metal-only; CUDA renderer is a documented extension point).

**Explicit non-goal:** general/large dense GEMM stays on vendor libraries (MPS/Accelerate/cuBLASLt); generation targets exploding route matrices, not library GEMM.

## Arc 2 — Prefill flash attention (default-on)

`termite_paged_attention_kv_prefill_sg` was correct but opt-in (slower than the scalar path) because it gathered every 32-key chunk of K and V into threadgroup memory. For contiguous full chunks (the common prefill case) it now `simdgroup_load`s K (transposed) and V **directly from device memory**, skipping both gathers. Bit-identical tokens (scalar == gather == direct verified); prefill at 1k tokens flips from 7.5% slower to **12.7% faster** than the scalar path; default-on (`TERMITE_METAL_DISABLE_PREFILL_SG_ATTENTION` to opt out, `TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD` for the fast path alone).

## Arc 3 — Speculative/MTP decode: from broken-and-3×-slower to plain-decode parity

**Headline: spec decode (k=1 + bonus + deferred-materialize) went 8.4 → 25.6-27.2 tok/s vs plain 24.9-27.1 (cooled interleaved A/B) — ~3×, at parity with plain — tokens bit-identical to plain decode in every configuration throughout.**

Three real bugs fixed:
- **Use-after-free** (`termite_metal_decode_runtime_copy_grown_buffer` + `termite_copy_bytes` kernel): span-buffer growth opened a blit encoder while the frame's planned compute encoder was open — the deterministic `--speculative-k 2` crash AND the long-standing `ANTFLY_GEMMA4_MTP_ACCEPT_BONUS` segfault, one root cause.
- **Shared env-flag cache**: `getenvBool`/`getenvFlagValue` in metal_compute/metal_runtime cached ONE value across all env names — every `TERMITE_METAL_*` toggle read through them was order-dependent (pre-existing).
- **Dedicated-runtime double-argmax**: the verify commit recomputed logits through a dense fallback every no-cached-choice round (55-81ms each).

Performance work (each step identity-gated):
- **Batched multi-row verify argmax**: one sync for k+1 rows (was one frame teardown per row); on-device unit test.
- **Deferred-materialization fold** (`ANTFLY_GEMMA4_MTP_DEFER_MATERIALIZE`): pending-token state with KV-invariant asserts + five deterministic round-shape tests; landed default-OFF on measured economics, flipped to the winning config once verify costs fell.
- **Verify frame at rows 2-8**: threadgroup-reduce RMS norm (was a 2-thread serial kernel, ~28ms/frame), q4_0 pair-activation shared-read small-batch kernels, prepared lm_head tail + 2-row shared-read q6_k mmv (pairs of rows share one pass over the ~550MB head). Verify @2 rows: 139 → ~37ms.
- **Draft step 18 → ~5-6ms**: donated slot attention encoded on the draft frame via a new cross-runtime variant (was: host-download of donor KV + full frame flush + host-shim attention ≈5 GPU round trips/step); masked-embedding argmax default-off (acceptance-neutral, thrice-validated).
- **MTP seed re-forward eliminated**: Metal now captures the prefill's last hidden (was CUDA-gated; seeding re-ran the whole prompt through the generic forward, scaling with prompt length).
- **Bonus default-on for Metal**; kill switches on every new route.
- **Fast identity hard-gate**: `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1` (plain-golden + MTP identity, ~4 min; previously no fast mode ran them) + `scripts/test_gemma4_mtp_defer_materialize.sh` (six-arm fold/bonus sweep).

Follow-ups (scoped in-tree hand-offs + memory, not in this PR): fold the verify-tail lm_head apply into the verify frame (~13% of decode wall), fold-acceptance recovery (488 vs ~640‰), auto-policy retune, PLE row-stride hoist, tail r2 occupancy. ⚠️ Flagged latent issue: the **CUDA** prefill hidden-capture passes a *pre-norm* row where Metal captures *post-final-norm* — verify on a CUDA machine before trusting CUDA MTP seeding.

## Correctness gates (all green at HEAD)

- Full test suite: 2083 passed / 0 failed / 12 skipped (`zig build test -Dmetal=true -Dcuda=false`).
- `quant-kernel-codegen -- --check` byte-sync (region + 38 generated files + manifests); `--check-metal` xcrun-compiles 28 standalone artifacts.
- `quant-kernel-metal-production-regression-check` 22/22 promotion-ready, zero blockers; blocker-strict clean.
- Gemma4 model-token gates: plain golden EXACT; MTP identity (auto == force == force-k1 == target) EXACT; generated attention/flash routes bit-identical flag-on vs flag-off.
- On-device conformance: matmul routes max_abs_error ~1e-6; RMSNorm exact-0 to 3.3e-6; decode-1x attention exact-0; flash ~4e-5.

## Behavior changes & new flags

- **Env-gate semantics**: `TERMITE_METAL_*`/`ANTFLY_METAL_*` boolean gates now parse values — `FLAG=0/false/no/off` is falsy (previously any set value enabled). Audit any scripts relying on `FLAG=0` to enable.
- Flash prefill attention is **default-on** (was opt-in).
- Generated kernel routes are opt-in unless promoted; promoted routes are default-on with `TERMITE_METAL_DISABLE_ANTFLY_*` opt-outs.
- `ANTFLY_GEMMA4_MTP_ACCEPT_BONUS` no longer crashes on Metal (still default-off there; enabling is a measured follow-up).
- Notable new dev flags: `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY`, `TERMITE_METAL_ENABLE_RMS_NORM_GENERATED`, `TERMITE_METAL_ENABLE_ATTENTION_1X_GENERATED`, `TERMITE_METAL_ENABLE_FLASH_PREFILL_GENERATED`, `TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD`.

## Remaining / follow-ups (tracked, not in this PR)

- Spec-decode P2 (fold materialization into verify) — in flight; hand-off in-tree.
- CUDA renderer parity (multi-backend single-sourcing) — other machine; Linux/CUDA CI build still needed before merge.
- Autotune-loop automation; attention-family extension routes (paged/quantized-KV read, head-rope/KV microkernels).

## Docs

`QUANT_KERNEL_COMPILER.md` documents the op-kind framework, single-sourcing, sweep/promotion workflow, worked examples (add a format; add a microkernel route), and the explicit GEMM non-goal. Hand-off docs for in-flight work live at repo root of `zig/pkg/inference/`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
