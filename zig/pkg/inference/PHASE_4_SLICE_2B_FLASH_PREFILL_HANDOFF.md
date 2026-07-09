# Phase 4 Slice 2b Hand-off: flash prefill attention as a tunable `--sweep` route

All paths relative to `zig/pkg/inference/`. Branch: `codex/quant-kernel-metal-compiler`. Zig 0.16.0.
Anchored against the Slice 2a working tree (decode-1x attention). Effort: **large, comparable to Slice 2a** (most complex kernel + new sweep-for-attention wiring). This is the route with a real (if modest) perf lever.

## Goal
Bring `termite_paged_attention_kv_prefill_sg` (`metal_kernels.m:5405-5469`) — the simdgroup-MMA flash prefill attention kernel — under the compiler as a second `op_kind=.attention` route (`AttentionKind.prefill_flash`), **tunable via `--sweep`**. Then sweep its schedule knobs, decode-runtime-A/B the winners, and promote if a variant beats the current default. Opt-in until promoted; hand-written stays default.

**Perf expectation — set it honestly.** The big prefill-attention win (contiguous direct-device K/V load) is *already shipped* (default-on in the hand-written kernel). The remaining sweep levers are **key-chunk 32→64** (halves barrier/gather count per chunk) and **rescale-skip** (skip the O-accumulator rescale when no row-max changed) — both flagged "smaller/optional" when deferred. Realistic upside: a few % of prefill *attention*, which is ≈ half of a 1k-token prefill → low-single-digit % total prefill. The value is as much "prove the compiler autotunes attention" as raw speed.

## Reuse everything Slice 2a built (the attention scaffolding is done)
- `quant_kernel_metal_renderer.zig`: `AttentionKind` enum (`:52`), self-contained `helper_paged_attention_1x_params`/`_page_token` (`:145/:150`), `renderAttention` (`:440`), `attentionHelpers` (`:468`), `validateAttentionSchedule` (`:474`), `renderAttentionBody` dispatch (`:495`), `RegionKernel.attention` field (`:567`) + the two-pass `.attention` arm in `renderRuntimeRegion`.
- `quant_kernel_compiler.zig`: `first_decode_attention_1x_metal_*` consts (`:2062+`), `first_generated_attention_artifacts` list, `renderMetalAttentionSource`, the `paged_attention_params_field_body` drift guard.
- `native_quant_kernel_codegen.zig`: the combined `.microkernel,.attention` arm + attention loops in the file/count/check paths.
- `metal_kernels.m`: `attention_1x_generated_pipeline` env-gated build + dispatch-swap pattern; the `termite_metal_run_generated_attention_check` FFI (sits after the params typedef `~2436`).
- The self-contained params-struct + `page_token` helpers **already handle the external-dependency crux** — the flash kernel uses the same struct + `termite_attention_page_token`, so it emits the same helpers. No new self-containment work.

## Implementation
### 1. Schedule: add the flash knobs
The current attention schedule reuses `KernelSchedule` (threads/cols/reduction). Flash needs more. Add flash-specific fields — cleanest is an `AttentionSchedule` for the attention op-kind (or extend `KernelSchedule` with defaulted `key_chunk: u16 = 32`, `skip_rescale: bool = false`; keep matmul untouched). Sweep space: `key_chunk ∈ {32, 64}` × `skip_rescale ∈ {false, true}` (4 variants). **Do NOT sweep SG-count / query-tile / gather-vs-direct** in this slice: query-tile=8 and 4-SG are structural, and gather-vs-direct is a runtime `fast` branch (direct always wins when contiguous), not a schedule knob.

### 2. Renderer: `renderPrefillFlashBody`
Add `prefill_flash` to `AttentionKind`; extend `attentionHelpers` (same two helpers), `validateAttentionSchedule` (flash: hd%32==0, threads=128/4-SG fixed), and the `renderAttentionBody` dispatch. Transcribe `metal_kernels.m:5405-5469` **verbatim as the baseline** (key_chunk=32, skip_rescale=false) — the body is byte-identical to the hand-written kernel, so the baseline variant must reproduce it exactly (diff-check like Slice 2a did). Then parametrize:
- **key_chunk**: replace the literal `32u` in the `kc += 32u` loop, the `sphys` populate (`tid < 32u`), the K/V gather bounds (`i < 32u * hd`), the `sp[j * 32u + kk]`/`ss[j * 32u]` strides, the `kk8 < 4u` P·V inner loop (= key_chunk/8), and **the shmem layout** with `key_chunk`. ⚠️ **SEV-HIGH: the `fb` offsets (`1024,1536,1664,1696,1760` at `:5413-5418`) and `skv = sq + 8*hd` / `fb = shmem + (8+32)*hd*2` are chunk-32-specific.** For chunk-64: `skv` doubles (64×hd), `ss`/`sp`/`sphys` sizes double, and every `fb` offset shifts. Recompute them symbolically from `key_chunk`, and update the host `setThreadgroupMemoryLength` at the dispatch site to match. Get this wrong and you get silent corruption.
- **skip_rescale**: guard the O-rescale (`:5453-5454`, `simdgroup_load(mcorr,sdiag)` + the `d_tiles` `simdgroup_multiply`) behind "did any row's max change this chunk?" — track a threadgroup flag set when `m_new > m_old`; skip the rescale (identity multiply) when unset. First chunk (`m_old == -inf → corr == 0`) must NOT be treated as unchanged.

### 3. Compiler + codegen + dispatch (mirror Slice 2a)
`first_prefill_flash_metal_*` consts + a `first_generated_attention_artifacts` entry (op_kind=.attention, prefill_flash kind, the chosen schedule); `renderMetalAttentionSource` handles `prefill_flash`; RegionKernel appended; codegen file/count/check loops already iterate the attention list. Dispatch: an env-gated `attention_flash_generated_pipeline` (`TERMITE_METAL_ENABLE_FLASH_PREFILL_GENERATED`), swapped in where `attention_paged_prefill_sg_pipeline` dispatches (that path is *already default-on* from the direct-load work — so the swap replaces the hand-written flash with the generated one; same buffers/grid((q_len+7)/8, heads)/128-threads, but **threadgroup-memory length must track key_chunk**).

### 4. ⭐ New: wire `--sweep` for attention (the point of this slice)
Today `runSweep` (`quant_kernel_metal_runtime_check.zig:2161`) iterates `metal_production_schedules` (matmul only) and `enumerateSweepVariants` enumerates threads/cols/reduction. Add an attention sweep path:
- `enumerateAttentionSweepVariants(kind) → [{key_chunk, skip_rescale}]` (4 flash variants).
- Extend `runSweep`/`runSweepRoute` to also iterate the attention routes and render each variant via `renderAttention`, benchmarking on-device against the current production flash kernel (reuse the existing standalone A/B + repeat-runs + `unstable_benchmark_timing` machinery).
- Emit the same `antfly.quant_kernel_metal_sweep.v1` evidence.
- **Sweep is a directional filter only** (documented caveat — it under-reports); the real arbiter is a decode-runtime / prefill A/B on E4B/E2B.

### 5. Conformance + correctness gate
- Isolated: extend the Slice 2a `referenceGqaAttention1x` harness to a multi-query prefill reference (or add `referenceFlashPrefill`) for a sanity float check. Flash summation order differs from the decode reference, so tolerance is loose — treat as compile/dispatch sanity, not the gate.
- **Real gate = bit-identical (or coherent) model tokens.** The baseline variant (chunk-32, rescale-on) is byte-identical to the hand-written kernel → must be bit-identical tokens (`scripts/compare_metal_gemma4_e4b_qat.sh`, or the `--print-token-ids` A/B I ran for Slice 2a). The **chunk-64 / rescale-skip variants change summation order** → tokens may shift at near-ties; validate they stay *coherent* (oracle text) and A/B the answer, exactly like the earlier flash-kernel work. Only promote a variant that (a) is coherent and (b) wins the prefill decode-runtime A/B beyond noise.

## Verification
```
zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dcuda=false
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false -- --write   # + flash .metal
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false              # --check + --check-metal (flash compiles standalone; shmem math must be valid MSL)
zig build test -Dmetal=true -Dcuda=false -- --test-filter "quant kernel"   # + flash render/schedule tests
zig build test -Dmetal=true -Dcuda=false                                   # full suite 0 fail; matmul 22/22 unaffected (separate list)
# baseline byte-identity: generated prefill_flash (chunk32/rescale-on) diff-identical to hand-written termite_paged_attention_kv_prefill_sg
# sweep: zig build quant-kernel-metal-sweep ... (now covers the attention route) → 4 flash variants ranked
# on-device: TERMITE_METAL_ENABLE_FLASH_PREFILL_GENERATED=1 + a long-prompt prefill A/B (E4B ~1k tok) per variant; tokens coherent; promote a winner only if it beats default beyond noise
```

## Risks (ranked)
1. **SEV-HIGH — chunk-64 shmem re-layout.** The `fb` offsets + `skv`/`ss`/`sp`/`sphys` sizes are chunk-32-specific; chunk-64 must recompute every offset symbolically AND bump the host `setThreadgroupMemoryLength`. A wrong offset = silent wrong tokens. Land chunk-32 baseline (byte-identical) first, prove it, then add chunk-64 as a separate step.
2. **SEV-HIGH — baseline byte-identity.** The chunk-32/rescale-on rendered body must diff-identical to the hand-written kernel (modulo the self-contained helper renames), so the "generated == hand-written" token gate holds. Same discipline as Slice 2a.
3. **SEV-MED — sweep-for-attention wiring** is new; keep the matmul sweep path untouched.
4. **SEV-MED — variant token drift.** chunk-64/rescale-skip change summation order → gate on coherence + answer A/B, not raw floats.
5. **SEV-LOW — modest payoff.** Direct-load (the big win) is already in; set expectations at low-single-digit % prefill.

## After this
The attention family is then fully expressible + tunable. Remaining optional routes: paged/quantized-KV read, fused head-rope/KV microkernels. And the two out-of-plan tracks: CUDA parity (multi-backend) and speculative decode (orthogonal perf). See `PHASE_4_ATTENTION_MICROKERNELS_HANDOFF.md`, `[[gemma4-prefill-flash-attention]]`.
