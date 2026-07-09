# Phase 4 Slice 2a Hand-off: decode-1x paged attention as the first `op_kind=.attention` route

All paths relative to `zig/pkg/inference/`. Branch: `codex/quant-kernel-metal-compiler`. Zig 0.16.0.
Verified against HEAD `87aa871cc`. Effort estimate: **~6–11 days** (dense kernel + model-token correctness gate).

## Goal
Bring `termite_paged_attention_kv_1x` (the scalar decode paged-attention kernel) under the compiler as the first `op_kind=.attention` generated route, mirroring the RMSNorm microkernel slice. Opt-in, hand-written stays production baseline, evidence-gated. This proves the attention path end-to-end; the flash prefill kernel (tunable via `--sweep`) is a later slice.

## The pattern to mirror (RMSNorm microkernel, commit `d3a92dd4e`)
`RegionKernel` is already op-kind-aware (`quant_kernel_metal_renderer.zig:397`):
```zig
pub const RegionKernel = struct {
    kernel_id: []const u8,
    op_kind: OpKind = .small_batch_matmul,
    decoder: FormatDecoder = decoder_q4_0,
    schedule: KernelSchedule,
    epilogue: Epilogue = .none,
    microkernel: MicrokernelKind = .rms_norm,
};
```
`renderRuntimeRegion` (`~:423`) has a two-pass switch on `op_kind` with a `.attention => return error.UnsupportedRegionOpKind` arm — **that's the seam to fill.** The compiler side (`first_generated_microkernel_artifacts` at `quant_kernel_compiler.zig:2665`, `renderMetalMicrokernelSource` at `~:3955`, the RMSNorm RegionKernel appended to `renderMetalRuntimeQuantRegion` at `~:3818`), the codegen `.microkernel` arm (`native_quant_kernel_codegen.zig:278`), and the conformance path (`quant_kernel_metal_runtime_check.zig`: `op_kind` on `CheckCase`, sister FFI `termite_metal_run_generated_microkernel_check`, `referenceRmsNorm`, isolated pass in `main()`) are the templates — replicate each with an `attention` variant.

## The kernel to render: `termite_paged_attention_kv_1x` (`metal_kernels.m:5232-5240`)
Buffers: `q[buffer0]`, `encoded_key[1]`, `v_bytes[2]`, `block_table[3]`, `sinks[4]`, `output[5]`, `params[6]`, `threadgroup float *shmem`. Grid `(q_len, num_heads)`, threadgroup 256. Body (verbatim, 8 lines): setup → per-KV-token score (simd_sum dot over head_dim, causal+sliding mask, paged lookup) → tree-reduce max → softmax (tree-reduce sum) → normalize → strided P·V accumulate.

Params struct `termite_metal_paged_attention_params` (`metal_kernels.m:2287`): `q_len, kv_tokens, num_heads, num_kv_heads, head_dim, key_row_bytes, base_key_row_bytes, query_position_offset, kv_position_offset, sliding_window, v_row_stride, page_size, block_count, contiguous_base_token, contiguous_blocks, format, v_element_bytes, has_sinks, softcap`.

**Parametrizable knobs** (the `AttentionSchedule`): `NT` (threads/threadgroup, currently 256), `NSG = NT/32`. **Structurally fixed:** paged KV via `block_table`/`termite_attention_page_token`, GQA via `heads_per_group`, causal+sliding masking, `format==3` (f16 KV), softcap. Keep those fixed for the first slice; only `NT`/`NSG` vary.

## ⚠️ Key complication (not present for RMSNorm): external dependencies
The kernel references, from the hand-written part of `metal_kernels.m` **outside** the codegen region:
- `termite_metal_paged_attention_params` (the constant struct).
- `inline uint termite_attention_page_token(device const uint*, constant termite_metal_paged_attention_params&, uint)`.

In the runtime region (precise_library) these resolve fine. But the standalone `.metal` file (compiled by the `xcrun` conformance check) will **not** — it needs both definitions. Two options:
- **(A, recommended) Emit them as renderer helper fragments** — add the params-struct text + `termite_attention_page_token` as `HelperFragment`s the attention skeleton emits (deduped by the existing `emitHelper`/`EmittedNames` machinery). Self-contained `.metal`, no special-casing.
- **(B) Route through `metal_runtime_external_helpers`** (the mechanism `termite_q8_0_block_scale` uses) — but that only covers the runtime region, not the standalone `.metal`, so you'd still need (A) for the file. Prefer (A).
Either way, the params struct becomes shared text owned by the compiler with a `--check` guard against the `metal_kernels.m` copy (mirror the `termite_q8_0_block_scale` external-helper drift check).

## Implementation steps (mirror the microkernel slice)
1. **Renderer** (`quant_kernel_metal_renderer.zig`): add an `AttentionKind` (or extend the region enum), an `AttentionSchedule` (or reuse `KernelSchedule` carrying `NT`), `renderAttention1xKernel`/`renderAttention1xBody` (transcribe the 8-line body with `NT`/`NSG` parametrized), the params-struct + `page_token` helper fragments, and fill the `.attention` arm of `renderRuntimeRegion`'s two passes.
2. **Compiler** (`quant_kernel_compiler.zig`): `first_decode_attention_1x_metal_*` constants (kernel_id `antfly_paged_attention_1x_generated_msl_v1`, source_path `src/ops/metal/generated/attention_decode_1x.metal`, an `AttentionSchedule`), a `first_generated_attention_artifacts` list (`op_kind=.attention`; set the meaningless `format=.f32,row_bucket=.rows_2_8,epilogue=.none` sentinels exactly like microkernel), `renderMetalAttentionSource` (comptime, mirrors `renderMetalMicrokernelSource`), and append its RegionKernel in `renderMetalRuntimeQuantRegion`.
3. **Codegen** (`native_quant_kernel_codegen.zig`): add the `.attention` arm to `compiledSourceForArtifact` (same shape as `.microkernel`); include `first_generated_attention_artifacts.len` in `generatedSourceFileCount`/`generatedFiles`/`checkMetalArtifacts`.
4. **Conformance** (`quant_kernel_metal_runtime_check.zig` + `metal_kernels.m`): add a sister FFI `termite_metal_run_generated_attention_check` (binds q/key/v/block_table/output + the params struct; simplest first case = **contiguous single block, f16 KV, no sinks, softcap 0**, so `block_table` is trivial and `page_token` is the contiguous fast path). Add a self-contained `referenceGqaAttention1x` CPU oracle. Isolated float tolerance will be loose (softmax summation-order); use it as a **compile/dispatch sanity check only**.
5. **Dispatch** (`metal_kernels.m`): opt-in `attention_1x_generated_pipeline` behind `TERMITE_METAL_ENABLE_ATTENTION_1X_GENERATED`, from precise_library; a `termite_metal_encode_paged_attention_1x_generated` that binds the same buffers as the hand-written path. Hand-written `termite_paged_attention_kv_1x` stays the default.
6. **Update the Phase 2 invariant test** (all-artifacts-small_batch_matmul) to allow the attention artifact, and add an attention-artifact invariant.

## Correctness gate (the real one)
**Primary = bit-identical model tokens.** Softmax is too summation-sensitive for a tight isolated float gate. Enable the generated 1x kernel and run `scripts/compare_metal_gemma4_e4b_qat.sh` (oracle text + greedy token-id prefix); the generated route must produce the same tokens as the hand-written baseline. Decode-1x is the simplest case (q_len=1, kv grows monotonically), so an A/B of generated-on vs default on E2B/E4B decode is the acceptance test. Keep the isolated float conformance as a fast pre-check for compile/dispatch/gross-logic errors.

## Verification
```
zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dcuda=false
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false -- --write   # regen attention .metal + manifests
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false              # --check byte-sync + xcrun standalone compile (needs the emitted helpers!)
zig build test -Dmetal=true -Dcuda=false -- --test-filter "quant kernel"   # + updated invariants
zig build test -Dmetal=true -Dcuda=false                                   # full suite 0 fail
zig build quant-kernel-metal-production-regression-check ...               # matmul 22/22 UNAFFECTED (separate list)
# On device: enable TERMITE_METAL_ENABLE_ATTENTION_1X_GENERATED, run compare_metal_gemma4_e4b_qat.sh → tokens identical
```

## Risks (ranked)
1. **SEV-HIGH — silent paged-KV / masking bugs.** A block-table off-by-one or a causal-mask slip produces *plausible-but-wrong* tokens. Gate hard on bit-identical tokens; test with kv spanning >1 page (non-contiguous block_table) once the contiguous case works.
2. **SEV-HIGH — external-helper self-containment** (above). The standalone `.metal` must compile with the emitted params struct + `page_token`; add a `--check` drift guard vs the `metal_kernels.m` copies.
3. **SEV-MED — keep the matmul path byte-identical.** Attention artifacts in a separate list (like microkernels); the 22-case regression must not move.
4. **SEV-MED — GQA/sliding/softcap coverage.** First slice can fix `format==3`, no sinks; but GQA (`num_heads>num_kv_heads`) and sliding-window are live in Gemma4 decode — validate both via model tokens before promoting.
5. **SEV-LOW — perf is not the point here.** The generated 1x kernel is a single-sourced copy; expect parity, not gains. Perf comes later from the flash kernel as a `--sweep`-tuned route.

## After this slice
Flash prefill kernel (`termite_paged_attention_kv_prefill_sg`) as a tunable `op_kind=.attention` route — that one carries the real perf lever (`--sweep` over key-chunk 32/64, SG count, gather-vs-direct, rescale-skip). It reuses all the attention plumbing this slice builds. See `PHASE_4_ATTENTION_MICROKERNELS_HANDOFF.md` Slice 2+ and `[[gemma4-prefill-flash-attention]]`.
