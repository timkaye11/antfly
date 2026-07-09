# Phase 4 Hand-off: bring non-matmul kernels (microkernels + attention) under the compiler

All paths relative to `zig/pkg/inference/`. Branch: `codex/quant-kernel-metal-compiler`.
Zig 0.16.0. Line numbers marked **(verify)** came from an exploration sub-agent and may drift a few lines — confirm before editing. The rest were directly verified.

## Goal & framing

Turn the quant *matmul* compiler into a general kernel compiler by generating a **family of narrowly-routed non-matmul kernels** with the same single-sourcing + conformance-evidence discipline: fused **microkernels** (RMSNorm, head-rope, KV read/write) and eventually the **attention** route matrix (decode hot paths, paged/quantized-KV, causal/windowed, prefill flash). This is primarily a **maintainability / single-sourcing** win; perf upside is modest and selective (attention-schedule autotuning; microkernel fusion may reduce the per-op CPU-encode overhead that shows up as the ~0.65× decode gap on high-bandwidth machines like M4 Max — measure, don't assume).

**Explicit non-goal:** general/large dense GEMM stays on vendor libs (MPS/Accelerate/cuBLASLt). `mm_sg` is already a ggml-port near vendor-optimal — do not "generate" it.

## Already in place (do not redo)

- **Phase 1 (single-sourcing) — committed** (`fb8a25136` region, `481355cf1` .metal). The renderer is the single source of truth; `renderRuntimeRegion(allocator, kernels)` (in `src/graph/quant_kernel_metal_renderer.zig`) dedups helpers + emits bodies; `.metal` files render at comptime via `renderMetalSmallBatchSource`.
- **Phase 2 (op-kind seam) — committed** (`f14df7c98`). `OpKind { small_batch_matmul, attention, microkernel }` enum exists; `GeneratedArtifact.op_kind` field (defaults `small_batch_matmul`); and the codegen routing seam is already wired in `src/native_quant_kernel_codegen.zig::compiledSourceForArtifact`:
  ```zig
  switch (artifact.op_kind) {
      .small_batch_matmul => {},
      .attention, .microkernel => return error.UnsupportedGeneratedOpKind,  // <-- YOU WIRE THIS ARM
  }
  ```
  There is an invariant test ("generated artifacts are all small_batch_matmul op_kind") that must be updated when the first non-matmul artifact lands.

## The crux (why this is ~850–1250 lines / multi-week, not small)

The conformance/evidence harness (`src/quant_kernel_metal_runtime_check.zig`) is **matmul-shaped throughout**:
- `CheckCase` **(verify ~606–620)** holds `rows/in_dim/out_dim/threads/cols/format/epilogue` — matmul dims.
- The on-device runner `termite_metal_run_generated_quant_kernel_check` **(verify ~1767–1787)** binds an `(input, weight[quantized], bias, output)` **matmul buffer layout**.
- The CPU reference is hard-wired to `quant_kernel_compiler.referenceMatmulEpilogue()` (dequant + matmul + epilogue) **(verify ~1744–1755)**.
- Dims come from a matmul benchmark table via `metalRuntimeDimsForArtifact` **(verify ~654–669)**.
- Evidence JSON hard-codes `rows/in_dim/out_dim` **(verify ~2878–2950)**.

**The first non-matmul kernel forces generalizing all of this (~400–600 lines).** After that, each additional microkernel/attention route is ~200–300 lines. Do the first slice (RMSNorm) precisely because it pays this whole tax with the simplest possible kernel (pure f32, no quantization) — so a bug is in the harness, not the math.

## Slice 1 — RMSNorm end-to-end (the framework proof; do this first, land it green, commit)

RMSNorm: `input[n,d]`, `weight[d]`, `eps` → `output[n,d]`, `out[i] = in[i] * rsqrt(mean(in²)+eps) * weight[i%d]`. Hand-written kernel + CPU ref already exist to validate bit-identical against.

### 1a. Generalize the conformance harness (the bulk of the work)
In `src/quant_kernel_metal_runtime_check.zig`:
- Add `op_kind: OpKind` to `CheckCase`; default `small_batch_matmul` so existing cases are unchanged.
- In `runCheckImpl` (the per-case runner), branch on `op_kind`:
  - `small_batch_matmul` → today's path unchanged.
  - `microkernel` → new buffer setup (input `[n*d]` f32, weight `[d]` f32, output `[n*d]` f32, params {n, d, eps}) + a new on-device runner.
- **On-device runner:** add a sister C FFI `termite_metal_run_generated_microkernel_check(source, kernel_name, input, weight, params, out, ...)` rather than overloading the matmul one — it compiles the source on-device, binds the microkernel buffer order, dispatches, returns output for host comparison. (Model it on the existing `termite_metal_run_generated_quant_kernel_check`.)
- **CPU reference:** add `referenceRmsNorm(input, weight, n, d, eps, out)` — wrap `activations.zig::rmsNorm` **(verified ~124–144)**. Tolerance: f32 rmsnorm is near bit-exact; use a tight abs tol (e.g. 1e-5) but confirm on-device.
- **Dims generator:** a small per-op-kind function producing RMSNorm shapes (n ∈ {1,2,4}, d ∈ {64,128,512,2048,4096}) — do NOT reuse the matmul dims table.

### 1b. Evidence/artifact shape
- `GeneratedArtifact.op_kind` already exists. For a microkernel artifact, `format`/`row_bucket`/`epilogue` are meaningless — **recommend adding a `format`/`epilogue` "none" sentinel** (cleanest) OR documenting that they're ignored when `op_kind != small_batch_matmul`. Prefer a real `.none` sentinel to avoid confusing field reuse.
- Evidence JSON: add a **top-level `"op_kind"`** and a small `"op_params": { "n":…, "d":…, "eps":… }` object; make `rows/in_dim/out_dim/format/row_bucket` optional (present only for matmul). This is a schema bump — version the evidence schema and update the parser + any golden-string tests. (The lower-churn alternative is to reuse `rows→n, in_dim→d, out_dim=0` with a top-level `op_kind`; acceptable but muddier.)

### 1c. Renderer: `renderMicrokernel`
In `src/graph/quant_kernel_metal_renderer.zig`:
- Add `renderRmsNormKernel(allocator, kernel_id, schedule) ![]u8` (or a small `renderMicrokernel` dispatch). It's ~80–120 lines: 1D grid (one threadgroup per row), threads parallelize the `d` reduction (reuse the existing tree-reduction / `HelperFragment` + `emitHelper` + `EmittedNames` dedup machinery — that part is **op-agnostic and reusable**). Signature: `(device const float* input, device const float* weight, device float* output, constant params&, ...)`. No decoder, no epilogue, no two-col variant.
- Generalize the region assembler: `RegionKernel` currently carries `{kernel_id, decoder, schedule, epilogue}` (matmul-specific). Make it op-kind-aware — simplest is a tagged union or an `op_kind` tag with optional decoder/epilogue — and have `renderRuntimeRegion` dispatch body rendering per op-kind while still globally deduping helpers across all op-kinds.
- `.metal` file: mirror `renderMetalSmallBatchSource` with a `renderMetalMicrokernelSource(comptime …)` so the microkernel `.metal` is single-sourced the same way (comptime render → source constant; keeps the fingerprint/evidence machinery intact — see how Phase 1b kept it comptime).

### 1d. Runtime wiring (`src/backends/metal_kernels.m`)
- The generated microkernel body flows into the codegen region (via the RegionKernel dispatch) and compiles into `precise_library`.
- Add a pipeline field + creation: `runtime->rms_norm_generated_pipeline = termite_metal_make_pipeline(device, precise_library, @"antfly_rms_norm_generated_msl_v1")`, gated by a kill switch (opt-in until promoted), mirroring the small-batch pipeline pattern.
- Add a dispatch/encode function `termite_metal_encode_rms_norm_generated(...)` binding the microkernel buffer order.
- The hand-written `termite_apply_rms_norm_rows` **(verify ~3418–3428, dispatch ~10886–10930, pipeline field line ~534)** stays wired as the baseline; the generated kernel is evidence-gated against it and bit-identical-validated before any promotion.

### 1e. Codegen arm
Wire the `microkernel` arm of the `compiledSourceForArtifact` switch (currently `return error.UnsupportedGeneratedOpKind`) to render the microkernel `.metal` source. Update the Phase 2 invariant test ("all artifacts small_batch_matmul") to allow the new microkernel artifact.

### 1f. Register the artifact + evidence
Add the RMSNorm `GeneratedArtifact` (op_kind=microkernel, kernel_id, source_path `.metal`, check_command) to `first_generated_artifacts`. Run the conformance harness on-device; if bit-identical vs the hand-written baseline and within tol vs the CPU ref, add its evidence. Keep it a dev-only candidate (opt-in kill switch) until you decide to promote.

## Verification (Slice 1, before commit)
```
zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dpjrt=false        # clean
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false -- --write            # regenerate (.metal + manifests incl new microkernel)
zig build quant-kernel-codegen -Dmetal=true -Dcuda=false                       # --check byte-sync
zig build test -Dmetal=true -Dcuda=false -- --test-filter "quant kernel"       # all pass (+ new RMSNorm cases, updated invariant test)
zig build test -Dmetal=true -Dcuda=false                                       # full suite 0 failures
zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false           # on-device conformance incl RMSNorm (max_abs_error within tol; bit-identical vs handwritten)
zig build quant-kernel-metal-production-regression-check ...                    # existing 22/22 unaffected (matmul routes unchanged)
```
Also A/B the generated RMSNorm vs `termite_apply_rms_norm_rows` on-device (bit-identical output on representative shapes) before wiring/promoting.

## Slice 2+ — scale to the attention family (the real value; separate commits)
Once the harness is generalized, each of these is ~200–300 lines (render fn + conformance case + dispatch + evidence):
1. **Decode attention hot paths** (fixed Gemma/Qwen head layouts, paged, GQA/MQA, causal/windowed) — highest-priority route matrix.
2. **Paged / quantized-KV read** kernels.
3. **More fused microkernels**: head-rope, KV write/read.
4. **Prefill flash** (`termite_paged_attention_kv_prefill_sg`) as a *tunable* route — schedule knobs {key-chunk 32/64, SG count, gather-vs-direct, rescale-skip}, autotuned via `--sweep`. (This kernel was recently hand-tuned with a contiguous direct-load fast path — bring it in as the baseline, then sweep.)
CPU conformance oracles live in `src/ops/native_compute.zig` (`causalSelfAttentionOp`, `gqaPagedAttention`, `rmsNormOp`, `ropeOp`). Correctness gate for attention = **bit-identical model tokens** (`scripts/compare_metal_gemma4_e4b_qat.sh` oracle text + greedy token-id prefix), since attention feeds the whole model. Keep hand-written kernels wired until a generated route wins its evidence gate.

## Risks (ranked)
1. **SEV-HIGH — harness generalization is the crux.** ~400–600 lines touching CheckCase, the FFI runner, CPU refs, dims, and evidence JSON. Do it with RMSNorm (simplest kernel) so failures are harness bugs, not math. Keep the matmul path byte-identical (existing 22/22 must not move).
2. **SEV-MED — evidence schema churn.** Adding op_kind/op_params bumps the evidence schema; update parser + golden-string tests. Prefer the schema bump over field-reuse for clarity.
3. **SEV-MED — RegionKernel/region-assembler generalization.** The dedup+emit is reusable, but the body-render dispatch and comptime `.metal` source constants must handle multiple op-kinds without breaking Phase 1b's comptime-fingerprint design.
4. **SEV-LOW — attention numerics.** Generated attention won't be byte-identical to hand-written (summation order); gate on bit-identical *tokens* (argmax-robust), not raw floats, per the flash-attention precedent.

## Key files
- `src/quant_kernel_metal_runtime_check.zig` — harness generalization (the bulk).
- `src/graph/quant_kernel_metal_renderer.zig` — `renderMicrokernel`/`renderAttention`, RegionKernel op-kind dispatch.
- `src/graph/quant_kernel_compiler.zig` — new artifacts/evidence, `.none` format/epilogue sentinels, `renderMetalMicrokernelSource` comptime source, update Phase 2 invariant test.
- `src/native_quant_kernel_codegen.zig` — wire the `microkernel`/`attention` arm of `compiledSourceForArtifact`.
- `src/backends/metal_kernels.m` — generated pipeline fields/creation/dispatch + kill switches; hand-written baselines stay.
- `src/ops/native_compute.zig`, `src/backends/activations.zig` — CPU conformance oracles.
