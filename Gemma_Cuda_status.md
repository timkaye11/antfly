# Gemma CUDA Status

Last updated: 2026-06-12 16:15 UTC

## Latest Status - 2026-06-12

CUDA Gemma4 now returns valid tokens for the local formats we have been testing.

Root cause found and fixed:

- CUDA `reshape2d` is a non-owning view.
- Several Gemma4 norm helpers were returning a reshaped view of `normed_flat` while also scheduling `defer cb.free(normed_flat)`.
- With CUDA's async stream and temp-buffer cache, that left returned tensors pointing at freed/reusable device storage.
- Symptoms matched the bad-token failure exactly: BF16/Q8 session-bound weights matched, but activations changed between trace points (`v_norm` vs `v_attn`) and `q_attn` could pick up K-norm-scale-like values.
- Patch: add CUDA `cloneTensorShape` as an owned device-to-device shape clone and use `reshape2dOwned` for returned Gemma4 norm reshape results. Also synchronize before uncached device frees to keep the temp-cache-disabled diagnostic path honest.

Validated after the fix:

- Q8_0 GGUF CUDA:
  - Path: `.models/google/gemma-4-12B-it-q8_0`
  - Prompt: `Write one sentence about ants.`
  - Output: `Ants are highly social insects that live in complex colonies and work together to build intricate underground structures.`
  - Token ids start with `14054`.
  - 20 tokens, generation time about 7.7s after load.
- Q4_K GGUF CUDA:
  - Path: `.models/google/gemma-4-12B-it-q4_k`
  - Prompt: `Write one sentence about ants.`
  - Output: `Ants are highly social insects that work together in complex colonies to gather food and build intricate underground structures.`
  - Token ids start with `14054`.
  - 21 tokens, generation time about 8.2s after load.
- Full BF16 safetensors CUDA:
  - Path: `.models/google/gemma-4-12B-it`
  - First-token control returns token id `14054`, text `Ant`, matching Q8/Q4.
  - 32-token streamed BF16 generation timed out at 300s, so correctness is good enough as a first-token/logit control but performance remains the major issue.

Post-fix activation trace:

- Q8_0 vs BF16 row 1: `native_top1=14054`, `reference_top1=14054`, no layer-0 suspects, `q_attn rel_rmse=0.005300`.
- Q8_0 vs BF16 row 0: `native_top1=14054`, `reference_top1=14054`, `v_norm_eq_v_attn` passes on both sides. Some early row-0 thresholds remain conservative, but final row/layer outputs align tightly.

Remaining work:

- BF16 streamed dense/FFN throughput is still poor: the 1-token control took about 123s, dominated by dense streaming and FFN reads/uploads.
- Q4_K quality is now coherent on the smoke prompt, but it still needs a real KL/top-1 eval suite against BF16 before calling it production-ready.
- Q4_K fast decode reports fallbacks (`cuda_q4k_fast_counts: decode_fallbacks=4389`), so there is still performance work in the quantized kernels.
- The old `row0_v_attn_eq_attn_out` trace invariant is shape-invalid for GQA (`2048` vs `4096`) and should be removed or made GQA-aware.

Older sections below are historical diagnostics. Treat them as superseded where they say Q8_0/Q4_K CUDA generation returns garbage.

## Goal

Get valid CUDA token outputs for both:

- Full Google Gemma4 12B instruction model from HF safetensors.
- Quantized Gemma4 GGUF exports, currently Q8_0 and Q4_K.

## Current Summary

BF16/full safetensors CUDA is the healthy correctness control. It runs on the L4 and returns plausible first tokens.

Q8_0 and Q4_K GGUF CUDA paths still return bad first tokens. Phase 0 config/overlay checks and the new session-bound weight-binding audit both closed major hypotheses: the Q8 session is not obviously running with different scalar config, and layers 0-5 are not misbound at the weight-slot level. A real-session activation trace now shows the Q8-vs-BF16 divergence starts in layer 0. The clean last-row trace has healthy input/q/k/v prep and then a large break at attention output, so the next target is Gemma4 attention trajectory/runtime semantics rather than more direct weight-slot auditing.

The current strongest result:

- Q8_0 CUDA math matches CPU math for the Q8_0 artifact very tightly.
- Q8_0 dequantized weights are close to HF in direct tensor parity and grouped RMSE. Attention and FFN projection groups are around `rel_rmse=0.0055-0.0057`, sampled embeddings are around `rel_rmse=0.0108`, and norms/scales are exact.
- Lazy-dequantizing Q8 projection weights and the tied embedding/head still produces bad first tokens.
- A layer0-dense Q8 hybrid still produces the same bad first token class as normal Q8.
- Sequential BF16-vs-Q8 first-token compare now runs under CUDA budgets and shows the top-logit distributions are badly divergent, not merely slightly reordered.
- Effective configs match between BF16 safetensors and Q8 GGUF sessions; disabling the GGUF config overlay did not fix Q8.
- Session-bound weight audit for layers 0-5 found no suspect graph weight slots: `ok=84`, `suspect=0`, `skipped=2` (embedding/tied head shape-only). Q8-vs-BF16 matrix RMS-relative differences are around `0.00003-0.00005`, and norms/scales are exact.
- Real-session activation trace found the first clean last-row divergence at `layer0.attn_out`: input, `attn_norm`, `q_raw`, `k_raw`, `v_norm`, `q_attn`, `k_attn`, and `v_attn` are all within normal Q8 noise, then `attn_out` jumps to `cosine=0.8275`, `rel_rmse=0.5615`.
- Q4_K diverges much more severely after one full layer.

So the quantized artifacts are internally consistent and the early session binding is clean, but the full prompt trajectory diverges immediately once layer-0 attention consumes the prompt K/V trajectory.

## Environment

- GPU: NVIDIA L4
- Visible VRAM from `nvidia-smi`: 23034 MiB
- CUDA driver reported by `cuda-info`: driver version 12020, device `sm_89`
- Workspace disk after cleanup and completed hybrid export: about 99 GB total, about 7.3 GB free

## Local Model Artifacts

Current important artifacts:

- `.models/google/gemma-4-12B-it/model.safetensors`
- `.models/google/gemma-4-12B-it-q8_0.gguf`
- `.models/google/gemma-4-12B-it-q4_k.gguf`
- `.models/google/mmproj.gguf`
- `.models/google/mmproj-q8_0.gguf`
- `.models/google/gemma-4-12B-it-q8_0-layer0dense.gguf`
- `.models/google/gemma-4-12B-it-q8_0-layer0dense/` wrapper directory with tokenizer/config symlinks
- `.models/google/mmproj-q8_0-layer0dense.gguf`

Removed to free space:

- `.models/google/gemma-4-12B-it-q4_k_dense_embd.gguf`
- `.models/google/mmproj-q4_k_dense_embd.gguf`
- `zig/pkg/inference/.zig-cache`

## Confirmed Working

### Full BF16 Safetensors

The full HF safetensors model works as the CUDA control:

- Model path: `.models/google/gemma-4-12B-it`
- Backend: CUDA
- Prompt examples tested:
  - `Ants`
  - `Write one sentence about ants.`
- First-token behavior was plausible. Example prior control:
  - `Ants` produced token id `236761`, text `.`
  - `Write one sentence about ants.` produced token id `45518`, text `thought`

Performance is slow because the model runs with streaming/offload, but correctness is acceptable enough to use as the control.

### CUDA Device And Primitive Health

`cuda-info --smoke` passes:

- fill f32
- dense f32
- Q8_0 matmul
- Q4_0 matmul
- Q4_K matmul
- Gemma4 primitives
- cuBLASLt BF16

Q8_0 real-tensor CUDA parity also passes tightly against CPU for the Q8_0 artifact:

- Layer0 full subgraph parity: ok
- Layer5 attention parity: ok
- Final norm parity: ok
- Sampled tied LM-head parity: ok

This means CUDA is matching the quantized artifact. It does not mean the quantized artifact matches BF16 generation behavior.

## Still Broken

### Q8_0 Normal CUDA Generation

Path:

- `.models/google/gemma-4-12B-it-q8_0`

Observed bad output examples:

- Prompt `Ants` returns bad/special/garbage first tokens.
- Prompt `Write one sentence about ants.` returns bad tokens such as `сто` or `NAN` depending on path/settings.

The top logits are suspicious. Example Q8 lazy-dequant projection run:

- Raw top token id `258882`, empty/special text
- Post-suppress top candidates included `zA`, `DSA`, `Sp`, multilingual fragments, etc.

### Q4_K Normal CUDA Generation

Path:

- `.models/google/gemma-4-12B-it-q4_k`

Observed behavior:

- Produces garbage or blank/special output.
- Q4_K with dense embedding did not fix the issue.
- Q4_K layer0 cross-drift is much larger than Q8_0, so Q4_K is currently less healthy than Q8_0.

## Diagnostics Added Or Changed

### CUDA GGUF Lazy Dequant Diagnostic

Added a gated diagnostic loader policy in `session_factory.zig`.

Gate:

- `TERMITE_CUDA_DEQUANTIZE_QUANT_WEIGHTS=1`
- or `ANTFLY_INFERENCE_CUDA_LAZY_GGUF_QUANT_DEQUANT=1`

Behavior:

- For GGUF Gemma quantized projection weights, put them into the lazy store.
- Dequantize and upload them one at a time through the existing lazy device budget path.
- Keep the embedding table quantized/resident to avoid a ~4 GB dense embedding allocation.
- Optional follow-up gate added for embedding/head dequant:
  - `TERMITE_CUDA_DEQUANTIZE_QUANT_EMBEDDING=1`
  - or `ANTFLY_INFERENCE_CUDA_LAZY_GGUF_QUANT_EMBEDDING=1`

This let Q8 run with layer projections dequantized to f32 without requiring a 24 GB resident dense model.

Result:

- The projection-only lazy-dequant Q8 run completed.
- It still produced a bad first token.
- The projection-plus-embedding lazy-dequant Q8 run also completed when backend budget was lowered to force earlier eviction.
- It still produced a bad first token (`TI`).
- Therefore the normal Q8 quantized matmul kernel is not the only culprit.

### Lazy Device LRU Fix

Fixed `noteLazyDeviceAccess` so resident lazy weights refresh their LRU epoch on reuse even when `bytes == 0`.

Reason:

- Before this, lazy resident weights could be treated as old even after reuse.
- This matters for streamed/offloaded CUDA runs near the L4 memory limit.

### CUDA Fused QKV Disable Switch

Added:

- `ANTFLY_CUDA_DISABLE_FUSED_QKV=1`

Purpose:

- Force Q/K/V projection through separate dense linears instead of fused QKV.

Result:

- Q8 lazy-dequant snapshots were effectively unchanged.
- `cuda_qkv_counts` dropped to zero, but bad hidden states and tokens remained.
- Fused QKV is ruled out as the primary issue.

### Export CLI Alias

Added `--format` as an alias for `--quantize` in `native_export_gguf.zig`.

Reason:

- Usage advertised `--format`, but the parser only accepted `--quantize`.
- This likely does not explain the existing artifacts if they were produced with `--quantize`, but it prevents future silent confusion.

### Cross-Layer Diagnostic

Added:

```bash
antfly-inference cuda-info --gemma4-cross-layer0 <gguf> <hf-model-dir>
```

Purpose:

- Compute a CPU layer0 forward pass using dequantized GGUF weights.
- Compute the same CPU layer0 forward pass using HF safetensors BF16 weights.
- Print per-stage drift.

This avoids creating another huge GGUF just to answer whether quantized weights are already diverging from BF16 after one layer.

### Cross-Parity Diagnostic Extension

Extended `cuda-info --gemma4-cross-parity` to compare 1D tensors and added `output_norm.weight` to the sampled cross-parity set.

Result:

- `output_norm.weight` matches HF exactly.
- Existing sampled `token_embd.weight` row comparisons show small Q8_0 row/dot errors.
- Final-logit failure is therefore not explained by a bad exported final norm or an obviously broken tied embedding/head tensor.

### Grouped Weight-Space RMSE Diagnostic

Added:

```bash
antfly-inference cuda-info --gemma4-cross-rmse <gguf> <hf-model-dir>
```

Purpose:

- Sweep GGUF tensor values against the source HF safetensors values.
- Group errors by tensor category.
- Sample `token_embd.weight` rows instead of materializing the full ~4 GB dense embedding.
- Sample oversized 2D quantized tensors row-wise when they exceed `ANTFLY_INFERENCE_CUDA_RMSE_MAX_ELEMENTS`.
- Print full per-tensor lines only with `ANTFLY_INFERENCE_CUDA_RMSE_VERBOSE=1`.
- Cap fully materialized tensors with `ANTFLY_INFERENCE_CUDA_RMSE_MAX_ELEMENTS`; default is `40000000`.

Important implementation note:

- The first full sweep tried to load 58,982,400-element FFN matrices and got stuck in I/O while holding most system RAM.
- The diagnostic now checks GGUF tensor element count before `loadTensorRef`, so oversized tensors are skipped before allocation.
- Large FFN tensors now use sampled row coverage instead of full materialization, so the grouped summary includes `ffn_gate`, `ffn_up`, and `ffn_down` without exhausting RAM.

### First-Token Compare Diagnostic

Extended `compare_generate.zig` so native/reference comparisons retain last-token logits and print:

- top-k overlap
- each model's top-1 rank in the other model's logits
- top-1 logit margins
- decoded empty-text and manifest-special flags

This is intended for BF16-vs-Q8 first-token comparisons once memory allows loading both models safely, or for external GGUF cross-checks.

Sequential mode also works for pairs that would otherwise exceed memory when
loaded together. It analyzes the native model, releases it, then analyzes the
reference model before comparing retained logits.

`compare` now accepts the same memory-budget flags used by generation:

- `--host-budget-mb`
- `--backend-budget-mb`
- `--combined-budget-mb`
- `--kv-budget-mb`
- `--scratch-budget-mb`

### Session Weight-Binding Audit

Added:

```bash
antfly-inference compare <q8-model-dir> <bf16-model-dir> <prompt> \
  --weight-binding-audit \
  --binding-audit-layer-limit N \
  --native-backend native \
  --reference-backend native
```

Purpose:

- Load weights through the built session and `gpt_arch.getModelWeight`, not through the standalone GGUF tensor diagnostics.
- Print and diff per-slot shape, l2, RMS, mean-abs, max-abs, and first values.
- Exercise `weight_prefix` handling and generation-style semantic fallbacks such as `pre_feedforward_layernorm -> post_attention_layernorm`, tied head fallback to embedding, and Gemma4 `layer_scalar`/`per_layer_input.layer_output_scale` naming.
- Fall back from `toFloat32` to `exportTensorData` so native quantized GGUF handles can be dequantized for the audit.

Result for layers 0-5:

- Command log: `/tmp/gemma4-weight-binding-audit-layer6.log`
- Totals: `ok=84`, `suspect=0`, `skipped=2`.
- The only skipped slots are the huge embedding table and tied head, intentionally shape-only.
- Attention/FFN matrices in layers 0-5 all match within normal Q8 noise:
  - typical `rms_rel=0.00003-0.00005`
  - norms and layer scales are exact
  - layer 5 full-attention `v_proj` is omitted as expected by Gemma4 config
- Q8 GGUF uses `model.layers.N.per_layer_input.layer_output_scale.weight`; BF16 safetensors resolves the same slot through the `model.layers.N.layer_scalar` fallback. Values match exactly.

Conclusion:

- The leading session-level misbinding hypothesis is falsified for the layers where the existing hidden-state evidence said divergence had already appeared.
- The next bisection should trace activations through the real session graphs and identify the first divergent op, rather than continuing direct weight-slot audits.

### Real-Session Activation Trace

Added:

```bash
antfly-inference compare <q8-model-dir> <bf16-model-dir> <prompt> \
  --activation-trace \
  --activation-trace-layer-limit N \
  --activation-trace-layer L \
  --native-backend cuda \
  --reference-backend cuda
```

Purpose:

- Run Q8 GGUF and BF16 safetensors sequentially through the built sessions.
- Capture activation rows from the actual GPT forward path, not standalone CPU reimplementations.
- Compare cosine similarity, rel-RMSE, RMS, mean/max absolute error, and first suspect surface.
- By default capture last-row input, layer outputs, pre-final norm, and final norm.
- With `--activation-trace-layer L`, capture detailed layer internals: attention norm, q/k/v projections, q/k/v attention inputs, attention output/projection/post-norm/residual, FFN stages, PLE, output scale.
- With `--activation-trace-all-rows`, capture full prompt-row tensors for selected surfaces. This is useful but can perturb lazy CUDA materialization, so interpret token changes from that mode cautiously.

Layer-output result:

```bash
timeout 900s zig/pkg/inference/zig-out/bin/antfly-inference compare \
  .models/google/gemma-4-12B-it-q8_0 \
  .models/google/gemma-4-12B-it \
  "Write one sentence about ants." \
  --activation-trace \
  --native-backend cuda \
  --reference-backend cuda \
  --activation-trace-layer-limit 8 \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024
```

Important output:

- Prompt/tokenization matched: 19 tokens.
- Q8 top1: `258882`; BF16 top1: `45518`.
- `input`: `cosine=0.999984722`, `rel_rmse=0.005529`, healthy.
- `layer0.out`: `cosine=0.910530952`, `rel_rmse=0.437977`, first suspect.
- Later layers rapidly collapse: by `layer5.out`, `cosine=0.091752563`, `rel_rmse=1.463662`.

Layer-0 detailed last-row result:

```bash
timeout 900s zig/pkg/inference/zig-out/bin/antfly-inference compare \
  .models/google/gemma-4-12B-it-q8_0 \
  .models/google/gemma-4-12B-it \
  "Write one sentence about ants." \
  --activation-trace \
  --activation-trace-layer 0 \
  --activation-trace-layer-limit 1 \
  --native-backend cuda \
  --reference-backend cuda \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024
```

Important output:

- Healthy through layer-0 attention prep:
  - `layer0.attn_norm`: `cosine=0.999987210`, `rel_rmse=0.005058`
  - `layer0.q_raw`: `cosine=0.999975815`, `rel_rmse=0.006971`
  - `layer0.k_raw`: `cosine=0.999981072`, `rel_rmse=0.006154`
  - `layer0.v_norm`: `cosine=0.999959300`, `rel_rmse=0.009022`
  - `layer0.q_attn`: `cosine=0.999970340`, `rel_rmse=0.007702`
  - `layer0.k_attn`: `cosine=0.999976543`, `rel_rmse=0.006849`
  - `layer0.v_attn`: `cosine=0.999959300`, `rel_rmse=0.009022`
- First clean last-row suspect:
  - `layer0.attn_out`: `cosine=0.827528788`, `rel_rmse=0.561487`
- The rest of layer 0 remains divergent, ending at:
  - `layer0.out`: `cosine=0.910530952`, `rel_rmse=0.437977`

All-rows layer-0 trace:

- `--activation-trace-all-rows` was also run for layer 0.
- It changed the Q8 selected token (`181812` instead of `258882`), so it likely perturbs lazy CUDA graph/materialization and should not be treated as a clean generation-path result.
- It did reveal a lead: full prompt-row `layer0.q_attn` diverged (`cosine=0.704349288`, `rel_rmse=0.996879`) even though `q_raw`/`k_raw` were healthy. This needs a non-perturbing row-wise bisection before blaming the attention core directly.

Conclusion:

- The failure is not layer-5-or-later accumulation. It is already present in layer 0.
- Last-row q/k/v prep is healthy, then attention output is not.
- Next work should isolate prior-token K/V rows, Q head-norm/scale behavior across prompt rows, and attention scores/probabilities in a way that does not change the selected token.

## Key Results

### Export/Dry-Run Policy

Q8_0 dry run:

- 666 tensors total
- 329 quantized
- Norms and scalar weights remain BF16/dense
- Embedding and projection matrices are quantized

Q4_K dry run:

- Same tensor policy: 666 tensors total, 329 quantized
- Norms and scalar weights remain BF16/dense
- Embedding and projection matrices are quantized

This does not look like an obvious “we quantized norms/scalars by accident” problem.

### Q8_0 Layer0 Cross Drift

Command:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-layer0 \
  .models/google/gemma-4-12B-it-q8_0/model.gguf \
  .models/google/gemma-4-12B-it
```

Important output:

- `attn_norm`: `max_abs=0`, `mean_abs=0`, `rel_rmse=0`
- `q`: `max_abs=1.287127`, `mean_abs=0.090192`, `rel_rmse=0.008197`
- `k`: `max_abs=1.112339`, `mean_abs=0.091787`, `rel_rmse=0.006436`
- `v`: `max_abs=1.063522`, `mean_abs=0.106662`, `rel_rmse=0.007269`
- `attn_out`: `max_abs=0.200207`, `mean_abs=0.006587`, `rel_rmse=0.012559`
- `attn_proj`: `max_abs=0.098482`, `mean_abs=0.009522`, `rel_rmse=0.013102`
- `ffn_gate`: `max_abs=1.008003`, `mean_abs=0.045652`, `rel_rmse=0.016603`
- `ffn_up`: `max_abs=1.321127`, `mean_abs=0.054306`, `rel_rmse=0.019140`
- `ffn_gated`: `max_abs=29.256592`, `mean_abs=0.134072`, `rel_rmse=0.024929`
- `ffn_raw`: `max_abs=10.852203`, `mean_abs=0.490257`, `rel_rmse=0.021192`
- `out`: `max_abs=0.497786`, `mean_abs=0.016557`, `rel_rmse=0.013840`

Interpretation:

- Q8_0 is not catastrophically wrong after one layer.
- But it is not numerically equivalent to BF16.
- The error can plausibly accumulate over 48 layers, especially with Gemma4’s large normalization behavior.

### Q4_K Layer0 Cross Drift

Command:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-layer0 \
  .models/google/gemma-4-12B-it-q4_k/model.gguf \
  .models/google/gemma-4-12B-it
```

Important output:

- `q`: `max_abs=16.111282`, `mean_abs=1.397964`
- `k`: `max_abs=14.041732`, `mean_abs=1.373720`
- `v`: `max_abs=22.639640`, `mean_abs=1.629242`
- `ffn_gated`: `max_abs=623.778300`, `mean_abs=1.725435`
- `ffn_raw`: `max_abs=159.456270`, `mean_abs=5.786045`
- `out`: `max_abs=10.740052`, `mean_abs=0.218122`

Interpretation:

- Q4_K is far more divergent than Q8_0 after only one layer.
- Q4_K bad tokens are currently explainable as quantization-quality/policy failure, not just CUDA runtime failure.

### Q8_0 Projection Lazy-Dequant Generation

With layer projection weights lazy-dequantized to f32 and embedding kept quantized:

- `native_resident_count=338`
- `native_resident_mb=1021`
- `native_lazy_count=328`
- `native_lazy_mb=11044`
- Completed generation without OOM using lower accepted backend budget.

But output was still bad:

- Example token: `zA`
- Top logits still included empty/special or garbage candidates.

Interpretation:

- Native CUDA Q8_0 matmul is not the only issue.
- Accumulated layer drift is enough to ruin the first token.

### Q8_0 Projection Plus Embedding Lazy-Dequant Generation

With quantized projections and `token_embd.weight` lazy-dequantized to f32:

- Initial run with `--backend-budget-mb 19000` OOMed while allocating the ~4 GB f32 embedding/head.
- Rerun with `--backend-budget-mb 12500` completed after forcing earlier lazy eviction.
- Lazy profile shape:
  - `native_resident_count=337`
  - `native_resident_mb=1`
  - `native_lazy_count=329`
  - `native_lazy_mb=12064`
- Output was still bad:
  - Raw top token id `258882`, empty/special text
  - Post-suppress token id `17578`, text `TI`

Interpretation:

- Embedding/head quantization is not the primary cause.
- Dequantizing Q8_0 weights after export cannot recover BF16 behavior; the drift is already baked into the quantized values or into the mixed quantized execution trajectory.

### Prompt Snapshot Comparison

Prompt: `Ants`, chat template enabled, one-token prefill.

BF16 control:

- `input` l2: `4105.589202`
- `layer5_out` l2: `7362.323303`
- `layer23_out` l2: `51113.970307`
- `layer47_out` / `pre_final_norm` l2: `129.300868`
- `final_norm` l2: `33272.501934`
- First token: id `236761`, text `.`

Q8_0:

- `input` l2: `4106.250115`
- `layer5_out` l2: `9597.835481`
- `layer23_out` l2: `12815.925682`
- `layer47_out` / `pre_final_norm` l2: `118.037614`
- `final_norm` l2: `152547.765386`
- First token in that snapshot run: id `118965`, text `знако`

Interpretation:

- The prompt embedding scale starts close.
- Q8_0 hidden-state direction diverges through the stack even though direct tensor parity is close.
- `pre_final_norm` total l2 is similar, but the final normalized vector lands in a very different direction/distribution, causing the tied head to rank garbage/special tokens above punctuation.

### Q8_0 Cross-Parity

Important results from `cuda-info --gemma4-cross-parity`:

- `output_norm.weight`: `weight_max_abs=0`, `weight_mean_abs=0`
- Sampled projection weights generally have Q8_0 `weight_mean_abs` around `0.00005` to `0.00014`.
- Sampled embedding rows are close:
  - Token `236751`: `weight_mean_abs=0.000118`, `dot_abs=0.002739`
  - Token `236761`: `weight_mean_abs=0.000096`, `dot_abs=0.003478`
  - Token `258882`: `weight_mean_abs=0.000619`, `dot_abs=0.024163`

Interpretation:

- Exported norms and sampled embedding/head rows look sane.
- The failure is not an obvious tensor-name mixup or final-norm export bug.

### Q8_0 Grouped Weight-Space RMSE

Command:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-rmse \
  .models/google/gemma-4-12B-it-q8_0/model.gguf \
  .models/google/gemma-4-12B-it
```

Completed with the default safety cap and sampled oversized FFN coverage:

- `max_full_elements=40000000`
- skipped tensors: `0`
- sampled oversized tensors: all `ffn_down`, `ffn_gate`, and `ffn_up` tensors because each has `58,982,400` elements

Grouped results:

- `token_embd`: `samples=13`, `rel_rmse=0.010837`, `rmse=0.000402`, `max_abs=0.016113`
- `output_norm`: `rel_rmse=0`, exact
- `norm`: `rel_rmse=0`, exact
- `layer_scale`: `rel_rmse=0`, exact
- `attn_q`: `rel_rmse=0.005564`, `rmse=0.000151`, `max_abs=0.006332`
- `attn_k`: `rel_rmse=0.005667`, `rmse=0.000145`, `max_abs=0.002289`
- `attn_v`: `rel_rmse=0.005665`, `rmse=0.000142`, `max_abs=0.006012`
- `attn_output`: `rel_rmse=0.005493`, `rmse=0.000146`, `max_abs=0.004761`
- `ffn_gate`: `samples=432`, `rel_rmse=0.005539`, `rmse=0.000148`, `max_abs=0.001776`
- `ffn_up`: `samples=432`, `rel_rmse=0.005575`, `rmse=0.000153`, `max_abs=0.001553`
- `ffn_down`: `samples=432`, `rel_rmse=0.005632`, `rmse=0.000138`, `max_abs=0.002312`

Interpretation:

- Direct Q8 tensor errors are small and category-consistent for embeddings, attention projections, and sampled FFN projections.
- Norms/scales/output norm are exact.
- No sampled category points to a corrupted export, name/layout mismatch, or broken Q8 dequant scale.
- The current evidence keeps pointing at accumulated trajectory sensitivity or quantization policy, not a single bad tensor.

## Latest Hybrid Export

Exported a hybrid Q8_0 artifact with layer 0 kept dense/BF16:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference export \
  .models/google/gemma-4-12B-it \
  --target gguf \
  --format q8_0 \
  --quantize-exclude model.language_model.layers.0.,blk.0. \
  --output .models/google/gemma-4-12B-it-q8_0-layer0dense.gguf \
  --projector-output .models/google/mmproj-q8_0-layer0dense.gguf
```

Status:

- Export completed successfully.
- Main GGUF size: about 12 GB.
- Projector GGUF size: about 100 MB.
- Disk free after export: about 7.3 GB.
- Dry-run policy for this export reports 666 tensors total and 322 quantized, down from 329 quantized in the normal Q8_0 export.
- The dry-run confirms `blk.0.*` projection and FFN weights are dense BF16, while layer 1 and later projection/FFN weights remain Q8_0.
- Added `.models/google/gemma-4-12B-it-q8_0-layer0dense/` wrapper directory so generation can find tokenizer/config sidecars.

Purpose:

- If keeping layer0 dense makes early snapshots match BF16, then layer0 quantization drift is confirmed as meaningful.
- If first token remains bad, we can try keeping more early layers dense or keeping embedding/output dense.
- Because disk is tight again, run this test before producing another hybrid artifact.

### Layer0-Dense Cross Drift

Command:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-layer0 \
  .models/google/gemma-4-12B-it-q8_0-layer0dense.gguf \
  .models/google/gemma-4-12B-it
```

Important output:

- `attn_norm`: `max_abs=0`, `mean_abs=0`, `rel_rmse=0`
- `q`: `max_abs=0.283093`, `mean_abs=0.019712`, `rel_rmse=0.001795`
- `k`: `max_abs=0.437840`, `mean_abs=0.021729`, `rel_rmse=0.001676`
- `v`: `max_abs=0.375464`, `mean_abs=0.023014`, `rel_rmse=0.001650`
- `attn_out`: `max_abs=0.076716`, `mean_abs=0.001432`, `rel_rmse=0.003354`
- `attn_proj`: `max_abs=0.033675`, `mean_abs=0.002620`, `rel_rmse=0.003907`
- `ffn_gate`: `max_abs=0.250687`, `mean_abs=0.010417`, `rel_rmse=0.003891`
- `ffn_up`: `max_abs=0.377448`, `mean_abs=0.012500`, `rel_rmse=0.004387`
- `ffn_gated`: `max_abs=10.894043`, `mean_abs=0.031010`, `rel_rmse=0.006084`
- `ffn_raw`: `max_abs=2.579498`, `mean_abs=0.111853`, `rel_rmse=0.004686`
- `out`: `max_abs=0.181189`, `mean_abs=0.004447`, `rel_rmse=0.004034`

Interpretation:

- Layer0-dense materially reduces drift versus normal Q8_0. The final layer0 `out rel_rmse` drops from `0.013840` to `0.004034`, and `ffn_gated rel_rmse` drops from `0.024929` to `0.006084`.
- It does not make the layer0 path exactly match the HF BF16 control.
- The exclude pattern did match layer0, so the remaining drift needs a narrower check of the BF16 GGUF export path and/or quantized tensors that still feed the first-token path, especially `token_embd.weight` and the tied output head.

### Layer0-Dense Generation

Prompt: `Ants`, chat template enabled.

Result:

- Layer0-dense Q8 still produced token id `236751`, text `s`.
- Raw top token was still id `258882`, empty/special text.
- Normal Q8 with the same prompt also produced token id `236751`, text `s`.

Interpretation:

- Keeping only layer0 dense is not enough.
- The problem is not isolated to layer0 projection/FFN quantization.

### Q8 Normal vs Layer0-Dense First-Token Compare

Command:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference compare \
  .models/google/gemma-4-12B-it-q8_0 \
  .models/google/gemma-4-12B-it-q8_0-layer0dense \
  "Ants" \
  --backend cuda \
  --top-k 10 \
  --sequential
```

Result:

- Normal Q8 and layer0-dense Q8 both had raw top token id `258882`, decoded as empty text and not flagged as manifest-special.
- Both models ranked that same token first in the other model's logits.
- The top-10 overlap was `3/10`.
- Layer0-dense changed lower-ranked logits, but did not dislodge the bad raw top token.

Interpretation:

- Layer0-dense improves layer0 numeric drift, but the final first-token failure survives.
- The next comparison needs a true BF16 or known-good external GGUF reference, not only another artifact from the same exporter/profile family.

### Q8 vs BF16 Sequential First-Token Compare

These runs use the new budgeted sequential compare path. Q8 is loaded and
analyzed first, then released before the BF16 safetensors control is loaded.

Common flags:

```bash
ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN=1 \
ANTFLY_INFERENCE_CUDA_TEMP_CACHE_MB=512 \
zig/pkg/inference/zig-out/bin/antfly-inference compare \
  .models/google/gemma-4-12B-it-q8_0 \
  .models/google/gemma-4-12B-it \
  "<prompt>" \
  --backend cuda \
  --top-k 10 \
  --sequential \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024
```

Prompt: `Ants`, chat template enabled.

- Q8 top-1: id `258882`, empty text, logit `29.422495`.
- BF16 top-1: id `236761`, text `.`, logit `19.473986`.
- Top-10 overlap: `0/10`.
- Q8 top-1 rank in BF16 logits: `249777`.
- BF16 top-1 rank in Q8 logits: `51897`.

Prompt: `Write one sentence about ants.`, chat template enabled.

- Q8 top-1: id `200797`, text `වීම`, logit `22.781525`.
- BF16 top-1: id `45518`, text `thought`, logit `17.998144`.
- Top-10 overlap: `1/10`.
- Q8 top-1 rank in BF16 logits: `135642`.
- BF16 top-1 rank in Q8 logits: `180266`.

Prompt: `Write one sentence about ants.`, raw prompt with `--raw-prompt --no-chat-template`.

- Q8 top-1: id `194569`, text `refe`, logit `23.225800`.
- BF16 top-1: id `236761`, text `.`, logit `8.468363`.
- Top-10 overlap: `0/10`.
- Q8 top-1 rank in BF16 logits: `11378`.
- BF16 top-1 rank in Q8 logits: `203321`.

Interpretation:

- Q8 is not close to the BF16 control at the logits distribution level.
- The failure is larger than a greedy tie-break, tokenization issue, or small top-k reorder.
- Since direct Q8 tensor RMSE is small and CUDA-vs-CPU primitive parity was tight, the next useful split is exporter/profile/runtime trajectory: compare against a known-good external GGUF, or add a CPU-vs-CUDA snapshot diagnostic that avoids the slow native generation path.

## Current Hypotheses

Most likely:

1. A Gemma4 attention trajectory/runtime semantic mismatch in the real session path. The clean last-row trace points to layer-0 attention output after healthy q/k/v prep. The leading split is now: prior-token K/V row trajectory vs attention-core behavior vs trace/materialization side effect.
2. An artifact/runtime interoperability issue specific to our GGUF export/loader path, still best split with an external known-good GGUF and/or llama.cpp cross-run.
3. Q4_K is too lossy under the current all-projection quantization policy for this Gemma4 model. Q4_K should stay deferred until Q8 has a healthy baseline.

Less likely now:

1. CUDA visibility or driver issue.
2. Tokenizer/config mismatch.
3. Fused QKV CUDA kernel issue.
4. Direct Q8_0 CUDA matmul primitive bug.
5. Norm/scalar tensors accidentally quantized.
6. Bad exported `output_norm.weight`.
7. Embedding/head quantization as the sole cause.
8. Session-level GGUF weight-slot misbinding for layers 0-5.
9. First divergence later than layer 0.

Still possible:

1. Our Q8_0/Q4_K quantizer differs enough from llama.cpp/ggml quantization quality to matter, though the layer 0-5 bound-weight audit is much cleaner than the observed activation/logit failure.
2. Gemma4 needs a model-specific quantization profile, but this should be validated with KL/top-1 eval and external GGUF before guessing layer exclusions.
3. CUDA quantized execution still has a full-stack bug not covered by current layer0/layer5 parity. A native CPU Q8 generation attempt died without producing a token, so CPU-vs-CUDA full generation remains inconclusive.

## Recommended Next Steps

1. Add a focused layer-0 attention bisection that compares prompt-row K/V and attention scores/probabilities without perturbing lazy CUDA execution. The all-rows trace is useful but changes Q8 top1, so treat it as a lead, not final proof.
2. Run same-input layer-0 graph parity through the real session graph: feed BF16 captured hidden/attn_norm/Q/K/V into the Q8 session attention path and vice versa to split inputs from attention op semantics.
3. Compare our Q8_0 export against an externally produced known-good ggml/llama.cpp GGUF if available; this remains the fastest way to distinguish artifact/export issues from runtime issues.
4. Run or add a full-stack Q8 CPU-vs-CUDA diagnostic that avoids the slow native generation path but compares selected prompt snapshots against CUDA.
5. If our export/policy is the issue, implement a Gemma4 quantization profile:
   - keep selected sensitive layers dense/BF16 or use a safer format
   - avoid revisiting Q4_K until Q8 is healthy
   - test layer windows rather than only layer0
6. Clean disk before more exports. The layer0-dense artifact is useful as evidence but can be removed once its result is no longer needed.

## Useful Commands

CUDA smoke:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info --smoke
```

Full BF16 control:

```bash
ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN=1 \
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/google/gemma-4-12B-it \
  "Ants" \
  --backend cuda \
  --combined-budget-mb 22000 \
  --backend-budget-mb 19000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 1 \
  --temperature 0 \
  --print-token-ids \
  --print-chat-template-status
```

Q8 normal generation:

```bash
ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TRACE=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TOP_K=10 \
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/google/gemma-4-12B-it-q8_0 \
  "Ants" \
  --backend cuda \
  --backend-budget-mb 14000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 1 \
  --temperature 0 \
  --print-token-ids \
  --print-chat-template-status
```

Q8 layer0-dense generation:

```bash
ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TRACE=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TOP_K=10 \
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/google/gemma-4-12B-it-q8_0-layer0dense \
  "Ants" \
  --backend cuda \
  --combined-budget-mb 22000 \
  --backend-budget-mb 15000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 1 \
  --temperature 0 \
  --print-token-ids \
  --print-chat-template-status
```

Q8 cross-layer diagnostic:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-layer0 \
  .models/google/gemma-4-12B-it-q8_0/model.gguf \
  .models/google/gemma-4-12B-it
```

Q4_K cross-layer diagnostic:

```bash
zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-cross-layer0 \
  .models/google/gemma-4-12B-it-q4_k/model.gguf \
  .models/google/gemma-4-12B-it
```

Q8 projection lazy-dequant diagnostic:

```bash
TERMITE_CUDA_DEQUANTIZE_QUANT_WEIGHTS=1 \
ANTFLY_INFERENCE_CUDA_TEMP_CACHE_MB=512 \
ANTFLY_INFERENCE_CUDA_LAZY_PROFILE=1 \
ANTFLY_INFERENCE_CUDA_PREFILL_FIRST_TOKEN=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TRACE=1 \
ANTFLY_INFERENCE_GENERATE_FIRST_TOKEN_TOP_K=10 \
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/google/gemma-4-12B-it-q8_0 \
  "Ants" \
  --backend cuda \
  --host-budget-mb 6000 \
  --combined-budget-mb 26000 \
  --backend-budget-mb 14000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 1 \
  --temperature 0 \
  --print-token-ids \
  --print-chat-template-status
```

Disable fused QKV during diagnostics:

```bash
ANTFLY_CUDA_DISABLE_FUSED_QKV=1
```

Disable head-norm/RoPE fusion during diagnostics:

```bash
ANTFLY_CUDA_DISABLE_HEAD_NORM_ROPE_FUSION=1
```
