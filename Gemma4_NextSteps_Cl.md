# Gemma4 CUDA Next Steps — Critical Reassessment and Plan

Last revised: 2026-06-11
Authors: AI research/engineering review of `Gemma_Cuda_status.md` (2026-06-11) and `Gemma4_Cuda_claude_plan.md` (2026-06-10), plus targeted code reading of `session_factory.zig`, `gpt.zig`, `native_export_gguf.zig`, and `cuda_info.zig`.

Goal: fast, correct inference of Gemma4 12B-it on a 16 GB-class L4 budget in full (BF16 control), Q8_0, and Q4_K formats, with a production-ready path (named profiles, eval metrics, CI, perf).

---

## STATUS UPDATE — 2026-06-11: Phase 0 executed, overlay hypothesis falsified

Phase 0 was implemented and run (`ANTFLY_INFERENCE_PRINT_EFFECTIVE_CONFIG=1` dump, `ANTFLY_INFERENCE_DISABLE_GGUF_CONFIG_OVERLAY=1` kill-switch, `cuda-info --gguf-meta` audit tool). Results:

- **Effective configs match.** BF16 safetensors and Q8 GGUF sessions agree on all structural Gemma4 scalar fields (sliding pattern, rope theta/local theta, KV heads, global head dim, rope active dim, …).
- **Overlay kill-switch did not fix Q8.** Same bad blank/control-token class with the overlay disabled.
- **One real round-trip bug found, currently masked:** GGUF metadata reconstruction yields `rope_partial_factor=0.5` vs the correct `0.25`; the overlay doesn't apply that field so the runtime stays correct *when the config.json sidecar is present*. A bare-GGUF load (no sidecar — i.e., any standalone deployment of the artifact) would run with the wrong value. **Must fix in the exporter/reader regardless** (`rope.dimension_count` write or its read-side derivation). It is also an existence proof that the metadata write→read round-trip has defects, which raises the prior on a sibling defect in tensor handling.

**Hypothesis #1 (config/overlay divergence) is falsified at the scalar-config level.** The revised leading suspect is its sibling, previously ranked #3:

### Revised leading hypothesis: session-level GGUF tensor **binding/wiring** divergence

The decisive contradiction in the evidence: layer0/layer5 pass parity and cross-drift **when loaded by the standalone diagnostics' own loaders**, yet the *session* trajectory is already ~30% off in l2 by layer5. The diagnostics fetch GGUF tensors with their own name mapping; the session binds GGUF tensors into graph slots through a different path. A misbinding there — swapped `ffn_gate`/`ffn_up`, swapped q/k norms, wrong pre/post-norm assignment, a per-layer tensor silently missing and defaulting — explains every observation at once: per-tensor parity perfect, configs identical, CPU==CUDA (both consume the same wrong wiring), dequantization useless, layer0-dense useless, structural trajectory blowup, and BF16 healthy (the safetensors session uses entirely different name mapping).

### New cheapest-decisive test → run before the rest of Phase 1

**Phase 1, Step 0 — Weight-binding audit (zero disk, minutes per run).** For every graph weight slot in the *built session* (`layer3.ffn_gate`, `layer3.post_attn_norm`, tied head, …), print shape + l2 norm + mean-abs + first-k dequantized values, for both the Q8 GGUF session and the BF16 safetensors session, and diff slot-by-slot. Q8 noise is ~0.5%, so any slot whose statistics differ materially is misbound; a swap appears as two slots with exchanged norms. This audits the binding layer that every existing diagnostic bypasses. If it comes back clean, proceed to Phase 1 steps 1–2 (first-divergent-layer cosine trace, same-input single-layer parity through the real session graphs) to localize the divergence to a specific op within layers 1–5.

## STATUS UPDATE — 2026-06-11: Phase 1 Step 0 executed, binding hypothesis falsified for layers 0-5

Implemented `compare --weight-binding-audit` and ran it through the built session binding path (`gpt_arch.getModelWeight`, including `weight_prefix` and generation-style fallbacks). CUDA shape/name audit was clean but CUDA cannot export bf16/quantized weight tensors to f32 today, so the numeric audit was run with both sessions on the native backend, which can export dense safetensors and dequantized GGUF handles.

Result for Q8 GGUF vs BF16 safetensors, layers 0-5:

- Full log: `/tmp/gemma4-weight-binding-audit-layer6.log`
- Totals: `ok=84`, `suspect=0`, `skipped=2`
- Skips: huge embedding table and tied head, intentionally shape-only
- Norms/scales: exact
- Attention/FFN matrices: `rms_rel` typically `0.00003-0.00005`
- Layer 5 full-attention `v_proj`: omitted as expected by Gemma4 config
- Naming alias observed but healthy: Q8 uses `per_layer_input.layer_output_scale.weight`, BF16 resolves through `layer_scalar`; value matches exactly

Conclusion: the revised leading hypothesis — session-level GGUF tensor slot misbinding — is falsified for the early layer window where the hidden-state evidence said divergence had already appeared. Continue with Phase 1 Step 1/2: first-divergent-layer cosine/rel-RMSE trace through the real session graphs, then same-input single-layer graph parity for the identified layer/op.

## STATUS UPDATE — 2026-06-12: Phase 1 Step 1 executed, divergence starts in layer-0 attention

Implemented `compare --activation-trace` with real-session activation capture from `gpt.zig`. It runs Q8 GGUF and BF16 safetensors sequentially, captures last-row activations from the actual forward path, and reports cosine/rel-RMSE. Optional `--activation-trace-layer N` captures detailed internals for one layer; `--activation-trace-all-rows` captures full prompt-row tensors but can perturb lazy CUDA materialization.

Layer-output CUDA trace on `Write one sentence about ants.`:

- Prompt/tokenization matched: 19 tokens
- Q8 top1: `258882`; BF16 top1: `45518`
- `input`: `cosine=0.999984722`, `rel_rmse=0.005529` — healthy
- `layer0.out`: `cosine=0.910530952`, `rel_rmse=0.437977` — first suspect
- `layer5.out`: `cosine=0.091752563`, `rel_rmse=1.463662`

Detailed layer-0 last-row trace:

- Healthy through attention prep:
  - `attn_norm`: `rel_rmse=0.005058`
  - `q_raw`: `rel_rmse=0.006971`
  - `k_raw`: `rel_rmse=0.006154`
  - `v_norm`: `rel_rmse=0.009022`
  - `q_attn`: `rel_rmse=0.007702`
  - `k_attn`: `rel_rmse=0.006849`
  - `v_attn`: `rel_rmse=0.009022`
- First clean suspect: `attn_out`, `cosine=0.827528788`, `rel_rmse=0.561487`
- `layer0.out` remains divergent: `cosine=0.910530952`, `rel_rmse=0.437977`

All-rows layer-0 trace:

- `--activation-trace-all-rows` changed Q8 top1 from `258882` to `181812`, so it likely perturbs lazy CUDA graph/materialization and should be treated as a lead rather than final proof.
- It showed full prompt-row `q_attn` divergence (`cosine=0.704349288`, `rel_rmse=0.996879`) while `q_raw`/`k_raw` stayed healthy, suggesting the next split should inspect Q head-norm/scale/row materialization and prior-token K/V rows without changing execution behavior.

Conclusion: Phase 1 Step 1 is complete. The failure is layer 0, not accumulated quant noise over many layers. The cleanest next action is a non-perturbing layer-0 attention bisection: same-input graph parity for attention, prompt-row K/V and attention-score diagnostics, and an explicit check for whether trace/materialization changes CUDA lazy execution.

> Note on the GPU budget: `nvidia-smi` reports ~23 GB visible VRAM (L4 is a 24 GB card), but this plan budgets to a **16 GB resident target** as requested — that gives real headroom for KV/scratch/fragmentation on the 24 GB card and keeps the artifacts viable on genuinely 16 GB GPUs. Where the two budgets diverge, both are called out.

---

## Part 1 — Critical Reassessment of the Evidence

### 1.1 The leading hypothesis in the status doc is quantitatively implausible

The status doc's current best explanation is: *"Q8_0 per-tensor error is small, but hidden-state/logit trajectory drift over 48 layers is enough to break first-token logits."*

We should treat this as **almost certainly wrong**, for three reasons:

1. **Q8_0 is empirically near-lossless everywhere else.** In llama.cpp and every published quantization study, Q8_0 (8-bit, per-32 block scales) produces perplexity deltas under ~0.1% and first-token agreement with BF16 in the high 90s percent, including on Gemma-family models, which are notorious for activation outliers and still survive Q8_0 fine. A 0.55% weight rel-RMSE simply does not produce the observed failure.
2. **The observed divergence is structural, not noise-shaped.** The sequential compare shows top-10 overlap of 0/10, Q8's top-1 ranked ~250,000th in BF16's logits, and BF16's top-1 ranked ~52,000th in Q8's. The prompt snapshots show `layer5_out` l2 already 30% off (9,598 vs 7,362), `layer23_out` l2 off by 4x (12,816 vs 51,114), and `final_norm` l2 off by 4.6x (152,548 vs 33,273) despite similar `pre_final_norm` magnitude. Independent ~0.5% weight noise per layer cannot bend the trajectory that hard that early. Something is computing a *different function*, not the same function with noisy weights.
3. **The dequant experiments already falsify "quantized values are the problem."** With *all* projections AND the embedding/head lazy-dequantized to f32 — i.e., running effectively a dense model whose weights are within 0.55% of BF16 — the first token is still garbage. If the quantized values were the cause, this run should have been nearly healthy. It wasn't. The cause therefore lives somewhere that dequantization does not touch.

### 1.2 What variable is shared by every failing run and absent from every passing run?

| Run | Load path | Result |
|---|---|---|
| BF16 safetensors, CUDA | safetensors session (config.json) | ✅ plausible |
| Q8_0 GGUF, CUDA | GGUF session | ❌ garbage |
| Q8_0 GGUF, lazy-dequant projections (f32 math) | GGUF session | ❌ garbage |
| Q8_0 GGUF, lazy-dequant projections + embedding (f32 math) | GGUF session | ❌ garbage |
| Q8_0 layer0-dense GGUF | GGUF session | ❌ garbage |
| Q4_K GGUF | GGUF session | ❌ garbage (worse) |

Every failing configuration goes through the **GGUF session-construction path**; the only passing configuration does not. Quantization format, dequantization, fused QKV, layer0 density — all varied with no effect on the outcome class. The load path never varied.

### 1.3 Code reading confirms the GGUF path can build a *structurally different model*

Targeted reading of the load paths found a concrete mechanism, not just a suspicion:

- **The GGUF session merges two config sources.** `session_factory.zig:1752-1790` loads HF `config.json` first (the quantized model dirs symlink config/tokenizer back to the BF16 dir), then calls `detectArchitectureFromGguf()` and **overlays** GGUF header metadata on top via `overlayGptStructuralConfig()` (`session_factory.zig:1924-1967`). The safetensors session uses config.json alone. So the two sessions agree *except* for whatever the overlay changes.
- **The overlay touches exactly the Gemma4 structural fields** that would produce this failure signature: `num_kv_shared_layers`, `global_head_dim`, `num_global_key_value_heads`, `shared_layer_intermediate_size`, `sliding_window_pattern`, `rope_local_theta`, `ple_hidden_size`. Several overlay conditions are fragile, e.g. `if (source.sliding_window_pattern != 6) target.sliding_window_pattern = ...` and the `rope_local_theta` condition `(source != 10000.0 || target == 10000.0)` — if the GGUF-derived config has a default-ish or mis-parsed value, it silently overwrites the good config.json value. A wrong sliding-window pattern or local/global rope theta assignment changes attention structure on most of the 48 layers — precisely the kind of error that leaves weights pristine, keeps CPU and CUDA in perfect agreement with each other (both consume the same wrong config), and wrecks the trajectory by mid-stack.
- **The GGUF metadata reader collapses per-layer arrays.** `gpt.zig:1100-1128` reads only the *first element* of per-layer `attention.head_count_kv` and `feed_forward_length` arrays that the exporter writes (`native_export_gguf.zig:4624-4691`), then reconstructs Gemma4's heterogeneity from other keys (`gpt.zig:1184-1193`). Any asymmetry between what export writes and what load reconstructs lands in the overlay.
- **The cross-layer0 diagnostic is blind to all of this.** `cuda_info.zig:1049-1057` hardcodes hidden size, head dims, eps, and `theta = 10000.0` and runs *both* the GGUF and HF weights through that same hardcoded forward. It validates weights in isolation — which is exactly why it reports small, quantization-noise-shaped drift while real generation is broken. Every "weights look fine" result in the status doc is consistent with a config bug; none of them tests against one.

### 1.4 Revised hypothesis ranking

1. **(New, most likely) GGUF-session effective-config divergence.** The overlay/metadata round-trip gives the GGUF session different structural hyperparameters (sliding-window pattern, local-vs-global rope theta, shared-KV layout, global head dims) than the safetensors session. Explains *every* observation, including the ones the old hypotheses couldn't (dequantized-weights-still-bad; CPU==CUDA on the same artifact; structural l2 blowups).
2. **Localized tensor corruption missed by sampling.** The RMSE sweep sampled 13 of 262k embedding rows and row-sampled all FFN tensors. A layout/transpose/padding bug on a *subset* of blocks or rows could hide from sampled RMSE. Less likely (cross-parity also probed the specific bad token's row), but cheap to close out.
3. **GGUF runtime graph wiring difference** not captured by config (e.g., a per-layer branch keyed off GGUF-only state). Same diagnostic as #1 catches it.
4. **Quantizer quality (Q4_K only).** Q4_K layer0 drift (`mean_abs` ~1.4 on q/k/v vs ~0.09 for Q8_0, `ffn_gated max_abs` 624) is far beyond healthy 4-bit noise — and the prior review already found a real codec bug (payload quantized against *unrounded* sub-scales while dequant uses the rounded packed metadata, `quant_codec.zig`). Q4_K has its own genuine quality problem **in addition to** whatever breaks Q8. But Q4_K shares the GGUF load path, so fix the load path first; the Q4_K codec fix is Phase 3.
5. *(Demoted)* Accumulated Q8 trajectory drift, CUDA kernel bugs, tokenizer issues, norm/scalar mis-quantization — all either falsified or quantitatively insufficient.

---

## Part 2 — The Plan

Phases are ordered by information-per-dollar: each early phase is cheap, needs no new 12 GB artifacts, and can fully redirect everything after it.

### Phase 0 — Effective-config bisection (zero disk, ~1 day) ← do this first

Goal: prove or kill hypothesis #1 with direct evidence instead of more weight-space forensics.

1. **Dump and diff the effective config.** Add an env-gated dump (e.g. `ANTFLY_INFERENCE_PRINT_EFFECTIVE_CONFIG=1`) that prints the final `Config` struct — every field, especially `rope_theta`, `rope_local_theta`, `sliding_window`, `sliding_window_pattern`, `num_kv_shared_layers`, `global_head_dim`, `num_global_key_value_heads`, `num_key_value_heads`, `attention_head_dim`, `rope_partial_factor`, `norm_eps`, `final_logit_softcapping`, intermediate sizes — at session build time. Run it once for the safetensors session and once for the Q8 GGUF session; diff the two dumps. **This single diff likely names the root cause.**
2. **Overlay kill-switch experiment.** Add a gate (e.g. `ANTFLY_INFERENCE_DISABLE_GGUF_CONFIG_OVERLAY=1`) that skips `overlayGptStructuralConfig()` so the GGUF session runs on pure config.json (the symlinked sidecars make this well-defined). Re-run Q8 generation on the two status-doc prompts. If first tokens become sane → root cause confirmed, jump to Phase 2. This is the decisive experiment; the config dump is the explanation.
3. **GGUF header audit.** Add/extend a metadata dump (`cuda-info --gguf-meta <file>` or similar) listing every KV the exporter wrote for the Q8 artifact, side-by-side with what `parseGgufMetadata()` reconstructs from it (the write→read round-trip, including the per-layer array collapse at `gpt.zig:1100-1128`). Compare against config.json ground truth.
4. **Fix the blind diagnostic.** Make `--gemma4-cross-layer0` derive hyperparameters from the actual loaded configs (both sides) instead of the hardcoded block at `cuda_info.zig:1049-1057`, and have it *print* the hyperparameters it used per side. This converts our main weights diagnostic into one that can also catch config drift, permanently.

Exit criterion: we can state, with a config diff in hand, either "the GGUF session was computing a structurally different model — field X" or "effective configs are bit-identical; hypothesis #1 is dead."

**Decision table:**

| Phase 0 outcome | Conclusion | Next |
|---|---|---|
| Configs differ; overlay-disable fixes Q8 tokens | Export/overlay metadata bug | Phase 2 (fix + harden), skip Phase 1 |
| Configs differ; overlay-disable does *not* fix | Config bug plus something else | Fix config, re-run Phase 0, then Phase 1 |
| Configs identical, Q8 still bad | Hypothesis #1 dead | Phase 1 |

### Phase 1 — Close the remaining bisection gaps ← ACTIVE (Phase 0 did not resolve; see status update above)

0. **Weight-binding audit (do first — see status update).** Slot-by-slot statistics diff of bound weights between the Q8 GGUF session and the BF16 safetensors session, auditing the session's GGUF→graph-slot binding that all standalone diagnostics bypass.
1. **First-divergent-layer trace with direction, not just magnitude.** Extend the existing snapshot compare to record per-layer hidden-state **cosine similarity** and rel-RMSE between the Q8 GGUF session and the BF16 safetensors session on the same prompt (sequential, budgeted, like `compare --sequential`). l2 norms hid the story; direction will pinpoint the first layer/op where the trajectory breaks. The layer5 30% gap says divergence starts in layers 1–5 — that's a small search space.
2. **Same-input single-layer graph parity.** Feed an *identical* captured input vector through layer N of both sessions (not the standalone hardcoded CPU re-implementation — the real session graphs). This isolates graph wiring + config with weights held near-equal, the exact complement of cross-layer0.
3. **Exhaustive (non-sampled) error sweep for the first-token-critical tensors.** Full `token_embd.weight` (stream it; don't materialize 4 GB — accumulate per-row error online) and per-*block* max-abs-error histograms for the layers identified in step 1. We're hunting localized corruption (a transposed slab, a bad block range) that sampled RMSE averages away. Report worst-1000 rows/blocks, not means.
4. **Fix native CPU Q8 generation.** It "died without producing a token" — that's an unresolved bug *and* a missing reference point. A working CPU full-stack run on the Q8 artifact cleanly splits artifact vs CUDA-runtime for the whole stack, not just layers 0/5.

### Phase 2 — External cross-validation and the definitive exporter split (needs disk)

Run regardless of which branch fixed Q8, before declaring victory — it validates against the wider ecosystem rather than our own loop.

1. **Disk plan first.** Free: `gemma-4-12B-it-q8_0-layer0dense.gguf` (+12 GB, its evidence is captured), Q4_K artifact (+~7 GB, Q4_K work is deferred to Phase 3 and re-export is cheap), spare mmproj variants. Target ≥ 26 GB free.
2. **Known-good external GGUF, both directions.** (a) Run a llama.cpp-produced Q8_0 GGUF of the same checkpoint through our engine (CPU + CUDA) using the symlink-wrapper layout. (b) Run *our* Q8_0 GGUF through llama.cpp. Cheap pre-step: fetch only the header of the remote GGUF (metadata lives at the file start; an HTTP range request suffices) and diff its KVs against ours before downloading 12 GB.
3. **Dense (`--format none`) GGUF end-to-end** if any ambiguity survives: a no-quantization GGUF that still fails indicts the export/load path with zero confounders; one that passes exonerates it completely.

| Result | Meaning |
|---|---|
| llama.cpp Q8 works in our runtime; ours fails in llama.cpp | Our exporter/quantizer/metadata is bad |
| llama.cpp Q8 fails in our runtime; ours works in llama.cpp | Our GGUF loader/runtime is bad |
| Both work cross-wise after the Phase 0 fix | Fixed; metadata round-trip was the whole story |
| Dense GGUF fails | Export/load path bug independent of quantization |

### Phase 3 — Q4_K codec quality (after Q8 is healthy)

Q4_K has known, concrete defects independent of the load-path bug. Once Q8 is the healthy baseline:

1. **Fix the rounded-scale mismatch** in `quantizeQ4_KBlock` (`gguf/quant_codec.zig`): quantize payload nibbles against the *reconstructed* `d * scs[sub]` / `dmin * mins[sub]` that dequantization will actually use, not the pre-rounding floats.
2. **Port an iterative scale/min search** (`make_qkx2_quants`-style) — the single-pass min/max quantizer leaves large 4-bit error on the table.
3. **Golden tests:** fixed deterministic input blocks, round-trip RMSE bounds for Q4_K and Q8_0, small reference vectors checked in (no large fixtures). Re-run `--gemma4-cross-layer0` and require Q4_K layer0 drift within ~2–4× of Q8_0, not 15×.
4. **Named Gemma4 Q4 profile** (see 4.2) with sensitive-tensor exceptions, constrained to CUDA-executable formats (Q4_K/Q8_0/dense only — no Q5_K/Q6_K until kernels exist).

### Phase 4 — Productize correctness: profiles and a real quality metric

1. **Replace first-token equality with a KL-divergence eval.** Add a small evaluation mode: N prompts (~50–100, mixed chat/raw), teacher-forced next-token distributions from the BF16 control, report mean KL divergence and top-1 agreement of the quantized model. This is the industry-standard quant acceptance metric, it's cheap (one forward per token), and it turns "looks coherent" into a number with a regression threshold. Acceptance gates: Q8_0 mean KL < 0.02, top-1 agreement > 97%; Q4_K mean KL < 0.10, top-1 agreement > 90% (tune after first measurement; the point is *having* gates).
2. **Named exporter profiles, not flag strings.** `--profile gemma4-q8-cuda` / `--profile gemma4-q4-cuda` in `native_export_gguf.zig`, encoding the evidence-based exception list (dense/Q8 for sensitive tensors), CUDA-format constraints, and the metadata KVs — versioned and printed into the GGUF as a provenance KV. Kill hand-maintained `--quantize-exclude` strings.
3. **Round-trip metadata test in CI.** Export a tiny fixture model → load it → assert the effective `Config` struct is field-identical to the source config. This permanently pins the class of bug Phase 0 targets. This is the single highest-value regression test this incident can leave behind.
4. **Diagnostic hygiene.** Promote the gates worth keeping (`compare --sequential`, effective-config dump, budget flags) to documented flags; delete the rest (the `TERMITE_*` lazy-dequant experiment gates, the dirty diagnostic edits in `session_factory.zig`) before merge. The branch must not ship experiment scaffolding as accidental API.

### Phase 5 — Fit and performance on the L4

Start only when a Q8 profile passes the Phase 4 KL gate. Correctness fixes change numerics; perf work before that is rework.

**5.1 Memory budget (the production shape):**

| Component | Q8_0 | Q4_K (fixed) |
|---|---|---|
| Weights resident | ~12.1 GB | ~6.9 GB |
| KV cache (8k ctx; shared-KV + sliding layers help) | ~0.5–1 GB | ~0.5–1 GB |
| Scratch/workspace/cuBLASLt | ~1–1.5 GB | ~1–1.5 GB |
| **Total** | **~14–15 GB** | **~9–10 GB** |

Q8_0 fits a 16 GB budget *tightly* (fine with margin on the actual 23 GB-visible L4); **Q4_K is the comfortable 16 GB production target**, which is why Phase 3 matters and why a mixed profile (Q4_K base + Q8/dense sensitive tensors, ~7.5–8.5 GB) is the likely sweet spot. BF16 (~24 GB weights) remains a streamed correctness control only — never a production serving target on this card.

**5.2 Performance roadmap** (ordered by expected wall-clock return; carried forward from the prior plan, still sound):

1. **Benchmark harness first** — tokens/sec, per-phase timing, kernel/sync/KV-fallback counters; record profile, prompt, budgets, GPU, driver in every result. No kernel change lands without a before/after on the same harness.
2. **Tiled/fused Q8_0 matmul family** — port the Q4_K tiled/fusion approach (`termite_linear_q8_0_f32` is currently scalar, one thread per output); include fused QKV and the LM-head path the chosen profile actually uses.
3. **Device-side greedy argmax** — stop downloading 262k-vocab logits per token; final-logit softcap is monotonic so plain greedy can skip it pre-argmax; fall back to host when sampling/penalties/grammar need mutable logits.
4. **KV-cache device-path health** — measure `device_kv_fail_*` counters before touching code; eliminate host round-trips per token.
5. **Sync/launch reduction**, then CUDA graph capture for steady-state decode once kernels stabilize.
6. **Int8 activation path (`__dp4a`/Q8_1 activations)** — last, because it changes numerics; must re-pass the KL gate.
7. **Flash-style prefill attention** — later pass; decode latency dominates the serving target.

Every `.cu` change regenerates `inference_cuda_kernels.ptx` and extends `cuda-info --smoke` parity coverage, and must hold the Phase 4 KL gate.

### Phase 6 — Production hardening

1. CI smoke: tiny-fixture quantize → load → config round-trip assert → deterministic generate with top-k sanity; CUDA smoke where GPUs are available.
2. Document the known-good profiles, L4 budget recipes (both 16 GB and 23 GB variants), and the eval gates in-repo.
3. Branch cleanup: productize-or-delete every diagnostic added during this investigation; commit history currently carries `wip` commits and dirty working-tree diagnostics (`session_factory.zig`, `compare_generate.zig`, `cuda_info.zig`) that need intentional disposition before merge to main.

---

## Part 3 — Success criteria

- **Correctness:** Q8_0 and Q4_K pass the KL-divergence gate vs BF16; CPU and CUDA agree per artifact; deterministic outputs coherent on the status-doc prompts; effective-config round-trip test green in CI.
- **Fit:** Q8_0 ≤ 15 GB resident @ 8k ctx; mixed-Q4 profile ≤ 10 GB; no lazy/streamed weights in the production profile.
- **Speed (L4, batch 1, after Phase 5):** target ≥ 15–20 tok/s decode for resident Q8_0 and ≥ 25–30 tok/s for Q4-mixed as initial bars; refine after the first harness baseline — the harness, not the guess, sets the contract.
- **Process:** no hand-rolled exclude strings, no undocumented env gates, every quant artifact stamped with its profile version.

## Part 4 — Risks

- **Phase 0 finds identical configs.** Then hypothesis #1 dies cheaply (a day) and Phase 1's direction/graph-parity tools take over — they were the prior plan's gap anyway. Nothing is wasted; the diagnostics built are permanent.
- **Disk (7.3 GB free)** blocks Phase 2 artifacts until cleanup; the plan sequences deletions only after their evidence is captured.
- **llama.cpp interop ambiguity:** our GGUF may legitimately differ in conventions (per-layer arrays) without being *wrong*; treat llama.cpp results as evidence, not as the spec — the config round-trip test is the spec.
- **16 GB target vs 23 GB card:** Q8-only on a true 16 GB device has thin scratch headroom; if Q4-mixed slips, the fallback is documenting Q8 as 24 GB-card-only.
- **Perf work invalidating correctness:** every numerics-touching optimization (dp4a, fusion) re-runs the KL gate; the gate is the contract that lets perf work proceed safely.

## Part 5 — Immediate next actions (first working session)

1. Implement the effective-config dump and run the safetensors-vs-GGUF diff (Phase 0.1).
2. Implement the overlay kill-switch and re-run Q8 generation (Phase 0.2).
3. Capture GGUF header KVs for the Q8 artifact and diff the write→read round-trip (Phase 0.3).
4. Based on the diff: either fix the metadata/overlay bug and re-validate, or pivot to Phase 1 with the divergent-layer cosine trace.
