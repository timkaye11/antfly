# Spec-decode P3c Hand-off: run the MTP verify through the FUSED multi-row path (129-op plan, not the 468-op prefill plan)

Workdir `zig/pkg/inference/`, branch `codex/quant-kernel-metal-compiler`. Zig 0.16.0.
Base: HEAD including `5ed771e97` (P3b pair kernels). Anchors are `fn-name:~line` — re-grep, they drift.
Memory of record: `~/.claude/.../memory/project_gemma4_spec_decode_fix.md` (read the P3/P3b sections — each prior phase corrected its predecessor's attribution; this doc encodes the current, measurement-verified picture).

## The verified structure of the problem (do not re-litigate, but do re-verify Phase-A style)

Committed arc: crash fix `396c0cc01` → batched argmax `5c77099f2` → P2 fold (default-OFF) → P3 `46bdd65dd` (lm_head tail 55→13ms, double-argmax kill, env-cache bug, bonus default-on) → P3b `5ed771e97` (q4_0 pair-activation shared-read kernels rows 2-8). Spec: **16.3 tok/s (k=2+bonus fold-off) vs plain 21.5** on recent thermals; tokens bit-identical everywhere.

**The remaining verify gap is STRUCTURAL, not a kernel class:**
- rows-1 target forwards run the **FUSED decode plan**: 129 ops/frame = 3 megaops/layer (`attention` = qkv+rope+KV-write+paged-attention+out-proj+residual; `ffn_pre_norm_scale`; `ffn_gate_up_activation` = gate+up+activation+down+post-norm+residual+PLE) + 3 tail ops. Frame ≈ **34ms**.
- rows 2-3 verify runs the **UNFUSED prefill plan**: 468 ops/frame (separate qkv/rope×2/v_norm/attention/gate_up/down/norms/PLE×3) ≈ **64ms @2 rows, 89ms @3**. Encode is NOT the bottleneck (plan cache-hit ~5µs, execute/encode ~1.3ms; `ANTFLY_GEMMA4_MTP_VERIFY_TRACE=1` prints these) — the cost is ~340 extra GPU dispatch latencies + unfused small-op inefficiency at tiny rows.
- ⚠️ **The frame-trace `quant_dispatch`/`command_operators` counters are CONTRACT RECORDS, partly hardcoded** (e.g. gate_up recorded as MUL_MM unconditionally at the record site `metal_kernels.m:~33775`). P3b partially mis-scoped off them. Attribute with wall/GPU measurements, not those counters.

**Payoff:** fused verify @2 rows ≈ 40-45ms → verify wall ~50ms incl. tail → spec ≈ 20-24 tok/s (≈/> plain on current thermals). Then **re-run the P2 fold A/B and k-sweep** — the fold (default-OFF) was priced on expensive verifies and likely flips ON, adding more.

## Why this is routing work, not kernel work (the de-risking facts)

1. **The fused multi-token chain already RUNS at rows=4 in production**: the assistant's donated-KV seeding pass executes `runActiveGatedLayerFromAttentionInputDeviceMt` → `runGatedDecoderBlockF32KvDeviceMt` (metal_compute.zig:~16449/~12780) → `termite_metal_decode_runtime_apply_attention_f32_gated_block_quantized_device_kv_device` (metal_kernels.m:~38820) with seq_len=4 (this was the P0 k=2 crash site — *in the multi-row path, since fixed*). The `Mt` (multi-token) plumbing exists end-to-end.
2. **FFN megaop rows 2-8: DONE** — P3b's `termite_q4_0_pair_activation_multiply_r{2,3,4,5}_ext` + the gated-FFN impl (`apply_gated_ffn_residual_q8_0_slots_device_impl`, despite the q8_0 name it handles q4_0 blocks; my rows-2-8 branch at `:~33865`, opt-out `TERMITE_METAL_DISABLE_Q4_0_PAIR_ACTIVATION_SMALL_BATCH`).
3. **Paged attention at q_len 2-8: EXISTS** — `termite_paged_attention_kv_1x` grid is `(q_len, heads)` with per-row positions/causal+sliding masking (flash prefill_sg needs q_len≥8, irrelevant here).
4. **Multi-row tail: DONE (P3)** — prepared lm_head slot + per-row mmv + `argmaxLogitsRowsSuppressDevice` batched argmax already consume a multi-row hidden.

## What P3c must actually do

**Phase A (diagnose-first, ~half a day):** trace precisely why the verify forward routes to the prefill plan:
- `generation.zig:~1642-1671` builds the verify DecodeContext with `attention_mode = .paged_prefill` when q_len>1 (paged_decode only at q_len==1).
- `decoder_gated_runtime.zig` `shouldUseDecoderRuntimeFrame:~1783-1805`: `.greedy/.sampled` phases hard-require `query_sequence_len == 1`; `.prefill` phase → the unfused prefill plan.
- Map the rows-1 fused 129-op plan's build/execute path (the decode-phase lowerer + its executor cases) and confirm which pieces take a `rows` parameter today vs assume 1. Also map what the SEEDING path (rows=4) does differently — it's the existing proof-of-concept for fused multi-row.

**Phase B (the routing):** two candidate designs — pick per Phase-A findings:
- **B1 (likely cheaper): drive the verify through the gated-block Mt chain directly** (the same calls the seeding pass makes), bypassing plan selection: per-layer `runActiveGatedLayerFromAttentionInputDeviceMt`-style loop at q_len 2-8 inside one frame, then the P3 multi-row tail. Needs: all-rows final-hidden output (see below), PLE at rows 2-8 verified, KV suffix write for the verify rows (the seeding pass already writes multi-row KV via the quantized seed path — reuse its discipline).
- **B2: extend the fused decode-plan lowerer to rows 2-8** (a `rows` dimension through the 129-op plan build + executor megaop cases). Cleaner long-term (plan-cached), more surface.
- Either way, gate behind an env flag first (e.g. `ANTFLY_GEMMA4_MTP_FUSED_MULTIROW_VERIFY=1`), A/B, flip default when green.

**The known real gaps (budget these):**
1. **All-rows final hidden**: the fused decode path produces the LAST row's hidden for the lm head; verify needs hidden for rows 0..k (acceptance + pure-prefix reuse + P2's fold row-0). The prefill path returns multi-row hidden (`forwardFinalHiddenRows`); the fused path must expose the same (the residual stream tensor holds all rows — it's an output-plumbing question, not compute).
2. **PLE at rows 2-8** inside the FFN megaop: verify the ple_gate/ple_proj encodes take rows (they may be 1x-only like pair-activation was — same treatment as P3b if so; PLE mats are small, per-row reduce may be fine — measure first).
3. **KV write semantics**: verify rows must land in the reserved slots with correct positions (the unfused path does this via the paged suffix write; the fused attention megaop's KV write at rows>1 is exercised by seeding — confirm position math for the verify offsets).
4. **q6_k etc. non-q4_0 layers** (LM head is tail — done; if any backbone layer is non-q4_0, check its megaop rows support; E4B QAT backbone is q4_0).

## Acceptance criteria
- Frame trace during spec decode shows verify frames with **~129-op-scale fused command counts** (not 468) at rows 2-3; `ANTFLY_GEMMA4_MTP_VERIFY_TRACE=1` wall @2 rows ≤ ~50ms (from 64ms), @3 rows ≤ ~60ms (from 89ms).
- **Tokens bit-identical** to plain at k=1/2/4, bonus on AND off, fold on AND off: `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1 bash scripts/compare_metal_gemma4_e4b_qat.sh` exit 0 + direct 48-tok diffs + `scripts/test_gemma4_mtp_defer_materialize.sh` all pass. Near-tie argmax flips possible (different summation order in the fused ops) — gates arbitrate, evaluate don't ignore.
- Plain decode + real prefill byte-identical and perf-unchanged (shared machinery!).
- Full suite 0 fail. Then: **fold re-A/B + k-sweep** (`/private/tmp/spec_cost_model.py`) → pick the new default config; report spec vs plain vs llama honestly.

## Risks
1. **SEV-HIGH — shared machinery**: the gated megaops serve plain decode (rows-1) and the seeding pass; the plan lowerer serves prefill. Every change kill-switched; plain golden + prefill A/B after each step.
2. **SEV-HIGH — KV/position correctness at multi-row**: silent-wrong-token class. The identity gates + long-generation runs (128+ tok) are the defense; test kv crossing page boundaries (the P0 crash taught us depth>8 matters).
3. **SEV-MED — all-rows hidden plumbing** may touch the frame's output contract; keep the prefill path as fallback (env flag) so a decline degrades to today's behavior, not breakage.
4. **SEV-MED — attribution drift**: this is the third re-attribution of this gap. Phase-A trace FIRST; if the fused path @2 rows measures ≥60ms in a spike, STOP and re-model before building.

## Follow-ups after landing (queued, do not scope-creep into P3c)
- Fold re-A/B + k-sweep + defaults (P5); `SPEC_DECODE_P4_DRAFT_OVERHEAD_HANDOFF.md` (draft 13.2ms/step) for final margin; the pre-existing `kv_compacted` multi-token-append bug (P2 audit) if `cache_compaction_ratio` is ever enabled.

## Key files
- `src/backends/decoder_gated_runtime.zig` — phase/frame gates `:~1783-1805`, direct forward `:~1926`, `forwardFinalHiddenRows:~6063`.
- `src/pipelines/generation.zig` — verify context `:~1642-1671`, verify chain `:~4963-5074`.
- `src/ops/metal_compute.zig` — Mt chain `:~12780/:~16449/:~16652`, decode-plan executor, prefill plan `decoderRuntimePlanPrefillFrame`.
- `src/backends/metal_kernels.m` — gated attention block `:~38820`, gated FFN impl + P3b branch `:~33865`, cosmetic record sites `:~33775` (do not trust).
- Gates: `compare_metal_gemma4_e4b_qat.sh` (POLICY_ONLY), `test_gemma4_mtp_defer_materialize.sh`, `ANTFLY_GEMMA4_MTP_VERIFY_TRACE=1`.

---

## OUTCOME (2026-07-10, implemented — premise re-modeled per Risk 4)

**Phase A overturned all three de-risking premises; the routing designs (B1/B2) were moot:**
1. `runActiveGatedLayerFromAttentionInputDeviceMt` is hard-gated `paged_decode && q_len==1` (metal_compute.zig `:15982`) — it never ran at rows=4. The MTP prompt-seed pass runs on `initContiguous` state → `.full_recompute` → never touches the Mt chain at all.
2. **The verify ALREADY funnels per layer into the fused gated-block C megaop** (`termite_metal_decode_runtime_apply_attention_f32_gated_block_quantized_device_kv_device`): planned prefill executor → `runGatedDecoderBlockOp` paged_prefill leg → `runGatedPrefillFrameLayerQuantized` → `runGatedDecoderBlockF32KvDeviceMt` → megaop. "Route the verify through the fused path" described the status quo.
3. The 468-vs-129 record counts do not measure dispatches: rows-2 own-KV layer = 20 real dispatches vs rows-1 = 19 (both frames ≈ 735-750 dispatches). `mm=84` in the trace is two HARDCODED cosmetic record sites (PLE gate/proj, metal_kernels.m `:26100/:26244`); no MM kernel runs at rows 2-8 (`q4_0_mm_sg` is rows≥9-gated). All matmuls at rows 2-8 already ride shared-read small-batch kernels.

**Measured real cause of the 62.5ms-vs-37.5ms GPU frame gap:** at rows≥2 the frame dispatches `termite_apply_rms_norm_rows` — a THREAD-PER-ROW kernel (one thread serially scanning hidden_size floats twice) — for attn pre-norm, FFN pre-norm (2560 wide, 84 dispatches/frame) and v_norm. At rows=2 these are 2-thread dispatches on the serial planned encoder ≈ pure GPU stalls ≈ ~28ms/frame. (Confirmed by fix A/B, not inference.) Secondary: kill-switch A/Bs showed FFN small-batch −6ms headroom, generic small-batch −5.7ms, pair-activation −1ms; concurrent planned dispatch = no change + token divergence (dependent chains).

**The fix (this commit):** `termite_apply_rms_norm_rows_reduce` — threadgroup-per-row 256-thread reduce (identical summation structure to `termite_apply_rms_norm_1x_reduce`), routed in `termite_metal_dispatch_rms_norm_rows` when `rows 2-8 && hidden_size >= 1024`. The narrow gate keeps plain-decode v_norm (rows=kv_heads over head_dim=256) and real prefill (rows>8) byte-identical. Kill switch: `TERMITE_METAL_DISABLE_SMALL_ROWS_NORM_REDUCE`.

**Results (M4, same-session, oracle prompt, 96 tok):**
- Verify wall: rows-2 **64-69 → 36-41ms** (target ≤50 ✓; now ≈ the rows-1 planned frame itself), rows-3 **89 → 57ms** (target ≤60 ✓), rows-4 64.5ms, rows-5 77ms.
- Plain 24.3 tok/s (warm). Spec: **k1+bonus+fold 17.4** (fold FLIPS to a win at k1: 12.1→17.4), **k2+bonus fold-off 16.9**, k2+fold 15.0, k4 fold-off 15.7. Best ≈ 0.72× plain on these thermals (plain ran 21.5 in the P3 session — parity there is plausible but unverified).
- Identity: tokens bit-identical to plain in ALL arms (k=1/2/4 × bonus on/off × fold on/off, 48-96 tok), POLICY_ONLY exit 0, `test_gemma4_mtp_defer_materialize.sh` pass, plain decode byte-identical vs pre-change binary, prefill identity on long (rows≫8) and tiny prompts.
- k1+fold acceptance = 500‰ (fold's known Option-B staleness penalty, see P2) vs ~700‰ fold-off.

**Follow-up levers (do not re-derive):** (a) verify TAIL is now 13ms of the ~57ms verify: 2× per-row lm_head mmv each re-reading the ~550MB q6_k tied lm_head — a 2-row shared-read mmv would halve it (+~1.5-2 tok/s at k1/k2); (b) draft 12.2ms/step (P4 handoff, draft_argmax 2.9ms is the soft spot); (c) fold acceptance penalty (P2 Option-B stale activation); (d) rows-3 marginal (57 vs 41ms — r_ext row-group scaling: rows 3-4 pay ~2× threadgroup work of rows 2); (e) PLE per-layer slice dispatch+alloc at rows≥2 (~2-4ms/frame, rows-1 gets a free view — hoist to one frame-level pre-slice); (f) paged attention kv_1x rescans full KV per query row — irrelevant at short KV, linear-in-rows at long context.
