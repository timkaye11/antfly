# Spec-decode P3 Hand-off: route the multi-row VERIFY forward onto the planned decoder frame

## OUTCOME (2026-07-09, commit 46bdd65dd) — DONE, premise was stale, real fixes landed

**Phase A verdict: the routing premise below was WRONG at this HEAD.** The multi-row verify
already plans AND executes as one planned frame (`prefill_plan=15/15, prefill_execute=15/15`,
`small_batch=126 mm=84` per verify frame). The doc's `small_batch=0` trace was taken through the
dedicated-runtime path and/or misattributed the draft model's `dense_linear` fragments (the doc
itself flagged that risk). The measured 122ms verify decomposed as:
- **~55ms lm_head+argmax tail** — `linearNoBiasArgmaxRowsSuppress` at rows 2-3 × vocab 262144
  collapses in the multi-row small-batch kernel (both dense-f16 AND prepared q8_0 slots ~52ms).
  FIXED: prepared final-lm-head slot + per-row mmv + batched argmax → ~13ms @2 rows; op-level
  per-row split for rows 2-8 × out_dim≥32768 protects all other callers.
- **~62ms GPU frame** (vs 34ms rows-1) — q4_0 layer linears ride reduce-family kernels whose
  grid scales with rows (~27ms marginal/row). NOT FIXED — this is the remaining lever.
- **~3ms host** (embed 0.3, PLE 1.2, plan 5µs cache-hit, encode 1.4).

Also fixed: **dedicated-runtime verify commit recomputed logits through the dense fallback each
round** (55-81ms, ran on every fold-on round since `ANTFLY_GEMMA4_MTP_DEDICATED_RUNTIME`
defaults true) — now skipped when prepared choices exist. And a **critical env-cache bug**:
`metal_compute.getenvBool` / `metal_runtime.getenvFlagValue` shared ONE cache across all env
names (nested struct didn't reference the comptime param) — every TERMITE_METAL_* toggle read
through them was order-dependent; re-run any A/B that relied on those toggles.

**Results (E4B QAT, 96 tok, force k=2+bonus fold-off):** verify 122→84ms, decode 12.1→16.3 tok/s
(plain 21.5 same session). Fold re-A/B: fold-on 6.7→12.2 but fold-off still wins (fold-on's
rows-3 verify pays the frame's per-row scaling). Bonus now default-on for Metal. Gates:
POLICY_ONLY exit 0, defer-materialize gates pass, suite 2085/0/12, token ids bit-identical.

**Remaining endgame lever (next handoff):** the frame's per-row scaling. rows-1 34ms → rows-2
64ms → rows-3 89ms. q4_0 small_batch r2/r4/r8(+wide) shared-read kernels exist for plain f32
linears but the pair-activation (gate_up), activation_rhs (PLE) and f16-activation classes have
only per-row reduce variants. Fix that and: verify @2 rows ≈ 50ms total, fold-on flips to a win
(kills the 41ms materialize), spec ≈ 24-27 tok/s > plain. Note `TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH=1`
(generated kernel) measured SLOWER (9.9 vs 16.3 tok/s) — hand-written r2_wide stays.
Diagnosis tooling: `ANTFLY_GEMMA4_MTP_VERIFY_TRACE=1` prints per-verify stage/tail/round traces.

---

Workdir `zig/pkg/inference/`, branch `codex/quant-kernel-metal-compiler`. Zig 0.16.0.
Base: HEAD including the P2 fold commit. `file:line` anchors verified on that tree but drift — re-grep.
Memory of record: `~/.claude/.../memory/project_gemma4_spec_decode_fix.md`. Prior hand-offs in-tree: P2 (fold, landed default-OFF), P4 (draft overhead, queued).

## Why this is THE endgame lever

Committed arc so far: crash fix (`396c0cc01`, unblocked k=2 + bonus), batched verify argmax (`5c77099f2`), P2 fold (landed, default-OFF per measured verdict). Best config: **k=2+bonus fold-off ≈ 13.2 tok/s vs plain 24.3**.

**The measured facts that define this task:**
- Verify (q_len=2) costs **139ms/call**, marginal row **~57ms** (196ms @ 3 rows). A planned rows-1 decode step is ~41ms; the rows-2 quant matmul floor is **1.05×** (microbench: rows2 0.241ms vs rows1 0.230ms per op) — so a planned rows-2-3 verify should cost ~45-65ms, not 139-196ms.
- Frame trace (`TERMITE_METAL_TRACE_FRAME=all` on a spec run): only the rows=1 forwards run as planned frames (129 ops, `quant_dispatch mmv=43`); the multi-row verify executes as **dozens of tiny generic per-op frames** (`region: other total=14 rms_norm=4 dense_linear=4`, `attention total=2 head_rope=2`), and **`small_batch=0` for the entire run** — the rows-2-8 quant kernels never fire in the path they were built for.
- **Payoff:** verify(2-3 rows) at ~1.1-1.6× plain step makes the already-built stack (fold + bonus + batched argmax + k=1/2) land ~24-31 tok/s → beats plain (24.3) and llama (~30). It also flips P2's economics: **re-run the fold A/B after this lands** (`scripts/test_gemma4_mtp_defer_materialize.sh` + the profile A/B; the fold's plumbing is gated and waiting).

## The routing chain (code-verified by an explorer — re-verify the starred inference)

1. `speculativeDecodeGemma4Mtp` (generation.zig:~5599) → verify via `forwardMtpTargetHiddenDevice(tokens[verify_start..verify_seq], …)` (:~5855).
2. → `forwardMtpTargetHiddenDeviceReplayMode` (:~4963-4981) → Metal branch (:~5023-5040): `decoder_gated_runtime.forwardFinalHiddenRows(...)`; **null → generic `gpt_arch.forwardFinalHiddenTensorFromEmbeddingsWithLayer0Overrides` fallback** (:~5042-5074) — the fragment path in the trace.
3. `forwardFinalHiddenRows` (decoder_gated_runtime.zig:~6063): gates on `supportsConfig`, `query_sequence_len == input_ids.len`, `attention_mode ∈ {paged_prefill, paged_decode}` (:~6072-6074). Verify's context builds `attention_mode = .paged_prefill` for q_len>1 (generation.zig:~1657). → `forwardFinalHiddenTensorGemmaDirect` (:~1926).
4. `shouldUseDecoderRuntimeFrame` (:~1783-1805): `.prefill` phase has **no q_len upper gate** → frame goes active for verify.
5. Multi-row plan attempted (:~1984): `decoderRuntimePlanPrefillFrame` when `phase==.prefill and query_sequence_len > 1`. Plan acceptance gates live at metal_compute.zig:~17638-17659 (`rows <= 1` reject, hidden/head checks, **prepared-linear-slot format checks** ~17707-17754, `preparedLinearMatmulFormatForLinearSlot` ~18344).
6. Execution gate (:~2027): `prefill_frame_planned and ple_vectors != null and …` → `decoderRuntimeExecuteGraphCommandPlanFrame` → on success returns planned multi-row hidden (`frame_hidden`, total_rows) — **the planned path DOES produce multi-row final hidden**, exactly what verify needs.
7. **THE TRAP (★ verify this precisely):** when the plan or execution declines, the code falls to per-op `decoderRuntimeApplyLinear` — but with `decoder_frame_active=true`, BOTH `tryApplyQuantizedRuntimeLinear` (metal_runtime.zig:~11691) and `tryApplyDenseRuntimeLinear` (:~12423) hard-return null on `frame_active` → the whole Gemma-direct path returns null → generic interpreter fragments. So a *declined plan* silently costs 3-5× instead of failing loudly.

Real prefill (q_len≫1) succeeds through the same machinery (one planned frame, `mul_mm=210` at 450 tok; small-batch kernels fire via prefill-chunking in E2B configs) — so the planned path handles rows 2-8 **in principle**; verify is failing a specific gate before it.

## Phase A — DIAGNOSE THE DECLINE (do this first; don't guess)
Add temporary (or env-gated) trace to the decline points and run the spec repro:
- `decoderRuntimePlanPrefillFrame` acceptance path (metal_compute.zig:~17638+): log WHICH check rejects for the verify call (rows/hidden/head checks vs the prepared-slot format loop ~17707-17754).
- The execution gate (decoder_gated_runtime.zig:~2027): is `prefill_frame_planned` false (plan declined) or does `ExecuteGraphCommandPlanFrame` itself fail? Is `ple_vectors == null` for the verify context (Gemma4 PLE vectors are computed per-forward — the verify path may not supply them)?
- Repro (~40s): `ANTFLY_GEMMA4_MTP_ACCEPT_BONUS=1 ANTFLY_GEMMA4_MTP_ADAPTIVE_K=0 ./zig-out/bin/antfly-inference generate <e4b> "<prompt>" --backend metal --draft-model <assistant> --speculative-k 2 --speculation-policy force --max-tokens 24 --temperature 0` with `TERMITE_METAL_TRACE_FRAME=all` + your new trace.
- Also confirm which model the `dense_linear` fragments belong to (draft assistant is bf16/dense — they may be draft steps, not verify; don't chase the wrong model's ops).

## Phase B — FIX (ranked by likely correctness; pick per Phase-A findings)
1. **Fix the actual decline cause** so the planned prefill frame executes for verify's q_len 2-3 (e.g., supply `ple_vectors` for the verify forward; prepare/format-register the missing linear slots the plan gate wants — the slots ARE prepared for the rows-1 planned path, so the gap may be a format/capability mark for multi-row). This is the real prize: ONE planned frame per verify, `small_batch>0`.
2. **Make declined plans fail soft, not catastrophically:** when the plan declines with `decoder_frame_active`, END the frame and retry the per-op path with `frame_active=false` (today it cascades to the generic interpreter). Worth doing regardless of #1 as a guard — but it only gets per-op quantized calls (~40 round trips), not the planned frame; measure, don't assume it's enough.
3. **Do NOT** remove the `frame_active` null-gates inside tryApplyQuantized/DenseRuntimeLinear (metal_runtime.zig:~11691/~12423) — that's the WAF-hazard class (generic ops under active frames caused the historical heap corruptions; the discipline exists for a reason).
4. If the planner fundamentally can't do rows 2-3 (Phase A says why), extend it — the quant dispatch selector already routes rows 2-8 to the small-batch kernels; the launch-shape machinery is table-driven (quant-kernel compiler work).

## Acceptance criteria
- Frame trace during a spec run shows **one planned frame per verify call** with `quant_dispatch small_batch > 0` (rows 2-3) — this is the definitive signal.
- Verify wall: 139ms → ≤ ~70ms @ 2 rows (marginal row ≤ ~15ms). Profile via `--json-timing` (`target_verify_ns / target_verify_calls`).
- **Identity gates**: `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1 bash scripts/compare_metal_gemma4_e4b_qat.sh` exit 0; direct token diffs at k=1/2/4 (+bonus, +fold) vs plain — bit-identical. NOTE: moving verify from the generic path to the planned frame changes summation order → argmax is robust in practice, but the gates arbitrate; evaluate any flip, don't ignore.
- Plain decode + prefill untouched (their planned paths share code — re-run the plain golden + a prefill A/B).
- Full suite 0 fail. THEN: **re-run the P2 fold A/B** (its verdict was priced on 139/196ms verifies) and the k-sweep (k=1/2 + bonus + fold) with the cost model (`/private/tmp/spec_cost_model.py`) → pick the new default config; consider bonus default-on for Metal if identity holds (it does today).

## Risks
1. **SEV-HIGH — shared machinery**: the planned prefill frame is production prefill's path. Any planner change must keep real prefill byte-identical (golden gates + prefill perf A/B).
2. **SEV-MED — near-tie argmax flips** from the summation-order change (precedent says fine; gates arbitrate).
3. **SEV-MED — the decline may be structural** (e.g., planner assumes prefill-sized chunks); then this becomes a planner extension, not a gate fix — budget accordingly, and land option-2 (soft-fail) as the interim win.
4. **SEV-LOW — KV semantics**: verify writes KV for its rows via the same suffix write in both paths; the planned frame's KV write ordering is already exercised by prefill.

## Key files
- `src/backends/decoder_gated_runtime.zig` — `forwardFinalHiddenRows:~6063`, `forwardFinalHiddenTensorGemmaDirect:~1926`, `shouldUseDecoderRuntimeFrame:~1783`, plan/exec gates `:~1984/:~2027`.
- `src/ops/metal_compute.zig` — `decoderRuntimePlanPrefillFrame` acceptance `:~17638-17754`, `preparedLinearMatmulFormatForLinearSlot:~18344`, `decoderRuntimeExecuteGraphCommandPlanFrame`.
- `src/backends/metal_runtime.zig` — the `frame_active` null-gates `:~11691/:~12423` (understand, don't remove).
- `src/pipelines/generation.zig` — verify chain `:~4963-5074`, context `:~1642-1671`.
- Gates: `scripts/compare_metal_gemma4_e4b_qat.sh` (POLICY_ONLY), `scripts/test_gemma4_mtp_defer_materialize.sh` (fold re-A/B).
