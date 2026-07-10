# Spec-decode P4 Hand-off: draft-step + cross-runtime overhead (the post-fold levers)

## OUTCOME (2026-07-10) — DONE. Spec k1+bonus+fold reaches plain parity; the handoff's cost model was largely stale, the real levers were found by `sample`-profiling.

**Results (M4, cooled, interleaved A/B, 128 tok, oracle prompt, all six arms token-IDENTICAL):**
plain 24.9/27.1 tok/s; spec k1+bonus+fold 25.6/27.2 tok/s (**~1.0× plain — was 0.75-0.80× at session start**, 20.3 vs 25.3). Draft 11.9 → **5.0-6.4 ms/step** (assistant 7.1 → **0.45ms**); the one-time MTP seed forward (~0.85s at a 21-token prompt) is GONE. Acceptance 488‰ unchanged in every arm. Gates: POLICY_ONLY exit 0, defer-materialize gates exit 0, suite green.

**What the profiling actually showed (handoff model corrections):**
- The "~2×{2,512} donated-KV backend-cache host clones per step" (lever 1) were STALE — those fire only in the warm-up round. The real draft-step cost: **the donated-KV attention fell off the paged-slot path entirely** — per step it gathered the full donor KV via `MetalKvStorage.gatherLayerKv` (2 private-buffer downloads), flushed the whole 16-op draft frame just to read Q on the host, and ran the host-shim attention on its own command buffer (~5 syncs + a CPU/host cascade that also pushed one rms_norm per layer onto the host path). The `metal_compute.zig` donated branch at the OLD no-frame site was never hit; the active-frame branch (`gqaPagedAttentionOp` skip_kv_write leg) was.
- The draft was already ONE frame per step (the earlier "5 frames/step" reading was five draft STEPS); the serializer was the attention fallback above plus the separate argmax round-trip.
- **Lever 4's "unattributed bucket" was mostly the MTP seed re-forward**: Metal never captured the prefill's last hidden (`prefill_last_hidden` was CUDA-gated), so `replaceGemma4MtpActivationFromPrompt` re-ran the whole prompt through the generic per-op forward (~0.85s at 21 tokens, scaling with prompt length) between prefill and round 1 — invisible in per-round profiling.

**What landed:**
1. **Donated slot attention on the draft frame** (the big one): `termite_metal_decode_runtime_encode_paged_attention_slot_ex` splits slot-owner vs frame-owner runtimes; new `..._attention_paged_slot_device_on_frame` encodes the KV owner's span-slot attention directly onto the draft's active frame (guards: same device, KV owner quiescent, no planned encoder); wired into BOTH donated read sites in `gqaPagedAttentionOp` (the active-frame skip_kv_write leg is the one the draft hits). No draft-frame flush, no KV gather, no host attention. Kill switch `TERMITE_METAL_DISABLE_DONATED_SLOT_ATTENTION_ON_FRAME`. Draft assistant 7.1 → 0.45 ms/step.
2. **Metal prefill prepared-tail greedy + hidden capture** (`tryMetalPreparedTailPrefillGreedy`): when MTP seeding is wanted (pure-greedy, no grammar), the last prefill chunk runs `forwardPrefillLastPreparedTail(prefer_greedy_token=true)` and captures the last hidden row POST-final-norm (matching what verify harvests — note the CUDA capture path hands the MTP a PRE-norm row, a likely latent CUDA inconsistency, left untouched). Kills the seed re-forward. Kill switch `ANTFLY_GEMMA4_MTP_DISABLE_METAL_PREFILL_HIDDEN_CAPTURE`.
3. **Draft argmax frame-tail encode** (`..._apply_linear_argmax_device_frame_encode` + reordered submit in `draftTokenDevice`): folds the lm-head+argmax into the draft frame for one submit+wait per step. Measured NEUTRAL-to-slightly-negative vs flush+own-cb on M4 (both iterations), so **default OFF** — opt-in `TERMITE_METAL_ENABLE_LINEAR_ARGMAX_FRAME_FINAL`. The reorder itself stays (fallbacks self-flush; `argmaxLastRowOp` gained an active-frame drain guard against stale reads).

**Traps for the next session:** `decode_tok_per_s` swings ±35% with thermals on this M4 — trust only interleaved A/B on a cooled machine (the plan already says this; it bit twice this session). `sample`-profile against symbol addresses (atos with the vmmap slide) beats every counter/trace in this codebase for attribution; the frame-trace counters remain partly cosmetic.

**Next levers (measured, in order):** (a) verify tail lm_head apply runs on its OWN command buffer + wait (~407/3071 samples ≈ 13% of decode wall inside verify_ns 49-51ms) — encode it (and the batched argmax) onto/after the verify frame before its single wait; (b) fold acceptance (488‰ with fold vs ~570-640‰ without — P2 Option-B staleness) is now the dominant spec-side loss; (c) k/policy retune + auto defaults (P5) from this new cost basis; (d) draft argmax kernel itself (~2-3.5ms; 268MB f32 dense lm_head read — f16/quantized draft head would halve it, acceptance risk unmeasured).

---

## Original handoff (pre-session; cost model now stale — see OUTCOME)

Workdir `zig/pkg/inference/`, branch `codex/quant-kernel-metal-compiler`.
**Prerequisite: P2 (fold) has landed** — see `SPEC_DECODE_P2_FOLD_MATERIALIZE_HANDOFF.md`. Re-baseline before starting; numbers below are post-P1 (pre-P2). Memory of record: `~/.claude/.../memory/project_gemma4_spec_decode_fix.md`.

## Why this matters
Post-P1 wall at k=2+bonus is 10.1s/128tok: verify ~5.1s wall (GPU ≈ floor), materialize 2.6s (P2 kills), **draft 1.7s (13.2ms/step)** and **~3.5s unattributed cross-runtime/frame overhead**. After P2 lands (~17 tok/s expected), the draft + overhead bucket IS the remaining gap to plain (24.3). The draft is a **4-layer / hidden-256 / ffn-2048 (~6M backbone)** assistant — 13.2ms/step is ~10× its compute; it's all orchestration.

## Known facts (measured in prior sessions; re-verify with the phase tracer)
- Phase tracer: `ANTFLY_GEMMA4_MTP_DRAFT_PHASE_TRACE=1` → `mtp_phase` lines; profile fields `draft_assistant_ns / draft_argmax_ns / draft_lm_head_ns / draft_selection_ns / draft_preprojection_ns` in `--json-timing`.
- Post-P1 split (warm run, 161 steps): assistant forward ~11.4ms, argmax ~5.4ms (lm_head 2.6 + selection 2.5), pre/post-proj ~1ms. (An earlier cool run had forward 6.2ms — machine-state sensitive; re-measure.)
- **Residual host flushes**: ~2×`{2,512}` host materializations per step in the donated-KV backend-cache write path (`updateBackendKvLayerCache`, `metal_compute.zig:~8368-8457` — `cloneMetalTensorRows` HOST clones) force ~4 mid-frame flushes/step. Trace with `TERMITE_METAL_TRACE_HOST_MATERIALIZE_LIMIT=N`.
- **One frame submit+WAIT per draft step** (`gemma4_mtp.zig` `draftTokenDevice:~820-946`: begin frame → encode → `decoderRuntimeSubmitAndWaitFrame` → argmax). k steps = k round-trips; the argmax→next-embedding dependency is the serializer.
- Draft lm-head argmax uses `linearNoBiasArgmaxLastRow` (~1.6ms after first-call slot prep) + selection ~2.5ms host-side.
- `ANTFLY_GEMMA4_MTP_DISABLE_MASKED_EMBEDDING=1`: ~3ms/step saved, acceptance unchanged (verified twice) — candidate default flip.

## Levers, in expected-value order
1. **Devicize the donated-KV backend-cache write** (kill the ~4 host flushes/step): make `updateBackendKvLayerCache`'s clone path device-side (or skip the host mirror for device-resident donated KV). Expect draft ≈ −3-5ms/step.
2. **Masked-embedding default flip**: re-verify acceptance-neutrality with the fold on, then default `DISABLE_MASKED_EMBEDDING`. −~3ms/step.
3. **Chain draft steps on-device (bigger build)**: the plain-decode pipelined path already has device-token feedback (`..._quant_embedding_lookup_prepared_device_token` reads `runtime->token_buffer`) — the draft loop could encode k steps into one frame with the argmax writing token_buffer and the next step's embedding lookup reading it on-device. Removes k−1 submit-waits + host selection. This is the draft analog of the pipelined greedy loop; budget it like a mini-project (frame semantics under the assistant's generic-op forward — the WAF class of hazards is FIXED but be deliberate).
4. **The ~3.5s unattributed bucket**: decompose first (don't guess) — per-runtime frame stats (`TERMITE_DEBUG_METAL_TIMING=1` prints per-runtime `metal_decoder_frame` lines; the draft runtime's wait vs gpu), `dedicated_runtime_hits=0` anomaly (the fused `gemma4MtpVerifyCommit` path never engages — find its gate and why), and the k=1 finding that matched rounds pay a hidden cached-choice target forward accounted in RESIDUAL not verify_ns.
5. **k/policy retune + defaults** (after the above): cost model (`/private/tmp/spec_cost_model.py`) → pick default k (data says 1-2, not 4), tune `ANTFLY_GEMMA4_MTP_AUTO_*` so `auto` enables only when profitable, and decide `ACCEPT_BONUS` default-on for Metal (it works now; identity-gated).

## Gates (every change)
`ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1 bash scripts/compare_metal_gemma4_e4b_qat.sh` (exit 0) + fold/bonus-on token-identity diffs + acceptance within noise + full suite. Perf verdicts on a cool machine, interleaved A/B, cost model as arbiter vs plain (and llama ~30 tok/s for the endgame).

## Collision note
Do NOT start until P2's `generation.zig` changes land (profile counters + round loop overlap). Lever 1/3 live in `metal_compute.zig`/`gemma4_mtp.zig`/`metal_kernels.m` and are mostly disjoint from P2, but rebase after P2 merges.
