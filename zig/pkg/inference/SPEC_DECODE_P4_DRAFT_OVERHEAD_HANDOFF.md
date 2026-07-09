# Spec-decode P4 Hand-off: draft-step + cross-runtime overhead (the post-fold levers)

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
