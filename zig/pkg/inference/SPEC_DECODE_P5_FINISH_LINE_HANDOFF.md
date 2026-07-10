# Spec-decode P5 Hand-off: the finish line (tail frame-fold, acceptance, policy defaults, verdict)

Workdir `zig/pkg/inference/`, branch `codex/quant-kernel-metal-compiler`. Base: HEAD including the P4 commit (donated-slot-attention-on-frame + prefill hidden capture; `ec285a568` + P4). Memory of record: `~/.claude/.../memory/project_gemma4_spec_decode_fix.md` — **read the full phase history: nearly every phase overturned its predecessor's attribution. Phase-A-first (measure the live process; `sample`+`atos` worked where profile counters lied), and treat this doc's cost claims as hypotheses to re-verify.**

**State at base:** spec k=1+bonus+fold = 25.6-27.2 tok/s vs plain 24.9-27.1 (cooled interleaved A/B) — parity. Tokens bit-identical in all six sweep arms; acceptance 488‰ (fold) vs ~640‰ (fold-off); draft ~5-6.4ms/step; verify ~37ms@2rows. Gates: `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1 bash scripts/compare_metal_gemma4_e4b_qat.sh` (exit 0) + `scripts/test_gemma4_mtp_defer_materialize.sh` + suite 2085/0. Measurement traps: fanless-M4 thermals (interleave + cooldowns; a false regression mid-P4 came from this), `TERMITE_METAL_TRACE_FRAME` must be `=all`, `ANTFLY_GEMMA4_MTP_ADAPTIVE_K` stays on even under force policy, frame-trace op counters are contract records (cosmetic) — never attribute from them.

## Item 1 — Fold the verify-tail lm_head apply into the verify frame (top item, ~13% of decode wall)
Today `forwardPreparedLmHeadArgmaxRows` (decoder_gated_runtime.zig:~6160) runs AFTER the verify frame with its own synchronous apply (`decoderRuntimeApplyLinear` → commit+wait) + a batched argmax that itself does a frame flush/restore. That's 2 extra waits per verify outside the ~37ms frame.
- The pieces exist: the rows-1 planned decode frame already encodes `tail_lm_head` + `tail_argmax` IN-frame; the 2-row q6_k mmv (`termite_q6_k_linear_r2_reduce`, routed in the generic q6_k dispatch at rows 2-8/out≥32768) and the batched argmax encode helpers with `_at` offset params (`termite_metal_encode_argmax_logits*_on_encoder_at`) are all encoder-level and frame-compatible.
- Work: an on-encoder path that, at the end of the verify frame (inside `forwardFinalHiddenTensorGemmaDirect`'s multi-row branch or just after the block loop), encodes final-norm→lm_head(r2 mmv from the prepared slot)→rows argmax into the SAME frame, one wait, read k+1 token ids from the shared token buffer. Mind the planned-range hazards (the argmax helpers already take them) and pre-size token buffer/topk scratch BEFORE the frame (growth mid-frame = the P0.5 hazard class; use the existing pre-size discipline from `..._argmax_from_logits_rows_suppress_device`).
- Expected: verify wall −8-11ms → spec +1.5-2.5 tok/s. Kill switch + identity gates as always.

## Item 2 — Fold acceptance recovery (488‰ → target ~640‰)
Three documented drivers (P2 adversarial review, priced-in then, addressable now):
- **Missing donor-KV row for the pending token during post-fold drafting** (drafts run before the append). P4's donated-slot-attention-on-frame rework changed this path — re-check whether the pending token's row can now be made visible to the draft's slot attention (the target KV slot IS written by the time of… verify, not draft — the fix may be ordering or an extra slot-view row). Measure acceptance delta alone.
- **Option-C draft-1 activation** (verify row `logit_base_row + accepted - 1` hidden instead of Option-B chain activation) — temporary experiment patch per the P2 handoff; A/B acceptance.
- **Bonus adoption one chain position short** (slot[k−1] for the bonus token) — inherent, likely unfixable cheaply; document if so.
Each: fold-on vs fold-off acceptance + tok/s, tokens identical (these only change PROPOSALS). If acceptance reaches ~600‰+, fold’s margin widens meaningfully.

## Item 3 — P5 policy + defaults (the original plan’s Phase 5)
- **Pick shipped defaults from a fresh k-sweep** (k∈{1,2}, bonus on, fold on/off, cost model `/private/tmp/spec_cost_model.py`): likely k=1+bonus+fold default when a draft model is provided.
- **Consider `ANTFLY_GEMMA4_MTP_DEFER_MATERIALIZE` default-ON** (it's the winning config; the fold has its own deterministic test suite) — gate on the six-arm sweep staying identical.
- **Auto-policy retune**: tune `ANTFLY_GEMMA4_MTP_AUTO_*` (AUTO_MAX_K, MIN_ACCEPTED_PER_ROUND_MILLI, cost-probe rounds) so `--speculation-policy auto` + calibration enables spec exactly when profitable.
- **Replace the "auto must not self-enable" gate** (compare script forces `ANTFLY_GEMMA4_MTP_ENABLE_METAL_AUTO=0` and can't validate the new behavior) with two explicit tests: *calibrated-enable* (auto + positive calibration → enables, tokens match target) and *uncalibrated-disable* (auto + none → stays off). Update `run_mtp_policy_check` accordingly.
- **Final verdict**: cooled A/B spec vs plain vs llama.cpp (~30 tok/s) at 128-256 tok; report honestly (parity with plain is achieved; llama needs items 1-2 to land well).

## Item 4 — Small retunes (opportunistic)
- Tail r2 kernel occupancy (11→~7ms possible; yl0+yl1=32 regs/thread — try NR0=1×wider or half-staging).
- PLE row-stride hoist: `sliceLastDim(ple_vectors, layer*ple_dim…)` per layer (decoder_gated_runtime.zig:~1124) materializes a copy kernel per layer at rows≥2 (free view at rows-1) — thread an input row-stride through applyPleResidual→encode→PLE kernels (~2-4ms/verify; touches plain decode → full gates).

## Cross-cutting
- ⚠️ **CUDA latent bug (ticket, needs the CUDA machine)**: CUDA's prefill hidden-capture passes a PRE-norm row where Metal captures POST-final-norm — verify/align before trusting CUDA MTP seeding.
- `kv_compacted` multi-token-append bug (P2 audit) if `cache_compaction_ratio` is ever enabled.
- After any lever lands: POLICY_ONLY gate, defer-materialize gates, six-arm identity, plain golden + prefill untouched, suite 0 fail. Perf verdicts cooled + interleaved only.
