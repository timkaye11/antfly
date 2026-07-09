# Spec-decode P2 Hand-off: fold materialization into the next verify (pending-token design)

> **STATUS (2026-07-09): IMPLEMENTED + VALIDATED — flag ships DEFAULT OFF per the measured verdict below.**
>
> - Implementation: `ANTFLY_GEMMA4_MTP_DEFER_MATERIALIZE=1`; pending state on `Gemma4MtpActivationState` (`pending_unmaterialized` + `pendingCount`/`assertKvTokensInSync`/`notePending*`); append `draft_count + entry_pending`, error truncates use live `pendingCount()`; pending+cached-choice = `error.InvalidSpeculativeState`; Option-B via per-step retained projected activations (correction → slot `matched_drafts`, bonus → last); `flushPendingGemma4MtpMaterialize` before both standardDecode fallbacks; `deferred_materializations`/`pending_flushes` counters (profile struct + JSON + debug line).
> - Tests: unit (`gemma4 mtp pending token bookkeeping holds kv invariant across round shapes`, opt-in flag) + `scripts/test_gemma4_mtp_defer_materialize.sh` (fold-on identity k=1/2/4+bonus with materializations==0, zero-match/acceptance/cost fallback gates incl. `pending_flushes==1`, fold-off control). POLICY_ONLY gate exit 0. All fold-on runs BIT-IDENTICAL to plain target at 48 and 128 tok.
> - **Measured A/B (M4, 128 tok):** k=1+bonus: 9.33→11.32 tok/s bench prompt, 7.70→9.14 oracle prompt (**WIN +19–21%**). k=2+bonus: 13.0/13.2→11.6/11.8 bench (−10%), 12.47→7.73 oracle (**LOSS −38%**). Root cause of the k≥2 loss: exactly the "economics honesty" risk — fold-off ran 30/67 oracle rounds on the zero-forward cached-reject path, fold-on runs a full verify every round, AND the verify's marginal row is ~57ms (2-row 139ms/call → 3-row 196ms/call), so absorbing the pending row is not free. Acceptance dropped 8–18pts from Option-B stale activations (secondary). materializations→0 confirmed in all fold-on runs.
> - Adversarial review (post-implementation, 2 confirmed findings, both acceptance-quality-only, no correctness bugs): (a) every post-fold round drafts against a donor KV view missing the pending token's row (drafts run before the append; assistant is read-only donor-KV) — a second acceptance-drop driver alongside the stale activation, invisible to identity gates by construction; (b) the bonus-round Option-B adoption (slot[k−1]) is one chain position short of the bonus token (no k+1-th draft step exists), a strictly larger approximation than the correction case (slot[m] is positionally exact). Both are inherent to the design, are priced into the measured acceptance deltas above, and are documented at the fold site. Restricting the fold to corrections-only would NOT help: at k=1 the win came almost entirely from eliminating bonus materializes (63/63 at k=1 were bonus).
> - Decision: keep default OFF. Revisit if P3/P4 shrink verify call overhead / per-row cost — the fold's plumbing is in place and gated.
> - Pre-existing bug found by audit (NOT fold-related, unfixed): `syncPagedKvViewForDecode`'s `kv_compacted` branch assumes +1 token per append — wrong for any multi-token speculative append when `cache_compaction_ratio` is set.

Workdir `zig/pkg/inference/`, branch `codex/quant-kernel-metal-compiler`. Zig 0.16.0.
Base: HEAD after `5c77099f2` (P1 batched argmax). All `file:line` refs verified on that tree but WILL drift — re-grep before editing (anchors given as `fn-name:~line`).

## Context — what's already done and measured

Speculative/MTP decode fix arc, committed so far:
- `8715a9cfc` — `ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1` mode in `scripts/compare_metal_gemma4_e4b_qat.sh`: the fast (~4min) hard gate running plain-golden + MTP identity (target vs auto/force/force-k1 token ids). **Run it after every change.**
- `396c0cc01` — fixed the blit-under-open-encoder use-after-free (k=2 crash AND the historical `ANTFLY_GEMMA4_MTP_ACCEPT_BONUS` segfault — one root cause). **Bonus now works on Metal** and is central to the economics.
- `5c77099f2` — P1: batched multi-row verify argmax (one command buffer/sync/download for all k+1 rows; was one full frame teardown per row). On-device unit test included.

**Measured stack post-P1 (E4B QAT, k=2 + bonus, force, 128 tok, warm base M4):** spec 12.66 tok/s vs plain 24.3. Wall 10.1s = verify 5.1s wall (40 calls × 128ms; target-frame GPU≈wait, verify GPU ≈ its ~60-70ms kernel floor) + **materialize 2.6s (61 calls × 42ms — with bonus ON, EVERY round materializes)** + draft 1.7s (13.2ms/step) + residual. Acceptance 65.3% at k=2 (70.8% at k=1). Profile via `--json-timing` (`speculative.mtp_profile` block; round counters under `speculative`: rounds/drafted/matched/accepted/corrections/bonus).

**This task (P2): eliminate the materialize forward** — the single biggest remaining lever (~2.4-2.6s/128tok). Expected landing ≈ 17 tok/s at k=2+bonus; parity/win then needs the residual overhead + draft trims (separate tasks).

## The design (validated by a Plan agent against the code — corrections baked in)

Today, when a round ends with a **correction** (mismatch at index m → commit `target_choices[m]`) or a **bonus** (all matched → commit `target_choices[k]`), that committed token c has never been forwarded through the target: `materializeAcceptedTokenKvForMtp` (`generation.zig`, materialize block `:~6157-6208`, impls `:~6860-7061`) runs a 1-row full forward to (a) write c's KV, (b) produce c's hidden (next round's draft predictor activation), (c) produce `next_cached_target_choice`.

**The fold:** don't materialize. Leave c *pending*; the NEXT round's verify prepends c as row 0.
- This is **literally today's non-cached full verify shape**: with `cached_first_target_choice == null`, `target_query_len = verify_len` and `verify_start = verify_seq - verify_len = seq_len - 1` (`:~5740`) — rows = `[last_committed_token, d1..dk]`. The fold just makes "last committed" the un-forwarded c. **No new verify code needed.**
- Verify's paged-attention **suffix write covers its query rows** (`writeLayerKvSuffix`, `metal_compute.zig:~8875-8965`): c's reserved KV slot gets written by the next verify. Positions come from `query_position_offset = total - query_len` — correct with c at row 0.
- Row-0's argmax = the target choice after c = exactly what `next_cached_target_choice` supplied. The acceptance loop (`acceptVerifiedDraftTokenChoicesGreedy:~6393-6469`) consumes `choices[i]` vs `draft_tokens[i]` identically.
- **Bonus rounds fold the same way**: the pending token is the bonus token instead of the correction.

### Required mechanics (each was code-verified)
1. **Pending-state helper** on `Gemma4MtpActivationState` (`:~4619`) — one small struct owning `pending_unmaterialized: bool` (or count 0/1) and the append/truncate arithmetic, asserting the invariant **`decode_state.total_tokens + pending_count == seq_len` at every transition**. Do NOT scatter flags.
2. **Append/truncate +1 when pending**: `appendGeneratedTokens(draft_count)` at `:~5735` → `draft_count + 1`; the error-path truncates at `:~5778 / :~6029 / :~6084` also +1. The success rollback `:~6152-6154` (`draft_count - matched_drafts`) is already correct — it trims only the tail, leaving c's now-written slot.
3. **⚠️ SEV-HIGH silent-corruption hazard:** the mid-generation `standardDecode` fallbacks (`generation.zig:~2892` and `:~2925`, reached via the auto cost/acceptance/zero-match gates) build a decode context assuming `total_tokens == seq_len`. With a pending c they would **overwrite the last verified token's KV and never write c's** — silent wrong tokens, no error. **Flush the pending token (run one materialize) before BOTH fallbacks.** Same for any other exit that continues generating outside the MTP loop.
4. **Final round pending is fine**: on EOS/max-tokens/grammar stop, c is already committed from an exact argmax and its KV is never needed again.
5. **Draft-1 activation (the acceptance question):** round N+1's first draft step normally consumes c's hidden (from materialize). Use **Option B**: the assistant's own chained `projected_activation` from the draft step that emitted the *rejected* token (mismatch index m → step m+1's output; bonus rounds → the final step's). This is exactly the approximation draft steps 2..k already make (they consume the assistant chain — verified: `device_chain_activation = draft_result.projected_activation`, `:~5639-5640` / host `:~5662-5663`). **Retain per-step projected activations in a small array** (currently overwritten each step); pick index m (or last, for bonus). Option C (verify-row hidden via the pure-prefix reuse expression at `:~5856/:~5936/:~6002`, guards relaxed) is a **temporary experiment patch** for the acceptance A/B only — do not ship a second knob.
6. **The dead scaffold is your hook:** `have_predictor_activation` branch at `:~6161-6163` is provably dead today (all its setters are guarded `!correction_added and !had_bonus`) — it was built for exactly this. Under the fold, correction/bonus rounds set the (Option-B) activation and skip materialize.
7. **One env flag**, default OFF until validated: e.g. `ANTFLY_GEMMA4_MTP_DEFER_MATERIALIZE=1`. Telemetry: with the flag on, `materializations` must → ~0 (only pending-flushes before fallbacks remain — consider a `pending_flushes` counter); watch `mtp_acceptance_permille` and `accepted/rounds`.

### Economics honesty (why this could still disappoint)
The fold **removes the cached-first rejection fast path** (`rejectMtpDraftFromCachedFirstChoice:~6278`): today a repeat-mismatch round costs draft + cached-reject (zero forward) + materialize; under the fold it becomes a full (k+2-row) verify. Net win depends on stale-activation drafting keeping step-1 acceptance near today's 65-71%. If acceptance craters, leave the flag off and record the finding — the cost model (`/private/tmp/spec_cost_model.py`, feed `cat json log`) arbitrates.

## Deterministic tests (REQUIRED before the fold lands — an 8-token gate may never hit these)
Write tests (unit-level where possible, else scripted CLI runs with forced env) for:
1. correction → pending → next verify (tokens identical to plain; invariant holds).
2. correction → pending → **each** standardDecode fallback — force via the auto gates (`ANTFLY_GEMMA4_MTP_ZERO_MATCH_FALLBACK_ROUNDS`, `ANTFLY_GEMMA4_MTP_MIN_ACCEPTANCE_PERMILLE` high, auto cost gate) — pending must flush first; tokens identical.
3. bonus → pending → next verify.
4. error rollback while pending (inject/force a verify error path; truncate arithmetic must keep the invariant).
5. final-token/EOS while pending (clean exit).
Each asserts the boundary invariant and token identity vs plain.

## Verification (after: build; the 5 tests; then)
```sh
cd zig/pkg/inference
zig build -Doptimize=ReleaseFast -Dmetal=true -Donnx=false -Dcuda=false
ANTFLY_INFERENCE_GEMMA4_COMPARE_POLICY_ONLY=1 bash scripts/compare_metal_gemma4_e4b_qat.sh   # exit 0
# Fold-on identity (48-tok direct diff, all k, bonus on):
B=./zig-out/bin/antfly-inference; M=<e4b-dir>; D=<assistant-dir>; P="<any prompt>"
# target vs (k=1,2,4 × ANTFLY_GEMMA4_MTP_DEFER_MATERIALIZE=1 ANTFLY_GEMMA4_MTP_ACCEPT_BONUS=1) --print-token-ids diff
# Profile A/B (cool machine): fold on vs off at k=1/k=2+bonus; expect materializations→0, wall −~2.4s,
# acceptance within ~5pts; feed /private/tmp/spec_cost_model.py for the verdict.
zig build test -Dmetal=true -Dcuda=false    # full suite 0 fail (2083 passing at base)
```

## Risks (ranked)
1. **SEV-HIGH — silent KV corruption** via the fallbacks / any `total_tokens == seq_len` assumption outside the MTP loop. The pending-flush + invariant asserts + test #2 are the defense. Grep broadly for `makeDecodeContext(seq_len` -style callers reachable mid-MTP.
2. **SEV-MED — acceptance drop from stale-activation drafting** turns the fold into a regression on exactly the rounds it targets. Flag stays off unless the A/B says net-positive.
3. **SEV-MED — near-tie argmax flips**: c's hidden now comes from a (k+2)-row forward instead of a 1-row forward (fp tiling differences). Same-in-kind as today's accepted-draft commits; the identity gates are the arbiter — a flip is detectable, evaluate rather than ignore (precedent: multi-row verify already passes the gates).
4. **SEV-LOW — draft seeding sees kv_len = seq_len−1 on pending rounds** (c's donor KV missing for the assistant's attention): draft-quality-only, folds into the acceptance A/B.

## Key files
- `src/pipelines/generation.zig` — `speculativeDecodeGemma4Mtp:~5551` (round loop, draft loop `:~5611-5693`, append `:~5735`, verify paths `:~5766-6138`, accept `:~6393-6469`, materialize block `:~6157-6208`, dead scaffold `:~6161`, cached-reject `:~6278`, fallbacks `:~2892/:~2925`, activation state `:~4619`, materialize impls `:~6860-7061`).
- `src/ops/metal_compute.zig` — KV suffix write `:~8875-8965` (read-only; understand the position math).
- `scripts/compare_metal_gemma4_e4b_qat.sh` — POLICY_ONLY gate.
- Memory of record: `~/.claude/.../memory/project_gemma4_spec_decode_fix.md` (full measurement history); master plan `~/.claude/plans/great-lets-create-a-distributed-catmull.md`.
