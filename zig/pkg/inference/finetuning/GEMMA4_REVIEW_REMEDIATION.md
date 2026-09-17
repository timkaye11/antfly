# Gemma training review remediation and performance work

Status: local follow-up verification complete, September 16, 2026. September 15
E2B/E4B Metal-versus-MLX captures are complete. E4B passes the configured
single-seed quality gates; E2B fails the reward-improvement gate in both
backends. GRPO remains experimental. These results support PR review, not
full production promotion or a general performance claim.

## September 16 follow-up review

The September 15 results below are retained historical evidence. The follow-up
changes are based on `334fed88151e4d8eb3b08ff641943e9bef8b7746` and require their
own verification; neither the earlier binary hash nor its MLX capture qualifies
this changed worktree. The user owns the commit and submission.

| Follow-up finding | Disposition |
| --- | --- |
| Every merge-queue batch runs the GPU gate | Diff against `merge_group.base_sha`; unrelated and documentation-only changes skip the job. Pathspec regression covers platform/build/runtime/serving dependencies. |
| Missing fourth dense attention scale | Paired GQA kernel now honors the explicit scale; analytic regression also checks the paired dispatch counter. |
| Renderer test unreachable and stale | Select the entire renderer test module, fix empty Gemma BOS expectation, and exclude an unanswered tool-response marker from target labels. |
| Packed fused VJP unverified | Independent scalar GQA/RMSNorm evaluators support graph finite differences, including packed-gradient slicing, mixed heads, windows, custom scale, and frozen/trainable norm weights. |
| Native BF16 borrowing lacks CPU logits check | Extend resident/lazy borrowed embedding regression through the tied output head, comparing explicit F32 weights and analytic logits. |
| New broad native/PJRT store refinement | Restrict both early refinement sites to Gemma4 channel architecture; ordinary GPT/Gemma configurations retain their previous path. |
| Brittle CI name selection | CI audits every added named inference test against qualified test-log names; missing selections fail. The serving target is explicitly invoked. |
| Compiler memory budget | Pass `--maxrss 10737418240` to focused builds as well as the target estimate. This is Zig scheduling admission, not an operating-system RSS limit. |
| Preference environment changes reused v5 | Bump fingerprint domain to v6 and explain incompatible restore failures; preserve the historical AdamW default of 0.01. |
| Clone metadata committed during encoding | Per-frame completion receipts publish cloned coverage only after a successful wait. Canceling growth restores prior backing buffers; retry reads actual capacities. Regression covers cancel, unrelated successful frame, retry, and second-hop fan-out. |
| Process-global training policy | Scope training policy and captured enablement to the owning thread; another thread cannot inherit it. Acquisition/release remain synchronous and thread-affine. |
| GQA dispatch inferred from attributes | Add explicit training intent to graph attention attributes and route on that intent, including default scale/window. |
| Missing dot-general gradients | Implement every rank-two single-contraction layout; unsupported layouts return `NoVjpRule` instead of dropping gradients. |
| GRPO input edge cases | Validate reward/epsilon/clip inputs; count opaque sparse prompt IDs without overflow; constant groups with zero epsilon yield zero advantages. Invalid all-masked configurations fail closed. |
| CCE dense allocation risk | Bound each fallback logits tensor to 64 MiB, checking overflow before allocation. Capture require-CCE and tile policy at backend creation. Larger unsupported shapes fail explicitly. |
| Multimodal GRPO temperature mismatch | The public lane remains rejected pending multimodal evaluation; its internal legacy path now explicitly rejects non-unit temperature. No multimodal parity claim. |
| Rejected option serialization | Remove layer-name, LLRD, and schedule-free fields from effective run reports; keep clear parser rejection. |
| Docs and helper issues | Remove nonexistent public GGUF flag, personal paths, and misleading temporary evidence paths. Document missing numerical toggles. Share the remaining CUDA-gate hash implementation; fixture cache is per-work-directory and validation survives Python optimization. Add PEFT round-trip helper unit tests. |

Intentional policies are now explicit: the canonical renderer is non-thinking
and strips thought content, including from targets; reasoning-trace training is
not supported. Hashing the whole renderer source deliberately invalidates
prepared inputs even after comment edits. Evaluation data and acceptance
settings remain in preference identity so resumed publication cannot silently
change its contract. These are documented constraints, not parity fixes.

The two remaining `sha256_file` definitions are thin adapters around the shared
streaming implementation, preserving distinct prefixes/error contracts. JSON
helpers with different canonicalization/error behavior are not interchangeable.
A complete immutable low-level kernel policy and per-lane recipe extraction
remain architecture work. The review's unspecified “everything in grpo.zig”
and “four getenv sites” require concrete examples to claim complete closure;
this pass fixes reproducible GRPO defects and the identified CCE hot reads.

Follow-up local verification:

- Debug, ReleaseSafe, and ReleaseFast required-Metal gates: **450 passed, two
  optional real-model fixture skips** in each configuration (452 selected).
- ML graph suite: **546/546**. Gemma Python suite: **787/787**.
- Isolated graph gate: **24/24**; serving gate: **8/8**; CLI/server gate:
  **9/9 selected**. CPU-only build executes both native BF16 logits and the
  narrowed store-refinement regression successfully.
- Branch-to-log audit: **340/340 added named tests selected**, no missing
  tests, one explicitly reported optional GGUF fixture skip.
- Public ReleaseFast CLI: **37/37 build steps**. Stock PEFT CPU load/save/reload
  of the exported tiny structural fixture passes with **zero logit difference**;
  this is export compatibility, not Gemma4 numerical qualification.
- Five CI scope/audit tests, toolchain policy, workflow YAML parsing, all
  85 checked Python files' Ruff formatting, changed Zig formatting, whitespace,
  and conflict-marker checks pass.

The optional public build initially rejected a 10 GiB scheduler budget before
compiling: its existing full-application targets declare up to 20 GiB. Repeating
with the workflow's normal serialized settings succeeds; the inference compiler
reports 13 GiB peak RSS. The focused test targets retain their 10 GiB admission
budget. This was a local invocation error, not a machine crash or failed test.

Logs and source hashes are retained under
`.benchmark-assets/gemma4-review-20260916/`. Hosted CI and new full-model MLX
capture remain separate release evidence; September 15 captures are not
relabeled as current.

## September 15 confirmed findings and changes


| Finding | Remediation | Verification status |
| --- | --- | --- |
| CI missing graph target, wrong revision/toolchain, advisory dependency | Restored `test-gemma-graph`; use requested head and toolchain policy; require Metal result in base/full aggregate | Policy checker and validation-scope test pass; hosted execution pending |
| Python formatting failures | Formatted 49 affected Gemma scripts with Ruff | 785/785 tests pass with localhost access; all 81 Gemma Python files pass Ruff format check |
| F16 backward routed through BF16 prefix/tail | Guard prefix route by weight dtype | GPU regression covers 193 rows and 65,536 output dimensions |
| F16 backward flushes tiny gradients | Preserve F32 gradient staging and accumulation | GPU identity regression includes 2e-8 and values beyond F16 range |
| Training renderer differs from serving | Canonical separate system turns, inline tool results, call/content ordering, history-channel stripping; exclude observations from labels | New golden compares 14 prefixes/prompt modes directly with checked-in Jinja |
| Recipe drops seed/options or uses wrong backend | Forward Gemma bootstrap seed; reject unsupported native preference configuration and non-preference execution modes | Focused contract tests |
| GRPO KL references raw base for SFT-started run | Freeze bootstrap adapter before restore; use same reference for heldout evaluation; report snapshot mode | Real E2B and E4B nonzero-start diagnostics select the initial-adapter snapshot; baseline mean KL 5.06e-12 and 3.19e-11 |
| Resume ignores numerical environment | Shared stable sorted environment digest for SFT and preference fingerprints; native preference admission checks; report overrides | Digest tests and actual CLI mismatch rejection added |
| Resume recomputes baseline from checkpoint | Evaluate immutable bootstrap before restore in SFT and GRPO | Real tiny Metal final-boundary recovery regression added |
| Implicit AdamW decay | Explicit SFT flag and Gemma recipe field; record and fingerprint effective value | Preserve historical 0.01; explicit zero exercised by CLI regression |
| Missing fused-gradient coverage | Include existing RMSNorm finite differences and Metal/native GQA parity in focused gate; add independent GQA finite differences | Debug and ReleaseSafe gates pass |
| Dense attention drops custom score scale | Honor scale in three dense simdgroup kernels | Analytic GPU test passes for 8, 16, 33 rows |
| Training attention silently stages through host | Device training request declines without host fallback | Strict execution fails closed on decline; no change to serving fallback |
| Stale quantized aliases after slot release | Remove every alias of retired slot before reuse | GPU-backed slot lifecycle regression passes |
| Dense CE shared-memory race | Synchronize all readers of shared state before next row writes it | Native-reference loss/gradient stress passes: 259 rows, 521 vocabulary entries, eight repeats, mixed/all ignored labels and softcap on/off; barriers also protect adjacent shared-state reductions |
| CCE can materialize full logits | Warn with shape/dtype/allocation estimate and explain existing require-CCE switch | Explicit memory contract documented; bounded general fallback remains open |
| Explicit `default` initializer invalid in PEFT JSON | Serialize boolean true; canonicalize manifest comparison so explicit default and PEFT true agree | Full config/manifest regression and fresh stock PEFT load/save/reload pass |
| Constant chat digest | Bind renderer source | Prepared artifacts require refresh |
| Publication rejects unknown directory-entry types | Resolve unknown type with no-follow stat | Unknown file/directory/symlink regression passes in ReleaseFast |
| Focused ReleaseSafe compiler exceeds declared memory | Per-target 10 GiB allowance, based on 8.6 GB observed compile | ReleaseSafe compile and required Metal/oracle gate pass |

## September 15 verification

The expanded required-Metal Debug, ReleaseSafe, and ReleaseFast gates each
pass 431 tests with two optional real-model fixture skips; Debug reports no
leaks. The final CLI build passes all 37 steps, and the separate CLI/server
gate passes. Unresolved-conflict, whitespace, Zig-format, and source-marker
audits are clean.
The Python suite passes 785/785; all 81 Gemma Python files pass Ruff formatting.
The toolchain-policy checker and validation-scope tests pass.

The prior remediation verification also passes 24 graph tests, 536 ML graph
tests, and stock PEFT load/save/reload with exactly zero logit difference on
the structural fixture. Pinned MLX matches all 12 temperature/head analytic
fixtures within 1.73e-7. The earlier verification logs and model hash/size lock
checks are retained under `.benchmark-assets/gemma4-review-20260915/`.
Hosted CI has not run for this uncommitted worktree.

The test-selection audit found 77 added named regressions missing from the
original focused gate. The focused target now explicitly selects 72 shared
regressions; CI selects the remaining five CLI/server tests through their
own test roots. Those roots use compile-time filtering. The declaration-to-log
audit now observes all 331 of 331 added named tests across expanded Debug
and the separate CLI/server gate, with none missing. One of those added tests
is the optional real-GGUF fixture skip; the other skipped fixture predates
this branch. The audit distinguishes selection coverage from execution.

The new whole-decoder serving regression compares cached BF16 Metal with
native full-prefix logits over prefill and ten decode steps. Local attention
wraps four two-token ring pages while the global layer retains full history.
Maximum logit error is 3.09945e-6 (tolerance 3e-4); the negative control proves
that ignoring the local window changes results. Debug execution exposed and
now verifies fixes for direct-input, unused reserved-carrier, and KV-seed-list
allocation leaks. The final E4B trainer checkpoint and published adapter are
byte-identical before and after these serving ownership fixes.

Dense CE now protects shared-state readers before buffer reuse. Stress
coverage compares native loss and dHidden for 259 rows and 521 vocabulary
entries, with mixed/all ignored labels, softcap enabled/disabled, and eight
repetitions. Provenance hashing streams through 64 KiB rather than mapping
whole model shards; byte-domain compatibility and boundary sizes are tested.

## September 15 matched Metal and MLX evidence

Artifacts: `.benchmark-assets/gemma4-pr-20260915/`. Both models use locked
BF16 weights, seed 17, 64 training groups, 32 heldout groups, 16 completions
per group, and a one-token completion horizon. The acceptance thresholds,
optimizer settings, and dataset hashes were fixed before execution.

| Result | E2B | E4B |
| --- | --- | --- |
| Optimizer updates / zero-variance skips | 29 / 35 | 61 / 3 |
| Mean reward, baseline → final | 0.759765625 → 0.759765625 | 0.236328125 → 0.23828125 |
| Top-ranked reward, baseline → final | 0.75 → 0.75 | 0.28125 → 0.40625 |
| Positive-group rate, baseline → final | 0.90625 → 0.90625 | 0.96875 → 0.96875 |
| Metal / MLX final KL loss | 5.34647e-7 / 5.33501e-7 | 3.824733e-5 / 3.824573e-5 |
| Adapter tensors compared | 552 | 686 |
| Update-vector relative L2 error | 0.0308315% | 0.0113339% |
| Update-vector cosine | 0.999999952478 | 0.999999993580 |
| Maximum absolute update difference | 1.74119e-8 | 1.42354e-7 |
| Quality acceptance / publication | Improvement gate fails; no adapter | All configured gates pass; adapter published |

MLX independently samples each group during the trace replay comparison;
all 1024 training and 512 heldout completion tokens match their Metal order
for each model. Updates consume recorded training decisions, while final
heldout evaluation samples from the resulting MLX adapter. Reward metrics
match exactly. The unchanged adapter direction/norm/vector gates pass, but
bounded numerical agreement is not bitwise equality or broad statistical parity.
E2B's flat reward is shared across backends at this recipe/horizon; no
threshold was relaxed to make publication succeed.

| Diagnostic process measurement | E2B Metal | E2B MLX | E4B Metal | E4B MLX |
| --- | --- | --- | --- | --- |
| Wall seconds | 251.20 | 363.78 | 628.45 | 571.40 |
| Peak sampled physical footprint, decimal GB | 6.67 | 13.34 | 10.34 | 15.39 |
| Sampled swap growth | 0 | 0 | 0 | 0 |

These are single-run diagnostic measurements. E4B MLX uses an exact, fixed-input
frozen-PLE row cache; E2B keeps the full frozen embedding tables. Both use
aligned-F32 activations and eager coalesced single-token backward. The host
already has approximately 4 GiB swap in use, so zero growth does not establish
clean-host zero-paging qualification. E4B's compact replay is explicitly
ineligible for performance qualification. Do not infer a general speed ranking
from this table.

The frozen-PLE cache preserves the BF16 gather bytes, rejects unknown input
tokens, leaves the trainable PLE projection intact, and records its token set
and source shapes. Its independent E2B control produces a byte-identical
adapter to the full-table capture (all 552 tensors). Eager versus compiled
coalesced MLX is separately checked: 0.0123837% update-vector relative L2,
cosine 0.9999999923. Dataset prefix admission honors `max_examples` while
retaining the full-file fingerprint and selected source-row checks.

The final public CLI SHA-256 is
`caa38d1d293bf49616df2bf2fdf07bcdb40ab1c95af2b94a9e76cf6cf471196d`.
`final-binary-identity.json` binds the binary to the dirty worktree inventory
at build time. The final checkpoint and E4B adapter also match the earlier
accepted capture byte-for-byte. Primary results are
`e2b-final-64groups-metal-mlx-comparison.json`,
`e4b-final-metal-mlx-comparison.json`, and the corresponding quality summaries.

## Nonzero initial adapter reference

Both model sizes now have real-model checks starting from nonzero adapters.
E4B starts from the accepted trained adapter, takes two updates, and reports
`compiled-initial-adapter-snapshot`. Its baseline KL loss is 1.27670e-12
(mean KL 3.19176e-11); sampling/rescoring error is zero, and initial
policy/reference log-probability error is at most 1.52588e-5. The check takes
42.27 seconds, peaks at 10.55 GB sampled physical footprint, and adds no swap.
Its two-row reward-improvement gate fails as expected and prevents publication.
This verifies the reference choice at bounded scale; it is not an additional
quality qualification. See `e4b-nonzero-reference-summary.json`.

The retained E2B nonzero-start diagnostic reports baseline mean KL 5.0591e-12
and maximum policy/reference log-probability error 1.9074e-5. Its two-row
improvement requirement also fails. These reference errors are bounded, not
bitwise zero.

## Excluded attempts and evidence boundaries

Earlier E4B attempts stopped at the unchanged 256 MiB swap-growth guard.
The compiled compact MLX attempt reached 22.59 GB physical footprint and
1518.81 MiB swap growth at its first backward. A separate attempt failed
full-file/prefix validation before the runner correction. Those directories
and explicit process exits remain retained but are excluded from successful
quality, numerical, and timing claims. Smaller 16/2 E2B and 2/2 E4B probes
are retained as controls and superseded by the matched 64/32 results above.
No CUDA or HF/PEFT numerical qualification was attempted in this MLX-first work.
Stock PEFT coverage is structural export compatibility only.

## Remaining release and architecture work

- Run required hosted CI against the final submitted revision.
- Keep GRPO experimental until the larger multi-seed, longer-horizon quality
  campaign and the separate sealed holdout pass. The fresh E2B improvement
  gate remains open; the single-seed E4B pass does not qualify a distribution.
- Repeat on a clean host for zero-paging release evidence. CCE fallback now
  rejects logits tensors larger than 64 MiB; scalable chunked fallback remains
  future work.
- Finish per-lane recipe execution/planning extraction and immutable low-level
  kernel policy. Schema and objective validation are extracted, six repeated
  failure-reporting paths are consolidated, and unreachable surrogate and
  duplicate optimizer code are removed. Duplicate kernel routing remains.
- Historical notes are separated into `GEMMA4_HISTORY.md`; shared streaming
  hashing replaces eight Python owners. Serving/config regressions cover the
  narrowed Gemma4 metadata refinement and native BF16 borrowing lifetimes.

## Performance sequence after correctness gates

1. Freeze model/tokenizer/adapter/data/source identities and explicit optimizer
   settings. Refresh captures affected by renderer or numerics changes.
2. Run the complete same-host MLX-first matrix: E2B/E4B DPO fixed and Q128
   length policies, plus GRPO training, evaluation, and total-cycle timing.
   Pin work, sampler decisions, warmup, synchronization, dtype, topology and seeds.
3. Measure wall/GPU time, dispatch/submit/wait counts, transfers, physical
   footprint, RSS, and paging; identify the measured bottleneck before optimizing.
4. Change one critical path at a time. Keep experimental defaults gated until
   tensor bounds, discrete decisions, checkpoint recovery, reward/KL and memory
   acceptance pass.
5. Require repeated paired measurements showing at least 5% improvement over
   pinned MLX in the agreed matrix with quality and memory acceptance intact.
   The fresh numerical replay measurements above do not satisfy this gate.
