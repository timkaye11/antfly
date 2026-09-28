# Gemma training review remediation and performance work

Status: accepted-adapter recovery and development quality, September 24, 2026.
Local implementation checks and real-model diagnostics are in progress. GRPO
remains experimental; production qualification and hosted CI for the submitted
revision remain open.

The user has deferred E4B qualification because of the host memory limit.
The active qualification scope is E2B; E4B results and failures remain retained,
with no production qualification claimed. E2B multi-token SFT can proceed once
its own quality and recovery prerequisites pass. This scope change does not
relax any E2B quality threshold, resource guard or release requirement.

The full-size E2B DPO development campaign passes with the memory-corrected
binary: 1,960 distinct training pairs for one epoch, seeds 17/42/991, and the
unchanged 256 evaluation pairs. It retains learning rate 1e-6 and all quality
thresholds from the passing bounded candidate. The data audit verifies the
previous 64 training pairs are an exact prefix, all evaluation pairs are
unchanged, and tokenized prompts have no overlap or truncation. Evidence is
`full-e2b-dpo/input-audit.json` and `full-e2b-dpo/plan.json`.

Full DPO seeds 17, 42, and 991 each complete all 1,960 Metal optimizer steps
and pass the 256-pair evaluation. Held-out loss falls from 0.69314611 to
0.53620815, 0.53727198, and 0.53580046 respectively; all reach 75.78125% DPO
preference accuracy. This measures chosen/rejected policy-versus-reference
margins, not answer accuracy. Mean final loss is 0.53642686 across 5,880 total
updates. Independent review verifies report hashes, counts, metrics, and
unchanged absolute and baseline-relative thresholds. The resource guard exits
successfully after 11,216.44 seconds, with a sampled peak physical footprint
of 2.283 GB and no swap growth. The 1,015 host pageouts and existing swap keep
this a development diagnostic rather than clean-host performance evidence.
See `full-e2b-dpo/campaign/campaign_report.json` and
`full-e2b-dpo/independent-campaign-review.json`. Full midpoint recovery remains
pending.

The E2B multi-token SFT plan now has its own version, with the deferred E4B
prerequisite removed and every recipe, generation, scoring and correctness
threshold unchanged. E2B's three-seed quality and accepted-adapter recovery
prerequisites pass. The serial queue verified successful DPO completion and
process shutdown, then started multi-token SFT. A failed stage stops it without
retry. See `multi-token-squad/campaign-preflight-e2b.json` and
`e2b-squad-queue-plan.json`.

Multi-token SFT seed 17 passes its development quality gate after all 1,960
updates. On 256 questions from 88 held-out articles, token F1 rises from
78.1798% to 83.6141%, exact match from 54.2969% to 69.5313%, and completion
remains 100%. Article-level F1 improves on 43 articles and declines on 17,
with 28 ties (one-sided p = 0.000532883). Independent rescoring matches every
reported score, and all 16 cached/uncached checks agree. Reload loss exactly
matches 0.2732937959837675. Training peaks at 12.081 GB physical footprint
with no sampled swap growth; existing swap and 2,596 training pageouts exclude
clean-host performance claims. See
`multi-token-squad/e2b/seed-17/answer-quality-independent-review.json` and
`multi-token-squad/e2b/seed-17/training-reload-interim-review.json`.

Seed 42 also passes all 1,960 updates, exact 256-example reload, and development
answer quality. F1 rises from the same 78.1798% baseline to 84.6149%, exact match
from 54.2969% to 70.7031%, and completion remains 100%. Article-level F1 improves
on 42 articles and declines on 19, with 27 ties (one-sided p = 0.002222010).
Independent rescoring matches the recorded scores; all 16 cached/uncached checks
agree. Reload loss exactly matches 0.28447476921083414. Training peaks at
12.107 GB with no sampled swap growth. Reload samples 0.5 MiB of host swap
growth, and all phases retain host pageouts, so these remain functional and
development-quality results. See
`multi-token-squad/e2b/seed-42/answer-quality-independent-review.json`,
`training-interim-review.json`, and `reload-interim-review.json` in that seed
directory.

Seed 991 also passes all 1,960 Metal updates and exact reload over 256 examples
(loss 0.301092142807131). Answer F1 rises to 83.1006%, exact match to 66.7969%,
and completion remains 100%. Its article test has 44 wins, 23 losses and 21
ties (p = 0.006966938). Independent rescoring confirms every seed and the full
aggregate: mean F1 rises from 78.1798% to 83.7765%, mean exact match from
54.2969% to 69.0104%, and the aggregate article test has 44 wins, 23 losses and
21 ties (p = 0.006966938). All 5,880 optimizer updates, three exact reloads and
48 cached/uncached checks pass. The controller exits successfully with sealed
data unopened. See `multi-token-squad/e2b/independent-campaign-review.json`.
Paging remains present, so this is development-quality evidence. Multi-token
SFT recovery is prepared but unexecuted; its one-epoch checkpoint check covers
interruption before publication, not mid-epoch recovery. Native validation of
the GRPO stop correction passes after verified campaign shutdown, as detailed
below.

Review of all three complete training-process memory traces covers 7,845
samples across approximately 88 minutes per seed. Comparing the fixed 10–20
minute and 70–80 minute windows, peak physical footprint increases by only
344,088, 294,912 and 1,359,920 bytes for seeds 17, 42 and 991 respectively.
Every ten-minute window and the final partial window are retained in
`multi-token-squad/e2b/full-memory-envelope-review.json`. These bounded sampled
peaks support the observed memory-growth fix; they do not exclude unsampled
transients, other leak classes or the recorded host paging.

Completed SFT seeds 17 and 42 report 1,653 and 1,651 graph builds across 1,960
updates, respectively. A CPU replay of their target shapes accounts for every
build: sparse SFT uses the exact supervised-token count in the graph signature,
and the one-entry cache rebuilds when that count changes. Both datasets contain
33 target shapes, and 58.05% of scheduled input tokens are padding. This identifies
optimization candidates, but does not isolate their runtime cost or establish
that a larger cache fits memory. The frozen campaigns remain unchanged; see
`multi-token-squad/graph-rebuild-attribution.json`.

Two subsequent E2B campaigns are prepared but have not started: full-size GRPO
with 1,960 distinct groups per seed and unchanged quality thresholds, and DPO
recovery with SIGTERM at the durable midpoint checkpoint (980 of 1,960 pairs).
Recovery must reproduce the adapter from the accepted full-size seed-42 run.
The plans are `full-e2b-grpo/plan-v2.json` and
`full-e2b-dpo-recovery-plan-v2.json`; their preflight checks do not establish
model quality or recovery results.

A separate CPU process-lifecycle probe reproduced a guard-v4 shutdown bug:
the controller exited successfully while its sleep child remained alive, and
the guard incorrectly returned success. The probe cleaned up its own child.
Guard v5 now terminates remaining members of its owned process group, verifies
shutdown, and rejects the incomplete run. Twenty regression checks, the same
real orphan scenario, and a normal 100-child shutdown smoke pass. Resource
thresholds are unchanged. The completed DPO and active SFT bindings remain intact;
their queue independently checks that model processes have exited. Subsequent
GRPO and recovery plans use v5. See `guard-v5-validation.json`.

Further multi-token GRPO review found a stop-policy bug. The E2B model config
declares EOS IDs `[1, 106]`, but training passed only tokenizer EOS 1. A CPU
probe using the verbatim sampler with scripted logits produced
`[42, 106, 43, 43]` instead of stopping at `[42, 106]`, and the truncation
predicate also rejected 106 as a terminal token. Three of four checks failed.
The source now consults the existing model-configured EOS set in ranked,
eager-group and incremental-KV sampling, applies the same predicate to text
GRPO truncation, and binds the stop policy into GRPO checkpoint identity.
The corrected isolated sampler and native policy tests pass all seven checks.
The completed DPO/SFT campaigns retain their original binary and unchanged
bindings. Subsequent GRPO and recovery executions use the corrected binary.
Evidence is `grpo-stop-token-audit/result.json`.
After both campaign controllers and model workers exited, Debug, ReleaseSafe,
and ReleaseFast each passed 466 of 469 selected Metal tests, with three optional
fixture skips. All three new stop-policy tests passed in each mode. The public
CLI build completed all 37 steps, its help check passed, and guard v5 verified
shutdown. Peak physical footprint was 17.923 GB, with no swap growth; the host
recorded 410 pageouts. See `grpo-stop-native/result.json` and its guard summary.

The build also recorded a 3,263,823,872-byte CLI compiler peak above its former
3 GiB scheduling reservation. Zig reported this diagnostic without failing the
overall build. The macOS CLI reservation is now 4 GiB; Linux reservations and
the outer resource guard are unchanged. The existing scheduling-profile unit
test and a separate cached public CLI rebuild pass, retaining the same binary
SHA-256 `7b9114343d814a4ce01d478da521479b0b98abcd81af3b964df94db05d0de8ff`.
This validates the corrected build graph, not another cold compiler measurement.
Evidence is `grpo-stop-cli-rebuild/acceptance.json`; the rebuilt binary is under
that directory's `installed/bin/antfly`. The owned temporary build cache was
removed after verified compiler shutdown, reclaiming 2.93 GB of free space.

The multi-token GRPO development inputs are prepared from the same SQuAD
split: 1,960 training questions across 354 articles and 256 evaluation
questions across 88 disjoint articles. All prompt tokens match the native
prepared SFT inputs exactly; the longest prompts are 433 and 404 tokens,
leaving room for 64 completion tokens within the 512-token limit. The pinned
verifier embeds the reference answers and existing SQuAD scoring functions,
returns normalized token F1 only for answers ending with token 106, and gives
incomplete answers zero reward. Twelve CPU contract checks pass, including
all 2,216 gold answers and execution with the native empty environment.
The candidate recipe and protocol are frozen in `multi-token-grpo/protocol.json`.
The article scorer additionally passes 11 CPU checks: it replays retained
verifier exchanges, rejects changed evidence, and avoids treating questions
from one article or repeated training seeds as independent observations.
Every seed must improve reward and article direction without exact-match
regression, achieve at least 99% completed answers, and the aggregate article
test must pass p <= 0.05. Existing canonical GRPO thresholds remain required.
A bounded one-group probe completed on the corrected binary, retaining the
16-completion group, 64-token limit, reward contract and resource guards. Its
baseline produced 221 tokens across 16 completions, all ending at token 106;
independent replay verified every reward exchange. Baseline sampling took
462.61 seconds, or 94.69% of the 488.55-second evaluation loop. A one-second
stack sample confirms eager Metal generation; this diagnostic sample and the
existing host swap exclude release performance claims.

The longest-prompt training fixture then produced 16 identical five-token
answers, all ending at 106 and earning independently verified reward 1.0.
GRPO correctly rejected this zero-variance group with `NoGrpoLearningSignal`.
No optimizer update or final evaluation was demonstrated. The process exited
after 684.39 seconds with verified shutdown, 1.090 GB peak physical footprint,
and no swap growth, pageouts or swapouts. This is stop/reward-contract evidence,
not training integration. See `multi-token-grpo/integration-probe/review.json`.

A matched diagnostic using the existing opt-in compiled sampler produces the
same baseline and training completion tokens and rewards, and also rejects
zero learning signal. Baseline sampling takes 335.46 seconds versus 462.61
seconds eager (1.379x), and the evaluation loop takes 361.38 versus 488.55
seconds (1.352x). Overall duration is 512.77 versus 684.39 seconds. The compiled
run peaks at 10.571 GB rather than 1.090 GB, with 68.12 MiB swap growth, 1,021
pageouts and 4,360 swapouts. Both guards verify shutdown. This single diagnostic
pair is not clean-host performance or optimizer-integration qualification.
The compiled sampler's mid-epoch recovery rejection remains unchanged.

A broader compiled training diagnostic used eight records from
distinct articles, selected solely by source-ID hashes without consulting
answers or model outputs. All original sampling, optimizer and evaluation
thresholds remain unchanged. It completed baseline evaluation, then the guard
terminated training at 294.19 MiB host swap growth, above the unchanged 256 MiB
limit. Peak sampled physical footprint was 18.400 GB; shutdown was verified
after 784.22 seconds. No training reward trace, optimizer report, final
evaluation or accepted trained adapter was produced. Initial adapter and
source bindings remained unchanged. This leaves compiled multi-token GRPO
optimizer integration and its memory requirements unqualified; it does not
invalidate the completed SFT memory evidence. The retained result is
`multi-token-grpo/broader-probe/execution/result.json`.
Its prepared reviewer checks all 128 training completions,
per-prompt coverage and actual optimizer accounting; 13 CPU review regressions
pass. Full three-seed quality and exact recovery remain open. See
`multi-token-grpo/compiled-probe/zero-signal-comparison.json`,
`multi-token-grpo/broader-probe-inputs/selection.json` and
`multi-token-grpo/broader-probe/`.

## September 23 continued quality qualification

Fresh answer evaluation of the published seed-42 adapter reproduces every
prediction from the earlier staging evaluation: complete-answer accuracy is
80.078125% versus 75.0%, with 23 wins, 10 losses and p=0.01754102. The frozen
seed-17/991 confirmation campaign has passed with the memory-corrected binary.
Seed 17 now passes all 1,960 updates, publication, exact reload and answer
quality: 80.078125% versus 75.0%, with 26 wins, 13 losses and p=0.02662596.
Peak training physical memory is 4,233,517,296 bytes with no sampled swap
growth. Seed 991 also passes: 79.296875% versus 75.0%, with 24 wins, 13 losses
and p=0.04943587. Final and independent reload loss are both exactly
0.22630479299937178; all 1,960 updates execute on Metal. Its completed command
peaks at 4,233,795,896 bytes with no sampled swap growth. The first seed-991
attempt was stopped after 1,128 seconds when host swap grew by 455.31 MiB;
training peak physical memory remained 4,233,337,120 bytes. A coincident CPU
model-file check was stopped by the same guard. Both attempts are retained.
After a standalone CPU check and a host recovery observation, seed 991
completed alone with identical ordered records, prepared examples, recipe and
resource limits. Model-file reads are now serialized with GPU jobs as well.
All individual gates and the aggregate test pass. Averaging seed effects
within each of the 256 prompts gives 28 wins, 16 losses and p=0.04807088;
independent rescoring reproduces the result. Evidence is
`qkv-confirmations/three-seed-independent-rescore.json`. These are development
results; reserved acceptance remains separate.

The accepted seed-42 recipe now passes final-checkpoint publication recovery.
After 1,960 updates, SIGTERM exits with signal 15 before immutable adapter
publication. Resume reproduces the 421,809,976-byte trainer checkpoint
(`41d7f5837ad30300c2f7895dcce03f4f4c7339046fc7337e27eed2894b666758`)
and accepted adapter exactly; final loss is again 0.2405380331704805 on all
256 examples, with every strict Metal violation counter zero. The guarded
campaign takes 3,419 seconds and peaks at 4,252,441,320 bytes physical memory.
Swap does not grow, but 1,058 host pageouts and pre-existing swap exclude this
run from zero-paging performance evidence. This covers publication recovery
at the final epoch boundary; no additional optimizer steps run after resume.
Evidence is `qkv-full-seed42-recovery/qualification_report.json` and
`qkv-full-recovery-process/summary.json` in the same evidence root.
The E4B three-seed runner has completed seed-42 training with the same frozen
recipe and quality thresholds: 1,960 Metal updates, 256-example baseline and
final evaluations, and immutable adapter publication. Evaluation loss falls
from 2.8048525808844715 to 0.12501887674119416; every strict Metal violation
counter is zero. The command completes in 2,818 seconds at 5,964,079,168 bytes
peak physical memory. Host swap peaks 1.87 MiB above baseline, with 2,267
pageouts and 120 swapouts, so this remains diagnostic evidence. Independent
reload reproduces the final loss exactly on all 256 examples, at 3.46 GB
peak physical memory. The following MLX answer evaluator was stopped after
18.49 seconds by the unchanged 256 MiB swap-growth guard: growth was 479.88
MiB, sampled peak physical memory was 16.05 GB, and no quality result was
produced. The quality queue and dependent deployment queue both stopped.
An isolated base-load diagnostic then completed without swap growth, showing
14.93 GB of active MLX model memory. One fresh answer-evaluation attempt used
the same inputs and limits but again hit the swap guard after 16.80 seconds:
332.81 MiB growth at the stop sample, 489.44 MiB by shutdown. Both attempts
are retained; no answer-quality result was produced and no further E4B retry
is running. Completed training and exact reload remain valid. See
`e4b-seed42-training-review.json`, `e4b-seed42-reload-and-resource-stop.json`,
and `e4b-full-quality/seed-42/answers-retry1-plan.json` in the same evidence
root. Lower loss alone is not a quality pass.

The stock PEFT export/import boundary is now covered by a nonzero CPU fixture.
Direct native validation of the exported stock keys rejects
`UnexpectedAdapterTensor`, as expected. A qualification-only converter uses an
explicit Antfly origin, verifies base/tokenizer/template provenance and exact
tensor mapping, and publishes a separately validated internal artifact.
All four tensor payloads and the origin initialization seed survive the round
trip; changed stock weights are also preserved. Its 21 checks reject invalid
metadata, keys, shapes, nonfinite payloads, provenance mismatches, input drift,
and output overwrite attempts. Export sidecars establish the original
checkpoint binding; plain stock artifacts explicitly retain caller-asserted
lineage. Evidence is `peft-import-validation.json`. This does not establish
real E2B/E4B HF/PEFT numerical agreement or a public reverse-import command.

The accepted E2B seed-42 adapter now passes public CLI materialization and full
tensor verification. The first verifier stopped because it lacked the three
Antfly-to-HF per-layer projection aliases. A separate corrected verifier
passes all 12 original checks plus three alias checks, retaining every
arithmetic bound and the original failure evidence. The existing exported
checkpoint was verified without generating another model copy: all 276
adapted tensors pass the independent FP64 reference with the fixed FP32/BF16
bound, and all 1,735 untouched tensors are byte-identical. In total,
302,737,143 weight elements change; config and tokenizer files match their
sources exactly. The merged checkpoint SHA-256 is
`fddbcc40d93e02157f997d8727f685200f901615440ebfcead14482404a7ef6c`.
Native export takes 64.64 seconds and corrected verification takes 24.24
seconds, both without sampled swap growth. These are diagnostics, not
zero-paging performance measurements. Evidence is
`materialized-e2b/result.json`, `tensor-verification.json`, and
`verification-recovery-plan.json` under that materialization directory.
Generation and HTTP serving qualification remain separate.

The first merged-model CLI comparison passed all 32 raw-prompt F32-cache
cases, then rejected the first public-chat case because the fixture appended
four final-channel header tokens absent from the built-in non-thinking
template. The checked-in template and its explicit-thinking-mode tests agree
with the actual CLI prompt. Versioned corrected fixtures preserve all selected
example IDs and generation/resource limits; raw and public-chat entry paths
now require the same canonical tokens. All 21 generation/CLI contract checks
pass, including the observed prompt regression. A fresh MLX capture completes
64 cases and four cache-agreement checks, reproducing every unchanged raw
prediction exactly. The corrected 128-case CLI comparison now passes.
The original failed run remains under `merged-serving/e2b/cli`; corrected
evidence is under `merged-serving-v2/`, with
`merged-prompt-correction-validation.json` describing the fixture correction.

The corrected CLI run passed its first 20 comparisons before the resource
guard stopped on a non-positive footprint probe. The controller was still
live, so the existing root-process shutdown check could not admit a departing
CLI child. A versioned guard now verifies the failed member's PID, process
group, birth time and departure within 250 ms; a persistently live member or
unreadable process state still fails. Twelve checks pass, including a
deterministic regression that fails with the old guard. A CPU workload with
100 reaped children also passes and exercises one verified member-exit race.
Swap, RSS, physical-memory, disk and timeout limits are unchanged. The original
failure remains retained. All 128 comparisons pass under the corrected guard:
32 cases each for raw/F32, chat/F32, raw/F16 and chat/F16, with exact prompt and
output token IDs and finish reasons. Independent rescoring of the saved
responses reproduces every pass. The command takes 431.75 seconds with no
sampled swap growth, pageouts or swapouts; existing host swap still excludes
release performance admission. One verified child-zombie race is retained in
the guard report. All CLI runs log `ProcessIsolationRequired` during optional
decoder-runtime prewarm; compiled generation completes with exact outputs.
These checks establish output parity, not serving throughput or full device
residency. See `guard-v4-validation.json`, `e2b-cli-independent-review.json`,
and `merged-serving-v2/e2b/cli-guard-v4-plan.json`.

All 132 HTTP comparisons now pass on the same resident E2B server: 33 requests
per cache dtype and response format, including the first prompt repeated after
every other prompt. Streaming and non-streaming text, finish reasons and all
usage counts match the merged MLX reference. Independent replay of every
saved response confirms the result and both repeated-prompt pairs. The run
takes 558.80 seconds and peaks at 925,931,976 bytes sampled physical memory,
with no sampled swap growth or swapouts. Existing swap and 250 host pageouts
exclude zero-paging performance qualification. The owned server exits on
SIGTERM, is reaped, and is confirmed absent afterward. Its model bundle uses
verified hardlinks, so the HTTP checks create no second model-weight copy.
Evidence is `e2b-http-independent-review.json` and
`merged-serving-v2/e2b/http/result.json`. E2B merged-model export and serving
integration pass this recorded workload; E4B and broader release gates remain
open.

The additional multi-token QA protocol is frozen before model evaluation.
Its official SQuAD training snapshot yields 1,960 training examples from 354
articles and 256 evaluation examples from 88 separate articles, one question
per passage. Native preparation completes without truncation; evaluation
answers contain 2–40 tokens. The seven preparation, ten scoring, and eight
generation/input contract tests pass. The metrics agree with the pinned
official scorer on 1,000 comparisons (maximum F1 rounding difference
1.12e-16). The combined current Python contract suite passes all 833 tests
without skips; 14 CI path/selection tests also pass. The existing Gemma4 CI
lane discovers the new test files automatically. These local checks do not
replace hosted CI for the submitted revision. Real model generation, cache agreement, training quality, and reserved
acceptance are still pending. The existing BoolQ holdout and reserved SQuAD
development labels remain unopened. Evidence is under
`.benchmark-assets/gemma4-sft-production-20260923/qkv-confirmations/` and
`multi-token-squad/` in that same root.
The serial multi-token runner now binds its protocol, data, helpers and binary
to a frozen campaign plan. CPU preflight checks the real prepared data and
rejects nonzero native fallback counters and incorrect step counts. Each model
must complete all three seeds, publication, independent reload and paired
generation before its aggregate can pass. This preflight is not model-quality
evidence.

Performance input preparation now supplies exact, untruncated train/evaluation
rows at 128, 512 and 2,048 tokens for E2B and E4B. All 24 existing benchmark
binding checks pass across both models, both splits and accumulation 1/4;
the models share identical token and label workloads. Source metadata and
native provenance are verified. These are deterministic shape fixtures, not
task-quality evidence. Measurements for the 24-cell model/preset/length/
accumulation matrix and clean-host admission remain pending. The canonical
input evidence is `performance-inputs-v3/manifest.json` in the same root;
earlier attempts retain the rejected over-limit calibration and missing-source
metadata diagnostics.
Both Q/V seed adapters are now published, and standalone CPU verification
admits all 24 model/preset/length/accumulation cells with identical native/MLX
workload hashes and matching adapter provenance. The check used 58,311,136
bytes peak physical memory without sampled swap growth. Evidence and native
diagnostic command arguments are in `performance-matrix-preflight-v2/result.json`;
this remains input admission rather than measured performance.
The installed Python/package versions and Xcode SDK 26.2 meet the reference
requirements. The retained quality runtime uses pinned wheels; performance
still needs the locked source checkouts and native-build attestation required
by `run_gemma4_lora_mlx_benchmark.py`, in addition to clean-host admission.

## September 23 LoRA Q/K/V partial-execution memory fix

A fused LoRA Q/K/V attempt allocated Q's result before discovering that K's
base projection was unavailable at the current graph position. The region
declined, and ordinary fallback execution overwrote Q's value slot without
releasing that result. Allocation tracing found 15 unreleased Q buffers per
example, accounting for roughly 23 MB of growth per example.

The executor now checks all three branches' inputs before executing any branch.
A regression fails on the previous implementation and passes with the fix; it
covers each missing-base case and the fully ready path. Debug, ReleaseSafe,
and ReleaseFast each pass 463 of 466 selected tests, with three optional fixture
skips. The public ReleaseFast CLI build passes.

On the same 256 examples and staging adapter, loss remains exactly
0.2405380331704805 with zero strict Metal violations. Peak physical footprint
falls from 8,666,110,672 to 2,689,358,352 bytes, with no sampled swap growth.
This closes the standalone evaluation-growth reproduction. A guarded two-update
lifecycle also completes both 256-example evaluations and adapter publication
at 3,939,571,976 bytes peak physical footprint, with no sampled swap growth.
The full E2B all-linear seed-42 run also passes: 1,960 training and Metal
optimizer updates, 256-example baseline and final evaluation, successful
publication, and independent 256-example reload. The published adapter SHA-256
is `c36c55d6f7b561f668d13bcd204bdf28150505629780c1127c1a42d9941fc834`,
exactly matching the previous full trajectory. Final and reload loss are both
0.2405380331704805, with all five strict Metal violation counters zero.
The full command takes 1,656.997 seconds and peaks at 4,234,467,616 physical
bytes; reload peaks at 2,673,498,208 bytes. Neither process shows sampled swap
growth or a guard violation. This closes the measured E2B memory/publication
regression. The host had existing swap, so this does not establish zero-paging
production qualification.

Evidence is under `.benchmark-assets/gemma4-sft-production-20260923/`:
`qkv-memory-fix-impact.json`, `qkv-readiness-metal-validation.json`, and
`qkv-full-lifecycle-comparison.json`. The frozen lifecycle plan binds the
binary, native sources, inputs, runner, and resource guard by SHA-256.

## September 23 planned Metal encoder lifetime correction

The shared-buffer fix below does not resolve evaluation's large memory growth.
Further stack capture identifies unpooled Metal command encoders returned by
`termite_metal_decode_runtime_begin_planned_compute_scope`. That C entry point
now supplies a local autorelease pool; its successful scope remains owned by
the runtime's strong encoder field. A real-Metal regression proves cancellation
releases the encoder before the caller's outer pool drains: it fails with the
previous implementation and passes with this correction.

Debug, ReleaseSafe, and ReleaseFast each select 465 tests, with 462 passing and
three optional fixture skips. The public binary preserves exact loss across
256 examples and removes unpooled buffer/encoder warnings, but peak physical
footprint remains 8,666,110,672 bytes. The Q/K/V correction above addresses the
separate large allocation leak.

## September 23 shared Metal buffer lifetime correction

The remaining SFT evaluation investigation reproduced growth with in-frame
buffer reuse disabled and identical loss. Objective-C runtime diagnostics
reported 512 unpooled buffer references across 256 examples. Bounded stack
instrumentation traced them to `termite_metal_buffer_contents`, called while
reading embedding indices. Metal's interior-pointer annotation makes ARC
retain/autorelease the owning buffer; the Zig caller supplies no outer pool.
The accessor now drains that temporary reference locally while the caller's
owned handle keeps the returned pointer valid.

An actual Metal-buffer ownership regression fails with the previous accessor
and passes after the fix: releasing the handle must clear its weak reference
before the caller's outer pool drains. It also checks repeated shared contents
access, null handles, and rejection of private-buffer host access. Each focused
Metal mode (Debug, ReleaseSafe, ReleaseFast) selected 464 tests, with 461 passing
and three optional fixture skips. Its public binary completed the full 256
examples with exactly the same loss (0.2405380331704805), zero fallback, no
sampled swap growth, and no remaining unpooled buffer warnings. The leaked
buffers measured only 1 KiB each; peak physical footprint stayed essentially
unchanged (8,709,069,568 versus 8,713,132,728 bytes). This fixes a real ownership
error; the separate Q/K/V correction above resolves the measured memory blocker.

## September 23 SFT evaluation lifetime correction

Final SFT evaluation loaded a fresh model while the training backend, graph,
gradients, and optimizer slots remained live. The all-linear trial finished
all 1,960 updates and saved a staging adapter before its resource guard stopped
the command. The CLI now saves the adapter and reporting counters, releases
training resources, and then loads the independent evaluation backend. The
benchmark's deferred baseline evaluation also runs after this release. The
numerical-oracle path retains its trainer until gradient/moment capture.

A separate, guarded, read-only evaluation of the retained staging adapter
passes the unchanged development answer gate: 80.078125% complete-answer and
forced-choice accuracy versus 75.0% and 76.5625% respectively. Its paired test
has 23 wins, 10 losses, and 223 ties (p=0.01754102). All baseline predictions
match the earlier evaluator on the same 256 prompts. This is evidence about
the unpublished adapter's quality; it does not make the stopped training
command successful or qualify the recipe for production.

Metal Debug, ReleaseSafe, and ReleaseFast each pass 460 tests with three
optional fixture skips. The public ReleaseFast build passes after correcting
the build scheduler's initial memory-budget admission. A two-update real E2B
all-linear check completes publication and independent reload evaluation with
zero native fallback counters and no sampled swap growth. These checks cover
the lifetime change; they do not replace the full quality campaign.

The full rerun under `.benchmark-assets/gemma4-sft-production-20260923/`
completed all 1,960 updates and reproduced the staged adapter SHA-256 exactly,
but stopped during final evaluation at 364.81 MiB of sampled swap growth.
The lifetime correction was insufficient; the command did not publish its
output. A separate native evaluation of that same adapter completed all 256
examples with loss 0.2405380331704805, zero fallback counters, and no sampled
swap growth. Allocation tracing subsequently identified the Q/K/V ownership
error described above. The corrected full command now finishes publication
and independent reload; the retained seed 17/991 confirmation protocol has
not run yet.
The release plan preserves E2B/E4B,
multi-token, sealed-evaluation, recovery, numerical-oracle, PEFT, zero-paging
performance, and hosted-CI requirements.

The SFT recovery qualifier now supports the final checkpoint of a one-epoch
recipe. Its v3 contract requires exact final checkpoint bytes, terminal
evaluation agreement with strict Metal execution, stable prepared inputs and
binary, and optional reproduction of an accepted adapter hash. Report
publication preserves colliding temporary files and concurrent destinations.
Six regressions reproduced failures before these fixes; nine added tests cover
these cases and accepted-adapter/input binding. All 808 Python contract tests
pass after fixing a digest-prefix mismatch caught during validation. Real-model
recovery of the full SFT recipe remains pending; these are harness checks.

## September 22 checkpoint and publication hardening

Based on `aefbeb0d9e3b14982cacc7af61b34e81d4a2763e` plus the changes in this
worktree. Earlier binaries and captures do not qualify this patch.

- Immutable and mutable file publication now register cleanup only after
  exclusive creation succeeds. A colliding temporary file survives a failed
  creation; the existing destination stays intact and a later retry succeeds.
  The new regression reproduced deletion of the unowned file before the fix.
- Checkpoint inspection and restore reject negative Adam second moments before
  changing weights, optimizer slots, or counters. Such values are finite but
  invalid for the optimizer's square root. The regression corrupts a later
  slot, checks that earlier slots remain untouched, then verifies that valid
  negative weights/first moments and a zero second moment still restore.
- The main guide now separates historical evidence from current release status.

Current-source validation:

| Check | Result |
| --- | --- |
| Required Metal Debug / ReleaseSafe / ReleaseFast | 460 passed and three optional fixture skips in each mode |
| ML unit tests | 584/584 |
| Python contracts, pinned Python 3.12 | 787/787 |
| Isolated serving / graph / CLI contracts | 8/8, 24/24, 9/9 |
| Branch-added named-test selection | 342/342 selected; no missing tests |
| CI scope / selection-audit tests | 12/12 and 2/2 |
| Oracle lock, changed Zig formatting, whitespace | Pass |
| Public ReleaseFast CLI build | 37/37 build steps |

The three skipped Metal tests require optional GGUF or multimodal model
fixtures. They do not establish coverage of those real-model paths. Logs,
commands, source hashes, model identities, and diagnostic artifacts are retained
under `.benchmark-assets/gemma4-hardening-20260922/`.

Fresh real-model recovery diagnostics use the hash-verified BF16 E2B model,
rank-16 `peft-qv` adapters, seed 17, and the retained public binary with SHA-256
`1b403b24e528ec5f7670a84424154baecda26fe9be67c160388b4f805069bfc1`:

- SFT: two training and two evaluation BoolQ prompts, two epochs/four updates.
  A real interruption after epoch one resumes to a byte-identical adapter and
  checkpoint, with exact post-boundary metrics and no native fallback.
- DPO: eight training and eight evaluation BoolQ prompts, two epochs/16 updates.
  Interrupted/resumed and uninterrupted runs have byte-identical adapters,
  checkpoints and evaluation artifacts; terminal metric comparison is exact.
  This uses functional smoke acceptance, not the production quality campaign.

The public CLI also rejects a copy of the real E2B checkpoint with one
negative Adam second moment as `InvalidCheckpointSecondMoment`; no final
output directory is published. See `corrupt-resume.json`. Stock PEFT
load/save/reload passes with zero logit difference on the tiny structural
fixture; this is interchange coverage, not Gemma4 numerical qualification.

The unchanged 64-training/32-heldout-group E2B GRPO diagnostic completes
29 optimizer updates and 35 zero-variance skips. Mean held-out reward remains
0.759765625 and top-ranked reward remains 0.75. Absolute quality floors pass,
but baseline-relative improvement fails, returning
`GrpoBaselineRelativeEvaluationGateFailed` and withholding the trained
adapter. Its final KL loss is 5.341463520380785e-7. The run takes 256.23 seconds
and adds no sampled swap; this single-run diagnostic is not a performance
comparison. See `grpo-64groups/grpo_report.json` and `grpo-process/summary.json`.

Both recovery checks have zero sampled swap growth. Their reports are
`sft-resume/qualification_report.json` and `dpo-resume/qualification_report.json`
within the retained evidence directory. These bounded recovery checks do not
qualify task quality, a larger workload, E4B, or cross-backend numerical parity.

The host has 24 GiB RAM and had about 1.3 GiB swap in use at preflight. Runs on
this host are diagnostic evidence; a no-growth result cannot establish the
clean-host zero-paging release gate. Current-source E2B/E4B quality, parity,
sealed-holdout and repeated performance qualification remain required.

## September 23 SFT training-order investigation

The failed full-development SFT dataset contains 980 `yes` and 980 `no`
examples, but ends with 342 consecutive `no` examples. Balanced source-order
selection can create this tail after the other label's quota fills. Native SFT
processes prepared examples in file order; the trainer seed does not shuffle
them. A controlled repeat sorts records by SHA-256 of a seed and source ID.
Every record, prepared token, label, and initial adapter remains unchanged;
the longest same-label run becomes 11, and the last 128 examples contain 62
`yes` and 66 `no` examples. Learning rate stays 1e-5 for this comparison.

The shuffle-only run improves complete-answer accuracy from the failed
candidate's 63.28125% to 78.125%, compared with the 75.0% initial baseline.
Native development loss is 0.26290617 and reload agrees exactly. However, the
paired test has 19 wins, 11 losses, and 226 ties (p=0.10024421), so this
candidate **still fails** the unchanged p ≤ 0.05 gate. Forced-choice accuracy
is 78.125%. The result supports fixing record order, but does not qualify
the recipe. The predeclared 1e-5 confirmation seeds were therefore not launched.

The reusable `shuffle_gemma4_sft_dataset.py` preserves record bytes and writes
a seed/hash manifest; it reproduces the tested order byte for byte. The paired
`evaluate_gemma4_sft_answers_mlx.py` checks exact prepared prompts and adapter
tensors against pinned model/runtime/source archives, recomputes correctness
by prompt identity, and enforces complete-answer significance and forced-choice
non-regression. Twelve new tests cover permutation identity, output collisions,
pairing, termination, regression, and significance. All 799 selected Python
tests pass across the sandbox run and an isolated retry of the existing
localhost socket test (798 + 1). The native runtime and executable are unchanged.

Reducing only learning rate to 1e-6 also fails: native loss is 0.65511857,
reload agrees exactly, and complete-answer accuracy is 75.390625% (9 paired
wins, 8 losses, p=0.5). Forced-choice accuracy is also 75.390625%, below its
76.5625% baseline. The reusable evaluator's baseline predictions match the
original evaluator exactly on all 256 prompts. This lower-rate candidate
does not launch confirmation seeds.

Eight-example gradient accumulation at the shuffled control's 1e-5 rate also
fails significance: 77.34375% complete/forced-choice accuracy, 26 wins and 20
losses (p=0.23069559). Its 1,960 examples produce exactly 245 optimizer updates;
native loss is 0.37029628 and reload agrees exactly. Code inspection confirms
the accumulated mean is clipped before the optimizer update, with no partial
window in this run. Its confirmation seeds are not launched.

The `text-all-linear` candidate uses 276 targets instead of 50 Q/V targets,
retaining the shuffled control's learning rate, rank 16, alpha 32, seed 42,
one epoch, and single-example updates. Its fresh initial adapter has zero B
tensors; it is not a previously trained GRPO adapter. After 1,639 seconds,
the resource guard stops it because sampled host swap grows by 296 MiB,
exceeding the unchanged 256 MiB limit. Its log records 1,960 training updates,
but the command does not finish, publish its final output, or produce held-out
loss, reload, or answer-quality reports. The staging files remain diagnostic
evidence only. The process group has exited and no confirmation seeds launch.
A fresh resource-admitted run is needed to assess this recipe's quality.

Evidence is under `.benchmark-assets/gemma4-sft-quality-20260923/`, including
the three completed failures, the resource-stopped trial, frozen plans, source
snapshots, test logs, and a SHA-256 artifact index. No SFT recipe passes the
development qualification gates in this investigation. Acceptance thresholds
remain unchanged; the sealed holdout is unopened.

## September 23 accepted-adapter recovery and full SFT development data

The hardened executable passes the canonical mid-epoch recovery qualifier for
the accepted seed-42 DPO candidate at learning rate 1e-6: 64 training pairs for
eight epochs, 256 evaluation pairs, and 512 optimizer updates. The fresh control
run reproduces the previously accepted adapter. A second run is interrupted
at epoch index 3 after 32 examples (224 total updates), publishes no final
adapter, and resumes to the same result.

The final adapter SHA-256 is
`2284055fec3c4fec4dddcd3da73790e2aa8c978f1f85dc236706596e221dc81b`.
The resumed adapter, trainer checkpoint, and preference checkpoint sidecar are
byte-identical to uninterrupted execution; terminal report and evaluation
metrics match exactly. Evidence is under
`.benchmark-assets/gemma4-production-20260923/dpo-accepted-resume/`.
The campaign completed in 2,625.58 seconds without a resource stop. Host paging
counters increased despite no sampled swap growth, so this remains diagnostic
memory/performance evidence.

The same evidence directory also freezes a one-epoch SFT development candidate
with all 1,960 training examples, 256 evaluation examples, seed 42, rank-16 Q/V
adapters, and learning rate 1e-5. Preparation has no truncation or train/eval
overlap by source ID, group ID, record hash, prompt tokens, or full tokens.
The candidate **fails** the frozen answer-quality gates. Native held-out loss
falls from 6.21888673 to 0.65242779, and saved-adapter reload reproduces that loss
exactly with zero graph fallback counters. However, greedy complete-answer
accuracy falls from 75.00% to 63.28125%: 15 paired wins, 45 losses, 196 ties;
the one-sided improvement p-value is 0.99997888. Forced yes/no accuracy falls
from 76.5625% to 63.28125%. The final model predicts `no` on 216 of 256 balanced
examples, compared with 156 before training. No acceptance threshold changed.

Answer evaluation uses exact native-prepared prompt tokens in pinned MLX and
requires a correct yes/no token immediately followed by end-of-turn. This is
a task diagnostic, not native-generation or locked CUDA parity. The failed
adapter and reports remain available for diagnosis and are not accepted for
release. The reserved holdout remains unopened.

Post-failure teacher-forced evaluation in the same pinned MLX runtime attributes
about 97% of its loss reduction to the supervised trailing newline: mean loss
on that token falls from 16.05263 to 0.00006927. Answer-token loss falls from
2.40160 to 1.95631, while end-of-turn loss is effectively zero in both runs.
This explains how total loss can improve despite worse answer accuracy; it
does not establish the cause of the answer regression. The next SFT experiment
must evaluate answer quality independently of formatting-token loss and retain
the frozen acceptance gates. Native/CUDA parity remains a separate gate.
See `sft-loss-attribution.json` in the same evidence directory. The attribution
helper's first run failed on a variable-name collision; its logs are retained,
and a corrected diagnostic-only helper completed without changing the SFT run.

The clean-source oracle still rejects the dirty checkout. Existing CI runner
configuration covers CUDA inference and Metal tests, not the complete
finetuning qualification campaign; GitHub credentials and runner admission
remain unavailable locally. No source commit or remote mutation was performed.

## September 22 development quality campaigns

The same hardened executable passes the canonical v7 E2B GRPO development
quality qualifier for seeds 17, 42, and 991. Each seed uses a freshly bootstrapped
rank-16 all-linear adapter, 64 distinct training prompts for eight epochs,
256 evaluation prompts, 16 completions per group, one completion token, and
learning rate 5e-8. Acceptance thresholds were unchanged.

| Seed | Mean evaluation reward, baseline → final | Optimizer updates | Prompt wins / losses / ties |
| --- | --- | --- | --- |
| 17 | 0.68383789 → 0.68798828 | 209 | 19 / 11 / 226 |
| 42 | 0.69165039 → 0.69506836 | 224 | 17 / 7 / 232 |
| 991 | 0.68896484 → 0.69262695 | 233 | 20 / 6 / 230 |

All seeds pass directionality, reward, positive-group, optimizer-coverage, and
KL gates. Averaging paired reward changes across seeds at each independent
evaluation prompt gives 36 wins, 17 losses, and 203 ties; the exact one-sided
sign-test p-value is 0.00633017. Mean completion reward improves by 0.00374349.
There are 666 optimizer updates and no KL-budget rejections. The retained
one-epoch 64/32 smoke remains flat; this longer campaign is a distinct workload,
including seeded input permutations and the expanded evaluation set.

The initial sigmoid DPO candidate used rank-16 Q/V adapters, the same 64/256
training/evaluation prompt counts, eight epochs, and learning rate 1e-5. Seed
17 failed the baseline-relative held-out loss gate after 512 updates: loss
increased from 0.693146 to 0.822459, despite preference accuracy 0.777344. The
trained adapter was withheld. The failure is retained in
`dpo-quality/campaign_report.json`. The controlled candidate at learning rate
1e-6 passes all three seeds; all other recipe semantics, initial weights, data,
and acceptance thresholds are unchanged.

| Seed | Held-out DPO loss, baseline → final | Preference accuracy | Reward margin |
| --- | --- | --- | --- |
| 17 | 0.69314611 → 0.63417566 | 0.72265625 | 0.14240971 |
| 42 | 0.69314611 → 0.63367379 | 0.72656250 | 0.14348866 |
| 991 | 0.69314611 → 0.63434398 | 0.72656250 | 0.14161763 |

Each seed completes 512 optimizer updates and publishes its trained adapter.
Mean held-out loss improves by 0.05908163; all absolute and baseline-relative
gates pass. DPO preference accuracy measures positive policy/reference reward
margins, not answer accuracy. The canonical passing report is
`dpo-lr1e-6-quality/campaign_report.json`. All six published GRPO/DPO adapters
pass finite-tensor, target-pair, checkpoint-hash, and tokenizer-fingerprint
checks, retained in `adapter-integrity.json`.

Evidence and reproducible commands are under
`.benchmark-assets/gemma4-production-20260922/`; the canonical result is
`grpo-quality/campaign_report.json`. `release-gates.json` and
`release-command-templates.json` retain the remaining plan. The current oracle
admits 36 trace producers and 24 paired comparisons; twelve historical
HF-to-native comparison requests are outside its accepted producer roles.

This pass does not replace the retained 1,960-training-group E2B/E4B plan,
multi-token qualification, accepted-adapter recovery, or the reserved
254-prompt holdout. Development and reserved evaluation IDs are disjoint;
reserved labels were not read or evaluated in this work. The host began with
1,333.56 MiB swap in use. Sampled swap usage stayed constant, but host counters
increased by 1,243 pageouts and 64 swapins in an observed interval; these changes
cannot be attributed solely to training. This is not zero-paging evidence.
The passing DPO campaign separately records five additional host pageouts and
67 swapins, with unchanged swap usage and no resource-guard stop. It also
remains diagnostic memory/performance evidence.

Full qualification also requires clean committed source and a matching build,
the locked CUDA HF reference environment, more artifact capacity, clean-host
performance runs, and hosted CI. A read-only GitHub check could not run because
the local GitHub CLI is unauthenticated. No source commit or remote mutation
was performed.

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
