# EmbeddingGemma 2 production review

## Recommendation

The branch remains **unqualified for production**. Qualification is scoped to
**NVIDIA L4 / SM89**, as requested. The completed inference sweep attempted all
48 cells: **42 pass** over **4,200 paired requests**, with no measured release
speed-gate failures. Six B8/B32, 8,192-token cells remain inconclusive because
PyTorch reference selection encountered CUDA out-of-memory errors.

The preceding fused right-padding server, SHA-256
`e87f5e0acf22085c446b8f236625713476e9143b6bc483791e5d7cf2ac8b7f3c`, passes
fresh CLI, HTTP, media, contracts, input-limit, concurrency and capacity checks.
The unchanged release thresholds require cosine at least `0.999`, throughput
CI lower bound at least `0.9`, and p95 CUDA/PyTorch latency at most `1.1`.
Minimum cosine over the 42 paired workloads is `0.9998586051130127`.

| Workload | CUDA p50/p95 ms | PyTorch p50/p95 ms | Throughput ratio CI95 |
| --- | --- | --- | --- |
| B1/S512 prose | 13.286 / 13.668 | 14.686 / 16.331 | 1.1091–1.1291 |
| B1/S512 code | 12.054 / 12.791 | 14.085 / 15.593 | 1.1587–1.1829 |
| B8/S512 heterogeneous padding | 52.656 / 54.074 | 68.145 / 70.279 | 1.2895–1.2992 |
| B32/S512 heterogeneous padding | 222.739 / 227.162 | 269.071 / 275.608 | 1.2010–1.2087 |
| B32/S512 prose | 291.682 / 296.748 | 288.278 / 291.837 | 0.9863–0.9915 |
| B1/S2048 prose | 38.894 / 40.599 | 52.058 / 53.588 | 1.3262–1.3368 |
| B1/S8192 prose | 228.896 / 230.975 | 477.643 / 483.976 | 2.0864–2.0919 |

Eight of nine 512-token text workloads meet the stronger parity target:
throughput CI lower bound at least `1`, p95 ratio at most `1`.
B32/S512 prose is about 1.2% slower at p50; it passes the release thresholds
and remains an explicit parity miss. All references are selected per workload
from the six eager/SDPA compilation combinations. Flex attention is optional.

The reference worker now splits ordered batches only on typed CUDA OOM,
including compiler-wrapped OOMs. It releases failed-forward tensors before
retrying, remembers smaller successful batch limits, and preserves single-row
resource failures and unrelated errors. HTTP errors retain their typed response
body. A fresh 48-cell sweep will prioritize the six previous long-context
failures after current binaries and serving are requalified. Old raw evidence
is retained under its original binary/reference identity.

The real recovery probe exposed a second issue: B8/S8192 native inference
returns HTTP 500 with `CudaDriverError` when the PyTorch model is resident and
`cuMemAlloc` reports physical OOM. Native allocation now maps only CUDA's typed
OOM result to its existing memory-limit path, reclaims idle buffers even when
its own quota has room, and returns `ResourceLimitExceeded` if it still cannot
allocate. The embedding pipeline can then split the request; unrelated driver
errors remain unchanged. All three standalone allocator/budget tests pass,
including reservation rollback and subsequent successful allocation. The corrected L4 recovery probe now passes all three cases: B8/S8192 prose
and B32/S8192 heterogeneous padding, including an identical repeated request.
Minimum cosine is `0.9999340973611892`; the repeated request reuses the recovered
PyTorch batch limit without another OOM. This proves recovery and correctness,
not speed. The final CLI build, fresh serving/numerical checks and full paired
inference sweep remain pending.

Native source-inventory loading now admits an inclusive **64 MiB** limit while
keeping prepared shard manifests bounded at 16 MiB. Preparation, copying,
authentication and reopening use the same source bound. Large-inventory tamper
checks and exact/oversized boundary tests pass. The current build prepares
**25,000 training examples**, plus six retained nontraining rows; the previous
`StreamTooLong` preparation failure is resolved. Native dataset/job tests pass
**51/51** and the current Python qualification harness passes **108/108**, with
no skips. The inventory-only trainer, exporter and preparer rebuilds pass. The physical-OOM repair has rebuilt those tools and the benchmark server.
The CLI build continues before fresh serving and numerical checks, then the
48-cell paired sweep.

The latest guided candidate passes **45/48 independent development bounds**.
Three Recall@100 confidence lower bounds remain negative despite positive mean
deltas; all 24 retention bounds and target MRR/nDCG bounds pass. The predeclared
25,000-example trial keeps rank 16, alpha 32, batch 64, microbatch 2, query/document
lengths 128/512, four negatives and unchanged confidence thresholds. It runs
only after formal inference timing finishes. No new larger-trial quality result
exists yet; no finetuned candidate is release-qualified. Final three-seed
qualification still requires 60 official retrieval cells and 180 metric bounds.

Current work and job receipts are in
`/tmp/antfly-eg2-production-l4-20261011/resource-retry-production-progress.json`.
The new inference reports use `remaining-inference-matrix-progress-v12.json`;
previous sweep reports remain unchanged. Commits and Git operations are left to
the user.

## Active L4 qualification, October 11–13, 2026

The restored reference uses CUDA Torch 2.14.0+cu130, cuDNN 9.24.0.43, and Transformers
commit `a96730c8c97b8efbf35bbaf7f5da33ec99231a49`. Default CUDA passes the fresh HTTP
oracle's 71/71 gates, the 9/9 audio matrix, and the expanded 23/23 extended matrix.
Minimum audio cosine is `0.9991016600972747`; extended-matrix minimum cosine is
`0.9997612203954042`. These measurements belong to the 146,107,664-byte server with
SHA-256 `80541afdbfada1c878ce6b5051eb964faa3b6ba1d0fb51b80b7f50c3f6882817`.

The audio discrepancy came from CUDA contracting depthwise multiply/add operations
into FMA. Explicit separately rounded multiplication and addition match the host
accumulation contract. Direct L4 differentials require exact agreement for kernel
sizes 5 and 15, including adversarial cancellation across 1,024 channels per shape;
both pass with zero error.

The structural HTTP preflight fix now accepts valid manual-marker layouts,
including reordered mixed content. The grouped test target has explicit test ownership; it passes
33 checks, with 12 unavailable-backend checks skipped.

All five pinned official evaluation corpora are materialized and authenticated:
MS MARCO 8,841,823 documents / 6,980 judged queries; CodeSearchNet Python
22,176 / 22,176 and Go 14,291 / 14,291; SciFact 5,183 / 300; NFCorpus 3,633 / 323.
The quality gate independently pins these counts and all converted corpus/query/qrels
digests. The evaluator
preserves document titles, uses `CODE_RETRIEVAL` for code queries, and bounds
text preprocessing and HTTP batches. A float64 streaming evaluator scores all four
dimensions and all three seeds in one corpus pass. Each HTTP request binds the exact
native model identity to its authenticated bundle. Before the positive-alias fix,
the full Python qualification harness passed 96 checks. The current full harness
passes 102 checks with no skips, including four mining/dataset regressions and
two tests for the required reference configurations. This is harness validation,
not measured finetuning quality.

The selected image reference is SDPA with default compilation, after all nine
attention/compilation configurations finished without timeout. Six are valid; the
three stock Flex configurations cannot fit their shared-memory requirements on L4.
Before device preprocessing, the clean 100-pair comparison measures CUDA p50/p95 `72.680/74.104 ms`
versus PyTorch `60.599/61.825 ms`, with throughput CI `[0.8319, 0.8395]` and p95
ratio `1.1986`. Both gates fail. Retaining one device position-table copy per
tower removes redundant transfers but does not materially improve request latency.
The combined server/training/preparation/export build passes all 26 steps.

The current candidate moves Pillow-compatible RGB8 resize and scaled patch packing
to two CUDA kernels. The shared CPU supplies exact integer coefficient plans.
Seven L4 component differentials match every FP32 output bit, including padded
rows, extreme aspect ratios, and downsampling. For the benchmark image, the two
kernels take `0.048 ms`; the standalone CPU resize/normalization takes `8.885 ms`
p50. These component timings exclude transfers and request execution and do not
establish end-to-end speed or qualification. The rebuilt server passes all 26
build steps; all 2,132 recorded native/build/shared source files remain unchanged.

The fresh uncontended 100-pair image benchmark passes both speed gates against
the selected compiled-default SDPA reference. Native p50/p95 is
`53.348/54.881 ms`; PyTorch is `61.476/62.371 ms`. The throughput ratio's 95%
confidence interval is `[1.1508, 1.1628]`, and p95 latency ratio is `0.8799`.
Cosine remains `0.9999641518184414`. These measurements qualify this image
workload on L4; they do not complete the broader inference or training matrices.

Full evaluation decontamination removes one training query. Native preparation
also catches CodeSearchNet identifier collisions; the converter now binds IDs to
source and code content. The regenerated, decontaminated fixture has 952 training
records and prepares successfully. It remains a bounded functional training fixture.
The current native two-step run matches all 338 initial adapter tensors and resumes
all 1,014 state tensors exactly; PyTorch resume passes independently. The initial
broader BF16 comparison fails at `4.12%/5.95%` gradient relative L2. Layer traces
locate the first propagating difference on longer documents at RoPE: CUDA `powf`
recomputation changes BF16 coefficients at rounding boundaries. Using the
reference's 384 CPU-initialized FP32 frequencies matches every tested coefficient
over 512 positions in both head geometries. The rebuilt active-token document
trace then matches all 24 layer outputs and the final projection bit for bit.

The corrected trainer has SHA-256
`cdb54df50b5c60df5f61a89a14b6d75430065415d93361fd277051a2ddbf7bc7`.
Its fresh two-step gate passes all 1,356 checks: step-zero gradients are
bit-identical, and step one has cosine `0.99994507` and relative L2 `1.04809%`,
within the unchanged `2%` limit. Initial adapters and all 1,014 resumed state
tensors remain exact. The initial three reviewed seed runs completed preparation for
the official retrieval-quality matrix. Each completes all 952 records in 15
optimizer steps. Independent export verification passes for all three: every
one of 117,833,728 merged weight elements matches PyTorch's FP32 merge/BF16
rounding, all 1,207 untouched tensors remain byte-identical, all 338 FP32 adapters
reload exactly through PEFT, and copied sidecars and Adam tensor contracts pass.
Their training timings overlap the server build and are not performance evidence.

Four full-corpus quality roles now complete 48/60 cells. CodeSearchNet Python
covers all 22,176 documents and queries; its mean MRR/nDCG/Recall deltas are
nonnegative across the twelve cells. Go covers all 14,291 documents and queries.
Their paired confidence diagnostics fail 7/36 Python and 13/36 Go metric bounds
against the unchanged zero-delta target threshold. Positive mean deltas alone do
not establish non-regression. The twelve MS MARCO cells and the final authenticated
sixty-cell gate remain outstanding. The unfinished MS MARCO run for these already
failed candidates was stopped on October 12 to redirect L4 to candidate repairs;
the completed 48 cells remain preserved.

The first two full-corpus quality roles cover all 3,633 NFCorpus documents / 323
judged queries and 5,183 SciFact documents / 300 judged queries, across all three
seeds and four dimensions (24/60 cells). Mean nDCG deltas range from
`-0.00201` to `+0.00652` on NFCorpus and `-0.00437` to `+0.00645` on SciFact.
Three MRR confidence bounds fail the unchanged `-0.01` retention threshold:
NFCorpus seed 17 / 128 dimensions (`-0.01024`), and SciFact seeds 29 / 43 at
256 dimensions (`-0.01135/-0.01246`). The initial training recipe is therefore
not quality-qualified. The release thresholds remain unchanged.
The matching fifteen-step PyTorch eager BF16 run for seed 29 compares all 338
final FP32 adapters. Overall relative L2 is `0.25054%`; relative L2 of the changes
from initialization is `7.58536%`, with update cosine `0.997123`. A direct SciFact
diagnostic reproduces the cached native metrics exactly. The PyTorch-trained
candidate also fails the same 256-dimension MRR retention gate: its lower bound
is `-0.010554`, versus native `-0.011352`. This demonstrates a recipe failure in
the reference as well; it does not establish the cause of every other failing cell.

Candidate repair uses an expanded pool of 8,822 training examples, excluding all
release-test queries/functions and all development queries/functions. The new
native miner authenticates the frozen base identity and CUDA provenance, uses
code-query/document prompts with the training token caps, mines within each
language's training positives, and excludes repositories and known positives.
It mines 3,822 code queries and preserves MS MARCO's supplied negatives. The older
SentenceTransformer mining option now applies the same prompt and selection rules;
it previously encoded unprompted queries and documents.

Recipe selection uses fixed development queries against full validation corpora:
512 Python queries / 23,107 documents, 512 Go / 14,242, 512 SciFact training
queries / 5,183 documents, and 323 NFCorpus development queries / 3,633 documents.
These sampled development queries are ineligible for release promotion. A
predeclared learning-rate grid (`1e-5`, `3e-5`, `1e-4`) uses common seed 29, then
confirms the selected recipe with seeds 17, 29 and 43 before fresh full held-out
evaluation. All three learning-rate trials and the three-seed confirmation finish.
The `3e-5` and `1e-4` trials fail development retention; `1e-5` is selected by the
original macro objective and passes all 24 development retention bounds for each
seed. For seed 29, mean validation nDCG deltas are `+0.002418` for Go and
`-0.000802` for Python, giving macro delta `+0.000808`.

Fresh held-out evaluation of these three repaired candidates completes 48 cells.
All 72 NFCorpus/SciFact retention bounds pass, resolving the original measured
retention failures. CodeSearchNet still fails 32/72 target bounds: 25 Go and seven
Python. These candidates are rejected. MS MARCO is not started for an already
failing candidate. Checkpoints, PEFT adapters, exact-export receipts and complete
per-query arrays remain preserved; superseded merged weight files are reclaimable
derivatives with authenticated rebuild instructions.

The original development selector allowed a positive macro average to hide a
failing language, dimension or metric. The new `select_finetune_recipe.py`
requires all 48 distinct development bounds, using the unchanged target `0`
and retention `-0.01` thresholds. It rejects all three preceding candidates on
their existing development evidence. Its CLI binds complete eligible query
coverage and corpus/query/qrel digests to the report, rejects sampled development
sets, and never marks a candidate production-qualified. Seven regressions cover
masked losses, boundary thresholds, invalid numbers, missing/duplicate cells,
query inventory mismatches and evidence scope.

A separate predeclared v2 experiment evaluates every eligible development query:
23,078 Python, 14,191 Go, 807 SciFact and 323 NFCorpus, against the same complete
development corpora. Training excludes all of these queries and code functions.
It tests a code-focused mix of all 3,822 authenticated mined code examples plus
500 retrieval examples chosen by a fixed hash, retaining source negatives.
The learning-rate grid remains `1e-5`, `3e-5`, `1e-4`, with seed 29 for exploration
and seeds 17/29/43 required for confirmation. Every target language/dimension/
metric must pass before a recipe becomes eligible for separate held-out testing.
The first code-focused trial completes all 68 optimizer steps and passes the
independent exact-export check again. A token-level audit finds no identical
positive/negative inputs after clipping and no shared clipped training queries.
Expanded development evaluation is running; these trials provide no new
release-quality evidence yet. Separate held-out collection is queued only after
every confirmation seed passes development and the inference sweep releases L4.
The first expanded development evaluation finishes: Python passes all twelve
bounds, both retention datasets pass all twenty-four, and seven Go bounds fail.
The next predeclared `3e-5` trial is training. Positive means with confidence
intervals crossing zero remain ineligible.

The confidence gate now reuses resampling multiplicities across metrics with the
same query count. It keeps the reviewed seed, 5,000 draws and percentile indexes,
and sums repeated differences with `math.fsum` before dividing by the full query
count. It neither multiplies differences nor changes sample membership. All 144
completed metric bounds match the cached diagnostics bit for bit; direct checks
also match the original routine on full Python and Go query arrays. Four new
regressions cover dense, sparse, cancelling and zero differences, mixed query
counts, custom seeds and invalid inputs.

Controlled performance fixtures now populate all seven reviewed training shapes.
An independent 100-batch census verifies full active token caps and distinct
positive/negative candidates. These repeat public text to fill the caps, are
explicitly ineligible for retrieval-quality evidence. All seven workloads complete
100 measured pairs after ten warmups each. All performance and externally measured
memory gates pass against resident, uncompiled PyTorch/PEFT BF16 SDPA. Microbatch
size is two, LoRA rank is 16, and alpha is 32 throughout.

| Logical batch | Query/document tokens | Negatives | Native p50 (ms) | PyTorch p50 (ms) | Throughput ratio 95% CI |
| --- | --- | --- | --- | --- | --- |
| 16 | 32/128 | 0 | 690.476 | 6,291.240 | 9.070–9.150× |
| 32 | 32/128 | 0 | 1,373.412 | 12,034.662 | 8.712–8.765× |
| 64 | 32/128 | 0 | 2,738.110 | 24,066.683 | 8.748–8.794× |
| 32 | 128/512 | 0 | 2,942.096 | 12,314.434 | 4.182–4.199× |
| 64 | 128/512 | 0 | 5,921.461 | 24,230.486 | 4.094–4.112× |
| 32 | 128/512 | 1 | 5,184.906 | 18,516.965 | 3.567–3.582× |
| 64 | 128/512 | 4 | 23,761.949 | 71,885.269 | 3.000–3.020× |

The worst p95 native/reference ratio is `0.3271`. Maximum externally sampled
native peaks are 1,304,428,544 device bytes and 837,054,464 host bytes, below the
20 GiB / 8 GiB limits. `training-performance-qualification.json` binds the current
trainer, source manifest, workload audit, gate report and all 700 raw measured
pairs. This completes training performance qualification on L4; retrieval quality
and the broader inference performance matrix still prevent production promotion.
The remaining inference sweep runs without competing GPU qualification work.
Fresh standalone CLI parity, cancellation/session reuse,
manifest input caps, concurrency and maximum-length batch capacity checks precede
timing; those checks pass on the corrected server and CLI binaries. The sweep
prioritizes the 8,192-token text case and requires six eager/SDPA reference
configurations and 100 measured pairs per workload. Unexpected
reference runtime errors leave selection inconclusive; recorded hardware failures
remain distinct from timeouts and numerical parity failures.

Bulk artifacts remain under `/tmp/antfly-eg2-production-l4-20261011`; generated
corpora and per-query reports are not added as repository fixtures. The initial image
run overlapped unrelated work and is explicitly ineligible for qualification.

The current Python harness passes 102 checks with no skips. The standalone ReleaseFast shared
image suite passes 9/10; its existing Torchvision uint8 fixture also fails in the
unchanged indexed source. Pillow checks and the scoped L4 preprocessing path pass;
this does not qualify the separate CPU/Torchvision path.

The deployed SM89 CUDA artifact passes 190 fresh FP32 primitive differentials
against CPU PyTorch autograd: contrastive loss/coefficients, MRL normalization
and attention softmax/adjoints, including invalid targets and fully masked rows.
The separately built stripped production CLI passes all sixty oracle checks.
All sixty CLI checks and 190 primitive checks also pass afresh after adopting
the corrected build. The stripped production CLI has SHA-256
`a764c2fd3a2e73062e88f666b9685aeb6693ec6886fd598823b8bb3560898ac2`.
CUDA artifact bytes remain unchanged. Fresh B8/B32 maximum-context capacity,
all nine audio cases and all 23 extended media cases also pass. Corrected-server
paired performance still remains unmeasured.

The corrected server passes the fresh 71-check HTTP oracle, 14 contract checks
and 10 narrowed-input-limit checks. The full 26-check concurrency suite passes,
including 24 simultaneous requests and both indexed-error checks. Those two
checks initially used an unsupported mixed
array of strings and ordered groups; the collector now expresses every item as
an ordered group, retaining the same sibling preservation, deduplication and
error-code assertions. Their failed report is retained; the corrected fixture
passes sibling preservation and exactly one error for each failed item.

The operational audio comparison initially used eager PyTorch attention and
failed all nine pure-audio cases (minimum cosine `0.9777310`). The unchanged
fresh native session reproduces those failures exactly with eager, while SDPA
passes all nine (minimum `0.99910166`), isolating this discrepancy to reference
attention mode. The completed operational media checks use pinned SDPA without
compilation; eager diagnostic reports remain retained. The full performance
sweep considers all six eager/SDPA attention/compilation configurations and requires
the fastest parity-valid reference for each measured workload.

The v2 code-focused grid finishes without an eligible recipe: `1e-5`, `3e-5`
and `1e-4` fail seven, four and nine development bounds respectively. The middle
trial's remaining failures are Recall@100; it passes development MRR/nDCG and
retention. No v2 candidate is sent to held-out release testing.

A predeclared v3 experiment uses a frozen-teacher relative positive-score margin
of `0.05` with `3e-5` learning rate, one epoch and the same three required seeds.
It supports recording/removing anchors without an admissible guided negative.
All development bounds must still pass, with positive macro code nDCG gain;
the full held-out thresholds remain unchanged. Mining defaults remain unchanged,
and all previous checkpoints, corpus inventories and measured arrays are retained.
Training-only mining has completed with the corrected server: all 3,822 code
anchors retain an admissible negative, and 52 selections change (42 Python,
10 Go). Every selected cosine meets the declared margin; all other rows and
anchor fields remain unchanged. Native preparation passes for all 4,328 records
(4,322 training), bound to the mined dataset and manifest digests. Candidate
Seed 29 completes all 68 optimizer steps, and its
independent export verifier passes (169 adapted tensors, 1,207 untouched tensors
and exact PEFT reload). Full development checks finish with three failing
Recall@100 bounds: Python 256 and Go 256/512. All 24 retention bounds and all
target MRR/nDCG bounds pass; mean macro code nDCG delta is `+0.00372936`.
The candidate remains ineligible, so seed confirmation and held-out testing
are skipped. Its checkpoint, export verification and full measured arrays remain.

The three misses are small in absolute terms. The following deltas are percentage
points, measured on the full independent development inventories, not the official
held-out release corpora. The unchanged target gate requires the lower bound to
be at least zero.

| Recall@100 check | Mean delta (pp) | 95% CI lower bound (pp) | Queries improved / regressed |
| --- | ---: | ---: | ---: |
| Python, 256 dimensions | +0.008666 | -0.008666 | 3 / 1 |
| Go, 256 dimensions | +0.028187 | -0.042280 | 15 / 11 |
| Go, 512 dimensions | +0.042280 | -0.021140 | 13 / 7 |

The current server's first fresh inference cell passes all 100 measured pairs:
B1/S8192 natural text has native p50/p95 `228.591/232.150 ms`, versus
`478.804/486.258 ms` for the selected default-compiled SDPA reference. Its
throughput ratio CI is `[2.088458, 2.095167]`, p95 latency ratio is `0.477421`,
and cosine is `0.9999591679853352`. All nine baseline configurations conclude;
six qualify and three FlexAttention configurations exceed L4 shared memory.
This is one of 48 required workloads, bound to the corrected server SHA-256
`7f29092bb9e074eb37b5f9714e6fa18c48b1641f7dfe02d571f13df6c5c37591`.

The same server's B1/S16 natural-text cell also passes all 100 pairs: native
p50/p95 is `8.913/9.886 ms`, versus `13.033/16.327 ms` for the selected
default-compiled eager reference. Throughput CI is `[1.465336, 1.504367]`, and
p95 latency ratio is `0.605500`. B1/S16 source code also passes: native
p50/p95 is `8.874/9.220 ms`, versus `13.164/14.336 ms` for default-compiled
eager. Throughput CI is `[1.479392, 1.504435]`, p95 ratio is `0.643118`, and
cosine is `0.9999389731197371`. B1/S16 heterogeneous padding passes with the
six-reference profile: native p50/p95 `8.827/9.272 ms`, versus `12.830/13.610 ms`
for default-compiled eager; throughput CI `[1.443314, 1.466775]`, p95 ratio
`0.681283`, and cosine `0.9999323654671057`. Four of 48 workloads are complete.

The qualification runner now retains its private compiler cache up to 2 GiB,
reclaiming it when less than 3 GiB remains free and requiring at least 2 GiB
before the next compilation. The transition occurred between completed cells;
both existing reports were authenticated and retained. Compilation remains
outside the measurements, and every workload selects among all six required
configurations before collecting 100 paired samples. Four cache-policy and
measurement-validation checks pass.

FlexAttention is now optional, as requested. It was a PyTorch comparison backend,
not a dependency of the native CUDA kernels. The required reference profile is
eager and SDPA, each with compilation disabled, `default`, and `max-autotune`;
`--include-flex-attention` retains the additional diagnostic configurations.
The earlier optional-Flex compiler failures are archived and affected cells are
being rerun with the six-reference profile. The three completed paired reports
were independently revalidated against that profile. All 48 workloads, 100 pairs
per workload, numerical thresholds, performance gates and finetuning quality
requirements remain in force.

The inference sweep continues collecting the other 44 workloads. A predeclared
larger-data experiment targets 10,000 clean code
examples per language plus 5,000 retained MS MARCO examples, with the same
guided margin and an initial learning rate of `3e-5 * 68 / 391` to preserve the
approximate integrated learning-rate budget. Complete development and official
exclusions remain enforced. Source preparation yields during formal inference
timings; this experiment has no candidate quality result yet.

Training preparation now closes a reproduced false-negative path: deduplicating
a query could retain an alternate positive's ID while losing its content-to-ID
binding, allowing an identical code body with another ID to become a negative.
Known positives now expand to every same-split normalized-content alias before
query deduplication. The streaming collector preserves discarded positive-ID
bindings and resolves them against its retained training pool. Cross-split
aliases remain excluded, and conflicting document-ID bindings are rejected.
Duplicate-query rows also reserve their repository and alternate positive
content before merging; regressions cover both paths into a lower-priority split.
The idle larger-data collector was replaced before collecting or training on
the new source; its manifest will bind the corrected helper and collector.
Previous candidate measurements remain preserved. This repair does not establish
that alias errors caused the earlier quality misses or that the new trial will pass.

## Earlier final-review snapshot, October 11, 2026

Reviewed `feat/embeddinggemma2-cuda`, including the resolved incoming main merge,
model identity and admission, grouped serving, native training/checkpoint/export,
qualification collectors, and generated SDK contracts.

The review fixes grouped serving's narrowed-token-limit bypass, including media
delegation on other native backends. Ordered groups now use the loaded manifest's
token and modality limits, rechecking the active model after acquisition. Fail-fast
responses include backend provenance. CUDA overflow and modality failures retain
their nonretryable client-error codes in per-item responses; a missing model identity
returns a mismatch instead of an unchecked nullable unwrap. Boundary regressions
exercise 512 tokens, 513-token overflow, and inputs larger than the tokenizer window.
The narrowed-cap HTTP collector now includes singleton groups, grouped arrays,
audio, and a mixed-success per-item batch.

Fresh L4 warmup exposed a CPU/CUDA handoff regression: the CPU loader kept all
EmbeddingGemma 2 weights lazy, while CUDA copied only the resident inventory and
closed the temporary host session. The CUDA handoff now materializes those weights.
The subsequent endpoint run exposed BF16 audio subsampling filters being passed to
the FP32 convolution kernel. CUDA now widens those two small filters once during
upload. The encoder and media linear projections retain their BF16 execution.

The corrected ReleaseFast CUDA/SM89 `bench-server` build passes all 17 build steps
and fresh L4 warmup. Its 146,032,960-byte binary has SHA-256
`5a703dc7a4645a8b0b8cd42d964faf498fb9e2ac4cb62f5cd007da9b4678e937`.
The hardware smoke passes 13/13 endpoint cases, and the updated persistent
`qualify_input_cap_endpoint.py` collector passes its overlapping 10/10 cases.
Coverage includes exact 512/513-token boundaries in plain, singleton-group, and
group-array requests; image expansion; successful 16 kHz audio execution; permanent
per-item overflow errors; CUDA provenance; finite unit vectors; and 128/256/512 MRL
consistency. All 13 pinned model artifacts match their expected sizes and hashes.

These new measurements cover serving contracts and native consistency. They do not
execute a PyTorch reference model or establish fresh accuracy or performance parity.
The restarted environment lacks the earlier cuDNN runtime, so the smoke also exercises
the CUDA fallback configuration. The exact binary, source manifest, and reports are
bound in `/tmp/antfly-eg2-final-review.json`; raw endpoint reports are
`/tmp/antfly-eg2-final-review-endpoint.json` and `/tmp/antfly-eg2-review-input-cap.json`.
The release still needs fresh oracle and paired-performance evidence with the
qualified runtime configuration.

Earlier in this final review, all 104 native training/checkpoint/export tests and
the focused grouped-serving regressions passed. After the environment restart,
all 55 Python qualification-harness tests pass without skips in the restored
model-free environment (CPU Torch 2.14.0, NumPy 2.4.6, Pillow 12.3.0, SciPy 1.17.1),
and all 306 Python SDK tests pass again. Python generated-client regeneration and
Rust specification synchronization are clean; Go SDK packages passed after the
merge resolution. The checks cover integration and contracts, and do not establish
new model-quality or performance results.

The inference measurements below belong to the October 8 binary. The V22 finetuning
measurements belong to October 9. Neither set qualifies the current merged branch or
the new serving fixes. See the compact indexes
[`results/l4-2026-10-08.json`](results/l4-2026-10-08.json) and
[`results/finetune-l4-2026-10-09-draft.json`](results/finetune-l4-2026-10-09-draft.json).

## Finetuning qualification

Native CUDA text LoRA supports ranks 8/16/32, 169 adapter targets, checkpoint/resume,
FP32 optimizer masters, merged BF16 bundles, and PEFT export. Adapted bundles retain
their text-only capability restriction.

V22's two-step smoke passes 1,356 numerical checks: step-zero gradients are
bit-identical to PyTorch, and step one has cosine `0.99989947` and relative L2
`1.41792%`. Resume reproduces all 1,014 final-state tensors exactly. Independent
PyTorch export verification matches all 117,833,728 adapted elements across 169
targets, keeps 1,207 untouched tensors byte-identical, and reloads all 338 FP32 PEFT
adapter tensors exactly.

The tiny paired collector measured native steps at `143.01/91.68 ms` versus
PyTorch SDPA at `1217.94/858.24 ms`, with one warmup and two pairs. This validates the
collector only; it is insufficient for a production performance claim.

Outstanding required qualification:

- Official full-corpus retrieval quality: five dataset/language roles, four MRL
  dimensions, and three seeds (60 cells).
- Seven fixed training workloads with 100 resident alternating pairs per workload.
- L4 numerical, resume, and export evidence for the current merged branch.

The recorded result remains `qualified=false`; the bounded retrieval smoke remains
`production_quality_eligible=false`.

## Scope

This change adds BF16 `google/embeddinggemma-2` inference to the Zig CUDA runtime for
text, image, audio, and ordered mixed content. It includes:

- the official 24-layer text encoder, task prefixes, 768-dimensional output, and
  trained 128/256/512/768 MRL dimensions;
- native image and audio preprocessing and the official vision/audio towers;
- resident BF16 CUDA execution, cuBLASLt projections, cuDNN vision and bounded/local
  text attention, strict topology and weight validation, and cancellation-safe cleanup;
- strict 8192-token expanded-input accounting, bounded media admission, per-item error
  behavior, malformed-placeholder rejection, and controlled execution;
- manifest/session/server/CLI integration plus a pinned, verified Hugging Face bundle;
- reproducible oracle, contract, concurrency, capacity, and paired-performance tools.

Video and quantized EmbeddingGemma 2 tiers are outside this change.

## Historical inference model and build

The catalog pins `google/embeddinggemma-2` at Hugging Face commit
`914f7f89142e33e77833254d9c9b90c3cef7303b`. The 1,488,915,288-byte weight file has
SHA-256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`;
all 13 bundle artifacts are verified by size and digest and installed with a managed v2
receipt.

```sh
zig/pkg/inference/zig-out/bin/antfly-inference pull \
  google/embeddinggemma-2:bf16-safetensors-bundle-v1 --models-dir .models
```

The October 8 reviewed source manifest is
`results/l4-2026-10-08-source-manifest.json`, SHA-256
`e0975c97287f640dc874a805cf4e586931e41bf8ae78d67435f0dab0d2c2f51e`.
The stripped ReleaseFast CUDA/SM89 binary is 46,233,640 bytes, timestamped
2026-10-08 02:55:33.823 UTC, with SHA-256
`bda59bbadd9fdc2d1af83abb2312d751eb8df780a54140bc75b676d76f1f73b9`.
Portable CUDA artifacts are shipped, but the architecture-specific WMMA path has been
qualified only on SM89/L4.

## Historical inference correctness and contract evidence

The October 8 binary passes all 60 CLI oracle gates across document, query, code-query,
image, audio, and ordered mixed inputs at 768/512/256/128 dimensions. The minimum
cosine is `0.9998520645` on the 16 kHz audio fixture; every output is unit-normalized,
reports CUDA provenance and exact expanded token usage, and each MRL result equals
truncate-and-renormalize of the 768-dimensional result. The report is
`results/l4-2026-10-08-raw/cli-final-oct8-bda59bb.json` (SHA-256
`4450a3f84cb6b8ce84fe93251fa490669c032f50044fc0b9d85b4765346e6a00`).
The corresponding final HTTP oracle passes 71/71 gates; its report is
`results/l4-2026-10-08-raw/http-final-bda59bb.json` (SHA-256
`d60229ae986ddfcbab18af82161ceb9f8a8bda167187d8ec5edc24d9eae2ed77`).

Model-free validation includes 35/35 focused inference tests, 6/6 client tests with
loopback enabled, and the final sparse-mel helper tests. The sparse mel path preserves
ascending accumulation order, falls back to the dense loop for any nonfinite magnitude,
and is bitwise identical for all 12,672 representative finite sums. Its standalone
ReleaseFast benchmark reduces the 99-frame accumulation from about 4.25 ms to 0.096 ms;
the BLAS path is unchanged. Evidence is recorded in
`results/l4-2026-10-08-sparse-mel.json`.

The tracked CUDA attention differentials cover local/global text shapes through sequence
8192 and vision attention, with minimum cosine above `0.9999946`. The cuDNN diagonal-band
proof implements the official inclusive ±512 local mask and has direct B32/S2048 and
B1/S8192 numerical coverage. Maximum-context B8/B32 admission and execution have also
been exercised on the L4 without relying on quadratic scratch estimates.

## Historical qualification gaps

The October 8 nine-case audio matrix **fails the production quality gate**. The 1-second,
44.1 kHz case reaches cosine `0.998923725795`, below the required `0.999`; see
`results/l4-2026-10-08-raw/audio-matrix-final-bda59bb.json`. This is materially below
the final 16 kHz fixture and the other modality cases. A diagnostic run forcing all
audio operations onto the host reaches `0.999106404`, but that is not the production
path and is not counted as a pass. The antialiased Zig resampler has independent
SciPy/PyTorch parity tests. The October 11 qualification traced the remaining
discrepancy to depthwise FMA contraction and corrected it, as recorded above.

The final `bda59bb…` binary passes six of seven 100-pair L4 performance cells. Each row
also passes its numerical cosine requirement. Throughput is the 95% native/reference
ratio confidence interval; latency is native/reference p95.

| Cell | Native p50/p95 (ms) | Compiled reference p50/p95 (ms) | Throughput 95% CI | p95 ratio | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| B1/S128 | 10.680 / 11.942 | 17.113 / 18.601 | 1.5794–1.6217 | 0.6420 | Pass |
| B8/S512 | 63.597 / 65.000 | 67.635 / 70.123 | 1.0606–1.0692 | 0.9269 | Pass |
| B32/S2048 | 1359.863 / 1378.999 | 1652.928 / 1674.936 | 1.2136–1.2173 | 0.8233 | Pass |
| B1/S8192 | 227.035 / 230.316 | 472.038 / 476.759 | 2.0773–2.0828 | 0.4831 | Pass |
| Image | 69.091 / 70.294 | 61.830 / 64.788 | **0.8942–0.9031** | 1.0850 | **Fail: throughput lower bound < 0.90** |
| Audio (16 kHz) | 30.882 / 32.796 | 47.017 / 58.057 | 1.5443–1.6273 | 0.5649 | Pass |
| Ordered mixed | 86.653 / 101.722 | 84.351 / 96.937 | 0.9598–1.0010 | 1.0494 | Pass |

The sparse-mel optimization closes the preceding mixed p95 miss; the final mixed cell
passes. Image remains a measured production performance failure even though its p95
ratio passes. Raw reports are named `formal-final-bda-*.json` under
`results/l4-2026-10-08-raw/`. The compact, authoritative qualification index is
`results/l4-2026-10-08.json`.

These four text cells cover 4 of the 45-cell text matrix; the remaining 41 cells are
unmeasured. Secondary GPU architectures are likewise unmeasured. Neither limitation is
inferred from the passing SM89 cells.

Operational contract (14/14), concurrency (26/26), narrowed-capability (4/4),
large-context, and extended multimodal (20/20) evidence was run on the immediately
preceding `4aa4dda…` binary. The only runtime change between those October 8 binaries
is the bitwise-identical sparse mel summation helper, so this is retained as precise
historical regression evidence rather than represented as a fresh `bda59bb…` run.

## Historical SDK and generated-source validation

Generated OpenAPI clients are synchronized. The generator also carries pre-existing
ChatGPT and Apple schema drift that the repository checker requires; those unrelated
lines are retained so regeneration is stable rather than hand-edited away.

- Go: all packages pass.
- Python: 266 tests pass, including the generator check.
- TypeScript: typecheck passes.
- Rust: specification synchronization passes; a Rust compiler was not available in the
  qualification environment.

Reviewers can reproduce local and GPU checks from `README.md`. Resolve the measured
inference failures, complete the required training qualification, and bind fresh GPU
evidence to the release source and binary before production promotion.
