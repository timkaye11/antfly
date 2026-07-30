# Gemma 4 QAT Metal Baseline Performance Roadmap

## Status

Updated 2026-07-29 for the Apple M4 `ReleaseFast` path and the
`google/gemma-4-E4B-it-qat-q4_0-gguf` model.

The objective is to make ordinary, single-token Gemma 4 generation beat the
pinned llama.cpp comparator end to end. Multi-token prediction can remain an
additional accelerator, but it is neither a dependency nor a substitute for a
fast baseline model path.

The current conclusion is deliberately conservative:

- the production baseline is close enough that sub-percent effects matter;
- no experiment in the latest tranche cleared its declared whole-model
  promotion gate;
- all new Q4_0 pair-activation and concurrent-dispatch routes therefore remain
  default-off;
- the most recent pinned llama.cpp artifact predates the current binary and
  working-tree tranche, so it is a prior anchor rather than the current gap.

The first next step is a fresh, provenance-complete comparator run. Do not quote
a new llama.cpp gap until that run exists.

## Scope and guardrails

This roadmap covers the compiled whole-model Metal route for baseline Gemma 4
QAT prefill and greedy decode.

It follows these constraints:

- Preserve the canonical prompt and generated token IDs exactly. A different
  token stream is a correctness failure, not a performance result.
- Keep accumulations in F32. Any reassociation must pass a declared strict
  tensor tolerance plus exact token and long-context gates; reduced-precision
  intermediates require a separate precision contract.
- Prefer reusable Q4_0, paged-attention, graph-planning, and Metal runtime
  primitives over a Gemma-named monolithic kernel.
- Change and promote one route at a time. Every candidate starts behind an
  explicit enable switch and retains a disable/rollback switch.
- Use fresh processes for policy A/Bs because Metal policy is captured during
  runtime construction.
- Require route counters, no fallbacks, no host output, and binary/model/device
  provenance in every model-level claim.
- Never promote from a kernel microbenchmark alone.
- Do not change command-buffer or encoder topology merely to collect a timing
  number. Diagnostic captures that alter execution are qualitative evidence
  only.
- Treat MTP and speculative results as a separate track.

Explicitly deferred or rejected:

- unsafe f16 FFN variants that changed the token prefix;
- additional generic row-one Q4_0 MMV variants without evidence of at least 2%
  plausible end-to-end headroom;
- performance work on concurrent planned dispatch until determinism is fixed;
- enabling the down-projection sum-of-squares epilogue before its previous
  crash is root-caused;
- rowwise prefill and PLE-only fusion, which are lower-payoff than the measured
  attention and matrix-multiply surfaces.

## Canonical workload

The primary acceptance workload is the SearchAF-shaped raw prompt already
owned by
[`benchmark_metal_gemma4_long_output.sh`](./scripts/benchmark_metal_gemma4_long_output.sh):

| Contract | Required value |
| --- | --- |
| Model | Gemma 4 E4B QAT Q4_0 GGUF, exact file SHA-256 recorded |
| Prompt | canonical raw prompt, 2,003 tokens, exact prompt-ID digest recorded |
| Output | 300 tokens, greedy, `--ignore-eos`, exact generated-ID digest recorded |
| Antfly route | Metal, compiled, whole-model |
| KV cache | f16 for both implementations |
| Comparator | pinned `llama-completion` build 10182 and binary SHA-256 |
| Process order | balanced ABBA/BAAB ordering with a fresh process per sample |
| Thermal control | warmup, 45-second cooldown, interleaving, and CV gate |
| Reporting | prefill latency, decode latency/throughput, total latency, route counters |

Shorter outputs are screening gates, not the success criterion. Use 64 tokens
for determinism and route checks, 128 tokens for candidate screening, and the
full 300-token output for promotion.

The workload matrix used before default-on promotion must also cover:

- prompt lengths near 512, 2,003, 4,096, and 8,192 tokens;
- 64- and 300-token outputs;
- both sliding-window and global-attention rollover boundaries;
- aligned and ragged prefill row counts.

If tokenizer behavior prevents the stock prompt from landing on a requested
boundary, record the exact realized token count and digest rather than calling
it the nominal size.

## Latest experiment ledger

All latency ratios below are candidate divided by baseline, so lower is better.
Throughput ratios are candidate divided by baseline, so higher is better. These
were three-pair, 2K+128 screens; they are not promotion evidence.

| Candidate | Total | Prefill | Decode latency | Decode throughput | Correctness | Decision |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Q4_0 pair decode `nr8-nsg2` | 0.99575 | 0.99735 | 0.99338 | 1.00667 | exact tokens and routes | Parked: missed the 0.995 total and 0.99 decode gates |
| Q4_0 pair prefill `m32-n64` | 0.99351 | 0.98457 | 1.00839 | 0.99168 | exact tokens and routes | One cooled retest: prefill signal was promising, but decode regressed and CV reached 19.7% |
| Concurrent planned dispatch | n/a | n/a | n/a | n/a | all three 64-token outputs diverged | Blocked on correctness; do not benchmark |
| Dispatch-boundary stage timing | n/a | n/a | n/a | n/a | failed closed with `supported=0`, `complete=0` on this M4 | Use whole-frame counters, calibrated signature costs, and offline capture |
| f16 FFN experiment | n/a | n/a | n/a | n/a | token prefix changed | Rejected |

The decode pair result is especially useful negative evidence: it materially
reduced the number of split Q4_0 linear dispatches, but cumulative whole-frame
GPU time was effectively unchanged in the first paired sample. Dispatch count
alone is therefore not the next decode objective.

## Promotion ladder

### Gate 1: tensor and route correctness

A candidate must pass before any model timing is interpreted:

- strict F32-oracle tensor tolerance for every supported aligned and tail
  shape;
- NaN, Inf, zero-length, aliasing, and bounds cases;
- expected one-hot policy counter and exact route count;
- zero fallback, unsupported-route, host-output, and diagnostic-report counts;
- unchanged non-Metal behavior.

Any mismatch stops the experiment immediately.

### Gate 2: isolated kernel screening

Use the actual model shapes and buffer encodings. Require:

- at least five interleaved measurements after warmup;
- at least 5% median GPU-time improvement for a new kernel family;
- no tested shape regression greater than 1%;
- exact-F32 oracle parity.

This only earns a whole-model experiment. It never earns default enablement.

### Gate 3: whole-model baseline A/B

Use
[`benchmark_metal_gemma4_ab.py`](./scripts/benchmark_metal_gemma4_ab.py) with
fresh processes and balanced ordering.

Screen with six pairs and 45-second cooldown. Require:

- exact prompt and generated token digests;
- target phase at least 1% faster;
- total latency at least 0.5% faster;
- non-target phase regression no worse than 0.5%;
- at least five of six target-phase wins;
- CV no greater than 3%;
- exact expected route counts and no fallback.

CV above 3% is inconclusive, not a win or loss. Rerun once after a full cooldown.
Two independent misses park the candidate.

### Gate 4: pinned llama.cpp win

Run the canonical 2K+300 comparator in two independent sessions, six balanced
pairs per session. Promotion requires:

- median Antfly/llama.cpp total-latency ratio at or below 0.98;
- paired 95% confidence interval for total latency entirely below 1.0;
- median Antfly/llama.cpp decode-throughput ratio at or above 1.02, with its
  paired confidence interval entirely above 1.0;
- no prefill or decode latency regression greater than 1% on another workload
  in the matrix;
- exact tokens, stable routes, CV at or below 3%, and reproduction in both
  sessions.

The benchmark summarizer should compute paired bootstrap confidence intervals
before this gate is used for promotion. Until then, medians and CV are screening
evidence only.

## Prioritized implementation tranches

### T0 — Re-anchor the current binary

**Purpose:** establish the actual remaining gap before changing another kernel.

Deliverables:

1. Build `ReleaseFast` from the intended review state.
2. Run six balanced, cooled 2K+300 pairs against pinned llama.cpp b10182.
3. Record Antfly and comparator binary hashes, Git revision and tracked-diff
   hash, model hash, device registry/name, OS, Zig version, route counters, and
   prompt/output digests.
4. Repeat in a second session if the result is within 2% of parity.
5. Add paired bootstrap confidence intervals and balanced-order validation to
   the long-output summarizer.

Exit criteria: a reproducible current prefill, decode, and total gap with no
route or correctness ambiguity.

Stop condition: token mismatch, comparator drift, fallback, dirty provenance
without a recorded diff hash, or CV above 3%.

### T1 — Complete topology-preserving cost attribution

**Purpose:** select work by estimated end-to-end payoff, not kernel intuition.

The runtime already owns a bounded, per-regime workload census with exact
format, row, input/output dimension, encoding, epilogue, call count, logical
bytes, and MAC signatures. It can calibrate isolated signatures and records
whole-frame GPU time. The first tuning slice currently excludes the dominant
row-one Q4_0 decode shapes and does not price all fused attention/FFN families.

Extend that existing mechanism rather than adding ad-hoc timers:

1. Calibrate Q4_0 row-one, Q6_K row-one tail, Q4_0 pair/activation, and the
   decode GQA split signatures used by this model.
2. Export separate ranked prefill and decode tables, including uncovered GPU
   time, dropped signatures, and the smallest set explaining 90% of calibrated
   cost.
3. Compare summed calibrated estimates with whole-frame GPU duration. Treat the
   estimates as a cost model, never as claimed hardware DRAM counters.
4. Take one offline Metal System Trace/GPU capture for a steady decode frame and
   one prefill frame to validate ordering and find missing families. Do not use
   capture timings as benchmark results.
5. Split end-to-end time into whole-frame GPU, command-buffer wait, runtime
   encode/orchestration, tokenization, and residual host time without adding
   production encoder boundaries.

Exit criteria: at least 90% of eligible quant work and 80% of whole-frame GPU
time are explained, with no dropped signatures and a ranked top-three target
list for each phase.

Stop condition: if the model explains less than 80% of frame GPU time, add the
missing operation family before tuning a candidate.

### T2 — Decode GQA split schedule portfolio

**Hypothesis:** long-context decode is leaving more headroom in the two-pass GQA
split kernel than in another row-one Q4_0 MMV variant. The current route uses a
single policy: up to 32 splits, 32-token chunks, and 256 threads for both stage
and reduction. It runs across all 42 decoder layers and its cost grows with
context.

Implementation:

1. Replace the fixed policy with explicit, generic schedule descriptors keyed
   by device family, head dimension, sliding/global attention, and KV-length
   bucket.
2. Qualify separate portfolios for the 256-wide SWA-512 shape and the 512-wide
   global-attention shape. Candidate dimensions include split counts
   `8/16/24/32`, key chunks `32/64/128`, and `128/256` stage/reduce threads.
3. Keep one conservative generic fallback and per-variant one-hot counters.
4. Size scratch from the selected split count rather than the maximum policy.
5. Keep all accumulations in F32, enforce the tensor oracle, and never route an
   unqualified shape to a candidate.

Correctness matrix:

- KV lengths `511/512/513`, `1023/1024`, `2003`, `4095`, and `8191`;
- head dimensions 256 and 512;
- query lengths 1 and 2;
- SWA ring wrap and global KV history;
- tensor oracle plus exact 64- and 300-token model digests.

Performance gate: at least 2% decode improvement, 1% total improvement, four of
five wins, CV at or below 2%, and prefill regression no worse than 0.5% in the
initial route A/B. Then apply the normal six-pair and comparator gates.

Stop condition: park a schedule family after two cooled runs miss 1% total, or
immediately on any reduction/token drift.

### T3 — General Q4_0 prefill MM portfolio

**Hypothesis:** the isolated `m32-n64` pair result contains a real prefill signal,
but the useful policy should cover all high-cost Q4_0 projections and row
buckets rather than only gate/up pairing.

Implementation:

1. First rerun the existing `m32-n64` pair candidate with six balanced pairs and
   45-second cooldown. Require prefill ratio at or below 0.99, total at or below
   0.995, decode latency at or below 1.005, decode throughput at or above 0.995,
   and CV at or below 3%.
2. If confirmed, extend the generic MM portfolio across `m16/m32/m64` by
   `n64/n128` tiles, with distinct aligned and tail kernels.
3. Key selection by device, row bucket, input/output dimensions, alignment, and
   encoding for the actual E4B shapes: `2560x10240`, `10240x2560`,
   `2560x4096`, and `4096x2560`.
4. Test rows `256/511/512/1024/2003/4096/8192`; never infer a ragged-tail winner
   from an aligned shape.
5. Keep direct-KV policy independent so two experimental changes are not mixed
   in one result.

Performance gate: at least 2% prefill improvement, 0.5% total improvement, four
of five wins, CV at or below 2%, and decode regression no worse than 0.5% before
the normal promotion ladder.

Stop condition: if the cooled confirmation repeats the high variance or decode
regression, park pair prefill and continue only with independently measured
generic MM shapes.

### T4 — True Q4_0 Q/K/V projection fusion

**Hypothesis:** non-shared-KV layers can reuse the row-one input and remove one
dispatch by producing Q, K, and V from one exact-F32 superkernel. The current
Q4_0 route is effectively Q plus a packed K/V pair, not a true triple kernel.

Implementation requirements:

- generic three-weight descriptor and buffer-range contract;
- exact per-output accumulation order;
- independent Q/K/V offsets and tail bounds;
- route counter and master rollback;
- no coupling to Gemma metadata outside the graph lowering decision.

Proceed to a model A/B only if the isolated Q+K/V projection group improves by
at least 8%. Model promotion requires at least 0.5% decode and total improvement,
exact head/rope tensors, and exact 64-/300-token output.

### T5 — Long-context memory and thermal stability

**Purpose:** reduce variance and make 8K-16K qualification sustainable even if
2K latency is neutral.

Right-size and recycle GQA scratch and graph-plan buffers by active query,
head-dimension, split-count, and context buckets. After warmup there must be no
per-token allocation churn.

Exit criteria:

- at least 20% reduction in peak scratch/plan resident bytes on 8K and 16K
  prompts;
- exact output through SWA rollover and global history;
- five-run CV at or below 2%;
- no 2K prefill, decode, or total regression greater than 1%.

### T6 — Concurrency correctness, then portable rollout

Concurrency is a correctness project before it is a performance project.

1. Reduce the digest divergence to a minimal planned-range test.
2. Build one fail-closed resource-access DAG for complete read/write spans.
   Unknown or alias-ambiguous accesses serialize.
3. Require sequential/concurrent parity for at least 20 repeated runs across
   multiple prompts, 64- and 300-token outputs, SWA wrap, and long global KV.
4. Only then measure whether concurrency clears the normal whole-model gate.

For every promoted optimization, replace development overrides with a
device/shape-qualified runtime policy. Keep a master rollback switch, emit the
chosen route in timing JSON, and add model-neutral tensor coverage plus Gemma 4
route coverage. A hardware nightly should run the screening workload matrix;
ordinary CI should enforce contracts and correctness without pretending to
measure Metal performance.

## Definition of done

This roadmap is complete when the baseline, no-MTP route:

- beats the pinned llama.cpp build on the canonical 2K+300 workload under the
  final statistical gate;
- preserves exact prompt and generated token IDs;
- has no material prefill or decode regression across the workload matrix;
- reproduces across independent cooled sessions;
- has device/shape-qualified, default-on policy with a tested rollback;
- remains fully Metal-resident with no fallback or diagnostic reports;
- passes the Metal tensor, route, benchmark-contract, API-validation, and
  non-Metal regression suites.

Only after those conditions hold should MTP results be layered on top and
reported as an additional speedup.

## Commands

Build and contract tests:

```sh
cd zig/pkg/inference
zig build -Doptimize=ReleaseFast
zig build test-metal-gemma4-benchmark-contracts
zig build test-metal-q4-0-linear
```

Pinned comparator re-anchor:

```sh
cd zig/pkg/inference
RUNS=6 \
COOLDOWN_SECONDS=45 \
OUTPUT_TOKENS=300 \
MAX_TOTAL_RATIO=0.98 \
MIN_DECODE_RATIO=1.02 \
MAX_CV=0.03 \
EXPECTED_TOKEN_IDS_SHA256=<approved-token-digest> \
OUT_DIR=/tmp/antfly-gemma4-current-b10182 \
scripts/benchmark_metal_gemma4_long_output.sh
```

Candidate A/Bs use the checked contract runner. It deliberately requires
approved prompt and generated-token digests and explicit route profiles:

```sh
cd zig/pkg/inference
python3 scripts/benchmark_metal_gemma4_ab.py run \
  --out-dir /tmp/antfly-gemma4-candidate \
  --experiment-id <candidate-id> \
  --target-phase <prefill-or-decode> \
  --model "$HOME/.antfly/inference/models/google/gemma-4-E4B-it-qat-q4_0-gguf" \
  --antfly-bin zig-out/bin/antfly-inference \
  --expected-prompt-tokens 2003 \
  --expected-prompt-token-ids-sha256 <approved-prompt-digest> \
  --expected-token-ids-sha256 <approved-token-digest> \
  --runs 6 \
  --cooldown-seconds 45 \
  --min-target-wins 5 \
  --max-cv 0.03 \
  --baseline-route-profile split_ffn \
  --candidate-route-profile <qualified-profile> \
  --candidate-env <candidate-enable-variable>=1
```

Run `python3 scripts/benchmark_metal_gemma4_ab.py run --help` for candidate
policy, route, determinism, and stage-diagnostic options. Never reuse an output
directory across experiments.

## Source map

- [Long-output comparator harness](./scripts/benchmark_metal_gemma4_long_output.sh)
- [Comparator parser and statistical contract](./scripts/gemma4_metal_long_output.py)
- [Baseline/candidate A/B runner](./scripts/benchmark_metal_gemma4_ab.py)
- [Metal Q4_0 route benchmark](./src/bench/metal_q4_0_linear.zig)
- [Metal kernels and scheduling policy](./src/backends/metal_kernels.m)
- [Metal runtime, workload census, and calibration](./src/backends/metal_runtime.zig)
- [Decoder gated runtime](./src/backends/decoder_gated_runtime.zig)
- [Compiled Metal graph executor and route counters](./src/graph/metal_executor.zig)
- [Generation timing and JSON provenance](./src/native_generate.zig)
