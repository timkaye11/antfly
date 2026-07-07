# Quant Kernel Compiler

## Scope

The quant kernel compiler is a build-time Antfly inference tool for GGUF quant
matmul kernels. It is deliberately small: descriptors, tiny IR, backend-specific
lowering metadata, generated source files, route manifests, and evidence gates.

It is not a runtime JIT and not a general GPU compiler. Production runtime must
not depend on Python, Triton, CUDA toolkit, cuBLAS, XLA, or network access.

## Source Of Truth

The dispatch-facing API is still the existing graph planning contract:

- `src/graph/quant_matmul.zig`: format, row bucket, dispatch bucket, and generic
  quant matmul planning vocabulary.
- `src/graph/quant_kernel_compiler.zig`: compiler descriptors, IR, generated
  artifacts, route registry, benchmark manifests, and promotion evidence policy.
- `src/backends/metal_kernels.m`: native Metal runtime and provider lowering.

Generated artifacts are checked in:

- `src/ops/metal/generated/*.metal`: generated Metal candidates.
- `src/ops/metal/artifacts/*.metal`: checked-in artifact copies for candidates
  that have a runtime artifact path.
- `src/ops/cuda/generated/quant_kernel_*.cu`: generated CUDA candidates.
- `src/ops/cuda/artifacts/quant_kernel_*.cu`: checked-in promoted copies of
  generated CUDA kernels; the same kernels are embedded in the production
  `inference_cuda_kernels.{cu,ptx,fatbin,_sm89.cubin}` bundle via
  `scripts/regen-cuda-artifacts.sh`.
- `src/ops/cuda/generated/evidence/*.json`: checked-in CUDA promotion
  benchmark evidence.
- `src/ops/cuda/generated/quant_kernel_*.json`: spec, artifact, benchmark, and
  conformance manifests. The path is historical; the manifests cover both CUDA
  and Metal.

`zig build quant-kernel-codegen -- --check` verifies the primary source path,
generated source sidecar path, and Metal artifact source path for every
manifested artifact. Missing sidecars or artifact copies fail the codegen check
instead of becoming a packaging surprise during review.

## Compile Flow

The internal compile API is intentionally small. Callers build or look up a
`QuantKernelCompileRequest`, then use `compileQuantKernelSource(...)` or the
Metal convenience wrapper `compileMetalKernelSource(format, row_bucket,
epilogue)`. The result ties together the descriptor, IR, route lowering,
generated artifact metadata, canonical source text, check command, artifact
path, and runtime gate.

`compileQuantKernelSource(...)` is also a guardrail. Before returning a compiled
source record, it verifies that the descriptor, IR, route lowering, generated
artifact, canonical source text, generated/artifact paths, check command,
production bit, and Metal runtime gate all describe the same route. Drift returns
`null` and fails the codegen path instead of emitting an orphan kernel.

Typical Metal use is:

```zig
const compiled = quant_kernel_compiler.compileMetalKernelSource(.q6_k, .rows_2_8, .bias_gelu) orelse
    return error.UnsupportedQuantKernelRoute;
const emitted = try quant_kernel_compiler.emitCompiledSource(allocator, compiled);
defer emitted.deinit(allocator);

// emitted.data is the canonical generated source text.
// compiled.source_path, artifact_source_path, check_command, runtime_gate_env,
// and production_enabled describe how that source is checked and routed.
```

`emitCompiledSource(...)` is the source emission boundary. Migrated Metal
families emit source from descriptor data and return owned source text; families
not yet migrated return the checked-in canonical template. Both paths are
intentional: generated files stay checked in, production dispatch stays
artifact-backed, and source changes remain visible in review.

The codegen executable is always built for the build host, not the selected
runtime target, so source checks still work during Linux/CUDA cross builds.

Current descriptor-emitted Metal families:

- `Q2_K` small batch: `none`, `bias`, `bias_gelu`.
- `Q3_K` small batch: `none`, `bias`, `bias_gelu`.
- `Q4_0` small batch: `none`.
- `Q4_1` small batch: `none`.
- `Q4_K` small batch: `none`, `bias`, `bias_gelu`.
- `Q5_0` small batch: `none`.
- `Q5_1` small batch: `none`.
- `Q5_K` small batch: `none`, `bias`, `bias_gelu`.
- `Q6_K` small batch: `none`, `bias`, `bias_gelu`.
- `Q8_0` small batch: `none`, `bias`, `bias_gelu`, `relu`.
- `Q8_1` small batch: `none`.
- `Q8_K` small batch: `none`.

The focused compiler golden test proves emitted source is byte-for-byte equal
to the checked-in canonical source for these families. It also keeps at least
one non-Metal borrowed-source case so the fallback path remains covered.

1. Add or change a `QuantKernelSpec` in
   `src/graph/quant_kernel_compiler.zig`.
2. Add or update `GeneratedArtifact` route metadata.
3. For a migrated family, update `emitCompiledSource(...)` and the small
   backend-specific emitter. For an unmigrated family, add the canonical
   checked-in source template.
4. From `zig/pkg/inference`, run:

   ```sh
   zig build quant-kernel-codegen -- --write
   ```

5. Review the generated Metal/CUDA manifest diffs.
6. Run the focused compiler tests:

   ```sh
   zig build test -Dmetal=true -Dcuda=false -- --test-filter "quant kernel compiler"
   ```

7. Run the Metal gates before treating any Metal route as locally ready:

   ```sh
   zig build quant-kernel-metal-production-regression-check -Dmetal=true -Dcuda=false
   zig build quant-kernel-metal-local-check -Dmetal=true -Dcuda=false
   zig build quant-kernel-metal-industry-local-check -Dmetal=true -Dcuda=false
   ```

   These Metal evidence targets require a macOS target with `xcrun`/Metal
   available. On non-macOS targets they fail closed instead of passing as
   source-only checks.

   The generic `quant-kernel-local-check` stays usable on non-macOS for
   generated-source freshness and CUDA source-policy checks. On macOS it also
   depends on the Metal local gate.

## Promotion Policy

Generated kernels start as dev candidates. A candidate may become production
only when all of these are true:

- Generated source and checked-in artifact source match the manifest
  fingerprint.
- Runtime route evidence proves the generated route is actually selected.
- Provider route evidence exists when the provider surface exposes the same
  candidate.
- Promotion evidence is sequential, uses `--repeat-runs 5 --measure-iters 500`,
  and names the exact kernel in the evidence path.
- Repeated promotion and production-regression checks run two unrecorded warmup
  repeats before the measured repeats; evidence records this as
  `warmup_repeat_runs`, and generated manifests expose
  `metal_promotion_warmup_repeat_runs`, so stale evidence is rejected when the
  policy changes.
- Promotion evidence times the generated decode-runtime route in an active-frame
  batch when that route is selected, while still running the standalone
  generated source check for compile/correctness coverage.
- Every promoted case clears the minimum speedup gate and the worst-repeat
  speedup gate.
- `quant-kernel-metal-production-regression-check` passes after promotion.

Cleared blocker evidence is not enough by itself. It is only a signal to
investigate. Production promotion requires checked-in evidence plus the
production-regression gate.

CUDA promotion uses the same shape with CUDA-specific mechanics: the promoted
source copy must live outside `src/ops/cuda/generated/`, the benchmark command
is sequential `bench-cuda` with `--warmup-iters 5 --measure-iters 50
--quant-compiler-repeat-runs 3` and an `--quant-compiler-evidence-out` path, the
measured geomean speedup must clear `minimum_speedup` (1.0) against the named
handwritten baseline with CPU-reference correctness inside tolerance, and a
matching comptime `BenchmarkEvidence` record plus the checked-in evidence JSON
are required before `production_enabled` may flip. Promoted CUDA kernels are
embedded in the production artifact bundle, so runtime dispatch stays
driver-only (`cuModuleLoadDataEx`; nvcc is dev-time only).

## Current CUDA State

The current checked-in state has 3 promoted generated CUDA production routes,
all Q4_0, promoted on sequential benchmark evidence measured on an NVIDIA L4
(driver 580.159.03, CUDA 13.2 nvcc; evidence JSONs under
`src/ops/cuda/generated/evidence/`):

| kernel | plan | handwritten baseline | measured speedup (geomean) |
|---|---|---|---|
| `antfly_q4_0_mmv_f32_v1` | `cuda/q4_0/rows_1/none/mmv` | `termite_linear_q4_0_f32_tile4` | 1.18 (worst shape 1.02, LM head 1.30) |
| `antfly_q4_0_mm_f32_v1` | `cuda/q4_0/rows_9_64/none/mm` | `termite_linear_q4_0_f32` | 3.53 (up to 10.23 on FFN-down; 0.75 at in_dim=256, runtime-gated) |
| `antfly_q4_0_pair_mmv_f32_v1` | `cuda/q4_0/rows_1/pair/mmv` | `termite_linear_q4_0_pair_nobias_f32_tile4_w4` | 1.29 |

Speedups are geomeans across the Gemma4 E2B QAT dispatch shapes recorded in the
bench targets (`bench-cuda --quant-compiler-q4-0-{mmv,mm,pair}-ptx`), with
CPU-reference and baseline-output max-abs-diff at or below 3e-6. The mm
evidence also records a losing shape: `in_dim=256` (E2B PLE projection) runs at
about 0.73x the handwritten baseline, so the runtime route only claims rows
9..64 shapes with `in_dim >= 512` (`cuda_q4_0_generated_mm_min_in_dim` in
`cuda_compute.zig`); narrower shapes stay on the handwritten kernel. The mmv
bench additionally cross-checks the Gemma4 LM-head shape (`1536 -> 262144`)
generated-vs-baseline on device (the CPU reference is skipped above
`quant_compiler_q4_0_max_reference_out_dim` because it dequantizes the full
weight tensor to dense f32).

End-to-end impact, Gemma4 E2B QAT (`--backend cuda`, 128 tokens, generated
kernels default-on vs disabled, interleaved runs, bit-identical output):

- prefill: 146-151 ms vs 200-201 ms (about -25%).
- decode: 61.1/60.8 tok/s vs 59.6/59.5 tok/s (about +2.4%).
- end-to-end total: 2241/2257 ms vs 2349/2353 ms (about -4.5%).
- per-run generated launches: 16256 mmv + 92 mm + 4445 pair, zero fallbacks.

Q4_K-only models are unaffected by construction (the promoted routes are keyed
to Q4_0): clipclap embeddings are bit-identical with the kernels on/off (all
quant matmuls ride the q4_k tensor-core route), and gliner2 passes the
`scripts/verify_gliner2_cuda.sh` native/CUDA parity gate with identical entity
counts and warm extraction timing unchanged (12.0 vs 12.1 ms).

Runtime dispatch for the promoted CUDA routes is default-on and per-kernel
opt-out via `ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_MMV`,
`ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_MM`, and
`ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_PAIR` (recorded per artifact as
`runtime_gate_env` in `quant_kernel_artifacts.json`). Actual launch counts are
reported by `generate --print-timing` as `cuda_q4_0_generated_counts:` with
per-kernel hit/fallback counters; a fallback dispatches the handwritten
baseline for that call.

The Q4_0 `pair` epilogue (two no-bias projections sharing one input) is
CUDA-only; Metal is carved out in `supportsEpilogueForBackend` and keeps its
own Q4_0 small-batch route unchanged.

Dispatch precedence note: in the rows==1 Q4_0 pair route the promoted
generated kernel runs ahead of the default-on handwritten
`ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_TILE4_W4` path (that env defaults to true, so
it is the production default rather than an opt-in experiment). To select the
handwritten pair kernel — for A/B timing or to rule out the generated kernel —
set `ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_PAIR=1`. The plain
`linearNoBias` route keeps the opposite order: the genuinely opt-in tile8 and
tile4_w4 experiment envs (default off) still take precedence over the
generated mmv kernel when explicitly enabled.

Local reproduction note: the manifest-pinned `nvcc -ptx -arch=compute_75`
check commands emit PTX ISA matched to the pinned CUDA 13.2 toolkit. Drivers
older than the toolkit (for example 580.x = CUDA 13.0) cannot JIT that PTX;
compile `-cubin -arch=sm_89` to the pinned `/tmp/*.ptx` paths instead when
running the benchmark/evidence commands locally. The production runtime is
unaffected: it loads the checked-in fatbin/cubin artifacts.

The remaining non-promoted CUDA candidate is the first lazy Q4_K
`rows_2_8`/`bias_gelu` kernel, which loses to its handwritten baseline (0.74x)
and stays dev-only with blocker evidence, as designed.

## Current Metal State

The current checked-in state has 8 promoted generated Metal production routes:
`Q2_K none`, `Q3_K none`, `Q4_K bias`, `Q5_K bias`, `Q6_K none`,
`Q6_K bias`, `Q8_0 none`, and `Q8_0 bias`.
The production-regression gate covers 16 generated-vs-handwritten benchmark
cases for those routes.

Runtime-wired generated candidates that are not promoted stay on handwritten
production dispatch by default. They remain opt-in through
`TERMITE_METAL_ENABLE_ANTFLY_*` gates, or all at once with
`ANTFLY_METAL_GENERATED_QUANT=1` / `TERMITE_METAL_ENABLE_ANTFLY_GENERATED_QUANT=1`,
and are tracked with explicit blockers.
The route-all evidence covers 50 generated cases: all 50 must be route-ready,
46 must have provider-route evidence, and every non-promoted candidate must
explain itself with an explicit blocker such as `speedup_gate_missing`,
`runtime_route_only`, or `unsupported_handwritten_baseline`. Benchmark-ready
counts for non-promoted candidates may vary with timing noise; production
promotion does not rely on route-all speedup alone.

The dedicated blocker evidence table remains intentionally stricter than
route-all: 17 candidate kernels are guarded, 12 have benchmark-evidence paths
(`speedup_gate_missing` for Q4_0, Q5_0, Q5_1, Q4_K none, Q6_K bias+GELU, and
Q8_0 bias+GELU; `unstable_benchmark_timing` for Q4_1, Q4_K bias+GELU,
Q5_K none, Q5_K bias+GELU, Q8_1 none, and Q8_K none), and 5 are
route-evidence-only
because their handwritten baseline is unsupported. Cleared one-off evidence for
these kernels is production-regression guarded and does not promote the kernel
by itself.

This is intentional. A slow or noisy candidate is useful for compiler coverage,
route testing, and future tuning, but it must not silently become the production
route.

Current Q4_0, Q4_1, Q5_0, Q5_1, Q4_K none/bias+GELU, Q5_K none/bias+GELU,
Q6_K bias+GELU, Q8_0 bias+GELU, Q8_1 none, and Q8_K none note: these
generated kernels are correct and useful for route coverage, but they are not
production defaults. Their blocker evidence is production-regression guarded; a
one-off local promotion-ready result is a signal to investigate, not enough to
promote.

Current Q2_K/Q3_K bias and bias+GELU plus Q8_0 relu note: these generated
kernels prove correctness and dispatch wiring through route-all, but they do not
have comparable handwritten baselines. They cannot be promoted by the sequential
speedup gate until a non-speedup promotion policy or comparable baseline exists.

Current Q4_K/Q5_K/Q6_K/Q8_0 compiler note: their small-batch Metal source is now
descriptor-emitted where listed above, but promotion remains per-kernel and
evidence-gated. Descriptor emission is source generation, not production
selection.

Current local validation snapshot:

- `zig build test -Dmetal=true -Dcuda=false -- --test-filter "quant kernel compiler"`
  covers descriptor determinism, IR generation, route summaries, manifests,
  generated source freshness, doc guardrails, and CPU/reference checks.
- `zig build quant-kernel-codegen -- --check-metal` compiles the generated and
  promoted Metal sources through `xcrun`.
- `zig build quant-kernel-metal-production-regression-check -Dmetal=true -Dcuda=false`
  runs the 8 promoted kernels across 16 generated-vs-handwritten cases.
  `src/ops/cuda/generated/quant_kernel_benchmarks.json` enumerates those 16
  Metal production-regression cases with shape, dims, tolerance, source
  fingerprint, and benchmark command metadata. The target fails on hard route
  blockers such as missing generated/provider routes, unsupported production
  baselines, or average speedup below the promotion threshold. Timing drift from
  an individual repeated run is reported as `production_regression_timing_drift`
  with slow-fallback counters, but does not hide the route/provider evidence.
- `zig build quant-kernel-metal-runtime-route-all -Dmetal=true -Dcuda=false`
  checks all generated Metal routes. The latest local run was route-ready for
  50/50 cases, provider-checked 46 cases, and reported explicit non-promoted
  blockers instead of silently promoting candidates.
- `zig build quant-kernel-metal-industry-local-check -Dmetal=true -Dcuda=false`
  adds blocker evidence refresh/strict checks and the Gemma4 generated-route
  prefill-frame smoke. The generated-route smoke now gates model-level
  generated-family coverage: the local Gemma4 E4B path must report at least two
  generated quant families, with `q4_0` as the top family and Q8_0/Q4_0
  generated dispatch counters present.

Metal-named evidence targets are macOS-only by contract. They report a clear
build failure off macOS so a Linux/CUDA VM can still run CUDA/source checks
without producing false Metal runtime evidence. Use `quant-kernel-local-check`
for the cross-platform source/policy gate and `quant-kernel-metal-local-check`
when Metal runtime evidence is required.

## Evidence Commands

Metal runtime evidence uses the `antfly.quant_kernel_metal_runtime_evidence.v9`
schema. In addition to per-case timings, each evidence file records the
machine-checked compiled-vs-handwritten summary:
`benchmark_supported_count`, `benchmark_speedup_pass_count`,
`benchmark_speedup_min`, `benchmark_speedup_min_case`, `benchmark_speedup_max`,
`benchmark_speedup_max_case`, and `benchmark_speedup_avg`. The runtime evidence
checker recomputes these fields from the case list and rejects stale summaries.
Repeated timing evidence also records `repeat_gate_index`, allowing the
promotion gate to tolerate one isolated outlier while still requiring the sorted
repeat speedup gate to clear.

Route-all evidence is an observability check, not a promotion decision. Promoted
generated-production routes report an empty `promotion_blocker` when route
evidence and benchmark evidence are healthy. Non-promoted generated candidates
continue to report explicit blockers such as `runtime_route_only`,
`speedup_gate_missing`, or `unsupported_handwritten_baseline`.
Unsupported-handwritten-baseline candidates are route-evidence-only. They may
prove correctness and dispatch wiring through route-all. They cannot be promoted
by the sequential speedup gate until a comparable handwritten baseline or
explicit non-speedup promotion policy is added.

Production-regression evidence must match the compiler-owned
`metal_production_regression_cases` manifest exactly. Each production-regression
evidence file also records the compiler benchmark manifest schema, case count,
and case fingerprint, and the evidence checker rejects stale or partial case
sets before trusting benchmark speedups. Successful production-regression checks
also print the identity in the summary line as:
`benchmark_manifest=antfly.quant_kernel_benchmarks.v4:34:<fingerprint>`.

## Runtime Observability

Per-run stats expose `quant_kernel_plan` in Metal compact JSON, plain timing
logs, and Prometheus metrics. The important counters are:

- `planned`: quant matmul ops observed by the shared plan.
- `generated_production`: generated kernels selected as production routes.
- `generated_candidates`: generated candidates that were planned or, for
  runtime-derived Metal counters, actually dispatched while still marked as
  candidates.
- `fast_path_misses`: the sum of explicit, mutually exclusive fallback reasons.
  This includes
  generated candidates that fell back to handwritten production because the
  generated artifact is missing or not wired, plus unsupported format, shape,
  epilogue, backend, and tensor-core-repack reasons. The `unsupported` counter is
  still exposed separately as the roll-up for unsupported format/shape/epilogue/
  backend routes.
- `top_fallback_reason` / `top_fallback_count`: the largest concrete fallback
  family for the run.

Actual generated Metal dispatch counters are not treated as handwritten
fallbacks. If a dev candidate is opt-in enabled and executes, it increments
`generated_candidates` without adding a fast-path miss. If the same planned
candidate does not execute and falls back to handwritten production, its
fallback reason contributes to `fast_path_misses`.

Generate promotion evidence for one candidate:

```sh
zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- \
  --evidence-out /private/tmp/antfly-quant-metal-<kernel_id>-promotion-evidence.json \
  --repeat-runs 5 \
  --measure-iters 500 \
  --promotion-ready-kernel <kernel_id>
```

Check promotion evidence:

```sh
zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- \
  --check-evidence /private/tmp/antfly-quant-metal-<kernel_id>-promotion-evidence.json \
  --require-promotion-ready \
  --require-kernel <kernel_id>
```

Refresh and audit blocker evidence:

```sh
zig build quant-kernel-metal-blocker-evidence-refresh -Dmetal=true -Dcuda=false
zig build quant-kernel-metal-blocker-strict-check -Dmetal=true -Dcuda=false
```

## Adding A Format Or Epilogue

The minimal path is:

1. Add the descriptor data: block values, block bytes, fields, decode ops,
   schedules, epilogues, dtype policy, and backend support.
2. Add generated source/artifact metadata for one schedule.
3. Add descriptor-driven emission only when it removes real duplicated source
   work for that family. Keep the emitter backend-specific and byte-for-byte
   golden-tested against the checked-in source.
4. Add route/conformance expectations.
5. Regenerate manifests.
6. Prove CPU/reference correctness and route coverage.
7. Leave the route opt-in until promotion evidence and production regression
   both pass.

Do not add a general compiler abstraction unless it removes real duplicated
format/schedule work in this file and the backend lowering.
