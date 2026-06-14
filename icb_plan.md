# Beat the Metal encode wall: slot-bound buffers -> compiled-replay -> ICB

## Context

Branch `gliner2_finetuning_parity`. GLiNER2 DeBERTa LoRA training on Metal. After the committed wins
(fused attention, sumsq fix) the step is **host-encode-bound**: each step re-encodes ~4500 Metal
compute dispatches on the host (~130–180ms), while GPU compute is only ~4.5ms. The graph is **static
across steps** (same node ids/shapes/op order; only tensor DATA changes). Measurement this session
proved op-count fusion of cheap ops (LayerNorm) is wall-time-NEUTRAL — the encode cost isn't uniform,
and we're encode-bound not compute-bound.

**Ecosystem research (PyTorch/vLLM/llama.cpp-Metal/MLX/MPSGraph):** the proven lever for an
encode-bound static graph is **capture-once / replay-many** - CUDA Graphs (vLLM, the core lever;
`enforce_eager` off is much slower) and its Metal analog, **MTLIndirectCommandBuffer (ICB)**. The hard
prerequisite both share: **static buffer addresses** — preallocated buffers reused across steps with
data written in place (vLLM/CUDA-graph practice). Our executor currently blocks this: it's
**handle-flow** (each op allocates a fresh output buffer per call; downstream consume by CT handle),
so buffers aren't stable. The `buffer_plan` ALREADY computes a static liveness-based allocation map
(`allocations[]`, reuse via `lifetimesOverlap`) — we just don't bind outputs to it on the hot path.

**Goal:** encode the static command stream once into an ICB, replay each step with data written in
place - collapsing the ~130-180ms encode to ~the GPU submit/wait. Done in 3 flag-gated,
parity-gated phases; each lands independently and is validated by the in-Zig graph-vs-direct parity
gate (rel ~1e-7) + identical loss trajectory before the next.

## Current-code corrections to preserve

- `buffer_plan.zig` already has the right logical model: one `LogicalSlot` per node; `.allocation`
  slots own storage; `.view` slots alias their source allocation; `.runtime_input` and `.constant`
  slots intentionally have `invalid_allocation`. `PhysicalAllocation.reusable=false` already implies
  `slot_count==1` in validation, and graph outputs are non-reusable because `computeLastUse` extends
  them to `node_count`.
- Do not blindly map buffer-plan allocations into the existing fixed `graph_plan_buffers` array. The
  current C runtime capacity is `TERMITE_METAL_GRAPH_PLAN_SLOT_CAPACITY == 29`, and those indices are
  already semantically assigned to decoder/runtime scratch slots. Phase 0 needs either:
  - a new allocation-id-keyed output pool owned by `MetalPartitionExecutor`, with ObjC helpers to
    allocate/retain/bind MTLBuffers; or
  - an explicit expansion/renumbering of the runtime graph-plan slot table, with a hard capacity
    assert and migration of all existing slot constants.
  The safer first implementation is the executor-owned pool. Reuse the allocation policy and
  `MetalTensor.deviceBorrowed`/`retainedStorageView` ownership model, not the fixed graph-plan slot
  numbers.
- A pooled output CT must be a retained non-owning device view over the pool buffer. In this codebase
  the free no-op property is not a generic CT tag; it comes from `MetalTensor.deviceBorrowed`
  (`release_on_drop=false`) and the CT handle owning only that retained wrapper. Still guard
  `freeExpiredInputs` and alias-clone paths so they do not drop or clone pooled aliases incorrectly.
- Graph-output slots are special even after slot binding. The executor currently deep-copies graph
  outputs in `copyPartitionGraphOutputsToOwnedStorage` after `decoderRuntimeSubmitAndWaitFrame`, then
  syncs host mirrors. Keep that behavior unless Phase 0 gives graph outputs dedicated non-reused pool
  buffers with the same post-frame ordering guarantee.
- ICB resource residency should be an availability-gated optimization. On the current SDK,
  `MTLResidencySet` is available on newer Metal headers, but the baseline ICB path can still use
  per-frame `useResource:usage:` calls on the compute encoder. Implement that fallback first and add
  residency sets only behind an OS/API check.

## Phase 0 - Slot-bound persistent output buffers (the prerequisite; ~neutral wall-time)

Bind every node's OUTPUT to a persistent pooled `MTLBuffer` keyed by `buffer_plan` `AllocationId`,
reused across steps (same address every step), instead of allocating fresh per op.

- **Pool:** add `OutputBufferPool` to the persistent `MetalPartitionExecutor` struct
  (metal_partition_executor.zig ~580, beside the cached `partition_view`/`runtime_region_plan`).
  One `MTLBuffer` per non-view `PhysicalAllocation` (`kind==.tensor`, `backend==.metal`), sized `byte_size`, allocated
  once / refreshed on the existing partition fingerprint (`CachedPartitionBufferView.matches`).
  Prefer a new executor-owned allocation-id pool over overloading the current 29-entry
  `graph_plan_buffers` table. Exclusions are driven by data already in `buffer_plan`: runtime
  inputs/params (flow via `rt_map` and weight stores), constants (pre-materialized), transfer-only
  allocations (`kind==.transfer`), and **views** (reshape/slice - no own buffer, alias source
  allocation). Graph outputs get a pooled buffer only if it is non-reused and survives post-frame
  readback; otherwise preserve the existing owned-copy path.
- **Bind outputs:** thread an optional `output_hint: ?CT` or low-level `?MetalTensor` retained view
  over the pooled slot buffer, shaped to `slot.desc.shape`, from the node loop (~1162) through
  `tryExecuteMetalCommand` -> the op switch -> the op. The hint should be created from the
  allocation-id pool using `MetalTensor.deviceBorrowed` / `retainedStorageView`, then wrapped as a CT
  with normal CT lifetime for the wrapper only. Ops with an existing `*Into` device variant (several
  exist: `decoderRuntimeApplyMultiplyInto`, `copyTensorInto`, ...) write directly and return that
  non-owning CT view; add `*Into` variants for the hot ops missing them (dot_general, transpose,
  reductions) in metal_compute.zig/metal_runtime.zig/metal_kernels.m. Reshape/slice stay views (no
  hint). Any op without an `*Into` path falls back to allocate-and-return (perf no-op, not a
  correctness risk) — keeps the phase incremental.
- **Heuristics:** add an explicit pooled-output predicate (for example by allocation id / CT storage
  handle) and use it to guard `freeExpiredInputs` (~1274) and
  `cloneOutputIfAliasedInputWouldBeFreed` (~1261). `cb.free` may still destroy the CT wrapper, so
  correctness must not depend on a generic free no-op; the underlying MTLBuffer must remain owned by
  the pool until the executor cache invalidates.
- **Riskiest part — in-place input/output aliasing:** the plan may legally place a node's output in
  the same allocation as an input at its last use; a kernel that reads the input while writing the
  output can corrupt mid-dispatch. Mitigation: refuse to bind an `output_hint` whose allocation
  equals ANY input allocation that is still live at that node, or whose source view aliases such an
  allocation. Fall back to allocate-and-return; debug-assert the decision. Also assert every
  graph-output slot bound to the pool is `!reusable && slot_count==1`.
- **Flag:** `TERMITE_METAL_SLOT_BOUND_OUTPUTS` (default-off; off = byte-for-byte today's path).
  Land in slices (dot_general → +elementwise/transpose → views → fused ops → flip heuristics), each
  parity-validated. Add stats: pooled-hit, allocate-fallback, alias-refused, unsupported-op, and
  graph-output-owned-copy counts.

## Phase 1 - CompiledStepPlan: record once, branch-light replay (reclaims ~40ms/step)

> **⚠️ CORRECTION (2026-06-13, measured): Phase 1's standalone premise is REFUTED. CUT it.**
> Loop-profile at b1 s128 (warm step): `execution_ms=143`, of which `command_path_ms=134.4`
> (4524 dispatches, ~29.7µs/op) is the entire wall. The decision overhead this phase was meant to
> skip — `alias_clone_ms=0.099` + `free_expired_ms=0.619` + `stats_ms=0.093` ≈ **0.8ms**. The
> "~40ms" estimate below is wrong for this graph; record/replay-to-skip-decisions reclaims <1ms.
> Phase-0 flag OFF→ON moved command_path only 140.4→134.4ms (~2.7µs/op = the saved `deviceAllocate`);
> the remaining ~27µs/op is the irreducible per-op Metal encode/dispatch + Zig wrapping. The ONLY
> lever that collapses it is **Phase 2 (ICB)**. The dispatch-descriptor RECORD pass below survives
> ONLY as Phase 2's build substrate — fold it into Phase 2, do not ship eager replay as its own phase.

Separate planning from emission. Record per-op **dispatch descriptors** (a fused op contributes
several; interpreter ops contribute zero):
`DispatchDesc { pso_id, inputs: []BufHandle(allocId|rt-node|param|constant, offset, arg_index),
output: BufHandle, grid, tg, tg_mem_len, params: []u8, barriers, kind: icb_ok|eager_only }`.

- Add `compiled_step: ?CompiledStepPlan` to the persistent executor, invalidated on the same
  fingerprint as `partition_view`/`runtime_region_plan`. First step runs today's ladder in **record
  mode** (a cheap capture branch in the per-op encode helpers appends a `DispatchDesc`); later steps
  **replay** a tight loop over `dispatches`, skipping the 4-way ladder, defer heuristics, alias-clone,
  and free-expired — this alone reclaims the ~40ms/step decision overhead measured this session.
- `kind`: a node is `icb_ok` iff it executed via planned-region/fused/command (`execution_kind ==
  .command`) AND all bindings are static pooled/rt buffers; else `eager_only` (the ~19
  interpreter-fallback ops). Phase 1 replay still encodes eagerly (validates the record before ICB).
- Record invalidation must include more than node count: include partition index/count, first/last
  slot node, slot count, transfer count, backend, output pool allocation ids/sizes, op kind, shape,
  PSO/function id, dispatch geometry, threadgroup memory length, static params bytes, and every
  static buffer binding's allocation id + byte offset. Reject replay if any command's actual runtime
  binding differs from the recorded one.
- Keep existing graph-output elision protection in the replay path. It is not just overhead; it is
  the correctness guard that materializes the scalar loss tail and gradient leaves.

## Phase 2 - MTLIndirectCommandBuffer encode-once-replay (collapses the ~130ms)

> **⚠️⚠️ REFUTED BY MEASUREMENT (2026-06-13). DO NOT BUILD. The premise of this whole plan is wrong.**
> A host-encode micro-benchmark (3 stable runs, N=2000 multiply dispatches) found ICB replay is
> **~4× SLOWER** than eager re-encode: eager = 176–191 ns/op, ICB (executeCommandsInBuffer +
> memoryBarrier per op) = 728–827 ns/op (win_ratio 0.22–0.26×, per-op saving ≈ −550 ns). The deep
> dependency chain forces a barrier between nearly every op (the naive single-range version RACES —
> proven), so the ICB can't batch and pays executeCommandsInBuffer+barrier per op, which exceeds the
> cost of just encoding the dispatch.
> **Root cause of the misdiagnosis:** the loop-profile `command_path` ≈ **29.6µs/op**, but the pure
> Metal encode is only **~0.18µs/op** — a 168× gap. The ~130ms wall is **~99% Zig-side per-op
> overhead** (CT wrapping, `retainedCopy` ObjC-retain atomics, `ownedDeviceMetalTensorFromCt`, shape
> conversions, hazard-tracker range scan), NOT the Metal encode. ICB targets the wrong 0.18µs.
> **The real lever (new investigation, NOT in this plan): profile + cut the ~29µs/op Zig-side overhead.**
> Prime suspects: the `prepare_planned_compute_accesses` hazard scan (possible O(active_ranges) →
> quadratic per frame), per-operand retain/release atomics, per-op CT alloc + shape i32 conversions.
> A resolution-caching record/replay (skip re-resolution + hazard re-scan on replay steps) is the
> plausible win — but profile first to localize the 29µs. Phase 0 (the ~50% output pool, ~6ms real
> win + stable addresses) stands; Phases 1 & 2 are both resolved (cut / refuted) by measurement.

- **Build once** (keyed by fingerprint), new ObjC in metal_kernels.m + Zig wrappers in
  metal_runtime.zig: `MTLIndirectCommandBufferDescriptor` (`commandTypes =
  ConcurrentDispatchThreads`/`ConcurrentDispatch`, `maxKernelBufferBindCount` = max arg count,
  `inheritPipelineState=NO`, `inheritBuffers=NO`); `newIndirectCommandBufferWithDescriptor:
  maxCommandCount:options:` (N = count of `icb_ok` dispatches). Encode each
  `indirectComputeCommandAtIndex:k`
  once: `setComputePipelineState:` (PSOs already cached), `setKernelBuffer:offset:atIndex:` per
  binding, `setThreadgroupMemoryLength:atIndex:`, `concurrentDispatchThreadgroups:threadsPerThreadgroup:`
  or `concurrentDispatchThreads:threadsPerThreadgroup:` as recorded. **Params:** ICBs can't
  `setBytes`; convert each op's static params to a small per-op param `MTLBuffer` (allocated once at
  record; params are static across steps) bound via `setKernelBuffer`.
- **Residency/resources:** build the dedup'd buffer set during record. Baseline: call
  `useResource:usage:` for every referenced `MTLBuffer` on the compute encoder before
  `executeCommandsInBuffer:withRange:`. Optional: add a single `MTLResidencySet` on runtimes/OSes
  that expose it. Assert every BufHandle has a live handle and is included in the resource set.
- **Per-step data — write IN PLACE (no re-encode):** the trainer ALREADY does this —
  `ensureCachedRuntimeTensor`/`trainingOverwriteF32` (real_autodiff_trainer.zig ~1056-1081) overwrite
  the cached runtime-input CT in place when shape matches; LoRA/regular params bind the persistent
  `device.weight` (~653-668). So input_ids/attn_bias/targets/params keep the SAME buffer address —
  the ICB's bindings never re-encode; only contents change via the existing uploads.
- **Replay each step:** one compute encoder on `active_frame_cb` -> `executeCommandsInBuffer:withRange:`.
  **Hybrid/piecewise** (like vLLM piecewise cudagraph): partition `dispatches` into maximal
  contiguous `icb_ok` runs; per run emit one `executeCommandsInBuffer:` (pre-stored NSRanges),
  eager-encode each `eager_only` op inline on the same encoder.
- **In-flight:** `decoderRuntimeSubmitAndWaitFrame` blocks before the next step → one ICB reused with
  no mutation-while-in-flight hazard (single ICB suffices; if we later pipeline, double-buffer only
  the in-place input buffers).
- **Flag:** `TERMITE_METAL_ICB_STEP` (default-off).
- **Device support check:** gate build/replay behind `supportsFamily`/selector checks for compute ICB
  support. If unsupported, automatically fall back to Phase 1 eager replay and emit a one-line stat,
  not a hard training failure.

## Critical files

- `src/graph/metal_partition_executor.zig` — pool + output_hint + heuristic guards (Ph0);
  `CompiledStepPlan` record/replay on the persistent struct (Ph1); ICB build/replay + hybrid stitch (Ph2)
- `src/ops/metal_compute.zig` — thread `output_hint`; route hot ops to `*Into`; wrap pooled
  outputs as non-owning `MetalTensor` device views while normal CT freeing only drops the wrapper
- `src/backends/metal_tensor.zig` — existing `deviceBorrowed`, `retainedView`,
  `retainedStorageView`, and release-on-drop semantics for non-owning pooled CT wrappers
- `src/backends/metal_runtime.zig` + `src/backends/metal_kernels.m` — `*Into` variants (Ph0);
  allocation-id output-pool buffer ABI; ICB descriptor/encode/execute, resource binding/residency
  set, per-op param buffers (Ph2)
- `src/graph/buffer_plan.zig` — (read-only reuse: `allocations[]`, `slotForNode`, `lifetimesOverlap`);
  optional debug assert on graph-output allocations
- `src/finetune/real_autodiff_trainer.zig` — (reuse) in-place runtime-input/param uploads target the fixed buffers
- Reuse: persistent executor struct + partition fingerprint, the graph-plan allocation policy as a
  reference point only (`reserveGraphPlanSlots`/`graph_plan_buffers` are fixed-slot runtime scratch today),
  the in-Zig parity gate (`training.zig` `training_graph_executor_parity` ~1616 + per-node ~560),
  loop-profile sub-timers, the watchdog + ladder scripts.

## Verification (after EACH phase; gate before proceeding)

```bash
cd zig/pkg/inference
# parity: in-Zig graph-vs-direct must stay green (rel ~1e-7) + identical loss trajectory, flag ON
env \
  TERMITE_METAL_SLOT_BOUND_OUTPUTS=1 \
  TERMITE_DEBERTA_FUSED_ATTENTION=1 \
  TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR=1 \
  TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK=1 \
  scripts/gliner2_memory_watchdog.sh output/p.log -- \
  zig build -Dmetal=true -Doptimize=ReleaseFast train-gliner2-autodiff -- \
  --model-dir /private/tmp/termite-models/gliner2 --train-data /tmp/gliner2_hard.jsonl \
  --out-dir output/p --epochs 4 --batch-size 2 --max-examples 4 --seq-len 64 --backend metal \
  --objective gliner2-total-loss --lora-rank 4 --lora-alpha 8 --lora-dropout 0 --max-span-width 4 \
  --lora-only-trainables --span-loss-reduction sum --span-positive-weight 1 --span-negative-weight 1 \
  --span-hard-negative-weight 1 --seed 42 --learning-rate 1e-3
# Add TERMITE_METAL_COMPILED_STEP=1 for Phase 1.
# Add both TERMITE_METAL_COMPILED_STEP=1 and TERMITE_METAL_ICB_STEP=1 for Phase 2.
# wall: TERMITE_METAL_PARTITION_LOOP_PROFILE=1 at b1 s128 — track command_path_ms/execution_ms
#   Ph0 expect ~neutral; Ph1 expect ~40ms off (decision overhead); Ph2 expect command_path → ~0,
#   step → ~GPU submit/wait + boundary. Read trainer_total_ms from training_metrics.jsonl.
```

Also add focused unit/smoke coverage before the real GLiNER2 ladder:

- `buffer_plan`/pool mapping test: views share source allocation, runtime inputs/constants are excluded,
  graph-output pooled slots are non-reused, and allocation-id -> buffer mappings rebuild on fingerprint
  mismatch.
- Phase 0 executor smoke: a small graph with output/input last-use aliasing refuses the hint and still
  matches direct execution.
- Phase 1 record/replay smoke: record hash changes on shape/op/binding changes; replay refuses stale
  descriptors.
- Phase 2 ObjC smoke: create a one-kernel compute ICB, bind an input/output buffer plus param buffer,
  execute through a compute encoder, and verify output bytes before wiring the full graph.

**Success:** Ph0 parity-clean + ~neutral (prerequisite); Ph1 parity-clean + ~40ms off; Ph2
parity-clean + the ~130ms encode collapses (step approaches the ~4.5ms GPU + ~31ms boundary +
overhead). Each flag default-off and flippable on.

## Risks

1. **Phase 0 in-place aliasing** (highest) → refuse output_hint overlapping a live input; debug-assert.
2. **Residency completeness** (Ph2) → a missing `useResource`/residency-set entry = garbage/crash;
   assert every recorded BufHandle is resident.
3. **ICB arg-index / `maxKernelBufferBindCount` limits** (~31) → assert max-arg-count at record;
   params-as-buffer adds one binding per op.
4. **Fixed graph-plan slot collision** → do not consume the 29 existing runtime graph-plan slots for
   arbitrary node outputs without a capacity audit and explicit constant migration; prefer the
   executor-owned allocation-id pool.
5. **Plan-cache invalidation** → rebuild pool/record/ICB on any shape change (seq-len/batch); key on
   the SAME fingerprint as `runtime_region_plan`; the in-Zig parity check guards every run.
6. **Graph output lifetime** → pooled graph outputs must not regress the current post-frame owned-copy
   guarantee; otherwise training extraction can read recycled or stale storage.
7. **Scope/uncertainty** → ~1.5–2.2k LOC across 3 phases. Phase 0 is neutral-but-mandatory; if Phase
   1 alone (the ~40ms) underwhelms or Phase 2 hits an ICB wall (incompatible op mix), the phases are
   independently shippable and the flag is default-off — measurement gates each step (12-false-fix
   discipline).

---

## Provenance / supporting findings (this session)

- **Per-step cost map (b1, post-fused-attention, post-sumsq-fix):** ~130–180ms per-op Metal encode
  (4500 ops × ~27µs) is the wall; ~40ms per-node decision overhead (defer/elision/skip/free, pure
  functions of the static graph); ~31ms boundary-output host readback; ~6ms submit; ~4.5ms GPU.
- **Refuted levers (measurement-driven):** frozen-param device residency (GPU already trivial);
  decision-only plan-replay alone (only ~8ms of the dispatch decisions; the loop machinery is the
  ~40ms — addressed by Phase 1); LayerNorm-backward op-fusion (correct + parity-clean + −18% op-count
  but WALL-TIME-NEUTRAL — cheap ops dominate the removed count; proved per-op encode is non-uniform).
- **Why ICB, not more fusion:** PyTorch keeps LayerNorm as 1 fwd + 1 fused bwd (native_layer_norm_backward)
  and is compute-bound on CPU; vLLM's decisive lever is CUDA Graphs (capture-replay), fused kernels
  secondary; llama.cpp-Metal re-encodes + parallelizes but most Metal engines avoid ICB; MPSGraph is
  true compile-once-replay. ICB is the Metal analog of CUDA Graphs and the correct primitive for our
  static-graph/data-only-changes case. Static buffers are the shared prerequisite (Phase 0).
