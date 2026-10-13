# Qwen3.8 on Apple Silicon: branch review and 48 GB Studio handoff plan

Date: 2026-10-13. Branch `qwen38-flash-next-metal` at `0e4c6aaa7f` (3 commits
over `e8a1bfcabc`). Reviewed on the 16 GiB M4 MacBook Air (Mac16,12). Nothing
was built or executed for this review; findings come from reading code, the
checked-in evidence, the GGUF headers on `/Volumes/Extreme`, and the janas
source at commit `94b72c96`. Throughput figures marked *estimate* are
arithmetic, not measurements.

This extends, and does not replace,
[`qwen38-ssd-production-review-2026-10.md`](qwen38-ssd-production-review-2026-10.md),
which targeted the 16 GiB host only. Its three open release gates (independent
near-limit parity, clean-swap qualification, session endurance) carry over.

## 1. Summary

1. **The "resident router, streamed experts" design applies to Flash only.**
   Qwen3.8-27B is dense (`qwen35`, 64 trunk layers, no router). On the Air it
   must read 14.34 GB per token and is bound to ~0.06 tok/s by the USB SSD; no
   kernel work changes that. On the Studio the entire 16.45 GB model fits in
   memory, so the right design there is fully resident, with no streaming.
2. **No wrong-output defect was found on either default decode path.** Pager
   transactions, lease/GPU ordering, dequant math, struct layouts and buffer
   lifetimes were traced and hold. One real wrong-output bug exists in shared
   kernels outside these runners (§3, F1).
3. **The branch cannot use a 48 GB machine as written.** Flash silently clamps
   to a 10 GiB process cap and 128 slots/layer; 27B rejects budgets above
   4 GiB and has exactly two layer buffers. Caps are duplicated in the Python
   harnesses.
4. **Kernels are correct but not competitive.** Flash decode issues about
   4,100 GPU dispatches (one encoder each) and 98–146 command-buffer waits per
   token and bypasses almost all of the tuned Gemma4 machinery in this repo.
   27B runs 37% of its trunk bytes through a per-element megakernel.
5. **janas is a useful design reference but not a like-for-like benchmark.**
   It is x86/Vulkan only, has no Flash-Next architecture, and its headline
   80B numbers use ~45–50% expert residency, 4-bit `down` planes, an 8-bit KV
   cache and a cache pre-warmed on the identical prompt (§4).

## 2. Model facts that drive the plan

Sizes from the GGUF tensor tables (decimal GB).

| | Qwen3.8-Flash-Next UD-Q4_K_XL | Qwen3.8-27B UD-Q4_K_M |
| --- | --- | --- |
| Architecture | `qwen4exp`: 48 layers, 512 experts, top 10, shared expert, 3-gram PLE | `qwen35`: 64 trunk layers + 1 MTP block (`blk.64`, `nextn.*`) |
| Total | 111.3 GB, 4 shards | 16.45 GB |
| Routed experts | 77.0 GB = 24,576 experts × ~3.13 MB (gate Q4_K, up Q4_K, down Q5_1/Q8_0) | none |
| Row-read tables | 28.8 GB n-gram table (IQ4_NL, 160-wide rows) | 0.72 GB `token_embd` |
| Must be resident | ~5.3 GB: Q8_0 trunk, routers (F32), head, shared experts | 13.4 GB trunk + 1.04 GB head |
| MTP tensors in GGUF | none | yes |
| All-miss bytes per token | ~1.5 GB (480 experts) | 14.34 GB |

Consequences:

- An expert's gate, up and down live in three separate per-layer tensors, so
  one expert is three non-contiguous reads (six lanes ≈ two experts in flight).
- Flash weight bytes touched per decoded token are roughly 4 GB of Q8_0 trunk
  plus 1.5 GB of experts (*estimate*). At all-hit steady state the dense trunk,
  not the experts, sets the memory-bandwidth floor.
- `/Volumes/Extreme` is attached over USB and measured 0.87–0.97 GB/s. Every
  10% of expert misses costs ~150 MB ≈ 0.15 s/token on that link.

Memory plans (to be replaced by measured values on each host):

| Host | Flash | 27B |
| --- | --- | --- |
| Air, 16 GiB | 5.3 GB resident + ≤3.5 GiB cache ≈ 23 slots/layer, 4.5% of experts | streamed only; functional/qualification use |
| Studio, 48 GB | 5.3 GB resident + ~30 GB cache ≈ 199 slots/layer, 39% of experts; ~10 GB left for OS | fully resident, ~16 GiB including 4K state |

## 3. Review findings

Severity: P0 wrong output/crash, P1 robustness or handoff blocker, P2 minor.
"Confirmed" means verified by reading the cited code; nothing was reproduced
at runtime.

### Shared code (affects other models)

| ID | Sev | Finding |
| --- | --- | --- |
| F1 | P0, confirmed, pre-existing on main | `backends/metal_kernels.m:20683-20701`: the IQ4_NL and IQ4_XS `1x_reduce` kernels partition input blocks over `NSG = 4` SIMD groups (kernel source `:10084`, `:10098`) but are dispatched with the default `reduce_threads = 64` (`:20213`), i.e. two groups. Blocks 16–31 of every 32 are never summed, so any single-row matvec with `in_dim > 512` is silently wrong. The branch fixed exactly this for Q5_1 (`:20515`). Q5_0 (`:20498`), Q8_1 (`:20663`) and MXFP4 (`:20703`) follow the same pattern; Q8_K needs checking. Neither Qwen runner reaches these paths, but any GGUF with IQ4 tensors on the general path does. Existing fixtures are 256 wide and pass. |
| F2 | P1, confirmed | `zig/lib/tokenizer/src/hf_tokenizer.zig:~1154`: the Qwen Split pattern is recognised by exact string compare against two literals; any other pattern silently falls back to the GPT-2 scanner. The two runners are protected by the tokenizer hash pin; other Qwen-family tokenizers are not. |
| F3 | P2, confirmed | `models/tensor_store.zig:~1066`: a file carrying `split.count > 1` but renamed or standing alone now fails to open where it previously loaded as a single file. `describeTensorRange` now errors instead of returning null (CUDA callers tolerate it). |
| F4 | P1, confirmed | `.github/workflows/zig-tests.yml:651-659` runs only `-Dmetal=false` suites on Linux. No Metal kernel is executed in CI; Metal tests return `SkipZigTest`. |

Jinja, tokenizer and `native_generate.zig` flag changes were traced for
regressions against existing models; none found. The `expert_slot_arena.zig`
rewrite is a comptime genericisation with unchanged Gemma A4B behaviour.

### Flash (`qwen4exp`)

| ID | Sev | Finding |
| --- | --- | --- |
| F5 | P1, confirmed | Hard caps: `architectures/qwen4exp_runtime.zig:72` clamps to 10 GiB and to Metal's recommended working set (comparing host+backend bytes against a Metal-only figure); `models/qwen4exp.zig:246,257` stops at 128 slots/layer; slot indices are `u8` (`runtime/moe/expert_slot_arena.zig:31,34,56`, `qwen4exp_pager.zig:73,90`). Oversized `--moe-cache-mb` is truncated silently while the receipt reports the requested value. |
| F6 | P1, confirmed | One transient read error ends the session: `util/c_file.zig:514` flattens every `pread` error to `IncompleteRead` and the Darwin branch (`:757`) does not retry EINTR; `qwen4exp_pager.zig:232` has no retry although rollback is clean and the retry point is safe. On the Studio one EIO discards a 30 GB warm cache. |
| F7 | P1, confirmed | `qwen4exp_runtime.zig:689-691` (and `:555`) registers `errdefer self.failed = true` before the first pressure/cancel check, so a momentary memory-pressure blip at a token boundary poisons an intact decoder. `local_text_session.zig:222,237,252` makes the same condition fatal, and there is no per-request cancel: SIGINT always exits the process. |
| F8 | P1, confirmed | Every process start re-hashes all 111.33 GB with `F_NOCACHE` (16.46 GB for 27B). `qwen_local/benchmark.py` starts a process per case: one default Flash run reads ~668 GB for hashing alone. Flash opens each shard by path three times (parse, hash, read), so the hash does not bind the bytes later read; 27B correctly uses one descriptor. |
| F9 | P1, plausible | The sparse-attention block selection (`backends/qwen4exp.metal:215-228`) only begins pruning past position ~2051. Independent reference parity has been run at 12 tokens; behaviour beyond the pruning threshold is covered by a synthetic test only. |
| F10 | P2, confirmed | Prefill updates the eviction state once per group of ten experts in ascending ID order, +1 per expert regardless of how many tokens used it (`qwen4exp_runtime.zig:595-609`); decode starts with the highest IDs cached rather than the most used. Each prefill chunk also computes the full 248320×2560 head (`:658-662`). |
| F11 | P2, confirmed | `read_lane_count` looks tunable but `qwen4exp_pager.zig:149,193` hard-code the 6-lane layout; changing it is undefined behaviour or out of bounds. Lease invariants are debug asserts that compile out in ReleaseFast. Staging→slot adds a second copy of every miss (`:239`). |
| F12 | P2, confirmed | No prefetch of any kind, no cross-process cache persistence, eviction counters live on slots so frequency is forgotten on eviction. Context is capped at 4096 in three places plus a `[4096]i32` stack history (`local_text_session.zig:255`) that becomes an out-of-bounds write if only the cap is raised. Residency-set creation is fatal below macOS 15. |

Decode control flow per layer today: batch 1 (attention/linear branch, HC,
shared expert, router, GPU top-10) → **wait** → CPU reads ten IDs, starts
reads → hit-expert batch → join reads, copy, publish → **wait** → miss
experts + accumulate → **wait**. With embedding and head: 98 waits per token
all-miss, 146 at high hit rate. The all-hit case is therefore the slowest in
synchronisation terms.

Why the opt-in experiments lost: `GROUPED_EXPERTS` dequantises one weight per
thread with no scale reuse and still runs twice per layer; `BATCH_LINEAR`
re-reads weights per row (no arithmetic saved) while tripling batches;
`PREFILL_MPS` re-dequantises a full matrix per ≤32-row group and most experts
in a short prompt see 1–3 rows.

### 27B (`qwen35` streamed)

| ID | Sev | Finding |
| --- | --- | --- |
| F13 | P1, confirmed | Two-buffer/4 GiB design is structural: cap rejected above 4096 MiB (`architectures/qwen35_streamed.zig:41`), `buffers: [2]`/`l % 2` indexing (`:85`, `:198-216`, `:350-355`), fixed 288 MiB slots, per-layer `endBatch`, lease release drops the layer (`runtime/dense_weight_pager.zig:94`). |
| F14 | P2, confirmed | Lookup verification (2–7 rows) uses scalar per-output kernels with a different summation order from single-row decode (`backends/qwen35_streamed_kernels.zig:105`), so speculation is not bit-identical to plain greedy on near-ties. Rejection replays a full target pass although only recurrent state is missing (`qwen35_streamed.zig:299-311`). |
| F15 | P2, confirmed | Failure inside the decode loop overwrites the detailed receipt with a minimal one (`native_qwen35_generate.zig:69-75,179`). Greedy only, although the artifact's defaults are temperature 1.0, top-k 20, top-p 0.95. |

The existing general-path `qwen3_5` family (`architectures/gpt.zig:7927`)
cannot load this GGUF: GGUF mapping is refused (`models/gpt.zig:1806`), its
conv/delta-rule/gated-norm run on the host with four readbacks per layer, head
ordering differs, and IQ3_S has no Metal kernel. The streamed runner is the
shorter path to a resident 27B.

Per-token GPU time (~1.6 s on the Air) by cause, ranked: IQ4_XS/IQ4_NL/IQ3_S
(37% of trunk bytes) through the per-element megakernel; Q5_K (34%) on a
second-tier reduce kernel; Q3_K scalar at one row; Q4_K/Q6_K missing
slot-keyed paths and gate/up fusion because every matrix rebinds slot 0; the
tuned head kernel family hard-gated to another model's shape; 65 synchronous
frames and ~1,300 encoders per token.

## 4. Competitive reference: janas

Verified in source unless marked (doc).

| Technique | janas | This branch |
| --- | --- | --- |
| Expert cache | one global arena sized from free RAM | equal per-layer split, ≤128 slots |
| Eviction | global LRU; Belady-style during oversized prefill blocks | per-layer LFU with halving decay, slot-keyed counters |
| On-disk layout | `.jns`: gate+up+down contiguous per expert, 4096-aligned, CRC | three GGUF tensors per layer, three reads per expert |
| Read path | `O_DIRECT`; io_uring for small batches, pread pool otherwise; read straight into the slot | six `pread` lanes into staging, then memcpy |
| Prefetch | next layer's router applied early, top 10, ≤2 reads, hysteresis | none |
| Warm start | persisted per-expert hotness, idle-time warmer | none |
| Decode compute | CPU, int8 activations, experts grouped by ID in one dispatch | GPU, FP32, 51 dispatches per layer for experts |
| Drafts | model MTP block (recovered from checkpoint), n-gram lookup, cost planner; 74–89% acceptance claimed (doc) | 27B prompt lookup prototype; none for Flash |
| Quality-affecting defaults | 4-bit `down` planes when cache < experts, 8-bit KV, int8 activations, non-exact GPU prefill; expert skipping below 20% coverage | none; FP32 throughout |

Fair-comparison caveats: janas's 62/24 tok/s figure is a 48.4 GB model on
32 GB with about half its experts resident, on a 5.7 GB/s NVMe, 16 generated
tokens, one round, replaying a sequence the cache has already seen. It does
not run on macOS and cannot load Flash-Next. It reports 4.4 tok/s writing for
dense 27B on that laptop (doc). The meaningful same-artifact baseline on the
Studio is llama.cpp Metal at the revision pinned in `qwen4exp/artifact.json`.

## 5. Plan

Ordering principle: unblock the Studio, measure, then optimise the measured
bottleneck. Each phase ends with a parity gate (full-vocabulary logits,
relative L2 ≤ 0.002, matching argmax, against the existing oracles) and the
session/lifecycle campaigns.

### Phase 0 — on the Air, before handoff

0.1 **Fix F1** as a separate change to main: set 128 threads for IQ4_NL,
IQ4_XS, Q5_0, Q8_1, MXFP4 (audit Q8_K and the sibling encoder near `:18433`);
add fixtures at widths 1024 and 2048 so the reduce path is compared with the
scalar kernel above 512.

0.2 **Machine profile instead of literals (F5, F13, F12).** One place derives
process cap, expert-cache bytes and context from physical memory and Metal's
recommended working set, with explicit overrides. Remove the 10 GiB and
4 GiB literals, the 128-slot cap, widen slot indices to `u16`, replace the
`[4096]i32` histories with context-sized allocations, and make truncation an
error. Receipts report actual slot counts and bytes. Mirror in `chat.py`,
`evidence.py`, `benchmark.py`.

0.3 **Robustness (F6, F7, F11).** Preserve errno and retry EINTR on Darwin;
bounded retry of failed expert reads before poisoning; move the pre-mutation
checks ahead of the `errdefer`; make memory pressure a recoverable turn error
(stop the turn, optionally shrink the cache); add per-request cancellation
that keeps the session and cache alive; promote lease invariants to release
checks; derive lane layout from `read_lane_count`.

0.4 **Verification without the tax (F8).** Bind hash to bytes by hashing and
reading through the same descriptors for Flash. Add a verification stamp
keyed by path, size, mtime, inode and digest, with `--verify full|stamp` and
the mode recorded in receipts; qualification runs keep `full`.

0.5 **Go/no-go measurement: hit rate versus cache size.** Record route traces
for ≥500 decoded tokens on each of chat, code and document workloads plus a
multi-turn session (`ANTFLY_QWEN4EXP_ROUTE_TRACE`; ~12 minutes each on the
Air). Extend `cache_replay.py` to sweep 1.5–36 GB and compare current
LFU-decay, global LRU, expert-keyed LFU with ghost history, weighted
per-layer allocation and a Belady oracle. Output: bytes read per token at
30 GB. This decides whether 39% residency gives a usable hit rate and which
policy Phase 3 implements.

0.6 **Handoff kit.** Commit this plan and Phase 0. Copy both model
directories preserving layout (`UD-Q4_K_XL/` under the Flash sidecar
directory), verify against `artifact.json`. Pinned and non-portable items:
shard names and digests (`models/qwen4exp.zig:22-27`,
`models/qwen35_streamed.zig:23-24`), three tokenizer sidecar digests, vocab
248320, the exact Split regex, the llama.cpp reference build under `/tmp`,
and all `*-m4.json` evidence (historical, tied to this host).

### Phase 1 — Studio bring-up

1. Record chip, RAM, macOS, `recommendedMaxWorkingSetSize`, storage placement.
2. Place the models on internal NVMe or a Thunderbolt enclosure; run
   `storage_probe.py` on each candidate volume. USB at ~1 GB/s caps Flash at
   roughly 6 tok/s even at 90% hits (*estimate*).
3. Build both runners and the focused Metal suites; rerun
   `qualify_sessions.py`, `qualify_lifecycle.py` and the parity checks at the
   **old** caps to establish that the port reproduces the Air's results.
4. Build the pinned llama.cpp and record its Metal throughput on both
   artifacts as the baseline.

### Phase 2 — 27B fully resident (first Studio win)

1. N-slot mode in the streamed runner with N = 64: buffers sized to actual
   layer spans (13.4 GiB), retained leases, no reads after load, hashing
   folded into the load pass, one frame per token. Expected to remove the
   ~14.8 s/token of read wait.
2. Route IQ4_XS/IQ4_NL through the fixed reduce kernels; prebuilt tensor table
   instead of per-layer name scans; drop the per-`mm` pressure snapshot.
3. Kernel work in byte-share order: Q5_K port of the Q4_K v2 design; Q3_K and
   IQ3_S reduce kernels (or load-time transcode of those 11 tensors, behind a
   parity gate); generalise the tuned head kernels to 5120×248320; per-tensor
   slots to enable gate/up pair fusion; GPU embedding lookup.
4. Sampling (temperature/top-k/top-p) by reusing the existing GPU sampling
   kernels; then lift the 4096 context cap with the split-GQA/flash attention
   kernels and an FP16 KV cache.
5. Later: move the conv/delta-rule ops and a GGUF mapper into the general
   path for HTTP serving.

Exit: ≥0.9× llama.cpp Metal decode and prefill on the same GGUF and host,
parity at 5, 128, 512 and 4094 tokens.

### Phase 3 — Flash cache and I/O at scale

1. Cache sized from the machine profile; per-layer allocation from the
   Phase 0.5 result (or a shared pool if global wins); expert-keyed frequency
   with ghost history; prefill counts tokens per expert and uses
   wanted-again-latest when the union exceeds capacity (F10).
2. **Packed expert sidecar**: one contiguous, page-aligned record per expert
   (gate, up, down), bit-identical payload, versioned and hashed, generated
   from the pinned GGUF; original GGUF untouched. Read directly into the
   destination slot (no staging copy). Costs ~77 GB of extra storage.
3. Next-layer router prefetch gated by measured miss rate; depth tuned per
   storage speed (deeper on USB).
4. Persisted per-artifact hotness profile and idle-time warming, so a fresh
   process starts with the hot set loading.
5. Skip the vocabulary head on non-final prefill chunks.

Exit: measured hit rate and bytes/token at the Studio budget match the
Phase 0.5 replay within tolerance; cold-start time to steady state recorded.

### Phase 4 — Flash kernels

Ranked by expected payoff at high hit rate:

1. Single-batch all-hit fast path (146 → 98 waits), embedding folded into the
   first batch.
2. One fused indexed expert kernel: gate+up+SwiGLU for all ten routes, then
   down with weighted accumulate (51 → 2–3 dispatches per layer), built on the
   existing pair-activation reduce inner loops rather than op 28.
3. One compute encoder per frame, then prepared frames; shapes are static per
   layer type.
4. GPU-side routing: expert→slot table in a buffer, miss flag written by the
   kernel, replay from the first missing layer only (98 → ~1–2 waits per
   all-hit token). Reuses the arena's `writeExpertToSlotMap` and the Gemma MoE
   checkpoint machinery; needs Q4_K/Q5_K/Q5_1 forward kernels.
5. Dense trunk fusion with the existing Q8_0 qkv/pair/rms-scale kernels; this
   is where most bytes are.
6. Long-context attention: parallel rank selection for the block index, a
   compact selected-token list, fused flash-style kernel, FP16 KV. Close F9
   with independent parity past position 2051 first.
7. Prefill: batch dense Q8_0 projections across the chunk, lift the 32-row
   cap, overlap group reads with GPU work.
8. Hoist ~500 `getenv` calls per token, name-hash weight lookups and
   per-layer pressure checks out of the loop.

### Phase 5 — speculation

- 27B: load `blk.64` and drive `verifyDraft` from the model's own MTP head
  (input construction must be confirmed against the HF reference); per-row
  GPU state snapshots instead of replay; multi-row kernels with single-row
  summation order (F14).
- Flash: the GGUF has no MTP tensors. Start with prompt/n-gram lookup; each
  extra verified token routes to different experts, so gate drafting on
  measured miss cost. Recovering an MTP block from the original checkpoint is
  a separate, optional artifact.

### Phase 6 — the Air

- Flash: raise the cache to the largest admitted size, apply Phases 3–4
  (hotness, prefetch, packed sidecar matter most on the slow link). Realistic
  ceiling with ~4.5% residency is I/O bound; the replay from 0.5 gives the
  number.
- Quality-trading options (reduced-bit expert planes, cache-conditional
  routing as janas does below 20% coverage) only as explicit, receipt-reported
  opt-ins with a quality gate, never defaults.
- 27B stays a functional/qualification configuration on 16 GiB.

### Phase 7 — release gates

1. Independent full-vocabulary parity matrix on final binaries: short,
   chunk boundaries, past the sparse-attention pruning threshold, near 4K.
2. Clean-swap qualification (zero swapin and swapout deltas) and a
   multi-hour session endurance run with footprint trend.
3. Benchmark protocol: fixed chat/code/document corpus, ≥256 generated
   tokens, cold and warm separately, AB/BA order, TTFT and p50/p95 inter-token
   latency, hit rate and bytes/token in every receipt, llama.cpp Metal on the
   same artifact as baseline.
4. A Metal gate that actually runs (self-hosted macOS runner or a required
   local script whose receipt is checked in) covering kernel fixtures and the
   focused suites (F4).

## 6. Decisions needed

1. **Studio storage**: internal NVMe (needs ~128 GB, plus ~77 GB if the
   packed sidecar is adopted) or a Thunderbolt enclosure. Staying on USB
   bounds Flash throughput regardless of kernel quality.
2. **Scope of "production"**: local single-owner CLI/chat (the previous
   review's scope) or HTTP serving through the general inference path. The
   latter adds Phase 2 step 5 and its Flash equivalent.
3. **Bit-exactness**: the plan keeps weights bit-identical and FP32
   activations. Matching janas-style numbers at low residency requires
   quality-trading options; confirm these stay opt-in.
4. **Throughput targets**: proposed as relative gates (≥0.9× llama.cpp Metal
   for 27B; Flash measured against its own all-hit ceiling and the Phase 0.5
   replay) until Studio measurements exist.

## 7. Phase 0 status

Implemented on the 16 GiB M4 on 2026-10-13, on top of `638bbbd52d`. Phases 1–7
need the Studio and are untouched. Figures below are from this host with other
applications open: system swap counters moved during every model run, so they
are functional results, not clean-memory or timing evidence.

| Item | State | What changed | Check |
| --- | --- | --- | --- |
| 0.1 F1 | done | 128-thread dispatch for the Q5_0, Q8_1, IQ4_NL, IQ4_XS and MXFP4 `1x_reduce` kernels, with a fallback to the scalar kernel if the pipeline cannot take 128 threads. Audit: Q8_K, Q4_1 and Q5_K stripe over two SIMD groups and were already correct, as is the pair encoder. | New `backends/metal_quant_reduce_test.zig`: 11 formats × widths 256/512/1024/2048, single-row reduce and three-row batch kernels against the CPU codec. On the unfixed kernels it fails at Q5_0, width 1024 (−1.874 against −2.613); 256 and 512 pass, as predicted. |
| 0.2 Machine profile | done | `runtime/machine_profile.zig` derives the process cap, Flash expert cache and context limit; the 10 GiB and 4 GiB literals and the 128-slot cap are gone; slot indices are 16-bit for arenas wider than 254 slots (A4B keeps bytes); token histories are context-sized; out-of-range budgets are errors. `--dry-run` works for both runners. Harness caps removed from `chat.py`, `evidence.py`, both `benchmark.py`; receipts are checked against the enforced values. | The Air derives a 10.0 GiB cap with a 3.71 GB cache (24–25 slots, 4.8%) for Flash and a 3.99 GiB cap for 27B. Unit tests cover 512 slots per layer and a 48 GB host. |
| 0.3 Robustness | done, one option open | `RangeReader.read` keeps the errno class and retries EINTR on Darwin; expert, n-gram and embedding reads retry transient errors three times; lease and transaction invariants are release checks; lane layout follows `read_lane_count`. Both decoders separate terminal failure from "sequence needs reset" and run their pressure/cancel checks before anything is mutated; a single token is never interrupted for pressure or a request cancel, only a multi-token chunk is. Sessions gained `{"cancel":true}` and non-fatal, classified `error` events. | Fault-injection tests for retry, exhaustion and rollback; session input and recovery tests; session qualification below. **Not done:** shrinking the expert cache under pressure (the plan marks it optional; it belongs with the Phase 3 allocation rework). |
| 0.4 Verification | done | Flash hashes and reads through one descriptor per shard, each checked against the parsed file's identity. `--verify full\|stamp`; receipts record mode and outcome; qualification harnesses force `full`. | Stamp test covers fallback, replacement and corruption. Startup with a valid stamp: Flash 173.8 s → 7.2 s, 27B 29.5 s → 1.7 s. |
| 0.5 Go/no-go | done: go | `record_routes.py`, and `cache_replay.py --manifest` with seven policies and a budget sweep. | Replay model tests; the replay reproduces the three one-shot runs' measured hits and misses exactly. Results in §7.1. |
| 0.6 Handoff kit | done except the copy | `verify_install.py` and the "Handoff to another Mac" section of `scripts/qwen_local/README.md` (pinned and non-portable items, bring-up order). | The copy itself and the commit are operator steps. |

Token IDs after the change match the checked-in M4 records for the chat
benchmark prompt: Flash `[29108, 4009, 1100, 9872, 303, 660, 17425, 430]`
(1536 MiB cache, derived 8.04 GiB cap, full and stamp verification) and 27B
`[29108, 4009, 5435, 660]` (derived cap). Focused suites: Metal 27B 36 passed /
1 skipped, Flash 52 / 4; CPU-only 27B 31 / 6, Flash 39 / 16; Python 24 passed.

`qualify_sessions.py --family flash --cache-mb 1536` (full verification, derived
cap) passed all 46 functional cases in raw and chat mode, with the recorded
IDs (`[11751, 13]`, `[21047, 13]`, `[14556]`, `[17981, 27450]`), exact prefix
reuse, reset parity, the new cancel case, and a post-warmup footprint range of
128 KiB / 400 KiB over eight cycles. Its clean-swap condition failed (148,609
swapins, 173,231 swapouts in the window), as every run on this loaded host did.
That campaign ran one build before the last two changes (explicit caps must
hold the admission reserve; a single token is never interrupted). On the final
binaries: a reply cancelled during decode ended with
`finish_reason: "cancelled"`, `state_reset: false`, and the next turn reused all
22 consumed tokens; `qualify_lifecycle.py --family 27b` passed all eight phases
(longest cooperative exit 0.15 s) and the Flash prefill, decode and idle phases
passed; 27B prompt lookup, one-shot and in a session, accepted 6 of 6 proposals
with no replay, as in the M4 record; and all four route recordings in §7.1
completed, the session extending one sequence across five turns.

Two things the implementation found that the review did not:

- **A derived cap must be tight.** The admission controller treats a cap that
  exceeds current availability as a dynamic-pressure envelope and reserves the
  node headroom again inside it, which no realistic Flash plan survives. A
  derived cap is therefore the plan plus 768 MiB, and the 256 MiB of that for
  the process's own pre-admission footprint is needed: without it the 27B plan
  was refused by 20 MB.
- **`errdefer |err|` no longer parses in Zig 0.17**, so the failure
  classification wraps the token and chunk bodies instead of capturing in an
  `errdefer`.

F1 is isolated so it can go to `main` separately: the `metal_kernels.m`
dispatch hunks (plus the same hunk for Q5_1, which `main` does not have yet),
`metal_quant_reduce_test.zig`, and its import in `inference.zig`. That
combination was checked to apply to `main` at `b467a7f19f`.

### 7.1 Hit rate versus cache size (Phase 0.5)

Routes were recorded on the Air with the final Phase 0 binary at a 1536 MiB
cache (routes do not depend on cache size): chat, code and document prompts
with 519 decoded tokens each, and a five-turn chat session with 545. The
summary, with per-workload results for every budget and policy, is
[`route-replay-m4.json`](../../../../../zig/pkg/inference/scripts/qwen4exp/route-replay-m4.json).
Each run starts with an empty cache, so first touches are counted as misses.

The recordings themselves measured 0.61–0.78 tok/s, 854–1004 MB read per
decoded token and a 33–44% hit rate at 10–11 slots per layer. Replaying the
runtime's policy at those slot counts reproduces the measured prefill and
decode hits and misses exactly for all three one-shot runs, so the model of
acquisitions matches the executor.

Predicted decode megabytes read per token, mean of the four workloads (hit
rate in parentheses). All-miss is 1504 MB per token.

| Cache | Runtime (per-layer LFU-decay) | Per-layer LRU | Global LRU | Expert-keyed LFU with ghosts | Same, weighted slots | Belady per layer | Belady global |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.6 GB | 940 (37.7%) | 949 | 958 | 934 | 932 | 908 | 676 |
| 4 GB | 709 (53.0%) | 708 | 733 | 669 | 671 | 481 | 424 |
| 8 GB | 509 (66.3%) | 502 | 503 | 466 | 462 | 279 | 246 |
| 12 GB | 371 (75.5%) | 370 | 354 | 339 | 329 | 182 | 156 |
| 20 GB | 202 (86.7%) | 201 | 177 | 187 | 171 | 90 | 76 |
| 30 GB | 105 (93.1%) | 97 | 81 (94.6%) | 96 | 84 (94.4%) | 50 | 43 (97.2%) |
| 36 GB | 67 (95.6%) | 66 | 55 | 68 | 59 | 41 | 36 |

At 30 GB by workload (runtime policy / global LRU / weighted / Belady global):
chat 55 / 42 / 45 / 32, code 122 / 97 / 99 / 51, document 142 / 98 / 110 / 38,
session 99 / 89 / 81 / 49 MB per token. Over the second half of each decode the
means are 113 / 84 / 85 / 33.

What this decides:

- **Go.** 39% residency turns a 1.5 GB all-miss token into roughly 0.1 GB with
  the policy the runtime already has: about 0.12 s of read service per token on
  the USB SSD and a few hundredths of a second on NVMe. That is from a cold
  cache over about 500 tokens; a warmed cache can only do better.
- **Residency is not the whole answer.** In 500 to 1,700 tokens each workload
  touched 280–418 of the 512 experts per layer on average, so no policy removes
  the first-touch misses. Online policies at 30 GB are still about twice the
  offline bound, which is the headroom prefetch and a persisted hotness profile
  (Phase 3.3–3.4) can go after.
- **Policy for Phase 3.** The equal per-layer split wastes slots: layers need
  between 84 and 358 slots at 30 GB when allocated from the other workloads'
  miss curves. From 12 GB up, a global LRU pool and weighted per-layer slots
  with expert-keyed, ghost-remembering LFU both beat the runtime policy, by
  20–23% at 30 GB. Below 8 GB the order reverses for the global pool, which is
  the worst choice on the Air, where expert-keyed LFU with ghosts leads.
  Phase 3.1 should therefore implement expert-keyed frequency with ghost
  history and weighted per-layer allocation: it is first or second among the
  online policies at every budget from 1.6 to 36 GB and does not regress the
  Air. A shared LRU pool is slightly better at 30 GB and above and is the
  simpler alternative if the Studio is the only target.
- **Frequency alone misleads on long shifts.** On the code workload the
  runtime's slot-keyed LFU is the worst online policy at 30 GB and gets worse
  in the second half (160 MB per token against 137 for plain per-layer LRU):
  it keeps experts that were hot for the previous part of the answer.

Limits: four prompts on one host, about 500 decoded tokens each, greedy decoding,
thinking disabled; bytes are application read ranges of whole experts, and the
weighted allocation is calibrated leave-one-out on the other three workloads.
The Studio should repeat the recording at its derived cache and compare its
measured hit rate with these predictions (Phase 3 exit).
