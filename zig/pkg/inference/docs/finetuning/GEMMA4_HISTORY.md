# Gemma4 historical experiments and qualification notes

These dated notes preserve past measurements and decisions. They do not qualify
the current source. See [the shipping contract](GEMMA4.md) and
[the current remediation ledger](GEMMA4_REVIEW_REMEDIATION.md).

## Validation Status

### Current source verification

The current branch adds prepared-input v6 causal-tokenization plus
source/group/media provenance, mandatory separate eval,
vocabulary/sequence/aggregate admission, PEFT-schema config plus Antfly
sidecar, typed public Gemma4 operations, four-step recipes and workflows,
immutable file publication, loss-only compiled Metal evaluation, strict
promotion/upload telemetry, exact-resume substrate, and packed Q4_0/Q4_K/Q6_K
frozen-linear input gradients.

The 2026-08-19 integrated source snapshot passed these local gates:

- final required-device ReleaseFast `test-gemma4-finetune`: 263 selected, 261
  passed, two optional real-model tests skipped, zero failed. This includes the
  strict tiny BF16 CLI optimizer step, packed-format tile-boundary numerical
  checks, Q4_0 graph-executor dispatch, direct-GGUF loss repeatability, and the
  real subprocess `model-command` success/failure/attestation contract;
- the earlier non-Metal ReleaseFast `test-gemma4-finetune`: 180 passed, eight
  Metal-only tests skipped, zero failed, confirming that the training API and
  unsupported-backend stubs remained portable at that checkpoint;
- current-source ReleaseFast inference-edition `antfly`: all ten build steps
  passed and `antfly inference finetune --help` reached the unified dispatcher;
- 125 offline Gemma4 oracle, publication, PEFT-key, MLX-build-attestation,
  benchmark-runner, and campaign-contract tests, all passing; and
- the checked-in oracle lock validation at
  `sha256:c848acb5fa38abda012f52c31cc122927b26775896d4a460e58cc9336cf27383`.

One bounded real `google/gemma-4-E2B-it` BF16 run completed through the public
Metal CLI with prepared-v6 disjoint inputs, rank-4 Q/V LoRA, one optimizer
step, one Metal optimizer update, finite gradient norm `2.42837`, and zero
recorded graph/interpreter fallback. Held-out loss moved from `6.82213` before
the step to `6.20983` after it. The immutable output and run manifest are under
`/private/tmp/antfly-gemma4-e2b-real-smoke-20260811-1555/` on the qualification
host. The trained adapter also exported through the public stock-key PEFT
boundary and completed a full E2B stock-PEFT load/forward smoke.

The SDK 26.2 MLX reference lane is now executable. On 2026-08-12, a separate
diagnostic-only pinned MLX 0.31.2 / MLX-LM 0.31.3 E2B run completed the exact
sequence-128, accumulation-1, rank-16/alpha-32 `peft-qv` workload. Twenty
synchronized measured optimizer steps had median/mean latency
`0.263281/0.263396 s` (about `485.96` input tokens/s); allocator and process
physical-footprint peaks were `10.014` and `10.708 GiB`, with zero swap. The
measured window did incur 1,097,728 bytes of page-ins and 589,824 bytes of
page-outs, so it would fail the release lane's zero-paging threshold. The run
loaded the SDK-required attested `libjaccl.dylib` and
closed all BF16-base/F32-LoRA/F32-gradient/F32-AdamW precision inventories.
This artifact is explicitly not release evidence: the Antfly checkout was
dirty, release memory thresholds were not enforced, and there is not yet an
alternating five-repeat campaign. It establishes that the MLX reference runs
end to end.

The matching Antfly diagnostic lane now also completes end to end. Its current
default-on Metal route uses simdgroup matrix multiply for both BF16
frozen-linear forward and input-gradient products when rows and both matrix
dimensions are at least 128. Exact 64-row-compatible shapes use a 64-row by
64-column by 32-K tile with eight simdgroups and 256 threads, doubling weight
reuse relative to the preceding 32-row tile. Devices that cannot admit the
256-thread pipeline, irregular matrix dimensions, and non-64-multiple row
counts retain the 32-row path. Both paths convert BF16 weights and F32
activations or output gradients into transient FP16 threadgroup tiles, perform
half 8x8 simdgroup MMA, and accumulate into F32. The routes have independent
emergency rollback controls:

- `TERMITE_METAL_DISABLE_BF16_SIMDGROUP_MM=1` disables the forward route; and
- `TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_MM=1` disables the
  input-gradient route.

`TERMITE_METAL_DISABLE_BF16_SIMDGROUP_M64=1` and
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64=1` disable only the 64-row
forward and backward specializations, respectively, and fall back to the
proven 32-row simdgroup route.

The older 16x32 tiled routes remain available behind their existing rollback
controls. Disabling both simdgroup routes with the final binary restores the
previous promoted loss `7.611953259`, gradient norm `3.143428326`, and
approximately `1.370 s` synchronized frame.

On the identical sequence-128, accumulation-1, rank-16/alpha-32 `peft-qv`
workload, the final guarded 64-row binary's strict 20-step diagnostic measured
`0.568763/0.568748 s` median/mean (`225.06` input tokens/s). This is `1.235x`
faster than the preceding 32-row simdgroup binary (`0.702349 s`), `2.495x`
faster than the 16x32-tiled binary (`1.418835 s`), and `3.815x` faster than the
original `2.169801 s` baseline. Process peak physical footprint stayed
effectively flat at `1,774,864,856` bytes. The measured window recorded 16,384
bytes of page-ins, zero page-outs, and zero swap; the nonzero page-in count still
fails the release lane's zero-paging gate. The exact executable SHA-256 is
`18ceaa0a0025b3e728118584488c7772d1bc4f8a658cc9fda923da4e688c2c6f`.
The diagnostic artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-simdgroup-m64-final-diagnostic-v1.json`
with SHA-256
`06cf51a0128119d263326c0313fbc13be866189befcc97d5fc58efe662c7683f`.

This route deliberately trades exact BF16/F32 arithmetic identity for the
simdgroup's FP16 tile inputs. Against the preceding tiled route, one-step loss
moved from `7.611953259` to `7.611551762` and gradient norm from `3.143428326`
to `3.143764734`. Across all 100 adapter tensors, the resulting one-step adapter
had maximum absolute delta `0.000399718`, relative L2 delta `0.001254`, and
cosine `0.999999214`; the update vector itself had relative L2 delta `4.09%`
and cosine `0.999165`. This is small model-state drift, but it is not exact
numerical parity. The 64-row scheduling change itself adds no further drift:
at both sequence lengths 128 and 512, its one-step adapter checkpoint was byte
identical to the 32-row rollback. A five-update sequence-128 run also produced
an identical final adapter (`SHA-256
95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`),
identical loss/gradient history, held-out loss `4.839618`, one Metal optimizer
update, 50 runtime LoRA regions, and zero fallback per step.

MLX-LM remains `2.159x` faster on the sequence-128 cell, down from `2.667x`
before the 64-row pass and `5.387x` before the first simdgroup pass, while its
process peak is `6.479x` larger. On the exact 512-token prepared workload, the
final Antfly binary measured `1.566196/1.565846 s` median/mean (`326.98` input
tokens/s), a `1.306x` improvement over the 32-row mean of `2.045645 s`, with a
`5,987,846,664`-byte process peak. Matched MLX-LM measured `0.990706 s`
(`516.80` input tokens/s) and a `15,999,361,256`-byte peak. MLX is therefore
`1.581x` faster at 512 tokens but uses `2.672x` the process footprint. The final
Antfly sample recorded 3,784,704 bytes of page-ins, 114,688 bytes of page-outs,
and zero swap; it still fails the zero-paging gate. The Antfly and MLX artifacts
are respectively
`/private/tmp/antfly-gemma4-zig-e2b-seq512-simdgroup-m64-final-diagnostic-v1.json`
(SHA-256
`a23a5eefe5ecc48bdd53a63dd732d51730920f21d66be9132bc3640e398359ff`)
and `/private/tmp/antfly-gemma4-mlx-e2b-seq512-diagnostic-v1.json` (SHA-256
`d7d6a7518193963e7bfaf83ffaf289dd8af176f49791e16a1231dd80aa75513c`).

The sparse causal-loss graph now batches up to eight supervised rows through
each tied vocabulary projection. The former one-row graph reread Gemma4 E2B's
approximately 805 MiB BF16 embedding table once per target token. Eight rows
match the existing Metal backward-input tile height, reduce those frozen-weight
scans from eight to one on the locked workload, and add only an 8 MiB F32
logits tensor. `TERMITE_GEMMA4_DISABLE_BATCHED_SPARSE_LOSS=1` restores the
one-row graph. `TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS=<1..64>` supplies a
bounded diagnostic override; invalid values retain the default of eight.

In synchronized no-frame profiling, batching reduced the eight
`1x262144 * 262144x1536` vocabulary input-gradient products from `102.590 ms`
to one `8x262144 * 262144x1536` product at `22.836 ms`. The matching forward
projection fell from `29.239 ms` across eight products to `20.182 ms` in one.
Total training `dot_general` time fell from `596.780 ms` to `515.718 ms`, and
the compiled command count fell from `3,852` to `3,449`, with no fallback.

The final same-binary sequence-128 diagnostic measured
`0.470076/0.470017 s` median/mean (`272.33` input tokens/s), a further `1.210x`
improvement over the 64-row result and `4.616x` over the original baseline.
Peak physical footprint was `1,808,025,952` bytes, 33,161,096 bytes above the
one-row M64 sample. MLX-LM is now `1.784x` faster while using `6.359x` the
process footprint. The sample recorded 3,047,424 bytes of page-ins, no
page-outs, and no swap. Its artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`63d16c86dd9aaeb89699e1b38632a4a6d97edab0ae47e9e4b4c09037e3623052`.

At sequence 512, the same binary measured `1.466413/1.466469 s`
median/mean (`349.14` input tokens/s), a further `1.068x` improvement over M64.
Peak physical footprint was `6,023,989,768` bytes. MLX-LM is `1.480x` faster
while using `2.656x` the process footprint. This sample recorded 8,404,992
bytes of page-ins, 16,384 bytes of page-outs, and no swap. Its artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-sparse-loss-chunk8-diagnostic-v1.json`
at SHA-256
`1035fd111fc79e5d2103d02ba074f11531cacf5d03f1f65127e91eeee1c243e2`.
The diagnostic executable SHA-256 is
`c265e78692dd29da4bfa198e0e0e00e50412d769d7519dd2abf2ceca2cc28d9a`.

The graph change adds no observed optimizer-state drift. For the locked
eight-target example, one- and five-update adapters are byte-identical to the
one-row M64 graph; the five-update artifact remains SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`
with identical loss and gradient history and held-out loss `4.839618`. A
separate fifteen-target update differed by one F32 loss ULP
(`6.822133064` versus `6.822132587`), had an identical gradient norm, and
produced the same byte-identical adapter (SHA-256
`629a459767d7d826cc047b6cb6584e47021b1c046fa1e727da88ec19d20e71a0`).

The batched graph exposed one remaining fixed vocabulary cost: the frozen BF16
input-gradient product `8x262144 * 262144x1536 -> 8x1536`. Metal now admits a
narrow exact-arithmetic M8/N32/K64 specialization for that product. It uses 128
threads, F32 inputs/accumulators/output, the precise Metal library, and the same
ascending K accumulation order as the generic tile. Admission requires exactly
eight rows, `in_dim >= 128`, `out_dim >= 65536`, `in_dim % 32 == 0`, and
`out_dim % 64 == 0`; all other shapes retain the generic path.
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SMALL_ROWS=1` is the same-binary rollback.

Two repeated no-frame profiles reduced this kernel from `23.7375 ms` mean to
`20.2905 ms`, a `14.5%` local improvement and `3.447 ms` saved. The trimmed
unified binary then passed the locked same-binary 20-step diagnostic at both
sequence lengths. At 128 tokens, enabled measured `0.466987/0.466848 s`
median/mean (`274.18` input tokens/s) against `0.470053/0.470101 s`
(`272.28` tokens/s) with only this route disabled: `3.253 ms` saved, or
`0.697%`. At 512 tokens, enabled measured `1.464261/1.464651 s`
(`349.57` tokens/s) against `1.471423/1.472216 s` (`347.78` tokens/s):
`7.564 ms` saved, or `0.516%`. The enabled process peaks were
`1,808,107,968` and `6,024,006,080` bytes respectively. Against the retained
MLX-LM cells, MLX remains `1.774x` faster at sequence 128 and `1.478x` faster
at sequence 512, while using `6.356x` and `2.656x` the process footprint.

The sequence-128 enabled/rollback artifacts are
`/private/tmp/antfly-gemma4-zig-e2b-seq128-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`6cb67bd0cf13f5c3f7560fb453072d94ab8359c095e11b1c31c3090558d11cd5`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq128-vocab-backward-n32k64-disabled-diagnostic-v1.json`
(SHA-256
`352a76aa177edcc27d029affe7dbf5d66a5fa6fe2a9d32381d6b97c7d24d86e3`).
The sequence-512 pair is
`/private/tmp/antfly-gemma4-zig-e2b-seq512-vocab-backward-n32k64-enabled-diagnostic-v1.json`
(SHA-256
`2d9fe10409584f3c0f39b85c3854e2f6888a6994cdc363a3c36148ee2dada57f`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq512-vocab-backward-n32k64-disabled-diagnostic-v1.json`
(SHA-256
`fafd5fd7069df1317483919e8d9f6dfec16b83f4895015ea603365d92d2253d3`).
All four bind executable SHA-256
`55796f5424996465b5b0a3aab16fe40bfbdd1a1780d5aa76acba68a543b3ebd7`.

The specialization is training-state exact. The trimmed binary reproduced
one-step loss `7.611551761627197`, gradient norm `3.1437647342681885`, and
adapter SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.
After five updates it reproduced adapter SHA-256
`95401988d8d98af88f2608dcb2bcd80937341adecfbd50f5ab20db296b35b384`
and the accepted loss/gradient history, with zero fallback. Current-source
required-device ReleaseFast `test-gemma4-finetune` passed 201 tests with two
optional skips, and inference-edition `antfly-main-test` passed its unified CLI
test with all ten build steps successful. The sequence-512 measurements still
incurred page-ins and page-outs, and each A/B arm is one diagnostic sample, so
this result remains optimization evidence rather than a release-campaign PASS.

### Integrated hardening and final v8 diagnostic (2026-08-12)

The final integrated pass hardened two paths that only appeared under a full
multi-step run. Gemma4's fused RMSNorm VJP is now selected only for Metal, so
native execution remains an independent decomposed oracle. The Metal VJP
requires its activation and output gradient to already be device resident. A
frozen norm weight is served from a prepared runtime slot, while a trainable
norm weight must be directly device resident; neither case may silently upload
a host operand during backward. Recreated graph constants now carry a stable
content identity through device residency and output cloning. The runtime keys
frozen RMSNorm slots by that identity instead of a transient device address,
preventing semantically identical constants from consuming all 512 dynamic
slots across a 25-step benchmark.

The packed-weight path also stopped unconditionally locking the prefetch queue.
Synthetic stores and intentionally synchronous callers do not initialize that
queue, so the old lock could sleep forever before inspecting an already
available Q4_0 tensor. Both affected load paths now lock only when
`prefetch_initialized` is true. The isolated packed-Q4 graph regression passed
in 300 ms, and the final real-device ReleaseFast aggregate completed with 201
tests passed, two optional skips, and zero failures. The seven offline Python
Gemma4 modules completed 125 tests with zero failures; all changed Zig sources
passed `zig fmt --check`, and `git diff --check` was clean.

The exact inference-edition v8 CLI has executable SHA-256
`d8e4819493d1d24b9aaba9548e87907aba9a8f3dea949374d5fb7ddf44ed4400`.
It completed every cold, warmup, first-steady, and measured optimizer window in
both strict diagnostic cells. The comparison uses the retained, matching MLX
cell rather than a new unmatched run:

| Sequence | Antfly median / mean | MLX median / mean | MLX speed advantage | Antfly / MLX peak footprint |
| --- | ---: | ---: | ---: | ---: |
| 128 | `0.467937 / 0.468015 s` | `0.263110 / 0.263169 s` | `1.778x` | `1,809,451,384 / 11,492,007,952` bytes |
| 512 | `1.464869 / 1.465439 s` | `0.992220 / 0.990706 s` | `1.479x` | `6,025,595,400 / 15,999,361,256` bytes |

MLX therefore used `6.351x` Antfly's process footprint at sequence 128 and
`2.655x` at sequence 512. The Antfly sequence-128 measured window recorded
720,896 bytes of page-ins, no page-outs, and zero swap. Sequence 512 recorded
360,448 bytes of page-ins, 294,912 bytes of page-outs, and zero swap, so it
still fails the release lane's zero-paging gate. These are diagnostic samples
from an intentionally dirty checkout and are never admissible as release
evidence. The artifacts are:

- `/private/tmp/antfly-gemma4-zig-e2b-seq128-integrated-v8-final-v1.json`
  (`sha256:c81cf70ad66d5fdedf6d990327366959abd494c4927b8f5057929c8a6f7eb845`);
- `/private/tmp/antfly-gemma4-zig-e2b-seq512-integrated-v8-final-v1.json`
  (`sha256:9a4108ce17118088b6dbda69b71982953b3a8972d6c85f90acdbe7c6ec4b426d`);
- `/private/tmp/antfly-gemma4-mlx-e2b-seq128-diagnostic-v1.json`; and
- `/private/tmp/antfly-gemma4-mlx-e2b-seq512-diagnostic-v1.json`.

This pass also added guarded, separately attributable Gemma4 gate/up kernels.
The forward kernel reuses one activation tile across both BF16 projections;
the backward kernel keeps two independent F32 accumulation streams and writes
their sum directly. The most recent focused backward microbenchmark printed
`6.482 ms` for two qualified products plus the ordinary add and `6.074 ms` for
the fused route. Both routes remain opt-in and require exactly 64-compatible
rows. Their forward and backward counters were zero in both final E2E cells
because the real graph presents 128 rows. Telemetry now proves that zero-hit
fact instead of allowing an enabled-but-unused optimization to receive credit.
The next MLP kernel must target the observed 128-row graph shape and earn an
end-to-end win before promotion.

### Rejected MLP storage and seq128 fusion candidates (2026-08-13)

The follow-up pass first tested F16 mirrors of the frozen BF16 MLP weights so
MPSGraph could own the dense products. A full forward-only mirror reduced the
locked sequence-128 mean from the `0.468884 s` control to `0.458693 s`
(`2.17%`) but raised process peak physical footprint from `1,809,778,992` to
`4,957,474,680` bytes and failed the one-step adapter parity gate. A
backward-only mirror reached `0.441966 s` (`5.74%` faster) at a
`4,956,721,016`-byte peak. Its aggregate adapter comparison looked close, but
43 of 100 target tensors failed the required per-tensor numerical gate; the
worst relative L2 delta was `0.0519005` with cosine `0.998653`. Both mirror
routes and their model-sized duplicate storage were removed. The diagnostic
artifacts are:

- `/private/tmp/antfly-gemma4-zig-e2b-seq128-f16-mps-mlp-v15-v1.json`
  (`sha256:0e31d90f82c45d684246e5de517227f6fa373de98fdd7f57813d469a0510b881`);
  and
- `/private/tmp/antfly-gemma4-zig-e2b-seq128-f16-mps-backward-v16-v1.json`
  (`sha256:4da3b394dbe4a247166cecfca049719314d9093f4d94e113b1fd22a36bfe2089`).

The graph dump was then corrected to skip the smaller 2,933-node loss-only
evaluation graph and inspect the 6,693-node training/autodiff graph. The new
debug-only `TERMITE_DUMP_GRAPH_MIN_NODES` threshold composes with
`TERMITE_DUMP_GRAPH_NODES` for that purpose. It showed the production layer-34
gate/up input gradients as nodes `3367` and `3368`, both `[128,1536]`, followed
by add node `3369`. This explained why the rows-64 experimental matcher had
correctly reported zero calls.

Two bit-exact rows-128 simdgroup-M64 fusion variants were evaluated rather
than inferred from the local kernel timing. Both produced the control adapter
SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`,
executed exactly 35 fused calls per optimizer step, and reduced graph commands
from 3,449 to 3,414. Neither reduced the 1,253 compute encoders, however, and
both lost the strict 20-step end-to-end gate:

| seq128 route | Mean / median | GPU mean | Versus `0.467500 s` control |
| --- | ---: | ---: | ---: |
| 32 KiB two-product threadgroup tile | `0.575830 / 0.576104 s` | `0.523477 s` | `23.17%` slower |
| 16 KiB tile plus exact destination add | `0.513459 / 0.513238 s` | `0.460887 s` | `9.83%` slower |

The 16 KiB kernel was locally bit-exact and measured `6.571 ms` versus
`7.618 ms` for two isolated products plus add, demonstrating why a
microbenchmark is insufficient here: combining the two products reduced GPU
scheduling parallelism across the full training frame. The rows-128 matcher,
kernel, and runtime admission were removed. The original rows-64 experiment
remains opt-in and production seq128 training remains on the faster independent
M64 products plus add. The rejected end-to-end artifacts are
`/private/tmp/antfly-gemma4-zig-e2b-seq128-bf16-gateup-pairsum-v20-v1.json`
(`sha256:58748405b58b5e60866df1ff036f4f5c573c4b8d4442c36f4f0e1e5af8d93d2e`)
and
`/private/tmp/antfly-gemma4-zig-e2b-seq128-bf16-gateup-pairsum-v21-v1.json`
(`sha256:846f6638bf21e61e982b17942cf87dca908e196e8971b4858e08d216856dd499`).

### Qualified coalesced BF16 M64 backward loader (2026-08-13)

Same-binary rollback attribution identified the standalone frozen-linear input
gradient as the next useful target. Disabling only the forward M64 route raised
the sequence-128 mean from `0.467500 s` to `0.492410 s` (`5.33%`), while
disabling only backward M64 raised it to `0.573880 s` (`22.76%`). The result
also explained the failed pair fusion above: the independent backward products
are valuable, but their per-product transposed-weight loader still had room to
improve.

The legacy M64 backward kernel assigned one output column to four adjacent
threads, making the global BF16 weight reads stride by `in_dim`. The promoted
kernel instead assigns each thread an eight-column contiguous weight segment
and swizzles that segment into the exact existing threadgroup tile. Its tile
shape, output-gradient loader, three simdgroup barriers per K fragment, MMA
order, F32 accumulators, and stores are unchanged. This follows the relevant
design lesson in the pinned MLX 0.31.2 wheel's Apple Steel GEMM sources: use a
transpose-aware block loader, while retaining the synchronization required by
the shared-memory MMA implementation. It does not embed or call MLX.

The real-Metal regression uses the production Gemma 4 E2B dimensions
`rows=128`, `in_dim=1536`, and `out_dim=6144` and requires bit-identical output
against the legacy M64 kernel. It passed. Independent strict CLI one-step runs
also produced byte-identical adapters at SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.
Two 20-step sequence-128 A/B pairs, with the execution order reversed in the
second pair, measured:

| Pair | Legacy mean | Coalesced mean | Legacy GPU mean | Coalesced GPU mean |
| --- | ---: | ---: | ---: | ---: |
| control then candidate | `0.468076 s` | `0.445565 s` | `0.420842 s` | `0.397391 s` |
| candidate then control | `0.468155 s` | `0.445322 s` | `0.420647 s` | `0.397410 s` |

Across those two pairs, mean wall time improved `4.84%` and mean Metal frame
time improved `5.55%`. The route is therefore default-on;
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_COALESCED=1` restores the
qualified legacy loader without disabling the broader M64 specialization.

The final default-on binary then passed strict paired cells at both locked
sequence lengths:

| Sequence | Default median / mean | Rollback median / mean | Default / rollback GPU mean | Wall improvement | Default / MLX mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.444689 / 0.444468 s` | `0.468757 / 0.468896 s` | `0.397388 / 0.420521 s` | `5.21%` | `1.689x` |
| 512 | `1.363282 / 1.366533 s` | `1.467746 / 1.467627 s` | `1.259479 / 1.363107 s` | `6.89%` | `1.379x` |

Peak physical footprint was effectively unchanged: `1,809,992,080` bytes at
sequence 128 and `6,025,562,536` bytes at sequence 512. The retained MLX cells
use `11,492,007,952` and `15,999,361,256` bytes respectively, so MLX remains
faster while using `6.35x` and `2.65x` Antfly's process footprint. The final
binary SHA-256 is
`fd3a023742ca50fbe0bcc2bffcd0f896ed6858f0271512d3446aab50e472f318`.
The default and rollback artifacts are:

- sequence 128:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-coalesced-default-v33-v1.json`
  (`sha256:ddc98ccb6b47aa3c9d370266511745d2e3dab1be8b159aa714b432458a3f533a`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-coalesced-rollback-v34-v1.json`
  (`sha256:22f715eb11fc74c39ebedc05d6575a711e2936084dc17572b9008590815c86dd`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-coalesced-default-v35-v1.json`
  (`sha256:bd6f8140d2871323d373e07a3cafd38b4b5c5de8cbded0db24006d8a62f27fee`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-coalesced-rollback-v36-v1.json`
  (`sha256:2c25e65395a785d0168b32351c650aa0dc618381008179cf5b29f866cb05eb3f`).

These remain diagnostic artifacts from a dirty checkout. The sequence-128
default cell recorded 16,384 bytes of page-ins, and the sequence-512 cell
recorded page-ins and page-outs, so neither is a release-campaign PASS.

### Qualified packed BF16 M64 backward loads (2026-08-13)

The coalesced loader above still issued eight scalar BF16 loads per thread.
Replacing them with ordinary `ushort4` loads looked faster in microbenchmarks,
but a strict one-step CLI run rejected that implementation: the adapter changed
from the qualified
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`
to
`d7da901c07ce5d6d182e020865043b33ad85fb317e3065ae2a9d3d915c359c85`,
and the mean gradient norm changed from `3.143764734` to `2.776347160`.

The fault was an alignment-contract violation rather than an MMA or scheduling
error. Persistent Safetensors BF16 weights are borrowed with
`newBufferWithBytesNoCopy`; they are guaranteed to be dtype-aligned, not
eight- or sixteen-byte aligned. The real E2B checkpoint's projection payloads
start at address modulo eight equal to two. An ordinary Metal `ushort4 *`
therefore asserted alignment the storage does not provide. The promoted kernel
uses `packed_ushort4` for BF16 weights and `packed_float4` for gradient views,
then retains the existing vector conversion, threadgroup layout, barriers, MMA
order, F32 accumulation, and stores.

The real-Metal regression now covers every one of the 11 E2B text projection
geometries that can reach the 128-row M64 route, a nonzero F32 gradient-buffer
view offset, and a borrowed BF16 weight base deliberately offset by two bytes.
It is bit-identical to the scalar-coalesced route. Final user-facing CLI runs
through `antfly inference finetune train gemma4-lora` also produced
byte-identical adapters for the default packed and scalar rollback paths, both
at the qualified adapter SHA-256 above. Training loss (`7.611551762`), gradient
norm (`3.143764734`), and post-update evaluation loss (`6.111435890`) matched
exactly.

The final same-binary diagnostic pairs measured:

| Sequence | Packed median / mean | Scalar rollback median / mean | Packed / rollback GPU mean | Wall improvement | Packed / MLX mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.428906 / 0.428800 s` | `0.446268 / 0.446523 s` | `0.380754 / 0.397433 s` | `3.97%` | `1.628x` |
| 512 | `1.287999 / 1.288199 s` | `1.367928 / 1.368350 s` | `1.182072 / 1.260595 s` | `5.86%` | `1.300x` |

The packed route is default-on.
`TERMITE_METAL_DISABLE_BF16_BACKWARD_SIMDGROUP_M64_PACKED=1` restores the
qualified scalar-coalesced loader without disabling the broader M64 route.
Peak physical footprint remained effectively unchanged at `1,809,811,856`
bytes for sequence 128 and `6,025,775,528` bytes for sequence 512. The retained
MLX 0.31.2 cells measured `0.263396 s` and `0.990706 s` with peaks of
`11,497,430,816` and `15,999,361,256` bytes, so MLX is still faster while using
`6.35x` and `2.66x` Antfly's process footprint. The root CLI binary SHA-256 is
`8fd482628ddbcb3a2b0190980cd1a940cc41215cc7d66aed6220d49822756545`.

The final default and rollback artifacts are:

- sequence 128:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-default-final-v62-v1.json`
  (`sha256:60465e62d32f02c2079bd99464bb35a76a0d2fea7caad34718a343d39997810a`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-rollback-final-v63-v1.json`
  (`sha256:a1e935426585825c0de8dfe9d72c5331f3cad3bd66c5e70e58587131b40d3c78`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-packed-default-final-v64-v1.json`
  (`sha256:cbbaf6e39cb0fe0add1f8874f33d9bb4c6dee0133a160c663a71030fd74f9260`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-packed-rollback-final-v65-v1.json`
  (`sha256:a7ea29475b87df77adba3f25649d38d74c823f3566cd241b422dea17a40241f3`).

These are bounded diagnostic artifacts from a dirty checkout. The default
sequence-128 cell recorded 65,536 bytes of page-ins, and the sequence-512 cell
recorded page-ins and page-outs, so this kernel promotion is qualified but the
overall branch still does not have a locked release-campaign PASS.

### Qualified packed BF16 M64 forward loads (2026-08-13)

The packed-backward default was reprofiled before changing another kernel. A
diagnostic no-frame run captured the complete 6,693-node training graph and
then failed closed at the expected strict post-step evaluation boundary; its
absolute per-command times are synchronization-distorted and are used only for
ranking. The training graph executed 1,057 dot/GEMM commands in `468.977 ms`
of `961.819 ms` total execution. The largest individual family was the
40-call `128x1536 * 1536x12288` projection at `62.627 ms`; the corresponding
reverse projection and the 6,144-wide MLP projections were also among the top
shapes. Dense dispatch tracing confirmed that eligible forward projections
still used the scalar-load `bf16_simdgroup_m64` kernel while input gradients
used the packed sibling. The retained profile is
`/private/tmp/antfly-gemma4-packed-default-profile-v68.log`
(`sha256:bee9039ef26a98b771a6e18df23b08064f786df3a330071cbbfd24c4d5768331`).

The promoted forward sibling replaces each eight-value scalar BF16 and F32
global-load sequence with two `packed_ushort4` and `packed_float4` loads. It
does not change the bias seed, threadgroup indices, simdgroup loads, barriers,
MMA order, F32 accumulators, or output stores. Packed types are required here:
the persistent Safetensors view is allowed to be only two-byte aligned, and a
compiled F32 activation view is allowed to begin at a four-byte offset.

The real-Metal regression compares the packed route bit-for-bit with the
retained scalar kernel for all 11 E2B projection geometries at 128 rows. It
also pairs a borrowed BF16 base deliberately offset by two bytes with an F32
device view offset by four bytes and requires both outputs to remain
device-resident. A strict user-facing CLI step selected the packed forward
kernel for every eligible projection, completed one Metal optimizer update
with zero graph/interpreter fallback, and matched the scalar rollback exactly:
before/train/after loss was
`6.822133064 / 7.611551762 / 6.111435890`, mean gradient norm was
`3.143764734`, and both adapter files had SHA-256
`bbb4b27ecabbf68864e7f8b6199fe9fe31fb424f913dec223e4bc096a3a15491`.

Two sequence-128 pairs reversed execution order, and the cleaned final binary
repeated the packed-then-scalar order. A sequence-512 pair exercised the
longer-context pressure regime on the same kernel build:

| Sequence / order | Packed median / mean | Scalar median / mean | Packed / scalar GPU mean | Wall improvement |
| --- | ---: | ---: | ---: | ---: |
| 128, packed then scalar | `0.409764 / 0.409210 s` | `0.428644 / 0.429399 s` | `0.363127 / 0.380234 s` | `4.70%` |
| 128, scalar then packed | `0.409909 / 0.409623 s` | `0.427586 / 0.427572 s` | `0.362807 / 0.380513 s` | `4.20%` |
| 128, cleaned final binary | `0.409577 / 0.409291 s` | `0.428065 / 0.427702 s` | `0.363286 / 0.380291 s` | `4.30%` |
| 512, packed then scalar | `1.215124 / 1.209975 s` | `1.285686 / 1.286470 s` | `1.112794 / 1.181448 s` | `5.95%` |

Across all three sequence-128 pairs, wall mean improved `4.40%` and GPU-frame
mean improved `4.54%`. Peak physical footprint remained effectively unchanged:
the largest packed sequence-128 observation was `1,809,893,800` bytes and the
sequence-512 observation was `6,025,628,048` bytes. Against the retained MLX
0.31.2 means, the remaining wall-time gaps are now `1.554x` at sequence 128 and
`1.221x` at sequence 512. This improves on the immediately preceding
packed-backward default by `4.53%` and `6.07%`, respectively.

The packed forward route is default-on.
`TERMITE_METAL_DISABLE_BF16_FORWARD_SIMDGROUP_M64_PACKED=1` restores the exact
scalar M64 loader without disabling the broader M64 specialization. The
attested inference-edition benchmark binary SHA-256 is
`a11b433d319aba536db70f508874d609c43bf5c97eb89d72b17ebf84cd93f546`.
After removing the temporary no-frame profiling relaxation, the cleaned
attested ReleaseFast inference binary SHA-256 is
`2bf227a22a918351315e938543858d455daef4a6fe94a04a151acadeb1f4ec0c`.
That binary repeated the strict one-step loss, gradient, update, fallback, and
adapter-SHA result above. The real-device ReleaseFast `test-gemma4-finetune`
aggregate also exited successfully with the all-shape packed-forward regression
included.
The diagnostic artifacts are:

- sequence 128, packed then scalar:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-candidate-v71-v1.json`
  (`sha256:b5f6ac1e2738f351f09fa504c8f8c2c6cae792075b8a329edd45deac4788175e`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-scalar-v72-v1.json`
  (`sha256:73dbe10127b3c2d15bf8462b70efc12e6a582bde139baea81c9c739a2868ac8e`);
- sequence 128, scalar then packed:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-scalar-v73-v1.json`
  (`sha256:85d971f50ed634b2dd76956927e691ac2127f263b153d8aaf4cb4462b0c223d6`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-candidate-v74-v1.json`
  (`sha256:9935222ee07f3dfedb3b9ca4c68be0324e0e3cac583a851cd4d0c613c043583e`);
  and
- sequence 512:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-forward-packed-candidate-v75-v1.json`
  (`sha256:a99b1d97dd08e28a49d8e84b8350c7e2c7d596f993342a8cb06e8225a261c505`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-forward-packed-scalar-v76-v1.json`
  (`sha256:de73e784986e1c8fc112ad6dedacab07922ac4806ea7d49cff33bbbbe437eaec`);
  and
- sequence 128, cleaned final binary:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-packed-final-v78-v1.json`
  (`sha256:e040e9470507e73e17b82f4b5f401c3faedd28dadb29a19d951bbceef2a42647`)
  and
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-forward-scalar-final-v79-v1.json`
  (`sha256:83d179970ee5e5eb6b3b12d0308e5e94325f023388d2e2378ac0710fa445f58e`).

### Rejected zero-bias seed specialization and packed-route audit (2026-08-13)

A follow-up review confirmed that the promoted packed forward kernel changes
only its alignment-safe global loads; the bias seed, shared-memory layout,
barriers, simdgroup MMA order, accumulators, and stores remain identical to the
scalar M64 control. The regression now also observes a runtime
`bf16_forward_simdgroup_m64_packed_calls` counter. With the packed rollback set,
the counter must not move; with the packed route admitted, every tested
projection must increment it exactly once. This closes the prior gap where
bitwise output equality alone did not independently prove route execution.

Gemma 4's frozen dense projections are represented by positive-zero bias
buffers, so a bounded candidate initialized the packed kernel's simdgroup
accumulators directly to zero and skipped the 1,024-value threadgroup bias
tile plus its two barriers. Admission required a slot-preparation scan proving
every bias value was bitwise positive zero, and
`TERMITE_METAL_DISABLE_BF16_FORWARD_SIMDGROUP_M64_ZERO_BIAS=1` restored the
ordinary packed bias-tile path. The same binary
(`sha256:0916896d1ca025389a897fa3a69c9dde9a148128b08101fe792ece59f247cf3e`)
ran default, rollback, rollback, default at both sequence lengths:

| Sequence | Zero-bias wall mean | Rollback wall mean | Zero-bias GPU mean | Rollback GPU mean | Wall effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | `0.409631 s` | `0.410623 s` | `0.363361 s` | `0.363319 s` | `0.24%` faster |
| 512 | `1.216293 s` | `1.215180 s` | `1.114177 s` | `1.113450 s` | `0.09%` slower |

The sequence-128 wall result was not corroborated by GPU time, and both
sequence-512 measures regressed. The candidate was therefore removed rather
than promoted. The retained change is only the packed-route counter and its
rollback/admission assertions. The post-revert required-device ReleaseFast
Gemma 4 aggregate passed 204 tests with two optional skips and zero failures.

The final reviewed inference-edition binary is
`/private/tmp/antfly-gemma4-packed-reviewed-root-v92/bin/antfly`
(`sha256:98acf9abd3820917832e1352324ab880809d44e1dcaf84553163f0c4950c322b`).
Its 20-step sequence-128 confirmation measured `0.410223 s` wall mean,
`0.410580 s` wall median, and `0.363887 s` synchronized Metal-frame mean. It
retained the same workload and initial-adapter semantic digests, used no
diagnostic overrides, and reported 3,449 command dispatches, 1,253 compute
encoders, and 1,650 planned barriers. The retained artifact is
`/private/tmp/antfly-gemma4-zig-e2b-seq128-packed-reviewed-final-v93.json`
(`sha256:16e787ab12c36ea317a145d35d6a5cff5222663b67f69f97302a02b6de62951a`).

The rejected A/B artifacts are:

- sequence 128 default A / rollback A / rollback B / default B:
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-default-v84-a.json`
  (`sha256:472d2ca16e8ea0aa680d50a2fddfd1370c5fac5c08cf85191d2ff475d0c00bb9`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-rollback-v85-a.json`
  (`sha256:10af4d858f4478ac81aa22999002fb445a67aa2a0f7b119c8e1e67a84bd77e6f`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-rollback-v86-b.json`
  (`sha256:9e898b69a55d58432db13e649445333a65720103bbbfac9dedf2fa5298913847`),
  and `/private/tmp/antfly-gemma4-zig-e2b-seq128-zero-bias-default-v87-b.json`
  (`sha256:8c44a9b1b2c9afb65d65ef50ec5e0d94c6c06cd5cc8515089b9dadd52585f37f`);
- sequence 512 default A / rollback A / rollback B / default B:
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-default-v88-a.json`
  (`sha256:c5a55ba5b901b70a6b332d5cc57ea17fe742feab2b201653aada83d0086cf998`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-rollback-v89-a.json`
  (`sha256:c6d43f567cbff8ca2316673642a66efd01d894122d53fccdc5c4d1838e20a004`),
  `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-rollback-v90-b.json`
  (`sha256:0b2fb829d6f1a96f41b3cddfebee2da19afb5aab57bdd88ab5527dc1839bbb41`),
  and `/private/tmp/antfly-gemma4-zig-e2b-seq512-zero-bias-default-v91-b.json`
  (`sha256:e70c67ebeff3eafcb9d94fb1d578e04d9de4dc4057edd1d320cb6d7b92ae1d04`).

These are still bounded diagnostics from a dirty checkout. Every cell recorded
page-ins, and both sequence-512 cells recorded page-outs; the packed sequence-
512 cell also observed a `-21%` memory-pressure availability delta. The kernel
promotion is qualified, but none of these artifacts is a release-campaign PASS.

The current `TERMITE_ENABLE_FUSED_LINEAR_CROSS_ENTROPY=1` experiment is also
not a cut cross-entropy implementation: it remains default-off and still
materializes the complete `[rows, vocabulary]` logits and gradient-logits
tensors. It must not be used to claim Unsloth-style or Apple-style memory
behavior. The production design is a new forward/backward op pair that saves
only per-row log-sum-exp, valid-count, and mean-loss state, tiles vocabulary in
both directions, and accumulates `d_hidden` and optional `d_weight` without a
global logits tensor. On the current 8-row by 262,144-vocabulary cell, the
existing route owns about 8 MiB of forward logits and about 16 MiB of
logits/gradient working storage during backward. A 64 KiB vocabulary tile plus
the small saved state would cut those CE intermediates by more than 99%.

That design follows the useful part of the industry architecture without
copying CUDA performance claims onto Metal. The pinned
[MLX-LM trainer](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py)
still obtains model logits before its ordinary cross-entropy call. Apple's
[Cut Cross-Entropy implementation](https://github.com/apple/ml-cross-entropy)
and [research paper](https://arxiv.org/abs/2411.09009) establish the
logits-free vocabulary-tiled algorithm, while
[Unsloth's CCE integration](https://github.com/unslothai/cut-cross-entropy)
builds on that work. Antfly should implement the algorithm as first-class Zig
graph and Metal operations, qualify BF16 first, and then add Q4_0, Q4_K, and
Q6_K frozen-output-weight readers. It should not hide it behind the current
misnamed experimental switch.

With both M64 directions using qualified alignment-safe packed loaders, the
normal synchronized frame is now reprofiled. Its 3,449 command dispatches,
1,253 compute encoders, and 1,650 planned barriers make graph/dispatch fusion
the next performance target; another tile-seed micro-optimization is unlikely
to close the MLX gap. The next pass should first attribute and coalesce the
1,057 dot/GEMM, 1,084 elementwise, 399 transpose, and 309 activation-backward
dispatches without reviving the already rejected independent-product fusions.
True cut cross-entropy remains the parallel memory and longer-context target.
Promotion also requires a run-scoped, manifest-bound numerical-kernel policy
fingerprint so environment changes cannot alter approximate kernel admission
mid-run. Direct GGUF QLoRA, E4B, multi-seed convergence, interrupted resume
equivalence, and a clean alternating five-pair release campaign remain open
gates.

These ratios are bounded diagnostic A/B evidence from a dirty Antfly checkout,
not the locked alternating multi-repeat performance campaign, a zero-paging
PASS, or a numerical-parity claim.

Two other candidates were rejected rather than left on by default. MPS BF16
matrix multiplication hit Apple's mixed-dtype assertion because MPS admits the
current F32-output contract only for an F16 right-hand matrix. A shared-row
BF16 reduction remained numerically exact at 128 rows but regressed the training
frame by about `5.2x` because it collapsed row parallelism.

### Measured Metal optimization pass (2026-08-11)

The same one-example, sequence-64, rank-4 Q/V E2B case was used to optimize
the training graph without changing its model, prepared inputs, seed, or
hyperparameters. A custom RMSNorm VJP now replaces the decomposed primitive
backward. For LoRA, norm weights are frozen, so the VJP also returns only
`d_input`; the Metal kernel skips the dead `d_weight` reduction and its private
inverse-RMS buffer. The full `d_input + d_weight` path remains covered for
callers that train norm weights.

| Training-step measure | Pre-fusion | Current RMSNorm VJP | Change |
| --- | ---: | ---: | ---: |
| Logical graph commands | 9,434 | 4,362 | -53.8% |
| Elementwise commands | 6,630 | 1,558 | -76.5% |
| Reduce commands | 649 | 168 | -74.1% |
| Interpreter fallbacks | 0 | 0 | unchanged |
| Warm peak physical footprint | 1,119,488,640 bytes | 1,011,255,864 bytes | -9.7% |

The input-only specialization removes another 239 training graph nodes and
executions (`8,346 -> 8,107` and `6,476 -> 6,237`). In a same-minute profiled
A/B, Metal frame submission moved from `5,779.021 ms` to `5,748.113 ms`
(-30.9 ms, about 0.54%). Whole-process warm time was tied within noise
(`18.25 s` full-gradient control versus `18.24 s` input-only), so this is a
dispatch/memory improvement, not evidence of a material end-to-end speedup.
Loss before/train/after remained `6.822132587 / 9.059597015 / 6.209827900`,
mean gradient norm remained `2.428374052`, and strict Metal fallback counters
remained zero. Native finite differences and native-versus-Metal tests over
hidden widths 3 through 768 bound the worst observed RMSNorm-backward absolute
delta at `7.1525574e-7`.

Two LoRA-backward candidates were not promoted. The full three-gradient region
needs a recomputed checkpoint activation that is unavailable when the region
is scheduled, so canonical direct-B matching caused strict interpreter
fallback and was removed. The safer low-rank `d_after_a + dB` region executed
50 times with zero fallback and reduced commands from 4,362 to 4,262, but its
paired warm time was neutral/slightly worse (`18.07 s` versus `18.01 s`); it
therefore remains opt-in. The dominant measured training work is now the 1,184
dot/GEMM commands and the attention/linear backward path.

The existing raw-linear training-region route was also evaluated rather than
enabled by default. It matched 297 regions with zero fallback and produced a
byte-identical adapter, but alternating warm runs regressed from `17.69 s` and
`17.68 s` without the route to `17.91 s` and `18.19 s` with it. The route stays
disabled for training; region-hit count alone is not a performance result.

Attention batched-dot VJPs now choose operand order and contracting axes that
emit `dQ`, `dK`, `dP`, and `dV` in their final physical layouts. The Metal
batched-dot kernel accepts either matrix axis as the contraction, eliminating
the input/output transpose materializations around those contractions. On the
same E2B case this reduced training commands from 4,362 to 4,224 (-3.2%),
transpose commands from 537 to 399 (-25.7%), graph nodes from 8,107 to 7,969,
and executed nodes from 6,237 to 6,099. A profiled Metal frame moved from
`5,804.462 ms` to `5,778.212 ms` (-0.45%). Three alternating warm whole-process
runs measured `17.93/18.03/18.08 s` before and `17.98/18.00/18.05 s` after
(medians `18.03 s` and `18.00 s`), which is effectively tied. The adapter was
byte-identical (`sha256:df62cf44593f4305e6496470afcd5f5a0e7cbe8174095cf0a6248bf6bfe94168`),
loss and gradient metrics were unchanged, and fallback and host-output counts
remained zero. This is a real graph/dispatch reduction, not a material
end-to-end speedup claim.

Attention-sized batched dot products now use one cached
`MPSMatrixMultiplication` across every batch matrix, including all four
left/right contracting-axis layouts. Large 2D projections already used the
same Apple MPS primitive, so this targets the previously scalar-per-output
batched path rather than duplicating that optimized projection route. The
default admission gate is deliberately bounded to `batch_count >= 2`,
`m >= 128`, `n >= 8`, and `k >= 32`; sequence-length-64 alternating runs were
tied at `17.88 s` median with and without MPS. At sequence length 256, the
profiled training frame improved from `5,603.455 ms` to `5,369.956 ms`
(-4.17%), while peak RSS changed by only 49,152 bytes. Three alternating
whole-process runs improved from `16.95 s` to `16.81 s` median (-0.83%). The
scalar and MPS paths produced byte-identical adapters
(`sha256:7cfa386de0e2d76ec0f7c01a40bed375fdbc83e556611568d853b5059b6cc9fd`),
identical training loss and gradient norm, one optimizer update, and zero
fallback or host-output events. `TERMITE_METAL_DISABLE_DOT_GENERAL_BATCHED_MPS`
provides the paired control, while
`TERMITE_METAL_REQUIRE_DOT_GENERAL_BATCHED_MPS` makes route admission a hard
test assertion. This is a bounded one-example optimization result, not MLX-LM
parity evidence.

These are Antfly old-versus-new measurements, not an MLX-LM comparison. The
locked alternating multi-repeat same-Mac campaign described above is still
required before claiming MLX performance parity.

This is useful execution evidence, not an oracle or quality result: it is one
example and one update, the before/after rows differ in supervised-token count,
and no HF/native/Metal gradient or update trace comparison was produced.
Broad multi-step convergence, deterministic overfit, bounded peak memory,
repeated adapter reload/generation, and required CI remain open. Exact
epoch-boundary process-kill/resume has now passed for bounded real E2B and E4B
Metal jobs; those narrow gates are no longer roadmap-only claims.

Local results are not substitutes for required CI:

```sh
zig build test-gemma4-finetune
zig build test-gemma-graph
python3 scripts/gemma4/compare_gemma4_lora_hf_zig.py validate-lock
python3 -m unittest discover -v -s scripts -p 'test_*gemma4*py'
```

The macOS job is a synthetic real-GPU gate, not an E2B/E4B scale gate. It must
run successfully in CI and be made required in repository branch protection;
workflow source alone is not evidence that either has happened. The Python
contract tests validate the lock, schemas, deterministic fixtures, producer
roles, evidence ledgers, numerical checks, runner coordination, and benchmark
pairing on synthetic data; they do not execute the locked HF or MLX-LM model
campaigns.

The official E2B QAT Q4_0 GGUF (`sha256:fa401b55...dec6634`) now passes the
explicit direct-GGUF Metal lane for one real UltraChat row at sequence length
64, rank 2, Q/V targets. Two fresh production-default processes produced exact
gradient fingerprints and the same adapter
`sha256:881e74dd...27016e`. The complete two-epoch process-kill/resume gate then
produced adapter `sha256:e38d0dd8...d99d961`, with uninterrupted and resumed
epoch-2 loss `3.6382982731`, gradient norm `0.3051027656`, one Metal optimizer
step, and zero fallback. The qualification report is
`/private/tmp/antfly-gemma4-e2b-gguf-qlora-resume-acceptance-20260819-v3/qualification_report.json`.

The important negative result is retained: wrapping the quantized tied-head
projection, CE, and input-gradient projection inside one builder-level fused
loss node was nondeterministic even after internal encoder fences. The exact
decomposed graph was repeatable, so direct GGUF selects
`linear_cross_entropy_mode = "decomposed-gguf"` automatically and binds that
mode into the checkpoint fingerprint. This closes optimizer and resume
correctness for the exercised E2B shape; it does not yet close peak-memory,
MLX/HF numerical parity, multi-seed task quality, E4B GGUF, or deployment
generation gates. The public `qlora-sft` recipe therefore remains a typed
error.

### Qualified gate/up backward graph fusion (2026-08-13)

Gemma 4 LoRA product training now recognizes the same-layer gate/up
backward-input pair
`d_gate @ W_gate + d_up @ W_up` as one prepared runtime region. The Metal
entry point consumes both cached BF16 weight slots, produces the sum directly,
and keeps the non-anchor dot elided after the fused add has been satisfied.
This removes one dot dispatch and one add dispatch per transformer layer. The
runtime-region plan owns the match, so steady-state execution does not rescan
the graph.

The fusion is enabled by default only for the qualified 64-, 128-, and
512-row lanes. `TERMITE_METAL_DISABLE_GEMMA4_BF16_GATE_UP_BACKWARD_INPUT_SUM=1`
is the same-binary production rollback. Explicit
`TERMITE_METAL_ENABLE_GEMMA4_BF16_GATE_UP_BACKWARD_INPUT_SUM=1` retains an
experimental surface for other kernel-supported 64-row multiples. Matcher
misses and runtime misses retain the two-dot-plus-add path.

A CPU sample caught an initially hidden host regression: generic Gemma
residency telemetry recursively classified the fused backward add 35 times per
step. The fusion now records its statically proven `residual_add` category
directly. Planned-region time on the profiled sequence-128 step fell from about
10.25 ms to 6.57 ms, slightly below the 6.88 ms rollback profile.

On the locked diagnostic E2B Q/V rank-16 workload, the final default-on binary
and its disable override produced the sequence-128 cells below. The sequence-512
cells used the immediately preceding attested qualification binary with the
fusion explicitly enabled; its region, prepared-slot kernel, liveness, and
telemetry paths are identical, and only the subsequent default-admission policy
changed.

| Cell | Fused mean | Rollback mean | Wall improvement | Structural evidence |
| --- | ---: | ---: | ---: | --- |
| seq128, accumulation 1, repetition 1 | 413.357 ms | 414.585 ms | 1.227 ms (0.30%) | 35 fusions; 3,344 -> 3,274 commands |
| seq128, accumulation 1, repetition 2 | 414.189 ms | 415.637 ms | 1.448 ms (0.35%) | reverse order; same counts |
| seq128, accumulation 4 | 1,687.872 ms | 1,695.919 ms | 8.048 ms (0.47%) | 140 fusions; four plan-cache hits |
| seq512, accumulation 1, repetition 1 | 1,204.384 ms | 1,210.696 ms | 6.311 ms (0.52%) | 35 fusions; 3.87 ms GPU reduction |
| seq512, accumulation 1, repetition 2 | 1,204.381 ms | 1,215.415 ms | 11.034 ms (0.91%) | reverse order; 3.62 ms GPU reduction |

The expanded Metal parity test covers both Gemma FFN widths at rows 128 and
512. Rows 64 are bit-exact. At rows 512, the worst observed maximum absolute
delta was `4.005e-5` and worst relative L2 was `2.80e-6`. A full one-step
adapter comparison against rollback had identical loss
(`7.611551761627197`), gradient-norm delta `0.000263`, state maximum absolute
delta `0.0019914`, relative L2 `0.0032884`, and cosine `0.9999946`. Those state
metrics pass the checked-in native/Metal BF16 profile (`0.02`, `0.005`, and
`0.9999`). The packed accumulation order is not byte-identical to rollback, so
promotion relies on the numerical oracle rather than an adapter SHA claim.

The locked seq2048 diagnostic exceeded its 1,800-second subprocess watchdog
while still actively executing and published no sample. It is therefore not a
pass or a regression. Rows 2048 remain outside the default-qualified set; the
legacy path remains active there until the long-sequence harness can measure a
complete paired cell within a practical watchdog. These local artifacts come
from a dirty diagnostic checkout and are performance/qualification evidence,
not release evidence or an MLX parity claim.

### Matched E2B DPO optimization benchmark (2026-08-17)

The locked one-token DPO case now recognizes the exact structural condition in
which chosen and rejected responses share one prompt and each contain one
completion token. Policy and frozen-reference scoring project the final prompt
row once for both candidates. Their opposing logprob gradients are then encoded
as two weighted targets on that same causal row, so one compiled backward
microbatch replaces the previous chosen and rejected sequence microbatches.
General multi-token or non-shared-prompt data remains on the existing sequence
path.

Five fresh-process runs of the final strict-Metal binary produced identical
25-update loss trajectories and byte-identical trained adapters. The table uses
the median of the five per-run measured medians for the optimized Antfly cell;
the prior Antfly and pinned MLX-LM cells retain their matched 20-update measured
medians.

| Implementation | Measured seconds / update | Peak process footprint | Result |
| --- | ---: | ---: | --- |
| Prior Antfly pair path | `1.846122` | `3,659,354,480` bytes | baseline |
| Optimized Antfly paired-row path | `0.760557` | `3,656,355,992` bytes | `2.427x` faster than prior Antfly |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `0.520192` | `12,782,264,168` bytes | Antfly is `1.462x` duration / `46.21%` slower |

Antfly uses `71.40%` less comparable peak process footprint than MLX-LM, and
MLX-LM uses `3.496x` Antfly's footprint. All five Antfly adapters have SHA-256
`fd2060d835e2c35a052dec593f02091b8df5b7aaa61cd0c891bddccb72dfd430`;
all 100 adapter tensors changed. Adapter movement was L2 `1.907393` versus
MLX-LM's `1.824274`. Across cold, first, warmup, and measured updates, the
optimized trajectory differs from the prior Antfly route by mean absolute
`0.000434` and maximum absolute `0.002761`; final loss differs by
`1.78e-6`. Against MLX-LM those values are `0.009541`, `0.040490`, and
`2.93e-5` respectively.

An important correctness guard remains explicit. Reuse-enabled repetitions
occasionally exposed a stale compiled update even after eager scoring was
isolated. Gemma4 DPO therefore disables the Metal in-frame private-buffer pool
for the complete preference run while retaining compiled execution and all
qualified fused kernels. The report records
`metal_buffer_reuse_mode = "disabled-for-dpo-run"`. This restores exact
five-run determinism without giving back the speedup, but it postpones the
lower-memory `~2.57 GB` experimental result until cross-scope alias dependencies
can be proven globally.

Initial same-graph policy/reference error is exactly zero. The Antfly-to-MLX
base reference chosen/rejected logprob deltas remain `0.213154` and `0.037575`,
with a `0.175579` preference-margin delta, so this is matched training-behavior
evidence rather than exact cross-framework forward-logit parity. It is also a
bounded E2B sequence-128 one-token diagnostic, not multi-token, E4B, or release
campaign qualification. The retained comparison artifact is
`/private/tmp/antfly-gemma4-e2b-dpo-final-mlx-comparison-20260817-v1.json`.

### Real UltraFeedback multi-token DPO parity (2026-08-17)

The general sequence path now also has a provenance-locked real-data result.
`scripts/gemma4/materialize_gemma4_dpo_hf_parity.py` consumes
`HuggingFaceH4/ultrafeedback_binarized` `test_prefs` at revision
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85`. The source Parquet SHA-256 is
`e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389`.
The materializer verifies the exact local Gemma4 tokenizer contract, forbids
truncation, requires distinct chosen/rejected token IDs and a score margin, and
selects one example from each of five total-token buckets through sequence 512.
A second-admitted-per-bucket mode produced a disjoint five-example holdout.

The public recipe CLI trained the five-example set for five epochs through 25
complete DPO updates from the same rank-16, alpha-32 Q/V seed adapter. The
pinned MLX `0.31.2` / MLX-LM `0.31.3` runner consumed the exact materialized
token IDs and update order. Both adapters were then loaded and scored by
`scripts/gemma4/evaluate_gemma4_dpo_adapters_mlx.py` under one MLX oracle, eliminating
base-runtime differences from the result comparison.

| Shared MLX evaluation | Compiled-score Antfly adapter | MLX-trained adapter |
| --- | ---: | ---: |
| Training preference accuracy | `1.0` | `1.0` |
| Training mean DPO loss | `0.00163453` | `0.000813062` |
| Training mean reward margin | `12.01240` | `12.88266` |
| Disjoint holdout preference accuracy | `0.6` | `0.6` |
| Disjoint holdout mean DPO loss | `0.652640` | `0.664016` |
| Disjoint holdout mean reward margin | `0.429506` | `0.634040` |

Holdout preference-decision agreement is `1.0` across all five rows. The
Antfly/MLX holdout mean-loss ratio is `0.98287`, with absolute accuracy delta
zero. This is a bounded behavioral-parity pass on real, disjoint preferences;
five training and five holdout examples are not a broad quality or convergence
campaign.

The original general sequence path materialized large vocabulary-shaped work
for both scoring and the signed DPO backward objective. The production path now
uses four reusable supervised-row buckets (`64`, `128`, `256`, and `512`), a
compact uniform `[row, token, scale, ...]` target contract, and frozen tied-head
fused linear cross-entropy. Policy and frozen-reference scoring return a single
device-reduced sequence logprob; backward returns the exact signed summed-logp
gradient without owning a global logits tensor. Chosen and rejected backward
passes remain separate batch-1 microbatches because a batch-2 experiment
regressed both throughput and memory. The live policy scorer now executes those
two loss-only forwards through the trainer's cached pruned compiled session.
Frozen-reference precomputation keeps explicit zero-LoRA bindings on its
separate route, so scoring never swaps or mutates optimizer-owned weights.

The exact final ReleaseFast binary SHA-256 is
`ca5cb5b3bbbbf7816a88379d950bfcde57e9b6ac5505010788d6e24b167ce257`.
Its fresh fixed-work result is:

| Implementation | Measured median / mean | Reference precompute | Lifetime peak physical footprint |
| --- | ---: | ---: | ---: |
| Prior Antfly sequence path | `41.4782 / 45.5115 s` | `23.9595 s` | `32,555,638,368` bytes |
| Intermediate Antfly fused path | `6.97423 / 7.48183 s` | `13.1064 s` | `14,775,393,072` bytes |
| Final Antfly compiled-score path | `6.55229 / 6.97276 s` | `11.5299 s` | `15,034,801,184` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.05975 / 2.06250 s` | `5.31961 s` | `19,168,812,736` bytes |

The compiled scorer reduces the intermediate fused path's median by `6.05%`
and mean by `6.80%`. Relative to the original sequence path, the retained path
is `6.330x` faster by median and `6.527x` faster by mean. Against pinned
MLX-LM, Antfly is now `3.181x` the median duration and `3.381x` the mean
duration, while using `21.57%` less comparable peak process footprint. The
conservative footprint is the larger of two repetitions; it is `1.76%` above
the intermediate fused run and `53.82%` below the original path. MLX reference
precompute remains `2.167x` faster.

Two independent compiled-score repetitions published byte-identical adapter
payload SHA-256
`e49063871260b65baf09bdbf277ce210afe195db4d739d7cc5bde9b2f7acf337`
and identical 25-loss trajectories. Their medians were `6.552286 s` and
`6.459342 s`; the table retains the slower result. Against the intermediate
fused adapter, the final update has cosine `0.996205`, norm ratio `1.101293`,
relative L2 difference `0.136449`, and `97.445%` nonzero sign agreement. The
compiled reduction order therefore is not a byte-equivalent multi-step route,
even though the first update is byte-identical. The shared MLX oracle above is
the acceptance evidence: training and holdout decisions remain identical, and
the final Antfly holdout mean loss is closer to MLX than the intermediate
adapter's (`0.652640` versus `0.612447`, with MLX at `0.664016`).

The next high-value target is the compiled backward graph, not another scoring
special case. A sequence-512 frame trace attributes the chosen and rejected
backward passes to `1,311` and `1,316` Metal compute commands. Each carries
`1,336` planned scopes and `1,060` planned barriers, for `1.784 s` and
`1.129 s` of GPU time in the traced cold update. MLX compiles scoring, loss,
backward, and optimizer as one step; Antfly still submits the two signed
sequence backwards independently. The next pass should coarsen safe graph
regions and remove redundant cross-region barriers while retaining batch-1
memory behavior, then reprofile before attempting another shape-specific
kernel.

Retained artifacts:

- training case semantic SHA-256
  `9f788b5cb6090c66f257ec6d35cea02c8e0d5170a8e0b0f1d9076e4879af2ade`;
- holdout case semantic SHA-256
  `afea7cdb02f4b4ed838a905872d82d5ad678cd31d60b911e5196529ba1c5c5c1`;
- prior Antfly report and adapter under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-run-v1`;
- exact final optimized Antfly report and adapter under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-compiled-policy-final-v1`;
- deterministic repetition under
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-compiled-policy-repeat-v1`;
- matched MLX report
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/mlx-run-v3-cross-eval.json`; and
- optimized training and disjoint shared-oracle results
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/compiled-policy-{training,heldout}-mlx-cross-eval-v1.json`.

### Exact detached-gradient DPO and graph coalescing follow-up (2026-08-17)

The general multi-token Metal route no longer needs a score-only policy forward
followed by a second backward execution for each side of the preference pair.
For logical gradient accumulation one and a non-recursive adapter, chosen and
rejected each execute exactly once with a coefficient of one, producing the raw
summed sequence logprob and its gradient. The chosen device gradients are
detached before the rejected branch runs; both device sets are then combined
with the exact host DPO coefficients and flushed through AdamW once. Reports
identify this route as
`policy_scoring_mode = "backward-loss-reuse-device-detached"` and
`training_microbatch_mode = "chosen-rejected-raw-gradients-device-combined"`.
`ANTFLY_GEMMA4_DPO_DETACHED_GRADIENTS=0` restores the proven two-backward
rollback. Recursive LoRA, explicit pair-graph experiments, and logical gradient
accumulation above one conservatively retain their older route.

The one-step detached/rollback adapter comparison measured gradient cosine
`0.9997975674`, norm ratio `0.9999953928`, relative L2 delta `0.020121`, and
`99.99048%` sign agreement. Independent detached repetitions were byte-identical
at adapter SHA-256
`37c5a0672306a4a81c8aec3c6bdcd75adc486ab9d0c02adc122859e71843b959`.
The complete 25-update final adapter is SHA-256
`5bf72497c5c57208d0903a4878ae2d9490d6a58407d8756ff7fa31acd6a58ae5`;
the retained run before the final graph-fusion pass has the same bytes and the
same 25-loss trajectory.

Two command-level changes survived same-binary rollback gates:

- The prepared static execution plan now groups qualified rank-16 LoRA-A MPS
  dots by exact parameter family and shape. Fifteen Q/V pairs per backward
  branch coalesce, reducing branch compute-command counts from `1,311/1,316`
  to `1,296/1,301`. Pair-backward GPU time improved `0.54%` in the controlled
  A/B. `TERMITE_METAL_DISABLE_GROUPED_LORA_A_R16=1` is the rollback.
- A precise-library add3 kernel fuses 18 single-use, same-shape F32 add chains
  per branch while preserving `(a+b)+c` versus `c+(a+b)` order. It rejects
  broadcasts, multi-use producers, two deferred children, and chains deeper
  than one level. Planned scopes fell from `1,321` to `1,303` and barriers from
  `1,060` to `1,042` per branch. Across two alternating pairs, pair-backward GPU
  time averaged `2,906.220 ms` versus `2,913.438 ms` with
  `TERMITE_METAL_DISABLE_ADD3_FUSION=1`, a small but repeatable `0.248%` win.

The custom batched-MPS rollback and a packed K=16 expansion GEMM were rejected.
Disabling batched MPS reduced encoder count but raised chosen/rejected GPU time
to roughly `2.704/2.052 s`; the fallback kernels are compute-bound. The packed
`512x16 * 16x1536` candidate passed exact output parity but was slower in both
same-binary pairs, so its kernel, admission, and tests were removed. These
results make dispatch-count reduction alone an insufficient promotion rule.

The final fixed-work result from the production-default ReleaseFast path is:

| Implementation | Measured median / mean | Reference precompute | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Prior compiled-score Antfly path | `6.55229 / 6.97276 s` | `11.5299 s` | `15,034,801,184` bytes |
| Final detached-gradient Antfly path | `5.05956 / 5.24470 s` | `11.5553 s` | `14,744,034,000` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.05975 / 2.06250 s` | `5.31961 s` | `19,146,055,360` bytes |

This table is retained as the detached-gradient attribution checkpoint; the
compact-attention and allocator-lifetime sections below supersede it for the
current shipping result.

The exact hardened CLI for the final Antfly row is SHA-256
`371634ba358c528100b2752f27f2779e8a9f186927e2548bc29d04fd0c2a186a`.
Detached gradients reduce the prior Antfly median by `22.78%` and mean by
`24.78%`. Antfly remains `2.456x` the MLX median duration and `2.543x` the MLX
mean duration, while its peak physical footprint is `22.99%` lower. The final
Antfly report records loss `0.1946493`, mean reward margin `4.299646`, accuracy
`0.84`, exact initial same-base policy/reference logprobs, 25 optimizer steps,
and 50 physical branch microbatches.

The new adapter also passed the shared-runtime behavioral gate rather than
inheriting the older adapter's result. Under the pinned MLX oracle, Antfly and
MLX both reached training accuracy `1.0` and holdout accuracy `0.6`, with
preference-decision agreement `1.0` on all five training and five disjoint
holdout rows. Final Antfly/MLX mean DPO loss was
`0.00187977/0.000813062` on training and `0.612271/0.664016` on holdout. The
bounded holdout mean-loss ratio is `0.922073`; this remains five-row diagnostic
evidence, not a broad convergence claim.

Evidence:

- final report and adapter:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-detached-final-full-v2`;
- training shared-oracle result:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/detached-final-training-mlx-cross-eval-v1.json`;
- disjoint holdout shared-oracle result:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/detached-final-heldout-mlx-cross-eval-v1.json`; and
- matched MLX training report:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/mlx-run-v3-cross-eval.json`.

The remaining gap was not in policy score reuse or a single uncovered rank-16
GEMM. Both backward branches still owned about 1,300 compute commands and
roughly `1.78/1.12 s` of GPU work. A retained trace attributed enough of that
work to the decomposed attention VJP to justify the compact-GQA campaign below.

### Compact GQA attention VJP and E2B-only promotion (2026-08-18)

The retained production trace showed `140` masked-softmax calls, `70`
softmax-backward calls, and the surrounding attention contractions in one DPO
update. The accepted implementation packs token-major Q/K/V/dO once into a
reused head-major private buffer, evaluates QK-transpose, dO-V-transpose,
dScore-transpose-Q, dScore-K, and P-transpose-dO as five batched MPS matrix
multiplications, applies causal/sliding softmax VJP in one Metal kernel, then
unpacks dQ and reduces the expanded per-query-head dK/dV back into compact
shared-KV tensors. The workspace reuses Q/K/V regions only after their last
encoded read. Sequence lengths below `128` retain the scalar compact route.

The final shipping policy is deliberately architecture-specific. The compact
VJP is default-on only for the exact qualified E2B topology: hidden size
`1536`, `35` layers, `8Q/1KV`, local/global head dimensions `256/512`,
intermediate size `6144`, sliding window/pattern `512/5`, `20` shared-KV tail
layers, and PLE width `256`. `TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION=1`
restores the decomposed production graph. The explicit
`TERMITE_METAL_ENABLE_GEMMA_GQA_ATTENTION_FUSION=1` variable remains a research
override for non-qualified shapes, and
`TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_MPS_VJP=1` selects the slower scalar
compact diagnostic route. `TERMITE_METAL_TRACE_GEMMA_GQA_ATTENTION_FUSION=1`
reports the selected compact kernel geometry.

The final same-binary, fixed-25 E2B comparison used the real fingerprinted
UltraFeedback slice, rank-16/alpha-32 Q/V LoRA, sequence length `512`, and the
locked cold/first/three-warmup/20-measured protocol:

| Implementation | Measured median / mean | Reference precompute | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Final binary, decomposed rollback | `4.286381 / 4.422427 s` | `8.692538 s` | `15,206,603,736` bytes |
| Final binary, production E2B default | `3.868544 / 3.958108 s` | `12.087822 s` | `14,622,085,632` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.059747 / 2.062500 s` | `5.319610 s` | `19,146,055,360` bytes |

The production default won `18/20` paired measured updates and reduced median
and mean update time by `9.75%` and `10.50%`, respectively. It saved
`584,518,104` bytes (`3.84%`) versus rollback. Relative to pinned MLX, the
conservative final-binary result is `1.878x/1.919x` the median/mean duration
while using `23.63%` less peak process footprint. Two earlier optimized runs,
before the memory-pressure-heavy E4B campaign, measured `3.696854` and
`3.696983 s` medians; all three runs produced the same complete loss trajectory
and byte-identical adapter payload
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`.
This table isolates the compact-attention change before the allocator-lifetime
promotion below.

Against the decomposed 25-step adapter, the accepted E2B update has cosine
`0.99998331`, norm ratio `1.00004211`, relative L2 difference `0.00577777`, and
`99.9234%` sign agreement. Initial policy/reference logprobs remain exact. In
the shared pinned MLX evaluator, the default adapter and MLX both reached
training accuracy `1.0` and disjoint holdout accuracy `0.6`, with decision
agreement `1.0` on all ten rows. Their training mean DPO losses were
`0.00191398/0.000813062`; holdout losses were `0.631058/0.664016`. This is a
bounded real-data behavior gate, not a broad convergence claim.

E4B was tested and explicitly rejected from the default. A sequence-512
one-step run exercised both `8Q/2KV` local `head_dim=256` and global
`head_dim=512` MPS routes. Its update remained close to rollback (cosine
`0.999815`, norm ratio `0.999998`, relative L2 `0.01924`), but the matched
25-step trajectory diverged materially:

| E4B implementation | Median / mean | Final loss / reward / accuracy | Peak physical footprint |
| --- | ---: | ---: | ---: |
| Decomposed rollback | `32.5795 / 39.7838 s` | `0.11109 / 182.904 / 0.88` | `21,505,631,064` bytes |
| Compact MPS research override | `31.9462 / 36.8085 s` | `0.24120 / 3.351 / 0.80` | `21,170,132,992` bytes |

The final E4B update cosine fell to `0.807816`, relative L2 difference rose to
`0.661817`, and sign agreement fell to `82.00%`. The small speed and memory wins
therefore do not satisfy the quality/parity gate. The final no-override E4B CLI
smoke emitted no compact trace and reproduced the decomposed one-step adapter
byte-for-byte. E4B needs a numerically tighter compact reduction or a different
bounded-memory attention strategy before reconsideration.

### Planned-encoder DPO buffer reuse and E2B promotion (2026-08-18)

The remaining coarse DPO guard disabled all same-frame private-buffer reuse.
The lifetime audit found two independent contracts. First, the executor must
not free a pre-materialized parameter or constant after an ahead-of-order fused
consumer while its own graph position is still pending. Second, a private
buffer may enter the same-frame pool only while a barrier-capable planned
encoder owns its last use; reuse forces an encoder barrier. Releases outside
that scope are quarantined until command-buffer completion. Focused tests cover
ahead-of-order graph liveness, same-encoder reuse fences, unscoped quarantine,
completion publication, live escaped tensors, and cancelled frames.

The scoped policy now configures in-frame and completion-fenced reuse as one
drained-boundary transaction and restores both on every exit path. It defaults
on only for the exact qualified E2B topology. Set
`ANTFLY_GEMMA4_DPO_IN_FRAME_BUFFER_REUSE=0` for the production rollback. Setting
it to `1` remains a research override for other shapes; E4B and unknown
topologies otherwise stay fail-closed.

Three alternating reuse-enabled fixed-25 runs produced byte-identical adapters
and exactly identical complete loss arrays around a same-binary reuse-disabled
run. Their medians were `2.494348`, `2.540593`, and `2.545429 s/update`. The
final no-override shipping-binary run measured `2.536096 / 2.825541 s`
median/mean and `8,758,956,424` bytes peak footprint:

| E2B buffer policy | Measured median / mean | Peak physical footprint |
| --- | ---: | ---: |
| Completion-fenced cache; in-frame reuse disabled | `4.089158 / 4.145905 s` | `14,621,594,016` bytes |
| Planned-encoder in-frame reuse plus completion-fenced cache | `2.536096 / 2.825541 s` | `8,758,956,424` bytes |
| Pinned MLX `0.31.2` / MLX-LM `0.31.3` | `2.059747 / 2.062500 s` | `19,146,055,360` bytes |

Against the same-hot-path rollback, the promoted default is `37.98%` faster by
median, `31.85%` faster by mean, and uses `40.10%` less peak memory. It is now
`1.231x` the MLX median duration and `1.370x` the MLX mean duration while using
`54.25%` less peak process footprint. All candidate, rollback, and final-default
runs produced adapter SHA-256
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`
with exact initial policy/reference parity and identical final loss, reward,
and accuracy. A final E4B one-step smoke reported in-frame reuse disabled and
reproduced its decomposed adapter SHA-256
`75ff9a99fc02b74c797a0da4f74aaf63082fe97aba53fa15e7e0a3dcf6ac62b5`.

The final ReleaseFast CLI is SHA-256
`b3af4a952966e354346745b05cb00bcf3c4111e3a1c9d8fb339389fd8e94b4f6`.
The focused Metal gate selected `244` tests: `242` passed and two optional
real-artifact tests skipped. The portable non-Metal gate selected `229` tests:
`212` passed and `17` Metal-dependent tests skipped. The direct native
comparison covers scalar sequence `8` and the MPS admission boundary at
sequence `128`, including full causal and sliding-window forward/backward
paths. A final fixed-25 CLI run on that exact binary selected the E2B
planned-encoder reuse default and reproduced the qualified adapter payload.

Evidence:

- final production-default E2B report and adapter:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-reuse-default-final-v1`;
- three reuse-enabled qualification runs:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-in-frame-reuse-v{1,2,3}`;
- same-hot-path reuse-disabled qualification run:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-reuse-rollback-same-binary-v1`;
- final E4B fail-closed policy smoke:
  `/private/tmp/antfly-gemma4-e4b-dpo-one-step-reuse-policy-final-v1`;
- pre-reuse compact-GQA production report:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-gqa-default-final-v7`;
- E2B training and disjoint shared-MLX evaluation:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/gqa-mps-v4-{training,heldout}-mlx-cross-eval.json`;
- E4B compact and rollback fixed-25 reports:
  `/private/tmp/antfly-gemma4-e4b-dpo-real-fixed25-{gqa-mps,baseline}-20260818-v1`; and
- E4B final update comparison:
  `/private/tmp/antfly-gemma4-e4b-gqa-mps-vs-baseline-fixed25-20260818-v1.json`.

### Production in-place RMSNorm-backward residual-add fusion (2026-08-18)

A node-level sequence-512 E2B trace narrowed the next normalization candidate
to `69` adjacent `fused_rms_norm_backward -> reshape -> add` chains in each
chosen/rejected backward, or `138` opportunities per DPO optimizer update. The
initial candidate extended the frozen-weight RMSNorm-backward Metal kernel with
an exact-operand-order residual add, rejected graph outputs and protected
scalar tails, and retained the ordinary path for trainable norm weights. Its
standalone output allocation was bit-exact but failed the production memory
gate: the fixed-25 footprint rose by `429,539,304` bytes (`4.90%`) for a
noise-level `0.08%/0.12%` median/mean improvement, so that implementation was
kept default-off.

The production follow-up writes into the residual's existing Metal range. It
requires the residual to be an internal add, its exact final graph use to be
the matched add, and its storage to be executor-owned. Runtime alias checking
compares exact byte ranges rather than rejecting disjoint retained views of a
shared allocation. Overlapping live aliases, graph outputs, protected values,
borrowed runtime inputs, trainable norm weights, shape changes, and unavailable
planned-encoder barriers all fail closed. The write emits an explicit
buffer-scope barrier before reusing the range. Chained matches may consume a
preceding fusion's already-materialized add output even though that producer
is marked skipped for traversal bookkeeping.

The in-place one-step trace executed all `138` chosen/rejected chains without
a decline, warning, or fallback. Its adapter remained byte-identical to the
prior final profile at SHA-256
`99c41810b809f6ee7ed3f96fc5a884a00d9e5d0296a8040bbdb03a73998c7732`.
The same-binary fixed-25 in-place and rollback runs also had identical complete
loss arrays and the canonical final adapter SHA-256
`1a17373b0092b60cc24593b265cb073fb46188d4fc5dca97d1a3fd610ffe301a`:

| RMS backward path | Measured median / mean | Reference precompute | Lifetime peak footprint | Completion-cache peak | Metal allocation requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| In-place final-use residual | `2.518822 / 2.817389 s` | `8.773315 s` | `8,556,564,872` bytes | `5,244,553,216` bytes | `138,760` |
| Same-binary rollback | `2.502889 / 2.813252 s` | `8.796792 s` | `8,760,152,432` bytes | `5,421,238,272` bytes | `145,660` |

Throughput is neutral: the in-place route was `0.64%` slower by median and
`0.15%` slower by mean, while total process wall time was `0.22%` faster. The
memory result is material in the controlled run: lifetime peak fell `203,587,560`
bytes (`2.32%`), completion-cache peak fell `176,685,056` bytes (`3.26%`), and
allocation requests fell by exactly `6,900` (`138` fusions times `50`
microbatches). The route is therefore production-default as a memory and
allocator-pressure optimization, not claimed as a throughput win. Automatic
enablement is restricted to the qualified E2B hidden width (`1536`); E4B
remains default-off pending a multi-step same-binary qualification.
`TERMITE_METAL_DISABLE_RMS_NORM_BACKWARD_RESIDUAL_ADD_FUSION=1` is the rollback
switch and takes precedence; the legacy enable variable remains accepted and
can be set false for an equivalent process-local opt-out or true to force an
explicit research run on an otherwise unqualified width.

The final no-enable-override ReleaseFast CLI is SHA-256
`625a1fde0109587eecf9cd79415e6c306839eab86d53aa654dd3536f0b69480c`.
Its traced E2B one-step smoke executed all `138` in-place fusions with zero
decline, warning, or fallback markers and reproduced the canonical one-step
adapter. The final portable matrix selected `230` tests (`212` passed and `18`
Metal/optional tests skipped); the required-device matrix selected `238`
tests (`236` passed and two optional real-artifact tests skipped).

Evidence:

- in-place one-step fusion trace:
  `/private/tmp/antfly-gemma4-e2b-rms-residual-inplace-one-step-v6-20260818.log`;
- byte-exact one-step run:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-one-step-rms-residual-inplace-v6`;
- timed fixed-25 in-place candidate:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-rms-residual-inplace-candidate-v3`;
- timed same-binary rollback:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-fixed25-rms-residual-inplace-rollback-v2`;
- qualified A/B ReleaseFast binary SHA-256:
  `53e6bb1203ec07e7a0d475d7e73982dbd07bbfe98b93d37c85f42b641838ad5b`;
- final default-on smoke trace:
  `/private/tmp/antfly-gemma4-e2b-rms-residual-inplace-default-one-step-v7-20260818.log`;
- final default-on one-step artifacts:
  `/private/tmp/antfly-gemma4-ultrafeedback-dpo-parity-20260817-v1/antfly-dpo-one-step-rms-residual-inplace-v7-default`.

### Matched E2B GRPO benchmark (2026-08-17)

Gemma4 text GRPO now scores its frozen same-base reference through the exact
compiled policy graph using reusable zero-valued LoRA device bindings. The
first real E2B probe selected ranked tokens `[7001, 711]`, produced prefix-match
rewards `[1, 0]`, and reported zero sampling/rescore and policy/reference error.
The previous eager-reference result had a false nonzero initial KL term; the
corrected cold loss, policy-gradient loss, and KL loss are all exactly zero
while the reward-derived gradient still updates every adapter tensor.

The locked rank-16 Q/V, sequence-128 diagnostic uses one rendered prompt, two
ranked one-token completions, AdamW at `1e-4`, KL coefficient `0.04`, and 25
complete groups split as cold 1, first 1, warmup 3, and measured 20. The
algorithmically matched MLX-LM runner uses the pinned MLX `0.31.2` and MLX-LM
`0.31.3` source revisions. It caches the immutable base prompt distribution
once and evaluates each completion at batch 1. A batch-2 MLX rescore changed
the second initial logprob by `0.124493`, so it was rejected rather than hidden
inside a nominally faster but semantically different baseline.

The optimized Antfly route now runs one shared ranked prompt forward, reuses
the sampled on-policy logprobs, caches exact frozen-reference rows, coalesces a
one-token completion group into one weighted gradient row, and projects only
the selected causal row. Five independent production-default runs produced
the same complete 25-point loss trajectory and byte-identical adapter payload
`a7c98538931e1fb20457343c491c287282b9f83d32b183aa4484cf9c3408178b`.

| Metric | Antfly Zig/Metal | Pinned MLX-LM |
| --- | ---: | ---: |
| Measured median per complete group | 0.773689 s | 0.629479 s |
| Measured mean per complete group | 0.777806 s | 0.629428 s |
| Peak process physical footprint | 1,871,007,776 bytes | 11,767,013,080 bytes |
| Adapter delta L2 | 0.782024 | 0.715478 |

The Antfly value is the median of five run medians; the five medians ranged
from `0.772952 s` to `0.774567 s`. It is `1.229x` the MLX-LM duration, or
`22.91%` slower. This is a `4.772x` speedup over the first matched Antfly
baseline (`3.692312 s`) without changing the locked workload. Antfly uses
`84.10%` less comparable peak process footprint, or MLX-LM uses `6.289x` as
much. The exact reference cache held two rows and served 48 hits; total
reference-scoring time fell from `22.722172 s` to `0.836700 s` for the full
25-group run.

The original in-frame private-buffer pool could recycle a Metal buffer under a
new logical tensor identity without a provable encoder dependency. Repeated
runs exposed exact stale losses from earlier updates. The production allocator
now reuses such a buffer only when an active planned encoder can emit a
buffer-scope barrier; otherwise the buffer remains quarantined until the frame
ends and the allocation is fresh. The eager sampling path also closes and
synchronizes its Metal frame at the logits readback boundary. This combination
preserved the low-memory result and passed five repeated trajectories; merely
staging gradients or adding a readback wait did not.

Cross-framework numerical parity is still not exact. The optimized Antfly and
MLX-LM initial second-token logprobs are `-8.051079` and `-7.940530`, an
absolute delta of `0.110549`; the ranked-margin delta is `0.110499`. Their
25-step loss trajectories have MAE `0.287661`, maximum absolute delta
`0.656068`, and final delta `0.327514`. Antfly's adapter-update L2 is `9.30%`
larger. This qualifies the command path, deterministic Antfly trajectory, and
bounded performance diagnostic; it does not establish broad GRPO quality or
cross-framework result parity.

The next optimization target is general multi-token GRPO: batch active
divergent prefixes by decode step, cache immutable reference rows by exact
prompt/prefix identity, and profile the remaining compiled backward path at
realistic sequence lengths before adding more specialized kernels.

Evidence:

- optimized compact comparison:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-mlx-comparison-v1.json`
- original pre-optimization comparison:
  `/private/tmp/antfly-gemma4-e2b-grpo-mlx-comparison-v1.json`
- Antfly five-run reports:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-20260817-v{1,2,3,4,5}/grpo_report.json`
- Antfly timed run:
  `/private/tmp/antfly-gemma4-e2b-grpo-safe-reuse-20260817-v1.time.txt`
- MLX-LM report:
  `/private/tmp/antfly-gemma4-e2b-grpo-mlx-repo-run-v2.json`
- rebuilt one-step cross-framework-logprob probe:
  `/private/tmp/antfly-gemma4-e2b-grpo-logprob-probe-20260817-v2/grpo_report.json`

The final five-run Antfly result is bound to binary
`sha256:932ab0d54dc0fa9826fc1c624485568a63827a6a13d51ab28018e530f64190e7`.
The working tree was intentionally dirty; no commit or push was performed.

### Real BoolQ GRPO acceptance, MLX parity, and stability boundary (2026-08-20)

`scripts/gemma4/materialize_gemma4_grpo_boolq.py` pins `google/boolq` at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. It verifies the local tokenizer,
forbids rendered-prompt truncation, requires one-token `yes`/`no` targets, and
materializes balanced, source-disjoint 64-row train and validation artifacts.
Their JSONL SHA-256 digests are respectively
`36e2a2759413914466e7670583794e803b81b25d6f487faac1aad971f54fc3d7`
and `03ccbae22059e529e4abcd54977367ac8b64fd9f47feee06bf53ace20d9af7cc`.

ReleaseFast binary
`sha256:9b81aea87ff1e83e30ead95e3247f034d4e302ada63918f34ef720efff6f26a1`
completed the bounded cell through the public
`antfly inference finetune run <recipe.json>` command with no `TERMITE_*`
overrides. The cell uses the first eight pinned train rows, group size eight,
one epoch, learning rate `1e-7`, and all 64 held-out rows. It performed eight
optimizer updates, started with exact sampling/rescore and policy/reference
parity, published a changed adapter, and reported training mean reward
`0.328125` with mean KL `3.59379e-7`. Held-out mean reward was `0.36328125`,
top-ranked mean reward `0.703125`, positive-reward group rate `1.0`, and mean
KL `5.83758e-7`. Those values pass the unchanged floors `0.125`, `0.55`,
`0.75`, and maximum KL `1.0`.

The final run is byte-identical to the accepted 2026-08-19 trajectory. Its
adapter, train reward trace, and evaluation reward trace SHA-256 values are
respectively
`bd49818094089e7203e9844d528909e6c8f725c14e09d122e947990dfd5321b3`,
`990f206f2f58ae0034aa369bc8fa8b08837ac3ed087e8aef0b22a4aacc3b3d27`,
and `71685d49a5e039300567feb83d8fc3e2b74a7e4e8a7f9ebc483f327f509731fb`.
The held-out policy digest is
`sha256:42036e6ed505932978d5794db244917de810440b6eb8099a2dfe26d9a4836cd6`.
The complete final evidence root is
`/private/tmp/antfly-gemma4-e2b-boolq-grpo-direct-gather-acceptance-20260820-v13`.

The earlier intermittent Metal trajectory drift was traced to BF16 mmap
embedding-row staging across multiple runtimes in one optimizer-backed GRPO
process. GRPO now owns a nested process-scoped suppression guard and keeps
ordered planned encoders through the complete trainer lifetime. The remaining
fifth-evaluation-call failure was a separate ABA cache bug: compiled sparse
scoring sent ephemeral `[128,1536]` hidden activations through the generic
embedding-table cache, whose device-handle key could be recycled after the
activation was freed. Tensor-id embedding lookup now selects ordinary dense
device rows with the direct axis-0 Metal gather, bypassing the cache and a
`786,432`-byte activation copy per scorer call. The legacy device-table
preparation fallback also refuses pointer-only hits. A 12-lifetime real-Metal
regression passed without host fallback, and a cache-traced five-prompt run was
byte-identical to the canonical evaluation prefix while exposing only the two
stable mmap-backed model tables. The required-device ReleaseFast focused gate
selected 266 tests, passed 264, and skipped two optional local-model fixtures.

The final public run took `70.81 s`, with `569,163,776` bytes maximum RSS and
`3,465,826,216` bytes peak physical footprint. This is `4.378x` faster than
the previous exact Antfly acceptance and closes the prior `7.411x` MLX gap.
A fresh pinned MLX 0.31.2 / MLX-LM 0.31.3 comparison took `43.0698 s` and
peaked at `11,298,022,880` physical bytes. MLX is therefore `1.644x` faster in
this full bounded campaign, while Antfly uses `69.324%` less peak physical
memory. The comparator again classifies the result as **behavioral parity with
numerical drift**: all behavioral checks pass, including exact baseline mean
reward, `0.980469` baseline candidate recall, and `0.984375` top-1 agreement.
Native MLX held-out mean/top-ranked reward is `0.357422` / `0.6875`, within the
locked bounds. Numerical parity does not pass: adapter-delta cosine is
`0.572289`, relative L2 is `0.014554`, maximum absolute difference is
`1.58776e-6`, and held-out KL differs by `5.94119e-4`. The comparison artifact
is `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-parity-20260820-direct-gather-v3.json`.

This pass is deliberately bounded. Two larger, preserved campaigns exposed a
sharp stability boundary rather than being relabeled as successes: 192 updates
at `1e-5` and 64 updates at `1e-6` both failed the held-out gate, left the
trained adapter unpublished, and reported mean KL `1.62371584e9`. Their roots
are `/private/tmp/antfly-gemma4-e2b-boolq-grpo-acceptance-20260819-v5` and
`/private/tmp/antfly-gemma4-e2b-boolq-grpo-acceptance-20260819-v7`.
That historical acceptance predated the train-time KL budget and adaptive
controller qualified below. Those controls close the missing fail-safe, but a
multi-seed update-count sweep is still required: the bounded pass proves the
real dataset, optimizer, evaluator, reward trace, and publication contract,
not long-horizon convergence. That acceptance was one-token only; the bounded
multi-token E2B/E4B execution and MLX boundaries are qualified separately
below.

### Real E2B/E4B multi-token GRPO sparse-row qualification (2026-08-20)

The production Metal multi-token route now projects only the active causal row at
each ranked decode step and only the completion predictor rows during policy
and frozen-reference rescoring. Sampling and rescoring use the same projection
geometry under one default-on switch; this preserves the fail-closed on-policy
sampling/rescore check. Set
`ANTFLY_GEMMA4_GRPO_SPARSE_MULTI_TOKEN=0` to restore full sequence-vocabulary
projection for both phases. The optimization applies only when
`max_completion_tokens > 1` on Metal; native execution and the
already-qualified coalesced one-token route are unchanged.

This is deliberately not whole-transformer candidate batching. A first
candidate flattened the completion group into a padded transformer batch, but
real E2B Metal checks rejected it: mixed sparse rescoring differed from legacy
sampling by up to `0.002059937`, and padding projected rows to the legacy
sequence width still increased the maximum error to `0.010498047`. The
experimental batch switch remains default-off and retains the strict parity
failure. The promoted path keeps each divergent transformer prefix at batch
one while removing the dominant unused vocabulary rows.

Real pinned BoolQ cells used one train prompt, one source-disjoint evaluation
prompt, group size three, sequence length 128, rank-16 LoRA, learning rate
`1e-7`, and prefix-match reward. Both the train and held-out groups exercised
variable completion lengths `[4, 4, 3]` in the four-token cells.

| Workload | Legacy wall | Sparse wall | Maximum RSS, legacy -> sparse | Peak physical, legacy -> sparse |
| --- | ---: | ---: | ---: | ---: |
| E2B, two-token cap | `22.71 s` | `19.29 s` | `1,189,740,544 -> 568,836,096` bytes | `6,056,481,112 -> 5,517,381,448` bytes |
| E2B, four-token cap | `20.69 s` | `19.59 s` | `1,222,656,000 -> 573,947,904` bytes | `6,034,477,280 -> 5,524,983,624` bytes |
| E4B, four-token cap, median of two runs per mode | `37.14 s` | `34.52 s` | promotion-build same-binary `1,222,197,248 -> 632,438,784` bytes | promotion-build same-binary `7,693,342,384 -> 7,201,461,624` bytes |

The bounded E4B timing is noisy: the pre-promotion pair favored sparse
`31.70 s` to `40.91 s`, while the promotion-build same-binary
default/rollback pair was `37.33 s` to `33.36 s`. The two-run median is a
modest `7.1%` sparse win, not a stable throughput distribution. The memory
result was consistent and is the
stronger promotion signal: promotion-build RSS fell `48.3%` and peak physical
footprint fell `6.4%` versus rollback. E2B improved wall time by `15.1%` at the
two-token cap and `5.3%` at the four-token cap while cutting RSS by more than
half.

Every E2B and E4B cell retained exact-zero sampling/rescore and initial
policy/reference error within its selected geometry. Legacy and sparse modes
produced byte-identical train/evaluation reward traces and byte-identical
trained adapters. The E4B adapter SHA-256 is
`837ee12fad644025600aef1a39e5008d4426aa2281892e3baefd55878e6bb120`;
its train and evaluation trace SHA-256 values are respectively
`cfbc93de6eef294239e417f0d538f937ab12b71f6ba1e8fe4e3869e313277a4a`
and `e7d6d2d21af7db58821dd6fdbfb7502c4731eb54775ecda9f74cf013edcbffe7`.
The two projection geometries are not bit-identical internally: E4B diagnostic
first-token logprobs differ by at most `0.000261069`, but selected tokens,
rewards, loss, optimizer counts, adapter bytes, and replay traces are exact.

The final no-override ReleaseFast Metal CLI is SHA-256
`d866bce982a94e5dc7c5e3c614a928001534b6083c2e3b6daf7a79d549c7ed70`.
Its exact-source E4B run took `37.85 s`, used `625,639,424` bytes maximum
RSS and `7,194,482,016` bytes peak physical footprint, selected the qualified
sparse modes, and reproduced the accepted adapter and trace digests above.
Its focused Gemma4 gate selected `267` tests, passed `234`, and skipped `33`
optional hardware or local-fixture cases, with no failures. Final E4B evidence
is under
`/private/tmp/antfly-gemma4-e4b-boolq-grpo-four-token-final-20260820-v3`;
the promotion-build same-binary rollback is under
`/private/tmp/antfly-gemma4-e4b-boolq-grpo-four-token-rollback-20260820-v2`.
This closes the bounded multi-token projection boundary, not long-horizon
quality or MLX parity. The next performance target is true incremental KV
reuse and prompt/prefix-aware active-candidate scheduling, so transformer work
is shared rather than merely reducing LM-head projection rows.

### Matched multi-token E2B/E4B GRPO with train-time KL control (2026-08-20)

Gemma4 GRPO now applies a raw mean-token K3 budget before every optimizer
mutation. `grpo.train_max_kl` defaults to `0.1`; a non-finite, negative, or
larger observation writes a `budget-exceeded` record and aborts without
admitting that group. Setting `adaptive_kl = true` additionally requires a
positive `target_kl < train_max_kl` and `kl_horizon >= 1`. The bounded
proportional controller uses the current beta for the current group and updates
the coefficient for the next group after every finite KL observation, including
a hard-budget rejection. Its horizon is measured in sampled completion episodes,
so each group contributes `group_size` controller steps. The result is clamped
to `min_kl_coef` / `max_kl_coef`
(defaults `0.001` / `1.0`). The next group always uses the updated coefficient,
including after a rejected observation. GRPO v10 train reports expose this
contract through v5 KL trace rows whose `objective_kl_coef` must exactly match
`kl_coef_before`; a mismatch fails before optimizer mutation. GRPO evaluation reports
carry raw `mean_kl`; the atomic `grpo_kl_control_trace.jsonl` binds every
observation, decision, optimizer-step count, and before/after coefficient.
The same admission path covers text and multimodal Gemma4 GRPO.

The matched campaign used pinned `google/boolq` revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The materializer selected 64
balanced train and 64 balanced, source-disjoint validation rows, forbade prompt
truncation at sequence length 128, required one-token `yes`/`no` targets, and
bound a four-token rollout cap. The materialization manifest SHA-256 is
`1b476974011cec1e725f2097b6d8581c9f0884f2e32e559763122681405ec7e3`;
train/eval JSONL SHA-256 values are
`01b35bec10abba5d540a29c0c4b0600c44bf54305d18bea0a7ea58d19b464057`
and `00b2c0fd7cacb639e2912a50e835b3a1d29c3d4dcad5b2fb13be83d582724f1e`.
Both E2B and E4B used the first eight train rows, 16 held-out rows, group size
four, rank-16/alpha-32 Q/V LoRA, learning rate `1e-7`, beta `0.04`, target KL
`0.01`, horizon `100`, and hard raw-KL budget `0.1` through the public
`antfly inference finetune run <recipe.json>` surface. The ReleaseFast binary
SHA-256 is
`e9bb7af988686e5b2730fc9c13af6b4bb67166a17e0ec4b4d0fb847453b3c57e`.

The pinned MLX comparator uses official MLX `0.31.2`, MLX-LM `0.31.3` at
commit `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, and runner SHA-256
`1a87f81e860204ae468d49d4c787a2c8a5b1f507671ade10321af6c141cf8062`.
It runs an exact Antfly completion/reward replay and an independent ranked MLX
rollout from the same seed adapter. An initial implementation batched divergent
MLX candidates and produced up to `0.359` sampling/rescore logprob error, so
those timings were rejected. The accepted comparator keeps sampling, policy
scoring, reference scoring, and differentiation at physical batch size one;
it token-normalizes and accumulates all four gradients, clips once, and makes
one optimizer update. All accepted E2B/E4B train lanes had exactly zero
sampling/rescore and differentiable-rescore error.

| Model | Antfly / MLX train seconds | Antfly / MLX eval seconds | Train / eval time ratio | Antfly / MLX peak physical | Antfly memory reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B | `100.231608 / 27.909717` | `113.877259 / 42.069351` | `3.591x / 2.707x` | `5,539,827,648 / 14,524,769,952` bytes | `61.86%` |
| E4B | `168.024695 / 50.238484` | `134.783294 / 66.170529` | `3.345x / 2.037x` | `7,201,281,352 / 20,408,988,776` bytes | `64.72%` |

Antfly admitted all eight groups. E2B's maximum raw train KL was
`4.68729e-5` and beta decreased to `0.03936446`; native MLX reached
`0.0315538` and beta increased to `0.04032064`. Held-out mean/top/positive-group
reward was exactly `0.34375 / 0.5625 / 1.0` in both implementations. E4B's
maximum was `1.12171e-5` with final Antfly beta `0.03936446`; native MLX
reached `0.01630496` with final beta `0.03952224`. E4B held-out mean and
positive-group reward matched at `0.234375 / 0.9375`; top-rank reward was weak
in both implementations (`0.0625` Antfly, `0.0` MLX). A first E4B run
correctly failed and withheld its adapter because a provisional absolute
top-rank floor of `0.25` was unsupported. The preserved v2 run kept nonzero
mean/positive reward and KL gates but treated top-rank reward as a parity
measurement; it reproduced the failed run's train, evaluation, and KL traces
byte-for-byte.

These are behavioral, not numerical-update, passes. Exact-trace adapter-delta
cosine was only `0.533616` for E2B and `0.598704` for E4B, despite relative L2
differences of `0.0316425` and `0.0241640`. Both comparison artifacts therefore
classify the result as `bounded-campaign-with-measured-drift` and explicitly set
`broad_grpo_performance_parity = false` and
`long_horizon_quality_parity = false`. Closing that boundary requires a
predeclared longer horizon, multiple seeds and tasks, baseline-relative quality
gates, and repeated same-Mac timing distributions; a single deterministic
BoolQ campaign is not sufficient.

Evidence:

- E2B Antfly root:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-adaptive-campaign-20260820-v1`
  (adapter SHA-256
  `d8c40074fa0a3e3d9784276c8ad19eea415377c8456f1904d05821eb6a59627c`);
- E2B MLX comparison:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-multitoken-adaptive-campaign-20260820-v1.json`
  (SHA-256
  `901166b34870576c2f5d6e5b9242107456bffb2160c826344b06a228eb951d27`);
- preserved fail-closed E4B quality-gate root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-adaptive-campaign-20260820-v1`;
- accepted E4B Antfly root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-adaptive-campaign-20260820-v2`
  (adapter SHA-256
  `6cd296df2eddd03fa87dfcc31d2666215844b08868b931a7d6c8cfa512999cc3`);
  and
- E4B MLX comparison:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-multitoken-adaptive-campaign-20260820-v1.json`
  (SHA-256
  `392d4603791e0ccf7bc2277040aa4cf9f1edf263b99e399267db6294a4fa3658`).

### Experimental incremental KV reuse and exact E2B/E4B gate (2026-08-20)

The sparse multi-token Metal sampler now has an opt-in paged-decode lane. Set
`ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV=1` to prefill each prompt once, share only
complete 16-token F32 KV pages with every candidate, replay each prompt tail,
and decode one token at a time with the live Q/V LoRA adapter. Ranked token
selection remains device-resident. Setting
`ANTFLY_GEMMA4_GRPO_INCREMENTAL_KV_SHADOW_EXACT=1` additionally runs the
qualified full-prefix sampler for the first group and rejects any selected-token
or F32 logprob-bit drift.

Paged decode and the qualified fixed-shape graph use different reduction
orders. The incremental route therefore canonicalizes each sampled
completion's policy logprobs with one qualified sparse full-sequence rescore.
This preserves exact optimizer and artifact behavior while still proving that
token selection uses the shared KV cache. GRPO v5 train and v3 held-out reports
record canonical/tail prefill counts, decode forwards, exact rescoring,
device-resident ranked selections, host fallbacks, shared/reused prompt tokens,
page size, and cache dtype.

`validate_gemma4_grpo_incremental_kv_parity.py` compared each opt-in campaign
with its full-prefix baseline. Both gates passed exact hashes for the training
reward trace, held-out reward trace, KL-control trace, and final adapter:

| Model | Train / eval canonical prefills | Train / eval decode forwards | Train / eval exact rescoring | Host fallback | Final adapter SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| E2B | `8 / 16` | `88 / 176` | `32 / 64` | `0` | `d8c40074fa0a3e3d9784276c8ad19eea415377c8456f1904d05821eb6a59627c` |
| E4B | `8 / 16` | `69 / 123` | `32 / 64` | `0` | `6cd296df2eddd03fa87dfcc31d2666215844b08868b931a7d6c8cfa512999cc3` |

The exact gate is a correctness success but the first implementation is a
performance rejection. Fresh matched MLX `0.31.2` / MLX-LM `0.31.3` campaigns
used the same pinned BoolQ rows, adapters, eight train groups, 16 held-out
groups, group size four, and four-token cap:

| Model | Full-prefix Antfly train / eval | Incremental Antfly train / eval | Incremental slowdown | Fresh MLX train / eval | Incremental Antfly / MLX |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B | `100.231608 / 113.877259 s` | `157.262167 / 215.813487 s` | `1.569x / 1.895x` | `27.769290 / 38.714746 s` | `5.663x / 5.574x` |
| E4B | `168.024695 / 134.783294 s` | `256.107200 / 278.369770 s` | `1.524x / 2.065x` | `51.297557 / 62.883978 s` | `4.993x / 4.427x` |

Sampling alone regressed `2.260x / 2.228x` for E2B train/eval and
`2.525x / 2.584x` for E4B. MLX timing moved only by single-digit percentages
versus the prior campaign, so the gap is not reference noise. The dominant
boundary is serial eager one-token decoder dispatch with live LoRA plus one exact full-sequence
rescore per completion. Incremental KV reuse remains default-off. Promotion
requires prompt/prefix-bucketed active-candidate batching or a LoRA-aware
compiled paged decoder, removal of canonical rescoring only after logprobs are
bit-exact, the same exact E2B/E4B artifact gate, and a measured win over the
full-prefix rollback. The previously rejected padded whole-transformer batch
is not an acceptable shortcut.

Evidence:

- E2B incremental root:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-incremental-kv-20260820-v6`;
- E2B fresh MLX comparison:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-incremental-reprofile-20260820-v1.json`
  (SHA-256
  `302608afceea524917a1cd8acf1cfeaa1d9cb4bc9c7706c663064ab25eb2857d`);
- E4B incremental root:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-incremental-kv-20260820-v1`; and
- E4B fresh MLX comparison:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-incremental-reprofile-20260820-v1.json`
  (SHA-256
  `d2dc9636a5f7f345d4de191fd051d67e3a8786090f463679210b15f606edb570`).

### Segmented prompt prefill and Unsloth transfer audit (2026-08-20)

The incremental sampler now copies each candidate's non-page-aligned prompt
tail from one canonical prefill instead of replaying that tail independently.
This is an exact optimization: the E2B and E4B campaigns reproduced the
accepted train reward trace, held-out reward trace, KL-control trace, and final
adapter byte-for-byte. Relative to the first active-candidate implementation,
train/eval sampling improved from `82.286618 / 158.290720 s` to
`74.673257 / 140.767838 s` on E2B (`9.3% / 11.1%`) and from
`124.513679 / 210.227072 s` to `104.816143 / 182.845191 s` on E4B
(`15.8% / 13.0%`). Peak physical footprint was `5.956 GB` and `8.063 GB`.

The same-binary full-prefix sampler is still faster: segmented incremental
sampling is `1.656x / 1.692x` slower on E2B train/eval and
`1.846x / 2.041x` slower on E4B. Fresh pinned MLX native rollouts took
`28.940351 / 42.193658 s` for E2B train/eval and
`51.768443 / 66.379926 s` for E4B. Comparing complete accounted Antfly train
and evaluation loops, not just the optimized sampling phase, leaves gaps of
approximately `4.50x / 4.06x` on E2B and `4.20x / 3.45x` on E4B. Antfly used
about `58.9%` and `59.8%` less peak physical memory than MLX, respectively.

Unsloth's [padding-free and explicit sequence-packing
work](https://unsloth.ai/docs/new/3x-faster-training-packing) identifies the
next high-value direction, but the existing Antfly `sequence_packing.zig`
utility is not wired into Gemma training. It already emits reset `position_ids`,
`doc_ids`, `cu_seqlens`, and a block-diagonal causal mask; however,
`GemmaAutodiffCtx.buildForward` currently ignores its `attention_mask`, and
`gemma_graph.zig` creates one fixed causal/sliding mask and global RoPE
positions. Concatenating examples now would therefore allow cross-example
attention and apply incorrect positions. It is not an admissible optimization.

The production implementation should use a segment-aware/ragged GQA graph:

1. start with length-bucketed independent batch rows to remove right-padding
   work without changing attention semantics;
2. add `cu_seqlens`-driven causal/sliding attention and per-segment RoPE reset,
   without materializing a quadratic block mask;
3. map sparse supervised rows through the packed layout for LoRA-SFT, preserve
   independent chosen/rejected units for DPO, and preserve independent
   completions plus shared-prompt accounting for GRPO;
4. retain sample order, optimizer boundaries, and stable reductions so the
   existing E2B/E4B trace and adapter-hash gates remain meaningful; and
5. promote only after both models beat the unpacked rollback with bounded peak
   physical memory.

An opt-in intermediate experiment batches independent GRPO completion rows for
backward only. Set
`ANTFLY_GEMMA4_GRPO_BATCH_MULTI_TOKEN_BACKWARD=1`; cap the physical batch with
`ANTFLY_GEMMA4_GRPO_MULTI_TOKEN_BACKWARD_MAX_BATCH` (default `4`). GRPO v5
reports record the physical batch size and physical micro-batches per group.
This lane is deliberately default-off:

| Model/lane | Backward/update | Gain vs serial | Peak physical | Exact adapter gate |
| --- | ---: | ---: | ---: | --- |
| E2B serial | `4.827423 s` | - | `5.611 GB` | canonical |
| E2B batch 2 | `3.067223 s` | `36.5%` | `8.587 GB` | fail, max abs `1.993e-7` |
| E2B batch 4 | `2.164943 s` | `55.2%` | `14.029 GB` | fail, max abs `1.996e-7` |
| E4B serial | `10.930324 s` | - | accepted campaign `8.063 GB` | canonical |
| E4B batch 2 | `7.817791 s` | `28.5%` | `12.076 GB` | fail, max abs `1.986e-7` |

The E2B one-group wall times were effectively flat (`32.42-32.55 s`), so the
faster backward did not yet improve end-to-end latency. The numerical drift is
small and expected from a changed floating-point reduction order, but it is
still outside the exact promotion contract. The lane remains useful for a
future predeclared tolerance-based quality campaign; it is not a production
default.

Unsloth's [long-context activation
offload](https://unsloth.ai/blog/long-context) is also not a current
sequence-128 Metal target. Apple Silicon uses unified physical memory, so
moving saved activations to CPU address space does not free a separate VRAM
pool and may add copies or synchronization. Reconsider it only for profiled
long-context runs (at least sequence 512) that fail a physical-memory gate.
Likewise, the [Unsloth GRPO memory
optimization](https://www.unsloth.ai/blog/grpo) that avoids retaining full
generation-by-token-by-vocabulary logits is largely already represented here
by sparse completion-row rescoring and fused/cut cross-entropy; current
profiles point to decoder dispatch, canonical rescoring, and backward rather
than retained full-vocab logits.

Evidence:

- exact E2B segmented campaign:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-kv-segmented-canonical-campaign-20260820-v1`;
- exact E4B segmented campaign:
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-kv-segmented-canonical-campaign-20260820-v1`;
- fresh pinned E2B/E4B MLX comparisons:
  `/private/tmp/antfly-gemma4-e2b-boolq-grpo-mlx-segmented-canonical-reprofile-20260820-v1.json`
  and
  `/private/tmp/antfly-gemma4-e4b-boolq-grpo-mlx-segmented-canonical-reprofile-20260820-v1.json`;
- E2B serial, batch-2, and batch-4 probes:
  `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-serial-control-20260820-v1`,
  `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-batch2-probe-20260820-v1`,
  and `/private/tmp/antfly-gemma4-e2b-grpo-group-backward-probe-20260820-v3`;
  and
- E4B batch-2 probe:
  `/private/tmp/antfly-gemma4-e4b-grpo-group-backward-batch2-probe-20260820-v2`.

### Opt-in independent-row length buckets (2026-08-21)

Text LoRA-SFT can now avoid executing every example at the prepared artifact's
maximum sequence length. The default remains the established fixed-shape path.
Enable the new policy explicitly:

```sh
antfly inference finetune train gemma4-lora \
  --model "$BASE" --adapter "$ADAPTER" \
  --train-prepared "$TRAIN_PREPARED" --eval-prepared "$EVAL_PREPARED" \
  --out "$OUT" --backend metal \
  --sequence-length-bucket-quantum 16 \
  --sequence-length-bucket-min 32 \
  --graph-cache-capacity 4
```

Recipes expose the same policy under `runtime`:

```json
{
  "runtime": {
    "sequence_length_bucket_quantum": 16,
    "sequence_length_bucket_min": 32,
    "graph_cache_capacity": 4
  }
}
```

Each example remains an independent causal row and keeps its original sample,
optimizer, attention, and RoPE order. The scheduler only rounds its logical
token count upward to a bounded graph shape; it never concatenates examples or
truncates logical tokens. A deterministic bounded graph cache retains the
resulting signatures. Run fingerprints bind the bucket policy and effective
cache capacity. The unchanged fixed policy retains the prior v2 fingerprint
domain so compatible fixed-shape checkpoints are not invalidated by this
feature.

Reports record logical, scheduled, and fixed-baseline rows; avoided padding;
bucketed-example and min/max-shape counts; and graph cache builds, hits, active
reuses, evictions, and peak residency. `epoch_history` additionally records
`antfly.gemma4.epoch-timing/v1` monotonic wall time and example, logical-token,
scheduled-token, supervised-token, and optimizer-update rates, plus the cache
delta for that epoch. `phase_timing` uses
`antfly.gemma4.phase-timing/v1` to separate initialization/restore, initial
evaluation, total epoch training, adapter save, and final evaluation. The
timers rely on the trainer step's existing completion contract and add no GPU
synchronization. Bucket and cache flags are rejected by the locked benchmark
contract. A cache capacity without a bucket quantum is also rejected. Public
multimodal training fails closed when bucketing is requested.

The first real-data systems campaign used the pinned 64-train/64-eval
`google/boolq` materialization at revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`, with four train rows and two eval
rows at prepared maximum 512. The source's `target` field was mechanically
renamed to the SFT loader's `response` field. Because the materialization's
prompts were already rendered for the GRPO campaign and were rendered again by
the SFT loader, this proves real-data execution and performance behavior, not a
canonical BoolQ quality-training result.

The four train rows contained `493` logical tokens. Quantum-16 scheduling used
`544` transformer rows per epoch instead of `2048`, avoiding `1504` rows
(`73.44%`). Both models built three signatures, reused them without eviction,
made eight strict-Metal optimizer updates, and reported zero fallback steps.

| Model | Fixed wall / peak physical | Bucketed wall / peak physical | Update parity from common input |
| --- | ---: | ---: | --- |
| E2B | `22.08 s / 6.732 GB`; reverse-order repeat `19.64 s / 6.732 GB` | `20.46 s / 2.113 GB`; repeat `32.66 s / 2.110 GB` | cosine `0.9999804`; relative L2 `0.0062656`; magnitude delta `0.0201%` |
| E4B | `72.49 s / 11.400 GB` | `42.12 s / 3.472 GB` | cosine `0.9999438`; relative L2 `0.0106013`; magnitude delta `0.00665%` |

The E4B pair improved end-to-end wall time by `41.9%` and peak physical memory
by `69.5%`. E2B peak physical memory improved by about `68.6%`, but its wall
time is inconclusive: the repeat suffered hundreds of thousands of macOS page
faults and reversed the first pair's small apparent gain. Do not use these
full-process E2B timings as a throughput claim. Darwin maximum RSS remained
large for both policies because it includes the mapped base artifact; peak
physical footprint is the relevant working-set measurement.

The fixed and bucketed E2B runs were each byte-reproducible across two fresh
processes (adapter SHA-256
`d962d9055a1c3d8e59808080616b3cb8ba00d0219590c530025f134ef9c14339`
and `3e27b97b335140bbccab51a841e3e88006d564d82b91052941f9b3c7d7eaf370`,
respectively). Fixed versus bucketed updates are not
bit-exact because the changed BF16/Metal shapes change floating-point reduction
order. The measured update cosines and magnitudes support a production-safe
opt-in, not a silent default change or exact-trajectory claim.

The larger steady-state campaign used 16 train rows, four disjoint evaluation
rows, three epochs, 48 strict-Metal optimizer updates, learning rate `1e-4`,
and seed `42`. The 20 rows span 18 distinct logical lengths but only four Q16
shapes (`96`, `112`, `128`, and `144`); every completion has seven supervised
tokens. Epochs two and three are the steady-state gate. Both bucketed arms
reported `reuse_only = true`, four resident signatures, zero builds, and zero
evictions in those epochs. Each policy avoided fallback for all updates.

| Model | Fixed steady epoch / logical tok/s | Q16 steady epoch / logical tok/s | Through-final-eval wall | Peak physical footprint | Update parity from common input |
| --- | ---: | ---: | ---: | ---: | --- |
| E2B | `18.796 s / 102.15` | `19.384 s / 99.05` | fixed `68.688 s`; Q16 `83.386 s` | fixed `6.732 GB`; Q16 `2.142 GB` | cosine `0.9999901`; relative L2 `0.0044516`; max abs `5.7765e-4` |
| E4B | `72.287 s / 26.56` | `37.114 s / 51.73` | fixed `257.486 s`; Q16 `140.876 s` | fixed `11.379 GB`; Q16 `3.525 GB` | cosine `0.9999829`; relative L2 `0.0058628`; max abs `7.5147e-4` |

Q16 removes `75.20%` of scheduled training rows in both pairs. On E4B that
becomes a `1.948x` steady-state speedup, a `1.828x` through-final-eval speedup,
and a `69.02%` footprint reduction. On E2B it saves `68.18%` of peak physical
footprint but is `3.13%` slower per steady epoch and `21.40%` slower through
final evaluation. The E2B result is now conclusive rather than page-fault
noise: at these short shapes, per-example launch/optimizer cost and lower GPU
occupancy outweigh avoided padded rows. Fixed shape therefore remains the
cross-model default. Q16 is the qualified E4B throughput/memory policy for
similarly heterogeneous short rows, but should stay explicit until broader
length distributions establish an automatic selection threshold.

The held-out results also agree: fixed/Q16 loss is
`0.2253238`/`0.2253264` for E2B and `0.00365768`/`0.00365325` for E4B. Shape
dependent BF16 reductions prevent byte identity, so the cosine/relative-L2
gate remains the correct numerical contract.

The final instrumented ReleaseFast Metal binary is 29,632,896 bytes with
SHA-256
`8654d156e2bc868757dd90362fab515aa757378ca7066149734f64f63f1e6e45`.
It produced all four steady-state roots above. The full focused ReleaseSafe
Metal gate selected 275 tests: 273 passed and two optional real-artifact tests
skipped.

Steady-state evidence:

- consolidated campaign artifact:
  `/private/tmp/antfly-gemma4-boolq-sft-steady16-length-bucket-campaign-20260821-v1.json`
  (SHA-256
  `bbc6ac638fbeca623521766cd29428c29f9e664b9c80fe9482c8dfbe214118d9`);
- prepared E2B train/eval artifacts:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-20260821-train-prepared.json`
  and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-20260821-eval-prepared.json`;
- prepared E4B train/eval artifacts use the corresponding `e4b` paths;
- E2B fixed and Q16 roots:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-fixed-20260821-v1` and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-steady16-bucket16-20260821-v1`;
  and
- E4B fixed and Q16 roots use the corresponding `e4b` paths.

Initial evidence:

- transformed real-data inputs:
  `/private/tmp/antfly-gemma4-boolq-sft-length-buckets-20260821-train.jsonl`
  and
  `/private/tmp/antfly-gemma4-boolq-sft-length-buckets-20260821-eval.jsonl`;
- E2B fixed and bucketed roots:
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-fixed-20260821-v1` and
  `/private/tmp/antfly-gemma4-e2b-boolq-sft-bucket16-20260821-v1`; reverse-order
  repeats use the corresponding `v2` roots;
- E4B fixed root and adapter SHA-256:
  `/private/tmp/antfly-gemma4-e4b-boolq-sft-fixed-20260821-v1`,
  `932c45051efafc91b852a648a1d083921a006f042e77f98333419ba45b8c1e7e`;
  and
- E4B bucketed root and adapter SHA-256:
  `/private/tmp/antfly-gemma4-e4b-boolq-sft-bucket16-20260821-v1`,
  `5eb178e1c9fba0c3834879eabc8e2fe49dbdde7a634b7a7a81e0d44948fe6faf`.

The independent-row policy remains SFT-specific, but DPO now has a separate
pair-safe scheduler. One rounded shape is computed from the maximum logical
chosen/rejected row; reference precompute, both policy branches, backward, and
held-out evaluation share that sequence length and the pair's maximum
weighted-target bucket. Recipe/report provenance binds the quantum, minimum,
cache capacity, scheduled rows, graph signatures, and phase snapshots. The
bucketed route defaults to the qualified four-graph cache; eight is an explicit
upper bound, not the implicit policy. Every distinct bucket graph signature is
checked for zero-adapter policy/reference parity before optimizer mutation.
Fixed padding remains the default and rollback. GRPO still requires its own
group-safe policy that preserves completion-group boundaries, shared-prompt
accounting, ranked selection, and canonical rescore semantics. True
padding-free packing remains the longer-term target: `cu_seqlens`-driven
causal/sliding attention, per-segment RoPE reset, and packed sparse-target row
mapping without a quadratic block mask.

### Pair-safe DPO buckets and matched MLX reprofile (2026-08-21)

The final campaign used `HuggingFaceH4/ultrafeedback_binarized` at revision
`3949bf5f8c17c394422ccfab0c31ea9c20bdeb85` (source parquet SHA-256
`e9dab2789f419d4204d73ec2c860af6d88d466b906e0109e69b96075467eb389`).
Both frameworks trained the same five pairs for five epochs: 25 optimizer
updates, sequence maximum 512, beta `0.1`, AdamW learning rate `1e-4`, and
rank-16/alpha-32 Q/V adapters. Q128/min256 scheduled pair shapes
`[384, 384, 512, 256, 256]` instead of five fixed 512-row shapes, reducing
executed branch rows from `5120` to `3584` (`30%`). Antfly used cache capacity
four, which held all four training signatures after one bootstrap eviction.

Each performance row below is one fresh process. Times are the median and mean
of the 20 measured updates after the locked cold/first/three-warmup sequence;
memory is macOS peak physical footprint. This is a same-Mac matched reprofile,
not yet the five-process alternating distribution gate.

| Model / policy | Antfly median / mean | MLX-LM median / mean | Antfly / MLX median | Antfly peak | MLX peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2B fixed 512 | `2.4951 / 2.8161 s` | `2.1114 / 2.1124 s` | `1.182x` | `8.572 GB` | `18.945 GB` |
| E2B Q128/min256 | `1.8727 / 2.0616 s` | `1.5354 / 1.4590 s` | `1.220x` | `8.258 GB` | `18.841 GB` |
| E4B fixed 512 | `32.9120 / 38.0635 s` | `37.3854 / 38.4071 s` | `0.880x` | `21.573 GB` | `26.390 GB` |
| E4B Q128/min256 | `26.6671 / 34.3067 s` | `15.3322 / 16.3493 s` | `1.739x` | `26.629 GB` | `26.366 GB` |

Within Antfly, Q128 improved median time by `1.332x` on E2B and `1.234x` on
E4B. It reduced E2B peak memory by `3.66%`, but increased E4B peak by `23.44%`
because four specialized sequence executables remain resident. Fixed E4B is
the strongest current result: Antfly is `11.97%` faster by median, effectively
tied by mean, and uses `18.26%` less peak memory than MLX-LM. The optimized
E4B boundary is not parity: MLX-LM Q128 is `1.739x/2.098x` faster by
median/mean with effectively equal memory. MLX converts the same row reduction
into a `2.438x` median speedup while Antfly retains substantial per-shape and
per-branch overhead.

Quality was scored under one pinned MLX oracle on source rows disjoint from
training. E2B Q128 Antfly and MLX agreed on all five decisions and both reached
`0.60` accuracy. The expanded E4B gate used 25 unique rows with zero training
overlap: Antfly/MLX Q128 agreement was `22/25` (`0.88`) with `0.60`/`0.64`
accuracy; fixed agreement was also `0.88`. Fixed-versus-Q128 agreement within
each framework was `0.96`. This passes a bounded behavioral gate but fails
exact decision parity and does not establish broad, multi-seed, or long-horizon
quality parity. Shape-dependent BF16 reductions also make fixed and bucketed
adapter bytes intentionally non-identical. Fixed scheduling therefore remains
the default and rollback; Q128/min256 remains an explicit throughput policy.

The tested ReleaseFast CLI SHA-256 is
`63bd3647bfb8c6c1c9cddc4bb1a9f091b291e6abde7aba271ccdd1b8843c5bf2`.
The bounded diagnostic comparison used MLX `0.31.2` at
`68cf2fddd8de5edd8ab3d926391772b2e2cedad8` and MLX-LM `0.31.3` at
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`; its recovered SDK 26.2 Metal
library was hash-bound in the campaign. That archived runner predates the v3
closed-native-runtime contract: it did not bind the loaded Python extension,
both runtime dylibs, package inventory, and native-build receipt as one
postflight-rechecked bundle. Its measurements remain useful same-machine
diagnostics, but are not promotable cross-framework release evidence. New DPO
MLX training and shared-oracle evaluation runs require the same strict
`--mlx-build-attestation` used by the LoRA oracle runner. Consolidated evidence is at
`/private/tmp/antfly-gemma4-dpo-length-buckets-20260821-v1/campaign.json`
(SHA-256
`541ea5ab5cbf66d16618cf987f9f1e56e89269d53d03db8e031e486326dfc321`).
The expanded E4B quality summary is beside it as
`e4b-holdout-25-summary.json` (SHA-256
`8b6cd22bb905e8db532c7106c45d174fed44708ed29f86b2b7b590d321e94b73`).

The next performance target is a memory-safe compiled whole-pair DPO objective
or segment-aware packed chosen/rejected graph. It must preserve pair-shared
attention/RoPE/target semantics and optimizer order while amortizing the two
branch passes and short-shape launch overhead. More bucket-quantum sweeps are
unlikely to close the optimized E4B gap: the schedule already removes 30% of
rows, while Antfly's four graph signatures and branch execution dominate the
remaining difference.

The post-campaign production review emits DPO report v5. It replaced quadratic
graph-signature and train/eval-overlap scans with hash-indexed linear passes,
made cache capacity four the bucketed default, added the pre-update parity gate
for every distinct bucket signature, and made DPO MLX training report v3 and
shared-oracle evaluation v2 verify and postflight-recheck the full model/input
surface, installed package versions, clean source revisions, loaded native
runtime, and strict build attestation.
These changes do not alter the explicit cache-four Antfly training schedule
used for the measured campaign; its binary hash remains the immutable Antfly
performance provenance above.

The production-review source snapshot was rebuilt as the shipping
`antfly` CLI (77,181,608 bytes, SHA-256
`b93433368ab36a53eb7b607be453e2f981a9d9fe15ef650da7c2741a5a87ca05`).
Its real-device ReleaseSafe gate selected 276 tests: 274 passed and the two
optional fixture-dependent tests skipped. Fresh CLI acceptances then trained
one epoch over the same five pinned UltraFeedback pairs on both E2B and E4B.
Each emitted report v5, executed five optimizer updates, admitted the implicit
four-graph cache, checked all four observed graph signatures with exactly zero
initial policy/reference error, published a changed adapter, and passed a
five-row held-out evaluation with zero prompt overlap. Independent adapter
validation found 50 E2B and 66 E4B Q/V target modules at rank 16/alpha 32. The
immutable roots are
`/private/tmp/antfly-gemma4-production-review-e2b-dpo-v5-20260821-v1` and
`/private/tmp/antfly-gemma4-production-review-e4b-dpo-v5-20260821-v1`.
These are mechanics and artifact acceptances; their deliberately permissive
smoke thresholds do not replace the longer campaign's bounded quality result.

## Preference Recovery and Terminal Evaluation Boundary (2026-08-21)

DPO and GRPO report v6 close the epoch-boundary recovery contract that report
v5 left implicit. A preference checkpoint is admitted only with a matching
`antfly_gemma4_preference_checkpoint_state/v1` content-addressed sidecar. The
sidecar restores exact DPO/GRPO aggregates; GRPO additionally restores reward
trace state, raw and weighted KL totals, adaptive-KL controller state, and the
initial diagnostic prefix. Publication orders sidecar-before-checkpoint, so a
visible checkpoint cannot name unpublished aggregate state.

The preference run fingerprint is v5. Besides model, adapter, train/eval data,
optimizer, reward, and graph/scheduling controls, it hashes the resolved
`antfly_gemma4_metal_numerical_policy/v2` contract: sparse-loss and CCE tile
geometry, BF16 kernels, eager LoRA/dot/quant routes, and graph-executor fusion
routes. A change to any covered route therefore rejects resume before optimizer
mutation. Unknown `TERMITE_METAL_*`, `TERMITE_GEMMA4_*`, or Gemma preference
environment controls fail closed until they are explicitly reviewed and added
to this attestation boundary. Admitted boolean values are canonical `0`/`1`;
presence-only kill switches accept only `1`. The fingerprint also binds DPO
activation-checkpoint interval/recursion and the exact GRPO backward batch size.
Checkpointed GRPO rejects custom reward trace/exchange paths rather than sharing
mutable files across artifact roots. The main report records the same policy
and the checkpoint sidecar path/digest.

The product and both preference qualifiers now share
`src/finetune/gemma4_preference_environment.policy` as the single typed source
for environment scope, admission, sanitization, and strict bindings. The Zig
binary embeds it; Python qualification reads it directly and records its
SHA-256. Qualification strips every inherited `TERMITE_*` and
`ANTFLY_GEMMA4_*` override before installing the strict executor bindings.
Product planning rejects unreviewed semantic/debug controls and explicitly
denies graph-output ownership/elision, output host-mirror resync, and paged-KV
kill switches before printing a plan or publishing any run artifact.

Held-out preference evaluation no longer inherits the training allocation
history. The source trainer drains Metal, synchronizes the final trainables to
host, retires compiled graphs and device optimizer slots, and becomes
terminal-only. A separately initialized backend receives the exact host
snapshot and runs evaluation with in-frame and completion-cache private-buffer
reuse disabled. Reports attest the policy string
`terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled`;
any attempted optimizer step after that boundary fails.

`scripts/gemma4/qualify_gemma4_preference_resume.py` schema v2 runs the same pinned
recipe uninterrupted, kills a second process only after an epoch-1 checkpoint
and sidecar are durable, then resumes into a new immutable output root. It
requires byte-identical adapters, exact final sidecars, training metrics,
completion/reward/KL-control traces, discrete evaluation behavior, and Metal
policy. It also opens the standalone evaluation reports and all three GRPO
trace artifacts, verifies their paths and content digests, and compares their
semantic contents. It rejects symlink leaves and output/input overlap, validates
the complete adapter tree plus manifest, preserves every immutable seed
companion, and requires a changed tensor payload. DPO terminal metrics are
exact. Repeated E2B evaluation-only replays
from the same byte-identical checkpoint showed fresh-process variation only in
GRPO's terminal Metal KL reductions, so the qualifier removes no other field
and permits only:

- `evaluation.kl_loss`: absolute delta at most `1e-6`;
- `evaluation.mean_kl`: absolute delta at most `1e-5`.

The standalone evaluation report's derived `loss` uses a `1e-6` bound because
it includes weighted KL; `pg_loss` remains exact. All observed values and
bounds are emitted in the qualification report. A value outside a bound, any
adapter/checkpoint/sidecar/trace/discrete drift, a missing artifact, or a
missing numerical-policy attestation still fails closed.

The 2026-08-21 recovery-boundary source snapshot was rebuilt as a
29,795,824-byte ReleaseFast Metal
binary with SHA-256
`b6b8cd957cee34058ffd0cae3a1a9ca0794ba8ad7562647253775b1bb0540304`.
The real-device focused gate selected 279 tests: 277 passed and the two optional
fixture-dependent tests skipped. All four same-binary v5-fingerprint/v2-policy
SIGTERM/resume campaigns then passed:

| Task / model | Recovery comparison | Adapter SHA-256 | Held-out result |
| --- | --- | --- | --- |
| DPO E2B | exact checkpoint, sidecar, adapter tree, metrics, and evaluation | `3e690f8bff46af295028328d46fcdaee6f484f04cf57d82ce2feafd5005555bd` | loss `0.68912619`, accuracy `0.60`, passed |
| DPO E4B | exact checkpoint, sidecar, adapter tree, metrics, and evaluation | `657bbd8f89b738cdd6453e147955eb099da53e6b25ad3778fabb2c324071df78` | loss `0.64424115`, accuracy `0.80`, passed |
| GRPO E2B | exact training/discrete trajectory; terminal `kl_loss` / `mean_kl` deltas `1.13e-7 / 2.81e-6` | `43e6d637a316b38247737c5e3879843e5576b95ed1c81655850f91962666a787` | mean/top-rank reward `0.34375 / 0.5625`, passed |
| GRPO E4B | exact training/discrete trajectory; terminal `kl_loss` / `mean_kl` deltas `9.59e-9 / 2.40e-7` | `0efbf1ad76f9e145382c18aaa24813d081220e8ceb373988bf33b0967e92a8c7` | mean/top-rank reward `0.234375 / 0.0625`, passed |

The reports for that recovery snapshot are:

- `/private/tmp/antfly-gemma4-e2b-dpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-dpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e2b-grpo-preference-resume-acceptance-20260821-v11/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-grpo-preference-resume-acceptance-20260821-v11/qualification_report.json`.

The pre-v5/pre-policy-v2 reports are retained at:

- `/private/tmp/antfly-gemma4-e2b-grpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-grpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e2b-dpo-preference-resume-acceptance-20260821-v2/qualification_report.json`;
- `/private/tmp/antfly-gemma4-e4b-dpo-preference-resume-acceptance-20260821-v4/qualification_report.json`.

The first E4B GRPO qualifier invocation failed safely because an older seed
directory had a tensor checkpoint but no `adapter_config.json`. The accepted
campaign uses
`/private/tmp/antfly-gemma4-dpo-length-buckets-20260821-v1/e4b-seed-qv-r16-a32`,
whose adapter tensor SHA-256 is identical
(`44c328721be136ce6795d7870109ca704a04b7094e8fc9553947620e6a30ddc5`)
and whose complete
rank-16/alpha-32 Q/V contract names the pinned E4B base. The rejected root was
preserved rather than repaired in place.

The v11 artifacts close the recovery and terminal-evaluation boundary for the
exercised short E2B/E4B DPO/GRPO campaigns. They do not establish broad
long-horizon quality or distribution-level performance parity. The subsequent
checkpoint contract admits incremental-KV GRPO only at a drained,
zero-live-sequence epoch boundary and restores cumulative telemetry while
rebuilding transient pages. Canonical full-prefix direct-GGUF E2B DPO/GRPO is
an explicitly gated research lane; combining that Q4_0 base with incremental
KV remains fail-closed because the paged decoder changed the canonical token
trajectory.

## Final Preference PR Qualification (2026-08-22)

The audited source was rebuilt once as a 29,831,008-byte ReleaseFast Metal
binary with SHA-256
`6e0dde73f07fd7bbbb7ef5d953979580c52fd9d7b91a1d9fe323b012965d3402`.
All qualification below uses that exact binary. The post-documentation local
mirror of the required PR workflow ran the same 280-test finetuning root in
Debug, ReleaseSafe, and ReleaseFast; each passed 278, skipped only the two
optional fixture-dependent tests, and failed none. The Debug and ReleaseSafe
Gemma graph roots passed `7/7`, the filtered ownership/lifecycle gate passed
`4/4`, the pinned Python 3.12 suite passed `593/593`, and deterministic fixture
regeneration passed. Hosted required CI remains pending until the branch is
pushed; documentation changes do not alter the qualified executable.

The long-horizon harness used seeds `17`, `42`, and `991`, eight epochs, strict
Metal, and a fixed initialized adapter. Each seed receives a deterministic
permutation of the same immutable row multiset. DPO runs use five pinned
UltraFeedback pairs per epoch and a disjoint five-pair holdout; GRPO runs use
eight pinned BoolQ prompts per epoch, group size four, a four-token completion
cap, and 16 disjoint held-out groups. The training and held-out dataset
SHA-256 values are respectively
`45d12cce115dfd0b9ec40b1e0c98d7b805f56adee559467b940762e9b8240f2f` /
`29a34e7d08e045fb50eebcf441a29c02b909d50eb33d112027141bc9dfb0a1de`
for UltraFeedback and
`01b35bec10abba5d540a29c0c4b0600c44bf54305d18bea0a7ea58d19b464057` /
`00b2c0fd7cacb639e2912a50e835b3a1d29c3d4dcad5b2fb13be83d582724f1e`
for BoolQ.

| Campaign | Result | Optimizer horizon | Held-out evidence |
| --- | --- | ---: | --- |
| E2B DPO | **PASS**, three distinct adapter digests | `3 x 40` updates | accuracy `0.60-0.80` (mean `0.6667`), loss `0.5888-0.7921`, positive margin `0.2648-0.8901` |
| E4B DPO | **PASS**, three distinct adapter digests | `3 x 40` updates | accuracy `0.60-0.80` (mean `0.7333`), loss `0.5939-0.7355`, positive margin `0.2986-0.4979` |
| E2B GRPO | **PASS**, three distinct adapter digests and zero host-logit fallback | `3 x 64` updates | mean/top-rank reward `0.34375 / 0.5625`, positive-group rate `1.0`, KL loss `2.73e-6-3.43e-6` |
| E4B GRPO | **FAIL** at seed 17; seeds 42/991 not run after fail-fast | `64` updates completed | mean reward `0.234375` and positive-group rate `0.9375` pass, KL loss `1.16e-6` passes, but top-rank reward `0.0625` misses the predeclared `0.125` floor |

The passing campaign reports and SHA-256 values are:

- `campaigns/e2b-dpo/campaign_report.json`:
  `e2424c14e4c690ad5c2740369ea27454a7c37b0e99a7c9d92a87d513f8aa6247`;
- `campaigns/e4b-dpo/campaign_report.json`:
  `c5ef75f4fac2b82c357aaa540fc195260cf944a9a9bbfff59bb98909ca893c09`;
- `campaigns/e2b-grpo/campaign_report.json`:
  `7f3de930dbd833b6af2de172e35268bfa63b35c72264d21c85fd143dff241532`.

The preserved E4B GRPO failure report is
`campaigns/e4b-grpo/campaign_report.json` at SHA-256
`84a8f27b2b40df2fdc037edc44b72dcc2b79a88bde87869007252b5c4d207668`;
its completed seed-17 adapter is
`39ed6a3aa48c59511324fe420c15cd05dde8826f8234754471b51c703f315684`.
All paths above are relative to the immutable evidence root
`/private/tmp/antfly-gemma4-prready-20260822-v1`. The E4B run completed cleanly
and changed the adapter, but its training reward repeated exactly across all
eight epochs and only one of 16 held-out top-ranked candidates was correct.
The campaign floor was not weakened after observing this result. E4B GRPO
therefore remains a quality blocker, not a runtime or numerical-policy failure.

The same binary also reran the experimental direct-GGUF E2B recovery boundary
against official Q4_0 GGUF SHA-256
`fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634`.
Canonical full-prefix DPO passed exact interruption/resume with adapter
`ead771802f218710253e09f993da8a510bed40b44d5305af4dda397124fdec5f`
and qualification-report SHA-256
`5e9683ed0a3c819676e3ced992eba6b2387b4c5e5865745f7f74d97cd7543546`.
Canonical full-prefix GRPO passed exact training/discrete recovery plus the
documented bounded terminal Metal KL comparison, with adapter
`9ce1b03dc532740961f189b589930efbc46c2fbd7c9108ece0e4262d901893b3`
and report SHA-256
`6d612a6867517a04aa6543336ae97cc891a173e85172ca0cf07612fa92207eab`.
The negative direct-GGUF/incremental-KV dry run still exits nonzero with
`DirectGgufGrpoIncrementalKvNotQualified`; exact shadowing previously exposed
the divergent token prefix `3771 236761 7993 236743` versus canonical
`3771 236761 108 16907`.

SafeTensors E2B and E4B incremental-KV GRPO separately passed the two-epoch
SIGTERM/resume gate with active-candidate batching, prompt-tail cloning, exact
full-prefix shadowing, and zero host-logit fallbacks. Both runs reproduced
their uninterrupted adapters and terminal KL values exactly:

| Model | Adapter SHA-256 | Qualification-report SHA-256 | Held-out mean / top-rank reward |
| --- | --- | --- | ---: |
| E2B | `43e6d637a316b38247737c5e3879843e5576b95ed1c81655850f91962666a787` | `57b9f701cf0ae3374af37f5469e637fff47c77fa4911482e36036470823514ff` | `0.34375 / 0.5625` |
| E4B | `0efbf1ad76f9e145382c18aaa24813d081220e8ceb373988bf33b0967e92a8c7` | `eec8f3936d8eaadfa36935d4480653b671bca005b3e80854cdc2e5d314b238d5` | `0.234375 / 0.0625` |

These reports live under `boundaries/e2b-incremental-resume` and
`boundaries/e4b-incremental-resume` in the same evidence root. They prove
frame-drained epoch-boundary recovery and telemetry continuity; the E4B row
does not override the separate long-horizon top-rank quality failure.

The final-source same-Mac DPO reprofile uses pinned MLX `0.31.2`, MLX-LM
`0.31.3` at revision
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, clean MLX revision
`68cf2fddd8de5edd8ab3d926391772b2e2cedad8`, and native-build attestation
SHA-256
`96a04c5e5176788310f373863054821ee35e90ec0e609cec35df854831b78f09`.
Every cell uses one cold update, one first update, three warmups, and 20
measured updates over the identical five-pair order. Peak physical footprint
covers the complete process lifetime. Q128 means pair-safe 128-row buckets
with a 256-row minimum.

| Model / schedule | Antfly median / mean | MLX median / mean | Antfly / MLX peak physical bytes | Median result |
| --- | ---: | ---: | ---: | ---: |
| E2B fixed | `2.498859 / 2.810538 s` | `2.047034 / 2.046491 s` | `13,012,883,360 / 18,910,715,824` | MLX `1.221x` faster; Antfly `31.19%` lower memory |
| E2B Q128 | `1.861362 / 2.049095 s` | `1.530776 / 1.453914 s` | `11,442,083,456 / 18,957,165,112` | MLX `1.216x` faster; Antfly `39.64%` lower memory |
| E4B fixed | `29.944145 / 32.246341 s` | `24.459206 / 25.994102 s` | `21,924,705,720 / 26,414,804,728` | MLX `1.224x` faster; Antfly `17.00%` lower memory |
| E4B Q128 | `22.326479 / 26.784846 s` | `11.221655 / 10.058360 s` | `26,676,814,464 / 26,362,818,560` | MLX `1.990x` faster; Antfly `1.19%` higher memory |

Q128 improves Antfly median update time by `1.342x` on E2B and `1.341x` on
E4B. It reduces E2B peak memory by `12.07%`, but increases E4B peak memory by
`21.67%`. The fresh fixed-E4B cell no longer reproduces the earlier Antfly
timing lead; MLX is faster in all four final-source cells. Common-oracle MLX
evaluation of every Antfly adapter still prefers all five chosen responses.
These are single-process diagnostics, not an alternating repeated-process
distribution.

The Antfly wrapper / task-report and MLX report SHA-256 pairs are:

- E2B fixed:
  `66a86326d03569d2f336664c8631cb96c74bbb6836cedd97bdfe35cd5ea8fc2b` /
  `333f9e7e58d557b9e4cbc5a7dc27887d88fda06aede3e07a9b24ed82dd36599d`,
  MLX `b8bee961ad4f33baec0329c9fa2312285b29c21339f69b50afa07f9b0321c3cc`;
- E2B Q128:
  `35199634cf49b3ffc3369adafd196b8554c377d9c3e391e88e2f657e36e59c9f` /
  `83dec9dcbdd654afd5c550f030b267dc1fc148df3f92b20c738823f920ee63f1`,
  MLX `3379ae4a5b5e43622e5e18478783cb7b1ae8918284da3645e27c59c64e2bdd48`;
- E4B fixed:
  `a9b405ff63b9fea4c0ba09d3ecd6bc68b90e5acbadab6d12d4b1d341d27e6095` /
  `cc281bcb8e8888a2d60f3f04c167cbe4c01b0f72acb9aee60032a4553b49c364`,
  MLX `c1135b3edfc6cf891960ba2eb583528a69b5cefa4ba719af0eccd005220fa533`;
- E4B Q128:
  `4d977802c0c0e70a288a6f97a695532570a619fd72a7a7c7d6d45766272568b8` /
  `bba65f42c9b4b4b81e28cb07f9a9ca64b676bca08982e24ccee8fc94e9ebb19c`,
  MLX `043283a3e02eb3713c24a62fe13b2660e5f0b4565efe52e4cb3ad11554e88d36`.

The first E2B fixed wrapper attempt lost only its final process-exit footprint
sample after training succeeded; the corrected sampler tolerates that bounded
post-exit race. The interrupted E4B Q128 root has no wrapper report and remains
excluded. Both incomplete roots were preserved, and accepted reruns used new
immutable `v2` roots.

These archived campaigns establish bounded RNG/data-order robustness only:
all three historical seeds start from the same adapter and use absolute floors.
The current campaign contract replaces that design with independently seeded
adapter bootstrap plus strict baseline-relative held-out gates, but no archived
result is retroactively upgraded; the pinned campaigns must be rerun. The
historical E2B high-learning-rate GRPO collapse remains valid negative
evidence. None of these results enables public `qlora-sft` or broad E4B GRPO
quality claims.

## Preference P1 Hardening (2026-08-23)

The final production review closed two public-boundary defects without
expanding the qualified training surface. First, mutable output paths are no
longer compared only after lexical normalization. The shared fine-tuning path
guard canonicalizes the deepest existing ancestor and retains any missing
suffix, so an output routed through a symlink into an immutable model, dataset,
adapter, checkpoint, reward input, or planned report fails before creation.
Second, product admission and Python qualification use the shared typed
environment policy described above. The four previously unqualified
correctness switches fail closed, and the qualifiers cannot inherit them.

The exact-source PR gate selected 284 Gemma4 fine-tuning tests in each of
Debug, ReleaseSafe, and ReleaseFast: 282 passed and only the two optional
fixture-dependent tests skipped. The focused graph gate passed 7/7 in Debug
and ReleaseSafe, the lifecycle smoke passed 4/4, the shared policy and path
suites passed 3/3 each, and the deterministic 16-row oracle fixture regenerated
and checked byte-for-byte. The final arm64 ReleaseFast binary is
`/private/tmp/antfly-gemma4-p1-fixed-release-20260823-v3/bin/antfly-inference`
at SHA-256
`0b97bb30396f14e15fc126ca156f0da79b7a8b6a98c9366d21a2754c7fbe2b97`;
its embedded policy source is SHA-256
`62333c528648a43a4711f7ab4a04e0fc70876f54b37549443de8469d8ec88dd5`.
Public probes against that binary rejected the symlinked output as
`PreferenceArtifactInputConflict` and each of
`TERMITE_DISABLE_GRAPH_OUTPUT_OWNED_COPY`,
`TERMITE_DISABLE_GRAPH_OUTPUT_ELISION_OVERRIDE`,
`TERMITE_DISABLE_OUTPUT_HOST_MIRROR_RESYNC`, and
`TERMITE_DISABLE_PAGED_KV` as
`UnattestedGemma4PreferenceEnvironmentOverride`; neither probe created an
output manifest, report, adapter, or model subdirectory. These fixes do not
change the preserved E4B GRPO quality failure or enable direct-GGUF plus
incremental-KV GRPO.

## E2B Preference Performance Candidate (2026-08-26)

This pass treated performance as a correctness-constrained release gate. It
used the retained ReleaseFast rollback binary for DPO A/B tests and a separately
frozen candidate binary for GRPO. Every accepted comparison used the same E2B
BF16 model, rank-16 / alpha-32 Q/V adapter, sequence length 128, data order,
seed, and update/evaluation schedule. All benchmark processes reported zero
swap-in and swap-out. The host already had system swap allocated before the
campaign, so these runs are diagnostic evidence rather than a fresh-host
zero-paging release attestation.

The fixed-shape 25-update DPO control had a `2.473715 s` median update. Four
behaviorally exact candidates did not improve it:

| DPO candidate | Median update | Change from control | Decision |
| --- | ---: | ---: | --- |
| Coalesced gradient snapshot | `2.478296 s` | `+0.18%` | reject |
| Ping-pong gradient buffers | `2.478647 s` | `+0.20%` | reject |
| Slot-bound outputs | `2.520215 s` | `+1.88%` | reject; lower footprint did not offset the regression |
| BF16 gate/up MLP pair | `2.559719 s` | `+3.48%` | reject |

No DPO optimization from this pass is promoted. The closest pinned same-Mac
MLX comparisons above remain `1.216x` to `1.221x` faster on E2B, so the next
DPO target is graph-region and command-boundary coarsening around the branch
forward/backward path, not another output-buffer lifetime variant.

For GRPO, the candidate adds a forward-output-only compiled session that keeps
the inference fused-GQA opcode, captures the complete first output explicitly,
and caches that session with the trainer. It is gated by
`ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING=1`, admitted only for multi-token Metal
sampling, conflicts with incremental KV, changes the execution fingerprint,
and is off by default. The first implementation redirected fused GQA to its
training/VJP alternate and correctly failed initial policy/reference parity;
the admitted candidate preserves the original forward opcode. A separate
cached-direct experiment reached `50,338,567,152` peak physical bytes and
terminated without publishing a result, so it is rejected.

The largest matched A/B used eight training groups and sixteen held-out groups:

| E2B GRPO metric | Control | Compiled sampling | Result |
| --- | ---: | ---: | ---: |
| End-to-end wall time | `226.68 s` | `178.36 s` | `21.32%` faster (`1.271x`) |
| Training sampling | `43.225250 s` | `26.355352 s` | `39.03%` faster |
| Policy rescore | `1.970764 s` | `1.323338 s` | `32.85%` faster |
| Backward/update | `38.055118 s` | `38.282704 s` | effectively flat |
| Held-out evaluation loop | `119.541959 s` | `88.403845 s` | `26.05%` faster |
| Peak physical footprint | `5,530,913,480` | `5,671,013,160` bytes | `2.53%` higher |

Both paths produced adapter SHA-256
`8985cf51ff52b93f8d026ed4cc210a190096a6a45f897b37a57f27e5de1f6448`.
Their training reward trace, evaluation reward trace, and KL-control trace were
byte-identical at SHA-256
`03e3ffe47ddfab31e4525df9c9db6df5b65335c9b225b9b1cee865fb6d590714`,
`2f13d36195139ef8db751b5560d0e1d2ef8fd31c9467d2b5ec7d0be35bd55a75`,
and `2928c08feca474ae9de7e9d3ed384cb2a7213bbef32609e6a49917090c077473`.
Training KL/loss was exactly equal; held-out mean KL differed by
`9.2408845e-7`, while reward aggregates remained exact. Initial
policy/rescore and policy/reference maximum errors were zero.

The benchmarked candidate is frozen at
`/private/tmp/antfly-gemma4-opt-20260826/compiled-sampling-345c515b/bin/antfly-inference`,
binary SHA-256
`e6137279b66a305ee8f293a6dc58b0430020c73027fcd880bc8fbddd5f380b38`,
with benchmark-source patch digest
`345c515b544bf0185cf90a5a5c92b961dd86e5fa3d8f4dfd283290a1c5d988e0`.
The immutable A/B roots are
`/private/tmp/antfly-gemma4-opt-20260826/grpo-e2b-control-8x16-r1` and
`/private/tmp/antfly-gemma4-opt-20260826/grpo-e2b-compiled-8x16-r1`.

The `56071123...` / `9b61c4e6...` snapshot and two-update smoke were the first
in-process candidate. A later fresh-process interruption/resume qualification
found a compatibility defect that the matched same-process A/B could not see:
selected tokens and rewards remained exact, but raw compiled log-probabilities
changed after execution-cache recreation and changed the optimizer trajectory.
The earlier table remains useful historical performance evidence, but that
raw-log-probability candidate is not promotable.

### Restart-safe compiled GRPO qualification

The corrected path keeps compiled multi-token selection, then explicitly
rescores every selected completion through the restart-stable eager sparse
policy graph at batch size one. It verifies every selected token at its
expected ranked position and uses those canonical eager values for both GRPO
old-policy and current-policy log-probabilities. Batch-wide eager rescoring was
rejected because tied-head reduction geometry changed log-probabilities by
approximately `0.022267`. The report mode is
`compiled-token-selection-with-eager-per-completion-token-validated-logprob-rescore`.
Checkpoint restore also recreates the frozen-reference cache, retires both
uninterrupted and resumed compiled execution caches, and synchronizes and
byte-validates restored Metal weight, optimizer, and gradient-accumulator
state before the next graph.

The full two-epoch E2B interruption/resume qualifier passed 16 groups and 64
completions. All 64 completions were eagerly rescored. Uninterrupted and
resumed runs produced adapter SHA-256
`43e6d637a316b38247737c5e3879843e5576b95ed1c81655850f91962666a787`,
training checkpoint SHA-256
`936f03e81589c1e3af59c6bc043226959300042a1a03c42e0608977a9c27172c`,
and checkpoint-sidecar SHA-256
`89f54c9945f531ed3c12ad4bcaf42be92f51c1bf2f891a2aa3a8ed83d70f1280`.
Adapter, checkpoint, reward, and KL-control trajectories were exact. The only
non-exact values were the separately admitted fresh-backend terminal Metal
derived KL/loss values, which passed their narrow absolute bounds. The terminal
evaluation passed at mean/top-rank reward `0.34375 / 0.5625` and positive-group
rate `1.0`. Qualification report SHA-256 is
`503382159f640cd936fc69655765b1187ec261c486325a5eca8adce5ad1b6fd8`;
the retained root is
`/private/tmp/antfly-gemma4-opt-20260826/compiled-resume-e2b-token-shadow-full-r23`.

The matched corrected eight-group/16-holdout benchmark is still faster than
the eager Antfly control, but the correctness rescore makes the real remaining
gap visible:

| E2B GRPO metric | Eager control | Restart-safe compiled | Result |
| --- | ---: | ---: | ---: |
| End-to-end wall time | `226.68 s` | `189.46 s` | `16.42%` faster (`1.196x`) |
| Training sampling | `43.225250 s` | `26.111656 s` | `39.59%` faster |
| Policy rescore | `1.970764 s` | `15.011305 s` | correctness cost; 32/32 completions |
| Backward/update | `38.055118 s` | `38.260600 s` | effectively flat |
| Held-out evaluation loop | `119.541959 s` | `87.347253 s` | `26.93%` faster |

The corrected result is `6.22%` slower than the non-promotable raw-log-prob
compiled result. Its training phases total `93.538358 s` versus pinned MLX
`27.769290 s` (`3.368x` slower), and evaluation is `87.347253 s` versus MLX
`38.714746 s` (`2.256x` slower). Its GRPO report and adapter SHA-256 values are
`3bf07b18e63707a755a311986258ea7c7b3e4e8401f2f79d3aa2df9ab60bb076`
and `d8c40074fa0a3e3d9784276c8ad19eea415377c8456f1904d05821eb6a59627c`;
the retained root is
`/private/tmp/antfly-gemma4-opt-20260826/grpo-e2b-token-shadow-8x16-r24`.
The next safe performance target is reuse of the eager batch-one execution
plan/session without changing allocation or reduction geometry. Raw compiled
log-probabilities are not an acceptable optimization target.

The exact-resume adapter was then materialized into the pinned E2B base. The
streaming merge copied and verified 1,961 base tensors, merged 50 LoRA tensors,
and published a `10,246,621,942`-byte model with SHA-256
`c04ffd7255c8fd189243ceeb7cc7b2b34b969afd714157b88c8a9747296c1919`.
Three independent fresh-process Metal reloads on the same deterministic prompt
all emitted token ID `1904` (`no`), stopped after one token, reproduced the
same attention dispatch, and reported zero process swaps. Their combined
semantic digest is
`4bfd387aa897cae44acad9e90e4b8e624e31e6f3fc16777d2c6db79c060aa068`.
The retained materialized model is
`/private/tmp/antfly-gemma4-opt-20260826/materialized-e2b-qualified-grpo-v14-r1`;
the token-ID timing records are the three
`materialized-generation-tokenids-v14-r1-run*.json` files in the same campaign
root.

The restart-safe executable-source diff, excluding this status document, is
SHA-256
`73b72f8a94351ff2c1c6ef8c3ea10fb38e70f40e8cb91927952090a5bf5f492f`.
Stable Zig 0.16.0 produced ReleaseFast binary SHA-256
`74d71f175cc70005db84adc48d43186a3a60e33b9067d18e49237f1a4b8f6af6`;
a fresh rebuild was byte-identical to the frozen binary. Debug, ReleaseSafe,
and ReleaseFast each selected 295 focused Gemma4 tests, passed 293, and skipped
the same two optional local-model fixtures. Dedicated Debug and ReleaseSafe
Gemma graph gates passed. The Python Gemma4 discovery suite passed all 602
tests, including the localhost warm-server case outside the socket-restricted
sandbox. `zig fmt --check` and `git diff --check` are clean. The installed
`0.16.0-dev.3144+ac6fb0b59` snapshot crashes on this tree and is not accepted
as a release toolchain.

The exact final binary passes the stronger independent-initialization E2B DPO
campaign for all three seeds at eight epochs and learning rate `1e-5`.
Held-out accuracy is
`0.60-0.80`, loss is `0.663723-0.687369`, and reward margin is
`0.013310-0.062920`, all improved from each seed's initialized baseline. Its
campaign report SHA-256 is
`b6c3e5f12b1e5805a0dd63a4710d7239776ea3224e11e693ab9ab23989d1a208`;
the root is
`/private/tmp/antfly-gemma4-opt-20260826/quality-e2b-dpo-lr1e5-74d71f17-r1`.
The three final adapter SHA-256 values are
`6adf382f5d90e786aa8db6034e8fa11a298f32440baba8d0a135faed40635f56`,
`3f1df992a18daaf9251cffb37fc7c698650c92243ad7750a51e24d032ddf628d`,
and `d8d02816fbe0dae2e6a00df2100165c416091fdf08baff69ac4d89cd069cce4b`.

The equivalent current-binary E2B GRPO gate remains negative evidence. The
seed-17 fail-fast campaigns at `5e-7`, `7.5e-7`, and `1e-6` all completed 64
optimizer groups without a KL-budget rejection, but baseline and final
mean/top-rank/positive-group rewards stayed exactly
`0.34375 / 0.5625 / 1.0`. At `1e-6`, training reward finally rose from
`0.28125` to `0.3125` in epoch eight, but held-out reward did not move; only
two lower-ranked zero-reward completions exchanged order. Held-out KL loss was
`0.0000847082` against the `0.004` ceiling, and maximum training mean KL was
`0.00453018` against the hard `0.1` budget. No failed run published an adapter,
and seeds 42/991 correctly did not run. Campaign report SHA-256 values are
`e951237500a305e5fdcae549e47a1bd7639347374028197e53006ad52d81037c`,
`873fc0daff9440cd9f3173eaf403059a499dcf0825f938f63a076e66968f991d`,
and `9b3ef1a87009f20f1b1b2bc3e922b2c5ca21617a977343db5b49196b4a452882`.
The gates were not weakened after observing the result, and learning rate is
not raised past the historical `1e-6` stability boundary. The next quality
experiment must increase representative train/evaluation resolution or use a
predeclared longer-horizon schedule, not tune against these 16 holdout groups.

Promotion status therefore remains **default-off candidate**, not production
default. Exact E2B resume and materialized reload are closed. Remaining
promotion blockers are baseline-relative E2B GRPO quality, E4B GRPO quality and
memory/performance evidence, a clean fresh-host zero-paging run, and the hosted
`gemma4-metal-training / macos-arm64-gpu` CI gate on an immutable commit.
During that campaign the 24 GiB host already had swap allocated and only about
5 GiB free, so it was not admissible for E4B or fresh-host attestation. Current
free space is lower, as recorded below.

## Algorithm Hardening Candidate (2026-08-26)

The current source closes the highest-risk semantic gaps identified in the
production review. It is an implementation candidate, not a promoted release:

- Gemma4 GRPO now samples from a seeded categorical distribution with typed
  temperature, top-p, and top-k filtering. SplitMix-derived logical streams
  bind seed, epoch, group, completion, and token position, so interruption and
  resume recreate the same samples without depending on process state. Duplicate
  samples are valid. Reports record the exact sampling controls and selected
  token/log-probability evidence.
- Zero-variance reward groups, groups with every completion masked as
  truncated, non-finite loss inputs, and KL-budget violations cannot silently
  mutate the optimizer. KL overflow skips the group by default or aborts under
  an explicit policy. Reports and checkpoints carry group counts and fractions.
- GRPO exposes group/no reward scaling, GRPO/BNPO/DR-GRPO/DAPO loss reductions,
  asymmetric `epsilon_high`, and whole-completion truncation masking. The core
  implements batch scaling for multi-group score fixtures, while model training
  rejects it until prompt groups share a real rollout batch. DAPO rejects
  gradient accumulation greater than one until a truthful cross-group
  active-token denominator exists.
- DPO exposes sigmoid DPO, conservative label smoothing, length-normalized IPO,
  and reference-free SimPO. A non-base-equivalent initial adapter is snapshotted
  as the frozen reference instead of accidentally comparing the policy to the
  raw base. ORPO/CPO/KTO remain typed unsupported errors because the paired
  trainer does not yet provide their differentiable auxiliary-SFT or unpaired
  contracts. DPO report/evaluation schemas are v7/v3.
- A final partial gradient-accumulation window is renormalized from the
  configured `1/N` micro-batch scale to its actual `1/M` mean before norm,
  clipping, and AdamW. The Metal path folds that correction into norm and
  optimizer scaling without adding a gradient-buffer rewrite command.
- The historical MLX GRPO comparators implement ranked candidate selection.
  They now fail closed when handed a stochastic Antfly report rather than
  presenting candidate-set overlap as native-rollout parity. A replacement
  comparator must implement the same categorical sampler or use predeclared
  distribution-level gates.

The stable Zig `0.16.0` focused gate now selects 301 tests: 299 pass and the
same two optional local-model fixtures skip. The direct preference-loss suite
passes 17 analytic, finite-difference, and fail-closed checks for DPO/cDPO, IPO,
and SimPO; the direct GRPO suite passes 14 checks. The current top-level Python
discovery passes 243 tests. The separate 405-case Gemma4 discovery passed 404
tests in the restricted sandbox; its sole localhost-bind permission error
passed when rerun with localhost access. The installed
`0.16.0-dev.3144+ac6fb0b59` compiler still crashes while compiling the focused
root and remains rejected as a release toolchain.

One real E2B diagnostic reached the strict Metal trainer with the new sampler.
Its first two-group input correctly failed `NoGrpoLearningSignal` because every
reward was equal. A disjoint two-group retry completed one optimizer group from
16 sampled completions and 63 completion tokens, with one zero-variance group,
zero sampling/rescore and policy/reference consistency errors, and a passing
held-out result (`0.1875` mean reward, `0.75` top-rank reward, `0.75` positive
group rate). Training sampling, policy rescore, reference scoring, and backward
took approximately `22.17 s`, `3.66 s`, `6.89 s`, and `9.35 s`; held-out
sampling and reference scoring took `43.68 s` and `13.23 s`. This proves the
real path executes and learns, not that it is competitive or release-qualified.

This host remains inadmissible for a fresh-host promotion attestation. At the
2026-09-01 final merge review the data volume had about `87 GiB` free, but
encrypted swap was already allocated (`2048 MiB` total, `747.44 MiB` used).
Do not use this process lifetime for a zero-paging release claim. The next
release evidence must come from a clean immutable source revision on a fresh
host, followed by three-seed E2B and E4B DPO/GRPO quality, exact resume,
cross-framework correctness, and same-workload performance distributions.

## Final Main-Merge Review (2026-09-01)

The merge conflicts are mechanically resolved with no unmerged index entries,
conflict markers, or diff-whitespace errors, but the merge remains intentionally
uncommitted. This review found and fixed five integration or fail-closed gaps:

- IPO no longer multiplies the completion-mean policy/reference log-ratio
  margin by DPO `beta` before comparing it with `1 / (2 * tau)`. When `ipo_tau`
  is omitted, the recipe maps `beta` to IPO tau rather than applying both.
- Paired preference loss now rejects invalid DPO/IPO scaling and non-finite
  log-probabilities. GRPO rejects invalid KL coefficients, non-finite
  log-probabilities/advantages, overflowing policy ratios, and non-finite loss
  or gradient intermediates before optimizer mutation.
- The shared graph LoRA injector rejects zero rank and non-positive or
  non-finite alpha before cloning or mutating a graph.
- The secondary GGUF test store implements main's new
  `preserveFileCacheOnDeinit` VTable contract; the full inference-root lifecycle
  build exposed this merge omission after the narrower Gemma4 target compiled.
- The macOS Gemma4 workflow now triggers on the shared DPO/GRPO/LoRA, CLI,
  tool, script, and fixture surfaces and explicitly discovers the nested
  `scripts/gemma4/gemma4` Python suite.

Pinned Zig `0.16.0`, `-j1`, strict Metal, and isolated caches pass the final
301-case focused gate in Debug, ReleaseSafe, and ReleaseFast (`299 passed`, two
optional-artifact skips in each mode). The Debug and ReleaseSafe graph target
passes `7/7`; the full ML graph build passes `534/534`; the full inference-root
ownership/lifecycle filter passes `4/4`; direct preference and GRPO suites pass
`17/17` and `14/14`; the top-level Python discovery passes `243/243`; the oracle
lock validates; and the workflow YAML parses. The 405-case nested Python suite
has the restricted-sandbox/localhost split result described above.

This closes the locally reproducible code and merge-integration findings, not
the production promotion. This exact source still lacks fresh BF16 E2B/E4B
multi-seed DPO/GRPO quality, current-algorithm interruption/resume and
materialized-reload evidence, pinned native/Metal/HF-or-MLX numerical traces,
repeated matched performance/memory distributions, a fresh-host zero-paging
attestation, and the hosted required macOS GPU check on an immutable revision.
The known baseline-relative E2B GRPO and E4B top-rank quality failures also
remain open. Therefore the branch remains a default-off production candidate,
not production-ready.

## Production Qualification Refresh (2026-09-01)

The final review used ReleaseFast Metal binary SHA-256
`b8bee85094859b80e36ab118bf46982d226e32c3bf23fc15749d2daacfd1ad45`.
The pinned official BF16 model revisions were E2B
`3e22461f65e89153144f8adb70e3b8c2cc9845a7` and E4B
`ee0ef6023621cff504d758262d4e04895a5af4a2`; their monolithic model
SHA-256 values were
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`
and
`cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`.
Both inventories and the rank-16 / alpha-32 Q/V adapter targets passed the
pinned oracle lock. The E4B DPO and GRPO recipes passed the public recipe
dry-run, but no E4B training campaign was admitted after the upstream E2B GRPO
quality failure and on a host that already had encrypted swap in use.

The current binary passed the real E2B DPO gate for independent initialization
seeds 17, 42, and 991. Held-out accuracy was `0.6`, `0.8`, and `0.8`; loss was
`0.6846736073`, `0.6753409505`, and `0.6649171114`; and reward margin was
`0.0187718216`, `0.0379370116`, and `0.0600500479`. Every metric improved from
its fresh initialized baseline and every final adapter digest was distinct.
Campaign report SHA-256 is
`ee632b74c5d14932908c48fb1e8dec10a3644418d92f38bb90e2d3dae97cdc50`;
the retained root is
`/private/tmp/antfly-gemma4-production-evidence/quality-e2b-dpo-b8bee850-r2`.

GRPO did not pass, and the failed gates were not weakened after observation:

- The first compiled 32-prompt/eight-epoch run contained only `7/256` useful
  reward-variation groups. This exposed a qualification blind spot: nominal
  horizon length alone did not prove a meaningful optimizer campaign. Report
  SHA-256 is
  `11934c8e9a333cb80f555469a2af2eff3fcfc616468d448f29bdace92c087796`.
- With group size eight, temperature two, and a two-token compiled rollout,
  seed 17 admitted `135/256` groups, but held-out mean reward fell from
  `0.5390625` to `0.533203125`; top-rank reward and positive-group rate stayed
  at `0.71875` and `0.75`. Report SHA-256 is
  `7bc6c9317430815ea3ae8dc9d8f735df6893bf4f91ad8211d7b54bbee3243aa6`.
- A broader one-token campaign used 128 balanced training examples and a new
  128-example validation slice with zero source-ID overlap against the prior
  64-example holdout. Its v2 materialization manifest SHA-256 is
  `07055940b1302a371330aca70da4a72c494c95bc766edb41bd9eadf45f2a7851`;
  train/eval JSONL SHA-256 values are
  `80f4ce1116f7bdbfccf9b30b7fcd5d6e46fa83378debf8fcff0d7c277715c752`
  and
  `8254b2d7712eb6f14ca5b75f200713759f1a2df6e18ceef1a0a666b1f74af2d3`.
  Under the explicit abort policy, the fourth observation of one repeated
  prompt reached raw mean KL `0.20524150` at logical group 479, exceeding the
  unchanged `0.1` hard budget before mutation. Report SHA-256 is
  `c6945becd3a418b01270391fd16df08b655502979a00fa8577c2cad1de3557e5`.
- Repeating that exact campaign with the documented safe `skip_group` policy
  completed training with `310/512` optimizer groups, `201` zero-variance
  groups, and only `1/512` KL rejection. Final held-out mean reward improved
  from `0.578125` to `0.58984375`, but top-rank reward remained `0.703125` and
  positive-group rate regressed from `0.7578125` to `0.75`. Final KL loss
  `0.0004560637` stayed below `0.004`, but the baseline-relative quality gate
  correctly failed. No adapter was published and seeds 42/991 did not run.
  The self-bound failure report and final evaluation SHA-256 values are
  `18e90ab0d23d7d2f54dd5637a423f4715ccf8df3b613e327eb7d7e5b9c6a96ae`
  and
  `06ebe21b1dadb0b693caf92841f78f20b9b1fe16b3eb0a9a76ddc0b8de15d545`.

Qualification is now fail-closed around those observations. GRPO campaigns
must account for the complete logical horizon, admit at least 25% of groups to
the optimizer, reject no more than 1% at the train-time KL budget, and match
KL-controller admissions to realized optimizer groups. CLI overrides may only
tighten those two production bounds. Both passing and failing reports bind the
quality qualifier, shared resume qualifier, environment policy, binary, model,
adapter, recipe, and exact train/evaluation inputs. BoolQ materialization v2 can
exclude exact source IDs from any number of digest-bound prior manifests; a
per-label offset alone is not accepted as evidence of holdout rotation when a
token-budget change alters admission.

After these changes, top-level script discovery passes `261/261`. Nested Gemma4
discovery passes 439 of 440 cases in the socket-restricted sandbox; the sole
localhost warm-server case passes independently with localhost access, for an
effective `440/440`. `git diff --check`, the unmerged-index check, and the
conflict-marker scan pass. The previously recorded current-source Zig gates
remain `299/301` focused cases (the same two optional artifact skips) in Debug,
ReleaseSafe, and ReleaseFast, plus `534/534` ML graph, `7/7` Gemma graph,
`4/4` lifecycle, `17/17` preference-loss, and `14/14` direct GRPO checks.

This refresh makes the implementation and qualification path safer, but it is
negative release evidence, not a production promotion. Current blockers are a
fresh three-seed baseline-relative E2B GRPO quality pass, equivalent current
E4B DPO/GRPO quality, accepted-adapter resume/materialized-reload and
HF-or-MLX parity, repeated matched performance/memory evidence, a fresh-host
zero-paging attestation, and the hosted required macOS GPU check on an immutable
revision. Do not enable GRPO by default or call this branch production-ready
until those gates pass.

## Production Qualification Closeout (2026-09-02)

The final source audit closes the remaining local merge, full-target optimizer,
and failed-campaign evidence gaps. It does **not** promote GRPO: the predeclared
real E2B all-linear campaign completed healthy optimizer work but failed its
strict held-out greedy-reward improvement gate. That gate was not weakened or
reinterpreted after observation, seeds 17 and 991 were not run after seed 42
failed, and the reserved final holdout remains unspent.

Full `text-all-linear` E2B rank-16 / alpha-32 LoRA contains 276 target modules
and 552 trainable A/B tensors. The first real GRPO attempt exposed that Metal's
single-dispatch gradient sum-of-squares bridge accepted at most 256 inputs and
therefore returned `TooManyTrainingSumSquaresInputs`. The public reduction now
chunks arbitrarily long input lists into bridge-sized batches, completes each
temporary result before release, accumulates batch totals in host `f64`, and
returns the established `f32` result. The focused Metal regression exercises
552 input tensors (`256 + 256 + 40`) before the AdamW update. The runtime's 256
input limit is now one shared constant rather than a duplicated magic number.

Baseline-relative DPO and GRPO rejection now preserves the complete typed task
report before returning the gate error. A failed gate leaves
`trained_adapter_dir` null and does not save or validate a trained adapter;
successful runs retain the previous save, validation, report, and print order.
The outer quality campaign records the active failed seed and hash-addressed
task, evaluation, baseline, runner, manifest, reward-trace, and KL-trace
artifacts. JSON summaries copy only allowlisted fields and refuse to parse an
input larger than 16 MiB; trace payloads remain external and are represented by
path, size, and SHA-256. This closes the prior observability hole where the
trainer returned the correct baseline-relative error but the outer campaign
could preserve only stderr and separately emitted evaluation files.

Pinned Zig `0.16.0`, strict Metal, isolated caches, and `-j1` pass the final
304-case focused gate in Debug, ReleaseSafe, and ReleaseFast: 302 passed and the
same two optional-artifact cases skipped in every mode. The new 552-input
gradient-norm case passes in all three. The clean ReleaseFast focused compile
took about three minutes at 9 GiB MaxRSS; the standalone server build took about
four minutes at 13 GiB MaxRSS. Top-level Python discovery passes 265 tests,
nested Gemma4 discovery passes 440 tests with localhost enabled, the workflow
YAML parses, and the final formatting, unmerged-index, conflict-marker, and
whitespace checks pass. The final Zig executable is a 30,396,304-byte Mach-O
arm64 ReleaseFast binary at
`/private/tmp/antfly-gemma4-production-build-report-final/install/bin/antfly-inference`
with SHA-256
`a653231a375b74a8d7be80a6d9e444adb4f615d1b2cb62c372b60a04802c46e1`.
Its finetuning help contract loads, and it accepts the reserved final recipe in
dry-run mode.

The decisive real-model run used official BF16 E2B revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`, model SHA-256
`2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc5503`,
and the 552-tensor all-linear adapter. Its exact ReleaseFast training binary was
SHA-256
`3391548dd58bdf318508e60d3283322a826f812868204adf1b3ea623f6c57de6`.
Seed 42 trained for one seeded-shuffle epoch over 1,024 BoolQ prompts with group
size 16, temperature 2, top-k 32, one completion token, BNPO, and learning rate
`1e-7`. Of 1,024 logical groups, 357 were optimizer-admitted, 663 had zero
reward variance, and four were rejected before mutation for exceeding the
unchanged raw mean-KL budget of 0.1. The optimizer-group rate was 34.86% and the
KL-rejection rate was 0.391%, so both predeclared training-coverage gates
passed. Informative groups were balanced across target labels: 183 `no` and
178 `yes`.

| Held-out metric | Initialized baseline | Trained policy | Decision |
| --- | ---: | ---: | --- |
| mean reward | `0.722412109375` | `0.730712890625` | improved |
| first/greedy completion reward | `0.75` | `0.75` | **failed strict improvement** |
| positive-reward group rate | `0.83203125` | `0.8359375` | improved |
| KL loss | `0` | `0.0004473277` | below `0.004` ceiling |
| mean KL | `0` | `0.0111831930` | diagnostic |

The exact greedy comparison gained one prompt and lost one prompt, producing
the observed net zero. The trained policy digest was
`645f57f0a07b0e1aa843dbfcb95f5114b6b088281f73eb6beb6828cab546af3b`.
The preserved campaign report is
`/private/tmp/antfly-gemma4-production-evidence/diagnostic-e2b-grpo-all-linear-train1024-observed-eval256-g16-t2-k32-max1-e1-lr1e7-v8-gradnorm552-3391548d/campaign_report.json`
at SHA-256
`60eadb4a92b4ff437f697e63d056749a665904effcf6d2869c8d64201f5b8cb5`.
The final evaluation and KL-control trace SHA-256 values are respectively
`74c5dd45afd18f763f1b06661cde1e29de93a437b7ea7857e1014c011680c46b`
and
`909bf7e4647d05cd0706108a5933d65c2b360962080256485fcf316662176d6b`.
This run predates the task-report-on-failure hardening above; no trained adapter
was published, but its outer report necessarily lacks the new bounded
`failed_run.task_report` entry.

The reserved performance-unobserved final split has 1,024 training rows and 254
balanced evaluation rows, zero train/evaluation source-ID overlap, and excludes
448 IDs from every previously observed evaluation manifest. Its manifest,
train, and evaluation SHA-256 values are
`6cc421a967769f6e62fe8069f2a6eb5f7aa545531bfc7a1513207cdfa7d0ef43`,
`14c7df586fb0cc10b5758049f9f4326b274f35927aae9faa3cbe76cd8512046b`,
and
`45c48f2e00525859f4de5313e689c190b9ecf8155b50e91e462cebd5cd16b6ea`.
The predeclared final recipe SHA-256 is
`b3ef9946a3f483ef53419ed026867d35b32292ca62ed05f9937246ac0a418901`.
It remains reserved for a principled algorithm or training-contract candidate,
not another post-hoc learning-rate guess.

That historical result is superseded for candidate selection by the fresh v7
diagnostic below. It remains useful failure evidence, but it is not the current
E2B GRPO readiness decision.

## E2B GRPO Quality-Gate Refresh (2026-09-02)

The real E2B all-linear Metal candidate now passes a fresh three-seed
diagnostic under `antfly_gemma4_preference_quality_campaign/v7`. This is a
meaningful quality-gate pass, not a release-holdout pass: the final split remains
sealed until it can be evaluated on a fresh host with zero swap.

Schema v7 makes the statistical unit match the independent evaluation unit.
For each seed, baseline and trained completions use common random numbers and
are reduced to one summed reward delta per prompt. Every seed must be
directionally positive (`wins > losses`), but an individual seed is not also
required to reach `p <= 0.05`. Once at least three seeds exist, each prompt's
delta is averaged across seeds and a one-sided exact sign test is applied over
the 256 independent prompts. This avoids treating the 4,096 correlated
completions per seed, or the three training seeds themselves, as independent
quality samples. The gate also requires strict mean-reward improvement and
non-regressing top-completion reward per seed, no more than one positive-
reward-group regression, at least 25% optimizer coverage, at most 1% KL
rejection, and evaluation KL loss at most `0.004`. This campaign's top-
completion reward improved strictly in all three seeds even though the
predeclared contract permitted a tie.

The diagnostic used:

- ReleaseFast Metal binary SHA-256
  `26963fdaa9dcc24fb6cbcb1161dbcca8ee91a955798e9d1b90b620ecef469526`;
- qualifier source SHA-256
  `bbe8b0763827d22a14480c7668e897a444a2ed12f6fbbae3f7dc38cdc511938d`;
- 1,960-row balanced training data SHA-256
  `c388cf422c4f1cbffb0bd07ebea3ea9f7ac05de5ac36809bfcfb33bd130c5cab`;
- fresh balanced 256-prompt evaluation data SHA-256
  `b61f5aa75c37b07ccc1e68098a40fd0eef6bfa659697ed78947379ee10e52beb`;
  the split excludes all 702 source IDs used by earlier evaluation and the
  reserved final split; and
- predeclared diagnostic recipe SHA-256
  `eb35f8dbb42af2a70c83d4448616d2ac43b942c74f9c30579de66b98c151b654`.

All three seeds passed the per-seed quality and training-health contracts:

| Seed | Mean reward improvement | Top-completion improvement | Positive groups | Prompt wins/losses/ties | Optimizer / KL-reject rate | Evaluation KL loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | `+0.01171875` | `+0.01171875` | `208 -> 213` | `46 / 16 / 194` | `36.33% / 0.255%` | `0.0004957861` |
| 42 | `+0.0048828125` | `+0.01171875` | `209 -> 208` | `23 / 14 / 219` | `35.71% / 0.306%` | `0.0004411542` |
| 991 | `+0.0068359375` | `+0.01171875` | `211 -> 210` | `36 / 14 / 206` | `35.61% / 0.510%` | `0.0004898956` |

After averaging each prompt's group-reward delta across the three seeds, the
campaign recorded 73 wins, 30 losses, and 153 ties, with net group reward
`+32`, mean completion-reward improvement `+0.0078125`, and one-sided exact
`p = 0.000013588206562850904`. The write-once diagnostic outer report is
`/private/tmp/antfly-gemma4-production-evidence/diagnostic-confirmation-e2b-grpo-all-linear-train1960-observed-eval256-fresh160-g16-t2-k32-max1-e1-lr5e8-paired-v7-26963fda-hostmetal-r1/campaign_report.json`
at SHA-256
`4a5ad06a7075d7ecfd6f4eab7d31deff8ca92a0fb2e7d9bb0f4ab83d87375d98`.
It binds the recipes, initialized and trained adapters, training/evaluation
reports, and reward/KL traces for seeds 17, 42, and 991.
Because the root is under `/private/tmp`, it is retained diagnostic evidence,
not durable release storage; promotion must archive the complete tree in the
immutable release artifact store and verify its recorded digests there.

A repeated seed-42 evaluation is a useful numerical caveat. Its trained adapter
and training reward/KL traces were byte-identical to the immediately preceding
campaign, while four of 4,096 final completion token IDs differed and the net
reward differed by one. Both executions used the same recorded strict Metal
policy. The cause is not proven; the evidence bounds the observed difference
but does not justify a claim of byte-reproducible fresh-process Metal sampling.
The prompt-level multi-seed diagnostic remains positive under either result.

The reserved final evaluation remains unopened at SHA-256
`45c48f2e00525859f4de5313e689c190b9ecf8155b50e91e462cebd5cd16b6ea`.
Its frozen recipe is SHA-256
`3cfde8d2c8c6fbfa93a8c9caa48973d94783deb96e35aeb5be4395022b6aa93e`.
After the diagnostic, the host reported 2,061.19 MiB of encrypted swap, so
running the final cell here would violate the predeclared zero-paging release
condition and spend the holdout on non-promotable evidence. Promotion remains
blocked on that fresh-host final run, current-source E4B quality, accepted-
adapter interruption/resume, pinned cross-framework numerical parity, repeated
performance/memory distributions, and required hosted macOS GPU CI on an
immutable revision. The current-source CLI serving build and loopback gate are
closed below. GRPO therefore remains default-off despite the diagnostic pass.

## Accepted E2B GRPO Deployment Diagnostic (2026-09-02 through 2026-09-03)

Seed 42 from the passing v7 quality campaign was selected before deployment as
the canonical default seed, not by choosing the strongest observed seed. Its
adapter SHA-256 is
`b17b52dea3de90e35e8d2fad6442448100fa072eb58f7bebb6d8ab0619459f8a`.
The streaming, dtype-preserving materializer merged all 276 LoRA tensors and
copied 1,735 base tensors. The resulting 10,246,621,942-byte checkpoint is
`/private/tmp/antfly-gemma4-production-evidence/deployment-diagnostic-e2b-grpo-seed42-all-linear-v7-26963fda-r1/materialized-model/model.safetensors`
at SHA-256
`12b238b351bf7b7377ab2fd4a84383087ed9eba1ec1bca59824bb6749b187448`.

Three fresh processes reloaded that checkpoint through the eager Metal path.
Every process produced public text `yes`, token ID `4443`, one completion token,
the `length` finish reason, and zero process swaps, with no runtime command-
buffer fallback. The write-once reload report is
`/private/tmp/antfly-gemma4-production-evidence/deployment-diagnostic-e2b-grpo-seed42-all-linear-v7-26963fda-r1/materialized-reload-capture-report-v2.json`
at SHA-256
`97937adf70513377d6d734e79bfdfd143f8616d0f55650b2586a5f59a1250f19`.

The first loopback CLI serving attempt exposed a real request-translation
defect: `inference generate --server --disable-thinking` did not serialize its
explicit thinking mode. The server therefore returned empty public content and
private reasoning `The` for the one-token request. A controlled raw-JSON A/B
probe returned that same split when the field was omitted and returned public
content `yes` with no private reasoning when
`chat_template_kwargs.enable_thinking=false` was present. The CLI bridge now
forwards explicit `false` and `true` values while preserving omission for the
default, with a three-case serialization regression test.

The corrected raw HTTP diagnostic then passed three serial requests. All three
returned HTTP 200, exact public content `yes`, no private reasoning, one
completion token, and identical semantic accounting. The resident server
selected Metal, emitted the expected three admitted and three executed prefill
plans plus four generation timing events including warmup, used no runtime
command-buffer fallback, recorded zero child-process swaps, and did not grow
host swap. The materialized checkpoint remained byte-identical. The report is
`/private/tmp/antfly-gemma4-production-evidence/deployment-diagnostic-e2b-grpo-seed42-all-linear-v7-26963fda-r1/http-serving-diagnostic-report-v3.json`
at SHA-256
`ed682f60985f6acf3be0001b182a55e9aa9fb3ff4b9789a78c9153aa5276c31f`.

The installed development compiler `0.16.0-dev.3144+ac6fb0b59` terminated
current-source Debug and ReleaseFast builds with compiler `SIGSEGV`/`SIGBUS`.
An isolated baseline compile with this serving patch removed failed identically,
so the crash was not evidence against the patch. The pinned stable Zig `0.16.0`
toolchain then closed that build blocker: with `-j1`, ReleaseFast, and Metal
enabled it produced the 30,396,320-byte current-source executable
`/private/tmp/antfly-gemma4-production-build-e2b-v7-serving-fix-zig0160/install/bin/antfly-inference`
at SHA-256
`2ddb7517d954f6688ec7df6ee42ddfe3a9d5a190292ca22c6957fefdccefc502`.
The explicit-thinking serialization regression passes in Debug and in a fresh
isolated ReleaseFast Metal test build. The complete post-fix ReleaseFast Metal
Gemma4 gate also selects 304 cases, passes 302, and skips only the same two
documented optional-artifact cases.

That current-source executable passed the CLI-facing loopback gate, not just
the raw-JSON control. Three serial invocations of
`inference generate --server --disable-thinking` returned exact public content
`yes`, no private reasoning, one completion token, and `length`; all exited
zero. The server selected Metal, recorded three admitted and three executed
prefill plans and four generation-timing events including warmup, used no
runtime command-buffer fallback, and recorded zero child-process swap delta.
The executable, source file, and materialized checkpoint remained byte-identical
before and after the gate. The report is
`/private/tmp/antfly-gemma4-production-evidence/deployment-diagnostic-e2b-grpo-seed42-all-linear-v7-26963fda-r1/http-serving-diagnostic-report-v4.json`
at SHA-256
`4b30d5f5e2279adeea7052b581ced59f0093bdd3c083571a05240909330ca683`.

These are paged-host diagnostics under `/private/tmp`, not immutable release
evidence. Host swap was unchanged across the current-source serving gate but
remained 1,909.12 MiB allocated. The current top-level Gemma4 Python discovery
passes `264/264`; nested Gemma4 discovery passes `440/440` after its sole
localhost integration case is run with loopback permission; and the focused
quality/materializer/resume selection passes `60/60`. Keep the final holdout
sealed until a fresh zero-swap host is available.

The accepted seed-42 quality trajectory itself is one epoch, and the previous
recovery qualifier required at least two epochs with epoch-boundary-only
interruption, so no recovery evidence could apply to it. That contract gap is
now closed in source (2026-09-03): preference training writes durable
mid-epoch checkpoints on an eager-only `checkpoint.every_examples` cadence,
and `qualify_gemma4_preference_resume.py` gained `--interrupt-after-examples`
with lexicographic `(epoch, cursor)` boundary tracking. A real one-epoch E2B
GRPO qualification then passed on this host with the exact accepted model
snapshot, seed-42 bootstrap adapter, all-linear preset, and v7
hyperparameters on a 24-group subset of the accepted training data and the
pinned fresh 256-prompt evaluation split (16 evaluated groups): SIGTERM at
the durable `(epoch 0, cursor 8)` v2 sidecar, resume, and byte-identical
final trainer checkpoint, sidecar, and 105,409,779-byte adapter
(`sha256:14ef8aa64279298b259...`), with the semantic trajectory exact except
the pre-declared bounded terminal Metal GRPO KL floats. The report is
`/private/tmp/antfly-gemma4-midepoch-resume-e2b-r1/qualification_report.json`
at SHA-256
`e160ef9347fdc0de7dece5f5a525f4c204972e85bde60455f57529e0551ccd5c`, produced
by the current-source stable-Zig ReleaseFast Metal binary at SHA-256
`546761c17e4cce3997f3c52876bca2ae739dbf0f9e2bf490965a3e1629ce1737`. The
current-source Debug and ReleaseFast Metal Gemma4 gates select 306 cases and
pass 304 with the same two documented optional-artifact skips, the Gemma4
Python discovery passes `267/267`, and the resume-qualifier unit suite passes
`25/25` including stub end-to-end mid-epoch DPO and GRPO cells. This is
paged-host contract evidence for the mid-epoch recovery surface, not yet
recovery of the accepted adapter itself: that still requires interrupting and
resuming the full-length 1,960-group accepted recipe and matching its
recorded adapter digest, which remains a deliberate scheduled run.

## Final Local PR-Readiness Audit (2026-09-03)

The remaining locally actionable contract gaps are now closed in source. First,
`qualify_gemma4_preference_resume.py` publishes a v4 report that retains and
re-hashes the exact interrupted checkpoint plus content-addressed preference
sidecar after the resumed child finishes, validates the checkpoint's structural
counters and zero final cursor, and optionally requires the final adapter's
predeclared SHA-256. The real 24-group E2B run above remains subset-scale
evidence; the full 1,960-group accepted-recipe digest match is still required.

Second, the Zig numerical-oracle producer is implemented. The typed trainer's
paired private `--oracle-request` and `--oracle-capture-out` boundary admits
only the locked one-, two-, or eight-step AdamW trajectory. It captures a
deterministic logit projection at every supervised causal position, final raw
F32 gradients, post-update weights and Adam moments, and strict execution
counters. On Metal, a read-only final-gradient observer preserves the normal
direct-device AdamW path; it does not select the accumulation/reduction path.
`export_gemma4_lora_zig_oracle.py` independently binds the clean source
revision, release executable, lock, model, prepared row and raw source,
provenance-bound adapter, target inventory, checkpoint counters, capture file
set, and no-fallback Metal evidence before publishing a closed
`gemma4_oracle_trace/v1` directory. The required-device tiny-BF16 CLI
integration exercises that complete producer path.

Third, the v2 BoolQ materialization contract is consumable by both MLX parity
runners. They retain compatibility with historical v1 manifests, but current
v2 inputs must pass the exact selection/exclusion contract and their
self-attested semantic SHA-256 before any dataset or model work begins. This
closes the handoff that otherwise prevented the full 1,960-group E4B v7-style
reference rerun from starting. Failed quality campaigns now retain the
baseline evaluation trace plus child stdout/stderr alongside the already-bound
candidate evidence, and GRPO loss assembly rejects a finite-component sum that
would overflow `f32` instead of publishing an infinite loss.

The Antfly and MLX performance runners accept both historical v2 and current
v3 provenance-bound internal adapter manifests. Direct regression coverage
keeps the internal tensor-key layout compatible with v3 while unknown policy
sources and mismatched layouts remain fail-closed.

Finally, all Gemma4 LoRA/DPO/GRPO scripts, schemas, locks, requirements, and
tests now live under `scripts/gemma4/`, matching the family layout introduced
by `main`. Every internal path identity, workflow command, schema constant,
testdata lookup, and documentation command was updated with the move; the
combined family suite exercises the relocated tools together with the merged
serving and CUDA tooling.

The current local source passes all three revision-bound, required-device
focused Metal gates with the pinned stable Zig 0.16.0 toolchain and `-j1`:
Debug, ReleaseSafe, and ReleaseFast each select `311` tests, pass `309`, and
skip only the same two documented optional model-artifact cases. The embedded
source revision is now mandatory in required-device mode, so the private
oracle capture integration cannot silently go unexercised. The separate Debug
and ReleaseSafe Gemma4 graph targets, ML graph unit suite, and four selected
Metal lifecycle regressions also pass. Relocated Gemma4 family discovery passes
`723/723` on the host, including the feature-specific LoRA/DPO/GRPO contract
suites; the known localhost integration cell was run with loopback access after
the restricted sandbox rejected the bind. The focused resume
qualifier passes `28/28`, and the new Zig-oracle exporter suite passes `7/7`. Relevant
Python modules compile; the checked-in oracle v3 lock validates at SHA-256
`c848acb5fa38abda012f52c31cc122927b26775896d4a460e58cc9336cf27383`; and
`git diff --check` is clean.

This makes the Gemma4 change set **locally PR-ready**, subject to staging the
remaining intended files and completing the already-open merge on one clean
revision. It does not promote the feature to production. No locked real-model
Zig trace can be validly produced from the current dirty/in-progress merge, and
the paired HF/PEFT trace still needs the pinned CUDA environment and model
snapshots. The final 2026-09-03 host preflight reports `1,669.12 MiB` swap in use
and only `31 GiB` available on the data volume, so the sealed E2B holdout, E4B
v7 campaign, and other zero-paging release cells were deliberately not run.

## Production Roadmap and Release Gates

1. **Keep one green product contract.** Compile the typed CLI, four-step
   recipe/workflows, v6 admission, PEFT sidecar, checkpoint/resume, and strict
   Metal path together; require focused Debug, ReleaseSafe, and shipping-mode
   macOS gates from the final clean branch. Snapshot the numerical-kernel
   admission policy once per run, bind its fingerprint into the run manifest
   and telemetry, and reject policy drift before optimizer mutation.
2. **Expand the oracle and real-data preference loops.** The bounded E2B/E4B
   UltraFeedback DPO and BoolQ GRPO comparisons are archived with disjoint
   evaluation and matched MLX evidence. Three-seed/eight-epoch absolute-floor
   campaigns now pass for E2B/E4B DPO and E2B GRPO; the legacy tiny-recipe E4B
   GRPO campaign fails its predeclared top-rank floor. A follow-up independent
   MLX-LM diagnostic also remained flat on that legacy workload and produced
   zero top-rank improvement; because no immutable MLX diagnostic bundle was
   retained, that narrows the failure away from an Antfly-only defect but is
   not promotion evidence. Independent-initialization, baseline-relative
   E2B DPO passes. The current compiled E2B GRPO candidate passes the v7 fresh
   diagnostic three-seed prompt-level gate, while its separate final holdout
   remains sealed for a zero-swap host. The Zig trace producer now exists; next
   run that frozen final cell, rerun E4B on the full 1,960-group v7-style recipe,
   and run pinned HF/PEFT one-, two-, and eight-step traces against Zig native
   and Metal for both target presets.
3. **Finish optional-lane provenance.** Bind teacher targets to their
   teacher/base identity and extend the closed run ledger when optional
   artifacts are admitted.
4. **Complete artifact transactions.** Add a typed staged-artifact evaluator,
   stale-staging recovery, and power-loss failure injection, and replace
   whole-buffer prepared JSON with immutable streaming shards.
5. **Extend the qualified durable-resume surface.** Real E2B and E4B BF16 Metal
   SFT/DPO/GRPO interrupted-and-resumed training trajectories now match
   uninterrupted execution exactly, with byte-identical adapters; terminal
   GRPO KL floats carry the separately attested narrow GPU-evaluation bounds
   above. Checkpointed incremental-KV GRPO also passes E2B/E4B epoch-boundary
   recovery with cumulative telemetry and rebuilt transient pages. Experimental
   canonical direct-GGUF E2B SFT, DPO, and GRPO have recovery passes. The
   accepted one-epoch GRPO adapter justified mid-epoch scheduling: eager
   preference training now writes durable `every_examples` mid-epoch
   checkpoints, and a real one-epoch E2B GRPO mid-epoch interruption/resume
   passes byte-identically on a 24-group subset of the accepted recipe.
   Compiled-sampling and incremental-KV lanes stay epoch-boundary-only, and
   add retained generations only if operational evidence justifies the extra
   state surface.
6. **Finish production-shape compute.** The gate/up backward-input sum is now a
   qualified default runtime region at rows 64, 128, and 512, and frozen-head
   fused linear cross-entropy now owns the strict-Metal hard-label and uniform
   DPO sequence objectives without global logits. Independent-row SFT length
   buckets and pair-safe DPO buckets are qualified opt-ins with fixed-shape
   rollback, bounded graph caching, and real E2B/E4B evidence. Next build a
   memory-safe whole-pair or segment-aware packed DPO graph to close the E4B
   Q128 gap, finish group-safe GRPO scheduling, and coalesce the forward
   gate/up projection and saved-value path without violating autodiff
   lifetimes. Extend the fused loss reader to packed Q4_0, Q4_K, and Q6_K only
   when direct QLoRA admission is ready. Prove native parity and peak-memory
   bounds at intended E2B/E4B sequence lengths before promotion.
7. **Extend the same-Mac performance baseline.** The realistic fixed/Q128
   E2B/E4B DPO matrix and four-token GRPO diagnostics now establish bounded
   single-process baselines. The final-source refresh finds MLX faster in all
   four DPO cells; Antfly retains a material memory advantage except at E4B
   Q128, where footprint is effectively tied and slightly worse. Restart-safe
   compiled E2B GRPO is `1.196x` faster than eager Antfly but remains `3.368x`
   behind MLX in training phases. Reuse the exact eager batch-one rescore plan,
   and repeat the DPO matrix as an alternating five-process distribution on the
   pinned host before making a stable performance claim.
8. **Real E2B acceptance.** Pinned BF16 DPO/GRPO pass bounded historical
   absolute quality floors and exact epoch-boundary recovery. Baseline-relative
   DPO passes three independently initialized seeds. Compiled GRPO now passes
   three independently initialized seeds on the fresh schema-v7 diagnostic;
   its frozen final holdout must pass on a fresh zero-swap host. Then add the
   remaining native-versus-Metal/HF per-target parity cells before a broad
   claim.
9. **Real E4B acceptance.** Pinned BF16 DPO passes the bounded multi-seed gate
   and incremental GRPO recovery is exact, but the archived tiny-recipe GRPO
   campaign misses its predeclared top-rank quality floor. The same lack of
   learning reproduced in an independent MLX-LM diagnostic, while E4B was never
   run on the 1,960-group v7-style recipe that made E2B pass. Run that campaign
   on a fresh zero-swap host, then repeat the broader E2B gates with sequences
   and target presets that exercise shared KV, PLE, and the larger
   adapter/optimizer footprint without fallback or unbounded growth.
10. **Deployment and materialization.** The streaming, dtype-preserving writer
    now accepts sharded Safetensors and validates its staged output. The
    baseline-relative quality-accepted E2B GRPO adapter passes streaming merge,
    three deterministic fresh-process reloads, three serial raw-HTTP Metal
    serving requests, and three serial current-source CLI requests through
    `inference generate --server --disable-thinking`. The mid-epoch recovery
    contract required for a one-epoch trajectory is implemented and passes a
    real subset-scale E2B qualification; next interrupt and resume the exact
    full-length accepted recipe and match its recorded adapter digest, archive
    the complete accepted evidence tree immutably, and repeat the quality,
    disk/memory, and repeated-generation gates on E4B. Separately design any
    QAT-Q4/GGUF merge contract.
11. **Finish optional direct Q4/QLoRA qualification.** The pinned official E2B
    Q4_0 inventory now passes strict optimizer and process-kill/resume gates for
    canonical SFT/DPO/GRPO without host dequantization. Direct-GGUF GRPO plus
    incremental KV remains rejected after exact token divergence. Next prove
    peak memory, MLX/HF multi-step parity, multi-seed task quality, adapter
    reload generation, and E4B GGUF. Until those artifacts pass, the public
    `qlora-sft` recipe remains unsupported.
12. **Scale model-based GRPO rewards deliberately.** The typed weighted-rule,
    pinned external-verifier, and pinned `model-command` contracts now provide
    bounded failures, artifact/tokenizer/template/calibration identities,
    response attestation, subprocess integration coverage, structured failure
    traces, and replayable train/eval evidence. Keep built-in exact,
    case-insensitive, prefix, and token-match modes scoped to verifiable tasks.
    Add true request batching only with an explicit resource and ordering
    contract; the current implementation correctly enforces
    `max_batch_size = 1`.

Finally, run the macOS real-GPU workflow, make it a required branch-protection
check, and archive its artifacts. Keep separate opt-in gates for pinned real
E2B/E4B models because the synthetic runner job is not a production-scale test.

Do not call the path production-ready until every gate above has a reproducible
artifact, pinned model/dataset provenance, and an explicit pass threshold.

## Merge refresh (2026-09-09)

Resolved 31 conflict blocks across the five inference/Metal files while merging
`26e332ed8c335d75eecd875548c6c36bcb47d183` into branch head
`3fd4d22304`. The resolution preserves Gemma4 training kernels and content-based
RMSNorm slot identity alongside F16 embedding/linear paths, cached MPS views,
weight-handle ownership, ModernBERT normalization, and serving admission floors.
It also removes duplicate automatically merged declarations and reconciles the
attention call signature and owned-backend factory.

GPU validation exposed a cleanup hang when a synchronous weight store had no
prefetch queue. Both per-handle and dense-cache pin release now guard the queue
lock just as loading does. The lifetime tests cover dense and quantized weights
with and without a queue, including allocation failures. Transient RMSNorm slot
retirement now uses the same content key as slot preparation and has a regression
covering reconstructed equal-content tensors and slot reclamation.

Local validation with pinned Zig 0.16.0 and actual Metal access:

- Gemma4 ReleaseFast gate: 314 selected, 312 passed, two optional fixture skips.
- Expanded inference Debug gate: 247 selected, 245 passed, two optional model
  skips; generated-kernel companion checks also passed 23/23.
- Gemma4 Python discovery: 723/723 passed, including localhost server harnesses.
- ReleaseFast Metal CLI build and both top-level/finetuning help smokes passed.
- Resolved Zig files pass formatting; tracked text has no conflict markers;
  worktree and staged whitespace checks pass.

Logs and the resolution patch are retained under
`/tmp/antfly-merge-prready-20260909/` as local review evidence. The merge was
subsequently committed as `6c36b2e015a9f18f6339173cf748e322a22ebab0`, with no
unmerged index entries. These checks establish local source readiness only;
they do not refresh the real-model production qualification described above.

### Final review fixes (2026-09-09)

Checkpoint path isolation now resolves existing symlinks before interpreting
parent traversal. An `alias/../model.safetensors` checkpoint can no longer pass
preflight as an unrelated path when `alias` points inside the immutable model.
Missing suffixes containing `.` or `..`, and dangling symlinks, fail closed;
ordinary missing suffixes still normalize repeated separators. Regression tests
cover absolute and relative paths, existing and missing destinations, dangling
links with trailing separators, and checkpoint rejection without modifying the
input file or creating an output directory.

The GRPO planning test now checks Metal admission only when Metal is compiled,
and expects `BackendUnavailable` otherwise while retaining the remaining native
contract assertions. The bootstrap/output collision fixture creates the parent
directory it traverses, so it tests a real filesystem alias.

Validation with Zig 0.16.0:

- Both new path regressions fail against the original helper; all five standalone
  path-isolation tests pass against the fix.
- Gemma4 Debug without Metal: 298 selected, 274 passed, 24 skipped, no failures.
- Strict actual-Metal Gemma4 Debug: 317 selected, 315 passed, two optional fixture
  skips, no failures. The strict run embeds the base merge revision, with the
  uncommitted source patch retained separately.
- Formatting, whitespace, and unresolved-index checks pass.

Fix validation logs and the source patch are retained under
`/tmp/antfly-review-fixes-20260909/`. These results close the two local review
findings without refreshing the real-model production qualification gates.

### Qualification restart (2026-09-09)

The new E4B GRPO acceptance and native/Metal/HF parity campaign is in input
recovery and preflight. The old `/private/tmp` campaign directories remain, but
their model, adapter, report, prepared-input, and environment files are missing.
The recorded commands recover the two frozen recipes byte-for-byte and recreate
the 1,960 training rows, 256 fresh diagnostic prompts, and 254 reserved final
prompts at the exact SHA-256 values recorded above. The fresh diagnostic IDs
exclude all 448 previously observed IDs plus all 254 reserved IDs. The reserved
split has been reconstructed and hashed, but has not been evaluated.

Pinned model restoration now targets the durable, Git-ignored
`.benchmark-assets/gemma4-qualification-20260909/` directory. The oracle lock
validates, and the 1,536-row numerical contract fixture regenerates exactly.
The predeclared plan covers E4B/E2B diagnostic and final campaigns plus 36
native/Metal/HF traces and 36 comparisons across both target presets and the
locked 1/2/8-step AdamW trajectories. Full-length recovery is tied to the
current accepted seed-42 adapter before interruption.

The latest preflight still reports 687.62 MiB host swap on the 24 GiB Mac, and
the reviewed source changes remain uncommitted. Zero-paging acceptance therefore
has not started; the Zig oracle also requires a clean source revision, and the
HF/PEFT oracle needs an available CUDA host with the pinned environment. No
quality, parity, or recovery gap is claimed closed by this preparation.

### Qualification restart: committed build and checkpoint reader fix (2026-09-09)

The user committed the reviewed changes as
`e7edd66f399edef41691a71be1530a86722bf4d7`. The clean-source ReleaseFast Metal
build passed with the pinned Zig 0.16.0 toolchain. Its executable SHA-256 is
`4a0f8d0f2aa9175bffa58729b17f8cb8b95dd463ed637a5bd1cc65abef651d7e`.
Both locked E2B/E4B model snapshots were restored and hash-verified; both
model-specific oracle datasets and both rank-16 target presets were prepared.

The pinned stock PEFT 0.19.1 export/load/save/reload smoke passed against that
binary, preserving the tiny fixture's logits exactly at both boundaries.
This establishes adapter interoperability for the smoke's scope, not real
Gemma4 training parity or general PEFT feature parity.

The first real E2B q/v one-step Metal oracle exposed a reader defect:
`RealAutodiffTrainer` serializes checkpoint weights and optimizer slots as
flat F32 vectors, but the Python oracle expected matrix-shaped storage.
The reader now checks the exact flat element count and dtype, then reshapes
using the independently validated adapter dimensions. The regression fixture
now matches the actual writer. Wrong-rank, wrong-length, and wrong-dtype slots
remain rejected; numerical tolerances are unchanged. The corrected fixture
failed before the fix, and all 26 oracle-related Python tests passed afterward.
The broader Python suite passed all 724 tests: 723 in the sandbox and the one
loopback-server test on a separate permitted rerun after its bind was blocked.

A retained real E2B Metal diagnostic validated the corrected reader and
packaged all 100 adapter tensors and 500 trace tensors, including raw gradients
and optimizer state. Its one-step loss was `1.4149742126464844` and raw gradient
norm was `2.37408334452001`. These values are capture evidence, not a quality
or parity pass. The diagnostic uses the committed executable and records the
modified Python reader's digest; it does not publish an immutable oracle
`COMPLETE.json`.

The matching native diagnostic was terminated by its resource guard after
12.57 seconds when observed host swap grew by 2,001.06 MiB. No native trace or
native/Metal comparison completed. The 24 GiB M4 Pro began this qualification
with 689.75 MiB of swap already in use, so the frozen zero-paging quality and
acceptance campaigns remain unrun, and the reserved final holdout remains
unevaluated. Continue native/Metal captures on a host with sufficient memory;
the separately pinned HF/PEFT numerical reference, full-length accepted-adapter
resume, and hosted CI also remain outstanding. Antfly CUDA implementation is
outside this qualification's requested scope.

Durable build/preparation records, the PEFT smoke, and both raw diagnostics
are under `/Users/tim/Documents/af/antfly-qualification/20260909/`. The reader
fix and this status update must be included in the next source revision before
producing immutable oracle bundles with the corrected exporter.
Run plans, reconstructed recipes, source-ID exclusion checks, and current
preflight evidence are under
`.benchmark-results/gemma4-qualification-20260909/QUALIFICATION.md`. Numerical
outputs are planned outside the source tree at
`/Users/tim/Documents/af/antfly-qualification/20260909/` as required by the oracle.


### 2026-09-09 MLX drift investigation (diagnostic, not acceptance)

Restored the pinned MLX-LM source at
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd` and ran it with MLX/MLX-Metal
0.31.2 on this Mac. These wheel-based runs lack the release build attestation.
They do not replace the locked CUDA/BF16 HF/PEFT oracle or zero-paging GRPO
acceptance. The reserved final BoolQ split remains unevaluated.

The fresh E2B `peft-qv` comparison binds the same restored Google weights,
prepared token IDs and labels, and rank-16/alpha-32 initial adapter to the
committed-source Metal capture. The first adapter has zero B matrices. The
forward mismatch therefore precedes an optimizer update:

| Execution | Initial mean supervised CE |
| --- | ---: |
| Antfly Metal, committed binary | 1.41497421 |
| MLX stock BF16 activations | 0.92414445 |
| MLX F32 activation diagnostic | 0.92415619 |
| Pinned Transformers 5.5.2, CPU BF16 base forward | 0.93010724 |

The independent HF CPU check is a base-forward diagnostic, not a PEFT training
or release-oracle pass. Its sampled logits agree with stock MLX at cosine
0.99972751 and relative vector error 0.02456654. Against Antfly, the MLX F32
activation run has sampled-logit cosine 0.59124101, relative vector error
0.97956205, and maximum absolute difference 33.54383945. Its raw-gradient
cosine is -0.01134146 and relative vector error is 1.04546500.

With AdamW bias correction explicitly enabled, the F32 MLX update has cosine
0.00467156 and relative vector error 1.41129308. The difference between update
norms is only 0.00054805. The historical field
`delta_l2_relative_difference` measured this magnitude-only difference; it
was never the relative vector distance. The comparators now additionally emit
`delta_vector_l2_relative_error` and `delta_vector_l2_error`, preserving the
historical field and all existing gate thresholds. Equal-norm orthogonal and
opposite-vector regressions protect the distinction.

Disabling BF16 embedding-row staging did not change Antfly's loss or raw
gradient norm. Padding the MLX F32 forward to 512 tokens likewise
did not resolve the logit mismatch. That padded run was terminated by the
swap-growth guard during gradient evaluation; its forward result is usable,
but no completed gradient or optimizer result is claimed. The first stock MLX
diagnostic accidentally used MLX's default uncorrected AdamW; only its forward
and raw gradients are valid comparisons. Its optimizer results are excluded.

A derived one-layer pretrained fixture, with the first PLE slice and original
sliding-attention semantics retained, fits native and Metal. One-step initial
losses were 20.70525932 (native) and 20.70529556 (Metal), versus 22.22519684
(HF BF16) and 22.23039627 (HF F32). A four-token intermediate-value capture then localized the issue to RoPE
position assignment. The graph passed `[S*B*N, D]` rows to a backend contract
that infers heads per token from row width. Both native and Metal consequently
rotated individual heads as consecutive positions. The fix passes
`[S, B*N*D]`, preserving the position-major layout and restoring the original
attention shape afterward. A multi-batch/multi-head graph regression checks that contract; a numerical
regression exercises the actual native RoPE boundary and checks every head
against explicit token-position rotations. The four-token native prefix
rerun reduced the final-hidden absolute-sum discrepancy against HF F32 from
23.22117684 to 0.00116684. These are stage fingerprints, not exhaustive tensor
parity; full-model comparisons are recorded below. The fixture is diagnostic and cannot
qualify the full E2B model.

Evidence is retained under
`/Users/tim/Documents/af/antfly-qualification/20260909/diagnostics/`, including
`e2b-qv-step1-metal`, `mlx-e2b-qv-step1-f32-bias-corrected`,
`hf-e2b-base-cpu`, and `e2b-prefix-1/disjoint`. Reproduction scripts and guarded
execution logs are under `.benchmark-results/gemma4-qualification-20260909/`.
The legacy GRPO comparison's ranked sampler differs from the current seeded
categorical sampler; no native-rollout parity refresh is claimed from these
SFT diagnostics. Full E4B GRPO acceptance remains pending. The original numerical mismatch
was an observed correctness blocker, not merely missing evidence.


The layout-only full E2B rerun reduced initial CE to **0.92412513**, versus
MLX F32 **0.92415619**. Sampled-logit cosine improved to **0.9999999275**,
relative vector error to **0.00038125**, and maximum absolute difference to
**0.02382123**. Gradients still differed (cosine **0.79235790**, relative
vector error **0.65790286**), motivating a separate backward-path check.

That check exposed an independent inverse-RoPE defect: autodiff negated the
sine input, but native and Metal reconstruct rotation angles from attributes
and ignore the sine table. The analytic gradient regression failed with
`actual +0.84147096` versus `expected -0.84147096` at token position one.
Autodiff now negates `freq_scale` as well as the sine table, so both runtime
representations encode the inverse angle. The focused native regression then
passed (12 linked checks). Full-model results with both fixes follow below.


With **both fixes**, the same E2B one-step Metal capture versus retained MLX
F32 activations produces:

| Metric | Before fixes | Both fixes |
| --- | ---: | ---: |
| Mean supervised CE (MLX: 0.92415619) | 1.41497421 | 0.92412513 |
| Sampled-logit cosine | 0.59124101 | 0.99999993 |
| Raw-gradient cosine | -0.01134146 | 0.99995060 |
| Raw-gradient relative vector error | 1.04546500 | 0.01057247 |
| Update cosine | 0.00467156 | 0.99511984 |
| Update relative vector error | 1.41129308 | 0.09879426 |

This is a substantial correction, **not a locked numerical-parity pass**.
The F32 activation variant is a diagnostic precision alignment. A fresh stock
MLX BF16 step with explicit AdamW bias correction has loss 0.92414445,
sampled-logit cosine 0.99986742, gradient cosine 0.99222286, and update cosine
0.92838995. Its gradient and update relative vector errors remain 0.13117116
and 0.37843476. Reporting only the near-identical losses or update norms would
hide those differences.

An independent NumPy F32 AdamW replay using the **Zig raw gradients** matches
the captured first update at relative vector error **2.04816e-7**, maximum
absolute difference **9.31323e-10**. Comparing Zig and MLX F32 gradients,
3,899 of 1,449,984 nonzero gradient elements have opposite signs (0.26890%);
those coordinates account for **99.3000%** of the squared update discrepancy.
Their median absolute Zig gradient is 6.51317e-7, versus 7.37123e-5 across
nonzero coordinates. This directly identifies first-step Adam amplification
of small gradient differences as the remaining update-error mechanism for
this fixture. It does not qualify later optimizer steps or explain every
source of the remaining forward/gradient rounding difference.

The final diagnostic executable is ReleaseFast Metal, SHA-256
`bdf64c876fff11855429c7ce0033e2d9a0d5d3d0105cf364c53c73557f687678`, based
on `e7edd66f399edef41691a71be1530a86722bf4d7` with the retained uncommitted
Zig patch SHA-256
`871e507c3a73ecd263b6521695d6fb6e04d1a2739106123c2ac9199cd5bf50fd`.
The raw output is `diagnostics/e2b-qv-step1-metal-rope-layout-vjp-fixed`.
At that checkpoint, the build's recorded Zig patch matched the Zig diff. No immutable
oracle completion was produced.

Validation after both fixes: required-device Metal Debug **317 passed,
two optional skips**; no-Metal Debug **276 passed, 24 skips**; Python
**728 passed**; ML library tests passed. The initial strict-device invocation
omitted its required embedded-revision flag and rejected the CLI test; the
correctly configured rerun passed. The inverse-RoPE analytic regression
failed with the original VJP and passed after the fix. Formatting and Git
whitespace/conflict checks pass. No Git mutation was performed.

E4B full-recipe GRPO acceptance, the current categorical-sampler MLX rollout
comparison, the complete 1/2/8-step native/Metal/HF matrix, and accepted-adapter
full-length recovery remain pending. The latest stock-MLX preflight had
3,315.75 MiB swap in use, so this Mac remains outside frozen zero-paging
acceptance admission. The final holdout remains unevaluated.


The final derived-prefix rerun with both fixes completed on both backends:
initial CE was **22.23034477 native**, **22.23071098 Metal**, and
**22.23039627 HF F32**. Native/Metal adapter-update cosine was **0.99873925**,
with **0.05021295** relative vector error. This confirms the large forward
mismatch is corrected on the native path too, while preserving visibility
into the residual update discrepancy. The one-layer fixture does not replace
the full native E2B/E4B numerical matrix.

The investigation handoff, reproduction scripts, source snapshots and patches,
validation logs, binary identities, and artifact hash manifest are retained at
`/Users/tim/Documents/af/antfly-qualification/20260909/drift-investigation/`.

## F32 staging correction and refreshed references (2026-09-09)

This follow-up identifies two additional precision defects after the RoPE
corrections above. **One-step E2B numerical agreement is substantially better;
production promotion and the full numerical matrix remain unqualified.**

A full 35-layer E2B CPU HF/PEFT reference fits on this Mac with frozen BF16
weight storage and F32 operators. Sparse embedding-row conversion avoids
copying the large embedding tables. The reference uses torch 2.10.0,
transformers 5.5.2, PEFT 0.19.1, the locked Google E2B revision, the exact 100
initial q/v LoRA tensors, and explicit bias-corrected AdamW. This is an
independent F32 correctness diagnostic, not the locked CUDA/BF16 oracle.

Intermediate captures localized the first material error to Metal's BF16
linear SIMD kernels: they silently converted F32 activations to F16. The
first q projection matched an independent F16-rounded-input calculation at
6.44e-9 relative error, versus 6.57e-5 for the F32-input calculation. Backward
kernels similarly rounded gradients; CCE could also store gradient logits in
F16. These conversions can erase small gradients and turn finite F32 values
outside F16's exponent range into infinities and NaNs.

The BF16 linear kernels now retain F32 operands in forward, backward, packed,
row-tail, and fused gate/up variants. Their two staging tiles use 16 KiB, and
all corresponding host dispatch allocations were updated. Frozen weights
remain stored in BF16. CCE retains F32 gradient logits by default; the old
buffer is an explicit `TERMITE_METAL_ENABLE_LINEAR_CCE_F16_GRAD` experiment,
with the existing disable switch taking precedence. It is not the default
numerical contract. The separate F16-weight backward dispatch retains its
original scratch allocation.

A required-device regression covers aligned and tail rows, fractional F32
activations, gradients below F16's subnormal range, and finite values above
its range. The original kernel produced NaNs. The corrected kernel passes.
An analytic uniform-logit CCE case separately checks tiny loss gradients:
the default passes, while explicitly enabling the old F16 buffer fails.

Across the full model, final normalized hidden-state relative error against
HF fell from **8.97822e-4 to 5.08460e-6**, about **177x smaller**. Full arrays
were retained for linear, normalization, rotary and block outputs; these
are not just first-value fingerprints.

The fresh training capture uses the complete E2B model, q/v rank 16 and
alpha 32, seed 42, **154 physical/actual tokens** and 71 supervised tokens.
The original prepared artifact declares a 512-token maximum. A derived
160-token maximum preserved all 1,024 examples exactly, and successful runs
with both artifacts produced **all 500 trace tensors bitwise identically**,
including identical probes and loss. Both run the same 154-row graph.
Initial losses are **0.92415500 Antfly**, **0.92415243 HF**, and
**0.92415619 MLX F32**.

Correction to earlier investigation descriptions: oracle admission sets
`training_max_seq_len` to the selected example's `num_input_tokens`, not the
prepared maximum. Changing the prepared bound did not reduce graph memory.
Names containing `160` and `512` identify prepared bounds, not physical
execution shapes. The successful original-artifact rerun is the primary
one-step numerical result.

| One-step comparison | Sampled-logit relative error | Raw-gradient relative error | Update-vector relative error |
| --- | ---: | ---: | ---: |
| Before this fix vs HF F32 | 0.038102% | 0.997023% | 9.579454% |
| After this fix vs HF F32 | 0.000405% | 0.039145% | 1.404868% |
| After this fix vs MLX F32 | 0.000434% | 0.027452% | 0.986330% |
| Independent HF F32 vs MLX F32 | 0.000321% | 0.065706% | 1.907193% |

Relative vector errors use the reference direction recorded in each JSON.
After the fix, HF raw-gradient cosine is **0.9999999391**, maximum absolute
gradient difference **1.88313e-5**, and update cosine **0.9999013173**.
MLX raw-gradient cosine is **0.9999999661** and update cosine
**0.9999513573**. The Antfly differences are smaller than the HF/MLX
cross-framework gradient and update differences for this fixture. This is
not a claim of bitwise equivalence or a formal locked tolerance PASS.

**Stock BF16 MLX remains a different numerical lane:** retained stock
references give 13.09917% raw-gradient and 37.94638% update-vector error.
The close numbers above require explicitly aligned F32 computation. Neither
this comparison nor the small loss differences establish stock HF/PEFT BF16
parity, real-model native CPU parity, or downstream GRPO quality.

The primary original-artifact one-step capture completed in 26.22 seconds,
with peak process RSS 11,758,336 KiB. Swap was 2,904.50 MiB before execution,
sampled peak 6,851.44 MiB and 3,296.81 MiB afterward; it stayed within the
bounded 4 GiB paging-growth allowance. The identical derived-artifact run
also completed, at 12,978,448 KiB peak RSS; its peak swap was not retained.
These wall times are diagnostic execution time, **not throughput measurements**.
Earlier attempts hit the tighter 1 GiB paging-growth guard. The two-step
capture hit its 4 GiB guard: sampled peak swap 7,527.69 MiB, growth
4,239.44 MiB, exit -15. No two-step parity result was published. The
eight-step capture and prepared HF longer-trajectory scripts were not run.

Validation after the precision fix: **318 required-device Metal tests
passed, two optional skips; native Debug 276 passed, 24 skips; all 728 Python
tests passed; ReleaseFast Metal
build passed.** The Python suite's first invocation failed only because the
sandbox denied its loopback-port bind; the permitted rerun passed. An early
shader iteration had a vector-cast compilation error and skipped device
execution; it is excluded. Only the corrected, required-device runs above
count. Temporary real-model graph hooks were removed from shipping source.

The release diagnostic binary SHA-256 is
`390ace476e00b83bf4f4de8bc3b3ed9e81c10d914fcab71ec94e6fda24e3a1b3`,
based on `e7edd66f399edef41691a71be1530a86722bf4d7` with source-patch SHA-256
`fe19a5427e3c29eddf73a64e95bc8809a712346acd715988bb729f5e3a8197fd`.
The successful raw capture is
`diagnostics/e2b-qv-step1-metal-f32-staging-fixed-512-paging-diagnostic`.
No immutable oracle COMPLETE or acceptance PASS was produced.

Retained source, binary, intermediate arrays, HF/MLX comparisons, scripts,
validation logs, and exclusions are under
`/Users/tim/Documents/af/antfly-qualification/20260909/residual-investigation/`,
with a verified SHA-256 inventory in `MANIFEST.json`. The earlier
`drift-investigation/` archive remains intact. The first broad activation
capture and aborted raw captures are excluded regardless of partial output.

The complete 1/2/8-step native/Metal/HF matrix, stock BF16 qualification,
full E4B GRPO acceptance, current categorical-sampler MLX rollout comparison,
accepted-adapter full-length recovery, controlled performance and hosted CI
remain pending. The frozen final holdout remains unevaluated. Nonzero paging
continues to block the frozen acceptance campaign; CUDA implementation
remains outside this work's scope.

## Full parity follow-up: embedding residency and CCE range (2026-09-09)

**Full parity and production acceptance remain unproven.** This follow-up
fixes native embedding residency and CCE exponent-range correctness, preserves
sparse embedding forwards through autodiff, and bounds oracle logit workspace.
CUDA implementation remains outside scope. The reserved final holdout has
not been scored.

Native embedding borrowing recognized only legacy `model.embed_tokens.weight`
and `wte.weight` names. Gemma4's `model.language_model` and normalized
per-layer embedding names therefore expanded entire BF16 tables to F32.
The loader now preserves their BF16 backing storage. A regression checks both
resident and lazy stores, fused lookup and lowered gather, repeated IDs,
exact row values, and the absence of a full F32 allocation. Before this fix,
the full native capture exceeded its 4 GiB swap-growth guard in 16.6 seconds.
The first loader-fixed attempt had no swap growth but timed out at 180 seconds.
The longer retest retained the same 18 GiB RSS / 4 GiB swap-growth limits:
it completed and validated in 753.4 seconds with no swap growth; peak RSS 13.43 GiB and peak swap 3038.00 MiB (baseline 3038.00 MiB).
The successful capture contains all 100 adapter tensors and 500 trace tensors,
with loss 0.9241400361 and raw-gradient norm 0.6976542280. Its complete one-step
comparisons are:

| Comparison | Gradient relative error | Update-vector relative error |
| --- | ---: | ---: |
| Native versus HF F32 (HF reference) | 0.049805% | 1.685759% |
| Native versus MLX F32 (MLX reference) | 0.021730% | 0.760342% |
| Earlier Metal versus native (native reference) | 0.016634% | 0.733851% |

Native uses the loader-only binary
`25336ec8ba0f65316d2bb66d9dc7e0812753256216b316a1b935e4423fb525d2`
and source patch `6154bdc47d07460ac09f53445bf6684c63901df44a102daed5d7a5badde36afb`.
It predates the sparse-forward/capture-bound build. The direct Metal comparison
uses the prior F32-staging capture and records both build identities. These
complete captures close a diagnostic evidence gap, without attesting the final
source or a full trajectory. No swap growth is not equivalent to a zero-paging
host; this CPU run is not a native/Metal throughput benchmark.

Autodiff now preserves the sparse embedding forward and supplies its
scatter-add VJP directly. The prior general Metal gather prepared the entire
BF16 table. An analytic repeated-token regression checks the retained forward,
loss, and accumulated table gradient. Real-model memory benefit and trajectory
agreement remain unqualified; the combined Metal capture exceeded its guard.

Large-row CCE backward also silently built an F16 mirror of BF16 weights.
The analytic test uses the finite, exactly representable BF16 weight 69632:
the old mirror returns `-inf` instead of `-0.0054399166`. The default now uses
the BF16 path. `TERMITE_METAL_ENABLE_LINEAR_CCE_F16_MPS_BACKWARD=1` explicitly
enables the old experiment; its existing disable flag takes precedence.
The negative control fails as expected, while the default passes. The large-row
dispatch requires at least 128 rows and is not exercised by the 71-row E2B
fixture. This closes that default conversion; it does not qualify unrelated
inference or experimental kernels.

Oracle logit capture now limits each full-vocabulary projection to 16 MiB
(or one row when a single row exceeds that budget). CCE training streams
vocabulary tiles, so its 512-row chunk setting was an unsuitable capture
workspace bound. E2B capture now projects at most 16 rows per call instead
of all 71 supervised rows. Every requested probe and all 154 physical input
tokens remain present. This bound did not resolve total Metal paging:
the latest shipping-source attempt exceeded 4 GiB swap growth in 26.5 seconds.
A separate `vmmap` snapshot measured **19.7 GiB physical footprint**, including
large heap and GPU allocations. The sampling and allocation-calltree captures
returned no usable call stacks, so the allocation source is not localized.
No further memory bound was relaxed, and no partial capture is counted as PASS.

Independent full E2B HF/PEFT F32 and MLX F32 trajectories completed at both
2 and 8 optimizer steps, using the same initial adapter and prepared example.
These are CPU HF/PEFT versus MLX GPU diagnostics, not the locked CUDA/BF16 oracle.
All 100 adapter tensors, initial values, shapes and probe identities were checked.
Relative vector errors below use HF as the reference:

| HF/PEFT versus MLX | Step 2 | Step 8 |
| --- | ---: | ---: |
| Sampled-logit relative error | 0.013974% | 0.201805% |
| Raw-gradient relative error | 0.207068% | 5.590517% |
| Cumulative update-vector relative error | 1.843557% | 0.850340% |
| HF loss | 0.87873411 | 0.01052650 |
| MLX loss | 0.87851971 | 0.01052698 |

Nearly equal final losses do not imply equal gradients. These trajectories
evolve independently. To separate state divergence from operator differences,
both frameworks then loaded the exact same HF step-8 adapter. That common-state
comparison has sampled-logit relative error **0.000284%** (maximum absolute
error 0.000152), raw-gradient relative error **0.558960%**, and fresh-Adam
update-vector relative error **6.157790%**. Losses are 0.0050749755 / 0.0050827451.
The reduced gradient disagreement is evidence of trajectory amplification,
with a residual at identical weights. This snapshot uses fresh optimizer state;
it is **not a ninth continuation step**. Independent F32 AdamW replay reproduces
each framework's updates closely (relative error 2.30e-7 for HF, 6.67e-6 for MLX).
Only 4,411 of 2,678,784 gradient components (0.1647%) change sign; those account
for 94.37% of squared update disagreement. Their median HF gradient magnitude
is 3.75e-8, versus 8.73e-6 over active components. This attributes most update
amplification to near-zero gradient signs rather than mismatched optimizer
formulas; it does not remove the residual forward/backward gap. No threshold
was changed or matrix PASS issued. HF reference captures had no swap growth; MLX captures paged and
cannot support clean memory or performance qualification.

The prior successful shipping Metal one-step diagnostic still belongs to the
F32-staging build (`390ace476e00b83bf4f4de8bc3b3ed9e81c10d914fcab71ec94e6fda24e3a1b3`):
its raw-gradient relative errors were 0.039145% versus HF and 0.027452% versus
aligned MLX F32. Those results do not attest the newer builds. Stock BF16 MLX
remains a separate, materially divergent comparison.

Current-source validation: required-device Metal Debug **322 passed, 2 optional
skips**; native Debug **279 passed, 24 skips**; shared ML library tests passed;
focused Python oracle/contract/resume tests **77 passed**. The prior full Python
728-test result was not rerun. An initial Metal invocation omitted the required
embedded source revision and failed only that CLI attestation test; the full
corrected invocation passed. The explicit F16 negative control fails as intended.
ReleaseFast Metal build, Zig formatting, whitespace and unresolved-conflict
checks passed. No Git mutation was performed.

Current build identity:

- Source HEAD: `e7edd66f399edef41691a71be1530a86722bf4d7`, dirty tracked source.
- Shipping-source patch SHA256: `9d9887ffc17c69461d9afda431a44aca40ec6c1730ec0d95724ed0701d4476fb`.
- Binary SHA256: `d8d17bdf8428ebb9a773dd08d202a93a0aee253136d36ed19649eb5b7d56918a`.
- Final source patch and binary hashes were rechecked against build metadata.

Reports, source snapshots, exact runners, failed attempts, references and all
three build-stage binaries are retained under
`/Users/tim/Documents/af/antfly-qualification/20260909/full-parity-investigation/`.
`MANIFEST.json` records SHA256 for each copied artifact. The previous
`drift-investigation/` and `residual-investigation/` archives remain unchanged.
The machine still has nonzero paging. Remaining gates are a completed
current-source native/Metal 1/2/8-step matrix, stock BF16 and E4B numerical
comparisons, current categorical-sampler MLX GRPO comparison, E4B GRPO acceptance,
reserved holdout, zero-paging performance/memory evidence, accepted-adapter recovery refresh and hosted CI.

### 2026-09-10 parity continuation: memory, CCE and stock PEFT fixes

This continuation supersedes the preceding current-status statements; the older
captures and failures remain historical evidence. The Metal/HF/MLX diagnostic
matrix now contains E2B and E4B, `peft-qv` and `text-all-linear`, and independent
1/2/8-step trajectories. Full numerical parity and production acceptance remain
unqualified. These runs use one locked prepared example (154 physical tokens,
71 supervised tokens, prepared maximum 512), BF16 frozen weights with F32-aligned
activations/operators, rank 16, alpha 32, seed 42 and the locked AdamW recipe.
CPU HF/PEFT F32 and activation-aligned MLX F32 are diagnostic references; they
do not replace the locked CUDA/BF16 oracle or stock BF16 MLX qualification.

The continuation fixes four independently reproduced defects:

- Deferred forward Metal dots now retain native BF16/F16 frozen weights. The
  generic dot previously made both a full F32 host peer and a device clone.
  A direct unplanned-dot regression failed before the dispatch fix and passes
  afterward. The initial E2B one-step trace remained bitwise identical across
  all 500 tensors; both models now finish the bounded Metal matrix.
- Cold live-logit evaluation now binds persistent resident adapter weights
  before training starts. Transient host allocations could collide with cached
  device bindings. A regression observes repeated nonzero device updates with
  stale host copies and no optimizer allocation. At common trained E2B q/v
  weights, sampled-logit error versus MLX fell from 5.08% to 0.000299%.
- Tiled CCE retains the row maximum separately from the logarithmic exponential
  sum, computes shifted loss/gradients, and compensates tile summation. For an
  analytic large-offset fixture, the previous loss was 0.00048828125 versus an
  expected 0.00046083788; the fixed kernel passes the unchanged 2e-7 tolerance.
  Splitting the state without compensated summation still failed. This fixes
  an actual numerical defect but does not explain the residual common-state HF
  gradient difference in the real q/v model.
- Canonicalization expands root PLE aliases before adding `model.`. Both PEFT
  exporters preserve the real multimodal root and translate PLE names in tensor
  keys and configuration. The Zig manifest hashes the resulting destination
  configuration. Neither exporter changes the tensor payload values.

Stock `PeftModel.from_pretrained` loading passed **all eight cases**: Python and
public Zig exporters, both models and both presets. Tests use trained step-8
adapters with every LoRA B tensor nonzero and verify every loaded float32 value,
shape and target: E2B q/v 100 tensors, E2B full 552, E4B q/v 132, E4B full 686.
The actual pinned `Gemma4ForConditionalGeneration` loads frozen text weights
from the checkpoint; unused modality weights stay meta. This establishes export
and stock loading compatibility, not forward quality or reverse PEFT-to-Antfly
round-trip qualification. Packages: torch 2.10.0, transformers 5.5.2, PEFT 0.19.1.

Independent Metal versus HF results below are **percent relative L2 error**;
the update column compares cumulative parameter updates from the initial adapter.

| Model | Preset | Steps | Gradient error | Update error |
| --- | --- | ---: | ---: | ---: |
| E2B | q/v | 1 | 0.039036% | 1.403575% |
| E2B | q/v | 2 | 0.160232% | 1.433394% |
| E2B | q/v | 8 | 2.220013% | 0.648025% |
| E4B | q/v | 1 | 0.004026% | 0.183772% |
| E4B | q/v | 2 | 0.014960% | 0.422064% |
| E4B | q/v | 8 | 0.399732% | 0.177634% |
| E2B | full | 1 | 0.023339% | 0.622435% |
| E2B | full | 2 | 2.367138% | 2.099288% |
| E2B | full | 8 | 96.586537% | 14.412970% |
| E4B | full | 1 | 0.003871% | 0.170286% |
| E4B | full | 2 | 0.035521% | 0.330731% |
| E4B | full | 8 | 29.312955% | 1.931189% |

Against aligned MLX, full-preset eight-step gradient/update errors are
2.946032% / 0.925510% for E2B and 19.484668% / 1.135896% for E4B. Similar
training losses do not establish parity: E2B full step-8 sampled-logit error
versus HF is 34.3954%, despite losses 0.0332142 and 0.0328521.

Applying the unchanged `hf-zig-bf16` numeric values to these F32 diagnostics
identifies the following out-of-bound per-target states (gradient, updated
weight, Adam m and v). This is triage, not the locked BF16 validator or a PASS.

| Model | Preset | Compared per step | Exceeded at 1 / 2 / 8 steps |
| --- | --- | ---: | ---: |
| E2B | q/v | 400 | 12 / 7 / 98 |
| E4B | q/v | 528 | 0 / 0 / 1 |
| E2B | full | 2208 | 9 / 114 / 2203 |
| E4B | full | 2744 | 0 / 0 / 1815 |

At identical HF step-8 full-preset weights with a **fresh optimizer**, the latest
Metal build versus HF has gradient errors 0.012216% (E2B) and 0.076140% (E4B),
and sampled-logit errors 0.000181% and 0.000105%. All 2208 / 2744 per-target
states remain within the diagnostic bounds at these common snapshots. Fresh
update-vector errors are 0.238889% / 1.695014%; comparison of updated weights
has a different denominator. E2B versus MLX at the same full-preset snapshot
has gradient error 0.007973% and fresh-update error 0.082041%. These results
support substantial trajectory amplification; they are not ninth continuation
steps and do not eliminate the remaining numerical differences.

An independent algebraic F32 Adam replay uses each capture's raw gradients and
recorded preclip norm, including the unchanged clipping epsilon. Across both
full presets at the initial and common-trained snapshots, replay update errors
are at most 2.53e-7 relative L2 for Metal and 2.28e-7 for HF. At initial E2B
weights, only 1243 of 15,204,352 active gradient components change sign, yet
they contribute 69.97% of squared update disagreement. E4B has 241 of
21,331,872 active components changing sign, contributing 54.72%. At the common
E4B snapshot, the corresponding share is 81.30%. This rules against an Adam
formula mismatch in these first-step cases; it does not prove that every
subsequent step is correct. The retained report is
`parity-continuation-full-adam-amplification.json`.

Lower-memory reference methods retain their numerical evidence:

- HF frozen-linear F32 rematerialization is bitwise identical across all 600
  E2B one-step tensors. Output-head rematerialization additionally matches all
  3312 E2B full-preset one-step tensors and admits E4B full step 8 under 16 GiB.
- MLX layer checkpointing matches all 600 E2B q/v two-step tensors exactly.
  Streaming output and releasing validated Python adapter objects preserve the
  entire E2B full-preset eight-step tensor file byte-for-byte. The resulting
  E4B full-preset 1/2/8 references complete without swap growth.
- Failed reference attempts remain excluded even if they wrote a complete
  tensor file. This includes earlier E4B HF/MLX memory-bound attempts and the
  latest auxiliary E4B common-snapshot MLX attempt (+1465.13 MiB swap). The
  main independent MLX matrix is complete. A subsequent isolated-validation
  retry closes the auxiliary comparison as described below.

The common-snapshot runner now performs heavyweight adapter/capture validation
in a separately guarded CPU process, then checks a hash-bound proof before MLX
loads. All original validation and exact model adapter loading checks remain.
The E2B common-reference tensor file is byte-identical to the previous method
(SHA256 `74899c5171fb4714f99ef8b1f943e8714407490e5c160a25c61b3ff68719ce90`).
The E4B retry completes in 22.80 seconds, peak sampled RSS 8,196,960 KiB and
586.07 MiB swap growth, within the unchanged 16 GiB / 1 GiB guards. At identical
trained full-preset weights, E4B Metal versus MLX gradient error is 0.006388%,
fresh-update error 0.154427% and sampled-logit error 0.000109%. The prior failed
attempt remains excluded; the successful retry is numerical evidence only.

Model jobs ran serially. Metal/native diagnostics retain 18 GiB sampled RSS and
4 GiB swap-growth guards; HF/MLX retain 16 GiB and 1 GiB growth guards. The host
had pre-existing swap throughout. These runs cannot qualify zero-paging memory
or performance. The reserved final 254-example holdout remains untouched.

The latest-build native E2B q/v one-step refresh completed in 751.36 seconds,
with peak sampled RSS 14,546,416 KiB and no swap growth (2844.31 MiB initially,
2692.12 MiB finally). All 500 trace tensors are byte-identical to the earlier
loader-fixed native capture: SHA256
`3cae501ac12a8823416ecefa5d582257ba977a67485d05cf88bde05795aaa854`.
Gradient/update-vector errors are 0.049805% / 1.685759% versus HF and
0.021730% / 0.760342% versus aligned MLX. Against the current Metal trajectory
build, 24 of 400 per-target states exceed the unchanged native/Metal numeric
bounds; all are updated LoRA B weights. This refresh closes source freshness
for one native cell, not the complete native matrix. A three-second stack
sample placed all 173 active-thread samples in the scalar F64 generic-dot
loop at `native_compute.zig:38220`; this is diagnostic localization, not a
zero-paging performance measurement.

Current validation: actual-device Metal Debug **326 passed, 2 optional skips**;
native Debug **280 passed, 26 skips**; full Gemma4 Python discovery **730 passed**.
The initial sandbox Python run failed only because its fake HTTP server could
not bind localhost; the complete authorized rerun passed. ReleaseFast built.

Build/source identities (dirty source, HEAD
`e7edd66f399edef41691a71be1530a86722bf4d7`):

- Full numerical trajectory build (`stable-cce-fixed`): binary SHA256
  `ca89f6bf48c89bfa601ab9d6118c7ffeee5a57d8b405dd051cef9e256b6b76ee`,
  compiled-source patch SHA256
  `7c23b02c153afcbdcb6977f89b6aa737e1d6ddfded69b42203662f4df31ce07e`.
- Latest export/common-snapshot build (`peft-namespace-fixed`): binary SHA256
  `88be734cfcea017b36f9552b8a918a18cafc245c206357b5c254b6534a9dc418`,
  compiled-source patch SHA256
  `26f5eb715504eb0b92cd3b087ff9d5f1c6e76f1acf2abd84471cc537a91feb30`.
  Its additional compiled changes affect PEFT export. Python source snapshots
  are retained separately; the compiled-source patch is not an all-file digest.

Reproduction runners and comparisons live in
`.benchmark-results/gemma4-qualification-20260909/`. In particular,
`parity-continuation-all-hf-numeric-bounds.json` retains every checked target
state, `parity-continuation-{stable,all-linear}-*-vs-*.json` records numerical
comparisons, and `parity-continuation-final-common-*-vs-*.json` records common
snapshots. Stock loading reports live under
`/Users/tim/Documents/af/antfly-qualification/20260909/diagnostics/parity-continuation-stock-peft-*/`.
`/Users/tim/Documents/af/antfly-qualification/20260909/parity-continuation-investigation/`
retains exact runners, source, builds, successful and excluded raw captures with
a SHA256 manifest; previous sealed archives stay intact. The manifest digest is
recorded separately in `parity-continuation-archive-receipt.json` in the results
directory, avoiding a self-referential archive digest.

Remaining gates: independent-trajectory numerical failures; completion of the
current native matrix; stock BF16 parity and the locked HF/BF16 reference;
categorical-sampler MLX GRPO comparison; E4B GRPO acceptance and reserved final
holdout; reverse PEFT adapter recovery qualification; accepted-adapter recovery
refresh; zero-paging performance/memory evidence and hosted CI. CUDA implementation
remains outside scope. No tolerance was loosened or oracle COMPLETE / acceptance
PASS published.

### 2026-09-10 native matrix and chronological GRPO replay continuation

The native reference now uses a packed F64 kernel for large rank-2 dots,
including dense, native-storage and strided-view operands. Each SIMD lane
computes an independent output in the same left-to-right F64 accumulation order
as the scalar implementation; there is no reduction across lanes or F32
intermediate rounding. Eight output columns share a panel of `64 * k` bytes.
Existing source-tensor BLAS dispatch retains its previous behavior. Exact tests
cover both operand orientations, strided views, odd row/column tails, BF16/F16
storage, and a cancellation case whose correct F64 result is 1 but whose F32
result is 0. Representative kernel diagnostics show 8.0–9.5x speedups with zero
output-bit differences. These timings do not qualify real-model performance.

The first optimization only covered transposed dense/source weights. Its live
model profile exposed a remaining scalar strided-view path. That attempt was
intentionally stopped after 376.76 seconds and is excluded from parity evidence;
it did not trigger the resource guard and did not publish a complete trace.
The final implementation covers that path too. The serial native E2B/E4B,
q/v/full, 1/2/8-step matrix retains the 18 GiB sampled RSS, 4 GiB swap-growth and
900-second per-capture limits. Its first cell must reproduce the sealed native
E2B trace byte-for-byte before the remaining cells can start. Live results are
recorded in `native-f64-strided-campaign-status.json` in the existing results
directory; a running cell is not a completed comparison.

Both single-token and multi-token MLX GRPO comparison runners now preserve the
chronological group order recorded in Antfly reward traces. The previous loader
sorted groups by original dataset prompt index, silently undoing epoch shuffling.
The runners bind each chronological group to its original prompt and reject
interleaved optimizer groups. New shuffled-order regressions fail before the fix;
all 30 focused comparison tests pass afterward. The categorical-sampler guard
remains: correct ordering alone does not qualify the retired ranked sampler.

Current source validation: native Debug **282 passed, 26 skipped**; required-device
Metal Debug **328 passed, 2 optional skips**; full Gemma4 Python **733 passed**.
ReleaseFast and source/binary identity checks pass. The current compiled-source
patch includes the new untracked helper file explicitly, in addition to tracked
changes:

- Binary SHA256: `c82c30aa5c844e6bc59ca4910287b0eb60cf48259e393b597bbe8f127caaaa1c`.
- Compiled-source patch SHA256: `b3993194335aed256945bd71e98d53be6e825c73b5d58e15df61a67aca272164`.
- `native_f64_dot.zig` SHA256: `e63ca701578933f16e2c6058b6758643aa3959d3dde6c4de84d5fdb9f1fa2299`.
- HEAD remains `e7edd66f399edef41691a71be1530a86722bf4d7`; source is dirty.

These changes enable missing reference measurements and correct GRPO replay;
they do not waive the independently measured trajectory failures, stock BF16
qualification, categorical rollout comparison, acceptance/holdout, recovery,
zero-paging or hosted-CI gates documented above. No Git mutation was performed.

<!-- native-matrix-results:start -->

The real-model preservation gate passed: all 500 E2B q/v trace tensors are
byte-identical to the sealed native reference (trace SHA256
`3cae501ac12a8823416ecefa5d582257ba977a67485d05cf88bde05795aaa854`).
Diagnostic one-step wall time fell from 751.36 to 96.70 seconds (7.77x),
without swap growth. Pre-existing swap excludes these timings from release
performance evidence.

Completed native cells: **6/12**. These are independent F32-aligned
trajectories; relative errors use the named HF or MLX reference.

| Model / preset | Steps | Native/HF gradient error | Native/MLX gradient error | Native/HF update error |
| --- | ---: | ---: | ---: | ---: |
| E2B / qv | 1 | 0.0498% | 0.0217% | 1.6858% |
| E2B / qv | 2 | 0.1889% | 0.0754% | 1.6688% |
| E2B / qv | 8 | 4.0760% | 1.5552% | 0.7568% |
| E2B / all-linear | 1 | 0.0286% | 0.0154% | 0.8366% |
| E2B / all-linear | 2 | 2.3661% | 0.0755% | 2.3128% |
| E2B / all-linear | 8 | 96.4793% | 2.9548% | 14.3841% |

Unfinished or excluded cells: [{"model": "e4b", "preset": "qv", "steps": 1, "status": "failed"}].

`native-f64-strided-{hf,native-metal}-bounds-summary.json` records per-target
failures against the unchanged numerical values. Passing capture validation is
not a numerical-parity PASS. References and current native builds are attested
separately. The trajectory failures remain open.

The host-side categorical sampler now reproduces Zig 0.16 Xoshiro256++ and its
F64 uniform draws, v2 group/completion seed derivation, F32 logit differences,
F64 CDF accumulation, temperature/top-k/nucleus filtering, stable token-ID ties,
and evaluation's greedy first completion. Executing verbatim production Zig
sampling functions generated **384 bit-identical random draws and 1,920 matching
token selections** across five policies. The source-bound fixture is reproducible;
13 primitive tests include rare extra RNG draws, EOS-independent streams,
shuffled prompt identity and invalid inputs. These are primitive diagnostics;
the categorical acceptance guard remains until skipped-group and behavioral
qualification are implemented. Both legacy rollout runners now use stable
ranked tie selection through the shared helper.

Both GRPO classifiers now require actual adapter-update vector distance at their
existing relative-error limits (5% single-token / 10% multi-token), in addition
to magnitude and direction. The previous magnitude-only gate falsely accepted
same-length rotated updates, missing vector metrics and NaNs in the new field.
The negative regression failed three cases before the fix. An actual-MLX tensor
check confirms that a 10% vector error with negligible magnitude difference is
rejected while identical updates pass. Missing, non-finite or invalid vector
metrics fail closed. Historical magnitude-only output fields are retained.

Final Python validation: **748 passed in 67.346 seconds**; 44 GRPO/MLX runner
tests and 13 sampling primitive tests pass. The full test count supersedes the
733-test intermediate run above. For this sampler/vector-gate stage, compiled Zig source and binary hashes
remained unchanged; subsequent native storage builds are recorded separately below.
Sampler evidence is indexed in `grpo-sampling-status.json`.

<!-- native-matrix-results:end -->

### Native E4B frozen-weight memory follow-up

The first E4B q/v native capture using `native-f64-strided-fixed` was stopped by
the unchanged 18 GiB sampled-RSS guard after 138.43 seconds (peak RSS 19,710,880
KiB; peak swap 5,856.94 MiB from a 2,510.69 MiB baseline). It produced no complete
trace and is excluded. The six successful E2B cells and excluded E4B attempt,
Python sampler/vector-gate fixes, source and binaries were sealed in
`/Users/tim/Documents/af/antfly-qualification/20260909/native-matrix-investigation`:
424 files, manifest SHA256
`05962acd857f2d010614dc0cf0ee7f37892ebba8f19dc4098c912f7f676216a5`.

Native Gemma4 backend creation now opts into source-backed frozen rank-2
BF16/F16 handles, extending the embedding storage policy to linear weights.
This prevents retained training parameter handles from owning persistent full
F32 copies. Individual operators still own temporary F32 views; the bound is
the largest needed matrix, not a promise of zero conversion. The dense dot's
F64 reduction and eager F32 GEMM shape remain unchanged. Native inference's
ordinary loading policy is unchanged. The biased source-linear path now keeps
its bias through GEMM's beta=1 instead of calling the overwriting no-bias helper.
Regression coverage checks resident/lazy handles, BF16/F16, both contraction
layouts and biased/unbiased linear output bytes. The model checks below belong
to these new builds; the preceding six-cell matrix retains its older build identity.

The first source-backed build (`native-frozen-linear-fixed`, binary SHA256
`ba50e107ae16fef4c22721f422bf70ec27f238d3e1ad8b8d7f5644b5b9375a40`, compiled
patch `8a768cb71f387ec5f48e8a10919a569e35e217b9519a3b492df9d07f2b6fce9e`)
passed the E2B gate: all 500 trace tensors remain byte-identical, peak RSS fell
from 14.6 to 10.2 GiB, and there was no swap growth. Its E4B attempt stayed below
the RSS cap but crossed the swap-growth limit after 61.08 seconds. A separate
allocation-profile attempt under unchanged limits also failed after 60.90
seconds. Both are excluded. The retained vmmap summaries distinguish about
8.7 GiB of resident clean file mappings from about 3 GiB of dirty heap during
probing; they do not establish accepted E4B training memory.

The follow-up `native-frozen-reclaim` variant extends the existing
completed-operation mapped-page reclamation hook to resident source-backed
handles, using the source tensor name for the tensor-store range lookup.
It also covers lowered dot, transpose and gather consumers. The mapping and
parameter handles stay valid; only consumed clean pages are eligible for
reclamation. A regression verifies the source-name binding, repeated reads,
and exclusion of ordinary native and non-mapped handles. Results are recorded
in `native-frozen-reclaim-status.json` and `native-frozen-reclaim-gate-status.json`.

The final resident-mapping build passed **284 native tests (26 skips)** and
**330 required-device Metal tests (2 optional skips)**, followed by ReleaseFast.
Binary SHA256 `d7d5aed0ea83457dc45c16e4f8b83b5a23aa1dc6bc2a94d2af0e12b669ef232c`;
compiled patch SHA256
`0bc59922901a0fb5d00001e98fdb8c0c04649fc3adb82007ee391ae0b78c572d`.
The source identity, formatting, whitespace and unresolved-conflict audits pass.
Python remains **748 passed**; no Python implementation changed after that run.

E2B q/v one step completed in 102.746 seconds, with peak sampled RSS 11,749,520
KiB and zero swap growth from a 2,601.00 MiB baseline. The entire 500-tensor trace
is byte-identical to the sealed earlier native trace (SHA256
`3cae501ac12a8823416ecefa5d582257ba977a67485d05cf88bde05795aaa854`).
Its gradient/update relative L2 errors remain 0.049805%/1.685759% against CPU
HF/PEFT F32 and 0.021730%/0.760342% against aligned MLX F32. These aggregate
numbers do not override the earlier per-target diagnostic failures.

E4B q/v one step stopped after 58.948 seconds with exit -15. Peak sampled RSS was
15,002,224 KiB, below 18 GiB, but swap grew from 2,561.00 to 6,743.19 MiB:
**4,182.19 MiB growth exceeded the unchanged 4,096 MiB cap**. No complete trace
was produced; this attempt is excluded. Resident-page advice and source-backed
storage therefore do not establish native E4B memory qualification. No further
unchanged-source retries were launched. All model jobs are terminal, the reserved
254-example holdout remains untouched, and no oracle COMPLETE or acceptance PASS
was published. The E4B native matrix, independent-trajectory numerical failures,
stock BF16 reference, end-to-end categorical GRPO behavior, E4B acceptance,
adapter recovery, zero-paging qualification and hosted CI remain open.

The frozen-weight builds, exact-trace check, failed attempts, allocation profiles,
source snapshots and current PR description are retained separately in
`/Users/tim/Documents/af/antfly-qualification/20260909/native-frozen-weight-investigation`.
Its manifest digest and file count are recorded in
`.benchmark-results/gemma4-qualification-20260909/native-frozen-weight-archive-receipt.json`;
the archive hash-binds the preceding native-matrix evidence and preserves all
excluded attempts. These archives are diagnostics, not release qualification.


### 2026-09-10 gradient-norm precision follow-up

The second-step optimizer audit replays 14 retained pairs: native E2B q/v and
full presets, plus Metal, HF and aligned MLX for both models and presets. Each
pair uses its own saved step-1 weights/moments and step-2 raw gradients; initial
losses match exactly, native request bindings and training options match, and
reference initial adapter tensors match exactly. This tests the second step,
not the unobserved intervening steps 3-8. Native F32-beta arithmetic reproduces
both saved moments exactly and incremental updates within 1.998e-7 relative L2.
The Python-scalar-beta replay reproduces HF updates within 1.526e-7. The initial
formula variant used Python scalar beta complements for every backend; its
small native discrepancy disappears when using the native F32 beta semantics.
No optimizer hyperparameter or tolerance was changed. Full results and both
controls are retained in `second-step-optimizer-replay.json`.

The audit exposed a separate Metal clipping issue: the production
`termite_training_sumsq_f32` shader used a single thread to add every squared
value sequentially in F32. On the exact production shader, a vector beginning
with 1 followed by 1,048,576 values of alternating +/-1e-4 produced squared sum
1 instead of 1.01048575947 (1.037695% relative error). A 257-element case also
failed the predeclared 2e-7 relative bound. The replacement uses 256 lanes,
compensated local sums and a balanced reduction tree, compiled through the
precise-math library. Both single-input and batched dispatches provide their
own threadgroup scratch. The corrected shader passes the same analytic bounds;
the million-element case has 8.756e-9 relative error.

Replaying the **same saved real-model gradients** through both production
shaders isolates the reduction from all model arithmetic:

| Step-2 full preset | Tensors | F64 reference norm | Old norm relative error | Corrected norm relative error |
| --- | ---: | ---: | ---: | ---: |
| E2B | 552 | 12.0020461540 | 1.817873e-6 | 1.098551e-8 |
| E4B | 686 | 4.0824963298 | 1.822980e-5 | 4.975231e-9 |

The old norm errors agree with clipping-scale discrepancies independently
inferred from saved first moments. These are reduction and optimizer-state
checks; updated full trajectories still require fresh model captures. Shader
source, F64 controls, tensor payloads, input-source hashes and results are in
`training-sumsq-before`, `training-sumsq-after`,
`training-sumsq-real-gradients`, and `training-sumsq-root-cause.json`.

A separate native page-allocation experiment preserved all 500 E2B trace tensors
exactly (104.757 seconds, no swap growth), but E4B still exceeded the unchanged
4 GiB swap-growth limit after 64.855 seconds: 5,194.81 MiB growth, peak RSS
14,922,224 KiB. The incomplete capture is excluded. The experiment was removed
from shipping source because it did not improve E4B admission; the preceding
source-backed resident-mapping implementation was restored byte-for-byte.
Experimental source, patch, binary and results remain retained under
`native-frozen-pages-*`; binary SHA256
`7162300be13eb54a10a50079a515d0defd6af2f4dffcc5e660e7e2be2830e547`,
patch SHA256 `ab3a9d20c1698eaf6f88991276167e9e9ffd5fb6cba1a93a99b6bfb18b09e1dd`.
No allocator environment settings or process-wide purge were introduced.

<!-- training-sumsq-results:start -->
The corrected build passed **284 native tests (26 skips)** and **331 required-device
Metal tests (2 optional skips)**, including the integrated analytic single/batched
norm regression. Python remains **748 passed**; no shipping Python source changed
after that run. ReleaseFast and compiled-source identity checks pass.

Binary SHA256 `12546440265368956e0806d2f3866ed4e568330f11cb6c04167951798359ba79`;
compiled patch SHA256 `976f909f086d7e5aa97a2a1bde81169fd420544c2cadd9886b8c516c92528a94`.

Fresh independent trajectories below use unchanged F32 diagnostic references and
unchanged resource limits. All errors are **percent relative L2**; updates are
cumulative from the exact initial adapter. They are not stock BF16 qualification.

| Model | Preset | Steps | HF gradient | HF update | MLX gradient | MLX update |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E2B | all-linear | 1 | 0.023339% | 0.622435% | 0.020048% | 0.476215% |
| E2B | all-linear | 2 | 2.367092% | 2.098454% | 0.113366% | 1.035204% |
| E2B | all-linear | 8 | 96.584720% | 14.410316% | 2.978692% | 0.930807% |
| E4B | all-linear | 1 | 0.003871% | 0.170286% | 0.002915% | 0.136007% |
| E4B | all-linear | 2 | 0.035437% | 0.330417% | 0.023666% | 0.318632% |
| E4B | all-linear | 8 | 29.508488% | 1.936556% | 21.898395% | 1.253675% |
| E2B | qv | 1 | 0.039036% | 1.403575% | 0.027572% | 0.990024% |
| E2B | qv | 2 | 0.160232% | 1.433394% | 0.109306% | 1.009484% |
| E2B | qv | 8 | 2.326447% | 0.648633% | 3.198099% | 0.457820% |
| E4B | qv | 1 | 0.004026% | 0.183772% | 0.002587% | 0.181795% |
| E4B | qv | 2 | 0.014960% | 0.422063% | 0.023118% | 0.443063% |
| E4B | qv | 8 | 0.389547% | 0.177452% | 0.340878% | 0.198202% |

Completed 12 of the 12 planned cells. Campaign terminal state:
`diagnostic-matrix-captured`. Per-target bounds remain unchanged:

| Model | Preset | Step | States outside diagnostic bounds |
| --- | --- | ---: | ---: |
| E2B | all-linear | 1 | 9 / 2208 |
| E2B | all-linear | 2 | 114 / 2208 |
| E2B | all-linear | 8 | 2203 / 2208 |
| E4B | all-linear | 1 | 0 / 2744 |
| E4B | all-linear | 2 | 0 / 2744 |
| E4B | all-linear | 8 | 1803 / 2744 |
| E2B | qv | 1 | 12 / 400 |
| E2B | qv | 2 | 7 / 400 |
| E2B | qv | 8 | 98 / 400 |
| E4B | qv | 1 | 0 / 528 |
| E4B | qv | 2 | 0 / 528 |
| E4B | qv | 8 | 1 / 528 |

The fresh second-step replay reduces the inferred clipping-scale discrepancy to
at most 1.267e-7 across all four Metal model/preset pairs, versus 1.825e-5 before.
Its F32-beta first-moment replay errors are at most 1.748e-7. Residual second-moment
and incremental-update replay errors remain up to 1.626e-5 and 8.246e-6;
these paired independent captures do not yet prove all Metal optimizer states
or all intermediate steps. `training-sumsq-second-step-optimizer-replay.json`
retains those limits rather than collapsing the result into an optimizer PASS.

The corrected reduction closes the demonstrated clipping precision defect. It
does not waive remaining independent-trajectory failures, native E4B coverage,
stock BF16/HF oracle, categorical GRPO behavior, E4B acceptance, reserved holdout,
adapter recovery, zero-paging qualification or hosted CI. No numerical tolerance
was loosened, no reserved holdout was scored, and no acceptance PASS was issued.

Evidence is retained in `training-sumsq-*` and sealed separately in
`/Users/tim/Documents/af/antfly-qualification/20260909/training-norm-investigation`.
The manifest receipt is `training-sumsq-archive-receipt.json`; prior evidence is
hash-bound without modifying the older archives.
<!-- training-sumsq-results:end -->


### 2026-09-10 precise Metal AdamW follow-up

The residual second-moment discrepancy was reproduced directly in the production
`termite_training_adamw_f32` shader with identical supplied tensors and scalar
parameters. Compiling that shader through the ordinary fast-math library gave
2.0014e-5 second-moment relative L2 error on 32,768 controlled values; compiling
the same source with precise math reduced it to 1.8764e-8. First-moment errors
were 1.2597e-7 / 3.4413e-8 and incremental-update errors were 3.5965e-6 /
2.5011e-7, respectively. This isolates compiler arithmetic from model execution
and does not depend on paired runs having identical hidden intermediate states.

A reduced eight-element fixture uses gradients +/-2.1, +/-3, +/-1.3 and +/-1e-4,
prior m=0.009, prior v=8e-6 and gradient scale 0.244949072599411. The old shader's
variance error reaches 5.0550e-5 relative; precise compilation matches every
expected F32 variance value exactly. The integrated regression uses an independent
F64 control over the actual F32 parameters, verifies m/v at 5e-7 relative, weights
at 2e-9 absolute, and zeroed gradients through both single and batched APIs.
These new regression bounds do not change any oracle or campaign tolerance.

Only the AdamW pipeline's math-library selection changes. The optimizer formula,
hyperparameters, norm correction, clipping threshold and checkpoint layout stay
the same. The required-device Metal suite passed **332 tests, 2 optional skips**.
Native code is byte-identical to the preceding audited build (284 tests passed,
26 skipped); shipping Python remains unchanged from its 748-test pass.
Probe inputs, exact shader source, fast/precise outputs and hashes are retained
in `optimizer-shader-math-probe`.

<!-- precise-adam-results:start -->
Native code retains its **284-test pass (26 skips)**; the corrected build passed **332 required-device
Metal tests (2 optional skips)**, including the integrated analytic single/batched
AdamW regression. Python remains **748 passed**; no shipping Python source changed
after that run. ReleaseFast and compiled-source identity checks pass.

Binary SHA256 `46620bf9dba98f4708a0a529980d52ee9bc7cf7ffdcae944534b6e199f85279b`;
compiled patch SHA256 `5abb24ace25ecfd7be87b7192b17f58ce6fac8d796a5201449723c15d95edda9`.

Fresh independent trajectories below use unchanged F32 diagnostic references and
unchanged resource limits. All errors are **percent relative L2**; updates are
cumulative from the exact initial adapter. They are not stock BF16 qualification.

| Model | Preset | Steps | HF gradient | HF update | MLX gradient | MLX update |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E2B | all-linear | 1 | 0.023339% | 0.622435% | 0.020048% | 0.476215% |
| E2B | all-linear | 2 | 2.367102% | 2.097940% | 0.113980% | 1.036690% |
| E2B | all-linear | 8 | 96.577096% | 14.409863% | 2.985211% | 0.933048% |
| E4B | all-linear | 1 | 0.003871% | 0.170286% | 0.002915% | 0.136007% |
| E4B | all-linear | 2 | 0.035500% | 0.330446% | 0.023635% | 0.318414% |
| E4B | all-linear | 8 | 28.889947% | 1.916215% | 20.341833% | 1.176305% |
| E2B | qv | 1 | 0.039036% | 1.403575% | 0.027572% | 0.990024% |
| E2B | qv | 2 | 0.160219% | 1.432765% | 0.109327% | 1.010626% |
| E2B | qv | 8 | 2.550247% | 0.650022% | 2.977879% | 0.455724% |
| E4B | qv | 1 | 0.004026% | 0.183772% | 0.002587% | 0.181795% |
| E4B | qv | 2 | 0.015092% | 0.423047% | 0.023042% | 0.440718% |
| E4B | qv | 8 | 0.401481% | 0.178079% | 0.324882% | 0.197156% |

Completed 12 of the 12 planned cells. Campaign terminal state:
`diagnostic-matrix-captured`. Per-target bounds remain unchanged:

| Model | Preset | Step | States outside diagnostic bounds |
| --- | --- | ---: | ---: |
| E2B | all-linear | 1 | 9 / 2208 |
| E2B | all-linear | 2 | 114 / 2208 |
| E2B | all-linear | 8 | 2203 / 2208 |
| E4B | all-linear | 1 | 0 / 2744 |
| E4B | all-linear | 2 | 0 / 2744 |
| E4B | all-linear | 8 | 1806 / 2744 |
| E2B | qv | 1 | 12 / 400 |
| E2B | qv | 2 | 7 / 400 |
| E2B | qv | 8 | 99 / 400 |
| E4B | qv | 1 | 0 / 528 |
| E4B | qv | 2 | 0 / 528 |
| E4B | qv | 8 | 1 / 528 |

Replaying the second update from captured first-step moments and second-step
gradients confirms the correction across all four model/preset combinations.
The control uses the actual F32 beta values and clipping scale; errors below are
**relative L2 (fractions, not percentages)**. This verifies the captured update,
not uncaptured steps 3–8 or full trajectory parity.

| Model | Preset | First moment | Variance | Incremental update |
| --- | --- | ---: | ---: | ---: |
| E2B | qv | 2.6618e-08 | 1.0980e-08 | 2.7225e-07 |
| E2B | all-linear | 3.2739e-08 | 2.0859e-08 | 2.6832e-07 |
| E4B | qv | 2.8552e-08 | 2.2008e-09 | 3.0146e-07 |
| E4B | all-linear | 3.2965e-08 | 3.6316e-08 | 2.7564e-07 |

Precise AdamW compilation closes the demonstrated variance-update precision defect. It
does not waive remaining independent-trajectory failures, native E4B coverage,
stock BF16/HF oracle, categorical GRPO behavior, E4B acceptance, reserved holdout,
adapter recovery, zero-paging qualification or hosted CI. No numerical tolerance
was loosened, no reserved holdout was scored, and no acceptance PASS was issued.

Evidence is retained in `precise-adam-*` and sealed separately in
`/Users/tim/Documents/af/antfly-qualification/20260909/optimizer-precision-investigation`.
The manifest receipt is `precise-adam-archive-receipt.json`; prior evidence is
hash-bound without modifying the older archives.
<!-- precise-adam-results:end -->


### 2026-09-10 categorical rollout and skipped-update integration

The multi-token MLX comparison runner now accepts current seeded categorical
reports through `--categorical-diagnostic`. It validates the explicit sampling
policy, training seed, v8 prompt-order contract and v4 evaluation policy. Sampling
uses original dataset prompt indices after reordering, independent completion
streams, a shared prompt forward, and greedy completion zero only in evaluation.
EOS remains included; the supported recipe keeps budget-truncated completions
unmasked. Policy log-probabilities are scored without temperature or filtering.

The new path accepts duplicate completions and counts their multiplicities when
measuring overlap. A real MLX train-loop probe reproduced the previous distinct-
completion guard failure before this correction. Equal-reward groups and groups
over the raw KL budget now skip Adam without advancing its step or KL coefficient.
The KL trace validator checks chronological group/prompt IDs, optimizer step IDs,
admission decisions, coefficient continuity, reward-derived skips and report
counts. An exact F32 budget-boundary regression covers admission at the limit.
All-skipped lanes return zero updates and absent update metrics, without dividing
by zero or comparing an unmodified adapter as a successful update.

This is deliberately a diagnostic artifact contract. The new v2 result has
`status=diagnostic-completed` and `classification=categorical-diagnostic-only`;
it cannot return the historical bounded-parity classification even when the
single-seed thresholds pass. Failed Antfly evaluation reports can be inspected
only through this diagnostic mode. The default acceptance guard remains in place.
The existing bounded recipe is unchanged: q/v, rank 16/alpha 32, length 128,
learning rate 1e-7, one epoch, 2–8 completions and a 2–32-token budget. It does not
yet cover the release recipe's all-linear, group-16, single-token campaign or the
legacy single-token MLX runner.

Four serial real-model rollout probes cover E2B/E4B and stock BF16/aligned F32
MLX forwards. Each uses the exact seed adapter and locked prepared diagnostic
prompt, seeds 17/991, train/evaluation domains, four completions and three tokens.
The retained full-vocabulary logits are replayed through verbatim production Zig
sampling and normalization functions using Zig 0.16. **All 192 tokens and their
192 F32 log-probabilities match exactly.** This isolates the sampling contract on
identical input logits; it is not native-versus-MLX forward or training parity.
Independent full-sequence MLX rescoring remains inside the unchanged 1e-4 bound:

| Model | Activation | Exact selections / scores | Max rescore error | Sampled RSS | Swap growth |
| --- | --- | ---: | ---: | ---: | ---: |
| E2B | aligned F32 | 48 / 48 | 3.3004e-07 | 5041.4 MiB | 0.00 MiB |
| E4B | aligned F32 | 48 / 48 | 9.4298e-07 | 6181.0 MiB | 917.25 MiB |
| E2B | stock BF16 | 48 / 48 | 4.7684e-07 | 6639.8 MiB | 0.00 MiB |
| E4B | stock BF16 | 48 / 48 | 7.1526e-07 | 4812.0 MiB | 0.00 MiB |

All probes completed within 900 seconds, 16 GiB sampled RSS and 1 GiB swap growth.
Every run started with nonzero swap. These are numerical diagnostics; no memory,
performance or zero-paging qualification is claimed. The captured logits and
helper hashes are retained, including both activation paths and independent Zig
replays. Frozen model weights and prepared-source identities were verified.

The production nested train loop also ran with controlled completions/gradients
and actual MLX compiled accumulation, clipping and AdamW. Both trace replay and
native rollout pass mixed-group and all-skipped cases (four cases total). Mixed
runs produce exactly two Adam steps, with parameter values and all optimizer
state bit-identical to a two-update control. Zero-variation and KL-rejected groups
preserve the coefficient; the F32 KL boundary is admitted. All-skipped runs keep
parameters, optimizer step and coefficient unchanged. This verifies the loop's
state transitions with controlled model outputs, not real-model GRPO gradients.

<!-- categorical-validation:start -->
Gemma4 Python: **758 tests passed** on the final source. The initial sandboxed
run hit one loopback-bind permission error; the isolated retest and final full
suite passed with loopback access. Source audit confirms the compiled Zig patch
and ReleaseFast binary still match the precise-Adam build. Unresolved-conflict,
whitespace and Zig-format checks pass. No Zig implementation changed in this
categorical follow-up; its earlier 332 Metal and 284 native test passes are
carried forward.
<!-- categorical-validation:end -->

Evidence: `.benchmark-results/gemma4-qualification-20260909/categorical-runner-status.json`,
`categorical-mlx-update-state.json`, `categorical-production-replay-summary.json`
and `categorical-stock-production-replay-summary.json`. Source-bound diagnostic
scripts, full logits, excluded failed probe logs and final test logs are retained.
The separate `categorical-rollout-investigation` archive binds the preceding
`optimizer-precision-investigation` manifest
`928f55c52251cc276df8aedf71ae7b5819c257db1f9a716be6039f0f70241b5a`.

Full independent trajectory failures remain unchanged: full-preset eight-step
Metal/HF gradient errors are 96.5771% E2B and 28.8899% E4B. Native E4B coverage,
locked HF/stock BF16 numerical qualification, optimizer-backed real-model
categorical campaigns, statistical behavior, E4B GRPO quality acceptance, the
reserved holdout, reverse adapter import/recovery, zero-paging qualification and
hosted CI remain open. The reserved 254 examples were not read or scored. CUDA
implementation remains outside scope. No tolerance was loosened or acceptance
PASS published.


### 2026-09-10 precise parallel training RMSNorm follow-up

The full-trajectory investigation identified a separate forward precision defect.
The default Metal RMSNorm row kernel serially accumulates F32 squares. Replaying
that exact kernel on retained E2B embedding activations reproduces the earlier
first-layer RMSNorm drift: 1.8215372e-6 relative L2 versus the saved HF output.
Precise compilation alone leaves 1.7521467e-6 error. The existing parallel row
kernel, compiled precisely, reduces that error to 8.4371117e-8. Against an
independent F64 control, error falls from 1.7105782e-6 to 5.9091236e-8.

An analytic row with one large element and 1535 values of 1e-4 reproduces loss
of small squared contributions: the serial result has 6.3201941e-6 relative L2
error versus F64, while the parallel result has 2.6379716e-8. The source-bound
probe retains all four combinations of serial/parallel and fast/precise math,
including the control that shows precise compilation alone is insufficient.

Gemma4 training now requests a precise 256-lane row reduction explicitly through
its Metal compute backend. The device API reuses the existing access tracking,
frame ownership and bounds checks; the precision request fails if the precise
pipeline is unavailable. Three optional inference norm fusions decline this
training request so the ordinary precise norm fallback can execute. Inference
defaults are unchanged. The public legacy device entry point keeps its ABI and
behavior. Native arithmetic and the AdamW recipe are unchanged.

The new required-device regression checks F64-derived normalized outputs at
widths 128, 1536 and 2560, including signs and non-unit norm weights. It passes
the 2e-7 relative bound; the reproduced serial error exceeds that bound.
Required-device Metal Debug: **333 passed, 2 optional skips**.

A separate HF self-control repeats the unchanged recipe at 1/2/8 steps. All
3312 arrays at each captured step are bit-identical with the original six-thread
reference, including gradients, weights and optimizer state. Running with one
CPU thread changes the eight-step gradient by 1.165219% and cumulative update
by 0.112575%. This measures some reduction-order sensitivity, but is far below
the outstanding 96.58% Metal/HF full-preset gradient difference and cannot
justify waiving it. The three byte-identical six-thread captures were replaced
with verified APFS clones of their references; paths, contents, hashes and
timestamps are preserved. The clone receipt records 1,897,478,736 shared bytes.

<!-- training-rmsnorm-results:start -->
ReleaseFast and source/binary identity checks pass. The native-only Debug build
passed **284 tests (26 skips)**. The prior **758-test Python pass** is carried
forward: every changed shipping script and fixture remains hash-identical to the
sealed categorical-runner source; no Python implementation changed here.

Binary SHA256 `b6587c5b9b166fbbc98fed6eb2e19a3a018f249b5fe562b59a8e8206da1d34b4`;
compiled patch SHA256 `d91c2472382bb095eac5f5b181649f10c41d0deaba343c5ab5e42c686cb16f1f`.

Completed 12 / 12 planned cells; terminal campaign state
`diagnostic-matrix-captured`. References, recipes and tolerances are unchanged.
All errors below are **percent relative L2**; updates are cumulative from the
exact initial adapter. HF CPU F32 and aligned MLX are diagnostic references.

| Model | Preset | Steps | HF gradient | HF update | MLX gradient | MLX update |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E2B | all-linear | 1 | 0.023176% | 0.595130% | 0.022688% | 0.505020% |
| E2B | all-linear | 2 | 2.368048% | 2.070529% | 0.122138% | 1.087040% |
| E2B | all-linear | 8 | 96.522230% | 14.400830% | 3.219215% | 0.999742% |
| E4B | all-linear | 1 | 0.003807% | 0.165992% | 0.002635% | 0.131526% |
| E4B | all-linear | 2 | 0.035186% | 0.330624% | 0.022701% | 0.312394% |
| E4B | all-linear | 8 | 29.935476% | 2.212523% | 7.961315% | 0.776499% |
| E2B | qv | 1 | 0.036678% | 1.351174% | 0.030092% | 1.051475% |
| E2B | qv | 2 | 0.155219% | 1.370880% | 0.116927% | 1.084301% |
| E2B | qv | 8 | 2.253734% | 0.619507% | 3.268900% | 0.494194% |
| E4B | qv | 1 | 0.003923% | 0.167720% | 0.002312% | 0.176397% |
| E4B | qv | 2 | 0.010845% | 0.286552% | 0.021304% | 0.517671% |
| E4B | qv | 8 | 0.289726% | 0.119526% | 0.500604% | 0.223024% |

Per-target states outside the unchanged diagnostic bounds:

| Model | Preset | Steps | Exceeded / compared |
| --- | --- | ---: | ---: |
| E2B | all-linear | 1 | 8 / 2208 |
| E2B | all-linear | 2 | 110 / 2208 |
| E2B | all-linear | 8 | 2203 / 2208 |
| E4B | all-linear | 1 | 0 / 2744 |
| E4B | all-linear | 2 | 0 / 2744 |
| E4B | all-linear | 8 | 1907 / 2744 |
| E2B | qv | 1 | 10 / 400 |
| E2B | qv | 2 | 5 / 400 |
| E2B | qv | 8 | 97 / 400 |
| E4B | qv | 1 | 0 / 528 |
| E4B | qv | 2 | 0 / 528 |
| E4B | qv | 8 | 0 / 528 |

E4B q/v Metal/HF CPU F32 is within every unchanged per-target bound at steps 1, 2 and 8.
In particular, step eight improves from one failed state to **0 / 528**.
This closes the measured F32 diagnostic target-state gap for that preset;
it does not qualify stock BF16 or independent all-linear trajectories. The
additional scalar and per-position probe audit also has no failures for E4B q/v:
144 checks at step one, 145 at step two and 151 at step eight. It covers the full
captured loss history, raw gradient norm, sampled logits and logsumexp values
using the existing profile functions and limits. This still samples logits and
checks one prepared training example, not the full quality dataset.

Independent second-update replay uses the captured first-step weights/moments,
second-step gradients, recorded clipping scale and actual F32 beta values.
The errors below are **relative L2 fractions**, not percentages. They verify
these captured updates, not uncaptured optimizer steps 3–8.

| Model | Preset | First moment | Variance | Incremental update |
| --- | --- | ---: | ---: | ---: |
| E2B | qv | 8.8791e-08 | 1.7619e-07 | 2.7783e-07 |
| E2B | all-linear | 3.6045e-08 | 1.4804e-08 | 2.6824e-07 |
| E4B | qv | 2.8687e-08 | 2.3553e-09 | 3.0125e-07 |
| E4B | all-linear | 3.2922e-08 | 3.7233e-08 | 2.7562e-07 |

The current full-preset first-step amplification audit still finds opposite-sign
gradients in 1,189 / 15,204,352 active E2B components and 237 / 21,331,872 E4B
components. Those positions account for 69.37% and 51.43% of squared update
difference respectively. Algebraic first-step Adam replay agrees with each
framework's captured update within 2e-7 relative L2. These observations locate
substantial initial amplification at near-zero gradients; they do not establish
the cause of every subsequent trajectory difference.

An additional HF control extends the physical tensor from 154 to 160 rows with
causal tail padding and ignored labels. Original tokens, supervised labels,
adapter, six-thread CPU execution and AdamW recipe are unchanged. Its eight-step
gradient/update drift against the 154-row reference is **0.418578% / 0.040194%**.
This is a sensitivity experiment, not a replacement oracle or a changed gate.
Only the final eight-step tensor state is retained for this padding control.

The RMSNorm precision defect is corrected, but full independent trajectory
parity remains unqualified. Existing paging excludes memory/performance acceptance.
Native E4B coverage, locked HF/stock BF16 numerical qualification, real-model
categorical campaigns and statistical behavior, E4B GRPO quality acceptance,
reserved holdout, reverse adapter import/recovery, zero-paging qualification and
hosted CI remain open. CUDA implementation remains outside scope. No numerical
tolerance was changed, reserved holdout scored, or acceptance PASS issued.

Evidence is sealed in `training-rmsnorm-investigation`, with receipt
`training-rmsnorm-archive-receipt.json`. The archive binds the preceding
`categorical-rollout-investigation` manifest
`6ee92bb24743996991a7e902a137245c7a8ecf1a2f6d6ffef047aeb30160e6d2`.
Raw captures, controls, source/binary snapshots and excluded attempts are retained.
<!-- training-rmsnorm-results:end -->

### 2026-09-10 backward isolation and full GRPO recipe coverage

The next numerical control captures selected RMSNorm inputs and upstream
loss derivatives from the unchanged HF CPU F32 E2B all-linear first step.
All **552 adapter gradients remain byte-identical** to the retained reference,
so the hooks do not alter this captured training computation. The control
covers 20 norm operations in layers 0, 17 and 34 and the final norm. Frozen
operations without a backward graph are not included.

Both production Metal RMSNorm backward reductions were replayed verbatim, each
with fast and precise compilation. The default SIMD-group kernel's worst
relative L2 error across those inputs is **1.3448e-7 versus isolated HF autograd**
and **7.6377e-8 versus an independent F64 derivative**. There are no sign
reversals against the F64 control. Precise compilation improves 16/20 cases
against F64, but is not uniformly better; it is not established as a fix for
the outstanding trajectory divergence. The rollback tree was checked separately
and is not confused with the default SIMD-group dispatch. Native serial F32
emulation reaches 8.1908e-7 error; that emulation is not a native-runtime capture.
No RMSNorm backward implementation was changed on this evidence.

The MLX campaign runner now exposes two explicit, pinned recipe profiles:

| Profile | Targets | Group | Completion budget | Sequence length | Learning rate | Advantage epsilon |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qv-multitoken` (default) | `peft-qv` | 2–8 | 2–32 | 128 | 1e-7 | 1e-4 |
| `all-linear-single-token` | `text-all-linear` | 16 | 1 | 160 | 5e-8 | 1e-8 |

Select the full profile with
`--recipe-profile all-linear-single-token --group-size 16 --max-completion-tokens 1 --categorical-diagnostic`
on `run_gemma4_grpo_boolq_mlx_multitoken.py`, alongside its existing attested
runtime, dataset, seed adapter and Antfly campaign arguments. It requires
categorical reports, explicit all-linear targets and the pinned temperature 2,
top-p 1, top-k 32 policy. Profile values flow through materialization checks,
row admission, model target selection, physical padding, advantages, AdamW and
the output contract. Full-profile results remain diagnostics and cannot publish
statistical parity or production acceptance. The historical default is retained.

Validation: **760 Gemma4 Python tests passed**. Contract fixtures exercise both
profiles, shuffled duplicate completions, failed-quality diagnostic evidence,
recipe mismatches, fixed group/token budgets and the 159-token prompt boundary
that reserves one position in a 160-token sequence. Source-bound probes execute
the production padding functions, optimizer factory and nested compiled MLX
training loop for both profiles. All eight mixed/all-skipped and trace/native
cases pass. Mixed groups produce two admitted Adam updates with weights and
complete optimizer state byte-identical to their controls. All-skipped groups
leave weights, optimizer step and KL coefficient unchanged. Model scoring and
gradients are controlled in these loop probes; they do not establish real-model
GRPO gradient or quality acceptance. An initial probe's incorrect expected tensor
shape failed before training; its source and log remain excluded and retained.

No compiled Zig/Metal source changed in this follow-up. The sealed precise
RMSNorm build, its 333 Metal / 284 native test passes and the 12-cell diagnostic
matrix remain applicable. E2B q/v and both independent all-linear trajectories
still fail; E4B q/v's measured HF CPU F32 case remains within bounds. Full
optimizer-backed real-model categorical campaigns, E4B GRPO quality acceptance,
stock BF16/locked HF numerical qualification, native E4B coverage, reverse adapter
import/recovery, zero-paging qualification and hosted CI remain open. The reserved
254-example holdout remains untouched; CUDA implementation remains out of scope.

Evidence: `.benchmark-results/gemma4-qualification-20260909/recipe-profile-status.json`,
`rmsnorm-backward-simdgroup-probe/report.json`, `recipe-profile-python-tests.log`
and the two `recipe-profile-mlx-update-state-*.json` reports. The separate
`backward-and-recipe-investigation` archive binds the previous
`training-rmsnorm-investigation` manifest
`1c6f93b015d1a86b81012fcc86a94c3c9f422c952b112c55f85c9d24d94e1a13`.

### 2026-09-10 optimizer-backed GRPO integration and failed-run evidence

The retained all-linear/group-16/one-token recipe was exercised on original
training prefixes and the first two fresh diagnostic evaluation rows. These
small integration cases are not the 1960-row training / 256-row diagnostic
campaign and do not replace the reserved acceptance holdout. Learning rate,
clipping, sampling, reward computation and quality thresholds are unchanged.

The first two training groups have no within-group reward variation. The CLI
correctly returns `NoGrpoLearningSignal`. A 16-group prefix instead completes
two optimizer steps, skips 14 zero-variation groups and then fails the absolute
evaluation gate: mean reward 0.46875 and positive-reward group rate 0.5, below
the unchanged 0.75 group-rate minimum. This exposed an evidence-retention defect:
the terminal evaluator returned before `grpo_report.json` could be written.

The GRPO loop now writes the evaluation and training summaries before returning
the quality error. Accepted-adapter publication requires **both** absolute and
baseline-relative gates. A real rerun verifies two optimizer steps in the saved
report, `evaluation.passed=false`, `trained_adapter_dir=null`, and no
`adapter-trained` directory. Reward traces, KL-controller traces and evaluation
metrics are byte-identical before/after this control-flow change. The child
still returns `GrpoEvaluationGateFailed`; neither the threshold nor the adapter
publication gate was bypassed.

Required-device Metal Debug: **333 passed, 2 skips**; native-only Debug:
**284 passed, 26 skips**. ReleaseFast completes with binary SHA256
`2420d0f9c1956b9dcf98e27a46232d750a9a62bbc9f7ff6bdfdf5b4920b9744a`
and compiled patch SHA256
`c9c091579c6dbd338f73181937e8dab4c4b5f49c656060d2cb80b78c68853036`.
The numerical kernels are unchanged from the prior RMSNorm matrix. The real
E2B rerun finishes in 58.37 diagnostic seconds, with sampled peak RSS 10,093,328
KiB and no growth from 2,101.81 MiB existing swap. This is not zero-paging
qualification or a performance result.

The production MLX loader now binds the original training manifest and the
separately materialized evaluation manifest through
`--evaluation-dataset-manifest`. It checks each source manifest independently,
requires matching dataset revision, tokenizer and dependency identities, checks
cross-source disjointness, and records both manifest digests and selection
policies. The 128-token training admission limit remains distinct from the
160-token evaluation admission and physical execution lengths. No rows or
source manifests are relabeled to imply a new materialization.

Real serialized reports also exposed fixture gaps: categorical sampling includes
an algorithm and stream-derivation identity, F32 clip bounds serialize as
0.20000000298023224, and omitted `normalize_advantage` serializes as null with a
true default. The loader now checks these actual representations and rejects
unknown sampling identities, altered clip values and explicit false advantage
normalization. `failed-quality-gate` is admitted only as diagnostic evidence.
When an explicit failed quality summary has no accepted adapter, diagnostic
replay/rollout may proceed but adapter-vector parity remains unavailable; a
missing adapter cannot be silently accepted for a successful quality report.

The first MLX preflight correctly failed source attestation: the retained source
was an extracted archive, and Git reported the parent Antfly revision. A fresh
archive fetched from the official pinned MLX-LM commit matches the retained
archive SHA256
`67e1a52f9b86551a24eab1aa2681c26a391819925bb10d14792b96a88303ebc7`.
`--mlx-lm-source-archive` now supports that explicitly pinned archive, verifying
all 212 source files and the complete directory inventory. Unknown revisions,
changed archives/files, symlinks, escaping members and generated bytecode caches
are rejected. Archive execution suppresses bytecode writes. The existing Git
checkout path remains available. Installed MLX 0.31.2 runtime files also match
the exact pinned MLX and MLX-Metal wheel contents; no Git state was modified.

<!-- full-recipe-integration-results:start -->
Final Python validation: **766 tests passed**, with source hashes verified
unchanged across the suite. The actual failed-quality E2B report passes the
production loader using both original dataset manifests and no accepted adapter.

The matched stock-BF16 MLX run passes wheel/source/dataset/report admission but
stops after 157.99 seconds at the **2 GiB free-disk guard**. Its last sample has
1,874,395,136 free bytes (1.75 GiB), below 2,147,483,648. Sampled peak RSS is
5,428,704 KiB, below 16 GiB. Swap grows 970.19 MiB from 2,101.81 MiB, below the
1 GiB growth cap; the disk limit triggers first. No completed MLX campaign
result exists. The full Python suite starts more than 100 seconds after this
model process exits, so it does not cause that resource stop. No guard was
relaxed, incomplete capture promoted, or statistical/quality PASS emitted.

Current disk headroom prevents further admitted model runs. The inference
`.zig-cache` occupies about 40 GiB (39 GiB in object entries), but cache/evidence
preservation constraints require approval before deletion. Existing binary,
source and raw diagnostic evidence are retained independently. Cache cleanup
alone would not satisfy the separate zero-paging qualification requirement.

The new `full-recipe-integration-investigation` archive retains the source,
binary, tests, original failed attempts, failed-quality reports/traces, source
archive/wheel attestations, and MLX resource samples. It binds the previous
`backward-and-recipe-investigation` manifest
`6619dfe1d664979ae71d6290f347e5fcc1b1f67c938da5d76adf96cf29441f2d`.
Full independent numerical parity, complete E4B GRPO quality/statistical
acceptance, native E4B coverage, locked HF/stock BF16 qualification, accepted
adapter recovery, reserved holdout, zero-paging performance/memory and hosted CI
remain open. CUDA implementation remains outside scope.
<!-- full-recipe-integration-results:end -->

### 2026-09-10 parity resume: observable MLX attempts

The disk preflight still reports approximately 1.3 GiB free, below the unchanged
2 GiB diagnostic reserve. No further model job was launched and the request to
remove the approximately 40 GiB inference build cache remains pending. The
retained source, executable and failed-run evidence remain available in the
separate `full-recipe-integration-investigation` archive.

The MLX campaign now flushes structured progress events to stderr before input
and runtime validation, provenance hashing, model loading/materialization,
adapter installation, reference scoring, rollout, and optimizer updates. Events
include elapsed time, lane and group where applicable. They do not constitute
completed results or acceptance evidence. This addresses the empty log from the
prior resource stop; its exact execution stage remains unknown.

The retry wrapper
`.benchmark-results/gemma4-qualification-20260909/run_full_recipe_mlx_stage_retry.py`
requires a distinct attempt suffix and creates log/execution artifacts exclusively,
preserving previous failures. It retains the 900-second, 16 GiB RSS, 1 GiB swap
growth and 2 GiB free-disk limits and records which resource limit triggered.
The focused campaign/source-attestation suite passes **27 tests** after the
logging change; `git diff --check` passes and no unresolved conflict paths exist.
The earlier 766-test suite applies to the previously sealed source revision.
No numerical tolerance, optimizer computation, quality threshold or publication
gate changed. Full numerical and E4B GRPO acceptance gaps remain open.

### 2026-09-10 approved cache cleanup and completed E2B GRPO comparison

Explicitly approved removal of `zig/pkg/inference/.zig-cache` recovered
42,420,445,184 bytes (39.5 GiB). The preserved executable and separate archive
hashes were checked before deletion. Other caches, source, models and diagnostic
evidence were retained. `approved-inference-cache-cleanup.json` records the action.

The first resumed MLX attempt was stopped when review found final result
serialization still dereferenced an absent accepted adapter. The nullable digest
now matches failed-quality admission; an advertised checkpoint must still exist.
The next attempt reached its first optimizer update and exceeded the unchanged
1 GiB swap-growth guard. Both excluded attempts and their distinct causes remain
recorded.

`--completion-execution sequential` now evaluates each completion's gradient
before proceeding to the next. Accumulation order, clipping and one optimizer
update per admitted group are preserved. The default remains `compiled-group`.
Source-bound controls cover both recipe profiles and modes: mixed/skipped cases
retain exact Adam state, and a nonlinear model's weights, optimizer state and
metrics are byte-identical across eight updates (192 compared arrays).

The real E2B stock-BF16 sequential comparison completes in 365.17 diagnostic
seconds with sampled peak RSS 8,926,896 KiB and 0.13 MiB swap growth from
3,383.25 MiB existing swap. Replay and native rollout both match all 256 Antfly
training completion tokens, update at group indices 10 and 14, skip 14 groups,
and reject none for KL. Final report serialization succeeds with no accepted
Antfly adapter. This closes the execution/evidence gap for this small case.

Held-out behavior still differs: MLX mean reward is 0.4375 versus Antfly 0.46875;
both have top-ranked reward 0.5 and positive-group rate 0.5. MLX's evaluation
token difference is already present at baseline. Its post-training KL loss is
6.8528961565e-5 versus Antfly 1.9728758904e-10. This result is explicitly
`categorical-diagnostic-only`, with no adapter-vector, quality or statistical
parity claim. It uses 16 original training rows and two fresh diagnostic rows.

The optional `--activation-mode aligned-f32` uses the retained numerical
comparison method: F32 text/per-layer input embeddings with frozen BF16 weights.
It is admitted only for categorical diagnostics. Its first sequential attempt
stops at the swap guard during backward; `--gradient-checkpointing` exposes the
pinned MLX-LM layer helper for a further bounded attempt. Source hash and execution
options are recorded. Stock BF16 and aligned F32 remain distinct reference lanes.

<!-- cache-cleared-grpo-final-results:start -->
The optional `--shared-single-token-scoring` path gathers one-token completion
scores from the shared causal predictor row, retaining physical batch 1 and
sequence length 160. BF16/F32 causal-model controls match all 96 selected scores
exactly at prompt lengths 1, 17 and 159 while reducing 16 scoring forwards to one.
Multi-token input is rejected. The original per-completion path remains the
default, and real training retains its differentiable-rescore error checks.

With shared scoring and pinned checkpointing, the aligned-F32 E2B comparison
completes in 130.94 seconds, with sampled peak RSS 8,150,368 KiB and no growth
from 2,546.44 MiB swap. Replay and native rollout both match all 256 training and
32 evaluation completion tokens, the two updates, 14 skips, and reward metrics.
Native-rollout KL loss is 2.0152669190e-10 versus Zig 1.9728758904e-10; replay is
1.8271487151e-10. Maximum differentiable-rescore error is 2.4587e-7 in native
rollout and zero in replay. The E2B quality gate still fails in both frameworks,
and no accepted E2B adapter exists for vector comparison. This closes the
observed token/behavior discrepancy for this small precision-aligned case.

The original E4B 16-group integration reaches terminal two-row evaluation but
hits the 180-second limit while serializing a candidate adapter. A partial
SafeTensors file remains inside an unpublished staging directory. Gemma4 bundle
publication now uses the existing buffered SafeTensors writer instead of issuing
an individual write for each four-byte float. The writer preserves tensor bytes,
aligns the payload and flushes/syncs before atomic publication. Post-publication
change validation reuses canonical tensor identity, so metadata, order and
supported PEFT namespace differences cannot make unchanged weights appear trained.

The corrected E4B run completes in **155.52 seconds**, with sampled peak RSS
15,649,888 KiB and no growth from 2,634.44 MiB existing swap. It performs all
16 optimizer updates, writes its complete GRPO report and publishes the adapter.
Mean reward improves from 0.15625 to 0.1875, top-ranked reward is 0.5,
positive-group rate is 1.0, and KL loss is 7.4133022281e-6. Both absolute and
baseline-relative gates pass for these **two diagnostic evaluation rows**.
Training, KL-control and final evaluation traces are byte-identical to the prior
attempt. Baseline traces differ only in their configuration digest, which binds
the distinct baseline trace/exchange paths under each output root. Evaluation
metrics are exactly equal. This is not the full 1960/256/reserved-holdout campaign.

ReleaseFast binary SHA256:
`707daaa63bbf578c487ab7efc5d6024b93a1076c92cac406f050356383ff4bf0`;
compiled patch SHA256:
`e2819ef44745441febae8966207e19b2b9fba02fd7747f895a45feccfe74d01a`.
E4B GRPO report SHA256:
`34f4de9d571d483fe9c8bb3c67a87ed650108c845544f778b3a57431341ee975`;
published adapter checkpoint SHA256:
`7cef793db70bcea9afbc209b2b2508934f2dc38beb683c6db85961f6f91e04ca`.

The matched stock-BF16 E4B MLX comparison completes in 672.84 seconds, sampled
peak RSS 8,512,192 KiB, with no growth from 2,546.44 MiB swap. Across 686 adapter
tensors, replay delta cosine is 0.9983599 and vector relative-L2 error is
0.0572903 (5.73%), within the existing bounded diagnostic check. Native rollout
passes the small absolute quality gate with mean reward 0.1875 and positive-group
rate 1.0. Tokens still differ: mean training multiset recall is 0.8828125 and
evaluation recall is 0.875. Evaluation KL loss is 2.3552683878e-4. Classification
remains `categorical-diagnostic-only`.

The aligned-F32 E4B comparison completes as two independently guarded processes:
`--execution-lane trace-replay` and `--execution-lane native-rollout`. Each restores
the same initial adapter and fresh optimizer; explicit categorical sampling seeds
are unchanged. Individual lanes require categorical diagnostic mode, emit
`diagnostic-lane-completed`, and use null for unmeasured lane results/checks.
The default `both` behavior is preserved. The earlier combined attempt was
explicitly stopped because its projected total exceeded the 900-second guard;
it is excluded and retains its stop reason. No time or memory guard was raised.

Replay completes in 549.17 seconds (6,914,016 KiB sampled peak RSS, 484.50 MiB
swap growth from 2,473.69 MiB); independent rollout completes in 546.92 seconds
(8,806,928 KiB RSS, no growth from 2,902.19 MiB swap). Both complete all 16 updates,
reject no groups for KL, and pass the unchanged absolute and baseline-relative
checks on the two diagnostic evaluation rows. The pairing receipt verifies exact
model, data, seed-adapter, source and runtime identities, plus identical baseline
results apart from timing.

Across **686 adapter tensors**, aligned replay update relative-L2 error is
**0.0002383601 (0.023836%)**, versus stock BF16's 5.73%; cosine is
**0.9999999728**, maximum absolute delta difference is 6.9362e-8, and update norm
relative difference is 5.8850e-7. This is a comparison on forced Antfly completion
traces; independent-rollout adapter-vector distance is not measured.

Independent MLX rollout matches **255/256 training tokens and all 32 evaluation
tokens**. The only mismatch is group 0, prompt index 14, completion index 4:
MLX token 2717 versus Zig token 507; both receive zero reward. It occurs before
any optimizer update. The corresponding forward logits were not captured, so
this does not establish a rounding or sampling-boundary cause. Subsequent sampled
groups match in both replay and independent rollout. Mean evaluation reward is
0.1875 versus baseline 0.15625 in both frameworks, top-ranked reward is 0.5,
and positive-group rate is 1.0. MLX replay/native KL losses are
7.4210485168e-6 / 7.4380689799e-6 versus Zig 7.4133022281e-6.
Maximum differentiable-rescore error is zero in replay and 9.5367e-7 in independent
rollout. No exact-token or general numerical parity claim is issued.

The newly published E4B checkpoint also passes a fresh stock-PEFT load check:
343 target modules, **686 exact F32 tensors**, and 343 nonzero trained B tensors.
The check maps the pinned HF text weights and loads the Python-translated adapter
using unmodified PEFT; unused modality weights remain meta. It completes in
18.14 seconds with sampled peak RSS 3,661,840 KiB and no swap growth. This verifies
loading and values, not forward quality, reverse import, or checkpoint recovery.
See `cache-cleared-e4b-aligned-lane-comparison.json`,
`cache-cleared-e4b-token-difference.json`, and
`cache-cleared-e4b-published-stock-peft/report.json`.

Required-device Metal validation passes **334 tests with two skips**; native-only
validation passes **285 with 26 skips**. These include buffer-boundary/signed-zero
preservation and unchanged-weight rejection despite header/order/namespace changes.
An initial test fixture misused an ownership-taking writer and aborted; its source
and log are retained, and the corrected fixture passes. The final Python suite
passes **770 tests in 70.47 seconds**, with identical full-checkout hashes before
and after execution and matching Python source hashes. The earlier final-suite
attempt had one harness error while this document was being edited; that harness
checks full-checkout identity, making the edit a likely cause, but its original
child stderr was not retained. The isolated test and frozen-checkout full rerun
pass. The excluded attempt, timestamp evidence and subsequent logs remain archived.
An earlier Python attempt also required host access for a localhost-bind test.
No numerical tolerance or quality threshold was relaxed.

All these model runs are local diagnostics with pre-existing swap. Full independent
numerical trajectories, full E4B GRPO/statistical acceptance, accepted-adapter
recovery, native E4B coverage, locked HF/stock BF16 qualification, zero-paging
memory/performance and hosted CI remain open. The reserved holdout is untouched.
See `cache-cleared-status.json` and distinct execution reports under
`.benchmark-results/gemma4-qualification-20260909/`. The completed evidence is
sealed in `/Users/tim/Documents/af/antfly-qualification/20260909/cache-cleared-grpo-investigation`,
with a per-file SHA-256 manifest, the final executable/source, the exact pre-lane
runner source used by the completed E2B aligned/E4B stock comparisons, and the
published E4B/translated PEFT artifacts. `cache-cleared-archive-receipt.json`
records the manifest hash. The prior integration archive remains unchanged.
<!-- cache-cleared-grpo-final-results:end -->

### 2026-09-10 MLX-first predictor and common-optimizer-state investigation

MLX is the active on-Mac numerical reference. HF/PEFT comparisons and CUDA work
are deferred by request. This section supersedes the preceding unmeasured
independent-rollout adapter distance and unexplained initial-token mismatch.
It adds diagnostic evidence; full statistical acceptance remains open.

The MLX campaign can now optionally retain the first training predictor row with
`--capture-initial-training-logits`, admitted only in categorical diagnostic mode.
It captures the original logits, prompt IDs/index, source identity, physical
sequence length and predictor position, without changing sampling. Later
completion forwards cannot overwrite it. Final adapter-vector comparison now
also runs for independent rollout when an accepted Antfly adapter is available.
Neither addition changes numerical tolerances or the diagnostic classification.
Runner SHA256: `147082eff0626402a5b50147f37b6fd60fab8e58be2597976a088bb6074b91ab`.

An isolated snapshot of 3837 Zig-tree source files adds a single diagnostic hook
at the original E4B training predictor. It writes the raw F32 row, prompt, copied
PRNG draws and production-sampled tokens, then exits with the deliberate
`DiagnosticPredictorCaptured` error before the first update. Shipping Zig source
and the previously verified ReleaseFast binary are unchanged. The probe's
40.26-second run reproduces all 16 original group-zero tokens and all 16 random
draws exactly. Its intentional exit is capture evidence, not a training pass.
The first sandboxed build failed to create a compiler-cache manifest; the same
source/arguments built with host access. Both build records are retained.

The refreshed independent aligned-F32 E4B MLX lane completes in 554.99 diagnostic
seconds, with sampled peak RSS 6,778,400 KiB and no growth from 2844.69 MiB swap.
It completes 16 updates, matches **255/256 training and 32/32 evaluation tokens**,
and passes the unchanged quality minimums on the same two diagnostic rows:
mean reward 0.1875, top-ranked reward 0.5, positive-group rate 1.0, and KL loss
7.4380689799e-6. Across **686 adapter tensors**, independent update-vector
relative-L2 error is **0.0404199885 (4.0420%)**, cosine **0.9991838985**, and maximum
absolute delta difference **5.78684e-7**. The prior replay result remains
**0.023836%** on forced Zig completion traces. Independent and replay results
measure different trajectories and are reported separately.

The first refreshed MLX attempt stopped after 14.50 seconds at the unchanged
1 GiB swap-growth guard, before any update or predictor capture. It is excluded.
After the host settled, one retry completed under the same 900-second, 16 GiB
RSS, 1 GiB swap-growth and 2 GiB free-disk limits. Logs and execution records for
both attempts are preserved.

The captured initial row identifies the one-token mismatch precisely:

| Initial predictor, group 0 / prompt 14 | Zig | Aligned-F32 MLX |
| --- | ---: | ---: |
| Token 507 logit | 15.6763658524 | 15.6764316559 |
| Token 2717 logit | 15.6764192581 | 15.6764259338 |
| Rank 24 / rank 25 (one-based) | 2717 / 507 | 507 / 2717 |
| Completion 4 token | 507 | 2717 |

Both have the same top-32 set, but these adjacent candidates swap order. The
identical random draw **0.969369504498047** lands in rank 25's interval in both
runs, approximately **[0.9664584, 0.9712862)**; it is not close to the interval
boundary. Thus the changed token is caused by the near-tied logits exchanging
rank, rather than a PRNG or CDF-boundary discrepancy. Full-row relative-L2 error
is **5.42315e-6**, RMS difference **3.23678e-5**, and maximum absolute difference
**1.68800e-4**. The row has 262144 logits, 85 prompt tokens, and physical sequence
length 160. Verbatim production Zig sampling functions and the Python sampler
both reproduce all 16 recorded draws/tokens on each captured row. No logit
rounding or tie tolerance was introduced to force token agreement.

A separate E2B all-linear diagnostic starts both implementations from identical
**MLX step-two weights, Adam first/second moments, and optimizer step 2**. The
production Zig CLI generates valid epoch-boundary checkpoints for a three-epoch
run. The retained epoch-two checkpoint supplies its original fingerprint,
trainer/RNG counters and zero gradient accumulators; only its weights and moments
are replaced with verified MLX tensors. The normal `--resume` path validates it
and executes exactly epoch three. No oracle admission check is bypassed.

The original first training example is prepared with physical length 154 and
71 supervised tokens; its IDs and labels exactly match the retained numerical
fixture. A distinct one-record evaluation fixture satisfies CLI disjointness.
This is numerical instrumentation, not a quality data set. Both frameworks use
all-linear/r16/a32, learning rate 0.001, AdamW betas 0.9/0.999, epsilon 1e-8,
weight decay 0.01, gradient clip 1, accumulation 1, seed 42 and F32 activations
with frozen BF16 base weights. MLX restores and checks every weight/moment array
and the step counter before updating, with pinned layer checkpointing.

Across **552 adapter tensors**, the common-state one-update comparison is:

| Quantity | Zig versus MLX |
| --- | ---: |
| Loss | 0.4549368322 versus 0.4549361765 |
| Absolute loss difference | 6.55651e-7 |
| Update-vector relative-L2 error | **0.0006823248 (0.0682325%)** |
| Update cosine | 0.9999997672 |
| Final weight relative-L2 error | 0.0001011656 |
| Adam first-moment relative-L2 error | 0.0001701098 |
| Adam second-moment relative-L2 error | 6.97587e-5 |

The normal Zig resume finishes in 11.79 seconds with sampled peak RSS
8,709,456 KiB and no swap growth. MLX finishes in 14.24 wrapper seconds with
6,561,776 KiB RSS and 266.50 MiB swap growth. Both start with existing swap and
remain excluded from memory/performance qualification.

The largest coordinate difference is layer 27 `gate_proj` LoRA B at canonical
flat index 192840. Its first moment becomes -7.44368e-9 in Zig versus -4.08135e-8
in MLX, from identical initial state. Applying the AdamW equation to each run's
own retained moments reproduces its final weight within 2.7e-9, versus the
observed cross-framework weight difference 7.27151e-4. This points to sensitivity
to differences in very small backward values. It is an algebraic diagnostic,
not proof of global optimizer parity; raw Zig gradients are not captured by
the normal resume path. The smaller common-state error also supports accumulated
trajectory drift as a contributor to the independent multi-step discrepancies.

Evidence lives under `.benchmark-results/gemma4-qualification-20260909/`:
`mlx-first-predictor-comparison/report.json`, its
`production-sampler-receipt.json`, `mlx-first-common-state/comparison.json`, and
`coordinate-diagnostics.json`. Wrappers, source inventory, isolated hook/binary,
raw tensors, exact initial checkpoint, successful runs and excluded attempts
are retained. `mlx-first-source-audit.json` records final local validation;
`mlx-first-archive-receipt.json` identifies the separate sealed evidence archive.

Remaining MLX work is to trace small backward-value differences at common state,
refresh independent multi-step comparisons after any justified fix, and complete
the full GRPO quality/statistical campaign. E2B's existing small quality failure
is still open; E4B's two-row pass does not establish general acceptance. The
1960/256 multi-seed campaign and reserved 254-example holdout are not consumed by
these diagnostics. No full numerical-parity or production-ready claim is made.

The final Python suite passes **772 tests in 70.19 seconds**, with identical
checkout hashes before and after execution and matching Python source digests.
The unchanged compiled Zig source retains its verified **334 Metal passes / two
skips** and **285 native passes / 26 skips**. The final source audit confirms
no unresolved conflicts or whitespace errors.

### 2026-09-10 MLX-first backward-gradient localization

This follow-up measures raw gradients before clipping and AdamW, using the same
E2B step-two weights/moments/counters and 154-token, 71-supervised-token example
as the common-state comparison above. All changes are confined to isolated
source snapshots and diagnostic scripts. Shipping numerical code is unchanged;
HF/PEFT and CUDA remain deferred.

The first isolated Metal probe captures **552 raw-gradient tensors** immediately
before clipping. Its resumed checkpoint matches all **2763 prior checkpoint
arrays bit-for-bit**, including weights, moments, accumulators and counters.
Raw-gradient relative-L2 error versus the unchanged MLX reference is
**0.0002645960 (0.0264596%)**, cosine **0.9999999677**, and maximum absolute
error **0.0001340210**. The previously measured one-update error remains
**0.0682325%**. The probe completes in 28.20 seconds with sampled peak RSS
9,534,880 KiB and no growth from 2569.94 MiB existing swap.

At layer 27 `gate_proj` LoRA B, canonical index 192840 (output 12052, rank 8),
the directly captured raw gradient is **1.29624505e-6** in Zig versus
**-4.27576566e-7** in MLX. Thus the moment discrepancy observed earlier is
already present before clipping/AdamW. Across **26,333,184 coordinates**, 2355
have opposite nonzero gradient signs. Those coordinates contribute 20.43% of
squared update error. Coordinates whose raw-gradient magnitude is at most
1e-5 in both implementations contribute **78.90%** of squared update error;
the largest 20 individual coordinates contribute 24.57%. These are diagnostic
concentration measurements, not relaxed parity criteria.

A custom identity/VJP tap captures MLX's actual layer-27 LoRA-B reduction
operands while preserving all **3312 retained reference arrays bit-for-bit**.
The selected gradient is the sum of 154 products. Their absolute sum is
0.0014805743; their float64 sum is **-4.27624947e-7**, versus MLX's actual
**-4.27576566e-7**. MLX's own reduction error is **4.83813e-11**, over 35,000
times smaller than the **1.72382e-6** cross-framework gradient difference at
this coordinate. Cancellation is present, but MLX's final summation alone does
not explain the observed gap. This run completes in 14.18 wrapper seconds,
with sampled peak RSS 6,375,984 KiB and no swap growth.

Two bounded controls preserve the reference and quality contracts:

- Disabling the selected BF16 backward matrix implementations produces exactly
  the same 552 raw-gradient arrays and final checkpoint. It establishes no
  improvement and does not prove those switches changed the active route.
- Replacing only MLX's GELU VJP evaluation order with the analytic tanh derivative
  preserves the pinned forward loss. Gradient error changes only from 0.0264596%
  to 0.0264480%; update error slightly worsens to **0.0682691%**. This modified
  derivative is an explanatory ablation, not a new stock-MLX reference or fix.

The attempted decomposed Zig GELU backward path fails closed with
`StrictMetalInterpreterFallback`; no numerical result is admitted from it.
A subsequent optional LoRA-region operand hook produces no operand capture,
although its 26.19-second resumed update preserves the original checkpoint
bit-for-bit. That attempt is excluded from operand analysis. Its source and
execution records are retained. No strict-execution check is bypassed.

The final isolated probe retains the two inputs of the actual layer-27 LoRA-B
gradient dot product as additional graph outputs. It completes in **26.20
seconds**, sampled peak RSS **9,585,232 KiB**, with no growth from 2497.94 MiB
swap. All **552 raw-gradient arrays and 2763 checkpoint arrays remain bitwise
identical** to the baseline capture. The initial build's use of a nonexistent
`Shape.rank` field was corrected to the existing `Shape.rank()` accessor;
the failed source/log and successful retry are retained.

Comparing the two actual operand sets gives a direct decomposition:

| Measured surface | Zig versus unchanged MLX |
| --- | ---: |
| Forward low-rank input relative-L2 error | 3.16683e-6 |
| Incoming gradient relative-L2 error | **0.0002619351 (0.0261935%)** |
| Layer-27 LoRA-B raw-gradient relative-L2 error | 0.0003959590 |
| Zig reduction versus its own float64 product sum, relative-L2 | 2.00123e-7 |
| MLX reduction versus its own float64 product sum, relative-L2 | 1.93672e-7 |

At the worst update coordinate, the observed gradient gap is **1.72382161e-6**.
Float64 cross-substitution attributes **1.72475464e-6** to the differing incoming
gradient, **-8.56542e-10** to the forward operand, and **4.41121e-12** to their
interaction. Zig's and MLX's own reduction errors are **-3.25184e-11** and
**4.83813e-11** respectively. These contributions reconstruct the observed gap
within 1e-15. The numerical discrepancy is therefore already present in the
backward signal entering this LoRA-B reduction; changing the final summation
would not address its principal source.

This closes the missing raw-gradient/operand evidence and localizes the next
investigation to the backward chain feeding that projection. It does **not**
identify a faulty upstream operator or establish a numerical fix. The tested
GELU derivative-order change is not promoted. The next useful capture is the
MLP/normalization backward inputs preceding this signal, followed by independent
multi-step reruns only after a justified implementation change. Full GRPO
acceptance remains separate and open; the reserved holdout is untouched.

Reproduction wrappers and results are under
`.benchmark-results/gemma4-qualification-20260909/`: `run_mlx_backward_probe.py`,
`run_mlx_backward_layer27.py`, `run_mlx_backward_analytic_gelu.py`, and
`run_mlx_backward_graph_operands.py`, with bounded guard wrappers for MLX.
`mlx-backward/graph-operands/operand-decomposition.json` binds the actual tensors,
`mlx-backward/sensitivity.json` records the coordinate analysis, and
`mlx-backward/capture-identity-final.json` verifies bitwise capture fidelity.
`mlx-backward-status.json` and `mlx-backward-source-audit.json` summarize scope
and validation; `mlx-backward-archive-receipt.json` identifies the separate
sealed archive. All model jobs ran serially under the unchanged guards with
pre-existing swap; no memory/performance or full statistical claim is issued.

Shipping Zig and Python sources still match the previously validated binary and
**334 Metal / 285 native / 772 Python** passing-test evidence. Those suites are
not rerun for isolated diagnostic-only source changes. The new validation is
real-model execution, exact checkpoint/gradient fidelity, operand reconstruction,
and a fresh source/conflict/whitespace audit.


### 2026-09-10 MLX-first MLP/normalization and loss-head localization

The next common-state E2B diagnostic follows the backward signal through the
layer-27 MLP and 41 normalization operations (five per layer for layers 27–34,
plus the final norm). The same step-two checkpoint, 154-token training example,
71 supervised tokens, recipe and pinned aligned-F32 MLX reference are retained.
Shipping numerical code is unchanged. This closes a localization question; it
does not reduce the retained 0.0264596% raw-gradient or 0.0682325% one-update gap.

An isolated graph-output probe retains **216 tensors**. The metadata-only run
and tensor-capture run both preserve all **552 raw gradients and 2763 checkpoint
arrays bitwise**. The corrected MLX capture preserves all **3312 original
reference arrays bitwise**. Node IDs are bound to the captured 10,352-node graph
and checked before execution. Complete source, node metadata and build identity
are retained; these hard-coded diagnostic hooks are not shipping code.

Float64 local derivatives use each implementation's actual captured operands:

| Local backward check | Zig relative-L2 error versus own F64 | MLX relative-L2 error versus own F64 |
| --- | ---: | ---: |
| Worst of 41 RMSNorm VJPs | 9.87508e-8 | 7.36117e-8 |
| Layer-27 gated GELU, gate derivative | 1.05803e-7 | 6.54135e-8 |
| Layer-27 gated GELU, up derivative | 8.27932e-8 | 6.81612e-8 |
| Layer-27 down projection, including LoRA | 7.16632e-7 | 7.16029e-7 |

The down-projection F64 control reads the exact BF16 base tensor and common
LoRA weights. All captured normalization weights agree exactly. These local
errors are far smaller than the roughly 2.6e-4 incoming-gradient discrepancy;
no MLP or normalization arithmetic change is supported by these measurements.

The final normalization's incoming gradient already differs by **5.17239e-5**.
An isolated MLX loss-head replay reproduces its captured gradient bitwise. Feeding
the same Zig final-normalized inputs to MLX leaves **5.06671e-5** relative error
against Zig, while changing only the inputs within MLX has **7.86022e-6** effect.
The remaining head discrepancy therefore cannot be attributed solely to forward
activation drift.

A CPU F64 control streams the exact tied BF16 vocabulary weights in 8192-row
tiles, applies the configured softcap, stable cross-entropy and backward
projection over all **262,144 vocabulary entries and 71 supervised rows**:

| Loss-head backward control | Relative-L2 error versus F64 |
| --- | ---: |
| Zig on its own captured inputs | **5.77420e-5** |
| MLX on the same Zig inputs | **8.38806e-6** |
| MLX on its own captured inputs | **8.03859e-6** |
| F64 effect of changing only the captured inputs | 6.61257e-6 |

This makes the **tiled loss-head backward path** the next concrete target:
separate raw-logit projection, softcap/softmax derivatives and BF16
vocabulary-projection accumulation on identical inputs. It does not yet
identify which of those operations needs a fix, or establish that fixing this
head alone will close independent training-trajectory parity.

Instrumentation failures are retained and excluded appropriately. The initial
MLX export failed on BF16-to-NumPy serialization. Casting only exported values
to F32 resolved that issue, but the first broad input taps regrouped the shared
MLP input gradients and changed 2652 reference arrays. Removing gate/up input
taps restored full bitwise fidelity. Only the first broad run's loss-head
control is reused: all five final-norm operand/weight arrays are verified
bitwise identical to the corrected capture, and the MLX head replay is exact.
The first CPU F64 script had a syntax error before execution; its corrected
retry is the admitted result.

The Metal stage run completes in **26.17 seconds**, sampled peak RSS
**8,880,304 KiB**, without growth from 2489.94 MiB existing swap. The corrected
MLX run completes in **14.26 seconds**, sampled peak RSS **7,623,392 KiB**,
without growth from 3338.88 MiB swap. The earlier broad MLX capture/head control
stays within its guard but grows swap by **984.94 MiB**. The CPU F64 control
completes in 1.02 internal seconds / 2.03 wrapper seconds; it finishes between
RSS samples, so its recorded zero sampled RSS is **not memory qualification**.
All runs remain numerical diagnostics, not zero-paging or performance evidence.

`mlx-mlp-source-audit.json` confirms unchanged shipping source/binary and the
previous **334 Metal / 285 native / 772 Python** test evidence; these suites
were not rerun for isolated capture changes. No conflicts or whitespace errors
remain. The report is `mlx-mlp-status.json`; evidence is sealed in
`mlx-mlp-investigation`, bound by `mlx-mlp-archive-receipt.json` to the preceding
`mlx-backward-investigation` archive. Full MLX trajectory parity, broad GRPO
acceptance and the reserved holdout remain open. HF/PEFT stays deferred and
CUDA implementation remains outside this scope.


### 2026-09-10 compensated loss-head backward fix and E4B trajectory blocker

This follow-up **fixes a loss-head backward numerical defect**, but it does not
close full MLX parity. E2B improves; E4B improves at steps one and two but its
independent eight-step comparison worsens. That E4B result remains an explicit
promotion blocker. The current implementation and all results, including that
regression, are retained for review.

The standalone probe calls the original Metal runtime's CCE projection,
statistics, probability-gradient and backward-product implementations using the
actual E2B final-normalized inputs and tied BF16 weights. Its hidden gradient
matches the full training capture **bitwise**. At the default 65,536-entry tile,
relative-L2 error against the full F64 head is 5.77420e-5. Replaying only its
actual probability gradients through an F64 matrix product leaves an own-reduction
error of **5.75268e-5**. Logit-projection and softcap/CE effects on the hidden
gradient are only 2.36480e-6 and 2.59746e-7 respectively. Thus the long backward
matrix sum dominates this measured defect.

An 8192-entry tile control first reduces common-state raw-gradient error from
0.0264596% to 0.00151352% and update error to 0.00301196%. That is a localization
control. The shipping fix retains the existing tile policy and forward math:

- CCE backward tiles wider than 4096 use dedicated precise kernels. SIMD
  products accumulate 256-term blocks and compensate their merge. Small-row
  scalar products compensate each term.
- The kernels reuse the existing threadgroup scratch and output buffers.
  Ordinary linear backward dispatch, frozen BF16 storage, F16/MPS opt-in
  experiments, optimizer settings, manifests and quality thresholds remain
  unchanged. The existing SIMD disable control selects the compensated scalar
  path, including for larger batches.
- Both new pipelines are checked during runtime creation and CCE admission,
  released with the runtime, and identified in existing dense-linear traces.

The final numerical candidate preserves **all raw logits, loss, cached state
and probability gradients bitwise**. Its own backward-reduction error falls to
**3.18223e-7 (about 181 times lower)** and total F64 head error to
**2.43943e-6 (about 24 times lower)**. For comparison, stock aligned-F32 MLX on
the same captured Zig inputs has 8.38806e-6 error against F64. This is a local
numerical accuracy result, not a claim of complete MLX training parity.

The initial 1024-term candidate still left excessive small-row cancellation.
The final kernels pass the exact-zero hidden-gradient fixture with identical
vocabulary rows for batch sizes **1, 3, 17, 64, 65, 71, 128 and 129**, hidden
dimension 5 and vocabulary 65,545. This covers both row paths and the nine-entry
vocabulary tail. Original maximum absolute error reaches **8.08829e-5**; the
final maximum is **7.04866e-7**, within a 1e-6 allowance for F32 exp/log
normalization. The same cases pass with SIMD disabled, and traces confirm the
compensated scalar route. A matching regression is part of the required-device
Metal suite, which passes **335 tests, 2 optional skips** after integration and
again after preserving the SIMD control.

The normal production CLI resumes from the identical MLX E2B step-two weights,
Adam moments and counters. All checkpoint counters/accumulators remain exact,
and the forward loss stays **0.45493683218955994 bitwise**. Across 552 tensors,
one-update relative-L2 error falls from **0.0682325% to 0.00969259%**, about
**7.04 times lower**. This run captures a normal checkpoint, not raw gradients.

Independent all-linear trajectories retain the original initial adapters,
example, optimizer and pinned aligned-F32 MLX references:

| Model / steps | Raw-gradient error before → after | Update-vector error before → after |
| --- | ---: | ---: |
| E2B / 8 | 3.21921% → **1.50205%** | 0.999742% → **0.340642%** |
| E4B / 1 | 0.00263524% → **0.00175556%** | 0.131526% → **0.0744023%** |
| E4B / 2 | 0.0227013% → **0.0106306%** | 0.312394% → **0.198880%** |
| E4B / 8 | 7.96131% → **23.9561%** | 0.776499% → **1.13938%** |

The E4B eight-step worsening is not dismissed or converted into a PASS. Its
first-step gradient comparison uses the same initial state and improves, as
does step two, while later independently updated trajectories diverge further.
At E4B step one, coordinates whose raw gradients are at most 1e-5 in both
implementations contribute **99.9931%** of squared update error. At step two,
that fraction is **95.4542%**; **683 opposite-sign coordinates** contribute
**82.2140%**. These measurements identify tiny-gradient update sensitivity as
the next investigation target; they do not establish an AdamW arithmetic bug
or justify changing epsilon, clipping, learning rate or tolerances.

All current model runs complete within the existing 180-second / 18-GiB RSS /
4-GiB swap-growth / 2-GiB disk guards. Exact resource samples are recorded in
`loss-head-status.json` and the individual executions. Existing swap remains,
so these are numerical diagnostics, not zero-paging performance qualification.
The full 1960/256 multi-seed GRPO campaign and reserved holdout remain unrun;
previous small GRPO quality results belong to the preceding binary and have
not been refreshed for this loss-head change.

`loss-head-source-audit.json` binds the final compiled patch and binary to the
335-test Metal pass. Python source hashes still match the retained **772-test**
pass; that suite was not rerun. Native math is unchanged. Conflicts and both
staged/unstaged whitespace checks are clean. Wrapper-description corrections
are recorded separately in `loss-head-scope-clarifications.json`, preserving
executed scripts and their hashes. Full evidence and the current PR description
are sealed in `loss-head-backward-investigation`, with receipt
`loss-head-archive-receipt.json` binding the preceding `mlx-mlp-investigation`
archive. HF/PEFT remains deferred; CUDA implementation remains outside scope.


### 2026-09-10 tiny-gradient sensitivity and long forward projection fix

This follow-up isolates the E4B sensitivity and improves its independent
trajectory, but **does not close the full-parity blocker**. The integrated
change compensates long BF16 forward projections on the existing 32-row SIMD
route. The independently tested candidate improves E2B as well as E4B; the
older E4B result before the loss-head change is still better at eight steps.

The diagnostics retain the original example (154 physical / 71 supervised
tokens), initial rank-16/alpha-32 all-linear adapters, seed 42, BF16 frozen
weights, aligned-F32 pinned MLX and AdamW settings. No epsilon, learning rate,
clipping policy, parity tolerance or quality threshold changes. These are
one-example SFT numerical controls, **not GRPO acceptance**.

**Optimizer sensitivity is measured, rather than inferred from small gradients.**
An F64 first-step AdamW reconstruction covers all 38,879,232 E4B coordinates.
Using the actual differing gradients explains essentially all update-error
energy; the residual is 0.0080604%. A separate actual-GPU replay covers 174,978
selected coordinates. Each implementation reproduces its own captured weights
bitwise. Given identical raw gradients and clipping scale, production Metal
and pinned MLX differ by at most 7.10133e-9 in the resulting weights. This does
not support changing the optimizer to address the observed trajectory gap.

A causal control replaces MLX's first-step weights and Adam moments with the
exact shipping Metal state, then runs steps two through eight entirely in MLX.
That intervention alone produces 1.29441% update error versus uninterrupted
MLX. Shipping Metal versus the intervened continuation is only 0.180598%.
This identifies early state sensitivity as a major cause of later divergence;
the intervened run is explicitly **not a replacement golden reference**.

The instrumented Metal capture is nonperturbing: all 686 raw-gradient arrays
and 3432 checkpoint arrays match shipping bitwise. Across five worst LoRA-B
targets, local final-dot errors versus F64 are about 2e-7, while incoming branch
cotangents differ by about 1.7e-5 to 2e-5. Replacing both implementations' local
dots with F64 preserves all five worst sign mismatches. The remaining signal
therefore arrives from upstream. E4B's loss-head replay also matches the full
capture bitwise; its own backward reduction error is 2.06943e-7. Its measured
projection and input effects are larger.

**The selected fix reduces forward accumulation error.** Actual E4B layer-19
MLP down-projection operands have a 10,240-term reduction. Relative-L2 error
against F64 is 1.61594e-6 for the previous Metal kernel, 1.07763e-6 for stock
MLX on those same operands, and 2.06726e-7 for the new kernel: a 7.82-fold Metal
accuracy improvement. The precise kernel forms 256-term SIMD partial sums and
compensates their merge, reusing the existing 16-KiB staging allocation. Both
ordinary forward entry points select it when their 32-row BF16 SIMD route is
active and the input width exceeds 4096. Existing m64 dispatch and SIMD disable
controls remain intact. Runtime admission, diagnostics and teardown include
the new pipeline. The cancellation regression covers rows 128, 129 and 154,
input width 8193 and output width 129, including row, input and output tails.
Its initial 65-column fixture correctly failed because that width selects the
unchanged scalar route; the corrected fixture targets the actual SIMD contract.

A separate compensated ordinary-backward candidate improves its isolated F64
product by 6.44 times but worsens E4B's eight-step update error to 1.14838%.
It is **rejected and excluded from shipping source**; its negative evidence is
retained. Local accuracy alone is insufficient for selecting a training fix.

Comparison results, relative-L2 percentages (independent eight-step rows
refreshed on the final binary; first-step/common-state rows use the validated
candidate):

| Model / scope | Raw-gradient error before → after | Update-vector error before → after |
| --- | ---: | ---: |
| E4B / step 1 | 0.00175556% → 0.00195003% | 0.0744023% → 0.0720630% |
| E4B / independent step 8 | 23.9561% → 9.45365% | 1.13938% → 0.967597% |
| E2B / independent step 8 | 1.50205% → 0.458272% | 0.340642% → 0.153303% |
| E4B / identical MLX step-7 state, one CLI update | not captured | 0.00189961% → 0.00150343% |

The common-state run uses normal admitted checkpoint resume, including all
weights, moments, counters and zero accumulators. It captures a checkpoint,
not raw gradients. Its maximum update discrepancy is 8.47154e-7. The earlier
pre-loss-head independent E4B errors were 7.96131% / 0.776499%; the new forward
fix **has not fully recovered that older result**. First-step loss also changes
with the forward fix; this is not a claim of bitwise-preserved forward math.

**E4B MLX reference refreshes now finish within the memory guards.** Three
initial attempts hit the swap-growth guard and remain recorded as excluded.
The successful fixed-token diagnostic caches the exact frozen token/PLE
embedding gathers and releases the unused 5.25-GiB PLE table. The trainable
per-layer projection remains in the normal forward graph. Both refreshed
one-step and eight-step runs reproduce all 4116 saved stock-reference arrays
bitwise, finish in 26.3 / 40.4 seconds, peak at about 5.7 / 6.1 GiB RSS, and
show no sampled swap growth. This optimization applies only to these fixed
inputs; it is not general MLX backend or performance qualification.

The final ReleaseFast binary (`ad2b31256bf73c9f03e78d1abcedc0e0030e3539d9cf47bf2043bbd2448b2cdf`)
passes **336 required-device Metal tests, 2 optional skips**. Fresh independent
E2B and E4B eight-step runs reproduce the candidate's complete loss history
and all gradient/update/Adam summary metrics exactly. Their full trace files
also match the candidate byte-for-byte (`tiny-final-trace-identity.json`).
E2B completes in 22.24
seconds at 6.87 GiB sampled peak RSS; E4B completes in 52.55 seconds at
14.28 GiB. Both pass capture validation and packaging with no sampled swap
growth. The tested source patch is
`f9aaba7d198715e19b9283d57f66cef0cfd781b9a2c3f82d009f48c56d4c5226`.
Python source hashes still match the retained 772-test pass; that suite was
not rerun. Native math is unchanged and its earlier results remain historical.
Unresolved-conflict and staged/unstaged whitespace audits are clean.

Final integrated-build verification and immutable evidence bindings are
recorded in `tiny-gradient-status.json`, `tiny-gradient-source-audit.json` and
`tiny-gradient-archive-receipt.json`. The archive `tiny-gradient-investigation`
retains successful and excluded attempts, exact operands, F64 controls,
optimizer replay, causal intervention, source, binaries and wrapper scope
corrections, and binds the preceding `loss-head-backward-investigation` archive.

Existing swap remains, so all numerical runs are excluded from zero-paging
performance qualification. Full MLX trajectory parity, full 1960/256 multi-seed
GRPO, reserved-holdout acceptance, recovery/native E4B coverage and hosted CI
remain open. Earlier small GRPO results belong to earlier binaries and have
not been refreshed here. HF/PEFT remains deferred; CUDA is outside scope.


### 2026-09-10 earliest E4B divergence: initial RMSNorm rounding

The next investigation **locates the first remaining differences from identical
inputs and weights**. The attention input and per-layer-input (PLE) branches
both first show nonzero forward-output differences at RMSNorm. These are
small F32 rounding differences; this evidence does not establish a remaining
normalization formula bug or explain the entire independent trajectory gap.
Shipping mathematical source, optimizer settings and tolerances are unchanged.

The same fixed example, initial adapters and pinned aligned-F32 MLX are used.
Both isolated Metal captures retain all **686 raw-gradient arrays and 3432
checkpoint arrays** exactly; those files also match the shipping-equivalent
forward candidate byte-for-byte. The successful MLX capture compares all
**4116 actual arrays** bitwise with the unchanged stock reference and saves
individual array hashes without duplicating the complete reference file.
The isolated source snapshots differ from shipping only in graph output
retention and diagnostic saving. This rules out instrumentation drift before
interpreting the saved operands.

The first attention RMSNorm and the PLE projection are parallel consumers of
scaled token embeddings; there is no unique serial order between these branches.
Measured initial stages are:

| Stage | Metal versus MLX |
| --- | ---: |
| Token embedding lookup | bitwise identical |
| Token embedding scaling | bitwise identical |
| Scaled frozen PLE token embeddings | bitwise identical |
| Layer-0 input RMSNorm | 2.17332e-8 relative L2; 19209 / 394240 coordinates differ |
| PLE base projection and complete projection output | bitwise identical |
| Scaled PLE projection / RMSNorm input | bitwise identical |
| PLE projection RMSNorm | 4.15873e-8 relative L2 |

The PLE low-rank intermediate differs by 1.81142e-7 relative L2. Its initial
LoRA-B factor is zero, so the branch contribution and complete forward
projection still match exactly. That intermediate remains relevant to the
LoRA-B gradient; matching forward outputs alone does not prove backward parity.

**Exact-input F64 controls classify the measured errors.** Standalone production
Metal RMSNorm reproduces the full capture bitwise; stock MLX fast RMSNorm on
the same input and frozen weight reproduces its full capture bitwise. Their
errors against F64 are **5.92378e-8 / 5.85825e-8** respectively. All observed MLX
outputs can be reconstructed with the same two F32 multiplications, using the
same per-row inverse as Metal for 144 rows, one representable F32 step lower
for eight rows and one step higher for two rows. This is an inference from
observed outputs, not direct access to MLX's internal reduction statistics.

The parallel PLE base projection has **7.68566e-7** error against F64 in both
implementations, with bitwise-identical outputs. It is not a source of their
initial forward disagreement. PLE RMSNorm errors against each implementation's
own F64 normalization are **5.25547e-8 / 5.40489e-8**; its input is identical.
These results are consistent with F32 rounding at normalization, rather than
a missing weight, scaling factor, or different initial embedding.

**A concrete compiler precision distinction is verified.** Production sets
`MTLCompileOptions.mathMode = MTLMathModeSafe`, but leaves
`mathFloatingPointFunctions = MTLMathFloatingPointFunctionsFast`. The observed
option values are 0/0, consistent with the retained SDK header. A statistics
probe using these exact options reproduces production bitwise. Changing only
the function option to `Precise` (0/1) preserves every sum, mean and denominator
but changes reciprocal square roots in three rows, affecting 5867 output
coordinates. Replaying those changed inverses reproduces the complete output
difference. Safe arithmetic and precise math functions are separate settings.
The explicit-precision control is **not integrated or qualified as a training
fix**. Earlier statistics probes using `fastMathEnabled = NO` did not reproduce
production; their outputs and failed fidelity assertion are retained and are
not interpreted as production internal statistics.

**Causal controls show why the first difference is not the whole explanation.**
One fixed-input MLX step is repeated with Metal's exact initial RMSNorm output,
then with both initial RMSNorm outputs matched. The two-normalization control
uses stopped-gradient output corrections, verifies inputs/outputs bitwise,
and preserves stock MLX backward through the trainable PLE projection. These
are labeled interventions, not replacement references:

| MLX comparison with unchanged Metal step | Gradient relative L2 | Update relative L2 |
| --- | ---: | ---: |
| Stock MLX | 0.00195003% | 0.0720630% |
| First input RMSNorm forward output matched | 0.00213512% | 0.0770754% |
| Both initial RMSNorm forward outputs matched | 0.00204047% | 0.0742755% |

The interventions themselves shift MLX's update by **0.0555848% / 0.0615617%**
relative to stock MLX. Small rounding differences demonstrably affect updates,
but matching these two early outputs does not close the overall gradient or
update gap. A useful next control is matching subsequent layer-boundary state
and backward inputs, then separating RMSNorm backward and attention/MLP
arithmetic. Choosing a global precision setting solely to match this initial
fixture would be premature.

The original MLX attempt stops at the existing **2-GiB free-disk guard** and
remains excluded even though its completed tensor bytes match the reference.
Byte-verified independent APFS clones reclaim duplicate physical storage
without removing logical artifacts. The successful MLX retry completes in
**28.27 seconds**, at **5.93 GiB sampled peak RSS**, with **504.50 MiB swap
growth**, within its unchanged guard. The two Metal captures complete in
**46.55 / 46.49 seconds**; the extended capture peaks at **14.58 GiB RSS** with
no sampled swap growth. The one-/two-normalization interventions complete in
**28.27 / 30.33 seconds** with no sampled swap growth. Existing swap excludes
all of these runs from zero-paging or performance qualification. A recorded
zero sampled RSS for a short standalone probe means it finished between samples.

Source audit retains the prior **336 Metal / 772 Python** passing evidence;
these suites were not rerun because shipping code is unchanged. No unresolved
conflicts or whitespace errors remain. Details, exact source/binary hashes,
per-array digests, actual operands, F64 controls, excluded attempts and scope
corrections are bound by `early-divergence-status.json` and sealed in
`early-divergence-investigation`. Receipt `early-divergence-archive-receipt.json`
binds the preceding `tiny-gradient-investigation` archive. Full independent
MLX parity and GRPO acceptance remain open; the reserved holdout is untouched.
HF/PEFT remains deferred and CUDA implementation stays outside scope.


### 2026-09-10 E4B RMSNorm backward: nine matched-input controls

**RMSNorm backward is not the dominant source of the measured step-one drift
at these nine sites.** The isolated capture and pinned aligned-F32 MLX use the
same original example, weights, adapters and AdamW settings as the preceding
investigation. Shipping mathematical source and binary are unchanged. This
is a one-example SFT arithmetic diagnostic, not a new GRPO acceptance result.

The capture's full raw-gradient, trainer-checkpoint and adapter files match
the shipping-equivalent baseline byte-for-byte (686 gradient / 3432 checkpoint
arrays). The MLX hooks preserve all **4116 reference arrays bitwise**. Hooks
retain each norm's forward input/output and backward input/cotangent/output;
MLX uses an identity output VJP to retain the incoming cotangent. Every frozen
norm weight is checked against its exact BF16 model tensor. Standalone
production Metal backward reproduces all nine captured outputs **bitwise**.
MLX backward is evaluated using its stock fast-RMSNorm VJP on those same
operands, with both backends' input sets tested independently.

Relative L2 below is a fraction, not a percentage. The full-path column uses
each backend's own captured inputs; the matched-input column uses the same
Metal-captured input, weight and cotangent for both implementations.

| Site | Full-path backward difference | Matched-input Metal–MLX | Metal vs own F64 | MLX vs same F64 |
| --- | ---: | ---: | ---: | ---: |
| Final norm | 6.73132e-06 | 6.79590e-08 | 5.18775e-08 | 5.63595e-08 |
| Shared PLE projection norm | 2.18423e-05 | 6.46254e-08 | 4.74432e-08 | 5.02334e-08 |
| Layer 0 post-attention | 1.17689e-05 | 6.39263e-08 | 4.92327e-08 | 5.03963e-08 |
| Layer 0 pre-FFN | 1.20577e-05 | 6.59490e-08 | 5.50858e-08 | 5.83087e-08 |
| Layer 0 post-FFN | 1.23914e-05 | 7.09921e-08 | 5.20314e-08 | 4.59423e-08 |
| Layer 0 PLE post-norm | 1.29828e-05 | 6.18428e-08 | 4.71538e-08 | 5.15217e-08 |
| Layer 0 query norm | 1.28622e-05 | 5.47914e-08 | 4.99253e-08 | 4.98927e-08 |
| Layer 0 key norm | 1.10933e-05 | 7.08921e-08 | 6.59532e-08 | 7.31955e-08 |
| Layer 19 post-FFN | 2.02578e-05 | 7.88939e-08 | 5.00042e-08 | 6.68040e-08 |

Replacing both local backward calculations with F64, while preserving their
actual different inputs/cotangents, retains **99.765–100.152% of the original
gap's L2 magnitude** across the sites. The difference between the observed
and F64 gap vectors is **0.272–1.077% of the observed gap norm**. These are
local counterfactuals, not whole-model F64 training or a trajectory intervention.
The shared PLE norm has bitwise-identical forward inputs, but its incoming
cotangent already differs by **2.329e-5** relative L2. Its backward operation
therefore propagates disagreement received from the rest of the graph.

Production backward uses the ordinary Metal shader library. Replaying its
same kernels with `mathMode=Safe`, and then additionally precise math
functions, does **not** consistently improve accuracy against F64. These
controls are retained; neither setting is integrated as a fix. The next
bounded target is the layer-19 MLP activation/product chain and its backward
operands, adjacent to the previously sensitive down-projection gradient.
That should separate nonlinear amplification of forward differences from
new local backward error. The current evidence does not establish an
attention/MLP bug or rule out differences at unmeasured sites/steps.

The Metal capture completes in **46.49 s**, sampled peak RSS **14.36 GiB**,
with no sampled swap growth; pinned MLX completes in **30.35 s**, peak RSS
**6.75 GiB**, with **260.37 MiB swap growth**. Both pass their original model
execution guards. Standalone Metal/MLX controls complete in **6.06 / 2.04 s**
without sampled swap growth. Existing swap still excludes performance and
zero-paging qualification. Zero sampled RSS for the short MLX replay means
it finished between samples. Initial Metal and standalone preflight attempts
stop before process launch because free disk is below 3 GiB. Storage receipts
retain byte-verified APFS clones; newly generated F64 scratch is losslessly
compressed with round-trip SHA verification. The small standalone MLX replay
uses an explicit output-sized preflight reserve: the unchanged 2-GiB stop
floor plus 35,323,904 output bytes and 128 MiB margin. Model preflights, runtime
stop limits, numerical tolerances and quality gates are unchanged.

Reproduction is recorded by `build_norm_backward_capture.py`,
`run_norm_backward_metal.py`, `guard_norm_backward_mlx.py`,
`prepare_norm_backward_control.py`, `compact_norm_backward_truth.py`,
`guard_norm_backward_control.py`, `guard_norm_backward_control_mlx.py` and
`analyze_norm_backward.py`. `norm-backward-status.json` binds source/binary
hashes, capture fidelity, per-array reference hashes, arithmetic comparisons,
execution guards and scope clarifications. The sealed
`norm-backward-investigation` archive binds the preceding
`early-divergence-investigation` manifest.

Read-only source/conflict/whitespace checks retain the prior **336 Metal /
772 Python** passing evidence; suites are not rerun for this documentation
and isolated diagnostic work. Full independent MLX trajectory parity, full
1960/256 multi-seed GRPO, reserved-holdout acceptance, recovery/native E4B
coverage, zero-paging performance and hosted CI remain open. The reserved
holdout is untouched; HF/PEFT remains deferred and CUDA stays outside scope.


### 2026-09-10 layer-19 GEGLU: inherited gradient drift dominates

**The measured layer-19 GELU/product forward and backward arithmetic is
healthy against matched-input MLX and F64.** This closes another local
arithmetic evidence gap without establishing full independent trajectory
parity. Shipping code, optimizer settings, tolerances and quality gates are
unchanged. This remains one fixed-example, initial-state E4B SFT arithmetic
work; it is not a GRPO or held-out acceptance campaign.

An isolated source snapshot retains the layer-19 MLP input, gate/up projection
outputs, GELU/product, down-projection boundary and backward operands. The
first default-fusion capture stops with `MissingValue` and yields no valid
training evidence. The diagnostic retry disables the gated-GELU forward and
backward fusions, FFN-GELU backward runtime region, and gated-FFN graph fusion.
It passes capture validation and its complete **686 raw-gradient arrays,
3432 checkpoint arrays, and adapter file match the production-equivalent
baseline byte-for-byte**. Thus the rollback permits inspection without
changing the observed step. Default-fusion capture of these added intermediate
outputs remains a diagnostic limitation; no shipping fusion change is made.

Pinned MLX preserves its original compiled `geglu` function. Identity output
VJP taps capture its product cotangent and gate/up gradients; all **4116
reference arrays remain bitwise identical**. Every inspected native output
replays bitwise with production Metal APIs. MLX compiled product and both
backward outputs replay bitwise from its actual captured operands. Fused and
unfused standalone Metal backward outputs also match bitwise on both input
sets. The separately replayed MLX GELU value is not claimed to be an observed
internal intermediate of compiled GEGLU.

All relative L2 values below are fractions, not percentages. Matched-input
controls use the same native-captured gate, up projection and product
cotangent for both implementations; the report also checks the MLX operand
set independently. F64 uses the tanh GELU approximation and analytic
derivative with the captured F32 operands converted exactly to float64.

| Stage | Full-path difference | Matched-input Metal–MLX | Metal vs F64 | MLX vs F64 |
| --- | ---: | ---: | ---: | ---: |
| GELU forward (separate MLX replay) | not captured inside MLX fusion | 6.54341e-08 | 6.04774e-08 | 4.90207e-08 |
| Gated product | 3.58280e-06 | 6.16974e-08 | 6.33876e-08 | 5.41920e-08 |
| Gate gradient | 2.18682e-05 | 8.00218e-08 | 6.47283e-08 | 6.44983e-08 |
| Up gradient | 2.26583e-05 | 6.45052e-08 | 6.41552e-08 | 5.42676e-08 |

The MLP input already differs by **3.73456e-6** relative L2; the gate/up
projections differ by **2.87414e-6 / 3.47006e-6**. The gradient arriving at the
down-projection output differs by **2.02578e-5**, and the gradient arriving at
the gated product differs by **1.99507e-5**. Replacing just local GEGLU
arithmetic with F64 retains **99.9239%** of the product gap's L2 magnitude and
**100.0153% / 100.0375%** of the gate/up gradient gap magnitudes. The residual
between observed and F64 gap vectors is **2.2057% / 0.4056% / 0.3667%** of
those respective observed gap norms. These are local counterfactuals with
different captured operands retained, not whole-model F64 trajectories.

Holding the other inputs at the MLX values gives this local F64 attribution:

| Changed operands | Gate-gradient difference | Up-gradient difference |
| --- | ---: | ---: |
| Forward gate/up values only | 5.35775e-6 | 4.95488e-6 |
| Incoming product cotangent only | 2.14048e-5 | 2.19122e-5 |
| Both | 2.18715e-5 | 2.26668e-5 |

The differences are vectors, so these norms are not additive percentages of
causation. They show inherited cotangent drift dominates this local check;
forward-state differences add smaller nonlinear effects. Matched-input
Metal/MLX and F64 comparisons show no sign reversals at the inspected GEGLU
outputs/gradient coordinates. This does not rule out tiny cancellation-driven
LoRA-gradient sign changes in later matrix reductions.

An explicit precise-math-function control reduces native GELU forward error
against F64 from **6.04774e-8 to 4.90089e-8**, and gate-gradient error from
**6.47283e-8 to 6.16320e-8** on the native operand set. This is a small local
rounding improvement; it has not been qualified as a whole-trajectory fix and
is not integrated. The next bounded experiment is a complete layer-19 MLP
replay with identical boundary input and incoming gradient, including its
projection products, to isolate the block's contribution from earlier state
drift. Existing evidence does not identify an attention or MLP formula bug.

The successful Metal capture completes in **40.44 s**, sampled peak RSS
**14.74 GiB**, with no sampled swap growth. MLX completes in **30.34 s**, peak
RSS **7.56 GiB**, with **396.56 MiB swap growth**. Standalone Metal/MLX replays
complete in **6.06 / 2.03 s** with no sampled swap growth. All pass the original
3-GiB disk preflight and runtime guards; existing swap excludes zero-paging
and performance qualification. A zero sampled RSS for the short MLX replay
means it completed between samples.

To restore disk headroom before the build, idle compiler objects receive
transparent lossless APFS compression. Every original path and logical byte
is retained and SHA-verified before/after replacement; no cache entry or
sealed evidence is deleted. A sandbox compression attempt creates an invalid
zero-length staging file and fails its hash check before replacing the
original; the host retry succeeds. These storage receipts and the excluded
capture attempt remain explicit in the evidence.

Reproduction: `build_mlp19_capture.py`, `run_mlp19_metal_unfused.py`,
`guard_mlp19_mlx.py`, `prepare_mlp19_control.py`, `guard_mlp19_control.py`,
`guard_mlp19_control_mlx.py`, and `analyze_mlp19.py`. The status, operands,
compiler controls, execution guards, exact source/binary hashes and per-array
MLX digests are sealed in `mlp19-investigation`, chained to
`norm-backward-investigation` manifest
`f4d3aa58d7abf091fcc86539c9581f17637223d6b7930a7356b19212a3e6f9d1`.

Source/conflict/whitespace audit retains the prior **336 Metal / 772 Python**
passing evidence; those suites are not rerun because shipping math is
unchanged. Diagnostic scripts compile. Full independent MLX trajectory
parity, full 1960/256 multi-seed GRPO and reserved-holdout acceptance,
recovery/native E4B coverage, zero-paging performance and hosted CI remain
open. The reserved holdout is untouched. HF/PEFT remains deferred and CUDA
implementation remains outside scope.

The next producer target is the layer-19 backward residual add (`node 9403`).
Its two inputs are the MLP-output gradient (`node 9374`) and the residual-path
gradient (`node 9402`). Existing whole-block evidence shows the add output
drift is inherited from those branches: the native-boundary cotangent differs
by **2.02578e-5 relative L2**, while the matched-boundary block remains
bitwise consistent. The dedicated Metal capture completed in 44.47 s with
`target_tensor_count=686`; the retained tensors satisfy `node9403 ==
node9374 + node9402` exactly (`max_abs=0`, `relative_l2=0`). This validates the
add arithmetic and shows no new producer defect. A fresh guarded end-to-end
MLX capture now confirms relative-L2 drift of **1.72255e-5** at node 9374,
**1.88070e-5** at node 9402, and **1.72201e-5** at their node-9403 sum. MLX
also reconstructs the add bitwise. Node 9374 dominates the absolute error, so
its scalar-multiply input at node 9373 was traced next. Fresh MLX/Metal results
show **1.72245e-5** relative-L2 drift at node 9373 and **1.72255e-5** after the
scalar multiply at node 9374, clearing that multiply. Node 9373 is an exact add
on both backends. Its layer-20 residual branch at node 9278 has **1.71519e-5**
drift and dominates absolute error; its attention branch at node 9372 has
**2.04834e-5** drift but a 3.37x smaller L2 norm. Node 9278 is an exact add on
both backends. Its feed-forward residual input at node 9237 differs by
**1.62185e-5** relative L2, while the pre-feed-forward branch at node 9277
differs by **1.96164e-5**. Splitting node 9237 again shows node 9208 carries
effectively all absolute disagreement (`difference_l2=1.23803e-5`); the
per-layer gate branch at node 9236 contributes only `1.23512e-7`. Node 9208 is
a scalar multiply of node 9207. This continues the upstream drift without
identifying an incorrect add, scalar multiply, RMSNorm, or MLP implementation.

A fresh final-boundary capture localizes the first measured backward difference
to the fused loss-head cotangent at node 6350: **6.29780e-6 relative L2** with
`max_abs=1.21363e-8`. Its scatter and reshape through node 6352 are bitwise
exact, including zero nonsupervised rows. Final RMSNorm backward reaches
**6.73134e-6** at node 6354. Existing matched-input controls bound the local
compensated loss-head error at **2.43943e-6** and final-norm backward near
`7e-8`; most of the fresh full-path disagreement therefore arrives in differing
forward hidden states. No backward operation measured in this trace provides a
justified shipping fix.

The follow-up whole-block replay uses the exact layer-19 boundary input and
incoming gradient from each backend. On those backend-native boundaries, the
block's input-gradient difference is **2.20977e-5 relative L2**. Replaying the
complete gate projection, up projection, GEGLU product, down projection and
input VJP with the same boundary tensors makes the Metal and MLX block outputs
bitwise consistent with their respective reference replays; the Metal replay
also matches the captured native input gradient bitwise. This isolates the
remaining difference to the boundary state and cotangent arriving from earlier
layers, rather than the layer-19 block's local arithmetic. The block controls
complete in **4.09 s / 2.03 s** (Metal / MLX), with no sampled swap growth; they
are standalone diagnostics and do not qualify full-model parity or performance.
