# EmbeddingGemma 2 production review

## Recommendation

Open this as a **draft PR**. The implementation is complete enough for review and its
correctness, API, admission, cancellation, and model-distribution contracts are well
covered. Promotion is blocked by the final 44.1 kHz audio accuracy failure and image
throughput failure described below.

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

## Reproducible model and build

The catalog pins `google/embeddinggemma-2` at Hugging Face commit
`914f7f89142e33e77833254d9c9b90c3cef7303b`. The 1,488,915,288-byte weight file has
SHA-256 `197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79`;
all 13 bundle artifacts are verified by size and digest and installed with a managed v2
receipt.

```sh
zig/pkg/inference/zig-out/bin/antfly-inference pull \
  google/embeddinggemma-2:bf16-safetensors-bundle-v1 --models-dir .models
```

The final reviewed source manifest is
`results/l4-2026-10-08-source-manifest.json`, SHA-256
`e0975c97287f640dc874a805cf4e586931e41bf8ae78d67435f0dab0d2c2f51e`.
The stripped ReleaseFast CUDA/SM89 binary is 46,233,640 bytes, timestamped
2026-10-08 02:55:33.823 UTC, with SHA-256
`bda59bbadd9fdc2d1af83abb2312d751eb8df780a54140bc75b676d76f1f73b9`.
Portable CUDA artifacts are shipped, but the architecture-specific WMMA path has been
qualified only on SM89/L4.

## Correctness and contract evidence

The final binary passes all 60 CLI oracle gates across document, query, code-query,
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

## Remaining qualification gap

The final nine-case audio matrix **fails the production quality gate**. The 1-second,
44.1 kHz case reaches cosine `0.998923725795`, below the required `0.999`; see
`results/l4-2026-10-08-raw/audio-matrix-final-bda59bb.json`. This is materially below
the final 16 kHz fixture and the other modality cases. A diagnostic run forcing all
audio operations onto the host reaches `0.999106404`, but that is not the production
path and is not counted as a pass. The antialiased Zig resampler has independent
SciPy/PyTorch parity tests, so the remaining full-path discrepancy must be understood
and corrected before production promotion.

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
preceding `4aa4dda…` binary. The only final runtime change is the bitwise-identical sparse
mel summation helper, so this is retained as precise regression evidence rather than
represented as a fresh `bda59bb…` run.

## SDK and generated-source validation

Generated OpenAPI clients are synchronized. The generator also carries pre-existing
ChatGPT and Apple schema drift that the repository checker requires; those unrelated
lines are retained so regeneration is stable rather than hand-edited away.

- Go: all packages pass.
- Python: 266 tests pass, including the generator check.
- TypeScript: typecheck passes.
- Rust: specification synchronization passes; a Rust compiler was not available in the
  qualification environment.

Reviewers can reproduce local and GPU checks from `README.md`. Keep the PR in draft and
resolve both failed gates before production promotion.
