# Media and OCR optimization, October 2026

Current contracts: [batching](../../../../../zig/pkg/inference/BATCHING.md#gpu-residency-for-media-embeddings)
and [generative OCR](../../gemma4-ocr-optimization.md). The single
[results JSON](../../media-throughput-results.json) retains the final timing
samples, source/binary/harness pins, corpus hashes, exact OCR flags and gates.

Detailed reports and former guides are preserved in the [pinned evidence snapshot](https://github.com/antflydb/antfly/tree/01cb3636fe502a48a83f9881bd8df9d925d36a70/docs/design/inference).
[Download the repository snapshot ZIP](https://github.com/antflydb/antfly/archive/01cb3636fe502a48a83f9881bd8df9d925d36a70.zip)
and inspect `docs/design/inference/` for the detailed reports. A separate
219,486-byte evidence-only ZIP was generated locally as
`.benchmark-results/pr-1088-evidence-01cb3636fe.zip`; its checksum is in the results
JSON. It contains published reports/guides and the unabridged diaries, not model
assets or ignored raw transcripts/logs. Original full diary:
[OCR at 144f7dc](https://github.com/antflydb/antfly/blob/144f7dc49c17bb4b9b747b58d8ae2a81b918f3e6/docs/design/inference/gemma4-ocr-optimization.md).

## Final results

M4 Max (32 GPU cores, 36 GiB). These are distinct local campaigns and timing
scopes, not a deployment SLA or cross-runtime ranking.

| Workload | Before → after | Qualified conclusion |
| --- | --- | --- |
| Eight square-image embeddings | 11.827 → 2.377 s p50 | Flash + frames + resident handoff; final cosine >= 0.999999987 |
| Eight photo-image embeddings | 12.128 → 2.388 s p50 | Final cosine >= 0.999999962; borrowed raster scope |
| Gemma4 eight short audio clips | 1.268 → 1.168 s p50 | Downstream embedding encoder batches; conformer stays per clip |
| CLAP sixteen short audio clips | 1.018 → 0.900 s p50 | Native batches plus bounded CPU preparation overlap |
| Eight-page Gemma4 OCR | Ordinary brackets 55.339–57.572 s → 48.819–49.304 s | 12.8% less mean recognition time; all 4,057 tokens/transcripts exact |
| Earlier Apple accurate OCR | 2.087 s for the same eight pages | Warm recognition baseline excludes PNG preparation/HTTP |

## Image and audio embeddings

Retain numeric intermediates and final prompt/encoder handoffs on the admitted
backend; only final vectors download in qualified paths. CPU codecs, exact resize,
audio resampling/frame packing and unsupported provider operations remain host
boundaries. Synthetic native/Metal tests establish parity/ownership/transfer
behavior; CUDA cross-compilation does not qualify CUDA throughput.

Initial same-grid image batching gained ~11% at four rows; eight rows were slower
and doubled activation storage. Serving stays at four. Model-scoped vision Flash
and per-layer command frames then yielded the larger embedding gains above.
Geometry-qualified brokers preserve model generation, grid axes, task/dimensions,
source/options and original indexes. Bounded lookahead joins on every exit and
acquires no speculative GPU lease. Lookahead's independent gain and saturated
HTTP throughput remain unqualified because of drift/low coalescing occupancy.

Audio-only Gemma encoder groups use at most four rows, <=25% padding and a
1,024-token cap. CLAP retains native batches up to eight and fusion/fallback.
Mixed lengths, stereo, PCM, MP3/AAC/Ogg and indexed failures preserve reference
vectors. Concurrent compilation limits the short-clip timing claims; these
are not audio text-generation or GPU codec-decoding measurements.

## PDF rendering and codecs

Production PDF rendering stays CPU. Resident model handoff cut four-page tensor
downloads by ~99.8%, but full PDF embedding comparisons varied/reversed by process
order. **No overall PDF throughput improvement is qualified.**

The benchmark-only Metal painter supports simple opaque fills/images with exact
CPU fallback for unsupported pages. Snapping fractional raster extent fixed an
initial scan pixel failure. Tile culling, preserved paint order, bounded reusable
buffers and overlapped private-reader preparation improved mixed render-only
~62→38 ms, roughly matching parallel CPU. Synthetic fill gains do not establish
real-document coverage. CPU parsing/codecs and the host raster copy remain.

Borrowed JPEG Huffman prefix tables improved isolated scan decode ~9%, with exact
RGBA at four resolutions. WebP lossless/compressed-alpha prefix changes retained
exact fixture pixels; ordinary VP8 is unchanged. A VP8 normalization trial was
slower and rejected. Codec gains did not qualify end-to-end embedding speedups.

## OCR correctness and rejected speedups

Row-dependent matrix arithmetic intermittently changed a fragile newline/EOS
choice, omitting a paragraph despite normal stop status. At sequence length
1,011 the free-running trace selected EOS rather than newline, losing 204
characters/41 tokens. Canonical-prefix diagnostics concealed accumulated drift;
scoped rollbacks/EOS confirmation did not establish a robust repair.
Retaining the complete prepared singleton operation family fixed the serving
path. Faster incomplete matrix runs are excluded from speed claims.

The final invariant Q4/tail/QKV/carrier/head-frame configuration reduced mean
recognition time by 12.8% with fresh ordinary brackets. Shared-weight kernels were
mostly slower and excluded; widened accumulators changed bits and were rejected.
Recorded native conditions match 128 page observations/79,222 completion tokens,
including reversed order and memory instrumentation. These are transcript parity
gates against ordinary output, not certified OCR accuracy. Ordinary itself failed
one diagram page; the expanded text corpus does not erase that failure.

No Antfly full-corpus MTP gain is qualified. Less drafting or fewer submissions
still left verification/recovery cost; cached seeds initially failed paragraph
parity, and deferred recovery increased verification work. MLX F32 improved a
restricted seven-page workload with overlapping ranges; different weights,
precision/output and a capped eighth page prevent runtime ranking.

## Qualification boundaries

Rendering/startup are excluded from OCR recognition. Embedding benchmarks use
separate checkpoints/scopes. Amortized seconds/page is not individual latency.
Diagnostics/probes are excluded from timing; sampled RSS and Metal allocations
overlap and cannot be added. Frozen-runtime gates remain historical after merges.
All OCR experiments stay opt-in. Replicate gains on idle hosts and representative,
manually checked pages/audio before broader promotion; Apple generation is unmeasured.

## CUDA image residency and video follow-up, October 14–15

The [CUDA video results](../../gemma4-video-cuda-results.json) retain the model
and fixture hashes, timing scopes, and individual warm samples. On an NVIDIA
L4, the six-second synthetic clip's preparation fell from 89.385 to 11.297 s;
generation excluding model loading fell from 104.904 to 26.700 s. These compare
the initial and optimized video implementations in the same worktree. Main had
no corresponding video route. Model loading was 39.225 s in the optimized run.
Both generation runs stopped at the requested 96-token cap; they are smoke
tests, not a video-description accuracy benchmark.

NVDEC maps only the selected presentation timestamps. A CUDA kernel performs
NV12 conversion, resize, normalization, and patch packing before the resident
projector and prompt handoff. Position gathers, QAT clamps, and vision attention
stay on device. F16/BF16 projector weights upload compactly and widen exactly to
F32 on device; activation precision is unchanged. Request-scoped weight caches
retain each weight across frames. Compatible decoder surfaces survive successful
requests; errors discard them, and cancellation is polled during demux and decode.

Five warm decode-through-projector runs had a median of 8.412 s and a p95 of
9.165 s, with zero intermediate tensor readbacks. Each request still uploaded
372,312,032 bytes of weights/metadata. Decode-through-patches alone had a
12.17 ms median, which excludes model projection and generation. NVDEC internal
packet transfers are not tensor-counter events. Full generation downloads final
logits/token IDs and therefore does not have zero total readbacks.

Reproduce from `zig/pkg/inference`, using the repository's Zig version and
worktree-local caches:

```bash
export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-cache/global"
export ZIG_LOCAL_CACHE_DIR="$PWD/.zig-cache"
zig build test-video-media -Dcuda=false -j1
zig build test -Dcuda=true -Dcuda-artifacts=sm89 \
  -Dtest-filter='video ' -Dtest-filter='gemma4 image batching' \
  -Dtest-filter='gemma4 resident position' \
  -Dtest-filter='embeddinggemma2 resident' -j1
zig build test -Dcuda=false -Dtest-filter='video CLI' -j1
zig build -Dcuda=true -Dcuda-artifacts=sm89 -Dstrip=true -j1
zig build build-gemma4-video-bench -Dcuda=true -Dcuda-artifacts=sm89 -j1
zig-out/bin/antfly-gemma4-video-bench \
  src/testdata/video/closed.mp4 .zig-cache/video-projector.json 5 /models/gemma4/mmproj.gguf
zig-out/bin/antfly-inference generate /models/gemma4 \
  'Describe what is visible in this video.' \
  --video src/testdata/video/closed.mp4 --backend cuda \
  --max-tokens 96 --temperature 0 --disable-thinking \
  --print-timing --json-timing .zig-cache/video-generation.json
```

CPU tests cover bounded demux, allocation failures, every fixture truncation and
single-byte corruption, edit-list sampling, timestamp overflow, and fake-driver
decoder callback failures/reuse. CUDA tests cover NVDEC presentation ordering,
cancellation/recovery, CPU/GPU pixel and projector parity, both position layouts,
attention across frame boundaries, QAT clamps, and all 65,536 F16/BF16 bit patterns.
Transfer guards reject intermediate projector exports. The embedded PTX is checked
against its source hash; regenerate it with `tools/cuda/generate_video_ptx.py`
using NVRTC 11.8, without requiring a GPU.

Shared EmbeddingGemma2 packing/pooling improvements do not remove the base CUDA
architecture gate: full-model EmbeddingGemma2 CUDA remains unsupported here.
Metal regression execution requires a separate Mac; this campaign ran on Linux.
The supported local video envelope and runtime dependencies are documented in
the [inference guide](../../../../guides/inference.mdx#local-gemma4-video-prompts).
