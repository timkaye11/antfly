# Florence-2 Metal residency

The Metal reader keeps image features on-device from the patch convolution
through the DaViT stages, position/temporal embeddings, spatial pooling,
projection, prompt assembly, and text encoder. It reuses the shared Metal
gather, concat, reduction, matrix-multiply and command-frame infrastructure.
The existing resident incremental decoder and KV cache remain in use.

The projection table is stored as `[vision_dim, projection_dim]`. A GPU row
lookup converts packed GGUF weights to FP32, then resident matrix multiplication
uses that layout directly. Preparing a transposed dynamic linear slot would
otherwise download its weight buffer. Position indices and prompt token IDs
are small CPU inputs; image decoding/resizing remains CPU preprocessing.

The device path supports the standard Florence feature-source order
`[spatial_avg_pool, temporal_avg_pool]`, including Metal batches. Other layouts
retain their existing fallback. CUDA retains its existing single-image path.
No API or model-lifecycle changes are involved.

Set `ANTFLY_INFERENCE_FLORENCE_DISABLE_DEVICE_VISION_TAIL=1` to restore the
previous host-tail path. Set `ANTFLY_INFERENCE_READ_PROFILE=1` to log
`florence_metal_phase` counters for the complete encoder. Successful warm reads
on the qualified model must report zero `tensor_download_bytes` and zero
`tensor_device_readbacks`; output token sampling is outside that encoder span.

## Downsampling convolution

The dense 3x3, stride-2, padding-1 convolution uses a four-pixel tile when
there are at least 16 input channels. Each thread reuses weights and index
calculations across four neighboring output pixels. The precise FP32 kernel
preserves the original channel/kernel accumulation order, including bias.
Partial tiles and padded edges are supported; other convolution shapes retain
the generic kernel. No intermediate buffers or host transfers are added.

Set `TERMITE_FLORENCE2_METAL_DISABLE_CONV3X3=1` to restore the generic kernel.
`TERMITE_FLORENCE2_METAL_REQUIRE_CONV3X3=1` makes eligible calls fail if the new
kernel is unavailable. The focused suite requires this dispatch and covers
batches, odd dimensions, partial tiles, optional bias, and command frames.

Use `--comparison conv3x3` with the harness below to compare only this kernel,
keeping the resident vision tail enabled on both sides. The report records
recovery coverage, vision time, and downsampling time for each request.

## Validation

From `zig/`, use the bounded build wrapper and worktree-local caches:

```sh
ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-cache/global" \
python3 tools/run_bounded_zig_build.py --max-rss-cap 10737418240 -- \
  build inference-test-florence-metal -Dmetal=true -Doptimize=ReleaseSafe -j1
```

The focused suite requires a real Metal device. Its tensor tests compare
position/pooling, projection, and prompt assembly with the CPU calculations
for single images, two-image batches, and empty/nonempty prompts, while
checking that intermediate tensors do not download.

From the repository root, qualify an installed model with:

```sh
python3 zig/pkg/inference/scripts/benchmark_resources.py \
  --output .zig-cache/florence-qualification/resources.json \
  --timeout 600 --max-rss-mib 8192 --max-swap-growth-mib 0 -- \
  python3 zig/pkg/inference/scripts/florence2/qualify_florence2_residency.py \
  --binary zig/zig-out/bin/antfly-inference-http-server \
  --model-dir /path/to/models/antflydb/Florence-2-base \
  --output .zig-cache/florence-qualification/qualification.json \
  --rounds 2 --iters 3
```

The harness alternates rollback/resident order, warms each case, requires exact
response/token parity, rejects encoder readbacks and regressions above 5%, and
checks malformed input, concurrent reads, and recovery after a client disconnect.
`--photo /path/to/photo.jpg` adds a caption case. Use an unused local port
(default 18091). Model/binary/request hashes are recorded with the evidence.

## Local convolution result (2026-10-15)

On a 16 GiB Apple M4 with Florence-2-base Q4_K, two alternating rounds with
three measured requests per case/profile produced the following medians.
Both profiles used the same ReleaseSmall executable and resident vision tail.

| Case | Generic convolution | Tiled convolution | Request latency reduction |
| --- | ---: | ---: | ---: |
| sparse | 1120.0 ms | 1050.7 ms | 6.2% |
| portrait | 1559.5 ms | 1423.8 ms | 8.7% |
| landscape_table | 1461.4 ms | 1302.8 ms | 10.9% |
| batch | 4192.3 ms | 3911.1 ms | 6.7% |
| photo | 1983.7 ms | 1854.8 ms | 6.5% |

The three downsampling stages took 27.9–34.7% less time. All measured responses
and tokens matched exactly, including the previous residency campaign's
outputs. Both paths reported zero encoder tensor downloads/readbacks.
All 76 focused ReleaseSafe Metal tests passed with no skips. Tests/builds
added no swap; the benchmark recorded zero new swap-outs, although the host
already had about 2.2 GiB of swap allocated. This is local serial evidence;
concurrent serving and disconnect recovery were excluded.

The qualification harness writes machine-readable reports with model, binary,
request and output evidence, per-case timings, and resource limits. Keep these
reports as local or CI artifacts rather than committing them to the repository.

## Serving limitation found during local qualification

Concurrent Florence reads can return HTTP 500 with `INFERENCE_FAILED/QueueFull`
when the shared Metal provider is occupied. This reproduces with both the
host-tail rollback and the resident path; a subsequent serial read succeeds.
The default harness treats this as a failure. `--skip-recovery` runs only the
serial parity/residency/latency gate and does **not** qualify concurrent serving.
The shared-provider lock deliberately uses admission rather than blocking a
worker; the reader error response needs a separate serving fix.
