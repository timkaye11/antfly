# Gemma 4

This note tracks Antfly inference's Gemma 4 generation support, especially Google's
Gemma 4 Multi-Token Prediction (MTP) assistant models.

## Current Status

Antfly inference supports Gemma 4 as a native decoder family through the shared GPT
runtime. The model config already carries Gemma 4-specific metadata such as
sliding/global attention, shared KV-tail metadata, grouped-query dimensions,
per-layer embeddings (PLE), MoE metadata, and final logit softcapping.

For MTP, Antfly inference has a generic native speculative decoding path:

1. A draft model proposes up to `speculative_k` tokens.
2. The target model verifies the drafted span in one forward pass.
3. Matching draft tokens are accepted.
4. On the first mismatch, the target's token is accepted instead.
5. If all drafted tokens match, the target supplies one bonus token.

This is exposed through the server API with `draft_model` and `speculative_k`.
The CLI also supports it:

```sh
antfly inference generate /path/to/google/gemma-4-E2B-it \
  "Explain speculative decoding in one paragraph." \
  --draft-model /path/to/google/gemma-4-E2B-it-assistant \
  --speculative-k 4 \
  --backend metal \
  --print-timing
```

The drafter must use the same tokenizer vocabulary and special token ids as the
target. Speculative decoding is currently native text-only generation; it is not
enabled for multimodal prompts or the ONNX direct path.

## Google Gemma 4 MTP Design

Google's MTP assistants are not just arbitrary smaller language models. They
are paired with a target Gemma 4 checkpoint and are designed to reduce drafting
overhead:

- The assistant shares the target input embedding table.
- The assistant consumes target last-layer activations.
- The assistant concatenates target activations with token embeddings and
  down-projects into the drafter width.
- The assistant can share target-side KV/cache state instead of recomputing the
  whole prompt independently.
- E2B and E4B assistants include an efficient clustered embedder to avoid a
  full-vocabulary projection for every draft step.

## Source and Artifact Confirmation

Sources checked on 2026-05-05:

- Google's launch post and Hugging Face model cards describe Gemma 4 MTP as
  assisted/speculative decoding: an assistant proposes multiple tokens, and the
  target model verifies them in parallel while preserving target quality.
- Hugging Face exposes `google/gemma-4-*-assistant` as Transformers
  `AutoModelForCausalLM` artifacts, with `generation_config.json` marking
  `"is_assistant": true`, `"num_assistant_tokens": 6`, and a constant assistant
  token schedule.
- The public Transformers `v5.7.0` and `v5.8.0` Gemma 4 source does not expose
  `Gemma4AssistantForCausalLM` or `gemma4_assistant` implementation details.
  The public source trail for runtime behavior is currently LiteRT-LM, not the
  tagged Transformers Gemma 4 model files.
- LiteRT-LM's MTP drafter runtime loads a `tf_lite_mtp_drafter` model section,
  uses a base-model `verify` signature, and drafts greedily. It concatenates
  token embeddings with the verifier/base activations into an `activations`
  input, runs the drafter repeatedly, and verifies `G + 1` target positions in
  one pass. On mismatch it accepts the verifier token; on full match it accepts
  the verifier bonus token.

Confirmed assistant artifact structure:

- `antfly inference pull google/gemma-4-E2B-it-assistant` downloads the official
  safetensors assistant into
  `~/.antfly/inference/models/google/gemma-4-E2B-it-assistant`.
- `google/gemma-4-E2B-it-assistant` config:
  - `architectures`: `Gemma4AssistantForCausalLM`
  - `model_type`: `gemma4_assistant`
  - `backbone_hidden_size`: 1536
  - compact text stack: 4 layers, hidden size 256, 4 attention heads, 1 KV head,
    sliding attention for layers 0-2 and full attention for layer 3
  - `use_ordered_embeddings`: true, `num_centroids`: 2048,
    `centroid_intermediate_top_k`: 32
- E2B assistant safetensors header:
  - `pre_projection.weight`: `[256, 3072]`
  - `post_projection.weight`: `[1536, 256]`
  - `model.embed_tokens.weight`: `[262144, 256]`
  - `masked_embedding.token_ordering`: `[262144]`
  - `masked_embedding.centroids.weight`: `[2048, 256]`
- E4B assistant safetensors header:
  - `pre_projection.weight`: `[256, 5120]`
  - `post_projection.weight`: `[2560, 256]`
  - otherwise follows the E2B compact 256-wide, 4-layer drafter shape
- 26B-A4B assistant safetensors header:
  - `pre_projection.weight`: `[1024, 5632]`
  - `post_projection.weight`: `[2816, 1024]`
  - `model.embed_tokens.weight`: `[262144, 1024]`
  - no `masked_embedding.*` tensors in the inspected safetensors header

The projection shapes confirm LiteRT-LM's runtime contract: MTP drafter input is
`concat(token_embedding, verifier_or_target_activation)` at
`2 * backbone_hidden_size`, the compact assistant stack runs at its own hidden
size, and `post_projection` returns to target/backbone hidden size for the next
chained draft step.

Runtime findings from implementation:

- The assistant is query-only. It owns Q/O projections and MLP weights, but no
  K/V projections. All assistant layers must read target K/V banks.
- The 4 assistant layers do not map to target layers 0-3. They map by attention
  type to the target's last non-shared KV donor layers:
  - E2B target: sliding donor layer 13, full-attention donor layer 14.
  - E4B community LiteRT extraction reports the analogous banks as layers 22
    and 23.
- E2B/E4B `masked_embedding.token_ordering` is a full vocabulary permutation,
  and `masked_embedding.centroids.weight` is `[2048, 256]`. This supports a
  clustered output head: score centroids, keep the configured top 32 centroid
  groups, then select the best token inside those groups from assistant
  embedding logits.
- The official E2B assistant config says `tie_word_embeddings = true` and does
  not include an explicit `lm_head.weight`; the current implementation uses the
  assistant embedding matrix for logits, then applies the clustered mask when
  `masked_embedding.*` tensors are present.
- MLX-VLM's public Gemma 4 assistant implementation and the SeatownSin
  extracted PyTorch drafter both highlight runtime details that are easy to get
  subtly wrong:
  - the target activation passed to the drafter is the target hidden state that
    predicted the current token, not the hidden state after consuming that
    token;
  - the drafter position id is held constant during an autoregressive MTP draft
    block.
- The extracted PyTorch drafter captures the output of `text_model.norm`, so
  Antfly inference now uses final-RMSNorm target hidden states for both target logits and
  MTP drafter handoff. The older pre-final-RMSNorm handoff is retained only as
  implementation scaffolding for comparison.
- The public `masked_embedder.py` implementation treats
  `masked_embedding.token_ordering` as centroid-to-token ordering:
  `ordering[c * cluster_size .. (c + 1) * cluster_size]` is the token set for
  centroid `c`. That matches the current baseline implementation; the inverse
  interpretation is now only a debug experiment.

The current Antfly inference implementation uses the same acceptance/verification
algorithm. Phase 1 used an independent decoder drafter. Phase 2 now has a
Gemma-specific MTP draft step that consumes target hidden activations, reads
target K/V, and chains projected activations. Verification is still target-owned.

## Implementation Plan

### Phase 1: Generic Assistant Drafters

Status: implemented for the native server API and CLI.

- Load an optional `draft_model` alongside the target model.
- Validate tokenizer compatibility before generation.
- Allocate a separate draft KV manager and decode state.
- Prefill target and draft with the same text prompt.
- Run the existing draft/verify speculative loop.
- Report speculative rounds, drafted tokens, accepted draft tokens, rejected
  draft tokens, corrections, and bonus tokens in CLI timing output.
- Disable direct ONNX and one-token artifact shortcuts when a drafter is
  requested so generation cannot silently ignore the assistant.

This should work with Gemma 4 `*-assistant` checkpoints if they are exported in
a format the native loader understands as a decoder-only model.

### Phase 2: Gemma 4 MTP Runtime

Status: implemented with Gemma-specific runtime ownership and remaining
acceptance-rate investigation.

Add a Gemma-specific drafter runtime that understands assistant checkpoints as
MTP heads instead of independent decoders:

1. Extend model metadata parsing for MTP assistant structure: done.
   - `model_type = "gemma4_assistant"` and
     `architectures = ["Gemma4AssistantForCausalLM"]`,
   - `backbone_hidden_size`,
   - assistant layer count and hidden size,
   - `pre_projection.weight` and `post_projection.weight`,
   - clustered embedder metadata for E2B/E4B where present,
   - explicit target-model compatibility identifiers when available.
2. Expose target drafter activations from the target decode pass: done for
   native generation through `forwardAllLogitsAndHiddenHost` and
   `materializeAcceptedTokenKvAndReturnHidden`. The MTP path uses final
   RMSNorm hidden for the drafter handoff, matching the extracted PyTorch
   reference's `text_model.norm` hook.
3. Add a Gemma 4 MTP draft helper in `src/architectures/gemma4_mtp.zig`: done.
   - borrow or alias target token embeddings at the target/backbone width,
   - consume target final hidden activations,
   - build drafter inputs from `concat(token_embedding, target_hidden)`,
   - run the assistant transformer stack,
   - produce assistant logits and clustered candidate logits,
   - retain the drafter's `projected_activations`/post-projection output so the
     next draft step can chain from the prior assistant step without rerunning
     the target.
4. Replace independent draft prompt prefill with target-activation seeding: done
   for `gemma4_assistant` draft configs.
5. Keep the existing verification path unchanged: done. Target-side verification is
   what preserves output quality and sampling semantics.
6. Extend telemetry: partially done. `ANTFLY_GEMMA4_MTP_DEBUG=1` prints drafted
   token ids and verifier choices for acceptance debugging.
7. Move Gemma 4 runtime-specific construction into
   `src/architectures/gemma4_runtime.zig`: done.
   - the explicit backend contract is `gemma4_gated_ple_shared_kv`,
   - shared-KV layer specs, PLE slots, head-norm slots, and final/tail slots are
     built by the Gemma 4 architecture module,
   - per-layer output scales are resolved to scalar runtime metadata for the
     whole-frame path instead of retained backend tensors,
   - Gemma 4 MTP assistants skip standalone shared-decoder prewarm so valid
     assistant artifacts no longer emit the stale `MissingWeight` warning.

Current smoke result:

```sh
antfly inference generate ~/.antfly/inference/models/ggml-org/gemma-4-e2b-it-gguf \
  "Write one short sentence about databases." \
  --backend metal \
  --draft-model ~/.antfly/inference/models/google/gemma-4-E2B-it-assistant \
  --speculative-k 2 \
  --max-tokens 4 \
  --print-token-ids \
  --print-timing
```

This runs end-to-end and preserves target-owned verification. After fixing the
activation handoff to use the hidden state that predicted the current token, the
same short smoke accepted one draft token:

```text
speculative: rounds=2 drafted=4 matched=1 rejected=3 accepted=3 corrections=2 bonus=0
```

A longer `--speculative-k 6 --max-tokens 12` smoke accepted 2 of 41 drafted
tokens against the local `ggml-org` Q8_0 GGUF target. The runtime is now
productive, but acceptance is still far below the published best-case numbers.
The remaining likely causes are source/model pairing differences between the
official safetensors assistant and the local GGUF target, quantization effects in
the target, or a still-missing detail in the clustered output head.

### CUDA Branch Status

Status checked on 2026-06-25 on branch `gemma4-e4b-qat`:

- `zig build -Dcuda=true` succeeds.
- `antfly-inference cuda-info --smoke` succeeds on an NVIDIA L4 (`sm_89`).
- The CUDA smoke now covers the Gemma4-specific WIP primitives:
  add-multiply-scalar, RMSNorm-add-multiply-scalar, head-norm+RoPE, GQA, RoPE,
  and MTP masked argmax.
- Local validation artifacts used in this pass:
  `.models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
  for E2B CUDA smoke, `.models/google/gemma-4-12B-it-q4_k` for 12B Q4 CUDA
  validation, and `.models/unsloth/gemma-4-E2B-it-qat-GGUF/MTP/` for E2B MTP
  assistant GGUFs.
- E4B QAT status checked on 2026-06-24 on branch `gemma4-e4b-qat`:
  `google/gemma-4-E4B-it-qat-q4_0-gguf` pulls as
  `gemma-4-E4B_q4_0-it.gguf` plus `gemma-4-E4B-it-mmproj.gguf`; CUDA generation
  succeeds on NVIDIA L4 with the resident Gemma4 path. An 8-token target-only
  smoke reported `load_model=25.274s`, `generate=1.076s`, and
  `decode_tok_per_s=11.096` with `/tmp/gemma4-e4b-qat-cuda-8.json`.
- Q4_0 CUDA decode now has fused QAT-aware SIMT paths for QKV projection,
  gate/up pair projection, gated-down projection, and standalone single-row
  linear decode. A 32-token E4B QAT run on NVIDIA L4 reported
  `decode_tok_per_s=17.307`, `generate=2.446s`, and `load_model=26.481s` with
  `/tmp/gemma4-e4b-qat-cuda-32-fused-tile4.json`. The run hit
  `qkv_fused_q4_0=768`, `linear_pair_fused_q4_0=1344`, and
  `gated_down_fused_q4_0=1344` with zero QKV, pair, or gated-down fallbacks.
- The E4B QAT CUDA production gate now defaults to 512 generated tokens and
  requires the Q4_0 fused counters to be non-zero with zero QKV, pair, and
  gated-down fallbacks. It also requires the fast f32 GQA decode counter to be
  non-zero with zero fast-GQA fallbacks, persistent decode graph replay via
  `ANTFLY_INFERENCE_CUDA_DECODE_GRAPH_REPLAY=required`, stable temp reuse,
  delayed capture (`ANTFLY_INFERENCE_CUDA_CAPTURE_MIN_ALLOC_SEQ=10000`), and a
  launch-density ceiling of 30 launches/token. It also requires raw i32 greedy
  token export with `cuda_generate.to_float32_calls=0` and
  `cuda_generate.to_float32_bytes=0`, so the token readback path does not
  regress to float conversion, and delayed CUDA token readback with
  `ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK=1` plus
  `cuda.download_syncs<=4`. The 128-token tuned replay run on 2026-06-25 reported
  `decode_tok_per_s=15.038`,
  `graph_capture_persistent_replays=116`, `launches_per_token=76.469`,
  `launch_attention_gqa_decode_fast=42`, and
  `launch_attention_gqa_decode_fast_fallbacks=0` with
  `/tmp/gemma4-e4b-qat-fastgqa-128.json`.
- Before the branch-hoisted Q4_0 changes, 512-token f32-KV E4B QAT CUDA decode
  was at the 15 tok/s floor but not yet comfortably above it. The tuned
  stable-reuse graph replay run
  `/tmp/gemma4-e4b-qat-fastgqa-final-512.json` reported
  `decode_tok_per_s=14.987`, `graph_capture_persistent_replays=500`,
  `launches_per_token=25.867`, `launch_attention_gqa_decode_fast=42`, and
  `launch_attention_gqa_decode_fast_fallbacks=0`. The later default-path
  validation `/tmp/gemma4-e4b-qat-gated-default-512.json` reported
  `decode_tok_per_s=15.019`, `device_token_handoff_hits=511`,
  `device_token_handoff_fallbacks=0`, `device_token_handoff_seeds=1`,
  `graph_capture_persistent_replays=500`, `launches_per_token=25.867`, and zero
  fast-GQA or Q4_0 fused-kernel fallbacks. This proved the default graph replay,
  fast-GQA, device-token handoff, and Q4_0 fused paths were stable enough to
  focus the next pass on Q4_0 kernel work and token export.
- The 2026-06-25 post-merge QAT pass added pinned host staging for tiny scalar
  uploads and downloads. On the 512-token f32-KV E4B QAT replay benchmark it
  reduced `upload_syncs` from 1024 to 0 and staged all 512 token downloads
  through pinned memory while preserving `graph_capture_persistent_replays=500`,
  fast-GQA hits, and zero Q4_0 fused fallbacks. This intermediate pass was still
  borderline before the later Q4_0 branch-hoisting, PLE fusion, and token-export
  work:
  `/tmp/gemma4-e4b-qat-pinned-d2h-512.json` reported
  `decode_tok_per_s=14.974` with `pinned_scalar_downloads=512`, while repeat
  `/tmp/gemma4-e4b-qat-pinned-d2h-512-r2.json` reported
  `decode_tok_per_s=14.829`.
- The same pass enabled CUDA greedy device-token handoff for Gemma4 PLE/QAT in
  the native generation pipeline. After one host seed, the next-token tensor is
  passed directly into the following decode step; JSON timing reports this under
  `generation_decoder_runtime.device_token_handoff_*`. The 512-token validation
  run `/tmp/gemma4-e4b-qat-device-token-active-512.json` reported
  `device_token_handoff_attempts=511`, `device_token_handoff_hits=511`, and
  `decode_tok_per_s=15.048`, but the repeat
  `/tmp/gemma4-e4b-qat-device-token-active-512-r2.json` reported
  `decode_tok_per_s=14.898`. Treat this as correctness/telemetry groundwork; the
  later delayed-readback path keeps the token tensor on device, enqueues async
  pinned scalar downloads, and reduces the production 512-token QAT target to two
  generate-time download syncs for host-visible emission and stop handling.
- A deeper Gemma4 gated-runtime token-tensor decode probe is available behind
  `ANTFLY_INFERENCE_CUDA_GATED_TOKEN_TENSOR_DECODE=1`, but it is not production
  default. The 16-token L4 diagnostic
  `/tmp/gemma4-e4b-qat-gated-tensor-16.json` regressed to
  `decode_tok_per_s=11.503`, `graph_capture_persistent_replays=0`, and zero
  device-token handoff hits. With the probe off again,
  `/tmp/gemma4-e4b-qat-gated-default-16.json` returned to
  `decode_tok_per_s=18.265`, `device_token_handoff_hits=15`,
  `device_token_handoff_fallbacks=0`, and
  `graph_capture_persistent_replays=4`. Keep this path as implementation
  scaffolding until the gated Gemma4 PLE token contract is graph-replay-safe.
- Q4_0 tile8 decode kernels are available behind
  `ANTFLY_INFERENCE_CUDA_Q4_0_DECODE_TILE8=1`, with per-kernel overrides
  `ANTFLY_INFERENCE_CUDA_Q4_0_QKV_TILE8=1`,
  `ANTFLY_INFERENCE_CUDA_Q4_0_PAIR_TILE8=1`,
  `ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE8=1`, and
  `ANTFLY_INFERENCE_CUDA_Q4_0_LINEAR_TILE8=1`. The old all-tile8 comparison was
  slower (`/tmp/gemma4-e4b-qat-pinned-f32-tile8-512.json` reported
  `decode_tok_per_s=14.791`), and the current L4 production gate defaults the
  tile8 family off. Q4_0 gated-down now uses the precompute path on the tile4
  kernel by default; QKV, gate/up-pair, gated-down tile8, and linear tile8 remain
  opt-in A/B controls.
  `ANTFLY_INFERENCE_CUDA_Q4_0_GATED_DOWN_TILE16=1` is also compiled as an
  experiment, but the 128-token L4 check
  `/tmp/gemma4-e4b-qat-tile-split-down16-128.json` reported
  `decode_tok_per_s=16.028`, slower than the tile4 baseline and gated-down
  tile8, so it is not promoted into the gate.
- Q4_0 decode kernels now hoist the packed-nibble offset and high-nibble flag
  outside the inner column loop for the tile, embedding, QKV, pair, gated-down,
  and argmax paths. A warp-broadcast scale probe regressed
  (`/tmp/gemma4-e4b-qat-warp-scale-128.json` reported
  `decode_tok_per_s=14.735`) and was removed. The branch-hoisted production
  path improved the 128-token L4 check to
  `/tmp/gemma4-e4b-qat-q4nibble-128.json` at `decode_tok_per_s=19.997`; after
  adding the same helper to Q4_0 embedding lookup,
  `/tmp/gemma4-e4b-qat-q4nibble-embed-128.json` reported
  `decode_tok_per_s=19.913`. The
  512-token repeats `/tmp/gemma4-e4b-qat-q4nibble-512.json` and
  `/tmp/gemma4-e4b-qat-q4nibble-512-r2.json` were historical tile8 A/B runs and
  reported `17.376` and `17.227` tok/s with
  `graph_capture_persistent_replays=500`, all post-seed device-token handoff
  hits, and zero Q4_0 fused-kernel fallbacks. Later 512-token production gates
  defaulted tile8 off after the tile4 gated-down precompute path proved faster.
  Raw i32 token export then removed the final token-path `toFloat32`
  conversion:
  `/tmp/gemma4-e4b-qat-i32-export-128.json` reported
  `decode_tok_per_s=19.953` with `cuda_generate.to_float32_calls=0` and
  `cuda_generate.to_float32_bytes=0`. PLE combine now uses CUDA
  `addWeightedScalars`, replacing the previous add-multiply-plus-scale sequence
  with one weighted combine launch. The hot device-token PLE path also has an
  opt-in Q6_K RMS/embed construction fusion, but that path is not the default
  because it has not beaten the default gate reliably. The current
  target-vs-Q4_K production gate at
  `/tmp/gemma4-q4-target-vs-q4k-new-defaults-gate-20260625-r1` ran one 512-token
  E4B Q4_K baseline and one 512-token E4B QAT Q4_0 target pass with delayed token
  readback, PLE fusion, scaled embedding lookup, and fused PLE
  embedding-combine enabled. The Q4_K baseline reported
  `decode_tok_per_s=16.676`; QAT reported `30.125` tok/s, a `1.806x`
  QAT-over-Q4_K ratio against the gate's default
  `MIN_E4B_QAT_OVER_Q4K_RATIO=1.25` floor. The QAT run kept
  `graph_capture_persistent_replays=500`, `launches_per_token=21.787`,
  `launch_embedding=513`, `launch_scalar=0`, `add_mul_scalar_fused=512`,
  `linear_activation_slice_fused_q4_0=504`, `qkv_fused_q4_0=288`,
  `linear_pair_fused_q4_0=504`, `gated_down_fused_q4_0=504`,
  `gated_down_fused_q4_0_precompute=462`, `gated_down_fused_q4_0_tile8=0`,
  `device_token_handoff_hits=511`, Q4_0 fused fallbacks at zero,
  `graph_capture_capacity_skips=0`, and `download_syncs=2`.
  The E4B QAT production gate default is now `MIN_E4B_QAT_TOK_S=24.0`,
  `MIN_E4B_QAT_RUN_TOK_S=24.0`, `MIN_E4B_QAT_OVER_Q4K_RATIO=1.25`,
  `E4B_QAT_REPEATS=2`, `E4B_QAT_PENDING_TOKEN_READBACK=1`,
  `E4B_QAT_MAX_DOWNLOAD_SYNCS=4`, `E4B_QAT_REQUIRE_PLE_FUSION=1`,
  `E4B_QAT_Q4_0_GATED_DOWN_TILE8=0`, `E4B_QAT_REQUIRE_GATED_DOWN_TILE8=0`,
  `E4B_QAT_Q4_0_PLE_GATE_FUSION=1`, `E4B_QAT_PLE_RMS_EMBED_FUSION=0`, and
  `E4B_QAT_MAX_LAUNCHES_PER_TOKEN=22.5`.
- The long-context replay gate is opt-in with `RUN_E4B_QAT_LONG=required`. The
  current integrated run at `/tmp/gemma4-cuda-production-gate-20260625-integrated`
  requested 1024 tokens, forced graph replay KV capacity to 2048, required at
  least 900 generated tokens, and generated 936 tokens before EOS. It reported
  `decode_tok_per_s=15.844`, `graph_capture_persistent_replays=926`,
  `graph_capture_capacity_skips=0`, `launches_per_token=14.238`,
  `launch_embedding=939` (`tokens+3`, allowed for EOS/lookahead cleanup),
  `launch_scalar=0`, `add_mul_scalar_fused=938`,
  `device_token_handoff_hits=937`, `device_token_handoff_fallbacks=0`,
  `download_syncs=1`, and zero Q4_0/GQA fallback counters.
- The preloaded resident serving gate is opt-in with
  `RUN_E4B_QAT_RESIDENT=required` in the package-local production gate. The
  current forced-replay run at
  `/tmp/gemma4-cuda-qat-resident-replay-20260625-r3` started
  `antfly-inference run` with `--preload-model generator:cuda:<E4B_QAT>` and
  completed two 128-token warm HTTP requests at `17.013` and `17.043` tok/s E2E
  (`avg=17.028`) against the 12.0 tok/s floor. The gate now also asserts serving
  graph replay from the server log and reported `replays=246` with `floor=42`.
  The package-local QAT-vs-Q4_K serving ratio gate at
  `/tmp/gemma4-cuda-pkg-resident-q4k-ratio-20260625-r1` reported QAT resident
  warm requests at `17.049` and `17.030` tok/s E2E (`avg=17.040`) and Q4_K
  resident warm requests at `12.861` and `12.831` tok/s E2E (`avg=12.846`), for
  a serving ratio of `1.326x` against the `1.05x` floor.
  The package-local 512-token serving gate at
  `/tmp/gemma4-cuda-pkg-resident-q4k-ratio-512-20260625-r1` reported QAT
  resident warm requests at `16.211` and `16.061` tok/s E2E (`avg=16.136`) and
  Q4_K resident warm requests at `12.711` and `12.827` tok/s E2E
  (`avg=12.769`), for a serving ratio of `1.264x` against the `1.05x` floor
  with `graph_replays=1014` for both resident paths. The resident gate now
  defaults to 512 generated tokens, requires CUDA graph replay, and uses a
  15.0 tok/s warm floor; set `E4B_QAT_RESIDENT_TOKENS=128` only for shorter
  smoke checks.
  The top-level production gate exposes the same opt-in resident check; the run
  at `/tmp/gemma4-cuda-top-resident-replay-20260625-r2` reported `17.111` and
  `17.134` tok/s E2E (`avg=17.123`) with `graph_replays=246`, and its
  `readiness.json` includes the resident step outcome. New top-level runs also
  write `cuda_environment.json` and copy it into `readiness.json` under
  `environment.cuda_smoke`, so cross-GPU threshold reviews should compare
  `device_name`, `compute_capability`, `driver_version`, `artifacts`, and the
  CUDA smoke/capability map along with tok/s. The package-local gate writes the
  same `cuda_environment.json` and summary line by default; use
  `RUN_CUDA_ENV=required` when metadata capture is part of the release gate.
  Both gates also write `e4b_qat_production_summary.json`, a single structured
  artifact with QAT/Q4_K target ratios, resident warm ratios, long-context QAT
  throughput, soak latency/aggregate throughput, backpressure queue metrics,
  graph replay counts, and the CUDA environment for CI dashboards. Its
  `verdict` section is threshold-aware: enabled gate phases pass their required
  floors into the summary so CI can fail on missing evidence or a regressed
  QAT/Q4_K ratio without scraping free-form logs.
  For provider-competitive throughput work, measure E4B QAT base decode and
  MTP/speculative decode as separate products. Community numbers in the
  `40+ tok/s` range often come from different hardware/runtime stacks, while
  the `120 tok/s` 12B QAT report used a Gemma 4 MTP assistant. The top-level
  production gate has an E4B QAT MTP matrix:

  ```sh
  RUN_E4B_QAT_MTP=required \
  E4B_QAT_ASSISTANT_Q8=.models/google/gemma-4-E4B-it-assistant \
  E4B_QAT_MTP_TOKENS=512 \
  E4B_QAT_MTP_SPEC_KS="2 4 6" \
  E4B_QAT_MTP_PROMPT_FILTER="ants_chat code_chat" \
  RUN_12B_MTP=0 RUN_E2B_MTP=0 \
  scripts/gemma4_cuda_production_gate.sh
  ```

  This writes `mtp_e4b_qat/summary.tsv`, and
  `e4b_qat_production_summary.json` exposes the parsed matrix under
  `mtp.mtp_e4b_qat` with `best` and `best_active` rows. Use those rows to track
  acceptance rate, active policy decisions, and speedup over target-only before
  claiming MTP is competitive with external providers.
  The 2026-06-25 focused MTP follow-up added explicit benchmark knobs for
  `MTP_UNSAFE_TARGET_REPLAY`, `MTP_REPLAY_VERIFY_ROWS`,
  `MTP_ASSISTANT_REPLAY`, `MTP_MATERIALIZE_REPLAY`,
  `CUDA_CAPTURE_PERSISTENT_REPLAY`, and `DRAFT_EMBED_CACHE`, plus draft
  embedding-cache and draft graph-replay counters in the JSON/TSV summaries.
  Runtime default `ANTFLY_GEMMA4_MTP_DRAFT_EMBED_CACHE` is now `256`; the memory
  cost is only a few MiB for E4B and avoids repeat target-embedding copies when
  prompts or generated text reuse token ids. The forced/profile baseline at
  `/tmp/gemma4-e4b-qat-mtp-profile-baseline-20260625-r1` still did not beat
  target-only: f32-KV target rows were `35.316` and `34.356` tok/s, while Q8
  assistant MTP ranged from `26.056` to `31.149` tok/s for `k=1` and from
  `27.831` to `28.798` tok/s for `k=2`. Opt-in unsafe target replay at
  `/tmp/gemma4-e4b-qat-mtp-profile-target-replay-20260625-r1` captured
  persistent target replays and cut target launch counts (`k=1` ants
  `42378 -> 30410`), but still topped out at `31.250` tok/s and changed one
  prompt's stop behavior, so it remains an investigation knob, not a production
  default. Opt-in assistant persistent replay at
  `/tmp/gemma4-e4b-qat-mtp-profile-assistant-replay-persistent-20260625-r2`
  reduced draft launches sharply (`k=1` ants `2769 -> 470`) and recorded draft
  persistent replays, but acceptance fell (`461 -> 187` permille on `k=1`
  ants), so it is also not production-safe. Cache dtype retests with replay off
  kept f32 as the MTP baseline: polar4 at
  `/tmp/gemma4-e4b-qat-mtp-profile-polar4-20260625-r1` reached target-only
  `35.724`/`35.184` tok/s but MTP remained `27.462`-`31.750` tok/s and changed
  output length; turbo3 at `/tmp/gemma4-e4b-qat-mtp-profile-turbo3-20260625-r1`
  regressed target-only to about `27` tok/s and MTP to `20.786`-`24.680` tok/s.
  Production policy remains: keep E4B QAT MTP behind auto/probe gating with zero
  active candidates until a matrix beats target-only with stable acceptance and
  matching stop behavior.
  The 2026-06-26 replay-correctness pass made MTP final-hidden graph replay
  context-keyed by default via `ANTFLY_GEMMA4_MTP_REPLAY_CONTEXT_KEY=1`.
  The replay key now includes batch, sequence length, query rows, position
  offset, KV length, total sequence length, and KV position offset, preventing a
  graph captured at one decode/KV context from replaying at another. The old
  broad label-only key remains available for profiling with
  `ANTFLY_GEMMA4_MTP_REPLAY_CONTEXT_KEY=0`, but it is not production-safe.
  Assistant replay is also gated independently from target replay, so
  `MTP_TARGET_REPLAY=off MTP_ASSISTANT_REPLAY=1` exercises only the drafter. The
  validation artifact
  `/tmp/gemma4-e4b-qat-mtp-assistant-replay-context-targetoff-20260626-r1`
  matched the no-replay acceptance counters exactly (`k=1` ants `39/18/57`,
  code `15/13/28`; `k=2` ants `64/25/57`, code `20/18/28` for
  drafted/matched/accepted). With strict keys, `k=2` still got safe within-round
  assistant replay (`31` and `9` draft persistent replays) and reduced draft
  launches (`4544 -> 2752`, `1420 -> 904`) without changing acceptance. Strict
  target replay at `/tmp/gemma4-e4b-qat-mtp-target-replay-context-20260626-r1`
  also matched baseline tokens and acceptance, but produced no persistent target
  replays because target verification contexts changed each round; keep target
  replay opt-in until a context-safe reuse strategy is proven.
  The top-level production gate now has first-class MTP replay and acceptance
  diagnostics for this path:
  `RUN_E4B_QAT_MTP_REPLAY_STABILITY=auto|required|off` runs a no-replay
  baseline and a strict assistant-replay matrix, then writes
  `mtp_replay_stability.json` after comparing status, generated token counts,
  finish reasons, drafted/matched/accepted counters, and acceptance permille.
  `RUN_E4B_QAT_MTP_REPLAY_512=required` writes a production 512-token strict
  assistant-replay matrix at `mtp_replay_512/summary.tsv`.
  `RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX=required` writes eight forced/profile
  diagnostic matrices named `mtp_acceptance_matrix_*` for the
  position-mode, target-hidden-source, and concat-order combinations. These
  diagnostic rows are summarized under `mtp_acceptance_matrix` in
  `e4b_qat_production_summary.json`; production speed verdicts only apply to
  the normal `mtp` matrices.

  ```sh
  RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=off \
  RUN_E4B_QAT_COMPRESSED_KV=off RUN_12B_MTP=0 RUN_E2B_MTP=0 \
  RUN_E4B_QAT_MTP=off \
  RUN_E4B_QAT_MTP_REPLAY_STABILITY=required \
  RUN_E4B_QAT_MTP_REPLAY_512=required \
  RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX=required \
  E4B_QAT_ASSISTANT_Q8=.models/google/gemma-4-E4B-it-assistant \
  scripts/gemma4_cuda_production_gate.sh
  ```

  Package-local runs can reproduce the same MTP replay path through
  `RUN_MTP=required`, `MTP_TARGET_REPLAY=off`, `MTP_ASSISTANT_REPLAY=1`,
  `MTP_REPLAY_CONTEXT_KEY=1`, `CUDA_CAPTURE_PERSISTENT_REPLAY=1`,
  `CUDA_TEMP_SLOT_PERIOD=1`, and `CUDA_TEMP_SLOT_SKIP=0`.
  The top-level production gate also has an opt-in compressed-KV QAT check:

  ```sh
  RUN_TARGET_ONLY=0 RUN_E4B_QAT_LONG=off RUN_E4B_QAT_RESIDENT=off \
  RUN_E4B_QAT_COMPRESSED_KV=required RUN_12B_MTP=0 RUN_E2B_MTP=0 \
  RUN_E4B_QAT_MTP=off \
  scripts/gemma4_cuda_production_gate.sh
  ```

  It forces `--cache-dtype` with
  `E4B_QAT_COMPRESSED_KV_DTYPE=polar4` and
  `E4B_QAT_COMPRESSED_KV_TURBOQUANT_MIN_TOKENS=0`, then validates 512 tokens,
  decode tok/s, graph replays, download syncs, capacity skips, and compressed-V
  read/write counters. The 2026-06-25 artifact
  `/tmp/gemma4-q4-compressed-kv-new-defaults-gate-20260625-r1` reported
  `38.823 tok/s`, `500` persistent replays, `501` graph replays, `2` download
  syncs, `0` capacity skips, `462` fast compressed-GQA launches, `504`
  compressed-V reads, `288` compressed-V writes, `24` paged block-table uploads,
  `504` fused PLE gate projections, `462` Q4_0 gated-down precompute hits, zero
  Q4_0 gated-down tile8 hits, and `22.303` launches/token. The attention-read
  path now elides the block-table lookup when the paged KV allocation is
  identity-contiguous while retaining the normal block-table path for
  non-contiguous pages. The final validation artifact
  `/tmp/gemma4-q4-compressed-kv-identity-attn-only-gate-20260625-r1` reported
  `38.956 tok/s`, `504` identity attention reads, `24` paged block-table
  uploads, `500` persistent graph replays, `2` download syncs, and `0` capacity
  skips; earlier attention-only repeats landed at `39.354` and `39.219` tok/s.
  Treat this as a production-candidate throughput path with the page-boundary
  graph-break blocker closed, paged/block-table Polar4 compressed attention
  active, and the row-1 Q4_0 gate/up pair kernel defaulting to the 4-warp tile4
  variant. Q4_0 gated-down now defaults to materializing
  `activation(gate) * up` once before the down projection; the PLE RMS/embed
  construction fusion remains opt-in because its launch reduction has not
  reliably beaten the default path in gate runs. The combined gate/up
  activation kernel remains opt-in via
  `ANTFLY_INFERENCE_CUDA_Q4_0_GATE_UP_ACTIVATION_PRECOMPUTE=1` because it
  regressed to `37.185 tok/s` on the 512-token gate. Write-side block-table
  elision was tested and backed out after regressing to `38.850 tok/s`. Treat
  the repeatable near-39.5 tok/s result as the focused production target; the
  full-release blocking floor is set to `36.0 tok/s` to avoid flaking on
  mixed-gate run-to-run variance, with `40 tok/s` kept as a
  stretch/provider-reference target rather than a hard blocker. The
  industry-grade focused gate at
  `/tmp/gemma4-e4b-qat-industry-compressed-gate-20260625-r1` passed with
  `39.102 tok/s`, `500` persistent graph replays, `2` download syncs, `0`
  capacity skips, `504` compressed-V reads, `288` compressed-V writes, `24`
  paged block-table uploads, `504` identity attention reads, `462` fast GQA
  launches, and `0` compressed-KV write fallbacks.
  To require a fixed competitive floor without a provider baseline, enable the
  competitive-floor verdict:

  ```sh
  RUN_E4B_QAT_COMPETITIVE_FLOOR=required \
  RUN_E4B_QAT_COMPRESSED_KV=required \
  E4B_QAT_COMPETITIVE_FLOORS="compressed_kv_decode_tok_s=36.0" \
  scripts/gemma4_cuda_production_gate.sh
  ```

  When `RUN_E4B_QAT_COMPRESSED_KV=required`, the gate and
  `e4b_qat_production_summary.json` now require more than speed: at least the
  configured token count, persistent graph replays, bounded download syncs, zero
  capacity skips, compressed V reads/writes, paged block-table uploads, identity
  attention fast-path hits for the fresh contiguous benchmark, fast GQA launches,
  and zero compressed-KV write fallbacks. This keeps a provider-readiness
  artifact from passing if the benchmark silently falls back to f32 KV,
  non-paged attention, or host-heavy decode.

  Supported floor metrics are `compressed_kv_decode_tok_s`,
  `target_decode_tok_s`, `long_decode_tok_s`, `resident_e2e_tok_s`,
  `soak_aggregate_tok_s`, and `backpressure_accepted_e2e_tok_s`. The fixed
  local floor should target `compressed_kv_decode_tok_s` because that is the paged
  Polar4 KV production speed path; the f32 target metric remains useful for
  QAT/Q4_K apples-to-apples kernel comparisons. This is intentionally separate
  from the QAT/Q4_K ratio check so the gate can distinguish local kernel
  progress from provider-competitive throughput.
  Provider comparisons can be enforced with
  `RUN_E4B_QAT_PROVIDER_COMPARISON=required` plus either
  `E4B_QAT_PROVIDER_BASELINE_JSON=/path/to/provider-baselines.json` or
  `E4B_QAT_PROVIDER_BASELINE_INLINE='{"baselines":[...]}'`. Baseline entries
  specify a `provider`, a `metric` such as `compressed_kv_decode_tok_s`,
  `resident_e2e_tok_s`, or `target_decode_tok_s`, and a `tok_s` value;
  `MIN_E4B_QAT_PROVIDER_RATIO` controls the local/provider ratio floor unless
  an entry supplies `min_ratio`.
  The summary records comparisons under `provider_comparison` and fails the
  verdict when local QAT is below the provider floor.
  `E4B_QAT_REQUIRE_PROVIDER_METADATA=1` is enabled by default, so release
  baselines must also include `model`, `hardware`, `tokens`, `workload`,
  `measured_at`, and `source_url` to keep provider comparisons auditable.
  Use `zig/pkg/inference/scripts/gemma4_qat_provider_baselines.example.json` as
  the template for real external measurements. Validate the file before turning
  provider comparison into a required CI phase:

  ```sh
  python3 zig/pkg/inference/scripts/gemma4_qat_production_summary.py \
    --validate-provider-baselines-only \
    --provider-baseline /path/to/provider-baselines.json \
    --output /tmp/gemma4-provider-baseline-validation.json
  ```

  This standalone mode checks loading, supported metrics, and provenance fields;
  the full production gate still compares local throughput from that run against
  the provider ratios.
  OpenAI-compatible providers can be measured directly with the provider helper:

  ```sh
  PROVIDER_API_KEY=... \
  python3 zig/pkg/inference/scripts/gemma4_qat_provider_benchmark.py \
    --base-url https://provider.example/v1 \
    --api-key-env PROVIDER_API_KEY \
    --provider provider-name \
    --model google/gemma-4-E4B-it-qat-q4_0-gguf \
    --hardware "provider GPU or instance class" \
    --source-url https://provider.example/run-or-dashboard \
    --tokens 512 \
    --min-completion-tokens 512 \
    --repeats 2 \
    --warmup 1 \
    --baseline-stats avg,median,min \
    --output /tmp/gemma4-provider-baselines.json \
    --rows-tsv /tmp/gemma4-provider-baselines.tsv
  ```

  The generated JSON is ready for `E4B_QAT_PROVIDER_BASELINE_JSON`; the TSV
  preserves per-request timing rows for review. Production runs should keep
  `avg,median,min` so the final summary produces separate labeled comparison
  checks from one provider measurement.
  Streaming provider throughput can also be collected with
  `--stream --rate-source stream_decode --metric target_decode_tok_s`; this
  requests usage-bearing stream chunks by default so the token count remains
  provider-reported.
  The production gates can run the same collector inline and feed its generated
  JSON directly into the provider comparison verdict:

  ```sh
  PROVIDER_API_KEY=... \
  RUN_E4B_QAT_PROVIDER_BENCHMARK=required \
  RUN_E4B_QAT_PROVIDER_COMPARISON=required \
  E4B_QAT_PROVIDER_BASE_URL=https://provider.example/v1 \
  E4B_QAT_PROVIDER_API_KEY_ENV=PROVIDER_API_KEY \
  E4B_QAT_PROVIDER_NAME=provider-name \
  E4B_QAT_PROVIDER_HARDWARE="provider GPU or instance class" \
  E4B_QAT_PROVIDER_SOURCE_URL=https://provider.example/run-or-dashboard \
  E4B_QAT_PROVIDER_BASELINE_STATS=avg,median,min \
  scripts/gemma4_cuda_production_gate.sh
  ```

  For streamed provider comparisons, add
  `E4B_QAT_PROVIDER_STREAM=1`,
  `E4B_QAT_PROVIDER_RATE_SOURCE=stream_decode`, and usually
  `E4B_QAT_PROVIDER_METRIC=target_decode_tok_s`.
  For merge/release readiness, run the CUDA release wrapper:

  ```sh
  scripts/ci/gemma4-qat-cuda-release-gate.sh
  ```

  The wrapper requires long-context decode, Polar4 compressed-KV decode,
  resident warm, resident soak, resident backpressure, resident Q4_K
  comparison, the E4B QAT MTP safety matrix, strict MTP replay stability,
  512-token strict assistant replay, and the forced acceptance diagnostic
  matrix. It also re-checks `readiness.json` and
  `e4b_qat_production_summary.json` for the exact compressed-KV fast-path
  counters and MTP replay/acceptance artifacts used by the release gate. The
  matching GitHub Actions workflow is `Gemma4 QAT CUDA Release Gate`; it is manual
  because it requires a GPU runner, local model artifacts, and optional provider
  credentials. Use `provider_mode=baseline` with `provider_baseline_json` to
  validate a pre-collected provider artifact, or `provider_mode=benchmark` with
  provider inputs plus `PROVIDER_API_KEY` to collect a fresh OpenAI-compatible
  provider baseline. Do not make an external provider claim from the local gate
  alone.
  The local release-wrapper validation at
  `/tmp/gemma4-e4b-qat-release-wrapper-gate-20260625-r2` passed `readiness=ok`
  and the wrapper post-check with compressed-KV `36.726` tok/s, `500`
  persistent graph replays, `504` compressed-V reads, `288` compressed-V writes,
  `24` paged block-table uploads, `504` identity attention reads, `462` fast GQA
  launches, and zero compressed-KV write fallbacks. The same run reported
  long-context QAT `26.181` tok/s over 936 generated tokens, resident QAT warm
  `27.093` tok/s E2E, resident Q4_K warm `17.818` tok/s E2E, resident ratio
  `1.521x`, soak aggregate `29.613` tok/s, soak p95 `17390.3` ms, backpressure
  `2` accepted and `2` rejected requests with max reject latency `6.7` ms, no
  unsafe resident graph markers, and zero active MTP candidates.

  This adds `e4b_qat_provider_baselines.json`,
  `e4b_qat_provider_baselines.tsv`, `e4b_qat_provider_benchmark.log`, and
  `e4b_qat_provider_baseline_validation.json` to the gate output directory. The
  top-level gate also exposes these artifacts and the parsed validation result
  in `readiness.json` under `provider_benchmark`. In `auto` mode the provider
  benchmark skips when no provider base URL is configured.
  The current required-phase top-level run at
  `/tmp/gemma4-current-local-required-20260625-r3` required CUDA smoke, target
  QAT/Q4_K, long-context QAT, resident QAT, resident Q4_K, soak, and
  backpressure phases after merging latest `main`. It passed `readiness=ok`
  with `verdict=ok`, zero verdict failures, L4 `sm_89` metadata, target QAT
  `avg=16.982` tok/s, target Q4_K `avg=11.596` tok/s, target ratio `1.464x`,
  long-context QAT `15.847` tok/s over 936 generated tokens, resident QAT
  `avg=16.108` tok/s E2E, resident Q4_K `avg=12.758` tok/s E2E, resident ratio
  `1.263x`, soak aggregate `16.896` tok/s, soak p95 `15476.5` ms, and
  backpressure with one accepted and three HTTP 503 rejected requests in at
  most `8.1` ms.
  The serving QAT-vs-Q4_K ratio gate at
  `/tmp/gemma4-cuda-top-resident-q4k-ratio-20260625-r3` reported QAT resident
  warm requests at `17.000` and `17.013` tok/s E2E (`avg=17.007`) and Q4_K
  resident warm requests at `12.796` and `12.785` tok/s E2E (`avg=12.790`), for
  a serving ratio of `1.330x` against the `1.05x` floor.
  The top-level 512-token serving gate at
  `/tmp/gemma4-cuda-top-resident-q4k-ratio-512-20260625-r1` reported QAT
  resident warm requests at `16.563` and `16.384` tok/s E2E (`avg=16.474`) and
  Q4_K resident warm requests at `12.745` and `12.815` tok/s E2E
  (`avg=12.780`), for a serving ratio of `1.289x` against the `1.05x` floor
  with `graph_replays=1014` for both resident paths.
  After tightening the resident defaults, the top-level default-settings run at
  `/tmp/gemma4-cuda-top-resident-defaults-512-20260625-r1` omitted explicit
  token, replay, and warm-floor overrides and still ran 512-token resident
  requests. It reported QAT `avg=16.215` tok/s E2E, Q4_K `avg=12.768` tok/s
  E2E, `ratio=1.270x`, and `graph_replays=1014`. The package-local
  default-settings run at
  `/tmp/gemma4-cuda-pkg-resident-defaults-512-20260625-r1` likewise used
  512-token resident requests by default and reported QAT `avg=16.179` tok/s
  E2E, Q4_K `avg=12.762` tok/s E2E, `ratio=1.268x`, and
  `graph_replays=1014`.
  Both production gate variants fail before starting a resident server if the
  soak or backpressure probes are enabled without `RUN_E4B_QAT_RESIDENT`, or if
  backpressure is enabled without `E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS`.
  The resident server exposes the same admission state through `/metrics` using
  `antfly_inference_request_queue_capacity`,
  `antfly_inference_request_queue_depth`,
  `antfly_inference_request_queue_available`,
  `antfly_inference_request_queue_active_requests`,
  `antfly_inference_request_queue_rejections_total`, and
  `antfly_inference_request_queue_rejected_units_total`. The top-level
  metrics-backed backpressure run at
  `/tmp/gemma4-cuda-top-resident-backpressure-metrics-20260625-r1` reported two
  accepted and two rejected requests, `queue_rejections=2`,
  `queue_rejected_units=6`, queue depth `0`, and queue available `6` after the
  accepted requests completed.
  The package-local gate now also has post-warm soak and CLI-backed overload
  probes. `/tmp/gemma4-cuda-pkg-resident-soak-backpressure-20260625-r1` used
  `--max-concurrent-requests 6`, reported QAT warm `avg=16.505` tok/s E2E,
  completed the six-request concurrency-two soak at aggregate `16.966` tok/s
  with minimum request `8.444` tok/s E2E and `p95=30317.3` ms, then accepted
  two and rejected two overload requests with HTTP 503 in at most `3.5` ms. It
  kept graph replay coverage at `2544` against a `2496` soak floor and `3054`
  against a `1504` backpressure floor. The metrics-backed package-local
  backpressure run at
  `/tmp/gemma4-cuda-pkg-resident-backpressure-metrics-20260625-r1` reported QAT
  warm `avg=16.158` tok/s E2E, accepted two and rejected two requests,
  `queue_rejections=2`, `queue_rejected_units=6`, and post-run queue depth `0`.
  The top-level QAT resident soak gate at
  `/tmp/gemma4-cuda-top-resident-soak-20260625-r1` kept the preloaded QAT
  server alive after the default warm pass, then issued six 256-token HTTP
  requests with concurrency two. It reported aggregate `16.933` tok/s,
  per-request `avg=9.875` tok/s E2E with queue time included,
  `graph_replays=2544`, and `graph_floor=2496`, with no unsafe graph-capture
  markers in the server log.
  The stricter latency-gated soak run at
  `/tmp/gemma4-cuda-top-resident-soak-latency-20260625-r1` completed the same
  profile with QAT warm `avg=16.336` tok/s E2E, soak aggregate `16.934` tok/s,
  minimum request `8.430` tok/s E2E against the 8.0 tok/s floor,
  `p50=30146.0` ms, `p95=30366.4` ms against the 35000 ms ceiling,
  `p99=30366.4` ms, `graph_replays=2544`, and `graph_floor=2496`.
  The longer 12-request soak at
  `/tmp/gemma4-cuda-top-resident-soak-12req-20260625-r1` kept the same
  concurrency-two, 256-token request shape and passed the same latency gates. It
  reported QAT warm `avg=16.154` tok/s E2E, soak aggregate `16.972` tok/s,
  minimum request `8.477` tok/s E2E, `p50=30176.4` ms, `p95=30200.7` ms,
  `p99=30200.7` ms, `graph_replays=4074`, and `graph_floor=3984`.
  The unconstrained concurrency-four probe at
  `/tmp/gemma4-cuda-top-resident-soak-c4-20260625-r1` kept aggregate throughput
  stable at `16.962` tok/s, but failed the per-request latency gate because
  queued requests stretched to `p95=105720.2` ms and individual request rates
  fell as low as `2.421` tok/s E2E. That is a production backpressure signal,
  not a decode kernel regression. The CLI-backed capacity gate at
  `/tmp/gemma4-cuda-top-resident-backpressure-cli-20260625-r2` started
  `antfly-inference run` with `--max-concurrent-requests 6`; with four
  concurrent 256-token requests it accepted two, rejected two with HTTP 503 in
  at most `5.2` ms, completed the accepted requests at `avg=12.854` tok/s E2E,
  and kept graph replay coverage at `1524` against a `1504` floor.
  The serving path attaches device KV storage to the decode state, and CUDA graph
  replay slots are invalidated when a new per-request KV device hook is
  provisioned so cross-request replay cannot target freed KV buffers.
- CUDA uses the current device-side Gemma4 fast paths for Q4_K Q/K/V
  projection, Q4_K embedding lookup, fused head-norm+RoPE, device KV
  read/write, dense GQA attention, MTP masked argmax, and optional paged
  TurboQuant KV.
- CUDA TurboQuant KV status, measurements, and validation steps live in
  `CUDA.md` under "Gemma4 And TurboQuant KV Status". Gemma4 CUDA defaults remain
  `f32` KV for exactness. `--cache-dtype polar4` is the current
  production-candidate opt-in compressed-K/compressed-V path; `--cache-dtype
  turbo3` is resident and functional but still experimental.

The E4B QAT GGUF pull command is:

```sh
zig/pkg/inference/zig-out/bin/antfly-inference pull \
  hf:google/gemma-4-E4B-it-qat-q4_0-gguf:gguf:Q4_0 \
  --models-dir .models \
  --tasks generate,read \
  --capabilities text,image,audio \
  --projector auto
```

The most direct E4B QAT CUDA smoke is:

```sh
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/google/gemma-4-E4B-it-qat-q4_0-gguf \
  "Write one sentence about ants." \
  --backend cuda \
  --host-budget-mb 8000 \
  --backend-budget-mb 12000 \
  --combined-budget-mb 18000 \
  --kv-budget-mb 512 \
  --scratch-budget-mb 1024 \
  --max-tokens 8 \
  --temperature 0 \
  --raw-prompt \
  --no-chat-template \
  --print-token-ids \
  --print-timing
```

The most direct user-facing E2B smoke is:

```sh
zig/pkg/inference/zig-out/bin/antfly-inference generate \
  .models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  "Give a one sentence summary of Korean history." \
  --backend cuda \
  --max-tokens 128 \
  --print-timing \
  --print-token-count
```

If running from `zig/pkg/inference/zig-out/bin`, use the absolute model path
instead of `.models/...`; CUDA.md includes that copy-paste form.

Once the model artifacts are available, run the CUDA validation ladder in this
order:

```sh
zig build -Dcuda=true

zig/pkg/inference/zig-out/bin/antfly-inference cuda-info --smoke

zig/pkg/inference/zig-out/bin/antfly-inference cuda-info \
  --gemma4-parity /path/to/gemma4-12b-target.gguf

zig/pkg/inference/zig-out/bin/antfly-inference compare \
  /path/to/gemma4-12b-target /path/to/gemma4-12b-target \
  "Write one sentence about ants." \
  --runtime-parity \
  --backend cuda \
  --top-k 8 \
  --no-chat-template

zig/pkg/inference/zig-out/bin/antfly-inference generate \
  /path/to/gemma4-12b-target \
  "Write one sentence about ants." \
  --backend cuda \
  --max-tokens 16 \
  --temperature 0 \
  --print-token-ids \
  --print-timing

zig/pkg/inference/zig-out/bin/antfly-inference generate \
  /path/to/gemma4-12b-target \
  "Write one sentence about ants." \
  --backend cuda \
  --draft-model /path/to/gemma4-assistant \
  --speculative-k 2 \
  --max-tokens 16 \
  --temperature 0 \
  --debug-mtp \
  --print-token-ids \
  --print-timing
```

Useful CUDA/MTP isolation flags:

- `ANTFLY_CUDA_DISABLE_GEMMA4_MTP_DEVICE=1`: use the host clustered-output
  fallback instead of the CUDA MTP masked-argmax kernel.
- `ANTFLY_CUDA_ENABLE_Q4K_DECODE_FAST=1`: enable the experimental Q4_K tile8
  decode path.
- `ANTFLY_CUDA_DISABLE_HEAD_NORM_ROPE_FUSION=1`: disable fused
  head-norm+RoPE.
- `ANTFLY_CUDA_ENABLE_ADD_MUL_SCALAR_FUSION=1` and
  `ANTFLY_CUDA_ENABLE_RMSNORM_ADD_MUL_SCALAR_FUSION=1`: enable experimental
  output-scale fusions.
- `ANTFLY_GEMMA4_MTP_ALLOW_UNSHARED_TARGET=1`: force experimental MTP against
  targets missing shared-KV metadata.

Follow-up smoke after adding the earlier pre-final-RMSNorm target activation
path:

```text
bundle: pkg/inference/.debug/metal-command-20260505-162324
validation: MTL_DEBUG_LAYER=1, MTL_SHADER_VALIDATION=0
exit_code=0
diagnostic-reports: none
speculative: rounds=3 drafted=5 matched=0 rejected=5 accepted=3 corrections=3 bonus=0
```

This confirms the path is Metal-stable under API validation for the local repro,
but it did not improve acceptance against the quantized GGUF target. Later
source comparison with the extracted PyTorch drafter moved the handoff back to
final-RMSNorm hidden states.

Official safetensors target status:

```text
bundle: pkg/inference/.debug/metal-command-20260505-170606
validation: MTL_DEBUG_LAYER=1, MTL_SHADER_VALIDATION=0
command: antfly inference generate ~/.antfly/inference/models/google/gemma-4-E2B-it hi --backend metal --max-tokens 1
exit_code=0
diagnostic-reports: none
token_ids: 239863
timing_ms: load_model=2591 generate=1330 total=3926
```

Memory note: the first Metal-only safetensors attempt preserved BF16 in the
tensor store but then expanded rank-2 BF16 weights into cached f32 host slices
and duplicated BF16 bytes for the decoder runtime. That explains the observed
multi-10GB footprint. The Metal cache now keeps BF16 rank-2 dense weights as
mmap-backed native bytes, only materializing f32 for vectors and fallback paths
that actually require host math. A traced smoke peaked around 2.6GB physical
footprint instead of the earlier 30GB+ behavior.

Official target + official assistant status:

```text
bundle: pkg/inference/.debug/metal-command-20260506-161752
validation: MTL_DEBUG_LAYER=1, MTL_SHADER_VALIDATION=0
command: antfly inference generate ~/.antfly/inference/models/google/gemma-4-E2B-it hi --backend auto --draft-model ~/.antfly/inference/models/google/gemma-4-E2B-it-assistant --speculative-k 2 --max-tokens 4
exit_code=0
diagnostic-reports: none
token_ids: 10979 236888 2088 740
speculative: rounds=1 drafted=2 matched=2 rejected=0 accepted=3 corrections=0 bonus=1
```

This confirms the full Metal/safetensors target + assistant runtime runs without
Metal diagnostic reports and can accept the assistant's drafted span on the
short anchor prompt. Mixed GGUF target plus official safetensors assistant is
also supported for local smoke coverage, but acceptance-rate conclusions should
prefer official target+assistant pairs and the proper Gemma 4 chat template.

The repo smoke wrapper is
`scripts/test_metal_gemma4_assistant_speculative.sh`. It uses `--backend auto`
by default so the normal backend selector can pick Metal when available; set
`ANTFLY_INFERENCE_GEMMA4_ASSISTANT_BACKEND=metal` to force Metal for crash/debug runs.
The official target currently needs the wrapper's default
`ANTFLY_INFERENCE_GEMMA4_ASSISTANT_HOST_BUDGET_MB=12288` and
`ANTFLY_INFERENCE_GEMMA4_ASSISTANT_COMBINED_BUDGET_MB=17408` preflight budgets.

### Metal GGUF Runtime Status

The Metal GGUF path now routes explicit compiled generation through graph
execution instead of a separate live whole-model shortcut:

```sh
antfly inference generate ~/.antfly/inference/models/ggml-org/gemma-4-e2b-it-gguf \
  hi \
  --backend metal \
  --mode compiled \
  --compiled-target whole-model \
  --max-tokens 4 \
  --print-token-ids \
  --print-timing
```

Under the graph route, Metal uses the resident decoder runtime directly for
whole-model prefill/decode. Pure greedy generation can return the selected token
without downloading full logits, so the short anchor prompt now reports
`prefill cached_logits=false greedy_token=true` and decode-side
`greedy_fallback=0`.

Validator smoke on 2026-05-07:

```text
bundle: pkg/inference/.debug/metal-command-20260507-142101
validation: MTL_DEBUG_LAYER=1, MTL_SHADER_VALIDATION=0
command: antfly inference generate ~/.antfly/inference/models/ggml-org/gemma-4-e2b-it-gguf hi --backend metal --mode compiled --compiled-target whole-model --max-tokens 4
exit_code=0
diagnostic-reports: none
token_ids: 10979 236888 2088 740
generate_timing_ms: prefill=875 decode=149 total=1024
metal_executor_ms: prefill_direct_family=871 greedy_calls=3 greedy_direct=149 greedy_fallback=0
metal_runtime_encoders: compute=21 blit=0 planned_scopes=35 planned_barriers=457
```

The generic quant runtime surface has been separated from the Q8_0-specific
kernel implementation. Public runtime scratch/setup exports, debug env vars,
and timing labels use `quant` names. The existing Q8_0 fused kernels remain
internal fast paths; adding Q4/K-quants should extend the quant-format dispatch
behind those generic entrypoints instead of creating more public `q80` API.
The direct whole-layer block planner now follows that shape too: it asks for a
direct quantized block format and currently selects the Q8_0 implementation
only when every participating linear slot is Q8_0. Unsupported or mixed formats
fall back through the staged generic quant linear path.
The staged FFN side can still use existing fused Metal kernels for non-Q8
families: homogeneous Q4_K, Q6_K, I2_S, TL1/TL2, Q8_0, plus mixed
Q4_K/Q5_K-down, Q4_K/Q6_K-down, and Q4_0/Q8_0-down layouts. The planner now
marks those combinations as direct-eligible instead of logging them as mixed or
unsupported before the runtime has a chance to use the fused path.
The device-resident FFN residual path follows the same generic shape: Q8_0
keeps the monolithic fused kernel, while non-Q8 formats that have staged pair
and single-stage Metal kernels compose gate/up, activation, multiply,
optional RMS norms, down projection, and residual add without leaving device
memory. That removes the old Q8-only boundary without adding format-specific
public APIs.

A standalone prepared-tail greedy shortcut that directly encoded
`rms_norm + quantized lm_head + argmax` outside a planned frame was tested and
backed out after a 2026-05-07 SoC watchdog reset under Metal API validation
(`pkg/inference/.debug/metal-command-20260507-214829`, panic
`/Library/Logs/DiagnosticReports/Retired/panic-base-2026-05-07-214909.panic`).
Keep that path on the materialized-logits argmax route until the command
lifetime/barrier issue is isolated. After reverting, minimal API-validation
smoke `pkg/inference/.debug/metal-command-20260507-215452` completed with
`token_ids: 10979` and no new diagnostic reports.

The native Metal GGUF route must not depend on MLX availability when Antfly inference is
built with both backends enabled. A later 4-token compiled whole-model smoke was
failing before model execution with `MlxMetalUnavailable`; the long-term fix is
to keep `.metal` sessions on the native Metal provider/stream path and reserve
MLX streams/providers for the `.mlx` backend. The repaired smoke
`pkg/inference/.debug/metal-command-20260507-222950` passed API validation with
`token_ids: 10979 236888 2088 740`, `prefill=157ms`, `decode=149ms`,
`total=1006ms`, and no diagnostic reports.

### Phase 3: Performance Work

- Tune `speculative_k`; start at 4 and compare against 2, 6, and 8.
- Add a heuristic schedule that increases draft length after full acceptance
  and decreases it after rejection.
- Benchmark dense Gemma 4 separately from the 26B MoE model. MoE verification
  can lose speedup at batch size 1 because drafted tokens may route to
  different experts.
- Prefer batched server benchmarks for MoE models, where expert reuse is more
  likely.
- Keep the compiled graph route as the single public Metal whole-model path.
  Avoid reintroducing a second live-executor CLI route; graph execution should
  own attachment, fallback, greedy-token shortcuts, and future prefill/decode
  scheduling.
- Fill the remaining GGML-shaped gaps behind generic quant entrypoints:
  extend the monolithic whole-block kernels beyond Q8_0, add any missing
  format-specific fused matvec kernels, and keep the layer/block planner
  independent of the physical quant format.

## Correctness Rules

- The target model always owns final token acceptance.
- Sampling, repetition penalties, and grammar masks must be applied from the
  target logits during verification.
- Rejected draft suffixes must be rolled back from KV state.
- Correction and bonus tokens must be materialized into target KV before the
  next round. Gemma 4 MTP assistants have no drafter KV; they keep only the
  target-prediction activation needed to seed the next draft round.
- MTP must fall back to standard decoding if the assistant is missing,
  incompatible, or slower for the current backend.

## Open Questions

- What is the exact public Transformers implementation for
  `Gemma4AssistantForCausalLM`? The tagged public Gemma 4 files do not yet show
  it, so implementation should follow confirmed artifacts plus LiteRT-LM
  behavior until upstream source is visible.
- Do assistant checkpoints expose enough metadata to validate exact target
  compatibility, or do we need a local compatibility table?
- Should the experimental inverse `masked_embedding.token_ordering` environment
  override be removed now that MLX-VLM confirms centroid-to-token ordering?
- Should the clustered embedder move into a cached backend-native path? The
  baseline implementation currently materializes the ordering on host per draft
  step for correctness/debuggability.
- Should speculative scheduling be per-request, per-model, or learned from
  recent acceptance-rate telemetry?
