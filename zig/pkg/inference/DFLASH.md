# DFlash

DFlash is an inference-native speculative decoding mode for block-diffusion
draft checkpoints. The first Antfly milestone targets Gemma4 on Metal with
z-lab Gemma4 DFlash draft checkpoints as the reference artifact format.

## Command

```sh
antfly inference generate /path/to/google/gemma-4-E2B-it \
  "Explain block diffusion drafting briefly." \
  --backend metal \
  --draft-model /path/to/z-lab/gemma-4-E2B-it-dflash \
  --speculative-method dflash \
  --speculative-k 16 \
  --print-timing
```

`--speculative-method` accepts:

- `ar`: existing autoregressive draft-model speculative decoding.
- `dflash`: Gemma4 DFlash block drafting plus target verification.

For `dflash`, `--speculative-k` is interpreted as the DFlash block size. If it
is omitted, Antfly uses the draft checkpoint block size when declared, otherwise
it defaults to 16.

The HTTP generate API accepts the same extension:

```json
{
  "model": "google/gemma-4-E2B-it",
  "messages": [{ "role": "user", "content": "hi" }],
  "draft_model": "z-lab/gemma-4-E2B-it-dflash",
  "speculative_method": "dflash",
  "speculative_k": 16
}
```

## Draft Metadata

Antfly recognizes DFlash draft checkpoints from `config.json` metadata such as:

- `architectures` or `model_type` containing `DFlash`.
- `speculative_method: "dflash"`.
- `dflash_config`, `dflash`, or root fields including `block_size`,
  `target_hidden_size`, `draft_layer_count`, `target_feature_layers`,
  `feature_bank_capacity`, `cross_context`, `mask_token_id`,
  `shared_embeddings`, and `shared_lm_head`.

The loader validates that:

- The target is Gemma4-compatible.
- The draft declares DFlash metadata.
- Target and draft tokenizer vocabulary/special tokens match.
- The draft target hidden width matches the target when declared.
- Feature-bank capacity and cross-context window fit the native device-resident
  ring limit.
- MoE and sliding-window DFlash drafts are rejected for this milestone.

## Reference Implementations

Antfly does not depend on external DFlash runtimes, but BeeLlama.cpp is a useful
design reference for the device path: it keeps target hidden states in a bounded
per-layer ring and lets the drafter attend over a configurable recent
cross-context window. Antfly mirrors that shape as DFlash metadata
(`feature_bank_capacity`, default 4096, and `cross_context`, default 1024), but
keeps the V1 contract stricter on macOS: no CPU ring fallback in production
DFlash.

BeeLlama.cpp also points to follow-up work that is intentionally out of Device
V1: adaptive draft-depth control, sampled DFlash verification, branch/tree
verification, and multimodal flat DFlash.

## Current Implementation

The native DFlash branch is separate from AR speculative decoding and Gemma4 MTP.
Device V1 is fail-closed: target hidden features, draft feature fusion, injected
draft K/V, draft block logits, and target verification must remain resident on
Metal. Scalar token IDs and counters may cross to host, but full hidden, logits,
or K/V tensors must not be downloaded during DFlash rounds.

The current native milestone owns a Metal-resident `DFlashFeatureBank`, captures
selected target layers during target forward, selects the first DFlash token with
device argmax, and validates/fuses target features against draft projection
weights. For z-lab Gemma4 drafts, the preferred fusion path is `fc.weight`
followed by `hidden_norm.weight`, with older feature-projection aliases retained
only as compatibility fallbacks.

After fusion, the drafter materializes one device-resident injected K/V context
per DFlash draft layer from `layers.N.self_attn.{k,v}_proj.weight` and
`layers.N.self_attn.k_norm.weight`, prepares the masked/noise embedding block on
device, and runs the z-lab-style draft-layer sequence through native
`gqaCrossAttentionFull`. That primitive has both a native reference path and a
purpose-built Metal kernel for non-causal GQA cross-attention with independent
query and K/V lengths. The DFlash path does not reuse the causal decoder
attention kernel.

Use the checkpoint inspector to confirm a local draft schema without loading
tensor payloads:

```bash
zig/pkg/inference/scripts/inspect_dflash_checkpoint.py /path/to/dflash-draft
```

`--print-timing` emits DFlash counters:

- Draft block time.
- Target feature fusion time.
- K/V injection time.
- Verification time.
- Feature extraction count.
- Device feature capture count.
- Host fallback count.
- Full tensor download bytes.
- Maximum accepted block length.

The old host-observable bootstrap path is still available only for parity
debugging by setting `ANTFLY_DFLASH_DEBUG_HOST_FALLBACK=1`. That mode is not
mergeable production behavior and increments `host_fallbacks` plus full tensor
download byte counters.

## Limitations

- Gemma4 text targets only.
- Metal/native path only; CUDA and Qwen3.5 are follow-ups.
- Deterministic decoding only: `temperature=0` with no top-p/top-k/min-p or
  repetition/frequency/presence penalties.
- No multimodal prompts.
- No grammar-constrained decoding or JSON schema response format.
- DFlash drafts that require SWA, masks/biases in draft cross-attention, sampled
  verification, or unsupported checkpoint schema variants still fail closed.
- Target verification by device-selected token ids is wired for deterministic
  DFlash rounds once draft tokens are available.
- No vLLM/SGLang delegation.
- DFlash training is out of scope; Antfly consumes trained draft checkpoints.

## Local Smoke

```sh
cd zig/pkg/inference
zig build test-metal-gemma4-dflash-speculative -Dmetal=true -Dmlx=false
```

Useful environment overrides:

- `ANTFLY_INFERENCE_GEMMA4_TARGET_MODEL`
- `ANTFLY_INFERENCE_GEMMA4_DFLASH_DRAFT_MODEL`
- `ANTFLY_INFERENCE_GEMMA4_DFLASH_PROMPT`
- `ANTFLY_INFERENCE_GEMMA4_DFLASH_EXPECTED_TOKEN_IDS`
