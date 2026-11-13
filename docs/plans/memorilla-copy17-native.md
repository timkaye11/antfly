# Memorilla copy17 native serving plan

Status: milestone 1 reached 2026-11-12 (offline parity tool reproduces the Python candidate); milestone 2 (managed serving) not started. Port of the encoder-LoRA research candidate
`tokens-copy17-gate3-1024` (schema `antfly.memorilla.encoder-lora-research.v1`)
to the Zig CUDA inference runtime. The released Memorilla path stays untouched;
copy17 arrives as a separate, separately qualified variant, following item 4 of
"Remaining integration work" in `memorilla-native-memory.md`.

## 1. What the candidate computes at inference

Reference: `zig/pkg/inference/scripts/memorilla/encoder_lora_eval.py` with
`--snap-memory all --copy-contiguous-runs --copy-complete-words`; helpers in
`encoder_lora.py`. Python runs the per-case loop under `inference_mode` and
`autocast(cuda, bf16)`; the copy head and the snapper run in FP32 with autocast
off. Weights in all three artifacts are stored FP32.

### 1.1 Artifacts (candidate directory)

| file | tensors | notes |
|---|---|---|
| `reader.safetensors` | 37 F32 | upstream `MemoryModule` state, packed `in_proj_weight [7680,2560]` / `in_proj_bias [7680]` per attention |
| `encoder-lora.safetensors` | 144 F32 | `layers.{0..35}.self_attn.{q_proj,v_proj}.{A,B}`; A `[8,2560]`, B `[4096,8]` (q) / `[1024,8]` (v); scale alpha/rank = 16/8 = 2.0 |
| `copy-head.safetensors` | 3 F32 | `query.weight [4096,4096]` (no bias), `gate.weight [1,4096]`, `gate.bias [1]` |
| `protocol.json`, `candidate.json`, `base-proofs.json` | | `memory_slots` 16, `document_features` "tokens-dual", `dual_layer` 24, `document_space_prefix` true, `document_self_context` true, `encoder_special_rows` true, `copy_head_temperature` 0.05, `reader_loops` 1, `entity_tags` "none" |

Reader geometry is the release geometry: d 2560, out 4096, 16 memories, 8 heads
(head_dim 320), 2 cross-attention blocks, LayerNorm eps 1e-5, erf GELU.

### 1.2 Encoder input (Qwen3-Embedding-4B tokenizer, BF16 encoder + LoRA)

1. Space prefix: each document that does not start with " " gets one leading
   space. The question gets no prefix and no instruction.
2. Self-context: for document D, `single = encode(D)` (no specials);
   `doubled = D + D` (joined directly, the second copy's leading space is the
   separator); require `encode(doubled)[-len(single):] == single`, else retry
   with `single = encode(" " + D)`, else error. `copy_ids = cat(single_i)`
   (length T = sum n_i) are the copy sources.
3. Batch `[D1+D1, D2+D2, ..., question]`, right padded. The tokenizer appends
   `<|endoftext|>` (151643) to every row (no BOS); pad id is also 151643 with
   mask 0. Padded length must be <= 128.
4. One causal forward with hidden states. `final` = output after the final
   RMSNorm; `layer` = `hidden_states[24]` (raw output of layer index 23, not
   normalized). Position ids `arange(L)`.
5. LoRA on every `q_proj` and `v_proj`: `y = base(x) + ((x A^T) B^T) * 2.0`,
   added before q_norm/k_norm and RoPE. Under autocast these run in BF16.
6. Row selection: for each document keep the last `n_i + 1` rows (second copy's
   tokens, then its EOS state); keep the question row whole. Apply to both
   `final` and `layer`.

### 1.3 tokens-dual features

- Question query `[1,2560]` FP32: last active row of `final` for the question
  (its EOS state), L2-normalized (eps 1e-12), then `feature_channel`:
  FP32 -> FP16 (RNE, subnormals kept) -> BF16 (RNE) -> FP32.
- Document rows per layer X in (final, layer24): concatenate every document's
  `n_i + 1` selected rows in order -> `[M, 2560]`, row-wise L2 normalize in
  FP32, `feature_channel`.
- Reader input `docs = cat([docs_final, docs_layer24], dim=1)` -> `[1, 2M, 2560]`
  FP32: all final-layer rows first, then all layer-24 rows, same order. Key
  padding mask all False. Question tokens are never document rows.

### 1.4 Reader forward (upstream MemoryModule, eval mode, BF16 autocast)

```
question = q[None]                              # [1,1,2560]
slots = memory_queries                          # [1,16,2560] FP32
conditioned = MHA_query_cond(Q=slots, K=V=question)   # single key => weight 1.0
                                                # == out_proj(W_v q + b_v) for every slot
slots = LN_query_cond_norm(slots + conditioned) # post-norm
docs = LN_doc_norm(docs)
for l in 0,1:
    slots = slots + MHA_l(Q=LN_cross_norm_l(slots), K=V=docs)
    slots = slots + Lin2_l(GELU_erf(Lin1_l(LN_ffn_norm_l(slots))))   # 2560->10240->2560
out = memory_projection(LN_layer_norm(slots))   # [1,16,4096]
```

MHA details (torch slow path): packed weights split q = rows 0:2560,
k = 2560:5120, v = 5120:7680 (same for bias); 8 heads of 320; scores
`(q * 320^-0.5) @ k^T`; softmax in FP32, probabilities rounded to BF16 before
`p @ v`; GEMMs in BF16 with BF16 outputs; LayerNorm in FP32; residual sums in
FP32; GELU on the BF16 tensor. Output memories `[16,4096]` are BF16.

### 1.5 Copy head (FP32, autocast off)

Sources `E = decoder.embed_tokens.weight[copy_ids].float()` `[T,4096]`: the
encoder-tokenizer ids of the space-prefixed documents (single copy), EOS rows
excluded. Encoder and decoder ids are assumed equal (asserted on one dev case).

```
m = memories.float()                            # [16,4096]
uq = normalize(m @ Wq^T); ut = normalize(E)
logits = (uq @ ut^T) / 0.05                     # [16,T]
probs = softmax(logits); copied = probs @ E
g = sigmoid(m @ gate.w^T + gate.b)              # [16]
soft = g*copied + (1-g)*m
```

### 1.6 Decoding rules, in order

a. `copy_with_contiguous_runs(min_gate=0.5)`: `log_probs = log_softmax(logits)`;
   `copying[s] = g[s] >= 0.5`; runs = maximal consecutive copying slots
   `(begin,end)`. For each run of length L <= T, `scores[s] = sum_k
   log_probs[begin+k, s+k]` for s in 0..T-L (runs longer than T keep the soft
   output and yield no span). Process runs by descending max score (stable);
   within a run take positions in descending score order, choose the first
   finite start whose L positions are all unclaimed, else the first finite start
   (overlap allowed), else skip. Claim positions; set
   `output[begin+k] = g*E[chosen+k] + (1-g)*m`. Return output, spans sorted.
b. Snap all: `ids[s] = argmax_v cos(output[s], table[v])` over the full decoder
   input-embedding table `[151936,4096]` (untrained tail rows included), FP32
   matmul over normalized rows. No blank handling.
c. `complete_copied_words(pad_id = 220 (" "), overflow truncate)`, only when
   spans exist. `continues(id)`: decoded text does not start with " ", body =
   text.strip() non-empty, every char alnum (Unicode) or '-'. Walk spans in
   order: emit snapped ids before `begin`, then extend `[start, start+L)` back
   while `continues(src[first])` and forward while `continues(src[last])`, emit
   those SOURCE ids (overriding snapped ids), cursor = end; emit the rest; drop
   trailing pads while > 16, then truncate to 16, then pad with 220.
   Final memories = `embed.weight[ids]` BF16.

### 1.7 Decoder prompt and generation (Qwen3-8B)

`<|memory|>` is an added special token, id 151669. Messages: system =
`SYSTEM + "\nMemory" + "<|memory|>"*16 + "Memory"`, user = question.strip();
chat template with `add_generation_prompt`, `enable_thinking=False`:

```
<|im_start|>system\n{SYSTEM}\nMemory<|memory|>x16Memory<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

SYSTEM = "Answer using the supplied memory. Return only the complete exact name
or code, preserving case, hyphens and leading zeros. If the requested fact is
missing, return UNKNOWN."

Injection is by embedding rows: `emb[pos] = bf16(e_mem + bf16(v - e_mem))`
where `e_mem` is the table row of 151669 (nonzero). This differs from the exact
table row in about 22% of elements (max abs 0.002); the existing Zig `compose`
already mirrors the same delta trick. Prefill BF16 in Python (F32 in Zig),
greedy argmax, stop on 151645 or 151643, max 24 new tokens. Answer =
decode(skip specials).strip(); exact match; "UNKNOWN" literal for missing facts.

Serving inputs are only the documents, the question and the frozen artifacts.
Priors and corpus data are evaluation-only.

## 2. What the Zig runtime has today

- Reader forward `src/pipelines/memorilla.zig forward()` is generic over row
  count (`documents [B*N, d]`, mask per row) and already matches the math above
  on split q/k/v weights, F32 strict, with trace hooks for the verifier.
  Question conditioning, two blocks, FFN, final projection all present.
- Everything above it is pooled-only: request `document_embeddings` 1..64 x
  2560 unit vectors, `memorilla_service.validateInputs`, `storedValue`
  FP16->BF16 channel, `encoder.encodeQuestion` (last-token pooled).
- Bundle loader `src/models/memorilla.zig`: schema `antfly.memorilla.bundle.v1`,
  exactly `17 + 16L` tensors, execution strings pinned (`encoder_pooling`
  "last-token-l2-normalized"), release pins to five checkpoints.
  `memorilla_runtime.Owner.open` refuses non-release bundles and requires
  `parameter_bytes == 776_495_104`, compute capability 12.1, strict F32.
- Decoder injection `src/pipelines/memorilla_generation.zig`: recipe with
  marker id 151669 (= vocab size), 16 rows, F32 `[16,4096]` MemoryRows,
  `compose` applies `original + (imported - original)`, paged F32 KV, greedy.
  Default system prompt differs from copy17's SYSTEM (recipe field).
- No per-token encoder state export, no intermediate-layer tap, no CUDA LoRA
  linear (native CPU `linearLoRA` only), no attention-probability op (the copy
  head needs probabilities, `crossAttention` returns weighted values only), no
  snapping, no span rules, no tokenizer-level word-boundary logic.
- Verifier `src/bench/memorilla_verify.zig` compares `6 + 8L` boundaries per
  case against `scripts/memorilla/reference.py` fixtures (pooled inputs).
- Config: `inference.memorilla.profiles[] {name, encoder, decoder,
  memory_module_path, max_documents, max_new_tokens}`
  (`zig/lib/platform/src/inference_settings.zig`). OpenAPI in
  `specs/openapi/inference/{config,api}.yaml`; `make generate` regenerates.
- Tests: add new files to the `test {}` block in `src/inference.zig` or a
  dedicated root like `src/test_memorilla_generation.zig` with a `build.zig`
  step; run `zig build test-memorilla-source -- --test-filter "<name>"` with
  `ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-cache/global" ZIG_LOCAL_CACHE_DIR="$PWD/.zig-cache"`.

### 2.1 Encoder specifics (from the CUDA path survey)

- Encoder profile load: `session_factory.zig:1945` with geometry pin (36 layers,
  hidden 2560, 32/8 heads, head_dim 128) and byte-exact lock on the two F32
  shards (`memorilla_v1_base_models.json`, `verifyArtifacts`); 398 resident F32
  weights required. Any modified shard fails closed, and adapter files next to
  the model are rejected (`optional_paths`).
- Run path `archRunImpl .gpt` -> `gpt_arch.hiddenForward` returns one host F32
  tensor `last_hidden_state [B,S,2560]` (after final RMSNorm); causal, no mask,
  right padding is harmless for active rows. Pooling happens afterwards on the
  host in `pipelines/embedding.zig pool3D` (`.last`) + `l2Normalize`.
- No intermediate-layer tap on this path. Minimal change: in
  `gpt.zig hiddenForwardFromEmbeddingsResidentWithOverrides` layer loop after
  `hidden = new_hidden;` (around :4331), copy `hidden` to host when
  `layer + 1 == tap_layer` (HF `hidden_states[k]` = output of 0-based layer
  k-1, before the final norm). Plumb through `hiddenForward` and
  `archRunImpl` as a second output tensor; `embedding.zig` reads only
  `outputs[0]`.
- LoRA: CUDA has no `linearLoRA`; the only load-time merge is Jina v5 specific
  (`gpu_hosted_store.zig JinaLoraAdapter.mergeIntoLoadedWeight`). Route: merge
  offline, `W_f32 + 2.0 * B @ A`, store F32, same tensor count and bytes.
- Diagnostic session without the v1 lock:
  `createQwen3EmbeddingCudaReferenceSession` (`session_factory.zig:1700`,
  strict F32, requires `model_manifest.json` without `memorilla_profile`);
  template `src/bench/memorilla_encoder_verify.zig`. Decoder equivalent:
  `createQwen3CudaReferenceSession` as used by `memorilla_qwen_verify.zig`.
- Tokenizer: `zig/lib/tokenizer` `HfTokenizer` loads `tokenizer.json`;
  `decodeToken(id, at_start=false)` returns " Depot"-style pieces (special ids
  decode to ""); encoder and decoder share vocab/merges; EOS 151643 is appended
  by the pipeline (`ensureTrailingEos`), not the tokenizer.

## 3. Port design

Two milestones. Milestone 1 proves the numerics end to end with an offline
verify tool against Python fixtures; milestone 2 wires the variant into the
managed service. Precision: Zig runs strict F32 everywhere (the qualified
decoder profile); Python ran BF16 autocast, so parity is measured at
boundaries with tolerances and, decisively, on snapped ids, completed ids and
answer text.

### 3.1 Offline export (Python, `scripts/memorilla/encoder_lora_export.py`)

1. `merge-encoder`: read the F32 managed encoder shards
   (`.benchmark-assets/memorilla/managed-f32/encoder`), add `2.0 * B @ A` to
   every `model.layers.N.self_attn.{q,v}_proj.weight` in FP32, write a complete
   HF directory (F32 shards, index, config, tokenizer files) plus a
   `merge.json` receipt (candidate sha, adapter sha, per-tensor max delta).
2. `export-reader`: `reader.safetensors` -> split q/k/v bundle via
   `exporter.split_projections`, non-release manifest, `execution.encoder_pooling`
   = `"per-token-dual-layer-l2-normalized-v1"` (new value; Zig accepts it only
   for non-release bundles), `config.json` = release geometry.
3. `copy-head` is used as-is (`copy-head.safetensors`, 3 F32 tensors).
4. `reference`: run the Python pipeline (the eval's own functions) on a case
   list and write `reference.json` + `reference.safetensors`:
   - json: `schema` `antfly.memorilla.encoder-lora-pipeline-reference.v1`,
     candidate/adapter/reader/copy-head sha256, `dual_layer`, `cases[]` with
     `name, documents[], question, expected_answer, copy_ids[], spans[[begin,end,start]],
     snapped_ids[16], completed_ids[16], prompt_ids[], output_ids[], answer`.
   - safetensors (F32 unless noted): `<case>.docs [2M,2560]`, `<case>.query [2560]`,
     `<case>.memories [16,4096]` (BF16 upcast), `<case>.copy_output [16,4096]`,
     `<case>.gate [16]`, `<case>.logits [16,T]`.

### 3.2 Zig pieces

- `gpt.zig` layer tap (see 2.1) exposed through the session run as a second
  output, `hidden_states_<k>`.
- `src/pipelines/memorilla_token_features.zig`: space prefix, self-context
  doubling with the fallback rule, EOS append, right padding, row selection
  (last `n_i+1` rows per document, whole question row), per-row L2 normalize,
  feature channel (reuse `memorilla_encoder.storedValue`), `docs [2M,2560]`
  + `query [2560]` + `copy_ids`. Pure host code over the two returned tensors.
- `src/pipelines/memorilla_copy.zig` (host F32): copy head forward, contiguous
  run selection, snap (device dot products via `linear` against the embedding
  table plus host table norms computed once), `complete_copied_words` with the
  tokenizer, all mirroring section 1.5 and 1.6 with unit tests ported from
  `test_encoder_lora.py`.
- Reader: existing `pipelines/memorilla.forward` on the exported bundle with
  `allow_synthetic`; `models/memorilla.zig validateExecution` accepts the new
  `encoder_pooling` value for non-release bundles only.
- Decoder: `memorilla_generation.prepareText` with `Recipe.system_prompt` set
  to copy17's SYSTEM, memory rows = embedding rows of `completed_ids` (F32, via
  `embeddingLookup`), existing `compose` delta trick, greedy, 24 new tokens.
- `src/bench/memorilla_lora_verify.zig` + `build-memorilla-lora-verify`: loads
  merged encoder (reference session), reader bundle, copy head, decoder
  (reference session), runs every fixture case, compares each boundary with
  tolerances and reports id/answer agreement as JSONL.

### 3.3 Milestone 2 (managed serving)

New `Profile` variant with its own base-model lock, catalog bundle, manifest
and `executionIdentity`; `inference.memorilla.profiles[]` gains the candidate
directory; latent requests carry `document_texts` (per-token path needs text,
not pooled vectors); OpenAPI + `make generate`; admission for per-token rows;
docs in `docs/design/inference/memorilla-serving.md`. Not started until
milestone 1 reproduces the Python answers.

## 4. Milestone 1 results (2026-11-12)

Tool: `zig build build-memorilla-lora-verify -Dcuda=true -Doptimize=ReleaseFast`,
binary `zig-out/bin/antfly-memorilla-lora-verify --candidate <copy17 dir>
--reader-bundle <lora-reader-bundle> --encoder <lora-merged-encoder> --decoder
<managed-f32/decoder> --reference <lora-ref-dev|lora-ref-fresh> [--output
summary.json] [--max-cases N] [--skip-decoder] [--reference-features]
[--reference-memories]`. Fixtures from `encoder_lora_export.py reference`.

| set | cases | copy ids = Python | spans = | snapped = | completed = | prompt = | answer = Python | Zig correct | Python correct |
|---|---|---|---|---|---|---|---|---|---|
| dev | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| fresh (8 groups, seed 20261110) | 80 | 80 | 79 | 79 | 79 | 80 | 79 | 78 | 79 |

Stage numerics (Zig strict F32 vs Python BF16 autocast): document rows
relative L2 <= 2.6%, min row cosine 0.999; reader memories relative L2 2-6%
typically, worst 17% (cosine 0.86) on one held-out-phrasing owner case that
still decoded identically; copy gate within 1e-4 except where a gate sits at
the 0.5 threshold.

The two fresh divergences:
- `ood-0002-code-a`: one gate crosses 0.5 in the other direction, so Zig's
  site run ends one slot earlier (" Depot" slot becomes a pad). Both answers
  are the correct code.
- `ood-0002-code-b`: all sixteen memory ids and the prompt are identical, yet
  the F32 decoder answers UNKNOWN where the BF16 decoder returned the code.
  The record is a cut-site record (" code Ly5-009275 at Mozeba" + pads), the
  class already known to sit on the decoder's acceptance boundary.

Mean per-case timings on GB10, strict F32, offline tool (no batching, host
readbacks included): encoder 620 ms (3 rows x ~140 tokens through the generic
CUDA Qwen3 path), feature assembly 0.7 ms, reader 19 ms, copy head 9 ms
(host), snap 26 ms (GPU dot products + 10 MB readback), word completion
<0.1 ms, decoder 1.3-1.8 s (prompt ~100 tokens, up to 24 new tokens, F32 KV).
The Python BF16 harness decoded the same answers in ~0.65 s; the qualified F32
profile is the cost of strict parity, and a BF16 serving profile would need
its own qualification.

Unit tests: `zig build test-memorilla-source -Dtest-filter='memorilla copy'
-Dtest-filter='memorilla token features' -Dtest-filter='gpt hidden-state tap'
-Dtest-filter='memorilla execution contract'` (models/memorilla.zig,
memorilla_copy.zig and memorilla_token_features.zig are imported from the
`src/inference.zig` test block).

## 5. Milestone 2 design: managed serving of the research candidate

Principles: the released path keeps every pin unchanged; the research path is
a second, explicitly configured profile kind that fails closed on its own
pins; nothing is promoted to "qualified". Interfaces below are fixed so the
pieces can be built in parallel.

### 5.1 Artifacts (produced by `encoder_lora_export.py`)

Merged encoder directory (`merge-encoder`, sidecars refreshable with
`--sidecars-only`):
- `model_manifest.json` = the catalog encoder manifest with
  `"memorilla_profile":"qwen3_encoder_lora_research_f32_v1"` (byte-exact
  canonical form is `memorilla_catalog.research_encoder_manifest`).
- `memorilla_lock.json`: `{"schema":"antfly.memorilla.research-encoder-lock.v1",
  "candidate_sha256":"<sha256 of candidate.json>", "base":{"model":"Qwen/Qwen3-Embedding-4B",
  "revision":"5cf2132a..."}, "files":[{"path","size_bytes","sha256"}...]}` listing
  every shard, `model.safetensors.index.json`, `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`, `merge.json`,
  `1_Pooling/config.json`, `modules.json`, `config_sentence_transformers.json`,
  `generation_config.json` (not `model_manifest.json`, not itself).
- `.antfly-download-complete.json` managed receipt (version 1) listing every
  regular file in the directory with size and sha256, including
  `model_manifest.json`, `memorilla_lock.json` and `merge.json`, so the
  directory is a managed publication (`publication_generation` = sha256 of
  the receipt).

Research reader bundle directory (`export-reader`): `manifest.json`
(`release:false`, `execution.encoder_pooling` per-token, `files` pins
`config.json`, `memory.safetensors`, `copy-head.safetensors`, `candidate.json`,
`protocol.json`; `source.candidate_sha256` = sha256 of `candidate.json`),
plus those five files. The Zig side reads `system_prompt`,
`copy_head_temperature` and `memory_slots` from `protocol.json`.

Cross-check at load: encoder lock `candidate_sha256` == bundle
`source.candidate_sha256` == sha256(`candidate.json` in the bundle).

### 5.2 Qualification profile (Zig)

- `memorilla_qualification.Profile` gains `qwen3_encoder_lora_research_f32_v1`
  (role encoder). `requireRoleRecipe`, `residencyPlan`, `role` handle it; the
  geometry pin, 398-weight / 16,087,097,344-byte residency checks and the
  `archRunAdmitted` gating apply to both encoder tags (role based).
- Lock source: v1 keeps `@embedFile`; the research tag reads
  `<dir>/memorilla_lock.json` (<= 1 MiB), validates the schema, and
  `verifyArtifacts` hashes its `files`, requires `model_manifest.json` ==
  `research_encoder_manifest`, and `artifactGeneration` fences the lock file,
  `merge.json` and the receipt. `sourceBytes`/host residency come from the
  lock. `optional_paths` stays forbidden.
- `executionIdentity(.qwen3_encoder_lora_research_f32_v1)` is fixed per tag
  (tag, canonical research manifest, arithmetic, sm_121, question recipe,
  marker `"research-lock-in-directory"`); the per-candidate pin is surfaced
  as `candidate_sha256` in the response, not folded into the identity, because
  clients never produce vectors for this path.
- `memorilla_encoder.Producer.profile` is the actual tag;
  `requireProducer(producer, publication, expected)`.

### 5.3 Research reader runtime (Zig, `src/models/memorilla_research_runtime.zig`)

Mirror of `memorilla_runtime.zig` for `ReaderCache`: `parameter_bytes` =
reader 776_495_104 + copy head 67_125_252 = 843_620_356 host-resident copy
head (device-resident reader weights only: backend_weight_bytes 776_495_104);
`weightPlan`, `requireLimits`, `requireAdmission`, `Owner{bundle, compute,
copy_head: memorilla_copy.CopyHead, candidate_sha256: [64]u8, protocol:
{system_prompt: []u8, copy_head_temperature: f32, memory_slots: usize},
execution_mutex, open(a, io, dir, admission, limits, control),
computeBackendWithBudget, releaseRequestResources, deinit}`. `open` requires
`release == false`, the per-token pooling, the five pinned files, the
candidate cross-check, SM 12.1, strict F32, resident bytes == reader bytes.
`memorilla_reader_cache.ReaderCache` becomes `pub` with an identity file list
taken from the Owner (`Owner.identity_files`); `pub const ResearchCache =
ReaderCache(research.Owner)`. `LoadedModel` (decoder) carries both caches.

### 5.4 Per-token encoder entry (Zig, `memorilla_encoder.zig`)

`encodeTokenFeatures(a, handle, documents: []const []const u8, question,
limits, control) !TokenFeatures` with `TokenFeatures{ features:
memorilla_token_features.Features, question_tokens: usize, producer:
Producer, lease, deinit }`. Same guards as `encodeImpl` (profile tag
research, publication, identity before/after, embedding asset lease, lane),
CPU lease sized for `(docs+1) * 128 * 2560 * 4 * 2` plus features, session
run through the admitted path with explicit amounts and the
`hidden_states_tap` input (layer 24), then `memorilla_token_features.prepare`
/ `features`. Documents <= 64, each <= 8192 bytes, batch rows <= 128 tokens.

### 5.5 Service (Zig, `src/server/memorilla_research_service.zig`)

`generate(a, io, manager, profile, paths, question, documents: []const []const u8,
identity, max_new_tokens, limits, request_limits, control, options)
!OwnedResult` where `OwnedResult` extends the latent one with
`candidate_sha256`. Flow = `memorilla_service.generateImpl` with: identity ==
research encoder identity; `prepareText` recipe `system_prompt` from the
bundle protocol; `encodeTokenFeatures`; `ResearchCache.acquire`; reader
forward with `num_documents = rows`, `max_document_rows = 16384`,
`backend_scratch = 128 MiB + rows * 2560 * 4 * 6`; copy head on the host
(sources = decoder embedding rows fetched on the decoder backend under its
execution scope, released before generation); snap via `linearNoBias`
against the resident embedding table on the decoder backend (16 x vocab
readback, table norms cached per loaded decoder); `completeCopiedWords` with
the decoder tokenizer (pad id = tokenizer encode of " " must equal 220);
memory rows = embedding rows of the completed ids uploaded to the reader
backend; `decode.generate` unchanged. Extra decoder-budget scratch reserved
for the copy/snap transients (32 MiB).

### 5.6 Settings, request, response (Zig + OpenAPI + docs)

- `MemorillaProfile.reader_kind: enum { released, encoder_lora_research } =
  .released` (JSON `"released"` / `"encoder_lora_research"`); for
  `encoder_lora_research` the `encoder` is the merged directory and
  `memory_module_path` the research bundle.
- Request: `document_embeddings` optional (required for `released`,
  rejected for research); research latent requires `document_texts`
  (1..64, same text limits as source_verified); `source_verified` with a
  research profile returns `UnsupportedMemorillaProfile` (409) for now.
  `encoder_execution_identity` must equal the research encoder identity.
- Response: `context_source` gains `supplied_documents`; new optional
  `candidate_sha256`; `memory_tokens` stays 16; `input_documents` = texts.
- Handler: item count = texts for research; `memorillaFailureResponse` maps
  `SyntheticMemorillaBundleNotAllowed`, `UnsupportedMemorillaProfile`,
  `MemorillaCandidateMismatch`, `MemorillaEncoderBoundExceeded` (413).
- `make generate` after `config.yaml` / `api.yaml`; docs in
  `docs/design/inference/memorilla-serving.md` ("research profile" section).

### 5.7 Verification

Unit tests per module; HTTP tests with `/missing` paths for the new
validation branches; offline `antfly-memorilla-lora-serve-verify` (template:
`memorilla_composed_verify.zig`) building a `ModelManager`, the research
profile and running the fixture cases through `memorilla_research_service`,
comparing answers with `lora-ref-{dev,fresh}` (target: same 143/144).

## 6. Milestone 2 results (2026-11-12)

The research profile serves end to end through the managed path
(`memorilla_research_service`, invoked by the HTTP handler for
`reader_kind: "encoder_lora_research"`), verified offline with
`antfly-memorilla-lora-serve-verify` over a real `ModelManager`:

| set | cases | answer = Python | Zig correct | Python correct | admission refunded |
|---|---|---|---|---|---|
| dev | 64 | 64 | 64 | 64 | 64 |
| fresh | 80 | 79 | 78 | 79 | 80 |

The single divergence is the same cut-site record as in milestone 1
(F32 decoder answers UNKNOWN; identical memory ids). Warm request latency
(median, F32 strict, GB10): 2.0 s total = 0.62 s per-token encoding + 1.35 s
reader, copy head, snapping and decoding. Cold loads: encoder 52 s, decoder
95 s, reader bundle 4 s. Operator cap: the shared unified admission limit is
75% of the process cap; two F32 models plus one request need about 73 GiB.

Shipped in this milestone: research profile tag with in-directory lock and
fixed execution identity; research bundle pins (copy head, candidate,
protocol) with candidate cross-check against the encoder lock; research
reader runtime and cache; per-token managed encoder entry; text-input
service; `reader_kind` setting; request/response contract (`document_texts`
in latent mode, `context_source: supplied_documents`, `candidate_sha256`);
HTTP handler branch and error mapping; OpenAPI + Zig/Python/Rust generated
code; serving docs section; Python sidecar export (`merge-encoder`
`--sidecars-only`, pinned bundle extras).

Open items: Go and TypeScript SDK regeneration (`make generate` needs `go`
and Node 24 on the build host); `source_verified` mode for research
profiles (texts are already present, the verifier can consume them); a BF16
serving profile to recover decoder latency would need its own qualification.

## 7. Wide challenge through the managed path (2026-11-12)

Fixtures: `encoder_lora_export.py fresh-cases --groups 16` for seeds
20261110, 7331, 4242 (160 cases each) and `reference` for each; the Python
reference reproduces the research evaluator's answers on all 480 cases.
Managed service via `antfly-memorilla-lora-serve-verify`; comparison with
`.benchmark-results/memorilla/encoder-lora-next/compare_wide.py`.

| source | correct | wrong | abstain |
|---|---|---|---|
| research evaluator (Python BF16) | 464 / 480 (96.7%) | 2 | 14 |
| Python fixture reference | 464 / 480 | 2 | 14 |
| managed Zig service (F32) | 462 / 480 (96.2%) | 2 | 16 |

477 of 480 answers are identical. The three differences, all on records whose
copy gates or decoder decision sit at a numerical threshold:
- seed 20261110 `ood-0002-code-b`: cut-site code record; F32 decoder answers
  UNKNOWN (Python: correct code).
- seed 20261110 `ood-0008-owner`: gates 0.587 / 0.479 next to the 0.5 run
  threshold; the F32 path resolves the runs differently and the decoder echoes
  the memory frame ("Memory key holder with the brass key for ... Memory"),
  a wrong answer where Python was correct.
- seed 7331 `ood-0005-owner-b`: F32 abstains where Python returned a wrong name.

Per-request latency over the 480 managed requests: median 1.97 s, p90 2.52 s
(including the three cold loads). Per seed: 151 / 153 / 158 versus research
153 / 153 / 158.

Stage-level confirmation (`antfly-memorilla-lora-verify` over seed 20261110,
160 cases): copy sources 160/160 identical, spans/snapped/completed ids
157/160, prompt ids 160/160, answers 158/160. In `ood-0008-owner` the gate of
the " Z" slot (0.479 in Python) crosses 0.5 in F32 (max gate difference 0.09),
so the name run [2,7) and the site run [8,11) merge into one nine-slot run
whose best diagonal starts inside the sentence ("with the brass key for ..."),
the name leaves the record, and the decoder echoes the frame. Mitigations, in
order of cost: an answer-format guard turning implausible decoder output into
an abstention; a decode rule that refuses to merge runs across a slot whose
gate lies within a margin of the threshold (needs its own evaluation); or
training with a gate margin loss.

### 7.1 Answer-format guard (2026-11-12)

`memorilla_research_service.guardAnswer` runs on the decoder text after
`decode.generate` in the managed research path. Trimmed text equal to `UNKNOWN`
is accepted. Otherwise the text is rejected when it is empty, longer than 64
bytes, more than six whitespace-separated words, contains a control character
or newline, contains the frame literal `Memory` as a standalone word or the
`<|memory|>` marker, or contains a byte outside letters, digits, space, `-`,
`.` and `'`. A rejection replaces the text with `UNKNOWN` and sets
`finish_reason` to `abstained`; the token ids are kept. The HTTP response
reports the outcome in `answer_guard` (`accepted` / `rejected`).

Rationale: the task's answers are short exact facts (a one-to-three-word name,
a `Xx0-00dddd` code, or `UNKNOWN`) and the frozen decoder has no format
constraint in direct-answer mode. The only garbled wrong answer observed in the
480-case wide challenge (`ood-0008-owner`, section 7) was a frame echo,
"Memory key holder with the brass key for Zehisi Depot  \nMemory", which the
guard converts into an abstention. The rules are general (no corpus-specific
regex) so a correct name or code is never rejected; the remaining wrong answers
in the table above are well-formed names or codes and are not affected. This
is serving policy for the research profile only, unit-tested in the service;
a GPU re-run of the wide challenge should record 462 correct / 1 wrong / 17
abstain for the managed path if nothing else changes.

### 7.2 Bridge-margin merge rule: evaluated and rejected (2026-11-12)

Rule (`memorilla_copy.contiguousRunsWithMargin`): a copying slot whose gate is
within `margin` above the 0.5 threshold and that lies between confident slots
of the same run stops copying, splitting the run. Evaluated with
`antfly-memorilla-lora-verify --copy-bridge-margins 0,0.05,0.1,0.15` on wide
seed 20261110 (160 cases, F32):

| margin | correct | wrong | answers = Python |
|---|---|---|---|
| 0 (current) | 151 | 1 | 158 |
| 0.05 | 151 | 2 | 157 |
| 0.1 | 151 | 3 | 156 |
| 0.15 | 151 | 4 | 155 |

No case became correct at any margin. The target case `ood-0008-owner` turns
from the frame echo into "Gemebo Bunokaoka Zehisi Depot" (still wrong), and
three other cases turn from abstentions into wrong answers of the shape
"<name><suffix> <site> Depot": splitting a run at an uncertain slot leaves the
site run adjacent to the name, and the frozen decoder reads the concatenation
as the answer. The sweep was stopped after the first seed; the rule stays
available behind the flag (margin 0 in the service) and is not adopted. The
answer-format guard (7.1) remains the mitigation for garbled output; the
remaining fence-sitting records are a training-side matter (gate margin loss
or more held-out-phrasing data).

### 7.3 Wide challenge, managed path with the answer guard (2026-11-12)

Rerun of section 7 with the guard of 7.1 active (`lora-serve-wide-<seed>-guard.jsonl`):

| source | correct | wrong | abstain |
|---|---|---|---|
| research evaluator (Python BF16) | 464 / 480 (96.7%) | 2 | 14 |
| managed Zig service, F32, guarded | 462 / 480 (96.2%) | 1 | 17 |

The guard fired exactly once in 480 requests, on the frame-echo answer of
`ood-0008-owner`, and rejected nothing else. 477 of 480 answers remain
identical to Python. Latency unchanged (median 1.97 s, p90 2.53 s).

## 8. Qwen hybrid memory: general soft recall plus exact records (started 2026-11-13)

Goal: one memory that restates passages like Memorilla and recalls names and
codes exactly like copy17. Observation: the released reader consumes pooled
document vectors (general, soft), copy17 consumes per-token states (exact,
snapped); pooled features cannot pinpoint codes (section 2 of the research
notes), so the hybrid composes two readers rather than widening one.

Design (no training for the first experiment):
- Rows 1-16: the released Memorilla reader (stage3) over pooled
  Qwen3-Embedding-4B vectors, injected as soft rows exactly as the released
  profile does.
- Rows 17-32: copy17's snapped record (16 real-token rows: contiguous runs,
  word completion), or only its copying spans with the remainder dropped.
- Decoder: stock Qwen3-8B, prompt = system text + "Memory" + 32 placeholder
  slots + "Memory", question in the user turn; system text to be chosen by
  measurement (released "helpful assistant" prompt vs copy17's exact-answer
  prompt vs a merged one).

Evaluation (Python first, `scripts/memorilla/qwen_hybrid_eval.py`): three
arms (soft only, record only, hybrid) on (a) the depot dev and fresh sets
(exact match), (b) squad_v2 / drop validation subsets (EM, F1), (c) an enwiki
restatement sample (ROUGE-L against the passage). If the hybrid keeps copy17's
exact accuracy and the released reader's free-form accuracy, the follow-ups
are: a short decoder-in-the-loop fine-tune of the soft slots on Stage 2 data
with the 32-slot frame if the extra rows perturb them, then the Zig service
change (mixed soft/token rows through the existing injection path).

### 7.4 Fallback trigger signals (2026-11-13, CPU analysis over the 480 wide cases)

Per-slot gates and snap cosines from the Python fixtures joined with the
guarded managed-service outcomes (462 correct, 1 wrong, 17 abstained):

| trigger | flagged | wrong caught | abstentions caught | correct flagged |
|---|---|---|---|---|
| min copy-slot snap cosine < 0.85 | 10 | 0/1 | 5/17 | 5/462 |
| min copy-slot snap cosine < 0.80 | 5 | 0/1 | 4/17 | 1/462 |
| min snap cosine (all slots) < 0.80 | 45 | 1/1 | 6/17 | 38/462 |
| gate within 0.1 of 0.5 | 12 | 0/1 | 5/17 | 7/462 |

Reading: the one wrong answer (`ood-0007-owner`, a fence-sitting owner record)
is only caught by the all-slot cosine at the price of flagging 8% of correct
answers; the fence-case abstentions are caught cheaply by the copy-slot
cosine (5 of 17 at one false positive per caught case); the remaining
abstentions are cut-site records whose signals are clean (gates saturated,
cosines 1.0) and whose only signal is the UNKNOWN itself. Practical routing:
UNKNOWN -> full-context fallback (catches all 17 abstentions after the fact),
copy-slot cosine < 0.85 -> fallback before decoding (catches the fence cases
early), optional all-slot cosine < 0.80 for a precision-first deployment that
accepts 8% extra slow-path traffic to remove the last wrong answer. Both
cosines are available in the service (snap scores) and should be surfaced in
the response alongside `answer_guard`.

### 8.1 First hybrid results (2026-11-13, no training; `qwen_hybrid_eval.py`)

| set | soft (released stage3) | record (copy17) | hybrid (16 soft + 16 record rows, merged prompt) |
|---|---|---|---|
| depot dev, exact (64) | 0 | 64 | 51 (62 case-insensitive) |
| depot fresh, exact (80) | 0 | 79 | 69 (75 case-insensitive) |
| squad_v2 val 200, EM / F1 | 0.230 / 0.377 | 0.050 / 0.091 (73% abstain) | 0.285 / 0.449 |

The composition is additive on free-form QA: the hybrid beats both parts on
squad because the record rows often hold the exact span under depot-style
labels (" key holder in Medieval Latin, 9th century") that the record arm's
prompt turns into UNKNOWN, while the soft rows and prompt keep the decoder
answering. On exact facts the record text is complete and correct in every
hybrid miss; the loss is surface form (case flips of code prefixes, a dropped
prefix or first name), attributed to the merged system prompt diluting
"preserving case" and to the soft rows beside the record. Abstention is intact
(16/16 UNKNOWN on both depot sets, zero false abstentions; the hybrid even
removed copy17's one false abstention). Pooled soft memory alone cannot carry
codes (0/144 on depot). Pending: drop and enwiki arms; then prompt (`exact`),
row order (record first) and `--record-rows spans` variants; then the short
soft-slot fine-tune in the 32-row frame.

### 8.2 Hybrid variants (2026-11-13, no training)

| variant | depot dev exact (ci) | depot fresh exact (ci) | squad EM / F1 | drop F1_any |
|---|---|---|---|---|
| soft only | 0 (0) | 0 (0) | 0.230 / 0.377 | 0.246 |
| record only (copy17) | 64 (64) | 79 (79) | 0.050 / 0.091 (73% abstain) | 0.009 |
| hybrid, merged prompt, soft first (base) | 51 (62) | 69 (75) | 0.285 / 0.449 | 0.238 |
| hybrid, exact prompt | 51 (62) | 66 (74) | 0.275 / 0.441 | 0.215 |
| hybrid, record gated (clean span, cosine >= 0.85) | 51 (62) | 68 (74) | 0.290 / 0.453 | 0.233 |
| hybrid, record rows first | 36 (40) | 52 (54) | 0.285 / 0.436 | 0.244 |
| hybrid, span-only record rows | 20 (31) | 25 (29) | 0.295 / 0.458 | 0.250 |

enwiki ROUGE-L F1 (50 passages): soft 0.068, record 0.061, hybrid 0.086; every
arm emits one sentence (recall about 5%).

Conclusions: the prompt wording is not the cause of the exact-recall loss
(the exact prompt reproduces the misses one for one); the soft rows' presence
is, and putting them after the record is worse; the record's label tokens are
needed for the decoder to read it (span-only collapses depot). Gating the
record on a clean copied span is neutral on depot and best on squad, but the
0.85 threshold fires on 92% of squad and 87% of drop cases. Passage
restatement is absent for the stage3 reader under stock Qwen3-8B in this frame;
the stage1 checkpoint is the test of whether it exists at all.

### 8.3 Answer canonicalization to the copied span (CPU re-scoring of the same outputs)

Every hybrid depot miss had the exact text in the record. Re-scoring the base
and gated hybrid outputs with a post-decode rule, "if the answer matches a
record substring case-insensitively, or is the tail of a record word
sequence, return the record's exact text", recovers the case flips with no
regressions: dev 51 -> 62/64, fresh 69 -> 75/80 (gated 68 -> 74), 0 correct
answers broken. The copy slots are the verbatim source and the decoder is
the selector, so letting the record spell the answer is the principled fix;
to be implemented as a decode rule in the evaluator and the service, with the
answer guard, and checked on squad/drop for side effects.

### 8.4 Production composition and the restatement check (2026-11-13 05:40Z)

Gated hybrid with answer canonicalization (soft rows first, full record rows,
merged prompt, record gate at cosine 0.85, canonicalize to the record span):

| set | copy17 alone | soft alone | gated + canonicalized hybrid |
|---|---|---|---|
| depot dev exact (64) | 64 | 0 | 63 (12 answers canonicalized) |
| depot fresh exact (80) | 79 | 0 | 78 (10 canonicalized) |
| squad EM / F1 (200) | 0.050 / 0.091 | 0.230 / 0.377 | 0.290 / 0.453 |
| drop F1_any (200) | 0.009 | 0.246 | 0.223 |

Exact recall is back to within one case per set of copy17 while free-form QA
gains +0.08 F1 over the released reader; drop (arithmetic) loses 0.02. This
is the recommended no-training composition.

Restatement: the released STAGE1 checkpoint restates passages (enwiki 50,
soft arm): ROUGE-L F1 0.237 / P 0.248 / R 0.234, 141 generated tokens on
average, paraphrased gist of the passage; stage3 gives 0.068 and one sentence.
The capability exists in the released readers and is lost through the QA
stages, which validates the Gemma Stage 1 objective and suggests testing the
hybrid with stage1 (or stage2) soft slots instead of stage3.

### 8.5 Soft half by released checkpoint (2026-11-13 08:20Z, gated + canonicalized hybrid)

| soft half | depot dev | depot fresh | squad EM / F1 | drop F1_any | enwiki hybrid ROUGE-L (tokens) | enwiki soft-only ROUGE-L (tokens) |
|---|---|---|---|---|---|---|
| stage1 | 45/64 (7/16 missing abstained) | 45/80 | 0.000 / 0.038 | 0.064 | 0.230 (140) | 0.237 (141) |
| stage2 | 55/64 (11/16) | 67/80 | 0.300 / 0.448 | 0.255 | 0.154 (67) | 0.166 |
| stage3 | 63/64 (16/16) | 78/80 | 0.290 / 0.453 | 0.223 | 0.086 (24, base hybrid) | 0.068 (18) |

The released readers are task-tuned: stage1 restates but makes the decoder
ignore the question (QA collapses, abstention fails), stage3 answers but does
not restate, stage2 sits between on every axis. Production hybrid keeps the
stage3 soft half (exact + QA); passage restatement is a separate mode served
by the stage1 reader when a request asks for it, and a single soft half that
does both needs training (the decoder-in-the-loop tune of the soft slots, or
the Gemma Stage 1 -> Stage 2 path with restatement kept in the mixture).

### 8.6 Source citations on the research serving path (2026-11-13)

The research response now returns, for every span the copy head copied into
the 16-slot record, the document it came from and its exact UTF-8 byte range
in the caller's text: `citations: [{document_index, start, end, text,
supports_answer, match}]` and `citations_unavailable` (serving doc, "Research
profile"). Mechanism:

- Source map (`memorilla_token_features.sourceMap`): each document's
  single-copy ids are decoded piece by piece (`tokenBytes`) and checked byte
  for byte against the encoded text (the caller's text after the added space,
  plus one more space on the `encode(" " + D)` fallback). A piece's cumulative
  range, shifted back by the added spaces and clamped at 0, is its range in
  the caller's text. If the pieces do not reproduce the text (NFC composing a
  decomposed accent, a special token), there is no map: citations are
  unavailable and the request still succeeds.
- Spans (`memorilla_copy.completeCopiedWords`): `Completed.spans` reports, per
  copy-head span, the source range emitted after word extension and its
  record slots, clipped by the overflow truncation.
- Citations (`memorilla_copy.buildCitations`): each emitted range is split
  per document, because a contiguous run over the concatenated sources can
  cross a document boundary. Regions of one document that touch or overlap
  are merged, widened to whole UTF-8 characters and trimmed of whitespace.
  `supports_answer` and `match` (`exact`, `case_insensitive`, `none`) compare
  the published answer after the guard; UNKNOWN never counts as support.

CPU replay over the Python reference receipts (real Qwen3 tokenizer, the
reference copy-head spans, snapped ids and answers; word completion
reproduces every reference record exactly; every citation equals its
document bytes):

| set | cases | mapped | citations | correct non-UNKNOWN answers with a supporting citation | supported but wrong |
|---|---|---|---|---|---|
| dev | 64 | 64 | 112 | 48/48 | 0 |
| fresh | 80 | 80 | 143 | 63/63 | 0 |
| wide 20261110 | 160 | 160 | 283 | 121/121 | 0 |
| wide 4242 | 160 | 160 | 288 | 126/126 | 0 |
| wide 7331 | 160 | 160 | 287 | 121/121 | 0 |

Example (dev-0000-code-a, answer `Kg9-009369`): `{document_index 0, start
95, end 105, text "Kg9-009369", supports_answer true, match exact}` from
record slots 1..10, and `{document_index 0, start 23, end 35, text "Pipofo
Depot", supports_answer false}` from slots 11..15.

Managed service (`antfly-memorilla-lora-serve-verify`, first 8 dev cases,
shared GPU): 8/8 answers correct and equal to the reference, 14 citations,
0 mismatches against `document_texts[i][start..end]`, 0 unavailable, 6/6
correct non-UNKNOWN answers with a supporting citation (the two UNKNOWN cases
cite only the site name), admission refunded on every case.

### 8.7 Hybrid serving in Zig: one encoder, run-time encoder LoRA (2026-11-13)

The production composition of 8.4 (stage3 soft half, record gate at 0.85,
merged prompt, soft rows first, canonicalization with the three-character
tail minimum) is served by a third profile kind, `reader_kind:
hybrid_research` (serving doc, "Hybrid research profile"). Design:

- One resident encoder: the stock managed `qwen3_encoder_f32_v1`. The pooled
  pass runs it unchanged; the per-token record pass applies copy17's encoder
  LoRA to the q/v projections at run time (`gpt.EncoderLora`, `y = W x +
  (alpha/rank) B (A x)` on the raw projection output before q/k norm and
  RoPE). The adapters travel as two run inputs of the tapped hidden-state run
  (`encoder_lora` F32 `[36, 2 * 8 * 2560 + (4096 + 1024) * 8]`,
  `encoder_lora_scale` F32 `[1]`), are uploaded for the run and freed with it,
  so the resident weights and their residency pins are untouched. Residency
  drops from about 67 GB (two F32 encoders) to about 51 GB.
- Artifacts: `memory_module_path` = the hybrid bundle (`export-reader
  --include-encoder-lora`): the research bundle plus `encoder-lora.safetensors`
  pinned in `manifest.json` and by `candidate.json` `encoder_lora_sha256`;
  `protocol.json` `rank`/`alpha`/`target_modules`. `released_memory_module_path`
  = the stage3 bundle. The merged research encoder is not used.
  `encoder_execution_identity` is the stock encoder's.
- Service (`src/server/memorilla_hybrid_service.zig`): adapter and prompt pins
  before any model load; a tokenizer preflight on the stock encoder (question
  tokens, pooled texts, the record batch rows); one admission covering the
  larger decoder plan (32 rows, merged prompt) or the smaller (16 rows,
  helpful prompt), copy/snap transients, the released reader over up to 64
  pooled rows and the research reader over the exact per-token rows; pooled
  pass; record pass with adapters (padded rows up to the evaluator's 1024
  tokens, at most 16384 positions); released reader (`Cache`) borrowed and
  retired first, then the research reader (`ResearchCache`), copy head, runs,
  snapping (with the top-1 cosines), word completion; the gate on the
  four-decimal cosines as the evaluator records them; on a closed gate the
  32-row reservation is swapped in place for the 16-row plan; generation;
  canonicalization (gate open), frame-echo guard, citations (gate open only).
- Deviations from `qwen_hybrid_eval.py`: F32 throughout (the evaluator is
  BF16, adapters in BF16); texts above the 256-token pooled limit are chunked
  sentence-greedily at 128 tokens (the evaluator pools any length; no
  verification text reaches the limit); a 16384-position cap on the record
  batch; ASCII-only case folding in canonicalization; the frame-echo guard
  (empty, control characters, `<|memory|>`, standalone `Memory`) is serving
  policy the evaluator lacks.
- Tests: native-backend equality of run-time LoRA and the merged q/v weights
  (two-layer Qwen3, two padded rows, relative error <= 1e-5, plain forward
  bit-identical), adapter file packing and pin checks, canonicalization on the
  evaluator's 19 cases, the gate on the evaluator's cases, guard, chunker,
  admission maximum, settings, request shape and HTTP validation branches.
- Verification: `antfly-memorilla-lora-hybrid-verify` reads the Python
  `gated-canon` rows (`hybrid-{depot-dev,depot-fresh,squad}.jsonl`) with case
  files for the documents (depot `lora-ref-{dev,fresh}.json`; squad exported
  to `.benchmark-results/memorilla/qwen-hybrid-serve/squad-cases.json` from
  the same 200 validation rows) and reports per set agreement with the Python
  answer and gate, exact (depot) or EM/F1 (squad), record-text agreement,
  citation checks and latency.
