# Latent memory for the EmbeddingGemma 2 / Gemma 4 pair

Status: step 1 (compatibility probes) done 2026-11-12. Companion to
`memorilla-copy17-native.md`, whose reader, copy head, snapping and serving
path this plan reuses with a different encoder/decoder pair.

## 1. Why the recipe transfers

Snapping makes the memory sixteen real decoder tokens, so the decoder is never
trained and any frozen instruct model that reads short text records works.
What is trained: the reader (two cross-attention blocks), the copy head and an
encoder LoRA, with the lexical token-CE recipe against the decoder's embedding
table. The copy head requires encoder and decoder token ids to coincide.

## 2. Step 1 probes (2026-11-12)

**Tokenizer sharing.** `google/embeddinggemma-2` `tokenizer.json` and the Gemma 4
tokenizer (from the local `gemma-4-E4B-it-qat-q4_0-unquantized-assistant`
directory) have identical vocabularies: 262,144 token-to-id entries, no
differences either way; both sample records encode to the same ids. The copy
head and word completion carry over unchanged. (The two files differ in
post-processor bytes, not in vocabulary.)

**Readability.** Gemma 4 E4B (QAT q4_0 GGUF, Zig CUDA runtime,
`antfly inference generate --disable-thinking`, copy17 system prompt with the
record inlined as `Memory<record>     Memory` ahead of the question):

| record | question | answer |
|---|---|---|
| ` code Kg9-009369 at Mozeba Depot` | code for Mozeba Depot | `Kg9-009369` |
| ` key holder Komode Tuloco at Kopene Depot` | key holder at Kopene Depot | `Komode Tuloco` |
| ` code Kg9-009369 at Mozeba Depot` | key holder at Mozeba Depot | `UNKNOWN` |
| ` code Ly5-009275 at Mozeba` (cut site) | code for Mozeba Depot | `Ly5-009275` |
| ` code Zx5-005023 at Nerute Depot` | code for Kopene Depot | `UNKNOWN` |
| ` key holder Gemebo Bunoka is Z Zehisi Depot` (garbled) | key holder at Zehisi Depot | `Gemebo Bunoka` |

Six for six, including the cut-site and garbled records that the F32 Qwen3-8B
decoder got wrong. Because snapped slots are real tokens, the Gemma decoder
can be driven with a plain text prompt; no embedding-row injection is needed
for the text path.

**Zig plumbing survey (what exists, what is new).**
- EmbeddingGemma 2: hidden 512 with a learned 512->768 projection, 24 layers,
  bidirectional (no self-context doubling needed), 8192-token budget, media
  (image up to 280 soft tokens, video 140/frame, audio ~1020) spliced as rows
  of the same sequence. CUDA path returns pooled vectors only and is BF16-only;
  no per-token output and no intermediate-layer tap exist on CUDA (native and
  Metal return per-token 768-wide projected rows). New plumbing: CUDA per-token
  output (stop before the mean-L2 kernel) and a layer tap, as done for Qwen in
  `gpt.zig`.
- Gemma 4: E2B hidden 1536, E4B 2560, 12B 3840; GGUF QAT is the production
  format, HF safetensors also load. The Memorilla soft-prompt path is
  Qwen-only (embedding scale, PLE, softcap are rejected); the text route makes
  that irrelevant for snapped memories. No strict-F32 Gemma CUDA profile exists;
  a research profile would be BF16 and needs its own precision decision.
- Training: native Zig LoRA exists for both models but only with pooled
  contrastive (encoder) or causal-LM (decoder) losses; the per-token reader and
  copy head need the Python trainer. The Memorilla trainer pins
  transformers 4.53.3, too old for Gemma 4 / EmbeddingGemma 2 (5.x required).

## 3. Prerequisites for step 2 (training on the existing corpus)

1. BF16 Gemma 4 weights for training: `google/gemma-4-E4B-it` and
   `google/gemma-4-12B-it` are NOT gated on the Hub (checked 2026-11-12), so
   E4B downloads anonymously into the shared asset root
   (`plan-memorilla-native-memory/.benchmark-assets/memorilla/models/gemma-4-e4b-it`).
   E4B text geometry from its config: hidden 2560, 42 layers, 8 query / 2 KV
   heads (head_dim 256), vocab 262144, tied embeddings, per-layer input width
   256, final logit softcap 30, sliding window 512. EmbeddingGemma 2 BF16 is
   local (`.worktrees/fix-cuda-audio/.models/embeddinggemma2`, 1.49 GB).
2. A Python environment on transformers 5.x: `.venv-gemma` in this worktree
   (uv; torch cu130 wheels, transformers 5.22.2, uv cache under
   `zig/pkg/inference/.zig-cache/memorilla-uv`).
3. Trainer changes: encoder width 512 (pre-projection) or 768, no self-context,
   dual layer chosen among 24 layers (start at 16), reader output width =
   decoder hidden (2560 for E4B), record format unchanged, decoder prompt via
   the Gemma chat template with the system text in the first user turn.
4. Decoder first target: Gemma 4 E4B (local QAT for serving probes; BF16 HF for
   training). 12B after the recipe holds.

## 4. Step 2 design (trainer `scripts/memorilla/gemma_memory.py`)

- Features: EmbeddingGemma 2 `last_hidden_state` (512, after the final norm,
  before the 768 projection) plus `hidden_states[16]`, every document token row,
  L2-normalized, feature channel, concatenated [final rows; tap rows]; no
  self-context (bidirectional encoder); question query = mean of the final rows
  at width 512. LoRA rank 8 on q/v.
- Reader: `MemoryModule` with d 512, 8 heads, 2 blocks, output 2560; fresh
  init; slots default 24 (the decoder is not trained on a 16-token frame, so
  the record can carry the full site name; 16/24/32 selectable).
- Losses: the copy17 lexical recipe against the Gemma 4 embedding table, copy
  pointer and gate losses (gate weight 3).
- Decoding: contiguous runs, snap all, word completion, then the decoder reads
  the record as plain text through the Gemma chat template (single user turn:
  system text, `Memory<record>Memory`, question), greedy, 24 new tokens.
- Evaluation: the same dev split and fresh-challenge generator as copy17 so the
  numbers compare directly (target: copy17's 96.7% on the wide challenge).

## 5. Summarizing reader (decided 2026-11-12 23:50Z: soft slots through the frozen decoder, full data)

Why: a text-token memory cannot restate a passage (Stage 1 of Memorilla's
recipe maps a ~150-token passage to 16 slots), so passage-level memory needs
soft slots trained through the frozen decoder. Exact strings stay exact
through the copy head.

Data: `memorilla/Memorilla-Data` (Hub, public, 9.5 GB parquet, columns
collection_id / question / answer / documents): Stage 1 `enwiki` (~2M
passages, "Reconstruct the information you have in memory."), Stage 2
mixture of ten QA/summarisation configs (1.34M), Stage 3 benchmarks. Mirrored
to `plan-memorilla-native-memory/.benchmark-assets/memorilla/data/memorilla-data`.

Design (`scripts/memorilla/gemma_soft_memory.py`):
- Encoder: frozen EmbeddingGemma 2 per-token states (final norm + layer-16
  tap, 512 wide), computed on the fly; at most 16 documents per example
  (selected by pooled cosine to the question), 256 tokens each, 4096 rows.
- Reader: upstream MemoryModule d 512 -> 2560, K = 24 slots: 16 soft slots
  trained only through the decoder, 8 copy slots with the copy head
  (pointer + gate) supervised when the answer appears verbatim in a document.
- Decoder: frozen Gemma 4 E4B, memory rows injected as input embeddings at K
  placeholder positions, answer-token cross-entropy, gradient checkpointing,
  BF16 autocast, FP32 trainable weights. Recipe as upstream: AdamW wd 0.01,
  lr 3e-4, warmup + cosine to 10%, clip 1.0, effective batch 128, one epoch
  per stage.
- Inference: copy slots with gate >= 0.5 snap to tokens (contiguous runs);
  the rest stay soft. Serving later needs the Gemma embedding-row injection
  path in Zig (multimodal prefill route), not the plain-text prompt.
- Stages: stage1 (enwiki) -> stage2 (mixture) -> depot fine-tune as the
  Stage-3 analog (replaces the from-scratch depot run, whose slow start is
  what Stages 1-2 remove).

Cost estimate (to be replaced by smoke measurements): the 8B decoder forward
and backward per example dominates; Stage 1 at 2M examples is multiple days on
one GB10, Stage 2 about two thirds of that. Checkpoints every 1000 steps and
periodic validation allow stopping early.

### Step 2 result: from-scratch depot run (2026-11-13)

`pair-tap16-slots24-2048` (tap 16, 24 slots, 2048 steps, 2.2 s/step, 75 min):
dev 16 -> 17 -> 17 -> 16 -> 23 -> 37 -> 46 -> 50 -> 53 of 64 at steps
0..2048 (codes 31/32, names 6/16 at the end). Still rising steeply when the
cosine schedule ended; the reader learned the record format first and the
question-to-document binding only after ~1000 steps, as expected without a
pretrained reader. Not continued: the summarizing reader's Stage 1/2 replace
this from-scratch start, and the depot corpus becomes its Stage-3 fine-tune.
(`gemma_memory.py --resume` requires an identical protocol, so a longer run
would be a fresh run or needs an `--init` flag.)

### 5.1 Keeping restatement through Stage 2 (decided 2026-11-13)

Stanford's Stage 2 mixture (squad_v2, drop, coqa, pubmedqa, quail, pwc,
cnn_dailymail, samsum, dialogsum, msmarco) contains no enwiki restatement,
and their released checkpoints show the cost: restatement ROUGE-L 0.237
(stage1) -> 0.166 (stage2) -> 0.068 (stage3), while QA rises. Our goal is one
reader that both restates and answers, so Stage 2 deviates from their preset
by replaying Stage 1 data: launch with
`--datasets squad_v2,drop,coqa,pubmedqa,quail,pwc,cnn_dailymail,samsum,dialogsum,msmarco,enwiki:200000`
(roughly 13% restatement in the mixture; the cap is the tuning knob) and track
both the enwiki validation loss and the QA validation losses at every
evaluation. The depot fine-tune (Stage 3 analog) gets the same treatment: a
small replay of enwiki and Stage 2 QA alongside the depot corpus. If one
reader still cannot hold both, the fallback is the Qwen arrangement: keep the
Stage 1 checkpoint as a separate restatement mode.

## 6. Changes suggested by the Memorilla paper (arXiv 2610.21375v3, read 2026-11-13)

Paper facts that bear on our design:
- Their reader reads ONE pooled vector per chunk (documents chunked: TriviaQA
  sentence-greedy at 128 tokens, NarrativeQA paragraphs, FactKG one triple per
  chunk), stored once (5 KB per document) and reused; reader cost is flat from
  1 to 512 documents (2 ms), end to end 0.74 s on an H100. This is what lets
  them compress whole books (NarrativeQA, >4,000x).
- Architecture ablation (Table 13): query-conditioning the slots before they
  read is the key; a hard cosine top-16 selection reaches 33.86 vs 44.04 on
  PMv2. K=128 at training does not beat K=16. A larger encoder of the same
  family barely matters (4B vs 8B).
- Limitations: loses fine-grained factual details; encoder never optimized for
  compression; freezing both backbones keeps the document store portable.
- Section 5: "who"/"which" NarrativeQA questions favor RAG, "why"/"how" favor
  memory; Figure 5 shows memory OR RAG beats either alone (a hybrid oracle).
- Stage 3 recipe: lr 1e-4, warmup 0.1, max decoder length 2048, batch 64, up
  to 5 epochs, validation every quarter epoch; per-benchmark checkpoint
  selection only for PV4/PMv2. Generation limits: 64 tokens for QA, 512 for
  EnWiki reconstruction. Reader dropout 0.1, Xavier slot init.
- Sanity checks they use: Gaussian-noise documents (FactKG -5.93) and a
  decoder LoRA at matched parameters as a baseline.

What to change:
1. Dual-stream reader (before Stage 2): soft slots attend over pooled chunk
   vectors of the WHOLE collection (mean of the 512-wide final-norm states per
   chunk, cacheable, encoder frozen so the store stays portable); copy slots
   attend over per-token rows of the top few chunks only (computed at query
   time, encoder LoRA allowed there). Today the Gemma reader reads per-token
   rows of at most 16 documents x 256 tokens and picks those 16 by cosine,
   which is the hard top-k the paper's ablation found weak, and truncates long
   documents. Implementation: one key set = [pooled rows + type embedding;
   token rows + type embedding] with a per-slot mask (soft slots: pooled rows,
   copy slots: token rows, or both), initialized from the Stage 1 checkpoint.
2. Chunk documents instead of truncating them (paper's per-benchmark rules).
3. Stage 3 = their eight benchmark training splits (in Memorilla-Data) plus
   our depot corpus, with their Stage 3 hyperparameters, so results are
   comparable to their Table 11.
4. Evaluate with their protocol and metrics, including the NarrativeQA
   question-type breakdown (the direct test of the copy slots), closed-book /
   RAG-k / full-context baselines with the same decoder, document tokens per
   query, the noise test, and EnWiki reconstruction at 512 new tokens (our
   restatement numbers so far were capped at 160 and are underestimates).
5. Small deltas: reader dropout 0.1 (ours 0); keep 24 slots (16 + 8).
6. Later: the paper's continual-learning setup (memory trained with GRPO on
   agent trajectories, K=8) is a natural fit for agent memory in Antfly.
