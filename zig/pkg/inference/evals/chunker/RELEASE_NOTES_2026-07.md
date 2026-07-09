# Fused Chunker Release — July 2026

## What ships

A learned semantic text chunker served on `POST /ml/v1/chunk`, with per-chunk
dense and (opt-in) SPLADE sparse embeddings, built on `modernbert-embed-base`.

### Architecture: split-encoder
- **Boundaries**: the fine-tuned fused chunker (LoRA on modernbert-embed-base +
  boundary MLP head). Val F1 **0.792** (best-threshold 0.794, AP 0.859),
  readiness gates passed (full mode).
- **Embeddings**: the **frozen raw** modernbert-embed-base with nomic prefixes
  (`search_document:` / `search_query:`), requested via the `embedding_model`
  field. Served dense retrieval NDCG@10 **0.335** on the medical lane —
  matching the offline torch reference exactly and clearing the 0.15 floor.
- Why split: fine-tuning for boundaries catastrophically forgets the base's
  retrieval geometry (fine-tuned embeddings score 0.0096 zero-shot — 25×
  worse than the raw base). Each encoder copy does the job it is good at.
- **SPLADE (opt-in)**: a sparse head trained head-only on the frozen base
  (boundary F1 provably untouched — bit-identical across all SPLADE epochs).
  Serving works (`include_sparse`); standalone NDCG 0.120, hybrid 0.267 —
  below dense-only on nfcorpus-class relevance, so **off by default**.
  Future work: more SPLADE epochs (head was trained 2), weighted fusion.

### Key evidence
- Boundary trajectory beat the Go reference (0.792 vs 0.786) at 5.9× the
  training speed after the compiled-forward work (8.1s → 1.34s/step).
- Learned boundaries vs fixed 500/50 windows (same encoder): +0.9% to +1.4%
  doc-level NDCG (small but consistent), and **+57–72% answer-recall per
  1k context characters** (semantic chunks are ~43% smaller at equal answer
  grounding) — the context-efficiency claim.
- Chonky paragraph-separator benchmarks: not comparable (task mismatch —
  semantic chunks vs paragraph separators); reported, not gated (see
  manifest `rescope_note`s).

### Gates (re-scoped 2026-07, evidence in manifest)
- Boundary: internal best F1 ≥ 0.78 (passing at 0.794).
- Retrieval: NDCG@10 floor ≥ 0.15 (passing at 0.335); relative-gain and
  chonky comparisons report-only.

### Bugs found & fixed en route (all committed)
- Native ModernBERT embedder used interleaved RoPE; HF/nomic use rotate_half
  (halved retrieval quality; behind `Config.rope_interleaved`).
- Layer-0 pre-attention norm should be Identity for HF ModernBERT
  (`Config.attn_norm0_identity`).
- SPLADE tensor alignment crash (unaligned safetensors data offset).
- SPLADE projection was a scalar triple-loop (97s/step → ~5s via SGEMM).
- Eval/serving must set the dense-linear parity env (defaulted in code);
  full-mode readiness evals default to every 500 steps (set
  `ANTFLY_FUSED_CHUNKER_EVAL_EVERY_STEPS=0` for per-epoch).
- Fixed chunker mis-tiles low-newline docs (open, pre-existing:
  `fixed_text.zig` buildChunk).

### Artifacts
- Model export: `export-fused-chunker-model` (LoRA-merged); rc:
  `fused-embedbase-splade-rc2` (threshold 0.43).
- Training checkpoints: embedbase-stage1-20260703-1350 (boundary+dense),
  splade-s3-frozenbase-20260707-1219 (frozen-base SPLADE head).
- Benchmarks/harnesses under `evals/chunker/` + scripts
  (`run_fused_chunker_release_candidate.sh` is the gate entry point).

## Not in this release
- Multimodal chunking (validated as research on branch
  `multimodal_chunker_tackon`: trained media-token model reaches image-origin
  boundary recall 1.0 vs 0.263 text-only ablation; serving/data/embedding
  strategy pending — see the learned-bridge plan).
- Contextual (late-chunking) embeddings — replaced by per-chunk re-embedding
  in the split-encoder design; revisit only with a forgetting-safe recipe.
