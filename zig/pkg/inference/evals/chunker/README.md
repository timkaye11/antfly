# Fused Chunker Evaluation Contract

This directory defines the public benchmark contract for graduating the Zig/Metal
fused chunker from mechanical readiness to production quality.

## 2026-07 Gate Re-scope (split-encoder serving)

The fine-tuned fused chunker learned excellent boundaries (internal best F1
0.79) but the fine-tune catastrophically forgot the base encoder's retrieval
ability (doc-level NDCG@10 0.0096 versus 0.335+ for the raw embed-base). The
validated production fix is SPLIT-ENCODER serving: boundaries from the fused
model (boundary-only forward), chunk embeddings from the raw frozen embed-base
with nomic retrieval prefixes ("search_document: " for chunks,
"search_query: " for queries), which restores doc-level NDCG ~35x. Against the
best same-encoder local baseline the learned boundaries add about +1%
doc-level NDCG, and the chonky character-separator benchmark is a task
mismatch for retrieval-oriented boundaries. The release gates were re-scoped
accordingly:

- Boundary lane: gate only on `min_internal_phase20_best_f1 >= 0.78` (plus the
  fixed-threshold calibration delta). Per-dataset chonky numbers and
  base/large margins are still computed and REPORTED by the verifier, but
  `must_beat_chonky_base_each_dataset` and
  `must_match_or_beat_chonky_large_mean` are report-only (`false`).
- Retrieval lane: gate on the absolute health floor
  `min_overall_ndcg_at_10 >= 0.15` (the fused fine-tune's 0.0096 collapse must
  never pass). `relative_gain_over_best_local_baseline` is still computed and
  REPORTED, but `relative_gain_report_only: true` disables the former 5%
  must-beat gate.

## Boundary F1 Lane

The boundary lane mirrors chonky's published evaluation:

- Build paragraph-boundary datasets with the same sources and paragraph split
  semantics as chonky.
- Evaluate the first 1,000,000 tokens per dataset.
- Treat separator positions as Chonky-compatible character offsets, including
  the final document boundary, and report exact separator F1. When the first
  1,000,000-token stream is split into model-sized windows, only the true final
  prepared record contributes the final-boundary separator.
- Report per-dataset scores for `bookcorpus`, `en_judgements`,
  `paul_graham`, and `20_newsgroups`.

Release gates on the internal Go Phase-20 validation floor
(`internal_phase20_best_f1 >= 0.78`) and the fixed-threshold calibration delta.
Chonky base/large comparisons are reported per dataset for visibility but are
not release-gating (see the 2026-07 re-scope above).

## Retrieval NDCG Lane

The retrieval lane mirrors Voyage's reporting shape where possible:

- Report chunk-level and document-level `NDCG@10`.
- Keep single-vector, contextual chunk-vector, SPLADE-only, and dense+SPLADE
  hybrid lanes separate.
- Use only open/reproducible datasets for local claims.
- Treat Voyage Context 4 spreadsheet values as external targets unless a direct
  same-dataset/API run is added.
- For split-encoder runs, produce chunk vectors by POSTing `/chunk` with
  `config.embedding_model` set to the frozen embed-base (document prefix
  applied automatically from the embedder manifest), and query vectors via
  `/embeddings` against the same embedder with its query prefix
  (e.g. `search_query: `).

Release requires overall `NDCG@10` at or above the manifest's absolute floor
(`min_overall_ndcg_at_10`, currently 0.15). The relative gain over the best
local fixed/chonky baseline is reported alongside but no longer gated
(`relative_gain_report_only`); SPLADE-on-frozen-base lanes are a follow-up
while that head is still training.

## Candidate Model Build

Production evidence starts from a trained checkpoint, not from benchmark tooling
alone. Run the full readiness wrapper to produce a gated candidate checkpoint
and manifest:

```bash
ANTFLY_FUSED_CHUNKER_READINESS_MODE=full \
ANTFLY_FUSED_CHUNKER_TRAIN_DATA=/path/to/fused_train.jsonl \
ANTFLY_FUSED_CHUNKER_MODEL_DIR=/path/to/modernbert-model \
ANTFLY_FUSED_CHUNKER_MIN_BEST_F1=0.766 \
ANTFLY_FUSED_CHUNKER_MIN_FIXED_F1=0.766 \
ANTFLY_FUSED_CHUNKER_MIN_PROBABILITY_GAP=0.05 \
scripts/run_fused_chunker_production_readiness.sh
```

The candidate directory must contain `checkpoint_final.safetensors`,
`fused_training_manifest.json`, `fused_training_metrics.jsonl`, `train.log`,
and `readiness_summary.json`. Full readiness gates both best validation quality
and latest validation quality, so a run that peaks and then collapses cannot
pass on an old best score. When validation fails, the wrapper writes
`validation_failure_analysis.json` with a stop/continue recommendation and a
failure class such as underconfident fixed threshold, inverted probability gap,
overprediction, buried gold ranks, or train-loss-down/validation-rank-poor.
You can also run the analyzer directly:

```bash
zig build -Dskip-openapi=true analyze-fused-chunker-validation-failure -- \
  --out-dir /path/to/readiness-run \
  --analysis-out /path/to/readiness-run/validation_failure_analysis.json
```

Step validation metrics include gold-boundary rank diagnostics
(`gold_positive_*_rank_percentile`, `gold_positive_top_5x_recall`, and
`gold_positive_top_10x_recall`) so a failed run can distinguish calibration
problems from genuinely poor ranking. To compare train and validation ranking
at the same step gate, set
`ANTFLY_FUSED_CHUNKER_STEP_TRAIN_EVAL_MAX_EXAMPLES=<n>` on a bounded probe; the
trainer will emit `train_validation_step` records alongside `validation_step`
records, and the analyzer will classify train-good/validation-poor versus
train-and-validation-poor failures. Export the checkpoint for Go/runtime parity
when needed:

Before launching a long candidate run after a boundary-quality failure, use the
frozen-feature probe with non-overlapping offsets to separate same-split
generalization from train/validation distribution shift:

```bash
zig build -Dskip-openapi=true -Dmetal=true train-fused-chunker -- \
  --data /path/to/fused_train.jsonl \
  --val-data /path/to/fused_train.jsonl \
  --split train \
  --val-split train \
  --output /tmp/frozen-feature-same-split \
  --model-dir /path/to/modernbert-model \
  --backend metal \
  --lora-rank 0 \
  --lambda-embed 0 \
  --deterministic \
  --disable-encoder-neftune \
  --debug-frozen-feature-probe \
  --debug-frozen-feature-train-examples 16 \
  --debug-frozen-feature-val-examples 16 \
  --debug-frozen-feature-val-offset 1024 \
  --debug-frozen-feature-epochs 30
```

```bash
zig build -Dskip-openapi=true convert-fused-chunker-checkpoint-for-go -- \
  --checkpoint /path/to/candidate/checkpoint_final.safetensors \
  --out /tmp/fused_chunker_go_runtime.safetensors \
  --summary /tmp/fused_chunker_go_runtime.summary.json
```

Generate retrieval query/chunk embedding JSONL from the candidate model with
contextual dense embeddings and SPLADE enabled, then rank and score those
embeddings with the tools below. The release artifact is promotable only if the
candidate checkpoint passes readiness, boundary F1, retrieval NDCG, runtime
parity, and API response-shape checks.

Retrieval scoring consumes explicit ranked run JSONL. Normalize candidate
chunker responses into strict embedding JSONL first, generate the ranked run,
then score it. The materializer accepts raw `/chunk` response JSONL, wrappers
such as `{"document_id":"d1","response":{"data":[...]}}`, or direct embedding
records. It fails by default if dense or SPLADE vectors are missing, if dense
dimensions are inconsistent, or if SPLADE indices are not strictly ascending.

```bash
zig build -Dskip-openapi=true materialize-fused-chunker-retrieval-embeddings -- \
  --input /tmp/candidate_docs.chunk_responses.jsonl \
  --out /tmp/chunks.embeddings.jsonl \
  --kind docs \
  --output-dimension 256

zig build -Dskip-openapi=true materialize-fused-chunker-retrieval-embeddings -- \
  --input /tmp/candidate_queries.chunk_responses.jsonl \
  --out /tmp/queries.embeddings.jsonl \
  --kind queries \
  --output-dimension 256
```

The normalized query records include `query_id` and `relevant_ids`; document
records include `id`/`document_id`. Dense vectors use
`dense_embedding`; SPLADE vectors use `sparse_embedding` with `indices` and
`values`.

```bash
zig build -Dskip-openapi=true rank-fused-chunker-retrieval-benchmark -- \
  --queries /tmp/queries.embeddings.jsonl \
  --docs /tmp/chunks.embeddings.jsonl \
  --out /tmp/dense_splade_hybrid.run.jsonl \
  --mode hybrid \
  --top-k 100
```

The ranked run format is also accepted directly. Each record must include
`relevant_ids` or `relevant`, plus either `ranked_ids` or `results`/`hits` with
`id`, `doc_id`, or `chunk_id` fields:

```json
{"query_id":"q1","relevant_ids":["doc:7"],"ranked_ids":["doc:2","doc:7","doc:9"]}
```

Score a contextual dense+SPLADE run and local baselines with:

```bash
zig build -Dskip-openapi=true score-fused-chunker-retrieval-benchmark -- \
  --run /tmp/dense_splade_hybrid.run.jsonl \
  --out /tmp/retrieval.fused_chunker_benchmark_results.json \
  --output-dimension 256 \
  --baseline fixed_500_50_same_encoder=/tmp/fixed_500_50.run.jsonl \
  --baseline chonky_boundaries_same_encoder=/tmp/chonky.run.jsonl
```

## Result File

Materialize each Chonky/HuggingFace token-label dataset into fused-chunker
JSONL first. The input may be Chonky-style `tokens`/`ner_tags`, a
`paragraphs` array, or newline-separated paragraph `text`.

```bash
zig build -Dskip-openapi=true prepare-fused-chunker-boundary-eval -- \
  --input /path/to/chonky/bookcorpus.tokens.jsonl \
  --out /tmp/bookcorpus.fused_eval.jsonl \
  --dataset-name bookcorpus \
  --max-total-tokens 1000000 \
  --max-record-tokens 384
```

Benchmark runs must write a JSON result file with schema
`fused_chunker_benchmark_results/v1`. `dataset_results[].f1` in the boundary
lane must use `chonky_character_separator_f1`; internal fixed/best threshold
fields remain token-boundary quality diagnostics from the trained head:

```json
{
  "schema_version": "fused_chunker_benchmark_results/v1",
  "boundary_f1": {
    "dataset_metric": "chonky_character_separator_f1",
    "internal_phase20_best_f1": 0.80,
    "fixed_threshold_f1": 0.78,
    "best_threshold_f1": 0.80,
    "dataset_results": [
      { "name": "bookcorpus", "f1": 0.80 }
    ]
  },
  "retrieval_ndcg": {
    "output_dimension": 256,
    "overall_ndcg_at_10": 0.74,
    "voyage_context_4_target_ndcg_at_10": 0.8054,
    "distance_to_voyage_context_4_target": 0.0654,
    "best_local_baseline_ndcg_at_10": 0.70,
    "relative_gain_over_best_local_baseline": 0.0571,
    "baselines": [
      { "name": "fixed_500_50_same_encoder", "overall_ndcg_at_10": 0.68 }
    ]
  }
}
```

`eval-fused-chunker` can emit this schema for a single dataset run:

```bash
zig build -Dskip-openapi=true eval-fused-chunker -- \
  --data /tmp/bookcorpus.fused_eval.jsonl \
  --checkpoint /path/to/checkpoint.safetensors \
  --model-dir /path/to/modernbert-model \
  --benchmark-dataset-name bookcorpus \
  --results-out /tmp/bookcorpus.fused_chunker_benchmark_results.json \
  --baseline fixed_500_50_same_encoder:0.68 \
  --baseline chonky_boundaries_same_encoder:0.70
```

For the full release gate, aggregate one `dataset_results` entry per manifest
dataset into a single result file before running the verifier.

```bash
zig build -Dskip-openapi=true aggregate-fused-chunker-benchmark -- \
  --out /tmp/all.fused_chunker_benchmark_results.json \
  --input /tmp/bookcorpus.fused_chunker_benchmark_results.json \
  --input /tmp/en_judgements.fused_chunker_benchmark_results.json \
  --input /tmp/paul_graham.fused_chunker_benchmark_results.json \
  --input /tmp/20_newsgroups.fused_chunker_benchmark_results.json
```

Verify release evidence with:

```bash
zig build -Dskip-openapi=true verify-fused-chunker-benchmark -- \
  --manifest evals/chunker/manifest.json \
  --results /path/to/fused_chunker_benchmark_results.json
```
