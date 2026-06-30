# Fused Chunker Evaluation Contract

This directory defines the public benchmark contract for graduating the Zig/Metal
fused chunker from mechanical readiness to production quality.

## Boundary F1 Lane

The boundary lane mirrors chonky's published evaluation:

- Build paragraph-boundary datasets with the same sources and paragraph split
  semantics as chonky.
- Evaluate the first 1,000,000 tokens per dataset.
- Treat separator positions as sorted exact boundary offsets and report F1.
- Report per-dataset scores for `bookcorpus`, `en_judgements`,
  `paul_graham`, and `20_newsgroups`.

Release requires beating chonky base on each English dataset and matching or
beating the chonky large English mean, after the internal Go Phase-20 validation
gate has recovered.

## Retrieval NDCG Lane

The retrieval lane mirrors Voyage's reporting shape where possible:

- Report chunk-level and document-level `NDCG@10`.
- Keep single-vector, contextual chunk-vector, SPLADE-only, and dense+SPLADE
  hybrid lanes separate.
- Use only open/reproducible datasets for local claims.
- Treat Voyage Context 4 spreadsheet values as external targets unless a direct
  same-dataset/API run is added.

Release requires the contextual dense+SPLADE lane to beat local fixed/chonky
chunk baselines by at least 5% relative average `NDCG@10`.

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
and `readiness_summary.json`. Export the checkpoint for Go/runtime parity when
needed:

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
`fused_chunker_benchmark_results/v1`:

```json
{
  "schema_version": "fused_chunker_benchmark_results/v1",
  "boundary_f1": {
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
