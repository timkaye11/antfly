# Split-Encoder Serving — Fused Boundaries + Frozen Embed-Base: GPU E2E Runbook

The fine-tuned fused chunker has excellent boundaries (internal best F1 0.79)
but its fine-tune catastrophically forgot the base encoder's retrieval ability
(doc-level NDCG@10 0.0096 vs 0.335+ for the raw embed-base). SPLIT-ENCODER
serving fixes this at the API level:

- Boundaries come from the fused model via a cheap **boundary-only** forward
  (`forwardBoundaryOnly` — no token-embedding heads).
- Each resulting chunk's **text** is embedded through the raw frozen embed-base
  named by `config.embedding_model`, resolved via the same path `/embeddings`
  uses (`resolveModelPath` → `loadFromDir` → `ensureEmbeddingAssets` →
  `embeddingPipeline`).
- For nomic prefix-conditioned embedders the manifest knob
  (`model_manifest.json` → `"embedding_prefixes": {"document":
  "search_document: ", "query": "search_query: "}`) prepends
  `search_document: ` to every chunk text; `config.embedding_prefix` overrides
  per request (`""` disables).
- Queries must be embedded via `/embeddings` against the **same** embedder with
  the matching query prefix (`search_query: `) prepended to the query text.
- `output_dimension` truncate+renormalize is honored on the embed-base vectors.
  `include_sparse` together with `embedding_model` returns 400 for now
  (SPLADE-on-frozen-base is still training; follow-up).
- With `embedding_model` unset the fused path is byte-for-byte unchanged
  (regression invariant).

This runbook is the **GPU** validation that cannot run on the CPU-only build
(needs real fused-chunker + embed-base forwards). Do NOT run it while a GPU
training job is active — it will contend for the device.

## 0. Prerequisites

- A GPU-capable build of the inference server (Metal on macOS).
- The fused chunker candidate installed under
  `~/.antfly/inference/models/chunkers/<fused-chunker>/`.
- The raw frozen embed-base installed under
  `~/.antfly/inference/models/embedders/<embed-base>/`, with
  `model_manifest.json` declaring the nomic prefixes:

  ```json
  {
    "type": "embedder",
    "embedding_prefixes": {
      "document": "search_document: ",
      "query": "search_query: "
    }
  }
  ```

- `jq`, `python3`, and `curl` on PATH.

## 1. Build and start the server (GPU)

```bash
cd zig/pkg/inference
zig build -Dmetal=true
./zig-out/bin/<inference-binary> serve --port 8081 &
until curl -sf localhost:8081/health >/dev/null; do sleep 0.5; done
```

## 2. POST /chunk with embedding_model (split-encoder)

```bash
CHUNKER=<fused-chunker>
EMBEDDER=<embed-base>

curl -sf localhost:8081/chunk -H 'content-type: application/json' -d @- <<JSON | tee /tmp/split_encoder_chunks.json | jq '.data | length'
{
  "input": "First topic paragraph...\n\nSecond topic paragraph about something else...\n\nThird topic...",
  "config": {
    "model": "$CHUNKER",
    "embedding_model": "$EMBEDDER",
    "include_embeddings": true,
    "include_boundary_scores": true
  }
}
JSON
```

Checks:

- `data[].text`, `start_char`/`end_char`, `start_token`/`end_token`,
  `boundary_score` present (fused boundaries preserved, order by `index`).
- `data[].embedding` present with the embed-base dimension (NOT the fused
  chunker's), `embedding_dimension` matches.
- Repeat without `embedding_model`: identical chunk boundaries; embeddings now
  come from the fused model (regression invariant).
- Repeat with `"include_sparse": true` alongside `embedding_model`: expect 400.
- Repeat with `"output_dimension": 256`: `embedding_dimension == 256`, unit
  L2 norm.

## 3. Verify the chunk embedding equals a direct /embeddings call

Chunk vectors must equal a direct `/embeddings` call on the SAME prefixed
text — this proves the raw base (with `search_document: `) produced them:

```bash
CHUNK_TEXT=$(jq -r '.data[0].text' /tmp/split_encoder_chunks.json)

curl -sf localhost:8081/embeddings -H 'content-type: application/json' -d @- <<JSON | jq '.data[0].embedding' > /tmp/direct_embedding.json
{ "model": "$EMBEDDER", "input": "search_document: $CHUNK_TEXT" }
JSON

python3 - <<'PY'
import json
chunks = json.load(open('/tmp/split_encoder_chunks.json'))
chunk_vec = chunks['data'][0]['embedding']
direct_vec = json.load(open('/tmp/direct_embedding.json'))
assert len(chunk_vec) == len(direct_vec), (len(chunk_vec), len(direct_vec))
dot = sum(a*b for a, b in zip(chunk_vec, direct_vec))
na = sum(a*a for a in chunk_vec) ** 0.5
nb = sum(b*b for b in direct_vec) ** 0.5
cos = dot / (na * nb)
print('cosine(chunk_vec, direct_prefixed_embed) =', cos)
assert cos > 0.999, cos
PY
```

If the embedder applies an unconditional pipeline `text_prefix` of its own
(e.g. jina v5 "Document: "), drop the manual `search_document: ` in the direct
call and compare against the manifest-declared document prefix instead.

## 4. Query-side symmetry sanity

```bash
curl -sf localhost:8081/embeddings -H 'content-type: application/json' \
  -d "{\"model\":\"$EMBEDDER\",\"input\":\"search_query: second topic\"}" \
  | jq '.data[0].embedding' > /tmp/query_embedding.json
```

Cosine of the query vector against each chunk vector should rank the matching
chunk first; with the fused model's own (forgotten) embeddings it does not.

## 5. Retrieval benchmark lane (optional, full evidence)

Generate `/chunk` responses with `embedding_model` set for the benchmark corpus
and score per `evals/chunker/README.md` (materialize → rank → score → verify).
The 2026-07 re-scoped gate expects `overall_ndcg_at_10 >= 0.15` with the
relative gain over the best local baseline reported only.
