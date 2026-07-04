# Route A — Multimodal Chunk Embedding: GPU E2E Runbook

Route A makes the deterministic multimodal (`fixed`) chunk path return
**shared-space** dense embeddings per chunk, so a **text** query can retrieve
**image / audio** chunks (cross-modal retrieval over chunks).

- One request selects ONE embedder model (`config.embedding_model`). Every chunk
  of the document is embedded through that same model, so all chunk vectors
  share a space.
- Chunk → tower routing by modality:
  - text chunk (`text/*`)  → text tower   (`EmbeddingPipeline.embed`)
  - image chunk (`image/*`) → vision tower (`EmbeddingPipeline.embedImages`)
  - audio chunk (`audio/*`) → audio tower  (`EmbeddingPipeline.embedEncodedAudio`)
- CLIP serves text+image; CLAP serves text+audio. There is no tri-modal space.
  A chunk modality the chosen model cannot embed (e.g. an audio chunk with a
  CLIP model) returns a clear 400.

This runbook is the **GPU** validation that cannot run on the CPU-only build
(needs a real CLIP/CLAP forward pass). Do NOT run it while a GPU training job is
active — it will contend for the device.

## 0. Prerequisites

- A GPU-capable build of the inference server (Metal on macOS, CUDA on Linux).
- A CLIP embedder installed under `~/.antfly/inference/models/embedders/`, e.g.
  `clip-vit-base-patch32`. Export via
  `scripts/export_model_to_registry.py` (supports CLIP multimodal). The model
  manifest must advertise the vision tower (native `clip` arch hint or a
  `visual_model_path`) — this is what `ModalityCaps.fromManifest` keys off.
- `jq`, `python3`, and `curl` on PATH.

## 1. Build and start the server (GPU)

```bash
cd zig/pkg/inference
# macOS / Metal
zig build -Dmetal=true
# Launch the inference server (adjust flags to your deployment).
./zig-out/bin/<inference-binary> serve --port 8081 &
SERVER_PID=$!
# Wait for readiness
until curl -sf localhost:8081/health >/dev/null; do sleep 0.5; done
```

You can also run it via Antfly swarm mode (inference enabled by default):
`cd go/pkg/antfly && go run ./cmd swarm`.

## 2. Build an interleaved text+image document

The `fixed` multimodal chunker takes a single `input`. Two useful shapes:

- **Animated GIF** → the fixed chunker emits one `image/png` chunk per frame
  (`chunkGif`), so a single GIF yields several image chunks.
- **Text string** → text chunks (`fixed_text`).

For a text-vs-image cross-modal check, embed an image document and a text query
separately with the SAME model and compare. Prepare a base64 image:

```bash
IMG_B64=$(base64 -i cat.png | tr -d '\n')       # a single PNG frame
GIF_B64=$(base64 -i clip.gif | tr -d '\n')      # animated GIF (multi-frame)
```

## 3. POST /chunk with an image document + embedding_model

```bash
curl -sf localhost:8081/chunk -H 'content-type: application/json' -d @- <<JSON | tee /tmp/route_a_image_chunks.json | jq '.data | length'
{
  "input": { "type": "media", "mime_type": "image/gif", "data": "$GIF_B64" },
  "config": {
    "model": "fixed",
    "include_embeddings": true,
    "embedding_model": "clip-vit-base-patch32"
  }
}
JSON
```

Assertions:

```bash
# Every chunk must carry an embedding of the model's projection dim.
jq -e '.data | all(.embedding != null and (.embedding | length) > 0)' /tmp/route_a_image_chunks.json
jq -e '.data | all(.mime_type | startswith("image/"))' /tmp/route_a_image_chunks.json
jq '.data[0].embedding_dimension' /tmp/route_a_image_chunks.json
```

Optional Matryoshka truncation (dense-only): add `"output_dimension": 256` to
`config` and assert `.embedding | length == 256` and unit L2 norm.

## 4. Text-query vs image-chunk cosine sanity check

Embed a matching and a mismatching text query through the SAME CLIP model via
`/embeddings`, then cosine against an image chunk vector. The matching caption
should score higher (shared space works).

```bash
emb() {  # $1 = text -> prints JSON array
  curl -sf localhost:8081/embeddings -H 'content-type: application/json' \
    -d "{\"model\":\"clip-vit-base-patch32\",\"input\":$(jq -Rn --arg t "$1" '$t')}" \
    | jq -c '.data[0].embedding'
}
MATCH=$(emb "a photo of a cat")
MISS=$(emb "a city skyline at night")
IMGVEC=$(jq -c '.data[0].embedding' /tmp/route_a_image_chunks.json)

python3 - "$MATCH" "$MISS" "$IMGVEC" <<'PY'
import sys, json, math
def cos(a,b):
    d=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
    return d/(na*nb+1e-9)
match, miss, img = map(json.loads, sys.argv[1:4])
cm, cx = cos(match,img), cos(miss,img)
print(f"cos(match caption, image chunk) = {cm:.4f}")
print(f"cos(mismatch caption, image chunk) = {cx:.4f}")
assert cm > cx, "cross-modal retrieval broken: matching caption did not win"
print("OK: text query retrieves the image chunk in shared space")
PY
```

## 5. Error-taxonomy checks (all should 4xx, server stays up)

```bash
# Unknown embedder -> 404 MODEL_NOT_FOUND
curl -s -o /dev/null -w '%{http_code}\n' localhost:8081/chunk -d \
  '{"input":{"type":"media","mime_type":"image/gif","data":"'$GIF_B64'"},"config":{"model":"fixed","include_embeddings":true,"embedding_model":"does-not-exist"}}'   # expect 404

# Non-multimodal embedder (plain text embedder) -> 400
curl -s -o /dev/null -w '%{http_code}\n' localhost:8081/chunk -d \
  '{"input":{"type":"media","mime_type":"image/gif","data":"'$GIF_B64'"},"config":{"model":"fixed","include_embeddings":true,"embedding_model":"bge-small-en"}}'      # expect 400

# Modality unsupported by chosen model: audio chunk with a CLIP model -> 400
AUD_B64=$(base64 -i clip.wav | tr -d '\n')
curl -s -o /dev/null -w '%{http_code}\n' localhost:8081/chunk -d \
  '{"input":{"type":"media","mime_type":"audio/wav","data":"'$AUD_B64'"},"config":{"model":"fixed","include_embeddings":true,"embedding_model":"clip-vit-base-patch32"}}'  # expect 400

# Sparse/boundary not supported on the multimodal dense path -> 400
curl -s -o /dev/null -w '%{http_code}\n' localhost:8081/chunk -d \
  '{"input":{"type":"media","mime_type":"image/gif","data":"'$GIF_B64'"},"config":{"model":"fixed","include_embeddings":true,"include_sparse":true,"embedding_model":"clip-vit-base-patch32"}}'  # expect 400
```

## 6. CLAP variant (text+audio)

Repeat sections 3–4 with a CLAP embedder (e.g. `clap-htsat-unfused`) and a
`audio/wav` input. The fixed chunker windows the WAV into `audio/wav` chunks
(`chunkWav`); each is routed to the audio tower. A text query should retrieve
the matching audio chunk in the shared CLAP space.

## 7. Teardown

```bash
kill $SERVER_PID
```

## What this exercises that CPU tests cannot

- Real CLIP/CLAP vision/audio/text forward passes and shared-space geometry
  (the CPU unit tests use a synthetic pipeline that only checks routing, order
  preservation, output_dimension truncation, and the error taxonomy).
- The `model_manager` session loading path
  (`ensureEmbeddingAssets` → `embeddingPipeline`) for a multimodal embedder.
- End-to-end wiring from `/chunk` request → `embedFixedChunksMultimodal` →
  `multimodal_chunk_embedding.embedChunks` → `ChunkObject.embedding`.
