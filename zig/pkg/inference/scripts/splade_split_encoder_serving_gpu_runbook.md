# Frozen-Base SPLADE in Split-Encoder /chunk Serving: GPU E2E Runbook

Builds on `split_encoder_chunk_embedding_gpu_runbook.md`. That runbook lifted
retrieval quality by embedding each fused-chunker chunk's text through the raw
frozen embed-base (dense). This one adds the **sparse** half: a SPLADE head
trained on the **frozen** embed-base (not the fine-tuned fused encoder) produces
a vocabulary-space sparse vector per chunk from the SAME frozen-base per-token
hidden states, so `POST /chunk` with `embedding_model` + `include_sparse`
returns BOTH `embedding` (dense) and `sparse_embedding` (SPLADE) per chunk.

Design (Option B — SPLADE packaged WITH the embedder): the SPLADE projection
weight is copied into the embed-base model dir as `splade_head.safetensors`.
When present, the embedder is "SPLADE-capable" (`manifest.has_splade`) and its
`EmbeddingPipeline.embedSplade` runs the encoder, projects raw per-token hidden
states → vocab logits via the **threaded SGEMM** (`fused_chunker_splade`
`projectSpladeLogits` / `computeSpladeActivation` — never the scalar triple loop
that was the 97s/step regression), applies `log(1+relu)`, max-pools over the
chunk's masked tokens, and the serving layer sparsifies (positive-only, top_k,
ascending indices).

Key input decision: **SPLADE is fed the RAW chunk text** (no nomic
`search_document: ` retrieval prefix). SPLADE max-pools over content tokens, so
a retrieval prefix would inject spurious `search`/`document` vocab dims into
every vector. The dense half still uses the document prefix. The embedder's own
unconditional manifest `text_prefix` (empty for nomic) still applies to both,
matching the dense path's non-stacking behavior. If the s3 SPLADE was trained
WITH a prefix, flip `embedTextChunksSparse` to pass that prefix and re-run §4.

This is the **GPU** validation that cannot run on the CPU-only build (needs real
fused-chunker + embed-base forwards). Do NOT run it while a GPU training/eval
job is active — it will contend for the device.

## 0. Prerequisites

- A GPU-capable build of the inference server (Metal on macOS).
- The fused chunker candidate under
  `~/.antfly/inference/models/chunkers/<fused-chunker>/`.
- The raw frozen embed-base under
  `~/.antfly/inference/models/embedders/<embed-base>/`, with `model_manifest.json`
  declaring the nomic prefixes (see the split-encoder runbook §0).
- The frozen-base SPLADE weight:
  `/private/tmp/zig-fused-chunker-readiness/splade-s3-frozenbase-20260707-1219/splade_w_epoch_2.safetensors`
  (tensor `splade_proj_weight`, F32 `[50368, 768]` over the ModernBERT vocab).
- `jq`, `python3`, `curl` on PATH.

## 1. Package the SPLADE head with the embed-base

The loader (`LoadedModel.ensureSpladeHead`) reads tensor `splade_proj_weight`
(falling back to the fused-export key). The checkpoint already uses that key, so
packaging is a pure copy — no rename:

```bash
EMBED_DIR=~/.antfly/inference/models/embedders/<embed-base>
SPLADE_SRC=/private/tmp/zig-fused-chunker-readiness/splade-s3-frozenbase-20260707-1219/splade_w_epoch_2.safetensors

cp "$SPLADE_SRC" "$EMBED_DIR/splade_head.safetensors"

# (optional) confirm the tensor name/shape the loader expects
python3 - "$EMBED_DIR/splade_head.safetensors" <<'PY'
import sys, struct, json
p = sys.argv[1]
with open(p, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
for k, v in hdr.items():
    if k != '__metadata__':
        print(k, v['dtype'], v['shape'])
PY
# expect: splade_proj_weight F32 [50368, 768]
```

File presence alone flips `manifest.has_splade` true — no manifest edit needed.
If the embedder's hidden size or vocab differs from the defaults (768 / 50368),
set `"splade_vocab_size"` in `model_manifest.json`; the loader validates
`splade_vocab_size * hidden_size == element_count` and 400s on mismatch.

## 2. Build and start the server (GPU)

```bash
cd zig/pkg/inference
zig build -Dmetal=true
./zig-out/bin/<inference-binary> serve --port 8081 &
until curl -sf localhost:8081/health >/dev/null; do sleep 0.5; done
```

## 3. POST /chunk with embedding_model + include_sparse

```bash
CHUNKER=<fused-chunker>
EMBEDDER=<embed-base>

curl -sf localhost:8081/chunk -H 'content-type: application/json' -d @- <<JSON \
  | tee /tmp/splade_chunks.json | jq '.data | length'
{
  "input": "First topic paragraph...\n\nSecond topic paragraph about something else...\n\nThird topic...",
  "config": {
    "model": "$CHUNKER",
    "embedding_model": "$EMBEDDER",
    "include_embeddings": true,
    "include_sparse": true,
    "sparse_top_k": 128
  }
}
JSON
```

Per-chunk checks (`jq`):

```bash
python3 - <<'PY'
import json
doc = json.load(open('/tmp/splade_chunks.json'))
data = doc['data']
assert data, 'no chunks'
for i, c in enumerate(data):
    assert c['index'] == i, ('order broken', i, c['index'])   # chunk order preserved
    assert c.get('embedding'), ('missing dense', i)           # dense present
    sv = c.get('sparse_embedding')
    assert sv and sv['indices'], ('missing sparse', i)        # sparse present + nonzero
    idx, val = sv['indices'], sv['values']
    assert len(idx) == len(val)
    assert len(idx) <= 128, ('top_k exceeded', len(idx))      # sparse_top_k honored
    assert idx == sorted(idx) and len(set(idx)) == len(idx)   # ascending, unique
    assert all(v > 0 for v in val)                            # positive-only (log1p(relu))
    assert all(0 <= j < 50368 for j in idx)                   # in-vocab
print('OK', len(data), 'chunks carry dense + SPLADE sparse')
PY
```

Error-path checks:

- Same request against a **dense-only** embedder (no `splade_head.safetensors`):
  expect `400 SPARSE_EMBEDDINGS_UNSUPPORTED`.
- `include_sparse: true` + `embedding_model` but `include_embeddings: false`:
  expect `400 INVALID_REQUEST` (split-encoder SPLADE needs `include_embeddings`).
- Drop `include_sparse`: response is byte-for-byte the split-encoder dense result
  (regression invariant — no `sparse_embedding` field emitted).
- Dense `embedding` is unchanged whether or not `include_sparse` is set (the
  sparse forward does not perturb the dense vector).

## 4. Sanity: served sparse == direct SPLADE-head application

The served vector must equal applying the SPLADE head directly to the chunk's
frozen-base hidden states for the RAW (un-prefixed) chunk text. Reference in
Python against the packaged weight (uses HF for the encoder; adjust model id to
your embed-base):

```bash
CHUNK_TEXT=$(jq -r '.data[0].text' /tmp/splade_chunks.json)
jq '.data[0].sparse_embedding' /tmp/splade_chunks.json > /tmp/served_sparse.json

python3 - "$CHUNK_TEXT" <<'PY'
import sys, json, struct, numpy as np
from transformers import AutoTokenizer, AutoModel   # or your embed-base loader
import torch

text = sys.argv[1]                    # RAW chunk text — NO search_document: prefix
MODEL = "nomic-ai/modernbert-embed-base"
SPLADE = "/root/.antfly/inference/models/embedders/<embed-base>/splade_head.safetensors"
TOP_K = 128

# SPLADE projection weight [vocab, hidden]
with open(SPLADE, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n)); base = 8 + n
    meta = hdr['splade_proj_weight']; s, e = meta['data_offsets']
    f.seek(base + s); W = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(meta['shape'])

tok = AutoTokenizer.from_pretrained(MODEL)
enc = tok(text, return_tensors='pt', truncation=True)
with torch.no_grad():
    hidden = AutoModel.from_pretrained(MODEL)(**enc).last_hidden_state[0].numpy()  # [seq, hidden]
mask = enc['attention_mask'][0].numpy().astype(bool)
hidden = hidden[mask]                                    # active tokens only

logits = hidden @ W.T                                    # [tokens, vocab]  (SGEMM in prod)
act = np.log1p(np.maximum(logits, 0.0))                  # log(1 + relu)
dense = act.max(axis=0)                                  # max-pool over tokens
nz = np.nonzero(dense > 0)[0]
order = nz[np.argsort(-dense[nz])][:TOP_K]
ref = {int(i): float(dense[i]) for i in order}

served = json.load(open('/tmp/served_sparse.json'))
srv = {int(i): float(v) for i, v in zip(served['indices'], served['values'])}

inter = set(ref) & set(srv)
jac = len(inter) / len(set(ref) | set(srv))
maxdiff = max((abs(ref[i] - srv[i]) for i in inter), default=0.0)
print(f'jaccard(top-{TOP_K})={jac:.4f}  max|Δval|={maxdiff:.4g}  '
      f'served={len(srv)} ref={len(ref)}')
assert jac > 0.98, jac                                   # index sets agree
assert maxdiff < 1e-2, maxdiff                           # values agree (fp/backend tol)
print('OK: served SPLADE matches direct frozen-base application')
PY
```

Notes:
- Small index-set churn near the top_k boundary is expected when many dims tie;
  loosen the Jaccard threshold or raise `sparse_top_k` if the tail is dense.
- If the direct check disagrees systematically, the likely cause is a **prefix
  mismatch**: production feeds RAW text to SPLADE. Confirm the reference uses the
  same raw text (this script does).
- Metal vs. host SGEMM produce tiny fp differences; `max|Δval| < 1e-2` covers it.

## 5. Latency note

`include_sparse` adds ONE extra frozen-base encoder forward per /chunk request
(the sparse path calls `embedSplade`, which runs its own forward on the raw chunk
texts, in addition to the dense `embedTextChunks` forward). The SPLADE projection
itself is the tiled/multithreaded SGEMM, NOT the scalar loop, so it is a few ms —
not the 97s-class regression. Sanity-check serving latency:

```bash
for sparse in false true; do
  t=$( { /usr/bin/time -p curl -sf localhost:8081/chunk -H 'content-type: application/json' \
    -d "{\"input\":\"...long doc...\",\"config\":{\"model\":\"$CHUNKER\",\"embedding_model\":\"$EMBEDDER\",\"include_embeddings\":true,\"include_sparse\":$sparse,\"sparse_top_k\":128}}" \
    >/dev/null; } 2>&1 | awk '/real/{print $2}' )
  echo "include_sparse=$sparse real=${t}s"
done
```

Expect `include_sparse=true` ≈ 2× the dense-only wall time (two encoder
forwards), NOT orders of magnitude more. A future optimization can fuse dense +
sparse into a single forward (pool + SPLADE head off the same hidden states);
they currently diverge only because the sparse path drops the document prefix.
```
