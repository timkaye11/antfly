# GPU Validation Runbook — Compiled Segment Forward in /chunk Serving

Validates the compiled MPSGraph segment forward wired into the fused
chunker's SERVING path (`src/pipelines/fused_chunking.zig`,
branch `serving_compiled_forward`):

- default ON for the Metal backend, over the MERGED export (no LoRA
  injection: `CompiledEvalForward.initMergedBase` → rank-0
  `ModernBertSegmentForwardSession`s, base weights only)
- exactly-full windows (batch=1, seq=max_seq_len) run compiled; the ragged
  final window and any post-failure window run the unchanged eager forward
- kill switch: `ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD=0` forces eager
- segment sessions cold-compile at pipeline load (expect a few extra seconds
  of model load and a `compiled serving segment forward enabled` log line)

**Do not run while the MTCB benchmark sweep is active** (tmux session
`mtcb`, server on port 8099, CPU-heavy). Wait for it to finish; never reuse
its port or tmux session. Use a spare port (8123 below).

Expected outcome: identical chunk spans, boundary-score deltas below
threshold granularity (the documented ~2e-3 compiled-vs-eager cross-kernel
noise class at the feature level), and roughly 6–9x on the per-window
encoder forward (~276 ms/window eager at seq 384 → tens of ms compiled),
i.e. a ~2 min 158 KB boundary-only document dropping to ~15–25 s.

CPU-side verification already done (no GPU required, re-runnable any time):

```bash
cd zig/pkg/inference
zig build -Dskip-openapi=true test-bin                              # non-Metal build+link green
zig build -Dskip-openapi=true -Dmetal=true test-bin                 # Metal build green
zig build -Dskip-openapi=true test-fused-chunker-compiled-forward   # policy tests (kill switch, window gating)
zig build -Dskip-openapi=true test-fused-chunker-train              # CompiledEvalForward tests incl. initMergedBase
zig build -Dskip-openapi=true test-fused-chunker-pipeline           # serving pipeline vs smoke export, native CPU
zig build -Dskip-openapi=true -Druntime-test-filter test -- \
  --test-filter fused_chunking --test-filter forwardFromHidden      # forwardFromHidden head-parity tests
```

## 0. Pick a packaged model dir and a real ~150 KB document

Any export produced by `export_fused_chunker_model` works (needs
`model.safetensors`, `tokenizer.json`, `config.json`, `model_manifest.json`).
The smoke fixture at `/private/tmp/fused_export_smoke` is complete; prefer
the current release export if one is packaged.

```bash
export MODEL_DIR=/private/tmp/fused_export_smoke
ls "$MODEL_DIR"   # model.safetensors tokenizer.json config.json model_manifest.json

# ~150 KB of real prose (concatenate docs; avoid binary):
export DOC=/private/tmp/fused_serving_ab/doc_150k.txt
mkdir -p /private/tmp/fused_serving_ab
cat zig/pkg/inference/docs/*.md docs/*.mdx 2>/dev/null | head -c 158000 > "$DOC"
wc -c "$DOC"
```

## 1. Direct-pipeline latency A/B (bench tool, no HTTP)

The e2e bench drives `chunkTextTimed` directly and reports the per-phase
`forward_avg_ms`, which is where the whole win lives. 32768 tokens is the
serving cap (85 full windows + 1 ragged at max_seq_len 384).

```bash
cd zig/pkg/inference

# Eager baseline (kill switch ON):
ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD=0 \
zig build -Dskip-openapi=true -Dmetal=true bench-fused-chunker-e2e -- \
  --model-dir "$MODEL_DIR" --backend metal --mode both \
  --doc-tokens 2048,8192,32768 --iterations 3 \
  --results-out /private/tmp/fused_serving_ab/bench_eager.json

# Compiled (default ON — expect the load log:
# "fused chunker: compiled serving segment forward enabled (batch=1 seq=384 ...)"):
zig build -Dskip-openapi=true -Dmetal=true bench-fused-chunker-e2e -- \
  --model-dir "$MODEL_DIR" --backend metal --mode both \
  --doc-tokens 2048,8192,32768 --iterations 3 \
  --results-out /private/tmp/fused_serving_ab/bench_compiled.json

# Speedup + invariants (same windows and chunk counts row-for-row):
python3 - <<'EOF'
import json
eager = json.load(open("/private/tmp/fused_serving_ab/bench_eager.json"))
comp  = json.load(open("/private/tmp/fused_serving_ab/bench_compiled.json"))
for e, c in zip(eager["rows"], comp["rows"]):
    assert (e["doc_tokens"], e["mode"]) == (c["doc_tokens"], c["mode"])
    assert e["windows"] == c["windows"], (e, c)
    assert e["chunks"] == c["chunks"], f"chunk count changed: {e} vs {c}"
    speedup = e["forward_avg_ms"] / max(c["forward_avg_ms"], 1e-9)
    print(f'{e["doc_tokens"]:>6} tok {e["mode"]:<8} forward {e["forward_avg_ms"]:9.1f} -> '
          f'{c["forward_avg_ms"]:9.1f} ms  ({speedup:.1f}x)  e2e {e["e2e_avg_ms"]:9.1f} -> {c["e2e_avg_ms"]:9.1f} ms')
print("PASS: chunk counts identical; record the forward speedup (expect ~6-9x on full windows)")
EOF
```

Also note `cold_load_ms` in the compiled results: the delta vs eager is the
one-time 22-segment MPSGraph compile paid at load (by design, so the first
request isn't slow). Peak RSS grows by roughly one host-side f32 copy of the
encoder weights (~450 MB) from the per-session base-weight feed cache — the
same cost the eval compiled forward pays.

## 2. HTTP /chunk boundary-identity A/B (spare port, AFTER the sweep)

```bash
# Install the packaged model under a scratch models dir:
export MDIR=/private/tmp/fused_serving_ab/models
mkdir -p "$MDIR/chunkers"
ln -sfn "$MODEL_DIR" "$MDIR/chunkers/fused-serving-ab"

cd zig/pkg/inference
zig build -Dskip-openapi=true -Dmetal=true    # builds zig-out/bin/antfly-inference

# Request body (boundary-only, all chunks, with scores):
python3 - <<'EOF'
import json
doc = open("/private/tmp/fused_serving_ab/doc_150k.txt").read()
body = {"input": doc, "config": {"model": "fused-serving-ab",
        "max_chunks": 0, "include_boundary_scores": True}}
json.dump(body, open("/private/tmp/fused_serving_ab/req.json", "w"))
EOF

# --- Run A: eager (kill switch) ---
ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD=0 \
ANTFLY_INFERENCE_MODELS_DIR="$MDIR" \
./zig-out/bin/antfly-inference --port 8123 > /private/tmp/fused_serving_ab/server_eager.log 2>&1 &
SRV=$!; sleep 5
# warm-up (model load happens on first request), then timed run:
curl -s -o /dev/null -X POST localhost:8123/chunk -H 'content-type: application/json' \
  --data-binary @/private/tmp/fused_serving_ab/req.json
time curl -s -X POST localhost:8123/chunk -H 'content-type: application/json' \
  --data-binary @/private/tmp/fused_serving_ab/req.json \
  -o /private/tmp/fused_serving_ab/resp_eager.json
kill $SRV; wait $SRV 2>/dev/null

# --- Run B: compiled (default) ---
ANTFLY_INFERENCE_MODELS_DIR="$MDIR" \
./zig-out/bin/antfly-inference --port 8123 > /private/tmp/fused_serving_ab/server_compiled.log 2>&1 &
SRV=$!; sleep 5
curl -s -o /dev/null -X POST localhost:8123/chunk -H 'content-type: application/json' \
  --data-binary @/private/tmp/fused_serving_ab/req.json
time curl -s -X POST localhost:8123/chunk -H 'content-type: application/json' \
  --data-binary @/private/tmp/fused_serving_ab/req.json \
  -o /private/tmp/fused_serving_ab/resp_compiled.json
kill $SRV; wait $SRV 2>/dev/null

grep "compiled serving segment forward" /private/tmp/fused_serving_ab/server_compiled.log
grep "disabled via"                      /private/tmp/fused_serving_ab/server_eager.log
grep -i "falling back to eager"          /private/tmp/fused_serving_ab/server_compiled.log && echo "LATCHED (investigate)" || true

# --- Assert identical chunk spans + boundary-score tolerance ---
python3 - <<'EOF'
import json
a = json.load(open("/private/tmp/fused_serving_ab/resp_eager.json"))["chunks"]
b = json.load(open("/private/tmp/fused_serving_ab/resp_compiled.json"))["chunks"]
assert len(a) == len(b), f"chunk count differs: {len(a)} vs {len(b)}"
span = lambda c: (c["start_char"], c["end_char"], c.get("start_token"), c.get("end_token"))
mism = [(i, span(x), span(y)) for i, (x, y) in enumerate(zip(a, b)) if span(x) != span(y)]
assert not mism, f"SPANS DIFFER: {mism[:5]}"
deltas = [abs(x["boundary_score"] - y["boundary_score"]) for x, y in zip(a, b)]
print(f"PASS: {len(a)} chunks, spans identical; max boundary-score delta {max(deltas):.2e} (expect < ~1e-3)")
EOF
```

Record the two `time` results; the boundary-only /chunk on the 150 KB doc
should drop from ~2 min to ~15–25 s (forward is ~6–9x; tokenize/decode are
unchanged).

## 3. Embedding cosine check (fused-native dense path)

The split-encoder embedder path is unaffected (chunk embeddings come from
the separate frozen embed-base session, which does not use the fused
encoder). Only the fused model's OWN token-embedding head consumes the
compiled encoder features, so check that path explicitly:

```bash
python3 - <<'EOF'
import json
doc = open("/private/tmp/fused_serving_ab/doc_150k.txt").read()
body = {"input": doc, "config": {"model": "fused-serving-ab",
        "max_chunks": 0, "include_embeddings": True, "include_boundary_scores": True}}
json.dump(body, open("/private/tmp/fused_serving_ab/req_emb.json", "w"))
EOF
# repeat the Run A / Run B server dance above with req_emb.json ->
#   resp_emb_eager.json / resp_emb_compiled.json, then:
python3 - <<'EOF'
import json, math
a = json.load(open("/private/tmp/fused_serving_ab/resp_emb_eager.json"))["chunks"]
b = json.load(open("/private/tmp/fused_serving_ab/resp_emb_compiled.json"))["chunks"]
assert len(a) == len(b)
worst = 1.0
for x, y in zip(a, b):
    u, v = x["embedding"], y["embedding"]
    dot = sum(p*q for p, q in zip(u, v))
    nu, nv = math.sqrt(sum(p*p for p in u)), math.sqrt(sum(q*q for q in v))
    worst = min(worst, dot / (nu * nv))
print(f"PASS: min per-chunk cosine {worst:.6f} (require >= 0.999; typically >= 0.99999)")
EOF
```

## 4. Kill switch + fallback sanity

- Step 2 Run A already proves `ANTFLY_FUSED_CHUNKER_SERVING_COMPILED_FORWARD=0`
  forces eager (log: `compiled serving forward disabled via ...`).
- Fallback latch: the compiled path logs
  `compiled serving forward failed (...); falling back to eager forward`
  exactly once and serves every later window eagerly. If Run B's log shows
  the latch fired, the run is still CORRECT (eager numerics) but the perf
  win is gone — capture the error name and file it.

## Pass criteria

1. Load log shows the compiled forward enabled; no fallback latch during runs.
2. Chunk spans byte-identical (char and token) between eager and compiled on
   the 150 KB doc, boundary-only and with embeddings.
3. Max boundary-score delta < ~1e-3 (threshold granularity).
4. Min per-chunk embedding cosine >= 0.999 on the fused-native dense path.
5. Bench `forward_avg_ms` speedup ~6–9x on 8192/32768-token docs; e2e
   boundary-only 150 KB latency ~15–25 s vs ~2 min eager.
6. Kill switch verifiably restores today's eager behavior.
