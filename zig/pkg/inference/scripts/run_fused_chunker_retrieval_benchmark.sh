#!/usr/bin/env bash
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Retrieval-NDCG benchmark lane for the fused chunker release gate
# (evals/chunker/manifest.json, retrieval_ndcg). Orchestrates, per domain:
#
#   datasets     Materialize open BEIR/LongEmbed corpora + queries-with-qrels
#                JSONL via generate_fused_chunker_retrieval_datasets.py (CPU +
#                network only; safe to run without a GPU).
#   materialize  Normalize candidate/baseline /chunk-response JSONL into strict
#                embedding JSONL (materialize-fused-chunker-retrieval-embeddings).
#                GPU-produced inputs required (see "Embedding inputs" below).
#   rank         rank-fused-chunker-retrieval-benchmark for every mode + baseline.
#   score        score-fused-chunker-retrieval-benchmark: contextual dense+SPLADE
#                run vs local baselines, per domain, at --output-dimension.
#   aggregate    aggregate-fused-chunker-benchmark: merge per-domain retrieval
#                results into one fused_chunker_benchmark_results/v1 file.
#
# Stages are comma-selectable and run independently. Run the CPU-only dataset
# stage anywhere:
#   ANTFLY_FUSED_CHUNKER_RETRIEVAL_STAGES=datasets \
#   scripts/run_fused_chunker_retrieval_benchmark.sh
# then, once a trained checkpoint has produced embeddings on a free GPU:
#   ANTFLY_FUSED_CHUNKER_RETRIEVAL_STAGES=materialize,rank,score,aggregate \
#   scripts/run_fused_chunker_retrieval_benchmark.sh
#
# Embedding inputs (GPU-produced, consumed by the materialize stage)
# --------------------------------------------------------------------
# The materialize tool consumes raw /chunk responses (or wrappers/embedding
# records), NOT corpus text. For each domain, generate the following by POSTing
# the domain's corpus.jsonl / queries.jsonl text to the inference /chunk
# endpoint with the candidate chunker model and config
#   {"include_embeddings": true, "include_sparse": true,
#    "output_dimension": <dim>}
# then wrapping each response so the tool can recover ids:
#
#   $embed_in/<domain>/<variant>.docs.chunk_responses.jsonl
#     one line per corpus document, either
#       {"document_id":"<id>","response":{"data":[<chunk>...]}}   (wrapper) or
#       {"document_id":"<id>","data":[<chunk>...]}
#   $embed_in/<domain>/queries.chunk_responses.jsonl
#     one line per query, carrying the DOCUMENT-level qrels verbatim:
#       {"query_id":"<qid>","relevant_ids":["<document_id>",...],
#        "data":[<chunk>...]}     (query uses first chunk's embedding)
#
# <variant> is one of the ranked lanes/baselines below. Each variant differs
# only in how the corpus was chunked before embedding:
#   contextual   contextual chunk embeddings from the candidate checkpoint
#   fixed_500_50 fixed 500-token / 50-overlap chunks, same encoder
#   chonky       chonky-boundary chunks, same encoder
# single_embedding, splade_only, dense_only_contextual and dense_splade_rrf all
# reuse the contextual variant's embeddings (only the ranking mode differs).
#
# Scoring is DOCUMENT-level: ranked chunk ids ("<doc>#chunk:<n>") are collapsed
# to their document id before scoring against document-level relevant_ids.
# Chunk-level NDCG is intentionally NOT computed here because open BEIR/LongEmbed
# qrels are document-level; per-chunk relevance judgments do not exist.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pkg_root="$(cd -- "$script_dir/.." && pwd)"

zig_bin="${ANTFLY_ZIG_BIN:-}"
if [[ -z "$zig_bin" ]]; then
  if command -v zig >/dev/null 2>&1; then
    zig_bin="$(command -v zig)"
  elif [[ -x "$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig" ]]; then
    zig_bin="$HOME/.local/share/zigup/0.16.0-dev.3144+ac6fb0b59/files/zig"
  else
    echo "missing Zig compiler; set ANTFLY_ZIG_BIN=/path/to/zig" >&2
    exit 1
  fi
fi

python_bin="${ANTFLY_PYTHON_BIN:-python3}"

stages="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_STAGES:-datasets,materialize,rank,score,aggregate}"
domains="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_DOMAINS:-technical_docs web code medical conversation law finance long_context}"
data_dir="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_DATA_DIR:-/private/tmp/fused_retrieval_datasets}"
out_dir="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_OUT:-/private/tmp/fused-chunker-retrieval-benchmark}"
embed_in="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_EMBED_IN:-$out_dir/embeddings_in}"
manifest_path="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_MANIFEST:-$pkg_root/evals/chunker/manifest.json}"

output_dimension="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_OUTPUT_DIMENSION:-256}"
top_k="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_TOP_K:-100}"
rrf_k="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_RRF_K:-60}"

# dataset caps (laptop-friendly; see generator header)
max_docs="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_MAX_DOCS:-5000}"
max_queries="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_MAX_QUERIES:-500}"
max_doc_chars="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_MAX_DOC_CHARS:-60000}"
seed="${ANTFLY_FUSED_CHUNKER_RETRIEVAL_SEED:-42}"

# Baselines: name -> embedding variant that supplies its doc embeddings.
baseline_variants=(
  "fixed_500_50_same_encoder:fixed_500_50:hybrid"
  "chonky_boundaries_same_encoder:chonky:hybrid"
  "dense_only_contextual_chunks:contextual:dense"
  "splade_only:contextual:sparse"
  "dense_splade_rrf:contextual:hybrid"
)

prepared_dir="$data_dir"
materialized_dir="$out_dir/materialized"
runs_dir="$out_dir/runs"
results_dir="$out_dir/results"
aggregate_results="$out_dir/retrieval.fused_chunker_benchmark_results.json"

has_stage() {
  case ",$stages," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

zig_tool() {
  (
    cd "$pkg_root"
    "$zig_bin" build -Dskip-openapi=true -Doptimize=ReleaseFast "$@"
  )
}

# Collapse chunk-level ranked ids to document ids (split on '#chunk:'/'#'/':'),
# dedup keeping best rank, so document-level relevant_ids score correctly.
collapse_run_to_documents() {
  local in_path="$1" out_path="$2"
  "$python_bin" - "$in_path" "$out_path" <<'PY'
import json, re, sys
src, dst = sys.argv[1], sys.argv[2]
def doc_of(cid):
    return re.split(r"#chunk:|#|:", cid, maxsplit=1)[0]
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        relevant = rec.get("relevant_ids") or rec.get("relevant") or []
        ranked = rec.get("ranked_ids")
        if ranked is None:
            hits = rec.get("results") or rec.get("hits") or []
            ranked = [h.get("id") or h.get("doc_id") or h.get("chunk_id") for h in hits]
        seen, collapsed = set(), []
        for cid in ranked:
            did = doc_of(cid)
            if did not in seen:
                seen.add(did)
                collapsed.append(did)
        fout.write(json.dumps({
            "query_id": rec.get("query_id") or rec.get("id"),
            "relevant_ids": relevant,
            "ranked_ids": collapsed,
        }) + "\n")
PY
}

echo "fused chunker retrieval-NDCG benchmark lane"
echo "  manifest=$manifest_path"
echo "  domains=$domains"
echo "  stages=$stages"
echo "  data_dir=$data_dir"
echo "  embed_in=$embed_in"
echo "  out_dir=$out_dir"
echo "  output_dimension=$output_dimension top_k=$top_k rrf_k=$rrf_k"

if [[ ! -f "$manifest_path" ]]; then
  echo "missing benchmark manifest at $manifest_path" >&2
  exit 1
fi

mkdir -p "$materialized_dir" "$runs_dir" "$results_dir"

if has_stage datasets; then
  echo "== stage: datasets (CPU + network) =="
  domains_csv="$(echo "$domains" | tr ' ' ',')"
  "$python_bin" "$script_dir/generate_fused_chunker_retrieval_datasets.py" \
    --out-dir "$data_dir" \
    --domains "$domains_csv" \
    --max-docs "$max_docs" \
    --max-queries "$max_queries" \
    --max-doc-chars "$max_doc_chars" \
    --seed "$seed"
fi

if has_stage materialize; then
  echo "== stage: materialize =="
  for domain in $domains; do
    dom_out="$materialized_dir/$domain"
    mkdir -p "$dom_out"
    q_in="$embed_in/$domain/queries.chunk_responses.jsonl"
    if [[ ! -f "$q_in" ]]; then
      echo "missing query embedding input at $q_in" >&2
      echo "generate it from $prepared_dir/$domain/queries.jsonl via the /chunk endpoint" >&2
      echo "(include_embeddings=true, include_sparse=true, output_dimension=$output_dimension)" >&2
      exit 1
    fi
    echo "-- materialize $domain queries"
    zig_tool materialize-fused-chunker-retrieval-embeddings -- \
      --input "$q_in" \
      --out "$dom_out/queries.embeddings.jsonl" \
      --kind queries \
      --output-dimension "$output_dimension"
    for variant in contextual fixed_500_50 chonky; do
      d_in="$embed_in/$domain/$variant.docs.chunk_responses.jsonl"
      if [[ ! -f "$d_in" ]]; then
        echo "missing doc embedding input at $d_in" >&2
        echo "generate it by chunking $prepared_dir/$domain/corpus.jsonl with the $variant strategy" >&2
        echo "and embedding via /chunk (include_embeddings=true, include_sparse=true)" >&2
        exit 1
      fi
      echo "-- materialize $domain docs/$variant"
      zig_tool materialize-fused-chunker-retrieval-embeddings -- \
        --input "$d_in" \
        --out "$dom_out/$variant.docs.embeddings.jsonl" \
        --kind docs \
        --output-dimension "$output_dimension"
    done
  done
fi

if has_stage rank; then
  echo "== stage: rank =="
  for domain in $domains; do
    dom_mat="$materialized_dir/$domain"
    dom_runs="$runs_dir/$domain"
    mkdir -p "$dom_runs"
    q_emb="$dom_mat/queries.embeddings.jsonl"
    ctx_docs="$dom_mat/contextual.docs.embeddings.jsonl"
    if [[ ! -f "$q_emb" || ! -f "$ctx_docs" ]]; then
      echo "missing materialized embeddings for $domain (run the materialize stage first)" >&2
      exit 1
    fi
    # Primary contextual dense+SPLADE hybrid run (the gated lane).
    echo "-- rank $domain dense_splade_hybrid"
    zig_tool rank-fused-chunker-retrieval-benchmark -- \
      --queries "$q_emb" --docs "$ctx_docs" \
      --out "$dom_runs/dense_splade_hybrid.chunk_run.jsonl" \
      --mode hybrid --top-k "$top_k" --rrf-k "$rrf_k"
    collapse_run_to_documents \
      "$dom_runs/dense_splade_hybrid.chunk_run.jsonl" \
      "$dom_runs/dense_splade_hybrid.run.jsonl"
    # Manifest modes on the contextual embeddings (diagnostic, same encoder).
    zig_tool rank-fused-chunker-retrieval-benchmark -- \
      --queries "$q_emb" --docs "$ctx_docs" \
      --out "$dom_runs/single_embedding.chunk_run.jsonl" \
      --mode dense --top-k "$top_k"
    collapse_run_to_documents \
      "$dom_runs/single_embedding.chunk_run.jsonl" \
      "$dom_runs/single_embedding.run.jsonl"
    # Local baselines.
    for entry in "${baseline_variants[@]}"; do
      name="${entry%%:*}"; rest="${entry#*:}"
      variant="${rest%%:*}"; mode="${rest#*:}"
      docs="$dom_mat/$variant.docs.embeddings.jsonl"
      if [[ ! -f "$docs" ]]; then
        echo "missing $variant embeddings for baseline $name in $domain" >&2
        exit 1
      fi
      echo "-- rank $domain baseline $name (variant=$variant mode=$mode)"
      zig_tool rank-fused-chunker-retrieval-benchmark -- \
        --queries "$q_emb" --docs "$docs" \
        --out "$dom_runs/$name.chunk_run.jsonl" \
        --mode "$mode" --top-k "$top_k" --rrf-k "$rrf_k"
      collapse_run_to_documents \
        "$dom_runs/$name.chunk_run.jsonl" \
        "$dom_runs/$name.run.jsonl"
    done
  done
fi

if has_stage score; then
  echo "== stage: score =="
  for domain in $domains; do
    dom_runs="$runs_dir/$domain"
    primary="$dom_runs/dense_splade_hybrid.run.jsonl"
    if [[ ! -f "$primary" ]]; then
      echo "missing primary run at $primary (run the rank stage first)" >&2
      exit 1
    fi
    score_args=(
      --run "$primary"
      --out "$results_dir/$domain.retrieval.fused_chunker_benchmark_results.json"
      --output-dimension "$output_dimension"
    )
    for entry in "${baseline_variants[@]}"; do
      name="${entry%%:*}"
      run="$dom_runs/$name.run.jsonl"
      if [[ ! -f "$run" ]]; then
        echo "missing baseline run at $run (run the rank stage first)" >&2
        exit 1
      fi
      score_args+=(--baseline "$name=$run")
    done
    echo "-- score $domain"
    zig_tool score-fused-chunker-retrieval-benchmark -- "${score_args[@]}"
  done
fi

if has_stage aggregate; then
  echo "== stage: aggregate =="
  # aggregate-fused-chunker-benchmark merges BOTH lanes and requires boundary
  # evidence, so it only produces the combined release file. Here we emit a
  # retrieval-only fused_chunker_benchmark_results/v1 file (query-weighted, the
  # same math the zig aggregator applies to the retrieval lane) that
  # `verify-fused-chunker-benchmark --lane retrieval_ndcg` accepts. For the full
  # release file, run aggregate-fused-chunker-benchmark with the per-dataset
  # boundary results AND these per-domain retrieval results together.
  agg_inputs=()
  for domain in $domains; do
    results_path="$results_dir/$domain.retrieval.fused_chunker_benchmark_results.json"
    if [[ ! -f "$results_path" ]]; then
      echo "missing per-domain results at $results_path (run the score stage first)" >&2
      exit 1
    fi
    agg_inputs+=("$results_path")
  done
  "$python_bin" - "$aggregate_results" "${agg_inputs[@]}" <<'PY'
import json, sys
out_path, inputs = sys.argv[1], sys.argv[2:]
ndcg_sum = w_sum = target_sum = target_w = 0.0
r1_sum = r1_w = r10_sum = r10_w = mrr_sum = mrr_w = 0.0
queries_total = 0
dims = set()
baselines = {}  # name -> [weighted_ndcg_sum, weight_sum]
for path in inputs:
    lane = json.load(open(path))["retrieval_ndcg"]
    ndcg = lane.get("overall_ndcg_at_10", lane.get("ndcg_at_10"))
    q = lane.get("queries") or 1
    w = float(max(q, 1))
    ndcg_sum += ndcg * w; w_sum += w; queries_total += q
    tgt = lane.get("voyage_context_4_target_ndcg_at_10")
    if tgt is not None:
        target_sum += tgt * w; target_w += w
    if lane.get("recall_at_1") is not None:
        r1_sum += lane["recall_at_1"] * w; r1_w += w
    if lane.get("recall_at_10") is not None:
        r10_sum += lane["recall_at_10"] * w; r10_w += w
    if lane.get("mrr") is not None:
        mrr_sum += lane["mrr"] * w; mrr_w += w
    if lane.get("output_dimension") is not None:
        dims.add(lane["output_dimension"])
    for b in lane.get("baselines", []):
        bn = b.get("overall_ndcg_at_10", b.get("ndcg_at_10"))
        acc = baselines.setdefault(b["name"], [0.0, 0.0])
        acc[0] += bn * w; acc[1] += w
overall = ndcg_sum / w_sum
out_baselines = [{"name": n, "overall_ndcg_at_10": s / wt}
                 for n, (s, wt) in sorted(baselines.items())]
best = max((b["overall_ndcg_at_10"] for b in out_baselines), default=None)
dim = dims.pop() if len(dims) == 1 else None
target = (target_sum / target_w) if target_w > 0 else None
lane = {
    "output_dimension": dim,
    "overall_ndcg_at_10": overall,
    "voyage_context_4_target_ndcg_at_10": target,
    "distance_to_voyage_context_4_target": (target - overall) if target is not None else None,
    "best_local_baseline_ndcg_at_10": best,
    "relative_gain_over_best_local_baseline": ((overall - best) / best) if best else None,
    "recall_at_1": (r1_sum / r1_w) if r1_w > 0 else None,
    "recall_at_10": (r10_sum / r10_w) if r10_w > 0 else None,
    "mrr": (mrr_sum / mrr_w) if mrr_w > 0 else None,
    "queries": queries_total,
    "baselines": out_baselines,
}
json.dump({"schema_version": "fused_chunker_benchmark_results/v1",
           "retrieval_ndcg": lane}, open(out_path, "w"), indent=2)
print(f"aggregated {len(inputs)} domains: overall_ndcg@10={overall:.4f} "
      f"best_baseline={best} rel_gain={lane['relative_gain_over_best_local_baseline']}")
PY
  echo "aggregated retrieval results -> $aggregate_results"
  echo "verify the retrieval lane with:"
  echo "  $zig_bin build -Dskip-openapi=true verify-fused-chunker-benchmark -- \\"
  echo "    --manifest $manifest_path --results $aggregate_results --lane retrieval_ndcg"
fi
