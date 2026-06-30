#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-/tmp/antfly-gemma4-qat-cuda-release-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT_DIR"

if [[ -n "${E4B_QAT_PROVIDER_BASELINE_JSON:-}" || -n "${E4B_QAT_PROVIDER_BASELINE_INLINE:-}" ]]; then
  RUN_E4B_QAT_PROVIDER_COMPARISON="${RUN_E4B_QAT_PROVIDER_COMPARISON:-required}"
else
  RUN_E4B_QAT_PROVIDER_COMPARISON="${RUN_E4B_QAT_PROVIDER_COMPARISON:-off}"
fi

if [[ -n "${E4B_QAT_PROVIDER_BASE_URL:-}" ]]; then
  RUN_E4B_QAT_PROVIDER_BENCHMARK="${RUN_E4B_QAT_PROVIDER_BENCHMARK:-required}"
  RUN_E4B_QAT_PROVIDER_COMPARISON="${RUN_E4B_QAT_PROVIDER_COMPARISON:-required}"
else
  RUN_E4B_QAT_PROVIDER_BENCHMARK="${RUN_E4B_QAT_PROVIDER_BENCHMARK:-off}"
fi

export OUT_DIR
export RUN_BUILD="${RUN_BUILD:-1}"
export RUN_SMOKE="${RUN_SMOKE:-1}"
export RUN_MICROBENCH="${RUN_MICROBENCH:-0}"
export RUN_DEFAULT_POLICY="${RUN_DEFAULT_POLICY:-0}"
export RUN_TARGET_ONLY="${RUN_TARGET_ONLY:-0}"
export RUN_E4B_QAT="${RUN_E4B_QAT:-required}"
export RUN_E4B_Q4K_BASELINE="${RUN_E4B_Q4K_BASELINE:-required}"
export RUN_E4B_QAT_LONG="${RUN_E4B_QAT_LONG:-required}"
export RUN_E4B_QAT_COMPRESSED_KV="${RUN_E4B_QAT_COMPRESSED_KV:-required}"
export RUN_E4B_QAT_COMPETITIVE_FLOOR="${RUN_E4B_QAT_COMPETITIVE_FLOOR:-required}"
export RUN_E4B_QAT_RESIDENT="${RUN_E4B_QAT_RESIDENT:-required}"
export RUN_E4B_QAT_RESIDENT_SOAK="${RUN_E4B_QAT_RESIDENT_SOAK:-required}"
export RUN_E4B_QAT_RESIDENT_BACKPRESSURE="${RUN_E4B_QAT_RESIDENT_BACKPRESSURE:-required}"
export RUN_E4B_Q4K_RESIDENT_BASELINE="${RUN_E4B_Q4K_RESIDENT_BASELINE:-required}"
export RUN_E4B_QAT_MTP="${RUN_E4B_QAT_MTP:-required}"
export RUN_E4B_QAT_MTP_TARGET_EQUIV="${RUN_E4B_QAT_MTP_TARGET_EQUIV:-required}"
export RUN_E4B_QAT_MTP_REPLAY_STABILITY="${RUN_E4B_QAT_MTP_REPLAY_STABILITY:-required}"
export RUN_E4B_QAT_MTP_REPLAY_512="${RUN_E4B_QAT_MTP_REPLAY_512:-required}"
export RUN_E4B_QAT_MTP_HIDDEN_AB="${RUN_E4B_QAT_MTP_HIDDEN_AB:-required}"
export RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX="${RUN_E4B_QAT_MTP_ACCEPTANCE_MATRIX:-required}"
export RUN_E4B_QAT_MTP_DONOR_MATRIX="${RUN_E4B_QAT_MTP_DONOR_MATRIX:-required}"
export RUN_E4B_QAT_MTP_BENEFIT="${RUN_E4B_QAT_MTP_BENEFIT:-off}"
export RUN_12B_MTP="${RUN_12B_MTP:-0}"
export RUN_E2B_MTP="${RUN_E2B_MTP:-0}"
export RUN_E4B_QAT_PROVIDER_BENCHMARK
export RUN_E4B_QAT_PROVIDER_COMPARISON
export RUN_TIMEOUT="${RUN_TIMEOUT:-900}"

export E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS="${E4B_QAT_RESIDENT_MAX_CONCURRENT_REQUESTS:-16}"
export E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS="${E4B_QAT_RESIDENT_BACKPRESSURE_REQUESTS:-8}"
export E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY="${E4B_QAT_RESIDENT_BACKPRESSURE_CONCURRENCY:-8}"
export E4B_QAT_COMPETITIVE_FLOORS="${E4B_QAT_COMPETITIVE_FLOORS:-compressed_kv_decode_tok_s=36.0}"
export E4B_QAT_COMPRESSED_KV_MIN_TOK_S="${E4B_QAT_COMPRESSED_KV_MIN_TOK_S:-36.0}"
export E4B_QAT_COMPRESSED_KV_MIN_TOKENS="${E4B_QAT_COMPRESSED_KV_MIN_TOKENS:-512}"
export E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS="${E4B_QAT_COMPRESSED_KV_MIN_GRAPH_REPLAYS:-auto}"
export E4B_QAT_COMPRESSED_KV_MAX_GRAPH_DISCARDS="${E4B_QAT_COMPRESSED_KV_MAX_GRAPH_DISCARDS:-1}"
export E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS="${E4B_QAT_COMPRESSED_KV_MAX_DOWNLOAD_SYNCS:-4}"
export E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS="${E4B_QAT_COMPRESSED_KV_MAX_CAPACITY_SKIPS:-0}"
export E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS="${E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_READS:-1}"
export E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES="${E4B_QAT_COMPRESSED_KV_MIN_COMPRESSED_V_WRITES:-1}"
export E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS="${E4B_QAT_COMPRESSED_KV_MIN_PAGED_UPLOADS:-1}"
export E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS="${E4B_QAT_COMPRESSED_KV_MIN_IDENTITY_ATTENTION_READS:-1}"
export E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA="${E4B_QAT_COMPRESSED_KV_MIN_FAST_GQA:-1}"
export E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES="${E4B_QAT_COMPRESSED_KV_MAX_FAIL_WRITES:-0}"
export E4B_QAT_MTP_TARGET_EQUIV_TOKENS="${E4B_QAT_MTP_TARGET_EQUIV_TOKENS:-16}"
export E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER="${E4B_QAT_MTP_TARGET_EQUIV_PROMPT_FILTER:-ants_chat factual_chat explain_chat code_chat}"
export E4B_QAT_MTP_HIDDEN_AB_REPEATS="${E4B_QAT_MTP_HIDDEN_AB_REPEATS:-2}"
export E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO="${E4B_QAT_MTP_HIDDEN_AB_MIN_RATIO:-1.03}"
export MTP_MIN_ACTIVE_SPEED_TOKENS="${MTP_MIN_ACTIVE_SPEED_TOKENS:-1}"

echo "gemma4_qat_cuda_release_gate_out_dir=$OUT_DIR"
scripts/gemma4_cuda_production_gate.sh

python3 - "$OUT_DIR" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
readiness_path = out_dir / "readiness.json"
summary_path = out_dir / "e4b_qat_production_summary.json"

readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
summary = json.loads(summary_path.read_text(encoding="utf-8"))
errors = []
if not readiness.get("ok"):
    errors.append("readiness.json ok=false")
if not summary.get("ok"):
    errors.append("e4b_qat_production_summary.json ok=false")
verdict = summary.get("verdict") or {}
if not verdict.get("ok"):
    errors.append("summary verdict ok=false")
compressed = ((summary.get("compressed_kv") or {}).get("qat") or {})
counter_mins = compressed.get("cuda_counter_mins") or {}
counter_maxs = compressed.get("cuda_counter_maxs") or {}
required_positive = {
    "graph_capture_persistent_replays": 1,
    "device_kv_compressed_v_reads": 1,
    "device_kv_compressed_v_writes": 1,
    "device_kv_paged_block_table_uploads": 1,
    "device_kv_paged_identity_attention_reads": 1,
    "launch_attention_gqa_decode_fast": 1,
    "gated_down_fused_q4_0_precompute": 1,
}
for name, floor in required_positive.items():
    value = counter_mins.get(name)
    if value is None or float(value) < floor:
        errors.append(f"{name} min={value} floor={floor}")
for name, ceiling in {
    "graph_capture_discards": 1,
    "graph_capture_capacity_skips": 0,
    "device_kv_fail_write": 0,
    "gated_down_fused_q4_0_tile4": 0,
}.items():
    value = counter_maxs.get(name)
    if value is None or float(value) > ceiling:
        errors.append(f"{name} max={value} ceiling={ceiling}")
replay_stability = summary.get("mtp_replay_stability") or {}
if not replay_stability.get("present") or not replay_stability.get("ok"):
    errors.append(
        "mtp_replay_stability missing_or_failed "
        f"present={replay_stability.get('present')} ok={replay_stability.get('ok')}"
    )
if int(replay_stability.get("row_count") or 0) <= 0:
    errors.append(f"mtp_replay_stability row_count={replay_stability.get('row_count')}")
target_equivalence = summary.get("mtp_target_equivalence") or {}
if not target_equivalence.get("present") or not target_equivalence.get("ok"):
    errors.append(
        "mtp_target_equivalence missing_or_failed "
        f"present={target_equivalence.get('present')} ok={target_equivalence.get('ok')} "
        f"errors={target_equivalence.get('errors')}"
    )
if int(target_equivalence.get("candidate_count") or 0) <= 0:
    errors.append(f"mtp_target_equivalence candidate_count={target_equivalence.get('candidate_count')}")
if int(target_equivalence.get("target_case_count") or 0) < 4:
    errors.append(f"mtp_target_equivalence target_case_count={target_equivalence.get('target_case_count')}")
replay_512 = summary.get("mtp_replay_512") or {}
if not replay_512.get("present"):
    errors.append("mtp_replay_512 missing")
if int(replay_512.get("candidate_count") or 0) <= 0:
    errors.append(f"mtp_replay_512 candidate_count={replay_512.get('candidate_count')}")
hidden_ab = summary.get("mtp_hidden_ab") or {}
if not hidden_ab.get("present"):
    errors.append("mtp_hidden_ab missing")
if int(hidden_ab.get("pair_count") or 0) < 2:
    errors.append(f"mtp_hidden_ab pair_count={hidden_ab.get('pair_count')}")
if not hidden_ab.get("default_hidden_selector_disabled"):
    errors.append("mtp_hidden_ab default selector unexpectedly enabled")
if not hidden_ab.get("hidden_selector_counters_ok"):
    errors.append("mtp_hidden_ab hidden selector counters failed")
if hidden_ab.get("promotion_recommendation") == "fail":
    errors.append("mtp_hidden_ab promotion recommendation failed")
acceptance_matrix = summary.get("mtp_acceptance_matrix") or {}
if not acceptance_matrix.get("present"):
    errors.append("mtp_acceptance_matrix missing")
if int(acceptance_matrix.get("combo_count") or 0) < 8:
    errors.append(f"mtp_acceptance_matrix combo_count={acceptance_matrix.get('combo_count')}")
if not acceptance_matrix.get("best_acceptance"):
    errors.append("mtp_acceptance_matrix best_acceptance missing")
donor_matrix = summary.get("mtp_donor_matrix") or {}
if not donor_matrix.get("present"):
    errors.append("mtp_donor_matrix missing")
if int(donor_matrix.get("combo_count") or 0) < 2:
    errors.append(f"mtp_donor_matrix combo_count={donor_matrix.get('combo_count')}")
if not donor_matrix.get("best_acceptance"):
    errors.append("mtp_donor_matrix best_acceptance missing")
if errors:
    for error in errors:
        print(f"release_gate_failure: {error}", file=sys.stderr)
    raise SystemExit(1)
print(
    "release_gate_ok "
    f"compressed_kv_tok_s={compressed.get('avg_decode_tok_per_s')} "
    f"out_dir={out_dir}"
)
PY
