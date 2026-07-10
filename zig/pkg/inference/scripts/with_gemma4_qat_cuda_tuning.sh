#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/gemma4_qat_cuda_tuning.sh"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi

# The server resets the pinned-temp capture sequence whenever it provisions a
# new request-owned KV storage. Batching stays disabled so replay state cannot
# cross concurrently active requests.
antfly_decode_graph_replay="${ANTFLY_SERVER_DECODE_GRAPH_REPLAY:-required}"
gemma4_qat_cuda_tuning_env "${ANTFLY_CAPTURE_FORCE_KV_CAPACITY:-544}"
exec env "${GEMMA4_QAT_CUDA_ENV[@]}" \
  ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET=1 \
  ANTFLY_INFERENCE_DISABLE_CONTINUOUS_BATCHING=1 \
  "$@"
