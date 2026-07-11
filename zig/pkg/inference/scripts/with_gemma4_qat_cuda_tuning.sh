#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/gemma4_qat_cuda_tuning.sh"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARGS...]" >&2
  exit 2
fi

# The server resets the pinned-temp capture sequence whenever it provisions a
# new request-owned KV storage. Batching stays disabled by default so replay
# state cannot cross concurrently active requests. The batching gate sets this
# to 0 together with graph replay off.
case "${ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING:-1}" in
  1|true|yes|on)
    disable_continuous_batching=1
    ;;
  0|false|no|off)
    disable_continuous_batching=0
    ;;
  *)
    echo "ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING must be a boolean" >&2
    exit 2
    ;;
esac
antfly_decode_graph_replay="${ANTFLY_SERVER_DECODE_GRAPH_REPLAY:-required}"
gemma4_qat_cuda_tuning_env "${ANTFLY_CAPTURE_FORCE_KV_CAPACITY:-544}"
exec env "${GEMMA4_QAT_CUDA_ENV[@]}" \
  ANTFLY_INFERENCE_CUDA_SERVER_REQUEST_GRAPH_RESET=1 \
  ANTFLY_INFERENCE_DISABLE_CONTINUOUS_BATCHING="$disable_continuous_batching" \
  "$@"
