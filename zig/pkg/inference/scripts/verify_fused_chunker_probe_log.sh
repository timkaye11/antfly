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

set -euo pipefail

log_path="${1:-}"
if [[ -z "$log_path" || ! -f "$log_path" ]]; then
  echo "usage: verify_fused_chunker_probe_log.sh <log-file>" >&2
  exit 2
fi

if grep -Eiq '(^|[^[:alpha:]])nan([^[:alpha:]]|$)|(^|[^[:alpha:]])inf([^[:alpha:]]|$)|nonfinite' "$log_path"; then
  echo "probe failed: nonfinite value found in $log_path" >&2
  exit 1
fi

if ! grep -q 'segment_vjp_execution=mpsgraph_required' "$log_path"; then
  echo "probe failed: run did not advertise mpsgraph_required execution" >&2
  exit 1
fi

metrics_path="$(dirname -- "$log_path")/fused_training_metrics.jsonl"
if [[ -f "$metrics_path" ]]; then
  if ! grep -q '"vjp_runtime":"mpsgraph"' "$metrics_path"; then
    echo "probe failed: metrics did not record an MPSGraph VJP step" >&2
    exit 1
  fi
  if grep -Eq '"vjp_interpreter_fallbacks":[1-9][0-9]*' "$metrics_path"; then
    echo "probe failed: metrics reported interpreter fallbacks" >&2
    exit 1
  fi
else
  if ! grep -q 'runtime mpsgraph' "$log_path"; then
    echo "probe failed: no MPSGraph VJP profile line found" >&2
    exit 1
  fi

  if grep -E 'runtime mpsgraph .* fallbacks [1-9][0-9]*' "$log_path" >/dev/null; then
    echo "probe failed: MPSGraph profile reported interpreter fallbacks" >&2
    exit 1
  fi
fi

if ! grep -q 'training complete' "$log_path" && ! grep -q 'max steps reached' "$log_path"; then
  echo "probe failed: run did not complete or reach max steps" >&2
  exit 1
fi

echo "probe ok: $log_path"
