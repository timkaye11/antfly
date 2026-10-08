#!/usr/bin/env bash
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
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
here="$(cd "$(dirname "$0")" && pwd)"
nvcc="${NVCC:-/usr/local/cuda-13.2/bin/nvcc}"
"$nvcc" -O3 -std=c++17 -shared -Xcompiler=-fPIC -arch=sm_89 \
  "$here/vision_attention_differential.cu" -lcublasLt -lcublas \
  -o /tmp/vision_lt_q1024_scaled.so
printf 'wrote %s\n' /tmp/vision_lt_q1024_scaled.so
