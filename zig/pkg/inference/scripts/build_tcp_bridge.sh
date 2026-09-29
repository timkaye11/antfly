#!/bin/sh
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)"
output_root="${ANTFLY_TCP_OUTPUT_ROOT:-${repo_root}/zig/pkg/inference/zig-out/lib}"
mkdir -p "${output_root}"
clang++ -std=c++20 -O2 -fPIC -dynamiclib \
  "${repo_root}/zig/pkg/inference/src/finetune/distributed/tcp_bridge.cpp" \
  -o "${output_root}/libantfly_tcp.dylib"
echo "${output_root}/libantfly_tcp.dylib"
