#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: build_zig_release_archive.sh --version VERSION --target TARGET --archive-name NAME --out-dir DIR [--metal true|false] [--system-blas true|false] [--onnx true|false] [--optimize MODE] [--jobs N]

  --onnx  Enable the ONNX Runtime backend for embedded inference (default: true).
          ORT libraries are auto-provisioned via
          zig/pkg/inference/scripts/download-onnxruntime.sh. Upstream ORT is
          glibc/macOS only; for *-linux-musl targets ORT is skipped with a
          warning (the binary still builds, without ORT).

Builds the native Antfly Zig runtime and writes a release archive whose root
contains:
  antfly
  share/
  lib/
  include/
  README.md
  LICENSE
EOF
}

version=
target=
archive_name=
out_dir=
metal=false
system_blas=false
optimize=ReleaseFast
jobs=
# Production/release builds enable ONNX Runtime so the embedded inference server
# serves imported-ONNX models (e.g. SigLIP2@512 image tower) through ORT
# (~0.30 s/img) instead of the native interpreter (~17 s/img). Dev builds are
# unaffected: they use zig/Makefile or `zig build` directly, where -Donnx
# defaults OFF (zig/build.zig). Overridable with --onnx false.
onnx=true

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)
      version="${2:?missing --version value}"
      shift 2
      ;;
    --target)
      target="${2:?missing --target value}"
      shift 2
      ;;
    --archive-name)
      archive_name="${2:?missing --archive-name value}"
      shift 2
      ;;
    --out-dir)
      out_dir="${2:?missing --out-dir value}"
      shift 2
      ;;
    --metal)
      metal="${2:?missing --metal value}"
      shift 2
      ;;
    --system-blas)
      system_blas="${2:?missing --system-blas value}"
      shift 2
      ;;
    --onnx)
      onnx="${2:?missing --onnx value}"
      shift 2
      ;;
    --optimize)
      optimize="${2:?missing --optimize value}"
      shift 2
      ;;
    --jobs)
      jobs="${2:?missing --jobs value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$version" ] || [ -z "$target" ] || [ -z "$archive_name" ] || [ -z "$out_dir" ]; then
  usage
  exit 2
fi

if [ -n "$jobs" ] && ! [[ "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  usage
  echo "--jobs must be a positive integer, got: $jobs" >&2
  exit 2
fi

case "$optimize" in
  Debug|ReleaseSafe|ReleaseFast|ReleaseSmall) ;;
  *)
    usage
    echo "--optimize must be one of Debug, ReleaseSafe, ReleaseFast, ReleaseSmall; got: $optimize" >&2
    exit 2
    ;;
esac

case "$onnx" in
  true|false) ;;
  *)
    usage
    echo "--onnx must be true or false; got: $onnx" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/antfly-zig-release-${target}"
prefix="${work_root}/zig-out"
stage="${work_root}/stage"
local_cache="${work_root}/zig-cache"
cache_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}/zig-cache"

if [ -d /mnt/cache ] && [ -w /mnt/cache ]; then
  cache_root=/mnt/cache/zig
fi

lite_library_name() {
  case "$1" in
    *macos*) echo "libantfly.dylib" ;;
    *windows*) echo "antfly.dll" ;;
    *) echo "libantfly.so" ;;
  esac
}

<<<<<<< HEAD
# ---------------------------------------------------------------------------
# ONNX Runtime provisioning.
#
# When --onnx is enabled (default for release/production), download the pinned
# ORT libraries (via zig/pkg/inference/scripts/download-onnxruntime.sh, the same
# vendored fetch used by go/Dockerfile.omni) into the location the Zig build
# discovers by default: zig/pkg/inference/onnxruntime/<os>-<arch>. The embedded
# inference graph then links libonnxruntime and serves imported-ONNX models
# through ORT.
#
# Upstream (Microsoft) ORT is glibc + macOS only. For *-linux-musl targets there
# is no compatible ORT build, so ORT is skipped with a warning and the binary is
# built without it (non-fatal). Provisioning/download failures also degrade to a
# no-ORT build rather than breaking the release.
# ---------------------------------------------------------------------------
onnx_root=
if [ "$onnx" = "true" ]; then
  case "$target" in
    *-linux-musl|*musl*)
      echo "[onnx] target '$target' is musl; upstream ONNX Runtime is glibc-only -> building WITHOUT ORT (native interpreter). See report: the Linux runtime image (Alpine/musl) needs a glibc base to serve imported-ONNX via ORT." >&2
      onnx=false
      ;;
    *)
      case "$target" in
        x86_64-*) ort_arch=amd64 ;;
        aarch64-*) ort_arch=arm64 ;;
        *) ort_arch= ;;
      esac
      case "$target" in
        *-macos*|*-darwin*) ort_os=darwin ;;
        *-linux-*|*-linux) ort_os=linux ;;
        *) ort_os= ;;
      esac
      if [ -z "$ort_os" ] || [ -z "$ort_arch" ]; then
        echo "[onnx] cannot map target '$target' to an ORT platform -> building WITHOUT ORT" >&2
        onnx=false
      else
        ort_platform="${ort_os}-${ort_arch}"
        ort_base="$repo_root/zig/pkg/inference/onnxruntime"
        onnx_root="$ort_base/$ort_platform"
        if [ ! -f "$onnx_root/include/onnxruntime_c_api.h" ]; then
          echo "[onnx] provisioning ONNX Runtime for $ort_platform into $ort_base"
          if ! ONNXRUNTIME_ROOT="$ort_base" "$repo_root/zig/pkg/inference/scripts/download-onnxruntime.sh"; then
            echo "[onnx] download-onnxruntime.sh reported failures (partial platform availability is expected); continuing" >&2
          fi
        fi
        if [ -f "$onnx_root/include/onnxruntime_c_api.h" ] && [ -d "$onnx_root/lib" ]; then
          echo "[onnx] ONNX Runtime ready at $onnx_root -> building WITH ORT"
        else
          echo "[onnx] ONNX Runtime not available at $onnx_root -> building WITHOUT ORT" >&2
          onnx=false
          onnx_root=
        fi
      fi
      ;;
  esac
fi

zig_build_args=(
=======
lite_library_archive_path() {
  case "$1" in
    *windows*) echo "./bin/$(lite_library_name "$1")" ;;
    *) echo "./lib/$(lite_library_name "$1")" ;;
  esac
}

lite_lib_name="$(lite_library_name "$target")"
lite_lib_archive_path="$(lite_library_archive_path "$target")"
lite_lib_prefix_path="$prefix/${lite_lib_archive_path#./}"

rm -rf "$work_root"
mkdir -p "$prefix" "$stage" "$local_cache" "$cache_root/global" "$out_dir"

zig_build_options=(
>>>>>>> main
  -Dtarget="$target"
  -Doptimize="$optimize"
  -Dcpu=baseline
  -Dedition=full
  -Dantfly-bin-name=antfly
  -Dantfly-version="$version"
  -Donnx="$onnx"
  -Dmetal="$metal"
  -Dsystem-blas="$system_blas"
)

zig_install_args=(
  --prefix "$prefix"
  --cache-dir "$local_cache"
  --global-cache-dir "$cache_root/global"
)
if [ "$onnx" = "true" ] && [ -n "$onnx_root" ]; then
  zig_build_args+=(-Donnx-root="$onnx_root")
fi

(
  cd "$repo_root/zig"
  if [ -n "$jobs" ]; then
    zig build "-j$jobs" "${zig_build_options[@]}" install "${zig_install_args[@]}"
    zig build "-j$jobs" "${zig_build_options[@]}" lite-capi "${zig_install_args[@]}"
  else
    zig build "${zig_build_options[@]}" install "${zig_install_args[@]}"
    zig build "${zig_build_options[@]}" lite-capi "${zig_install_args[@]}"
  fi
)

test -x "$prefix/bin/antfly"
test -f "$prefix/include/antfly.h"
if [ ! -f "$lite_lib_prefix_path" ]; then
  echo "missing Antfly C ABI library: $lite_lib_prefix_path" >&2
  find "$prefix" -maxdepth 3 -type f | sort >&2
  exit 1
fi
cp "$prefix/bin/antfly" "$stage/antfly"
if [ -d "$prefix/share" ]; then
  cp -R "$prefix/share" "$stage/share"
fi
if [ -d "$prefix/lib" ]; then
  cp -R "$prefix/lib" "$stage/lib"
fi
if [ -d "$prefix/include" ]; then
  cp -R "$prefix/include" "$stage/include"
fi
cp "$repo_root/README.md" "$stage/README.md"
cp "$repo_root/LICENSE" "$stage/LICENSE"

tar -C "$stage" -czf "$out_dir/$archive_name" .
tar -tzf "$out_dir/$archive_name" > "$work_root/archive-contents.txt"
grep -Fx "./include/antfly.h" "$work_root/archive-contents.txt" >/dev/null
grep -Fx "$lite_lib_archive_path" "$work_root/archive-contents.txt" >/dev/null
echo "wrote $out_dir/$archive_name"
