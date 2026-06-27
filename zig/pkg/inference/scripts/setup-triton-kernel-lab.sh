#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s [--check|--install] [--venv PATH] [--with-torch]\n' "$0"
  printf 'Creates or checks the isolated Triton kernel lab used for Qwen3.6 CUDA experiments.\n'
  printf '\n'
  printf 'Environment overrides:\n'
  printf '  ANTFLY_TRITON_PACKAGE   default: triton==3.4.0\n'
  printf '  ANTFLY_TORCH_PACKAGE    default: torch==2.8.0\n'
}

mode="install"
with_torch=0
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../../../.." && pwd)"
venv_dir="$repo_dir/.tools/triton-qwen36-venv"

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --check)
      mode="check"
      ;;
    --install)
      mode="install"
      ;;
    --venv)
      shift
      if [ $# -eq 0 ]; then
        usage >&2
        exit 2
      fi
      venv_dir="$1"
      ;;
    --with-torch)
      with_torch=1
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

python_bin="${PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'error: python not found: %s\n' "$python_bin" >&2
  exit 1
fi

"$python_bin" - <<'PY'
import sys

if sys.version_info < (3, 10):
    raise SystemExit("error: Triton lab requires Python >= 3.10")
if sys.version_info >= (3, 15):
    raise SystemExit("error: Triton lab has not been validated on Python >= 3.15")
print("python=%d.%d.%d" % sys.version_info[:3])
PY

if [ "$mode" = "check" ]; then
  if [ ! -x "$venv_dir/bin/python" ]; then
    printf 'missing Triton lab venv: %s\n' "$venv_dir"
    exit 0
  fi
  "$venv_dir/bin/python" "$script_dir/compile_triton_qwen36.py" --check-env
  exit 0
fi

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m ensurepip --upgrade >/dev/null

triton_package="${ANTFLY_TRITON_PACKAGE:-triton==3.4.0}"
torch_package="${ANTFLY_TORCH_PACKAGE:-torch==2.8.0}"

packages=("$triton_package")
if [ "$with_torch" -eq 1 ]; then
  packages+=("$torch_package")
fi

PIP_DISABLE_PIP_VERSION_CHECK=1 "$venv_dir/bin/python" -m pip install "${packages[@]}"
"$venv_dir/bin/python" -m pip freeze > "$venv_dir/antfly-triton-versions.txt"
"$venv_dir/bin/python" "$script_dir/compile_triton_qwen36.py" --check-env

printf 'Triton lab ready: %s\n' "$venv_dir"
printf 'Version manifest: %s\n' "$venv_dir/antfly-triton-versions.txt"
