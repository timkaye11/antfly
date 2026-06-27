#!/usr/bin/env python3
"""Compile experimental Triton probes for the Qwen3.6 CUDA path.

The probes are intentionally separate from checked-in CUDA artifacts. They let us
compare Triton-generated PTX/cubin against hand-written kernels before routing
production inference through them.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import sys
from typing import Any, Callable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
DEFAULT_OUT_DIR = REPO_ROOT / ".tools" / "triton-qwen36-artifacts"
DEFAULT_TRITON_CACHE_DIR = REPO_ROOT / ".tools" / "triton-cache"
os.environ.setdefault("TRITON_CACHE_DIR", str(DEFAULT_TRITON_CACHE_DIR))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_report() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "triton_installed": importlib.util.find_spec("triton") is not None,
        "triton_version": package_version("triton"),
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torch_version": package_version("torch"),
        "cuda_home": os.environ.get("CUDA_HOME"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }


def print_check_env() -> int:
    print(json.dumps(env_report(), indent=2, sort_keys=True))
    return 0


def require_triton() -> tuple[Any, Any]:
    try:
        import triton
        import triton.language as tl
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "error: Triton is not installed. Run scripts/setup-triton-kernel-lab.sh --install first."
        ) from exc
    return triton, tl


def build_probe_kernels(triton: Any, tl: Any) -> dict[str, Callable[..., Any]]:
    @triton.jit
    def qwen36_kv_write_suffix_decode_scalars_f32_probe(
        key_cache,
        value_cache,
        key_src,
        value_src,
        token_offset,
        kv_values,
        total_values,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_values
        dst = token_offset * kv_values + offsets
        key = tl.load(key_src + offsets, mask=mask, other=0.0)
        value = tl.load(value_src + offsets, mask=mask, other=0.0)
        tl.store(key_cache + dst, key, mask=mask)
        tl.store(value_cache + dst, value, mask=mask)

    @triton.jit
    def qwen36_kv_write_suffix_device_scalars_f32_probe(
        key_cache,
        value_cache,
        key_src,
        value_src,
        decode_scalars,
        suffix_tokens,
        kv_values,
        total_values,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_values
        kv_seq_len = tl.load(decode_scalars + 2)
        token_offset = kv_seq_len - suffix_tokens
        dst = token_offset * kv_values + offsets
        key = tl.load(key_src + offsets, mask=mask, other=0.0)
        value = tl.load(value_src + offsets, mask=mask, other=0.0)
        tl.store(key_cache + dst, key, mask=mask)
        tl.store(value_cache + dst, value, mask=mask)

    @triton.jit
    def qwen36_attention_decode_f32_probe(
        output,
        query,
        key_cache,
        value_cache,
        kv_seq_len,
        num_heads,
        num_kv_heads,
        query_position_offset,
        kv_position_offset,
        sliding_window,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % num_heads
        kv_head = head // (num_heads // num_kv_heads)
        head_offsets = tl.arange(0, HEAD_DIM)
        token_offsets = tl.arange(0, BLOCK_N)

        q_base = head * HEAD_DIM
        kv_hidden = num_kv_heads * HEAD_DIM
        q = tl.load(query + q_base + head_offsets)
        scale = 1.0 / tl.sqrt(HEAD_DIM + 0.0)
        query_pos = query_position_offset

        max_score = tl.full((), -3.4028234663852886e38, tl.float32)
        for start in range(0, kv_seq_len, BLOCK_N):
            tokens = start + token_offsets
            key_pos = kv_position_offset + tokens
            valid = tokens < kv_seq_len
            valid = valid & (key_pos <= query_pos)
            valid = valid & ((sliding_window == 0) | ((query_pos - key_pos) < sliding_window))
            k_offsets = tokens[:, None] * kv_hidden + kv_head * HEAD_DIM + head_offsets[None, :]
            k = tl.load(key_cache + k_offsets, mask=valid[:, None], other=0.0)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -3.4028234663852886e38)
            max_score = tl.maximum(max_score, tl.max(scores, axis=0))

        denom = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((HEAD_DIM,), tl.float32)
        for start in range(0, kv_seq_len, BLOCK_N):
            tokens = start + token_offsets
            key_pos = kv_position_offset + tokens
            valid = tokens < kv_seq_len
            valid = valid & (key_pos <= query_pos)
            valid = valid & ((sliding_window == 0) | ((query_pos - key_pos) < sliding_window))
            k_offsets = tokens[:, None] * kv_hidden + kv_head * HEAD_DIM + head_offsets[None, :]
            k = tl.load(key_cache + k_offsets, mask=valid[:, None], other=0.0)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -3.4028234663852886e38)
            weights = tl.exp(scores - max_score)
            weights = tl.where(valid, weights, 0.0)
            denom += tl.sum(weights, axis=0)
            v_offsets = tokens[:, None] * kv_hidden + kv_head * HEAD_DIM + head_offsets[None, :]
            v = tl.load(value_cache + v_offsets, mask=valid[:, None], other=0.0)
            acc += tl.sum(v * weights[:, None], axis=0)

        result = acc / denom
        result = tl.where(denom > 0.0, result, 0.0)
        tl.store(output + q_base + head_offsets, result)

    @triton.jit
    def qwen36_attention_decode_f32_tiled_probe(
        output,
        query,
        key_cache,
        value_cache,
        kv_seq_len,
        num_heads,
        num_kv_heads,
        query_position_offset,
        kv_position_offset,
        sliding_window,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % num_heads
        kv_head = head // (num_heads // num_kv_heads)
        head_offsets = tl.arange(0, HEAD_DIM)
        token_offsets = tl.arange(0, BLOCK_N)

        q_base = head * HEAD_DIM
        kv_hidden = num_kv_heads * HEAD_DIM
        q = tl.load(query + q_base + head_offsets)
        scale = 1.0 / tl.sqrt(HEAD_DIM + 0.0)
        query_pos = query_position_offset

        running_max = tl.full((), -3.4028234663852886e38, tl.float32)
        running_denom = tl.full((), 0.0, tl.float32)
        running_acc = tl.zeros((HEAD_DIM,), tl.float32)

        for start in range(0, kv_seq_len, BLOCK_N):
            tokens = start + token_offsets
            key_pos = kv_position_offset + tokens
            valid = tokens < kv_seq_len
            valid = valid & (key_pos <= query_pos)
            valid = valid & ((sliding_window == 0) | ((query_pos - key_pos) < sliding_window))

            kv_offsets = tokens[:, None] * kv_hidden + kv_head * HEAD_DIM + head_offsets[None, :]
            key_tile = tl.load(key_cache + kv_offsets, mask=valid[:, None], other=0.0)
            scores = tl.sum(key_tile * q[None, :], axis=1) * scale
            scores = tl.where(valid, scores, -3.4028234663852886e38)

            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp(running_max - next_max)
            weights = tl.exp(scores - next_max)
            weights = tl.where(valid, weights, 0.0)

            value_tile = tl.load(value_cache + kv_offsets, mask=valid[:, None], other=0.0)
            running_acc = running_acc * old_scale + tl.sum(value_tile * weights[:, None], axis=0)
            running_denom = running_denom * old_scale + tl.sum(weights, axis=0)
            running_max = next_max

        result = running_acc / running_denom
        result = tl.where(running_denom > 0.0, result, 0.0)
        tl.store(output + q_base + head_offsets, result)

    @triton.jit
    def qwen36_mlp_gate_up_f32_probe(
        activated,
        input_vec,
        gate_weight,
        up_weight,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        out_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_K)
        acc_gate = tl.zeros((BLOCK_M,), tl.float32)
        acc_up = tl.zeros((BLOCK_M,), tl.float32)
        for k0 in range(0, HIDDEN, BLOCK_K):
            k = k0 + k_offsets
            x = tl.load(input_vec + k, mask=k < HIDDEN, other=0.0)
            weight_offsets = out_offsets[:, None] * HIDDEN + k[None, :]
            valid = (out_offsets[:, None] < INTERMEDIATE) & (k[None, :] < HIDDEN)
            gate = tl.load(gate_weight + weight_offsets, mask=valid, other=0.0)
            up = tl.load(up_weight + weight_offsets, mask=valid, other=0.0)
            acc_gate += tl.sum(gate * x[None, :], axis=1)
            acc_up += tl.sum(up * x[None, :], axis=1)
        silu = acc_gate * tl.sigmoid(acc_gate)
        tl.store(activated + out_offsets, silu * acc_up, mask=out_offsets < INTERMEDIATE)

    @triton.jit
    def qwen36_mlp_down_residual_f32_probe(
        output,
        activated,
        down_weight,
        residual,
        HIDDEN: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        out_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        k_offsets = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for k0 in range(0, INTERMEDIATE, BLOCK_K):
            k = k0 + k_offsets
            x = tl.load(activated + k, mask=k < INTERMEDIATE, other=0.0)
            weight_offsets = out_offsets[:, None] * INTERMEDIATE + k[None, :]
            valid = (out_offsets[:, None] < HIDDEN) & (k[None, :] < INTERMEDIATE)
            w = tl.load(down_weight + weight_offsets, mask=valid, other=0.0)
            acc += tl.sum(w * x[None, :], axis=1)
        res = tl.load(residual + out_offsets, mask=out_offsets < HIDDEN, other=0.0)
        tl.store(output + out_offsets, acc + res, mask=out_offsets < HIDDEN)

    return {
        "kv-write-probe": qwen36_kv_write_suffix_decode_scalars_f32_probe,
        "kv-write-scalars-probe": qwen36_kv_write_suffix_device_scalars_f32_probe,
        "attention-decode-f32-probe": qwen36_attention_decode_f32_probe,
        "attention-decode-f32-tiled-probe": qwen36_attention_decode_f32_tiled_probe,
        "attention-decode-f32-tiled4-probe": qwen36_attention_decode_f32_tiled_probe,
        "attention-decode-f32-tiled16-probe": qwen36_attention_decode_f32_tiled_probe,
        "attention-decode-f32-tiled32-probe": qwen36_attention_decode_f32_tiled_probe,
        "mlp-gate-up-f32-probe": qwen36_mlp_gate_up_f32_probe,
        "mlp-down-f32-probe": qwen36_mlp_down_residual_f32_probe,
    }


def compile_kernel(
    kernel_name: str,
    kernel: Callable[..., Any],
    out_dir: pathlib.Path,
    sm: int,
) -> dict[str, Any]:
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, compile as triton_compile

    if kernel_name == "kv-write-probe":
        signature = {
            "key_cache": "*fp32",
            "value_cache": "*fp32",
            "key_src": "*fp32",
            "value_src": "*fp32",
            "token_offset": "i32",
            "kv_values": "i32",
            "total_values": "i32",
        }
        constants = {"BLOCK": 256}
        options = {"num_warps": 4, "num_ctas": 1}
    elif kernel_name == "kv-write-scalars-probe":
        signature = {
            "key_cache": "*fp32",
            "value_cache": "*fp32",
            "key_src": "*fp32",
            "value_src": "*fp32",
            "decode_scalars": "*i32",
            "suffix_tokens": "i32",
            "kv_values": "i32",
            "total_values": "i32",
        }
        constants = {"BLOCK": 256}
        options = {"num_warps": 4, "num_ctas": 1}
    elif kernel_name == "attention-decode-f32-probe":
        signature = {
            "output": "*fp32",
            "query": "*fp32",
            "key_cache": "*fp32",
            "value_cache": "*fp32",
            "kv_seq_len": "i32",
            "num_heads": "i32",
            "num_kv_heads": "i32",
            "query_position_offset": "i32",
            "kv_position_offset": "i32",
            "sliding_window": "i32",
        }
        constants = {"HEAD_DIM": 256, "BLOCK_N": 1}
        options = {"num_warps": 8, "num_ctas": 1}
    elif kernel_name in {
        "attention-decode-f32-tiled-probe",
        "attention-decode-f32-tiled4-probe",
        "attention-decode-f32-tiled16-probe",
        "attention-decode-f32-tiled32-probe",
    }:
        signature = {
            "output": "*fp32",
            "query": "*fp32",
            "key_cache": "*fp32",
            "value_cache": "*fp32",
            "kv_seq_len": "i32",
            "num_heads": "i32",
            "num_kv_heads": "i32",
            "query_position_offset": "i32",
            "kv_position_offset": "i32",
            "sliding_window": "i32",
        }
        block_n_by_kernel = {
            "attention-decode-f32-tiled-probe": 8,
            "attention-decode-f32-tiled4-probe": 4,
            "attention-decode-f32-tiled16-probe": 16,
            "attention-decode-f32-tiled32-probe": 32,
        }
        constants = {"HEAD_DIM": 256, "BLOCK_N": block_n_by_kernel[kernel_name]}
        options = {"num_warps": 8, "num_ctas": 1}
    elif kernel_name == "mlp-gate-up-f32-probe":
        signature = {
            "activated": "*fp32",
            "input_vec": "*fp32",
            "gate_weight": "*fp32",
            "up_weight": "*fp32",
        }
        constants = {"HIDDEN": 5120, "INTERMEDIATE": 17408, "BLOCK_M": 64, "BLOCK_K": 128}
        options = {"num_warps": 8, "num_ctas": 1}
    elif kernel_name == "mlp-down-f32-probe":
        signature = {
            "output": "*fp32",
            "activated": "*fp32",
            "down_weight": "*fp32",
            "residual": "*fp32",
        }
        constants = {"HIDDEN": 5120, "INTERMEDIATE": 17408, "BLOCK_M": 64, "BLOCK_K": 128}
        options = {"num_warps": 8, "num_ctas": 1}
    else:
        raise ValueError(f"unknown kernel: {kernel_name}")

    source = ASTSource(kernel, signature=signature, constexprs=constants)
    compiled = triton_compile(source, target=GPUTarget("cuda", sm, 32), options=options)
    kernel_dir = out_dir / kernel_name
    kernel_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for ext, blob in compiled.asm.items():
        suffix = "cubin" if isinstance(blob, bytes) else ext
        path = kernel_dir / f"{kernel_name}.sm{sm}.{suffix}"
        if isinstance(blob, bytes):
            path.write_bytes(blob)
        else:
            path.write_text(blob)
        written[ext] = str(path)

    metadata = compiled.metadata._asdict() if hasattr(compiled.metadata, "_asdict") else vars(compiled.metadata)
    manifest = {
        "kernel": kernel_name,
        "sm": sm,
        "signature": signature,
        "constexprs": constants,
        "options": options,
        "metadata": metadata,
        "artifacts": written,
        "environment": env_report(),
    }
    manifest_path = kernel_dir / f"{kernel_name}.sm{sm}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-env", action="store_true", help="print Triton/Torch availability and exit")
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sm", type=int, default=89, help="CUDA SM target, for example 89 for Ada")
    parser.add_argument(
        "--kernel",
        choices=(
            "all",
            "kv-write-probe",
            "kv-write-scalars-probe",
            "attention-decode-f32-probe",
            "attention-decode-f32-tiled-probe",
            "attention-decode-f32-tiled4-probe",
            "attention-decode-f32-tiled16-probe",
            "attention-decode-f32-tiled32-probe",
            "mlp-gate-up-f32-probe",
            "mlp-down-f32-probe",
        ),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_env:
        return print_check_env()

    triton, tl = require_triton()
    kernels = build_probe_kernels(triton, tl)
    names = list(kernels) if args.kernel == "all" else [args.kernel]
    manifests = [compile_kernel(name, kernels[name], args.out_dir, args.sm) for name in names]
    print(json.dumps({"out_dir": str(args.out_dir), "compiled": manifests}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
