#!/usr/bin/env python3
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

"""Fast contract test for the paired Gemma4 long-output benchmark."""

import json
import os
from pathlib import Path
import subprocess
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK = SCRIPT_DIR / "benchmark_metal_gemma4_long_output.sh"


def executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gemma4-long-output-test-") as raw_tmp:
        tmp = Path(raw_tmp)
        model = tmp / "model.gguf"
        model.write_bytes(b"fake gguf")
        antfly = tmp / "antfly"
        llama = tmp / "llama-completion"
        out = tmp / "results"

        executable(
            antfly,
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
tokens = int(args[args.index("--max-tokens") + 1])
split_enable = os.environ.get("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT")
split_gqa = split_enable in (None, "1") and os.environ.get("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT") != "1"
timing_path = Path(args[args.index("--json-timing") + 1])
timing_path.write_text(json.dumps({
    "tokens": tokens,
    "finish_reason": "length",
    "timing_ms": {
        "generate": 10000,
        "prefill_inner": 4000,
        "decode_inner": 6000,
    },
}))
print("generate-setup: live whole-model executor skipped")
print("gen_debug: executePrefill whole-model fast path seq_len=2003")
print("prompt_token_ids:", " ".join(str(i) for i in range(2003)))
print("token_ids:", " ".join(str(i) for i in range(tokens)))
print(f"metal_attention_dispatch: paged_1x={0 if split_gqa else (tokens - 1) * 42} decode_gqa_split={(tokens - 1) * 42 if split_gqa else 0}")
print("metal_runtime_memory: frame_retained_mb=0")
print(f"metal_q4_0_dispatch: linear_reduce_rows={(tokens - 1) * 210}/0/0/0")
print(f"metal_q4_q6_k_dispatch: q6_linear_reduce_rows={tokens}/0/0/0")
print("metal_q4_0_encode_us: linear_reduce=1234")
""",
        )
        executable(
            llama,
            """#!/usr/bin/env python3
import sys

args = sys.argv[1:]
if "--version" in args:
    print("fake llama.cpp")
    raise SystemExit(0)
tokens = int(args[args.index("-n") + 1])
print("common_perf_print: prompt eval time = 4000.00 ms / 2003 tokens")
print(f"common_perf_print: eval time = 5600.00 ms / {tokens - 1} runs")
print("common_perf_print: sampling time = 50.00 ms")
print("common_perf_print: total time = 9500.00 ms")
""",
        )

        env = os.environ.copy()
        env.update({
            "MODEL": str(model),
            "GGUF": str(model),
            "ANTFLY_BIN": str(antfly),
            "LLAMA_CPP_BIN": str(llama),
            "OUT_DIR": str(out),
            "OUTPUT_TOKENS": "300",
            "WARMUPS": "0",
            "RUNS": "2",
            "COOLDOWN_SECONDS": "0",
            "MAX_CV": "0.01",
        })
        env.pop("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT", None)
        env.pop("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT", None)
        completed = subprocess.run(
            ["bash", str(BENCHMARK)],
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "prompt=2003 output=300" in completed.stdout
        summary = json.loads((out / "summary.json").read_text())
        assert summary["prompt_tokens"] == 2003
        assert summary["output_tokens"] == 300
        assert summary["total_ratio"] <= 1.10
        assert summary["decode_ratio"] >= 0.90
        assert summary["rows"][0]["paged_1x_calls"] == 0
        assert summary["rows"][0]["decode_gqa_split_calls"] == 12_558
        assert summary["rows"][0]["frame_retained_mb"] == 0
        assert summary["rows"][0]["q4_0_linear_reduce_rows_1"] == 62_790
        assert summary["rows"][0]["q6_k_linear_reduce_rows_1"] == 300
        assert summary["rows"][0]["q4_0_linear_reduce_encode_us"] == 1234
        assert summary["metadata"]["gguf_sha256"]
        assert summary["metadata"]["llama_cpp_version"] == "fake llama.cpp"
        assert summary["metadata"]["antfly_cache_dtype"] == "f16"
        assert summary["metadata"]["llama_cache_type_k"] == "f16"
        assert summary["metadata"]["llama_cache_type_v"] == "f16"
        assert summary["metadata"]["warmup_output_tokens"] == 4
        assert summary["metadata"]["pipelined_decode_frame_enable"] is None

        rollback_out = tmp / "paged-results"
        env["OUT_DIR"] = str(rollback_out)
        env["TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT"] = "1"
        subprocess.run(["bash", str(BENCHMARK)], env=env, check=True, text=True, capture_output=True)
        rollback_summary = json.loads((rollback_out / "summary.json").read_text())
        assert rollback_summary["rows"][0]["paged_1x_calls"] == 12_558
        assert rollback_summary["rows"][0]["decode_gqa_split_calls"] == 0


if __name__ == "__main__":
    main()
