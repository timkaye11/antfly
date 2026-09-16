#!/usr/bin/env python3
"""Generate sampler fixtures by executing verbatim production Zig functions.

Requires the repository's pinned Zig 0.16 compiler, no model or GPU. The fixture
binds the extracted source and compiler version; it is not rollout acceptance.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def extract_sampling_source() -> str:
    source_path = (
        Path(__file__).resolve().parents[2] / "src/finetune/gemma4_real_autodiff.zig"
    )
    source = source_path.read_text()
    sections = [
        source[
            source.index("pub const GrpoSamplingOptions = struct {") : source.index(
                "/// Derive the independent PRNG stream used"
            )
        ],
        source[
            source.index("fn selectTopRankedTokens(") : source.index(
                "fn logProbAtToken("
            )
        ],
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = subprocess.check_output([str(args.zig), "version"], text=True).strip()
    if version != "0.16.0":
        raise RuntimeError(f"expected Zig 0.16.0, found {version}")
    extracted = extract_sampling_source()
    harness = r"""
pub fn main() !void {
    const logits = [_]f32{ -2.3, 1, 1, -0.1, 3.5, 3.5, -100, 0.7, 2.1, -1.1, 0, 0.01, 0.8 };
    const policies = [_]GrpoSamplingOptions{
        .{ .seed = 0 },
        .{ .seed = 0, .temperature = 0.8, .top_k = 4 },
        .{ .seed = 0, .temperature = 1.2, .top_p = 0.7 },
        .{ .seed = 0, .temperature = 0.6, .top_p = 0.95, .top_k = 7 },
        .{ .seed = 0, .top_k = 1 },
    };
    for ([_]u64{ 0, 42, std.math.maxInt(u64) }) |run_seed| {
        for ([_]u64{ 0x4752504f54524149, 0x4752504f4556414c }) |domain| {
            for ([_]usize{ 0, 7 }) |epoch| {
                for ([_]usize{ 0, 17 }) |prompt| {
                    const group = deriveGrpoSamplingGroupSeed(run_seed, domain, epoch, prompt);
                    for ([_]usize{ 0, 3 }) |completion| {
                        const seed = deriveGrpoCompletionSamplingSeed(group, completion);
                        var rng = std.Random.DefaultPrng.init(seed);
                        for (0..8) |step| {
                            const draw = rng.random().float(f64);
                            std.debug.print("[{d},{d},{d},{d},{d},{d},{d},{d},{d}", .{ run_seed, domain, epoch, prompt, completion, group, seed, step, @as(u64, @bitCast(draw)) });
                            for (policies) |policy| {
                                const token = try sampleGrpoTokenFromLogits(std.heap.page_allocator, &logits, policy, draw);
                                std.debug.print(",{d}", .{token});
                            }
                            std.debug.print("]\n", .{});
                        }
                    }
                }
            }
        }
    }
}
"""
    args.work_dir.mkdir(parents=True, exist_ok=False)
    zig_path = args.work_dir / "sampling.zig"
    zig_path.write_text('const std = @import("std");\n' + extracted + harness)
    command = [
        str(args.zig),
        "run",
        str(zig_path),
        "-O",
        "ReleaseFast",
        "--global-cache-dir",
        "/tmp/antfly-zig-global-cache",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stderr.splitlines()]
    assert len(rows) == 384
    fixture = {
        "schema_version": "antfly_gemma4_grpo_sampling_fixture/v1",
        "zig_version": version,
        "source": "src/finetune/gemma4_real_autodiff.zig",
        "extracted_source_sha256": hashlib.sha256(extracted.encode()).hexdigest(),
        "harness_sha256": hashlib.sha256(zig_path.read_bytes()).hexdigest(),
        "logits": [-2.3, 1, 1, -0.1, 3.5, 3.5, -100, 0.7, 2.1, -1.1, 0, 0.01, 0.8],
        "policies": [
            {},
            {"temperature": 0.8, "top_k": 4},
            {"temperature": 1.2, "top_p": 0.7},
            {"temperature": 0.6, "top_p": 0.95, "top_k": 7},
            {"top_k": 1},
        ],
        "row_fields": [
            "run_seed",
            "domain",
            "epoch",
            "prompt",
            "completion",
            "group_seed",
            "completion_seed",
            "step",
            "draw_bits",
            "policy_tokens...",
        ],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        rows = fixture.pop("rows")
        metadata = json.dumps(fixture, indent=2)
        handle.write(metadata[:-2] + ',\n  "rows": [\n')
        handle.write(
            ",\n".join("    " + json.dumps(row, separators=(",", ":")) for row in rows)
        )
        handle.write("\n  ]\n}\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "token_selections": len(rows) * 5,
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
