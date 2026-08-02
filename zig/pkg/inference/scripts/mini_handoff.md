# Gemma4 26B-A4B Metal benchmark handoff

This runbook validates branch `codex/gemma4-a4b-compact2g` on the 24 GB Apple-silicon
machine that has the real TurboFieldfare binary and `.gturbo` model. Run it from
`<repo>/zig`. Replace only the four shell placeholders below; no source edits or
environment routing overrides are required.

```sh
export ANTFLY_MODEL=/absolute/path/to/gemma-4-26B_q4_0-it.gguf
export TURBO_BIN=/absolute/path/to/turbo-fieldfare
export TURBO_MODEL=/absolute/path/to/model.gturbo
export RESULT_ROOT=/tmp/a4b-final
```

The two performance claims are deliberately separate:

- `compact-2g`: Antfly is capped at 2048 MiB with device routing off. This is the
  direct low-memory comparison to TurboFieldfare.
- `full-residency`: Antfly uses a 16384 MiB budget and explicitly requires all 128
  experts per layer resident. This measures the fastest zero-drain route, not a
  2 GiB claim.

`off` is the production default. `auto`, `partial`, and `full` are explicit opt-ins
until this machine produces correctness and performance evidence.

## 1. Clean build and deterministic tests

```sh
cd <repo>/zig
export UV_CACHE_DIR=/tmp/antfly-a4b-uv-cache
export ZIG_LOCAL_CACHE_DIR=/tmp/antfly-a4b-zig-local
export ZIG_GLOBAL_CACHE_DIR=/tmp/antfly-a4b-zig-global
zig build -Doptimize=ReleaseFast
python3 -m unittest discover -s pkg/inference/scripts -p 'test_*a4b*.py'
python3 -m unittest discover -s pkg/inference/scripts -p 'test_measure_serving_ttft.py'
zig build test
```

All commands must exit zero. Do not waive named failures. Save the complete output.

## 2. Prove full residency engages safely

```sh
ANTFLY_GEMMA4_COMPACT_TIMING=1 \
python3 pkg/inference/scripts/run_guarded_a4b.py \
  --kill-gib 15.0 --log "$RESULT_ROOT-full-smoke.log" 900 -- \
  zig-out/bin/antfly-inference generate \
  "$ANTFLY_MODEL" 'Hello upon-' \
  --backend metal --memory-profile 2gbs --memory-budget-mb 16384 \
  --compact-device-routing full --raw-prompt --temperature 0 --seed 42 \
  --cache-dtype f16 --ignore-eos --print-token-ids --print-timing --max-tokens 128
```

Required evidence in the log:

- `resident_slots=3840/3840` (128 experts times 30 MoE layers).
- no expert loads or evictions after warm residency is established.
- roughly one submitted decode frame/flush per output token rather than one
  router readback per layer.
- `decode_tok_per_s` is present, watchdog breach is `None`, and the exact
  `peak_phys_footprint_bytes` is below the configured ceiling.

Any missing routing telemetry, fallback, OOM, timeout, or watchdog breach fails this
step. Close other applications and rerun once if system pressure caused the failure.

## 3. Exact streamed-versus-full token gate

The comparison tool launches the two routes sequentially with the same model, prompt,
dtype, token count, and seed. Output directories must not already exist.

```sh
python3 pkg/inference/scripts/verify_gemma4_a4b_parity.py compare-compact-routes \
  --antfly-bin zig-out/bin/antfly-inference --model "$ANTFLY_MODEL" \
  --prompt 'Hello upon-' --tokens 256 --seed 42 \
  --streamed-memory-budget-mb 2048 --full-memory-budget-mb 16384 \
  --out-dir "$RESULT_ROOT-route-parity"
```

Repeat with at least three materially different prompts. A single ordered token-ID
divergence is a correctness finding: keep full routing opt-in and preserve both logs.

If the pinned instrumented llama.cpp binary at commit `32703b42d6...` is available,
also run the strict anchor. Stock llama.cpp is not accepted because it lacks the
`reference_token_id:` instrumentation.

```sh
python3 pkg/inference/scripts/verify_gemma4_a4b_parity.py record-reference \
  --llama-bin /absolute/path/to/instrumented-llama-cli \
  --model "$ANTFLY_MODEL" --prompt 'Hello upon-' --tokens 32 \
  --cache-dtype both --out-dir "$RESULT_ROOT-llama-ref"

python3 pkg/inference/scripts/verify_gemma4_a4b_parity.py verify-reference \
  --antfly-bin zig-out/bin/antfly-inference --model "$ANTFLY_MODEL" \
  --reference "$RESULT_ROOT-llama-ref" --memory-budget-mb 2048 \
  --device-routing off --seed 42 --out-dir "$RESULT_ROOT-llama-verify"
```

## 4. Guarded budget matrix

Run each row only after the prior process exits. The explicit route makes the result
auditable; it does not rely on an environment variable or an `auto` heuristic.

```sh
while read BUDGET ROUTE CEILING; do
  ANTFLY_GEMMA4_COMPACT_TIMING=1 \
  python3 pkg/inference/scripts/run_guarded_a4b.py \
    --kill-gib "$CEILING" --log "$RESULT_ROOT-budget-$BUDGET.log" 900 -- \
    zig-out/bin/antfly-inference generate \
    "$ANTFLY_MODEL" 'Hello upon-' --backend metal --memory-profile 2gbs \
    --memory-budget-mb "$BUDGET" --compact-device-routing "$ROUTE" \
    --raw-prompt --temperature 0 --seed 42 --cache-dtype f16 --ignore-eos \
    --print-token-ids --print-timing --max-tokens 128
done <<'MATRIX'
2048 off 2.2
6144 partial 7.0
8192 partial 9.0
16384 full 15.0
MATRIX
```

Each row must remain below its ceiling and emit the requested effective route. Do not
compare throughput from a row that shows a fallback, missing footprint, or memory
pressure breach.

## 5. Paired Turbo benchmarks

The harness performs one warmup per engine, then six measured runs in AB/BA order. It
fails unless every run produces the same normalized generated text, token counts
match, footprint telemetry is present, decode CV is at most 3%, and Antfly's median
decode throughput is no more than 5% below Turbo. TTFT for both engines is the same
fresh-process `wall - decode` proxy. A different generated output is a correctness
finding, not a benchmark result, because it exercises a different expert-route trace.

```sh
python3 pkg/inference/scripts/benchmark_gemma4_a4b_turbo.py \
  --lane compact-2g \
  --antfly-bin zig-out/bin/antfly-inference --antfly-model "$ANTFLY_MODEL" \
  --turbo-bin "$TURBO_BIN" --turbo-model "$TURBO_MODEL" \
  --prompt 'Hello upon-' --tokens 128 --seed 42 --warmups 1 --repeats 6 \
  --max-decode-cv 0.03 --out-dir "$RESULT_ROOT-turbo-compact2g"

python3 pkg/inference/scripts/benchmark_gemma4_a4b_turbo.py \
  --lane full-residency \
  --antfly-bin zig-out/bin/antfly-inference --antfly-model "$ANTFLY_MODEL" \
  --turbo-bin "$TURBO_BIN" --turbo-model "$TURBO_MODEL" \
  --prompt 'Hello upon-' --tokens 128 --seed 42 --warmups 1 --repeats 6 \
  --max-decode-cv 0.03 --require-antfly-win \
  --out-dir "$RESULT_ROOT-turbo-full"
```

Keep both `summary.json` files and all raw logs. The compact result supports the 2 GiB
claim; the full-residency result supports the fastest-route claim. Never blend them.

## 6. Optional single-model serving gate

```sh
python3 pkg/inference/scripts/measure_serving_ttft.py \
  --binary zig-out/bin/antfly-inference \
  --model-dir "$(dirname "$ANTFLY_MODEL")" \
  --budget-mb 2048 --device-routing off --seed 42 \
  --json-out "$RESULT_ROOT-serving.json"
```

The default serving gate enforces one preloaded compact generator, one loaded model,
`allow_unknown_models=false`, deterministic cold replay, zero cached prompt tokens,
zero retained prompt-cache bytes, and the same 2048 MiB process-footprint ceiling.
Compact prompt reuse remains disabled because logical cache blocks do not retain their
Metal device-KV slots after sequence teardown.

After a device-KV retention fix lands, validate the experimental seam separately:

```sh
python3 pkg/inference/scripts/measure_serving_ttft.py \
  --binary zig-out/bin/antfly-inference \
  --model-dir "$(dirname "$ANTFLY_MODEL")" \
  --budget-mb 2048 --device-routing off --seed 42 \
  --experimental-compact-prompt-cache-reuse --max-ttft-ms 1200 \
  --json-out "$RESULT_ROOT-serving-cache-experimental.json"
```

That command must show positive cached-token and hit-metric evidence while repeated
prompt output remains exact. It is not a release gate until the underlying device-KV
ownership fix is reviewed.

## Release decision

The branch is benchmark-ready only when Steps 1-3 are green. Publish a performance
claim only from a stable Step 5 lane. Keep full routing opt-in if exact token parity
fails or if full residency ever silently falls back. Changing the production default
is a separate reviewed change after these artifacts are available.
