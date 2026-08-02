# Gemma 4 26B-A4B — 24 GB Mac mini hand-off runbook

Run this top-to-bottom on a **24 GB Apple-silicon Mac mini** to validate and measure
the **full-residency device-routing decode path** — the mechanism intended to beat
TurboFieldfare (25-26 tok/s) by eliminating the per-layer router-readback GPU drains.

Everything here targets branch `codex/gemma4-a4b-compact2g`. All paths assume the repo
at `/Users/…/antfly`; adjust the prefix for your checkout. The model is the **`.gguf`
FILE** (not the directory):
`…/models/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf` (14.4 GB).

---

## Background: what this validates and the one open question

Device routing runs the MoE FFN on-device. At **full residency** (all 128 experts/
layer resident, ≈13 GB) the router top-k also runs on-device (`termite_moe_topk`), so
the host reads **nothing** back per layer — the 30 per-token router drains vanish.
That is only reachable at a budget deriving 128 slots (≥ 14024 MiB), i.e. the 24 GB
mini; the 16 GB Air can't hold it. `device_routing=auto` (the default) selects this
`full` mode automatically at 128 slots.

**Certified already (unit level, no mini needed):** the device route *selection* is
bit-exact to the CPU/llama path (same 8 experts, same order, incl. near-ties), and
the device FFN kernels are bit-exact to the streamed path.

**The one open question this runbook answers on the mini:** the device softmax
*weights* differ from the CPU/llama path by ~5e-8 — **irreducible** (GPU `exp` vs CPU
`libm expf`; Apple GPUs have no fp64; computing softmax on the CPU to match would
require a logit readback = the very drain we removed). So the full path is a
*distinct-but-~5e-8-off* numerical path. Step 3 measures **how often, if ever, that
flips a decoded token** — the data behind the final decision (make full-residency
device routing the unconditional default, vs keep it opt-in).

---

## Prerequisites
- 24 GB Apple-silicon Mac mini, otherwise idle (thermals + memory matter).
- Zig 0.16.0 (`/opt/homebrew/bin/zig`).
- The exact A4B Q4_0 model file above (only this geometry qualifies — 30 layers,
  128 experts, top-8, `encoded_expert_bytes=3_345_408`; any other GGUF fails closed).
- **For the beat-Turbo A/B (Step 4):** the real TurboFieldfare binary + `.gturbo`
  model that produced the 25-26 tok/s reference. Not in this repo.
- **For the strict llama.cpp token anchor (Step 3, optional):** a *monolithic,
  instrumented* `llama-cli` built at commit `32703b42d6…` patched to print
  `reference_token_id:` per token. Stock/homebrew llama.cpp will **not** work (its
  `--version` lacks `32703b4`, it emits no reference markers, and split builds reject
  `--no-conversation`). Step 3's primary gate does **not** need this.

Use the **standard** `scripts/run_guarded_a4b.py` on the mini (NOT the Air-only
`guard_relaxed.py`) — but you MUST raise `--kill-gb` to `15.0` (default 2.20 SIGKILLs
a 13 GB run instantly). The 25 % system-pressure guard won't false-trip on 24 GB.

---

## Step 1 — Build + unit suite
```
cd <repo>/zig/pkg/inference
zig build -Doptimize=ReleaseFast
ls -la zig-out/bin/antfly-inference
zig build test 2>&1 | tail -40
```
Expect the suite green except the **3 known pre-existing failures**
(`metal native paged attention f16 single row …` + two `quant_kernel_compiler` drift
guards). A `gemma3_projector … FileNotFound` is a disk-full flake, not a regression.
Confirm these two device-routing certification tests PASS:
`device MoE topk route selection is bit-exact to the CPU selector` and
`full device MoE chain (device topk -> FFN) matches streamed chain`.

## Step 2 — Engage full residency and prove the drains are gone
```
cd <repo>/zig/pkg/inference
ANTFLY_GEMMA4_COMPACT_TIMING=1 python3 scripts/run_guarded_a4b.py --kill-gb 15.0 \
  --log /tmp/a4b-full.log 900 \
  zig-out/bin/antfly-inference generate \
  <model.gguf> "Hello upon-" \
  --backend metal --memory-budget-mb 16384 --raw-prompt --temperature 0 \
  --cache-dtype f16 --ignore-eos --print-token-ids --print-timing --max-tokens 128
```
`--memory-budget-mb 16384` derives **128 slots → auto → full** (no env needed; the
*budget*, not an env flag, creates residency — forcing `ANTFLY_GEMMA4_DEVICE_ROUTING=
full` at a smaller budget does NOT enlarge residency). Check `/tmp/a4b-full.log`:
- `metal_compact_expert_cache: resident_slots=3840/3840 … step_frames=<N> step_flushes=<~N>`
  — **the win:** `step_flushes` should be ≈ `step_frames` (≈ **1 flush/token**, only
  the final argmax), vs the partial path's ~31/token. `resident_slots` should be full
  (128×30=3840) with **misses≈0**.
- `gemma4_device_route_layer: layer=… experts=8 joined_frame=true` for every layer.
- `watchdog: … peak_phys_footprint≈13GB breach=None`, and `decode_tok_per_s=…`.

If it OOMs/kills, the mini is memory-contended — close other apps and retry.

## Step 3 — Correctness gate (the ~5e-8 softmax question)

**3a — Primary gate (no llama.cpp needed): does the full device path decode the same
tokens as the streamed reference?** Run both at the same 128-slot budget and diff the
token IDs:
```
# full device routing (auto -> full):
zig-out/bin/antfly-inference generate <model.gguf> "Hello upon-" \
  --backend metal --memory-budget-mb 16384 --raw-prompt --temperature 0 \
  --cache-dtype f16 --ignore-eos --print-token-ids --max-tokens 256 \
  | grep '^token_ids:' > /tmp/tok_full.txt
# streamed reference (device routing forced OFF), same budget:
ANTFLY_GEMMA4_DEVICE_ROUTING=off zig-out/bin/antfly-inference generate <model.gguf> "Hello upon-" \
  --backend metal --memory-budget-mb 16384 --raw-prompt --temperature 0 \
  --cache-dtype f16 --ignore-eos --print-token-ids --max-tokens 256 \
  | grep '^token_ids:' > /tmp/tok_streamed.txt
diff /tmp/tok_full.txt /tmp/tok_streamed.txt && echo "IDENTICAL — softmax residual did not flip any token"
```
Repeat over several diverse prompts (and longer `--max-tokens`, e.g. 256). The
streamed path is the llama-faithful reference (it matches the llama oracle bit-for-bit
elsewhere), so **full == streamed on tokens ⟹ full == llama on tokens.**
- **All identical:** the ~5e-8 never flips in practice → the full path is
  effectively correct → safe to make default (Step 5, option A).
- **Occasional divergence:** note which prompt/token; that's a real ~5e-8 near-tie
  flip → keep opt-in or gate by workload (Step 5, option B). Record the rate.

**3b — Per-layer divergence proof (optional, diagnostic):** dual-execute mode logs
device-vs-CPU FFN divergence per layer:
```
ANTFLY_GEMMA4_DEVICE_ROUTING_VERIFY=1 ANTFLY_GEMMA4_COMPACT_TIMING=1 \
  zig-out/bin/antfly-inference generate <model.gguf> "Hello upon-" \
  --backend metal --memory-budget-mb 16384 --raw-prompt --temperature 0 \
  --cache-dtype f16 --ignore-eos --max-tokens 8 2>&1 | grep gemma4_device_route_verify
```
Expect `max_abs_diff` ~1e-7 uniformly (the softmax-weight residual), no gross outlier.

**3c — Strict llama.cpp token anchor (optional, needs the instrumented llama-cli):**
confirms the *streamed/compact* engine matches llama.cpp exactly. Note the parity tool
runs Antfly at the hard-coded **2 GB compact profile** (streamed, not full residency),
so this anchors streamed↔llama; combined with 3a (full↔streamed) it transitively
anchors full↔llama.
```
# record once (writes a bundle; --out-dir must NOT exist):
python3 scripts/verify_gemma4_a4b_parity.py record-reference \
  --llama-bin <instrumented-llama-cli> --model <model.gguf> \
  --out-dir /tmp/gemma4-ref --prompt 'Hello upon-' --tokens 8 --cache-dtype both --llama-gpu-layers 0
# verify Antfly against it:
python3 scripts/verify_gemma4_a4b_parity.py verify-reference \
  --antfly-bin zig-out/bin/antfly-inference --model <model.gguf> \
  --reference /tmp/gemma4-ref --out-dir /tmp/gemma4-verify
```
If your instrumented build's `--version` lacks `32703b4`, add `--expected-llama-commit ''`
to record-reference. (Do not use stock llama.cpp — it emits no `reference_token_id:`
markers and the parser aborts.)

## Step 4 — Budget matrix + beat-Turbo A/B

**4a — Budget matrix (confirm routing tiers + relative speed):**
```
cd <repo>/zig/pkg/inference
for BUD in 6144 8192 16384 24576; do
  case $BUD in 6144) KG=7.0;; 8192) KG=9.0;; *) KG=15.0;; esac
  ANTFLY_GEMMA4_COMPACT_TIMING=1 python3 scripts/run_guarded_a4b.py --kill-gb $KG \
    --log /tmp/a4b-b$BUD.log 900 \
    zig-out/bin/antfly-inference generate <model.gguf> "Hello upon-" \
    --backend metal --memory-budget-mb $BUD --raw-prompt --temperature 0 \
    --cache-dtype f16 --ignore-eos --print-timing --print-token-ids --max-tokens 128
  sleep 30
done
grep -hE 'decode_tok_per_s=|resident_slots=|watchdog:' /tmp/a4b-b*.log
```
Expected routing: 6144→partial(45 slots), 8192→partial(67), 16384→full(128),
24576→full(128). `auto` is bit-exact to streamed at every tier — only speed changes.
Full-residency rows are where decode should jump (drains eliminated).

**4b — Paired Turbo A/B (needs the real Turbo binary + model).** The shipped
`benchmark_gemma4_a4b_turbo.py` **hard-codes `--memory-profile 2gbs`** (2 GB floor =
routing OFF) and defaults `--max-phys-footprint-bytes` to 2 GB — so **as-shipped it
does NOT test full residency.** Two required edits before the beat-Turbo run:
1. In `command_lines()` add `"--memory-budget-mb", "16384"` to the Antfly argv.
2. Pass `--max-phys-footprint-bytes 16106127360` (15 GiB) so the ~13 GB run isn't
   footprint-failed.
Then:
```
python3 scripts/benchmark_gemma4_a4b_turbo.py \
  --antfly-bin zig-out/bin/antfly-inference --antfly-model <model.gguf> \
  --turbo-bin <TURBO_BIN> --turbo-model <TURBO_MODEL>.gturbo \
  --prompt "Hello upon-" --tokens 128 --warmups 1 --repeats 3 \
  --max-phys-footprint-bytes 16106127360 --max-decode-cv 0.03 \
  --require-antfly-win --out-dir /tmp/a4b-turbo-full
```
It alternates engine order per repeat (thermal-fair), takes the median, and gates on:
decode CV ≤ 3 % (else "no claim" — let the mini cool, add repeats, don't loosen),
exact token counts, and `--require-antfly-win` (Antfly decode > Turbo AND Antfly
TTFT < Turbo). **Beating Turbo = Antfly full-residency decode tok/s > Turbo's 25-26.**

If you don't have the Turbo binary, the manual interleaved A/B in 4a (compare the
16384 full-residency `decode_tok_per_s` against Turbo's known 25-26) is the fallback.

## Step 5 — The decision this produces
- **Step 3a all-identical + Step 4 beats Turbo →** make full-residency device routing
  the unconditional default at the 128-slot tier (`auto` already selects it; nothing
  to change) and record the beat-Turbo number.
- **Step 3a shows occasional flips →** the ~5e-8 softmax is user-visible on near-ties.
  Keep `full` opt-in (via `ANTFLY_GEMMA4_DEVICE_ROUTING=full`) rather than the auto
  default, or accept it as a documented distinct-but-correct path. Record the flip
  rate so the trade (irreducible ~5e-8 for zero-drain speed) is an informed choice.

---

## Cheat-sheet
| Budget (`--memory-budget-mb`) | Slots | `auto` route | Notes |
|---|---|---|---|
| 2048 / 4096 | 8 / 24 | off (streamed) | below the 45-slot device floor |
| 6144 / 8192 | 45 / 67 | **partial** | device FFN, per-layer drains remain (+~9.5 %) |
| 16384 / 24576 | 128 | **full** | device top-k, **zero per-layer drains** (beat-Turbo) |

- Guard: standard `run_guarded_a4b.py`, `--kill-gb 15.0` for full residency; do NOT
  use `guard_relaxed.py` (Air-only 8 % pressure hack).
- `ANTFLY_GEMMA4_DEVICE_ROUTING=off|full|partial` forces a mode for A/B (wins over
  `auto`); `…_VERIFY=1` dual-executes device vs CPU; `ANTFLY_GEMMA4_COMPACT_TIMING=1`
  prints the per-layer + flush-count telemetry.
- Model is the `.gguf` FILE; only the A4B Q4_0 geometry qualifies.
