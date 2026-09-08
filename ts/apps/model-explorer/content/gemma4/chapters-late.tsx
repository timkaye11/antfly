"use client";

/**
 * Gemma4 chapters 8–11: the kernel compiler, the LM-head case study,
 * device-resident sampling, and MTP.
 */
import Link from "next/link";
import { CodeLink } from "@/components/code/code-link";
import { QuantChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import {
  CompilerPipelineFigure,
  GpuSamplingFigure,
  KernelCensusFigure,
  LmHeadPathsFigure,
  LmHeadZoomFigure,
  MtpScoreboardFigure,
  MtpSideBySideFigure,
  ProposeVerifyFigure,
  ScheduleRowFigure,
  TokenHandoffFigure,
} from "./figures-late";

function KernelChip({ name }: { name: string }) {
  return (
    <span className="inline-flex items-center rounded-sm border bg-muted/50 px-1.5 py-px font-mono text-[10px]">
      {name}
    </span>
  );
}

export function Gemma4LateChapters({ spec, routes }: ChaptersProps) {
  const isE4b = spec.id === "gemma4-e4b";
  const q4kRoute = routes.find((r) => r.id === "q4_k/rows_2_8/none");

  return (
    <div>
      {/* ── Ch 8 · Kernels are compiled, not written ─────────────── */}
      <ScrollyChapter
        id="ch-8"
        number={8}
        title="Kernels are compiled, not written (mostly)"
        intro={
          <p>
            The matvec kernels that move most of the bytes are not hand-typed Metal. They are
            rendered at build time from a schedule table — and the build fails if regeneration
            changes a byte.
          </p>
        }
      >
        <Scene id="pipeline" graphic={<CompilerPipelineFigure />}>
          <p>
            One schedule table describes every route as <code>format × row_bucket × epilogue</code>{" "}
            plus tuning knobs; one renderer expands each row through a shared MSL skeleton into a
            checked-in <code>.metal</code> file — 25 quant-kernel files today, byte-identical on
            every regen (<code>zig build quant-kernel-codegen -- --check</code>). This is a
            build-time renderer, not a runtime JIT: at serve time the kernels are as static as
            anyone else's.
          </p>
          <p>
            <CodeLink link={L("compiler-schedules")} /> · <CodeLink link={L("renderer")} />
          </p>
        </Scene>
        <Scene id="row" graphic={<ScheduleRowFigure route={q4kRoute} />}>
          <p>
            Here is the row behind chapter 9's LM-head kernel: <QuantChip format="q4_k" />, the
            2–8-row bucket, no epilogue — 128 threads per threadgroup, 16 columns, 2 rows,
            simdgroup-tiled reduction. When a sweep finds a better configuration, the fix is an edit
            to this row and a regenerate, and the diff shows exactly what changed in the emitted
            Metal.
          </p>
        </Scene>
        <Scene id="census" graphic={<KernelCensusFigure />}>
          <p>
            The full inventory is 413 kernels. The generated matvec family is the biggest block; the
            hand-written remainder is where generation hasn't paid for itself yet — attention,
            fusion epilogues, sampling, KV plumbing.
          </p>
          <Divergence
            others={
              <p>
                llama.cpp hand-maintains a kernel zoo per quant format, tuned by accumulated
                patches.
              </p>
            }
            antfly={
              <p>
                one schedule table is the single source of truth; re-tuning a route is a table edit
                plus regenerate.
              </p>
            }
            link={<CodeLink link={L("compiler-schedules")} />}
          />
          <p className="text-xs">
            Browse all 413:{" "}
            <Link className="text-primary underline" href="/systems/kernels">
              the kernel census →
            </Link>
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 9 · The LM head problem ───────────────────────────── */}
      <ScrollyChapter
        id="ch-9"
        number={9}
        title="The LM head problem (a worked example)"
        intro={
          <p>
            One tensor, one fifth of all traffic. This is what a perf investigation actually looks
            like in this codebase: a straggler, a fix, and a refuted branch kept in the record.
          </p>
        }
      >
        <Scene id="zoom" graphic={<LmHeadZoomFigure isE4b={isE4b} />}>
          <p>
            Every decode step ends by multiplying the hidden state against all 262,144 vocabulary
            rows — a <code>[{isE4b ? "2560" : "1536"} × 262144]</code> matvec. On E4B that segment
            is 550 MB per token, 19.5% of all weight traffic, and it ran through an un-tuned{" "}
            <QuantChip format="q6_k" /> path: the gap analysis priced a proper kernel at +7–12
            tok/s.
          </p>
        </Scene>
        <Scene id="paths" graphic={<LmHeadPathsFigure />}>
          <p>
            <strong>The fix that shipped:</strong> repack the Q6_K head to{" "}
            <QuantChip format="q4_k" /> in a streaming pass at load, then dispatch the tuned{" "}
            <KernelChip name="termite_q4_k_linear_1x_reduce_v2" /> MMV. E2B went 52.5→54.3–55.2
            tok/s and E4B 28.9→30.1 (+4–5%) on the Air, token-identical. The sampler's candidate
            pass still rescores through <KernelChip name="termite_lm_head_q6_k_rescore_top8" />{" "}
            where the Q6_K weights apply.
          </p>
          <p>
            <strong>The branch that didn't:</strong> a <QuantChip format="q4_0" /> head looked like
            free bytes and collapsed generation to an instant end-of-turn token. Measured, struck
            through, and kept — the ghost path is part of the engineering record, not an
            embarrassment to hide.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 10 · Sampling never leaves the GPU ────────────────── */}
      <ScrollyChapter
        id="ch-10"
        number={10}
        title="Sampling never leaves the GPU"
        intro={
          <p>
            The obvious way to sample: copy a million logit bytes to the CPU and pick there, every
            token. Antfly doesn't.
          </p>
        }
      >
        <Scene id="onchip" graphic={<GpuSamplingFigure />}>
          <p>
            The whole sampler is dispatches inside the frame: a top-8 candidate rescore over the
            logits (<KernelChip name="termite_lm_head_top8_reduce" />
            ), Gumbel-max partials for temperature sampling, and a final argmax (
            <KernelChip name="termite_argmax_logits_reduce" />) that writes a single token id into a
            device buffer. Greedy and sampled decoding share the path — the crossed-out arrow is the
            per-token logits readback that never happens.
          </p>
          <p>
            <CodeLink link={L("kernel-gumbel")} />
          </p>
          <Divergence
            others={
              <p>llama.cpp reads the logits back to the host and samples on the CPU each token.</p>
            }
            antfly={
              <p>top-8 rescore + Gumbel-max + argmax run on device; only text ever crosses back.</p>
            }
            link={<CodeLink link={L("kernel-gumbel")} />}
          />
        </Scene>
        <Scene id="handoff" graphic={<TokenHandoffFigure />}>
          <p>
            The payoff is bigger than saved copies: because the sampled id lives in a GPU buffer,
            the next frame's embedding lookup can consume it directly. That device-resident handoff
            is what makes{" "}
            <Link className="text-primary underline" href="#ch-7">
              chapter 7
            </Link>
            &apos;s pipelined decode frame legal — the CPU encodes frame N+1 against a token the
            host has never seen.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 11 · MTP ──────────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-11"
        number={11}
        title="MTP: a draft model that reads the main model's mind"
        intro={
          <p>
            Gemma-4 ships an official speculative drafter — a 4-layer, hidden-256 stack with 4
            attention heads and 1 KV head that doesn't keep its own memory. It borrows the main
            model's.
          </p>
        }
      >
        <Scene id="donor" graphic={<MtpSideBySideFigure isE4b={isE4b} />}>
          <p>
            The drafter is query-only: it has Q/O projections and MLPs but no K/V projections at
            all. Its layers cross-attend the main model's KV banks through donor layers —{" "}
            {isE4b
              ? "layers 22 and 23 on E4B"
              : "sliding donor layer 13 and full-attention donor layer 14 on E2B"}{" "}
            — with a projection pair bridging its 256-wide stack to the {isE4b ? "2560" : "1536"}
            -wide backbone. A draft model that literally reads the target&apos;s working memory.
          </p>
          <p>
            <CodeLink link={L("mtp-kv-donor")} /> · <CodeLink link={L("mtp-draft-request")} />
          </p>
        </Scene>
        <Scene id="verify" graphic={<ProposeVerifyFigure />}>
          <p>
            The loop: the drafter proposes k tokens cheaply, then the main model verifies all k
            positions in one batched step. Every accepted token is a full decode step the big model
            skipped; the first mismatch truncates the tail and costs nothing but the draft. Output
            is exactly what the main model alone would have produced.
          </p>
        </Scene>
        <Scene id="board" graphic={<MtpScoreboardFigure />}>
          <p>
            <strong>And here's the honest part.</strong> On CUDA and in llama.cpp this drafter is
            worth 2–3×. On Antfly's Metal path it is currently net-slower — E2B measured 75.9 tok/s
            target-only vs 63.7 with the BF16 draft at 64% acceptance, so the auto-gate disables it:{" "}
            <em>&quot;MTP correctness/fallback PASS; Metal-auto performance FAIL.&quot;</em> The
            machinery — dedicated draft runtime, draft frames, device-resident accept path — is
            built, measured, and default-off. It's the lever that works everywhere but here, yet.
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
