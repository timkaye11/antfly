"use client";

import { CodeLink } from "@/components/code/code-link";
import { EnvFlagChip, QuantChip } from "@/components/primitives/chips";
import { Divergence, Scene, ScrollyChapter } from "@/components/scrollytelling/scrolly";
import { KernelTimelineFrame } from "@/components/viz/kernel-timeline-frame";
import { BytesBar, ComparisonBars, JourneyChart } from "@/components/viz/perf-charts";
import { SankeyFlow } from "@/components/viz/sankey-flow";
import { bytesBreakdown, comparisonSamples, finalsM4Pro, journeyE2bAir, machines, rooflineCeiling } from "@/content/perf";
import { L } from "@/lib/links";
import type { ChaptersProps } from "../registry";
import { ColdOpenFigure, GraphToFrameFigure, LayerStackFigure } from "./figures";
import { Gemma4EarlyChapters } from "./chapters-early";
import { Gemma4LateChapters } from "./chapters-late";
import Link from "next/link";

export function Gemma4Chapters({ spec, routes, frames }: ChaptersProps) {
  const isE4b = spec.id === "gemma4-e4b";
  const tokS = String(spec.stats.tokS ?? "");

  return (
    <div>
      {/* ── Ch 0 · Cold open ─────────────────────────────────────── */}
      <ScrollyChapter
        id="ch-0"
        number={0}
        title={`One token, ${isE4b ? "~18" : "~12"} milliseconds`}
        intro="Everything below is this moment, slowed down."
      >
        <Scene id="replay" graphic={<ColdOpenFigure tokS={`${tokS} decode · Q4_0 · greedy`} />}>
          <p>
            Watch one decode step. A token id enters on the left; {spec.stats.layers} transformer layers, a{" "}
            {Number(spec.stats.vocab).toLocaleString()}-word vocabulary matvec, and a GPU-side sampler later, a
            new token appears in the stream — {tokS.split(" ")[0]} times per second on{" "}
            {isE4b ? "an M4 Pro" : "an M4 Pro"}.
          </p>
          <p>
            The whole step is <em>one</em> Metal command frame: a single compute encoder, 143 planned dispatch
            scopes, zero barriers. The rest of this page slows that frame down until every box on this strip
            becomes a chapter.
          </p>
        </Scene>
      </ScrollyChapter>

      {/* ── Ch 1 · The shape of the model ─────────────────────────── */}
      <ScrollyChapter
        id="ch-1"
        number={1}
        title={`The shape of ${spec.displayName}`}
        intro={
          <p>
            {spec.stats.layers} layers, hidden size {spec.stats.hidden} — but the interesting structure is in
            which layers get which attention, and which layers own a KV cache at all.
          </p>
        }
      >
        <Scene id="iswa" graphic={<LayerStackFigure spec={spec} emphasis="attention" />}>
          <p>
            <strong>The 5:1 iSWA rhythm.</strong> Five of every six layers attend through a sliding window
            (shuttered rows); every sixth layer is a full global-attention layer. The pattern is one line of
            config logic — <code>(layer + 1) % sliding_window_pattern != 0</code> — with the pattern defaulting
            to 6.
          </p>
          <p>
            <CodeLink link={L("config-layer-uses-sliding")} /> · <CodeLink link={L("config-sliding-pattern")} />
          </p>
        </Scene>
        <Scene id="headdim" graphic={<LayerStackFigure spec={spec} emphasis="headdim" />}>
          <p>
            <strong>Two head sizes.</strong> Global layers run <code>head_dim 512</code>; sliding layers run{" "}
            <code>256</code>. Long-range retrieval gets double the rank exactly where it's needed, and the
            attention kernels carry both shapes (<code>hd512</code> has its own generated flash-prefill
            variant).
          </p>
          <p>
            <CodeLink link={L("config-global-head-dim")} />
          </p>
        </Scene>
        <Scene id="sharedkv" graphic={<LayerStackFigure spec={spec} emphasis="kv" />}>
          <p>
            <strong>Most layers don't own a KV cache.</strong> On {spec.displayName}, only{" "}
            {spec.stats.kvOwners} of {spec.stats.layers} layers write K/V; the {spec.stats.sharedKv} tail
            layers skip the K/V projection entirely and read a donor layer's blocks. That's{" "}
            {isE4b ? "18" : "20"} layers of KV memory and KV-write bandwidth that simply don't exist.
          </p>
          <Divergence
            others={<p>every layer projects and stores its own K/V; KV memory scales with depth.</p>}
            antfly={<p>shared-KV tail layers hold zero pages — the predicate is one comparison in the config.</p>}
            link={<CodeLink link={L("config-shared-kv")} />}
          />
        </Scene>
        <Scene id="bytes" graphic={<div className="flex h-full flex-col justify-center">{spec.sankey && <SankeyFlow spec={spec.sankey} />}</div>}>
          <p>
            <strong>Follow the bytes.</strong> Decode is memory-bound, so architecture <em>is</em> the byte
            budget: the FFN moves 65.7% of all weight traffic per token, the Q6_K LM head 19.5%, attention
            11.7%. Keep this flow in mind — chapters 8, 9 and 12 are about shaving these ribbons.
          </p>
        </Scene>
      </ScrollyChapter>

      <Gemma4EarlyChapters spec={spec} routes={routes} frames={frames} />

      {/* ── Ch 7 · From graph to frames ──────────────────────────── */}
      <ScrollyChapter
        id="ch-7"
        number={7}
        title="From graph to frames"
        intro={
          <p>
            This is the heart of the runtime: the op graph is not walked, it is <em>planned</em> — once — into
            a command frame the Metal runtime replays every step.
          </p>
        }
      >
        <Scene id="graph" graphic={<GraphToFrameFigure step={0} />}>
          <p>
            A decode step starts as a DAG of ops per layer: norm, QKV, rope, attention, projections, the gated
            FFN. llama.cpp walks a graph like this and encodes commands op-by-op, every token.
          </p>
          <p>
            Antfly's planner instead compiles it into a <code>FrameDescriptor</code>: an ordered list of{" "}
            <code>PlannedOp</code>s grouped into <code>EncoderScope</code>s, with family-specific lowerers for
            attention setup, gated layers, PLE, and the tail.
          </p>
          <p>
            <CodeLink link={L("planner-frame-descriptor")} /> · <CodeLink link={L("planner-encoder-scope")} />
          </p>
        </Scene>
        <Scene id="barriers" graphic={<GraphToFrameFigure step={1} />}>
          <p>
            <strong>Barriers only at real hazards.</strong> Within a frame, the runtime tracks read/write byte
            ranges and emits a Metal buffer barrier only where a genuine RAW/WAR/WAW hazard exists. The old
            Q8_0 anchor frame still carried <span className="font-mono">41 encoders and 422 barriers</span> —
            every red tick a point where the GPU serialized.
          </p>
        </Scene>
        <Scene id="zero" graphic={<GraphToFrameFigure step={2} />}>
          <p>
            The barrier reframe finished the job: the live Q4_0 decode frame submits{" "}
            <span className="font-mono text-primary">1 compute encoder, 143 planned scopes, planned_barriers = 0</span>{" "}
            — whole-frame scoped suppression on a serial encoder, and the hazard scan itself dropped from 0.6
            to 0.1 ms per frame.
          </p>
          <Divergence
            others={<p>ggml re-encodes its graph per token with coarse encoder granularity.</p>}
            antfly={<p>one planned frame, re-bound per step; barriers exist only where byte ranges actually collide.</p>}
            link={<CodeLink link={L("runtime-submit-frame")} />}
          />
        </Scene>
        <Scene
          id="pipelined"
          graphic={
            <div className="flex h-full flex-col justify-center">
              <KernelTimelineFrame scenario={frames.q40} prevScenario={frames.q40} />
            </div>
          }
        >
          <p>
            <strong>And then the frames overlap.</strong> Because the sampled token id stays device-resident
            (chapter 10), frame N+1 can be <em>encoded from a token that doesn't exist yet on the host</em> —
            the CPU encodes the next frame before waiting on the current one. The submit→wait→encode bubble
            disappears: +9–12% on E2B, +5–8% on E4B, token-identical.
          </p>
          <p>
            <CodeLink link={L("executor-pipelined-decode")} />{" "}
            <EnvFlagChip name="TERMITE_METAL_ENABLE_PIPELINED_DECODE_FRAME" defaultOn />
          </p>
          <p className="text-xs">
            Full view: <Link className="text-primary underline" href="/systems/timeline">frame timeline →</Link>
          </p>
        </Scene>
      </ScrollyChapter>

      <Gemma4LateChapters spec={spec} routes={routes} frames={frames} />

      {/* ── Ch 12 · Scoreboard ───────────────────────────────────── */}
      <ScrollyChapter
        id="ch-12"
        number={12}
        title="The scoreboard: 44 → 56.5, and the road to the roofline"
        intro={
          <p>
            Every claim on this page is measured, machine-labeled, and carries its caveats. Decode is
            memory-bound, so the ceiling is arithmetic: bytes-per-token ÷ bandwidth.
          </p>
        }
      >
        <Scene id="journey" graphic={<div className="flex h-full flex-col justify-center"><JourneyChart entries={journeyE2bAir} /></div>}>
          <p>
            <strong>The journey.</strong> On the fanless M4 Air worktree, E2B went from ~44 tok/s at branch
            start to 56.5 (+25%): pipelined decode frames, the Q4_K LM-head repack, pair-fusion, PLE Q8_0
            staging. Refuted levers are part of the record too — the sumsq fusion was <em>−12%</em>, measured
            and struck.
          </p>
          <p className="text-xs text-muted-foreground">{machines.air}</p>
        </Scene>
        <Scene
          id="compare"
          graphic={
            <div className="flex h-full flex-col justify-center gap-4">
              <ComparisonBars samples={comparisonSamples} ceiling={rooflineCeiling} />
            </div>
          }
        >
          <p>
            <strong>Against the field</strong> (E4B Q4_0, 64-token circus, M4 Pro): Antfly 62.4 internal / 54.9
            end-to-end vs llama.cpp 72.2, Ollama 75.3, vLLM-Metal 86.6 — against a ~96 tok/s roofline at 2.829
            GB/token. Hover the asterisks: vLLM may include its MTP proposer, and some llama.cpp builds skip
            PLE entirely.
          </p>
          <p>
            <CodeLink link={L("perf-plan-comparison")} />
          </p>
        </Scene>
        <Scene
          id="finals"
          graphic={
            <div className="flex h-full flex-col justify-center gap-4">
              <BytesBar entries={bytesBreakdown} />
              <div className="grid gap-3">
                {finalsM4Pro.map((f) => (
                  <div key={f.model} className="rounded-lg border p-3">
                    <span className="text-2xl font-bold tabular-nums">{f.antfly}</span>
                    <span className="ml-2 text-sm text-muted-foreground">
                      tok/s · {f.model} · {f.pct} of llama.cpp
                    </span>
                  </div>
                ))}
              </div>
            </div>
          }
        >
          <p>
            <strong>Where it stands now.</strong> After the split-GQA floor retune, the pinned M4 Pro box
            measures E2B at 80.7 tok/s (74% of llama.cpp) and E4B at 56.1 (89%) — with the full model running,
            PLE included. The remaining gap is mostly the byte diet: the un-tuned Q6_K tail and vLLM's leaner
            4-bit packing.
          </p>
          <p className="text-xs">
            Full data with caveats: <Link className="text-primary underline" href="/systems/perf">the scoreboard →</Link>
          </p>
        </Scene>
      </ScrollyChapter>
    </div>
  );
}
