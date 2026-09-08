"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@antfly/design-system";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { Divergence } from "@/components/scrollytelling/scrolly";
import { BytesBar, ComparisonBars, GapWaterfall, JourneyChart } from "@/components/viz/perf-charts";
import {
  bytesBreakdown,
  censusStory,
  comparisonSamples,
  finalsM4Pro,
  gapSegments,
  journeyE2bAir,
  machines,
  refutedLevers,
  rooflineCeiling,
  splitGqaRetune,
} from "@/content/perf";

export function PerfClient({
  snippets,
  gitCommit,
  permalinkBase,
}: {
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
}) {
  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="mx-auto max-w-6xl space-y-12 px-4 py-8">
        <header className="max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight">The scoreboard</h1>
          <p className="mt-2 text-muted-foreground">
            Every number on this page is hand-transcribed from{" "}
            <code className="font-mono text-sm">GEMMA4_PERF_PLAN.md</code> with its caveats attached (hover the{" "}
            <span className="text-primary">*</span>). Decode is memory-bound: the roofline is bytes-per-token ÷
            memory bandwidth, so machine identity is on every number.
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1fr_320px]">
          <div>
            <h2 className="mb-1 text-lg font-semibold">Gemma4 E4B Q4_0 · 64-token decode circus</h2>
            <p className="mb-4 text-xs text-muted-foreground">{machines.pro}</p>
            <ComparisonBars samples={comparisonSamples} ceiling={rooflineCeiling} />
            <Divergence
              className="mt-6"
              others={
                <p>
                  vLLM-Metal leads at 227 GB/s effective — on ~7.5% fewer bytes/token (MLX 4-bit packing), and
                  possibly with its MTP proposer engaged.
                </p>
              }
              antfly={
                <p>
                  Antfly runs the full model — PLE included, which some llama.cpp builds skip (issue #22243) —
                  and closed the gap to 89% of llama.cpp on E4B by the final retune below.
                </p>
              }
            />
          </div>
          <aside>
            <h3 className="mb-2 font-mono text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Where the bytes go (E4B, per token)
            </h3>
            <BytesBar entries={bytesBreakdown} />
            <p className="mt-3 text-xs text-muted-foreground">
              This bar explains the whole page: decode speed is bytes moved per token. FFN dominates, the LM
              head is the single biggest tensor, and everything else is noise.
            </p>
          </aside>
        </section>

        <section>
          <h2 className="mb-1 text-lg font-semibold">Gap decomposition — from 62.4 to the roofline</h2>
          <p className="mb-4 text-xs text-muted-foreground">
            5.66 ms/token of excess vs the 2.829 GB/token roofline, bucketed. Green = fix landed.
          </p>
          <GapWaterfall
            base={{ label: "Antfly internal decode", value: 62.4 }}
            segments={gapSegments}
            ceiling={{ label: rooflineCeiling.label, value: rooflineCeiling.value }}
          />
          <p className="mt-3 max-w-3xl text-sm text-muted-foreground">
            The barrier bucket is the census story: the Q8_0 anchor frame ran{" "}
            <span className="font-mono">{censusStory.q8Anchor.encoders} encoders / {censusStory.q8Anchor.barriers} barriers</span>;
            the live Q4_0 frame submits{" "}
            <span className="font-mono">
              {censusStory.q4Live.encoders} compute encoder / {censusStory.q4Live.plannedScopes} planned scopes /{" "}
              {censusStory.q4Live.plannedBarriers} barriers
            </span>{" "}
            under whole-frame scoped suppression.
          </p>
        </section>

        <section>
          <h2 className="mb-1 text-lg font-semibold">The journey — E2B on the fanless Air</h2>
          <p className="mb-4 text-xs text-muted-foreground">
            {machines.air}. Branch start ~44 tok/s → 56.5 (+25%). Hover the points for A/B numbers.
          </p>
          <JourneyChart entries={journeyE2bAir} />
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg border p-4">
              <h3 className="mb-2 text-sm font-semibold">Refuted levers (measured, then struck)</h3>
              <ul className="space-y-1.5 text-xs text-muted-foreground">
                {refutedLevers.map((r) => (
                  <li key={r.label}>
                    <span className="line-through">{r.label}</span> — {r.note}
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-lg border p-4">
              <h3 className="mb-2 text-sm font-semibold">{splitGqaRetune.label}</h3>
              <p className="text-xs text-muted-foreground">{splitGqaRetune.detail}</p>
            </div>
          </div>
        </section>

        <section>
          <h2 className="mb-4 text-lg font-semibold">Finals on the pinned reference box</h2>
          <div className="grid gap-4 sm:grid-cols-2">
            {finalsM4Pro.map((f) => (
              <Card key={f.model}>
                <CardHeader>
                  <CardTitle className="text-base">Gemma4 {f.model}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-baseline gap-2">
                    <span className="text-3xl font-bold tabular-nums">{f.antfly}</span>
                    <span className="text-sm text-muted-foreground">tok/s · {f.pct} of llama.cpp ({f.llama})</span>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">{f.context}</p>
                </CardContent>
              </Card>
            ))}
          </div>
          <p className="mt-3 text-xs text-muted-foreground">{machines.pro}</p>
        </section>
      </div>
    </SnippetProvider>
  );
}
