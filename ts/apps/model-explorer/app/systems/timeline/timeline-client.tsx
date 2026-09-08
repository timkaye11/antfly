"use client";

import { Tabs, TabsList, TabsTrigger } from "@antfly/design-system";
import { parseAsString, useQueryState } from "nuqs";
import { Suspense } from "react";
import { CodeLink } from "@/components/code/code-link";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { KernelTimelineFrame } from "@/components/viz/kernel-timeline-frame";
import { L } from "@/lib/links";
import type { FrameScenario } from "@/lib/schema";

interface TimelineProps {
  frames: { q40: FrameScenario; q80: FrameScenario };
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
}

export function TimelineClient(props: TimelineProps) {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <TimelineInner {...props} />
    </Suspense>
  );
}

function TimelineInner({ frames, snippets, gitCommit, permalinkBase }: TimelineProps) {
  const [frame, setFrame] = useQueryState("frame", parseAsString.withDefault("q40"));
  const scenario = frame === "q80" ? frames.q80 : frames.q40;
  const isQ40 = frame !== "q80";

  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="mx-auto max-w-6xl space-y-8 px-4 py-8">
        <header className="max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight">Frame timeline</h1>
          <p className="mt-2 text-muted-foreground">
            One Metal command frame — one decode step — laid out Perfetto-style. Both views here are{" "}
            <em>planned</em> mode: the structure the planner submits (encoder scopes, planned ops, barriers),
            with op widths proportional to <em>estimated bytes moved</em>, not measured GPU time. A captured
            mode with real timings would draw the same picture from <code className="font-mono text-sm">gpuNanos</code>{" "}
            instead.
          </p>
        </header>

        <section>
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-lg font-semibold">{scenario.title}</h2>
            <Tabs value={isQ40 ? "q40" : "q80"} onValueChange={setFrame}>
              <TabsList className="h-7">
                <TabsTrigger value="q80" className="h-6 px-2 text-xs">
                  Q8_0 anchor (41 encoders · 422 barriers · 36 scopes)
                </TabsTrigger>
                <TabsTrigger value="q40" className="h-6 px-2 text-xs">
                  live Q4_0 (1 encoder · 143 scopes · 0 barriers)
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          {scenario.description && (
            <p className="mb-4 max-w-3xl text-sm text-muted-foreground">{scenario.description}</p>
          )}
          <div className="rounded-lg border bg-card p-4">
            <KernelTimelineFrame scenario={scenario} prevScenario={isQ40 ? frames.q40 : undefined} />
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <div className="rounded-lg border p-4">
            <h3 className="mb-2 text-sm font-semibold">Encoder scopes</h3>
            <p className="text-xs text-muted-foreground">
              The brackets over the GPU lane are <code className="font-mono">EncoderScope</code>s — the
              planner's grouping of planned ops under one compute encoder. In the Q8_0 anchor, scope boundaries
              were also encoder boundaries (41 of them); the live Q4_0 frame keeps the scopes as planning
              structure but submits them all on a single serial encoder.
            </p>
            <p className="mt-2 text-xs">
              <CodeLink link={L("planner-encoder-scope")} /> · <CodeLink link={L("runtime-begin-frame")} />
            </p>
          </div>
          <div className="rounded-lg border p-4">
            <h3 className="mb-2 text-sm font-semibold">Barriers (hover the red ticks)</h3>
            <p className="text-xs text-muted-foreground">
              Each red tick is a memory barrier the planner emitted for a RAW/WAR/WAW hazard between tracked
              byte ranges — a point where the GPU serializes. The anchor frame carried 422 of them; the barrier
              reframe proved them unnecessary under whole-frame scoped suppression on a serial encoder, and the
              live frame plans zero. Ticks shown are representative, not all 422.
            </p>
          </div>
          <div className="rounded-lg border p-4">
            <h3 className="mb-2 text-sm font-semibold">The pipelined third lane</h3>
            <p className="text-xs text-muted-foreground">
              On the live Q4_0 view, a third lane appears: because the sampled token id stays device-resident,
              the CPU encodes frame N+1 <em>before</em> waiting on frame N. The submit→wait→encode bubble
              disappears — worth +9–12% on E2B, token-identical.
            </p>
            <p className="mt-2 text-xs">
              <CodeLink link={L("executor-pipelined-decode")} />
            </p>
          </div>
        </section>

        <p className="text-xs text-muted-foreground">
          Scope contents are representative (one layer plus the tail), drawn from the planner's structure —
          widths are log-scaled estimated bytes, so a 30 MB FFN matvec and a 4 KB norm are both visible.
        </p>
      </div>
    </SnippetProvider>
  );
}
