"use client";

import { Button } from "@antfly/design-system";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { type ClientSnippet, SnippetProvider } from "@/components/code/snippet-context";
import { SpineStrip, type SpineStageId } from "@/components/spine-strip";
import { getChapters } from "@/content/registry";
import type { FrameScenario, KernelRoute, ModelSpec } from "@/lib/schema";

export interface ModelFrames {
  q40: FrameScenario;
  q80Anchor: FrameScenario;
}

export function ModelPageClient({
  spec,
  routes,
  frames,
  snippets,
  gitCommit,
  permalinkBase,
}: {
  spec: ModelSpec;
  routes: KernelRoute[];
  frames: ModelFrames;
  snippets: Record<string, ClientSnippet>;
  gitCommit: string;
  permalinkBase?: string;
}) {
  const Chapters = getChapters(spec.id);
  const modified = [...new Set(spec.stages.map((s) => s.spine).filter((s): s is SpineStageId => !!s))];

  return (
    <SnippetProvider snippets={snippets} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div>
        <header className="border-b bg-muted/20">
          <div className="mx-auto max-w-7xl px-4 py-10">
            <p className="font-mono text-xs uppercase tracking-wider text-muted-foreground">{spec.family}</p>
            <h1 className="mt-1 text-4xl font-bold tracking-tight">{spec.displayName}</h1>
            <p className="mt-3 max-w-2xl text-lg text-muted-foreground">{spec.tagline}</p>
            <div className="mt-6 flex flex-wrap gap-x-8 gap-y-3">
              {Object.entries(spec.stats).map(([k, v]) => (
                <div key={k}>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
                  <div className="font-mono text-sm font-semibold">{v}</div>
                </div>
              ))}
            </div>
            <div className="mt-6 flex flex-wrap items-center gap-4">
              <SpineStrip modified={modified} withVision={spec.id === "qwen3-vl"} />
              <Button variant="outline" size="sm" asChild>
                <Link href={`/explore/${spec.id}`}>
                  Open in DAG explorer <ArrowRight className="size-3.5" />
                </Link>
              </Button>
            </div>
          </div>
        </header>
        <Chapters spec={spec} routes={routes} frames={frames} />
      </div>
    </SnippetProvider>
  );
}
