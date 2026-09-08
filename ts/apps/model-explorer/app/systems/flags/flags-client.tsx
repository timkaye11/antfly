"use client";

import { Badge, cn, Input } from "@antfly/design-system";
import { parseAsString, useQueryState } from "nuqs";
import { Suspense, useMemo } from "react";
import { CodeLink } from "@/components/code/code-link";
import { SnippetProvider } from "@/components/code/snippet-context";
import type { SourceLink } from "@/lib/schema";

interface FlagRow {
  name: string;
  kind: string;
  occurrences: number;
  source: SourceLink;
}

const PREFIXES = [
  ["all", "all"],
  ["TERMITE_METAL_", "metal"],
  ["TERMITE_", "termite (other)"],
  ["ANTFLY_GEMMA4_", "gemma4 / MTP"],
  ["ANTFLY_CUDA_", "cuda"],
  ["ANTFLY_", "antfly (other)"],
] as const;

export function FlagsClient(props: { flags: FlagRow[]; gitCommit: string; permalinkBase?: string }) {
  return (
    <Suspense fallback={<div className="min-h-screen" />}>
      <FlagsInner {...props} />
    </Suspense>
  );
}

function FlagsInner({ flags, gitCommit, permalinkBase }: { flags: FlagRow[]; gitCommit: string; permalinkBase?: string }) {
  const [q, setQ] = useQueryState("q", parseAsString.withDefault(""));
  const [prefix, setPrefix] = useQueryState("prefix", parseAsString.withDefault("all"));

  const filtered = useMemo(() => {
    const needle = q.toUpperCase();
    return flags.filter((f) => {
      if (prefix !== "all") {
        if (!f.name.startsWith(prefix)) return false;
        // Keep the buckets disjoint: "TERMITE_" excludes "TERMITE_METAL_", etc.
        if (prefix === "TERMITE_" && f.name.startsWith("TERMITE_METAL_")) return false;
        if (prefix === "ANTFLY_" && (f.name.startsWith("ANTFLY_GEMMA4_") || f.name.startsWith("ANTFLY_CUDA_"))) return false;
      }
      return needle === "" || f.name.includes(needle);
    });
  }, [flags, q, prefix]);

  return (
    <SnippetProvider snippets={{}} gitCommit={gitCommit} permalinkBase={permalinkBase}>
      <div className="mx-auto max-w-5xl space-y-6 px-4 py-8">
        <header className="max-w-3xl">
          <h1 className="text-3xl font-bold tracking-tight">Tuning env flags</h1>
          <p className="mt-2 text-muted-foreground">
            Every kernel route, fusion, and frame behavior in the runtime is gated by an environment flag —{" "}
            {flags.length} of them, scanned from the Zig source. Most are kill-switches for default-on
            optimizations (green = enabling, others disable/trace/force).
          </p>
        </header>

        <div className="flex flex-wrap gap-1.5">
          {PREFIXES.map(([p, label]) => (
            <button
              key={p}
              type="button"
              onClick={() => setPrefix(p)}
              className={cn(
                "rounded-full border px-2.5 py-0.5 font-mono text-xs transition-colors",
                prefix === p ? "border-primary text-primary" : "text-muted-foreground hover:bg-accent",
              )}
            >
              {label}
            </button>
          ))}
        </div>
        <Input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Filter flags… (e.g. PIPELINED, GQA_SPLIT, MTP)"
          className="max-w-md font-mono text-sm"
        />

        <div className="space-y-0.5">
          {filtered.slice(0, 200).map((f) => (
            <div key={f.name} className="flex items-center gap-3 rounded border-b px-2 py-1.5 text-sm last:border-0">
              <span className="min-w-0 flex-1 truncate font-mono text-xs">{f.name}</span>
              <Badge className="shrink-0 text-[9px]">{f.kind}</Badge>
              <span className="w-14 shrink-0 text-right font-mono text-[10px] text-muted-foreground">
                ×{f.occurrences}
              </span>
              <CodeLink link={f.source} className="shrink-0" />
            </div>
          ))}
        </div>
        {filtered.length > 200 && (
          <p className="text-xs text-muted-foreground">Showing 200 of {filtered.length} — narrow the filter.</p>
        )}
      </div>
    </SnippetProvider>
  );
}
