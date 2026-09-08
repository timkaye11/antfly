"use client";

import { cn } from "@antfly/design-system";
import Link from "next/link";

export const SPINE_STAGES = [
  { id: "http", label: "HTTP" },
  { id: "session", label: "Session" },
  { id: "tokenizer", label: "Tokenizer" },
  { id: "graph", label: "Graph" },
  { id: "frames", label: "Frames" },
  { id: "kernels", label: "Kernels" },
  { id: "kv", label: "KV" },
  { id: "sample", label: "Sample" },
] as const;

export type SpineStageId = (typeof SPINE_STAGES)[number]["id"] | "vision";

/**
 * The persistent runtime-pipeline breadcrumb. On model pages, stages the
 * model specializes get a "modified" dot; clicking jumps to that model's
 * chapter for the stage if one exists, else to /runtime#<stage>.
 */
export function SpineStrip({
  modified = [],
  active,
  links = {},
  withVision = false,
  className,
}: {
  modified?: SpineStageId[];
  active?: SpineStageId;
  /** stage id -> href override (e.g. "#ch-7" on a model page). */
  links?: Partial<Record<SpineStageId, string>>;
  withVision?: boolean;
  className?: string;
}) {
  const stages: Array<{ id: SpineStageId; label: string }> = withVision
    ? [
        { id: "http", label: "HTTP" },
        { id: "session", label: "Session" },
        { id: "tokenizer", label: "Tokenizer" },
        { id: "vision", label: "Vision" },
        { id: "graph", label: "Graph" },
        { id: "frames", label: "Frames" },
        { id: "kernels", label: "Kernels" },
        { id: "kv", label: "KV" },
        { id: "sample", label: "Sample" },
      ]
    : [...SPINE_STAGES];

  return (
    <nav aria-label="Runtime pipeline" className={cn("flex flex-wrap items-center gap-1 text-xs", className)}>
      {stages.map((stage, i) => {
        const href = links[stage.id] ?? `/runtime#${stage.id}`;
        const isModified = modified.includes(stage.id);
        const isActive = active === stage.id;
        return (
          <span key={stage.id} className="flex items-center gap-1">
            {i > 0 && <span className="text-muted-foreground/50">→</span>}
            <Link
              href={href}
              className={cn(
                "relative rounded-full border px-2.5 py-0.5 font-mono transition-colors hover:bg-accent",
                isActive ? "border-primary bg-primary/10 text-primary" : "text-muted-foreground",
              )}
            >
              {stage.label}
              {isModified && (
                <span
                  className="absolute -top-0.5 -right-0.5 size-2 rounded-full bg-primary"
                  title="This model specializes this stage"
                />
              )}
            </Link>
          </span>
        );
      })}
    </nav>
  );
}
