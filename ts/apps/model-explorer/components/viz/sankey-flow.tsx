"use client";

import { cn } from "@antfly/design-system";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";
import { useId, useMemo, useState } from "react";
import type { SankeySpec } from "@/lib/schema";

interface SankeyNodeDatum {
  id: string;
  label: string;
  colorVar?: string;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
}
interface SankeyLinkDatum {
  source: SankeyNodeDatum | string | number;
  target: SankeyNodeDatum | string | number;
  value: number;
  label?: string;
  width?: number;
}

const WIDTH = 720;
const HEIGHT = 360;

/**
 * Forward-pass flow: d3-sankey computes the layout, React owns the SVG.
 * Link width ∝ value (typically bytes/token share).
 */
export function SankeyFlow({
  spec,
  highlight,
  className,
}: {
  spec: SankeySpec;
  /** Node ids to emphasize; everything else dims. */
  highlight?: string[];
  className?: string;
}) {
  const descriptionId = useId();
  const [hovered, setHovered] = useState<string | null>(null);

  const { nodes, links } = useMemo(() => {
    const layout = sankey<SankeyNodeDatum, SankeyLinkDatum>()
      .nodeId((d) => d.id)
      .nodeWidth(14)
      .nodePadding(18)
      .extent([
        [8, 8],
        [WIDTH - 8, HEIGHT - 8],
      ]);
    return layout({
      nodes: spec.nodes.map((n) => ({ ...n })),
      links: spec.links.map((l) => ({ source: l.source, target: l.target, value: l.value, label: l.label })),
    });
  }, [spec]);

  const linkPath = sankeyLinkHorizontal();
  const emphasized = (id: string) =>
    (!highlight || highlight.includes(id)) && (hovered === null || hovered === id);

  return (
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={cn("w-full", className)} role="group" aria-label="Forward-pass flow" aria-describedby={descriptionId}>
      <title>Forward-pass flow</title>
      <desc id={descriptionId}>
        Focus a stage to highlight its connections. {spec.links.map((link) => `${spec.nodes.find((n) => n.id === link.source)?.label ?? link.source} to ${spec.nodes.find((n) => n.id === link.target)?.label ?? link.target}${link.label ? `: ${link.label}` : ""}`).join(". ")}
      </desc>
      {links.map((link) => {
        const source = link.source as SankeyNodeDatum;
        const target = link.target as SankeyNodeDatum;
        const active =
          hovered === null || hovered === source.id || hovered === target.id;
        return (
          <path
            key={`${source.id}:${target.id}:${link.label ?? ""}:${link.value}`}
            d={linkPath(link) ?? undefined}
            fill="none"
            stroke={source.colorVar ?? "var(--muted-foreground)"}
            strokeWidth={Math.max(1.5, link.width ?? 1)}
            strokeOpacity={active ? 0.35 : 0.08}
            className="transition-[stroke-opacity]"
          >
            <title>{`${source.label} → ${target.label}${link.label ? ` · ${link.label}` : ""}`}</title>
          </path>
        );
      })}
      {nodes.map((node) => {
        const active = emphasized(node.id);
        return (
          <g
            key={node.id}
            role="button"
            tabIndex={0}
            aria-label={`Highlight connections for ${node.label}`}
            onMouseEnter={() => setHovered(node.id)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(node.id)}
            onBlur={() => setHovered(null)}
            onClick={() => setHovered(node.id)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setHovered(node.id);
              }
            }}
            className="cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary"
          >
            <rect
              x={node.x0}
              y={node.y0}
              width={(node.x1 ?? 0) - (node.x0 ?? 0)}
              height={(node.y1 ?? 0) - (node.y0 ?? 0)}
              rx={3}
              fill={node.colorVar ?? "var(--muted-foreground)"}
              fillOpacity={active ? 0.85 : 0.3}
            />
            <text
              x={(node.x0 ?? 0) < WIDTH / 2 ? (node.x1 ?? 0) + 6 : (node.x0 ?? 0) - 6}
              y={((node.y0 ?? 0) + (node.y1 ?? 0)) / 2}
              dominantBaseline="central"
              textAnchor={(node.x0 ?? 0) < WIDTH / 2 ? "start" : "end"}
              fontSize={11}
              className="fill-foreground font-mono"
            >
              {node.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
