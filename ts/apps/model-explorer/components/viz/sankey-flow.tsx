"use client";

import { cn } from "@antfly/design-system";
import { sankey, sankeyLinkHorizontal } from "d3-sankey";
import { useMemo, useState } from "react";
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
    <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className={cn("w-full", className)} role="img" aria-label="Forward-pass flow">
      {links.map((link, i) => {
        const source = link.source as SankeyNodeDatum;
        const target = link.target as SankeyNodeDatum;
        const active =
          hovered === null || hovered === source.id || hovered === target.id;
        return (
          <path
            key={i}
            d={linkPath(link) ?? undefined}
            fill="none"
            stroke={source.colorVar ?? "var(--muted-foreground)"}
            strokeWidth={Math.max(1.5, link.width ?? 1)}
            strokeOpacity={active ? 0.35 : 0.08}
            className="transition-[stroke-opacity]"
          >
            <title>
              {source.label} → {target.label}
              {link.label ? ` · ${link.label}` : ""}
            </title>
          </path>
        );
      })}
      {nodes.map((node) => {
        const active = emphasized(node.id);
        return (
          <g
            key={node.id}
            onMouseEnter={() => setHovered(node.id)}
            onMouseLeave={() => setHovered(null)}
            className="cursor-default"
            opacity={active ? 1 : 0.35}
          >
            <rect
              x={node.x0}
              y={node.y0}
              width={(node.x1 ?? 0) - (node.x0 ?? 0)}
              height={(node.y1 ?? 0) - (node.y0 ?? 0)}
              rx={3}
              fill={node.colorVar ?? "var(--muted-foreground)"}
              fillOpacity={0.85}
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
