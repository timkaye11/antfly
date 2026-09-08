import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  type EnvFlagGate,
  type KernelInventoryEntry,
  type KernelRoute,
  ModelSpec,
  SCHEMA_VERSION,
  type SourceLink,
  type Stage,
} from "../lib/schema/index.ts";
import { appRoot, verifyAnchor } from "./lib.ts";

const CURATED_DIR = join(appRoot, "data", "curated");

interface MergeContext {
  opKinds: Set<string>;
  kernels: Set<string>;
  routes: Set<string>;
  flags: Set<string>;
  gitCommit: string;
  generatedAt: string;
  warnings: string[];
}

export function buildMergeContext(
  opKinds: { name: string }[],
  inventory: KernelInventoryEntry[],
  routes: KernelRoute[],
  flags: EnvFlagGate[],
  gitCommit: string,
  generatedAt: string,
): MergeContext {
  return {
    opKinds: new Set(opKinds.map((o) => o.name)),
    kernels: new Set(inventory.map((k) => k.name)),
    routes: new Set(routes.map((r) => r.id)),
    flags: new Set(flags.map((f) => f.name)),
    gitCommit,
    generatedAt,
    warnings: [],
  };
}

function healLink(link: SourceLink | undefined, ctx: MergeContext, where: string): void {
  if (!link?.anchor) return;
  try {
    const result = verifyAnchor(link.path, link.line, link.anchor);
    if (result.healed) {
      ctx.warnings.push(`healed anchor for ${where}: ${link.path}:${link.line} -> :${result.line}`);
      link.line = result.line;
    }
  } catch (err) {
    throw new Error(`${where}: ${(err as Error).message}`);
  }
}

/**
 * A curated model file is a ModelSpec minus `schemaVersion`/`sources`, or a
 * family file with `variants` (one topology emitted per variant with
 * stat/repeat overrides).
 */
export function mergeCuratedModels(ctx: MergeContext): ModelSpec[] {
  if (!existsSync(CURATED_DIR)) return [];
  const specs: ModelSpec[] = [];
  for (const file of readdirSync(CURATED_DIR).sort()) {
    if (!file.endsWith(".model.json")) continue;
    const raw = JSON.parse(readFileSync(join(CURATED_DIR, file), "utf8"));
    const variants: Array<Record<string, unknown>> = raw.variants ?? [
      { id: raw.id, displayName: raw.displayName, tagline: raw.tagline, stats: raw.stats },
    ];
    for (const variant of variants) {
      const candidate = structuredClone({
        schemaVersion: SCHEMA_VERSION,
        id: variant.id,
        family: raw.family,
        displayName: variant.displayName ?? raw.displayName,
        tagline: variant.tagline ?? raw.tagline,
        stats: { ...(raw.stats ?? {}), ...((variant.stats as object) ?? {}) },
        stages: raw.stages,
        graphs: raw.graphs,
        sankey: raw.sankey,
        sources: { gitCommit: ctx.gitCommit, generatedAt: ctx.generatedAt },
      });
      const repeatOverrides = (variant.repeatOverrides ?? {}) as Record<string, number>;
      for (const stage of candidate.stages as Stage[]) {
        const count = repeatOverrides[stage.id];
        if (count !== undefined && stage.repeat) stage.repeat.count = count;
      }
      const spec = ModelSpec.parse(candidate);
      validateSpec(spec, ctx, file);
      specs.push(spec);
    }
  }
  return specs;
}

function validateSpec(spec: ModelSpec, ctx: MergeContext, file: string): void {
  const stageIds = new Set(spec.stages.map((s) => s.id));
  for (const stage of spec.stages) healLink(stage.source, ctx, `${file}/${stage.id}`);

  for (const [phase, graph] of Object.entries(spec.graphs)) {
    if (!graph) continue;
    const nodeIds = new Set(graph.nodes.map((n) => n.id));
    for (const node of graph.nodes) {
      const where = `${file}/${phase}/${node.id}`;
      if (!ctx.opKinds.has(node.opKind)) {
        throw new Error(`${where}: unknown opKind "${node.opKind}"`);
      }
      if (!stageIds.has(node.stageId)) {
        throw new Error(`${where}: unknown stageId "${node.stageId}"`);
      }
      for (const k of node.kernels) {
        if (!ctx.kernels.has(k)) throw new Error(`${where}: unknown kernel "${k}"`);
      }
      for (const r of node.kernelRouteIds) {
        if (!ctx.routes.has(r)) throw new Error(`${where}: unknown kernel route "${r}"`);
      }
      for (const f of node.envFlagNames) {
        if (!ctx.flags.has(f)) ctx.warnings.push(`${where}: env flag not found in scan: "${f}"`);
      }
      healLink(node.source, ctx, where);
      healLink(node.lowererSource, ctx, `${where} (lowerer)`);
    }
    for (const edge of graph.edges) {
      if (!nodeIds.has(edge.from) || !nodeIds.has(edge.to)) {
        throw new Error(`${file}/${phase}/${edge.id}: edge endpoint missing (${edge.from} -> ${edge.to})`);
      }
    }
  }
}

/**
 * data/curated/links.json: a registry of named SourceLinks used by TSX
 * content ({ "id": { path, anchor, line? } }). Anchors are verified and
 * line numbers healed, so prose links never rot.
 */
export function mergeNamedLinks(ctx: MergeContext): Record<string, SourceLink> {
  const file = join(CURATED_DIR, "links.json");
  if (!existsSync(file)) return {};
  const raw = JSON.parse(readFileSync(file, "utf8")) as Record<string, SourceLink>;
  for (const [id, link] of Object.entries(raw)) {
    healLink(link, ctx, `links.json/${id}`);
  }
  return raw;
}

/** Collect every SourceLink in the emitted data, for the snippet cache. */
export function collectLinks(values: unknown[]): SourceLink[] {
  const out: SourceLink[] = [];
  const visit = (v: unknown): void => {
    if (Array.isArray(v)) {
      for (const item of v) visit(item);
    } else if (v && typeof v === "object") {
      const obj = v as Record<string, unknown>;
      if (typeof obj.path === "string" && (obj.path as string).startsWith("zig/")) {
        out.push(obj as unknown as SourceLink);
      } else {
        for (const value of Object.values(obj)) visit(value);
      }
    }
  };
  visit(values);
  return out;
}
