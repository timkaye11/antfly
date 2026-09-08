import envFlagsJson from "@/data/generated/env-flags.json";
import kernelsJson from "@/data/generated/kernels.json";
import linksJson from "@/data/generated/links.json";
import manifestJson from "@/data/generated/manifest.json";
import opKindsJson from "@/data/generated/op-kinds.json";
import snippetsJson from "@/data/generated/snippets.json";
import { EnvFlagsFile, KernelsFile, Manifest, type SourceLink } from "@/lib/schema";

export const manifest = Manifest.parse(manifestJson);
export const kernels = KernelsFile.parse(kernelsJson);
export const envFlags = EnvFlagsFile.parse(envFlagsJson);

export const opKinds = (opKindsJson as { opKinds: Array<{ name: string; group: string; source: SourceLink }> })
  .opKinds;

export interface Snippet {
  path: string;
  line: number;
  startLine: number;
  html: string;
  lang: string;
}

const snippets = (snippetsJson as { snippets: Record<string, Snippet> }).snippets;

export const namedLinks = (linksJson as { links: Record<string, SourceLink> }).links;

/** Look up a drift-checked named link from data/curated/links.json. */
export function L(id: string): SourceLink {
  const link = namedLinks[id];
  if (!link) throw new Error(`unknown named link "${id}" — add it to data/curated/links.json`);
  return link;
}

export function getSnippet(link: SourceLink): Snippet | undefined {
  if (link.line === undefined) return undefined;
  return snippets[`${link.path}:${link.line}`];
}

/** Collect every SourceLink nested anywhere in `values` (for SnippetProvider). */
export function collectSourceLinks(values: unknown): SourceLink[] {
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

/** Subset of the snippet cache covering `links` — pass to SnippetProvider. */
export function snippetsFor(links: SourceLink[]): Record<string, Snippet> {
  const out: Record<string, Snippet> = {};
  for (const link of links) {
    if (link.line === undefined) continue;
    const key = `${link.path}:${link.line}`;
    const snippet = snippets[key];
    if (snippet) out[key] = snippet;
  }
  return out;
}

export function permalinkFor(link: SourceLink): string | undefined {
  if (!manifest.permalinkBase) return undefined;
  const anchor = link.line !== undefined ? `#L${link.line}` : "";
  return `${manifest.permalinkBase}/${manifest.gitCommit}/${link.path}${anchor}`;
}
