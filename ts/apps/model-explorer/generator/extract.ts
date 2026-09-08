import { readdirSync } from "node:fs";
import { join } from "node:path";
import type { EnvFlagGate, KernelInventoryEntry, KernelRoute, SourceLink } from "../lib/schema/index.ts";
import { lineOf, readRepoFile, repoRoot } from "./lib.ts";

const NODE_ZIG = "zig/lib/ml/src/graph/node.zig";
const COMPILER_ZIG = "zig/pkg/inference/src/graph/quant_kernel_compiler.zig";
const KERNELS_M = "zig/pkg/inference/src/backends/metal_kernels.m";
const GENERATED_DIR = "zig/pkg/inference/src/ops/metal/generated";
const INFERENCE_SRC = "zig/pkg/inference/src";

export interface OpKindEntry {
  name: string;
  group: "primitive" | "fused";
  source: SourceLink;
}

/** Parse PrimitiveOp + FusedOp enum members from node.zig. */
export function extractOpKinds(): OpKindEntry[] {
  const content = readRepoFile(NODE_ZIG);
  const out: OpKindEntry[] = [];
  for (const [enumName, group] of [
    ["PrimitiveOp", "primitive"],
    ["FusedOp", "fused"],
  ] as const) {
    const start = content.indexOf(`pub const ${enumName} = enum`);
    if (start < 0) throw new Error(`enum ${enumName} not found in ${NODE_ZIG}`);
    const bodyStart = content.indexOf("{", start);
    const bodyEnd = content.indexOf("\n};", bodyStart);
    const body = content.slice(bodyStart, bodyEnd);
    const memberRe = /^\s{4}([a-z][a-z0-9_]*),\s*$/gm;
    let m: RegExpExecArray | null;
    while ((m = memberRe.exec(body)) !== null) {
      out.push({
        name: m[1],
        group,
        source: { path: NODE_ZIG, line: lineOf(content, bodyStart + m.index), anchor: `${m[1]},` },
      });
    }
  }
  if (out.length < 50) throw new Error(`suspiciously few op kinds parsed: ${out.length}`);
  return out;
}

/** Parse metal_production_schedules struct-literal rows. */
export function extractSchedules(): KernelRoute[] {
  const content = readRepoFile(COMPILER_ZIG);
  const tableStart = content.indexOf("pub const metal_production_schedules");
  if (tableStart < 0) throw new Error(`metal_production_schedules not found in ${COMPILER_ZIG}`);
  const tableEnd = content.indexOf("\n};", tableStart);
  const table = content.slice(tableStart, tableEnd);
  const rowRe =
    /\.\{ \.format = \.(\w+), \.row_bucket = \.(\w+), \.epilogue = \.(\w+), \.schedule = \.\{ ([^}]*) \} \}/g;
  const routes: KernelRoute[] = [];
  let m: RegExpExecArray | null;
  while ((m = rowRe.exec(table)) !== null) {
    const [, format, rowBucket, epilogue, schedBody] = m;
    const sched: Record<string, number | string> = {};
    for (const fieldMatch of schedBody.matchAll(/\.(\w+) = \.?([\w]+)/g)) {
      const [, key, raw] = fieldMatch;
      sched[key] = /^\d+$/.test(raw) ? Number(raw) : raw;
    }
    routes.push({
      id: `${format}/${rowBucket}/${epilogue}`,
      format,
      rowBucket,
      epilogue,
      schedule: {
        threadsPerThreadgroup: sched.threads_per_threadgroup as number | undefined,
        colsPerThreadgroup: sched.cols_per_threadgroup as number | undefined,
        rowsPerThreadgroup: sched.rows_per_threadgroup as number | undefined,
        reduction: sched.reduction as string | undefined,
        keyChunk: sched.key_chunk as number | undefined,
        skipRescale: sched.skip_rescale === "true" ? true : undefined,
      },
      generated: true,
      generatedFile: guessGeneratedFile(format, epilogue),
      source: {
        path: COMPILER_ZIG,
        line: lineOf(content, tableStart + m.index),
        anchor: `.format = .${format}, .row_bucket = .${rowBucket}, .epilogue = .${epilogue}`,
      },
    });
  }
  if (routes.length < 10) throw new Error(`suspiciously few schedule rows parsed: ${routes.length}`);
  return routes;
}

function guessGeneratedFile(format: string, epilogue: string): string | undefined {
  const suffix = epilogue === "none" ? "" : `_${epilogue}`;
  const candidate = `quant_kernel_${format}_small_batch${suffix}.metal`;
  try {
    const files = readdirSync(join(repoRoot, GENERATED_DIR));
    return files.includes(candidate) ? `${GENERATED_DIR}/${candidate}` : undefined;
  } catch {
    return undefined;
  }
}

const FAMILY_RULES: Array<[RegExp, KernelInventoryEntry["family"]]> = [
  [/(lora|adamw|training|_bwd|backward|bce|grad)/, "training"],
  [/moe/, "moe"],
  [/(gliner|deberta|disentangled)/, "gliner"],
  [/(florence|vision|conv2d|conv1d|window_pack|window_unpack|channel_scores)/, "vision"],
  [/(attention|sdpa|paged)/, "attention"],
  [/_mm(_|$)|_mm_sg/, "mm_sg"],
  [/(pair_activation|pair_linear|head_rms_rope|rms_norm_add|post_residual|pair_)/, "fusion"],
  [/(sample|argmax|top8|topk|lm_head)/, "sampling"],
  [/(kv|polar4|turbo3|compressed_attention|_key$|_key_)/, "kv"],
  [/(rms|norm|rope)/, "norm_rope"],
  [/(cpy|get_rows|set_rows|transpose|concat|slice|broadcast|convert|embed|dtype)/, "data_movement"],
  [/linear/, "matvec"],
];

function classifyKernel(name: string): KernelInventoryEntry["family"] {
  for (const [re, family] of FAMILY_RULES) if (re.test(name)) return family;
  return "other";
}

/** Scan metal_kernels.m + generated/*.metal for `kernel void <name>`. */
export function extractKernelInventory(): KernelInventoryEntry[] {
  const seen = new Map<string, KernelInventoryEntry>();

  const mContent = readRepoFile(KERNELS_M);
  const beginMarker = mContent.indexOf("quant-kernel-codegen:begin generated quant kernels");
  const endMarker = mContent.indexOf("quant-kernel-codegen:end generated quant kernels");
  const kernelRe = /kernel void ([a-zA-Z0-9_]+)/g;
  let m: RegExpExecArray | null;
  while ((m = kernelRe.exec(mContent)) !== null) {
    const name = m[1];
    if (seen.has(name)) continue;
    const inGenerated = beginMarker >= 0 && m.index > beginMarker && m.index < endMarker;
    seen.set(name, {
      name,
      family: classifyKernel(name),
      generated: inGenerated,
      source: { path: KERNELS_M, line: lineOf(mContent, m.index), anchor: `kernel void ${name}` },
    });
  }

  for (const file of readdirSync(join(repoRoot, GENERATED_DIR))) {
    if (!file.endsWith(".metal")) continue;
    const rel = `${GENERATED_DIR}/${file}`;
    const content = readRepoFile(rel);
    let g: RegExpExecArray | null;
    const re = /kernel void ([a-zA-Z0-9_]+)/g;
    while ((g = re.exec(content)) !== null) {
      const name = g[1];
      const entry: KernelInventoryEntry = {
        name,
        family: classifyKernel(name),
        generated: true,
        source: { path: rel, line: lineOf(content, g.index), anchor: `kernel void ${name}` },
      };
      // Prefer the generated-file location for generated kernels.
      seen.set(name, entry);
    }
  }

  const inventory = [...seen.values()].sort((a, b) => a.name.localeCompare(b.name));
  if (inventory.length < 300) throw new Error(`suspiciously few kernels found: ${inventory.length}`);
  return inventory;
}

function inferFlagKind(name: string): EnvFlagGate["kind"] {
  if (/_MB$/.test(name)) return "mb";
  if (/(_BYTES|_CAPACITY)$/.test(name)) return "bytes";
  if (/(_COUNT|_THREADS|_CHUNK|_MIN_KV|_SIZE|_LIMIT|_MAX|_ROWS|_COLS)$/.test(name)) return "int";
  if (/(DISABLE|ENABLE|FORCE|STRICT|TRACE|DEBUG)/.test(name)) return "bool";
  return "unknown";
}

/** Scan inference src for TERMITE_METAL_* env flags with occurrence locations. */
export function extractEnvFlags(): EnvFlagGate[] {
  const byName = new Map<string, { links: SourceLink[]; count: number }>();
  const walk = (relDir: string) => {
    for (const entry of readdirSync(join(repoRoot, relDir), { withFileTypes: true })) {
      const rel = `${relDir}/${entry.name}`;
      if (entry.isDirectory()) {
        if (entry.name === "generated") continue;
        walk(rel);
      } else if (entry.name.endsWith(".zig") || entry.name.endsWith(".m")) {
        const content = readRepoFile(rel);
        const re = /(?:TERMITE|ANTFLY)_[A-Z0-9][A-Z0-9_]{3,}/g;
        let m: RegExpExecArray | null;
        while ((m = re.exec(content)) !== null) {
          const name = m[0];
          const rec = byName.get(name) ?? { links: [], count: 0 };
          rec.count++;
          if (rec.links.length < 5) {
            rec.links.push({ path: rel, line: lineOf(content, m.index), anchor: name });
          }
          byName.set(name, rec);
        }
      }
    }
  };
  walk(INFERENCE_SRC);
  const flags = [...byName.entries()]
    .map(([name, rec]) => ({
      name,
      kind: inferFlagKind(name),
      occurrences: rec.count,
      sources: rec.links,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
  if (flags.length < 200) throw new Error(`suspiciously few env flags found: ${flags.length}`);
  return flags;
}
