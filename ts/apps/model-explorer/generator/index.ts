/**
 * Model-explorer data generator.
 *
 * Extracts ground truth from the Zig runtime (op vocabulary, kernel
 * inventory, production schedules, env flags), merges hand-curated model
 * graphs from data/curated/, verifies + self-heals file:line anchors, and
 * emits validated JSON into data/generated/.
 *
 *   pnpm gen         — regenerate in place
 *   pnpm gen:check   — regenerate in memory and diff (exit 1 on drift)
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { SCHEMA_VERSION } from "../lib/schema/index.ts";
import { extractEnvFlags, extractKernelInventory, extractOpKinds, extractSchedules } from "./extract.ts";
import { appRoot, gitCommit, toJson } from "./lib.ts";
import { buildMergeContext, collectLinks, mergeCuratedModels, mergeNamedLinks } from "./merge.ts";
import { buildSnippets } from "./snippets.ts";

const OUT_DIR = join(appRoot, "data", "generated");
const VOLATILE_KEYS = new Set(["generatedAt", "gitCommit"]);
const SIZE_WARN_BYTES = 800 * 1024;

function stripVolatile(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripVolatile);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) {
      if (!VOLATILE_KEYS.has(k)) out[k] = stripVolatile(v);
    }
    return out;
  }
  return value;
}

async function generate(): Promise<Map<string, string>> {
  const commit = gitCommit();
  const generatedAt = new Date().toISOString();

  const opKinds = extractOpKinds();
  const routes = extractSchedules();
  const inventory = extractKernelInventory();
  const flags = extractEnvFlags();

  const ctx = buildMergeContext(opKinds, inventory, routes, flags, commit, generatedAt);
  const models = mergeCuratedModels(ctx);
  const namedLinks = mergeNamedLinks(ctx);
  for (const warning of ctx.warnings) console.warn(`  warn: ${warning}`);

  const files = new Map<string, string>();
  files.set("op-kinds.json", toJson({ schemaVersion: SCHEMA_VERSION, opKinds }));
  files.set("kernels.json", toJson({ schemaVersion: SCHEMA_VERSION, routes, inventory }));
  files.set("env-flags.json", toJson({ schemaVersion: SCHEMA_VERSION, flags }));
  files.set("links.json", toJson({ schemaVersion: SCHEMA_VERSION, links: namedLinks }));
  for (const spec of models) {
    files.set(`models/${spec.id}.json`, toJson(spec));
  }

  // Pass curated frame scenarios through (validated lazily by the app).
  const framesDir = join(appRoot, "data", "curated", "frames");
  if (existsSync(framesDir)) {
    for (const file of readdirSync(framesDir).sort()) {
      if (file.endsWith(".json")) {
        files.set(`frames/${file}`, readFileSync(join(framesDir, file), "utf8"));
      }
    }
  }

  // Snippets only for links the UI shows hover peeks for (model graphs +
  // schedule rows) — the full kernel/flag inventories link out without peeks.
  const links = collectLinks([routes, models, Object.values(namedLinks)]);
  const snippets = await buildSnippets(links);
  files.set("snippets.json", toJson({ schemaVersion: SCHEMA_VERSION, snippets }));

  files.set(
    "manifest.json",
    toJson({
      schemaVersion: SCHEMA_VERSION,
      gitCommit: commit,
      generatedAt,
      permalinkBase: "https://github.com/antflydb/antfly/blob",
      models: models.map((m) => m.id),
      counts: {
        opKinds: opKinds.length,
        kernels: inventory.length,
        routes: routes.length,
        envFlags: flags.length,
        snippets: Object.keys(snippets).length,
      },
    }),
  );

  for (const [name, content] of files) {
    if (Buffer.byteLength(content) > SIZE_WARN_BYTES) {
      console.warn(`  warn: ${name} is ${(Buffer.byteLength(content) / 1024).toFixed(0)} KB (budget 800 KB)`);
    }
  }
  return files;
}

function check(files: Map<string, string>): number {
  let drift = 0;
  for (const [name, content] of files) {
    const target = join(OUT_DIR, name);
    if (!existsSync(target)) {
      console.error(`MISSING  ${name}`);
      drift++;
      continue;
    }
    const existing = readFileSync(target, "utf8");
    const a = JSON.stringify(stripVolatile(JSON.parse(existing)));
    const b = JSON.stringify(stripVolatile(JSON.parse(content)));
    if (a !== b) {
      console.error(`DRIFT    ${name}`);
      drift++;
    }
  }
  // Extra files on disk that would no longer be generated are drift too.
  const walk = (dir: string, prefix: string) => {
    if (!existsSync(dir)) return;
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
      if (entry.isDirectory()) walk(join(dir, entry.name), rel);
      else if (!files.has(rel)) {
        console.error(`STALE    ${rel}`);
        drift++;
      }
    }
  };
  walk(OUT_DIR, "");
  return drift;
}

const checkMode = process.argv.includes("--check");
const files = await generate();

if (checkMode) {
  const drift = check(files);
  if (drift > 0) {
    console.error(`\ngen:check: ${drift} file(s) drifted — run \`pnpm gen\` and review.`);
    process.exit(1);
  }
  console.log(`gen:check: ${files.size} files match.`);
} else {
  rmSync(OUT_DIR, { recursive: true, force: true });
  for (const [name, content] of files) {
    const target = join(OUT_DIR, name);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, content);
  }
  console.log(`generated ${files.size} files into data/generated/`);
}
