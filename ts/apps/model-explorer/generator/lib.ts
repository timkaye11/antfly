import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Walk up from this file until we find the antfly repo root (.git). */
export function findRepoRoot(): string {
  let dir = dirname(fileURLToPath(import.meta.url));
  while (dir !== "/") {
    if (existsSync(join(dir, ".git"))) return dir;
    dir = resolve(dir, "..");
  }
  throw new Error("could not locate repo root (.git) above generator/");
}

export const repoRoot = findRepoRoot();
export const appRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

export function readRepoFile(relPath: string): string {
  return readFileSync(join(repoRoot, relPath), "utf8");
}

export function gitCommit(): string {
  return execSync("git rev-parse HEAD", { cwd: repoRoot }).toString().trim();
}

/** 1-indexed line lookup. */
export function lineOf(content: string, index: number): number {
  let line = 1;
  for (let i = 0; i < index; i++) if (content.charCodeAt(i) === 10) line++;
  return line;
}

export interface AnchorResult {
  line: number;
  healed: boolean;
}

/**
 * Verify that `anchor` appears at `line` in the file; if not, search for a
 * unique occurrence and heal the line number. Throws if the anchor is gone
 * or ambiguous.
 */
export function verifyAnchor(relPath: string, line: number | undefined, anchor: string): AnchorResult {
  const content = readRepoFile(relPath);
  const lines = content.split("\n");
  if (line !== undefined && lines[line - 1]?.includes(anchor)) {
    return { line, healed: false };
  }
  const hits: number[] = [];
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].includes(anchor)) hits.push(i + 1);
  }
  if (hits.length === 1) return { line: hits[0], healed: true };
  if (hits.length === 0) {
    throw new Error(`anchor not found in ${relPath}: "${anchor}"`);
  }
  throw new Error(`anchor ambiguous (${hits.length} hits) in ${relPath}: "${anchor}"`);
}

/** Stable stringify with 1-space indent to keep generated JSON diffs readable. */
export function toJson(value: unknown): string {
  return `${JSON.stringify(value, null, 1)}\n`;
}
