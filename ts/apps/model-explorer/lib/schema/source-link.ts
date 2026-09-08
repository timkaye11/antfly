import { z } from "zod";

/**
 * A pointer into the Antfly repo. `path` is repo-root-relative
 * (e.g. "zig/pkg/inference/src/graph/quant_kernel_compiler.zig").
 * `anchor` is an expected substring at `line`, used by the generator for
 * drift detection + self-healing line numbers.
 */
export const SourceLink = z.object({
  path: z.string(),
  line: z.number().int().positive().optional(),
  endLine: z.number().int().positive().optional(),
  anchor: z.string().optional(),
  symbol: z.string().optional(),
});
export type SourceLink = z.infer<typeof SourceLink>;
