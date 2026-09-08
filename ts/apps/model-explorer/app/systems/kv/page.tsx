import { L, manifest, snippetsFor } from "@/lib/data";
import { KvClient } from "./kv-client";

const LINK_IDS = ["kv-manager", "kv-sliding-window", "kv-turboquant-polar4", "config-shared-kv"];

export default function KvPage() {
  return (
    <KvClient
      snippets={snippetsFor(LINK_IDS.map(L))}
      gitCommit={manifest.gitCommit}
      permalinkBase={manifest.permalinkBase}
    />
  );
}
