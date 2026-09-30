import { describe, expect, it, vi } from "vitest";
import { AntflyClient } from "../src/client.js";

describe("training API", () => {
  it("preserves auth, cancellation, cursor and stable job keys", async () => {
    const fetcher = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("{}", { status: 200 }));
    try {
      const client = new AntflyClient({
        baseUrl: "http://localhost:8080",
        auth: { username: "admin", password: "secret" },
      });
      const controller = new AbortController();
      await client.training.logs("job/one", "1", 65536, controller.signal);
      const [url, options] = fetcher.mock.calls[0];
      expect(String(url)).toContain("/db/v1/training/jobs/job%2Fone/logs?rank=1&cursor=65536");
      expect(options?.signal).toBe(controller.signal);
      expect(new Headers(options?.headers).get("authorization")).toBe("Basic YWRtaW46c2VjcmV0");
      fetcher.mockResolvedValue(new Response("{}", { status: 202 }));
      const spec = {
        peer_id: "mini",
        coordinator: "192.0.2.1:32132",
        request_id: "stable-request",
        kind: "transport" as const,
      };
      await client.training.preflight(spec, controller.signal);
      expect(JSON.parse(String(fetcher.mock.calls[1][1]?.body))).toEqual(spec);
      fetcher.mockResolvedValue(new Response("{}", { status: 202 }));
      const local = {
        request_id: "local-request",
        execution_mode: "local" as const,
        family: "gliner25" as const,
        gliner25_config: "/data/job.json",
      };
      await client.training.start(local, controller.signal);
      expect(JSON.parse(String(fetcher.mock.calls[2][1]?.body))).toEqual(local);
      fetcher.mockResolvedValue(new Response("{}", { status: 202 }));
      const form = {
        request_id: "form-request",
        execution_mode: "local" as const,
        family: "gliner25" as const,
        base_model: "/models/gliner25",
        dataset_id: "prepared-dataset",
        gliner25_options: { batch_size: 1, accumulation: 2, memory_total_gib: 12 },
      };
      await client.training.start(form, controller.signal);
      expect(JSON.parse(String(fetcher.mock.calls[3][1]?.body))).toEqual(form);
    } finally {
      fetcher.mockRestore();
    }
  });
});
