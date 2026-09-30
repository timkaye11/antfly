import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useTrainingResource } from "./use-training";

const { endpoint, client } = vi.hoisted(() => ({
  endpoint: { current: "http://server-a" },
  client: {},
}));
vi.mock("@/hooks/use-api-config", () => ({
  useApiConfig: () => ({ apiUrl: endpoint.current, client }),
}));
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  endpoint.current = "http://server-a";
});

describe("training resources", () => {
  it("ignores late A responses after switching A to B and back to A", async () => {
    const requests: { signal: AbortSignal; resolve: (value: string) => void }[] = [];
    const load = (signal: AbortSignal) =>
      new Promise<string>((resolve) => requests.push({ signal, resolve }));
    const view = renderHook(() => useTrainingResource(load));
    expect(requests).toHaveLength(1);
    endpoint.current = "http://server-b";
    view.rerender();
    expect(requests[0].signal.aborted).toBe(true);
    endpoint.current = "http://server-a";
    view.rerender();
    expect(requests[1].signal.aborted).toBe(true);
    await act(async () => requests[2].resolve("new A"));
    await waitFor(() => expect(view.result.current.data).toBe("new A"));
    await act(async () => {
      requests[0].resolve("old A");
      requests[1].resolve("B");
    });
    expect(view.result.current.data).toBe("new A");
    view.unmount();
    expect(requests[2].signal.aborted).toBe(true);
  });

  it("waits while hidden, resumes when visible, and never overlaps requests", async () => {
    let hidden = true;
    vi.spyOn(document, "hidden", "get").mockImplementation(() => hidden);
    let resolve!: (value: string) => void;
    const load = vi.fn(
      () =>
        new Promise<string>((done) => {
          resolve = done;
        })
    );
    const view = renderHook(() => useTrainingResource(load, 10));
    expect(load).not.toHaveBeenCalled();
    hidden = false;
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(load).toHaveBeenCalledTimes(1);
    hidden = true;
    await act(async () => resolve("visible result"));
    await waitFor(() => expect(view.result.current.data).toBe("visible result"));
  });
});
