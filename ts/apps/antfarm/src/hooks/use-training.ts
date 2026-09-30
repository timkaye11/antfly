import { useCallback, useEffect, useRef, useState } from "react";
import { useApiConfig } from "@/hooks/use-api-config";

/** Endpoint-scoped polling. Old endpoint responses and mutations cannot update a new view. */
export function useTrainingResource<T>(load: (signal: AbortSignal) => Promise<T>, interval = 5000) {
  const { apiUrl, client } = useApiConfig();
  const [state, setState] = useState<{ data?: T; error?: string; loading: boolean }>({
    loading: true,
  });
  const [revision, setRevision] = useState(0);
  const generation = useRef(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  // biome-ignore lint/correctness/useExhaustiveDependencies: Endpoint identity and explicit refresh are cancellation boundaries, even with a stable load callback.
  useEffect(() => {
    const current = ++generation.current;
    const controller = new AbortController();
    let pending = false;
    setState({ loading: true });
    const poll = async () => {
      if (pending || document.hidden) return;
      pending = true;
      try {
        const data = await load(controller.signal);
        if (!controller.signal.aborted && current === generation.current)
          setState({ data, loading: false });
      } catch (error) {
        if (!controller.signal.aborted && current === generation.current)
          setState((previous) => ({ ...previous, error: String(error), loading: false }));
      } finally {
        pending = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), interval);
    document.addEventListener("visibilitychange", poll);
    return () => {
      ++generation.current;
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", poll);
    };
  }, [apiUrl, client, load, interval, revision]);
  return { ...state, refresh };
}
