/** Typed node-local training API, with optional two-Mac execution. */
import type { components } from "./public-api.js";

export type TrainingPeer = components["schemas"]["TrainingPeer"];
export type TrainingPeerRegistration = components["schemas"]["TrainingPeerRegistration"];
export type TrainingPeersResponse = components["schemas"]["TrainingPeersResponse"];
export type TrainingJobSpec = components["schemas"]["TrainingJobSpec"];
export type TrainingJob = components["schemas"]["TrainingJob"];
export type TrainingJobsResponse = components["schemas"]["TrainingJobsResponse"];
export type TrainingLogsResponse = components["schemas"]["TrainingLogsResponse"];
export type TrainingDatasetSpec = components["schemas"]["TrainingDatasetSpec"];
export type TrainingDataset = components["schemas"]["TrainingDataset"];
export type TrainingDatasetsResponse = components["schemas"]["TrainingDatasetsResponse"];
export type TrainingHuggingFaceRequest = components["schemas"]["TrainingHuggingFaceRequest"];
export type TrainingHuggingFaceResponse = components["schemas"]["TrainingHuggingFaceResponse"];

type Transport = <T>(
  method: string,
  path: string,
  body?: unknown,
  signal?: AbortSignal
) => Promise<T>;

export function createTrainingClient(request: Transport) {
  const base = "/db/v1/training";
  const id = encodeURIComponent;
  return {
    datasets: (signal?: AbortSignal) =>
      request<TrainingDatasetsResponse>("GET", `${base}/datasets`, undefined, signal),
    createDataset: (spec: TrainingDatasetSpec, signal?: AbortSignal) =>
      request<TrainingDataset>("POST", `${base}/datasets`, spec, signal),
    dataset: (datasetId: string, signal?: AbortSignal) =>
      request<TrainingDataset>("GET", `${base}/datasets/${id(datasetId)}`, undefined, signal),
    uploadDatasetChunk: (datasetId: string, offset: number, data: string, signal?: AbortSignal) =>
      request<TrainingDataset>(
        "PUT",
        `${base}/datasets/${id(datasetId)}/chunks`,
        { offset, data },
        signal
      ),
    prepareDataset: (datasetId: string, signal?: AbortSignal) =>
      request<TrainingDataset>(
        "POST",
        `${base}/datasets/${id(datasetId)}/prepare`,
        undefined,
        signal
      ),
    cancelDataset: (datasetId: string, signal?: AbortSignal) =>
      request<TrainingDataset>(
        "POST",
        `${base}/datasets/${id(datasetId)}/cancel`,
        undefined,
        signal
      ),
    removeDataset: (datasetId: string, signal?: AbortSignal) =>
      request<{ removed: boolean }>(
        "DELETE",
        `${base}/datasets/${id(datasetId)}`,
        undefined,
        signal
      ),
    huggingFace: (source: TrainingHuggingFaceRequest, signal?: AbortSignal) =>
      request<TrainingHuggingFaceResponse>("POST", `${base}/datasets/huggingface`, source, signal),
    peers: (signal?: AbortSignal) =>
      request<TrainingPeersResponse>("GET", `${base}/peers`, undefined, signal),
    registerPeer: (peer: TrainingPeerRegistration, signal?: AbortSignal) =>
      request<TrainingPeer>("POST", `${base}/peers`, peer, signal),
    removePeer: (peerId: string, signal?: AbortSignal) =>
      request<{ removed: boolean }>("DELETE", `${base}/peers/${id(peerId)}`, undefined, signal),
    refreshPeer: (peerId: string, signal?: AbortSignal) =>
      request<TrainingPeer>("POST", `${base}/peers/${id(peerId)}/refresh`, undefined, signal),
    preflight: (spec: TrainingJobSpec, signal?: AbortSignal) =>
      request<TrainingJob>("POST", `${base}/preflights`, spec, signal),
    start: (spec: TrainingJobSpec, signal?: AbortSignal) =>
      request<TrainingJob>("POST", `${base}/jobs`, spec, signal),
    jobs: (signal?: AbortSignal) =>
      request<TrainingJobsResponse>("GET", `${base}/jobs`, undefined, signal),
    job: (jobId: string, signal?: AbortSignal) =>
      request<TrainingJob>("GET", `${base}/jobs/${id(jobId)}`, undefined, signal),
    logs: (jobId: string, rank: "launcher" | "0" | "1", cursor = 0, signal?: AbortSignal) =>
      request<TrainingLogsResponse>(
        "GET",
        `${base}/jobs/${id(jobId)}/logs?rank=${rank}&cursor=${cursor}`,
        undefined,
        signal
      ),
    cancel: (jobId: string, signal?: AbortSignal) =>
      request<TrainingJob>("POST", `${base}/jobs/${id(jobId)}/cancel`, undefined, signal),
    pause: (jobId: string, signal?: AbortSignal) =>
      request<TrainingJob>("POST", `${base}/jobs/${id(jobId)}/pause`, undefined, signal),
    resume: (jobId: string, requestId: string, signal?: AbortSignal) =>
      request<TrainingJob>(
        "POST",
        `${base}/jobs/${id(jobId)}/resume`,
        { request_id: requestId },
        signal
      ),
  };
}
