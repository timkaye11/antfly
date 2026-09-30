import type { TrainingJob, TrainingJobSpec } from "@antfly/sdk";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TrainingPage from "./TrainingPage";

const { client } = vi.hoisted(() => ({
  client: {
    training: {
      peers: vi.fn(),
      jobs: vi.fn(),
      job: vi.fn(),
      logs: vi.fn(),
      preflight: vi.fn(),
      start: vi.fn(),
    },
  },
}));
vi.mock("@/hooks/use-api-config", () => ({
  useApiConfig: () => ({ apiUrl: "http://local-training", client }),
}));
vi.mock("@/components/training-peers", () => ({
  TrainingPeers: () => <div>Peer setup</div>,
}));
vi.mock("@/components/training-datasets", () => ({
  TrainingDatasets: ({ onSelect }: { onSelect: (id: string) => void }) => (
    <section id="training-datasets" aria-label="Dataset configuration">
      Dataset configuration
      <button type="button" onClick={() => onSelect("prepared-fixture")}>
        Use fixture dataset
      </button>
    </section>
  ),
}));

beforeEach(() => {
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  const jobs: TrainingJob[] = [];
  client.training.peers.mockResolvedValue({
    peers: [{ id: "mini", name: "Mini", status: "connected" }],
    nearby: [],
  });
  client.training.jobs.mockImplementation(async () => ({ jobs }));
  client.training.job.mockImplementation(async (id: string) => jobs.find((job) => job.id === id));
  client.training.logs.mockResolvedValue({ text: "", cursor: 0 });
  client.training.preflight.mockImplementation(async (spec: TrainingJobSpec) => {
    const job = { id: "readiness", kind: "preflight", status: "complete", spec } as TrainingJob;
    jobs.push(job);
    return job;
  });
  client.training.start.mockImplementation(async (spec: TrainingJobSpec) => ({
    id: "run",
    kind: "training",
    status: "queued",
    spec,
  }));
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("training execution selection", () => {
  it("defaults to local training without peer fields and puts optional machines after the dataset", async () => {
    render(<TrainingPage />);
    expect(
      (screen.getByRole("checkbox", { name: /Use another Mac/ }) as HTMLInputElement).checked
    ).toBe(false);
    expect(screen.queryByLabelText("Remote Mac")).toBeNull();
    expect(screen.queryByRole("button", { name: "Test TCP connection" })).toBeNull();
    const dataset = screen.getByRole("region", { name: "Dataset configuration" });
    const execution = screen.getByRole("region", { name: "Execution" });
    expect(
      dataset.compareDocumentPosition(execution) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    const readiness = screen.getByRole("button", { name: "Check training readiness" });
    expect((readiness as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByLabelText(/Job JSON path/)).toBeNull();
    expect(screen.queryByText(/Use job template/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Use fixture dataset" }));
    expect((readiness as HTMLButtonElement).disabled).toBe(false);
    fireEvent.click(readiness);
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(1));
    const spec = client.training.preflight.mock.calls[0][0];
    expect(spec.execution_mode).toBe("local");
    expect(spec.base_model).toBe("/Users/Shared/antfly-training/models/gliner25");
    expect(spec.dataset_id).toBe("prepared-fixture");
    expect(spec.gliner25_options).toMatchObject({
      mode: "lora",
      rank: 8,
      epochs: 1,
      batch_size: 2,
    });
    expect(spec).not.toHaveProperty("gliner25_config");
    expect(spec).not.toHaveProperty("train_file");
    expect(spec).not.toHaveProperty("peer_id");
    expect(spec).not.toHaveProperty("coordinator");
    const start = screen.getByRole("button", { name: "Start training" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(start);
    await waitFor(() => expect(client.training.start).toHaveBeenCalledTimes(1));
    expect(client.training.start.mock.calls[0][0].execution_mode).toBe("local");
    expect(screen.queryByRole("option", { name: "Remote Mac · rank 1" })).toBeNull();
  });

  it("submits form settings and invalidates readiness when training parameters change", async () => {
    render(<TrainingPage />);
    fireEvent.click(screen.getByRole("button", { name: "Use fixture dataset" }));
    fireEvent.change(screen.getByLabelText("Adapter"), { target: { value: "dora" } });
    fireEvent.change(screen.getByLabelText("Epochs"), { target: { value: "3" } });
    fireEvent.change(screen.getByLabelText("Batch size per Mac"), { target: { value: "1" } });
    fireEvent.change(screen.getByLabelText("Learning rate"), { target: { value: "0.0002" } });
    fireEvent.click(screen.getByText("Advanced training settings"));
    fireEvent.change(screen.getByLabelText("Gradient accumulation"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Maximum sequence tokens"), {
      target: { value: "256" },
    });
    fireEvent.change(screen.getByLabelText("Total memory budget (GiB)"), {
      target: { value: "10" },
    });
    fireEvent.change(screen.getByLabelText("Dataset memory budget (MiB)"), {
      target: { value: "32" },
    });
    fireEvent.change(screen.getByLabelText("Model loading overhead (MiB)"), {
      target: { value: "128" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Check training readiness" }));
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(1));
    expect(client.training.preflight.mock.calls[0][0].gliner25_options).toMatchObject({
      mode: "dora",
      epochs: 3,
      batch_size: 1,
      task_lr: 0.0002,
      accumulation: 4,
      max_sequence_tokens: 256,
      memory_total_gib: 10,
      dataset_memory_mib: 32,
      source_auxiliary_mib: 128,
    });
    const start = screen.getByRole("button", { name: "Start training" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.change(screen.getByLabelText("Adapter rank"), { target: { value: "16" } });
    expect((start as HTMLButtonElement).disabled).toBe(true);
  });

  it("keeps an existing job JSON as an optional path without replacing its settings", async () => {
    render(<TrainingPage />);
    fireEvent.change(screen.getByLabelText("Configuration method"), {
      target: { value: "template" },
    });
    expect(screen.queryByLabelText(/Base model directory/)).toBeNull();
    fireEvent.change(screen.getByLabelText(/Job JSON path/), {
      target: { value: "/data/existing.json" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Check training readiness" }));
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(1));
    const request = client.training.preflight.mock.calls[0][0];
    expect(request.gliner25_config).toBe("/data/existing.json");
    expect(request).not.toHaveProperty("base_model");
    expect(request).not.toHaveProperty("train_file");
    expect(request).not.toHaveProperty("gliner25_options");
    fireEvent.click(screen.getByRole("checkbox", { name: /Override the JSON training settings/ }));
    expect(screen.getByLabelText("Compute")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Check training readiness" }));
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(2));
    expect(client.training.preflight.mock.calls[1][0].gliner25_options).toMatchObject({
      mode: "lora",
      rank: 8,
    });
  });

  it("requires connection settings only for two Macs and invalidates readiness when the mode changes", async () => {
    render(<TrainingPage />);
    fireEvent.change(screen.getByLabelText("Model family"), { target: { value: "gemma4" } });
    const examples = screen.getByLabelText("Selected examples") as HTMLInputElement;
    expect(examples.min).toBe("1");
    expect(examples.step).toBe("1");
    fireEvent.change(examples, { target: { value: "3" } });
    const toggle = screen.getByRole("checkbox", { name: /Use another Mac/ });
    fireEvent.click(toggle);
    expect((screen.getByLabelText("Selected examples (even)") as HTMLInputElement).step).toBe("2");
    expect(
      (screen.getByRole("button", { name: "Check training readiness" }) as HTMLButtonElement)
        .disabled
    ).toBe(true);
    await screen.findByRole("option", { name: "Mini · connected" });
    fireEvent.change(screen.getByLabelText("Remote Mac"), { target: { value: "mini" } });
    fireEvent.change(screen.getByLabelText("This Mac’s reachable address"), {
      target: { value: "192.0.2.1:32132" },
    });
    fireEvent.change(examples, { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Check training readiness" }));
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(1));
    expect(client.training.preflight.mock.calls[0][0]).toMatchObject({
      execution_mode: "two_mac",
      peer_id: "mini",
      coordinator: "192.0.2.1:32132",
    });
    const start = screen.getByRole("button", { name: "Start training" });
    await waitFor(() => expect((start as HTMLButtonElement).disabled).toBe(false));
    fireEvent.click(toggle);
    expect((start as HTMLButtonElement).disabled).toBe(true);
    expect(screen.queryByLabelText("Remote Mac")).toBeNull();
    fireEvent.change(screen.getByLabelText("Selected examples"), { target: { value: "3" } });
    fireEvent.click(screen.getByRole("button", { name: "Check training readiness" }));
    await waitFor(() => expect(client.training.preflight).toHaveBeenCalledTimes(2));
    const local = client.training.preflight.mock.calls[1][0];
    expect(local).toMatchObject({ execution_mode: "local", max_examples: 3 });
    expect(local).not.toHaveProperty("peer_id");
    expect(local).not.toHaveProperty("coordinator");
  });
});
