import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TrainingDatasets } from "./training-datasets";

const { endpoint, client } = vi.hoisted(() => ({
  endpoint: { current: "http://server-a" },
  client: {
    training: {
      datasets: vi.fn(),
      createDataset: vi.fn(),
      uploadDatasetChunk: vi.fn(),
      prepareDataset: vi.fn(),
      huggingFace: vi.fn(),
      cancelDataset: vi.fn(),
      removeDataset: vi.fn(),
    },
  },
}));
vi.mock("@/hooks/use-api-config", () => ({
  useApiConfig: () => ({ apiUrl: endpoint.current, client }),
}));
const props = {
  family: "gliner25" as const,
  modelDir: "/models/gemma",
  selectedId: "",
  onSelect: vi.fn(),
  calibrationId: "template",
  onCalibration: vi.fn(),
  testId: "template",
  onTest: vi.fn(),
};

function uploadFile(size: number) {
  const bytes = new TextEncoder().encode("x".repeat(size));
  const file = new File([bytes], "training.jsonl", { type: "application/jsonl" });
  Object.defineProperty(file, "slice", {
    value: (start: number, end: number) => ({
      arrayBuffer: async () => bytes.slice(start, end).buffer,
      text: async () => new TextDecoder().decode(bytes.slice(start, end)),
    }),
  });
  fireEvent.change(screen.getByLabelText("CSV or JSONL file"), { target: { files: [file] } });
}

beforeEach(() => {
  endpoint.current = "http://server-a";
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  client.training.datasets.mockResolvedValue({ datasets: [] });
  client.training.createDataset.mockResolvedValue({
    id: "dataset-1",
    uploaded_bytes: 0,
    status: "uploading",
  });
  client.training.uploadDatasetChunk.mockImplementation(async (_id, offset, data) => ({
    id: "dataset-1",
    uploaded_bytes: offset + atob(data).length,
  }));
  client.training.prepareDataset.mockResolvedValue({ id: "dataset-1", status: "preparing" });
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe("training dataset import", () => {
  it("offers imported datasets or a JSONL path without a job-template dependency", async () => {
    const changeFile = vi.fn();
    render(
      <TrainingDatasets {...props} useJobTemplate={false} trainFile="" onTrainFile={changeFile} />
    );
    expect(screen.queryByRole("option", { name: /job template/i })).toBeNull();
    expect(screen.getByRole("option", { name: "Select a prepared dataset" })).toBeTruthy();
    fireEvent.click(screen.getByText("Use a training file already on this Mac"));
    fireEvent.change(screen.getByLabelText(/Training JSONL path/), {
      target: { value: "/data/train.jsonl" },
    });
    expect(changeFile).toHaveBeenCalledWith("/data/train.jsonl");
    fireEvent.click(screen.getByText("Calibration & evaluation datasets"));
    expect((screen.getByLabelText("Calibration dataset") as HTMLSelectElement).value).toBe("none");
    expect((screen.getByLabelText("Held-out test dataset") as HTMLSelectElement).value).toBe(
      "none"
    );
    await waitFor(() => expect(client.training.datasets).toHaveBeenCalled());
  });

  it("allows odd local datasets and applies the two-Mac row constraint only when enabled", async () => {
    client.training.datasets.mockResolvedValue({
      datasets: [
        {
          id: "odd",
          name: "Odd dataset",
          family: "gliner25",
          status: "ready",
          row_count: 3,
          spec: { source: "upload" },
        },
      ],
    });
    const view = render(<TrainingDatasets {...props} />);
    const option = await within(
      screen.getByRole("combobox", { name: /Training dataset/ })
    ).findByRole("option", { name: /Odd dataset/ });
    expect((option as HTMLOptionElement).disabled).toBe(false);
    view.rerender(<TrainingDatasets {...props} distributed />);
    expect((option as HTMLOptionElement).disabled).toBe(true);
    view.rerender(<TrainingDatasets {...props} />);
    expect((option as HTMLOptionElement).disabled).toBe(false);
  });

  it("uploads bounded chunks in order and prepares only after the complete file", async () => {
    render(<TrainingDatasets {...props} />);
    uploadFile(70000);
    await screen.findByText("Source preview (truncated)");
    fireEvent.click(screen.getByRole("button", { name: "Import and prepare dataset" }));
    await waitFor(() => expect(client.training.prepareDataset).toHaveBeenCalledTimes(1));
    const chunks = client.training.uploadDatasetChunk.mock.calls;
    expect(chunks.map((call) => call[1])).toEqual([0, 32768, 65536]);
    expect(chunks.map((call) => atob(call[2]).length)).toEqual([32768, 32768, 4464]);
    expect(client.training.createDataset.mock.calls[0][0]).toMatchObject({
      source: "upload",
      filename: "training.jsonl",
      size_bytes: 70000,
      format: "gliner25",
    });
  });

  it("does not send later upload chunks or prepare after switching endpoints", async () => {
    let finish!: (value: unknown) => void;
    client.training.uploadDatasetChunk.mockImplementation(
      () =>
        new Promise((resolve) => {
          finish = resolve;
        })
    );
    const view = render(<TrainingDatasets {...props} />);
    uploadFile(70000);
    await screen.findByText("Source preview (truncated)");
    fireEvent.click(screen.getByRole("button", { name: "Import and prepare dataset" }));
    await waitFor(() => expect(client.training.uploadDatasetChunk).toHaveBeenCalledTimes(1));
    const signal = client.training.uploadDatasetChunk.mock.calls[0][3] as AbortSignal;
    endpoint.current = "http://server-b";
    view.rerender(<TrainingDatasets {...props} />);
    expect(signal.aborted).toBe(true);
    await act(async () => finish({ id: "dataset-1", uploaded_bytes: 32768 }));
    expect(client.training.uploadDatasetChunk).toHaveBeenCalledTimes(1);
    expect(client.training.prepareDataset).not.toHaveBeenCalled();
  });

  it("imports the previewed Hugging Face split with explicit Gemma columns and tokenizer", async () => {
    client.training.huggingFace
      .mockResolvedValueOnce({
        splits: [{ config: "default", split: "train" }],
        features: [],
        rows: [],
      })
      .mockResolvedValueOnce({
        splits: [],
        features: [{ name: "question" }, { name: "answer" }],
        rows: ['{"question":"Hello","answer":"World"}'],
        total_rows: 20,
      });
    render(<TrainingDatasets {...props} family="gemma4" />);
    fireEvent.click(screen.getByRole("button", { name: "Hugging Face" }));
    fireEvent.change(screen.getByLabelText("Hugging Face dataset"), {
      target: { value: "org/data" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Load subsets and splits" }));
    await screen.findByRole("option", { name: "default / train" });
    fireEvent.click(screen.getByRole("button", { name: "Preview rows" }));
    await screen.findByText("20 rows in this split");
    fireEvent.change(screen.getByLabelText("Prompt column"), { target: { value: "question" } });
    fireEvent.change(screen.getByLabelText("Response column"), { target: { value: "answer" } });
    fireEvent.click(screen.getByRole("button", { name: "Import and prepare dataset" }));
    await waitFor(() => expect(client.training.prepareDataset).toHaveBeenCalledTimes(1));
    expect(client.training.createDataset.mock.calls[0][0]).toMatchObject({
      source: "huggingface",
      family: "gemma4",
      hf_dataset: "org/data",
      hf_config: "default",
      hf_split: "train",
      columns: { prompt: "question", response: "answer" },
      model_dir: "/models/gemma",
      max_seq_len: 512,
    });
    expect(client.training.uploadDatasetChunk).not.toHaveBeenCalled();
  });

  it("aborts a resumed upload when switching endpoints", async () => {
    const completions: ((value: unknown) => void)[] = [];
    client.training.uploadDatasetChunk.mockImplementation(
      () => new Promise((resolve) => completions.push(resolve))
    );
    const view = render(<TrainingDatasets {...props} />);
    uploadFile(70000);
    await screen.findByText("Source preview (truncated)");
    fireEvent.click(screen.getByRole("button", { name: "Import and prepare dataset" }));
    await waitFor(() => expect(client.training.uploadDatasetChunk).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Stop request" }));
    fireEvent.click(screen.getByRole("button", { name: "Import and prepare dataset" }));
    await waitFor(() => expect(client.training.uploadDatasetChunk).toHaveBeenCalledTimes(2));
    const signal = client.training.uploadDatasetChunk.mock.calls[1][3] as AbortSignal;
    expect(signal.aborted).toBe(false);
    endpoint.current = "http://server-b";
    view.rerender(<TrainingDatasets {...props} />);
    expect(signal.aborted).toBe(true);
    await act(async () => {
      for (const finish of completions) finish({ id: "dataset-1", uploaded_bytes: 32768 });
    });
    expect(client.training.uploadDatasetChunk).toHaveBeenCalledTimes(2);
    expect(client.training.prepareDataset).not.toHaveBeenCalled();
  });
});
