#!/usr/bin/env python3
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import evaluate_finetune_full_corpus as subject


def full(*values):
    row = list(values) + [0.0] * (768 - len(values))
    norm = math.sqrt(sum(value * value for value in row))
    return [value / norm for value in row]


class FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def __call__(self, texts, task):
        self.calls.append((list(texts), task))
        return [self.vectors[text] for text in texts]


class FullCorpusTest(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_graded_metrics_and_binary_metrics(self):
        values = subject.metrics(["weak", "miss", "strong"], {"strong": 3, "weak": 1})
        self.assertEqual(values["mrr_at_10"], 1.0)
        expected = ((2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(4)) / (
            (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3))
        self.assertAlmostEqual(values["ndcg_at_10"], expected)
        self.assertEqual(values["recall_at_100"], 1.0)

    def test_top_k_ties_are_doc_id_ascending(self):
        top = subject.TopK(1, 3)
        top.add([[1.0, 1.0]], ["z", "b"])
        top.add([[1.0, .5]], ["a", "c"])
        self.assertEqual(top.ids(0), ["a", "b", "z"])

    def test_numpy_and_dependency_free_scoring_choose_same_near_tie_order(self):
        queries = [[1.0, 2.0, 3.0]]
        documents = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0 + 1e-12]]
        accelerated = subject.similarities(queries, documents)
        saved = subject._np
        try:
            subject._np = None
            fallback = subject.similarities(queries, documents)
        finally:
            subject._np = saved
        accelerated_top, fallback_top = subject.TopK(1), subject.TopK(1)
        accelerated_top.add(accelerated, ["a", "b"]); fallback_top.add(fallback, ["a", "b"])
        self.assertEqual(accelerated_top.ids(0), fallback_top.ids(0))

    def test_streams_chunks_and_never_embeds_more_than_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); corpus = root / "corpus.jsonl"; queries = root / "queries.jsonl"; qrels = root / "qrels.jsonl"
            docs = [{"_id": f"d{i}", "title": "T" if i == 0 else "", "text": f"doc{i}"} for i in range(5)]
            qs = [{"_id": f"q{i}", "text": f"query{i}"} for i in range(3)]
            self.write_jsonl(corpus, docs); self.write_jsonl(queries, qs)
            self.write_jsonl(qrels, [{"query_id": f"q{i}", "corpus_id": f"d{i}", "score": 2} for i in range(3)])
            vectors = {f"query{i}": full(1 if i == 0 else 0, 1 if i == 1 else 0, 1 if i == 2 else 0) for i in range(3)}
            for i in range(5): vectors[("title: T | text: doc0" if i == 0 else f"doc{i}")] = full(1 if i == 0 else 0, 1 if i == 1 else 0, 1 if i == 2 else 0, 1 if i > 2 else 0)
            base, candidate = FakeEmbedder(vectors), FakeEmbedder(vectors)
            ids, base_values, candidate_values, passes = subject.evaluate(corpus=corpus, queries=queries, qrels=qrels,
                base_embed=base, candidate_embed=candidate, dimension=128, query_chunk_size=2, corpus_chunk_size=2)
            self.assertEqual(ids, ["q0", "q1", "q2"]); self.assertEqual(passes, 2)
            self.assertEqual(base_values, candidate_values)
            self.assertTrue(all(len(texts) <= (2 if task == "RETRIEVAL_DOCUMENT" else 2) for texts, task in base.calls))
            self.assertEqual(sum(task == "RETRIEVAL_DOCUMENT" for _, task in base.calls), 6)
            self.assertEqual(base.calls[1][0][0], "title: T | text: doc0")

    def test_full_vector_is_required_and_mrl_is_renormalized(self):
        with self.assertRaisesRegex(ValueError, "full 768"):
            subject.truncate_normalize([[1.0, 2.0]], 2)
        value = subject.truncate_normalize([full(3, 4)], 1)
        self.assertAlmostEqual(math.sqrt(sum(item * item for item in value[0])), 1.0)

    def test_seed_paths_require_exact_distinct_quality_seeds(self):
        values=[f"{seed}=/tmp/model-{seed}" for seed in subject.QUALITY_SEEDS]
        self.assertEqual(set(subject.seed_paths(values,"candidate")),set(subject.QUALITY_SEEDS))
        with self.assertRaisesRegex(ValueError,"exactly seeds"):
            subject.seed_paths(values[:-1],"candidate")

    def test_qrels_reject_nonfinite_grades_and_missing_positive_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); corpus = root / "corpus.jsonl"; queries = root / "queries.jsonl"; qrels = root / "qrels.jsonl"
            self.write_jsonl(corpus, [{"id": "present", "text": "doc"}])
            self.write_jsonl(queries, [{"id": "q", "text": "query"}])
            self.write_jsonl(qrels, [{"query_id": "q", "corpus_id": "missing", "score": float("nan")}])
            with self.assertRaisesRegex(ValueError, "nonnegative numeric"):
                subject.load_qrels(qrels, {"q"})
            self.write_jsonl(qrels, [{"query_id": "q", "corpus_id": "missing", "score": 1}])
            embedder = FakeEmbedder({"query": full(1), "doc": full(1)})
            with self.assertRaisesRegex(ValueError, "absent from the full corpus"):
                subject.evaluate(corpus=corpus, queries=queries, qrels=qrels, base_embed=embedder,
                    candidate_embed=embedder, dimension=128, query_chunk_size=1, corpus_chunk_size=1)

    def test_merge_accepts_only_complete_cells_with_exact_artifacts_and_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = "a" * 64
            inventories = []
            for dataset, language, role in subject.CELL_KEYS:
                official = {"source_repo": "owner/data", "source_revision": "1" * 40,
                    "corpus_config": "corpus", "corpus_split": "corpus",
                    "queries_config": "queries", "queries_split": "queries",
                    "qrels_source_repo": "owner/qrels", "qrels_source_revision": "2" * 40,
                    "qrels_config": "default", "qrels_split": "test",
                    "corpus_sha256": digest, "queries_sha256": digest,
                    "qrels_sha256": digest, "corpus_count": 2, "query_count": 1}
                inventories.append({"dataset": dataset, "language": language, "role": role,
                                    "official_corpus": official})
            source = root / "source.json"
            source.write_text(json.dumps({"schema": "antfly.embedding_gemma2.official_retrieval_sources.v1",
                "official_inventories": inventories, "sources": []}))
            source_digest = subject.file_sha256(source)
            training_source = root / "training-source.json"; training_source.write_text('{"training":true}')
            training_digest = subject.file_sha256(training_source)
            prepared = root / "prepared.json"; prepared.write_text(json.dumps({"source_manifest_sha256": training_digest}))
            bundles = {}; models = {}; checkpoints = {}
            for seed in subject.QUALITY_SEEDS:
                checkpoints[seed] = root / f"checkpoint-{seed}"; checkpoints[seed].write_bytes(f"candidate-{seed}".encode())
                bundle_dir=root/f"bundle-{seed}";bundle_dir.mkdir()
                models[seed]=bundle_dir/"model.safetensors";models[seed].write_bytes(f"model-{seed}".encode())
                bundles[seed]=bundle_dir/"antfly_finetune_provenance.json"
                bundles[seed].write_text(json.dumps({"schema":"antfly.embedding_gemma2.finetune_export.v1",
                    "base_weight_sha256":subject.BASE_WEIGHT_SHA256,
                    "merged_weight_sha256":subject.file_sha256(models[seed]),
                    "training":{"schema":"antfly.embedding_gemma2.native_training.v1","seed":seed,
                        "dataset_sha256":subject.file_sha256(prepared),
                        "checkpoint_weight_sha256":subject.file_sha256(checkpoints[seed]),
                        "public_source_manifest_sha256":training_digest}}))
            cells = []
            official_by_key = {(row["dataset"], row["language"], row["role"]): row["official_corpus"] for row in inventories}
            for dataset, language, role in subject.CELL_KEYS:
                for dimension in subject.MRL_DIMENSIONS:
                    for seed in subject.QUALITY_SEEDS:
                        artifacts = {"bundle_manifest": subject.artifact(bundles[seed]),
                            "model": subject.artifact(models[seed]),
                            "checkpoint": subject.artifact(checkpoints[seed]),
                            "prepared_manifest": subject.artifact(prepared),
                            "training_source_manifest": subject.artifact(training_source),
                            "source_manifest": subject.artifact(source)}
                        cell = {"qualification_scope": "official-full-corpus", "dataset": dataset,
                            "language": language, "role": role, "dimension": dimension, "seed": seed,
                            "base_query_ids": ["q"], "candidate_query_ids": ["q"],
                            "base_per_query": {name: [0.5] for name in subject.METRICS},
                            "candidate_per_query": {name: [0.6] for name in subject.METRICS},
                            "official_corpus": official_by_key[(dataset, language, role)],
                            "execution": {"backend": "cuda"}, "artifacts": artifacts}
                        path = root / f"cell-{len(cells)}.json"; path.write_text(json.dumps(cell)); cells.append(path)
            output = root / "merged.json"
            args = Namespace(cells=cells,
                candidate_checkpoint=[f"{seed}={checkpoints[seed]}" for seed in subject.QUALITY_SEEDS],
                candidate_model=[f"{seed}={models[seed]}" for seed in subject.QUALITY_SEEDS],
                prepared_manifest=prepared, training_source_manifest=training_source,
                source_manifest=source, output=output,
                base_weight_sha256="b" * 64)
            self.assertEqual(subject.command_merge(args), 0)
            merged = json.loads(output.read_text())
            self.assertEqual(len(merged["cells"]), 60)
            self.assertEqual(merged["provenance"]["public_source_manifest_sha256"], source_digest)
            self.assertEqual(set(merged["provenance"]["candidates_by_seed"]), {"17", "29", "43"})
            bad = json.loads(cells[0].read_text()); bad["execution"]["backend"] = "cpu"; cells[0].write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "CUDA"):
                subject.command_merge(args)


if __name__ == "__main__":
    unittest.main()
