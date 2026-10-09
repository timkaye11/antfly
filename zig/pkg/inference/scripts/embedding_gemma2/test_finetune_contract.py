#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import hashlib
import json
import struct
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from finetune_contract import BASE_WEIGHT_SHA256, PERFORMANCE_CELLS, file_sha256, normalized_content_hash, paired_ratio_ci, retrieval_metrics, validate_candidate_artifacts, validate_dataset_manifest, validate_performance_cells, validate_performance_workload
from finetune_reference import TOKENIZER_SHA256, PreparedDataset, batch_indices, learning_rate, pad_ids, shuffle_epoch, sliced_inputs
from benchmark_finetune_paired import reference_source_digest, validate_report_provenance
from qualify_finetune_numeric import compare_array, reviewed_targets, validate_bf16_provenance, validate_resume
from qualify_finetune_quality import OFFICIAL_SOURCE_CONTRACTS, bootstrap_delta, main as quality_main
from prepare_training_dataset import add_train_only_negatives, decontaminate


def manifest():
    return {
        "schema": "antfly.embedding_gemma2.finetune_dataset.v1",
        "sources": [{"dataset": "public/x", "revision": "2" * 40, "config": "default", "license": "apache-2.0", "snapshot_sha256": "1" * 64}],
        "splits": {
            "train_content_sha256": ["a" * 64], "validation_content_sha256": ["b" * 64], "heldout_content_sha256": ["c" * 64],
            "train_corpus_ids": ["d1", "d2"], "train_positive_pairs": [["q1", "d1"]],
        },
        "miner": {"policy":"deterministic-train-only-cross-group-random-v1","frozen_base_model":None,"corpus_ids": ["d2"], "negative_pairs": [["q1", "d2"]]},
    }


def official_quality_fixture(query_count=1):
    digest="a"*64
    contracts=[];inventories=[];sources=[]
    for key,identity in OFFICIAL_SOURCE_CONTRACTS.items():
        dataset,language,role=key
        corpus={**identity,"corpus_sha256":digest,"queries_sha256":digest,"qrels_sha256":digest,
                "corpus_count":100,"query_count":query_count}
        contracts.append({"dataset":dataset,"language":language,"role":role,
                          "fixture_scope":"official-full-corpus","official_corpus":corpus})
        inventories.append({"dataset":dataset,"language":language,"role":role,"official_corpus":corpus})
        for repo,revision,config in ((corpus["source_repo"],corpus["source_revision"],corpus["corpus_config"]),
                                     (corpus["source_repo"],corpus["source_revision"],corpus["queries_config"]),
                                     (corpus["qrels_source_repo"],corpus["qrels_source_revision"],corpus["qrels_config"])):
            source={"source_repo":repo,"source_revision":revision,"snapshot_sha256":digest,
                    "config":config,"license":"public"}
            if source not in sources:sources.append(source)
    provenance={key:digest for key in ("prepared_manifest_sha256","training_source_manifest_sha256",
               "public_source_manifest_sha256")}
    provenance["base_weight_sha256"]=BASE_WEIGHT_SHA256;provenance["public_sources"]=sources
    provenance["candidates_by_seed"]={str(seed):{
        "candidate_checkpoint_sha256":hashlib.sha256(f"checkpoint-{seed}".encode()).hexdigest(),
        "candidate_model_sha256":hashlib.sha256(f"model-{seed}".encode()).hexdigest(),
        "candidate_bundle_manifest_sha256":hashlib.sha256(f"bundle-{seed}".encode()).hexdigest()}
        for seed in (17,29,43)}
    return ({"schema":"antfly.embedding_gemma2.finetune_quality_input.v1",
             "qualification_scope":"official-full-corpus","seeds":[17,29,43],"cell_contract":contracts,
             "cells":[],"provenance":provenance},
            {"schema":"antfly.embedding_gemma2.official_retrieval_sources.v1",
             "official_inventories":inventories})


class ContractTests(unittest.TestCase):
    def test_dataset_split_and_miner_contract(self):
        self.assertEqual(validate_dataset_manifest(manifest())["miner_corpus"], 1)
        value = manifest(); value["splits"]["heldout_content_sha256"] = ["a" * 64]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_dataset_manifest(value)
        value = manifest(); value["miner"]["negative_pairs"] = [["q1", "d1"]]
        with self.assertRaisesRegex(ValueError, "known positives"):
            validate_dataset_manifest(value)
        value = manifest(); value["sources"][0]["revision"] = "moving-main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_dataset_manifest(value)
        value = manifest(); value["splits"]["train_positive_pairs"] = [["q1"]]
        with self.assertRaisesRegex(ValueError, "two nonempty"):
            validate_dataset_manifest(value)

    def test_legacy_miner_policy_is_bounded_and_cannot_claim_frozen_mining(self):
        value=manifest();value["fixture_scope"]="bounded-functional-not-official-benchmark"
        value["miner"]["policy"]="deterministic-train-only-v1"
        result=validate_dataset_manifest(value)
        self.assertTrue(result["legacy_bounded_policy"])
        value["fixture_scope"]="official-full-corpus"
        with self.assertRaisesRegex(ValueError,"only for bounded fixtures"):
            validate_dataset_manifest(value)
        value["fixture_scope"]="bounded-functional-not-official-benchmark";value["miner"]["frozen_base_model"]="/model"
        with self.assertRaisesRegex(ValueError,"without a frozen-model claim"):
            validate_dataset_manifest(value)

    def test_normalization_catches_trivial_leakage(self):
        self.assertEqual(normalized_content_hash("  Hello\nWORLD "), normalized_content_hash("hello world"))

    def test_completed_step_schedule_matches_zig_boundary(self):
        self.assertAlmostEqual(learning_rate(0, 100), 2e-5)
        self.assertAlmostEqual(learning_rate(4, 100), 1e-4)
        self.assertAlmostEqual(learning_rate(5, 100), 1e-4)
        self.assertAlmostEqual(learning_rate(99, 100), 1e-4 / 95)

    def test_splitmix_epoch_order_and_cross_epoch_batch_are_stable(self):
        self.assertEqual(shuffle_epoch(8, 42, 0), [3, 1, 6, 2, 4, 0, 7, 5])
        self.assertEqual(batch_indices(4, 42, 1, 4), [3, 2, 0, 1])
        first_epoch = shuffle_epoch(5, 42, 0)
        second_epoch = shuffle_epoch(5, 42, 1)
        self.assertEqual(batch_indices(5, 42, 0, 4), first_epoch[:4])
        self.assertEqual(batch_indices(5, 42, 1, 4), first_epoch[4:])
        self.assertEqual(batch_indices(5, 42, 2, 4), second_epoch[:4])
        self.assertEqual(batch_indices(5, 42, 3, 4), second_epoch[4:])
        self.assertEqual(batch_indices(5, 42, 2, 4), batch_indices(5, 42, 2, 4))

    def test_unequal_length_microbatches_trim_to_their_active_width(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is optional for model-free contract tests")
        encoded=pad_ids([[1,2,3,4,5],[1,2],[1,2,3],[1]],"cpu")
        first=sliced_inputs(encoded,0,2);second=sliced_inputs(encoded,2,4)
        self.assertEqual(tuple(first["input_ids"].shape),(2,5))
        self.assertEqual(tuple(second["input_ids"].shape),(2,3))
        self.assertEqual(set(second),{"attention_mask","input_ids"})

    def test_native_prepared_index_is_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload = b'{"split":"train","query_ids":[2,1]}\n'
            (root / "data-000000.jsonl").write_bytes(payload)
            entry = struct.pack("<IIQ", 0, len(payload), 0) + hashlib.sha256(payload).digest() + bytes([0]) + bytes(7)
            index = b"EG2IDX1\0" + struct.pack("<Q", 1) + entry; (root / "index.bin").write_bytes(index)
            manifest = {"format":"antfly.embeddinggemma2.retrieval-shards/v1","records":1,"train_records":1,
                        "tokenizer_sha256":TOKENIZER_SHA256,"index_sha256":hashlib.sha256(index).hexdigest(),
                        "shards":[{"name":"data-000000.jsonl","bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            prepared = PreparedDataset(root); self.assertEqual(prepared[0]["query_ids"], [2,1]); prepared.close()
            (root / "data-000000.jsonl").write_bytes(payload + b"x")
            with self.assertRaisesRegex(ValueError, "shard mismatch"): PreparedDataset(root)

    def test_paired_bootstrap_preserves_covariance(self):
        lower, upper = paired_ratio_ci([10.0] * 20, [9.5] * 20, samples=500)
        self.assertAlmostEqual(lower, .95)
        self.assertAlmostEqual(upper, .95)

    def test_performance_contract_rejects_substituted_small_cell(self):
        cells=[{"logical_batch_size":b,"microbatch_size":m,"query_max_tokens":q,"document_max_tokens":d,
                "max_negatives":n,"lora":{"rank":r,"alpha":a}} for b,m,q,d,n,r,a in PERFORMANCE_CELLS]
        self.assertEqual(set(validate_performance_cells(cells)),PERFORMANCE_CELLS)
        cells[0]={"logical_batch_size":2,"microbatch_size":2,"query_max_tokens":8,"document_max_tokens":8,
                  "max_negatives":0,"lora":{"rank":16,"alpha":32}}
        with self.assertRaisesRegex(ValueError,"contract differs"):
            validate_performance_cells(cells)

    def test_performance_workload_rejects_underfilled_negatives_and_tokens(self):
        cell={"batch_size":32,"negatives":1,"query_tokens":128,"document_tokens":512,
              "observed_query_counts":[32]*100,"observed_candidate_counts":[64]*100,
              "active_query_tokens":{"minimum":110,"maximum":128,"mean":120,"microbatches":100},
              "active_document_tokens":{"minimum":470,"maximum":512,"mean":480,"microbatches":100}}
        validate_performance_workload(cell,100)
        cell["observed_candidate_counts"]=[63]*100
        with self.assertRaisesRegex(ValueError,"negative workload"):
            validate_performance_workload(cell,100)
        cell["observed_candidate_counts"]=[64]*100
        cell["active_document_tokens"]["mean"]=450
        with self.assertRaisesRegex(ValueError,"token workload"):
            validate_performance_workload(cell,100)

    def test_paired_provenance_binds_binary_manifest_and_reference_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);binary=root/"trainer";binary.write_bytes(b"binary")
            sources=root/"sources";sources.mkdir()
            for name in ("finetune_contract.py","finetune_reference.py","run_finetune_paired.py"):
                (sources/name).write_text(name)
            ids=[f"cell-{index}" for index in range(7)]
            mapping={cell:"a"*64 for cell in ids}
            manifest=root/"manifest.json"
            manifest.write_text(json.dumps({"schema":"antfly.embedding_gemma2.training-source-manifest/v1",
                                             "binary_sha256":file_sha256(binary),
                                             "files":[{"path":"trainer.zig","bytes":1,"sha256":"f"*64}]}))
            common={"schema":"antfly.embedding_gemma2.finetune_samples.v1","cell_contract":ids,
                    "provenance":{"hardware":"NVIDIA L4","software":{},"base_weight_sha256":BASE_WEIGHT_SHA256,
                                  "dataset_sha256":mapping,"job_fingerprint":mapping}}
            native=json.loads(json.dumps(common));reference=json.loads(json.dumps(common))
            native["provenance"].update(binary_sha256=file_sha256(binary),source_sha256=file_sha256(manifest))
            reference["provenance"]["source_sha256"]=reference_source_digest(sources)
            validate_report_provenance(native,reference,binary,manifest,sources)
            binary.write_bytes(b"mutated")
            with self.assertRaisesRegex(ValueError,"binding differs"):
                validate_report_provenance(native,reference,binary,manifest,sources)
            binary.write_bytes(b"binary");reference["schema"]="unreviewed"
            with self.assertRaisesRegex(ValueError,"schema is unsupported"):
                validate_report_provenance(native,reference,binary,manifest,sources)

    def test_retrieval_metrics(self):
        result = retrieval_metrics(["bad", "good", "also"], {"good"})
        self.assertEqual(result["mrr_at_10"], .5)
        self.assertEqual(result["recall_at_100"], 1.0)

    def test_numeric_gate_rejects_nonfinite_and_large_error(self):
        self.assertTrue(compare_array([1.0, 2.0], [1.0, 2.0], 1e-5, 1e-4)["pass"])
        self.assertFalse(compare_array([1.0, 2.1], [1.0, 2.0], 1e-5, 1e-4)["pass"])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare_array([float("nan")], [0.0], 1e-5, 1e-4)

    def test_numeric_bf16_provenance_rejects_attention_and_trace_stage_mismatch(self):
        shared={"logical_batch_size":2,"microbatch_size":2,"max_negatives":1,"rank":16,"alpha":32,"seed":42,
                "base_weight_sha256":BASE_WEIGHT_SHA256,"prepared_manifest_sha256":"b"*64,"preprocessing_sha256":"c"*64,
                "padding_policy":"per-micro-active-v1","compute_profile":"bf16-edges-fp32-masters-v1",
                "gradient_trace_stage":"pre-clip"}
        native={"provenance":{**shared,"attention":"eager"}}
        reference={"provenance":{**shared,"attention":"eager","loaded_attention":"eager","targets":sorted(reviewed_targets())}}
        validate_bf16_provenance(native,reference)
        reference["provenance"]["loaded_attention"]="sdpa"
        with self.assertRaisesRegex(ValueError,"loaded eager"): validate_bf16_provenance(native,reference)
        reference["provenance"]["loaded_attention"]="eager";native["provenance"]["gradient_trace_stage"]="post-clip"
        with self.assertRaisesRegex(ValueError,"gradient_trace_stage"): validate_bf16_provenance(native,reference)
        native["provenance"]["gradient_trace_stage"]="pre-clip";native["provenance"].pop("attention")
        with self.assertRaisesRegex(ValueError,"explicitly identify eager"): validate_bf16_provenance(native,reference)

    def test_resume_gate_rejects_bare_pass_and_mutated_state(self):
        with self.assertRaisesRegex(ValueError,"unsupported schema"):
            validate_resume({"pass":True},"bare")
        value={"schema":"antfly.embedding_gemma2.finetune_resume.v1","pass":True,
               "provenance_equal":True,"uninterrupted_next_step_sha256":"a"*64,
               "resumed_next_step_sha256":"a"*64}
        validate_resume(value,"complete")
        value["resumed_next_step_sha256"]="b"*64
        with self.assertRaisesRegex(ValueError,"equal authenticated state"):
            validate_resume(value,"mutated")

    def test_quality_bootstrap_is_paired(self):
        lower, upper = bootstrap_delta([.6] * 20, [.5] * 20)
        self.assertAlmostEqual(lower, .1)
        self.assertAlmostEqual(upper, .1)
        with self.assertRaisesRegex(ValueError, "finite"):
            bootstrap_delta([float("nan")], [0.5])

    def test_candidate_artifacts_bind_actual_model_checkpoint_seed_and_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);training_source=root/"training-source.json";training_source.write_text("{}")
            prepared=root/"prepared.json";prepared.write_text(json.dumps({"source_manifest_sha256":file_sha256(training_source)}))
            checkpoint=root/"checkpoint.safetensors";checkpoint.write_bytes(b"checkpoint")
            bundle=root/"bundle";bundle.mkdir();model=bundle/"model.safetensors";model.write_bytes(b"weights")
            provenance={"schema":"antfly.embedding_gemma2.finetune_export.v1","base_weight_sha256":BASE_WEIGHT_SHA256,
                "merged_weight_sha256":file_sha256(model),"training":{"schema":"antfly.embedding_gemma2.native_training.v1",
                "seed":17,"dataset_sha256":file_sha256(prepared),"checkpoint_weight_sha256":file_sha256(checkpoint),
                "public_source_manifest_sha256":file_sha256(training_source)}}
            (bundle/"antfly_finetune_provenance.json").write_text(json.dumps(provenance))
            validated=validate_candidate_artifacts(checkpoint=checkpoint,model=model,prepared_manifest=prepared,
                training_source_manifest=training_source,seed=17)
            self.assertEqual(validated["candidate_model_sha256"],file_sha256(model))
            model.write_bytes(b"substituted")
            with self.assertRaisesRegex(ValueError,"merged model bytes"):
                validate_candidate_artifacts(checkpoint=checkpoint,model=model,prepared_manifest=prepared,
                    training_source_manifest=training_source,seed=17)
            model.write_bytes(b"weights");provenance["training"]["seed"]=29
            (bundle/"antfly_finetune_provenance.json").write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError,"seed/data/checkpoint/source"):
                validate_candidate_artifacts(checkpoint=checkpoint,model=model,prepared_manifest=prepared,
                    training_source_manifest=training_source,seed=17)

    def test_quality_gate_rejects_small_fixture_declared_as_official_corpus(self):
        value,public=official_quality_fixture(query_count=5)
        cell={"dataset":"msmarco","language":None,"role":"target","dimension":128,"seed":17,
              **value["provenance"]["candidates_by_seed"]["17"],
              "candidate_query_ids":["q1","q2","q3","q4"],"base_query_ids":["q1","q2","q3","q4"],
              "candidate_per_query":{metric:[1.0]*4 for metric in ("mrr_at_10","ndcg_at_10","recall_at_100")},
              "base_per_query":{metric:[1.0]*4 for metric in ("mrr_at_10","ndcg_at_10","recall_at_100")}}
        value["cells"]=[cell]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"input.json";report=root/"report.json";manifest_path=root/"public.json";training=root/"training.json";prepared=root/"prepared.json";base=root/"base"
            source.write_text(json.dumps(value));manifest_path.write_text(json.dumps(public));training.write_text("{}");prepared.write_text(json.dumps({"source_manifest_sha256":"a"*64}));base.write_bytes(b"x")
            argv=["quality","--input",str(source),"--base-weight",str(base),"--prepared-manifest",str(prepared),"--training-source-manifest",str(training),"--public-source-manifest",str(manifest_path),"--report",str(report)]
            digest_by_name={"base":BASE_WEIGHT_SHA256,"prepared.json":"a"*64,"training.json":"a"*64,"public.json":"a"*64}
            for seed in (17,29,43):
                checkpoint=root/f"checkpoint-{seed}";model=root/f"model-{seed}.safetensors";checkpoint.write_bytes(b"x");model.write_bytes(b"x")
                argv.extend(["--candidate-checkpoint",f"{seed}={checkpoint}","--candidate-model",f"{seed}={model}"])
                digest_by_name[checkpoint.name]=value["provenance"]["candidates_by_seed"][str(seed)]["candidate_checkpoint_sha256"]
                digest_by_name[model.name]=value["provenance"]["candidates_by_seed"][str(seed)]["candidate_model_sha256"]
            candidate_validator=lambda **kwargs:value["provenance"]["candidates_by_seed"][str(kwargs["seed"])]
            with mock.patch.object(sys,"argv",argv),mock.patch("qualify_finetune_quality.file_sha256",side_effect=lambda path:digest_by_name[path.name]),mock.patch("qualify_finetune_quality.validate_candidate_artifacts",side_effect=candidate_validator):
                with self.assertRaisesRegex(ValueError,"cover the declared official query corpus"):quality_main()

    def test_quality_gate_requires_corpus_source_in_authenticated_provenance(self):
        value,public=official_quality_fixture();value["provenance"]["public_sources"]=value["provenance"]["public_sources"][1:]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/"input.json";report=root/"report.json";manifest_path=root/"public.json";training=root/"training.json";prepared=root/"prepared.json";base=root/"base"
            source.write_text(json.dumps(value));manifest_path.write_text(json.dumps(public));training.write_text("{}");prepared.write_text(json.dumps({"source_manifest_sha256":"a"*64}));base.write_bytes(b"x")
            argv=["quality","--input",str(source),"--base-weight",str(base),"--prepared-manifest",str(prepared),"--training-source-manifest",str(training),"--public-source-manifest",str(manifest_path),"--report",str(report)]
            digest_by_name={"base":BASE_WEIGHT_SHA256,"prepared.json":"a"*64,"training.json":"a"*64,"public.json":"a"*64}
            for seed in (17,29,43):
                checkpoint=root/f"checkpoint-{seed}";model=root/f"model-{seed}.safetensors";checkpoint.write_bytes(b"x");model.write_bytes(b"x")
                argv.extend(["--candidate-checkpoint",f"{seed}={checkpoint}","--candidate-model",f"{seed}={model}"])
                digest_by_name[checkpoint.name]=value["provenance"]["candidates_by_seed"][str(seed)]["candidate_checkpoint_sha256"]
                digest_by_name[model.name]=value["provenance"]["candidates_by_seed"][str(seed)]["candidate_model_sha256"]
            candidate_validator=lambda **kwargs:value["provenance"]["candidates_by_seed"][str(kwargs["seed"])]
            with mock.patch.object(sys,"argv",argv),mock.patch("qualify_finetune_quality.file_sha256",side_effect=lambda path:digest_by_name[path.name]),mock.patch("qualify_finetune_quality.validate_candidate_artifacts",side_effect=candidate_validator):
                with self.assertRaisesRegex(ValueError,"absent from public-source provenance"):quality_main()

    def test_public_fixture_mining_is_train_only_and_deterministic(self):
        records = [
            {"query":"q1","query_id":"q1","group_id":"g1","split":"train","positive":{"id":"d1","text":"one"},"negatives":[],"known_positive_ids":["d1"]},
            {"query":"q2","query_id":"q2","group_id":"g2","split":"train","positive":{"id":"d2","text":"two"},"negatives":[],"known_positive_ids":["d2"]},
            {"query":"q3","query_id":"q3","group_id":"g3","split":"test","positive":{"id":"d3","text":"three"},"negatives":[],"known_positive_ids":["d3"]},
        ]
        add_train_only_negatives(records, 42)
        self.assertEqual(records[0]["negatives"][0]["id"], "d2")
        self.assertEqual(records[1]["negatives"][0]["id"], "d1")
        self.assertEqual(records[2]["negatives"], [])

    def test_public_fixture_decontamination_prefers_heldout(self):
        def row(query_id, split, query, group=None):
            document_id=query_id+"d"
            return {"query":query,"query_id":query_id,"group_id":group or query_id,"split":split,
                    "positive":{"id":document_id,"text":query_id+" doc"},"known_positive_ids":[document_id]}
        accepted, hashes = decontaminate([row("train","train","same"), row("test","test","same")])
        self.assertEqual([value["query_id"] for value in accepted], ["test"])
        self.assertFalse(hashes["train"])
        accepted, _ = decontaminate([row("train","train","unique","project"), row("test","test","other","project")])
        self.assertEqual([value["query_id"] for value in accepted], ["test"])

    def test_public_fixture_decontamination_merges_same_split_duplicate_query(self):
        first={"query":"same","query_id":"a","group_id":"a","split":"train","positive":{"id":"d1","text":"one"},"known_positive_ids":["d1"]}
        second={"query":"same","query_id":"b","group_id":"b","split":"train","positive":{"id":"d2","text":"two"},"known_positive_ids":["d2"]}
        accepted, _ = decontaminate([second, first])
        self.assertEqual(len(accepted),1)
        self.assertEqual(accepted[0]["query_id"],"a")
        self.assertEqual(accepted[0]["known_positive_ids"],["d1","d2"])


if __name__ == "__main__":
    unittest.main()
