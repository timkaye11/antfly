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

"""Gate three-seed retrieval quality per dataset, language and MRL dimension."""

from __future__ import annotations
import argparse,itertools,json,math,random,statistics
from collections import Counter
from pathlib import Path
from finetune_contract import BASE_WEIGHT_SHA256,MRL_DIMENSIONS,QUALITY_SEEDS,REQUIRED_QUALITY_CELLS,file_sha256,load_json,valid_sha256,validate_candidate_artifacts,validate_official_source_manifest

REQUIRED_CELLS=REQUIRED_QUALITY_CELLS
REQUIRED_PROVENANCE=("base_weight_sha256","prepared_manifest_sha256",
                     "training_source_manifest_sha256","public_source_manifest_sha256")
def source_contract(repo,revision,corpus_config,corpus_split,queries_config,queries_split,
                    qrels_repo,qrels_revision,qrels_config,qrels_split):
    return {"source_repo":repo,"source_revision":revision,"corpus_config":corpus_config,
            "corpus_split":corpus_split,"queries_config":queries_config,"queries_split":queries_split,
            "qrels_source_repo":qrels_repo,"qrels_source_revision":qrels_revision,
            "qrels_config":qrels_config,"qrels_split":qrels_split}

OFFICIAL_SOURCE_CONTRACTS={
    ("msmarco",None,"target"):source_contract("BeIR/msmarco","a918e0d11a77ed33f42f29d98340b655593b96ad","corpus","corpus","queries","queries","BeIR/msmarco-qrels","253fbf8a3f8d4a0932b63882b5162bedc84779f5","default","dev"),
    ("codesearchnet","python","target"):source_contract("code-search-net/code_search_net","bd0cf261e357a3eb5c8fba490d23ec1a1cd59555","python","test","python","test","code-search-net/code_search_net","bd0cf261e357a3eb5c8fba490d23ec1a1cd59555","python","test"),
    ("codesearchnet","go","target"):source_contract("code-search-net/code_search_net","bd0cf261e357a3eb5c8fba490d23ec1a1cd59555","go","test","go","test","code-search-net/code_search_net","bd0cf261e357a3eb5c8fba490d23ec1a1cd59555","go","test"),
    ("scifact",None,"retention"):source_contract("BeIR/scifact","b3b5335604bf5ee3c4447671af975ea25143d4f5","corpus","corpus","queries","queries","BeIR/scifact-qrels","2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029","default","test"),
    ("nfcorpus",None,"retention"):source_contract("BeIR/nfcorpus","b5026a0e96e8a7ac4f95f482a596389289d46269","corpus","corpus","queries","queries","BeIR/nfcorpus-qrels","a451b3b26d3ae1358f259c1a3a4dd61fcea35a65","default","test"),
}

# Counts are reviewed against the complete, pinned dataset files. A caller's
# self-declared inventory cannot turn a sampled fixture into release evidence.
OFFICIAL_COUNTS = {
    ("msmarco", None, "target"): (8841823, 6980),
    ("codesearchnet", "python", "target"): (22176, 22176),
    ("codesearchnet", "go", "target"): (14291, 14291),
    ("scifact", None, "retention"): (5183, 300),
    ("nfcorpus", None, "retention"): (3633, 323),
}

# Canonical JSONL from the complete authenticated snapshots, produced by
# materialize_official_retrieval_sources.py. Pin bytes as well as row counts.
OFFICIAL_FILES = {
    ("msmarco", None, "target"): (
        "046ff8d3cd246416222b7b76ecd59b8c95041ddf22eecfe6ecfd10e8bfd77e9b",
        "f07e3e817f5403aaf42263052feed61e4dc16530eb27733405da34e01edf65d7",
        "0cb5e3d75cf6fe2566f754be18177d0d2b24eb94dbd2ef3cbf12203b99c044fb"),
    ("codesearchnet", "python", "target"): (
        "28bd0104dcb3266726166751d22f3dd63d8f21ca9c8822e2cd7ad00291f35a21",
        "3a6e834bed3fc36b1a66a72165aae24594cddd30915fbb267b6e88c07cf867c1",
        "c95d3f3616e9a068b35dd3379e94aa2cd47013a872b2f37ffb7716e229427d35"),
    ("codesearchnet", "go", "target"): (
        "fa353e51eb06b27372fc08facf435acf365c915186ab20df6b3a3642baf0bfba",
        "90b33467dfa9947ccdda71ca0f7d6ce8432ae52158197a956ddbd5bf56fa1838",
        "90b62dca6c80c2454c7f4fcb780ee3009014206115821c63451391746ba0cd1e"),
    ("scifact", None, "retention"): (
        "83824fbe6112ee2614496dbdbaead3d32ad9cf0507c3b6ca244850f2303cea7e",
        "8a2f8d87d612b71c697ca7fe30c68d0afa4b9234dcf98b353ab15f571134d768",
        "c26de8b8c8910eb889b3576890042c2a2b9102396fbec1ac2e747c168d6eceec"),
    ("nfcorpus", None, "retention"): (
        "b5147cf309c8c8e09a18fa874d8aa2c1036f2c4779b6df64e71b79c6370a16d0",
        "07dedb839bad657b00782de502f02452482dc2498d3d597b01109eb709e04188",
        "e8bfbb5b33ba966426cabbe271a932c831f65d63a20f67616152aa83b268e2a5"),
}


def validate_reviewed_inventory(inventory):
    if set(inventory) != set(OFFICIAL_SOURCE_CONTRACTS):
        raise ValueError("official inventory must contain the five reviewed roles")
    for key, corpus in inventory.items():
        if any(corpus.get(name) != value for name, value in OFFICIAL_SOURCE_CONTRACTS[key].items()):
            raise ValueError("inventory differs from the reviewed official source contract")
        if (corpus.get("corpus_count"), corpus.get("query_count")) != OFFICIAL_COUNTS[key]:
            raise ValueError("inventory differs from reviewed official corpus/query counts")
        if tuple(corpus.get(name + '_sha256') for name in ('corpus', 'queries', 'qrels')) != OFFICIAL_FILES[key]:
            raise ValueError("inventory differs from reviewed official corpus/query/qrel bytes")

def bootstrap_delta(candidate:list[float],base:list[float],seed:int=20261008)->tuple[float,float]:
    if len(candidate)!=len(base) or not candidate: raise ValueError("paired query arrays required")
    if not all(isinstance(value,(int,float)) and math.isfinite(value) for value in candidate+base): raise ValueError("quality arrays must be finite")
    rng=random.Random(seed); vals=[]
    for _ in range(5000):
        ids=[rng.randrange(len(base)) for _ in base]; vals.append(statistics.fmean(candidate[i]-base[i] for i in ids))
    vals.sort(); return vals[125],vals[4875]


def bootstrap_deltas(pairs, seed=20261008):
    """Reuse paired draws without changing the reference bootstrap's bits.

    Every metric starts from the same Random seed. Metrics with the same
    query count can share draw multiplicities. fmean is fsum divided by that
    count: omit zero differences and repeat each remaining difference exactly
    as often as sampled, without multiplying or changing the denominator.
    Retain only one draw's counts, rather than a full query-by-draw matrix.
    """
    grouped = {}
    for index, (candidate, base) in enumerate(pairs):
        if len(candidate) != len(base) or not candidate:
            raise ValueError("paired query arrays required")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in candidate + base):
            raise ValueError("quality arrays must be finite")
        sparse = [(i, a - b) for i, (a, b) in enumerate(zip(candidate, base)) if a != b]
        grouped.setdefault(len(base), []).append((index, sparse))
    result = [None] * len(pairs)
    for count, metrics in grouped.items():
        rng = random.Random(seed)
        draws = [[] for _ in metrics]
        for _ in range(5000):
            multiplicities = Counter(rng.randrange(count) for _ in range(count))
            for values, (_, sparse) in zip(draws, metrics):
                total = math.fsum(itertools.chain.from_iterable(
                    itertools.repeat(value, multiplicities[i]) for i, value in sparse))
                values.append(total / count)
        for (index, _), values in zip(metrics, draws):
            values.sort()
            result[index] = (values[125], values[4875])
    return result

def seed_paths(values:list[str],label:str)->dict[int,Path]:
    result={}
    for value in values:
        seed_text,separator,path=value.partition("=")
        if not separator or not seed_text.isdecimal() or int(seed_text) in result or not path:
            raise ValueError(f"{label} must contain unique SEED=PATH entries")
        result[int(seed_text)]=Path(path)
    if set(result)!=set(QUALITY_SEEDS): raise ValueError(f"{label} requires exact seeds {QUALITY_SEEDS}")
    return result

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--candidate-checkpoint",action="append",default=[],metavar="SEED=PATH");p.add_argument("--candidate-model",action="append",default=[],metavar="SEED=PATH");p.add_argument("--base-weight",type=Path,required=True);p.add_argument("--prepared-manifest",type=Path,required=True);p.add_argument("--training-source-manifest",type=Path,required=True);p.add_argument("--public-source-manifest",type=Path,required=True);p.add_argument("--target-min-delta",type=float,default=0.0);p.add_argument("--retention-min-delta",type=float,default=-0.01);p.add_argument("--report",type=Path,required=True);a=p.parse_args(); data=load_json(a.input)
    if data.get("schema")!="antfly.embedding_gemma2.finetune_quality_input.v1": raise ValueError("unsupported quality input schema")
    if not math.isfinite(a.target_min_delta) or not math.isfinite(a.retention_min_delta): raise ValueError("quality thresholds must be finite")
    if a.target_min_delta < 0 or a.retention_min_delta < -0.01: raise ValueError("release quality thresholds cannot be weakened")
    if tuple(data.get("seeds",()))!=QUALITY_SEEDS: raise ValueError("quality promotion requires exact three-seed contract")
    if data.get("qualification_scope")!="official-full-corpus": raise ValueError("bounded fixtures are not quality-promotion evidence")
    provenance=data.get("provenance",{})
    if any(not valid_sha256(provenance.get(key)) for key in REQUIRED_PROVENANCE): raise ValueError("quality report provenance is incomplete")
    checkpoints=seed_paths(a.candidate_checkpoint,"candidate checkpoints")
    models=seed_paths(a.candidate_model,"candidate models")
    candidates=provenance.get("candidates_by_seed")
    if not isinstance(candidates,dict) or set(candidates)!=set(map(str,QUALITY_SEEDS)):
        raise ValueError("quality provenance requires exact per-seed candidate artifacts")
    for seed in QUALITY_SEEDS:
        expected=candidates[str(seed)]
        actual=validate_candidate_artifacts(checkpoint=checkpoints[seed],model=models[seed],
            prepared_manifest=a.prepared_manifest,training_source_manifest=a.training_source_manifest,seed=seed)
        if (not isinstance(expected,dict) or expected!=actual
                or not all(valid_sha256(value) for value in expected.values())):
            raise ValueError(f"seed {seed} candidate artifacts differ from supplied immutable artifacts")
    artifacts={"base_weight_sha256":a.base_weight,"prepared_manifest_sha256":a.prepared_manifest,
               "training_source_manifest_sha256":a.training_source_manifest,
               "public_source_manifest_sha256":a.public_source_manifest}
    if any(file_sha256(path)!=provenance[key] for key,path in artifacts.items()): raise ValueError("quality provenance differs from supplied immutable artifacts")
    if provenance["base_weight_sha256"]!=BASE_WEIGHT_SHA256: raise ValueError("quality report does not use the pinned base weight")
    training_source_sha=provenance["training_source_manifest_sha256"]
    prepared=load_json(a.prepared_manifest)
    if prepared.get("source_manifest_sha256")!=training_source_sha:
        raise ValueError("prepared training data is not bound to the supplied training-source manifest")
    source_manifest=load_json(a.public_source_manifest)
    inventory_by_key=validate_official_source_manifest(source_manifest)
    validate_reviewed_inventory(inventory_by_key)
    if not isinstance(provenance.get("public_sources"),list) or len(provenance["public_sources"])<5: raise ValueError("official public-source provenance is incomplete")
    public_sources=set()
    for source in provenance["public_sources"]:
        if (not isinstance(source,dict) or not isinstance(source.get("source_repo"),str)
                or not valid_sha256(source.get("snapshot_sha256"))
                or not isinstance(source.get("config"),str) or not source["config"]
                or not isinstance(source.get("license"),str) or not source["license"]
                or not isinstance(source.get("source_revision"),str)
                or len(source["source_revision"]) not in (40,64)
                or any(c not in "0123456789abcdef" for c in source["source_revision"])):
            raise ValueError("official public-source entry is incomplete")
        public_sources.add((source["source_repo"],source["source_revision"],source["config"]))
    contract=data.get("cell_contract",[])
    contract_keys=[]
    contract_queries={}
    for cell in contract:
        if cell.get("role") not in ("target","retention"): raise ValueError("cell contract requires explicit target or retention role")
        key=(cell.get("dataset"),cell.get("language"),cell["role"])
        if not key[0] or key in contract_keys: raise ValueError("duplicate or invalid quality cell contract")
        corpus=cell.get("official_corpus",{})
        if cell.get("fixture_scope")!="official-full-corpus": raise ValueError("quality cell lacks official corpus metadata")
        expected_source=OFFICIAL_SOURCE_CONTRACTS.get(key)
        if expected_source is None or any(corpus.get(name)!=value for name,value in expected_source.items()): raise ValueError("quality cell differs from the reviewed official source contract")
        if corpus!=inventory_by_key[key]: raise ValueError("quality cell differs from the authenticated official query inventory")
        required_sources={(corpus["source_repo"],corpus["source_revision"],corpus["corpus_config"]),
                          (corpus["source_repo"],corpus["source_revision"],corpus["queries_config"]),
                          (corpus["qrels_source_repo"],corpus["qrels_source_revision"],corpus["qrels_config"])}
        if not required_sources.issubset(public_sources): raise ValueError("official corpus source is absent from public-source provenance")
        contract_keys.append(key)
        contract_queries[key]=corpus["query_count"]
    if set(contract_keys)!=REQUIRED_CELLS: raise ValueError("quality promotion requires exact MSMARCO, CodeSearchNet Python/Go, SciFact and NFCorpus cells")
    rows=[]
    metric_pairs=[]
    seen=set()
    paired_ids={}
    for cell in data.get("cells",[]):
        if cell.get("dimension") not in MRL_DIMENSIONS: raise ValueError("missing/invalid MRL dimension")
        if cell.get("seed") not in QUALITY_SEEDS: raise ValueError("missing/invalid promotion seed")
        if cell.get("role") not in ("target","retention"): raise ValueError("quality cell has invalid role")
        key=(cell.get("dataset"),cell.get("language"),cell["role"],cell["dimension"],cell["seed"])
        if key in seen: raise ValueError("duplicate quality cell")
        seen.add(key)
        candidate_identity=candidates[str(cell["seed"])]
        if (cell.get("candidate_checkpoint_sha256")!=candidate_identity["candidate_checkpoint_sha256"]
                or cell.get("candidate_model_sha256")!=candidate_identity["candidate_model_sha256"]
                or cell.get("candidate_bundle_manifest_sha256")!=candidate_identity["candidate_bundle_manifest_sha256"]):
            raise ValueError("quality cell candidate identity differs from its declared seed artifacts")
        candidate_ids=cell.get("candidate_query_ids");base_ids=cell.get("base_query_ids")
        if (not isinstance(candidate_ids,list) or candidate_ids!=base_ids
                or not all(isinstance(value,str) and value for value in candidate_ids)
                or len(candidate_ids)!=len(set(candidate_ids)) or not candidate_ids): raise ValueError("base/candidate query IDs must be paired, unique nonempty strings")
        contract_key=(cell.get("dataset"),cell.get("language"),cell["role"])
        if len(candidate_ids)!=contract_queries[contract_key]: raise ValueError("query IDs do not cover the declared official query corpus")
        canonical_ids=paired_ids.setdefault(contract_key,candidate_ids)
        if candidate_ids!=canonical_ids: raise ValueError("query ID order/set differs across seeds or MRL dimensions")
        for metric in ("mrr_at_10","ndcg_at_10","recall_at_100"):
            candidate=cell["candidate_per_query"][metric];base=cell["base_per_query"][metric]
            if (not isinstance(candidate,list) or not isinstance(base,list)
                    or len(candidate)!=len(candidate_ids) or len(base)!=len(base_ids)
                    or any(isinstance(value,bool) or not isinstance(value,(int,float))
                           or not math.isfinite(value) or not 0<=value<=1 for value in candidate+base)):
                raise ValueError("per-query metrics must align with IDs and remain finite in [0,1]")
            threshold=a.target_min_delta if cell["role"]=="target" else a.retention_min_delta
            metric_pairs.append((candidate,base))
            rows.append({"dataset":cell["dataset"],"language":cell.get("language"),"dimension":cell["dimension"],"seed":cell["seed"],"role":cell["role"],"metric":metric,"threshold":threshold})
    for row,(lo,hi) in zip(rows,bootstrap_deltas(metric_pairs)):
        row.update({"delta_ci95":[lo,hi],"pass":lo>=row["threshold"]})
    expected={(dataset,language,role,d,s) for dataset,language,role in contract_keys for d in MRL_DIMENSIONS for s in QUALITY_SEEDS}
    complete=seen==expected
    report={"schema":"antfly.embedding_gemma2.finetune_quality.v1","pass":complete and bool(rows) and all(r["pass"] for r in rows),"complete":complete,"provenance":provenance,"rows":rows}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"pass":report["pass"],"rows":len(rows)}));return 0 if report["pass"] else 1
if __name__=="__main__":raise SystemExit(main())
