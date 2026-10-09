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

"""Gate paired native/PyTorch training-step samples for seven exact cells."""

from __future__ import annotations
import argparse,json,math
from pathlib import Path
from finetune_contract import BASE_WEIGHT_SHA256,MAX_DEVICE_BYTES,MAX_HOST_BYTES,MAX_P95_RATIO,MIN_THROUGHPUT_RATIO,file_sha256,load_json,paired_ratio_ci,percentile,valid_sha256,validate_performance_cells,validate_performance_workload

def reference_source_digest(root:Path)->str:
    import hashlib
    digest=hashlib.sha256()
    for relative in ("finetune_contract.py","finetune_reference.py","run_finetune_paired.py"):
        digest.update(relative.encode()+b"\0");digest.update(bytes.fromhex(file_sha256(root/relative)))
    return digest.hexdigest()

def validate_report_provenance(n:dict,r:dict,native_binary:Path,native_manifest:Path,reference_source_dir:Path)->None:
    for label,value in (("native",n),("reference",r)):
        if value.get("schema")!="antfly.embedding_gemma2.finetune_samples.v1": raise ValueError(f"{label} sample report schema is unsupported")
        provenance=value.get("provenance",{})
        required={"hardware","software","source_sha256","base_weight_sha256","dataset_sha256","job_fingerprint"}
        missing=required-set(provenance)
        if missing: raise ValueError(f"{label} provenance missing {sorted(missing)}")
        if not isinstance(provenance["hardware"],str) or not provenance["hardware"].strip() or not isinstance(provenance["software"],dict): raise ValueError(f"{label} hardware/software provenance is invalid")
        if not valid_sha256(provenance["source_sha256"]) or provenance["base_weight_sha256"]!=BASE_WEIGHT_SHA256: raise ValueError(f"{label} source/base provenance is invalid")
        for key in ("dataset_sha256","job_fingerprint"):
            if not isinstance(provenance[key],dict) or set(provenance[key])!=set(value.get("cell_contract",[])) or not all(valid_sha256(item) for item in provenance[key].values()): raise ValueError(f"{label} {key} provenance is invalid")
    binary_sha=file_sha256(native_binary);manifest_sha=file_sha256(native_manifest)
    manifest=load_json(native_manifest)
    if (manifest.get("schema")!="antfly.embedding_gemma2.training-source-manifest/v1"
            or not isinstance(manifest.get("files"),list) or not manifest["files"]
            or any(not isinstance(entry,dict) or not isinstance(entry.get("path"),str)
                   or not valid_sha256(entry.get("sha256")) or not isinstance(entry.get("bytes"),int)
                   or entry["bytes"]<0 for entry in manifest["files"])):
        raise ValueError("native source manifest schema or file inventory is invalid")
    if manifest.get("binary_sha256")!=binary_sha or n["provenance"].get("binary_sha256")!=binary_sha or n["provenance"]["source_sha256"]!=manifest_sha: raise ValueError("native binary/source manifest binding differs from measured report")
    if r["provenance"]["source_sha256"]!=reference_source_digest(reference_source_dir): raise ValueError("reference source digest differs from measured report")

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--native",type=Path,required=True);p.add_argument("--reference",type=Path,required=True);p.add_argument("--native-binary",type=Path,required=True);p.add_argument("--native-source-manifest",type=Path,required=True);p.add_argument("--reference-source-dir",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args(); n,r=load_json(a.native),load_json(a.reference)
    validate_report_provenance(n,r,a.native_binary,a.native_source_manifest,a.reference_source_dir)
    for label,value in (("native",n),("reference",r)):
        if value.get("qualification_shape") is not True: raise ValueError(f"{label} samples were not collected with the production qualification shape")
    for key in ("hardware","base_weight_sha256","dataset_sha256","job_fingerprint"):
        if n["provenance"].get(key)!=r["provenance"].get(key): raise ValueError(f"native/reference {key} provenance differs")
    n_contract=n.get("cell_contract");r_contract=r.get("cell_contract")
    if not isinstance(n_contract,list) or n_contract!=r_contract or len(n_contract)!=7 or len(set(n_contract))!=7: raise ValueError("qualification requires one shared exact seven-cell contract")
    if set(n.get("cells",{}))!=set(n_contract) or set(r.get("cells",{}))!=set(n_contract): raise ValueError("cell reports do not exactly match the declared contract")
    validate_performance_cells([n["cells"][cell_id] for cell_id in n_contract])
    rows=[]
    for cell_id in sorted(set(n["cells"])|set(r["cells"])):
        if cell_id not in n["cells"] or cell_id not in r["cells"]: raise ValueError("cell set mismatch")
        nc,rc=n["cells"][cell_id],r["cells"][cell_id]; nt,rt=nc["step_ms"],rc["step_ms"]
        contract_keys=("batch_size","microbatch_size","query_tokens","document_tokens","negatives","rank","alpha",
                       "dataset_sha256","job_fingerprint","initial_adapter_sha256","pairing","warmup_steps")
        contract_keys=(*contract_keys,"native_attention","reference_attention","padding_policy","compute_profile","active_query_tokens","active_document_tokens","observed_query_counts","observed_candidate_counts")
        if any(nc.get(key)!=rc.get(key) for key in contract_keys) or any(nc.get(key) is None for key in contract_keys): raise ValueError(f"{cell_id} native/reference cell contracts differ")
        for key in ("dataset_sha256","job_fingerprint","initial_adapter_sha256"):
            if not valid_sha256(nc[key]): raise ValueError(f"{cell_id} {key} is not a lowercase SHA256 identity")
        if n["provenance"]["dataset_sha256"][cell_id]!=nc["dataset_sha256"] or n["provenance"]["job_fingerprint"][cell_id]!=nc["job_fingerprint"]: raise ValueError(f"{cell_id} cell/provenance identity differs")
        if nc["pairing"]!="resident-alternating-two-barriers-v1" or not isinstance(nc["warmup_steps"],int) or nc["warmup_steps"]<1: raise ValueError("timings require resident alternating two-barrier collection and warmup")
        if nc["padding_policy"]!="per-micro-active-v1" or nc["compute_profile"]!="bf16-edges-fp32-masters-v1": raise ValueError("timings require the reviewed padding and compute profiles")
        if nc["native_attention"]!="eager" or nc["reference_attention"]!="sdpa": raise ValueError("production timing requires native eager against the explicit SDPA reference")
        expected_observations=100
        validate_performance_workload(nc,expected_observations)
        if nc.get("trace_mode") or rc.get("trace_mode"): raise ValueError("trace-mode timings are not performance-qualified")
        if len(nt)!=100 or len(rt)!=100: raise ValueError("qualification requires exactly 100 paired steps")
        if not all(isinstance(value,(int,float)) and math.isfinite(value) and value>0 for value in [*nt,*rt]): raise ValueError("step timings must be finite and positive")
        for measured in (nc,rc):
            if measured.get("memory_measurement")!="external_peak": raise ValueError("memory must be measured externally, not statically estimated")
            if not all(isinstance(measured.get(key),(int,float)) and math.isfinite(measured[key]) and measured[key]>0 for key in ("peak_device_bytes","peak_host_bytes")): raise ValueError("memory peaks must be finite and positive")
        lo,hi=paired_ratio_ci(nt,rt);p95=percentile(nt,.95)/percentile(rt,.95)
        memory=nc["peak_device_bytes"]<=MAX_DEVICE_BYTES and nc["peak_host_bytes"]<=MAX_HOST_BYTES
        rows.append({"cell":cell_id,"samples":len(nt),"throughput_ratio_ci95":[lo,hi],"p95_latency_ratio":p95,"peak_device_bytes":nc["peak_device_bytes"],"peak_host_bytes":nc["peak_host_bytes"],"pass":lo>=MIN_THROUGHPUT_RATIO and p95<=MAX_P95_RATIO and memory})
    report={"schema":"antfly.embedding_gemma2.finetune_paired.v1","pass":len(rows)==7 and all(x["pass"] for x in rows),"gates":{"throughput_ratio_lower_95":MIN_THROUGHPUT_RATIO,"p95_latency_ratio_max":MAX_P95_RATIO,"device_bytes_max":MAX_DEVICE_BYTES,"host_bytes_max":MAX_HOST_BYTES},"native_provenance":n.get("provenance"),"reference_provenance":r.get("provenance"),"rows":rows}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"pass":report["pass"],"cells":len(rows)}));return 0 if report["pass"] else 1
if __name__=="__main__":raise SystemExit(main())
