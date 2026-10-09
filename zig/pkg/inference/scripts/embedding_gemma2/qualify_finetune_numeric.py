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

"""Compare native and PyTorch LoRA step traces, including resume equivalence."""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from finetune_contract import BASE_WEIGHT_SHA256,canonical_hash,file_sha256,load_json,valid_sha256

def reviewed_targets()->set[str]:
    suffixes=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj")
    return {*(f"language_model.layers.{layer}.{suffix}" for layer in range(24) for suffix in suffixes),"language_model.embedding_projection"}

def adapter_target(name:str)->str:
    value=name
    for prefix in ("base_model.model.model.","base_model.model.","model."):
        if value.startswith(prefix): value=value[len(prefix):];break
    for suffix in (".lora_A.weight",".lora_B.weight"):
        if value.endswith(suffix): return value[:-len(suffix)]
    raise ValueError(f"invalid canonical adapter tensor name {name}")

def validate_bf16_provenance(native:dict,ref:dict)->None:
    contract_fields=("logical_batch_size","microbatch_size","max_negatives","rank","alpha","seed",
                     "base_weight_sha256","prepared_manifest_sha256","preprocessing_sha256","padding_policy",
                     "compute_profile","gradient_trace_stage")
    for field in contract_fields:
        left=native["provenance"].get(field);right=ref["provenance"].get(field)
        if left is None or left!=right: raise ValueError(f"BF16 native/reference provenance differs for {field}")
    for field in ("base_weight_sha256","prepared_manifest_sha256","preprocessing_sha256"):
        if not valid_sha256(native["provenance"][field]): raise ValueError(f"BF16 {field} is not a lowercase SHA256 identity")
    if native["provenance"]["base_weight_sha256"]!=BASE_WEIGHT_SHA256: raise ValueError("BF16 trace does not use the pinned base weight")
    if native["provenance"]["padding_policy"]!="per-micro-active-v1": raise ValueError("BF16 qualification requires per-microbatch active padding")
    if native["provenance"]["compute_profile"]!="bf16-edges-fp32-masters-v1": raise ValueError("BF16 qualification requires the reviewed compute profile")
    if native["provenance"]["gradient_trace_stage"]!="pre-clip": raise ValueError("BF16 qualification requires pre-clip gradient traces")
    if ref["provenance"].get("attention")!="eager" or ref["provenance"].get("loaded_attention")!="eager": raise ValueError("BF16 numeric qualification requires the explicit loaded eager attention oracle")
    if native["provenance"].get("attention")!="eager": raise ValueError("native numeric trace must explicitly identify eager attention")
    reference_targets={target.removeprefix("model.") for target in ref["provenance"].get("targets",[])}
    if reference_targets!=reviewed_targets(): raise ValueError("reference BF16 trace lacks the exact reviewed 169-target inventory")

def validate_resume(report:dict,label:str)->None:
    if report.get("schema")=="antfly.embedding_gemma2.native_training_smoke.v1":
        resume=report.get("resume",{})
        if (report.get("pass") is not True or resume.get("pass") is not True
                or resume.get("provenance_equal") is not True
                or not valid_sha256(resume.get("next_step_state_sha256"))
                or not valid_sha256(report.get("binary_sha256"))
                or not valid_sha256(report.get("config_sha256"))):
            raise ValueError(f"{label} native resume smoke lacks authenticated equivalence evidence")
        left=load_json(Path(resume.get("uninterrupted_summary","")));right=load_json(Path(resume.get("resumed_summary","")))
        left_state=left.get("resume",{}).get("next_step_state_sha256");right_state=right.get("resume",{}).get("next_step_state_sha256")
        if left.get("provenance")!=right.get("provenance") or not valid_sha256(left_state) or left_state!=right_state or left_state!=resume["next_step_state_sha256"]:
            raise ValueError(f"{label} native resume summaries do not prove equal state and provenance")
        return
    if report.get("schema")!="antfly.embedding_gemma2.finetune_resume.v1":
        raise ValueError(f"{label} resume report has unsupported schema")
    left=report.get("uninterrupted_next_step_sha256");right=report.get("resumed_next_step_sha256")
    provenance_proven = report.get("provenance_equal") is True
    summary_paths=(report.get("uninterrupted_summary"),report.get("resumed_summary"))
    if any(path is not None for path in summary_paths):
        if not all(isinstance(path,str) and path for path in summary_paths):
            raise ValueError(f"{label} resume summary evidence is incomplete")
        summaries=[load_json(Path(path)) for path in summary_paths]
        expected_hashes=(report.get("uninterrupted_summary_sha256"),report.get("resumed_summary_sha256"))
        if (not all(valid_sha256(value) for value in expected_hashes)
                or any(file_sha256(Path(path))!=expected for path,expected in zip(summary_paths,expected_hashes))):
            raise ValueError(f"{label} resume summary evidence digest differs")
        summary_states=[summary.get("resume",{}).get("next_step_state_sha256") for summary in summaries]
        if summary_states != [left,right] or summaries[0].get("provenance")!=summaries[1].get("provenance"):
            raise ValueError(f"{label} resume summaries do not prove equal state and provenance")
        if (not valid_sha256(report.get("provenance_sha256"))
                or canonical_hash(summaries[0]["provenance"])!=report["provenance_sha256"]):
            raise ValueError(f"{label} resume provenance digest differs")
        provenance_proven=True
    if report.get("trajectory_identity_reused") is True:
        reused=load_json(Path(report.get("reused_resume_proof","")))
        validate_resume(reused,label+" reused")
        provenance_proven=True
        if left!=reused.get("uninterrupted_next_step_sha256") or right!=reused.get("resumed_next_step_sha256"):
            raise ValueError(f"{label} reused resume state differs")
    if report.get("pass") is not True or not provenance_proven or not valid_sha256(left) or left!=right:
        raise ValueError(f"{label} resume report lacks equal authenticated state identities")

def tensor_values(value):
    if isinstance(value,list):
        def flatten(node):
            if not isinstance(node,list): return (),[node]
            if not node: raise ValueError("empty tensor")
            children=[flatten(child) for child in node]
            if any(shape!=children[0][0] for shape,_ in children): raise ValueError("ragged tensor")
            return (len(node),*children[0][0]),[item for _,items in children for item in items]
        shape,values=flatten(value);return values,shape
    if not isinstance(value,dict) or set(value)!={"file","tensor"}: raise ValueError("invalid tensor reference")
    from safetensors import safe_open
    with safe_open(value["file"],framework="numpy") as tensors:
        tensor=tensors.get_tensor(value["tensor"]);return tensor.reshape(-1).astype("float64").tolist(),tuple(tensor.shape)

def compare_array(a:list[float], b:list[float], atol:float, rtol:float)->dict:
    if len(a)!=len(b) or not a: raise ValueError("tensor shape/length mismatch")
    if not all(math.isfinite(x) for x in a+b): raise ValueError("nonfinite tensor")
    dot=sum(x*y for x,y in zip(a,b)); aa=sum(x*x for x in a); bb=sum(y*y for y in b)
    cosine=1.0 if aa==bb==0 else (0.0 if aa==0 or bb==0 else dot/math.sqrt(aa*bb))
    diff_sq=sum((x-y)*(x-y) for x,y in zip(a,b))
    ref_l2=math.sqrt(bb); candidate_l2=math.sqrt(aa)
    rel_l2=0.0 if diff_sq==0 else (math.inf if ref_l2==0 else math.sqrt(diff_sq)/ref_l2)
    max_abs=max(abs(x-y) for x,y in zip(a,b)); max_scaled=max(abs(x-y)/(atol+rtol*max(abs(x),abs(y))) for x,y in zip(a,b))
    return {"cosine":cosine,"max_abs":max_abs,"max_scaled_error":max_scaled,
            "reference_l2":ref_l2,"candidate_l2":candidate_l2,"relative_l2":rel_l2,
            "_dot":dot,"_candidate_sq":aa,"_reference_sq":bb,"_diff_sq":diff_sq,
            "pass":max_scaled<=1.0}

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--native",type=Path,required=True); p.add_argument("--reference",type=Path,required=True); p.add_argument("--native-resume",type=Path,required=True); p.add_argument("--reference-resume",type=Path,required=True); p.add_argument("--precision",choices=("fp32","bf16"),required=True); p.add_argument("--report",type=Path,required=True); a=p.parse_args()
    native,ref=load_json(a.native),load_json(a.reference)
    if native.get("schema")!="antfly.embedding_gemma2.finetune_native.v1" or ref.get("schema")!="antfly.embedding_gemma2.finetune_reference.v1": raise ValueError("numeric trace schema is unsupported")
    minimum_steps=1 if a.precision=="fp32" else 2
    native_steps,reference_steps=native.get("steps",[]),ref.get("steps",[])
    if len(native_steps)!=len(reference_steps) or not minimum_steps<=len(native_steps)<=10: raise ValueError("trace step counts must match and remain within the qualified bound")
    for label,value in (("native",native),("reference",ref)):
        if value.get("provenance",{}).get("precision")!=a.precision: raise ValueError(f"{label} precision provenance mismatch")
    if a.precision=="bf16":
        validate_bf16_provenance(native,ref)
    atol,rtol=((2e-5,2e-4) if a.precision=="fp32" else (3e-3,2e-2))
    active_norm=1e-7
    tensor_cosine_min,tensor_rel_l2_max,global_cosine_min,global_rel_l2_max=(
        (0.995,0.02,0.9999,0.005) if a.precision=="fp32" else (0.98,0.10,0.999,0.02)
    )
    rows=[]; gradient_summaries=[]
    for step,(ns,rs) in enumerate(zip(native_steps,reference_steps)):
        if ns["step"]!=rs["step"] or ns["step"]!=step: raise ValueError("step mismatch")
        for field in ("embedding","gradients","adapter_parameters"):
            nk,rk=ns[field],rs[field]
            if set(nk)!=set(rk): raise ValueError(f"{field} tensor names mismatch")
            if a.precision=="bf16" and field in ("gradients","adapter_parameters"):
                names=set(nk)
                if len(names)!=338 or sum(".lora_A.weight" in name for name in names)!=169 or sum(".lora_B.weight" in name for name in names)!=169:
                    raise ValueError(f"{field} lacks the exact official 338-tensor LoRA inventory")
                if {adapter_target(name) for name in names}!=reviewed_targets(): raise ValueError(f"{field} adapter names differ from the reviewed 169-target inventory")
            gradient_totals={"dot":0.0,"candidate_sq":0.0,"reference_sq":0.0,"diff_sq":0.0}
            active=[]; inactive=[]
            for name in sorted(nk):
                nv,nshape=tensor_values(nk[name]);rv,rshape=tensor_values(rk[name])
                if nshape!=rshape: raise ValueError(f"{field}.{name} shape mismatch")
                metrics=compare_array(nv,rv,atol,rtol)
                if field=="gradients":
                    for key,source in (("dot","_dot"),("candidate_sq","_candidate_sq"),("reference_sq","_reference_sq"),("diff_sq","_diff_sq")):
                        gradient_totals[key]+=metrics[source]
                    metrics["active_reference"]=metrics["reference_l2"]>=active_norm
                    if metrics["active_reference"]:
                        metrics["pass"]=metrics["cosine"]>=tensor_cosine_min and metrics["relative_l2"]<=tensor_rel_l2_max
                        active.append((name,metrics))
                    else:
                        metrics["pass"]=metrics["max_abs"]<=atol
                        inactive.append((name,metrics))
                for key in ("_dot","_candidate_sq","_reference_sq","_diff_sq"): metrics.pop(key)
                rows.append({"step":step,"field":field,"name":name,"shape":nshape,**metrics})
            if field=="gradients":
                ref_sq=gradient_totals["reference_sq"]; candidate_sq=gradient_totals["candidate_sq"]
                global_cosine=1.0 if ref_sq==candidate_sq==0 else (0.0 if ref_sq==0 or candidate_sq==0 else gradient_totals["dot"]/math.sqrt(ref_sq*candidate_sq))
                global_relative_l2=0.0 if gradient_totals["diff_sq"]==0 else (math.inf if ref_sq==0 else math.sqrt(gradient_totals["diff_sq"]/ref_sq))
                family_counts={family:sum(f".lora_{family}.weight" in name for name,_ in active) for family in ("A","B")}
                expected_activity=(family_counts["A"]==0 and family_counts["B"]>0) if step==0 else (family_counts["A"]>0 and family_counts["B"]>0)
                summary={"step":step,"active_reference_tensors":len(active),"inactive_reference_tensors":len(inactive),
                         "active_lora_A_tensors":family_counts["A"],"active_lora_B_tensors":family_counts["B"],
                         "expected_activity":expected_activity,"global_cosine":global_cosine,"global_relative_l2":global_relative_l2,
                         "minimum_active_tensor_cosine":min((m["cosine"] for _,m in active),default=1.0),
                         "maximum_active_tensor_relative_l2":max((m["relative_l2"] for _,m in active),default=0.0),
                         "minimum_active_reference_l2":min((m["reference_l2"] for _,m in active),default=0.0),
                         "maximum_active_reference_l2":max((m["reference_l2"] for _,m in active),default=0.0),
                         "worst_cosine_tensor":min(active,key=lambda item:item[1]["cosine"])[0] if active else None,
                         "worst_relative_l2_tensor":max(active,key=lambda item:item[1]["relative_l2"])[0] if active else None}
                summary["pass"]=expected_activity and global_cosine>=global_cosine_min and global_relative_l2<=global_rel_l2_max and all(m["pass"] for _,m in active+inactive)
                gradient_summaries.append(summary)
        nl,rl=float(ns["loss"]),float(rs["loss"])
        if not math.isfinite(nl) or not math.isfinite(rl): raise ValueError("nonfinite loss")
        loss_error=abs(nl-rl); rows.append({"step":step,"field":"loss","name":"loss","max_abs":loss_error,"pass":loss_error<=atol+rtol*abs(rl)})
    # Optimizer serialization differs across implementations. Each side proves
    # interrupted-versus-uninterrupted equivalence independently.
    native_resume,reference_resume=load_json(a.native_resume),load_json(a.reference_resume)
    validate_resume(native_resume,"native");validate_resume(reference_resume,"reference")
    resume=True
    report={"schema":"antfly.embedding_gemma2.finetune_numeric.v1","precision":a.precision,
            "gradient_gates":{"active_reference_l2_min":active_norm,"per_tensor_cosine_min":tensor_cosine_min,
                              "per_tensor_relative_l2_max":tensor_rel_l2_max,"global_cosine_min":global_cosine_min,
                              "global_relative_l2_max":global_rel_l2_max},
            "pass":all(r["pass"] for r in rows) and all(s["pass"] for s in gradient_summaries) and resume,
            "resume_equivalent":resume,"gradient_summaries":gradient_summaries,"rows":rows}
    a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps({"pass":report["pass"],"rows":len(rows)})); return 0 if report["pass"] else 1
if __name__=="__main__": raise SystemExit(main())
