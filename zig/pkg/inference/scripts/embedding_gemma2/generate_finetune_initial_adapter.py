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

"""Generate the exact native SplitMix64 LoRA initialization as PEFT tensors."""

from __future__ import annotations
import argparse,hashlib,json,math,struct
from pathlib import Path

BASE_SHA="197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
SUFFIXES=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj")

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def splitmix(state):
    mask=(1<<64)-1;state=(state+0x9E3779B97F4A7C15)&mask;z=state
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&mask;z=((z^(z>>27))*0x94D049BB133111EB)&mask
    return state,(z^(z>>31))&mask

def header(path):
    with path.open("rb") as f:n=struct.unpack("<Q",f.read(8))[0];return json.loads(f.read(n))

def main():
    p=argparse.ArgumentParser();p.add_argument("--model",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--rank",type=int,choices=(8,16,32),default=16);p.add_argument("--alpha",type=float,default=32);p.add_argument("--seed",type=int,default=42);a=p.parse_args()
    if not 0 <= a.seed < 1 << 64: p.error("seed must be in [0,2^64)")
    if not math.isfinite(a.alpha) or not 0 < a.alpha <= 1024: p.error("alpha must be finite and in (0,1024]")
    weight=a.model/"model.safetensors"
    if sha(weight)!=BASE_SHA:raise ValueError("base model identity mismatch")
    import numpy as np,torch
    from safetensors.torch import save_file
    metadata=header(weight);names=[]
    for layer in range(24):names.extend(f"language_model.layers.{layer}.{suffix}.weight" for suffix in SUFFIXES)
    names.append("language_model.embedding_projection.weight")
    state=a.seed; tensors={}
    for name in names:
        shape=metadata[name]["shape"]
        if len(shape)!=2:raise ValueError(f"nonmatrix target {name}")
        output_dim,input_dim=shape;bound=np.float32(1.0)/np.sqrt(np.float32(input_dim))
        values=np.empty((a.rank,input_dim),dtype=np.float32)
        flat=values.reshape(-1)
        for index in range(flat.size):
            state,bits=splitmix(state);unit=np.float32(bits>>40)*np.float32(1.0/16777216.0)
            flat[index]=np.float32((unit*np.float32(2.0)-np.float32(1.0))*bound)
        peft=f"base_model.model.{name[:-len('.weight')]}"
        tensors[f"{peft}.lora_A.weight"]=torch.from_numpy(values)
        tensors[f"{peft}.lora_B.weight"]=torch.zeros((output_dim,a.rank),dtype=torch.float32)
    a.output_dir.mkdir(parents=True,exist_ok=False)
    adapter=a.output_dir/"adapter_model.safetensors";save_file(tensors,str(adapter),metadata={"format":"pt","antfly.initialization":"splitmix64-v1"})
    config={"base_model_name_or_path":"google/embeddinggemma-2","bias":"none","inference_mode":False,"lora_alpha":a.alpha,"lora_dropout":0.0,"peft_type":"LORA","r":a.rank,"task_type":"FEATURE_EXTRACTION","target_modules":[*SUFFIXES,"embedding_projection"]}
    (a.output_dir/"adapter_config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    report={"schema":"antfly.embedding_gemma2.initial_adapter.v1","base_weight_sha256":BASE_SHA,"seed":a.seed,"rank":a.rank,"alpha":a.alpha,"tensor_count":len(tensors),"adapter_sha256":sha(adapter),"final_splitmix_state":state}
    (a.output_dir/"initialization_report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
