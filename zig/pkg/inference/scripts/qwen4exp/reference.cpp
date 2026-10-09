// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Validation only. Pinned llama.cpp graph; bounded FP32 activation adapter.
#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#include <string>
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
// Reference-only precision adapter: preserve F32 activations. Stock ggml CPU
// matvec quantizes activations before its dot products; Antfly Metal does not.
// Dequantize at most 64 weight rows with upstream GGUF routines, then BLAS.
bool precise_weight_mm(ggml_tensor *t) {
    if(t->op!=GGML_OP_MUL_MAT && t->op!=GGML_OP_MUL_MAT_ID)return false;
    auto *w=t->src[0];auto *x=t->src[1];
    return w->type!=GGML_TYPE_F32 && ggml_get_type_traits(w->type)->to_float && x->type==GGML_TYPE_F32;
}
void recompute_mm(ggml_tensor *t) {
    auto *w=t->src[0];auto *x=t->src[1]; const auto dequant=ggml_get_type_traits(w->type)->to_float;
    const int k=w->ne[0],n=w->ne[1];
    std::vector<float> block(64*k),input(k),output(n);
    const int routes=t->op==GGML_OP_MUL_MAT_ID ? t->ne[1] : 1;
    if(t->op==GGML_OP_MUL_MAT && (x->ne[1]!=1 || w->ne[2]!=1 || w->ne[3]!=1)) {fprintf(stderr,"unexpected precision shape %s\n",t->name);abort();}
    for(int route=0;route<routes;++route) {
        int expert=t->op==GGML_OP_MUL_MAT_ID ? ((int32_t*)t->src[2]->data)[route] : 0;
        const char *xp=(const char*)x->data+(route%x->ne[1])*x->nb[1];
        for(int j=0;j<k;++j)memcpy(&input[j],xp+j*x->nb[0],4);
        for(int start=0;start<n;start+=64) {
            int rows=std::min(64,n-start);
            for(int r=0;r<rows;++r)dequant((const char*)w->data+expert*w->nb[2]+(start+r)*w->nb[1],block.data()+r*k,k);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,rows,k,1,block.data(),k,input.data(),1,0,output.data()+start,1);
        }
        ggml_backend_tensor_set(t,output.data(),route*t->nb[1],n*4);
    }
}
bool capture(ggml_tensor *t,bool ask,void*) {
    std::string name=t->name;
    bool wanted=(name.find("hc_")!=std::string::npos || name.find("linear_attn_")!=std::string::npos || name.find("final_output")!=std::string::npos || name.find("ffn_moe_out")!=std::string::npos || name.find("ffn_moe_topk")!=std::string::npos || name.find("ffn_moe_logits")!=std::string::npos || name.find("l_last")!=std::string::npos || name.find("model.input_embed")!=std::string::npos || name.find("ple_")!=std::string::npos || name.find("result_norm")!=std::string::npos);
    if(ask)return precise_weight_mm(t) || (wanted && std::getenv("QWEN4EXP_REFERENCE_CAPTURE_DIR") && (t->type==GGML_TYPE_F32 || t->type==GGML_TYPE_I32) && ggml_is_contiguous(t));
    if(precise_weight_mm(t))recompute_mm(t);
    if(!wanted || !std::getenv("QWEN4EXP_REFERENCE_CAPTURE_DIR"))return true;
    std::vector<unsigned char> bytes(ggml_nbytes(t));ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    const char *dir=std::getenv("QWEN4EXP_REFERENCE_CAPTURE_DIR");
    if(dir){FILE *f=fopen((std::string(dir)+"/"+name+".f32").c_str(),"wb");if(f){fwrite(bytes.data(),1,bytes.size(),f);fclose(f);}}return true;
}
int main(int argc,char **argv) {
    if(argc<4) return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();
    ggml_backend_load_all();
    mp.tensor_buft_overrides=nullptr;
    ggml_backend_dev_t devices[]={nullptr};
    mp.devices=devices;
    mp.n_gpu_layers=0; mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode=LLAMA_LAZY_MODE_ON; mp.use_extra_bufts=false; mp.load_mtp=false;
    auto *model=llama_model_load_from_file(argv[1],mp); if(!model)return 3;
    auto cp=llama_context_default_params();
    cp.n_ctx=4096; cp.n_batch=1; cp.n_ubatch=1; cp.n_threads=4; cp.n_threads_batch=4;
    // The flash graph casts F32 K/V to F16; compare matching FP32 attention.
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval=capture;
    cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.offload_kqv=false;cp.op_offload=false;cp.moe_cache_size=0;
    auto *ctx=llama_init_from_model(model,cp); if(!ctx)return 4;
    auto *vocab=llama_model_get_vocab(model);const int n=llama_vocab_n_tokens(vocab);
    FILE *out=fopen(argv[2],"wb"); if(!out)return 5;
    for(int i=3;i<argc;++i) {
        llama_token token=atoi(argv[i]);
        auto batch=llama_batch_get_one(&token,1);
        if(llama_decode(ctx,batch)!=0)return 6;
        const float *logits=llama_get_logits_ith(ctx,-1);
        int best=0;for(int j=0;j<n;++j) {if(!std::isfinite(logits[j]))return 7;if(logits[j]>logits[best])best=j;}
        if(fwrite(logits,sizeof(float),n,out)!=(size_t)n)return 8;
        fprintf(stderr,"REFERENCE position=%d next=%d vocab=%d\n",i-3,best,n);
    }
    fclose(out);llama_free(ctx);llama_model_free(model);llama_backend_free();
}
