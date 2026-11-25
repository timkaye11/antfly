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

// native-program:entry antfly_paged_attention_prefill_flash_hd512_twohead_register_v1
// native-program:indent 11
#include <metal_stdlib>
using namespace metal;
struct Params { uint q_len,kv_tokens,num_heads,num_kv_heads,head_dim,key_row_bytes,base_key_row_bytes,query_position_offset,kv_position_offset,sliding_window,v_row_stride,page_size,block_count,contiguous_base_token,contiguous_blocks,format,v_element_bytes,has_sinks; float softcap; uint swa_scan_clamp,has_explicit_positions; };
inline uint page_token(device const uint*bt,constant Params&p,uint t){if(p.contiguous_blocks)return p.contiguous_base_token+t;uint b=t/p.page_size,r=t-b*p.page_size;return b<p.block_count?bt[b]+r:0xffffffffu;}
kernel void antfly_paged_attention_prefill_flash_hd512_twohead_register_v1(device const float*q[[buffer(0)]],device const uchar*kb[[buffer(1)]],device const uchar*vb[[buffer(2)]],device const uint*bt[[buffer(3)]],device const float*sinks[[buffer(4)]],device float*out[[buffer(5)]],constant Params&p[[buffer(6)]],threadgroup char*sm[[threadgroup(0)]],ushort tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],uint2 tg[[threadgroup_position_in_grid]]){
 const uint q0=tg.x*8u,h0=tg.y*2u,qs=8u*512u,kr=p.key_row_bytes/2u,ds=uint(sg)*64u,g=p.num_heads/p.num_kv_heads,kvh=(h0/g)*512u;const float scale=rsqrt(512.0f);
 if(q0>=p.q_len||h0+1u>=8u||p.num_heads!=8u||(p.num_kv_heads!=1u&&p.num_kv_heads!=2u)||p.head_dim!=512u||p.sliding_window!=0u||p.format!=3u||p.page_size!=16u||p.has_sinks||p.softcap!=0.0f||p.has_explicit_positions||g<2u||(h0/g)!=((h0+1u)/g))return;
 threadgroup half*sq=(threadgroup half*)sm;                       // 16384 [2][8][512]
 threadgroup float*ss=(threadgroup float*)(sm+16384u);           // 4096 [2][8][64]
 threadgroup half*sp=(threadgroup half*)(sm+20480u);             // 2048 [2][8][64]
 threadgroup uint*phys=(threadgroup uint*)(sm+22528u);            // 256 [64]
 threadgroup float*M=(threadgroup float*)(sm+22784u),*S=M+16u,*diag=S+16u; // 512 diag
 threadgroup float*tail=diag+128u;                               // 2048 shared per-head tail
 const device half*kh=(const device half*)kb,*vh=(const device half*)vb;
 for(uint i=uint(tid);i<8192u;i+=256u){uint hi=i/4096u,j=(i/512u)&7u,d=i&511u,qi=q0+j;sq[i]=qi<p.q_len?half(q[qi*qs+(h0+hi)*512u+d]*scale):half(0);}
 if(tid<16u){M[tid]=-3.402823466e38f;S[tid]=0;}if(tid<128u)diag[tid]=0;
 simdgroup_float8x8 O0[8],O1[8];for(uint z=0;z<8u;z++){O0[z]=make_filled_simdgroup_matrix<float,8>(0);O1[z]=make_filled_simdgroup_matrix<float,8>(0);}
 for(uint kc=0;kc<p.kv_tokens;kc+=64u){
  threadgroup_barrier(mem_flags::mem_threadgroup);uint qlast=min(q0+7u,p.q_len-1u),last=p.query_position_offset+qlast,keyfirst=p.kv_position_offset+kc;if(keyfirst>last)break;
  if(tid<64u){uint ki=kc+uint(tid);phys[tid]=ki<p.kv_tokens?page_token(bt,p,ki):0xffffffffu;}threadgroup_barrier(mem_flags::mem_threadgroup);
  uint kt=uint(sg),ph=phys[kt*8u];const device half*kbase=kh+(ph!=0xffffffffu?ph:0u)*kr+kvh;simdgroup_float8x8 A0=make_filled_simdgroup_matrix<float,8>(0),A1=make_filled_simdgroup_matrix<float,8>(0);
  for(uint d=0;d<512u;d+=8u){simdgroup_half8x8 Q0,Q1,K;simdgroup_load(Q0,sq+d,512u);simdgroup_load(Q1,sq+4096u+d,512u);simdgroup_load(K,kbase+d,kr,0,true);simdgroup_multiply_accumulate(A0,Q0,K,A0);simdgroup_multiply_accumulate(A1,Q1,K,A1);}simdgroup_store(A0,ss+kt*8u,64u,0,false);simdgroup_store(A1,ss+512u+kt*8u,64u,0,false);threadgroup_barrier(mem_flags::mem_threadgroup);
  uint j=uint(sg),qi=q0+j,posq=p.query_position_offset+qi;
  for(uint hi=0;hi<2u;hi++){float scores[2],rm=-3.402823466e38f;for(uint kl=0;kl<2u;kl++){uint kk=uint(lane)+kl*32u,ki=kc+kk;bool ok=ki<p.kv_tokens&&qi<p.q_len&&phys[kk]!=0xffffffffu&&p.kv_position_offset+ki<=posq;float x=ok?ss[hi*512u+j*64u+kk]:-3.402823466e38f;if(!isfinite(x))x=-3.402823466e38f;scores[kl]=x;rm=max(rm,x);}rm=simd_max(rm);float old=M[hi*8u+j],nm=max(old,rm),corr=1.0f;if(nm>-3e38f)corr=old>-3e38f?exp(old-nm):0;float sl=0;for(uint kl=0;kl<2u;kl++){uint kk=uint(lane)+kl*32u;float x=scores[kl],e=(nm>-3e38f&&x>-3e38f)?exp(x-nm):0;sp[hi*512u+j*64u+kk]=half(e);sl+=e;}float sum=simd_sum(sl);if(lane==0){S[hi*8u+j]=S[hi*8u+j]*corr+sum;M[hi*8u+j]=nm;diag[hi*64u+j*8u+j]=corr;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_float8x8 C0,C1;simdgroup_load(C0,diag,8u);simdgroup_load(C1,diag+64u,8u);for(uint z=0;z<8u;z++){simdgroup_float8x8 T0,T1;simdgroup_multiply(T0,C0,O0[z]);simdgroup_multiply(T1,C1,O1[z]);O0[z]=T0;O1[z]=T1;}
  for(uint k8=0;k8<8u;k8++){uint vph=phys[k8*8u];simdgroup_half8x8 P0,P1;simdgroup_load(P0,sp+k8*8u,64u);simdgroup_load(P1,sp+512u+k8*8u,64u);for(uint z=0;z<8u;z++){uint d8=ds+z*8u;simdgroup_half8x8 V;if(kc+k8*8u+8u<=p.kv_tokens){simdgroup_load(V,vh+vph*p.v_row_stride+kvh+d8,p.v_row_stride);}else{threadgroup half*sv=(threadgroup half*)ss+uint(sg)*64u;for(uint vi=uint(lane);vi<64u;vi+=32u){uint vr=vi/8u,vc=vi&7u,xph=phys[k8*8u+vr];sv[vi]=xph!=0xffffffffu?vh[xph*p.v_row_stride+kvh+d8+vc]:half(0);}simdgroup_barrier(mem_flags::mem_threadgroup);simdgroup_load(V,sv,8u);}simdgroup_multiply_accumulate(O0[z],P0,V,O0[z]);simdgroup_multiply_accumulate(O1[z],P1,V,O1[z]);}}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);if(tid<16u){uint hi=uint(tid)/8u,j=uint(tid)&7u;diag[hi*64u+j*8u+j]=S[hi*8u+j]>0?1/S[hi*8u+j]:0;}threadgroup_barrier(mem_flags::mem_threadgroup);
 simdgroup_float8x8 I0,I1;simdgroup_load(I0,diag,8u);simdgroup_load(I1,diag+64u,8u);for(uint z=0;z<8u;z++){simdgroup_float8x8 T0,T1;simdgroup_multiply(T0,I0,O0[z]);simdgroup_multiply(T1,I1,O1[z]);O0[z]=T0;O1[z]=T1;}
 if(q0+8u<=p.q_len){for(uint z=0;z<8u;z++){simdgroup_store(O0[z],out+q0*qs+h0*512u+ds+z*8u,qs);simdgroup_store(O1[z],out+q0*qs+(h0+1u)*512u+ds+z*8u,qs);}return;}
 uint valid=p.q_len-q0;for(uint hi=0;hi<2u;hi++)for(uint owner=0;owner<8u;owner++){threadgroup_barrier(mem_flags::mem_threadgroup);if(uint(sg)==owner){if(hi==0)for(uint z=0;z<8u;z++)simdgroup_store(O0[z],tail+z*8u,64u);else for(uint z=0;z<8u;z++)simdgroup_store(O1[z],tail+z*8u,64u);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=uint(tid);i<valid*64u;i+=256u){uint j=i/64u,d=i&63u;out[(q0+j)*qs+(h0+hi)*512u+owner*64u+d]=tail[i];}}
}
// Keep Q4_K half staging and Q6_K float staging as distinct compile-time policies.
template<typename Input, typename Tile, bool Q6> inline void termite_k_mm_matrix_m64_n64(
    device const Input *input, device const uchar *weight, device float *output,
    int rows, int in_dim, int out_dim, uint3 thread_pos, uint3 group_pos,
    ushort simdgroup_id, threadgroup uchar *shared) {
    const uint tid = thread_pos.x; const uint first_o = group_pos.x * 64u; const uint first_r = group_pos.y * 64u;
    if (rows < 2 || in_dim <= 0 || (uint(in_dim) & 255u) != 0u || out_dim <= 0 || first_r >= uint(rows) || first_o >= uint(out_dim)) return;
    threadgroup Tile *weight_tile = (threadgroup Tile *)shared; threadgroup Tile *input_tile = weight_tile + 2048;
    simdgroup_matrix<Tile, 8, 8> mw[8]; simdgroup_matrix<Tile, 8, 8> mx; simdgroup_float8x8 acc[8]; for (uint i = 0u; i < 8u; ++i) acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint block_count = uint(in_dim) >> 8; const uint sg_row = uint(simdgroup_id) * 8u;
    for (uint k0 = 0u; k0 < uint(in_dim); k0 += 32u) {
        if (Q6) {
        for (uint index = tid; index < 2048u; index += 256u) { uint k = index >> 6; uint col = index & 63u; uint global_col = first_o + col; uint global_k = k0 + k; float value = 0.0f; if (global_col < uint(out_dim)) { device const uchar *block = weight + (global_col * block_count + (global_k >> 8)) * 210u; value = antfly_q6_k_dequant_lane_v2(block, int(global_k & 255u)); } weight_tile[k * 64u + col] = value; }
        } else {
        { const uint col = tid & 63u; const uint k_lane = tid >> 6u; const uint global_col = first_o + col; if (global_col < uint(out_dim)) { const uint block_index = k0 >> 8u; const uint sub = (k0 & 255u) >> 5u; device const uchar *block = weight + (global_col * block_count + block_index) * 144u; float raw_scale = 0.0f; float raw_min = 0.0f; antfly_qk_unpack_scale_min_6bit(block + 4u, int(sub), raw_scale, raw_min); const float dsc = antfly_qk_half_le_to_float(block) * raw_scale; const float dmn = antfly_qk_half_le_to_float(block + 2u) * raw_min; device const uchar *qs = block + 16u + (sub >> 1u) * 32u; for (uint step = 0u; step < 8u; ++step) { const uint k = k_lane + step * 4u; const uchar packed = qs[k]; const uint qv = (sub & 1u) == 0u ? uint(packed & 0x0fu) : uint(packed >> 4u); weight_tile[k * 64u + col] = half(dsc * float(qv) - dmn); } } else { for (uint step = 0u; step < 8u; ++step) { const uint k = k_lane + step * 4u; weight_tile[k * 64u + col] = half(0.0f); } } }
        }
        for (uint index = tid; index < 2048u; index += 256u) { uint row = index >> 5; uint k = index & 31u; uint global_row = first_r + row; input_tile[row * 32u + k] = global_row < uint(rows) ? Tile(input[global_row * uint(in_dim) + k0 + k]) : Tile(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 32u; k += 8u) { for (uint i = 0u; i < 8u; ++i) simdgroup_load(mw[i], weight_tile + k * 64u + i * 8u, 64u); simdgroup_load(mx, input_tile + sg_row * 32u + k, 32u); for (uint i = 0u; i < 8u; ++i) simdgroup_multiply_accumulate(acc[i], mx, mw[i], acc[i]); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float *result_tile = (threadgroup float *)shared; for (uint i = 0u; i < 8u; ++i) simdgroup_store(acc[i], result_tile + sg_row * 64u + i * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < 4096u; index += 256u) { uint row = index >> 6; uint col = index & 63u; uint global_row = first_r + row; if (global_row < uint(rows) && first_o + col < uint(out_dim)) output[global_row * uint(out_dim) + first_o + col] = result_tile[index]; }
}
