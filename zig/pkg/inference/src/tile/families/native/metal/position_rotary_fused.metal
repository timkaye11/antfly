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

// native-program:entry termite_head_rms_rope_hd256_512_hybrid
// native-program:indent 11
kernel void termite_head_rms_rope_hd256_512_hybrid(device const float *input [[buffer(0)]],
                                   device const float *weight [[buffer(1)]],
                                   device float *output [[buffer(2)]],
                                   constant termite_metal_head_rms_rope_params &p [[buffer(3)]],
                                   threadgroup float *partial [[threadgroup(0)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   uint head [[threadgroup_position_in_grid]]) {
    if(head>=p.total_heads)return;
    uint base=head*p.head_dim;
    float sum=0.0f;
    for(uint i=tid;i<p.head_dim;i+=256u){float x=input[base+i];sum+=x*x;}
    sum=simd_sum(sum);if(lane==0u)partial[uint(sg)]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0u){float v=lane<8u?partial[uint(lane)]:0.0f;v=simd_sum(v);if(lane==0u)partial[0]=v;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms=rsqrt(partial[0]/float(p.head_dim)+p.eps);
    uint rope_half=p.rope_dim/2u,head_half=p.head_dim/2u;
    for(uint d=tid;d<p.head_dim;d+=256u){
        float value=input[base+d]*inv_rms*weight[d]*p.value_scale;
        uint idx0=0u,idx1=0u,j=0u;bool apply=p.rope_dim>=2u;
        if(apply&&p.consecutive_pairs!=0u){if(d>=p.rope_dim)apply=false;else{j=d/2u;idx0=2u*j;idx1=idx0+1u;}}
        else if(apply){if(d<rope_half){j=d;idx0=d;idx1=head_half+d;}else if(d>=head_half&&d<head_half+rope_half){j=d-head_half;idx0=j;idx1=d;}else apply=false;}
        if(!apply){output[base+d]=value;continue;}
        float x0=input[base+idx0]*inv_rms*weight[idx0]*p.value_scale;
        float x1=input[base+idx1]*inv_rms*weight[idx1]*p.value_scale;
        float freq=1.0f/pow(p.theta,float(2u*j)/float(p.rope_dim));
        uint row=head/p.heads_per_row;if(p.position_period!=0u)row%=p.position_period;
        float angle=float(p.position+row)*p.freq_scale*freq,c=cos(angle),s=sin(angle);
        output[base+d]=d==idx0?x0*c-x1*s:(d==idx1?x0*s+x1*c:value);
    }
}
// native-program:entry termite_head_rms_rope_cossin
// native-program:indent 11
kernel void termite_head_rms_rope_cossin(device const float *input [[buffer(0)]],
                                   device const float *weight [[buffer(1)]],
                                   device float *output [[buffer(2)]],
                                   constant termite_metal_head_rms_rope_params &p [[buffer(3)]],
                                   device const float2 *cossin [[buffer(4)]],
                                   threadgroup float *partial [[threadgroup(0)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   uint head [[threadgroup_position_in_grid]]) {
    if(head>=p.total_heads)return;
    uint base=head*p.head_dim;
    float sum=0.0f;
    for(uint i=tid;i<p.head_dim;i+=256u){float x=input[base+i];sum+=x*x;}
    sum=simd_sum(sum);if(lane==0u)partial[uint(sg)]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0u){float v=lane<8u?partial[uint(lane)]:0.0f;v=simd_sum(v);if(lane==0u)partial[0]=v;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms=rsqrt(partial[0]/float(p.head_dim)+p.eps);
    uint rope_half=p.rope_dim/2u,head_half=p.head_dim/2u;
    for(uint d=tid;d<p.head_dim;d+=256u){
        float value=input[base+d]*inv_rms*weight[d]*p.value_scale;
        uint idx0=0u,idx1=0u,j=0u;bool apply=p.rope_dim>=2u;
        if(apply&&p.consecutive_pairs!=0u){if(d>=p.rope_dim)apply=false;else{j=d/2u;idx0=2u*j;idx1=idx0+1u;}}
        else if(apply){if(d<rope_half){j=d;idx0=d;idx1=head_half+d;}else if(d>=head_half&&d<head_half+rope_half){j=d-head_half;idx0=j;idx1=d;}else apply=false;}
        if(!apply){output[base+d]=value;continue;}
        float x0=input[base+idx0]*inv_rms*weight[idx0]*p.value_scale;
        float x1=input[base+idx1]*inv_rms*weight[idx1]*p.value_scale;
                uint row=head/p.heads_per_row;if(p.position_period!=0u)row%=p.position_period;
        float2 cs=cossin[(p.position+row)*(p.rope_dim/2u)+j];float c=cs.x,s=cs.y;
        output[base+d]=d==idx0?x0*c-x1*s:(d==idx1?x0*s+x1*c:value);
    }
}

// native-program:entry termite_head_rms_rope_prefill_cossin
// native-program:indent 11
kernel void termite_head_rms_rope_prefill_cossin(device const float *input [[buffer(0)]],
                               device const float *weight [[buffer(1)]],
                               device float *output [[buffer(2)]],
                               constant termite_metal_head_rms_rope_params &p [[buffer(3)]], device const float2 *cossin [[buffer(4)]],
                               threadgroup float *shmem [[threadgroup(0)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint tpg [[threads_per_threadgroup]],
                               uint head [[threadgroup_position_in_grid]]) {
    if (head >= p.total_heads || p.head_dim == 0u || p.rope_dim == 0u) return;
    uint base = head * p.head_dim;
    float mean_square = 0.0f;
    for (uint i = tid; i < p.head_dim; i += tpg) { float x = input[base + i]; mean_square += x * x; }
    shmem[tid] = mean_square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) { if (tid < stride) shmem[tid] += shmem[tid + stride]; threadgroup_barrier(mem_flags::mem_threadgroup); }
    float inv_rms = rsqrt(shmem[0] / float(p.head_dim) + p.eps);
    uint rope_half = p.rope_dim / 2u, head_half = p.head_dim / 2u;
    for (uint d = tid; d < p.head_dim; d += tpg) {
        float value = input[base + d] * inv_rms * weight[d] * p.value_scale;
        if (p.rope_dim < 2u) { output[base + d] = value; continue; }
        uint idx0 = 0u, idx1 = 0u, j = 0u; bool apply = true;
        if (p.consecutive_pairs != 0u) { if (d >= p.rope_dim) apply = false; else { j = d / 2u; idx0 = 2u*j; idx1 = idx0+1u; } }
        else { if (d < rope_half) { j=d; idx0=d; idx1=head_half+d; } else if (d >= head_half && d < head_half+rope_half) { j=d-head_half; idx0=j; idx1=d; } else apply=false; }
        if (!apply) { output[base+d]=value; continue; }
        float x0=input[base+idx0]*inv_rms*weight[idx0]*p.value_scale;
        float x1=input[base+idx1]*inv_rms*weight[idx1]*p.value_scale;
        float freq=1.0f/pow(p.theta,float(2u*j)/float(p.rope_dim));
        uint row=p.heads_per_row==0u?0u:head/p.heads_per_row;
        if(p.position_period!=0u)row%=p.position_period;
        float2 cs=cossin[(p.position+row)*(p.rope_dim/2u)+j];float c=cs.x,s=cs.y;
        output[base+d]=d==idx0?x0*c-x1*s:(d==idx1?x0*s+x1*c:value);
    }
}

// native-program:entry termite_head_rms_rope_qk_fused
// native-program:indent 11
kernel void termite_head_rms_rope_qk_fused(device const float *q_input [[buffer(0)]], device const float *q_weight [[buffer(1)]], device float *q_output [[buffer(2)]], constant termite_metal_head_rms_rope_params &q [[buffer(3)]], device const float2 *cossin [[buffer(4)]], device const float *k_input [[buffer(5)]], device const float *k_weight [[buffer(6)]], device float *k_output [[buffer(7)]], constant termite_metal_head_rms_rope_params &k [[buffer(8)]],
                                   threadgroup float *partial [[threadgroup(0)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   uint head [[threadgroup_position_in_grid]]) {
    uint qheads=q.total_heads,khead=head-qheads;bool is_k=head>=qheads;if(is_k&&khead>=k.total_heads)return;termite_metal_head_rms_rope_params p=is_k?k:q;device const float *input=is_k?k_input:q_input;device const float *weight=is_k?k_weight:q_weight;device float *output=is_k?k_output:q_output;head=is_k?khead:head;
    uint base=head*p.head_dim;
    float sum=0.0f;
    for(uint i=tid;i<p.head_dim;i+=256u){float x=input[base+i];sum+=x*x;}
    sum=simd_sum(sum);if(lane==0u)partial[uint(sg)]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0u){float v=lane<8u?partial[uint(lane)]:0.0f;v=simd_sum(v);if(lane==0u)partial[0]=v;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms=rsqrt(partial[0]/float(p.head_dim)+p.eps);
    uint rope_half=p.rope_dim/2u,head_half=p.head_dim/2u;
    for(uint d=tid;d<p.head_dim;d+=256u){
        float value=input[base+d]*inv_rms*weight[d]*p.value_scale;
        uint idx0=0u,idx1=0u,j=0u;bool apply=p.rope_dim>=2u;
        if(apply&&p.consecutive_pairs!=0u){if(d>=p.rope_dim)apply=false;else{j=d/2u;idx0=2u*j;idx1=idx0+1u;}}
        else if(apply){if(d<rope_half){j=d;idx0=d;idx1=head_half+d;}else if(d>=head_half&&d<head_half+rope_half){j=d-head_half;idx0=j;idx1=d;}else apply=false;}
        if(!apply){output[base+d]=value;continue;}
        float x0=input[base+idx0]*inv_rms*weight[idx0]*p.value_scale;
        float x1=input[base+idx1]*inv_rms*weight[idx1]*p.value_scale;
                uint row=head/p.heads_per_row;if(p.position_period!=0u)row%=p.position_period;
        float2 cs=cossin[(p.position+row)*(p.rope_dim/2u)+j];float c=cs.x,s=cs.y;
        output[base+d]=d==idx0?x0*c-x1*s:(d==idx1?x0*s+x1*c:value);
    }
}

// native-program:entry termite_head_rms_rope_qkv_fused
// native-program:indent 11
kernel void termite_head_rms_rope_qkv_fused(device const float*q_input[[buffer(0)]],device const float*q_weight[[buffer(1)]],device float*q_output[[buffer(2)]],constant termite_metal_head_rms_rope_params&q[[buffer(3)]],device const float2*cossin[[buffer(4)]],device const float*k_input[[buffer(5)]],device const float*k_weight[[buffer(6)]],device float*k_output[[buffer(7)]],constant termite_metal_head_rms_rope_params&k[[buffer(8)]],device const float*v_input[[buffer(9)]],device float*v_output[[buffer(10)]],constant termite_metal_head_rms_rope_params&v[[buffer(11)]],threadgroup float*partial[[threadgroup(0)]],uint tid[[thread_index_in_threadgroup]],ushort lane[[thread_index_in_simdgroup]],ushort sg[[simdgroup_index_in_threadgroup]],uint group[[threadgroup_position_in_grid]]){
 uint qk=q.total_heads+k.total_heads;if(group>=qk){uint row=group-qk;if(row>=v.total_heads)return;if(sg==0u)partial[uint(lane)]=0.0f;uint base=row*v.head_dim,vectors=v.head_dim>>2u;float sum=0.0f;if(tid<128u)for(uint z=tid;z<vectors;z+=128u){uint i=base+(z<<2u);float4 x=float4(v_input[i],v_input[i+1],v_input[i+2],v_input[i+3]);sum+=dot(x,x);}sum=simd_sum(sum);threadgroup_barrier(mem_flags::mem_threadgroup);if(tid<128u&&lane==0u)partial[uint(sg)]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);float total=tid<128u?simd_sum(partial[uint(lane)]):0.0f;float inv=rsqrt(total/float(v.head_dim)+v.eps);if(tid<128u)for(uint z=tid;z<vectors;z+=128u){uint i=base+(z<<2u);float4 x=float4(v_input[i],v_input[i+1],v_input[i+2],v_input[i+3]);float4 y=x*inv;v_output[i]=y.x;v_output[i+1]=y.y;v_output[i+2]=y.z;v_output[i+3]=y.w;}return;}
 bool is_k=group>=q.total_heads;uint head=is_k?group-q.total_heads:group;termite_metal_head_rms_rope_params p=is_k?k:q;device const float*input=is_k?k_input:q_input;device const float*weight=is_k?k_weight:q_weight;device float*output=is_k?k_output:q_output;uint base=head*p.head_dim;float sum=0.0f;for(uint i=tid;i<p.head_dim;i+=256u){float x=input[base+i];sum+=x*x;}sum=simd_sum(sum);if(lane==0u)partial[uint(sg)]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);if(sg==0u){float z=lane<8u?partial[uint(lane)]:0.0f;z=simd_sum(z);if(lane==0u)partial[0]=z;}threadgroup_barrier(mem_flags::mem_threadgroup);float inv=rsqrt(partial[0]/float(p.head_dim)+p.eps);uint rh=p.rope_dim/2u,hh=p.head_dim/2u;for(uint d=tid;d<p.head_dim;d+=256u){float value=input[base+d]*inv*weight[d]*p.value_scale;uint i0=0,i1=0,j=0;bool apply=p.rope_dim>=2u;if(apply&&p.consecutive_pairs!=0u){if(d>=p.rope_dim)apply=false;else{j=d/2u;i0=2u*j;i1=i0+1u;}}else if(apply){if(d<rh){j=d;i0=d;i1=hh+d;}else if(d>=hh&&d<hh+rh){j=d-hh;i0=j;i1=d;}else apply=false;}if(!apply){output[base+d]=value;continue;}float x0=input[base+i0]*inv*weight[i0]*p.value_scale,x1=input[base+i1]*inv*weight[i1]*p.value_scale;uint row=head/p.heads_per_row;if(p.position_period)row%=p.position_period;float2 cs=cossin[(p.position+row)*(p.rope_dim/2u)+j];output[base+d]=d==i0?x0*cs.x-x1*cs.y:(d==i1?x0*cs.y+x1*cs.x:value);}}

struct TableParams { uint rope_dim, positions; float theta, freq_scale; };
// native-program:entry termite_init_rope_cossin
// native-program:indent 11
kernel void termite_init_rope_cossin(device float2 *out [[buffer(0)]], constant TableParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 uint pairs=p.rope_dim/2u;if(i>=p.positions*pairs)return;uint pos=i/pairs,j=i%pairs;
 float freq=1.0f/pow(p.theta,float(2u*j)/float(p.rope_dim));
 float angle=float(pos)*p.freq_scale*freq;out[i]=float2(cos(angle),sin(angle));
}
