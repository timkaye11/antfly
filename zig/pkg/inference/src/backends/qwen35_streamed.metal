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

#include <metal_stdlib>
using namespace metal;
struct Q4Params {
    uint op, n, dim, groups, position, stride, mode, index;
    float epsilon, scale, theta, reserved;
};
inline float q4sig(float x) { return 1.0f/(1.0f+exp(-x)); }
inline float q4silu(float x) { return x*q4sig(x); }
// Random-access GGUF decoding: keep encoded weights resident and fuse decode
// into MMV, or fill a bounded tile for batched MPS GEMM.
inline float q35packed(device const uchar *w, uint element, uint kind, device const uint *grid) {
    uint j=element%256;
    if(kind==11) { // Q3_K
        w+=(element/256)*110;
        uint sub=j/16;
        uint scale=((w[96+sub%8]>>(sub>=8 ? 4 : 0))&15)
                 | (((w[104+sub%4]>>(2*(sub/4)))&3)<<4);
        uint low=(w[32+(j/128)*32+j%32]>>(2*((j%128)/32)))&3;
        uint high=(w[j%32]>>(j/32))&1;
        ushort bits=ushort(w[108]) | (ushort(w[109])<<8);
        return (float(as_type<half>(bits))*float(int(scale)-32))*float(int(low)+int(high)*4-4);
    }
    uint group=j/32;
    if(kind==21) { // IQ3_S, the shared CPU codebook is uploaded once.
        w+=(element/256)*110;
        uint lane=(j%32)/8, component=j%8;
        uint code=w[2+group*8+2*lane+component/4]
                 | (((w[66+group]>>(2*lane+component/4))&1)<<8);
        uint magnitude=(grid[code]>>(8*(component%4)))&255;
        uint scale=(w[106+group/2]>>(4*(group%2)))&15;
        ushort bits=ushort(w[0]) | (ushort(w[1])<<8);
        float value=(float(as_type<half>(bits))*float(1+2*scale))*float(magnitude);
        return (w[74+group*4+lane]&(1<<component)) ? -value : value;
    }
    // IQ4_XS
    constexpr int values[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    w+=(element/256)*136;
    ushort bits=ushort(w[0]) | (ushort(w[1])<<8);
    uint high=uint(w[2]) | (uint(w[3])<<8);
    uint scale=((w[4+group/2]>>(4*(group%2)))&15) | (((high>>(2*group))&3)<<4);
    uint packed=w[8+group*16+j%16];
    uint code=(j%32)<16 ? packed&15 : packed>>4;
    return (float(as_type<half>(bits))*float(int(scale)-32))*float(values[code]);
}
kernel void qwen35_streamed(
    device float *a [[buffer(0)]], device float *b [[buffer(1)]],
    device float *c [[buffer(2)]], device float *d [[buffer(3)]],
    device float *e [[buffer(4)]], device float *f [[buffer(5)]],
    device float *g [[buffer(6)]], device float *o [[buffer(7)]],
    constant Q4Params &p [[buffer(8)]], uint id [[thread_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (id >= p.n) return;
    switch (p.op) {
    case 0: o[id] = 0; break;
    case 1: o[id] = a[id % p.dim]; break;
    case 2: { // grouped RMS norm, optional shared gamma or L2 scaling
        // One SIMD group per normalization group, rather than recomputing
        // the entire sum independently for every output component.
        uint base = (id/32)*p.dim;
        float sum = 0;
        for(uint j=lane;j<p.dim;j+=32) sum += a[base+j]*a[base+j];
        sum=simd_sum(sum);
        float norm = rsqrt((p.mode==2 ? sum : sum/float(p.dim))+p.epsilon);
        for(uint j=lane;j<p.dim;j+=32)
            o[base+j] = a[base+j]*norm*(p.mode==2 ? 1.0f : b[p.mode==1 ? j : base+j]);
        break;
    }
    case 9: { // causal four-tap depthwise conv, history stored oldest first
        uint history=3*p.stride;
        float value=a[id]*b[id*4+3];
        for(uint k=0;k<3;++k) value+=c[(k*p.stride)*p.n+id]*b[id*4+k];
        for(uint t=0;t+1<history;++t) c[t*p.n+id]=c[(t+1)*p.n+id];
        c[(history-1)*p.n+id]=a[id];
        o[id]=q4silu(value); break;
    }
    case 10: { // recurrent delta rule, one lane per value column
        uint head=id/128, v=id%128;
        // GGUF converter reorders GDN heads for ggml repeat semantics.
        uint kh=head%16;
        float decay=exp(d[head]* (max(c[head]+e[head],0.0f)+log(1.0f+exp(-abs(c[head]+e[head])))));
        float beta=q4sig(b[head]);
        uint base=head*128*128;
        float prediction=0;
        for(uint k=0;k<128;++k) {
            f[base+k*128+v]*=decay;
            prediction+=f[base+k*128+v]*a[2048+kh*128+k];
        }
        float delta=(a[4096+head*128+v]-prediction)*beta, result=0;
        for(uint k=0;k<128;++k) {
            float state=f[base+k*128+v]+a[2048+kh*128+k]*delta;
            f[base+k*128+v]=state;
            result+=state*a[kh*128+k]*0.08838834764831845f;
        }
        o[id]=result; break;
    }
    case 12: o[id]=q4silu(a[id])*b[id]; break;
    case 16: o[id]=a[id]+b[id]; break;
    case 18: { // split interleaved attention Q|gate
        uint h=id/256, j=id%256;
        o[id]=a[h*512+j]; b[id]=a[h*512+256+j]; break;
    }
    case 19: { // partial Neox RoPE on normalized contiguous heads; text positions
        uint j=id%p.dim, base=id-j;
        if(j>=64) {o[id]=a[id]; break;}
        uint k=j%32; float angle=float(p.position)*pow(p.theta,-2.0f*float(k)/64.0f);
        o[id]=a[id]*cos(angle)+(j<32 ? -a[base+k+32] : a[base+k])*sin(angle); break;
    }
    case 20: o[p.position*p.dim+id]=a[id]; break;
    case 25: { // softmax each attention head, in place after scores complete
        uint n=p.position+1, base=id*n; float maximum=-INFINITY;
        for(uint j=0;j<n;++j) maximum=max(maximum,a[base+j]);
        float sum=0;for(uint j=0;j<n;++j) {o[base+j]=exp(a[base+j]-maximum);sum+=o[base+j];}
        for(uint j=0;j<n;++j) o[base+j]/=sum;break;
    }
    case 27: { // greedy token selection, deterministic id tie break
        float best=-INFINITY; uint winner=0xffffffffu; bool finite=true;
        for(uint j=lane;j<p.dim;j+=32) {
            finite &= isfinite(a[j]);
            if(a[j]>best || (a[j]==best && j<winner)) {best=a[j];winner=j;}
        }
        float maximum=simd_max(best);
        winner=simd_min(best==maximum ? winner : 0xffffffffu);
        if(!simd_all(finite)) winner=0xffffffffu;
        if(lane==0) ((device uint*)o)[0]=winner;
        break;
    }
    case 4: { // GPU owns tile writes; subsequent GEMM and reuse are ordered.
        o[id]=q35packed(reinterpret_cast<device const uchar *>(a),p.index*p.dim+id,p.mode,reinterpret_cast<device const uint *>(b));
        break;
    }
    case 5: { // IQ3_S / IQ4_XS fused packed matrix-vector product.
        uint task=id/32, column=task%p.groups, row=task/p.groups;
        float sum=0;
        for(uint j=lane;j<p.dim;j+=32)
            sum+=b[row*p.dim+j]*q35packed(reinterpret_cast<device const uchar *>(a),column*p.dim+j,p.mode,reinterpret_cast<device const uint *>(c));
        sum=simd_sum(sum);
        if(lane==0) o[row*p.groups+column]=sum;
        break;
    }
    case 6: { // bounded FP32 tile, output rows stay in their original matrix
        uint task=id/32, column=task%p.groups, row=task/p.groups;
        float sum=0;
        for(uint j=lane;j<p.dim;j+=32) sum+=a[column*p.dim+j]*b[row*p.dim+j];
        sum=simd_sum(sum);
        if(lane==0) o[row*p.stride+p.index+column]=sum;
        break;
    }
    case 8: o[(id/p.groups)*p.stride+p.index+id%p.groups]=a[id]; break;
    case 3: { // Bounded IQ4_NL tile, decoded once for all prompt rows.
        constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        device const uchar *weights=reinterpret_cast<device const uchar *>(a);
        uint element=p.index*p.dim+id;
        uint offset=(element/32)*18;
        ushort bits=ushort(weights[offset]) | (ushort(weights[offset+1])<<8);
        uchar packed=weights[offset+2+(element%16)];
        uint code=(element%32)<16 ? packed&15 : packed>>4;
        o[id]=float(as_type<half>(bits))*float(values[code]);
        break;
    }
    case 7: { // IQ4_NL: one SIMD group per output, including all wide-row blocks.
        constexpr int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        uint task=id/32, column=task%p.groups, row=task/p.groups;
        device const uchar *weights=reinterpret_cast<device const uchar *>(a);
        uint weight_row=column*(p.dim/32)*18;
        float sum=0;
        for(uint j=lane;j<p.dim;j+=32) {
            uint offset=weight_row+(j/32)*18;
            ushort bits=ushort(weights[offset]) | (ushort(weights[offset+1])<<8);
            uchar packed=weights[offset+2+(j%16)];
            uint code=(j%32)<16 ? packed&15 : packed>>4;
            sum+=b[row*p.dim+j]*float(as_type<half>(bits))*float(values[code]);
        }
        sum=simd_sum(sum);
        if(lane==0) o[row*p.groups+column]=sum;
        break;
    }
    case 11: o[id]=a[id]*q4silu(b[id]); break; // Qwen3.5 GDN uses SiLU, not sigmoid.
    case 24: { // dense causal attention, four KV heads / six query heads per KV
        uint tokens=p.position+1, h=id/tokens, t=id%tokens;
        float dot=0;for(uint j=0;j<256;++j) dot+=a[h*256+j]*b[t*1024+(h/6)*256+j];
        o[id]=dot*0.0625f;break;
    }
    case 26: {
        uint h=id/256,j=id%256,n=p.position+1;float sum=0;
        for(uint t=0;t<n;++t) sum+=a[h*n+t]*b[t*1024+(h/6)*256+j];
        o[id]=sum*q4sig(c[id]);break;
    }
    }
}
