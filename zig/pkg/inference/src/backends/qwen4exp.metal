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
constant short iq4_values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
// Scalar definitions match Antfly's GGUF dequantization contracts. One SIMD
// group reduces one output row; a route mask permits resident/missing batches.
inline float q4exp_weight(device const uchar *base, uint k, uint format) {
    if (format == 8) {
        device const uchar *q=base+(k/32)*34;
        return float(*(device const half*)q)*float(((device const char*)(q+2))[k%32]);
    }
    if (format == 7) {
        device const uchar *q=base+(k/32)*24;
        uint j=k%32;
        uint high=uint(q[4])|(uint(q[5])<<8)|(uint(q[6])<<16)|(uint(q[7])<<24);
        uint low=(q[8+j%16]>>(j<16 ? 0 : 4))&15;
        return float(*(device const half*)q)*float(low|(((high>>j)&1)<<4))+float(*(device const half*)(q+2));
    }
    uint blockbytes=format==12 ? 144 : 176;
    device const uchar *q=base+(k/256)*blockbytes;
    uint j=k%256, sub=j/32;
    device const uchar *sc=q+4;
    uint scale=sub<4 ? sc[sub]&63 : (sc[sub+4]&15)|((sc[sub-4]>>6)<<4);
    uint minimum=sub<4 ? sc[sub+4]&63 : (sc[sub+4]>>4)|((sc[sub]>>6)<<4);
    device const uchar *qs=q+(format==12 ? 16 : 48);
    uint low=(qs[(sub/2)*32+j%32]>>((sub%2)*4))&15;
    if (format==13) low|=((q[16+j%32]>>sub)&1)<<4;
    return float(*(device const half*)q)*float(scale)*float(low)-float(*(device const half*)(q+2))*float(minimum);
}
// All numerical model operations execute here or through Antfly's existing
// quantized Metal matvec kernels. Zig controls file I/O and route leases.
kernel void qwen4exp_eager(
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
    case 3: o[id] = q4silu(a[id]*p.scale); break;
    case 4: { // hyperconnection mixer
        float sum=0; for(uint j=0;j<4;++j) sum += a[j*p.dim+id]*q4sig(b[j*p.dim+id]);
        o[id]=sum*0.25f; break;
    }
    case 5: o[id] = a[id]+b[id%p.dim]*2.0f*q4sig(c[id/p.dim]*0.25f); break;
    case 6: { // F32 or BF16 matvec, one SIMD group per output row
        uint row=id/32;
        float sum=0;
        for(uint j=lane;j<p.dim;j+=32) {
            uint k=row*p.dim+j;
            float w=p.mode==30 ? as_type<float>(uint(((device ushort*)a)[k])<<16) : a[k];
            sum+=w*b[j];
        }
        sum=simd_sum(sum); if(lane==0) o[row]=sum; break;
    }
    case 28: { // grouped quantized projections, output [route, row]
        uint row=id/32, lane=id%32, route=row/p.groups, col=row%p.groups;
        if ((p.index&(1u<<route))==0) break;
        uint block=p.mode==7 || p.mode==8 ? 32 : 256;
        uint bytes=p.mode==7 ? 24 : p.mode==8 ? 34 : p.mode==12 ? 144 : 176;
        device const uchar *w=(device const uchar*)a+ulong(((device uint*)c)[route])*p.stride+p.position+ulong(col)*(p.dim/block)*bytes;
        device float *x=b+(p.scale!=0 ? route*p.dim : 0);
        float total=0;
        for(uint k=lane;k<p.dim;k+=32) total+=q4exp_weight(w,k,p.mode)*x[k];
        total=simd_sum(total);
        if(lane==0) o[row]=total;
        break;
    }
    case 29: { // masked grouped SwiGLU; never read uncomputed route rows
        if (p.index&(1u<<(id/p.dim))) o[id]=q4silu(a[id])*b[id];
        break;
    }
    case 32: o[(id/p.dim)*p.stride+p.index+id%p.dim]=a[id]; break;
    case 30: { // gather token activations for one expert from route locations
        uint location=((device const uint*)b)[id/p.dim];
        o[id]=a[(location/10)*p.dim+id%p.dim];
        break;
    }
    case 31: { // scatter expert outputs into their original token/route positions
        uint location=((device const uint*)b)[id/p.dim];
        o[location*p.dim+id%p.dim]=a[id];
        break;
    }
    case 7: { // encoded Q8_0 embedding row; 32 values per 34 bytes
        device uchar *raw=(device uchar*)a;
        uint k=p.index*p.dim+id%p.dim, block=k/32;
        float scale=float(*(device half*)(raw+block*34));
        o[id]=scale*float(((device char*)(raw+block*34+2))[k%32]); break;
    }
    case 8: { // IQ4_NL ngram rows; packed blocks of 32 values
        device uchar *raw=(device uchar*)a;
        uint block=id/32, j=id%32;
        float scale=float(*(device half*)(raw+block*18));
        uint nibble=(raw[block*18+2+j%16]>>(j<16 ? 0 : 4))&15;
        o[id]=scale*float(iq4_values[nibble]); break;
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
    case 11: o[id]=a[id]*q4sig(b[id]); break;
    case 12: o[id]=q4silu(a[id])*b[id]; break;
    case 13: { // deterministic top ten, normalized selected softmax
        bool finite=true;
        for(uint j=lane;j<512;j+=32) finite &= isfinite(a[j]);
        if(!simd_all(finite)) {
            if(lane==0) for(uint k=0;k<10;++k) ((device uint*)o)[k]=0xffffffffu;
            return;
        }
        float selected[10]; uint ids[10];
        for(uint k=0;k<10;++k) {
            float best=-INFINITY; uint winner=0xffffffffu;
            for(uint j=lane;j<512;j+=32) {
                bool used=false;for(uint t=0;t<k;++t) used|=ids[t]==j;
                if(!used && (a[j]>best || (a[j]==best && j<winner))) { best=a[j]; winner=j; }
            }
            float maximum=simd_max(best);
            winner=simd_min(best==maximum ? winner : 0xffffffffu);
            ids[k]=winner; selected[k]=maximum;
        }
        if(lane!=0) break;
        float sum=0; float maximum=a[ids[0]];
        for(uint k=0;k<10;++k) { selected[k]=exp(a[ids[k]]-maximum); sum+=selected[k]; }
        for(uint k=0;k<10;++k) { ((device uint*)o)[k]=ids[k]; o[10+k]=selected[k]/sum; }
        break;
    }
    case 14: o[id]+=a[id]*b[p.index]; break;
    case 15: o[id]=a[id]*q4sig(b[0]); break;
    case 16: o[id]=a[id]+b[id]; break;
    case 17: { // PLE signed-root gate, repeated once per stream component
        uint base=(id/p.dim)*p.dim; float dot=0;
        for(uint j=0;j<p.dim;++j) dot+=a[base+j]*b[base+j];
        dot*=rsqrt(float(p.dim));
        o[id]=c[id%p.dim]*q4sig(sign(dot)*sqrt(max(abs(dot),1e-6f))); break;
    }
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
    case 21: { // average four raw index keys, no future or incomplete block
        uint base=(p.position-3)*128;
        o[id]=(a[base+id]+a[base+128+id]+a[base+256+id]+a[base+384+id])*0.25f; break;
    }
    case 22: { // pooled index scores over four query heads
        float score=0;
        for(uint h=0;h<4;++h) { float dot=0;for(uint j=0;j<128;++j) dot+=a[h*128+j]*b[id*128+j];score+=max(dot,0.0f); }
        o[id]=score*0.08838834764831845f; break;
    }
    case 23: { // exact top-512 block mask, chronological tie break
        if(id!=0) break;
        uint blocks=(p.position+1)/4;
        for(uint j=0;j<blocks;++j) o[j]=0;
        for(uint k=0;k<min(blocks,512u);++k) {
            float best=-INFINITY;uint winner=0xffffffffu;
            for(uint j=0;j<blocks;++j) if(o[j]==0 && (a[j]>best || (a[j]==best && j<winner))) {best=a[j];winner=j;}
            if(winner!=0xffffffffu) o[winner]=1;
        } break;
    }
    case 24: { // QK attention scores, cache in sequence order
        uint tokens=p.position+1, h=id/tokens, t=id%tokens;
        uint blocks=tokens/4;
        if(p.mode==0 && t/4<blocks && c[t/4]==0) {o[id]=-INFINITY;break;} // mode 1: every block kept
        float dot=0;for(uint j=0;j<256;++j) dot+=a[h*256+j]*b[t*512+(h/12)*256+j];
        o[id]=dot*0.0625f; break;
    }
    case 25: { // softmax each attention head, in place after scores complete
        uint n=p.position+1, base=id*n; float maximum=-INFINITY;
        for(uint j=0;j<n;++j) maximum=max(maximum,a[base+j]);
        float sum=0;for(uint j=0;j<n;++j) {o[base+j]=exp(a[base+j]-maximum);sum+=o[base+j];}
        for(uint j=0;j<n;++j) o[base+j]/=sum;break;
    }
    case 26: { // weighted values with sigmoid output gate
        uint h=id/256,j=id%256,n=p.position+1; float sum=0;
        for(uint t=0;t<n;++t) sum+=a[h*n+t]*b[t*512+(h/12)*256+j];
        o[id]=sum*q4sig(c[id]);break;
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
    }
}
