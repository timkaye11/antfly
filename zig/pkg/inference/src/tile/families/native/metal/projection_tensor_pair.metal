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

// native-program:entry termite_q4_0_pair_activation_mpp_m64_n32_k64
// native-program:indent 8
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct termite_metal_linear_params { uint rows,in_dim,out_dim,row_blocks,weight_offset,second_weight_offset; };
struct termite_metal_apply_activation_params { uint activation_kind,rows,dim; };
struct termite_block_q4_0 { half d; uchar qs[16]; };
inline float termite_gelu_tanh(float x) {
    if (!isfinite(x)) return 0.0f;
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    if (inner > 10.0f) return x;
    if (inner < -10.0f) return 0.0f;
    float y = 0.5f * x * (1.0f + tanh(inner));
    return isfinite(y) ? y : 0.0f;
}
inline float termite_silu(float x) {
    if (!isfinite(x)) return 0.0f;
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    float y = x / (1.0f + exp(-x));
    return isfinite(y) ? y : 0.0f;
}
inline float termite_gated_activation_product(float gate, float up, uint kind) {
    float activated = kind == 2u ? termite_silu(gate) : termite_gelu_tanh(gate);
    return activated == 0.0f ? 0.0f : activated * up;
}
inline void termite_dequantize_q4_0_4x4_half(device const termite_block_q4_0 *xb, short il, thread half4x4 &reg) {
    device const ushort *qs = reinterpret_cast<device const ushort *>(&xb->qs[0]);
    const float d1 = il ? (float(xb->d) / 16.0f) : float(xb->d);
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * float(xb->d);
    const ushort mask0 = il ? ushort(0x00F0) : ushort(0x000F);
    const ushort mask1 = mask0 << 8;
    for (short i = 0; i < 8; ++i) { reg[i / 2][2 * (i % 2) + 0] = half(d1 * float(qs[i] & mask0) + md); reg[i / 2][2 * (i % 2) + 1] = half(d2 * float(qs[i] & mask1) + md); }
}
template<uint NK,uint N,uint M> inline void termite_q4_pair_mpp_impl(device const float *input,device const uchar *wg,device const uchar *wu,device float *output,constant termite_metal_linear_params &p,constant termite_metal_apply_activation_params &ap,threadgroup char *scratch,ushort tid,uint3 tg){
 const uint col0=tg.x*N,row0=tg.y*M;
 threadgroup half *wa=(threadgroup half*)scratch,*xb=wa+2u*N*NK;
 auto a=tensor<threadgroup half,dextents<int,2>,tensor_inline>(xb,dextents<int,2>(NK,M));
 auto b=tensor<threadgroup half,dextents<int,2>,tensor_inline>(wa,dextents<int,2>(NK,2u*N));
 constexpr auto desc=matmul2d_descriptor(M,2u*N,NK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<desc,execution_simdgroups<4>> op;
 auto accum=op.template get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
 for(ushort i=0;i<accum.get_capacity();++i)accum[i]=0.0f;
 for(uint k0=0;k0<p.in_dim;k0+=NK){
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint wr=uint(tid)/2u;wr<2u*N;wr+=64u){device const uchar *weight=wr<N?wg:wu;device const termite_block_q4_0 *block=(device const termite_block_q4_0 *)(weight+(col0+(wr%N))*p.row_blocks*18u+k0/32u*18u);short il=short(tid)%2;
   for(uint kb=0u;kb<NK;kb+=32u){half4x4 v;termite_dequantize_q4_0_4x4_half(block+kb/32u,il,v);for(short j=0;j<16;j++)wa[wr*NK+kb+uint(il)*16u+uint(j)]=v[j/4][j%4];}}
  const uint xc=(uint(tid)%4u)*8u;
  for(uint row=uint(tid)/4u;row<M;row+=32u){uint xr=min(row,p.rows-row0-1u);for(uint kb=0u;kb<NK;kb+=32u)*(threadgroup half2x4 *)(xb+row*NK+kb+xc)=(half2x4)(*((device const float2x4 *)(input+(row0+xr)*p.in_dim+k0+kb+xc)));}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  op.run(a,b,accum);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 auto dest=tensor<threadgroup float,dextents<int,2>,tensor_inline>((threadgroup float*)scratch,dextents<int,2>(2u*N,M));accum.store(dest);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint idx=tid;idx<min(M,p.rows-row0)*N;idx+=128u){uint row=idx/N,col=idx%N;threadgroup float *v=(threadgroup float*)scratch;output[(row0+row)*p.out_dim+col0+col]=termite_gated_activation_product(v[row*2u*N+col],v[row*2u*N+col+N],ap.activation_kind);}
}
kernel void termite_q4_0_pair_activation_mpp_m64_n32_k64(
    device const float *input [[buffer(0)]], device const uchar *gate [[buffer(1)]],
    device const uchar *up [[buffer(2)]], device float *output [[buffer(3)]],
    constant termite_metal_linear_params &p [[buffer(4)]],
    constant termite_metal_apply_activation_params &ap [[buffer(5)]],
    threadgroup char *scratch [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    termite_q4_pair_mpp_impl<64u, 32u, 64u>(input, gate, up, output, p, ap, scratch, tid, tg);
}
// native-program:entry termite_q4_0_pair_activation_mm_m64_n32_nk64_aligned
// native-program:indent 11
inline void termite_q4_0_pair_activation_mm_m64_n32_nk64_impl(device const float *input,device const uchar *wg,device const uchar *wu,device float *output,constant termite_metal_linear_params&p,constant termite_metal_apply_activation_params&ap,threadgroup char*shmem,ushort tid,ushort sg,uint3 tg,bool tail){
 threadgroup half*sa=(threadgroup half*)shmem;threadgroup half*sb=(threadgroup half*)(shmem+4096u);uint r0=tg.x*32u,r1=tg.y*64u,nr1=tail?min(64u,p.rows-r1):64u;uint col=uint(tid)/4u,block=(uint(tid)/2u)&1u,il=uint(tid)&1u;device const termite_block_q4_0*xg=(device const termite_block_q4_0*)(wg+(r0+col)*p.row_blocks*18u);device const termite_block_q4_0*xu=(device const termite_block_q4_0*)(wu+(r0+col)*p.row_blocks*18u);simdgroup_half8x8 ma[2],mb[4];simdgroup_float8x8 ga[8],ua[8];for(short i=0;i<8;i++){ga[i]=make_filled_simdgroup_matrix<float,8>(0);ua[i]=make_filled_simdgroup_matrix<float,8>(0);}
 for(uint k=0;k<p.in_dim;k+=64u){half4x4 qg;termite_dequantize_q4_0_4x4_half(xg+block,short(il),qg);threadgroup_barrier(mem_flags::mem_threadgroup);TERMITE_FOR_UNROLL(short i=0;i<16;i++){uint sx=block*4u+2u*il+uint(i/8),sy=col/8u,lx=col%8u,ly=uint(i%8),ib=4u*sx+sy;sa[64u*ib+8u*ly+lx]=qg[i/4][i%4];}TERMITE_FOR_UNROLL(uint qm=0;qm<2u;qm++)TERMITE_FOR_UNROLL(uint qk=0;qk<2u;qk++){uint row=min(qm*32u+uint(tid)/4u,nr1-1u),ks=qk*32u+8u*(uint(tid)%4u),sx=qk*4u+uint(tid)%4u,sy=qm*4u+(uint(tid)/4u)/8u,ly=(uint(tid)/4u)%8u,ib=8u*sx+sy;device const float*yp=input+(r1+row)*p.in_dim+k+ks;*(threadgroup half2x4*)(sb+64u*ib+8u*ly)=(half2x4)(*((device const float2x4*)yp));}threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup const half*wa=sa+2u*64u*(uint(sg)&1u),*ia=sb+4u*64u*(uint(sg)>>1u);TERMITE_FOR_UNROLL(short ik=0;ik<8;ik++){TERMITE_FOR_UNROLL(short i=0;i<2;i++)simdgroup_load(ma[i],wa+64*i,8,0,false);TERMITE_FOR_UNROLL(short i=0;i<4;i++)simdgroup_load(mb[i],ia+64*i,8,0,false);TERMITE_FOR_UNROLL(short i=0;i<8;i++)simdgroup_multiply_accumulate(ga[i],mb[i/2],ma[i%2],ga[i]);wa+=4*64;ia+=8*64;}
 half4x4 qu;termite_dequantize_q4_0_4x4_half(xu+block,short(il),qu);threadgroup_barrier(mem_flags::mem_threadgroup);TERMITE_FOR_UNROLL(short i=0;i<16;i++){uint sx=block*4u+2u*il+uint(i/8),sy=col/8u,lx=col%8u,ly=uint(i%8),ib=4u*sx+sy;sa[64u*ib+8u*ly+lx]=qu[i/4][i%4];}threadgroup_barrier(mem_flags::mem_threadgroup);wa=sa+2u*64u*(uint(sg)&1u);ia=sb+4u*64u*(uint(sg)>>1u);TERMITE_FOR_UNROLL(short ik=0;ik<8;ik++){TERMITE_FOR_UNROLL(short i=0;i<2;i++)simdgroup_load(ma[i],wa+64*i,8,0,false);TERMITE_FOR_UNROLL(short i=0;i<4;i++)simdgroup_load(mb[i],ia+64*i,8,0,false);TERMITE_FOR_UNROLL(short i=0;i<8;i++)simdgroup_multiply_accumulate(ua[i],mb[i/2],ma[i%2],ua[i]);wa+=4*64;ia+=8*64;}xg+=2;xu+=2;}
 threadgroup_barrier(mem_flags::mem_threadgroup);threadgroup float*gt=(threadgroup float*)shmem,*ut=gt+2048u;uint tb=16u*(uint(sg)&1u)+32u*32u*(uint(sg)>>1u);for(short i=0;i<8;i++){uint mo=8u*uint(i%2)+8u*32u*uint(i/2);simdgroup_store(ga[i],gt+tb+mo,32u,0,false);simdgroup_store(ua[i],ut+tb+mo,32u,0,false);}threadgroup_barrier(mem_flags::mem_threadgroup);for(uint i=uint(tid);i<nr1*32u;i+=128u){uint row=i/32u,c=i-row*32u;output[(r1+row)*p.out_dim+r0+c]=termite_gated_activation_product(gt[i],ut[i],ap.activation_kind);}}
kernel void termite_q4_0_pair_activation_mm_m64_n32_nk64_aligned(device const float*x[[buffer(0)]],device const uchar*g[[buffer(1)]],device const uchar*u[[buffer(2)]],device float*y[[buffer(3)]],constant termite_metal_linear_params&p[[buffer(4)]],constant termite_metal_apply_activation_params&a[[buffer(5)]],threadgroup char*s[[threadgroup(0)]],ushort t[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){termite_q4_0_pair_activation_mm_m64_n32_nk64_impl(x,g,u,y,p,a,s,t,sg,tg,false);}
// native-program:entry termite_q4_0_pair_activation_mm_m64_n32_nk64_tail
// native-program:indent 11
kernel void termite_q4_0_pair_activation_mm_m64_n32_nk64_tail(device const float*x[[buffer(0)]],device const uchar*g[[buffer(1)]],device const uchar*u[[buffer(2)]],device float*y[[buffer(3)]],constant termite_metal_linear_params&p[[buffer(4)]],constant termite_metal_apply_activation_params&a[[buffer(5)]],threadgroup char*s[[threadgroup(0)]],ushort t[[thread_index_in_threadgroup]],ushort sg[[simdgroup_index_in_threadgroup]],uint3 tg[[threadgroup_position_in_grid]]){termite_q4_0_pair_activation_mm_m64_n32_nk64_impl(x,g,u,y,p,a,s,t,sg,tg,true);}
