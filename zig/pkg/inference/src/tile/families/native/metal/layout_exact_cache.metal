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

// native-program:entry termite_exact_kv_head_major
// native-program:indent 11
kernel void termite_exact_kv_head_major(device const ushort *src [[buffer(0)]], device ushort *dst [[buffer(1)]], constant uint4 &shape [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  uint rows=shape.x, heads=shape.y, dim=shape.z, cap=shape.w; uint n=rows*heads*dim; if(gid>=n)return; uint d=gid%dim, q=gid/dim, h=q%heads, r=q/heads; dst[(h*cap+r)*dim+d]=src[(r*heads+h)*dim+d];
}
// native-program:entry termite_exact_kv_ring
// native-program:indent 11
kernel void termite_exact_kv_ring(device const ushort *src [[buffer(0)]], device ushort *dst [[buffer(1)]], constant uint4 &shape [[buffer(2)]], uint gid [[thread_position_in_grid]]) { uint rows=shape.x,width=shape.y,cap=shape.z,start=shape.w,n=rows*width;if(gid>=n)return;uint r=gid/width,c=gid%width;dst[((start+r)%cap)*width+c]=src[gid]; }
