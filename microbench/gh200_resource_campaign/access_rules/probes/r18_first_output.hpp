#pragma once
// Five local control-flow cuts, deferred until the consumer exits its work loop.
// No completion/arrival or hardware clock-enclosure claim is made by this helper.
struct R18FirstOutput { uint64_t stamp[10] = {}; };
__device__ __forceinline__ bool r18_first_writer() {
  unsigned c = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
  return (threadIdx.x==128 || threadIdx.x==256) &&
    (c==0 || c==1 || c==6 || c==7 || c==14 || c==15 || c==22 || c==23 ||
     c==30 || c==31 || c==38 || c==39);
}
template<int Point>
__device__ __forceinline__ void r18_first_stamp(R18FirstOutput* out, int tile) {
  if (out && tile==0 && r18_first_writer() && r18_first_output_observer) {
    out->stamp[2*Point]=v06_clock();
    out->stamp[2*Point+1]=v06_ns();
  }
}
__device__ __forceinline__ void r18_first_flush(R18FirstOutput const& value) {
  if (!r18_first_writer()) return;
  auto out=v06_base()+1296+(threadIdx.x/128-1)*11;
  out[0]=0x52313810u+r18_first_output_observer;
  #pragma unroll
  for (int i=0;i<10;++i) out[1+i]=value.stamp[i];
}
