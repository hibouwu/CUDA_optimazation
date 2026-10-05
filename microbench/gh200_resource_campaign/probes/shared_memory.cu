#include "../common/probe_runtime.cuh"
#include <functional>

__device__ unsigned long long shared_ns() {
  unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;
}
__device__ unsigned shared_smid() {
  unsigned x;asm volatile("mov.u32 %0, %%smid;":"=r"(x));return x;
}
template<int Bytes> __device__ __forceinline__ uint4 read_vector(unsigned address) {
  uint4 v=make_uint4(0,0,0,0);
  if constexpr(Bytes==4) asm volatile("ld.volatile.shared.u32 %0,[%1];":"=r"(v.x):"r"(address):"memory");
  if constexpr(Bytes==8) asm volatile("ld.volatile.shared.v2.u32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"r"(address):"memory");
  if constexpr(Bytes==16) asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(address):"memory");
  return v;
}
template<int Bytes> __device__ __forceinline__ void write_vector(unsigned address,uint4 v) {
  if constexpr(Bytes==4) asm volatile("st.volatile.shared.u32 [%0],%1;"::"r"(address),"r"(v.x):"memory");
  if constexpr(Bytes==8) asm volatile("st.volatile.shared.v2.u32 [%0],{%1,%2};"::"r"(address),"r"(v.x),"r"(v.y):"memory");
  if constexpr(Bytes==16) asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(address),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w):"memory");
}
// Mode 0 read, 1 write, 2 independent duplex. Broadcast is read-only.
template<int Bytes,int Mode,bool Broadcast>
__global__ void shared_access(int iterations,int stride,unsigned seed,gh::Stamp* stamp,unsigned* sums,unsigned* errors) {
  extern __shared__ __align__(16) unsigned data[];
  const unsigned tid=threadIdx.x;
  for(unsigned i=tid;i<8192;i+=blockDim.x) {
    data[i]=i*17u+seed;data[8192+i]=0xdeadbeefu;
  }
  __syncthreads();
  auto c0=static_cast<unsigned long long>(clock64());auto t0=shared_ns();
  __syncthreads();
  unsigned sum=0;
  #pragma unroll 1
  for(int it=0;it<iterations;++it) {
    #pragma unroll
    for(int q=0;q<8;++q) {
      unsigned base=Broadcast?(tid/32)*8+q:((tid+q*256)*stride*(Bytes/4))&8191u;
      if constexpr(Mode!=1) {
        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(data+base));
        auto v=read_vector<Bytes>(addr);sum+=v.x+v.y+v.z+v.w;
      }
      if constexpr(Mode!=0) {
        // Address-derived stores have no data dependency on the loaded values.
        auto v=make_uint4(base*29u+seed+7,(base+1)*29u+seed+7,(base+2)*29u+seed+7,(base+3)*29u+seed+7);
        unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(data+8192+base));
        write_vector<Bytes>(addr,v);
      }
    }
  }
  __syncthreads();
  auto t1=shared_ns();auto c1=static_cast<unsigned long long>(clock64());
  if(tid==0)stamp[0]={t0,t1,c0,c1,shared_smid()};
  unsigned bad=0;
  if constexpr(Mode!=0) {
    for(int q=0;q<8;++q)for(int lane=0;lane<Bytes/4;++lane) {
      unsigned index=((tid+q*256)*(Bytes/4))&8191u;
      bad+=data[8192+index+lane]!=(index+lane)*29u+seed+7;
    }
  }
  sums[tid]=sum;errors[tid]=bad;
}
struct SharedCase {
  std::string id;int bytes,mode,stride;bool broadcast;
  std::function<void(int,unsigned,gh::Stamp*,unsigned*,unsigned*)> launch;
  cudaFuncAttributes attributes{};
  int occupancy=0;
};
template<int Bytes,int Mode,bool Broadcast=false> SharedCase make_case(int stride=1) {
  SharedCase c;
  c.bytes=Bytes;c.mode=Mode;c.stride=stride;c.broadcast=Broadcast;
  c.id=std::string(Mode==0?"read":Mode==1?"write":"duplex")+"_w"+std::to_string(Bytes)+"_"+
       (Broadcast?"broadcast":"stride"+std::to_string(stride));
  GH_CUDA(cudaFuncSetAttribute(shared_access<Bytes,Mode,Broadcast>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536));
  GH_CUDA(cudaFuncGetAttributes(&c.attributes,shared_access<Bytes,Mode,Broadcast>));
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&c.occupancy,shared_access<Bytes,Mode,Broadcast>,256,65536));
  c.launch=[stride](int it,unsigned seed,gh::Stamp* s,unsigned* sums,unsigned* errors) {
    shared_access<Bytes,Mode,Broadcast><<<1,256,65536>>>(it,stride,seed,s,sums,errors);
  };
  return c;
}
int main(int argc,char**argv) try {
  auto device=gh::device();gh::emit_device(device);
  if(argc==2&&std::string(argv[1])=="device")return 0;
  if(argc!=4)throw std::runtime_error("CASE_ID ITERATIONS SEED");
  int iterations=gh::integer(argv[2],1,65536);unsigned seed=gh::integer(argv[3],1,1000000);
  if(iterations!=8192)throw std::runtime_error("frozen iteration mismatch");
  std::vector<SharedCase> all;
  for(int s:{1,2,4,8,16,32})all.push_back(make_case<4,0>(s));
  all.push_back(make_case<8,0>());all.push_back(make_case<16,0>());
  all.push_back(make_case<4,1>());all.push_back(make_case<8,1>());all.push_back(make_case<16,1>());
  all.push_back(make_case<4,0,true>());
  all.push_back(make_case<4,2>());all.push_back(make_case<8,2>());all.push_back(make_case<16,2>());
  auto it=std::find_if(all.begin(),all.end(),[&](const SharedCase& c){return c.id==argv[1];});
  if(it==all.end())throw std::runtime_error("unknown case");
  SharedCase c=*it;
  if(c.attributes.sharedSizeBytes!=0 || c.occupancy<1)throw std::runtime_error("unexpected allocation/occupancy");
  gh::Stamp* ds;unsigned *sums,*errors;
  GH_CUDA(cudaMalloc(&ds,sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&sums,256*sizeof(unsigned)));GH_CUDA(cudaMalloc(&errors,256*sizeof(unsigned)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  auto execute=[&]() {
    GH_CUDA(cudaEventRecord(begin));c.launch(iterations,seed,ds,sums,errors);GH_CUDA(cudaGetLastError());
    GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    gh::Observation o;o.stamps.resize(1);float ms;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));o.event_ms=ms;
    GH_CUDA(cudaMemcpy(o.stamps.data(),ds,sizeof(gh::Stamp),cudaMemcpyDeviceToHost));
    std::vector<unsigned> got(256),bad(256);
    GH_CUDA(cudaMemcpy(got.data(),sums,256*sizeof(unsigned),cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(bad.data(),errors,256*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(int t=0;t<256;++t) {
      gh::u64 expected=0;
      if(c.mode!=1)for(int q=0;q<8;++q)for(int lane=0;lane<c.bytes/4;++lane) {
        unsigned base=c.broadcast?(t/32)*8+q:((t+q*256)*c.stride*(c.bytes/4))&8191u;
        expected+=unsigned((base+lane)*17u+seed);
      }
      o.errors+=(got[t]!=unsigned(expected*iterations));o.errors+=bad[t];
    }
    o.checked_elements=256+(c.mode?256*8*(c.bytes/4):0);
    o.method="host_address_checksum_and_post_timer_per_word_store_check";
    o.input_conditions="read[i]=17*i+seed; write[i]=29*i+seed+7; poison; independent stores";
    gh::envelope(o);return o;
  };
  auto warm=gh::warmup(execute);auto o=execute();
  gh::u64 amount=gh::u64(256)*8*c.bytes*iterations,rb=c.mode==1?0:amount,wb=c.mode==0?0:amount;
  std::set<unsigned> unique;
  for(int t=0;t<256;++t)for(int q=0;q<8;++q)for(int l=0;l<c.bytes/4;++l) {
    unsigned base=c.broadcast?(t/32)*8+q:((t+q*256)*c.stride*(c.bytes/4))&8191u;unique.insert(base+l);
  }
  std::ostringstream extra;extra<<"\"dynamic_smem_bytes\":65536,\"registers_per_thread\":"<<c.attributes.numRegs
    <<",\"occupancy_limit_ctas_per_sm\":"<<c.occupancy<<",\"unique_bytes_per_direction_iteration\":"<<unique.size()*4
    <<",\"access_bytes\":"<<c.bytes<<",\"stride\":"<<c.stride<<",\"broadcast\":"<<(c.broadcast?"true":"false");
  gh::emit_trial(c.id,iterations,seed,256,"one_cta","byte",rb+wb,rb,wb,o,warm,extra.str());
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(ds));GH_CUDA(cudaFree(sums));GH_CUDA(cudaFree(errors));return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 2;}
