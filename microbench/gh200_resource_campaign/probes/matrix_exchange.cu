#include "../common/probe_runtime.cuh"

__device__ __forceinline__ unsigned long long mx_ns(){unsigned long long x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__device__ __forceinline__ unsigned mx_sm(){unsigned x;asm volatile("mov.u32 %0, %%smid;":"=r"(x));return x;}
__host__ __device__ unsigned short mx_value(unsigned row,unsigned col,unsigned matrix,unsigned slot,unsigned seed){return static_cast<unsigned short>(1+row+11*col+97*matrix+997*slot+seed%8191);}

template<int N,bool Trans> __device__ __forceinline__ void mx_load(unsigned* r,unsigned address){
#define MX_LD(NUM,OUT,REGS) if constexpr(Trans) asm volatile("ldmatrix.sync.aligned.m8n8.x" #NUM ".trans.shared.b16 " REGS ", [%" #NUM "];":OUT:"r"(address):"memory"); else asm volatile("ldmatrix.sync.aligned.m8n8.x" #NUM ".shared.b16 " REGS ", [%" #NUM "];":OUT:"r"(address):"memory");
  if constexpr(N==1){MX_LD(1,"=r"(r[0]),"{%0}")}
  // Multiple asm operands need literal lists rather than a comma-containing macro argument.
  if constexpr(N==2){
    if constexpr(Trans)asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(r[0]),"=r"(r[1]):"r"(address):"memory");
    else asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];":"=r"(r[0]),"=r"(r[1]):"r"(address):"memory");
  }
  if constexpr(N==4){
    if constexpr(Trans)asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address):"memory");
    else asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(address):"memory");
  }
#undef MX_LD
}
template<int N,bool Trans> __device__ __forceinline__ void mx_store(const unsigned* r,unsigned address){
  if constexpr(N==1){
    if constexpr(Trans)asm volatile("stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%0], {%1};"::"r"(address),"r"(r[0]):"memory");
    else asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};"::"r"(address),"r"(r[0]):"memory");
  }
  if constexpr(N==2){
    if constexpr(Trans)asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1,%2};"::"r"(address),"r"(r[0]),"r"(r[1]):"memory");
    else asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};"::"r"(address),"r"(r[0]),"r"(r[1]):"memory");
  }
  if constexpr(N==4){
    if constexpr(Trans)asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};"::"r"(address),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]):"memory");
    else asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"::"r"(address),"r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]):"memory");
  }
}

// Mode 0 load, 1 store, 2 roundtrip. One full warp per CTA.
template<int N,bool Trans,int Mode> __global__ void matrix_exchange(int iterations,unsigned seed,gh::Stamp* stamp,unsigned* output){
  extern __shared__ __align__(16) unsigned short smem[];
  unsigned short* src=smem;unsigned short* dst=smem+2048;
  const unsigned lane=threadIdx.x;
  for(unsigned i=lane;i<2048;i+=32){unsigned slot=i/256,m=(i%256)/64,r=(i%64)/8,c=i%8;src[i]=mx_value(r,c,m,slot,seed);dst[i]=0xdead;}
  unsigned fragments[8][N]={},base[8][N]={};
  if constexpr(Mode==1){
    #pragma unroll
    for(int q=0;q<8;++q){
      #pragma unroll
      for(int m=0;m<N;++m){
        unsigned packed=0;
        #pragma unroll
        for(int h=0;h<2;++h){unsigned row=Trans?2*(lane%4)+h:lane/4,col=Trans?lane/4:2*(lane%4)+h;packed|=unsigned(mx_value(row,col,m,q,seed))<<(16*h);}
        base[q][m]=packed;
      }
    }
  }
  unsigned sum=0;
  __shared__ volatile unsigned drain[32];__shared__ unsigned long long t0,c0;
  __syncthreads();if(lane==0){t0=mx_ns();c0=clock64();}__syncthreads();
  #pragma unroll 1
  for(int it=0;it<iterations;++it){
    #pragma unroll
    for(int q=0;q<8;++q){
      unsigned offset=q*256+((lane/8)%N)*64+(lane%8)*8;
      if constexpr(Mode!=1){mx_load<N,Trans>(fragments[q],static_cast<unsigned>(__cvta_generic_to_shared(src+offset)));}
      else {
        #pragma unroll
        for(int m=0;m<N;++m){unsigned mask=unsigned(static_cast<unsigned short>(it+1));fragments[q][m]=base[q][m]^(mask|(mask<<16));}
      }
      if constexpr(Mode!=0)mx_store<N,Trans>(fragments[q],static_cast<unsigned>(__cvta_generic_to_shared(dst+offset)));
      if constexpr(Mode==2)asm volatile("bar.warp.sync 0xffffffff;":::"memory");
      if constexpr(Mode!=1){
        #pragma unroll
        for(int m=0;m<N;++m)sum+=fragments[q][m];
      }
    }
  }
  drain[lane]=Mode==1?fragments[7][N-1]:sum;
  __syncthreads();if(lane==0){auto c1=static_cast<unsigned long long>(clock64());auto t1=mx_ns();*stamp={t0,t1,c0,c1,mx_sm()};}
  output[lane]=sum;
  #pragma unroll
  for(int q=0;q<8;++q){
    #pragma unroll
    for(int m=0;m<N;++m)output[32+(lane*8+q)*N+m]=fragments[q][m];
  }
  for(unsigned i=lane;i<2048;i+=32)output[32+32*8*N+i]=dst[i];
}

template<int Threads,int Streams> __global__ void warp_exchange(int steps,unsigned seed,gh::Stamp* stamp,unsigned* output){
  unsigned values[Streams];const unsigned tid=threadIdx.x,lane=tid%32,warp=tid/32;
  #pragma unroll
  for(int s=0;s<Streams;++s)values[s]=seed+17*lane+131*s+8191*warp;
  __shared__ volatile unsigned drain[Threads];__shared__ unsigned long long t0,c0;
  __syncthreads();if(tid==0){t0=mx_ns();c0=clock64();}__syncthreads();
  // Formal runs have steps=8192*8: one loop-control operation per eight steps.
  // The remainder uses the same routing for short checks at 1, 3 and 33 steps.
  #pragma unroll 1
  for(int iteration=0;iteration<steps/8;++iteration){
    #pragma unroll
    for(int q=0;q<8;++q){
      #pragma unroll
      for(int s=0;s<Streams;++s)
        asm volatile("shfl.sync.idx.b32 %0, %0, %1, 31, 0xffffffff;"
                     :"+r"(values[s]):"r"((lane+1)%32));
    }
  }
  #pragma unroll 1
  for(int step=0;step<steps%8;++step){
    #pragma unroll
    for(int s=0;s<Streams;++s)
      asm volatile("shfl.sync.idx.b32 %0, %0, %1, 31, 0xffffffff;"
                   :"+r"(values[s]):"r"((lane+1)%32));
  }
  unsigned sum=0;
  #pragma unroll
  for(int s=0;s<Streams;++s)sum+=values[s];
  drain[tid]=sum;__syncthreads();if(tid==0){auto c1=static_cast<unsigned long long>(clock64());auto t1=mx_ns();*stamp={t0,t1,c0,c1,mx_sm()};}
  #pragma unroll
  for(int s=0;s<Streams;++s)output[tid*Streams+s]=values[s];
}

struct MXCase {std::string id;void(*kernel)(int,unsigned,gh::Stamp*,unsigned*);int matrices,mode,threads,streams;bool transpose;};
template<int N,bool T,int M> void mx_add(std::vector<MXCase>& out){out.push_back({std::string(M==0?"load":M==1?"store":"roundtrip")+"_x"+std::to_string(N)+(T?"_trans":"_normal"),matrix_exchange<N,T,M>,N,M,32,0,T});}
template<int N> void mx_add_n(std::vector<MXCase>& out){mx_add<N,false,0>(out);mx_add<N,true,0>(out);mx_add<N,false,1>(out);mx_add<N,true,1>(out);mx_add<N,false,2>(out);mx_add<N,true,2>(out);}
std::vector<unsigned> mx_reference(const MXCase& c,int length,unsigned seed){
  if(c.mode==3){std::vector<unsigned> result(c.threads*c.streams);for(int t=0;t<c.threads;++t)for(int s=0;s<c.streams;++s)result[t*c.streams+s]=seed+17*((t%32+length)%32)+131*s+8191*(t/32);return result;}
  const int n=c.matrices;std::vector<unsigned> result(32+32*8*n+2048,0);
  for(int lane=0;lane<32;++lane){unsigned sum=0;for(int q=0;q<8;++q)for(int m=0;m<n;++m){unsigned packed=0;for(int h=0;h<2;++h){unsigned r=c.transpose?2*(lane%4)+h:lane/4,col=c.transpose?lane/4:2*(lane%4)+h;unsigned value=mx_value(r,col,m,q,seed);if(c.mode==1)value^=static_cast<unsigned short>(length);packed|=value<<(16*h);}result[32+(lane*8+q)*n+m]=packed;sum+=packed;}result[lane]=c.mode==1?0:sum*length;}
  for(int i=0;i<2048;++i){int q=i/256,m=i%256/64,r=i%64/8,col=i%8;unsigned value=0xdead;if(c.mode!=0 && m<n){value=mx_value(r,col,m,q,seed);if(c.mode==1)value^=static_cast<unsigned short>(length);}result[32+32*8*n+i]=value;}
  return result;
}
int mx_formal_main(int argc,char**argv)try{
  auto device=gh::device();gh::emit_device(device);if(argc==2&&std::string(argv[1])=="device")return 0;
  if(argc!=4)throw std::runtime_error("CASE_ID ITERATIONS SEED required");
  int iterations=gh::integer(argv[2],8192,8192);unsigned seed=gh::integer(argv[3],0,4294967295ull);
  std::vector<MXCase> cases;mx_add_n<1>(cases);mx_add_n<2>(cases);mx_add_n<4>(cases);
  cases.push_back({"shuffle_t32_streams1",warp_exchange<32,1>,0,3,32,1,false});cases.push_back({"shuffle_t32_streams4",warp_exchange<32,4>,0,3,32,4,false});
  cases.push_back({"shuffle_t128_streams1",warp_exchange<128,1>,0,3,128,1,false});cases.push_back({"shuffle_t128_streams4",warp_exchange<128,4>,0,3,128,4,false});
  auto found=std::find_if(cases.begin(),cases.end(),[&](auto c){return c.id==argv[1];});if(found==cases.end())throw std::runtime_error("unknown case");auto c=*found;
  int dynamic=c.mode==3?0:8192;cudaFuncAttributes attr{};GH_CUDA(cudaFuncGetAttributes(&attr,reinterpret_cast<const void*>(c.kernel)));
  if(attr.localSizeBytes)throw std::runtime_error("unexpected local/spill");int occupancy=0;GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.kernel,c.threads,dynamic));if(occupancy<1)throw std::runtime_error("no resident CTA");
  int length=c.mode==3?iterations*8:iterations;auto expected=mx_reference(c,length,seed);
  gh::Stamp* stamp;unsigned* output;GH_CUDA(cudaMalloc(&stamp,sizeof(gh::Stamp)));GH_CUDA(cudaMalloc(&output,expected.size()*sizeof(unsigned)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  auto execute=[&](int count,bool measured=true){auto want=mx_reference(c,count,seed);std::vector<unsigned> got(want.size());GH_CUDA(cudaMemset(output,0xff,want.size()*4));void* args[]={&count,&seed,&stamp,&output};
    GH_CUDA(cudaEventRecord(begin));GH_CUDA(cudaLaunchKernel(reinterpret_cast<const void*>(c.kernel),dim3(1),dim3(c.threads),args,dynamic,nullptr));GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
    gh::Observation o;o.stamps.resize(1);float ms=0;GH_CUDA(cudaEventElapsedTime(&ms,begin,end));o.event_ms=ms;GH_CUDA(cudaMemcpy(o.stamps.data(),stamp,sizeof(gh::Stamp),cudaMemcpyDeviceToHost));GH_CUDA(cudaMemcpy(got.data(),output,got.size()*4,cudaMemcpyDeviceToHost));for(size_t i=0;i<got.size();++i)o.errors+=(got[i]!=want[i]);o.checked_elements=got.size();o.method="full_fragments_checksum_output_and_padding_v1";o.input_conditions="coordinate+seed matrix uint16; step-XOR independent stores; lane/warp/stream shuffle seed";if(o.errors)throw std::runtime_error("matrix/exchange full-output mismatch");if(measured)gh::envelope(o);return o;};
  if(c.mode==3){for(int count:{1,3,33})execute(count,false);}else{execute(1,false);execute(2,false);}
  auto launch=[&](){return execute(length);};auto warm=gh::warmup(launch);auto o=launch();
  gh::u64 direction=c.mode==3?0:gh::u64(iterations)*8*c.matrices*128,read=(c.mode==0||c.mode==2)?direction:0,write=(c.mode==1||c.mode==2)?direction:0;
  gh::u64 work=c.mode==3?gh::u64(iterations)*8*c.streams*(c.threads/32):read+write;
  std::ostringstream extra;extra<<"\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes<<",\"dynamic_smem_bytes\":"<<dynamic<<",\"local_size_bytes\":"<<attr.localSizeBytes<<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"nonuniform_short_checks\":"<<(c.mode==3?"[1,3,33]":"[1,2]")<<",\"short_check_errors\":0";
  gh::emit_trial(c.id,iterations,seed,c.threads,"one_cta",c.mode==3?"operation":"byte",work,read,write,o,warm,extra.str());
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(stamp));GH_CUDA(cudaFree(output));return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}

// The independently reviewed short interface is host-only. Everything above
// retains the original device functions and formal branch (main renamed only).
#include "../common/word_artifacts.hpp"

struct MXShortCheck {
  int length;
  unsigned long long checked=0,errors=0;
  bool completed=false,stamp_complete=false;
  std::string artifact_path,artifact_sha256;
};

static std::string mx_symbol(const MXCase& c) {
  if(c.mode==3)
    return "_Z13warp_exchangeILi"+std::to_string(c.threads)+"ELi"+
      std::to_string(c.streams)+"EEvijPN2gh5StampEPj";
  return "_Z15matrix_exchangeILi"+std::to_string(c.matrices)+"ELb"+
    std::to_string(int(c.transpose))+"ELi"+std::to_string(c.mode)+
    "EEvijPN2gh5StampEPj";
}

static std::vector<MXCase> mx_short_cases() {
  std::vector<MXCase> cases;
  mx_add_n<1>(cases);mx_add_n<2>(cases);mx_add_n<4>(cases);
  cases.push_back({"shuffle_t32_streams1",warp_exchange<32,1>,0,3,32,1,false});
  cases.push_back({"shuffle_t32_streams4",warp_exchange<32,4>,0,3,32,4,false});
  cases.push_back({"shuffle_t128_streams1",warp_exchange<128,1>,0,3,128,1,false});
  cases.push_back({"shuffle_t128_streams4",warp_exchange<128,4>,0,3,128,4,false});
  return cases;
}

static void mx_emit_validation(const MXCase& c,const std::string& profile,
    unsigned seed,const cudaFuncAttributes& attr,int occupancy,int dynamic,
    const std::vector<MXShortCheck>& checks,bool failed) {
  unsigned long long errors=failed?1:0;
  for(const auto& check:checks)errors+=check.errors;
  const size_t count=c.mode==3?c.threads*c.streams:32+256*c.matrices+2048;
  std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"
    <<gh::quote(c.id)<<",\"profile_id\":"<<gh::quote(profile)<<",\"seed\":"<<seed
    <<",\"scope\":\"one_cta\",\"threads\":"<<c.threads<<",\"blocks\":1,\"errors\":"<<errors
    <<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[";
  for(size_t i=0;i<checks.size();++i) {
    std::cout<<(i?",":"")<<"{\"launch_index\":"<<i<<",\"iterations\":"<<checks[i].length
      <<",\"input_profile\":\"coordinate_seed_nonuniform\",\"threads\":"<<c.threads<<",\"blocks\":1}";
  }
  std::cout<<"],\"checks\":[";
  for(size_t i=0;i<checks.size();++i) {
    const auto& check=checks[i];
    std::cout<<(i?",":"")<<"{\"launch_index\":"<<i
      <<",\"reference_model\":\"matrix_exchange_python_coordinate_reference_v1\""
      <<",\"reference_sha256\":\"a3829a670b078ffa38a9c2c8b82de562f1cfba081775dc46c528d2e4a3f330bc\""
      <<",\"comparison\":\"exact\",\"tolerance_id\":null,\"checked_elements\":"<<check.checked
      <<",\"expected_elements\":"<<count<<",\"errors\":"<<check.errors
      <<",\"completed\":"<<(check.completed?"true":"false")
      <<",\"verified_CTA_ids\":"<<(check.stamp_complete?"[0]":"[]")<<",\"output_artifacts\":[";
    if(!check.artifact_sha256.empty())
      std::cout<<"{\"path\":"<<gh::quote(check.artifact_path)<<",\"sha256\":"<<gh::quote(check.artifact_sha256)
        <<",\"dtype\":\"uint32\",\"shape\":["<<count<<"],\"evidence_kind\":\"full_values\"}";
    std::cout<<"]}";
  }
  std::cout<<"],\"resource_identity\":{\"kernel_symbol\":"<<gh::quote(mx_symbol(c))
    <<",\"registers_per_thread\":"<<attr.numRegs<<",\"static_smem_bytes\":"<<attr.sharedSizeBytes
    <<",\"dynamic_smem_bytes\":"<<dynamic<<",\"local_size_bytes\":"<<attr.localSizeBytes
    <<",\"occupancy_limit_ctas_per_sm\":"<<occupancy<<",\"extensions\":{}}}\n";
}

static int mx_validate_main(int argc,char** argv) try {
  const auto device=gh::device();gh::emit_device(device);
  if(argc!=5)throw std::runtime_error("validate-only CASE_ID PROFILE_ID 3 required");
  unsigned seed=gh::integer(argv[4],3,3);
  const auto cases=mx_short_cases();
  const auto found=std::find_if(cases.begin(),cases.end(),[&](const MXCase& c){return c.id==argv[2];});
  if(found==cases.end())throw std::runtime_error("unknown short case");
  const auto& c=*found;
  const std::string profile=c.mode==3?"shuffle_short_1_8_33_v1":"matrix_short_1_2_v1";
  if(profile!=argv[3])throw std::runtime_error("profile does not apply to selected case");
  const std::vector<int> lengths=c.mode==3?std::vector<int>{1,8,33}:std::vector<int>{1,2};
  const int dynamic=c.mode==3?0:8192;
  cudaFuncAttributes attr{};
  GH_CUDA(cudaFuncGetAttributes(&attr,reinterpret_cast<const void*>(c.kernel)));
  int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,c.kernel,c.threads,dynamic));
  if(occupancy<1||attr.localSizeBytes)throw std::runtime_error("short target resource/spill failure");
  const auto count=mx_reference(c,lengths.front(),seed).size();
  gh::Stamp* stamp=nullptr;unsigned* output=nullptr;
  GH_CUDA(cudaMalloc(&stamp,sizeof(gh::Stamp)));
  GH_CUDA(cudaMalloc(&output,count*sizeof(unsigned)));
  cudaEvent_t begin,end;
  GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  std::vector<MXShortCheck> checks;unsigned details=0;
  try {
    for(const int requested:lengths) {
      auto want=mx_reference(c,requested,seed);
      std::vector<unsigned> poison(want.size()),got(want.size());
      for(size_t i=0;i<want.size();++i)poison[i]=want[i]^~unsigned(0);
      GH_CUDA(cudaMemcpy(output,poison.data(),poison.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
      GH_CUDA(cudaMemset(stamp,0xff,sizeof(gh::Stamp)));
      GH_CUDA(cudaEventRecord(begin));
      // Record only the target actually attempted; setup failures do not invent a launch.
      checks.push_back(MXShortCheck{requested});
      auto& check=checks.back();int actual_count=requested;
      void* args[]={&actual_count,&seed,&stamp,&output};
      GH_CUDA(cudaLaunchKernel(reinterpret_cast<const void*>(c.kernel),dim3(1),dim3(c.threads),args,dynamic,nullptr));
      GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
      gh::Stamp completed{};
      GH_CUDA(cudaMemcpy(&completed,stamp,sizeof(completed),cudaMemcpyDeviceToHost));
      GH_CUDA(cudaMemcpy(got.data(),output,got.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
      const auto unset=std::numeric_limits<gh::u64>::max();
      check.stamp_complete=completed.begin_ns!=unset&&completed.end_ns!=unset&&
        completed.begin_cycle!=unset&&completed.end_cycle!=unset&&completed.smid!=~unsigned(0);
      if(!check.stamp_complete) {
        ++check.errors;
        if(details++<8)std::cerr<<"launch="<<checks.size()-1<<" incomplete CTA stamp\n";
      }
      for(size_t i=0;i<got.size();++i) {
        if(got[i]!=want[i]) {
          ++check.errors;
          if(details++<8)std::cerr<<"launch="<<checks.size()-1<<" index="<<i
            <<" actual="<<got[i]<<" expected="<<want[i]<<'\n';
        }
        ++check.checked;
      }
      check.artifact_path="mx_output_"+std::to_string(checks.size()-1)+".u32le";
      check.artifact_sha256=word_artifacts::write_words(check.artifact_path,got);
      check.completed=check.stamp_complete;
      if(check.errors)throw std::runtime_error("short full-output check failed");
    }
  } catch(const std::exception& error) {
    mx_emit_validation(c,profile,seed,attr,occupancy,dynamic,checks,true);
    std::cerr<<error.what()<<'\n';return 2;
  }
  mx_emit_validation(c,profile,seed,attr,occupancy,dynamic,checks,false);
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(stamp));GH_CUDA(cudaFree(output));return 0;
} catch(const std::exception& error) {
  std::cerr<<error.what()<<'\n';return 2;
}

int main(int argc,char** argv) {
  if(argc>1&&std::string(argv[1])=="validate-only")return mx_validate_main(argc,argv);
  return mx_formal_main(argc,argv);
}
