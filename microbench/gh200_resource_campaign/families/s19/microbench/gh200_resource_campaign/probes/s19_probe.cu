#include "../common/probe_runtime.cuh"
#include "../common/word_artifacts.hpp"
#include "../common/s19_reference.hpp"
#include "../common/s19_reference_id.hpp"
#include <cstring>

__device__ __forceinline__ void issue_tile(const unsigned* input,unsigned* shared,int t,int slot) {
  #pragma unroll
  for(int q=0;q<4;++q) {
    int offset=int(threadIdx.x)*4+q*512;
    unsigned dst=unsigned(__cvta_generic_to_shared(shared+slot*2048+offset));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(dst),"l"(input+t*2048+offset):"memory");
  }
  asm volatile("cp.async.commit_group;":::"memory");
}
__device__ __forceinline__ void wait_oldest(int pending) {
  if(pending==3)asm volatile("cp.async.wait_group 3;":::"memory");
  else if(pending==2)asm volatile("cp.async.wait_group 2;":::"memory");
  else if(pending==1)asm volatile("cp.async.wait_group 1;":::"memory");
  else asm volatile("cp.async.wait_group 0;":::"memory");
}

// One symbol keeps register allocation and static/dynamic resources identical
// across the five controls and three depths. Runtime branches are uniform.
extern "C" __global__ void s19_pipeline(const unsigned* input,unsigned* output,
    unsigned* digest,unsigned* final_slots,unsigned* trace_input,unsigned* trace_c,
    gh::Stamp* stamps,int mode,int stages,int tiles,int iterations) {
  extern __shared__ __align__(16) unsigned storage[];
  int tid=int(threadIdx.x), column=tid%32, row0=tid/32;
  for(int i=tid;i<8192;i+=128)storage[i]=s19::poison;
  __syncthreads();
  if(mode==0) {
    int active=stages<tiles?stages:tiles;
    for(int slot=0;slot<active;++slot)issue_tile(input,storage,slot,slot);
    wait_oldest(0);__syncthreads();
  }
  float accum[8]={};unsigned checksum=0;
  gh::Stamp stamp{};
  if(tid==0) {
    asm volatile("mov.u32 %0, %%smid;":"=r"(stamp.smid));
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(stamp.begin_ns)::"memory");
    asm volatile("mov.u64 %0, %%clock64;":"=l"(stamp.begin_cycle)::"memory");
  }
  __syncthreads();
  #pragma unroll 1
  for(int rep=0;rep<iterations;++rep) {
    #pragma unroll
    for(int i=0;i<8;++i)accum[i]=0;
    int active=stages<tiles?stages:tiles;
    if(mode==1 || mode>=3) for(int t=0;t<active;++t)issue_tile(input,storage,t,t);
    #pragma unroll 1
    for(int t=0;t<tiles;++t) {
      int slot=t%stages;
      if(mode==2) {issue_tile(input,storage,t,slot);wait_oldest(0);}
      if(mode==1 || mode>=3) {int later=tiles-t-1;wait_oldest(later<stages-1?later:stages-1);}
      __syncthreads();
      volatile unsigned* words=storage+slot*2048;
      if(trace_input)for(int i=tid;i<2048;i+=128)trace_input[(rep*tiles+t)*2048+i]=words[i];
      if(mode==1) {
        #pragma unroll
        for(int q=0;q<4;++q) {
          #pragma unroll
          for(int j=0;j<4;++j)checksum+=words[((tid+1)%128)*4+q*512+j];
        }
      } else {
        volatile float* a=reinterpret_cast<volatile float*>(storage+slot*2048);
        volatile float* b=a+1024;
        #pragma unroll 1
        for(int k=0;k<32;++k) {
          float bv=b[k*32+column];
          #pragma unroll
          for(int i=0;i<8;++i) {
            float av=a[(row0+4*i)*32+k];
            asm volatile("fma.rn.f32 %0, %1, %2, %0;":"+f"(accum[i]):"f"(av),"f"(bv));
          }
        }
      }
      if(trace_c) {
        #pragma unroll
        for(int i=0;i<8;++i)trace_c[(rep*tiles+t)*1024+(row0+4*i)*32+column]=__float_as_uint(accum[i]);
      }
      // All SMEM readers finish before any thread reuses this slot.
      __syncthreads();
      if((mode==1 || mode>=3) && t+stages<tiles)issue_tile(input,storage,t+stages,slot);
    }
    wait_oldest(0);
    if(mode==4) {
      #pragma unroll
      for(int i=0;i<8;++i) {
        float value;
        float bias=float(column-16)/16;
        asm volatile("fma.rn.f32 %0, %1, 0f3f000000, %2;":"=f"(value):"f"(accum[i]),"f"(bias));
        unsigned* address=output+(row0+4*i)*32+column;
        unsigned bits=__float_as_uint(value);
        asm volatile("st.global.wb.u32 [%0], %1;"::"l"(address),"r"(bits):"memory");
      }
      // Device-scope ordering followed by a CTA join includes all writers.
      __threadfence();
    }
    #pragma unroll
    for(int i=0;i<8;++i)asm volatile("" : "+f"(accum[i]) :: "memory");
    asm volatile("" : "+r"(checksum) :: "memory");
    __syncthreads();
  }
  if(tid==0) {
    asm volatile("mov.u64 %0, %%clock64;":"=l"(stamp.end_cycle)::"memory");
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(stamp.end_ns)::"memory");
    stamps[0]=stamp;
  }
  __syncthreads();
  // Common correctness artifacts are outside the clock64 interval.
  if(mode!=4) {
    #pragma unroll
    for(int i=0;i<8;++i)output[(row0+4*i)*32+column]=__float_as_uint(accum[i]);
  }
  digest[tid]=checksum;
  for(int i=tid;i<8192;i+=128)final_slots[i]=storage[i];
}

struct Buffer {
  unsigned* pointer=nullptr;std::vector<unsigned> host;
  explicit Buffer(size_t n):host(n+16,s19::poison) {
    for(int i=0;i<8;++i){host[i]=0xd1900000u+i;host[host.size()-8+i]=0xd1910000u+i;}
    GH_CUDA(cudaMalloc(&pointer,host.size()*4));upload();
  }
  ~Buffer(){if(pointer)cudaFree(pointer);}
  unsigned* data(){return pointer+8;}
  void upload(){GH_CUDA(cudaMemcpy(pointer,host.data(),host.size()*4,cudaMemcpyHostToDevice));}
  std::vector<unsigned> download() {
    GH_CUDA(cudaMemcpy(host.data(),pointer,host.size()*4,cudaMemcpyDeviceToHost));
    return {host.begin()+8,host.end()-8};
  }
  void check_guards() const {
    for(int i=0;i<8;++i)if(host[i]!=0xd1900000u+unsigned(i)||host[host.size()-8+i]!=0xd1910000u+unsigned(i))
      throw std::runtime_error("S19 artifact guard changed");
  }
  std::vector<unsigned> guards() const {
    std::vector<unsigned> values(host.begin(),host.begin()+8);
    values.insert(values.end(),host.end()-8,host.end());return values;
  }
};
struct Artifact {std::string name,sha;size_t words;};
static void save(std::vector<Artifact>& artifacts,const std::string& name,const std::vector<unsigned>& data) {
  artifacts.push_back({name,word_artifacts::write_words(name,data),data.size()});
}
static void compare(const std::vector<unsigned>& actual,const std::vector<unsigned>& expected,const char* name) {
  if(actual.size()!=expected.size())throw std::runtime_error(std::string(name)+" shape");
  for(size_t i=0;i<actual.size();++i)if(actual[i]!=expected[i])
    throw std::runtime_error(std::string(name)+" mismatch at "+std::to_string(i));
}
int main(int argc,char** argv) try {
  // No command grants formal eligibility; suite admission remains external.
  if(argc<2)throw std::runtime_error("device | capability | validate-only CASE PROFILE SEED | diagnose CASE ITERATIONS SEED periodic");
  std::string command=argv[1];
  if(command!="device"&&command!="capability"&&command!="validate-only"&&command!="diagnose")throw std::runtime_error("unknown command");
  bool validation=command=="validate-only",diagnostic=command=="diagnose";
  if(argc!=(validation?5:diagnostic?6:2))throw std::runtime_error("argument count");
  const auto device=gh::device();gh::emit_device(device);
  if(command=="device")return 0;
  cudaFuncAttributes attributes{};
  GH_CUDA(cudaFuncGetAttributes(&attributes,s19_pipeline));
  int occupancy=0;
  GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,s19_pipeline,128,32768));
  if(occupancy<1 || attributes.localSizeBytes!=0)throw std::runtime_error("S19 requires legal occupancy and no local spill");
  std::string resource="\"resource_identity\":{\"kernel_symbol\":\"s19_pipeline\",\"registers_per_thread\":"+std::to_string(attributes.numRegs)+
    ",\"static_smem_bytes\":"+std::to_string(attributes.sharedSizeBytes)+",\"dynamic_smem_bytes\":32768,\"local_size_bytes\":"+std::to_string(attributes.localSizeBytes)+
    ",\"occupancy_limit_ctas_per_sm\":"+std::to_string(occupancy)+",\"extensions\":{}}";
  if(command=="capability"){std::cout<<"{\"type\":\"s19_capability\",\"GPU_target_launches\":0,"<<resource<<"}\n";return 0;}
  auto c=s19::parse(argv[2]);std::string profile_id=validation?argv[3]:"diagnostic";
  int iterations=validation?(profile_id=="periodic_1"||profile_id=="tagged_1"?1:3):int(gh::integer(argv[3],1,1024));
  if(validation&&profile_id!="periodic_1"&&profile_id!="periodic_3"&&profile_id!="tagged_1"&&profile_id!="tagged_3")throw std::runtime_error("short profile");
  unsigned seed=unsigned(gh::integer(argv[4],0,4294967295ull));
  std::string profile=validation?(profile_id.rfind("tagged",0)==0?"tagged":"periodic"):argv[5];
  if(profile!="periodic"&&profile!="tagged")throw std::runtime_error("input profile");
  if(diagnostic&&profile!="periodic")throw std::runtime_error("diagnostic requires periodic input");
  auto input=s19::input(c,seed,profile=="tagged");
  auto expected=s19::expected(c,iterations,seed,profile=="tagged",validation);
  Buffer in(input.size()),out(1024),digest(128),slots(8192);
  Buffer trace_input(validation?size_t(iterations)*c.tiles*2048:0),trace_c(validation?size_t(iterations)*c.tiles*1024:0);
  std::copy(input.begin(),input.end(),in.host.begin()+8);in.upload();
  gh::Stamp *stamp=nullptr;GH_CUDA(cudaMalloc(&stamp,sizeof(gh::Stamp)));GH_CUDA(cudaMemset(stamp,0xff,sizeof(gh::Stamp)));
  cudaEvent_t begin,end;GH_CUDA(cudaEventCreate(&begin));GH_CUDA(cudaEventCreate(&end));
  GH_CUDA(cudaEventRecord(begin));
  s19_pipeline<<<1,128,32768>>>(in.data(),out.data(),digest.data(),slots.data(),validation?trace_input.data():nullptr,
      validation?trace_c.data():nullptr,stamp,c.mode,c.stages,c.tiles,iterations);
  GH_CUDA(cudaGetLastError());GH_CUDA(cudaEventRecord(end));GH_CUDA(cudaEventSynchronize(end));
  float elapsed=0;GH_CUDA(cudaEventElapsedTime(&elapsed,begin,end));
  gh::Stamp host_stamp{};GH_CUDA(cudaMemcpy(&host_stamp,stamp,sizeof(host_stamp),cudaMemcpyDeviceToHost));
  auto output=out.download(),checksums=digest.download(),final_slots=slots.download();
  auto ti=trace_input.download(),tc=trace_c.download();auto original=in.download();
  std::vector<Artifact> artifacts;
  save(artifacts,"input.u32le",original);save(artifacts,"output.u32le",output);save(artifacts,"digest.u32le",checksums);save(artifacts,"slots.u32le",final_slots);
  if(validation){save(artifacts,"trace_input.u32le",ti);save(artifacts,"trace_c.u32le",tc);}
  std::vector<unsigned> guards;
  for(const Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c}) {
    auto g=buffer->guards();guards.insert(guards.end(),g.begin(),g.end());
  }
  save(artifacts,"guards.u32le",guards);
  std::vector<unsigned> stamp_words;
  for(gh::u64 value:{host_stamp.begin_ns,host_stamp.end_ns,host_stamp.begin_cycle,host_stamp.end_cycle}) {
    stamp_words.push_back(unsigned(value));stamp_words.push_back(unsigned(value>>32));
  }
  stamp_words.push_back(host_stamp.smid);stamp_words.push_back(s19::bits(elapsed));save(artifacts,"stamp.u32le",stamp_words);
  // Persist complete values before reporting a mismatch so failed launches can be replayed.
  unsigned errors=0;std::string failure;
  try {
    for(const Buffer* buffer:{&in,&out,&digest,&slots,&trace_input,&trace_c})buffer->check_guards();
    compare(original,input,"input");compare(output,expected.output,"output");compare(checksums,expected.digest,"digest");compare(final_slots,expected.slots,"slots");
    if(validation){compare(ti,expected.trace_input,"trace_input");compare(tc,expected.trace_c,"trace_c");}
    if(!std::isfinite(elapsed)||elapsed<=0||host_stamp.smid==0xffffffffu||host_stamp.end_cycle<=host_stamp.begin_cycle||host_stamp.end_ns<host_stamp.begin_ns||
       (!validation&&host_stamp.end_ns==host_stamp.begin_ns)||
       double(host_stamp.end_ns-host_stamp.begin_ns)>double(elapsed)*1.05e6)throw std::runtime_error("timer/event boundary");
  }catch(const std::exception& e){errors=1;failure=e.what();}
  if(validation) {
    size_t checked=input.size()+1024+128+8192+ti.size()+tc.size()+96+10;
    std::cout<<"{\"schema_version\":2,\"validation_schema_version\":1,\"type\":\"validation\",\"case_id\":"<<gh::quote(c.id)
      <<",\"profile_id\":"<<gh::quote(profile_id)<<",\"seed\":"<<seed<<",\"scope\":\"one_cta\",\"threads\":128,\"blocks\":1,\"errors\":"<<errors
      <<",\"performance_eligible\":false,\"warmup_executed\":false,\"pilot_executed\":false,\"target_launches\":[{\"launch_index\":0,\"iterations\":"<<iterations
      <<",\"input_profile\":"<<gh::quote(profile)<<",\"threads\":128,\"blocks\":1}],\"checks\":[{\"launch_index\":0,\"reference_model\":\"s19_integer_gemm_v1\",\"reference_sha256\":\""<<S19_REFERENCE_SHA256
      <<"\",\"comparison\":\"exact\",\"tolerance_id\":null,\"checked_elements\":"<<checked<<",\"expected_elements\":"<<checked<<",\"errors\":"<<errors
      <<",\"completed\":true,\"verified_CTA_ids\":[0],\"output_artifacts\":[";
    for(size_t i=0;i<artifacts.size();++i) {const auto& a=artifacts[i];std::cout<<(i?",":"")<<"{\"path\":"<<gh::quote(a.name)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"dtype\":\"uint32\",\"shape\":["<<a.words<<"],\"evidence_kind\":\"full_values\"}";}
    std::cout<<"]}],"<<resource<<"}\n";
    if(errors)std::cerr<<failure<<"\n";
  } else {
  std::cout<<std::setprecision(17)<<"{\"schema_version\":2,\"type\":"<<gh::quote(validation?"s19_validation":"s19_diagnostic")
    <<",\"case_id\":"<<gh::quote(c.id)<<",\"iterations\":"<<iterations<<",\"seed\":"<<seed<<",\"input_profile\":"<<gh::quote(profile)
    <<",\"performance_eligible\":false,\"warmup_executed\":false,\"errors\":"<<errors<<",\"failure\":"<<gh::quote(failure)
    <<",\"GPU_target_launches\":1,\"scope\":\"one_cta\",\"threads\":128,\"blocks\":1,\"event_ms\":"<<elapsed
    <<",\"work\":{\"fma_flop\":"<<(c.mode==1?0:65536ull*c.tiles*iterations)<<",\"input_payload_bytes\":"<<(c.mode==0?0:8192ull*c.tiles*iterations)
    <<",\"output_payload_bytes\":"<<(c.mode==4?4096ull*iterations:0)<<",\"epilogue_flop\":"<<(c.mode==4?2048ull*iterations:0)<<"},"
    <<resource<<",\"blocks_detail\":[{\"block_id\":0,\"smid\":"<<host_stamp.smid<<",\"start_ns\":"<<host_stamp.begin_ns<<",\"stop_ns\":"<<host_stamp.end_ns
    <<",\"start_cycle\":"<<host_stamp.begin_cycle<<",\"stop_cycle\":"<<host_stamp.end_cycle<<"}],\"artifacts\":[";
  for(size_t i=0;i<artifacts.size();++i) {const auto& a=artifacts[i];std::cout<<(i?",":"")<<"{\"path\":"<<gh::quote(a.name)<<",\"sha256\":"<<gh::quote(a.sha)<<",\"words\":"<<a.words<<",\"encoding\":\"u32le\"}";}
  std::cout<<"]}\n";
  }
  GH_CUDA(cudaEventDestroy(begin));GH_CUDA(cudaEventDestroy(end));GH_CUDA(cudaFree(stamp));
  return errors?1:0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
