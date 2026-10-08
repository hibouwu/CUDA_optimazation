// R16 quota: release before/after a dependent integer delay; two consumer WGs.
#include "r01_support.hpp"
#include <cutlass/arch/reg_reconfig.h>
struct QuotaStamp { uint64_t start,enter,leave,delay_start,delay_end,compute_start,compute_end;unsigned sm,last_sm; };
__device__ uint64_t quota_clock(uint64_t x){uint64_t y;asm volatile("{.reg .pred p;setp.ne.u64 p,%1,0;mov.u64 %0,0;@p mov.u64 %0,%%clock64;}":"=l"(y):"l"(x):"memory");return y;}
__global__ __launch_bounds__(384,1) void quota16(const float* initial,int delay,bool before,
                                    QuotaStamp* stamps,float* output,unsigned* integers){
  __shared__ volatile uint64_t gate[384];
  int t=threadIdx.x,wg=t/128;gate[t]=(uint64_t(1)<<48)+t+1;
  __syncthreads();QuotaStamp s{};s.sm=r_sm();s.start=quota_clock(gate[(t+128)%384]);
  if(wg==0){
    unsigned x=t+17;
    auto work=[&](){s.delay_start=clock64();
#pragma unroll 1
      for(int i=0;i<delay;++i)asm volatile("mad.lo.u32 %0,%0,1664525,1013904223;":"+r"(x)::"memory");
      s.delay_end=quota_clock(uint64_t(x)+1);
    };
    if(before)work();
    s.enter=clock64();cutlass::arch::warpgroup_reg_dealloc<40>();s.leave=clock64();
    if(!before)work();
    integers[t]=x;
  }else{
    s.enter=clock64();cutlass::arch::warpgroup_reg_alloc<232>();s.leave=clock64();
    float live[192];
#pragma unroll
    for(int j=0;j<192;++j){live[j]=reinterpret_cast<const volatile float*>(initial)[(t-128)*192+j];asm volatile("":"+f"(live[j])::"memory");}
    float sum=0;
    // Timed dependent reduction; loads precede this window, stores follow it.
    s.compute_start=quota_clock(uint64_t(__float_as_uint(live[191]))+1);
#pragma unroll
    for(int j=0;j<192;++j)asm volatile("add.rn.f32 %0,%0,%1;":"+f"(sum):"f"(live[j]):"memory");
    s.compute_end=quota_clock(uint64_t(__float_as_uint(sum))+1);
#pragma unroll
    for(int j=0;j<192;++j)output[(t-128)*192+j]=live[j];
    integers[t]=__float_as_uint(sum);
  }
  s.last_sm=r_sm();if(t%32==0)stamps[t/32]=s;
}
int main(int argc,char**argv){try{
  int delay=0,before=1;for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("missing option");std::string k=argv[i];int v=std::stoi(argv[i+1]);if(k=="--delay")delay=v;else if(k=="--before")before=v;else throw std::runtime_error("unknown option");}
  if((delay!=0&&delay!=64&&delay!=256)||before<0||before>1)throw std::runtime_error("invalid quota coordinate");
  std::vector<float> input(256*192);for(int t=0;t<256;++t)for(int j=0;j<192;++j)input[t*192+j]=(t+1)/1024.f+j/32.f;
  DeviceBuffer<float> initial(input.size()),output(input.size());DeviceBuffer<unsigned> integers(384);DeviceBuffer<QuotaStamp> stamp(12);
  CUDA_CHECK(cudaMemcpy(initial.pointer,input.data(),input.size()*4,cudaMemcpyHostToDevice));
  cudaFuncAttributes attr{};CUDA_CHECK(cudaFuncGetAttributes(&attr,quota16));std::vector<QuotaStamp> times(12);
  auto launch=[&](){quota16<<<1,384>>>(initial.pointer,delay,before,stamp.pointer,output.pointer,integers.pointer);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(times.data(),stamp.pointer,times.size()*sizeof(QuotaStamp),cudaMemcpyDeviceToHost));std::vector<uint64_t> waits;for(int i=4;i<12;++i)waits.push_back(times[i].leave-times[i].enter);std::sort(waits.begin(),waits.end());return (waits[3]+waits[4])/2;};
  Windows warm=warm_up(launch);launch();std::vector<float> out(output.count);std::vector<unsigned> values(384);
  CUDA_CHECK(cudaMemcpy(out.data(),output.pointer,out.size()*4,cudaMemcpyDeviceToHost));CUDA_CHECK(cudaMemcpy(values.data(),integers.pointer,values.size()*4,cudaMemcpyDeviceToHost));
  save_binary("values.f32",out);save_binary("integers.u32",values);
  std::cout<<std::setprecision(17)<<"{\"delay\":"<<delay<<",\"before\":"<<before<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes<<",\"threads\":384,\"warmup_converged\":"<<(warm.converged?"true":"false")<<",\"warmup\":";vector_json(warm.warmup);
  std::cout<<",\"schema_version\":2,\"warmup_metric\":\"median_consumer_inc_cycles\",\"stamps\":[";for(int i=0;i<12;++i){auto s=times[i];std::cout<<(i?",":"")<<'['<<s.start<<','<<s.enter<<','<<s.leave<<','<<s.delay_start<<','<<s.delay_end<<','<<s.sm<<','<<s.compute_start<<','<<s.compute_end<<','<<s.last_sm<<']';}std::cout<<"]}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
