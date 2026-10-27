// Four organizations of the same input and real cfg_b output services.
#include "r00_common.hpp"
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/arch/barrier.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstddef>

using namespace cute;
constexpr int M=128,N=128,K=256,Threads=384,Groups=4,PayloadElements=Groups*2*128*64,Guards=16;
constexpr uint32_t GuardBits=0x7fc12345u;
using JointTile=Shape<_128,_128,_64>;
using Cluster=Shape<_1,_1,_1>;
using EpiTile=Shape<_64,_32>;
using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,JointTile,Cluster,EpiTile,float,float,
    void,cutlass::layout::RowMajor,4,float,cutlass::layout::RowMajor,4,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    cutlass::half_t,cutlass::layout::RowMajor,8,cutlass::half_t,cutlass::layout::RowMajor,8,
    float,JointTile,Cluster,cutlass::gemm::collective::StageCount<6>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
using Full=cutlass::arch::ClusterTransactionBarrier;
static_assert(sizeof(Mainloop::TensorStorage)==196608);
static_assert(Kernel::SharedStorageSize==215040);
static_assert(Epilogue::DispatchPolicy::StagesD==2 && Epilogue::DispatchPolicy::DelayTmaStore);
constexpr int StageElements=128*64,PreparedElements=128*128,PreparedOperandFloats=8192;
static_assert(cosize_v<Mainloop::SmemLayoutA> == 6*StageElements);
static_assert(cosize_v<Mainloop::SmemLayoutB> == 6*StageElements);
static_assert(int(Mainloop::SmemLayoutA{}(make_coord(0,0,4)))==4*StageElements);
static_assert(int(Mainloop::SmemLayoutB{}(make_coord(0,0,4)))==4*StageElements);
using JointTensorStorage=typename Kernel::SharedStorage::TensorStorage;
constexpr size_t MainloopOffset=offsetof(Kernel::SharedStorage,tensors)+offsetof(JointTensorStorage,mainloop);
constexpr size_t EpilogueOffset=offsetof(Kernel::SharedStorage,tensors)+offsetof(JointTensorStorage,epilogue);
static_assert(EpilogueOffset+sizeof(Epilogue::TensorStorage)<=MainloopOffset);
static_assert(offsetof(Mainloop::TensorStorage,smem_A)+6*StageElements*2<=offsetof(Mainloop::TensorStorage,smem_B));
enum Organization { Input=0,Output=1,Serial=2,Concurrent=3 };
struct Point { uint64_t cycle=0,ns=0; };
struct Record { Point begin,end; uint32_t smid_begin=0,smid_end=0; };
__device__ __forceinline__ Point clock_pair() {
  Point p;
  asm volatile("mov.u64 %0, %%clock64; mov.u64 %1, %%globaltimer;"
      : "=l"(p.cycle),"=l"(p.ns) :: "memory");
  return p;
}
__device__ __forceinline__ uint32_t read_smid() {
  uint32_t s;asm volatile("mov.u32 %0, %%smid;":"=r"(s));return s;
}
__host__ __device__ float output_value(int row,int column,int seed) {
  return ((7*row+11*column+3*seed)%17-8)/32.f;
}

template<class CollectiveEpilogue>
__global__ __launch_bounds__(Threads,1) void q07_joint(
    const __grid_constant__ Mainloop::Params input,
    const __grid_constant__ typename CollectiveEpilogue::Params output,
    int organization,int seed,Record* records,cutlass::half_t* payload,int prepared) {
  extern __shared__ __align__(1024) unsigned char memory[];
  auto& shared=*reinterpret_cast<Kernel::SharedStorage*>(memory);
  __shared__ uint64_t full[6];
  auto shape=make_shape(M,N,K,1);
  auto sA=make_tensor(make_smem_ptr(shared.tensors.mainloop.smem_A.data()),Mainloop::SmemLayoutA{});
  auto sB=make_tensor(make_smem_ptr(shared.tensors.mainloop.smem_B.data()),Mainloop::SmemLayoutB{});
  auto inputs=Mainloop{}.load_init(shape,input);
  auto a_slice=input.tma_load_a.get_slice(0);auto b_slice=input.tma_load_b.get_slice(0);
  auto gA=get<0>(inputs)(_,_,0,_,0);auto gB=get<1>(inputs)(_,_,0,_,0);
  auto tAgA=a_slice.partition_S(gA);auto tBgB=b_slice.partition_S(gB);
  auto tAsA=a_slice.partition_D(sA);auto tBsB=b_slice.partition_D(sB);
  typename Mainloop::TiledMma mma;
  auto accumulators=partition_fragment_C(mma,take<0,2>(JointTile{}));
  auto coordinates=mma.get_slice(threadIdx.x%128).partition_C(make_identity_tensor(make_shape(_128{},_128{})));
  CUTLASS_PRAGMA_UNROLL
  for(int i=0;i<size(accumulators);++i) {
    auto coordinate=coordinates(i);
    accumulators(i)=output_value(get<0>(coordinate),get<1>(coordinate),seed);
  }
  static_assert(decltype(size(accumulators))::value==128);
  if(prepared && threadIdx.x>=128 && threadIdx.x<256) {
    // Two disjoint 32 KiB regions: physical mainloop A/B stages 4 and 5.
    volatile float* scratch_a=reinterpret_cast<volatile float*>(shared.tensors.mainloop.smem_A.data()+4*StageElements);
    volatile float* scratch_b=reinterpret_cast<volatile float*>(shared.tensors.mainloop.smem_B.data()+4*StageElements);
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<size(accumulators);++i) {
      int linear=i*128+(int(threadIdx.x)-128);
      if(linear<PreparedOperandFloats) scratch_a[linear]=accumulators(i);
      else scratch_b[linear-PreparedOperandFloats]=accumulators(i);
    }
  }
  typename CollectiveEpilogue::LoadPipeline::Params load_options;
  load_options.role=CollectiveEpilogue::LoadPipeline::ThreadCategory::Consumer;
  load_options.dst_blockid=0;
  load_options.producer_arv_count=32;load_options.consumer_arv_count=128;
  typename CollectiveEpilogue::LoadPipeline load_pipeline(shared.pipelines.epi_load,load_options);
  typename CollectiveEpilogue::StorePipeline::Params store_options;store_options.always_wait=true;
  typename CollectiveEpilogue::StorePipeline store_pipeline(store_options);
  typename CollectiveEpilogue::LoadPipelineState load_state;
  auto store_state=cutlass::make_producer_start_state<typename CollectiveEpilogue::StorePipeline>();
  CollectiveEpilogue epilogue(output,shared.tensors.epilogue);
  if(threadIdx.x==0) {
    for(int slot=0;slot<6;++slot) Full::init(full+slot,1);
    cutlass::arch::fence_barrier_init();
    prefetch_tma_descriptor(input.tma_load_a.get_tma_descriptor());
    prefetch_tma_descriptor(input.tma_load_b.get_tma_descriptor());
    prefetch_tma_descriptor(output.tma_store_d.get_tma_descriptor());
  }
  __syncthreads();
  Record record;
  if(threadIdx.x==0) { record.smid_begin=read_smid();record.begin=clock_pair(); }
  __syncthreads();
#pragma unroll 1
  for(int phase=0;phase<2;++phase) {
    if(phase==0 && organization!=Output && threadIdx.x==0) {
      CUTLASS_PRAGMA_UNROLL
      for(int group=0;group<Groups;++group) {
        Full::arrive_and_expect_tx(full+group,32768);
        copy(input.tma_load_a.with(full[group]),tAgA(_,_,_,group),tAsA(_,_,_,group));
        copy(input.tma_load_b.with(full[group]),tBgB(_,_,_,group),tBsB(_,_,_,group));
      }
      CUTLASS_PRAGMA_UNROLL
      for(int group=0;group<Groups;++group) Full::wait(full+group,0);
    }
    if(organization!=Input && phase==(organization==Serial ? 1 : 0) && threadIdx.x>=128 && threadIdx.x<256) {
      auto next=epilogue.store(load_pipeline,load_state,store_pipeline,store_state,
          shape,JointTile{},make_coord(0,0,_,0),accumulators,mma,threadIdx.x-128,shared.tensors.epilogue);
      epilogue.store_tail(load_pipeline,get<0>(next),store_pipeline,get<1>(next));
    }
    __syncthreads();
  }
  if(threadIdx.x==0) { record.end=clock_pair();record.smid_end=read_smid();records[0]=record; }
  __syncthreads();
  // All input groups retain distinct slots; numeric reads and export follow the end stamp.
  for(int q=threadIdx.x;q<PayloadElements;q+=Threads) {
    int slot=q/(2*128*64),operand=(q/(128*64))%2,index=q%(128*64);
    payload[q]=organization==Output ? cutlass::half_t(0.f)
        : operand==0 ? sA(index/64,index%64,slot) : sB(index/64,index%64,slot);
  }
  __syncthreads();
  if(threadIdx.x==0) for(int slot=0;slot<6;++slot) Full::invalidate(full+slot);
}

struct OptionsQ07 {
  int organization=Concurrent,seed=3,warmup_limit=30,prepared=0;bool inspect=false;
  std::string mode="concurrent";
  OptionsQ07(int argc,char**argv) {
    for(int i=1;i<argc;++i) {
      std::string key=argv[i];
      if(key=="--inspect-program") {inspect=true;continue;}
      if(i+1==argc) throw std::runtime_error("missing CLI value");
      std::string value=argv[++i];
      if(key=="--organization") {
        mode=value;
        if(value=="input") organization=Input;else if(value=="output") organization=Output;
        else if(value=="serial") organization=Serial;else if(value=="concurrent") organization=Concurrent;
        else throw std::runtime_error("invalid organization");
      } else {
        size_t end=0;int parsed=std::stoi(value,&end);
        if(end!=value.size()) throw std::runtime_error("noninteger value");
        if(key=="--seed") seed=parsed;else if(key=="--warmup-limit") warmup_limit=parsed;
        else if(key=="--prepared") prepared=parsed;
        else throw std::runtime_error("unknown CLI option");
      }
    }
    if(seed!=3 && seed!=22 && seed!=41) throw std::runtime_error("seed must be 3,22,41");
    if(warmup_limit<8 || warmup_limit>60) throw std::runtime_error("warmup-limit must be 8..60");
    if(prepared!=0 && prepared!=1) throw std::runtime_error("prepared must be 0 or 1");
  }
};
std::string uuid_string(cudaDeviceProp const&p) {
  std::ostringstream out;out<<"GPU-"<<std::hex<<std::setfill('0');
  for(int i=0;i<16;++i) {if(i==4||i==6||i==8||i==10) out<<'-';out<<std::setw(2)<<unsigned(uint8_t(p.uuid.bytes[i]));}
  return out.str();
}
void inspect() {
  std::cout<<"{\"schema\":\"gh200-q07-program/1\",\"cuda_api_calls\":0,\"threads\":384,\"input_groups\":4,\"input_stages\":6,\"input_bytes\":131072,\"output_bytes\":65536,\"output_subtiles\":8,\"factory\":{\"shared_storage_bytes\":"<<Kernel::SharedStorageSize
      <<",\"mainloop_tensor_bytes\":"<<sizeof(Mainloop::TensorStorage)<<",\"epilogue_tensor_bytes\":"<<sizeof(Epilogue::TensorStorage)
      <<",\"StagesC\":"<<Epilogue::DispatchPolicy::StagesC<<",\"StagesD\":"<<Epilogue::DispatchPolicy::StagesD
      <<",\"DelayTmaStore\":"<<(Epilogue::DispatchPolicy::DelayTmaStore?"true":"false")<<"}"
      <<",\"prepared_scratch\":{\"elements\":16384,\"bytes\":65536,\"mainloop_offset\":"<<MainloopOffset
      <<",\"epilogue_offset\":"<<EpilogueOffset<<",\"a_offset\":"<<MainloopOffset+offsetof(Mainloop::TensorStorage,smem_A)+4*StageElements*2
      <<",\"b_offset\":"<<MainloopOffset+offsetof(Mainloop::TensorStorage,smem_B)+4*StageElements*2<<",\"operand_bytes\":32768}}\n";
}
int run(OptionsQ07 const&o) {
  int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));if(count!=1) throw std::runtime_error("one visible GPU required");
  cudaDeviceProp device{};CUDA_CHECK(cudaGetDeviceProperties(&device,0));
  if(device.major!=9||device.minor!=0||device.multiProcessorCount!=132||std::string(device.name).find("GH200")==std::string::npos)
    throw std::runtime_error("GH200 SM90/132SM required");
  DeviceBuffer<cutlass::half_t> a(M*K),b(K*N),payload(PayloadElements);
  DeviceBuffer<float> d(M*N+2*Guards);DeviceBuffer<Record> records(1);
  std::vector<cutlass::half_t> ha(M*K),hb(K*N),hp(PayloadElements);
  for(int r=0;r<M;++r) for(int k=0;k<K;++k) ha[r*K+k]=cutlass::half_t(input_value(r,k,o.seed,true));
  for(int k=0;k<K;++k) for(int c=0;c<N;++c) hb[k*N+c]=cutlass::half_t(input_value(k,c,o.seed,false));
  CUDA_CHECK(cudaMemcpy(a.pointer,ha.data(),ha.size()*sizeof(ha[0]),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.pointer,hb.data(),hb.size()*sizeof(hb[0]),cudaMemcpyHostToDevice));
  auto shape=make_shape(M,N,K,1);
  auto sa=make_stride(int64_t(K),Int<1>{},int64_t(M*K));
  auto sb=make_stride(Int<1>{},int64_t(N),int64_t(K*N));
  auto sd=make_stride(int64_t(N),Int<1>{},int64_t(M*N));
  typename Mainloop::Arguments input_args{a.pointer,sa,b.pointer,sb};
  typename Epilogue::Arguments output_args{{1.f,0.f},nullptr,sd,d.pointer+Guards,sd};
  auto input=Mainloop::to_underlying_arguments(shape,input_args,nullptr);
  auto output=Epilogue::to_underlying_arguments(shape,output_args,nullptr);
  CUDA_CHECK(cudaFuncSetAttribute(q07_joint<Epilogue>,cudaFuncAttributeMaxDynamicSharedMemorySize,Kernel::SharedStorageSize));
  cudaFuncAttributes attributes{};CUDA_CHECK(cudaFuncGetAttributes(&attributes,q07_joint<Epilogue>));
  int occupancy=0;CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,q07_joint<Epilogue>,Threads,Kernel::SharedStorageSize));
  if(occupancy!=1||attributes.localSizeBytes) throw std::runtime_error("occupancy1/local0 required");
  std::vector<uint32_t> initial(M*N+2*Guards,GuardBits),actual(initial.size());
  Record record;uint64_t errors=0,checked=0;
  auto execute=[&]() {
    CUDA_CHECK(cudaMemcpy(d.pointer,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));
    q07_joint<Epilogue><<<1,Threads,Kernel::SharedStorageSize>>>(input,output,o.organization,o.seed,records.pointer,payload.pointer,o.prepared);
    CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&record,records.pointer,sizeof(record),cudaMemcpyDeviceToHost));
    if(record.end.cycle<=record.begin.cycle||record.end.ns<=record.begin.ns||record.smid_begin!=record.smid_end)
      throw std::runtime_error("invalid timing/SM identity");
    return double(record.end.ns-record.begin.ns);
  };
  std::vector<double> warmup;bool converged=false;double cv=0;
  for(int i=0;i<o.warmup_limit;++i) {
    warmup.push_back(execute());
    if(i>=7) {
      double mean=0,var=0;for(int j=i-4;j<=i;++j) mean+=warmup[j]/5;
      for(int j=i-4;j<=i;++j) var+=(warmup[j]-mean)*(warmup[j]-mean)/4;
      cv=std::sqrt(var)/mean;if(cv<=0.02){converged=true;break;}
    }
  }
  execute();
  CUDA_CHECK(cudaMemcpy(hp.data(),payload.pointer,hp.size()*sizeof(hp[0]),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual.data(),d.pointer,actual.size()*4,cudaMemcpyDeviceToHost));
  for(int q=0;q<PayloadElements;++q) {
    int slot=q/(2*128*64),op=(q/(128*64))%2,index=q%(128*64);
    float expected=o.organization==Output ? 0.f : op==0 ? input_value(index/64,slot*64+index%64,o.seed,true)
        : input_value(slot*64+index%64,index/64,o.seed,false);
    errors+=float(hp[q])!=expected;++checked;
  }
  for(int q=0;q<M*N+2*Guards;++q) {
    uint32_t expected=GuardBits;
    if(o.organization!=Input && q>=Guards && q<Guards+M*N) {
      float value=output_value((q-Guards)/N,(q-Guards)%N,o.seed);std::memcpy(&expected,&value,4);
    }
    errors+=actual[q]!=expected;++checked;
  }
  std::ofstream po("payload.f16",std::ios::binary),out("output-guarded.u32",std::ios::binary);
  po.write(reinterpret_cast<char const*>(hp.data()),hp.size()*sizeof(hp[0]));
  out.write(reinterpret_cast<char const*>(actual.data()),actual.size()*4);
  if(!po||!out) throw std::runtime_error("could not write witnesses");
  std::cout<<std::setprecision(17)<<"{\"schema\":\"gh200-q07-input-output/1\",\"status\":\"measured\",\"organization\":\""<<o.mode
      <<"\",\"seed\":"<<o.seed<<",\"samples\":1,\"performance_eligible\":"<<(converged&&!errors?"true":"false")
      <<",\"device\":{\"uuid\":\""<<uuid_string(device)<<"\",\"name\":\""<<device.name<<"\",\"sms\":"<<device.multiProcessorCount<<"}"
      <<",\"resources\":{\"threads\":384,\"grid_ctas\":1,\"dynamic_smem_bytes\":"<<Kernel::SharedStorageSize
      <<",\"static_smem_bytes\":"<<attributes.sharedSizeBytes<<",\"registers_per_thread\":"<<attributes.numRegs
      <<",\"local_size_bytes\":"<<attributes.localSizeBytes<<",\"occupancy_ctas_per_sm\":"<<occupancy<<"}"
      <<",\"timing\":{\"begin_cycle\":"<<record.begin.cycle<<",\"end_cycle\":"<<record.end.cycle
      <<",\"begin_ns\":"<<record.begin.ns<<",\"end_ns\":"<<record.end.ns
      <<",\"elapsed_cycles\":"<<record.end.cycle-record.begin.cycle<<",\"elapsed_ns\":"<<record.end.ns-record.begin.ns
      <<",\"smid_begin\":"<<record.smid_begin<<",\"smid_end\":"<<record.smid_end<<"}"
      <<",\"warmup\":{\"iterations\":"<<warmup.size()<<",\"converged\":"<<(converged?"true":"false")<<",\"tail_cv\":"<<cv<<"}"
      <<",\"correctness\":{\"checked_elements\":"<<checked<<",\"errors\":"<<errors<<"},\"trace\":false,\"prepared\":"<<(o.prepared?"true":"false")<<"}\n";
  return errors?2:0;
}
int main(int argc,char**argv) try {OptionsQ07 o(argc,argv);if(o.inspect){inspect();return 0;}return run(o);}
catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 2;}
