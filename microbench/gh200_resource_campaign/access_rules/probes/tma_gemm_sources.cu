// cfg_b source coordinates with the tensor probe controller; warm both output work lists.
#include "r00_common.hpp"
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/arch/barrier.h>
#include <fstream>

using namespace cute;
using ProbeTile = Shape<_128, _128, _64>;
using ProbeCluster = Shape<_1, _1, _1>;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8, float, ProbeTile, ProbeCluster,
    cutlass::gemm::collective::StageCount<4>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
constexpr int Threads = 128;
constexpr int SharedBytes = sizeof(Mainloop::TensorStorage) + 1024;
static_assert(sizeof(Mainloop::TensorStorage) == 131072);
static_assert(Mainloop::TmaTransactionBytes == 32768);

struct Point { uint64_t cycle, ns; };
struct GroupRecord { Point issue_begin, issue_end, wait_begin, wait_end, released; };
struct CtaRecord { Point begin, end; uint64_t errors, checked; unsigned smid; };
__device__ __forceinline__ Point stamp() {
  Point p;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(p.cycle) :: "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(p.ns) :: "memory");
  return p;
}
__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}

__global__ void tensor_probe(const __grid_constant__ Mainloop::Params params,
    int m, int n, int k, int iterations, int active, int verify,
    int trace, int seed, int release_delay_ns, int source_phase_stride, int source_tile_shift,
    int launch_skew_ns, int2 const* worklist, int work_index, GroupRecord* records,
    CtaRecord* ctas, uint64_t* gate) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& tensors = *reinterpret_cast<Mainloop::TensorStorage*>(storage);
  auto* barriers = reinterpret_cast<uint64_t*>(storage + sizeof(Mainloop::TensorStorage));
  auto sA = make_tensor(make_smem_ptr(tensors.smem_A.data()), Mainloop::SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(tensors.smem_B.data()), Mainloop::SmemLayoutB{});
  auto inputs = Mainloop{}.load_init(make_shape(m, n, k, 1), params);
  auto a_slice = params.tma_load_a.get_slice(0);
  auto b_slice = params.tma_load_b.get_slice(0);
  int tile_index = (int(blockIdx.x)+source_tile_shift)%int(gridDim.x);
  int2 tile = worklist[work_index*gridDim.x+tile_index];
  int source_phase = (int(blockIdx.x)*source_phase_stride) % (k/64);
  auto gA = get<0>(inputs)(_, _, tile.x, _, 0);
  auto gB = get<1>(inputs)(_, _, tile.y, _, 0);
  auto tAgA = a_slice.partition_S(gA);
  auto tBgB = b_slice.partition_S(gB);
  auto tAsA = a_slice.partition_D(sA);
  auto tBsB = b_slice.partition_D(sB);
  auto* out = records + size_t(blockIdx.x)*iterations;
  if (threadIdx.x == 0) {
    for (int slot=0; slot<4; ++slot) Barrier::init(barriers+slot, 1);
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(params.tma_load_b.get_tma_descriptor());
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  auto grid = cooperative_groups::this_grid();
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) *gate = global_ns() + 100000;
  grid.sync();
  uint64_t start = *gate;
  int64_t offset = launch_skew_ns>=0 ? int64_t(blockIdx.x)*launch_skew_ns
      : int64_t(gridDim.x-1-blockIdx.x)*-int64_t(launch_skew_ns);
  while (global_ns() < start+uint64_t(offset)) {}
  __syncthreads();
  if (!verify && threadIdx.x != 0) return;
  Point begin = stamp();
  unsigned smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  uint64_t errors = 0, checked = 0;

  auto issue_group = [&](int g) {
    int slot = g % 4, kt = (g+source_phase) % (k/64);
    Point before{};
    if (trace) before=stamp();
    Barrier::arrive_and_expect_tx(barriers+slot, 32768);
    copy(params.tma_load_a.with(barriers[slot]), tAgA(_,_,_,kt), tAsA(_,_,_,slot));
    copy(params.tma_load_b.with(barriers[slot]), tBgB(_,_,_,kt), tBsB(_,_,_,slot));
    if (trace) {
      Point after=stamp();
      out[g].issue_begin=before; out[g].issue_end=after;
    }
  };
  if (threadIdx.x == 0) {
    for (int g=0; g<active; ++g) issue_group(g);
  }
  for (int g=0; g<iterations; ++g) {
    Point wb{}, we{};
    if (threadIdx.x == 0) {
      if (trace) wb=stamp();
      Barrier::wait(barriers+g%4, (g/4)%2);
      if (trace) we=stamp();
    }
    if (verify) {
      __syncthreads();
      int row_base=tile.x*128, col_base=tile.y*128, k_base=((g+source_phase)%(k/64))*64;
      for (int q=threadIdx.x; q<128*64; q+=Threads) {
        int row=q/64, col=q%64;
        float ea=input_value(row_base+row, k_base+col, seed, true);
        float eb=input_value(k_base+col, col_base+row, seed, false);
        errors += float(sA(row,col,g%4)) != ea;
        errors += float(sB(row,col,g%4)) != eb;
        checked += 2;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (release_delay_ns) {
        uint64_t until=global_ns()+release_delay_ns;
        while (global_ns()<until) {}
      }
      if (trace) {
        Point released=stamp();
        out[g].wait_begin=wb; out[g].wait_end=we; out[g].released=released;
      }
      if (g+active<iterations) issue_group(g+active);
    }
  }
  Point end=stamp();
  if (threadIdx.x == 0) {
    ctas[blockIdx.x].begin=begin; ctas[blockIdx.x].end=end;
    ctas[blockIdx.x].smid=smid;
    for (int slot=0; slot<4; ++slot) Barrier::invalidate(barriers+slot);
  }
  if (verify) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&ctas[blockIdx.x].errors),
              static_cast<unsigned long long>(errors));
    atomicAdd(reinterpret_cast<unsigned long long*>(&ctas[blockIdx.x].checked),
              static_cast<unsigned long long>(checked));
  }
}

static std::string descriptor_hex(void const* pointer) {
  auto bytes=static_cast<unsigned char const*>(pointer);
  char digit[]="0123456789abcdef";
  std::string result;
  for (int i=0; i<128; ++i) { result+=digit[bytes[i]>>4]; result+=digit[bytes[i]&15]; }
  return result;
}
static void print_point(Point p) { std::cout<<'['<<p.cycle<<','<<p.ns<<']'; }

__global__ void write_pollution(uint32_t* data, size_t words) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<words; i+=gridDim.x*blockDim.x) {
    asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(data+i), "r"(uint32_t(1)) : "memory");
  }
}

__global__ void fill_allocation_tail(cutlass::half_t* data, size_t count) {
  for (size_t i=blockIdx.x*blockDim.x+threadIdx.x; i<count; i+=gridDim.x*blockDim.x)
    data[i]=cutlass::half_t(65504.f);
}

int main(int argc, char** argv) try {
  int blocks=1, active=1, iterations=8, ring=32, verify=1, trace=1;
  int seed=17, delay=0, warmup=0, source_phase_stride=0, source_tile_shift=0, launch_skew_ns=0, ldb=0, pollute_kib=0;
  int warmup_skew_ns=0, work_index=0, warmup_work_offset=0, record_warmup=0;
  int measure_groups=0, measure_source_phase_stride=-1;
  int alloc_lda=0;
  std::string worklist_path;
  bool warmup_skew_set=false;
  for (int i=1; i<argc; i+=2) {
    if (i+1==argc) throw std::runtime_error("missing value");
    std::string name=argv[i];
    if (name=="--worklist") {worklist_path=argv[i+1];continue;}
    int value=std::stoi(argv[i+1]);
    if (name=="--ctas") blocks=value;
    else if (name=="--active") active=value;
    else if (name=="--groups") iterations=value;
    else if (name=="--ring") ring=value;
    else if (name=="--verify") verify=value;
    else if (name=="--trace") trace=value;
    else if (name=="--seed") seed=value;
    else if (name=="--release-delay-ns") delay=value;
    else if (name=="--warmup") warmup=value;
    else if (name=="--source-phase-stride") source_phase_stride=value;
    else if (name=="--ldb") ldb=value;
    else if (name=="--alloc-lda") alloc_lda=value;
    else if (name=="--pollute-kib") pollute_kib=value;
    else if (name=="--source-tile-shift") source_tile_shift=value;
    else if (name=="--launch-skew-ns") launch_skew_ns=value;
    else if (name=="--work-index") work_index=value;
    else if (name=="--warmup-work-offset") warmup_work_offset=value;
    else if (name=="--record-warmup") record_warmup=value;
    else if (name=="--measure-groups") measure_groups=value;
    else if (name=="--measure-source-phase-stride") measure_source_phase_stride=value;
    else if (name=="--warmup-skew-ns") {warmup_skew_ns=value;warmup_skew_set=true;}
    else throw std::runtime_error("unknown argument: "+name);
  }
  if (blocks<1 || (active!=1 && active!=2 && active!=4) || iterations<active || ring<1 ||
      (verify!=0 && verify!=1) ||
      (trace!=0 && trace!=1) || delay<0 || warmup<0 || source_phase_stride<0 ||
      measure_groups<0 || measure_groups>iterations || (measure_groups && measure_groups<active) ||
      measure_source_phase_stride < -1) throw std::runtime_error("invalid input");
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if (prop.major!=9 || !prop.cooperativeLaunch || blocks>prop.multiProcessorCount)
    throw std::runtime_error("requires SM90 cooperative grid within SM count");
  int m=0,n=0,k=0,work_ctas=0;
  std::ifstream input(worklist_path);
  if (!(input>>m>>n>>k>>work_ctas) || work_ctas!=blocks || k!=64*ring ||
      m<=0 || n<=0 || m%128 || n%128 || work_index<0 || work_index>1 ||
      warmup_work_offset<0 || warmup_work_offset>1 || (record_warmup!=0 && record_warmup!=1))
    throw std::runtime_error("worklist header must match CTA count, K ring and full cfg_b tiles");
  std::vector<int2> coordinates(size_t(blocks)*2);
  for (auto& tile:coordinates) {
    if (!(input>>tile.x>>tile.y) || tile.x<0 || tile.y<0 || tile.x>=m/128 || tile.y>=n/128)
      throw std::runtime_error("invalid cfg_b work coordinate");
  }
  DeviceBuffer<int2> worklist(coordinates.size());
  CUDA_CHECK(cudaMemcpy(worklist.pointer,coordinates.data(),coordinates.size()*sizeof(int2),cudaMemcpyHostToDevice));
  if (source_tile_shift<0 || source_tile_shift>=blocks)
    throw std::runtime_error("source-tile-shift must be in 0..ctas-1");
  if (ldb==0) ldb=n;
  if (ldb<n || ldb%8) throw std::runtime_error("ldb must cover N and align to 8 elements");
  if (alloc_lda==0) alloc_lda=k;
  if (alloc_lda<k || alloc_lda%8) throw std::runtime_error("alloc-lda must cover K and align to 8 elements");
  DeviceBuffer<cutlass::half_t> a(size_t(m)*alloc_lda), b(size_t(ldb)*k);
  fill_input<<<128,256>>>(a.pointer,m,k,seed,true,false);
  if (alloc_lda>k) fill_allocation_tail<<<1,256>>>(a.pointer+size_t(m)*k,size_t(m)*(alloc_lda-k));
  fill_input<<<128,256>>>(b.pointer,k,ldb,seed,false,false);
  CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
  Mainloop::Arguments args{a.pointer, make_stride(int64_t(k),_1{},int64_t(0)),
                          b.pointer, make_stride(_1{},int64_t(ldb),int64_t(0))};
  Mainloop::Params params=Mainloop::to_underlying_arguments(make_shape(m,n,k,1),args,nullptr);
  CUDA_CHECK(cudaFuncSetAttribute(tensor_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,SharedBytes));
  int occupancy=0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,tensor_probe,Threads,SharedBytes));
  if (occupancy!=1) throw std::runtime_error("expected one CTA per SM");
  cudaFuncAttributes attr{}; CUDA_CHECK(cudaFuncGetAttributes(&attr,tensor_probe));
  if (pollute_kib<0 || pollute_kib>32768) throw std::runtime_error("pollute-kib must be 0..32768");
  DeviceBuffer<uint32_t> pollution(size_t(32)*1024*1024/4);
  size_t pollution_words=size_t(pollute_kib)*1024/4;
  DeviceBuffer<GroupRecord> records(size_t(blocks)*iterations);
  DeviceBuffer<CtaRecord> ctas(blocks); DeviceBuffer<uint64_t> gate(1);
  auto launch=[&]() {
    if (pollution_words) {
      write_pollution<<<132,256>>>(pollution.pointer,pollution_words);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemset(ctas.pointer,0,blocks*sizeof(CtaRecord)));
    void* values[]={&params,&m,&n,&k,&iterations,&active,&verify,&trace,&seed,&delay,
                   &source_phase_stride,&source_tile_shift,&launch_skew_ns,&worklist.pointer,&work_index,&records.pointer,&ctas.pointer,&gate.pointer};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(tensor_probe),dim3(blocks),
                                          dim3(Threads),values,SharedBytes));
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  if (!warmup_skew_set) warmup_skew_ns=launch_skew_ns;
  int measured_skew_ns=launch_skew_ns;
  launch_skew_ns=warmup_skew_ns;
  int measured_work_index=work_index;
  int warmup_groups=iterations, warmup_source_phase_stride=source_phase_stride;
  std::vector<std::vector<CtaRecord>> warmup_records;
  for (int i=0; i<warmup; ++i) {
    work_index=(i+warmup_work_offset)%2;launch();
    if (record_warmup) {
      warmup_records.emplace_back(blocks);
      CUDA_CHECK(cudaMemcpy(warmup_records.back().data(),ctas.pointer,
                            blocks*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
    }
  }
  work_index=measured_work_index;
  launch_skew_ns=measured_skew_ns;
  if (measure_groups) iterations=measure_groups;
  if (measure_source_phase_stride>=0) source_phase_stride=measure_source_phase_stride;
  launch();
  std::vector<CtaRecord> host(blocks);
  CUDA_CHECK(cudaMemcpy(host.data(),ctas.pointer,blocks*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
  std::vector<GroupRecord> events(size_t(blocks)*iterations);
  if (trace) CUDA_CHECK(cudaMemcpy(events.data(),records.pointer,events.size()*sizeof(GroupRecord),cudaMemcpyDeviceToHost));
  uint64_t total_errors=0,total_checked=0;
  std::vector<uint32_t> pollution_host(pollution_words);
  if (pollution_words) CUDA_CHECK(cudaMemcpy(pollution_host.data(),pollution.pointer,
                                            pollution_words*4,cudaMemcpyDeviceToHost));
  uint64_t pollution_errors=0;
  for (uint32_t value:pollution_host) pollution_errors += value!=1;
  std::set<unsigned> sms;
  for (auto const& c:host) {total_errors+=c.errors;total_checked+=c.checked;sms.insert(c.smid);}
  char uuid[80]; auto u=reinterpret_cast<unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  std::cout<<"{\"schema\":\"cfg-b-gemm-source-probe/1\",\"gpu_uuid\":\""<<uuid
    <<"\",\"ctas\":"<<blocks<<",\"active_groups\":"<<active<<",\"groups\":"<<iterations
    <<",\"ring\":"<<ring<<",\"verify\":"<<verify<<",\"trace\":"<<trace
    <<",\"seed\":"<<seed<<",\"release_delay_ns\":"<<delay<<",\"warmup\":"<<warmup
    <<",\"warmup_groups\":"<<warmup_groups
    <<",\"warmup_source_phase_stride\":"<<warmup_source_phase_stride
    <<",\"source_phase_stride\":"<<source_phase_stride
    <<",\"source_tile_shift\":"<<source_tile_shift
    <<",\"launch_skew_ns\":"<<launch_skew_ns
    <<",\"warmup_skew_ns\":"<<warmup_skew_ns
    <<",\"pollute_kib\":"<<pollute_kib<<",\"pollution_checked\":"<<pollution_words
    <<",\"pollution_errors\":"<<pollution_errors
    <<",\"warmup_work_offset\":"<<warmup_work_offset
    <<",\"record_warmup\":"<<record_warmup
    <<",\"work_index\":"<<work_index<<",\"m\":"<<m<<",\"n\":"<<n
    <<",\"lda\":"<<k<<",\"ldb\":"<<ldb
    <<",\"alloc_lda\":"<<alloc_lda<<",\"allocated_a_elements\":"<<a.count
    <<",\"source_a_address\":"<<reinterpret_cast<uintptr_t>(a.pointer)
    <<",\"source_b_address\":"<<reinterpret_cast<uintptr_t>(b.pointer)
    <<",\"records_address\":"<<reinterpret_cast<uintptr_t>(records.pointer)
    <<",\"ctas_address\":"<<reinterpret_cast<uintptr_t>(ctas.pointer)
    <<",\"worklist_address\":"<<reinterpret_cast<uintptr_t>(worklist.pointer)
    <<",\"gate_address\":"<<reinterpret_cast<uintptr_t>(gate.pointer)
    <<",\"pollution_address\":"<<reinterpret_cast<uintptr_t>(pollution.pointer)
    <<",\"group_record_bytes\":"<<sizeof(GroupRecord)<<",\"cta_record_bytes\":"<<sizeof(CtaRecord)
    <<",\"smem\":"<<SharedBytes<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes
    <<",\"max_active_ctas_per_sm\":"<<occupancy<<",\"observed_sms\":"<<sms.size()
    <<",\"errors\":"<<total_errors<<",\"checked\":"<<total_checked
    <<",\"descriptor_a\":\""<<descriptor_hex(params.tma_load_a.get_tma_descriptor())
    <<"\",\"descriptor_b\":\""<<descriptor_hex(params.tma_load_b.get_tma_descriptor())<<"\",\"ctas_data\":[";
  for (int c=0;c<blocks;++c) {
    if(c)std::cout<<',';
    std::cout<<"{\"cta\":"<<c<<",\"smid\":"<<host[c].smid<<",\"tile\":["<<coordinates[work_index*blocks+(c+source_tile_shift)%blocks].x<<','<<coordinates[work_index*blocks+(c+source_tile_shift)%blocks].y<<"],\"begin\":";print_point(host[c].begin);
    std::cout<<",\"end\":";print_point(host[c].end);std::cout<<'}';
  }
  std::cout<<"],\"warmup_ctas\":[";
  for (size_t i=0;i<warmup_records.size();++i) {
    if(i)std::cout<<',';
    std::cout<<"{\"work_index\":"<<(i+warmup_work_offset)%2<<",\"ctas\":[";
    for (int c=0;c<blocks;++c) {
      if(c)std::cout<<',';
      auto const& record=warmup_records[i][c];
      std::cout<<"{\"cta\":"<<c<<",\"smid\":"<<record.smid<<",\"begin\":";print_point(record.begin);
      std::cout<<",\"end\":";print_point(record.end);std::cout<<'}';
    }
    std::cout<<"]}";
  }
  std::cout<<"],\"group_events\":[";
  if(trace)for(size_t i=0;i<events.size();++i) {
    if(i)std::cout<<',';
    auto const& e=events[i];
    std::cout<<"{\"cta\":"<<i/iterations<<",\"group\":"<<i%iterations<<",\"points\":[";
    print_point(e.issue_begin);std::cout<<',';print_point(e.issue_end);std::cout<<',';
    print_point(e.wait_begin);std::cout<<',';print_point(e.wait_end);std::cout<<',';print_point(e.released);
    std::cout<<"]}";
  }
  std::cout<<"]}\n";
  return total_errors || pollution_errors || sms.size()!=size_t(blocks) ||
         (verify && total_checked!=uint64_t(blocks)*iterations*16384) ? 2 : 0;
} catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
