// cfg_b tensor requests with a separate warpgroup issuing controlled WGMMA background.
#include "r00_common.hpp"
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/arch/barrier.h>
#include <fstream>
#include <cstring>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <utility>

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
constexpr int Threads = 256;
namespace G = cute::SM90::GMMA;
using BackgroundMma = G::MMA_64x128x16_F32F16F16_SS<G::Major::K, G::Major::MN>;
using BackgroundMmaRS = G::MMA_64x128x16_F32F16F16_RS<G::Major::K, G::Major::MN>;
constexpr int SharedBytes = sizeof(Mainloop::TensorStorage) + 1024 + 24576;
static_assert(sizeof(Mainloop::TensorStorage) == 131072);
static_assert(Mainloop::TmaTransactionBytes == 32768);

struct Point { uint64_t cycle, ns; };
struct GroupRecord { Point issue_begin, issue_end, wait_begin, wait_end, released; };
struct CtaRecord { Point begin, end, background_begin, background_end; uint64_t errors, checked; unsigned smid, background_rounds; };
template <size_t... I>
__device__ void background_mma(uint64_t a, uint64_t b, float* d, std::index_sequence<I...>) {
  BackgroundMma::fma(a, b, d[I]..., G::ScaleOut::One);
}
template <size_t... I>
__device__ void background_mma_rs(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint64_t b, float* d, std::index_sequence<I...>) {
  BackgroundMmaRS::fma(a0, a1, a2, a3, b, d[I]..., G::ScaleOut::One);
}
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
    int m, int n, int k, int iterations, int active, int independent, int verify,
    int trace, int seed, int release_delay_ns, int source_phase_stride, int background,
    int tma_ctas, int background_skip_targets, int target_rs, int target_scalar_warp,
    int issuer_warp, float* background_output, GroupRecord* records,
    CtaRecord* ctas, uint64_t* gate) {
  extern __shared__ __align__(1024) unsigned char storage[];
  auto& tensors = *reinterpret_cast<Mainloop::TensorStorage*>(storage);
  auto* barriers = reinterpret_cast<uint64_t*>(storage + sizeof(Mainloop::TensorStorage));
  auto sA = make_tensor(make_smem_ptr(tensors.smem_A.data()), Mainloop::SmemLayoutA{});
  auto sB = make_tensor(make_smem_ptr(tensors.smem_B.data()), Mainloop::SmemLayoutB{});
  using Half = cutlass::half_t;
  auto bg_layout_a = tile_to_shape(G::Layout_K_SW128_Atom<Half>{}, Shape<_64, _64>{});
  auto bg_layout_b = tile_to_shape(G::Layout_MN_SW128_Atom<Half>{}, Shape<_128, _64>{});
  auto* bg_storage = reinterpret_cast<Half*>(storage+sizeof(Mainloop::TensorStorage)+1024);
  auto bg_a = make_tensor(make_smem_ptr(bg_storage), bg_layout_a);
  auto bg_b = make_tensor(make_smem_ptr(bg_storage+4096), bg_layout_b);
  for (int q=threadIdx.x; q<4096; q+=Threads) bg_a(q/64,q%64)=Half(background==2 ? .0625f : 0.f);
  for (int q=threadIdx.x; q<8192; q+=Threads) bg_b(q/64,q%64)=Half(background==2 ? .0625f : 0.f);
  auto av = local_tile(bg_a, make_shape(_64{}, _16{}), make_coord(_0{}, _0{}));
  auto bv = local_tile(bg_b, make_shape(_128{}, _16{}), make_coord(_0{}, _0{}));
  uint64_t ad = G::make_gmma_desc<G::Major::K>(av), bd = G::make_gmma_desc<G::Major::MN>(bv);
  __shared__ volatile int bg_continue;
  auto inputs = Mainloop{}.load_init(make_shape(m, n, k, 1), params);
  auto a_slice = params.tma_load_a.get_slice(0);
  auto b_slice = params.tma_load_b.get_slice(0);
  int tile_index = independent ? int(blockIdx.x) : 0;
  int source_phase = (int(blockIdx.x)*source_phase_stride) % (k/64);
  auto gA = get<0>(inputs)(_, _, tile_index, _, 0);
  auto gB = get<1>(inputs)(_, _, tile_index, _, 0);
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
  while (global_ns() < start) {}
  __syncthreads();
  if (threadIdx.x >= 128) {
    if (target_scalar_warp >= 0 && int(blockIdx.x) < tma_ctas) {
      uint32_t a=1, b=2, c=3, d=4;
      unsigned rounds=0;
      if (int(threadIdx.x/32)==target_scalar_warp) {
        if (threadIdx.x%32==0) ctas[blockIdx.x].background_begin=stamp();
        while (__shfl_sync(0xffffffff, int(global_ns()<start+500000), 0)) {
#pragma unroll
          for (int q=0; q<16; ++q) {
            asm volatile("add.u32 %0, %0, 1; add.u32 %1, %1, 3; add.u32 %2, %2, 5; add.u32 %3, %3, 7;"
                         : "+r"(a), "+r"(b), "+r"(c), "+r"(d));
          }
          ++rounds;
        }
        if (threadIdx.x%32==0) {
          ctas[blockIdx.x].background_end=stamp();
          ctas[blockIdx.x].background_rounds=rounds;
        }
      }
      auto* scalar_output=reinterpret_cast<uint32_t*>(background_output);
      size_t offset=(size_t(blockIdx.x)*128+threadIdx.x-128)*64;
      for (int q=0; q<64; ++q) scalar_output[offset+q]=
          q==0 && int(threadIdx.x/32)==target_scalar_warp ? a^b^c^d : 0;
      return;
    }
    float accum[64] = {};
    auto* packed_a = reinterpret_cast<volatile uint32_t const*>(bg_storage);
    uint32_t ra0=packed_a[0], ra1=packed_a[1], ra2=packed_a[2], ra3=packed_a[3];
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
    unsigned rounds = 0;
    if (threadIdx.x == 128) ctas[blockIdx.x].background_begin = stamp();
    for (;;) {
      if (threadIdx.x == 128) bg_continue = global_ns() < start+500000;
      asm volatile("bar.sync 15, 128;" ::: "memory");
      if (!bg_continue) break;
      if (background && (!background_skip_targets || int(blockIdx.x)>=tma_ctas)) {
        if (target_rs && int(blockIdx.x)<tma_ctas) {
#pragma unroll
          for (int q=0; q<16; ++q)
            background_mma_rs(ra0,ra1,ra2,ra3,bd,accum,std::make_index_sequence<64>{});
        } else {
#pragma unroll
          for (int q=0; q<16; ++q) background_mma(ad,bd,accum,std::make_index_sequence<64>{});
        }
        asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 1;" ::: "memory");
        ++rounds;
      }
      asm volatile("bar.sync 15, 128;" ::: "memory");
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    if (threadIdx.x == 128) {
      ctas[blockIdx.x].background_end = stamp();
      ctas[blockIdx.x].background_rounds = rounds;
    }
#pragma unroll
    for (int q=0; q<64; ++q) background_output[(size_t(blockIdx.x)*128+threadIdx.x-128)*64+q]=accum[q];
    return;
  }
  while (global_ns() < start+50000) {}
  asm volatile("bar.sync 14, 128;" ::: "memory");
  if (int(blockIdx.x)>=tma_ctas) {
    if (threadIdx.x==0) {
      ctas[blockIdx.x].begin=stamp();ctas[blockIdx.x].end=stamp();
      unsigned id;asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
      ctas[blockIdx.x].smid=id;
    }
    return;
  }
  int issuer=32*issuer_warp;
  if (!verify && threadIdx.x != issuer) return;
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
  if (threadIdx.x == issuer) {
    for (int g=0; g<active; ++g) issue_group(g);
  }
  for (int g=0; g<iterations; ++g) {
    Point wb{}, we{};
    if (threadIdx.x == issuer) {
      if (trace) wb=stamp();
      Barrier::wait(barriers+g%4, (g/4)%2);
      if (trace) we=stamp();
    }
    if (verify) {
      asm volatile("bar.sync 14, 128;" ::: "memory");
      int row_base=tile_index*128, k_base=((g+source_phase)%(k/64))*64;
      for (int q=threadIdx.x; q<128*64; q+=128) {
        int row=q/64, col=q%64;
        float ea=input_value(row_base+row, k_base+col, seed, true);
        float eb=input_value(k_base+col, row_base+row, seed, false);
        errors += float(sA(row,col,g%4)) != ea;
        errors += float(sB(row,col,g%4)) != eb;
        checked += 2;
      }
      asm volatile("bar.sync 14, 128;" ::: "memory");
    }
    if (threadIdx.x == issuer) {
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
  if (threadIdx.x == issuer) {
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

int main(int argc, char** argv) try {
  int blocks=1, active=1, iterations=8, ring=32, independent=0, verify=1, trace=1;
  int seed=17, delay=0, warmup=0, source_phase_stride=0, ldb=0, background=0;
  int tma_ctas=0, background_skip_targets=0, target_rs=0, target_scalar_warp=-1, issuer_warp=0;
  for (int i=1; i<argc; i+=2) {
    if (i+1==argc) throw std::runtime_error("missing value");
    std::string name=argv[i]; int value=std::stoi(argv[i+1]);
    if (name=="--ctas") blocks=value;
    else if (name=="--active") active=value;
    else if (name=="--groups") iterations=value;
    else if (name=="--ring") ring=value;
    else if (name=="--independent") independent=value;
    else if (name=="--verify") verify=value;
    else if (name=="--trace") trace=value;
    else if (name=="--seed") seed=value;
    else if (name=="--release-delay-ns") delay=value;
    else if (name=="--warmup") warmup=value;
    else if (name=="--source-phase-stride") source_phase_stride=value;
    else if (name=="--ldb") ldb=value;
    else if (name=="--background") background=value;
    else if (name=="--tma-ctas") tma_ctas=value;
    else if (name=="--background-skip-targets") background_skip_targets=value;
    else if (name=="--target-rs") target_rs=value;
    else if (name=="--target-scalar-warp") target_scalar_warp=value;
    else if (name=="--issuer-warp") issuer_warp=value;
    else throw std::runtime_error("unknown argument: "+name);
  }
  if (tma_ctas==0) tma_ctas=blocks;
  if (blocks<1 || tma_ctas<1 || tma_ctas>blocks ||
      (target_scalar_warp!=-1 && (target_scalar_warp<4 || target_scalar_warp>7)) ||
      issuer_warp<0 || issuer_warp>3 ||
      (target_rs!=0 && target_rs!=1) ||
      (background_skip_targets!=0 && background_skip_targets!=1) ||
      (active!=1 && active!=2 && active!=4) || iterations<active || ring<1 ||
      (independent!=0 && independent!=1) || (verify!=0 && verify!=1) ||
      (trace!=0 && trace!=1) || delay<0 || warmup<0 || source_phase_stride<0 || background<0 || background>2) throw std::runtime_error("invalid input");
  cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
  if (prop.major!=9 || !prop.cooperativeLaunch || blocks>prop.multiProcessorCount)
    throw std::runtime_error("requires SM90 cooperative grid within SM count");
  int m=128*(independent?blocks:1), n=m, k=64*ring;
  if (ldb==0) ldb=n;
  if (ldb<n || ldb%8) throw std::runtime_error("ldb must cover N and align to 8 elements");
  DeviceBuffer<cutlass::half_t> a(size_t(m)*k), b(size_t(ldb)*k);
  fill_input<<<128,256>>>(a.pointer,m,k,seed,true,false);
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
  DeviceBuffer<float> background_output(size_t(blocks)*128*64);
  DeviceBuffer<GroupRecord> records(size_t(blocks)*iterations);
  DeviceBuffer<CtaRecord> ctas(blocks); DeviceBuffer<uint64_t> gate(1);
  auto launch=[&]() {
    CUDA_CHECK(cudaMemset(ctas.pointer,0,blocks*sizeof(CtaRecord)));
    void* values[]={&params,&m,&n,&k,&iterations,&active,&independent,&verify,&trace,&seed,&delay,
                   &source_phase_stride,&background,&tma_ctas,&background_skip_targets,&target_rs,
                   &target_scalar_warp,&issuer_warp,
                   &background_output.pointer,&records.pointer,&ctas.pointer,&gate.pointer};
    CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(tensor_probe),dim3(blocks),
                                          dim3(Threads),values,SharedBytes));
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  for (int i=0; i<warmup; ++i) launch();
  launch();
  std::vector<CtaRecord> host(blocks);
  CUDA_CHECK(cudaMemcpy(host.data(),ctas.pointer,blocks*sizeof(CtaRecord),cudaMemcpyDeviceToHost));
  std::vector<GroupRecord> events(size_t(tma_ctas)*iterations);
  if (trace) CUDA_CHECK(cudaMemcpy(events.data(),records.pointer,events.size()*sizeof(GroupRecord),cudaMemcpyDeviceToHost));
  uint64_t total_errors=0,total_checked=0;
  std::set<unsigned> sms;
  for (auto const& c:host) {total_errors+=c.errors;total_checked+=c.checked;sms.insert(c.smid);}
  std::vector<float> bg_values(size_t(blocks)*128*64);
  CUDA_CHECK(cudaMemcpy(bg_values.data(),background_output.pointer,bg_values.size()*sizeof(float),cudaMemcpyDeviceToHost));
  uint64_t bg_errors=0; bool overlap=true;
  for (int c=0; c<blocks; ++c) {
    float expected=background==2 ? float(host[c].background_rounds) : 0.f;
    for (int q=0; q<128*64; ++q) {
      if (target_scalar_warp>=0 && c<tma_ctas) {
        uint32_t rounds=host[c].background_rounds;
        uint32_t wanted=q%64==0 && (q/64+128)/32==target_scalar_warp ?
            (1+16*rounds)^(2+48*rounds)^(3+80*rounds)^(4+112*rounds) : 0;
        uint32_t actual; std::memcpy(&actual,&bg_values[size_t(c)*128*64+q],sizeof(actual));
        bg_errors += actual!=wanted;
      } else bg_errors += bg_values[size_t(c)*128*64+q] != expected;
    }
    overlap &= host[c].background_begin.ns <= host[c].begin.ns && host[c].background_end.ns >= host[c].end.ns;
  }
  char uuid[80]; auto u=reinterpret_cast<unsigned char*>(prop.uuid.bytes);
  std::snprintf(uuid,sizeof(uuid),"GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  std::cout<<"{\"schema\":\"cfg-b-tensor-probe/1\",\"gpu_uuid\":\""<<uuid
    <<"\",\"ctas\":"<<blocks<<",\"active_groups\":"<<active<<",\"groups\":"<<iterations
    <<",\"ring\":"<<ring<<",\"independent\":"<<independent<<",\"verify\":"<<verify<<",\"trace\":"<<trace
    <<",\"seed\":"<<seed<<",\"release_delay_ns\":"<<delay<<",\"warmup\":"<<warmup
    <<",\"source_phase_stride\":"<<source_phase_stride
    <<",\"tma_ctas\":"<<tma_ctas<<",\"background_skip_targets\":"<<background_skip_targets
    <<",\"target_rs\":"<<target_rs
    <<",\"target_scalar_warp\":"<<target_scalar_warp<<",\"issuer_warp\":"<<issuer_warp
    <<",\"background\":"<<background<<",\"background_errors\":"<<bg_errors<<",\"background_overlap\":"<<(overlap?1:0)
    <<",\"lda\":"<<k<<",\"ldb\":"<<ldb
    <<",\"smem\":"<<SharedBytes<<",\"registers\":"<<attr.numRegs<<",\"local_bytes\":"<<attr.localSizeBytes
    <<",\"max_active_ctas_per_sm\":"<<occupancy<<",\"observed_sms\":"<<sms.size()
    <<",\"errors\":"<<total_errors<<",\"checked\":"<<total_checked
    <<",\"descriptor_a\":\""<<descriptor_hex(params.tma_load_a.get_tma_descriptor())
    <<"\",\"descriptor_b\":\""<<descriptor_hex(params.tma_load_b.get_tma_descriptor())<<"\",\"ctas_data\":[";
  for (int c=0;c<blocks;++c) {
    if(c)std::cout<<',';
    std::cout<<"{\"cta\":"<<c<<",\"smid\":"<<host[c].smid<<",\"begin\":";print_point(host[c].begin);
    std::cout<<",\"end\":";print_point(host[c].end);
    std::cout<<",\"background_begin\":";print_point(host[c].background_begin);
    std::cout<<",\"background_end\":";print_point(host[c].background_end);
    std::cout<<",\"background_rounds\":"<<host[c].background_rounds<<'}';
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
  return total_errors || bg_errors || (!verify && !overlap) || sms.size()!=size_t(blocks) ||
         (verify && total_checked!=uint64_t(tma_ctas)*iterations*16384) ? 2 : 0;
} catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
