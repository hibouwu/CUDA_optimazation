// Build as an isolated shared module, without trace overlay or observer macros.
#if defined(V06_TRACE) || defined(R18_DUAL_CLOCK) || defined(R18_MATCH_DUAL_LAYOUT)
#error "plain prefix module cannot contain observer instrumentation"
#endif
#if !defined(V06_CFG) || V06_CFG != 1
#error "plain prefix ABI 1 currently supports cfg_b only"
#endif
#define R09_PLAIN_PREFIX_LIBRARY
#include "r18.cu"
#include "r09_plain_prefix.h"

static_assert(Mainloop::DispatchPolicy::Stages == 6, "prefix ABI 1 requires stage6");
namespace {
struct Prefix {
  typename Gemm::Params params;
  std::unique_ptr<DeviceBuffer<unsigned char>> workspace;
};
void report_error(char* error, size_t capacity, const char* message) {
  if (error && capacity) std::snprintf(error, capacity, "%s", message);
}
}
#define R09_EXPORT extern "C" __attribute__((visibility("default")))
R09_EXPORT int r09_plain_prefix_create(const R09PlainPrefixSpec* s, void** handle,
    R09PlainPrefixInfo* info, char* error, size_t capacity) {
  if (handle) *handle = nullptr;
  try {
    if (!s || !handle || !info || s->abi_version != 1)
      throw std::runtime_error("invalid plain-prefix ABI arguments");
    // This is the declared observation target, not a general GEMM service.
    if (s->m != 2816 || s->n != 1536 || s->k != 4096 || s->sm_count != 132 ||
        s->lda != 4096 || s->ldb != 1536 || s->ldd != 1536 || !s->a || !s->b || !s->d)
      throw std::runtime_error("plain-prefix ABI 1 target mismatch");
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 9 || properties.multiProcessorCount != s->sm_count)
      throw std::runtime_error("plain-prefix ABI 1 requires the declared SM90 device");
    typename Kernel::StrideA sa = make_stride(s->lda, Int<1>{}, int64_t(s->m)*s->lda);
    typename Kernel::StrideB sb = make_stride(Int<1>{}, s->ldb, int64_t(s->k)*s->ldb);
    typename Kernel::StrideD sd = make_stride(s->ldd, Int<1>{}, int64_t(s->m)*s->ldd);
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {s->m,s->n,s->k,1},
        {reinterpret_cast<cutlass::half_t*>(s->a),sa,reinterpret_cast<cutlass::half_t*>(s->b),sb},
        {{1.f,0.f},nullptr,sd,reinterpret_cast<float*>(s->d),sd}};
    args.hw_info.device_id = device;
    args.hw_info.sm_count = s->sm_count;
    args.scheduler.max_swizzle_size = 1;
    using RasterOption = decltype(args.scheduler.raster_order);
    args.scheduler.raster_order = RasterOption::Heuristic;
    auto prefix = std::make_unique<Prefix>();
    Gemm gemm;
    cutlass_check(gemm.can_implement(args), "prefix can_implement");
    size_t bytes = Gemm::get_workspace_size(args);
    prefix->workspace = std::make_unique<DeviceBuffer<unsigned char>>(std::max<size_t>(1,bytes));
    cutlass_check(gemm.initialize(args,prefix->workspace->pointer), "prefix initialize");
    prefix->params = gemm.params();
    dim3 grid = Gemm::get_grid_shape(prefix->params);
    if (grid.x != 1 || grid.y != 132 || grid.z != 1)
      throw std::runtime_error("plain-prefix persistent grid mismatch: "+std::to_string(grid.x)+","+std::to_string(grid.y)+","+std::to_string(grid.z));
    *info = {1,6,int(grid.x),int(grid.y),int(grid.z),bytes};
    *handle = prefix.release();
    report_error(error,capacity,"");
    return 0;
  } catch (const std::exception& e) {
    report_error(error,capacity,e.what());
    return 1;
  }
}
R09_EXPORT int r09_plain_prefix_run(void* handle, char* error, size_t capacity) {
  try {
    if (!handle) throw std::runtime_error("null plain-prefix handle");
    cutlass_check(Gemm::run(static_cast<Prefix*>(handle)->params), "prefix run");
    report_error(error,capacity,"");
    return 0;
  } catch (const std::exception& e) {
    report_error(error,capacity,e.what());
    return 1;
  }
}
R09_EXPORT void r09_plain_prefix_destroy(void* handle) { delete static_cast<Prefix*>(handle); }
