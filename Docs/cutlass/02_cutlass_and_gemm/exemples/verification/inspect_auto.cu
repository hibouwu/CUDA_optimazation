// 编译原示例的设备路径，并在 Host 输出实际实例化的类型；不启动 GEMM。
// CASE_ID: 0 Dense，1 NVFP4，2 Grouped，3 MoE，4 Attention（两个 GEMM）。
#ifndef CASE_ID
#define CASE_ID 0
#endif
#define main example_main_not_run
#if CASE_ID == 0
#include "../dense_baseline.cu"
#elif CASE_ID == 1
#include "../nvfp4_block_scaled.cu"
#elif CASE_ID == 2
#include "../grouped_gemm.cu"
#elif CASE_ID == 3
#include "../moe_expert_gemm.cu"
#elif CASE_ID == 4
#include "../attention_unfused.cu"
#else
#error Unsupported CASE_ID
#endif
#undef main
#include <cxxabi.h>
#include <cstdlib>
#include <typeinfo>

template<class T>
void show_type(char const* label) {
  int status = 0;
  char* name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status);
  std::cout << label << '=' << (status == 0 ? name : typeid(T).name()) << '\n';
  std::free(name);
}

template<class Kernel>
void inspect(char const* name) {
  using Mainloop = typename Kernel::CollectiveMainloop;
  using Epilogue = typename Kernel::CollectiveEpilogue;
  std::cout << "CASE=" << name << '\n';
  show_type<typename Mainloop::DispatchPolicy>("Mainloop.DispatchPolicy");
  show_type<typename Mainloop::TileShape>("Mainloop.TileShape");
  show_type<typename Mainloop::TiledMma>("Mainloop.TiledMma");
  show_type<typename Mainloop::TiledMma::MMA_Op>("Mainloop.MMA_Op");
  show_type<typename Kernel::CtaShape_MNK>("Kernel.CtaShape_MNK");
  show_type<typename Kernel::TileScheduler>("Kernel.TileScheduler");
  show_type<typename Epilogue::DispatchPolicy>("Epilogue.DispatchPolicy");
  show_type<typename Epilogue::EpilogueTile>("Epilogue.Tile");
  show_type<Mainloop>("Mainloop.FullType");
  show_type<Epilogue>("Epilogue.FullType");
  std::cout << "Epilogue.SharedStorage.bytes=" << sizeof(typename Epilogue::SharedStorage) << '\n';
  std::cout << "Kernel.SharedStorage.bytes=" << sizeof(typename Kernel::SharedStorage) << '\n';
}

int main() {
#if CASE_ID == 4
  inspect<qk::GemmKernel>("attention_qk");
  inspect<pv::GemmKernel>("attention_pv");
#else
  inspect<GemmKernel>(("case_" + std::to_string(CASE_ID)).c_str());
#endif
}
