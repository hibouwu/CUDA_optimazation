#include <cute/layout.hpp>
#include <cute/layout_composed.hpp>
#include <cute/swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <iostream>
using namespace cute;
constexpr int NumThreadsPerWarpGroup=128;
cute::tuple<bool, uint32_t> spread_arrivals_to_warpgroup(int thread_idx_in_warpgroup, int warp_idx) {
  constexpr uint32_t MaxClusterSize = 16;
  bool is_signaling_thread = (thread_idx_in_warpgroup % (NumThreadsPerWarpGroup / MaxClusterSize)) == 0;
  auto layout = cute::composition(Swizzle<2,0,-2>{},
                                  Layout<Shape<_4,_4>,Stride<_4,_1>>{});
  uint32_t thread_row = warp_idx % 4;
  uint32_t thread_col = (thread_idx_in_warpgroup / 8) % 4;
  uint32_t dst_blockid = layout(thread_row, thread_col);
  return cute::make_tuple(is_signaling_thread, dst_blockid);
}
int main() {
  for (int physical=128; physical<384; ++physical) {
    auto [signal,destination]=spread_arrivals_to_warpgroup(physical%128,physical/32);
    // cluster2x1: both destinations share N=0, hence row/column filtering is true.
    if (signal && destination<2) std::cout << physical << " " << destination << "\n";
  }
}
