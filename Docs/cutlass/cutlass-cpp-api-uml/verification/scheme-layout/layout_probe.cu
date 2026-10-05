// Host-only checks of fixed CUTLASS layout APIs. No GEMM/GPU kernel is launched.
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <cutlass/detail/layout.hpp>
#include <cutlass/layout/matrix.h>
#include <cute/layout.hpp>

int main() {
  constexpr int M = 256, N = 256, K = 128, L = 1;
  using AStride = cutlass::detail::TagToStrideA_t<cutlass::layout::RowMajor>;
  using BStride = cutlass::detail::TagToStrideB_t<cutlass::layout::RowMajor>;
  static_assert(std::is_same_v<AStride, cute::Stride<int64_t, cute::Int<1>, int64_t>>);
  static_assert(std::is_same_v<BStride, cute::Stride<cute::Int<1>, int64_t, int64_t>>);

  int checked = 0;
  int packed_mismatches = 0;
  int b_coordinate_mismatches = 0;
  for (int padding : {0, 8}) {
    const int64_t lda = K + padding, ldb = N + padding;
    const auto a = cute::make_layout(cute::make_shape(M, K, L),
                                    AStride{lda, cute::Int<1>{}, int64_t(0)});
    const auto b = cute::make_layout(cute::make_shape(N, K, L),
                                    BStride{cute::Int<1>{}, ldb, int64_t(0)});
    const cutlass::layout::RowMajor mathematical_a(lda);
    const cutlass::layout::RowMajor mathematical_b(ldb);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        const auto offset = a(cute::make_coord(m, k, 0));
        if (offset != int64_t(m) * lda + k ||
            offset != mathematical_a(cutlass::MatrixCoord(m, k))) return 1;
        if (padding && offset != int64_t(m) * K + k) ++packed_mismatches;
        ++checked;
      }
    }
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        // CuTe B uses (n,k,l); mathematical B uses (k,n).
        const auto offset = b(cute::make_coord(n, k, 0));
        if (offset != int64_t(k) * ldb + n ||
            offset != mathematical_b(cutlass::MatrixCoord(k, n))) return 2;
        if (padding && offset != int64_t(k) * N + n) ++packed_mismatches;
        if (offset != int64_t(n) * K + k) ++b_coordinate_mismatches;
        ++checked;
      }
    }
    std::printf("padding=%d lda=%lld ldb=%lld A(1,0)=%lld B(1,1)=%lld\n",
                padding, static_cast<long long>(lda), static_cast<long long>(ldb),
                static_cast<long long>(a(cute::make_coord(1, 0, 0))),
                static_cast<long long>(b(cute::make_coord(1, 1, 0))));
  }
  if (!packed_mismatches || !b_coordinate_mismatches) return 3;
  std::printf("HOST_LAYOUT_PASS checked=%d ignored_padding_mismatches=%d wrong_B_coordinate_mismatches=%d gpu_kernel_launched=false\n",
              checked, packed_mismatches, b_coordinate_mismatches);
}
