#include "shape_policy.hpp"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " M N K\n";
    return 2;
  }

  gemm_shape_opt::Shape shape{
      std::atoi(argv[1]),
      std::atoi(argv[2]),
      std::atoi(argv[3]),
  };
  const auto features = gemm_shape_opt::classify_shape(shape);
  std::cout << "class=" << gemm_shape_opt::to_string(features.shape_class)
            << " m_tail=" << features.m_tail
            << " n_tail=" << features.n_tail
            << " k_tail=" << features.k_tail
            << " flops=" << features.flops
            << " default_backends="
            << gemm_shape_opt::default_backend_suite(features.shape_class)
            << '\n';
  return 0;
}
