#include "transformer_common.cuh"

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
  TransformerShape shape{1, 1024, 4096, 32, 128};
  if (argc == 6) {
    shape.batch = std::atoi(argv[1]);
    shape.seq_len = std::atoi(argv[2]);
    shape.hidden = std::atoi(argv[3]);
    shape.num_heads = std::atoi(argv[4]);
    shape.head_dim = std::atoi(argv[5]);
  } else if (argc != 1) {
    std::cerr << "Usage: " << argv[0]
              << " [batch seq_len hidden num_heads head_dim]\n";
    return EXIT_FAILURE;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::cout << "Transformer optimization workspace\n";
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  print_shape(shape);
  std::cout << "Benchmark policy: warmup=" << kTransformerWarmup
            << ", timed repeats=" << kTransformerRepeat << '\n';
  std::cout << "Planned operators: LayerNorm, Softmax, QKV projection, "
               "Attention, FFN, KV cache\n";
  return 0;
}
