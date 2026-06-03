#include "transformer_benchmark.cuh"
#include "transformer_kernels.cuh"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

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

  if (shape.batch <= 0 || shape.seq_len <= 0 || shape.hidden <= 0 ||
      shape.num_heads <= 0 || shape.head_dim <= 0) {
    std::cerr << "All shape values must be positive\n";
    return EXIT_FAILURE;
  }

  const int rows = shape.batch * shape.seq_len;
  const int hidden = shape.hidden;
  const float eps = 1e-5f;
  const size_t tensor_elements = static_cast<size_t>(rows) * hidden;
  const size_t tensor_bytes = tensor_elements * sizeof(float);
  const size_t vector_bytes = static_cast<size_t>(hidden) * sizeof(float);

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

  std::cout << "Transformer LayerNorm benchmark\n";
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";
  print_shape(shape);
  std::cout << "Benchmark policy: warmup=" << kTransformerWarmup
            << ", timed repeats=" << kTransformerRepeat
            << " per backend before moving to the next backend\n";

  std::vector<float> h_input(tensor_elements);
  std::vector<float> h_gamma(hidden);
  std::vector<float> h_beta(hidden);
  std::vector<float> h_ref(tensor_elements);
  std::vector<float> h_out(tensor_elements);
  fill_layernorm_inputs(h_input, h_gamma, h_beta);
  cpu_layernorm(h_input, h_gamma, h_beta, h_ref, rows, hidden, eps);

  float *d_input = nullptr, *d_gamma = nullptr, *d_beta = nullptr;
  float* d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, tensor_bytes));
  CHECK_CUDA(cudaMalloc(&d_gamma, vector_bytes));
  CHECK_CUDA(cudaMalloc(&d_beta, vector_bytes));
  CHECK_CUDA(cudaMalloc(&d_output, tensor_bytes));
  CHECK_CUDA(
      cudaMemcpy(d_input, h_input.data(), tensor_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_gamma, h_gamma.data(), vector_bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(d_beta, h_beta.data(), vector_bytes, cudaMemcpyHostToDevice));

  std::ofstream csv("transformer_benchmark.csv");
  csv << "Operator,Version,Batch,SeqLen,Hidden,NumHeads,HeadDim,TimeMs,"
         "BandwidthGBps,Matched\n";

  dim3 grid(rows);
  dim3 one_thread(1);
  benchmark_layernorm(
      "v1 naive",
      [&]() {
        layernorm_v1_naive<<<grid, one_thread>>>(
            d_input, d_gamma, d_beta, d_output, rows, hidden, eps);
        CHECK_CUDA(cudaGetLastError());
      },
      rows, hidden, d_output, tensor_bytes, h_ref, h_out, csv, shape);

  dim3 block(kTransformerBlockSize);
  benchmark_layernorm(
      "v2 block reduce",
      [&]() {
        layernorm_v2_block_reduce<<<grid, block>>>(
            d_input, d_gamma, d_beta, d_output, rows, hidden, eps);
        CHECK_CUDA(cudaGetLastError());
      },
      rows, hidden, d_output, tensor_bytes, h_ref, h_out, csv, shape);

  if (hidden % 4 == 0) {
    benchmark_layernorm(
        "v3 vectorized float4",
        [&]() {
          layernorm_v3_vectorized<<<grid, block>>>(
              d_input, d_gamma, d_beta, d_output, rows, hidden, eps);
          CHECK_CUDA(cudaGetLastError());
        },
        rows, hidden, d_output, tensor_bytes, h_ref, h_out, csv, shape);
  } else {
    std::cout << "v3 vectorized float4: skipped because hidden must be a "
                 "multiple of 4\n";
    csv << "LayerNorm,v3 vectorized float4 skipped," << shape.batch << ","
        << shape.seq_len << "," << shape.hidden << "," << shape.num_heads
        << "," << shape.head_dim << ",0,0,0\n";
  }

  csv.close();

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_gamma));
  CHECK_CUDA(cudaFree(d_beta));
  CHECK_CUDA(cudaFree(d_output));
  return 0;
}
