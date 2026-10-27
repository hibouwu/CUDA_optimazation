// Same allocation; four arms pair stage and source-to-SM binding.
#define GH200_BATCH_SOURCE
#define GH200_SM_SOURCE_BINDING
#define GH200_NATIVE_TMA
#define GH200_MULTI_CTA_TMA
#define main q02_unused_single_cta_main
#include "cfg_b_tma_chain.cu"
#undef main
#include <algorithm>

struct Q02Options {
  int ctas = 1, stages = 6;
#ifdef GH200_BATCH_SOURCE
  int source_mode = 1;
  int source_binding = 0;
#endif
  U32 seed = 3;
  Q02Options(int argc, char** argv) {
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc) throw std::runtime_error("missing option value");
      std::string key = argv[i], value = argv[i + 1];
      if (key == "--ctas") ctas = int(integer(value, 132));
      else if (key == "--stages") stages = int(integer(value, 6));
      else if (key == "--seed") seed = U32(integer(value, std::numeric_limits<U32>::max()));
#ifdef GH200_BATCH_SOURCE
      else if (key == "--source-mode") source_mode = int(integer(value, 1));
#endif
      else throw std::runtime_error("unknown option: " + key);
    }
    if ((ctas != 1 && ctas != 132) || (stages != 4 && stages != 6))
      throw std::runtime_error("ctas must be 1/132 and stages 4/6");
  }
};

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "--inspect-program" && std::string(argv[2]) == "1") {
      std::cout << "{\"schema\":\"gh200-q02-cta-program/1\",\"cuda_api_calls\":0,"
          "\"ktiles\":13,\"source_tiles\":32,\"physical_slots\":6,\"threads\":256,"
          "\"dynamic_smem_bytes\":196608,\"tma_requests_per_tile\":3,\"wgmma_per_tile\":8,"
          "\"shared_source_bytes\":1048576,\"cta_record_stride\":" << sizeof(Record)
          << ",\"cta_output_stride_bytes\":" << OutputElements * sizeof(float)
#ifdef GH200_BATCH_SOURCE
          << ",\"source_batches\":132,\"source_batch_stride_bytes\":524288,\"source_allocation_bytes\":138412032"
#endif
          << ",\"paired_arms\":4,\"source_bindings\":[\"cta\",\"smid\"]}\n";
      return 0;
    }
    Q02Options options(argc, argv);
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count != 1) throw std::runtime_error("exactly one visible GPU required");
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));
    if (device.major != 9 || device.minor != 0 || device.multiProcessorCount != 132 ||
        std::string(device.name).find("GH200") == std::string::npos)
      throw std::runtime_error("GH200 with 132 SMs required");
    auto kernel = tma_chain<false>;
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SmemBytes));
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    int occupancy = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, Threads, SmemBytes));
    if (occupancy != 1 || attributes.localSizeBytes != 0)
      throw std::runtime_error("one CTA/SM capacity and zero local memory required");
    std::ostringstream uuid;
    uuid << "GPU-";
    for (int i = 0; i < 16; ++i) {
      if (i == 4 || i == 6 || i == 8 || i == 10) uuid << '-';
      uuid << std::hex << std::setw(2) << std::setfill('0')
          << unsigned(static_cast<unsigned char>(device.uuid.bytes[i]));
    }
    if (uuid.str() != "GPU-43269fbc-449d-3e0f-908a-9c81229546d3")
      throw std::runtime_error("reference GPU required");
    std::vector<cutlass::half_t> a, b;
    fill_sources(a, b, 32, options.seed, true);
#ifdef GH200_BATCH_SOURCE
    constexpr int SourceBatches = 132;
#else
    constexpr int SourceBatches = 1;
#endif
    DeviceBuffer<cutlass::half_t> source_a(a.size() * SourceBatches), source_b(b.size() * SourceBatches);
    auto upload = [&]() {
      for (int batch = 0; batch < SourceBatches; ++batch) {
        CUDA_CHECK(cudaMemcpy(source_a.pointer + batch * a.size(), a.data(), a.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(source_b.pointer + batch * b.size(), b.data(), b.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
      }
    };
    upload();
    Mainloop::Arguments arguments{source_a.pointer, cute::make_stride(int64_t(2048), cute::_1{}, int64_t(262144)),
        source_b.pointer, cute::make_stride(cute::_1{}, int64_t(128), int64_t(262144))};
    auto params = Mainloop::to_underlying_arguments(cute::make_shape(128, 128, 2048, SourceBatches), arguments, nullptr);
    DeviceBuffer<Record> records(options.ctas);
    DeviceBuffer<float> output(size_t(options.ctas) * OutputElements);
    std::vector<Record> host_records(options.ctas);
    auto launch = [&]() {
      CUDA_CHECK(cudaMemset(records.pointer, 0xff, size_t(options.ctas) * sizeof(Record)));
      CUDA_CHECK(cudaMemset(output.pointer, 0xff, size_t(options.ctas) * OutputElements * sizeof(float)));
      kernel<<<options.ctas, Threads, SmemBytes>>>(params, 13, options.stages, 32, true,
          records.pointer, nullptr, output.pointer
#ifdef GH200_BATCH_SOURCE
          , options.source_mode, options.source_binding
#endif
          );
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(host_records.data(), records.pointer, size_t(options.ctas) * sizeof(Record), cudaMemcpyDeviceToHost));
      U64 begin = std::numeric_limits<U64>::max(), end = 0;
      for (const auto& record : host_records) {
        if (record.end.ns <= record.begin.ns || record.end.cycle <= record.begin.cycle)
          throw std::runtime_error("invalid CTA timing window");
        begin = std::min(begin, record.begin.ns);
        end = std::max(end, record.end.ns);
      }
      return end - begin;
    };
    struct Arm { int stages, binding; };
    std::vector<Arm> arms{{4,0},{6,1},{6,0},{4,1}};
    if (options.seed == 22) std::reverse(arms.begin(), arms.end());
    if (options.seed == 41) std::rotate(arms.begin(), arms.begin()+2, arms.end());
    std::cout << "{\"schema\":\"gh200-q02-fixed-source-pair/1\",\"arms\":[";
    U64 all_errors = 0;
    for (size_t arm_index = 0; arm_index < arms.size(); ++arm_index) {
      options.stages = arms[arm_index].stages;
      options.source_binding = arms[arm_index].binding;
      std::string prefix = std::string(options.source_binding ? "sm_s" : "cta_s")+std::to_string(options.stages);
      if (arm_index) std::cout << ',';
#ifdef GH200_BATCH_SOURCE
    U64 witness_errors = 0, witness_generation_errors = 0;
    std::vector<std::vector<U32>> witness_smids(2);
    for (int operand = 0; operand < 2; ++operand) {
      for (int batch = 0; batch < SourceBatches; ++batch) {
        std::fill(a.begin(), a.end(), cutlass::half_t(operand == 0 ? (256 + batch) / 256.f : 1 / 32.f));
        std::fill(b.begin(), b.end(), cutlass::half_t(operand == 1 ? (256 + batch) / 256.f : 1 / 32.f));
        CUDA_CHECK(cudaMemcpy(source_a.pointer + batch * a.size(), a.data(), a.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(source_b.pointer + batch * b.size(), b.data(), b.size() * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
      }
      launch();
      std::vector<float> values(size_t(options.ctas) * OutputElements);
      CUDA_CHECK(cudaMemcpy(values.data(), output.pointer, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int cta = 0; cta < options.ctas; ++cta) {
        witness_smids[operand].push_back(host_records[cta].smid_start);
        int batch = options.source_binding ? int(host_records[cta].smid_start) : cta;
        float expected = 26.f * (256 + batch) / 256.f;
        for (int i = 0; i < OutputElements; ++i)
          witness_errors += values[size_t(cta) * OutputElements + i] != expected;
        witness_generation_errors += host_records[cta].generation_errors;
      }
      std::ofstream file(prefix+(operand == 0 ? "_batch_a.f32" : "_batch_b.f32"), std::ios::binary);
      file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
      if (!file) throw std::runtime_error("batch witness output write failed");
    }
    if (witness_errors || witness_generation_errors)
      throw std::runtime_error("batch address witness failed");
    fill_sources(a, b, 32, options.seed, true);
    upload();
#endif
    std::vector<U64> warmup;
    bool stable = false;
    for (int i = 0; i < 30; ++i) {
      warmup.push_back(launch());
      if (warmup.size() >= 8 && tail_cv(warmup) <= 0.02) { stable = true; break; }
    }
    U64 window = launch();
    std::vector<float> values(size_t(options.ctas) * OutputElements);
    CUDA_CHECK(cudaMemcpy(values.data(), output.pointer, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
    auto expected = reference(13, options.stages, options.seed, true);
    U64 errors = 0, generation_errors = 0;
    double max_abs = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      float actual = values[i], want = expected[i % OutputElements];
      if (!std::isfinite(actual) || actual != want) ++errors;
      if (std::isfinite(actual)) max_abs = std::max(max_abs, std::abs(double(actual) - want));
    }
    for (const auto& record : host_records) generation_errors += record.generation_errors;
    std::ofstream file(prefix+"_output.f32", std::ios::binary);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!file) throw std::runtime_error("output write failed");
    std::cout << std::setprecision(17)
#ifdef GH200_BATCH_SOURCE
        << "{\"schema\":\"gh200-q02-source-batch/1\",\"source_mode\":" << options.source_mode
        << ",\"source_binding\":" << options.source_binding << ",\"arm_id\":" << quote(prefix)
        << ",\"source_batches\":132,\"source_batch_stride_bytes\":524288,\"source_allocation_bytes\":138412032,"
        "\"witness_output_files\":[" << quote(prefix+"_batch_a.f32") << ',' << quote(prefix+"_batch_b.f32")
        << "],\"witness_checked_elements\":" << 2 * values.size()
        << ",\"witness_generation_checked\":" << 2 * U64(options.ctas) * 13 * 128
        << ",\"witness_errors\":" << witness_errors << ",\"witness_generation_errors\":" << witness_generation_errors
        << ",\"ctas\":" << options.ctas
#else
        << "{\"schema\":\"gh200-q02-tma-compute-ctas/1\",\"ctas\":" << options.ctas
#endif
        << ",\"stages\":" << options.stages << ",\"ktiles\":13,\"source_tiles\":32,\"seed\":" << options.seed
        << ",\"gpu_uuid\":" << quote(uuid.str()) << ",\"physical_slots\":6,\"threads\":256,"
        "\"dynamic_smem_bytes\":196608,\"static_smem_bytes\":" << attributes.sharedSizeBytes
        << ",\"registers\":" << attributes.numRegs << ",\"local_bytes\":" << attributes.localSizeBytes
        << ",\"occupancy_ctas_per_sm\":" << occupancy << ",\"source_profile\":\"generation\","
        "\"source_row_stride_bytes\":[4096,256],\"shared_source_bytes\":1048576,\"tma_requests_per_tile\":3,"
        "\"wgmma_per_tile\":8,\"full_arrivals\":1,\"empty_arrivals\":128,\"trace\":false,"
        "\"warmup_stable\":" << (stable ? "true" : "false") << ",\"warmup_cv\":" << tail_cv(warmup)
        << ",\"witness_smids\":[";
    for (int operand = 0; operand < 2; ++operand) {
      std::cout << (operand ? ",[" : "[");
      for (size_t cta = 0; cta < witness_smids[operand].size(); ++cta)
        std::cout << (cta ? "," : "") << witness_smids[operand][cta];
      std::cout << ']';
    }
    std::cout << "],\"warmup_ns\":[";
    for (size_t i = 0; i < warmup.size(); ++i) std::cout << (i ? "," : "") << warmup[i];
    std::cout << "],\"window_ns\":" << window << ",\"a_descriptor_hex\":"
        << quote(descriptor_hex(params.tma_load_a.get_tma_descriptor())) << ",\"b_descriptor_hex\":"
        << quote(descriptor_hex(params.tma_load_b.get_tma_descriptor()))
        << ",\"checked_elements\":" << values.size() << ",\"errors\":" << errors
        << ",\"max_abs_error\":" << max_abs << ",\"generation_checked\":" << U64(options.ctas) * 13 * 128
        << ",\"generation_errors\":" << generation_errors << ",\"output_file\":" << quote(prefix+"_output.f32") << ",\"records\":[";
    for (size_t cta = 0; cta < host_records.size(); ++cta) {
      const auto& record = host_records[cta];
      std::cout << (cta ? "," : "") << "{\"cta\":" << cta << ",\"smid_start\":" << record.smid_start
          << ",\"smid_end\":" << record.smid_end << ",\"begin\":";
      print_pair(record.begin); std::cout << ",\"end\":"; print_pair(record.end);
      std::cout << ",\"generation_errors\":" << record.generation_errors << '}';
    }
    std::cout << "]}";
    all_errors += errors+generation_errors;
    }
    std::cout << "]}\n";
    return all_errors ? 2 : 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
