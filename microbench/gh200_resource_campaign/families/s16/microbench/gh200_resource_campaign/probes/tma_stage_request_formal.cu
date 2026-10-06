#define main ts_short_main
#include "tma_stage_request.cu"
#undef main

static void ts_write_identity(const std::string& path, const std::string& text) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0) throw std::runtime_error("exclusive S16 identity file creation failed");
  struct Close { int fd; ~Close() { ::close(fd); } } close{fd};
  size_t offset = 0;
  while (offset < text.size()) {
    const auto count = ::write(fd, text.data() + offset, text.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::runtime_error("S16 identity file write failed");
    offset += size_t(count);
  }
  if (::fsync(fd) != 0) throw std::runtime_error("S16 identity file fsync failed");
}

// Host extension only: the original 18 device entry points and four-launch
// short check are included unchanged. The measured path sets capture=false.
int ts_formal_main(int argc, char** argv) try {
  const auto device = gh::device();
  gh::emit_device(device);
  const std::string mode = argc > 1 ? argv[1] : "";
  const bool resources_only = mode == "resources";
  const bool pilot = mode == "pilot-only";
  if ((resources_only && argc != 3) ||
      (!resources_only && (argc != 5 || (!pilot && mode != "formal-only"))))
    throw std::runtime_error("resources CASE or pilot-only/formal-only CASE ITERATIONS SEED required");
  const TsCase* selected = nullptr;
  for (const auto& item : ts_cases)
    if (item.id == std::string(argv[2])) selected = &item;
  if (!selected) throw std::runtime_error("unknown original S16 coordinate");
  const auto& c = *selected;
  namespace ref = tma_stage_reference;
  const auto shared = ref::shared(c.stages, c.requests);
  cudaFuncAttributes attributes{};
  GH_CUDA(cudaFuncGetAttributes(&attributes, c.function));
  int occupancy = 0;
  const bool capacity_ok = shared + attributes.sharedSizeBytes <= device.prop.sharedMemPerBlockOptin;
  if (capacity_ok) {
    GH_CUDA(cudaFuncSetAttribute(c.function, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    GH_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, c.function, ts_T, shared));
  }
  const bool legal = capacity_ok && occupancy > 0 && attributes.localSizeBytes == 0;
  if (resources_only) {
    std::cout << "{\"schema_version\":2,\"type\":\"resources\",\"case_id\":" << gh::quote(c.id)
      << ",\"legal\":" << (legal ? "true" : "false") << ",\"target_launches\":0"
      << ",\"kernel_symbol\":" << gh::quote(c.symbol)
      << ",\"software_stages\":" << c.stages << ",\"requests_per_item\":" << c.requests
      << ",\"dynamic_smem_bytes\":" << shared << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
      << ",\"registers_per_thread\":" << attributes.numRegs << ",\"local_size_bytes\":" << attributes.localSizeBytes
      << ",\"occupancy_limit_ctas_per_sm\":" << occupancy
      << ",\"smem_per_cta_optin_bytes\":" << device.prop.sharedMemPerBlockOptin
      << ",\"reason\":" << gh::quote(legal ? "legal_capacity_and_occupancy_bound" :
          !capacity_ok ? "shared_capacity_exceeded" : attributes.localSizeBytes ? "local_memory_not_admitted" : "occupancy_zero")
      << "}\n";
    return 0;
  }
  if (!legal) throw std::runtime_error("S16 resource_reject_before_launch");
  unsigned iterations = gh::integer(argv[3], 128, pilot ? 128 : 65536);
  unsigned seed = gh::integer(argv[4], 0, 4294967295ull);
  const unsigned blocks = c.all_gpu ? device.prop.multiProcessorCount * std::min(4, occupancy) : 1;
  const gh::u64 allocation = ref::allocation(blocks, c.requests);
  const gh::u64 ring_words = (allocation - 32) / 4;
  const gh::u64 final_words = gh::u64(blocks) * c.stages * c.requests * ts_W;
  const gh::u64 other = final_words * 4 + blocks * (sizeof(gh::Stamp) + 12 * sizeof(std::uint64_t));
  size_t free_bytes = 0, total_bytes = 0;
  GH_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (allocation > free_bytes || other > free_bytes - allocation)
    throw std::runtime_error("S16 complete output device memory budget");
  unsigned *global = nullptr, *final_slots = nullptr;
  gh::Stamp* stamps = nullptr;
  std::uint64_t* counts = nullptr;
  GH_CUDA(cudaMalloc(&global, allocation));
  GH_CUDA(cudaMalloc(&final_slots, final_words * 4));
  GH_CUDA(cudaMalloc(&stamps, blocks * sizeof(gh::Stamp)));
  GH_CUDA(cudaMalloc(&counts, blocks * 12 * sizeof(std::uint64_t)));
  std::vector<unsigned> initial(ring_words + 8), final_poison(final_words);
  for (unsigned k = 0; k < 4; ++k) {
    initial[k] = 0xd15ea5e0u + k;
    initial[ring_words + 4 + k] = 0xd15ea5e4u + k;
  }
  for (unsigned b = 0; b < blocks; ++b)
    for (unsigned slot = 0; slot < ts_G; ++slot)
      for (unsigned request = 0; request < c.requests; ++request) {
        const gh::u64 at = 4 + ((gh::u64(b) * ts_G + slot) * c.requests + request) * ts_W;
        for (unsigned word = 0; word < ts_W; ++word)
          initial[at + word] = c.g2s ? ref::g2s(b, slot, request, word, c.requests, seed) :
            ~ref::s2g(b, slot % c.stages, request, word, c.stages, c.requests, seed);
      }
  for (unsigned b = 0; b < blocks; ++b)
    for (unsigned slot = 0; slot < c.stages; ++slot)
      for (unsigned request = 0; request < c.requests; ++request) {
        const gh::u64 at = ((gh::u64(b) * c.stages + slot) * c.requests + request) * ts_W;
        for (unsigned word = 0; word < ts_W; ++word)
          final_poison[at + word] = ~ref::final(c.g2s, b, slot, request, word,
                                             c.stages, c.requests, iterations, seed);
      }
  std::vector<unsigned> ring(ring_words + 8), finals(final_words);
  std::vector<std::uint64_t> counters(blocks * 12);
  cudaEvent_t begin, end;
  GH_CUDA(cudaEventCreate(&begin));
  GH_CUDA(cudaEventCreate(&end));
  unsigned* payload = global + 4;
  unsigned* trace = nullptr;
  std::uint64_t* lifecycle = nullptr;
  bool capture = false;
  void* args[] = {&payload, &iterations, &seed, &capture, &trace, &final_slots, &stamps, &counts, &lifecycle};
  unsigned invocation = 0;
  TsCheck final_check;
  auto execute = [&](bool save) {
    ++invocation;
    GH_CUDA(cudaMemcpy(global, initial.data(), allocation, cudaMemcpyHostToDevice));
    GH_CUDA(cudaMemcpy(final_slots, final_poison.data(), final_words * 4, cudaMemcpyHostToDevice));
    GH_CUDA(cudaMemset(stamps, 0xff, blocks * sizeof(gh::Stamp)));
    GH_CUDA(cudaMemset(counts, 0xff, blocks * 12 * sizeof(std::uint64_t)));
    GH_CUDA(cudaEventRecord(begin));
    GH_CUDA(cudaLaunchKernel(c.function, dim3(blocks), dim3(ts_T), args, shared));
    GH_CUDA(cudaGetLastError());
    GH_CUDA(cudaEventRecord(end));
    GH_CUDA(cudaEventSynchronize(end));
    gh::Observation observation;
    float elapsed = 0;
    GH_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
    observation.event_ms = elapsed;
    observation.stamps.resize(blocks);
    GH_CUDA(cudaMemcpy(observation.stamps.data(), stamps, blocks * sizeof(gh::Stamp), cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(counters.data(), counts, blocks * 12 * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(ring.data(), global, allocation, cudaMemcpyDeviceToHost));
    GH_CUDA(cudaMemcpy(finals.data(), final_slots, final_words * 4, cudaMemcpyDeviceToHost));
    auto compare = [&](unsigned actual, unsigned expected) {
      ++observation.checked_elements;
      observation.errors += actual != expected;
    };
    for (unsigned k = 0; k < 4; ++k) {
      compare(ring[k], 0xd15ea5e0u + k);
      compare(ring[ring_words + 4 + k], 0xd15ea5e4u + k);
    }
    for (unsigned b = 0; b < blocks; ++b) {
      for (unsigned slot = 0; slot < ts_G; ++slot)
        for (unsigned request = 0; request < c.requests; ++request) {
          const gh::u64 at = 4 + ((gh::u64(b) * ts_G + slot) * c.requests + request) * ts_W;
          for (unsigned word = 0; word < ts_W; ++word)
            compare(ring[at + word], ref::ring(c.g2s, b, slot, request, word,
                                             c.stages, c.requests, iterations, seed));
        }
      for (unsigned slot = 0; slot < c.stages; ++slot)
        for (unsigned request = 0; request < c.requests; ++request) {
          const gh::u64 at = ((gh::u64(b) * c.stages + slot) * c.requests + request) * ts_W;
          for (unsigned word = 0; word < ts_W; ++word)
            compare(finals[at + word], ref::final(c.g2s, b, slot, request, word,
                                                c.stages, c.requests, iterations, seed));
        }
      const auto& stamp = observation.stamps[b];
      const auto sentinel = std::numeric_limits<gh::u64>::max();
      observation.errors += stamp.begin_ns == sentinel || stamp.end_ns == sentinel ||
        stamp.begin_cycle == sentinel || stamp.end_cycle == sentinel || stamp.smid == ~unsigned(0) ||
        stamp.end_ns <= stamp.begin_ns || stamp.end_cycle <= stamp.begin_cycle;
      observation.checked_elements += 10;
      const std::uint64_t expected[] = {std::uint64_t(iterations) * c.requests, c.g2s ? 0u : iterations,
        c.g2s ? iterations : 0u, iterations, 0, iterations,
        iterations > c.stages ? iterations - c.stages : 0u, c.g2s ? 0u : 1u, 0, 0,
        c.g2s ? c.stages : 0u, c.g2s ? c.stages : 0u};
      for (unsigned field = 0; field < 12; ++field) {
        const auto value = counters[gh::u64(b) * 12 + field];
        if (field == 9 && c.g2s) {
          observation.errors += value < iterations || value == sentinel;
          observation.checked_elements += 2;
        } else {
          compare(unsigned(value), unsigned(expected[field]));
          compare(unsigned(value >> 32), unsigned(expected[field] >> 32));
        }
      }
    }
    observation.method = "S16_complete32slot_ring_finalSMEM_allCTA_counts_and_stamps_v1";
    observation.input_conditions = "nonuniform_uint32_stage_request_pattern;capture_false;32slot_ring;exact";
    std::string validation_error;
    try { gh::envelope(observation); }
    catch (const std::exception& error) { validation_error = error.what(); }
    if (save || !validation_error.empty()) {
      TsCheck saved;
      saved.checked = observation.checked_elements;
      ts_save(saved, invocation, "ring_guards.u32le", ring, {ring_words + 8});
      ts_save(saved, invocation, "final_slots.u32le", finals, {blocks, c.stages, c.requests, ts_W});
      ts_save(saved, invocation, "counts.u32le", word_artifacts::split_u64(counters), {blocks, 12, 2});
      std::vector<std::uint64_t> encoded;
      for (const auto& stamp : observation.stamps)
        encoded.insert(encoded.end(), {stamp.begin_ns, stamp.end_ns, stamp.begin_cycle, stamp.end_cycle, stamp.smid});
      ts_save(saved, invocation, "stamps.u32le", word_artifacts::split_u64(encoded), {blocks, 5, 2});
      std::ostringstream metadata;
      metadata << "{\"case_id\":" << gh::quote(c.id) << ",\"phase\":" << gh::quote(pilot ? "pilot" : "formal")
        << ",\"iterations\":" << iterations << ",\"seed\":" << seed << ",\"invocation\":" << invocation
        << ",\"capture\":false,\"errors\":" << observation.errors
        << ",\"checked_elements\":" << observation.checked_elements << ",\"blocks\":" << blocks
        << ",\"software_stages\":" << c.stages << ",\"requests_per_item\":" << c.requests
        << ",\"validation_failed\":" << (validation_error.empty() ? "false" : "true")
        << ",\"diagnostic\":" << gh::quote(validation_error) << "}";
      ts_write_identity("stage_" + std::to_string(invocation) + "_identity.json", metadata.str() + "\n");
      if (save) final_check = std::move(saved);
    }
    if (!validation_error.empty()) throw std::runtime_error(validation_error);
    return observation;
  };
  gh::Warmup warm;
  if (!pilot) warm = gh::warmup([&]() { return execute(false); });
  const auto observed = execute(true);
  std::ostringstream extra;
  extra << "\"phase\":" << gh::quote(pilot ? "pilot" : "formal")
    << ",\"performance_eligible\":" << (pilot ? "false" : "true")
    << ",\"warmup_executed\":" << (pilot ? "false" : "true") << ",\"capture\":false"
    << ",\"software_stages\":" << c.stages << ",\"requests_per_item\":" << c.requests
    << ",\"payload_bytes\":16384,\"global_slots_per_cta\":32"
    << ",\"kernel_symbol\":" << gh::quote(c.symbol)
    << ",\"registers_per_thread\":" << attributes.numRegs << ",\"static_smem_bytes\":" << attributes.sharedSizeBytes
    << ",\"dynamic_smem_bytes\":" << shared << ",\"local_size_bytes\":" << attributes.localSizeBytes
    << ",\"occupancy_limit_ctas_per_sm\":" << occupancy
    << ",\"post_timing_final_export_bytes\":" << final_words * 4
    << ",\"device_timer_boundary\":\"pipeline_fill_steady_drain_and_allCTA_consumer_release\""
    << ",\"event_timer_boundary\":\"complete_kernel_including_initialization_and_final_export\""
    << ",\"final_invocation\":" << invocation << ",\"full_output_artifacts\":[";
  for (size_t index = 0; index < final_check.artifacts.size(); ++index) {
    const auto& item = final_check.artifacts[index];
    extra << (index ? "," : "") << "{\"path\":" << gh::quote(item.path) << ",\"sha256\":" << gh::quote(item.sha)
      << ",\"dtype\":\"uint32\",\"shape\":[";
    for (size_t axis = 0; axis < item.shape.size(); ++axis) extra << (axis ? "," : "") << item.shape[axis];
    extra << "]}";
  }
  extra << ']';
  const auto work = ref::payload(blocks, iterations, c.requests);
  gh::emit_trial(c.id, iterations, seed, ts_T, c.all_gpu ? "all_gpu" : "one_cta", "byte",
                 work, c.g2s ? work : 0, c.g2s ? 0 : work, observed, warm, extra.str());
  GH_CUDA(cudaEventDestroy(begin));
  GH_CUDA(cudaEventDestroy(end));
  GH_CUDA(cudaFree(counts));
  GH_CUDA(cudaFree(stamps));
  GH_CUDA(cudaFree(final_slots));
  GH_CUDA(cudaFree(global));
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 2;
}

int main(int argc, char** argv) {
  if (argc > 1 && (std::string(argv[1]) == "validate-only" || std::string(argv[1]) == "device"))
    return ts_short_main(argc, argv);
  return ts_formal_main(argc, argv);
}
