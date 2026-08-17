// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef GUIDE_CASE_ID
#define GUIDE_CASE_ID "unknown"
#endif
#ifndef GUIDE_CUTLASS_SHA
#define GUIDE_CUTLASS_SHA "unknown"
#endif
#ifndef GUIDE_TOOLCHAIN_LOCK_SHA
#define GUIDE_TOOLCHAIN_LOCK_SHA "unknown"
#endif
#ifndef GUIDE_SOURCE_SHA
#define GUIDE_SOURCE_SHA "unknown"
#endif

namespace guide {

enum class Status { pass, fail, skip, not_run };

inline char const* status_name(Status value) {
  switch (value) {
    case Status::pass: return "PASS";
    case Status::fail: return "FAIL";
    case Status::skip: return "SKIP";
    case Status::not_run: return "NOT_RUN";
  }
  return "FAIL";
}

struct CaseDescriptor {
  std::string case_id = GUIDE_CASE_ID;
  std::string title;
  std::array<int, 4> problem_mnkl{};
  std::array<int, 3> mma_tile_mnk{};
  std::array<int, 3> instruction_mnk{};
  std::array<int, 3> cluster_mnk{};
  int cta_group = 1;
  int scale_vector_size = 0;
  std::string element_a;
  std::string element_b;
  std::string element_accumulator;
  std::string element_d;
  std::string operand_source;
  std::string mainloop_schedule;
  std::string reference = "independent_cpu_reference";
};

struct VerificationResult {
  Status status = Status::not_run;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::string message;
  std::string gpu_name;
  int compute_major = 0;
  int compute_minor = 0;
};

struct ParsedArgs {
  bool describe = false;
  bool verify = false;
  bool json = false;
  std::uint64_t seed = 20260817;
  std::string json_out;
};

inline std::string json_escape(std::string_view text) {
  std::ostringstream out;
  for (char ch : text) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << ch; break;
    }
  }
  return out.str();
}

template <std::size_t N>
inline void write_array(std::ostream& out, std::array<int, N> const& values) {
  out << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

inline void write_descriptor_json(std::ostream& out, CaseDescriptor const& d) {
  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"case_id\": \"" << json_escape(d.case_id) << "\",\n"
      << "  \"title\": \"" << json_escape(d.title) << "\",\n"
      << "  \"target_arch\": \"sm_110a\",\n"
      << "  \"cutlass_arch_tag\": \"cutlass::arch::Sm100\",\n"
      << "  \"problem_mnkl\": ";
  write_array(out, d.problem_mnkl);
  out << ",\n  \"mma_tile_mnk\": ";
  write_array(out, d.mma_tile_mnk);
  out << ",\n  \"instruction_mnk\": ";
  write_array(out, d.instruction_mnk);
  out << ",\n  \"cluster_mnk\": ";
  write_array(out, d.cluster_mnk);
  out << ",\n"
      << "  \"cta_group\": " << d.cta_group << ",\n"
      << "  \"scale_vector_size\": " << d.scale_vector_size << ",\n"
      << "  \"types\": {\"a\": \"" << json_escape(d.element_a)
      << "\", \"b\": \"" << json_escape(d.element_b)
      << "\", \"accumulator\": \"" << json_escape(d.element_accumulator)
      << "\", \"d\": \"" << json_escape(d.element_d) << "\"},\n"
      << "  \"operand_source\": \"" << json_escape(d.operand_source) << "\",\n"
      << "  \"mainloop_schedule\": \"" << json_escape(d.mainloop_schedule) << "\",\n"
      << "  \"reference\": \"" << json_escape(d.reference) << "\"\n"
      << "}\n";
}

inline void write_result_json(std::ostream& out, CaseDescriptor const& d,
                              VerificationResult const& r, std::uint64_t seed) {
  out << std::setprecision(17)
      << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"case_id\": \"" << json_escape(d.case_id) << "\",\n"
      << "  \"status\": \"" << status_name(r.status) << "\",\n"
      << "  \"target_arch\": \"sm_110a\",\n"
      << "  \"problem_mnkl\": ";
  write_array(out, d.problem_mnkl);
  out << ",\n"
      << "  \"seed\": " << seed << ",\n"
      << "  \"cutlass_git_sha\": \"" << GUIDE_CUTLASS_SHA << "\",\n"
      << "  \"toolchain_lock_sha256\": \"" << GUIDE_TOOLCHAIN_LOCK_SHA << "\",\n"
      << "  \"source_sha256\": \"" << GUIDE_SOURCE_SHA << "\",\n"
      << "  \"reference\": \"" << json_escape(d.reference) << "\",\n"
      << "  \"max_abs_error\": " << r.max_abs_error << ",\n"
      << "  \"max_rel_error\": " << r.max_rel_error << ",\n"
      << "  \"message\": \"" << json_escape(r.message) << "\",\n"
      << "  \"device\": {\"name\": \"" << json_escape(r.gpu_name)
      << "\", \"compute_capability\": \"" << r.compute_major << '.' << r.compute_minor << "\"},\n"
      << "  \"evidence\": {\n"
      << "    \"documented\": true,\n"
      << "    \"source_present\": true,\n"
      << "    \"compile_passed\": true,\n"
      << "    \"ptx_verified\": false,\n"
      << "    \"sass_verified\": false,\n"
      << "    \"runtime_correct\": " << (r.status == Status::pass ? "true" : "false") << ",\n"
      << "    \"performance_measured\": false\n"
      << "  }\n"
      << "}\n";
}

inline ParsedArgs parse_args(int argc, char** argv) {
  ParsedArgs parsed;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--describe") parsed.describe = true;
    else if (arg == "--verify") parsed.verify = true;
    else if (arg == "--json") parsed.json = true;
    else if (arg == "--seed" && i + 1 < argc) parsed.seed = std::stoull(argv[++i]);
    else if (arg == "--json-out" && i + 1 < argc) parsed.json_out = argv[++i];
    else if (arg == "--help") {
      std::cout << "Usage: " << argv[0]
                << " --describe [--json] | --verify [--seed N] [--json-out PATH]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
  }
  if (parsed.describe == parsed.verify) {
    throw std::invalid_argument("select exactly one of --describe or --verify");
  }
  return parsed;
}

template <class VerifyFn>
int run_case_main(int argc, char** argv, CaseDescriptor descriptor, VerifyFn&& verify_fn) {
  try {
    auto args = parse_args(argc, argv);
    if (args.describe) {
      write_descriptor_json(std::cout, descriptor);
      return 0;
    }

    VerificationResult result = verify_fn(args.seed);
    std::ostringstream json;
    write_result_json(json, descriptor, result, args.seed);
    std::cout << json.str();
    if (!args.json_out.empty()) {
      std::filesystem::path path(args.json_out);
      if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
      std::ofstream file(path);
      if (!file) throw std::runtime_error("cannot write " + path.string());
      file << json.str();
    }
    if (result.status == Status::pass) return 0;
    if (result.status == Status::skip || result.status == Status::not_run) return 77;
    return 1;
  } catch (std::exception const& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

}  // namespace guide
