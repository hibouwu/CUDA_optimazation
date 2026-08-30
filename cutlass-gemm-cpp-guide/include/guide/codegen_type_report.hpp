// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <type_traits>

namespace guide {

template <class Type, class = void>
struct HasBlockScaleDetails : std::false_type {};

template <class Type>
struct HasBlockScaleDetails<Type, std::void_t<
    typename Type::ElementSF,
    typename Type::GmemTiledCopySFA,
    typename Type::GmemTiledCopySFB,
    typename Type::SmemLayoutAtomSFA,
    typename Type::SmemLayoutAtomSFB>> : std::true_type {};

template <class Type, class = void>
struct HasSparseMetadataDetails : std::false_type {};

template <class Type>
struct HasSparseMetadataDetails<Type, std::void_t<
    typename Type::SparseConfig,
    typename Type::SmemLayoutAtomE>> : std::true_type {};

inline std::string escape_codegen_json(std::string const& value) {
  std::string result;
  result.reserve(value.size());
  for (char ch : value) {
    switch (ch) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default: result += ch; break;
    }
  }
  return result;
}

template <class Type>
std::string codegen_type_name() {
  int status = 0;
  std::unique_ptr<char, decltype(&std::free)> name(
      abi::__cxa_demangle(typeid(Type).name(), nullptr, nullptr, &status), &std::free);
  if (status != 0 || !name) {
    return std::string("abi-mangled:") + typeid(Type).name();
  }
  return name.get();
}

template <class Config>
void write_codegen_type_report(std::ostream& out, std::string const& instance_id) {
  using Mainloop = typename Config::CollectiveMainloop;
  using Epilogue = typename Config::CollectiveEpilogue;
  using Kernel = typename Config::GemmKernel;
  using Dispatch = typename Mainloop::DispatchPolicy;
  using DispatchSchedule = typename Dispatch::Schedule;
  using TiledMma = typename Mainloop::TiledMma;
  using MmaAtom = typename TiledMma::Atom;

  auto field = [&](char const* name, std::string const& value, bool comma = true) {
    out << "    \"" << name << "\": \"" << escape_codegen_json(value) << '"';
    if (comma) out << ',';
    out << '\n';
  };

  out << "{\n"
      << "  \"schema_version\": 1,\n"
      << "  \"instance_id\": \"" << escape_codegen_json(instance_id) << "\",\n"
      << "  \"resolved_types\": {\n";
  field("collective_mainloop", codegen_type_name<Mainloop>());
  field("mainloop_builder", codegen_type_name<typename Config::MainloopBuilder>());
  field("mainloop_builder_collective_op",
        codegen_type_name<typename Config::MainloopBuilder::CollectiveOp>());
  field("epilogue_builder", codegen_type_name<typename Config::EpilogueBuilder>());
  field("epilogue_builder_collective_op",
        codegen_type_name<typename Config::EpilogueBuilder::CollectiveOp>());
  field("dispatch_policy", codegen_type_name<Dispatch>());
  field("dispatch_schedule", codegen_type_name<DispatchSchedule>());
  field("tiled_mma", codegen_type_name<TiledMma>());
  field("mma_atom", codegen_type_name<MmaAtom>());
  field("mainloop_dispatch_policy", codegen_type_name<typename Mainloop::DispatchPolicy>());
  field("mainloop_tiled_mma", codegen_type_name<typename Mainloop::TiledMma>());
  field("tiled_mma_atom", codegen_type_name<typename TiledMma::Atom>());
  field("mma_value_type_a", codegen_type_name<typename TiledMma::ValTypeA>());
  field("mma_value_type_b", codegen_type_name<typename TiledMma::ValTypeB>());
  field("mma_value_type_c", codegen_type_name<typename TiledMma::ValTypeC>());
  field("gmem_tiled_copy_a", codegen_type_name<typename Mainloop::GmemTiledCopyA>());
  field("gmem_tiled_copy_b", codegen_type_name<typename Mainloop::GmemTiledCopyB>());
  field("smem_layout_atom_a", codegen_type_name<typename Mainloop::SmemLayoutAtomA>());
  field("smem_layout_atom_b", codegen_type_name<typename Mainloop::SmemLayoutAtomB>());
  field("smem_copy_atom_a", codegen_type_name<typename Mainloop::SmemCopyAtomA>());
  field("smem_copy_atom_b", codegen_type_name<typename Mainloop::SmemCopyAtomB>());
  field("collective_epilogue", codegen_type_name<Epilogue>());
  field("kernel_collective_mainloop", codegen_type_name<typename Kernel::CollectiveMainloop>());
  field("kernel_collective_epilogue", codegen_type_name<typename Kernel::CollectiveEpilogue>());
  field("config_arch_tag", codegen_type_name<typename Config::ArchTag>());
  field("config_operator_class", codegen_type_name<typename Config::OperatorClass>());
  field("config_element_a", codegen_type_name<typename Config::ElementA>());
  field("config_element_b", codegen_type_name<typename Config::ElementB>());
  field("config_element_accumulator", codegen_type_name<typename Config::ElementAccumulator>());
  field("config_element_d", codegen_type_name<typename Config::ElementD>());
  field("config_layout_a", codegen_type_name<typename Config::LayoutA>());
  field("config_layout_b", codegen_type_name<typename Config::LayoutB>());
  field("config_layout_d", codegen_type_name<typename Config::LayoutD>());
  field("config_mainloop_schedule", codegen_type_name<typename Config::MainloopSchedule>());
  field("config_epilogue_schedule", codegen_type_name<typename Config::EpilogueSchedule>());
  field("config_stage_policy", codegen_type_name<typename Config::StagePolicy>());
  field("config_problem_shape", codegen_type_name<typename Config::ProblemShape>());
  field("config_tile_scheduler", codegen_type_name<typename Config::TileScheduler>());
  field("gemm_kernel", codegen_type_name<Kernel>(), false);
  out << "  },\n"
      << "  \"resolved_optional_types\": {\n";
  bool optional_first = true;
  auto optional_field = [&](char const* name, std::string const& value) {
    if (!optional_first) out << ",\n";
    out << "    \"" << name << "\": \"" << escape_codegen_json(value) << '"';
    optional_first = false;
  };
  if constexpr (HasBlockScaleDetails<Mainloop>::value) {
    optional_field("scale_element", codegen_type_name<typename Mainloop::ElementSF>());
    optional_field("gmem_tiled_copy_sfa", codegen_type_name<typename Mainloop::GmemTiledCopySFA>());
    optional_field("gmem_tiled_copy_sfb", codegen_type_name<typename Mainloop::GmemTiledCopySFB>());
    optional_field("smem_layout_atom_sfa", codegen_type_name<typename Mainloop::SmemLayoutAtomSFA>());
    optional_field("smem_layout_atom_sfb", codegen_type_name<typename Mainloop::SmemLayoutAtomSFB>());
  }
  if constexpr (HasSparseMetadataDetails<Mainloop>::value) {
    optional_field("sparse_config", codegen_type_name<typename Mainloop::SparseConfig>());
    optional_field("smem_layout_atom_e", codegen_type_name<typename Mainloop::SmemLayoutAtomE>());
    optional_field("element_e", codegen_type_name<typename Mainloop::ElementE>());
    optional_field("gmem_copy_atom_e", codegen_type_name<typename Mainloop::GmemCopyAtomE>());
    optional_field("smem_layout_e", codegen_type_name<typename Mainloop::SmemLayoutE>());
  }
  if (!optional_first) out << '\n';
  out << "  },\n"
      << "  \"resolved_values\": {\n"
      << "    \"mainloop_stages\": " << Dispatch::Stages << ",\n"
      << "    \"scheduler_stages\": " << DispatchSchedule::SchedulerPipelineStageCount << ",\n"
      << "    \"accumulator_stages\": " << DispatchSchedule::AccumulatorPipelineStageCount << ",\n"
      << "    \"atom_shape_mnk\": ["
      << static_cast<int>(cute::size<0>(typename TiledMma::AtomShape_MNK{})) << ','
      << static_cast<int>(cute::size<1>(typename TiledMma::AtomShape_MNK{})) << ','
      << static_cast<int>(cute::size<2>(typename TiledMma::AtomShape_MNK{})) << "],\n"
      << "    \"mma_tile_mnk\": ["
      << static_cast<int>(cute::size<0>(typename Config::MmaTileShape{})) << ','
      << static_cast<int>(cute::size<1>(typename Config::MmaTileShape{})) << ','
      << static_cast<int>(cute::size<2>(typename Config::MmaTileShape{})) << "],\n"
      << "    \"cluster_mnk\": ["
      << static_cast<int>(cute::size<0>(typename Config::ClusterShape{})) << ','
      << static_cast<int>(cute::size<1>(typename Config::ClusterShape{})) << ','
      << static_cast<int>(cute::size<2>(typename Config::ClusterShape{})) << "],\n"
      << "    \"scale_vector_size\": ";
  if constexpr (HasBlockScaleDetails<Mainloop>::value) {
    out << Mainloop::SFVecSize;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"scale_vector_size_a\": ";
  if constexpr (HasBlockScaleDetails<Mainloop>::value) {
    out << Mainloop::SFVecSize;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"scale_vector_size_b\": ";
  if constexpr (HasBlockScaleDetails<Mainloop>::value) {
    out << Mainloop::SFVecSize;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"element_a_sparsity\": ";
  if constexpr (HasSparseMetadataDetails<Mainloop>::value) {
    out << Mainloop::ElementASparsity;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"alignment_a\": " << Config::AlignmentA << ",\n"
      << "    \"alignment_b\": " << Config::AlignmentB << ",\n"
      << "    \"alignment_d\": " << Config::AlignmentD << ",\n"
      << "    \"mainloop_shared_storage_bytes\": " << sizeof(typename Mainloop::SharedStorage) << ",\n"
      << "    \"epilogue_shared_storage_bytes\": " << sizeof(typename Epilogue::SharedStorage) << ",\n"
      << "    \"kernel_shared_storage_bytes\": " << sizeof(typename Kernel::SharedStorage) << "\n"
      << "  }\n"
      << "}\n";
}

}  // namespace guide
