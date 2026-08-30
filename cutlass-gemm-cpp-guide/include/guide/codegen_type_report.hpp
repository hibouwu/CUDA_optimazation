// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/container/tuple.hpp>

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

template <class Type, class = void>
struct HasClassicGmemCopy : std::false_type {};
template <class Type>
struct HasClassicGmemCopy<Type, std::void_t<
    typename Type::GmemTiledCopyA, typename Type::GmemTiledCopyB>> : std::true_type {};

template <class Type, class = void>
struct HasClassicSmemLayout : std::false_type {};
template <class Type>
struct HasClassicSmemLayout<Type, std::void_t<
    typename Type::SmemLayoutAtomA, typename Type::SmemLayoutAtomB>> : std::true_type {};

template <class Type, class = void>
struct HasClassicSmemCopy : std::false_type {};
template <class Type>
struct HasClassicSmemCopy<Type, std::void_t<
    typename Type::SmemCopyAtomA, typename Type::SmemCopyAtomB>> : std::true_type {};

template <class Type, class = void>
struct HasInputComputeCopy : std::false_type {};
template <class Type>
struct HasInputComputeCopy<Type, std::void_t<
    typename Type::SmemLayoutAtomsA, typename Type::SmemLayoutAtomsB,
    typename Type::InputCopyAtomA, typename Type::InputCopyAtomB,
    typename Type::ComputeCopyAtomA, typename Type::ComputeCopyAtomB>> : std::true_type {};

template <class Type, class = void>
struct HasPlanarMmaPair : std::false_type {};
template <class Type>
struct HasPlanarMmaPair<Type, std::void_t<
    typename Type::TiledMmaPair, typename Type::TiledMmaANeg>> : std::true_type {};

template <class Type, class = void>
struct HasScaleFactorMma : std::false_type {};
template <class Type>
struct HasScaleFactorMma<Type, std::void_t<typename Type::TiledMma_SF>> : std::true_type {};

template <class Type, class = void>
struct HasBlockwiseDetails : std::false_type {};
template <class Type>
struct HasBlockwiseDetails<Type, std::void_t<
    typename Type::ScaleConfig, typename Type::LayoutSFA, typename Type::LayoutSFB,
    decltype(Type::ScaleGranularityM), decltype(Type::ScaleGranularityN),
    decltype(Type::ScaleGranularityK)>> : std::true_type {};

template <class Type, class = void>
struct HasBlockwiseScaleElementAliases : std::false_type {};
template <class Type>
struct HasBlockwiseScaleElementAliases<Type, std::void_t<
    typename Type::ElementSFA, typename Type::ElementSFB>> : std::true_type {};

template <class Type, class = void>
struct HasMixedInputDetails : std::false_type {};
template <class Type>
struct HasMixedInputDetails<Type, std::void_t<
    typename Type::ElementScale, typename Type::ElementZero,
    typename Type::LayoutScale, typename Type::GmemTiledCopyScale,
    typename Type::SmemLayoutAtomScale>> : std::true_type {};

template <class Type, class = void>
struct HasFastAlgorithmDetails : std::false_type {};
template <class Type>
struct HasFastAlgorithmDetails<Type, std::void_t<
    decltype(Type::NumComputeMtxs), decltype(Type::NumBandsToCompute),
    decltype(Type::ScalingFactor), decltype(Type::AccPromotionInterval)>> : std::true_type {};

template <class Type, class = void>
struct HasStages : std::false_type {};
template <class Type>
struct HasStages<Type, std::void_t<decltype(Type::Stages)>> : std::true_type {};

template <class Type, class = void>
struct HasLoadTransformStages : std::false_type {};
template <class Type>
struct HasLoadTransformStages<Type, std::void_t<
    decltype(Type::Load2TransformPipelineStageCount),
    decltype(Type::Transform2MmaPipelineStageCount)>> : std::true_type {};

template <class Type, class = void>
struct HasComplexStages : std::false_type {};
template <class Type>
struct HasComplexStages<Type, std::void_t<
    decltype(Type::ComputationPipelineStageCount),
    decltype(Type::TransformationPipelineStageCount)>> : std::true_type {};

template <class Type, class = void>
struct HasElementASparsity : std::false_type {};
template <class Type>
struct HasElementASparsity<Type, std::void_t<decltype(Type::ElementASparsity)>> : std::true_type {};

template <class Type, class = void>
struct HasElementAMmaSparsity : std::false_type {};
template <class Type>
struct HasElementAMmaSparsity<Type, std::void_t<decltype(Type::ElementAMmaSparsity)>>
    : std::true_type {};

template <class Type, class = void>
struct HasElementEMmaSparsity : std::false_type {};
template <class Type>
struct HasElementEMmaSparsity<Type, std::void_t<decltype(Type::ElementEMmaSparsity)>>
    : std::true_type {};

template <class Dispatch>
constexpr int mainloop_stage_count() {
  if constexpr (HasStages<Dispatch>::value) {
    return Dispatch::Stages;
  } else if constexpr (HasLoadTransformStages<Dispatch>::value) {
    return Dispatch::Load2TransformPipelineStageCount;
  } else if constexpr (HasComplexStages<Dispatch>::value) {
    return Dispatch::ComputationPipelineStageCount;
  } else {
    return 0;
  }
}

template <class Mainloop>
constexpr int element_a_sparsity() {
  if constexpr (HasElementASparsity<Mainloop>::value) {
    return Mainloop::ElementASparsity;
  } else if constexpr (HasElementAMmaSparsity<Mainloop>::value) {
    return Mainloop::ElementAMmaSparsity;
  } else {
    return 0;
  }
}

template <class Mainloop>
constexpr int element_e_sparsity() {
  if constexpr (HasElementEMmaSparsity<Mainloop>::value) {
    return Mainloop::ElementEMmaSparsity;
  } else {
    return 0;
  }
}

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

template <class Fragment>
std::string codegen_mma_operand_source() {
  std::string const type_name = codegen_type_name<Fragment>();
  if (type_name.find("sparse_smem_desc") != std::string::npos) {
    return "SPARSE_SMEM_DESCRIPTOR";
  }
  if (type_name.find("smem_desc") != std::string::npos) {
    return "SMEM_DESCRIPTOR";
  }
  if (type_name.find("tmem_frg") != std::string::npos) {
    return "TMEM_FRAGMENT";
  }
  return "UNKNOWN";
}

template <class Type>
constexpr int codegen_tuple_arity() {
  if constexpr (cute::is_tuple<Type>::value) {
    return static_cast<int>(cute::tuple_size<Type>::value);
  } else {
    return 0;
  }
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
  field("mma_fragment_type_a", codegen_type_name<typename TiledMma::FrgTypeA>());
  field("mma_fragment_type_b", codegen_type_name<typename TiledMma::FrgTypeB>());
  field("mma_operand_source_a", codegen_mma_operand_source<typename TiledMma::FrgTypeA>());
  field("mma_operand_source_b", codegen_mma_operand_source<typename TiledMma::FrgTypeB>());
  if constexpr (HasClassicGmemCopy<Mainloop>::value) {
    field("gmem_tiled_copy_a", codegen_type_name<typename Mainloop::GmemTiledCopyA>());
    field("gmem_tiled_copy_b", codegen_type_name<typename Mainloop::GmemTiledCopyB>());
  } else {
    field("gmem_tiled_copy_a", "not_applicable");
    field("gmem_tiled_copy_b", "not_applicable");
  }
  if constexpr (HasClassicSmemLayout<Mainloop>::value) {
    field("smem_layout_atom_a", codegen_type_name<typename Mainloop::SmemLayoutAtomA>());
    field("smem_layout_atom_b", codegen_type_name<typename Mainloop::SmemLayoutAtomB>());
  } else {
    field("smem_layout_atom_a", "not_applicable");
    field("smem_layout_atom_b", "not_applicable");
  }
  if constexpr (HasClassicSmemCopy<Mainloop>::value) {
    field("smem_copy_atom_a", codegen_type_name<typename Mainloop::SmemCopyAtomA>());
    field("smem_copy_atom_b", codegen_type_name<typename Mainloop::SmemCopyAtomB>());
  } else {
    field("smem_copy_atom_a", "not_applicable");
    field("smem_copy_atom_b", "not_applicable");
  }
  field("collective_epilogue", codegen_type_name<Epilogue>());
  field("kernel_collective_mainloop", codegen_type_name<typename Kernel::CollectiveMainloop>());
  field("kernel_collective_epilogue", codegen_type_name<typename Kernel::CollectiveEpilogue>());
  field("config_arch_tag", codegen_type_name<typename Config::ArchTag>());
  field("config_operator_class", codegen_type_name<typename Config::OperatorClass>());
  field("mainloop_operator_class", codegen_type_name<typename Config::MainloopOperatorClass>());
  field("epilogue_operator_class", codegen_type_name<typename Config::EpilogueOperatorClass>());
  field("config_element_a", codegen_type_name<typename Config::ElementA>());
  field("config_element_b", codegen_type_name<typename Config::ElementB>());
  field("config_element_c", codegen_type_name<typename Config::ElementC>());
  field("config_element_compute", codegen_type_name<typename Config::ElementCompute>());
  field("config_element_accumulator", codegen_type_name<typename Config::ElementAccumulator>());
  field("config_element_d", codegen_type_name<typename Config::ElementD>());
  field("config_layout_a", codegen_type_name<typename Config::LayoutA>());
  field("config_layout_b", codegen_type_name<typename Config::LayoutB>());
  field("config_layout_c", codegen_type_name<typename Config::LayoutC>());
  field("config_layout_d", codegen_type_name<typename Config::LayoutD>());
  field("builder_element_a", codegen_type_name<typename Config::BuilderElementA>());
  field("builder_element_b", codegen_type_name<typename Config::BuilderElementB>());
  field("builder_layout_a", codegen_type_name<typename Config::BuilderLayoutA>());
  field("builder_layout_b", codegen_type_name<typename Config::BuilderLayoutB>());
  field("epilogue_element_c", codegen_type_name<typename Config::EpilogueElementC>());
  field("epilogue_element_d", codegen_type_name<typename Config::EpilogueElementD>());
  field("epilogue_layout_c", codegen_type_name<typename Config::EpilogueLayoutC>());
  field("epilogue_layout_d", codegen_type_name<typename Config::EpilogueLayoutD>());
  field("epilogue_tile", codegen_type_name<typename Config::EpilogueTile>());
  field("fusion_operation", codegen_type_name<typename Config::FusionOperation>());
  field("config_mainloop_schedule", codegen_type_name<typename Config::MainloopSchedule>());
  field("config_epilogue_schedule", codegen_type_name<typename Config::EpilogueSchedule>());
  field("config_stage_policy", codegen_type_name<typename Config::StagePolicy>());
  field("config_problem_shape", codegen_type_name<typename Config::ProblemShape>());
  field("config_cluster_shape", codegen_type_name<typename Config::ClusterShape>());
  field("config_cluster_default_shape", codegen_type_name<typename Config::ClusterDefaultShape>());
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
  if constexpr (HasInputComputeCopy<Mainloop>::value) {
    optional_field("smem_layout_atoms_a", codegen_type_name<typename Mainloop::SmemLayoutAtomsA>());
    optional_field("smem_layout_atoms_b", codegen_type_name<typename Mainloop::SmemLayoutAtomsB>());
    optional_field("input_copy_atom_a", codegen_type_name<typename Mainloop::InputCopyAtomA>());
    optional_field("input_copy_atom_b", codegen_type_name<typename Mainloop::InputCopyAtomB>());
    optional_field("compute_copy_atom_a", codegen_type_name<typename Mainloop::ComputeCopyAtomA>());
    optional_field("compute_copy_atom_b", codegen_type_name<typename Mainloop::ComputeCopyAtomB>());
  }
  if constexpr (HasPlanarMmaPair<Mainloop>::value) {
    optional_field("planar_tiled_mma_pair", codegen_type_name<typename Mainloop::TiledMmaPair>());
    optional_field("planar_tiled_mma_a_negative", codegen_type_name<typename Mainloop::TiledMmaANeg>());
  }
  if constexpr (HasScaleFactorMma<Mainloop>::value) {
    optional_field("scale_factor_tiled_mma", codegen_type_name<typename Mainloop::TiledMma_SF>());
    optional_field("scale_factor_mma_atom", codegen_type_name<typename Mainloop::TiledMma_SF::Atom>());
  }
  if constexpr (HasBlockwiseDetails<Mainloop>::value) {
    using InternalLayoutSFA = std::remove_pointer_t<typename Mainloop::LayoutSFA>;
    using InternalLayoutSFB = std::remove_pointer_t<typename Mainloop::LayoutSFB>;
    optional_field("blockwise_scale_config", codegen_type_name<typename Mainloop::ScaleConfig>());
    if constexpr (HasBlockwiseScaleElementAliases<Mainloop>::value) {
      optional_field("blockwise_element_sfa", codegen_type_name<typename Mainloop::ElementSFA>());
      optional_field("blockwise_element_sfb", codegen_type_name<typename Mainloop::ElementSFB>());
    } else {
      // The array/grouped specialization sources software scale values as ElementAccumulator.
      optional_field("blockwise_element_sfa", codegen_type_name<typename Mainloop::ElementAccumulator>());
      optional_field("blockwise_element_sfb", codegen_type_name<typename Mainloop::ElementAccumulator>());
    }
    optional_field("blockwise_layout_sfa", codegen_type_name<typename Mainloop::LayoutSFA>());
    optional_field("blockwise_layout_sfb", codegen_type_name<typename Mainloop::LayoutSFB>());
    optional_field("blockwise_major_a",
                   cute::size<0, 1>(InternalLayoutSFA{}.stride()) == 1 ? "MN" : "K");
    optional_field("blockwise_major_b",
                   cute::size<0, 1>(InternalLayoutSFB{}.stride()) == 1 ? "MN" : "K");
  }
  if constexpr (HasMixedInputDetails<Mainloop>::value) {
    optional_field("mixed_element_scale", codegen_type_name<typename Mainloop::ElementScale>());
    optional_field("mixed_element_zero", codegen_type_name<typename Mainloop::ElementZero>());
    optional_field("mixed_layout_scale", codegen_type_name<typename Mainloop::LayoutScale>());
    optional_field("mixed_gmem_tiled_copy_scale", codegen_type_name<typename Mainloop::GmemTiledCopyScale>());
    optional_field("mixed_smem_layout_atom_scale", codegen_type_name<typename Mainloop::SmemLayoutAtomScale>());
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
      << "    \"mainloop_stages\": " << mainloop_stage_count<Dispatch>() << ",\n"
      << "    \"scheduler_stages\": " << DispatchSchedule::SchedulerPipelineStageCount << ",\n"
      << "    \"accumulator_stages\": " << DispatchSchedule::AccumulatorPipelineStageCount << ",\n"
      << "    \"load_to_transform_stages\": ";
  if constexpr (HasLoadTransformStages<Dispatch>::value) {
    out << Dispatch::Load2TransformPipelineStageCount;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"transform_to_mma_stages\": ";
  if constexpr (HasLoadTransformStages<Dispatch>::value) {
    out << Dispatch::Transform2MmaPipelineStageCount;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"computation_stages\": ";
  if constexpr (HasComplexStages<Dispatch>::value) {
    out << Dispatch::ComputationPipelineStageCount;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"transformation_stages\": ";
  if constexpr (HasComplexStages<Dispatch>::value) {
    out << Dispatch::TransformationPipelineStageCount;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"atom_shape_mnk\": ["
      << static_cast<int>(cute::size<0>(typename TiledMma::AtomShape_MNK{})) << ','
      << static_cast<int>(cute::size<1>(typename TiledMma::AtomShape_MNK{})) << ','
      << static_cast<int>(cute::size<2>(typename TiledMma::AtomShape_MNK{})) << "],\n"
      << "    \"mma_tile_mnk\": ["
      << static_cast<int>(cute::size<0>(typename Config::MmaTileShape{})) << ','
      << static_cast<int>(cute::size<1>(typename Config::MmaTileShape{})) << ','
      << static_cast<int>(cute::size<2>(typename Config::MmaTileShape{})) << "],\n"
      << "    \"cluster_mnk\": ["
      << static_cast<int>(cute::size<0>(typename Config::ClusterDefaultShape{})) << ','
      << static_cast<int>(cute::size<1>(typename Config::ClusterDefaultShape{})) << ','
      << static_cast<int>(cute::size<2>(typename Config::ClusterDefaultShape{})) << "],\n"
      << "    \"builder_tuple_arity_a\": "
      << codegen_tuple_arity<typename Config::BuilderElementA>() << ",\n"
      << "    \"builder_tuple_arity_b\": "
      << codegen_tuple_arity<typename Config::BuilderElementB>() << ",\n"
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
    out << element_a_sparsity<Mainloop>();
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"element_e_sparsity\": " << element_e_sparsity<Mainloop>() << ",\n"
      << "    \"blockwise_granularity_m\": ";
  if constexpr (HasBlockwiseDetails<Mainloop>::value) {
    out << Mainloop::ScaleGranularityM;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"blockwise_granularity_n\": ";
  if constexpr (HasBlockwiseDetails<Mainloop>::value) {
    out << Mainloop::ScaleGranularityN;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"blockwise_granularity_k\": ";
  if constexpr (HasBlockwiseDetails<Mainloop>::value) {
    out << Mainloop::ScaleGranularityK;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"num_compute_matrices\": ";
  if constexpr (HasFastAlgorithmDetails<Mainloop>::value) {
    out << Mainloop::NumComputeMtxs;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"num_bands_to_compute\": ";
  if constexpr (HasFastAlgorithmDetails<Mainloop>::value) {
    out << Mainloop::NumBandsToCompute;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"fast_scaling_factor\": ";
  if constexpr (HasFastAlgorithmDetails<Mainloop>::value) {
    out << Mainloop::ScalingFactor;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"acc_promotion_interval\": ";
  if constexpr (HasFastAlgorithmDetails<Mainloop>::value) {
    out << Mainloop::AccPromotionInterval;
  } else {
    out << 0;
  }
  out << ",\n"
      << "    \"alignment_a\": " << Config::AlignmentA << ",\n"
      << "    \"alignment_b\": " << Config::AlignmentB << ",\n"
      << "    \"alignment_c\": " << Config::AlignmentC << ",\n"
      << "    \"alignment_d\": " << Config::AlignmentD << ",\n"
      << "    \"mainloop_shared_storage_bytes\": " << sizeof(typename Mainloop::SharedStorage) << ",\n"
      << "    \"epilogue_shared_storage_bytes\": " << sizeof(typename Epilogue::SharedStorage) << ",\n"
      << "    \"kernel_shared_storage_bytes\": " << sizeof(typename Kernel::SharedStorage) << "\n"
      << "  }\n"
      << "}\n";
}

}  // namespace guide
