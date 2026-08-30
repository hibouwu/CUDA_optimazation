#!/usr/bin/env python3
"""Generate canonical compile-only instances for the official-C++ Phase 2 campaign."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from codegen_v2.model import load_strict_json  # noqa: E402
TARGET = {
    "cutlass_git_sha": "e05f953a5b3d38adc240df2ff928e0421c2abba3",
    "cutlass_arch_tag": "cutlass::arch::Sm100",
    "virtual_arch": "compute_110a",
    "binary_arch": "sm_110a",
}
LAYOUT = {
    "RowMajor": "cutlass::layout::RowMajor",
    "ColumnMajor": "cutlass::layout::ColumnMajor",
}


def layout_cpp(value: str) -> str:
    pointer = value.endswith("*")
    base = value[:-1] if pointer else value
    if base not in LAYOUT:
        raise ValueError(f"unsupported logical layout label: {value}")
    return LAYOUT[base] + ("*" if pointer else "")


def base_mechanism() -> dict:
    return {
        "pointer_mode": "single",
        "transforms": {"a": "identity", "b": "identity"},
        "block_scaled": {
            "enabled": False,
            "scale_a": None,
            "scale_b": None,
            "vector_size_a": None,
            "vector_size_b": None,
        },
        "blockwise": {
            "enabled": False,
            "granularity_m": None,
            "granularity_n": None,
            "granularity_k": None,
            "major_a": None,
            "major_b": None,
            "element_sfa": None,
            "element_sfb": None,
        },
        "sparse": {
            "enabled": False,
            "metadata_element": None,
            "a_sparsity": None,
            "e_sparsity": None,
        },
        "mixed_input": {
            "enabled": False,
            "mode": None,
            "operands_swapped": None,
            "transformed_operand": None,
            "tuple_arity": None,
            "narrow_type": None,
            "wide_type": None,
            "scale_type": None,
            "zero_type": None,
        },
        "fast_fp32": {
            "enabled": False,
            "atom_model": None,
            "num_compute_matrices": None,
            "num_bands": None,
            "scaling_factor": None,
            "acc_promotion_interval": None,
        },
        "complex": {"enabled": False, "representation": None},
    }


def spec(**values) -> dict:
    defaults = {
        "mainloop_op": "cutlass::arch::OpClassTensorOp",
        "epilogue_op": "cutlass::arch::OpClassTensorOp",
        "compute": "float",
        "stage_epi_bound": False,
        "problem": "cute::Shape<int,int,int,int>",
        "problem_mode": "dense",
        "scheduler": "void",
        "epilogue_tile": "cutlass::epilogue::collective::EpilogueTileAuto",
        "fusion": "void",
        "prelude": "",
        "builder_a": None,
        "builder_b": None,
        "builder_layout_a": None,
        "builder_layout_b": None,
        "epilogue_c": None,
        "epilogue_d": None,
        "epilogue_layout_c": None,
        "epilogue_layout_d": None,
        "cluster_type": None,
        "cluster_dynamic": False,
        "load_mode": "tma",
        "sparse_mma": False,
        "block_scale_mma": False,
        "tmem_copy": False,
        "extra_includes": "",
        "source_note": "",
        "generation_risk": "",
        "expected_outcome": "STATIC_PASS",
        "failure_domain": None,
        "failure_layer": None,
        "control_instance_id": None,
    }
    defaults.update(values)
    defaults["builder_a"] = defaults["builder_a"] or defaults["a"]
    defaults["builder_b"] = defaults["builder_b"] or defaults["b"]
    defaults["builder_layout_a"] = defaults["builder_layout_a"] or layout_cpp(defaults["layout_a"])
    defaults["builder_layout_b"] = defaults["builder_layout_b"] or layout_cpp(defaults["layout_b"])
    defaults["epilogue_c"] = defaults["epilogue_c"] or defaults["c"]
    defaults["epilogue_d"] = defaults["epilogue_d"] or defaults["d"]
    defaults["epilogue_layout_c"] = defaults["epilogue_layout_c"] or layout_cpp(defaults["layout_c"])
    defaults["epilogue_layout_d"] = defaults["epilogue_layout_d"] or layout_cpp(defaults["layout_d"])
    defaults["cluster_type"] = defaults["cluster_type"] or cpp_shape(defaults["cluster"])
    return defaults


def cpp_shape(values: list[int]) -> str:
    return "cute::Shape<" + ",".join(f"cute::_{value}" for value in values) + ">"


SPEC_PATHS = (
    ROOT / "tests/codegen/phase2_nonblock_specs.json",
    ROOT / "tests/codegen/phase2_blockscale_sparse_specs.json",
)
REQUIRED_SPEC_KEYS = {
    "instance_id",
    "tag",
    "a",
    "b",
    "c",
    "d",
    "accumulator",
    "layout_a",
    "layout_b",
    "layout_c",
    "layout_d",
    "align",
    "tile",
    "cluster",
    "cta",
    "epilogue",
    "mma_kind",
    "mechanism_patch",
    "source_note",
    "generation_risk",
}


def load_phase2_specs() -> list[dict]:
    documents = [load_strict_json(path) for path in SPEC_PATHS]
    for path, document in zip(SPEC_PATHS, documents, strict=True):
        if (
            document.get("schema_version") != 1
            or document.get("scope") != "PHASE2_OFFICIAL_CPP_CANONICAL_INSTANCES"
            or document.get("cutlass_git_sha") != TARGET["cutlass_git_sha"]
            or document.get("expected_count") != len(document.get("specs", []))
        ):
            raise ValueError(f"invalid Phase 2 spec document metadata: {path}")
    raw_specs = [entry for document in documents for entry in document["specs"]]
    for index, entry in enumerate(raw_specs):
        missing = REQUIRED_SPEC_KEYS - set(entry)
        if missing:
            raise ValueError(f"Phase 2 spec {index} lacks required fields: {sorted(missing)}")
        for field in ("align", "tile", "cluster"):
            values = entry[field]
            if len(values) != (4 if field == "align" else 3) or any(
                not isinstance(value, int) or value <= 0 for value in values
            ):
                raise ValueError(f"Phase 2 spec {index} has invalid {field}")
        if entry["cta"] not in {1, 2}:
            raise ValueError(f"Phase 2 spec {index} has invalid CTA group")
    items = [spec(**entry) for entry in raw_specs]
    tags = [item["tag"] for item in items]
    instance_ids = [item["instance_id"] for item in items]
    if len(items) != 34 or len(set(tags)) != 34 or len(set(instance_ids)) != 34:
        raise ValueError("Phase 2 specs must contain 34 unique Tags and instance IDs")

    inventory = load_strict_json(
        ROOT / "tests/codegen/sm110a_schedule_reference_inventory.json"
    )
    official_tags = {
        entry["tag"]
        for entry in inventory["entries"]
        if entry["reference_class"] == "official_cpp_explicit_or_conditional"
    }
    contract = load_strict_json(ROOT / "tests/codegen/static_codegen_contract.json")
    phase1_tags = {
        load_strict_json(
            ROOT / "tests/codegen/instances" / instance_id / "instance.json"
        )["subject"]["id"]
        for instance_id in contract["phase1_fresh_replay_instances"]
    }
    expected_tags = official_tags - phase1_tags
    if set(tags) != expected_tags:
        raise ValueError(
            f"Phase 2 Tag set mismatch: missing={sorted(expected_tags - set(tags))} "
            f"extra={sorted(set(tags) - expected_tags)}"
        )
    return items


def merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def render_config(item: dict) -> str:
    fusion_argument = "" if item["fusion"] == "void" else ",\n      FusionOperation"
    planar_epilogue_include = ""
    if item.get("mechanism_patch", {}).get("complex", {}).get("representation") == "planar":
        planar_header = (
            "sm100_epilogue_array_planar_complex_tma_warpspecialized.hpp"
            if "PtrArray" in item["tag"]
            else "sm100_epilogue_planar_complex_tma_warpspecialized.hpp"
        )
        planar_epilogue_include = (
            f"#include <cutlass/epilogue/collective/{planar_header}>\n"
        )
    stage = (
        "cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>"
        if item["stage_epi_bound"]
        else "cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>"
    )
    return f'''// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/mma_sm100.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
{planar_epilogue_include}#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
{item["extra_includes"]}

namespace guide::codegen::{item["instance_id"]} {{

{item["prelude"]}
struct Config {{
  using ElementA = {item["a"]};
  using ElementB = {item["b"]};
  using ElementC = {item["c"]};
  using ElementD = {item["d"]};
  using ElementAccumulator = {item["accumulator"]};
  using ElementCompute = {item["compute"]};
  using LayoutA = {layout_cpp(item["layout_a"])};
  using LayoutB = {layout_cpp(item["layout_b"])};
  using LayoutC = {layout_cpp(item["layout_c"])};
  using LayoutD = {layout_cpp(item["layout_d"])};
  static constexpr int AlignmentA = {item["align"][0]};
  static constexpr int AlignmentB = {item["align"][1]};
  static constexpr int AlignmentC = {item["align"][2]};
  static constexpr int AlignmentD = {item["align"][3]};
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = {item["mainloop_op"]};
  using MainloopOperatorClass = {item["mainloop_op"]};
  using EpilogueOperatorClass = {item["epilogue_op"]};
  using BuilderElementA = {item["builder_a"]};
  using BuilderElementB = {item["builder_b"]};
  using BuilderLayoutA = {item["builder_layout_a"]};
  using BuilderLayoutB = {item["builder_layout_b"]};
  using EpilogueElementC = {item["epilogue_c"]};
  using EpilogueElementD = {item["epilogue_d"]};
  using EpilogueLayoutC = {item["epilogue_layout_c"]};
  using EpilogueLayoutD = {item["epilogue_layout_d"]};
  using MmaTileShape = {cpp_shape(item["tile"])};
  using ClusterShape = {item["cluster_type"]};
  using ClusterDefaultShape = {cpp_shape(item["cluster"])};
  using MainloopSchedule = cutlass::gemm::{item["tag"]};
  using EpilogueSchedule = {item["epilogue"]};
  using EpilogueTile = {item["epilogue_tile"]};
  using FusionOperation = {item["fusion"]};
  using ProblemShape = {item["problem"]};
  using TileScheduler = {item["scheduler"]};

  using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, MmaTileShape, ClusterShape,
      EpilogueTile, ElementAccumulator, ElementCompute,
      EpilogueElementC, EpilogueLayoutC, AlignmentC,
      EpilogueElementD, EpilogueLayoutD, AlignmentD,
      EpilogueSchedule{fusion_argument}>;
  using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
  using StagePolicy = {stage};
  using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass,
      BuilderElementA, BuilderLayoutA, AlignmentA,
      BuilderElementB, BuilderLayoutB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      StagePolicy, MainloopSchedule>;
  using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}};

}}  // namespace guide::codegen::{item["instance_id"]}
'''


def opcode_contract(item: dict) -> tuple[dict, dict]:
    ptx_required = []
    sass_required = []
    if item["load_mode"] in {"tma", "mixed"}:
        ptx_required.append({"id": "tma_load", "regex": r"cp\.async\.bulk\.tensor(?:\..*)?", "min_count": 1, "max_count": None})
        sass_required.append({"id": "tma_load", "regex": r"UTMALDG(?:\..*)?", "min_count": 1, "max_count": None})
    if item["load_mode"] in {"cpasync", "mixed"}:
        ptx_required.append({"id": "cpasync_load", "regex": r"cp\.async\.(?:ca|cg)\.shared\.global(?:\..*)?", "min_count": 1, "max_count": None})
        sass_required.append({"id": "cpasync_load", "regex": r"LDGSTS(?:\..*)?", "min_count": 1, "max_count": None})
    ptx_required.append({"id": "tmem_load", "regex": r"tcgen05\.ld(?:\..*)?", "min_count": 1, "max_count": None})
    sass_required.append({"id": "tmem_load", "regex": r"LDTM(?:\..*)?", "min_count": 1, "max_count": None})
    if item["tmem_copy"]:
        ptx_required.append({"id": "tmem_copy", "regex": r"tcgen05\.cp(?:\..*)?", "min_count": 1, "max_count": None})
        sass_required.append({"id": "tmem_copy", "regex": r"UTCCP(?:\..*)?", "min_count": 1, "max_count": None})
    sparse = r"\.sp" if item["sparse_mma"] else ""
    suffix = r"(?:\..*)?block_scale(?:\..*)?" if item["block_scale_mma"] else r"(?:\..*)?"
    ptx_required.append({
        "id": "mma",
        "regex": rf"tcgen05\.mma{sparse}\.cta_group::{item['cta']}(?:\..*)?kind::{item['mma_kind']}{suffix}",
        "min_count": 1,
        "max_count": None,
    })
    sass_required.append({"id": "mma", "regex": r"UTC[A-Z0-9_]*MMA(?:\..*)?", "min_count": 1, "max_count": None})
    opposite = 2 if item["cta"] == 1 else 1
    ptx_forbidden = [{"id": "opposite_cta", "regex": rf"tcgen05\.mma(?:\.sp)?\.cta_group::{opposite}(?:\..*)?", "min_count": 0, "max_count": 0}]
    sass_forbidden = []
    if item["cta"] == 1:
        sass_forbidden.append({"id": "mma_2cta", "regex": r"UTC[A-Z0-9_]*MMA\.2CTA(?:\..*)?", "min_count": 0, "max_count": 0})
    return {"required": ptx_required, "forbidden": ptx_forbidden}, {"required": sass_required, "forbidden": sass_forbidden}


def render_instance(item: dict, config: Path, kernel: Path, witness: Path) -> dict:
    inventory_path = ROOT / "tests/codegen/sm110a_schedule_reference_inventory.json"
    inventory = load_strict_json(inventory_path)
    reference = next(entry for entry in inventory["entries"] if entry["tag"] == item["tag"])
    tags = load_strict_json(ROOT / "tests/codegen/sm110a_tensor_schedule_tags.json")
    group = next(entry["group"] for entry in tags["entries"] if entry["tag"] == item["tag"])
    mechanism = merge(base_mechanism(), item.get("mechanism_patch", {}))
    if "PtrArray" in item["tag"]:
        mechanism["pointer_mode"] = "array"
    ptx, sass = opcode_contract(item)
    return {
        "schema_version": 1,
        "contract_sha256": hashlib.sha256((ROOT / "tests/codegen/static_codegen_contract.json").read_bytes()).hexdigest(),
        "objective_sha256": "463fa7015808acd883b28d115fa33708f66064aaceed96f728996a01ca0e4091",
        "freshness_epoch": "sm110a-codegen-v2-20260830",
        "instance_id": item["instance_id"],
        "subject": {"kind": "explicit_schedule_tag", "id": item["tag"], "group": group},
        "provenance": {
            "reference_inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "reference_class": reference["reference_class"],
            "source_anchors": [{"path": reference["path"], "line_start": reference["line"], "line_end": reference["line"], "sha256": reference["anchor_line_sha256"]}],
            "parent_instance_id": None,
            "derivation_axis": reference["derivation_axis"],
        },
        "target": TARGET,
        "translation_units": {
            "config": {"path": config.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
            "kernel": {"path": kernel.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(kernel.read_bytes()).hexdigest()},
            "type_witness": {"path": witness.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(witness.read_bytes()).hexdigest()},
        },
        "declared_config": {
            "family": group,
            "operator_class": item["mainloop_op"],
            "elements": {"a": item["a"], "b": item["b"], "c": item["c"], "d": item["d"], "accumulator": item["accumulator"], "compute": item["compute"]},
            "layouts": {"a": item["layout_a"], "b": item["layout_b"], "c": item["layout_c"], "d": item["layout_d"]},
            "alignments": {"a": item["align"][0], "b": item["align"][1], "c": item["align"][2], "d": item["align"][3]},
            "builder_contract": {
                "mainloop_operator_class": item["mainloop_op"], "epilogue_operator_class": item["epilogue_op"],
                "element_a": item["builder_a"], "element_b": item["builder_b"],
                "layout_a": item["builder_layout_a"], "layout_b": item["builder_layout_b"],
                "epilogue_element_c": item["epilogue_c"], "epilogue_element_d": item["epilogue_d"],
                "epilogue_layout_c": item["epilogue_layout_c"], "epilogue_layout_d": item["epilogue_layout_d"],
                "epilogue_tile": item["epilogue_tile"], "fusion_operation": item["fusion"],
                "problem_mode": item["problem_mode"], "cluster_type_cpp": item["cluster_type"],
                "cluster_is_dynamic": item["cluster_dynamic"], "cluster_default_mnk": item["cluster"],
            },
            "mma_tile_mnk": item["tile"], "cluster_mnk": item["cluster"],
            "stage_policy": "cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>" if item["stage_epi_bound"] else "cutlass::gemm::collective::StageCountAutoCarveout<EpilogueSharedStorage>",
            "mainloop_schedule": f"cutlass::gemm::{item['tag']}", "epilogue_schedule": item["epilogue"],
            "kernel_problem_shape": item["problem"], "tile_scheduler": item["scheduler"],
            "cta_group": item["cta"], "mechanism": mechanism,
        },
        "static_contract": {
            "required_layers": ["DECLARED_BUILDER_CONFIG", "BUILDER_SPECIALIZATION", "DISPATCH_POLICY", "COLLECTIVE", "STAGE", "COPY_LAYOUT", "TILED_MMA", "ATOM", "KERNEL_COMPOSITION", "PTX_EMISSION", "SM110A_ASSEMBLY", "FUNCTION_BINDING", "FUNCTION_CONTRACT"],
            "target_entity": "cutlass::device_kernel<GemmKernel>",
            "function_selector": "sole_ptx_entry_equals_sole_elf_sto_entry_equals_unique_nvdisasm_function",
            "ptx": ptx,
            "sass": sass,
        },
        "hypothesis": {
            "expected_outcome": item["expected_outcome"],
            "failure_domain": item["failure_domain"],
            "failure_layer": item["failure_layer"],
            "diagnostic_patterns": [],
            "control_instance_id": item["control_instance_id"],
        },
    }


def render_kernel(item: dict) -> str:
    return f'''// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include <cutlass/device_kernel.h>
using GemmKernel = guide::codegen::{item["instance_id"]}::Config::GemmKernel;
template __global__ void cutlass::device_kernel<GemmKernel>(
    CUTLASS_GRID_CONSTANT GemmKernel::Params const params);
'''


def render_type_witness(item: dict) -> str:
    return f'''// SPDX-License-Identifier: BSD-3-Clause
#include "config.hpp"
#include "guide/codegen_type_report.hpp"
#include <iostream>
int main() {{
  guide::write_codegen_type_report<guide::codegen::{item["instance_id"]}::Config>(
      std::cout, "{item["instance_id"]}");
  return 0;
}}
'''


def write_instance(item: dict) -> None:
    directory = ROOT / "tests/codegen/instances" / item["instance_id"]
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "config.hpp"
    kernel = directory / "kernel.cu"
    witness = directory / "type_witness.cu"
    config.write_text(render_config(item), encoding="utf-8")
    kernel.write_text(render_kernel(item), encoding="utf-8")
    witness.write_text(render_type_witness(item), encoding="utf-8")
    (directory / "instance.json").write_text(
        json.dumps(render_instance(item, config, kernel, witness), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    for item in load_phase2_specs():
        write_instance(item)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
