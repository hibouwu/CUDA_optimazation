#!/usr/bin/env python3
"""Mechanical one-time migration of the six Phase 1 static instance records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpp_layout(value: str) -> str:
    if "RowMajor" in value:
        return "cutlass::layout::RowMajor"
    if "ColumnMajor" in value:
        return "cutlass::layout::ColumnMajor"
    raise ValueError(f"cannot normalize layout {value!r}")


def static_shape(values: list[int]) -> str:
    return "cute::Shape<" + ",".join(f"cute::_{value}" for value in values) + ">"


def main() -> int:
    contract_hash = sha256(ROOT / "tests/codegen/static_codegen_contract.json")
    for instance_path in sorted((ROOT / "tests/codegen/instances").glob("*/instance.json")):
        value = json.loads(instance_path.read_text(encoding="utf-8"))
        if value["instance_id"] not in {
            "dense_f16_1sm",
            "dense_f16_2sm",
            "dense_bs_nvfp4_1sm",
            "dense_bs_mxf4_1sm",
            "dense_bs_mxf8_1sm",
            "sparse_bs_nvfp4_1sm",
        }:
            continue
        declared = value["declared_config"]
        declared["elements"] = {
            "a": declared["elements"]["a"],
            "b": declared["elements"]["b"],
            "c": "void",
            "d": declared["elements"]["d"],
            "accumulator": declared["elements"]["accumulator"],
            "compute": "float",
        }
        declared["layouts"] = {
            "a": declared["layouts"]["a"],
            "b": declared["layouts"]["b"],
            "c": "RowMajor",
            "d": declared["layouts"]["d"],
        }
        alignment_c = 8 if value["instance_id"] == "sparse_bs_nvfp4_1sm" else 1
        declared["alignments"] = {
            "a": declared["alignments"]["a"],
            "b": declared["alignments"]["b"],
            "c": alignment_c,
            "d": declared["alignments"]["d"],
        }
        declared["builder_contract"] = {
            "mainloop_operator_class": declared["operator_class"],
            "epilogue_operator_class": declared["operator_class"],
            "element_a": declared["elements"]["a"],
            "element_b": declared["elements"]["b"],
            "layout_a": cpp_layout(declared["layouts"]["a"]),
            "layout_b": cpp_layout(declared["layouts"]["b"]),
            "epilogue_element_c": declared["elements"]["c"],
            "epilogue_element_d": declared["elements"]["d"],
            "epilogue_layout_c": cpp_layout(declared["layouts"]["c"]),
            "epilogue_layout_d": cpp_layout(declared["layouts"]["d"]),
            "epilogue_tile": "cutlass::epilogue::collective::EpilogueTileAuto",
            "fusion_operation": "void",
            "problem_mode": "dense",
            "cluster_type_cpp": static_shape(declared["cluster_mnk"]),
            "cluster_is_dynamic": False,
            "cluster_default_mnk": declared["cluster_mnk"],
        }
        mechanism = declared["mechanism"]
        mechanism["blockwise"] = {
            "enabled": False,
            "granularity_m": None,
            "granularity_n": None,
            "granularity_k": None,
            "major_a": None,
            "major_b": None,
            "element_sfa": None,
            "element_sfb": None,
        }
        mechanism["mixed_input"] = {
            "enabled": False,
            "mode": None,
            "operands_swapped": None,
            "transformed_operand": None,
            "tuple_arity": None,
            "narrow_type": None,
            "wide_type": None,
            "scale_type": None,
            "zero_type": None,
        }
        if value["instance_id"] == "sparse_bs_nvfp4_1sm":
            mechanism["sparse"] = {
                "enabled": True,
                "metadata_element": "uint8_t",
                "a_sparsity": 2,
                "e_sparsity": 16,
            }
        else:
            mechanism["sparse"] = {
                "enabled": False,
                "metadata_element": None,
                "a_sparsity": None,
                "e_sparsity": None,
            }
        fast = mechanism["fast_fp32"]
        fast.update(
            num_compute_matrices=None,
            num_bands=None,
            scaling_factor=None,
            acc_promotion_interval=None,
        )
        declared.pop("operand_source", None)
        value["contract_sha256"] = contract_hash
        for role, reference in value["translation_units"].items():
            reference["sha256"] = sha256(ROOT / reference["path"])
        instance_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
