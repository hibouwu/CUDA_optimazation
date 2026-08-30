#!/usr/bin/env python3
"""Adversarial unit tests for exact-symbol PTX/SASS attribution."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from codegen_v2.attribution import AttributionError, attribute_codegen  # noqa: E402
from codegen_v2.attribution import parse_ptx_entries  # noqa: E402


PTX_HEADER = ".version 9.0\n.target sm_110a\n.address_size 64\n"
CODE_OBJECT_METADATA = "arch = sm_110a\ncode for sm_110a\n.target sm_110a\n"


def ptx_function(symbol: str, *opcodes: str) -> str:
    body = "\n".join(f"  {opcode} %r1, %r2;" for opcode in opcodes)
    return f".visible .entry {symbol}()\n{{\n{body}\n}}\n"


def sass_function(symbol: str, *opcodes: str) -> dict:
    return {
        "function-name": symbol,
        "start": 0,
        "length": len(opcodes) * 16,
        "sass-instructions": [{"opcode": opcode, "operands": "R1,R2"} for opcode in opcodes],
    }


def sass_json(*functions: dict, arch: str = "sm_110a") -> str:
    metadata = {
        "SM": {"version": {"major": 11, "minor": 0}},
        ".note.nv.tkinfo": {"tki_toolOptions": f"-arch {arch} -m 64 "},
    }
    return json.dumps([metadata, list(functions)], separators=(",", ":"))


def elf_symbols(*symbols: str) -> str:
    return "symbols:\n" + "\n".join(
        f"STT_FUNC STB_LOCAL STO_ENTRY {symbol}" for symbol in symbols
    )


def inspect(
    ptx: str,
    sass: str,
    symbol: str = "kernel_A",
    *,
    entries: str | None = None,
    metadata: str = CODE_OBJECT_METADATA,
):
    return attribute_codegen(
        ptx_text=ptx,
        nvdisasm_json_text=sass,
        elf_symbols_text=entries if entries is not None else elf_symbols(symbol),
        code_object_metadata_text=metadata,
        expected_symbol=symbol,
        declared_cta_group=1,
        expected_ptx_target="sm_110a",
        expected_sass_arch="sm_110a",
        required_ptx=[
            r"cp\.async\.bulk\.tensor(?:\..*)?",
            r"tcgen05\.mma\.cta_group::1(?:\..*)?",
            r"tcgen05\.ld(?:\..*)?",
        ],
        forbidden_ptx=[
            r"tcgen05\.mma\.cta_group::2(?:\..*)?",
            r"tcgen05\.cp(?:\..*)?",
        ],
        required_sass=[r"UTMALDG(?:\..*)?", r"UTCHMMA(?:\..*)?", r"LDTM(?:\..*)?"],
        forbidden_sass=[
            r"UTMALDG(?:\..*)?\.2CTA",
            r"UTCHMMA\.2CTA(?:\..*)?",
            r"UTCCP(?:\..*)?",
        ],
    )


def inspect_group(
    ptx: str, sass: str, cta_group: int, *, require_tma_group_match: bool = True
):
    return attribute_codegen(
        ptx_text=ptx,
        nvdisasm_json_text=sass,
        elf_symbols_text=elf_symbols("kernel_A"),
        code_object_metadata_text=CODE_OBJECT_METADATA,
        expected_symbol="kernel_A",
        declared_cta_group=cta_group,
        require_tma_group_match=require_tma_group_match,
        expected_ptx_target="sm_110a",
        expected_sass_arch="sm_110a",
        required_ptx=[
            r"cp\.async\.bulk\.tensor(?:\..*)?",
            r"tcgen05\.mma(?:\..*)?",
            r"tcgen05\.ld(?:\..*)?",
        ],
        forbidden_ptx=[],
        required_sass=[r"UTMALDG(?:\..*)?", r"UTC[A-Z0-9_]*MMA(?:\..*)?", r"LDTM(?:\..*)?"],
        forbidden_sass=[],
    )


def require_error(name: str, expected: str, callback) -> None:
    try:
        callback()
    except AttributionError as error:
        if expected not in str(error):
            raise AssertionError(f"{name}: wrong error {error!s}") from error
    else:
        raise AssertionError(f"{name}: deliberate corruption was accepted")


def main() -> int:
    valid_ptx = PTX_HEADER + ptx_function(
        "kernel_A",
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes",
        "tcgen05.mma.cta_group::1.kind::f16",
        "tcgen05.ld.sync.aligned",
    )
    valid_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D", "UTCHMMA", "LDTM.16")
    )
    result = inspect(valid_ptx, valid_sass)
    if not result.passed:
        raise AssertionError(f"positive fixture failed: {result}")

    split_ptx = (
        PTX_HEADER
        + ptx_function("kernel_A", "cp.async.bulk.tensor.2d", "tcgen05.mma.cta_group::1.kind::f16")
        + ptx_function("kernel_B", "tcgen05.ld.sync.aligned")
    )
    if inspect(split_ptx, valid_sass).ptx_contract.passed:
        raise AssertionError("split required PTX instructions were merged across functions")

    duplicate_ptx = valid_ptx + ptx_function("kernel_A", "tcgen05.ld.sync.aligned")
    require_error("duplicate_ptx_symbol", "occur exactly once", lambda: inspect(duplicate_ptx, valid_sass))

    duplicate_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D", "UTCHMMA", "LDTM.16"),
        sass_function("kernel_A", "LDTM.16"),
    )
    require_error("duplicate_sass_symbol", "occur exactly once", lambda: inspect(valid_ptx, duplicate_sass))

    wrong_sass_symbol = sass_json(
        sass_function("kernel_B", "UTMALDG.3D", "UTCHMMA", "LDTM.16")
    )
    require_error("ptx_sass_symbol_split", "found 0", lambda: inspect(valid_ptx, wrong_sass_symbol))

    helper_only_ptx = (
        PTX_HEADER
        + ptx_function("kernel_A", "mov.u32")
        + ptx_function(
            "helper",
            "cp.async.bulk.tensor.2d",
            "tcgen05.mma.cta_group::1.kind::f16",
            "tcgen05.ld.sync.aligned",
        )
    )
    if inspect(helper_only_ptx, valid_sass).ptx_contract.passed:
        raise AssertionError("helper instructions were attributed to the declared target")

    comment_ptx = PTX_HEADER + ptx_function("kernel_A", "mov.u32").replace(
        "  mov.u32 %r1, %r2;",
        "  // cp.async.bulk.tensor tcgen05.mma.cta_group::1 tcgen05.ld\n  mov.u32 %r1, %r2;",
    )
    if inspect(comment_ptx, valid_sass).ptx_contract.passed:
        raise AssertionError("PTX comments were counted as instructions")

    forbidden_ptx = valid_ptx.replace(
        "tcgen05.ld.sync.aligned %r1, %r2;",
        "tcgen05.ld.sync.aligned %r1, %r2;\n  tcgen05.cp.cta_group::1 %r1, %r2;",
    )
    if inspect(forbidden_ptx, valid_sass).ptx_contract.passed:
        raise AssertionError("forbidden PTX opcode did not fail the contract")

    forbidden_sass = sass_json(
        sass_function(
            "kernel_A", "UTMALDG.3D", "UTCHMMA", "LDTM.16", "UTCCP.128"
        )
    )
    if inspect(valid_ptx, forbidden_sass).sass_contract.passed:
        raise AssertionError("forbidden SASS opcode did not fail the contract")

    helper_forbidden_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D", "UTCHMMA", "LDTM.16"),
        sass_function("helper", "UTCCP.128"),
    )
    if not inspect(valid_ptx, helper_forbidden_sass).passed:
        raise AssertionError("forbidden opcode in a helper polluted the target contract")

    require_error("missing_ptx", "found 0", lambda: inspect(PTX_HEADER, valid_sass))
    require_error(
        "wrong_ptx_target",
        "PTX target sm_100a",
        lambda: inspect(valid_ptx.replace("sm_110a", "sm_100a"), valid_sass),
    )
    require_error(
        "wrong_sass_arch",
        "SASS architecture sm_100a",
        lambda: inspect(
            valid_ptx,
            valid_sass,
            metadata=CODE_OBJECT_METADATA.replace("sm_110a", "sm_100a"),
        ),
    )
    require_error(
        "multiple_sass_arches",
        "exactly one architecture",
        lambda: inspect(valid_ptx, valid_sass, metadata=CODE_OBJECT_METADATA + ".target sm_100a\n"),
    )

    two_sm_ptx = valid_ptx.replace("cta_group::1", "cta_group::2")
    two_sm_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D.2CTA", "UTCHMMA.2CTA", "LDTM.16")
    )
    two_sm_result = inspect(two_sm_ptx, two_sm_sass)
    if two_sm_result.passed:
        raise AssertionError("2SM function satisfied the 1SM contract")

    require_error(
        "wildcard_required_pattern",
        "wildcard-only",
        lambda: attribute_codegen(
            ptx_text=valid_ptx,
            nvdisasm_json_text=valid_sass,
            elf_symbols_text=elf_symbols("kernel_A"),
            code_object_metadata_text=CODE_OBJECT_METADATA,
            expected_symbol="kernel_A",
            declared_cta_group=1,
            expected_ptx_target="sm_110a",
            expected_sass_arch="sm_110a",
            required_ptx=[".*"],
            forbidden_ptx=[],
            required_sass=[r"LDTM(?:\..*)?"],
            forbidden_sass=[],
        ),
    )
    require_error(
        "invalid_regex",
        "invalid regex",
        lambda: attribute_codegen(
            ptx_text=valid_ptx,
            nvdisasm_json_text=valid_sass,
            elf_symbols_text=elf_symbols("kernel_A"),
            code_object_metadata_text=CODE_OBJECT_METADATA,
            expected_symbol="kernel_A",
            declared_cta_group=1,
            expected_ptx_target="sm_110a",
            expected_sass_arch="sm_110a",
            required_ptx=["["],
            forbidden_ptx=[],
            required_sass=[r"LDTM(?:\..*)?"],
            forbidden_sass=[],
        ),
    )
    require_error(
        "missing_sass_boundaries",
        "found 0",
        lambda: inspect(valid_ptx, sass_json()),
    )

    require_error(
        "missing_entry_symbol",
        "STO_ENTRY",
        lambda: inspect(valid_ptx, valid_sass, entries="symbols:\n"),
    )
    require_error(
        "duplicate_entry_symbol",
        "STO_ENTRY",
        lambda: inspect(valid_ptx, valid_sass, entries=elf_symbols("kernel_A", "kernel_A")),
    )

    helper_required_sass = sass_json(
        sass_function("kernel_A", "MOV", "UTCHMMA", "LDTM.16"),
        sass_function("helper", "UTMALDG.3D"),
    )
    if inspect(valid_ptx, helper_required_sass).sass_contract.passed:
        raise AssertionError("required SASS opcode in helper was attributed to target")

    operand_only_sass = json.loads(valid_sass)
    operand_only_sass[1][0]["sass-instructions"][0] = {
        "opcode": "MOV",
        "operands": "UTMALDG.3D,UTCHMMA,LDTM.16",
    }
    if inspect(valid_ptx, json.dumps(operand_only_sass)).sass_contract.passed:
        raise AssertionError("SASS operands were interpreted as opcodes")

    require_error(
        "malformed_nvdisasm_json",
        "nvdisasm JSON is invalid",
        lambda: inspect(valid_ptx, "not-json"),
    )
    duplicate_key_json = valid_sass.replace(
        '"function-name":"kernel_A"',
        '"function-name":"evil","function-name":"kernel_A"',
        1,
    )
    require_error(
        "duplicate_nvdisasm_key",
        "duplicate key",
        lambda: inspect(valid_ptx, duplicate_key_json),
    )
    nonfinite_json = valid_sass.replace('"start":0', '"start":NaN', 1)
    require_error(
        "nonfinite_nvdisasm_number",
        "non-finite",
        lambda: inspect(valid_ptx, nonfinite_json),
    )

    missing_metadata = json.loads(valid_sass)
    missing_metadata[0][".note.nv.tkinfo"] = "not-an-object"
    require_error(
        "missing_nvdisasm_tool_options",
        "tki_toolOptions",
        lambda: inspect(valid_ptx, json.dumps(missing_metadata)),
    )

    missing_opcode = json.loads(valid_sass)
    missing_opcode[1][0]["sass-instructions"][0] = {"operands": "R1,R2"}
    require_error(
        "missing_sass_opcode",
        "requires opcode",
        lambda: inspect(valid_ptx, json.dumps(missing_opcode)),
    )

    real_header_metadata = (
        "arch = sm_110a\ncode for sm_110a\n.target sm_110a\n"
        '.headerflags @"EF_CUDA_SM110 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM110)"\n'
    )
    if not inspect(valid_ptx, valid_sass, metadata=real_header_metadata).passed:
        raise AssertionError("real SM110 headerflags caused an sm_110a false rejection")

    ordinary_function = "symbols:\nSTT_FUNC STB_LOCAL STV_DEFAULT kernel_A\n"
    require_error(
        "ordinary_function_not_entry",
        "STO_ENTRY",
        lambda: inspect(valid_ptx, valid_sass, entries=ordinary_function),
    )

    zero_operand_ptx = valid_ptx.replace(
        "tcgen05.ld.sync.aligned %r1, %r2;",
        "tcgen05.ld.sync.aligned %r1, %r2;\n"
        "  tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;",
    )
    zero_opcodes = parse_ptx_entries(zero_operand_ptx)[0].opcodes
    if "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned" not in zero_opcodes:
        raise AssertionError("zero-operand PTX lifecycle opcode was not parsed")

    bracket_operand_ptx = valid_ptx.replace(
        "  tcgen05.ld.sync.aligned %r1, %r2;",
        "  tcgen05.st.sync.aligned.32x32b.x32.b32[%r1],{%r2};",
    )
    bracket_opcodes = parse_ptx_entries(bracket_operand_ptx)[0].opcodes
    if "tcgen05.st.sync.aligned.32x32b.x32.b32" not in bracket_opcodes:
        raise AssertionError("PTX opcode adjacent to a bracket operand was not parsed")

    nested_ptx = valid_ptx.replace(
        "tcgen05.mma.cta_group::1.kind::f16 %r1, %r2;",
        "{\n  .reg .u32 nested;\n"
        "  tcgen05.mma.cta_group::1.kind::f16 %r1, %r2;\n}",
    )
    if not inspect(nested_ptx, valid_sass).passed:
        raise AssertionError("nested PTX braces truncated the target entry")

    func_only_ptx = (
        PTX_HEADER
        + ptx_function("kernel_A", "mov.u32")
        + ".func helper()\n{\n  cp.async.bulk.tensor.2d %r1, %r2;\n"
        + "  tcgen05.mma.cta_group::1.kind::f16 %r1, %r2;\n"
        + "  tcgen05.ld.sync.aligned %r1, %r2;\n}\n"
    )
    if inspect(func_only_ptx, valid_sass).ptx_contract.passed:
        raise AssertionError("required opcodes in a trailing .func polluted the entry")

    mixed_group_ptx = valid_ptx.replace(
        "tcgen05.ld.sync.aligned %r1, %r2;",
        "tcgen05.ld.sync.aligned %r1, %r2;\n"
        "  tcgen05.mma.cta_group::2.kind::f16 %r1, %r2;",
    )
    mixed_result = inspect_group(mixed_group_ptx, valid_sass, 1)
    if mixed_result.cta_group_contract.passed:
        raise AssertionError("mixed PTX CTA-groups passed the structural gate")

    two_group_ptx = valid_ptx.replace("cta_group::1", "cta_group::2")
    two_group_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D.2CTA", "UTCHMMA.2CTA", "LDTM.16")
    )
    if inspect_group(two_group_ptx, valid_sass, 2).cta_group_contract.passed:
        raise AssertionError("2SM PTX plus 1SM SASS passed")
    if inspect_group(valid_ptx, two_group_sass, 1).cta_group_contract.passed:
        raise AssertionError("1SM PTX plus 2SM SASS passed")
    input_transform_sass = sass_json(
        sass_function("kernel_A", "UTMALDG.3D.MULTICAST", "UTCHMMA.2CTA", "LDTM.16")
    )
    if inspect_group(two_group_ptx, input_transform_sass, 2).cta_group_contract.passed:
        raise AssertionError("strict 2SM TMA policy accepted a 1CTA input-transform load")
    if not inspect_group(
        two_group_ptx,
        input_transform_sass,
        2,
        require_tma_group_match=False,
    ).cta_group_contract.passed:
        raise AssertionError("input-transform TMA policy rejected valid 2CTA MMA with 1CTA TMA")

    pseudo_opcode = sass_json(
        sass_function("kernel_A", "NOTUTMALDG", "NOTUTCHMMA", "NOTLDTM")
    )
    if inspect(valid_ptx, pseudo_opcode).sass_contract.passed:
        raise AssertionError("substring-like pseudo opcodes passed fullmatch contracts")

    print("ATTRIBUTION_V2_ADVERSARIAL_PASS cases=34 positive=7")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
