"""Function-scoped PTX/SASS parsing and instruction contracts.

The parser selects a declared symbol first. Instruction patterns are evaluated
only inside that function and only against parsed opcode tokens. It never picks
a function because its body happens to contain the requested mnemonics.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


class AttributionError(ValueError):
    """The artifact cannot be attributed to one declared target function."""


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise AttributionError(f"nvdisasm JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise AttributionError(f"nvdisasm JSON contains non-finite number: {value}")


@dataclass(frozen=True)
class FunctionBlock:
    symbol: str
    text: str
    opcodes: tuple[str, ...]


@dataclass(frozen=True)
class ContractResult:
    required_matches: dict[str, tuple[str, ...]]
    forbidden_matches: dict[str, tuple[str, ...]]

    @property
    def passed(self) -> bool:
        return all(self.required_matches.values()) and not any(self.forbidden_matches.values())


@dataclass(frozen=True)
class CtaGroupResult:
    declared_cta_group: int
    ptx_mma_opcodes: tuple[str, ...]
    sass_mma_opcodes: tuple[str, ...]
    sass_tma_load_opcodes: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class AttributionResult:
    symbol: str
    ptx_target: str
    sass_arch: str
    sass_total_function_count: int
    ptx_function: FunctionBlock
    sass_function: FunctionBlock
    ptx_contract: ContractResult
    sass_contract: ContractResult
    cta_group_contract: CtaGroupResult

    @property
    def passed(self) -> bool:
        return (
            self.ptx_contract.passed
            and self.sass_contract.passed
            and self.cta_group_contract.passed
        )


def _strip_comments(text: str) -> str:
    def preserve_newlines(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    text = re.sub(r"/\*[\s\S]*?\*/", preserve_newlines, text)
    return re.sub(r"//[^\n]*", "", text)


def validate_patterns(patterns: list[str], field: str) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for index, pattern in enumerate(patterns):
        location = f"{field}[{index}]"
        if not isinstance(pattern, str) or not pattern.strip():
            raise AttributionError(f"{location}: pattern must be a nonempty string")
        if pattern.strip() in {".*", ".+", "^.*$", "^.+$"}:
            raise AttributionError(f"{location}: wildcard-only patterns are forbidden")
        try:
            expression = re.compile(pattern, re.IGNORECASE)
        except re.error as error:
            raise AttributionError(f"{location}: invalid regex: {error}") from error
        if expression.search("") is not None:
            raise AttributionError(f"{location}: zero-width/empty-string matches are forbidden")
        if len(re.findall(r"[A-Za-z0-9]", pattern)) < 3:
            raise AttributionError(f"{location}: pattern lacks a meaningful opcode literal")
        compiled.append(expression)
    return tuple(compiled)


def _balanced_block(text: str, opening_brace: int) -> tuple[str, int]:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening_brace : index + 1], index + 1
    raise AttributionError("PTX entry has no balanced closing brace")


def _ptx_opcodes(function_text: str) -> tuple[str, ...]:
    opcodes: list[str] = []
    for line in function_text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith((".", "{", "}", "(")):
            continue
        while True:
            label = re.match(r"^[A-Za-z_$][A-Za-z0-9_.$]*:\s+", candidate)
            if label is None:
                break
            candidate = candidate[label.end() :].lstrip()
        predicate = re.match(r"^@!?%[A-Za-z0-9_.$]+\s+", candidate)
        if predicate is not None:
            candidate = candidate[predicate.end() :]
        opcode = re.match(r"^([A-Za-z][A-Za-z0-9_.:]*)(?=\s|;|\{|\[)", candidate)
        if opcode is not None:
            opcodes.append(opcode.group(1))
    return tuple(opcodes)


def parse_ptx_entries(text: str) -> list[FunctionBlock]:
    cleaned = _strip_comments(text)
    header = re.compile(
        r"(?m)^\s*(?:(?:\.visible|\.weak)\s+)*\.entry\s+([A-Za-z_$][A-Za-z0-9_.$]*)\s*\("
    )
    blocks: list[FunctionBlock] = []
    for match in header.finditer(cleaned):
        opening_brace = cleaned.find("{", match.end())
        if opening_brace < 0:
            raise AttributionError(f"PTX entry {match.group(1)} has no body")
        body, end = _balanced_block(cleaned, opening_brace)
        block_text = cleaned[match.start() : end]
        blocks.append(FunctionBlock(match.group(1), block_text, _ptx_opcodes(body)))
    return blocks


def parse_ptx_target(text: str) -> str:
    targets = re.findall(r"(?m)^\s*\.target\s+([A-Za-z0-9_]+)", _strip_comments(text))
    if len(targets) != 1:
        raise AttributionError(f"PTX module must contain exactly one .target, found {targets}")
    return targets[0].lower()


def _sass_opcodes(function_text: str) -> tuple[str, ...]:
    opcodes: list[str] = []
    for line in function_text.splitlines():
        candidate = re.sub(r"/\*[0-9A-Fa-fx]+\*/", "", line).strip()
        if not candidate or candidate.startswith((".", "//", "{")):
            continue
        label = re.match(r"^[A-Za-z_$][A-Za-z0-9_.$]*:\s+", candidate)
        if label is not None:
            candidate = candidate[label.end() :].lstrip()
        predicate = re.match(r"^@!?[UP]?P[0-9]+\s+", candidate, re.IGNORECASE)
        if predicate is not None:
            candidate = candidate[predicate.end() :]
        opcode = re.match(r"^([A-Z][A-Z0-9_.]*)\s", candidate, re.IGNORECASE)
        if opcode is not None:
            opcodes.append(opcode.group(1).upper())
    return tuple(opcodes)


def parse_sass_functions(text: str) -> list[FunctionBlock]:
    section_matches = list(
        re.finditer(r"(?m)^\s*\.section\s+\.text\.([^,\s]+)[^\n]*$", text)
    )
    if section_matches:
        blocks: list[FunctionBlock] = []
        for index, match in enumerate(section_matches):
            end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(text)
            block_text = text[match.start() : end]
            blocks.append(FunctionBlock(match.group(1), block_text, _sass_opcodes(block_text)))
        return blocks

    function_matches = list(re.finditer(r"(?m)^\s*Function\s*:\s*(\S+)\s*$", text))
    blocks = []
    for index, match in enumerate(function_matches):
        end = function_matches[index + 1].start() if index + 1 < len(function_matches) else len(text)
        block_text = text[match.start() : end]
        blocks.append(FunctionBlock(match.group(1), block_text, _sass_opcodes(block_text)))
    return blocks


def parse_sass_arch(text: str) -> str:
    candidates = {
        match.lower()
        for match in re.findall(
            r"(?im)(?:^\s*arch\s*=\s*|^\s*code\s+for\s+|^\s*\.target\s+)(sm_[0-9]+[a-z]?)\b",
            text,
        )
    }
    if len(candidates) != 1:
        raise AttributionError(
            f"SASS artifact must identify exactly one architecture, found {sorted(candidates)}"
        )
    return next(iter(candidates))


def parse_entry_symbols(text: str) -> tuple[str, ...]:
    entries: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^\s*STT_FUNC\s+\S+\s+STO_ENTRY\s+(\S+)\s*$", line)
        if match is not None:
            entries.append(match.group(1))
    return tuple(entries)


def parse_nvdisasm_json(text: str) -> tuple[str, list[FunctionBlock]]:
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise AttributionError(f"nvdisasm JSON is invalid: {error}") from error
    if (
        not isinstance(payload, list)
        or len(payload) != 2
        or not isinstance(payload[0], dict)
        or not isinstance(payload[1], list)
    ):
        raise AttributionError("nvdisasm JSON must be [metadata, functions]")
    metadata = payload[0]
    tkinfo = metadata.get(".note.nv.tkinfo")
    if not isinstance(tkinfo, dict) or not isinstance(tkinfo.get("tki_toolOptions"), str):
        raise AttributionError("nvdisasm metadata requires .note.nv.tkinfo.tki_toolOptions")
    tool_options = tkinfo["tki_toolOptions"]
    arch_matches = re.findall(r"(?:^|\s)-arch\s+(sm_[0-9]+[a-z]?)(?:\s|$)", tool_options)
    if len(arch_matches) != 1:
        raise AttributionError(
            f"nvdisasm metadata must identify one -arch value, found {arch_matches}"
        )
    functions: list[FunctionBlock] = []
    for index, function in enumerate(payload[1]):
        if not isinstance(function, dict):
            raise AttributionError(f"nvdisasm functions[{index}] must be an object")
        symbol = function.get("function-name")
        instructions = function.get("sass-instructions")
        if not isinstance(symbol, str) or not isinstance(instructions, list):
            raise AttributionError(
                f"nvdisasm functions[{index}] requires function-name and sass-instructions"
            )
        opcodes: list[str] = []
        for instruction_index, instruction in enumerate(instructions):
            if not isinstance(instruction, dict) or not isinstance(instruction.get("opcode"), str):
                raise AttributionError(
                    f"nvdisasm functions[{index}].sass-instructions[{instruction_index}] "
                    "requires opcode"
                )
            opcodes.append(instruction["opcode"].upper())
        functions.append(FunctionBlock(symbol, json.dumps(function, separators=(",", ":")), tuple(opcodes)))
    return arch_matches[0].lower(), functions


def select_unique(blocks: list[FunctionBlock], symbol: str, artifact: str) -> FunctionBlock:
    matches = [block for block in blocks if block.symbol == symbol]
    if len(matches) != 1:
        raise AttributionError(
            f"{artifact}: declared symbol {symbol!r} must occur exactly once, found {len(matches)}"
        )
    return matches[0]


def evaluate_contract(
    opcodes: tuple[str, ...],
    required: list[str],
    forbidden: list[str],
    field: str,
) -> ContractResult:
    required_regex = validate_patterns(required, f"{field}.required")
    forbidden_regex = validate_patterns(forbidden, f"{field}.forbidden")
    required_matches = {
        pattern.pattern: tuple(opcode for opcode in opcodes if pattern.fullmatch(opcode))
        for pattern in required_regex
    }
    forbidden_matches = {
        pattern.pattern: tuple(opcode for opcode in opcodes if pattern.fullmatch(opcode))
        for pattern in forbidden_regex
    }
    return ContractResult(required_matches, forbidden_matches)


def evaluate_cta_group(
    ptx_opcodes: tuple[str, ...],
    sass_opcodes: tuple[str, ...],
    declared_cta_group: int,
    require_tma_group_match: bool = True,
) -> CtaGroupResult:
    if declared_cta_group not in (1, 2):
        raise AttributionError("declared_cta_group must be 1 or 2")
    ptx_group_pattern = re.compile(
        r"tcgen05\.mma(?:\.[A-Za-z0-9_]+)*\.cta_group::([12])(?:\..*)?",
        re.IGNORECASE,
    )
    ptx_mma = tuple(opcode for opcode in ptx_opcodes if ptx_group_pattern.fullmatch(opcode))
    sass_mma_pattern = re.compile(r"UTC[A-Z0-9_]*MMA(?:\..*)?", re.IGNORECASE)
    sass_mma = tuple(opcode for opcode in sass_opcodes if sass_mma_pattern.fullmatch(opcode))
    sass_tma = tuple(
        opcode
        for opcode in sass_opcodes
        if re.fullmatch(r"UTMALDG(?:\..*)?", opcode, re.IGNORECASE)
    )
    errors: list[str] = []
    if not ptx_mma:
        errors.append("target PTX contains no tcgen05.mma opcode with CTA-group")
    else:
        observed_groups = {int(ptx_group_pattern.fullmatch(opcode).group(1)) for opcode in ptx_mma}
        if observed_groups != {declared_cta_group}:
            errors.append(
                f"target PTX MMA CTA-groups {sorted(observed_groups)} != [{declared_cta_group}]"
            )
    if not sass_mma:
        errors.append("target SASS contains no tensor MMA opcode")
    elif declared_cta_group == 1 and any(".2CTA" in opcode for opcode in sass_mma):
        errors.append("1SM target SASS contains a 2CTA MMA opcode")
    elif declared_cta_group == 2 and any(".2CTA" not in opcode for opcode in sass_mma):
        errors.append("2SM target SASS contains a non-2CTA MMA opcode")
    if require_tma_group_match and declared_cta_group == 1 and any(".2CTA" in opcode for opcode in sass_tma):
        errors.append("1SM target SASS contains a 2CTA TMA-load opcode")
    if require_tma_group_match and declared_cta_group == 2 and sass_tma and not any(".2CTA" in opcode for opcode in sass_tma):
        errors.append("2SM target SASS contains no 2CTA TMA-load opcode")
    return CtaGroupResult(declared_cta_group, ptx_mma, sass_mma, sass_tma, tuple(errors))


def attribute_codegen(
    *,
    ptx_text: str,
    nvdisasm_json_text: str,
    elf_symbols_text: str,
    code_object_metadata_text: str,
    expected_symbol: str,
    declared_cta_group: int,
    expected_ptx_target: str,
    expected_sass_arch: str,
    required_ptx: list[str],
    forbidden_ptx: list[str],
    required_sass: list[str],
    forbidden_sass: list[str],
    require_tma_group_match: bool = True,
) -> AttributionResult:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expected_symbol):
        raise AttributionError("expected_symbol must be a stable C identifier")
    ptx_target = parse_ptx_target(ptx_text)
    sass_arch = parse_sass_arch(code_object_metadata_text)
    nvdisasm_arch, sass_functions = parse_nvdisasm_json(nvdisasm_json_text)
    if ptx_target != expected_ptx_target.lower():
        raise AttributionError(f"PTX target {ptx_target} != {expected_ptx_target.lower()}")
    if sass_arch != expected_sass_arch.lower():
        raise AttributionError(f"SASS architecture {sass_arch} != {expected_sass_arch.lower()}")
    if nvdisasm_arch != sass_arch:
        raise AttributionError(f"nvdisasm architecture {nvdisasm_arch} != code object {sass_arch}")
    ptx_function = select_unique(parse_ptx_entries(ptx_text), expected_symbol, "PTX")
    entry_symbols = parse_entry_symbols(elf_symbols_text)
    if entry_symbols.count(expected_symbol) != 1:
        raise AttributionError(
            f"ELF: declared symbol {expected_symbol!r} must occur exactly once as STO_ENTRY, "
            f"found {entry_symbols.count(expected_symbol)}"
        )
    sass_function = select_unique(sass_functions, expected_symbol, "SASS")
    if ptx_function.symbol != sass_function.symbol:
        raise AttributionError(
            f"PTX symbol {ptx_function.symbol!r} != SASS symbol {sass_function.symbol!r}"
        )
    return AttributionResult(
        expected_symbol,
        ptx_target,
        sass_arch,
        len(sass_functions),
        ptx_function,
        sass_function,
        evaluate_contract(ptx_function.opcodes, required_ptx, forbidden_ptx, "ptx"),
        evaluate_contract(sass_function.opcodes, required_sass, forbidden_sass, "sass"),
        evaluate_cta_group(
            ptx_function.opcodes,
            sass_function.opcodes,
            declared_cta_group,
            require_tma_group_match,
        ),
    )
