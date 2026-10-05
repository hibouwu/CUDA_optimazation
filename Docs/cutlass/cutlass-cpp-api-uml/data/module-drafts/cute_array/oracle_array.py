#!/usr/bin/env python3
"""Six bounded host-C++17 syntax-only oracles for snapshot cute/container/array.hpp.

Print complete source and actual compiler results as JSON. No C++ program is run,
no GPU is used, and this script writes no files. Compilation source is sent on
stdin so each recorded command is reproducible with its recorded source.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


PRELUDE = """#include <cute/container/array.hpp>
#include <type_traits>
#include <utility>
"""

SAMPLES = [
    {
        "id": "clear_zero_member_vs_free",
        "source_lines": [326, 332, 351, 355],
        "question": "N=0 的成员 clear 不要求 T(0)，free clear 是否仍要求？",
        "source": PRELUDE + """
struct NoZero {
  NoZero() = delete;
  NoZero(int) = delete;
};
void probe() {
  cute::array<NoZero, 0> a{};
#if FREE_CLEAR
  cute::clear(a);
#else
  a.clear();
#endif
}
""",
        "variants": [
            {"id": "member", "defines": ["FREE_CLEAR=0"], "expected_exit_code": 0},
            {"id": "free", "defines": ["FREE_CLEAR=1"], "expected_exit_code": 1,
             "diagnostic_contains": ["functional-style cast from 'int' to 'NoZero' uses deleted function", "a.fill(T(0))"]},
        ],
    },
    {
        "id": "equality_zero_still_checks_not_equal",
        "source_lines": [339, 349],
        "question": "普通 for 的 N=0 是否免去 T 的 != 要求？仅有 == 是否足够？",
        "source": PRELUDE + """
struct Element {};
#if COMPARISON == 1
constexpr bool operator==(Element const&, Element const&) { return true; }
#elif COMPARISON == 2
constexpr bool operator!=(Element const&, Element const&) { return false; }
#endif
constexpr cute::array<Element, 0> a{}, b{};
static_assert(a == b);
""",
        "variants": [
            {"id": "no_comparison", "defines": ["COMPARISON=0"], "expected_exit_code": 1,
             "diagnostic_contains": ["invalid operands to binary expression", "lhs[i] != rhs[i]"]},
            {"id": "only_equal", "defines": ["COMPARISON=1"], "expected_exit_code": 1,
             "diagnostic_contains": ["invalid operands to binary expression", "lhs[i] != rhs[i]"]},
            {"id": "not_equal_available", "defines": ["COMPARISON=2"], "expected_exit_code": 0},
        ],
    },
    {
        "id": "reverse_zero_vs_nonzero_element_requirements",
        "source_lines": [372, 385],
        "question": "if constexpr 的零长度分支与非零分支分别实例化哪些元素操作？",
        "source": PRELUDE + """
#if REVERSE_CASE == 0
struct Element {
  Element() = delete;
  Element(Element const&) = delete;
  Element(Element&&) = delete;
  Element& operator=(Element const&) = delete;
};
void probe() {
  cute::array<Element, 0> a{};
  auto result = cute::reverse(a);
  static_assert(std::is_empty_v<decltype(result)>);
}
#elif REVERSE_CASE == 1
struct Element {
  Element() = delete;
  Element(Element const&) = default;
  Element& operator=(Element const&) = default;
};
static_assert(std::is_aggregate_v<Element>);
static_assert(!std::is_default_constructible_v<Element>);
void probe(cute::array<Element, 1> const& a) { (void)cute::reverse(a); }
#elif REVERSE_CASE == 2
struct Element {
  Element() = default;
  Element(Element const&) = default;
  Element& operator=(Element const&) = delete;
};
void probe(cute::array<Element, 1> const& a) { (void)cute::reverse(a); }
#elif REVERSE_CASE == 3
struct Element {
  Element() = default;
  Element(Element const&) = delete;
  Element(Element&&) = default;
  Element& operator=(Element const&) = default;
};
static_assert(!std::is_copy_constructible_v<Element>);
static_assert(std::is_move_constructible_v<Element>);
void probe(cute::array<Element, 1> const& a) { (void)cute::reverse(a); }
#elif REVERSE_CASE == 4
struct Element {
  Element() = delete;
  explicit Element(int) {}
  Element(Element const&) = default;
  Element& operator=(Element const&) = default;
};
static_assert(!std::is_aggregate_v<Element>);
void probe(cute::array<Element, 1> const& a) { (void)cute::reverse(a); }
#endif
""",
        "variants": [
            {"id": "zero_no_element_construction_or_assignment", "defines": ["REVERSE_CASE=0"], "expected_exit_code": 0},
            {"id": "nonzero_aggregate_deleted_default_constructor_still_passes", "defines": ["REVERSE_CASE=1"], "expected_exit_code": 0},
            {"id": "nonzero_deleted_copy_assignment", "defines": ["REVERSE_CASE=2"], "expected_exit_code": 1,
             "diagnostic_contains": ["overload resolution selected deleted operator '='", "t_r[k] = t[N - k - 1]"]},
            {"id": "nonzero_no_copy_constructor_but_movable", "defines": ["REVERSE_CASE=3"], "expected_exit_code": 0},
            {"id": "nonzero_nonaggregate_deleted_default_constructor", "defines": ["REVERSE_CASE=4"], "expected_exit_code": 1,
             "diagnostic_contains": ["call to deleted constructor of 'element_type'", "t_r{}"]},
        ],
    },
    {
        "id": "get_cvref_and_original_storage",
        "source_lines": [404, 426],
        "question": "三个 get 重载的 cv/ref 返回类型、const&& 回退以及原存储别名。",
        "source": PRELUDE + """
using A = cute::array<int, 1>;
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A&>())), int&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A const&>())), int const&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A&&>())), int&&>);
static_assert(std::is_same_v<decltype(cute::get<0>(std::declval<A const&&>())), int const&>);
template<class X, class = void> struct accepts_get : std::false_type {};
template<class X> struct accepts_get<X, std::void_t<decltype(cute::get<0>(std::declval<X>()))>>
    : std::true_type {};
static_assert(!accepts_get<A volatile&>::value);
static_assert(!accepts_get<A const volatile&>::value);
constexpr bool refers_to_original_element() {
  A a{{7}};
  int&& element = cute::get<0>(static_cast<A&&>(a));
  element = 11;
  return &element == a.data() && a[0] == 11;
}
static_assert(refers_to_original_element());
""",
        "variants": [{"id": "cvref_and_constexpr_alias", "defines": [], "expected_exit_code": 0}],
    },
    {
        "id": "tuple_element_vs_get_body_bound_check",
        "source_lines": [404, 426, 438, 442],
        "question": "tuple_element 不检查 I<N；get 的 static_assert 何时触发？",
        "source": PRELUDE + """
using Empty = cute::array<int, 0>;
static_assert(std::tuple_size<Empty>::value == 0);
static_assert(std::is_same_v<std::tuple_element<999, Empty>::type, int>);
// Unevaluated return-type inspection does not instantiate this function body.
static_assert(std::is_same_v<decltype(cute::get<999>(std::declval<Empty&>())), int&>);
#if CALL_GET
void probe(Empty& a) { (void)cute::get<999>(a); }
#endif
""",
        "variants": [
            {"id": "traits_and_unevaluated_get", "defines": ["CALL_GET=0"], "expected_exit_code": 0},
            {"id": "instantiate_get_body", "defines": ["CALL_GET=1"], "expected_exit_code": 1,
             "diagnostic_contains": ["static assertion failed", "Index out of range", "999UL < 0UL"]},
        ],
    },
    {
        "id": "empty_logical_size_not_zero_object_size",
        "source_lines": [198, 337],
        "question": "N=0 是无元素/逻辑长度零，不是 sizeof 为零。",
        "source": PRELUDE + """
struct NoDefault { NoDefault() = delete; };
using Empty = cute::array<NoDefault, 0>;
static_assert(std::is_empty_v<Empty>);
static_assert(std::is_default_constructible_v<Empty>);
static_assert(sizeof(Empty) > 0);
// This exact size is a recorded property of this host compiler/ABI, not a portable API guarantee.
static_assert(sizeof(Empty) == 1);
constexpr Empty a{};
static_assert(a.empty() && a.size() == 0 && a.max_size() == 0);
static_assert(a.data() == nullptr && a.begin() == nullptr && a.end() == nullptr);
""",
        "variants": [{"id": "logical_zero_object_nonzero", "defines": [], "expected_exit_code": 0}],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="/usr/bin/clang++-21")
    parser.add_argument("--cuda-include", default="/usr/local/cuda-13.0/targets/x86_64-linux/include")
    args = parser.parse_args()
    atlas = Path(__file__).resolve().parents[3]
    header = atlas / "snapshot/include/cute/container/array.hpp"
    version = subprocess.run([args.compiler, "--version"], capture_output=True, text=True, check=True)
    header_hash = subprocess.run(["sha256sum", str(header)], capture_output=True, text=True, check=True).stdout.split()[0]
    flags = ["-std=c++17", "-fsyntax-only", "-fno-color-diagnostics", "-x", "c++",
             "-I", str(atlas / "snapshot/include"), "-I", args.cuda_include,
             "-I", str(Path(args.cuda_include) / "cccl")]
    results = []
    for sample in SAMPLES:
        item = {key: value for key, value in sample.items() if key != "variants"}
        item["variants"] = []
        for variant in sample["variants"]:
            command = [args.compiler, *flags, *["-D" + define for define in variant["defines"]], "-"]
            process = subprocess.run(command, input=sample["source"], capture_output=True, text=True)
            diagnostic_matches = all(fragment in process.stderr for fragment in variant.get("diagnostic_contains", []))
            entry = {**variant, "command": command, "exit_code": process.returncode,
                     "stdout": process.stdout, "stderr": process.stderr,
                     "expectation_met": process.returncode == variant["expected_exit_code"] and diagnostic_matches}
            item["variants"].append(entry)
        results.append(item)
    variants = [variant for item in results for variant in item["variants"]]
    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_header": str(header),
        "source_header_sha256": header_hash,
        "compiler": args.compiler,
        "compiler_version": version.stdout.strip(),
        "scope": "Host Clang 21 C++17 syntax-only and constant-expression checks. No object, executable, GPU execution, runtime safety, codegen, or performance validation.",
        "sample_count": len(results),
        "compilation_count": len(variants),
        "expectations_met": sum(variant["expectation_met"] for variant in variants),
        "results": results,
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0 if all(variant["expectation_met"] for variant in variants) else 1


if __name__ == "__main__":
    sys.exit(main())
