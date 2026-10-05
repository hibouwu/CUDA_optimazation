"""Independent fixed-source and Clang alignment oracles; no GPU execution.

These tests establish the original contract before testing the extraction
adapter. In particular, outer alignof alone cannot detect lost field alignment.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scan_candidates import lex

ALIGNMENT = "include/cute/container/alignment.hpp"
ARRAY = "include/cute/container/array_aligned.hpp"
FP8 = "include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp"
MIXED = "include/cutlass/gemm/collective/sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp"
ARRAY_MIXED = "include/cutlass/gemm/collective/sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input.hpp"
EXPECTED = [(ALIGNMENT, 60 + i, str(2 ** i), "class", "aligned_struct") for i in range(9)] + [
    (ARRAY, 40, "Alignment", "class", "array_aligned"),
    (FP8, 216, "128", "field", "smem_SFA"),
    (FP8, 217, "128", "field", "smem_SFB"),
    (MIXED, 291, "SmemAlignmentA", "field", "smem_A"),
    (MIXED, 292, "SmemAlignmentB", "field", "smem_B"),
    (ARRAY_MIXED, 271, "SmemAlignmentA", "field", "smem_A"),
    (ARRAY_MIXED, 272, "SmemAlignmentB", "field", "smem_B"),
]


def lexical_calls(path, raw):
    """Independent token inventory, not extractor output or repaired AST."""
    tokens, _, _ = lex(raw)
    calls = []
    for index, token in enumerate(tokens[:-1]):
        if token.text != "CUTE_ALIGNAS" or tokens[index + 1].text != "(":
            continue
        line_start = raw.rfind(b"\n", 0, token.start) + 1
        if re.match(rb"\s*#\s*define\s*$", raw[line_start:token.start]):
            continue
        depth, end = 1, index + 2
        while end < len(tokens) and depth:
            depth += (tokens[end].text == "(") - (tokens[end].text == ")")
            end += 1
        if depth:
            raise AssertionError((path, token.start, "unclosed macro call"))
        close = tokens[end - 1]
        calls.append({"path": path, "line": raw[:token.start].count(b"\n") + 1,
                      "start": token.start, "end": close.end,
                      "argument": raw[tokens[index + 1].end:close.start].decode().strip()})
    return calls


class CuteAlignmentSourceReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope = json.loads((ROOT / "data/scope.json").read_text())
        cls.calls, cls.sources = [], {}
        if len(scope["files"]) != 824:
            raise AssertionError("The frozen file scope must remain 824")
        for record in scope["files"]:
            raw = (ROOT / "snapshot" / record["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise AssertionError((record["path"], "snapshot hash mismatch"))
            calls = lexical_calls(record["path"], raw)
            cls.calls.extend(calls)
            if calls or record["path"] == ALIGNMENT:
                cls.sources[record["path"]] = raw

    def test_frozen_scope_contains_exactly_sixteen_uses_in_five_files(self):
        actual = {(c["path"], c["line"], c["argument"]) for c in self.calls}
        self.assertEqual(actual, {(p, line, arg) for p, line, arg, _, _ in EXPECTED})
        self.assertEqual(len(self.calls), 16)
        self.assertEqual(len(self.sources), 5)

    def test_class_key_and_field_prefix_have_distinct_owners(self):
        for path, line, expression, kind, owner in EXPECTED:
            raw = self.sources[path]
            call = next(c for c in self.calls if (c["path"], c["line"]) == (path, line))
            text = raw.decode().splitlines()[line - 1]
            before, after = text.split("CUTE_ALIGNAS", 1)
            with self.subTest(path=path, line=line):
                if kind == "class":
                    self.assertRegex(before, r"\bstruct\s+$")
                    self.assertRegex(after, r"\)\s+" + owner + r"\b")
                else:
                    self.assertFalse(before.strip())
                    self.assertRegex(after, r"\b" + owner + r"\s*;")
                    # These fixed-source uses are within the immediately
                    # preceding TensorStorage declaration, not its parent.
                    prefix = raw[:call["start"]].decode()
                    self.assertGreater(prefix.rfind("struct TensorStorage"),
                                       prefix.rfind("struct SharedStorage"))

    def test_dependent_alignment_expressions_are_not_literal_128(self):
        dependent = Counter(c["argument"] for c in self.calls if not c["argument"].isdigit())
        self.assertEqual(dependent, {"Alignment": 1, "SmemAlignmentA": 2, "SmemAlignmentB": 2})
        for path in [MIXED, ARRAY_MIXED]:
            text = self.sources[path].decode()
            self.assertIn("alignment_for_swizzle(SmemLayoutA{})", text)
            self.assertIn("alignment_for_swizzle(SmemLayoutB{})", text)
            self.assertIn("SmemAlignmentA >= 128 and SmemAlignmentB >= 128", text)

    def test_both_fixed_macro_definitions_and_exact_condition_are_retained(self):
        lines = self.sources[ALIGNMENT].decode().splitlines()
        self.assertEqual(lines[50:55], ["#if defined(__CUDACC__)",
            "#  define CUTE_ALIGNAS(n) __align__(n)", "#else",
            "#  define CUTE_ALIGNAS(n) alignas(n)", "#endif"])


class CuteAlignmentClangReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which("clang++")
        if not cls.clang:
            raise unittest.SkipTest("Clang is required for the independent C++ oracle")
        candidates = [Path("/usr/local/cuda-13.0/targets/x86_64-linux/include"),
                      Path("/usr/local/cuda/targets/x86_64-linux/include")]
        cls.cuda_include = next((p for p in candidates if (p / "cccl/cuda/std/utility").is_file()), None)
        if cls.cuda_include is None:
            raise unittest.SkipTest("CUDA/CCCL headers needed by the original snapshot headers are unavailable")

    def compile(self, source, extra=()):
        return subprocess.run([self.clang, "-std=c++17", "-I", str(ROOT / "snapshot/include"),
            "-I", str(self.cuda_include), "-I", str(self.cuda_include / "cccl"),
            "-x", "c++", "-fsyntax-only", *extra, "-"],
            input=source, text=True, capture_output=True, timeout=60)

    @staticmethod
    def layout_source(attributes=True, original_expectations=True):
        attr = "CUTE_ALIGNAS(128) " if attributes else ""
        a, b, size = (128, 256, 384) if original_expectations else (4, 16, 128)
        return f'''#include <cstddef>
#include "cute/container/array_aligned.hpp"
struct Probe : cute::aligned_struct<128> {{
  char prefix;
  {attr}cute::array<float,3> a;
  {attr}cute::array<float,3> b;
}};
static_assert(alignof(Probe)==128, "review_outer_alignment");
static_assert(offsetof(Probe,a)=={a}, "review_offset_a");
static_assert(offsetof(Probe,b)=={b}, "review_offset_b");
static_assert(sizeof(Probe)=={size}, "review_total_size");
static_assert(alignof(cute::array_aligned<float,3,64>)==64, "review_dependent_alignment");
'''

    def test_original_headers_preserve_field_offsets(self):
        result = self.compile(self.layout_source())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_removing_only_field_alignment_is_detected_even_with_same_outer_alignof(self):
        result = self.compile(self.layout_source(attributes=False))
        self.assertNotEqual(result.returncode, 0)
        for marker in ["review_offset_a", "review_offset_b", "review_total_size"]:
            self.assertIn(marker, result.stderr)
        self.assertNotIn("file not found", result.stderr)

    def test_removed_fields_have_the_counterexample_offsets(self):
        result = self.compile(self.layout_source(attributes=False, original_expectations=False))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cuda_and_host_macro_branches_preprocess_to_different_spellings(self):
        block = "\n".join((ROOT / "snapshot" / ALIGNMENT).read_text().splitlines()[50:55])
        for cuda, expected in [(False, "alignas(Alignment)"), (True, "__align__(Alignment)")]:
            result = subprocess.run([self.clang, "-E", "-P", "-x", "c++",
                "-D__CUDACC__" if cuda else "-U__CUDACC__", "-"],
                input=block + "\nCUTE_ALIGNAS(Alignment)\n", text=True,
                capture_output=True, timeout=30)
            with self.subTest(cuda=cuda):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_local_redefinition_changes_the_alignment_contract(self):
        source = '''#include "cute/container/alignment.hpp"
#undef CUTE_ALIGNAS
#define CUTE_ALIGNAS(n) alignas(2*(n))
struct CUTE_ALIGNAS(16) Local { char x; };
static_assert(alignof(Local)==32, "review_local_redefinition");
'''
        result = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_undef_can_expose_an_ordinary_same_spelling_function(self):
        source = '''#include "cute/container/alignment.hpp"
#undef CUTE_ALIGNAS
constexpr int CUTE_ALIGNAS(int n) { return n+1; }
static_assert(CUTE_ALIGNAS(16)==17, "review_not_an_alignment_attribute");
'''
        result = self.compile(source)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fixed_macro_wrong_arity_and_expression_position_are_not_valid_attributes(self):
        cases = ["struct CUTE_ALIGNAS(16,32) Bad {char x;};",
                 "struct CUTE_ALIGNAS() Bad {char x;};",
                 "int bad() { return CUTE_ALIGNAS(16); }"]
        for statement in cases:
            result = self.compile('#include "cute/container/alignment.hpp"\n' + statement)
            with self.subTest(statement=statement):
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("file not found", result.stderr)


class CuteAlignmentAdapterReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import declaration_headers
        from extract_declarations import Extractor
        cls.headers = declaration_headers
        cls.raw = {p: (ROOT / "snapshot" / p).read_bytes() for p in {x[0] for x in EXPECTED}}
        cls.analysis = {p: cls.headers.analyze_headers(p, raw) for p, raw in cls.raw.items()}
        cls.extracted = {}
        for path, raw in cls.raw.items():
            extractor = Extractor("independent_alignment_review")
            record = extractor.extract(path, raw)
            cls.extracted[path] = (record, extractor.result())

    @staticmethod
    def cute_edits(result):
        return [e for e in result["projection_edits"] if e["kind"] == "cute_alignment_attribute"]

    def analyze_fixture(self, tail, include=True):
        prefix = '#include "cute/container/alignment.hpp"\n' if include else ""
        return self.headers.analyze_headers("include/cute/review_alignment.hpp", (prefix + tail).encode())

    def test_all_sixteen_have_exact_attribute_expression_and_owner_spans(self):
        seen = set()
        for path, line, expression, kind, name in EXPECTED:
            result, raw = self.analysis[path], self.raw[path]
            attrs = [a for a in result["alignment_attributes"] if a["kind"] == "cute_requested_alignment" and a["start_line"] == line]
            with self.subTest(path=path, line=line):
                self.assertEqual(len(attrs), 1)
                a = attrs[0]
                self.assertEqual(a["alignment_expression"].strip(), expression)
                self.assertEqual(raw[a["start_byte"]:a["end_byte"]].decode(), a["semantic_spelling"])
                x = a["expression_span"]
                self.assertEqual(raw[x["start_byte"]:x["end_byte"]].decode(), a["alignment_expression"])
                self.assertEqual(a["owner_hint"]["kind"], "type" if kind == "class" else "field")
                owner = a["owner_hint"]["declaration_span"]
                text = raw[owner["start_byte"]:owner["end_byte"]].decode()
                self.assertLessEqual(owner["start_byte"], a["start_byte"])
                self.assertGreaterEqual(owner["end_byte"], a["end_byte"])
                self.assertRegex(text, r"\b" + name + (r"\s*;" if kind == "field" else r"\b"))
                if kind == "field":
                    self.assertNotIn("struct TensorStorage", text)
                    self.assertEqual(a["owner_hint"]["syntax_kind"], "field_declaration")
                else:
                    self.assertTrue(text.startswith("struct "))
                self.assertEqual(a["alignment_unit"], "bytes")
                self.assertIn("not_evaluated", a["value_evaluation"])
                seen.add((path, line))
        self.assertEqual(len(seen), 16)
        self.assertEqual(sum(len(self.cute_edits(r)) for r in self.analysis.values()), 16)

    def test_two_definition_branches_and_each_include_edge_are_source_backed(self):
        provider = self.raw[ALIGNMENT]
        for path, result in self.analysis.items():
            for a in result["alignment_attributes"]:
                if a["kind"] != "cute_requested_alignment":
                    continue
                variants = a["conditional_expansions"]
                self.assertEqual(len(variants), 2)
                by_body = {v["definition"]["body"]: v for v in variants}
                self.assertEqual(set(by_body), {"__align__(n)", "alignas(n)"})
                for body, v in by_body.items():
                    definition = v["definition"]
                    self.assertEqual(definition["path"], ALIGNMENT)
                    self.assertEqual(definition["parameters"], ["n"])
                    self.assertEqual(v["definition_source_sha256"], hashlib.sha256(provider).hexdigest())
                    self.assertIn(body, provider[definition["start_byte"]:definition["end_byte"]].decode())
                    self.assertEqual(v["expanded_spelling"], body.replace("(n)", "(" + a["alignment_expression"] + ")"))
                    self.assertEqual(len(v["conditions"]), 1)
                    condition = v["conditions"][0]
                    self.assertIn("defined(__CUDACC__)", condition["expression"])
                    self.assertEqual(condition["directive"], "if" if body == "__align__(n)" else "else")
                    self.assertEqual(condition["expression"].startswith("!"), body == "alignas(n)")
                    span = condition["directive_span"]
                    self.assertEqual(span["path"], ALIGNMENT)
                    self.assertTrue(provider[span["start_byte"]:span["end_byte"]].lstrip().startswith(b"#"))
                chain = a["definition_include_chain"]
                if path == ALIGNMENT:
                    self.assertEqual(chain, [])
                else:
                    self.assertEqual(chain[0]["path"], path)
                    self.assertLess(chain[0]["start_line"], a["start_line"])
                    self.assertEqual(chain[-1]["target_path"], ALIGNMENT)
                    for i, edge in enumerate(chain):
                        text = (ROOT / "snapshot" / edge["path"]).read_text().splitlines()[edge["start_line"] - 1]
                        self.assertIn(edge["target_path"].removeprefix("include/"), text)
                        if i:
                            self.assertEqual(chain[i - 1]["target_path"], edge["path"])

    def test_core_attaches_each_attribute_only_to_its_real_occurrence(self):
        for path, line, expression, kind, name in EXPECTED:
            record, result = self.extracted[path]
            owners = []
            for o in result["occurrences"]:
                for a in o.get("alignment_specifiers", []):
                    if a.get("kind") == "cute_requested_alignment" and a["start_line"] == line:
                        owners.append(o)
                        self.assertEqual(a["owner_occurrence_id"], o["declaration_occurrence_id"])
                        self.assertEqual(a["owner_entity_id"], o["entity_id"])
                        self.assertEqual(a["alignment_expression"].strip(), expression)
                        self.assertIn(name, o["name"])
                        self.assertEqual(o["kind"], "struct_specifier" if kind == "class" else "member")
                        self.assertEqual(o["parse_status"], "parsed")
            with self.subTest(path=path, line=line):
                self.assertTrue(owners)
                header = record["syntax_analysis"]["header_analysis"]
                a = next(a for a in header["alignment_attributes"] if a.get("kind") == "cute_requested_alignment" and a["start_line"] == line)
                self.assertEqual(set(a["owner_occurrence_ids"]), {o["declaration_occurrence_id"] for o in owners})
                self.assertEqual(set(a["owner_entity_ids"]), {o["entity_id"] for o in owners})

    def test_unrelated_fp8_pending_is_not_cleared_by_alignment(self):
        _, result = self.extracted[FP8]
        self.assertTrue(any(d["blocks_phase_1"] and d["start_line"] == 879 for d in result["diagnostics"]))
        self.assertFalse(any(d["category"] == "scope_closure_mismatch" for d in result["diagnostics"]))

    def test_local_undef_redefinition_and_conditional_undef_are_not_fixed_macro_proofs(self):
        undefined = self.analyze_fixture("#undef CUTE_ALIGNAS\nint CUTE_ALIGNAS(int n);\n")
        self.assertFalse(self.cute_edits(undefined))
        self.assertFalse(undefined["alignment_attributes"])
        self.assertTrue(undefined["non_alignment_spellings"])
        for tail in ["#undef CUTE_ALIGNAS\n#define CUTE_ALIGNAS(n) alignas(2*(n))\nstruct CUTE_ALIGNAS(16) X {};",
                     "#if CHANGED\n#undef CUTE_ALIGNAS\n#endif\nstruct CUTE_ALIGNAS(16) X {};",
                     "#define __align__(n) unknown(n)\nstruct CUTE_ALIGNAS(16) X {};"]:
            with self.subTest(tail=tail):
                result = self.analyze_fixture(tail)
                self.assertFalse(self.cute_edits(result))
                self.assertTrue(any(d["category"] == "alignment_macro_override_pending" for d in result["diagnostics"]))

    def test_missing_include_wrong_arity_and_unknown_owner_remain_unresolved(self):
        cases = [("struct CUTE_ALIGNAS(16) X {};", False, "alignment_macro_availability_pending"),
                 ("struct CUTE_ALIGNAS(16,32) X {};", True, "alignment_arguments_pending"),
                 ("struct CUTE_ALIGNAS() X {};", True, "alignment_arguments_pending"),
                 ("int f(){return CUTE_ALIGNAS(16);}", True, "alignment_context_pending"),
                 ("CUTE_ALIGNAS(16) int global_object;", True, "alignment_context_pending")]
        for tail, include, expected in cases:
            with self.subTest(tail=tail):
                result = self.analyze_fixture(tail, include)
                self.assertFalse(self.cute_edits(result))
                self.assertTrue(any(d["category"] == expected for d in result["diagnostics"]))
                self.assertFalse(any(a.get("owner_hint") for a in result["alignment_attributes"]))


if __name__ == "__main__":
    unittest.main()
