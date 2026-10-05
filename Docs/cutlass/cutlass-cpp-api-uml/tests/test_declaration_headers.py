"""Independent header/alignment fixtures with Clang and exact source mapping."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import unittest

from tree_sitter import Language, Parser
import tree_sitter_cpp

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("declaration_headers", ROOT / "scripts/declaration_headers.py")
headers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(headers)


class DeclarationHeadersTests(unittest.TestCase):
    def analyze(self, source):
        return headers.analyze_headers("fixture.hpp", source.encode())

    def clang(self, source, success=True):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)

    def assert_pieces(self, source, variant):
        pieces = b"".join(source[s["physical_span"]["start_byte"]:s["physical_span"]["end_byte"]] for s in variant["segments"])
        self.assertEqual(pieces.decode(), variant["virtual_source"])
        span = variant["signature_span"]
        self.assertEqual(source[span["start_byte"]:span["end_byte"]].decode(), variant["semantic_spelling"])
        for parameter in variant["signature"].get("parameters", []):
            parameter_bytes = b"".join(source[s["start_byte"]:s["end_byte"]] for s in parameter["source_segments"])
            self.assertEqual(parameter_bytes.decode(), parameter["raw"])

    def test_named_and_anonymous_parameter_header_variants(self):
        source = "struct S {static inline int\n#if FEATURE\ninit(void const* p)\n#else\ninit(void const* /* p */)\n#endif\n{\n#if FEATURE\nreturn p?1:0;\n#else\nreturn 0;\n#endif\n}};"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual([v["signature"]["parameters"][0]["name"] for v in variants], ["p", None])
        self.assertEqual({v["signature"]["parameters"][0]["type"] for v in variants}, {"void const*"})
        self.assertEqual({v["signature"]["return_type"] for v in variants}, {"int"})
        self.assertEqual(len({v["source_declaration_id"] for v in variants}), 1)
        self.assertEqual(len({v["conditional_variant_id"] for v in variants}), 2)
        self.assertIn("/* p */", variants[1]["virtual_signature"])
        for index, variant in enumerate(variants):
            self.assertEqual(variant["status"], "header_parsed")
            self.assertEqual(variant["body_status"], "retained_for_core_extraction_all_source_branches")
            self.assert_pieces(source.encode(), variant)
            self.clang(f"#define FEATURE {1-index}\nstruct Probe {{" + variant["virtual_source"] + "};")
        self.assertFalse(result["diagnostics"])

    def test_fixed_cluster_init_two_headers_full_body_not_discarded(self):
        path = "include/cutlass/cluster_launch.hpp"
        source = (ROOT / "snapshot" / path).read_bytes()
        result = headers.analyze_headers(path, source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual({v["signature"]["name"] for v in variants}, {"init"})
        self.assertEqual([v["signature"]["parameters"][0]["name"] for v in variants], ["kernel_function", None])
        self.assertEqual({v["signature"]["parameters"][0]["type"] for v in variants}, {"void const*"})
        self.assertTrue(all("CUTLASS_HOST" in v["signature"]["attributes"] for v in variants))
        self.assertEqual(len({(v["body_span"]["start_byte"], v["body_span"]["end_byte"]) for v in variants}), 1)
        for variant in variants:
            self.assert_pieces(source, variant)
            self.assertIn("cudaFuncSetAttribute", variant["virtual_source"])
            self.assertIn("#else", variant["virtual_source"])
            self.assertTrue(variant["conditions"])
            self.clang("#define CUTLASS_HOST\nenum class Status { kSuccess,kInvalid }; struct Probe {" + variant["virtual_signature"] + ";};")
        self.assertFalse(result["diagnostics"])

    def test_if_zero_and_elif_headers_are_all_retained(self):
        source = "int\n#if 0\nf(int x)\n#elif A\nf(long x)\n#else\nf(char)\n#endif\n{return 0;}"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 3)
        self.assertEqual([v["signature"]["parameters"][0]["type"] for v in variants], ["int", "long", "char"])
        self.assertTrue(any(c["constant_false"] for c in variants[0]["conditions"]))
        self.assertFalse(result["diagnostics"])

    def test_entire_header_is_conditional_without_shared_prefix(self):
        source = "#if A\nint f(int value)\n#else\nlong g(long)\n#endif\n{return 0;}"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual([v["signature"]["name"] for v in variants], ["f", "g"])
        self.assertEqual([v["signature"]["return_type"] for v in variants], ["int", "long"])
        self.assertEqual(variants[0]["declaration_span"]["start_byte"], 0)
        self.assertFalse(result["diagnostics"])

    def test_braced_defaults_not_silently_skipped_as_function_bodies(self):
        source = "int\n#if A\nf(int value={})\n#else\nf(long value={})\n#endif\n{return 0;}"
        result = self.analyze(source)
        self.assertEqual(len(result["conditional_declaration_variants"]), 2)
        self.assertTrue(result["diagnostics"])
        self.assertTrue(all(v["status"] == "header_parse_pending" for v in result["conditional_declaration_variants"]))
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)

    def test_branch_local_terminators_after_shared_prefix_report_gap(self):
        source = "static int\n#if A\nf(int);\n#else\nf(long);\n#endif\n"
        result = self.analyze(source)
        self.assertFalse(result["conditional_declaration_variants"])
        self.assertTrue(any(d["category"] == "conditional_function_branch_boundary_pending" for d in result["diagnostics"]))

    def test_parameter_subsection_and_prototype(self):
        source = "int f(\n#if A\nint x\n#else\nlong\n#endif\n);"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual([v["signature"]["parameters"][0]["name"] for v in variants], ["x", None])
        self.assertTrue(all(v["body_span"] is None for v in variants))
        self.assertFalse(result["diagnostics"])
        for variant in variants:
            self.clang(variant["virtual_source"])

    def test_template_prefix_and_cv_ref_are_retained(self):
        source = "struct S {template<class T> T\n#if A\nf(T const& x) const &\n#else\nf(T&& x) &&\n#endif\n{return T{};}};"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertTrue(all(v["signature"]["template_parameters_raw"] == "<class T>" for v in variants))
        self.assertIn("const", variants[0]["signature"]["qualifiers"])
        self.assertIn("&", variants[0]["signature"]["qualifiers"])
        self.assertIn("&&", variants[1]["signature"]["qualifiers"])
        for variant in variants:
            self.clang("struct Probe {" + variant["virtual_source"] + "};")
        self.assertFalse(result["diagnostics"])

    def test_utf8_offsets_and_parameter_comment_do_not_become_name(self):
        source = "// 中文\nint\n#if A\nf(int /* anonymous */)\n#else\nf(int named)\n#endif\n{return 0;}"
        result = self.analyze(source)
        self.assertEqual(result["conditional_declaration_variants"][0]["declaration_span"]["start_byte"], len("// 中文\n".encode()))
        self.assertIsNone(result["conditional_declaration_variants"][0]["signature"]["parameters"][0]["name"])
        for variant in result["conditional_declaration_variants"]:
            self.assert_pieces(source.encode(), variant)

    def test_constructor_initializer_is_an_explicit_gap(self):
        source = "struct S {int value;\n#if A\nS(int x)\n#else\nS(long x)\n#endif\n:value(x) {}};"
        result = self.analyze(source)
        self.assertFalse(result["conditional_declaration_variants"])
        self.assertTrue(any(d["category"] == "conditional_function_header_pending" for d in result["diagnostics"]))
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)

    def test_existing_conditional_aliases_and_complete_functions_not_reclassified(self):
        source = "using X=\n#if A\ndecltype(1)\n#else\ndecltype(2L)\n#endif\n;\n#if A\nint f(){return 1;}\n#else\nlong f(){return 2;}\n#endif\n"
        result = self.analyze(source)
        self.assertFalse(result["conditional_declaration_variants"])
        self.assertFalse(result["diagnostics"])

    def test_condition_inside_inline_assembly_is_not_a_header(self):
        source = 'void f(){asm volatile(\n#if A\n""\n#else\n""\n#endif\n: : "r"(1));}'
        result = self.analyze(source)
        self.assertFalse(result["conditional_declaration_variants"])
        self.assertFalse(result["diagnostics"])
        self.clang("#define A 1\n" + source)

    def test_fixed_alignment_specializations_13_attributes_and_layout(self):
        path = "include/cutlass/platform/platform.h"
        source = (ROOT / "snapshot" / path).read_bytes()
        result = headers.analyze_headers(path, source)
        self.assertEqual(len(result["projection_edits"]), 13)
        self.assertEqual([int(a["alignment_expression"]) for a in result["alignment_attributes"]], [2**i for i in range(13)])
        for attribute in result["alignment_attributes"]:
            span = attribute["expression_span"]
            self.assertEqual(source[span["start_byte"]:span["end_byte"]].decode(), attribute["alignment_expression"])
            self.assertTrue(attribute["conditions"])
        self.assertFalse(result["diagnostics"])
        # Compiler cross-check uses the GNU expansion actually used by CUDA's
        # current host_defines.h on GNU/Linux and in CUDA RTC mode. This is a
        # reduced layout check, not a proof for all compiler/target branches.
        declarations = "template<int N> struct aligned_chunk;\n"
        for value in [2**i for i in range(13)]:
            declarations += f"template<> struct __align__({value}) aligned_chunk<{value}>{{char data[{value}];}};\n"
            declarations += f"static_assert(alignof(aligned_chunk<{value}>)=={value});\n"
        reduced = self.analyze(declarations)
        projected = headers._syntax.project_for_parser(declarations.encode(), reduced["projection_edits"]).decode()
        self.clang("#define __align__(n) __attribute__((aligned(n)))\n" + declarations)
        self.clang(projected)
        self.assertFalse(Parser(Language(tree_sitter_cpp.language())).parse(projected.encode()).root_node.has_error)

    def test_alignment_expression_not_evaluated_or_discarded(self):
        source = "template<int N> struct __align__((N << 1)) A {char data;};"
        result = self.analyze(source)
        self.assertEqual(result["alignment_attributes"][0]["alignment_expression"], "(N << 1)")
        self.assertEqual(result["projection_edits"][0]["semantic_spelling"], "__align__((N << 1))")
        self.assertIn("not_evaluated", result["alignment_attributes"][0]["value_evaluation"])
        projected = headers._syntax.project_for_parser(source.encode(), result["projection_edits"]).decode()
        self.clang(projected + "\nstatic_assert(alignof(A<16>)==32);")

    def test_alignment_does_not_weaken_natural_alignment_semantics(self):
        source = "struct __align__(1) A {double value;}; static_assert(alignof(A)==alignof(double));"
        result = self.analyze(source)
        projected = headers._syntax.project_for_parser(source.encode(), result["projection_edits"]).decode()
        self.clang("#define __align__(n) __attribute__((aligned(n)))\n" + source)
        self.clang(projected)
        self.assertNotIn("alignas", projected)

    def test_alignment_macro_override_not_assumed_builtin(self):
        source = "#define __align__(N)\nstruct __align__(16) A {};"
        result = self.analyze(source)
        self.assertFalse(result["projection_edits"])
        self.assertTrue(any(d["category"] == "alignment_macro_override_pending" for d in result["diagnostics"]))
        self.assertEqual(len(result["alignment_attributes"]), 1)

    def test_alignment_comments_strings_nonclass_and_arity(self):
        source = '// struct __align__(64) Fake;\nconst char* text="__align__(32)";\n__align__(16) int value;\nstruct __align__(16,32) Bad {};'
        result = self.analyze(source)
        self.assertEqual(len(result["alignment_attributes"]), 2)
        self.assertFalse(result["projection_edits"])
        self.assertEqual({d["category"] for d in result["diagnostics"]}, {"alignment_context_pending", "alignment_arguments_pending"})


if __name__ == "__main__":
    unittest.main()
