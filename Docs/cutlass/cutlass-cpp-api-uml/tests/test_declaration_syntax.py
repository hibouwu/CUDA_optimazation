import importlib.util
from pathlib import Path
import shutil
import subprocess
import unittest

from tree_sitter import Language, Parser
import tree_sitter_cpp

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("declaration_syntax", PROJECT / "scripts/declaration_syntax.py")
syntax = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(syntax)


class DeclarationSyntaxTests(unittest.TestCase):
    def parse(self, source):
        return Parser(Language(tree_sitter_cpp.language())).parse(source)

    def analyze(self, source, **kwargs):
        return syntax.analyze_syntax("include/cute/fixture.hpp", source.encode(), **kwargs)

    def clang(self, source, standard="c++17", pedantic=True):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ not available for independent syntax check")
        options = [compiler, "-std=" + standard, "-fsyntax-only", "-x", "c++", "-"]
        if pedantic:
            options.append("-pedantic-errors")
        result = subprocess.run(options, input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_conditional_alias_preserves_every_type_and_span(self):
        source = "using X =\n#if MODE\nint\n#else\nlong\n#endif\n;"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertEqual({v["target_type"].strip() for v in variants}, {"int", "long"})
        self.assertTrue(all(v["status"] == "parsed" for v in variants))
        self.assertFalse(result["diagnostics"])
        for variant in variants:
            reconstructed = b"".join(source.encode()[s["physical_span"]["start_byte"]:s["physical_span"]["end_byte"]] for s in variant["segments"])
            self.assertEqual(reconstructed.decode(), variant["virtual_source"])
            self.clang(variant["virtual_source"])
        self.assertEqual(len({v["source_declaration_id"] for v in variants}), 1)

    def test_nested_conditional_alias_and_if_zero_kept(self):
        source = "#if OUTER\nusing X =\n#if 0\nint\n#elif A\n#if B\nlong\n#else\nshort\n#endif\n#else\nchar\n#endif\n;\n#endif\n"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 4)
        self.assertTrue(all(v["conditions"][0]["expression"] == "OUTER" for v in variants))
        self.assertTrue(any(any(c["constant_false"] for c in v["conditions"]) for v in variants))
        self.assertFalse(result["diagnostics"])

    def test_missing_else_empty_type_is_explicit_error(self):
        result = self.analyze("using X =\n#if A\nint\n#endif\n;")
        self.assertEqual(len(result["conditional_declaration_variants"]), 2)
        self.assertTrue(result["diagnostics"])
        self.assertEqual({v["status"] for v in result["conditional_declaration_variants"]}, {"parsed", "parse_pending"})

    def test_dependent_type_construct_keeps_typename_and_initializer(self):
        source = "struct S {using X=int;}; template<class T> constexpr auto f() {return typename T::X{};} static_assert(f<S>()==0);"
        result = self.analyze(source)
        edits = [e for e in result["projection_edits"] if e["kind"] == "dependent_type_braced_construction"]
        self.assertEqual(len(edits), 1)
        self.assertEqual(edits[0]["semantic_spelling"], "typename T::X")
        projected = syntax.project_for_parser(source.encode(), edits)
        self.assertFalse(self.parse(projected).root_node.has_error)
        self.assertIn(b"(typename T::X){}", projected)
        self.clang(source)  # Original is valid strict ISO C++17.
        self.clang(projected.decode(), standard="gnu++17", pedantic=False)

    def test_dependent_variable_declaration_not_treated_as_temporary(self):
        result = self.analyze("template<class T> void f(){ typename T::X value{}; }")
        self.assertFalse(any(e["kind"] == "dependent_type_braced_construction" for e in result["projection_edits"]))

    def test_braced_default_restoration_contract_and_strict_original(self):
        source = "template<class T> void f(T const& value = {}); template<class T> void g(T&& value = {});"
        result = self.analyze(source)
        edits = [e for e in result["projection_edits"] if e["kind"] == "braced_parameter_default"]
        self.assertEqual(len(edits), 2)
        self.assertTrue(all(e["semantic_spelling"] == "{}" and e["semantic_node"]["kind"] == "initializer_list" for e in edits))
        self.assertFalse(self.parse(syntax.project_for_parser(source.encode(), edits)).root_node.has_error)
        self.clang(source)

    def test_comments_strings_and_nonparameter_initializers_untouched(self):
        source = '// typename T::X{}\nconst char* text="typename T::X{}"; int array[] = {}; struct S {int a = {};};'
        result = self.analyze(source)
        self.assertFalse(result["projection_edits"])

    def test_keyword_macro_has_both_definition_variants(self):
        source = "template<int N> int f(){if CUTLASS_CONSTEXPR_IF_CXX17 (N) return N; return 0;}"
        result = self.analyze(source)
        edits = result["projection_edits"]
        self.assertEqual(len(edits), 1)
        self.assertEqual({v["expanded"] for v in edits[0]["variants"]}, {"constexpr", ""})
        self.assertTrue(all(v["conditions"] for v in edits[0]["variants"]))
        self.assertFalse(self.parse(syntax.project_for_parser(source.encode(), edits)).root_node.has_error)
        self.clang("#define CUTLASS_CONSTEXPR_IF_CXX17 constexpr\n" + source)
        self.clang("#define CUTLASS_CONSTEXPR_IF_CXX17\n" + source, standard="c++14")

    def test_missing_macro_definition_never_guessed(self):
        result = self.analyze("int f(){if CUTLASS_CONSTEXPR_IF_CXX17 (true) return 1; return 0;}", macro_sources={})
        self.assertFalse(result["projection_edits"])
        self.assertTrue(result["diagnostics"])

    def test_fixed_exmy_alias_variants_and_keyword_count(self):
        path = "include/cutlass/exmy_base.h"
        source = (PROJECT / "snapshot" / path).read_bytes()
        result = syntax.analyze_syntax(path, source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 4)
        self.assertEqual({v["name"] for v in variants}, {"BitRepresentation", "FP32BitRepresentation"})
        self.assertTrue(all(v["status"] == "parsed" for v in variants))
        self.assertGreaterEqual(sum(e["kind"] == "conditional_keyword_macro" for e in result["projection_edits"]), 35)
        self.assertFalse(result["diagnostics"])

    def test_fixed_exmy_variants_cross_checked_by_clang(self):
        path = "include/cutlass/exmy_base.h"
        result = syntax.analyze_syntax(path, (PROJECT / "snapshot" / path).read_bytes())
        prelude = '''
namespace cutlass { namespace detail {
enum class FpEncoding { E8M23 };
template<FpEncoding> struct Representation {};
template<FpEncoding E> Representation<E> fp_encoding_selector();
template<FpEncoding E> struct FpEncodingSelector { using type = Representation<E>; };
} }
using namespace cutlass;
'''
        for variant in result["conditional_declaration_variants"]:
            source = prelude + "template<detail::FpEncoding T> struct Check {\n" + variant["virtual_source"] + "\n};\n"
            self.clang(source)

    def test_fixed_axpby_braced_signature_defaults(self):
        path = "include/cute/algorithm/axpby.hpp"
        source = (PROJECT / "snapshot" / path).read_bytes()
        result = syntax.analyze_syntax(path, source)
        edits = [e for e in result["projection_edits"] if e["kind"] == "braced_parameter_default"]
        self.assertEqual(len(edits), 2)
        self.assertEqual([e["start_line"] for e in edits], [54, 73])

    def test_multiline_directive_utf8_and_exact_variant_segments(self):
        source = "// 中文位置\nusing X =\n#if A && \\\n B\nint\n#else\nlong\n#endif\n;"
        result = self.analyze(source)
        variants = result["conditional_declaration_variants"]
        self.assertEqual(len(variants), 2)
        self.assertTrue(all(v["status"] == "parsed" for v in variants))
        self.assertIn("B", variants[0]["conditions"][0]["expression"])
        self.assertEqual(variants[0]["declaration_span"]["start_byte"], len("// 中文位置\n".encode()))
        self.assertFalse(result["diagnostics"])

    def test_annotated_constructor_unnamed_braced_defaults(self):
        source = "template<class T> struct S { CUTLASS_DEVICE S(int a,T = {},T = {}) : x(a) {} int x; };"
        result = self.analyze(source)
        defaults = [e for e in result["projection_edits"] if e["kind"] == "braced_parameter_default"]
        self.assertEqual(len(defaults), 2)

    def test_fixed_sm100_all_unnamed_defaults_found(self):
        path = "include/cutlass/pipeline/sm100_pipeline.hpp"
        source = (PROJECT / "snapshot" / path).read_bytes()
        result = syntax.analyze_syntax(path, source)
        lines = [e["start_line"] for e in result["projection_edits"] if e["kind"] == "braced_parameter_default"]
        self.assertEqual(lines.count(327), 2)
        self.assertEqual(lines.count(642), 2)

    def test_typename_inside_base_template_does_not_swallow_class_body(self):
        source = "template<class P> struct is_gmem<P, void_t<typename P::iterator>> : is_gmem<typename P::iterator> {};"
        result = self.analyze(source)
        self.assertFalse(result["projection_edits"])
        self.assertFalse(result["diagnostics"])

    def test_direct_initializer_assignment_is_not_parameter_default(self):
        source = "struct S {S(int);}; int y; S x(y = {});"
        result = self.analyze(source)
        self.assertFalse(result["projection_edits"])
        self.clang(source)


if __name__ == "__main__":
    unittest.main()
