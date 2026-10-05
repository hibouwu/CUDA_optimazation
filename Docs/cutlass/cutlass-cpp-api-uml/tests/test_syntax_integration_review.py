"""Independent semantic-field/source-provenance checks for syntax integration."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("syntax_integration_review_extractor", ROOT / "scripts/extract_declarations.py")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class SyntaxIntegrationIndependentReview(unittest.TestCase):
    def extract(self, source):
        result = extractor.Extractor("syntax-integration-independent-review")
        result.extract("fixture.hpp", source.encode())
        return result

    def clang(self, source):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_braced_default_restored_not_temporary_identifier(self):
        source = "template<class T> void f(T const& value={});"
        self.clang(source)
        result = self.extract(source)
        f = next(o for o in result.occurrences if o["name"] == "f")
        self.assertEqual(f["parameters"][0]["default"], "{}")
        self.assertEqual(f["parameters"][0]["name"], "value")
        self.assertEqual(f["raw_signature"], source)
        self.assertNotIn("__codex_parser_", f["expanded_signature"])
        self.assertFalse(result.diagnostics)

    def test_constraint_and_default_projection_compose_physical_spans(self):
        source = "template<class T,__CUTE_REQUIRES(sizeof(T)>1)> void f(T value={});"
        result = self.extract(source)
        f = next(o for o in result.occurrences if o["name"] == "f")
        self.assertIn("enable_if", f["expanded_signature"])
        self.assertEqual(f["parameters"][0]["default"], "{}")
        self.assertNotIn("__codex_parser_", f["expanded_signature"])
        for occurrence in result.occurrences:
            span = occurrence["signature_range"]
            self.assertEqual(source.encode()[span["start_byte"]:span["end_byte"]].decode().rstrip(), occurrence["raw_signature"])
        self.assertFalse(result.diagnostics)

    def test_conditional_aliases_share_entity_and_have_distinct_variants(self):
        source = "struct S {\nusing X=\n#if A\nint\n#else\nlong\n#endif\n;};"
        result = self.extract(source)
        aliases = [o for o in result.occurrences if o["name"] == "X"]
        self.assertEqual(len(aliases), 2)
        self.assertEqual(len({o["entity_id"] for o in aliases}), 1)
        self.assertEqual(len({o["variant_id"] for o in aliases}), 2)
        self.assertEqual({o["target_type"] for o in aliases}, {"int", "long"})
        self.assertEqual(len({o["conditional_declaration_origin"]["source_declaration_id"] for o in aliases}), 1)
        for alias in aliases:
            origin = alias["conditional_declaration_origin"]
            pieces = b"".join(source.encode()[s["physical_span"]["start_byte"]:s["physical_span"]["end_byte"]] for s in origin["segments"])
            self.assertEqual(pieces.decode(), origin["virtual_source"])
            span = alias["signature_range"]
            self.assertEqual(source.encode()[span["start_byte"]:span["end_byte"]].decode().rstrip(), alias["raw_signature"])
        self.assertFalse(result.diagnostics)

    def test_conditional_keyword_not_unconditional_constexpr(self):
        source = "template<class T> CUTLASS_CONSTEXPR_IF_CXX17 T f(T value={});"
        result = self.extract(source)
        f = next(o for o in result.occurrences if o["name"] == "f")
        self.assertNotIn("constexpr", f["qualifiers"])
        self.assertEqual({v["expanded"] for v in f["conditional_specifiers"][0]["variants"]}, {"constexpr", ""})
        self.assertIn("CUTLASS_CONSTEXPR_IF_CXX17", f["expanded_signature"])
        self.assertFalse(result.diagnostics)

    def test_restoration_cannot_rewrite_unrelated_sizeof_return_type(self):
        source = "template<class T> auto f()->decltype(typename T::X{});\ntemplate<class T> auto g()->decltype(sizeof(typename T::X));"
        self.clang(source)
        result = self.extract(source)
        g = next(o for o in result.occurrences if o["name"] == "g")
        self.assertEqual(g["return_type"], "decltype(sizeof(typename T::X))")

    def test_restoration_cannot_rewrite_string_literal(self):
        source = 'template<class T> auto f()->decltype(typename T::X{});\nconst char* text="(typename T::X)";'
        self.clang(source)
        result = self.extract(source)
        constant = next(o for o in result.occurrences if o["name"] == "text")
        self.assertEqual(constant["initializer"], '"(typename T::X)"')

    def test_conditional_alias_restores_private_access(self):
        source = "struct S {private:\nusing X=\n#if A\nint\n#else\nlong\n#endif\n;};"
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)
        result = self.extract(source)
        aliases = [o for o in result.occurrences if o["name"] == "X"]
        self.assertEqual(len(aliases), 2)
        self.assertEqual({o["access"] for o in aliases}, {"private"})

    def test_generated_type_restores_private_access(self):
        source = "#define MAKE(X) struct X {};\nstruct S {private: MAKE(A)};"
        self.clang(source)
        result = self.extract(source)
        a = next(o for o in result.occurrences if o["name"] == "A")
        self.assertIn("macro_origin", a)
        self.assertEqual(a["access"], "private")

    def test_simple_member_macro_is_expanded_or_explicit_pending(self):
        source = "#define MAKE(X) int X;\nstruct S {private: MAKE(x)};"
        self.clang(source)
        result = self.extract(source)
        members = [o for o in result.occurrences if o["name"] == "x"]
        self.assertTrue(result.diagnostics or any(o.get("declared_type") == "int" and "macro_origin" in o for o in members), members)

    def test_unnamed_missing_punctuation_is_a_real_diagnostic(self):
        for source in ("struct S { int x };", "void f(int x;", "int x"):
            with self.subTest(source=source):
                result = self.extract(source)
                self.assertTrue(result.files[0]["normalized_parse_has_error"])
                self.assertTrue(any(d["category"] == "missing_syntax" and d["blocks_phase_1"] for d in result.diagnostics), result.diagnostics)


if __name__ == "__main__":
    unittest.main()
