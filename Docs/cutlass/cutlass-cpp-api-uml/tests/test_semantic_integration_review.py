"""Independent follow-up checks of positional recovery and conditional access."""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_integration_review_extractor", ROOT / "scripts/extract_declarations.py")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class SemanticIntegrationIndependentReview(unittest.TestCase):
    def extract(self, source):
        result = extractor.Extractor("semantic-integration-review")
        result.extract("fixture.hpp", source.encode())
        return result

    def clang(self, source, success=True):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)

    def access(self, occurrence, bindings):
        resolution = occurrence["access_resolution"]
        if resolution["kind"] == "unconditional":
            return resolution["value"]
        for case in resolution["ordered_cases"]:
            values = []
            for condition in case["conditions"]:
                expression = condition["expression"]
                expression = re.sub(r"\b[A-Z]\b", lambda m: str(bindings[m[0]]), expression)
                self.assertRegex(expression, r"^[01()!&| \t]+$")
                expression = expression.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
                values.append(bool(eval(expression, {"__builtins__": {}}, {})))
            if all(values):
                return case["access"]
        return resolution["otherwise"]

    def test_same_signature_repeated_type_and_literal_two_projections(self):
        source = 'template<class T, __CUTE_REQUIRES(sizeof(T)>1)> auto f(T value={}) -> decltype((typename T::X{}, typename T::X{}, sizeof(typename T::X), "(typename T::X)"));'
        self.clang("#define __CUTE_REQUIRES(...) int=0\n" + source)
        result = self.extract(source)
        f = next(o for o in result.occurrences if o["name"] == "f")
        expected = 'decltype((typename T::X{}, typename T::X{}, sizeof(typename T::X), "(typename T::X)"))'
        self.assertEqual(f["return_type"], expected)
        self.assertEqual(f["parameters"][0]["default"], "{}")
        identity = result.entities[f["entity_id"]]["identity_signature"]
        self.assertEqual(identity["dependent_return"], extractor.canonical(expected, {"T": "$T0_0"}))
        self.assertNotIn("__codex_parser_", str(result.entities))
        self.assertEqual(f["raw_signature"], source)
        self.assertFalse(result.diagnostics)

    def test_last_active_access_matches_clang_for_four_configurations(self):
        source = "class S {\n#if A\npublic:\n#endif\n#if B\nprivate:\n#endif\nint x; };"
        result = self.extract(source)
        x = next(o for o in result.occurrences if o["name"] == "x")
        for a in (0, 1):
            for b in (0, 1):
                expected = "public" if a and not b else "private"
                self.assertEqual(self.access(x, {"A": a, "B": b}), expected)
                self.clang(f"#define A {a}\n#define B {b}\n" + source + "\nint use(S& s){return s.x;}", success=expected == "public")
        self.assertFalse(result.diagnostics)

    def test_conditional_access_of_real_macro_member_matches_clang(self):
        source = "#define FIELD(N) int N;\nstruct S {\n#if A\nprivate:\n#else\npublic:\n#endif\nFIELD(x)};"
        result = self.extract(source)
        x = next(o for o in result.occurrences if o["name"] == "x")
        self.assertIn("macro_origin", x)
        self.assertEqual(x["declared_type"], "int")
        for a in (0, 1):
            expected = "private" if a else "public"
            self.assertEqual(self.access(x, {"A": a}), expected)
            self.clang(f"#define A {a}\n" + source + "\nint use(S& s){return s.x;}", success=not a)
        self.assertFalse(result.diagnostics)

    def test_friend_callable_does_not_inherit_conditional_private_access(self):
        source = "namespace n { struct S {\n#if A\nprivate:\n#else\nprotected:\n#endif\nfriend int f(S const&); }; int f(S const& x){return 1;} }"
        result = self.extract(source)
        fs = [o for o in result.occurrences if o["name"] == "f"]
        self.assertEqual(len(fs), 2)
        self.assertEqual(len({o["entity_id"] for o in fs}), 1)
        self.assertEqual({o["access"] for o in fs}, {"public"})
        for a in (0, 1):
            self.clang(f"#define A {a}\n" + source + "\nint use(n::S const& s){return n::f(s);}")
        self.assertFalse(result.diagnostics)

    def test_friend_type_forward_occurrence_is_not_silently_lost(self):
        source = "namespace n { struct S {private: friend struct F; }; struct F {}; }"
        self.clang(source + "\nstatic_assert(sizeof(n::F)==1);")
        result = self.extract(source)
        fs = [o for o in result.occurrences if o["name"] == "F"]
        self.assertEqual(len(fs), 2, fs)
        self.assertEqual({o["qualified_name"] for o in fs}, {"n::F"})
        self.assertEqual(len({o["entity_id"] for o in fs}), 1)

    def test_two_projection_expanded_parameter_text_has_no_parser_identifier(self):
        source = "template<class T, __CUTE_REQUIRES(sizeof(T)>1)> void f(T value={});"
        result = self.extract(source)
        f = next(o for o in result.occurrences if o["name"] == "f")
        parameter = f["parameters"][0]
        self.assertNotIn("__codex_parser_", parameter.get("expanded_raw", parameter["raw"]))


if __name__ == "__main__":
    unittest.main()
