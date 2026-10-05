"""Independent namespace counterexamples; failures represent real open findings."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("namespace_review_extractor", ROOT / "scripts/extract_declarations.py")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class NamespaceIndependentReview(unittest.TestCase):
    def extract(self, source):
        result = extractor.Extractor("namespace-independent-review")
        result.extract("fixture.hpp", source.encode())
        return result

    def clang_owner(self, source, owner, prelude=""):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        fixture = prelude + source + "\nstatic_assert(sizeof(" + owner + "::X) == 1);\n"
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=fixture, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def xs(self, result):
        return [o for o in result.occurrences if o["name"] == "X"]

    def test_inactive_undef_does_not_disable_imported_macro(self):
        source = "#if 0\n#undef CUTE_STL_NAMESPACE\n#endif\nnamespace CUTE_STL_NAMESPACE {struct X {}; }"
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.xs(result)}, {"std::X", "cuda::std::X"})
        self.clang_owner(source, "std", "#define CUTE_STL_NAMESPACE std\n")

    def test_undef_in_comment_is_not_a_macro_event(self):
        source = "/*\n#undef CUTE_STL_NAMESPACE\n*/\nnamespace CUTE_STL_NAMESPACE {struct X {}; }"
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.xs(result)}, {"std::X", "cuda::std::X"})
        self.clang_owner(source, "std", "#define CUTE_STL_NAMESPACE std\n")

    def test_line_spliced_undef_takes_effect(self):
        source = "#define API_NS n\n#undef API_\\\nNS\nnamespace API_NS {struct X {}; }"
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.xs(result)}, {"API_NS::X"})
        self.assertFalse(result.diagnostics)
        self.clang_owner(source, "API_NS")

    def test_unknown_conditional_undef_not_false_resolved(self):
        source = "#if FEATURE\n#undef CUTE_STL_NAMESPACE\n#endif\nnamespace CUTE_STL_NAMESPACE {struct X {}; }"
        result = self.extract(source)
        owners = {o["qualified_name"] for o in self.xs(result)}
        self.assertTrue(result.diagnostics or {"CUTE_STL_NAMESPACE::X", "std::X", "cuda::std::X"} <= owners)
        self.clang_owner(source, "std", "#define CUTE_STL_NAMESPACE std\n#define FEATURE 0\n")
        self.clang_owner(source, "CUTE_STL_NAMESPACE", "#define CUTE_STL_NAMESPACE std\n#define FEATURE 1\n")

    def test_define_undef_redefine_uses_current_definition(self):
        source = "#define API_NS first\n#undef API_NS\n#define API_NS second\nnamespace API_NS {struct X {}; }"
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.xs(result)}, {"second::X"})
        self.assertFalse(result.diagnostics)
        self.clang_owner(source, "second")

    def test_macro_component_of_compound_namespace_expands(self):
        source = "#define API_NS a\nnamespace API_NS::b {struct X {}; }"
        self.clang_owner(source, "a::b")
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.xs(result)}, {"a::b::X"})

    def test_compound_cutlass_retains_parameterized_owner(self):
        source = "namespace cutlass::detail {struct X {}; }"
        prelude = "#define CUTLASS_NAMESPACE review\n#define concat_tok(a,b) a ## b\n#define mkcutlassnamespace(pre,ns) concat_tok(pre,ns)\n#define cutlass mkcutlassnamespace(cutlass_,CUTLASS_NAMESPACE)\n"
        self.clang_owner(source, "cutlass_review::detail", prelude)
        result = self.extract(source)
        xs = self.xs(result)
        self.assertTrue(any(o["qualified_name"] == "cutlass::detail::X" for o in xs))
        self.assertTrue(any(o["qualified_name_resolution"] == "parameterized_macro_binding" for o in xs), xs)

    def test_local_alias_to_imported_macro_is_rescanned_or_pending(self):
        source = "#define API_NS CUTE_STL_NAMESPACE\nnamespace API_NS {struct X {}; }"
        self.clang_owner(source, "std", "#define CUTE_STL_NAMESPACE std\n")
        self.clang_owner(source, "cuda::std", "#define CUTE_STL_NAMESPACE cuda::std\n")
        result = self.extract(source)
        owners = {o["qualified_name"] for o in self.xs(result)}
        self.assertTrue(result.diagnostics or owners == {"std::X", "cuda::std::X"}, owners)

    def test_repeated_macro_uses_share_one_configuration(self):
        source = "namespace CUTE_STL_NAMESPACE {namespace CUTE_STL_NAMESPACE {struct X {}; }}"
        self.clang_owner(source, "std::std", "#define CUTE_STL_NAMESPACE std\n")
        self.clang_owner(source, "cuda::std::cuda::std", "#define CUTE_STL_NAMESPACE cuda::std\n")
        result = self.extract(source)
        feasible = [o for o in self.xs(result) if not o.get("condition_feasibility") == "unsatisfiable"
                    and not any(c.get("constant_false") for c in o["preprocessor_conditions"])]
        self.assertEqual({o["qualified_name"] for o in feasible}, {"std::std::X", "cuda::std::cuda::std::X"})


if __name__ == "__main__":
    unittest.main()
