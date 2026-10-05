"""Independent Clang CPP oracles for macro paths currently claiming success.

Known-pending expansion capabilities are not required to become implemented by
these tests. A conservative explicit diagnostic is acceptable where asserted;
silently publishing a different successful expansion is not.
"""
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "review_macro_expansion", ROOT / "scripts/macro_expansion.py"
)
macro = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = macro
SPEC.loader.exec_module(macro)
CLANG = shutil.which("clang++")


def tokens(source):
    # Whitespace outside literals is immaterial; inside literals it is evidence.
    return re.findall(r'"(?:\\.|[^"\\])*"|[A-Za-z_]\w*|\S', source)


@unittest.skipUnless(CLANG, "clang++ is required for the independent CPP oracle")
class MacroOracleReviewTests(unittest.TestCase):
    def oracle(self, source):
        result = subprocess.run(
            [CLANG, "-E", "-P", "-x", "c++", "-"],
            input=source, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def assert_success_matches_or_is_explicitly_pending(self, source):
        expected = self.oracle(source)
        result = macro.expand_file("include/cute/review_macro.hpp", source.encode())
        if not result["expansions"] and result["diagnostics"]:
            return  # Pending is honest; these tests target false successful output.
        self.assertEqual(
            tokens("".join(e["virtual_source"] for e in result["expansions"])),
            tokens(expected),
            "A successful declaration macro expansion differs from Clang CPP",
        )

    def test_argument_macro_prescan_cannot_silently_change_entity_name(self):
        self.assert_success_matches_or_is_explicitly_pending(
            "#define NAME Actual\n"
            "#define MAKE(X) struct X {};\n"
            "MAKE(NAME)\n"
        )

    def test_token_paste_result_requires_rescan_or_pending(self):
        self.assert_success_matches_or_is_explicitly_pending(
            "#define A_type Actual\n"
            "#define MAKE(X) struct X ## _type {};\n"
            "MAKE(A)\n"
        )

    def test_spliced_line_comment_must_not_generate_fake_api(self):
        source = "#define MAKE(X) struct X {};\n// disabled \\\nMAKE(Fake)\n"
        self.assertEqual(tokens(self.oracle(source)), [])
        result = macro.expand_file("include/cute/review_macro.hpp", source.encode())
        self.assertFalse(result["expansions"], "A commented invocation generated a fake struct")

    def test_spliced_undef_must_not_reuse_dead_macro(self):
        source = "#define MAKE(X) struct X {};\n#un\\\ndef MAKE\nMAKE(Fake)\n"
        self.assertEqual(tokens(self.oracle(source)), ["MAKE", "(", "Fake", ")"])
        result = macro.expand_file("include/cute/review_macro.hpp", source.encode())
        self.assertFalse(result["expansions"], "A macro expanded after its spliced #undef")

    def test_stringification_collapses_comment_adjacent_whitespace(self):
        self.assert_success_matches_or_is_explicitly_pending(
            "#define MAKE(X) struct Y { const char* s = #X; };\n"
            "MAKE(a /**/ b)\n"
        )

    def test_existing_explicit_undef_pending_is_not_a_new_failure(self):
        source = "#define MAKE(X) struct X {};\n#undef MAKE\nMAKE(Fake)\n"
        self.assertEqual(tokens(self.oracle(source)), ["MAKE", "(", "Fake", ")"])
        result = macro.expand_file("include/cute/review_macro.hpp", source.encode())
        self.assertFalse(result["expansions"])
        self.assertTrue(any(
            d["kind"] == "macro_definition_lifetime_unresolved"
            for d in result["diagnostics"]
        ))

    def test_literal_token_paste_text_is_preserved(self):
        self.assert_success_matches_or_is_explicitly_pending(
            '#define MAKE(X) struct X { const char* s = " ## "; };\n'
            "MAKE(Real)\n"
        )


if __name__ == "__main__":
    unittest.main()
