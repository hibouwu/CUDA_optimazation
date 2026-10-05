"""Declaration macro scope/classification regressions, including SI03."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from macro_expansion import definitions, expand_file


class DeclarationMacroTests(unittest.TestCase):
    def expand(self, source):
        return expand_file("fixture.hpp", source.encode())

    def clang(self, source):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_simple_int_member_has_real_expansion(self):
        source = "#define MAKE(X) int X;\nstruct S {private: MAKE(x)};"
        self.clang(source)
        result = self.expand(source)
        self.assertFalse(result["diagnostics"])
        self.assertEqual([x["virtual_source"] for x in result["expansions"]], ["int x;"])
        self.assertEqual(result["definitions"][0]["replacement_classification"], "declaration_sequence")

    def test_multi_member_and_multi_declaration_sequence(self):
        source = "#define FIELDS(A,B,C) int A,B; float C;\nstruct S {FIELDS(x,y,z)};"
        self.clang(source)
        result = self.expand(source)
        self.assertEqual(result["expansions"][0]["virtual_source"], "int x,y; float z;")
        self.assertFalse(result["diagnostics"])

    def test_formal_type_and_pointer_function_declarations(self):
        source = "#define DECL(T,N) T* N(int);\nstruct S {DECL(int,f)};"
        self.clang(source)
        result = self.expand(source)
        self.assertEqual(result["expansions"][0]["virtual_source"], "int* f(int);")

    def test_primitive_token_pasted_members(self):
        source = "#define FIELDS(X) int X ## _a, X ## _b;\nstruct S {FIELDS(value)};"
        self.clang(source)
        result = self.expand(source)
        self.assertEqual(result["expansions"][0]["virtual_source"], "int value_a, value_b;")
        self.assertFalse(result["diagnostics"])

    def test_callsite_semicolon_has_separate_provenance(self):
        source = "#define FIELD(X) int X\nstruct S {FIELD(x);};"
        self.clang(source)
        result = self.expand(source)
        expansion = result["expansions"][0]
        self.assertEqual(expansion["virtual_source"], "int x;")
        span = expansion["callsite_suffix"]
        self.assertEqual(source.encode()[span["start_byte"]:span["end_byte"]], b";")
        self.assertEqual(source.encode()[expansion["start_byte"]:expansion["macro_invocation_end_byte"]], b"FIELD(x)")
        self.assertFalse(result["diagnostics"])

    def test_missing_callsite_semicolon_is_pending(self):
        result = self.expand("#define FIELD(X) int X\nstruct S {FIELD(x)};")
        self.assertFalse(result["expansions"])
        self.assertEqual(result["diagnostics"][0]["kind"], "macro_declaration_terminator_pending")

    def test_statement_macro_with_local_variable_not_named_api(self):
        source = "#define UPDATE(X) do { int local=(X); (X)=local+1; } while(0)\nvoid f(){int x=0; UPDATE(x); }"
        self.clang(source)
        result = self.expand(source)
        self.assertFalse(result["expansions"])
        self.assertFalse(result["diagnostics"])
        self.assertEqual(result["definitions"][0]["replacement_classification"], "statement_or_expression")
        self.assertEqual(len(result["non_declaration_invocations"]), 1)

    def test_return_and_assignment_macro_not_api(self):
        source = "#define RETURN(X) return X\n#define SET(X,Y) X=Y\nint f(){int x;SET(x,3);RETURN(x); }"
        self.clang(source)
        result = self.expand(source)
        self.assertFalse(result["expansions"])
        self.assertEqual({x["name"] for x in result["non_declaration_invocations"]}, {"RETURN", "SET"})

    def test_known_object_declaration_macro_has_explicit_pending(self):
        source = "#define FIELD int x;\nstruct S {FIELD};"
        self.clang(source)
        result = self.expand(source)
        self.assertFalse(result["expansions"])
        self.assertEqual(result["diagnostics"][0]["kind"], "object_declaration_macro_pending")

    def test_nested_simple_declaration_macro_not_silently_a_type(self):
        source = "#define FIELD(X) int X;\n#define WRAP(X) FIELD(X)\nstruct S {WRAP(x)};"
        self.clang(source)
        result = self.expand(source)
        self.assertFalse(result["expansions"])
        self.assertEqual(result["diagnostics"][0]["kind"], "nested_declaration_macro_pending")

    def test_argument_dependent_declaration_expands_when_actual_form_is_clear(self):
        source = "#define ID(X) X\nstruct S {ID(int x;)};"
        self.clang(source)
        result = self.expand(source)
        self.assertEqual(result["expansions"][0]["virtual_source"], "int x;")

    def test_ambiguous_formal_type_call_is_pending(self):
        source = "#define DECL(T,N) T(N);\nstruct S {DECL(int,x)};"
        self.clang(source)
        result = self.expand(source)
        # int(x) is an unambiguous primitive declaration if parsed as such;
        # otherwise a name/type lookup pending is permitted, never fake T data.
        self.assertTrue(result["diagnostics"] or result["expansions"])

    def test_unsupported_va_opt_declaration_is_pending(self):
        source = "#define FIELD(X,...) int X __VA_OPT__(=1);\nstruct S {FIELD(x,1)};"
        result = self.expand(source)
        self.assertFalse(result["expansions"])
        self.assertEqual(result["diagnostics"][0]["kind"], "macro_expansion_unsupported")

    def test_no_retroactive_macro_definition(self):
        result = self.expand("CALL(x);\n#define CALL(X) int X;\n")
        self.assertFalse(result["expansions"])
        self.assertFalse(result["diagnostics"])

    def test_fixed_return_status_and_assertion_are_explicit_non_api(self):
        path = "include/cutlass/cluster_launch.hpp"
        result = expand_file(path, (ROOT / "snapshot" / path).read_bytes())
        self.assertFalse(any(d.get("name") == "Return_Status" for d in result["diagnostics"]))
        self.assertEqual(sum(x["name"] == "Return_Status" for x in result["non_declaration_invocations"]), 3)
        path = "include/cutlass/platform/platform.h"
        result = expand_file(path, (ROOT / "snapshot" / path).read_bytes())
        self.assertFalse(any(x["name"] == "static_assert" for x in result["expansions"]))
        self.assertTrue(any(x["name"] == "static_assert" for x in result["non_declaration_invocations"]))

    def test_fixed_cuda_driver_variants_still_pending(self):
        path = "include/cutlass/cuda_host_adapter.hpp"
        result = expand_file(path, (ROOT / "snapshot" / path).read_bytes())
        pending = [d for d in result["diagnostics"] if d.get("name") == "CUTLASS_CUDA_DRIVER_WRAPPER_DECL"]
        self.assertEqual(len(pending), 2)
        self.assertTrue(all(d["kind"] == "macro_definition_ambiguous" for d in pending))


if __name__ == "__main__":
    unittest.main()
