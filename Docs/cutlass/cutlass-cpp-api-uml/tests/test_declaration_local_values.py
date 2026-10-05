"""Source-binding counterexamples plus independent Clang VarDecl oracles."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_local_values", ROOT / "scripts/declaration_local_values.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def analyze(source):
    return module.analyze_local_values("include/cute/local_fixture.hpp", source.encode())


def candidate(source, name="tmp"):
    return next(c for c in analyze(source)["candidates"] if c["name"] == name)


def walk(node):
    yield node
    for child in node.get("inner", []):
        yield from walk(child)


def compiler(source):
    result = subprocess.run(["clang++", "-x", "c++", "-std=c++17", "-fsyntax-only", "-Xclang", "-ast-dump=json", "-"],
                            input=source, text=True, capture_output=True, check=False)
    return result, json.loads(result.stdout) if result.stdout.strip() else {}


class LocalValueProofTests(unittest.TestCase):
    def test_parameter_binding_is_direct_evidence(self):
        source = "void f(float lhs){float tmp(lhs);}"
        c = candidate(source)
        self.assertEqual(c["status"], "classified_non_api")
        binding = c["argument_bindings"][0]["binding"]
        self.assertEqual((binding["name"], binding["kind"]), ("lhs", "function_parameter"))
        a, b = binding["name_range"]["start_byte"], binding["name_range"]["end_byte"]
        self.assertEqual(source[a:b], "lhs")
        self.assertTrue(c["local_object_id"])
        self.assertIn("initialization/conversion call target", c["phase2_obligations"])

    def test_deleted_parameter_and_argument_type_do_not_prove_value(self):
        self.assertEqual(candidate("void f(){float tmp(lhs);}")["status"], "pending")
        c = candidate("using lhs=int; void f(){float tmp(lhs);}")
        self.assertEqual(c["status"], "pending")
        self.assertEqual(c["argument_bindings"][0]["binding"]["category"], "type")

    def test_nearest_type_alias_shadows_outer_parameter(self):
        c = candidate("void f(float lhs){{using lhs=int;float tmp(lhs);}}")
        self.assertEqual(c["status"], "pending")
        self.assertEqual(c["argument_bindings"][0]["binding"]["kind"], "using_alias")
        c = candidate("void f(float lhs){{typedef int lhs;float tmp(lhs);}}")
        self.assertEqual(c["status"], "pending")

    def test_prior_block_object_and_proven_initialization_chain(self):
        source = "void f(float lhs){float first(lhs);float tmp(first);}"
        c = candidate(source)
        self.assertEqual(c["status"], "classified_non_api")
        self.assertEqual(c["argument_bindings"][0]["binding"]["kind"], "proven_local_object")
        c = candidate("void f(){int value=1;float tmp(value);}")
        self.assertEqual(c["argument_bindings"][0]["binding"]["kind"], "block_object")
        self.assertEqual(candidate("void f(){float tmp(value);int value=1;}")["status"], "pending")

    def test_block_objects_do_not_escape_their_scope(self):
        self.assertEqual(candidate("void f(){{int value=1;}float tmp(value);}")["status"], "pending")
        self.assertEqual(candidate("void f(){if(int value=1){}float tmp(value);}")["status"], "pending")

    def test_explicit_member_type_and_value_can_be_proven(self):
        source = "struct S{typedef int* pointer;pointer _ptr;void release(){pointer p(_ptr);}};"
        c = candidate(source, "p")
        self.assertEqual(c["status"], "classified_non_api")
        self.assertEqual(c["target_type_binding"]["kind"], "typedef")
        self.assertEqual(c["argument_bindings"][0]["binding"]["kind"], "member_object")

    def test_class_complete_scope_includes_later_member_but_not_later_type_as_value(self):
        good = "struct S{using pointer=int*;void release(){pointer p(_ptr);}pointer _ptr;};"
        self.assertEqual(candidate(good, "p")["status"], "classified_non_api")
        bad = "struct S{using pointer=int*;void release(){pointer p(_ptr);}using _ptr=int;};"
        self.assertEqual(candidate(bad, "p")["status"], "pending")
        hidden = "struct S{using pointer=int*;pointer _ptr;void release(){{using _ptr=int;pointer p(_ptr);}}};"
        self.assertEqual(candidate(hidden, "p")["status"], "pending")

    def test_local_class_does_not_capture_enclosing_automatic_value(self):
        source = "void outer(int lhs){struct S{void f(){float tmp(lhs);}};}"
        c = candidate(source)
        self.assertEqual(c["status"], "pending")
        self.assertIn("enclosing_function_automatic_value_not_accessible_to_local_class", c["pending_reasons"])

    def test_condition_visibility_and_opposite_branch_shadow(self):
        proven = "void f(int lhs){\n#if FLAG\nint value=lhs;\nfloat tmp(value);\n#endif\n}"
        self.assertEqual(candidate(proven)["status"], "classified_non_api")
        unavailable = "void f(){\n#if FLAG\nint value=1;\n#endif\nfloat tmp(value);\n}"
        self.assertEqual(candidate(unavailable)["status"], "pending")
        opposite = "void f(int lhs){\n#if FLAG\nusing lhs=int;\n#else\nfloat tmp(lhs);\n#endif\n}"
        self.assertEqual(candidate(opposite)["status"], "classified_non_api")
        uncertain = "void f(int lhs){\n#if FLAG\nusing lhs=int;\n#endif\nfloat tmp(lhs);\n}"
        self.assertEqual(candidate(uncertain)["status"], "pending")

    def test_separate_similarly_spelled_guards_are_not_assumed_equivalent(self):
        source = "void f(){\n#if FLAG\nint value=1;\n#endif\n#if FLAG\nfloat tmp(value);\n#endif\n}"
        self.assertEqual(candidate(source)["status"], "pending")

    def test_lambda_and_for_scopes_remain_explicit_pending(self):
        for source in ("void f(int lhs){auto w=[] {float tmp(lhs);};}",
                       "void f(int lhs){auto w=[lhs] {float tmp(lhs);};}",
                       "void f(int lhs){for(int i=0;i<1;++i){float tmp(lhs);}}"):
            c = candidate(source)
            self.assertEqual(c["status"], "pending")
            self.assertIsNone(c["target_type_binding"])
            self.assertFalse(any("binding" in a for a in c["argument_bindings"]))

    def test_extern_unknown_typename_and_empty_parens_are_not_objects(self):
        for source in ("void f(float lhs){extern float tmp(lhs);}",
                       "void f(float lhs){Unknown tmp(lhs);}",
                       "template<class T> void f(float lhs){typename T::Unknown tmp(lhs);}",
                       "void f(){float tmp();}"):
            self.assertEqual(candidate(source)["status"], "pending", source)

    def test_declared_template_type_parameter_is_not_an_unknown_typename(self):
        c = candidate("template<class T> void f(T lhs){T tmp(lhs);}")
        self.assertEqual(c["status"], "classified_non_api", c["pending_reasons"])
        self.assertEqual(c["target_type_binding"]["kind"], "template_type_parameter")

    def test_macro_cannot_be_overridden_by_an_apparent_value_binding(self):
        source = "void f(int lhs){\n#define lhs int\nfloat tmp(lhs);\n}"
        c = candidate(source)
        self.assertEqual(c["status"], "pending")
        self.assertIn("macro_can_change_initializer_identifier", c["pending_reasons"])

    def test_repeated_lookup_is_stable_and_unknown_range_is_pending(self):
        source = b"void f(float lhs){float tmp(lhs);}"
        analyzer = module.LocalValueAnalyzer("x.hpp", source)
        first = analyzer.analyze()
        self.assertEqual(first, analyzer.analyze())
        c = first["candidates"][0]
        self.assertEqual(analyzer.classify_range(c["start_byte"], c["end_byte"]), c)
        self.assertEqual(analyzer.classify_range(0, 3)["status"], "pending")

    def test_fixed_source_three_regression_families(self):
        expected = {"include/cutlass/bfloat16.h": ("float tmp(lhs);", 4, "classified_non_api"),
                    "include/cutlass/platform/platform.h": ("pointer p(_ptr);", 1, "classified_non_api"),
                    "include/cutlass/array_subbyte.h": ("reference ref(storage, i);", 1, "pending")}
        for path, (raw, count, status) in expected.items():
            source = (ROOT / "snapshot" / path).read_bytes()
            rows = [c for c in module.analyze_local_values(path, source)["candidates"] if c["raw"] == raw]
            self.assertEqual(len(rows), count)
            self.assertTrue(all(c["status"] == status for c in rows))
            self.assertTrue(all(source[c["start_byte"]:c["end_byte"]].decode() == c["raw"] for c in rows))


@unittest.skipUnless(shutil.which("clang++"), "clang++ required for independent AST oracle")
class ClangLocalValueOracle(unittest.TestCase):
    def declaration(self, source, name):
        result, tree = compiler(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        nodes = {n["id"]: n for n in walk(tree) if n.get("name") == name and n.get("kind") in {"VarDecl", "FunctionDecl"}}
        self.assertEqual(len(nodes), 1)
        return next(iter(nodes.values())), tree

    def test_parameter_initializer_is_vardecl_and_references_parameter(self):
        source = "void f(float lhs){float tmp(lhs);}"
        declaration, tree = self.declaration(source, "tmp")
        self.assertEqual(declaration["kind"], "VarDecl")
        self.assertTrue(any(n.get("kind") == "DeclRefExpr" and n.get("referencedDecl", {}).get("kind") == "ParmVarDecl" and n["referencedDecl"]["name"] == "lhs" for n in walk(tree)))
        self.assertEqual(candidate(source)["status"], "classified_non_api")

    def test_member_and_later_member_are_vardecl(self):
        for source in ("struct S{typedef int* pointer;pointer _ptr;void release(){pointer p(_ptr);}};",
                       "struct S{typedef int* pointer;void release(){pointer p(_ptr);}pointer _ptr;};"):
            declaration, tree = self.declaration(source, "p")
            self.assertEqual(declaration["kind"], "VarDecl")
            self.assertTrue(any(n.get("kind") == "MemberExpr" and n.get("name") == "_ptr" for n in walk(tree)))
            self.assertEqual(candidate(source, "p")["status"], "classified_non_api")

    def test_type_shadow_really_changes_parse_to_functiondecl(self):
        for source, name in (("void f(float lhs){{using lhs=int;float tmp(lhs);}}", "tmp"),
                             ("struct S{using pointer=int*;void release(){pointer p(_ptr);}using _ptr=int;};", "p")):
            declaration, _ = self.declaration(source, name)
            self.assertEqual(declaration["kind"], "FunctionDecl")
            self.assertEqual(candidate(source, name)["status"], "pending")

    def test_for_variable_oracle_does_not_expand_module_scope_claim(self):
        source = "struct reference{reference(int*,int);};void f(int* storage){for(int i=0;i<1;++i){reference ref(storage,i);}}"
        declaration, _ = self.declaration(source, "ref")
        self.assertEqual(declaration["kind"], "VarDecl")
        c = candidate(source, "ref")
        self.assertEqual(c["status"], "pending")
        self.assertIn("for_initializer_and_iteration_scope_pending", c["pending_reasons"])

    def test_extern_and_noncapturing_lambda_recovery_are_not_semantic_proof(self):
        for source in ("void f(float lhs){extern float tmp(lhs);}",
                       "void f(float lhs){auto w=[] {float tmp(lhs);};}"):
            result, _ = compiler(source)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(candidate(source)["status"], "pending")


if __name__ == "__main__":
    unittest.main()
