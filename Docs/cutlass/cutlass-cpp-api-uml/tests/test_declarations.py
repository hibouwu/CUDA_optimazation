import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_declarations.py"
spec = importlib.util.spec_from_file_location("extract_declarations", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class DeclarationTests(unittest.TestCase):
    def extract(self, source, path="include/cute/test.hpp"):
        extractor = module.Extractor("test_commit")
        extractor.extract(path, source.encode())
        return extractor.result()

    def test_overloads_cv_ref_and_parameter_names(self):
        result = self.extract("namespace n { int f(int x); int f(int y) {return y;} int f(float x); struct A { int f() const &; int f() &&; }; }")
        fs = [o for o in result["occurrences"] if o["name"] == "f"]
        self.assertEqual(len(fs), 5)
        self.assertEqual(fs[0]["entity_id"], fs[1]["entity_id"])
        self.assertNotEqual(fs[0]["entity_id"], fs[2]["entity_id"])
        self.assertNotEqual(fs[3]["entity_id"], fs[4]["entity_id"])

    def test_template_forward_and_specializations(self):
        result = self.extract("template<class T, int N=3> struct A; template<class T,int N> struct A {}; template<class T> struct A<T,2> {}; template<> struct A<int,2> {};")
        types = [o for o in result["occurrences"] if o["kind"] == "struct_specifier"]
        self.assertEqual(len(types), 4)
        self.assertEqual(types[0]["entity_id"], types[1]["entity_id"])
        self.assertEqual([o["specialization"] for o in types], ["primary", "primary", "partial", "explicit"])
        self.assertEqual(len({o["entity_id"] for o in types}), 3)

    def test_conditions_and_if_zero(self):
        result = self.extract("#if 0\nint never();\n#elif A\nint maybe();\n#else\nint other();\n#endif\n")
        funcs = [o for o in result["occurrences"] if o["kind"] == "function"]
        self.assertEqual(len(funcs), 3)
        self.assertTrue(funcs[0]["preprocessor_conditions"][0]["constant_false"])
        self.assertIn("!", funcs[1]["preprocessor_conditions"][0]["expression"])
        self.assertIn("A", funcs[2]["preprocessor_conditions"][0]["expression"])

    def test_annotations_offsets_and_methods(self):
        source = "struct A { private: CUTLASS_HOST_DEVICE A(int x=3) {} CUTE_HOST_DEVICE ~A(); int operator()(int x) const; operator bool() const; };"
        result = self.extract(source)
        methods = [o for o in result["occurrences"] if o["kind"] in {"constructor", "destructor", "operator"}]
        self.assertEqual(len(methods), 4)
        self.assertTrue(all(o["access"] == "private" for o in methods))
        self.assertIn("CUTLASS_HOST_DEVICE", methods[0]["raw_signature"])
        self.assertEqual(methods[0]["parameters"][0]["default"], "3")
        for occurrence in result["occurrences"]:
            span = occurrence["signature_range"]
            self.assertEqual(source.encode()[span["start_byte"]:span["end_byte"]].decode().rstrip(), occurrence["raw_signature"])

    def test_locals_excluded_local_classes_owned(self):
        result = self.extract("void f() { int local; struct L { int member; void g() {int inner;} }; } void h() { struct L {int z;}; }")
        names = [o["name"] for o in result["occurrences"]]
        self.assertNotIn("local", names)
        self.assertNotIn("inner", names)
        self.assertIn("member", names)
        local_types = [o for o in result["occurrences"] if o["name"] == "L"]
        self.assertEqual(len(local_types), 2)
        self.assertNotEqual(local_types[0]["entity_id"], local_types[1]["entity_id"])
        self.assertTrue(any(s["kind"] == "function" for s in local_types[0]["scope_chain"]))

    def test_alias_enum_multi_members_and_function_pointer(self):
        result = self.extract("using A=int; typedef int I,J; enum class E:int { X, Y=2 }; struct S { int a,b; }; int (*p)(int);")
        names = [o["name"] for o in result["occurrences"]]
        self.assertEqual(set(names), {"A", "I", "J", "E", "X", "Y", "S", "a", "b", "p"})
        pointer = next(o for o in result["occurrences"] if o["name"] == "p")
        self.assertEqual(pointer["kind"], "variable")

    def test_static_does_not_merge_across_files(self):
        extractor = module.Extractor("test")
        extractor.extract("include/cute/a.hpp", b"static int f();")
        extractor.extract("include/cute/b.hpp", b"static int f();")
        self.assertEqual(len(extractor.entities), 2)

    def test_macro_definition_retained_and_expansion_materialized(self):
        result = self.extract("#define MAKE(T) struct T {};\nMAKE(A)\n")
        self.assertTrue(any(o["kind"] == "macro_definition" and "struct" in o["replacement"] for o in result["occurrences"]))
        self.assertTrue(any(o["name"] == "A" and "macro_origin" in o for o in result["occurrences"]))
        self.assertFalse(result["completion"]["phase_1_passed"])

    def test_parse_error_not_silenced_by_function_body(self):
        result = self.extract("struct S { void good(); void broken() { @@@; } int after; };")
        names = [o["name"] for o in result["occurrences"]]
        self.assertIn("good", names)
        self.assertIn("after", names)
        self.assertTrue(any(d["category"] == "parse_error" and d["blocks_phase_1"] for d in result["diagnostics"]))

    def test_generated_macro_entities_have_physical_and_virtual_provenance(self):
        source = "#define OP(OP) template<class T> int operator OP(T a) {return 0;}\nnamespace n {\nOP(+)\nOP(-)\n}\n"
        result = self.extract(source)
        generated = [o for o in result["occurrences"] if "macro_origin" in o]
        self.assertEqual(len(generated), 2)
        self.assertEqual({o["qualified_name"] for o in generated}, {"n::operator +", "n::operator -"})
        for o in generated:
            self.assertEqual(source.encode()[o["start_byte"]:o["end_byte"]].decode(), "OP(" + o["name"][-1] + ")")
            self.assertIn("virtual_range", o)
            self.assertEqual(o["macro_origin"]["definition"]["start_line"], 1)
        self.assertNotEqual(generated[0]["entity_id"], generated[1]["entity_id"])

    def test_variadic_and_const_return_are_preserved(self):
        result = self.extract("const int* f(int a,...); const int* f(int b);")
        funcs = [o for o in result["occurrences"] if o["kind"] == "function"]
        self.assertEqual(funcs[0]["parameters"][-1]["type"], "...")
        self.assertIn("const", funcs[0]["return_type"])
        self.assertNotEqual(funcs[0]["entity_id"], funcs[1]["entity_id"])

    def test_constraint_macro_projection_keeps_original_signature(self):
        source = "namespace cute { template<class T, __CUTE_REQUIRES(sizeof(T)>1)> int f(T a); }"
        result = self.extract(source)
        function = next(o for o in result["occurrences"] if o["name"] == "f")
        self.assertIn("__CUTE_REQUIRES", function["raw_signature"])
        self.assertIn("enable_if", function["expanded_signature"])
        self.assertNotIn("__codex_parser_", function["expanded_signature"])
        self.assertIsNone(function["template_parameters"][-1]["parameters"][-1]["name"])
        self.assertFalse(result["diagnostics"])
        for o in result["occurrences"]:
            span = o["signature_range"]
            self.assertEqual(source.encode()[span["start_byte"]:span["end_byte"]].decode().rstrip(), o["raw_signature"])

    def test_preprocessor_else_if_does_not_escape_function_scope(self):
        source = "int f() { if constexpr (true) {return 1;}\n#if X\nelse if constexpr(false) {return 2;}\n#endif\nelse {return 3;} }\nint after;"
        result = self.extract(source)
        names = [o["name"] for o in result["occurrences"]]
        self.assertEqual(names, ["f", "after"])
        self.assertFalse(result["diagnostics"])
        self.assertTrue(result["files"][0]["preprocessor_regions"])

    def test_reference_return_overloads_are_methods(self):
        result = self.extract("struct S { S& f(int) {return *this;} S& f(double) {return *this;} S& operator++(); S operator++(int); };")
        methods = [o for o in result["occurrences"] if o["name"] == "f"]
        self.assertEqual([o["kind"] for o in methods], ["method", "method"])
        self.assertNotEqual(methods[0]["entity_id"], methods[1]["entity_id"])
        self.assertEqual(methods[0]["return_type"], "S &")
        operators = [o for o in result["occurrences"] if o["kind"] == "operator"]
        self.assertEqual(len(operators), 2)
        self.assertNotEqual(operators[0]["entity_id"], operators[1]["entity_id"])

    def test_friend_injected_into_namespace(self):
        result = self.extract("namespace n { struct S { friend int f(S const&); }; int f(S const& x) {return 1;} }")
        funcs = [o for o in result["occurrences"] if o["name"] == "f"]
        self.assertEqual(len(funcs), 2)
        self.assertEqual([o["qualified_name"] for o in funcs], ["n::f", "n::f"])
        self.assertEqual(funcs[0]["entity_id"], funcs[1]["entity_id"])

    def test_compound_namespace_identity(self):
        result = self.extract("namespace a::b { int f(); } namespace a { namespace b { int f(); } }")
        funcs = [o for o in result["occurrences"] if o["name"] == "f"]
        self.assertEqual(funcs[0]["entity_id"], funcs[1]["entity_id"])

    def test_literal_namespace_macro_is_resolved(self):
        result = self.extract("#define API_NS std\nnamespace API_NS { struct X {}; }")
        self.assertFalse(result['diagnostics'])
        x = next(o for o in result["occurrences"] if o["name"] == "X")
        self.assertEqual(x['qualified_name'],'std::X')
        self.assertEqual(x["scope_chain"][0]["name_resolution"]["status"], "literal_macro_binding")

    def test_friend_prefix_qualifiers_retained(self):
        result = self.extract("struct S { CUTE_HOST_DEVICE constexpr friend int f(S const& a) {return 0;} };")
        f = next(o for o in result["occurrences"] if o["name"] == "f")
        self.assertTrue(f["raw_signature"].startswith("CUTE_HOST_DEVICE constexpr friend"))
        self.assertIn("CUTE_HOST_DEVICE", f["attributes"])
        self.assertIn("constexpr", f["prefix_specifiers"])


if __name__ == "__main__":
    unittest.main()
