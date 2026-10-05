"""Independent source/Clang checks for the standalone physical bitfield module."""
from pathlib import Path
import json
import shutil
import subprocess
import unittest

from tree_sitter import Parser

from scripts.declaration_bitfields import LANGUAGE, analyze_bitfields, analyze_macro_bitfield_patterns, _Source, _LexicalColonRoles
from scripts.declaration_projection import SourceProjection

ROOT = Path(__file__).resolve().parents[1]


class BitfieldTests(unittest.TestCase):
    def lexical_role(self, text, marker):
        source = text.encode() if isinstance(text, str) else text
        position = source.index(marker.encode() if isinstance(marker, str) else marker)
        view = _Source("fixture.hpp", source)
        proof = _LexicalColonRoles(view, Parser(LANGUAGE).parse(source).root_node).classify(position)
        return source, proof

    def analyze(self, text, path="fixture.hpp"):
        source = text.encode() if isinstance(text, str) else text
        result = analyze_bitfields(path, source)
        for field in result["bitfields"]:
            for key in ("declaration", "declarator"):
                item = field[key]
                self.assertEqual(source[item["start_byte"]:item["end_byte"]].decode(), item["raw"])
            width = field["bit_width_range"]
            self.assertEqual(source[width["start_byte"]:width["end_byte"]].decode(), field["bit_width"])
            colon = field["colon_range"]
            self.assertEqual(source[colon["start_byte"]:colon["end_byte"]], b":")
            if field["name"] is None:
                self.assertIsNone(field["name_range"])
            else:
                name = field["name_range"]
                self.assertEqual(source[name["start_byte"]:name["end_byte"]].decode(), field["name"])
        self.assertEqual(len(result["candidates"]), sum(result["summary"]["candidate_classifications"].values()))
        return source, result

    def test_zero_width_multi_declarators_and_unique_anonymous_identity(self):
        source, result = self.analyze("struct S { unsigned a:3, :0, :2, b:4, :1; };")
        fields = result["bitfields"]
        self.assertEqual([(f["name"], f["bit_width"]) for f in fields],
                         [("a", "3"), (None, "0"), (None, "2"), ("b", "4"), (None, "1")])
        self.assertEqual(len({f["bitfield_source_id"] for f in fields}), 5)
        self.assertEqual(len({f["source_declaration_id"] for f in fields}), 1)
        self.assertEqual([f["declarator_index"] for f in fields], list(range(5)))
        self.assertTrue(all(f["initializer"] is None for f in fields))
        self.assertFalse(result["diagnostics"])
        self.assertEqual(result, analyze_bitfields("fixture.hpp", source))

    def test_comment_recovery_position_is_not_insertion_anchor(self):
        source, result = self.analyze("struct S { unsigned a:4, // previous field\n :2, b:1, // second\n :0; };")
        self.assertEqual(len(result["projection_edits"]), 2)
        for edit in result["projection_edits"]:
            self.assertEqual(edit["start_byte"], edit["end_byte"])
            self.assertEqual(source[edit["start_byte"]:edit["start_byte"] + 1], b":")
            self.assertIn(edit["start_line"], (2, 3))
            self.assertTrue(edit["parser_only"])
            self.assertEqual(edit["semantic_spelling"], "")
            self.assertIsNone(edit["restored_name"])
        projection = SourceProjection("fixture.hpp", source, result["projection_edits"])
        self.assertFalse(Parser(LANGUAGE).parse(projection.projected).root_node.has_error)
        reader = projection.semantic_reader()
        self.assertEqual(reader(0, len(projection.projected)), source)
        for edit in result["projection_edits"]:
            mapped = projection.span(edit["projected_start_byte"], edit["projected_end_byte"])
            self.assertEqual(mapped["start_byte"], edit["start_byte"])
            self.assertEqual(mapped["end_byte"], edit["end_byte"])

    def test_utf8_positions_and_null_names_are_physical_bytes(self):
        source, result = self.analyze('// 中文前缀 😀\nstruct S { unsigned 位:3, /* 中文 */ :2; };')
        fields = result["bitfields"]
        self.assertEqual([(f["name"], f["bit_width"]) for f in fields], [("位", "3"), (None, "2")])
        self.assertEqual(fields[0]["name_range"]["start_byte"], source.index("位".encode()))
        self.assertEqual(result["projection_edits"][0]["start_byte"], source.index(b":2"))
        other = analyze_bitfields("other.hpp", source)
        self.assertNotEqual(fields[1]["bitfield_source_id"], other["bitfields"][1]["bitfield_source_id"])

    def test_template_width_and_conditional_colon(self):
        _, result = self.analyze("template<class T, unsigned N> struct S { unsigned x:T::bits, :(N > 4 ? 4 : N); };")
        self.assertEqual([(f["name"], f["bit_width"]) for f in result["bitfields"]],
                         [("x", "T::bits"), (None, "(N > 4 ? 4 : N)")])
        self.assertEqual(result["summary"]["colon_candidates"], 3)
        self.assertFalse(result["diagnostics"])

    def test_cxx20_initializer_is_not_width(self):
        _, result = self.analyze("struct S { unsigned x:3 = 2; unsigned y:4 {3}; };")
        self.assertEqual([(f["name"], f["bit_width"], f["initializer"]) for f in result["bitfields"]],
                         [("x", "3", "2"), ("y", "4", "{3}")])
        self.assertEqual([e["kind"] for e in result["projection_edits"]],
                         ["bitfield_initializer_parser_mask", "bitfield_initializer_parser_mask"])

    def test_cxx20_comma_list_recovers_second_width_and_full_semantics(self):
        source, result = self.analyze("struct S {unsigned a:3=1, b:4{2};};")
        self.assertEqual([(f["name"], f["bit_width"], f["initializer"]) for f in result["bitfields"]],
                         [("a", "3", "1"), ("b", "4", "{2}")])
        self.assertFalse(result["diagnostics"])
        self.assertEqual(result["summary"]["parser_name_insertions"], 0)
        self.assertEqual(len(result["projection_edits"]), 2)
        for edit in result["projection_edits"]:
            self.assertTrue(edit["parser_only"])
            self.assertEqual(len(edit["parse_projection"].encode()), edit["end_byte"] - edit["start_byte"])
            self.assertEqual(edit["semantic_spelling"], source[edit["start_byte"]:edit["end_byte"]].decode())
        projection = SourceProjection("fixture.hpp", source, result["projection_edits"])
        self.assertFalse(Parser(LANGUAGE).parse(projection.projected).root_node.has_error)
        self.assertEqual(projection.semantic_reader()(0, len(projection.projected)), source)
        self.assertTrue(result["raw_parse_has_error"])

    def test_cxx20_unproven_more_complex_initializer_ranges_remain_pending(self):
        for source in ("struct S {unsigned a:3=1, b:4{2}, c:5=3;};",
                       "// 中文\nstruct S {unsigned a:3 = (1 /* 注释 */), b:4 {2};};"):
            with self.subTest(source=source):
                _, result = self.analyze(source)
                self.assertTrue(result["diagnostics"])
                self.assertFalse(result["projection_edits"])

    def test_cxx20_initializer_projection_preserves_utf8_and_comments(self):
        source, result = self.analyze("// 中文\nstruct S {unsigned a:3 = 1, b:4 {2 /* 注释 */};};")
        self.assertEqual([(f["name"], f["initializer"]) for f in result["bitfields"]],
                         [("a", "1"), ("b", "{2 /* 注释 */}")])
        self.assertFalse(result["diagnostics"])
        projection = SourceProjection("fixture.hpp", source, result["projection_edits"])
        self.assertEqual(projection.semantic_reader()(0, len(projection.projected)), source)

    def test_comments_strings_raw_strings_and_scope_operators(self):
        text = '''// fake :5\nstruct S { const char* s = "x:9"; const char* r = R"tag(x:7)tag";
        /* unsigned ignored:8; */ unsigned good:2; };\nusing X = A::B;'''
        _, result = self.analyze(text)
        self.assertEqual(result["summary"]["colon_candidates"], 1)
        self.assertEqual([f["name"] for f in result["bitfields"]], ["good"])

    def test_backslash_continued_line_comment_hides_fake_field(self):
        _, result = self.analyze("// comment \\\nunsigned fake:8;\nstruct S { unsigned x:1; };")
        self.assertEqual(result["summary"]["colon_candidates"], 1)
        self.assertEqual([f["name"] for f in result["bitfields"]], ["x"])

    def test_recovered_constructor_clauses_are_not_bitfields(self):
        _, result = self.analyze("struct S { CUTE_HOST_DEVICE S() noexcept : x(0) {} int x; };" )
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["projection_edits"])
        self.assertFalse(result["diagnostics"])

    def test_specialized_class_constructor_uses_unspecialized_name(self):
        _, result = self.analyze("template<class T> struct S; template<> struct S<int> { CUTE_HOST_DEVICE S() : x(0) {} int x; };")
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["projection_edits"])
        self.assertFalse(result["diagnostics"])

    def test_ternary_in_recovered_constructor_is_not_bitfield(self):
        text = '''struct S { CUTE_HOST_DEVICE S(int* p) :
          x(int(p ? *p : 1)), y(int(p ? *p : 2)) { x = (p ? *p : 0); }
          int x, y; };'''
        _, result = self.analyze(text)
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["projection_edits"])

    def test_other_colon_roles_and_function_body_are_not_deleted(self):
        text = '''enum E:unsigned { V }; struct B {}; struct S:B { public:
          unsigned x:2;
          void f(int* p) { label: for (int a : {1,2}) { switch(a) { case 1: break; default: break; } }
          asm volatile ("" : : "r"(p)); }
        };'''
        source, result = self.analyze(text)
        self.assertEqual([f["name"] for f in result["bitfields"]], ["x"])
        self.assertFalse(result["projection_edits"])
        self.assertFalse(result["diagnostics"])
        self.assertIn(b'asm volatile ("" : : "r"(p))', source)

    def test_missing_width_stays_pending_without_name_projection(self):
        _, result = self.analyze("struct S { unsigned x:; unsigned :; };")
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["projection_edits"])
        self.assertTrue(result["diagnostics"])
        self.assertTrue(all(d["blocks_phase_1"] for d in result["diagnostics"]))

    def test_unrelated_asm_syntax_error_remains_visible_without_projection(self):
        source, result = self.analyze('void f(int x) { asm volatile ("" : : "r"(x) "r"(x)); }')
        self.assertTrue(result["raw_parse_has_error"])
        self.assertGreater(result["raw_syntax_error_count"], 0)
        self.assertFalse(result["projection_edits"])
        self.assertFalse(result["bitfields"])
        self.assertIn(b'"r"(x) "r"(x)', source)

    def test_attributes_not_supported_are_pending_not_dropped(self):
        for text in ("struct S { unsigned x [[deprecated]]:3; };",
                     "struct S { unsigned x __attribute__((packed)):3; };"):
            with self.subTest(source=text):
                _, result = self.analyze(text)
                self.assertFalse(result["bitfields"])
                self.assertEqual(len(result["diagnostics"]), 1)
                self.assertFalse(result["projection_edits"])

    def test_macro_definition_and_all_conditional_branches_remain_visible(self):
        _, result = self.analyze("#define BF unsigned from_macro:3;\nstruct S {\n#if A\nunsigned x:2;\n#else\nunsigned :4;\n#endif\n};")
        self.assertEqual([(f["name"], f["bit_width"]) for f in result["bitfields"]], [("x", "2"), (None, "4")])
        self.assertEqual(result["bitfields"][0]["preprocessor_conditions"][0]["expression"], "A")
        self.assertIn("!", result["bitfields"][1]["preprocessor_conditions"][0]["expression"])
        self.assertEqual([d["message"] for d in result["diagnostics"]], ["preprocessor_colon_requires_expansion"])

    def test_macro_generated_declarator_is_pending_not_constructor(self):
        _, result = self.analyze("#define FIELD(x) unsigned x\nstruct S { FIELD(x):3; };")
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["projection_edits"])
        self.assertEqual(len(result["diagnostics"]), 1)

    def test_lexical_constructor_needs_initializer_list_and_separate_body(self):
        positive = "struct S { S(int x) noexcept(true) : value_{x}, other_(0) {} int value_, other_; };"
        source, proof = self.lexical_role(positive, ": value_")
        self.assertEqual(proof["classification"], "lexically_verified_constructor_initializer_separator")
        self.assertEqual(len(proof["evidence"]["initializer_ranges"]), 2)
        name = proof["evidence"]["constructor_name_range"]
        self.assertEqual(source[name["start_byte"]:name["end_byte"]], b"S")
        for negative in ("#define FIELD(x) unsigned x\nstruct S { FIELD(x):3; };",
                         "#define S(x) unsigned x\nstruct S { S(x):3; };",
                         "#define S(x) unsigned x\nstruct S { S(x):width(1); };",
                         "#define S(x) unsigned x\nstruct S { S(x):3 {1}; };",
                         "#define S(x) unsigned x\nstruct S { S(x):width(1) {}; };"):
            with self.subTest(source=negative):
                _, result = self.lexical_role(negative, ":")
                self.assertIsNone(result)

    def test_same_named_macro_bitfield_cannot_masquerade_as_constructor_body(self):
        source = "constexpr unsigned width(unsigned x) { return x; }\n#define S(x) unsigned x\nstruct S { S(x):width(3) {}; };"
        _, result = self.analyze(source)
        self.assertFalse(result["bitfields"])
        self.assertEqual(len(result["diagnostics"]), 1)
        self.assertEqual(result["diagnostics"][0]["message"], "macro_generated_declarator_requires_expansion")

    def test_lexical_template_class_scope_and_braced_default_parameters(self):
        text = "template<class T> struct S; template<> struct S<int> { struct P { int n; }; CUTE_DEVICE S(P p={}, int n=0) : p_(p), n_{n} {} P p_; int n_; };"
        source, proof = self.lexical_role(text, ": p_")
        self.assertEqual(proof["classification"], "lexically_verified_constructor_initializer_separator")
        owner = proof["evidence"]["owner_name_range"]
        self.assertEqual(source[owner["start_byte"]:owner["end_byte"]], b"S")
        self.assertEqual(proof["evidence"]["owner_header_basis"], "physical_class_keyword_name_template_base_header")

    def test_lexical_class_header_does_not_invent_owner_from_template_parameter(self):
        for text in ("template<class S> void f() { S(x):value(1) {} }",
                     "struct S *pointer; void f() { S(x):value(1) {} }"):
            with self.subTest(source=text):
                _, proof = self.lexical_role(text, ":value")
                self.assertIsNone(proof)

    def test_lexical_asm_recovers_only_direct_argument_separators(self):
        text = '''void f(int x) { asm volatile ("prefix"
#if A
"one"
#else
"two"
#endif
: : "r"(x)); }'''
        _, result = self.analyze(text)
        self.assertFalse(result["diagnostics"])
        self.assertEqual(result["summary"]["colon_candidates"], 2)
        self.assertTrue(all(c["status"] == "excluded_non_bitfield" for c in result["candidates"]))
        view = _Source("fixture.hpp", text.encode())
        roles = _LexicalColonRoles(view, Parser(LANGUAGE).parse(text.encode()).root_node)
        self.assertTrue(all(roles.classify(c["start_byte"])["classification"] == "lexically_verified_asm_argument_separator"
                            for c in result["candidates"]))
        nested = 'void f() { asm volatile("" : : "r"([] { struct S { unsigned x:3; }; return 0; }())); }'
        _, proof = self.lexical_role(nested, ":3")
        self.assertIsNone(proof)
        _, result = self.analyze(nested)
        self.assertEqual([(f["name"], f["bit_width"]) for f in result["bitfields"]], [("x", "3")])

    def test_decltype_base_has_class_header_not_member_declarator(self):
        text = "struct Base {}; struct Derived : decltype(Base{}) {};"
        _, proof = self.lexical_role(text, ": decltype")
        self.assertEqual(proof["classification"], "lexically_verified_decltype_base_separator")
        _, result = self.analyze(text)
        self.assertFalse(result["bitfields"])
        self.assertFalse(result["diagnostics"])
        _, proof = self.lexical_role("struct S { unsigned x : decltype(value)::bits; };", ": decltype")
        self.assertIsNone(proof)

    def test_real_pending_cases_use_physical_scope_when_raw_owner_is_wrong(self):
        cases = [("include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp", 337, "lexically_verified_constructor_initializer_separator"),
                 ("include/cutlass/conv/threadblock/conv2d_dgrad_filter_tile_access_iterator_optimized.h", 136, "lexically_verified_constructor_initializer_separator"),
                 ("include/cute/arch/mma_sm100_umma.hpp", 1657, "lexically_verified_asm_argument_separator"),
                 ("include/cute/container/tuple.hpp", 274, "lexically_verified_decltype_base_separator")]
        for path, line, expected in cases:
            with self.subTest(path=path):
                _, result = self.analyze((ROOT / "snapshot" / path).read_bytes(), path)
                found = [c for c in result["candidates"] if c["start_line"] == line]
                self.assertEqual(len(found), 1)
                self.assertEqual(found[0]["classification"], expected)
                self.assertIn("classification_evidence", found[0])

    def test_macro_definition_shape_probe_retains_physical_colon_candidates(self):
        source = b'#define BF(X,W) unsigned X:W; \\\n unsigned :0;\n'
        before = analyze_bitfields("fixture.hpp", source)
        patterns = analyze_macro_bitfield_patterns("fixture.hpp", source)
        self.assertEqual(len(patterns), 1)
        definition = patterns[0]
        self.assertEqual(definition["status"], "confirmed_class_member_bitfield_pattern")
        self.assertFalse(definition["changes_physical_candidate_classification"])
        self.assertEqual([(p["name_template"], p["bit_width_template"]) for p in definition["bitfield_patterns"]],
                         [("X", "W"), (None, "0")])
        self.assertEqual({p["colon_range"]["start_byte"] for p in definition["bitfield_patterns"]},
                         {c["start_byte"] for c in before["candidates"]})
        self.assertEqual(len(before["diagnostics"]), 2)
        self.assertEqual(before, analyze_bitfields("fixture.hpp", source))

    def test_macro_definition_piecewise_width_mapping_across_line_splice(self):
        source = b'#define BF(X,W) unsigned X:(W \\\n + 1);\n'
        definition = analyze_macro_bitfield_patterns("fixture.hpp", source)[0]
        self.assertEqual(definition["status"], "confirmed_class_member_bitfield_pattern")
        field = definition["bitfield_patterns"][0]
        self.assertEqual(len(field["bit_width_ranges"]), 2)
        reconstructed = b"".join(source[s["start_byte"]:s["end_byte"]] for s in field["bit_width_ranges"])
        self.assertEqual(reconstructed.decode(), field["bit_width_template"])

    def test_macro_definition_non_member_and_token_paste_shapes_stay_unproved(self):
        source = b'''#define EXPR(X) ((X)?1:0)
#define JOIN(X) unsigned f_##X:3;
#define ASM(X) asm volatile("" : : "r"(X));
#define UNKNOWN(X) FIELD(X):3;
#define ESCAPE(X) unsigned X:3; }; struct Other { unsigned y:2;
'''
        results = analyze_macro_bitfield_patterns("fixture.hpp", source)
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r["status"] == "unproved_class_member_bitfield_pattern" for r in results))
        self.assertTrue(all(not r["bitfield_patterns"] for r in results))

    def test_fixed_descriptor_files_have_explicit_independent_anonymous_ids(self):
        for relative, total, anonymous in [("include/cute/arch/mma_sm100_desc.hpp", 51, 12),
                                            ("include/cute/arch/mma_sm90_desc.hpp", 11, 6)]:
            with self.subTest(path=relative):
                _, result = self.analyze((ROOT / "snapshot" / relative).read_bytes(), relative)
                self.assertEqual(result["summary"]["bitfields"], total)
                self.assertEqual(result["summary"]["anonymous_bitfields"], anonymous)
                self.assertEqual(result["summary"]["unique_anonymous_source_ids"], anonymous)
                self.assertFalse(result["diagnostics"])
                self.assertTrue(all(f["initializer"] is None for f in result["bitfields"]))

    @unittest.skipUnless(shutil.which("clang++"), "Clang is required for independent syntax evidence")
    def test_clang_ast_matches_all_fixed_descriptor_fields(self):
        # Independently selected physical struct-body ranges, not generated from
        # this module's candidates or recovered trees. Only the anonymous owner
        # is given a fixture name; every original field spelling is unchanged.
        sections = [("include/cute/arch/mma_sm100_desc.hpp", [(103,112), (277,285), (417,433), (446,462)]),
                    ("include/cute/arch/mma_sm90_desc.hpp", [(109,125)])]
        unit = "using uint8_t=unsigned char; using uint16_t=unsigned short; using uint32_t=unsigned int;\n"
        expected = []
        serial = 0
        for path, ranges in sections:
            source = (ROOT / "snapshot" / path).read_bytes()
            result = analyze_bitfields(path, source)
            for first, last in ranges:
                unit += "struct Probe" + str(serial) + " {\n"
                unit += b"".join(source.splitlines(keepends=True)[first-1:last]).decode()
                unit += "};\n"
                serial += 1
                expected.extend((f["name"], f["bit_width"]) for f in result["bitfields"]
                                if first <= f["colon_range"]["start_line"] <= last)
        run = subprocess.run(["clang++", "-std=c++17", "-pedantic-errors", "-fsyntax-only",
                              "-Xclang", "-ast-dump=json", "-x", "c++", "-"],
                             input=unit, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        observed, stack = [], [json.loads(run.stdout)]
        while stack:
            node = stack.pop()
            if node.get("kind") == "FieldDecl" and node.get("isBitfield"):
                observed.append((node.get("name"), node["inner"][0]["value"]))
            stack.extend(reversed(node.get("inner", [])))
        self.assertEqual(len(observed), 62)
        self.assertEqual(sum(name is None for name, _ in observed), 18)
        self.assertEqual(observed, expected)

    @unittest.skipUnless(shutil.which("clang++"), "Clang is required for independent syntax evidence")
    def test_clang_original_source_not_parser_temporary_names(self):
        cases = [
            ("struct S { unsigned x:3, :0, :2, y:4; };", "c++17", 0),
            ("struct S { unsigned x:3, // comment\n :2; };", "c++17", 0),
            ("// 中文\nstruct S { unsigned 位:3, :2; };", "c++17", 0),
            ("template<class T,unsigned N> struct S { unsigned x:T::bits, :(N>4?4:N); }; struct T { static constexpr unsigned bits=2; }; S<T,3> s;", "c++17", 0),
            ("struct S { unsigned x:3=2; unsigned y:4 {3}; };", "c++20", 0),
            ("struct S {unsigned a:3=1, b:4{2};};", "c++20", 0),
            ("struct S {unsigned a:3=1, b:4{2}, c:5=3;};", "c++20", 0),
            ("struct S {unsigned a:3=(1 /* 注释 */), b:4{2};};", "c++20", 0),
            ("struct S { unsigned named:0; };", "c++17", 1),
            ("struct S { unsigned x:; };", "c++17", 1),
            ("#define FIELD(x) unsigned x\nstruct S { FIELD(x):3; };", "c++17", 0),
            ("#define S(x) unsigned x\nstruct S { S(x):3; };", "c++17", 0),
            ("#define S(x) unsigned x\nstruct S { S(x):3 {1}; };", "c++20", 0),
            ("constexpr unsigned width(unsigned x) { return x; }\n#define S(x) unsigned x\nstruct S { S(x):width(3) {}; };", "c++20", 0),
            ("struct Base {}; struct Derived : decltype(Base{}) {};", "c++17", 0),
            ("struct S { S(int x) noexcept(true) : value_{x}, other_(0) {} int value_, other_; };", "c++17", 0),
            ('void f(int x) { asm volatile(""\n#if A\n"one"\n#else\n"two"\n#endif\n: : "r"(x)); }', "c++17", 0),
            ('#define BF(X,W) unsigned X:W; unsigned :0;\nstruct S { BF(x,3) };', "c++17", 0),
        ]
        for code, standard, expected in cases:
            with self.subTest(source=code):
                run = subprocess.run(["clang++", "-std=" + standard, "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                     input=code, text=True, capture_output=True)
                self.assertEqual(run.returncode, expected, run.stderr)


if __name__ == "__main__":
    unittest.main()
