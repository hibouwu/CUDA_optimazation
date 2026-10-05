"""Independent source/token/identity checks of integrated bitfields."""
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_declarations import Extractor
from declaration_projection import SourceProjection


class BitfieldIndependentReview(unittest.TestCase):
    def extract(self, source, path="fixture.hpp"):
        result = Extractor("bitfield-independent-review")
        result.extract(path, source.encode() if isinstance(source, str) else source)
        return result

    def fields(self, result):
        return [o for o in result.occurrences if o.get("is_bitfield")]

    def clang(self, source):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++20", "-pedantic-errors", "-fsyntax-only", "-x", "c++", "-"],
                                input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def assert_physical_field(self, source, field):
        for field_name, expected in (("colon_range", ":"), ("bit_width_range", field["bit_width"]),
                                     ("declarator_range", field["declarator"])):
            span = field[field_name]
            self.assertEqual(source[span["start_byte"]:span["end_byte"]].decode(), expected)
        if field["name"] is None:
            self.assertIsNone(field["name_range"])
        else:
            span = field["name_range"]
            self.assertEqual(source[span["start_byte"]:span["end_byte"]].decode(), field["name"])
        if field["initializer"] is None:
            self.assertIsNone(field["initializer_range"])
        else:
            span = field["initializer_range"]
            self.assertEqual(source[span["start_byte"]:span["end_byte"]].decode(), field["initializer"])
        signature = field["signature_range"]
        self.assertEqual(source[signature["start_byte"]:signature["end_byte"]].decode().rstrip(), field["raw_signature"])

    def assert_model_coverage(self, artifact):
        model_ids = {f["bitfield_source_id"] for record in artifact["files"] for f in record.get("bitfield_analysis", {}).get("bitfields", [])}
        occurrence_ids = {o["bitfield_source_id"] for o in artifact["occurrences"] if o.get("is_bitfield")}
        self.assertTrue(model_ids <= occurrence_ids, sorted(model_ids - occurrence_ids))

    def test_fixed_descriptors_every_field_points_to_real_tokens(self):
        expected = {"include/cute/arch/mma_sm100_desc.hpp": (51, 12),
                    "include/cute/arch/mma_sm90_desc.hpp": (11, 6)}
        anonymous_entities = set()
        for path, (count, anonymous) in expected.items():
            source = (ROOT / "snapshot" / path).read_bytes()
            result = self.extract(source, path)
            fields = self.fields(result)
            self.assertEqual(len(fields), count)
            self.assertEqual(sum(f["name"] is None for f in fields), anonymous)
            for field in fields:
                self.assert_physical_field(source, field)
                self.assertIsNone(field["initializer"])
                self.assertEqual(field["access"], "public")
                if field["name"] is None:
                    anonymous_entities.add(field["entity_id"])
            self.assert_model_coverage(result.result())
            self.assertFalse(result.diagnostics)
        self.assertEqual(len(anonymous_entities), 18)

    def test_three_layer_origin_span_preserves_both_colons(self):
        original = b"aa :3, :0;"
        p1 = SourceProjection("fixture.hpp", original, [{"start_byte": 0, "end_byte": 2,
                              "expanded": "aa", "semantic_spelling": "aa", "parse_projection": "long_parser_type"}])
        insertion = p1.projected.rindex(b":")
        p2 = SourceProjection("fixture.hpp", p1.projected, [{"start_byte": insertion, "end_byte": insertion,
                              "expanded": "", "semantic_spelling": "", "parse_projection": "__parser_anon"}])
        p3 = SourceProjection("fixture.hpp", p2.projected, [
            {"start_byte": 0, "end_byte": 0, "expanded": "", "semantic_spelling": "", "parse_projection": "struct Shell{"},
            {"start_byte": len(p2.projected), "end_byte": len(p2.projected), "expanded": "", "semantic_spelling": "", "parse_projection": "};"},
        ])
        reader = p3.semantic_reader(p2.semantic_reader(p1.semantic_reader()))
        self.assertEqual(reader(0, len(p3.projected)), original)
        origins = [reader.origin_span(m.start(), m.end()) for m in re.finditer(b":", p3.projected)]
        self.assertEqual([x["start_byte"] for x in origins], [original.index(b":"), original.rindex(b":")])
        self.assertTrue(all(original[x["start_byte"]:x["end_byte"]] == b":" for x in origins))

    def test_initializers_widths_and_anonymous_zero_width_are_distinct(self):
        source = "struct S {unsigned first:3=1, second:4{2}; unsigned :0; unsigned :2;};"
        self.clang(source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual([(f["name"], f["bit_width"], f["initializer"]) for f in fields],
                         [("first", "3", "1"), ("second", "4", "{2}"), (None, "0", None), (None, "2", None)])
        for field in fields:
            self.assert_physical_field(source.encode(), field)
        self.assertFalse(result.diagnostics)

    def test_identical_field_names_in_different_scopes_do_not_merge(self):
        source = "struct A {unsigned value:1, :2;}; struct B {unsigned value:1, :2;};"
        self.clang(source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 4)
        self.assertEqual(len({f["entity_id"] for f in fields}), 4)
        self.assertEqual({f["qualified_name"] for f in fields if f["name"]}, {"A::value", "B::value"})
        self.assertFalse(result.diagnostics)

    def test_source_delete_removes_named_field_not_anonymous_neighbor(self):
        source = "#define FIELDS(N) unsigned N:1, :2;\nstruct S {FIELDS(x)};"
        deleted = source.replace("unsigned N:1, :2;", "unsigned :2;")
        self.clang(source)
        self.clang(deleted)
        before, after = self.extract(source), self.extract(deleted)
        self.assertEqual([(f["name"], f["bit_width"]) for f in self.fields(before)], [("x", "1"), (None, "2")])
        self.assertEqual([(f["name"], f["bit_width"]) for f in self.fields(after)], [(None, "2")])
        self.assertEqual(len(before.files[0]["macro_bitfield_patterns"][0]["classified_colons"]), 2)
        self.assertEqual(len(after.files[0]["macro_bitfield_patterns"][0]["classified_colons"]), 1)
        self.assertFalse(before.diagnostics + after.diagnostics)

    def test_deleted_output_is_rejected_by_independent_model_check(self):
        result = self.extract("struct S {unsigned first:1, :2;};")
        artifact = result.result()
        self.assert_model_coverage(artifact)
        damaged = copy.deepcopy(artifact)
        damaged["occurrences"] = [o for o in damaged["occurrences"] if not o.get("is_bitfield") or o["name"] is not None]
        with self.assertRaises(AssertionError):
            self.assert_model_coverage(damaged)

    def test_macro_definition_pattern_points_to_actual_colon_not_comment(self):
        source = "#define FIELDS(N) unsigned N /* : comment */ : 1, \\\n : 2;\nstruct S {FIELDS(x)};"
        result = self.extract(source)
        patterns = result.files[0]["macro_bitfield_patterns"]
        self.assertEqual(len(patterns), 1)
        pattern = patterns[0]
        self.assertFalse(pattern["is_concrete_api_instance"])
        self.assertEqual(len(pattern["classified_colons"]), 2)
        comment_colon = source.index(": comment")
        proof = pattern["definition_shape_proof"]
        self.assertEqual(proof["status"], "confirmed_class_member_bitfield_pattern")
        model = {f["pattern_id"]: f for f in proof["bitfield_patterns"]}
        for classified in pattern["classified_colons"]:
            physical = classified["physical_colon"]
            self.assertEqual(source.encode()[physical["start_byte"]:physical["end_byte"]], b":")
            self.assertNotEqual(physical["start_byte"], comment_colon)
            field = model[classified["pattern_bitfield_source_id"]]
            self.assertEqual(field["colon_range"], physical)
            pieces = [s for s in proof["source_segments"]
                      if s["physical_range"]["start_byte"] <= physical["start_byte"] < s["physical_range"]["end_byte"]]
            self.assertEqual(len(pieces), 1)
            piece = pieces[0]
            virtual_colon = piece["virtual_start_byte"] + physical["start_byte"] - piece["physical_range"]["start_byte"]
            self.assertEqual(proof["virtual_source"].encode()[virtual_colon:virtual_colon+1], b":")
        # A remaining raw preprocessor parse error is not hidden by this proof.
        self.assertTrue(result.diagnostics or len(self.fields(result)) == 2)

    def test_macro_fragment_has_real_scope_and_separate_virtual_token_spans(self):
        source = "#define FIELDS(N) unsigned N:1, :2;\nclass A {FIELDS(x)}; struct B {protected: FIELDS(x)};"
        self.clang(source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 4)
        self.assertEqual(len({f["entity_id"] for f in fields}), 4)
        expansions = {e["invocation_id"]: e for e in result.files[0]["macro_expansions"]}
        for field in fields:
            expansion = expansions[field["macro_origin"]["invocation_id"]]
            self.assertEqual(field["access"], "private" if field["qualified_name"].startswith("A::") else "protected")
            self.assertEqual(field["raw_signature"], expansion["virtual_source"].rstrip())
            for key, expected in (("colon_range", ":"), ("bit_width_range", field["bit_width"])):
                span = field[key]["virtual_range"]
                self.assertEqual(expansion["virtual_source"].encode()[span["start_byte"]:span["end_byte"]].decode(), expected)
            owner = field["bitfield_origin"]["owner"]
            self.assertTrue(owner["range_is_fragment_grammar_context"])
            self.assertEqual(owner["source_scope_entity_id"], field["scope_chain"][-1]["entity_id"])
        self.assertNotIn("__codex_parser_", json.dumps(list(result.entities.values())))
        self.assertFalse(result.diagnostics)

    def test_macro_cpp20_initializers_are_not_bit_widths(self):
        source = "#define FIELDS(N) unsigned N:3=1, other:4{2};\nstruct S {FIELDS(x)};"
        self.clang(source)
        result = self.extract(source)
        self.assertEqual([(f["name"], f["bit_width"], f["initializer"]) for f in self.fields(result)], [("x", "3", "1"), ("other", "4", "{2}")])
        self.assertFalse(result.diagnostics)

    def test_named_conditional_width_variants_share_entity_not_variant(self):
        source = "struct S {\n#if A\nunsigned value:1;\n#else\nunsigned value:2;\n#endif\n};"
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 2)
        self.assertEqual(len({f["entity_id"] for f in fields}), 1)
        self.assertEqual(len({f["variant_id"] for f in fields}), 2)
        self.assertEqual({f["bit_width"] for f in fields}, {"1", "2"})
        self.assertTrue(all(f["preprocessor_conditions"] for f in fields))
        self.assertFalse(result.diagnostics)

    def test_external_and_callsite_conditions_keep_their_own_source_lines(self):
        source = "#define FIELDS(N) unsigned N:1, :2;\n#if LOCAL\nnamespace cutlass {class S {FIELDS(x)};}\n#endif"
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 4)
        helper_path = "include/cutlass/detail/helper_macros.hpp"
        helper_lines = (ROOT / "snapshot" / helper_path).read_text().splitlines()
        for field in fields:
            guards = field["preprocessor_conditions"]
            local = next(g for g in guards if g["expression"] == "LOCAL")
            external = next(g for g in guards if "CUTLASS_NAMESPACE" in g["expression"])
            self.assertEqual((local["directive_path"], local["directive_line"]), ("fixture.hpp", 2))
            self.assertEqual(external["directive_path"], helper_path)
            self.assertIn("ifdef CUTLASS_NAMESPACE", helper_lines[external["directive_line"]-1])
        self.assertFalse(result.diagnostics)

    def test_macro_cannot_escape_fragment_wrapper_without_pending(self):
        source = "#define FIELDS(N) unsigned N:1; }; struct Other { unsigned other:2;\nstruct S {FIELDS(x)};"
        self.clang(source)
        result = self.extract(source)
        self.assertTrue(any(d["category"] == "macro_member_context_escape_pending" and d["blocks_phase_1"] for d in result.diagnostics))
        self.assertTrue(all(e["status"] != "expanded_and_parsed" for e in result.files[0]["macro_expansions"]))
        self.assertFalse(result.files[0]["macro_bitfield_patterns"])

    def test_shared_conditional_header_body_preserves_anonymous_physical_model(self):
        source = "static int\n#if A\nf(void const* x)\n#else\nf(void const*)\n#endif\n{struct Local {unsigned value:1, :2;};return 0;}"
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 4)
        self.assertFalse(result.diagnostics)
        with self.subTest(contract="anonymous_name_and_identity"):
            self.assertEqual([f["name"] for f in fields], ["value", None, "value", None])
            self.assertNotIn("__codex_parser_", json.dumps(list(result.entities.values())))
            self.assertEqual(len({f["entity_id"] for f in fields}), 2)
        with self.subTest(contract="physical_source_model_coverage"):
            self.assert_model_coverage(result.result())
            self.assertEqual(len({f["bitfield_source_id"] for f in fields}), 2)
        with self.subTest(contract="raw_signature_and_declarator"):
            for field in fields:
                self.assert_physical_field(source.encode(), field)

    def test_prefix_anonymous_field_does_not_shift_header_body_identity(self):
        source = "struct Prefix {unsigned :3;};\nstatic int\n#if A\nf(void const* x)\n#else\nf(void const*)\n#endif\n{struct Local {unsigned value:1, :2;};return 0;}"
        self.clang("#define A 1\n" + source)
        self.clang("#define A 0\n" + source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 5)
        model_ids = {f["bitfield_source_id"] for f in result.files[0]["bitfield_analysis"]["bitfields"]}
        self.assertEqual(len(model_ids), 3)
        self.assertEqual({f["bitfield_source_id"] for f in fields}, model_ids)
        self.assertEqual(len({f["entity_id"] for f in fields}), 3)
        locals_ = [o for o in result.occurrences if o["name"] == "Local"]
        self.assertEqual(len(locals_), 2)
        self.assertEqual(len({o["entity_id"] for o in locals_}), 1)
        for field in fields:
            self.assert_physical_field(source.encode(), field)
            self.assertNotIn("__codex_parser_", field.get("expanded_signature", ""))
        self.assertNotIn("__codex_parser_", json.dumps(list(result.entities.values())))
        self.assertFalse(result.diagnostics)

    def test_prefix_and_nested_body_conditions_keep_final_physical_models(self):
        source = "// 前缀位置\nstruct Prefix {unsigned :0, padding:2;};\nstatic int\n#if A\nf(void const* x)\n#else\nf(void const*)\n#endif\n{\n#if B\nstruct Local {unsigned value:1, :2;};\n#else\n#if C\nstruct Local {unsigned value:3, :4;};\n#else\nstruct Local {unsigned value:5, :6;};\n#endif\n#endif\nreturn 0;}"
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    self.clang(f"#define A {a}\n#define B {b}\n#define C {c}\n" + source)
        result = self.extract(source)
        fields = self.fields(result)
        self.assertEqual(len(fields), 14)
        model_ids = {f["bitfield_source_id"] for f in result.files[0]["bitfield_analysis"]["bitfields"]}
        self.assertEqual(len(model_ids), 8)
        self.assertEqual({f["bitfield_source_id"] for f in fields}, model_ids)
        self.assertEqual(sum(f["name"] is None for f in fields), 7)
        locals_ = [o for o in result.occurrences if o["name"] == "Local"]
        self.assertEqual(len(locals_), 6)
        self.assertEqual(len({o["entity_id"] for o in locals_}), 1)
        physical_model_uses = {}
        for field in fields:
            self.assert_physical_field(source.encode(), field)
            physical_model_uses.setdefault(field["bitfield_source_id"], []).append(field)
            if field["qualified_name"].startswith("f::"):
                self.assertTrue(field["preprocessor_conditions"])
                expressions = " ".join(c["expression"] for c in field["preprocessor_conditions"])
                self.assertIn("A", expressions)
                self.assertIn("B", expressions)
                if field["bit_width"] in {"3", "4", "5", "6"}:
                    self.assertIn("C", expressions)
        self.assertEqual(sorted(len(uses) for uses in physical_model_uses.values()), [1, 1, 2, 2, 2, 2, 2, 2])
        self.assertNotIn("__codex_parser_", json.dumps(list(result.entities.values())))
        self.assertFalse(result.diagnostics)


if __name__ == "__main__":
    unittest.main()
