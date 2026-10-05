"""Adversarial reconciliation tests; fixtures do NOT call the entity extractor."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_reconciler", ROOT / "scripts/reconcile_candidates.py")
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)
PATH = "include/cute/reconcile_fixture.hpp"


def occurrence(source, signature, name, kind, *, body=None, end=None, declarator=None, condition_lines=(), **extra):
    raw = source.encode()
    begin = raw.index(signature.encode())
    stop = begin + len(signature.encode())
    return {
        "id": "occ_" + module.sha(f"{begin}:{name}:{kind}".encode())[:12],
        "entity_id": "ent_" + module.sha(f"{begin}:{name}:{kind}".encode())[:12],
        "kind": kind, "name": name, "qualified_name": name,
        "range": [begin, end if end is not None else (body[1] if body else stop)],
        "signature": [begin, stop], "declarator": declarator, "body": body,
        "syntax": [begin, stop], "raw_signature_sha256": module.sha(signature.rstrip().encode()),
        "parse_status": "parsed", "attributes": [], "parameter_count": 0,
        "condition_lines": list(condition_lines), "macro_origin": None,
        "scope_chain": [], "name_resolution": {}, "qualified_name_resolution": "source_name", **extra,
    }


def candidates(source):
    return module.LEXER.scan_sources({PATH: source.encode()})["candidates"]


def reconcile(source, occurrences=(), diagnostics=(), cs=None, file_record=None):
    cs = candidates(source) if cs is None else cs
    defs = {c["candidate_id"]: c for c in cs if c["kind"] == "macro_definition"}
    return module.Reconciler(PATH, source.encode(), cs, list(occurrences), diagnostics, defs, file_record).run()


HELPER = "include/cutlass/detail/helper_macros.hpp"
HELPER_SOURCE = b"#ifdef CUTLASS_NAMESPACE\n#define concat_tok(a,b) a ## b\n#define mkcutlassnamespace(pre,ns) concat_tok(pre,ns)\n#define cutlass mkcutlassnamespace(cutlass_,CUTLASS_NAMESPACE)\n#endif\n"


def definition_record(path, data, name):
    view = module.SourceProof(path, data)
    d = next(d for d in view.scan.definitions if d["name"] == name)
    a, b = d["byte_range"]
    x, y = d["body_byte_range"]
    return {"path": path, "name": name, "start_byte": a, "end_byte": b - 1,
            "parameters": None if not d["function_like"] else ["".join(p) for p in view._comma_parts(d["parameter_tokens"])],
            "body": data[x:y].decode(),
            "preprocessor_conditions": [{"expression": p["expression"], "directive_line": p["line"]} for p in view.conditions_at(a)]}


def binding_fixture():
    source = "namespace cutlass { int f(); }"
    chain = [definition_record(HELPER, HELPER_SOURCE, name) for name in ("cutlass", "mkcutlassnamespace", "concat_tok")]
    key = HELPER + "::cutlass"
    records = []
    for choice, namespace in (("default_namespace", "cutlass"), ("custom_namespace", "<cutlass_namespace(CUTLASS_NAMESPACE)>")):
        resolution = {"status": "literal_macro_binding" if choice == "default_namespace" else "parameterized_macro_binding",
                      "spelling": "cutlass", "binding_key": key, "choice_key": choice, "definition_chain": copy.deepcopy(chain)}
        if choice == "default_namespace":
            resolution.update(expanded_namespace="cutlass", missing_bindings=[])
            guard = {"expression": "!defined(CUTLASS_NAMESPACE)", "directive_line": 1}
        else:
            resolution.update(scope_expression='rescan(token_paste("cutlass_", expand(CUTLASS_NAMESPACE)))', is_literal_namespace=False,
                              missing_bindings=["CUTLASS_NAMESPACE tokens and argument/result rescanning"], requirements=["Final result must be a valid namespace name"])
            guard = {"expression": "defined(CUTLASS_NAMESPACE)", "directive_line": 1}
        owner = occurrence(source, "namespace cutlass ", namespace, "namespace", end=len(source),
                           source_id="source_namespace_" + namespace, name_resolution=resolution, conditions=[guard], expanded_namespace_component=0)
        f = occurrence(source, "int f();", "f", "function", source_id="source_f", conditions=[guard],
                       id="instance_f_" + choice, entity_id="entity_f_" + choice, qualified_name=namespace + "::f",
                       qualified_name_resolution="source_name" if choice == "default_namespace" else "parameterized_macro_binding",
                       namespace_choices={key: choice}, namespace_conditions=[guard],
                       scope_chain=[{"name": namespace, "kind": "namespace", "entity_id": owner["entity_id"], "name_resolution": resolution}])
        records.extend((owner, f))
    return source, records


def bound_rows(source, records):
    return module.Reconciler(PATH, source.encode(), candidates(source), records,
                             source_reader=lambda path: HELPER_SOURCE if path == HELPER else (_ for _ in ()).throw(ValueError("unknown source"))).run()


def conditional_fixture():
    source = "using X =\n#if MODE\nint\n#else\nlong\n#endif\n;"
    view = module.SourceProof(PATH, source.encode())
    records = []
    for index, (segments, guards) in enumerate(view.conditional_projections(0, len(source))):
        virtual, parts, cursor = b"", [], 0
        for a, b in segments:
            payload = source.encode()[a:b]
            parts.append({"virtual_start_byte": cursor, "virtual_end_byte": cursor + len(payload),
                          "physical_span": {"path": PATH, "start_byte": a, "end_byte": b}, "mapping": "exact_source_slice"})
            cursor += len(payload)
            virtual += payload
        conditions = [{"expression": g["expression"], "directive_span": {"path": g["path"], "start_byte": g["span"][0], "end_byte": g["span"][1], "start_line": g["line"]}} for g in guards]
        target = "int" if index == 0 else "long"
        origin = {"source_declaration_id": "source_alias_X", "conditional_variant_id": "variant_" + str(index),
                  "kind": "alias", "name": "X", "declaration_span": {"path": PATH, "start_byte": 0, "end_byte": len(source)},
                  "semantic_spelling": source, "virtual_source": virtual.decode(), "segments": parts,
                  "conditions": conditions, "target_type": target, "status": "parsed", "parse_errors": []}
        records.append(occurrence(source, source, "X", "alias", id="alias_instance_" + str(index), source_id="source_alias_X",
                                  conditions=conditions, conditional_origin=origin, expanded_signature=virtual.decode().rstrip(), target_type=target))
    return source, records


def bitfield_fixture():
    source = "struct S { unsigned : 0; unsigned : 3; };"
    end_type = source.rindex("}") + 1
    owner = occurrence(source, "struct S ", "S", "struct_specifier", end=end_type, syntax=[0, end_type])
    records = [owner]
    for width in ("0", "3"):
        declaration = "unsigned : " + width + ";"
        start = source.index(declaration)
        colon = source.index(":", start)
        value = source.index(width, colon)
        field = lambda a, b: {"path": PATH, "start_byte": a, "end_byte": b, "raw": source[a:b]}
        location = lambda a, b: {"path": PATH, "start_byte": a, "end_byte": b}
        identifier = "bitfield_" + width
        origin = {"bitfield_source_id": identifier, "source_declaration_id": "declaration_" + width,
                  "declarator_index": 0, "kind": "anonymous_bitfield", "name": None, "name_range": None,
                  "declaration": field(start, start + len(declaration)), "declarator": field(colon, value + 1),
                  "colon_range": location(colon, colon + 1), "bit_width_range": location(value, value + 1), "bit_width": width,
                  "owner": {"kind": "struct_specifier", "name": "S", "range": location(0, end_type)},
                  "initializer": None, "initializer_range": None}
        records.append(occurrence(source, declaration, None, "member", source_id="source_" + identifier,
                                  is_bitfield=True, bitfield_source_id=identifier, bitfield_origin=origin,
                                  bit_width=width, colon_range=[colon, colon + 1], bit_width_range=[value, value + 1],
                                  declarator=[colon, value + 1], scope_chain=[{"name": "S", "kind": "struct_specifier", "entity_id": owner["entity_id"]}]))
    return source, records


def select(rows, source, text, kind="syntax_interval"):
    return next(r for r in rows if r["candidate_kind"] == kind
                and source.encode()[r["byte_range"][0]:r["byte_range"][1]].decode() == text)


class ReconciliationTests(unittest.TestCase):
    def test_directive_path_cannot_be_replaced_by_same_line_in_another_file(self):
        source, records = binding_fixture()
        records[3]["conditions"][0]["directive_path"] = "include/cute/wrong.hpp"
        self.assertEqual(select(bound_rows(source, records), source, "int f();")["status"], "pending")

    def test_new_conditional_function_form_is_explicit_pending_not_alias_or_macro(self):
        source = "void f();"
        f = occurrence(source, source, "f", "function", conditional_origin={"kind": "conditional_function_header", "source_declaration_id": "future_header"})
        row = select(reconcile(source, [f]), source, source)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["obligation"], "phase1_conditional_declaration_pending")
        self.assertIn("conditional_function_header", row["pending_reasons"][0])

    def test_anonymous_zero_width_bitfield_has_source_identity_not_invented_name(self):
        source, records = bitfield_fixture()
        rows = reconcile(source, records)
        for width in ("0", "3"):
            row = select(rows, source, "unsigned : " + width + ";")
            self.assertEqual(row["status"], "mapped_occurrences", row["pending_reasons"])
            self.assertEqual(row["bitfields"][0]["name"], None)
            self.assertEqual(row["bitfields"][0]["bit_width"], width)
        self.assertNotEqual(records[1]["bitfield_source_id"], records[2]["bitfield_source_id"])

    def test_deleted_bitfield_or_type_owner_stays_pending(self):
        source, records = bitfield_fixture()
        self.assertEqual(select(reconcile(source, [records[0], records[2]]), source, "unsigned : 0;")["status"], "pending")
        self.assertEqual(select(reconcile(source, records[1:]), source, "unsigned : 0;")["status"], "pending")

    def test_bitfield_wrong_kind_width_colon_or_invented_name_is_rejected(self):
        for mutation in ("kind", "width", "colon", "name", "initializer"):
            source, records = bitfield_fixture()
            field = records[1]
            if mutation == "kind": field["kind"] = "function"
            if mutation == "width": field["bit_width"] = "99"
            if mutation == "colon": field["colon_range"][0] += 1
            if mutation == "name": field["name"] = "invented"
            if mutation == "initializer": field["initializer"] = "0"
            self.assertEqual(select(reconcile(source, records), source, "unsigned : 0;")["status"], "pending", mutation)

    def test_alignas_attribute_does_not_turn_member_into_function(self):
        source = "alignas(16) Storage value;"
        member = occurrence(source, source, "value", "member", qualifiers=["alignas(16)"], declared_type="Storage")
        rows = reconcile(source, [member])
        self.assertEqual(select(rows, source, source)["status"], "mapped_occurrences")
        self.assertEqual(select(rows, source, "alignas(16)", "macro_invocation")["classification"], "alignment_attribute_expression")
        self.assertEqual(select(reconcile(source), source, "alignas(16)", "macro_invocation")["status"], "pending")
        wrong = copy.deepcopy(member); wrong["kind"] = "function"
        self.assertEqual(select(reconcile(source, [wrong]), source, "alignas(16)", "macro_invocation")["status"], "pending")

    def test_operator_name_candidate_maps_operator_not_ordinary_call(self):
        source = "int operator()(int x) const;"
        operator = occurrence(source, source, "operator()", "operator")
        self.assertEqual(select(reconcile(source, [operator]), source, "operator()", "macro_invocation")["status"], "mapped_occurrences")
        self.assertEqual(select(reconcile(source), source, "operator()", "macro_invocation")["status"], "pending")
        operator["kind"] = "member"
        self.assertEqual(select(reconcile(source, [operator]), source, "operator()", "macro_invocation")["status"], "pending")

    def test_alias_rhs_and_delimiter_fragment_need_exact_alias_owner(self):
        source = "using Layout = decltype(size(Shape{}));"
        alias = occurrence(source, source, "Layout", "alias", target_type="decltype(size(Shape{}))")
        rows = reconcile(source, [alias])
        self.assertEqual(select(rows, source, "));" )["classification"], "signature_delimiter_continuation")
        self.assertEqual(select(rows, source, "size(Shape{})", "macro_invocation")["classification"], "declaration_subexpression")
        self.assertEqual(select(reconcile(source), source, "));" )["status"], "pending")
        alias["kind"] = "variable"
        self.assertEqual(select(reconcile(source, [alias]), source, "));" )["status"], "pending")

    def test_braced_initializer_literal_fragment_is_not_a_new_api(self):
        source = "void* ptr{nullptr};"
        owner = occurrence(source, source, "ptr", "variable", initializer="{nullptr}", declared_type="void")
        self.assertEqual(select(reconcile(source, [owner]), source, "nullptr}")["classification"], "declaration_subexpression")
        self.assertEqual(select(reconcile(source), source, "nullptr}")["status"], "pending")
        owner["kind"] = "namespace"
        self.assertEqual(select(reconcile(source, [owner]), source, "nullptr}")["status"], "pending")

    def test_parameter_default_call_is_expression_not_declaration(self):
        source = "void f(int x = make());"
        parameter = {"range": [source.index("int"), source.rindex(")")], "name": "x", "type": "int", "default": "make()"}
        owner = occurrence(source, source, "f", "function", parameters=[parameter])
        self.assertEqual(select(reconcile(source, [owner]), source, "make()", "macro_invocation")["classification"], "declaration_subexpression")
        self.assertEqual(select(reconcile(source), source, "make()", "macro_invocation")["status"], "pending")
        owner["kind"] = "member"
        self.assertEqual(select(reconcile(source, [owner]), source, "make()", "macro_invocation")["status"], "pending")

    def test_constructor_initializers_are_not_comma_declarators(self):
        source = "S(int n): a(n), b(n) {}"
        header = source[:source.index("{")]
        owner = occurrence(source, header, "S", "constructor", declarator=[0, source.index(":")], body=[source.index("{"), len(source)])
        rows = reconcile(source, [owner])
        self.assertEqual(select(rows, source, header + "{")["status"], "mapped_occurrences")
        self.assertEqual(select(rows, source, "a(n)", "macro_invocation")["classification"], "declaration_subexpression")
        self.assertEqual(select(reconcile(source), source, "a(n)", "macro_invocation")["status"], "pending")
        owner["kind"] = "function"
        self.assertEqual(select(reconcile(source, [owner]), source, "a(n)", "macro_invocation")["status"], "pending")

    def test_constexpr_compatibility_macro_use_can_be_verified_specifier(self):
        source = "#define constexpr\nconstexpr int f();"
        f = occurrence(source, "constexpr int f();", "f", "function", qualifiers=["constexpr"])
        self.assertEqual(select(reconcile(source, [f]), source, "constexpr", "macro_invocation")["classification"], "language_declaration_specifier")
        self.assertEqual(select(reconcile(source), source, "constexpr", "macro_invocation")["status"], "pending")
        f["kind"] = "member"
        self.assertEqual(select(reconcile(source, [f]), source, "constexpr", "macro_invocation")["status"], "pending")

    def test_body_only_error_does_not_make_unrelated_statement_a_library_api(self):
        source = "void f() { g(); @; }"
        f = occurrence(source, "void f() ", "f", "function", body=[source.index("{"), len(source)], parse_status="contains_parse_error")
        at = source.index("@")
        diag = {"diagnostic_id": "body_error", "category": "parse_error", "start_byte": at, "end_byte": at + 1, "blocks_phase_1": True}
        rows = reconcile(source, [f], [diag])
        self.assertEqual(select(rows, source, "g();")["status"], "classified_non_api")
        self.assertEqual(select(rows, source, "@;")["status"], "pending")
        self.assertEqual(select(reconcile(source, [], [diag]), source, "g();")["status"], "pending")
        self.assertEqual(select(reconcile(source, [f]), source, "g();")["status"], "pending")
        diag["start_byte"], diag["end_byte"] = 0, 4
        self.assertEqual(select(reconcile(source, [f], [diag]), source, "g();")["status"], "pending")

    def test_expression_context_does_not_swallow_local_type_or_declaration_macro(self):
        source = "int value = [] { struct Local { int member; }; return 0; }();"
        owner = occurrence(source, source, "value", "variable", initializer="[] { struct Local { int member; }; return 0; }()")
        self.assertEqual(select(reconcile(source, [owner]), source, "int member;")["status"], "pending")
        source = "#define MAKE(X) struct X {};\nvoid f() { MAKE(Local); }"
        f = occurrence(source, "void f() ", "f", "function", body=[source.index("{", source.index("void")), len(source)])
        rows = reconcile(source, [f])
        self.assertEqual(select(rows, source, "MAKE(Local);")["status"], "pending")
        self.assertEqual(select(rows, source, "MAKE(Local)", "macro_invocation")["status"], "pending")

    def test_bound_namespace_instances_keep_source_and_instance_identities_separate(self):
        source, records = binding_fixture()
        rows = bound_rows(source, records)
        header = select(rows, source, "namespace cutlass {")
        call = select(rows, source, "int f();")
        self.assertEqual(header["status"], "mapped_occurrences", header["pending_reasons"])
        self.assertEqual(call["status"], "mapped_occurrences", call["pending_reasons"])
        self.assertEqual(call["source_occurrence_ids"], ["source_f"])
        self.assertEqual(len(call["occurrence_ids"]), 2)
        parameter = next(i for i in call["binding_instances"] if i["namespace_binding_kind"] == "parameterized")
        self.assertTrue(parameter["missing_bindings"])
        self.assertEqual(parameter["namespace_choices"][HELPER + "::cutlass"], "custom_namespace")

    def test_delete_one_namespace_instance_cannot_be_hidden_by_same_physical_range(self):
        source, records = binding_fixture()
        records = [r for r in records if r["id"] != "instance_f_custom_namespace"]
        result = select(bound_rows(source, records), source, "int f();")
        self.assertEqual(result["status"], "pending")
        self.assertIn("namespace_binding_instance_alternatives_incomplete", result["pending_reasons"])

    def test_namespace_binding_label_without_chain_is_not_a_proof(self):
        source, records = binding_fixture()
        for record in records:
            if record["kind"] == "namespace":
                record["name_resolution"]["definition_chain"] = []
        self.assertEqual(select(bound_rows(source, records), source, "int f();")["status"], "pending")

    def test_namespace_binding_body_guard_expression_and_missing_binding_are_checked(self):
        for mutation in ("definition_body", "definition_guard", "scope_expression", "missing_bindings", "choice", "namespace_conditions"):
            source, records = binding_fixture()
            custom = records[2]["name_resolution"]
            if mutation == "definition_body": custom["definition_chain"][2]["body"] = "a + b"
            if mutation == "definition_guard": custom["definition_chain"][0]["preprocessor_conditions"] = []
            if mutation == "scope_expression": custom["scope_expression"] = 'token_paste("different_", CUTLASS_NAMESPACE)'
            if mutation == "missing_bindings": custom["missing_bindings"] = []
            if mutation == "choice": records[3]["namespace_choices"] = {HELPER + "::cutlass": "default_namespace"}
            if mutation == "namespace_conditions": records[3]["namespace_conditions"] = []
            row = select(bound_rows(source, records), source, "int f();")
            self.assertEqual(row["status"], "pending", mutation)

    def test_external_binding_condition_is_required_but_not_a_local_branch(self):
        source, records = binding_fixture()
        row = select(bound_rows(source, records), source, "int f();")
        self.assertEqual(row["conditions"], [])
        self.assertEqual(row["status"], "mapped_occurrences")
        records[3]["conditions"] = []
        self.assertEqual(select(bound_rows(source, records), source, "int f();")["status"], "pending")

    def test_parameterized_label_cannot_forge_qualified_name(self):
        source, records = binding_fixture()
        records[3]["qualified_name"] = "invented::f"
        self.assertEqual(select(bound_rows(source, records), source, "int f();")["status"], "pending")

    def test_conditional_alias_maps_piecewise_source_without_macro_origin(self):
        source, records = conditional_fixture()
        rows = reconcile(source, records)
        header = select(rows, source, "using X =")
        self.assertEqual(header["status"], "mapped_occurrences", header["pending_reasons"])
        self.assertEqual(header["obligation"], "phase1_conditional_declaration")
        self.assertEqual(header["source_occurrence_ids"], ["source_alias_X"])
        self.assertEqual(len(header["occurrence_ids"]), 2)
        self.assertEqual(len(select(rows, source, "int")["occurrence_ids"]), 1)
        self.assertEqual(len(select(rows, source, "long")["occurrence_ids"]), 1)

    def test_deleted_conditional_variant_cannot_hide_behind_full_statement_span(self):
        source, records = conditional_fixture()
        row = select(reconcile(source, records[:1]), source, "using X =")
        self.assertEqual(row["status"], "pending")
        self.assertIn("conditional_source_branch_variants_incomplete", row["pending_reasons"])

    def test_conditional_source_piece_and_guard_tampering_is_rejected(self):
        for mutation in ("physical_piece", "virtual_text", "condition", "target", "source_text"):
            source, records = conditional_fixture()
            origin = records[0]["conditional_origin"]
            if mutation == "physical_piece": origin["segments"][1]["physical_span"]["start_byte"] += 1
            if mutation == "virtual_text": origin["virtual_source"] = origin["virtual_source"].replace("int", "float")
            if mutation == "condition": origin["conditions"][0]["expression"] = "OTHER_MODE"
            if mutation == "target": records[0]["target_type"] = "float"
            if mutation == "source_text": origin["semantic_spelling"] = "using X = int;"
            self.assertEqual(select(reconcile(source, records), source, "using X =")["status"], "pending", mutation)

    def test_binding_instance_id_collision_and_source_id_retargeting_are_rejected(self):
        source, records = binding_fixture()
        records[3]["id"] = records[1]["id"]
        with self.assertRaisesRegex(ValueError, "instance ID"):
            bound_rows(source, records)
        source, records = binding_fixture()
        records[3]["range"] = [0, len(source)]
        with self.assertRaisesRegex(ValueError, "source occurrence ID"):
            bound_rows(source, records)

    def test_boolean_condition_comparison_does_not_erase_precedence(self):
        self.assertEqual(module.condition_form("!((defined(MODE)))"), module.condition_form("!defined(MODE)"))
        self.assertNotEqual(module.condition_form("A && (B || C)"), module.condition_form("(A && B) || C"))

    def test_enclosing_namespace_cannot_cover_missing_declaration(self):
        source = "namespace n { int a; int b; }"
        ns = occurrence(source, "namespace n ", "n", "namespace", end=len(source))
        a = occurrence(source, "int a;", "a", "variable")
        rows = reconcile(source, [ns, a])
        self.assertEqual(select(rows, source, "int a;")["status"], "mapped_occurrences")
        self.assertEqual(select(rows, source, "int b;")["status"], "pending")
        self.assertEqual(select(rows, source, "int b;")["occurrence_ids"], [])

    def test_class_full_body_range_is_not_declaration_signature(self):
        source = "struct S { int missing; };"
        cls = occurrence(source, "struct S ", "S", "struct_specifier", end=len(source))
        rows = reconcile(source, [cls])
        self.assertEqual(select(rows, source, "int missing;")["status"], "pending")

    def test_forged_whole_namespace_signature_still_cannot_cover_children(self):
        source = "namespace n { int missing; }"
        fake = occurrence(source, source, "n", "namespace")
        rows = reconcile(source, [fake])
        self.assertEqual(select(rows, source, "int missing;")["status"], "pending")
        self.assertEqual(select(rows, source, "namespace n {")["status"], "pending")

    def test_large_initializer_signature_cannot_cover_local_type_member(self):
        source = "int member = [] { struct Local { int member; }; return 0; }();"
        outer = occurrence(source, source, "member", "variable")
        rows = reconcile(source, [outer])
        self.assertEqual(select(rows, source, "int member;")["status"], "pending")

    def test_wrong_kind_reference_return_cannot_be_marked_mapped(self):
        source = "S& f(int);"
        wrong = occurrence(source, source, "f", "member")
        self.assertEqual(select(reconcile(source, [wrong]), source, source)["status"], "pending")
        right = occurrence(source, source, "f", "function")
        self.assertEqual(select(reconcile(source, [right]), source, source)["status"], "mapped_occurrences")

    def test_shared_statement_range_cannot_hide_deleted_comma_declarator(self):
        source = "int a,b;"
        a = occurrence(source, source, "a", "variable", declarator=[4, 5])
        b = occurrence(source, source, "b", "variable", declarator=[6, 7])
        good = select(reconcile(source, [a, b]), source, source)
        self.assertEqual(good["status"], "mapped_occurrences")
        for surviving in (a, b):
            bad = select(reconcile(source, [surviving]), source, source)
            self.assertEqual(bad["status"], "pending")
        a["declarator"] = None
        b["declarator"] = None
        self.assertEqual(select(reconcile(source, [a, b]), source, source)["status"], "pending")

    def test_same_line_two_statements_require_two_occurrences(self):
        source = "int a; int b;"
        rows = reconcile(source, [occurrence(source, "int a;", "a", "variable")])
        self.assertEqual(select(rows, source, "int b;")["status"], "pending")

    def test_condition_branch_cannot_be_satisfied_by_other_branch(self):
        source = "#if MODE\nint f();\n#else\nint f(int);\n#endif\n"
        first = occurrence(source, "int f();", "f", "function", condition_lines=[1])
        wrong_branch = occurrence(source, "int f(int);", "f", "function", condition_lines=[1])
        rows = reconcile(source, [first, wrong_branch])
        self.assertEqual(select(rows, source, "int f();")["status"], "mapped_occurrences")
        self.assertEqual(select(rows, source, "int f(int);")["status"], "pending")

    def test_if_zero_candidate_is_not_removed(self):
        source = "#if 0\nint never();\n#endif\n"
        cs = candidates(source)
        rows = reconcile(source, cs=cs)
        self.assertEqual(len(rows), len(cs))
        self.assertEqual(select(rows, source, "int never();")["status"], "pending")

    def test_renaming_diagnostic_symbolic_does_not_fill_missing_api(self):
        source = "int missing();"
        diagnostic = {"diagnostic_id": "diag_fake", "category": "symbolic", "start_byte": 0,
                      "end_byte": len(source), "blocks_phase_1": False}
        rows = reconcile(source, diagnostics=[diagnostic])
        self.assertEqual(select(rows, source, source)["status"], "pending")

    def test_parse_status_and_raw_signature_are_verified(self):
        source = "int f();"
        for field, bad in (("parse_status", "contains_parse_error"), ("raw_signature_sha256", "bad"), ("name", "wrong_name")):
            o = occurrence(source, source, "f", "function")
            o[field] = bad
            self.assertEqual(select(reconcile(source, [o]), source, source)["status"], "pending")

    def test_macro_dependent_namespace_is_pending_even_with_occurrence(self):
        source = "namespace API_NS { int x; }"
        ns = occurrence(source, "namespace API_NS ", "API_NS", "namespace", end=len(source), name_resolution={"status": "macro_dependent"})
        x = occurrence(source, "int x;", "x", "variable", qualified_name_resolution="macro_dependent")
        rows = reconcile(source, [ns, x])
        self.assertEqual(select(rows, source, "namespace API_NS {")["status"], "pending")
        self.assertEqual(select(rows, source, "int x;")["status"], "pending")

    def test_function_body_call_is_phase2_not_declaration_gap(self):
        source = "void f() { g(); int local; }"
        o = occurrence(source, "void f() ", "f", "function", body=[source.index("{"), len(source)])
        rows = reconcile(source, [o])
        call = select(rows, source, "g();")
        self.assertEqual(call["status"], "classified_non_api")
        self.assertEqual(call["obligation"], "phase2_expression")
        self.assertEqual(call["relationship_status"], "pending_phase2")
        self.assertEqual(select(rows, source, "int local;")["status"], "classified_non_api")

    def test_missing_local_class_and_member_not_hidden_as_local_expression(self):
        source = "void f() { struct Local { int member; }; }"
        f = occurrence(source, "void f() ", "f", "function", body=[source.index("{"), len(source)])
        rows = reconcile(source, [f])
        self.assertEqual(select(rows, source, "struct Local {")["status"], "pending")
        self.assertEqual(select(rows, source, "int member;")["status"], "pending")

    def test_bad_body_extent_does_not_authorize_non_api_classification(self):
        source = "void f() { g(); } int outside;"
        f = occurrence(source, "void f() ", "f", "function", body=[source.index("{"), len(source)])
        rows = reconcile(source, [f])
        self.assertEqual(select(rows, source, "g();")["status"], "pending")
        self.assertEqual(select(rows, source, "int outside;")["status"], "pending")

    def test_includes_and_condition_macros_keep_separate_classifications(self):
        source = "#include CUDA_STD_HEADER(tuple)\n#if defined(CUDA_ARCH_FAMILY) && (CUDA_ARCH_FAMILY(1000) || CUDA_ARCH_FAMILY(1010))\n#endif\n#include <vector>\n"
        rows = reconcile(source)
        includes = [r for r in rows if r["candidate_kind"] == "include_dependency"]
        self.assertEqual([r["obligation"] for r in includes], ["phase2_file_dependency"] * 2)
        self.assertEqual(includes[0]["expression_macro"], "CUDA_STD_HEADER")
        self.assertEqual(includes[1]["include_classification"], "external")
        self.assertEqual(sum(r.get("condition_macro_uses", {}).get("CUDA_ARCH_FAMILY", 0) for r in rows), 2)

    def test_comments_and_strings_do_not_become_api_candidates(self):
        source = '// struct Fake {};\nchar const* s="struct Other {};";'
        rows = reconcile(source)
        self.assertFalse(any('Fake' in r["evidence"]["excerpt"] for r in rows if r["candidate_kind"] == "syntax_interval"))
        self.assertEqual(sum(r["candidate_kind"] == "syntax_interval" for r in rows), 1)

    def test_deleted_generated_operator_stays_pending(self):
        source = "#define OP(S) template <auto T> int operator S() {return 0;}\nOP(+);\nOP(-);\n"
        cs = candidates(source)
        definition = next(c for c in cs if c["kind"] == "macro_definition")
        invs = [c for c in cs if c["kind"] == "macro_invocation" and c.get("origin") == "source"]
        generated = []
        for i, invocation in enumerate(invs):
            a, b = invocation["byte_range"]
            o = occurrence(source, source[a:b], "operator " + ("+" if i == 0 else "-"), "operator")
            o["macro_origin"] = {"invocation": {"start_byte": a, "end_byte": b}, "definition": {
                "path": PATH, "name": "OP", "start_byte": definition["byte_range"][0], "end_byte": definition["byte_range"][1]}}
            generated.append(o)
        complete = reconcile(source, generated, cs=cs)
        self.assertTrue(all(select(complete, source, source[c["byte_range"][0]:c["byte_range"][1]], "macro_invocation")["status"] == "mapped_occurrences" for c in invs))
        missing = reconcile(source, generated[:1], cs=cs)
        self.assertEqual(select(missing, source, "OP(-)", "macro_invocation")["status"], "pending")
        self.assertEqual(select(missing, source, "OP(-);")["status"], "pending")

    def test_duplicate_candidate_id_is_rejected(self):
        source = "int x;"
        cs = candidates(source)
        cs.append(copy.deepcopy(cs[-1]))
        with self.assertRaisesRegex(ValueError, "cardinality/identity"):
            reconcile(source, cs=cs)

    def test_candidate_hash_tampering_is_rejected(self):
        source = "int x;"
        cs = candidates(source)
        cs[-1]["raw_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "range/hash"):
            reconcile(source, cs=cs)

    def test_candidate_deletion_cannot_redefine_frozen_denominator(self):
        source = "int first; int missing;"
        cs = candidates(source)
        module.verify_frozen_candidates(PATH, source.encode(), cs, {}, {PATH})
        damaged = [c for c in cs if source[c["byte_range"][0]:c["byte_range"][1]] != "int missing;"]
        with self.assertRaisesRegex(ValueError, "differs from source rescan"):
            module.verify_frozen_candidates(PATH, source.encode(), damaged, {}, {PATH})

    def test_macro_owner_pending_does_not_make_ordinary_local_statement_a_library_api(self):
        source = "void f() { g(); }"
        f = occurrence(source, "void f() ", "f", "function", body=[source.index("{"), len(source)], qualified_name_resolution="macro_dependent")
        rows = reconcile(source, [f])
        self.assertEqual(select(rows, source, "void f() {")["status"], "pending")
        self.assertEqual(select(rows, source, "g();")["status"], "classified_non_api")
        self.assertEqual(select(rows, source, "g();")["obligation"], "phase2_expression")

    def test_stream_reader_handles_small_chunks_nested_arrays_and_escaped_keys(self):
        data = {"count": 123456789, "ignored": [{"items": ['"occurrences": [', {"x": "中文"}]}],
                "occurrences": [{"path": "a", "value": 'escaped \\" text'}, {"path": "b"}], "done": True}
        with tempfile.TemporaryDirectory(prefix="reconcile-stream-") as temp:
            path = Path(temp) / "input.json"
            path.write_text(json.dumps(data, ensure_ascii=False))
            observed = list(module.top_items(path, {"occurrences"}, chunk_size=3))
        self.assertEqual([v for k, v, item in observed if item], data["occurrences"])
        self.assertEqual(dict((k, v) for k, v, item in observed if not item), {"count": 123456789, "done": True})

    def test_end_to_end_preserves_every_candidate_and_does_not_claim_phase_pass(self):
        source = "int x;"
        with tempfile.TemporaryDirectory(prefix="reconcile-build-") as temp:
            root = Path(temp)
            (root / "data").mkdir()
            snapshot = root / "snapshot" / PATH
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text(source)
            scope = {"commit": "fixture", "file_count": 1, "files": [{"path": PATH}]}
            scope_bytes = json.dumps(scope).encode()
            (root / "data/scope.json").write_bytes(scope_bytes)
            ledger = module.LEXER.scan_sources({PATH: source.encode()}, "fixture", module.sha(scope_bytes))
            ledger["scanner_sha256"] = module.file_sha(Path(module.LEXER.__file__))
            (root / "data/candidates.json").write_text(json.dumps(ledger))
            signature = {"start_byte": 0, "end_byte": len(source)}
            o = {"declaration_occurrence_id": "occ_x", "entity_id": "ent_x", "kind": "variable", "name": "x", "qualified_name": "x", "path": PATH,
                 **signature, "signature_range": signature, "syntax_node_range": signature,
                 "declarator_range": {"start_byte": 4, "end_byte": 5}, "raw_signature": source, "parse_status": "parsed"}
            declarations = {"commit": "fixture", "is_subset": False, "entities": [], "occurrences": [o], "diagnostics": [],
                            "files": [{"path": PATH, "sha256": module.sha(source.encode()), "macro_expansions": []}]}
            (root / "data/declarations.json").write_text(json.dumps(declarations))
            with patch("builtins.print"):
                checks = module.build(root)
            result = json.loads((root / "data/candidate-mapping.json").read_bytes())
            self.assertEqual({r["candidate_id"] for r in result["mappings"]}, {c["candidate_id"] for c in ledger["candidates"]})
            self.assertFalse(checks["phase_1_passed"])
            self.assertTrue(checks["all_candidates_reproduced_from_source_and_frozen_scanner"])
            self.assertEqual(checks["statuses"]["mapped_occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
