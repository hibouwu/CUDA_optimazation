"""Adversarial tests for the independent lexical ledger, not API coverage tests."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("candidate_scanner", ROOT / "scripts/scan_candidates.py")
scanner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scanner
SPEC.loader.exec_module(scanner)


def scan(text, path="include/cute/fixture.hpp", extra=None):
    sources = {path: text.encode("utf-8")}
    sources.update(extra or {})
    return scanner.scan_sources(sources)


def kind(ledger, name):
    return [c for c in ledger["candidates"] if c["kind"] == name]


def raw(candidate, source):
    a, b = candidate["byte_range"]
    return source.encode()[a:b].decode()


class CandidateTests(unittest.TestCase):
    def test_comments_literals_and_raw_literals_cannot_make_fake_identifiers(self):
        source = '''// #define FAKE(x) struct Lost {};
/* namespace false_ns { FAKE(abc); } */
char const* a = "MACRO(fake); namespace not_real {}";
char const* b = R"tag(#if 0\nHIDDEN(1); " and } ;)tag";
char quote = '\\'';
struct Actual { int member; };
'''
        result = scan(source)
        self.assertFalse(kind(result, "macro_definition"))
        self.assertFalse(kind(result, "preprocessor_directive"))
        self.assertFalse(kind(result, "macro_invocation"))
        self.assertFalse(result["files"][0]["diagnostics"])
        self.assertTrue(any("Actual" in raw(c, source) for c in kind(result, "syntax_interval")))

    def test_line_spliced_comment_and_macro_definition(self):
        source = "// comment \\\nFAKE(1)\n#define MAKE(x) \\\nstruct x { int a; };\nMAKE(A)\n"
        result = scan(source)
        self.assertEqual([m["name"] for m in kind(result, "macro_definition")], ["MAKE"])
        self.assertEqual([m["name"] for m in kind(result, "macro_invocation")], ["MAKE"])
        self.assertIn("struct", kind(result, "macro_definition")[0]["declaration_generation_hints"])

    def test_every_branch_and_if_zero_are_retained(self):
        source = "#if 0\nint disabled;\n#if FLAG\nint nested;\n#endif\n#elif MODE == 2\nint second;\n#else\nint third;\n#endif\n"
        result = scan(source)
        intervals = kind(result, "syntax_interval")
        self.assertEqual(len(intervals), 4)
        self.assertEqual([i["constant_false"] for i in intervals], [True, True, False, False])
        self.assertEqual([len(i["conditions"]) for i in intervals], [1, 2, 1, 1])
        branches = {c["candidate_id"]: c for c in kind(result, "preprocessor_branch")}
        self.assertTrue(all(ref in branches for c in result["candidates"] for ref in c["conditions"]))
        self.assertIn("MODE == 2", branches[intervals[3]["conditions"][0]]["predicate"])

    def test_namespace_nesting_multiple_declarations_and_comma_declarators(self):
        source = "namespace a { namespace b::c { int x; int y, z; } }"
        result = scan(source)
        intervals = kind(result, "syntax_interval")
        self.assertEqual(len(intervals), 6)
        self.assertEqual(sum("namespace" in c["keyword_hints"] for c in intervals), 2)
        self.assertEqual(sum(len(c["comma_byte_offsets"]) for c in intervals), 1)
        self.assertEqual(sum(c["token_count"] for c in intervals), result["files"][0]["lexical_accounting"]["all_tokens"])

    def test_all_named_macro_definitions_and_lowercase_unknown_candidates(self):
        source = "#if A\n#define MAKE(x) struct x {};\n#else\n#define MAKE(x) using x = int;\n#endif\nMAKE(T)\nexternal_lowercase_macro(U)\n"
        result = scan(source)
        invocation = next(c for c in kind(result, "macro_invocation") if c["name"] == "MAKE")
        self.assertEqual(len(invocation["definition_candidates"]), 2)
        self.assertEqual(invocation["resolution"], "lexical_spelling_only")
        self.assertTrue(any(c["name"] == "external_lowercase_macro" for c in kind(result, "macro_invocation")))

    def test_macro_body_nested_generation_is_not_discarded(self):
        source = "#define INNER(x) struct x {};\n#define OUTER(x) INNER(x)\nOUTER(A)\n"
        result = scan(source)
        nested = [c for c in kind(result, "macro_invocation") if c["name"] == "INNER"]
        self.assertEqual(len(nested), 1)
        self.assertEqual(nested[0]["origin"], "macro_body")
        self.assertTrue(nested[0]["owner_candidate_id"])

    def test_utf8_positions_are_bytes(self):
        source = "// 中文\nstruct X { int y; };\n"
        result = scan(source)
        candidate = kind(result, "syntax_interval")[0]
        self.assertEqual(candidate["byte_range"][0], source.encode().index(b"struct"))
        self.assertEqual(candidate["line_range"], [2, 2])
        self.assertEqual(raw(candidate, source), "struct X {")

    def test_numeric_token_cannot_swallow_adjacent_macro_name(self):
        source = "#define VALUE 2\n#define MACRO(x) (x)\nint x = 1+VALUE; int y = 1e+2; int z = 1'000; int w = 0-MACRO(X); float v = 1e-2;\n"
        result = scan(source)
        invocations = kind(result, "macro_invocation")
        self.assertEqual([c["name"] for c in invocations], ["VALUE", "MACRO"])
        self.assertTrue(all(not c["is_confirmed_macro"] for c in invocations))
        numbers = [t.text for t in scanner.lex(source.encode())[0] if t.kind == "number"]
        self.assertEqual(numbers, ["2", "1", "1e+2", "1'000", "0", "1e-2"])
        self.assertFalse(result["files"][0]["diagnostics"])

    def test_translation_phase_line_splicing_preserves_original_offsets(self):
        source = "#define MA\\\nKE\\\n(x) struct x {};\nMA\\\nKE(T)\n/\\\n/ FALSE(1)\n"
        result = scan(source)
        definition = kind(result, "macro_definition")[0]
        self.assertEqual(definition["name"], "MAKE")
        self.assertTrue(definition["function_like"])
        invocations = kind(result, "macro_invocation")
        self.assertEqual([c["name"] for c in invocations], ["MAKE"])
        self.assertEqual(raw(invocations[0], source), "MA\\\nKE(T)")

    def test_relative_include_normalizes_without_reading_outside_scope(self):
        result = scan('#include "../target.hpp"\n', path="include/cute/sub/fixture.hpp",
                      extra={"include/cute/target.hpp": b""})
        dependency = kind(result, "include_dependency")[0]
        self.assertEqual(dependency["classification"], "in_scope")
        self.assertEqual(dependency["target_path"], "include/cute/target.hpp")

    def test_standard_external_missing_generated_and_tools_are_distinct(self):
        headers = ["vector", "cuda_runtime.h", "cute/exists.hpp", "cutlass/util/packed_stride.hpp", "cutlass/arch/array.h", "cutlass/arch/numeric_types.h", "cutlass/version_extended.h"]
        source = "".join(f"#include <{h}>\n" for h in headers) + "#include GENERATED_HEADER\n"
        result = scan(source, extra={"include/cute/exists.hpp": b""})
        dependencies = kind(result, "include_dependency")
        by_name = {c["header"]: c["classification"] for c in dependencies}
        self.assertEqual(by_name, {"vector": "external", "cuda_runtime.h": "external", "cute/exists.hpp": "in_scope", "cutlass/util/packed_stride.hpp": "source_outside_scope", "cutlass/arch/array.h": "missing_source", "cutlass/arch/numeric_types.h": "missing_source", "cutlass/version_extended.h": "conditionally_generated", None: "unresolved_expression"})

    def test_malformed_input_has_visible_diagnostics(self):
        result = scan("#else\nint x;\n#if A\n/* unclosed")
        names = {d["kind"] for d in result["files"][0]["diagnostics"]}
        self.assertEqual(names, {"unmatched_preprocessor_branch", "unterminated_preprocessor_conditional", "unterminated_comment"})

    def test_determinism_and_candidate_identity(self):
        first = scan("struct X { int x; int f(); };\n")
        second = scan("struct X { int x; int f(); };\n")
        self.assertEqual(first, second)
        ids = [c["candidate_id"] for c in first["candidates"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(c["mapping_status"] == "pending" for c in first["candidates"]))
        self.assertEqual(first["summary"]["api_coverage_status"], "not_claimed")

    def create_temporary_scope(self, root):
        path = "include/cute/fixture.hpp"
        source = b"struct X { int x; };\n"
        target = root / "snapshot" / path
        target.parent.mkdir(parents=True)
        target.write_bytes(source)
        (root / "data").mkdir()
        manifest = {"commit": "fixture", "file_count": 1,
                    "files": [{"path": path, "sha256": scanner.digest(source)}]}
        (root / "data/scope.json").write_text(json.dumps(manifest))
        return target

    def test_frozen_ledger_rescan_rejects_deleted_candidate_and_implicit_replacement(self):
        with tempfile.TemporaryDirectory(prefix="cutlass-candidates-test-") as temp:
            root = Path(temp)
            self.create_temporary_scope(root)
            with patch("builtins.print"):
                scanner.build(root)
                scanner.build(root, check=True)
            output = root / "data/candidates.json"
            damaged = json.loads(output.read_bytes())
            damaged["candidates"].pop()
            output.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "differs from independent rescan"):
                scanner.build(root, check=True)
            with self.assertRaisesRegex(ValueError, "Refusing to mutate frozen"):
                scanner.build(root)

    def test_changed_snapshot_is_rejected_before_candidate_scan(self):
        with tempfile.TemporaryDirectory(prefix="cutlass-candidates-test-") as temp:
            root = Path(temp)
            target = self.create_temporary_scope(root)
            target.write_bytes(b"struct Different {};\n")
            with self.assertRaisesRegex(ValueError, "Snapshot SHA-256 differs"):
                scanner.build(root)

    def test_fixed_source_5_unary_18_binary_macro_invocations(self):
        path = "include/cute/numeric/integral_constant.hpp"
        result = scanner.scan_sources({path: (ROOT / "snapshot" / path).read_bytes()})
        invocations = [c for c in kind(result, "macro_invocation") if c["origin"] == "source"]
        self.assertEqual(sum(c["name"] == "CUTE_LEFT_UNARY_OP" for c in invocations), 5)
        self.assertEqual(sum(c["name"] == "CUTE_BINARY_OP" for c in invocations), 18)
        self.assertFalse(result["files"][0]["diagnostics"])


if __name__ == "__main__":
    unittest.main()
