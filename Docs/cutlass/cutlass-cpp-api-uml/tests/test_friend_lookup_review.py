"""Independent Clang-backed boundary checks for friend type name lookup."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("friend_lookup_review_extractor", ROOT / "scripts/extract_declarations.py")
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


class FriendLookupIndependentReview(unittest.TestCase):
    def extract(self, source):
        result = extractor.Extractor("friend-lookup-independent-review")
        result.extract("fixture.hpp", source.encode())
        return result

    def clang_friends(self, source):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        result = subprocess.run([compiler, "-std=c++17", "-pedantic-errors", "-Xclang", "-ast-dump=json",
                                 "-fsyntax-only", "-x", "c++", "-"], input=source, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        found = []
        def walk(node):
            if node.get("kind") == "FriendDecl" and "type" in node:
                found.append(node["type"].get("desugaredQualType", node["type"]["qualType"]))
            for child in node.get("inner", []):
                walk(child)
        walk(json.loads(result.stdout))
        return found

    def friend_occurrences(self, result):
        return [o for o in result.occurrences if o["friend"]]

    def assert_owner_or_explicit_pending(self, result, expected):
        if result.diagnostics:
            self.assertTrue(any(d["blocks_phase_1"] for d in result.diagnostics))
            return
        self.assertEqual({o["qualified_name"] for o in self.friend_occurrences(result)}, expected)

    def test_using_declaration_makes_existing_type_visible(self):
        source = "namespace a {struct F {}; } namespace b {using a::F; struct S {friend struct F;};}"
        self.assertEqual(self.clang_friends(source), ["a::F"])
        result = self.extract(source)
        self.assert_owner_or_explicit_pending(result, {"a::F"})

    def test_using_directive_does_not_behave_like_using_declaration(self):
        source = "namespace a {struct F {}; } namespace b {using namespace a; struct S {friend struct F;};}"
        self.assertEqual(self.clang_friends(source), ["b::F"])
        result = self.extract(source)
        self.assertEqual({o["qualified_name"] for o in self.friend_occurrences(result)}, {"b::F"})
        self.assertFalse(result.diagnostics)

    def test_visible_base_nested_type_not_new_namespace_type(self):
        source = "struct Base {struct F {};}; struct S:Base {friend struct F;};"
        self.assertEqual(self.clang_friends(source), ["Base::F"])
        result = self.extract(source)
        self.assert_owner_or_explicit_pending(result, {"Base::F"})

    def test_dependent_base_not_searched_as_a_known_existing_type(self):
        source = "template<class T> struct S:T {friend struct F;};"
        self.assertEqual(self.clang_friends(source), ["F"])
        result = self.extract(source)
        # This extra boundary need not force a broader lookup implementation in
        # this checkpoint. A source-preserving pending reference is acceptable;
        # falsely resolving an existing dependent-base member is not.
        self.assert_owner_or_explicit_pending(result, {"F"})
        if result.diagnostics:
            self.assertTrue(all(o["kind"] == "friend_type_reference" and o.get("lookup_status") == "extraction_pending"
                                for o in self.friend_occurrences(result)))

    def test_inactive_nested_type_does_not_capture_friend_lookup(self):
        source = "struct S {\n#if 0\nstruct F {};\n#endif\nfriend struct F;}; struct F {};"
        self.assertEqual(self.clang_friends(source), ["F"])
        result = self.extract(source)
        # The inactive source F must remain inventoried, but it is not visible
        # to the subsequent active friend declaration.
        self.assertTrue(any(o["qualified_name"] == "S::F" and any(c.get("constant_false") for c in o["preprocessor_conditions"]) for o in result.occurrences))
        self.assert_owner_or_explicit_pending(result, {"F"})

    def test_conditional_prior_type_requires_owner_variants_or_pending(self):
        source = "struct S {\n#if A\nstruct F {};\n#endif\nfriend struct F;}; struct F {};"
        self.assertEqual(self.clang_friends("#define A 1\n" + source), ["S::F"])
        self.assertEqual(self.clang_friends("#define A 0\n" + source), ["F"])
        result = self.extract(source)
        self.assert_owner_or_explicit_pending(result, {"S::F", "F"})
        if not result.diagnostics:
            self.assertTrue(all(o["preprocessor_conditions"] for o in self.friend_occurrences(result)))

    def test_existing_nested_type_keeps_enclosing_template_identity(self):
        source = "template<class T> struct S {struct F {};friend struct F;};"
        self.assertEqual(self.clang_friends(source), ["S::F"])
        result = self.extract(source)
        fs = [o for o in result.occurrences if o["name"] == "F"]
        self.assertEqual(len(fs), 2)
        self.assertEqual(len({o["entity_id"] for o in fs}), 1)
        self.assertFalse(result.diagnostics)

    def test_simple_alias_friend_is_reference_not_new_class(self):
        source = "struct Real {}; using Alias=Real; struct S {private:friend Alias;};"
        self.assertEqual(self.clang_friends(source), ["Real"])
        result = self.extract(source)
        refs = [o for o in result.occurrences if o["friend"]]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["kind"], "friend_type_reference")
        self.assertEqual(refs[0]["target_type"], "Alias")
        self.assertFalse(any(o["kind"] in extractor.TYPE_NODES and o["name"] == "Alias" for o in result.occurrences))

    def test_qualified_type_friend_is_reference_not_new_declaration(self):
        source = "namespace n {struct F {}; } struct S {friend class n::F;};"
        self.assertEqual(self.clang_friends(source), ["n::F"])
        result = self.extract(source)
        refs = [o for o in result.occurrences if o["friend"]]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["kind"], "friend_type_reference")
        self.assertEqual(refs[0]["target_type"], "n::F")
        self.assertEqual(sum(o["kind"] in extractor.TYPE_NODES and o["qualified_name"] == "n::F" for o in result.occurrences), 1)


if __name__ == "__main__":
    unittest.main()
