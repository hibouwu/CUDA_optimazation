"""Independent canonical-join mutations; never write stage/module artifacts.

Use the real enrichment/stager, but replace its ledger stream with in-memory
records from the four small relevant fixed headers. dump is always captured.
These tests guard the join contract, not all parser semantics or full-library
completeness. They do not load or rewrite the multi-gigabyte canonical ledger.
"""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_atlas as atlas
import extract_declarations
import stage_cute_alignment as stage

A='include/cute/container/alignment.hpp'
R='include/cute/container/array_aligned.hpp'

class AlignmentStageIndependentReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[]
        for path in (A,R,'include/cute/container/array.hpp','include/cute/numeric/math.hpp'):
            extractor=extract_declarations.Extractor('8f50b052e1099fb982392a622caab69b97b63128')
            extractor.extract(path,(ROOT/'snapshot'/path).read_bytes())
            cls.records.extend(extractor.occurrences)

    def target(self,records):
        return next(o for o in records if o['path']==A and o['kind']=='struct_specifier'and o['signature_range']['start_line']==67)

    def run_stage(self,mutation=None):
        records=copy.deepcopy(self.records)
        if mutation:mutation(records)
        captured=[]
        def items(*args,**kwargs):return (('occurrences',o,True)for o in records)
        def fingerprint(path):
            if Path(path)==ROOT/'data/declarations.json':return 'read-only-memory-ledger-fixture'
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        with patch.object(atlas,'top_items',items),patch.object(stage,'dump',lambda p,d:captured.append((str(p),copy.deepcopy(d)))),\
             patch.object(stage,'file_sha',fingerprint),contextlib.redirect_stdout(io.StringIO()):
            stage.main()
        self.assertEqual(len(captured),2)
        return captured[0][1]

    def rejected(self,mutation):
        with self.assertRaises(ValueError):self.run_stage(mutation)

    def test_positive_retains_two_namespace_positions_and_zero_manual(self):
        output=self.run_stage()
        self.assertEqual(output['canonical_integration']['physical_declarations_linked'],16)
        self.assertEqual(output['canonical_integration']['current_manual_overrides'],0)
        ns=next(n for n in output['nodes']if n['id']=='alignment.namespace.cute')['source_declaration_occurrences']
        self.assertEqual({o['path'] for o in ns},{A,R})
        self.assertEqual(len({o['declaration_occurrence_id']for o in ns}),2)
        self.assertEqual(len({o['entity_id']for o in ns}),1)

    def test_wrong_type_owner_rejected(self):
        self.rejected(lambda rs:self.target(rs).update(qualified_name='cute::Wrong::aligned_struct<128,Child>',entity_id='wrong'))

    def test_missing_entire_alignment_rejected(self):
        self.rejected(lambda rs:self.target(rs).update(alignment_specifiers=[]))

    def test_original_missing_cuda_branch_counterexample_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0]['conditional_expansions'].pop(0))

    def test_original_wrong_attribute_owner_counterexample_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0].update(owner_entity_id='wrong',owner_occurrence_id='wrong'))

    def test_original_missing_function_declattr_counterexample_rejected(self):
        self.rejected(lambda rs:next(o for o in rs if o['name']=='is_byte_aligned').update(attributes=[],qualifiers=[]))

    def test_original_wrong_namespace_owner_counterexample_rejected(self):
        self.rejected(lambda rs:next(o for o in rs if o['kind']=='namespace'and o['path']==A).update(qualified_name='wrong::cute',entity_id='wrong'))

    def test_original_duplicate_namespace_drop_other_position_rejected(self):
        def mutation(records):
            first=next(o for o in records if o['kind']=='namespace'and o['path']==A)
            records[:]=[o for o in records if not(o['kind']=='namespace'and o['path']==R)]
            records.append(copy.deepcopy(first))
        self.rejected(mutation)

    def test_namespace_same_spelling_different_entity_rejected(self):
        self.rejected(lambda rs:next(o for o in rs if o['kind']=='namespace'and o['path']==R).update(entity_id='other_cute_identity'))

    def test_alignment_unit_corruption_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0].update(alignment_unit='bits'))

    def test_alignment_expression_span_corruption_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0]['expression_span'].__setitem__('end_byte',3026))

    def test_expansion_definition_body_corruption_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0]['conditional_expansions'][0]['definition'].update(body='__align__(64)'))

    def test_expansion_condition_location_corruption_rejected(self):
        self.rejected(lambda rs:self.target(rs)['alignment_specifiers'][0]['conditional_expansions'][0]['conditions'][0]['directive_span'].update(start_line=53))

    def test_duplicate_branch_rejected(self):
        def mutation(rs):
            variants=self.target(rs)['alignment_specifiers'][0]['conditional_expansions']
            variants[1]=copy.deepcopy(variants[0])
        self.rejected(mutation)

if __name__=='__main__':unittest.main()
