"""Attribute ownership must survive the final source ledger audit."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_declarations import alignment_issues
from extract_declarations import Extractor


class AlignmentMechanicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path='include/cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized_fp8_blockwise_scaling.hpp'
        cls.source=(ROOT/'snapshot'/path).read_bytes()
        extractor=Extractor('8f50b052e1099fb982392a622caab69b97b63128');extractor.extract(path,cls.source)
        cls.occurrences=[o for o in extractor.occurrences if o.get('alignment_specifiers')]

    def test_field_attributes_have_exact_source_and_owner(self):
        self.assertEqual(len(self.occurrences),4)
        self.assertFalse([issue for o in self.occurrences for issue in alignment_issues(o,self.source)])

    def test_wrong_owner_is_rejected(self):
        o=copy.deepcopy(self.occurrences[0]);o['alignment_specifiers'][0]['owner_occurrence_id']='wrong'
        self.assertIn('owner_backreference_mismatch',alignment_issues(o,self.source)[0]['reasons'])

    def test_wrong_expression_is_rejected(self):
        o=copy.deepcopy(self.occurrences[0]);o['alignment_specifiers'][0]['alignment_expression']='64'
        self.assertIn('expression_source_mismatch',alignment_issues(o,self.source)[0]['reasons'])

    def test_type_field_confusion_is_rejected(self):
        o=copy.deepcopy(self.occurrences[0]);o['alignment_specifiers'][0]['owner_hint']['kind']='type'
        self.assertIn('type_attribute_on_non_type',alignment_issues(o,self.source)[0]['reasons'])


if __name__=='__main__':unittest.main()
