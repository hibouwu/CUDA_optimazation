import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_declarations import audit
from extract_declarations import Extractor


class MechanicalAuditTests(unittest.TestCase):
    def fixture(self, change=None):
        source = b'#define MAKE(X) struct X {};\nMAKE(A)\nint f(int x); int f(double x);'
        e = Extractor('fixture'); e.extract('include/test.hpp',source)
        data = e.result()
        generators = {'scripts/'+name for name in ('extract_declarations.py','declaration_projection.py','declaration_syntax.py','macro_expansion.py','namespace_bindings.py','declaration_bitfields.py','declaration_headers.py','declaration_scope_integrity.py')}
        data['generator_provenance'] = {'sources_changed_during_run':[], 'data_matches_end_source_files':True,
            'source_hashes_at_end':{name:hashlib.sha256(b'# test fixture\n').hexdigest()for name in generators}}
        if change: change(data)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'data').mkdir(); (root/'snapshot/include').mkdir(parents=True)
            (root/'scripts').mkdir()
            for name in generators:(root/name).write_bytes(b'# test fixture\n')
            (root/'snapshot/include/test.hpp').write_bytes(source)
            (root/'data/scope.json').write_text(json.dumps({'commit':'fixture','files':[{'path':'include/test.hpp','sha256':hashlib.sha256(source).hexdigest()}]}))
            path = root/'data/declarations.json'; path.write_text(json.dumps(data))
            return audit(root,path)

    def test_real_and_macro_signatures_are_checked(self):
        result = self.fixture()
        self.assertEqual(result['issues'],[])
        self.assertTrue(result['mechanical_audit_passed'])
        self.assertFalse(result['phase_1_passed'])

    def test_duplicate_occurrence_is_not_hidden_by_matching_count(self):
        def change(data):
            data['occurrences'].append(copy.deepcopy(data['occurrences'][-1]))
            data['summary']['occurrences'] += 1
            data['files'][0]['occurrence_count'] += 1
        result = self.fixture(change)
        self.assertIn('duplicate_occurrence',{i['kind']for i in result['issues']})

    def test_wrong_source_signature_is_rejected(self):
        result = self.fixture(lambda d:next(o for o in d['occurrences']if o['name']=='f').update(raw_signature='int f(char x);'))
        self.assertIn('physical_signature_mismatch',{i['kind']for i in result['issues']})

    def test_wrong_backreference_is_rejected(self):
        result = self.fixture(lambda d:d['entities'][-1].update(declaration_occurrence_ids=[]))
        self.assertIn('entity_occurrence_backreference_mismatch',{i['kind']for i in result['issues']})

    def test_parser_placeholder_in_return_is_rejected(self):
        result = self.fixture(lambda d:d['occurrences'][-1].update(return_type='__codex_parser_fake'))
        self.assertIn('parser_placeholder_in_semantic_field',{i['kind']for i in result['issues']})

    def test_wrong_macro_signature_is_rejected(self):
        def change(d):
            next(o for o in d['occurrences']if o.get('macro_origin'))['raw_signature']='struct Wrong'
        result = self.fixture(change)
        self.assertIn('virtual_signature_mismatch',{i['kind']for i in result['issues']})

    def test_empty_provenance_cannot_claim_current_generator(self):
        result = self.fixture(lambda d:d['generator_provenance'].update(source_hashes_at_end={}))
        self.assertIn('generator_provenance_denominator_mismatch',{i['kind']for i in result['issues']})

    def test_wrong_commit_is_rejected(self):
        result = self.fixture(lambda d:d.update(commit='other'))
        self.assertIn('commit_mismatch',{i['kind']for i in result['issues']})


if __name__ == '__main__': unittest.main()
