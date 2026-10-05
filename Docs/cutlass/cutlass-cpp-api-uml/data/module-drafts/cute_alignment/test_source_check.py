import copy
import json
from pathlib import Path
import unittest
from check_source import verify

class AlignmentDraftMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base=json.loads((Path(__file__).parent/'relations.json').read_text())
    def fails(self,change):
        data=copy.deepcopy(self.base);change(data)
        with self.assertRaises((AssertionError,KeyError,ValueError)):verify(data)
    def test_original(self):self.assertTrue(verify(self.base)['source_accounting_check_passed'])
    def test_removed_partial_specialization(self):
        self.fails(lambda d:d['nodes'].__setitem__(slice(None),[n for n in d['nodes']if n['id']!='alignment.type.aligned_256']))
    def test_missing_cuda_macro_branch(self):
        self.fails(lambda d:next(n for n in d['nodes']if n.get('semantic_class')=='requested_alignment_attribute')['expansion_variants'].pop())
    def test_cast_misrepresented_as_call(self):
        self.fails(lambda d:next(e for e in d['edges']if e['id']=='alignment.edge.evaluate_cast').update(relation='calls'))
    def test_static_assert_call_misrepresented_as_runtime(self):
        self.fails(lambda d:next(e for e in d['edges']if e['relation']=='calls').update(evaluation='potentially_evaluated_runtime'))
    def test_external_typedef_misrepresented_as_builtin(self):
        self.fails(lambda d:next(o for o in d['coverage']['obligations']if o['category']=='type_spelling_token'and o['source_range']['raw']=='uintptr_t').update(semantic_role='builtin_type'))
    def test_dropped_alignment_owner_contract(self):
        self.fails(lambda d:d['edges'].__setitem__(slice(None),[e for e in d['edges']if e['id']!='alignment.edge.attach_attribute_aligned_60']))
    def test_changed_default_alignment(self):
        self.fails(lambda d:next(n for n in d['nodes']if n['id']=='alignment.type.array_aligned')['template_parameters'][2].update(default='128'))
    def test_wrong_old_global_identity(self):
        self.fails(lambda d:next(n for n in d['nodes']if n['id']=='alignment.type.array_aligned').update(entity_id='ent_wrong_CUTE_ALIGNAS'))
    def test_lost_operator_obligation(self):
        self.fails(lambda d:d['coverage']['obligations'].__setitem__(slice(None),[o for o in d['coverage']['obligations']if o['category']!='operator_expression']))
    def test_macro_namespace_is_not_cute_member(self):
        self.fails(lambda d:next(n for n in d['nodes']if n['id']=='alignment.macro.cuda').update(qualified_name='cute::CUTE_ALIGNAS'))
    def test_false_child_inheritance(self):
        self.fails(lambda d:next(n for n in d['nodes']if n['id']=='alignment.type.aligned_128').update(bases=['Child']))
    def test_false_acceptance_claim(self):
        self.fails(lambda d:d['coverage'].update(api_coverage_passed=True))
    def test_dependency_signature_must_stop_before_body(self):
        self.fails(lambda d:next(n for n in d['nodes']if n['id']=='alignment.dep.has_single_bit')['signature_range'].__setitem__('end_byte',3984))
    def test_type_reference_does_not_include_parameter_name(self):
        self.fails(lambda d:next(e for e in d['edges']if e['source']=='alignment.type.array_aligned'and e['target']=='alignment.external.size_t').update(source_expression='size_t N'))

if __name__=='__main__':unittest.main()
