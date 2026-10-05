"""Deliberate mutations of the authored draft, never write artifacts."""
import copy
import json
from pathlib import Path
import unittest
import check_source

class ArraySourceDraftReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=json.loads((Path(__file__).parent/'relations.json').read_text())
    def rejected(self,mutation):
        data=copy.deepcopy(self.original);mutation(data)
        with self.assertRaises((AssertionError,KeyError,StopIteration)):check_source.verify(data)
    def node(self,d,id):return next(n for n in d['nodes']if n['id']==id)
    def call(self,d,line,text):return next(e for e in d['edges']if e['relation']=='calls'and e['callsite']['start_line']==line and e['source_expression']==text)
    def delete_edge_coherently(self,d,e):
        d['edges'].remove(e)
        for v in d['views']:
            v['edge_ids']=[i for i in v['edge_ids']if i!=e['id']]
        for o in d['coverage']['obligations']:o['graph_refs']=[i for i in o['graph_refs']if i!=e['id']]
    def test_positive(self):self.assertEqual(check_source.verify(self.original)['bound_declarations'],92)
    def test_delete_physical_declaration(self):self.rejected(lambda d:d['coverage']['physical_declarations'].pop())
    def test_delete_one_overload(self):self.rejected(lambda d:d['nodes'].remove(self.node(d,'array.api.fn_124')))
    def test_same_signature_short_name_wrong_owner(self):
        self.rejected(lambda d:self.node(d,'array.type.cuda_std.tuple_size')['source_selector'].update(qualified_name='std::tuple_size<cute::array<T,N>>'))
    def test_merge_std_cuda_node_owner(self):
        self.rejected(lambda d:self.node(d,'array.type.cuda_std.tuple_size').update(qualified_name='std::tuple_size<cute::array<T,N>>'))
    def test_drop_std_bridge_physical_occurrence(self):
        self.rejected(lambda d:self.node(d,'array.type.std.tuple_size')['source_selectors'].pop())
    def test_entity_condition_falsely_first_occurrence_only(self):
        self.rejected(lambda d:self.node(d,'array.type.std.tuple_size').update(preprocessor_conditions=[{'expression':'!defined(__CUDACC_RTC__)'}]))
    def test_wrong_entity_availability_and_instead_of_or(self):
        self.rejected(lambda d:self.node(d,'array.type.std.tuple_size')['availability'].update(operator='all_of'))
    def test_drop_condition_on_cuda_include(self):
        self.rejected(lambda d:next(o for o in d['coverage']['obligations']if o['category']=='include'and o['source_range']['start_line']==452).update(preprocessor_conditions=[]))
    def test_preprocessor_as_runtime_evaluation(self):
        self.rejected(lambda d:next(e for e in d['edges']if e['relation']=='evaluates'and e['evidence'][0]['start_line']==450).update(evaluation='potentially_evaluated'))
    def test_template_parameter_wrong_cute_owner(self):
        self.rejected(lambda d:next(e for e in d['edges']if e['relation']=='type_uses'and e['evidence'][0]['start_line']==433).update(source='array.namespace.cute'))
    def test_nonconst_cbegin_wrong_const_overload(self):self.rejected(lambda d:self.call(d,120,'begin()').update(target='array.api.fn_112'))
    def test_zero_back_wrong_primary_begin(self):self.rejected(lambda d:self.call(d,239,'begin()').update(target='array.api.fn_106'))
    def test_free_clear_falsely_calls_member_clear(self):self.rejected(lambda d:self.call(d,355,'a.fill(T(0))').update(target='array.api.fn_331'))
    def test_get_rvalue_wrong_const_index(self):self.rejected(lambda d:self.call(d,425,'a[I]').update(target='array.api.fn_62'))
    def test_get_rvalue_mistaken_std_move(self):self.rejected(lambda d:self.call(d,425,'cute::move(a[I])').update(target='array.dependency.remove_cv_import'))
    def test_adl_swapped_for_fixed_std_target(self):
        def mutate(d):
            b=self.node(d,self.call(d,190,'swap((*this)[i], other[i])')['target'])
            b['known_candidates']=[c for c in b['known_candidates']if c['kind']!='dependent_lookup']
        self.rejected(mutate)
    def test_range_for_faked_literal_call(self):
        self.rejected(lambda d:next(e for e in d['edges']if e['relation']=='implicit_calls').update(source_expression_is_not_literal_call=False))
    def test_hide_ordinary_for_instantiation(self):
        self.rejected(lambda d:self.call(d,344,'lhs[i]').update(control_contexts=[]))
    def test_reverse_constexpr_branch_erased(self):self.rejected(lambda d:self.call(d,382,'t_r[k]').update(control_contexts=[]))
    def test_zero_injected_class_name_misbound_primary(self):
        self.rejected(lambda d:next(e for e in d['edges']if e['relation']=='type_uses'and e['evidence'][0]['start_line']==335 and e['source_expression']=='array').update(target='array.type.decl_42'))
    def test_std_bridge_inherits_wrong_integral_constant(self):
        self.rejected(lambda d:next(n for n in d['nodes']if n['id'].startswith('array.binding.base.465.')).update(qualified_name='std::integral_constant<size_t,N>'))
    def test_drop_source_parameter(self):self.rejected(lambda d:d['coverage']['parameter_obligations'].pop())
    def test_default_argument_fabricated(self):self.rejected(lambda d:d['coverage']['parameter_obligations'][0].update(default='0'))
    def test_change_exact_callsite_text(self):self.rejected(lambda d:self.call(d,425,'cute::move(a[I])')['callsite'].update(raw_source_expression='std::move(a[I])'))
    def test_lose_zero_empty_body(self):self.rejected(lambda d:self.node(d,'array.api.fn_331')['body_range'].update(raw='{ fill(T(0)); }'))
    def test_delete_call_and_all_references(self):self.rejected(lambda d:self.delete_edge_coherently(d,self.call(d,425,'a[I]')))
    def test_delete_type_reference_and_all_references(self):
        self.rejected(lambda d:self.delete_edge_coherently(d,next(e for e in d['edges']if e['relation']=='type_uses'and e['source_expression']=='ptrdiff_t')))
    def test_delete_alias_target_and_all_references(self):
        self.rejected(lambda d:self.delete_edge_coherently(d,next(e for e in d['edges']if e['relation']=='aliases'and e['source']=='array.alias.decl_44')))
    def test_delete_membership_and_all_references(self):
        self.rejected(lambda d:self.delete_edge_coherently(d,next(e for e in d['edges']if e['relation']=='member_of'and e['source']=='array.api.fn_56')))
    def test_fabricate_written_constructor_api(self):
        self.rejected(lambda d:d['nodes'].append({'id':'array.fake_ctor','kind':'api','name':'array','path':check_source.PATH,'line':42}))
    def test_skip_visible_old_rtc_tuple_forward(self):
        self.rejected(lambda d:self.delete_edge_coherently(d,next(e for e in d['edges']if e['relation']=='specializes'and e['target']=='array.type.decl_457')))

if __name__=='__main__':unittest.main()
