"""Independent source contracts for production using_syntax.

The frozen oracle module is unchanged and never imports production code. Its
separate written-using enumeration is compared per physical span across the
fixed 824 files. Negative fixtures also check names, syntax ranges and limits.
"""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from using_syntax import scan_source_usings

SPEC = importlib.util.spec_from_file_location('frozen_using_oracle_for_syntax_test', ROOT/'tests/test_using_declaration_independent_review.py')
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)
COMMIT = '8f50b052e1099fb982392a622caab69b97b63128'


def slice_text(source, where):
    return source[where['start_byte']:where['end_byte']].decode()


class UsingSyntaxFixtures(unittest.TestCase):
    def scan(self, source, *, clean=True, path='include/cute/using_source_fixture.hpp'):
        source = source.encode() if isinstance(source, str) else source
        data = scan_source_usings(path, source, COMMIT)
        if clean:
            self.assertFalse(data['diagnostics'], data['diagnostics'])
        for record in data['sources'] + data['namespace_aliases']:
            self.check_ranges(source, record)
            forbidden = {'owner', 'scope_chain', 'qualified_name', 'entity_id', 'target_type', 'target_entity_id', 'access'}
            self.assertFalse(forbidden.intersection(record))
            self.assertEqual(record['semantic_resolution'], 'not_attempted_source_syntax_only')
        self.assertFalse(data['scope_or_target_resolution_performed'])
        return data

    def check_ranges(self, source, record):
        where = record['source_range']
        self.assertEqual(slice_text(source, where), record['raw_signature'])
        self.assertEqual(hashlib.sha256(source[where['start_byte']:where['end_byte']]).hexdigest(), record['raw_sha256'])
        for field, text in [('target_source_range', 'target_source_expression'), ('target_name_source_range', 'target_name_source_expression')]:
            if record[field]:
                self.assertEqual(slice_text(source, record[field]), record[text])
        for segment in record['qualifier_segments']:
            self.assertEqual(slice_text(source, segment['source_range']), segment['source_expression'])
        def visit(value):
            if isinstance(value, dict):
                if {'start_byte','end_byte','start_line','end_line'} <= value.keys():
                    a,b = value['start_byte'],value['end_byte']
                    self.assertTrue(0 <= a <= b <= len(source), value)
                    self.assertEqual(source[:a].count(b'\n')+1,value['start_line'])
                    self.assertEqual(source[:max(a,b-1)].count(b'\n')+1,value['end_line'])
                    if 'raw' in value:self.assertEqual(slice_text(source,value),value['raw'])
                for item in value.values():visit(item)
            elif isinstance(value,list):
                for item in value:visit(item)
        visit(record)

    def test_alias_attribute_operator_assignment_and_typename_are_distinct(self):
        source='''using red [[deprecated("message = ;")]] = atomic_add<T>;
using Base::operator=;
using typename Base<T>::type;
using namespace N;
'''
        records=self.scan(source)['sources']
        self.assertEqual([r['kind']for r in records],['alias','import','import','directive'])
        self.assertEqual(records[0]['declared_name'],'red')
        self.assertEqual(records[0]['target_source_expression'],'atomic_add<T>')
        self.assertEqual(len(records[0]['attribute_ranges']),1)
        self.assertEqual(records[1]['terminal_name'],'operator=')
        self.assertEqual(records[1]['terminal_kind'],'operator_function_id')
        self.assertEqual(records[2]['target_source_expression'],'typename Base<T>::type')
        self.assertEqual(records[2]['target_name_source_expression'],'Base<T>::type')
        self.assertEqual(records[2]['typename_range']['raw'],'typename')
        self.assertEqual(records[2]['terminal_name'],'type')
        self.assertIsNone(records[3]['declared_name'])

    def test_operator_punctuation_is_not_a_template_or_import_list(self):
        records=self.scan('using B::operator<; using B::operator,; using B::operator<<; using B::operator[];')['sources']
        self.assertEqual([r['terminal_name']for r in records],['operator<','operator,','operator<<','operator[]'])
        self.assertEqual({r['kind']for r in records},{'import'})
        self.assertTrue(all(not r['constructor_spelling_candidate']for r in records))

    def test_nested_template_scopes_are_not_namespace_separators(self):
        data=self.scan('using ::alpha::Outer<beta::T, Inner<gamma::U>>::Member<delta::V>::name;')
        r=data['sources'][0]
        self.assertEqual([s['source_expression']for s in r['qualifier_segments']],
                         ['alpha','Outer<beta::T, Inner<gamma::U>>','Member<delta::V>'])
        self.assertEqual(len(r['scope_separator_ranges']),3)
        self.assertEqual(r['terminal_name'],'name')
        self.assertEqual(r['leading_global_scope_range']['end_byte']-r['leading_global_scope_range']['start_byte'],2)

    def test_parenthesized_comparison_inside_template_does_not_close_qualifier(self):
        r=self.scan('using Base<(A::value > 0), typename other::T>::Base;')['sources'][0]
        self.assertEqual(len(r['qualifier_segments']),1)
        self.assertEqual(r['qualifier_segments'][0]['source_expression'],'Base<(A::value > 0), typename other::T>')
        self.assertEqual(r['terminal_name'],'Base')
        self.assertTrue(r['constructor_spelling_candidate'])
        self.assertEqual(r['constructor_candidate_evidence']['semantic_constructor_classification'],'not_attempted_requires_owner_and_base_binding')

    def test_constructor_candidate_is_only_a_spelling_candidate(self):
        records=self.scan('using iter_adaptor<P, gmem_ptr<P>>::iter_adaptor; using Q::Base::Base; using Base<T>::member; using typename Base<T>::type;')['sources']
        self.assertEqual([r['constructor_spelling_candidate']for r in records],[True,True,False,False])
        self.assertEqual({r['kind']for r in records},{'import'})
        self.assertEqual(records[0]['terminal_name'],'iter_adaptor')
        self.assertEqual(len(records[0]['qualifier_segments']),1)
        # No class owner was supplied, so even Q::Base::Base is not classified
        # as a constructor; an equal qualifier spelling is insufficient.
        self.assertNotIn('inherited_constructor_import',json.dumps(records))

    def test_comments_literals_and_spliced_keywords_preserve_physical_bytes(self):
        source=b'// using fake::a;\nconst char* s="using fake::b;";\nconst char* r=R"tag(using fake::c;)tag";\nus\\\ning ::n /* middle */ :: fo\\\no;\n'
        d=self.scan(source)
        self.assertEqual(d['counts']['written_using'],1)
        r=d['sources'][0]
        self.assertEqual(r['keyword_range']['raw'],'us\\\ning')
        self.assertEqual(r['keyword_range']['logical_token'],'using')
        self.assertEqual(r['terminal_name'],'foo')
        self.assertEqual(slice_text(source,r['terminal_name_range']),'fo\\\no')
        self.assertEqual(r['target_source_expression'],'::n /* middle */ :: fo\\\no')

    def test_crlf_spliced_comment_cannot_create_using(self):
        source=b'// comment \\\r\nusing fake::name;\r\nusing n::value;\r\n'
        r=self.scan(source)['sources']
        self.assertEqual(len(r),1)
        self.assertEqual(r[0]['raw_signature'],'using n::value;')
        self.assertEqual(r[0]['source_range']['start_line'],3)

    def test_namespace_alias_has_own_family_and_never_increases_using_count(self):
        d=self.scan('namespace UMMA { namespace TMEM = cute::TMEM; using namespace SM90; }')
        self.assertEqual(d['counts']['written_using'],1)
        self.assertEqual(d['counts']['namespace_alias'],1)
        r=d['namespace_aliases'][0]
        self.assertEqual(r['kind'],'namespace_alias')
        self.assertEqual(r['declared_name'],'TMEM')
        self.assertEqual(r['terminal_name'],'TMEM')
        self.assertEqual(r['target_source_expression'],'cute::TMEM')
        self.assertIn('source_namespace_alias_id',r)
        self.assertNotIn('source_using_id',r)
        self.assertFalse(r['constructor_spelling_candidate'])

    def test_dependent_alias_rhs_remains_complete_type_expression(self):
        d=self.scan('using X = typename Base<T>::type; using P = const int*; using F = int(*)(int);')
        self.assertEqual([r['kind']for r in d['sources']],['alias']*3)
        self.assertEqual(d['sources'][0]['typename_range']['raw'],'typename')
        self.assertEqual(d['sources'][0]['target_source_expression'],'typename Base<T>::type')
        self.assertEqual(d['sources'][1]['target_source_expression'],'const int*')
        self.assertIsNone(d['sources'][1]['terminal_name'])
        self.assertEqual(d['sources'][2]['target_source_expression'],'int(*)(int)')
        self.assertIsNone(d['sources'][2]['terminal_name'])

    def test_decltype_and_builtin_types_are_not_template_names(self):
        d=self.scan('using A = decltype(foo()); using B = int; using C = auto; using D = decltype(foo())::type;')
        for r in d['sources'][:3]:
            self.assertIsNone(r['terminal_name'])
            self.assertEqual(r['target_form'],'type_expression')
        qualified=d['sources'][3]
        self.assertEqual(qualified['terminal_name'],'type')
        self.assertEqual(qualified['qualifier_segments'][0]['syntax_kind'],'decltype_specifier')
        self.assertIsNone(qualified['qualifier_segments'][0]['head_name'])

    def test_semicolon_inside_alias_lambda_is_not_statement_terminator(self):
        d=self.scan('using X = decltype([] { int x=0; return x; }); using n::next;')
        self.assertEqual(d['counts']['written_using'],2)
        self.assertEqual(d['sources'][0]['raw_signature'],'using X = decltype([] { int x=0; return x; });')
        self.assertEqual(d['sources'][1]['terminal_name'],'next')

    def test_inline_type_alias_keeps_member_semicolon_and_outer_terminator(self):
        source='using TmaDescriptor = struct alignas(64) { char bytes[128]; };'
        r=self.scan(source)['sources'][0]
        self.assertEqual(r['raw_signature'],source)
        self.assertEqual(r['target_source_expression'],'struct alignas(64) { char bytes[128]; }')
        self.assertEqual(r['kind'],'alias')
        self.assertIsNone(r['terminal_name'])
        self.assertEqual(r['target_form'],'type_expression')

    def test_conditions_are_not_evaluated_or_deleted(self):
        d=self.scan('#if 0\nusing ns::cold;\n#else\nusing ns::hot;\n#endif\n')
        self.assertEqual([r['terminal_name']for r in d['sources']],['cold','hot'])
        self.assertNotIn('preprocessor_conditions',d['sources'][0])
        # Context/condition binding belongs to the caller, not this layer.
        self.assertNotIn('active',d['sources'][0])

    def test_unterminated_using_retained_with_blocking_diagnostic(self):
        d=self.scan('namespace n { using Base::name }',clean=False)
        self.assertEqual(d['counts']['written_using'],1)
        self.assertEqual(d['sources'][0]['kind'],'unsupported')
        self.assertEqual(d['sources'][0]['syntax_status'],'malformed')
        self.assertTrue(d['sources'][0]['diagnostic_refs'])
        self.assertTrue(all(x['blocks_using_source_contract']for x in d['diagnostics']))

    def test_unsupported_enum_and_multi_import_are_not_silently_ordinary(self):
        d=self.scan('using enum Color; using A::a, B::b;',clean=False)
        self.assertEqual([r['kind']for r in d['sources']],['unsupported','unsupported'])
        self.assertEqual(d['counts']['written_using'],2)
        self.assertEqual(d['sources'][0]['target_source_expression'],'Color')
        self.assertEqual(d['sources'][1]['target_source_expression'],'A::a, B::b')
        self.assertEqual({x['category']for x in d['diagnostics']},{'using_enum_outside_fixed_shapes','using_declarator_list_outside_fixed_shapes'})

    def test_empty_alias_and_malformed_literals_cannot_report_clean(self):
        d=self.scan('using X = ; "unterminated',clean=False)
        self.assertEqual(d['sources'][0]['kind'],'alias')
        self.assertEqual(d['sources'][0]['syntax_status'],'malformed')
        self.assertEqual({x['category']for x in d['diagnostics']},{'empty_alias_target','source_lexical_unterminated_quoted_literal'})

    def test_physical_ids_are_repeatable_distinct_and_context_free(self):
        source=b'using ns::x; using ns::x;'
        a=self.scan(source);b=self.scan(source)
        ids=[r['source_using_id']for r in a['sources']]
        self.assertEqual(ids,[r['source_using_id']for r in b['sources']])
        self.assertEqual(len(set(ids)),2)
        other=scan_source_usings('include/cute/other.hpp',source,COMMIT)
        self.assertNotEqual(ids[0],other['sources'][0]['source_using_id'])
        other=scan_source_usings(a['path'],source,'another-commit')
        self.assertNotEqual(ids[0],other['sources'][0]['source_using_id'])

    def test_importlib_loading_does_not_depend_on_cwd_or_sys_path(self):
        code='''import importlib.util,json,sys
spec=importlib.util.spec_from_file_location("isolated_using_module",sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
result=module.scan_source_usings("fixture.hpp",b"using n::x;","fixture")
assert result["sources"][0]["terminal_name"]=="x"
assert not result["diagnostics"]
print(json.dumps(result["counts"]))
'''
        result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'scripts/using_syntax.py')],
            cwd=tempfile.gettempdir(),text=True,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['written_using'],1)


class FixedUsingSyntaxInventory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scope=json.loads((ROOT/'data/scope.json').read_text())
        assert len(scope['files'])==824
        cls.counts=Counter();cls.nonalias=[];cls.aliases=[];cls.ids=set();cls.namespace_aliases=[]
        cls.examined_files=0;cls.mismatches=[];cls.diagnostics=[];cls.oracle_span_refinements=[]
        mapping={'alias_declaration':'alias','using_import':'import','using_directive':'directive','using_enum':'unsupported'}
        # The frozen independent inventory counts these two aliases correctly,
        # but stops at the inline struct member's first semicolon. Refine ONLY
        # these exact source literals; do not derive expected ends from the
        # production result or change the old oracle/test file.
        inline_aliases={
            ('include/cute/arch/copy_sm90_desc.hpp',295):'using TmaDescriptor = struct alignas(64) { char bytes[128]; };',
            ('include/cute/arch/copy_sm90_desc.hpp',296):'using Im2ColTmaDescriptor = struct alignas(64) { char bytes[128]; };',
        }
        for entry in scope['files']:
            source=(ROOT/'snapshot'/entry['path']).read_bytes()
            assert hashlib.sha256(source).hexdigest()==entry['sha256']
            result=scan_source_usings(entry['path'],source,scope['commit'])
            independently_enumerated=ORACLE.written_using_forms(source)
            observed=Counter((r['source_range']['start_byte'],r['source_range']['end_byte'],r['kind'],r['raw_signature'])for r in result['sources'])
            expected=Counter()
            for r in independently_enumerated:
                exact=inline_aliases.get((entry['path'],r['start_line']))
                if exact:
                    a=r['start_byte'];b=a+len(exact.encode())
                    assert source[a:b].decode()==exact
                    assert r['raw']==exact[:-3] and r['end_byte']==b-3
                    cls.oracle_span_refinements.append((entry['path'],r['start_line'],r['end_byte'],b))
                    expected[(a,b,mapping[r['kind']],exact)]+=1
                else:expected[(r['start_byte'],r['end_byte'],mapping[r['kind']],r['raw'])]+=1
            if observed!=expected:cls.mismatches.append(entry['path'])
            cls.counts.update(result['counts']);cls.diagnostics.extend(result['diagnostics']);cls.examined_files+=1
            for r in result['sources']:
                assert r['source_using_id']not in cls.ids
                cls.ids.add(r['source_using_id'])
                if r['kind']!='alias':cls.nonalias.append(r)
                elif entry['path']=='include/cutlass/functional.h'and r['source_range']['start_line']==990:cls.aliases.append(r)
            cls.namespace_aliases.extend(result['namespace_aliases'])

    def test_exact_fixed_inventory_and_per_span_oracle_comparison(self):
        self.assertEqual(self.examined_files,824)
        self.assertFalse(self.mismatches)
        self.assertFalse(self.diagnostics)
        self.assertEqual(self.counts,{'written_using':59819,'alias':59136,'import':499,'directive':184,'unsupported':0,
                                     'namespace_alias':1,'constructor_spelling_candidates':83})
        self.assertEqual(len(self.ids),59819)
        self.assertEqual(len({r['path']for r in self.nonalias}),162)
        self.assertEqual(self.oracle_span_refinements,[
            ('include/cute/arch/copy_sm90_desc.hpp',295,12067,12070),
            ('include/cute/arch/copy_sm90_desc.hpp',296,12138,12141)])

    def test_fixed_directive_targets_typename_and_namespace_alias(self):
        self.assertEqual(Counter(r['terminal_name']for r in self.nonalias if r['kind']=='directive'),{'cute':172,'detail':11,'SM90':1})
        typed=[r for r in self.nonalias if r['typename_range']]
        self.assertEqual([r['source_range']['start_line']for r in typed],list(range(73,79)))
        self.assertEqual({r['path']for r in typed},{'include/cutlass/gemm/kernel/gemm_grouped_per_group_scale.h'})
        self.assertEqual(len(self.namespace_aliases),1)
        ns=self.namespace_aliases[0]
        self.assertEqual((ns['path'],ns['source_range']['start_line'],ns['raw_signature']),
                         ('include/cute/atom/mma_traits_sm100.hpp',428,'namespace TMEM = cute::TMEM;'))

    def test_fixed_attribute_alias_and_qualifier_macro_spelling(self):
        self.assertEqual(len(self.aliases),1)
        alias=self.aliases[0]
        self.assertEqual(alias['declared_name'],'red')
        self.assertEqual(alias['target_source_expression'],'atomic_add<T>')
        self.assertEqual(len(alias['attribute_ranges']),1)
        r=next(r for r in self.nonalias if r['path']=='include/cute/util/type_traits.hpp'and r['source_range']['start_line']==92)
        self.assertEqual(r['terminal_name'],'remove_cv_t')
        self.assertEqual([s['head_name']for s in r['qualifier_segments']],['CUTE_STL_NAMESPACE'])
        self.assertEqual(r['target_source_expression'],'CUTE_STL_NAMESPACE::remove_cv_t')
        self.assertFalse(r['constructor_spelling_candidate'])
        r=next(r for r in self.nonalias if r['path']=='include/cutlass/fast_math.h'and r['source_range']['start_line']==58)
        self.assertIsNotNone(r['leading_global_scope_range'])
        self.assertEqual(r['terminal_name'],'swap')
        self.assertEqual([s['head_name']for s in r['qualifier_segments']],['cuda','std'])


if __name__=='__main__':unittest.main()
