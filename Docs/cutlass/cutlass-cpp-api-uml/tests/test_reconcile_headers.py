"""Independent conditional-function fixtures; never call the declaration extractor."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import reconcile_candidates as r
PATH='include/cute/header_reconcile_fixture.hpp'


def fixture(body='{ sink(1); return 0; }'):
    source='int\n#if A\nf(int x)\n#else\nf(int y)\n#endif\n'+body
    data=source.encode();end_header=data.index(b'{');end=len(data)
    view=r.SourceProof(PATH,data);occurrences=[]
    for i,(spans,guards) in enumerate(view.conditional_projections(0,end_header)):
        pieces=[];segments=[];offset=0
        for a,b in spans+[(end_header,end)]:
            piece=data[a:b];pieces.append(piece)
            segments.append({'virtual_start_byte':offset,'virtual_end_byte':offset+len(piece),
                'physical_span':{'path':PATH,'start_byte':a,'end_byte':b},'mapping':'exact_source_slice',
                'role':'header'if b<=end_header else'shared_body'})
            offset+=len(piece)
        header=b''.join(pieces[:-1]).decode()
        conditions=[{'expression':g['expression'],'directive_path':PATH,'directive_line':g['line']}for g in guards]
        name='x'if i==0 else'y';pstart=data.index(('int '+name).encode())
        origin={'kind':'function_header_variant','source_declaration_id':'physical_f','conditional_variant_id':'variant_'+str(i),
            'declaration_span':{'path':PATH,'start_byte':0,'end_byte':end},
            'signature_span':{'path':PATH,'start_byte':0,'end_byte':end_header},
            'body_span':{'path':PATH,'start_byte':end_header,'end_byte':end},
            'semantic_spelling':data[:end_header].decode(),'virtual_signature':header,'virtual_source':b''.join(pieces).decode(),
            'segments':segments,'conditions':conditions,'status':'header_parsed','signature':{'name':'f'}}
        occurrences.append({'id':'f_variant_'+str(i),'entity_id':'entity_f','source_id':'source_f',
            'kind':'function','name':'f','qualified_name':'f','range':[0,end],'signature':[0,end_header],
            'declarator':[data.index(('f(int '+name).encode()),end_header],'syntax':[0,end],'body':[end_header,end],
            'raw_signature_sha256':r.sha(data[:end_header].decode().rstrip().encode()),'parse_status':'parsed','attributes':[],
            'parameter_count':1,'parameters':[{'range':[pstart,pstart+5],'name':name,'type':'int','default':None}],
            'return_type':'int','expanded_signature':header.rstrip(),'conditions':conditions,'condition_lines':[g['line']for g in guards],
            'scope_chain':[],'name_resolution':{},'qualified_name_resolution':'source_name','macro_origin':None,'conditional_origin':origin})
    return source,occurrences


def run(source,occurrences):
    cs=r.LEXER.scan_sources({PATH:source.encode()})['candidates']
    reconciler=r.Reconciler(PATH,source.encode(),cs,occurrences)
    return reconciler,list(reconciler.run())


class HeaderReconciliationTests(unittest.TestCase):
    def header_rows(self,source,rows):
        stop=source.index('{')
        return [x for x in rows if x['byte_range'][0]<x['byte_range'][1]<=stop and x['candidate_kind']=='syntax_interval' and x['status']!='classified_non_api']

    def test_all_branches_and_shared_body_are_verified(self):
        source,occurrences=fixture();checker,rows=run(source,occurrences)
        self.assertTrue(all(not checker.invalid_occurrence(o)for o in occurrences))
        self.assertTrue(self.header_rows(source,rows))
        self.assertTrue(all(x['status']=='mapped_occurrences'for x in self.header_rows(source,rows)))
        sink=[x for x in rows if source[x['byte_range'][0]:x['byte_range'][1]].strip()=='sink(1)']
        self.assertTrue(sink)
        self.assertTrue(all(x['status']=='classified_non_api'for x in sink))

    def test_deleted_header_branch_cannot_use_other_branch(self):
        source,occurrences=fixture();_,rows=run(source,occurrences[:1])
        self.assertTrue(any(x['status']=='pending'for x in self.header_rows(source,rows)))

    def test_changed_parameter_type_and_return_type_are_rejected(self):
        for which in ('parameter','return'):
            with self.subTest(which=which):
                source,occurrences=fixture()
                if which=='parameter':occurrences[0]['parameters'][0]['type']='float'
                else:occurrences[0]['return_type']='float'
                checker,_=run(source,occurrences)
                self.assertTrue(checker.invalid_occurrence(occurrences[0]))

    def test_changed_virtual_body_or_truncated_body_is_rejected(self):
        for which in ('virtual','range','segment'):
            with self.subTest(which=which):
                source,occurrences=fixture();o=occurrences[0]
                if which=='virtual':o['conditional_origin']['virtual_source']=o['conditional_origin']['virtual_source'].replace('sink(1)','sink(2)')
                elif which=='range':o['body'][1]-=1
                else:o['conditional_origin']['segments'][-1]['physical_span']['end_byte']-=1
                checker,_=run(source,occurrences)
                self.assertTrue(checker.invalid_occurrence(o))

    def test_guard_and_segment_forgery_are_rejected(self):
        source,occurrences=fixture();occurrences[0]['conditional_origin']['conditions'][0]['expression']='B'
        checker,_=run(source,occurrences)
        self.assertTrue(checker.invalid_occurrence(occurrences[0]))

    def test_missing_local_type_not_covered_by_function_body(self):
        source,occurrences=fixture('{ struct Hidden {int value;}; return 0; }')
        _,rows=run(source,occurrences)
        selected=[x for x in rows if 'Hidden' in source[x['byte_range'][0]:x['byte_range'][1]] and x['candidate_kind']=='syntax_interval']
        self.assertTrue(selected)
        self.assertTrue(any(x['status']=='pending'for x in selected))


if __name__=='__main__':unittest.main()
