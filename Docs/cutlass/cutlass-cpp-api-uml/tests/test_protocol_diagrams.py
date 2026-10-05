"""Protocol views preserve explicit causality, endpoint identity and evidence."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from protocol_diagrams import verify_protocol,protocol_puml,protocol_dot,subdivide_protocol_views
from build_atlas import puml_label


def fixture():
    nodes={'api':{'id':'api','name':'Allocator::free'}}
    evidence=[{'id':'source','path':'include/a.hpp','start_line':1,'end_line':4}]
    actors=[{'id':key,'label':key,'evidence_refs':['source']}for key in ['leader','peer']]
    events=[{'id':key,'label':'free '+actor,'actor_id':actor,'api_node_ref':'api',
             'anchors':['enter','return'],'callsite':{'path':'include/a.hpp','start_line':4},'evidence_refs':['source']}
            for key,actor in [('l_free','leader'),('p_free','peer')]]
    orders=[{'id':key+'.intra','before':{'event_id':key,'anchor':'enter'},'after':{'event_id':key,'anchor':'return'},
             'condition':'same invocation','evidence_refs':['source']}for key in ['l_free','p_free']]
    resource={'id':'r','states':[{'id':'allocated','label':'allocated'},{'id':'released','label':'released'}]}
    transitions=[{'id':'release','resource_id':'r','from':'allocated','to':'released',
        'trigger':{'event_id':'l_free','anchor':'return'},'preconditions':['peer uses completed before free'],
        'evidence_refs':['source']}]
    views=[{'id':'partial','kind':'partial_order','title':'No global free order','event_ids':['l_free','p_free'],
            'order_ids':[x['id']for x in orders]},
           {'id':'state','kind':'state','title':'Resource','resource_id':'r','transition_ids':['release']}]
    return {'id':'protocol','actors':actors,'resources':[resource],'events':events,'partial_order':orders,
            'state_transitions':transitions,'views':views,'evidence':evidence},nodes,{}


class ProtocolTests(unittest.TestCase):
    def test_explicit_partial_order_is_valid(self):
        p,n,e=fixture();self.assertEqual(len(verify_protocol(p,n,e)['events']),2)

    def test_same_api_same_line_different_actors_are_not_merged(self):
        p,n,e=fixture();text=protocol_puml(p,p['views'][0],n,e,puml_label)
        self.assertIn('event=l_free',text);self.assertIn('event=p_free',text)
        for order in p['partial_order']:self.assertIn(order['id'],text)

    def test_array_order_does_not_add_cross_actor_edges(self):
        p,n,e=fixture();text=protocol_puml(p,p['views'][0],n,e,puml_label)
        p['events'].reverse();again=protocol_puml(p,p['views'][0],n,e,puml_label)
        self.assertEqual(text,again)
        self.assertEqual(text.count(' --> '),0)
        for order in p['partial_order']:self.assertIn(order['id'],text)

    def test_async_effect_is_not_implicitly_equal_to_return(self):
        p,n,e=fixture();p['events'][0]['anchors'].append('effect')
        text=protocol_puml(p,p['views'][0],n,e,puml_label)
        self.assertNotIn('<<effect>>',text)

    def test_missing_api_and_missing_anchor_fail(self):
        p,n,e=fixture();p['events'][0]['api_node_ref']='invented'
        with self.assertRaisesRegex(ValueError,'Unknown protocol API'):verify_protocol(p,n,e)
        p,n,e=fixture();p['partial_order'][0]['after']['anchor']='effect'
        with self.assertRaisesRegex(ValueError,'Unknown event anchor'):verify_protocol(p,n,e)

    def test_order_without_source_or_with_cycle_fails(self):
        p,n,e=fixture();p['partial_order'][0]['evidence_refs']=[]
        with self.assertRaisesRegex(ValueError,'without source'):verify_protocol(p,n,e)
        p,n,e=fixture();back=copy.deepcopy(p['partial_order'][0]);back['id']='back';back['before'],back['after']=back['after'],back['before'];p['partial_order'].append(back)
        with self.assertRaisesRegex(ValueError,'Cyclic'):verify_protocol(p,n,e)

    def test_hidden_order_and_wrong_resource_fail(self):
        p,n,e=fixture();p['views'][0]['order_ids'].pop()
        with self.assertRaisesRegex(ValueError,'not visible'):verify_protocol(p,n,e)
        p,n,e=fixture();p['state_transitions'][0]['resource_id']='unknown'
        with self.assertRaisesRegex(ValueError,'Unknown transition resource'):verify_protocol(p,n,e)

    def test_state_transition_keeps_trigger_and_guard(self):
        p,n,e=fixture();text=protocol_puml(p,p['views'][1],n,e,puml_label)
        self.assertEqual(p['state_transitions'][0]['preconditions'],['peer uses completed before free'])
        self.assertIn('event=l_free',text)
        self.assertEqual(text.count(' --> '),1)

    def test_dot_keeps_distinct_row_ports_and_all_constraint_ids(self):
        p,n,e=fixture();text=protocol_dot(p,p['views'][0],n,e)
        self.assertIn('PORT="enter"',text);self.assertIn('PORT="return"',text)
        for order in p['partial_order']:self.assertIn('order='+order['id'],text)
        self.assertNotIn('e0:return:e -> e1:',text)

    def test_different_api_events_still_have_separate_port_edges(self):
        p,n,e=fixture();p['partial_order'].append({'id':'cross','before':{'event_id':'l_free','anchor':'enter'},'after':{'event_id':'p_free','anchor':'enter'},'evidence_refs':['source']})
        p['views'][0]['order_ids'].append('cross');text=protocol_dot(p,p['views'][0],n,e)
        self.assertIn('e0:enter:e -> e1:enter:e',text)
        self.assertIn('id="cross"',text)

    def test_subdivision_preserves_all_authored_constraints(self):
        p,n,e=fixture();split=subdivide_protocol_views(p)
        ids={oid for v in split for oid in v.get('order_ids',[])}
        self.assertEqual(ids,{o['id']for o in p['partial_order']})
        for v in split:
            self.assertTrue(set(v.get('order_ids',[]))<=ids)
        p['views']=split;verify_protocol(p,n,e)


if __name__=='__main__':unittest.main()
