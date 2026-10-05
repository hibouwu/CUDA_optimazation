"""Independent logical arrays, real C++ arithmetic reference; no GPU evidence."""
import unittest,tempfile,subprocess,json,copy
from pathlib import Path
from auditors import cluster_dsm_reference_v2 as ref
ROOT=Path(__file__).resolve().parents[1]

def fixture(form,c,g,i,seed):
 b=c*g;s=ref.shapes(form,c,g,i);a={k:[0]*__import__('math').prod(v) for k,v in s.items()}
 def set64(key,at,value):a[key][2*at]=value&0xffffffff;a[key][2*at+1]=value>>32
 for block in range(b):
  for k,v in enumerate([i,block%c,c,1,1,0]):set64('completion',block*6+k,v)
  for k,v in enumerate([1,2,3,4,200+block]):set64('stamps',block*5+k,v)
  for item in range(i):
   off=(block*i+item)*12;a['lifecycle'][off:off+12]=ref.lifecycle(form,c,block,item)
   if 'trace' in a:
    w=4096 if form.startswith('bulk_') else 1024
    a['trace'][(block*i+item)*w:(block*i+item+1)*w]=[ref.payload(form,c,block,item,word,seed) for word in range(w)]
   if 'tokens' in a and ref.mask(form,c)&(1<<(block%c)):set64('tokens',block*i+item,2**64-1) # opaque valid domain; not forbidden encoding
  if 'guards' in a:a['guards'][block*8:block*8+8]=[0xd15ea5e0+k for k in range(8)]
  if 'sums' in a:
   for t in range(128):a['sums'][block*128+t]=sum(ref.payload(form,c,block,item,t+128*j,seed) for item in range(i) for j in range(8))&0xffffffff
 if 'source_ring' in a:
  a['source_ring']=[(17*word+seed)&0xffffffff for word in range(g*32*4096)]
 return a

class ClusterReferenceTests(unittest.TestCase):
 def test_all_fixed_forms_clusters_pairs_and_exact_count(self):
  for c in (2,4,8):
   for form in ref.FORMS:
    for i,seed in ref.PAIRS:
     a=fixture(form,c,1,i,seed);r=ref.audit(form,c,1,i,seed,a)
     self.assertEqual(r['exact_checked_elements'],sum(map(len,a.values()))-r['opaque_token_words'])
 def test_wrong_rank_payload_mask_consumerdone_exit_and_bytes(self):
  for form in ref.FORMS:
   a=fixture(form,4,1,2,3)
   mutations=[('lifecycle',0,99),('lifecycle',3,99),('lifecycle',10,0),('lifecycle',11,0),('completion',8,0)]
   if 'trace' in a:mutations.append(('trace',0,a['trace'][0]^1))
   for key,index,v in mutations:
    b=copy.deepcopy(a);b[key][index]=v
    with self.assertRaises(ValueError):ref.audit(form,4,1,2,3,b)
   bad=copy.deepcopy(a);bad['lifecycle'].pop()
   with self.assertRaises(ValueError):ref.audit(form,4,1,2,3,bad)
 def test_multicast_source_request_and_receiver_accounting(self):
  one=ref.ledger('bulk_single_target',4,1,2);all_=ref.ledger('bulk_all_targets',4,1,2)
  self.assertEqual(one['source_request_bytes'],32768);self.assertEqual(all_['source_request_bytes'],32768)
  self.assertEqual(one['received_bytes'],32768);self.assertEqual(all_['received_bytes'],131072)
  self.assertEqual(one['collective_sync_stages'],4);self.assertEqual(all_['collective_sync_stages'],4)
 def test_nonreceiver_token_noarrival_and_stamp_integer_domain(self):
  a=fixture('bulk_single_target',4,1,1,0);a['tokens'][0]=1
  with self.assertRaises(ValueError):ref.audit('bulk_single_target',4,1,1,0,a)
  a=fixture('cluster_sync',2,1,1,0);a['stamps'][0]=1.0
  with self.assertRaises(ValueError):ref.audit('cluster_sync',2,1,1,0,a)
 def test_machine_coordinates_and_cpp_reference_agree(self):
  contract=json.loads((ROOT/'contracts/cluster_dsm_multicast_short_v1.draft.json').read_text());self.assertEqual(len(contract['cases']),42);self.assertEqual(len({c['id'] for c in contract['cases']}),42)
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);source=p/'ref.cpp';source.write_text('#include "'+str(ROOT/'common/cluster_dsm_reference_v1.hpp')+'"\n#include <cstdio>\nint main(){for(unsigned f=0;f<7;++f)for(unsigned c:{2u,4u,8u})for(unsigned b=0;b<c*2;++b){if(f<4)std::printf("%u %u %u %u\\n",f,c,b,cluster_reference::dsm(f>=2,f==1||f==3,c,b,4,127,0xffffffffu));for(unsigned k=0;k<12;++k)std::printf("L %u %u %u %u %u\\n",f,c,b,k,cluster_reference::life(f,c,b,4,k));}}')
   subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(p/'ref')],check=True,capture_output=True)
   result=subprocess.run([str(p/'ref')],check=True,capture_output=True,text=True)
   for line in result.stdout.splitlines():
    fields=line.split()
    if fields[0]=='L':f,c,b,k,v=map(int,fields[1:]);self.assertEqual(v,ref.lifecycle(ref.FORMS[f],c,b,4)[k])
    else:f,c,b,v=map(int,fields);self.assertEqual(v,ref.payload(ref.FORMS[f],c,b,4,127,4294967295))
 def test_adapter_abi_core_envelope_and_failclosed_sass(self):
  from auditors import cluster_dsm_validation_v2 as adapter
  from runners.validation_diagnostic import validation_envelope,profile_from,POLICY
  contract=json.loads((ROOT/'contracts/cluster_dsm_multicast_short_v1.draft.json').read_text());p=json.loads((ROOT/'contracts/cluster_dsm_multicast_validation_profiles_v1.draft.json').read_text())['profiles'][0]
  for case in contract['cases']:self.assertEqual(adapter.validation_argv('binary/probe',case,p,3),['binary/probe','validate-only',case['id'],p['id'],'3'])
  with self.assertRaises(ValueError):adapter.audit_sass('',contract)
  case=contract['cases'][0];form=case['form'];c=case['cluster_size'];b=c
  row={'schema_version':2,'validation_schema_version':1,'type':'validation','case_id':case['id'],'profile_id':p['id'],'seed':3,'scope':case['scope'],'threads':128,'blocks':b,'errors':0,'performance_eligible':False,'warmup_executed':False,'pilot_executed':False,'target_launches':[],'checks':[],'resource_identity':{'kernel_symbol':adapter.symbol(form,c),'registers_per_thread':32,'static_smem_bytes':4128,'dynamic_smem_bytes':0,'local_size_bytes':0,'occupancy_limit_ctas_per_sm':0,'extensions':{'cluster_launch_supported':1,'potential_cluster_size':8,'active_cluster_capacity':1,'cluster_size':c,'clusters':1,'mask':0,'opaque_token_words_preserved_domain_only':[0,0,0],'occupancy_is_upper_bound':True}}}
  for n,(i,seed) in enumerate(ref.PAIRS):
   layout=ref.shapes(form,c,1,i);count=sum(__import__('math').prod(v) for v in layout.values())
   row['target_launches'].append({'launch_index':n,'iterations':i,'input_profile':'nonuniform_uint32_cluster_paired_seeds_v1','threads':128,'blocks':b})
   row['checks'].append({'launch_index':n,'reference_model':p['reference_identity']['model'],'reference_sha256':p['reference_identity']['sha256'],'comparison':'exact','tolerance_id':None,'completed':True,'errors':0,'checked_elements':count,'expected_elements':count,'verified_CTA_ids':list(range(b)),'output_artifacts':[{'path':f'cluster_{n}_{leaf}.u32le','sha256':'1'*64,'dtype':'uint32','evidence_kind':'full_values','shape':shape} for leaf,shape in layout.items()]})
  policy=json.loads((ROOT.parents[1]/POLICY).read_text());document=json.loads((ROOT/'contracts/cluster_dsm_multicast_validation_profiles_v1.draft.json').read_text());self.assertEqual(profile_from(document,p['id'],policy),p);validation_envelope(row,case,p,3,policy)
  bad_profile=copy.deepcopy(document);bad_profile['profiles'][0]['input_profiles']=['old_seed0','old_seed3','old_seedUINTMAX']
  with self.assertRaisesRegex(ValueError,'profile exceeds launch budget'):profile_from(bad_profile,p['id'],policy)
  device={'schema_version':2,'type':'device','name':'GH200 CPUfixture','cc':'9.0','uuid':'GPU-00000000-0000-0000-0000-000000000000','sms':132,'driver_version':1,'runtime_version':12090,'registers_per_sm':65536,'smem_per_sm_bytes':233472,'smem_per_cta_optin_bytes':232448}
  from unittest.mock import patch
  with patch.dict(adapter.BASELINE,{},clear=True):
   with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,p,3) # absent actual B2 must still failclosed
  from unittest.mock import patch
  with patch.dict(adapter.BASELINE,{adapter.symbol(form,c):{'registers_per_thread':32,'static_smem_bytes':4128,'local_size_bytes':0}}):
   self.assertEqual(adapter.validate_validation(device,row,case,p,3)['target_launches'],3)
   for key,value in [('active_cluster_capacity',0),('cluster_size',8),('mask',1)]:
    bad=copy.deepcopy(row);bad['resource_identity']['extensions'][key]=value
    with self.assertRaises(ValueError):adapter.validate_validation(device,bad,case,p,3)
if __name__=='__main__':unittest.main()
