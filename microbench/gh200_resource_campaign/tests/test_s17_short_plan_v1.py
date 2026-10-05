"""Finite planner/query-role CPU fixtures only; no hardware support evidence."""
import unittest,json,copy
from pathlib import Path
from runners import s17_short_plan_v1 as p
from auditors.cluster_dsm_sass_baseline_v1 import BASELINE
from auditors.cluster_dsm_reference_v1 import FORMS
from auditors.cluster_dsm_validation_v1 import symbol
ROOT=Path(__file__).resolve().parents[1]
CONTRACT=json.loads((ROOT/'contracts/cluster_dsm_multicast_short_v1.draft.json').read_text());PROFILE=json.loads((ROOT/'contracts/cluster_dsm_multicast_validation_profiles_v1.draft.json').read_text())['profiles'][0]
DEVICE={'schema_version':2,'type':'device','cc':'9.0','name':'GH200 CPU fixture','uuid':'GPU-00000000-0000-0000-0000-000000000000','sms':96,'driver_version':1,'runtime_version':12090,'registers_per_sm':65536,'smem_per_sm_bytes':233472,'smem_per_cta_optin_bytes':232448}

def rows():
 out=[]
 for c in (2,4,8):
  for f in FORMS:
   s=symbol(f,c);r=BASELINE[s];out.append({'type':'cluster_capability','kernel_symbol':s,'cluster_size':c,'cluster_launch_supported':1,'potential_cluster_size':8,'active_cluster_capacity':7,'cuda_error':0,'registers_per_thread':r['registers_per_thread'],'static_smem_bytes':r['static_smem_bytes'],'local_size_bytes':0,'occupancy_is_upper_bound':True})
 return out
class PlannerTests(unittest.TestCase):
 def test_42_profiles_grid_from_query_not_sms_and_complete_bytes(self):
  plan=p.plan(CONTRACT,PROFILE,DEVICE,rows());self.assertEqual(plan['supported_coordinates'],42);self.assertEqual(plan['planned_target_launches'],126)
  for point in plan['coordinates']:self.assertEqual(point['clusters'],1 if point['scope']=='one_cluster' else 7)
  maximum=plan['coordinates'][plan['maximum_index']];self.assertEqual(maximum['artifact_bytes_upper_bound'],plan['maximum_artifact_bytes_upper_bound'])
  device=dict(DEVICE,sms=132);self.assertEqual(p.plan(CONTRACT,PROFILE,device,rows())['total_artifact_bytes_upper_bound'],plan['total_artifact_bytes_upper_bound'])
 def test_query_error_unsupported_zero_capacity_and_protocol_bound_not_pass(self):
  for key,value,status in [('cuda_error',801,'query_error_before_target'),('cluster_launch_supported',0,'cluster_launch_not_supported'),('potential_cluster_size',1,'selected_C_not_supported'),('active_cluster_capacity',0,'zero_active_cluster_capacity'),('active_cluster_capacity',1025,'protocol_G_bound_exceeded')]:
   a=rows();a[0][key]=value;plan=p.plan(CONTRACT,PROFILE,DEVICE,a);matched=[x for x in plan['coordinates'] if x['form']=='local_read' and x['cluster_size']==2];self.assertEqual(len(matched),2)
   for x in matched:self.assertEqual(x['status'],status);self.assertEqual(x['target_launches'],0);self.assertEqual(x['artifact_bytes_upper_bound'],0)
 def test_missing_duplicate_wrong_actual_resources_and_domains_reject(self):
  for mutate in [lambda a:a.pop(),lambda a:a.reverse(),lambda a:a[0].update(registers_per_thread=999),lambda a:a[0].update(cluster_size=4),lambda a:a[0].update(active_cluster_capacity=1.5),lambda a:a[0].update(extra='unregistered')]:
   a=rows();mutate(a)
   with self.assertRaises(ValueError):p.plan(CONTRACT,PROFILE,DEVICE,a)
 def test_query_JSON_duplicate_nonfinite_and_missing_rows_reject(self):
  raw=('\n'.join(json.dumps(x) for x in [DEVICE,*rows()])+'\n').encode();d,r=p.strict_query_rows(raw);self.assertEqual(d,DEVICE);self.assertEqual(len(r),21)
  for bad in (raw.replace(b'"type": "device"',b'"type":"device","type":"device"',1),raw.replace(b'96',b'NaN',1),b'{}\n'):
   with self.assertRaises(ValueError):p.strict_query_rows(bad)
if __name__=='__main__':unittest.main()
