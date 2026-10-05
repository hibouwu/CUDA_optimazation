"""S16 controller A: finite CPU budget/state/archive checks, no GPU or query."""
import copy,hashlib,io,json,os,tarfile,tempfile,time,unittest,subprocess,sys
from pathlib import Path
from common.suite_io import digest
from runners.s16_short_controller import coordinates,footprint,artifact_bytes,quota_budget,quota_headroom,next_action,validate_archived,device_binding,SUITE,MAX_FIRST
ROOT=Path(__file__).resolve().parents[1]
class ControllerTests(unittest.TestCase):
 def setUp(self):
  self.contract=json.loads((ROOT/'contracts/tma_stage_request_short_v1.draft.json').read_text());self.profiles=json.loads((ROOT/'contracts/tma_stage_request_validation_profiles_v1.draft.json').read_text());self.profile=self.profiles['profiles'][0];self.device={'sms':132,'smem_per_sm_bytes':233472,'smem_per_cta_optin_bytes':232448,'registers_per_sm':65536};self.points=coordinates(self.contract,self.profiles);self.identity={'uuid':'CPU-only','source':'a'*64,'controller':'b'*64}
 def test36_grouped128launches_max_first_and4capacityreject(self):
  result=footprint(self.contract,self.profiles,self.device);self.assertEqual(self.points[0]['case_id'],MAX_FIRST);self.assertEqual(len(self.points),36);self.assertEqual(len({r['case_id'] for r in self.points}),36);self.assertEqual(result['legal_target_launches'],128);self.assertEqual(result['capacity_reject_cases'],4);self.assertEqual(result['capacity_reject_target_launches'],0);self.assertEqual(result['max_artifact_bytes_upper_bound'],4490994944);self.assertTrue(all(p['seed']==3 for p in self.points));self.assertFalse(result['actual_occupancy_claim'])
 def test_raw_footprint_includes4rings_allslots_all41items_and_opaque(self):
  case=next(c for c in self.contract['cases'] if c['id']==MAX_FIRST);result=artifact_bytes(case,self.profile,self.device);b=result['blocks_upper_bound'];self.assertEqual(b,396);self.assertEqual(result['trace_bytes'],b*41*4*16384);self.assertEqual(result['final_shared_slot_bytes'],4*b*1*4*16384);self.assertEqual(result['ring_and_guard_bytes'],4*(b*32*4*16384+32));self.assertEqual(result['lifecycle_bytes_including_opaque_token'],b*41*64);self.assertEqual(result['opaque_token_preserved_domain_only_words'],2*b*41);self.assertFalse(result['actual_occupancy_claim'])
 def test_quota_budget_keeps_worst_seal_and_no_compression_assumption(self):
  size=4490994944;headroom=20*1024**3;good=quota_budget(0,size,headroom);self.assertTrue(good['allow']);self.assertGreater(good['worst_case_seal_bytes'],size);self.assertFalse(quota_budget(0,size,headroom,4*1024**3)['allow']);self.assertFalse(quota_budget(0,size,size)['allow']);self.assertEqual(quota_budget(0,size,size)['status'],'quota_checkpoint_before_next_target')
 def test_fresh_quota_rejects_stale_user_path_and_digest(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td)/'repo';repo.mkdir();(repo/'quota.stdout').write_text('original query');data={'schema_version':1,'status':'current_quota_query_verified','queried_unix_ns':time.time_ns(),'execution_uid':os.geteuid(),'filesystem_root':td,'hard_bytes':20*1024**3,'used_bytes':1024**3,'reserve_bytes':512*1024**2,'query_stdout':{'path':'quota.stdout','sha256':hashlib.sha256((repo/'quota.stdout').read_bytes()).hexdigest()}};p=repo/'quota.json';p.write_text(json.dumps(data));self.assertGreater(quota_headroom(repo,p,{'execution_uid':os.geteuid()}),0)
   for field,value in [('queried_unix_ns',time.time_ns()-61_000_000_000),('execution_uid',-1),('filesystem_root','/'),('reserve_bytes',0),('used_bytes',20*1024**3)]:
    bad=dict(data);bad[field]=value;p.write_text(json.dumps(bad))
    with self.assertRaises(ValueError):quota_headroom(repo,p,{'execution_uid':os.geteuid()})
   bad=copy.deepcopy(data);bad['query_stdout']['sha256']='0'*64;p.write_text(json.dumps(bad))
   with self.assertRaises(ValueError):quota_headroom(repo,p,{'execution_uid':os.geteuid()})
 def test_source_controller_uuid_drift_and_missing_passed_never_run(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);root=repo/'run';p=self.points[0];entry={'coordinate':p,'state':'resident_passed','family_identity_sha256':digest(self.identity)}
   with self.assertRaises(ValueError):next_action(root,entry,repo,p,self.identity)
   changed=dict(self.identity,controller='changed')
   with self.assertRaises(ValueError):next_action(root,entry,repo,p,changed)
   changed=dict(self.identity,uuid='changed')
   with self.assertRaises(ValueError):next_action(root,entry,repo,p,changed)
 def test_pending_resume_passed_audit_and_failed_stop(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);root=repo/'run';root.mkdir();(root/'attempts').mkdir();p=self.points[0];entry={'coordinate':p,'state':'pending','family_identity_sha256':digest(self.identity)}
   (root/'validation_state.json').write_text('{"status":"checkpoint"}');self.assertEqual(next_action(root,entry,repo,p,self.identity),'resume');(root/'attempts/attempt_00').mkdir()
   with self.assertRaises(ValueError):next_action(root,entry,repo,p,self.identity)
   (root/'validation_state.json').write_text('{"status":"case_diagnostic_passed"}');self.assertEqual(next_action(root,entry,repo,p,self.identity),'audit')
   entry['state']='failed'
   with self.assertRaises(ValueError):next_action(root,entry,repo,p,self.identity)
 def test_capacity_terminal_is_zero_target_and_keeps_no_diagnostic_run(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);p=next(x for x in self.points if '_s4_r4_' in x['case_id']);entry={'coordinate':p,'state':'resource_reject_before_launch','target_launches':0,'family_identity_sha256':digest(self.identity)};self.assertEqual(next_action(repo/'absent',entry,repo,p,self.identity),'verify_capacity_terminal');legal=copy.deepcopy(entry);legal['coordinate']=self.points[0]
   with self.assertRaises(ValueError):next_action(repo/'absent',legal,repo,self.points[0],self.identity)
   entry['target_launches']=1
   with self.assertRaises(ValueError):next_action(repo/'absent',entry,repo,p,self.identity)
 def test_archive_missing_resident_full_decode_prevents_remeasurement(self):
  from s16_controller_archive_fixture import payload,pack
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);point,run,data=payload(repo);entry=pack(repo,point,run,data,self.identity,'good')
   self.assertEqual(validate_archived(entry,repo,point),'archived_verified')
   self.assertEqual(next_action(repo/run,entry,repo,point,self.identity),'archive_audit')
   bad=copy.deepcopy(entry);bad['manifest_sha256']='0'*64
   with self.assertRaises(ValueError):next_action(repo/run,bad,repo,point,self.identity)
   bad=copy.deepcopy(entry);bad['offhost_receipt']['sha256']='0'*64
   with self.assertRaises(ValueError):next_action(repo/run,bad,repo,point,self.identity)
 def test_coherent_archive_wrong_launch_role_geometry_and_attempt_negatives(self):
  from s16_controller_archive_fixture import payload,pack
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);point,run,original=payload(repo);raw_name='attempts/attempt_00/raw.jsonl'
   def mutate_raw(data,fn):
    rows=[json.loads(x) for x in data[raw_name].decode().splitlines()];fn(rows);data[raw_name]=('\n'.join(json.dumps(x) for x in rows)+'\n').encode()
   mutations=[lambda d:mutate_raw(d,lambda rows:rows[1]['target_launches'][0].update(iterations=2)),lambda d:mutate_raw(d,lambda rows:rows[1]['target_launches'].reverse()),lambda d:mutate_raw(d,lambda rows:rows[1]['target_launches'][0].pop('input_profile')),lambda d:mutate_raw(d,lambda rows:rows[1]['target_launches'][0].update(blocks=2)),lambda d:mutate_raw(d,lambda rows:rows[1].update(threads=64)),lambda d:mutate_raw(d,lambda rows:rows[1].update(blocks=2)),lambda d:mutate_raw(d,lambda rows:rows[1]['resource_identity'].update(occupancy_limit_ctas_per_sm=99)),lambda d:d.update({'attempts/attempt_01/raw.jsonl':d[raw_name]}),lambda d:d.update({'failure_summary.json':b'{"status":"case_diagnostic_failed"}'})]
   for index,mutation in enumerate(mutations):
    data=dict(original);mutation(data);entry=pack(repo,point,run,data,self.identity,'bad'+str(index))
    with self.assertRaises(ValueError):validate_archived(entry,repo,point)
   for index,(file,field,value) in enumerate([('validation_spec.json','family_B3_eligible',True),('diagnostic_summary.json','status','case_diagnostic_failed'),('attempts/attempt_00/raw.jsonl.registry.json','state','cleanup_unconfirmed')]):
    data=dict(original);obj=json.loads(data[file]);obj[field]=value;data[file]=json.dumps(obj).encode();entry=pack(repo,point,run,data,self.identity,'role'+str(index))
    with self.assertRaises(ValueError):validate_archived(entry,repo,point)
   data=dict(original);reg=json.loads(data['attempts/attempt_00/raw.jsonl.registry.json']);reg['registry_id']='overlap';reg['argv']=[reg['argv'][0],'device'];reg['cleanup_receipt']['argv']=reg['argv'];data['environment/device_overlap.stdout.registry.json']=json.dumps(reg).encode();entry=pack(repo,point,run,data,self.identity,'overlap')
   with self.assertRaisesRegex(ValueError,'overlapping registered'):validate_archived(entry,repo,point)
 def test_bound_readonly_query_binary_uuid_job_receipt_negatives(self):
  with tempfile.TemporaryDirectory() as td:
   repo=Path(td);device={**self.device,'uuid':'GPU-CPU','type':'device','cc':'9.0','name':'CPU fixture GH200'};p=repo/'device.json';p.write_text(json.dumps(device));stdout=repo/'query.stdout';stdout.write_bytes(p.read_bytes());binary=repo/'reviewed-binary';binary.write_bytes(b'CPU fixture, never executable');reviewpath=repo/'source-B.json';review={'gate_files':{'original/diagnostics/tma_stage_request/probe':hashlib.sha256(binary.read_bytes()).hexdigest()}};reviewpath.write_text(json.dumps(review));env={'uuid':'GPU-CPU','name':'CPU fixture GH200','driver':'fixture','compiler':'fixture12.9','tools':{},'execution_uid':os.geteuid()};allocation={**env,'job':'123'};argv=[str(binary),'device'];receipt={'returncode':0,'timed_out':False,'cleanup_confirmed':True,'timeout_seconds':30,'argv':argv,'stdout_sha256':hashlib.sha256(stdout.read_bytes()).hexdigest(),'gpu_process_registration':{'state':'cleanup_confirmed','gpu_uuid':'GPU-CPU','argv':argv}};binding={'schema_version':1,'device_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'environment':env,'allocation_job':'123','source_review_sha256':hashlib.sha256(reviewpath.read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'stdout':{'path':'query.stdout','sha256':hashlib.sha256(stdout.read_bytes()).hexdigest()},'receipt':receipt};self.assertEqual(device_binding(repo,p,binding,allocation,review,reviewpath),device)
   for field,value in [('allocation_job','999'),('binary_sha256','0'*64),('source_review_sha256','0'*64)]:
    bad=copy.deepcopy(binding);bad[field]=value
    with self.assertRaises(ValueError):device_binding(repo,p,bad,allocation,review,reviewpath)
   for field,value in [('timeout_seconds',120),('argv',[str(binary),'case']),( 'returncode',1)]:
    bad=copy.deepcopy(binding);bad['receipt'][field]=value
    with self.assertRaises(ValueError):device_binding(repo,p,bad,allocation,review,reviewpath)
   binary.write_bytes(b'changed executable')
   with self.assertRaises(ValueError):device_binding(repo,p,binding,allocation,review,reviewpath)
 def test_plan_and_run_draft_do_not_query_or_launch(self):
  env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONDONTWRITEBYTECODE':'1'}
  plan=subprocess.run([sys.executable,'-B','-m','runners.s16_short_controller','plan','--repo',str(ROOT.parents[1]),'--start','0','--stop','1'],env=env,capture_output=True,text=True);self.assertEqual(plan.returncode,0,plan.stderr);self.assertEqual(json.loads(plan.stdout)['case_id'],MAX_FIRST)
  run=subprocess.run([sys.executable,'-B','-m','runners.s16_short_controller','run','--repo',str(ROOT.parents[1])],env=env,capture_output=True,text=True);self.assertNotEqual(run.returncode,0);self.assertIn('execution disabled',run.stderr)
if __name__=='__main__':unittest.main()
