"""Actual S14 formal host + real bounded warmup/emitter on CPU CUDA shim."""
import json,os,sys,subprocess,tempfile,unittest,hashlib,struct,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from test_tma_bulk_validation import SHIM,pairs
from auditors import tma_bulk_formal_v1,tma_bulk_formal_validation_v1 as adapter
from runners import validation_diagnostic as core
PROFILE=json.loads((ROOT/'contracts/tma_bulk_formal_validation_profiles_v1.draft.json').read_text())['profiles'][0]
POLICY=json.loads((ROOT/'contracts/early_validation_v1.json').read_text())
CONTRACT=json.loads((ROOT/'contracts/tma_bulk.json').read_text())
PROTOCOL=json.loads((ROOT/'contracts/protocol.json').read_text())
FAKE_LAUNCH=r'''
int cudaEventElapsedTime(float* x,int,int){*x=1.1f;return 0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3,void** args,size_t shared){
 const std::string failure=std::getenv("S14_STUB_FAILURE")?std::getenv("S14_STUB_FAILURE"):"";
 if(failure=="cuda")return 1;
 auto* global=*static_cast<unsigned**>(args[0]);int iterations=*static_cast<int*>(args[1]);unsigned seed=*static_cast<unsigned*>(args[2]);
 if(*static_cast<bool*>(args[3]))throw std::runtime_error("formal capture enabled");
 auto* final=*static_cast<unsigned**>(args[4]);auto* stamps=*static_cast<gh::Stamp**>(args[5]);auto* counts=*static_cast<std::uint64_t**>(args[6]);
 const unsigned words=selected_q/4;
 if(shared!=selected_q+32)return 2;
 for(unsigned b=0;b<grid.x;++b){
  if(stamps[b].smid!=~unsigned(0)||counts[b*5]!=~std::uint64_t(0))throw std::runtime_error("control poison missing");
  for(unsigned slot=0;slot<32;++slot)for(unsigned w=0;w<words;++w){
   unsigned expected=selected_g2s?17u*unsigned((size_t(b)*32+slot)*words+w)+seed:~(29u*unsigned(size_t(b)*words+w)+seed);
   if(global[(size_t(b)*32+slot)*words+w]!=expected)throw std::runtime_error("ring init missing");
  }
  if(selected_g2s)for(unsigned w=0;w<words;++w){
   const unsigned value=17u*unsigned((size_t(b)*32+(iterations-1)%32)*words+w)+seed;
   if(final[size_t(b)*words+w]!=~value)throw std::runtime_error("final tile poison missing");
   if(!(failure=="missing_final_last"&&b==grid.x-1&&w==words-1))final[size_t(b)*words+w]=value;
  }
  else for(unsigned slot=0;slot<unsigned(std::min(iterations,32));++slot)for(unsigned w=0;w<words;++w)
   global[(size_t(b)*32+slot)*words+w]=29u*unsigned(size_t(b)*words+w)+seed;
  stamps[b]={1000,1001000,2000,2002000,b%2};counts[b*5]=iterations;counts[b*5+1]=selected_g2s?iterations*2:0;counts[b*5+2]=0;counts[b*5+3]=0;counts[b*5+4]=selected_g2s?0:iterations;
 }
 if(failure=="zero_timer")for(unsigned b=0;b<grid.x;++b){stamps[b].end_ns=stamps[b].begin_ns;stamps[b].end_cycle=stamps[b].begin_cycle;}
 if(failure=="guard")global[-1]^=1;
 if(failure=="last_ring"&&!selected_g2s)global[size_t(grid.x)*32*words-1]^=1;
 if(failure=="stamp")stamps[grid.x-1].smid=~unsigned(0);
 if(failure=="timeout")counts[2]=1;
 if(failure=="full_wait")counts[4]=~std::uint64_t(0);
 static unsigned launched=0;++launched;
 if(failure=="measurement_guard"&&launched==9)global[-1]^=1;
 if(const char* log=std::getenv("S14_LAUNCH_LOG")){std::ofstream f(log,std::ios::app);f<<iterations<<" "<<seed<<"\n";}
 return 0;
}
'''

class TmaBulkFormalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.binary=Path(cls.temp.name)/'host'
  shim=SHIM[:SHIM.index('int cudaLaunchKernel')]
  runtime=(ROOT/'common/probe_runtime.cuh').read_text();actual=runtime[runtime.index('struct Observation {'):].rsplit('}',1)[0]
  shim='#include <cmath>\n#include <iomanip>\n#include <utility>\n#define GH_TIMER_RESOLUTION_V3 0\n'+shim
  shim+='\nnamespace gh {enum class TimerPolicy {StrictPositiveNs,ZeroWorkOneCta,ZeroWorkGrid};\n'+actual+'\n}\n'+FAKE_LAUNCH
  old=(ROOT/'probes/tma_bulk.cu').read_text();host=old[old.index('struct TbCase'):].replace('int main(','int tma_bulk_original_short_main(')
  # Original short main needs its capture-aware CUDA shim; keep it uncalled
  # in these formal-path tests and omit its body while retaining tables/emitter.
  host=host[:host.index('int tma_bulk_original_short_main')]+'int tma_bulk_original_short_main(int,char**){return 9;}\n'
  new=(ROOT/'probes/tma_bulk_formal_v1.cu').read_text();new=new[new.index('int main('):]
  source=Path(cls.temp.name)/'host.cpp';source.write_text(shim.replace('REFERENCE_PATH',str(ROOT/'common/tma_bulk_reference.hpp')).replace('ARTIFACT_PATH',str(ROOT/'common/word_artifacts.hpp'))+host+new)
  result=subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(cls.binary)],capture_output=True,text=True,timeout=30)
  if result.returncode:raise RuntimeError(result.stderr)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def run_host(self,case,length=128,seed=3,failure='',short=False):
  with tempfile.TemporaryDirectory(dir=self.temp.name) as d:
   log=Path(d)/'launch.log';argv=[str(self.binary),'validate-only',case['id'],'formal_final_1_2_33_v1','3'] if short else [str(self.binary),case['id'],str(length),str(seed)]
   run=subprocess.run(argv,cwd=d,capture_output=True,text=True,timeout=30,env={**os.environ,'S14_STUB_FAILURE':failure,'S14_LAUNCH_LOG':str(log)})
   rows=[json.loads(line,object_pairs_hook=pairs) for line in run.stdout.splitlines()]
   launches=log.read_text().splitlines() if log.exists() else []
   if short and run.returncode==0:
    self.assertEqual(len(core.verify_output_artifacts(Path(d),rows[1])),12)
    arrays=[]
    for check in rows[1]['checks']:
     values={}
     for artifact in check['output_artifacts']:
      data=(Path(d)/artifact['path']).read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),artifact['sha256'])
      values[artifact['path']]={'shape':artifact['shape'],'values':list(struct.unpack('<'+'I'*(len(data)//4),data))}
     arrays.append(values)
    core.validation_envelope(rows[1],case,PROFILE,3,POLICY)
    adapter.validate_validation(rows[0],rows[1],case,PROFILE,3)
    tma_bulk_formal_v1.audit_formal_path_values(rows[0],case,rows[1],arrays)
    self.last_arrays=arrays
   return run,rows,launches
 def test_all24_formal_real_warmup_and_emitter(self):
  for case in CONTRACT['cases']:
   for seed in (0,3,4294967295):
    with self.subTest(case=case['id'],seed=seed):
     run,rows,launches=self.run_host(case,seed=seed);self.assertEqual(run.returncode,0,run.stderr)
     self.assertEqual(launches,[f'128 {seed}']*9)
     tma_bulk_formal_v1.validate_trial(rows[1],case,rows[0],seed,PROTOCOL)
 def test_all24_short_exact_three_pairs_no_warmup(self):
  for case in CONTRACT['cases']:
   run,rows,launches=self.run_host(case,short=True);self.assertEqual(run.returncode,0,run.stderr)
   self.assertEqual(launches,['1 0','2 3','33 4294967295'])
   self.assertFalse(rows[1]['warmup_executed']);self.assertEqual(len(rows[1]['checks']),3)
 def test_numeric_guard_control_failure_never_emits_trial(self):
  for case in (CONTRACT['cases'][0],CONTRACT['cases'][2]):
   for failure in ('guard','stamp','timeout','full_wait','cuda','missing_final_last' if case['parameters']['direction']=='gmem_to_smem' else 'last_ring'):
    run,rows,launches=self.run_host(case,failure=failure)
    self.assertNotEqual(run.returncode,0);self.assertFalse(any(r.get('type')=='trial' for r in rows));self.assertLessEqual(len(launches),1)

 def test_independent_final_word_lifecycle_and_quantity_mutations(self):
  for case in (CONTRACT['cases'][0],CONTRACT['cases'][2]):
   run,rows,_=self.run_host(case,short=True);self.assertEqual(run.returncode,0,run.stderr)
   for kind in ('payload','guard','count','shape','missing','pair','quantity','stamp'):
    row=copy.deepcopy(rows[1]);arrays=copy.deepcopy(self.last_arrays)
    if kind=='pair':row['target_launches'][2]['iterations']=2
    elif kind=='quantity':row['checks'][0]['checked_elements']+=1
    elif kind=='stamp':arrays[0]['formal_0_stamps.u32le']['values'][8]=4294967295
    else:
     name=next(n for n in arrays[2] if (('final_tile' in n or 'ring' in n) if kind in ('payload','shape','missing') else ('guard' in n if kind=='guard' else 'completion' in n)))
     if kind=='missing':arrays[2].pop(name)
     elif kind=='shape':arrays[2][name]['shape'][-1]+=1
     else:arrays[2][name]['values'][-1]^=1
    with self.subTest(kind=kind),self.assertRaises(ValueError):tma_bulk_formal_v1.audit_formal_path_values(rows[0],case,row,arrays)
 def test_formal_measured_launch_corruption_after_valid_warmup(self):
  run,rows,launches=self.run_host(CONTRACT['cases'][0],failure='measurement_guard')
  self.assertNotEqual(run.returncode,0);self.assertEqual(len(launches),9);self.assertFalse(any(r.get('type')=='trial' for r in rows))

 def test_formal_length_bounds_and_independent_request_accounting(self):
  case=CONTRACT['cases'][0]
  for length in (1,2,33,128,65536):
   run,rows,_=self.run_host(case,length=length,seed=4294967295);self.assertEqual(run.returncode,0,run.stderr)
   tma_bulk_formal_v1.validate_trial(rows[1],case,rows[0],4294967295,PROTOCOL)
   for key in ('work_count','read_payload_bytes','write_payload_bytes','completed_requests','global_allocation_bytes','capture_enabled'):
    row=copy.deepcopy(rows[1]);row[key]=not row[key] if isinstance(row[key],bool) else row[key]+1
    with self.subTest(length=length,key=key),self.assertRaises(ValueError):tma_bulk_formal_v1.validate_trial(row,case,rows[0],4294967295,PROTOCOL)
  for length in (0,65537):
   run,rows,launches=self.run_host(case,length=length);self.assertNotEqual(run.returncode,0);self.assertEqual(launches,[])

 def test_zero_resolution_short_is_correctness_only_formal_rejects(self):
  case=CONTRACT['cases'][0]
  run,rows,launches=self.run_host(case,short=True,failure='zero_timer');self.assertEqual(run.returncode,0,run.stderr)
  self.assertEqual(len(launches),3);self.assertFalse(rows[1]['performance_eligible'])
  run,rows,_=self.run_host(case,failure='zero_timer');self.assertNotEqual(run.returncode,0);self.assertFalse(any(r.get('type')=='trial' for r in rows))

 def test_short_core_abi_profile_and_impossible_resources_rejected(self):
  core.profile_from({'schema_version':1,'profiles':[PROFILE]},PROFILE['id'],POLICY)
  case=CONTRACT['cases'][0];self.assertEqual(adapter.validation_argv('binary/probe',case,PROFILE,3),['binary/probe','validate-only',case['id'],PROFILE['id'],'3'])
  run,rows,_=self.run_host(case,short=True);self.assertEqual(run.returncode,0,run.stderr)
  for kind in ('regs','occupancy','one_cta_grid','wrong_device','timestamp_float','timestamp_bool','timestamp_negative','timestamp_overflow','smid_overflow'):
   device=copy.deepcopy(rows[0]);row=copy.deepcopy(rows[1]);arrays=copy.deepcopy(self.last_arrays)
   if kind=='regs':row['resource_identity']['registers_per_thread']=999
   elif kind=='occupancy':row['resource_identity']['occupancy_limit_ctas_per_sm']=99
   elif kind=='one_cta_grid':row['blocks']=2
   elif kind=='wrong_device':device['cc']='8.0'
   else:
    words=arrays[0]['formal_0_stamps.u32le']['values']
    if kind=='timestamp_float':words[0]=1000.5
    elif kind=='timestamp_bool':words[0]=True
    elif kind=='timestamp_negative':words[0]=-1
    elif kind=='timestamp_overflow':words[0]=2**32
    else:words[9]=1
   with self.subTest(kind=kind),self.assertRaises(ValueError):tma_bulk_formal_v1.audit_formal_path_values(device,case,row,arrays)
  for seed in (0,4294967295):
   with self.assertRaises(ValueError):adapter.validation_argv('binary/probe',case,PROFILE,seed)

 def test_group_cli_seed_reject_and_old_profile_forwarding(self):
  case=CONTRACT['cases'][0]
  for seed in ('0','4294967295'):
   run=subprocess.run([str(self.binary),'validate-only',case['id'],PROFILE['id'],seed],capture_output=True,text=True,timeout=30)
   self.assertEqual(run.returncode,2);self.assertFalse('"type":"validation"' in run.stdout)
  # The CPU fixture marks the original short entry with return9; actual old
  # short CUDA/host coverage remains in the untouched original test suite.
  run=subprocess.run([str(self.binary),'validate-only',case['id'],'bulk_short_1_seed0_v1','0'],capture_output=True,text=True,timeout=30)
  self.assertEqual(run.returncode,9)

if __name__=='__main__':unittest.main()
