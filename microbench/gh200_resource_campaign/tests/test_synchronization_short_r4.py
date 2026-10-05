"""Actual r4 short host, CPU CUDA shim and independent full-array validation."""
import copy,hashlib,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auditors import synchronization_short_r4 as adapter
from runners import validation_diagnostic as core
LEGACY=json.loads((ROOT/'contracts/synchronization_lowering_v3.json').read_text())['cases']
PROFILES=json.loads((ROOT/'contracts/synchronization_short_profiles_r4.draft.json').read_text())['profiles']
POLICY=json.loads((ROOT/'contracts/early_validation_v1.json').read_text())
PAIR_CASES=[{'id':n,'scope':'one_cta','threads':32} for n in ('warp_divergent_shared_pair','cta_cross_half_shared_pair')]
SHIM=r'''
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "REFERENCE_PATH"
#include "PAIR_REFERENCE_PATH"
#include "BINDINGS_PATH"
#include "WRITER_PATH"
namespace gh {using u64=unsigned long long;struct Stamp{u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};struct Device{};Device device(){return {};}
void emit_device(const Device&){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"CPU shim GH200\",\"uuid\":\"GPU-00000000-0000-0000-0000-000000000000\",\"sms\":2,\"driver_version\":12090,\"runtime_version\":12090,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":232448,\"l2_cache_bytes\":62914560,\"global_memory_bytes\":1073741824}\n";}
std::string quote(const std::string& s){return "\""+s+"\"";}u64 integer(const char* s,u64 lo,u64 hi){char* end;auto x=std::strtoull(s,&end,10);if(*end||x<lo||x>hi)throw std::runtime_error("integer");return x;}}
struct SyncThread{unsigned value,errors,timeout,smid;gh::u64 completed,attempts,checked,arrival_cycle,departure_cycle;};
struct cudaFuncAttributes{int numRegs=0;size_t sharedSizeBytes=0,localSizeBytes=0;};struct dim3{unsigned x;dim3(unsigned n):x(n){}};
int selected_mode=0,selected_threads=32,selected_purpose=0;bool selected_skew=false,selected_pair=false;cudaFuncAttributes selected_attrs;
template<int Mode,int Threads,bool Skew,int Purpose> void simulated_sync(int,unsigned,gh::Stamp*,SyncThread*){
 selected_mode=Mode;selected_threads=Threads;selected_purpose=Purpose;selected_skew=Skew;selected_pair=false;
 unsigned regs=Purpose==2?(Mode==2?25:22):Purpose==1?(Mode==2?24:20):Mode==2?21:Mode<2?(Skew?18:20):Mode==5?20:18;
 selected_attrs={int(regs),size_t(Threads*8+(Mode==2?32:16)),0};}
using SyncKernel=void(*)(int,unsigned,gh::Stamp*,SyncThread*);
struct SyncCase{const char* id;int mode,threads,iterations;bool skew;SyncKernel measured,correctness,arrival;};
template<int Mode,int Threads,bool Skew> SyncCase make_case(const char* id){return {id,Mode,Threads,Mode<3?2048:8192,Skew,simulated_sync<Mode,Threads,Skew,0>,simulated_sync<Mode,Threads,Skew,1>,simulated_sync<Mode,Threads,Skew,2>};}
void warp_divergent_pair_v2(int,unsigned,unsigned,unsigned*,std::uint64_t*){selected_pair=true;selected_attrs={22,136,0};}
void cta_divergent_pair_v2(int,unsigned,unsigned,unsigned*,std::uint64_t*){selected_pair=true;selected_attrs={25,136,0};}
void select(const void* f){if(f==reinterpret_cast<const void*>(warp_divergent_pair_v2))warp_divergent_pair_v2(0,0,0,nullptr,nullptr);else if(f==reinterpret_cast<const void*>(cta_divergent_pair_v2))cta_divergent_pair_v2(0,0,0,nullptr,nullptr);else reinterpret_cast<SyncKernel>(const_cast<void*>(f))(0,0,nullptr,nullptr);}
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA CPU failure: " #call);}while(0)
int cudaFuncGetAttributes(cudaFuncAttributes* a,const void* f){select(f);*a=selected_attrs;return 0;}int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* x,const void*,int,size_t){*x=1;return 0;}
template<class T> int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::malloc(n));return *p?0:1;}int cudaFree(void* p){std::free(p);return 0;}int cudaMemset(void* p,int c,size_t n){std::memset(p,c,n);return 0;}int cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}int cudaDeviceSynchronize(){return 0;}
int synchronization_original_main(int,char**){return 9;}
int cudaLaunchKernel(const void* target,dim3 grid,dim3 threads,void** args,size_t){select(target);if(grid.x!=1)return 1;
 const std::string failure=std::getenv("S11_STUB_FAILURE")?std::getenv("S11_STUB_FAILURE"):"";const int length=*static_cast<int*>(args[0]);
 if(selected_pair){
  if(threads.x!=32||*static_cast<unsigned*>(args[1])!=0xffffffffu)return 1;
  const unsigned seed=*static_cast<unsigned*>(args[2]);auto* output=*static_cast<unsigned**>(args[3]);auto* clocks=*static_cast<std::uint64_t**>(args[4]);
  for(unsigned tid=0;tid<32;++tid){unsigned initial=seed+17*tid,peer=tid^16u,peer_initial=seed+17*peer+(peer>=16?324508639u:0u),sum=0,last=0;for(unsigned phase=1;phase<=unsigned(length)*8;++phase){last=peer_initial^(phase*2246822519u);sum+=last;}
   unsigned expected[]={initial,last,sum,unsigned(length)*8};for(unsigned j=0;j<4;++j){if(output[tid*4+j]!=~expected[j])throw std::runtime_error("pair poison missing");output[tid*4+j]=expected[j];}}
  if(clocks[0]!=~std::uint64_t(0)||clocks[1]!=~std::uint64_t(0))throw std::runtime_error("pair clock poison missing");clocks[0]=(1ull<<53)+3;clocks[1]=clocks[0]+64;if(failure=="last")output[127]^=1;if(failure=="clock")clocks[1]=~std::uint64_t(0);
 }else{
  if(threads.x!=unsigned(selected_threads))return 1;
  const unsigned seed=*static_cast<unsigned*>(args[1]);auto* stamp=*static_cast<gh::Stamp**>(args[2]);auto* output=*static_cast<SyncThread**>(args[3]);unsigned phases=unsigned(length)*(selected_purpose==0?8:1);
  if(stamp->smid!=~unsigned(0))throw std::runtime_error("stamp poison missing");
  for(unsigned tid=0;tid<threads.x;++tid){auto& row=output[tid];unsigned value=seed+17*tid;bool selected=selected_mode==0?tid%32>=16:tid/32==threads.x/32-1;if(selected_skew&&selected)for(unsigned n=0;n<phases*256;++n)value=value*1664525u+1013904223u;
   unsigned checked=selected_purpose==1?2:0;if(row.value!=~value||row.completed!=~gh::u64(phases)||row.checked!=~gh::u64(checked)||row.errors!=~unsigned(0)||row.timeout!=~unsigned(0)||row.smid!=~unsigned(0)||row.attempts!=~gh::u64(0)||row.arrival_cycle!=~gh::u64(0)||row.departure_cycle!=~gh::u64(0))throw std::runtime_error("perfield poison missing");
   const gh::u64 arrival=selected_purpose==2?(1ull<<53)+3+tid*2:0;row={value,0,0,137,phases,selected_mode==2?gh::u64(phases)*(selected_purpose==1?2:1):0,checked,arrival,arrival+(selected_purpose==2?1:0)};
  }
  *stamp={1000,1000,(1ull<<53)+1,(1ull<<53)+2048,137};
  if(failure=="last")output[threads.x-1].value^=1;if(failure=="timeout")output[0].timeout=1;if(failure=="attempts")output[0].attempts=~gh::u64(0);if(failure=="smid")output[0].smid=~unsigned(0);if(failure=="arrival")output[0].arrival_cycle=~gh::u64(0);if(failure=="clock")stamp->end_cycle=~gh::u64(0);
 }
 if(const char* p=std::getenv("S11_LAUNCH_LOG")){std::ofstream f(p,std::ios::app);f<<length<<"\n";}return 0;
}
'''
class SyncShortHost(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.binary=Path(cls.temp.name)/'host';src=Path(cls.temp.name)/'host.cpp'
  replacements={'REFERENCE_PATH':'synchronization_reference.hpp','PAIR_REFERENCE_PATH':'synchronization_pair_reference_v2.hpp','BINDINGS_PATH':'synchronization_role_bindings_r4.hpp','WRITER_PATH':'word_artifacts.hpp'};shim=SHIM
  for key in sorted(replacements,key=len,reverse=True):shim=shim.replace(key,str(ROOT/'common'/replacements[key]))
  source=(ROOT/'probes/synchronization_short_r4.cu').read_text();source=source[source.index('struct SyncShortArtifact'):];src.write_text(shim+source)
  run=subprocess.run(['g++','-std=c++17','-O2',str(src),'-o',str(cls.binary)],capture_output=True,text=True,timeout=30)
  if run.returncode:raise RuntimeError(run.stderr)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def host(self,case,profile,failure=''):
  with tempfile.TemporaryDirectory(dir=self.temp.name) as d:
   log=Path(d)/'launches';run=subprocess.run([str(self.binary),'validate-only',case['id'],profile['id'],'3'],cwd=d,capture_output=True,text=True,timeout=30,env={**os.environ,'S11_LAUNCH_LOG':str(log),'S11_STUB_FAILURE':failure});rows=[json.loads(s) for s in run.stdout.splitlines()];launches=log.read_text().splitlines() if log.exists() else []
   if run.returncode==0:
    device,row=rows;core.validation_envelope(row,case,profile,3,POLICY);adapter.validate_validation(device,row,case,profile,3);core.verify_output_artifacts(Path(d),row);arrays={}
    for check in row['checks']:
     for a in check['output_artifacts']:
      data=(Path(d)/a['path']).read_bytes();arrays[a['path']]={'shape':a['shape'],'values':list(struct.unpack('<'+'I'*(len(data)//4),data))}
    adapter.audit_values(device,row,case,profile,3,arrays);self.last_arrays=arrays
   return run,rows,launches
 def test_all13_measured_and_auxiliary_profiles_exact65launches(self):
  total=0
  for case in LEGACY:
   for profile in PROFILES[:2]:
    run,rows,launches=self.host(case,profile);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(launches,[str(n) for n in profile['target_iterations']]);total+=len(launches)
  self.assertEqual(total,65)
 def test_two_supplemental_cases_exact6launches_and_wide_clocks(self):
  for case in PAIR_CASES:
   run,_,launches=self.host(case,PROFILES[2]);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(launches,['1','2','33'])
 def test_every_complete_value_field_corruption_rejects(self):
  case=next(c for c in LEGACY if c['id']=='mbarrier_t128_aligned');profile=PROFILES[1];run,rows,_=self.host(case,profile);self.assertEqual(run.returncode,0,run.stderr)
  for name in self.last_arrays:
   arrays=copy.deepcopy(self.last_arrays)
   if 'threads' in name:arrays[name]['values'][0]^=1
   else:arrays[name]['values'][-1]=0xffffffff
   with self.subTest(array=name),self.assertRaises(ValueError):adapter.audit_values(rows[0],rows[1],case,profile,3,arrays)
  for failure in ('last','timeout','attempts','smid','arrival','clock'):
   run,_,_=self.host(case,profile,failure);self.assertNotEqual(run.returncode,0)
  for case in PAIR_CASES:
   for failure in ('last','clock'):
    run,_,_=self.host(case,PROFILES[2],failure);self.assertNotEqual(run.returncode,0)

 def test_missing_role_wrong_resource_role_length_and_uint_domains_rejected(self):
  case=LEGACY[0];profile=PROFILES[1];run,rows,_=self.host(case,profile);self.assertEqual(run.returncode,0,run.stderr)
  for kind in ('missing_role','role_reg','role_identity','primary','length','phase_quantity','checked_bool','float_word','sentinel'):
   row=copy.deepcopy(rows[1]);arrays=copy.deepcopy(self.last_arrays)
   if kind=='missing_role':row['resource_identity']['extensions']['role_resources'].pop('arrival')
   elif kind=='role_reg':row['resource_identity']['extensions']['role_resources']['arrival']['registers_per_thread']+=1
   elif kind=='role_identity':row['resource_identity']['extensions']['role_target_identities']['arrival']['target_identity_sha256']='0'*64
   elif kind=='primary':row['resource_identity']['kernel_symbol']=row['resource_identity']['extensions']['role_resources']['arrival']['kernel_symbol']
   elif kind=='length':row['target_launches'][0]['iterations']=1
   elif kind=='phase_quantity':row['checks'][0]['checked_elements']-=1
   elif kind=='checked_bool':row['checks'][0]['errors']=False
   elif kind=='float_word':arrays['sync_stamp_0.u32le']['values'][0]=1000.5
   else:arrays['sync_threads_1.u32le']['values'][12:14]=[4294967295,4294967295]
   with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.audit_values(rows[0],row,case,profile,3,arrays)
  for case in PAIR_CASES:
   run,rows,_=self.host(case,PROFILES[2]);self.assertEqual(run.returncode,0,run.stderr)
   for index in (0,1,2,3):
    arrays=copy.deepcopy(self.last_arrays);arrays['sync_pair_threads_2.u32le']['values'][index]^=1
    with self.subTest(pair=case['id'],field=index),self.assertRaises(ValueError):adapter.audit_values(rows[0],rows[1],case,PROFILES[2],3,arrays)
 def test_fixed_core_profiles_and_seed_and_old_entry_forwarding(self):
  for profile in PROFILES:core.profile_from({'schema_version':1,'profiles':[profile]},profile['id'],POLICY)
  for seed in (0,4294967295):
   with self.assertRaises(ValueError):adapter.validation_argv('binary/probe',LEGACY[0],PROFILES[0],seed)
  run=subprocess.run([str(self.binary),'warp_t32_aligned','2048','3'],capture_output=True,text=True,timeout=30);self.assertEqual(run.returncode,9)

if __name__=='__main__':unittest.main()
