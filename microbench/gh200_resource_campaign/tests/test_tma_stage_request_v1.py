"""S16 actual host CPU simulation and independently replayed full artifacts; no GPU."""
import copy,hashlib,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors import tma_stage_request_validation_v1 as adapter
CONTRACT=json.loads((ROOT/'contracts/tma_stage_request_short_v1.draft.json').read_text())
PROFILE=json.loads((ROOT/'contracts/tma_stage_request_validation_profiles_v1.draft.json').read_text())['profiles'][0]
SHIM='\n#include <algorithm>\n#include <cstdint>\n#include <cstdlib>\n#include <cstring>\n#include <fstream>\n#include <iostream>\n#include <limits>\n#include <sstream>\n#include <stdexcept>\n#include <string>\n#include <vector>\n#include "REFERENCE_PATH"\n#include "ARTIFACT_PATH"\nnamespace gh {\nusing u64=unsigned long long;\nstruct Stamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};\nstruct Prop {int multiProcessorCount=2;size_t sharedMemPerBlockOptin=232448;};\nstruct Device {Prop prop;};Device device(){return {};}\nvoid emit_device(const Device&){std::cout<<"{\\"schema_version\\":2,\\"type\\":\\"device\\",\\"cc\\":\\"9.0\\",\\"name\\":\\"CPU shim GH200\\",\\"uuid\\":\\"GPU-00000000-0000-0000-0000-000000000000\\",\\"sms\\":2,\\"driver_version\\":12090,\\"runtime_version\\":12090,\\"registers_per_sm\\":65536,\\"smem_per_sm_bytes\\":233472,\\"smem_per_cta_optin_bytes\\":232448,\\"l2_cache_bytes\\":62914560,\\"global_memory_bytes\\":1073741824}\\n";}\nstd::string quote(const std::string& s){return "\\""+s+"\\"";}\nu64 integer(const char* s,u64 lo,u64 hi){char* end;auto v=std::strtoull(s,&end,10);if(*end||v<lo||v>hi)throw std::runtime_error("integer");return v;}\n}\nconstexpr unsigned ts_Q=16384,ts_W=4096,ts_T=128,ts_G=32;\nunsigned selected_s=0,selected_r=0;bool selected_g=false;\n#define TS_FAKE(S,R) void ts_g2s_s##S##_r##R(){selected_s=S;selected_r=R;selected_g=true;} void ts_s2g_s##S##_r##R(){selected_s=S;selected_r=R;selected_g=false;}\nTS_FAKE(1,1) TS_FAKE(1,2) TS_FAKE(1,4) TS_FAKE(2,1) TS_FAKE(2,2) TS_FAKE(2,4) TS_FAKE(4,1) TS_FAKE(4,2) TS_FAKE(4,4)\nstruct cudaFuncAttributes {int numRegs=32;size_t sharedSizeBytes=0,localSizeBytes=0;};\nstruct dim3 {unsigned x;dim3(unsigned n):x(n){}};\nusing cudaEvent_t=int;\nconstexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1,cudaFuncAttributeMaxDynamicSharedMemorySize=1;\n#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA shim failure: " #call);}while(0)\nint cudaFuncSetAttribute(const void* f,int,size_t){reinterpret_cast<void(*)()>(const_cast<void*>(f))();return 0;}\nint cudaFuncGetAttributes(cudaFuncAttributes* a,const void*){*a={};const int g[9]={40,40,40,32,32,32,40,40,40},s[9]={32,40,40,40,32,32,40,32,32};unsigned k=(selected_s==1?0:selected_s==2?1:2)*3+(selected_r==1?0:selected_r==2?1:2);a->numRegs=selected_g?g[k]:s[k];return 0;}\nint cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* o,const void*,unsigned,size_t shared){*o=std::min(4,int(233472/shared));return 0;}\nint cudaMemGetInfo(size_t* f,size_t* t){*f=*t=1<<30;return 0;}\ntemplate<class T> int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::malloc(n));return *p?0:1;}\nint cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}\nint cudaMemset(void* a,int v,size_t n){std::memset(a,v,n);return 0;}\nint cudaFree(void* p){std::free(p);return 0;}\nint cudaEventCreate(int* p){*p=0;return 0;}int cudaEventRecord(int){return 0;}\nint cudaEventSynchronize(int){return 0;}int cudaEventDestroy(int){return 0;}\n\nint cudaLaunchKernel(const void*,dim3 grid,dim3,void** args,size_t shared){\n if(const char* log=std::getenv("S16_LAUNCH_LOG")){std::ofstream out(log,std::ios::app);out<<"target\\n";}\n std::string failure=std::getenv("S16_FAILURE")?std::getenv("S16_FAILURE"):"";if(failure=="cuda")return 1;\n unsigned S=selected_s,R=selected_r;if(shared!=S*R*16384+8*S+32)return 2;\n auto* global=*static_cast<unsigned**>(args[0]);unsigned I=*static_cast<unsigned*>(args[1]),seed=*static_cast<unsigned*>(args[2]);\n if(!*static_cast<bool*>(args[3]))return 3;\n auto* trace=*static_cast<unsigned**>(args[4]);auto* final=*static_cast<unsigned**>(args[5]);auto* stamps=*static_cast<gh::Stamp**>(args[6]);auto* counts=*static_cast<std::uint64_t**>(args[7]);auto* life=*static_cast<std::uint64_t**>(args[8]);\n for(unsigned b=0;b<grid.x;++b){\n  std::vector<unsigned> tile(S*R*4096);\n  for(unsigned slot=0;slot<S;++slot)for(unsigned r=0;r<R;++r)for(unsigned w=0;w<4096;++w){unsigned x=selected_g?17u*unsigned(((size_t(b)*32+slot)*R+r)*4096+w)+seed:29u*unsigned(((size_t(b)*S+slot)*R+r)*4096+w)+seed;tile[(slot*R+r)*4096+w]=selected_g?~x:x;}\n  for(unsigned i=0;i<I;++i){for(unsigned r=0;r<R;++r)for(unsigned w=0;w<4096;++w){size_t go=((size_t(b)*32+i%32)*R+r)*4096+w,to=((size_t(b)*I+i)*R+r)*4096+w,so=(i%S*R+r)*4096+w;unsigned x=selected_g?global[go]:tile[so];if(selected_g)tile[so]=x;else global[go]=x;trace[to]=x;}\n   std::uint64_t e[]={i%S,i/S,selected_g?(1ull<<60)+i:0,1,1,1,R,selected_g?0u:i+1};std::copy(e,e+8,life+(size_t(b)*I+i)*8);\n  }\n  std::copy(tile.begin(),tile.end(),final+size_t(b)*S*R*4096);\n  std::uint64_t c[]={std::uint64_t(I)*R,selected_g?0u:I,selected_g?I:0u,I,I,I,I>S?I-S:0u,selected_g?0u:1u,0,selected_g?2u*I:0u,selected_g?S:0u,selected_g?S:0u};std::copy(c,c+12,counts+b*12);\n  stamps[b]={(1ull<<53)+7,(1ull<<53)+7,(1ull<<53)+11,(1ull<<53)+11,137};\n }\n if(failure=="trace")trace[size_t(grid.x)*I*R*4096-1]^=1;\n if(failure=="final")final[size_t(grid.x)*S*R*4096-1]^=1;\n if(failure=="guard")global[-1]^=1;\n if(failure=="ring")global[size_t(grid.x)*32*R*4096-1]^=1;\n if(failure=="consumer_done")life[4]=0;\n if(failure=="release")life[5]=0;\n if(failure=="requests")counts[0]++;\n if(failure=="commits")counts[1]++;\n if(failure=="generation")life[1]++;\n if(failure=="stamp")stamps[0].end_cycle=~gh::u64(0);\n if(failure=="smid")stamps[0].smid=~unsigned(0);\n return 0;\n}\n'
class StageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name)/'host.cpp';source=(ROOT/'probes/tma_stage_request.cu').read_text();host=source[source.index('struct TsCase'):];p.write_text(SHIM.replace('REFERENCE_PATH',str(ROOT/'common/tma_stage_request_reference.hpp')).replace('ARTIFACT_PATH',str(ROOT/'common/word_artifacts.hpp'))+host);cls.bin=Path(cls.tmp.name)/'host';r=subprocess.run(['g++','-std=c++17','-O2',str(p),'-o',str(cls.bin)],capture_output=True,text=True);assert r.returncode==0,r.stderr
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def launch(self,case,failure='',seed=3):
  folder=tempfile.TemporaryDirectory(dir=self.tmp.name);p=Path(folder.name);r=subprocess.run([str(self.bin),'validate-only',case['id'],PROFILE['id'],str(seed)],cwd=p,env={**os.environ,'S16_FAILURE':failure,'S16_LAUNCH_LOG':str(p/'launches')},capture_output=True,text=True,timeout=30);rows=[json.loads(x) for x in r.stdout.splitlines()];return folder,r,rows,p
 def arrays(self,row,p):
  a={}
  for check in row['checks']:
   for x in check['output_artifacts']:
    data=(p/x['path']).read_bytes();self.assertEqual(hashlib.sha256(data).hexdigest(),x['sha256']);a[x['path']]=list(struct.unpack('<'+str(len(data)//4)+'I',data))
  return a
 def test_all36_host_coordinates_capacity_and_full_values(self):
  for case in CONTRACT['cases']:
   with self.subTest(case=case['id']):
    tmp,r,rows,p=self.launch(case)
    try:
     if case['software_stages']==4 and case['requests_per_item']==4:self.assertNotEqual(r.returncode,0);self.assertIn('resource_reject_before_launch',r.stderr);self.assertFalse((p/'launches').exists());continue
     self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len((p/'launches').read_text().splitlines()),4);adapter.audit_values(rows[0],rows[1],case,PROFILE,3,self.arrays(rows[1],p))
     from runners.validation_diagnostic import validation_envelope
     from runners.validation_diagnostic import POLICY
     policy=json.loads((ROOT.parents[1]/POLICY).read_text())
     validation_envelope(rows[1],case,PROFILE,3,policy)
    finally:tmp.cleanup()
 def test_actual_host_data_lifecycle_and_completion_negatives(self):
  for case in (CONTRACT['cases'][4],CONTRACT['cases'][6]):
   for failure in ('trace','final','guard','ring','consumer_done','release','requests','commits','generation','stamp','smid','cuda'):
    tmp,r,rows,p=self.launch(case,failure)
    try:self.assertNotEqual(r.returncode,0);self.assertEqual(len((p/'launches').read_text().splitlines()),1)
    finally:tmp.cleanup()
 def test_offline_artifact_and_row_negatives(self):
  case=CONTRACT['cases'][0];tmp,r,rows,p=self.launch(case)
  try:
   self.assertEqual(r.returncode,0,r.stderr);device,row=rows;a=self.arrays(row,p);adapter.audit_values(device,row,case,PROFILE,3,a)
   for leaf in ('trace','final_slots','ring_guards','counts','lifecycle','stamps'):
    key='stage_0_'+leaf+'.u32le';old=a[key][-1];a[key][-1]^=1
    with self.assertRaises(ValueError):adapter.audit_values(device,row,case,PROFILE,3,a)
    a[key][-1]=old
   key='stage_0_trace.u32le';old=a[key][0];a[key][0]=float(old)
   with self.assertRaises(ValueError):adapter.audit_values(device,row,case,PROFILE,3,a)
   a[key][0]=old
   for field,value in [('blocks',2),('errors',1),('warmup_executed',True)]:
    bad=copy.deepcopy(row);bad[field]=value
    with self.assertRaises(ValueError):adapter.validate_validation(device,bad,case,PROFILE,3)
   bad=copy.deepcopy(row);bad['resource_identity']['registers_per_thread']=999
   with self.assertRaises(ValueError):adapter.validate_validation(device,bad,case,PROFILE,3)
   bad=copy.deepcopy(row);bad['resource_identity']['occupancy_limit_ctas_per_sm']=99
   with self.assertRaises(ValueError):adapter.validate_validation(device,bad,case,PROFILE,3)
  finally:tmp.cleanup()
 def test_formal_and_wrong_seed_never_launch(self):
  case=CONTRACT['cases'][0];tmp,r,rows,p=self.launch(case,seed=0)
  try:self.assertNotEqual(r.returncode,0);self.assertFalse((p/'launches').exists())
  finally:tmp.cleanup()
  r=subprocess.run([str(self.bin),case['id'],'128','3'],capture_output=True,text=True);self.assertNotEqual(r.returncode,0);self.assertIn('formal admission disabled',r.stderr)
 def test_opaque_token_is_preserved_domain_only(self):
  case=CONTRACT['cases'][0];tmp,r,rows,p=self.launch(case)
  try:
   self.assertEqual(r.returncode,0,r.stderr);device,row=rows;arrays=self.arrays(row,p)
   domain=row['resource_identity']['extensions']['opaque_token_evidence']
   self.assertEqual(domain,{'classification':'preserved_domain_only','exact_checked':False,'words_per_launch':[2*row['blocks']*i for i in (1,2,5,33)]})
   for n,i in enumerate((1,2,5,33)):
    stored=sum(len(arrays[x['path']]) for x in row['checks'][n]['output_artifacts'])
    self.assertEqual(row['checks'][n]['checked_elements'],stored-2*row['blocks']*i)
   key='stage_0_lifecycle.u32le';arrays[key][4]=0xffffffff;arrays[key][5]=0xffffffff
   result=adapter.audit_values(device,row,case,PROFILE,3,arrays)
   self.assertEqual(result['opaque_token_words_preserved_domain_only'],82*row['blocks'])
   self.assertFalse(result['opaque_token_exact_checked'])
   arrays[key][4]=1.0
   with self.assertRaises(ValueError):adapter.audit_values(device,row,case,PROFILE,3,arrays)
   bad=copy.deepcopy(row);bad['checks'][0]['checked_elements']+=2*row['blocks'];bad['checks'][0]['expected_elements']=bad['checks'][0]['checked_elements']
   with self.assertRaises(ValueError):adapter.validate_validation(device,bad,case,PROFILE,3)
  finally:tmp.cleanup()
 def test_unknown_target_sass_is_rejected(self):
  with self.assertRaises(ValueError):adapter.audit_sass('synthetic SASS',CONTRACT)
  with self.assertRaises(ValueError):adapter.validation_argv('../probe',CONTRACT['cases'][0],PROFILE,3)
if __name__=='__main__':unittest.main()
