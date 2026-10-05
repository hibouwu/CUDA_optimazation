"""Actual generated host runs on CPU CUDA shim; no GPU evidence."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from auditors import low_precision_bounded_fp8_validation as adapter
CONTRACT=json.loads((B/'contracts/low_precision_bounded_fp8_v1.json').read_text());PROFILE=json.loads((B/'contracts/low_precision_bounded_fp8_validation_profiles_v1.json').read_text())['profiles'][0]
SHIM=r'''
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "low_precision_bounded_fp8_reference_v1.hpp"
namespace gh {
using u64=unsigned long long;struct Stamp{u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
struct Device {struct Prop{int multiProcessorCount=2;}prop;};Device device(){return {};}
void emit_device(Device){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"GH200 CPU shim\",\"sms\":2,\"uuid\":\"GPU-00000000-0000-0000-0000-000000000001\",\"runtime_version\":12090,\"driver_version\":13010,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":232448}\n";}
std::string quote(std::string s){return "\""+s+"\"";}
u64 integer(const char* s,u64 lo,u64 hi){char* end;auto v=std::strtoull(s,&end,10);if(*end||v<lo||v>hi)throw std::runtime_error("integer");return v;}
struct Observation{std::vector<Stamp>stamps;double event_ms;u64 errors=0,checked_elements=0;std::string method,input_conditions;};
void envelope(Observation&){};template<class F> int warmup(F f){f();return 1;}
void emit_trial(const std::string&,int,unsigned,int,const char*,const char*,u64,int,int,const Observation&,int,const std::string& s){std::cout<<"{"<<s<<"}\n";}
}
int kind=0,groups=1,launches=0;
#define FN(K,G,N) void lp_bounded_v1_wgmma_##N##_g##G(int,unsigned,bool,gh::Stamp*,double*){kind=K;groups=G;}
FN(0,1,e4m3) FN(0,2,e4m3) FN(1,1,e5m2) FN(1,2,e5m2)
struct LowCase {const char* id;void(*kernel)(int,unsigned,bool,gh::Stamp*,double*);int kind,groups,width;bool all_gpu;};
struct cudaFuncAttributes {int numRegs=148;unsigned sharedSizeBytes=0,localSizeBytes=0;};
struct dim3{unsigned x;dim3(unsigned v):x(v){}};using cudaEvent_t=int;
constexpr int cudaMemcpyDeviceToHost=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA failure");}while(0)
int cudaFuncGetAttributes(cudaFuncAttributes* p,const void* f){reinterpret_cast<void(*)(int,unsigned,bool,gh::Stamp*,double*)>(const_cast<void*>(f))(0,0,0,nullptr,nullptr);p->sharedSizeBytes=4096+groups*128*8+16;return 0;}
template<class F>int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* p,F,int,int){*p=groups==1?2:1;return 0;}
template<class T>int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::malloc(n));return 0;}
int cudaMemset(void* p,int v,size_t n){std::memset(p,v,n);return 0;}
int cudaMemcpy(void* p,const void* q,size_t n,int){std::memcpy(p,q,n);return 0;}
int cudaFree(void* p){std::free(p);return 0;}
int cudaEventCreate(int* p){*p=0;return 0;}int cudaEventRecord(int){return 0;}int cudaEventSynchronize(int){return 0;}int cudaEventDestroy(int){return 0;}int cudaGetLastError(){return 0;}
int cudaEventElapsedTime(float* p,int,int){*p=0;return 0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3 block,void** args,int,void*){
 ++launches;std::cerr<<"TARGET "<<launches<<"\n";
 std::string fail=std::getenv("BFP8_FAIL")?std::getenv("BFP8_FAIL"):"";
 if(fail=="cuda")return 1;
 int length=*static_cast<int*>(args[0]);unsigned seed=*static_cast<unsigned*>(args[1]);bool non=*static_cast<bool*>(args[2]);
 auto stamps=*static_cast<gh::Stamp**>(args[3]);auto output=*static_cast<double**>(args[4]);
 for(unsigned b=0;b<grid.x;++b){
  stamps[b]={0,0,0,0,137+b};
  for(unsigned t=0;t<block.x;++t)for(unsigned c=0;c<2;++c)for(unsigned f=0;f<32;++f){
   unsigned g=t/128,local=t%128,lane=t%32,row=16*(local/32)+lane/4+8*((f/2)%2),col=2*(lane%4)+f%2+8*(f/4);
   long long sum=0;if(non)for(int k=0;k<32;++k)sum+=(int((row+2ull*k+seed)%7)-3)*(int((col+3ull*k+seed)%7)-3);
   double value=non?double(sum*length*16)/64+double(1+g+c)/8:double(1+2*g+c)/8;
   output[((b*block.x+t)*2+c)*32+f]=value;
  }
 }
 auto last=grid.x*block.x*64-1;
 if(fail=="last")output[last]+=1;
 if(fail=="nan")output[last]=std::numeric_limits<double>::quiet_NaN();
 if(fail=="nonf32")output[last]+=1e-12;
 if(fail=="stamp")stamps[grid.x-1].smid=0xffffffffu;
 if(fail=="zero")output[last]=0;
 return 0;
}
'''
class Host(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();p=Path(cls.tmp.name);cls.exe=p/'host'
  source=(B/'probes/low_precision_bounded_fp8_v1.cu').read_text();host=source[source.index('const std::vector<LowCase> cases='):]
  (p/'host.cpp').write_text(SHIM+host)
  subprocess.run(['c++','-std=c++17','-O2','-I',str(B/'common'),str(p/'host.cpp'),'-o',str(cls.exe)],check=True,capture_output=True)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def run_host(self,case,fail='',seed=3):
  return subprocess.run([str(self.exe),'validate-only',case['id'],PROFILE['id'],str(seed)],env={**os.environ,'BFP8_FAIL':fail},text=True,capture_output=True)
 def test_all_eight_actual_host_four_targets(self):
  for c in CONTRACT['cases']:
   result=self.run_host(c);self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stderr.count('TARGET '),4)
   device,row=map(json.loads,result.stdout.splitlines());self.assertEqual(adapter.validate_validation(device,row,c,PROFILE,3)['target_launches'],4)
 def test_partial_failure_and_wrong_seed(self):
  c=CONTRACT['cases'][-1]
  for failure in ('last','nan','nonf32','stamp','zero','cuda'):
   result=self.run_host(c,failure);self.assertNotEqual(result.returncode,0);self.assertEqual(result.stderr.count('TARGET '),1)
   device,row=map(json.loads,result.stdout.splitlines());self.assertGreater(row['errors'],0);self.assertEqual(len(row['checks']),1)
   with self.assertRaises(ValueError):adapter.validate_validation(device,row,c,PROFILE,3)
  result=self.run_host(c,seed=4);self.assertNotEqual(result.returncode,0);self.assertNotIn('TARGET ',result.stderr)

if __name__=='__main__':unittest.main()
