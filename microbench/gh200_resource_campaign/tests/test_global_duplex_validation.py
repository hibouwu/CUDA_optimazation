"""Actual S13 host branch CPU simulation; no GPU or compiled-device qualification."""
import copy,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors import global_duplex_validation as adapter
CONTRACT=json.loads((ROOT/'contracts/global_duplex.json').read_text())
PROFILE=json.loads((ROOT/'contracts/global_duplex_validation_profiles_v1.json').read_text())['profiles'][0]

SHIM=r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "REFERENCE_PATH"
struct uint4 {unsigned x,y,z,w;};
namespace gh {
using u64=unsigned long long;
struct Stamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
struct Observation {std::vector<Stamp> stamps;double event_ms=0;u64 errors=0,checked_elements=0;std::string method,input_conditions;};
struct Prop {int multiProcessorCount=2;int l2CacheSize=131072;};
struct Device {Prop prop;};
Device device(){return {};}
void emit_device(const Device&){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"CPU shim GH200\",\"uuid\":\"GPU-00000000-0000-0000-0000-000000000000\",\"sms\":2,\"driver_version\":12090,\"runtime_version\":12090,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":227328,\"l2_cache_bytes\":131072,\"global_memory_bytes\":1073741824}\n";}
std::string quote(const std::string& s){return "\""+s+"\"";}
u64 integer(const char* s,u64 lo,u64 hi){char* end;auto x=std::strtoull(s,&end,10);if(*end||x<lo||x>hi)throw std::runtime_error("integer");return x;}
template<class F> int warmup(F){throw std::runtime_error("FORBIDDEN warmup");}
template<class... T> void emit_trial(T...){throw std::runtime_error("FORBIDDEN formal trial");}
}
constexpr unsigned gd_threads=256;
int selected_reads=0,selected_writes=0;bool selected_copy=false;
#define MOCK_KERNEL(NAME,R,W,COPY) void NAME(){selected_reads=R;selected_writes=W;selected_copy=COPY;}
MOCK_KERNEL(gd_read_ca_r1_w0,1,0,false)
MOCK_KERNEL(gd_read_cg_r1_w0,1,0,false)
MOCK_KERNEL(gd_write_wb_r0_w1,0,1,false)
MOCK_KERNEL(gd_copy_cg_r1_w1,1,1,true)
MOCK_KERNEL(gd_independent_cg_r1_w1,1,1,false)
MOCK_KERNEL(gd_independent_cg_r2_w1,2,1,false)
MOCK_KERNEL(gd_independent_cg_r4_w1,4,1,false)
MOCK_KERNEL(gd_independent_cg_r1_w2,1,2,false)
MOCK_KERNEL(gd_independent_cg_r1_w4,1,4,false)
struct cudaFuncAttributes {int numRegs=32;size_t sharedSizeBytes=1024,localSizeBytes=0;};
struct dim3 {unsigned x;dim3(unsigned n):x(n){}};
using cudaEvent_t=int;
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA shim failure: " #call);}while(0)
int cudaFuncGetAttributes(cudaFuncAttributes* a,const void* f){*a={};reinterpret_cast<void(*)()>(const_cast<void*>(f))();return 0;}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* o,const void*,unsigned,size_t){*o=4;return 0;}
int cudaMemGetInfo(size_t* f,size_t* t){*f=*t=1<<30;return 0;}
template<class T> int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::malloc(n));return *p?0:1;}
int cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}
int cudaMemset(void* a,int v,size_t n){std::memset(a,v,n);return 0;}
int cudaFree(void* p){std::free(p);return 0;}
int cudaEventCreate(int* p){*p=0;return 0;}int cudaEventRecord(int){return 0;}
int cudaEventSynchronize(int){return 0;}int cudaEventDestroy(int){return 0;}
int cudaEventElapsedTime(float* p,int,int){*p=0;return 0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3,void** args,size_t){
 static int launch=0;++launch;
 std::string fail=std::getenv("S13_STUB_FAILURE")?std::getenv("S13_STUB_FAILURE"):"";
 if(fail=="cuda"||(fail=="cuda_second"&&launch==2))return 1;
 const auto input=*static_cast<uint4**>(args[0]);auto output=*static_cast<uint4**>(args[1]);
 const auto vectors=*static_cast<gh::u64*>(args[2]);int iterations=*static_cast<int*>(args[3]);
 unsigned seed=*static_cast<unsigned*>(args[4]);auto stamps=*static_cast<gh::Stamp**>(args[5]);auto sums=*static_cast<unsigned**>(args[6]);
 const size_t threads=size_t(grid.x)*256,groups=vectors/threads;
 // Enumerate owner vectors directly: a full sweep reads each owned vector r
 // times. This model does not reproduce the device's wrapped group issue order.
 for(size_t t=0;t<threads;++t){
  unsigned sum=0;
  for(int iteration=0;iteration<iterations;++iteration)for(size_t group=0;group<groups;++group){
   const size_t index=group*threads+t;
   for(int read=0;read<selected_reads;++read){
    auto v=input[index];sum+=v.x+v.y+v.z+(fail=="omit_lane"?0:v.w);
   }
   for(int write=0;write<selected_writes;++write){
    if(fail=="missing_destination"&&index==0)continue;
    if(selected_copy)output[index]=input[index];
    else {const unsigned word=unsigned(index*4);output[index]={29u*word+seed,29u*(word+1)+seed,29u*(word+2)+seed,29u*(word+3)+seed};}
   }
  }
  if(!(fail=="missing_checksum"&&t==0))sums[t]=sum;
 }
 for(unsigned b=0;b<grid.x;++b)stamps[b]={0,0,0,0,b+100};
 if(fail=="stamp")stamps[grid.x-1].end_ns=~gh::u64(0);
 if(fail=="copy_dependency"&&selected_copy)output[0].x^=1;
 if(fail=="many")for(unsigned i=0;i<256;++i)sums[i]^=1;
 return 0;
}
'''


def unique_object(pairs):
    obj={}
    for k,v in pairs:
        if k in obj:raise ValueError('duplicate JSON key '+k)
        obj[k]=v
    return obj


class GlobalDuplexValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();p=Path(cls.temp.name)/'host.cpp';cls.binary=Path(cls.temp.name)/'host'
        source=(ROOT/'probes/global_duplex.cu').read_text();host=source[source.index('struct GdCase'):]
        p.write_text(SHIM.replace('REFERENCE_PATH',str(ROOT/'common/global_duplex_reference.hpp'))+host)
        subprocess.run(['g++','-std=c++17','-O2',str(p),'-o',str(cls.binary)],check=True,capture_output=True,timeout=30)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_host(self,case,failure='',seed=0xa5a5a5a5):
        run=subprocess.run([str(self.binary),'validate-only',case['id'],PROFILE['id'],str(seed)],text=True,capture_output=True,timeout=20,env={**os.environ,'S13_STUB_FAILURE':failure})
        rows=[json.loads(s,object_pairs_hook=unique_object) for s in run.stdout.splitlines()]
        self.assertEqual(len(rows),2,run.stderr);return run,rows

    def test_all18_real_host_rows_against_ABI(self):
        for case in CONTRACT['cases']:
            with self.subTest(case=case['id']):
                run,(device,row)=self.run_host(case);self.assertEqual(run.returncode,0,run.stderr)
                result=adapter.validate_validation(device,row,case,PROFILE,0xa5a5a5a5)
                self.assertEqual(result['target_launches'],2)
                r=case['parameters']['read_requests_per_group'];w=case['parameters']['write_requests_per_group']
                ext=row['resource_identity']['extensions']
                self.assertEqual(result['checked_elements'],2*((row['blocks']*256 if r else 0)+(ext['array_bytes']//4 if w else 0)))
                self.assertEqual(adapter.validation_argv('binary/probe',case,PROFILE,0),['binary/probe','validate-only',case['id'],PROFILE['id'],'0'])

    def test_old_poison_collision_and_all_completion_failure_paths(self):
        # At seed0xa5a5a5a5 destination word0 equals the rejected constant
        # poison. New per-item inversion must still detect an omitted store.
        self.assertEqual((29*0+0xa5a5a5a5)&0xffffffff,0xa5a5a5a5)
        for mode in ('write','copy','independent'):
            case=next(c for c in CONTRACT['cases'] if c['parameters']['mode']==mode)
            run,(device,row)=self.run_host(case,'missing_destination')
            self.assertNotEqual(run.returncode,0)
            with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,PROFILE,0xa5a5a5a5)
        for mode in ('read','write','copy','independent'):
            case=next(c for c in CONTRACT['cases'] if c['parameters']['mode']==mode)
            for failure in ('missing_checksum','stamp','cuda','cuda_second','many'):
                with self.subTest(mode=mode,failure=failure):
                    run,(device,row)=self.run_host(case,failure)
                    self.assertNotEqual(run.returncode,0)
                    self.assertEqual(len(row['target_launches']),2 if failure=='cuda_second' else 1)
                    self.assertLessEqual(sum(l.startswith('length=') for l in run.stderr.splitlines()),8)
                    with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,PROFILE,0xa5a5a5a5)
        for failure,mode in [('omit_lane','read'),('copy_dependency','copy')]:
            case=next(c for c in CONTRACT['cases'] if c['parameters']['mode']==mode)
            run,_=self.run_host(case,failure);self.assertNotEqual(run.returncode,0)

    def test_schema_coverage_allocation_and_flags_rejected(self):
        case=CONTRACT['cases'][-1];_,(device,original)=self.run_host(case)
        mutations=[lambda r:r.update(errors=1),lambda r:r.update(errors=False),lambda r:r.update(pilot_executed=True),
                   lambda r:r.update(warmup_executed=True),lambda r:r.update(performance_eligible=True),
                   lambda r:r.update(blocks=1),lambda r:r['checks'].pop(),lambda r:r['target_launches'].pop(),
                   lambda r:r['target_launches'][1].update(iterations=1),lambda r:r['target_launches'][0].update(input_profile='uniform'),
                   lambda r:r['checks'][1].update(launch_index=0),lambda r:r['checks'][0].update(completed=False),
                   lambda r:r['checks'][0]['verified_CTA_ids'].pop(),lambda r:r['checks'][0].update(reference_sha256='0'*64),
                   lambda r:r['resource_identity'].update(static_smem_bytes=0),lambda r:r['resource_identity'].update(local_size_bytes=4),
                   lambda r:r['resource_identity'].update(occupancy_limit_ctas_per_sm=32),
                   lambda r:r['resource_identity']['extensions'].update(array_bytes=16),
                   lambda r:r['resource_identity']['extensions'].update(aggregate_array_bytes=16),
                   lambda r:r['resource_identity']['extensions'].update(read_requests_per_group=0)]
        for mutate in mutations:
            row=copy.deepcopy(original);mutate(row)
            with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,PROFILE,0xa5a5a5a5)
        for field in original:
            row=copy.deepcopy(original);del row[field]
            with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,PROFILE,0xa5a5a5a5)
        write=next(c for c in CONTRACT['cases'] if c['parameters']['mode']=='write');_,(device,row)=self.run_host(write)
        for check in row['checks']:
            check['checked_elements']+=row['blocks']*256;check['expected_elements']+=row['blocks']*256
        with self.assertRaises(ValueError):adapter.validate_validation(device,row,write,PROFILE,0xa5a5a5a5)

    def test_no_profile_or_prior_evidence_bypass(self):
        case=CONTRACT['cases'][0]
        for path in ('../probe','/tmp/probe','a/../probe','a//probe'):
            with self.assertRaises(ValueError):adapter.validation_argv(path,case,PROFILE,0)
        p=copy.deepcopy(PROFILE);p['target_iterations']=[1,16]
        with self.assertRaises(ValueError):adapter.validation_argv('binary/probe',case,p,0)
        request={'case':case,'profile':PROFILE}
        self.assertEqual(adapter.validate_prior_evidence({},request,{})['status'],'insufficient')
        with self.assertRaises(ValueError):adapter.validate_prior_evidence({'source_artifacts_sha256':{'raw':'a'*64}},request,{'raw':'b'*64})


if __name__=='__main__':unittest.main()
