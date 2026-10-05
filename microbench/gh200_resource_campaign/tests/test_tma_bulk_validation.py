"""Actual S14 host branch and full persisted artifacts on a CPU CUDA shim.

No device kernel is executed. Real GPU correctness remains a separate gate.
"""
import copy,hashlib,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors import tma_bulk_validation as adapter
from auditors.tma_bulk import audit_artifact_values,artifact_layouts
CONTRACT=json.loads((ROOT/'contracts/tma_bulk.json').read_text())
PROFILES=json.loads((ROOT/'contracts/tma_bulk_validation_profiles_v1.json').read_text())['profiles']
SHIM=r'''
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "REFERENCE_PATH"
#include "ARTIFACT_PATH"
namespace gh {
using u64=unsigned long long;
struct Stamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
struct Prop {int multiProcessorCount=2;size_t sharedMemPerBlockOptin=232448;};
struct Device {Prop prop;};Device device(){return {};}
void emit_device(const Device&){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"CPU shim GH200\",\"uuid\":\"GPU-00000000-0000-0000-0000-000000000000\",\"sms\":2,\"driver_version\":12090,\"runtime_version\":12090,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":232448,\"l2_cache_bytes\":62914560,\"global_memory_bytes\":1073741824}\n";}
std::string quote(const std::string& s){return "\""+s+"\"";}
u64 integer(const char* s,u64 lo,u64 hi){char* end;auto v=std::strtoull(s,&end,10);if(*end||v<lo||v>hi)throw std::runtime_error("integer");return v;}
}
constexpr unsigned tb_threads=128,tb_slots=32;
unsigned selected_q=0;bool selected_g2s=false,selected_release=false;
#define TB_FAKE(Q) \
void tb_g2s_q##Q(){selected_q=Q;selected_g2s=true;selected_release=false;} \
void tb_s2g_q##Q(){selected_q=Q;selected_g2s=false;selected_release=false;} \
void tb_release_q##Q(){selected_q=Q;selected_g2s=false;selected_release=true;}
TB_FAKE(1024) TB_FAKE(4096) TB_FAKE(8192) TB_FAKE(16384) TB_FAKE(32768) TB_FAKE(65536)
struct cudaFuncAttributes {int numRegs=32;size_t sharedSizeBytes=0,localSizeBytes=0;};
struct dim3 {unsigned x;dim3(unsigned n):x(n){}};
using cudaEvent_t=int;
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1,cudaFuncAttributeMaxDynamicSharedMemorySize=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA shim failure: " #call);}while(0)
int cudaFuncSetAttribute(const void* f,int,size_t){reinterpret_cast<void(*)()>(const_cast<void*>(f))();return 0;}
int cudaFuncGetAttributes(cudaFuncAttributes* a,const void*){*a={};return 0;}
int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* o,const void*,unsigned,size_t shared){*o=std::min(4,int(233472/shared));return 0;}
int cudaMemGetInfo(size_t* f,size_t* t){*f=*t=1<<30;return 0;}
template<class T> int cudaMalloc(T** p,size_t n){*p=static_cast<T*>(std::malloc(n));return *p?0:1;}
int cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}
int cudaMemset(void* a,int v,size_t n){std::memset(a,v,n);return 0;}
int cudaFree(void* p){std::free(p);return 0;}
int cudaEventCreate(int* p){*p=0;return 0;}int cudaEventRecord(int){return 0;}
int cudaEventSynchronize(int){return 0;}int cudaEventDestroy(int){return 0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3,void** args,size_t shared){
 if(const char* log=std::getenv("S14_LAUNCH_LOG")){std::ofstream out(log,std::ios::app);out<<"one target\n";}
 const std::string failure=std::getenv("S14_STUB_FAILURE")?std::getenv("S14_STUB_FAILURE"):"";
 if(failure=="cuda")return 1;
 if(shared!=selected_q+32)return 2;
 auto* global=*static_cast<unsigned**>(args[0]);
 unsigned seed=*static_cast<unsigned*>(args[selected_release?1:2]);
 int iterations=selected_release?1:*static_cast<int*>(args[1]);
 auto* capture=*static_cast<unsigned**>(args[selected_release?2:4]);
 auto* stamps=*static_cast<gh::Stamp**>(args[selected_release?3:5]);
 auto* completion=*static_cast<std::uint64_t**>(args[selected_release?4:6]);
 auto* clocks=selected_release?*static_cast<std::uint64_t**>(args[5]):nullptr;
 const unsigned words=selected_q/4;
 for(unsigned b=0;b<grid.x;++b){
  for(int iteration=0;iteration<iterations;++iteration){
   const size_t slot=size_t(b)*32+iteration%32;
   for(unsigned word=0;word<words;++word){
    const unsigned value=selected_g2s?global[slot*words+word]:29u*unsigned(size_t(b)*words+word)+seed;
    if(!selected_g2s)global[slot*words+word]=value;
    size_t index=selected_release?size_t(b)*words+word:(size_t(b)*iterations+iteration)*words+word;
    if(!(failure=="missing_capture_last"&&b==grid.x-1&&iteration==iterations-1&&word==words-1))
      capture[index]=selected_release?~value:value;
   }
  }
  stamps[b]={0,0,0,0,b+100};const size_t at=size_t(b)*5;
  completion[at]=iterations;completion[at+1]=selected_g2s?iterations*2:0;completion[at+2]=0;
  completion[at+3]=selected_release?1:0;completion[at+4]=selected_g2s?0:iterations;
  if(clocks){clocks[b*3]=(1ull<<53)+3;clocks[b*3+1]=(1ull<<53)+3;clocks[b*3+2]=(1ull<<53)+3;}
 }
 if(failure=="guard")global[-1]^=1;
 if(failure=="untouched_slot"&&!selected_g2s&&iterations<32)global[(32-1)*words]^=1;
 if(failure=="last_ring_word"&&!selected_g2s)global[size_t(grid.x)*32*words-1]^=1;
 if(failure=="stamp")stamps[grid.x-1].end_ns=~gh::u64(0);
 if(failure=="timeout")completion[2]=1;
 if(failure=="missing_full_wait")completion[4]=0;
 if(failure=="unwritten_count")completion[0]=~std::uint64_t(0);
 if(failure=="release_clock"&&clocks)clocks[2]=clocks[0]-1;
 if(failure=="source_not_overwritten"&&selected_release)capture[0]^=0xffffffff;
 return 0;
}
'''


def pairs(items):
    obj={}
    for k,v in items:
        if k in obj:raise ValueError('duplicate JSON key '+k)
        obj[k]=v
    return obj


class TmaBulkValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();src=Path(cls.temp.name)/'host.cpp';cls.binary=Path(cls.temp.name)/'host';cls.sequence=0
        source=(ROOT/'probes/tma_bulk.cu').read_text();host=source[source.index('struct TbCase'):]
        src.write_text(SHIM.replace('REFERENCE_PATH',str(ROOT/'common/tma_bulk_reference.hpp')).replace('ARTIFACT_PATH',str(ROOT/'common/word_artifacts.hpp'))+host)
        subprocess.run(['g++','-std=c++17','-O3',str(src),'-o',str(cls.binary)],check=True,capture_output=True,timeout=30)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_host(self,case,profile,failure='',seed=None):
        type(self).sequence+=1;folder=Path(self.temp.name)/str(self.sequence);folder.mkdir();log=folder/'launches'
        run=subprocess.run([str(self.binary),'validate-only',case['id'],profile['id'],str(profile['required_seed'] if seed is None else seed)],cwd=folder,text=True,capture_output=True,timeout=30,
            env={**os.environ,'S14_STUB_FAILURE':failure,'S14_LAUNCH_LOG':str(log)})
        rows=[json.loads(s,object_pairs_hook=pairs) for s in run.stdout.splitlines()]
        return run,rows,folder

    def load_arrays(self,row,folder):
        arrays={};buffers=[]
        for item in row['checks'][0]['output_artifacts']:
            content=(folder/item['path']).read_bytes();buffers.append(content)
            self.assertEqual(hashlib.sha256(content).hexdigest(),item['sha256'])
            count=1
            for d in item['shape']:count*=d
            self.assertEqual(len(content),count*4)
            self.assertEqual(sys.byteorder,'little')
            arrays[item['path']]={'shape':item['shape'],'values':memoryview(content).cast('I')}
        return arrays

    def test_all84_profiles_actual_host_and_full_word_replay(self):
        total=0
        for case in CONTRACT['cases']:
            for profile in PROFILES:
                if profile['role']=='source_release' and case['parameters']['direction']=='gmem_to_smem':continue
                with self.subTest(case=case['id'],profile=profile['id']):
                    run,rows,folder=self.run_host(case,profile);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(len(rows),2)
                    device,row=rows;result=adapter.validate_validation(device,row,case,profile,profile['required_seed'])
                    arrays=self.load_arrays(row,folder);replay=audit_artifact_values(case,profile,profile['required_seed'],arrays)
                    self.assertEqual(result['checked_elements'],replay['checked_elements']);self.assertTrue(replay['full_value_replay'])
                    self.assertEqual((folder/'launches').read_text(),'one target\n');total+=1
        self.assertEqual(total,84)

    def test_single_word_guard_timeout_and_lifecycle_failures(self):
        g2s=next(c for c in CONTRACT['cases'] if c['id']=='gmem_to_smem_64kib_one_cta')
        s2g=next(c for c in CONTRACT['cases'] if c['id']=='smem_to_gmem_64kib_one_cta')
        coordinates=[(g2s,PROFILES[2],mode) for mode in ('missing_capture_last','guard','stamp','timeout','unwritten_count','cuda')]
        coordinates += [(s2g,PROFILES[0],mode) for mode in ('untouched_slot','last_ring_word','missing_full_wait')]
        coordinates += [(s2g,PROFILES[3],mode) for mode in ('release_clock','source_not_overwritten','missing_full_wait')]
        for case,profile,mode in coordinates:
            with self.subTest(mode=mode):
                run,rows,folder=self.run_host(case,profile,mode);self.assertNotEqual(run.returncode,0);self.assertEqual(len(rows),2)
                self.assertEqual((folder/'launches').read_text(),'one target\n')
                self.assertLessEqual(sum(line.startswith('S14 ') and ('index=' in line or 'CTA=' in line) for line in run.stderr.splitlines()),8)
                with self.assertRaises(ValueError):adapter.validate_validation(rows[0],rows[1],case,profile,profile['required_seed'])

    def test_mismatched_profile_seed_never_launches(self):
        case=CONTRACT['cases'][0]
        for profile,badseed in [(PROFILES[0],3),(PROFILES[1],0),(PROFILES[2],3),(PROFILES[3],3)]:
            run,rows,folder=self.run_host(case,profile,seed=badseed)
            self.assertNotEqual(run.returncode,0);self.assertFalse((folder/'launches').exists())
            with self.assertRaises(ValueError):adapter.validation_argv('binary/probe',case,profile,badseed)
        folder=Path(self.temp.name)/'formal-forbidden';folder.mkdir()
        run=subprocess.run([str(self.binary),case['id'],'32','3'],cwd=folder,text=True,capture_output=True,env={**os.environ,'S14_LAUNCH_LOG':str(folder/'launches')})
        self.assertNotEqual(run.returncode,0);self.assertFalse((folder/'launches').exists())

    def test_metadata_only_pass_does_not_replace_full_value_replay(self):
        case=next(c for c in CONTRACT['cases'] if c['id']=='smem_to_gmem_1kib_one_cta');profile=PROFILES[0]
        run,(device,row),folder=self.run_host(case,profile);self.assertEqual(run.returncode,0)
        arrays=self.load_arrays(row,folder);array=arrays['bulk_ring.u32le'];corrupted=list(array['values']);corrupted[-1]^=1
        arrays['bulk_ring.u32le']={'shape':array['shape'],'values':corrupted}
        self.assertEqual(adapter.validate_validation(device,row,case,profile,0)['status'],'pass')
        with self.assertRaises(ValueError):audit_artifact_values(case,profile,0,arrays)
        for mutate in [lambda r:r['checks'][0]['output_artifacts'].pop(),lambda r:r['checks'][0].update(completed=False),
                       lambda r:r['resource_identity']['extensions'].update(global_slots_per_cta=2),
                       lambda r:r['checks'][0]['output_artifacts'][0].update(dtype='float64'),
                       lambda r:r['checks'][0].update(checked_elements=1),lambda r:r.update(pilot_executed=True)]:
            changed=copy.deepcopy(row);mutate(changed)
            with self.assertRaises(ValueError):adapter.validate_validation(device,changed,case,profile,0)


if __name__=='__main__':unittest.main()
