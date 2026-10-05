"""Real S10 host code under a CPU CUDA shim; never GPU evidence."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from auditors import matrix_exchange_validation as adapter
from auditors.matrix_exchange_validation_baseline import BASELINE
from runners.validation_diagnostic import verify_output_artifacts
from test_matrix_exchange import function_body
CONTRACT = json.loads((BASE/'contracts/matrix_exchange.json').read_text())
PROFILES = json.loads((BASE/'contracts/matrix_exchange_validation_profiles_v1.json').read_text())['profiles']

SHIM = r'''
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
namespace gh {
using u64=unsigned long long;
struct Stamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
struct Observation {std::vector<Stamp> stamps;double event_ms=0;u64 errors=0,checked_elements=0;std::string method,input_conditions;};
struct Device {};
Device device(){return {};}
void emit_device(const Device&){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"CPU shim GH200\",\"uuid\":\"GPU-00000000-0000-0000-0000-000000000000\",\"sms\":2,\"driver_version\":13010,\"runtime_version\":12090,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":232448,\"l2_cache_bytes\":62914560,\"global_memory_bytes\":102005473280}\n";}
std::string quote(const std::string& value){return "\""+value+"\"";}
u64 integer(const char* s,u64 lo,u64 hi){char* end;auto n=std::strtoull(s,&end,10);if(*end||n<lo||n>hi)throw std::runtime_error("integer");return n;}
template<class F> int warmup(F){throw std::runtime_error("FORBIDDEN warmup");}
template<class... T> void emit_trial(T...){throw std::runtime_error("FORBIDDEN formal emit");}
void envelope(const Observation&){throw std::runtime_error("FORBIDDEN envelope");}
}
int mode=0,matrices=0,threads=0,streams=0;bool trans=false;
template<int N,bool T,int M> void matrix_exchange(int,unsigned,gh::Stamp*,unsigned*) {mode=M;matrices=N;threads=32;streams=0;trans=T;}
template<int T,int S> void warp_exchange(int,unsigned,gh::Stamp*,unsigned*) {mode=3;matrices=0;threads=T;streams=S;trans=false;}
struct cudaFuncAttributes {int numRegs=0;size_t sharedSizeBytes=0,localSizeBytes=0;};
struct dim3 {unsigned x;dim3(unsigned value):x(value){}};
using cudaEvent_t=int;
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CUDA shim failure: " #call);}while(0)
int cudaFuncGetAttributes(cudaFuncAttributes* a,const void* f){
 reinterpret_cast<void(*)(int,unsigned,gh::Stamp*,unsigned*)>(const_cast<void*>(f))(0,0,nullptr,nullptr);
 RESOURCE_ASSIGNMENTS
 return 0;
}
template<class F> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* n,F,int,int){*n=1;return 0;}
template<class T> int cudaMalloc(T** p,size_t count){*p=static_cast<T*>(std::malloc(count));return *p?0:1;}
int cudaMemcpy(void* dst,const void* src,size_t count,int){std::memcpy(dst,src,count);return 0;}
int cudaMemset(void* dst,int value,size_t count){std::memset(dst,value,count);return 0;}
int cudaFree(void* pointer){std::free(pointer);return 0;}
int cudaEventCreate(int* value){*value=0;return 0;}
int cudaEventRecord(int){return 0;}int cudaEventDestroy(int){return 0;}
int cudaEventElapsedTime(float* value,int,int){*value=0;return 0;}
int cudaGetLastError(){return 0;}
std::string failure(){return std::getenv("MX_STUB_FAILURE")?std::getenv("MX_STUB_FAILURE"):"";}
int launches=0;
int cudaEventSynchronize(int){return failure()=="event"?1:0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3 block,void** args,int,void*) {
 ++launches;
 int length=*static_cast<int*>(args[0]);unsigned seed=*static_cast<unsigned*>(args[1]);
 auto stamp=*static_cast<gh::Stamp**>(args[2]);auto output=*static_cast<unsigned**>(args[3]);
 std::cerr<<"CPU_TARGET "<<length<<'\n';
 if(failure()=="cuda"||(failure()=="cuda_second"&&launches==2))return 1;
 if(grid.x!=1||block.x!=unsigned(threads))return 1;
 std::vector<unsigned> values;
 if(mode==3) {
  values.resize(threads*streams);
  for(int t=0;t<threads;++t)for(int stream=0;stream<streams;++stream)
   values[t*streams+stream]=seed+17*(t%32)+131*stream+8191*(t/32);
  // Iterative simultaneous lane permutation, independent of the closed oracle.
  const int steps=failure()=="wrong_steps"?length*8:length;
  for(int step=0;step<steps;++step) {
   auto old=values;
   for(int t=0;t<threads;++t)for(int stream=0;stream<streams;++stream)
    values[t*streams+stream]=old[((t/32)*32+(t%32+1)%32)*streams+stream];
  }
 } else {
  values.assign(32+256*matrices+2048,0);
  std::vector<unsigned> tile(2048);
  for(int q=0;q<8;++q)for(int m=0;m<4;++m)for(int row=0;row<8;++row)for(int col=0;col<8;++col)
   tile[q*256+m*64+row*8+col]=(1+row+11*col+97*m+997*q+seed%8191)&65535;
  std::fill(values.begin()+32+256*matrices,values.end(),0xdead);
  for(int row=0;row<8;++row)for(int col=0;col<8;++col) {
   int lane=trans?col*4+row/2:row*4+col/2;
   int half=trans?row%2:col%2;
   for(int q=0;q<8;++q)for(int m=0;m<matrices;++m) {
    auto value=tile[q*256+m*64+row*8+col];
    if(mode==1)value^=length&65535;
    values[32+(lane*8+q)*matrices+m]|=value<<(16*half);
    if(mode!=0)values[32+256*matrices+q*256+m*64+row*8+col]=value;
   }
  }
  if(mode!=1)for(int lane=0;lane<32;++lane)
   for(int iteration=0;iteration<length;++iteration)for(int q=0;q<8;++q)for(int m=0;m<matrices;++m)
    values[lane]+=values[32+(lane*8+q)*matrices+m];
 }
 for(size_t i=0;i<values.size();++i) {
  if(failure()=="omit_output"&&i==0)continue;
  output[i]=values[i];
 }
 if(failure()=="checksum")output[0]^=1;
 if(failure()=="fragment"&&mode!=3)output[32]^=1;
 if(failure()=="padding")output[values.size()-1]^=1;
 if(failure()=="many")for(size_t i=0;i<values.size();++i)output[i]^=1;
 *stamp={0,0,0,0,511};
 if(failure()=="stamp")stamp->end_ns=~gh::u64(0);
 if(failure()=="smid")stamp->smid=~unsigned(0);
 return 0;
}
'''


def no_duplicates(pairs):
    out={}
    for key,value in pairs:
        if key in out:raise ValueError('duplicate JSON field')
        out[key]=value
    return out


class MatrixValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='s10-host-CPU-')
        cls.root=Path(cls.temp.name);cls.binary=cls.root/'host'
        source=(BASE/'probes/matrix_exchange.cu').read_text()
        host=source[source.index('struct MXCase'):]
        host=host.replace('../common/word_artifacts.hpp',str(BASE/'common/word_artifacts.hpp'))
        assigns=[]
        for case in CONTRACT['cases']:
            p=case['parameters'];base=BASELINE[case['id']];m={'load':0,'store':1,'roundtrip':2,'shuffle':3}[p['mode']]
            assigns.append(f'if(mode=={m}&&matrices=={p.get("matrices",0)}&&threads=={case["threads"]}&&streams=={p.get("streams",0)}&&trans=={str(p.get("transpose",False)).lower()}){{a->numRegs={base["registers_per_thread"]};a->sharedSizeBytes={base["static_smem_bytes"]};}}')
        unit=SHIM.replace('RESOURCE_ASSIGNMENTS','\n'.join(assigns))
        unit+=function_body(source,'unsigned short mx_value(')+'\n'+host
        (cls.root/'host.cpp').write_text(unit)
        subprocess.run(['g++','-std=c++17','-O2',str(cls.root/'host.cpp'),'-o',str(cls.binary)],check=True,capture_output=True,timeout=30)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def run_host(self,case,failure='',profile=None,seed=3,conflict=False):
        profile=profile or PROFILES[1 if case['parameters']['mode']=='shuffle' else 0]
        folder=Path(tempfile.mkdtemp(dir=self.root))
        if conflict:(folder/'mx_output_0.u32le').write_bytes(b'existing immutable evidence')
        run=subprocess.run([str(self.binary),'validate-only',case['id'],profile['id'],str(seed)],cwd=folder,
                           env={**os.environ,'MX_STUB_FAILURE':failure},capture_output=True,text=True,timeout=20)
        rows=[json.loads(line,object_pairs_hook=no_duplicates) for line in run.stdout.splitlines()]
        return folder,run,rows,profile

    def test_actual_host_all22_cases_48_launches_and_full_offline_replay(self):
        launches=words=0
        for case in CONTRACT['cases']:
            folder,run,(device,row),profile=self.run_host(case)
            self.assertEqual(run.returncode,0,run.stderr)
            result=adapter.validate_validation(device,row,case,profile,3)
            verify_output_artifacts(folder,row)
            arrays={}
            for check in row['checks']:
                item=check['output_artifacts'][0];raw=(folder/item['path']).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),item['sha256'])
                arrays[item['path']]=list(struct.unpack('<'+'I'*(len(raw)//4),raw))
            replay=adapter.audit_artifact_values(case,profile,3,arrays)
            self.assertEqual(replay['checked_elements'],result['checked_elements'])
            self.assertFalse(replay['family_B3_qualified'])
            lengths=[int(line.split()[1]) for line in run.stderr.splitlines() if line.startswith('CPU_TARGET')]
            self.assertEqual(lengths,profile['target_iterations'])
            launches+=len(lengths);words+=result['checked_elements']
        self.assertEqual((launches,words),(48,98784))

    def test_all_output_regions_missing_stamp_and_CUDA_failures(self):
        case=next(c for c in CONTRACT['cases'] if c['id']=='store_x1_normal')
        for failure in ('omit_output','checksum','fragment','padding','many','stamp','smid','cuda','cuda_second','event'):
            folder,run,(device,row),profile=self.run_host(case,failure)
            self.assertNotEqual(run.returncode,0)
            self.assertLessEqual(sum(line.startswith('launch=') for line in run.stderr.splitlines()),8)
            with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,profile,3)
            self.assertEqual(len(row['target_launches']),2 if failure=='cuda_second' else 1)
            if failure=='cuda_second':self.assertTrue((folder/'mx_output_0.u32le').is_file())
        shuffle=CONTRACT['cases'][-1]
        _,run,(device,row),profile=self.run_host(shuffle,'wrong_steps')
        self.assertNotEqual(run.returncode,0)
        with self.assertRaises(ValueError):adapter.validate_validation(device,row,shuffle,profile,3)

    def test_artifact_exclusive_creation_preserves_previous_file(self):
        case=CONTRACT['cases'][0]
        folder,run,rows,profile=self.run_host(case,conflict=True)
        self.assertNotEqual(run.returncode,0)
        self.assertEqual((folder/'mx_output_0.u32le').read_bytes(),b'existing immutable evidence')
        with self.assertRaises(ValueError):adapter.validate_validation(rows[0],rows[1],case,profile,3)

    def test_profile_seed_and_actual_steps_rejected(self):
        case=CONTRACT['cases'][0]
        for profile,seed in [(PROFILES[1],3),(PROFILES[0],4)]:
            _,run,rows,_=self.run_host(case,profile=profile,seed=seed)
            self.assertNotEqual(run.returncode,0)
            self.assertNotIn('CPU_TARGET',run.stderr)
            with self.assertRaises(ValueError):adapter.validation_argv('binary/probe',case,profile,seed)
        for path in ('../probe','/tmp/probe','a//probe'):
            with self.assertRaises(ValueError):adapter.validation_argv(path,case,PROFILES[0],3)

    def test_metadata_resources_partial_artifact_and_value_tamper(self):
        case=CONTRACT['cases'][-1]
        folder,run,(device,original),profile=self.run_host(case)
        mutations=[lambda r:r.update(errors=False),lambda r:r.update(pilot_executed=True),lambda r:r.update(blocks=2),
          lambda r:r['target_launches'][1].update(iterations=64),lambda r:r['checks'].pop(),
          lambda r:r['checks'][0].update(completed=False),lambda r:r['checks'][0].update(verified_CTA_ids=[]),
          lambda r:r['checks'][0]['output_artifacts'][0].update(shape=[1]),
          lambda r:r['checks'][1]['output_artifacts'][0].update(path='mx_output_0.u32le'),
          lambda r:r['resource_identity'].update(registers_per_thread=1),
          lambda r:r['resource_identity'].update(occupancy_limit_ctas_per_sm=32)]
        for mutate in mutations:
            row=copy.deepcopy(original);mutate(row)
            with self.assertRaises(ValueError):adapter.validate_validation(device,row,case,profile,3)
        arrays={item['path']:list(struct.unpack('<'+'I'*item['shape'][0],(folder/item['path']).read_bytes()))
                for check in original['checks'] for item in check['output_artifacts']}
        arrays['mx_output_0.u32le'][0]^=1
        with self.assertRaises(ValueError):adapter.audit_artifact_values(case,profile,3,arrays)
        path=folder/'mx_output_0.u32le';path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaises(ValueError):verify_output_artifacts(folder,original)

    def test_original_device_and_formal_bytes_preserved(self):
        old=BASE.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/s10-validation-interface/matrix_exchange.initial.cu'
        original=old.read_text();current=(BASE/'probes/matrix_exchange.cu').read_text()
        self.assertTrue(current.startswith(original.replace('int main(int argc,char**argv)try{','int mx_formal_main(int argc,char**argv)try{')))
        self.assertEqual(current.split('struct MXCase')[0],original.split('struct MXCase')[0])
        request={'case':CONTRACT['cases'][0],'profile':PROFILES[0],'seed':3}
        self.assertEqual(adapter.validate_prior_evidence({},request,{})['status'],'insufficient')
        with self.assertRaises(ValueError):adapter.validate_prior_evidence({'source_artifacts_sha256':{'raw':'a'*64}},request,{'raw':'b'*64})

    def test_baseline_SASS_all22_and_changed_instruction_rejected(self):
        path=BASE.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-matrix-b/diagnostics/matrix_exchange/sass.txt'
        text=path.read_text();self.assertEqual(len(adapter.audit_sass(text,CONTRACT)),22)
        mutated=text.replace('SHFL.IDX','SHFL.BFLY',1)
        with self.assertRaises(ValueError):adapter.audit_sass(mutated,CONTRACT)


if __name__=='__main__':unittest.main()
