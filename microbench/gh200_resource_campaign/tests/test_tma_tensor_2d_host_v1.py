"""Real S15 short host on CPU CUDA/descriptor simulation; no GPU claims."""
import copy,hashlib,json,os,struct,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auditors import tma_tensor_2d_validation_v1 as adapter
from runners import validation_diagnostic as core
CONTRACT=json.loads((ROOT/'contracts/tma_tensor_2d.json').read_text())
PROFILES=json.loads((ROOT/'contracts/tma_tensor_2d_validation_profiles_v1.draft.json').read_text())['profiles']
POLICY=json.loads((ROOT/'contracts/early_validation_v1.json').read_text())
SHIM=r'''
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "REFERENCE_PATH"
#include "WRITER_PATH"
namespace gh {
using u64=unsigned long long;struct Stamp {u64 begin_ns,end_ns,begin_cycle,end_cycle;unsigned smid;};
struct Prop {int multiProcessorCount=2;size_t sharedMemPerBlockOptin=232448;};struct Device {Prop prop;};Device device(){return {};}
void emit_device(const Device&){std::cout<<"{\"schema_version\":2,\"type\":\"device\",\"cc\":\"9.0\",\"name\":\"CPU shim GH200\",\"uuid\":\"GPU-00000000-0000-0000-0000-000000000000\",\"sms\":2,\"driver_version\":12090,\"runtime_version\":12090,\"registers_per_sm\":65536,\"smem_per_sm_bytes\":233472,\"smem_per_cta_optin_bytes\":232448,\"l2_cache_bytes\":62914560,\"global_memory_bytes\":1073741824}\n";}
std::string quote(const std::string& s){return "\""+s+"\"";}
u64 integer(const char* s,u64 lo,u64 hi){char* end;auto v=std::strtoull(s,&end,10);if(*end||v<lo||v>hi)throw std::runtime_error("integer");return v;}
}
using cuuint64_t=std::uint64_t;using cuuint32_t=std::uint32_t;
struct alignas(64) CUtensorMap {unsigned char bytes[128];};struct MapFields {std::uint64_t base,w,rows,pitch;unsigned width,height,sw;};
constexpr int CUDA_SUCCESS=0,CU_TENSOR_MAP_DATA_TYPE_UINT16=16,CU_TENSOR_MAP_INTERLEAVE_NONE=0,CU_TENSOR_MAP_SWIZZLE_128B=128,CU_TENSOR_MAP_SWIZZLE_NONE=0,CU_TENSOR_MAP_L2_PROMOTION_NONE=0,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0;
int cuTensorMapEncodeTiled(CUtensorMap* map,int dtype,unsigned rank,void* base,const std::uint64_t* dim,const std::uint64_t* stride,const std::uint32_t* box,const std::uint32_t* step,int interleave,int sw,int l2,int oob){
 if(std::getenv("S15_DESCRIPTOR_FAIL"))return 7;
 if(dtype!=16||rank!=2||interleave||l2||oob||step[0]!=1||step[1]!=1||std::uintptr_t(base)%128||std::uintptr_t(map)%64||box[0]>256||box[1]>256||stride[0]%16||stride[0]<box[0]*2||(sw&&box[0]*2>128))return 8;
 std::memset(map,0,sizeof(*map));MapFields fields{std::uint64_t(std::uintptr_t(base)),dim[0],dim[1],stride[0],box[0],box[1],unsigned(sw)};std::memcpy(map,&fields,sizeof(fields));return 0;
}
unsigned selected_q=0;bool selected_g2s=false,selected_sw=false;
#define TT_FAKE(Q,SW,NAME) \
void tt_g2s_q##Q##_##NAME(){selected_q=Q;selected_g2s=true;selected_sw=SW;} \
void tt_s2g_q##Q##_##NAME(){selected_q=Q;selected_g2s=false;selected_sw=SW;}
TT_FAKE(1024,false,none) TT_FAKE(4096,false,none) TT_FAKE(8192,false,none) TT_FAKE(16384,false,none) TT_FAKE(32768,false,none) TT_FAKE(65536,false,none)
TT_FAKE(1024,true,sw128) TT_FAKE(4096,true,sw128) TT_FAKE(8192,true,sw128) TT_FAKE(16384,true,sw128) TT_FAKE(32768,true,sw128)
struct cudaFuncAttributes{int numRegs=40;size_t sharedSizeBytes=0,localSizeBytes=0;};struct dim3{unsigned x;dim3(unsigned n):x(n){}};
constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1,cudaFuncAttributeMaxDynamicSharedMemorySize=1;
#define GH_CUDA(call) do{if((call)!=0)throw std::runtime_error("CPU CUDA failure: " #call);}while(0)
int cudaFuncSetAttribute(const void* f,int,size_t){reinterpret_cast<void(*)()>(const_cast<void*>(f))();return 0;}
int cudaFuncGetAttributes(cudaFuncAttributes* a,const void*){*a={};return 0;}int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* o,const void*,unsigned,size_t){*o=1;return 0;}
int cudaMemGetInfo(size_t* f,size_t* t){*f=*t=1ull<<30;return 0;}
template<class T> int cudaMalloc(T** p,size_t n){void* raw=nullptr;int status=posix_memalign(&raw,256,n);*p=static_cast<T*>(raw);return status;}
int cudaMemcpy(void* a,const void* b,size_t n,int){std::memcpy(a,b,n);return 0;}int cudaMemset(void* p,int v,size_t n){std::memset(p,v,n);return 0;}
int cudaFree(void* p){std::free(p);return 0;}int cudaDeviceSynchronize(){return 0;}
int cudaLaunchKernel(const void*,dim3 grid,dim3 threads,void** args,size_t shared){
 auto* map=static_cast<CUtensorMap*>(args[0]);MapFields d{};std::memcpy(&d,map,sizeof(d));auto* global=*static_cast<std::uint16_t**>(args[1]);unsigned pitch=*static_cast<unsigned*>(args[2]);int n=*static_cast<int*>(args[3]);unsigned seed=*static_cast<unsigned*>(args[4]);
 if(!*static_cast<bool*>(args[5])||threads.x!=128||shared!=selected_q+1056||pitch!=d.pitch||std::uintptr_t(global)!=d.base||bool(d.sw)!=selected_sw)return 9;
 auto* logical=*static_cast<std::uint16_t**>(args[6]);auto* physical=*static_cast<std::uint16_t**>(args[7]);auto* stamps=*static_cast<gh::Stamp**>(args[8]);auto* counts=*static_cast<std::uint64_t**>(args[9]);const unsigned e=selected_q/2,w=d.width,h=d.height,stride=pitch/2;
 const std::string failure=std::getenv("S15_STUB_FAILURE")?std::getenv("S15_STUB_FAILURE"):"";
 for(unsigned b=0;b<grid.x;++b){
  if(stamps[b].smid!=~unsigned(0)||counts[b*5]!=~std::uint64_t(0))throw std::runtime_error("controls not poisoned");
  for(int i=0;i<n;++i){unsigned slot=unsigned(i)%32;
   for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){const auto at=(size_t(b)*n+i)*e+y*w+x;const auto value=std::uint16_t(17u*x+31u*y+(selected_g2s?73u*slot:0u)+151u*b+seed);
    if(logical[at]!=std::uint16_t(value^65535))throw std::runtime_error("logical poison missing");
    const unsigned group=(x/8)^(selected_sw?y%8:0);const unsigned physical_n=y*w+group*8+x%8;const auto pos=(size_t(b)*n+i)*e+physical_n;
    if(physical[pos]!=std::uint16_t(value^65535))throw std::runtime_error("physical poison missing");
    const auto addr=((size_t(b)*32+slot)*h+y)*stride+x;
    if(selected_g2s&&global[addr]!=value)throw std::runtime_error("input layout changed");
    if(!selected_g2s)global[addr]=value;logical[at]=value;physical[pos]=value;
   }
  }
  stamps[b]={1000,1000,2000,2000,b+137};counts[b*5]=n;counts[b*5+1]=selected_g2s?n*2:0;counts[b*5+2]=0;counts[b*5+3]=0;counts[b*5+4]=selected_g2s?0:n;
 }
 if(failure=="last_logical")logical[size_t(grid.x)*n*e-1]^=1;
 if(failure=="last_physical")physical[size_t(grid.x)*n*e-1]^=1;
 if(failure=="padding"&&stride>w)global[w]^=1;
 if(failure=="guard")global[-1]^=1;
 if(failure=="last_slot")global[(size_t(grid.x)*32*h-1)*stride+w-1]^=1;
 if(failure=="timeout")counts[2]=1;
 if(failure=="stamp")stamps[grid.x-1].smid=~unsigned(0);
 if(const char* log=std::getenv("S15_LAUNCH_LOG")){std::ofstream f(log,std::ios::app);f<<n<<"\n";}
 return 0;
}
'''
class TensorHost(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory();cls.binary=Path(cls.temp.name)/'host';source=Path(cls.temp.name)/'host.cpp'
  host=(ROOT/'probes/tma_tensor_2d.cu').read_text();host=host[host.index('struct TtCase'):]
  source.write_text('#include <fstream>\n'+SHIM.replace('REFERENCE_PATH',str(ROOT/'common/tma_tensor_2d_reference.hpp')).replace('WRITER_PATH',str(ROOT/'common/tensor_artifacts.hpp'))+host)
  result=subprocess.run(['g++','-std=c++17','-O2',str(source),'-o',str(cls.binary)],capture_output=True,text=True,timeout=30)
  if result.returncode:raise RuntimeError(result.stderr)
 @classmethod
 def tearDownClass(cls):cls.temp.cleanup()
 def host(self,case,profile,failure='',full_values=False):
  with tempfile.TemporaryDirectory(dir=self.temp.name) as folder:
   log=Path(folder)/'launches';run=subprocess.run([str(self.binary),'validate-only',case['id'],profile['id'],str(profile['required_seed'])],cwd=folder,capture_output=True,text=True,timeout=30,env={**os.environ,'S15_STUB_FAILURE':failure,'S15_LAUNCH_LOG':str(log)})
   rows=[json.loads(line) for line in run.stdout.splitlines()];launches=log.read_text().splitlines() if log.exists() else []
   if run.returncode==0:
    device,row=rows;core.validation_envelope(row,case,profile,profile['required_seed'],POLICY);adapter.validate_validation(device,row,case,profile,profile['required_seed']);core.verify_output_artifacts(Path(folder),row)
    if full_values:
     arrays={}
     for a in row['checks'][0]['output_artifacts']:
      data=(Path(folder)/a['path']).read_bytes();width,code={'uint16':(2,'H'),'uint32':(4,'I'),'uint8':(1,'B')}[a['dtype']]
      arrays[a['path']]={'shape':a['shape'],'values':list(struct.unpack('<'+code*(len(data)//width),data))}
     adapter.audit_values(device,row,case,profile,profile['required_seed'],arrays);self.last_arrays=arrays
   return run,rows,launches
 def test_actual_host_all68_coordinates_three_fixed_profiles(self):
  for case in CONTRACT['cases']:
   for profile in PROFILES:
    with self.subTest(case=case['id'],profile=profile['id']):
     run,rows,launches=self.host(case,profile);self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(launches,[str(profile['target_iterations'][0])])
 def test_full_value_replay_all_layouts_directions_with_wrap(self):
  wanted=['gmem_to_smem_1kib_continuous_none_one_cta','gmem_to_smem_1kib_padding_none_one_cta','gmem_to_smem_1kib_continuous_sw128_one_cta','smem_to_gmem_1kib_continuous_none_one_cta','smem_to_gmem_1kib_padding_none_one_cta','smem_to_gmem_1kib_continuous_sw128_one_cta']
  for case in CONTRACT['cases']:
   if case['id'] in wanted:
    run,_,_=self.host(case,PROFILES[2],full_values=True);self.assertEqual(run.returncode,0,run.stderr)
 def test_corruption_of_last_word_padding_guards_lifecycle_rejects(self):
  case=next(c for c in CONTRACT['cases'] if c['id']=='smem_to_gmem_1kib_padding_none_one_cta')
  for failure in ('last_logical','last_physical','padding','guard','last_slot','timeout','stamp'):
   run,rows,_=self.host(case,PROFILES[1],failure);self.assertNotEqual(run.returncode,0);self.assertTrue(rows[1]['errors']>0)

 def test_actual_host_metadata_and_full_value_rejections(self):
  case=CONTRACT['cases'][0];profile=PROFILES[1];run,rows,_=self.host(case,profile,full_values=True);self.assertEqual(run.returncode,0,run.stderr)
  for kind in ('regs','occupancy','grid','descriptor_failure','descriptor_dimensions','descriptor_float','base_alignment','shared_bytes','seed','profile','quantity'):
   row=copy.deepcopy(rows[1]);device=copy.deepcopy(rows[0]);e=row['resource_identity']['extensions']
   if kind=='regs':row['resource_identity']['registers_per_thread']=999
   elif kind=='occupancy':row['resource_identity']['occupancy_limit_ctas_per_sm']=99
   elif kind=='grid':row['blocks']=2
   elif kind=='descriptor_failure':e['descriptor_encode_status']=7
   elif kind=='descriptor_dimensions':e['global_dimensions'][1]+=1
   elif kind=='descriptor_float':e['global_dimensions'][0]=64.0
   elif kind=='base_alignment':e['global_base_address']+=2
   elif kind=='shared_bytes':row['resource_identity']['dynamic_smem_bytes']-=1024
   elif kind=='seed':row['seed']=0
   elif kind=='profile':row['profile_id']=PROFILES[0]['id']
   else:row['checks'][0]['checked_elements']+=1
   with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.validate_validation(device,row,case,profile,3)
  for name in self.last_arrays:
   arrays=copy.deepcopy(self.last_arrays);arrays[name]['values'][-1]^=1
   with self.subTest(artifact=name),self.assertRaises(ValueError):adapter.audit_values(rows[0],rows[1],case,profile,3,arrays)
  arrays=copy.deepcopy(self.last_arrays);arrays['tensor_stamps.u32le']['values'][0]=1000.5
  with self.assertRaises(ValueError):adapter.audit_values(rows[0],rows[1],case,profile,3,arrays)
 def test_descriptor_failure_before_target_and_illegal_matrix(self):
  case=CONTRACT['cases'][0];profile=PROFILES[0]
  with tempfile.TemporaryDirectory(dir=self.temp.name) as folder:
   log=Path(folder)/'launches';run=subprocess.run([str(self.binary),'validate-only',case['id'],profile['id'],'0'],cwd=folder,capture_output=True,text=True,timeout=30,env={**os.environ,'S15_DESCRIPTOR_FAIL':'1','S15_LAUNCH_LOG':str(log)})
   self.assertNotEqual(run.returncode,0);self.assertFalse(log.exists())
  for kind in ('sw64','pitch','box','padding_sw'):
   bad=copy.deepcopy(case)
   if kind=='sw64':bad['parameters'].update(payload_bytes=65536,row_stride_bytes=256,swizzle='SW128',box_dim=[128,256])
   elif kind=='pitch':bad['parameters']['row_stride_bytes']=140
   elif kind=='box':bad['parameters']['box_dim']=[64,257]
   else:bad['parameters'].update(row_stride_bytes=144,swizzle='SW128')
   with self.subTest(kind=kind),self.assertRaises(ValueError):adapter.geometry(bad)

if __name__=='__main__':unittest.main()
