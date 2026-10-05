"""CPU-only regressions for versioned zero-work timer acceptance and real C++ host code."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from auditors import memory_baseline,legacy_compute
from auditors.observation import TIMER_REVISION


class TimerResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol=json.loads((BASE/'contracts/protocol.json').read_text())
        cls.contracts={name:json.loads((BASE/'contracts'/(name+'.json')).read_text())
                       for name in ('memory_baseline','memory_baseline_timer_v3','legacy_fma_timer_v3',
                                    'legacy_mma_timer_v3','legacy_wgmma_timer_v3')}
        cls.device={'schema_version':2,'type':'device','uuid':'GPU-43269fbc-449d-3e0f-908a-9c81229546d3',
                    'name':'NVIDIA GH200 120GB','cc':'9.0','sms':2,'driver_version':13010,'runtime_version':12090}

    def case(self,name='empty_one_cta',contract='memory_baseline_timer_v3'):
        return copy.deepcopy(next(c for c in self.contracts[contract]['cases'] if c['id']==name))

    def row(self,case):
        revised=case.get('timer_resolution_revision')==TIMER_REVISION
        one=case['scope']=='one_cta';compute=case['work_model'].startswith('compute_') or case['work_model'].endswith('_issue_v1')
        blocks=1 if one else self.device['sms']*4
        if compute:
            e=legacy_compute.arithmetic(case,blocks)
            correctness={'method':'uniform_and_nonuniform_full_output_v1','checked_elements':e['checked'],
                         'input_conditions':'legacy uniform constants; separate seeded nonuniform validation; D reset each launch'}
        else:
            e=memory_baseline.work(case,self.device);e['work']=e['read']+e['write']
            empty=case['work_model']=='empty_window';shared=case['work_model']=='shared_scalar8'
            correctness={'method':'empty_window_timestamps' if empty else 'host_per_thread_address_checksum' if shared else 'host_checksum_and_optional_full_array_store_validation',
                         'checked_elements':e['checked'],'input_conditions':'no memory workload' if empty else 'shared[i]=17*i+3; write initialization poison' if shared else 'global[i]=17*i+seed modulo 2^32'}
        details=[{'block_id':i,'smid':i%2,'start_ns':1000+i*10,'stop_ns':1100+i*10,
                  'start_cycle':1205937031242820+i*1000,'stop_cycle':1205937031242845+i*1000} for i in range(blocks)]
        row={'schema_version':2,'type':'trial','case_id':case['id'],'iterations':case['iterations'],'seed':7,
             'threads':case['threads'],'blocks':blocks,'scope':case['scope'],'errors':0,'work_unit':case['work_unit'],
             'work_count':e['work'],'read_payload_bytes':e['read'],'write_payload_bytes':e['write'],
             'start_ns':1000,'stop_ns':1100+(blocks-1)*10,'event_ms':.01,'blocks_detail':details,
             'warmup_samples_ns':[100]*8,'warmup_converged':True,'correctness':correctness,
             'cache_residency_proven':False,'physical_hbm_bytes_proven':False}
        if compute:
            row.update(timing_model='v2_cta_start_gate_result_drain_v1',phase=case['parameters']['phase'],
                       registers_per_thread=32,static_smem_bytes=4096,local_size_bytes=0,occupancy_limit_ctas_per_sm=8,
                       max_abs_error=0,output_elements_checked=e['checked'],
                       nonuniform_validation={'iterations':[1,2],'checked_elements':e['checked']*2,'input_seed':7,'errors':0,
                         'reference_model':'integer_dyadic_rne_fma_v1' if case['parameters']['ptx'].startswith('fma.') else 'integer_dyadic_logical_matrix_v1'})
        else:row.update(requested_working_set_bytes=case['parameters']['bytes'],allocation_per_array_bytes=e['allocation'],stride=case['parameters']['stride'])
        if revised:row.update(timer_resolution_revision=TIMER_REVISION,
            warmup_basis='cta_clock64_cycles' if case['timer_acceptance']=='zero_work_one_cta_local_cycles' else 'elapsed_ns',
            warmup_samples_cycles=[25]*8 if one else [],globaltimer_distinguishable=True,zero_ns_ctas=0)
        return row

    def audit(self,row,case,compute=False):
        return (legacy_compute if compute else memory_baseline).validate_trial(row,case,self.device,7,self.protocol)

    def equal_ns(self,row):
        for d in row['blocks_detail']:d['stop_ns']=d['start_ns']
        row['stop_ns']=max(d['stop_ns'] for d in row['blocks_detail'])
        row['globaltimer_distinguishable']=row['stop_ns']>row['start_ns']
        row['zero_ns_ctas']=len(row['blocks_detail'])
        row['warmup_samples_ns']=[row['stop_ns']-row['start_ns']]*8

    def test_new_contracts_parse_and_old_contracts_are_strict(self):
        for name,c in self.contracts.items():
            (memory_baseline if name.startswith('memory') else legacy_compute).validate_contract(c)
        old=self.case(contract='memory_baseline');row=self.row(old)
        self.assertEqual(self.audit(row,old)['value'],100)
        self.equal_ns(row)
        with self.assertRaises(ValueError):self.audit(row,old)
        revised=self.case();row=self.row(revised)
        with self.assertRaises(ValueError):self.audit(row,old)
        unversioned=copy.deepcopy(self.contracts['memory_baseline'])
        grid=next(c for c in unversioned['cases'] if c['id']=='empty_all_gpu')
        grid.update(timer_resolution_revision=TIMER_REVISION,timer_acceptance='zero_work_grid_ns')
        with self.assertRaises(ValueError):memory_baseline.validate_contract(unversioned)

    def test_captured_equal_ns_is_cycles_not_zero_latency(self):
        case=self.case();row=self.row(case);self.equal_ns(row)
        d=row['blocks_detail'][0];d['start_ns']=d['stop_ns']=1790838389952521376
        row['start_ns']=row['stop_ns']=d['start_ns']
        result=self.audit(row,case)
        self.assertEqual((result['value'],result['unit'],result['warmup_basis']),(25,'cycles/window','cta_clock64_cycles'))
        self.assertFalse(result['globaltimer_distinguishable'])
        self.assertEqual(row['warmup_samples_ns'],[0]*8)

    def test_ordinary_equal_ns_and_false_empty_or_nonzero_work_rejected(self):
        for name in ('smem_read_stride1','global_read_ca_8m'):
            case=self.case(name);row=self.row(case);self.equal_ns(row)
            with self.assertRaises(ValueError):self.audit(row,case)
        case=self.case();row=self.row(case);self.equal_ns(row)
        for field in ('work_count','read_payload_bytes','write_payload_bytes'):
            bad=copy.deepcopy(row);bad[field]=1
            with self.subTest(field=field),self.assertRaises(ValueError):self.audit(bad,case)
        badcase=copy.deepcopy(case);badcase['parameters']['mode']='smem_read'
        with self.assertRaises(ValueError):self.audit(row,badcase)

    def test_local_cycles_and_warmup_unit_checks(self):
        case=self.case();row=self.row(case);self.equal_ns(row)
        bads=[]
        bad=copy.deepcopy(row);bad['blocks_detail'][0]['stop_cycle']=bad['blocks_detail'][0]['start_cycle'];bads.append(bad)
        bad=copy.deepcopy(row);bad['blocks_detail'][0]['stop_ns']-=1;bads.append(bad)
        bad=copy.deepcopy(row);bad['warmup_basis']='elapsed_ns';bads.append(bad)
        bad=copy.deepcopy(row);bad['warmup_samples_cycles']=[25]*7;bads.append(bad)
        bad=copy.deepcopy(row);bad['warmup_samples_cycles'][0]=0;bads.append(bad)
        bad=copy.deepcopy(row);bad.pop('warmup_samples_cycles');bads.append(bad)
        bad=copy.deepcopy(row);bad['globaltimer_distinguishable']=True;bads.append(bad)
        bad=copy.deepcopy(row);bad['zero_ns_ctas']=0;bads.append(bad)
        for bad in bads:
            with self.assertRaises(ValueError):self.audit(bad,case)
        # NS array can be non-convergent; only cycle series controls this case.
        row['warmup_samples_ns']=[0,32]*4
        self.assertTrue(self.audit(row,case)['warmup_converged'])
        row['warmup_samples_cycles']=[25,100]*4
        with self.assertRaises(ValueError):self.audit(row,case)

    def test_grid_equal_cta_ns_requires_positive_grid_without_cycle_metric(self):
        case=self.case('empty_all_gpu');row=self.row(case);self.equal_ns(row)
        result=self.audit(row,case);self.assertEqual((result['value'],result['unit']),(70,'ns/window'))
        bad=copy.deepcopy(row);bad['warmup_samples_cycles']=[25]*8
        with self.assertRaises(ValueError):self.audit(bad,case)
        badcase=copy.deepcopy(case);badcase['metric']={'numerator':'cta_clock64_cycles','denominator':'one','scale':1,'unit':'cycles/window'}
        with self.assertRaises(ValueError):self.audit(row,badcase)
        for d in row['blocks_detail']:d['start_ns']=d['stop_ns']=1000
        row['stop_ns']=1000;row['globaltimer_distinguishable']=False;row['warmup_samples_ns']=[0]*8
        with self.assertRaises(ValueError):self.audit(row,case)

    def test_compute_empty_models_and_positive_work_paths(self):
        for family in ('legacy_fma','legacy_mma','legacy_wgmma'):
            c=self.contracts[family+'_timer_v3']
            empty=next(x for x in c['cases'] if x['work_model']=='compute_empty_window_v1')
            row=self.row(empty);self.equal_ns(row)
            self.assertEqual(self.audit(row,empty,True)['value'],25)
            measured=next(x for x in c['cases'] if x['work_model']!='compute_empty_window_v1')
            row=self.row(measured);self.equal_ns(row)
            with self.assertRaises(ValueError):self.audit(row,measured,True)

    def test_real_cpp_header_acceptance_and_emission_without_cuda(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('host C++ compiler unavailable')
        stub='''#pragma once
#include <cstddef>
using cudaError_t=int;constexpr int cudaSuccess=0;
struct cudaUUID_t{char bytes[16];};
struct cudaDeviceProp{int major,minor;char name[256];cudaUUID_t uuid;int multiProcessorCount;std::size_t totalGlobalMem;int l2CacheSize,regsPerMultiprocessor;std::size_t sharedMemPerMultiprocessor,sharedMemPerBlockOptin;};
const char* cudaGetErrorString(int);int cudaGetDeviceCount(int*);int cudaGetDeviceProperties(cudaDeviceProp*,int);int cudaDriverGetVersion(int*);int cudaRuntimeGetVersion(int*);
'''
        source='''#include "probe_runtime.cuh"
int main(int argc,char**argv) {int mode=std::atoi(argv[1]);try {
 gh::Observation o;o.event_ms=.001;o.timer_policy=gh::control_timer_policy(mode!=1&&mode!=8,mode!=2&&mode!=3);
 o.stamps.push_back({1790838389952521376ull,1790838389952521376ull,1205937031242820ull,1205937031242845ull,124});
 if(mode==8)o.stamps[0].end_ns+=32;
 if(mode==2||mode==3){auto s=o.stamps[0];s.smid=125;s.begin_cycle+=1000000;s.end_cycle+=1000000;if(mode==2){s.begin_ns+=32;s.end_ns+=32;}o.stamps.push_back(s);}
 if(mode==4)o.stamps.push_back(o.stamps[0]);
 if(mode==6)o.stamps[0].end_cycle=o.stamps[0].begin_cycle;
 auto warm=gh::warmup([&](){return o;});
 gh::emit_trial("CPU_TEST_ONLY",1,7,256,(mode==2||mode==3)?"all_gpu":"one_cta","operation",mode==5?1:0,0,0,o,warm);
 }catch(const std::exception&e){std::cout<<"rejected: "<<e.what()<<"\\n";}}
'''
        with tempfile.TemporaryDirectory(prefix='timer-header-cpu-') as tmp:
            p=Path(tmp);(p/'cuda_runtime.h').write_text(stub);(p/'main.cpp').write_text(source)
            def compile_mode(revision):
                binary=p/('v'+str(revision));subprocess.run([compiler,'-std=c++17','-O2',f'-DGH_TIMER_RESOLUTION_V3={revision}',
                    '-I',str(p),'-I',str(BASE/'common'),str(p/'main.cpp'),'-o',str(binary)],check=True,capture_output=True,timeout=30);return binary
            def run(binary,mode):return subprocess.run([str(binary),str(mode)],check=True,capture_output=True,text=True,timeout=10).stdout
            revised=compile_mode(1);row=json.loads(run(revised,0))
            self.assertEqual(row['warmup_samples_ns'],[0]*8);self.assertEqual(row['warmup_samples_cycles'],[25]*8)
            self.assertEqual(row['warmup_basis'],'cta_clock64_cycles');self.assertFalse(row['globaltimer_distinguishable'])
            grid=json.loads(run(revised,2));self.assertEqual(grid['warmup_basis'],'elapsed_ns');self.assertEqual(grid['warmup_samples_cycles'],[])
            for mode in (1,3,4,5,6):self.assertTrue(run(revised,mode).startswith('rejected:'),mode)
            original=compile_mode(0);self.assertTrue(run(original,0).startswith('rejected:'))
            ordinary=json.loads(run(original,8));self.assertNotIn('timer_resolution_revision',ordinary)


if __name__=='__main__':unittest.main()
