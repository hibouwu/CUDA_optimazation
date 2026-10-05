"""Real S13 host + real common warmup/JSON emitter on CPU CUDA simulation.

This does not execute a device kernel or qualify GPU correctness.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE/'tests')]
from test_global_duplex_validation import SHIM, unique_object
from auditors import global_duplex, global_duplex_formal_v1

CONTRACT = json.loads((BASE/'contracts/global_duplex.json').read_text())
PROTOCOL = json.loads((BASE/'contracts/protocol.json').read_text())


class GlobalDuplexFormalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='s13-formal-host-')
        cls.binary = Path(cls.temporary.name)/'host'
        runtime = (BASE/'common/probe_runtime.cuh').read_text()
        actual_host_runtime = runtime[runtime.index('struct Observation {'):].rsplit('}',1)[0]
        shim = '#include <cmath>\n#include <iomanip>\n#include <fstream>\n#define GH_TIMER_RESOLUTION_V3 0\n'+SHIM
        shim = shim.replace('struct Observation {std::vector<Stamp> stamps;double event_ms=0;u64 errors=0,checked_elements=0;std::string method,input_conditions;};','enum class TimerPolicy { StrictPositiveNs, ZeroWorkOneCta, ZeroWorkGrid };')
        begin = shim.index('template<class F> int warmup')
        end = shim.index('\n}\nconstexpr unsigned gd_threads', begin)
        shim = shim[:begin]+actual_host_runtime+shim[end:]
        shim = shim.replace('cudaEventElapsedTime(float* p,int,int){*p=0;', 'cudaEventElapsedTime(float* p,int,int){*p=1.1f;')
        shim = shim.replace('stamps[b]={0,0,0,0,b+100}', 'stamps[b]={1000,1001000,2000,2002000,b%2}')
        # Before each simulated launch, inspect all real host-initialized poison.
        check = '''
 if(iterations!=16)throw std::runtime_error("formal length changed");
 for(size_t t=0;t<threads;++t){
   const unsigned expected=global_duplex_reference::read_checksum(vectors*16,threads,t,selected_reads,iterations,seed);
   if(sums[t]!=(expected^~unsigned(0)))throw std::runtime_error("checksum poison missing");
 }
 if(output)for(size_t word=0;word<vectors*4;++word){
   const unsigned expected=(selected_copy?17u:29u)*unsigned(word)+seed;
   if(reinterpret_cast<unsigned*>(output)[word]!=(expected^~unsigned(0)))throw std::runtime_error("destination poison missing");
 }
 for(unsigned b=0;b<grid.x;++b)if(stamps[b].begin_ns!=~gh::u64(0)||stamps[b].smid!=~unsigned(0))throw std::runtime_error("stamp poison missing");
 if(const char* p=std::getenv("S13_LAUNCH_LOG")){std::ofstream f(p,std::ios::app);f<<iterations<<"\\n";}
'''
        shim = shim.replace(' // Enumerate owner vectors directly:',check+' // Enumerate owner vectors directly:')
        shim = shim.replace(' if(fail=="many")', ' if(fail=="measurement_corrupt"&&launch==9)sums[threads-1]^=1;\n if(fail=="many")')
        source = (BASE/'probes/global_duplex.cu').read_text()
        path = Path(cls.temporary.name)/'host.cpp'
        path.write_text(shim.replace('REFERENCE_PATH',str(BASE/'common/global_duplex_reference.hpp'))+source[source.index('struct GdCase'):])
        subprocess.run(['g++','-std=c++17','-O2',str(path),'-o',str(cls.binary)],check=True,capture_output=True,timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def execute(self,case,seed=3,failure=''):
        with tempfile.TemporaryDirectory(dir=self.temporary.name) as name:
            log=Path(name)/'launches'
            result=subprocess.run([str(self.binary),case['id'],'16',str(seed)],capture_output=True,text=True,
                timeout=30,env={**os.environ,'S13_STUB_FAILURE':failure,'S13_LAUNCH_LOG':str(log)})
            launches=log.read_text().splitlines() if log.exists() else []
            rows=[json.loads(line,object_pairs_hook=unique_object) for line in result.stdout.splitlines()]
            return result,rows,launches

    def test_all18_formal_hosts_three_seeds_real_warmup_and_emitter(self):
        for case in CONTRACT['cases']:
            for seed in (0,3,4294967295):
                with self.subTest(case=case['id'],seed=seed):
                    run,rows,launches=self.execute(case,seed)
                    self.assertEqual(run.returncode,0,run.stderr)
                    self.assertEqual(launches,['16']*9)
                    device,row=rows
                    result=global_duplex.validate_trial(row,case,device,seed,PROTOCOL)
                    self.assertEqual(result['warmup_cv'],0)
                    self.assertEqual(row['warmup_samples_ns'],[1000000]*8)
                    p=case['parameters'];size=row['array_bytes']
                    self.assertEqual(row['work_count'],16*size*(p['read_requests_per_group']+p['write_requests_per_group']))
                    self.assertEqual(result['value'],row['work_count']/1000000)

    def test_formal_warmup_and_measurement_corruption_reject(self):
        case=next(c for c in CONTRACT['cases'] if c['id']=='dependent_copy_small')
        for failure in ('missing_checksum','missing_destination','omit_lane','copy_dependency','stamp','measurement_corrupt','cuda'):
            with self.subTest(failure=failure):
                run,rows,launches=self.execute(case,failure=failure)
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(any(r.get('type')=='trial' for r in rows))
                self.assertLessEqual(len(launches),9)

    def test_wrapper_requires_all_B3_resources_working_set_and_admission(self):
        case=CONTRACT['cases'][0]
        run,(device,row),_=self.execute(case)
        self.assertEqual(run.returncode,0,run.stderr)
        resource={k:row[k] for k in ('kernel_symbol','registers_per_thread','static_smem_bytes','dynamic_smem_bytes','local_size_bytes','occupancy_limit_ctas_per_sm')}
        resource['extensions']={k:row[k] for k in ('requested_array_bytes','array_bytes','aggregate_array_bytes','read_requests_per_group','write_requests_per_group')}
        admitted={**case,'b3_blocks':row['blocks'],'b3_resource_identity':resource}
        global_duplex_formal_v1.validate_trial(row,admitted,device,3,PROTOCOL)
        with self.assertRaises(ValueError):global_duplex_formal_v1.validate_trial(row,case,device,3,PROTOCOL)
        for key in [*resource['extensions'],'blocks','kernel_symbol','registers_per_thread','occupancy_limit_ctas_per_sm','work_count']:
            bad=copy.deepcopy(row);bad[key]=bad[key]+1 if isinstance(bad[key],int) else bad[key]+'bad'
            with self.subTest(key=key),self.assertRaises(ValueError):
                global_duplex_formal_v1.validate_trial(bad,admitted,device,3,PROTOCOL)


if __name__=='__main__':unittest.main()
