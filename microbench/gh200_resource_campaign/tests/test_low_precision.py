"""S08 CPU-only evidence. No test launches the CUDA probe or asserts GPU support."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from auditors import low_precision as audit
from auditors import low_precision_reference as reference

spec = importlib.util.spec_from_file_location('lp_generator', BASE/'probes/generate_low_precision.py')
generator = importlib.util.module_from_spec(spec); spec.loader.exec_module(generator)


class LowPrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((BASE/'contracts/low_precision.json').read_text())
        cls.lowered = json.loads((BASE/'contracts/low_precision_lowering_v3.json').read_text())
        cls.protocol = json.loads((BASE/'contracts/protocol.json').read_text())
        cls.device = {'schema_version': 2, 'type': 'device', 'name': 'NVIDIA GH200', 'cc': '9.0',
                      'uuid': 'GPU-00000000-0000-0000-0000-000000000001', 'sms': 132,
                      'driver_version': 13010, 'runtime_version': 12090,
                      'smem_per_sm_bytes': 233472, 'smem_per_cta_optin_bytes': 232448, 'registers_per_sm': 65536}

    def compile_run(self, source, stdin=''):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('CPU C++ compiler unavailable')
        with tempfile.TemporaryDirectory(prefix='gh200-lowprecision-cpu-') as directory:
            root = Path(directory); (root/'test.cpp').write_text(source)
            subprocess.run([compiler, '-std=c++17', '-O2', '-I', str(BASE/'common'),
                            str(root/'test.cpp'), '-o', str(root/'test')], check=True, capture_output=True, timeout=30)
            return subprocess.run([str(root/'test')], input=stdin, text=True, check=True,
                                  capture_output=True, timeout=30).stdout

    def test_frozen_matrix_and_generator_replay(self):
        audit.validate_contract(self.contract)
        self.assertEqual(generator.generate(), (BASE/'probes/low_precision.cu').read_text())
        self.assertEqual(len(self.contract['cases']), 32)
        self.assertEqual(generator.generate().count('__global__ void lp_'), 16)

    def test_wgmma_register_boundaries_cover_float_and_integer_accumulators(self):
        source=generator.generate()
        checks=audit.validate_source_register_boundaries(source)
        self.assertEqual(len(checks),8)
        self.assertTrue(all(x['registers_per_boundary']==64 and x['boundaries']==2 for x in checks))
        self.assertEqual({x['constraint'] for x in checks},{'+f','+r'})
        for constraint,label in itertools.product(('f','r'),('BEFORE_WGMMA_FENCE','AFTER_FINAL_WAIT')):
            boundary=generator.register_boundary(constraint,label)
            self.assertIn(boundary,source)
            with self.assertRaises(ValueError):
                audit.validate_source_register_boundaries(source.replace(boundary,'',1))
        with self.assertRaises(ValueError):
            audit.validate_source_register_boundaries(source.replace('for(int j=0;j<32;++j) asm volatile','for(int j=0;j<31;++j) asm volatile',1))

    def test_contract_mutations_rejected(self):
        for mutate in (
            lambda x: x['cases'].pop(),
            lambda x: x['cases'].append(deepcopy(x['cases'][0])),
            lambda x: x['cases'][0]['parameters'].update(groups=2),
            lambda x: x['cases'][0]['parameters'].update(chains=1),
            lambda x: x['cases'][0]['parameters'].update(shape=[16, 8, 16]),
            lambda x: x['cases'][0]['parameters'].update(saturating=True),
            lambda x: x['cases'][0].update(work_unit='OP'),
            lambda x: x['cases'][0].update(iterations=0),
            lambda x: x['cases'][0]['metric'].update(denominator='elapsed_ns'),
            lambda x: x['cases'][0]['parameters'].update(ptx='mma.sp.fake'),
            lambda x: x['fragments']['mma_A'].update(row='lane//4'),
            lambda x: x['fragments']['wgmma_AB_smem'].update(stride_offset_bytes=256),
            lambda x: x['correctness'].update(S8_nonuniform='all zeros'),
        ):
            changed = deepcopy(self.contract); mutate(changed)
            with self.assertRaises(ValueError):
                audit.validate_contract(changed)

    def test_fragment_bijections_and_smem_bytes(self):
        for path, role, threads, elements, shape in [
            ('mma', 'A', 32, 16, (16, 32)), ('mma', 'B', 32, 8, (32, 8)),
            ('mma', 'D', 32, 4, (16, 8)), ('wgmma', 'D', 128, 32, (64, 64))]:
            coords = [reference.coordinate(path, role, t, e) for t in range(threads) for e in range(elements)]
            self.assertEqual(len(coords), len(set(coords)))
            self.assertEqual(set(coords), set(itertools.product(range(shape[0]), range(shape[1]))))
        addresses = [(o%8)*16+(o//8)*128+k%16+(k//16)*1024 for o in range(64) for k in range(32)]
        self.assertEqual(sorted(addresses), list(range(2048)))

    def test_actual_device_byte_encoder_against_independent_fp8_decoder(self):
        encoder = generator.PREFIX.split('template<int Kind>', 1)[1].split('struct LowCase', 1)[0]
        source = '#include <iostream>\n#define __device__\n#define __forceinline__ inline\ntemplate<int Kind>' + encoder
        source += r'''
template<int K>void emit(){for(unsigned seed:{0u,3u,4294967295u})
for(unsigned o=0;o<64;++o)for(unsigned k=0;k<32;++k)for(bool a:{false,true})for(bool varied:{false,true})
std::cout<<K<<" "<<seed<<" "<<o<<" "<<k<<" "<<a<<" "<<varied<<" "<<unsigned(lp_operand<K>(o,k,seed,a,varied))<<"\n";}
int main(){emit<0>();emit<1>();emit<2>();emit<3>();}
'''
        for line in self.compile_run(source).splitlines():
            kind, seed, outer, inner, a, varied, bits = map(int, line.split()); name = audit.KINDS[kind]
            expected = reference.input_value(name, outer, inner, seed, bool(a)) if varied else Fraction(1, 16) if kind < 2 else 1
            got = reference.fp8_value(bits, name) if kind < 2 else bits - 256 if kind == 2 and bits >= 128 else bits
            self.assertEqual(got, expected, (name, seed, outer, inner, a, varied, bits))

    def test_actual_mma_packing_and_byte_order(self):
        encoder = 'template<int Kind>' + generator.PREFIX.split('template<int Kind>', 1)[1].split('struct LowCase', 1)[0]
        fragments = []
        for kind in audit.KINDS:
            setup = generator.kernel('mma', kind, 1).split('gh::Stamp* stamps,double* output) {', 1)[1].split('__shared__ volatile double drain', 1)[0]
            fragments.append('{'+setup+r'''
for(int r=0;r<4;++r)std::cout<<Kind<<" A "<<threadIdx.x<<" "<<r<<" "<<a[r]<<"\n";
for(int r=0;r<2;++r)std::cout<<Kind<<" B "<<threadIdx.x<<" "<<r<<" "<<b[r]<<"\n";
}''')
        source = '#include <iostream>\n#define __device__\n#define __forceinline__ inline\n'+encoder
        source += 'struct {unsigned x;}threadIdx;int main(){unsigned seed=4294967295u;bool nonuniform=true;for(unsigned t=0;t<32;++t){threadIdx.x=t;'
        source += '\n'.join(fragments)+'}}'
        for line in self.compile_run(source).splitlines():
            kind, role, lane, register, word = line.split(); name=audit.KINDS[int(kind)]; lane=int(lane); register=int(register); word=int(word)
            expected=[]
            for j in range(4):
                row,col=reference.coordinate('mma',role,lane,4*register+j)
                expected.append(reference.input_value(name,row,col,4294967295,True) if role=='A'
                                else reference.input_value(name,col,row,4294967295,False))
            self.assertEqual(word,reference.packed_word(expected,name))
        self.assertEqual(reference.packed_word([-2,-1,0,1], 's8'),0x0100fffe)
        with self.assertRaises(ValueError):reference.packed_word([-1,0,1,2], 'u8')

    def test_cpp_full_output_reference_against_fraction_oracle(self):
        source = r'''
#include <iostream>
#include <iomanip>
#include "low_precision_reference.hpp"
int main(){int kind,wg,groups,length,varied;unsigned seed;std::cout<<std::setprecision(17);
while(std::cin>>kind>>wg>>groups>>length>>seed>>varied){
auto v=low_precision_reference::outputs(kind,wg,groups,length,seed,varied);
for(auto x:v)std::cout<<x<<" ";std::cout<<"\n";}}
'''
        specs = [(k,w,2,it,4294967295,1) for k in range(4) for w in (0,1) for it in (1,2)]
        specs += [(k,w,1,8192,3,0) for k in range(4) for w in (0,1)]
        output = self.compile_run(source, '\n'.join(' '.join(map(str,s)) for s in specs)).splitlines()
        self.assertEqual(len(output),len(specs))
        for line, (kind,wg,groups,length,seed,varied) in zip(output,specs):
            expected = reference.output_reference(audit.KINDS[kind], 'wgmma' if wg else 'mma', groups,length,seed,bool(varied))
            self.assertEqual(list(map(float,line.split())),list(map(float,expected)))

    def test_actual_wgmma_shared_operand_initialization(self):
        encoder = 'template<int Kind>' + generator.PREFIX.split('template<int Kind>', 1)[1].split('struct LowCase', 1)[0]
        fragments = []
        for kind in audit.KINDS:
            setup = generator.kernel('wgmma', kind, 1).split('gh::Stamp* stamps,double* output) {', 1)[1].split('__syncthreads();', 1)[0]
            fragments.append('{'+setup+r'''
for(unsigned i=0;i<2048;++i)std::cout<<Kind<<" "<<i<<" "<<unsigned(as[i])<<" "<<unsigned(bs[i])<<"\n";
}''')
        source = '#include <iostream>\n#define __device__\n#define __forceinline__ inline\n#define __shared__\n#define __align__(N) alignas(N)\n'+encoder
        source += 'struct {unsigned x;}threadIdx,blockDim;int main(){threadIdx.x=0;blockDim.x=1;unsigned seed=4294967295u;bool nonuniform=true;'
        source += '\n'.join(fragments)+'}'
        inverse = {(o%8)*16+(o//8)*128+k%16+(k//16)*1024:(o,k) for o in range(64) for k in range(32)}
        for line in self.compile_run(source).splitlines():
            kind, address, a, b = map(int,line.split()); encoding=audit.KINDS[kind];outer,inner=inverse[address]
            for bits,is_a in [(a,True),(b,False)]:
                got=reference.fp8_value(bits,encoding) if kind<2 else bits-256 if kind==2 and bits>=128 else bits
                self.assertEqual(got,reference.input_value(encoding,outer,inner,4294967295,is_a))

    def test_actual_failure_coordinate_diagnostic(self):
        helper=generator.HOST.split('struct LowValidationCheck',1)[0]
        source='#include <iostream>\n#include <sstream>\n#include <iomanip>\n#include <string>\n'+helper
        source+='int main(){std::cout<<lp_difference(8192+64*73+32+19,8192,128,0.5,-0.25)<<"\\n";}'
        line=self.compile_run(source).strip()
        self.assertIn('block=1 thread=73 group=0 lane=9 chain=1 fragment=19 row=42 col=35',line)
        self.assertIn('actual=0.5 expected=-0.25',line)
        self.assertIn('actual_hex=0x1p-1 expected_hex=-0x1p-2',line)
        self.assertIn('if(o.errors<=8)',generator.HOST)
        self.assertIn('!std::isfinite(got[i])||got[i]!=expected[i%per_cta]',generator.HOST)

    def trial(self, case):
        expected = audit.work(case,self.device,2); p=case['parameters']; blocks=expected['blocks']; checked=expected['correctness']['checked_elements']
        detail=[{'block_id':i,'smid':i%132,'start_ns':1000,'stop_ns':2000,'start_cycle':10,'stop_cycle':210} for i in range(blocks)]
        row={'schema_version':2,'type':'trial','case_id':case['id'],'iterations':8192,'seed':3,
                'threads':case['threads'],'blocks':blocks,'scope':case['scope'],'errors':0,
                'correctness':expected['correctness'],'work_unit':case['work_unit'],'work_count':expected['work'],
                'read_payload_bytes':0,'write_payload_bytes':0,'start_ns':1000,'stop_ns':2000,'event_ms':0.002,
                'warmup_samples_ns':[1000]*8,'warmup_converged':True,'blocks_detail':detail,
                'cache_residency_proven':False,'physical_hbm_bytes_proven':False,'registers_per_thread':64,
                'static_smem_bytes':case['threads']*8+16+(4096 if p['path']=='wgmma' else 0),'local_size_bytes':0,
                'occupancy_limit_ctas_per_sm':2,'timing_model':audit.TIMING_MODEL,'phase':'measure',
                'input_encoding':p['input_type'],'max_abs_error':0,'output_elements_checked':checked,
                'nonuniform_validation':{'iterations':[1,2],'checked_elements':checked*2,'input_seed':3,'errors':0,
                                         'reference_model':'low_precision_logical_integer_v1'}}
        if case['work_model']=='ptx_logical_mma_lowering_v1':
            row['compiled_demand_contract']=audit.compiled_demand(case)
        return row

    def test_work_and_full_output_record_acceptance_all_32_cases(self):
        for case in self.contract['cases']:
            result=audit.validate_trial(self.trial(case),case,self.device,3,self.protocol)
            self.assertEqual(result['unit'],case['metric']['unit'])
        self.assertEqual(audit.work(self.contract['cases'][0],self.device,2)['work'],2147483648)

    def test_raw_tampering_missing_outputs_and_occupancy_rejected(self):
        case=self.contract['cases'][0]
        for key,value in [('work_count',1),('work_unit','OP'),('output_elements_checked',1),
                          ('read_payload_bytes',32),('iterations',0),('input_encoding','e5m2'),
                          ('max_abs_error',1),('local_size_bytes',8),('occupancy_limit_ctas_per_sm',0),
                          ('static_smem_bytes',0),('errors',1)]:
            row=self.trial(case);row[key]=value
            with self.assertRaises(ValueError):audit.validate_trial(row,case,self.device,3,self.protocol)
        row=self.trial(case);row['nonuniform_validation']['checked_elements']-=1
        with self.assertRaises(ValueError):audit.validate_trial(row,case,self.device,3,self.protocol)
        row=self.trial(case);row['blocks_detail'][0]['stop_cycle']=10
        with self.assertRaises(ValueError):audit.validate_trial(row,case,self.device,3,self.protocol)

    def synthetic_sass(self, lowered=False):
        functions=[]
        for path,kind,groups in itertools.product(('mma','wgmma'),audit.KINDS,(1,2)):
            floating=kind in audit.KINDS[:2]
            if path=='wgmma':
                opcode=f'QGMMA.64x64x32.F32.{kind.upper()}.{kind.upper()}' if floating else f'IGMMA.64x64x32.{kind.upper()}.{kind.upper()}'
            else:opcode='HMMA.16816.F32' if floating else f'IMMA.16832.{kind.upper()}.{kind.upper()}'
            ops=['WARPGROUP.ARRIVE;', 'CS2R R0, SR_GLOBALTIMERLO;', 'CS2R R2, SR_CLOCKLO;', 'BAR.SYNC 0;']
            if path=='mma' and floating:
                ops += [f'F2FP.F16.{kind.upper()}.UNPACK_B R4, R5;' for _ in range(12)]
            # Strict original A expects 32 PTX products per loop: two half-K HMMA each.
            count=(2 if lowered else 64) if path=='mma' and floating else 32
            ops += [f'{opcode} R8, R4, R6, R8;' for _ in range(count)]
            if lowered and path=='mma' and floating:
                ops += ['FADD R8, R4, R8;' for _ in range(128)]
            if path=='wgmma':
                ops[-1]=ops[-1].replace(';',', gsb0;')
                ops+=['WARPGROUP.DEPBAR.LE gsb0, 0x0;']
            ops+=['@P0 BRA 0x40;']
            if path=='wgmma':ops+=['WARPGROUP.DEPBAR.LE gsb0, 0x0;']
            ops+=['BAR.SYNC 0;', 'CS2R R2, SR_CLOCKLO;', 'CS2R R0, SR_GLOBALTIMERLO;']
            functions.append(f'Function : lp_{path}_{kind}_g{groups}\n'+'\n'.join(f'/*{i*16:04x}*/ {op}' for i,op in enumerate(ops)))
        return '\n'.join(functions)

    def test_sass_scoped_loop_barrier_type_and_spill_rejection(self):
        sass=self.synthetic_sass();self.assertEqual(len(audit.audit_sass(sass,self.contract)),16)
        for damaged in [sass.replace('HMMA.16816.F32','NOP',1),
                        sass.replace('/*0030*/ BAR.SYNC 0;','/*0030*/ NOP;',1),
                        sass.replace('0x40;','0x3000;',1),
                        sass.replace('E4M3.E4M3','E4M3.E5M2',1),
                        sass.replace('S8.S8','S8.U8',1),
                        sass.replace('QGMMA.64x64x32.F32','QGMMA.64x64x32.F16',1),
                        sass.replace('F2FP.F16.E4M3.UNPACK_B','F2FP.F16.E5M2.UNPACK_B',1),
                        sass.replace('/*0040*/','/*0038*/ LDL R0, [R1];\n/*0040*/',1),
                        sass.replace('HMMA.16816','HMMA.16832',1),
                        sass.replace(', gsb0;',';',1)]:
            with self.assertRaises(ValueError):audit.audit_sass(damaged,self.contract)

    def test_type_parser_rejects_wrong_a_b_and_accumulator(self):
        good=[('QGMMA.64x64x32.F32.E4M3.E4M3 R8, gdesc[UR8], R8;', 'wgmma','e4m3'),
              ('IGMMA.64x64x32.S8.S8 R8, gdesc[UR8], R8;', 'wgmma','s8'),
              ('IMMA.16832.U8.U8 R8, R4, R6, R8;', 'mma','u8'),
              ('HMMA.16816.F32 R8, R4, R6, R8;', 'mma','e4m3')]
        for op,path,kind in good:audit.decode_compute_opcode(op,path,kind)
        for op,path,kind in [('QGMMA.64x64x32.F32.E4M3.E5M2 R8, R4, R8;', 'wgmma','e4m3'),
                             ('QGMMA.64x64x32.F16.E4M3.E4M3 R8, R4, R8;', 'wgmma','e4m3'),
                             ('IGMMA.64x64x32.S8.U8 R8, R4, R8;', 'wgmma','s8'),
                             ('IMMA.16832.S8.U8 R8, R4, R6, R8;', 'mma','u8'),
                             ('HMMA.16816.F16 R8, R4, R6, R8;', 'mma','e4m3')]:
            with self.assertRaises(ValueError):audit.decode_compute_opcode(op,path,kind)

    def test_lowering_revision_retains_logical_work_and_denies_native_export(self):
        audit.validate_contract(self.lowered)
        lowered=[c for c in self.lowered['cases'] if c['work_model']=='ptx_logical_mma_lowering_v1']
        self.assertEqual(len(lowered),8)
        for case in self.lowered['cases']:
            result=audit.validate_trial(self.trial(case),case,self.device,3,self.protocol)
            if case in lowered:
                self.assertFalse(case['exportable']);self.assertTrue(result['unit'].startswith('logical_'))
        for mutate in [lambda c:c['cases'][0].update(exportable=True),
                       lambda c:c['cases'][0].update(work_model='dense_lowprecision_collective_v1'),
                       lambda c:c['cases'][0]['metric'].update(unit='FLOP/clock64_cycle/CTA'),
                       lambda c:c['lowering_contract']['observed_sass_per_warp_iteration'].update(FADD=127)]:
            c=deepcopy(self.lowered);mutate(c)
            with self.assertRaises(ValueError):audit.validate_contract(c)
        case=lowered[0];row=self.trial(case);row['compiled_demand_contract']['instruction_counts']['FADD']=127
        with self.assertRaises(ValueError):audit.validate_trial(row,case,self.device,3,self.protocol)

    def test_lowering_exact_demand_and_old_contract_still_rejects(self):
        sass=self.synthetic_sass(lowered=True)
        self.assertEqual(len(audit.audit_sass(sass,self.lowered)),16)
        with self.assertRaisesRegex(ValueError,'shape/count'):
            audit.audit_sass(sass,self.contract)
        for damaged in [sass.replace('FADD R8','NOP R8',1),
                        sass.replace('F2FP.F16.E4M3.UNPACK_B','F2FP.F16.E5M2.UNPACK_B',1),
                        sass.replace('HMMA.16816.F32','HMMA.16816.F16',1),
                        sass.replace('HMMA.16816.F32','NOP',1)]:
            with self.assertRaises(ValueError):audit.audit_sass(damaged,self.lowered)


if __name__=='__main__':unittest.main()
