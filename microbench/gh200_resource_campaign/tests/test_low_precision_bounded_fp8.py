"""Independent exact oracle, actual byte encoder, unchanged measured-body checks."""
import hashlib,json,struct,subprocess,sys,tempfile,unittest
from fractions import Fraction
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from auditors import low_precision_bounded_fp8 as audit
SOURCE=(B/'probes/low_precision_bounded_fp8_v1.cu').read_text()
CONTRACT=json.loads((B/'contracts/low_precision_bounded_fp8_v1.json').read_text())

class Bounded(unittest.TestCase):
 def run_cpp(self,body):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory);(p/'x.cpp').write_text(body)
   subprocess.run(['c++','-std=c++17','-O2','-I',str(B/'common'),str(p/'x.cpp'),'-o',str(p/'x')],check=True,capture_output=True)
   return subprocess.run([str(p/'x')],check=True,capture_output=True).stdout
 def test_actual_encoder_and_all_reference_words(self):
  encoder=SOURCE[SOURCE.index('template<int Kind>'):SOURCE.index('struct LowCase')].replace('__device__','').replace('__forceinline__','inline')
  body='#include <iostream>\n#include "low_precision_bounded_fp8_reference_v1.hpp"\n'+encoder+r'''
int main(){
 for(int kind=0;kind<2;++kind)for(unsigned seed:{0u,3u,4294967295u})for(int non=0;non<2;++non)
 for(unsigned outer=0;outer<64;++outer)for(unsigned k=0;k<32;++k)for(bool a:{false,true}) {
  unsigned char value=kind==0?lp_operand<0>(outer,k,seed,a,non):lp_operand<1>(outer,k,seed,a,non);
  std::cout.write(reinterpret_cast<char*>(&value),1);
 }
 for(int kind=0;kind<2;++kind)for(int groups=1;groups<=2;++groups)for(int non=0;non<2;++non)
 for(int length:{1,2,8192}) {if(non&&length>2)continue;
  auto values=low_precision_bounded_fp8_reference_v1::outputs(kind,true,groups,length,3,non);
  std::cout.write(reinterpret_cast<const char*>(values.data()),values.size()*8);
 }
}
'''
  raw=self.run_cpp(body);at=0
  for kind,mant,bias in ((0,3,7),(1,2,15)):
   for seed in (0,3,4294967295):
    for non in (0,1):
     for outer in range(64):
      for k in range(32):
       for a in (False,True):
        bits=raw[at];at+=1;exp=(bits&127)>>mant;frac=bits&((1<<mant)-1)
        value=Fraction(0) if exp==frac==0 else Fraction((1<<mant)+frac,1<<mant)*Fraction(2)**(exp-bias)
        if bits&128:value=-value
        expected=Fraction((outer+(2 if a else 3)*k+seed)%7-3,8) if non else Fraction(-1 if not a and k%2 else 1,16)
        self.assertEqual(value,expected)
  for kind in range(2):
   for groups in (1,2):
    for non in (0,1):
     for length in (1,2,8192):
      if non and length>2:continue
      for t in range(groups*128):
       g=t//128;local=t%128;lane=t%32
       for chain in range(2):
        for f in range(32):
         m=16*(local//32)+lane//4+8*((f//2)%2);n=2*(lane%4)+f%2+8*(f//4)
         dot=sum(((m+2*k+3)%7-3)*((n+3*k+3)%7-3) for k in range(32)) if non else 0
         expected=Fraction(1+g+chain,8)+Fraction(dot*16*length,64) if non else Fraction(1+2*g+chain,8)
         got=struct.unpack_from('d',raw,at)[0];at+=8;self.assertEqual(Fraction(got),expected)
  self.assertEqual(at,len(raw))
 def test_measured_bodies_and_regeneration(self):
  old=(B/'probes/low_precision.cu').read_text();before=hashlib.sha256(SOURCE.encode()).hexdigest()
  subprocess.run([sys.executable,str(B/'probes/generate_low_precision_bounded_fp8_v1.py')],check=True)
  self.assertEqual(before,hashlib.sha256((B/'probes/low_precision_bounded_fp8_v1.cu').read_bytes()).hexdigest())
  for kind in ('e4m3','e5m2'):
   for g in (1,2):
    bodies=[]
    for text,symbol in [(old,f'lp_wgmma_{kind}_g{g}'),(SOURCE,f'lp_bounded_v1_wgmma_{kind}_g{g}')]:
     start=text.index('__global__ void '+symbol+'(');body=text[start:text.index('\n}\n',start)];bodies.append(body[body.index('  __syncthreads();'):])
    self.assertEqual(*bodies)
    self.assertIn('for(int q=0;q<16;++q)',bodies[0]);self.assertIn('for(int c=0;c<2;++c)',bodies[0]);self.assertEqual(bodies[0].count('wgmma.wait_group.sync.aligned 0;'),2)
 def test_contract_work_and_blind_spots(self):
  import copy
  audit.validate_contract(CONTRACT)
  from common.suite_snapshot import required_reviews
  reviews=required_reviews(CONTRACT,preflight=True,case_diagnostic=True)
  self.assertIn(('S08','A','reviews/S08-A-review.json'),reviews)
  self.assertIn(('S08','bounded-fp8-A','reviews/S08-bounded-fp8-A-review.json'),reviews)
  for c in CONTRACT['cases']:
   plan=audit.work(c,{'sms':132},2);self.assertEqual(plan['work'],plan['blocks']*c['parameters']['groups']*32*8192*262144)
   for key,value in [('groups',3),('chains',1),('wait',1),('uniform_input_value',0.0625)]:
    bad=copy.deepcopy(c);bad['parameters'][key]=value
    with self.assertRaises(ValueError):audit.case_identity(bad)
  self.assertEqual(sum(1 if k%2==0 else -1 for k in range(32)),0)
  self.assertEqual(Fraction(1,8)+8192*16*0,Fraction(1,8)) # omitted loop indistinguishable
  self.assertNotEqual(Fraction(1,8)+8192*16*Fraction(2,256),Fraction(1,8))
  self.assertIn('std::signbit(got[i])!=std::signbit(expected[i%per_cta])',SOURCE)

 def test_formal_trial_conditions_and_mutations(self):
  import copy
  device={'sms':132,'smem_per_sm_bytes':233472,'smem_per_cta_optin_bytes':232448,'registers_per_sm':65536}
  protocol=json.loads((B/'contracts/protocol.json').read_text())
  for case in CONTRACT['cases']:
   occ=2 if case['parameters']['groups']==1 else 1;plan=audit.work(case,device,occ);blocks=plan['blocks'];p=case['parameters'];positive=24 if p['input_type']=='e4m3' else 44
   row={'schema_version':2,'type':'trial','case_id':case['id'],'iterations':8192,'seed':1031,'threads':case['threads'],'blocks':blocks,'scope':case['scope'],'errors':0,'correctness':plan['correctness'],'work_unit':'FLOP','work_count':plan['work'],'read_payload_bytes':0,'write_payload_bytes':0,'start_ns':1000,'stop_ns':2000,'event_ms':0.002,'warmup_samples_ns':[1000]*8,'warmup_converged':True,'blocks_detail':[{'block_id':i,'smid':i%132,'start_ns':1000,'stop_ns':2000,'start_cycle':10,'stop_cycle':210} for i in range(blocks)],'cache_residency_proven':False,'physical_hbm_bytes_proven':False,'registers_per_thread':148,'static_smem_bytes':4096+case['threads']*8+16,'local_size_bytes':0,'occupancy_limit_ctas_per_sm':occ,'timing_model':'low_precision_start_gate_result_drain_v1','phase':'measure','input_encoding':p['input_type'],'max_abs_error':0,'output_elements_checked':plan['correctness']['checked_elements'],'input_profile_id':'nonzero_K_alternating_cancellation_v1','dynamic_counter_verified':False,'D0_rule':'(1+2*group+chain)/8','nominal_WGMMA_per_group_iteration':32,'bounded_operand_bits':{'A_positive':positive,'B_even':positive,'B_odd':positive+128},'operand_source_form':'SS','groups_per_CTA':p['groups'],'chains':2,'batch':16,'wait':0,'shape':[64,64,32]}
   audit.validate_trial(row,case,device,1031,protocol)
   for key,value in [('seed',3),('max_abs_error',False),('D0_rule','(1+group+chain)/8'),('dynamic_counter_verified',True),('work_count',1),('nominal_WGMMA_per_group_iteration',31),('bounded_operand_bits',{}),('output_elements_checked',1),('registers_per_thread',1),('static_smem_bytes',9999),('occupancy_limit_ctas_per_sm',32),('local_size_bytes',4),('errors',1),('groups_per_CTA',True)]:
    bad=copy.deepcopy(row);bad[key]=value
    with self.assertRaises(ValueError):audit.validate_trial(bad,case,device,1031,protocol)

if __name__=='__main__':unittest.main()
