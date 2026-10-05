"""Actual 730679 instruction mutations; baseline needs independent review."""
import json,re,sys,unittest
from pathlib import Path
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from auditors.low_precision_bounded_fp8 import audit_sass
C=json.loads((B/'contracts/low_precision_bounded_fp8_v1.json').read_text())
SASS=B.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-bounded-fp8-a/diagnostics/low_precision_bounded_fp8_v1/sass.txt'

def mutations(text):
 for part in re.split(r'Function\s*:\s*',text)[1:]:
  symbol=part.splitlines()[0].strip()
  for name,pattern in [('compute',r'QGMMA\.[^;]+;'),('wait',r'WARPGROUP.DEPBAR.LE[^;]+;'),('proxy_fence',r'MEMBAR.ALL.CTA[^;]*;'),('CTA_gate',r'BAR.SYNC[^;]+;'),('counter_increment',r'UIADD3 UR4, UR4, 0x1, URZ ;'),('counter_bound',r'ISETP.LE.AND P0, PT, R0, UR4, PT ;'),('backedge',r'@!P0 BRA 0xe30 ;'),('initial_D0',r'FMUL R55,[^;]+;'),('shared_input',r'STS.U8[^;]+;'),('output',r'STG.E.64[^;]+;'),('drain',r'STS.64[^;]+;'),('globaltimer',r'CS2R[^;]+SR_GLOBALTIMERLO[^;]*;')]:
   matches=list(re.finditer(pattern,part));assert matches,(symbol,name)
   for n,m in enumerate(matches if name in ('compute','wait') else matches[:1]):
    changed=part[:m.start()]+'NOP;'+part[m.end():]
    yield symbol,name+str(n),text.replace(part,changed,1)
  m=re.search(r'QGMMA\.[^;]+;',part);changed=part[:m.start()]+'@!PT '+m[0]+part[m.end():]
  yield symbol,'never_executed_compute',text.replace(part,changed,1)
  yield symbol,'wrong_second_chain',text.replace(part,part.replace('R24, gdesc[UR8], R24','R56, gdesc[UR8], R56',1),1)

class Actual(unittest.TestCase):
 def test_four_targets_and_mutations(self):
  text=SASS.read_text();rows=audit_sass(text,C);self.assertEqual(len(rows),4)
  for row in rows:self.assertEqual(row['timed_loops'][0]['sass_compute_instructions'],32)
  cases=list(mutations(text));self.assertGreaterEqual(len(cases),180)
  for symbol,name,bad in cases:
   with self.subTest(symbol=symbol,mutation=name),self.assertRaises(ValueError):audit_sass(bad,C)
 def test_target_identity(self):
  text=SASS.read_text();part=re.split(r'Function\s*:\s*',text)[1]
  for bad in (text+'\nFunction : '+part,text.replace('_Z27lp_bounded_v1_wgmma_e4m3_g1ijbPN2gh5StampEPd','_Z27lp_bounded_v1_wgmma_e4m3_g1ijbPN2gh5StampEPd_fake')):
   with self.assertRaises(ValueError):audit_sass(bad,C)

if __name__=='__main__':unittest.main()
