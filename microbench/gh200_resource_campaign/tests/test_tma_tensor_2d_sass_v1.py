"""S15 actual731028 target proof and real instruction/encoding/resource negatives."""
import json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auditors.tma_tensor_2d_sass_v1 import audit_sass,target_rows,audit_compile_resources
from auditors.tma_tensor_2d_sass_baseline_v1 import BASELINE
CONTRACT=json.loads((ROOT/'contracts/tma_tensor_2d_short_v1.draft.json').read_text())
BUILD=ROOT.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-s15-short-a/diagnostics/tma_tensor_2d'

def mutations(text):
    for symbol in sorted(BASELINE):
        rows=target_rows(text,symbol);g2s='_g2s_' in symbol
        patterns=[('tensor_request',r'^UTMA(?:LDG|STG)\.2D'),('shared_proxy',r'^MEMBAR\.ALL\.CTA'),('CTA_gate',r'^BAR\.SYNC'),('timestamp',r'SR_GLOBALTIMERLO'),('uint16_consumer',r'\bLDS(?:\.|\s)')]
        if g2s:patterns += [('expectQ',r'SYNCS\.ARRIVE\.TRANS64\.RED\.A0TR'),('arrival_token',r'SYNCS\.ARRIVE\.TRANS64\.A1T0'),('acquire',r'SYNCS\.PHASECHK'),('invalidate',r'SYNCS\.CCTL\.IV'),('timeout_trap',r'BPT\.TRAP')]
        else:patterns += [('commit',r'^UTMACMDFLUSH'),('fullwait',r'^DEPBAR\.LE'),('full_visibility',r'^CCTL\.IVALL'),('global_reset_proxy',r'^FENCE\.VIEW\.ASYNC\.G'),('global_consumer_proxy',r'^FENCE\.VIEW\.ASYNC\.G')]
        if symbol.endswith('_sw128'):patterns.append(('swizzle_logic',r'\bLOP3'))
        for label,pattern in patterns:
            candidates=[r for r in rows if re.search(pattern,r['instruction'])];assert candidates,(symbol,label)
            row=candidates[-1] if label=='global_consumer_proxy' else candidates[0];pc=row['pc'];op=row['instruction']
            parts=re.split(r'Function\s*:\s*',text);part=next(part for part in parts[1:] if part.splitlines()[0].strip()==symbol)
            changed=re.sub(r'(/\*'+f'{pc:04x}'+r'\*/\s*)'+re.escape(op)+r'\s*;',lambda m:m[1]+'@!PT '+op+';',part,count=1);assert changed!=part
            yield symbol,label,text.replace(part,changed,1)
        parts=re.split(r'Function\s*:\s*',text);part=next(part for part in parts[1:] if part.splitlines()[0].strip()==symbol)
        instruction=next(r for r in rows if re.match(r'UTMA(?:LDG|STG)\.2D',r['instruction']))['instruction']
        yield symbol,'tensor_wrong_operand',text.replace(part,part.replace(instruction,instruction.replace('[UR','[URZ+UR',1),1),1)
        match=re.search(r'/\* (0x[0-9a-f]{16}) \*/',part);changed=part[:match.start(1)]+'0x'+format(int(match[1],16)^1,'016x')+part[match.end(1):]
        yield symbol,'encoding_only',text.replace(part,changed,1)

class TensorTarget(unittest.TestCase):
 def test_actual22_targets_resources_and_code_only_not_GPU(self):
    rows=audit_sass((BUILD/'sass.txt').read_text(),CONTRACT);self.assertEqual(len(rows),22)
    self.assertEqual(len(audit_compile_resources((BUILD/'compile.stderr').read_text())),22)
    self.assertTrue(all(r['GPU_qualification'] is False for r in rows))
 def test_actual_instruction_predicate_operand_encoding_negatives(self):
    text=(BUILD/'sass.txt').read_text();cases=list(mutations(text));self.assertEqual(len(cases),274)
    for symbol,label,bad in cases:
        with self.subTest(symbol=symbol,label=label),self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
 def test_missing_duplicate_target_resource_and_metadata_boundary(self):
    text=(BUILD/'sass.txt').read_text()
    for bad in (text.replace('Function : tt_g2s_q1024_none','Function : absent',1),text+'\nFunction : '+re.split(r'Function\s*:\s*',text)[1]):
        with self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
    self.assertEqual(len(audit_sass('different compiler title\n'+text,CONTRACT)),22)
    resources=(BUILD/'compile.stderr').read_text()
    for old,new in [('Used 40 registers','Used 41 registers'),('used 1 barriers','used 2 barriers'),('0 bytes stack frame','8 bytes stack frame'),('0 bytes spill loads','4 bytes spill loads')]:
        with self.subTest(resource=old),self.assertRaises(ValueError):audit_compile_resources(resources.replace(old,new,1))

if __name__=='__main__':unittest.main()
