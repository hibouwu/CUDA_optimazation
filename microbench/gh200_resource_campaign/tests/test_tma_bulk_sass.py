"""Actual compile-only target instruction mutations; no GPU execution."""
import copy,json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from auditors.tma_bulk import audit_sass,_ops
CONTRACT=json.loads((ROOT/'contracts/tma_bulk.json').read_text())
SASS=ROOT.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation/target-compile-followup-b/diagnostics/tma_bulk/sass.txt'


def mutations(text):
    for part in re.split(r'Function\s*:\s*',text)[1:]:
        symbol=part.splitlines()[0].strip()
        if not symbol.startswith('tb_'):continue
        ops=_ops(part)
        patterns=[('bulk_direction',r'^UBLKCP\.'),('proxy_fence',r'^MEMBAR\.ALL\.CTA'),
                  ('CTA_gate',r'^BAR\.SYNC'),('global_timer',r'SR_GLOBALTIMERLO')]
        if '_g2s_' in symbol:
            patterns += [('expect_tx',r'SYNCS\.ARRIVE\.TRANS64\.RED\.A0TR'),('arrival_token',r'SYNCS\.ARRIVE\.TRANS64\.A1T0'),
                         ('acquire_wait',r'SYNCS\.PHASECHK'),('invalidate',r'SYNCS\.CCTL\.IV'),('fatal_timeout',r'BPT\.TRAP')]
        else:
            patterns += [('commit',r'^UTMACMDFLUSH'),('full_wait',r'^DEPBAR\.LE'),('full_visibility',r'^CCTL\.IVALL')]
        if '_release_' in symbol:patterns += [('source_overwrite',r'^STS'),('release_clock',r'SR_CLOCKLO')]
        for name,pattern in patterns:
            matches=[(pc,op) for pc,op in ops if re.search(pattern,op)]
            if not matches:raise AssertionError((symbol,name))
            pc,op=matches[-1] if name=='source_overwrite' else matches[0]
            # The mutation retains all textual opcodes but makes this operation dead.
            changed=re.sub(r'(/\*'+f'{pc:04x}'+r'\*/\s*)'+re.escape(op)+r'\s*;',lambda m:m[1]+'@!PT '+op+';',part,count=1)
            if changed==part:raise AssertionError((symbol,name,'mutation not applied'))
            yield symbol,name,text.replace(part,changed,1)
        pc,op=next((pc,op) for pc,op in ops if op.startswith('UBLKCP.'))
        yield symbol,'wrong_bulk_operand',text.replace(part,part.replace(op,op.rsplit(',',1)[0]+', URZ',1),1)


class TargetSass(unittest.TestCase):
    def test_actual_targets_and_dead_operation_negatives(self):
        text=SASS.read_text();rows=audit_sass(text,CONTRACT)
        self.assertEqual(len(rows),18)
        cases=list(mutations(text));self.assertEqual(len(cases),168)
        for symbol,name,bad in cases:
            with self.subTest(symbol=symbol,mutation=name),self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
    def test_missing_duplicate_and_metadata_only(self):
        text=SASS.read_text();parts=re.split(r'Function\s*:\s*',text)
        for bad in (text.replace('Function : tb_g2s_q1024','Function : absent',1),text+'\nFunction : '+parts[1]):
            with self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
        # Tool metadata is intentionally outside the instruction identity.
        self.assertEqual(len(audit_sass('unrelated build metadata\n'+text,CONTRACT)),18)

if __name__=='__main__':unittest.main()
