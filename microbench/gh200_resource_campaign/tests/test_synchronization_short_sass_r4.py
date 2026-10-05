"""Actual731079 comparison of39roles+2outlined-pair targets, no GPU execution."""
import json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from auditors.synchronization_short_sass_r4 import audit_sass,audit_resources,target_rows
from auditors.synchronization_short_baseline_r4 import BASELINE
BASE=ROOT.parents[1]/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation'
NEW=BASE/'target-compile-s11-short-a/diagnostics/synchronization_short_r4'
OLD=BASE/'target-compile-synchronization-a/diagnostics/synchronization'
PAIR=BASE/'target-compile-warp-divergent-a/diagnostics'
CONTRACT=json.loads((ROOT/'contracts/synchronization_short_r4.draft.json').read_text())

def mutations(text):
    parts=re.split(r'Function\s*:\s*',text)[1:]
    for symbol in sorted(BASELINE):
        part=next(p for p in parts if p.splitlines()[0].strip()==symbol);rows=target_rows(text,symbol)
        patterns=[('clock',r'SR_CLOCKLO'),('participant_output',r'\bSTG')]
        if 'sync_' in symbol:
            patterns += [('globaltimer',r'SR_GLOBALTIMERLO'),('CTA_gate',r'\bBAR\.SYNC')]
            if 'ILi2E' in symbol:patterns += [('arrive_token',r'SYNCS\.ARRIVE'),('acquire_wait',r'SYNCS\.PHASECHK'),('abort',r'ATOMS\.EXCH')]
            elif 'ILi1E' in symbol:patterns.append(('CTA_target',r'BAR\.SYNC\.DEFER_BLOCKING 0x1'))
            elif 'ILi3E' in symbol or 'ILi4E' in symbol or 'ILi5E' in symbol:patterns.append(('fence',r'MEMBAR'))
        else:
            patterns += [('shared_write',r'\bSTS'),('shared_read',r'\bLDS')]
            if symbol.startswith('warp'):patterns += [('outlined_collective',r'WARPSYNC\.COLLECTIVE'),('helper_dispatch',r'BRA\.DIV'),('warp_drain',r'WARPSYNC\.ALL')]
            else:patterns.append(('CTA_pair',r'BAR\.SYNC'))
        for label,pattern in patterns:
            candidates=[r for r in rows if re.search(pattern,r['instruction'])];assert candidates,(symbol,label)
            row=candidates[0];op=row['instruction'];pc=row['pc'];changed=re.sub(r'(/\*'+f'{pc:04x}'+r'\*/\s*)'+re.escape(op)+r'\s*;',lambda m:m[1]+'@!PT '+op+';',part,count=1);assert changed!=part
            yield symbol,label,text.replace(part,changed,1)
        match=re.search(r'/\* (0x[0-9a-f]{16}) \*/',part);changed=part[:match.start(1)]+'0x'+format(int(match[1],16)^1,'016x')+part[match.end(1):]
        yield symbol,'encoding_only',text.replace(part,changed,1)

class SyncActualTargets(unittest.TestCase):
 def test_all41_actual_code_and_resources_equal_originals(self):
    text=(NEW/'sass.txt').read_text();old=(OLD/'sass.txt').read_text();pair=(PAIR/'sass.txt').read_text()
    for symbol in BASELINE:self.assertEqual(target_rows(text,symbol),target_rows(pair if symbol.endswith('pair_v2') else old,symbol))
    self.assertEqual(len(audit_sass(text,CONTRACT)['targets']),41);self.assertEqual(len(audit_resources((NEW/'compile.stderr').read_text())),41)
 def test_actual_instruction_predicate_and128bit_negatives(self):
    text=(NEW/'sass.txt').read_text();cases=list(mutations(text));self.assertEqual(len(cases),266)
    for symbol,label,bad in cases:
        with self.subTest(symbol=symbol,label=label),self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
 def test_missing_duplicate_resource_metadata_and_outlined_helpers(self):
    text=(NEW/'sass.txt').read_text();first=next(iter(BASELINE));parts=re.split(r'Function\s*:\s*',text)
    for bad in (text.replace('Function : '+first,'Function : absent',1),text+'\nFunction : '+parts[1]):
        with self.assertRaises(ValueError):audit_sass(bad,CONTRACT)
    warp=target_rows(text,'warp_divergent_pair_v2');exit_pc=next(r['pc'] for r in warp if re.search(r'\bEXIT',r['instruction']));self.assertTrue(any(r['pc']>exit_pc and 'WARPSYNC.COLLECTIVE' in r['instruction'] for r in warp))
    resources=(NEW/'compile.stderr').read_text()
    for before,after in [('0 bytes stack frame','8 bytes stack frame'),('0 bytes spill stores','4 bytes spill stores'),('used 1 barriers','used 3 barriers'),('Used 20 registers','Used 21 registers')]:
        changed=resources.replace(before,after,1);self.assertNotEqual(changed,resources)
        with self.assertRaises(ValueError):audit_resources(changed)
    self.assertEqual(len(audit_sass('new tool heading\n'+text,CONTRACT)['targets']),41)

if __name__=='__main__':unittest.main()
