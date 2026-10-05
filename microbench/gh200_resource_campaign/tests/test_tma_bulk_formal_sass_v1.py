"""Actual S14 host-inclusive compilation compared with original device targets.

No GPU kernel executes here. All18 functions must retain complete128-bit
instructions and actual ptxas resources, not merely mnemonic counts.
"""
import hashlib,json,re,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from common.family_b3 import target_identity,compile_resources
from auditors.tma_bulk import audit_sass
from test_tma_bulk_sass import mutations
REPO=ROOT.parents[1]
BASE=REPO/'results/gh200_resource_campaign/20261001-resource-suite-v2/implementation'
OLD=BASE/'target-compile-followup-b/diagnostics/tma_bulk'
NEW=BASE/'target-compile-s14-formal-host-a/diagnostics/tma_bulk_formal_v1'
CONTRACT=json.loads((ROOT/'contracts/tma_bulk_formal_host_v1.draft.json').read_text())
SYMBOLS={f'tb_{mode}_q{q}' for q in (1024,4096,8192,16384,32768,65536) for mode in ('g2s','s2g','release')}

def exact_targets(text,compile_text):
    expected=target_identity((OLD/'sass.txt').read_text(),SYMBOLS)
    if target_identity(text,SYMBOLS)!=expected:raise ValueError('complete18 original device machine-code identity differs')
    if compile_resources(compile_text,SYMBOLS)!=compile_resources((OLD/'compile.stderr').read_text(),SYMBOLS):raise ValueError('actual target resource identity differs')
    return audit_sass(text,CONTRACT)

class FormalTargetSass(unittest.TestCase):
 def test_actual18_targets_resources_and_source_receipts(self):
    text=(NEW/'sass.txt').read_text();compile_text=(NEW/'compile.stderr').read_text()
    self.assertEqual(len(exact_targets(text,compile_text)),18)
    for kind in ('compile','sass'):
        receipt=json.loads((NEW/(kind+'.receipt.json')).read_text());self.assertEqual(receipt['returncode'],0)
        self.assertTrue(receipt['cleanup_confirmed']);self.assertFalse(receipt['timed_out'])
        stdout=NEW/('compile.stdout' if kind=='compile' else 'sass.txt')
        stderr=NEW/(kind+'.stderr')
        self.assertEqual(receipt['stdout_sha256'],hashlib.sha256(stdout.read_bytes()).hexdigest())
        self.assertEqual(receipt['stderr_sha256'],hashlib.sha256(stderr.read_bytes()).hexdigest())
    summary=json.loads((NEW.parent/'compile_summary.json').read_text())[0]
    self.assertEqual(summary['source_sha256'],hashlib.sha256((ROOT/'probes/tma_bulk_formal_v1.cu').read_bytes()).hexdigest())
    self.assertEqual(summary['binary_sha256'],hashlib.sha256((NEW/'probe').read_bytes()).hexdigest())
    self.assertEqual(summary['sass_sha256'],hashlib.sha256((NEW/'sass.txt').read_bytes()).hexdigest())
 def test_all168_actual_semantic_negatives_and18_encoding_only_changes(self):
    text=(NEW/'sass.txt').read_text();compile_text=(NEW/'compile.stderr').read_text()
    candidates=list(mutations(text));self.assertEqual(len(candidates),168)
    for symbol,label,bad in candidates:
        with self.subTest(symbol=symbol,label=label),self.assertRaises(ValueError):exact_targets(bad,compile_text)
    for part in re.split(r'Function\s*:\s*',text)[1:]:
        symbol=part.splitlines()[0].strip()
        if symbol not in SYMBOLS:continue
        match=re.search(r'/\* (0x[0-9a-f]{16}) \*/',part)
        bad_part=part[:match.start(1)]+'0x'+format(int(match[1],16)^1,'016x')+part[match.end(1):]
        with self.subTest(symbol=symbol,label='encoding_only'),self.assertRaises(ValueError):exact_targets(text.replace(part,bad_part,1),compile_text)
 def test_actual_resource_and_target_set_drift(self):
    text=(NEW/'sass.txt').read_text();compile_text=(NEW/'compile.stderr').read_text()
    for old,new in [('0 bytes stack frame','8 bytes stack frame'),('0 bytes spill stores','4 bytes spill stores'),('used 1 barriers','used 2 barriers'),('Used 32 registers','Used 33 registers')]:
        changed=compile_text.replace(old,new)
        if changed==compile_text:
            if old.startswith('Used '):
                match=re.search(r'Used (\d+) registers',compile_text)
                changed=compile_text[:match.start(1)]+str(int(match[1])+1)+compile_text[match.end(1):]
            else:self.fail('resource mutation not applied: '+old)
        with self.subTest(resource=old),self.assertRaises(ValueError):exact_targets(text,changed)
    for wrong in (text.replace('Function : tb_g2s_q1024','Function : missing',1),text+'\nFunction : '+re.split(r'Function\s*:\s*',text)[1]):
        with self.assertRaises(ValueError):exact_targets(wrong,compile_text)
    self.assertEqual(len(exact_targets('different tool heading\n'+text,compile_text)),18)

if __name__=='__main__':unittest.main()
