"""Freeze one 16-block diagnostic using the already checked binaries."""
from pathlib import Path
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
from analyze import CONTRASTS

root=Path(__file__).resolve().parent
prior=root.parent/'20261011-R13-early-wait-sum-correctness/attempts/remaining3'
original=root.parent/'20261011-R13-early-wait-sum-correctness'
manifest=json.loads((prior/'manifest.json').read_text())
arms=json.loads((original/'manifest.json').read_text())['samples']
assert [(x['stage'],x['flag']) for x in arms]==[(4,0),(4,1),(6,0),(6,1)]
assert not (root/'freeze.json').exists(), 'do not overwrite a frozen diagnostic'
for dirname in ('analysis_snapshot','build'):
    shutil.copytree(prior/dirname,root/dirname,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('run.sh','validate_allocation.py','submit.py'):
    shutil.copyfile(prior/name,root/name)
collector=(prior/'collect.py').read_text()
collector=collector.replace("Path('raw').mkdir();records=[]", "Path('raw').mkdir();records=[];observed_sm=None;decoded=[]")
collector=collector.replace("    Path('raw',sample['id']+'.decoded.json')", "    if observed_sm is None:observed_sm=result['sm']\n    if result['sm']!=observed_sm:raise ValueError('SM mapping changed; retain and stop')\n    decoded.append(result)\n    Path('raw',sample['id']+'.decoded.json')")
collector=collector.replace("Path('complete.json').write_text", "from analyze import block_contrasts,summarize\nblocks=[block_contrasts(decoded[i:i+4]) for i in range(0,len(decoded),4)]\nPath('analysis.json').write_text(json.dumps(summarize(blocks),indent=2)+'\\n')\nPath('complete.json').write_text")
(root/'collect.py').write_text(collector)
rng=random.Random(20261011)
samples=[]
for block in range(16):
    order=list(range(4));rng.shuffle(order)
    for position,index in enumerate(order):
        sample=copy.deepcopy(arms[index])
        sample.update(id=f'b{block:02d}-p{position}-s{sample["stage"]}-f{sample["flag"]}',block=block,position=position)
        samples.append(sample)
for key in ('prior_attempt',):manifest.pop(key)
manifest.update(purpose='Fixed randomized-block mechanism diagnostic; no observation or performance acceptance',
    samples=samples,blocks=16,randomization=dict(seed=20261011,method='independent uniform permutation of four arms in each block'),
    contrasts=list(CONTRASTS),inference=dict(target='medians of jointly formed within-block contrasts',
        simultaneous_confidence=0.95,method='exact binomial ranks 3 and 14 of 16, Bonferroni for eight contrasts',
        assumption='IID block contrast vectors; random ordering alone does not establish this',
        interpretation='conditional intervals plus complete empirical block data; no stopping for significance'),
    stopping='Exactly 64 targets; retain and stop at any failed check or CTA0 SM change. No replacement, sample-size extension or partial scoring.',
    performance_decision='No equivalence tolerance or observation qualification. Off/on D can diagnose total-window changes but cannot identify uninstrumented Q.',
    scope_preserved='cfg_a/b/c full objective unchanged; this fixed cfg_b protocol chooses the next mechanism to investigate.')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
command=json.loads((prior/'command.json').read_text())
command=[x.replace('remaining3','diagnostic').replace('--time=00:10:00','--time=00:20:00') for x in command]
(root/'command.json').write_text(json.dumps(command,indent=2)+'\n')
test=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis.py'],cwd=root,capture_output=True,text=True)
(root/'cpu-tests.json').write_text(json.dumps(dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr),indent=2)+'\n')
assert test.returncode==0
names=['manifest.json','command.json','collect.py','run.sh','validate_allocation.py','analyze.py']
names += ['analysis_snapshot/'+name for name in manifest['analysis_sha256']]
names += ['build/'+name for name in manifest['binary_sha256']]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in manifest['binary_sha256'].items():assert sha(root/'build'/name)==digest
for name,digest in manifest['analysis_sha256'].items():assert sha(root/'analysis_snapshot'/name)==digest
freeze=dict(files={name:sha(root/name) for name in names})
(root/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
with tarfile.open(root/'input.tar.gz','w:gz') as archive:
    for name in names+['freeze.json']:archive.add(root/name,arcname=name)
(root/'input-receipt.json').write_text(json.dumps(dict(sha256=sha(root/'input.tar.gz'),bytes=(root/'input.tar.gz').stat().st_size),indent=2)+'\n')
print(json.dumps(dict(targets=len(samples),manifest_sha256=sha(root/'manifest.json'),freeze_sha256=sha(root/'freeze.json')),indent=2))
