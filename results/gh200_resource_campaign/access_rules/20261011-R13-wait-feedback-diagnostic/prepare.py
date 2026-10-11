"""Freeze the six-arm feedback diagnostic only after all eight correctness targets verify."""
from pathlib import Path
import copy
import hashlib
import json
import random
import shutil
import subprocess
import sys
import tarfile
from analyze import SERIES

root=Path(__file__).resolve().parent
correctness=root.parent/'20261011-R13-wait-feedback-correctness'
prior=correctness/'attempts/executable-bit-fix'
verified=json.loads((correctness/'verification.json').read_text())
assert verified['total_original_targets']==8 and verified['mode_format_and_sampled_correctness_passed']
assert {(r['stages'],r['observer']) for r in verified['records']}=={(s,m) for s in (4,6) for m in range(4)}
assert not (root/'freeze.json').exists()
manifest=json.loads((prior/'manifest.json').read_text())
for dirname in ('analysis_snapshot','build'):
    shutil.copytree(prior/dirname,root/dirname,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('run.sh','validate_allocation.py','submit.py'):shutil.copy2(prior/name,root/name)
collector=(prior/'collect.py').read_text()
collector=collector.replace("Path('raw').mkdir();records=[]","Path('raw').mkdir();records=[];decoded=[];observed_sm=None")
collector=collector.replace("    Path('raw',sample['id']+'.decoded.json')",
    "    if observed_sm is None:observed_sm=result['sm']\n    if result['sm']!=observed_sm:raise ValueError('SM mapping changed')\n    decoded.append(result)\n    Path('raw',sample['id']+'.decoded.json')")
collector=collector.replace("Path('complete.json').write_text",
    "from analyze import summarize\nPath('analysis.json').write_text(json.dumps(summarize(decoded),indent=2)+'\\n')\nPath('complete.json').write_text")
(root/'collect.py').write_text(collector)
arms=[sample for sample in manifest['samples'] if sample['flag'] in (1,2,3)]
assert [(x['stage'],x['flag']) for x in arms]==[(s,m) for s in (4,6) for m in (1,2,3)]
rng=random.Random(20261011);samples=[]
for block in range(16):
    order=list(range(6));rng.shuffle(order)
    for position,index in enumerate(order):
        sample=copy.deepcopy(arms[index])
        sample.update(id=f'b{block:02d}-p{position}-s{sample["stage"]}-m{sample["flag"]}',block=block,position=position)
        samples.append(sample)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest.pop('prior_attempt')
manifest.update(purpose='Six predeclared contrasts of ALL15 vs single-group observation, using one binary per stage',
    samples=samples,blocks=16,randomization=dict(seed=20261011,method='independent uniform permutation of six arms within each block'),
    correctness_report_sha256=sha(correctness/'verification.json'),correctness_archive_sha256=verified['archive_sha256'],
    contrasts=list(SERIES),inference=dict(target='separate medians of six jointly formed block contrasts',
        simultaneous_confidence=0.95,method='exact binomial rank3..14 of 16, Bonferroni for six predeclared quantities',
        assumption='IID block contrast vectors, not established by randomized order'),
    stopping='Exactly 96 targets, retain and stop on any numerical/format/SM check failure; no replacement, extension or partial scoring.',
    performance_decision='Instrumented bundle feedback only; no equivalence, no uninstrumented Q, no service price or model acceptance.')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
command=[x.replace('r13-wait-feedback-correctness-execfix','r13-wait-feedback-diagnostic').replace('--time=00:10:00','--time=00:20:00')
    for x in json.loads((prior/'command.json').read_text())]
(root/'command.json').write_text(json.dumps(command,indent=2)+'\n')
test=subprocess.run([sys.executable,'-m','unittest','-v','test_analysis.py'],cwd=root,capture_output=True,text=True)
(root/'cpu-tests.json').write_text(json.dumps(dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr),indent=2)+'\n')
assert test.returncode==0
names=['manifest.json','command.json','collect.py','run.sh','validate_allocation.py','analyze.py']
names += ['analysis_snapshot/'+name for name in manifest['analysis_sha256']]
names += ['build/'+name for name in manifest['binary_sha256']]
executables={}
for name,digest in manifest['binary_sha256'].items():
    assert sha(root/'build'/name)==digest
    assert (root/'build'/name).stat().st_mode & 0o777 == 0o755
    executables['build/'+name]=0o755
freeze=dict(files={name:sha(root/name) for name in names},executables=executables)
(root/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
with tarfile.open(root/'input.tar.gz','w:gz') as archive:
    for name in names+['freeze.json']:archive.add(root/name,arcname=name)
with tarfile.open(root/'input.tar.gz') as archive:
    for name,mode in executables.items():assert archive.getmember(name).mode & 0o777 == mode
(root/'input-receipt.json').write_text(json.dumps(dict(sha256=sha(root/'input.tar.gz'),bytes=(root/'input.tar.gz').stat().st_size),indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(root/'manifest.json'),freeze_sha256=sha(root/'freeze.json'),targets=len(samples)),indent=2))
