"""Freeze one target per stage/mode, using the successful CPU build."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import tarfile

root=Path(__file__).resolve().parent
old=root.parent/'20261011-R13-early-wait-sum-correctness/attempts/remaining3'
build=root.parent/'20261011-R13-wait-feedback-build'
audit=json.loads((build/'build-audit.json').read_text())
manifest=json.loads((old/'manifest.json').read_text())
assert not (root/'freeze.json').exists()
(root/'build').mkdir();(root/'analysis_snapshot').mkdir()
for record in audit['binaries']:
    shutil.copy2(build/'attempts/prepare2/returned/build'/record['name'],root/'build'/record['name'])
modules=Path('microbench/gh200_resource_campaign/access_rules')
for name in ('decode_r13_wait_feedback.py','analyze_r18.py','v06_model.py','v06_run.py'):
    shutil.copyfile(modules/name,root/'analysis_snapshot'/name)
for name in ('run.sh','validate_allocation.py','submit.py'):shutil.copyfile(old/name,root/name)
(root/'collect.py').write_text((old/'collect.py').read_text().replace('from decode_r13_wait_sum import decode','from decode_r13_wait_feedback import decode'))
samples=[]
for stage in (4,6):
    template=next(x for x in manifest['samples'] if x['stage']==stage)
    for mode in range(4):
        sample=copy.deepcopy(template);sample.update(id=f'stage{stage}_mode{mode}',flag=mode)
        sample['command'][sample['command'].index('--wait-sum-observer')+1]=str(mode)
        samples.append(sample)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest.pop('prior_attempt')
manifest.update(purpose='Eight original correctness targets for the new feedback protocol; no effect estimate',
    samples=samples,binary_sha256={r['name']:r['binary_sha256'] for r in audit['binaries']},
    source_manifest_sha256=audit['source_manifest_sha256'],build_audit_sha256=sha(build/'build-audit.json'),
    analysis_sha256={p.name:sha(p) for p in sorted((root/'analysis_snapshot').iterdir())},
    trace_requirement='Mode-specific mask/count, group2/group16 interval containment; full early Q/R only in ALL15; all other missing components stay None.',
    stopping='Exactly eight declared targets, one per stage/mode, no replacements; stop and retain on any failure.',
    performance_decision='No performance, feedback or perturbation conclusion from these single-call correctness targets.')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
command=[x.replace('r13-early-wait-sum-remaining3','r13-wait-feedback-correctness') for x in json.loads((old/'command.json').read_text())]
(root/'command.json').write_text(json.dumps(command,indent=2)+'\n')
names=['manifest.json','collect.py','run.sh','validate_allocation.py','command.json']
names += ['build/'+name for name in manifest['binary_sha256']]
names += ['analysis_snapshot/'+name for name in manifest['analysis_sha256']]
(root/'freeze.json').write_text(json.dumps(dict(files={name:sha(root/name) for name in names}),indent=2)+'\n')
with tarfile.open(root/'input.tar.gz','w:gz') as archive:
    for name in names+['freeze.json']:archive.add(root/name,arcname=name)
(root/'input-receipt.json').write_text(json.dumps(dict(sha256=sha(root/'input.tar.gz'),bytes=(root/'input.tar.gz').stat().st_size),indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(root/'manifest.json'),freeze_sha256=sha(root/'freeze.json')),indent=2))
