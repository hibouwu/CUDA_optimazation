"""Preserve the pre-execution failure and prepare all eight unexecuted targets."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
failed_sha='2137a5ab121b622f05bc53c9e84cdb54c7acffe186ee9c289415abf5ce7bdf38'
assert sha(root/'output.tar.gz')==failed_sha
with tarfile.open(root/'output.tar.gz') as archive:
    assert './records.json' not in archive.getnames()
    assert './setup.stdout' not in archive.getnames()
    assert b'PermissionError: [Errno 13]' in archive.extractfile('./run.log').read()
    assert int(archive.extractfile('./exit_code.txt').read())==1
new=root/'attempts/executable-bit-fix';new.mkdir(parents=True)
for dirname in ('build','analysis_snapshot'):
    shutil.copytree(root/dirname,new/dirname,ignore=shutil.ignore_patterns('__pycache__'))
for name in ('run.sh','validate_allocation.py','submit.py'):
    shutil.copy2(root/name,new/name)
manifest=json.loads((root/'manifest.json').read_text())
manifest['prior_attempt']=dict(output_archive_sha256=failed_sha,manifest_sha256=sha(root/'manifest.json'),
    status='failed before setup process launch: missing executable file permission',executed_targets=0,retaken_targets=0)
manifest['purpose']='All eight original unexecuted correctness targets after executable permission repair; binary bytes unchanged'
(new/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
code=(root/'collect.py').read_text()
anchor="if not os.environ.get('CUDA_VISIBLE_DEVICES')"
assert code.count(anchor)==1
code=code.replace(anchor,"for name,mode in json.loads(Path('freeze.json').read_text())['executables'].items():\n    if Path(name).stat().st_mode & 0o777 != mode:raise ValueError('binary executable metadata mismatch')\n"+anchor)
(new/'collect.py').write_text(code)
command=[x.replace('r13-wait-feedback-correctness','r13-wait-feedback-correctness-execfix')
         for x in json.loads((root/'command.json').read_text())]
(new/'command.json').write_text(json.dumps(command,indent=2)+'\n')
executables={}
for name,digest in manifest['binary_sha256'].items():
    path=new/'build'/name
    assert sha(path)==digest
    path.chmod(0o755);executables['build/'+name]=0o755
old_freeze=json.loads((root/'freeze.json').read_text())
freeze=dict(files={name:sha(new/name) for name in old_freeze['files']},executables=executables)
(new/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
with tarfile.open(new/'input.tar.gz','w:gz') as archive:
    for name in list(freeze['files'])+['freeze.json']:archive.add(new/name,arcname=name)
with tarfile.open(new/'input.tar.gz') as archive:
    for name,mode in executables.items():assert archive.getmember(name).mode & 0o777 == mode
(new/'input-receipt.json').write_text(json.dumps(dict(sha256=sha(new/'input.tar.gz'),bytes=(new/'input.tar.gz').stat().st_size),indent=2)+'\n')
print(json.dumps(dict(manifest_sha256=sha(new/'manifest.json'),freeze_sha256=sha(new/'freeze.json'),original_targets=8,executed_before_repair=0),indent=2))
