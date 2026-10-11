import subprocess,json,hashlib,shlex,tarfile
from pathlib import Path
r=Path(__file__).resolve().parent
command=json.loads((r/'command.json').read_text())
with (r/'input.tar.gz').open('rb') as src,(r/'output.tar.gz').open('wb') as out,(r/'slurm.log').open('wb') as log:
 result=subprocess.run(['ssh','-o','BatchMode=yes','romeo',shlex.join(command)],stdin=src,stdout=out,stderr=log)
output=r/'output.tar.gz'
receipt=dict(returncode=result.returncode,bytes=output.stat().st_size,sha256=hashlib.sha256(output.read_bytes()).hexdigest())
(r/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
if output.stat().st_size:
 with tarfile.open(output) as archive:archive.extractall(r/'returned',filter='data')
raise SystemExit(result.returncode)
