import ctypes,gzip,hashlib,json,os,subprocess,sys,shutil
from pathlib import Path
root=Path.cwd();m=json.loads(Path('manifest.json').read_text());rows={r['id']:r for r in json.loads(Path('cases.json').read_text())}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,digest in json.loads(Path('freeze.json').read_text())['files'].items():
 if sha(Path(name))!=digest:raise RuntimeError('frozen execution input changed: '+name)
for name,digest in m['analysis_sha256'].items():
 if sha(Path('analysis_snapshot')/name)!=digest:raise RuntimeError('analysis snapshot changed')
sys.path.insert(0,str(root/'analysis_snapshot'));from run_r18 import case_args
for name,digest in m['binary_sha256'].items():
 if sha(Path('build')/name)!=digest:raise RuntimeError('binary identity differs')
if not os.environ.get('CUDA_VISIBLE_DEVICES') or not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('missing assigned CUDA visibility')
original={k:os.environ.get(k) for k in ('SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_STEP_GPUS','CUDA_VISIBLE_DEVICES')}
cuda_root=Path(shutil.which('nvcc')).resolve().parent.parent
candidates=[cuda_root/'lib64/libcudart.so.12',cuda_root/'lib/libcudart.so.12',*cuda_root.glob('targets/*/lib/libcudart.so.12')]
libraries=sorted({p.resolve() for p in candidates if p.is_file()})
if len(libraries)!=1:raise RuntimeError('expected one CUDA12 runtime under '+str(cuda_root)+': '+str(libraries))
Path('runtime-library.json').write_text(json.dumps(dict(path=str(libraries[0]),sha256=sha(libraries[0])),indent=2)+'\n')
lib=ctypes.CDLL(str(libraries[0]));count=ctypes.c_int();rc=lib.cudaGetDeviceCount(ctypes.byref(count))
if rc:raise RuntimeError('cudaGetDeviceCount failed: '+str(rc))
assigned=[]
for device in range(count.value):
 uuid=(ctypes.c_ubyte*16)();rc=lib.cudaDeviceGetUuid(ctypes.byref(uuid),device)
 if rc:raise RuntimeError('cudaDeviceGetUuid failed: '+str(rc))
 h=bytes(uuid).hex();assigned.append('GPU-'+h[:8]+'-'+h[8:12]+'-'+h[12:16]+'-'+h[16:20]+'-'+h[20:])
Path('assigned-devices.json').write_text(json.dumps(dict(original=original,runtime_visible_uuids=assigned,selected=m['gpu_uuid']),indent=2)+'\n')
if m['gpu_uuid'] not in assigned:raise RuntimeError('requested GPU is not in this step assigned CUDA visibility')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=m['gpu_uuid']);Path('raw').mkdir();Path('full-d').mkdir()
def events(data):return {v['event']:v for l in data.decode().splitlines() if l.startswith('{') for v in [json.loads(l)]}
checks=[]
for stage in (4,6):
 row=next(r for r in rows.values() if r['stages']==stage and r['release_warp_observer']==0)
 for variant in ('plain','stamped'):
  name=f'cfg_b_s{stage}_{variant}';cmd=['build/'+name,'--mode','check-release-warps','--check-output','full-d/'+name+'.bin',*case_args(row)]
  p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);Path('raw/'+name+'_check.jsonl').write_bytes(p.stdout);Path('raw/'+name+'_check.stderr').write_bytes(p.stderr)
  es=events(p.stdout)
  if p.returncode or es['setup']['gpu_uuid']!=m['gpu_uuid'] or es['release_warps_check']['different'] or es['release_warps_check']['nonfinite']:raise RuntimeError('full D off/on or GPU check failed')
  checks.append(dict(binary=name,elements=es['release_warps_check']['elements'],D_sha256=sha(Path('full-d')/(name+'.bin'))))
 if (Path('full-d')/f'cfg_b_s{stage}_plain.bin').read_bytes()!=(Path('full-d')/f'cfg_b_s{stage}_stamped.bin').read_bytes():raise RuntimeError('full D plain/stamped mismatch')
Path('full-D-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
records=[]
for sample in m['samples']:
 row=rows[sample['case']];cmd=['build/'+sample['binary'],'--mode','warm',*case_args(row)]
 p=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 raw=Path('raw')/(sample['id']+'.jsonl.gz');raw.write_bytes(gzip.compress(p.stdout,mtime=0));Path('raw/'+sample['id']+'.stderr').write_bytes(p.stderr)
 records.append(dict(sample,raw=str(raw),raw_sha256=sha(raw),returncode=p.returncode))
 Path('records.json').write_text(json.dumps(records,indent=2)+'\n')
Path('collection-complete.json').write_text(json.dumps(dict(samples=len(records),nonzero_processes=sum(r['returncode']!=0 for r in records),manifest_sha256=sha(Path('manifest.json'))),indent=2)+'\n')
