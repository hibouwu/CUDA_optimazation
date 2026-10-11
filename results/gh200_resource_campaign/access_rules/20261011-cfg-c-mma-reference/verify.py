from pathlib import Path
from array import array
import hashlib,json,statistics,sys
from validate_allocation import validate
root=Path(__file__).resolve().parent;data=root/'returned';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((root/'manifest.json').read_text());build=Path(m['build_run'])
for name,digest in json.loads((root/'freeze.json').read_text())['files'].items():assert sha(root/name)==sha(data/name)==digest
receipt=json.loads((root/'receipt.json').read_text());assert receipt['returncode']==0 and sha(root/'output.tar.gz')==receipt['sha256']
assert sha(data/'build/cfg_c_mma_reference')==m['binary_sha256']==sha(build/'returned/build/cfg_c_mma_reference')
for name,digest in json.loads((root/'origin-source_hashes.json').read_text()).items():assert sha(build/name)==digest
validate((data/'allocation.txt').read_text())
records=json.loads((data/'records.json').read_text());assert [{k:r[k] for k in ('id','trial','iterations','command')} for r in records]==m['samples']
assert json.loads((data/'complete.json').read_text())==dict(records=6,manifest_sha256=sha(root/'manifest.json'))
results=[]
for row in records:
 assert row['returncode']==0
 for name,digest in row['files'].items():assert sha(data/name)==digest
 directory=data/'samples'/row['id'];r=json.loads((directory/'stdout.json').read_text());n=row['iterations']
 fixed=dict(status='measured',dtype='fp16',source='SS',scope='one_cta',n=128,k=16,chains=2,groups=2,threads=256,blocks=1,iterations=n,per_chain_batch=4,wait=1,b_major='MN',gpu_uuid=m['gpu_uuid'],warmup_calls=16,warmup_policy='fixed16_same_probe',unit='clock64_cycle/CTA',local_bytes_per_thread=0,coordinate_checks=[1,2],output_elements=32768,max_output_error=0)
 for key,value in fixed.items():assert r[key]==value,(key,r[key],value)
 work=2*64*128*16*2*4*2*n;assert r['work_flop']==work
 stamps=r['stamps'];assert len(stamps)==1
 ns0,ns1,c0,c1,sm=stamps[0];assert 0<c0<c1 and 0<ns0<ns1 and 0<=sm<132 and r['elapsed']==c1-c0
 values=array('f');values.frombytes((directory/'output.f32').read_bytes())
 if sys.byteorder!='little':values.byteswap()
 assert len(values)==32768 and all(v==n/4 for v in values)
 results.append(dict(id=row['id'],iterations=n,cycles=r['elapsed'],work_flop=work,flop_per_SM_cycle=work/r['elapsed'],output_values_checked=len(values),sm=sm))
summary=[dict(iterations=n,cycles=[r['cycles'] for r in results if r['iterations']==n],median_cycles=statistics.median(r['cycles'] for r in results if r['iterations']==n),median_flop_per_SM_cycle=statistics.median(r['flop_per_SM_cycle'] for r in results if r['iterations']==n)) for n in (64,1024)]
report=dict(results=results,summary=summary,formal_output_values_checked=196608,short_coordinate_checks_from_host=True,observation_qualified=False,prediction_qualified=False,limits=m['limits'])
(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
