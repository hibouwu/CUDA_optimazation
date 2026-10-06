"""S16 finite read-only queries and complete per-run lossless storage.

No target kernel launches. Queries require a reviewed binary and Slurm; sealing,
verification and explicit scratch retirement are separate commands.
"""
from pathlib import Path
import argparse,getpass,os,socket,tarfile,time,shutil,json,subprocess,shlex,uuid
from urllib.parse import unquote
from common.suite_io import require,sha,read_json,atomic_json,verify_files,file_lock,bounded,process_ok,gpu_lock
from common.packed_evidence import PackedEvidence,logical_path
from runners.s16_short_controller import SUITE,coordinates,CONTRACT,PROFILES,CORE_CAMPAIGN,validate_archived,device_binding
from common.s16_quota_v1 import query_quota


def controller_gates(repo):
    from common.suite_io import validate_gate
    suite=Path(repo)/SUITE
    validate_gate(suite/'reviews/S16-short-controller-A-review.json',repo,'S16','controller-A')
    validate_gate(suite/'reviews/S16-short-controller-source-B-review.json',repo,'S16','controller-source-B')


def query_device(repo,binary,output):
    from runners.environment import inspect_allocation,environment_identity,budget
    from common.suite_io import validate_gate
    repo=Path(repo).resolve();binary=Path(binary).resolve();output=Path(output).absolute()
    controller_gates(repo);suite=repo/SUITE
    source_path=suite/'reviews/S16-short-source-B-review.json'
    review=validate_gate(source_path,repo,'S16','early-validation-source-B')
    require(sha(binary) in {v for n,v in review['gate_files'].items() if n.endswith('/diagnostics/tma_stage_request/probe')},'reviewed actual S16 binary required')
    require(output.is_relative_to(repo),'S16 device evidence inside deployment')
    cursor=output
    while cursor!=repo:require(not cursor.is_symlink(),'device query namespace symlink');cursor=cursor.parent
    env=inspect_allocation();budget(env,30);output.parent.mkdir(parents=True,exist_ok=True)
    queries=output.parent/(output.stem+'.queries');require(not queries.is_symlink(),'device query namespace regular');queries.mkdir(exist_ok=True)
    import uuid
    query=queries/(str(time.time_ns())+'-'+str(os.getpid())+'-'+uuid.uuid4().hex);query.mkdir()
    stdout=query/'stdout';stderr=query/'stderr'
    with gpu_lock(env['uuid']):receipt=bounded([str(binary),'device'],repo,stdout,stderr,30,uuid=env['uuid'])
    atomic_json(query/'process.json',receipt)
    require(process_ok(receipt),'device query failure; no alias publication')
    device=read_json(stdout);device_path=query/'device.json';atomic_json(device_path,device)
    binding={'schema_version':1,'device_sha256':sha(device_path),'stdout':{'path':str(stdout.relative_to(repo)),'sha256':sha(stdout)},'binary_sha256':sha(binary),'source_review_sha256':sha(source_path),'environment':environment_identity(env),'allocation_job':env['job'],'receipt':receipt}
    device_binding(repo,device_path,binding,env,review,source_path)
    atomic_json(query/'binding.json',binding);atomic_json(output,device);atomic_json(output.with_suffix('.binding.json'),binding)
    return device


def coordinate(repo,index):
    code=Path(repo)/CORE_CAMPAIGN;points=coordinates(read_json(code/CONTRACT),read_json(code/PROFILES));require(type(index)is int and 0<=index<36,'S16 finite index');return points[index]


def run_name(c):return SUITE+'/tma_stage_request/short-v1-'+c['case_id']+'--'+c['profile_id']


def inventory(repo,run):
    root=Path(repo)/run
    require(root.resolve().is_relative_to(Path(repo).resolve()),'resident namespace inside deployment')
    cursor=Path(repo)
    for part in Path(run).parts:
        cursor=cursor/part;require(not cursor.is_symlink(),'no symlink in resident namespace')
    require(root.is_dir() and not root.is_symlink(),'resident run directory')
    for item in root.rglob('*'):require(not item.is_symlink() and (item.is_file() or item.is_dir()),'regular evidence only')
    manifest=read_json(root/'validation_manifest.json');verify_files(root,manifest)
    names={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    require(names==set(manifest)|{'validation_manifest.json','validation_state.json'},'resident exact complete manifest file set')
    require(read_json(root/'diagnostic_summary.json')['status']=='case_diagnostic_passed','only complete diagnostic storage')
    for p in root.rglob('*.registry.json'):require(read_json(p)['state']=='cleanup_confirmed','unresolved registered process')
    for p in root.rglob('*.receipt.json'):require(read_json(p).get('cleanup_confirmed') is True,'unconfirmed process receipt')
    return {run+'/'+n:{'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in sorted(names)}


def verifier_identity(archive):
    return {'host':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'uid':os.geteuid(),'archive_path':str(Path(archive).resolve())}


def local_destination(repo,name,*,new=False):
    """Validate every existing ancestor before mkdir, transfer or publication."""
    repo=Path(repo).absolute();logical_path(name)
    destination=repo/name
    cursor=Path(destination.anchor)
    for part in destination.parts[1:]:
        cursor=cursor/part
        require(not cursor.is_symlink(),'local transport symlink forbidden: '+str(cursor))
        if cursor.exists() and cursor!=destination:require(cursor.is_dir(),'local transport parent must be directory')
    require(destination.resolve().is_relative_to(repo.resolve()),'local transport path escapes deployment')
    if new:require(not destination.exists(),'collection never overwrites local evidence')
    elif destination.exists():require(destination.is_file(),'local transport member must be regular')
    return destination


def verify_pack(repo,entry,location,output):
    require(location in ('remote','offhost'),'finite verifier location')
    repo=Path(repo);index=read_json(repo/entry['index']['path']);closure=read_json(repo/entry['closure']['path'])
    bundle=PackedEvidence(repo/entry['archive']['path'],repo/entry['index']['path'],expected_index_sha256=entry['index']['sha256'],expected_archive_sha256=entry['archive']['sha256'],expected_closure=closure['members'])
    result=bundle.verify_all();receipt={'schema_version':1,'status':'full_members_verified','location':location,'archive_sha256':entry['archive']['sha256'],'index_sha256':entry['index']['sha256'],'closure_sha256':entry['closure']['sha256'],'run_path':entry['run_path'],'manifest_sha256':entry['manifest_sha256'],'members':closure['members'],'member_count':result['members'],'uncompressed_bytes':result['uncompressed_bytes'],'verification':'complete_decode_all_members_sha256'}
    atomic_json(output,receipt)
    attestation={'schema_version':1,'receipt_sha256':sha(output),'archive_sha256':entry['archive']['sha256'],'verifier':verifier_identity(repo/entry['archive']['path'])}
    atomic_json(Path(output).with_suffix('.origin.json'),attestation);return receipt


def seal(repo,index,output_dir):
    repo=Path(repo).resolve();out=Path(output_dir).resolve();require(out.is_relative_to(repo),'pack directory within deployment');require(not out.exists(),'never overwrite sealed archive')
    c=coordinate(repo,index);run=run_name(c);suite=repo/SUITE
    require(not out.is_relative_to(repo/run),'sealed output cannot mutate source run tree')
    with file_lock(suite/'.tma_stage_request-short-v1-controller.lock'):
        before=inventory(repo,run);out.mkdir(parents=True);archive=out/'run.tar.xz'
        with tarfile.open(archive,'w:xz',preset=1) as stream:
            for name in before:stream.add(repo/name,arcname=name,recursive=False)
        require(inventory(repo,run)==before,'source content or member drift while sealing')
        ah=sha(archive);mh=sha(repo/run/'validation_manifest.json')
        idx={'schema_version':1,'namespace':'repository_relative_v1','archive_sha256':ah,'archive_bytes':archive.stat().st_size,'members':before,'roots':{'original_manifest':run+'/validation_manifest.json'}}
        closure={'schema_version':1,'run_path':run,'manifest_sha256':mh,'members':before}
        atomic_json(out/'index.json',idx);atomic_json(out/'closure.json',closure)
        ledger=read_json(suite/'tma_stage_request-short-v1-ledger.json');resident=ledger['entries'][str(c['index'])]
        require(resident['state']=='resident_passed' and resident['coordinate']==c and resident['manifest_sha256']==mh,'seal original resident ledger')
        entry={'coordinate':c,'state':'archive_remote_only','run_path':run,'manifest_sha256':mh,'family_identity_sha256':resident['family_identity_sha256']}
        for key,file in [('archive',archive),('index',out/'index.json'),('closure',out/'closure.json')]:entry[key]={'path':str(file.relative_to(repo)),'sha256':sha(file)}
        verify_pack(repo,entry,'remote',out/'remote.json');entry['archive_verification']={'path':str((out/'remote.json').relative_to(repo)),'sha256':sha(out/'remote.json')};atomic_json(out/'entry.json',entry)
    return entry


def collect(remote_repo,remote_entry,local_repo):
    """Only fetch exact sealed files; both source identities are checked remotely."""
    remote_repo=Path(remote_repo);remote_entry=Path(remote_entry);local_repo=Path(local_repo).absolute()
    require(remote_repo.is_absolute() and remote_repo.is_relative_to(Path('/gpfs/scratch')),'finite ROMEO scratch source')
    require(not remote_entry.is_absolute() and '..' not in remote_entry.parts,'relative remote entry')
    script="""import json,hashlib,pathlib,sys
r=pathlib.Path(sys.argv[1]);ep=r/sys.argv[2];e=json.loads(ep.read_text());names=[sys.argv[2]]+[e[k]['path'] for k in ['archive','index','closure','archive_verification']];names.append(str(pathlib.Path(e['archive_verification']['path']).with_suffix('.origin.json')))
out={}
for n in names:
 p=pathlib.Path(n)
 assert not p.is_absolute() and '..' not in p.parts
 f=r/p
 assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(r.resolve())
 for parent in f.parents:
  if parent==r:break
  assert not parent.is_symlink()
 h=hashlib.sha256()
 with f.open('rb') as v:
  for b in iter(lambda:v.read(1048576),b''):h.update(b)
 out[n]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps({'files':out,'entry':e},sort_keys=True))
"""
    cmd='python3 -c '+shlex.quote(script)+' '+shlex.quote(str(remote_repo))+' '+shlex.quote(str(remote_entry))
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','romeo',cmd]
    def remote_inventory():
        result=subprocess.run(ssh,capture_output=True,text=True,timeout=180);require(result.returncode==0,'remote sealed inventory failed');return json.loads(result.stdout)
    source_inventory=remote_inventory();before=source_inventory['files'];source_entry=source_inventory['entry']
    code=Path(__file__).resolve().parents[1]
    fixed=coordinates(read_json(code/CONTRACT),read_json(code/PROFILES))
    point=source_entry['coordinate'];require(type(point['index']) is int and 0<=point['index']<204 and point==fixed[point['index']],'collected entry is an exact finite S16 coordinate')
    require(source_entry['state']=='archive_remote_only' and source_entry['run_path']==run_name(point),'collected original run namespace')
    # Preflight the whole fixed transfer/output set before creating any local
    # directory or invoking SCP. Reject even a valid remote pack if the local
    # namespace would follow an existing symlink.
    for n in before:local_destination(local_repo,n,new=True)
    local_destination(local_repo,str(remote_entry),new=True)
    receipt_name=logical_path(source_entry['archive_verification']['path'])
    offhost=(local_repo/receipt_name).with_name('offhost.json')
    outputs=(offhost,offhost.with_suffix('.origin.json'),offhost.with_name('collection.json'))
    for path in outputs:local_destination(local_repo,str(path.relative_to(local_repo)),new=True)
    for n,record in before.items():
        destination=local_destination(local_repo,n,new=True);destination.parent.mkdir(parents=True,exist_ok=True)
        local_destination(local_repo,n,new=True)
        result=subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','romeo:'+shlex.quote(str(remote_repo/n)),str(destination)],capture_output=True,text=True,timeout=180)
        local_destination(local_repo,n)
        require(result.returncode==0 and destination.is_file() and destination.stat().st_size==record['bytes'] and sha(destination)==record['sha256'],'offhost transfer byte identity')
    require(remote_inventory()==source_inventory,'remote sealed source drift during collection')
    entry=read_json(local_repo/remote_entry);require(entry==source_entry,'transferred entry identity differs from preflight')
    for path in outputs:local_destination(local_repo,str(path.relative_to(local_repo)),new=True)
    verify_pack(local_repo,entry,'offhost',offhost)
    atomic_json(offhost.with_name('collection.json'),{'schema_version':1,'remote_repo':str(remote_repo),'entry_path':str(remote_entry),'before_after_members':before,'offhost_receipt_sha256':sha(offhost),'GPU_execution':False})
    return entry


def admit(repo,entry_path,offhost_receipt,offhost_origin):
    repo=Path(repo).resolve();entry=read_json(entry_path);remote_origin=Path(repo/entry['archive_verification']['path']).with_suffix('.origin.json');remote=read_json(remote_origin);off=read_json(offhost_origin)
    require(off['receipt_sha256']==sha(offhost_receipt) and remote['receipt_sha256']==entry['archive_verification']['sha256'] and off['archive_sha256']==remote['archive_sha256']==entry['archive']['sha256'],'verifier attestation bound pack/receipt')
    require(off['verifier']['host']!=remote['verifier']['host'],'offhost must be a distinct verifier host, not another remote path')
    entry['offhost_receipt']={'path':str(Path(offhost_receipt).resolve().relative_to(repo)),'sha256':sha(offhost_receipt)};entry['state']='archived_verified';validate_archived(entry,repo,entry['coordinate'])
    entry['verification_origins']={k:{'path':str(Path(v).resolve().relative_to(repo)),'sha256':sha(v)} for k,v in [('remote',remote_origin),('offhost',offhost_origin)]}
    suite=repo/SUITE
    with file_lock(suite/'.tma_stage_request-short-v1-controller.lock'):
        ledger_path=suite/'tma_stage_request-short-v1-ledger.json';ledger=read_json(ledger_path);old=ledger['entries'][str(entry['coordinate']['index'])]
        require(old['state']=='resident_passed' and old['coordinate']==entry['coordinate'] and old['manifest_sha256']==entry['manifest_sha256'] and old['family_identity_sha256']==entry['family_identity_sha256'],'admit must preserve an actual original resident ledger identity')
        ledger['entries'][str(entry['coordinate']['index'])]=entry;atomic_json(ledger_path,ledger)
    return entry


def retire(repo,index):
    repo=Path(repo).resolve();require(repo.is_relative_to(Path('/gpfs/scratch')),'retire restricted to ROMEO scratch deployment, never canonical local archive');suite=repo/SUITE
    with file_lock(suite/'.tma_stage_request-short-v1-controller.lock'):
        ledger=read_json(suite/'tma_stage_request-short-v1-ledger.json');entry=ledger['entries'][str(index)];c=coordinate(repo,index);validate_archived(entry,repo,c)
        origins=entry['verification_origins'];remote=read_json(repo/origins['remote']['path']);off=read_json(repo/origins['offhost']['path'])
        for ref in origins.values():require(sha(repo/ref['path'])==ref['sha256'],'verification origin changed')
        require(remote['verifier']['host']!=off['verifier']['host'],'independent offhost verifier required')
        root=repo/run_name(c);require(inventory(repo,run_name(c))==read_json(repo/entry['closure']['path'])['members'],'resident source still exact before explicit retire')
        from common.gpu_registry import registry_dirs
        uuid=read_json(root/'environment/device.json')['uuid']
        with gpu_lock(uuid):
            active,_=registry_dirs(uuid);require(not list(active.glob('*.json')),'no active UUID process before retire')
            shutil.rmtree(root)
        atomic_json(suite/'implementation/s16-retire'/('coordinate-'+str(index)+'.json'),{'index':index,'run_path':entry['run_path'],'manifest_sha256':entry['manifest_sha256'],'archived_evidence_retained':True,'retired_unix_ns':time.time_ns()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['quota','device','seal','verify','collect','admit','retire']);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--output',type=Path);p.add_argument('--binary',type=Path);p.add_argument('--index',type=int);p.add_argument('--entry',type=Path);p.add_argument('--location',choices=['remote','offhost']);p.add_argument('--offhost-receipt',type=Path);p.add_argument('--offhost-origin',type=Path);p.add_argument('--remote-repo',type=Path);a=p.parse_args()
    controller_gates(a.repo)
    if a.command=='quota':query_quota(a.repo,a.output)
    elif a.command=='device':query_device(a.repo,a.binary,a.output)
    elif a.command=='seal':seal(a.repo,a.index,a.output)
    elif a.command=='verify':verify_pack(a.repo,read_json(a.entry),a.location,a.output)
    elif a.command=='collect':collect(a.remote_repo,a.entry,a.repo)
    elif a.command=='admit':admit(a.repo,a.entry,a.offhost_receipt,a.offhost_origin)
    else:retire(a.repo,a.index)
if __name__=='__main__':main()
