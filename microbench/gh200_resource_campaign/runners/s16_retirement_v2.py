"""S16 explicit resident retirement on its producer node. No GPU execution/cleanup."""
from pathlib import Path
import argparse,json,os,re,shutil,socket,subprocess,time
from common.suite_io import require,read_json,sha,digest,file_lock,gpu_paths
from common.gpu_registry import registry_root
from runners.s16_storage_v1 import (SUITE,coordinate,run_name,inventory,validate_archived,
                                    controller_gates,pack_destination,publish_new_json,local_destination)


def scratch_repo(repo):
    repo=Path(repo).absolute()
    require('..' not in repo.parts and repo.is_relative_to(Path('/gpfs/scratch')),
            'retirement restricted to explicit ROMEO scratch repo')
    cursor=Path(repo.anchor)
    for part in repo.parts[1:]:
        cursor=cursor/part
        require(not cursor.is_symlink() and cursor.is_dir(),'scratch repo ancestors must be real directories')
    return repo


def cpu_allocation(producer):
    job=os.environ.get('SLURM_JOB_ID','');require(re.fullmatch(r'[0-9]+',job) is not None,'active CPU Slurm allocation required')
    records=[]
    def capture(argv):
        start=time.time_ns();result=subprocess.run(argv,capture_output=True,text=True,timeout=15)
        records.append({'argv':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'start_ns':start,'stop_ns':time.time_ns()})
        require(result.returncode==0,'CPU Slurm query failed');return result.stdout
    config=capture(['scontrol','show','config'])
    require(re.search(r'ClusterName\s*=\s*\S*romeo',config,re.I) is not None,'ROMEO CPU allocation required')
    info=capture(['scontrol','show','job','-o',job])
    require(re.search(r'\bJobState=RUNNING\b',info) and re.search(r'\bTimeLimit=00:10:00\b',info),'RUNNING 10 minute CPU allocation')
    require(re.search(r'\bUserId=\S*\('+str(os.geteuid())+r'\)',info),'CPU allocation UID')
    require(re.search(r'\bNumNodes=1\b',info),'single producer CPU node')
    cpus=re.search(r'\bNumCPUs=([0-9]+)\b',info);require(cpus and int(cpus[1])>0,'positive allocated CPUs')
    allocated=re.search(r'\b(?:AllocTRES|TRES)=(\S+)',info)
    require(allocated and re.search(r'(?:^|,)cpu=[1-9][0-9]*(?:,|$)',allocated[1]),'explicit allocated CPU TRES')
    require(not re.search(r'gres/gpu(?:[:=]|\b)|\b(?:Gres|TresPerNode|TresPerJob)=\S*gpu',info,re.I),'CPU retirement must allocate no GPU')
    nodes=re.search(r'\bNodeList=(\S+)',info);require(nodes,'allocated producer NodeList')
    hosts=capture(['scontrol','show','hostnames',nodes[1]]).split()
    require(len(hosts)==1 and hosts[0].split('.')[0]==producer.split('.')[0],'CPU allocation on original producer')
    return {'job':job,'producer_host':producer,'allocated_CPUs':int(cpus[1]),'GPU_allocation':False,'scontrol_records':records}


def source_fingerprint(repo):
    parent=repo.parent
    source_path=parent/'source-manifest.json';launch_path=parent/'launch-manifest.json'
    require(source_path.is_file() and launch_path.is_file() and not source_path.is_symlink() and not launch_path.is_symlink(),'original package source and launch manifests')
    source=read_json(source_path);launch=read_json(launch_path)
    require(isinstance(source,dict) and source and isinstance(launch,dict) and 'source-manifest.json' in launch,'nonempty original source and launcher closure')
    for name,digest_value in source.items():
        path=local_destination(repo,name)
        require(path.is_file() and sha(path)==digest_value,'original frozen source file drift '+name)
    for name,digest_value in launch.items():
        require(isinstance(name,str) and len(Path(name).parts)==1 and name not in ('.','..'),'flat original launcher name')
        path=parent/name
        require(not path.is_symlink() and path.is_file() and sha(path)==digest_value,'original launch file drift '+name)
    return {'source_manifest_sha256':sha(source_path),'launch_manifest_sha256':sha(launch_path),'source_files':len(source),'launch_files':len(launch)}


def no_active(uuid):
    lock,quarantine=gpu_paths(uuid)
    require(not quarantine.exists() and not quarantine.is_symlink(),'UUID quarantine blocks retirement')
    registry=registry_root(uuid)
    for path in (registry,registry/'active'):
        require(not path.is_symlink(),'UUID registry symlink')
        require(path.exists() and path.is_dir() and path.stat().st_uid==os.geteuid()
                and path.stat().st_mode & 0o777==0o700,'existing private UUID registry directory ownership/mode')
    active=registry/'active'
    require(not any(active.iterdir()),'nonempty active UUID registry blocks retirement')


def origins(repo,entry):
    refs=entry.get('verification_origins',{})
    require(set(refs)=={'remote','offhost'},'two preserved verifier origins required')
    result={}
    for role,ref in refs.items():
        require(set(ref)=={'path','sha256'},'origin reference fields')
        path=pack_destination(repo,ref['path'])
        require(path.is_file() and sha(path)==ref['sha256'],'origin exact current SHA')
        result[role]=read_json(path)
    remote,off=result['remote'],result['offhost']
    require(remote['archive_sha256']==off['archive_sha256']==entry['archive']['sha256'],
            'both origin archive identities')
    require(remote['receipt_sha256']==entry['archive_verification']['sha256']
            and off['receipt_sha256']==entry['offhost_receipt']['sha256'],'both receipt identities')
    require(remote['verifier']['host']!=off['verifier']['host'],'distinct offhost verifier')
    return result


def family_device(identity,device_path,initial):
    required={'schema_version','family','environment','device_sha256','source_files_sha256','controller_files_sha256'}
    require(isinstance(identity,dict) and set(identity)==required,'original S16 family_identity schema')
    require(type(identity['schema_version'])is int and identity['schema_version']==1 and identity['family']=='tma_stage_request','original S16 family identity version')
    require(isinstance(identity['environment'],dict) and isinstance(identity['source_files_sha256'],dict) and identity['source_files_sha256'] and isinstance(identity['controller_files_sha256'],dict) and identity['controller_files_sha256'],'original condition/source/controller maps')
    require(isinstance(identity['device_sha256'],str) and re.fullmatch('[0-9a-f]{64}',identity['device_sha256']) and sha(device_path)==identity['device_sha256'],'resident device bytes must equal original query device SHA')
    device=read_json(device_path);uuid=device['uuid']
    require(initial['uuid']==identity['environment'].get('uuid')==uuid,'original environment/initial/device physical UUID')
    require(type(initial['execution_uid'])is int and initial['execution_uid']==identity['environment'].get('execution_uid'),'original environment/initial execution UID')
    return device


def receipt_root(repo,index,manifest_sha):
    require(len(manifest_sha)==64 and all(ch in '0123456789abcdef' for ch in manifest_sha),'manifest SHA domain')
    parent=repo/SUITE/'implementation/s16-retire-v2'
    cursor=repo
    for part in parent.relative_to(repo).parts:
        cursor=cursor/part
        require(not cursor.is_symlink(),'retirement receipt ancestor symlink')
        if cursor.exists():require(cursor.is_dir(),'retirement receipt ancestor regular directory')
    output=parent/('index-'+str(index)+'-'+manifest_sha)
    require(not output.exists() and not output.is_symlink(),
            'prior retirement reservation exists; inspect offline, never auto reenter')
    return output


def retire(repo,index,expected_run):
    repo=scratch_repo(repo);integer_index=type(index)is int and 0<=index<36
    require(integer_index,'finite S16 coordinate index')
    c=coordinate(repo,index);name=run_name(c)
    require(expected_run==name,'explicit run must equal original fixed coordinate')
    root=repo/name
    # Inspect existing paths without touching a GPU lock or creating receipts.
    cursor=repo
    for part in Path(name).parts:
        cursor=cursor/part;require(not cursor.is_symlink(),'resident ancestor symlink')
    require(root.is_dir(),'original resident exists')
    initial=read_json(root/'environment/initial.json')
    current_host=socket.gethostname()
    require(current_host.split('.')[0]==initial['host'].split('.')[0],
            'retirement must run on the actual producer node, never login/offhost')
    require(os.geteuid()==initial['execution_uid'],'original producer execution UID')
    allocation=cpu_allocation(initial['host'])
    controller_gates(repo)
    suite=repo/SUITE
    with file_lock(suite/'.tma_stage_request-short-v1-controller.lock'):
        ledger_path=suite/'tma_stage_request-short-v1-ledger.json';ledger=read_json(ledger_path)
        entry=ledger['entries'][str(index)]
        require(entry['state']=='archived_verified' and entry['coordinate']==c and entry['run_path']==name,
                'original explicitly admitted archived coordinate')
        require(entry['family_identity_sha256']==digest(ledger['identity']),'original family condition identity')
        output=receipt_root(repo,index,entry['manifest_sha256'])
        device=family_device(ledger['identity'],root/'environment/device.json',initial);uuid=device['uuid']
        lock,_=gpu_paths(uuid)
        # Plain UUID flock only. Never gpu_lock(), reconcile(), stop_group() or kill.
        with file_lock(lock):
            no_active(uuid)
            source_before=source_fingerprint(repo)
            validate_archived(entry,repo,c)
            origins(repo,entry)
            closure=read_json(pack_destination(repo,entry['closure']['path']))
            resident=inventory(repo,name)
            require(resident==closure['members'],'all current resident bytes equal both archived evidence')
            # Confirm the frozen environment read before locking was not replaced.
            require(read_json(root/'environment/initial.json')==initial and read_json(root/'environment/device.json')==device,
                    'producer identity changed during admission')
            no_active(uuid)
            output=receipt_root(repo,index,entry['manifest_sha256'])
            output.parent.mkdir(parents=True,exist_ok=True);output.mkdir(exist_ok=False)
            record={'schema_version':1,'index':index,'run_path':name,'producer_host':initial['host'],
                    'actual_host':current_host,'execution_uid':os.geteuid(),'uuid':uuid,
                    'manifest_sha256':entry['manifest_sha256'],'family_identity_sha256':entry['family_identity_sha256'],
                    'ledger_sha256':sha(ledger_path),'CPU_allocation':allocation,'original_source_fingerprint':source_before,
                    'archive':entry['archive'],'index_ref':entry['index'],
                    'closure':entry['closure'],'archive_verification':entry['archive_verification'],
                    'offhost_receipt':entry['offhost_receipt'],'retirement_source_sha256':sha(Path(__file__)),
                    'verification_origins':entry['verification_origins'],
                    'verified_resident_files':len(resident),'verified_resident_bytes':sum(v['bytes'] for v in resident.values()),
                    'started_unix_ns':time.time_ns(),'GPU_execution':False,'process_cleanup_attempted':False}
            publish_new_json(output/'started.json',{**record,'status':'retirement_started'})
            try:
                shutil.rmtree(root)
                require(not root.exists() and not root.is_symlink(),'resident deletion completion unknown')
                require(sha(ledger_path)==record['ledger_sha256'],'retirement does not alter ledger')
                require(source_fingerprint(repo)==source_before,'original source/launch closure changed during retirement')
                # All retained sealed references remain exact after deleting the run.
                for ref in (entry['archive'],entry['index'],entry['closure'],entry['archive_verification'],entry['offhost_receipt'],*entry['verification_origins'].values()):
                    require(sha(pack_destination(repo,ref['path']))==ref['sha256'],'retained packed evidence changed')
                final={**record,'status':'resident_retired_archived_evidence_retained','completed_unix_ns':time.time_ns()}
                publish_new_json(output/'complete.json',final)
                return final
            except BaseException as exc:
                # A partial deletion or uncertain publication is never retried.
                failure={**record,'status':'retirement_failed_or_unknown','error_type':type(exc).__name__,
                         'error':str(exc),'resident_exists_observed':root.exists(),'automatic_reentry':False}
                try:publish_new_json(output/'failed.json',failure)
                except Exception:pass  # immutable started reservation remains authoritative
                raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--index',type=int,required=True)
    p.add_argument('--run-path',required=True);p.add_argument('--expected-source-sha256',required=True)
    a=p.parse_args();require(sha(Path(__file__))==a.expected_source_sha256,'dispatched retirement source drift')
    # The dispatch operator must verify the independent retirement review before
    # deployment; this SHA check is artifact identity, not self-authorization.
    print(json.dumps(retire(a.repo,a.index,a.run_path)),flush=True)

if __name__=='__main__':main()
