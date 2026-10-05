"""S16 finite short admission. Own independent source B is required before run.

Reuse the signed A rules and diagnostic CLI; no calibration or performance run.
"""
from pathlib import Path
import argparse,json,sys,subprocess
from common.suite_io import require,read_json,sha,digest,atomic_json,file_lock,validate_gate,process_ok
from runners import s16_short_controller as rules
from runners import s16_storage_v1 as storage
from common.s16_quota_v2 import query_quota

WRITER_SOURCE='probes/feasibility/s16_writer_budget_v1.cpp'

def writer_budget(package,code,required_bytes):
    """Admission uses collected actual ARM evidence, never a CPU fixture label."""
    code=Path(code).resolve();repo=code.parents[1];package=Path(package).resolve();out=package/'writer_budget'
    gate_path=repo/rules.SUITE/'reviews/S16-ARM-writer-result-B-review.json'
    gate=validate_gate(gate_path,repo,'S16','ARM-writer-result-B')
    grant=gate['writer_budget_authorization']
    require(set(grant)=={'evidence_root','producer_root','source_manifest_sha256','collection','raw_archive'},'actual writer authorization fields')
    from common.packed_evidence import logical_path
    evidence_name=logical_path(grant['evidence_root']);require(package==repo/evidence_name,'only actual independently reviewed ARM evidence location; arbitrary CLI package rejected')
    for field in ('collection','raw_archive'):
        ref=grant[field];require(set(ref)=={'path','sha256'},'actual writer evidence ref')
        p=repo/logical_path(ref['path']);require(gate['gate_files'].get(ref['path'])==ref['sha256'] and sha(p)==ref['sha256'],'actual raw/collection reviewer-bound identity')
    collection=read_json(repo/grant['collection']['path'])
    require(collection['status']=='all62_actual_ARM_writer_files_and_full_XZ_stream_verified_offhost' and collection['archive_sha256']==grant['raw_archive']['sha256'] and len(collection['files'])==62 and collection['GPU_execution']is False,'actual full62 raw archive source verification')
    required=['source_manifest.json','writer_budget/summary.json','writer_budget/allocation.json','writer_budget/writer','writer_budget/writer.receipt.json','writer_budget/writer.stdout','writer_budget/writer.stderr']
    for name in required:
        relative=evidence_name+'/'+name
        require(gate['gate_files'].get(relative)==sha(package/name) and collection['files'][name]['sha256']==sha(package/name),'review must bind actual writer metadata/binary/output')
    require(sha(package/'source_manifest.json')==grant['source_manifest_sha256'],'actual exact source package manifest')
    manifest=read_json(package/'source_manifest.json');require(len(manifest)==9,'actual reviewed9 source members')
    for name,h in manifest.items():require(sha(package/name)==h,'ARM writer frozen package source changed')
    env=read_json(out/'allocation.json');summary=read_json(out/'summary.json')
    require(env['machine']=='aarch64' and env['scheduler_CPUs']==1 and env['GPU_kernel_launches']==0 and env['CUDA_queries']==0,'actual single CPU ARM writer environment')
    require(summary['status']=='CPU_writer_necessary_budget_pass' and summary['GPU_execution']is False and summary['numeric_B3_qualified']is False,'actual writer necessary budget must pass')
    require(summary['source_sha256']==sha(code/WRITER_SOURCE) and summary['word_artifacts_sha256']==sha(code/'common/word_artifacts.hpp') and summary['binary_sha256']==sha(out/'writer'),'actual writer source/helper/binary identity')
    receipt=read_json(out/'writer.receipt.json')
    require(process_ok(receipt) and receipt['stdout_sha256']==sha(out/'writer.stdout') and receipt['stderr_sha256']==sha(out/'writer.stderr') and receipt['argv']==[str(Path(grant['producer_root'])/'writer_budget/writer'),str(Path(grant['producer_root'])/'writer_budget/raw'),'396','30'],'actual writer receipt and output identity')
    rows=[json.loads(x) for x in (out/'writer.stdout').read_text().splitlines()]
    result=rows[-1];files=rows[:-1]
    require(result==summary['CPU_writer_summary'] and result['type']=='CPU_writer_budget' and result['full_shape_completed']is True and result['completed_bytes']==4490994944 and result['files']==24 and 0<=result['writer_seconds']<=30,'full 24-file writer measurement within necessary30s budget')
    require(Path(grant['producer_root']).is_absolute() and str(grant['producer_root']).startswith('/gpfs/scratch/'),'actual original producer execution namespace')
    require(result['budget_seconds']==30 and result['GPU_execution']is False,'writer CPU only fixed budget')
    expected=[]
    for launch,i in enumerate((1,2,5,33)):
        b=396;r=4;q=16384
        sizes={'trace':b*i*r*q,'final_slots':b*r*q,'ring_guards':b*32*r*q+32,'lifecycle':b*i*64,'counts':b*96,'stamps':b*40}
        for leaf,nbytes in sizes.items():expected.append((launch,i,leaf,'budget_'+str(launch)+'_'+leaf+'.u32le',nbytes))
    require([(r['launch_index'],r['iterations'],r['leaf'],r['path'],r['bytes']) for r in files]==expected,'fixed maximum writer geometry')
    require(len(files)==24 and len({r['path'] for r in files})==24 and all(r['type']=='CPU_writer_file' for r in files) and sum(r['bytes'] for r in files)==4490994944,'writer full file ledger')
    require(type(required_bytes)is int and 0<required_bytes<=4490994944,'actual device geometry exceeds measured writer shape; new CPU budget required')
    for row in files:
        raw=collection['files']['writer_budget/raw/'+row['path']]
        require(raw['bytes']==row['bytes'] and raw['sha256']==row['sha256'] and raw['stored_in_archive']is True,'actual24 raw arrays bound full decode evidence')
    return {'actual_result_review_sha256':sha(gate_path),'raw_archive_sha256':grant['raw_archive']['sha256'],'summary_sha256':sha(out/'summary.json'),'source_manifest_sha256':sha(package/'source_manifest.json'),'receipt_sha256':sha(out/'writer.receipt.json'),'covered_artifact_bytes':4490994944,'necessary_only':True}

def identity(code,env,device_path):
    source=read_json(code/rules.MANIFEST)['files_sha256']
    files=[Path(__file__),code/'runners/s16_storage_v1.py',code/'runners/s16_short_controller.py',code/'common/s16_quota_v1.py',code/'common/s16_quota_v2.py']
    return rules.family_identity(env,sha(device_path),source,{str(p.relative_to(code)):sha(p) for p in files})

def resident_bytes(repo,points):
    total=0
    for c in points:
        root=repo/storage.run_name(c)
        if root.exists():
            for p in root.rglob('*'):
                require(not p.is_symlink(),'resident symlink blocks admission')
                if p.is_file():total+=p.stat().st_size
    return total

def max_first_ready(ledger):
    entry=ledger['entries'].get('0')
    require(entry is not None and entry['coordinate']['case_id']==rules.MAX_FIRST and entry['state'] in ('resident_passed','archived_verified'),'max-first actual short process must pass before any downstream target')

def execute(repo,start,stop,device_path,quota_path,writer_package):
    repo=Path(repo).resolve();code=repo/rules.CORE_CAMPAIGN;suite=repo/rules.SUITE
    require(type(start)is int and type(stop)is int and 0<=start<stop<=36 and stop-start<=4,'finite1..4-coordinate slice')
    storage.controller_gates(repo)
    validate_gate(suite/'reviews/S16-ARM-writer-package-B-review.json',repo,'S16','ARM-writer-package-B')
    from runners.environment import inspect_allocation,environment_identity,budget
    allocation=inspect_allocation();budget(allocation,30);env=environment_identity(allocation)
    review_path=suite/'reviews/S16-short-source-B-review.json'
    review=validate_gate(review_path,repo,'S16','early-validation-source-B')
    device=rules.device_binding(repo,device_path,read_json(Path(device_path).with_suffix('.binding.json')),allocation,review,review_path)
    contract=read_json(code/rules.CONTRACT);profiles=read_json(code/rules.PROFILES);points=rules.coordinates(contract,profiles)
    rows=rules.footprint(contract,profiles,device)['coordinates'];writer=writer_budget(writer_package,code,rows[0]['artifact_bytes_upper_bound'])
    ident=identity(code,env,device_path);idh=digest(ident);ledger_path=suite/'tma_stage_request-short-v1-ledger.json'
    with file_lock(suite/'.tma_stage_request-short-v1-controller.lock'):
        if ledger_path.exists():ledger=read_json(ledger_path);require(ledger['identity']==ident,'changed family source/controller/UUID cannot resume')
        else:ledger={'schema_version':1,'identity':ident,'writer_necessary_budget':writer,'entries':{}};atomic_json(ledger_path,ledger)
        for c in points[start:stop]:
            key=str(c['index']);root=repo/storage.run_name(c);row=rows[c['index']]
            action=rules.next_action(root,ledger['entries'].get(key),repo,c,ident)
            if row['target_launches']==0:
                require('_s4_r4_' in c['case_id'] and not root.exists() and action in ('run','verify_capacity_terminal'),'finite actual capacity terminal only')
                terminal={'coordinate':c,'state':'resource_reject_before_launch','family_identity_sha256':idh,'target_launches':0,'device_sha256':sha(device_path),'capacity':row}
                old=ledger['entries'].get(key);require(old is None or old==terminal,'capacity terminal changed')
                ledger['entries'][key]=terminal;atomic_json(ledger_path,ledger);continue
            if action=='archive_audit':continue
            if action in ('run','resume'):
                if c['index']!=0:max_first_ready(ledger)
                # Cooperative filesystem lock shared across S15/S16 deployments;
                # root must also serialize producers that do not take this lock.
                with file_lock(repo.parent/'.raw-storage-admission.lock'):
                    query_quota(repo,quota_path)
                    headroom=rules.quota_headroom(repo,quota_path,env)
                    resident=resident_bytes(repo,points);admit=rules.quota_budget(resident,row['artifact_bytes_upper_bound'],headroom)
                    if not admit['allow']:
                        atomic_json(suite/'implementation/s16-storage-run-B/checkpoint.json',{'coordinate':c,'resident_bytes':resident,**admit,'GPU_target_launches':0});return 0
                    result=invoke(code,suite,root,c,action,ledger,ledger_path,idh)
            else:result=invoke(code,suite,root,c,action,ledger,ledger_path,idh)
            if result:return result
    return 0

def invoke(code,suite,root,c,action,ledger,ledger_path,identity_sha):
    require(action in ('run','resume','audit'),'only existing diagnostic CLI')
    key=str(c['index'])
    if action!='audit':
        ledger['entries'][key]={'coordinate':c,'state':'started_unknown','family_identity_sha256':identity_sha,'run_path':storage.run_name(c)};atomic_json(ledger_path,ledger)
    argv=[sys.executable,'-B',str(code/'validate_suite.py'),action]
    if action=='run':argv+=['--suite',str(suite),'--output',str(root),'--contract',str(code/rules.CONTRACT),'--profiles',str(code/rules.PROFILES),'--adapter-manifest',str(code/rules.MANIFEST),'--case',c['case_id'],'--profile',c['profile_id'],'--seed','3']
    else:argv+=[str(root)]
    result=subprocess.run(argv,check=False)
    if result.returncode:
        ledger['entries'][key]={'coordinate':c,'state':'failed','family_identity_sha256':identity_sha,'run_path':storage.run_name(c),'returncode':result.returncode};atomic_json(ledger_path,ledger);return result.returncode
    storage.inventory(code.parents[1],storage.run_name(c))
    ledger['entries'][key]={'coordinate':c,'state':'resident_passed','family_identity_sha256':identity_sha,'run_path':storage.run_name(c),'manifest_sha256':sha(root/'validation_manifest.json')};atomic_json(ledger_path,ledger)
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=Path.cwd());p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=1);p.add_argument('--device-json',type=Path,required=True);p.add_argument('--quota-receipt',type=Path,required=True);p.add_argument('--writer-package',type=Path,required=True);a=p.parse_args()
    return execute(a.repo,a.start,a.stop,a.device_json,a.quota_receipt,a.writer_package)
if __name__=='__main__':raise SystemExit(main())
