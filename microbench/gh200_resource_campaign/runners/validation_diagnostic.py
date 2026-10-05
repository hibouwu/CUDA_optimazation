"""Versioned, case-only B3 diagnostics; no calibration, warmup or performance path."""
from __future__ import annotations
import importlib
import ast
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

from common.suite_io import (atomic_json, bounded, digest, file_lock, gate_mapping,
    gpu_lock, process_ok, read_json, relative, require, sha, suite_lock_path,
    utc, validate_gate, verify_files)
from common.suite_snapshot import (CAMPAIGN, check_snapshot, freeze,
    inventory_depfile, local_cpp_closure, python_closure_check, required_reviews)
from runners.environment import Checkpoint, budget, environment_identity, inspect_allocation
from runners.suite_runner import run_command, verify_interrupted_gpu_groups

CORE_REVIEW='reviews/S02-early-validation-B-review-r6.json'
POLICY=CAMPAIGN+'/contracts/early_validation_v1.json'
ENTRY=CAMPAIGN+'/validate_suite.py'
CORE_FILES={ENTRY,POLICY,*[CAMPAIGN+'/'+name for name in (
    'runners/validation_diagnostic.py','common/suite_io.py','common/suite_snapshot.py',
    'common/gpu_registry.py','common/calibration.py','runners/environment.py',
    'runners/suite_runner.py','auditors/suite.py','common/family_b3.py','auditors/async_copy.py',
    'auditors/async_copy_sass_baseline.py','auditors/async_copy_validation.py')]}


def adapter_dependencies(repo,contract,manifest):
    """Conservative project import/include closure, including function-local imports."""
    module_path=CAMPAIGN+'/'+manifest['module'].replace('.','/')+'.py'
    required=local_cpp_closure(repo,{contract['source'],*contract['dependencies']})
    pending=[module_path];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);required.add(name)
        for node in ast.walk(ast.parse(relative(repo,name).read_text())):
            imports=[]
            if isinstance(node,ast.Import):imports=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                require(node.level==0,'relative adapter imports need an explicit supported closure')
                if node.module in ('common','auditors','runners'):
                    imports=[node.module+'.'+a.name for a in node.names]
                elif node.module:imports=[node.module]
            for imported in imports:
                if imported.split('.')[0] in ('common','auditors','runners'):
                    pending.append(CAMPAIGN+'/'+imported.replace('.','/')+'.py')
            if isinstance(node,ast.Call):
                require(not (isinstance(node.func,ast.Name) and node.func.id=='__import__')
                        and not (isinstance(node.func,ast.Attribute) and node.func.attr=='import_module'),
                        'dynamic adapter imports require an explicit supported closure')
    return required


def review_requirements(contract,manifest,policy):
    requested=[{'stage':s,'phase':p,'path':n} for s,p,n in required_reviews(contract,True,case_diagnostic=True)]
    requested += [{'stage':'S02','phase':'early-validation-B','path':CORE_REVIEW},
                  *policy['review_dependencies'],*manifest['review_dependencies']]
    return sorted({(r['stage'],r['phase'],r['path']) for r in requested})


def authorized_dependencies(repo,reviewed,contract,manifest,paths):
    required=set(paths)|set(manifest['files_sha256'])|adapter_dependencies(repo,contract,manifest)
    source_gates={name for dep,review in reviewed
                  if dep['stage']==contract['stage'] and dep['phase']=='early-validation-source-B'
                  for name in gate_mapping(review)}
    require(required<=source_gates,'source-B omits probe/include/adapter dependency: '+str(sorted(required-source_gates)))
    core_gates={name for dep,review in reviewed
                if dep['stage']=='S02' and dep['phase']=='early-validation-B'
                for name in gate_mapping(review)}
    require(CORE_FILES<=core_gates,'core B omits runner or safety dependency')


def profile_from(document,profile_id,policy):
    require(document.get('schema_version')==1 and isinstance(document.get('profiles'),list),'profile document')
    matches=[p for p in document['profiles'] if p.get('id')==profile_id]
    require(len(matches)==1,'unknown/duplicate profile')
    profile=matches[0]
    require(set(policy['profile_required_fields'])<=set(profile),'incomplete executable profile')
    require(type(profile['maximum_target_launches']) is int and 1<=profile['maximum_target_launches']<=4,'short target launch bound')
    require(profile['target_iterations'] and all(type(i) is int and 0<=i<=64 for i in profile['target_iterations']),'short iteration bound')
    require(len(set(profile['target_iterations']))==len(profile['target_iterations']),'duplicate short lengths')
    require(isinstance(profile['input_profiles'],list) and profile['input_profiles'] and all(isinstance(x,str) and x for x in profile['input_profiles']),'input profiles')
    require(len(set(profile['input_profiles']))==len(profile['input_profiles']),'duplicate input profiles')
    require(len(profile['target_iterations'])*len(profile['input_profiles'])<=profile['maximum_target_launches'],'profile exceeds launch budget')
    require(type(profile['maximum_explicit_auxiliary_launches']) is int and profile['maximum_explicit_auxiliary_launches']>=0,'auxiliary launch bound')
    require(profile['output_evidence_kind'] in policy['output_evidence_kinds'],'output evidence kind')
    return profile


def manifest_check(repo,contract_path,profiles_path,manifest_path):
    repo=Path(repo);contract=read_json(contract_path);manifest=read_json(manifest_path)
    policy=read_json(relative(repo,POLICY))
    require(set(manifest)==set(policy['adapter_manifest_required_fields']),'adapter manifest fields')
    require(type(manifest['adapter_abi_version']) is int and manifest['adapter_abi_version']==1,'adapter ABI')
    require(isinstance(manifest['adapter_id'],str) and re.fullmatch(r'[a-z][a-z0-9_]*',manifest['adapter_id']),'adapter identity')
    require(manifest['contract_sha256']==sha(contract_path) and manifest['profiles_sha256']==sha(profiles_path),'contract/profile binding')
    require(manifest['exports']==policy['new_adapter_exports'],'adapter exports')
    module=manifest['module']
    require(isinstance(module,str) and re.fullmatch(r'auditors\.[a-z][a-z0-9_]*',module),'restricted adapter namespace')
    module_path=CAMPAIGN+'/'+module.replace('.','/')+'.py'
    require(module_path in manifest['files_sha256'],'adapter module unbound')
    verify_files(repo,manifest['files_sha256'])
    require(adapter_dependencies(repo,contract,manifest)<=set(manifest['files_sha256']),
            'manifest omits actual probe/include/adapter dependency')
    require(any(x['stage']==contract['stage'] and x['phase']=='early-validation-source-B' for x in manifest['review_dependencies']),'family diagnostic source review missing')
    return contract,manifest,policy


def case_from(contract,case_id):
    cases=[c for c in contract['cases'] if c['id']==case_id]
    require(len(cases)==1,'one existing case required for diagnostic')
    return cases[0]


def append_event(root,event,**fields):
    path=Path(root)/'journal.jsonl'
    with path.open('a') as out:
        out.write(json.dumps({'event':event,'utc':utc(),**fields},sort_keys=True,allow_nan=False)+'\n')
        out.flush();os.fsync(out.fileno())


def initialize(repo,suite,root,contract_path,profiles_path,manifest_path,case_id,profile_id,seed):
    repo,suite,root=map(lambda p:Path(p).resolve(),(repo,suite,root))
    require(root.parent.parent==suite,'output must be suite/family/run')
    require(type(seed) is int and 0<=seed<=4294967295,'seed domain')
    contract,manifest,policy=manifest_check(repo,contract_path,profiles_path,manifest_path)
    require(root.parent.name==contract['family'],'family output directory')
    case=case_from(contract,case_id)
    profiles=read_json(profiles_path)
    require(profiles.get('family')==contract['family'],'profile family')
    profile=profile_from(profiles,profile_id,policy)
    require(0 not in profile['target_iterations'] or case.get('exportable') is False,'zero iterations require explicit zero-work profile')
    additional=[{'stage':'S02','phase':'early-validation-B','path':CORE_REVIEW},*policy['review_dependencies'],*manifest['review_dependencies']]
    reviewed=[]
    for dep in additional:
        review=validate_gate(relative(suite,dep['path']),repo,dep['stage'],dep['phase'])
        reviewed.append((dep,review))
    # The family source gate must bind the profile, manifest and every adapter dependency.
    authorized_dependencies(repo,reviewed,contract,manifest,
        {str(Path(p).resolve().relative_to(repo)) for p in (contract_path,profiles_path,manifest_path)})
    with file_lock(suite_lock_path(suite)):
        require(not root.exists(),'new diagnostic directory required')
        env=inspect_allocation()
        root.mkdir(parents=True)
        for folder in ('environment','build','binary','attempts'):(root/folder).mkdir()
        frozen=freeze(repo,suite,root,contract_path,preflight=True,case_diagnostic=True)
        paths={ENTRY,POLICY,str(Path(profiles_path).resolve().relative_to(repo)),str(Path(manifest_path).resolve().relative_to(repo))}|set(manifest['files_sha256'])
        for _,review in reviewed:paths.update(gate_mapping(review))
        paths=local_cpp_closure(repo,paths)
        hashes=read_json(root/'snapshot/manifest.json')
        for name in sorted(paths):
            dest=root/'snapshot/repo'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(relative(repo,name),dest);hashes['repo/'+name]=sha(dest)
        for dep,review in reviewed:
            dst=root/'snapshot/suite'/dep['path'];dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(relative(suite,dep['path']),dst);hashes['suite/'+dep['path']]=sha(dst)
            binding={**dep,'sha256':sha(dst)}
            if binding not in frozen['reviews']:frozen['reviews'].append(binding)
        python_closure_check(repo,{k.removeprefix('repo/') for k in hashes if k.startswith('repo/')})
        atomic_json(root/'snapshot/manifest.json',hashes)
        frozen['snapshot_manifest_sha256']=sha(root/'snapshot/manifest.json')
        spec={'schema_version':2,'validation_schema_version':1,'kind':'case_diagnostic',
              'selection_scope':'case_diagnostic','family':contract['family'],'stage':contract['stage'],
              'case_id':case_id,'profile_id':profile_id,'seed':seed,'performance_eligible':False,'execution_root':str(root),
              'family_B3_eligible':False,'profiles_path':'snapshot/repo/'+str(Path(profiles_path).resolve().relative_to(repo)),
              'adapter_manifest_path':'snapshot/repo/'+str(Path(manifest_path).resolve().relative_to(repo)),**frozen}
        atomic_json(root/'validation_spec.json',spec)
        atomic_json(root/'environment/initial.json',env)
        atomic_json(root/'validation_state.json',{'status':'pending','selection_scope':'case_diagnostic'})
        append_event(root,'initialized',case_id=case_id,profile_id=profile_id)
    return root/'snapshot/repo'/ENTRY


def load(root):
    root=Path(root).resolve();spec=read_json(root/'validation_spec.json')
    require(spec['kind']=='case_diagnostic' and spec['selection_scope']=='case_diagnostic' and spec['family_B3_eligible'] is False and spec['performance_eligible'] is False,'diagnostic cannot grant family/performance qualification')
    contract=check_snapshot(root,spec)
    policy=read_json(relative(root,'snapshot/repo/'+POLICY))
    repo=root/'snapshot/repo'
    _,manifest,_=manifest_check(repo,relative(root,spec['contract_path']),relative(root,spec['profiles_path']),relative(root,spec['adapter_manifest_path']))
    expected=review_requirements(contract,manifest,policy)
    actual=[(r['stage'],r['phase'],r['path']) for r in spec['reviews']]
    require(len(actual)==len(set(actual)) and sorted(actual)==expected,'required review set changed or missing')
    reviewed=[]
    for stage,phase,name in expected:
        binding=next(r for r in spec['reviews'] if (r['stage'],r['phase'],r['path'])==(stage,phase,name))
        path=relative(root/'snapshot/suite',name)
        require(sha(path)==binding['sha256'],'review binding changed')
        review=validate_gate(path,repo,stage,phase)
        reviewed.append(({'stage':stage,'phase':phase,'path':name},review))
    authorized_dependencies(repo,reviewed,contract,manifest,
        {spec[key].removeprefix('snapshot/repo/') for key in ('contract_path','profiles_path','adapter_manifest_path')})
    profile=profile_from(read_json(relative(root,spec['profiles_path'])),spec['profile_id'],policy)
    case=case_from(contract,spec['case_id'])
    # Entrypoint dispatches into this frozen tree before this import.
    module=importlib.import_module(manifest['module'])
    expected=relative(repo,CAMPAIGN+'/'+manifest['module'].replace('.','/')+'.py')
    require(Path(module.__file__).resolve()==expected,'adapter loaded outside frozen tree')
    require(all(callable(getattr(module,name,None)) for name in manifest['exports']),'adapter callable ABI')
    require(module.ADAPTER_ID==manifest['adapter_id'] and module.ADAPTER_ABI_VERSION==1,'loaded adapter identity')
    require(all(callable(getattr(module,name,None)) for name in ('audit_sass','validate_device')),'family SASS/device checks missing before compile')
    return spec,contract,profile,case,module


def strict_lines(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate raw field');result[key]=value
        return result
    lines=Path(path).read_text().splitlines()
    require(len(lines)==2,'exactly device and validation JSONL lines required')
    values=[json.loads(line,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite raw JSON'))) for line in lines]
    require(values[0].get('type')=='device' and values[1].get('type')=='validation','validation is not trial/performance evidence')
    return values


def verify_output_artifacts(directory,row):
    kinds={'uint32':4,'uint16':2,'float32':4,'float64':8,'uint8':1}
    result={}
    for check in row['checks']:
        for item in check['output_artifacts']:
            require(set(item)=={'path','sha256','dtype','shape','evidence_kind'},'output artifact fields')
            require(item['dtype'] in kinds and item['shape'] and all(type(x) is int and x>0 for x in item['shape']),'output artifact type/shape')
            count=1
            for size in item['shape']:count*=size
            path=relative(directory,item['path'])
            require(path.stat().st_size==count*kinds[item['dtype']] and sha(path)==item['sha256'],'output artifact size/hash')
            require(item['path'] not in result,'output artifact reused across checks')
            result[item['path']]=item['sha256']
    return result


def validation_envelope(row,case,profile,seed,policy):
    """Core short-only invariants remain enforced even if an adapter has a bug."""
    require(set(row)==set(policy['validation_required_fields']),'validation row fields')
    require(type(row['schema_version']) is int and row['schema_version']==2
            and type(row['validation_schema_version']) is int and row['validation_schema_version']==1,'validation schema')
    require(row['type']=='validation' and row['case_id']==case['id']
            and row['profile_id']==profile['id'] and row['seed']==seed,'validation identity')
    require(all(row[key] is False for key in policy['fixed_false_fields']),'short validation cannot warm up, pilot or export performance')
    require(type(row['errors']) is int and row['errors']==0,'validation errors')
    require(row['scope']==case['scope'] and row['threads']==case['threads']
            and type(row['blocks']) is int and row['blocks']>0,'validation launch geometry')
    launches,checks=row['target_launches'],row['checks']
    require(isinstance(launches,list) and isinstance(checks,list) and 0<len(launches)<=profile['maximum_target_launches'] and len(checks)==len(launches),'validation target/check count')
    for collection in (launches,checks):
        require(all(type(item.get('launch_index')) is int for item in collection)
                and [item['launch_index'] for item in collection]==list(range(len(launches))),'launch/check identities and ordering')
    for launch,check in zip(launches,checks):
        require(set(launch)==set(policy['launch_fields']) and set(check)==set(policy['check_fields']),'per-launch fields')
        require(type(launch['iterations']) is int and launch['iterations'] in profile['target_iterations']
                and launch['input_profile'] in profile['input_profiles'],'declared short lengths/inputs')
        require(launch['threads']==row['threads'] and launch['blocks']==row['blocks'],'per-launch participants')
        require(check['completed'] is True and type(check['errors']) is int and check['errors']==0,'incomplete or incorrect target launch')
        require(type(check['checked_elements']) is int and type(check['expected_elements']) is int
                and check['checked_elements']==check['expected_elements'] and check['checked_elements']>=0,'full output coverage')
        require(isinstance(check['verified_CTA_ids'],list) and all(type(i) is int for i in check['verified_CTA_ids'])
                and sorted(check['verified_CTA_ids'])==list(range(row['blocks'])),'complete CTA participation')
    require(set(row['resource_identity'])==set(policy['resource_identity_fields']),'resource identity fields')


def attempt_evidence(root,spec,profile,case,module,directory):
    receipt=read_json(directory/'receipt.json')
    require(process_ok(receipt),'diagnostic process failed; never import as success')
    require(receipt['stdout_sha256']==sha(directory/'raw.jsonl') and receipt['stderr_sha256']==sha(directory/'stderr'),'diagnostic output hash')
    expected=['validate-only',case['id'],profile['id'],str(spec['seed'])]
    require(receipt['argv']==[str(Path(spec['execution_root'])/'binary/probe'),*expected],'short-only command identity')
    require(type(receipt.get('timeout_seconds')) is int and receipt['timeout_seconds']==30 and type(receipt.get('returncode')) is int,'short process receipt types/budget')
    require(type(receipt.get('host_start_ns')) is int and type(receipt.get('host_stop_ns')) is int and receipt['host_stop_ns']>=receipt['host_start_ns'],'process time order')
    registration=receipt.get('gpu_process_registration',{})
    require(registration.get('state')=='cleanup_confirmed','GPU registration cleanup')
    require(type(receipt.get('pid')) is int and receipt['pid']>0 and receipt.get('pgid')==receipt['pid']
            and registration.get('pid')==receipt['pid'] and registration.get('pgid')==receipt['pgid'],
            'registered process/group identity')
    device,row=strict_lines(directory/'raw.jsonl')
    require(device==read_json(root/'environment/device.json'),'raw device mismatch')
    require(registration.get('gpu_uuid')==device['uuid'].lower() and registration.get('argv')==receipt['argv'],'process registration identity')
    cleanup=registration.get('cleanup_receipt',{})
    require(cleanup=={k:v for k,v in receipt.items() if k!='gpu_process_registration'},'registered cleanup receipt')
    policy=read_json(relative(root,'snapshot/repo/'+POLICY))
    validation_envelope(row,case,profile,spec['seed'],policy)
    result=module.validate_validation(device,row,case,profile,spec['seed'])
    require(set(result)==set(policy['validation_result_fields']) and result.get('status')=='pass'
            and result.get('performance_eligible') is False and result['case_id']==case['id']
            and result['profile_id']==profile['id'],'adapter result qualification')
    require(type(result['target_launches']) is int and result['target_launches']==len(row['target_launches'])
            and result['checked_elements']==sum(c['checked_elements'] for c in row['checks'])
            and result['output_evidence_kind']==profile['output_evidence_kind'],'adapter result coverage')
    artifacts=verify_output_artifacts(directory,row)
    return {'result':result,'receipt':str((directory/'receipt.json').relative_to(root)),
            'receipt_sha256':sha(directory/'receipt.json'),'raw_sha256':sha(directory/'raw.jsonl'),
            'output_artifacts':artifacts}


def compile_probe(root,spec,contract,module,env):
    if 'binary_sha256' in spec:
        require(sha(root/'binary/probe')==spec['binary_sha256'],'retained diagnostic binary changed')
        return
    source=relative(root,spec['source_path'])
    command=[env['tools']['nvcc']['path'],*contract['build']['flags'],'-MD','-MF',str(root/'build/probe.d'),str(source),'-o',str(root/'binary/probe')]
    run_command(root,'build/compile',command,180,env,cwd=root/'snapshot/repo')
    inventory_depfile(root,root/'build/probe.d',toolchain_roots=[Path(env['tools']['nvcc']['path']).resolve().parent.parent])
    run_command(root,'build/sass',[env['tools']['cuobjdump']['path'],'--dump-sass',str(root/'binary/probe')],120,env)
    require(callable(getattr(module,'audit_sass',None)),'validation adapter must expose family SASS audit')
    checks=module.audit_sass((root/'build/sass.stdout').read_text(),contract)
    atomic_json(root/'build/sass_audit.json',checks)
    run_command(root,'build/ldd',['ldd',str(root/'binary/probe')],120,env)
    libraries={str(Path(token).resolve()):sha(token) for line in (root/'build/ldd.stdout').read_text().splitlines() for token in line.split() if token.startswith('/') and Path(token).is_file()}
    atomic_json(root/'build/shared_libraries.json',libraries)
    spec.update(binary_sha256=sha(root/'binary/probe'),sass_sha256=sha(root/'build/sass.stdout'),dependencies_sha256=sha(root/'build/dependencies.json'),shared_libraries_sha256=sha(root/'build/shared_libraries.json'))
    atomic_json(root/'validation_spec.json',spec)


def execute(root):
    root=Path(root).resolve();suite=root.parent.parent
    with file_lock(suite_lock_path(suite)):
        spec,contract,profile,case,module=load(root)
        require(root==Path(spec['execution_root']),'relocated archives are audit-only; retain original execution root for resume')
        status=read_json(root/'validation_state.json')['status']
        require(status in ('pending','running','checkpoint'),'failed or completed diagnostic requires audit; no blind rerun')
        env=inspect_allocation()
        require(environment_identity(env)==environment_identity(read_json(root/'environment/initial.json')),'environment drift')
        atomic_json(root/'environment'/('allocation_'+str(time.time_ns())+'.json'),env)
        try:
            compile_probe(root,spec,contract,module,env)
            verify_files(Path('/'),{str(Path(p).relative_to('/')):h for p,h in read_json(root/'build/shared_libraries.json').items()})
            with gpu_lock(env['uuid']) as recovered:
                if recovered:atomic_json(root/'environment'/('recovery_'+str(time.time_ns())+'.json'),recovered)
                verify_interrupted_gpu_groups(root,env)
                name='environment/device_'+str(time.time_ns())
                run_command(root,name,[str(root/'binary/probe'),'device'],30,env,gpu=True)
                device=read_json(root/(name+'.stdout'))
                require(callable(getattr(module,'validate_device',None)),'validation adapter must expose device validation')
                module.validate_device(device)
                require(device['uuid'].lower()==env['uuid'].lower(),'device UUID mismatch')
                if (root/'environment/device.json').exists():require(device==read_json(root/'environment/device.json'),'runtime device drift')
                else:atomic_json(root/'environment/device.json',device)
                folders=sorted((root/'attempts').glob('attempt_*'))
                for prior in folders:
                    receipt=prior/'receipt.json'
                    if receipt.exists():
                        evidence=attempt_evidence(root,spec,profile,case,module,prior)
                        return seal(root,spec,evidence)
                    append_event(root,'interrupted_attempt_retained',path=str(prior.relative_to(root)))
                budget(env,30)
                directory=root/'attempts'/f'attempt_{len(folders):02d}';directory.mkdir()
                argv=module.validation_argv('binary/probe',case,profile,spec['seed'])
                require(argv==['binary/probe','validate-only',case['id'],profile['id'],str(spec['seed'])],'adapter attempted nonvalidation command')
                atomic_json(root/'validation_state.json',{'status':'running','attempt':directory.name,'selection_scope':'case_diagnostic'})
                append_event(root,'launch_started',case_id=case['id'],attempt=directory.name)
                receipt=bounded([str(root/'binary/probe'),*argv[1:]],directory,directory/'raw.jsonl',directory/'stderr',30,uuid=env['uuid'])
                atomic_json(directory/'receipt.json',receipt)
                require(process_ok(receipt),'diagnostic failed; raw/stderr preserved')
                evidence=attempt_evidence(root,spec,profile,case,module,directory)
            return seal(root,spec,evidence)
        except Checkpoint as error:
            atomic_json(root/'validation_state.json',{'status':'checkpoint','reason':str(error),'selection_scope':'case_diagnostic'})
            append_event(root,'checkpoint',reason=str(error));return 75
        except Exception as error:
            atomic_json(root/'validation_state.json',{'status':'failed','reason':str(error),'selection_scope':'case_diagnostic'})
            append_event(root,'failed',reason=str(error))
            atomic_json(root/'failure_summary.json',{'status':'case_diagnostic_failed','reason':str(error),
                        'case_id':spec['case_id'],'performance_eligible':False,'family_B3_eligible':False})
            mapping={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()
                     and p.name not in ('failure_manifest.json','validation_state.json')}
            atomic_json(root/'failure_manifest.json',mapping)
            raise


def seal(root,spec,evidence):
    atomic_json(root/'diagnostic_summary.json',{'status':'case_diagnostic_passed','case_id':spec['case_id'],'performance_eligible':False,'family_B3_eligible':False,'evidence':evidence})
    append_event(root,'case_diagnostic_passed',case_id=spec['case_id'])
    files={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name not in ('validation_manifest.json','validation_state.json')}
    atomic_json(root/'validation_manifest.json',files)
    atomic_json(root/'validation_state.json',{'status':'case_diagnostic_passed','selection_scope':'case_diagnostic','family_B3_eligible':False})
    return 0


def audit(root):
    root=Path(root).resolve();spec,contract,profile,case,module=load(root)
    if (root/'failure_summary.json').exists():
        manifest=read_json(root/'failure_manifest.json');verify_files(root,manifest)
        expected_paths={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
                        and p.name not in ('failure_manifest.json','validation_state.json')}
        require(set(manifest)==expected_paths,'failure evidence manifest coverage')
        summary=read_json(root/'failure_summary.json')
        require(summary['status']=='case_diagnostic_failed' and summary['case_id']==spec['case_id']
                and summary['performance_eligible'] is False and summary['family_B3_eligible'] is False,'failure cannot grant qualification')
        for receipt_path in (root/'attempts').glob('attempt_*/receipt.json'):
            receipt=read_json(receipt_path)
            require(sha(receipt_path.parent/'raw.jsonl')==receipt['stdout_sha256']
                    and sha(receipt_path.parent/'stderr')==receipt['stderr_sha256'],'failed process output hashes')
        return {**summary,'artifact_integrity':'verified','GPU_correctness':'failed_or_unproven'}
    require(sha(root/'binary/probe')==spec['binary_sha256'] and sha(root/'build/sass.stdout')==spec['sass_sha256'],'build identity')
    require(sha(root/'build/dependencies.json')==spec['dependencies_sha256'],'compiled dependencies')
    summary=read_json(root/'diagnostic_summary.json')
    directory=relative(root,summary['evidence']['receipt']).parent
    expected={'status':'case_diagnostic_passed','case_id':spec['case_id'],'performance_eligible':False,'family_B3_eligible':False,'evidence':attempt_evidence(root,spec,profile,case,module,directory)}
    require(summary==expected,'diagnostic summary mismatch')
    manifest=read_json(root/'validation_manifest.json');verify_files(root,manifest)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name not in ('validation_manifest.json','validation_state.json')}
    require(set(manifest)==actual,'diagnostic manifest coverage')
    return summary
