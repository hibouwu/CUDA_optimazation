"""Single-launch, full-output S08 observation. No numerical/performance promotion.

Execution uses the reviewed process/lock primitives. Audit reads only relative
archive paths; original absolute paths are retained solely as receipt identity.
"""
from __future__ import annotations
import ast
import json
import os
from pathlib import Path
import shutil
import time

from common.suite_io import (atomic_json, bounded, file_lock, gate_mapping, gpu_lock,
    process_ok, read_json, relative, require, sha, suite_lock_path, utc,
    validate_gate, verify_files)
from common.suite_snapshot import (CAMPAIGN, check_snapshot, freeze,
    local_cpp_closure, python_closure_check, required_reviews)
from runners.environment import Checkpoint, budget, environment_identity, inspect_allocation
from runners.suite_runner import compile_probe, run_command, verify_interrupted_gpu_groups

KIND = 'long_accumulation_diagnostic'
ENTRY = CAMPAIGN+'/run_accumulation_diagnostic.py'
BASE_CONTRACT = CAMPAIGN+'/contracts/low_precision_lowering_v3.json'
DIAGNOSTIC = CAMPAIGN+'/contracts/low_precision_accumulation_diagnostic_v1.json'
WRAPPER = CAMPAIGN+'/probes/diagnostics/low_precision_accumulation.cu'
AUDITOR = CAMPAIGN+'/auditors/low_precision_accumulation.py'
BASELINE = 'low_precision/diagnostic-wgmma-e4m3-v2'
EXTRA_REVIEWS = [('S08','long-accumulation-A','reviews/S08-long-accumulation-A-review.json'),
                 ('S08','long-accumulation-source-B','reviews/S08-long-accumulation-source-B-review.json')]
DIAGNOSTIC_SHA256 = 'd9c2f883c4acfb0e4908a4e2b222219e156bed17ed48754ea52fd39b712adaf2'
FALSE_FLAGS = ('performance_eligible','family_B3_eligible','numerical_kernel_qualification')
BASELINE_BINDINGS = ('validation_spec.json','validation_manifest.json','snapshot/manifest.json',
    'environment/initial.json','environment/device.json','build/sass.stdout',
    'build/dependencies.json','build/shared_libraries.json','attempts/attempt_00/raw.jsonl',
    'attempts/attempt_00/receipt.json','attempts/attempt_00/stderr','binary/probe')


def python_closure(repo, paths):
    """Static project imports plus the explicitly selected original family auditor."""
    paths=set(paths);pending=list(paths)
    while pending:
        name=pending.pop()
        if not name.endswith('.py'):continue
        for node in ast.walk(ast.parse(relative(repo,name).read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[v.name for v in node.names]
            elif isinstance(node,ast.ImportFrom):
                require(node.level==0,'relative imports unsupported in diagnostic closure')
                if node.module in ('common','runners','auditors'):modules=[node.module+'.'+v.name for v in node.names]
                elif node.module:modules=[node.module]
            for module in modules:
                if module.split('.')[0] not in ('common','runners','auditors'):continue
                dep=CAMPAIGN+'/'+module.replace('.','/')+'.py'
                if dep not in paths:paths.add(dep);pending.append(dep)
    return paths


def source_dependencies(repo):
    return local_cpp_closure(repo,python_closure(repo,{ENTRY,WRAPPER,AUDITOR,BASE_CONTRACT,DIAGNOSTIC,
        CAMPAIGN+'/auditors/low_precision.py',CAMPAIGN+'/auditors/low_precision_validation.py',
        CAMPAIGN+'/tests/test_accumulation_diagnostic.py',CAMPAIGN+'/tests/test_low_precision_accumulation.py'}))


def review_requirements(base):
    return sorted(set(required_reviews(base,True)+EXTRA_REVIEWS))


def required_source_bindings(repo,suite):
    prefix=Path(suite).resolve().relative_to(Path(repo).resolve()).as_posix()+'/'+BASELINE+'/'
    return source_dependencies(repo)|{prefix+name for name in BASELINE_BINDINGS}


def verify_authorization(repo,suite,base,bindings=None):
    expected=review_requirements(base)
    if bindings is not None:
        actual=[(r['stage'],r['phase'],r['path']) for r in bindings]
        require(len(actual)==len(set(actual)) and sorted(actual)==expected,'mandatory review set missing or changed')
    reviewed=[]
    for stage,phase,name in expected:
        path=relative(suite,name)
        if bindings is not None:
            saved=next(r for r in bindings if (r['stage'],r['phase'],r['path'])==(stage,phase,name))
            require(sha(path)==saved['sha256'],'review binding changed')
        review=validate_gate(path,repo,stage,phase)
        reviewed.append(({'stage':stage,'phase':phase,'path':name,'sha256':sha(path)},review))
    return reviewed


def same_json(left,right):
    return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)


def strict_rows(path,second_type):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON field');result[key]=value
        return result
    lines=Path(path).read_text().splitlines()
    require(len(lines)==2,'exactly two stdout JSONL rows required')
    rows=[json.loads(line,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON'))) for line in lines]
    require(all(isinstance(r,dict) for r in rows) and rows[0].get('type')=='device' and rows[1].get('type')==second_type,'diagnostic stdout type')
    return rows


def baseline_evidence(root):
    """Verify the full old archive and snapshot without executing old code."""
    spec=read_json(relative(root,'validation_spec.json'))
    require(spec.get('kind')=='case_diagnostic' and spec.get('case_id')=='wgmma_e4m3_g1_one_cta','baseline short identity')
    base=check_snapshot(root,spec)
    manifest=read_json(relative(root,'validation_manifest.json'));verify_files(root,manifest)
    actual={p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file() and p.name not in ('validation_manifest.json','validation_state.json')}
    require(set(manifest)==actual,'baseline archive manifest coverage')
    for key,name in (('binary_sha256','binary/probe'),('sass_sha256','build/sass.stdout'),
                     ('dependencies_sha256','build/dependencies.json'),('shared_libraries_sha256','build/shared_libraries.json')):
        require(sha(relative(root,name))==spec[key],'baseline build identity')
    device,row=strict_rows(relative(root,'attempts/attempt_00/raw.jsonl'),'validation')
    require(device==read_json(relative(root,'environment/device.json')),'baseline raw device')
    receipt=read_json(relative(root,'attempts/attempt_00/receipt.json'))
    verify_receipt(receipt,root/'attempts/attempt_00/raw.jsonl',root/'attempts/attempt_00/stderr',
        [str(Path(spec['execution_root'])/'binary/probe'),'validate-only',spec['case_id'],spec['profile_id'],str(spec['seed'])],30,device['uuid'])
    from auditors.low_precision_validation import validate_validation
    case=next(c for c in base['cases'] if c['id']==spec['case_id'])
    profile=read_json(relative(root,spec['profiles_path']))['profiles'][0]
    validate_validation(device,row,case,profile,spec['seed'])
    return spec,base,device,row['resource_identity']


def verify_receipt(receipt,stdout,stderr,argv,timeout,uuid):
    require(type(receipt.get('returncode')) is int and receipt.get('timed_out') is False and receipt.get('cleanup_confirmed') is True and process_ok(receipt),'unsuccessful command cannot be imported')
    require(receipt.get('argv')==argv and type(receipt.get('timeout_seconds')) is int and receipt['timeout_seconds']==timeout,'receipt argv/timeout identity')
    require(receipt.get('stdout_sha256')==sha(stdout) and receipt.get('stderr_sha256')==sha(stderr),'process output hash')
    require(type(receipt.get('host_start_ns')) is int and type(receipt.get('host_stop_ns')) is int and receipt['host_stop_ns']>=receipt['host_start_ns'],'receipt time order')
    registration=receipt.get('gpu_process_registration',{})
    require(registration.get('state')=='cleanup_confirmed' and registration.get('gpu_uuid')==uuid.lower() and registration.get('argv')==argv,'registered cleanup/UUID/argv')
    require(type(receipt.get('pid')) is int and receipt['pid']>0 and receipt.get('pgid')==receipt['pid']
            and registration.get('pid')==receipt['pid'] and registration.get('pgid')==receipt['pgid'],'registered process identity')
    require(registration.get('cleanup_receipt')=={k:v for k,v in receipt.items() if k!='gpu_process_registration'},'registered cleanup receipt binding')


def append_event(root,event,**fields):
    with (root/'journal.jsonl').open('a') as stream:
        stream.write(json.dumps({'utc':utc(),'event':event,**fields},sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())


def initialize(repo,suite,root):
    repo,suite,root=map(lambda p:Path(p).resolve(),(repo,suite,root))
    require(root.parent.parent==suite and root.parent.name=='low_precision','output must be suite/low_precision/new-ID')
    require(sha(relative(repo,DIAGNOSTIC))==DIAGNOSTIC_SHA256,'fixed A diagnostic contract changed')
    base=read_json(relative(repo,BASE_CONTRACT));reviewed=verify_authorization(repo,suite,base)
    required=required_source_bindings(repo,suite)
    source_review=next(review for dep,review in reviewed if dep['phase']=='long-accumulation-source-B')
    require(required<=set(gate_mapping(source_review)),'source-B omits diagnostic code/dependency/baseline binding')
    baseline=suite/BASELINE
    baseline_evidence(baseline)
    baseline_env=read_json(relative(baseline,'environment/initial.json'))
    with file_lock(suite_lock_path(suite)):
        require(not root.exists(),'new diagnostic run ID required; never overwrite/rerun completed evidence')
        env=inspect_allocation()
        require(environment_identity(env)==environment_identity(baseline_env),'environment differs from original baseline')
        root.mkdir(parents=True)
        for name in ('environment','build','binary','attempts'):(root/name).mkdir()
        frozen=freeze(repo,suite,root,repo/BASE_CONTRACT,preflight=True)
        paths=required|{ENTRY}
        for _,review in reviewed:paths.update(gate_mapping(review))
        # Preserve the complete original archive so all nested snapshot/hash
        # checks remain independently reproducible after relocation.
        paths|={p.relative_to(repo).as_posix() for p in baseline.rglob('*') if p.is_file()}
        paths=local_cpp_closure(repo,paths)
        mapping=read_json(root/'snapshot/manifest.json')
        for name in sorted(paths):
            dest=root/'snapshot/repo'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(relative(repo,name),dest);mapping['repo/'+name]=sha(dest)
        bindings=[]
        for dep,_ in reviewed:
            dest=root/'snapshot/suite'/dep['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(relative(suite,dep['path']),dest);mapping['suite/'+dep['path']]=sha(dest);bindings.append(dep)
        python_closure_check(repo,{k.removeprefix('repo/') for k in mapping if k.startswith('repo/'+CAMPAIGN+'/')})
        atomic_json(root/'snapshot/manifest.json',mapping)
        spec={**frozen,'schema_version':2,'kind':KIND,'fixture':False,'family':'low_precision','stage':'S08',
              'case_id':'wgmma_e4m3_g1_one_cta','profile_id':'wgmma_e4m3_uniform_8192_full_output',
              'execution_root':str(root),'source_path':'snapshot/repo/'+WRAPPER,
              'diagnostic_path':'snapshot/repo/'+DIAGNOSTIC,
              'baseline_path':'snapshot/repo/'+baseline.relative_to(repo).as_posix(),
              'required_source_bindings':sorted(required),'reviews':bindings,
              'snapshot_manifest_sha256':sha(root/'snapshot/manifest.json'),
              'environment_sha256':None,**{key:False for key in FALSE_FLAGS}}
        atomic_json(root/'environment/initial.json',env);spec['environment_sha256']=sha(root/'environment/initial.json')
        atomic_json(root/'run_spec.json',spec)
        atomic_json(root/'diagnostic_state.json',{'status':'pending','kind':KIND})
        append_event(root,'initialized')
    return root/'snapshot/repo'/ENTRY


def load(root):
    root=Path(root).resolve();spec=read_json(relative(root,'run_spec.json'))
    require(spec.get('schema_version')==2 and spec.get('kind')==KIND and spec.get('fixture') is False,'long diagnostic identity')
    require(all(spec.get(key) is False for key in FALSE_FLAGS),'diagnostic cannot grant numerical/performance/B3 qualification')
    require(spec.get('source_path')=='snapshot/repo/'+WRAPPER and spec.get('contract_path')=='snapshot/repo/'+BASE_CONTRACT
            and spec.get('diagnostic_path')=='snapshot/repo/'+DIAGNOSTIC,'fixed compiler source/contract paths')
    require(spec.get('case_id')=='wgmma_e4m3_g1_one_cta' and spec.get('profile_id')=='wgmma_e4m3_uniform_8192_full_output','unique long diagnostic coordinate')
    base=check_snapshot(root,spec);repo=root/'snapshot/repo'
    require(sha(relative(root,spec['diagnostic_path']))==DIAGNOSTIC_SHA256,'frozen A contract identity')
    diagnostic=read_json(relative(root,spec['diagnostic_path']))
    reviewed=verify_authorization(repo,root/'snapshot/suite',base,spec['reviews'])
    baseline_path=Path(spec['baseline_path'])
    require(baseline_path.as_posix().startswith('snapshot/repo/') and baseline_path.as_posix().endswith('/'+BASELINE),'fixed baseline relative path')
    baseline=root/baseline_path
    relative(root,baseline_path.as_posix()+'/validation_spec.json')
    prefix=baseline.relative_to(repo).as_posix()+'/'
    required=source_dependencies(repo)|{prefix+name for name in BASELINE_BINDINGS}
    require(spec.get('required_source_bindings')==sorted(required),'required source binding set changed')
    review=next(r for d,r in reviewed if d['phase']=='long-accumulation-source-B')
    require(required<=set(gate_mapping(review)),'source-B incomplete dependency authorization')
    baseline_spec,baseline_base,device,resource=baseline_evidence(baseline)
    require(base==baseline_base,'original base contract changed')
    for name in local_cpp_closure(repo,{base['source'],*base['dependencies']}):
        require(sha(relative(repo,name))==sha(relative(baseline,'snapshot/repo/'+name)),'target source/include changed from baseline')
    require(sha(relative(repo,base['source']))==diagnostic['source_reuse']['sha256'],'original target source identity')
    require(sha(relative(root,'environment/initial.json'))==spec['environment_sha256'],'initial environment changed')
    require(environment_identity(read_json(root/'environment/initial.json'))==environment_identity(read_json(baseline/'environment/initial.json')),'original environment equivalence')
    from auditors import low_precision_accumulation as module
    require(Path(module.__file__).resolve()==relative(repo,AUDITOR),'auditor must load from frozen tree')
    return spec,base,diagnostic,baseline,device,resource,module


def verify_build(root,spec,base,baseline,module,live=False):
    for key,name in (('binary_sha256','binary/probe'),('sass_sha256','build/sass.txt'),
                     ('dependencies_sha256','build/dependencies.json'),('shared_libraries_sha256','build/shared_libraries.json')):
        require(sha(relative(root,name))==spec[key],'diagnostic build identity')
    env=read_json(root/'environment/initial.json')
    command=read_json(root/'build/command.json')
    original=Path(spec['execution_root']);flags=base['build']['flags']
    expected=[env['tools']['nvcc']['path'],*flags,'-MD','-MF',str(original/'build/probe.d'),str(original/spec['source_path']),'-o',str(original/'binary/probe')]
    require(command=={'argv':expected,'source_relative':spec['source_path'],'target_relative':'binary/probe'},'compile source/flags identity')
    for name,argv,timeout in [('compile',expected,180),('disassemble',[env['tools']['cuobjdump']['path'],'--dump-sass',str(original/'binary/probe')],120),('ldd',['ldd',str(original/'binary/probe')],120)]:
        receipt=read_json(root/'build'/f'{name}.receipt.json')
        require(process_ok(receipt) and receipt.get('argv')==argv and receipt.get('timeout_seconds')==timeout,'build command receipt')
        require(receipt['stdout_sha256']==sha(root/'build'/f'{name}.stdout') and receipt['stderr_sha256']==sha(root/'build'/f'{name}.stderr'),'build command hashes')
    from auditors.low_precision import audit_sass
    text=(root/'build/sass.txt').read_text()
    require(read_json(root/'build/sass_audit.json')==audit_sass(text,base),'full original family SASS audit')
    comparison=module.compare_target_sass(text,(baseline/'build/sass.stdout').read_text(),'lp_wgmma_e4m3_g1')
    deps=read_json(root/'build/dependencies.json');old=read_json(baseline/'build/dependencies.json')
    mapping=read_json(root/'snapshot/manifest.json')
    require(set(old['local'])<=set(deps['local']),'target local compiler dependency omitted')
    for name in deps['local']:
        require(name in mapping and sha(relative(root/'snapshot',name))==mapping[name],'compiled local dependency hash')
    require('repo/'+WRAPPER in deps['local'],'wrapper missing from compiled dependency inventory')
    require(all(deps['external_toolchain'].get(p)==h for p,h in old['external_toolchain'].items()),'original external toolchain dependency changed')
    libraries=read_json(root/'build/shared_libraries.json')
    require(libraries==read_json(baseline/'build/shared_libraries.json'),'shared library identity differs from baseline')
    if live:
        verify_files(Path('/'),{str(Path(p).relative_to('/')):h for p,h in {**deps['external_toolchain'],**libraries}.items()})
    return comparison


def attempt_evidence(root,spec,contract,baseline_device,baseline_resource,module,directory):
    device,row=strict_rows(directory/'raw.jsonl',KIND)
    require(same_json(device,read_json(root/'environment/device.json')) and same_json(device,baseline_device),'collection device differs from prelaunch/baseline')
    receipt=read_json(directory/'receipt.json')
    verify_receipt(receipt,directory/'raw.jsonl',directory/'stderr',[str(Path(spec['execution_root'])/'binary/probe'),'collect'],120,device['uuid'])
    result=module.audit_values(device,row,contract,baseline_device,baseline_resource)
    require(result.get('status') in ('collected_exact_mathematical_match','collected_finite_difference'),'diagnostic auditor result')
    return {'result':result,'receipt':str((directory/'receipt.json').relative_to(root)),
            'receipt_sha256':sha(directory/'receipt.json'),'raw_sha256':sha(directory/'raw.jsonl')}


def evidence_manifest(root,excluded):
    return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name not in excluded}


def seal(root,spec,evidence):
    summary={'status':'diagnostic-collected','kind':KIND,'case_id':spec['case_id'],**{k:False for k in FALSE_FLAGS},'evidence':evidence}
    atomic_json(root/'diagnostic_summary.json',summary);append_event(root,'diagnostic-collected')
    atomic_json(root/'diagnostic_manifest.json',evidence_manifest(root,{'diagnostic_manifest.json','diagnostic_state.json'}))
    atomic_json(root/'diagnostic_state.json',{'status':'diagnostic-collected','kind':KIND})
    return 0


def execute(root):
    root=Path(root).resolve()
    with file_lock(suite_lock_path(root.parent.parent)):
        spec,base,contract,baseline,baseline_device,baseline_resource,module=load(root)
        require(root==Path(spec['execution_root']),'relocated archive is audit-only')
        status=read_json(root/'diagnostic_state.json')['status']
        require(status in ('pending','running','checkpoint') and not (root/'diagnostic_summary.json').exists() and not (root/'failure_summary.json').exists(),'failed/completed evidence cannot be rerun')
        env=inspect_allocation()
        require(environment_identity(env)==environment_identity(read_json(root/'environment/initial.json')),'environment drift')
        atomic_json(root/'environment'/('allocation_'+str(time.time_ns())+'.json'),env)
        try:
            if 'binary_sha256' not in spec:compile_probe(root,spec,base,env)
            # Revalidate immutable source/review identities after the compiler.
            load(root)
            comparison=verify_build(root,spec,base,baseline,module,live=True)
            atomic_json(root/'build/target_sass_equivalence.json',comparison)
            with gpu_lock(env['uuid']) as recovered:
                if recovered:atomic_json(root/'environment'/('recovery_'+str(time.time_ns())+'.json'),recovered)
                verify_interrupted_gpu_groups(root,env)
                folders=sorted((root/'attempts').glob('attempt_*'))
                # Do not launch anything after an already completed collection.
                for folder in folders:
                    if (folder/'receipt.json').exists():
                        evidence=attempt_evidence(root,spec,contract,baseline_device,baseline_resource,module,folder)
                        return seal(root,spec,evidence)
                    append_event(root,'interrupted_attempt_retained',path=str(folder.relative_to(root)))
                name='environment/device_'+str(time.time_ns())
                receipt=run_command(root,name,[str(root/'binary/probe'),'device'],30,env,gpu=True)
                device,resource=strict_rows(root/(name+'.stdout'),'resource')
                require(set(resource)=={'type','resource_identity'} and same_json(device,baseline_device) and same_json(resource['resource_identity'],baseline_resource),'target resource/device differs before launch')
                require(device['uuid'].lower()==env['uuid'].lower(),'UUID lock/device mismatch')
                verify_receipt(receipt,root/(name+'.stdout'),root/(name+'.stderr'),[str(root/'binary/probe'),'device'],30,env['uuid'])
                atomic_json(root/'environment/device.json',device)
                atomic_json(root/'environment/resource.json',resource)
                atomic_json(root/'environment/device_query.json',{'prefix':name,'receipt_sha256':sha(root/(name+'.receipt.json'))})
                # Fresh budget and immutable inputs immediately before the sole target.
                load(root);verify_build(root,spec,base,baseline,module,live=True);budget(env,120)
                folder=root/'attempts'/f'attempt_{len(folders):02d}';folder.mkdir()
                atomic_json(root/'diagnostic_state.json',{'status':'running','kind':KIND,'attempt':folder.name})
                append_event(root,'collect_started',attempt=folder.name)
                receipt=bounded([str(root/'binary/probe'),'collect'],folder,folder/'raw.jsonl',folder/'stderr',120,uuid=env['uuid'])
                atomic_json(folder/'receipt.json',receipt)
                require(process_ok(receipt),'collection failed; full raw/stderr retained')
                evidence=attempt_evidence(root,spec,contract,baseline_device,baseline_resource,module,folder)
            return seal(root,spec,evidence)
        except Checkpoint as error:
            atomic_json(root/'diagnostic_state.json',{'status':'checkpoint','kind':KIND,'reason':str(error)})
            append_event(root,'checkpoint',reason=str(error));return 75
        except Exception as error:
            atomic_json(root/'diagnostic_state.json',{'status':'failed','kind':KIND,'reason':str(error)})
            append_event(root,'failed',reason=str(error))
            atomic_json(root/'failure_summary.json',{'status':'diagnostic-failed','kind':KIND,'reason':str(error),**{k:False for k in FALSE_FLAGS}})
            atomic_json(root/'failure_manifest.json',evidence_manifest(root,{'failure_manifest.json','diagnostic_state.json'}))
            raise


def audit(root):
    root=Path(root).resolve();spec,base,contract,baseline,device,resource,module=load(root)
    if (root/'failure_summary.json').exists():
        manifest=read_json(root/'failure_manifest.json');verify_files(root,manifest)
        require(manifest==evidence_manifest(root,{'failure_manifest.json','diagnostic_state.json'}),'failure manifest coverage')
        summary=read_json(root/'failure_summary.json')
        require(summary.get('status')=='diagnostic-failed' and summary.get('kind')==KIND and all(summary.get(k) is False for k in FALSE_FLAGS),'failure qualification')
        for path in (root/'attempts').glob('attempt_*/receipt.json'):
            receipt=read_json(path)
            require(receipt['stdout_sha256']==sha(path.parent/'raw.jsonl') and receipt['stderr_sha256']==sha(path.parent/'stderr'),'failed output hashes')
        return summary
    comparison=verify_build(root,spec,base,baseline,module)
    require(comparison==read_json(root/'build/target_sass_equivalence.json'),'SASS equivalence replay')
    query=read_json(root/'environment/device_query.json');prefix=query['prefix']
    require(prefix.startswith('environment/device_') and '/' not in prefix.removeprefix('environment/'),'device query path')
    receipt_path=relative(root,prefix+'.receipt.json');require(sha(receipt_path)==query['receipt_sha256'],'prelaunch resource query receipt')
    raw_device,raw_resource=strict_rows(relative(root,prefix+'.stdout'),'resource')
    require(same_json(raw_device,device) and same_json(raw_resource,{'type':'resource','resource_identity':resource}),'prelaunch resource equivalence')
    verify_receipt(read_json(receipt_path),relative(root,prefix+'.stdout'),relative(root,prefix+'.stderr'),[str(Path(spec['execution_root'])/'binary/probe'),'device'],30,device['uuid'])
    summary=read_json(root/'diagnostic_summary.json');folder=relative(root,summary['evidence']['receipt']).parent
    expected={'status':'diagnostic-collected','kind':KIND,'case_id':spec['case_id'],**{k:False for k in FALSE_FLAGS},
              'evidence':attempt_evidence(root,spec,contract,device,resource,module,folder)}
    require(summary==expected,'complete values summary replay mismatch')
    manifest=read_json(root/'diagnostic_manifest.json');verify_files(root,manifest)
    require(manifest==evidence_manifest(root,{'diagnostic_manifest.json','diagnostic_state.json'}),'collection manifest coverage')
    return summary
