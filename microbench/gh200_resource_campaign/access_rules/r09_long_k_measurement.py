#!/usr/bin/env python3
"""Authorized measurement-v2 preparation, CPU gate and one-card sampling.

The original ARM binaries/probes remain unchanged. This is calibration, not V10.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics as st
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

GPU = 'GPU-43269fbc-449d-3e0f-908a-9c81229546d3'
FROZEN = 'Q15-full-time/V09-component-validation/20261009-V09-freeze-job738972-r2'
ENDS = 'Q09-frequency/R07-R09-inkernel-clock/20261009-R09-input-clock-calibration-job738376'
MEASUREMENT = 'Q15-full-time/long-k-mainline/20261011-long-k-event-bounded/measurement-v2'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(base, out):
    if out.exists() and {p.name for p in out.iterdir()} != {'attempts'}:
        raise ValueError('new run or preserved preparation-attempt directory required')
    out.mkdir(parents=True, exist_ok=True)
    spec = read(base/MEASUREMENT/'conditions.json')
    refs = read(base/FROZEN/'inputs/reference-cases.json')
    g6 = [r for r in read(base/FROZEN/'cases.json') if r.get('group')=='G6_base']
    if (len(spec['new_conditions']),len(refs),len(g6),spec['total_planned_processes'])!=(18,6,3,660):
        raise ValueError('authorized matrix changed')
    rows=[]
    for old in spec['new_conditions']:
        row={k:old[k] for k in ('id','config','m','n','k','lda','ldb','ldd','input_mode','seed','swizzle','sm_count')}
        row.update(set='calib',kind='ordinary',storage_m=row['m'],storage_n=row['n'],
                   zero_m=-1,zero_n=-1,evict=0,group='new-long-k-calibration')
        rows.append(row)
    rows += refs+g6
    write(out/'cases.json',rows)
    schedule=[]
    for phase,chosen in (('reference',refs),('development',rows[:18]+g6)):
        for trial in range(10):
            ordered=chosen[:];random.Random(f'20261012-{phase}-{trial}').shuffle(ordered)
            for r in ordered:
                variants=['plain','dual']
                if r in rows[:18] and r['k'] in (2048,32768):variants.append('ends')
                random.Random(f'20261012-{r["id"]}-{trial}').shuffle(variants)
                schedule.extend(dict(phase=phase,case=r['id'],trial=trial,variant=v) for v in variants)
    if len(schedule)!=660 or sum(s['variant']=='ends' for s in schedule)!=120:
        raise ValueError('fixed process schedule changed')
    write(out/'schedule.json',schedule)
    native={};commands={};bindings={}
    for protocol,name,variants in (('v09',FROZEN,('plain','dual')),('ends',ENDS,('ends',))):
        original=base/name;dest=out/'protocols'/protocol
        source_hashes=read(original/'source_hashes.json')
        declared = {name: digest for name, digest in source_hashes.items() if name.startswith('source/')}
        for name, digest in declared.items():
            p = original/name
            if sha(p)!=digest:
                raise ValueError('original source changed: '+str(p))
            target = dest/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p,target)
        (dest/'build').mkdir();(out/'cpu-precheck'/protocol/'build').mkdir(parents=True)
        original_commands=read(original/'build/commands.json')
        original_hashes=read(original/'build/binary_hashes.json')
        for cfg in ('cfg_a','cfg_b','cfg_c'):
            for variant in variants:
                key=f'{cfg}_{variant}';binary=original/'build'/key
                if sha(binary)!=original_hashes[f'build/{key}']:
                    raise ValueError('original ARM binary identity changed')
                shutil.copy2(binary,dest/'build'/key)
                shutil.copy2(original/'build'/f'{key}.sass',dest/'build'/f'{key}.sass')
                shutil.copy2(original/'build'/f'{key}.log',dest/'build'/f'{key}.log')
                native[f'protocols/{protocol}/build/{key}']=sha(binary)
                commands[key]=dict(protocol=protocol,argv=original_commands[key])
        bindings[protocol]=dict(original=str(original),source_hashes_sha256=sha(original/'source_hashes.json'),
            compiler=(original/'build/nvcc-version.txt').read_text(),
            ignored_unlisted_files=[str(p.relative_to(original)) for p in sorted((original/'source').rglob('*'))
                                    if p.is_file() and str(p.relative_to(original)) not in declared],
            source_files={str(p.relative_to(dest/'source')):sha(p) for p in sorted((dest/'source').rglob('*')) if p.is_file()})
    (out/'source').mkdir()
    shutil.copy2(Path(__file__),out/'source'/Path(__file__).name)
    write(out/'build-commands.json',commands);write(out/'native-binary-hashes.json',native)
    write(out/'protocol-bindings.json',bindings)
    write(out/'run_config.json',dict(calibration=True,V10=False,reference_gpu=GPU,
        reference_node='romeo-a057',planned_processes=660,max_gpu_submissions=1,max_wall_minutes=60,
        cases_sha256=sha(out/'cases.json'),schedule_sha256=sha(out/'schedule.json'),
        runner_sha256=sha(out/'source'/Path(__file__).name),
        warmup='Original binaries: adaptive 8..30, last-five CV <=2%; original correctness/tolerance',
        schedule_seed='20261012-phase-case-trial',
        source_inputs={str(base/MEASUREMENT/p):sha(base/MEASUREMENT/p) for p in ('README.md','conditions.json','static-checks.json')},
        baseline_frozen_prediction_sha256=sha(base/FROZEN/'frozen/predictions.json'),
        baseline_frozen_manifest_sha256=sha(base/FROZEN/'frozen/manifest.json')))
    write(out/'reference-expected.json',read(base/FROZEN/'frozen/predictions.json')['reference_metrics'])
    write(out/'reference-policy.json',read(base/FROZEN/'inputs/policy.json'))
    write(out/'submissions.json',dict(new_gpu_submissions=[],gpu_submission_count=0,
        gpu_limit=1,cpu_activity='Login-node CUDA12.9 static build; no CPU Slurm or GPU allocation.'))
    print(out)


def gemm_instructions(text):
    bodies=[s for s in re.split(r'\n\s*Function : ',text) if 'GemmUniversal' in s.split('\n',1)[0]]
    result=[]
    for body in bodies:
        result.append([re.sub(r'/\*.*?\*/','',line).strip() for line in body.splitlines()
                       if re.match(r'\s*/\*[0-9a-fA-F]+\*/',line)])
    if not result:raise ValueError('GemmUniversal SASS missing')
    return result


def gemm_resources(log):
    blocks = re.split(r'ptxas info\s*: Function properties for ',log)
    values = []
    for block in blocks:
        if 'GemmUniversal' not in block.split('\n',1)[0]:
            continue
        registers = re.search(r'Used (\d+) registers, used (\d+) barriers',block)
        spills = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block)
        if not registers or not spills:
            raise ValueError('GEMM resource declaration missing')
        values.append(dict(registers=int(registers[1]),barriers=int(registers[2]),
                           stack_bytes=int(spills[1]),spill_store_bytes=int(spills[2]),spill_load_bytes=int(spills[3])))
    if not values:
        raise ValueError('GEMM resource properties missing')
    return values


def cpu_build(root, toolkit):
    nvcc=toolkit/'bin/nvcc';dump=toolkit/'bin/cuobjdump'
    version=subprocess.check_output([str(nvcc),'--version'],text=True)
    if 'V12.9.41' not in version:raise ValueError('CUDA12.9.41 required')
    (root/'cpu-precheck/compiler.txt').write_text(version)
    (root/'cpu-precheck/host-compiler.txt').write_text(subprocess.check_output(['g++','--version'],text=True))
    results={}
    def build_one(name,item):
        protocol=item['protocol'];folder=root/'cpu-precheck'/protocol/'build';binary=folder/name
        cmd=item['argv'][:];cmd[0]=str(nvcc);cmd[cmd.index('-o')+1]=str(binary)
        with (folder/f'{name}.log').open('w') as log:
            subprocess.run(cmd,cwd=root/'protocols'/protocol,stdout=log,stderr=subprocess.STDOUT,check=True)
        sass=subprocess.check_output([str(dump),'--dump-sass',str(binary)],text=True)
        (folder/f'{name}.sass').write_text(sass)
        resource=subprocess.check_output([str(dump),'--dump-resource-usage',str(binary)],text=True)
        (folder/f'{name}.resources').write_text(resource)
        original=root/'protocols'/protocol/'build'/f'{name}.sass'
        current=gemm_instructions(sass);prior=gemm_instructions(original.read_text())
        log=(folder/f'{name}.log').read_text()
        count=sum(sum(bool(re.search(r'\bHGMMA\.',line)) for line in body) for body in current)
        expected=8 if name.startswith('cfg_a_') else 16
        spills=[x for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',log) if any(map(int,x))]
        same=current==prior
        resources=gemm_resources(log)
        prior_resources=gemm_resources((original.with_suffix('.log')).read_text())
        resources_same=resources==prior_resources and all(r['registers']==168 for r in resources)
        result=dict(command=cmd,native_gemm_instruction_sequence_equal=same,
            gemm_resources=resources,native_gemm_resources=prior_resources,resource_identity_equal=resources_same,
            hgmma=count,expected_hgmma=expected,spills=spills,C7510=log.count('C7510'),
            cpu_binary_sha256=sha(binary),cpu_sass_sha256=sha(folder/f'{name}.sass'),
            original_sass_sha256=sha(original),resource_usage=resource,
            passed=same and resources_same and count==expected and not spills and 'C7510' not in log)
        print(name,'PASS' if result['passed'] else 'FAIL',flush=True)
        return name,result
    def one(name,item):
        try:
            return build_one(name,item)
        except Exception:
            error=traceback.format_exc()
            (root/'cpu-precheck'/item['protocol']/'build'/f'{name}.failure.txt').write_text(error)
            print(name,'FAIL',flush=True)
            return name,dict(passed=False,error=error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,result in pool.map(lambda pair:one(*pair),read(root/'build-commands.json').items()):results[name]=result
    write(root/'cpu-precheck/gate.json',dict(compiler=version,host=os.uname().nodename,
        original_arm_binaries_unchanged=True,results=results,passed=all(r['passed'] for r in results.values())))
    if not all(r['passed'] for r in results.values()):raise ValueError('CPU/SASS identity gate failed; do not submit GPU')


def sample(root):
    if os.uname().nodename!='romeo-a057' or not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('reference-node Slurm allocation required')
    allocation=subprocess.check_output(['scontrol','show','job','-dd',os.environ['SLURM_JOB_ID']],text=True)
    (root/'allocation.txt').write_text(allocation)
    # V09 reference job allocated all four cards. Establish membership before
    # narrowing visibility; a node-wide device list alone is insufficient.
    if not re.search(r'AllocTRES=.*gres/gpu=4(?:,|\s|$)',allocation):
        raise ValueError('reference job requires the original four-GPU allocation')
    if not re.search(r'NumNodes=1(?:\s|$)',allocation) or not re.search(r'NodeList=romeo-a057(?:\s|$)',allocation):
        raise ValueError('single original reference-node allocation required')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    (root/'allocation-visible-devices.json').write_text(json.dumps(dict(
        SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),
        CUDA_VISIBLE_DEVICES_before=visible,device_map=devices),indent=2)+'\n')
    if len(devices.strip().splitlines())!=4 or GPU not in devices:
        raise ValueError('required UUID is not in the allocated four-card node')
    if visible and set(visible.split(',')) not in (set('0123'),{line.split(',')[1].strip() for line in devices.splitlines()}):
        raise ValueError('job visibility does not prove access to every allocated GPU')
    allocated=os.environ.get('SLURM_STEP_GPUS') or os.environ.get('SLURM_JOB_GPUS','')
    if set(allocated.split(',')) not in (set('0123'),{line.split(',')[1].strip() for line in devices.splitlines()}):
        raise ValueError('job/step GPU membership does not prove the required four-card allocation')
    os.environ['V08_GPU']=GPU;os.environ['CUDA_VISIBLE_DEVICES']=GPU
    if not read(root/'cpu-precheck/gate.json')['passed']:raise ValueError('CPU gate absent')
    config=read(root/'run_config.json')
    if sha(Path(__file__))!=config['runner_sha256']:
        raise ValueError('prepared runner identity changed')
    for name in ('cases','schedule'):
        if sha(root/f'{name}.json')!=config[f'{name}_sha256']:
            raise ValueError('fixed input changed: '+name)
    for protocol,binding in read(root/'protocol-bindings.json').items():
        for name,digest in binding['source_files'].items():
            if sha(root/'protocols'/protocol/'source'/name)!=digest:
                raise ValueError('protocol source changed: '+name)
    for name,digest in read(root/'native-binary-hashes.json').items():
        if sha(root/name)!=digest:raise ValueError('native binary changed')
    # Use the original V09 process/check decoder; ends retains its recorded ABI.
    for variant in ('plain','dual','ends'):
        protocol='ends' if variant=='ends' else 'v09'
        for cfg in ('cfg_a','cfg_b','cfg_c'):
            shutil.copy2(root/'protocols'/protocol/'build'/f'{cfg}_{variant}',root/'build'/f'{cfg}_{variant}')
    sys.path.insert(0,str(root/'source'))
    sys.path.insert(1,str(root/'protocols/v09/source'))
    import run_v08
    import v06_run as common
    import analyze_r18
    import v08_model
    if not read(root/'ends-decoder-precheck.json')['passed']:
        raise ValueError('original ends ABI has not passed the V09 decoder check')
    # The original wrappers verify source/build hashes at setup.
    write(root/'source_hashes.json',{str(p.relative_to(root)):sha(p) for p in (root/'source').rglob('*') if p.is_file()})
    write(root/'build/binary_hashes.json',{str(p.relative_to(root)):sha(p) for p in (root/'build').iterdir() if p.is_file()})
    rows={r['id']:r for r in read(root/'cases.json')}
    lock=open('/tmp/gh200-measurement-'+GPU+'.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Keep the original process and decoder. Its recursive warmup recovery
    # enters this wrapper, which permits only one recovery in the entire run.
    original_one=run_v08.run_one
    recovery_path=root/'recovery.json'
    recovered=read(recovery_path)['recovered_processes'] if recovery_path.exists() else []
    def bounded_one(folder,row,variant,trial,attempt=0):
        if attempt:
            key=dict(case=row['id'],variant=variant,trial=trial,attempt=attempt)
            if key not in recovered:
                if recovered:
                    raise ValueError('the authorized one-recovery allowance is exhausted')
                recovered.append(key)
                write(recovery_path,dict(limit=1,recovered_processes=recovered))
        return original_one(folder,row,variant,trial,attempt)
    run_v08.run_one=bounded_one
    original_run=subprocess.run
    ledger=root/'process-ledger.jsonl'
    launched=sum(1 for line in ledger.read_text().splitlines() if json.loads(line).get('event')=='start') if ledger.exists() else 0
    def recorded_run(*args,**kwargs):
        nonlocal launched
        command=args[0] if args else kwargs['args']
        if not isinstance(command,(list,tuple)) or Path(command[0]).parent!=root/'build':
            return original_run(*args,**kwargs)
        number=launched;launched+=1
        category='setup' if '--mode' in command and 'setup' in command else 'sampling'
        entry=dict(process=number,category=category,command=command,host_start_ns=time.time_ns())
        with ledger.open('a') as stream:
            stream.write(json.dumps(dict(event='start',**entry))+'\n');stream.flush();os.fsync(stream.fileno())
        try:
            process=original_run(*args,**kwargs)
            terminal=dict(event='return',process=number,returncode=process.returncode,host_stop_ns=time.time_ns())
        except Exception as error:
            folder=root/'failed-processes';folder.mkdir(exist_ok=True)
            for name in ('stdout','stderr'):
                value=getattr(error,name,None)
                if value:
                    with gzip.open(folder/f'{number}-{name}.gz','wb') as stream:
                        stream.write(value.encode() if isinstance(value,str) else value)
            terminal=dict(event='exception',process=number,error=repr(error),host_stop_ns=time.time_ns())
            raise
        finally:
            with ledger.open('a') as stream:stream.write(json.dumps(terminal)+'\n')
        return process
    subprocess.run=recorded_run
    run_v08.setup(root)
    observed_refs={};done=[]
    for index,item in enumerate(read(root/'schedule.json')):
        row=rows[item['case']]
        rec=run_v08.run_one(root,row,item['variant'],item['trial'])
        done.append(dict(index=index,**item,raw_sha256=rec['raw_sha256']))
        if item['phase']=='reference':
            values=observed_refs.setdefault(row['id'],dict(plain=[],envelope=[],frequency=[],extra=[]))
            if item['variant']=='plain':values['plain'].append(rec['elapsed_us'])
            else:
                cs=analyze_r18.replay(root,rec,row)['ctas'];most=max(len(c['tiles']) for c in cs)
                window=(max(c['end_ns'] for c in cs)-min(c['entry_ns'] for c in cs))/1000
                values['envelope'].append(window)
                values['frequency'].append(st.median((c['end_c']-c['entry_c'])/(c['end_ns']-c['entry_ns']) for c in cs if len(c['tiles'])==most))
                values['extra'].append(rec['elapsed_us']-window)
        write(root/'progress.json',dict(completed_slots=len(done),last=item))
        if (index+1)%60==0:
            print('completed slots',index+1,'of 660',flush=True)
        if index==119:
            expected=read(root/'reference-expected.json');policy=read(root/'reference-policy.json');changes={};passed=True
            for cid,values in observed_refs.items():
                actual=dict(plain_us=st.median(values['plain']),envelope_us=st.median(values['envelope']),
                            effective_ghz=st.median(values['frequency']),event_extra_us=st.median(values['extra']))
                changes[cid]=dict(actual=actual,expected=expected[cid])
                for key,limit in policy['reference_relative_limits'].items():passed &= abs(actual[key]/expected[cid][key]-1)<=limit
                for key,limit in policy['reference_absolute_limits'].items():passed &= abs(actual[key]-expected[cid][key])<=limit
            write(root/'reference-check.json',dict(passed=bool(passed),changes=changes,policy=policy))
            if not passed:raise ValueError('frozen reference gate failed; development targets stopped')
    write(root/'sampling-complete.json',dict(planned_slots=660,completed_slots=len(done),slots=done,
        GPU=GPU,job=os.environ['SLURM_JOB_ID'],development_only=True,V10=False))
    (root/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q','-i',GPU],text=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('step',choices=['prepare','cpu-build','sample']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path);p.add_argument('--toolkit',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.step=='prepare':prepare(a.base.resolve(),root)
    elif a.step=='cpu-build':cpu_build(root,a.toolkit.resolve())
    else:
        (root/'build').mkdir(exist_ok=True)
        try:
            sample(root)
        except Exception:
            write(root/'sampling-failure.json',dict(error=traceback.format_exc(),unix_ns=time.time_ns(),
                  job=os.environ.get('SLURM_JOB_ID'),progress=read(root/'progress.json') if (root/'progress.json').exists() else None))
            raise
