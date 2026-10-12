#!/usr/bin/env python3
"""Run the prelisted ordinary-SMEM development or frozen holdout in Slurm."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def command(argv,folder,stem,timeout=240):
    with (folder/(stem+'.stdout')).open('w') as out,(folder/(stem+'.stderr')).open('w') as err:
        result=subprocess.run(argv,cwd=folder,stdout=out,stderr=err,timeout=timeout)
    write(folder/(stem+'.process.json'),dict(command=argv,returncode=result.returncode))
    if result.returncode:
        raise RuntimeError('failed '+str(folder/stem))
    return (folder/(stem+'.stdout')).read_text()


def sass_audit(path):
    result=[]
    for block in path.read_text().split('Function : ')[1:]:
        name=block.splitlines()[0].strip()
        match=re.match(r'_Z10smem_probeILi([012])ELi(4|8|16)ELb([01])ELb([01])E',name)
        if not match:
            continue
        op,width,dependent,native=map(int,match.groups())
        if op==2:
            continue
        instructions=[]
        for line in block.splitlines():
            parsed=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?);',line)
            if parsed:
                instructions.append((int(parsed[1],16),parsed[2]))
        target='LDS' if op==0 else 'STS'
        expected=target+({4:'',8:'.64',16:'.128'}[width])
        loops=[]
        for pc,text in instructions:
            jump=re.search(r'\bBRA\s+(?:0x)?([0-9a-f]+)',text)
            if not jump or int(jump[1],16)>=pc:
                continue
            start=int(jump[1],16)
            body=[(at,t) for at,t in instructions if start<=at<=pc]
            operations=[(at,re.search(r'\b('+target+r'(?:\.[A-Z0-9]+)*)\b',t)[1])
                for at,t in body if re.search(r'\b'+target+r'(?:\.|\s)',t)]
            if len(operations)==8:
                loops.append(dict(begin_pc=start,end_pc=pc,target_instructions=operations,
                    instruction_count=len(body),dependent_address_chain=bool(dependent)))
        valid=[loop for loop in loops if all(mnemonic==expected for _,mnemonic in loop['target_instructions'])]
        if len(valid)!=1:
            if native:
                # This sequence is prelisted for the later holdout, not used
                # to calibrate ordinary kernels. Retain the coverage failure;
                # it prevents holdout qualification rather than silently
                # discarding it or blocking unrelated development samples.
                result.append(dict(kernel=name,op=op,width=width,dependent=bool(dependent),native=True,
                    status='native_window_unqualified',expected_mnemonic=expected,observed_loops=loops))
                continue
            raise ValueError('SASS target loop is not eight '+expected+': '+name)
        result.append(dict(kernel=name,op=op,width=width,dependent=bool(dependent),native=bool(native),
            status='verified_target_loop',**valid[0]))
    if len(result)!=11:
        raise ValueError('expected 11 ordinary target kernels, got '+str(len(result)))
    return result


def build(package,output,cutlass):
    source=package/'source';out=output/'build';out.mkdir()
    nvcc=subprocess.check_output(['nvcc','--version'],text=True)
    (out/'nvcc-version.txt').write_text(nvcc)
    if 'release 12.9' not in nvcc:
        raise ValueError('frozen calibration compiler requires CUDA 12.9')
    common=['nvcc','-std=c++17','-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','--ptxas-options=-v']
    command(common+['-I'+str(cutlass/'include'),str(source/'probes/r03_lds_sts.cu'),'-o',str(out/'smem')],out,'compile-smem')
    command(common+[str(source/'exp09/probes/shared_memory.cu'),'-o',str(out/'exp09')],out,'compile-exp09')
    command(common+[str(source/'old-r03/probes/r03.cu'),'-o',str(out/'r03')],out,'compile-r03')
    for name in ('smem','exp09','r03'):
        text=command(['cuobjdump','--dump-sass',str(out/name)],out,'disassemble-'+name)
        (out/(name+'.sass')).write_text(text)
    write(out/'sass-audit.json',sass_audit(out/'smem.sass'))
    write(out/'binary-sha256.json',{name:sha(out/name) for name in ('smem','exp09','r03')})
    return out


def run_case(binary,case,folder):
    folder.mkdir(parents=True,exist_ok=False)
    maps=folder/'addresses.map'
    maps.write_text('\n'.join(str(x) for row in case['addresses_words'] for x in row)+'\n')
    args=[str(binary),str(case['op']),str(case['width']),str(case['warps']),str(int(case['dependent'])),
        str(int(case['native'])),str(case['iterations']),str(case['seed']),str(maps)]
    record=json.loads(command(args,folder,'measure',timeout=120))
    record['output_sha256']=sha(folder/'output.u32')
    record['map_sha256']=sha(maps)
    if record['local_bytes'] or (case['op']!=2 and not record['warmup_converged']) or record['errors']:
        raise ValueError('spill, warmup, or correctness gate failed')
    if record['target_warp_instructions']!=(0 if case['op']==2 else 8*case['iterations']*case['warps']):
        raise ValueError('dynamic target count mismatch')
    write(folder/'result.json',record)
    return record


def main(package,output,phase,cutlass,expected_gpu):
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('an allocated Slurm GPU job is required')
    output.mkdir(exist_ok=False,parents=True)
    try:
        if phase=='development':
            for path,expected in json.loads((package/'source-sha256.json').read_text()).items():
                if sha(package/'source'/path)!=expected:raise ValueError('source identity mismatch '+path)
            for path,expected in json.loads((package/'cutlass-closure-sha256.json').read_text()).items():
                if sha(cutlass/path)!=expected:raise ValueError('CUTLASS closure mismatch '+path)
            binaries=build(package,output,cutlass)
        else:
            frozen=json.loads((package/'freeze/manifest.json').read_text())
            for path,expected in frozen['files_sha256'].items():
                if sha(package/path)!=expected:raise ValueError('freeze hash mismatch '+path)
            binaries=package/'development/build'
            if any(row['status']!='verified_target_loop' for row in json.loads((binaries/'sass-audit.json').read_text())):
                raise ValueError('prelisted CUTLASS sequence lacks a qualified repeated SASS window')
        device=json.loads(command([str(binaries/'smem'),'device'],output,'device'))
        environment=dict(**device,host=os.uname().nodename,job=os.environ['SLURM_JOB_ID'],
            CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            SLURM_JOB_GPUS=os.environ.get('SLURM_JOB_GPUS'),SLURM_STEP_GPUS=os.environ.get('SLURM_STEP_GPUS'),
            phase=phase,protocol_sha256=sha(package/'protocol.json'))
        write(output/'environment.json',environment)
        if expected_gpu and device['gpu_uuid']!=expected_gpu:
            raise ValueError('allocated GPU differs from frozen calibration UUID')
        with open('/tmp/gh200-access-'+device['gpu_uuid']+'.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if phase=='development':
                old=output/'compatibility';old.mkdir()
                cases=[f'read_w4_stride{s}' for s in (1,2,4,8,16,32)]+['read_w4_broadcast','read_w8_stride1','read_w16_stride1','write_w4_stride1']
                for trial in range(10):
                    for case in cases:
                        if trial>=3 and case!='read_w4_broadcast':continue
                        folder=old/(case+f'-{trial:02}');folder.mkdir()
                        command([str(binaries/'exp09'),case,'8192','17'],folder,'measure')
                for op in (0,1):
                    for width in (4,16):
                        folder=old/f'r03-o{op}-b{width}';folder.mkdir()
                        command([str(binaries/'r03'),'--op',str(op),'--width',str(width),'--warps','8','--iterations','8192','--seed','17'],folder,'measure')
                for trial in range(10):
                    command([str(binaries/'smem'),'grain'],output,f'grain-{trial:02}')
                cases=json.loads((package/'development-cases.json').read_text())+json.loads((package/'control-cases.json').read_text())
            else:
                cases=json.loads((package/'freeze/holdout-cases.json').read_text())
                # Prelisted across-job reference checks use original source window.
                refs=output/'references';refs.mkdir()
                for case in ('read_w4_stride4','read_w8_stride1','write_w4_stride1'):
                    folder=refs/case;folder.mkdir()
                    command([str(binaries/'exp09'),case,'8192','17'],folder,'measure')
            rng=random.Random(2026101130)
            failures=[]
            for trial in range(3):
                ordered=cases.copy();rng.shuffle(ordered)
                for case in ordered:
                    folder=output/'samples'/case['id']/f'trial-{trial:02}'
                    try:
                        run_case(binaries/'smem',case,folder)
                    except Exception as error:
                        failures.append(dict(case=case['id'],trial=trial,error=str(error)))
                        if folder.exists():write(folder/'failure.json',failures[-1])
                    print(case['id'],trial,flush=True)
            write(output/'failures.json',failures)
            write(output/'completion.json',dict(status='complete_with_failures' if failures else 'complete',
                prelisted_conditions=len(cases),processes=3*len(cases),failures=len(failures)))
    except Exception as error:
        write(output/'fatal.json',dict(error=str(error),phase=phase))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('development','holdout'),required=True)
    p.add_argument('--cutlass',type=Path,required=True);p.add_argument('--expected-gpu')
    a=p.parse_args();main(a.package.resolve(),a.output.resolve(),a.phase,a.cutlass.resolve(),a.expected_gpu)
