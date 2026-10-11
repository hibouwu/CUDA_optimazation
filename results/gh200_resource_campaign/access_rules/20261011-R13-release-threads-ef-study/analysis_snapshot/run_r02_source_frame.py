#!/usr/bin/env python3
"""R02 bounded diagnostic: two source forms in the same kernel/register frame."""
import argparse,fcntl,json,os,random,re,shutil,statistics,subprocess
from pathlib import Path
import run_r02 as runner
from analyze_r02 import replay

ROOT=Path(__file__).resolve().parent
CASES=[dict(id='frame_'+mode,subset='source',source='fixed_frame',chains=8,warps=4,rotate=rotate)
       for rotate,mode in [(0,'reuse_pair'),(1,'rotate_eight_pairs')]]


def sass(path):
    facts=[]
    for body in path.read_text().split('Function : ')[1:]:
        if 'r02_probeILi0ELi3ELi8ELb0ELi' not in body.splitlines()[0]:continue
        instructions=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',line)
            if m:instructions.append((int(m[1],16),m[2]))
        if any(re.search(r'\b(?:LDL|STL)\b',op) for _,op in instructions):raise ValueError('frame diagnostic spilled')
        loops=[]
        for address,op in instructions:
            m=re.search(r'BRA 0x([0-9a-f]+)',op)
            if not m or int(m[1],16)>=address:continue
            first=int(m[1],16);ops=[s for a,s in instructions if first<=a<=address]
            ffma=[s for s in ops if re.search(r'\bFFMA\b',s)]
            if len(ffma)!=64:continue
            operands=[s.split('FFMA',1)[1].strip().split(',') for s in ffma]
            pairs={tuple(x.strip().replace('.reuse','') for x in op[1:3]) for op in operands}
            dest={op[0].strip() for op in operands}
            # ptxas can rename an accumulator on the first/last iteration of
            # the unrolled body. Follow the C operand, not physical dest count.
            chains={};counts={};origins={}
            for operands_one in operands:
                d=operands_one[0].strip();c=operands_one[3].strip().replace('.reuse','')
                if c not in chains:
                    chains[c]=len(origins);origins[c]=chains[c];counts[chains[c]]=0
                chain=chains[c];chains[d]=chain;counts[chain]+=1
            if len(origins)!=8 or sorted(counts.values())!=[8]*8:
                raise ValueError('not eight dependent accumulator chains of eight operations')
            if any(chains[c]!=chain for c,chain in origins.items()):
                raise ValueError('loop backedge changes accumulator chains')
            loops.append(dict(begin=hex(first),end=hex(address),source_pairs=len(pairs),
                              accumulator_chains=8,destination_registers=sorted(dest),instructions=ops))
        if sorted(x['source_pairs'] for x in loops)!=[1,8]:raise ValueError('missing one/eight-pair branches')
        clocks=[(hex(a),op) for a,op in instructions if 'SR_CLOCKLO' in op]
        if len(clocks)!=3:raise ValueError('diagnostic clock boundaries changed')
        facts.append(dict(kernel=body.splitlines()[0],loops=loops,clocks=clocks))
    if len(facts)!=2:raise ValueError('two code-position specializations expected')
    return facts


def analyze(root,output=None):
    for n,h in json.loads((root/'source_hashes.json').read_text()).items():
        if runner.sha(root/n)!=h:raise ValueError('frozen source changed')
    identity=json.loads((root/'build/binary.json').read_text())
    if runner.sha(root/'build/r02')!=identity['sha256'] or runner.sha(root/'build/sass.txt')!=identity['sass_sha256']:
        raise ValueError('binary/SASS changed')
    facts=sass(root/'build/sass.txt');records=[]
    for path in sorted((root/'samples').glob('*/*/result.json')):
        r=json.loads(path.read_text());r['checked']=replay(path.parent,r);records.append(r)
    if not records:raise ValueError('no diagnostic samples')
    rows=[]
    for case in CASES:
        for position in [0,1]:
            for trace in [0,1]:
                rr=[r for r in records if r['case_id']==case['id'] and r['label'].startswith('formal-') and r['position']==position and r['intermediate']==bool(trace)]
                if len(rr)<10:raise ValueError('ten independent processes needed for zero-difference diagnosis')
                vals=[r['consumed_cycles'] for r in rr]
                rows.append(dict(case=case['id'],position=position,trace=trace,processes=len(rr),
                  cycles=statistics.median(vals),cv=runner.cv(vals),stddev=statistics.pstdev(vals),
                  registers=sorted({r['registers'] for r in rr}),kernel=sorted({r['kernel'] for r in rr}),
                  qualified=runner.cv(vals)<=.05 and all(r['warmup_converged'] and r['local_bytes']==0 and r['check_errors']==0 for r in rr)))
    for pos in [0,1]:
        group=[r for r in rows if r['position']==pos]
        if len({tuple(r['registers']) for r in group})!=1 or len({tuple(r['kernel']) for r in group})!=1:
            raise ValueError('source forms do not share the same allocated frame')
    dest=output or root/'analysis';dest.mkdir(parents=True,exist_ok=True)
    runner.write(dest/'sass-check.json',facts)
    runner.write(dest/'rules.json',dict(conditions=rows,processes=len(records),checked_values=sum(r['checked'] for r in records),
      flop=128*8192*2,scope='one CTA, four warps, eight chains, common kernel frame; two compiler-generated branch bodies, not fixed physical RF fields'))
    print('R02 fixed-frame checks:',len(records),'processes')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('step',choices=['prepare','sample','analyze'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--analysis-output',type=Path)
    a=p.parse_args();root=a.output.resolve()
    if a.step=='prepare':
        runner.prepare(root);shutil.copy2(Path(__file__),root/'source'/Path(__file__).name)
        runner.write(root/'cases.json',CASES)
        runner.write(root/'source_hashes.json',{str(x.relative_to(root)):runner.sha(x) for x in (root/'source').rglob('*') if x.is_file()})
        return
    if a.step=='analyze':analyze(root,a.analysis_output);return
    if not os.environ.get('SLURM_JOB_ID'):raise ValueError('Slurm allocation required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,power.limit','--format=csv,noheader'],text=True).strip()
    if 'GH200' not in gpu or '\n' in gpu:raise ValueError('single GH200 required')
    env=dict(job=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,gpu=gpu)
    locks=[open('/tmp/gh200-measurement-'+gpu.split(',')[1].strip()+'.lock','a'),(root/'.run.lock').open('a')]
    for lock in locks:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for n,h in json.loads((root/'source_hashes.json').read_text()).items():
        if runner.sha(root/n)!=h:raise ValueError('frozen source changed')
    if (root/'environment.json').exists():
        if json.loads((root/'environment.json').read_text())!=env:raise ValueError('environment changed')
    else:runner.write(root/'environment.json',env)
    (root/'nvidia-smi-before.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))
    if not (root/'build/binary.json').exists():runner.build(root,['-DR02_FRAME_DIAG'])
    sass(root/'build/sass.txt')
    for case in CASES:
        r=runner.run_one(root,case,'smoke',64,banked=case['rotate'],witness=1)
        replay(root/'samples'/case['id']/'smoke',r)
    for trial in range(10):
        for position in [0,1]:
            group=CASES[:];random.Random(20261008+trial+position*100).shuffle(group)
            for case in group:
                variants=[0,1];random.Random(20261008+trial).shuffle(variants)
                for trace in variants:
                    label=f'formal-{trial:02}-pos{position}-trace{trace}'
                    r=runner.run_one(root,case,label,8192,intermediate=trace,position=position,banked=case['rotate'])
                    replay(root/'samples'/case['id']/label,r)
    analyze(root)
    (root/'nvidia-smi-after.txt').write_text(subprocess.check_output(['nvidia-smi','-q'],text=True))

if __name__=='__main__':main()
