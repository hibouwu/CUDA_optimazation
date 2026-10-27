#!/usr/bin/env python3
"""Five prelisted same-observer repeat pairs; descriptive spatial reference only."""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_r13_sm import select_attempts
from analyze_r18 import replay
from r09_clock_brackets import paired_phase_summary
import v06_run as common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.base.resolve()/'Q09-frequency/R07-R09-inkernel-clock/20261010-R09-R13-R18-failure-controls'
    common.verify(root)
    rows = json.loads((root/'cases.json').read_text())
    row = next(r for r in rows if r['id']=='r13_stage_k4096_s6')
    selected,failed,_ = select_attempts(root,row['id'],('dual',))
    if len(selected)!=10 or failed:
        raise ValueError('ten successful stage6 dual repeats required; failures need review')
    decoded,inputs = {},{}
    for item in selected:
        rec = item['record']
        decoded[rec['trial']] = replay(root,rec,row,retain_roles=True)['ctas']
        inputs[str(root/rec['raw'])] = rec['raw_sha256']
    pairs = []
    for a,b in ((0,1),(2,3),(4,5),(6,7),(8,9)):
        left,right = ({c['cta']:c for c in decoded[t]} for t in (a,b))
        if set(left)!=set(right) or len(left)!=132:
            raise ValueError('full matching CTA set required')
        deltas = []
        for cta in sorted(left):
            x,y = left[cta],right[cta]
            for role,phase in ((0,'first'),(1,'later')):
                if x['role_work'][role]!=y['role_work'][role]:
                    raise ValueError('role work changed')
                ca,na = x['roles'][role]['tiles'][0],x['roles'][role]['tiles_ns'][0]
                cb,nb = y['roles'][role]['tiles'][0],y['roles'][role]['tiles_ns'][0]
                deltas.append(dict(cta=cta,role=role,phase=phase,work=x['role_work'][role],
                    baseline_sm=x['sm'],repeat_sm=y['sm'],
                    L_ns_delta=(nb[1]-nb[0])-(na[1]-na[0]),
                    L_cycles_delta=(cb[1]-cb[0])-(ca[1]-ca[0])))
        pairs.append(dict(trial_left=a,trial_right=b,cta_pairs=deltas,
                          phases=paired_phase_summary(deltas)))
    result = dict(scope='Same-observer sequential-process repeats; not instrument perturbation, pure noise, a tolerance calibration, or independent validation.',
        run=str(root),case=row['id'],pairs=pairs,input_sha256=inputs,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['Only five fixed disjoint repeat pairs; no distribution-free or population bound.',
            'State/history and CTA-to-SM assignment vary across sequential processes. All CTA/SM identities and native deltas retained.',
            'Existing 384/1024 ns gates are process-center screens, not accepted per-CTA spatial limits.',
            'Nonzero spatial changes remain unqualified for original-phenomenon attribution; no threshold is fitted here.'])
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'repeat-spatial-reference.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    summary = {phase:{key:max(p['phases'][phase]['L_ns_delta'][key] for p in pairs)
        for key in ('p95_centered_absolute','maximum_centered_absolute')} for phase in ('first','later')}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
