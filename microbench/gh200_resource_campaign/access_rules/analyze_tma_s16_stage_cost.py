#!/usr/bin/env python3
"""S16 static evidence and one-parameter development transfer, no GPU execution."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import subprocess

from scipy.optimize import brentq
from tma_input_events import Costs, GroupCosts, simulate
from analyze_tma_input_events import ROOT, S16, load_conditions

ACTUAL = S16.parent / 'storage-v2/collection-job734543-r2-CPU734612/actual'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instructions(sass, symbol):
    match = re.search(r'Function : '+re.escape(symbol)+r'\s(.*?)(?=Function :|\Z)', sass, re.S)
    if not match:
        raise ValueError(f'missing kernel: {symbol}')
    return [dict(address=int(m[1], 16), text=m[2].strip())
            for m in re.finditer(r'/\*([0-9a-f]+)\*/\s*(.*?);', match[1])]


def windows(code):
    # Straight-line instruction positions on the producer path, not runtime
    # lane-executed instruction counts. Predicated instructions remain listed.
    wait = next(i for i,e in enumerate(code) if 'TRYWAIT' in e['text'])
    previous_branch = max(i for i in range(wait) if 'BRA' in code[i]['text'])
    arrive = next(i for i,e in enumerate(code) if 'SYNCS.ARRIVE.TRANS64.A1T0' in e['text'])
    following_branch = next(i for i in range(arrive+1,len(code)) if 'BRA' in code[i]['text'])
    result = {}
    for name, values in [('before_wait',code[previous_branch+1:wait]),
                         ('after_arrive',code[arrive+1:following_branch])]:
        counts = Counter(e['text'].split()[1] if e['text'].startswith('@') else e['text'].split()[0] for e in values)
        result[name] = dict(positions=len(values), opcode_counts=dict(counts), instructions=values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'results/gh200_resource_campaign/access_rules/Q02-tma-request/S16-single-cta-control/20261013-tma-input-events-v2/analysis.json')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be new or empty')
    actual = ROOT/ACTUAL
    manifest = json.loads((actual/'source-manifest.json').read_text())
    published = json.loads((ROOT/S16/'qualified-parameters.json').read_text())['parameters'][0]
    assert digest(actual/'source-manifest.json') == published['source_manifest_sha256']
    assert digest(actual/'summary.json') == published['source_summary_sha256']
    binary = actual/'repo/bin/probe'
    assert digest(binary) == manifest['bin/probe'] == json.loads((actual/'binary-identity.json').read_text())['probe']
    source = actual/'repo/microbench/gh200_resource_campaign/probes/tma_stage_request.cu'
    assert digest(source) == manifest['microbench/gh200_resource_campaign/probes/tma_stage_request.cu']
    current = ROOT/'microbench/gh200_resource_campaign/families/s16/microbench/gh200_resource_campaign/probes/tma_stage_request.cu'
    assert digest(source) == digest(current)
    baseline = json.loads(args.baseline.read_text())
    assert baseline['status'] == 'development_only_not_hardware_qualification'
    # Match the baseline's published evidence to today's files; code is allowed
    # to evolve while the historical baseline result stays frozen.
    for name, expected in baseline['input_sha256'].items():
        assert digest(ROOT/name) == expected, name
    run = subprocess.run(['cuobjdump','--dump-sass',str(binary)], capture_output=True, text=True, check=True)
    cases, inputs = load_conditions()
    cases = [r for r in cases if r['family']=='s16']
    static = {}
    for s in (2,4):
        for r in (1,2):
            symbol = f'ts_g2s_s{s}_r{r}'
            code = instructions(run.stdout, symbol)
            row = next(c for c in cases if c['stages']==s and c['requests']==r)
            static[symbol] = dict(windows=windows(code), resources=row['resources'],
                                  local_memory_instructions=[e for e in code if re.search(r'\b(?:LDL|STL)\b',e['text'])])
    deltas = {}
    for r in (1,2):
        deltas[r] = {name: static[f'ts_g2s_s4_r{r}']['windows'][name]['positions']-
                    static[f'ts_g2s_s2_r{r}']['windows'][name]['positions']
                    for name in ('before_wait','after_arrive')}
    assert deltas[1] == deltas[2] == dict(before_wait=12,after_arrive=18)
    old = next(f for f in baseline['fits'] if f['candidate']=='s16_credits_unlimited')
    costs = Costs(**old['costs'])
    calibration = next(c for c in cases if c['stages']==4 and c['requests']==1)
    transfers = next(c for c in cases if c['stages']==4 and c['requests']==2)

    def predict(row, hypothesis, parameter):
        changed = costs
        group = GroupCosts()
        if row['stages']==4:
            if hypothesis=='software_token_positions':
                # Effective cycles per extra static position. This shared weight
                # is a candidate assumption, not an instruction-latency claim.
                group=GroupCosts(parameter*12,parameter*18)
            elif hypothesis=='completion_tail_only':
                changed=replace(costs, completion_tail=costs.completion_tail+parameter)
            elif hypothesis=='unassigned_group_return_gap':
                group=GroupCosts(after_issue=parameter)
            else:
                raise ValueError(hypothesis)
        return simulate('s16',row['items'],row['stages'],row['requests'],changed,
                        group_costs=group)['cycles_per_request']

    fits=[]
    for hypothesis in ('software_token_positions','completion_tail_only','unassigned_group_return_gap'):
        value=brentq(lambda x: predict(calibration,hypothesis,x)-calibration['observed'],0,10000,xtol=1e-9)
        scores=[]
        for row in cases:
            predicted=predict(row,hypothesis,value)
            role='old_development_fit' if row['stages']!=4 else ('new_development_calibration' if row['requests']==1 else 'development_transfer')
            scores.append(dict(case_id=row['case_id'],role=role,observed=row['observed'],predicted=predicted,
                               error_percent=100*(predicted/row['observed']-1)))
        fits.append(dict(hypothesis=hypothesis,parameter=value,
                         effective_extra_group_cycles=(value*30 if hypothesis=='software_token_positions'
                                                       else value if hypothesis=='unassigned_group_return_gap' else None),
                         scores=scores))
    paired=[]
    for r in (1,2):
        a=next(c for c in cases if c['stages']==2 and c['requests']==r)
        b=next(c for c in cases if c['stages']==4 and c['requests']==r)
        paired.append(dict(requests=r,observed_extra_cycles_per_group=(b['observed']-a['observed'])*r))
    result=dict(status='development_only_not_hardware_qualification',
                disassembler_version=subprocess.run(['cuobjdump','--version'],capture_output=True,text=True,check=True).stdout,
                disassembly_sha256=hashlib.sha256(run.stdout.encode()).hexdigest(),
                source_matches_prior_inspected_source=True,static_windows=static,
                extra_instruction_positions=deltas,paired_observed=paired,fits=fits,
                limitations=['Static positions include predicates; they are not executed-lane counts or calibrated instruction cycles.',
                             'R1 stage4 is now calibration. R2 stage4 was already seen; this is no fresh holdout.',
                             'Different resource allocation and physical barrier addresses remain confounded.',
                             'Global lifecycle count updates remain inside the formal timed loop.',
                             'Rejecting the tail-only form does not reject all hardware-side group costs.'],
                input_sha256={**inputs,str(binary.relative_to(ROOT)):digest(binary),
                              str(source.relative_to(ROOT)):digest(source),str(args.baseline):digest(args.baseline),
                              str((actual/'source-manifest.json').relative_to(ROOT)):digest(actual/'source-manifest.json')},
                code_sha256={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
                             'microbench/gh200_resource_campaign/access_rules/tma_input_events.py':digest(Path(__file__).with_name('tma_input_events.py'))})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'probe.sass').write_text(run.stdout)
    (args.output/'disassembly.stderr').write_text(run.stderr)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(deltas=deltas,paired=paired,fits=fits),indent=2))


if __name__=='__main__':
    main()
