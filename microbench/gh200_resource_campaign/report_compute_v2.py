#!/usr/bin/env python3
"""Reconstruct compute work/rates and figures outside an immutable v2 archive."""
import argparse
import csv
import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path
import shutil
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def work(case, raw):
    p = case['parameters']
    if p['phase'] == 'empty_control':
        return 0
    participants = raw['blocks'] * raw['threads']
    repetitions = raw['iterations'] * p['chains'] * p['batch']
    if case['work_model'] == 'fma_dense_issue_v1':
        return 2 * participants * p['packed_lanes'] * repetitions
    if p['shape'] is not None:
        m, n, k = p['shape']
        width = p['collective_width_threads']
        if participants % width:
            raise ValueError('partial collective')
        return 2 * m * n * k * (participants // width) * repetitions
    raise ValueError('unsupported work model')


def render(root, replay_path, output):
    root, output = root.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('new analysis directory outside run required')
    replay = read(replay_path)
    if replay['status'] != 'pass_under_bounded_float_replay_v1' or replay['failed_checks']:
        raise ValueError('accepted replay required')
    inventory = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    if inventory != replay['input_artifacts_sha256']:
        raise ValueError('complete replay input hashes differ')
    spec = read(root / 'run_spec.json'); summary = read(root / 'summary.json')
    contract_path = (root / spec['contract_path']).resolve()
    if not contract_path.is_relative_to(root):
        raise ValueError('contract outside archive')
    contract = read(contract_path)
    if contract['family'] not in ('legacy_fma', 'legacy_mma', 'legacy_wgmma') or not summary['terminal']:
        raise ValueError('terminal compute archive required')
    cases = {c['id']: c for c in contract['cases']}
    records, case_rows = [], []
    for result in summary['cases']:
        case = cases[result['case_id']]; p = case['parameters']; rows = []
        for batch in result['batches']:
            for sample in batch['samples']:
                raw_path = (root / sample['receipt']).resolve().parent / 'raw.jsonl'
                if not raw_path.is_relative_to(root) or sha(raw_path) != sample['raw_sha256']:
                    raise ValueError('raw path/hash mismatch')
                raw = next(json.loads(line) for line in raw_path.read_text().splitlines() if json.loads(line).get('type') == 'trial')
                expected = work(case, raw)
                if raw['work_count'] != expected:
                    raise ValueError('independent work count mismatch: ' + case['id'])
                blocks = raw['blocks_detail']
                if case['scope'] == 'one_cta':
                    if len(blocks) != 1:
                        raise ValueError('local cycle metric requires one CTA')
                    duration = blocks[0]['stop_cycle'] - blocks[0]['start_cycle']
                else:
                    duration = max(b['stop_ns'] for b in blocks) - min(b['start_ns'] for b in blocks)
                if duration <= 0:
                    raise ValueError('nonpositive duration')
                value = duration if p['phase'] == 'empty_control' else expected / duration
                if value != sample['value']:
                    raise ValueError('raw rate differs from summary')
                row = {'case_id': case['id'], 'input_type': p['input_type'], 'scope': case['scope'],
                    'phase': p['phase'], 'threads': raw['threads'], 'blocks': raw['blocks'],
                    'chains': p['chains'], 'batch_instructions': p['batch'], 'iterations': raw['iterations'],
                    'packed_lanes': p.get('packed_lanes', 1), 'wait': p.get('wait'),
                    'process_batch': sample['batch'], 'trial': sample['trial'], 'work_count': expected,
                    'duration': duration, 'unit': sample['unit'], 'value': value,
                    'warmup_converged': sample['warmup_converged'], 'status': result['status'],
                    'raw': raw_path.relative_to(root).as_posix(), 'raw_sha256': sha(raw_path)}
                if contract['family']=='legacy_wgmma':
                    row.update(operand_source_form=p['operand_source_form'], matrix_shape='x'.join(map(str,p['shape'])), collectives_per_CTA=raw['threads']//p['collective_width_threads'])
                records.append(row); rows.append(row)
        accepted = [r['value'] for r in rows if r['warmup_converged']]
        if len(accepted) < 2:
            raise ValueError('insufficient accepted samples for this report')
        median = statistics.median(accepted)
        if median != result['merged']['median'] or len(accepted) != result['merged']['samples']:
            raise ValueError('merged median/count differs')
        first = rows[0]
        case_rows.append({**{k: first[k] for k in ('case_id','input_type','scope','phase','threads','blocks','chains','batch_instructions','iterations','packed_lanes','wait','unit') + (('operand_source_form','matrix_shape','collectives_per_CTA') if contract['family']=='legacy_wgmma' else ())},
                          'status': result['status'], 'samples': len(accepted), 'median': median,
                          'min': min(accepted), 'max': max(accepted), 'cv': statistics.stdev(accepted)/statistics.mean(accepted),
                          'source_configuration': (case.get('legacy') or {}).get('configuration_id','new representative')})
    # Bind fits to the same actual compiled kernel, not just dtype labels.
    source = (root / spec['source_path']).resolve()
    if not source.is_relative_to(root):
        raise ValueError('source path outside archive')
    symbols = dict(re.findall(r'\{"([^"]+)","[^"]+",(lc_[a-z0-9]+),', source.read_text()))
    approved_symbols = {r['function'] for r in read(root / 'build/sass_audit.json')}
    fit_groups = {}
    for row in case_rows:
        case = cases[row['case_id']]
        if row['phase'] == 'empty_control' or case['iteration_policy']['kind'] != 'fixed':
            continue
        symbol = symbols.get(row['case_id'])
        if symbol not in approved_symbols:
            raise ValueError('fixed-length fit lacks SASS-bound symbol')
        key = (symbol, row['scope'], row['threads'], row['unit'], json.dumps(case['parameters'], sort_keys=True))
        fit_groups.setdefault(key, []).append(row)
    fits = []
    for key, rows in fit_groups.items():
        if len({r['iterations'] for r in rows}) < 3:
            continue
        points = []
        for row in rows:
            durations = sorted(r['duration'] for r in records if r['case_id'] == row['case_id'] and r['warmup_converged'])
            n = len(durations)
            median = Fraction(durations[(n-1)//2] + durations[n//2], 2)
            points.append((row['iterations'], median, row['case_id']))
        count = len(points); sx = sum(x for x,y,c in points); sy = sum(y for x,y,c in points)
        slope = (count * sum(x*y for x,y,c in points) - sx*sy) / (count * sum(x*x for x,y,c in points) - sx*sx)
        intercept = (sy - slope*sx) / count
        error = max(abs(intercept+slope*x-y)/y for x,y,c in points)
        fits.append({'kernel_symbol':key[0], 'scope':key[1], 'threads':key[2],
            'duration_unit':'clock64_cycle/CTA' if key[1]=='one_cta' else 'globaltimer_ns/GPU',
            'input_type':rows[0]['input_type'], 'chains':rows[0]['chains'],
            'points':[{'iterations':x,'median_duration':float(y),'case_id':c} for x,y,c in sorted(points)],
            'intercept':float(intercept),'increment_per_iteration':float(slope),
            'maximum_in_sample_relative_error':float(error), 'held_out_validation':False,
            'parameter_export':False, 'interpretation':'descriptive fixed-kernel fit only; no general latency or extrapolation claim'})
    output.mkdir(parents=True)
    (output/'fixed_length_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
    for name, data in [('samples.csv', records), ('cases.csv', case_rows)]:
        with (output / name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    examples = [next(r for r in records if r['scope'] == scope and r['phase'] != 'empty_control') for scope in ('one_cta','all_gpu')]
    (output/'worked_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gh200-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['svg.hashsalt'] = 'gh200-compute-v2'
    figures=[]
    groups=sorted({(r['input_type'],r['scope'],r['unit']) for r in case_rows if r['phase']!='empty_control'})
    for dtype,scope,unit in groups:
        selection=[r for r in case_rows if (r['input_type'],r['scope'],r['unit'])==(dtype,scope,unit) and r['phase']!='empty_control']
        for page in range((len(selection)+11)//12):
            chunk=selection[page*12:(page+1)*12]
            fig,ax=plt.subplots(figsize=(11,max(3.5,len(chunk)*.42+1.8)))
            labels=[]
            for y,row in enumerate(chunk):
                values=[r['value'] for r in records if r['case_id']==row['case_id'] and r['warmup_converged']]
                color='#24678d' if row['status']=='stable' else '#b85b19'
                ax.scatter(values,[y]*len(values),s=10,color=color,alpha=.45)
                ax.errorbar(row['median'],y,xerr=[[row['median']-row['min']],[row['max']-row['median']]],fmt='D',color=color,capsize=3)
                if contract['family']=='legacy_wgmma':
                    labels.append(f"wg{row['collectives_per_CTA']} c{row['chains']} q{row['batch_instructions']} wait{row['wait']} {row['operand_source_form']} n{row['iterations']} [{row['case_id'][-8:]}]")
                else:
                    labels.append(f"t{row['threads']} c{row['chains']} q{row['batch_instructions']} lanes{row['packed_lanes']} n{row['iterations']} [{row['case_id'][-8:]}]")
            ax.set_yticks(range(len(chunk)),labels);ax.invert_yaxis();ax.set_xlabel(unit);ax.grid(axis='x',alpha=.2)
            ax.set_title(f"GH200 {contract['family']}: {dtype}, {scope}\nAll accepted samples; median and min/max; C review pending")
            fig.tight_layout();name=f'{dtype}-{scope}-{page+1}'
            fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.svg'),metadata={'Date':None});plt.close(fig);figures.append(name+'.png')
    shutil.copyfile(__file__,output/Path(__file__).name)
    (output/'index.md').write_text('# Compute figures\n\nC review pending. Empty timing controls are in cases.csv; they are not FLOP rates.\n\n'+'\n\n'.join(f'![{name}]({name})' for name in figures)+'\n')
    manifest={'schema_version':1,'qualification':'analysis_pending_independent_C_review','family':contract['family'],
        'summary_sha256':sha(root/'summary.json'),'replay_sha256':sha(replay_path),'input_manifest_sha256':replay['input_manifest_sha256'],
        'matplotlib_version':matplotlib.__version__,'case_count':len(case_rows),'sample_rows':len(records),
        'error_bars':'min/max of all warmup-converged process samples, not confidence intervals','parameter_export':False,
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--replay',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();render(args.run,args.replay,args.output)
