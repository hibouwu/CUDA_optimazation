"""按冻结判据评分 tensor 首批条件；同一价格组须同时通过全部条件。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

from analyze_tma_tensor_probe import analyze


def score(root):
    binding = json.loads((root/'binding.json').read_text())
    for name, expected in binding['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
    plan = json.loads((root/'plan.json').read_text())
    predictions = json.loads((root/'predictions.json').read_text())
    invocations = json.loads((root/'invocations.json').read_text())
    from datetime import datetime
    frozen_ns = int(datetime.fromisoformat(binding['frozen_at']).timestamp()*1e9)
    assert min(i['started_ns'] for i in invocations) > frozen_ns
    observations = analyze(root)
    rows = {r['id']:r for r in observations['rows']}
    paired = predictions['predictions'][-2:]
    before, after = [rows[p['id']+'-t1'] for p in paired]
    observed_response = after['summary']['release_window_ns']['median']/before['summary']['release_window_ns']['median']-1
    plain_before, plain_after = [rows[p['id']+'-t0'] for p in paired]
    bridge_response = plain_after['summary']['median_cta_ns']['median']/plain_before['summary']['median_cta_ns']['median']-1
    paired_responses = []
    for i in range(plan['processes']):
        a = next(r for r in before['processes'] if r['repeat']==i)
        b = next(r for r in after['processes'] if r['repeat']==i)
        paired_responses.append(b['release_window_ns']/a['release_window_ns']-1)
    candidates = {}
    threshold = plan['scoring']['max_absolute_relative_error']
    tolerance = plan['scoring']['phase_response_tolerance']
    for candidate, parameters in predictions['retained'].items():
        scores = []
        for index in range(len(parameters)):
            errors = []
            for p in predictions['predictions']:
                predicted = p['candidates'][candidate][index]['median_cta_ns']
                observed = rows[p['id']+'-t1']['summary']['release_window_ns']['median']
                errors.append(dict(id=p['id'],predicted_ns=predicted,observed_ns=observed,error=predicted/observed-1))
            phase = errors[-1]['predicted_ns']/errors[-2]['predicted_ns']-1
            maximum = max(abs(e['error']) for e in errors)
            scores.append(dict(parameter_index=index,median_absolute_error=st.median(abs(e['error']) for e in errors),
                max_absolute_error=maximum,predicted_phase_response=phase,
                phase_response_error=phase-observed_response,
                pass_joint_time=maximum<=threshold,pass_phase=abs(phase-observed_response)<=tolerance,errors=errors))
        bridge = bridge_response>=.15 if candidate=='shared_source_fifo' else abs(bridge_response)<=.05
        passing = [s['parameter_index'] for s in scores if s['pass_joint_time'] and s['pass_phase'] and bridge]
        candidates[candidate] = dict(pass_trace_bridge=bridge,passing_parameter_indices=passing,
            best_joint_diagnostic=min(scores,key=lambda s:s['max_absolute_error']),parameter_scores=scores)
    for name, result in candidates.items():
        excluded = all(not other['passing_parameter_indices'] for key,other in candidates.items() if key!=name)
        result['conclusion'] = '支持：声明的冻结价格集合与条件内' if result['passing_parameter_indices'] and excluded else (
            '冻结价格集合未通过' if not result['passing_parameter_indices'] else '信息不足：竞争解释未排除')
    return dict(schema='tensor-discrimination-score/1',frozen_at=binding['frozen_at'],
        primary_phase_response=observed_response,paired_process_responses=paired_responses,
        untraced_phase_response=bridge_response,candidates=candidates,observations=observations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=score(args.root)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('phase response traced/untraced:',report['primary_phase_response'],report['untraced_phase_response'])
    for name,r in report['candidates'].items():
        print(name,r['conclusion'],'passing sets',len(r['passing_parameter_indices']),
              'best joint max error',r['best_joint_diagnostic']['max_absolute_error'])
