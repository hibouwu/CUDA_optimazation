"""WGMMA 背景对照：核对实际重叠，并按控制流分解同一窗口。"""
import argparse
import json
from pathlib import Path
import statistics as st

from analyze_tma_tensor_probe import analyze


def control_partition(sample, unit):
    """均值保持可加性；所有分项是软件打点区间。"""
    events = {(row['cta'], row['group']): row['points'] for row in sample['group_events']}
    q, count = sample['active_groups'], sample['groups']
    rows = []
    for cta in sample['ctas_data']:
        points = [events[cta['cta'], group] for group in range(count)]
        parts = dict(prefix=points[0][0][unit]-cta['begin'][unit],
            issue=sum(p[1][unit]-p[0][unit] for p in points),
            prefill_gap=sum(points[g][0][unit]-points[g-1][1][unit] for g in range(1, q)),
            wait_entry=0, wait=sum(p[3][unit]-p[2][unit] for p in points),
            consumer=sum(p[4][unit]-p[3][unit] for p in points),
            refill_gap=sum(points[g+q][0][unit]-points[g][4][unit] for g in range(count-q)),
            tail=cta['end'][unit]-points[-1][4][unit])
        for group, point in enumerate(points):
            previous = points[group+q-1][1][unit] if group+q <= count else points[group-1][4][unit]
            parts['wait_entry'] += point[2][unit]-previous
        assert min(parts.values()) >= 0, (cta['cta'], parts)
        total = cta['end'][unit]-cta['begin'][unit]
        assert sum(parts.values()) == total, (cta['cta'], sum(parts.values()), total)
        rows.append(dict(parts, total=total))
    return {name: st.fmean(row[name] for row in rows) for name in rows[0]}


def report(root):
    result = analyze(root)
    for row in result['rows']:
        for process in row['processes']:
            sample = json.loads((root/'samples'/process['file']).read_text())
            assert sample['background_errors'] == 0 and sample['background_overlap'] == 1
            ctas = sample['ctas_data']
            assert all(c['background_begin'][1] <= c['begin'][1] <= c['end'][1] <= c['background_end'][1]
                       for c in ctas)
            if sample['background']:
                assert all(c['background_rounds'] > 0 for c in ctas)
            process['background_rounds'] = [min(c['background_rounds'] for c in ctas),
                                            max(c['background_rounds'] for c in ctas)]
            if sample['trace']:
                process['control_partition'] = {unit: control_partition(sample, i)
                                                for i, unit in enumerate(('cycles', 'ns'))}
    contrasts = []
    for after in result['rows']:
        if after['arguments']['background'] == 0:
            continue
        baseline = dict(after['arguments'], background=0)
        before = next(row for row in result['rows'] if row['arguments'] == baseline)
        response = {key: after['summary'][key]['median']/before['summary'][key]['median']-1
                    for key in ('median_cta_cycles', 'median_cta_ns', 'effective_ghz')}
        paired = []
        for b in after['processes']:
            a = next(p for p in before['processes'] if p['repeat'] == b['repeat'])
            pair = dict(repeat=b['repeat'], window_ns_response=b['median_cta_ns']/a['median_cta_ns']-1,
                        window_cycle_response=b['median_cta_cycles']/a['median_cta_cycles']-1)
            if after['arguments']['trace']:
                pair['control_delta'] = {unit: {key: b['control_partition'][unit][key]-a['control_partition'][unit][key]
                    for key in a['control_partition'][unit]} for unit in ('cycles', 'ns')}
            paired.append(pair)
        delta = None
        if after['arguments']['trace']:
            delta = {unit: {key: st.fmean(p['control_delta'][unit][key] for p in paired)
                for key in paired[0]['control_delta'][unit]} for unit in ('cycles', 'ns')}
        contrasts.append(dict(before=before['id'], after=after['id'], response=response,
                              mean_control_delta=delta, paired=paired))
    result.update(contrasts=contrasts, interpretation='分项为同一探针的可加软件区间；硬件服务归因另由候选干预检验')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for contrast in result['contrasts']:
        print(contrast['after'], contrast['response'])
        print('mean control delta:', contrast['mean_control_delta'])
