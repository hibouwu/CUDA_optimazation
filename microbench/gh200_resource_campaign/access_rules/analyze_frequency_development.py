"""复核独立 plain/dual 进程，保留同一 SM 匹配窗口的周期、ns 与频率。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics as st

import analyze_r18
import v06_run


def summary(values):
    return dict(n=len(values), median=st.median(values), minimum=min(values), maximum=max(values),
                cv=st.pstdev(values)/st.fmean(values))


def analyze(root):
    v06_run.verify(root)
    config = json.loads((root/'run_config.json').read_text())
    assert hashlib.sha256((root/'cases.json').read_bytes()).hexdigest() == config['cases_sha256']
    cases = json.loads((root/'cases.json').read_text())
    setup = {row['case']: row['setup'] for row in json.loads((root/'static_setup.json').read_text())}
    environment = json.loads((root/'environment.json').read_text())
    assert environment['gpu'].split(',')[0] == config['reference_gpu']
    rows = []
    for case in cases:
        failed, processes = [], []
        records = [json.loads(path.read_text()) for path in sorted((root/'samples'/case['id']).glob('*.json'))]
        for record in records:
            if record['returncode']:
                failed.append(record)
                continue
            measured = analyze_r18.replay(root, record, case, retain_roles=True)
            process = dict(variant=record['variant'], trial=record['trial'], attempt=record['attempt'],
                host_start_ns=record['host_start_ns'], event_us=measured['elapsed_us'], raw=record['raw'],
                raw_sha256=record['raw_sha256'])
            if record['variant'] == 'dual':
                active = [cta for cta in measured['ctas'] if cta['tiles']]
                largest = max(len(cta['tiles']) for cta in active)
                frequencies, cycles, mainloop_frequencies = [], [], []
                for cta in active:
                    # 同一消费者的 final_c/final_ns 配对，避免独立选择两个时钟域的最大值。
                    last = max(cta['roles'], key=lambda role: role['final_ns'])
                    dc = last['final_c']-cta['entry_c']
                    dn = last['final_ns']-cta['entry_ns']
                    if len(cta['tiles']) == largest:
                        frequencies.append(dc/dn)
                        cycles.append(dc)
                    for role in cta['roles']:
                        for tc, tn in zip(role['tiles'], role['tiles_ns']):
                            if tn[1]-tn[0] >= 3200:
                                mainloop_frequencies.append((tc[1]-tc[0])/(tn[1]-tn[0]))
                first = min(cta['entry_ns'] for cta in measured['ctas'])
                last = max(cta['end_ns'] for cta in active)
                process.update(frequency_ghz=st.median(frequencies),
                    largest_work_cta_cycles=st.median(cycles), largest_work_tiles=largest,
                    window_us=(last-first)/1000, active_ctas=len(active),
                    distinct_sms=len({cta['sm'] for cta in active}),
                    common_overlap_ns=min(cta['end_ns'] for cta in active)-max(cta['entry_ns'] for cta in active),
                    mainloop_frequency_ghz=st.median(mainloop_frequencies) if mainloop_frequencies else None,
                    mainloop_frequency_windows=len(mainloop_frequencies))
            processes.append(process)
        for variant in config['variants']:
            trials = sorted(p['trial'] for p in processes if p['variant'] == variant)
            assert trials == list(range(config['processes_per_variant'])), (case['id'], variant, trials)
        plain = [p for p in processes if p['variant'] == 'plain']
        dual = [p for p in processes if p['variant'] == 'dual']
        metrics = {field: summary([p[field] for p in dual]) for field in
                   ('frequency_ghz', 'largest_work_cta_cycles', 'window_us', 'event_us', 'common_overlap_ns')}
        metrics['plain_us'] = summary([p['event_us'] for p in plain])
        metrics['dual_minus_envelope_us'] = summary([p['event_us']-p['window_us'] for p in dual])
        rows.append(dict(case=case, setup=setup[case['id']], metrics=metrics, processes=processes,
                         failed_attempts=failed))
    return dict(schema='frequency-development/1', environment=environment, rows=rows,
        input_hashes={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ('cases.json', 'run_config.json', 'source_hashes.json', 'build/binary_hashes.json')},
        decoder_sha256=hashlib.sha256(Path(analyze_r18.__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for row in result['rows']:
        print(row['case']['id'], {name: metric['median'] for name, metric in row['metrics'].items()})
