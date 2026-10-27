#!/usr/bin/env python3
"""普通arrival短链的局部时间线与已有plain响应；CPU，无参数拟合。"""
import argparse
import json
from pathlib import Path
import statistics

from cfg_b_handoff_chain import timed_lifecycle, validate
from l2_saturation import digest, read_json, require


def build_report(repo):
    inputs = {}

    def read(path, binary=False):
        inputs[str(path.relative_to(repo))] = digest(path)
        return path.read_bytes() if binary else read_json(path)

    observations = {}
    raw_processes = full_elements = 0
    prefix = repo/'results/gh200_resource_campaign/access_rules'
    names = ('Q04-sync-slot/SYNC-01-02-control/20261017-Q04-index-control-002', 'Q04-sync-slot/SYNC-01-02-control/20261017-Q04-control-localization-001')
    for name in names:
        root = prefix/name
        plan, score = read(root/'plan.json'), read(root/'score.json')
        read(root/'build/target-review.json')
        read(root/'source/cfg_b_handoff_chain.cu', binary=True)
        samples = {}
        for case in plan['cases']:
            if case['index_mode'] != 'incremental':
                continue
            for seed in plan['seeds']:
                folder = root/'runs'/case['id']/f'seed{seed}'
                raw = read(folder/'stdout')
                receipt = read(folder/'receipt.json')
                require(receipt['returncode'] == 0 and not receipt['timed_out'], '原始执行未成功')
                require(raw['index_mode'] == 'incremental', '原始索引模式不符')
                read(folder/raw['output_file'], binary=True)
                checked = validate(folder, raw, case, seed)
                require(checked['performance_eligible'], '原始性能资格未通过')
                frozen = next(row for row in score['rows'] if row['case_id'] == case['id'] and row['seed'] == seed)
                cycles = raw['timing']['elapsed_cycles']
                require(cycles == frozen['cycles'] and raw['device']['uuid'] == frozen['device_uuid'],
                        '原始周期或卡身份与评分不符')
                samples[case['ktiles'], case['delay_cycles'], seed] = cycles
                raw_processes += 1
                full_elements += checked['output_witness']['checked_elements']
        observations[name] = samples
    first, second = [observations[name] for name in names]
    measured_delay = statistics.median(first[13, 4096, seed]-first[13, 0, seed] for seed in (3, 22, 41))
    measured_growth = statistics.median(second[13, 0, seed]-second[4, 0, seed] for seed in (3, 22, 41))
    freeze = read(prefix/'Q05-wgmma/local-compute-chain/20261018-Q05-compute-wait-001/dev-freeze.json')
    lag = freeze['parameters']['completion_lag_cycles']
    require(freeze['selected'] == 'per_group_lag' and lag == 95, '驻留完成版本来源不符')
    profiles, comparisons = {}, []
    calls = 0
    for name, completion_lag in (('d0', 0), ('resident_d95', lag)):
        predictions = {}
        for stages in (4, 6):
            for length in (4, 13, 32):
                for delay in (0, 4096):
                    key = f's{stages}_k{length}_d{delay}'
                    predictions[key] = timed_lifecycle(length, stages, delay,
                        completion_lag_cycles=completion_lag)
                    calls += 1
        end = lambda key: predictions[key]['local_end_cycle']
        response = end('s6_k13_d4096')-end('s6_k13_d0')
        growth = end('s6_k13_d0')-end('s6_k4_d0')
        profiles[name] = predictions
        comparisons.append(dict(profile=name, completion_lag_cycles=completion_lag,
            length_growth=dict(predicted_cycle=growth, observed_cycle=measured_growth,
                relative_error=abs(growth-measured_growth)/abs(measured_growth),
                passes_five_percent=abs(growth-measured_growth) <= .05*abs(measured_growth)),
            delay_response=dict(predicted_cycle=response, observed_cycle=measured_delay,
                relative_error_to_prediction=abs(response-measured_delay)/abs(response),
                passes_original_five_percent=abs(response-measured_delay) <= .05*abs(response)),
            composite_window_remainder_cycle={str(length): statistics.median(
                second[length, 0, seed] for seed in (3, 22, 41))-end(f's6_k{length}_d0')
                for length in (4, 13)}))
    excluded = all(not row['delay_response']['passes_original_five_percent'] for row in comparisons)
    conclusion = ('两个零额外控制完成版本均未解释递增回绕的延迟响应，停止向GEMM扩大；保留局部接口'
                  if excluded else '局部响应保留相容版本，后续独立验证未完成')
    return dict(schema='gh200-q04-local-handoff-model/1', unit='SM clock64 cycle',
        qualification='已有开发证据回放；未采形状为工作假设时间线',
        inputs=inputs, raw_processes_checked=raw_processes, full_output_elements_checked=full_elements,
        local_model_calls=calls, parameter_fits=0, gpu_jobs=0,
        comparisons=comparisons, timelines=profiles,
        conclusion=conclusion,
        unresolved=['消费者控制与full等待返回', 'empty最后必要arrival及producer返回'],
        independent_C2=False, independent_C3=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.repo.resolve())
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('raw_processes_checked',
        'full_output_elements_checked', 'local_model_calls', 'comparisons', 'conclusion')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
