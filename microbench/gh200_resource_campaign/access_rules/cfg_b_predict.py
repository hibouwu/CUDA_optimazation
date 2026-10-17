"""用共同 case/setup 输入和冻结模块预测 cfg_b 的逐端点与完整时间。"""
import argparse
import hashlib
import json
from pathlib import Path

import cfg_b_composition as composition
import cfg_b_frequency
import cfg_b_program


def predict_cases(cases, setups, calibration, frequency_model):
    by_case = {row['case']: row['setup'] for row in setups}
    predictions = []
    for case in cases:
        program = cfg_b_program.build_program(case, by_case[case['id']])
        services = {'frequency': cfg_b_frequency.StaticFrequency(program, frequency_model)}
        predictions.append(dict(id=case['id'],
            prediction=composition.predict(program, calibration, services)))
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--setups', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--frequency-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in ('cases', 'setups', 'calibration', 'frequency_model')}
    contents = {name: json.loads(path.read_text()) for name, path in inputs.items()}
    predictions = predict_cases(contents['cases'], contents['setups'], contents['calibration'], contents['frequency_model'])
    report = dict(schema='cfg-b-time-prediction/1', qualification='半经验：V09 聚合组合替换运行前频率',
        frequency_model=contents['frequency_model']['schema'],
        input_sha256={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for name, path in inputs.items()},
        implementation_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (composition, cfg_b_program, cfg_b_frequency)}, predictions=predictions)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'{len(predictions)} cfg_b predictions written to {args.output}')


if __name__ == '__main__':
    main()
