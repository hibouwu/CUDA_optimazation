"""cfg_b 运行前有效频率预测：冻结的静态特征候选，单位 GHz。"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def features(case, observer='dual'):
    m, n, k = (case[name] for name in ('m', 'n', 'k'))
    tiles = math.ceil(m/128)*math.ceil(n/128)
    active = min(132, tiles, case.get('sm_count') or 132)
    max_work = math.ceil(tiles/active)
    nominal_cycles = tiles*math.ceil(k/64)*512/132
    unique_mib = (m*k+n*k)*2/(1024**2)
    return [math.log2(nominal_cycles), math.log2(unique_mib), math.log2(k),
            active/132, tiles/(132*max_work), float(case.get('input_mode', 'dyadic') == 'zero'),
            float(case.get('input_mode', 'dyadic') == 'random'), float(observer == 'dual')]


def fit(screen):
    rows = screen['rows']
    if 'formula' in screen:
        selected = screen['formula']
        return dict(schema='cfg-b-frequency/2', method=selected['method'],
            parameters=selected['parameters'], input_modes=['dyadic', 'zero'],
            feature_names=['log2_nominal_cycles_per_132_sms', 'log2_source_mib', 'log2_k',
                'active_fraction', 'last_wave_utilization', 'input_zero', 'input_random', 'observer_dual'],
            selection={key: selected[key] for key in ('median_abs', 'max_abs', 'rms')},
            training_input_hashes=screen['input_hashes'], training_rows=len(rows),
            training_geometry_groups=len({tuple(row['case'][key] for key in ('m', 'n', 'k')) for row in rows}))
    selected = screen['candidates'][0]
    x = np.asarray([features(row['case'], row['observer']) for row in rows])
    y = np.asarray([row['ghz'] for row in rows])
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .5)
    normalized = (x-center)/scale
    length, ridge = selected['length'], selected['ridge']
    kernel = np.exp(-np.sum((normalized[:, None, :]-normalized[None, :, :])**2, axis=2)/(2*length**2))
    mean = y.mean()
    weights = np.linalg.solve(kernel+ridge*np.eye(len(kernel)), y-mean)
    return dict(schema='cfg-b-frequency/1', method='RBF kernel ridge; geometry-group CV selection',
        feature_names=['log2_nominal_cycles_per_132_sms', 'log2_source_mib', 'log2_k',
            'active_fraction', 'last_wave_utilization', 'input_zero', 'input_random', 'observer_dual'],
        center=center.tolist(), scale=scale.tolist(), training_features=normalized.tolist(),
        mean_ghz=float(mean), weights=weights.tolist(), length=length, ridge=ridge,
        selection={key: selected[key] for key in ('median_abs', 'max_abs', 'rms')},
        training_input_hashes=screen['inputs'], training_rows=len(rows),
        training_geometry_groups=len({tuple(row['case'][key] for key in ('m', 'n', 'k')) for row in rows}))


def predict(model, case, observer='dual'):
    if model['schema'] == 'cfg-b-frequency/2':
        if case.get('input_mode', 'dyadic') not in model['input_modes']:
            raise ValueError('frequency/2 supports dyadic and zero inputs')
        x = features(case, observer)
        maximum_dyadic, amplitude_dyadic, maximum_zero, amplitude_zero, power, onset, scale, dual = model['parameters']
        maximum = maximum_zero if x[5] else maximum_dyadic
        amplitude = amplitude_zero if x[5] else amplitude_dyadic
        response = 1-math.exp(-max(0., 2**x[0]/1e5-onset)/scale)
        return maximum-amplitude*x[4]**power*response+dual*x[7]
    x = (np.asarray(features(case, observer))-model['center'])/model['scale']
    kernel = np.exp(-np.sum((np.asarray(model['training_features'])-x)**2, axis=1)/(2*model['length']**2))
    return float(model['mean_ghz']+kernel@model['weights'])


class StaticFrequency:
    """接入聚合组合的频率模块；包络只消费预测频率。"""
    version = 'cfg-b-static-frequency-rbf-1'

    def __init__(self, program, model):
        case = program['case']
        self.version = model['schema']
        self.frequency_ghz = predict(model, case)
        self.point = dict(zip(model['feature_names'], features(case)))
        self.zero_fraction = float(case['input_mode'] == 'zero')
        self.mixing = False

    def solve(self, envelope_us):
        return dict(window_us=envelope_us(self.frequency_ghz),
                    frequency_ghz=self.frequency_ghz, closure=0.)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('screen', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    model = fit(json.loads(args.screen.read_text()))
    model['screen_sha256'] = hashlib.sha256(args.screen.read_bytes()).hexdigest()
    model['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(model, indent=2)+'\n')
    print(model['training_rows'], model['selection'])
