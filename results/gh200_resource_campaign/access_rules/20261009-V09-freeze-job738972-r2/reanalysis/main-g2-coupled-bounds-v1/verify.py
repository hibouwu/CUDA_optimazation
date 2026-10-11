"""Verify the conditional G2 extrema by a one-dimensional parameter polytope."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    report = json.loads(args.report.read_text())
    for relative, digest in report['input_sha256'].items():
        assert hashlib.sha256((run/relative).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((args.report.parent.parent/'analyze.py').read_bytes()).hexdigest() == report['analysis_sha256']
    cal = json.loads((run/'frozen/calibration.json').read_text())
    base = np.array(cal['first_supply']['cfg_c']['parameters'])
    constraints = report['identification']['constraints']
    eq = np.array(constraints['A_eq'])
    u, singular, vh = np.linalg.svd(eq, full_matrices=True)
    assert len(singular) == 4 and min(singular) > 1e-5
    direction = vh[-1]
    assert np.max(abs(eq@direction)) < 1e-12
    A = np.vstack([constraints['A_ub'], -np.eye(len(base))])
    b = np.r_[constraints['b_ub'], np.zeros(len(base))]
    low, high = -np.inf, np.inf
    for coefficient, bound in zip(A, b):
        slope, slack = coefficient@direction, bound-coefficient@base
        if abs(slope) < 1e-10:
            assert slack >= -1e-7
        elif slope > 0:
            high = min(high, slack/slope)
        else:
            low = max(low, slack/slope)
    assert np.isfinite(low) and np.isfinite(high) and low <= 0 <= high
    alpha, offset = report['recurrence']['alpha'], report['recurrence']['offset_ns']
    features = np.array(report['unique_request_features'])
    compute, kt = 1024., 16
    # On this entire line segment, every CTA duration is a maximum of affine
    # functions of t. Its minimum occurs at an endpoint or pair intersection;
    # its maximum occurs at an endpoint. No LP solver is used here.
    def extrema(frequency):
        branches = []
        for z in features:
            for a, b0 in [(np.r_[alpha, np.zeros(len(base)-1)], offset+alpha*kt*compute/frequency),
                          (np.r_[alpha, alpha*kt*z], offset)]:
                branches.append((float(a@direction), float(a@base+b0)))
        candidates = [low, high]
        for i,(a,b0) in enumerate(branches):
            for c,d in branches[:i]:
                if abs(a-c) > 1e-10:
                    cross = (d-b0)/(a-c)
                    if low <= cross <= high:
                        candidates.append(cross)
        value = lambda t: max(a*t+b0 for a,b0 in branches)/1000
        minimizer = min(candidates, key=value)
        maximizer = max((low,high), key=value)
        return [(value(t),t) for t in (minimizer,maximizer)]

    sys.path.insert(0, str(run/'source'))
    import r09_input_source_clock as clock
    point, model = report['static_clock_work'], report['source_clock']
    roots = {}
    for index,name in enumerate(('minimum','maximum')):
        left, right = 0., model['coefficients'][0]
        for _ in range(100):
            frequency = (left+right)/2
            window,t = extrema(frequency)[index]
            mapped = clock.source_frequency(point,window,model)['frequency_ghz']
            if mapped > frequency:
                left = frequency
            else:
                right = frequency
        frequency = (left+right)/2
        window,t = extrema(frequency)[index]
        expected = report['coupled'][name]
        assert abs(window-expected['window_us']) < 1e-7
        assert abs(frequency-expected['frequency_ghz']) < 1e-8
        theta = np.array(expected['shared_parameters'])
        witness_t = float((theta-base)@direction)
        assert np.max(abs(theta-(base+witness_t*direction))) < 1e-7
        assert low-1e-7 <= witness_t <= high+1e-7
        roots[name] = dict(window_us=window,frequency_ghz=frequency,t=t,
            reported_parameter_t=witness_t,reported_window_difference_us=window-expected['window_us'])
    # Check interior feasible models against the coupled interval by executing
    # the original full predictor, independently of the analytic envelope.
    import v09_model as v09
    import copy
    setup = next(x['setup'] for x in json.loads((run/'freeze-setup.json').read_text()) if x['case']==report['case']['id'])
    checked=[]
    for t in np.linspace(low,high,21):
        theta = base+t*direction
        assert np.min(theta) >= -1e-7
        candidate = copy.deepcopy(cal)
        candidate['first_supply']['cfg_c']['parameters'] = theta.tolist()
        result = v09.predict_components(report['case'],setup,candidate)
        assert roots['minimum']['window_us']-1e-7 <= result['window_us'] <= roots['maximum']['window_us']+1e-7
        checked.append(dict(t=float(t),window_us=result['window_us']))
    out = dict(method='SVD rank4 yields a1D affine parameter set; halfspaces give its full bounded segment; affine-branch intersections solve each convex envelope without LP.',
        singular_values=singular.tolist(),direction=direction.tolist(),parameter_interval=[low,high],
        roots=roots,interior_predictor_checks=checked,
        input_sha_verified=len(report['input_sha256']),
        report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observation_qualified=False,prediction_qualified=False)
    with args.output.open('x') as stream:
        stream.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(roots,indent=2))


if __name__ == '__main__':
    main()
