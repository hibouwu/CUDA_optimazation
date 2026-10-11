"""Frozen development rule evaluated using only a declared completed prefix.

This can score a future locally frozen experiment without modifying its binary.
It does not establish observer validity or demonstrate an online runner hook.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import r09_prefix_ordinal_clock as clock


def source_hashes():
    return {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (__file__, clock.__file__)}


def predict_from_prefix(rule, prefix):
    """Produce local-window predictions; current target data is inadmissible."""
    if rule.get('schema') != 'r09-frozen-prefix-clock-rule/1' or rule.get('method') != 'common':
        raise ValueError('frozen common-coefficient rule required')
    if rule['source_sha256'] != source_hashes():
        raise ValueError('prediction source differs from frozen rule')
    if set(prefix) != {'setup', 'prefix_event_us'}:
        raise ValueError('only static setup and completed prefix events are allowed')
    if set(prefix['setup']) != set(clock.SETUP_FIELDS):
        raise ValueError('exact static setup required; target observations are inadmissible')
    events = prefix['prefix_event_us']
    if type(events) is not list or len(events) != 16:
        raise ValueError('all sixteen declared prefix events required')
    parsed = []
    for value in events:
        if type(value) not in (int, float, str):
            raise ValueError('finite positive prefix events required')
        try:
            event = Fraction(str(value))
        except (ValueError, ZeroDivisionError):
            raise ValueError('finite positive prefix events required') from None
        if event <= 0:
            raise ValueError('finite positive prefix events required')
        parsed.append(event)
    calibration = rule['calibration']
    rates = clock.predict_rates(prefix['setup'], parsed[-1], calibration, ordinal_profile=False)
    reference = rule['native_reference']
    if reference['gpu_uuid'] != prefix['setup']['gpu_uuid']:
        raise ValueError('independent native reference must match the target card')
    if (reference['iterations'] != 1024 or reference['work_flop'] != 4294967296
            or reference['protocol'] != 'cfg-c-mma-reference-v1'):
        raise ValueError('frozen long instruction reference required')
    samples = reference['cycle_samples']
    if len(samples) != 3 or any(type(v) is not int or v <= 0 for v in samples):
        raise ValueError('three positive native reference samples required')
    setup = prefix['setup']
    work = 2 * setup['tile'][0] * setup['tile'][1] * setup['k']
    cycles = Fraction(work * sorted(samples)[1], reference['work_flop'])
    return dict(schema='r09-frozen-prefix-clock-prediction/1',
        status='experimental_local_common_window_only',
        prefix_calls=len(parsed),last_prefix_event_us=str(parsed[-1]),
        work_flop=work,native_cycles=str(cycles),
        cycle_per_ns=[str(v) for v in rates],prediction_ns=[str(cycles/v) for v in rates],
        observation_qualified=False,prediction_qualified=False,
        prospective_holdout_qualified=False,cross_card_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule', type=Path, required=True)
    parser.add_argument('--rule-sha256', required=True)
    parser.add_argument('--prefix-input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new prediction output path required')
    rule_bytes = args.rule.read_bytes()
    rule_sha = hashlib.sha256(rule_bytes).hexdigest()
    if rule_sha != args.rule_sha256:
        raise ValueError('rule differs from caller-pinned freeze SHA')
    prefix_bytes = args.prefix_input.read_bytes()
    prediction = predict_from_prefix(json.loads(rule_bytes), json.loads(prefix_bytes))
    prediction.update(rule_sha256=rule_sha,
        prefix_input_sha256=hashlib.sha256(prefix_bytes).hexdigest())
    with args.output.open('x') as stream:
        stream.write(json.dumps(prediction, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
