"""Project an existing raw record to prefix-only input, without target scoring."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from r09_prefix_ordinal_clock import SETUP_FIELDS

CALL_PROTOCOL_FIELDS = (
    'warmup_policy', 'warmup_swizzle', 'warmup_fixed_calls', 'history_protocol',
    'history_source', 'source_copies', 'mirror_target_slot',
    'producer_tail_observer', 'producer_tail_prefix_flag', 'release_warp_observer',
    'epilogue_defer', 'a_row_rotation', 'second_prefill', 'first_supply_events',
    'producer_delay_ns', 'work_swap55_91', 'b_offset_bytes', 'b_allocation_extra_bytes',
)
PREFIX_ZERO_FIELDS = ('release_warp_warmup', 'epilogue_warmup_defer', 'second_prefill_warmup')


def extract_prefix_input(setup, call, binary_sha256):
    for name in CALL_PROTOCOL_FIELDS:
        if type(call[name]) is not type(setup[name]) or call[name] != setup[name]:
            raise ValueError('setup/call protocol mismatch: ' + name)
    if type(call['target_swizzle']) is not type(setup['swizzle']) or call['target_swizzle'] != setup['swizzle']:
        raise ValueError('setup/call protocol mismatch: target_swizzle')
    for name in PREFIX_ZERO_FIELDS:
        if type(call[name]) is not int or call[name] != 0:
            raise ValueError('unsupported prefix treatment: ' + name)
    for name in ('prefix_a_ptr', 'target_a_ptr', 'prefix_b_ptr', 'target_b_ptr'):
        if type(call[name]) is not int or call[name] <= 0:
            raise ValueError('positive integer buffer identities required')
    # These are execution-organization fields, not measured target results.
    derived = dict(binary_sha256=binary_sha256, prefix_calls=call['warmup_calls'],
        prefix_same_a=call['prefix_a_ptr'] == call['target_a_ptr'],
        prefix_same_b=call['prefix_b_ptr'] == call['target_b_ptr'],
        target_calls=call['target_calls'], preceding_target_calls=call['preceding_target_calls'])
    if (type(binary_sha256) is not str or len(binary_sha256) != 64
            or any(c not in '0123456789abcdef' for c in binary_sha256)):
        raise ValueError('recorded binary SHA256 required')
    projected = {}
    for name in SETUP_FIELDS:
        projected[name] = derived[name] if name in derived else setup[name]
    return dict(setup=projected, prefix_event_us=call['warmup_us'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--raw-sha256', required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('new prefix output path required')
    raw = args.raw.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.raw_sha256:
        raise ValueError('raw record differs from caller-pinned identity')
    content = gzip.decompress(raw) if args.raw.suffix == '.gz' else raw
    packets = {}
    for line in content.splitlines():
        packet = json.loads(line)
        name = packet['event']
        if name in packets:
            raise ValueError('duplicate raw event')
        packets[name] = packet
    if set(packets) != {'setup', 'call', 'check'}:
        raise ValueError('one setup/call/check record required')
    prefix = extract_prefix_input(packets['setup'], packets['call'], args.binary_sha256)
    # Protocol/value acceptance belongs to the pinned predictor. This extractor
    # does not inspect target measurements or reject on numerical outcomes.
    output = json.dumps(prefix, indent=2, allow_nan=False) + '\n'
    with args.output.open('x') as stream:
        stream.write(output)
    print(json.dumps(dict(raw_sha256=args.raw_sha256, binary_sha256=args.binary_sha256,
        prefix_input_sha256=hashlib.sha256(output.encode()).hexdigest(),
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        execution='post-acquisition prefix projection; not numeric precommit',
        target_validity_checked=False)))


if __name__ == '__main__':
    main()
