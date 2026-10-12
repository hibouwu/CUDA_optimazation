"""Address invariants for the prelisted m11 lane contrast; no fitted model."""
from collections import Counter
import hashlib
import json


def group_key(case):
    return f"op{case['op']}_b{case['width']}_w{case['warps']}_d{int(case['dependent'])}"


def address_features(case):
    words_per_lane = case['width'] // 4
    threads = 32 * case['warps']
    addresses = case['addresses_words']
    if len(addresses) != 8 or any(len(row) != threads for row in addresses):
        raise ValueError('eight complete address rows required')
    full_demand = 0
    subgroup_demand = 0
    distinct_words = set()
    bank_sets = []
    word_counts = []
    for row in addresses:
        for warp in range(case['warps']):
            banks = [set() for _ in range(32)]
            counts = Counter()
            for lane in range(32):
                word = row[warp * 32 + lane]
                if word % words_per_lane or not warp * 4096 <= word <= (warp + 1) * 4096 - words_per_lane:
                    raise ValueError('unaligned or out-of-region address')
                for at in range(word, word + words_per_lane):
                    banks[at % 32].add(at)
                    counts[at] += 1
                    distinct_words.add(at)
            full_demand += max(len(bank) for bank in banks)
            bank_sets.append([sorted(bank) for bank in banks])
            word_counts.append(sorted(counts.items()))
            subgroup_lanes = 32 // words_per_lane
            for start in range(0, 32, subgroup_lanes):
                subgroup_banks = [set() for _ in range(32)]
                for lane in range(start, start + subgroup_lanes):
                    word = row[warp * 32 + lane]
                    for at in range(word, word + words_per_lane):
                        subgroup_banks[at % 32].add(at)
                subgroup_demand += max(len(bank) for bank in subgroup_banks)
    return dict(
        full_warp_demand=full_demand,
        old_lane_subgroup_demand=subgroup_demand,
        requested_bytes_per_round=case['width'] * threads * 8,
        distinct_address_bytes=4 * len(distinct_words),
        bank_sets_sha256=hashlib.sha256(json.dumps(bank_sets).encode()).hexdigest(),
        address_multiplicities_sha256=hashlib.sha256(json.dumps(word_counts).encode()).hexdigest(),
        physical_batches_measured=False,
    )


def validate_plan(cases):
    if len(cases) != 48 or len({case['id'] for case in cases}) != 48:
        raise ValueError('exactly 48 distinct prelisted conditions required')
    indexed = {case['id']: case for case in cases}
    pairs = []
    for case in cases:
        if case['op'] != 0 or case['native'] or case['iterations'] not in (2048, 8192):
            raise ValueError('outside authorized diagnostic panel')
        features = address_features(case)
        for key, expected in case['static_features'].items():
            if features[key] != expected:
                raise ValueError('prelisted feature mismatch: ' + case['id'] + ':' + key)
        if '_contiguous_' not in case['id']:
            continue
        other = indexed[case['id'].replace('_contiguous_', '_interleaved_')]
        other_features = address_features(other)
        for key in ('op', 'width', 'warps', 'dependent', 'native', 'iterations', 'seed'):
            if case[key] != other[key]:
                raise ValueError('pair execution organization mismatch')
        for key in ('full_warp_demand', 'requested_bytes_per_round', 'distinct_address_bytes',
                    'bank_sets_sha256', 'address_multiplicities_sha256'):
            if features[key] != other_features[key]:
                raise ValueError('pair address invariant mismatch: ' + key)
        pairs.append(dict(contiguous=case['id'], interleaved=other['id'], invariant=features))
    if len(pairs) != 14:
        raise ValueError('expected 14 controlled layout pairs')
    return pairs
