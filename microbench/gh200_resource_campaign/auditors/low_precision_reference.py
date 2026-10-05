"""Independent, CPU-only S08 scalar math and documented fragment coordinates."""
from __future__ import annotations
from fractions import Fraction


def fp8_value(bits, encoding):
    if type(bits) is not int or not 0 <= bits < 256 or encoding not in ('e4m3', 'e5m2'):
        raise ValueError('FP8 encoding/byte')
    mantissa_bits, bias = (3, 7) if encoding == 'e4m3' else (2, 15)
    exponent = (bits & 127) >> mantissa_bits
    fraction = bits & ((1 << mantissa_bits) - 1)
    if (encoding == 'e5m2' and exponent == 31) or (encoding == 'e4m3' and exponent == 15 and fraction == 7):
        raise ValueError('nonfinite FP8 excluded')
    significand = fraction if exponent == 0 else (1 << mantissa_bits) + fraction
    shift = (1 - bias if exponent == 0 else exponent - bias) - mantissa_bits
    value = Fraction(significand * (1 << max(shift, 0)), 1 << max(-shift, 0))
    return -value if bits & 128 else value


def packed_word(values, encoding):
    """Small exact representable test values; no rounded conversion oracle."""
    if len(values) != 4:
        raise ValueError('four 8-bit lanes required')
    word = 0
    for lane, value in enumerate(values):
        if encoding in ('s8', 'u8'):
            lo, hi = (-128, 127) if encoding == 's8' else (0, 255)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('integer encoding range')
            bits = value & 255
        else:
            matches = []
            for byte in range(256):
                try:
                    if fp8_value(byte, encoding) == value:
                        matches.append(byte)
                except ValueError:
                    pass
            if not matches:
                raise ValueError('not exactly representable FP8')
            bits = min(matches)  # canonical +0
        word |= bits << (8 * lane)
    return word


def coordinate(path, role, thread, element):
    if path == 'mma':
        lane = thread
        if role == 'A':
            return lane // 4 + 8 * ((element // 4) % 2), 4 * (lane % 4) + element % 4 + 16 * (element // 8)
        if role == 'B':
            return 4 * (lane % 4) + element % 4 + 16 * (element // 4), lane // 4
        if role == 'D':
            return lane // 4 + 8 * (element // 2), 2 * (lane % 4) + element % 2
    elif path == 'wgmma' and role == 'D':
        return 16 * (thread // 32) + (thread % 32) // 4 + 8 * ((element // 2) % 2), 2 * (thread % 4) + element % 2 + 8 * (element // 4)
    raise ValueError('unknown register fragment')


def input_value(encoding, outer, inner, seed, a=True):
    index = outer + (2 if a else 3) * inner + seed
    if encoding in ('e4m3', 'e5m2'):
        return Fraction(index % 7 - 3, 8)
    if encoding == 's8':
        return index % 5 - 2
    if encoding == 'u8':
        return index % 3
    raise ValueError('unknown input type')


def logical_reference(encoding, row, col, group, chain, iterations, seed, nonuniform):
    if not nonuniform:
        return Fraction(iterations * 16 * 32, 256) if encoding in ('e4m3', 'e5m2') else iterations * 16 * 32
    if iterations not in (1, 2):
        raise ValueError('nonuniform validation bounded to 1/2 iterations')
    dot = sum(input_value(encoding, row, k, seed) * input_value(encoding, col, k, seed, False) for k in range(32))
    d0 = Fraction(1 + group + chain, 8) if encoding in ('e4m3', 'e5m2') else 1 + group + chain
    return dot * iterations * 16 + d0


def output_reference(encoding, path, groups, iterations, seed, nonuniform):
    width, count = (32, 4) if path == 'mma' else (128, 32)
    cache = {}
    result = []
    for thread in range(groups * width):
        for chain in range(2):
            for element in range(count):
                row, col = coordinate(path, 'D', thread % width, element)
                key = row, col, thread // width, chain
                if key not in cache:
                    cache[key] = logical_reference(encoding, *key, iterations, seed, nonuniform)
                result.append(cache[key])
    return result
