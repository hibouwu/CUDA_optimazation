"""S18 integer/bit-pattern reference. No CUDA and no timing/workload aggregation.

Long recurrences use affine exponentiation or closed forms, independently of
iteration-by-iteration device loops. This module alone grants no GPU admission.
"""
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def unsigned(value, bits, name):
    if type(value) is not int or not 0 <= value < (1 << bits):
        raise ValueError(name + ' must be an unsigned integer of the declared width')
    return value


def initial(seed, thread, word):
    unsigned(seed, 32, 'seed')
    unsigned(thread, 32, 'thread')
    unsigned(word, 32, 'word')
    return (seed + 65537 * (thread + 1) + 257 * (word + 1)) & MASK32


def affine(value, multiplier, addend, steps, bits=32):
    if type(steps) is not int or steps < 0 or bits not in (32, 64):
        raise ValueError('nonnegative integer steps and 32/64-bit domain required')
    mask = (1 << bits) - 1
    total_multiplier, total_addend = 1, 0
    while steps:
        if steps & 1:
            total_addend = (multiplier * total_addend + addend) & mask
            total_multiplier = multiplier * total_multiplier & mask
        addend = (multiplier * addend + addend) & mask
        multiplier = multiplier * multiplier & mask
        steps >>= 1
    return (total_multiplier * value + total_addend) & mask


def positive_iterations(iterations):
    if type(iterations) is not int or iterations < 1:
        raise ValueError('positive integer iteration count required')


def word_result(seed, thread, word, iterations):
    positive_iterations(iterations)
    return affine(initial(seed, thread, word), 1664525, 1013904223, iterations)


def service_result(operation, seed, thread, stream, iterations):
    positive_iterations(iterations)
    first = initial(seed, thread, stream)
    if operation == 'add_u64':
        start = (first << 32) | initial(seed, thread, stream + 17)
        increment = ((seed << 32) + 65537 * (thread + 1) + 257 * (stream + 1)) | 1
        return (start + iterations * increment) & MASK64
    if operation == 'mad_wide_u32':
        multiplier = initial(seed ^ 0xa5a5a5a5, thread, stream) | 1
        addend = (initial(seed ^ 0x5a5a5a5a, thread, stream) << 32) | initial(seed ^ 0xc3c3c3c3, thread, stream)
        # All earlier r values contribute only low32 to the next request.
        previous = affine(first, multiplier, addend & MASK32, iterations - 1)
        return (previous * multiplier + addend) & MASK64
    if operation in ('cvt_rn_f16_f32', 'cvt_f32_f16'):
        salt = (initial(seed ^ 0x9e3779b9, thread, stream) & 1023) | 1
        q = ((first & 1023) + iterations * salt + iterations * (iterations - 1) // 2) & 1023
        return (0x3c00 | q) if operation == 'cvt_rn_f16_f32' else (0x3f800000 | (q << 13))
    raise ValueError('unknown S18 operation: ' + str(operation))


def atomic_results(seed, threads, same_word, iterations):
    positive_iterations(iterations)
    if threads != 128 or type(threads) is not int or type(same_word) is not bool:
        raise ValueError('S18 atomic matrix fixes 128 threads and a boolean address pattern')
    if same_word:
        increment = sum(1 + thread % 7 for thread in range(threads))
        return [(initial(seed, 0, 0) + iterations * increment) & MASK32]
    return [(initial(seed, 0, thread) + iterations * (1 + thread % 7)) & MASK32
            for thread in range(threads)]
