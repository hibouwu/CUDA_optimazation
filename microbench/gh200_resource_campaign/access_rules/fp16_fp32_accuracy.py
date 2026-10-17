"""FP16 输入、连续 K16 FP32 点积的逐块前向误差界。"""
import numpy as np

RULE = 'fp16-fp32-k16-bound-v1'
ATOL = 2.0**-20
UNIT_ROUNDOFF = 2.0**-23
DOT_K = 16
GAMMA = DOT_K*UNIT_ROUNDOFF/(1-DOT_K*UNIT_ROUNDOFF)


def random_values(rows, columns, seed, is_a):
    value = np.uint32(seed) ^ np.uint32(0xa511e9b3 if is_a else 0x63d83595)
    value = value ^ (rows*np.uint32(0x9e3779b9)) ^ (columns*np.uint32(0x85ebca6b))
    value = (value ^ (value >> 16))*np.uint32(0x7feb352d)
    value = (value ^ (value >> 15))*np.uint32(0x846ca68b)
    value = value ^ (value >> 16)
    return ((value >> 8).astype(np.float32)*np.float32(2**-23)-np.float32(1)).astype(np.float16).astype(np.float64)


def dot_reference_bound(a, b):
    """每行一个点积；K16 内次序任意，允许定向舍入及FP16输入flush-to-zero。"""
    products = a*b
    reference = np.zeros(products.shape[0], dtype=np.float64)
    bound = np.zeros_like(reference)
    for begin in range(0, products.shape[1], DOT_K):
        end = begin+DOT_K
        part = products[:, begin:end]
        absolute = np.abs(part)
        small_a = (np.abs(a[:, begin:end]) < 2**-14) & (a[:, begin:end] != 0)
        small_b = (np.abs(b[:, begin:end]) < 2**-14) & (b[:, begin:end] != 0)
        flush = np.where(small_a | small_b, absolute, 0).sum(axis=1)
        bound = (1+GAMMA)*bound+GAMMA*(np.abs(reference)+absolute.sum(axis=1))+(1+GAMMA)*flush
        reference += part.sum(axis=1)
    return reference, ATOL+bound


def random_references(indices, m, n, k, seed):
    positions = np.asarray(indices, dtype=np.int64)
    rows = (positions//n).astype(np.uint32)
    columns = (positions%n).astype(np.uint32)
    inner = np.arange(k, dtype=np.uint32)[None, :]
    expected, bounds = [], []
    for begin in range(0, len(indices), 32):
        a = random_values(rows[begin:begin+32, None], inner, seed, True)
        b = random_values(inner, columns[begin:begin+32, None], seed, False)
        reference, bound = dot_reference_bound(a,b)
        expected.extend(reference.tolist())
        bounds.extend(bound.tolist())
    return expected, bounds
