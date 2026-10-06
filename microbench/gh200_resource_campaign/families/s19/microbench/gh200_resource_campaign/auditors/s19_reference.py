#!/usr/bin/env python3
"""Exact, CPU-only reference for the finite S19 short-validation contract.

All products and accumulations use integer numerators with denominator 256.
The maximum absolute partial numerator is 64 * 32 * 49 = 100352 < 2**24,
so every input, product and partial sum is exactly representable in FP32.
This reference proves expected values, never GPU execution or performance.
"""

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import struct


MODES = ("compute", "transport", "serial", "overlap", "output")
STAGES = (1, 2, 4)
K_TILES = (1, 2, 4, 8, 16, 32, 64)
PROFILES = ("periodic", "tagged")
SHORT_ITERATIONS = (1, 3)
POISON = 0x7FC0ABCD
UINT32_MAX = (1 << 32) - 1
CASE_FIELDS = {"mode", "stages", "k_tiles", "iterations", "seed", "profile"}
PAYLOAD_FIELDS = (
    "input_bits", "output_bits", "digest", "input_trace_bits", "c_trace_bits",
    "shared_bits",
)
ARTIFACT_FILES = {
    "input_bits": "input.u32le", "output_bits": "output.u32le",
    "digest": "digest.u32le", "shared_bits": "slots.u32le",
    "input_trace_bits": "trace_input.u32le", "c_trace_bits": "trace_c.u32le",
}


@dataclass(frozen=True)
class Case:
    mode: str
    stages: int
    k_tiles: int
    iterations: int = 1
    seed: int = 1
    profile: str = "periodic"

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in MODES:
            raise ValueError("mode must be compute/transport/serial/overlap/output")
        if type(self.stages) is not int or self.stages not in STAGES:
            raise ValueError("stages must be one of 1, 2, 4")
        if type(self.k_tiles) is not int or self.k_tiles not in K_TILES:
            raise ValueError("k_tiles must be one of 1, 2, 4, 8, 16, 32, 64")
        if type(self.iterations) is not int or self.iterations not in SHORT_ITERATIONS:
            raise ValueError("short iterations must be 1 or 3")
        if type(self.seed) is not int or not 0 <= self.seed <= UINT32_MAX:
            raise ValueError("seed must be a uint32 integer")
        if type(self.profile) is not str or self.profile not in PROFILES:
            raise ValueError("profile must be periodic or tagged")
        if self.mode == "compute" and self.profile != "periodic":
            raise ValueError("compute-only requires the periodic profile")

    @property
    def case_id(self):
        return (f"s19-{self.mode}-s{self.stages}-k{self.k_tiles}-"
                f"{self.profile}-i{self.iterations}-seed{self.seed}")

    @property
    def holdout(self):
        return self.k_tiles == 64


def parse_case(value):
    """Require exactly the six typed case fields; do not coerce JSON values."""
    if type(value) is not dict or set(value) != CASE_FIELDS:
        raise ValueError("case must contain exactly: " + ", ".join(sorted(CASE_FIELDS)))
    return Case(**value)


def case_matrix(include_holdout=True, profiles=PROFILES,
                iterations=SHORT_ITERATIONS, seeds=(1,)):
    """Return the finite short matrix, with holdout identified on each Case."""
    if type(include_holdout) is not bool:
        raise ValueError("include_holdout must be a bool")
    if not profiles or any(type(p) is not str or p not in PROFILES for p in profiles):
        raise ValueError("invalid or empty profiles")
    if not iterations or any(type(i) is not int or i not in SHORT_ITERATIONS for i in iterations):
        raise ValueError("invalid or empty short iterations")
    if not seeds or any(type(s) is not int or not 0 <= s <= UINT32_MAX for s in seeds):
        raise ValueError("invalid or empty seeds")
    if any(len(set(values)) != len(values) for values in (profiles, iterations, seeds)):
        raise ValueError("matrix selectors must not contain duplicates")
    cases = []
    for mode in MODES:
        for stages in STAGES:
            for k_tiles in K_TILES:
                if k_tiles == 64 and not include_holdout:
                    continue
                for profile in profiles:
                    if mode == "compute" and profile == "tagged":
                        continue
                    for count in iterations:
                        for seed in seeds:
                            cases.append(Case(mode, stages, k_tiles, count, seed, profile))
    return cases


def float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


@lru_cache(maxsize=512)
def tile_numerators(seed, tag):
    """A and B in row-major order, represented as exact integers over 16."""
    a = tuple((m * 3 + k * 5 + seed % 17 + tag * 7) % 15 - 7
              for m in range(32) for k in range(32))
    b = tuple((k * 7 + n * 3 + seed % 19 + tag * 11) % 15 - 7
              for k in range(32) for n in range(32))
    return a, b


@lru_cache(maxsize=512)
def tile_input_bits(seed, tag):
    a, b = tile_numerators(seed, tag)
    return tuple(float_bits(value / 16) for value in a + b)


@lru_cache(maxsize=512)
def tile_product_numerators(seed, tag):
    """Direct integer matrix multiplication; independent of CUDA thread layout."""
    a, b = tile_numerators(seed, tag)
    product = []
    for m in range(32):
        for n in range(32):
            value = 0
            for k in range(32):
                value += a[m * 32 + k] * b[k * 32 + n]
            product.append(value)
    return tuple(product)


@lru_cache(maxsize=64)
def prefix_c_bits(seed, profile, k_tiles):
    accumulator = [0] * 1024
    trace = []
    for tile in range(k_tiles):
        tag = 0 if profile == "periodic" else tile + 1
        product = tile_product_numerators(seed, tag)
        for index in range(1024):
            accumulator[index] += product[index]
            trace.append(float_bits(accumulator[index] / 256))
    return tuple(trace), tuple(accumulator)


def expected(case):
    """Return complete little-endian uint32 words in the agreed flat layouts.

    input_bits: [Ktiles, 2048]; output_bits: [32, 32]; digest: [128];
    input_trace_bits: [iterations, Ktiles, 2048];
    c_trace_bits: [iterations, Ktiles, 32, 32]; shared_bits: [4, 2048].
    The traces describe the consumed tile and C after its computation. C resets
    each sequence; the transport digest accumulates over all sequences.
    """
    if not isinstance(case, Case):
        raise TypeError("expected requires a validated Case")
    tiles = []
    for tile in range(case.k_tiles):
        tag = 0 if case.profile == "periodic" else tile + 1
        tiles.append(tile_input_bits(case.seed, tag))
    inputs = [value for tile in tiles for value in tile]
    digest = [0] * 128
    if case.mode == "transport":
        output = [0] * 1024
        c_trace = [0] * (case.k_tiles * 1024)
        for tid in range(128):
            total = 0
            for tile in tiles:
                for q in range(4):
                    for j in range(4):
                        index = ((tid + 1) % 128) * 4 + q * 512 + j
                        total += tile[index]
            digest[tid] = (total * case.iterations) & UINT32_MAX
    else:
        c_trace, accumulator = prefix_c_bits(case.seed, case.profile, case.k_tiles)
        if case.mode == "output":
            # 0.5 * (integer / 256) + (n - 16) / 16.
            output = [float_bits((value + (index % 32 - 16) * 32) / 512)
                      for index, value in enumerate(accumulator)]
        else:
            output = [float_bits(value / 256) for value in accumulator]
    shared = [POISON] * 8192
    if case.mode == "compute":
        for slot in range(min(case.stages, case.k_tiles)):
            shared[slot * 2048:(slot + 1) * 2048] = tiles[slot]
    else:
        # Solve the final tile occupying each ring slot without simulating the
        # probe's prefetch/consume schedule.
        for slot in range(min(case.stages, case.k_tiles)):
            last_tile = slot + ((case.k_tiles - 1 - slot) // case.stages) * case.stages
            shared[slot * 2048:(slot + 1) * 2048] = tiles[last_tile]
    return {
        "input_bits": inputs,
        "output_bits": output,
        "digest": digest,
        "input_trace_bits": inputs * case.iterations,
        "c_trace_bits": list(c_trace) * case.iterations,
        "shared_bits": shared,
    }


def verify_payload(case, payload):
    """Reject missing, surplus, truncated, mistyped or unequal full payloads."""
    if type(payload) is not dict or set(payload) != set(PAYLOAD_FIELDS):
        raise ValueError("payload must contain exactly: " + ", ".join(PAYLOAD_FIELDS))
    reference = expected(case)
    sizes = {}
    for field in PAYLOAD_FIELDS:
        actual = payload[field]
        wanted = reference[field]
        if type(actual) is not list or len(actual) != len(wanted):
            raise ValueError(f"{field}: expected list of {len(wanted)} uint32 words")
        for index, (value, target) in enumerate(zip(actual, wanted)):
            if type(value) is not int or not 0 <= value <= UINT32_MAX:
                raise ValueError(f"{field}[{index}]: value must be a uint32 integer")
            if value != target:
                raise ValueError(f"{field}[{index}]: got 0x{value:08x}, expected 0x{target:08x}")
        sizes[field] = len(wanted)
    return {"case_id": case.case_id, "holdout": case.holdout,
            "matched_words": sizes, "numerical_match": True,
            "evidence_scope": "complete_payload_values_only"}


def verify_artifacts(case, directory):
    """Read the probe's six actual raw files, retaining exact byte identities.

    This verifies values and dimensions only. Allocation provenance, executable
    identity, completion events and runtime status remain the runner's checks.
    """
    directory = Path(directory)
    lengths = {
        "input_bits": case.k_tiles * 2048, "output_bits": 1024, "digest": 128,
        "shared_bits": 8192,
        "input_trace_bits": case.iterations * case.k_tiles * 2048,
        "c_trace_bits": case.iterations * case.k_tiles * 1024,
    }
    payload = {}
    artifacts = {}
    for field, filename in ARTIFACT_FILES.items():
        raw = (directory / filename).read_bytes()
        count = lengths[field]
        if len(raw) != count * 4:
            raise ValueError(f"{filename}: got {len(raw)} bytes, expected {count * 4}")
        payload[field] = list(struct.unpack(f"<{count}I", raw))
        artifacts[filename] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    result = verify_payload(case, payload)
    result["artifacts"] = artifacts
    return result


def read_json(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def reject_constant(text):
        raise ValueError("non-finite JSON value: " + text)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys,
                      parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--exclude-holdout", action="store_true")
    expected_parser = subparsers.add_parser("expected")
    expected_parser.add_argument("case_json")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("case_json")
    verify_parser.add_argument("payload_json")
    artifact_parser = subparsers.add_parser("verify-artifacts")
    artifact_parser.add_argument("case_json")
    artifact_parser.add_argument("directory")
    args = parser.parse_args()
    if args.command == "matrix":
        result = [asdict(case) for case in case_matrix(not args.exclude_holdout)]
    elif args.command == "expected":
        result = expected(parse_case(read_json(args.case_json)))
    elif args.command == "verify":
        result = verify_payload(parse_case(read_json(args.case_json)), read_json(args.payload_json))
    else:
        result = verify_artifacts(parse_case(read_json(args.case_json)), args.directory)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
