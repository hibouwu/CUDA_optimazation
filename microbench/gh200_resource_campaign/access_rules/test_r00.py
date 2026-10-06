import unittest
import tempfile
import json
import gzip
import hashlib
import struct
from pathlib import Path
from analyze import recompute, analyze
from run_r00 import matrix


class R00Checks(unittest.TestCase):
    def test_matrix_and_heldout_shapes(self):
        cases = matrix()
        self.assertEqual(len(cases), 30)
        self.assertEqual(len({c["id"] for c in cases}), 30)
        self.assertEqual(sum(c["kind"] == "gemm" for c in cases), 15)
        for c in cases:
            if c.get("backend") == "cutlass":
                self.assertNotIn((c["m"], c["n"], c["k"]), [(2048, 2048, 2048), (2048, 2048, 8192)])

    def test_event_units(self):
        work, time, rate, unit = recompute(
            dict(kind="gemm", m=3, n=5, k=7, work_flop=210, elapsed_ms=0.1)
        )
        self.assertEqual(work, 210)
        self.assertAlmostEqual(rate, 0.0021)
        self.assertEqual(unit, "GFLOP/s/GPU")

    def test_work_and_clock_domain_rejections(self):
        record = dict(
            kind="wgmma",
            n=256,
            k=16,
            chains=1,
            per_chain_batch=16,
            groups=1,
            iterations=128,
            blocks=1,
            scope="one_cta",
            unit="clock64_cycle/CTA",
            stamps=[[100, 1100, 200, 4200, 0]],
            elapsed=4000,
            work_flop=1073741824,
        )
        self.assertEqual(recompute(record)[0], 1073741824)
        for field, value in [
            ("elapsed", 3999),
            ("work_flop", 1073741825),
            ("unit", "globaltimer_ns/GPU"),
        ]:
            with self.assertRaises(ValueError):
                recompute(dict(record, **{field: value}))

    def test_empty_or_invalid_window(self):
        for elapsed in [0, -1, float("nan")]:
            with self.assertRaises(ValueError):
                recompute(dict(kind="gemm", m=3, n=5, k=7, work_flop=210, elapsed_ms=elapsed))

    def test_tolerated_output_does_not_replace_rate_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "samples/synthetic/trial-00"
            sample.mkdir(parents=True)
            env = dict(
                mode="smoke",
                hostname="synthetic",
                gpu_uuid="synthetic",
                slurm_job_id="synthetic",
                cutlass_tag="synthetic",
            )
            (root / "environment.json").write_text(json.dumps(env))
            (root / "cases.json").write_text(json.dumps([dict(id="synthetic", kind="wgmma")]))
            output = struct.pack("<f", 1e-5) + bytes((16384 - 1) * 4)
            with gzip.open(sample / "output.f32.gz", "wb") as stream:
                stream.write(output)
            r = dict(
                kind="wgmma",
                status="measured",
                trial_role="smoke",
                n=256,
                k=32,
                chains=1,
                per_chain_batch=16,
                groups=1,
                iterations=1,
                blocks=1,
                threads=128,
                scope="one_cta",
                unit="clock64_cycle/CTA",
                stamps=[[100, 1100, 200, 1200, 0]],
                elapsed=1000,
                work_flop=16777216,
                output_elements=16384,
                output_sha256=hashlib.sha256(output).hexdigest(),
                input_profile="uniform_paired_sign_bounded",
                local_bytes_per_thread=0,
            )
            (sample / "result.json").write_text(json.dumps(r))
            rows = analyze(root)
            self.assertEqual(rows[0]["count"], 1)
            self.assertEqual(rows[0]["median_rate"], 16777.216)


if __name__ == "__main__":
    unittest.main()
