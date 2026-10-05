"""CPU coverage and rejection tests; never launches or modifies old experiments."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / "contracts/build_legacy_mapping.py"
SPEC = importlib.util.spec_from_file_location("build_legacy_mapping", PATH)
mapping = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapping)


class LegacyMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = mapping.build()
        cls.archive = mapping.ROOT / mapping.ARCHIVES

    def test_checked_in_mapping_is_reproducible(self):
        path = mapping.ROOT / mapping.OUTPUT
        self.assertEqual(path.read_text(), json.dumps(self.data, ensure_ascii=False, indent=2) + "\n")

    def test_exact_legacy_matrix_and_raw_coverage(self):
        expected = {"initial": (32, {128, 512, 2048}, {"single_cta"}, 7),
                    "sustained": (42, None, {"single_cta", "full_gpu"}, 5),
                    "audit": (7, {8192, 32768, 65536}, {"single_cta", "full_gpu"}, 12)}
        self.assertEqual(self.data["counts"]["basic_instruction_launch_combinations"], 62)
        self.assertEqual(len(self.data["configurations"]), 81)
        for batch, (count, lengths, scopes, repeats) in expected.items():
            rows = [m for m in self.data["mappings"] if m["legacy_configuration_id"].startswith(f"20260930-{batch}/")]
            names = {r["legacy_configuration_id"] for r in rows}
            self.assertEqual(len(names), count)
            for name in names:
                formal = [r for r in rows if r["legacy_configuration_id"] == name and r["key"]["phase"] == "measure"]
                self.assertEqual({r["key"]["scope"] for r in formal}, scopes)
                for scope in scopes:
                    scoped = [r for r in formal if r["key"]["scope"] == scope]
                    if lengths:
                        self.assertEqual({r["key"]["iteration_policy"]["iterations"] for r in scoped}, lengths)
                    else:
                        self.assertEqual(len(scoped), 1)
                        self.assertEqual(scoped[0]["key"]["iteration_policy"]["kind"], "dynamic_cuda_event_pilot")
                self.assertTrue(all(len(r["legacy_raw_lines"]) == repeats for r in formal))
            listed = [n for row in rows for n in row["legacy_raw_lines"]]
            auxiliary = self.data["batches"][batch]["auxiliary_records"]
            listed += [n for row in auxiliary for n in row["raw_lines"]]
            raw = (self.archive / f"20260930-{batch}" / "raw.jsonl").read_text().splitlines()
            self.assertEqual(sorted(listed), list(range(2, len(raw) + 1)))
            self.assertEqual(len(listed), len(set(listed)))

    def test_audit_empty_controls_are_single_cta_only(self):
        rows = [r for r in self.data["mappings"] if r["key"]["phase"] == "empty_control"]
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(r["key"]["scope"] == "single_cta" and
                            r["key"]["iteration_policy"] == {"kind": "fixed", "iterations": 0}
                            and len(r["legacy_raw_lines"]) == 12 for r in rows))
        auxiliary = self.data["batches"]["audit"]["auxiliary_records"]
        self.assertEqual(sum(len(r["raw_lines"]) for r in auxiliary if r["phase"] == "warmup"), 78)
        self.assertEqual(sum(len(r["raw_lines"]) for r in auxiliary if r["phase"] == "transition_warmup"), 504)

    def test_same_basic_instruction_keeps_protocol_differences(self):
        rows = [r for r in self.data["mappings"] if r["key"]["ptx"] ==
                "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16" and
                r["key"]["threads"] == 128 and r["key"]["chains"] == 2 and
                r["key"]["batch"] == 16 and r["key"]["wait"] == 0 and
                r["key"]["scope"] == "single_cta" and r["key"]["phase"] == "measure"]
        self.assertEqual(len(rows), 7)  # 3 initial + 1 calibrated + 3 audit.
        self.assertEqual(len({r["basic_combination_id"] for r in rows}), 1)
        self.assertEqual(len({r["mapping_id"] for r in rows}), 7)
        initial = next(r for r in rows if "initial/" in r["legacy_configuration_id"])
        sustained = next(r for r in rows if "sustained/" in r["legacy_configuration_id"])
        audit = next(r for r in rows if "audit/" in r["legacy_configuration_id"])
        self.assertIsNone(initial["key"]["drain"]["post_loop_wgmma_wait"])
        self.assertEqual(sustained["key"]["drain"]["post_loop_wgmma_wait"], 0)
        self.assertIsNone(sustained["key"]["timer"]["additional_cycle_basis"])
        self.assertIsNotNone(audit["key"]["timer"]["additional_cycle_basis"])

    def test_source_metadata_mismatch_is_rejected(self):
        directory = self.archive / "20260930-initial"
        cases = json.loads((directory / "summary.json").read_text())["cases"]
        source = (directory / "probe.cu").read_text()
        for field, value in (("threads", 64), ("chains", 3), ("batch", 8),
                             ("ptx", "fma.rz.f32"), ("work_per_collective", 1)):
            case = dict(cases[0], **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                mapping.extract_contract(case, source, "initial")
        wgmma = next(c for c in cases if c["kind"] == "wgmma_f16")
        changed = source.replace("wgmma.wait_group.sync.aligned 0;", "wgmma.wait_group.sync.aligned 3;")
        with self.assertRaisesRegex(ValueError, "wait/drain mismatch"):
            mapping.extract_contract(wgmma, changed, "initial")

    def test_missing_duplicate_and_incorrect_measurements_are_rejected(self):
        directory = self.archive / "20260930-initial"
        case = json.loads((directory / "summary.json").read_text())["cases"][0]
        all_rows = [json.loads(s) for s in (directory / "raw.jsonl").read_text().splitlines()]
        rows = [(i + 1, r) for i, r in enumerate(all_rows) if r.get("case") == case["name"] and r["iterations"] == 128]
        mapping.validate_records(rows, case, 7, 128, "single_cta", "measure")
        corruptions = [rows[:-1], rows[:-1] + rows[:1]]
        for field, value in (("work_flop", 1), ("iterations", 512), ("scope", "full_gpu"),
                             ("phase", "empty_control"), ("max_abs_error", 1)):
            changed = deepcopy(rows)
            changed[0][1][field] = value
            corruptions.append(changed)
        for changed in corruptions:
            with self.assertRaises(ValueError):
                mapping.validate_records(changed, case, 7, 128, "single_cta", "measure")

    def test_input_and_instruction_operands_affect_identity(self):
        case = json.loads((self.archive / "20260930-initial/summary.json").read_text())["cases"][0]
        source = (self.archive / "20260930-initial/probe.cu").read_text()
        original, _ = mapping.extract_contract(case, source, "initial")
        changed, _ = mapping.extract_contract(case, source.replace("(float)0.25", "(float)0.125"), "initial")
        self.assertNotEqual(original["input"]["arithmetic_inline_asm_bindings"],
                            changed["input"]["arithmetic_inline_asm_bindings"])
        self.assertNotEqual(mapping.key_id(original), mapping.key_id(changed))

    def test_relocated_replay_and_corrupt_archive_rejection(self):
        files = {mapping.PROTOCOL}
        for batch in self.data["batches"].values():
            files.update(Path(p) for p in batch["source_files"])
        before = {str(p): mapping.sha256((mapping.ROOT / p).read_bytes()) for p in files}
        with tempfile.TemporaryDirectory(prefix="gh200-mapping-") as temporary:
            target = Path(temporary)
            for path in files:
                (target / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(mapping.ROOT / path, target / path)
            self.assertEqual(mapping.build(target), self.data)
            raw = target / mapping.ARCHIVES / "20260930-initial/raw.jsonl"
            raw.write_text(raw.read_text().replace('"iterations":128', '"iterations":129', 1))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                mapping.build(target)
        after = {str(p): mapping.sha256((mapping.ROOT / p).read_bytes()) for p in files}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
