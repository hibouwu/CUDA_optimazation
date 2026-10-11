#!/usr/bin/env python3
"""R13 cfg_a pitch x requested-SM contrast; reuse the checked shared binaries.

CPU: list | cpu-check | prepare --harness-run CHECKED_RUN --output NEW_RUN
GPU (allocated node only): setup | sample --output RUN [--procs 1|10]
The inherited runner uses V08_GPU == CUDA_VISIBLE_DEVICES == the allocated UUID.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import run_r18
import run_v08
import v06_run as common
from v08_model import scheduled_work

ROOT = Path(__file__).resolve().parent
HARNESS_SHA = "6338653b5a4127a7cb947b6355c38642310f3dcc"


def cases():
    rows = []
    for sms in (32, 64, 96, 132):
        for pitch in ("aligned", "a16", "b16"):
            row = run_v08.row(f"cfg_a_sm{sms}_{pitch}", "cfg_a", "ctrl", 2304, 3072, 4096)
            row.update(group=f"sm{sms}", pitch=pitch, sm_count=sms, input_mode="dyadic", seed=17,
                       lda=4104 if pitch == "a16" else 4096,
                       ldb=3080 if pitch == "b16" else 3072)
            rows.append(row)
    return rows


def cpu_check():
    expected = {32: {13: 16, 14: 16}, 64: {6: 16, 7: 48},
                96: {4: 48, 5: 48}, 132: {3: 96, 4: 36}}
    rows = cases()
    assert len(rows) == len({r["id"] for r in rows}) == 12
    for row in rows:
        sms = row["sm_count"]
        args = run_r18.case_args(row)
        assert args[args.index("--sm-count") + 1] == str(sms)
        assert args[args.index("--input-mode") + 1] == "dyadic"
        work = scheduled_work("cfg_a", row["m"], row["n"], [sms, 1, 1], 1)
        assert Counter(map(len, work)) == expected[sms]
        assert len({xy for cta in work for xy in cta}) == 432
        assert max(map(len, work)) < 64  # Current trace capacity.
    overlap = {}
    for j in (0, 1, 2, 3):
        sets = []
        for sms in (96, 132):
            work = scheduled_work("cfg_a", 2304, 3072, [sms, 1, 1], 1)
            sets.append({cta[j] for cta in work if len(cta) == 4})
        overlap[j] = len(sets[0] & sets[1])
    assert overlap == {0: 0, 1: 24, 2: 24, 3: 0}
    result = dict(cases=12, output_tiles=432, expected_grid_note="prediction only; read actual setup.grid at runtime",
                  expected_total_tiles_per_CTA=expected, same_coordinate_T4_96_132=overlap)
    print(json.dumps(result, indent=2))
    return result


def prepare(root, harness):
    common.verify(harness)
    origin = json.loads((harness / "run_config.json").read_text())
    if origin["source_commit"] != HARNESS_SHA:
        raise ValueError("this batch expects the checked 6338653 harness")
    binaries = ("cfg_a_plain", "cfg_a_stamped")
    sass_hashes = json.loads((harness / "build/sass_hashes.json").read_text())
    for name in binaries:
        if common.sha(harness / "build" / (name + ".sass")) != sass_hashes[name + ".sass"]:
            raise ValueError("checked SASS changed: " + name)
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(harness / "source", root / "source", ignore=shutil.ignore_patterns("__pycache__"))
    (root / "build").mkdir()
    for name in binaries:
        for suffix in ("", ".sass", ".log"):
            shutil.copy2(harness / "build" / (name + suffix), root / "build" / (name + suffix))
    for name in ("run_r13_sm.py", "analyze_r13_sm.py"):
        shutil.copy2(ROOT / name, root / "source" / name)
    commands = json.loads((harness / "build/commands.json").read_text())
    hashes = json.loads((harness / "build/binary_hashes.json").read_text())
    resources = json.loads((harness / "build/resources.json").read_text())
    common.write_json(root / "build/commands.json", {k: commands[k] for k in binaries})
    common.write_json(root / "build/binary_hashes.json", {"build/"+k: hashes["build/"+k] for k in binaries})
    common.write_json(root / "build/sass_hashes.json", {k+".sass": sass_hashes[k+".sass"] for k in binaries})
    common.write_json(root / "build/resources.json", dict(facts={k: resources["facts"][k] for k in binaries},
                      problems=[k for k in resources["problems"] if k in binaries]))
    shutil.copy2(harness / "build/nvcc-version.txt", root / "build/nvcc-version.txt")
    common.write_json(root / "build/origin.json", dict(run=str(harness), source_commit=HARNESS_SHA,
                      original_binary_manifest_sha256=common.sha(harness / "build/binary_hashes.json"),
                      original_source_manifest_sha256=common.sha(harness / "source_hashes.json"),
                      analysis_sha256=common.sha(ROOT / "analyze_r13_sm.py")))
    common.write_json(root / "cases.json", cases())
    common.write_json(root / "run_config.json", dict(family="r13_sm", harness_source_commit=HARNESS_SHA,
                      cases_sha256=common.sha(root / "cases.json"), input_mode="dyadic", seed=17,
                      scope="same-grid pitch contrasts; cross-grid matched j/T/coordinate subset, not local/shared attribution"))
    common.write_json(root / "source_hashes.json", {str(p.relative_to(root)): common.sha(p)
                      for p in (root / "source").rglob("*") if p.is_file()})
    common.write_json(root / "cpu_geometry.json", cpu_check())
    (root / "run.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${SLURM_JOB_ID:?run inside the assigned Slurm allocation}"
: "${V08_GPU:?set to the allocated GPU UUID}"
: "${CUDA_VISIBLE_DEVICES:?must equal V08_GPU}"
if [[ ! -f static_setup.json ]]; then
  python3 source/run_r13_sm.py setup --output .
fi
python3 source/run_r13_sm.py sample --output . --procs "${R13_PROCS:-10}"
python3 source/analyze_r13_sm.py --run . --output "reanalysis/sm-${R13_PROCS:-10}procs"
''')
    (root / "run.sh").chmod(0o755)
    common.verify(root)
    print("Prepared 12 cases with checked cfg_a plain/stamped binaries:", root)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "list":
        print(json.dumps(cases(), indent=2))
    elif step == "cpu-check":
        cpu_check()
    elif step == "prepare":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("step", choices=["prepare"])
        parser.add_argument("--harness-run", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        prepare(args.output.resolve(), args.harness_run.resolve())
    elif step in ("setup", "sample"):
        # Reuse the shared run loop/locking/checks; select the one allocated UUID
        # through the existing V08 identity function (nvidia-smi may list 4 GPUs).
        common.identity = run_v08.identity
        run_r18.main("r13_sm")
    else:
        raise SystemExit(__doc__)
