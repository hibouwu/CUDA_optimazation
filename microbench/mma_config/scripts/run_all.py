#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import mma_config_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tcgen05 mma_config experiments in dependency order."
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + mma_config_runner.STAGES,
        default="all",
        help="Run all stages or a single experiment subfolder.",
    )
    parser.add_argument("--case-id", help="Run a single case_id within --stage.")
    parser.add_argument("--quick", action="store_true", help="Use the reduced smoke-test matrix.")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per performance case.")
    parser.add_argument("--seed", type=int, default=20260720, help="Randomization seed for performance case order.")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip regenerating Docs/ExperimentReport.md after the run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.case_id and args.stage == "all":
        raise SystemExit("--case-id requires a single --stage.")

    stages = mma_config_runner.STAGES if args.stage == "all" else [args.stage]
    latest_rows = {}
    for stage in stages:
        rows = mma_config_runner.run_stage(
            stage,
            quick=args.quick,
            repeats=args.repeats,
            seed=args.seed,
            single_case=args.case_id,
        )
        latest_rows[stage] = rows

        if stage == "00_validation":
            ok, missing = mma_config_runner.validation_core_passed(rows)
            if not ok:
                missing_s = ", ".join(f"{dtype}/N{n}" for dtype, n in missing)
                raise SystemExit(
                    "00_validation core cases failed; stopping before performance stages. "
                    f"Missing valid core rows: {missing_s}"
                )

    if not args.no_report:
        report = mma_config_runner.generate_final_report()
        print(f"wrote {report}")


if __name__ == "__main__":
    main()
