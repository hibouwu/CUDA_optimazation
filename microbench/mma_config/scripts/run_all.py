#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

import mma_config_runner


def parse_args():
    parser = argparse.ArgumentParser(
        description="按依赖顺序运行 tcgen05 mma_config 实验。"
    )
    parser.add_argument(
        "--stage",
        choices=["all"] + mma_config_runner.STAGES,
        default="all",
        help="运行全部 stage 或单个实验子目录。",
    )
    parser.add_argument("--case-id", help="在 --stage 内运行单个 case_id。")
    parser.add_argument("--quick", action="store_true", help="使用缩减版 smoke-test 矩阵。")
    parser.add_argument("--repeats", type=int, default=5, help="每个性能 case 的 repeat 次数。")
    parser.add_argument("--seed", type=int, default=20260720, help="性能 case 执行顺序的随机种子。")
    parser.add_argument(
        "--static",
        action="store_true",
        help="运行静态 single-case calibration runner，而不是 legacy stage runner。",
    )
    parser.add_argument(
        "--static-matrix",
        choices=["calibration", "collector", "ingress", "ldshared", "tmem", "all"],
        default="all",
        help="设置 --static 时要运行的静态矩阵。",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="运行后跳过重新生成 Docs/ExperimentReport.md。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.static:
        cmd = [
            sys.executable,
            str(ROOT / "02_latency_throughput" / "scripts" / "run_static_calibration.py"),
            "--matrix",
            args.static_matrix,
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
        ]
        if args.quick:
            cmd.append("--quick")
        if args.case_id:
            cmd.extend(["--case-id", args.case_id])
        subprocess.run(cmd, cwd=ROOT.parent, check=True)
        return

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
