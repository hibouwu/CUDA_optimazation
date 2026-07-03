#!/usr/bin/env python3
"""
Run Nsight Compute analysis on our bank-stride kernel
to capture register file access metrics
"""
import subprocess
import sys
from pathlib import Path

KERNEL = "./build/sass_register_bench"
OUTPUT_DIR = Path("ncu_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Metrics that might reveal bank information
METRICS = [
    "smsp__inst_executed",           # Instruction execution
    "smsp__inst_issued",              # Instruction issue
    "smsp__pipe_fu_core_active",     # FU core pipeline activity
    "smsp__inst_executed_op_fp32",   # FP32 ops
    "smsp__inst_executed_op_integer", # Integer ops
    "smsp__inst_executed_op_logic",  # Logic ops
]

# Try to collect bank-related metrics
for metric in METRICS:
    output_file = OUTPUT_DIR / f"profile_{metric}.ncu-rep"
    cmd = [
        "ncu",
        "--metrics", metric,
        "--csv",
        f"--output={output_file}",
        KERNEL
    ]
    print(f"Collecting {metric}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"  ✓ Success → {output_file}")
        else:
            print(f"  ✗ Error: {result.stderr[:100]}")
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout")
    except Exception as e:
        print(f"  ✗ Exception: {e}")

print("\nGenerating full report with multiple sections...")
output_file = OUTPUT_DIR / "profile_full.ncu-rep"
cmd = [
    "ncu",
    "--set", "full",
    f"--output={output_file}",
    KERNEL
]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if result.returncode == 0:
    print(f"✓ Full report saved to {output_file}")
else:
    print(f"Errors in full report: {result.stderr[:200]}")
