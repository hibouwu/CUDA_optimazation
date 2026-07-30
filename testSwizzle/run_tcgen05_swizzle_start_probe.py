#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "probe.cu"
BIN = ROOT / "probe.out"

def run(cmd):
    cmd = [str(arg) for arg in cmd]
    print("+", " ".join(cmd))
    try:
        return subprocess.check_output(
            cmd,
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        if exc.output:
            print(exc.output, end="" if exc.output.endswith("\n") else "\n")
        raise

def build():
    run([
        "nvcc",
        "-O3",
        "-std=c++17",
        "-gencode", "arch=compute_110a,code=sm_110a",
        SRC,
        "-lcuda",
        "-o", BIN
    ])

def main():
    build()
    out = run([BIN])
    print(out)

if __name__ == "__main__":
    main()
