#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))
import mma_config_runner

if __name__ == "__main__":
    mma_config_runner.main("00_validation")
