#!/usr/bin/env python3
"""Versioned, read-only post-audit; preserves the original frozen strict outcome."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from auditors.offline_replay_v1 import replay


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir',type=Path)
    ap.add_argument('--policy',type=Path,default=Path(__file__).resolve().parent/'contracts/offline_replay_v1.json')
    args=ap.parse_args()
    receipt=replay(args.run_dir,args.policy)
    print(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
    return 0 if receipt['status']=='pass_under_bounded_float_replay_v1' else 2


if __name__=='__main__':raise SystemExit(main())
