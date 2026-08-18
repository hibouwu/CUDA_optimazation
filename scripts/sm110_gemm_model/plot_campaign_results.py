#!/usr/bin/env python3
"""Generate deterministic SVG plots from one SM110 campaign summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign_plots import PlotError, generate_campaign_plots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="campaign summary.json or closure_analysis.json")
    parser.add_argument("--output-dir", type=Path,
                        help="default: <input-directory>/plots")
    args = parser.parse_args()
    try:
        manifest = generate_campaign_plots(args.input, args.output_dir)
    except (OSError, json.JSONDecodeError, PlotError, KeyError) as error:
        print(json.dumps({"pass": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps({"pass": True, **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
