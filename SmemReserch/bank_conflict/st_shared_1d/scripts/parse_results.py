#!/usr/bin/env python3
import csv
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / "results" / "basic_results.csv"
if not path.exists():
    raise SystemExit(f"Missing {path}; run scripts/run_basic.sh first.")
with path.open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit(f"{path} is empty")
columns = ["case", "stride", "avg_ms", "min_ms", "effective_GBps"]
widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in columns}
print("  ".join(c.ljust(widths[c]) for c in columns))
for row in rows:
    print("  ".join(row[c].ljust(widths[c]) for c in columns))
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit(0)
labels = [f"{r['case']}:{r['stride']}" if r["stride"] != "0" else r["case"] for r in rows]
plt.figure(figsize=(max(8, len(rows) * 0.8), 5))
plt.bar(labels, [float(r["avg_ms"]) for r in rows])
plt.xticks(rotation=30, ha="right")
plt.ylabel("avg_ms")
plt.tight_layout()
plt.savefig(root / "results" / "avg_ms.png", dpi=150)

