#!/usr/bin/env python3
"""Reproducible figures bound to an audited summary, with scopes kept separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from audit_campaign import audit
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gh200-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot(root):
    if (root/"REJECTED.md").exists():
        raise ValueError("run was rejected in post-audit")
    audit(root, require_complete=True)
    path=root/"summary.json";summary=json.loads(path.read_text())
    if summary["status"]!="timing_validated":
        raise ValueError("audited timing summary required")
    dest=root/"plots";dest.mkdir(exist_ok=True)
    created=[]
    for group in ("smem","global"):
        rows=[r for r in summary["cases"] if r["mode"].startswith(group)]
        fig,ax=plt.subplots(figsize=(10,5.5))
        values=[r["median"] for r in rows]
        errors=[[r["median"]-r["min"] for r in rows],[r["max"]-r["median"] for r in rows]]
        ax.barh(range(len(rows)),values,xerr=errors,color="#356C9B",capsize=3)
        ax.set_yticks(range(len(rows)),[r["id"] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("Requested B / clock64 cycle / CTA" if group=="smem" else "Requested payload GB/s / GPU")
        ax.set_title("GH200: SMEM access service" if group=="smem" else "GH200: global access paths (cache residency unproven)")
        ax.grid(axis="x",alpha=.2);ax.set_axisbelow(True)
        fig.tight_layout()
        for extension in ("svg", "png"):
            name=group+"-service."+extension
            fig.savefig(dest/name,dpi=160)
            created.append(name)
        plt.close(fig)
    shutil.copy2(Path(__file__),dest/"plot_results.py")
    manifest={"summary_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "plotter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "matplotlib_version":matplotlib.__version__,
              "figures":{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in created},
              "error_bars":"min/max of independent process trials, not confidence intervals"}
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (dest/"index.md").write_text("# GH200 资源路径图\n\n误差条为样本最小至最大值；单位与作用域分开，缓存驻留和物理 HBM 流量未证实。\n\n"+
                                "\n\n".join(f"![{name}]({name})" for name in created if name.endswith(".svg"))+"\n")
    print(dest)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run_dir",type=Path);plot(p.parse_args().run_dir)
