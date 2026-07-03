#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASIC_CSV = ROOT / "results" / "basic_results.csv"
PATCHED_CSV = ROOT / "results" / "sass_patched" / "results.csv"
ASSETS = ROOT / "assets"
OUTPUT = ASSETS / "register_experiment_results.png"

BASIC_ORDER = [
    "R0_imad_chain",
    "R1_imad_independent_x4",
    "R2_reuse_hot_x4",
    "R3_bank_dense_x4",
    "R4_bank_sparse_x4",
]
PATCHED_ORDER = [
    "S0_mod4_control_noreuse",
    "S1_mod4_conflict_noreuse",
    "S2_mod8_control_noreuse",
    "S3_mod8_conflict_noreuse",
]


def load_ordered(path, order):
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["case"]: row for row in csv.DictReader(stream)}
    missing = [name for name in order if name not in rows]
    if missing:
        raise SystemExit(f"{path} is missing cases: {', '.join(missing)}")
    return [rows[name] for name in order]


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required to plot experiment results")

    basic = load_ordered(BASIC_CSV, BASIC_ORDER)
    patched = load_ordered(PATCHED_CSV, PATCHED_ORDER)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#28332F",
            "axes.labelcolor": "#28332F",
            "xtick.color": "#28332F",
            "ytick.color": "#28332F",
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.8),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    figure.patch.set_facecolor("#F5F1E8")
    for axis in axes:
        axis.set_facecolor("#FCFAF5")
        axis.grid(axis="y", color="#A8B0A8", alpha=0.28, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    basic_labels = ["Dependent\nchain", "Independent\nx4", "Reuse hot\nx4",
                    "Bank dense\nx4", "Bank sparse\nx4"]
    basic_values = [float(row["median_cycles_per_op"]) for row in basic]
    basic_colors = ["#C54A34", "#287A74", "#D39A2C", "#557A46", "#7B6957"]
    bars = axes[0].bar(
        basic_labels,
        basic_values,
        color=basic_colors,
        width=0.68,
        edgecolor="#FCFAF5",
        linewidth=1.5,
        zorder=3,
    )
    axes[0].set_ylim(0, 4.75)
    axes[0].set_ylabel("Median cycles per IMAD")
    axes[0].set_title(
        "PTX-generated SASS", loc="left", fontsize=14, pad=32
    )
    axes[0].text(
        0,
        1.01,
        "Dependency latency vs. independent issue throughput",
        transform=axes[0].transAxes,
        color="#60706A",
        fontsize=9.5,
    )
    for bar, value in zip(bars, basic_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            color="#28332F",
            fontsize=9,
            fontweight="bold",
        )

    patched_values = [
        float(row["median_cycles_per_op"]) for row in patched
    ]
    hypotheses = ["R % 4", "R % 8"]
    control = [patched_values[0], patched_values[2]]
    conflict = [patched_values[1], patched_values[3]]
    x_positions = [0, 1]
    for x_value, control_value, conflict_value in zip(
        x_positions, control, conflict
    ):
        axes[1].plot(
            [x_value - 0.09, x_value + 0.09],
            [control_value, conflict_value],
            color="#7B8C85",
            linewidth=2,
            zorder=2,
        )
    axes[1].scatter(
        [value - 0.09 for value in x_positions],
        control,
        s=95,
        color="#287A74",
        edgecolor="#FCFAF5",
        linewidth=1.4,
        label="Control",
        zorder=4,
    )
    axes[1].scatter(
        [value + 0.09 for value in x_positions],
        conflict,
        s=95,
        color="#C54A34",
        marker="D",
        edgecolor="#FCFAF5",
        linewidth=1.4,
        label="Same-residue pair",
        zorder=4,
    )
    axes[1].set_xticks(x_positions, hypotheses)
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylim(2.04, 2.13)
    axes[1].set_ylabel("Median cycles per IMAD")
    axes[1].set_title(
        "Patched physical registers", loc="left", fontsize=14, pad=32
    )
    axes[1].text(
        0,
        1.01,
        "128 IMADs, fixed R<n>, all .reuse disabled",
        transform=axes[1].transAxes,
        color="#60706A",
        fontsize=9.5,
    )
    for x_value, control_value, conflict_value in zip(
        x_positions, control, conflict
    ):
        delta = conflict_value - control_value
        axes[1].text(
            x_value,
            max(control_value, conflict_value) + 0.012,
            f"{control_value:.6f}\nΔ = {delta:+.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#28332F",
            fontweight="bold",
        )
    axes[1].legend(
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=9,
    )

    figure.suptitle(
        "NVIDIA Thor Register-File Microbenchmark",
        x=0.055,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color="#1F2925",
    )
    figure.text(
        0.055,
        0.012,
        "CUDA 13 · sm_110 · 100,000 iterations · 20 repeats · 0 local bytes",
        color="#60706A",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.05, 0.99, 0.94), w_pad=3.0)
    ASSETS.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
