#!/usr/bin/env python3
"""Deterministic, dependency-free SVG plots for SM110 campaign summaries.

The JSON summary remains the auditable source of truth.  These plots are
derived presentation artifacts: they never promote a measurement to a strict
upper bound and they never connect values with incompatible work units.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


COLORS = (
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2",
    "#4f46e5", "#be123c", "#65a30d", "#7c3aed", "#0f766e", "#b45309",
)


class PlotError(ValueError):
    pass


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    low: float | None = None
    high: float | None = None
    x_label: str | None = None


@dataclass(frozen=True)
class Series:
    label: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Panel:
    title: str
    y_label: str
    series: tuple[Series, ...]
    x_ticks: tuple[tuple[float, str], ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PlotError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise PlotError(f"{field} must be finite")
    return result


def _optional_finite(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _finite(value, field)


def _runtime_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("results")
    if not isinstance(rows, list):
        raise PlotError("campaign summary must contain a results array")
    return [row for row in rows if isinstance(row, dict) and row.get("status") == "ok"]


def _style() -> str:
    return """
text { font-family: Arial, sans-serif; fill: #172033; }
.figure-title { font-size: 22px; font-weight: 700; }
.panel-title { font-size: 14px; font-weight: 700; }
.axis-title { font-size: 12px; font-weight: 600; }
.tick { font-size: 10px; fill: #52606d; }
.legend { font-size: 10px; }
.frame { fill: #ffffff; stroke: #9aa6b2; stroke-width: 1; }
.grid { stroke: #d9e0e7; stroke-width: 1; }
.axis { stroke: #596675; stroke-width: 1; }
.whisker { stroke-width: 1.2; opacity: 0.72; }
""".strip()


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    scaled = value / base
    step = 1.0 if scaled <= 1 else 2.0 if scaled <= 2 else 5.0 if scaled <= 5 else 10.0
    return step * base


def _fmt(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _line_figure(path: Path, title: str, x_label: str,
                 panels: Iterable[Panel], columns: int = 2) -> None:
    panel_rows = list(panels)
    if not panel_rows:
        raise PlotError(f"{title}: no panels")
    columns = max(1, min(columns, len(panel_rows)))
    width = 1240
    outer_x = 42
    gap_x = 34
    panel_w = (width - 2 * outer_x - gap_x * (columns - 1)) / columns
    panel_h = 310
    top = 66
    gap_y = 24
    rows = math.ceil(len(panel_rows) / columns)
    height = int(top + rows * panel_h + (rows - 1) * gap_y + 28)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<style>{_style()}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="figure-title">{escape(title)}</text>',
    ]
    for panel_index, panel in enumerate(panel_rows):
        column = panel_index % columns
        row = panel_index // columns
        px = outer_x + column * (panel_w + gap_x)
        py = top + row * (panel_h + gap_y)
        margin_left, margin_right = 62.0, 18.0
        legend_columns = min(3, max(1, len(panel.series)))
        if any(len(series.label) > 16 for series in panel.series):
            legend_columns = min(2, legend_columns)
        legend_rows = math.ceil(len(panel.series) / legend_columns)
        margin_top, margin_bottom = 45.0 + legend_rows * 16.0, 48.0
        x0, x1 = px + margin_left, px + panel_w - margin_right
        y0, y1 = py + margin_top, py + panel_h - margin_bottom
        all_points = [point for series in panel.series for point in series.points]
        if not all_points:
            continue
        x_values = [point.x for point in all_points]
        x_values.extend(value for value, _ in panel.x_ticks)
        y_values = [point.y for point in all_points]
        y_values.extend(point.low for point in all_points if point.low is not None)
        y_values.extend(point.high for point in all_points if point.high is not None)
        xmin, xmax = min(x_values), max(x_values)
        if xmin == xmax:
            xmin, xmax = xmin - 0.5, xmax + 0.5
        ymax = _nice_max(max(y_values) * 1.08)

        def sx(value: float) -> float:
            return x0 + (value - xmin) / (xmax - xmin) * (x1 - x0)

        def sy(value: float) -> float:
            return y1 - value / ymax * (y1 - y0)

        svg.extend([
            f'<text x="{px + panel_w / 2:.1f}" y="{py + 18:.1f}" text-anchor="middle" class="panel-title">{escape(panel.title)}</text>',
            f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" class="frame"/>',
        ])
        for tick_index in range(5):
            value = ymax * tick_index / 4
            yy = sy(value)
            svg.append(f'<line x1="{x0:.1f}" y1="{yy:.1f}" x2="{x1:.1f}" y2="{yy:.1f}" class="grid"/>')
            svg.append(f'<text x="{x0 - 7:.1f}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{escape(_fmt(value))}</text>')
        for value, label in panel.x_ticks:
            xx = sx(value)
            svg.append(f'<line x1="{xx:.1f}" y1="{y1:.1f}" x2="{xx:.1f}" y2="{y1 + 4:.1f}" class="axis"/>')
            svg.append(f'<text x="{xx:.1f}" y="{y1 + 17:.1f}" text-anchor="middle" class="tick">{escape(label)}</text>')
        svg.append(
            f'<text x="{px + 12:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 {px + 12:.1f} {(y0 + y1) / 2:.1f})" class="axis-title">{escape(panel.y_label)}</text>'
        )
        svg.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{py + panel_h - 7:.1f}" text-anchor="middle" class="axis-title">{escape(x_label)}</text>')
        legend_x = x0
        legend_cell_width = (x1 - x0) / legend_columns
        for series_index, series in enumerate(panel.series):
            color = COLORS[series_index % len(COLORS)]
            dash = ' stroke-dasharray="7 4"' if series_index % 3 == 1 else ""
            legend_column = series_index % legend_columns
            legend_row = series_index // legend_columns
            lx = legend_x + legend_column * legend_cell_width
            ly = py + 35 + legend_row * 15
            svg.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx + 18:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="2.2"{dash}/>')
            svg.append(f'<text x="{lx + 23:.1f}" y="{ly + 4:.1f}" class="legend">{escape(series.label)}</text>')
            ordered = sorted(series.points, key=lambda point: point.x)
            path_data = " ".join(
                ("M" if idx == 0 else "L") + f" {sx(point.x):.1f} {sy(point.y):.1f}"
                for idx, point in enumerate(ordered)
            )
            svg.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
            for point in ordered:
                xx, yy = sx(point.x), sy(point.y)
                if point.low is not None and point.high is not None:
                    low_y, high_y = sy(point.low), sy(point.high)
                    svg.extend([
                        f'<line x1="{xx:.1f}" y1="{high_y:.1f}" x2="{xx:.1f}" y2="{low_y:.1f}" stroke="{color}" class="whisker"/>',
                        f'<line x1="{xx - 4:.1f}" y1="{high_y:.1f}" x2="{xx + 4:.1f}" y2="{high_y:.1f}" stroke="{color}" class="whisker"/>',
                        f'<line x1="{xx - 4:.1f}" y1="{low_y:.1f}" x2="{xx + 4:.1f}" y2="{low_y:.1f}" stroke="{color}" class="whisker"/>',
                    ])
                tooltip = f"{series.label}; x={point.x_label or _fmt(point.x)}; y={_fmt(point.y)} {panel.y_label}"
                svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{color}"><title>{escape(tooltip)}</title></circle>')
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _horizontal_bars(path: Path, title: str,
                     groups: dict[str, list[tuple[str, float, float, float]]]) -> None:
    entries = sum(len(rows) for rows in groups.values())
    width, row_h = 1240, 29
    height = 78 + entries * row_h + len(groups) * 38
    left, right = 360.0, 55.0
    plot_w = width - left - right
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<style>{_style()}</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" class="figure-title">{escape(title)}</text>',
    ]
    y = 66.0
    for group, rows in groups.items():
        maximum = _nice_max(max(high for _, _, _, high in rows) * 1.05)
        svg.append(f'<text x="{left:.1f}" y="{y:.1f}" class="panel-title">{escape(group)}</text>')
        y += 19
        for index, (label, value, low, high) in enumerate(rows):
            color = COLORS[index % len(COLORS)]
            bar_w = value / maximum * plot_w
            low_x = left + low / maximum * plot_w
            high_x = left + high / maximum * plot_w
            svg.extend([
                f'<text x="{left - 10:.1f}" y="{y + 15:.1f}" text-anchor="end" class="tick">{escape(label)}</text>',
                f'<rect x="{left:.1f}" y="{y + 3:.1f}" width="{bar_w:.1f}" height="15" fill="{color}" opacity="0.78"><title>{escape(label)}: {_fmt(value)} {escape(group)}</title></rect>',
                f'<line x1="{low_x:.1f}" y1="{y + 10.5:.1f}" x2="{high_x:.1f}" y2="{y + 10.5:.1f}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{left + bar_w + 7:.1f}" y="{y + 15:.1f}" class="tick">{escape(_fmt(value))}</text>',
            ])
            y += row_h
        y += 19
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _compute_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    rows = _runtime_rows(data)
    artifacts: list[tuple[Path, str]] = []
    for launch, launch_label in (
        ("single_warp_block", "Single-warp compute throughput"),
        ("full_sm_4warp_block", "Full-SM compute throughput"),
    ):
        selected = [row for row in rows if str(row.get("case_id", "")).endswith(launch)]
        panels: list[Panel] = []
        for precision_id in sorted({str(row["precision_id"]) for row in selected}):
            precision_rows = [row for row in selected if row["precision_id"] == precision_id]
            unit = "TFLOP/s" if precision_rows[0].get("work_unit") == "flop" else "TOP/s"
            points = []
            for row in precision_rows:
                match = re.search(r"_m128n(\d+)k\d+_", str(row["case_id"]))
                if not match:
                    raise PlotError(f"cannot derive MMA N from {row['case_id']}")
                n = int(match.group(1))
                scale = 1e12
                points.append(Point(
                    x=math.log2(n),
                    x_label=str(n),
                    y=_finite(row["rate_per_second_median"], "compute median") / scale,
                    low=_finite(row["rate_per_second_min"], "compute min") / scale,
                    high=_finite(row["rate_per_second_max"], "compute max") / scale,
                ))
            panels.append(Panel(
                title=precision_id,
                y_label=unit,
                series=(Series("median (min–max)", tuple(points)),),
                x_ticks=tuple((math.log2(n), str(n)) for n in (64, 128, 256)),
            ))
        if panels:
            path = output / f"compute-{launch.replace('_block', '').replace('_', '-')}.svg"
            _line_figure(path, launch_label, "MMA atom N", panels, columns=3)
            artifacts.append((path, "compute throughput by MMA N"))
    return artifacts


def _payload_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    rows = _runtime_rows(data)
    panels = []
    for residency, label in (("hot_l2", "Hot L2, isolated per-SM ingress"),
                             ("cold_hbm", "Cold DRAM, full-GPU ingress")):
        selected = sorted((row for row in rows if row.get("residency") == residency),
                          key=lambda row: int(row["tile_bytes"]))
        points = tuple(Point(
            x=math.log2(int(row["tile_bytes"]) / 1024),
            x_label=f"{int(row['tile_bytes']) // 1024} KiB",
            y=_finite(row["rate_per_second_median"], "TMA median") / 1e9,
            low=_finite(row["rate_per_second_min"], "TMA min") / 1e9,
            high=_finite(row["rate_per_second_max"], "TMA max") / 1e9,
        ) for row in selected)
        if points:
            panels.append(Panel(
                label, "GB/s", (Series("median (min–max)", points),),
                tuple((math.log2(value), str(value)) for value in (4, 8, 16, 32, 64)),
            ))
    if not panels:
        return []
    path = output / "tma-throughput-by-payload.svg"
    _line_figure(path, "TMA payload service curves", "Payload (KiB)", panels)
    return [(path, "TMA throughput by payload and residency")]


def _duplex_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    rows = _runtime_rows(data)
    panels = []
    for prefix, label in (("hbm", "Cold DRAM duplex surface"),
                          ("l2", "Hot L2 duplex surface")):
        selected: list[tuple[float, dict[str, Any], int, int]] = []
        for row in rows:
            match = re.search(rf"{prefix}\.duplex\.r(\d+)_w(\d+)$", str(row.get("resource", "")))
            if match:
                read_ops, write_ops = int(match.group(1)), int(match.group(2))
                selected.append((read_ops / (read_ops + write_ops), row, read_ops, write_ops))
        selected.sort(key=lambda item: item[0])
        if not selected:
            continue
        total, read, write = [], [], []
        for share, row, _, _ in selected:
            median = _finite(row["rate_per_second_median"], "duplex median") / 1e9
            low = _finite(row["rate_per_second_min"], "duplex min") / 1e9
            high = _finite(row["rate_per_second_max"], "duplex max") / 1e9
            total.append(Point(share * 100, median, low, high, f"{share * 100:.0f}%"))
            read.append(Point(share * 100, median * share, x_label=f"{share * 100:.0f}%"))
            write.append(Point(share * 100, median * (1 - share), x_label=f"{share * 100:.0f}%"))
        panels.append(Panel(
            label, "GB/s",
            (Series("total", tuple(total)), Series("read share", tuple(read)),
             Series("write share", tuple(write))),
            tuple((value, f"{value}%") for value in (0, 25, 50, 75, 100)),
        ))
    if not panels:
        return []
    path = output / "memory-duplex-service-curves.svg"
    _line_figure(path, "Simultaneous read/write service curves", "Read share of issued bytes", panels)
    return [(path, "memory duplex total/read/write service curves")]


def _full_gemm_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    rows = _runtime_rows(data)
    throughput_panels, ratio_panels = [], []
    for precision_id in sorted({str(row["precision_id"]) for row in rows}):
        selected = sorted((row for row in rows if row["precision_id"] == precision_id),
                          key=lambda row: int(row["n"]))
        if not selected:
            continue
        unit = "TFLOP/s" if selected[0].get("work_unit") == "flop" else "TOP/s"
        candidate, reference, ratio = [], [], []
        for row in selected:
            n = int(row["n"])
            x = math.log2(n)
            candidate.append(Point(
                x, _finite(row["custom_rate_per_second_median"], "candidate median") / 1e12,
                _finite(row["custom_rate_per_second_min"], "candidate min") / 1e12,
                _finite(row["custom_rate_per_second_max"], "candidate max") / 1e12,
                str(n),
            ))
            reference.append(Point(
                x, _finite(row["reference_rate_per_second_median"], "reference median") / 1e12,
                x_label=str(n),
            ))
            ratio.append(Point(
                x, _finite(row["ratio_of_paired_medians"], "reference ratio") * 100,
                x_label=str(n),
            ))
        ticks = tuple((math.log2(n), str(n)) for n in (1024, 2048, 4096))
        throughput_panels.append(Panel(
            precision_id, unit,
            (Series("candidate", tuple(candidate)), Series("reference", tuple(reference))), ticks,
        ))
        ratio_panels.append(Panel(
            precision_id, "% of reference", (Series("candidate/reference", tuple(ratio)),), ticks,
        ))
    if not throughput_panels:
        return []
    throughput_path = output / "full-gemm-throughput-by-n.svg"
    ratio_path = output / "full-gemm-ratio-to-reference.svg"
    _line_figure(throughput_path, "Full-GEMM candidate vs same-precision reference",
                 "Square GEMM N", throughput_panels, columns=3)
    _line_figure(ratio_path, "Full-GEMM candidate/reference ratio",
                 "Square GEMM N", ratio_panels, columns=3)
    return [
        (throughput_path, "full-GEMM candidate and reference throughput"),
        (ratio_path, "full-GEMM candidate/reference ratio"),
    ]


def _component_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    rows = _runtime_rows(data)
    groups: dict[str, list[tuple[str, float, float, float]]] = {}
    resource_counts: dict[str, int] = {}
    for row in rows:
        resource = str(row.get("resource", row.get("case_id", "unknown")))
        resource_counts[resource] = resource_counts.get(resource, 0) + 1
    for row in rows:
        unit = str(row.get("rate_unit", "B/s"))
        resource = str(row.get("resource", row.get("case_id", "unknown")))
        if unit != "B/s":
            divisor, label = 1e9, "Epilogue (Gelement/s)"
        elif resource.startswith("tmem.readback"):
            divisor, label = 1e9, "TMEM readback (GB/s)"
        elif resource.startswith("tmem.scale"):
            divisor, label = 1e9, "TMEM scale ingress (GB/s)"
        else:
            divisor, label = 1e9, "Memory and TMA transport (GB/s)"
        case_label = (
            str(row.get("case_id", resource))
            if resource_counts[resource] > 1 else resource
        )
        groups.setdefault(label, []).append((
            case_label,
            _finite(row["rate_per_second_median"], "component median") / divisor,
            _finite(row["rate_per_second_min"], "component min") / divisor,
            _finite(row["rate_per_second_max"], "component max") / divisor,
        ))
    if not groups:
        return []
    for values in groups.values():
        values.sort(key=lambda item: item[0])
    path = output / "component-throughput-by-contract.svg"
    _horizontal_bars(path, "Component throughput by exact resource contract", groups)
    return [(path, "component throughput grouped by compatible unit")]


def _closure_plots(data: dict[str, Any], output: Path) -> list[tuple[Path, str]]:
    observations = data.get("observations")
    if not isinstance(observations, list):
        raise PlotError("closure analysis must contain observations")
    panels, ratio_panels = [], []
    for precision_id in sorted({str(row["precision_id"]) for row in observations}):
        selected = sorted((row for row in observations if row["precision_id"] == precision_id),
                          key=lambda row: int(row["n"]))
        observed, reference, empirical_low, empirical_high, upper_low, upper_high = [], [], [], [], [], []
        to_reference, to_empirical, to_upper = [], [], []
        for row in selected:
            n, scale = int(row["n"]), 1e12
            x = math.log2(n)
            observed_rate = _finite(row["observed_median_per_second"], "observed")
            reference_rate = _finite(row["reference_median_per_second"], "reference")
            empirical_min_rate = _optional_finite(
                row.get("empirical_ideal_min_per_second"), "empirical min")
            empirical_max_rate = _optional_finite(
                row.get("empirical_ideal_max_per_second"), "empirical max")
            upper_min_rate = _optional_finite(
                row.get("conditional_upper_min_per_second"), "upper min")
            upper_max_rate = _optional_finite(
                row.get("conditional_upper_max_per_second"), "upper max")
            observed.append(Point(x, observed_rate / scale, x_label=str(n)))
            reference.append(Point(x, reference_rate / scale, x_label=str(n)))
            if empirical_min_rate is not None:
                empirical_low.append(Point(x, empirical_min_rate / scale, x_label=str(n)))
            if empirical_max_rate is not None:
                empirical_high.append(Point(x, empirical_max_rate / scale, x_label=str(n)))
            if upper_min_rate is not None:
                upper_low.append(Point(x, upper_min_rate / scale, x_label=str(n)))
            if upper_max_rate is not None:
                upper_high.append(Point(x, upper_max_rate / scale, x_label=str(n)))
            if reference_rate > 0:
                to_reference.append(Point(x, observed_rate / reference_rate * 100, x_label=str(n)))
            if empirical_max_rate is not None and empirical_max_rate > 0:
                to_empirical.append(Point(x, observed_rate / empirical_max_rate * 100, x_label=str(n)))
            if upper_min_rate is not None and upper_min_rate > 0:
                to_upper.append(Point(x, observed_rate / upper_min_rate * 100, x_label=str(n)))
        unit = "TFLOP/s" if str(selected[0].get("performance_unit")) == "flop/s" else "TOP/s"
        absolute_series = [
            Series("observed", tuple(observed)), Series("reference", tuple(reference)),
            Series("empirical min", tuple(empirical_low)),
            Series("empirical max", tuple(empirical_high)),
            Series("upper min", tuple(upper_low)), Series("upper max", tuple(upper_high)),
        ]
        ratio_series = [
            Series("observed/reference", tuple(to_reference)),
            Series("observed/empirical max", tuple(to_empirical)),
            Series("observed/tight upper", tuple(to_upper)),
        ]
        panels.append(Panel(
            precision_id, unit,
            tuple(series for series in absolute_series if series.points),
            tuple((math.log2(n), str(n)) for n in (1024, 2048, 4096)),
        ))
        ratio_panels.append(Panel(
            precision_id, "%",
            tuple(series for series in ratio_series if series.points),
            tuple((math.log2(n), str(n)) for n in (1024, 2048, 4096)),
        ))
    if not panels:
        return []
    path = output / "closure-observed-envelope-upper.svg"
    ratio_path = output / "closure-efficiency-ratios.svg"
    _line_figure(path, "Observed performance, empirical envelope, and conditional upper",
                 "Square GEMM N", panels, columns=3)
    _line_figure(ratio_path, "Observed performance relative to reference and model ceilings",
                 "Square GEMM N", ratio_panels, columns=3)
    return [
        (path, "three-layer closure comparison"),
        (ratio_path, "observed/reference/envelope/upper ratios"),
    ]


def _kind(data: dict[str, Any]) -> str:
    if isinstance(data.get("observations"), list):
        return "closure"
    rows = data.get("results")
    if not isinstance(rows, list):
        raise PlotError("unsupported JSON: neither results nor observations are present")
    if not rows:
        return "empty"
    keys = {key for row in rows if isinstance(row, dict) for key in row}
    resources = [str(row.get("resource", "")) for row in rows if isinstance(row, dict)]
    case_ids = [str(row.get("case_id", "")) for row in rows if isinstance(row, dict)]
    if "custom_rate_per_second_median" in keys:
        return "full_gemm"
    if "tile_bytes" in keys:
        return "tma_payload"
    if resources and all(".duplex." in resource for resource in resources):
        return "memory_duplex"
    if "precision_id" in keys and any("_m128n" in case_id for case_id in case_ids):
        return "compute"
    if "resource" in keys:
        return "component"
    return "empty"


def generate_campaign_plots(summary_path: Path,
                            output_dir: Path | None = None) -> dict[str, Any]:
    summary_path = summary_path.resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PlotError("summary root must be a JSON object")
    output = (output_dir or summary_path.parent / "plots").resolve()
    output.mkdir(parents=True, exist_ok=True)
    kind = _kind(data)
    generators = {
        "compute": _compute_plots,
        "tma_payload": _payload_plots,
        "memory_duplex": _duplex_plots,
        "full_gemm": _full_gemm_plots,
        "component": _component_plots,
        "closure": _closure_plots,
        "empty": lambda _data, _output: [],
    }
    artifacts = generators[kind](data, output)
    source_hash = _sha256(summary_path)
    manifest = {
        "schema_version": 1,
        "generator_path": "scripts/sm110_gemm_model/campaign_plots.py",
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "source_json": os.path.relpath(summary_path, output),
        "source_sha256": source_hash,
        "campaign_kind": kind,
        "runtime_rows": (
            len(_runtime_rows(data)) if isinstance(data.get("results"), list)
            else len(data.get("observations", []))
        ),
        "chart_count": len(artifacts),
        "charts": [
            {"path": path.name, "description": description, "sha256": _sha256(path)}
            for path, description in artifacts
        ],
        "evidence_boundary": (
            "derived visualization only; source JSON remains authoritative; "
            "lines and ratios do not promote measured sustained rates to physical uppers"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# SM110 campaign plots",
        "",
        f"Source SHA-256: `{source_hash}`",
        "",
        "> Derived visualization only. The source JSON remains the auditable truth.",
        "",
    ]
    if artifacts:
        for path, description in artifacts:
            lines.extend([f"## {description}", "", f"![{description}]({path.name})", ""])
    else:
        lines.extend([
            "No runtime-rate plot was generated because this summary contains no `status=ok` rows.",
            "Static compilation/SASS success is not runtime performance evidence.",
            "",
        ])
    (output / "index.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest
