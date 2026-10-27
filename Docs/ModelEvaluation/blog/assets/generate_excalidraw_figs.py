#!/usr/bin/env python3
"""Generate Simon Boehm (siboehm.com/articles/22/CUDA-MMM) style Excalidraw figures."""
import math
import random
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent


class RoughCanvas:
    def __init__(self, width: int, height: int, seed: int = 42):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.elements = []

    def _jitter(self, val: float, amp: float = 1.2) -> float:
        return val + self.rng.uniform(-amp, amp)

    def _rough_line_path(self, x1: float, y1: float, x2: float, y2: float, roughness: float = 1.1) -> str:
        length = math.hypot(x2 - x1, y2 - y1)
        scale = min(1.0, length / 110.0) * roughness
        paths = []
        for pass_idx in range(2):
            amp = 1.35 * scale if pass_idx == 0 else 0.9 * scale
            sx = self._jitter(x1, amp)
            sy = self._jitter(y1, amp)
            ex = self._jitter(x2, amp)
            ey = self._jitter(y2, amp)
            mx1 = x1 + (x2 - x1) * 0.33 + self.rng.uniform(-2.2, 2.2) * scale
            my1 = y1 + (y2 - y1) * 0.33 + self.rng.uniform(-2.2, 2.2) * scale
            mx2 = x1 + (x2 - x1) * 0.67 + self.rng.uniform(-2.2, 2.2) * scale
            my2 = y1 + (y2 - y1) * 0.67 + self.rng.uniform(-2.2, 2.2) * scale
            paths.append(f"M {sx:.2f} {sy:.2f} C {mx1:.2f} {my1:.2f}, {mx2:.2f} {my2:.2f}, {ex:.2f} {ey:.2f}")
        return " ".join(paths)

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#1e1e1e",
             stroke_width: float = 1.8, opacity: float = 1.0, roughness: float = 1.0):
        d = self._rough_line_path(x1, y1, x2, y2, roughness)
        self.elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-linecap="round" opacity="{opacity}"/>'
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str = "#1e1e1e", stroke_width: float = 1.8, roughness: float = 1.0,
             fill_opacity: float = 1.0):
        if fill != "none":
            pts = [
                (self._jitter(x, 0.9), self._jitter(y, 0.9)),
                (self._jitter(x + w, 0.9), self._jitter(y, 0.9)),
                (self._jitter(x + w, 0.9), self._jitter(y + h, 0.9)),
                (self._jitter(x, 0.9), self._jitter(y + h, 0.9)),
            ]
            pt_str = " ".join(f"{px:.2f},{py:.2f}" for px, py in pts)
            self.elements.append(
                f'<polygon points="{pt_str}" fill="{fill}" fill-opacity="{fill_opacity}" stroke="none"/>'
            )
        if stroke != "none":
            d = " ".join([
                self._rough_line_path(x, y, x + w, y, roughness),
                self._rough_line_path(x + w, y, x + w, y + h, roughness),
                self._rough_line_path(x + w, y + h, x, y + h, roughness),
                self._rough_line_path(x, y + h, x, y, roughness),
            ])
            self.elements.append(
                f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

    def grid_rect(self, x: float, y: float, w: float, h: float, rows: int, cols: int,
                  fill: str = "#fa5252", stroke: str = "#1e1e1e", grid_opacity: float = 0.40):
        self.rect(x, y, w, h, fill=fill, stroke="none")
        for r in range(1, rows):
            gy = y + h * r / rows
            self.line(x, gy, x + w, gy, stroke=stroke, stroke_width=1.0, opacity=grid_opacity, roughness=0.65)
        for c in range(1, cols):
            gx = x + w * c / cols
            self.line(gx, y, gx, y + h, stroke=stroke, stroke_width=1.0, opacity=grid_opacity, roughness=0.65)
        self.rect(x, y, w, h, fill="none", stroke=stroke, stroke_width=1.8, roughness=1.0)

    def dashed_box(self, x: float, y: float, w: float, h: float, rx: float = 16.0,
                   stroke: str = "#1e1e1e", stroke_width: float = 2.1, dash: str = "10,8",
                   fill: str = "none", fill_opacity: float = 1.0):
        if fill != "none":
            self.elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
                f'fill="{fill}" fill-opacity="{fill_opacity}" stroke="none"/>'
            )
        for dx, dy, sw in [(0.0, 0.0, stroke_width), (0.8, -0.6, stroke_width * 0.6)]:
            self.elements.append(
                f'<rect x="{x + dx:.2f}" y="{y + dy:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{rx:.2f}" '
                f'fill="none" stroke="{stroke}" stroke-width="{sw:.2f}" stroke-dasharray="{dash}" '
                f'stroke-linecap="round"/>'
            )

    def curved_arrow(self, x1: float, y1: float, cx1: float, cy1: float,
                     cx2: float, cy2: float, x2: float, y2: float,
                     stroke: str = "#1e1e1e", stroke_width: float = 2.0, head_len: float = 15.0):
        paths = []
        for pass_idx in range(2):
            amp = 1.1 if pass_idx == 0 else 0.7
            sx, sy = self._jitter(x1, amp), self._jitter(y1, amp)
            ex, ey = self._jitter(x2, amp * 0.4), self._jitter(y2, amp * 0.4)
            p_cx1, p_cy1 = self._jitter(cx1, amp * 1.4), self._jitter(cy1, amp * 1.4)
            p_cx2, p_cy2 = self._jitter(cx2, amp * 1.4), self._jitter(cy2, amp * 1.4)
            paths.append(f"M {sx:.2f} {sy:.2f} C {p_cx1:.2f} {p_cy1:.2f}, {p_cx2:.2f} {p_cy2:.2f}, {ex:.2f} {ey:.2f}")
        d = " ".join(paths)
        self.elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round"/>'
        )
        angle = math.atan2(y2 - cy2, x2 - cx2)
        for sign in (-1, 1):
            a = angle + math.pi + sign * 0.40
            hx = x2 + head_len * math.cos(a)
            hy = y2 + head_len * math.sin(a)
            self.line(x2, y2, hx, hy, stroke=stroke, stroke_width=stroke_width, roughness=0.6)

    def text(self, x: float, y: float, content: str, size: int = 20, weight: str = "bold",
             color: str = "#1e1e1e", anchor: str = "middle", family: str = "'Comic Neue', sans-serif"):
        esc = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}px" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{esc}</text>'
        )

    def save(self, svg_path: Path, png_path: Path):
        svg_content = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'  <rect width="100%" height="100%" fill="#ffffff"/>\n  '
            + "\n  ".join(self.elements)
            + "\n</svg>\n"
        )
        svg_path.write_text(svg_content, encoding="utf-8")
        subprocess.run(
            ["rsvg-convert", "-d", "180", "-p", "180", str(svg_path), "-o", str(png_path)],
            check=True,
        )


def build_fig1():
    """Proportional matrix dimensions both outside (128x128 += 128x64 * 64x128)
    and inside the zoom-in callout (64x128 += 64x16 * 16x128)."""
    c = RoughCanvas(860, 560, seed=101)

    # Scale: 16 elements = 21 px -> 64 = 84 px, 128 = 168 px
    # Top-left: Accumulator D (128x128) = two 64x128 blocks (d[0] top, d[1] bottom)
    dx, dy, dw, dh = 70, 55, 168, 168
    c.text(dx + dw / 2, dy - 16, "D (128x128)", size=19, weight="bold")
    c.grid_rect(dx, dy, dw, dh / 2, rows=4, cols=8, fill="#f84f4f")
    c.grid_rect(dx, dy + dh / 2, dw, dh / 2, rows=4, cols=8, fill="#ff8787")
    c.dashed_box(dx - 8, dy - 8, dw + 16, dh / 2 + 16, rx=10, stroke="#1e1e1e", stroke_width=2.1, dash="8,6")
    c.text(dx - 32, dy + 48, "d[0]", size=19, weight="bold")
    c.text(dx - 32, dy + dh / 2 + 48, "d[1]", size=19, weight="bold")

    # "+="
    c.text(285, dy + dh / 2 + 10, "+=", size=30, weight="bold")

    # Middle: sA (128x64) = 2 M-slices (64 each) x 4 K-slices (16 each)
    ax, ay, aw, ah = 335, 55, 84, 168
    c.text(ax + aw / 2, ay - 16, "sA (128x64)", size=19, weight="bold")
    k_colors = ["#ffd43b", "#a9e34b", "#74c0fc", "#ffa94d"]
    kw = aw / 4.0  # 21 px per 16-col slice
    for k in range(4):
        c.grid_rect(ax + k * kw, ay, kw, ah / 2, rows=4, cols=1, fill=k_colors[k])
        c.grid_rect(ax + k * kw, ay + ah / 2, kw, ah / 2, rows=4, cols=1, fill=k_colors[k], grid_opacity=0.25)

    # "*"
    c.text(462, dy + dh / 2 + 14, "*", size=32, weight="bold")

    # Right: sB (64x128) = 4 K-slices (16 each) x 128 columns
    bx, by, bw, bh = 505, 55, 168, 84
    c.text(bx + bw / 2, by - 16, "sB (64x128)", size=19, weight="bold")
    kh = bh / 4.0  # 21 px per 16-row slice
    for k in range(4):
        c.grid_rect(bx, by + k * kh, bw, kh, rows=1, cols=8, fill=k_colors[k])

    # Bottom dashed container: Single WGMMA Atom (m64n128k16): 64x128 += (64x16) * (16x128)
    cx, cy, cw, ch = 165, 285, 645, 240
    c.dashed_box(cx, cy, cw, ch, rx=22, stroke="#1e1e1e", stroke_width=2.2, dash="10,8")
    c.text(cx + cw / 2, cy + 38, "1 of 8 WGMMA Atoms (m64n128k16, 64 cycles)", size=22, weight="bold")

    # True proportional shapes inside container (16 -> 21px, 64 -> 84px, 128 -> 168px)
    by_in = cy + 72
    # d[0]: 64x128 (height 84, width 168)
    c.grid_rect(cx + 38, by_in, 168, 84, rows=4, cols=8, fill="#f84f4f")
    c.text(cx + 38 + 84, by_in + 115, "d[0] (64x128)", size=18, weight="bold")

    c.text(cx + 242, by_in + 50, "+=", size=28, weight="bold")

    # desc_a: 64x16 (height 84, width 21)
    c.grid_rect(cx + 282, by_in, 24, 84, rows=4, cols=1, fill="#ffd43b")
    c.text(cx + 294, by_in + 115, "desc_a (64x16)", size=18, weight="bold")

    c.text(cx + 345, by_in + 54, "*", size=30, weight="bold")

    # desc_b: 16x128 (height 21, width 168)
    c.grid_rect(cx + 385, by_in + 30, 168, 24, rows=1, cols=8, fill="#ffd43b")
    c.text(cx + 385 + 84, by_in + 115, "desc_b (16x128)", size=18, weight="bold")

    # Swooping arrows
    c.curved_arrow(dx + 35, dy + dh / 2 + 10, 75, 325, 110, 395, cx - 8, cy + 120)
    c.curved_arrow(bx + bw + 4, by + kh * 0.5, 765, 95, 785, 225, cx + 470, cy - 6)

    c.save(OUT_DIR / "fig1-wgmma-tiling.svg", OUT_DIR / "fig1-wgmma-tiling.png")


def build_fig2():
    """Separate mainloop step comparison (607c vs 512c) from the warp software timeline (429+28+448=905c)."""
    c = RoughCanvas(860, 500, seed=202)

    # Rows 1-2 draw the fitted model T = a + 512*K_t + n*d on one scale: every tile, the prologue
    # included, is 8 WGMMA = 512c (180 px); d = 95c is 33 px. They are not a hardware trace.
    tile_w, lag_w, gap = 180, 33, 12

    # 1. Kernel 1 mainloop wait<0>
    c.text(40, 38, "Kernel 1 (mainloop wait<0>): 607 cycles / K-tile (K_t - 1 drains)", size=20, weight="bold", anchor="start")
    y0 = 54
    # Prologue tile 0 (no wait)
    c.grid_rect(40, y0, tile_w, 42, rows=1, cols=8, fill="#dee2e6")
    c.text(40 + tile_w / 2, y0 + 64, "Prologue t=0 (512c)", size=16, weight="bold")
    ox = 40 + tile_w + gap
    for i in range(1, 3):
        c.grid_rect(ox, y0, tile_w, 42, rows=1, cols=8, fill="#ffd43b")
        c.rect(ox + tile_w, y0, lag_w, 42, fill="#f84f4f")
        c.text(ox + tile_w / 2, y0 + 64, f"t={i} (512c)", size=16, weight="bold")
        c.text(ox + tile_w + lag_w / 2 + 6, y0 + 64, "+d=95c", size=15, weight="bold", color="#c92a2a")
        ox += tile_w + lag_w + gap

    # 2. Kernel 2 mainloop wait<1>
    c.text(40, 175, "Kernel 2 (mainloop wait<1>): 512 cycles / K-tile (single tail drain)", size=20, weight="bold", anchor="start")
    y1 = 191
    c.grid_rect(40, y1, tile_w, 42, rows=1, cols=8, fill="#dee2e6")
    c.text(40 + tile_w / 2, y1 + 64, "Prologue t=0 (512c)", size=16, weight="bold")
    colors = ["#a9e34b", "#74c0fc"]
    ox = 40 + tile_w + gap
    for i in range(1, 3):
        c.grid_rect(ox, y1, tile_w, 42, rows=1, cols=8, fill=colors[i - 1])
        c.text(ox + tile_w / 2, y1 + 64, f"t={i} (512c)", size=16, weight="bold")
        ox += tile_w + gap
    # Tail wait<0> at the end exposes d once
    c.rect(ox, y1, lag_w, 42, fill="#e599f7")
    c.text(ox + lag_w / 2 + 6, y1 + 64, "+d once", size=15, weight="bold", color="#862e9c")
    c.text(ox + lag_w + 8, y1 + 27, "mma_tail", size=16, weight="bold", anchor="start")
    c.text(820, 278, "Rows 1-2: fitted model T = a + 512 K_t + n d (schematic, not a hardware trace)",
           size=14, weight="normal", color="#495057", anchor="end")

    # 3. Separate full-width box: Warp Software %clock64 Timeline across (Last Mainloop Iter + mma_tail)
    zx, zy, zw, zh = 40, 300, 780, 175
    c.dashed_box(zx, zy, zw, zh, rx=18, stroke="#1e1e1e", stroke_width=2.1, dash="9,7")
    c.text(zx + zw / 2, zy + 30, "Warp Software %clock64 Endpoints: Last Loop Iter (t = K_t-1) + mma_tail", size=19, weight="bold")

    # Scale: 905 cycles total -> ~700 px (0.77 px / cycle)
    bx = zx + 40
    by = zy + 66
    # 429c issue->commit (330 px)
    c.rect(bx, by, 330, 42, fill="#74c0fc")
    c.text(bx + 165, by + 27, "issue -> commit (429c)", size=17, weight="bold")
    c.text(bx + 180, by + 68, "<--- Inside last loop body (429 + 28 = 457c) --->", size=16, weight="bold")

    # 28c commit->wait<1> (42 px)
    c.rect(bx + 330, by, 42, 42, fill="#ffd43b")
    c.text(bx + 351, by - 10, "wait<1> (28c)", size=15, weight="bold")

    # 448c loop_wait_after -> tail_wait0_after (330 px)
    c.rect(bx + 372, by, 330, 42, fill="#e599f7")
    c.text(bx + 372 + 165, by + 27, "mma_tail: wait<0> (448c)", size=17, weight="bold")
    c.text(bx + 372 + 165, by + 68, "<--- Outside loop (448c; commit -> tail = 476c) --->", size=16, weight="bold")

    c.save(OUT_DIR / "fig2-wait1-vs-wait0.svg", OUT_DIR / "fig2-wait1-vs-wait0.png")


def build_fig3():
    """Verified software call and barrier sequence in CUTLASS Ping-Pong (no speculative hardware TC queue or 448c label)."""
    c = RoughCanvas(860, 340, seed=303)

    # Row 1: WG 1 (Software call sequence)
    y0 = 65
    c.text(70, y0 + 28, "WG 1", size=20, weight="bold")
    c.grid_rect(125, y0, 200, 44, rows=1, cols=5, fill="#74c0fc")
    c.text(225, y0 - 12, "mma() (Tile 0)", size=17, weight="bold")

    c.rect(335, y0, 82, 44, fill="#ffd43b")
    c.text(376, y0 - 12, "arrive()", size=17, weight="bold")

    c.rect(427, y0, 150, 44, fill="#e599f7")
    c.text(502, y0 - 12, "mma_tail (wait<0>)", size=16, weight="bold")

    c.rect(587, y0, 62, 44, fill="#e9ecef")
    c.text(618, y0 - 12, "wait()", size=16, weight="bold")

    c.rect(659, y0, 160, 44, fill="#f84f4f")
    c.text(739, y0 - 12, "Epilogue (Tile 0)", size=17, weight="bold")

    # Row 2: WG 2 (Software call sequence)
    y1 = 205
    c.text(70, y1 + 28, "WG 2", size=20, weight="bold")
    c.rect(125, y1, 292, 44, fill="#e9ecef")
    c.text(271, y1 + 68, "order.wait()", size=17, weight="bold")

    c.grid_rect(427, y1, 150, 44, rows=1, cols=4, fill="#a9e34b")
    c.text(502, y1 + 68, "mma() (Tile 1)", size=17, weight="bold")

    c.rect(587, y1, 62, 44, fill="#ffd43b")
    c.text(618, y1 + 68, "arrive()", size=16, weight="bold")

    c.rect(659, y1, 160, 44, fill="#e599f7")
    c.text(739, y1 + 68, "mma_tail (Tile 1)", size=17, weight="bold")

    # Handoff 1: WG1 arrive() -> WG2 mma()
    c.curved_arrow(376, y0 + 44, 382, 130, 405, 175, 425, y1 + 20)

    # Handoff 2: WG2 arrive() -> WG1 Epilogue
    c.curved_arrow(618, y1, 622, 160, 638, 125, 657, y0 + 24)

    c.save(OUT_DIR / "fig3-pingpong-handoff.svg", OUT_DIR / "fig3-pingpong-handoff.png")


def build_fig4():
    """S-stage ring buffer wraparound (slot = g % S, gen = g // S), phase parity, and lagged release credit path."""
    c = RoughCanvas(860, 500, seed=404)

    c.text(40, 36, "Ring Buffer Slot Reuse (S = 6): slot = g % S, gen = g // S", size=20, weight="bold", anchor="start")

    # Generation 0 (g = 0..5)
    y0 = 56
    c.text(40, y0 + 27, "gen = 0", size=18, weight="bold", anchor="start")
    c.text(40, y0 + 46, "full p=0", size=14, weight="normal", color="#495057", anchor="start")
    sx, sw, sh, gap = 130, 106, 52, 10
    gen0_colors = ["#74c0fc", "#a9e34b", "#dee2e6", "#dee2e6", "#dee2e6", "#dee2e6"]
    for s in range(6):
        x = sx + s * (sw + gap)
        c.rect(x, y0, sw, sh, fill=gen0_colors[s])
        c.text(x + sw / 2, y0 + 23, f"Slot {s}", size=16, weight="bold")
        c.text(x + sw / 2, y0 + 43, f"tile g={s}", size=14, weight="normal")

    # Generation 1 (g = 6..11)
    y1 = 152
    c.text(40, y1 + 27, "gen = 1", size=18, weight="bold", anchor="start")
    c.text(40, y1 + 46, "full p=1", size=14, weight="normal", color="#495057", anchor="start")
    gen1_colors = ["#ffa94d", "#dee2e6", "#dee2e6", "#dee2e6", "#dee2e6", "#dee2e6"]
    for s in range(6):
        x = sx + s * (sw + gap)
        c.rect(x, y1, sw, sh, fill=gen1_colors[s])
        c.text(x + sw / 2, y1 + 23, f"Slot {s}", size=16, weight="bold")
        c.text(x + sw / 2, y1 + 43, f"tile g={s + 6}", size=14, weight="normal")

    # Wraparound arrow from Slot 0 (gen 0) to Slot 0 (gen 1)
    c.curved_arrow(sx + sw / 2, y0 + sh, sx + sw / 2 - 20, y0 + sh + 22,
                   sx + sw / 2 - 20, y1 - 22, sx + sw / 2, y1 - 2)
    c.text(sx + sw / 2 + 62, y0 + sh + 27, "reuse Slot 0 at g=S", size=15, weight="bold", color="#c92a2a")

    # Bottom dashed box: Why Slot 0 is released during tile g=1 and acquired by Producer at tile g=S
    bx, by, bw, bh = 40, 242, 780, 232
    c.dashed_box(bx, by, bw, bh, rx=18, stroke="#1e1e1e", stroke_width=2.1, dash="9,7")
    c.text(bx + bw / 2, by + 30, "Lagged Release (wait<1>) & First Wraparound Dependency (release(0) -> acquire(S))", size=18, weight="bold")

    # Consumer row inside box
    cy = by + 62
    c.text(60, cy + 26, "Consumer", size=18, weight="bold", anchor="start")
    # g=0 box
    c.rect(165, cy, 175, 42, fill="#74c0fc")
    c.text(165 + 87.5, cy + 26, "g=0: issue (Slot 0)", size=15, weight="bold")
    # g=1 box
    c.rect(355, cy, 185, 42, fill="#a9e34b")
    c.text(355 + 92.5, cy + 26, "g=1: issue + wait<1>", size=15, weight="bold")
    # release(0) box
    c.rect(555, cy, 235, 42, fill="#ffd43b")
    c.text(555 + 117.5, cy + 26, "arrive(empty[0]) x128", size=15, weight="bold")
    c.text(555 + 117.5, cy - 10, "confirms g=0 done -> release Slot 0", size=14, weight="bold", color="#2b8a3e")

    # Producer row inside box
    py = by + 158
    c.text(60, py + 26, "Producer", size=18, weight="bold", anchor="start")
    c.rect(165, py, 240, 42, fill="#e9ecef")
    c.text(165 + 120, py + 26, "g=0..S-1: empty p=1 (free)", size=15, weight="bold")

    c.rect(425, py, 175, 42, fill="#ffa94d")
    c.text(425 + 87.5, py + 26, "g=S: wait(empty[0])", size=15, weight="bold")

    c.rect(615, py, 175, 42, fill="#f84f4f")
    c.text(615 + 87.5, py + 26, "arrive(full[0], p=1)", size=15, weight="bold")

    # Arrow from Consumer arrive(empty[0]) to Producer wait(empty[0])
    c.curved_arrow(650, cy + 42, 625, cy + 78, 545, py - 32, 515, py - 2)

    c.save(OUT_DIR / "fig4-ring-wraparound.svg", OUT_DIR / "fig4-ring-wraparound.png")


def build_fig5():
    """First-wraparound producer delay (D4096 at tile = S): S=6 vs S=4 slack and +275c feedback."""
    c = RoughCanvas(860, 520, seed=505)

    c.text(40, 34, "First-Wraparound Producer Delay (D4096 at tile = S): Buffer Slack & +275c Feedback",
           size=19, weight="bold", anchor="start")

    # Scale: 0 to 5200 cycles -> ~560 px (0.108 px/cycle), x_origin = 165
    ox = 165
    scale = 0.108

    def cx(cyc: float) -> float:
        return ox + cyc * scale

    # --- Case A: S = 6 (tile = 6 wraparound) ---
    c.text(40, 74, "1. S = 6 (wraparound at g = 6): absorbs 2,304c (0-fb) or 2,029c (+275c fb) -> exposes 2,067c",
           size=17, weight="bold", anchor="start")

    y_c6 = 92
    c.text(45, y_c6 + 25, "Consumer (S=6)", size=16, weight="bold", anchor="start")
    # g=0..1 (768c before release(0)), then g=2..5 buffered slack (2304c until g=6 pipe stall at 3072c)
    c.rect(cx(0), y_c6, 768 * scale, 38, fill="#74c0fc")
    c.text(cx(384), y_c6 + 24, "g=0..1 (768c)", size=14, weight="bold")

    c.rect(cx(768), y_c6, 2304 * scale, 38, fill="#a9e34b")
    c.text(cx(768 + 1152), y_c6 + 24, "Buffered g=2..5 Slack (2,304c)", size=15, weight="bold")

    c.rect(cx(3072), y_c6, 2067 * scale, 38, fill="#f84f4f")
    c.text(cx(3072 + 1033), y_c6 + 24, "Exposed Stall (+2,067c)", size=15, weight="bold")

    c.rect(cx(5139), y_c6, 85, 38, fill="#dee2e6")
    c.text(cx(5139) + 42.5, y_c6 + 24, "g=6..", size=15, weight="bold")

    y_p6 = 148
    c.text(45, y_p6 + 25, "Producer (g=6)", size=16, weight="bold", anchor="start")
    c.rect(cx(768), y_p6, 4096 * scale, 38, fill="#ffd43b")
    c.text(cx(768 + 2048), y_p6 + 24, "Injected Delay D4096 (4,096c)", size=15, weight="bold")

    c.rect(cx(4864), y_p6, 275 * scale, 38, fill="#e599f7")
    c.text(cx(4864 + 137), y_p6 + 56, "+275c feedback", size=14, weight="bold", color="#862e9c")
    c.text(cx(768), y_p6 + 56, "release(0) @ 768c", size=14, weight="bold", color="#2b8a3e")

    # --- Case B: S = 4 (tile = 4 wraparound) ---
    c.text(40, 252, "2. S = 4 (wraparound at g = 4): 2 fewer buffered tiles (-1,024c slack) -> exposes 3,091c (meas 3,045c)",
           size=17, weight="bold", anchor="start")

    y_c4 = 270
    c.text(45, y_c4 + 25, "Consumer (S=4)", size=16, weight="bold", anchor="start")
    c.rect(cx(0), y_c4, 768 * scale, 38, fill="#74c0fc")
    c.text(cx(384), y_c4 + 24, "g=0..1 (768c)", size=14, weight="bold")

    c.rect(cx(768), y_c4, 1280 * scale, 38, fill="#a9e34b")
    c.text(cx(768 + 640), y_c4 + 24, "g=2..3 (1,280c)", size=14, weight="bold")

    c.rect(cx(2048), y_c4, 3091 * scale, 38, fill="#f84f4f")
    c.text(cx(2048 + 1545), y_c4 + 24, "Exposed Stall (+3,091c pred / +3,045c meas)", size=15, weight="bold")

    c.rect(cx(5139), y_c4, 85, 38, fill="#dee2e6")
    c.text(cx(5139) + 42.5, y_c4 + 24, "g=4..", size=15, weight="bold")

    y_p4 = 326
    c.text(45, y_p4 + 25, "Producer (g=4)", size=16, weight="bold", anchor="start")
    c.rect(cx(768), y_p4, 4096 * scale, 38, fill="#ffd43b")
    c.text(cx(768 + 2048), y_p4 + 24, "Injected Delay D4096 (4,096c)", size=15, weight="bold")

    c.rect(cx(4864), y_p4, 275 * scale, 38, fill="#e599f7")
    c.text(cx(4864 + 137), y_p4 + 56, "+275c feedback", size=14, weight="bold", color="#862e9c")

    # Bottom summary box explaining the algebraic equivalence of the 275c placement
    bx, by, bw, bh = 40, 408, 780, 92
    c.dashed_box(bx, by, bw, bh, rx=14, stroke="#1e1e1e", stroke_width=1.9, dash="8,6")
    c.text(bx + bw / 2, by + 30,
           "Model Slack Formula: Exposed(S) = max(0, D - (512*(S-2) + 256) + f_sync),  f_sync = 275c",
           size=16, weight="bold")
    c.text(bx + bw / 2, by + 58,
           "Observability note: placing f_sync = 275c on empty release visibility or on blocked full wait return",
           size=14, weight="normal", color="#495057")
    c.text(bx + bw / 2, by + 78,
           "yields identical window predictions across all S and K_t (schematic model, not a physical trace).",
           size=14, weight="normal", color="#495057")

    c.save(OUT_DIR / "fig5-stage-slack-d4096.svg", OUT_DIR / "fig5-stage-slack-d4096.png")


def build_fig6():
    """Why mainloop wait<0> on the synchronized B-chain costs +228.4c/step (+2,512c over 11 interior steps)."""
    c = RoughCanvas(860, 470, seed=606)

    c.text(40, 36, "Interior Step (g -> g+1) on Synchronized B-Chain: wait<1> vs wait<0>",
           size=20, weight="bold", anchor="start")

    # 1. wait<1> on B-chain: 512.0c / step (control work hidden inside post-wait<1> overlap window)
    c.text(40, 74, "1. B-chain wait<1>: 512.0c / step (133.4c sync/control hidden inside 161c overlap window)",
           size=17, weight="bold", anchor="start")
    y1 = 90
    c.text(45, y1 + 24, "TC Pipe", size=16, weight="bold", anchor="start")
    c.grid_rect(155, y1, 320, 38, rows=1, cols=8, fill="#a9e34b")
    c.text(155 + 160, y1 + 24, "Tile g WGMMA (512c)", size=15, weight="bold")
    c.grid_rect(475, y1, 240, 38, rows=1, cols=6, fill="#74c0fc")
    c.text(475 + 120, y1 + 24, "Tile g+1 WGMMA (512c)", size=15, weight="bold")

    y1_s = 138
    c.text(45, y1_s + 24, "Control", size=16, weight="bold", anchor="start")
    # wait<1> returns 161c before next batch must issue; 133.4c control fits inside
    c.rect(255, y1_s, 95, 34, fill="#dee2e6")
    c.text(255 + 47.5, y1_s + 22, "wait<1>", size=14, weight="bold")
    c.rect(350, y1_s, 105, 34, fill="#ffd43b")
    c.text(350 + 52.5, y1_s + 22, "sync (133.4c)", size=14, weight="bold")
    c.text(555, y1_s + 22, "<- hidden under ongoing Tile g (27.6c margin)", size=14, weight="bold", color="#2b8a3e", anchor="start")

    # 2. Pure d=95 prediction (Resident A-chain model): 607.0c / step (+1,045c over 11 steps - REFUTED on B-chain)
    c.text(40, 212, "2. Resident-only d = 95c hypothesis: 607.0c / step (+1,045c at K13 — refuted on B-chain)",
           size=17, weight="bold", anchor="start")
    y2 = 228
    c.text(45, y2 + 24, "Model", size=16, weight="bold", anchor="start")
    c.grid_rect(155, y2, 320, 38, rows=1, cols=8, fill="#dee2e6")
    c.text(155 + 160, y2 + 24, "Tile g (512c)", size=15, weight="bold")
    c.rect(475, y2, 62, 38, fill="#f84f4f")
    c.text(475 + 31, y2 + 24, "+95c", size=15, weight="bold")
    c.grid_rect(537, y2, 200, 38, rows=1, cols=5, fill="#dee2e6")
    c.text(537 + 100, y2 + 24, "Tile g+1", size=15, weight="bold")

    # 3. Actual wait<0> on B-chain: 740.4c / step (+228.4c/step = +2,512c over 11 interior steps)
    c.text(40, 308, "3. B-chain wait<0>: 740.4c / step (wait<0> drains pipe -> exposes +95c + 133.4c = +228.4c/step)",
           size=17, weight="bold", anchor="start")
    y3 = 324
    c.text(45, y3 + 24, "Critical", size=16, weight="bold", anchor="start")
    c.grid_rect(155, y3, 320, 38, rows=1, cols=8, fill="#ffd43b")
    c.text(155 + 160, y3 + 24, "Tile g (512c)", size=15, weight="bold")

    c.rect(475, y3, 62, 38, fill="#f84f4f")
    c.text(475 + 31, y3 + 24, "+95c", size=15, weight="bold")

    c.rect(537, y3, 96, 38, fill="#e599f7")
    c.text(537 + 48, y3 + 24, "+133.4c", size=15, weight="bold")

    c.grid_rect(633, y3, 180, 38, rows=1, cols=4, fill="#ffd43b")
    c.text(633 + 90, y3 + 24, "Tile g+1 (512c)", size=15, weight="bold")

    # Callout label under +95c and +133.4c
    c.text(554, y3 + 64, "11 interior steps at K_t=13: 11 * (95 + 133.36) = +2,512c (7,572 -> 10,084c)",
           size=15, weight="bold", color="#c92a2a")
    c.text(554, y3 + 88, "Held-out K_t=32 (30 interior steps): pred 24,151c vs meas 24,087c (0.27% error)",
           size=14, weight="normal", color="#495057")
    c.text(820, 448, "Schematic decomposition of fitted step model (three algebraic placements share identical predictions)",
           size=13, weight="normal", color="#495057", anchor="end")

    c.save(OUT_DIR / "fig6-wait0-exposed-control.svg", OUT_DIR / "fig6-wait0-exposed-control.png")


if __name__ == "__main__":
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    print("Generated 6 Simon-Boehm-style figures in", OUT_DIR)
