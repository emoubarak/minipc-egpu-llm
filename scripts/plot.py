#!/usr/bin/env python3
"""Render docs/throughput.svg from data/results.csv (stdlib only).

Rows with a non-empty `chart` column are plotted. The column reads `panel|label`.
When a measurement is a range (value_low != value_high), the bar goes to the low
value and a lighter segment shows the spread up to the high value.
The SVG follows the viewer's light/dark preference through a CSS media query.
"""
import csv
import html
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "results.csv"
OUT = ROOT / "docs" / "throughput.svg"

PANELS = [
    ("moe-tg", "Qwen3.6-35B-A3B Q4 (22 GB MoE, does not fit in 8 GB VRAM)", "generation, tokens/s"),
    ("moe-pp", "Same model, reading a 5,780-token chat prompt (2026-10-09)", "prompt processing, tokens/s"),
    ("dense-tg", "Qwen3-8B Q4_K_M (4.9 GB dense, fits in VRAM, 2026-05)", "generation, tokens/s"),
]

W = 860
LABEL_W = 330
BAR_X = LABEL_W + 16
BAR_MAX_W = W - BAR_X - 120
ROW_H = 26
BAR_H = 16
PANEL_GAP = 30
TITLE_H = 46


def kind(label):
    if label.startswith("780M"):
        return "igpu"
    if "+" in label or "Ollama" in label:
        return "other"
    return "egpu"


def fmt(v):
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}"


def main():
    rows = [r for r in csv.DictReader(SRC.open()) if r["chart"]]
    by_panel = {}
    for r in rows:
        panel, label = r["chart"].split("|", 1)
        by_panel.setdefault(panel, []).append(
            (label, float(r["value_low"]), float(r["value_high"]), r["date"]))

    parts = []
    y = 56
    for key, title, unit in PANELS:
        items = sorted(by_panel.get(key, []), key=lambda it: -it[2])
        if not items:
            continue
        vmax = max(hi for _, _, hi, _ in items)
        scale = BAR_MAX_W / (vmax * 1.05)
        parts.append(f'<text class="h" x="0" y="{y}">{html.escape(title)}</text>')
        parts.append(f'<text class="u" x="0" y="{y + 17}">{html.escape(unit)}</text>')
        y += TITLE_H - 16
        for label, lo, hi, _date in items:
            k = kind(label)
            w_lo = max(lo * scale, 1)
            w_hi = hi * scale
            cy = y + ROW_H / 2
            parts.append(f'<text class="l" x="{LABEL_W}" y="{cy + 4}" text-anchor="end">{html.escape(label)}</text>')
            parts.append(f'<rect class="b {k}" x="{BAR_X}" y="{cy - BAR_H / 2}" width="{w_lo:.1f}" height="{BAR_H}" rx="2"/>')
            if hi > lo:
                parts.append(f'<rect class="b {k} r" x="{BAR_X + w_lo:.1f}" y="{cy - BAR_H / 2}" '
                             f'width="{w_hi - w_lo:.1f}" height="{BAR_H}" rx="2"/>')
            val = fmt(lo) if hi == lo else f"{fmt(lo)}–{fmt(hi)}"
            parts.append(f'<text class="v" x="{BAR_X + w_hi + 6:.1f}" y="{cy + 4}">{val}</text>')
            y += ROW_H
        y += PANEL_GAP

    legend_y = y - 8
    legend = []
    lx = 0
    for k, text in (("egpu", "RTX 3060 Ti eGPU (OCuLink)"), ("igpu", "Radeon 780M iGPU"),
                    ("other", "split / other runtime"), ("egpu r", "measured range")):
        legend.append(f'<rect class="b {k}" x="{lx}" y="{legend_y - 10}" width="12" height="12" rx="2"/>')
        legend.append(f'<text class="u" x="{lx + 18}" y="{legend_y}">{text}</text>')
        lx += 18 + 7.2 * len(text) + 26
    height = legend_y + 34

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W + 28}" height="{height:.0f}" viewBox="0 0 {W + 28} {height:.0f}" role="img" aria-label="Throughput of llama.cpp on a Ryzen 7 8845HS mini PC with an RTX 3060 Ti over OCuLink and a Radeon 780M iGPU">
<style>
  text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .t {{ fill: #1f2328; }}
  .h {{ fill: #1f2328; }}
  .l {{ fill: #1f2328; }}
  .t {{ font-size: 17px; font-weight: 600; }}
  .h {{ font-size: 14px; font-weight: 600; }}
  .u {{ font-size: 12px; fill: #59636e; }}
  .l {{ font-size: 12.5px; }}
  .v {{ font-size: 12.5px; font-variant-numeric: tabular-nums; fill: #59636e; }}
  .egpu {{ fill: #2f6fd6; }}
  .igpu {{ fill: #d97a1f; }}
  .other {{ fill: #8c959f; }}
  .r {{ fill-opacity: 0.38; }}
  @media (prefers-color-scheme: dark) {{
    .t {{ fill: #e6edf3; }}
    .h {{ fill: #e6edf3; }}
    .l {{ fill: #e6edf3; }}
    .u {{ fill: #9198a1; }}
    .v {{ fill: #9198a1; }}
    .egpu {{ fill: #5b93f0; }}
    .igpu {{ fill: #f0a050; }}
    .other {{ fill: #6e7781; }}
    .r {{ fill-opacity: 0.5; }}
  }}
</style>
<g transform="translate(14,4)">
<text class="t" x="0" y="22">llama.cpp on a Ryzen 7 8845HS mini PC + RTX 3060 Ti 8 GB (OCuLink)</text>
{chr(10).join(parts)}
{chr(10).join(legend)}
</g>
</svg>
'''
    OUT.write_text(svg)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} bars)")


if __name__ == "__main__":
    main()
