"""Figure 5: duration by geometry against the post-battery reference.

Reads duration_geometry/results_summary.json from the software archive
and writes <output_dir>/figure5_duration_geometry.{pdf,png,svg}.
This script draws Figure 5 of the article.

Usage: python make_figure5_manuscript.py <results_summary.json> <output_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GREY = "#304d70", "#18847b", "#78828c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "savefig.facecolor": "white",
                     "svg.hashsalt": "figure5-duration"})

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
designs = {d["design"]: d for d in summary["designs"]}
contrasts = {c["contrast"]: c for c in summary["contrasts"]}

fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.7))
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.9, wspace=0.95)

ax = axs[0]
for placement, colour, marker, shift in [("contiguous", NAVY, "o", -0.06), ("distributed", TEAL, "s", 0.06)]:
    xs, ys, lo, hi = [], [], [], []
    for k, n in enumerate([5, 10, 20]):
        d = designs[f"{placement}_{n}"]
        xs.append(k + shift)
        ys.append(d["rho_tilde"])
        lo.append(d["rho_tilde"] - d["rho_tilde_ci95"][0])
        hi.append(d["rho_tilde_ci95"][1] - d["rho_tilde"])
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt=marker + "-", color=colour, capsize=2, ms=4, lw=1,
                label=placement.capitalize())
ax.set_xticks(range(3), ["20 s", "40 s", "80 s"])
ax.set_xlim(-0.4, 2.4)
ax.set_xlabel("Pre-battery sample duration")
ax.set_ylabel("Agreement with post-battery\nreference ($\\widetilde{\\rho}$)")
ax.legend(frameon=False, loc="lower right")

ax = axs[1]
rows = [
    ("distributed_5 - contiguous_5", "20 s: distributed − contiguous", TEAL),
    ("distributed_10 - contiguous_10", "40 s: distributed − contiguous", TEAL),
    ("distributed_20 - contiguous_20", "80 s: distributed − contiguous", TEAL),
    ("contiguous_20 - contiguous_10", "Contiguous: 80 s − 40 s", NAVY),
    ("distributed_10 - contiguous_20", "Distributed 40 s − contiguous 80 s", GREY),
]
for y, (key, label, colour) in zip(range(len(rows) - 1, -1, -1), rows):
    c = contrasts[key]
    e = c["delta_z"]
    q = c["delta_z_ci95"]
    ax.errorbar(e, y, xerr=[[e - q[0]], [q[1] - e]], fmt="o", color=colour, capsize=2, ms=4, lw=1)
ax.set_yticks(range(len(rows) - 1, -1, -1), [r[1] for r in rows])
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.axvline(0, color=GREY, ls="--", lw=0.8)
ax.set_xlabel("Fisher-scale difference ($\\Delta z$)")

for j, a in enumerate(axs):
    a.text(-0.12 if j == 0 else -0.95, 1.06, ["(a)", "(b)"][j], transform=a.transAxes, fontweight="bold", fontsize=10)
for ext in ["pdf", "png", "svg"]:
    fig.savefig(out / f"figure5_duration_geometry.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.04)
print("written", out)
