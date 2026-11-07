"""Figure 6 and Figure S6: scalp topography of the sampling-geometry contrast.

Reads topography/results_summary.json from the software archive and writes
figure6_topography.{pdf,png,svg} (Delta z maps) and
figureS6_topography_levels.{pdf,png,svg} (rho-tilde maps).

Usage: python make_figure6_topography.py <results_summary.json> <output_dir>
Requires matplotlib and mne (electrode positions from the standard_1005 montage).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import mne

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                     "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
                     "svg.hashsalt": "figure6-topography"})
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#24548a", "#8fb0d0", "#f0efec", "#e09a8a", "#a8322a"])
SEQUENTIAL = LinearSegmentedColormap.from_list("grey_teal", ["#f0efec", "#9fcfc8", "#3f9c92", "#0f5e57"])
ROI = ("O1", "Oz", "O2")

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
channels = summary["channels"]
info = mne.create_info(channels, 1000.0, "eeg")
info.set_montage("standard_1005")
mask = np.array([c in ROI for c in channels])
mask_params = dict(marker="o", markerfacecolor="white", markeredgecolor="black", linewidth=0, markersize=4.5)
TITLES = {"within": "Within the pre-battery recording\n(531 participants)",
          "between": "Pre- versus post-battery recording\n(524 participants)"}


def values(comparison, key):
    return np.array([row[key] for row in summary["comparisons"][comparison]["per_channel"]])


def topomap(ax, data, cmap, vlim):
    im, _ = mne.viz.plot_topomap(data, info, axes=ax, show=False, cmap=cmap, vlim=vlim, contours=0,
                                 sensors=True, mask=mask, mask_params=mask_params, outlines="head",
                                 extrapolate="head", image_interp="cubic", res=200)
    return im


# Figure 6: Delta z maps on one colour scale centred at zero.
deltas = {c: values(c, "delta_z") for c in ("within", "between")}
limit = max(np.abs(v).max() for v in deltas.values())
limit = float(np.ceil(limit * 20) / 20)
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.75))
fig.subplots_adjust(left=0.02, right=0.86, bottom=0.04, top=0.82, wspace=0.08)
for j, comparison in enumerate(("within", "between")):
    im = topomap(axs[j], deltas[comparison], DIVERGING, (-limit, limit))
    axs[j].set_title(("(a) " if j == 0 else "(b) ") + TITLES[comparison], pad=4)
    result = summary["comparisons"][comparison]
    lo, hi = result["channel_mean_delta_z_ci95"]
    digits = 2 if comparison == "within" else 3
    axs[j].text(0.5, -0.04, f"Mean over channels: {result['channel_mean_delta_z']:.{digits}f} ({lo:.{digits}f} to {hi:.{digits}f})",
                transform=axs[j].transAxes, ha="center", va="top", fontsize=7.5)
cax = fig.add_axes([0.885, 0.16, 0.022, 0.56])
bar = fig.colorbar(im, cax=cax)
bar.set_label("Fisher-scale difference ($\\Delta z$)\ndistributed $10\\times4$ s $-$ contiguous 40 s", fontsize=7.5)
bar.ax.tick_params(labelsize=7)
bar.outline.set_linewidth(0.5)
for ext in ("pdf", "png", "svg"):
    fig.savefig(out / f"figure6_topography.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)

# Figure S6: rho-tilde maps, one shared sequential scale.
levels = {(c, g): values(c, f"rho_{g}") for c in ("within", "between") for g in ("C", "F10")}
low = float(np.floor(min(v.min() for v in levels.values()) * 20) / 20)
fig, axs = plt.subplots(2, 2, figsize=(4.6, 4.7))
fig.subplots_adjust(left=0.12, right=0.84, bottom=0.03, top=0.9, wspace=0.06, hspace=0.18)
labels = {"C": "Contiguous 40 s", "F10": "Distributed $10\\times4$ s"}
for i, comparison in enumerate(("within", "between")):
    for j, geometry in enumerate(("C", "F10")):
        im = topomap(axs[i, j], levels[(comparison, geometry)], SEQUENTIAL, (low, 1.0))
        if i == 0:
            axs[i, j].set_title(labels[geometry], pad=4)
    axs[i, 0].text(-0.16, 0.5, TITLES[comparison], transform=axs[i, 0].transAxes,
                   rotation=90, ha="center", va="center", fontsize=7.5)
cax = fig.add_axes([0.87, 0.25, 0.025, 0.5])
bar = fig.colorbar(im, cax=cax)
bar.set_label("Rank reproducibility ($\\widetilde{\\rho}$)", fontsize=7.5)
bar.ax.tick_params(labelsize=7)
bar.outline.set_linewidth(0.5)
for ext in ("pdf", "png", "svg"):
    fig.savefig(out / f"figureS6_topography_levels.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("written", out, "delta limit", limit, "level floor", low)
