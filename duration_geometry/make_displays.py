#!/usr/bin/env python3
"""Build the duration-by-geometry table and figure from results_summary.json.

Usage (from the archive root):
    python -B duration_geometry/make_displays.py --output outputs/duration_geometry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GREY = "#304d70", "#18847b", "#78828c"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "svg.fonttype": "none", "savefig.facecolor": "white", "svg.hashsalt": "duration-geometry",
                     "path.simplify": False})
NAMES = {"contiguous": "Contiguous", "distributed": "Distributed"}


def f(x, d=4):
    return f"{x:.{d}f}".replace("-", "$-$")


def interval(q, d=4):
    return f"[{f(q[0], d)}, {f(q[1], d)}]"


def contrast_label(text):
    a, b = text.split(" - ")

    def name(design):
        kind, n = design.split("_")
        return f"{NAMES[kind].lower()} {4 * int(n)} s"
    return f"{name(a).capitalize()} minus {name(b)}"


def table(summary) -> str:
    rows = []
    for d in summary["designs"]:
        rows.append(f"{NAMES[d['placement']]} {d['duration_s']} s & {f(d['rho_tilde'])} & {interval(d['rho_tilde_ci95'])} & {f(d['mean_raw_rho'])} & {f(d['mcse_mean_z'], 5)} \\\\")
    rows.append("\\midrule")
    rows.append("Comparison & $\\Delta z$ & 95\\% interval & Raw difference & MCSE \\\\")
    rows.append("\\midrule")
    family = None
    for c in summary["contrasts"]:
        if family is not None and c["family"] != family:
            rows.append("\\addlinespace")
        family = c["family"]
        rows.append(f"{contrast_label(c['contrast'])} & {f(c['delta_z'])} & {interval(c['delta_z_ci95'])} & {f(c['raw_difference'])} & {f(c['mcse_delta_z'], 5)} \\\\")
    body = "\n".join(rows)
    return ("\\begin{table}[H]\n\\centering\n"
            f"\\caption{{Duration by geometry: one pre-battery sample against the all-cell post-battery reference (N = {summary['n']}; exploratory)}}\n"
            "\\label{tab:duration_geometry}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n"
            "Design & $\\widetilde{\\rho}$ & 95\\% interval & Mean raw $\\rho$ & MCSE ($z$) \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
            "\\par\\smallskip\n\\begin{minipage}{\\textwidth}\\footnotesize "
            "Spearman correlation across participants per draw, 200 draws per design. "
            "$\\widetilde{\\rho}$ is the back-transformed mean Fisher $z$. "
            "Intervals: 5000 participant-bootstrap resamples shared by all rows, 2.5th and 97.5th percentiles, no multiplicity correction. "
            "MCSE: Monte Carlo standard error over draws. "
            "Raw difference: difference of mean raw correlations.\\end{minipage}\n\\end{table}\n")


def figure(summary, out: Path):
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    fig.subplots_adjust(left=.19, right=.97, bottom=.17, top=.97)
    positions = [0, 1, 2]
    for kind, color, marker, offset in [("contiguous", NAVY, "o", -.07), ("distributed", TEAL, "s", .07)]:
        rows = sorted([d for d in summary["designs"] if d["placement"] == kind], key=lambda d: d["cells"])
        y = [d["rho_tilde"] for d in rows]
        lo = [d["rho_tilde"] - d["rho_tilde_ci95"][0] for d in rows]
        hi = [d["rho_tilde_ci95"][1] - d["rho_tilde"] for d in rows]
        x = [p + offset for p in positions]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=marker + "-", color=color, ms=4.5, lw=1.2, capsize=2.5,
                    label=NAMES[kind])
    ax.set_xticks(positions, ["20 s", "40 s", "80 s"])
    ax.set_xlim(-.4, 2.4)
    ax.set_xlabel("Pre-battery sample duration")
    ax.set_ylabel("Agreement with post-battery\nreference ($\\widetilde{\\rho}$)")
    ax.grid(axis="y", color="#e3e6e9", lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for ext in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        if ext == "pdf":
            kwargs["metadata"] = {"CreationDate": None, "ModDate": None}
        if ext == "svg":
            kwargs["metadata"] = {"Date": None}
        fig.savefig(out / f"figure_duration_geometry.{ext}", bbox_inches="tight", pad_inches=.04, **kwargs)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=HERE / "results_summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = json.loads(args.results.read_text(encoding="utf-8"))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tex = table(summary)
    if "\u2014" in tex:
        raise SystemExit("long dash in table")
    (out / "table_duration_geometry.tex").write_text(tex, encoding="utf-8", newline="\n")
    figure(summary, out)
    print(f"wrote table and figure to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
