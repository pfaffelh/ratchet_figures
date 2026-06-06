#!/usr/bin/env python3
"""Render a static figure that mirrors the interactive page (index.html).

For one fixed configuration this reproduces the same filter / transform pipeline
the web tool uses, but shows the *full per-path distribution* of
y = alpha * t_0 / log(theta) as one violin per (psi, delta) point (instead of
mean +/- SD error bars). Those per-path samples are not in data.js (which only
stores mean/std), so we read the raw summary_*.csv files directly and apply the
same grouping/filtering as build_data.py + index.html.

x: N*alpha (log)   y: alpha*t_0/log(theta) (linear)

Fonts are enlarged so the figure stays legible at half the width of an A4 page.
"""
import csv
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root; generated figures live here
DATA_DIRS = [
    HERE,
]

# ----- selection (matches the requested on-screen state) ---------------------
N_SEL = 10000
MODE = "one_minus_alpha_power"
PSI_MAX = 2.0            # psi <= 2
THETA_MIN = 1.0         # theta > THETA_MIN
ALPHA_MAX = 0.2         # alpha < ALPHA_MAX
MIN_FRAC_HIT = 1.0      # keep only points where 100% of paths clicked
OUT_BASENAME = "figure_Nalpha_alphat0_logtheta_N10000"

# Plotly PALETTE, indexed by position of psi in sorted psi_values
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
           "#ff7f0e", "#17becf", "#8c564b", "#e377c2"]


def fnum(s):
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def load_groups():
    """Group raw paths by (N, psi, delta, mode) -> dict with params + t0 list."""
    groups = {}
    for d in DATA_DIRS:
        if not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "summary_N_*_psi_*.csv"))):
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    mode = row["selection_mode"].strip()
                    key = (int(float(row["N"])), float(row["psi"]),
                           float(row["delta"]), mode)
                    g = groups.get(key)
                    if g is None:
                        g = groups[key] = {
                            "N": int(float(row["N"])), "psi": float(row["psi"]),
                            "delta": float(row["delta"]), "mode": mode,
                            "alpha": float(row["alpha"]), "theta": float(row["theta"]),
                            "n_total": 0, "n_hit": 0, "t0": [],
                        }
                    g["n_total"] += 1
                    if row["hit_x0_zero"].strip() == "True":
                        g["n_hit"] += 1
                    t0 = fnum(row.get("t_0"))
                    if t0 is not None and t0 > 0:
                        g["t0"].append(t0)
    return list(groups.values())


def main():
    groups = load_groups()
    psi_values = sorted({g["psi"] for g in groups})
    psi_color = {psi: PALETTE[i % len(PALETTE)] for i, psi in enumerate(psi_values)}

    def keep(g):
        return (g["N"] == N_SEL and g["mode"] == MODE and g["psi"] <= PSI_MAX
                and g["theta"] > THETA_MIN and g["alpha"] < ALPHA_MAX
                and g["n_total"] and g["n_hit"] / g["n_total"] >= MIN_FRAC_HIT
                and len(g["t0"]) > 0)

    sel = [g for g in groups if keep(g)]
    psis = sorted({g["psi"] for g in sel})

    plt.rcParams.update({
        "font.size": 15, "axes.labelsize": 20,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 15,
    })
    fig, ax = plt.subplots(figsize=(6.4 * 1.3 * 1.3, 5.0 * 1.3))

    # one violin per (psi, delta) point, positioned at log10(N*alpha);
    # collect per-psi means to connect them with a line
    for psi in psis:
        gs = sorted((g for g in sel if g["psi"] == psi),
                    key=lambda g: g["N"] * g["alpha"])
        mean_pts = []
        for g in gs:
            ys = [g["alpha"] * t0 / math.log(g["theta"]) for t0 in g["t0"]]
            pos = math.log10(g["N"] * g["alpha"])
            mean_pts.append((pos, sum(ys) / len(ys)))
            parts = ax.violinplot([ys], positions=[pos], widths=0.06,
                                  showmeans=True, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(psi_color[psi])
                body.set_edgecolor(psi_color[psi])
                body.set_alpha(0.25)
            if "cmeans" in parts:
                parts["cmeans"].set_color(psi_color[psi])
                parts["cmeans"].set_linewidth(2)
        # connect the means of this colour
        if mean_pts:
            ax.plot([x for x, _ in mean_pts], [y for _, y in mean_pts],
                    color=psi_color[psi], lw=2, zorder=5)

    # theory curve (dashed): alpha*(log theta/alpha)/log theta == 1 for all points
    xs_th = sorted(math.log10(g["N"] * g["alpha"]) for g in sel)
    if xs_th:
        ax.plot([xs_th[0], xs_th[-1]], [1, 1], ls="--", lw=2, color="0.4")

    ax.set_xlabel(r"$N\,\alpha$")
    ax.set_ylabel(r"$\alpha\, t_0 / \log\theta$")
    ax.set_ylim(0, 15)   # clip the long upper tails

    # log10 positions -> power-of-ten tick labels; tighten x to the data
    lo = math.floor(min(xs_th)); hi = math.floor(max(xs_th))
    ticks = list(range(lo, hi + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([rf"$10^{{{t}}}$" for t in ticks])
    ax.set_xlim(min(xs_th) - 0.15, max(xs_th) + 0.15)  # last: not widened by ticks

    handles = [Line2D([0], [0], color=psi_color[psi], lw=8, alpha=0.25)
               for psi in psis]
    labels = [f"ψ={psi:g}" for psi in psis]
    handles.append(Line2D([0], [0], color="0.4", lw=2, ls="--"))
    labels.append(r"$\frac{\log\theta}{\alpha}$")
    ax.legend(handles, labels, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(labels))

    pdf = os.path.join(ROOT, OUT_BASENAME + ".pdf")
    png = os.path.join(ROOT, OUT_BASENAME + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print("wrote:", os.path.basename(pdf), "and", os.path.basename(png))


if __name__ == "__main__":
    main()
