#!/usr/bin/env python3
"""Render a static figure that mirrors the interactive page (index.html).

For one fixed configuration this reproduces the same filter / transform pipeline
the web tool uses, but shows the *full per-path distribution* of
y = s * t_0 / log(u_div_s) as one violin per (psi, delta) point (instead of
mean +/- SD error bars). Those per-path samples are not in data.js (which only
stores mean/std), so we read the raw summary_*.csv files directly and apply the
same grouping/filtering as build_data.py + index.html.

This is the second figure: only psi in {2, 5, 10} and a LOGARITHMIC y-axis.

x: N*s (log)   y: s*t_0/log(u_div_s) (log)

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

HERE = os.path.dirname(os.path.abspath(__file__))  # data and generated figures live here
DATA_DIRS = [
    HERE,
]

# ----- selection (second figure: only psi 2/5/10, log y-axis) ----------------
N_SEL = 10000
MODE = "one_minus_s_power"
PSI_SET = {2.0, 5.0, 10.0}   # show only these psi
U_DIV_S_MIN = 1.0         # u_div_s > U_DIV_S_MIN
S_MAX = 0.2         # s < S_MAX
MIN_FRAC_HIT = 1.0      # keep only points where 100% of paths clicked
OUT_BASENAME = "figure_Ns_st0_logu_div_s_N10000_psi2_5_10_logy"

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
                            "s": float(row["s"]), "u_div_s": float(row["u_div_s"]),
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
        return (g["N"] == N_SEL and g["mode"] == MODE and g["psi"] in PSI_SET
                and g["u_div_s"] > U_DIV_S_MIN and g["s"] < S_MAX
                and g["n_total"] and g["n_hit"] / g["n_total"] >= MIN_FRAC_HIT
                and len(g["t0"]) > 0)

    sel = [g for g in groups if keep(g)]
    psis = sorted({g["psi"] for g in sel})

    plt.rcParams.update({
        "font.size": 15, "axes.labelsize": 26,
        "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 19,
    })
    fig, ax = plt.subplots(figsize=(6.4 * 1.3 * 1.3, 5.0 * 1.3))

    # one violin per (psi, delta) point, positioned at log10(N*s).
    # The y-axis is logarithmic: to get an UNDISTORTED violin we run the KDE on
    # log10(y) and draw on a linear axis (relabelled as powers of ten), rather
    # than computing the density in linear space and stretching it onto a log
    # axis. The connecting line marks the (arithmetic) mean, at log10(mean).
    all_logy = []
    for psi in psis:
        gs = sorted((g for g in sel if g["psi"] == psi),
                    key=lambda g: g["N"] * g["s"])
        mean_pts = []
        for g in gs:
            ys = [g["s"] * t0 / math.log(g["u_div_s"]) for t0 in g["t0"]]
            logys = [math.log10(y) for y in ys if y > 0]
            all_logy.extend(logys)
            pos = math.log10(g["N"] * g["s"])
            mean_pts.append((pos, math.log10(sum(ys) / len(ys))))
            parts = ax.violinplot([logys], positions=[pos], widths=0.06,
                                  showmeans=False, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(psi_color[psi])
                body.set_edgecolor(psi_color[psi])
                body.set_alpha(0.25)
        # connect the means of this colour (in log space)
        if mean_pts:
            ax.plot([x for x, _ in mean_pts], [y for _, y in mean_pts],
                    color=psi_color[psi], lw=2, zorder=5)

    # theory (dashed): s*(log u_div_s/s)/log u_div_s == 1, i.e. log10 == 0
    xs_th = sorted(math.log10(g["N"] * g["s"]) for g in sel)
    if xs_th:
        ax.plot([xs_th[0], xs_th[-1]], [0, 0], ls="--", lw=2, color="0.4")

    ax.set_xlabel(r"$N\,s$")
    ax.set_ylabel(r"$s\, t_0 / \log(u/s)$")

    # y is plotted in log10 units on a linear axis -> 10^k tick labels;
    # keep the theory line (log10 = 0) in view
    ylo = math.floor(min(min(all_logy), 0.0))
    yhi = math.ceil(max(all_logy))
    ax.set_yticks(list(range(ylo, yhi + 1)))
    ax.set_yticklabels([rf"$10^{{{k}}}$" for k in range(ylo, yhi + 1)])
    ax.set_ylim(min(all_logy) - 0.2, max(all_logy) + 0.2)

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
    labels.append(r"$\frac{\log(u/s)}{s}$")
    ax.legend(handles, labels, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(labels))

    pdf = os.path.join(HERE, OUT_BASENAME + ".pdf")
    png = os.path.join(HERE, OUT_BASENAME + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    print("wrote:", os.path.basename(pdf), "and", os.path.basename(png))


if __name__ == "__main__":
    main()
