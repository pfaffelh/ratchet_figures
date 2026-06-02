#!/usr/bin/env python3
"""Build a LaTeX table of the simulated values (mirrors index.html quantities).

Same source values as the figure (N fixed, one selection mode), but with NO
restriction on theta, alpha or psi. Layout:
  - one row per psi
  - one column per beta = 1 - delta
  - each cell stacks, in order:  N*alpha, N*lambda, theta, t0, t0*alpha/log(theta)

t0 is the per-group mean first-loss time (mean over finite, >0 paths), exactly
as stored by build_data.py. Reads data.js; writes a standalone .tex document.
Run:  python3 make_table.py
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

N_SEL = 10000
MODE = "exponential"
MIN_FRAC_HIT = 1.0          # show a cell only if 100% of paths clicked
OUT = os.path.join(HERE, "table_N10000_exponential.tex")


def load_data():
    with open(os.path.join(HERE, "data.js")) as fh:
        src = fh.read()
    src = re.sub(r"^//.*\n", "", src)
    src = src.replace("const RATCHET_DATA = ", "").rstrip().rstrip(";")
    return json.loads(src)


def fmt(v, spec):
    if v is None or not math.isfinite(v):
        return "--"
    return format(v, spec)


# the five quantities, in row order; label is the entry in the label column
QUANTITIES = [
    (r"$N\alpha$",                 lambda p: fmt(p["N"] * p["alpha"], ".4g")),
    (r"$N\lambda$",                lambda p: fmt(p["N"] * p["lambda"], ".4g")),
    (r"$N\,e^{-\theta}$",          lambda p: fmt(p["N"] * math.exp(-p["theta"]), ".4g")),
    (r"$t_0$",                     lambda p: fmt(p["mean_t0"], ".4g")),
    (r"$t_0\,\alpha/\log\theta$",  lambda p: _y_logtheta(p)),
]
PSI_ROW = 2   # 0-based: put psi on the 3rd of the five rows (height of N e^{-theta})


def _y_logtheta(p):
    theta = p["theta"]
    logth = math.log(theta) if theta > 0 else None
    if logth is not None and logth < 0:
        return "$<0$"                        # log(theta) < 0: report only "<0"
    if p["mean_t0"] is not None and logth not in (None, 0):
        return fmt(p["mean_t0"] * p["alpha"] / logth, ".3f")
    return "--"


def main():
    data = load_data()
    pts = [p for p in data["points"]
           if p["N"] == N_SEL and p["selection_mode"] == MODE
           and p["n_used"] > 0 and p["frac_hit"] >= MIN_FRAC_HIT]

    psis = sorted({p["psi"] for p in pts})
    deltas = sorted({round(p["delta"], 10) for p in pts})
    index = {(p["psi"], round(p["delta"], 10)): p for p in pts}

    colspec = "c l" + "c" * len(deltas)
    header = " & ".join([r"$\psi \backslash \delta$", ""]
                        + [f"${d:g}$" for d in deltas]) + r" \\"

    # each psi is a block of five real rows (one per quantity); psi sits on the
    # 3rd row so it aligns exactly with the 3rd entry (N e^{-theta}). Blocks are
    # separated by a horizontal rule.
    blocks = []
    for psi in psis:
        rows = []
        for q, (label, valfn) in enumerate(QUANTITIES):
            psi_col = f"${psi:g}$" if q == PSI_ROW else ""
            cells = [valfn(index[(psi, d)]) if (psi, d) in index else "--"
                     for d in deltas]
            rows.append(f"{psi_col} & {label} & " + " & ".join(cells) + r" \\")
        blocks.append("\n".join(rows))

    body = "\n\\midrule\n".join(blocks)

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.5cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage[export]{{adjustbox}}
\begin{{document}}

\begin{{table}}[t]
\centering
\caption{{Simulated values for population size $N={N_SEL}$, selection mode
\texttt{{{MODE}}} (no restriction on $\theta$, $\alpha$, $\psi$).
Rows: $\psi$ (each a block of five rows, one per quantity named in the label
column); columns: $\delta$. ``--'' marks parameter sets with fewer than
{int(MIN_FRAC_HIT*100)}\% clicking paths.}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{adjustbox}}{{max width=\textwidth, max totalheight=0.9\textheight, center}}
\scriptsize
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{adjustbox}}
\end{{table}}

\end{{document}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.basename(OUT)}: {len(psis)} rows (psi) x "
          f"{len(deltas)} cols (delta)")


if __name__ == "__main__":
    main()
