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
MIN_FRAC_HIT = 0.9          # same path-validity filter as the figure/web tool
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


def cell(p):
    """The five stacked values for one (psi, delta) parameter set."""
    n_alpha = p["N"] * p["alpha"]
    n_emtheta = p["N"] * math.exp(-p["theta"])
    theta = p["theta"]
    t0 = p["mean_t0"]
    logth = math.log(theta) if theta > 0 else None
    if logth is not None and logth < 0:
        y_str = "$<0$"                       # log(theta) < 0: report only "<0"
    elif t0 is not None and logth not in (None, 0):
        y_str = fmt(t0 * p["alpha"] / logth, ".3f")
    else:
        y_str = "--"
    rows = [fmt(n_alpha, ".4g"), fmt(n_emtheta, ".4g"), fmt(theta, ".3f"),
            fmt(t0, ".4g"), y_str]
    return r"\shortstack{" + r" \\ ".join(rows) + "}"


def main():
    data = load_data()
    pts = [p for p in data["points"]
           if p["N"] == N_SEL and p["selection_mode"] == MODE
           and p["n_used"] > 0 and p["frac_hit"] >= MIN_FRAC_HIT]

    psis = sorted({p["psi"] for p in pts})
    deltas = sorted({round(p["delta"], 10) for p in pts})
    index = {(p["psi"], round(p["delta"], 10)): p for p in pts}

    colspec = "l" + "c" * len(deltas)
    header = " & ".join([r"$\psi \backslash \delta$"]
                        + [f"${d:g}$" for d in deltas]) + r" \\"

    lines = []
    for psi in psis:
        cells = []
        for d in deltas:
            p = index.get((psi, d))
            cells.append(cell(p) if p else "--")
        lines.append(f"${psi:g}$ & " + " & ".join(cells) + r" \\")

    body = "\n\\addlinespace\n".join(lines)

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,landscape,margin=1.2cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\begin{{document}}

\begin{{table}}[t]
\centering
\caption{{Simulated values for population size $N={N_SEL}$, selection mode
\texttt{{{MODE}}} (no restriction on $\theta$, $\alpha$, $\psi$).
Rows: $\psi$; columns: $\delta$. Each cell lists, top to bottom:
$N\alpha$, $N\,e^{{-\theta}}$, $\theta$, $t_0$, $t_0\,\alpha/\log\theta$.
``--'' marks parameter sets with fewer than {int(MIN_FRAC_HIT*100)}\% clicking
paths.}}
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\scriptsize
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\end{{document}}
"""
    with open(OUT, "w") as fh:
        fh.write(tex)
    print(f"wrote {os.path.basename(OUT)}: {len(psis)} rows (psi) x "
          f"{len(deltas)} cols (delta)")


if __name__ == "__main__":
    main()
