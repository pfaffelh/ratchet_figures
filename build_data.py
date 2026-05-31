#!/usr/bin/env python3
"""Build data.js for the interactive ratchet figures.

Reads the per-path `summary_N_*_psi_*.csv` files, groups by (psi, delta) and
stores per-group mean/std of t_0 and 1/t_0 (computed over finite, strictly
positive t_0 only). All y-axis quantities the page offers are scalar multiples
of mean(t_0) or mean(1/t_0), so storing these four numbers is enough to render
every "transform-then-average" curve exactly.

Run again whenever the CSVs change:  python3 build_data.py
"""
import csv
import glob
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
# Data sources; each CSV's N is read from the file itself (not the filename),
# so we group robustly by (N, psi, delta) regardless of how the folder is named.
DATA_DIRS = [
    os.path.join(HERE, "ratchet_discrete_psi_delta_stop_at_t0_results"),
    os.path.join(HERE, "ratchet_discrete_psi_delta_stop_at_t0_results_large_N_different_psi"),
    os.path.join(HERE, "ratchet_discrete_psi_approx_delta_stop_at_t0_results_large_N_different_psi"),
]
OUT = os.path.join(HERE, "data.js")


def fnum(s):
    """Parse a CSV cell to a finite float, else None (empty / inf / NaN)."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


# -------- Neutral-diffusion theory: E_{x0}[tau_0] for dX = -lam X dt + sqrt(X(1-X)) dW.
# See neutral.tex for the derivation.

def _S_value(y, two_lam):
    """Scale function S(y) = int_0^y (1-w)^{-2 lam} dw."""
    if abs(two_lam - 1.0) < 1e-12:
        return -math.log1p(-y)
    return (1.0 - (1.0 - y) ** (1.0 - two_lam)) / (1.0 - two_lam)


def _Sm(y, two_alpha, m_scale):
    """Integrand S(y) * m(y), with m(y) = m_scale * (1-y)^{two_alpha - 1} / y.
    Removable limit at y = 0 is m_scale (since S(y) ~ y there)."""
    if y < 1e-15:
        return m_scale
    return _S_value(y, two_alpha) * m_scale * (1.0 - y) ** (two_alpha - 1.0) / y


def _I2_integrand(t, inv_two_alpha):
    """Integrand 1/(1 - t^{1/(2 N lam)}) of the substituted second integral."""
    if t <= 0.0:
        return 1.0
    return 1.0 / (1.0 - t ** inv_two_alpha)


def _simpson(f, a, b, n):
    """Composite Simpson on [a, b] with n (even) subintervals."""
    if n < 2:
        n = 2
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return s * h / 3.0


def t0_neutral_diffusion(lam, x0, N=1, n=4000):
    """Mean hitting time of 0 for dX = -lam X dt + sqrt(X(1-X)/N) dW, starting at x0.

    Uses the closed-form (see neutral.tex)
        u(x) = int_0^x S(y) m(y) dy + S(x) * int_x^1 m(y) dy,
    with S' = (1-x)^{-2 N lam}, m(y) = 2N (1-y)^{2 N lam - 1}/y, and the
    substitution t = (1-y)^{2 N lam} for the second integral.
    """
    if lam is None or lam <= 0 or N is None or N <= 0 or not (0.0 < x0 < 1.0):
        return None
    a = 2.0 * N * lam            # exponent in S and m
    m_scale = 2.0 * N            # m(y) = m_scale * (1-y)^{a-1}/y
    I1 = _simpson(lambda y: _Sm(y, a, m_scale), 0.0, x0, n)
    T = (1.0 - x0) ** a
    if T <= 0.0:
        I2 = 0.0
    else:
        inv_a = 1.0 / a
        # int_x^1 m dy = (1/lam) * int_0^T dt/(1 - t^{1/a})
        I2 = _simpson(lambda t: _I2_integrand(t, inv_a), 0.0, T, n) / lam
    return I1 + _S_value(x0, a) * I2


groups = {}
for data_dir in DATA_DIRS:
    if not os.path.isdir(data_dir):
        continue
    for path in sorted(glob.glob(os.path.join(data_dir, "summary_N_*_psi_*.csv"))):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                sel_mode = row["selection_mode"].strip()
                key = (int(float(row["N"])), float(row["psi"]), float(row["delta"]), sel_mode)
                g = groups.get(key)
                if g is None:
                    g = groups[key] = {
                        "N": int(float(row["N"])),
                        "psi": float(row["psi"]),
                        "delta": float(row["delta"]),
                        "selection_mode": sel_mode,
                        "alpha": float(row["alpha"]),
                        "lambda": float(row["lambda"]),
                        "theta": float(row["theta"]),
                        "effective_beta": float(row["effective_beta"]),
                        "n_total": 0,
                        "n_hit": 0,
                        "t0": [],
                    }
                g["n_total"] += 1
                if row["hit_x0_zero"].strip() == "True":
                    g["n_hit"] += 1
                t0 = fnum(row.get("t_0"))
                if t0 is not None and t0 > 0:
                    g["t0"].append(t0)

points = []
for key in sorted(groups):
    g = groups[key]
    t0 = g["t0"]
    n_used = len(t0)
    if n_used >= 1:
        inv = [1.0 / x for x in t0]
        mean_t0 = statistics.fmean(t0)
        mean_inv = statistics.fmean(inv)
        std_t0 = statistics.stdev(t0) if n_used >= 2 else 0.0
        std_inv = statistics.stdev(inv) if n_used >= 2 else 0.0
    else:
        mean_t0 = std_t0 = mean_inv = std_inv = None
    points.append({
        "psi": g["psi"],
        "delta": g["delta"],
        "N": g["N"],
        "selection_mode": g["selection_mode"],
        "alpha": g["alpha"],
        "lambda": g["lambda"],
        "theta": g["theta"],
        "effective_beta": g["effective_beta"],
        "n_total": g["n_total"],
        "n_hit": g["n_hit"],
        "frac_hit": g["n_hit"] / g["n_total"] if g["n_total"] else 0.0,
        "n_used": n_used,
        "mean_t0": mean_t0,
        "std_t0": std_t0,
        "mean_inv_t0": mean_inv,
        "std_inv_t0": std_inv,
        # neutral-diffusion theory baseline: E_{x0=e^-theta}[tau_0]
        # for dX = -lambda X dt + sqrt(X(1-X)/N) dW
        "t0_neutral": t0_neutral_diffusion(g["lambda"], math.exp(-g["theta"]), N=g["N"]),
    })

out = {
    "N_values": sorted({p["N"] for p in points}),
    "psi_values": sorted({p["psi"] for p in points}),
    "selection_modes": sorted({p["selection_mode"] for p in points}),
    "points": points,
}
with open(OUT, "w") as fh:
    fh.write("// auto-generated by build_data.py - do not edit by hand\n")
    fh.write("const RATCHET_DATA = ")
    json.dump(out, fh, indent=1, allow_nan=False)
    fh.write(";\n")

print(f"wrote {OUT}: {len(points)} (psi,delta) points; "
      f"psi={out['psi_values']}; N={out['N_values']}")
