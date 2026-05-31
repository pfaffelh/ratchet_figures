#!/usr/bin/env python3
"""One Wright-Fisher Muller's-ratchet trajectory until the first click.

Same skeleton as the GPU simulator in Mullers_ratchet/Intuition/Imitate_Peters_Figure.py
(class-count vector of length C = max(5*log N, ...); per generation: mutation by
matrix; selection by fitness vector; multinomial resample), but
  * mutation is **Poisson(lambda)** per individual per generation
    (matrix entries M[k, k+m] = e^{-lambda} lambda^m / m!, tail lumped at the
    boundary), so it stays consistent for any lambda > 0;
  * selection is configurable: w_k = (1-alpha)^k (matches CSVs with
    selection_mode = "one_minus_alpha_power") or w_k = exp(-alpha k)
    (matches selection_mode = "exponential").

Parameter mapping used elsewhere in this repo:
    theta  = delta * ln N
    alpha  = psi * N^{delta - 1}
    lambda = theta * alpha
"""
import csv
import math
import os
import time

import numpy as np


# ----------- configuration ----------------------------------------------------
N = 10_000
DELTA = 0.9
PSI = 1.0
SELECTION_MODE = "exponential"          # "exponential" or "one_minus_alpha_power"
SEED = 42
T_MAX = 1_000_000

THETA = DELTA * math.log(N)
ALPHA = PSI * N ** (DELTA - 1.0)
LAM = THETA * ALPHA

# Number of classes: enough room for the Poisson tails plus mutation drift.
C_MIN_LOG = int(5 * math.log(N))
C_FROM_THETA = int(math.ceil(THETA + 6.0 * math.sqrt(THETA + LAM) + 6.0))
NUM_CLASSES = max(C_MIN_LOG, C_FROM_THETA)

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = f"wf_single_path_psi{PSI:g}_delta{DELTA:g}_N{N}_{SELECTION_MODE}"
TRAJ_CSV = os.path.join(HERE, STEM + "_trajectory.csv")
PNG = os.path.join(HERE, STEM + ".png")


def poisson_mutation_matrix(lam, C):
    """M[k, k+m] = e^{-lam} lam^m / m!, with the tail past the last class lumped
    into M[k, C-1] so each row sums to 1."""
    M = np.zeros((C, C))
    pmf = np.zeros(C)
    pmf[0] = math.exp(-lam)
    for m in range(1, C):
        pmf[m] = pmf[m - 1] * lam / m
    for k in range(C):
        avail = C - k
        M[k, k:k + avail] = pmf[:avail]
        # lump remaining probability mass at the last class
        M[k, C - 1] += max(0.0, 1.0 - M[k, :].sum())
    return M


def selection_vector(alpha, C, mode):
    k = np.arange(C, dtype=np.float64)
    if mode == "exponential":
        return np.exp(-alpha * k)
    if mode == "one_minus_alpha_power":
        return np.power(1.0 - alpha, k)
    raise ValueError(f"unknown selection_mode: {mode}")


def truncated_poisson_pmf(theta, C):
    pmf = np.zeros(C)
    pmf[0] = math.exp(-theta)
    for k in range(1, C):
        pmf[k] = pmf[k - 1] * theta / k
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


def main():
    print(f"Wright-Fisher Muller's ratchet, single path")
    print(f"  N = {N}, delta = {DELTA}, psi = {PSI}")
    print(f"  selection_mode = {SELECTION_MODE}")
    print(f"  theta  = {THETA:.6f}")
    print(f"  alpha  = {ALPHA:.6f}")
    print(f"  lambda = {LAM:.6f}  (Poisson rate of new mutations / gen / individual)")
    print(f"  num_classes C = {NUM_CLASSES}")
    print(f"  E[X_0(0)] = N * exp(-theta) = {N * math.exp(-THETA):.4f}")
    print(f"  seed = {SEED}, T_MAX = {T_MAX}")

    rng = np.random.default_rng(SEED)

    M = poisson_mutation_matrix(LAM, NUM_CLASSES)
    sel = selection_vector(ALPHA, NUM_CLASSES, SELECTION_MODE)
    p0 = truncated_poisson_pmf(THETA, NUM_CLASSES)

    # initial generation: sample N individuals from the truncated Poisson
    counts = rng.multinomial(N, p0).astype(np.int64)

    gens = [0]
    x0 = [int(counts[0])]
    click_gen = None

    t_start = time.time()
    if x0[0] == 0:
        click_gen = 0
        print("X_0(0) = 0: initial multinomial happened to have no class-0 individual.")
    else:
        # cast counts to float so the matmul works without copies
        for gen in range(1, T_MAX + 1):
            p = counts.astype(np.float64) @ M       # mutation first
            p = p * sel                              # then selection
            s = p.sum()
            if not (s > 0):
                print(f"degenerate p at gen {gen}; stopping.")
                break
            p /= s
            counts = rng.multinomial(N, p).astype(np.int64)
            c0 = int(counts[0])
            gens.append(gen)
            x0.append(c0)
            if c0 == 0:
                click_gen = gen
                print(f"first click at generation t_0 = {gen}  "
                      f"(walltime {time.time() - t_start:.1f}s)")
                break
        else:
            print(f"no click within T_MAX = {T_MAX} generations "
                  f"(walltime {time.time() - t_start:.1f}s)")

    # CSV
    with open(TRAJ_CSV, "w", newline="") as fh:
        w_csv = csv.writer(fh)
        w_csv.writerow(["generation", "X_0_count", "X_0_frequency"])
        for t, c in zip(gens, x0):
            w_csv.writerow([t, c, f"{c / N:.6e}"])
    print(f"wrote {TRAJ_CSV} ({len(gens)} rows)")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.step(gens, x0, where="post", lw=0.9, color="#1f77b4")
    eq = N * math.exp(-THETA)
    ax.axhline(eq, color="gray", lw=0.8, ls="--",
               label=fr"$N\,e^{{-\theta}}\approx{eq:.2f}$")
    ax.axhline(0.0, color="black", lw=0.5)
    if click_gen is not None:
        ax.axvline(click_gen, color="#d62728", lw=0.8, ls=":",
                   label=fr"first click at $t_0={click_gen}$ gen")
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$X_0(t)$ (count of class 0)")
    ax.set_title(
        fr"Wright-Fisher Muller's ratchet, single path:  "
        fr"$\delta={DELTA},\ \psi={PSI},\ N={N}$  ({SELECTION_MODE})" "\n"
        fr"$\alpha={ALPHA:.4f},\ \lambda={LAM:.4f},\ \theta={THETA:.4f}$"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-0.3)
    fig.tight_layout()
    fig.savefig(PNG, dpi=140)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
