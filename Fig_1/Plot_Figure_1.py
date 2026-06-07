import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
# Input timeseries/metadata live alongside this script in Fig_1/.
INPUT_DIR = HERE
OUTPUT_DIR = os.path.join(HERE, "theory_comparison_one_minus_s_power_until_2000")

SELECTION_MODE = "one_minus_s_power"

TIME_MAX_PLOT = 2000

DEFAULT_N = 10000
DEFAULT_U = 0.04
DEFAULT_S = 0.01

CHUNKSIZE = 1_000_000
THEORY_SUBSTEPS_PER_GENERATION = 5

# Step size in original time units for the numerical second-order integral.
# Smaller values are more accurate but slower.
SECOND_ORDER_TIME_STEP = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Large plot style
# ============================================================

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.labelsize": 26,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 19,
        "axes.linewidth": 1.8,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
    }
)


# ============================================================
# Colors
# ============================================================

COLOR_M1_SIM = "#084594"
COLOR_M1_THEORY_1 = "#9ecae1"
COLOR_M1_THEORY_2 = "#4292c6"

COLOR_VAR_SIM = "#54278f"
COLOR_VAR_THEORY = "#cbc9e2"

COLOR_COV_SIM = "#006d2c"
COLOR_COV_THEORY = "#a1d99b"

X_COLORS = {
    0: {"sim": "#08519c", "theory": "#bdd7e7"},
    1: {"sim": "#238b45", "theory": "#c7e9c0"},
    2: {"sim": "#a50f15", "theory": "#fcbba1"},
    3: {"sim": "#54278f", "theory": "#cbc9e2"},
}


# ============================================================
# File helpers
# ============================================================

def find_metadata_file():
    paths = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, f"metadata_*selection_{SELECTION_MODE}.csv")
        )
    )

    if not paths:
        return None

    return paths[0]


def find_means_file():
    paths = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, f"timeseries_means_*selection_{SELECTION_MODE}.csv")
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No means file found for selection mode {SELECTION_MODE} in {INPUT_DIR}"
        )

    return paths[0]


def find_by_run_file():
    paths = sorted(
        glob.glob(
            os.path.join(INPUT_DIR, f"timeseries_by_run_*selection_{SELECTION_MODE}.csv")
        )
    )

    if not paths:
        return None

    return paths[0]


def load_parameters():
    metadata_path = find_metadata_file()

    if metadata_path is None:
        print("No metadata file found. Using fallback values.")

        N = DEFAULT_N
        u = DEFAULT_U
        s = DEFAULT_S
        u_div_s = u / s

        return {
            "N": N,
            "u": u,
            "s": s,
            "u_div_s": u_div_s,
        }

    metadata = pd.read_csv(metadata_path).iloc[0]

    return {
        "N": int(metadata["N"]),
        "u": float(metadata["u"]),
        "s": float(metadata["s"]),
        "u_div_s": float(metadata["u_div_s"]),
    }


# ============================================================
# First-order theory formulas
# ============================================================

def cumulative_trapezoid(y, x):
    out = np.zeros_like(y, dtype=float)

    dx = np.diff(x)
    avg = 0.5 * (y[:-1] + y[1:])

    out[1:] = np.cumsum(dx * avg)

    return out


def make_cumulative_integral_grid(times, s):
    times = np.asarray(times, dtype=float)

    u_max = s * float(np.max(times))
    du = s / float(THEORY_SUBSTEPS_PER_GENERATION)

    n_grid = int(math.ceil(u_max / du)) + 1
    u = np.linspace(0.0, u_max, n_grid)

    return u


def theory_m1_first_order(times, N, s, u_div_s):
    times = np.asarray(times, dtype=float)

    q = 1.0 - np.exp(-s * times)

    return u_div_s + (
        np.exp(u_div_s * q**2) - 1.0
    ) / (2.0 * N * s)


def theory_x0(times, N, s, u_div_s):
    times = np.asarray(times, dtype=float)

    u = make_cumulative_integral_grid(times, s)
    exp_minus_u = np.exp(-u)

    integrand = (
        np.exp(-u_div_s * exp_minus_u)
        * (
            1.0
            - np.exp(-u_div_s * exp_minus_u * (1.0 - exp_minus_u))
        )
    )

    cumulative = cumulative_trapezoid(integrand, u)
    integral_values = np.interp(s * times, u, cumulative)

    return math.exp(-u_div_s) - integral_values / (N * s)


def theory_xk(times, k, N, s, u_div_s):
    if k == 0:
        return theory_x0(times, N, s, u_div_s)

    times = np.asarray(times, dtype=float)

    u = make_cumulative_integral_grid(times, s)
    exp_minus_u = np.exp(-u)

    factorial_k = math.factorial(k)

    poisson_initial = math.exp(-u_div_s) * u_div_s**k / factorial_k

    first_term = (
        np.exp(-u_div_s * exp_minus_u)
        * np.exp(-u_div_s * exp_minus_u * (1.0 - exp_minus_u))
        * u_div_s**k
        / factorial_k
    )

    second_term = (
        np.exp(-u_div_s * exp_minus_u)
        * u_div_s**k
        * (1.0 - exp_minus_u * (1.0 - exp_minus_u))**k
        / factorial_k
    )

    integrand = first_term - second_term

    cumulative = cumulative_trapezoid(integrand, u)
    integral_values = np.interp(s * times, u, cumulative)

    return poisson_initial + integral_values / (N * s)


def theory_variance_m1(times, N, s, u_div_s):
    times = np.asarray(times, dtype=float)

    u = make_cumulative_integral_grid(times, s)
    exp_minus_u = np.exp(-u)

    integrand = (
        u_div_s
        * np.exp(-2.0 * u)
        * (1.0 + u_div_s * (1.0 - exp_minus_u) ** 2)
        * np.exp(u_div_s * (1.0 - exp_minus_u) ** 2)
    )

    cumulative = cumulative_trapezoid(integrand, u)
    integral_values = np.interp(s * times, u, cumulative)

    return integral_values / (N * s)


def theory_cov_m1_x0(times, N, s, u_div_s):
    times = np.asarray(times, dtype=float)

    q = 1.0 - np.exp(-s * times)

    return (
        math.exp(-u_div_s)
        / (2.0 * N * s)
        * (
            np.exp(u_div_s * q**2)
            - 2.0 * np.exp(u_div_s * q)
            + 1.0
        )
    )


# ============================================================
# Second-order theory for m_1
# ============================================================

def H_z(z, u_div_s):
    return np.exp(-u_div_s * (1.0 - np.exp(-z)))


def K_z(z, u_div_s):
    return u_div_s * np.exp(-z) * H_z(z, u_div_s)


def C_HH_z(r, s, u_div_s):
    return H_z(r + s, u_div_s) - H_z(r, u_div_s) * H_z(s, u_div_s)


def C_HK_z(r, s, u_div_s):
    return K_z(r + s, u_div_s) - H_z(r, u_div_s) * K_z(s, u_div_s)


def D_kappa_uv(u, v, u_div_s):
    # u = s * a, v = s * b

    U = H_z(v, u_div_s)
    V = H_z(u + v, u_div_s)
    W = H_z(2.0 * u + v, u_div_s)

    P = K_z(u + v, u_div_s)
    Q = K_z(2.0 * u + v, u_div_s)

    exp_minus_u = np.exp(-u)
    exp_minus_v = np.exp(-v)

    A = u_div_s * (1.0 - exp_minus_v) * (exp_minus_u - np.exp(-2.0 * u))

    M = np.exp(u_div_s * (1.0 - exp_minus_u) ** 2 * (1.0 - exp_minus_v))

    term = (
        ((-2.0 * A * V * W - 3.0 * P * W + 2.0 * Q * V) / V**4)
        * C_HH_z(v, u + v, u_div_s)
    )

    term += (
        ((A * V + P) / V**3)
        * C_HH_z(v, 2.0 * u + v, u_div_s)
    )

    term += (
        (W / V**3)
        * C_HK_z(v, u + v, u_div_s)
    )

    term += (
        (-1.0 / V**2)
        * C_HK_z(v, 2.0 * u + v, u_div_s)
    )

    term += (
        (3.0 * U * (A * V * W + 2.0 * P * W - Q * V) / V**5)
        * C_HH_z(u + v, u + v, u_div_s)
    )

    term += (
        (U * (-2.0 * A * V - 3.0 * P) / V**4)
        * C_HH_z(u + v, 2.0 * u + v, u_div_s)
    )

    term += (
        (-3.0 * U * W / V**4)
        * C_HK_z(u + v, u + v, u_div_s)
    )

    term += (
        (2.0 * U / V**3)
        * C_HK_z(u + v, 2.0 * u + v, u_div_s)
    )

    term += (
        (U / V**3)
        * C_HK_z(2.0 * u + v, u + v, u_div_s)
    )

    return M * term


def second_order_m1_integral(times, s, u_div_s):
    times = np.asarray(times, dtype=float)

    u_max = s * float(np.max(times))
    du = s * SECOND_ORDER_TIME_STEP

    if du <= 0:
        raise ValueError("s and SECOND_ORDER_TIME_STEP must be positive.")

    n_grid = int(math.ceil(u_max / du))
    u_grid = np.arange(n_grid + 1, dtype=float) * du

    diag_sums = np.zeros(n_grid + 1, dtype=np.float64)

    print(
        f"Computing second-order m1 integral on {n_grid + 1} grid points "
        f"with du={du:.6g}."
    )

    for i, u in enumerate(u_grid):
        m = n_grid - i + 1

        v_values = u_grid[:m]
        d_values = D_kappa_uv(u, v_values, u_div_s)

        diag_indices = i + np.arange(m)

        np.add.at(diag_sums, diag_indices, d_values)

    # Rectangular quadrature in u,v.
    # Since u = s*a and v = s*b, da db = du dv / s^2.
    triangle_integral_grid = np.cumsum(diag_sums) * du * du / (s**2)

    target_u = s * times

    return np.interp(target_u, u_grid, triangle_integral_grid)


def theory_m1_second_order(times, N, s, u_div_s):
    first_order = theory_m1_first_order(times, N, s, u_div_s)
    integral_second_order = second_order_m1_integral(
        times=times,
        s=s,
        u_div_s=u_div_s,
    )

    return first_order + integral_second_order / (N**2)


def build_theory_dataframe(times, N, s, u_div_s):
    return pd.DataFrame(
        {
            "time": times,
            "theory_kappa_1_mean_first_order": theory_m1_first_order(
                times,
                N,
                s,
                u_div_s,
            ),
            "theory_kappa_1_mean_second_order": theory_m1_second_order(
                times,
                N,
                s,
                u_div_s,
            ),
            "theory_X_0_mean": theory_xk(times, 0, N, s, u_div_s),
            "theory_X_1_mean": theory_xk(times, 1, N, s, u_div_s),
            "theory_X_2_mean": theory_xk(times, 2, N, s, u_div_s),
            "theory_X_3_mean": theory_xk(times, 3, N, s, u_div_s),
            "theory_kappa_1_var": theory_variance_m1(times, N, s, u_div_s),
            "theory_cov_kappa_1_X_0": theory_cov_m1_x0(times, N, s, u_div_s),
        }
    )


# ============================================================
# First loss times and variance/covariance from by-run data
# ============================================================

def aggregate_by_run_file(path, times):
    print(f"Aggregating by-run file in chunks up to time {TIME_MAX_PLOT}: {path}")

    max_time = int(np.max(times))
    size = max_time + 1

    count = np.zeros(size, dtype=np.float64)

    sum_kappa = np.zeros(size, dtype=np.float64)
    sum_kappa_sq = np.zeros(size, dtype=np.float64)

    sum_x0 = np.zeros(size, dtype=np.float64)
    sum_x0_sq = np.zeros(size, dtype=np.float64)

    sum_kappa_x0 = np.zeros(size, dtype=np.float64)

    first_loss_by_run = {}

    # collect (time, kappa_1) over all runs to get per-time percentiles of m_1
    t_chunks = []
    kappa_chunks = []

    usecols = ["run_index", "time", "X_0", "kappa_1"]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE):
        chunk = chunk[chunk["time"] <= max_time]

        if chunk.empty:
            continue

        t = chunk["time"].to_numpy(dtype=np.int64)
        x0 = chunk["X_0"].to_numpy(dtype=np.float64)
        kappa = chunk["kappa_1"].to_numpy(dtype=np.float64)

        t_chunks.append(t)
        kappa_chunks.append(kappa)

        count += np.bincount(t, minlength=size)
        sum_kappa += np.bincount(t, weights=kappa, minlength=size)
        sum_kappa_sq += np.bincount(t, weights=kappa**2, minlength=size)

        sum_x0 += np.bincount(t, weights=x0, minlength=size)
        sum_x0_sq += np.bincount(t, weights=x0**2, minlength=size)

        sum_kappa_x0 += np.bincount(t, weights=kappa * x0, minlength=size)

        zero_rows = chunk[chunk["X_0"] == 0.0]

        if not zero_rows.empty:
            first_hits = (
                zero_rows
                .groupby("run_index", as_index=False)["time"]
                .min()
            )

            for _, row in first_hits.iterrows():
                run_index = int(row["run_index"])
                hit_time = int(row["time"])

                if run_index not in first_loss_by_run:
                    first_loss_by_run[run_index] = hit_time
                else:
                    first_loss_by_run[run_index] = min(
                        first_loss_by_run[run_index],
                        hit_time,
                    )

    valid = count > 1

    kappa_mean = np.full(size, np.nan)
    x0_mean = np.full(size, np.nan)

    kappa_var = np.full(size, np.nan)
    x0_var = np.full(size, np.nan)
    cov_kappa_x0 = np.full(size, np.nan)

    kappa_mean[valid] = sum_kappa[valid] / count[valid]
    x0_mean[valid] = sum_x0[valid] / count[valid]

    kappa_var[valid] = (
        sum_kappa_sq[valid]
        - count[valid] * kappa_mean[valid] ** 2
    ) / (count[valid] - 1.0)

    x0_var[valid] = (
        sum_x0_sq[valid]
        - count[valid] * x0_mean[valid] ** 2
    ) / (count[valid] - 1.0)

    cov_kappa_x0[valid] = (
        sum_kappa_x0[valid]
        - count[valid] * kappa_mean[valid] * x0_mean[valid]
    ) / (count[valid] - 1.0)

    aggregate = pd.DataFrame(
        {
            "time": np.arange(size),
            "simulation_kappa_1_var": kappa_var,
            "simulation_X_0_var": x0_var,
            "simulation_cov_kappa_1_X_0": cov_kappa_x0,
        }
    )

    aggregate = aggregate[aggregate["time"].isin(times)].copy()

    # Per-time percentiles of m_1 across runs (for the spread band of the m1 plots).
    if t_chunks:
        qdf = pd.DataFrame({
            "time": np.concatenate(t_chunks),
            "kappa": np.concatenate(kappa_chunks),
        })
        qt = (
            qdf.groupby("time")["kappa"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .unstack()
        )
        qt.columns = [
            "kappa_1_q10", "kappa_1_q25", "kappa_1_q50", "kappa_1_q75", "kappa_1_q90"
        ]
        aggregate = aggregate.merge(qt.reset_index(), on="time", how="left")

    first_loss_df = pd.DataFrame(
        {
            "run_index": list(first_loss_by_run.keys()),
            "first_loss_time": list(first_loss_by_run.values()),
        }
    )

    if not first_loss_df.empty:
        first_loss_df = first_loss_df.sort_values("run_index")

    return aggregate, first_loss_df


# ============================================================
# Plot helpers
# ============================================================

def get_first_loss_summary(first_loss_df):
    if first_loss_df.empty:
        return {
            "n_hits_until_time_max": 0,
            "mean_first_loss_time": np.nan,
            "median_first_loss_time": np.nan,
            "q25_first_loss_time": np.nan,
            "q75_first_loss_time": np.nan,
            "min_first_loss_time": np.nan,
            "max_first_loss_time": np.nan,
        }

    values = first_loss_df["first_loss_time"].to_numpy(dtype=float)

    return {
        "n_hits_until_time_max": int(len(first_loss_df)),
        "mean_first_loss_time": float(np.mean(values)),
        "median_first_loss_time": float(np.median(values)),
        "q25_first_loss_time": float(np.quantile(values, 0.25)),
        "q75_first_loss_time": float(np.quantile(values, 0.75)),
        "min_first_loss_time": float(np.min(values)),
        "max_first_loss_time": float(np.max(values)),
    }


def add_first_loss_lines(ax, first_loss_summary):
    mean_t = first_loss_summary["mean_first_loss_time"]
    median_t = first_loss_summary["median_first_loss_time"]

    if not np.isnan(mean_t):
        ax.axvline(
            mean_t,
            color="#636363",
            linestyle=":",
            linewidth=3,
            label=rf"mean first loss = {mean_t:.1f}",
        )

    if not np.isnan(median_t):
        ax.axvline(
            median_t,
            color="#252525",
            linestyle="-.",
            linewidth=3,
            label=rf"median first loss = {median_t:.1f}",
        )


def add_m1_spread_band(ax, comparison):
    """Shade the 10-90% inter-path range of m_1 and draw its median.

    Conveys the spread of m_1 across paths (unlike +/- 2 SE, which is only the
    uncertainty of the mean). Needs the per-time percentiles from the by-run
    file; silently does nothing if they are unavailable."""
    if "kappa_1_q10" not in comparison.columns or not comparison["kappa_1_q10"].notna().any():
        return

    ax.fill_between(
        comparison["time"],
        comparison["kappa_1_q10"],
        comparison["kappa_1_q90"],
        color="#9e9e9e",
        alpha=0.30,
        linewidth=0,
        label="simulation 10–90%",
    )


def plot_first_loss_violin(first_loss_df, first_loss_summary):
    if first_loss_df.empty:
        print("Skipping first-loss violin plot because no first losses were found.")
        return

    values = first_loss_df["first_loss_time"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 7))

    parts = ax.violinplot(
        values,
        positions=[1],
        showmeans=True,
        showmedians=True,
        showextrema=True,
    )

    for body in parts["bodies"]:
        body.set_facecolor("#9ecae1")
        body.set_edgecolor("#084594")
        body.set_alpha(0.8)

    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#084594")
            parts[key].set_linewidth(2.3)

    ax.scatter(
        np.ones_like(values),
        values,
        color="#084594",
        s=16,
        alpha=0.25,
    )

    mean_t = first_loss_summary["mean_first_loss_time"]
    median_t = first_loss_summary["median_first_loss_time"]
    q25 = first_loss_summary["q25_first_loss_time"]
    q75 = first_loss_summary["q75_first_loss_time"]

    ax.axhline(mean_t, color="#636363", linestyle=":", linewidth=3, label=rf"mean = {mean_t:.1f}")
    ax.axhline(median_t, color="#252525", linestyle="-.", linewidth=3, label=rf"median = {median_t:.1f}")
    ax.axhline(q25, color="#969696", linestyle="--", linewidth=2.2, label=rf"25% = {q25:.1f}")
    ax.axhline(q75, color="#969696", linestyle="--", linewidth=2.2, label=rf"75% = {q75:.1f}")

    ax.set_xticks([1])
    ax.set_xticklabels([r"first loss"])
    ax.set_ylabel(r"first loss time")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=17)

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"first_loss_time_violin_until_{TIME_MAX_PLOT}_{SELECTION_MODE}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_m1_first_order(comparison, first_loss_summary, first_loss_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["time"],
        comparison["kappa_1_mean"],
        color=COLOR_M1_SIM,
        linewidth=3,
        label="simulation mean",
    )

    add_m1_spread_band(ax, comparison)

    add_first_loss_lines(ax, first_loss_summary)

    # Rug of the individual first-loss (click) times along the bottom axis:
    # one tick per path, so one sees *when* each path actually clicked.
    if first_loss_df is not None and not first_loss_df.empty:
        t0s = first_loss_df["first_loss_time"].to_numpy(dtype=float)
        t0s = t0s[t0s <= TIME_MAX_PLOT]
        if t0s.size:
            ax.plot(
                t0s,
                np.full_like(t0s, 0.02),
                marker="|",
                linestyle="none",
                markersize=14,
                markeredgewidth=1.2,
                color=COLOR_M1_SIM,
                alpha=0.35,
                transform=ax.get_xaxis_transform(),  # x in data, y in axes fraction
                label=f"first losses (n={t0s.size})",
            )

    # Theory drawn last (high zorder) so it stays on top of the band/curves.
    ax.plot(
        comparison["time"],
        comparison["theory_kappa_1_mean_first_order"],
        color=COLOR_M1_THEORY_1,
        linewidth=3,
        linestyle="--",
        label=r"theory up to $1/\nu$",
        zorder=10,
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{E}(m_1(X^N_t))$")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", ncol=2)

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{SELECTION_MODE}_m1_first_order_until_{TIME_MAX_PLOT}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_m1_second_order(comparison, first_loss_summary):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["time"],
        comparison["kappa_1_mean"],
        color=COLOR_M1_SIM,
        linewidth=3,
        label="simulation mean",
    )

    add_m1_spread_band(ax, comparison)

    add_first_loss_lines(ax, first_loss_summary)

    # Theory drawn last (high zorder) so it stays on top of the band/curves.
    ax.plot(
        comparison["time"],
        comparison["theory_kappa_1_mean_first_order"],
        color=COLOR_M1_THEORY_1,
        linewidth=3,
        linestyle="--",
        label=r"theory up to $1/\nu$",
        zorder=10,
    )

    ax.plot(
        comparison["time"],
        comparison["theory_kappa_1_mean_second_order"],
        color=COLOR_M1_THEORY_2,
        linewidth=3,
        linestyle="-.",
        label=r"theory up to $1/\nu^2$",
        zorder=11,
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{E}(m_1(X^N_t))$")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", ncol=2)

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{SELECTION_MODE}_m1_second_order_until_{TIME_MAX_PLOT}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_x0_to_x3_together(comparison):
    fig, ax = plt.subplots(figsize=(10, 6))

    for k in range(4):
        ax.plot(
            comparison["time"],
            comparison[f"X_{k}_mean"],
            color=X_COLORS[k]["sim"],
            linewidth=3,
            label=rf"simulation $X^N_{k}$",
        )

        ax.plot(
            comparison["time"],
            comparison[f"theory_X_{k}_mean"],
            color=X_COLORS[k]["theory"],
            linewidth=3,
            linestyle="--",
            label=rf"theory $X^N_{k}$",
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{E}(X^N_k(t))$")

    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=17)

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{SELECTION_MODE}_X0_X1_X2_X3_together_until_{TIME_MAX_PLOT}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_variance_comparison(comparison):
    if "simulation_kappa_1_var" not in comparison.columns:
        print("Skipping variance plot because by-run data are unavailable.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["time"],
        comparison["simulation_kappa_1_var"],
        color=COLOR_VAR_SIM,
        linewidth=3,
        label="simulation",
    )

    ax.plot(
        comparison["time"],
        comparison["theory_kappa_1_var"],
        color=COLOR_VAR_THEORY,
        linewidth=3,
        linestyle="--",
        label=r"theory up to $1/\nu$",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{V}(m_1(X^N_t))$")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{SELECTION_MODE}_variance_m1_until_{TIME_MAX_PLOT}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_covariance_comparison(comparison):
    if "simulation_cov_kappa_1_X_0" not in comparison.columns:
        print("Skipping covariance plot because by-run data are unavailable.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        comparison["time"],
        comparison["simulation_cov_kappa_1_X_0"],
        color=COLOR_COV_SIM,
        linewidth=3,
        label="simulation",
    )

    ax.plot(
        comparison["time"],
        comparison["theory_cov_kappa_1_X_0"],
        color=COLOR_COV_THEORY,
        linewidth=3,
        linestyle="--",
        label=r"theory up to $1/\nu$",
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\operatorname{Cov}(m_1(X^N_t),X^N_0(t))$")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{SELECTION_MODE}_cov_m1_X0_until_{TIME_MAX_PLOT}.png",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def add_error_columns(comparison):
    pairs = [
        ("kappa_1_first_order", "kappa_1_mean", "theory_kappa_1_mean_first_order"),
        ("kappa_1_second_order", "kappa_1_mean", "theory_kappa_1_mean_second_order"),
        ("X_0", "X_0_mean", "theory_X_0_mean"),
        ("X_1", "X_1_mean", "theory_X_1_mean"),
        ("X_2", "X_2_mean", "theory_X_2_mean"),
        ("X_3", "X_3_mean", "theory_X_3_mean"),
    ]

    for name, sim_col, theory_col in pairs:
        comparison[f"abs_error_{name}"] = np.abs(
            comparison[sim_col] - comparison[theory_col]
        )

        comparison[f"rel_error_{name}"] = (
            comparison[f"abs_error_{name}"]
            / np.maximum(np.abs(comparison[theory_col]), 1e-15)
        )

    if "simulation_kappa_1_var" in comparison.columns:
        comparison["abs_error_var_kappa_1"] = np.abs(
            comparison["simulation_kappa_1_var"]
            - comparison["theory_kappa_1_var"]
        )

    if "simulation_cov_kappa_1_X_0" in comparison.columns:
        comparison["abs_error_cov_kappa_1_X_0"] = np.abs(
            comparison["simulation_cov_kappa_1_X_0"]
            - comparison["theory_cov_kappa_1_X_0"]
        )

    return comparison


# ============================================================
# Main
# ============================================================

def main():
    means_path = find_means_file()
    by_run_path = find_by_run_file()

    params = load_parameters()

    N = params["N"]
    u = params["u"]
    s = params["s"]
    u_div_s = params["u_div_s"]

    print("\n" + "=" * 80)
    print(f"Selection mode: {SELECTION_MODE}")
    print(f"N={N}, u={u}, s={s}, u_div_s={u_div_s}")
    print(f"Using only time <= {TIME_MAX_PLOT}")
    print("=" * 80)

    means = pd.read_csv(means_path)

    required_mean_cols = [
        "time",
        "X_0_mean",
        "X_1_mean",
        "X_2_mean",
        "X_3_mean",
        "kappa_1_mean",
    ]

    missing = [col for col in required_mean_cols if col not in means.columns]

    if missing:
        raise ValueError(f"Missing columns in {means_path}: {missing}")

    means = means[means["time"] <= TIME_MAX_PLOT].copy()

    times = means["time"].to_numpy(dtype=float)

    theory = build_theory_dataframe(
        times=times,
        N=N,
        s=s,
        u_div_s=u_div_s,
    )

    comparison = means.merge(theory, on="time", how="left")

    first_loss_df = pd.DataFrame(
        columns=["run_index", "first_loss_time"]
    )

    if by_run_path is not None:
        aggregate, first_loss_df = aggregate_by_run_file(
            path=by_run_path,
            times=means["time"].to_numpy(dtype=int),
        )

        comparison = comparison.merge(aggregate, on="time", how="left")
    else:
        print(
            "No by-run file found. First loss time, variance, and covariance "
            "cannot be computed."
        )

    first_loss_path = os.path.join(
        OUTPUT_DIR,
        f"first_loss_times_until_{TIME_MAX_PLOT}_{SELECTION_MODE}.csv",
    )

    first_loss_df.to_csv(first_loss_path, index=False)
    print(f"Saved: {first_loss_path}")

    first_loss_summary = get_first_loss_summary(first_loss_df)

    first_loss_summary_path = os.path.join(
        OUTPUT_DIR,
        f"first_loss_summary_until_{TIME_MAX_PLOT}_{SELECTION_MODE}.csv",
    )

    pd.DataFrame([first_loss_summary]).to_csv(
        first_loss_summary_path,
        index=False,
    )

    print(f"Saved: {first_loss_summary_path}")
    print("\nFirst loss summary:")
    print(first_loss_summary)

    comparison = add_error_columns(comparison)

    comparison_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_until_{TIME_MAX_PLOT}_{SELECTION_MODE}.csv",
    )

    comparison.to_csv(comparison_path, index=False)
    print(f"Saved: {comparison_path}")

    plot_first_loss_violin(first_loss_df, first_loss_summary)
    plot_m1_first_order(comparison, first_loss_summary, first_loss_df)
    plot_m1_second_order(comparison, first_loss_summary)
    plot_x0_to_x3_together(comparison)
    plot_variance_comparison(comparison)
    plot_covariance_comparison(comparison)

    error_summary = {
        "selection_mode": SELECTION_MODE,
        "N": N,
        "u": u,
        "s": s,
        "u_div_s": u_div_s,
        "time_max_plot": TIME_MAX_PLOT,
        "n_first_losses_until_time_max": first_loss_summary["n_hits_until_time_max"],
        "mean_first_loss_time": first_loss_summary["mean_first_loss_time"],
        "median_first_loss_time": first_loss_summary["median_first_loss_time"],
        "q25_first_loss_time": first_loss_summary["q25_first_loss_time"],
        "q75_first_loss_time": first_loss_summary["q75_first_loss_time"],
        "max_abs_error_kappa_1_first_order": comparison["abs_error_kappa_1_first_order"].max(),
        "max_abs_error_kappa_1_second_order": comparison["abs_error_kappa_1_second_order"].max(),
        "max_abs_error_X_0": comparison["abs_error_X_0"].max(),
        "max_abs_error_X_1": comparison["abs_error_X_1"].max(),
        "max_abs_error_X_2": comparison["abs_error_X_2"].max(),
        "max_abs_error_X_3": comparison["abs_error_X_3"].max(),
    }

    if "abs_error_var_kappa_1" in comparison.columns:
        error_summary["max_abs_error_var_kappa_1"] = (
            comparison["abs_error_var_kappa_1"].max()
        )

    if "abs_error_cov_kappa_1_X_0" in comparison.columns:
        error_summary["max_abs_error_cov_kappa_1_X_0"] = (
            comparison["abs_error_cov_kappa_1_X_0"].max()
        )

    error_summary_path = os.path.join(
        OUTPUT_DIR,
        f"comparison_error_summary_until_{TIME_MAX_PLOT}_{SELECTION_MODE}.csv",
    )

    pd.DataFrame([error_summary]).to_csv(
        error_summary_path,
        index=False,
    )

    print(f"Saved: {error_summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
