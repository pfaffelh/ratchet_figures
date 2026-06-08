import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


#  Parameters

N_VALUES = [10000]
PSI_VALUES = [0.7, 1.0, 1.3, 2.0, 5.0, 10.0]
DELTA_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]

RUNS = 1000
BATCH_SIZE = 1000
SEED = 22

T_MAX = 1000000

# CPU pre-screen (gate). Before launching the expensive full GPU run for a
# parameter set, simulate PROBE_RUNS paths on the CPU and only proceed if ALL of
# them click (the fittest class X_0 empties) within T_MAX. The analysis keeps a
# parameter set only when 100% of paths click (MIN_FRAC_HIT=1.0 in make_figure_A /
# make_table), so a single non-clicking probe path already rules the set out --
# skipping it then avoids burning GPU time on a set that can never yield a data
# point. The probe runs on the CPU with the same T_MAX as the real run, so
# confirming a skip for a non-clicking set runs the probe to the full horizon.
# Set PROBE_ENABLED = False to always run the full simulation.
PROBE_ENABLED = True
PROBE_RUNS = 8

# Resume support. When True, a parameter set whose output CSV already exists in
# OUTPUT_DIR is skipped (neither probed nor re-simulated) and its existing data
# is reused for the combined summary. This lets an interrupted run be restarted
# without recomputing finished sets or overwriting their results. Note: sets the
# probe skipped (non-clicking) write no CSV, so they are re-probed on restart.
SKIP_EXISTING = True

# Within-generation update order.
#   False (default): selection then mutation; stationary marginal Poi(u/s),
#       started at Poi(u/s). Keeps the current behaviour and summary file names.
#   True : mutation then selection; stationary marginal Poi(u(1-s)/s), started
#       at that equilibrium. Summary files get a "_mutation_first" tag.
MUTATION_FIRST = True

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

REQUIRE_CUDA = True
TORCH_DTYPE = torch.float32

#  "exponential" fitness =  exp(-s k).
#  "one_minus_s_power": fitness =  (1-s)^k.
SELECTION_MODE = "one_minus_s_power"

MUTATION_TAIL_TOL = 1e-12
MAX_MUTATIONS_CAP = 250

# Number of mutation classes to track. The occupied band in Muller's ratchet
# stays narrow: validated max k_max = 43 across all 51 valid parameter sets at
# runs=1000, T_MAX=1000 (selection one_minus_s_power). A cap of 256 keeps a
# ~5.8x safety margin while making the per-generation multinomial far cheaper
# than tracking all N classes. The last class is an absorbing tail bucket, so
# keep this well above any band you expect; simulate_parameter_set prints a
# warning if the occupied band ever approaches the cap.
NUM_CLASSES_CAP = 256

# Draw the per-generation multinomial in chunks along the run axis. The
# transient (chunk, N) index buffer that torch.multinomial materialises is the
# real memory bottleneck; chunking keeps it small even with many parallel runs.
MULTINOMIAL_RUN_CHUNK = 8192


# Parameter grid

PARAMETER_SETS = [
    {"N": N, "psi": psi, "delta": delta}
    for N in N_VALUES
    for psi in PSI_VALUES
    for delta in DELTA_VALUES
]


# Device setup

def get_torch_device(require_cuda=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        return device

    if require_cuda:
        raise RuntimeError(
            "CUDA is not available in PyTorch. Install a CUDA-enabled PyTorch version "
            "or set REQUIRE_CUDA = False to run on CPU."
        )

    print("CUDA is not available. Running on CPU.")
    return torch.device("cpu")


# Parameter formulas

def parameters_from_psi_delta(N, psi, delta):
    u_div_s = delta * math.log(N)
    s = psi * math.exp(u_div_s) / N
    u = s * u_div_s

    effective_beta = -math.log(u) / math.log(N)

    psi_check = N * s * math.exp(-u_div_s)

    return {
        "u_div_s": u_div_s,
        "s": s,
        "u": u,
        "effective_beta": effective_beta,
        "psi_check": psi_check,
    }


# File helpers

def float_tag(x):
    return f"{x:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def summary_filename(N, psi, delta):
    tag = "_mutation_first" if MUTATION_FIRST else "_selection_first"
    return (
        f"summary"
        f"_N_{int(N)}"
        f"_psi_{float_tag(psi)}"
        f"_delta_{float_tag(delta)}"
        f"{tag}.csv"
    )


def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# Wright-Fisher Diffusion

def poisson_lumped_probs(u_div_s, num_classes, device, dtype):
    k = torch.arange(num_classes - 1, device=device, dtype=dtype)
    u_div_s_tensor = torch.tensor(u_div_s, device=device, dtype=dtype)

    log_probs = (
        -u_div_s_tensor
        + k * torch.log(u_div_s_tensor)
        - torch.lgamma(k + 1.0)
    )

    probs_without_tail = torch.exp(log_probs)
    tail = torch.clamp(1.0 - probs_without_tail.sum(), min=0.0)

    probs = torch.cat([probs_without_tail, tail.reshape(1)])
    probs = probs / probs.sum()

    return probs


def build_poisson_mutation_kernel(
    u,
    num_classes,
    device,
    dtype,
    tail_tol=1e-12,
    max_mutations_cap=250,
):
    probs = [math.exp(-u)]
    cumulative = probs[0]
    m = 0

    while (
        (1.0 - cumulative) > tail_tol
        and m < max_mutations_cap
        and m < num_classes - 1
    ):
        m += 1
        probs.append(probs[-1] * u / m)
        cumulative += probs[-1]

    probs_np = np.array(probs, dtype=np.float64)
    cdf_np = np.cumsum(probs_np)

    tail_by_d = np.zeros(num_classes, dtype=np.float64)
    tail_by_d[0] = 1.0

    for d in range(1, num_classes):
        idx = d - 1

        if idx < len(cdf_np):
            tail_by_d[d] = max(0.0, 1.0 - cdf_np[idx])
        else:
            tail_by_d[d] = 0.0

    d_by_source = np.arange(num_classes - 1, -1, -1, dtype=np.int64)
    tail_by_source_np = tail_by_d[d_by_source]

    poisson_probs = torch.tensor(
        probs_np,
        device=device,
        dtype=dtype,
    )

    tail_by_source = torch.tensor(
        tail_by_source_np,
        device=device,
        dtype=dtype,
    )

    return poisson_probs, tail_by_source


def mutate_counts_poisson_shift(counts, poisson_probs, tail_by_source):
    runs, num_classes = counts.shape
    last = num_classes - 1

    # Shifting by m with weight poisson_probs[m] is a causal 1D correlation
    # along the class axis: mutated[:, j] = sum_m counts[:, j - m] * probs[m].
    # A single left-padded conv1d replaces the per-m Python loop (one fused
    # kernel instead of len(poisson_probs) slice-adds), which dominated the
    # generation step.
    kernel_size = int(poisson_probs.numel())
    weight = poisson_probs.flip(0).reshape(1, 1, kernel_size)

    padded = F.pad(counts.unsqueeze(1), (kernel_size - 1, 0))
    mutated = F.conv1d(padded, weight).squeeze(1)

    # The last class collects the entire upper mutation tail exactly.
    mutated[:, last] = counts @ tail_by_source

    return mutated


def torch_multinomial_counts_batch(N, probs, generator=None, run_chunk=None):
    runs, num_classes = probs.shape

    if run_chunk is None:
        run_chunk = MULTINOMIAL_RUN_CHUNK

    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)

    counts = torch.empty(
        (runs, num_classes),
        device=probs.device,
        dtype=torch.long,
    )

    for start in range(0, runs, run_chunk):
        stop = min(start + run_chunk, runs)
        chunk_runs = stop - start

        samples = torch.multinomial(
            probs[start:stop],
            num_samples=N,
            replacement=True,
            generator=generator,
        )

        offsets = (
            torch.arange(chunk_runs, device=probs.device, dtype=torch.long)
            .unsqueeze(1)
            * num_classes
        )

        flat_samples = (samples + offsets).reshape(-1)

        counts[start:stop] = torch.bincount(
            flat_samples,
            minlength=chunk_runs * num_classes,
        ).reshape(chunk_runs, num_classes)

    return counts


def build_selection_vector(s, num_classes, device, dtype):
    k = torch.arange(num_classes, device=device, dtype=dtype)

    if SELECTION_MODE == "exponential":
        return torch.exp(-s * k)

    if SELECTION_MODE == "one_minus_s_power":
        if s >= 1.0:
            raise ValueError(
                f"s={s} is not valid for selection=(1-s)^k. "
                "Use SELECTION_MODE='exponential' or choose smaller s."
            )

        return (1.0 - s) ** k

    raise ValueError(
        "SELECTION_MODE must be 'exponential' or 'one_minus_s_power'."
    )


def one_wright_fisher_generation(
    counts,
    N,
    poisson_probs,
    tail_by_source,
    selection,
    generator=None,
):
    if MUTATION_FIRST:
        # Mutation first, then selection (S after M_u): stationary marginal
        # Poi(u (1 - s) / s) for fitness (1 - s)^k.
        mutated = mutate_counts_poisson_shift(
            counts=counts,
            poisson_probs=poisson_probs,
            tail_by_source=tail_by_source,
        )
        probs = mutated * selection
    else:
        # Selection first, then mutation (M_u after S): stationary marginal
        # Poi(u / s) for fitness (1 - s)^k.
        selected = counts * selection
        mutated = mutate_counts_poisson_shift(
            counts=selected,
            poisson_probs=poisson_probs,
            tail_by_source=tail_by_source,
        )
        probs = mutated

    probs = probs / probs.sum(dim=1, keepdim=True)

    new_counts = torch_multinomial_counts_batch(
        N=N,
        probs=probs,
        generator=generator,
    )

    return new_counts


# Simulation for one parameter set

def simulate_parameter_set(
    N,
    psi,
    delta,
    runs,
    batch_size,
    t_max,
    output_dir,
    seed,
    device,
    dtype,
):
    params = parameters_from_psi_delta(
        N=N,
        psi=psi,
        delta=delta,
    )

    u_div_s = params["u_div_s"]
    s = params["s"]
    u = params["u"]
    effective_beta = params["effective_beta"]
    psi_check = params["psi_check"]

    num_classes = min(int(N), NUM_CLASSES_CAP)

    print(
        f"Running N={N}, psi={psi}, delta={delta}, "
        f"u_div_s={u_div_s:.8g}, s={s:.8g}, u={u:.8g}, "
        f"effective_beta={effective_beta:.8g}, psi_check={psi_check:.8g}, "
        f"selection_mode={SELECTION_MODE}"
    )

    try:
        selection = build_selection_vector(
            s=s,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )
    except ValueError as error:
        print(f"Skipping parameter set: {error}")
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Start each path at the stationary marginal of the chosen update order:
    # Poi(u/s) for selection->mutation, Poi(u(1-s)/s) for mutation->selection.
    init_u_div_s = u_div_s * (1.0 - s) if MUTATION_FIRST else u_div_s
    init_probs = poisson_lumped_probs(
        u_div_s=init_u_div_s,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
    )

    poisson_probs, tail_by_source = build_poisson_mutation_kernel(
        u=u,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
        tail_tol=MUTATION_TAIL_TOL,
        max_mutations_cap=MAX_MUTATIONS_CAP,
    )

    summary_frames = []

    with torch.no_grad():
        for start in range(0, runs, batch_size):
            m = min(batch_size, runs - start)
            global_path_indices = np.arange(start, start + m, dtype=int)

            probs0 = init_probs.unsqueeze(0).repeat(m, 1)

            counts = torch_multinomial_counts_batch(
                N=N,
                probs=probs0,
                generator=generator,
            )

            # The only quantity of interest per path is t_0: the first time the
            # fittest class (index 0) empties. The path is stopped at t_0 and
            # nothing afterwards is simulated or recorded. `hit` doubles as the
            # done mask; paths that never hit run to t_max (counted as misses).
            hit = counts[:, 0] == 0

            first_loss_time = torch.full(
                (m,),
                -1,
                device=device,
                dtype=torch.long,
            )
            first_loss_time[hit] = 0

            t = 0

            while t < t_max and not torch.all(hit):
                active_indices = torch.nonzero(~hit, as_tuple=False).flatten()

                counts[active_indices] = one_wright_fisher_generation(
                    counts=counts[active_indices].to(dtype),
                    N=N,
                    poisson_probs=poisson_probs,
                    tail_by_source=tail_by_source,
                    selection=selection,
                    generator=generator,
                )

                t += 1

                new_hits = (~hit) & (counts[:, 0] == 0)
                if torch.any(new_hits):
                    first_loss_time[new_hits] = t
                    hit[new_hits] = True   # path is done at t_0

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            # Guard against silently truncating the occupied band: if it ever
            # approached the class cap, results past num_classes-1 would be lost.
            # (Cheap: one reduction per batch. The absorbing last class is
            # excluded so the t=0 init phantom does not trigger it.)
            occupied = counts[:, : num_classes - 1] > 0
            class_idx = torch.arange(num_classes - 1, device=device)
            band_top = int(
                torch.where(occupied, class_idx, torch.full_like(class_idx, -1)).max()
            )
            if band_top >= int(0.75 * num_classes):
                print(
                    f"  WARNING: occupied band reached class {band_top} of "
                    f"num_classes={num_classes}; raise NUM_CLASSES_CAP to avoid "
                    f"truncating the upper classes."
                )

            hit_cpu = hit.detach().cpu().numpy().astype(bool)
            first_loss_time_cpu = first_loss_time.detach().cpu().numpy()
            t0_float = first_loss_time_cpu.astype(float)
            t0_float[first_loss_time_cpu < 0] = np.nan

            summary_frames.append(
                pd.DataFrame(
                    {
                        "N": int(N),
                        "psi": float(psi),
                        "delta": float(delta),
                        "u": float(u),
                        "u_div_s": float(u_div_s),
                        "s": float(s),
                        "effective_beta": float(effective_beta),
                        "psi_check": float(psi_check),
                        "selection_mode": SELECTION_MODE,
                        "update_order": "mutation_first" if MUTATION_FIRST else "selection_first",
                        "path_index": global_path_indices,
                        "hit_x0_zero": hit_cpu,
                        "t_0": t0_float,
                        "first_loss_time": t0_float,
                        "num_classes": int(num_classes),
                        "T_MAX": int(t_max),
                    }
                )
            )

            print(
                f"  Finished paths {start} to {start + m - 1}; "
                f"hits: {int(hit_cpu.sum())}/{m}"
            )

    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_path = os.path.join(output_dir, summary_filename(N, psi, delta))
    save_dataframe(summary_df, summary_path)

    return summary_df


# CPU pre-screen

def probe_parameter_set(N, psi, delta, probe_runs, t_max, seed, dtype):
    """Cheap CPU gate for the full GPU run.

    Simulate ``probe_runs`` paths on the CPU and report whether they all click
    (the fittest class empties) within ``t_max``. Mirrors the inner loop of
    simulate_parameter_set but records nothing and writes no files. Returns a
    dict with:
        valid     -- False if the parameter set is skipped (e.g. s >= 1)
        all_click -- True iff every probe path clicked within t_max
        n_hit     -- number of probe paths that clicked
        t_reached -- generation at which the probe loop stopped
        reason    -- only present when valid is False
    """
    device = torch.device("cpu")

    params = parameters_from_psi_delta(N=N, psi=psi, delta=delta)
    u_div_s = params["u_div_s"]
    s = params["s"]
    u = params["u"]

    num_classes = min(int(N), NUM_CLASSES_CAP)

    try:
        selection = build_selection_vector(
            s=s,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )
    except ValueError as error:
        return {"valid": False, "reason": str(error)}

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # Start each path at the stationary marginal of the chosen update order:
    # Poi(u/s) for selection->mutation, Poi(u(1-s)/s) for mutation->selection.
    init_u_div_s = u_div_s * (1.0 - s) if MUTATION_FIRST else u_div_s
    init_probs = poisson_lumped_probs(
        u_div_s=init_u_div_s,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
    )

    poisson_probs, tail_by_source = build_poisson_mutation_kernel(
        u=u,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
        tail_tol=MUTATION_TAIL_TOL,
        max_mutations_cap=MAX_MUTATIONS_CAP,
    )

    with torch.no_grad():
        probs0 = init_probs.unsqueeze(0).repeat(probe_runs, 1)
        counts = torch_multinomial_counts_batch(
            N=N,
            probs=probs0,
            generator=generator,
        )

        # Same stop-at-t_0 logic as the full run: clicked paths are frozen
        # (excluded from the active set) so their empty class 0 stays empty.
        hit = counts[:, 0] == 0
        t = 0

        while t < t_max and not torch.all(hit):
            active_indices = torch.nonzero(~hit, as_tuple=False).flatten()

            counts[active_indices] = one_wright_fisher_generation(
                counts=counts[active_indices].to(dtype),
                N=N,
                poisson_probs=poisson_probs,
                tail_by_source=tail_by_source,
                selection=selection,
                generator=generator,
            )

            t += 1
            hit = hit | (counts[:, 0] == 0)

    n_hit = int(hit.sum())

    return {
        "valid": True,
        "all_click": bool(n_hit == probe_runs),
        "n_hit": n_hit,
        "t_reached": int(t),
    }


# Run all parameter sets

def run_all_parameter_sets(
    parameter_sets,
    output_dir,
    runs,
    batch_size,
    t_max,
    seed,
    require_cuda,
    dtype,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Parameter sets:")
    for params in parameter_sets:
        derived = parameters_from_psi_delta(
            N=int(params["N"]),
            psi=float(params["psi"]),
            delta=float(params["delta"]),
        )

        print(
            {
                **params,
                "u_div_s": derived["u_div_s"],
                "s": derived["s"],
                "u": derived["u"],
                "effective_beta": derived["effective_beta"],
                "psi_check": derived["psi_check"],
            }
        )

    device = get_torch_device(require_cuda=require_cuda)

    all_summaries = []

    for run_index, params in enumerate(parameter_sets):
        N = int(params["N"])
        psi = float(params["psi"])
        delta = float(params["delta"])
        set_seed = seed + run_index

        if SKIP_EXISTING:
            existing_path = os.path.join(
                output_dir, summary_filename(N, psi, delta)
            )

            if os.path.exists(existing_path):
                try:
                    existing_df = pd.read_csv(existing_path)
                except Exception as error:
                    print(
                        f"  Existing {os.path.basename(existing_path)} could "
                        f"not be read ({error}); re-running this set."
                    )
                else:
                    print(
                        f"Skipping N={N}, psi={psi}, delta={delta}: output "
                        f"already exists ({os.path.basename(existing_path)})."
                    )
                    all_summaries.append(existing_df)
                    continue

        if PROBE_ENABLED:
            probe = probe_parameter_set(
                N=N,
                psi=psi,
                delta=delta,
                probe_runs=PROBE_RUNS,
                t_max=t_max,
                seed=set_seed,
                dtype=dtype,
            )

            if not probe["valid"]:
                print(
                    f"Skipping N={N}, psi={psi}, delta={delta}: {probe['reason']}"
                )
                continue

            if not probe["all_click"]:
                print(
                    f"CPU probe: N={N}, psi={psi}, delta={delta} -> only "
                    f"{probe['n_hit']}/{PROBE_RUNS} paths clicked within t_max "
                    f"(reached t={probe['t_reached']}); skipping full run."
                )
                continue

            print(
                f"CPU probe: N={N}, psi={psi}, delta={delta} -> all "
                f"{PROBE_RUNS}/{PROBE_RUNS} clicked (max t={probe['t_reached']}); "
                f"launching full GPU simulation."
            )

        summary_df = simulate_parameter_set(
            N=N,
            psi=psi,
            delta=delta,
            runs=runs,
            batch_size=batch_size,
            t_max=t_max,
            output_dir=output_dir,
            seed=set_seed,
            device=device,
            dtype=dtype,
        )

        if summary_df is not None:
            all_summaries.append(summary_df)

    if not all_summaries:
        raise RuntimeError("No parameter set was simulated.")

    combined_summary = pd.concat(all_summaries, ignore_index=True)

    combined_summary_path = os.path.join(
        output_dir,
        "summary_all_parameter_sets.csv",
    )

    save_dataframe(combined_summary, combined_summary_path)

    return combined_summary


# Main

if __name__ == "__main__":
    combined_summary = run_all_parameter_sets(
        parameter_sets=PARAMETER_SETS,
        output_dir=OUTPUT_DIR,
        runs=RUNS,
        batch_size=BATCH_SIZE,
        t_max=T_MAX,
        seed=SEED,
        require_cuda=REQUIRE_CUDA,
        dtype=TORCH_DTYPE,
    )

    print(combined_summary)
