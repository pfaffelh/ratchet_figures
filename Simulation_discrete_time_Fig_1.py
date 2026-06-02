import os
import math
import numpy as np
import pandas as pd
import torch


# ============================================================
# User settings
# ============================================================

N = 10000
T = 10000
RUNS = 1000
BATCH_SIZE = 100

LAM = 0.04
ALPHA = 0.01
THETA = LAM / ALPHA

NUM_CLASSES = N

SELECTION_MODES = [
    "exponential",
    "one_minus_alpha_power",
]

OUTPUT_DIR = "ratchet_fixed_lambda_alpha_results"

REQUIRE_CUDA = True
TORCH_DTYPE = torch.float32
SEED = 22

WRITE_EVERY_GENERATIONS = 50

MUTATION_TAIL_TOL = 1e-12
MAX_MUTATIONS_CAP = 250


# ============================================================
# Device setup
# ============================================================

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


# ============================================================
# File helpers
# ============================================================

def float_tag(x):
    return f"{x:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def append_dataframe_csv(df, path):
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


# ============================================================
# Wright-Fisher helpers
# ============================================================

def poisson_lumped_probs(theta, num_classes, device, dtype):
    k = torch.arange(num_classes - 1, device=device, dtype=dtype)
    theta_tensor = torch.tensor(theta, device=device, dtype=dtype)

    log_probs = (
        -theta_tensor
        + k * torch.log(theta_tensor)
        - torch.lgamma(k + 1.0)
    )

    probs_without_tail = torch.exp(log_probs)
    tail = torch.clamp(1.0 - probs_without_tail.sum(), min=0.0)

    probs = torch.cat([probs_without_tail, tail.reshape(1)])
    probs = probs / probs.sum()

    return probs


def build_poisson_mutation_kernel(
    lam,
    num_classes,
    device,
    dtype,
    tail_tol=1e-12,
    max_mutations_cap=250,
):
    probs = [math.exp(-lam)]
    cumulative = probs[0]
    m = 0

    while (
        (1.0 - cumulative) > tail_tol
        and m < max_mutations_cap
        and m < num_classes - 1
    ):
        m += 1
        probs.append(probs[-1] * lam / m)
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

    mutated = torch.zeros_like(counts)

    max_m = int(poisson_probs.numel()) - 1

    for m in range(max_m + 1):
        if m < last:
            mutated[:, m:last] += counts[:, : last - m] * poisson_probs[m]

    mutated[:, last] = counts @ tail_by_source

    return mutated


def torch_multinomial_counts_batch(N, probs, generator=None):
    runs, num_classes = probs.shape

    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True)

    samples = torch.multinomial(
        probs,
        num_samples=N,
        replacement=True,
        generator=generator,
    )

    offsets = (
        torch.arange(runs, device=probs.device, dtype=torch.long)
        .unsqueeze(1)
        * num_classes
    )

    flat_samples = (samples + offsets).reshape(-1)

    counts = torch.bincount(
        flat_samples,
        minlength=runs * num_classes,
    ).reshape(runs, num_classes)

    return counts


def build_selection_vector(alpha, num_classes, selection_mode, device, dtype):
    k = torch.arange(num_classes, device=device, dtype=dtype)

    if selection_mode == "exponential":
        return torch.exp(-alpha * k)

    if selection_mode == "one_minus_alpha_power":
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                "For selection_mode='one_minus_alpha_power', alpha must satisfy 0 < alpha < 1."
            )

        return (1.0 - alpha) ** k

    raise ValueError(
        "selection_mode must be 'exponential' or 'one_minus_alpha_power'."
    )


def one_wright_fisher_generation(
    counts,
    N,
    poisson_probs,
    tail_by_source,
    selection,
    generator=None,
):
    mutated = mutate_counts_poisson_shift(
        counts=counts,
        poisson_probs=poisson_probs,
        tail_by_source=tail_by_source,
    )

    probs = mutated * selection
    probs = probs / probs.sum(dim=1, keepdim=True)

    new_counts = torch_multinomial_counts_batch(
        N=N,
        probs=probs,
        generator=generator,
    )

    return new_counts


def calc_observables(counts, kvec, N):
    x0_to_x3 = counts[:, :4].to(kvec.dtype) / float(N)
    kappa_1 = (counts.to(kvec.dtype) @ kvec) / float(N)

    return torch.cat(
        [
            x0_to_x3,
            kappa_1.unsqueeze(1),
        ],
        dim=1,
    )


# ============================================================
# Output helpers
# ============================================================

def make_timeseries_dataframe(
    values_cpu,
    run_indices,
    time,
):
    return pd.DataFrame(
        {
            "run_index": run_indices,
            "time": int(time),
            "X_0": values_cpu[:, 0],
            "X_1": values_cpu[:, 1],
            "X_2": values_cpu[:, 2],
            "X_3": values_cpu[:, 3],
            "kappa_1": values_cpu[:, 4],
        }
    )


def save_metadata(path, selection_mode):
    metadata = pd.DataFrame(
        [
            {
                "N": N,
                "T": T,
                "runs": RUNS,
                "batch_size": BATCH_SIZE,
                "lambda": LAM,
                "alpha": ALPHA,
                "theta": THETA,
                "num_classes": NUM_CLASSES,
                "selection_mode": selection_mode,
                "seed": SEED,
            }
        ]
    )

    metadata.to_csv(path, index=False)
    print(f"Saved metadata: {path}")


# ============================================================
# Simulation for one selection mode
# ============================================================

def simulate_selection_mode(selection_mode, device, seed):
    print("\n" + "=" * 80)
    print(f"Running selection mode: {selection_mode}")
    print("=" * 80)

    output_prefix = (
        f"N_{N}"
        f"_T_{T}"
        f"_lambda_{float_tag(LAM)}"
        f"_alpha_{float_tag(ALPHA)}"
        f"_runs_{RUNS}"
        f"_selection_{selection_mode}"
    )

    timeseries_path = os.path.join(
        OUTPUT_DIR,
        f"timeseries_by_run_{output_prefix}.csv",
    )

    means_path = os.path.join(
        OUTPUT_DIR,
        f"timeseries_means_{output_prefix}.csv",
    )

    metadata_path = os.path.join(
        OUTPUT_DIR,
        f"metadata_{output_prefix}.csv",
    )

    remove_if_exists(timeseries_path)
    remove_if_exists(means_path)
    remove_if_exists(metadata_path)

    save_metadata(metadata_path, selection_mode)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    init_probs = poisson_lumped_probs(
        theta=THETA,
        num_classes=NUM_CLASSES,
        device=device,
        dtype=TORCH_DTYPE,
    )

    poisson_probs, tail_by_source = build_poisson_mutation_kernel(
        lam=LAM,
        num_classes=NUM_CLASSES,
        device=device,
        dtype=TORCH_DTYPE,
        tail_tol=MUTATION_TAIL_TOL,
        max_mutations_cap=MAX_MUTATIONS_CAP,
    )

    selection = build_selection_vector(
        alpha=ALPHA,
        num_classes=NUM_CLASSES,
        selection_mode=selection_mode,
        device=device,
        dtype=TORCH_DTYPE,
    )

    kvec = torch.arange(
        NUM_CLASSES,
        device=device,
        dtype=TORCH_DTYPE,
    )

    sums = np.zeros((T + 1, 5), dtype=np.float64)
    sq_sums = np.zeros((T + 1, 5), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, RUNS, BATCH_SIZE):
            m = min(BATCH_SIZE, RUNS - start)

            run_indices = np.arange(start, start + m, dtype=np.int64)

            probs0 = init_probs.unsqueeze(0).repeat(m, 1)

            counts = torch_multinomial_counts_batch(
                N=N,
                probs=probs0,
                generator=generator,
            )

            chunk_frames = []

            for t in range(T + 1):
                values = calc_observables(
                    counts=counts,
                    kvec=kvec,
                    N=N,
                )

                values_cpu = values.detach().cpu().numpy().astype(np.float64)

                sums[t, :] += values_cpu.sum(axis=0)
                sq_sums[t, :] += np.square(values_cpu).sum(axis=0)

                df_t = make_timeseries_dataframe(
                    values_cpu=values_cpu,
                    run_indices=run_indices,
                    time=t,
                )

                chunk_frames.append(df_t)

                if (
                    len(chunk_frames) >= WRITE_EVERY_GENERATIONS
                    or t == T
                ):
                    chunk_df = pd.concat(chunk_frames, ignore_index=True)
                    append_dataframe_csv(chunk_df, timeseries_path)
                    chunk_frames = []

                if t < T:
                    counts = one_wright_fisher_generation(
                        counts=counts.to(TORCH_DTYPE),
                        N=N,
                        poisson_probs=poisson_probs,
                        tail_by_source=tail_by_source,
                        selection=selection,
                        generator=generator,
                    )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            print(f"Finished runs {start} to {start + m - 1}")

    means = sums / float(RUNS)

    if RUNS > 1:
        variances = (sq_sums - RUNS * np.square(means)) / float(RUNS - 1)
        variances = np.maximum(variances, 0.0)
    else:
        variances = np.zeros_like(means)

    sd = np.sqrt(variances)
    se = sd / math.sqrt(RUNS)

    means_df = pd.DataFrame(
        {
            "time": np.arange(T + 1, dtype=np.int64),

            "X_0_mean": means[:, 0],
            "X_1_mean": means[:, 1],
            "X_2_mean": means[:, 2],
            "X_3_mean": means[:, 3],
            "kappa_1_mean": means[:, 4],

            "X_0_sd": sd[:, 0],
            "X_1_sd": sd[:, 1],
            "X_2_sd": sd[:, 2],
            "X_3_sd": sd[:, 3],
            "kappa_1_sd": sd[:, 4],

            "X_0_se": se[:, 0],
            "X_1_se": se[:, 1],
            "X_2_se": se[:, 2],
            "X_3_se": se[:, 3],
            "kappa_1_se": se[:, 4],
        }
    )

    means_df.to_csv(means_path, index=False)

    print(f"Saved timeseries by run: {timeseries_path}")
    print(f"Saved timeseries means: {means_path}")

    return timeseries_path, means_path


# ============================================================
# Run
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_torch_device(require_cuda=REQUIRE_CUDA)

    output_paths = []

    for mode_index, selection_mode in enumerate(SELECTION_MODES):
        paths = simulate_selection_mode(
            selection_mode=selection_mode,
            device=device,
            seed=SEED + mode_index,
        )

        output_paths.append(paths)

    print("\nDone.")
    print("Output files:")

    for timeseries_path, means_path in output_paths:
        print(timeseries_path)
        print(means_path)


if __name__ == "__main__":
    main()
