import os
import math
import numpy as np
import pandas as pd
import torch


#  Parameters

N_VALUES = [10000]
PSI_VALUES = [1.0, 5.0, 10.0]
DELTA_VALUES = [round(0.1 * i, 1) for i in range(1, 10)]

RUNS = 100
BATCH_SIZE = 100
SEED = 22

T_MAX = 1000
STOP_MULTIPLIER_AFTER_T0 = 5

OUTPUT_DIR = "ratchet_discrete_psi_delta_results"

REQUIRE_CUDA = True
TORCH_DTYPE = torch.float32

#  "exponential" fitness =  exp(-\alpha k).
#  "one_minus_alpha_power": fitness =  (1-alpha)^k.
SELECTION_MODE = "exponential"

MUTATION_TAIL_TOL = 1e-12
MAX_MUTATIONS_CAP = 250


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
    theta = delta * math.log(N)
    alpha = psi * math.exp(theta) / N
    lam = alpha * theta

    effective_beta = -math.log(lam) / math.log(N)

    psi_check = N * alpha * math.exp(-theta)

    return {
        "theta": theta,
        "alpha": alpha,
        "lambda": lam,
        "effective_beta": effective_beta,
        "psi_check": psi_check,
    }


# File helpers

def float_tag(x):
    return f"{x:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def kappa_filename(N, psi, delta):
    return (
        f"kappa_timeseries"
        f"_N_{int(N)}"
        f"_psi_{float_tag(psi)}"
        f"_delta_{float_tag(delta)}.csv"
    )


def x0_filename(N, psi, delta):
    return (
        f"x0_until_t0"
        f"_N_{int(N)}"
        f"_psi_{float_tag(psi)}"
        f"_delta_{float_tag(delta)}.csv"
    )


def first_loss_filename(N, psi, delta):
    return (
        f"first_loss_events"
        f"_N_{int(N)}"
        f"_psi_{float_tag(psi)}"
        f"_delta_{float_tag(delta)}.csv"
    )


def summary_filename(N, psi, delta):
    return (
        f"summary"
        f"_N_{int(N)}"
        f"_psi_{float_tag(psi)}"
        f"_delta_{float_tag(delta)}.csv"
    )


def save_dataframe(df, path):
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# Wright-Fisher Diffusion

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


def build_selection_vector(alpha, num_classes, device, dtype):
    k = torch.arange(num_classes, device=device, dtype=dtype)

    if SELECTION_MODE == "exponential":
        return torch.exp(-alpha * k)

    if SELECTION_MODE == "one_minus_alpha_power":
        if alpha >= 1.0:
            raise ValueError(
                f"alpha={alpha} is not valid for selection=(1-alpha)^k. "
                "Use SELECTION_MODE='exponential' or choose smaller alpha."
            )

        return (1.0 - alpha) ** k

    raise ValueError(
        "SELECTION_MODE must be 'exponential' or 'one_minus_alpha_power'."
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


def calc_kappa_from_counts(counts, kvec, N):
    return (counts.to(kvec.dtype) @ kvec) / float(N)


def leading_zero_count_from_counts(counts):
    positive = counts > 0
    return torch.argmax(positive.to(torch.long), dim=1)


# Record helpers

def flag_to_numpy(flag, record_indices):
    n = int(record_indices.numel())

    if isinstance(flag, torch.Tensor):
        return flag[record_indices].detach().cpu().numpy().astype(bool)

    return np.full(n, bool(flag), dtype=bool)


def make_kappa_records(
    counts,
    kvec,
    record_indices,
    global_path_indices,
    current_time,
    is_final_record,
    N,
    psi,
    delta,
    lam,
    theta,
    alpha,
    effective_beta,
    psi_check,
):
    if record_indices.numel() == 0:
        return None

    kappa_1 = (
        calc_kappa_from_counts(
            counts=counts[record_indices],
            kvec=kvec,
            N=N,
        )
        .detach()
        .cpu()
        .numpy()
    )

    record_indices_cpu = record_indices.detach().cpu().numpy()
    path_index = global_path_indices[record_indices_cpu]

    return pd.DataFrame(
        {
            "N": int(N),
            "psi": float(psi),
            "delta": float(delta),
            "lambda": float(lam),
            "theta": float(theta),
            "alpha": float(alpha),
            "effective_beta": float(effective_beta),
            "psi_check": float(psi_check),
            "selection_mode": SELECTION_MODE,
            "path_index": path_index,
            "time": int(current_time),
            "kappa_1": kappa_1,
            "is_final_record": flag_to_numpy(is_final_record, record_indices),
        }
    )


def make_x0_records(
    counts,
    record_indices,
    global_path_indices,
    current_time,
    is_first_loss_record,
    N,
    psi,
    delta,
    lam,
    theta,
    alpha,
    effective_beta,
    psi_check,
):
    if record_indices.numel() == 0:
        return None

    x0_count = counts[record_indices, 0].detach().cpu().numpy()
    x0_value = x0_count / float(N)

    record_indices_cpu = record_indices.detach().cpu().numpy()
    path_index = global_path_indices[record_indices_cpu]

    return pd.DataFrame(
        {
            "N": int(N),
            "psi": float(psi),
            "delta": float(delta),
            "lambda": float(lam),
            "theta": float(theta),
            "alpha": float(alpha),
            "effective_beta": float(effective_beta),
            "psi_check": float(psi_check),
            "selection_mode": SELECTION_MODE,
            "path_index": path_index,
            "time": int(current_time),
            "X0_count": x0_count,
            "X_0": x0_value,
            "is_first_loss_record": flag_to_numpy(
                is_first_loss_record,
                record_indices,
            ),
        }
    )


def make_first_loss_records(
    counts,
    kvec,
    record_indices,
    global_path_indices,
    current_time,
    N,
    psi,
    delta,
    lam,
    theta,
    alpha,
    effective_beta,
    psi_check,
):
    if record_indices.numel() == 0:
        return None

    counts_selected = counts[record_indices]

    kappa_1 = (
        calc_kappa_from_counts(
            counts=counts_selected,
            kvec=kvec,
            N=N,
        )
        .detach()
        .cpu()
        .numpy()
    )

    leading_zero_count = (
        leading_zero_count_from_counts(counts_selected)
        .detach()
        .cpu()
        .numpy()
    )

    x0_count = counts_selected[:, 0].detach().cpu().numpy()
    x0_value = x0_count / float(N)

    record_indices_cpu = record_indices.detach().cpu().numpy()
    path_index = global_path_indices[record_indices_cpu]

    return pd.DataFrame(
        {
            "N": int(N),
            "psi": float(psi),
            "delta": float(delta),
            "lambda": float(lam),
            "theta": float(theta),
            "alpha": float(alpha),
            "effective_beta": float(effective_beta),
            "psi_check": float(psi_check),
            "selection_mode": SELECTION_MODE,
            "path_index": path_index,
            "t_0": int(current_time),
            "first_loss_time": int(current_time),
            "X0_count_at_t0": x0_count,
            "X_0_at_t0": x0_value,
            "kappa_1_at_t0": kappa_1,
            "leading_zero_count_at_t0": leading_zero_count,
            "first_nonzero_class_at_t0": leading_zero_count,
        }
    )


# Simulation for one parameter set

def simulate_parameter_set(
    N,
    psi,
    delta,
    runs,
    batch_size,
    t_max,
    stop_multiplier_after_t0,
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

    theta = params["theta"]
    alpha = params["alpha"]
    lam = params["lambda"]
    effective_beta = params["effective_beta"]
    psi_check = params["psi_check"]

    num_classes = N

    print(
        f"Running N={N}, psi={psi}, delta={delta}, "
        f"theta={theta:.8g}, alpha={alpha:.8g}, lambda={lam:.8g}, "
        f"effective_beta={effective_beta:.8g}, psi_check={psi_check:.8g}, "
        f"selection_mode={SELECTION_MODE}"
    )

    try:
        selection = build_selection_vector(
            alpha=alpha,
            num_classes=num_classes,
            device=device,
            dtype=dtype,
        )
    except ValueError as error:
        print(f"Skipping parameter set: {error}")
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    init_probs = poisson_lumped_probs(
        theta=theta,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
    )

    poisson_probs, tail_by_source = build_poisson_mutation_kernel(
        lam=lam,
        num_classes=num_classes,
        device=device,
        dtype=dtype,
        tail_tol=MUTATION_TAIL_TOL,
        max_mutations_cap=MAX_MUTATIONS_CAP,
    )

    kvec = torch.arange(num_classes, device=device, dtype=dtype)

    kappa_frames = []
    x0_frames = []
    first_loss_frames = []
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

            hit = counts[:, 0] == 0

            first_loss_time = torch.full(
                (m,),
                -1,
                device=device,
                dtype=torch.long,
            )

            stop_time = torch.full(
                (m,),
                int(t_max),
                device=device,
                dtype=torch.long,
            )

            leading_zero_count_at_t0 = torch.full(
                (m,),
                -1,
                device=device,
                dtype=torch.long,
            )

            kappa_1_at_t0 = torch.full(
                (m,),
                float("nan"),
                device=device,
                dtype=dtype,
            )

            done = torch.zeros(m, device=device, dtype=torch.bool)

            initial_kappa_1 = calc_kappa_from_counts(
                counts=counts,
                kvec=kvec,
                N=N,
            )

            if torch.any(hit):
                first_loss_time[hit] = 0
                stop_time[hit] = 0
                done[hit] = True

                leading_zero_count_at_t0[hit] = leading_zero_count_from_counts(
                    counts[hit]
                )

                kappa_1_at_t0[hit] = initial_kappa_1[hit]

            all_indices = torch.arange(m, device=device)

            frame = make_kappa_records(
                counts=counts,
                kvec=kvec,
                record_indices=all_indices,
                global_path_indices=global_path_indices,
                current_time=0,
                is_final_record=done,
                N=N,
                psi=psi,
                delta=delta,
                lam=lam,
                theta=theta,
                alpha=alpha,
                effective_beta=effective_beta,
                psi_check=psi_check,
            )
            kappa_frames.append(frame)

            frame = make_x0_records(
                counts=counts,
                record_indices=all_indices,
                global_path_indices=global_path_indices,
                current_time=0,
                is_first_loss_record=hit,
                N=N,
                psi=psi,
                delta=delta,
                lam=lam,
                theta=theta,
                alpha=alpha,
                effective_beta=effective_beta,
                psi_check=psi_check,
            )
            x0_frames.append(frame)

            if torch.any(hit):
                hit_indices = torch.nonzero(hit, as_tuple=False).flatten()

                frame = make_first_loss_records(
                    counts=counts,
                    kvec=kvec,
                    record_indices=hit_indices,
                    global_path_indices=global_path_indices,
                    current_time=0,
                    N=N,
                    psi=psi,
                    delta=delta,
                    lam=lam,
                    theta=theta,
                    alpha=alpha,
                    effective_beta=effective_beta,
                    psi_check=psi_check,
                )
                first_loss_frames.append(frame)

            t = 0

            while t < t_max and not torch.all(done):
                active = ~done
                active_indices = torch.nonzero(active, as_tuple=False).flatten()

                counts_active = counts[active_indices].to(dtype)

                counts_new_active = one_wright_fisher_generation(
                    counts=counts_active,
                    N=N,
                    poisson_probs=poisson_probs,
                    tail_by_source=tail_by_source,
                    selection=selection,
                    generator=generator,
                )

                counts[active_indices] = counts_new_active

                t += 1

                active = ~done

                x0_record_mask = active & (~hit)

                new_hits = x0_record_mask & (counts[:, 0] == 0)

                current_kappa_1 = calc_kappa_from_counts(
                    counts=counts,
                    kvec=kvec,
                    N=N,
                )

                if torch.any(new_hits):
                    first_loss_time[new_hits] = t

                    new_stop_time = min(
                        int(stop_multiplier_after_t0 * t),
                        int(t_max),
                    )

                    stop_time[new_hits] = new_stop_time
                    hit[new_hits] = True

                    leading_zero_count_at_t0[new_hits] = (
                        leading_zero_count_from_counts(counts[new_hits])
                    )

                    kappa_1_at_t0[new_hits] = current_kappa_1[new_hits]

                    new_hit_indices = torch.nonzero(
                        new_hits,
                        as_tuple=False,
                    ).flatten()

                    frame = make_first_loss_records(
                        counts=counts,
                        kvec=kvec,
                        record_indices=new_hit_indices,
                        global_path_indices=global_path_indices,
                        current_time=t,
                        N=N,
                        psi=psi,
                        delta=delta,
                        lam=lam,
                        theta=theta,
                        alpha=alpha,
                        effective_beta=effective_beta,
                        psi_check=psi_check,
                    )
                    first_loss_frames.append(frame)

                final_record = active & (t >= stop_time)

                kappa_record_indices = torch.nonzero(
                    active,
                    as_tuple=False,
                ).flatten()

                frame = make_kappa_records(
                    counts=counts,
                    kvec=kvec,
                    record_indices=kappa_record_indices,
                    global_path_indices=global_path_indices,
                    current_time=t,
                    is_final_record=final_record,
                    N=N,
                    psi=psi,
                    delta=delta,
                    lam=lam,
                    theta=theta,
                    alpha=alpha,
                    effective_beta=effective_beta,
                    psi_check=psi_check,
                )
                kappa_frames.append(frame)

                x0_record_indices = torch.nonzero(
                    x0_record_mask,
                    as_tuple=False,
                ).flatten()

                frame = make_x0_records(
                    counts=counts,
                    record_indices=x0_record_indices,
                    global_path_indices=global_path_indices,
                    current_time=t,
                    is_first_loss_record=new_hits,
                    N=N,
                    psi=psi,
                    delta=delta,
                    lam=lam,
                    theta=theta,
                    alpha=alpha,
                    effective_beta=effective_beta,
                    psi_check=psi_check,
                )

                if frame is not None:
                    x0_frames.append(frame)

                done[final_record] = True

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            final_kappa_1 = calc_kappa_from_counts(
                counts=counts,
                kvec=kvec,
                N=N,
            )

            final_x0_count = counts[:, 0]
            final_x0 = final_x0_count.to(dtype) / float(N)

            hit_cpu = hit.detach().cpu().numpy().astype(bool)

            first_loss_time_cpu = first_loss_time.detach().cpu().numpy()
            first_loss_time_float = first_loss_time_cpu.astype(float)
            first_loss_time_float[first_loss_time_cpu < 0] = np.nan

            stop_time_cpu = stop_time.detach().cpu().numpy()

            leading_zero_cpu = leading_zero_count_at_t0.detach().cpu().numpy()
            leading_zero_float = leading_zero_cpu.astype(float)
            leading_zero_float[leading_zero_cpu < 0] = np.nan

            kappa_t0_cpu = kappa_1_at_t0.detach().cpu().numpy()
            final_kappa_cpu = final_kappa_1.detach().cpu().numpy()
            final_x0_cpu = final_x0.detach().cpu().numpy()
            final_x0_count_cpu = final_x0_count.detach().cpu().numpy()

            summary_df = pd.DataFrame(
                {
                    "N": int(N),
                    "psi": float(psi),
                    "delta": float(delta),
                    "lambda": float(lam),
                    "theta": float(theta),
                    "alpha": float(alpha),
                    "effective_beta": float(effective_beta),
                    "psi_check": float(psi_check),
                    "selection_mode": SELECTION_MODE,
                    "path_index": global_path_indices,
                    "hit_x0_zero": hit_cpu,
                    "t_0": first_loss_time_float,
                    "first_loss_time": first_loss_time_float,
                    "stop_time": stop_time_cpu,
                    "simulated_until": stop_time_cpu,
                    "simulated_until_rule": f"min({stop_multiplier_after_t0} * t_0, {t_max})",
                    "leading_zero_count_at_t0": leading_zero_float,
                    "first_nonzero_class_at_t0": leading_zero_float,
                    "kappa_1_at_t0": kappa_t0_cpu,
                    "final_kappa_1": final_kappa_cpu,
                    "final_X0_count": final_x0_count_cpu,
                    "final_X_0": final_x0_cpu,
                    "num_classes": int(num_classes),
                    "T_MAX": int(t_max),
                }
            )

            summary_frames.append(summary_df)

            print(
                f"  Finished paths {start} to {start + m - 1}; "
                f"hits by stop: {int(hit_cpu.sum())}/{m}"
            )

    kappa_df = pd.concat(kappa_frames, ignore_index=True)
    x0_df = pd.concat(x0_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)

    if first_loss_frames:
        first_loss_df = pd.concat(first_loss_frames, ignore_index=True)
    else:
        first_loss_df = pd.DataFrame(
            columns=[
                "N",
                "psi",
                "delta",
                "lambda",
                "theta",
                "alpha",
                "effective_beta",
                "psi_check",
                "selection_mode",
                "path_index",
                "t_0",
                "first_loss_time",
                "X0_count_at_t0",
                "X_0_at_t0",
                "kappa_1_at_t0",
                "leading_zero_count_at_t0",
                "first_nonzero_class_at_t0",
            ]
        )

    merge_cols = [
        "path_index",
        "hit_x0_zero",
        "t_0",
        "stop_time",
        "leading_zero_count_at_t0",
        "first_nonzero_class_at_t0",
    ]

    kappa_df = kappa_df.merge(
        summary_df[merge_cols],
        on="path_index",
        how="left",
    )

    x0_df = x0_df.merge(
        summary_df[merge_cols],
        on="path_index",
        how="left",
    )

    kappa_path = os.path.join(output_dir, kappa_filename(N, psi, delta))
    x0_path = os.path.join(output_dir, x0_filename(N, psi, delta))
    first_loss_path = os.path.join(output_dir, first_loss_filename(N, psi, delta))
    summary_path = os.path.join(output_dir, summary_filename(N, psi, delta))

    save_dataframe(kappa_df, kappa_path)
    save_dataframe(x0_df, x0_path)
    save_dataframe(first_loss_df, first_loss_path)
    save_dataframe(summary_df, summary_path)

    return summary_df


# Run all parameter sets

def run_all_parameter_sets(
    parameter_sets,
    output_dir,
    runs,
    batch_size,
    t_max,
    stop_multiplier_after_t0,
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
                "theta": derived["theta"],
                "alpha": derived["alpha"],
                "lambda": derived["lambda"],
                "effective_beta": derived["effective_beta"],
                "psi_check": derived["psi_check"],
            }
        )

    device = get_torch_device(require_cuda=require_cuda)

    all_summaries = []

    for run_index, params in enumerate(parameter_sets):
        summary_df = simulate_parameter_set(
            N=int(params["N"]),
            psi=float(params["psi"]),
            delta=float(params["delta"]),
            runs=runs,
            batch_size=batch_size,
            t_max=t_max,
            stop_multiplier_after_t0=stop_multiplier_after_t0,
            output_dir=output_dir,
            seed=seed + run_index,
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
        stop_multiplier_after_t0=STOP_MULTIPLIER_AFTER_T0,
        seed=SEED,
        require_cuda=REQUIRE_CUDA,
        dtype=TORCH_DTYPE,
    )

    print(combined_summary)
