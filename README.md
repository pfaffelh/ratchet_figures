This is the code repository for the paper *"A semigroup approach towards the rate
of Muller's ratchet"* by C. S. Heinzel, P. Pfaffelhuber and A. Wakolbinger.

## Background

We consider a Fleming-Viot process with type space $\mathbb{N}_0$, which is the solution of the system of SDEs

$$dX_k=\left(\alpha \left(\sum_{j=0}^{\infty}(j-k)X_j\right)X_k+\lambda(X_{k-1}-X_k)\right)dt+\sum_{\ell \neq k}\sqrt{\frac{1}{N}X_kX_\ell}\, dW_{k\ell}\qquad (\ast)$$

for $k=0,1,\ldots$, with $X_{-1}:=0$, $\sum_k X_k = 1$, $\lambda \geq 0$
(the mutation rate), $\alpha \geq 0$ (the selection coefficient), and
$N>0$ (which determines the speed of the system), where
$`(W_{k\ell})_{k>\ell}`$ is a family of independent Brownian motions, and $`W_{k\ell}=-W_{\ell k}`$.

The unique weak solution of $(\ast)$ is referred to as *Muller's ratchet*.
In this model, $X_k(t)$ denotes the frequency of individuals carrying
$k$ mutations. The fitness of types decreases with $k$, and the mutation
process is unidirectional (at rate $\lambda$) towards less fit individuals.
Consequently, at time $t$, the fittest class is given by

$$
K^\ast(t):=\inf\{k : X_k(t)=0\},
$$

and the map $t \mapsto K^\ast(t)$ is almost surely non-decreasing.

The most interesting question in Muller's ratchet concerns the speed of
$t \mapsto K^\ast(t)$ as $t \to \infty$, which we refer to as the clicking
rate of Muller's ratchet. We give a partial answer by calculating $\mathbf E\left(\sum_{k=1}^\infty kX_k(t) \right)$ with an error of $\mathcal O(1/N^2)$ for fixed $t>0$. Our result suggests that $\psi_N:=N\alpha e^{-\lambda/\alpha}$ is an important quantity for determining the speed of the ratchet, as long as $\psi_N \gg 1$: If $\psi_N \ll 1$, clicks occur frequently on a time-scale of $\psi_N$ generations, while it clicks rarely on that time-scale if $\pso_N \gg 1$.
## Content
This repository contains 

Each figure/table lives in its own self-contained folder (simulation + plotting code + data + generated output):

`Fig_1/`: `Simulation_discrete_time_Fig_1.py` simulates the full per-generation timeseries of $m_1, X_0(t), \dots$; `Plot_Figure_1.py` compares the simulated $m_1$ and $X_k$ against the analytic theory up to $\mathcal O(1/N^2)$ and writes its plots here. <br>
`Fig_2_Table_1/`: `Simulation_discrete_time_Fig_2.py` simulates the discrete-time ratchet on a GPU (if available) and records the first-loss time $t_0$ per path; `make_figure_A.py`, `make_figure_B.py` and `make_table.py` build Figure 2 and Table 1 (written here), and `build_data.py` writes `data.js` for the interactive web view. <br>
`theory/`: LaTeX notes (`neutral.tex`, `wf_equilibria.tex`). <br>

The generated figures (`figure_*.pdf/png`) and the table (`table_N10000.*`) are written into `Fig_2_Table_1/`; `Plot_Figure_1.py` writes its plots into `Fig_1/`. Only `data.js` is written to the repository root, where `index.html` renders it as the interactive plot. <br>

## Interactive plot

An interactive version of Figure 2 / Table 1 is available at
**https://pfaffelh.github.io/ratchet_figures/** (rendered from `data.js` by `index.html`).

## Reproducing the figures

The Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The simulations run on a GPU if available (see `requirements.txt` for installing a
CUDA-enabled PyTorch). All scripts resolve their paths relative to their own
location, so they can be invoked from the repository root.

**Figure 1** (simulation vs. analytic theory up to $\mathcal O(1/N^2)$):

```bash
python Fig_1/Simulation_discrete_time_Fig_1.py   # simulate -> Fig_1/timeseries_*, metadata_*
python Fig_1/Plot_Figure_1.py                    # plots -> Fig_1/theory_comparison_*/
```

**Figure 2 and Table 1** (clicking rate / first-loss time $t_0$ over the $(\psi,\delta)$ grid):

```bash
python Fig_2_Table_1/Simulation_discrete_time_Fig_2.py   # simulate -> Fig_2_Table_1/summary_*.csv
python Fig_2_Table_1/build_data.py                       # aggregate -> data.js (repo root)
python Fig_2_Table_1/make_figure_A.py                    # Figure 2 (psi <= 2)  -> Fig_2_Table_1/
python Fig_2_Table_1/make_figure_B.py                    # Figure 2 (psi 2/5/10) -> Fig_2_Table_1/
python Fig_2_Table_1/make_table.py                       # Table 1 -> Fig_2_Table_1/table_N10000.tex
latexmk -pdf Fig_2_Table_1/table_N10000.tex             # compile the table PDF
```

The simulation step is only needed to regenerate the raw data. Figure 2, Table 1 and
the main Figure 1 panels can be rebuilt directly from the committed data (the
`summary_*.csv` files and the Figure 1 `timeseries_means_*` / `metadata_*`). The
~0.5 GB per-run timeseries `Fig_1/timeseries_by_run_*.csv` is not committed (it
exceeds GitHub's file-size limit and is git-ignored); rerun
`Simulation_discrete_time_Fig_1.py` to regenerate it for the variance / covariance /
first-loss panels.


### Funding Acknowledgement

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 499552394 – SFB 1597.
