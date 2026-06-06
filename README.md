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

`scripts/`: the general simulation code. <br>
&nbsp;&nbsp;`Simulation_discrete_time.py`: Simulates the discrete-time ratchet on a GPU (if available) and records the first-loss time $t_0$ per path. Writes its output to `Fig_2_Table_1/`. <br>
&nbsp;&nbsp;`Simulation_discrete_time_Fig_1.py`: Simulates the data for Figure 1 (full per-generation timeseries of $m_1, X_0(t), \dots$). Writes its output to `Fig_1/`. <br>
`Fig_1/`: data for Figure 1 plus `Plot_Figure_1.py`, which compares the simulated $m_1$ and $X_k$ against the analytic theory up to $\mathcal O(1/N^2)$. <br>
`Fig_2_Table_1/`: the simulation summaries plus the scripts that build Figure 2, Table 1 and the interactive web view: `make_figure.py`, `make_figure_2.py`, `make_table.py` and `build_data.py` (which writes `data.js`). <br>
`theory/`: LaTeX notes (`neutral.tex`, `wf_equilibria.tex`). <br>

The generated figures (`figure_*.pdf/png`) and the table (`table_N10000.*`) are written into `Fig_2_Table_1/`; `Plot_Figure_1.py` writes its plots into `Fig_1/`. Only `data.js` is written to the repository root, where `index.html` renders it as the interactive plot. <br>


### Funding Acknowledgement

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 499552394 – SFB 1597.
