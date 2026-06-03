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

Simulation_discrete_time.py: Simulates the discrete time version of the ratchet on a GPU (if available). We get $m_1, X_0(t), Cov(X_0(t), m1_(t))$ and the trajectory of the connection between $m_1(t)$ and $X_0(t)$. <br>
Simulation_discrete_time_Fig_1.py: Simulates the data for Figure 1. <br>
Plots:  Code to create the plots. <br>
ratchet_discrete_psi_approx_delta_stop_at_t0_results_large_N_different_psi : output of the python files <br>
ratchet_discrete_psi_delta_results: <br>
ratchet_discrete_psi_delta_stop_at_t0_results: output of the python files <br>
ratchet_discrete_psi_delta_stop_at_t0_results_large_N: output of the python files <br>
ratchet_discrete_psi_delta_stop_at_t0_results_large_N_different_psi: output of the python files <br>


### Funding Acknowledgement

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project-ID 499552394 – SFB 1597.
