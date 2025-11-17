Forward variable

$$
\alpha_t(i) = P(O_1,\dots,O_t,, q_t=i)
$$

Backward variable

$$
\beta_t(i) = P(O_{t+1},\dots,O_T \mid q_t=i)
$$

⸻

E-step

State posterior probability

$$
\gamma_t(i) =
\frac{\alpha_t(i),\beta_t(i)}
{\sum_{k=1}^N \alpha_t(k)\beta_t(k)}
$$

Transition posterior probability

$$
\xi_t(i,j) =
\frac{
\alpha_t(i), a_{ij}, b_j(O_{t+1}), \beta_{t+1}(j)
}{
\sum_{i’=1}^N \sum_{j’=1}^N
\alpha_t(i’), a_{i’j’}, b_{j’}(O_{t+1}), \beta_{t+1}(j’)
}
$$

⸻

M-step

Initial distribution update

$$
\pi_i^{\text{new}} = \gamma_1(i)
$$

Transition matrix update

$$
a_{ij}^{\text{new}} =
\frac{
\sum_{t=1}^{T-1} \xi_t(i,j)
}{
\sum_{t=1}^{T-1} \gamma_t(i)
}
$$

Emission matrix update

$$
b_j(k)^{\text{new}} =
\frac{
\sum_{t=1}^{T} \gamma_t(j), \mathbf{1}{{O_t = k}}
}{
\sum{t=1}^{T} \gamma_t(j)
}
$$

⸻

Iteration Loop
	1.	Initialize:
            $$pi, A, B$$
	2.	Compute $$\alpha_t(i)$$ using the forward algorithm
	3.	Compute $$\beta_t(i)$$ using the backward algorithm
	4.	Compute $$\gamma_t(i)$$ and $$\xi_t(i,j)$$
	5.	Update parameters using the M-step
	6.	Compute log-likelihood
$$
\log P(O \mid \lambda) = \log\left( \sum_{i=1}^N \alpha_T(i) \right)
$$
	7.	Repeat until convergence
