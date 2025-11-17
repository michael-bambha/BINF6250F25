# Introduction
Project 08 -- Viterbi, Forward, Backward, Forward-Backward,
and Baum-Welch algorithms for HMMs

# Pseudocode

## Forward:
```
We will keep the initialization the same from Viterbi:
    
    - first col of fwd = init probs * emission probs at t[0]
        ^ for each state s, find prob of starting in state S and
        observing the observation (great wording I know)
    - then for t in range(1:end):
        - for each state s:
            - initialize our prob
            - incrementally add to the prob the previous fwd mtx *
            trans probs
        - update fwd[s, t] as emission probs * prob from earlier
    
    - termination - sum over all states s of fwd[s, t] 


```
## Backward:

```
Now basically we do the same thing as fwd, but we iterate in
reverse. Can use reversed() for this.

will need to initialize to 1 (or to 0 for log)
and combine over next states instead of previous.
```

## Forward-Backward:

```
Call forward() and backward().
Multiply forward and backward mtxs
Then normalize to a conditional probability (as multiplying gives us joint)
```

## Baum-Welch:

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

```
"E" Step:
1) We need to call foward() and backward() to get alpha and beta.
2) Compute gamma (i) = posterior probability that the model is in state i
at time t.
3) Now we need another probability (ksi) - the posterior probability
that at time t, the model is in state i, AND at time t+1, it moved
to state j, GIVEN the observation.
"M" Step:
1) SET pi(i)* to gamma(i)(1) -- pi is initial probs
2) SET a(i,j)* to sum from t=1 to end: gamma(t)(i,j)/sum t=1:end gamma(t)(i)
3) SET b(j)(k)* to sum from t=1:end: gamma(t)(j) when O(t) = k for all k / sum(all t) gamma(t)(j)

In less math terms:

fwd() and bwd() produce 2 arrays of alpha(i,j) for all i->j
To get "ksi":
    - we can iterate across all t (or use np):
        - Get the forward and backward variable at t and t+1, respectively
        - Multiply: fwd(t) for state i, bwd(t+1) for j, trans(i->j), emission prob of 
        y(at t+1) for state j

Remaining calculations are going to depend on our posterior (gamma)
and ksi.


```

# Successes
We refactored the code to be class-based which was incredibly
helpful for implementations of multiple algorithms, and we have
implemented Viterbi, forward, backward, and forward-backward algorithms 
successfully.

# Struggles
One of the biggest issues was code organization, structure, and making something that is extendable
for future usage. One issue is handling log space, which was not necessarily conceptually challenging,
but from a clean code perspective was hard to implement. Initially we had if statements at each calculation,
which was becoming cumbersome and also just hard to read. We ended up solving this with
a neat design pattern (see AI note below). Beyond that, most of the struggles were more
regarding the mathematics of the algorithms themselves - while the concepts of what the 
algorithms took in, what they output, and what they're used for wasn't too complicated, understanding the subtelties of the
underlying math was more tricky.

Baum-Welch: This algorithm is honestly extremely math-heavy and implementing it was a challenge.
Part of the issue is that the "ksi" matrix is actually a 3D array and broadcasting operations over 3
dimensions became increasingly confusing to keep track of. Conceptually it didn't seem that
hard to understand what was going on, but specifically trying to vectorize multiple
calculations over several dimensions at once is hard to do in code. We ran into several
numpy broadcasting errors saying the shapes of the two arrays are not compatible.
Generally this was due to forgetting to explicitly reshape, or summing over
the incorrect axes.

# Personal Reflections
## Group Leader
Forward/Backward implementations were not super complex. We simply just
reused the logic from the Viterbi algorithm for the general 
dynamic programming approach, just with removing the traceback
and replacing max/argmax with our `sum_states` function. I realized
that summing + elementwise multiplication was just doing the
matrix dot product, so that was a nice simplification. Then
the forward-backward is just calling both of our previous
implementations to compute the mpp. In terms of coding, most of the time I spent was taken
to understand how to properly broadcast matrices and vectors
(initially I forgot to reshape and it was causing problems).
However a lot of the time was spent going over the math in the iteration
steps and termination step and understanding how matrix operations are being
applied in those contexts.

I did struggle with how to handle both log and probability space,
and it's not completely perfect as-is (I still coded in some `if`
statements here and there, didn't have time to continually refactor),
but I was pretty happy with what we had in general.

Also coding in the model as a global variable bothered me from a
programming standpoint and it looked messy so I just decided to load
the parameters from JSON instead with a simple class method.

**Baum-Welch:**

This really tested my coding abilities to the maximum, I will be completely honest.
Although I had done a couple of neat broadcasting tricks like in the forward/backward algorithms
and back in the neighbor joining algorithm, trying to do this across 3 dimensions
completely broke my brain. That single for loop `for t in range(T-1)` which was 3 lines of code took me approximately
3 hours of coding to get the dimensions correct.

## Other member
Other members' reflections on the project

# Generative AI Appendix
ChatGPT-5 was used to assist with some organizational and
design questions related to handling both log and probability space.
We found it was overly verbose to continually use `if` statements
to check if we were in log space constantly - AI assisted in 
a clean way to handle this by simply adding attributes in the
constructor to handle this.

Q. I am working on a project related to HMMs, and I need to handle
both log and probability space. We currently are using several
`if` statements, such as `if use_log_space=True ...` to handle
models with log space, but it is starting to become messy as this
needs to be done in every calculation step. Can you give some design
patterns to centralize or clean up this logic?

Additionally we used ChatGPT-5 to generate the LaTeX formatting
for the equations - this was mainly so we could keep a markdown file
of the math handy while we worked on the implementation to save some alt-tabbing effort.
