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
