# Introduction
Project 08 -- Viterbi, Forward, Backward, Forward-Backward,
and Baum-Welch algorithms for HMMs

# Pseudocode
Put pseudocode in this box:

```
Some pseudocode here
```

# Successes
We refactored the code to be class-based which was incredibly
helpful for implementations of multiple algorithms.

# Struggles
Description of the stumbling blocks the team experienced

# Personal Reflections
## Group Leader

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
