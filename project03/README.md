## Project 3: Gibbs Sampling

## Pseudocode

```
## MAIN FUNCTION
FUNCTION GibbsMotifFinder(seqs, k, seed=42):
    """
    Args:
        seqs: list of DNA sequences (strings)
        k: motif length
        seed: random seed for reproducibility
    Returns:
        Final PFM (4 x k matrix)
    """

    # Step 0: Set random seed so results can be reproduced
    SET numpy random seed to seed

    # Step 1: Initialize motifs randomly
    Motifs ← initialize_motifs(seqs, k)

    # Step 2: Iterative sampling
    FOR iteration = 1 to MAX_ITERS (e.g., 10000):

        # Step 2a: Randomly choose one sequence to leave out
        i ← choose_random_sequence_index(seqs)

        # Step 2b: Build PWM from all motifs except the ith one
        pwm ← build_pwm_from_motifs(Motifs, i, k)

        # Step 2c: Score every possible k-mer in seqs[i]
        Positions, Scores ← score_all_kmers(seqs[i], k, pwm)

        # Step 2d: Probabilistically select a new motif for seqs[i]
        new_motif ← sample_new_motif(seqs[i], Positions, Scores, k)

        # Step 2e: Update motif list
        Motifs[i] ← new_motif

        # Step 2f (optional): Track convergence by measuring information content
        IF iteration % CHECK_INTERVAL == 0:
            pfm ← build_pfm(Motifs, k)
            IC ← pfm_ic(pfm)
            PRINT "Iteration:", iteration, "IC:", IC

    # Step 3: Construct final PFM from all motifs
    FinalPFM ← build_pfm(Motifs, k)

    RETURN FinalPFM
```
```
##INITIALIZATION
FUNCTION initialize_motifs(seqs, k):
    """
    Randomly select a starting k-mer from each sequence.
    This forms the initial guess for the motif positions.
    """
    Motifs ← empty list
    FOR each sequence in seqs:
        pos ← random integer between 0 and (len(sequence) - k)
        kmer ← substring(sequence, pos, pos+k)
        Append kmer to Motifs
    RETURN Motifs
```
```
## RANDOM SEQUENCE CHOICE
FUNCTION choose_random_sequence_index(seqs):
    """
    Randomly choose which sequence to update this iteration.
    """
    N ← length of seqs
    i ← random integer between 0 and N-1
    RETURN i
```
```
## PWM CONSTRUCTION
FUNCTION build_pwm_from_motifs(Motifs, leave_out_index, k):
    """
    Build a probability weight matrix from all motifs except one.
    This ensures the left-out sequence is re-scored independently.
    """
    Subset ← Motifs with Motifs[leave_out_index] removed
    pfm ← build_pfm(Subset, k)    # provided helper
    pwm ← build_pwm(pfm)          # provided helper
    RETURN pwm
```
```
##SCORING KMERS
FUNCTION score_all_kmers(sequence, k, pwm):
    """
    For each possible position in the sequence:
      - extract k-mer
      - compute PWM score for forward strand
      - compute PWM score for reverse complement strand
      - record both scores separately
    """
    Positions ← empty list
    Scores ← empty list

    FOR m from 0 to (len(sequence) - k):
        kmer ← substring(sequence, m, m+k)
        rev_kmer ← reverse_complement(kmer)   # provided helper

        score_fwd ← score_kmer(kmer, pwm)     # provided helper
        score_rev ← score_kmer(rev_kmer, pwm)

        Append (m, "fwd") to Positions; Append score_fwd to Scores
        Append (m, "rev") to Positions; Append score_rev to Scores

    RETURN Positions, Scores
```
```
## SAMPLING NEW MOTIF
FUNCTION sample_new_motif(sequence, Positions, Scores, k):
    """
    Select a new motif position and strand probabilistically.
    Higher-scoring motifs are more likely to be chosen,
    but lower-scoring ones still have a chance.
    """
    Normalize Scores so that sum(Scores) = 1
    chosen_index ← np.random.choice(range(len(Scores)), p=Scores)

    pos, strand ← Positions[chosen_index]

    IF strand == "fwd":
        RETURN substring(sequence, pos, pos+k)
    ELSE:
        RETURN reverse_complement(substring(sequence, pos, pos+k))
```
```
FUNCTION measure_information_content(Motifs, k):
    """
    Calculate the information content of the current motif set.
    Used to see if the Gibbs sampler is converging.
    """
    pfm ← build_pfm(Motifs, k)
    IC ← pfm_ic(pfm)   # provided helper
    RETURN IC
```

## Successes
We were able to successfully implement the Gibbs sampler, and we learned a lot more about numpy usage,
vectorization, and KL divergence for measuring our convergence.

![]('figures/Shine_Delgarno.png')

## Challenges
One of the biggest issues was the convergence check implementation. Originally, we looked at information content across 10 iterations and measured the absolute mean difference, but the IC wasn't changing enough to be a meaningful check. It could have just been an implementation mistake on our part, but setting the mean difference to 1e^-8 seemed to not be a great measure for convergence. Instead, we used Kullbach-Liebler divergence, which hopefully is an appropriate measure for this task. Still, finding an appropriate threshold to consider the model "converged" is somewhat arbitrary and very experimental, so that was another challenge. 


### Comments from Group Leader

While the Gibbs' algorithm is fairly straightforward, trying to determine convergence is a challenge. I had to do a lot of outside research on how we consider models "converged" mathematically, and that took me into some rabbit holes aside from information content alone. I found a couple of additional ways to measure convergence, including heuristics like Hamming distance, but ultimately settled on KL as it was pretty easy to implement. 

### Comments from Other Members
