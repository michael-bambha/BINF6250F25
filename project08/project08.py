""""
File: project08.py
Description: BINF6250 project 8
Authors: Michael Bambha and Jason Bae
"""
from collections import defaultdict
import numpy as np
import pandas as pd

INIT_PROBS = {
    "I": 0.1,
    "G": 0.9
}

TRANS_PROBS = {
    "I": {"I": 0.6, "G": 0.4},
    "G": {"I": 0.1, "G": 0.9}
}

EMIT_PROBS = {
    "I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
    "G": {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}
}


def viterbi(observation: str):
    num_states = len(INIT_PROBS.keys())
    obs_len = len(observation)
    prob_matrix = pd.DataFrame(np.zeros((num_states, obs_len)))
    prob_matrix.index = INIT_PROBS.keys()
    prob_matrix.columns = [char for char in observation]
    traceback = {"States": [], "Probabilities": []}
    for i in range(len(observation)):
        prob_matrix, traceback = calc_probs(i, observation, prob_matrix, traceback)
    return prob_matrix, traceback


def calc_probs(j, observation, prob_matrix, traceback):
    observed_emission = observation[j]

    traceback_prob = 0
    traceback_state = ""

    if j == 0:
        prob_calcs = defaultdict(float)
        for idx, k in enumerate(INIT_PROBS):
            state_prob = INIT_PROBS[k] * EMIT_PROBS[k][observed_emission]
            prob_calcs[k] = state_prob
            prob_matrix.iloc[idx, j] = prob_calcs[k]
            max_state, max_prob = choose_max(INIT_PROBS, k=None)
            if state_prob > traceback_prob:
                traceback_prob = state_prob
                traceback_state = max_state

        traceback["States"].append(traceback_state)
        traceback["Probabilities"].append(traceback_prob)
        return prob_matrix, traceback



    for idx, k in enumerate(TRANS_PROBS):
        prob_calcs = defaultdict(defaultdict)
        previous_calc = prob_matrix.iloc[idx, j-1]
        max_state, max_prob = choose_max(TRANS_PROBS, k)
        state_prob = max_prob * previous_calc * EMIT_PROBS[k][observed_emission]
        prob_calcs[k][max_state] = state_prob
        prob_matrix.iloc[idx, j] = prob_calcs[k][max_state]

        if state_prob > traceback_prob:
            traceback_prob = state_prob
            traceback_state = max_state

    traceback["States"].append(traceback_state)
    traceback["Probabilities"].append(traceback_prob)

    return prob_matrix, traceback


def choose_max(prob_dict, k=None):
    if not k:
        return max(prob_dict.items(), key=lambda x: x[1])
    return max(prob_dict[k].items(), key=lambda x: x[1])


# df = (viterbi("ACGCGATC")[0])
# df.to_csv("something.csv")


for idx, k in enumerate(TRANS_PROBS):
    print(idx)
    print(k)