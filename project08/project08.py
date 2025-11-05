""""
File: project08.py
Description: BINF6250 project 8
Authors: Michael Bambha and Jason Bae
"""
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
    states = list(INIT_PROBS.keys())
    num_states = len(INIT_PROBS.keys())
    obs_len = len(observation)

    # Establish probability matrix as a dataframe, with the possible states as rows
    # and the emission indices as the columns.
    prob_df = pd.DataFrame(np.zeros((num_states, obs_len)), index=states)
    
    # Traceback array set to be empty, we will record the state with the highest 
    # probability for each emission.
    traceback_array = np.empty(obs_len, dtype=str)

    # Fill out probability matrix and traceback array
    calc_probs(states, observation, obs_len, prob_df, traceback_array)
    
    return prob_df, traceback_array

def calc_probs(states, observation, obs_len, prob_df, traceback_array):
    """
    Function to calculate the probabilities of each state-emission combination and
    record which is most probable in a traceback array. 
    """
    # 'recursion' -- all other up to the end
    for j in range(0, obs_len):
        max_prob = -1.0
        # cannot access previous state (start)
        if j == 0:
            # beginning state is the state with the biggest probability
            start_state = max(INIT_PROBS, key=INIT_PROBS.get)
            # update first observation w/ initial probs
            for k in states:
                prob_df.loc[k, j] = INIT_PROBS[k] * EMIT_PROBS[k][observation[j]]
        
        else:
            for k in states: # rows in probability dataframe
                for i in states: # transitions in each row
                    prob = prob_df.loc[i, j-1] * TRANS_PROBS[i][k]
    
                    if prob > max_prob:
                        max_prob = prob
                        best_state = i
    
                # Emission probability multiplication is constant, can do once per state.
                prob_df.loc[k, j] = EMIT_PROBS[k][observation[j]] * max_prob
                max_prob = -1.0 # reset max_prob for next row
        
        # find max within column
        maximum = prob_df[j].max() 
        # find state the max is found in
        max_state = prob_df.index[prob_df[j] == maximum][0]  
        # log state in traceback array
        traceback_array[j] = max_state
