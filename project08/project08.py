""""
File: project08.py
Description: BINF6250 project 8
Authors: Michael Bambha and Jason Bae
"""
from __future__ import annotations
import json
import numpy as np

class HiddenMarkovModel:
    """Class for storing our HMM"""
    def __init__(
            self,
            init_probs: dict[str, float],
            trans_probs: dict[str, dict[str, float]],
            emit_probs: dict[str, dict[str, float]],
            use_log_space: bool = False,
    ):
        self.use_log_space = use_log_space

        # store OG dicts before we convert to arrays
        self._init_dict = init_probs
        self._trans_dict = trans_probs
        self._emit_dict = emit_probs

        self.states = list(init_probs.keys())
        self.symbols = list(next(iter(emit_probs.values())).keys())
        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.sym_idx = {sym: i for i, sym in enumerate(self.symbols)}

        self.init_probs = np.array([init_probs[s] for s in self.states])
        self.trans_probs = np.array([
            [trans_probs[i][j] for j in self.states]
            for i in self.states
        ])
        self.emit_probs = np.array([
            [emit_probs[s][sym] for sym in self.symbols]
            for s in self.states
        ])

        # allow for both log space and prob space
        self.mul = np.add if use_log_space else np.multiply
        self.add = np.logaddexp if use_log_space else np.add
        self.zero = -np.inf if use_log_space else 0.0

    def viterbi(self, observation: str) -> tuple[np.ndarray, list[str]]:
        """
        Viterbi algorithm using vectorized array operations.

        Args:
            observation: String of observed symbols

        Returns:
            tuple of (viterbi_matrix, best_path)
        """
        obs_len = len(observation)
        num_states = len(self.states)
        obs_indices = np.array([self.sym_idx[symbol] for symbol in observation])
        viterbi = np.full((num_states, obs_len), self.zero)
        traceback = np.zeros((num_states, obs_len), dtype=int)

        # initialization
        # viterbi[s, 0] = P(state = s at t=0) * P(observation[0] | s)
        viterbi[:, 0] = self.mul(self.init_probs, self.emit_probs[:, obs_indices[0]])

        # recursion
        for t in range(1, obs_len):
            # viterbi[:, t-1] has shape (num_states,)
            # trans_probs has shape (num_states, num_states)
            # broadcasting: viterbi[:, t-1, None] -> (num_states, 1)
            # trans_scores[i,j] = viterbi[i,t-1] * P(state j | state i)
            trans_scores = self.mul(viterbi[:, t - 1][:, None], self.trans_probs)

            # best previous for each current state
            traceback[:, t] = np.argmax(trans_scores, axis=0)
            max_scores = np.max(trans_scores, axis=0)

            # * emission probs
            viterbi[:, t] = self.mul(max_scores, self.emit_probs[:, obs_indices[t]])

        # backtrack
        path_indices = np.zeros(obs_len, dtype=int)
        path_indices[-1] = np.argmax(viterbi[:, -1])

        for t in range(obs_len - 1, 0, -1):
            path_indices[t - 1] = traceback[path_indices[t], t]

        # indices to state names
        best_path = [self.states[idx] for idx in path_indices]

        return viterbi, best_path

    @classmethod
    def from_json(cls, path: str) -> HiddenMarkovModel:
        """Load HMM parameters from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str):
        """Save HMM parameters to JSON"""
        data = {
            "init_probs": self._init_dict,
            "trans_probs": self._trans_dict,
            "emit_probs": self._emit_dict
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def to_log_space(self) -> HiddenMarkovModel:
        """Convert probabilities to log space"""
        if self.use_log_space:
            return self
        log_init_probs = {state: np.log(prob) for state, prob in self._init_dict.items()}

        log_trans_probs = {
            state: {next_state: np.log(prob) for next_state, prob in transitions.items()}
            for state, transitions in self._trans_dict.items()
        }

        log_emit_probs = {
            state: {symbol: np.log(prob) for symbol, prob in emissions.items()}
            for state, emissions in self._emit_dict.items()
        }

        return HiddenMarkovModel(log_init_probs, log_trans_probs, log_emit_probs, use_log_space=True)

if __name__ == "__main__":
    INIT_PROBS = {"I": 0.1, "G": 0.9}
    TRANS_PROBS = {"I": {"I": 0.6, "G": 0.4},
                   "G": {"I": 0.1, "G": 0.9}}
    EMIT_PROBS = {"I": {"A": 0.1, "C": 0.4, "G": 0.4, "T": 0.1},
                  "G": {"A": 0.4, "C": 0.1, "G": 0.1, "T": 0.4}}

    hmm = HiddenMarkovModel(INIT_PROBS, TRANS_PROBS, EMIT_PROBS)
    hmm = hmm.to_log_space()

    obs = "ACGT"
    result = hmm.viterbi(obs)

    print(result)