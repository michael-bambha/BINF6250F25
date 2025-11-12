""""
File: project08.py
Description: BINF6250 project 8
Authors: Michael Bambha and Jason Bae
"""
from __future__ import annotations
import json
import numpy as np
from pprint import pformat

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

        if use_log_space:
            self.add = lambda *args: np.logaddexp(*args) if len(args) > 1 else np.logaddexp.reduce(args[0])
        else:
            self.add = lambda *args: np.add(*args) if len(args) > 1 else np.sum(args[0])
        self.zero = -np.inf if use_log_space else 0.0
        self.mul = np.add if use_log_space else np.multiply

    def __str__(self):
        return pformat(vars(self))

    def __getitem__(self, key):
        if isinstance(key, str):
            return {
                "init": self._init_dict.get(key),
                "transitions": self._trans_dict.get(key),
                "emissions": self._emit_dict.get(key),
            }

    def sum_states(self, matrix: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """
        In the forward and backward algorithms, we need different math operations
        for summing over states when in log space or probability space. We
        take the dot product of A @ B in prob space, while we need to use
        logaddexp.reduce for log space.
        :param matrix: matrix A
        :param prev: matrix B
        :return: A @ B if in probability space,
        """
        if getattr(self, "use_log_space", False):
            return np.logaddexp.reduce(matrix.T + prev[:, None], axis=0)
        # np.sum(A*B) is the same as A.T @ B in this case
        # A and B are 1d arrays of identical length
        return matrix.T @ prev

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

    def forward(self, observation) -> tuple[float, np.ndarray]:
        """
        Find the probability of seeing an observation given the model
        through the forward algorithm
        :param observation: Observation (sequence)
        :return: Tuple of the probability of seeing the observation and
        the final matrix
        """
        obs_len = len(observation)
        num_states = len(self.states)
        obs_indices = np.array([self.sym_idx[symbol] for symbol in observation])
        fwd = np.full((num_states, obs_len), self.zero)
        # initialization step is identical to viterbi
        fwd[:, 0] = self.mul(self.init_probs, self.emit_probs[:, obs_indices[0]])

        for t in range(1, obs_len):
            summed = self.sum_states(self.trans_probs, fwd[:, t - 1])
            fwd[:, t] = self.mul(self.emit_probs[:, obs_indices[t]], summed)

        tot_prob = self.add(fwd[:, -1])
        return tot_prob, fwd

    def backward(self, observation) -> tuple[float, np.ndarray]:
        """
        Find the probability of seeing an observation given the model
        through the backward algorithm
        :param observation: observation (sequence)
        :return: Tuple of the probability of seeing the observation and
        the final matrix
        """
        obs_len = len(observation)
        num_states = len(self.states)
        obs_indices = np.array([self.sym_idx[symbol] for symbol in observation])
        bwd = np.full((num_states, obs_len), self.zero)

        bwd[:, -1] = 0.0 if self.use_log_space else 1.0

        for t in reversed(range(obs_len - 1)):
            next_term = self.mul(self.emit_probs[:, obs_indices[t + 1]], bwd[:, t + 1])
            bwd[:, t] = self.sum_states(self.trans_probs.T, next_term)

        if self.use_log_space:
            total_prob = np.logaddexp.reduce(
                self.init_probs + self.emit_probs[:, obs_indices[0]] + bwd[:, 0]
            )
        else:
            total_prob = np.sum(
                self.init_probs * self.emit_probs[:, obs_indices[0]] * bwd[:, 0]
            )
        return total_prob, bwd

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
    hmm = HiddenMarkovModel.from_json("params.json")
    hmm = hmm.to_log_space()
    obs = "ACGT"
    bwd = hmm.backward(obs)
    fwd = hmm.forward(obs)
    vit = hmm.viterbi(obs)
    print(bwd)
    print(fwd)
    print(vit)
