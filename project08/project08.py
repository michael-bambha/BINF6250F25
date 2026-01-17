""""
File: project08.py
Description: BINF6250 project 8
Authors: Michael Bambha and Jason Bae
"""
# TODO: getting NaNs for alpha which probably has something to do with the loop range and the fact that xi is initialized to 0

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
            alphabet: list[str] = None,
            use_log_space: bool = False,

    ):
        self.use_log_space = use_log_space

        # store OG dicts before we convert to arrays
        self._init_dict = init_probs
        self._trans_dict = trans_probs
        self._emit_dict = emit_probs
        self.alphabet = alphabet
        if not self.alphabet:
            self.alphabet = "AGCT"
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
        self.div = np.subtract if use_log_space else np.divide


    def __str__(self):
        # pretty print format so it looks nice if we print out the class
        return pformat(vars(self))

    def __getitem__(self, key):
        if isinstance(key, str):
            return {
                "init": self._init_dict.get(key),
                "transitions": self._trans_dict.get(key),
                "emissions": self._emit_dict.get(key),
            }
    def set_init_probs(self, init):
        self.init_probs = init

    def set_trans_probs(self, trans):
        self.trans_probs = trans

    def set_emit_probs(self, emit):
        self.emit_probs = emit

    def update_probs(self, pi, aij, bi):
        self.set_init_probs(pi)
        self.set_trans_probs(aij)
        self.set_emit_probs(bi)

    def check_convergence(self, converged, convergence, pi, aij, bi):
        converged = True
        prev_vals = self.init_probs, self.trans_probs, self.emit_probs
        curr_vals = pi, aij, bi
        for i in range(3):
            difference = np.divide(curr_vals[i], prev_vals[i]) if self.use_log_space \
                else np.subtract(curr_vals[i], prev_vals[i])
            if np.any(difference > convergence):
                converged = False
                break
    @property
    def alphabet_size(self):
        return len(self.alphabet)

    def sum_states(self, matrix: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """
        In the forward and backward algorithms, we need different math operations
        for summing over states when in log space or probability space. We
        take the dot product of A @ B in prob space, while we need to use
        logaddexp.reduce for log space.
        :param matrix: alpha (transition probs)
        :param prev: 1d array of probabilities for all states at a time t
        :return: A @ B if in probability space, logaddexp.reduce() if logspace
        """
        # sum(p(xt | xt-1)alpha(xt-1))
        if getattr(self, "use_log_space", False):
            # we reshape prev to be explicitly 1D so it broadcasts along columns
            # axis=0 means collapses on rows -- so logaddexp.reduce() on cols
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
        # fwd[:, t-1] = alpha(xt - 1)
        # self.emit_probs[:, obs_indices[t]] = p(yt | xt)
        for t in range(1, obs_len):
            # summed = sum(p (xt | xt-1) alpha(xt-1))
            summed = self.sum_states(self.trans_probs, fwd[:, t - 1])
            # fwd[:, t] = p(yt | xt)sum(p(xt | xt-1)alpha(xt-1))
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

    def forward_backward(self, fwd: np.ndarray, bwd: np.ndarray):
        """
        Implementation of the forward backward algorithm - find the probability
        for a certain state at a specific point, given the model and observation.
        :param bwd: Backward matrix for a given observation
        :param fwd: Forward matrix for a given observation
        :return: posterior probability matrix showing P(xt = s | y1:t) for all s and t
        """
        posterior = self.mul(fwd, bwd)
        # posterior is currently a joint probability -- we need conditional P(xt = s | y1:t)
        # P(xt = s | y1:t) = P(xt = s, y1:t) / P(y1:t))
        # logaddexp.reduce() if log space, otherwise sum
        if self.use_log_space:
            posterior -= np.logaddexp.reduce(posterior, axis=0, keepdims=True)
        else:
            posterior /= posterior.sum(axis=0, keepdims=True)

        return posterior

    def baum_welch(self, observation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Implementation of the Baum-Welch algorithm to find probabilities for initial (pi), transition (alpha),
        and emission (b) matrices for a given number of hidden states and a known observation.

        This function assumes that the initial guesses for the probabilities have already been instantiated with
        the class constructor.

        Variable names reference the equations in the associated Baum_Welch.md file.

        :param observation: String corresponding to an observation of events. Assumes that the string is ordered
        in increasing time, and each event should be encoded as a single character.
        :return: initial (pi), transition (alpha) and emission (b) matrix updates, given the observation,
        guesses about the model, and a known number of hidden states.
        """
        iterations = 0
        convergence = 0.0001
        converged = False
        prev_vals = None
        curr_vals = None
        while iterations <= 1000 and not converged:

            # We will currently assume that the parameters are initialized when the model is called
            _, fwd = self.forward(observation)
            _, bwd = self.backward(observation)
            obs_indices = np.array([self.sym_idx[symbol] for symbol in observation])
            gamma = self.forward_backward(fwd, bwd)
            T = len(observation)
            # the full xi is a 3D array, since we are looking at state i and j across all t (3 dimensions)
            # aij* = sum(t=0:T-2) xi(t)(i,j) / sum(t=0:T-2)(gamma(t)(i))
            xi = np.zeros((T-1, len(self.states), len(self.states)))
            # we use some broadcasting tricks to vectorize the operation, but we still need to loop over each t
            # xi(t) needs to end up as an NxN matrix (the full xi will then be T-1xNxN)
            # fwd[:, t][:, None] = (N, 1); trans_probs = (N, N); emit_probs[t+1][None, :] = (1, N); bwd[:, t+1][None, :] = (1, N)
            # we will end up yielding an NxN matrix for all T-1
            for t in range(T-1):
                xi_num = self.mul(fwd[:, t][:, None], self.trans_probs, self.emit_probs[:, obs_indices[t+1]][None, :], bwd[:, t+1][None, :])
                if self.use_log_space:
                    xi_denom = np.logaddexp.reduce(xi_num.ravel())
                else:
                    xi_denom = xi_num.sum()
                xi[t] = self.div(xi_num, xi_denom)
            pi = gamma[:, 0]
            aij_num = self.agg(xi, axis=0) # (N, N)
            # transition matrix explicitly excludes the last entry (since you can't transition from the last thing to nothing)
            aij_denom = self.agg(gamma[:, :-1], axis=1) # (N, )
            aij = self.div(aij_num, aij_denom[:, None])
            # bi(vk) = sum(t=1:T) gamma(i)(t) s.t. yt=vk / gamma(i)(t)
            # since we need to find where yt=vk, we can vectorize this with a boolean mask
            # below creates a np array called "mask", consisting of booleans of shape TxK due to broadcasting`
            # note we reshaped obs_indices to be (T,1) and the alphabet vector to be (1,K) which produces TxK
            # results in a matrix where mask[t,k] = (obs_indices[t] == k)
            mask = obs_indices[:, None] == np.arange(self.alphabet_size)[None, :]
            # reshape gamma to (T, N, 1) and mask to (T, 1, K)
            # multiplying gives us shape (T, N, K)
            # this uses the mask from before to only count where obs_indices[t] == k
            # we have to transpose so that the time axis aligns properly
            # note we need to add a 3rd axis here because mask is (TxK) and gamma.T is (T, N) - we need all combos
            bi_num = self.agg(gamma.T[:, :, None] * mask[:, None, :], axis=0)
            bi_denom = self.agg(gamma, axis=1)
            # reshape the denominator to be a column vector so it can be broadcast across rows
            bi = self.div(bi_num, bi_denom[:, None])

            iterations += 1

            if iterations % 2 == 0 and iterations > 0:
                self.check_convergence(converged, convergence, pi, aij, bi)

            self.update_probs(pi, aij, bi)


        return pi, aij, bi

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
    obs = "ACGTTTAGC"
    print(hmm.trans_probs)
    print(hmm.state_idx)
    print(hmm.sym_idx)
    obs_indices = np.array([hmm.sym_idx[symbol] for symbol in obs])
    print(hmm.baum_welch(obs))

