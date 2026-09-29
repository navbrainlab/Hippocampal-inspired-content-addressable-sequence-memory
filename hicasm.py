"""HICASM: R -> C -> A -> B -> S -> R with hard WTA in A."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    N_R: int = 32
    N_C: int = 160
    N_B: int = 640
    N_S: int = 1024
    H: int = 6
    g: float = 1.05
    theta_A: float = 2.84
    p_A: float = 2.0
    a_B: float = 0.05
    beta: float = 0.375
    epsilon_factor: float = 1e-4
    adaptation_length: int = 16
    n_C_adaptation: int = 200
    n_S_adaptation: int = 1707
    model_seed: int = 321001
    adaptation_seed: int = 721001


@dataclass
class Memory:
    episodes: np.ndarray
    C_trajectory: np.ndarray
    barcodes: np.ndarray
    S_states: np.ndarray
    W_RS: np.ndarray
    slot_counts: np.ndarray
    context_values: np.ndarray
    barcode_values: np.ndarray

    @property
    def M(self) -> int:
        return len(self.episodes)

    @property
    def L(self) -> int:
        return self.episodes.shape[1]


def sample_unique_sequences(
    rng: np.random.Generator, N_R: int, length: int, count: int
) -> np.ndarray:
    """Sample unique sequences with no repeated relation within a sequence."""
    if length > N_R:
        raise ValueError("length cannot exceed N_R")
    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    while len(rows) < count:
        row = tuple(int(x) for x in rng.choice(N_R, length, replace=False))
        if row not in seen:
            seen.add(row)
            rows.append(row)
    return np.asarray(rows, dtype=np.int16)


class HICASM:
    """Canonical HICASM model with deterministic episode-level hard WTA."""

    def __init__(self, config: Config = Config()):
        self.config = config
        c = config
        if c.N_R != 32:
            raise ValueError("The canonical HICASM model uses N_R=32")

        # Canonical paired matrix construction.
        rng = np.random.default_rng(c.model_seed)
        W_CC = rng.normal(0.0, 1.0 / np.sqrt(c.N_C), (c.N_C, c.N_C))
        W_CC /= float(np.max(np.abs(np.linalg.eigvals(W_CC))))
        W_CR = rng.normal(0.0, 1.0, (c.N_C, c.N_R))
        W_CR /= np.linalg.norm(W_CR, axis=0, keepdims=True)
        W_CR *= 3.5
        self.W_CC, self.W_CR = W_CC, W_CR

        rng_B = np.random.default_rng(c.model_seed + 10_000)
        self.W_BC = rng_B.normal(0.0, 1.0 / np.sqrt(c.N_C), (1280, c.N_C))[:c.N_B]

        rng_S = np.random.default_rng(c.model_seed + 20_000)
        raw = rng_S.normal(0.0, 1.0, (c.N_S, c.N_S))
        Q, R = np.linalg.qr(raw)
        Q *= np.where(np.diag(R) < 0.0, -1.0, 1.0)
        # The frozen canonical builder was called with N_B_max=N_B=640;
        # sampling a wider matrix and slicing it would change RNG row layout.
        W_SB = rng_S.normal(0.0, 1.0, (c.N_S, c.N_B)) / np.sqrt(c.N_B)

        # Independent frozen adaptation.
        n_adapt = c.n_C_adaptation + c.n_S_adaptation
        adapt_rng = np.random.default_rng(c.adaptation_seed)
        adapt_episodes = sample_unique_sequences(
            adapt_rng, c.N_R, c.adaptation_length, n_adapt
        )
        adapt_C = self.run_C(adapt_episodes)
        self.mu_C = adapt_C[:c.n_C_adaptation].mean(axis=0)
        scores = self._barcode_scores(adapt_C[:c.n_C_adaptation])
        self.theta_B = float(np.quantile(scores, 1.0 - c.a_B))
        adapt_B = self.encode_B(adapt_C)

        self.mu_B = adapt_B[c.n_C_adaptation:].mean(axis=0)
        u0 = (adapt_B[c.n_C_adaptation:] - self.mu_B) @ W_SB.T
        raw_states = self._linear_replay(Q, u0, c.adaptation_length)
        flat = raw_states.reshape(-1, c.N_S)
        covariance = flat.T @ flat / len(flat)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        epsilon = float(c.epsilon_factor * eigenvalues.mean())
        gains = (eigenvalues + epsilon) ** (-c.beta)
        G_beta = (eigenvectors * gains) @ eigenvectors.T
        self.W_SB_star = G_beta @ W_SB
        self.Q_S = np.linalg.solve(G_beta.T, (G_beta @ Q).T).T

    def C_trajectory(self, episodes: np.ndarray) -> np.ndarray:
        episodes = np.asarray(episodes, dtype=int)
        if episodes.ndim != 2:
            raise ValueError("episodes must have shape (M, L)")
        C = np.zeros((len(episodes), self.config.N_C))
        trajectory = np.empty((len(episodes), episodes.shape[1], self.config.N_C))
        for t in range(episodes.shape[1]):
            C = np.tanh(
                self.config.g * (C @ self.W_CC.T)
                + self.W_CR[:, episodes[:, t]].T
            )
            trajectory[:, t] = C
        return trajectory

    def run_C(self, episodes: np.ndarray) -> np.ndarray:
        return self.C_trajectory(episodes)[:, -1]

    def _barcode_scores(self, C: np.ndarray) -> np.ndarray:
        return np.maximum((np.asarray(C) - self.mu_C) @ self.W_BC.T, 0.0)

    def encode_B(self, C: np.ndarray) -> np.ndarray:
        return (self._barcode_scores(C) > self.theta_B).astype(float)

    @staticmethod
    def _linear_replay(Q: np.ndarray, initial: np.ndarray, steps: int) -> np.ndarray:
        states = np.empty((len(initial), steps, initial.shape[1]), dtype=float)
        states[:, 0] = initial
        for t in range(1, steps):
            states[:, t] = states[:, t - 1] @ Q.T
        return states

    def replay_S(self, barcodes: np.ndarray, steps: int) -> np.ndarray:
        initial = (np.asarray(barcodes) - self.mu_B) @ self.W_SB_star.T
        return self._linear_replay(self.Q_S, initial, steps)

    def build_memory(self, episodes: np.ndarray) -> Memory:
        episodes = np.asarray(episodes, dtype=int)
        trajectory = self.C_trajectory(episodes)
        barcodes = self.encode_B(trajectory[:, -1])
        states = self.replay_S(barcodes, episodes.shape[1])

        c = self.config
        W_RS = np.zeros((c.N_R, c.H, c.N_S), dtype=float)
        counters = np.zeros(c.N_R, dtype=np.int8)
        counts = np.zeros((c.N_R, c.H), dtype=int)
        for state, relation in zip(states.reshape(-1, c.N_S), episodes.ravel()):
            slot = int(counters[relation])
            W_RS[relation, slot] += state
            counts[relation, slot] += 1
            counters[relation] = (slot + 1) % c.H

        return Memory(
            episodes=episodes,
            C_trajectory=trajectory,
            barcodes=barcodes,
            S_states=states,
            W_RS=W_RS,
            slot_counts=counts,
            context_values=trajectory - self.mu_C,
            barcode_values=barcodes - self.mu_B,
        )

    def address_from_C(self, memory: Memory, C_query: np.ndarray) -> dict[str, np.ndarray]:
        q = np.atleast_2d(np.asarray(C_query, dtype=float)) - self.mu_C
        evidence = q @ memory.context_values.reshape(-1, self.config.N_C).T
        branch = np.maximum(evidence - self.config.theta_A, 0.0) ** self.config.p_A
        D = branch.reshape(len(q), memory.M, memory.L).sum(axis=2)

        winner = D.argmax(axis=1)
        D_wta = np.zeros_like(D)
        D_wta[np.arange(len(D)), winner] = D[np.arange(len(D)), winner]
        total = D_wta.sum(axis=1, keepdims=True)
        u_B = D_wta @ memory.barcode_values
        recovered_B = self.mu_B + np.divide(
            u_B, total, out=np.zeros_like(u_B), where=total > 0
        )
        return {"D": D, "D_wta": D_wta, "winner": winner, "barcode": recovered_B}

    def address(self, memory: Memory, cues: np.ndarray) -> dict[str, np.ndarray]:
        return self.address_from_C(memory, self.run_C(np.asarray(cues, dtype=int)))

    @staticmethod
    def _readout(memory: Memory, states: np.ndarray) -> np.ndarray:
        slot_scores = np.einsum("...s,jhs->...jh", states, memory.W_RS, optimize=True)
        return slot_scores.max(axis=-1).argmax(axis=-1)

    def replay(self, memory: Memory, cues: np.ndarray) -> dict[str, np.ndarray]:
        addressed = self.address(memory, cues)
        states = self.replay_S(addressed["barcode"], memory.L)
        predicted = self._readout(memory, states)
        return {**addressed, "states": states, "predicted": predicted}

    def indexed_replay(self, memory: Memory) -> np.ndarray:
        """Downstream replay ceiling when the stored barcode is supplied directly."""
        return self._readout(memory, memory.S_states)


__all__ = ["Config", "Memory", "HICASM", "sample_unique_sequences"]
