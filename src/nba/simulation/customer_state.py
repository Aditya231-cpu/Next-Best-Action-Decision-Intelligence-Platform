"""Customer states used only by the simulated repeated-contact environment."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CustomerState:
    recency: np.ndarray
    history: np.ndarray
    mens: np.ndarray
    womens: np.ndarray
    newbie: np.ndarray
    contacts: np.ndarray

    @classmethod
    def sample(cls, n: int, rng: np.random.Generator) -> "CustomerState":
        return cls(recency=rng.integers(1, 13, n), history=rng.lognormal(4.5, .8, n),
            mens=rng.binomial(1, .48, n), womens=rng.binomial(1, .50, n),
            newbie=rng.binomial(1, .25, n), contacts=np.zeros(n, dtype=int))
