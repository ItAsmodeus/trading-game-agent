"""
Бейзлайн — случайный агент.
Если обученный агент не бьёт этот — что-то не так.
"""

import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int = 6, seed: int = 42):
        self.n_actions = n_actions
        self.rng       = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = False):
        action = self.rng.integers(0, self.n_actions)
        return action, None


class BuyAndHoldAgent:
    """Покупает в начале и держит весь эпизод. Классический бенчмарк."""

    def __init__(self):
        self.bought = False

    def predict(self, obs, deterministic: bool = False):
        if not self.bought:
            self.bought = True
            return 1, None   # BUY
        return 0, None       # HOLD

    def reset(self):
        self.bought = False
