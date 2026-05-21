"""
SelfPlayTradingEnv — TradingEnv with a frozen opponent for self-play training.

Each episode: main agent and frozen opponent start at the same bar, trade the
same market side-by-side. Reward blends own PnL signal with a competitive signal
(own_portfolio - opponent_portfolio), encouraging the agent to beat its past self.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import CFG
from env.trading_env import TradingEnv

COMPETITIVE_ALPHA = 0.3  # 30% competitive, 70% own PnL


class SelfPlayTradingEnv(TradingEnv):
    """TradingEnv that adds a frozen opponent agent running in parallel."""

    def __init__(
        self,
        opponent_model=None,
        competitive_alpha: float = COMPETITIVE_ALPHA,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.opponent_model = opponent_model
        self.competitive_alpha = competitive_alpha

        # Opponent env shares the same data DataFrame (no copy)
        self._opp_env = TradingEnv(
            symbol=self.symbol,
            data=self.data,
            random_start=False,
            noise_scale=0.0,
            reward_mode=self.reward_mode,
        )
        self._opp_obs: Optional[np.ndarray] = None
        self._opp_lstm_states = None
        self._opp_ep_starts: Optional[np.ndarray] = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Sync opponent to exact same starting bar as main agent
        self._opp_env.sim.reset()
        self._opp_env.milestones.reset()
        self._opp_env._episode_steps = 0
        self._opp_env._bars_in_position = 0
        self._opp_env._pnl_buffer.clear()
        self._opp_env._current_idx = self._current_idx

        self._opp_obs = self._opp_env._get_obs()
        self._opp_lstm_states = None
        self._opp_ep_starts = np.ones((1,), dtype=bool)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.opponent_model is not None and self._opp_obs is not None:
            # Opponent picks its action (deterministic, frozen weights)
            opp_action, self._opp_lstm_states = self.opponent_model.predict(
                self._opp_obs,
                state=self._opp_lstm_states,
                episode_start=self._opp_ep_starts,
                deterministic=True,
            )
            self._opp_ep_starts = np.zeros((1,), dtype=bool)

            self._opp_obs, _, _, _, opp_info = self._opp_env.step(int(opp_action))

            # Competitive signal: positive when we're ahead of the opponent
            own_value = info["portfolio_value"]
            opp_value = opp_info["portfolio_value"]
            competitive = float(np.clip(
                (own_value - opp_value) / CFG.STARTING_CAPITAL, -1.0, 1.0
            ))

            reward = float(np.clip(
                (1.0 - self.competitive_alpha) * reward
                + self.competitive_alpha * competitive,
                -1.0, 1.0,
            ))

        return obs, reward, terminated, truncated, info
