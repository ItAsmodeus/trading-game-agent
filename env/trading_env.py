"""
TradingEnv — OpenAI Gymnasium среда.
Агент видит окно из LOOKBACK свечей + состояние портфеля.
Действия: 0=HOLD, 1=BUY25%, 2=BUY50%, 3=BUY100%, 4=SELL25%, 5=SELL50%, 6=SELL100%
"""
from __future__ import annotations

import random
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import CFG
from env.data_loader import load_dataset
from env.market_sim import MarketSimulator, Portfolio
from env.milestones import MilestoneTracker


# Соответствие действий → параметрам исполнения
ACTION_MAP = {
    0:  ("hold",  0.00),
    1:  ("buy",   0.25),
    2:  ("buy",   0.50),
    3:  ("buy",   1.00),
    4:  ("sell",  0.25),
    5:  ("sell",  0.50),
    6:  ("sell",  1.00),
    7:  ("short", 0.25),
    8:  ("short", 0.50),
    9:  ("short", 1.00),
    10: ("cover", 0.25),
    11: ("cover", 0.50),
    12: ("cover", 1.00),
}


class TradingEnv(gym.Env):
    """
    Одноинструментная торговая среда.
    Для Self-Play запускать два экземпляра на одних данных.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        split: str = "train",
        data: Optional[pd.DataFrame] = None,
        random_start: bool = True,
        noise_scale: float = 0.0,
    ):
        super().__init__()
        self.symbol = symbol
        self.split = split
        self.random_start = random_start
        self.noise_scale = noise_scale

        self.data = data if data is not None else load_dataset(symbol, split=split)
        self.feature_cols = [c for c in self.data.columns if c not in ("open", "high", "low", "close", "volume", "close_raw")]
        # Observation: all normalized numeric cols except close_raw (which is used only for market sim)
        self.all_cols = [c for c in self.data.select_dtypes(include=[np.number]).columns if c != "close_raw"]
        n_features = len(self.all_cols)

        # State: [LOOKBACK × n_features] + [5 portfolio + 6 time features]
        # time features: ep_progress sin/cos, weekly sin/cos, hour-of-day sin/cos
        obs_size = CFG.LOOKBACK * n_features + 11
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(CFG.N_ACTIONS)

        self.sim: MarketSimulator = MarketSimulator()
        self.milestones = MilestoneTracker(CFG.STARTING_CAPITAL)
        self._step = 0
        self._start_idx = CFG.LOOKBACK
        self._current_idx = self._start_idx
        self._episode_steps = 0
        self._bars_in_position = 0

    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.milestones.reset()
        self._episode_steps = 0
        self._bars_in_position = 0

        max_start = len(self.data) - CFG.MAX_STEPS_PER_EPISODE - CFG.LOOKBACK
        if self.random_start and max_start > self._start_idx:
            self._current_idx = random.randint(self._start_idx, max_start)
        else:
            self._current_idx = self._start_idx

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        row = self.data.iloc[self._current_idx]
        price = float(self.data["close_raw"].iloc[self._current_idx])
        prices = {self.symbol: price}

        prev_value = self.sim.portfolio.total_value(prices)
        self.sim.portfolio.update_peak(prices)

        action_type, ratio = ACTION_MAP[action]
        was_in_position = bool(self.sim.portfolio.positions.get(self.symbol) and
                               self.sim.portfolio.positions[self.symbol].is_open)
        if action_type == "buy":
            self.sim.buy(self.symbol, price, ratio)
        elif action_type == "sell":
            self.sim.sell(self.symbol, price, ratio)
        elif action_type == "short":
            self.sim.short(self.symbol, price, ratio)
        elif action_type == "cover":
            self.sim.cover(self.symbol, price, ratio)

        # Track bars held in position for churn penalty
        pos_after = self.sim.portfolio.positions.get(self.symbol)
        now_in_position = bool(pos_after and pos_after.is_open)
        if now_in_position:
            self._bars_in_position += 1
        else:
            self._bars_in_position = 0

        self._current_idx += 1
        self._episode_steps += 1

        new_price = float(self.data["close_raw"].iloc[self._current_idx])
        new_prices = {self.symbol: new_price}
        new_value = self.sim.portfolio.total_value(new_prices)

        reward = self._compute_reward(
            prev_value=prev_value,
            new_value=new_value,
            action=action_type,
            prices=new_prices,
            ret_1h=float(row.get("ret_1", 0.0)),
            was_in_position=was_in_position,
        )

        # Milestone bonus (curriculum learning)
        milestone_bonus = self.milestones.update(
            portfolio=new_value,
            hours_alive=self._episode_steps,
            n_trades=self.sim.portfolio.n_trades,
        )
        reward += milestone_bonus

        terminated, reason = self._check_terminal(new_prices)
        truncated = self._episode_steps >= CFG.MAX_STEPS_PER_EPISODE or \
                    self._current_idx >= len(self.data) - 2

        info = {
            "portfolio_value": new_value,
            "cash": self.sim.portfolio.cash,
            "drawdown": self.sim.portfolio.drawdown(new_prices),
            "n_trades": self.sim.portfolio.n_trades,
            "total_fees": self.sim.portfolio.total_fees,
            "terminal_reason": reason,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        window = self.data[self.all_cols].iloc[
            self._current_idx - CFG.LOOKBACK: self._current_idx
        ].values.astype(np.float32)
        market_obs = window.flatten()

        prices = {self.symbol: float(self.data["close_raw"].iloc[self._current_idx])}
        portfolio_value = self.sim.portfolio.total_value(prices)
        pos = self.sim.portfolio.positions.get(self.symbol)

        position_ratio = 0.0
        unrealized_pnl = 0.0
        if pos and pos.is_open:
            pos_value = pos.value(prices[self.symbol])
            ratio = pos_value / max(portfolio_value, 1.0)
            # Signed: positive = long, negative = short
            position_ratio = ratio if pos.side == "long" else -ratio
            unrealized_pnl = pos.unrealized_pnl(prices[self.symbol]) / max(portfolio_value, 1.0)

        cash_ratio = self.sim.portfolio.cash / max(portfolio_value, 1.0)
        drawdown = self.sim.portfolio.drawdown(prices)
        value_ratio = portfolio_value / CFG.STARTING_CAPITAL

        portfolio_obs = np.array(
            [position_ratio, unrealized_pnl, cash_ratio, drawdown, value_ratio],
            dtype=np.float32,
        )

        # Time features: episode progress + weekly cycle + hour-of-day (H-009)
        ep_progress = self._episode_steps / max(CFG.MAX_STEPS_PER_EPISODE, 1)
        weekly_phase = (self._current_idx % 168) / 168.0
        hour_of_day = self.data.index[self._current_idx].hour / 24.0
        time_obs = np.array([
            np.sin(2 * np.pi * ep_progress),
            np.cos(2 * np.pi * ep_progress),
            np.sin(2 * np.pi * weekly_phase),
            np.cos(2 * np.pi * weekly_phase),
            np.sin(2 * np.pi * hour_of_day),   # hour-of-day cycle
            np.cos(2 * np.pi * hour_of_day),
        ], dtype=np.float32)

        obs = np.concatenate([market_obs, portfolio_obs, time_obs])
        if self.noise_scale > 0:
            obs = obs + np.random.normal(0, self.noise_scale, obs.shape).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def _compute_reward(
        self,
        prev_value: float,
        new_value: float,
        action: str,
        prices: dict,
        ret_1h: float,
        was_in_position: bool = False,
    ) -> float:
        pnl = (new_value - prev_value) / CFG.STARTING_CAPITAL

        pos = self.sim.portfolio.positions.get(self.symbol)
        in_cash = not (pos and pos.is_open)

        # Idle penalty: exempt when market volatility is extreme (QUANTUM_SKILL: σ>1.2 → sit out)
        if in_cash and action == "hold":
            sigma_raw = self._current_sigma()
            if sigma_raw <= CFG.SIGMA_IDLE_EXEMPT:
                pnl += CFG.INACTION_PENALTY

        # Churn penalty: penalize closing a position held for < CHURN_BARS bars
        just_closed = was_in_position and in_cash
        if just_closed and self._bars_in_position < CFG.CHURN_BARS:
            pnl += CFG.CHURN_PENALTY

        return float(np.clip(pnl, -1.0, 1.0))

    def _current_sigma(self) -> float:
        """Annualized 20-bar volatility from raw close prices."""
        start = max(0, self._current_idx - 21)
        closes = self.data["close_raw"].iloc[start:self._current_idx]
        if len(closes) < 2:
            return 0.5
        log_rets = np.log(closes / closes.shift(1)).dropna()
        return float(log_rets.std() * np.sqrt(24 * 365))

    def _check_terminal(self, prices: dict) -> tuple[bool, str]:
        value = self.sim.portfolio.total_value(prices)
        if value >= CFG.WIN_TARGET:
            return True, "WIN"
        if value <= CFG.STARTING_CAPITAL * CFG.LOSS_THRESHOLD:
            return True, "LOSS"
        return False, ""

    def render(self):
        prices = {self.symbol: float(self.data["close_raw"].iloc[self._current_idx])}
        value = self.sim.portfolio.total_value(prices)
        dd = self.sim.portfolio.drawdown(prices)
        print(
            f"Step {self._episode_steps:4d} | "
            f"Value: ${value:10.2f} | "
            f"Drawdown: {dd:.1%} | "
            f"Trades: {self.sim.portfolio.n_trades}"
        )
