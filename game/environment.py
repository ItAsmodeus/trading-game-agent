"""
TradingEnv — gymnasium-совместимая среда.
Один агент, один актив, дискретные действия.
Это уровень 1 из трёх запланированных.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from game.market import RealisticMarket
from game.rules import GAME_RULES, N_ACTIONS, N_FEATURES
from game.milestones import MilestoneTracker


class TradingEnv(gym.Env):
    """
    Прototip: один агент торгует одним активом на исторических OHLCV данных.

    Observation (20 признаков):
        price returns (3) + volatility (1) + RSI (1) + volume ratio (1) +
        MACD approx (1) + BB position (1) + ATR (1) + candle body (1) +
        high-low range (1) + portfolio state (4) + time features (4)

    Actions: 0=HOLD, 1=BUY, 2=SELL_SHORT, 3=CLOSE, 4=BUY_HALF, 5=SELL_HALF
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, config: dict = None):
        super().__init__()

        self.df     = df.reset_index(drop=True)
        self.config = config or GAME_RULES
        self.market = RealisticMarket(config=self.config)

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_FEATURES,), dtype=np.float32
        )

        self._init_state()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._observe(), {}

    def step(self, action: int):
        assert not self.done, "Вызван step() после done=True. Сначала reset()."

        price = float(self.df["close"].iloc[self.step_idx])

        # Исполнить действие (возвращает стоимость-комиссию)
        cost = self.market.execute_action(
            action=action,
            price=price,
            position=self.position,
            portfolio=self.portfolio,
            config=self.config,
        )

        # Обновить позицию
        self._update_position(action, price)

        # Следующий шаг
        self.step_idx += 1

        # P&L от удержания позиции
        next_price = float(self.df["close"].iloc[self.step_idx])
        pnl = self._compute_pnl(price, next_price)

        self.portfolio += pnl - cost
        self.portfolio  = max(self.portfolio, 0.0)

        if self.portfolio > self.peak_portfolio:
            self.peak_portfolio = self.portfolio

        self.trade_count += (1 if action != 0 else 0)

        # Проверить конец
        terminated = self._check_terminated()
        reward     = self._compute_reward(pnl, cost)

        # Бонус от игровых достижений (curriculum learning)
        returns = np.diff([self.config["starting_capital"], self.portfolio])
        sharpe  = float(returns[-1] / (abs(returns[-1]) + 1e-8)) if len(returns) > 0 else 0.0
        milestone_bonus = self.milestones.update(self.portfolio, self.trade_count, sharpe, self.step_idx)
        reward += milestone_bonus

        obs  = self._observe()
        info = {
            "portfolio":   self.portfolio,
            "position":    self.position,
            "step":        self.step_idx,
            "trade_count": self.trade_count,
        }

        return obs, reward, terminated, False, info

    def render(self):
        pct = (self.portfolio / self.config["starting_capital"] - 1) * 100
        print(
            f"Step {self.step_idx:4d} | "
            f"Portfolio: ${self.portfolio:,.0f} ({pct:+.1f}%) | "
            f"Position: {self.position:+.1f} | "
            f"Trades: {self.trade_count}"
        )

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _init_state(self):
        self.step_idx       = 50            # первые 50 баров нужны для индикаторов
        self.portfolio      = float(self.config["starting_capital"])
        self.peak_portfolio = self.portfolio
        self.position       = 0.0           # -1=short, 0=flat, 1=long
        self.entry_price    = 0.0
        self.trade_count    = 0
        self.done           = False
        self.max_steps      = len(self.df) - 2
        self.milestones     = MilestoneTracker(self.config["starting_capital"])

    def _update_position(self, action: int, price: float):
        from game.rules import ACTIONS
        name = ACTIONS.get(int(action) if hasattr(action, '__len__') else action, "HOLD")

        if name in ("BUY", "BUY_HALF"):
            if self.position <= 0:
                self.position    = 1.0 if name == "BUY" else 0.5
                self.entry_price = price
        elif name in ("SELL_SHORT", "SELL_HALF"):
            if self.position >= 0:
                self.position    = -1.0 if name == "SELL_SHORT" else -0.5
                self.entry_price = price
        elif name == "CLOSE":
            self.position    = 0.0
            self.entry_price = 0.0

    def _compute_pnl(self, price: float, next_price: float) -> float:
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        ret = (next_price - price) / price
        return self.portfolio * abs(self.position) * ret * np.sign(self.position)

    def _check_terminated(self) -> bool:
        if self.step_idx >= self.max_steps:
            self.done = True
            return True
        game_over_threshold = self.config["starting_capital"] * self.config["game_over_pct"]
        if self.portfolio < game_over_threshold:
            self.done = True
            return True
        return False

    def _compute_reward(self, pnl: float, cost: float) -> float:
        # Log-return (основной сигнал ~0.01 на типичный день)
        ret = pnl / max(self.portfolio, 1.0)

        # Drawdown penalty (квадратичный)
        dd = max(0, (self.peak_portfolio - self.portfolio) / max(self.peak_portfolio, 1))
        dd_penalty = 3.0 * (dd ** 2) if dd > 0.05 else 0.0
        if dd > 0.25:
            dd_penalty = 10.0 * dd

        # Overtrading penalty
        cost_penalty = cost / max(self.portfolio, 1.0) * 8.0

        reward = ret - dd_penalty - cost_penalty
        return float(np.clip(reward, -10.0, 10.0))

    def _observe(self) -> np.ndarray:
        i   = self.step_idx
        df  = self.df
        close = float(df["close"].iloc[i])

        # Returns
        r1  = np.log(df["close"].iloc[i]   / df["close"].iloc[i-1]  + 1e-8)
        r5  = np.log(df["close"].iloc[i]   / df["close"].iloc[i-5]  + 1e-8)
        r20 = np.log(df["close"].iloc[i]   / df["close"].iloc[i-20] + 1e-8)

        # Volatility
        vol = df["close"].pct_change().iloc[i-20:i].std()

        # RSI
        rsi = self._rsi(i)

        # Volume ratio
        vol_ratio = np.log(df["volume"].iloc[i] / (df["volume"].iloc[i-20:i].mean() + 1e-8) + 1e-8)

        # MACD approximation
        ema12 = df["close"].iloc[i-12:i].mean()
        ema26 = df["close"].iloc[i-26:i].mean()
        macd  = (ema12 / (ema26 + 1e-8)) - 1

        # Bollinger position
        bb = self._bb_position(i)

        # ATR normalized
        atr = self._atr(i) / (close + 1e-8)

        # Candle body and range
        body  = (df["close"].iloc[i] - df["open"].iloc[i]) / (close + 1e-8)
        hrange = (df["high"].iloc[i] - df["low"].iloc[i]) / (close + 1e-8)

        # Portfolio state
        drawdown = (self.peak_portfolio - self.portfolio) / max(self.peak_portfolio, 1)
        cap_ratio = self.portfolio / self.config["starting_capital"]
        progress  = self.portfolio / self.config["win_target"]

        # Time features (sin/cos encoding)
        t      = self.step_idx / max(self.max_steps, 1)
        sin_t  = np.sin(2 * np.pi * t)
        cos_t  = np.cos(2 * np.pi * t)
        sin_w  = np.sin(2 * np.pi * (self.step_idx % 7) / 7)
        cos_w  = np.cos(2 * np.pi * (self.step_idx % 7) / 7)

        # Funding rate (0.0 если колонка не добавлена в данные)
        fr_norm = float(df["fr_normalized"].iloc[i]) if "fr_normalized" in df.columns else 0.0
        fr_ma   = float(df["fr_ma_24h"].iloc[i])     if "fr_ma_24h" in df.columns else 0.0

        obs = np.array([
            r1, r5, r20,          # 3
            vol,                   # 1
            rsi,                   # 1
            vol_ratio,             # 1
            macd,                  # 1
            bb,                    # 1
            atr,                   # 1
            body,                  # 1
            hrange,                # 1
            self.position,         # 1
            drawdown,              # 1
            cap_ratio,             # 1
            progress,              # 1
            sin_t, cos_t,          # 2
            sin_w, cos_w,          # 2
            float(self.trade_count) / max(self.config["max_orders_per_day"], 1),  # 1
            fr_norm,               # 1  ← funding rate normalized [-1, 1]
            fr_ma,                 # 1  ← скользящее среднее FR 24ч
        ], dtype=np.float32)      # итого 22

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    # Технические индикаторы
    # ------------------------------------------------------------------

    def _rsi(self, idx: int, period: int = 14) -> float:
        if idx < period + 1:
            return 0.5
        delta = self.df["close"].iloc[idx - period:idx].diff().dropna()
        gain  = delta[delta > 0].sum()
        loss  = -delta[delta < 0].sum()
        rs    = gain / (loss + 1e-8)
        return float(1.0 - 1.0 / (1.0 + rs))

    def _bb_position(self, idx: int, period: int = 20) -> float:
        if idx < period:
            return 0.0
        window = self.df["close"].iloc[idx - period:idx]
        mean, std = window.mean(), window.std()
        upper = mean + 2 * std
        lower = mean - 2 * std
        if upper == lower:
            return 0.0
        pos = (self.df["close"].iloc[idx] - lower) / (upper - lower) * 2 - 1
        return float(np.clip(pos, -1.0, 1.0))

    def _atr(self, idx: int, period: int = 14) -> float:
        if idx < period + 1:
            return 0.0
        h = self.df["high"].iloc[idx - period:idx].values
        l = self.df["low"].iloc[idx - period:idx].values
        c = self.df["close"].iloc[idx - period - 1:idx - 1].values
        tr = np.maximum(h - l, np.maximum(np.abs(h - c), np.abs(l - c)))
        return float(tr.mean())
