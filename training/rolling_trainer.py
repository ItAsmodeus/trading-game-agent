"""
Rolling Trainer — скользящее обучение.

Вместо одного статичного обучения на всей истории:
  - Делим данные на окна по 6 месяцев
  - Обучаем на каждом окне по 500k шагов
  - Сдвигаем окно на 2 месяца вперёд
  - Transfer learning: каждое следующее окно стартует с весов предыдущего

Результат товарища: win rate 27% → 59% от этого одного изменения.
"""

import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from data.loader import load_from_exchange, load_from_csv
from game.environment import TradingEnv
from game.rules import GAME_RULES


@dataclass
class WindowResult:
    window_id:    int
    train_start:  str
    train_end:    str
    val_sharpe:   float
    val_return:   float
    val_drawdown: float
    n_trades:     int
    model_path:   str


@dataclass
class RollingReport:
    windows:          list[WindowResult] = field(default_factory=list)
    mean_sharpe:      float = 0.0
    mean_return:      float = 0.0
    best_sharpe:      float = 0.0
    worst_sharpe:     float = 0.0
    consistent_wins:  int   = 0   # окон с Sharpe > 1.0

    def summarize(self):
        if not self.windows:
            return
        sharpes = [w.val_sharpe for w in self.windows]
        returns = [w.val_return for w in self.windows]
        self.mean_sharpe     = float(np.mean(sharpes))
        self.mean_return     = float(np.mean(returns))
        self.best_sharpe     = float(np.max(sharpes))
        self.worst_sharpe    = float(np.min(sharpes))
        self.consistent_wins = sum(1 for s in sharpes if s > 1.0)

    def print(self):
        self.summarize()
        print("\n" + "="*60)
        print("ROLLING TRAINING REPORT")
        print("="*60)
        print(f"  Окон обучено:       {len(self.windows)}")
        print(f"  Средний Sharpe:     {self.mean_sharpe:.3f}")
        print(f"  Средняя доходность: {self.mean_return:+.1%}")
        print(f"  Лучший Sharpe:      {self.best_sharpe:.3f}")
        print(f"  Худший Sharpe:      {self.worst_sharpe:.3f}")
        print(f"  Окон с Sharpe>1.0:  {self.consistent_wins}/{len(self.windows)}")
        print("="*60)
        print("\nДетали по окнам:")
        for w in self.windows:
            status = "✅" if w.val_sharpe > 1.0 else "⚠️ " if w.val_sharpe > 0 else "❌"
            print(f"  {status} Окно {w.window_id}: {w.train_start[:7]}→{w.train_end[:7]} "
                  f"| Sharpe={w.val_sharpe:.2f} | Return={w.val_return:+.1%} "
                  f"| DD={w.val_drawdown:.1%} | Сделок={w.n_trades}")


class RollingTrainer:
    """
    Скользящее обучение RL-агента на временных рядах.

    Параметры:
        train_months  — длина обучающего окна (дефолт 6 месяцев)
        step_months   — шаг сдвига окна (дефолт 2 месяца)
        timesteps_per_window — шагов обучения на окно (дефолт 500k)
        n_envs        — параллельных сред (дефолт 4)
    """

    def __init__(
        self,
        train_months:          int = 6,
        step_months:           int = 2,
        timesteps_per_window:  int = 500_000,
        n_envs:                int = 4,
        checkpoints_dir:       str = "./checkpoints/rolling",
        config:                dict = None,
    ):
        self.train_months         = train_months
        self.step_months          = step_months
        self.timesteps_per_window = timesteps_per_window
        self.n_envs               = n_envs
        self.checkpoints_dir      = Path(checkpoints_dir)
        self.config               = config or GAME_RULES
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _bars_per_month(self, df: pd.DataFrame) -> int:
        """Вычисляет сколько баров примерно в одном месяце."""
        if len(df) < 2:
            return 30
        total_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
        bars_per_day = len(df) / max(total_days, 1)
        return max(1, int(bars_per_day * 30))

    def _make_windows(self, df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Нарезает данные на (train, val) окна со сдвигом."""
        bpm        = self._bars_per_month(df)
        train_bars = self.train_months * bpm
        step_bars  = self.step_months  * bpm
        val_bars   = max(step_bars, bpm)  # val = 1 шаг вперёд

        windows = []
        start = 0
        while start + train_bars + val_bars <= len(df):
            train = df.iloc[start : start + train_bars].reset_index(drop=True)
            val   = df.iloc[start + train_bars : start + train_bars + val_bars].reset_index(drop=True)
            if len(train) >= 60 and len(val) >= 20:  # минимум данных
                windows.append((train, val))
            start += step_bars

        return windows

    def _build_model(self, env, previous_model: Optional[PPO] = None) -> PPO:
        """Создаёт новую модель или продолжает обучение предыдущей (transfer learning)."""
        if previous_model is not None:
            # Transfer learning: копируем веса в новую среду
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=1e-4,   # меньший LR для дообучения
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,       # меньше энтропии — меньше exploration
                tensorboard_log="./logs/tensorboard_rolling",
                verbose=0,
            )
            # Копируем параметры политики
            model.policy.load_state_dict(
                copy.deepcopy(previous_model.policy.state_dict())
            )
        else:
            # Первое окно — обучаем с нуля
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log="./logs/tensorboard_rolling",
                verbose=0,
            )
        return model

    def _evaluate_window(self, model: PPO, val_df: pd.DataFrame) -> dict:
        """Оценивает модель на val данных. Возвращает метрики."""
        env     = TradingEnv(val_df, self.config)
        obs, _  = env.reset()
        portfolio_history = [self.config["starting_capital"]]
        done    = False
        n_trades = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            portfolio_history.append(info["portfolio"])
            n_trades = info["trade_count"]
            done = terminated or truncated

        returns = pd.Series(portfolio_history).pct_change().dropna()
        sharpe  = _sharpe(returns)
        total_return = (portfolio_history[-1] / portfolio_history[0]) - 1
        peak    = max(portfolio_history)
        drawdown = (peak - min(portfolio_history)) / peak if peak > 0 else 0

        return {
            "sharpe":   sharpe,
            "return":   total_return,
            "drawdown": drawdown,
            "n_trades": n_trades,
        }

    def train(self, df: pd.DataFrame) -> tuple[PPO, RollingReport]:
        """
        Основной метод. Принимает полный датафрейм, возвращает лучшую модель и отчёт.
        """
        windows = self._make_windows(df)
        if not windows:
            raise ValueError(f"Недостаточно данных для rolling training. Нужно >{self.train_months+self.step_months} месяцев.")

        print(f"\nRolling Training: {len(windows)} окон | "
              f"{self.train_months}мес обучение | {self.step_months}мес шаг | "
              f"{self.timesteps_per_window//1000}k шагов/окно\n")

        report        = RollingReport()
        previous_model: Optional[PPO] = None
        best_model: Optional[PPO]     = None
        best_sharpe   = -np.inf

        for i, (train_df, val_df) in enumerate(windows):
            train_start = str(train_df["timestamp"].iloc[0])[:10]
            train_end   = str(train_df["timestamp"].iloc[-1])[:10]
            print(f"Окно {i+1}/{len(windows)}: {train_start} → {train_end} "
                  f"({len(train_df)} баров обучение, {len(val_df)} val)")

            # Параллельные среды для обучения
            def make_env_fn(d=train_df):
                def _init():
                    return TradingEnv(d, self.config)
                return _init

            n = min(self.n_envs, os.cpu_count() or 4)
            train_envs = make_vec_env(make_env_fn(), n_envs=n)

            model = self._build_model(train_envs, previous_model)
            model.learn(
                total_timesteps=self.timesteps_per_window,
                reset_num_timesteps=(previous_model is None),
                tb_log_name=f"rolling_window_{i+1}",
            )

            # Оценка на val
            metrics = self._evaluate_window(model, val_df)
            sharpe  = metrics["sharpe"]

            # Сохраняем чекпоинт
            model_path = str(self.checkpoints_dir / f"window_{i+1:02d}_sharpe{sharpe:.2f}")
            model.save(model_path)

            result = WindowResult(
                window_id=i + 1,
                train_start=train_start,
                train_end=train_end,
                val_sharpe=sharpe,
                val_return=metrics["return"],
                val_drawdown=metrics["drawdown"],
                n_trades=metrics["n_trades"],
                model_path=model_path,
            )
            report.windows.append(result)

            status = "✅" if sharpe > 1.0 else "⚠️ " if sharpe > 0 else "❌"
            print(f"  {status} Sharpe={sharpe:.3f} | Return={metrics['return']:+.1%} | "
                  f"DD={metrics['drawdown']:.1%} | Сделок={metrics['n_trades']}")

            # Transfer learning: передаём веса следующему окну
            previous_model = model

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model  = copy.deepcopy(model)

        report.print()

        # Сохраняем лучшую модель
        best_path = str(self.checkpoints_dir / "best_rolling_model")
        if best_model:
            best_model.save(best_path)
            print(f"\nЛучшая модель сохранена: {best_path}.zip")

        return best_model, report


def _sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Sharpe ratio (annualized). Обрабатывает нулевую волатильность."""
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-10:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


# ──────────────────────────────────────────────
# CLI: python -m training.rolling_trainer
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rolling Training")
    parser.add_argument("--symbol",    default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--limit",     type=int, default=2000)
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--step-months",  type=int, default=2)
    parser.add_argument("--timesteps",    type=int, default=500_000)
    parser.add_argument("--data",      default=None, help="Путь к CSV (опционально)")
    args = parser.parse_args()

    if args.data:
        df = load_from_csv(args.data)
    else:
        print(f"Загружаю {args.symbol} с Binance...")
        df = load_from_exchange(args.symbol, args.timeframe, limit=args.limit)

    trainer = RollingTrainer(
        train_months=args.train_months,
        step_months=args.step_months,
        timesteps_per_window=args.timesteps,
    )
    best_model, report = trainer.train(df)
