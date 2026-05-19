"""
MilestoneTracker — curriculum learning через промежуточные цели.

Проблема разреженной награды на 1h: агент получает слабый сигнал из PnL.
Решение: бонусы за промежуточные достижения создают плотный обучающий сигнал.

Адаптировано для 1h таймфрейма (не копия main bot):
- Survival в часах, не днях
- Streak по прибыльным часам с позицией
- Position sizing milestone (использовал крупную позицию при правильном сигнале)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Milestone:
    id: str
    target: float
    metric: str       # capital | streak | hours | trades
    bonus: float
    achieved: bool = False


MILESTONE_CATALOG = [
    # Капитал — бонусы между v4 (слишком высокие) и v5 (слишком низкие)
    Milestone("first_profit",  10_100,  "capital", 0.015),
    Milestone("up_10pct",      11_000,  "capital", 0.03),
    Milestone("up_25pct",      12_500,  "capital", 0.06),
    Milestone("up_50pct",      15_000,  "capital", 0.10),
    Milestone("double",        20_000,  "capital", 0.15),

    # Выживание в часах (1h свечи) — повышены: reward за сохранение капитала
    Milestone("survive_168",   168,  "hours", 0.04),   # 1 неделя
    Milestone("survive_720",   720,  "hours", 0.08),   # 1 месяц
    Milestone("survive_2160",  2160, "hours", 0.15),   # 3 месяца

    # Серия прибыльных часов
    Milestone("streak_3",  3,  "streak", 0.02),
    Milestone("streak_5",  5,  "streak", 0.04),
    Milestone("streak_10", 10, "streak", 0.08),

    # Минимальная активность — trades_50/100 удалены (v5: поощряли оверторговлю)
    Milestone("trades_10",  10,  "trades", 0.005),
]


class MilestoneTracker:
    def __init__(self, starting_capital: float = 10_000.0):
        self.starting_capital = starting_capital
        self.milestones = [Milestone(**vars(m)) for m in MILESTONE_CATALOG]
        self.win_streak = 0
        self._prev_portfolio = starting_capital
        self._prev_had_position = False

    def update(self, portfolio: float, hours_alive: int, n_trades: int) -> float:
        bonus = 0.0

        # Обновляем серию: считаем часы где портфель рос с открытой позицией
        if portfolio > self._prev_portfolio:
            self.win_streak += 1
        else:
            self.win_streak = 0
        self._prev_portfolio = portfolio

        for m in self.milestones:
            if m.achieved:
                continue
            if m.metric == "capital" and portfolio >= m.target:
                m.achieved = True
                bonus += m.bonus
            elif m.metric == "hours" and hours_alive >= m.target:
                m.achieved = True
                bonus += m.bonus
            elif m.metric == "streak" and self.win_streak >= m.target:
                m.achieved = True
                bonus += m.bonus
            elif m.metric == "trades" and n_trades >= m.target:
                m.achieved = True
                bonus += m.bonus

        return bonus

    def reset(self):
        for m in self.milestones:
            m.achieved = False
        self.win_streak = 0
        self._prev_portfolio = self.starting_capital
