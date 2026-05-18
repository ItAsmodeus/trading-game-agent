"""
Milestones — игровые достижения агента.

Зачем это нужно для RL:
  Одна далёкая цель ($1M) = sparse reward → агент не учится.
  Промежуточные цели = dense reward → агент учится быстрее.
  Это называется Curriculum Learning — от простого к сложному.

Каждый милстоун даёт бонус к reward при достижении.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Milestone:
    id:          str
    name:        str
    description: str
    target:      float        # целевое значение
    metric:      str          # "capital" / "sharpe" / "streak" / "trades" / "return_pct"
    reward_bonus: float       # бонус к reward при достижении
    emoji:       str
    score_pts:   int          # очки за достижение
    achieved:    bool = False
    achieved_at: Optional[int] = None  # шаг на котором достигнуто


# ──────────────────────────────────────────
# Каталог милстоунов
# ──────────────────────────────────────────

MILESTONES = [
    # Капитал
    Milestone("first_profit",  "Первая прибыль",    "Капитал вырос выше стартового",     10_100,  "capital",    0.05, "💰", 100),
    Milestone("rising_star",   "Восходящая звезда", "+25% к стартовому капиталу",        12_500,  "capital",    0.10, "⭐", 250),
    Milestone("pro_trader",    "Профи-трейдер",     "+50% к стартовому капиталу",        15_000,  "capital",    0.20, "🎯", 500),
    Milestone("double_up",     "Удвоение",          "Удвоил стартовый капитал",          20_000,  "capital",    0.30, "🚀", 1000),
    Milestone("wealth",        "Состояние",         "Капитал $50k",                      50_000,  "capital",    0.50, "💎", 2500),
    Milestone("champion",      "Чемпион",           "Капитал $100k",                    100_000,  "capital",    1.00, "🏆", 5000),
    Milestone("legendary",     "Легендарный",       "Капитал $1M — ПОБЕДА!",          1_000_000,  "capital",    5.00, "👑", 50000),

    # Качество торговли
    Milestone("quality_trader","Качественный трейд","Sharpe > 1.0",                       1.0,   "sharpe",    0.10, "📊", 300),
    Milestone("sharp_trader",  "Острый трейдер",    "Sharpe > 1.5",                       1.5,   "sharpe",    0.20, "⚡", 600),
    Milestone("elite_trader",  "Элитный трейдер",   "Sharpe > 2.0",                       2.0,   "sharpe",    0.40, "🌟", 1200),

    # Серия побед
    Milestone("streak_3",      "Серия x3",          "3 прибыльные сделки подряд",           3,   "streak",    0.05, "🔥", 150),
    Milestone("streak_5",      "Серия x5",          "5 прибыльных сделок подряд",           5,   "streak",    0.10, "🔥🔥", 300),
    Milestone("streak_10",     "Серия x10",         "10 прибыльных сделок подряд",         10,   "streak",    0.20, "🔥🔥🔥", 750),

    # Выживаемость
    Milestone("survivor_30",   "Выживший",          "30 дней без game over",               30,   "days",      0.05, "🛡️", 200),
    Milestone("survivor_60",   "Стойкий",           "60 дней без game over",               60,   "days",      0.10, "🏰", 400),
    Milestone("survivor_100",  "Непобедимый",       "100 дней без game over",             100,   "days",      0.20, "⚔️", 800),

    # Активность
    Milestone("active_10",     "Активный",          "10 завершённых сделок",               10,   "trades",    0.02, "📈", 50),
    Milestone("active_50",     "Опытный",           "50 завершённых сделок",               50,   "trades",    0.05, "📊", 150),
    Milestone("active_100",    "Ветеран",           "100 завершённых сделок",             100,   "trades",    0.10, "🎖️", 300),
]


class MilestoneTracker:
    """
    Отслеживает достижения агента и начисляет бонусы к reward.
    """

    def __init__(self, starting_capital: float = 10_000.0):
        self.starting_capital = starting_capital
        self.milestones = [Milestone(**vars(m)) for m in MILESTONES]
        self.total_score = 0
        self.win_streak  = 0
        self.days_alive  = 0
        self.last_trade_profitable = False
        self._prev_portfolio = starting_capital

    def update(
        self,
        portfolio:    float,
        trade_count:  int,
        sharpe:       float,
        step:         int,
    ) -> float:
        """
        Проверяет достижения. Возвращает суммарный бонус к reward.
        Вызывать на каждом шаге среды.
        """
        bonus = 0.0
        self.days_alive = step

        # Обновляем серию побед
        if portfolio > self._prev_portfolio and trade_count > 0:
            self.win_streak += 1
        elif portfolio < self._prev_portfolio and trade_count > 0:
            self.win_streak = 0
        self._prev_portfolio = portfolio

        for m in self.milestones:
            if m.achieved:
                continue

            achieved = False
            if m.metric == "capital" and portfolio >= m.target:
                achieved = True
            elif m.metric == "sharpe" and sharpe >= m.target:
                achieved = True
            elif m.metric == "streak" and self.win_streak >= m.target:
                achieved = True
            elif m.metric == "days" and self.days_alive >= m.target:
                achieved = True
            elif m.metric == "trades" and trade_count >= m.target:
                achieved = True

            if achieved:
                m.achieved    = True
                m.achieved_at = step
                bonus += m.reward_bonus
                self.total_score += m.score_pts

        return bonus

    def achieved_list(self) -> list[Milestone]:
        return [m for m in self.milestones if m.achieved]

    def next_milestone(self) -> Optional[Milestone]:
        """Следующий незакрытый милстоун по капиталу."""
        capital_ms = [m for m in self.milestones if m.metric == "capital" and not m.achieved]
        return capital_ms[0] if capital_ms else None

    def progress_to_next(self, portfolio: float) -> float:
        """Прогресс к следующему капитальному милстоуну [0.0, 1.0]."""
        nxt = self.next_milestone()
        if not nxt:
            return 1.0
        prev_target = self.starting_capital
        achieved = [m for m in self.milestones if m.metric == "capital" and m.achieved]
        if achieved:
            prev_target = max(m.target for m in achieved)
        span = nxt.target - prev_target
        done = portfolio - prev_target
        return float(np.clip(done / max(span, 1), 0.0, 1.0))
