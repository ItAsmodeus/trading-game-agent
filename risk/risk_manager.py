"""
Risk Manager — квантовый аналитик проекта.

Отвечает за:
  - Kelly criterion: оптимальный размер позиции
  - Дневные лимиты потерь: автостоп при DD > 3% за день
  - VaR: максимальная потеря с вероятностью 99%
  - Комментарии к каждому решению агента

Правило №1: сначала не потерять, потом заработать.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Decision(Enum):
    ALLOW       = "ALLOW"
    REDUCE      = "REDUCE"        # уменьшить размер
    DENY_DAILY  = "DENY_DAILY"    # дневной лимит исчерпан
    DENY_DD     = "DENY_DD"       # максимальная просадка
    DENY_SIZE   = "DENY_SIZE"     # слишком большой ордер


@dataclass
class RiskComment:
    """Комментарий риск-аналитика к каждому решению."""
    decision:       Decision
    recommended_pct: float          # рекомендуемый размер позиции [0, 1]
    kelly_full:     float
    kelly_half:     float
    daily_pnl_pct:  float
    drawdown_pct:   float
    var_99:         float
    warnings:       list[str] = field(default_factory=list)

    def print(self, symbol: str = "BTC/USDT"):
        icon = {"ALLOW": "✅", "REDUCE": "⚠️ ", "DENY_DAILY": "🛑",
                "DENY_DD": "🛑", "DENY_SIZE": "🛑"}.get(self.decision.value, "?")
        print(f"\n[RISK ANALYST] {symbol}")
        print(f"  Решение:         {icon} {self.decision.value}")
        print(f"  Рек. размер:     {self.recommended_pct:.0%} от капитала")
        print(f"  Kelly full/half: {self.kelly_full:.1%} / {self.kelly_half:.1%}")
        print(f"  P&L за день:     {self.daily_pnl_pct:+.2%}")
        print(f"  Просадка:        {self.drawdown_pct:.2%}")
        print(f"  VaR(99%, 1д):    {self.var_99:.2%}")
        for w in self.warnings:
            print(f"  ⚠️  {w}")


class RiskManager:
    """
    Централизованный риск-менеджер.

    Использование:
        rm = RiskManager(capital=10_000)
        comment = rm.check(action=1, win_rate=0.55, win_loss_ratio=1.5)
        comment.print()
        if comment.decision == Decision.ALLOW:
            size = comment.recommended_pct * capital
    """

    def __init__(
        self,
        capital:            float = 10_000.0,
        max_risk_per_trade: float = 0.02,    # 2% на сделку
        daily_loss_limit:   float = 0.03,    # стоп при -3% за день
        max_drawdown:       float = 0.15,    # стоп стратегии при -15%
        max_position_pct:   float = 0.50,    # не более 50% в одной позиции
        max_orders_per_day: int   = 50,      # защита от overtrading и банов
    ):
        self.initial_capital    = capital
        self.capital            = capital
        self.peak_capital       = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_loss_limit   = daily_loss_limit
        self.max_drawdown       = max_drawdown
        self.max_position_pct   = max_position_pct
        self.max_orders_per_day = max_orders_per_day

        self._daily_pnl     = 0.0
        self._orders_today  = 0
        self._returns_log:  list[float] = []

    # ──────────────────────────────────────────
    # Основной метод проверки
    # ──────────────────────────────────────────

    def check(
        self,
        action:          int,
        win_rate:        float = 0.55,
        win_loss_ratio:  float = 1.5,
        funding_rate:    float = 0.0,
        symbol:          str   = "BTC/USDT",
        verbose:         bool  = False,
    ) -> RiskComment:
        """
        Проверяет решение агента. Возвращает RiskComment с рекомендацией.

        action: 0=HOLD, 1=BUY, 2=SELL_SHORT, 3=CLOSE, 4=BUY_HALF, 5=SELL_HALF
        """
        warnings = []

        # Текущие метрики
        daily_pnl_pct  = self._daily_pnl / max(self.capital, 1)
        drawdown_pct   = (self.peak_capital - self.capital) / max(self.peak_capital, 1)
        var_99         = self._calc_var()
        kelly_full     = self._kelly(win_rate, win_loss_ratio)
        kelly_half     = kelly_full / 2

        # Базовый рекомендуемый размер (Kelly/4 — консервативный режим)
        recommended_pct = min(kelly_full / 4, self.max_position_pct)

        # Если HOLD или CLOSE — всегда разрешаем
        if action in (0, 3):
            return RiskComment(
                decision=Decision.ALLOW,
                recommended_pct=0.0,
                kelly_full=kelly_full,
                kelly_half=kelly_half,
                daily_pnl_pct=daily_pnl_pct,
                drawdown_pct=drawdown_pct,
                var_99=var_99,
                warnings=warnings,
            )

        # ── Проверка 1: дневной лимит потерь ──
        if daily_pnl_pct < -self.daily_loss_limit:
            warnings.append(f"Дневной лимит исчерпан: {daily_pnl_pct:.2%} (лимит {-self.daily_loss_limit:.0%})")
            return RiskComment(
                decision=Decision.DENY_DAILY,
                recommended_pct=0.0,
                kelly_full=kelly_full, kelly_half=kelly_half,
                daily_pnl_pct=daily_pnl_pct, drawdown_pct=drawdown_pct,
                var_99=var_99, warnings=warnings,
            )

        # ── Проверка 2: максимальная просадка ──
        if drawdown_pct > self.max_drawdown:
            warnings.append(f"Макс просадка превышена: {drawdown_pct:.2%} (лимит {self.max_drawdown:.0%})")
            return RiskComment(
                decision=Decision.DENY_DD,
                recommended_pct=0.0,
                kelly_full=kelly_full, kelly_half=kelly_half,
                daily_pnl_pct=daily_pnl_pct, drawdown_pct=drawdown_pct,
                var_99=var_99, warnings=warnings,
            )

        # ── Проверка 3: лимит ордеров в день (защита от банов) ──
        if self._orders_today >= self.max_orders_per_day:
            warnings.append(f"Лимит ордеров исчерпан: {self._orders_today}/{self.max_orders_per_day} (защита от бана)")
            return RiskComment(
                decision=Decision.DENY_DAILY,
                recommended_pct=0.0,
                kelly_full=kelly_full, kelly_half=kelly_half,
                daily_pnl_pct=daily_pnl_pct, drawdown_pct=drawdown_pct,
                var_99=var_99, warnings=warnings,
            )

        # ── Проверка 4: высокий funding rate = рынок перегрет ──
        annual_fr = funding_rate * 3 * 365
        if annual_fr > 0.50:   # > 50% годовых — очень высокий
            warnings.append(f"Funding rate высокий ({annual_fr:.0%}/год) — риск разворота, снижаю размер")
            recommended_pct = min(recommended_pct, 0.25)  # не более 25%

        # ── Предупреждения (не блокируют) ──
        if drawdown_pct > self.max_drawdown * 0.7:
            warnings.append(f"Просадка приближается к лимиту: {drawdown_pct:.1%}")

        if daily_pnl_pct < -self.daily_loss_limit * 0.5:
            warnings.append(f"Половина дневного лимита использована: {daily_pnl_pct:.2%}")

        if var_99 < -0.05:
            warnings.append(f"VaR(99%) высокий: {var_99:.2%} — волатильный период")

        # ── Финальное решение ──
        if recommended_pct < 0.05:  # слишком мало смысла торговать
            warnings.append(f"Kelly рекомендует слишком малый размер ({recommended_pct:.1%})")
            decision = Decision.REDUCE
            recommended_pct = 0.10  # минимум 10% если решаем торговать
        else:
            decision = Decision.ALLOW

        comment = RiskComment(
            decision=decision,
            recommended_pct=recommended_pct,
            kelly_full=kelly_full,
            kelly_half=kelly_half,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=drawdown_pct,
            var_99=var_99,
            warnings=warnings,
        )

        if verbose:
            comment.print(symbol)

        return comment

    # ──────────────────────────────────────────
    # Обновление состояния
    # ──────────────────────────────────────────

    def update(self, pnl: float, traded: bool = False):
        """Вызывать после каждой сделки или шага."""
        self.capital    += pnl
        self._daily_pnl += pnl
        self._returns_log.append(pnl / max(self.capital, 1))
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital
        if traded:
            self._orders_today += 1

    def reset_day(self):
        """Вызывать в начале каждого торгового дня."""
        self._daily_pnl    = 0.0
        self._orders_today = 0

    # ──────────────────────────────────────────
    # Математика
    # ──────────────────────────────────────────

    def _kelly(self, win_rate: float, win_loss_ratio: float) -> float:
        """Kelly criterion: f* = (p*b - q) / b"""
        p = np.clip(win_rate, 0.01, 0.99)
        q = 1 - p
        b = max(win_loss_ratio, 0.01)
        kelly = (p * b - q) / b
        return max(0.0, float(kelly))

    def _calc_var(self, confidence: float = 0.99) -> float:
        """Historical VaR(99%, 1 day) на основе истории доходностей."""
        if len(self._returns_log) < 30:
            return -0.03   # дефолт 3% пока нет истории
        returns = pd.Series(self._returns_log[-252:])  # последний год
        return float(returns.quantile(1 - confidence))

    # ──────────────────────────────────────────
    # Отчёт
    # ──────────────────────────────────────────

    def status(self) -> dict:
        """Текущий статус риск-менеджера."""
        return {
            "capital":       self.capital,
            "peak_capital":  self.peak_capital,
            "drawdown":      (self.peak_capital - self.capital) / max(self.peak_capital, 1),
            "daily_pnl":     self._daily_pnl,
            "daily_pnl_pct": self._daily_pnl / max(self.capital, 1),
            "orders_today":  self._orders_today,
            "var_99":        self._calc_var(),
            "total_return":  (self.capital / self.initial_capital) - 1,
        }

    def print_status(self):
        s = self.status()
        print("\n[RISK STATUS]")
        print(f"  Капитал:     ${s['capital']:,.0f} ({s['total_return']:+.1%})")
        print(f"  Просадка:    {s['drawdown']:.2%}")
        print(f"  P&L сегодня: {s['daily_pnl_pct']:+.2%}")
        print(f"  Ордеров:     {s['orders_today']}/{self.max_orders_per_day}")
        print(f"  VaR(99%):    {s['var_99']:.2%}")
