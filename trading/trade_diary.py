"""
Trade Diary — дневник трейдера в Markdown.

Каждая сделка фиксируется с:
  - Датой и ценой входа
  - Минимум 2 основания (правило профессионала)
  - Планом выхода (стоп + тейк)
  - Новостным контекстом
  - Результатом (заполняется при закрытии)

Файл: TRADE_DIARY.md в корне проекта.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from trading.trade_analyzer import TradeDecision, MarketSnapshot


DIARY_PATH = Path("./TRADE_DIARY.md")

ACTION_EMOJI = {
    0: "⏸️",   # HOLD
    1: "🟢",   # BUY 100%
    2: "🔴",   # SELL SHORT 100%
    3: "⬜",   # CLOSE
    4: "🟡",   # BUY 50%
    5: "🟠",   # SELL SHORT 50%
}

DIRECTION = {
    1: "ЛОНГ 100%", 2: "ШОРТ 100%",
    4: "ЛОНГ 50%",  5: "ШОРТ 50%",
    3: "ЗАКРЫТИЕ",  0: "ПРОПУСК",
}


class TradeDiary:
    """
    Дневник трейдера. Пишет каждую сделку в Markdown файл.
    Подключается к train.py и читается человеком после каждого прогона.
    """

    def __init__(self, path: str = str(DIARY_PATH), session_label: str = ""):
        self.path          = Path(path)
        self.session_label = session_label or datetime.now().strftime("%Y-%m-%d %H:%M")
        self._trade_count  = 0
        self._session_pnl  = 0.0
        self._open_trades: dict[int, dict] = {}  # trade_id → данные открытой позиции
        self._session_started = False

    # ──────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────

    def start_session(self, symbol: str, capital: float, mode: str = "backtest"):
        """Начало новой торговой сессии."""
        self._session_started = True
        self._trade_count = 0
        self._session_pnl = 0.0

        self._append(_session_header(self.session_label, symbol, capital, mode))

    def log_trade(
        self,
        decision:  TradeDecision,
        snap:      MarketSnapshot,
        step:      int,
    ) -> Optional[int]:
        """
        Записывает сделку в дневник.
        Возвращает trade_id если позиция открыта, None если HOLD/заблокировано.
        """
        if not self._session_started:
            self.start_session(snap.symbol, snap.portfolio)

        # HOLD — не логируем как сделку
        if decision.action == 0 and decision.allowed:
            return None

        # Заблокированный вход — логируем кратко
        if not decision.allowed:
            self._append(_blocked_entry(decision, snap, step))
            return None

        # Закрытие позиции
        if decision.action == 3:
            self._log_close(snap, step)
            return None

        # Открытие позиции
        self._trade_count += 1
        trade_id = self._trade_count
        self._open_trades[trade_id] = {
            "entry_price": snap.price,
            "action":      decision.action,
            "step":        step,
            "capital_in":  snap.portfolio,
        }

        self._append(_trade_entry(trade_id, decision, snap, step))
        return trade_id

    def log_close(
        self,
        trade_id:    int,
        exit_price:  float,
        exit_reason: str,
        portfolio:   float,
        step:        int,
    ):
        """Записывает закрытие конкретной позиции."""
        if trade_id not in self._open_trades:
            return

        trade   = self._open_trades.pop(trade_id)
        is_long = trade["action"] in (1, 4)
        pnl_pct = ((exit_price - trade["entry_price"]) / trade["entry_price"])
        if not is_long:
            pnl_pct = -pnl_pct

        pnl_usd = pnl_pct * trade["capital_in"]
        self._session_pnl += pnl_usd

        self._append(_trade_close(trade_id, exit_price, exit_reason, pnl_pct, pnl_usd, portfolio, step))

    def end_session(self, final_portfolio: float, initial_capital: float, total_trades: int):
        """Итог сессии."""
        total_return = (final_portfolio / initial_capital - 1)
        self._append(_session_footer(
            trades=total_trades,
            final_portfolio=final_portfolio,
            total_return=total_return,
            session_pnl=self._session_pnl,
        ))

    # ──────────────────────────────────────────
    # Вспомогательные
    # ──────────────────────────────────────────

    def _append(self, text: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _log_close(self, snap: MarketSnapshot, step: int):
        """Закрытие без конкретного trade_id (для CLOSE action)."""
        if self._open_trades:
            trade_id = list(self._open_trades.keys())[-1]
            self.log_close(trade_id, snap.price, "Агент решил закрыть позицию", snap.portfolio, step)


# ──────────────────────────────────────────
# Форматирование блоков
# ──────────────────────────────────────────

def _session_header(label: str, symbol: str, capital: float, mode: str) -> str:
    mode_ru = {"backtest": "📊 Бэктест", "testnet": "🧪 Testnet", "live": "💰 Live"}.get(mode, mode)
    return f"""
---

## 🗓️ Сессия: {label}
**Инструмент:** {symbol} | **Режим:** {mode_ru} | **Стартовый капитал:** ${capital:,.0f}

"""


def _trade_entry(trade_id: int, decision: TradeDecision, snap: MarketSnapshot, step: int) -> str:
    emoji = ACTION_EMOJI.get(decision.action, "❓")
    direction = DIRECTION.get(decision.action, "?")
    date_str = snap.date or f"Бар #{step}"

    lines = [
        f"### Сделка #{trade_id} {emoji} | {date_str} | {direction}",
        f"",
        f"**Цена входа:** ${snap.price:,.2f} | **Капитал:** ${snap.portfolio:,.0f} | **Позиция до:** {snap.position:+.0f}",
        f"",
        f"#### 📋 Основания для входа ({decision.reasons_count()}/мин.2)",
        f"",
    ]

    for i, r in enumerate(decision.reasons, 1):
        icon = "✅" if r.is_bullish == (decision.action in (1, 4)) else "⚠️"
        lines.append(f"{icon} **Основание {i} [{r.source}]:** {r.description}")
        lines.append(f"   *Сигнал:* {r.signal} | *Сила:* {'█' * int(r.strength * 5)}{'░' * (5 - int(r.strength * 5))} {r.strength:.0%}")
        lines.append("")

    if decision.exit_plan:
        is_long = decision.action in (1, 4)
        lines.append(f"#### 🎯 План выхода")
        lines.append(f"- **Стоп-лосс:** ${decision.exit_plan.stop_loss_price:,.2f} ({-decision.exit_plan.stop_loss_pct:.1%})")
        lines.append(f"- **Тейк-профит:** ${decision.exit_plan.take_profit_price:,.2f} (+{decision.exit_plan.take_profit_pct:.1%})")
        lines.append(f"- **Макс. удержание:** {decision.exit_plan.max_hold_bars} баров")
        lines.append(f"- **Соотношение R:R:** 1:{decision.exit_plan.take_profit_pct/decision.exit_plan.stop_loss_pct:.1f}")
        lines.append("")

    lines.append(f"**Уверенность агента:** {decision.confidence:.0%}")
    lines.append("")
    return "\n".join(lines)


def _trade_close(
    trade_id: int,
    exit_price: float,
    reason: str,
    pnl_pct: float,
    pnl_usd: float,
    portfolio: float,
    step: int,
) -> str:
    icon  = "✅" if pnl_pct > 0 else "❌"
    emoji = "📈" if pnl_pct > 0 else "📉"
    return f"""#### {icon} Закрытие сделки #{trade_id} | Бар #{step}

{emoji} **Цена выхода:** ${exit_price:,.2f} | **P&L:** {pnl_pct:+.2%} (${pnl_usd:+,.0f})
**Причина закрытия:** {reason}
**Капитал после:** ${portfolio:,.0f}

---
"""


def _blocked_entry(decision: TradeDecision, snap: MarketSnapshot, step: int) -> str:
    date_str = snap.date or f"Бар #{step}"
    lines = [
        f"### 🚫 Заблокированный вход | {date_str}",
        f"",
        f"**Агент хотел:** {decision.action_name}",
        f"**Причина блокировки:** {decision.veto_reason}",
        f"",
    ]
    if decision.reasons:
        lines.append("Найденные (но недостаточные) сигналы:")
        for r in decision.reasons:
            lines.append(f"- [{r.source}] {r.description}")
    lines.append("")
    lines.append("*Действие заменено на HOLD. Ждём достаточно оснований.*")
    lines.append("")
    return "\n".join(lines)


def _session_footer(trades: int, final_portfolio: float, total_return: float, session_pnl: float) -> str:
    icon = "🏆" if total_return > 0 else "📉"
    return f"""
---

## {icon} Итог сессии

| Метрика | Значение |
|---------|---------|
| Всего сделок | {trades} |
| Итоговый капитал | ${final_portfolio:,.0f} |
| Общая доходность | {total_return:+.2%} |
| P&L сессии | ${session_pnl:+,.0f} |

---
"""


# ──────────────────────────────────────────
# Инициализация файла
# ──────────────────────────────────────────

def init_diary(path: str = str(DIARY_PATH)):
    """Создаёт файл дневника с шапкой если не существует."""
    p = Path(path)
    if not p.exists():
        with open(p, "w", encoding="utf-8") as f:
            f.write("""# 📒 Дневник трейдера — Trading Game Agent

> **Правило входа:** минимум 2 независимых основания + чёткий план выхода.
> Без этого — нет сделки. Так торгуют профессионалы.

**Источники сигналов:**
- 📊 Технические индикаторы (RSI, MACD, Bollinger Bands, Volume)
- 💰 Funding Rate (перегрев рынка лонгами/шортами)
- 📰 Fear & Greed Index + новостной сентимент
- 📈 Моментум (динамика цены за 1д и 5д)

**Интерпретация результатов:**
- ✅ Сделка открыта (≥2 оснований)
- 🚫 Вход заблокирован (<2 оснований)
- 📈 Прибыльное закрытие
- 📉 Убыточное закрытие

---
""")
        print(f"[Diary] Создан новый дневник: {p}")
