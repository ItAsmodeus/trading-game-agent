"""
Trade Analyzer — "мозг" который думает перед каждым входом.

Правило профессиональных трейдеров:
  НЕ входить в сделку без:
    1. Минимум 2 независимых основания
    2. Чёткого плана выхода (стоп + тейк)

Анализирует:
  - Технические индикаторы (RSI, MACD, BB, Volume)
  - Funding Rate сигнал
  - Fear & Greed Index + новости
  - Моментум (динамика цены)

Выдаёт:
  - TradeDecision с обоснованием на русском языке
  - Разрешение/запрет на вход
  - План выхода
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from data.news_fetcher import NewsFetcher, NewsSignal


# ──────────────────────────────────────────
# Данные
# ──────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Текущее состояние рынка для анализа."""
    price:          float
    rsi:            float           # 0–100
    macd:           float           # нормализованный
    bb_position:    float           # -1 (нижняя) до +1 (верхняя)
    volume_ratio:   float           # объём / средний объём
    ret_1d:         float           # доходность за 1 день
    ret_5d:         float           # доходность за 5 дней
    funding_rate:   float           # текущий funding rate
    position:       float           # текущая позиция (-1=short, 0=нет, 1=long)
    portfolio:      float           # текущий капитал
    peak_portfolio: float           # максимальный капитал
    date:           Optional[str] = None   # для бэктеста
    symbol:         str = "BTC/USDT"


@dataclass
class TradeReason:
    """Одно основание для входа/выхода."""
    source:      str    # "Технический" / "Funding Rate" / "Новости" / "Моментум"
    signal:      str    # краткое название сигнала
    description: str    # полное описание по-русски
    strength:    float  # сила сигнала 0.0–1.0
    is_bullish:  bool   # бычий или медвежий


@dataclass
class ExitPlan:
    """План выхода из позиции."""
    stop_loss_pct:   float   # % от цены входа
    take_profit_pct: float   # % от цены входа
    max_hold_bars:   int     # максимум баров держать
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0

    def calculate(self, entry_price: float, is_long: bool):
        if is_long:
            self.stop_loss_price   = entry_price * (1 - self.stop_loss_pct)
            self.take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:
            self.stop_loss_price   = entry_price * (1 + self.stop_loss_pct)
            self.take_profit_price = entry_price * (1 - self.take_profit_pct)
        return self

    def description_ru(self, is_long: bool) -> str:
        direction = "лонг" if is_long else "шорт"
        return (
            f"Стоп-лосс: ${self.stop_loss_price:,.0f} (-{self.stop_loss_pct:.1%}) | "
            f"Тейк-профит: ${self.take_profit_price:,.0f} (+{self.take_profit_pct:.1%}) | "
            f"Макс держать: {self.max_hold_bars} баров ({direction})"
        )


@dataclass
class TradeDecision:
    """Полное решение аналитика."""
    action:        int            # 0=HOLD, 1=BUY, 2=SELL_SHORT, 3=CLOSE, 4=BUY_HALF, 5=SELL_HALF
    action_name:   str
    allowed:       bool           # разрешён ли вход (2+ оснований)
    reasons:       list[TradeReason] = field(default_factory=list)
    exit_plan:     Optional[ExitPlan] = None
    veto_reason:   Optional[str] = None   # причина запрета если allowed=False
    confidence:    float = 0.0    # уверенность 0.0–1.0

    def reasons_count(self) -> int:
        return len(self.reasons)

    def summary_ru(self) -> str:
        lines = []
        if self.allowed:
            lines.append(f"✅ РЕШЕНИЕ: {self.action_name} | Уверенность: {self.confidence:.0%}")
            lines.append(f"Оснований: {self.reasons_count()} (необходимо минимум 2)")
            for i, r in enumerate(self.reasons, 1):
                icon = "🟢" if r.is_bullish else "🔴"
                lines.append(f"  {icon} Основание {i} [{r.source}]: {r.description}")
            if self.exit_plan:
                is_long = self.action in (1, 4)
                lines.append(f"📋 Выход: {self.exit_plan.description_ru(is_long)}")
        else:
            lines.append(f"🚫 ВХОД ЗАПРЕЩЁН: {self.veto_reason}")
            lines.append(f"Оснований найдено: {self.reasons_count()} (нужно ≥ 2)")
            if self.reasons:
                lines.append("Найденные сигналы (недостаточно):")
                for r in self.reasons:
                    icon = "🟢" if r.is_bullish else "🔴"
                    lines.append(f"  {icon} [{r.source}]: {r.description}")
        return "\n".join(lines)


# ──────────────────────────────────────────
# Аналитик
# ──────────────────────────────────────────

class TradeAnalyzer:
    """
    Анализирует рыночную ситуацию и принимает взвешенное решение.
    Применяет правило 2+ оснований как у профессиональных трейдеров.
    """

    MIN_REASONS = 2   # минимум оснований для входа

    def __init__(self, news_fetcher: Optional[NewsFetcher] = None):
        self.news_fetcher = news_fetcher or NewsFetcher()

    def analyze(self, snap: MarketSnapshot, raw_action: int) -> TradeDecision:
        """
        Анализирует предложенное действие агента.
        Если оснований < 2 — блокирует вход, меняет на HOLD.
        """
        action_name = {
            0: "HOLD", 1: "BUY (100%)", 2: "SELL SHORT (100%)",
            3: "CLOSE", 4: "BUY (50%)", 5: "SELL SHORT (50%)"
        }.get(raw_action, "HOLD")

        # HOLD и CLOSE не требуют обоснования
        if raw_action in (0, 3):
            return TradeDecision(
                action=raw_action, action_name=action_name,
                allowed=True, confidence=1.0,
            )

        is_long_intent = raw_action in (1, 4)

        # Собираем все сигналы
        news_signal = self.news_fetcher.get_signal(date=snap.date, symbol=snap.symbol.split("/")[0])
        reasons     = self._collect_reasons(snap, news_signal, is_long_intent)

        # Фильтруем: только релевантные для направления
        relevant = [r for r in reasons if r.is_bullish == is_long_intent]

        # Строим план выхода
        exit_plan = self._build_exit_plan(snap, is_long_intent)

        # Проверяем правило 2 оснований
        if len(relevant) >= self.MIN_REASONS:
            confidence = min(1.0, len(relevant) / 4.0 + sum(r.strength for r in relevant) / len(relevant) * 0.5)
            return TradeDecision(
                action=raw_action,
                action_name=action_name,
                allowed=True,
                reasons=relevant,
                exit_plan=exit_plan.calculate(snap.price, is_long_intent),
                confidence=confidence,
            )
        else:
            veto = (
                f"Найдено {len(relevant)} основание(й) — нужно минимум {self.MIN_REASONS}. "
                f"Действие заменено на HOLD."
            )
            return TradeDecision(
                action=0,             # заменяем на HOLD
                action_name="HOLD (заблокировано — мало оснований)",
                allowed=False,
                reasons=relevant,
                veto_reason=veto,
                confidence=0.0,
            )

    # ──────────────────────────────────────────
    # Сигналы
    # ──────────────────────────────────────────

    def _collect_reasons(
        self,
        snap: MarketSnapshot,
        news: NewsSignal,
        is_long: bool,
    ) -> list[TradeReason]:
        reasons = []

        # 1. RSI
        r = self._rsi_reason(snap.rsi, is_long)
        if r: reasons.append(r)

        # 2. MACD
        r = self._macd_reason(snap.macd, is_long)
        if r: reasons.append(r)

        # 3. Bollinger Bands
        r = self._bb_reason(snap.bb_position, is_long)
        if r: reasons.append(r)

        # 4. Volume
        r = self._volume_reason(snap.volume_ratio)
        if r: reasons.append(r)

        # 5. Моментум (цена)
        r = self._momentum_reason(snap.ret_1d, snap.ret_5d, is_long)
        if r: reasons.append(r)

        # 6. Funding Rate
        r = self._funding_reason(snap.funding_rate, is_long)
        if r: reasons.append(r)

        # 7. Fear & Greed + новости
        r = self._news_reason(news, is_long)
        if r: reasons.append(r)

        return reasons

    def _rsi_reason(self, rsi: float, is_long: bool) -> Optional[TradeReason]:
        rsi_pct = rsi * 100  # нормализован 0-1 → 0-100

        if is_long and rsi_pct < 35:
            strength = (35 - rsi_pct) / 35
            return TradeReason(
                source="Технический", signal="RSI перепроданность",
                description=f"RSI = {rsi_pct:.0f} — актив перепродан, вероятен отскок",
                strength=strength, is_bullish=True,
            )
        if not is_long and rsi_pct > 65:
            strength = (rsi_pct - 65) / 35
            return TradeReason(
                source="Технический", signal="RSI перекупленность",
                description=f"RSI = {rsi_pct:.0f} — актив перекуплен, вероятна коррекция",
                strength=strength, is_bullish=False,
            )
        return None

    def _macd_reason(self, macd: float, is_long: bool) -> Optional[TradeReason]:
        if is_long and macd > 0.002:
            return TradeReason(
                source="Технический", signal="MACD бычий",
                description=f"MACD гистограмма положительная ({macd:+.4f}) — бычий импульс",
                strength=min(1.0, macd * 100), is_bullish=True,
            )
        if not is_long and macd < -0.002:
            return TradeReason(
                source="Технический", signal="MACD медвежий",
                description=f"MACD гистограмма отрицательная ({macd:+.4f}) — медвежий импульс",
                strength=min(1.0, abs(macd) * 100), is_bullish=False,
            )
        return None

    def _bb_reason(self, bb_pos: float, is_long: bool) -> Optional[TradeReason]:
        if is_long and bb_pos < -0.7:
            return TradeReason(
                source="Технический", signal="Bollinger нижняя граница",
                description=f"Цена у нижней границы Bollinger Bands (позиция {bb_pos:.2f}) — зона поддержки",
                strength=abs(bb_pos), is_bullish=True,
            )
        if not is_long and bb_pos > 0.7:
            return TradeReason(
                source="Технический", signal="Bollinger верхняя граница",
                description=f"Цена у верхней границы Bollinger Bands (позиция {bb_pos:.2f}) — зона сопротивления",
                strength=bb_pos, is_bullish=False,
            )
        return None

    def _volume_reason(self, vol_ratio: float) -> Optional[TradeReason]:
        actual_ratio = np.exp(vol_ratio)  # денормализуем log
        if actual_ratio > 1.5:
            return TradeReason(
                source="Объём", signal="Повышенный объём",
                description=f"Объём торгов в {actual_ratio:.1f}x выше среднего — подтверждает движение",
                strength=min(1.0, (actual_ratio - 1) / 2), is_bullish=True,
            )
        return None

    def _momentum_reason(self, ret_1d: float, ret_5d: float, is_long: bool) -> Optional[TradeReason]:
        if is_long and ret_5d < -0.08 and ret_1d > 0:
            return TradeReason(
                source="Моментум", signal="Отскок после падения",
                description=f"Актив упал на {ret_5d:.1%} за 5 дней и сегодня показывает отскок ({ret_1d:+.1%}) — потенциальный разворот",
                strength=min(1.0, abs(ret_5d) * 5), is_bullish=True,
            )
        if not is_long and ret_5d > 0.08 and ret_1d < 0:
            return TradeReason(
                source="Моментум", signal="Разворот после роста",
                description=f"Актив вырос на {ret_5d:.1%} за 5 дней и сегодня разворачивается ({ret_1d:+.1%}) — потенциальная коррекция",
                strength=min(1.0, ret_5d * 5), is_bullish=False,
            )
        return None

    def _funding_reason(self, funding_rate: float, is_long: bool) -> Optional[TradeReason]:
        annual = funding_rate * 3 * 365

        if not is_long and annual > 0.50:
            return TradeReason(
                source="Funding Rate", signal="Рынок перегрет лонгами",
                description=f"Funding Rate = {funding_rate:.4%}/8ч ({annual:.0%}/год) — лонги перегреты, высокая вероятность разворота вниз",
                strength=min(1.0, annual), is_bullish=False,
            )
        if is_long and annual < -0.20:
            return TradeReason(
                source="Funding Rate", signal="Рынок перегрет шортами",
                description=f"Funding Rate = {funding_rate:.4%}/8ч ({annual:.0%}/год) — шорты перегреты, вероятен шорт-сквиз вверх",
                strength=min(1.0, abs(annual)), is_bullish=True,
            )
        return None

    def _news_reason(self, news: NewsSignal, is_long: bool) -> Optional[TradeReason]:
        reason_text = news.as_reason()
        if not reason_text:
            return None

        is_bullish_news = (
            news.fear_greed_score <= 25 or    # Extreme Fear = покупай
            news.news_sentiment > 0.3
        )
        is_bearish_news = (
            news.fear_greed_score >= 75 or    # Extreme Greed = продавай
            news.news_sentiment < -0.3
        )

        if is_long and is_bullish_news:
            return TradeReason(
                source="Новости/Сентимент", signal="Бычий фон",
                description=reason_text,
                strength=0.7, is_bullish=True,
            )
        if not is_long and is_bearish_news:
            return TradeReason(
                source="Новости/Сентимент", signal="Медвежий фон",
                description=reason_text,
                strength=0.7, is_bullish=False,
            )
        return None

    def _build_exit_plan(self, snap: MarketSnapshot, is_long: bool) -> ExitPlan:
        """Динамический план выхода на основе волатильности."""
        # Чем выше просадка — тем жёстче стоп
        dd = (snap.peak_portfolio - snap.portfolio) / max(snap.peak_portfolio, 1)
        base_stop = 0.03 + dd * 0.02   # базовый стоп 3%, растёт с просадкой

        return ExitPlan(
            stop_loss_pct=min(base_stop, 0.05),   # не более 5%
            take_profit_pct=base_stop * 2,         # R:R = 1:2
            max_hold_bars=10,
        )
