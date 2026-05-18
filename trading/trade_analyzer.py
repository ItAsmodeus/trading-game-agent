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
from data.orderbook_sim import OrderBookSimulator, OrderBookSnapshot


# ──────────────────────────────────────────
# Данные
# ──────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """Текущее состояние рынка для анализа."""
    price:          float
    rsi:            float           # 0–1 (нормализован)
    macd:           float           # нормализованный
    bb_position:    float           # -1 (нижняя) до +1 (верхняя)
    volume_ratio:   float           # log(объём / средний объём)
    ret_1d:         float           # доходность за 1 день
    ret_5d:         float           # доходность за 5 дней
    funding_rate:   float           # текущий funding rate
    position:       float           # текущая позиция (-1=short, 0=нет, 1=long)
    portfolio:      float           # текущий капитал
    peak_portfolio: float           # максимальный капитал
    # Новые поля — уже есть в obs агента
    ret_20d:        float = 0.0     # доходность за 20 дней (тренд)
    volatility:     float = 0.0     # волатильность (20-period std)
    atr:            float = 0.0     # ATR нормализованный
    candle_body:    float = 0.0     # тело свечи (close-open)/close
    # Quantum Leap поля
    z_score:        float = 0.0     # отклонение от справедливой цены
    regime:         float = 0.0     # режим рынка: + bull, - bear
    mu_60:          float = 0.0     # натуральный дрифт (60д)
    sigma_20:       float = 0.03    # волатильность annualized
    ma_cross:       float = 0.0     # MA20 vs MA50
    # Скальперские поля (заполняются в scalper режиме)
    orderbook:      Optional[OrderBookSnapshot] = None
    mode:           str = "swing"   # "swing" / "intraday" / "scalper"
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

    Режимы: swing (дефолт), intraday, scalper.
    В scalper режиме — другой набор сигналов (стакан, VWAP, iceberg).
    """

    MIN_REASONS = 2   # минимум оснований для входа

    def __init__(self, news_fetcher: Optional[NewsFetcher] = None, mode: str = "swing"):
        self.news_fetcher = news_fetcher or NewsFetcher()
        self.mode = mode

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

        # Выбираем набор сигналов по режиму
        mode = snap.mode or self.mode
        if mode == "scalper":
            reasons = self._collect_scalper_reasons(snap, is_long_intent)
            news_signal = None
        else:
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

        # 1. RSI (порог смягчён: 35→40 / 65→60)
        r = self._rsi_reason(snap.rsi, is_long)
        if r: reasons.append(r)

        # 2. MACD (порог смягчён: 0.002→0.001)
        r = self._macd_reason(snap.macd, is_long)
        if r: reasons.append(r)

        # 3. Bollinger Bands (порог смягчён: 0.7→0.6)
        r = self._bb_reason(snap.bb_position, is_long)
        if r: reasons.append(r)

        # 4. Volume
        r = self._volume_reason(snap.volume_ratio)
        if r: reasons.append(r)

        # 5. Моментум краткосрочный (порог смягчён: 8%→5%)
        r = self._momentum_reason(snap.ret_1d, snap.ret_5d, is_long)
        if r: reasons.append(r)

        # 6. Тренд 20 дней (НОВЫЙ — среднесрочное подтверждение)
        r = self._trend_20d_reason(snap.ret_20d, is_long)
        if r: reasons.append(r)

        # 7. Сильная свеча (НОВЫЙ — подтверждение движения телом свечи)
        r = self._candle_body_reason(snap.candle_body, is_long)
        if r: reasons.append(r)

        # 8. Сезонность крипто (НОВЫЙ — исторические паттерны)
        r = self._seasonality_reason(snap.date, is_long)
        if r: reasons.append(r)

        # 9. Низкая волатильность → пробой (НОВЫЙ — период сжатия)
        r = self._low_vol_breakout_reason(snap.volatility, snap.atr, snap.bb_position, is_long)
        if r: reasons.append(r)

        # 10. Funding Rate
        r = self._funding_reason(snap.funding_rate, is_long)
        if r: reasons.append(r)

        # 11. Fear & Greed + новости
        r = self._news_reason(news, is_long)
        if r: reasons.append(r)

        # 12. ⚛️ Z-Score (Quantum Leap)
        r = self._z_score_reason(snap.z_score, is_long)
        if r: reasons.append(r)

        # 13. 📈 Режим рынка (MA200 + дрифт)
        r = self._regime_reason(snap.regime, snap.mu_60, is_long)
        if r: reasons.append(r)

        return reasons

    def _rsi_reason(self, rsi: float, is_long: bool) -> Optional[TradeReason]:
        rsi_pct = rsi * 100  # нормализован 0-1 → 0-100

        if is_long and rsi_pct < 40:      # смягчено: 35 → 40
            strength = (40 - rsi_pct) / 40
            return TradeReason(
                source="Технический", signal="RSI перепроданность",
                description=f"RSI = {rsi_pct:.0f} — актив перепродан, вероятен отскок",
                strength=strength, is_bullish=True,
            )
        if not is_long and rsi_pct > 60:  # смягчено: 65 → 60
            strength = (rsi_pct - 60) / 40
            return TradeReason(
                source="Технический", signal="RSI перекупленность",
                description=f"RSI = {rsi_pct:.0f} — актив перекуплен, вероятна коррекция",
                strength=strength, is_bullish=False,
            )
        return None

    def _macd_reason(self, macd: float, is_long: bool) -> Optional[TradeReason]:
        if is_long and macd > 0.002:      # возвращаем 0.002 — фикс overtrading
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
        if is_long and bb_pos < -0.6:     # смягчено: -0.7 → -0.6
            return TradeReason(
                source="Технический", signal="Bollinger нижняя граница",
                description=f"Цена у нижней границы Bollinger Bands (позиция {bb_pos:.2f}) — зона поддержки",
                strength=abs(bb_pos), is_bullish=True,
            )
        if not is_long and bb_pos > 0.6:  # смягчено: 0.7 → 0.6
            return TradeReason(
                source="Технический", signal="Bollinger верхняя граница",
                description=f"Цена у верхней границы Bollinger Bands (позиция {bb_pos:.2f}) — зона сопротивления",
                strength=bb_pos, is_bullish=False,
            )
        return None

    def _volume_reason(self, vol_ratio: float) -> Optional[TradeReason]:
        actual_ratio = np.exp(vol_ratio)  # денормализуем log
        if actual_ratio > 1.3:            # смягчено: 1.5 → 1.3
            return TradeReason(
                source="Объём", signal="Повышенный объём",
                description=f"Объём торгов в {actual_ratio:.1f}x выше среднего — подтверждает движение",
                strength=min(1.0, (actual_ratio - 1) / 2), is_bullish=True,
            )
        return None

    def _momentum_reason(self, ret_1d: float, ret_5d: float, is_long: bool) -> Optional[TradeReason]:
        if is_long and ret_5d < -0.05 and ret_1d > 0:   # смягчено: 8% → 5%
            return TradeReason(
                source="Моментум", signal="Отскок после падения",
                description=f"Актив упал на {ret_5d:.1%} за 5 дней и сегодня показывает отскок ({ret_1d:+.1%}) — потенциальный разворот",
                strength=min(1.0, abs(ret_5d) * 8), is_bullish=True,
            )
        if not is_long and ret_5d > 0.05 and ret_1d < 0:  # смягчено: 8% → 5%
            return TradeReason(
                source="Моментум", signal="Разворот после роста",
                description=f"Актив вырос на {ret_5d:.1%} за 5 дней и сегодня разворачивается ({ret_1d:+.1%}) — потенциальная коррекция",
                strength=min(1.0, ret_5d * 8), is_bullish=False,
            )
        return None

    def _trend_20d_reason(self, ret_20d: float, is_long: bool) -> Optional[TradeReason]:
        """НОВЫЙ: среднесрочный тренд за 20 дней."""
        if is_long and ret_20d > 0.05:    # рост >5% за месяц = бычий тренд
            return TradeReason(
                source="Тренд", signal="Среднесрочный бычий тренд",
                description=f"Актив вырос на {ret_20d:.1%} за 20 дней — среднесрочный восходящий тренд",
                strength=min(1.0, ret_20d * 5), is_bullish=True,
            )
        if not is_long and ret_20d < -0.05:  # падение >5% за месяц = медвежий
            return TradeReason(
                source="Тренд", signal="Среднесрочный медвежий тренд",
                description=f"Актив упал на {ret_20d:.1%} за 20 дней — среднесрочный нисходящий тренд",
                strength=min(1.0, abs(ret_20d) * 5), is_bullish=False,
            )
        return None

    def _candle_body_reason(self, candle_body: float, is_long: bool) -> Optional[TradeReason]:
        """НОВЫЙ: сильное тело свечи = подтверждение движения."""
        if is_long and candle_body > 0.015:   # зелёная свеча >1.5%
            return TradeReason(
                source="Свеча", signal="Сильная бычья свеча",
                description=f"Сегодняшняя свеча имеет сильное бычье тело ({candle_body:.1%}) — подтверждение покупателей",
                strength=min(1.0, candle_body * 20), is_bullish=True,
            )
        if not is_long and candle_body < -0.015:  # красная свеча >1.5%
            return TradeReason(
                source="Свеча", signal="Сильная медвежья свеча",
                description=f"Сегодняшняя свеча имеет сильное медвежье тело ({candle_body:.1%}) — подтверждение продавцов",
                strength=min(1.0, abs(candle_body) * 20), is_bullish=False,
            )
        return None

    def _seasonality_reason(self, date: Optional[str], is_long: bool) -> Optional[TradeReason]:
        """НОВЫЙ: исторические сезонные паттерны крипто."""
        if not date:
            return None
        try:
            month = int(date[5:7])
        except (ValueError, IndexError):
            return None

        # Бычьи месяцы (исторически): январь, апрель, октябрь, ноябрь
        BULL_MONTHS = {
            1:  ("Январь", "исторически сильный старт года для BTC"),
            4:  ("Апрель", "исторически сильный месяц — весенний рост"),
            10: ("Октябрь", "Uptober — исторически лучший месяц BTC"),
            11: ("Ноябрь", "Q4 бычий сезон — исторически +60% в среднем"),
        }
        # Медвежьи месяцы: сентябрь, август
        BEAR_MONTHS = {
            9:  ("Сентябрь", "Rektember — исторически худший месяц BTC"),
            8:  ("Август", "исторически слабый период перед осенью"),
        }

        if is_long and month in BULL_MONTHS:
            name, reason = BULL_MONTHS[month]
            return TradeReason(
                source="Сезонность", signal=f"Бычий месяц ({name})",
                description=f"{name} — {reason}",
                strength=0.5, is_bullish=True,
            )
        if not is_long and month in BEAR_MONTHS:
            name, reason = BEAR_MONTHS[month]
            return TradeReason(
                source="Сезонность", signal=f"Медвежий месяц ({name})",
                description=f"{name} — {reason}",
                strength=0.5, is_bullish=False,
            )
        return None

    def _low_vol_breakout_reason(
        self, volatility: float, atr: float, bb_pos: float, is_long: bool
    ) -> Optional[TradeReason]:
        """НОВЫЙ: период низкой волатильности → потенциальный пробой."""
        if volatility <= 0 or atr <= 0:
            return None
        # Низкий ATR + цена у края BB = пружина сжата, готовится пробой
        if is_long and atr < 0.015 and bb_pos < -0.4:
            return TradeReason(
                source="Волатильность", signal="Сжатие → пробой вверх",
                description=f"Низкая волатильность (ATR={atr:.3f}) + цена у нижней BB ({bb_pos:.2f}) — период сжатия перед взрывным ростом",
                strength=0.6, is_bullish=True,
            )
        if not is_long and atr < 0.015 and bb_pos > 0.4:
            return TradeReason(
                source="Волатильность", signal="Сжатие → пробой вниз",
                description=f"Низкая волатильность (ATR={atr:.3f}) + цена у верхней BB ({bb_pos:.2f}) — период сжатия перед взрывным падением",
                strength=0.6, is_bullish=False,
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

    # ──────────────────────────────────────────
    # SCALPER — сигналы стакана
    # ──────────────────────────────────────────

    def _collect_scalper_reasons(self, snap: MarketSnapshot, is_long: bool) -> list[TradeReason]:
        """Скальперские сигналы: стакан, VWAP, iceberg, тик-моментум."""
        reasons = []
        ob = snap.orderbook

        if ob is None:
            # Нет стакана — используем упрощённые сигналы
            r = self._macd_reason(snap.macd, is_long)
            if r: reasons.append(r)
            r = self._volume_reason(snap.volume_ratio)
            if r: reasons.append(r)
            return reasons

        # 1. Order Book Imbalance (OBI) — главный скальперский сигнал
        r = self._obi_reason(ob.obi, is_long)
        if r: reasons.append(r)

        # 2. VWAP — цена выше/ниже среднего
        r = self._vwap_reason(ob.price_vs_vwap, is_long)
        if r: reasons.append(r)

        # 3. Iceberg ордера — крупный игрок в рынке
        r = self._iceberg_reason(ob.iceberg_signal, is_long)
        if r: reasons.append(r)

        # 4. Тик-моментум — последовательные движения
        r = self._tick_momentum_reason(ob.tick_direction, is_long)
        if r: reasons.append(r)

        # 5. Спред — ликвидность рынка
        r = self._spread_reason(ob.spread)
        if r: reasons.append(r)

        # 6. Объём (работает на всех таймфреймах)
        r = self._volume_reason(snap.volume_ratio)
        if r: reasons.append(r)

        # 7. Funding rate (даже для скальпера)
        r = self._funding_reason(snap.funding_rate, is_long)
        if r: reasons.append(r)

        return reasons

    def _obi_reason(self, obi: float, is_long: bool) -> Optional[TradeReason]:
        """Order Book Imbalance — давление покупателей vs продавцов."""
        if is_long and obi > 0.3:
            return TradeReason(
                source="Стакан", signal="OBI бычий",
                description=f"Дисбаланс стакана {obi:+.2f} — покупателей значительно больше чем продавцов, цена будет расти",
                strength=min(1.0, obi), is_bullish=True,
            )
        if not is_long and obi < -0.3:
            return TradeReason(
                source="Стакан", signal="OBI медвежий",
                description=f"Дисбаланс стакана {obi:+.2f} — продавцов значительно больше чем покупателей, цена будет падать",
                strength=min(1.0, abs(obi)), is_bullish=False,
            )
        return None

    def _vwap_reason(self, price_vs_vwap: float, is_long: bool) -> Optional[TradeReason]:
        """VWAP — отклонение цены от средневзвешенной."""
        if is_long and price_vs_vwap < -0.003:    # цена на 0.3%+ ниже VWAP
            return TradeReason(
                source="VWAP", signal="Цена ниже VWAP",
                description=f"Цена на {abs(price_vs_vwap):.2%} ниже VWAP — актив временно дёшев, вероятен возврат к VWAP",
                strength=min(1.0, abs(price_vs_vwap) * 100), is_bullish=True,
            )
        if not is_long and price_vs_vwap > 0.003:  # цена на 0.3%+ выше VWAP
            return TradeReason(
                source="VWAP", signal="Цена выше VWAP",
                description=f"Цена на {price_vs_vwap:.2%} выше VWAP — актив временно дорог, вероятен возврат к VWAP",
                strength=min(1.0, price_vs_vwap * 100), is_bullish=False,
            )
        return None

    def _iceberg_reason(self, iceberg_signal: str, is_long: bool) -> Optional[TradeReason]:
        """Iceberg ордера — крупный игрок скрывает объём."""
        if is_long and iceberg_signal == "BUY":
            return TradeReason(
                source="Маркетмейкер", signal="Iceberg покупка",
                description="Обнаружен крупный скрытый ордер на покупку (iceberg) — маркетмейкер или кит накапливает позицию",
                strength=0.8, is_bullish=True,
            )
        if not is_long and iceberg_signal == "SELL":
            return TradeReason(
                source="Маркетмейкер", signal="Iceberg продажа",
                description="Обнаружен крупный скрытый ордер на продажу (iceberg) — маркетмейкер или кит распределяет позицию",
                strength=0.8, is_bullish=False,
            )
        return None

    def _tick_momentum_reason(self, tick_direction: float, is_long: bool) -> Optional[TradeReason]:
        """Тик-моментум — последовательность роста/падения."""
        if is_long and tick_direction > 0.5:
            return TradeReason(
                source="Тик-моментум", signal="Бычий тик-поток",
                description=f"Последовательные бычьи тики (направление {tick_direction:+.2f}) — покупатели контролируют рынок",
                strength=tick_direction, is_bullish=True,
            )
        if not is_long and tick_direction < -0.5:
            return TradeReason(
                source="Тик-моментум", signal="Медвежий тик-поток",
                description=f"Последовательные медвежьи тики (направление {tick_direction:+.2f}) — продавцы контролируют рынок",
                strength=abs(tick_direction), is_bullish=False,
            )
        return None

    def _z_score_reason(self, z_score: float, is_long: bool) -> Optional[TradeReason]:
        """⚛️ Quantum Leap: цена далека от справедливой (Log-Normal модель)."""
        if is_long and z_score < -1.5:
            return TradeReason(
                source="⚛️ Quantum", signal="Z-Score: цена занижена",
                description=f"Цена на {abs(z_score):.1f}σ ниже справедливой (Log-Normal модель) — математически дёшево",
                strength=min(1.0, abs(z_score) / 3), is_bullish=True,
            )
        if not is_long and z_score > 1.5:
            return TradeReason(
                source="⚛️ Quantum", signal="Z-Score: цена завышена",
                description=f"Цена на {z_score:.1f}σ выше справедливой (Log-Normal модель) — математически дорого",
                strength=min(1.0, z_score / 3), is_bullish=False,
            )
        return None

    def _regime_reason(self, regime: float, mu_60: float, is_long: bool) -> Optional[TradeReason]:
        """📈 Режим рынка: цена vs MA(200) + натуральный дрифт."""
        if is_long and regime > 0.05 and mu_60 > 0:
            return TradeReason(
                source="Режим рынка", signal="Бычий режим (MA200)",
                description=f"Цена на {regime:.1%} выше MA(200) и дрифт μ={mu_60*100:.2f}%/день — бычий рынок подтверждён",
                strength=min(1.0, regime * 5), is_bullish=True,
            )
        if not is_long and regime < -0.05 and mu_60 < 0:
            return TradeReason(
                source="Режим рынка", signal="Медвежий режим (MA200)",
                description=f"Цена на {abs(regime):.1%} ниже MA(200) и дрифт μ={mu_60*100:.2f}%/день — медвежий рынок подтверждён",
                strength=min(1.0, abs(regime) * 5), is_bullish=False,
            )
        return None

    def _spread_reason(self, spread: float) -> Optional[TradeReason]:
        """Узкий спред = высокая ликвидность = хорошее время для входа."""
        if spread < 0.0005:   # спред < 0.05%
            return TradeReason(
                source="Ликвидность", signal="Узкий спред",
                description=f"Спред {spread:.3%} — высокая ликвидность рынка, минимальные издержки на вход",
                strength=1.0 - spread * 2000, is_bullish=True,
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
