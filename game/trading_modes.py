"""
Режимы торговли — конфигурации для разных стилей.

SWING    — дневной таймфрейм, позиции 3-10 дней (текущий режим)
INTRADAY — часовой таймфрейм, позиции внутри дня
SCALPER  — минутный таймфрейм, позиции минуты, стакан

Каждый режим имеет свой набор правил и допустимых сигналов.
"""

from dataclasses import dataclass, field
from typing import Literal

TradingModeType = Literal["swing", "intraday", "scalper"]


@dataclass
class TradingModeConfig:
    name:               str
    timeframe:          str         # ccxt формат: '1d', '1h', '5m'
    limit_bars:         int         # сколько баров качать
    min_hold_bars:      int         # минимум держать позицию
    max_hold_bars:      int         # максимум держать
    max_orders_per_day: int         # защита от банов
    min_delay_seconds:  int         # минимум секунд между ордерами
    stop_loss_pct:      float       # базовый стоп
    take_profit_pct:    float       # базовый тейк
    min_reasons:        int         # минимум оснований для входа
    use_orderbook:      bool        # нужен ли стакан
    use_news:           bool        # нужны ли новости
    use_funding:        bool        # нужен ли funding rate
    description_ru:     str = ""


MODES: dict[TradingModeType, TradingModeConfig] = {

    "swing": TradingModeConfig(
        name               = "SWING",
        timeframe          = "1d",
        limit_bars         = 2000,
        min_hold_bars      = 3,
        max_hold_bars      = 10,
        max_orders_per_day = 50,
        min_delay_seconds  = 60,
        stop_loss_pct      = 0.03,
        take_profit_pct    = 0.06,
        min_reasons        = 2,
        use_orderbook      = False,
        use_news           = True,
        use_funding        = True,
        description_ru     = "Свинг: дневной таймфрейм, позиции 3-10 дней. "
                             "Сигналы: RSI, MACD, Bollinger, Funding Rate, новости, сезонность.",
    ),

    "intraday": TradingModeConfig(
        name               = "INTRADAY",
        timeframe          = "1h",
        limit_bars         = 2000,    # ~83 дня часовых данных
        min_hold_bars      = 1,
        max_hold_bars      = 24,      # максимум 1 день
        max_orders_per_day = 100,
        min_delay_seconds  = 30,
        stop_loss_pct      = 0.015,   # уже 1.5% стоп
        take_profit_pct    = 0.03,    # 3% тейк
        min_reasons        = 2,
        use_orderbook      = False,
        use_news           = True,
        use_funding        = True,
        description_ru     = "Интрадей: часовой таймфрейм, позиции внутри дня. "
                             "Сигналы: RSI/MACD/BB на 1ч, VWAP, объём, funding rate.",
    ),

    "scalper": TradingModeConfig(
        name               = "SCALPER",
        timeframe          = "5m",
        limit_bars         = 2000,    # ~7 дней 5-минутных данных
        min_hold_bars      = 1,
        max_hold_bars      = 12,      # максимум 1 час (12 × 5мин)
        max_orders_per_day = 500,     # скальпер торгует много
        min_delay_seconds  = 30,      # но не меньше 30 сек между ордерами
        stop_loss_pct      = 0.005,   # 0.5% стоп (жёсткий)
        take_profit_pct    = 0.01,    # 1% тейк
        min_reasons        = 2,
        use_orderbook      = True,    # главный источник сигналов
        use_news           = False,   # новости слишком медленные для скальпа
        use_funding        = True,
        description_ru     = "Скальпер: 5-минутный таймфрейм, позиции минуты. "
                             "Сигналы: стакан (OBI), iceberg ордера, VWAP, тик-моментум, объём.",
    ),
}


def get_mode(mode_name: str) -> TradingModeConfig:
    key = mode_name.lower()
    if key not in MODES:
        raise ValueError(f"Режим '{mode_name}' не существует. Доступны: {list(MODES.keys())}")
    return MODES[key]


def get_game_rules(mode: TradingModeConfig) -> dict:
    """Возвращает GAME_RULES адаптированные под режим торговли."""
    from game.rules import GAME_RULES
    rules = dict(GAME_RULES)
    rules["max_orders_per_day"] = mode.max_orders_per_day
    rules["max_drawdown_pause"] = 0.10 if mode.name == "SCALPER" else 0.15
    rules["emergency_close"]    = 0.15 if mode.name == "SCALPER" else 0.25
    return rules
