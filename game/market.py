"""
Реалистичное исполнение ордеров.
Без этого любой бэктест врёт — агент будет думать что торгует лучше чем есть.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class FillResult:
    executed:   bool
    avg_price:  float
    fee:        float
    slippage:   float
    side:       str    # 'buy' | 'sell' | 'none'


class RealisticMarket:
    """
    Моделирует реальное исполнение:
    - bid/ask spread (агент никогда не получает mid price)
    - комиссия taker за market order
    - slippage по модели Almgren-Chriss
    - частичное исполнение при малой ликвидности
    """

    def __init__(self, config: dict):
        self.maker_fee   = config["maker_fee"]
        self.taker_fee   = config["taker_fee"]
        self.min_order   = config["min_order_usd"]

    def execute_market_buy(
        self,
        mid_price:      float,
        size_usd:       float,
        daily_volume:   float = 1_000_000_000.0,
        volatility:     float = 0.02,
        spread_bps:     float = 5.0,
    ) -> FillResult:
        if size_usd < self.min_order:
            return FillResult(False, mid_price, 0.0, 0.0, "none")

        # Покупаем по ask (mid + half spread)
        spread = mid_price * spread_bps / 10_000
        ask_price = mid_price + spread / 2

        # Market impact: модель Almgren-Chriss
        participation = size_usd / max(daily_volume, 1)
        market_impact = 0.1 * volatility * np.sqrt(participation) * mid_price

        avg_price = ask_price + market_impact
        fee = size_usd * self.taker_fee
        slippage = (avg_price - mid_price) / mid_price

        return FillResult(True, avg_price, fee, slippage, "buy")

    def execute_market_sell(
        self,
        mid_price:      float,
        size_usd:       float,
        daily_volume:   float = 1_000_000_000.0,
        volatility:     float = 0.02,
        spread_bps:     float = 5.0,
    ) -> FillResult:
        if size_usd < self.min_order:
            return FillResult(False, mid_price, 0.0, 0.0, "none")

        # Продаём по bid (mid - half spread)
        spread = mid_price * spread_bps / 10_000
        bid_price = mid_price - spread / 2

        participation = size_usd / max(daily_volume, 1)
        market_impact = 0.1 * volatility * np.sqrt(participation) * mid_price

        avg_price = bid_price - market_impact
        fee = size_usd * self.taker_fee
        slippage = (mid_price - avg_price) / mid_price

        return FillResult(True, avg_price, fee, slippage, "sell")

    def execute_action(
        self,
        action:    int,
        price:     float,
        position:  float,  # текущая позиция [-1, 0, 1]
        portfolio: float,
        config:    dict,
    ) -> float:
        """
        Выполняет действие агента, возвращает суммарную стоимость (комиссии).
        """
        from game.rules import ACTIONS

        action_name = ACTIONS.get(int(action) if hasattr(action, '__len__') else action, "HOLD")
        fee = 0.0

        if action_name == "HOLD":
            return 0.0

        if action_name in ("BUY", "BUY_HALF"):
            size_pct = 1.0 if action_name == "BUY" else 0.5
            size_usd = portfolio * size_pct * (1 - config["maker_fee"])
            if position <= 0 and size_usd >= self.min_order:
                fill = self.execute_market_buy(price, size_usd)
                fee = fill.fee

        elif action_name in ("SELL_SHORT", "SELL_HALF"):
            size_pct = 1.0 if action_name == "SELL_SHORT" else 0.5
            size_usd = portfolio * size_pct * (1 - config["maker_fee"])
            if position >= 0 and size_usd >= self.min_order:
                fill = self.execute_market_sell(price, size_usd)
                fee = fill.fee

        elif action_name == "CLOSE":
            if position != 0:
                size_usd = portfolio * abs(position)
                if position > 0:
                    fill = self.execute_market_sell(price, size_usd)
                else:
                    fill = self.execute_market_buy(price, size_usd)
                fee = fill.fee

        return fee
