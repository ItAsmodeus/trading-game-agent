"""
MarketSimulator — реалистичное исполнение ордеров.
Учитывает комиссии, slippage, bid/ask spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import CFG


@dataclass
class Position:
    symbol: str
    size: float = 0.0        # в базовой валюте (BTC, ETH, ...)
    entry_price: float = 0.0
    side: str = "none"       # "long", "short", "none"

    @property
    def is_open(self) -> bool:
        return self.size > 0 and self.side != "none"

    def unrealized_pnl(self, current_price: float) -> float:
        if not self.is_open:
            return 0.0
        if self.side == "long":
            return (current_price - self.entry_price) * self.size
        return (self.entry_price - current_price) * self.size

    def value(self, current_price: float) -> float:
        if self.side == "long":
            return self.size * current_price
        elif self.side == "short":
            # Margin (entry * size) + unrealized PnL
            return self.size * self.entry_price + self.unrealized_pnl(current_price)
        return 0.0


@dataclass
class Portfolio:
    cash: float = CFG.STARTING_CAPITAL
    positions: dict = field(default_factory=dict)  # symbol -> Position
    peak_value: float = CFG.STARTING_CAPITAL
    realized_pnl: float = 0.0
    total_fees: float = 0.0
    n_trades: int = 0

    def total_value(self, prices: dict[str, float]) -> float:
        pos_value = sum(
            pos.value(prices.get(pos.symbol, pos.entry_price))
            for pos in self.positions.values()
            if pos.is_open
        )
        return self.cash + pos_value

    def update_peak(self, prices: dict[str, float]) -> None:
        v = self.total_value(prices)
        if v > self.peak_value:
            self.peak_value = v

    def drawdown(self, prices: dict[str, float]) -> float:
        v = self.total_value(prices)
        return (self.peak_value - v) / self.peak_value if self.peak_value > 0 else 0.0


class MarketSimulator:
    """Исполняет ордера с реалистичными параметрами."""

    def __init__(self, portfolio: Optional[Portfolio] = None):
        self.portfolio = portfolio or Portfolio()

    def _execution_price(self, mid_price: float, side: str) -> float:
        """Цена исполнения с учётом slippage."""
        slip = np.random.uniform(0, CFG.SLIPPAGE * 2)
        if side == "buy":
            return mid_price * (1 + slip)
        return mid_price * (1 - slip)

    def _fee(self, notional: float) -> float:
        return notional * CFG.TAKER_FEE

    def buy(self, symbol: str, price: float, capital_ratio: float) -> dict:
        """
        Купить на capital_ratio * свободных средств.
        Возвращает dict с деталями исполнения.
        """
        available = self.portfolio.cash * min(capital_ratio, CFG.MAX_POSITION_RATIO)
        if available < 1.0:
            return {"executed": False, "reason": "insufficient_cash"}

        exec_price = self._execution_price(price, "buy")
        fee = self._fee(available)
        net_cash = available - fee
        size = net_cash / exec_price

        pos = self.portfolio.positions.get(symbol)
        if pos and pos.is_open and pos.side == "short":
            return {"executed": False, "reason": "close_short_first"}
        if pos and pos.is_open and pos.side == "long":
            # Усреднение: пересчитываем среднюю цену входа
            total_size = pos.size + size
            pos.entry_price = (pos.entry_price * pos.size + exec_price * size) / total_size
            pos.size = total_size
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol, size=size, entry_price=exec_price, side="long"
            )

        self.portfolio.cash -= available
        self.portfolio.total_fees += fee
        self.portfolio.n_trades += 1

        return {"executed": True, "price": exec_price, "size": size, "fee": fee}

    def sell(self, symbol: str, price: float, position_ratio: float) -> dict:
        """Продать position_ratio от текущей лонг-позиции."""
        pos = self.portfolio.positions.get(symbol)
        if not pos or not pos.is_open or pos.side != "long":
            return {"executed": False, "reason": "no_long_position"}

        exec_price = self._execution_price(price, "sell")
        sell_size = pos.size * min(position_ratio, 1.0)
        notional = sell_size * exec_price
        fee = self._fee(notional)

        pnl = (exec_price - pos.entry_price) * sell_size - fee
        self.portfolio.cash += notional - fee
        self.portfolio.realized_pnl += pnl
        self.portfolio.total_fees += fee
        self.portfolio.n_trades += 1

        pos.size -= sell_size
        if pos.size < 1e-8:
            pos.size = 0.0
            pos.side = "none"

        return {"executed": True, "price": exec_price, "size": sell_size, "fee": fee, "pnl": pnl}

    def short(self, symbol: str, price: float, capital_ratio: float) -> dict:
        """Открыть/добавить к шорт-позиции. Используем capital_ratio * cash как маржу."""
        pos = self.portfolio.positions.get(symbol)
        if pos and pos.is_open and pos.side == "long":
            return {"executed": False, "reason": "close_long_first"}

        margin = self.portfolio.cash * min(capital_ratio, CFG.MAX_POSITION_RATIO)
        if margin < 1.0:
            return {"executed": False, "reason": "insufficient_cash"}

        exec_price = self._execution_price(price, "sell")
        fee = self._fee(margin)
        net_margin = margin - fee
        size = net_margin / exec_price

        if pos and pos.is_open and pos.side == "short":
            total_size = pos.size + size
            pos.entry_price = (pos.entry_price * pos.size + exec_price * size) / total_size
            pos.size = total_size
        else:
            self.portfolio.positions[symbol] = Position(
                symbol=symbol, size=size, entry_price=exec_price, side="short"
            )

        self.portfolio.cash -= margin
        self.portfolio.total_fees += fee
        self.portfolio.n_trades += 1
        return {"executed": True, "price": exec_price, "size": size, "fee": fee}

    def cover(self, symbol: str, price: float, position_ratio: float) -> dict:
        """Закрыть position_ratio от шорт-позиции (buy to cover)."""
        pos = self.portfolio.positions.get(symbol)
        if not pos or not pos.is_open or pos.side != "short":
            return {"executed": False, "reason": "no_short_position"}

        exec_price = self._execution_price(price, "buy")
        cover_size = pos.size * min(position_ratio, 1.0)
        notional = cover_size * exec_price
        fee = self._fee(notional)

        gross_pnl = (pos.entry_price - exec_price) * cover_size
        margin_returned = pos.entry_price * cover_size
        self.portfolio.cash += margin_returned + gross_pnl - fee
        self.portfolio.realized_pnl += gross_pnl - fee
        self.portfolio.total_fees += fee
        self.portfolio.n_trades += 1

        pos.size -= cover_size
        if pos.size < 1e-8:
            pos.size = 0.0
            pos.side = "none"

        return {"executed": True, "price": exec_price, "size": cover_size, "fee": fee, "pnl": gross_pnl - fee}

    def reset(self) -> None:
        self.portfolio = Portfolio()
