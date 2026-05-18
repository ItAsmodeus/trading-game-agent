"""
Order Book Simulator — симуляция стакана из OHLCV данных.

В реальном трейдинге стакан (L2 книга) содержит все заявки.
В бэктесте исторического стакана нет → симулируем из OHLCV:
  - Spread ≈ ATR × коэффициент ликвидности
  - Глубина ≈ объём торгов
  - OBI (Order Book Imbalance) ≈ из направления движения цены
  - Iceberg паттерны ≈ из повторяющихся объёмных уровней

Для live торговли — подключение через Binance WebSocket (Phase 3).

ВАЖНО: симуляция приблизительна. В реальном скальпинге нужен настоящий стакан.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderBookSnapshot:
    """Снимок состояния стакана."""
    mid_price:          float    # средняя цена
    bid_price:          float    # лучший бид
    ask_price:          float    # лучший аск
    spread:             float    # спред в %
    spread_usd:         float    # спред в USD

    obi:                float    # Order Book Imbalance [-1, +1]
                                 # > 0 = бидов больше (бычье давление)
                                 # < 0 = асков больше (медвежье давление)

    bid_depth:          float    # объём на стороне покупки (симулировано)
    ask_depth:          float    # объём на стороне продажи (симулировано)
    depth_ratio:        float    # bid_depth / ask_depth

    large_bid_detected: bool     # обнаружен крупный бид (возможный iceberg)
    large_ask_detected: bool     # обнаружен крупный аск (возможный iceberg)
    iceberg_signal:     str      # "BUY" / "SELL" / "NONE"

    vwap:               float    # VWAP за последние N баров
    price_vs_vwap:      float    # (price - vwap) / vwap
    tick_direction:     float    # направление последних тиков [-1, +1]


class OrderBookSimulator:
    """
    Симулятор стакана на основе OHLCV данных.

    Используется для бэктеста скальперских стратегий.
    В Phase 3 заменяется на реальный WebSocket стакан.
    """

    def __init__(
        self,
        df:              pd.DataFrame,
        vwap_window:     int   = 20,     # баров для VWAP
        spread_factor:   float = 0.0002, # базовый спред 0.02%
        iceberg_thresh:  float = 2.5,    # объём > 2.5x среднего = крупный
    ):
        self.df             = df.reset_index(drop=True)
        self.vwap_window    = vwap_window
        self.spread_factor  = spread_factor
        self.iceberg_thresh = iceberg_thresh

        self._precompute()

    def _precompute(self):
        """Предварительно вычисляем VWAP и другие производные."""
        df = self.df
        df = df.copy()

        # VWAP (Volume Weighted Average Price)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (
            (typical_price * df["volume"]).rolling(self.vwap_window).sum() /
            df["volume"].rolling(self.vwap_window).sum()
        )

        # ATR для оценки спреда
        high_low = df["high"] - df["low"]
        df["atr_raw"] = high_low.rolling(14).mean()

        # Объём относительно среднего
        df["vol_mean"]  = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / (df["vol_mean"] + 1e-8)

        # Направление движения (tick direction)
        df["ret"]         = df["close"].pct_change()
        df["tick_dir_5"]  = df["ret"].rolling(5).mean()

        self.df = df

    def snapshot(self, idx: int) -> OrderBookSnapshot:
        """
        Возвращает симулированный снимок стакана на баре idx.
        """
        if idx < self.vwap_window + 1:
            return self._neutral_snapshot(float(self.df["close"].iloc[idx]))

        row   = self.df.iloc[idx]
        price = float(row["close"])

        # ── Спред ──────────────────────────────────
        atr_pct = float(row.get("atr_raw", price * 0.001)) / (price + 1e-8)
        spread  = max(self.spread_factor, atr_pct * 0.1)  # спред = 10% от ATR
        spread_usd = price * spread
        bid = price - spread_usd / 2
        ask = price + spread_usd / 2

        # ── Order Book Imbalance ────────────────────
        # Симулируем OBI из баланса High/Low/Close относительно диапазона
        bar_range = max(float(row["high"]) - float(row["low"]), 1e-8)
        close_pos = (price - float(row["low"])) / bar_range  # 0 = у Low, 1 = у High
        # Если цена закрылась у верха → бычье давление (OBI > 0)
        obi_raw = (close_pos - 0.5) * 2  # нормализуем в [-1, +1]

        # Усиливаем OBI через изменение объёма
        vol_ratio = float(row.get("vol_ratio", 1.0))
        ret = float(row.get("ret", 0.0))
        if ret > 0:
            obi = min(1.0, obi_raw + 0.2 * min(vol_ratio, 3.0))
        else:
            obi = max(-1.0, obi_raw - 0.2 * min(vol_ratio, 3.0))

        # ── Глубина стакана ─────────────────────────
        vol_usd = float(row["volume"]) * price
        if obi > 0:
            bid_depth = vol_usd * (0.5 + obi * 0.3)
            ask_depth = vol_usd * (0.5 - obi * 0.3)
        else:
            bid_depth = vol_usd * (0.5 + obi * 0.3)
            ask_depth = vol_usd * (0.5 - obi * 0.3)
        bid_depth = max(bid_depth, 1.0)
        ask_depth = max(ask_depth, 1.0)
        depth_ratio = bid_depth / ask_depth

        # ── Iceberg детекция ────────────────────────
        # Крупный объём = возможный iceberg
        large_bid = vol_ratio > self.iceberg_thresh and obi > 0.3
        large_ask = vol_ratio > self.iceberg_thresh and obi < -0.3

        iceberg_signal = "NONE"
        if large_bid:
            iceberg_signal = "BUY"    # крупный игрок покупает
        elif large_ask:
            iceberg_signal = "SELL"   # крупный игрок продаёт

        # ── VWAP ────────────────────────────────────
        vwap = float(row.get("vwap", price))
        if np.isnan(vwap):
            vwap = price
        price_vs_vwap = (price - vwap) / (vwap + 1e-8)

        # ── Tick direction ──────────────────────────
        tick_dir = float(row.get("tick_dir_5", 0.0))
        tick_dir_norm = np.tanh(tick_dir * 100)  # нормализуем в [-1, +1]

        return OrderBookSnapshot(
            mid_price          = price,
            bid_price          = bid,
            ask_price          = ask,
            spread             = spread,
            spread_usd         = spread_usd,
            obi                = float(obi),
            bid_depth          = bid_depth,
            ask_depth          = ask_depth,
            depth_ratio        = float(depth_ratio),
            large_bid_detected = large_bid,
            large_ask_detected = large_ask,
            iceberg_signal     = iceberg_signal,
            vwap               = vwap,
            price_vs_vwap      = float(price_vs_vwap),
            tick_direction     = float(tick_dir_norm),
        )

    def _neutral_snapshot(self, price: float) -> OrderBookSnapshot:
        """Нейтральный снимок когда нет истории."""
        return OrderBookSnapshot(
            mid_price=price, bid_price=price*0.9999, ask_price=price*1.0001,
            spread=0.0001, spread_usd=price*0.0001,
            obi=0.0, bid_depth=1e6, ask_depth=1e6, depth_ratio=1.0,
            large_bid_detected=False, large_ask_detected=False,
            iceberg_signal="NONE", vwap=price, price_vs_vwap=0.0, tick_direction=0.0,
        )

    def as_features(self, idx: int) -> np.ndarray:
        """
        Возвращает вектор признаков стакана для observation space.
        Размер: 6 float значений.
        """
        snap = self.snapshot(idx)
        return np.array([
            snap.obi,                                    # [-1, +1]
            snap.spread * 100,                           # спред в %
            np.log(snap.depth_ratio + 1e-8),             # log глубины
            snap.price_vs_vwap,                          # позиция vs VWAP
            snap.tick_direction,                         # тик-направление
            1.0 if snap.iceberg_signal == "BUY" else
            (-1.0 if snap.iceberg_signal == "SELL" else 0.0),  # iceberg
        ], dtype=np.float32)
