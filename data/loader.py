"""
Загрузка исторических OHLCV данных.
Два режима: из файла CSV или через ccxt с биржи.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_from_csv(path: str) -> pd.DataFrame:
    """Загружает OHLCV из CSV. Ожидает колонки: timestamp, open, high, low, close, volume."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    _validate(df)
    return df


def load_from_exchange(
    symbol:    str = "BTC/USDT",
    timeframe: str = "1d",
    limit:     int = 1000,
    exchange:  str = "binance",
) -> pd.DataFrame:
    """
    Загружает данные с биржи через ccxt.
    pip install ccxt
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("Установи ccxt: pip install ccxt")

    ex = getattr(ccxt, exchange)()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    _validate(df)
    return df


def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Хронологический split — никогда не перемешивай временные ряды!
    test = последние (1 - train_ratio - val_ratio) данных.
    """
    n     = len(df)
    n_tr  = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = df.iloc[:n_tr].reset_index(drop=True)
    val   = df.iloc[n_tr:n_tr + n_val].reset_index(drop=True)
    test  = df.iloc[n_tr + n_val:].reset_index(drop=True)

    print(f"Train: {len(train)} баров | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Базовые производные признаки (не утечка данных — только прошлое)."""
    df = df.copy()
    df["returns"]   = df["close"].pct_change()
    df["log_ret"]   = np.log(df["close"] / df["close"].shift(1))
    df["vol_20"]    = df["returns"].rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    return df.dropna().reset_index(drop=True)


def _validate(df: pd.DataFrame):
    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки: {missing}")
    assert (df["high"] >= df["close"]).all(), "high < close — битые данные"
    assert (df["low"]  <= df["close"]).all(), "low > close — битые данные"
    assert not df["close"].isna().any(),      "NaN в close"
