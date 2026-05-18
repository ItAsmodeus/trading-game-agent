"""
DataLoader — загрузка и препроцессинг OHLCV данных с Binance.
Кэширует в data/<symbol>_<timeframe>.parquet чтобы не качать повторно.
"""
from __future__ import annotations

import datetime
import os
import time
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from config import CFG


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    return symbol.replace("/", "") + f"_{timeframe}.parquet"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = CFG.TIMEFRAME,
    since: str = CFG.TRAIN_START,
    until: Optional[str] = None,
    data_dir: str = CFG.DATA_DIR,
) -> pd.DataFrame:
    """Загружает OHLCV с Binance. Кэш валиден если файл свежее 24ч И покрывает нужный диапазон."""
    if until is None:
        until = datetime.date.today().strftime("%Y-%m-%d")

    Path(data_dir).mkdir(exist_ok=True)
    cache_path = Path(data_dir) / _symbol_to_filename(symbol, timeframe)

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            cached = pd.read_parquet(cache_path)
            until_ts = pd.Timestamp(until, tz="UTC")
            # Cache valid only if it covers the requested range
            if cached.index[-1] >= until_ts - pd.Timedelta("48h"):
                return cached

    exchange = ccxt.binance({"enableRateLimit": True})
    since_ts = exchange.parse8601(f"{since}T00:00:00Z")
    until_ts = exchange.parse8601(f"{until}T00:00:00Z")

    all_candles = []
    current_since = since_ts
    print(f"Fetching {symbol} {timeframe} from {since} to {until}...")

    while current_since < until_ts:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        current_since = candles[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < pd.Timestamp(until, tz="UTC")]
    df = df[~df.index.duplicated(keep="first")]

    df.to_parquet(cache_path)
    print(f"  Saved {len(df)} candles to {cache_path}")
    return df


def fetch_futures_features(
    symbol: str,
    since: str,
    until: str,
    data_dir: str = CFG.DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Funding rate (8h) + Open Interest (1h) from Binance USDT-M Futures.
    Both resampled to 1h with forward-fill. Returns None on failure.
    """
    futures_symbol = symbol.replace("/USDT", "/USDT:USDT")
    safe_name = symbol.replace("/", "")
    cache_path = Path(data_dir) / f"{safe_name}_futures.parquet"

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            cached = pd.read_parquet(cache_path)
            until_ts = pd.Timestamp(until, tz="UTC")
            if cached.index[-1] >= until_ts - pd.Timedelta("48h"):
                return cached

    exchange = ccxt.binanceusdm({"enableRateLimit": True})
    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    until_ms = int(pd.Timestamp(until, tz="UTC").timestamp() * 1000)
    now_ms   = int(time.time() * 1000)
    frames   = []

    # ── Funding rate (every 8h) — full history available ──────────────────────
    try:
        all_fr: list = []
        current = since_ms
        while current < until_ms:
            batch = exchange.fetch_funding_rate_history(
                futures_symbol, since=current, limit=1000,
            )
            if not batch:
                break
            all_fr.extend(batch)
            next_ts = batch[-1]["timestamp"] + 1
            if next_ts <= current:
                break
            current = next_ts
            time.sleep(exchange.rateLimit / 1000)

        if all_fr:
            fr_df = pd.DataFrame([{
                "ts": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                "funding_rate": float(r["fundingRate"]),
            } for r in all_fr]).set_index("ts").sort_index()
            fr_df = fr_df.resample("1h").ffill()
            frames.append(fr_df)
            print(f"  FR: {len(all_fr)} records")
    except Exception as e:
        print(f"  [warn] Funding rate unavailable: {e}")

    # ── Open Interest (1h) — only last 30 days available ─────────────────────
    oi_cutoff = now_ms - 29 * 24 * 3600 * 1000
    if until_ms >= oi_cutoff:
        try:
            all_oi: list = []
            oi_since = max(since_ms, oi_cutoff)
            current  = oi_since
            while current < until_ms:
                batch = exchange.fetch_open_interest_history(
                    futures_symbol, "1h", since=current, limit=500,
                )
                if not batch:
                    break
                all_oi.extend(batch)
                next_ts = batch[-1]["timestamp"] + 1
                if next_ts <= current:
                    break
                current = next_ts
                time.sleep(exchange.rateLimit / 1000)

            if all_oi:
                oi_df = pd.DataFrame([{
                    "ts": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                    "open_interest": float(r["openInterestAmount"]),
                } for r in all_oi]).set_index("ts").sort_index()
                oi_df = oi_df.resample("1h").last().ffill()
                frames.append(oi_df)
                print(f"  OI: {len(all_oi)} records")
        except Exception as e:
            print(f"  [warn] Open interest unavailable: {e}")

    if not frames:
        return None

    result = pd.concat(frames, axis=1)
    result = result[~result.index.duplicated(keep="first")]
    result.to_parquet(cache_path)
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы. Возвращает новый DataFrame."""
    d = df.copy()
    c = d["close"]

    # Moving averages
    d["ma5"]  = c.rolling(5).mean()
    d["ma20"] = c.rolling(20).mean()
    d["ma50"] = c.rolling(50).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi"] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (width)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d["bb_upper"] = sma20 + 2 * std20
    d["bb_lower"] = sma20 - 2 * std20
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / sma20

    # Volume features
    d["vol_ma20"]   = d["volume"].rolling(20).mean()
    d["vol_ratio"]  = d["volume"] / d["vol_ma20"].replace(0, np.nan)

    # Returns
    d["ret_1"]  = c.pct_change(1)
    d["ret_5"]  = c.pct_change(5)
    d["ret_24"] = c.pct_change(24)

    # High-Low range
    d["hl_range"] = (d["high"] - d["low"]) / c

    # ── Order Book Imbalance (simulated from OHLCV) ───────────────────────────
    # OBI: bullish bar (close > open) = buying pressure, bearish = selling pressure
    bar_direction = np.sign(d["close"] - d["open"])
    d["obi"] = (bar_direction * d["volume"]).rolling(10).sum() / \
               d["volume"].rolling(10).sum().replace(0, np.nan)

    # VWAP deviation (price vs 20-bar VWAP)
    typical = (d["high"] + d["low"] + d["close"]) / 3
    d["vwap_dev"] = (c - (typical * d["volume"]).rolling(20).sum() /
                     d["volume"].rolling(20).sum().replace(0, np.nan)) / c

    d = d.dropna()
    return d


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score нормализация по обучающей выборке (только числовые колонки)."""
    cols = df.select_dtypes(include=[np.number]).columns
    mean = df[cols].mean()
    std  = df[cols].std().replace(0, 1)
    df[cols] = (df[cols] - mean) / std
    return df


def _normalize_split(data: pd.DataFrame, train_ref: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize `data` using stats from `train_ref`. Excludes close_raw."""
    cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != "close_raw"]
    mean = train_ref[cols].mean()
    std  = train_ref[cols].std().replace(0, 1)
    data[cols] = (data[cols] - mean) / std
    return data


def load_dataset(
    symbol: str,
    timeframe: str = CFG.TIMEFRAME,
    split: str = "train",
) -> pd.DataFrame:
    """Возвращает нормализованный датасет для нужного сплита."""
    raw = fetch_ohlcv(symbol, timeframe)
    feat = add_features(raw)

    splits = {
        "train": (CFG.TRAIN_START, CFG.TRAIN_END),
        "val":   (CFG.VAL_START,   CFG.VAL_END),
        "test":  (CFG.TEST_START,  CFG.TEST_END),
    }
    start, end = splits[split]
    mask = (feat.index >= start) & (feat.index < end)
    data = feat[mask].copy()

    # Keep raw close for market simulation BEFORE normalization
    data["close_raw"] = data["close"].copy()

    # Normalize feature columns using train-set statistics
    train_mask = (feat.index >= CFG.TRAIN_START) & (feat.index < CFG.TRAIN_END)
    train_data = feat[train_mask]
    cols = [c for c in data.select_dtypes(include=[np.number]).columns if c != "close_raw"]
    mean = train_data[cols].mean()
    std  = train_data[cols].std().replace(0, 1)
    data[cols] = (data[cols] - mean) / std

    return data


def load_window(
    symbol: str,
    train_start: str,
    train_end: str,
    val_end: str,
    timeframe: str = CFG.TIMEFRAME,
    use_futures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling window loader for walk-forward training.

    Train: [train_start, train_end)
    Val:   [train_end,   val_end)

    Normalization uses ONLY train-portion statistics (no lookahead bias).
    Returns (train_df, val_df) both with close_raw column.
    """
    # Fetch OHLCV covering the whole window
    raw = fetch_ohlcv(symbol, timeframe, since=CFG.TRAIN_START, until=val_end)

    # Optionally merge futures features (funding rate + OI)
    if use_futures:
        fut = fetch_futures_features(symbol, since=train_start, until=val_end)
        if fut is not None:
            raw = raw.join(fut, how="left")
            # Forward-fill futures columns so dropna() in add_features doesn't lose rows
            for col in fut.columns:
                if col in raw.columns:
                    raw[col] = raw[col].ffill()

    feat = add_features(raw)

    train_mask = (feat.index >= train_start) & (feat.index < train_end)
    val_mask   = (feat.index >= train_end)   & (feat.index < val_end)

    train_data = feat[train_mask].copy()
    val_data   = feat[val_mask].copy()

    # Preserve raw close before normalization
    train_data["close_raw"] = train_data["close"].copy()
    val_data["close_raw"]   = val_data["close"].copy()

    # Compute normalization stats from train BEFORE modifying it
    num_cols = [c for c in train_data.select_dtypes(include=[np.number]).columns if c != "close_raw"]
    mean = train_data[num_cols].mean()
    std  = train_data[num_cols].std().replace(0, 1)

    train_data[num_cols] = (train_data[num_cols] - mean) / std
    val_num_cols = [c for c in val_data.select_dtypes(include=[np.number]).columns if c != "close_raw"]
    shared = [c for c in val_num_cols if c in mean.index]
    val_data[shared] = (val_data[shared] - mean[shared]) / std[shared]

    return train_data, val_data


def load_all_symbols(split: str = "train") -> dict[str, pd.DataFrame]:
    """Загружает все инструменты из CFG.SYMBOLS."""
    datasets = {}
    for symbol in CFG.SYMBOLS:
        try:
            datasets[symbol] = load_dataset(symbol, split=split)
            print(f"Loaded {symbol}: {len(datasets[symbol])} rows")
        except Exception as e:
            print(f"Failed to load {symbol}: {e}")
    return datasets
