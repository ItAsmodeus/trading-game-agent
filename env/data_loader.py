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
import requests

from config import CFG


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    return symbol.replace("/", "") + f"_{timeframe}.parquet"


def fetch_fear_greed(
    since: str,
    until: str,
    data_dir: str = CFG.DATA_DIR,
) -> Optional[pd.Series]:
    """
    Fear & Greed Index from Alternative.me (free, no API key).
    Daily values (0-100) resampled to 1h via forward-fill.
    Cached to data/fear_greed.parquet.
    """
    cache_path = Path(data_dir) / "fear_greed.parquet"

    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 24:
            cached = pd.read_parquet(cache_path)["fear_greed"]
            since_ts = pd.Timestamp(since, tz="UTC")
            until_ts = pd.Timestamp(until, tz="UTC")
            if cached.index[0] <= since_ts + pd.Timedelta("7d") and \
               cached.index[-1] >= until_ts - pd.Timedelta("2d"):
                return cached

    try:
        days = (pd.Timestamp(until) - pd.Timestamp(since)).days + 30
        days = min(days, 2000)  # API limit
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        resp = requests.get(url, timeout=15)
        data = resp.json()["data"]

        records = []
        for item in data:
            ts = pd.Timestamp(int(item["timestamp"]), unit="s", tz="UTC").normalize()
            val = float(item["value"]) / 100.0  # normalize to [0, 1]
            records.append({"ts": ts, "fear_greed": val})

        df = pd.DataFrame(records).set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        # Resample daily → hourly with forward-fill
        df = df.resample("1h").ffill()

        Path(data_dir).mkdir(exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  F&G: {len(df)} hourly records fetched")
        return df["fear_greed"]

    except Exception as e:
        print(f"  [warn] Fear & Greed unavailable: {e}")
        return None


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
            since_ts = pd.Timestamp(since, tz="UTC")
            until_ts = pd.Timestamp(until, tz="UTC")
            # Cache valid only if it covers BOTH start and end of requested window
            if cached.index[0] <= since_ts + pd.Timedelta("48h") and \
               cached.index[-1] >= until_ts - pd.Timedelta("48h"):
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

    # ── H-006: Log-Normal Z-Score Features (Quantum Leap inspired) ───────────
    # Log returns for drift and vol estimation
    log_ret = np.log(c / c.shift(1))

    # 60-bar log drift (natural drift per 1h bar ≈ 2.5 days window)
    d["mu_60"] = log_ret.rolling(60).mean().clip(-0.05, 0.05)

    # 20-bar annualized volatility (1h data: annualize by sqrt(24*365) ≈ 93.6)
    d["sigma_20"] = log_ret.rolling(20).std() * np.sqrt(4 * 24 * 365)  # 4 bars/h at 15m
    d["sigma_20"] = d["sigma_20"].clip(0, 3)

    # Z-score: how far is price from its GBM "expected" value
    expected = c.shift(1) * np.exp(d["mu_60"])
    sigma_price = d["sigma_20"] * c / np.sqrt(24 * 365)  # per-bar vol in price units
    d["z_score_ln"] = ((c - expected) / sigma_price.replace(0, np.nan)).clip(-3, 3)

    # Market regime: price deviation from MA200 (200h ≈ 8 days)
    ma200 = c.rolling(200).mean()
    d["regime"] = ((c - ma200) / ma200.replace(0, np.nan)).clip(-0.5, 0.5)

    # MA cross: MA20 vs MA50 (trend direction signal, same as main bot feature 26)
    d["ma_cross"] = ((d["ma20"] - d["ma50"]) / d["ma50"].replace(0, np.nan)).clip(-0.5, 0.5)

    # ── H-013: Liquidation Cascade Proxy ─────────────────────────────────────
    # Mechanism: forced liquidations = abnormal volume + abnormal price move together.
    # After a liquidation cascade exhausts itself, price tends to rebound —
    # this is a causal signal, not a lagging indicator.
    vol_z = (d["volume"] - d["volume"].rolling(24).mean()) / \
            d["volume"].rolling(24).std().replace(0, np.nan)
    ret_abs = d["ret_1"].abs()
    ret_z = (ret_abs - ret_abs.rolling(24).mean()) / \
            ret_abs.rolling(24).std().replace(0, np.nan)

    # Intensity: both volume AND return must be abnormal (multiplicative gate)
    liq_intensity = (vol_z.clip(0) * ret_z.clip(0)).clip(0, 10)

    # Signed: positive = long liquidations (price fell → buy signal for rebound)
    #         negative = short liquidations (price rose → sell signal for reversal)
    d["liq_signal"] = (liq_intensity * -np.sign(d["ret_1"])).clip(-5, 5)

    # Memory: was there a liquidation event in the last 3 bars?
    d["liq_recent"] = liq_intensity.rolling(3).max().clip(0, 10)

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
