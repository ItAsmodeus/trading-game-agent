"""
Binance Funding Rate — загрузка и кэширование данных.

Funding rate выплачивается каждые 8 часов на perpetual futures.
Высокий FR (> 0.05%) = рынок перегрет лонгами = сигнал к осторожности.
Отрицательный FR = рынок перегрет шортами = возможность для лонгов.

Использование:
    loader = FundingRateLoader("BTCUSDT")
    df = loader.load(start="2023-01-01")
    # df: timestamp, funding_rate, annual_yield
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


CACHE_DIR = Path("./data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Пороги для сигналов (из strategy-funding-arb скилла)
FR_THRESHOLDS = {
    "very_high":  0.001,    # > 0.1% за 8ч = 109% годовых — выход из лонга
    "high":       0.0005,   # > 0.05% = 54% годовых — снижаем лонг
    "neutral_hi": 0.0001,   # > 0.01% — норма
    "neutral_lo": -0.0001,  # нейтральная зона
    "low":       -0.0005,   # < -0.05% — сигнал для лонгов (шорты платят)
}


class FundingRateLoader:
    """
    Загружает исторические funding rates с Binance Futures API.
    Кэширует результаты локально чтобы не спамить API.
    """

    BASE_URL = "https://fapi.binance.com"

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol     = symbol.replace("/", "").replace("-", "").upper()
        self.cache_path = CACHE_DIR / f"{self.symbol}_funding.parquet"

    def load(
        self,
        start: str = "2020-01-01",
        end:   Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Загружает funding rate историю. Возвращает DataFrame:
          timestamp, funding_rate, annual_yield, signal
        """
        if use_cache and self.cache_path.exists():
            cached = pd.read_parquet(self.cache_path)
            start_dt = pd.Timestamp(start, tz="UTC")
            if cached["timestamp"].max() >= start_dt:
                df = cached[cached["timestamp"] >= start_dt].reset_index(drop=True)
                print(f"[Funding] Кэш: {len(df)} записей для {self.symbol}")
                return self._enrich(df)

        df = self._fetch_from_api(start, end)
        if len(df) > 0:
            df.to_parquet(self.cache_path, index=False)
            print(f"[Funding] Сохранено в кэш: {self.cache_path}")

        return self._enrich(df)

    def _fetch_from_api(self, start: str, end: Optional[str]) -> pd.DataFrame:
        """Скачивает с Binance Futures REST API (лимит 1000 записей за запрос)."""
        try:
            import requests
        except ImportError:
            raise ImportError("Установи requests: pip install requests")

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms   = int(pd.Timestamp(end).timestamp() * 1000) if end else int(time.time() * 1000)

        all_records = []
        url = f"{self.BASE_URL}/fapi/v1/fundingRate"

        print(f"[Funding] Загружаю {self.symbol} с {start}...")
        while start_ms < end_ms:
            params = {
                "symbol": self.symbol,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"[Funding] Ошибка API: {e}. Возвращаю что есть.")
                break

            if not data:
                break

            all_records.extend(data)
            start_ms = data[-1]["fundingTime"] + 1
            time.sleep(0.1)  # соблюдаем rate limits

            if len(data) < 1000:
                break

        if not all_records:
            print(f"[Funding] ⚠️  Нет данных с API, возвращаю пустой DataFrame")
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(all_records)
        df["timestamp"]    = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)

        print(f"[Funding] Загружено {len(df)} записей")
        return df

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет производные признаки."""
        if df.empty:
            return df

        df = df.copy()

        # Годовая доходность (3 выплаты в день × 365)
        df["annual_yield"] = df["funding_rate"] * 3 * 365

        # Сигнал для агента: нормализованный FR [-1, 1]
        df["fr_normalized"] = np.tanh(df["funding_rate"] * 1000)

        # Скользящее среднее FR (тренд)
        df["fr_ma_24h"] = df["funding_rate"].rolling(3).mean()  # 3 × 8ч = 24ч

        # Категория сигнала
        df["signal"] = df["funding_rate"].apply(_classify_signal)

        return df

    def align_with_ohlcv(
        self,
        ohlcv: pd.DataFrame,
        funding: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Совмещает OHLCV и funding rate по времени.
        OHLCV дневной → берём последний известный FR на момент каждого бара.
        """
        if funding.empty:
            ohlcv = ohlcv.copy()
            ohlcv["funding_rate"]  = 0.0
            ohlcv["fr_normalized"] = 0.0
            return ohlcv

        # Убедимся что временные метки совместимы
        ohlcv_ts    = pd.to_datetime(ohlcv["timestamp"])
        funding_ts  = funding["timestamp"]

        if ohlcv_ts.dt.tz is None and funding_ts.dt.tz is not None:
            ohlcv_ts = ohlcv_ts.dt.tz_localize("UTC")
        elif ohlcv_ts.dt.tz is not None and funding_ts.dt.tz is None:
            funding_ts = funding_ts.dt.tz_localize("UTC")

        # Merge: последний известный FR до каждой свечи
        merged = pd.merge_asof(
            ohlcv.assign(timestamp=ohlcv_ts).sort_values("timestamp"),
            funding[["timestamp", "funding_rate", "fr_normalized", "fr_ma_24h"]].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

        # Заполняем NaN если нет данных по FR
        for col in ["funding_rate", "fr_normalized", "fr_ma_24h"]:
            merged[col] = merged[col].fillna(0.0)

        return merged.reset_index(drop=True)


def _classify_signal(fr: float) -> str:
    if fr > FR_THRESHOLDS["very_high"]:
        return "VERY_HIGH"    # рынок сильно перегрет — осторожно с лонгом
    elif fr > FR_THRESHOLDS["high"]:
        return "HIGH"          # повышенный — снижаем риск
    elif fr > FR_THRESHOLDS["neutral_hi"]:
        return "NEUTRAL_HIGH"
    elif fr > FR_THRESHOLDS["neutral_lo"]:
        return "NEUTRAL"
    elif fr > FR_THRESHOLDS["low"]:
        return "NEUTRAL_LOW"
    else:
        return "LOW"           # шорты платят — возможность для лонгов


def get_current_funding(symbol: str = "BTCUSDT") -> Optional[float]:
    """Получить текущий funding rate (для live торговли)."""
    try:
        import requests
        url    = "https://fapi.binance.com/fapi/v1/premiumIndex"
        resp   = requests.get(url, params={"symbol": symbol}, timeout=5)
        data   = resp.json()
        return float(data.get("lastFundingRate", 0))
    except Exception:
        return None


# ──────────────────────────────────────────────
# CLI: python -m data.binance_funding
# ──────────────────────────────────────────────
if __name__ == "__main__":
    loader = FundingRateLoader("BTCUSDT")
    df = loader.load(start="2023-01-01")

    print(f"\nФunding Rate статистика:")
    print(f"  Записей:          {len(df)}")
    print(f"  Период:           {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Средний FR:       {df['funding_rate'].mean():.4%}")
    print(f"  Макс FR:          {df['funding_rate'].max():.4%}")
    print(f"  Мин FR:           {df['funding_rate'].min():.4%}")
    print(f"  Средн. годовых:   {df['annual_yield'].mean():.1%}")
    print(f"\nРаспределение сигналов:")
    print(df["signal"].value_counts().to_string())

    current = get_current_funding("BTCUSDT")
    if current is not None:
        print(f"\nТекущий FR: {current:.4%} ({current * 3 * 365:.1%}/год)")
        print(f"Сигнал: {_classify_signal(current)}")
