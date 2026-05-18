"""
News Fetcher — новости и сентимент для торговых решений.

Источники (в порядке приоритета):
  1. Fear & Greed Index (api.alternative.me) — бесплатно, без ключа, всегда работает
  2. CryptoPanic API — агрегатор крипто-новостей (нужен бесплатный ключ)
  3. Fallback: нейтральный сентимент если API недоступен

Важно: в бэктесте используем исторические данные Fear&Greed.
В live режиме — текущие данные.
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

CACHE_DIR = Path("./data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ключевые слова для быстрого анализа заголовков
BULLISH_WORDS = [
    "bullish", "surge", "rally", "soar", "breakout", "adoption", "institutional",
    "etf approved", "халвинг", "рост", "бычий", "одобрен", "рекорд", "pump",
    "accumulation", "upgrade", "partnership", "launch", "mainstream", "ath"
]
BEARISH_WORDS = [
    "bearish", "crash", "dump", "ban", "hack", "regulation", "crackdown",
    "падение", "запрет", "взлом", "обвал", "медвежий", "sell-off", "liquidation",
    "fine", "lawsuit", "exploit", "vulnerability", "rug", "collapse"
]


@dataclass
class NewsSignal:
    """Результат анализа новостей."""
    fear_greed_score: int       # 0-100 (0=Extreme Fear, 100=Extreme Greed)
    fear_greed_label: str       # "Extreme Fear" / "Fear" / "Neutral" / "Greed" / "Extreme Greed"
    news_sentiment:   float     # -1.0 (очень медвежий) до +1.0 (очень бычий)
    top_headlines:    list[str] # топ-3 заголовка
    summary_ru:       str       # краткое описание по-русски для дневника
    is_significant:   bool      # есть ли значимые новости

    def as_reason(self) -> Optional[str]:
        """Возвращает основание для входа если новость значима."""
        if not self.is_significant:
            return None
        if self.fear_greed_score <= 25:
            return f"Индекс страха = {self.fear_greed_score} (Extreme Fear) — исторически лучшее время для покупки"
        if self.fear_greed_score >= 75:
            return f"Индекс жадности = {self.fear_greed_score} (Extreme Greed) — рынок перегрет, риск разворота"
        if self.news_sentiment > 0.3:
            return f"Новостной фон бычий (сентимент {self.news_sentiment:+.2f}): {self.top_headlines[0] if self.top_headlines else ''}"
        if self.news_sentiment < -0.3:
            return f"Новостной фон медвежий (сентимент {self.news_sentiment:+.2f}): {self.top_headlines[0] if self.top_headlines else ''}"
        return None


class NewsFetcher:
    """
    Получает актуальные рыночные данные о настроении.
    Работает в двух режимах:
      - live: реальные данные с API
      - backtest: исторические Fear&Greed данные
    """

    FEAR_GREED_URL   = "https://api.alternative.me/fng/"
    CRYPTOPANIC_URL  = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, cryptopanic_token: Optional[str] = None):
        self.cryptopanic_token = cryptopanic_token
        self._fg_cache: Optional[dict] = None
        self._fg_cache_time: float = 0.0
        self._historical_fg: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────
    # Основной метод
    # ──────────────────────────────────────────

    def get_signal(
        self,
        date: Optional[str] = None,    # для бэктеста: "2024-01-15"
        symbol: str = "BTC",
    ) -> NewsSignal:
        """
        Возвращает NewsSignal.
        date=None → live режим (текущие данные).
        date="YYYY-MM-DD" → бэктест режим (исторические данные).
        """
        if date is not None:
            return self._backtest_signal(date)
        return self._live_signal(symbol)

    # ──────────────────────────────────────────
    # Live режим
    # ──────────────────────────────────────────

    def _live_signal(self, symbol: str) -> NewsSignal:
        fg    = self._get_fear_greed_live()
        news  = self._get_news_live(symbol) if self.cryptopanic_token else []
        return self._build_signal(fg, news)

    def _get_fear_greed_live(self) -> dict:
        """Fear & Greed Index. Кэшируем на 1 час."""
        now = time.time()
        if self._fg_cache and (now - self._fg_cache_time) < 3600:
            return self._fg_cache

        try:
            import requests
            resp = requests.get(self.FEAR_GREED_URL, timeout=5)
            data = resp.json()["data"][0]
            self._fg_cache      = {"score": int(data["value"]), "label": data["value_classification"]}
            self._fg_cache_time = now
            return self._fg_cache
        except Exception:
            return {"score": 50, "label": "Neutral"}

    def _get_news_live(self, symbol: str) -> list[str]:
        """CryptoPanic новости. Требует токен."""
        if not self.cryptopanic_token:
            return []
        try:
            import requests
            params = {
                "auth_token": self.cryptopanic_token,
                "currencies": symbol,
                "filter": "hot",
                "public": "true",
            }
            resp = requests.get(self.CRYPTOPANIC_URL, params=params, timeout=5)
            results = resp.json().get("results", [])
            return [r["title"] for r in results[:5]]
        except Exception:
            return []

    # ──────────────────────────────────────────
    # Бэктест режим
    # ──────────────────────────────────────────

    def _backtest_signal(self, date: str) -> NewsSignal:
        """Исторические данные Fear & Greed для бэктеста."""
        if self._historical_fg is None:
            self._historical_fg = self._load_or_fetch_historical_fg()

        try:
            dt = pd.Timestamp(date)
            row = self._historical_fg.loc[self._historical_fg["date"] <= dt].iloc[-1]
            score = int(row["score"])
            label = _fg_label(score)
        except (IndexError, KeyError):
            score, label = 50, "Neutral"

        return self._build_signal({"score": score, "label": label}, [])

    def _load_or_fetch_historical_fg(self) -> pd.DataFrame:
        cache_path = CACHE_DIR / "fear_greed_historical.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            # Обновляем если данным больше суток
            if (pd.Timestamp.now() - df["date"].max()).days <= 1:
                return df

        try:
            import requests
            resp = requests.get(f"{self.FEAR_GREED_URL}?limit=2000&format=json", timeout=10)
            data = resp.json()["data"]
            df = pd.DataFrame(data)
            df["date"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s")
            df["score"] = df["value"].astype(int)
            df = df[["date", "score"]].sort_values("date").reset_index(drop=True)
            df.to_parquet(cache_path, index=False)
            print(f"[News] Загружено {len(df)} дней Fear&Greed данных")
            return df
        except Exception as e:
            print(f"[News] Не удалось загрузить исторические данные F&G: {e}")
            # Возвращаем пустой датафрейм — будем давать нейтральный сигнал
            return pd.DataFrame({"date": [pd.Timestamp("2020-01-01")], "score": [50]})

    # ──────────────────────────────────────────
    # Анализ
    # ──────────────────────────────────────────

    def _build_signal(self, fg: dict, headlines: list[str]) -> NewsSignal:
        score = fg.get("score", 50)
        label = fg.get("label", "Neutral")

        # Сентимент из заголовков
        sentiment = _analyze_headlines_sentiment(headlines)

        # Является ли сигнал значимым
        is_significant = (
            score <= 25 or score >= 75 or          # экстремальный F&G
            abs(sentiment) > 0.3 or                 # сильные новости
            (score <= 35 and sentiment < -0.1) or   # страх + плохие новости
            (score >= 65 and sentiment > 0.1)        # жадность + хорошие новости
        )

        summary_ru = _build_russian_summary(score, label, sentiment, headlines)

        return NewsSignal(
            fear_greed_score=score,
            fear_greed_label=label,
            news_sentiment=sentiment,
            top_headlines=headlines[:3],
            summary_ru=summary_ru,
            is_significant=is_significant,
        )


# ──────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────

def _analyze_headlines_sentiment(headlines: list[str]) -> float:
    """Быстрый keyword-based анализ сентимента заголовков. [-1, +1]"""
    if not headlines:
        return 0.0

    bull_count = 0
    bear_count = 0
    for h in headlines:
        h_lower = h.lower()
        bull_count += sum(1 for w in BULLISH_WORDS if w in h_lower)
        bear_count += sum(1 for w in BEARISH_WORDS if w in h_lower)

    total = bull_count + bear_count
    if total == 0:
        return 0.0
    return (bull_count - bear_count) / total


def _fg_label(score: int) -> str:
    if score <= 24:  return "Extreme Fear"
    if score <= 44:  return "Fear"
    if score <= 55:  return "Neutral"
    if score <= 74:  return "Greed"
    return "Extreme Greed"


def _build_russian_summary(score: int, label: str, sentiment: float, headlines: list[str]) -> str:
    label_ru = {
        "Extreme Fear": "Экстремальный страх",
        "Fear":         "Страх",
        "Neutral":      "Нейтральный",
        "Greed":        "Жадность",
        "Extreme Greed":"Экстремальная жадность",
    }.get(label, label)

    parts = [f"Fear & Greed: {score}/100 ({label_ru})"]

    if headlines:
        if sentiment > 0.2:
            parts.append(f"Новостной фон позитивный")
        elif sentiment < -0.2:
            parts.append(f"Новостной фон негативный")
        else:
            parts.append(f"Новости нейтральные")
        if headlines:
            parts.append(f"Топ: \"{headlines[0]}\"")
    else:
        parts.append("Данные о новостях недоступны (бэктест режим)")

    return " | ".join(parts)


# ──────────────────────────────────────────
# CLI: python -m data.news_fetcher
# ──────────────────────────────────────────
if __name__ == "__main__":
    fetcher = NewsFetcher()

    print("=== LIVE сигнал ===")
    signal = fetcher.get_signal()
    print(f"Fear & Greed: {signal.fear_greed_score} ({signal.fear_greed_label})")
    print(f"Сентимент:    {signal.news_sentiment:+.2f}")
    print(f"Сводка:       {signal.summary_ru}")
    reason = signal.as_reason()
    print(f"Основание:    {reason or 'нет значимого сигнала'}")
