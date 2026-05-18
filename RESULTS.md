# Results Comparison: Main Bot vs Agent 2

> Обновляется после каждого значимого прогона. Цель: честное сравнение двух подходов.

---

## Walk-Forward Sharpe (ключевая метрика)

| Version | Branch | WF Sharpe | Mean Return | Win Rate | Notes |
|---------|--------|-----------|-------------|----------|-------|
| Main v1 (PPO 200k) | master | 1.21 | +8.5% | — | OOS, медвежий рынок, 43 сделки/150 дней |
| Our v1 (200k) | agent2/dev | 0.270 | +2.49% | 59% | 17 окон, 1h candles |
| Our v2 (500k) | agent2/dev | ~0.27 | +2.7% | ~59% | Переобучение на тренировочном окне |
| Our v3 (TL+DD+Time) | agent2/dev | — | — | — | В планах: transfer learning + dd penalty + time features |

---

## Архитектурные различия

| Параметр | Main Bot (master) | Our Bot (agent2/dev) |
|----------|-------------------|----------------------|
| Таймфрейм | 1d | 1h |
| Observation | 22 flat features | 60×20 + 9 = 1209 |
| Шорт | ✅ | ❌ (в планах) |
| Transfer learning | ✅ | ✅ (v3+) |
| Drawdown penalty | ✅ quadratic | ✅ (v3+) |
| Idle penalty | ❌ | ✅ (v3+) |
| Funding rate | ✅ | ❌ (API broken) |
| Time features | ✅ sin/cos | ✅ (v3+) |
| Actions | 6 (incl. SHORT) | 7 (BUY/SELL 25/50/100%) |

---

## Гипотезы в работе

| # | Гипотеза | Статус | Ветка | Результат |
|---|----------|--------|-------|-----------|
| 1 | Transfer learning между окнами | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 2 | Time features (sin/cos) | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 3 | Quadratic DD penalty | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 4 | Volatility filter (торговать только волатильные сессии) | 📋 В очереди | — | — |
| 5 | SHORT capability | 📋 В очереди | — | — |
| 6 | Funding rate fix (API bug) | 📋 В очереди | — | — |
| 7 | Session-based episodes (150 candles) | 📋 В очереди | — | — |

---

## История сканов основной ветки

| Дата | Новые фичи в master | Взяли себе? | Комментарий |
|------|---------------------|-------------|-------------|
| 2026-05-18 | 23 изменений | — | Авто-скан |
| 2026-05-18 | Rolling trainer, risk_manager, binance_funding, trade_diary | Частично | TL + DD penalty взяли |
