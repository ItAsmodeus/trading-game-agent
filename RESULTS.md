# Results Comparison: Main Bot vs Agent 2

> Обновляется после каждого значимого прогона. Цель: честное сравнение двух подходов.

---

## Walk-Forward Sharpe (ключевая метрика)

| Version | Branch | WF Sharpe | Mean Return | Win Rate | Notes |
|---------|--------|-----------|-------------|----------|-------|
| Main v1 (PPO 200k) | master | 1.21 | +8.5% | — | OOS, медвежий рынок, 43 сделки/150 дней |
| Our v1 (200k) | agent2/dev | 0.270 | +2.49% | 59% | 17 окон, 1h candles |
| Our v2 (500k) | agent2/dev | ~0.27 | +2.7% | ~59% | Переобучение на тренировочном окне |
| Our v3 (TL+DD+Time+OBI+FR) | agent2/dev | -0.788 | — | — | ❌ DD penalty слишком агрессивный → 0 сделок |
| Our v4 (no-DD, milestones, churn, 300k) | agent2/dev | **+0.193** | +1.72% | 59% (10/17) | obs=1329, best win: +20.57% (BTC bull), worst: -14.37% (2025 bear) |
| Our v5 (H-006+H-007+H-008: z_score+soft_stop+anti-churn) | agent2/dev | 📋 плановый | — | — | obs ~1569, CHURN_BARS=10, SOFT_STOP_PCT=5%, старт сейчас |
| Our v6 (no-soft-stop, SIGMA_IDLE_EXEMPT, H-009 hour-of-day) | agent2/dev | **0.308** | +2.01% | 63% (12/19) | 19/23 окон (крэш на окне 20 из-за TL shape mismatch при смене N_ACTIONS 7→13); best +11.60%, worst -8.67% |
| Our v7 (v6 + SHORT/COVER capability) | agent2/dev | **+0.379** | +2.67% | 68% (13/19) | 13 actions, SHORT capability. Best window: +18.6% |
| Our v7.1 (500k steps) | agent2/dev | 0.365 | +2.54% | 65% (13/20) | More steps hurt — overfitting |
| Our v7.2 (ActionMasker + F&G + ma_cross) | agent2/dev | **-0.284** ❌ | -3.81% | 35% (8/23) | F&G = noise at 1h; obs 1691 too big for 300k steps |

---

## Архитектурные различия

| Параметр | Main Bot (master) | Our Bot (agent2/dev) |
|----------|-------------------|----------------------|
| Таймфрейм | 1d | 1h |
| Observation | 27 flat features (Quantum Leap v1) | 60×n_features + 9 (LOOKBACK window) |
| Observation size | 27 | ~1329 (v4), ~1571 (v6/v7 с H-006+H-009) |
| Шорт | ✅ | ✅ v7+ (SHORT 25/50/100% + COVER 25/50/100%) |
| Transfer learning | ✅ | ✅ (v3+) |
| Drawdown penalty | ✅ quadratic | ❌ Удалён в v4 (слишком агрессивный) |
| Idle penalty | ❌ | ✅ -0.0001 |
| Churn penalty | ✅ bars<3 → -0.002 | ✅ bars<10 → -0.003 (v6+) |
| Curriculum milestones | ✅ (игровые) | ✅ (1h-adapted: 168h/720h/2160h) |
| Funding rate | ✅ | ✅ (OI skip для окон >30 дней назад) |
| Time features | ✅ sin/cos | ✅ (v3+): episode + weekly cycle |
| Actions | 6 (incl. SHORT) | 7 (v1-v6), 13 (v7+: + SHORT/COVER 25/50/100%) |
| Z-score (log-normal) | ✅ rule-based TradeAnalyzer | ✅ raw feature → агент учится сам (v5+) |
| MA200 regime | ✅ | ✅ (v5+) |

---

## Гипотезы в работе

| # | Гипотеза | Статус | Ветка | Результат |
|---|----------|--------|-------|-----------|
| 1 | Transfer learning между окнами | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 2 | Time features (sin/cos) | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 3 | Quadratic DD penalty | 🔄 В коде | agent2/dev | Ждём прогон v3 |
| 4 | Volatility filter (торговать только волатильные сессии) | 📋 В очереди | — | — |
| 5 | SHORT capability | 📋 В очереди | — | — |
| 6 | Funding rate fix (API bug) | ✅ Готово | agent2/dev | OI skip для старых окон |
| 7 | OBI + VWAP deviation фичи | ✅ Готово | agent2/dev | obs: 1209→1329, первые в RL env |
| 8 | Session-based episodes (150 candles) | 📋 В очереди | — | — |

---

## История сканов основной ветки

| Дата | Новые фичи в master | Взяли себе? | Комментарий |
|------|---------------------|-------------|-------------|
| 2026-05-18 | OrderBook simulator, 3 режима (scalper/intraday/swing), стратпивот v2.0 | Частично | OBI идею взяли, режимы — нет (наш путь другой) |
| 2026-05-18 | Rolling trainer, risk_manager, binance_funding, trade_diary | Частично | TL + DD penalty взяли |
| 2026-05-18 | Quantum Leap v1: z_score_ln, regime, mu_60, sigma_20, ma_cross; anti-churn bars<3 | ✅ Взяли (H-006) | Наш подход: raw features в obs vs их rule-based пороги. +MA200 regime. Реализовано для v5 |
