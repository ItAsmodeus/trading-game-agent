# Контекст проекта Trading Game Agent
## Для Евы — читай в начале каждой сессии

---

## Кто работает над проектом

- **Карим Абушаев** — инициатор, Senior Java Dev, изучает квантовую физику
- **shohrux-btc** — collaborator (подключился 18.05.2026, активен)
- **Ева** — AI ассистент (Claude Code, 7 скиллов активны)

---

## Статус на 18.05.2026 (вечер)

### Phase 1 — ЗАВЕРШЕНА ✅

| Метрика | Результат | Цель |
|---------|-----------|------|
| OOS Sharpe | **1.21** | > 1.2 ✅ |
| Доходность агента | **+8.5%** | > 0 ✅ |
| Buy & Hold | -12.9% | — |
| Max Drawdown | 2.9% | < 20% ✅ |
| Сделок (150 дней) | 43 | < 100 ✅ |

Агент бьёт Buy & Hold на **+21.4%** на медвежьем рынке.

### Что сделано сегодня

- ✅ Запущено и отработало обучение PPO (200k шагов, 2:14 мин)
- ✅ Исправлены 2 бага: numpy array → int в market.py и environment.py
- ✅ Аудит reward function: откатили агрессивный фикс (потеряли Sharpe 1.21→-2.25)
- ✅ Активированы 7 скиллов: rl-engineer, multi-agent, market-microstructure, risk-engine, pre-mortem, backtest-python, strategy-funding-arb
- ✅ Создан `training/rolling_trainer.py` — скользящее обучение (transfer learning)
- ✅ Создан `risk/risk_manager.py` — Kelly criterion, VaR, комментарии решений
- ✅ Создан `data/binance_funding.py` — загрузка Funding Rate с Binance
- ✅ Обновлён `game/environment.py` — добавлены 2 признака funding rate (N=22)
- ✅ Обновлён `game/rules.py` — N_FEATURES: 20 → 22

### Инсайты от товарища (shohrux-btc)

Его результаты: Rolling обучение дало win rate 27% → 59%.
Его баг: нормализация цены давала фиктивные просадки (у нас нет — мы используем log returns).
Следующий шаг для него: Rolling v2 (500k шагов/окно).

---

## Ключевые документы проекта

| Файл | Для кого | Что внутри |
|------|----------|-----------|
| `FOR_DEVELOPER.md` | Новый разработчик | Навигация, как запустить, карта проекта |
| `AGENT_BRAIN.md` | Все | Мозг бота: принципы, сигналы, стратегии, уроки |
| `CONTEXT.md` | Ева | Текущий статус, что сделано, следующие шаги |
| `TRADING_GAME_DESIGN.md` | Все | Полная техническая спецификация v2.0 |
| `TRADE_DIARY.md` | Карим | Дневник сделок (создаётся при каждом запуске) |

## Структура проекта (актуальная)

```
trading-game-agent/
├── game/
│   ├── rules.py        — GAME_RULES, ACTIONS, N_FEATURES=22
│   ├── market.py       — RealisticMarket (Almgren-Chriss slippage)
│   └── environment.py  — TradingEnv(gym.Env), 22 признака, 6 действий
├── data/
│   ├── loader.py           — ccxt + CSV + train/val/test split
│   └── binance_funding.py  — Funding Rate с Binance Futures API ← NEW
├── training/
│   └── rolling_trainer.py  — Rolling Training (скользящее окно) ← NEW
├── risk/
│   └── risk_manager.py     — Kelly, VaR, лимиты, комментарии ← NEW
├── agents/
│   └── random_agent.py     — RandomAgent + BuyAndHoldAgent бейзлайны
├── evaluation/
│   └── metrics.py          — Sharpe, Calmar, Sortino, red flags
├── train.py                — базовый запуск (статика)
└── requirements.txt
```

### Как запустить базовое обучение
```bash
cd /Users/karim/trading-game-agent
source venv/bin/activate
python train.py --symbol BTC/USDT --timeframe 1d --timesteps 200000
```

### Как запустить Rolling Training (новое)
```bash
source venv/bin/activate
python -m training.rolling_trainer --symbol BTC/USDT --timeframe 1d \
  --train-months 6 --step-months 2 --timesteps 500000
```

### Мониторинг
```bash
# TensorBoard (в отдельном терминале)
tensorboard --logdir ./logs/tensorboard
# → http://localhost:6006
```

---

## Дневник трейдера (новое)

**Файл:** `TRADE_DIARY.md` — создаётся автоматически при каждом запуске.

**Правило входа (зашито в код):** минимум 2 независимых основания + план выхода.
Без этого — бот не входит в сделку, логирует причину блокировки.

**Источники оснований:**
1. RSI (перекупленность/перепроданность)
2. MACD (импульс)
3. Bollinger Bands (границы)
4. Volume (подтверждение объёмом)
5. Моментум (динамика цены 1д/5д)
6. Funding Rate (перегрев рынка)
7. Fear & Greed Index + новости (бесплатный API, без ключа)

**Модули:**
- `data/news_fetcher.py` — Fear & Greed + CryptoPanic (опционально)
- `trading/trade_analyzer.py` — думает перед каждым входом
- `trading/trade_diary.py` — пишет Markdown дневник

## Активные скиллы и их роли

| Скилл | Роль |
|-------|------|
| rl-engineer | Алгоритмы: PPO → SAC → LSTM архитектура |
| multi-agent | Phase 2: Альфа vs Бета, Self-Play, League Training |
| market-microstructure | Симулятор, стакан, slippage, Funding Rate |
| risk-engine | Kelly, VaR, дневные лимиты — "квантовый аналитик" |
| pre-mortem | Edge cases до реализации |
| backtest-python | Walk-forward validation, VectorBT, метрики |
| strategy-funding-arb | Funding Rate арбитраж — первый реальный доход |

---

## Защита от банов (критично для daily trading)

```python
SAFE_TRADING_RULES = {
    "max_orders_per_day": 50,       # лимит в risk_manager.py
    "min_delay_between_orders": 3,  # секунды (рандом 1-5)
    "use_websocket_for_data": True,  # не REST polling
    "max_position_pct_of_volume": 0.01,
    "timeframe": "1h",              # не тик, не минута
    "order_type": "limit",          # maker комиссия
}
# Binance лимиты: 1200 req/min, 10 orders/sec, 100k orders/24h
```

---

## Стратегический пивот (18.05.2026)

После разговора с Никитой Котковским (трейдер) — скорректировали стратегию.
Подробности: `STRATEGY.md` | Риски: `risks/RISKS.md` | Персона: `personas/nikita_kotkovsky.md`

**Новый фокус:** играть не против HFT, а в нишах которые им неинтересны.
**Цель изменена:** не $1M, а **20-40% годовых системно**.

## Roadmap (обновлённый)

| Фаза | Что | Статус |
|------|-----|--------|
| 1 | PPO Swing агент, Sharpe > 1.0 | ✅ Sharpe=1.11, +9.9% |
| 1.5 | Rolling Training (главный рычаг) | 🔄 Реализовано, запустить |
| 1.5 | Funding Rate Scanner | 🔄 Модуль готов, нужен запуск |
| 2 | OOS тест на 6 разных периодах | ⏳ Май-Июнь 2026 |
| 2 | Self-Play: Альфа vs Бета | ⏳ Июль 2026 |
| 3 | Testnet 30 дней | ⏳ Август 2026 |
| 4 | Real money $500-1000 | ⏳ Сентябрь 2026 |
| 5 | Масштабирование $5k-10k | ⏳ Декабрь 2026 |
| — | Scalper | 🚫 Отложен (конкурируем с HFT) |

---

## Ключевые находки (не забыть)

1. **Reward penalty** — большой штраф за просадку случайно дал хорошую стратегию (консерватизм на медвежьем рынке). Не трогать пока Sharpe > 1.
2. **Rolling > Static** — товарищ доказал: win rate 27%→59%. Наш следующий запуск.
3. **N_FEATURES = 22** — добавили funding rate. Старые модели несовместимы.
4. **Transfer learning** — rolling trainer передаёт веса между окнами (меньший LR 1e-4 на дообучении).
5. **Funding Rate сигнал** — высокий FR (>0.05%/8ч) = рынок перегрет = снижать лонг.

---

## Технологии

- RL: Stable-Baselines3 (PPO сейчас, SAC в Phase 2)
- Среда: Gymnasium + PettingZoo (для multi-agent Phase 2)
- Данные: ccxt (Binance) + Binance Futures API (Funding Rate)
- Нейросети: PyTorch (MLP сейчас, LSTM в Phase 2)
- Метрики: quantstats, ручной Sharpe/VaR
- Мониторинг: TensorBoard
- Риск: Kelly Criterion + Historical VaR(99%)
