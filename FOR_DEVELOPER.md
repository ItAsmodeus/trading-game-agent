# 👋 Добро пожаловать в Trading Game Agent

> Читай этот документ первым. Он даст полную картину за 10 минут.
> Авторы: Карим Абушаев + Ева (Claude Code) + shohrux-btc

---

## Что это за проект

Два соревнующихся RL-агента торгуют криптовалютой с одной целью: вырастить $10,000 → $1,000,000.

```
АГЕНТ АЛЬФА  vs  АГЕНТ БЕТА
     ↕                ↕
   PPO             SAC/PPO
     ↕                ↕
        РЫНОК (BTC/USDT)
             ↕
      Победитель = $1M
```

Агенты **не знают стратегий заранее** — только правила игры. Всё остальное находят сами через миллионы эпизодов обучения. Это RL (Reinforcement Learning), не алготрейдинг с хардкод-правилами.

---

## С чего начать (5 шагов)

### Шаг 1: Прочитай эти три документа

```
1. FOR_DEVELOPER.md   ← ты здесь, навигация по проекту
2. AGENT_BRAIN.md     ← мозг бота: принципы, сигналы, стратегии
3. CONTEXT.md         ← текущий статус, что сделано, что дальше
```

### Шаг 2: Установи окружение

```bash
git clone https://github.com/ItAsmodeus/trading-game-agent
cd trading-game-agent

python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install tensorboard            # для мониторинга
```

### Шаг 3: Запусти первый прогон

```bash
python train.py --symbol BTC/USDT --timeframe 1d --timesteps 200000
```

Ждёшь ~3 минуты. В конце видишь:
```
Результат на OOS тесте:
  Доходность агента:  +8.5%
  Buy & Hold:         -12.9%
  Sharpe:             1.21
```

### Шаг 4: Читай дневник трейдера

После прогона открой `TRADE_DIARY.md` — там каждая сделка с обоснованием на русском языке.

### Шаг 5: Смотри обучение в TensorBoard

```bash
# В отдельном терминале:
tensorboard --logdir ./logs/tensorboard
# Открой: http://localhost:6006
```

---

## Карта проекта — каждая папка и файл

```
trading-game-agent/
│
├── 📋 ДОКУМЕНТЫ (читай первыми)
│   ├── FOR_DEVELOPER.md      ← этот файл, старт
│   ├── AGENT_BRAIN.md        ← мозг бота, принципы и стратегии
│   ├── CONTEXT.md            ← текущий статус проекта
│   ├── TRADING_GAME_DESIGN.md← полная техническая спецификация v2.0
│   ├── TRADING_GAME_AGENT.md ← концептуальное исследование идеи
│   └── TRADE_DIARY.md        ← дневник сделок (создаётся при запуске)
│
├── 🎮 game/ — правила игры и торговая среда
│   ├── rules.py       ← GAME_RULES (комиссии, лимиты), ACTIONS, N_FEATURES=22
│   ├── environment.py ← TradingEnv(gym.Env) — главная среда обучения
│   └── market.py      ← RealisticMarket — симулятор с реальным slippage
│
├── 📊 data/ — загрузка и подготовка данных
│   ├── loader.py           ← OHLCV с Binance через ccxt + CSV + train/val/test split
│   ├── binance_funding.py  ← Funding Rate с Binance Futures (кэшируется)
│   └── news_fetcher.py     ← Fear & Greed Index + CryptoPanic новости
│
├── 🤖 agents/ — агенты (бейзлайны)
│   └── random_agent.py ← RandomAgent и BuyAndHoldAgent для сравнения
│
├── 🏋️ training/ — обучение
│   └── rolling_trainer.py ← Rolling Training: скользящее окно 6мес/шаг 2мес
│
├── ⚠️ risk/ — управление рисками
│   └── risk_manager.py ← Kelly, VaR, дневные лимиты, "квантовый аналитик"
│
├── 📒 trading/ — торговая логика
│   ├── trade_analyzer.py ← думает перед входом: 2+ оснований или HOLD
│   └── trade_diary.py    ← пишет Markdown дневник каждой сделки
│
├── 📈 evaluation/ — метрики
│   └── metrics.py ← Sharpe, Calmar, Sortino, красные флаги
│
├── 📁 papers/ — исследовательские статьи
│   └── DQN_Nature_2015_Mnih.pdf ← базовая статья DQN (Atari)
│
├── train.py          ← ТОЧКА ВХОДА: запуск обучения + тест + дневник
├── requirements.txt  ← все зависимости Python
└── venv/             ← виртуальное окружение (не коммитить)
```

---

## Как работает обучение (за 2 минуты)

```
1. Загружаем BTC данные с Binance (2000 дней)
   loader.py → ccxt → DataFrame

2. Разбиваем: 70% train | 15% val | 15% test
   Никогда не перемешиваем временные ряды!

3. Создаём торговую среду (TradingEnv)
   Агент видит 22 признака:
   - Цена: log returns за 1/5/20 дней
   - Технические: RSI, MACD, Bollinger, ATR, Volume
   - Портфель: позиция, просадка, прогресс к цели
   - Время: sin/cos кодирование
   - Новое: funding rate (fr_normalized, fr_ma_24h)

4. PPO агент учится (200k шагов = ~3 минуты на CPU)
   На каждом шаге: смотрит → действует → получает reward → обновляет сеть

5. Тест на OOS данных с дневником трейдера
   Каждая сделка записывается в TRADE_DIARY.md
   Правило: минимум 2 основания или вход заблокирован
```

---

## Как добавлять код

### Добавить новый сигнал в аналитик

Открой `trading/trade_analyzer.py`, метод `_collect_reasons()`.
Добавь метод `_my_signal_reason()` по аналогии с `_rsi_reason()`.
Зарегистрируй в `_collect_reasons()`.
Обнови `AGENT_BRAIN.md` — добавь сигнал в каталог.

### Добавить новую стратегию

1. Опиши стратегию в `AGENT_BRAIN.md` (Часть 3)
2. Если нужен новый тип действия — добавь в `game/rules.py → ACTIONS`
3. Если нужны новые признаки — добавь в `game/environment.py → _observe()`
4. Обнови `N_FEATURES` в `game/rules.py`

### Запустить Rolling Training

```bash
python -m training.rolling_trainer \
  --symbol BTC/USDT \
  --timeframe 1d \
  --train-months 6 \
  --step-months 2 \
  --timesteps 500000
```

Результаты: `./checkpoints/rolling/` + отчёт в терминале.

---

## Ключевые метрики — что значат числа

| Метрика | Плохо | Хорошо | Отлично | Текущий результат |
|---------|-------|--------|---------|-------------------|
| Sharpe | < 0 | > 1.0 | > 1.5 | **1.21** ✅ |
| Max Drawdown | > 25% | < 20% | < 10% | **2.9%** ✅ |
| Win Rate | < 40% | > 50% | > 60% | в работе |
| ep_rew_mean | стабильно негативный | растёт | приближается к 0 | -296 (было -1460) |

---

## Частые вопросы

**Q: Почему Python, а не C++?**
A: Python — для исследований и обучения. C++ будет в execution engine (Phase 4) когда будем торговать реальными деньгами. Сначала найди работающую стратегию, потом оптимизируй исполнение.

**Q: Почему reward отрицательный во время обучения?**
A: Нормально. ep_rew_mean = -1460 → -296 означает прогресс. Агент учится терять меньше.

**Q: Как проверить что модель не переобучилась?**
A: OOS Sharpe должен быть > 50% от in-sample Sharpe. Если in-sample = 3.0, а OOS = 0.3 → переобучение.

**Q: Можно ли запускать на реальных деньгах?**
A: Нет. Сейчас Phase 1 (симуляция). Реальные деньги — Phase 5. До этого: testnet 30 дней.

**Q: Как работает правило 2 оснований?**
A: `trade_analyzer.py` проверяет 7 источников сигналов. Если < 2 совпадают — вход заблокирован, действие заменяется на HOLD. Это в коде, не просто документация.

---

## Контакты и ссылки

- **GitHub:** https://github.com/ItAsmodeus/trading-game-agent
- **Карим:** @karim_product (Telegram)
- **Дизайн-документ:** `TRADING_GAME_DESIGN.md`
- **Теория:** papers/DQN_Nature_2015_Mnih.pdf

---

## Как вносить изменения

1. Сделал изменение → обнови `CONTEXT.md` (статус) и `AGENT_BRAIN.md` (если изменил логику)
2. Каждый коммит = одно логическое изменение
3. В сообщении коммита: `тип: описание` (feat: / fix: / docs: / refactor:)
4. Настрой git перед первым коммитом:

```bash
git config --global user.name "твоё имя"
git config --global user.email "твой@email.com"
```

---

*Документ обновляется при каждом значимом изменении проекта.*
*Последнее обновление: 18.05.2026*
