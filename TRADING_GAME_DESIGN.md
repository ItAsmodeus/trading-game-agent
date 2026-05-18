# Trading Game Design Document (TGD)
## Полная техническая спецификация v2.0

> Два соревнующихся RL-агента. Один рынок. Цель — $1 000 000.
> v1.0: 16 мая 2026 · v2.0: 18 мая 2026 · Карим Абушаев + Ева

---

## Часть 0: Концепция

```
TRADING GAME v1.0

Участники:   Агент Альфа  vs  Агент Бета
Арена:       Binance Futures + Bybit (USDT-M Perpetual)
Победа:      Первый кто достигает $1 000 000
Старт:       $5 000 каждому (итого $10 000)
Проигрыш:   Капитал упал ниже $1 000 (потеря 80%) → выбывает
Ничья:       12 месяцев — побеждает кто больше заработал

ГЛАВНОЕ ПРАВИЛО: Агенты НЕ знают стратегий заранее.
Только правила + инструменты → остальное сами.
```

---

## Часть 1: Multi-Agent Архитектура

### Выбранная парадигма: Asymmetric MARL + Fictitious Self-Play

Теоретическая база: Lanctot et al. (2017) "A Unified Game-Theoretic Approach to Multiagent RL" (DeepMind, arxiv:1711.00832). FSP сходится к Nash Equilibrium в zero-sum играх.

```
┌─────────────────────────────────────────────────────────────────┐
│                  ТОРГОВАЯ СРЕДА (Market Env)                     │
├─────────────────────────┬───────────────────────────────────────┤
│      АГЕНТ АЛЬФА        │           АГЕНТ БЕТА                  │
│                         │                                        │
│  PolicyNet_α (PPO)      │      PolicyNet_β (SAC)                │
│  ValueNet_α             │      ValueNet_β                       │
│  Капитал: $5,000        │      Капитал: $5,000                  │
│                         │                                        │
│  Observation:           │      Observation:                      │
│  - Market state (shared)│      - Market state (shared)          │
│  - Own portfolio        │      - Own portfolio                   │
│  - β's last action      │      - α's last action                │
│    (публично)           │        (публично)                     │
│  - β's P&L (скрыто)     │      - α's P&L (скрыто)              │
└─────────────────────────┴───────────────────────────────────────┘
```

### Инсайт из DQN Paper (Mnih et al., Nature 2015)

**Target Network = Self-Play Opponent** — это не случайное совпадение, это одна математическая идея:

В DQN: target network отстаёт от основной сети на C шагов → стабилизирует обучение.
В self-play: агент-бета = старая копия агента-альфа → стабилизирует обучение.

Нестационарная цель (изменяется на каждом шаге) → нестабильное обучение.
Решение в обоих случаях: зафиксировать цель, обновлять редко.

```python
# DQN: target network обновляется каждые C=10000 шагов
if step % C == 0:
    target_network.load_state_dict(policy_network.state_dict())

# Self-play: бета = старая альфа, обновляется каждые N эпизодов
if episode % N == 0:
    beta_agent = copy.deepcopy(alpha_agent)
```

**Второй инсайт из DQN:** один алгоритм победил на 43 из 49 игр Atari.
→ Нам не нужно отдельно настраивать агента под BTC vs ETH.
→ Нормализованные признаки + единая архитектура = торгует любой парой.

### League Training для стабильности

Проблема Self-Play: один агент намного сильнее → слабый не учится.
Решение: AlphaStar League (Vinyals et al., 2019, arxiv:1912.06680).

```python
class AgentLeague:
    """
    Агент торгует против:
    - 50% текущего противника (актуальность)
    - 35% лучшей исторической версии (стабильность)
    - 15% случайной версии (диверсификация)
    """
    def sample_opponent(self, agent_id) -> BaseAgent:
        r = np.random.random()
        if r < 0.50: return self.main_agents[1 - agent_id]
        elif r < 0.85: return self._best_historical()
        else: return self._random_historical()
```

---

## Часть 2: Биржи

### Binance Futures (USDT-M) — основная

**Инструменты:**
```
Tier 1 (старт):  BTCUSDT, ETHUSDT
Tier 2 (после):  SOLUSDT, BNBUSDT
Tier 3 (агент сам):  Альткоины с высоким funding
```

### Bybit — для Funding Rate Arbitrage

```
Binance funding = +0.05%, Bybit = +0.02%
Short Binance (получаем 0.05%) + Long Bybit (платим 0.02%)
Нетто: +0.03% за 8 часов = +27% годовых без плеча
```

---

## Часть 3: Observation Space (256 признаков)

```python
@dataclass
class MarketState:
    # Ценовые данные (нормализованы как log-returns)
    returns_1m:   float
    returns_5m:   float
    returns_1h:   float
    returns_4h:   float
    returns_24h:  float
    volatility_1h: float
    volatility_24h: float

    # Технические индикаторы
    rsi_14:        float  # (RSI - 50) / 50 → [-1,1]
    macd_hist:     float
    bb_position:   float  # [-1,1]
    atr_14:        float
    adx_14:        float
    volume_ratio:  float

    # Order Book (L2)
    bid_ask_spread:    float
    order_imbalance:   float
    depth_5bps:        float
    large_order_flag:  bool

    # Крипто-специфика
    funding_rate:          float
    funding_predicted:     float
    open_interest_change:  float
    liquidations_1h:       float

    # Сентимент
    fear_greed_index: float
    news_sentiment:   float
    on_chain_flow:    float

    # Собственное состояние
    position:          float
    unrealized_pnl:    float
    cash_ratio:        float
    drawdown_current:  float
    days_remaining:    float
    progress_to_goal:  float

    # Конкурент
    competitor_activity:    float
    competitor_last_action: int
```

---

## Часть 4: Action Space

```python
class HierarchicalActionSpace:
    """
    Уровень 1 (дискретный): ЧТО делать?
      0=HOLD, 1=OPEN_LONG, 2=OPEN_SHORT,
      3=CLOSE_LONG, 4=CLOSE_SHORT, 5=REDUCE, 6=SWITCH_ASSET

    Уровень 2 (непрерывный): СКОЛЬКО?
      size ∈ [0.01, 1.0]

    Уровень 3: КАК?
      order_type: MARKET=0, LIMIT=1
      price_offset ∈ [-0.005, +0.005]
    """
```

**Жёсткие ограничения (нарушить невозможно):**
```python
HARD_CONSTRAINTS = {
    "max_position_pct":    0.10,   # 10% в одном инструменте
    "max_leverage":         3.0,
    "max_daily_loss_pct":  0.03,   # 3% → стоп
    "max_drawdown":        0.15,   # 15% → пауза 24ч
    "max_orders_per_day": 100,
}
```

---

## Часть 5: Reward Function

```python
def compute_reward(state, next_state, action, costs):
    log_ret = np.log(next_state.portfolio / state.portfolio)

    drawdown = (state.peak - next_state.portfolio) / state.peak
    dd_penalty = 3.0 * (drawdown ** 2) if drawdown > 0.05 else 0
    if drawdown > 0.25:
        dd_penalty = 10.0 * drawdown

    cost_penalty = costs / state.portfolio * 8

    progress = next_state.portfolio / 1_000_000
    goal_reward = progress * 0.1

    reward = log_ret - dd_penalty - cost_penalty + goal_reward
    return float(np.clip(reward, -10.0, 10.0))
```

---

## Часть 6: Нейронная архитектура

### Multi-Timeframe Transformer

Temporal Fusion Transformer (Lim et al., 2021, arxiv:2012.09101) — лучший результат vs LSTM на 21 финансовом датасете.

**Инсайт DQN → трейдинг:**
DQN стекает 4 кадра подряд чтобы агент видел скорость (velocity).
Для трейдинга аналог — стек таймфреймов [1m, 5m, 1h, 1d]:
агент видит краткосрочный импульс И долгосрочный тренд одновременно.

```python
class MultiTimescaleTransformer(nn.Module):
    def __init__(self, state_dim=256, d_model=128, n_heads=8):
        super().__init__()
        self.tf_encoders = nn.ModuleDict({
            '1m': nn.TransformerEncoder(..., num_layers=2),
            '1h': nn.TransformerEncoder(..., num_layers=2),
            '1d': nn.TransformerEncoder(..., num_layers=2),
        })
        self.cross_attention = nn.MultiheadAttention(128, 8)
```

---

## Часть 7: Реалистичный симулятор

```
SIMULATOR_CHECKLIST:
✅ bid_ask_spread      # разница лучших цен
✅ slippage_model      # Almgren-Chriss
✅ partial_fills       # недостаточно ликвидности
✅ latency_ms          # 50-500мс
✅ maker_taker_fees    # разные комиссии
✅ funding_rates       # для перпетуальных
✅ liquidation_engine  # margin call
✅ market_impact       # наш ордер двигает цену
```

---

## Часть 8: Training Pipeline

### Curriculum Learning: 5 стадий

| Стадия | Условия | Target Sharpe | Эпизодов |
|--------|---------|--------------|---------|
| 1. Простой рынок | 1 актив, BUY/SELL/HOLD, нет комиссий | 0.5 | 10K |
| 2. Реалистичный | Комиссии + slippage | 0.8 | 50K |
| 3. Multi-asset | 3 актива, иерархические действия | 1.0 | 200K |
| 4. Self-Play | Добавляем конкурента | 1.2 | 1M |
| 5. League Training | Пул версий + sentiment | 1.5 | ∞ |

---

## Часть 9: Дыры в проекте (честный разбор)

### Критические

**1. $1M цель нереалистична как win condition**
$5k → $1M = 200x за 12 месяцев = ~+46% каждый месяц.
Даже лучшие хедж-фонды дают 30-50% в год, не в месяц.
→ Решение: $1M — это вдохновляющая цель для нарратива, а не критерий остановки.
Win condition = максимизировать Calmar ratio за эпизод.

**2. Нестационарность рынков**
Агент обученный на bull 2020-2021 провалится на bear 2022.
EWC частично решает, но не гарантирует.
→ Решение: Walk-forward training + постоянное дообучение на свежих данных.

**3. Multi-agent instability**
Пока учится Альфа, меняется Бета → среда нестационарна для обоих.
Это фундаментальная проблема MARL.
→ Решение: Opponent Lag (бета = Альфа минус 10k шагов).

**4. Reward hacking**
Агент найдёт способ максимизировать reward не через торговлю.
Например: никогда не торговать (нет комиссий = нет штрафа).
→ Решение: Добавить штраф за idle (нет позиции > N дней).

**5. Look-ahead bias в нормализации**
Если нормализуем данные по всему периоду — агент "видит" будущую статистику.
→ Решение: Только rolling нормализация (скользящее окно прошлого).

### Архитектурные

**6. Python + ZeroMQ + C++ — лишняя сложность для прototipa**
Три языка, три точки отказа, сложный дебаггинг.
→ Решение для прototipa: чистый Python. C++ только в Phase 3 (production).

**7. 64 параллельных среды нереально на MacBook Air**
MacBook Air = 8-16GB RAM. 64 envs × 1 episode = ~8GB только данных.
→ Решение: старт с 4-8 envs, масштабировать на облаке.

**8. Asymmetric PPO vs SAC делает сравнение нечестным**
Разные алгоритмы → разная скорость обучения → один всегда проигрывает не из-за стратегии.
→ Решение: для начала — оба PPO, разные seeds. SAC добавить в Phase 4.

**9. FinBERT требует новостей в датасете**
При бэктесте на исторических данных — где брать новости за 2020?
→ Решение: news sentiment добавить только при live trading. В симуляции — без него.

---

## Часть 10: Live Trading

### Shadow Mode → Постепенный деплой

| Фаза | Описание | Условие перехода |
|------|---------|-----------------|
| Shadow | Виртуальная торговля | 30 дней Sharpe > 1.0 |
| Micro | $100 реальных | 14 дней без крупных ошибок |
| Semi-auto | $1,000 | DD < 10% за месяц |
| Full-auto | Полная автономия | Sharpe > 1.5 за квартал |

### Kill Switch

```cpp
void trigger(TriggerReason reason) {
    emergency_close_all_positions();
    notify_telegram(reason);
    audit_log_.write(reason);
}
```

---

## Прototip (текущий статус)

### Что реализовано

```
trading-game-agent/
├── game/
│   ├── rules.py        ✅ Константы игры
│   ├── market.py       ✅ Almgren-Chriss slippage + fees
│   └── environment.py  ✅ Gymnasium среда (20 признаков, 6 действий)
├── data/
│   └── loader.py       ✅ CSV + ccxt + train/val/test split
├── agents/
│   └── random_agent.py ✅ RandomAgent + BuyAndHoldAgent бейзлайны
├── evaluation/
│   └── metrics.py      ✅ Sharpe, Calmar, Sortino, красные флаги
├── train.py            ✅ PPO через SB3, OOS тест, метрики
├── papers/
│   └── DQN_Nature_2015_Mnih.pdf  ✅
└── requirements.txt    ✅
```

### Как запустить прototip

```bash
cd /Users/karim/trading-game-agent
pip install -r requirements.txt

# Вариант 1: скачать данные с биржи и обучить
python train.py --symbol BTC/USDT --timeframe 1d --timesteps 200000

# Вариант 2: использовать свой CSV
python train.py --data my_btc_data.csv --timesteps 500000
```

### Метрики для перехода к реальным деньгам

```python
PRODUCTION_CRITERIA = {
    "oos_sharpe":     1.2,   # не in-sample!
    "max_drawdown":   0.15,
    "win_rate":       0.52,
    "profit_factor":  1.3,
    "calmar_ratio":   1.0,
    "bear_market":    True,  # работает на данных 2022
}
```

---

## Roadmap

| Фаза | Задача | Ключевой результат |
|------|--------|--------------------|
| **0 (сейчас)** | TradingEnv, RealisticMarket, DataLoader, PPO | Один агент учится, бьёт random |
| **1** | OOS Sharpe > 0.8, визуализация equity curve | Агент стабильно работает |
| **2** | Self-Play арена, два агента, League Training | Emergent strategies |
| **3** | C++ execution bridge, DualRiskManager | Production архитектура |
| **4** | Shadow mode 30 дней, kill switches | Testnet валидация |
| **5** | $100 → $1k → full auto | Реальные деньги |

---

## Related Projects (из deep research)

| Проект | Ссылка | Почему важен |
|--------|--------|-------------|
| FinRL | github.com/AI4Finance-Foundation/FinRL | 11k⭐, наша основа |
| TradeMaster | github.com/TradeMaster-NTU/TradeMaster | MARL бенчмарк |
| PyMarketSim | github.com/dipplestix/pymarketsim | LOB + multi-agent, ICAIF 2024 |
| JAX-LOB | github.com/KangOxford/jax-lob | GPU LOB для масштаба |
| Habr 2025 (RU) | habr.com/ru/articles/934258 | Единственная свежая RU статья с кодом |

---

## Ключевые papers

1. Mnih et al. (2015) — DQN: Nature 14236 → `papers/DQN_Nature_2015_Mnih.pdf`
2. Liu et al. (2021) — FinRL: arxiv:2011.09607
3. Schulman et al. (2017) — PPO: arxiv:1707.06347
4. Haarnoja et al. (2018) — SAC: arxiv:1801.01290
5. Lim et al. (2021) — TFT: arxiv:2012.09101
6. Lanctot et al. (2017) — MARL Self-Play: arxiv:1711.00832
7. Kirkpatrick et al. (2017) — EWC: arxiv:1612.00796
8. Vinyals et al. (2019) — AlphaStar League: arxiv:1912.06680

---

*v1.0: 16 мая 2026 · v2.0: 18 мая 2026 · Карим + Ева*
