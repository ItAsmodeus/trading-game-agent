"""
Правила игры — единственный источник истины.
Менять здесь, не в коде агента.
"""

GAME_RULES = {
    # Капитал
    "starting_capital":   10_000.0,   # $10k стартовый (на двоих агентов по $5k каждому)
    "game_over_pct":      0.20,        # остался <20% капитала → выбыл

    # Торговля
    "maker_fee":          0.0002,      # 0.02%
    "taker_fee":          0.0005,      # 0.05%
    "min_order_usd":      10.0,        # минимальный ордер
    "max_position_pct":   0.95,        # не более 95% в одной позиции
    "max_leverage":       3.0,

    # Ограничения (hard constraints)
    "max_orders_per_day": 100,         # > 100 — штраф, защита от overtrading
    "max_drawdown_pause": 0.15,        # просадка 15% → пауза 24ч
    "emergency_close":    0.25,        # просадка 25% → принудительное закрытие

    # Победа
    "episode_length":     252,         # дней (1 торговый год)
    "win_target":         1_000_000.0, # $1M (конечная цель, не критерий остановки)
}

# Инструменты которые агент может торговать
TRADEABLE_ASSETS = [
    "BTCUSDT",   # Tier 1 — старт
    "ETHUSDT",   # Tier 1 — старт
    # "SOLUSDT", # Tier 2 — после стабилизации
]

# Дискретный action space (уровень 1 прototipa)
ACTIONS = {
    0: "HOLD",         # ничего не делать
    1: "BUY",          # войти в long (100% свободного кэша)
    2: "SELL_SHORT",   # войти в short (100% свободного кэша)
    3: "CLOSE",        # закрыть текущую позицию
    4: "BUY_HALF",     # войти в long (50%)
    5: "SELL_HALF",    # войти в short (50%)
}

N_ACTIONS = len(ACTIONS)
N_FEATURES = 27  # 22 базовых + 3 Quantum Leap (mu, sigma, z_score) + 2 тренд (regime, ma_cross)
