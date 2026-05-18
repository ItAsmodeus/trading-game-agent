from dataclasses import dataclass, field
from typing import List


@dataclass
class GameConfig:
    # ── Капитал ───────────────────────────────────────────────────────────────
    STARTING_CAPITAL: float = 10_000.0   # USDT
    WIN_TARGET: float = 1_000_000.0
    LOSS_THRESHOLD: float = 0.20         # game over при капитале < 20% от старта

    # ── Комиссии Binance Futures ──────────────────────────────────────────────
    TAKER_FEE: float = 0.0004            # 0.04%
    MAKER_FEE: float = 0.0002            # 0.02%
    SLIPPAGE: float = 0.0003             # 0.03% среднее проскальзывание

    # ── Инструменты ───────────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    ])
    TIMEFRAME: str = "1h"

    # ── State space ───────────────────────────────────────────────────────────
    LOOKBACK: int = 60       # свечей в состоянии (60h ≈ 2.5 дня)
    N_FEATURES: int = 18     # признаков на свечу (OHLCV + 13 индикаторов)

    # ── Action space (дискретные) ─────────────────────────────────────────────
    # 0: HOLD
    # 1: BUY  25%  капитала
    # 2: BUY  50%  капитала
    # 3: BUY  100% капитала
    # 4: SELL 25%  позиции
    # 5: SELL 50%  позиции
    # 6: SELL 100% позиции (CLOSE)
    N_ACTIONS: int = 7

    # ── Reward ────────────────────────────────────────────────────────────────
    INACTION_PENALTY: float = -0.0001    # штраф за HOLD в кэше (нет открытой позиции)
    CHURN_PENALTY: float = -0.001       # штраф за закрытие позиции менее чем через 5 баров
    LEVERAGE_PENALTY_BASE: float = 0.01  # штраф за плечо > 3x
    DRAWDOWN_PENALTY_SCALE: float = 0.5  # масштаб штрафа за просадку > 10%

    # ── Ограничения риска ─────────────────────────────────────────────────────
    MAX_POSITION_RATIO: float = 0.95     # не более 95% капитала в позиции
    MAX_LEVERAGE: float = 5.0

    # ── Обучение ──────────────────────────────────────────────────────────────
    MAX_STEPS_PER_EPISODE: int = 24 * 30    # 30 дней в часах
    TRAIN_START: str = "2020-01-01"
    TRAIN_END: str = "2024-01-01"
    VAL_START: str = "2024-01-01"
    VAL_END: str = "2025-01-01"
    TEST_START: str = "2025-01-01"
    TEST_END: str = "2026-01-01"

    # ── Пути ──────────────────────────────────────────────────────────────────
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"


CFG = GameConfig()
