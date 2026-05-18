"""
Запуск обучения агента.
Использование:
    python train.py --data btc_daily.csv --timesteps 500000
    python train.py --exchange binance --symbol BTC/USDT --timesteps 200000
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from data.loader import load_from_csv, load_from_exchange, train_test_split
from game.environment import TradingEnv
from game.rules import GAME_RULES, ACTIONS
from game.trading_modes import get_mode, get_game_rules, MODES
from trading.trade_diary import TradeDiary, init_diary
from trading.trade_analyzer import TradeAnalyzer, MarketSnapshot
from data.news_fetcher import NewsFetcher
from data.orderbook_sim import OrderBookSimulator


def make_env(df, config):
    def _init():
        return TradingEnv(df, config)
    return _init


def train(args):
    # 0. Режим торговли
    mode_cfg  = get_mode(args.mode)
    game_rules = get_game_rules(mode_cfg)
    timeframe  = args.timeframe or mode_cfg.timeframe

    print(f"\n{'='*50}")
    print(f"Режим: {mode_cfg.name}")
    print(f"{mode_cfg.description_ru}")
    print(f"Таймфрейм: {timeframe} | Стоп: {mode_cfg.stop_loss_pct:.1%} | Тейк: {mode_cfg.take_profit_pct:.1%}")
    print(f"{'='*50}\n")

    # 1. Загрузить данные
    if args.data:
        df_full = load_from_csv(args.data)
    else:
        print(f"Загружаю {args.symbol} с {args.exchange}...")
        df_full = load_from_exchange(args.symbol, timeframe, limit=mode_cfg.limit_bars, exchange=args.exchange)

    train_df, val_df, test_df = train_test_split(df_full)

    # 2. Создать среды
    n_envs = min(args.n_envs, os.cpu_count() or 4)
    train_envs = make_vec_env(make_env(train_df, game_rules), n_envs=n_envs)
    eval_env   = TradingEnv(val_df, game_rules)

    # 3. Callbacks
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints/best/",
        log_path="./logs/",
        eval_freq=max(5_000 // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20_000 // n_envs, 1),
        save_path="./checkpoints/",
        name_prefix="ppo_trading",
    )

    # 4. Модель
    model = PPO(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # entropy для exploration
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="auto",
    )

    # 5. Обучение
    print(f"\nОбучаю {args.timesteps:,} шагов на {n_envs} параллельных средах...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save("./checkpoints/final_model")
    print("\nМодель сохранена: ./checkpoints/final_model")

    # 6. Тест на test set с дневником трейдера
    print("\nТест на OOS данных (test set)...")

    # Инициализируем стакан для скальпера
    ob_sim = OrderBookSimulator(test_df) if args.mode == "scalper" else None

    # Инициализируем дневник в папке режима
    diary_dir  = f"./bots/{args.mode}"
    diary_path = f"{diary_dir}/DIARY.md"
    os.makedirs(diary_dir, exist_ok=True)
    init_diary(diary_path)
    diary    = TradeDiary(path=diary_path, session_label=f"OOS тест [{mode_cfg.name}] — {args.symbol} {timeframe}")
    analyzer = TradeAnalyzer(news_fetcher=NewsFetcher(), mode=args.mode)
    diary.start_session(symbol=args.symbol, capital=game_rules["starting_capital"], mode="backtest")

    test_env  = TradingEnv(test_df, game_rules)
    obs, _    = test_env.reset()
    portfolio_history = [GAME_RULES["starting_capital"]]
    open_trade_id     = None

    step = 0
    while True:
        raw_action, _ = model.predict(obs, deterministic=True)
        action_int    = int(raw_action)

        # Текущее состояние для аналитика
        idx   = test_env.step_idx
        price = float(test_df["close"].iloc[min(idx, len(test_df)-1)])
        date  = str(test_df["timestamp"].iloc[min(idx, len(test_df)-1)])[:10]

        snap = MarketSnapshot(
            price         = price,
            rsi           = float(obs[4]),
            macd          = float(obs[6]),
            bb_position   = float(obs[7]),
            volume_ratio  = float(obs[5]),
            ret_1d        = float(obs[0]),
            ret_5d        = float(obs[1]),
            ret_20d       = float(obs[2]),
            volatility    = float(obs[3]),
            atr           = float(obs[8]),
            candle_body   = float(obs[9]),
            funding_rate  = float(obs[20]) * 0.001 if len(obs) > 20 else 0.0,
            position      = float(test_env.position),
            portfolio     = float(test_env.portfolio),
            peak_portfolio= float(test_env.peak_portfolio),
            orderbook     = ob_sim.snapshot(min(idx, len(test_df)-1)) if ob_sim else None,
            mode          = args.mode,
            date          = date,
            symbol        = args.symbol,
        )

        # Аналитик думает перед действием
        decision = analyzer.analyze(snap, action_int)

        # Логируем сделку если это не HOLD
        if decision.action != 0:
            trade_id = diary.log_trade(decision, snap, step)
            if trade_id:
                # Закрываем предыдущую если была
                if open_trade_id and open_trade_id != trade_id:
                    diary.log_close(open_trade_id, price, "Новая позиция открыта", snap.portfolio, step)
                open_trade_id = trade_id
            elif decision.action == 3 and open_trade_id:
                diary.log_close(open_trade_id, price, "Агент решил закрыть", snap.portfolio, step)
                open_trade_id = None

        # Исполняем (возможно заменённое) действие
        obs, reward, terminated, truncated, info = test_env.step(decision.action)
        portfolio_history.append(info["portfolio"])
        step += 1

        if terminated or truncated:
            # Закрываем открытую позицию
            if open_trade_id:
                diary.log_close(open_trade_id, price, "Конец тестового периода", info["portfolio"], step)
            break

    # Итоговые метрики
    final    = portfolio_history[-1]
    start    = portfolio_history[0]
    ret      = (final / start - 1) * 100
    bnh      = (float(test_df["close"].iloc[-1]) / float(test_df["close"].iloc[0]) - 1) * 100

    returns  = np.diff(portfolio_history) / np.array(portfolio_history[:-1])
    sharpe   = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
    peak     = max(portfolio_history)
    drawdown = (peak - min(portfolio_history[portfolio_history.index(peak):])) / peak * 100

    diary.end_session(final, start, info["trade_count"])

    print(f"\n{'='*50}")
    print(f"Результат на OOS тесте:")
    print(f"  Доходность агента:  {ret:+.1f}%")
    print(f"  Buy & Hold:         {bnh:+.1f}%")
    print(f"  Sharpe:             {sharpe:.2f}")
    print(f"  Max Drawdown:       {drawdown:.1f}%")
    print(f"  Сделок:             {info['trade_count']}")
    print(f"{'='*50}")
    print(f"\n📒 Дневник трейдера записан: {diary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Game Agent — обучение")
    parser.add_argument("--data",       type=str, default=None,       help="Путь к CSV файлу")
    parser.add_argument("--exchange",   type=str, default="binance",  help="Биржа (ccxt)")
    parser.add_argument("--symbol",     type=str, default="BTC/USDT", help="Пара")
    parser.add_argument("--timeframe",  type=str, default=None,       help="Таймфрейм (авто из режима)")
    parser.add_argument("--timesteps",  type=int, default=500_000,    help="Шагов обучения")
    parser.add_argument("--n_envs",     type=int, default=4,          help="Параллельных сред")
    parser.add_argument("--mode",       type=str, default="swing",
                        choices=["swing", "intraday", "scalper"],    help="Режим торговли")
    args = parser.parse_args()

    train(args)
