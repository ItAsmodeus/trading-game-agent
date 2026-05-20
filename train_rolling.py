"""
train_rolling.py — Walk-forward rolling window training.

Modes:
  walkforward  Train on 6mo -> eval next 1mo -> slide 1mo. Reports per-window Sharpe.
  latest       Train on most recent 6 months. Saves production model.

Usage:
  python train_rolling.py --mode walkforward --symbol BTC/USDT
  python train_rolling.py --mode latest --symbol BTC/USDT
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import CFG
from env.data_loader import load_window
from env.trading_env import TradingEnv


# ─── Rolling window parameters ──────────────────────────────────────────────
TRAIN_MONTHS  = 6
VAL_MONTHS    = 1
SLIDE_MONTHS  = 2     # default: slide 2 months (fewer windows, faster)
TIMESTEPS_PER_WINDOW = 200_000
N_ENVS        = 4
N_EVAL_EPISODES = 1   # one full-period run per val window (deterministic)

MIN_TRAIN_ROWS = CFG.LOOKBACK + CFG.MAX_STEPS_PER_EPISODE + 50
MIN_VAL_ROWS   = CFG.LOOKBACK + 50


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _add_months(ts: pd.Timestamp, n: int) -> pd.Timestamp:
    return ts + pd.DateOffset(months=n)


def date_windows(data_start: str, data_end: str, slide: int = SLIDE_MONTHS):
    """Generate (train_start, train_end, val_end) tuples."""
    start = pd.Timestamp(data_start, tz="UTC")
    end   = pd.Timestamp(data_end,   tz="UTC")
    ts = start
    while True:
        te  = _add_months(ts, TRAIN_MONTHS)
        ve  = _add_months(te, VAL_MONTHS)
        if ve > end:
            break
        yield ts.strftime("%Y-%m-%d"), te.strftime("%Y-%m-%d"), ve.strftime("%Y-%m-%d")
        ts = _add_months(ts, slide)


def linear_schedule(initial: float, final_frac: float = 0.1):
    def fn(progress: float) -> float:   # progress: 1.0 -> 0.0
        return final_frac * initial + (1 - final_frac) * initial * progress
    return fn


def build_model(train_env, tensorboard_log: str = "logs/tensorboard/") -> MaskablePPO:
    return MaskablePPO(
        "MlpPolicy", train_env,
        verbose=0,
        tensorboard_log=tensorboard_log,
        learning_rate=linear_schedule(1e-4, 0.1),
        n_steps=2048,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.15, 0.5),
        clip_range_vf=0.2,
        ent_coef=0.02,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
    )


def train_on_window(symbol: str, train_data: pd.DataFrame, timesteps: int,
                    previous_model: Optional[MaskablePPO] = None,
                    reward_mode: str = "pnl",
                    window_idx: int = 0) -> MaskablePPO:
    def make_env():
        def _init():
            env = TradingEnv(symbol=symbol, data=train_data, random_start=True,
                             noise_scale=0.02, reward_mode=reward_mode)
            env = ActionMasker(env, lambda e: e.action_masks())
            return Monitor(env)
        return _init

    train_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])

    use_tl = False
    if previous_model is not None:
        # Verify obs_size compatibility before transfer learning
        prev_obs = previous_model.observation_space.shape[0]
        curr_obs = train_env.observation_space.shape[0]
        if prev_obs == curr_obs:
            use_tl = True
        else:
            print(f"[warn] obs_size mismatch ({prev_obs} vs {curr_obs}), skipping TL")

    tb_log = f"logs/tensorboard/"
    if use_tl:
        model = MaskablePPO(
            "MlpPolicy", train_env,
            verbose=0,
            tensorboard_log=tb_log,
            learning_rate=linear_schedule(5e-5, 0.1),
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.1, 0.5),
            clip_range_vf=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64])),
        )
        try:
            prev_sd = copy.deepcopy(previous_model.policy.state_dict())
            curr_sd = model.policy.state_dict()
            # Partial TL: copy feature extraction layers only, reset action head
            # to avoid transferring directional bias between market regimes
            loaded, skipped = 0, 0
            for k in prev_sd:
                if k in curr_sd and curr_sd[k].shape == prev_sd[k].shape \
                        and not k.startswith("action_net"):
                    curr_sd[k] = prev_sd[k]
                    loaded += 1
                else:
                    skipped += 1
            model.policy.load_state_dict(curr_sd)
            print(f"  [TL] loaded={loaded} reset_action_head={skipped}", flush=True)
        except RuntimeError:
            print("  [TL skip] shape mismatch — training from scratch", flush=True)
            model = build_model(train_env, tb_log)
        model.learn(total_timesteps=timesteps, progress_bar=False,
                    reset_num_timesteps=False, tb_log_name=f"window_{window_idx:02d}")
    else:
        model = build_model(train_env, tb_log)
        model.learn(total_timesteps=timesteps, progress_bar=False,
                    tb_log_name=f"window_{window_idx:02d}")

    train_env.close()
    return model


def eval_on_window(model: PPO, symbol: str, val_data: pd.DataFrame,
                   reward_mode: str = "pnl") -> dict:
    """Run agent through full val period (no random start). Returns metrics."""
    env = TradingEnv(symbol=symbol, data=val_data, random_start=False, noise_scale=0.0,
                     reward_mode=reward_mode)

    returns = []
    drawdowns = []
    trades_list = []

    for _ in range(N_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        final_val = info["portfolio_value"]
        ep_return = (final_val - CFG.STARTING_CAPITAL) / CFG.STARTING_CAPITAL
        returns.append(ep_return)
        drawdowns.append(info["drawdown"])
        trades_list.append(info["n_trades"])

    return {
        "mean_return":  float(np.mean(returns)),
        "mean_dd":      float(np.mean(drawdowns)),
        "mean_trades":  float(np.mean(trades_list)),
    }


# ─── Modes ───────────────────────────────────────────────────────────────────
def walkforward(symbol: str, data_start: str, data_end: str, timesteps: int,
                slide: int = SLIDE_MONTHS, reward_mode: str = "pnl"):
    Path(CFG.MODELS_DIR).mkdir(exist_ok=True)
    windows = list(date_windows(data_start, data_end, slide))

    print(f"Walk-forward: {len(windows)} windows | "
          f"{TRAIN_MONTHS}mo train + {VAL_MONTHS}mo val, slide {SLIDE_MONTHS}mo")
    print(f"Timesteps/window: {timesteps:,} | Total ~{timesteps * len(windows):,}\n")

    results = []
    best_sharpe = -np.inf
    best_model_path = None
    previous_model: Optional[MaskablePPO] = None

    for i, (train_start, train_end, val_end) in enumerate(windows):
        prefix = f"[{i+1:3d}/{len(windows)}] {train_start}->{train_end} val->{val_end}"

        try:
            train_data, val_data = load_window(symbol, train_start, train_end, val_end)
        except Exception as e:
            print(f"{prefix} | SKIP (load error: {e})")
            continue

        if len(train_data) < MIN_TRAIN_ROWS:
            print(f"{prefix} | SKIP (train rows={len(train_data)} < {MIN_TRAIN_ROWS})")
            continue
        if len(val_data) < MIN_VAL_ROWS:
            print(f"{prefix} | SKIP (val rows={len(val_data)} < {MIN_VAL_ROWS})")
            continue

        tl_flag = " [TL]" if previous_model is not None else ""
        print(f"{prefix} | train={len(train_data)}rows{tl_flag} ... ", end="", flush=True)
        model = train_on_window(symbol, train_data, timesteps, previous_model,
                                reward_mode, window_idx=i + 1)
        previous_model = model
        metrics = eval_on_window(model, symbol, val_data, reward_mode)

        ret = metrics["mean_return"]
        results.append({"i": i + 1, "val_end": val_end, **metrics})

        flag = " *** BEST ***" if ret > best_sharpe else ""
        print(f"return={ret:+.2%} | DD={metrics['mean_dd']:.1%} | trades={metrics['mean_trades']:.0f}{flag}")

        if ret > best_sharpe:
            best_sharpe = ret
            model_name = symbol.replace("/", "") + "_ppo_rolling_best"
            best_model_path = str(Path(CFG.MODELS_DIR) / model_name)
            model.save(best_model_path)

        # Always overwrite latest model
        model_name_latest = symbol.replace("/", "") + "_ppo_rolling_latest"
        model.save(str(Path(CFG.MODELS_DIR) / model_name_latest))

    if not results:
        print("No windows completed.")
        return

    rets = [r["mean_return"] for r in results]
    mean_r = np.mean(rets)
    std_r  = np.std(rets) + 1e-8
    sharpe = mean_r / std_r

    print(f"\n{'='*60}")
    print(f"Walk-Forward Results ({len(results)} windows)")
    print(f"{'='*60}")
    print(f"Mean return : {mean_r:+.2%}")
    print(f"Std return  : {std_r:.2%}")
    print(f"WF Sharpe   : {sharpe:+.3f}  (target > 1.5)")
    print(f"Win rate    : {sum(r > 0 for r in rets)}/{len(rets)} ({sum(r > 0 for r in rets)/len(rets):.0%})")
    print(f"Best return : {max(rets):+.2%}  (saved to {best_model_path}.zip)")
    print(f"Latest model: {symbol.replace('/','')}_ppo_rolling_latest.zip (most recent window)")


def latest(symbol: str, timesteps: int):
    """Train on the most recent TRAIN_MONTHS of available data."""
    Path(CFG.MODELS_DIR).mkdir(exist_ok=True)

    today = pd.Timestamp("today", tz="UTC").normalize()
    val_end   = today
    train_end = _add_months(today, -VAL_MONTHS)     # last month -> val
    train_start = _add_months(train_end, -TRAIN_MONTHS)

    train_start_s = train_start.strftime("%Y-%m-%d")
    train_end_s   = train_end.strftime("%Y-%m-%d")
    val_end_s     = val_end.strftime("%Y-%m-%d")

    print(f"Latest-window training: {train_start_s} -> {train_end_s} (train), "
          f"{train_end_s} -> {val_end_s} (val)")
    print(f"Timesteps: {timesteps:,} | Symbol: {symbol}\n")

    train_data, val_data = load_window(symbol, train_start_s, train_end_s, val_end_s)
    print(f"Train rows: {len(train_data)} | Val rows: {len(val_data)}")

    model = train_on_window(symbol, train_data, timesteps)
    metrics = eval_on_window(model, symbol, val_data)

    model_name = symbol.replace("/", "") + "_ppo_rolling_latest"
    save_path = str(Path(CFG.MODELS_DIR) / model_name)
    model.save(save_path)

    print(f"\nVal return: {metrics['mean_return']:+.2%} | "
          f"DD: {metrics['mean_dd']:.1%} | Trades: {metrics['mean_trades']:.0f}")
    print(f"Saved: {save_path}.zip")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["walkforward", "latest"], default="walkforward")
    parser.add_argument("--symbol",     default="BTC/USDT")
    parser.add_argument("--data_start", default="2022-01-01",
                        help="Walk-forward start (walkforward mode only)")
    parser.add_argument("--data_end",   default="2026-05-01",
                        help="Walk-forward end (walkforward mode only)")
    parser.add_argument("--timesteps",   type=int, default=TIMESTEPS_PER_WINDOW)
    parser.add_argument("--slide",       type=int, default=SLIDE_MONTHS,
                        help="Months to slide window forward per iteration")
    parser.add_argument("--reward_mode", choices=["pnl", "sharpe"], default="pnl",
                        help="Reward signal: pnl (default) or rolling Sharpe ratio")
    args = parser.parse_args()

    if args.mode == "walkforward":
        walkforward(args.symbol, args.data_start, args.data_end, args.timesteps,
                    args.slide, args.reward_mode)
    else:
        latest(args.symbol, args.timesteps)
