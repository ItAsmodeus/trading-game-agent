"""
Метрики для оценки агента.
Правило: смотрим только на OOS (out-of-sample) данные.
In-sample Sharpe > 3.0 — почти наверняка overfitting.
"""

import numpy as np
import pandas as pd
from typing import Optional


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2 or returns.std() < 1e-8:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def calmar(portfolio: np.ndarray, periods_per_year: int = 252) -> float:
    """Calmar = annualized return / max drawdown. Лучше Sharpe для крипты."""
    if len(portfolio) < 2:
        return 0.0
    total_return = portfolio[-1] / portfolio[0] - 1
    ann_return   = (1 + total_return) ** (periods_per_year / len(portfolio)) - 1
    mdd          = max_drawdown(portfolio)
    return float(ann_return / (mdd + 1e-8))


def max_drawdown(portfolio: np.ndarray) -> float:
    """Максимальная просадка от пика до минимума."""
    peak = np.maximum.accumulate(portfolio)
    dd   = (peak - portfolio) / (peak + 1e-8)
    return float(dd.max())


def sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sortino = штрафует только за отрицательную волатильность."""
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std()
    return float(returns.mean() / (downside_std + 1e-8) * np.sqrt(periods_per_year))


def profit_factor(returns: np.ndarray) -> float:
    """sum(wins) / sum(losses). > 1.3 — хорошо."""
    wins   = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return float(wins / (losses + 1e-8))


def win_rate(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def full_report(
    portfolio_history:  list,
    benchmark_history:  Optional[list] = None,
    periods_per_year:   int = 252,
) -> dict:
    """
    Полный отчёт. Вызывай после эпизода.
    portfolio_history: список значений портфеля на каждом шаге.
    benchmark_history: buy & hold для сравнения.
    """
    p       = np.array(portfolio_history, dtype=float)
    returns = np.diff(p) / p[:-1]

    report = {
        "total_return_pct":  float((p[-1] / p[0] - 1) * 100),
        "sharpe":            sharpe(returns, periods_per_year),
        "calmar":            calmar(p, periods_per_year),
        "sortino":           sortino(returns, periods_per_year),
        "max_drawdown_pct":  float(max_drawdown(p) * 100),
        "profit_factor":     profit_factor(returns),
        "win_rate_pct":      float(win_rate(returns) * 100),
        "final_portfolio":   float(p[-1]),
        "n_steps":           len(p) - 1,
    }

    if benchmark_history is not None:
        b     = np.array(benchmark_history, dtype=float)
        b_ret = float((b[-1] / b[0] - 1) * 100)
        report["benchmark_return_pct"] = b_ret
        report["alpha_pct"]            = report["total_return_pct"] - b_ret

    return report


def print_report(report: dict):
    print("\n" + "=" * 50)
    print("  ОТЧЁТ ПО АГЕНТУ")
    print("=" * 50)
    print(f"  Доходность:      {report['total_return_pct']:+.1f}%")
    if "benchmark_return_pct" in report:
        print(f"  Buy & Hold:      {report['benchmark_return_pct']:+.1f}%")
        print(f"  Alpha:           {report['alpha_pct']:+.1f}%")
    print(f"  Sharpe:          {report['sharpe']:.2f}")
    print(f"  Calmar:          {report['calmar']:.2f}")
    print(f"  Sortino:         {report['sortino']:.2f}")
    print(f"  Max Drawdown:    {report['max_drawdown_pct']:.1f}%")
    print(f"  Profit Factor:   {report['profit_factor']:.2f}")
    print(f"  Win Rate:        {report['win_rate_pct']:.1f}%")
    print(f"  Финальный кэп:   ${report['final_portfolio']:,.0f}")
    print("=" * 50)

    # Красные флаги
    flags = []
    if report["sharpe"] > 3.0:
        flags.append("⚠️  Sharpe > 3.0 — вероятный overfitting")
    if report["win_rate_pct"] > 80:
        flags.append("⚠️  Win rate > 80% — проверь look-ahead bias")
    if report["max_drawdown_pct"] > 25:
        flags.append("🚨 Max Drawdown > 25% — агент слишком агрессивен")

    if flags:
        print("\nКРАСНЫЕ ФЛАГИ:")
        for f in flags:
            print(f"  {f}")
