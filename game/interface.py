"""
Game Interface — визуальный интерфейс игры в терминале.

Показывает:
  - Текущее состояние агента (капитал, позиция, прибыль)
  - Прогресс к следующему милстоуну
  - Шкала успешности (Score)
  - Достижения
  - Live метрики обучения

Использует только стандартную библиотеку + опционально rich.
"""

import os
import sys
from typing import Optional
from game.milestones import MilestoneTracker, Milestone


def _bar(filled: float, width: int = 20, fill="█", empty="░") -> str:
    """Прогресс-бар: filled=0.0..1.0"""
    n = int(filled * width)
    return fill * n + empty * (width - n)


def _capital_color(pct: float) -> str:
    """Цвет для капитала (ANSI если поддерживается)."""
    if pct > 0:  return "\033[92m"   # зелёный
    if pct < 0:  return "\033[91m"   # красный
    return "\033[0m"


RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[96m"
GOLD  = "\033[93m"
GREEN = "\033[92m"
RED   = "\033[91m"
DIM   = "\033[2m"


class GameInterface:
    """
    Терминальный интерфейс игры. Обновляется на каждом шаге.
    """

    def __init__(
        self,
        agent_name:       str   = "Agent Alpha",
        starting_capital: float = 10_000.0,
        mode:             str   = "swing",
        use_color:        bool  = True,
    ):
        self.agent_name       = agent_name
        self.starting_capital = starting_capital
        self.mode             = mode.upper()
        self.use_color        = use_color and sys.stdout.isatty()
        self.tracker          = MilestoneTracker(starting_capital)
        self._step            = 0

    def _c(self, code: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{RESET}"

    def render(
        self,
        portfolio:    float,
        position:     float,
        trade_count:  int,
        sharpe:       float,
        drawdown:     float,
        step:         int,
        action_name:  str = "HOLD",
        date:         str = "",
    ):
        """Выводит текущее состояние игры в терминал."""
        self._step = step

        # Обновляем трекер
        bonus = self.tracker.update(portfolio, trade_count, sharpe, step)

        # Основные метрики
        pct_change  = (portfolio / self.starting_capital - 1)
        next_ms     = self.tracker.next_milestone()
        progress    = self.tracker.progress_to_next(portfolio)
        score       = self.tracker.total_score

        # Цвета
        pct_color = GREEN if pct_change >= 0 else RED
        pos_str   = {1.0: "🟢 LONG", -1.0: "🔴 SHORT", 0.5: "🟡 LONG 50%",
                     -0.5: "🟠 SHORT 50%"}.get(position, "⬜ FLAT")

        # Строим интерфейс
        width = 56
        lines = []

        lines.append(self._c(BOLD + CYAN, "╔" + "═" * width + "╗"))
        lines.append(self._c(BOLD + CYAN, "║") +
                     self._c(BOLD, f"  🎮 TRADING GAME — {self.agent_name:<20}") +
                     self._c(DIM, f" [{self.mode}]  ") +
                     self._c(BOLD + CYAN, "║"))
        lines.append(self._c(CYAN, "╠" + "═" * width + "╣"))

        # Капитал
        cap_str = f"${portfolio:>10,.0f}"
        pct_str = f"{pct_change:+.1%}"
        lines.append(self._c(CYAN, "║") +
                     f"  💰 Капитал: " + self._c(BOLD, cap_str) +
                     "  " + self._c(pct_color, pct_str) +
                     " " * max(0, width - 28 - len(pct_str)) +
                     self._c(CYAN, "║"))

        # Прогресс к следующему уровню
        if next_ms:
            bar = _bar(progress, 22)
            ms_line = f"  {next_ms.emoji} {next_ms.name[:15]:<15} {self._c(GOLD, bar)} {progress:.0%}"
            pad = max(0, width - len(f"  {next_ms.emoji} {next_ms.name[:15]:<15}  {bar} {progress:.0%}") + 9)
            lines.append(self._c(CYAN, "║") + ms_line + " " * pad + self._c(CYAN, "║"))
        else:
            lines.append(self._c(CYAN, "║") + self._c(GOLD, "  👑 ВСЕ ЦЕЛИ ДОСТИГНУТЫ!") +
                         " " * (width - 24) + self._c(CYAN, "║"))

        lines.append(self._c(CYAN, "╠" + "─" * width + "╣"))

        # Метрики
        sharpe_color = GREEN if sharpe > 1.0 else (GOLD if sharpe > 0 else RED)
        dd_color     = GREEN if drawdown < 0.05 else (GOLD if drawdown < 0.15 else RED)

        lines.append(self._c(CYAN, "║") +
                     f"  📊 Sharpe: " + self._c(sharpe_color, f"{sharpe:+.2f}") +
                     f"  📉 DD: " + self._c(dd_color, f"{drawdown:.1%}") +
                     f"  🎯 Score: " + self._c(GOLD, f"{score:,}") +
                     " " * max(0, width - 44) +
                     self._c(CYAN, "║"))

        lines.append(self._c(CYAN, "║") +
                     f"  {pos_str:<14}  🔄 Сделок: {trade_count:<4}" +
                     f"  📅 {date[:10]}" +
                     " " * max(0, width - 42) +
                     self._c(CYAN, "║"))

        # Последнее действие
        lines.append(self._c(CYAN, "║") +
                     f"  Действие: " + self._c(BOLD, f"{action_name:<20}") +
                     f"  Шаг: {step:<6}" +
                     " " * max(0, width - 40) +
                     self._c(CYAN, "║"))

        # Достижения
        achieved = self.tracker.achieved_list()
        if achieved:
            lines.append(self._c(CYAN, "╠" + "─" * width + "╣"))
            lines.append(self._c(CYAN, "║") +
                         self._c(BOLD, "  🏆 Достижения: ") +
                         " " * (width - 17) +
                         self._c(CYAN, "║"))
            # По 3 в строке
            row = []
            for m in achieved[-6:]:
                row.append(f"{m.emoji} {m.name[:12]}")
                if len(row) == 3:
                    line = "  " + "  ".join(f"{r:<15}" for r in row)
                    lines.append(self._c(CYAN, "║") + line[:width] +
                                 " " * max(0, width - len(line)) + self._c(CYAN, "║"))
                    row = []
            if row:
                line = "  " + "  ".join(f"{r:<15}" for r in row)
                lines.append(self._c(CYAN, "║") + line[:width] +
                             " " * max(0, width - len(line)) + self._c(CYAN, "║"))

        # Новое достижение!
        if bonus > 0 and achieved:
            last = achieved[-1]
            lines.append(self._c(CYAN, "╠" + "─" * width + "╣"))
            new_line = f"  🎉 НОВОЕ ДОСТИЖЕНИЕ: {last.emoji} {last.name}  +{last.score_pts} очков!"
            lines.append(self._c(CYAN, "║") + self._c(GOLD + BOLD, new_line) +
                         " " * max(0, width - len(new_line) + 9) + self._c(CYAN, "║"))

        lines.append(self._c(BOLD + CYAN, "╚" + "═" * width + "╝"))

        # Выводим (очищаем предыдущий кадр)
        if step > 0:
            n_lines = len(lines)
            sys.stdout.write(f"\033[{n_lines}A\033[J")  # move up + clear
        print("\n".join(lines))
        sys.stdout.flush()


def render_final_score(tracker: MilestoneTracker, portfolio: float, starting: float):
    """Финальный экран после завершения эпизода."""
    pct = (portfolio / starting - 1)
    achieved = tracker.achieved_list()

    print("\n" + "═" * 58)
    print(f"  🏁 ИГРА ЗАВЕРШЕНА")
    print("═" * 58)
    print(f"  Финальный капитал: ${portfolio:,.0f}  ({pct:+.1%})")
    print(f"  Итоговый Score:    {tracker.total_score:,} очков")
    print(f"  Достижений:        {len(achieved)}/{len(tracker.milestones)}")
    print(f"  Серия побед:       {tracker.win_streak} 🔥")
    print()
    if achieved:
        print("  Заработанные достижения:")
        for m in achieved:
            print(f"    {m.emoji} {m.name:<22} +{m.score_pts} pts  (шаг {m.achieved_at})")
    else:
        print("  Достижений не получено. Продолжай учиться! 💪")
    print("═" * 58 + "\n")
