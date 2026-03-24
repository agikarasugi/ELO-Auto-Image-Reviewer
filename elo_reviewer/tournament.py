import random
from dataclasses import dataclass
from pathlib import Path

from . import console as cm
from .elo import EloRatings
from .judge import Judge


@dataclass
class RoundResult:
    round_num: int
    image_a: str
    image_b: str
    winner: str
    loser: str
    used_fallback: bool
    elo_winner_after: float
    elo_loser_after: float
    tokens: int


def sample_pair(image_paths: list[Path]) -> tuple[Path, Path]:
    a, b = random.sample(image_paths, 2)
    if random.random() < 0.5:
        a, b = b, a
    return a, b


def run_tournament(
    image_paths: list[Path],
    ratings: EloRatings,
    judge: Judge,
    rounds: int,
) -> list[RoundResult]:
    results: list[RoundResult] = []
    total_tokens = 0

    for i in range(rounds):
        image_a, image_b = sample_pair(image_paths)

        cm.log(
            f"[dim cyan][[/dim cyan][bold white]{i + 1}/{rounds}[/bold white][dim cyan]][/dim cyan]"
            f" [yellow]{image_a.name}[/yellow]"
            f" [dim]vs[/dim]"
            f" [yellow]{image_b.name}[/yellow]"
        )

        decision, _history, used_fallback, round_tokens = judge.compare(image_a, image_b)
        total_tokens += round_tokens

        if decision == "A":
            winner_path, loser_path = image_a, image_b
        else:
            winner_path, loser_path = image_b, image_a

        elo_w, elo_l = ratings.update(winner_path.name, loser_path.name)
        stats_w = ratings.get(winner_path.name)
        stats_l = ratings.get(loser_path.name)

        fallback_note = "  [bold red][fallback][/bold red]" if used_fallback else ""
        cm.log(
            f"  [green]winner:[/green] [bold green]{winner_path.name}[/bold green]"
            f"  [cyan](elo: {elo_w:.1f})[/cyan]{fallback_note}"
            f"  [dim]tokens: {round_tokens:,} | total: {total_tokens:,}[/dim]"
        )
        cm.log(
            f"  [dim]"
            f"W {winner_path.name}  elo {elo_w:.1f}  {stats_w.wins}W-{stats_w.losses}L ({stats_w.win_rate:.0f}%)"
            f"   |   "
            f"L {loser_path.name}  elo {elo_l:.1f}  {stats_l.wins}W-{stats_l.losses}L ({stats_l.win_rate:.0f}%)"
            f"[/dim]"
        )

        result = RoundResult(
            round_num=i + 1,
            image_a=image_a.name,
            image_b=image_b.name,
            winner=winner_path.name,
            loser=loser_path.name,
            used_fallback=used_fallback,
            elo_winner_after=elo_w,
            elo_loser_after=elo_l,
            tokens=round_tokens,
        )
        results.append(result)

    return results
