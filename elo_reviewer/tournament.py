import random
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

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
    verbose: bool = False,
) -> list[RoundResult]:
    results: list[RoundResult] = []

    with tqdm(total=rounds, desc="Tournament", unit="round") as pbar:
        for i in range(rounds):
            image_a, image_b = sample_pair(image_paths)

            if verbose:
                print(f"\nRound {i + 1}: {image_a.name}  vs  {image_b.name}")

            decision, _history, used_fallback = judge.compare(image_a, image_b)

            if decision == "A":
                winner_path, loser_path = image_a, image_b
            else:
                winner_path, loser_path = image_b, image_a

            elo_w, elo_l = ratings.update(winner_path.name, loser_path.name)

            result = RoundResult(
                round_num=i + 1,
                image_a=image_a.name,
                image_b=image_b.name,
                winner=winner_path.name,
                loser=loser_path.name,
                used_fallback=used_fallback,
                elo_winner_after=elo_w,
                elo_loser_after=elo_l,
            )
            results.append(result)

            postfix: dict = {"winner": winner_path.name[:18]}
            if used_fallback:
                postfix["fallback"] = "!"
            pbar.set_postfix(postfix)
            pbar.update(1)

    return results
