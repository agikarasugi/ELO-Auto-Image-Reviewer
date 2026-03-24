from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PlayerStats:
    filename: str
    elo: float
    wins: int = 0
    losses: int = 0

    @property
    def total_games(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games * 100.0


class EloRatings:
    def __init__(
        self,
        image_paths: list[Path],
        starting_elo: float = 1000.0,
        k_factor: float = 32.0,
    ) -> None:
        self.k_factor = k_factor
        self._ratings: dict[str, PlayerStats] = {
            p.name: PlayerStats(filename=p.name, elo=starting_elo)
            for p in image_paths
        }

    def get(self, filename: str) -> PlayerStats:
        return self._ratings[filename]

    def get_all(self) -> list[PlayerStats]:
        return list(self._ratings.values())

    def ranked(self) -> list[PlayerStats]:
        return sorted(self._ratings.values(), key=lambda s: s.elo, reverse=True)

    def update(self, winner: str, loser: str) -> tuple[float, float]:
        w = self._ratings[winner]
        lo = self._ratings[loser]

        e_w = 1.0 / (1.0 + 10.0 ** ((lo.elo - w.elo) / 400.0))
        e_l = 1.0 - e_w

        w.elo += self.k_factor * (1.0 - e_w)
        lo.elo += self.k_factor * (0.0 - e_l)

        w.wins += 1
        lo.losses += 1

        return w.elo, lo.elo
