"""Rank videos by their best-matching clip for a text query."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    video: str
    score: float
    start: float
    end: float


class Searcher:
    def __init__(self, vectors: np.ndarray, rows: list[tuple[str, float, float]]):
        self.vectors = vectors
        self.rows = rows

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Result]:
        if not self.rows:
            return []
        scores = self.vectors @ query_vec
        best: dict[str, Result] = {}
        # Highest-scoring clips first, so the first time a video appears is its best moment.
        for i in np.argsort(-scores):
            video, start, end = self.rows[i]
            if video not in best:
                best[video] = Result(video, float(scores[i]), start, end)
                if len(best) == k:
                    break
        return list(best.values())
