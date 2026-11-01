"""Rank moments in videos for a text query."""

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

    def search(self, query_vec: np.ndarray, k: int = 5, per_video: int = 1) -> list[Result]:
        """Best-scoring moments, at most per_video from one file.

        A clip touching one already chosen is the same moment, so it is skipped.
        """
        if not self.rows:
            return []
        scores = self.vectors @ query_vec
        chosen: list[Result] = []
        by_video: dict[str, list[Result]] = {}
        for i in np.argsort(-scores):
            video, start, end = self.rows[i]
            taken = by_video.setdefault(video, [])
            if len(taken) >= per_video or any(start <= r.end and end >= r.start for r in taken):
                continue
            result = Result(video, float(scores[i]), start, end)
            taken.append(result)
            chosen.append(result)
            if len(chosen) == k:
                break
        return chosen
