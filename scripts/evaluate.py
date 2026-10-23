"""Measure search quality on an indexed folder against hand-labelled queries.

Usage: python scripts/evaluate.py FOLDER [--queries scripts/eval_queries.json] [--model-size high|medium|small]

Metrics, per query, using the first relevant video in the ranking:
  hit@1  top result is relevant
  hit@5  a relevant video is in the top 5
  MRR    mean of 1/rank
"""

import argparse
import json
from pathlib import Path

from video_grep.embed import load_embedder
from video_grep.index import VideoIndex
from video_grep.models import SIZES
from video_grep.search import Searcher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--queries", default=str(Path(__file__).with_name("eval_queries.json")))
    p.add_argument("--model-size", choices=list(SIZES), default="high")
    args = p.parse_args()

    size = SIZES[args.model_size]
    queries = json.loads(Path(args.queries).read_text())["queries"]
    index = VideoIndex(Path(args.folder), size.index_id, size.key)
    if not size.is_downloaded() or not index.files or index.status().rebuild:
        raise SystemExit(f"Run `video-grep {args.folder} --model-size {size.key}` first to download and index.")
    searcher = Searcher(*index.load())
    n_videos = len({row[0] for row in searcher.rows})
    embedder = load_embedder(size)

    hit1 = hit5 = rr = 0.0
    print(f"{'query':45} rank  top result")
    for q, relevant in queries.items():
        results = searcher.search(embedder.embed_query(q), k=n_videos)
        names = [Path(r.video).stem for r in results]
        rank = min((names.index(r) + 1 for r in relevant if r in names), default=None)
        hit1 += rank == 1
        hit5 += rank is not None and rank <= 5
        rr += 1 / rank if rank else 0
        mark = "ok " if rank == 1 else ("~  " if rank and rank <= 5 else "MISS")
        print(f"{q[:45]:45} {rank or '-':>4}  {mark} {names[0]} ({results[0].score:.3f})")

    n = len(queries)
    print(f"\n{n} queries over {n_videos} videos  |  hit@1 {hit1 / n:.0%}  hit@5 {hit5 / n:.0%}  MRR {rr / n:.3f}")


if __name__ == "__main__":
    main()
