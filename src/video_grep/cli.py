"""video-grep: interactive, terminal-only search over a folder of videos.

  video-grep FOLDER

On first run you pick a model size, which is downloaded once. Then new or changed
videos are indexed (deleted ones dropped), and you type what you're looking for and
pick a result number. On a desktop the video opens at the matching moment; over SSH
or on a headless box you get its path, the time range, and a command to save just
that clip.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .models import SIZES

QUICKTIME_EXTS = {".mov", ".mp4", ".m4v"}
# On the test set, specific queries scored 0.67-0.80 on the right video, short vague ones
# ("blue light") ~0.65, and queries with no matching video topped out at 0.613.
# Rough, model-specific cut-off; revisit with real home videos.
MATCH_SCORE = 0.62
OPEN_VIDEOS = True  # set in main(): False when headless or --no-open


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def has_display() -> bool:
    """False over SSH or on headless machines (servers, boards) where a player can't open."""
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def show_details(path: Path, start: float, end: float):
    """Terminal-only result: where the video is, when it happens, and how to save just that part."""
    out = f"{path.stem}_at_{fmt_time(start).replace(':', '-')}{path.suffix}"
    cmd = ["ffmpeg", "-ss", f"{start:g}", "-i", str(path), "-t", f"{end - start:g}", "-c", "copy", out]
    print(f"  File:   {path}")
    print(f"  Moment: {fmt_time(start)} - {fmt_time(end)}")
    print(f"  Save just this part (a few MB instead of the whole video):\n    {shlex.join(cmd)}\n")


def open_video(path: Path, start: float):
    """Open in QuickTime at the matching moment when possible, else the default app."""
    if sys.platform != "darwin":
        try:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Opened in your default player; skip to {fmt_time(start)}.")
        except FileNotFoundError:
            print(f"  {path} (at {fmt_time(start)})")
        return
    if path.suffix.lower() in QUICKTIME_EXTS:
        script = """
            on run argv
                tell application "QuickTime Player"
                    activate
                    set doc to open POSIX file (item 1 of argv)
                    delay 0.5
                    set current time of doc to ((item 2 of argv) as real)
                    play doc
                end tell
            end run"""
        if subprocess.run(["osascript", "-e", script, str(path), str(start)], capture_output=True).returncode == 0:
            return
    subprocess.run(["open", str(path)])
    print(f"  Opened in your default player (QuickTime can't play this format); skip to {fmt_time(start)}.")


def update_index(index, embedder, force: bool):
    status = index.status()
    if status.rebuild:
        print("Model or settings changed since last index; re-indexing everything.")
    todo = len(status.pending) + (status.unchanged if force else 0)
    if not todo and not status.removed:
        return
    if force:
        print(f"Re-indexing all {todo} video(s).")
    else:
        print(f"Updating index: {len(status.new)} new, {len(status.changed)} changed, "
              f"{len(status.removed)} removed video(s).")
    if todo:
        print("This can take a while for big folders. Progress is saved as it goes;\n"
              "press Ctrl+C to stop and run again later to continue.\n")
    t = time.time()
    stats = index.update(embedder, force=force)
    if todo:
        print(f"Indexed {stats['indexed']} video(s) in {time.time() - t:.0f}s.")
    for rel, err in stats["failed"]:
        print(f"  Could not read {rel}: {err}")


def print_results(query: str, results):
    matches = [r for r in results if r.score >= MATCH_SCORE]
    if not matches:
        print(f'  No matches found for "{query}".\n')
        return []
    title = f"Top {len(matches)} matches" if len(matches) > 1 else "1 match"
    print(f'\n  {title} for "{query}":')
    for i, r in enumerate(matches, 1):
        print(f"    {i}. {r.video}  (best moment at {fmt_time(r.start)})")
    print(f"\n  {pick_hint(len(matches))}\n")
    return matches


def pick_hint(n: int) -> str:
    action = "open that video" if OPEN_VIDEOS else "see its full path and a command to save that clip"
    which = "1" if n == 1 else f"a number (1-{n})"
    return f"Type {which} to {action}, or type a new search."


def main():
    global OPEN_VIDEOS
    p = argparse.ArgumentParser(prog="video-grep",
                                description="Search a folder of videos in plain English, from the terminal.")
    p.add_argument("folder", help="folder containing your videos (searched recursively)")
    p.add_argument("-k", type=int, default=5, help="max results to show (default 5)")
    p.add_argument("--reindex", action="store_true", help="re-index every video from scratch")
    p.add_argument("--model-size", choices=list(SIZES),
                   help="switch model size (default: the one chosen on first run). "
                        "Each size keeps its own index, so a new size indexes the folder once.")
    p.add_argument("--no-open", action="store_true",
                   help="never launch a video player; show path, time range and a save command instead "
                        "(automatic over SSH / without a display)")
    args = p.parse_args()
    OPEN_VIDEOS = not args.no_open and has_display()

    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {args.folder}")

    from .embed import load_embedder
    from .index import VideoIndex
    from .models import ensure_model
    from .search import Searcher

    size = ensure_model(args.model_size)
    index = VideoIndex(folder, size.index_id, size.key)
    print(f"Loading {size.title} model ...", flush=True)
    embedder = load_embedder(size)
    try:
        update_index(index, embedder, args.reindex)
    except KeyboardInterrupt:
        sys.exit("\n\nStopped. Videos finished so far are saved; run again to continue indexing.")

    searcher = Searcher(*index.load())
    n = sum(1 for e in index.files.values() if "vectors" in e)
    if not n:
        sys.exit(f"No videos found in {index.folder}")
    print(f"\nReady to search {n} video(s) in {index.folder}")
    print("Type what you're looking for (e.g. \"dog playing in the snow\"), or 'q' to quit.\n")

    results = []
    while True:
        try:
            line = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            break
        if line.isdigit():
            if not results:
                print("  No results to pick from. Type what you're looking for.\n")
                continue
            n = int(line)
            if 1 <= n <= len(results):
                r = results[n - 1]
                if OPEN_VIDEOS:
                    print(f"  Opening {r.video} at {fmt_time(r.start)} ...\n")
                    open_video(index.folder / r.video, r.start)
                else:
                    show_details(index.folder / r.video, r.start, r.end)
            else:
                print(f"  {pick_hint(len(results))}\n")
            continue
        results = print_results(line, searcher.search(embedder.embed_query(line), k=args.k))


if __name__ == "__main__":
    main()
