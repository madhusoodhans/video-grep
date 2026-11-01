"""video-grep: terminal search over what happens inside a folder of videos.

  video-grep FOLDER                     interactive search (indexes first if needed)
  video-grep index FOLDER               build or update the index
  video-grep search FOLDER "query"      one search, then exit
  video-grep clip VIDEO --start 14:36   save part of a video
  video-grep status FOLDER              what's indexed, pending or failed
  video-grep doctor                     check this machine

`search` exits like grep: 0 when something matched, 1 when nothing did, 2 on an error.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from .models import SIZES

COMMANDS = {"shell", "index", "search", "clip", "status", "doctor"}
QUICKTIME_EXTS = {".mov", ".mp4", ".m4v"}
# Measured with the High model on 29 varied clips, 34 queries for things present and 24 for
# things absent. At 0.67 every present query matched and 18 of 24 absent ones were rejected;
# at 0.70 all 24 absent ones were rejected but 7 present ones were lost. Scores shift by up
# to ~0.014 on Small. These are starting points: re-measure on real footage.
POSSIBLE_SCORE = 0.67
STRONG_SCORE = 0.70
CLIP_SECONDS = 6.0
OPEN_VIDEOS = True  # set for the shell: False when headless or --no-open


def fmt_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_length(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f} seconds"
    if sec < 5400:
        return f"{sec / 60:.0f} minutes"
    return f"{sec / 3600:.1f} hours"


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def has_display() -> bool:
    """False over SSH or on headless machines (servers, boards) where a player can't open."""
    if os.environ.get("SSH_CONNECTION") and not os.environ.get("DISPLAY"):
        return False
    if sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


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


def require_ffmpeg():
    from .frames import missing_tools

    if missing := missing_tools():
        sys.exit(f"{' and '.join(missing)} not found. Install ffmpeg first "
                 "(macOS: brew install ffmpeg, Ubuntu/Debian: sudo apt install ffmpeg).")


def video_folder(args) -> Path:
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {args.folder}")
    return folder


def index_root(folder: Path, override: str | None) -> Path | None:
    """Where the index lives. None means the default, inside the video folder."""
    if override:
        return Path(override).expanduser()
    if os.access(folder, os.W_OK):
        return None
    from .models import HOME

    root = HOME / "indexes" / hashlib.sha1(str(folder.resolve()).encode()).hexdigest()[:16]
    print(f"{folder} is read-only, so the index is kept in {root}", file=sys.stderr)
    return root


def open_index(args, size):
    from .index import VideoIndex

    folder = video_folder(args)
    return VideoIndex(folder, size.index_id, size.key, index_root(folder, args.index_dir))


def load_model(size):
    from .embed import load_embedder

    print(f"Loading {size.title} model ...", file=sys.stderr, flush=True)
    embedder = load_embedder(size)
    device = {"mps": "Apple GPU", "cuda": "NVIDIA GPU"}.get(embedder.device)
    print(f"Running on {device}." if device else "Running on CPU, so indexing will be slow.", file=sys.stderr)
    return embedder


def report_failed(index):
    failed = {rel: entry["error"] for rel, entry in index.files.items() if "error" in entry}
    if failed:
        print(f"\n{len(failed)} video(s) could not be read and aren't searchable (--retry-failed tries again):")
        for rel, err in list(failed.items())[:5]:
            print(f"  {rel}: {err}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")


def update_index(index, get_embedder, args):
    """Bring the index up to date, showing what will happen first."""
    damaged = index.verify()
    if damaged:
        print(f"{damaged} video(s) had damaged index data and will be indexed again.")
    status = index.status()
    if status.rebuild:
        print("Model or settings changed since last index; re-indexing everything.")
    todo = index.plan(status, args.reindex, args.retry_failed)
    if todo or status.removed:
        if args.reindex:
            print(f"Re-indexing all {len(todo)} video(s).")
        else:
            print(f"Updating index: {len(status.new)} new, {len(status.changed)} changed, "
                  f"{len(status.partial)} unfinished, {len(todo) - len(status.pending)} to retry, "
                  f"{len(status.removed)} removed video(s).")
        durations = index.probe(todo)
        if todo:
            announce_work(index, durations)
        t = time.time()
        stats = index.update(get_embedder(), force=args.reindex, retry_failed=args.retry_failed,
                             prune=args.prune, durations=durations)
        if todo:
            print(f"Indexed {stats['indexed']} video(s) in {fmt_length(time.time() - t)}.")
        for rel, warning in stats["warnings"]:
            print(f"  Warning for {rel}: {warning}")
        if stats["kept_missing"]:
            print(f"  {stats['kept_missing']} indexed video(s) are missing. Their index data was kept in case a\n"
                  "  disk is unmounted. If they were really deleted, run again with --prune.")
    report_failed(index)


def announce_work(index, durations: dict):
    from .index import INDEX_BYTES_PER_SEC

    footage = sum(d or 0 for d in durations.values())
    needed = footage * INDEX_BYTES_PER_SEC
    free = shutil.disk_usage(nearest_dir(index.dir)).free
    print(f"To do: {len(durations)} video(s), {fmt_length(footage)} of footage. "
          f"The index will need about {fmt_bytes(needed)} ({fmt_bytes(free)} free).")
    if needed > free:
        sys.exit("Not enough disk space for the index. Free some up, or use --index-dir.")
    print("The time left appears once the first clips are done. Progress is saved as it goes; press\n"
          "Ctrl+C to stop and run again later to continue. Over SSH, use `video-grep index --background`.\n")


def nearest_dir(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def date_arg(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError("use the form YYYY-MM-DD") from None


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return n


def time_arg(text: str) -> float:
    from .frames import parse_time

    try:
        return parse_time(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def filtered_videos(index, args) -> set[str] | None:
    if not (args.path or args.after or args.before):
        return None
    chosen = index.select(args.path, args.after, args.before)
    if not chosen:
        sys.exit("No indexed videos match those filters.")
    return chosen


def describe(result) -> str:
    strength = "strong" if result.score >= STRONG_SCORE else "possible"
    return f"{result.video}  (at {fmt_time(result.start)}, {strength} match, {result.score:.2f})"


def matches_in(results, min_score: float) -> list:
    return [r for r in results if r.score >= min_score]


def print_matches(query: str, matches: list):
    title = f"Top {len(matches)} matches" if len(matches) > 1 else "1 match"
    print(f'\n  {title} for "{query}":')
    for i, r in enumerate(matches, 1):
        print(f"    {i}. {describe(r)}")


def no_matches(query: str):
    print(f'  No matches found for "{query}". Try describing it in more detail, or lower --min-score.\n')


def pick_hint(n: int) -> str:
    action = "open that video" if OPEN_VIDEOS else "see its full path"
    which = "1" if n == 1 else f"a number (1-{n})"
    example = "1" if n == 1 else "<number>"
    return f'Type {which} to {action}, "save {example}" to save that clip, or type a new search.'


def show_result(index, number: int, result):
    path = index.folder / result.video
    if OPEN_VIDEOS:
        print(f"  Opening {result.video} at {fmt_time(result.start)} ...")
        open_video(path, result.start)
        print(f"  File: {path}\n")
    else:
        print(f"  File:   {path}")
        print(f"  Moment: {fmt_time(result.start)} - {fmt_time(result.end)}")
        print(f'  Type "save {number}" to save just this part, or "save {number} exact" for a precise cut.\n')


def save_result(index, result, exact: bool, out_dir: str | None):
    from .frames import VideoError, clip_filename, save_clip

    path = index.folder / result.video
    folder = Path(out_dir).expanduser() if out_dir else Path.cwd()
    try:
        folder.mkdir(parents=True, exist_ok=True)
        out = save_clip(path, result.start, result.end, clip_filename(path, result.start, folder), exact)
    except (VideoError, OSError) as e:
        print(f'  Could not save the clip: {e}{"" if exact else ". Try save <number> exact."}\n')
        return
    print(f"  Saved {out} ({fmt_bytes(out.stat().st_size)})\n")


def cmd_shell(args):
    global OPEN_VIDEOS
    OPEN_VIDEOS = not args.no_open and has_display()
    require_ffmpeg()

    from .models import ensure_model
    from .search import Searcher

    video_folder(args)
    size = ensure_model(args.model_size)
    index = open_index(args, size)
    embedder = load_model(size)
    if index.is_indexing():
        print("Indexing is running in the background; searching what's finished so far.")
    else:
        try:
            update_index(index, lambda: embedder, args)
        except KeyboardInterrupt:
            sys.exit("\n\nStopped. Finished work is saved; run again to continue indexing.")
        except (OSError, RuntimeError) as e:
            sys.exit(f"\n\nIndexing stopped: {e}\nNo video was marked as failed. Fix the problem and "
                     "run again to continue.\nIf the index location can't be written to, use --index-dir.")

    searcher = Searcher(*index.load(filtered_videos(index, args)))
    count = len({row[0] for row in searcher.rows})
    if not count:
        sys.exit(f"No searchable videos in {index.folder}")
    print(f"\nReady to search {count} video(s) in {index.folder}")
    print("Type what you're looking for (e.g. \"dog playing in the snow\"), or 'q' to quit.\n")
    shell_loop(index, searcher, embedder, args)


def shell_loop(index, searcher, embedder, args):
    try:
        import readline  # noqa: F401  (gives input() arrow keys and history)
    except ImportError:
        pass
    results = []
    while True:
        try:
            line = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit"}:
            return
        if not re.fullmatch(r"\d+|save\s+\d+(\s+exact)?", line.lower()):
            found = searcher.search(embedder.embed_query(line), k=args.k, per_video=args.per_video)
            results = matches_in(found, args.min_score)
            if results:
                print_matches(line, results)
                print(f"\n  {pick_hint(len(results))}\n")
            else:
                no_matches(line)
            continue
        if not results:
            print("  No results to pick from. Type what you're looking for.\n")
            continue
        words = line.lower().split()
        number = int(words[1] if words[0] == "save" else words[0])
        if not 1 <= number <= len(results):
            print(f"  {pick_hint(len(results))}\n")
        elif words[0] == "save":
            save_result(index, results[number - 1], words[-1] == "exact", args.out_dir)
        else:
            show_result(index, number, results[number - 1])


def start_background(args, size, index):
    index.dir.mkdir(parents=True, exist_ok=True)
    log = index.dir / "index.log"
    cmd = [sys.executable, "-m", "video_grep", "index", str(index.folder), "--model-size", size.key]
    if args.index_dir:
        cmd += ["--index-dir", args.index_dir]
    cmd += [flag for flag, on in (("--reindex", args.reindex), ("--retry-failed", args.retry_failed),
                                  ("--prune", args.prune)) if on]
    with open(log, "w") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    print(f"Indexing in the background (process {proc.pid}). It keeps going if you disconnect.")
    print(f"  Check on it:  video-grep status {index.folder}")
    print(f"  Watch it:     tail -f {log}")
    print(f"  Stop it:      kill {proc.pid}   (finished work is kept)")


def cmd_index(args):
    require_ffmpeg()

    from .models import ensure_model

    video_folder(args)
    size = ensure_model(args.model_size)
    index = open_index(args, size)
    if index.is_indexing():
        sys.exit(f"This folder is already being indexed. See: video-grep status {index.folder}")
    if args.background:
        start_background(args, size, index)
        return
    embedder = None

    def get_embedder():
        nonlocal embedder
        embedder = embedder or load_model(size)
        return embedder

    try:
        update_index(index, get_embedder, args)
    except KeyboardInterrupt:
        sys.exit("\n\nStopped. Finished work is saved; run again to continue indexing.")
    except (OSError, RuntimeError) as e:
        sys.exit(f"\n\nIndexing stopped: {e}\nNo video was marked as failed. Fix the problem and "
                 "run again to continue.\nIf the index location can't be written to, use --index-dir.")
    print("Index is up to date.")


def cmd_search(args) -> int:
    from .models import ensure_model
    from .search import Searcher

    video_folder(args)
    size = ensure_model(args.model_size)
    index = open_index(args, size)
    if not any("vectors" in entry for entry in index.files.values()):
        if index.is_indexing():
            sys.exit(f"Indexing has started but nothing is finished yet. See: video-grep status {index.folder}")
        sys.exit(f"This folder isn't indexed yet. Run: video-grep index {index.folder}")
    status = index.status()
    if status.rebuild:
        sys.exit(f"The index was built with different settings. Run: video-grep index {index.folder}")
    if status.pending and not index.is_indexing():
        print(f"Note: {len(status.pending)} video(s) aren't indexed yet, so they aren't searched.", file=sys.stderr)
    searcher = Searcher(*index.load(filtered_videos(index, args)))
    embedder = load_model(size)
    found = searcher.search(embedder.embed_query(args.query), k=args.k, per_video=args.per_video)
    matches = matches_in(found, args.min_score)
    if not matches:
        print(f'No matches found for "{args.query}".', file=sys.stderr)
        return 1
    for r in matches:
        print(describe(r))
    return 0


def cmd_clip(args):
    from .frames import VideoError, clip_filename, probe_duration, save_clip

    require_ffmpeg()
    path = Path(args.video).expanduser()
    if not path.is_file():
        sys.exit(f"No such video: {args.video}")
    end = args.end if args.end is not None else args.start + CLIP_SECONDS
    try:
        duration = probe_duration(path)
        if duration and args.start >= duration:
            sys.exit(f"The start time is past the end of the video ({fmt_time(duration)}).")
        out = Path(args.output).expanduser() if args.output else clip_filename(path, args.start, Path.cwd())
        if out.exists():
            sys.exit(f"Already exists: {out}")
        save_clip(path, args.start, end, out, args.exact)
    except VideoError as e:
        sys.exit(f"Could not save the clip: {e}")
    print(f"Saved {out} ({fmt_bytes(out.stat().st_size)})")


def cmd_status(args):
    from .models import DEFAULT_SIZE, saved_size

    video_folder(args)
    size = SIZES[args.model_size or saved_size() or DEFAULT_SIZE]
    index = open_index(args, size)
    status = index.status()
    entries = [e for e in index.files.values() if "vectors" in e]
    footage = sum(e.get("duration", 0) for e in entries)
    stored = sum(f.stat().st_size for f in index.dir.rglob("*") if f.is_file()) if index.dir.exists() else 0
    print(f"Folder:   {index.folder}")
    print(f"Model:    {size.title}" + ("" if size.is_downloaded() else " (not downloaded yet)"))
    print(f"Index:    {index.dir} ({fmt_bytes(stored)})")
    print(f"Indexed:  {status.unchanged} video(s), {fmt_length(footage)} of footage searchable")
    if status.rebuild:
        print("Note:     built with different settings, so everything needs indexing again")
    waiting = [(len(status.new), "new"), (len(status.changed), "changed"), (len(status.partial), "unfinished"),
               (len(status.removed), "missing")]
    print("Waiting:  " + (", ".join(f"{n} {label}" for n, label in waiting if n) or "nothing"))
    if index.is_indexing():
        print(f"Running:  indexing now (log: {index.dir / 'index.log'})")
    elif status.pending:
        print(f"Next:     video-grep index {index.folder}")
    report_failed(index)


def cmd_doctor(args) -> int:
    from .frames import missing_tools
    from .models import DEFAULT_SIZE, HOME, has_gpu, ram_gb, recommended_size

    problems = 0

    def check(ok: bool | None, text: str):
        nonlocal problems
        problems += ok is False
        print(f"  {'OK  ' if ok else 'FAIL' if ok is False else 'NOTE'}  {text}")

    version = sys.version_info[:3]
    check((3, 12) <= version[:2] <= (3, 14) and version != (3, 14, 1), f"Python {'.'.join(map(str, version))}")
    if missing := missing_tools():
        check(False, f"{' and '.join(missing)} not found")
    else:
        line = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
        check(True, " ".join(line.split()[:3]))
    gpu, ram = has_gpu(), ram_gb()
    check(True if gpu else None, "GPU found" if gpu else "No usable GPU: indexing runs on the CPU and will be slow")
    check(True, f"{ram:.0f} GB RAM")
    free = shutil.disk_usage(nearest_dir(HOME)).free
    check(free > 2e9, f"{fmt_bytes(free)} free for models")
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        check(os.access(HOME, os.W_OK), f"Can write to {HOME}")
    except OSError as e:
        check(False, f"Can't write to {HOME}: {e}")
    downloaded = [s.title for s in SIZES.values() if s.is_downloaded()]
    check(True if downloaded else None, f"Models downloaded: {', '.join(downloaded) or 'none yet'}")
    best = recommended_size(gpu, ram)
    print(f"\n  Recommended model size for this machine: {SIZES[best].title}"
          + ("" if best == DEFAULT_SIZE else " (High needs about 8 GB RAM without a GPU)"))
    print("\n  Ready." if not problems else f"\n  {problems} problem(s) to fix first.")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="video-grep",
        description="Search what happens inside a folder of videos, from the terminal.",
        epilog="Run `video-grep FOLDER` for interactive search, or `video-grep COMMAND --help` for details.")
    commands = p.add_subparsers(dest="command", required=True)

    target = argparse.ArgumentParser(add_help=False)
    target.add_argument("folder", help="folder containing your videos (searched recursively)")
    target.add_argument("--model-size", choices=list(SIZES),
                        help="use this model size for this run (default: the one chosen on first run). "
                             "Each size keeps its own index, so a new size indexes the folder once.")
    target.add_argument("--index-dir", help="keep the index here instead of inside the video folder "
                                            "(automatic if the folder is read-only)")
    updating = argparse.ArgumentParser(add_help=False)
    updating.add_argument("--reindex", action="store_true", help="re-index every video from scratch")
    updating.add_argument("--retry-failed", action="store_true", help="try again on videos that could not be read")
    updating.add_argument("--prune", action="store_true",
                          help="drop index data for missing videos, even if many are missing")
    results = argparse.ArgumentParser(add_help=False)
    results.add_argument("-k", type=positive_int, default=5, help="max results to show (default 5)")
    results.add_argument("--per-video", type=positive_int, default=3,
                         help="max separate moments to show from one video (default 3)")
    results.add_argument("--min-score", type=float, default=POSSIBLE_SCORE,
                         help=f"hide results scoring below this (default {POSSIBLE_SCORE}). Lower it to see more "
                              "uncertain results; raise it to reduce false matches")
    results.add_argument("--path", help='only search this subfolder or file pattern, e.g. cam2 or "*.mp4"')
    results.add_argument("--after", type=date_arg, metavar="YYYY-MM-DD",
                         help="only videos last modified on or after this date")
    results.add_argument("--before", type=date_arg, metavar="YYYY-MM-DD",
                         help="only videos last modified on or before this date")

    s = commands.add_parser("shell", parents=[target, updating, results], help="interactive search (the default)")
    s.add_argument("--no-open", action="store_true",
                   help="never launch a video player; show the path and time range instead "
                        "(automatic over SSH / without a display)")
    s.add_argument("--out-dir", help="where `save` puts clips (default: the current folder)")
    s.set_defaults(func=cmd_shell)

    s = commands.add_parser("index", parents=[target, updating], help="build or update the index")
    s.add_argument("--background", action="store_true",
                   help="keep indexing after you disconnect; check on it with `video-grep status`")
    s.set_defaults(func=cmd_index)

    s = commands.add_parser("search", parents=[target, results], help="one search, then exit")
    s.add_argument("query", help="what happens in the moment you're looking for")
    s.set_defaults(func=cmd_search)

    s = commands.add_parser("clip", help="save part of a video")
    s.add_argument("video", help="the video file")
    s.add_argument("--start", type=time_arg, required=True, help="start time: 876, 14:36 or 1:02:03")
    s.add_argument("--end", type=time_arg, help=f"end time (default: {CLIP_SECONDS:g} seconds after the start)")
    s.add_argument("-o", "--output", help="output file (default: a new file in the current folder)")
    s.add_argument("--exact", action="store_true", help="precise cut, slower and without audio")
    s.set_defaults(func=cmd_clip)

    s = commands.add_parser("status", parents=[target], help="what's indexed, pending or failed")
    s.set_defaults(func=cmd_status)

    s = commands.add_parser("doctor", help="check that this machine is ready")
    s.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None):
    argv = sys.argv[1:] if argv is None else argv
    # `video-grep FOLDER` is the shell. A folder really named after a command needs a path: ./index
    if argv and argv[0] not in COMMANDS and argv[0] not in {"-h", "--help"}:
        argv = ["shell", *argv]
    args = build_parser().parse_args(argv)
    try:
        sys.exit(args.func(args) or 0)
    except SystemExit as e:
        if isinstance(e.code, str):  # error messages exit with 2, as grep does
            print(e.code, file=sys.stderr)
            sys.exit(2)
        raise


if __name__ == "__main__":
    main()
