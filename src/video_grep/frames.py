"""Find videos and turn them into short clips of sampled frames, using ffmpeg."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".mts", ".m2ts", ".ogv", ".wmv"}


@dataclass
class Clip:
    start: float
    end: float
    frames: np.ndarray  # (n, size, size, 3) uint8 RGB


def find_videos(folder: Path) -> list[Path]:
    """All video files under folder, skipping hidden files and directories."""
    return sorted(
        p for p in folder.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS
        and p.is_file()
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
    )


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, stdin=subprocess.DEVNULL,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def iter_clips(path: Path, window_sec: float, frames_per_clip: int, size: int, duration: float | None = None):
    """Yield consecutive, non-overlapping clips covering the whole video.

    One ffmpeg process decodes the video at frames_per_clip / window_sec fps and
    streams square, letterboxed RGB frames; every frames_per_clip frames become a clip.
    Videos shorter than one window become a single clip spread over their length.
    """
    duration = duration or probe_duration(path)
    window = min(window_sec, duration)
    fps = frames_per_clip / window
    vf = (f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
          f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2")
    # -nostdin: otherwise ffmpeg reads the terminal and eats keys typed during indexing.
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf", vf,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = size * size * 3
    # stderr is discarded: an unread pipe can fill up and stall ffmpeg on damaged files.
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        buf, start, emitted = [], 0.0, 0
        while raw := proc.stdout.read(frame_bytes):
            if len(raw) < frame_bytes:
                break
            buf.append(np.frombuffer(raw, np.uint8).reshape(size, size, 3))
            if len(buf) == frames_per_clip:
                end = min(start + window, duration)
                yield Clip(start, end, np.stack(buf))
                buf, start, emitted = [], end, emitted + 1
        # A tail of at least a quarter clip is worth keeping; shorter tails are noise.
        if buf and (len(buf) >= max(1, frames_per_clip // 4) or not emitted):
            yield Clip(start, duration, np.stack(buf))
        elif not emitted:
            raise RuntimeError("ffmpeg could not decode any frames")
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
