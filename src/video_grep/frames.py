"""Find videos and turn them into short clips of sampled frames, using ffmpeg."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VIDEO_EXTS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".3gp", ".ogv", ".wmv", ".flv", ".f4v", ".asf",
    ".mpg", ".mpeg", ".vob", ".ts", ".mts", ".m2ts", ".mxf",
    ".dav", ".h264", ".264", ".h265", ".265", ".hevc",  # common DVR / CCTV exports
}


class VideoError(Exception):
    """This file can't be read. Machine problems (disk full, out of memory) are not VideoErrors."""


def missing_tools() -> list[str]:
    return [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]


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


def probe_duration(path: Path) -> float | None:
    """Length in seconds, or None if the file doesn't record it (live recordings, raw streams)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL,
    )
    if r.returncode != 0:
        lines = r.stderr.strip().splitlines()
        raise VideoError(f"not a readable video ({lines[-1] if lines else 'ffprobe failed'})")
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except (KeyError, ValueError):
        return None


def iter_clips(path: Path, window_sec: float, frames_per_clip: int, size: int,
               duration: float | None = None, start_at: float = 0.0):
    """Yield consecutive, non-overlapping clips covering the video from start_at onwards.

    One ffmpeg process decodes the video at frames_per_clip / window_sec fps and
    streams square, letterboxed RGB frames; every frames_per_clip frames become a clip.
    Videos shorter than one window become a single clip spread over their length.
    """
    window = min(window_sec, duration) if duration else window_sec
    fps = frames_per_clip / window
    vf = (f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=decrease,"
          f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2")
    # -nostdin: otherwise ffmpeg reads the terminal and eats keys typed during indexing.
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_at:
        cmd += ["-ss", f"{start_at:g}"]
    cmd += ["-i", str(path), "-an", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = size * size * 3
    # stderr is discarded: an unread pipe can fill up and stall ffmpeg on damaged files.
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        buf, start, emitted = [], start_at, 0
        while raw := proc.stdout.read(frame_bytes):
            if len(raw) < frame_bytes:
                break
            buf.append(np.frombuffer(raw, np.uint8).reshape(size, size, 3))
            if len(buf) == frames_per_clip:
                yield Clip(start, start + window, np.stack(buf))
                buf, start, emitted = [], start + window, emitted + 1
        if buf:
            yield Clip(start, start + len(buf) / fps, np.stack(buf))
        elif not emitted and not start_at:
            raise VideoError("ffmpeg could not decode any frames")
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
