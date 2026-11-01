"""On-disk index, by default inside the searched folder at <folder>/.video-grep/<model size>/.

Each model size keeps its own index, so switching sizes and back doesn't re-index.

manifest.json maps each video (path relative to the folder) to its size, mtime,
clip time ranges and the .npy file holding one embedding per clip. Updates are
incremental: only new or changed videos are embedded, deleted ones are dropped.
Long videos are checkpointed as they go, and a partly indexed video is searchable.
"""

import fcntl
import fnmatch
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .frames import VideoError, find_videos, iter_clips, probe_duration

INDEX_DIR = ".video-grep"
MANIFEST_VERSION = 1
SETTINGS = {"window_sec": 6.0, "frames_per_clip": 8, "size": 384}
CHECKPOINT_CLIPS = 300  # save progress inside a video every 30 minutes of footage
# Losing this much of the index at once looks like an unmounted disk, not deleted videos.
MASS_REMOVAL_MIN = 5
MASS_REMOVAL_SHARE = 0.25
INDEX_BYTES_PER_SEC = 280  # measured: about 17 KB per minute of footage


@dataclass
class Status:
    new: list[str]
    changed: list[str]
    partial: list[str]
    failed: list[str]
    removed: list[str]
    unchanged: int
    rebuild: bool  # model or settings differ from what the index was built with

    @property
    def pending(self) -> list[str]:
        return self.new + self.changed + self.partial


class VideoIndex:
    def __init__(self, folder: Path, model_name: str, name: str, root: Path | None = None):
        self.folder = folder.resolve()
        self.dir = (root or self.folder / INDEX_DIR) / name
        self.vec_dir = self.dir / "vectors"
        self.manifest_path = self.dir / "manifest.json"
        self.model_name = model_name
        self.manifest = self._load()

    def _empty(self) -> dict:
        return {"version": MANIFEST_VERSION, "model": self.model_name, "settings": SETTINGS, "files": {}}

    def _load(self) -> dict:
        if not self.manifest_path.exists():
            return self._empty()
        try:
            manifest = json.loads(self.manifest_path.read_text())
            if isinstance(manifest.get("files"), dict):
                return manifest
        except (ValueError, AttributeError):
            pass
        os.replace(self.manifest_path, self.manifest_path.with_suffix(".corrupt"))
        print("The saved index was damaged, so this folder will be indexed again.")
        return self._empty()

    def _save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.manifest_path)

    def _write_vectors(self, name: str, vectors: np.ndarray):
        tmp = self.vec_dir / (name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vectors.astype(np.float16))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.vec_dir / name)

    @property
    def files(self) -> dict:
        return self.manifest["files"]

    def _incompatible(self) -> bool:
        m = self.manifest
        return m.get("version") != MANIFEST_VERSION or m.get("model") != self.model_name or m.get("settings") != SETTINGS

    @contextmanager
    def _locked(self):
        with open(self.dir / "lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another video-grep is already indexing this folder") from None
            yield

    def is_indexing(self) -> bool:
        """True if another process, such as a background index, holds the lock."""
        lock = self.dir / "lock"
        if not lock.exists():
            return False
        with open(lock) as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
        return False

    def status(self) -> Status:
        rebuild = self._incompatible() and bool(self.files)
        on_disk = {}
        for p in find_videos(self.folder):
            st = p.stat()
            on_disk[str(p.relative_to(self.folder))] = (st.st_size, st.st_mtime)
        new, changed, partial, failed, unchanged = [], [], [], [], 0
        for rel, (size, mtime) in on_disk.items():
            entry = self.files.get(rel)
            if entry is None:
                new.append(rel)
            elif rebuild or entry["size"] != size or entry["mtime"] != mtime:
                changed.append(rel)
            elif entry.get("partial"):
                partial.append(rel)
            elif "error" in entry:
                failed.append(rel)
            else:
                unchanged += 1
        removed = [rel for rel in self.files if rel not in on_disk]
        return Status(new, changed, partial, failed, removed, unchanged, rebuild)

    def verify(self) -> int:
        """Forget videos whose saved vectors are missing or don't match; they get re-indexed."""
        bad = []
        for rel, entry in self.files.items():
            if "vectors" not in entry:
                continue
            try:
                rows = np.load(self.vec_dir / entry["vectors"], mmap_mode="r").shape[0]
            except (OSError, ValueError, EOFError):
                rows = -1
            if rows != len(entry["segments"]):
                bad.append(rel)
        for rel in bad:
            self._drop(rel)
        if bad:
            self._save()
        return len(bad)

    def plan(self, status: Status, force: bool = False, retry_failed: bool = False) -> list[str]:
        """The videos an update would process."""
        todo = set(status.pending)
        if retry_failed:
            todo |= set(status.failed)
        if force:
            todo |= set(self.files) - set(status.removed)
        return sorted(todo)

    def probe(self, videos: list[str]) -> dict[str, float | None]:
        durations = {}
        for rel in tqdm(videos, disable=len(videos) < 50, leave=False,
                        bar_format="  Checking videos |{bar:30}| {n_fmt}/{total_fmt}"):
            try:
                durations[rel] = probe_duration(self.folder / rel)
            except VideoError:
                durations[rel] = None
        return durations

    def update(self, embedder, force: bool = False, retry_failed: bool = False, prune: bool = False,
               durations: dict | None = None) -> dict:
        """Embed new/changed videos and drop removed ones. Returns counts.

        Only VideoError marks a file as failed. Anything else (disk full, out of memory)
        is a machine problem: it stops the run and leaves the file to be tried again.
        """
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        with self._locked():
            return self._update(embedder, force, retry_failed, prune, durations)

    def _update(self, embedder, force, retry_failed, prune, durations) -> dict:
        status = self.status()
        if status.rebuild:  # old vectors can't be compared with new ones, so never keep a mix
            for rel in list(self.files):
                self._drop(rel)
        removed = status.removed
        kept_missing = 0
        if not prune and len(removed) >= MASS_REMOVAL_MIN and len(removed) > MASS_REMOVAL_SHARE * len(self.files):
            kept_missing, removed = len(removed), []
        for rel in removed:
            self._drop(rel)
        self.manifest.update(version=MANIFEST_VERSION, model=self.model_name, settings=SETTINGS)
        self._save()

        todo = self.plan(status, force, retry_failed)
        durations = durations if durations is not None else self.probe(todo)

        failed, warnings, gone = [], [], 0
        # Progress is measured in seconds of footage, so long videos move the bar proportionally.
        bar = tqdm(total=sum(d or 0 for d in durations.values()), disable=not todo, dynamic_ncols=True,
                   bar_format="  Indexing |{bar:30}| {percentage:3.0f}%  "
                              "{elapsed} elapsed, ~{remaining} left  {desc}")
        try:
            for i, rel in enumerate(todo, 1):
                bar.set_description_str(f"video {i}/{len(todo)}: {Path(rel).name[-40:]}")
                path = self.folder / rel
                try:
                    warning = self._index_file(rel, durations.get(rel), embedder, bar, force)
                    if warning:
                        warnings.append((rel, warning))
                except VideoError as e:
                    failed.append((rel, str(e)))
                    st = path.stat()
                    self._drop(rel)
                    # Remember it so a broken file isn't retried on every run.
                    self.files[rel] = {"size": st.st_size, "mtime": st.st_mtime, "error": str(e)[:200]}
                except FileNotFoundError:
                    if path.exists():
                        raise
                    gone += 1  # deleted while we were indexing, e.g. by NVR retention
                    self._drop(rel)
                self._save()
        finally:
            bar.close()
        return {"indexed": len(todo) - len(failed) - gone, "removed": len(removed), "failed": failed,
                "warnings": warnings, "kept_missing": kept_missing}

    def _index_file(self, rel: str, duration: float | None, embedder, bar, force: bool) -> str | None:
        path = self.folder / rel
        st = path.stat()
        s = SETTINGS
        vec_name = hashlib.sha1(os.fsencode(rel)).hexdigest()[:16] + ".npy"
        entry = self.files.get(rel, {})
        resume = (not force and entry.get("partial") and entry["size"] == st.st_size
                  and entry["mtime"] == st.st_mtime)
        segments = list(entry["segments"]) if resume else []
        vecs = [np.load(self.vec_dir / entry["vectors"]).astype(np.float32)] if resume else []
        start_at = segments[-1][1] if resume else 0.0
        bar.update(start_at)
        batch = []
        since_checkpoint = 0

        def flush():
            nonlocal since_checkpoint
            vecs.append(embedder.embed_clips([c.frames for c in batch]))
            segments.extend([round(c.start, 2), round(c.end, 2)] for c in batch)
            bar.update(batch[-1].end - batch[0].start)
            since_checkpoint += len(batch)
            batch.clear()

        def save(partial: bool):
            self._write_vectors(vec_name, np.concatenate(vecs))
            self.files[rel] = {"size": st.st_size, "mtime": st.st_mtime, "duration": round(duration or segments[-1][1], 2),
                               "vectors": vec_name, "segments": list(segments)}
            if partial:
                self.files[rel]["partial"] = True

        for clip in iter_clips(path, s["window_sec"], s["frames_per_clip"], s["size"], duration, start_at):
            batch.append(clip)
            if len(batch) == 4:
                flush()
            if since_checkpoint >= CHECKPOINT_CLIPS:
                save(partial=True)
                self._save()
                since_checkpoint = 0
        if batch:
            flush()
        save(partial=False)

        if duration and duration - segments[-1][1] > s["window_sec"]:
            return f"only the first {segments[-1][1]:.0f}s of {duration:.0f}s could be read; the file may be damaged"
        return None

    def _drop(self, rel: str):
        entry = self.files.pop(rel, None)
        if entry and entry.get("vectors"):
            (self.vec_dir / entry["vectors"]).unlink(missing_ok=True)

    def select(self, path: str | None = None, after: date | None = None, before: date | None = None) -> set[str]:
        """Indexed videos in a subfolder or matching a pattern, and modified within the dates."""
        chosen = set()
        for rel, entry in self.files.items():
            if "vectors" not in entry:
                continue
            if path and not (rel.startswith(path.rstrip("/") + "/") or fnmatch.fnmatch(rel, path)):
                continue
            day = date.fromtimestamp(entry["mtime"])
            if (after and day < after) or (before and day > before):
                continue
            chosen.add(rel)
        return chosen

    def load(self, videos: set[str] | None = None) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
        """Clip vectors stacked into one matrix, plus (video, start, end) per row.

        Reads only: damaged entries are skipped here, and verify() repairs them.
        """
        mats, rows = [], []
        for rel, entry in self.files.items():
            if "vectors" not in entry or (videos is not None and rel not in videos):
                continue
            try:
                vectors = np.load(self.vec_dir / entry["vectors"]).astype(np.float32)
            except (OSError, ValueError, EOFError):
                continue
            if len(vectors) != len(entry["segments"]):
                continue
            mats.append(vectors)
            rows.extend((rel, s, e) for s, e in entry["segments"])
        if not mats:
            return np.zeros((0, 0), np.float32), []
        return np.concatenate(mats), rows
