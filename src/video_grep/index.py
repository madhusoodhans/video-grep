"""On-disk index stored inside the searched folder, at <folder>/.video-grep/<model size>/.

Each model size keeps its own index, so switching sizes and back doesn't re-index.

manifest.json maps each video (path relative to the folder) to its size, mtime,
clip time ranges and the .npy file holding one embedding per clip. Updates are
incremental: only new or changed videos are embedded, deleted ones are dropped.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .frames import find_videos, iter_clips, probe_duration

INDEX_DIR = ".video-grep"
MANIFEST_VERSION = 1
SETTINGS = {"window_sec": 6.0, "frames_per_clip": 8, "size": 384}


@dataclass
class Status:
    new: list[str]
    changed: list[str]
    removed: list[str]
    unchanged: int
    rebuild: bool  # model or settings differ from what the index was built with

    @property
    def pending(self) -> list[str]:
        return self.new + self.changed


class VideoIndex:
    def __init__(self, folder: Path, model_name: str, name: str):
        self.folder = folder.resolve()
        self.dir = self.folder / INDEX_DIR / name
        self.vec_dir = self.dir / "vectors"
        self.manifest_path = self.dir / "manifest.json"
        self.model_name = model_name
        self.manifest = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"version": MANIFEST_VERSION, "model": self.model_name, "settings": SETTINGS, "files": {}}

    def _save(self):
        tmp = self.manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1))
        os.replace(tmp, self.manifest_path)

    @property
    def files(self) -> dict:
        return self.manifest["files"]

    def _incompatible(self) -> bool:
        m = self.manifest
        return m.get("version") != MANIFEST_VERSION or m.get("model") != self.model_name or m.get("settings") != SETTINGS

    def status(self) -> Status:
        rebuild = self._incompatible() and bool(self.files)
        on_disk = {}
        for p in find_videos(self.folder):
            st = p.stat()
            on_disk[str(p.relative_to(self.folder))] = (st.st_size, st.st_mtime)
        new, changed, unchanged = [], [], 0
        for rel, (size, mtime) in on_disk.items():
            entry = self.files.get(rel)
            if entry is None:
                new.append(rel)
            elif rebuild or entry["size"] != size or entry["mtime"] != mtime:
                changed.append(rel)
            else:
                unchanged += 1
        removed = [rel for rel in self.files if rel not in on_disk]
        return Status(new, changed, removed, unchanged, rebuild)

    def update(self, embedder, force: bool = False) -> dict:
        """Embed new/changed videos and drop removed ones. Returns counts."""
        self.vec_dir.mkdir(parents=True, exist_ok=True)
        status = self.status()
        todo = set(status.pending)
        if force:
            todo |= set(self.files) - set(status.removed)
        todo = sorted(todo)

        for rel in status.removed:
            self._drop(rel)
        self.manifest.update(version=MANIFEST_VERSION, model=self.model_name, settings=SETTINGS)
        self._save()

        durations = {}
        for rel in todo:
            try:
                durations[rel] = probe_duration(self.folder / rel)
            except Exception:
                durations[rel] = 0.0

        failed = []
        # Progress is measured in seconds of footage, so long videos move the bar proportionally.
        bar = tqdm(total=sum(durations.values()), disable=not todo, dynamic_ncols=True,
                   bar_format="  Indexing |{bar:30}| {percentage:3.0f}%  "
                              "{elapsed} elapsed, ~{remaining} left  {desc}")
        for i, rel in enumerate(todo, 1):
            bar.set_description_str(f"video {i}/{len(todo)}: {Path(rel).name[-40:]}")
            try:
                self._index_file(rel, durations[rel], embedder, bar)
            except Exception as e:
                failed.append((rel, str(e)))
                path = self.folder / rel
                st = path.stat()
                self._drop(rel)
                # Remember the failure so the same broken file isn't retried every run.
                self.files[rel] = {"size": st.st_size, "mtime": st.st_mtime, "error": str(e)[:200]}
            self._save()
        bar.close()
        return {"indexed": len(todo) - len(failed), "removed": len(status.removed),
                "unchanged": 0 if force else status.unchanged, "failed": failed}

    def _index_file(self, rel: str, duration: float, embedder, bar):
        path = self.folder / rel
        st = path.stat()
        s = SETTINGS
        segments, vecs, batch = [], [], []

        def flush():
            vecs.append(embedder.embed_clips([c.frames for c in batch]))
            segments.extend([round(c.start, 2), round(c.end, 2)] for c in batch)
            bar.update(batch[-1].end - batch[0].start)
            batch.clear()

        for clip in iter_clips(path, s["window_sec"], s["frames_per_clip"], s["size"], duration or None):
            batch.append(clip)
            if len(batch) == 4:
                flush()
        if batch:
            flush()

        vec_name = hashlib.sha1(rel.encode()).hexdigest()[:16] + ".npy"
        np.save(self.vec_dir / vec_name, np.concatenate(vecs).astype(np.float16))
        self.files[rel] = {"size": st.st_size, "mtime": st.st_mtime, "duration": round(duration, 2),
                           "vectors": vec_name, "segments": segments}

    def _drop(self, rel: str):
        entry = self.files.pop(rel, None)
        if entry and entry.get("vectors"):
            (self.vec_dir / entry["vectors"]).unlink(missing_ok=True)

    def load(self) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
        """All clip vectors stacked into one matrix, plus (video, start, end) per row."""
        mats, rows = [], []
        for rel, entry in self.files.items():
            if "vectors" not in entry:
                continue
            mats.append(np.load(self.vec_dir / entry["vectors"]).astype(np.float32))
            rows.extend((rel, s, e) for s, e in entry["segments"])
        if not mats:
            return np.zeros((0, 0), np.float32), []
        return np.concatenate(mats), rows
