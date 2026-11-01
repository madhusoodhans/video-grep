"""Model sizes the user can pick from, and downloading them with a progress bar.

Models live in ~/.cache/video-grep/models/<size>/ (override with VIDEO_GREP_HOME).
Downloads are pinned to a repo revision, verified by size and SHA-256, and resume if
interrupted; a .complete marker means a model is ready.
"""

import fnmatch
import hashlib
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

HOME = Path(os.environ.get("VIDEO_GREP_HOME", Path.home() / ".cache" / "video-grep"))
SETTINGS_FILE = HOME / "settings.json"
HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co")


@dataclass(frozen=True)
class ModelSize:
    key: str
    title: str
    summary: str
    repo: str
    patterns: tuple[str, ...]
    backend: str  # "torch" or "onnx"
    approx_mb: int
    revision: str
    onnx_suffix: str = ""

    @property
    def dir(self) -> Path:
        return HOME / "models" / self.key

    @property
    def index_id(self) -> str:
        """Recorded in each folder's index; changing it forces a re-index."""
        return self.repo + (f":{self.onnx_suffix.lstrip('_')}" if self.onnx_suffix else "")

    def is_downloaded(self) -> bool:
        return (self.dir / ".complete").exists()


GOOGLE_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
ONNX_REPO = "onnx-community/embeddinggemma-2-ONNX"
ONNX_REVISION = "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
ONNX_COMMON = ("*.json", "*.jinja", "tokenizer.json")

SIZES = {
    s.key: s for s in [
        ModelSize("high", "High", "best quality, fastest (uses your GPU)",
                  "google/embeddinggemma-2", ("*",), "torch", 1520, GOOGLE_REVISION),
        ModelSize("medium", "Medium", "nearly the same as High, but indexing is several times slower (CPU)",
                  ONNX_REPO, ONNX_COMMON + ("onnx/vision_encoder_quantized.onnx*", "onnx/model_quantized.onnx*"),
                  "onnx", 550, ONNX_REVISION, "_quantized"),
        ModelSize("small", "Small", "slightly lower quality, slowest indexing (CPU)",
                  ONNX_REPO, ONNX_COMMON + ("onnx/vision_encoder_q4.onnx*", "onnx/model_q4.onnx*"),
                  "onnx", 320, ONNX_REVISION, "_q4"),
    ]
}
DEFAULT_SIZE = "high"
HIGH_CPU_RAM_GB = 12  # High peaked at about 7.3 GB on CPU; leave room for everything else


class DownloadError(Exception):
    pass


def saved_size() -> str | None:
    try:
        key = json.loads(SETTINGS_FILE.read_text()).get("model_size")
    except (OSError, ValueError):
        return None
    return key if key in SIZES else None


def save_size(key: str):
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps({"model_size": key}))


def fmt_mb(mb: float) -> str:
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.0f} MB"


def has_gpu() -> bool:
    import torch

    return torch.backends.mps.is_available() or torch.cuda.is_available()


def ram_gb() -> float:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


def recommended_size(gpu: bool, ram: float) -> str:
    return DEFAULT_SIZE if gpu or ram >= HIGH_CPU_RAM_GB else "small"


def ask_size() -> ModelSize:
    """Interactive first-run menu. Enter picks the size recommended for this machine."""
    keys = list(SIZES)
    gpu, ram = has_gpu(), ram_gb()
    recommended = recommended_size(gpu, ram)
    print(f"This machine: {'GPU found' if gpu else 'no GPU found'}, {ram:.0f} GB RAM.\n")
    print("Choose a model size to download (needed once):\n")
    for i, key in enumerate(keys, 1):
        s = SIZES[key]
        summary = "best quality, needs about 8 GB RAM without a GPU" if key == "high" and not gpu else s.summary
        rec = "  (recommended)" if key == recommended else ""
        print(f"  {i}. {s.title:<7} ~{fmt_mb(s.approx_mb):>7}   {summary}{rec}")
    default = keys.index(recommended) + 1
    while True:
        try:
            ans = input(f"\nEnter 1-{len(keys)} [{default}]: ").strip()
        except EOFError:
            ans = ""
        if not ans:
            return SIZES[recommended]
        if ans.isdigit() and 1 <= int(ans) <= len(keys):
            return SIZES[keys[int(ans) - 1]]
        if ans.lower() in SIZES:
            return SIZES[ans.lower()]
        print(f"  Please enter a number from 1 to {len(keys)}.")


def _list_files(size: ModelSize) -> list[tuple[str, int, str | None]]:
    url = f"{HF}/api/models/{size.repo}/tree/{size.revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        tree = json.load(r)
    return [(f["path"], f.get("size", 0), f.get("lfs", {}).get("oid")) for f in tree
            if f["type"] == "file" and f["path"] != ".gitattributes"
            and any(fnmatch.fnmatch(f["path"], p) for p in size.patterns)]


def download(size: ModelSize):
    """Fetch the model's files into size.dir with one overall progress bar. Resumable."""
    files = _list_files(size)
    total = sum(n for _, n, _ in files)
    have = sum(min(_partial_size(size.dir / p), n) for p, n, _ in files)
    disk = size.dir
    while not disk.exists():
        disk = disk.parent
    free = shutil.disk_usage(disk).free
    if total - have > free - 200_000_000:
        sys.exit(f"Not enough disk space: the {size.title} model needs {fmt_mb((total - have) / 1e6)}, "
                 f"only {fmt_mb(free / 1e6)} free.")

    # delay: skip the first second, whose speed/ETA estimate is meaningless.
    bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024, dynamic_ncols=True, delay=1,
               bar_format="  Downloading |{bar:30}| {percentage:3.0f}%  {n_fmt}/{total_fmt}  "
                          "{rate_fmt}, ~{remaining} left")
    try:
        for path, n, sha256 in files:
            url = f"{HF}/{size.repo}/resolve/{size.revision}/{path}"
            _with_retries(lambda: _download_file(url, size.dir / path, n, sha256, bar))
    finally:
        bar.close()
    (size.dir / ".complete").write_text(json.dumps([p for p, _, _ in files]))


def _with_retries(fn, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == attempts - 1:
                raise
        except (OSError, http.client.HTTPException, DownloadError):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def _partial_size(dest: Path) -> int:
    if dest.exists():
        return dest.stat().st_size
    part = dest.with_name(dest.name + ".part")
    return part.stat().st_size if part.exists() else 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, dest: Path, expected: int, sha256: str | None, bar):
    if dest.exists() and dest.stat().st_size == expected:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start > expected:  # left over from something else; can't be resumed
        part.unlink()
        bar.update(-start)
        start = 0
    if start < expected or not part.exists():
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            if start and r.status != 206:  # server ignored the range; start over
                bar.update(-start)
                start = 0
            with open(part, "ab" if start else "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    got = part.stat().st_size
    if got != expected:  # the connection dropped; the .part file is kept so this can resume
        raise DownloadError(f"{dest.name} is incomplete ({got} of {expected} bytes)")
    if sha256 and _sha256(part) != sha256:
        part.unlink()
        bar.update(-got)
        raise DownloadError(f"{dest.name} failed its checksum")
    os.replace(part, dest)


def ensure_model(requested: str | None) -> ModelSize:
    """Pick the model size (flag > saved choice > menu) and make sure it's downloaded.

    Only the menu changes the saved choice, so --model-size affects a single run.
    A first run with no terminal stops instead of starting a large download unasked.
    """
    saved = saved_size()
    if requested:
        size = SIZES[requested]
    elif saved and SIZES[saved].is_downloaded():
        size = SIZES[saved]
    elif sys.stdin.isatty():
        size = ask_size()
        save_size(size.key)
    else:
        sys.exit("No model chosen yet. Run this once in a terminal, or pass "
                 "--model-size high, medium or small.")
    if not size.is_downloaded():
        print(f"\nDownloading the {size.title} model (~{fmt_mb(size.approx_mb)}) to {size.dir}", file=sys.stderr)
        try:
            download(size)
        except KeyboardInterrupt:
            sys.exit("\n\nDownload paused. Run again to resume where it stopped.")
        except (OSError, http.client.HTTPException, DownloadError) as e:
            sys.exit(f"\n\nCould not download the model: {e}\n"
                     f"Check your connection to {HF} and run again; it resumes where it stopped.")
        print("Download complete.\n", file=sys.stderr)
    if not saved:
        save_size(size.key)
    return size
