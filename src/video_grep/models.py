"""Model sizes the user can pick from, and downloading them with a progress bar.

Models live in ~/.cache/video-grep/models/<size>/ (override with VIDEO_GREP_HOME).
Downloads resume if interrupted; a .complete marker means a model is ready.
"""

import fnmatch
import json
import os
import shutil
import sys
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


ONNX_REPO = "onnx-community/embeddinggemma-2-ONNX"
ONNX_COMMON = ("*.json", "*.jinja", "tokenizer.json")

SIZES = {
    s.key: s for s in [
        ModelSize("high", "High", "best quality, fastest (uses your GPU)",
                  "google/embeddinggemma-2", ("*",), "torch", 1520),
        ModelSize("medium", "Medium", "same quality as High, but indexing is several times slower (CPU)",
                  ONNX_REPO, ONNX_COMMON + ("onnx/vision_encoder_quantized.onnx*", "onnx/model_quantized.onnx*"),
                  "onnx", 550, "_quantized"),
        ModelSize("small", "Small", "slightly lower quality, slowest indexing (CPU)",
                  ONNX_REPO, ONNX_COMMON + ("onnx/vision_encoder_q4.onnx*", "onnx/model_q4.onnx*"),
                  "onnx", 320, "_q4"),
    ]
}
DEFAULT_SIZE = "high"


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


def ask_size() -> ModelSize:
    """Interactive first-run menu. Enter picks the recommended default."""
    keys = list(SIZES)
    print("Choose a model size to download (needed once):\n")
    for i, key in enumerate(keys, 1):
        s = SIZES[key]
        rec = "  (recommended)" if key == DEFAULT_SIZE else ""
        print(f"  {i}. {s.title:<7} ~{fmt_mb(s.approx_mb):>7}   {s.summary}{rec}")
    default = keys.index(DEFAULT_SIZE) + 1
    while True:
        try:
            ans = input(f"\nEnter 1-{len(keys)} [{default}]: ").strip()
        except EOFError:
            ans = ""
        if not ans:
            return SIZES[DEFAULT_SIZE]
        if ans.isdigit() and 1 <= int(ans) <= len(keys):
            return SIZES[keys[int(ans) - 1]]
        if ans.lower() in SIZES:
            return SIZES[ans.lower()]
        print(f"  Please enter a number from 1 to {len(keys)}.")


def _list_files(size: ModelSize) -> list[tuple[str, int]]:
    url = f"{HF}/api/models/{size.repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        tree = json.load(r)
    return [(f["path"], f.get("size", 0)) for f in tree
            if f["type"] == "file" and f["path"] != ".gitattributes"
            and any(fnmatch.fnmatch(f["path"], p) for p in size.patterns)]


def download(size: ModelSize):
    """Fetch the model's files into size.dir with one overall progress bar. Resumable."""
    files = _list_files(size)
    total = sum(n for _, n in files)
    have = sum(min(_partial_size(size.dir / p), n) for p, n in files)
    free = shutil.disk_usage(size.dir.parent if size.dir.parent.exists() else Path.home()).free
    if total - have > free - 200_000_000:
        sys.exit(f"Not enough disk space: the {size.title} model needs {fmt_mb((total - have) / 1e6)}, "
                 f"only {fmt_mb(free / 1e6)} free.")

    # delay: skip the first second, whose speed/ETA estimate is meaningless.
    bar = tqdm(total=total, initial=have, unit="B", unit_scale=True, unit_divisor=1024, dynamic_ncols=True, delay=1,
               bar_format="  Downloading |{bar:30}| {percentage:3.0f}%  {n_fmt}/{total_fmt}  "
                          "{rate_fmt}, ~{remaining} left")
    for path, n in files:
        _download_file(f"{HF}/{size.repo}/resolve/main/{path}", size.dir / path, n, bar)
    bar.close()
    (size.dir / ".complete").write_text(json.dumps([p for p, _ in files]))


def _partial_size(dest: Path) -> int:
    if dest.exists():
        return dest.stat().st_size
    part = dest.with_name(dest.name + ".part")
    return part.stat().st_size if part.exists() else 0


def _download_file(url: str, dest: Path, expected: int, bar):
    if dest.exists() and dest.stat().st_size == expected:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if start and r.status != 206:  # server ignored the range; start over
            bar.update(-start)
            start = 0
        with open(part, "ab" if start else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)


def ensure_model(requested: str | None) -> ModelSize:
    """Pick the model size (flag > saved choice > menu) and make sure it's downloaded."""
    if requested:
        size = SIZES[requested]
    elif (key := saved_size()) and SIZES[key].is_downloaded():
        size = SIZES[key]
    elif sys.stdin.isatty():
        size = ask_size()
    else:
        size = SIZES[DEFAULT_SIZE]
    if not size.is_downloaded():
        print(f"\nDownloading the {size.title} model (~{fmt_mb(size.approx_mb)}) to {size.dir}")
        try:
            download(size)
        except KeyboardInterrupt:
            sys.exit("\n\nDownload paused. Run again to resume where it stopped.")
        print("Download complete.\n")
    save_size(size.key)
    return size
