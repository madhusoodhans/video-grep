"""Download a small set of freely licensed test videos from Wikimedia Commons.

Files get neutral names (clip_01.mov, ...) so filenames can't leak the answer.
The real topic, source and license for each file go in ground_truth.json.

Usage: python scripts/download_testdata.py [out_dir]
"""

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = "video-grep-testdata/0.1 (personal research script)"

TOPICS = [
    "dog playing", "cat sleeping", "baby crawling", "child trampoline",
    "beach waves", "mountain hiking", "skiing snow", "birthday cake candles",
    "cooking frying pan", "riding bicycle", "horse running", "birds flying flock",
    "waterfall", "fireworks", "city traffic night", "rain on window",
    "swimming pool", "soccer kick", "basketball", "playing guitar",
    "dancing", "train arriving station", "airplane takeoff", "aquarium fish",
    "elephant", "kite flying", "campfire", "sailing boat", "concert crowd",
    "bee flower",
]

MIN_SEC, MAX_SEC = 5, 120
MAX_BYTES = 40_000_000
PREFERRED = ["360p.mpeg4.mov", "360p.vp9.webm", "240p.vp9.webm", "480p.vp9.webm"]


PAUSE_SEC = 4  # Commons rate-limits bursts; stay polite


def urlopen(url, timeout):
    """GET with retries on HTTP 429, honouring Retry-After."""
    for attempt in range(5):
        time.sleep(PAUSE_SEC)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 4:
                raise
            wait = int(e.headers.get("Retry-After") or 30 * (attempt + 1))
            print(f"       rate limited, waiting {wait}s")
            time.sleep(wait)


def api(**params):
    params |= {"action": "query", "format": "json"}
    with urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=30) as r:
        return json.load(r)


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def find_clip(topic, taken):
    data = api(
        generator="search", gsrnamespace=6, gsrsearch=f"filetype:video {topic}", gsrlimit=20,
        prop="videoinfo", viprop="url|size|duration|derivatives|extmetadata",
    )
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    for p in pages:
        if p["title"] in taken:
            continue
        vi = p["videoinfo"][0]
        if not MIN_SEC <= (vi.get("duration") or 0) <= MAX_SEC:
            continue
        ders = {d.get("transcodekey"): d["src"] for d in vi.get("derivatives", []) if d.get("transcodekey")}
        key = next((k for k in PREFERRED if k in ders), None)
        if not key:
            continue
        meta = vi.get("extmetadata", {})
        return {
            "topic": topic,
            "title": p["title"],
            "duration": round(vi["duration"], 1),
            "source_page": vi["descriptionurl"],
            "download_url": ders[key],
            "license": strip_html(meta.get("LicenseShortName", {}).get("value")),
            "artist": strip_html(meta.get("Artist", {}).get("value")),
            "ext": "." + key.rsplit(".", 1)[-1],
        }
    return None


def download(url, dest):
    with urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        total = 0
        while chunk := r.read(1 << 16):
            total += len(chunk)
            if total > MAX_BYTES:
                raise ValueError("file too large")
            f.write(chunk)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata")
    out.mkdir(parents=True, exist_ok=True)
    truth_file = out / "ground_truth.json"
    truth = json.loads(truth_file.read_text()) if truth_file.exists() else {}
    taken = {c["title"] for c in truth.values()}
    done_topics = {c["topic"] for c in truth.values()}
    for i, topic in enumerate(TOPICS, 1):
        if topic in done_topics:
            continue
        try:
            clip = find_clip(topic, taken)
            if not clip:
                print(f"[skip] {topic}: no suitable clip")
                continue
            name = f"clip_{i:02d}{clip.pop('ext')}"
            download(clip["download_url"], out / name)
            taken.add(clip["title"])
            truth[name] = clip
            truth_file.write_text(json.dumps(truth, indent=2))
            print(f"[ok]   {name}  {topic!r:28} {clip['duration']:>6}s  {clip['title'][:60]}")
        except Exception as e:  # keep going; a partial set is still useful
            print(f"[fail] {topic}: {e}")
    print(f"\n{len(truth)} clips saved to {out}/")


if __name__ == "__main__":
    main()
