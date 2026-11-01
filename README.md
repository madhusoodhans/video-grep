# video-grep

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-lightgrey)
![Local](https://img.shields.io/badge/runs-100%25%20local-success)
![Terminal](https://img.shields.io/badge/interface-terminal%20%2F%20SSH-black?logo=gnubash&logoColor=white)

**Find the moment inside your videos by describing what happens in it, straight from the terminal.**

Type an action, scene or situation, such as *"person climbing over the fence"*, *"forklift reversing near the
dock"* or *"kid jumping on the bed"*. `video-grep` looks at the actual frames and returns the videos
where it happens, with the timestamp. It's not a file name or metadata search: your files can be called
`IMG_4521.mov` or `cam2_2300.mp4` and it still works.

No monitor, no video player, no downloading footage to scrub through it by hand.

```
search> person walking near the gate at night

  Top 3 matches for "person walking near the gate at night":
    1. cam2/2026-10-21_2300.mp4  (at 14:36, strong match, 0.74)
    2. cam2/2026-10-20_2300.mp4  (at 3:12, strong match, 0.71)
    3. cam1/2026-10-21_2300.mp4  (at 41:05, possible match, 0.68)

  Type a number (1-3) to see its full path, "save <number>" to save that clip, or type a new search.
```
<sub>Illustrative output. The match comes from what happens on screen; the file names say nothing about it.</sub>

## The problem

A lot of video lives on machines you only reach through a terminal: remote servers, cloud
VMs, CCTV/NVR boxes, embedded and edge boards (Jetson, Raspberry Pi-class devices),
lab rigs and storage boxes. To find one moment in that footage today you typically:

1. copy gigabytes of video to a machine with a screen,
2. open each file and scrub through it manually,
3. repeat until you find the part you need.

Search tools that understand video content exist, but they are built around graphical apps
and cloud services, not around an SSH session.

## What this does

`video-grep` goes through every video in a folder **where it already is**, sampling frames across
each 6-second stretch, and builds a searchable index of what happens in it: objects, actions, scenes
and settings. You then describe the
moment you need, such as "forklift near the loading dock", "dog chasing bubbles" or "crowd waving glow
sticks at a concert". You get the videos where that happens, ranked, each with the exact moment.

- **Terminal-first.** Built for SSH sessions. On a headless machine, picking a result shows the
  file path and time range, and `save` writes just that clip, so you can usually copy a few MB
  instead of the whole recording.
- **Works with read-only footage.** If the folder can't be written to, the index is kept elsewhere.
- **Handles long recordings.** One file can return several separate moments, and indexing saves
  progress every 30 minutes of footage.
- **Keeps going when you disconnect.** `video-grep index --background` carries on after your SSH
  session ends, and you can search whatever has finished while it runs.
- **Searches the content, not the file names.** Describe actions, activities, scenes or objects in
  plain English. No tags, captions, metadata or manual labelling needed.
- **Local and private.** After a one-time model download, everything runs offline on the machine.
  No video, frame or query leaves it.
- **Incremental.** Re-running only processes new or changed videos.
- **Built for macOS and Linux.** Uses a GPU if available, CPU otherwise. So far it has only been
  tested on macOS.
- **Choose your footprint.** Three model sizes, from 1.5 GB down to 320 MB.

On a desktop, picking a result opens the video at the matching moment instead
in QuickTime on macOS. On Linux it opens in the default player and you skip to the time shown.

## Quick start

Requires ffmpeg and Python 3.12, 3.13 or 3.14 (not 3.14.1). Check with `python3 --version`: Ubuntu
22.04, Debian 12 and JetPack 6 ship older versions, so you need to install a newer Python there first.
PyTorch has no builds for Intel Macs or 32-bit Raspberry Pi OS.

```bash
# macOS: brew install python ffmpeg      Ubuntu/Debian: sudo apt install python3-venv ffmpeg
git clone https://github.com/madhusoodhans/video-grep.git
cd video-grep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .

video-grep doctor            # check this machine is ready
video-grep /path/to/videos   # index the folder, then search it
```

**On Linux without an NVIDIA GPU**, install the CPU build of PyTorch first to avoid about 2.7 GB of
NVIDIA packages. Run this before the `pip install -r` line above (not yet tested):

```bash
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
```

On first run you see what hardware was found and choose a model size; Enter picks the one recommended
for your machine. It downloads once with a progress bar, then the folder is indexed:

```
This machine: GPU found, 24 GB RAM.

Choose a model size to download (needed once):

  1. High    ~ 1.5 GB   best quality, fastest (uses your GPU)  (recommended)
  2. Medium  ~ 550 MB   nearly the same as High, but indexing is several times slower (CPU)
  3. Small   ~ 320 MB   slightly lower quality, slowest indexing (CPU)

  Downloading |██████████████                |  47%  712M/1.42G  31.2MB/s, ~00:23 left
  Indexing    |█████████████▊                |  46%  01:24 elapsed, ~01:35 left  video 11/26: clip_15.mov
```

Progress is saved after every video and every 30 minutes of footage, so Ctrl+C is safe: the next run
continues, and the part already done is searchable in the meantime. Without a terminal (cron, `nohup`),
the first run needs `--model-size`, so a large download never starts unasked.

## Usage

### Interactive search

```bash
video-grep FOLDER
```

This indexes anything new, then gives you a `search>` prompt:

| Type | What happens |
|---|---|
| anything else | Searches for it |
| a result number | Opens the video at that moment, or shows its path on a headless machine |
| `save 2` | Saves that moment as a clip, usually in a few seconds |
| `save 2 exact` | Saves a precise cut: slower, and without audio |
| `q` | Quits |

The up arrow brings back earlier searches. On a headless machine it looks like this:

```
search> 1
  File:   /data/footage/cam2/2026-10-21_2300.mp4
  Moment: 14:36 - 14:42
  Type "save 1" to save just this part, or "save 1 exact" for a precise cut.

search> save 1
  Saved /home/you/2026-10-21_2300_at_14-36.mp4 (4.2 MB)
```

### Other commands

```bash
video-grep index FOLDER                # build or update the index, then exit
video-grep index FOLDER --background   # keep indexing after you disconnect
video-grep search FOLDER "dog at gate" # one search, then exit
video-grep clip VIDEO --start 14:36    # save 6 seconds from that time (or --end 14:50)
video-grep status FOLDER               # what's indexed, waiting or failed
video-grep doctor                      # check Python, ffmpeg, GPU, RAM, disk and models
```

`search` behaves like `grep`, so it works in scripts: results go to stdout, and it exits with 0 if
something matched, 1 if nothing did, and 2 on an error. It never indexes; it tells you if videos
are waiting.

### Long indexing over SSH

```bash
video-grep index /data/footage --background
video-grep status /data/footage     # check on it, even after reconnecting
```

The job keeps running if your connection drops. You can search what's finished while it runs, and
`kill` stops it without losing finished work. Before indexing starts you see the footage length and
the disk space the index needs; the time left appears once the first clips are done.

### Narrowing a search

These work with both the shell and `search`:

```bash
video-grep FOLDER --path cam2                # one subfolder, or a pattern like "*.mp4"
video-grep FOLDER --after 2026-10-01 --before 2026-10-21
video-grep FOLDER -k 10                      # up to 10 results (default 5)
video-grep FOLDER --per-video 1              # one moment per video (default: up to 3)
video-grep FOLDER --min-score 0.62           # show less certain results too
```

Dates use each file's modified time, which usually matches the recording time but changes if a
file is edited, and may change when it's copied.

### Other options

```bash
video-grep FOLDER --model-size small    # use another size for this run (each keeps its own index)
video-grep FOLDER --no-open             # never launch a player (automatic over SSH / no display)
video-grep FOLDER --out-dir ~/clips     # where `save` puts clips
video-grep FOLDER --index-dir /some/dir # keep the index outside the video folder
video-grep FOLDER --reindex             # rebuild the index from scratch
video-grep FOLDER --retry-failed        # try again on videos that could not be read
video-grep FOLDER --prune               # drop index data for missing videos, even if many are gone
```

A folder really named after a command needs a path: `video-grep ./index`.

### Reading the results

Each result shows how close the match is:

| Label | Score | What it means in testing |
|---|---|---|
| **strong match** | 0.70 and up | None of 24 searches for absent subjects reached this |
| **possible match** | 0.67 to 0.70 | Worth checking: 6 of 24 absent subjects reached this level |
| hidden | below 0.67 | You see `No matches found` if nothing is left |

Short or vague searches score lower, so "blue lights at a concert" works better than "blue light".
`--min-score` changes the cut-off. These levels come from a small test set with the High model, so
**check a match before relying on it**. See [Limitations](#limitations).

## Model sizes

All three are [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2); only the format
and precision differ. Audio is never used. Medium and Small download only the vision and text parts;
High's single weights file also contains the unused audio part (about 600 MB of its download).

| Size | Download | Runs on | Indexing speed* | Peak RAM | Embedding similarity to High** |
|---|---|---|---|---|---|
| **High** on GPU (default) | ~1.5 GB | Apple Metal / NVIDIA CUDA | ~6x faster than real time | not measured | reference |
| High on CPU | ~1.5 GB | CPU fallback | ~1.5x faster than real time | ~7.3 GB | reference |
| Medium (ONNX 8-bit) | ~550 MB | CPU, or NVIDIA via `onnxruntime-gpu` | ~real time | not measured | ~99.9% |
| Small (ONNX 4-bit) | ~320 MB | CPU, or NVIDIA via `onnxruntime-gpu` | ~2.5x slower than real time | ~3.3 GB | ~98% |

\* Measured on an Apple M5 MacBook with low-resolution test clips. High-resolution footage takes
longer to decode, and slower hardware will be much slower. "6x faster than real time" means 1 hour of
footage indexes in ~10 minutes.
\*\* Cosine similarity of embeddings on a few sample clips and queries. This is not the same as search
accuracy, which has only been measured for High.

The download is only part of the disk use. The Python packages take about 1 GB on macOS, and on Linux
the default `torch` install also pulls in NVIDIA packages (roughly 2.7 GB), even on a machine without
a GPU.

Models are stored in `~/.cache/video-grep/models/` (override with `VIDEO_GREP_HOME`).

## How it works

1. Each video is decoded with ffmpeg into 6-second clips, 8 frames per clip.
2. Each clip is embedded with EmbeddingGemma 2, a multimodal model that places video and text
   in the same vector space.
3. Vectors are stored next to the footage in `FOLDER/.video-grep/<size>/`, about 17 KB per minute
   of video.
4. A query is embedded the same way and compared to every clip (cosine similarity). The best
   clips are returned as moments, up to 3 per video by default.

Search is a brute-force dot product: ~25 ms for 30 videos on the test machine, and about 0.15 s for
1.4 million clips in a synthetic test. Memory is the larger limit, because all vectors are loaded
into RAM: about 44 MB for every 24 hours of footage indexed.

## Tested on

| | |
|---|---|
| Device | Apple MacBook, M5, 24 GB RAM |
| OS | macOS 27.0 |
| Python | 3.12.4 |
| Model sizes | High (GPU, Metal) fully evaluated; Small and Medium checked on sample clips only |
| Test set | 29 freely licensed clips (6–73 s) from Wikimedia Commons, 34 hand-labelled queries |
| Result (High) | right video ranked first for 97% of queries, top 5 for 100% |

**Not yet tested:** Linux (x86 or ARM), NVIDIA GPUs (`torch` CUDA or `onnxruntime-gpu`),
CPU-only machines running High, embedded boards (Jetson, Raspberry Pi), real CCTV footage, and
folders with thousands of videos. Reports are welcome.

## Limitations

- **Visual only.** Audio, speech and on-screen text are not searched, and there is no face or
  person identification, by design.
- **Short time window.** Each result reflects a 6-second clip. Events that only make sense over
  minutes ("the car that parked and left later") are not understood as a whole.
- **Generic concepts, not specific identities.** It finds "a red car", "a toddler on a bed", not
  "my car" or a particular person.
- **False matches and misses happen.** The score levels were set on 29 varied clips with 58 queries
  written while building this, and near-identical footage such as fixed CCTV angles may behave
  differently. At the default cut-off, 6 of 24 searches for absent subjects still returned possible
  matches. Don't treat a result, or its absence, as proof that an event did or didn't happen.
- **Small people and objects can be missed.** Frames are reduced to 384 pixels wide before embedding,
  so a distant person in a wide CCTV shot may be only a few pixels. Frames are also 0.75 s apart, so
  very quick events can fall between them.
- **Only known file extensions are read.** Most common video formats and DVR exports are covered.
- **The first index is the slow part.** Fast on a GPU, but on a CPU it can take as long as the
  footage itself, or longer. CPU-only machines and small boards will be slow.
- **Memory.** High on CPU peaked at about 7.3 GB of RAM and Small at about 3.3 GB, so High may not
  fit on small boards. The first-run menu takes this into account.
- **Index updates at startup only.** Videos added while the search prompt is open are picked up on
  the next run.
- **One-time internet access** is needed to download the model. Everything after that is offline.
- **English queries only so far.** The model is multilingual, but that hasn't been evaluated here.
- **QuickTime can't play WebM/MKV.** On macOS those open in the default app without jumping to
  the moment.

## Measuring quality on your own footage

`scripts/evaluate.py` runs labelled queries against an indexed folder and reports hit@1, hit@5
and MRR. `scripts/download_testdata.py` fetches the Wikimedia test set (videos are downloaded at
run time, not stored in this repo; licenses and authors are recorded in `ground_truth.json`).

```bash
python scripts/download_testdata.py ../video-grep-testdata
echo q | video-grep ../video-grep-testdata       # build the index
python scripts/evaluate.py ../video-grep-testdata
```

To evaluate your own videos, write a JSON file like `scripts/eval_queries.json` and pass `--queries`.

To run the automated tests (they use small generated videos and need no model):

```bash
pip install -e '.[dev]'
pytest
```

## Credits

- [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by Google (Apache 2.0).
- ONNX exports by [onnx-community](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX).
- Test clips from [Wikimedia Commons](https://commons.wikimedia.org/) contributors, under the
  licenses listed per clip.

## License

[Apache 2.0](LICENSE)
