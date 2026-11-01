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
    1. cam2/2026-10-21_2300.mp4  (best moment at 14:36)
    2. cam2/2026-10-20_2300.mp4  (best moment at 3:12)
    3. cam1/2026-10-21_2300.mp4  (best moment at 41:05)

  Type a number (1-3) to see its full path and a command to save that clip, or type a new search.
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

- **Terminal-first.** Built for SSH sessions. On a headless machine, picking a result prints the
  file path, the time range, and a ready-to-run `ffmpeg` command that saves just that clip, so you
  can usually copy a few MB instead of the whole recording. Run long indexing inside `tmux` or
  `screen`: a dropped SSH session stops it.
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

```bash
# macOS: brew install python ffmpeg      Ubuntu/Debian: sudo apt install python3-venv ffmpeg
git clone https://github.com/madhusoodhans/video-grep.git
cd video-grep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .

video-grep /path/to/videos
```

On first run you choose a model size (Enter = **High**, recommended). It downloads once with a
progress bar, then the folder is indexed:

```
Choose a model size to download (needed once):

  1. High    ~ 1.5 GB   best quality, fastest (uses your GPU)  (recommended)
  2. Medium  ~ 550 MB   nearly the same as High, but indexing is several times slower (CPU)
  3. Small   ~ 320 MB   slightly lower quality, slowest indexing (CPU)

  Downloading |██████████████                |  47%  712M/1.42G  31.2MB/s, ~00:23 left
  Indexing    |█████████████▊                |  46%  01:24 elapsed, ~01:35 left  video 11/26: clip_15.mov
```

Indexing progress is saved after every video, so Ctrl+C is safe: the next run continues, restarting
only the video that was in progress.

## Usage

```bash
video-grep FOLDER                       # index (if needed), then search interactively
video-grep FOLDER --model-size small    # switch model size (each size keeps its own index)
video-grep FOLDER --no-open             # never launch a player (automatic over SSH / no display)
video-grep FOLDER --reindex             # rebuild the index from scratch
video-grep FOLDER -k 10                 # show up to 10 results
```

Picking a result on a headless machine:

```
search> 1
  File:   /data/footage/cam2/2026-10-21_2300.mp4
  Moment: 14:36 - 14:42
  Save just this part (a few MB instead of the whole video):
    ffmpeg -ss 876 -i /data/footage/cam2/2026-10-21_2300.mp4 -t 6 -c copy 2026-10-21_2300_at_14-36.mp4
```

Results below a similarity of 0.62 are hidden, and if nothing passes you get `No matches found`.
This filter is weak: **a search for something that isn't in your footage can still return results,
so check a match before relying on it.** See [Limitations](#limitations).

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
4. A query is embedded the same way and compared to every clip (cosine similarity). Each video is
   ranked by its best clip, which also gives the timestamp.

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
- **False matches happen.** The 0.62 cut-off doesn't reliably reject searches for things that aren't
  in your footage: in a test on 29 varied clips, most such queries still returned results. The 97%
  figure above measures whether the right video ranks first when it exists, not whether the tool can
  tell that something is absent. Don't treat a result, or its absence, as proof that an event did or
  didn't happen.
- **One moment per video.** Each file returns only its single best moment, so a long recording with
  several matching events shows just one of them.
- **Long recordings are not a good fit yet.** A video is saved only when it finishes, so stopping
  or losing the session partway through a long file restarts that file.
- **The index must live inside the video folder,** so read-only folders don't work yet.
- **Some file types are skipped without warning,** including `.ts`, `.mpg`, `.flv` and `.dav`.
- **The first index is the slow part.** Fast on a GPU, but on a CPU it can take as long as the
  footage itself, or longer. CPU-only machines and small boards will be slow.
- **Memory.** High on CPU peaked at about 7.3 GB of RAM and Small at about 3.3 GB, so High may not
  fit on small boards. The first-run menu recommends High without checking for a GPU or RAM.
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

## Credits

- [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) by Google (Apache 2.0).
- ONNX exports by [onnx-community](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX).
- Test clips from [Wikimedia Commons](https://commons.wikimedia.org/) contributors, under the
  licenses listed per clip.

## License

[Apache 2.0](LICENSE)
