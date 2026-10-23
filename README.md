# video-grep-cli

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

- **Terminal-first.** Works fully over SSH. On a headless machine, picking a result prints the
  file path, the time range, and a ready-to-run `ffmpeg` command that saves just that clip, so you
  copy a few MB instead of the whole recording.
- **Searches the content, not the file names.** Describe actions, activities, scenes or objects in
  plain English. No tags, captions, metadata or manual labelling needed.
- **Local and private.** After a one-time model download, everything runs offline on the machine.
  No video, frame or query leaves it.
- **Incremental.** Re-running only processes new or changed videos.
- **Cross-platform.** macOS and Linux, GPU if available, CPU otherwise.
- **Choose your footprint.** Three model sizes, from 1.5 GB down to 320 MB.

On a desktop, picking a result opens the video at the matching moment instead
(QuickTime on macOS, the default player on Linux).

## Quick start

Requires Python 3.10+ and ffmpeg.

```bash
# macOS: brew install python ffmpeg      Ubuntu/Debian: sudo apt install python3-venv ffmpeg
git clone https://github.com/madhusoodhans/video-grep-cli.git
cd video-grep-cli
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
  2. Medium  ~ 550 MB   same quality as High, but indexing is several times slower (CPU)
  3. Small   ~ 320 MB   slightly lower quality, slowest indexing (CPU)

  Downloading |██████████████                |  47%  712M/1.42G  31.2MB/s, ~00:23 left
  Indexing    |█████████████▊                |  46%  01:24 elapsed, ~01:35 left  video 11/26: clip_15.mov
```

Indexing progress is saved after every video, so Ctrl+C is safe; the next run continues.

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

Results below a similarity of 0.62 are hidden. If nothing passes, you get
`No matches found`, not a list of guesses.

## Model sizes

All three are [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2); only the format
and precision differ. Only the vision and text parts are downloaded; audio is never used.

| Size | Download | Runs on | Indexing speed* | Quality vs High** |
|---|---|---|---|---|
| **High** (default) | ~1.5 GB | GPU (Apple Metal / NVIDIA CUDA), CPU fallback | ~6x faster than real time | reference |
| Medium (ONNX 8-bit) | ~550 MB | CPU, or NVIDIA via `onnxruntime-gpu` | ~real time | ~99.9% |
| Small (ONNX 4-bit) | ~320 MB | CPU, or NVIDIA via `onnxruntime-gpu` | ~2.5x slower than real time | ~98% |

\* Measured on an Apple M5 MacBook. "6x faster than real time" means 1 hour of footage indexes in ~10 minutes.
\*\* Cosine similarity of embeddings against High on sample clips and queries.

Models are stored in `~/.cache/video-grep/models/` (override with `VIDEO_GREP_HOME`).

## How it works

1. Each video is decoded with ffmpeg into 6-second clips, 8 frames per clip.
2. Each clip is embedded with EmbeddingGemma 2, a multimodal model that places video and text
   in the same vector space.
3. Vectors are stored next to the footage in `FOLDER/.video-grep/<size>/`, about 10 KB per minute
   of video.
4. A query is embedded the same way and compared to every clip (cosine similarity). Each video is
   ranked by its best clip, which also gives the timestamp.

Search is a brute-force dot product: ~25 ms for 30 videos on the test machine, comfortably
fast for thousands of videos.

## Tested on

| | |
|---|---|
| Device | Apple MacBook, M5, 24 GB RAM |
| OS | macOS 27.0 |
| Python | 3.12.4 |
| Model sizes | High (GPU, Metal), Medium and Small (CPU) |
| Test set | 29 freely licensed clips (5–120 s) from Wikimedia Commons, 34 hand-labelled queries |
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
- **The match cut-off is rough.** 0.62 was tuned on a small, easy test set. Real footage, especially
  many near-identical clips like fixed CCTV angles, may need a different value.
- **The first index is the slow part.** Fast on a GPU, but Medium and Small on a CPU take roughly
  as long as the footage itself, or longer. CPU-only machines and small boards will be slow.
- **Memory.** Expect a few GB of RAM while indexing (about 3.3 GB observed for Small on CPU).
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
