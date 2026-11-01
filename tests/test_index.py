import json

import numpy as np
import pytest

from video_grep import index as index_module
from video_grep.index import VideoIndex


def new_index(folder, root=None):
    return VideoIndex(folder, "test-model", "high", root)


def test_indexes_and_skips_unchanged(tmp_path, make_video, embedder):
    make_video(tmp_path / "a.mp4")
    index = new_index(tmp_path)
    assert index.update(embedder)["indexed"] == 1
    assert index.status().pending == []
    vectors, rows = index.load()
    assert len(vectors) == len(rows) == 2


def test_last_frames_are_kept(tmp_path, make_video, embedder):
    make_video(tmp_path / "a.mp4", seconds=6.8)
    index = new_index(tmp_path)
    index.update(embedder)
    assert index.files["a.mp4"]["segments"][-1][1] > 6.5


def test_unreadable_file_is_failed_then_retried(tmp_path, make_video, embedder):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    index = new_index(tmp_path)
    assert [f[0] for f in index.update(embedder)["failed"]] == ["bad.mp4"]
    assert index.status().failed == ["bad.mp4"]
    assert index.update(embedder)["failed"] == []  # not retried on its own
    make_video(tmp_path / "tmp.mp4")
    (tmp_path / "tmp.mp4").replace(bad)
    assert index.update(embedder, retry_failed=True)["indexed"] == 1
    assert "error" not in index.files["bad.mp4"]


@pytest.mark.parametrize("error", [RuntimeError("out of memory"), OSError(28, "No space left on device")])
def test_machine_problems_never_mark_videos_as_failed(tmp_path, make_video, embedder, error):
    make_video(tmp_path / "a.mp4")
    embedder.fail_after, embedder.error = 0, error
    index = new_index(tmp_path)
    with pytest.raises(type(error)):
        index.update(embedder)
    fresh = new_index(tmp_path)
    assert fresh.status().new == ["a.mp4"]


def test_file_deleted_during_indexing_does_not_stop_the_run(tmp_path, make_video, embedder, monkeypatch):
    make_video(tmp_path / "a.mp4")
    make_video(tmp_path / "b.mp4")
    real = index_module.iter_clips

    def vanish(path, *args):
        if path.name == "a.mp4":
            path.unlink()
            raise FileNotFoundError(path)
        return real(path, *args)

    monkeypatch.setattr(index_module, "iter_clips", vanish)
    index = new_index(tmp_path)
    stats = index.update(embedder)
    assert stats["indexed"] == 1 and stats["failed"] == []
    assert list(index.files) == ["b.mp4"]


def test_damaged_manifest_is_rebuilt(tmp_path, make_video, embedder):
    make_video(tmp_path / "a.mp4")
    index = new_index(tmp_path)
    index.update(embedder)
    index.manifest_path.write_text("")
    fresh = new_index(tmp_path)
    assert fresh.status().new == ["a.mp4"]
    assert fresh.update(embedder)["indexed"] == 1


@pytest.mark.parametrize("damage", ["delete", "wrong_rows", "garbage"])
def test_damaged_vectors_are_reindexed(tmp_path, make_video, embedder, damage):
    make_video(tmp_path / "a.mp4")
    index = new_index(tmp_path)
    index.update(embedder)
    path = index.vec_dir / index.files["a.mp4"]["vectors"]
    if damage == "delete":
        path.unlink()
    elif damage == "wrong_rows":
        np.save(path, np.zeros((1, 8), np.float16))
    else:
        path.write_bytes(b"junk")
    assert index.verify() == 1
    assert index.status().new == ["a.mp4"]


def test_interrupted_rebuild_never_mixes_old_and_new_vectors(tmp_path, make_video, embedder):
    make_video(tmp_path / "a.mp4")
    make_video(tmp_path / "b.mp4")
    new_index(tmp_path).update(embedder)
    index = VideoIndex(tmp_path, "other-model", "high")
    embedder.calls, embedder.fail_after = 0, 1
    with pytest.raises(RuntimeError):
        index.update(embedder)
    fresh = VideoIndex(tmp_path, "other-model", "high")
    assert list(fresh.files) == ["a.mp4"]  # redone with the new model; b's old vectors are gone
    assert fresh.status().new == ["b.mp4"]


def test_mass_removal_needs_prune(tmp_path, make_video, embedder):
    for i in range(6):
        make_video(tmp_path / f"{i}.mp4", seconds=1)
    index = new_index(tmp_path)
    index.update(embedder)
    for i in range(5):
        (tmp_path / f"{i}.mp4").unlink()
    assert index.update(embedder)["kept_missing"] == 5
    assert len(index.files) == 6
    assert index.update(embedder, prune=True)["removed"] == 5
    assert list(index.files) == ["5.mp4"]


def test_long_video_resumes_from_checkpoint(tmp_path, make_video, embedder, monkeypatch):
    make_video(tmp_path / "long.mp4", seconds=30)
    monkeypatch.setattr(index_module, "CHECKPOINT_CLIPS", 2)
    embedder.fail_after = 1  # first batch of 4 clips is saved, the second call dies
    index = new_index(tmp_path)
    with pytest.raises(RuntimeError):
        index.update(embedder)

    interrupted = new_index(tmp_path)
    assert interrupted.status().partial == ["long.mp4"]
    assert len(interrupted.load()[1]) == 4  # the finished part is already searchable

    embedder.fail_after, seen = None, []
    real = embedder.embed_clips
    embedder.embed_clips = lambda clips: (seen.append(len(clips)), real(clips))[1]
    assert interrupted.update(embedder)["indexed"] == 1
    assert sum(seen) == 1  # only the fifth clip was left to do
    segments = interrupted.files["long.mp4"]["segments"]
    assert segments == [[i * 6.0, (i + 1) * 6.0] for i in range(5)]
    assert "partial" not in interrupted.files["long.mp4"]


def test_truncated_video_gets_a_warning(tmp_path, make_video, embedder):
    video = make_video(tmp_path / "a.mkv", seconds=40)
    video.write_bytes(video.read_bytes()[:video.stat().st_size // 3])
    index = new_index(tmp_path)
    index.manifest["files"] = {}
    real = index_module.probe_duration
    index_module.probe_duration = lambda p: 40.0
    try:
        stats = index.update(embedder)
    finally:
        index_module.probe_duration = real
    assert stats["indexed"] == 1 and "could be read" in stats["warnings"][0][1]


def test_index_can_live_outside_a_read_only_folder(tmp_path, make_video, embedder):
    footage = tmp_path / "footage"
    footage.mkdir()
    make_video(footage / "a.mp4")
    footage.chmod(0o555)
    try:
        index = new_index(footage, tmp_path / "elsewhere")
        assert index.update(embedder)["indexed"] == 1
        assert not (footage / ".video-grep").exists()
        assert json.loads(index.manifest_path.read_text())["files"]["a.mp4"]["segments"]
    finally:
        footage.chmod(0o755)


def test_second_indexer_is_refused(tmp_path, make_video, embedder):
    make_video(tmp_path / "a.mp4")
    index = new_index(tmp_path)
    index.vec_dir.mkdir(parents=True)
    with index._locked():
        with pytest.raises(RuntimeError, match="already indexing"):
            new_index(tmp_path).update(embedder)
