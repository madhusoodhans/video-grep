import numpy as np

from video_grep.search import Searcher

ROWS = [("cam.mp4", 0.0, 6.0), ("cam.mp4", 6.0, 12.0), ("cam.mp4", 60.0, 66.0), ("cam.mp4", 120.0, 126.0),
        ("other.mp4", 0.0, 6.0)]
SCORES = np.array([0.9, 0.8, 0.7, 0.6, 0.5], np.float32)


def search(**kwargs):
    return Searcher(np.eye(5, dtype=np.float32) * SCORES, ROWS).search(np.ones(5, np.float32), **kwargs)


def test_one_result_per_video_by_default():
    assert [(r.video, r.start) for r in search(k=5)] == [("cam.mp4", 0.0), ("other.mp4", 0.0)]


def test_several_moments_from_one_recording():
    assert [r.start for r in search(k=3, per_video=3)] == [0.0, 60.0, 120.0]


def test_touching_clips_count_as_one_moment():
    assert 6.0 not in [r.start for r in search(k=5, per_video=5)]


def test_empty_index():
    assert Searcher(np.zeros((0, 0)), []).search(np.ones(5)) == []
