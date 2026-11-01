import subprocess

import numpy as np
import pytest


class FakeEmbedder:
    """Deterministic stand-in for the model: a vector from each clip's average colour."""

    device = "cpu"
    calls = 0
    fail_after = None
    error = RuntimeError("out of memory")

    def embed_clips(self, clips, batch_size=4):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise self.error
        vecs = np.stack([np.resize(c.astype(np.float32).mean(axis=(0, 1, 2)) + 1, 8) for c in clips])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def make_video():
    def make(path, seconds=7):
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                        f"testsrc=duration={seconds}:size=64x64:rate=10", "-pix_fmt", "yuv420p", str(path)], check=True)
        return path
    return make
