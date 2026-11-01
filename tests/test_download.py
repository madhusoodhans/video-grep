import hashlib
import http.server
import threading
from dataclasses import replace

import pytest
from tqdm import tqdm

from video_grep import models

BODY = b"model weights " * 1000
SHA = hashlib.sha256(BODY).hexdigest()


class Server(http.server.BaseHTTPRequestHandler):
    truncate_times = 0
    body = BODY

    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-")[6:-1])
        if start >= len(self.body):
            self.send_response(416)
            self.end_headers()
            return
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(self.body) - start))
        self.end_headers()
        data = self.body[start:]
        if Server.truncate_times:
            Server.truncate_times -= 1
            data = data[:len(data) // 2]
        self.wfile.write(data)


@pytest.fixture
def url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/file"
    server.shutdown()
    Server.truncate_times = 0
    Server.body = BODY


def fetch(url, dest, sha=SHA):
    with tqdm(disable=True) as bar:
        models._download_file(url, dest, len(BODY), sha, bar)


def test_download_and_verify(url, tmp_path):
    fetch(url, tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == BODY


def test_dropped_connection_is_not_accepted_and_resumes(url, tmp_path):
    Server.truncate_times = 1
    with pytest.raises(models.DownloadError, match="incomplete"):
        fetch(url, tmp_path / "f")
    assert not (tmp_path / "f").exists()
    fetch(url, tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == BODY


def test_full_part_file_does_not_cause_416(url, tmp_path):
    (tmp_path / "f.part").write_bytes(BODY)
    fetch(url, tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == BODY


def test_oversized_part_file_restarts(url, tmp_path):
    (tmp_path / "f.part").write_bytes(BODY + b"extra")
    fetch(url, tmp_path / "f")
    assert (tmp_path / "f").read_bytes() == BODY


def test_bad_checksum_is_rejected_and_removed(url, tmp_path):
    with pytest.raises(models.DownloadError, match="checksum"):
        fetch(url, tmp_path / "f", sha="0" * 64)
    assert not (tmp_path / "f").exists() and not (tmp_path / "f.part").exists()


def test_retries_then_succeeds(url, tmp_path):
    Server.truncate_times = 2
    models.time.sleep, real = (lambda s: None), models.time.sleep
    try:
        models._with_retries(lambda: fetch(url, tmp_path / "f"))
    finally:
        models.time.sleep = real
    assert (tmp_path / "f").read_bytes() == BODY


@pytest.mark.parametrize("gpu,ram,expected", [
    (True, 8, "high"), (False, 16, "high"), (False, 8, "small"),
])
def test_recommended_size(gpu, ram, expected):
    assert models.recommended_size(gpu, ram) == expected


def test_no_download_without_a_terminal_or_a_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(models.sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit, match="--model-size"):
        models.ensure_model(None)


def test_model_size_flag_does_not_change_the_saved_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(models, "HOME", tmp_path)
    models.save_size("high")
    small = models.SIZES["small"]
    monkeypatch.setitem(models.SIZES, "small", replace(small, repo="x"))
    monkeypatch.setattr(models.ModelSize, "is_downloaded", lambda self: True)
    models.ensure_model("small")
    assert models.saved_size() == "high"
