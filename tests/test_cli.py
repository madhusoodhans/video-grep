import os
import subprocess
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import FakeEmbedder
from video_grep import cli
from video_grep.frames import VideoError, clip_filename, parse_time, save_clip
from video_grep.index import VideoIndex
from video_grep.search import Result


class QueryEmbedder(FakeEmbedder):
    """Every query vector points the same way, so scores come from the stored vectors."""

    def __init__(self, dim=8):
        self.dim = dim

    def embed_query(self, text):
        return np.ones(self.dim, np.float32) / np.sqrt(self.dim)


@pytest.fixture
def indexed(tmp_path, make_video, monkeypatch, embedder):
    folder = tmp_path / "videos"
    (folder / "cam2").mkdir(parents=True)
    make_video(folder / "a.mp4")
    make_video(folder / "cam2" / "b.mp4")
    size = SimpleNamespace(key="high", title="High", index_id="m", is_downloaded=lambda: True)
    monkeypatch.setattr("video_grep.models.ensure_model", lambda requested: size)
    monkeypatch.setattr(cli, "load_model", lambda s: QueryEmbedder())
    monkeypatch.setitem(cli.SIZES, "high", size)
    VideoIndex(folder, "m", "high").update(embedder)
    return folder


def run(*argv):
    with pytest.raises(SystemExit) as e:
        cli.main(list(argv))
    return e.value.code


def test_folder_alone_means_the_shell(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_shell", lambda args: called.append(args.folder))
    assert run("/some/folder") == 0
    assert called == ["/some/folder"]


def test_folder_named_like_a_command_works_as_a_path(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "cmd_shell", lambda args: called.append(args.folder))
    run("./index")
    assert called == ["./index"]


@pytest.mark.parametrize("text,seconds", [("876", 876), ("14:36", 876), ("1:02:03", 3723), ("12.5", 12.5)])
def test_parse_time(text, seconds):
    assert parse_time(text) == seconds


@pytest.mark.parametrize("text", ["abc", "1:2:3:4", "-5", ""])
def test_parse_time_rejects_nonsense(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_search_prints_results_and_exits_zero(indexed, capsys):
    assert run("search", str(indexed), "anything", "--min-score", "0") == 0
    out = capsys.readouterr().out
    assert "a.mp4" in out and "match" in out and "Top" not in out


def test_search_with_no_match_exits_one_and_keeps_stdout_empty(indexed, capsys):
    assert run("search", str(indexed), "anything", "--min-score", "2") == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "No matches found" in captured.err


def test_errors_exit_two(indexed, tmp_path, capsys):
    assert run("search", str(tmp_path / "nowhere"), "x") == 2
    assert "Not a folder" in capsys.readouterr().err


def test_search_on_an_unindexed_folder_says_what_to_do(indexed, tmp_path, capsys):
    other = tmp_path / "other"
    other.mkdir()
    assert run("search", str(other), "x") == 2
    assert "video-grep index" in capsys.readouterr().err


def test_search_does_not_change_the_index(indexed, make_video):
    make_video(indexed / "new.mp4")
    manifest = indexed / ".video-grep" / "high" / "manifest.json"
    before = manifest.read_bytes()
    run("search", str(indexed), "x", "--min-score", "0")
    assert manifest.read_bytes() == before


def test_path_filter(indexed, capsys):
    assert run("search", str(indexed), "x", "--min-score", "0", "--path", "cam2") == 0
    out = capsys.readouterr().out
    assert "cam2/b.mp4" in out and "a.mp4" not in out


def test_date_filters(indexed, capsys):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert run("search", str(indexed), "x", "--min-score", "0", "--before", tomorrow) == 0
    capsys.readouterr()
    assert run("search", str(indexed), "x", "--after", tomorrow) == 2
    assert "match those filters" in capsys.readouterr().err


def test_filter_needs_no_exact_date_format_guessing():
    with pytest.raises(SystemExit):
        cli.main(["search", ".", "x", "--after", "21/10/2026"])


def test_status(indexed, make_video, capsys):
    bad = indexed / "bad.mp4"
    bad.write_bytes(b"nope")
    make_video(indexed / "new.mp4")
    index = VideoIndex(indexed, "m", "high")
    index.files["bad.mp4"] = {"size": 4, "mtime": bad.stat().st_mtime, "error": "unreadable"}
    index._save()
    assert run("status", str(indexed)) == 0
    out = capsys.readouterr().out
    assert "2 video(s)" in out and "1 new" in out and "bad.mp4: unreadable" in out


def test_clip_saves_a_part(tmp_path, make_video, monkeypatch, capsys):
    video = make_video(tmp_path / "a.mp4", seconds=10)
    monkeypatch.chdir(tmp_path)
    assert run("clip", str(video), "--start", "0:02", "--exact") == 0
    saved = tmp_path / "a_at_0-02.mp4"
    assert saved.exists()
    length = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", saved],
                            capture_output=True, text=True).stdout
    assert 5.5 < float(length) < 6.5
    assert "Saved" in capsys.readouterr().out


def test_clip_past_the_end_is_refused(tmp_path, make_video, capsys):
    video = make_video(tmp_path / "a.mp4", seconds=2)
    assert run("clip", str(video), "--start", "5:00") == 2
    assert "past the end" in capsys.readouterr().err


def test_clip_never_overwrites(tmp_path, make_video, capsys):
    video = make_video(tmp_path / "a.mp4")
    out = tmp_path / "taken.mp4"
    out.write_bytes(b"precious")
    assert run("clip", str(video), "--start", "1", "-o", str(out)) == 2
    assert out.read_bytes() == b"precious"


def test_clip_filename_avoids_existing_files(tmp_path):
    (tmp_path / "a_at_14-36.mp4").write_bytes(b"x")
    assert clip_filename(Path("/v/a.mp4"), 876, tmp_path).name == "a_at_14-36_2.mp4"
    assert clip_filename(Path("/v/a.mp4"), 3723, tmp_path).name == "a_at_1-02-03.mp4"


@pytest.mark.parametrize("name", ["-dash.mp4", "cam1:23.mp4", "with space.mp4"])
def test_odd_output_names(tmp_path, make_video, monkeypatch, name):
    video = make_video(tmp_path / "source.mp4")
    monkeypatch.chdir(tmp_path)
    assert save_clip(video, 1, 2, Path(name)).exists()


def test_save_rejects_backwards_times(tmp_path, make_video):
    with pytest.raises(VideoError):
        save_clip(make_video(tmp_path / "a.mp4"), 5, 2, tmp_path / "o.mp4")


def shell(index, lines, monkeypatch, tmp_path, capsys, matches):
    args = SimpleNamespace(k=5, per_video=3, min_score=0.0, out_dir=str(tmp_path / "clips"))
    feed = iter(lines)

    def fake_input(prompt):
        try:
            return next(feed)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    searcher = SimpleNamespace(search=lambda vec, k, per_video: matches)
    cli.shell_loop(index, searcher, QueryEmbedder(), args)
    return capsys.readouterr().out


def test_shell_save_writes_the_clip(indexed, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "OPEN_VIDEOS", False)
    index = VideoIndex(indexed, "m", "high")
    out = shell(index, ["a thing", "save 1"], monkeypatch, tmp_path, capsys, [Result("a.mp4", 0.8, 0.0, 6.0)])
    assert "Saved" in out and (tmp_path / "clips" / "a_at_0-00.mp4").exists()
    assert '"save 1" to save that clip' in out


def test_shell_pick_on_a_headless_machine_shows_the_path(indexed, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "OPEN_VIDEOS", False)
    index = VideoIndex(indexed, "m", "high")
    out = shell(index, ["a thing", "1"], monkeypatch, tmp_path, capsys, [Result("a.mp4", 0.8, 0.0, 6.0)])
    assert str(indexed / "a.mp4") in out and "Moment: 0:00 - 0:06" in out


def test_shell_save_with_nothing_to_save(indexed, monkeypatch, tmp_path, capsys):
    out = shell(VideoIndex(indexed, "m", "high"), ["save 1"], monkeypatch, tmp_path, capsys, [])
    assert "No results to pick from" in out


def test_shell_word_save_is_still_a_search(indexed, monkeypatch, tmp_path, capsys):
    out = shell(VideoIndex(indexed, "m", "high"), ["save the whale"], monkeypatch, tmp_path, capsys,
                [Result("a.mp4", 0.8, 0.0, 6.0)])
    assert 'for "save the whale"' in out


def test_load_skips_damaged_entries_without_changing_anything(indexed):
    index = VideoIndex(indexed, "m", "high")
    (index.vec_dir / index.files["a.mp4"]["vectors"]).write_bytes(b"junk")
    _, rows = index.load()
    assert {row[0] for row in rows} == {"cam2/b.mp4"}
    assert "a.mp4" in index.files


def test_is_indexing_tracks_the_lock(indexed):
    index = VideoIndex(indexed, "m", "high")
    assert not index.is_indexing()
    with index._locked():
        assert index.is_indexing()
    assert not index.is_indexing()


def test_doctor_reports_a_missing_ffmpeg(monkeypatch, capsys):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr("video_grep.models.has_gpu", lambda: False)
    assert run("doctor") == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "ffmpeg" in out and "problem(s)" in out


def test_background_spawn_is_detached(indexed, monkeypatch, capsys):
    spawned = {}

    def fake_popen(cmd, **kwargs):
        spawned.update(kwargs, cmd=cmd)
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    index = VideoIndex(indexed, "m", "high")
    args = SimpleNamespace(index_dir=None, reindex=False, retry_failed=True, prune=False)
    cli.start_background(args, SimpleNamespace(key="high"), index)
    assert spawned["start_new_session"] is True and spawned["stdin"] == subprocess.DEVNULL
    assert "--model-size" in spawned["cmd"] and "--retry-failed" in spawned["cmd"]
    assert os.path.exists(index.dir / "index.log") and "tail -f" in capsys.readouterr().out
