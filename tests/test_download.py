"""The downloader must resume dropped connections and never accept a partial file."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sonic.dataset import download

DATA = bytes(range(256)) * 4000  # ~1 MB


class FlakyHandler(BaseHTTPRequestHandler):
    """Serves DATA with Range support; the first response is cut off halfway."""

    requests = 0
    cut_all = False

    def do_GET(self):
        type(self).requests += 1
        start = 0
        if rng := self.headers.get("Range"):
            start = int(rng.removeprefix("bytes=").rstrip("-"))
        body = DATA[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).cut_all or type(self).requests == 1:
            body = body[: len(body) // 2]  # simulate the VPN dropping the connection
            self.close_connection = True
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    FlakyHandler.requests = 0
    FlakyHandler.cut_all = False
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/file.bin"
    httpd.shutdown()


def test_dropped_connection_is_resumed(server, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    target = download._download(server, tmp_path / "file.bin")
    assert target.read_bytes() == DATA
    assert FlakyHandler.requests == 2  # second request resumed with a Range header
    assert not (tmp_path / "file.bin.part").exists()


def test_extract_keeps_only_matching_members(tmp_path: Path):
    import zipfile

    archive = tmp_path / "6_dB_valve.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("valve/id_00/abnormal/00000000.wav", b"x")
        zf.writestr("valve/id_00/normal/00000000.wav", b"x")
    download._extract(archive, tmp_path / "out", r"/abnormal/")
    assert (tmp_path / "out/valve/id_00/abnormal/00000000.wav").exists()
    assert not (tmp_path / "out/valve/id_00/normal").exists()


def test_gives_up_without_renaming_partial_file(server, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    monkeypatch.setattr(FlakyHandler, "cut_all", True)
    with pytest.raises(RuntimeError, match="incomplete"):
        download._download(server, tmp_path / "file.bin", attempts=1)
    assert not (tmp_path / "file.bin").exists()
    assert (tmp_path / "file.bin.part").exists()
