"""Selective extraction from a zip served over HTTP with Range support."""

from __future__ import annotations

import os
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from sonic.dataset.remote_zip import RemoteZip, extract_many
from sonic.dataset.sources.fsd50k import classify


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler plus single-range 206 responses."""

    def do_GET(self):
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        start, end = 0, len(data) - 1
        if rng := self.headers.get("Range"):
            a, b = rng.removeprefix("bytes=").split("-")
            start, end = int(a), int(b) if b else len(data) - 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.wfile.write(data[start : end + 1])

    def log_message(self, *args):
        pass


@pytest.fixture
def served_zip(tmp_path: Path):
    files = {
        "audio/a.wav": os.urandom(5000),            # incompressible, stored
        "audio/b.wav": b"quiet " * 20000,           # compressible, deflated
        "other/c.txt": b"not wanted",
    }
    archive = tmp_path / "www" / "test.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("audio/a.wav", files["audio/a.wav"], compress_type=zipfile.ZIP_STORED)
        zf.writestr("audio/b.wav", files["audio/b.wav"], compress_type=zipfile.ZIP_DEFLATED)
        zf.writestr("other/c.txt", files["other/c.txt"])
    handler = partial(RangeHandler, directory=str(archive.parent))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/test.zip", files
    httpd.shutdown()


def test_lists_members_from_the_central_directory(served_zip):
    url, files = served_zip
    members = RemoteZip([url]).members()
    assert set(members) == set(files)
    assert members["audio/b.wav"].size == len(files["audio/b.wav"])


def test_extracts_only_selected_members(served_zip, tmp_path: Path):
    url, files = served_zip
    remote = RemoteZip([url])
    members = remote.members()
    wanted = [members["audio/a.wav"], members["audio/b.wav"]]
    out = tmp_path / "out"

    assert extract_many(remote, wanted, out) == []
    for name in ("audio/a.wav", "audio/b.wav"):
        assert (out / name).read_bytes() == files[name]
    assert not (out / "other").exists()
    # A second run skips files that are already there.
    assert extract_many(remote, wanted, out) == []


LABEL_MAP = {
    "_exclude_if_present": ["Telephone"],
    "_generic_labels": ["Alarm"],
    "Alarm": "Alarm or Siren",
    "Siren": "Alarm or Siren",
    "Vehicle_horn_and_car_horn_and_honking": "Vehicle Horn",
    "Screaming": "Panic Scream",
    "Shout": "Aggression",
}


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["Alarm"], ("Alarm or Siren", "")),
        (["Alarm", "Siren"], ("Alarm or Siren", "")),
        (["Alarm", "Vehicle_horn_and_car_horn_and_honking"], ("Vehicle Horn", "")),  # specific wins
        (["Alarm", "Telephone"], (None, "excluded")),
        (["Screaming", "Shout"], (None, "ambiguous")),
        (["Music"], (None, "unmapped")),
    ],
)
def test_fsd50k_classify(labels, expected):
    assert classify(labels, LABEL_MAP) == expected
