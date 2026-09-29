"""Importing the team's own recordings."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonic.dataset.recordings import import_recordings, parse_name


def test_parse_full_and_partial_names():
    info = parse_name("ali_pleasehelp_bedroom_2m_redmi9_1")
    assert info == {"speaker": "ali", "notes": "phrase=pleasehelp", "environment": "bedroom",
                    "distance_m": "2", "device": "redmi9"}
    assert parse_name("sara-take3")["speaker"] == "sara-take3"


def test_import_adds_rows_grouped_by_speaker(tmp_path: Path):
    folder = tmp_path / "custom" / "help_request"
    folder.mkdir(parents=True)
    for name in ("ali_helpme_room_1m_phone_1.wav", "ali_helpme_room_3m_phone_2.wav"):
        sf.write(folder / name, np.zeros(16000, np.float32), 16000)
    import_recordings(tmp_path)
    stats = import_recordings(tmp_path)  # second run adds nothing
    with open(tmp_path / "custom" / "recordings.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert stats["added"] == 0 and len(rows) == 2
    assert {r["group"] for r in rows} == {"ali"} and {r["distance_m"] for r in rows} == {"1", "3"}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_phone_formats_are_converted(tmp_path: Path):
    folder = tmp_path / "custom" / "help_request"
    folder.mkdir(parents=True)
    wav = folder / "tmp.wav"
    sf.write(wav, (np.sin(np.arange(16000) / 5) * 0.3).astype(np.float32), 16000)
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(folder / "sara_help_hall_2m_iphone_1.m4a")], check=True)
    wav.unlink()
    stats = import_recordings(tmp_path)
    assert stats["converted"] == 1
    assert (folder / "sara_help_hall_2m_iphone_1.wav").exists()
    assert (tmp_path / "custom" / "_originals" / "help_request" / "sara_help_hall_2m_iphone_1.m4a").exists()
