"""GTM export: only training audio, spread over recordings, manifest matches the files."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from sonic.gtm.export import PLAYBACK_SR, export, pick_segments


def test_pick_spreads_over_recordings():
    rows = [{"audio_id": f"A{i // 5}", "segment_id": str(i)} for i in range(50)]  # 10 clips x 5 segments
    picked = pick_segments(rows, 10, random.Random(0))
    assert len({r["audio_id"] for r in picked}) == 10  # one per recording first


def test_export_uses_training_segments_only(tmp_path: Path):
    seg, meta = tmp_path / "segments", tmp_path / "meta"
    meta.mkdir()
    rows = []
    for split in ("train", "test"):
        for i in range(4):
            rel = f"{split}/gunshot/GUN-{split}{i}_00.wav"
            (seg / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(seg / rel, np.full(16000, 0.1 if split == "train" else 0.9, np.float32), 16000)
            rows.append({"segment_id": f"GUN-{split}{i}_00", "audio_id": f"GUN-{split}{i}", "filename": rel,
                         "class_name": "Gunshot", "class_slug": "gunshot", "split": split, "source_dataset": "t"})
    with open(meta / "segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    counts = export(3, tmp_path / "gtm", seg, meta)
    audio, sr = sf.read(tmp_path / "gtm" / "gunshot.wav")
    with open(tmp_path / "gtm" / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))
    assert counts == {"Gunshot": 3} and sr == PLAYBACK_SR and len(audio) == 3 * PLAYBACK_SR
    assert all(m["split"] == "train" for m in manifest) and len(manifest) == 3
    assert np.abs(audio).max() < 0.5  # no test audio (0.9) leaked in
