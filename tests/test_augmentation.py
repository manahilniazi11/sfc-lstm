"""Augmentation tests: each transform alone, the chain, copy planning and a small build."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sonic.augmentation import transforms as T
from sonic.augmentation.build import build_augmented, plan_copies
from sonic.augmentation.chain import augment, load_config
from sonic.preprocessing.settings import AudioSettings
from sonic.preprocessing.steps import rms_dbfs

SR = 16000
S = AudioSettings()


def tone(freq: float = 440.0, amp: float = 0.1, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(seed: int = 0, amp: float = 0.05) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(SR) * amp).astype(np.float32)


# --- transforms ------------------------------------------------------------------

def test_every_transform_keeps_the_segment_length():
    rng = np.random.default_rng(0)
    y = tone()
    outputs = [
        T.time_shift(y, 2000), T.time_shift(y, -2000), T.pitch_shift(y, SR, 2), T.time_stretch(y, 0.85),
        T.time_stretch(y, 1.15), T.gain(y, -6), T.reverb(y, SR, 0.5, 0.3, rng),
        T.distance(y, SR, 3000, 0.3, rng), T.device(y, SR, (300, 3400), 0.5),
        T.mix_background(y, noise(), 10, rms_dbfs(y)),
    ]
    assert all(out.shape == y.shape and out.dtype == np.float32 for out in outputs)


def test_background_is_mixed_at_the_requested_snr():
    y, n = tone(), noise()
    mixed = T.mix_background(y, n, snr_db=10, reference_dbfs=rms_dbfs(y))
    assert rms_dbfs(y) - rms_dbfs(mixed - y) == pytest.approx(10, abs=0.1)


def test_quieter_source_ends_up_closer_to_the_noise():
    """Gain and distance lower the SNR, like a sound further away in the same room."""
    y, n = tone(), noise()
    ref = rms_dbfs(y)
    quiet = T.gain(y, -6)
    mixed = T.mix_background(quiet, n, snr_db=10, reference_dbfs=ref)
    assert rms_dbfs(quiet) - rms_dbfs(mixed - quiet) == pytest.approx(4, abs=0.1)


def test_time_shift_moves_content():
    y = np.zeros(SR, dtype=np.float32)
    y[1000] = 1.0
    assert np.argmax(T.time_shift(y, 500)) == 1500
    assert np.argmax(T.time_shift(y, -500)) == 500


def test_device_removes_frequencies_outside_the_band():
    low = tone(freq=100, amp=0.5)
    assert rms_dbfs(T.device(low, SR, (300, 3400), 0.9)) < rms_dbfs(low) - 15


def test_reverb_keeps_the_level():
    y = tone()
    assert rms_dbfs(T.reverb(y, SR, 0.6, 0.4, np.random.default_rng(1))) == pytest.approx(rms_dbfs(y), abs=0.5)


# --- chain ---------------------------------------------------------------------------

def test_chain_output_looks_like_a_preprocessed_segment():
    cfg = load_config()
    out, description = augment(tone(), noise(), cfg, S, np.random.default_rng(3))
    assert out.shape == (SR,)
    assert rms_dbfs(out) == pytest.approx(S.target_rms_dbfs, abs=1.5)
    assert np.count_nonzero(out == 0) == 0
    assert "noise(" in description  # background noise is always part of the chain


def test_chain_is_reproducible_from_the_seed():
    cfg = load_config()
    a = augment(tone(), noise(), cfg, S, np.random.default_rng(7))
    b = augment(tone(), noise(), cfg, S, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


# --- planning and build ------------------------------------------------------------------

def test_plan_tops_up_to_target_and_respects_max_copies():
    rows = [{"segment_id": str(i)} for i in range(10)]
    rng = np.random.default_rng(0)
    assert len(plan_copies(rows, target=25, max_copies=5, rng=rng)) == 15
    assert len(plan_copies(rows, target=100, max_copies=3, rng=rng)) == 30  # capped at 3 per segment
    assert plan_copies(rows, target=5, max_copies=5, rng=rng) == []  # already big enough


def test_build_augments_only_training_and_keeps_audio_ids(tmp_path: Path):
    seg_dir, meta_dir = tmp_path / "segments", tmp_path / "meta"
    meta_dir.mkdir()
    rows = []
    specs = [("Gunshot", "gunshot", "train", 3), ("Gunshot", "gunshot", "test", 2),
             ("Background Noise", "background_noise", "train", 4)]
    for class_name, slug, split, n in specs:
        for i in range(n):
            sid = f"{slug[:3].upper()}-{split[:2]}{i}_00"
            rel = f"{split}/{slug}/{sid}.wav"
            (seg_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(seg_dir / rel, tone(300 + 40 * i) if slug == "gunshot" else noise(i), SR)
            rows.append({"segment_id": sid, "audio_id": sid[:-3], "filename": rel, "class_name": class_name,
                         "class_slug": slug, "split": split, "source_dataset": "test"})
    with open(meta_dir / "segments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    result = build_augmented(seg_dir, tmp_path / "aug", meta_dir, workers=1)
    with open(meta_dir / "augmented_segments.csv", newline="") as f:
        aug = list(csv.DictReader(f))

    parents = {r["segment_id"]: r for r in rows}
    assert aug and all(r["split"] == "train" for r in aug)
    assert all(parents[r["parent_segment_id"]]["split"] == "train" for r in aug)
    assert all(r["audio_id"] == parents[r["parent_segment_id"]]["audio_id"] for r in aug)
    assert all((tmp_path / "aug" / r["filename"]).is_file() for r in aug)
    # Background clips are never mixed with themselves.
    assert all(r["noise_source"] != r["parent_segment_id"] for r in aug)
    assert result["copies"]["Gunshot"] == min(3 * 5, 1500 - 3)
