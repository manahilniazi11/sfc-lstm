"""Quality checks on synthetic recordings with one known problem each."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from sonic.quality import assess, assess_file

SR = 16000
RNG = np.random.default_rng(0)


def event_in_noise(seconds=3.0, event_amp=0.3, noise_amp=0.003) -> np.ndarray:
    """A 0.5 s tone burst over quiet background noise: a clean, typical event."""
    y = RNG.standard_normal(int(seconds * SR)) * noise_amp
    t = np.arange(SR // 2) / SR
    y[SR : SR + SR // 2] += event_amp * np.sin(2 * np.pi * 800 * t)
    return y.astype(np.float32)


def issues_text(result) -> str:
    return " | ".join(result.issues)


def test_clean_event_is_good():
    r = assess(event_in_noise(), SR)
    assert r.grade == "Good", r.issues


def test_steady_tone_is_not_called_noisy():
    """A siren or hum is steady but tonal; only noise-like steady audio is 'excessive noise'."""
    t = np.arange(3 * SR) / SR
    r = assess((0.2 * np.sin(2 * np.pi * 600 * t)).astype(np.float32), SR)
    assert "noise" not in issues_text(r)


def test_silence_is_unusable():
    r = assess(np.zeros(2 * SR, dtype=np.float32), SR)
    assert r.grade == "Unusable" and "silence" in issues_text(r)


def test_too_short_is_unusable():
    r = assess(event_in_noise()[: int(0.2 * SR)], SR)
    assert r.grade == "Unusable" and "duration" in issues_text(r)


def test_heavy_clipping_is_poor():
    y = np.clip(event_in_noise(event_amp=3.0), -1, 1)
    r = assess(y, SR)
    assert r.grade == "Poor" and "clipping" in issues_text(r)


def test_single_full_scale_peak_is_not_clipping():
    y = event_in_noise()
    y[SR + 100] = 1.0
    assert "clipping" not in issues_text(assess(y, SR))


def test_very_quiet_recording_is_poor():
    r = assess(event_in_noise(event_amp=0.003, noise_amp=0.0003), SR)
    assert r.grade == "Poor" and "low signal" in issues_text(r)


def test_pure_noise_is_excessive_noise_unless_it_is_background():
    noise = (RNG.standard_normal(3 * SR) * 0.1).astype(np.float32)
    assert "excessive noise" in issues_text(assess(noise, SR))
    assert "noise" not in issues_text(assess(noise, SR, expect_event=False))


def test_interior_dropouts_are_missing_frames_but_padding_is_not():
    y = event_in_noise()
    for start in (4000, 20000, 36000):
        y[start : start + 800] = 0  # three 50 ms dropouts
    assert "missing audio frames" in issues_text(assess(y, SR))
    padded = np.concatenate([np.zeros(8000, dtype=np.float32), event_in_noise(), np.zeros(8000, dtype=np.float32)])
    assert "missing" not in issues_text(assess(padded, SR))


def test_low_sample_rate_is_flagged():
    y = event_in_noise()[: 3 * 8000]
    assert "sample rate" in issues_text(assess(y, 8000))


def test_undecodable_file_is_an_encoding_problem(tmp_path: Path):
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"RIFF\x00\x00not really a wav file")
    r = assess_file(bad)
    assert r.grade == "Unusable" and "encoding" in issues_text(r)


def test_noise_floor_is_estimated(tmp_path: Path):
    path = tmp_path / "ok.wav"
    sf.write(path, event_in_noise(noise_amp=0.01), SR)
    r = assess_file(path)
    assert -45 < r.metrics["noise_floor_dbfs"] < -35  # 0.01 RMS noise is -40 dBFS
