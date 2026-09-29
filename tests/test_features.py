"""Feature tests: shapes, and that each feature responds to what it is meant to measure."""

from __future__ import annotations

import numpy as np
import pytest

from sonic.features import extract, feature_names, load_feature_settings

SR = 16000
S = load_feature_settings()
NAMES = feature_names(S)


def tone(freq: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(SR) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def white_noise(amp: float = 0.1, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(SR) * amp).astype(np.float32)


def feat(y: np.ndarray, name: str) -> float:
    return float(extract(y, SR, S).vector[NAMES.index(name)])


def test_vector_and_names_line_up():
    f = extract(white_noise(), SR, S)
    assert f.vector.shape == (len(NAMES),) == (123,)
    assert len(set(NAMES)) == len(NAMES)
    assert np.isfinite(f.vector).all()


def test_log_mel_shape_is_fixed_for_one_second():
    assert extract(white_noise(), SR, S).log_mel.shape == (S.n_mels, 63)


def test_extraction_is_deterministic():
    y = white_noise()
    assert np.array_equal(extract(y, SR, S).vector, extract(y, SR, S).vector)


def test_noise_is_flatter_and_crosses_zero_more_than_a_tone():
    assert feat(white_noise(), "flatness_mean") > 10 * feat(tone(440), "flatness_mean")
    assert feat(white_noise(), "zcr_mean") > feat(tone(440), "zcr_mean")


def test_centroid_and_rolloff_follow_the_pitch():
    low, high = tone(300), tone(3000)
    assert feat(high, "centroid_mean") > feat(low, "centroid_mean")
    assert feat(high, "rolloff_mean") > feat(low, "rolloff_mean")
    assert feat(high, "centroid_mean") == pytest.approx(3000, rel=0.1)


def test_chroma_finds_the_pitch_class():
    a4 = extract(tone(440), SR, S).vector  # A
    chroma = [a4[NAMES.index(f"chroma_{i}_mean")] for i in range(12)]
    assert int(np.argmax(chroma)) == 9  # librosa orders C, C#, D, ..., A is index 9


def test_onset_and_rms_peak_detect_an_impulse():
    steady = tone(440)
    click = steady.copy()
    click[8000:8080] += 0.8
    assert feat(click, "onset_max") > 3 * feat(steady, "onset_max")
    assert feat(click, "rms_max") > feat(steady, "rms_max")


def test_mel_bands_show_where_the_energy_is():
    v = extract(tone(200), SR, S).vector
    bands = [v[NAMES.index(f"mel_band_{i}_mean")] for i in range(S.mel_summary_bands)]
    assert int(np.argmax(bands)) < 4  # a 200 Hz tone lives in the lowest bands
