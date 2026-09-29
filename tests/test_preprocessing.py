"""Preprocessing tests on synthetic signals with known properties."""

from __future__ import annotations

import numpy as np
import pytest

from sonic.preprocessing import steps
from sonic.preprocessing.pipeline import INFERENCE, TRAINING, preprocess
from sonic.preprocessing.settings import AudioSettings, load_settings

S = AudioSettings()  # 16 kHz, 1 s segments, 0.5 s hop


def tone(seconds: float, sr: int = 16000, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_config_file_loads():
    s = load_settings()
    assert s.sample_rate == 16000 and s.segment_samples == 16000


# --- steps ------------------------------------------------------------------------

def test_mono_averages_stereo_but_takes_first_mic_of_an_array():
    stereo = np.stack([np.ones(10), np.zeros(10)], axis=1)
    assert np.allclose(steps.to_mono(stereo), 0.5)
    array8 = np.stack([np.full(10, i, dtype=float) for i in range(8)], axis=1)
    assert np.allclose(steps.to_mono(array8), 0.0)  # channel 0, not the mean (3.5)


def test_resampling_removes_content_above_the_new_nyquist():
    y = tone(1.0, sr=44100, freq=12000)  # 12 kHz cannot exist at 16 kHz
    out = steps.resample(y, 44100, 16000)
    assert out.size == 16000
    assert steps.rms_dbfs(out) < -50


def test_trim_bounds_cut_digital_silence():
    y = np.concatenate([np.zeros(8000), tone(1.0), np.zeros(24000)]).astype(np.float32)
    start, end = steps.trim_bounds(y, top_db=40)
    assert abs(start - 8000) <= 512 and abs(end - 24000) <= 512


def test_normalize_hits_target_level_and_leaves_silence_alone():
    y = tone(1.0, amp=0.01)
    assert steps.rms_dbfs(steps.normalize(y, -20, -1, -60)) == pytest.approx(-20, abs=0.1)
    quiet = tone(1.0, amp=1e-5)
    assert np.array_equal(steps.normalize(quiet, -20, -1, -60), quiet)


def test_normalize_respects_peak_limit():
    spiky = np.zeros(16000, dtype=np.float32)
    spiky[100] = 0.5  # one click: RMS target would need a huge gain
    out = steps.normalize(spiky, -20, -1, -80)
    assert np.abs(out).max() <= 10 ** (-1 / 20) + 1e-6


def test_place_centres_or_randomises_a_short_clip():
    rng = np.random.default_rng(0)
    clip = np.ones(4000, dtype=np.float32)
    centred = steps.place(clip, 16000, rng, centred=True)
    assert np.flatnonzero(centred)[0] == 6000
    offsets = {int(np.flatnonzero(steps.place(clip, 16000, rng))[0]) for _ in range(20)}
    assert len(offsets) > 1


def test_short_clips_get_the_same_noise_floor_as_long_ones():
    """Padding must not reveal how loud or short the original clip was."""
    quiet_short = preprocess(tone(0.3, amp=0.005), 16000, S, INFERENCE)[0].samples
    loud_short = preprocess(tone(0.3, amp=0.9), 16000, S, INFERENCE)[0].samples
    # The padded region (first 0.3 s of a centred 0.3 s clip is padding) is pure noise floor.
    for seg in (quiet_short, loud_short):
        assert np.count_nonzero(seg == 0) == 0
        assert steps.rms_dbfs(seg[:4000]) == pytest.approx(S.pad_noise_dbfs, abs=1)


# --- pipeline -----------------------------------------------------------------------

def test_every_segment_has_the_same_format_whatever_the_source():
    sources = [
        (np.stack([tone(3, 44100)] * 2, axis=1), 44100),   # stereo 44.1 kHz
        (np.stack([tone(3, 16000)] * 8, axis=1), 16000),   # 8-mic array 16 kHz (MIMII-like)
        (tone(3, 96000), 96000),                           # mono 96 kHz
    ]
    for audio, sr in sources:
        for seg in preprocess(audio, sr, S, INFERENCE):
            assert seg.samples.shape == (16000,) and seg.samples.dtype == np.float32
            assert steps.rms_dbfs(seg.samples) == pytest.approx(-20, abs=0.5)


def test_inference_covers_the_whole_recording_in_order():
    segs = preprocess(tone(3.2), 16000, S, INFERENCE)
    starts = [s.start_s for s in segs]
    assert starts == sorted(starts)
    assert starts[0] == 0.0 and segs[-1].end_s == pytest.approx(3.2, abs=0.05)


def test_training_keeps_the_event_not_the_background():
    rng = np.random.default_rng(1)
    background = (rng.standard_normal(4 * 16000) * 0.001).astype(np.float32)  # about -60 dBFS
    y = background.copy()
    y[40000:44000] += tone(0.25, amp=0.8)  # a 0.25 s "gunshot" at 2.5 s
    segs = preprocess(y, 16000, S, TRAINING)
    assert len(segs) == 1
    assert segs[0].start_s <= 2.5 <= segs[0].end_s


def test_training_limits_segments_per_clip_and_avoids_overlap():
    segs = preprocess(tone(20), 16000, S, TRAINING)  # steady sound: every window qualifies
    assert len(segs) == S.max_segments_per_clip
    starts = sorted(s.start_s for s in segs)
    assert all(b - a >= S.segment_seconds for a, b in zip(starts, starts[1:]))


def test_short_clip_is_padded_to_one_segment():
    segs = preprocess(tone(0.4), 16000, S, INFERENCE)
    assert len(segs) == 1 and segs[0].padded
    assert segs[0].samples.size == 16000


def test_silent_windows_are_flagged_in_inference_and_dropped_in_training():
    silence = np.zeros(32000, dtype=np.float32)
    segs = preprocess(silence, 16000, S, INFERENCE)
    assert segs and all(s.silent for s in segs)  # the live view can show "silence"
    assert preprocess(silence, 16000, S, TRAINING) == []


def test_same_seed_gives_identical_output():
    a = preprocess(tone(0.4), 16000, S, TRAINING, seed=5)[0].samples
    b = preprocess(tone(0.4), 16000, S, TRAINING, seed=5)[0].samples
    assert np.array_equal(a, b)


def test_interior_digital_silence_is_removed():
    y = np.concatenate([tone(1.0), np.zeros(16000, dtype=np.float32), tone(1.0)])
    for seg in preprocess(y, 16000, S, INFERENCE):
        assert np.count_nonzero(seg.samples == 0) == 0
