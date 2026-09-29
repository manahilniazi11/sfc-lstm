"""Decoding and metadata, audio-format support, near-duplicate fingerprints and the images."""

import io
import shutil

import numpy as np
import pytest
import soundfile as sf

from sonic.inference import audio_io, fingerprint, visuals

SR = 22050


def event_audio(seed=0, seconds=4.0):
    """Noise bursts and a chirp: a recording with structure, different per seed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    y = 0.02 * rng.standard_normal(t.size)
    for start in rng.uniform(0, seconds - 0.4, 4):
        i = int(start * SR)
        y[i : i + int(0.3 * SR)] += rng.uniform(0.2, 0.6) * rng.standard_normal(int(0.3 * SR)) * np.hanning(int(0.3 * SR))
    y += 0.2 * np.sin(2 * np.pi * (300 + 400 * seed) * t * (1 + t / seconds))
    return y.astype(np.float32)


@pytest.mark.parametrize("ext,fmt,subtype,depth", [
    (".wav", "WAV", "PCM_16", 16), (".flac", "FLAC", "PCM_24", 24), (".ogg", "OGG", "VORBIS", None), (".mp3", "MP3", "MPEG_LAYER_III", None),
])
def test_supported_formats_decode_with_metadata(tmp_path, ext, fmt, subtype, depth):
    path = tmp_path / f"clip{ext}"
    stereo = np.stack([event_audio(), event_audio(1)], axis=1)
    sf.write(path, stereo, SR, format=fmt, subtype=subtype)
    audio, sr, info = audio_io.decode(path)
    assert info.format == fmt and info.channels == 2 and sr == SR and info.bit_depth == depth
    assert info.duration_s == pytest.approx(4.0, abs=0.1)


@pytest.mark.skipif(audio_io.find_ffmpeg() is None, reason="FFmpeg not installed")
def test_m4a_is_decoded_through_ffmpeg(tmp_path):
    import subprocess

    wav = tmp_path / "a.wav"
    sf.write(wav, event_audio(), SR)
    m4a = tmp_path / "a.m4a"
    subprocess.run([shutil.which("ffmpeg"), "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "aac", str(m4a)], check=True)
    audio, sr, info = audio_io.decode(m4a)
    assert info.format == "M4A" and info.duration_s == pytest.approx(4.0, abs=0.1)


def test_unsupported_and_damaged_files_are_rejected(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello", encoding="utf-8")
    with pytest.raises(audio_io.AudioDecodeError, match="unsupported"):
        audio_io.decode(txt)
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk")
    with pytest.raises(audio_io.AudioDecodeError, match="damaged"):
        audio_io.decode(fake)


# --- near-duplicates (FR lxxiv) ---

def mp3_roundtrip(y):
    buffer = io.BytesIO()
    sf.write(buffer, y, SR, format="MP3")
    buffer.seek(0)
    return sf.read(buffer, dtype="float32")[0]


def test_fingerprint_finds_reencoded_quieter_and_trimmed_copies():
    original = event_audio(3, seconds=8)
    fp = fingerprint.fingerprint(original, SR)
    copies = {
        "mp3": mp3_roundtrip(original),
        "quieter": original * 0.25,
        "trimmed": original[int(1.5 * SR) : int(6.5 * SR)],
    }
    for name, copy in copies.items():
        assert fingerprint.similarity(fp, fingerprint.fingerprint(copy, SR)) > fingerprint.THRESHOLD, name


def test_fingerprint_separates_different_recordings():
    a = fingerprint.fingerprint(event_audio(3, seconds=8), SR)
    b = fingerprint.fingerprint(event_audio(7, seconds=8), SR)
    assert fingerprint.similarity(a, b) < fingerprint.THRESHOLD


def test_fingerprint_bytes_roundtrip():
    fp = fingerprint.fingerprint(event_audio(), SR)
    assert np.allclose(fingerprint.from_bytes(fingerprint.to_bytes(fp)), fp, atol=0.05)


def test_silent_audio_has_an_empty_fingerprint():
    assert fingerprint.fingerprint(np.zeros(SR * 2, np.float32), SR).shape == (0, fingerprint.N_BANDS)


def test_images_are_png():
    y = event_audio()
    for png in (visuals.waveform_png(y, SR, marks=[(0.5, 1.5, "Gunshot")]), visuals.spectrogram_png(y, SR)):
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000
    long = np.tile(y, 30)  # two minutes: drawn as an envelope
    assert visuals.waveform_png(long, SR)[:4] == b"\x89PNG"
