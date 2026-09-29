"""Synthetic "Person Asking for Help" clips with Microsoft Edge text-to-speech (edge-tts).

The SRS allows these samples to be "generated synthetically where
permitted"; the competition organisers gave the team written permission to
use edge-tts. Every clip is marked as synthetic in the dataset metadata.

What is generated, into ``<data>/raw/custom/``:

- ``help_request/``: 300 clips of the SRS safety phrases (Help me, Somebody
  help, Please help, Call for help, Emergency), every English voice saying
  every phrase once plus extra repeated calls ("Help me! Help me!"), each
  with its own speaking rate, pitch and loudness so they sound urgent in
  different ways.
- ``background_noise/``: one ordinary sentence per voice ("I'll see you at
  the bus stop"). Without these, clear single-voice speech would only ever
  mean Help, and the models could learn "a voice" instead of "a call for
  help" (the SRS names this confusion: help request vs. ordinary speech).

Each voice is one ``group`` in ``recordings.csv``, so all clips of a voice
land in the same split: the test set is spoken by voices the models never
heard in training. Run ``python -m sonic.dataset tts`` and then
``python -m sonic.dataset extend`` to add the clips to the built dataset.
"""

from __future__ import annotations

import asyncio
import csv
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_RAW_DIR

HELP_SLUG = "help_request"
SPEECH_SLUG = "background_noise"
HELP_TARGET = 300
LICENSE = ("synthetic speech: Microsoft Edge Read Aloud via edge-tts, used for this student competition "
           "with the organisers' written permission")

# The SRS limits the class to these safety phrases.
PHRASES = {
    "helpme": "Help me!",
    "somebodyhelp": "Somebody help!",
    "pleasehelp": "Please help!",
    "callforhelp": "Call for help!",
    "emergency": "Emergency!",
}
# Ordinary speech for the negatives: everyday sentences without "help" or "emergency".
ORDINARY = [
    "I'll see you at the bus stop in ten minutes.",
    "Can you pass me the remote control?",
    "The meeting has been moved to Thursday afternoon.",
    "We are out of milk, I will buy some later.",
    "Did you watch the match last night?",
    "Please close the window, it is getting cold.",
    "My phone battery is almost empty.",
    "The train was a little late this morning.",
    "Let's order some food for dinner.",
    "Have you seen my keys anywhere?",
    "The weather looks nice for a walk today.",
    "I think the printer is out of paper again.",
]


@dataclass
class Clip:
    voice: str
    slug: str
    name: str  # file stem
    text: str
    rate: str
    pitch: str
    volume: str
    note: str


def _signed(value: int, unit: str) -> str:
    return f"{value:+d}{unit}"


def plan(voices: list[str], seed: int = 42) -> list[Clip]:
    """Which clip each voice says, with its prosody; deterministic for a given voice list and seed."""
    rng = random.Random(seed)
    clips: list[Clip] = []

    def urgent(voice, key, text, take):
        # a call for help is usually faster, higher and louder than normal speech; a few are calmer
        calm = rng.random() < 0.15
        rate = rng.randint(-10, 5) if calm else rng.randint(5, 35)
        pitch = rng.randint(-10, 5) if calm else rng.randint(5, 40)
        volume = rng.randint(0, 10) if calm else rng.randint(10, 50)
        name = f"tts-{voice}_{key}_synthetic_0m_edgetts_{take}"
        clips.append(Clip(voice, HELP_SLUG, name, text, _signed(rate, "%"), _signed(pitch, "Hz"),
                          _signed(volume, "%"), f"phrase={key}{' calm' if calm else ''}"))

    for voice in voices:
        for key, text in PHRASES.items():
            urgent(voice, key, text, 1)
    extra = HELP_TARGET - len(clips)
    for i in range(max(extra, 0)):  # repeated calls, spread over the voices (take 2, 3, ... per round)
        voice = voices[i % len(voices)]
        key = rng.choice(list(PHRASES))
        urgent(voice, key, f"{PHRASES[key]} {PHRASES[key]}", 2 + i // len(voices))
    for i, voice in enumerate(voices):
        sentence = ORDINARY[i % len(ORDINARY)]
        clips.append(Clip(voice, SPEECH_SLUG, f"tts-{voice}_speech_synthetic_0m_edgetts_1", sentence,
                          _signed(rng.randint(-10, 10), "%"), _signed(rng.randint(-5, 5), "Hz"), "+0%",
                          "ordinary speech (negative for Person Asking for Help)"))
    return clips


async def english_voices() -> list[str]:
    import edge_tts

    voices = await edge_tts.list_voices()
    return sorted(v["ShortName"] for v in voices if v["Locale"].startswith("en-"))


async def _synthesise(clip: Clip, target: Path, attempts: int = 4) -> None:
    import edge_tts

    with tempfile.TemporaryDirectory() as tmp:
        mp3 = Path(tmp) / "clip.mp3"
        for attempt in range(attempts):
            try:
                await edge_tts.Communicate(clip.text, clip.voice, rate=clip.rate, pitch=clip.pitch,
                                           volume=clip.volume).save(str(mp3))
                break
            except Exception:  # the online service drops connections now and then: wait and retry
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(2 * (attempt + 1))
        target.parent.mkdir(parents=True, exist_ok=True)
        # 16-bit WAV at the service's 24 kHz, with 0.3 s of silence before and after like a real recording
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                        "-af", "adelay=300:all=1,apad=pad_dur=0.3", "-c:a", "pcm_s16le", str(target)], check=True)


async def _generate(clips: list[Clip], root: Path, parallel: int) -> int:
    limit = asyncio.Semaphore(parallel)
    made = 0

    async def one(clip):
        nonlocal made
        target = root / clip.slug / f"{clip.name}.wav"
        if target.exists():
            return
        async with limit:
            await _synthesise(clip, target)
            made += 1

    await asyncio.gather(*(one(c) for c in clips))
    return made


FIELDS = ["path", "environment", "device", "distance_m", "speaker", "group", "freesound_id", "license", "source", "notes"]


def _register(clips: list[Clip], root: Path) -> int:
    """Add a recordings.csv row per clip (existing rows stay as they are)."""
    path = root / "recordings.csv"
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig"))) if path.exists() else []
    known = {r["path"] for r in rows}
    added = 0
    for c in clips:
        rel = f"{c.slug}/{c.name}.wav"
        if rel in known:
            continue
        known.add(rel)
        rows.append({"path": rel, "environment": "synthetic (clean text-to-speech)", "device": "edge-tts",
                     "distance_m": "", "speaker": c.voice, "group": f"tts-{c.voice}", "freesound_id": "",
                     "license": LICENSE, "source": "edge-tts",
                     "notes": f"{c.note}; text={c.text!r}; rate={c.rate} pitch={c.pitch} volume={c.volume}"})
        added += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    return added


def generate(raw_dir: Path = DEFAULT_RAW_DIR, parallel: int = 6, seed: int = 42) -> dict[str, int]:
    root = raw_dir / "custom"
    voices = asyncio.run(english_voices())
    clips = plan(voices, seed)
    made = asyncio.run(_generate(clips, root, parallel))
    return {"voices": len(voices), "help_clips": sum(c.slug == HELP_SLUG for c in clips),
            "speech_clips": sum(c.slug == SPEECH_SLUG for c in clips), "generated_now": made,
            "registered": _register(clips, root)}
