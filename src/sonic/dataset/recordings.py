"""Import the team's own recordings into ``<data>/raw/custom``.

Name files ``<speaker>_<phrase>_<environment>_<distance>m_<device>_<take>.<ext>``,
e.g. ``ali_pleasehelp_bedroom_2m_redmi9_1.m4a``, and put them in
``<data>/raw/custom/<class slug>/`` (e.g. ``help_request/``). Then run
``python -m sonic.dataset recordings``, which:

1. converts formats the builder cannot read (phone .m4a, .aac, .opus,
   .webm, .3gp, .amr) to WAV with ffmpeg, moving the originals to
   ``custom/_originals/``;
2. adds a row per new file to ``custom/recordings.csv`` from its name:
   speaker, environment, distance, device, phrase (notes). ``group`` is set
   to the speaker, so each speaker's recordings stay in one split and the
   test set measures unseen voices.

Existing rows in ``recordings.csv`` are never changed.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

from .audio_check import AUDIO_EXTENSIONS
from .config import DEFAULT_RAW_DIR, load_classes

CONVERT = {".m4a", ".aac", ".opus", ".webm", ".3gp", ".amr", ".wma", ".mp4"}
FIELDS = ["path", "environment", "device", "distance_m", "speaker", "group", "freesound_id", "license", "source", "notes"]


def parse_name(stem: str) -> dict[str, str]:
    """Metadata from ``speaker_phrase_environment_2m_device_take``; missing parts stay empty."""
    parts = stem.split("_")
    info = {"speaker": "", "notes": "", "environment": "", "distance_m": "", "device": ""}
    if len(parts) >= 6:
        speaker, phrase, environment, distance, device = parts[:5]
        info.update(speaker=speaker, notes=f"phrase={phrase}", environment=environment,
                    distance_m=distance.removesuffix("m"), device=device)
    elif parts:
        info["speaker"] = parts[0]
    return info


def convert(path: Path, originals: Path) -> Path:
    """ffmpeg -> 16-bit WAV next to the file; the original is moved to ``originals``."""
    target = path.with_suffix(".wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:a", "pcm_s16le", str(target)], check=True)
    originals.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), originals / path.name)
    return target


def import_recordings(raw_dir: Path = DEFAULT_RAW_DIR) -> dict[str, int]:
    root = raw_dir / "custom"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "recordings.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    known = {r["path"] for r in rows}
    slugs = {c.slug for c in load_classes()}
    stats = {"converted": 0, "added": 0, "skipped_folders": 0}

    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if folder.name not in slugs:
            stats["skipped_folders"] += 1
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in CONVERT:
                path = convert(path, root / "_originals" / folder.name)
                stats["converted"] += 1
            if path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            if rel in known:
                continue
            info = parse_name(path.stem)
            rows.append({**dict.fromkeys(FIELDS, ""), "path": rel, **info, "group": info["speaker"],
                         "license": "own recording (team, with consent)", "source": "custom"})
            known.add(rel)
            stats["added"] += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    return stats
