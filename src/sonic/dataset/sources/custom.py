"""The team's own recordings and any hand-collected clips (MIVIA, Freesound, TTS).

Layout: ``<data>/raw/custom/<class slug or class name>/**/<file>``, e.g.
``<data>/raw/custom/help_request/ali_phone_room1_please_help.wav``.

An optional ``<data>/raw/custom/recordings.csv`` adds metadata per file. Its
``path`` column is relative to ``<data>/raw/custom``; every other column is
optional: environment, device, distance_m, speaker, group, freesound_id,
license, source, notes.

``group`` ties files from one original recording or session together (e.g.
several takes cut from one long phone recording). ``freesound_id`` does the
same across datasets, so a clip downloaded from Freesound that is also in
ESC-50 is recognised as the same recording.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from ..audio_check import AUDIO_EXTENSIONS
from ..candidates import Candidate
from ._util import read_csv

log = logging.getLogger(__name__)

METADATA_FILE = "recordings.csv"


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    """``label_map`` here maps lower-cased folder names (slug or class name) to classes."""
    if not root.is_dir():
        return
    extra = _load_metadata(root)
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        class_name = label_map.get(folder.name.lower())
        if not class_name:
            log.warning("custom: folder '%s' does not match any class slug or name; skipped", folder.name)
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rel = path.relative_to(root).as_posix()
            info = extra.get(rel, {})
            if info.get("group"):
                group_id = f"custom:{info['group']}"
            elif info.get("freesound_id"):
                group_id = f"freesound:{info['freesound_id']}"
            else:
                group_id = f"custom:{rel}"
            yield Candidate(
                source=info.get("source") or "custom",
                path=path,
                class_name=class_name,
                source_label=folder.name,
                group_id=group_id,
                origin_id=f"custom:{rel}",  # each file is its own original recording, even when grouped by speaker
                license=info.get("license") or "own recording (team)",
                freesound_id=info.get("freesound_id", ""),
                environment=info.get("environment") or "unknown",
                device=info.get("device") or "unknown",
                distance_m=info.get("distance_m", ""),
                speaker=info.get("speaker", ""),
                notes=info.get("notes", ""),
                priority=0,
            )


def _load_metadata(root: Path) -> dict[str, dict[str, str]]:
    path = root / METADATA_FILE
    if not path.is_file():
        return {}
    rows = {}
    for row in read_csv(path):
        rel = (row.get("path") or "").strip().replace("\\", "/")
        if rel:
            rows[rel] = {k: (v or "").strip() for k, v in row.items() if k}
    return rows
