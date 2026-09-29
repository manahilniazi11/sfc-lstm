"""Clips fetched with ``python -m sonic.dataset freesound`` (see sonic/dataset/freesound.py).

Layout: ``freesound/<class slug>/<sound id>.ogg`` plus ``sounds.json``. The
group is the Freesound ID, so a sound that is also in ESC-50, UrbanSound8K
or FSD50K is recognised as the same recording.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate
from ._util import license_from_url

# Freesound's API returns license names; the CC URL form is handled too.
_LICENSE_NAMES = {
    "Creative Commons 0": "CC0 1.0",
    "Attribution": "CC BY",
    "Attribution NonCommercial": "CC BY-NC",
}


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for index_path in sorted(root.glob("*/sounds.json")):
        sounds = json.loads(index_path.read_text(encoding="utf-8"))
        for sid, info in sorted(sounds.items()):
            path = index_path.parent / info["file"]
            if not path.exists():
                continue
            license_ = info.get("license", "")
            yield Candidate(
                source="freesound",
                path=path,
                class_name=info["class_name"],
                source_label=f"query: {info.get('query', '')}",
                group_id=f"freesound:{sid}",
                freesound_id=sid,
                license=_LICENSE_NAMES.get(license_) or license_from_url(license_),
                notes=f"freesound preview (lossy OGG); by {info.get('username', '?')}: {info.get('name', '')}",
            )
