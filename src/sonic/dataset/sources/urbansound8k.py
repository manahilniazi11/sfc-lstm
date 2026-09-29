"""UrbanSound8K: 8,732 slices (up to 4 s) cut from Freesound recordings."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate
from ._util import find_file, read_csv

LICENSE = "CC BY-NC 3.0 (UrbanSound8K)"
SALIENCE = {"1": "foreground", "2": "background"}


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    meta = find_file(root, "UrbanSound8K.csv")
    if meta is None:
        return
    audio_dir = meta.parent.parent / "audio"
    for row in read_csv(meta):
        class_name = label_map.get(row["class"])
        if not class_name:
            continue
        # Many slices share one fsID; grouping by it keeps them in one split.
        fs_id = row["fsID"]
        yield Candidate(
            source="urbansound8k",
            path=audio_dir / f"fold{row['fold']}" / row["slice_file_name"],
            class_name=class_name,
            source_label=row["class"],
            group_id=f"freesound:{fs_id}",
            freesound_id=fs_id,
            license=LICENSE,
            notes=f"salience={SALIENCE.get(row['salience'], row['salience'])}",
        )
