"""ESC-50 (github.com/karolpiczak/ESC-50): 2,000 five-second clips, 50 classes."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate
from ._util import find_file, read_csv

LICENSE = "CC BY-NC 3.0 (ESC-50)"


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    meta = find_file(root, "esc50.csv")
    if meta is None:
        return
    audio_dir = meta.parent.parent / "audio"
    for row in read_csv(meta):
        class_name = label_map.get(row["category"])
        if not class_name:
            continue
        # src_file is the Freesound ID of the recording the clip was cut from.
        fs_id = row["src_file"]
        yield Candidate(
            source="esc50",
            path=audio_dir / row["filename"],
            class_name=class_name,
            source_label=row["category"],
            group_id=f"freesound:{fs_id}",
            freesound_id=fs_id,
            license=LICENSE,
        )
