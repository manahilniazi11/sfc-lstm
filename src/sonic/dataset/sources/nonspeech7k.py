"""Nonspeech7k (zenodo.org/record/6967442): 7,014 human non-speech clips, 7 classes.

Layout: ``train/<file>.wav`` and ``test/<file>.wav`` plus one metadata CSV per
part. Clips come from Freesound, aigei.com and YouTube; clips cut from one
source file share its "File ID", so they are grouped together. Freesound
clips use the Freesound ID as group, so overlaps with FSD50K are detected.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from ..candidates import Candidate
from ._util import read_csv

LICENSE = "CC BY-NC-SA 4.0 (Nonspeech7k)"


def _norm(key: str) -> str:
    # Column names differ between the two CSVs ("Augmentation  type", "File ID", "file_id").
    return re.sub(r"[^a-z]", "", key.lower())


def labelled_files(root: Path, label_map: dict) -> Iterator[tuple[str, dict[str, str], str]]:
    """Yield ``(part, row, class_name)`` for every metadata row mapped to one of our classes."""
    for csv_path in sorted(root.glob("metadata*.csv")):
        part = "train" if "train" in csv_path.name.lower() else "test"
        for raw_row in read_csv(csv_path):
            row = {_norm(k): (v or "").strip() for k, v in raw_row.items() if k}
            class_name = label_map.get(row.get("classname", ""))
            if class_name and not row.get("augmentationtype", "orig").lower().startswith("aug"):
                yield part, row, class_name


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for part, row, class_name in labelled_files(root, label_map):
        path = root / part / row["filename"]
        if not path.exists():
            continue
        host = urlparse(row.get("source", "")).netloc.removeprefix("www.") or "unknown"
        file_id = row.get("fileid", path.stem)
        if host == "freesound.org":
            group_id, freesound_id = f"freesound:{file_id}", file_id
        else:
            group_id, freesound_id = f"nonspeech7k:{host}:{row['classname']}:{file_id}", ""
        yield Candidate(
            source="nonspeech7k",
            path=path,
            class_name=class_name,
            source_label=row["classname"],
            group_id=group_id,
            freesound_id=freesound_id,
            license=LICENSE,
            notes=f"origin={host}",
        )
