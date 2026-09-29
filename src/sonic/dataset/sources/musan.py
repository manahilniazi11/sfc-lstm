"""MUSAN (openslr.org/17): only the ``noise/`` part is used.

``noise/free-sound`` is mostly ambient and technical noise. ``noise/sound-bible``
contains assorted sound effects (some could be gunshots or alarms), so it is
disabled in the label map by default. Spot-check whichever subset you enable.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate

LICENSE = "MUSAN noise (per-file licenses in the subset's LICENSE file)"


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for subset, class_name in label_map.items():
        if not class_name:
            continue
        folders = [d for d in sorted(root.rglob(subset)) if d.is_dir() and d.parent.name == "noise"]
        for folder in folders:
            for wav in sorted(folder.glob("*.wav")):
                yield Candidate(
                    source="musan",
                    path=wav,
                    class_name=class_name,
                    source_label=subset,
                    group_id=f"musan:{subset}:{wav.stem}",
                    license=LICENSE,
                )
