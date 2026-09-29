"""The record every source adapter produces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Candidate:
    """One labelled audio file found in a raw dataset, before validation.

    ``group_id`` decides the split: files sharing a group (slices of one
    Freesound upload, the same MIMII recording at different SNRs, one actor's
    utterances) must land in the same split, otherwise the test set leaks
    into training.

    ``origin_id`` identifies the original recording, which is what the SRS
    counts as a unique clip. It defaults to ``group_id``; it differs when a
    group holds several original recordings (one actor's many utterances).
    """

    source: str
    path: Path
    class_name: str
    source_label: str
    group_id: str
    license: str
    freesound_id: str = ""
    environment: str = "unknown"
    device: str = "unknown"
    distance_m: str = ""
    speaker: str = ""
    notes: str = ""
    # Lower is picked first; the team's own recordings use 0.
    priority: int = 1
    origin_id: str = ""

    def __post_init__(self) -> None:
        if not self.origin_id:
            self.origin_id = self.group_id
