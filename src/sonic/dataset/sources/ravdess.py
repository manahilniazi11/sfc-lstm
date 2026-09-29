"""RAVDESS speech audio (zenodo.org/record/1188976): 24 actors, 8 emotions.

Filenames are seven 2-digit fields:
modality-vocal_channel-emotion-intensity-statement-repetition-actor.
Label-map keys are ``<emotion>:<intensity>``, e.g. ``angry:strong``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate

LICENSE = "CC BY-NC-SA 4.0 (RAVDESS)"
EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
INTENSITIES = {"01": "normal", "02": "strong"}
SPEECH = "01"
_NAME = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for wav in sorted(root.rglob("*.wav")):
        match = _NAME.match(wav.stem)
        if not match:
            continue
        _modality, channel, emotion, intensity, statement, _rep, actor = match.groups()
        if channel != SPEECH:
            continue
        label = f"{EMOTIONS.get(emotion, emotion)}:{INTENSITIES.get(intensity, intensity)}"
        class_name = label_map.get(label)
        if not class_name:
            continue
        yield Candidate(
            source="ravdess",
            path=wav,
            class_name=class_name,
            source_label=label,
            group_id=f"ravdess:actor:{actor}",
            origin_id=f"ravdess:{wav.stem}",
            license=LICENSE,
            environment="studio (acted speech)",
            speaker=f"ravdess:{actor}",
            notes=f"statement={statement}",
        )
