"""CREMA-D: 7,442 acted emotional utterances from 91 actors.

Filenames look like ``1001_DFA_ANG_XX.wav`` (actor_sentence_emotion_level).
Clips are grouped by actor so no speaker appears in both train and test.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate

LICENSE = "ODbL 1.0 (CREMA-D)"
_NAME = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for wav in sorted(root.rglob("*.wav")):
        match = _NAME.match(wav.stem)
        if not match:
            continue
        actor, sentence, emotion, level = match.groups()
        class_name = label_map.get(emotion)
        if not class_name:
            continue
        yield Candidate(
            source="crema_d",
            path=wav,
            class_name=class_name,
            source_label=f"{emotion}:{level}",
            group_id=f"crema_d:actor:{actor}",
            origin_id=f"crema_d:{wav.stem}",
            license=LICENSE,
            environment="studio (acted speech)",
            speaker=f"crema_d:{actor}",
            notes=f"sentence={sentence} level={level}",
        )
