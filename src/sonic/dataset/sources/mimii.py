"""MIMII (zenodo.org/record/3384388): normal/abnormal industrial machine sounds.

Layout after extraction: ``<snr>_dB_<machine>/<machine>/id_XX/{normal,abnormal}/*.wav``.
The same machine recording is released three times, mixed with factory noise
at 6, 0 and -6 dB SNR, so the group ignores the SNR: all three versions of a
recording must share a split.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..candidates import Candidate

LICENSE = "CC BY-SA 4.0 (MIMII)"
_SNR = re.compile(r"^(-?\d+)_dB")


def find(root: Path, label_map: dict[str, str | None]) -> Iterator[Candidate]:
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.relative_to(root).parts
        if len(parts) < 4:
            continue
        machine, machine_id, condition = parts[-4], parts[-3], parts[-2]
        class_name = label_map.get(condition)
        if not class_name:
            continue
        snr = next((m.group(1) for p in parts if (m := _SNR.match(p))), "unknown")
        yield Candidate(
            source="mimii",
            path=wav,
            class_name=class_name,
            source_label=f"{machine}:{condition}",
            group_id=f"mimii:{machine}:{machine_id}:{condition}:{wav.stem}",
            license=LICENSE,
            environment=f"factory, background noise at {snr} dB SNR",
            notes=f"machine={machine} id={machine_id} snr={snr}dB",
        )
