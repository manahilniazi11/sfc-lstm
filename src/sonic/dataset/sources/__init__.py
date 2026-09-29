"""Source adapters: one module per raw dataset.

Each adapter exposes ``find(root, label_map)``, which walks ``<data>/raw/<name>/``
and yields a :class:`~sonic.dataset.candidates.Candidate` for every file whose
label maps to one of our classes. Adapters only read metadata and paths; the
audio itself is checked later by the builder.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from ..candidates import Candidate
from . import (
    crema_d, custom, esc50, freesound, fsd50k, mimii, musan, nonspeech7k, ravdess, urbansound8k,
)

Finder = Callable[[Path, dict[str, str | None]], Iterator[Candidate]]

SOURCES: dict[str, Finder] = {
    "custom": custom.find,
    "esc50": esc50.find,
    "urbansound8k": urbansound8k.find,
    "fsd50k": fsd50k.find,
    "mimii": mimii.find,
    "crema_d": crema_d.find,
    "ravdess": ravdess.find,
    "musan": musan.find,
    "nonspeech7k": nonspeech7k.find,
    "freesound": freesound.find,
}
