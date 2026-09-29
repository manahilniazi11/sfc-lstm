from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path


def find_file(root: Path, name: str) -> Path | None:
    """First file called ``name`` anywhere under ``root`` (archives nest differently)."""
    return next((p for p in sorted(root.rglob(name)) if p.is_file()), None)


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        yield from csv.DictReader(f)


_CC_URL = re.compile(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/([\d.]+)")


def license_from_url(url: str) -> str:
    """'http://creativecommons.org/licenses/by-nc/3.0/' -> 'CC BY-NC 3.0'."""
    match = _CC_URL.search(url or "")
    if not match:
        return url or "unknown"
    kind, name, version = match.groups()
    if kind == "publicdomain":
        return f"CC0 {version}" if name == "zero" else f"Public Domain ({name})"
    return f"CC {name.upper()} {version}"
