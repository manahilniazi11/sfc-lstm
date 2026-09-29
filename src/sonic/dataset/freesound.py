"""Top up short classes from Freesound (freesound.org/docs/api).

Searches with the queries in ``config/freesound_queries.json``, keeps only
CC0 / CC BY / CC BY-NC sounds, and downloads each sound's high-quality
preview (OGG Vorbis, ~192 kbps), which needs only an API key: original files
would need a full OAuth2 login. Files go to ``<data>/raw/freesound/<slug>/``
with a ``sounds.json`` holding each sound's name, author, license and tags.

The key is read from FREESOUND_API_KEY (environment or .env) and sent in the
Authorization header, never in a URL, so it does not end up in logs.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .config import CONFIG_DIR, ConfigError, load_classes, setting

API = "https://freesound.org/apiv2/search/text/"
LICENSES = ("Creative Commons 0", "Attribution", "Attribution NonCommercial")
FIELDS = "id,name,tags,license,username,duration,previews"
PAGE_SIZE = 150

KEY_HELP = (
    "FREESOUND_API_KEY is not set. Create a key at https://freesound.org/apiv2/apply/ "
    "and add the line FREESOUND_API_KEY=<key> to the .env file in the repository root."
)


def load_queries(path: Path | None = None) -> dict[str, dict]:
    path = path or CONFIG_DIR / "freesound_queries.json"
    data = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    known = {c.name for c in load_classes()}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"{path.name}: unknown class(es) {sorted(unknown)}")
    return data


def fetch(raw_dir: Path, classes: list[str] | None = None, max_sounds: int | None = None) -> None:
    key = setting("FREESOUND_API_KEY")
    if not key:
        raise RuntimeError(KEY_HELP)
    slugs = {c.name: c.slug for c in load_classes()}
    for class_name, spec in load_queries().items():
        if classes and class_name not in classes:
            continue
        limit = max_sounds or spec.get("max_sounds", 200)
        fetch_class(class_name, spec, raw_dir / "freesound" / slugs[class_name], key, limit)


def fetch_class(class_name: str, spec: dict, dest: Path, key: str, limit: int) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    index_path = dest / "sounds.json"
    sounds: dict[str, dict] = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    exclude = {t.lower() for t in spec.get("exclude", [])}

    # Drop sounds fetched earlier that the (possibly updated) exclude list now rejects.
    for sid in [sid for sid, info in sounds.items() if _is_excluded(info, exclude)]:
        (dest / sounds.pop(sid)["file"]).unlink(missing_ok=True)
        print(f"  removed {sid}: now excluded")
    index_path.write_text(json.dumps(sounds, indent=1), encoding="utf-8")
    print(f"[freesound] {class_name}: {len(sounds)} already downloaded, target {limit}")

    for query in spec["queries"]:
        page = 1
        while len(sounds) < limit:
            results = _search(query, key, page)
            for sound in results["results"]:
                sid = str(sound["id"])
                if sid in sounds or _is_excluded(sound, exclude) or len(sounds) >= limit:
                    continue
                target = dest / f"{sid}.ogg"
                if not _download_preview(sound["previews"]["preview-hq-ogg"], target):
                    continue
                sounds[sid] = {
                    "file": target.name,
                    "class_name": class_name,
                    "query": query,
                    "name": sound["name"],
                    "username": sound["username"],
                    "license": sound["license"],
                    "tags": sound.get("tags", []),
                    "duration": sound.get("duration"),
                }
                index_path.write_text(json.dumps(sounds, indent=1), encoding="utf-8")
            print(f"\r  '{query}' page {page}: {len(sounds)}/{limit}", end="", flush=True)
            if not results.get("next"):
                break
            page += 1
        print()
        if len(sounds) >= limit:
            break
    print(f"[freesound] {class_name}: {len(sounds)} sounds in {dest}")


def _is_excluded(sound: dict, exclude: set[str]) -> bool:
    """True when an exclude term is one of the sound's tags or a word of its name."""
    tags = {t.lower() for t in sound.get("tags", [])}
    words = set(re.findall(r"[a-z]+", sound.get("name", "").lower()))
    return bool(exclude & (tags | words))


def _search(query: str, key: str, page: int) -> dict:
    license_filter = " OR ".join(f'"{name}"' for name in LICENSES)
    params = urllib.parse.urlencode({
        "query": query,
        "filter": f"duration:[0.5 TO 30] license:({license_filter})",
        "fields": FIELDS,
        "page_size": PAGE_SIZE,
        "page": page,
    })
    request = urllib.request.Request(f"{API}?{params}", headers={"Authorization": f"Token {key}"})
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError("Freesound rejected the API key (HTTP 401); check FREESOUND_API_KEY") from exc
            if exc.code == 404:
                return {"results": [], "next": None}  # past the last page
            if exc.code != 429 and exc.code < 500:
                raise
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2**attempt)  # the API allows 60 requests per minute
    raise RuntimeError(f"Freesound search for '{query}' kept failing")


def _download_preview(url: str, target: Path) -> bool:
    if target.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except (urllib.error.URLError, OSError):
        return False
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return True
