"""Download and unpack the public datasets into ``<data>/raw/<source>/``.

Downloads resume after an interruption (HTTP Range requests), and an item
that is already unpacked is skipped. Sizes are the compressed download.

Zip items marked "selective" fetch only the files we use (see remote_zip.py):
FSD50K clips whose labels map to one of our classes, MIMII abnormal clips.
Pass ``full=True`` (``--full``) to download and unpack the whole archive.
"""

from __future__ import annotations

import http.client
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import load_classes, load_label_map
from .remote_zip import RemoteZip, cached_members, extract_many
from .sources import fsd50k, nonspeech7k

ZENODO = "https://zenodo.org/records"
GB = 1024**3


@dataclass(frozen=True)
class Item:
    name: str
    dest: str  # folder under <data>/raw
    urls: list[str]
    size_gb: float
    description: str
    keep: str = ""  # regex: extract only archive members whose path matches (saves disk)
    split_zip: bool = False  # multi-part .z01/.zip archive: needs 7-Zip
    # "" = whole archive only; "keep" = fetch members matching `keep`;
    # "fsd50k" / "nonspeech7k" = fetch clips whose labels map to one of our classes.
    selector: str = ""
    separate_archives: bool = False  # each url is its own zip (not parts of one)
    extra_files: tuple[str, ...] = ()  # small files downloaded as-is (label CSVs)
    max_files: int = 0  # selective items: take this many matches, evenly spread (0 = all)
    exclude_dest: str = ""  # skip members already present under this <data>/raw folder


CATALOG: dict[str, Item] = {
    item.name: item
    for item in [
        Item("esc50", "esc50",
             ["https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"],
             0.6, "ESC-50: animals, glass breaking, siren, car horn, ambient noise"),
        Item("urbansound8k", "urbansound8k",
             [f"{ZENODO}/1203745/files/UrbanSound8K.tar.gz"],
             5.6, "UrbanSound8K: gunshot, car horn, siren, dog bark, air conditioner"),
        Item("ravdess", "ravdess",
             [f"{ZENODO}/1188976/files/Audio_Speech_Actors_01-24.zip"],
             0.2, "RAVDESS speech: strongly angry speech for Aggression"),
        *[
            Item(f"mimii-{m}", f"mimii/6_dB_{m}",
                 [f"{ZENODO}/3384388/files/6_dB_{m}.zip"],
                 size, f"MIMII {m} at 6 dB SNR: Machinery Fault (abnormal clips)",
                 keep=r"/abnormal/.+\.wav$", selector="keep")
            for m, size in [("fan", 9.5), ("pump", 7.1), ("valve", 6.4), ("slider", 6.6)]
        ],
        Item("mimii-valve-normal", "mimii/6_dB_valve",
             [f"{ZENODO}/3384388/files/6_dB_valve.zip"],
             6.4, "MIMII valve normal operation: factory background for augmentation (200 clips)",
             keep=r"/normal/.+\.wav$", selector="keep", max_files=200),
        *[
            Item(f"mimii-{m}-normal", f"mimii/6_dB_{m}",
                 [f"{ZENODO}/3384388/files/6_dB_{m}.zip"],
                 size, f"MIMII {m} normal operation: Normal Machinery class (50 clips)",
                 keep=r"/normal/.+\.wav$", selector="keep", max_files=50)
            for m, size in [("pump", 7.1), ("fan", 9.5)]
        ],
        Item("mimii-valve-normal-holdout", "mimii_holdout/6_dB_valve",
             [f"{ZENODO}/3384388/files/6_dB_valve.zip"],
             6.4, "MIMII valve normal clips never used in training: factory-shortcut test (60 clips)",
             keep=r"/normal/.+\.wav$", selector="keep", max_files=60, exclude_dest="mimii/6_dB_valve"),
        Item("musan", "musan",
             ["https://www.openslr.org/resources/17/musan.tar.gz"],
             10.3, "MUSAN: only noise/ is extracted, for Background Noise",
             keep=r"^musan/noise/"),
        Item("fsd50k-labels", "fsd50k",
             [f"{ZENODO}/4060432/files/FSD50K.ground_truth.zip",
              f"{ZENODO}/4060432/files/FSD50K.metadata.zip"],
             0.01, "FSD50K labels and licenses (small; needed with either audio part)"),
        Item("fsd50k-eval", "fsd50k",
             [f"{ZENODO}/4060432/files/FSD50K.eval_audio.z01",
              f"{ZENODO}/4060432/files/FSD50K.eval_audio.zip"],
             5.8, "FSD50K eval audio (10k clips): screams, shouts, gunshots, shatter",
             split_zip=True, selector="fsd50k"),
        Item("fsd50k-dev", "fsd50k",
             [f"{ZENODO}/4060432/files/FSD50K.dev_audio.z0{i}" for i in range(1, 6)]
             + [f"{ZENODO}/4060432/files/FSD50K.dev_audio.zip"],
             17.1, "FSD50K dev audio (41k clips): more of every FSD50K class",
             split_zip=True, selector="fsd50k"),
        Item("nonspeech7k", "nonspeech7k",
             [f"{ZENODO}/6967442/files/train.zip", f"{ZENODO}/6967442/files/test.zip"],
             2.4, "Nonspeech7k: screaming clips for Panic Scream",
             selector="nonspeech7k", separate_archives=True,
             extra_files=(f"{ZENODO}/6967442/files/metadata%20of%20train%20set%20.csv",
                          f"{ZENODO}/6967442/files/metadata%20of%20test%20set.csv")),
    ]
}

MANUAL_SOURCES = """\
Sources that cannot be downloaded automatically:
  crema_d  git clone https://github.com/CheyneyComputerScience/CREMA-D <data>/raw/crema_d
           (needs Git LFS; only the AudioWAV folder is used). Also mirrored on Kaggle.
  MIVIA    request access at https://mivia.unisa.it/datasets/audio-analysis/mivia-audio-events/
           then cut the annotated events into <data>/raw/custom/<class slug>/ with source=mivia
           in <data>/raw/custom/recordings.csv.
  Own recordings and TTS help phrases go in <data>/raw/custom/<class slug>/."""


def list_items() -> str:
    lines = [f"{'name':<15}{'full size':>10}  description", "-" * 78]
    for item in CATALOG.values():
        mode = "  [selective]" if item.selector else ""
        lines.append(f"{item.name:<15}{item.size_gb:>8.1f}GB  {item.description}{mode}")
    lines.append("\n[selective] items fetch only the clips we use, a small fraction of the full size.")
    return "\n".join(lines) + "\n\n" + MANUAL_SOURCES


def fetch(
    name: str,
    raw_dir: Path,
    keep_archives: bool = False,
    full: bool = False,
    classes: list[str] | None = None,
) -> None:
    item = CATALOG[name]
    if item.selector and not full:
        _fetch_selected(item, raw_dir, classes)
        return
    dest = raw_dir / item.dest
    marker = dest / f".{item.name}.done"
    if marker.exists():
        print(f"[{name}] already downloaded and unpacked in {dest}")
        return

    archive_dir = raw_dir / "_downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archives = [_download(url, archive_dir / url.rsplit("/", 1)[-1]) for url in item.urls]

    dest.mkdir(parents=True, exist_ok=True)
    if item.split_zip:
        if not _extract_split_zip(archives[-1], dest, marker):
            return  # instructions printed; keep the parts for a manual extraction
    else:
        for archive in archives:
            print(f"[{name}] unpacking {archive.name} ...")
            _extract(archive, dest, item.keep)

    marker.touch()
    if not keep_archives:
        for archive in archives:
            archive.unlink(missing_ok=True)
    print(f"[{name}] done -> {dest}")


def _fetch_selected(item: Item, raw_dir: Path, classes: list[str] | None) -> None:
    dest = raw_dir / item.dest
    dest.mkdir(parents=True, exist_ok=True)
    for url in item.extra_files:
        _download(url, dest / urllib.parse.unquote(url.rsplit("/", 1)[-1]))

    archives = [[url] for url in item.urls] if item.separate_archives else [item.urls]
    failed = []
    for urls in archives:
        remote = RemoteZip(urls)
        key = f"{item.name}-{Path(urls[-1]).stem}" if item.separate_archives else item.name
        members = cached_members(remote, raw_dir / "_downloads" / f"{key}.index.json")
        if item.selector == "keep":
            wanted = [m for name, m in members.items() if re.search(item.keep, name)]
        elif item.selector == "fsd50k":
            wanted = _fsd50k_wanted(item, members, raw_dir, classes)
        else:
            wanted = _nonspeech7k_wanted(members, dest, classes)
        if item.exclude_dest:
            used = raw_dir / item.exclude_dest
            wanted = [m for m in wanted if not (used / m.name).exists()]
        if item.max_files and len(wanted) > item.max_files:
            wanted = sorted(wanted, key=lambda m: m.name)
            step = len(wanted) / item.max_files  # evenly spread over machines and recordings
            wanted = [wanted[int(i * step)] for i in range(item.max_files)]
        failed += extract_many(remote, wanted, dest, label=f"[{key}]")
    if failed:
        raise RuntimeError(f"{len(failed)} files failed; run the same command again to retry them")
    print(f"[{item.name}] done -> {dest}")


def _nonspeech7k_wanted(members: dict, root: Path, classes: list[str] | None) -> list:
    label_map = load_label_map(load_classes())["nonspeech7k"]
    names = {
        f"{part}/{row['filename']}"
        for part, row, class_name in nonspeech7k.labelled_files(root, label_map)
        if not classes or class_name in classes
    }
    return [m for name, m in members.items() if name in names]


def _fsd50k_wanted(item: Item, members: dict, raw_dir: Path, classes: list[str] | None) -> list:
    """Archive members for clips that map to one of our classes (optionally only ``classes``)."""
    root = raw_dir / "fsd50k"
    if not list(root.rglob("dev.csv")):
        fetch("fsd50k-labels", raw_dir)
    label_map = load_label_map(load_classes())["fsd50k"]
    folder = "FSD50K.eval_audio" if item.name == "fsd50k-eval" else "FSD50K.dev_audio"
    wanted, missing = [], 0
    for audio_name, fs_id, class_name, _ in fsd50k.classified_clips(root, label_map):
        if audio_name != folder or (classes and class_name not in classes):
            continue
        member = members.get(f"{folder}/{fs_id}.wav")
        if member is None:
            missing += 1
        else:
            wanted.append(member)
    if missing:
        print(f"[{item.name}] warning: {missing} labelled clips are not in the archive")
    return wanted


def _download(url: str, target: Path, attempts: int = 20) -> Path:
    """Download ``url`` to ``target``, resuming after dropped connections.

    The file keeps a ``.part`` suffix until its size matches what the server
    announced, so an interrupted download is never mistaken for a finished one.
    """
    if target.exists():
        return target
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            done, total = _download_once(url, partial, target.name)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:  # range starts at the end: nothing left to fetch
                break
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            print(f"\n  connection problem: {exc}", file=sys.stderr)
        else:
            print(file=sys.stderr)
            if not total or done >= total:
                break
            print(f"  connection closed early at {done / GB:.2f} GB", file=sys.stderr)
        wait = min(60, 2**attempt)
        print(f"  resuming in {wait}s (attempt {attempt}/{attempts}) ...", file=sys.stderr)
        time.sleep(wait)
    else:
        raise RuntimeError(
            f"{target.name} is still incomplete after {attempts} attempts; "
            "run the same command again to resume"
        )
    partial.rename(target)
    return target


def _download_once(url: str, partial: Path, name: str) -> tuple[int, int]:
    """One HTTP request continuing from the end of ``partial``; returns (bytes on disk, total)."""
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "sonicsentinel-dataset"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if start and response.status != 206:
            start = 0  # server ignored the range: start over
        total = int(response.headers.get("Content-Length") or 0) + start
        done = start
        next_report = 0
        with open(partial, "ab" if start else "wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    _progress(name, done, total)
                    next_report = done + 100 * 1024**2
    _progress(name, done, total)
    return done, total


def _progress(name: str, done: int, total: int) -> None:
    pct = f"{100 * done / total:5.1f}%" if total else ""
    size = f"{done / GB:.2f}/{total / GB:.2f} GB" if total else f"{done / GB:.2f} GB"
    print(f"\r  {name}: {size} {pct}", end="", file=sys.stderr, flush=True)


def _extract(archive: Path, dest: Path, keep: str = "") -> None:
    wanted = re.compile(keep)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if wanted.search(m)]
            zf.extractall(dest, members=members)
    else:
        with tarfile.open(archive, "r:*") as tf:
            members = [m for m in tf if wanted.search(m.name)]
            tf.extractall(dest, members=members, filter="data")


def find_7zip() -> str | None:
    """7-Zip's Windows installer does not add itself to PATH, so check its default folders first."""
    for folder in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if folder and (exe := Path(folder) / "7-Zip" / "7z.exe").is_file():
            return str(exe)
    return shutil.which("7z") or shutil.which("7za")


def _extract_split_zip(first_zip: Path, dest: Path, marker: Path) -> bool:
    """Python's zipfile cannot read multi-part archives, so use 7-Zip if present."""
    seven_zip = find_7zip()
    command = [seven_zip or "7z", "x", str(first_zip), f"-o{dest}", "-y"]
    if seven_zip:
        print(f"unpacking {first_zip.name} with 7-Zip ...")
        if subprocess.run(command).returncode == 0:
            return True
    print(
        "\nThis is a multi-part zip. Install 7-Zip (https://7-zip.org) and run:\n  "
        + " ".join(f'"{c}"' if " " in c else c for c in command)
        + f"\nthen create the empty file {marker} so this step is skipped next time."
    )
    return False
