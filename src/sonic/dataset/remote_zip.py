"""Extract selected files from a remote zip without downloading the whole archive.

A zip keeps its table of contents (the "central directory") at the end.
Reading that with an HTTP Range request tells us where every file starts,
so each wanted file can be fetched as one small byte range. This also works
for multi-part archives (.z01, .z02, ..., .zip), where each part is one "disk".

Format reference: PKWARE APPNOTE.TXT, sections 4.3.7 (local header),
4.3.12 (central directory), 4.3.14-4.3.16 (ZIP64 and end records).
"""

from __future__ import annotations

import http.client
import json
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CENTRAL_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"
STORED, DEFLATED = 0, 8


class RemoteZipError(Exception):
    pass


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    crc: int
    compressed_size: int
    size: int
    disk: int  # which part the file starts in
    offset: int  # local header offset inside that part


class RemoteZip:
    """``urls`` are the archive parts in disk order; the ``.zip`` part is always last."""

    def __init__(self, urls: list[str], attempts: int = 8):
        self.urls = urls
        self.attempts = attempts
        self._sizes: dict[int, int] = {}

    # --- HTTP -------------------------------------------------------------------

    def _get(self, disk: int, start: int, end: int) -> bytes:
        """Bytes ``start..end`` (inclusive) of one part, retrying dropped connections."""
        url = self.urls[disk]
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "sonicsentinel-dataset"}
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        raise RemoteZipError(f"{url} does not support range requests")
                    total = response.headers.get("Content-Range", "").rsplit("/", 1)[-1]
                    if total.isdigit():
                        self._sizes[disk] = int(total)
                    data = response.read()
                if len(data) == end - start + 1:
                    return data
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504):
                    raise
                # Zenodo rate-limits bursts; honour Retry-After when it is sent.
                time.sleep(int(exc.headers.get("Retry-After") or 2**attempt))
                continue
            except (urllib.error.URLError, OSError, http.client.HTTPException):
                pass  # dropped connection or truncated response: retry
            time.sleep(min(30, 2**attempt))
        raise RemoteZipError(f"could not read bytes {start}-{end} of {url}")

    def _size(self, disk: int) -> int:
        if disk not in self._sizes:
            self._get(disk, 0, 0)
        return self._sizes[disk]

    def _read(self, disk: int, offset: int, length: int) -> bytes:
        """Read ``length`` bytes starting in ``disk``, continuing into later parts if needed."""
        chunks = []
        while length > 0:
            size = self._size(disk)
            if offset >= size:
                disk, offset = disk + 1, offset - size
                continue
            n = min(length, size - offset)
            chunks.append(self._get(disk, offset, offset + n - 1))
            length -= n
            offset += n
        return b"".join(chunks)

    # --- table of contents -------------------------------------------------------

    def members(self) -> dict[str, Member]:
        last = len(self.urls) - 1
        size = self._size(last)
        tail_start = max(0, size - 65536 - 22)  # EOCD comment can be up to 64 KiB
        tail = self._get(last, tail_start, size - 1)

        pos = tail.rfind(EOCD_SIG)
        if pos < 0:
            raise RemoteZipError("end of central directory not found")
        _, _, cd_disk, _, _, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", tail[pos : pos + 22])

        loc = tail.rfind(ZIP64_LOCATOR_SIG, 0, pos)
        if loc >= 0:  # ZIP64: the real values are in the ZIP64 end record
            z64 = tail.rfind(ZIP64_EOCD_SIG, 0, loc)
            if z64 < 0:
                raise RemoteZipError("ZIP64 end record not found")
            fields = struct.unpack("<IQHHIIQQQQ", tail[z64 : z64 + 56])
            cd_disk, cd_size, cd_offset = fields[5], fields[8], fields[9]

        directory = self._read(cd_disk, cd_offset, cd_size)
        return dict(_parse_central_directory(directory))

    # --- extraction ----------------------------------------------------------------

    def extract(self, member: Member, dest: Path) -> None:
        header = self._read(member.disk, member.offset, 30)
        if header[:4] != LOCAL_SIG:
            raise RemoteZipError(f"bad local header for {member.name}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        data = self._read(member.disk, member.offset + 30 + name_len + extra_len, member.compressed_size)

        if member.method == DEFLATED:
            data = zlib.decompress(data, -15)  # raw deflate stream, no zlib header
        elif member.method != STORED:
            raise RemoteZipError(f"{member.name}: unsupported compression method {member.method}")
        if zlib.crc32(data) != member.crc:
            raise RemoteZipError(f"{member.name}: CRC mismatch (corrupted transfer)")

        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)


def _parse_central_directory(directory: bytes):
    pos = 0
    while directory[pos : pos + 4] == CENTRAL_SIG:
        (method, crc, csize, usize, name_len, extra_len, comment_len, disk, offset) = struct.unpack(
            "<10xH4xIIIHHHH6xI", directory[pos : pos + 46]
        )
        name = directory[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
        extra = directory[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        usize, csize, offset, disk = _apply_zip64_extra(extra, usize, csize, offset, disk)
        yield name, Member(name, method, crc, csize, usize, disk, offset)
        pos += 46 + name_len + extra_len + comment_len


def _apply_zip64_extra(extra: bytes, usize: int, csize: int, offset: int, disk: int):
    """Values too big for the classic fields are stored as 0xFFFF... plus a ZIP64 extra field."""
    pos = 0
    while pos + 4 <= len(extra):
        header_id, length = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + length]
        if header_id == 0x0001:
            i = 0
            if usize == 0xFFFFFFFF:
                usize, i = struct.unpack("<Q", body[i : i + 8])[0], i + 8
            if csize == 0xFFFFFFFF:
                csize, i = struct.unpack("<Q", body[i : i + 8])[0], i + 8
            if offset == 0xFFFFFFFF:
                offset, i = struct.unpack("<Q", body[i : i + 8])[0], i + 8
            if disk == 0xFFFF:
                disk = struct.unpack("<I", body[i : i + 4])[0]
        pos += 4 + length
    return usize, csize, offset, disk


def cached_members(zip_: RemoteZip, cache: Path) -> dict[str, Member]:
    """The table of contents, cached on disk so it is only fetched once."""
    if cache.is_file():
        return {m["name"]: Member(**m) for m in json.loads(cache.read_text(encoding="utf-8"))}
    members = zip_.members()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([asdict(m) for m in members.values()]), encoding="utf-8")
    return members


def extract_many(zip_: RemoteZip, members: list[Member], dest_root: Path, workers: int = 4, label: str = "") -> list[str]:
    """Fetch ``members`` into ``dest_root/<member name>``; returns names that failed.

    Files already present are skipped, so an interrupted run simply continues.
    """
    todo = [m for m in members if not (dest_root / m.name).is_file()]
    total_bytes = sum(m.compressed_size for m in todo)
    done_bytes = done = 0
    failed = []
    print(f"{label}: {len(members) - len(todo)} already present, fetching {len(todo)} files "
          f"({total_bytes / 1024**3:.2f} GB)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(zip_.extract, m, dest_root / m.name): m for m in todo}
        for future in as_completed(futures):
            member = futures[future]
            try:
                future.result()
            except (RemoteZipError, zlib.error, OSError, http.client.HTTPException) as exc:
                failed.append(member.name)
                print(f"\n  failed: {exc}")
            done += 1
            done_bytes += member.compressed_size
            if done % 25 == 0 or done == len(todo):
                pct = 100 * done_bytes / total_bytes if total_bytes else 100
                print(f"\r  {done}/{len(todo)} files, {done_bytes / 1024**3:.2f} GB ({pct:.0f}%)", end="", flush=True)
    print()
    return failed
