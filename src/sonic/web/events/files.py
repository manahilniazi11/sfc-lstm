"""Serving stored audio with HTTP range support, so the browser's player can seek (FR ix)."""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from django.http import FileResponse, Http404, HttpResponse

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
CONTENT_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg", ".m4a": "audio/mp4"}


def ranged_file_response(request, path: Path):
    if not path.is_file():
        raise Http404("audio file not found")
    size = path.stat().st_size
    content_type = CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    match = RANGE.match(request.headers.get("Range", "").strip())
    if not match:
        response = FileResponse(open(path, "rb"), content_type=content_type)
        response["Accept-Ranges"] = "bytes"
        return response
    first, last = match.groups()
    if first == "":  # "bytes=-500": the last 500 bytes
        start, end = max(size - int(last or 0), 0), size - 1
    else:
        start, end = int(first), min(int(last) if last else size - 1, size - 1)
    if start > end or start >= size:
        response = HttpResponse(status=416)
        response["Content-Range"] = f"bytes */{size}"
        return response
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start + 1)
    response = HttpResponse(data, status=206, content_type=content_type)
    response["Content-Range"] = f"bytes {start}-{end}/{size}"
    response["Accept-Ranges"] = "bytes"
    response["Content-Length"] = str(len(data))
    return response
