"""Session document corpus into a local directory (LUC-1141).

Pure helpers behind ``client.resources.download``: validate the
``GET sdk/v2/sessions/<id>/documents`` listing, map each ``path`` safely under
the target directory, and write verified bytes atomically. No HTTP here; the
presigned bytes come from ``downloads.fetch_presigned``. Presigned URLs are
never logged or stored.
"""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, NamedTuple, Optional

import httpx

from ..core.errors import APIError

_CHUNK = 1 << 16


@dataclass(frozen=True)
class DownloadSummary:
    """What ``download`` did: corpus paths written / already current, and bytes written."""

    written: List[str]
    skipped: List[str]
    bytes: int


class Doc(NamedTuple):
    path: str
    target: Path
    sha256: str
    url: str


def parse_listing(body: Dict[str, Any], root: Path) -> List[Doc]:
    """Validate the whole listing before anything is fetched, so a bad entry writes nothing."""
    entries = body.get("documents") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        raise APIError("malformed documents listing")
    docs: List[Doc] = []
    seen = set()
    for entry in entries:
        fields = [entry.get(k) for k in ("path", "sha256", "url")] if isinstance(entry, dict) else []
        if len(fields) != 3 or not all(isinstance(f, str) for f in fields):
            raise APIError("malformed documents listing")
        path, sha256, url = fields
        if path in seen:
            raise APIError(f"duplicate document path {path!r}")
        seen.add(path)
        docs.append(Doc(path, target_path(root, path), sha256.lower(), url))
    return docs


def target_path(root: Path, rel: str) -> Path:
    """``root/rel``, refusing anything that could land outside ``root``."""
    parts = PurePosixPath(rel).parts
    if not parts or "\\" in rel or PurePosixPath(rel).is_absolute() or ".." in parts:
        raise APIError(f"unsafe document path {rel!r}")
    target = root.joinpath(*parts)
    # resolve() follows symlinks already inside root
    base = root.resolve()
    if base not in target.resolve().parents:
        raise APIError(f"unsafe document path {rel!r}")
    return target


def is_current(target: Path, sha256: str) -> bool:
    if not target.is_file():
        return False
    digest = hashlib.sha256()
    with target.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest() == sha256


def store(doc: Doc, data: bytes) -> int:
    """Write ``data`` to ``doc.target`` if its sha256 matches; returns bytes written."""
    if hashlib.sha256(data).hexdigest() != doc.sha256:
        raise APIError(f"sha256 mismatch for {doc.path}")
    doc.target.parent.mkdir(parents=True, exist_ok=True)
    part = doc.target.with_name(doc.target.name + ".part")
    part.write_bytes(data)
    os.replace(part, doc.target)
    return len(data)


def is_expired(exc: httpx.HTTPStatusError) -> bool:
    # s3 answers an expired url with 403 AccessDenied, or 400 ExpiredToken when the signing creds expired
    return exc.response.status_code in (400, 403)


def summarize(done: Dict[str, Optional[int]]) -> DownloadSummary:
    """``done`` maps corpus path to bytes written, or ``None`` when skipped."""
    return DownloadSummary(
        written=[p for p, n in done.items() if n is not None],
        skipped=[p for p, n in done.items() if n is None],
        bytes=sum(n for n in done.values() if n is not None),
    )
