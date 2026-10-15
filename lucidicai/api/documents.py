"""Session document corpus into a local directory (LUC-1141).

Pure helpers behind ``client.resources.download``: validate the
``GET sdk/v2/sessions/<id>/documents`` listing, map each ``path`` safely under
the target directory, and write verified bytes atomically. No HTTP here; the
presigned bytes come from ``downloads.fetch_presigned``. Presigned URLs are
never logged or stored.
"""
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

import httpx

from ..core.errors import APIError

_CHUNK = 1 << 16
_FIELDS = ("path", "sha256", "url")


@dataclass(frozen=True)
class DownloadSummary:
    """What ``download`` did: corpus paths written / already current, and bytes written."""

    written: List[str]
    skipped: List[str]
    bytes: int


@dataclass(frozen=True)
class Doc:
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
    targets = set()
    for entry in entries:
        if not isinstance(entry, dict) or not all(isinstance(entry.get(k), str) for k in _FIELDS):
            raise APIError("malformed documents listing")
        doc = Doc(entry["path"], target_path(root, entry["path"]), entry["sha256"].lower(), entry["url"])
        # compare targets, so `a/b` and `./a/b` collide too
        if doc.target in targets:
            raise APIError(f"duplicate document path {doc.path!r}")
        targets.add(doc.target)
        docs.append(doc)
    nested = [d.path for d in docs if not targets.isdisjoint(d.target.parents)]
    if nested:
        raise APIError(f"document path {nested[0]!r} is under another document")
    return docs


def target_path(root: Path, rel: str) -> Path:
    """``root/rel``, refusing anything that could land outside ``root``."""
    parts = PurePosixPath(rel).parts
    # `..` is refused even when it stays inside root; resolve() below catches the rest
    if "\\" in rel or "\x00" in rel or ".." in parts:
        raise APIError(f"unsafe document path {rel!r}")
    target = root.joinpath(*parts)
    # resolve() follows symlinks already inside root
    if root.resolve() not in target.resolve().parents:
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
    # random name so no corpus path collides; not mkstemp, whose 0600 hides docs from other uids
    part = doc.target.with_name(f"{doc.target.name}.{secrets.token_hex(8)}.part")
    with os.fdopen(os.open(part, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666), "wb") as f:
        f.write(data)
    os.replace(part, doc.target)
    return len(data)


def is_expired(exc: httpx.HTTPStatusError) -> bool:
    # s3 answers an expired url with 403 AccessDenied, or 400 ExpiredToken when the signing creds expired
    return exc.response.status_code in (400, 403)


def expired_again(exc: httpx.HTTPStatusError) -> APIError:
    """The second expiry, without the presigned URL that ``exc``'s message carries."""
    status = exc.response.status_code
    return APIError(f"document URL expired again after a re-list (S3 {status})", status_code=status)


def summarize(done: Dict[str, Optional[int]]) -> DownloadSummary:
    """``done`` maps corpus path to bytes written, or ``None`` when skipped."""
    return DownloadSummary(
        written=[p for p, n in done.items() if n is not None],
        skipped=[p for p, n in done.items() if n is None],
        bytes=sum(n for n in done.values() if n is not None),
    )
