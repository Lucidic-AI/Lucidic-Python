"""Typed response-model base for the SDK read surface (LUC-905).

Read methods return typed dataclasses instead of raw dicts. Each model is a
plain ``@dataclass`` that mixes in ``APIModel`` and is built from backend JSON
via ``from_dict`` — which ignores unrecognized keys (kept on ``.extra`` for
forward-compat, so a newer backend can add fields without breaking an older
SDK) and lets the dataclass surface a clear error for a genuinely missing
required field.

No pydantic — this extends the SDK's lone prior model precedent, the ``Prompt``
dataclass (``api/resources/prompt.py``). Nested typed models convert in the
owning model's ``from_dict`` override.

``CursorPage`` lives here too since it is the paginated container of models;
the iteration helpers that walk ``next`` live in ``api/pagination.py``.
"""
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Type, TypeVar
from urllib.parse import parse_qs, urlparse

T = TypeVar("T", bound="APIModel")


class APIModel:
    """Mixin for typed backend response models.

    Subclasses must be ``@dataclass``. ``APIModel`` is intentionally *not*
    itself a dataclass so that a base field (e.g. a defaulted ``extra``) does
    not force every subclass field to carry a default — the classic dataclass
    inheritance ordering trap. Unknown backend fields are stashed on a private
    ``_extra`` attribute and exposed read-only via ``.extra``.
    """

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Build a model from a backend JSON object.

        Unknown keys are ignored (kept on ``.extra`` for forward-compat). A
        ``null`` for a field that has a default (or ``default_factory``) is
        dropped so the default applies — turning a backend ``"tools": null``
        into ``[]`` rather than ``None``, which keeps a downstream
        ``for x in model.field`` safe. A genuinely missing required field still
        surfaces a clear ``TypeError`` from the dataclass constructor.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"{cls.__name__}.from_dict expected a dict, got "
                f"{type(data).__name__}"
            )
        field_map = {f.name: f for f in fields(cls)}
        known: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            f = field_map.get(key)
            if f is None:
                extra[key] = value
                continue
            has_default = f.default is not MISSING or f.default_factory is not MISSING
            if value is None and has_default:
                continue  # let the field default apply instead of a null
            known[key] = value
        obj = cls(**known)
        # object.__setattr__ works whether or not the subclass is frozen.
        object.__setattr__(obj, "_extra", extra)
        return obj

    @classmethod
    def from_list(cls: Type[T], items: List[Dict[str, Any]]) -> List[T]:
        """Build a list of models from a list of backend JSON objects."""
        return [cls.from_dict(item) for item in items]

    @property
    def extra(self) -> Dict[str, Any]:
        """Backend fields not mapped to a declared attribute (forward-compat)."""
        return getattr(self, "_extra", {})

    def to_dict(self) -> Dict[str, Any]:
        """Shallow dict of the declared dataclass fields (excludes ``extra``)."""
        return asdict(self)


def _extract_cursor(url: Optional[str]) -> Optional[str]:
    """Pull the ``cursor`` query param out of a paginated ``next``/``previous``
    URL. Returns None when there is no such page or no cursor present."""
    if not url:
        return None
    values = parse_qs(urlparse(url).query).get("cursor")
    return values[0] if values else None


@dataclass
class CursorPage:
    """One page of a cursor-paginated list endpoint.

    ``results`` are typed models when a ``model`` was supplied to
    ``from_body``, else raw dicts. ``next_cursor`` / ``previous_cursor`` are the
    extracted cursor tokens (None at the ends). ``raw`` is the untouched body.
    """

    results: List[Any]
    next_cursor: Optional[str]
    previous_cursor: Optional[str]
    raw: Dict[str, Any]

    @classmethod
    def from_body(
        cls, body: Dict[str, Any], *, model: Optional[Type[APIModel]] = None
    ) -> "CursorPage":
        raw_results = body.get("results", []) if isinstance(body, dict) else []
        results = [
            model.from_dict(item) if model is not None else item
            for item in raw_results
        ]
        return cls(
            results=results,
            next_cursor=_extract_cursor(body.get("next") if isinstance(body, dict) else None),
            previous_cursor=_extract_cursor(body.get("previous") if isinstance(body, dict) else None),
            raw=body if isinstance(body, dict) else {},
        )

    @property
    def has_next(self) -> bool:
        return self.next_cursor is not None
