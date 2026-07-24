"""Typed response model for client.usage (LUC-911).

The usage endpoint returns a *dynamic* ``{stat_name: total}`` map (stat names
are backend-normalized: spaces -> ``_``, ``/`` removed, lowercased), so — unlike
the fixed-field ``APIModel`` models — ``Usage`` is a thin read-only mapping
wrapper over the raw dict rather than a dataclass with declared fields.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, ItemsView, Iterator, KeysView


@dataclass
class Usage:
    """Org-aggregated usage / quota counters as a read-only mapping.

    Access counters by name — ``usage["num_sessions"]`` / ``usage.get("cost")``
    — or reach the raw dict via ``.stats``. An org-less key yields an empty map.
    """

    stats: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Usage":
        return cls(stats=dict(data) if isinstance(data, dict) else {})

    def get(self, key: str, default: Any = None) -> Any:
        return self.stats.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.stats)

    def __getitem__(self, key: str) -> Any:
        return self.stats[key]

    def __contains__(self, key: object) -> bool:
        return key in self.stats

    def __iter__(self) -> Iterator[str]:
        return iter(self.stats)

    def __len__(self) -> int:
        return len(self.stats)

    def keys(self) -> KeysView[str]:
        return self.stats.keys()

    def items(self) -> ItemsView[str, Any]:
        return self.stats.items()
