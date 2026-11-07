from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class Evidence:
    """
    Authoritative quantitative evidence container conforming to the
    Jane Street standard of absolute traceable arithmetic.
    Every engine metric must produce an Evidence object.
    """
    id: str                                      # e.g. "EV-LT-NVDA-001"
    metric: str                                  # e.g. "lookthrough_weight"
    value: Union[float, int, str, dict, list]    # Exact quantitative result
    formula: str                                 # Human-readable formula specification
    inputs: Dict[str, Any]                       # Raw inputs used in calculation
    data_sources: List[str]                      # File paths, disclosures, and as-of dates
    sample_size: Optional[int] = None            # e.g. Number of daily observations
    caveats: List[str] = field(default_factory=list)  # Missing data, proxies, assumptions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceRegistry:
    """In-memory store for evidence generated during an analysis session."""
    def __init__(self) -> None:
        self._items: Dict[str, Evidence] = {}

    def register(self, evidence: Evidence) -> Evidence:
        self._items[evidence.id] = evidence
        return evidence

    def get(self, evidence_id: str) -> Optional[Evidence]:
        return self._items.get(evidence_id)

    def all(self) -> List[Evidence]:
        return list(self._items.values())

    def clear(self) -> None:
        self._items.clear()
