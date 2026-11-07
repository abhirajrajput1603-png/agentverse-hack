from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class UserRiskProfile:
    """
    Authoritative investor risk profile.
    Rule 7: Personalise the interpretation, NEVER the maths.
    Quantitative portfolio metrics are strictly invariant; the profile guides
    the circuit breaker thresholds and contextual AI explanations.
    """
    name: str = "Everyday Investor"
    horizon_years: float = 3.0
    max_tolerable_loss_pct: float = 0.20        # e.g. 20% max tolerable drawdown
    max_single_stock_limit: float = 0.20        # e.g. 20% max single-issuer exposure
    objective: str = "Capital preservation with moderate growth"
    experience_level: str = "Intermediate"      # "Beginner", "Intermediate", "Advanced"
    financial_milestone: Optional[str] = None   # e.g. "£60,000 house deposit in 3 years"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserRiskProfile:
        return cls(
            name=data.get("name", "Everyday Investor"),
            horizon_years=float(data.get("horizon_years", 3.0)),
            max_tolerable_loss_pct=float(data.get("max_tolerable_loss_pct", 0.20)),
            max_single_stock_limit=float(data.get("max_single_stock_limit", 0.20)),
            objective=data.get("objective", "Capital preservation with moderate growth"),
            experience_level=data.get("experience_level", "Intermediate"),
            financial_milestone=data.get("financial_milestone"),
        )
