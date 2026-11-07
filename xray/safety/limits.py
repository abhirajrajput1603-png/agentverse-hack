from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from xray.engine.concentration import ConcentrationAnalysis
from xray.engine.lookthrough import LookThroughResult
from xray.evidence import Evidence
from xray.safety.audit import AuditLogger


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"
LIMITS_CONFIG_PATH = CONFIG_DIR / "limits.json"


@dataclass
class CircuitBreakerEvaluation:
    status: str                         # "PASS" or "BREACH"
    rule_name: str                      # e.g. "SINGLE_STOCK_LOOKTHROUGH_LIMIT"
    threshold: float                    # e.g. 0.20
    actual_value: float                 # e.g. 0.3471
    excess: float                       # actual_value - threshold (e.g. +0.1471)
    breaching_issuer: Optional[str]     # e.g. "NVDA"
    message: str                        # Human readable summary
    evidence_id: str                    # Cites exact evidence
    timestamp: str                      # UTC ISO timestamp


@dataclass
class SafetyCheckResult:
    is_safe: bool
    evaluations: List[CircuitBreakerEvaluation]
    evidence_list: List[Evidence] = field(default_factory=list)

    def has_breaches(self) -> bool:
        return any(e.status == "BREACH" for e in evaluations)

    @property
    def primary_breach(self) -> Optional[CircuitBreakerEvaluation]:
        for e in self.evaluations:
            if e.status == "BREACH":
                return e
        return None


class SafetyLimitsEngine:
    """
    Deterministic safety and circuit breaker engine.
    Rules are strictly enforced in Python code, never in prompts or LLM logic.
    Logs every evaluation to the append-only audit trail.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
    ) -> None:
        self.config_path = config_path or LIMITS_CONFIG_PATH
        self.audit_logger = audit_logger or AuditLogger()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {
                "default_max_single_stock_lookthrough_weight": 0.20,
                "default_max_top_3_lookthrough_weight": 0.50,
                "circuit_breaker_enabled": True,
            }
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_portfolio(
        self,
        lt_result: LookThroughResult,
        concentration: ConcentrationAnalysis,
        user_max_single_stock_limit: Optional[float] = None,
        persona_name: Optional[str] = None,
    ) -> SafetyCheckResult:
        """
        Evaluates portfolio look-through against safety limits:
        1. Single-stock look-through limit (persona override or config default 20%)
        2. Top-3 look-through concentration limit
        """
        evaluations: List[CircuitBreakerEvaluation] = []
        evidence_list: List[Evidence] = []

        # 1. Single-stock look-through check
        threshold_single = (
            user_max_single_stock_limit
            if user_max_single_stock_limit is not None
            else self.config.get("default_max_single_stock_lookthrough_weight", 0.20)
        )

        # Get top single stock excluding residual and cash
        top_stocks = lt_result.sorted_exposures(exclude_residual_and_cash=True)
        top_issuer = top_stocks[0] if top_stocks else None
        top_weight = top_issuer.total_weight if top_issuer else 0.0
        top_name = top_issuer.issuer if top_issuer else "NONE"

        excess_single = round(top_weight - threshold_single, 6)
        status_single = "BREACH" if excess_single > 1e-6 else "PASS"

        if status_single == "BREACH":
            msg_single = (
                f"BREACH: True {top_name} exposure ({top_weight*100:.1f}%) exceeds "
                f"{threshold_single*100:.1f}% policy limit by +{excess_single*100:.1f}%."
            )
        else:
            msg_single = (
                f"PASS: True max single-issuer exposure ({top_name} at {top_weight*100:.2f}%) "
                f"is within {threshold_single*100:.1f}% limit."
            )

        # Evidence for single stock limit
        ev_single = Evidence(
            id=f"EV-LIMIT-SINGLE-STOCK-{top_name}",
            metric="single_stock_circuit_breaker",
            value={
                "status": status_single,
                "issuer": top_name,
                "actual_weight": round(top_weight, 6),
                "threshold": threshold_single,
                "excess": excess_single,
            },
            formula="actual_lookthrough_weight <= user_threshold",
            inputs={
                "issuer": top_name,
                "actual_weight": round(top_weight, 6),
                "threshold": threshold_single,
                "persona": persona_name or "Standard Profile",
            },
            data_sources=["Look-Through Decomposition Engine", "config/limits.json"],
            sample_size=None,
            caveats=["Enforced on named underlying issuers; excludes cash and residual baskets."],
        )
        evidence_list.append(ev_single)

        eval_single = CircuitBreakerEvaluation(
            status=status_single,
            rule_name="SINGLE_STOCK_LOOKTHROUGH_LIMIT",
            threshold=threshold_single,
            actual_value=round(top_weight, 6),
            excess=excess_single,
            breaching_issuer=top_name if status_single == "BREACH" else None,
            message=msg_single,
            evidence_id=ev_single.id,
            timestamp="",  # Filled below
        )

        # Audit log event
        audit_entry = self.audit_logger.log_event(
            event_type="CIRCUIT_BREAKER_EVALUATION",
            details={
                "persona": persona_name,
                "rule": eval_single.rule_name,
                "status": eval_single.status,
                "actual_value": eval_single.actual_value,
                "threshold": eval_single.threshold,
                "excess": eval_single.excess,
                "message": eval_single.message,
                "evidence_id": eval_single.evidence_id,
            },
        )
        eval_single.timestamp = audit_entry["timestamp"]
        evaluations.append(eval_single)

        is_safe = all(e.status == "PASS" for e in evaluations)
        return SafetyCheckResult(
            is_safe=is_safe,
            evaluations=evaluations,
            evidence_list=evidence_list,
        )
