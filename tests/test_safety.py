import pytest
from pathlib import Path
from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.safety.audit import AuditLogger
from xray.safety.limits import SafetyLimitsEngine


def test_audit_logger(tmp_path: Path):
    log_file = tmp_path / "test_audit.jsonl"
    logger = AuditLogger(log_file)
    entry = logger.log_event("TEST_EVENT", {"foo": "bar", "num": 42})
    assert entry["event_type"] == "TEST_EVENT"
    assert entry["details"]["num"] == 42

    entries = logger.get_recent_entries()
    assert len(entries) == 1
    assert entries[0]["event_type"] == "TEST_EVENT"


def test_case_01_circuit_breaker_breach(tmp_path: Path):
    """
    Case 1: Liam's portfolio.
    Persona single stock tolerance: 20% (0.20).
    True look-through NVDA: 34.71%.
    Must trigger BREACH with excess +14.7%.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    lt_engine = LookThroughEngine(loader)
    lt_res = lt_engine.analyze(c1["holdings"])

    conc_engine = ConcentrationEngine()
    conc_analysis = conc_engine.analyze(lt_res)

    test_audit = AuditLogger(tmp_path / "audit_c1.jsonl")
    safety_engine = SafetyLimitsEngine(audit_logger=test_audit)

    result = safety_engine.evaluate_portfolio(
        lt_result=lt_res,
        concentration=conc_analysis,
        user_max_single_stock_limit=c1["max_risk_tolerance_single_stock"],
        persona_name=c1["persona"],
    )

    assert result.is_safe is False
    assert len(result.evaluations) == 1
    eval_0 = result.evaluations[0]
    assert eval_0.status == "BREACH"
    assert eval_0.breaching_issuer == "NVDA"
    assert abs(eval_0.actual_value - 0.3471) < 1e-3
    assert eval_0.threshold == 0.20
    assert abs(eval_0.excess - 0.1471) < 1e-3
    assert "BREACH: True NVDA exposure (34.7%) exceeds 20.0% policy limit by +14.7%." in eval_0.message

    # Check evidence object
    assert len(result.evidence_list) == 1
    assert result.evidence_list[0].id == "EV-LIMIT-SINGLE-STOCK-NVDA"

    # Check audit log written
    entries = test_audit.get_recent_entries()
    assert len(entries) == 1
    assert entries[0]["details"]["status"] == "BREACH"


def test_case_04_circuit_breaker_pass(tmp_path: Path):
    """
    Case 4: Passive portfolio.
    Persona single stock tolerance: 10% (0.10).
    Top single stock look-through: MSFT at ~5.13%.
    Must PASS.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    lt_engine = LookThroughEngine(loader)
    lt_res = lt_engine.analyze(c4["holdings"])

    conc_engine = ConcentrationEngine()
    conc_analysis = conc_engine.analyze(lt_res)

    test_audit = AuditLogger(tmp_path / "audit_c4.jsonl")
    safety_engine = SafetyLimitsEngine(audit_logger=test_audit)

    result = safety_engine.evaluate_portfolio(
        lt_result=lt_res,
        concentration=conc_analysis,
        user_max_single_stock_limit=c4["max_risk_tolerance_single_stock"],
        persona_name=c4["persona"],
    )

    assert result.is_safe is True
    eval_0 = result.evaluations[0]
    assert eval_0.status == "PASS"
    assert eval_0.threshold == 0.10
    assert abs(eval_0.actual_value - 0.05132) < 1e-4
    assert eval_0.excess < 0.0  # Under the limit
