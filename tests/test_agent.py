from pathlib import Path
import pytest

from xray.agent.investigator import (
    AgentToolRunner,
    Finding,
    MAX_AGENT_TOOL_CALLS,
    PortfolioInvestigator,
)
from xray.data_loader import DataLoader
from xray.evidence import Evidence, EvidenceRegistry
from xray.profile import UserRiskProfile
from xray.safety.audit import AuditLogger
from xray.safety.output_filter import OutputSafetyFilter


def test_validator_accepts_valid_citations():
    registry = EvidenceRegistry()
    registry.register(
        Evidence(
            id="EV-TEST-1",
            metric="test_m",
            value=0.34,
            formula="x",
            inputs={},
            data_sources=["test"],
        )
    )

    agent = PortfolioInvestigator()
    raw_findings = [
        {
            "title": "Valid Finding",
            "explanation": "This is a verified finding citing EV-TEST-1.",
            "severity": "CRITICAL",
            "evidence_ids": ["EV-TEST-1"],
        }
    ]

    findings, passed, reason = agent.validate_findings(raw_findings, registry)
    assert passed is True
    assert reason is None
    assert len(findings) == 1
    assert findings[0].title == "Valid Finding"
    assert findings[0].evidence_ids == ["EV-TEST-1"]


def test_validator_rejects_missing_citations():
    registry = EvidenceRegistry()
    registry.register(
        Evidence(
            id="EV-TEST-1",
            metric="test_m",
            value=0.34,
            formula="x",
            inputs={},
            data_sources=["test"],
        )
    )

    agent = PortfolioInvestigator()
    raw_findings = [
        {
            "title": "Hallucinated Finding",
            "explanation": "No proof given.",
            "severity": "WARNING",
            "evidence_ids": ["EV-NON-EXISTENT-ID"],
        }
    ]

    findings, passed, reason = agent.validate_findings(raw_findings, registry)
    assert passed is False
    assert "cited non-existent evidence ID" in reason


def test_validator_rejects_zero_citations():
    registry = EvidenceRegistry()
    agent = PortfolioInvestigator()
    raw_findings = [
        {
            "title": "Unsubstantiated Claim",
            "explanation": "I think tech is risky.",
            "severity": "INFO",
            "evidence_ids": [],
        }
    ]

    findings, passed, reason = agent.validate_findings(raw_findings, registry)
    assert passed is False
    assert "cites zero evidence IDs" in reason


def test_output_filter_blocks_buy_sell_and_predictions():
    """
    Safety v2: Verify OutputSafetyFilter blocks direct buy/sell instructions
    and stock price predictions in code.
    """
    registry = EvidenceRegistry()
    registry.register(
        Evidence(
            id="EV-LT-NVDA",
            metric="lookthrough_weight_NVDA",
            value=0.3471,
            formula="w",
            inputs={},
            data_sources=["test"],
        )
    )
    agent = PortfolioInvestigator()

    # 1. Prohibited trading instruction
    bad_trade_finding = [
        {
            "title": "High NVDA Concentration",
            "explanation": "You should sell NVDA immediately and execute a buy order for bonds.",
            "severity": "CRITICAL",
            "evidence_ids": ["EV-LT-NVDA"],
        }
    ]
    _, passed_trade, reason_trade = agent.validate_findings(bad_trade_finding, registry)
    assert passed_trade is False
    assert "OutputSafetyFilter" in reason_trade

    # 2. Prohibited stock price prediction
    bad_pred_finding = [
        {
            "title": "NVDA Outlook",
            "explanation": "NVDA has a price target of $250 and will rally to $300 with a guaranteed return.",
            "severity": "INFO",
            "evidence_ids": ["EV-LT-NVDA"],
        }
    ]
    _, passed_pred, reason_pred = agent.validate_findings(bad_pred_finding, registry)
    assert passed_pred is False
    assert "OutputSafetyFilter" in reason_pred


def test_output_filter_numerical_grounding():
    """
    Verify numerical grounding check accepts percentages matching Evidence values
    and rejects hallucinated percentages.
    """
    registry = EvidenceRegistry()
    registry.register(
        Evidence(
            id="EV-LT-NVDA",
            metric="lookthrough_weight_NVDA",
            value=0.3471,
            formula="w",
            inputs={"threshold": 0.20},
            data_sources=["test"],
        )
    )
    agent = PortfolioInvestigator()

    # Grounded numbers (34.71% and 20.0% are in registry)
    grounded = [
        {
            "title": "NVDA at 34.71%",
            "explanation": "Exposure of 34.71% exceeds the 20.0% limit.",
            "severity": "CRITICAL",
            "evidence_ids": ["EV-LT-NVDA"],
        }
    ]
    _, passed_ok, _ = agent.validate_findings(
        grounded, registry, enforce_numerical_grounding=True
    )
    assert passed_ok is True

    # Hallucinated percentage (83.45% is nowhere in registry)
    hallucinated = [
        {
            "title": "NVDA at 83.45%",
            "explanation": "Exposure of 83.45% is extremely high.",
            "severity": "CRITICAL",
            "evidence_ids": ["EV-LT-NVDA"],
        }
    ]
    _, passed_bad, reason_bad = agent.validate_findings(
        hallucinated, registry, enforce_numerical_grounding=True
    )
    assert passed_bad is False
    assert "Ungrounded percentage figure" in reason_bad


def test_agent_tool_runner_8_call_cap_and_audit_log(tmp_path: Path):
    """
    Verify AgentToolRunner enforces the hard cap of 8 tool calls and writes
    every call + cap halt event to the append-only audit log.
    """
    loader = DataLoader()
    c1 = loader.load_synthetic_portfolios()[0]
    audit = AuditLogger(tmp_path / "agent_tools_audit.jsonl")

    runner = AgentToolRunner(
        holdings=c1["holdings"],
        persona_name=c1["persona"],
        user_max_single_stock_limit=c1["max_risk_tolerance_single_stock"],
        data_loader=loader,
        audit_logger=audit,
        max_calls=MAX_AGENT_TOOL_CALLS,
    )

    for i in range(MAX_AGENT_TOOL_CALLS):
        res = runner.call_tool("lookthrough_and_concentration", reason=f"Call {i+1}")
        assert "lookthrough_n_eff" in res

    assert len(runner.tool_trace) == MAX_AGENT_TOOL_CALLS

    # 9th call must be halted by the hard cap
    ninth = runner.call_tool("circuit_breaker", reason="Call 9 exceeding cap")
    assert ninth["status"] == "HALTED_CAP_REACHED"
    assert len(runner.tool_trace) == MAX_AGENT_TOOL_CALLS

    entries = audit.get_recent_entries(limit=20)
    event_types = [e["event_type"] for e in entries]
    assert event_types.count("AGENT_TOOL_CALL") == MAX_AGENT_TOOL_CALLS
    assert "AGENT_TOOL_CALL_CAP_REACHED" in event_types


def test_autonomous_investigation_loop_end_to_end(tmp_path: Path):
    """
    Verify PortfolioInvestigator.investigate_autonomous runs the full 6-tool
    investigation loop, logs to AuditLogger, interprets against UserRiskProfile,
    and returns validated findings with grounded Evidence IDs.
    """
    loader = DataLoader()
    c1 = loader.load_synthetic_portfolios()[0]
    audit = AuditLogger(tmp_path / "autonomous_audit.jsonl")
    profile = UserRiskProfile(
        horizon_years=3,
        max_tolerable_loss_pct=15.0,
        objective="Capital Preservation",
        experience_level="Beginner",
    )

    agent = PortfolioInvestigator()
    report = agent.investigate_autonomous(
        holdings=c1["holdings"],
        user_goal=c1["stated_goal"],
        persona_name=c1["persona"],
        user_max_single_stock_limit=c1["max_risk_tolerance_single_stock"],
        user_profile=profile,
        data_loader=loader,
        audit_logger=audit,
        use_llm=False,
    )

    assert report.validation_passed is True
    assert report.tool_calls_count == 6
    assert report.tool_calls_count <= MAX_AGENT_TOOL_CALLS
    assert len(report.findings) >= 3
    assert len(report.raw_evidence) >= 10

    # Verify all findings pass strict numerical grounding too
    registry = EvidenceRegistry()
    for ev in report.raw_evidence:
        registry.register(ev)
    raw_dicts = [
        {
            "title": f.title,
            "explanation": f.explanation,
            "severity": f.severity,
            "evidence_ids": f.evidence_ids,
        }
        for f in report.findings
    ]
    _, num_passed, num_reason = agent.validate_findings(
        raw_dicts, registry, enforce_numerical_grounding=True
    )
    assert num_passed is True, num_reason
