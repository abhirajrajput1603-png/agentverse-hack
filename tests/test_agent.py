import pytest
from xray.agent.investigator import PortfolioInvestigator, Finding
from xray.evidence import Evidence, EvidenceRegistry


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
    # Hallucinated evidence ID
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
