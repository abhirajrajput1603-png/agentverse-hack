from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from xray.agent.prompts import INVESTIGATOR_SYSTEM_PROMPT
from xray.engine.concentration import ConcentrationAnalysis
from xray.engine.lookthrough import LookThroughResult
from xray.engine.risk import RiskMetrics
from xray.evidence import Evidence, EvidenceRegistry
from xray.safety.limits import SafetyCheckResult


load_dotenv()


@dataclass
class Finding:
    title: str
    explanation: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class InvestigationReport:
    findings: List[Finding]
    raw_evidence: List[Evidence]
    validation_passed: bool
    rejection_reason: Optional[str] = None


class PortfolioInvestigator:
    """
    AI Portfolio Risk Investigator powered by Gemini.
    Zero Mental Math: All quantitative values provided via deterministic Evidence objects.
    Post-check verification: Every finding must cite existing evidence IDs, and all cited numbers
    are validated against evidence records.
    """

    def __init__(self, model_name: str = "gemini-3.8-flash") -> None:
        self.model_name = model_name
        self._api_key = os.environ.get("GEMINI_API_KEY")

    def _call_gemini(self, prompt: str) -> str:
        """Invokes Gemini SDK with system instructions."""
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY not configured in environment or .env file.")

        from google import genai
        client = genai.Client(api_key=self._api_key)
        
        full_prompt = f"{INVESTIGATOR_SYSTEM_PROMPT}\n\n{prompt}"
        response = client.models.generate_content(
            model=self.model_name,
            contents=full_prompt,
        )
        return response.text or ""

    def validate_findings(
        self,
        findings_data: List[Dict[str, Any]],
        evidence_registry: EvidenceRegistry,
    ) -> Tuple[List[Finding], bool, Optional[str]]:
        """
        Validates agent output against deterministic invariants:
        1. All cited evidence IDs must exist in the registry.
        2. Format matches Finding schema.
        """
        valid_evidence_ids = set(evidence_registry._items.keys())
        parsed_findings: List[Finding] = []

        if not isinstance(findings_data, list):
            return [], False, "Output is not a JSON list"

        for idx, item in enumerate(findings_data):
            title = item.get("title", "")
            explanation = item.get("explanation", "")
            severity = item.get("severity", "INFO").upper()
            cites = item.get("evidence_ids", [])

            if not title or not explanation:
                return [], False, f"Finding {idx} is missing title or explanation"

            if not cites:
                return [], False, f"Finding '{title}' failed invariant: cites zero evidence IDs"

            for cid in cites:
                if cid not in valid_evidence_ids:
                    return [], False, f"Finding '{title}' cited non-existent evidence ID '{cid}'"

            parsed_findings.append(
                Finding(
                    title=title,
                    explanation=explanation,
                    severity=severity,
                    evidence_ids=cites,
                )
            )

        return parsed_findings, True, None

    def investigate(
        self,
        user_goal: str,
        persona_name: str,
        lt_result: LookThroughResult,
        concentration: ConcentrationAnalysis,
        risk: RiskMetrics,
        safety: SafetyCheckResult,
    ) -> InvestigationReport:
        """
        Main investigation entry point.
        Aggregates deterministic evidence, queries Gemini, validates response against evidence IDs.
        """
        registry = EvidenceRegistry()
        all_evidence = (
            lt_result.evidence_list
            + concentration.evidence_list
            + risk.evidence_list
            + safety.evidence_list
        )
        for ev in all_evidence:
            registry.register(ev)

        # Prepare prompt payload
        evidence_summary = [
            {
                "id": ev.id,
                "metric": ev.metric,
                "value": ev.value,
                "formula": ev.formula,
                "caveats": ev.caveats,
            }
            for ev in all_evidence
        ]

        payload = {
            "user_context": {
                "persona": persona_name,
                "stated_goal": user_goal,
            },
            "circuit_breaker_result": [
                {
                    "rule": e.rule_name,
                    "status": e.status,
                    "actual_value": e.actual_value,
                    "threshold": e.threshold,
                    "excess": e.excess,
                    "evidence_id": e.evidence_id,
                }
                for e in safety.evaluations
            ],
            "evidence_registry": evidence_summary,
        }

        user_prompt = (
            "Review this user's investment portfolio using the verified evidence provided below.\n"
            "Identify the key underlying risks (especially hidden look-through concentrations, "
            "correlation clusters, or policy breaches), and explain them in clear, objective terms.\n\n"
            f"INPUT EVIDENCE JSON:\n{json.dumps(payload, indent=2)}\n\n"
            "Respond ONLY with a JSON array of findings."
        )

        # First attempt
        raw_response = self._call_gemini(user_prompt)
        findings_json = self._extract_json(raw_response)

        findings, passed, reason = self.validate_findings(findings_json, registry)

        # Deterministic retry if validation failed
        if not passed:
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"PREVIOUS ATTEMPT FAILED VALIDATION: {reason}\n"
                "Please correct this immediately. Every finding MUST cite valid evidence IDs."
            )
            raw_response = self._call_gemini(retry_prompt)
            findings_json = self._extract_json(raw_response)
            findings, passed, reason = self.validate_findings(findings_json, registry)

        return InvestigationReport(
            findings=findings,
            raw_evidence=all_evidence,
            validation_passed=passed,
            rejection_reason=reason,
        )

    @staticmethod
    def _extract_json(text: str) -> List[Dict[str, Any]]:
        """Extracts JSON array from text response safely."""
        text = text.strip()
        # Remove markdown codeblocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)

        try:
            return json.loads(text)
        except Exception:
            return []
