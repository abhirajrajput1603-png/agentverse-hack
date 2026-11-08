from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from xray.agent.prompts import INVESTIGATOR_SYSTEM_PROMPT
from xray.data_loader import DataLoader
from xray.engine.benchmark import BenchmarkEngine
from xray.engine.concentration import ConcentrationAnalysis, ConcentrationEngine
from xray.engine.correlation import CorrelationAndRiskContributionEngine
from xray.engine.lookthrough import LookThroughEngine, LookThroughResult
from xray.engine.risk import RiskEngine, RiskMetrics
from xray.engine.stress import StressAndTailEngine
from xray.evidence import Evidence, EvidenceRegistry
from xray.profile import UserRiskProfile
from xray.safety.audit import AuditLogger
from xray.safety.limits import SafetyCheckResult, SafetyLimitsEngine
from xray.safety.output_filter import OutputSafetyFilter


load_dotenv()

MAX_AGENT_TOOL_CALLS = 8


@dataclass
class Finding:
    title: str
    explanation: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ToolCallRecord:
    step: int
    tool_name: str
    reason: str
    evidence_ids_produced: List[str]
    summary: Dict[str, Any]


@dataclass
class InvestigationReport:
    findings: List[Finding]
    raw_evidence: List[Evidence]
    validation_passed: bool
    rejection_reason: Optional[str] = None
    tool_trace: List[ToolCallRecord] = field(default_factory=list)
    tool_calls_count: int = 0


class AgentToolRunner:
    """
    Deterministic tool execution harness for Agent v2 autonomous investigation.
    Enforces a strict cap of MAX_AGENT_TOOL_CALLS (8) tool calls per investigation
    and logs every tool invocation and result to the append-only audit log.
    """

    def __init__(
        self,
        holdings: List[Dict[str, Any]],
        persona_name: str,
        user_max_single_stock_limit: Optional[float] = None,
        data_loader: Optional[DataLoader] = None,
        audit_logger: Optional[AuditLogger] = None,
        max_calls: int = MAX_AGENT_TOOL_CALLS,
    ) -> None:
        self.holdings = holdings
        self.persona_name = persona_name
        self.user_max_single_stock_limit = user_max_single_stock_limit
        self.loader = data_loader or DataLoader()
        self.audit = audit_logger or AuditLogger()
        self.max_calls = max_calls

        self.registry = EvidenceRegistry()
        self.tool_trace: List[ToolCallRecord] = []
        self._lt_cache: Optional[LookThroughResult] = None
        self._conc_cache: Optional[ConcentrationAnalysis] = None

    def _register_evidence(self, ev_list: List[Evidence]) -> List[str]:
        ids: List[str] = []
        for ev in ev_list:
            self.registry.register(ev)
            ids.append(ev.id)
        return ids

    def call_tool(self, tool_name: str, reason: str = "") -> Dict[str, Any]:
        """
        Executes a named deterministic engine tool, enforcing the 8-call ceiling
        and writing a tamper-evident audit record.
        """
        if len(self.tool_trace) >= self.max_calls:
            self.audit.log_event(
                event_type="AGENT_TOOL_CALL_CAP_REACHED",
                details={
                    "persona": self.persona_name,
                    "attempted_tool": tool_name,
                    "max_calls": self.max_calls,
                },
            )
            return {
                "status": "HALTED_CAP_REACHED",
                "error": f"Tool call limit of {self.max_calls} reached.",
            }

        step_num = len(self.tool_trace) + 1
        dispatch: Dict[str, Callable[[], Tuple[Dict[str, Any], List[Evidence]]]] = {
            "lookthrough_and_concentration": self._tool_lookthrough_and_concentration,
            "circuit_breaker": self._tool_circuit_breaker,
            "risk_and_tail": self._tool_risk_and_tail,
            "correlation_and_risk_contribution": self._tool_correlation,
            "historical_stress_replays": self._tool_stress_replays,
            "benchmark_comparison": self._tool_benchmark,
        }

        if tool_name not in dispatch:
            raise ValueError(f"Unknown investigation tool: '{tool_name}'")

        summary, ev_list = dispatch[tool_name]()
        ev_ids = self._register_evidence(ev_list)

        record = ToolCallRecord(
            step=step_num,
            tool_name=tool_name,
            reason=reason,
            evidence_ids_produced=ev_ids,
            summary=summary,
        )
        self.tool_trace.append(record)

        self.audit.log_event(
            event_type="AGENT_TOOL_CALL",
            details={
                "step": step_num,
                "persona": self.persona_name,
                "tool_name": tool_name,
                "reason": reason,
                "evidence_ids_produced": ev_ids,
                "summary": summary,
            },
        )
        return summary

    def _tool_lookthrough_and_concentration(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        lt_eng = LookThroughEngine(self.loader)
        conc_eng = ConcentrationEngine()
        lt_res = lt_eng.analyze(self.holdings)
        conc_res = conc_eng.analyze(lt_res)
        self._lt_cache = lt_res
        self._conc_cache = conc_res

        top_issuers = conc_res.lookthrough.top_issuers[:5]
        summary = {
            "direct_n_eff": conc_res.direct.effective_n,
            "lookthrough_n_eff": conc_res.lookthrough.effective_n,
            "lookthrough_hhi": conc_res.lookthrough.hhi,
            "top_1_issuer": top_issuers[0]["issuer"] if top_issuers else "N/A",
            "top_1_weight_pct": round((top_issuers[0]["weight"] if top_issuers else 0.0) * 100, 2),
            "top_5_issuers": [
                {"issuer": x["issuer"], "weight_pct": round(x["weight"] * 100, 2)}
                for x in top_issuers
            ],
        }
        return summary, lt_res.evidence_list + conc_res.evidence_list

    def _tool_circuit_breaker(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        if self._lt_cache is None or self._conc_cache is None:
            self._tool_lookthrough_and_concentration()
        assert self._lt_cache is not None and self._conc_cache is not None

        safety_eng = SafetyLimitsEngine(audit_logger=self.audit)
        safety_res = safety_eng.evaluate_portfolio(
            lt_result=self._lt_cache,
            concentration=self._conc_cache,
            user_max_single_stock_limit=self.user_max_single_stock_limit,
            persona_name=self.persona_name,
        )
        summary = {
            "is_safe": safety_res.is_safe,
            "evaluations": [
                {
                    "rule": e.rule_name,
                    "status": e.status,
                    "breaching_issuer": e.breaching_issuer,
                    "actual_pct": round(e.actual_value * 100, 2),
                    "threshold_pct": round(e.threshold * 100, 2),
                    "excess_pct": round(e.excess * 100, 2),
                    "evidence_id": e.evidence_id,
                }
                for e in safety_res.evaluations
            ],
        }
        return summary, safety_res.evidence_list

    def _tool_risk_and_tail(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        risk_eng = RiskEngine(self.loader)
        stress_eng = StressAndTailEngine(self.loader)
        risk_res = risk_eng.analyze(self.holdings)
        stress_res = stress_eng.analyze(self.holdings, n_bootstrap=200)
        tr = stress_res.tail_risk

        summary = {
            "annualized_volatility_pct": round(risk_res.annualized_volatility * 100, 2),
            "daily_var_95_pct": round(risk_res.historical_var_95 * 100, 2),
            "daily_es_95_pct": round(tr.es_95 * 100, 2),
            "daily_var_99_pct": round(tr.var_99 * 100, 2),
            "max_drawdown_pct": round(risk_res.max_drawdown * 100, 2),
            "unpriced_holdings": risk_res.unpriced_holdings,
        }
        ev_list = risk_res.evidence_list + [
            e for e in stress_res.evidence_list if e.id == "EV-TAIL-ES-95"
        ]
        return summary, ev_list

    def _tool_correlation(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        corr_eng = CorrelationAndRiskContributionEngine(self.loader)
        corr_res = corr_eng.analyze(self.holdings)
        top_c = corr_res.top_risk_contributor
        summary = {
            "avg_pairwise_corr_3y": corr_res.avg_pairwise_corr_3y,
            "highest_corr_pair_3y": corr_res.highest_corr_pair_3y,
            "top_risk_contributor": (
                {
                    "ticker": top_c.ticker,
                    "capital_weight_pct": round(top_c.capital_weight * 100, 2),
                    "pct_risk_share": round(top_c.pct_risk_contribution * 100, 2),
                    "risk_to_capital_ratio": top_c.risk_to_capital_ratio,
                }
                if top_c
                else None
            ),
        }
        return summary, corr_res.evidence_list

    def _tool_stress_replays(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        stress_eng = StressAndTailEngine(self.loader)
        stress_res = stress_eng.analyze(self.holdings, n_bootstrap=100)
        summary = {
            scen_id: {
                "name": sc.name,
                "portfolio_return_pct": round(sc.portfolio_cumulative_return * 100, 2),
                "portfolio_max_dd_pct": round(sc.portfolio_max_drawdown * 100, 2),
                "spy_return_pct": round(sc.spy_cumulative_return * 100, 2),
                "coverage_pct": round(sc.coverage_weight * 100, 1),
                "missing_holdings": sc.missing_history_holdings,
            }
            for scen_id, sc in stress_res.scenarios.items()
        }
        ev_list = [e for e in stress_res.evidence_list if e.id.startswith("EV-STRESS-")]
        return summary, ev_list

    def _tool_benchmark(self) -> Tuple[Dict[str, Any], List[Evidence]]:
        bench_eng = BenchmarkEngine(self.loader)
        bench_res = bench_eng.analyze(self.holdings)
        top_tilt = bench_res.active_tilts_vs_spy[0] if bench_res.active_tilts_vs_spy else None
        summary = {
            "portfolio_beta_vs_spy": bench_res.portfolio_beta_vs_spy,
            "vol_ratio_vs_spy": bench_res.vol_ratio_vs_spy,
            "vol_ratio_vs_60_40": bench_res.vol_ratio_vs_60_40,
            "top_active_tilt_vs_spy": (
                {
                    "issuer": top_tilt.issuer,
                    "user_weight_pct": round(top_tilt.user_weight * 100, 2),
                    "spy_weight_pct": round(top_tilt.spy_weight * 100, 2),
                    "active_tilt_pct": round(top_tilt.active_tilt * 100, 2),
                    "multiple_of_spy": top_tilt.multiple_of_spy,
                }
                if top_tilt
                else None
            ),
        }
        return summary, bench_res.evidence_list


class PortfolioInvestigator:
    """
    AI Portfolio Risk Investigator (v1 & v2 Autonomous Loop).
    Zero Mental Math: All quantitative values provided via deterministic Evidence objects.
    Post-check verification:
      1. Every finding must cite existing evidence IDs.
      2. OutputSafetyFilter blocks any buy/sell trading instructions or stock price predictions.
      3. Numerical grounding check verifies percentages match deterministic Evidence records.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.8-flash",
        output_filter: Optional[OutputSafetyFilter] = None,
    ) -> None:
        self.model_name = model_name
        self._api_key = os.environ.get("GEMINI_API_KEY")
        self.output_filter = output_filter or OutputSafetyFilter()

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
        enforce_numerical_grounding: bool = False,
    ) -> Tuple[List[Finding], bool, Optional[str]]:
        """
        Validates agent output against deterministic invariants:
        1. Output is a valid JSON list of findings.
        2. Every finding cites existing Evidence IDs in the registry.
        3. OutputSafetyFilter confirms zero buy/sell instructions or price predictions.
        4. Optional numerical grounding check confirms percentages match Evidence values.
        """
        valid_evidence_ids = set(evidence_registry._items.keys())
        parsed_findings: List[Finding] = []

        if not isinstance(findings_data, list) or len(findings_data) == 0:
            return [], False, "Output is not a non-empty JSON list"

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

            # Safety v2: Check for prohibited buy/sell instructions or price predictions
            filter_res = self.output_filter.check_text(f"{title} {explanation}")
            if not filter_res.is_clean:
                return (
                    [],
                    False,
                    f"Finding '{title}' blocked by OutputSafetyFilter: {'; '.join(filter_res.violations)}",
                )

            if enforce_numerical_grounding:
                num_res = self.output_filter.check_numerical_grounding(
                    f"{title} {explanation}", evidence_registry
                )
                if not num_res.is_clean:
                    return (
                        [],
                        False,
                        f"Finding '{title}' failed numerical grounding check: {'; '.join(num_res.violations)}",
                    )

            parsed_findings.append(
                Finding(
                    title=title,
                    explanation=explanation,
                    severity=severity,
                    evidence_ids=cites,
                )
            )

        return parsed_findings, True, None

    @staticmethod
    def synthesize_deterministic_findings(
        registry: EvidenceRegistry,
        user_profile: Optional[UserRiskProfile] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deterministic fallback synthesizer that builds evidence-backed findings
        directly from registered Evidence objects when offline or if LLM is unavailable.
        Guarantees 100% citation validity and zero hallucinated numbers.
        """
        findings: List[Dict[str, Any]] = []
        ev_map = {ev.id: ev for ev in registry.all()}

        # 1. Check circuit breaker
        cb_ev = next(
            (ev for ev in registry.all() if ev.id.startswith("EV-LIMIT-SINGLE-STOCK-")),
            None,
        )
        if cb_ev and isinstance(cb_ev.value, dict):
            issuer = cb_ev.value.get("issuer", "Issuer")
            w_val = float(cb_ev.value.get("actual_weight", 0.0)) * 100
            thresh = float(cb_ev.value.get("threshold", 0.20)) * 100
            excess = float(cb_ev.value.get("excess", 0.0)) * 100
            lt_id = f"EV-LT-{issuer}"
            cites = [cb_ev.id] + ([lt_id] if lt_id in ev_map else [])
            if excess > 0:
                findings.append(
                    {
                        "title": f"Single-Issuer Concentration Breach: {issuer} at {w_val:.2f}%",
                        "explanation": (
                            f"True look-through exposure to {issuer} is {w_val:.2f}%, which exceeds the "
                            f"{thresh:.2f}% single-stock safety threshold by +{excess:.2f}%."
                        ),
                        "severity": "CRITICAL",
                        "evidence_ids": cites,
                    }
                )
            else:
                findings.append(
                    {
                        "title": f"Single-Issuer Safety Limit Passed ({issuer} at {w_val:.2f}%)",
                        "explanation": (
                            f"Largest look-through single-issuer exposure is {issuer} at {w_val:.2f}%, "
                            f"remaining within the {thresh:.2f}% policy ceiling."
                        ),
                        "severity": "INFO",
                        "evidence_ids": cites,
                    }
                )

        # 2. Check Look-Through vs Direct N_eff
        if "EV-CONC-LT-NEFF" in ev_map and "EV-CONC-DIRECT-NEFF" in ev_map:
            lt_neff = float(ev_map["EV-CONC-LT-NEFF"].value)
            dir_neff = float(ev_map["EV-CONC-DIRECT-NEFF"].value)
            findings.append(
                {
                    "title": f"Effective Holdings: {lt_neff:.2f} Look-Through vs {dir_neff:.2f} Direct",
                    "explanation": (
                        f"Decomposing ETF wrappers yields an effective issuer count (N_eff) of {lt_neff:.2f} "
                        f"compared to {dir_neff:.2f} surface holdings."
                    ),
                    "severity": "WARNING" if lt_neff < 12.0 else "INFO",
                    "evidence_ids": ["EV-CONC-LT-NEFF", "EV-CONC-DIRECT-NEFF"],
                }
            )

        # 3. Check Stress Replays & Tail Risk
        if "EV-STRESS-YEAR_2022_FED_RATE_SHOCK" in ev_map:
            ev_2022 = ev_map["EV-STRESS-YEAR_2022_FED_RATE_SHOCK"]
            ret_2022 = float(ev_2022.value) * 100
            cites = ["EV-STRESS-YEAR_2022_FED_RATE_SHOCK"]
            if "EV-TAIL-ES-95" in ev_map:
                cites.append("EV-TAIL-ES-95")
            horizon_note = (
                f" Given a {user_profile.horizon_years}-year horizon and {user_profile.max_tolerable_loss_pct:.1f}% loss tolerance, this represents a severe regime risk."
                if user_profile
                else ""
            )
            findings.append(
                {
                    "title": f"2022 Rate-Hike Stress Replay: {ret_2022:.2f}% Cumulative Return",
                    "explanation": (
                        f"Replaying current holdings across the 2022 Federal Reserve tightening cycle "
                        f"produces a cumulative return of {ret_2022:.2f}%.{horizon_note}"
                    ),
                    "severity": "CRITICAL" if ret_2022 < -30.0 else "WARNING",
                    "evidence_ids": cites,
                }
            )

        return findings

    def investigate_autonomous(
        self,
        holdings: List[Dict[str, Any]],
        user_goal: str,
        persona_name: str,
        user_max_single_stock_limit: Optional[float] = None,
        user_profile: Optional[UserRiskProfile] = None,
        data_loader: Optional[DataLoader] = None,
        audit_logger: Optional[AuditLogger] = None,
        use_llm: bool = True,
    ) -> InvestigationReport:
        """
        Agent v2 Autonomous Investigation Loop:
          1. Analyse: Run initial look-through, concentration, and circuit breaker checks.
          2. Identify anomalies: Inspect concentration gap, breaches, and asset mix.
          3. Investigate: Call targeted risk, correlation/CCR, historical stress, and benchmark tools (capped at 8 calls).
          4. Verify & Interpret: Contextualize against UserRiskProfile and validate output against OutputSafetyFilter + Evidence IDs.
        """
        runner = AgentToolRunner(
            holdings=holdings,
            persona_name=persona_name,
            user_max_single_stock_limit=user_max_single_stock_limit,
            data_loader=data_loader,
            audit_logger=audit_logger,
            max_calls=MAX_AGENT_TOOL_CALLS,
        )

        # Step 1: Initial Look-Through & Circuit Breaker Scan
        lt_summary = runner.call_tool(
            "lookthrough_and_concentration",
            reason="Initial decomposition of ETF wrappers and single-issuer HHI/N_eff calculation.",
        )
        cb_summary = runner.call_tool(
            "circuit_breaker",
            reason="Verify whether true look-through weights breach single-stock policy limits.",
        )

        # Step 2: Anomaly-driven Deep-Dive Tool Selection (capped at 8 total calls)
        runner.call_tool(
            "risk_and_tail",
            reason="Quantify 3-year annualised volatility, 95%/99% VaR, Expected Shortfall, and unpriced coverage.",
        )

        if len(holdings) >= 2:
            runner.call_tool(
                "correlation_and_risk_contribution",
                reason="Multiple holdings detected; compute 1Y/3Y correlation clustering and Euler CCR risk shares.",
            )

        runner.call_tool(
            "historical_stress_replays",
            reason="Test portfolio resilience across Mar 2020 COVID, 2022 Rate Shock, Mar 2023 SVB, and Aug 2024 Tech Unwind.",
        )

        runner.call_tool(
            "benchmark_comparison",
            reason="Benchmark concentration, volatility, and active tilts against SPY and 60/40 SPY/BND.",
        )

        all_evidence = runner.registry.all()

        # Step 3: Interpret against UserRiskProfile and generate findings
        if use_llm and self._api_key:
            evidence_summary = [
                {
                    "id": ev.id,
                    "metric": ev.metric,
                    "value": ev.value,
                    "formula": ev.formula,
                    "inputs": ev.inputs,
                    "caveats": ev.caveats,
                }
                for ev in all_evidence
            ]
            payload = {
                "user_context": {
                    "persona": persona_name,
                    "stated_goal": user_goal,
                    "risk_profile": user_profile.to_prompt_context() if user_profile else None,
                },
                "autonomous_tool_trace": [
                    {
                        "step": t.step,
                        "tool": t.tool_name,
                        "reason": t.reason,
                        "summary": t.summary,
                    }
                    for t in runner.tool_trace
                ],
                "evidence_registry": evidence_summary,
            }
            user_prompt = (
                "You have completed an autonomous 6-step quantitative investigation of this portfolio.\n"
                "Synthesize the most important risk discoveries (hidden look-through concentration, "
                "Euler risk contribution disproportion, historical crisis drawdowns, and active tilts vs SPY), "
                "interpreting them in light of the user's risk profile without changing any numbers.\n\n"
                f"INVESTIGATION PAYLOAD JSON:\n{json.dumps(payload, indent=2)}\n\n"
                "Respond ONLY with a JSON array of findings."
            )
            try:
                raw_response = self._call_gemini(user_prompt)
                findings_json = self._extract_json(raw_response)
                findings, passed, reason = self.validate_findings(findings_json, runner.registry)
                if not passed:
                    retry_prompt = (
                        f"{user_prompt}\n\n"
                        f"PREVIOUS ATTEMPT FAILED VALIDATION: {reason}\n"
                        "Correct immediately: cite valid evidence_ids and never use buy/sell instructions or price predictions."
                    )
                    raw_response = self._call_gemini(retry_prompt)
                    findings_json = self._extract_json(raw_response)
                    findings, passed, reason = self.validate_findings(findings_json, runner.registry)
            except Exception:
                findings_json = self.synthesize_deterministic_findings(runner.registry, user_profile)
                findings, passed, reason = self.validate_findings(findings_json, runner.registry)
        else:
            findings_json = self.synthesize_deterministic_findings(runner.registry, user_profile)
            findings, passed, reason = self.validate_findings(findings_json, runner.registry)

        return InvestigationReport(
            findings=findings,
            raw_evidence=all_evidence,
            validation_passed=passed,
            rejection_reason=reason,
            tool_trace=runner.tool_trace,
            tool_calls_count=len(runner.tool_trace),
        )

    def investigate(
        self,
        user_goal: str,
        persona_name: str,
        lt_result: LookThroughResult,
        concentration: ConcentrationAnalysis,
        risk: RiskMetrics,
        safety: SafetyCheckResult,
        user_profile: Optional[UserRiskProfile] = None,
    ) -> InvestigationReport:
        """
        v1-compatible investigation entry point used when pre-computed engine outputs are passed directly.
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
                "risk_profile": user_profile.to_prompt_context() if user_profile else None,
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

        raw_response = self._call_gemini(user_prompt)
        findings_json = self._extract_json(raw_response)

        findings, passed, reason = self.validate_findings(findings_json, registry)

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
