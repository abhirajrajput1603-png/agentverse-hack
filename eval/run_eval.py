from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Tuple

from xray.agent.investigator import PortfolioInvestigator
from xray.data_loader import DataLoader
from xray.engine.benchmark import BenchmarkEngine
from xray.engine.concentration import ConcentrationEngine
from xray.engine.correlation import CorrelationAndRiskContributionEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.risk import RiskEngine
from xray.engine.stress import StressAndTailEngine
from xray.engine.whatif import WhatIfSimulator
from xray.safety.limits import SafetyLimitsEngine


def evaluate_all_cases() -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Automated evaluation harness for Portfolio X-Ray (Round 1 & Round 2).
    Validates all 4 synthetic benchmark portfolios against authoritative
    ground-truth criteria defined in data/synthetic_benchmark_portfolios.json.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    lt_engine = LookThroughEngine(loader)
    conc_engine = ConcentrationEngine()
    risk_engine = RiskEngine(loader)
    corr_engine = CorrelationAndRiskContributionEngine(loader)
    stress_engine = StressAndTailEngine(loader)
    bench_engine = BenchmarkEngine(loader)
    safety_engine = SafetyLimitsEngine()
    whatif_sim = WhatIfSimulator(data_loader=loader)
    investigator = PortfolioInvestigator()

    results: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    print("=" * 96)
    print("PORTFOLIO X-RAY BENCHMARK EVALUATION HARNESS (ROUND 1 & ROUND 2 COMPLETE)")
    print("Authoritative Ground Truth Validation · UCL AgentVerse 2026 Standard")
    print("=" * 96)

    for case in portfolios:
        cid = case["case_id"]
        name = case["name"]
        holdings = case["holdings"]
        limit = case["max_risk_tolerance_single_stock"]
        gt = case.get("ground_truth_lookthrough", {})

        # Execute full deterministic pipeline (Rounds 1 + 2)
        lt_res = lt_engine.analyze(holdings)
        conc = conc_engine.analyze(lt_res)
        risk = risk_engine.analyze(holdings)
        corr = corr_engine.analyze(holdings)
        stress = stress_engine.analyze(holdings, n_bootstrap=200)
        bench = bench_engine.analyze(holdings)
        safety = safety_engine.evaluate_portfolio(
            lt_result=lt_res,
            concentration=conc,
            user_max_single_stock_limit=limit,
            persona_name=case.get("persona"),
        )
        agent_rep = investigator.investigate_autonomous(
            holdings=holdings,
            user_goal=case.get("stated_goal", ""),
            persona_name=case.get("persona", ""),
            user_max_single_stock_limit=limit,
            data_loader=loader,
            use_llm=False,
        )

        case_passed = True
        failure_reasons = []

        # Universal Invariant 1: Look-Through Weight Conservation
        if abs(lt_res.total_weight - 1.0) > 1e-7:
            case_passed = False
            failure_reasons.append(f"Weight conservation failed: sum={lt_res.total_weight:.8f}")

        # Universal Invariant 2: Euler Risk Decomposition (sum(CCR) == sigma_p)
        if not corr.ccr_sum_check_passed:
            case_passed = False
            failure_reasons.append(f"Euler CCR check failed: {corr.ccr_sum} != {corr.portfolio_volatility}")

        # Universal Invariant 3: Tail Risk Ordering (ES_95 <= VaR_95)
        if stress.tail_risk.es_95 > stress.tail_risk.var_95 + 1e-9:
            case_passed = False
            failure_reasons.append(f"Tail risk ordering violated: ES_95 ({stress.tail_risk.es_95}) > VaR_95 ({stress.tail_risk.var_95})")

        # Universal Invariant 4: Agent v2 validation & tool cap
        if not agent_rep.validation_passed or agent_rep.tool_calls_count > 8:
            case_passed = False
            failure_reasons.append(f"Agent v2 validation failed ({agent_rep.rejection_reason}) or exceeded 8-call cap")

        # Case-specific ground truth validations
        if cid == "CASE-01-DIVERSIFICATION-ILLUSION":
            nvda_exp = lt_res.issuer_exposures.get("NVDA")
            if not nvda_exp:
                case_passed = False
                failure_reasons.append("NVDA not found in look-through exposures")
            else:
                expected_eff = gt.get("NVDA_total_effective", 0.3471)
                if abs(nvda_exp.total_weight - expected_eff) > 1e-3:
                    case_passed = False
                    failure_reasons.append(
                        f"NVDA look-through weight {nvda_exp.total_weight:.4f} != expected {expected_eff:.4f}"
                    )
                primary_eval = safety.evaluations[0]
                if primary_eval.status != "BREACH" or primary_eval.breaching_issuer != "NVDA":
                    case_passed = False
                    failure_reasons.append(f"Expected circuit breaker BREACH on NVDA, got {primary_eval.status}")

            # Adversarial What-If check (£15,000 into NVDA)
            whatif_res = whatif_sim.simulate_add_cash(
                current_holdings=holdings,
                target_ticker="NVDA",
                additional_cash_gbp=15000.0,
                persona_name=case.get("persona", ""),
                user_max_single_stock_limit=limit,
            )
            if whatif_res.after_safety.is_safe or whatif_res.after_lt.issuer_exposures["NVDA"].total_weight < 0.49:
                case_passed = False
                failure_reasons.append("Case 1 adversarial what-if (£15k NVDA) failed to trigger hard breach > 49%")

        elif cid == "CASE-02-AI-CAPEX-ENERGY-THEME":
            if "VST" not in lt_res.issuer_exposures or lt_res.issuer_exposures["VST"].total_weight != 0.25:
                case_passed = False
                failure_reasons.append("VST weight not preserved at 0.25")
            if "VST" not in risk.unpriced_holdings:
                case_passed = False
                failure_reasons.append("VST was not flagged as unpriced in risk engine")
            if "VST" not in stress.scenarios["YEAR_2022_FED_RATE_SHOCK"].missing_history_holdings:
                case_passed = False
                failure_reasons.append("VST was not flagged as missing history in stress replay")

        elif cid == "CASE-03-RATE-SHOCK-DURATION-TRAP":
            tlt_exp = lt_res.issuer_exposures.get("TLT")
            if not tlt_exp or tlt_exp.total_weight != 0.50 or not tlt_exp.is_fund_undisclosed:
                case_passed = False
                failure_reasons.append("TLT not properly maintained as 0.50 undisclosed aggregate fund")
            if "TSLA" not in lt_res.issuer_exposures:
                case_passed = False
                failure_reasons.append("ARKK constituent TSLA not found in lookthrough")
            # 2022 Fed Rate-Hike Stress Replay co-drawdown check
            s2022 = stress.scenarios["YEAR_2022_FED_RATE_SHOCK"]
            if s2022.portfolio_cumulative_return > -0.40:
                case_passed = False
                failure_reasons.append(f"Expected severe 2022 rate-shock co-drawdown < -40%, got {s2022.portfolio_cumulative_return:.4f}")

        elif cid == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK":
            top_stocks = lt_res.sorted_exposures(exclude_residual_and_cash=True)
            top_stock = top_stocks[0]
            if top_stock.total_weight >= 0.10:
                case_passed = False
                failure_reasons.append(f"Top stock {top_stock.issuer} ({top_stock.total_weight:.4f}) exceeded 0.10")
            primary_eval = safety.evaluations[0]
            if primary_eval.status != "PASS":
                case_passed = False
                failure_reasons.append(f"Expected circuit breaker PASS, got {primary_eval.status}")

        if case_passed:
            passed_count += 1
            status_str = "PASS"
        else:
            failed_count += 1
            status_str = "FAIL"

        result_row = {
            "case_id": cid,
            "name": name,
            "status": status_str,
            "failures": failure_reasons,
            "top_stock": lt_res.sorted_exposures(exclude_residual_and_cash=True)[0].issuer if lt_res.sorted_exposures(exclude_residual_and_cash=True) else "N/A",
            "top_stock_weight": lt_res.sorted_exposures(exclude_residual_and_cash=True)[0].total_weight if lt_res.sorted_exposures(exclude_residual_and_cash=True) else 0.0,
            "circuit_breaker": safety.evaluations[0].status if safety.evaluations else "N/A",
            "beta_vs_spy": bench.portfolio_beta_vs_spy,
            "es_95": stress.tail_risk.es_95,
            "evidence_count": len(agent_rep.raw_evidence),
        }
        results.append(result_row)

    print(
        f"\n{'CASE ID':<36} | {'STATUS':<6} | {'TOP ISSUER':<10} | {'TOP WT':<8} | {'CB STATUS':<9} | {'BETA':<6} | {'95% ES':<8} | {'EVIDENCE':<8}"
    )
    print("-" * 108)
    for r in results:
        print(
            f"{r['case_id']:<36} | {r['status']:<6} | {r['top_stock']:<10} | "
            f"{r['top_stock_weight']*100:>6.2f}% | {r['circuit_breaker']:<9} | "
            f"{r['beta_vs_spy']:>5.2f}x | {r['es_95']*100:>6.2f}% | {r['evidence_count']:>8}"
        )
        if r["failures"]:
            for f in r["failures"]:
                print(f"    --> REASON: {f}")

    print("-" * 108)
    print(f"TOTAL BENCHMARKS: {len(portfolios)} | PASSED: {passed_count} | FAILED: {failed_count}")
    print("=" * 108)
    return passed_count, failed_count, results


if __name__ == "__main__":
    passed, failed, _ = evaluate_all_cases()
    sys.exit(0 if failed == 0 else 1)
