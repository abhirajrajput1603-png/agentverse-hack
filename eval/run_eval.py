from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Tuple


from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.risk import RiskEngine
from xray.safety.limits import SafetyLimitsEngine


def evaluate_all_cases() -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    Automated evaluation harness for Portfolio X-Ray.
    Validates all 4 synthetic benchmark portfolios against authoritative
    ground-truth criteria defined in data/synthetic_benchmark_portfolios.json.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    lt_engine = LookThroughEngine(loader)
    conc_engine = ConcentrationEngine()
    risk_engine = RiskEngine(loader)
    safety_engine = SafetyLimitsEngine()

    results: List[Dict[str, Any]] = []
    passed_count = 0
    failed_count = 0

    print("=" * 84)
    print("PORTFOLIO X-RAY BENCHMARK EVALUATION HARNESS (ROUND 1)")
    print("Authoritative Ground Truth Validation · UCL AgentVerse 2026 Standard")
    print("=" * 84)

    for case in portfolios:
        cid = case["case_id"]
        name = case["name"]
        holdings = case["holdings"]
        limit = case["max_risk_tolerance_single_stock"]
        gt = case.get("ground_truth_lookthrough", {})

        # Execute deterministic pipeline
        lt_res = lt_engine.analyze(holdings)
        conc = conc_engine.analyze(lt_res)
        risk = risk_engine.analyze(holdings)
        safety = safety_engine.evaluate_portfolio(
            lt_result=lt_res,
            concentration=conc,
            user_max_single_stock_limit=limit,
            persona_name=case.get("persona"),
        )

        case_passed = True
        failure_reasons = []

        # Check 1: Weight conservation
        if abs(lt_res.total_weight - 1.0) > 1e-7:
            case_passed = False
            failure_reasons.append(f"Weight conservation failed: sum={lt_res.total_weight:.8f}")

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
                # Circuit breaker must BREACH
                primary_eval = safety.evaluations[0]
                if primary_eval.status != "BREACH" or primary_eval.breaching_issuer != "NVDA":
                    case_passed = False
                    failure_reasons.append(f"Expected circuit breaker BREACH on NVDA, got {primary_eval.status}")

        elif cid == "CASE-02-AI-CAPEX-ENERGY-THEME":
            # Direct holdings preserved, VST identified as unpriced
            if "VST" not in lt_res.issuer_exposures or lt_res.issuer_exposures["VST"].total_weight != 0.25:
                case_passed = False
                failure_reasons.append("VST weight not preserved at 0.25")
            if "VST" not in risk.unpriced_holdings:
                case_passed = False
                failure_reasons.append("VST was not flagged as unpriced in risk engine")

        elif cid == "CASE-03-RATE-SHOCK-DURATION-TRAP":
            # ARKK decomposed, TLT maintained as aggregate
            tlt_exp = lt_res.issuer_exposures.get("TLT")
            if not tlt_exp or tlt_exp.total_weight != 0.50 or not tlt_exp.is_fund_undisclosed:
                case_passed = False
                failure_reasons.append("TLT not properly maintained as 0.50 undisclosed aggregate fund")
            if "TSLA" not in lt_res.issuer_exposures:
                case_passed = False
                failure_reasons.append("ARKK constituent TSLA not found in lookthrough")

        elif cid == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK":
            # Max single stock < 10%, circuit breaker PASS
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
            "evidence_count": len(lt_res.evidence_list) + len(conc.evidence_list) + len(risk.evidence_list) + len(safety.evidence_list),
        }
        results.append(result_row)

    # Print summary table
    print(f"\n{'CASE ID':<36} | {'STATUS':<6} | {'TOP ISSUER':<10} | {'TOP WT':<8} | {'CB STATUS':<10} | {'EVIDENCE':<8}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['case_id']:<36} | {r['status']:<6} | {r['top_stock']:<10} | "
            f"{r['top_stock_weight']*100:>6.2f}% | {r['circuit_breaker']:<10} | {r['evidence_count']:>8}"
        )
        if r["failures"]:
            for f in r["failures"]:
                print(f"    --> REASON: {f}")

    print("-" * 88)
    print(f"TOTAL BENCHMARKS: {len(portfolios)} | PASSED: {passed_count} | FAILED: {failed_count}")
    print("=" * 88)

    return passed_count, failed_count, results


if __name__ == "__main__":
    passed, failed, _ = evaluate_all_cases()
    if failed > 0:
        sys.exit(1)
    print("\nALL BENCHMARK EVALUATIONS PASSED SUCCESSFULLY.")
