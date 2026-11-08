"""Portfolio X-Ray CLI: Run risk audits, what-if simulations, and stress tests directly from terminal."""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure v-ui directory and project root are in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from data import SAMPLE_PORTFOLIOS
    from engine import (
        Portfolio,
        Holding,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
    )
    from agent import ask_risk_analyst
except ImportError:
    from portfolio_xray.data import SAMPLE_PORTFOLIOS
    from portfolio_xray.engine import (
        Portfolio,
        Holding,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
    )
    from portfolio_xray.agent import ask_risk_analyst


def main():
    print("=" * 70)
    print("  PORTFOLIO X-RAY: AI-POWERED RISK INTELLIGENCE (v0 DEMO)")
    print("=" * 70)

    preset_key = "accidental_tech_overweight"
    data = SAMPLE_PORTFOLIOS[preset_key]
    print(f"\nAUDITING PORTFOLIO: {data['name']}")
    print(f"Description: {data['description']}")
    print(f"Profile: {data['user_profile']['risk_tolerance']} risk tolerance")
    print("-" * 70)

    portfolio = Portfolio(
        holdings=[Holding(ticker=h["ticker"], shares=h["shares"], price=h["price"]) for h in data["holdings"]],
        user_profile=data["user_profile"]
    )

    exposures = compute_lookthrough_exposure(portfolio)
    thematic = compute_thematic_exposure(exposures)
    metrics = compute_risk_metrics(portfolio, exposures)
    stress = run_stress_tests(portfolio, exposures)

    print("\n1. LOOK-THROUGH EXPOSURES (TOP 5 EFFECTIVE ASSETS):")
    for exp in exposures[:5]:
        indirect_info = ""
        if exp.indirect_weight > 0:
            sources = ", ".join(f"{k} (+{v*100:.1f}%)" for k, v in exp.contributing_sources.items())
            indirect_info = f" [Indirect from {sources}]"
        print(f"   * {exp.ticker:<6} ({exp.name:<30}): {exp.total_weight*100:>5.1f}% (${exp.total_dollar_value:>9,.2f}){indirect_info}")

    print("\n2. ECONOMIC & THEMATIC EXPOSURES:")
    for th in thematic[:3]:
        print(f"   * {th['theme']:<32}: {th['weight']*100:>5.1f}% portfolio weight")

    print("\n3. QUANTITATIVE RISK METRICS:")
    print(f"   * Portfolio Value           : ${metrics['portfolio_value']:,.2f}")
    print(f"   * Concentration (HHI)       : {metrics['hhi']} ({metrics['concentration_tier']})")
    print(f"   * Top 1 / Top 3 Concentration: {metrics['top_1_concentration']}% / {metrics['top_3_concentration']}%")
    print(f"   * Annualized Volatility     : {metrics['annual_volatility']}%")
    print(f"   * 1-Month 95% VaR           : ${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)")

    print("\n4. HISTORICAL STRESS TEST DRAWDOWN ESTIMATES:")
    for s in stress:
        print(f"   * {s['name']:<48}: {s['drawdown_pct']:>6.1f}% (${s['dollar_drawdown']:>10,.2f})")

    print("\n5. PRE-TRADE WHAT-IF SIMULATION (+50% INTO NVDA):")
    sim = simulate_what_if(portfolio, {"NVDA": 50.0, "SPY": 50.0})
    print(f"   * Volatility Change : {sim['current_metrics']['annual_volatility']}% -> {sim['proposed_metrics']['annual_volatility']}% ({sim['metrics_delta']['volatility_change']:+,.2f}%)")
    print(f"   * Top 1 Exposure    : {sim['current_metrics']['top_1_concentration']}% -> {sim['proposed_metrics']['top_1_concentration']}%")
    print(f"   * 1-M 95% VaR       : ${sim['proposed_metrics']['var_95_1m_dollar']:,.2f}")
    if sim["profile_warnings"]:
        for w in sim["profile_warnings"]:
            print(f"   * ⚠️  WARNING: {w}")

    print("\n" + "=" * 70)
    print("To explore the interactive Web UI, visit http://localhost:8080")
    print("=" * 70)


if __name__ == "__main__":
    main()
