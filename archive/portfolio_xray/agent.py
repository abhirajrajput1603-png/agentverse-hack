"""AI Risk Intelligence Analyst for Portfolio X-Ray.

Connects to Google Gemini API using standard library urllib.request (zero extra dependencies).
Includes deterministic rule-based fallback when offline or without an API key.
"""

from __future__ import annotations
import os
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional

from portfolio_xray.engine import (
    Portfolio,
    compute_lookthrough_exposure,
    compute_thematic_exposure,
    compute_risk_metrics,
    run_stress_tests,
    simulate_what_if,
)


def get_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment or .env file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    # Check local .env file
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return None


SYSTEM_INSTRUCTION = """You are Portfolio X-Ray, an institutional-grade autonomous Portfolio Risk Intelligence Analyst.
Your mission is to help everyday investors understand the true economic risks, hidden concentrations, and asset interactions in their portfolios.

CORE PRINCIPLES:
1. Educate, never predict: Never forecast price targets or recommend buying/selling specific assets. Frame everything around risk exposure, concentration, and historical scenario resilience.
2. Unpack Look-Through: Always highlight overlapping holdings (e.g., direct stock + ETFs holding the same stock).
3. Explain Jargon Simply: Explain terms like Value at Risk (VaR), Herfindahl Index (HHI), and Beta in clear, everyday language.
4. Stress Test Honestly: Show what happens in adverse market scenarios (rate shocks, recessions, tech selloffs).
5. Ground in User Profile: Compare risks against the user's stated risk tolerance, horizon, and loss capacity.
"""


def format_portfolio_context(portfolio: Portfolio) -> str:
    """Prepares structured, inspectable portfolio risk evidence for the AI agent."""
    exposures = compute_lookthrough_exposure(portfolio)
    thematic = compute_thematic_exposure(exposures)
    metrics = compute_risk_metrics(portfolio, exposures)
    stress = run_stress_tests(portfolio, exposures)

    lines = [
        f"TOTAL PORTFOLIO VALUE: ${metrics['portfolio_value']:,.2f}",
        f"INVESTOR PROFILE: Risk Tolerance: {portfolio.user_profile.get('risk_tolerance')}, Horizon: {portfolio.user_profile.get('horizon_years')} yrs, Loss Capacity: {portfolio.user_profile.get('loss_capacity')}",
        "",
        "DIRECT HOLDINGS:",
    ]
    for h in portfolio.holdings:
        lines.append(f"- {h.ticker}: {h.shares:.2f} shares @ ${h.price:.2f} = ${h.market_value:,.2f} ({h.market_value/portfolio.total_value*100:.1f}%)")

    lines.append("\nEFFECTIVE LOOK-THROUGH EXPOSURES (TOP 6):")
    for exp in exposures[:6]:
        overlap_detail = ""
        if exp.contributing_sources:
            sources_str = ", ".join(f"{src}: +{pct*100:.1f}%" for src, pct in exp.contributing_sources.items())
            overlap_detail = f" [Includes indirect from {sources_str}]"
        lines.append(f"- {exp.ticker} ({exp.name}): Total Effective: {exp.total_weight*100:.1f}% (${exp.total_dollar_value:,.2f}){overlap_detail}")

    lines.append("\nTHEMATIC & ECONOMIC EXPOSURES:")
    for th in thematic[:4]:
        lines.append(f"- {th['theme']}: {th['weight']*100:.1f}% portfolio weight")

    lines.append("\nQUANTITATIVE RISK METRICS:")
    lines.append(f"- Concentration Tier: {metrics['concentration_tier']} (HHI: {metrics['hhi']})")
    lines.append(f"- Top 1 Holding: {metrics['top_1_concentration']}%, Top 3 Holdings: {metrics['top_3_concentration']}%")
    lines.append(f"- Annualized Volatility: {metrics['annual_volatility']}%")
    lines.append(f"- 1-Month 95% Value at Risk (VaR): {metrics['var_95_1m_pct']}% (${metrics['var_95_1m_dollar']:,.2f})")
    lines.append(f"- Expected Shortfall (CVaR): {metrics['expected_shortfall_95_pct']}% (${metrics['expected_shortfall_95_dollar']:,.2f})")

    lines.append("\nHISTORICAL & MACRO STRESS TEST ESTIMATES:")
    for st in stress:
        lines.append(f"- {st['name']}: {st['drawdown_pct']}% drawdown (${st['dollar_drawdown']:,.2f})")

    return "\n".join(lines)


def ask_risk_analyst(
    portfolio: Portfolio,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generates an evidence-backed risk analysis response using Gemini or offline fallback."""
    context_str = format_portfolio_context(portfolio)
    api_key = get_api_key()

    if api_key and not api_key.startswith("AQ.Ab8RN6JqCrilO84"): # Check if mock or real
        try:
            return call_gemini_api(api_key, context_str, user_query, conversation_history)
        except Exception as e:
            # Fall back gracefully to structured offline analyst
            return generate_offline_risk_response(portfolio, user_query, str(e))
    else:
        return generate_offline_risk_response(portfolio, user_query)


def call_gemini_api(
    api_key: str,
    context: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Invokes Google Gemini REST API using standard library urllib.request."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    prompt = f"""{SYSTEM_INSTRUCTION}

HERE IS THE INVESTOR'S CURRENT PORTFOLIO RISK AUDIT:
---------------------------------------------------
{context}
---------------------------------------------------

USER QUESTION:
"{query}"

Please provide an insightful, structured, professional response covering:
1. Executive Risk Assessment
2. Hidden or Overlapping Exposures
3. Quantitative Risk & Downside Impact
4. Practical Risk Questions for the Investor to Consider
"""

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=12) as response:
        result = json.loads(response.read().decode("utf-8"))
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        return {
            "source": "gemini-2.5-flash",
            "response": text,
        }


def generate_offline_risk_response(
    portfolio: Portfolio,
    query: str,
    error_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Generates deterministic, institutional-quality risk analysis without network dependencies."""
    exposures = compute_lookthrough_exposure(portfolio)
    metrics = compute_risk_metrics(portfolio, exposures)
    thematic = compute_thematic_exposure(exposures)
    stress = run_stress_tests(portfolio, exposures)

    q_lower = query.lower()

    # Check for "what if" or allocation queries
    if "what if" in q_lower or "50%" in q_lower or "put" in q_lower or "buy" in q_lower:
        # Detect ticker mentioned
        target_ticker = "NVDA"
        for t in ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "TLT", "GLD", "SPY", "QQQ"]:
            if t.lower() in q_lower:
                target_ticker = t
                break

        sim = simulate_what_if(portfolio, {target_ticker: 50.0, "SPY": 50.0})
        prop_m = sim["proposed_metrics"]
        cur_m = sim["current_metrics"]
        delta_vol = sim["metrics_delta"]["volatility_change"]

        response_text = f"""### What-If Risk Simulation: 50% Allocation to {target_ticker}

**1. Exposure & Look-Through Impact**
* If you allocate 50% directly into **{target_ticker}**, your effective concentration in that single asset spikes from **{cur_m['top_1_concentration']}%** to **{prop_m['top_1_concentration']}%**.
* Your portfolio's concentration index (HHI) changes by **{sim['metrics_delta']['hhi_change']:+,.1f} points**, moving into the **{prop_m['concentration_tier']}** zone.

**2. Downside & Volatility Trade-Off**
* **Annualized Volatility:** Moves from **{cur_m['annual_volatility']}%** to **{prop_m['annual_volatility']}%** ({delta_vol:+,.2f}% shift).
* **1-Month 95% Value at Risk (VaR):** Increases to **${prop_m['var_95_1m_dollar']:,.2f}** ({prop_m['var_95_1m_pct']}% of total capital), meaning in 1 out of 20 months, you should statistically expect to lose at least this amount.

**3. Stress Scenario Resilience**
* Under a **2022 Fed Rate Shock & Multiple Compression**, the proposed portfolio would face an estimated **{sim['stress_comparison'][0]['proposed_drawdown_pct']}%** drawdown compared to **{sim['stress_comparison'][0]['current_drawdown_pct']}%** currently.

**4. Investor Profile Check**
{('⚠️ **Risk Threshold Warning:** ' + ' '.join(sim['profile_warnings'])) if sim['profile_warnings'] else '✅ This adjustment remains within your stated risk tolerance bounds.'}
"""
    else:
        # General Portfolio X-Ray Audit response
        top_exp = exposures[0] if exposures else None
        top_theme = thematic[0] if thematic else None
        worst_stress = min(stress, key=lambda s: s["drawdown_pct"]) if stress else None

        response_text = f"""### Portfolio X-Ray: Autonomous Risk Audit

**1. Effective Look-Through & Hidden Overlaps**
* **Top Exposure:** Your highest single economic exposure is **{top_exp.ticker if top_exp else 'N/A'} ({top_exp.name if top_exp else ''})** at **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}).
* {'Notice that holding both direct shares and broad ETFs (such as QQQ or VOO) concentrates your exposure far more than the nominal position size shows.' if top_exp and top_exp.indirect_weight > 0.02 else 'Your positions have minimal unintended ETF overlap.'}

**2. Key Thematic Factor Vulnerabilities**
* **Primary Economic Driver:** **{top_theme['theme'] if top_theme else 'Broad Market'}** accounts for **{top_theme['weight']*100:.1f}%** of your portfolio. Even assets in different sectors share this common sensitivity.
* **Concentration Tier:** Rated as **{metrics['concentration_tier']}** (HHI: {metrics['hhi']}), with your Top 3 holdings representing **{metrics['top_3_concentration']}%** of all assets.

**3. Quantified Downside Risk**
* **1-Month 95% Value at Risk (VaR):** **${metrics['var_95_1m_dollar']:,.2f}** ({metrics['var_95_1m_pct']}%).
* **Historical Stress Test:** In a scenario like the **{worst_stress['name'] if worst_stress else 'Market Selloff'}**, this portfolio has an estimated simulated drawdown of **{worst_stress['drawdown_pct'] if worst_stress else 0.0}%** (${worst_stress['dollar_drawdown'] if worst_stress else 0.0:,.2f}).

**4. Questions for Your Consideration:**
* If your primary theme ({top_theme['theme'] if top_theme else 'tech'}) faced a 25% earnings multiple contraction, could you comfortably ride out the drawdown without panicking?
* Does having {metrics['top_3_concentration']}% in just three underlying assets fit your {portfolio.user_profile.get('risk_tolerance')} risk tolerance?
"""

    return {
        "source": "deterministic-risk-engine" if not error_note else f"offline-analyst (API fallback: {error_note})",
        "response": response_text.strip(),
    }
