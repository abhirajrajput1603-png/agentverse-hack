"""AI Risk Intelligence Analyst for Portfolio X-Ray.

Connects to Google Gemini API using standard library urllib.request (zero extra dependencies).
Includes deterministic, context-aware rule-based fallback when offline or without an API key.
"""

from __future__ import annotations
import os
import re
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

    # Check local .env file in parent directories
    for candidate in [
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]:
        if candidate.is_file():
            try:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val
            except Exception:
                pass
    return None


SYSTEM_INSTRUCTION = """You are Portfolio X-Ray, an institutional-grade autonomous Portfolio Risk Intelligence Analyst.
Your mission is to help everyday investors understand the true economic risks, hidden concentrations, and asset interactions in their portfolios.

CORE PRINCIPLES:
1. Strict Scope Enforcement: You strictly answer questions about investment portfolios, asset allocations, risk metrics, financial scenarios, and economic exposures. If the user's question is unrelated to finance, investing, portfolios, economics, or risk management (for example: cooking recipes, general knowledge, programming, non-financial trivia, personal chit-chat, etc.), you MUST decline politely: "I can only assist with questions related to your portfolio, financial risk analysis, asset allocations, and market exposure. I am unable to answer questions outside of this scope."
2. Directly Answer the User's Question: Always address what the user actually asked, applying portfolio risk analytics to their specific inquiry.
3. Educate, Never Speculate: Never forecast price targets or recommend buying/selling specific assets. Frame everything around risk exposure, concentration, and historical scenario resilience.
4. Unpack Look-Through: Always highlight overlapping holdings (e.g., direct stock + ETFs holding the same stock).
5. Explain Jargon Simply: Explain terms like Value at Risk (VaR), Herfindahl Index (HHI), and Beta in clear, everyday language.
6. Stress Test Honestly: Show what happens in adverse market scenarios (rate shocks, recessions, tech selloffs).
7. Ground in User Profile: Compare risks against the user's stated risk tolerance, horizon, and loss capacity.
"""

FINANCIAL_KEYWORDS = {
    # Portfolio & asset concepts
    "portfolio", "portfolios", "holding", "holdings", "stock", "stocks", "equity", "equities",
    "etf", "etfs", "fund", "funds", "share", "shares", "ticker", "tickers",
    "asset", "assets", "position", "positions", "allocation", "allocations", "allocate", "allocating",
    "rebalance", "rebalancing", "diversify", "diversified", "diversification",
    "overlap", "overlaps", "overlapping", "lookthrough", "look-through", "underlying", "constituent", "constituents",
    # Financial metrics & risk
    "risk", "risks", "risky", "volatility", "volatile", "var", "cvar", "drawdown",
    "drawdowns", "beta", "hhi", "concentration", "concentrated", "exposure", "exposures",
    "safe", "safety", "loss", "losses", "lose", "losing", "return", "returns",
    "profit", "profits", "profitable", "yield", "dividend", "dividends", "capital",
    "money", "cash", "wealth", "wealthy", "dollar", "dollars", "invest", "invests",
    "investing", "investment", "investments", "investor", "investors",
    "vulnerability", "vulnerabilities", "weakness", "weaknesses", "downside", "upside",
    "balance", "worth", "broker", "brokerage",
    # Actions & trading
    "buy", "buys", "buying", "sell", "sells", "selling", "trade", "trades", "trading",
    "hedge", "hedging", "trim", "trimming", "short", "long", "hold", "holds",
    "what-if", "what if", "scenario", "stress", "simulate", "simulation",
    # Macro & markets
    "market", "markets", "crash", "crashes", "crashing", "recession", "recessions",
    "inflation", "deflation", "stagflation", "rate", "rates", "fed", "federal reserve",
    "interest", "yield curve", "macro", "economy", "economic", "gdp", "bear", "bull",
    "shock", "shocks", "selloff", "correction", "bubble", "liquidity",
    # Sectors & themes
    "tech", "technology", "semiconductor", "semiconductors", "chip", "chips",
    "energy", "utilities", "financials", "healthcare", "bonds", "treasury", "treasuries",
    "gold", "commodity", "commodities", "crypto", "bitcoin", "fixed income",
    # Tool / App scope
    "app", "dashboard", "x-ray", "xray", "analyst", "audit", "engine", "benchmark",
    "spy", "qqq", "voo", "vti", "smh", "xlk", "xlf", "xle", "tlt", "bnd", "gld",
}

KNOWN_COMPANIES = {
    "nvidia", "nvda", "apple", "aapl", "microsoft", "msft", "amazon", "amzn",
    "google", "googl", "goog", "alphabet", "meta", "facebook", "tesla", "tsla",
    "broadcom", "avgo", "amd", "berkshire", "brk", "jpmorgan", "jpm", "lilly", "lly",
    "constellation", "ceg"
}


def is_finance_or_portfolio_query(query: str) -> bool:
    """Checks whether the user's query is relevant to financial risk and portfolio analysis."""
    q = query.lower().strip()
    words = set(re.findall(r"\b[a-z0-9\$\%]+\b", q))

    if any(k in words for k in FINANCIAL_KEYWORDS):
        return True
    if any(c in words for c in KNOWN_COMPANIES):
        return True
    if any(p in q for p in ["what if", "look through", "look-through", "interest rate", "stress test", "make money"]):
        return True
    if "$" in q or "%" in q:
        return True
    return False


SCOPE_REFUSAL_MESSAGE = (
    "I can only assist with questions related to your portfolio, financial risk analysis, "
    "asset allocations, and market exposure. I am unable to answer questions outside of this scope."
)


def format_portfolio_context(portfolio: Portfolio) -> str:
    """Prepares structured, inspectable portfolio risk evidence for the AI agent."""
    exposures = compute_lookthrough_exposure(portfolio)
    thematic = compute_thematic_exposure(exposures)
    metrics = compute_risk_metrics(portfolio, exposures)
    stress = run_stress_tests(portfolio, exposures)

    profile = portfolio.user_profile or {}
    lines = [
        f"TOTAL PORTFOLIO VALUE: ${metrics['portfolio_value']:,.2f}",
        f"INVESTOR PROFILE: Risk Tolerance: {profile.get('risk_tolerance', 'Moderate')}, Horizon: {profile.get('horizon_years', 7)} yrs, Loss Capacity: {profile.get('loss_capacity', 'Moderate (15-20% max drawdown)')}",
        "",
        "DIRECT HOLDINGS:",
    ]
    total_val = portfolio.total_value or 1.0
    for h in portfolio.holdings:
        lines.append(f"- {h.ticker}: {h.shares:.2f} shares @ ${h.price:.2f} = ${h.market_value:,.2f} ({h.market_value/total_val*100:.1f}%)")

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
        lines.append(f"- {st['name']}: {st['drawdown_pct']}% drawdown (-${abs(st['dollar_drawdown']):,.2f})")

    return "\n".join(lines)


def ask_risk_analyst(
    portfolio: Portfolio,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Generates an evidence-backed risk analysis response using Gemini or offline fallback."""
    # Scope check: refuse queries unrelated to portfolio risk analysis
    if not is_finance_or_portfolio_query(user_query):
        return {
            "source": "scope-guardrail",
            "response": SCOPE_REFUSAL_MESSAGE,
        }

    context_str = format_portfolio_context(portfolio)
    api_key = get_api_key()

    if api_key and not api_key.startswith("AQ.Ab8RN6JqCrilO84"):
        try:
            return call_gemini_api(api_key, context_str, user_query, conversation_history)
        except Exception as e:
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

Please provide a direct, insightful, and professional response that directly answers the user's question using their actual portfolio numbers and risk exposures.
Format in clean Markdown with clear section headings or bullet points.
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
    """Generates deterministic, institutional-quality risk analysis tailored to the user's specific query."""
    exposures = compute_lookthrough_exposure(portfolio)
    metrics = compute_risk_metrics(portfolio, exposures)
    thematic = compute_thematic_exposure(exposures)
    stress = run_stress_tests(portfolio, exposures)

    top_exp = exposures[0] if exposures else None
    top_theme = thematic[0] if thematic else None
    worst_stress = min(stress, key=lambda s: s["drawdown_pct"]) if stress else None
    profile = portfolio.user_profile or {}
    risk_tol = profile.get("risk_tolerance", "Moderate")

    q_lower = query.lower().strip()
    words = set(re.findall(r"\b[a-z0-9]+\b", q_lower))

    # Extract any ticker mentioned in the query
    known_tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TLT", "GLD", "SPY", "QQQ", "VOO", "SMH", "VTI", "BND", "CASH"]
    mentioned_ticker = None
    for t in known_tickers:
        if t.lower() in words:
            mentioned_ticker = t
            break

    # 1. "Make money" / Growth / Returns / Profit / Upside / Wealth
    if any(p in q_lower for p in [
        "make money", "making money", "want to make money", "grow money", "how do i make money",
        "get rich", "maximize return", "upside", "compound",
    ]) or ({"money", "profit"} & words) or ({"return", "returns"} & words) or ({"growth", "grow"} & words):
        top_driver_name = f"{top_theme['theme']} ({top_theme['weight']*100:.1f}%)" if top_theme else "Equities"
        response_text = f"""### Pursuing Growth & Returns: What Drives Your Money

You want to make money and grow your capital. In investing, **expected return is directly coupled to risk**—there is no excess return without exposing your capital to market volatility and potential drawdowns.

Here is an audit of how your portfolio is currently set up to make money, and the specific trade-offs involved:

#### 1. What is Driving Your Potential Upside?
* **Primary Return Engine:** Your portfolio's growth is heavily tied to **{top_driver_name}**, led by **{top_exp.ticker if top_exp else 'your top stock'} ({top_exp.name if top_exp else ''})** at **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) effective exposure.
* **Leveraging Big Tech Momentum:** Holding both direct equities and growth ETFs (like QQQ/SMH) concentrates your upside. When the sector rallies, your gains will outpace the broader market.

#### 2. The Cost of Chasing High Returns: Downside Exposure
* **Annualized Volatility:** **{metrics['annual_volatility']}%**. This is significantly higher than a balanced baseline (~12-14%), meaning your account value will experience sharp swings.
* **1-Month 95% Value at Risk (VaR):** **${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)**. In 1 out of every 20 months, you should statistically anticipate a loss of at least this magnitude.
* **Shock Scenario Vulnerability:** In an adverse market event like **{worst_stress['name'] if worst_stress else 'a Market Selloff'}**, your estimated portfolio drawdown is **{worst_stress['drawdown_pct'] if worst_stress else 0.0}% (-${abs(worst_stress['dollar_drawdown']) if worst_stress else 0.0:,.2f})**.

#### 3. Actionable Questions for Long-Term Wealth Building
* **Can you endure the drawdown?** High-return strategies only work if you have a multi-year horizon and do not panic-sell during a 20-30% correction.
* **Is single-stock concentration too high?** Your top 3 holdings account for **{metrics['top_3_concentration']}%** of all assets. If one company stumbles, it can wipe out your gains.
* **Test adjustments in the What-If tool below:** You can simulate trimming extreme exposures to see how much downside volatility you can cut while maintaining healthy upside.

*(Note: Portfolio X-Ray analyzes risk and exposure. We do not provide speculative price predictions or specific buy/sell advice.)*
"""

    # 2. Safety / Capital Preservation / Is My Portfolio Safe?
    elif any(p in q_lower for p in [
        "is it safe", "is my portfolio safe", "how safe", "am i safe", "protect my money", "capital preservation"
    ]) or ({"safe", "safety", "secure"} & words) or ({"loss", "lose", "losing"} & words) or ({"danger", "worried", "risky"} & words):
        safety_verdict = "High Risk / Aggressive" if metrics['hhi'] > 2500 or metrics['annual_volatility'] > 22 else ("Moderately Balanced" if metrics['hhi'] > 1500 else "Well Diversified / Defensive")
        response_text = f"""### Portfolio Safety & Downside Assessment

Here is an objective assessment of your portfolio's safety:

#### 1. Overall Risk Verdict: **{safety_verdict}**
* **Concentration Tier:** **{metrics['concentration_tier']}** (HHI: {metrics['hhi']}).
* **Dominant Exposure:** Your single largest economic holding is **{top_exp.ticker if top_exp else 'N/A'}** representing **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) of your wealth.
* **Rule of Thumb:** A portfolio is generally not considered "safe" from company-specific shocks if any single stock exceeds 15-20% of total capital.

#### 2. Downside Losses in Numbers
* **Expected Monthly Drop (95% VaR):** **${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)**.
* **Tail Risk (Expected Shortfall):** If market losses exceed normal VaR, the average expected loss is **${metrics['expected_shortfall_95_dollar']:,.2f} ({metrics['expected_shortfall_95_pct']}%)**.
* **Stress Test Drawdown:** Under a scenario like **{worst_stress['name'] if worst_stress else 'Market Correction'}**, this portfolio drops by an estimated **{worst_stress['drawdown_pct'] if worst_stress else 0.0}% (-${abs(worst_stress['dollar_drawdown']) if worst_stress else 0.0:,.2f})**.

#### 3. How to Improve Safety
* Check if your stated risk tolerance (**{risk_tol}**) matches these numbers. If you cannot tolerate a 20%+ drawdown, consider trimming your top overlapping positions and adding broad market anchors or cash.
"""

    # 3. Buy / Sell Recommendations / Stock Picking Advice
    elif any(p in q_lower for p in [
        "should i buy", "should i sell", "what should i buy", "what should i sell",
        "what to buy", "what to sell", "stock recommendation", "stock tip", "which stock",
    ]) or ({"recommend", "recommendation", "tip", "advice"} & words):
        response_text = f"""### Risk Intelligence Policy: No Speculative Trade Calls

As an institutional-grade risk analyst, **Portfolio X-Ray does not predict stock prices or tell you to buy or sell specific assets.** 

Here is what we **can** do to guide your decision-making:

#### 1. What Your Holdings Need Right Now
* **Current Concentration:** Your Top 3 holdings represent **{metrics['top_3_concentration']}%** of your capital ({metrics['concentration_tier']}).
* If you are thinking of buying **more of {top_exp.ticker if top_exp else 'your top assets'}**, be aware that your concentration will cross even higher risk thresholds.
* If your goal is to reduce vulnerability, adding non-correlated assets (such as broad-market index funds, defensive sectors, or fixed income) reduces your overall portfolio volatility.

#### 2. Test Before You Trade
* Instead of guessing, use the **What-If Simulation** section below.
* Enter any hypothetical purchase or rebalance (e.g., allocating 20% to SPY or trimming a stock) to see the exact before-and-after impact on your volatility, HHI, and crisis drawdown.
"""

    # 4. Look-Through / Overlap / Hidden Concentration
    elif any(p in q_lower for p in [
        "look through", "look-through", "overlap", "hidden", "etf overlap", "diversified", "diversification"
    ]) or ({"overlap", "overlapping", "etf", "etfs"} & words):
        overlap_examples = []
        for exp in exposures:
            if exp.indirect_weight > 0.01 and exp.direct_weight > 0.01:
                sources_str = ", ".join(exp.contributing_sources.keys())
                overlap_examples.append(f"* **{exp.ticker} ({exp.name}):** You own {exp.direct_weight*100:.1f}% directly, but ETFs ({sources_str}) add an extra {exp.indirect_weight*100:.1f}%, making your real exposure **{exp.total_weight*100:.1f}%** (${exp.total_dollar_value:,.2f}).")
        if not overlap_examples:
            overlap_examples.append(f"* **{top_exp.ticker}:** Effective exposure is {top_exp.total_weight*100:.1f}% (${top_exp.total_dollar_value:,.2f}).")

        response_text = f"""### Understanding Look-Through & Hidden Overlap

Most retail investors look at their brokerage account and count the number of tickers, assuming 4-5 different ETFs provide broad diversification. **Look-Through Decomposition reveals what you actually own underneath.**

#### 1. Your Hidden Overlaps in This Portfolio:
{chr(10).join(overlap_examples[:4])}

#### 2. Why This Matters
* When you own both individual tech stocks and tech/broad ETFs (like QQQ, SMH, and VOO), **you are unwittingly multiplying your single-stock bet**.
* If that top company stumbles on earnings, the loss hits your direct shares and your ETF baskets simultaneously.
* Real diversification requires uncorrelated economic drivers, not just multiple fund tickets.
"""

    # 5. Stress Testing / Macro / Crashes / Recession / Inflation
    elif any(p in q_lower for p in [
        "crash", "recession", "inflation", "interest rate", "fed", "bear market", "rate hike", "drawdown", "stress test"
    ]) or ({"shock", "shocks", "crisis"} & words):
        stress_lines = []
        for st in stress:
            stress_lines.append(f"* **{st['name']}:** Estimated drawdown of **{st['drawdown_pct']}%** (-${abs(st['dollar_drawdown']):,.2f}). {st.get('description', '')}")

        response_text = f"""### Historical Stress Test Analysis

Here is how your current holdings would withstand historical market crises:

#### 1. Simulated Scenario Drawdowns:
{chr(10).join(stress_lines)}

#### 2. Biggest Vulnerability in a Crisis:
* Your primary economic exposure is **{top_theme['theme'] if top_theme else 'Growth'}** ({top_theme['weight']*100:.1f}%).
* During liquidity crunches or interest rate spikes (like 2022), high-duration tech valuations compress rapidly.
* With a 1-Month 95% VaR of **${metrics['var_95_1m_dollar']:,.2f}**, keep cash reserves available if you cannot afford short-term liquidation.
"""

    # 6. Specific Ticker Inquiry (e.g., NVDA, AAPL, MSFT, SPY, etc.)
    elif mentioned_ticker:
        target_exp = next((exp for exp in exposures if exp.ticker == mentioned_ticker), None)
        direct_h = next((h for h in portfolio.holdings if h.ticker == mentioned_ticker), None)

        if target_exp:
            indirect_sources = ", ".join(f"{k} (+{v*100:.1f}%)" for k, v in target_exp.contributing_sources.items()) if target_exp.contributing_sources else "None (held directly)"
            response_text = f"""### Holding Deep-Dive: **{target_exp.ticker} ({target_exp.name})**

Here is your exact look-through breakdown for **{target_exp.ticker}**:

#### 1. Effective Exposure Breakdown
* **Total Effective Weight:** **{target_exp.total_weight*100:.1f}%** (${target_exp.total_dollar_value:,.2f})
* **Direct Holding Weight:** {target_exp.direct_weight*100:.1f}% {f'({direct_h.shares:.1f} shares @ ${direct_h.price:.2f})' if direct_h else '(Not held directly)'}
* **Indirect Exposure via ETFs:** {target_exp.indirect_weight*100:.1f}%
* **Contributing ETF Sources:** {indirect_sources}

#### 2. Risk Role in Your Portfolio
* **Sector:** {target_exp.sector}
* **Themes:** {', '.join(target_exp.themes) if target_exp.themes else 'General Market'}
* **Annual Volatility:** {target_exp.annual_volatility*100:.1f}%
* With a total weight of {target_exp.total_weight*100:.1f}%, a 10% move in {target_exp.ticker} changes your entire net worth by **{target_exp.total_weight*10:.2f}% (${target_exp.total_dollar_value*0.10:,.2f})**.
"""
        else:
            response_text = f"""### Analysis for **{mentioned_ticker}**

* **Status in Portfolio:** **{mentioned_ticker}** is not currently in your direct holdings or top ETF look-through exposures.
* **Want to see what adding {mentioned_ticker} would do?** You can test adding a position to {mentioned_ticker} in the **What-If Simulator** below to observe how your concentration (HHI) and downside VaR change.
"""

    # 7. What-If Simulation Query
    elif any(p in q_lower for p in ["what if", "50%", "put", "allocate"]):
        target_ticker = mentioned_ticker or "NVDA"
        sim = simulate_what_if(portfolio, {target_ticker: 50.0, "SPY": 50.0})
        prop_m = sim["proposed_metrics"]
        cur_m = sim["current_metrics"]
        delta_vol = sim["metrics_delta"]["volatility_change"]

        response_text = f"""### What-If Risk Simulation: 50% Allocation to {target_ticker}

#### 1. Exposure & Look-Through Impact
* If you allocate 50% directly into **{target_ticker}**, your effective concentration in that single asset spikes from **{cur_m['top_1_concentration']}%** to **{prop_m['top_1_concentration']}%**.
* Your portfolio's concentration index (HHI) changes by **{sim['metrics_delta']['hhi_change']:+,.1f} points**, moving into the **{prop_m['concentration_tier']}** zone.

#### 2. Downside & Volatility Trade-Off
* **Annualized Volatility:** Moves from **{cur_m['annual_volatility']}%** to **{prop_m['annual_volatility']}%** ({delta_vol:+,.2f}% shift).
* **1-Month 95% Value at Risk (VaR):** Increases to **${prop_m['var_95_1m_dollar']:,.2f}** ({prop_m['var_95_1m_pct']}% of total capital).

#### 3. Stress Scenario Resilience
* Under **{sim['stress_comparison'][0]['scenario'] if sim.get('stress_comparison') else 'Historical Stress Shock'}**, the proposed portfolio faces an estimated **{sim['stress_comparison'][0]['proposed_drawdown_pct'] if sim.get('stress_comparison') else 0.0}%** drawdown compared to **{sim['stress_comparison'][0]['current_drawdown_pct'] if sim.get('stress_comparison') else 0.0}%** currently.

#### 4. Investor Profile Check
{('⚠️ **Risk Threshold Warning:** ' + ' '.join(sim['profile_warnings'])) if sim['profile_warnings'] else '✅ This adjustment remains within your stated risk tolerance bounds.'}
"""

    # 8. General / Fallback Question
    else:
        response_text = f"""### Analysis of Your Inquiry: *"{query}"*

Here is the risk perspective on your question based on your current holdings:

#### 1. Current Portfolio Context
* **Total Portfolio Capital:** ${metrics['portfolio_value']:,.2f}
* **Concentration Tier:** **{metrics['concentration_tier']}** (HHI: {metrics['hhi']}).
* **Top Look-Through Holding:** **{top_exp.ticker if top_exp else 'N/A'}** represents **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) of your wealth.
* **Annual Volatility:** **{metrics['annual_volatility']}%** (95% 1-Month VaR: **${metrics['var_95_1m_dollar']:,.2f}**).

#### 2. Key Considerations for You
* **Dominant Risk:** Your capital is {top_theme['weight']*100:.1f}% exposed to the **{top_theme['theme'] if top_theme else 'Tech'}** macro factor. Any market event hitting that theme will heavily influence your overall balance.
* **Alignment with Risk Profile:** Your current investor profile is set to **{risk_tol}**. Check whether potential historical drawdowns of up to **{worst_stress['drawdown_pct'] if worst_stress else 0.0}%** align with your time horizon.
* **Interactive Tools:** You can explore specific stress replays or test hypothetical trade adjustments in the dashboard widgets below.
"""

    return {
        "source": "deterministic-risk-engine" if not error_note else f"offline-analyst (API fallback: {error_note})",
        "response": response_text.strip(),
    }
