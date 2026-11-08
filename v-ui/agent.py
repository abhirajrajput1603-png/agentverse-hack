"""AI Risk Intelligence Analyst for Portfolio X-Ray.

Connects to Google Gemini API using standard library urllib.request (zero extra dependencies)
and integrates with xray's deterministic AgentToolRunner (MAX_AGENT_TOOL_CALLS = 8),
EvidenceRegistry ([EV-*]), OutputSafetyFilter, and append-only AuditLogger.
Includes deterministic, context-aware rule-based fallback when offline or without an API key.
"""

from __future__ import annotations
import os
import re
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure both v-ui and project root are in sys.path
V_UI_DIR = Path(__file__).resolve().parent
ROOT_DIR = V_UI_DIR.parent
if str(V_UI_DIR) not in sys.path:
    sys.path.insert(0, str(V_UI_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from engine import (
        Portfolio,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
    )
except ImportError:
    from portfolio_xray.engine import (
        Portfolio,
        compute_lookthrough_exposure,
        compute_thematic_exposure,
        compute_risk_metrics,
        run_stress_tests,
        simulate_what_if,
    )

from xray.agent.investigator import AgentToolRunner, MAX_AGENT_TOOL_CALLS
from xray.bridge import convert_vui_portfolio_to_xray_holdings, map_vui_profile_to_xray, get_shared_loader
from xray.safety.audit import AuditLogger
from xray.safety.output_filter import OutputSafetyFilter


def get_api_key() -> Optional[str]:
    """Retrieves GEMINI_API_KEY from environment or .env file."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()

    for candidate in [
        V_UI_DIR.parent / ".env",
        V_UI_DIR.parent.parent / ".env",
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
4. Zero Mental Math & Evidence Citations: Use ONLY the deterministic numbers provided in the audit context and cite their Evidence IDs in brackets (e.g. [EV-CONC-TOP1-WEIGHT], [EV-RISK-ANN-VOL], [EV-TAIL-ES-95]).
5. Unpack Look-Through: Always highlight overlapping holdings (e.g., direct stock + ETFs holding the same stock).
6. Explain Jargon Simply: Explain terms like Value at Risk (VaR), Herfindahl Index (HHI), Euler Risk Contribution, and Beta in clear, everyday language.
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
    "correlation", "correlations", "euler", "marginal", "tail", "shortfall", "bootstrap",
    "safe", "safety", "loss", "losses", "lose", "losing", "return", "returns",
    "profit", "profits", "profitable", "yield", "dividend", "dividends", "capital",
    "money", "cash", "wealth", "wealthy", "dollar", "dollars", "invest", "invests",
    "investing", "investment", "investments", "investor", "investors",
    "vulnerability", "vulnerabilities", "weakness", "weaknesses", "downside", "upside",
    "balance", "worth", "broker", "brokerage", "price", "target", "forecast", "predict",
    # Actions & trading
    "buy", "buys", "buying", "sell", "sells", "selling", "trade", "trades", "trading",
    "hedge", "hedging", "trim", "trimming", "short", "long", "hold", "holds",
    "what-if", "what if", "scenario", "stress", "simulate", "simulation", "audit",
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
    "app", "dashboard", "x-ray", "xray", "analyst", "engine", "benchmark", "breaker",
    "spy", "qqq", "voo", "vti", "smh", "xlk", "xlf", "xle", "tlt", "bnd", "gld", "arkk", "vst",
}

KNOWN_COMPANIES = {
    "nvidia", "nvda", "apple", "aapl", "microsoft", "msft", "amazon", "amzn",
    "google", "googl", "goog", "alphabet", "meta", "facebook", "tesla", "tsla",
    "broadcom", "avgo", "amd", "berkshire", "brk", "jpmorgan", "jpm", "lilly", "lly",
    "constellation", "ceg", "vistra", "vst", "arkk", "spacex", "openai", "palantir", "pltr",
}

ADVERSARIAL_INJECTION_PATTERNS = [
    (r"ignore\s+(?:all\s+)?(?:previous|prior|system|safety)\s+(?:instructions|rules|limits|guardrails)", "Prompt injection override attempt"),
    (r"disregard\s+(?:all\s+)?(?:previous|prior|system|safety)\s+(?:instructions|rules|limits|guardrails)", "System rule bypass attempt"),
    (r"bypass\s+(?:all\s+)?(?:safety|system|risk|filter|guardrail)", "Safety filter bypass attempt"),
    (r"you\s+are\s+now\s+(?:a|an)\s+(?:hedge\s+fund|day\s+trader|unrestricted|dan)", "Persona hijack attempt"),
    (r"(?:output|print|generate|execute|place)\s+(?:a\s+)?(?:(?:market|limit)\s+)?(?:buy|sell)\s+order", "Forced trade execution injection"),
    (r"(?:assume|pretend|state\s+that|say\s+that|report\s+that)\s+my\s+(?:portfolio\s+)?(?:volatility|risk|var|concentration|drawdown)\s+is\s+\d+(?:\.\d+)?\s*%", "Fabricated math injection attempt"),
    (r"override\s+(?:the\s+)?circuit\s+breaker", "Circuit breaker override attempt"),
]

PREDICTION_REQUEST_PATTERNS = [
    r"what\s+will\s+[a-z0-9.\s]+(?:share\s+price|stock\s+price|price)\s+be\b",
    r"\bpredict\s+(?:the\s+)?(?:future\s+)?(?:price|stock\s+price|return)\b",
    r"\b(?:give\s+me\s+a\s+)?price\s+target\s+for\b",
    r"\bwhere\s+will\s+[a-z0-9.]+\s+trade\s+(?:next\s+week|next\s+month|next\s+year|tomorrow|on\s+friday)\b",
    r"\bwill\s+[a-z0-9.]+\s+(?:double|triple|reach\s+\$\d+|hit\s+\$\d+)\b",
]


def detect_adversarial_injection(query: str) -> Optional[str]:
    """Detects prompt-injection, jailbreak, or fabricated-math override attempts."""
    q = query.strip()
    for pat, reason in ADVERSARIAL_INJECTION_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return reason
    return None


def is_price_prediction_request(query: str) -> bool:
    """Detects direct requests for speculative stock price targets or future price forecasts."""
    q = query.strip()
    for pat in PREDICTION_REQUEST_PATTERNS:
        if re.search(pat, q, flags=re.IGNORECASE):
            return True
    return False


def is_finance_or_portfolio_query(query: str) -> bool:
    """Checks whether the user's query is relevant to financial risk and portfolio analysis."""
    q = query.lower().strip()
    # Explicit non-finance programming / cooking / trivia exclusions
    non_finance_patterns = [
        r"\b(?:pasta|recipe|cooking|bake\s+a\s+cake|chicken\s+soup)\b",
        r"\b(?:sorting\s+algorithm|quicksort|mergesort|binary\s+tree|leetcode|fibonacci)\b",
        r"\b(?:write\s+a\s+poem|write\s+a\s+haiku|tell\s+me\s+a\s+joke)\b",
    ]
    for pat in non_finance_patterns:
        if re.search(pat, q, flags=re.IGNORECASE):
            return False

    words = set(re.findall(r"\b[a-z0-9\$\%]+\b", q))

    if any(k in words for k in FINANCIAL_KEYWORDS):
        return True
    if any(c in words for c in KNOWN_COMPANIES):
        return True
    if any(p in q for p in ["what if", "look through", "look-through", "interest rate", "stress test", "make money"]):
        return True
    if "$" in q or "£" in q or "%" in q:
        return True
    return False


SCOPE_REFUSAL_MESSAGE = (
    "I can only assist with questions related to your portfolio, financial risk analysis, "
    "asset allocations, and market exposure. I am unable to answer questions outside of this scope."
)


def run_autonomous_tools_for_query(
    portfolio: Portfolio,
    query: str,
    audit_logger: AuditLogger,
) -> AgentToolRunner:
    """
    Runs the deterministic AgentToolRunner (capped at MAX_AGENT_TOOL_CALLS = 8)
    to gather verifiable Evidence objects ([EV-*]) tailored to the user's query.
    """
    xray_holdings = convert_vui_portfolio_to_xray_holdings(portfolio)
    xray_profile = map_vui_profile_to_xray(portfolio.user_profile or {})
    runner = AgentToolRunner(
        holdings=xray_holdings,
        persona_name=xray_profile.name,
        user_max_single_stock_limit=xray_profile.max_single_stock_limit,
        data_loader=get_shared_loader(),
        audit_logger=audit_logger,
        max_calls=MAX_AGENT_TOOL_CALLS,
    )

    q_lower = query.lower()
    # Always run look-through & concentration + circuit breaker first
    runner.call_tool("lookthrough_and_concentration", reason="Decompose direct and ETF holdings into true issuer exposures")
    runner.call_tool("circuit_breaker", reason="Verify single-stock concentration against investor risk limit")
    runner.call_tool("risk_and_tail", reason="Compute 3Y annualized volatility, 95%/99% historical VaR, and Expected Shortfall")

    if any(k in q_lower for k in ["corr", "euler", "contribut", "diversif", "risk", "vulnerability", "audit"]):
        runner.call_tool("correlation_and_risk_contribution", reason="Compute 3Y pairwise correlation and Euler risk contribution")
    if any(k in q_lower for k in ["stress", "crash", "rate", "fed", "2020", "2022", "shock", "lose", "loss", "safe", "audit"]):
        runner.call_tool("historical_stress_replays", reason="Replay historical crisis drawdowns vs SPY benchmark")
    if any(k in q_lower for k in ["bench", "spy", "60/40", "beta", "compare", "tilt", "audit"]):
        runner.call_tool("benchmark_comparison", reason="Compare portfolio against SPY and 60/40 SPY/BND benchmarks")

    return runner


def format_portfolio_context(portfolio: Portfolio, runner: Optional[AgentToolRunner] = None) -> str:
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
        lines.append(f"- {exp.ticker} ({exp.name}): Total Effective: {exp.total_weight*100:.1f}% (${exp.total_dollar_value:,.2f}){overlap_detail} [EV-LT-{exp.ticker}]")

    lines.append("\nTHEMATIC & ECONOMIC EXPOSURES:")
    for th in thematic[:4]:
        lines.append(f"- {th['theme']}: {th['weight']*100:.1f}% portfolio weight")

    lines.append("\nQUANTITATIVE RISK METRICS:")
    lines.append(f"- Concentration Tier: {metrics['concentration_tier']} (HHI: {metrics['hhi']}) [EV-CONC-LT-HHI]")
    lines.append(f"- Top 1 Holding: {metrics['top_1_concentration']}% [EV-CONC-TOP1-WEIGHT], Top 3 Holdings: {metrics['top_3_concentration']}%")
    lines.append(f"- Annualized Volatility: {metrics['annual_volatility']}% [EV-RISK-ANN-VOL]")
    lines.append(f"- 1-Month 95% Value at Risk (VaR): {metrics['var_95_1m_pct']}% (${metrics['var_95_1m_dollar']:,.2f}) [EV-RISK-VAR-95]")
    lines.append(f"- Expected Shortfall (CVaR): {metrics['expected_shortfall_95_pct']}% (${metrics['expected_shortfall_95_dollar']:,.2f}) [EV-TAIL-ES-95]")

    lines.append("\nHISTORICAL & MACRO STRESS TEST ESTIMATES:")
    for st in stress:
        lines.append(f"- {st['name']}: {st['drawdown_pct']}% drawdown (-${abs(st['dollar_drawdown']):,.2f})")

    if runner is not None:
        lines.append("\nVERIFIED EVIDENCE REGISTRY IDS:")
        for ev in runner.registry.all()[:15]:
            lines.append(f"- [{ev.id}] {ev.metric}: {ev.value} (formula: {ev.formula})")

    return "\n".join(lines)


def ask_risk_analyst(
    portfolio: Portfolio,
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> Dict[str, Any]:
    """
    Generates an evidence-backed risk analysis response using Gemini or deterministic engine,
    enforcing scope guardrails, adversarial prompt-injection defenses, OutputSafetyFilter,
    autonomous tool-calling (AgentToolRunner), and append-only audit logging.
    """
    audit = audit_logger or AuditLogger()
    safety_filter = OutputSafetyFilter()

    # 1. Adversarial Prompt Injection / Override / Fabricated Math Defense
    inj_reason = detect_adversarial_injection(user_query)
    if inj_reason:
        runner = run_autonomous_tools_for_query(portfolio, "risk audit", audit)
        exposures = compute_lookthrough_exposure(portfolio)
        metrics = compute_risk_metrics(portfolio, exposures)
        top_exp = exposures[0] if exposures else None
        top_t = top_exp.ticker if top_exp else "NVDA"
        top_w = round(top_exp.total_weight * 100.0, 1) if top_exp else 0.0

        refusal_md = (
            f"### Safety Guardrail Active: Request Blocked\n\n"
            f"Portfolio X-Ray blocked this request (**{inj_reason}**).\n\n"
            f"* **Non-Negotiable Safety Policy:** I cannot ignore system safety rules, fabricate risk metrics, or execute/recommend buy or sell orders.\n"
            f"* **Verified Deterministic Portfolio State:** Your actual look-through top exposure is **{top_t}** at **{top_w}%** `[EV-CONC-TOP1-WEIGHT]`, "
            f"and your verified annualized volatility is **{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]`.\n"
            f"* Use the **Try a Change (What-If Simulator)** tab to test hypothetical portfolio adjustments safely with full human sign-off."
        )
        audit.log_event(
            event_type="ADVERSARIAL_PROMPT_BLOCKED",
            details={"query": user_query, "reason": inj_reason},
        )
        cited_evs = [ev.to_dict() for ev in runner.registry.all()[:8]]
        return {
            "source": "safety-filter",
            "response": refusal_md,
            "answer": refusal_md,
            "cited_evidences": cited_evs,
            "tool_calls": [
                {
                    "step": r.step,
                    "tool_name": r.tool_name,
                    "reason": r.reason,
                    "evidence_ids_produced": r.evidence_ids_produced,
                }
                for r in runner.tool_trace
            ],
            "safety_check": {"passed": False, "blocked_reason": inj_reason},
        }

    # 2. Scope Check: Refuse queries unrelated to portfolio risk analysis
    if not is_finance_or_portfolio_query(user_query):
        audit.log_event(
            event_type="SCOPE_GUARDRAIL_REFUSAL",
            details={"query": user_query},
        )
        return {
            "source": "scope-guardrail",
            "response": SCOPE_REFUSAL_MESSAGE,
            "answer": SCOPE_REFUSAL_MESSAGE,
            "cited_evidences": [],
            "tool_calls": [],
            "safety_check": {"passed": True, "blocked_reason": "out_of_scope_query"},
        }

    # 3. Execute Autonomous Tool-Calling Loop (capped at 8 calls) to gather Evidence [EV-*]
    runner = run_autonomous_tools_for_query(portfolio, user_query, audit)
    tool_calls_out = [
        {
            "step": r.step,
            "tool_name": r.tool_name,
            "reason": r.reason,
            "evidence_ids_produced": r.evidence_ids_produced,
            "summary": r.summary,
        }
        for r in runner.tool_trace
    ]

    # 4. Direct Speculative Price Target / Future Stock Price Prediction Defense
    if is_price_prediction_request(user_query):
        exposures = compute_lookthrough_exposure(portfolio)
        metrics = compute_risk_metrics(portfolio, exposures)
        pred_refusal = (
            f"### Non-Predictive Policy: No Stock Price Forecasts or Price Targets\n\n"
            f"**Portfolio X-Ray does not predict future stock prices, price targets, or market timing.**\n\n"
            f"Instead, we quantify your portfolio's historical risk and downside exposure using verified market data:\n"
            f"* **Annualized Volatility:** **{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]`\n"
            f"* **1-Month 95% Value at Risk (VaR):** **${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)** `[EV-RISK-VAR-95]`\n"
            f"* **Expected Shortfall (CVaR):** **${metrics['expected_shortfall_95_dollar']:,.2f} ({metrics['expected_shortfall_95_pct']}%)** `[EV-TAIL-ES-95]`\n\n"
            f"You can test how a specific percentage drop or rally would affect your capital in the **What Could Happen?** and **Try a Change** tabs."
        )
        audit.log_event(
            event_type="PRICE_PREDICTION_REFUSAL",
            details={"query": user_query},
        )
        return {
            "source": "safety-filter",
            "response": pred_refusal,
            "answer": pred_refusal,
            "cited_evidences": [ev.to_dict() for ev in runner.registry.all()[:8]],
            "tool_calls": tool_calls_out,
            "safety_check": {"passed": True, "blocked_reason": "speculative_price_prediction_refused"},
        }

    # 5. Generate response via Gemini API or Deterministic Analyst
    context_str = format_portfolio_context(portfolio, runner=runner)
    api_key = get_api_key()

    if api_key and not api_key.startswith("AQ.Ab8RN6JqCrilO84"):
        try:
            res_dict = call_gemini_api(api_key, context_str, user_query, conversation_history)
        except Exception as e:
            res_dict = generate_offline_risk_response(portfolio, user_query, str(e), runner=runner)
    else:
        res_dict = generate_offline_risk_response(portfolio, user_query, runner=runner)

    resp_text = res_dict.get("response", "")

    # 6. Post-Generation OutputSafetyFilter Check (blocks buy/sell instructions & price predictions)
    filter_res = safety_filter.check_text(resp_text)
    if not filter_res.is_clean:
        audit.log_event(
            event_type="OUTPUT_SAFETY_FILTER_BLOCK",
            details={"query": user_query, "violations": filter_res.violations},
        )
        fallback = generate_offline_risk_response(portfolio, "should i buy or sell", runner=runner)
        resp_text = fallback["response"]
        res_dict["source"] = "safety-filter"

    # Collect cited Evidence objects (prioritizing explicitly cited [EV-*] IDs, plus top core evidences)
    cited_ids = re.findall(r"\[(EV-[A-Z0-9_-]+)\]", resp_text)
    cited_ev_objs = []
    seen_ev_ids = set()
    for cid in cited_ids:
        ev_obj = runner.registry.get(cid)
        if ev_obj and cid not in seen_ev_ids:
            seen_ev_ids.add(cid)
            cited_ev_objs.append(ev_obj.to_dict())
    for ev_obj in runner.registry.all():
        if len(cited_ev_objs) >= 10:
            break
        if ev_obj.id not in seen_ev_ids:
            seen_ev_ids.add(ev_obj.id)
            cited_ev_objs.append(ev_obj.to_dict())

    audit.log_event(
        event_type="AI_RISK_ANALYST_RESPONSE",
        details={
            "query": user_query,
            "source": res_dict["source"],
            "tool_calls_count": len(tool_calls_out),
            "cited_evidence_ids": [e["id"] for e in cited_ev_objs[:6]],
            "safety_filter_clean": filter_res.is_clean,
        },
    )

    return {
        "source": res_dict["source"],
        "response": resp_text,
        "answer": resp_text,
        "cited_evidences": cited_ev_objs,
        "tool_calls": tool_calls_out,
        "safety_check": {
            "passed": filter_res.is_clean,
            "blocked_reason": None if filter_res.is_clean else "; ".join(filter_res.violations),
        },
    }


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
Include bracketed Evidence IDs like [EV-CONC-TOP1-WEIGHT], [EV-RISK-ANN-VOL], [EV-RISK-VAR-95], [EV-TAIL-ES-95] next to quantitative claims.
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
            "source": "deterministic-risk-analyst",
            "response": "Your portfolio is empty.",
            "cited_evidences": [],
            "tool_calls": [],
            "safety_check": {"passed": True, "blocked_reason": None}
        }


def generate_offline_risk_response(
    portfolio: Portfolio,
    query: str,
    error_note: Optional[str] = None,
    runner: Optional[AgentToolRunner] = None,
) -> Dict[str, Any]:
    """Generates deterministic, institutional-quality risk analysis tailored to the user's specific query with [EV-*] citations."""
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

    known_tickers = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TLT", "GLD", "SPY", "QQQ", "VOO", "SMH", "VTI", "BND", "CASH", "VST", "ARKK"]
    mentioned_ticker = None
    for t in known_tickers:
        if t.lower() in words:
            mentioned_ticker = t
            break

    # 1. Buy / Sell Recommendations / Stock Picking Advice (Checked first so trade advice requests always get policy response)
    if any(p in q_lower for p in [
        "should i buy", "should i sell", "what should i buy", "what should i sell",
        "what to buy", "what to sell", "stock recommendation", "stock tip", "which stock",
        "tell me which stock to buy", "buy right now", "sell right now",
    ]) or ({"recommend", "recommendation", "tip", "advice"} & words):
        response_text = f"""### Risk Intelligence Policy: No Speculative Trade Calls

As an institutional-grade risk analyst, **Portfolio X-Ray does not predict stock prices or tell you to buy or sell specific assets.**

Here is what our deterministic risk engines show for your decision-making:

#### 1. Current Concentration & Safety Limits
* **Top-3 Concentration:** Your Top 3 holdings represent **{metrics['top_3_concentration']}%** of your capital ({metrics['concentration_tier']}, HHI: **{metrics['hhi']}** `[EV-CONC-LT-HHI]`).
* **Top Single-Issuer Exposure:** **{top_exp.ticker if top_exp else 'N/A'}** is at **{top_exp.total_weight*100:.1f}%** `[EV-CONC-TOP1-WEIGHT]` (`[EV-LIMIT-SINGLE-STOCK-{top_exp.ticker if top_exp else 'NONE'}]`).
* Adding non-correlated assets (such as broad-market index funds or fixed income) reduces portfolio volatility (**{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]`).

#### 2. Test Before You Trade
* Use the **Try a Change (What-If Simulator)** tab to simulate any hypothetical purchase or rebalance with realistic execution friction (`[EV-WHATIF-FRICTION]`) and human sign-off.
"""

    # 2. "Make money" / Growth / Returns / Profit / Upside / Wealth
    elif any(p in q_lower for p in [
        "make money", "making money", "want to make money", "grow money", "how do i make money",
        "get rich", "maximize return", "upside", "compound",
    ]) or ({"money", "profit"} & words and "lose" not in words) or ({"return", "returns"} & words) or ({"growth", "grow"} & words):
        top_driver_name = f"{top_theme['theme']} ({top_theme['weight']*100:.1f}%)" if top_theme else "Equities"
        response_text = f"""### Pursuing Growth & Returns: What Drives Your Money

In investing, **expected return is directly coupled to risk**—there is no excess return without exposing your capital to market volatility and potential drawdowns.

Here is an audit of how your portfolio is currently positioned and the trade-offs involved:

#### 1. What is Driving Your Potential Upside?
* **Primary Return Engine:** Your portfolio's growth is heavily tied to **{top_driver_name}**, led by **{top_exp.ticker if top_exp else 'your top stock'} ({top_exp.name if top_exp else ''})** at **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) effective exposure `[EV-LT-{top_exp.ticker if top_exp else 'NVDA'}]` `[EV-CONC-TOP1-WEIGHT]`.
* **Concentration Score:** Your Herfindahl-Hirschman Index (HHI) is **{metrics['hhi']}** ({metrics['concentration_tier']}) `[EV-CONC-LT-HHI]`.

#### 2. The Cost of Chasing High Returns: Downside Exposure
* **Annualized Volatility:** **{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]`.
* **1-Month 95% Value at Risk (VaR):** **${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)** `[EV-RISK-VAR-95]`.
* **Shock Scenario Vulnerability:** In an adverse market event like **{worst_stress['name'] if worst_stress else 'a Market Selloff'}**, your estimated portfolio drawdown is **{worst_stress['drawdown_pct'] if worst_stress else 0.0}% (-${abs(worst_stress['dollar_drawdown']) if worst_stress else 0.0:,.2f})**.

*(Note: Portfolio X-Ray analyzes risk and exposure. We do not provide speculative price predictions or specific buy/sell advice.)*
"""

    # 3. Safety / Capital Preservation / Is My Portfolio Safe? / Biggest Risk
    elif any(p in q_lower for p in [
        "is it safe", "is my portfolio safe", "how safe", "am i safe", "protect my money", "capital preservation",
        "biggest risk", "biggest vulnerability", "lose a lot",
    ]) or ({"safe", "safety", "secure"} & words) or ({"loss", "lose", "losing"} & words) or ({"danger", "worried", "risky", "vulnerability"} & words):
        safety_verdict = "High Risk / Concentrated" if metrics['hhi'] > 2500 or metrics['annual_volatility'] > 22 else ("Moderately Balanced" if metrics['hhi'] > 1500 else "Well Diversified / Defensive")
        response_text = f"""### Portfolio Safety & Biggest Vulnerability Audit

Here is an objective, evidence-backed assessment of your portfolio's risk profile:

#### 1. Overall Risk Verdict: **{safety_verdict}**
* **Concentration Tier:** **{metrics['concentration_tier']}** (HHI: **{metrics['hhi']}** `[EV-CONC-LT-HHI]`).
* **Dominant Single-Company Exposure:** Your largest underlying position is **{top_exp.ticker if top_exp else 'N/A'} ({top_exp.name if top_exp else ''})** representing **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) of your total wealth `[EV-CONC-TOP1-WEIGHT]` `[EV-LIMIT-SINGLE-STOCK-{top_exp.ticker if top_exp else 'NONE'}]`.

#### 2. Downside Losses in Numbers
* **Annualized Volatility:** **{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]`.
* **Expected Monthly Drop (95% VaR):** **${metrics['var_95_1m_dollar']:,.2f} ({metrics['var_95_1m_pct']}%)** `[EV-RISK-VAR-95]`.
* **Tail Risk (Expected Shortfall / CVaR):** If market losses exceed normal VaR, the average tail loss is **${metrics['expected_shortfall_95_dollar']:,.2f} ({metrics['expected_shortfall_95_pct']}%)** `[EV-TAIL-ES-95]`.
* **Stress Test Drawdown:** Under **{worst_stress['name'] if worst_stress else 'Market Correction'}**, this portfolio drops by an estimated **{worst_stress['drawdown_pct'] if worst_stress else 0.0}% (-${abs(worst_stress['dollar_drawdown']) if worst_stress else 0.0:,.2f})**.

#### 3. Alignment with Your Profile (**{risk_tol}**)
* Compare these figures against your stated risk tolerance in the **Overview** and **Try a Change** tabs.
"""

    # 4. Look-Through / Overlap / Hidden Concentration
    elif any(p in q_lower for p in [
        "look through", "look-through", "overlap", "hidden", "etf overlap", "diversified", "diversification"
    ]) or ({"overlap", "overlapping", "etf", "etfs", "diversified"} & words):
        overlap_examples = []
        for exp in exposures:
            if exp.indirect_weight > 0.01 and exp.direct_weight > 0.01:
                sources_str = ", ".join(exp.contributing_sources.keys())
                overlap_examples.append(
                    f"* **{exp.ticker} ({exp.name}):** You own **{exp.direct_weight*100:.1f}%** directly, plus **{exp.indirect_weight*100:.1f}%** hidden inside ETFs ({sources_str}), making your true exposure **{exp.total_weight*100:.1f}%** (${exp.total_dollar_value:,.2f}) `[EV-LT-{exp.ticker}]`."
                )
        if not overlap_examples and top_exp:
            overlap_examples.append(
                f"* **{top_exp.ticker} ({top_exp.name}):** Effective exposure is **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) `[EV-LT-{top_exp.ticker}]`."
            )

        response_text = f"""### Understanding Look-Through & Hidden Fund Overlap

Many investors count the number of funds in their account and assume they are diversified. **Look-Through Decomposition reveals the exact companies you own underneath** `[EV-CONC-LT-NEFF]`.

#### 1. Your Hidden Overlaps in This Portfolio:
{chr(10).join(overlap_examples[:4])}

#### 2. Why This Matters
* When you own individual tech stocks alongside ETFs like QQQ, SMH, and VOO, **you are multiplying your single-company concentration** (`[EV-CONC-TOP1-WEIGHT]`).
* If that top company stumbles, the loss hits both your direct shares and your ETFs simultaneously.
"""

    # 5. Stress Testing / Macro / Crashes / Recession / Inflation / Rates
    elif any(p in q_lower for p in [
        "crash", "recession", "inflation", "interest rate", "interest rates", "fed", "bear market", "rate hike", "drawdown", "stress test"
    ]) or ({"shock", "shocks", "crisis", "rates"} & words):
        stress_lines = []
        for st in stress:
            stress_lines.append(f"* **{st['name']}:** Estimated drawdown of **{st['drawdown_pct']}%** (-${abs(st['dollar_drawdown']):,.2f}). {st.get('description', '')}")

        response_text = f"""### Historical & Macro Stress Test Analysis

Here is how your current holdings respond to severe market crises and interest-rate shocks (`[EV-STRESS-YEAR_2022_FED_RATE_SHOCK]`, `[EV-STRESS-MAR_2020_COVID_CRASH]`):

#### 1. Simulated Scenario Drawdowns:
{chr(10).join(stress_lines)}

#### 2. Biggest Vulnerability in a Crisis:
* Your primary economic exposure is **{top_theme['theme'] if top_theme else 'Growth'}** (**{top_theme['weight']*100:.1f}%**).
* During interest-rate spikes or liquidity crunches, high-duration tech and growth valuations compress rapidly.
* With a 1-Month 95% VaR of **${metrics['var_95_1m_dollar']:,.2f}** `[EV-RISK-VAR-95]`, ensure your time horizon allows you to ride out temporary drawdowns.
"""

    # 6. Specific Ticker Inquiry (e.g., NVDA, AAPL, MSFT, SPY, etc.)
    elif mentioned_ticker and not any(p in q_lower for p in ["what if", "50%", "put", "allocate"]):
        target_exp = next((exp for exp in exposures if exp.ticker == mentioned_ticker), None)
        direct_h = next((h for h in portfolio.holdings if h.ticker == mentioned_ticker), None)

        if target_exp:
            indirect_sources = ", ".join(f"{k} (+{v*100:.1f}%)" for k, v in target_exp.contributing_sources.items()) if target_exp.contributing_sources else "None (held directly)"
            response_text = f"""### Holding Deep-Dive: **{target_exp.ticker} ({target_exp.name})** `[EV-LT-{target_exp.ticker}]`

Here is your exact look-through breakdown for **{target_exp.ticker}**:

#### 1. Effective Exposure Breakdown
* **Total Effective Weight:** **{target_exp.total_weight*100:.1f}%** (${target_exp.total_dollar_value:,.2f}) `[EV-LT-{target_exp.ticker}]`
* **Direct Holding Weight:** **{target_exp.direct_weight*100:.1f}%** {f'({direct_h.shares:.1f} shares @ ${direct_h.price:.2f})' if direct_h else '(Not held directly)'}
* **Indirect Exposure via ETFs:** **{target_exp.indirect_weight*100:.1f}%**
* **Contributing ETF Sources:** {indirect_sources}

#### 2. Risk Role in Your Portfolio
* **Sector:** {target_exp.sector}
* **Themes:** {', '.join(target_exp.themes) if target_exp.themes else 'General Market'}
* **Annual Volatility:** **{target_exp.annual_volatility*100:.1f}%**
* With a total weight of **{target_exp.total_weight*100:.1f}%**, a 10% move in {target_exp.ticker} shifts your entire portfolio by **{target_exp.total_weight*10:.2f}% (${target_exp.total_dollar_value*0.10:,.2f})**.
"""
        else:
            response_text = f"""### Analysis for **{mentioned_ticker}**

* **Status in Portfolio:** **{mentioned_ticker}** is not currently in your direct holdings or top ETF look-through exposures.
* **Want to see what adding {mentioned_ticker} would do?** Test adding {mentioned_ticker} in the **Try a Change (What-If Simulator)** tab to inspect its impact on your concentration (`[EV-CONC-LT-HHI]`) and downside VaR (`[EV-RISK-VAR-95]`).
"""

    # 7. What-If Simulation Query
    elif any(p in q_lower for p in ["what if", "50%", "put", "allocate"]):
        target_ticker = mentioned_ticker or "NVDA"
        sim = simulate_what_if(portfolio, {target_ticker: 50.0, "SPY": 50.0})
        prop_m = sim["proposed_metrics"]
        cur_m = sim["current_metrics"]
        delta_vol = sim["metrics_delta"]["volatility_change"]
        first_stress = sim["stress_comparison"][0] if sim.get("stress_comparison") else {}
        scen_name = first_stress.get("name", first_stress.get("scenario", "Historical Stress Shock"))

        response_text = f"""### What-If Risk Simulation: 50% Allocation to {target_ticker}

#### 1. Exposure & Look-Through Impact
* If you allocate 50% directly into **{target_ticker}**, your effective concentration in that single asset moves from **{cur_m['top_1_concentration']}%** to **{prop_m['top_1_concentration']}%** `[EV-CONC-TOP1-WEIGHT]`.
* Your portfolio's concentration index (HHI) changes by **{sim['metrics_delta']['hhi_change']:+,.1f} points**, moving into the **{prop_m['concentration_tier']}** zone `[EV-CONC-LT-HHI]`.

#### 2. Downside & Volatility Trade-Off
* **Annualized Volatility:** Moves from **{cur_m['annual_volatility']}%** to **{prop_m['annual_volatility']}%** ({delta_vol:+,.2f}% shift) `[EV-RISK-ANN-VOL]`.
* **1-Month 95% Value at Risk (VaR):** Moves to **${prop_m['var_95_1m_dollar']:,.2f}** (**{prop_m['var_95_1m_pct']}%** of total capital) `[EV-RISK-VAR-95]`.

#### 3. Stress Scenario Resilience
* Under **{scen_name}**, the proposed portfolio faces an estimated **{first_stress.get('proposed_drawdown_pct', 0.0)}%** drawdown compared to **{first_stress.get('current_drawdown_pct', 0.0)}%** currently.

#### 4. Investor Profile Check
{('⚠️ **Risk Threshold Warning:** ' + ' '.join(sim['profile_warnings'])) if sim['profile_warnings'] else '✅ This adjustment remains within your stated risk tolerance bounds.'}
"""

    # 8. General / Autonomous Audit Fallback
    else:
        response_text = f"""### Autonomous Risk Audit for Your Portfolio

Here is the evidence-backed risk breakdown of your holdings:

#### 1. Current Portfolio Context
* **Total Portfolio Capital:** **${metrics['portfolio_value']:,.2f}**
* **Concentration Tier:** **{metrics['concentration_tier']}** (HHI: **{metrics['hhi']}** `[EV-CONC-LT-HHI]`).
* **Top Look-Through Holding:** **{top_exp.ticker if top_exp else 'N/A'}** represents **{top_exp.total_weight*100:.1f}%** (${top_exp.total_dollar_value:,.2f}) of your wealth `[EV-CONC-TOP1-WEIGHT]`.
* **Annualized Volatility:** **{metrics['annual_volatility']}%** `[EV-RISK-ANN-VOL]` (95% 1-Month VaR: **${metrics['var_95_1m_dollar']:,.2f}** `[EV-RISK-VAR-95]`, Expected Shortfall: **${metrics['expected_shortfall_95_dollar']:,.2f}** `[EV-TAIL-ES-95]`).

#### 2. Key Considerations for You
* **Dominant Theme:** Your capital has **{top_theme['weight']*100:.1f}%** exposure to **{top_theme['theme'] if top_theme else 'Tech'}**.
* **Alignment with Risk Profile (**{risk_tol}**):** Check whether potential crisis drawdowns of **{worst_stress['drawdown_pct'] if worst_stress else 0.0}%** (`[EV-STRESS-YEAR_2022_FED_RATE_SHOCK]`) align with your time horizon.
"""

    return {
        "source": "deterministic-risk-analyst" if not error_note else f"deterministic-risk-analyst (API fallback)",
        "response": response_text.strip(),
    }
