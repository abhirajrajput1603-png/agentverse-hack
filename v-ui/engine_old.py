"""Core quantitative risk engine for Portfolio X-Ray.

Pure, deterministic financial calculations using standard library only.
Testable without models or external data providers.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

try:
    from data import (
        ETF_LOOKTHROUGH,
        ASSET_METADATA,
        get_asset_info,
        STRESS_SCENARIOS,
        COMPANY_SYNONYMS,
    )
except ImportError:
    from portfolio_xray.data import (
        ETF_LOOKTHROUGH,
        ASSET_METADATA,
        get_asset_info,
        STRESS_SCENARIOS,
        COMPANY_SYNONYMS,
    )


@dataclass
class Holding:
    ticker: str
    shares: float
    price: float  # In dollars (or base currency)
    
    @property
    def market_value(self) -> float:
        return self.shares * self.price


@dataclass
class Portfolio:
    holdings: List[Holding] = field(default_factory=list)
    user_profile: Dict[str, Any] = field(default_factory=lambda: {
        "risk_tolerance": "Moderate",
        "horizon_years": 7,
        "loss_capacity": "Moderate (15-20% max drawdown)",
    })

    @property
    def total_value(self) -> float:
        return sum(h.market_value for h in self.holdings)


@dataclass
class EffectiveExposure:
    ticker: str
    name: str
    total_weight: float
    total_dollar_value: float
    direct_weight: float
    indirect_weight: float
    contributing_sources: Dict[str, float]  # Source ticker -> indirect weight contribution
    sector: str
    themes: List[str]
    annual_volatility: float


def compute_lookthrough_exposure(portfolio: Portfolio) -> List[EffectiveExposure]:
    """Decomposes ETF holdings to reveal overlapping company weights for assets the user has individually invested in.

    Index and ETF holdings are NOT separated into constituent components unless
    the investor has directly invested in that component individually.
    """
    total_val = portfolio.total_value
    if total_val <= 0:
        return []

    # Map holding weights
    holding_weights: Dict[str, float] = {}
    for h in portfolio.holdings:
        t = h.ticker.upper().strip()
        holding_weights[t] = holding_weights.get(t, 0.0) + (h.market_value / total_val)

    # Assets the investor has individually invested in (non-ETF holdings)
    individually_invested = {
        h.ticker.upper().strip()
        for h in portfolio.holdings
        if h.ticker.upper().strip() not in ETF_LOOKTHROUGH
    }

    # Accumulate exposures
    direct_weights: Dict[str, float] = {}
    indirect_weights: Dict[str, float] = {}
    contributions: Dict[str, Dict[str, float]] = {}  # target -> {etf: weight_contribution}

    for ticker, weight in holding_weights.items():
        if ticker in ETF_LOOKTHROUGH:
            # Only unpack constituents that the user has individually invested in
            extracted_weight_fraction = 0.0
            for constituent, constituent_weight in ETF_LOOKTHROUGH[ticker].items():
                if constituent in individually_invested:
                    effective_contribution = weight * constituent_weight
                    indirect_weights[constituent] = indirect_weights.get(constituent, 0.0) + effective_contribution
                    if constituent not in contributions:
                        contributions[constituent] = {}
                    contributions[constituent][ticker] = effective_contribution
                    extracted_weight_fraction += constituent_weight

            # The remaining portion of the ETF/index remains as the index itself
            remaining_index_weight = weight * (1.0 - extracted_weight_fraction)
            if remaining_index_weight > 0.00001:
                direct_weights[ticker] = direct_weights.get(ticker, 0.0) + remaining_index_weight
        else:
            direct_weights[ticker] = direct_weights.get(ticker, 0.0) + weight

    all_tickers = set(direct_weights.keys()) | set(indirect_weights.keys())
    results: List[EffectiveExposure] = []

    for t in all_tickers:
        d_wt = direct_weights.get(t, 0.0)
        ind_wt = indirect_weights.get(t, 0.0)
        tot_wt = d_wt + ind_wt
        info = get_asset_info(t)

        results.append(EffectiveExposure(
            ticker=t,
            name=info.get("name", t),
            total_weight=tot_wt,
            total_dollar_value=tot_wt * total_val,
            direct_weight=d_wt,
            indirect_weight=ind_wt,
            contributing_sources=contributions.get(t, {}),
            sector=info.get("sector", "General"),
            themes=info.get("themes", []),
            annual_volatility=info.get("annual_volatility", 0.25),
        ))

    # Sort descending by total weight
    results.sort(key=lambda x: x.total_weight, reverse=True)
    return results


def compute_thematic_exposure(exposures: List[EffectiveExposure], min_weight: float = 0.05) -> List[Dict[str, Any]]:
    """Sums effective exposures across macroeconomic, industry, and thematic drivers.

    Filters out any themes with exposure under min_weight (default: 0.05 / 5%).
    """
    theme_totals: Dict[str, float] = {}
    theme_assets: Dict[str, List[Dict[str, Any]]] = {}

    for exp in exposures:
        for theme in exp.themes:
            theme_totals[theme] = theme_totals.get(theme, 0.0) + exp.total_weight
            if theme not in theme_assets:
                theme_assets[theme] = []
            theme_assets[theme].append({
                "ticker": exp.ticker,
                "weight": exp.total_weight,
            })

    output = []
    for theme, total_weight in theme_totals.items():
        # Exclude themes under 5% exposure
        if round(total_weight, 4) < min_weight:
            continue
        assets = sorted(theme_assets[theme], key=lambda a: a["weight"], reverse=True)
        output.append({
            "theme": theme,
            "weight": total_weight,
            "top_assets": assets[:4],
        })

    output.sort(key=lambda x: x["weight"], reverse=True)
    return output


def compute_risk_metrics(portfolio: Portfolio, exposures: List[EffectiveExposure]) -> Dict[str, Any]:
    """Calculates quantitative risk metrics: HHI, Volatility, 95% VaR, and Expected Shortfall."""
    total_val = portfolio.total_value
    if total_val <= 0 or not exposures:
        return {
            "portfolio_value": 0.0,
            "hhi": 0.0,
            "concentration_tier": "None",
            "top_1_concentration": 0.0,
            "top_3_concentration": 0.0,
            "annual_volatility": 0.0,
            "var_95_1m_pct": 0.0,
            "var_95_1m_dollar": 0.0,
            "expected_shortfall_95_pct": 0.0,
            "expected_shortfall_95_dollar": 0.0,
        }

    # 1. Herfindahl-Hirschman Index (HHI)
    # Sum of squared weights * 10,000 (standard regulatory scale)
    hhi = sum((exp.total_weight * 100.0) ** 2 for exp in exposures)
    if hhi > 2500:
        concentration_tier = "High Concentration"
    elif hhi > 1500:
        concentration_tier = "Moderate Concentration"
    else:
        concentration_tier = "Diversified"

    # Top concentrations
    top_1 = exposures[0].total_weight if len(exposures) >= 1 else 0.0
    top_3 = sum(exp.total_weight for exp in exposures[:3])

    # 2. Portfolio Annualized Volatility
    # Sigma_p = sqrt( sum(w_i^2 * sigma_i^2) + sum_{i!=j}(w_i * w_j * rho * sigma_i * sigma_j) )
    # Pairwise average correlation rho ~= 0.40 for diversified equity / cross-asset
    rho = 0.40
    var_terms = 0.0
    covar_terms = 0.0

    n = len(exposures)
    for i in range(n):
        w_i = exposures[i].total_weight
        s_i = exposures[i].annual_volatility
        var_terms += (w_i * s_i) ** 2

        for j in range(i + 1, n):
            w_j = exposures[j].total_weight
            s_j = exposures[j].annual_volatility
            # Bonds and cash have lower or negative equity correlation
            local_rho = rho
            if any(k in exposures[i].ticker for k in ["US_TREASURY", "TLT", "BND"]) or any(k in exposures[j].ticker for k in ["US_TREASURY", "TLT", "BND"]):
                local_rho = -0.15
            elif exposures[i].ticker == "CASH" or exposures[j].ticker == "CASH":
                local_rho = 0.0
            covar_terms += 2.0 * w_i * w_j * local_rho * s_i * s_j

    port_vol = math.sqrt(max(0.0, var_terms + covar_terms))

    # 3. Parametric 95% 1-Month Value at Risk (VaR)
    # 1 month = 1/12 of year, Z_0.95 = 1.645
    monthly_vol = port_vol / math.sqrt(12.0)
    var_95_pct = 1.645 * monthly_vol
    var_95_dollar = var_95_pct * total_val

    # 4. Expected Shortfall (CVaR 95%)
    # For normal distribution, CVaR_95 ~= 1.25 * VaR_95
    cvar_95_pct = var_95_pct * 1.25
    cvar_95_dollar = cvar_95_pct * total_val

    return {
        "portfolio_value": round(total_val, 2),
        "hhi": round(hhi, 1),
        "concentration_tier": concentration_tier,
        "top_1_concentration": round(top_1 * 100.0, 2),
        "top_3_concentration": round(top_3 * 100.0, 2),
        "annual_volatility": round(port_vol * 100.0, 2),
        "var_95_1m_pct": round(var_95_pct * 100.0, 2),
        "var_95_1m_dollar": round(var_95_dollar, 2),
        "expected_shortfall_95_pct": round(cvar_95_pct * 100.0, 2),
        "expected_shortfall_95_dollar": round(cvar_95_dollar, 2),
    }


def select_stress_scenarios(
    portfolio: Portfolio,
    exposures: List[EffectiveExposure],
    limit: int = 4,
) -> List[Dict[str, Any]]:
    """Dynamically selects the most relevant stress scenarios tailored to the portfolio's factor concentrations.

    - High AI / Tech / Semiconductor exposure: prioritizes AI Capex, Taiwan Chip Chokepoint, LLM Margin Squeeze, Tech Derating.
    - FIG (Financial Institutions Group) / Banking / Rate-sensitive exposure: prioritizes Regional Banking Runs, Yield Curve Inversions, CRE Credit Spread Blowouts, and 2008 Credit Deleveraging.
    - Aerospace & Defense / Space: prioritizes Satellite Grounding, Defense Sequestration, Aviation Bottlenecks.
    - Energy & Commodities: prioritizes Stagflation Oil Shocks, Crude Demand Collapses.
    - Crypto & Speculative: prioritizes Crypto Winter & Stablecoin Contagion.
    - Broad / Macro Core: incorporates foundational baseline liquidity & duration shocks.
    """
    total_val = portfolio.total_value
    if not exposures or total_val <= 0:
        return STRESS_SCENARIOS[:limit]

    # Calculate thematic exposure weights
    theme_weights: Dict[str, float] = {}
    for exp in exposures:
        for theme in exp.themes:
            theme_weights[theme] = theme_weights.get(theme, 0.0) + exp.total_weight

    # Sector / Theme cluster weights
    ai_weight = sum(theme_weights.get(t, 0.0) for t in [
        "AI Infrastructure & Capex", "Semiconductor Supply Chain", "Generative AI",
        "High Growth Tech", "Cloud Computing", "Mega-Cap Tech", "Geopolitical Taiwan Risk"
    ])
    fig_weight = sum(theme_weights.get(t, 0.0) for t in [
        "Banking & Credit", "Financial Services", "Credit Spread Exposure",
        "Interest Rate Sensitive (Yield Curve)", "Interest Rate Duration", "Mortgage Rate Prepayment"
    ])
    defense_weight = sum(theme_weights.get(t, 0.0) for t in [
        "Aerospace & Defense", "Government Defense Prime", "High Growth Space Tech",
        "Satellite Communications (Starlink)", "Commercial Aviation", "Defense & Intelligence AI"
    ])
    energy_weight = sum(theme_weights.get(t, 0.0) for t in [
        "Energy & Commodities", "Nuclear Energy & Power Demand", "Inflation Hedge"
    ])
    consumer_weight = sum(theme_weights.get(t, 0.0) for t in [
        "Consumer Discretionary", "Consumer Hardware & Ecosystem", "Mobility & Gig Economy", "EV & Clean Tech"
    ])
    crypto_weight = sum(theme_weights.get(t, 0.0) for t in [
        "Crypto & Digital Assets", "Fintech Speculation"
    ])

    # Score each scenario based on relevance to portfolio
    scored_scenarios = []
    for scen in STRESS_SCENARIOS:
        cat = scen.get("category", "macro")
        target_themes = scen.get("target_themes", [])

        # 1. Base theme relevance score from matching portfolio themes
        base_relevance = sum(theme_weights.get(t, 0.0) for t in target_themes)

        # 2. Factor cluster boosts based on dominant exposure
        cluster_boost = 0.0
        if cat == "ai":
            if ai_weight >= 0.20:
                cluster_boost = ai_weight * 2.5
            elif ai_weight > 0.05:
                cluster_boost = ai_weight * 1.0
        elif cat == "fig":
            if fig_weight >= 0.12:
                cluster_boost = fig_weight * 3.0  # Strong boost for FIG / Rates
            elif fig_weight > 0.04:
                cluster_boost = fig_weight * 1.5
        elif cat == "defense":
            if defense_weight >= 0.12:
                cluster_boost = defense_weight * 3.0
            elif defense_weight > 0.04:
                cluster_boost = defense_weight * 1.5
        elif cat == "energy":
            if energy_weight >= 0.12:
                cluster_boost = energy_weight * 2.5
            elif energy_weight > 0.04:
                cluster_boost = energy_weight * 1.2
        elif cat == "crypto":
            if crypto_weight >= 0.05:
                cluster_boost = crypto_weight * 4.0
        elif cat == "consumer":
            if consumer_weight >= 0.15:
                cluster_boost = consumer_weight * 1.8
        elif cat in ("macro", "liquidity", "rates"):
            # Core macro scenarios have baseline relevance so they serve as foundational anchors
            cluster_boost = 0.25

        score = base_relevance + cluster_boost
        scored_scenarios.append((score, scen))

    # Sort scenarios by score descending
    scored_scenarios.sort(key=lambda x: x[0], reverse=True)

    # Pick top scenarios
    selected = [scen for _, scen in scored_scenarios[:limit]]
    return selected


def run_stress_tests(
    portfolio: Portfolio,
    exposures: List[EffectiveExposure],
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Evaluates portfolio drawdown impact across dynamically selected or specified shock scenarios."""
    total_val = portfolio.total_value
    results = []

    if scenarios is None:
        scenarios = select_stress_scenarios(portfolio, exposures, limit=4)

    for scenario in scenarios:
        shocks = scenario["shocks"]
        default_shock = scenario.get("default_equity_shock", -0.15)
        
        portfolio_shock = 0.0
        asset_impacts = []

        for exp in exposures:
            # Asset shock is determined by its thematic sensitivity
            matched_shocks = [shocks[theme] for theme in exp.themes if theme in shocks]
            if matched_shocks:
                asset_shock = sum(matched_shocks) / len(matched_shocks)
            elif exp.ticker == "CASH":
                asset_shock = shocks.get("Capital Preservation", 0.0)
            elif "US_TREASURY" in exp.ticker or exp.ticker in ("TLT", "BND"):
                asset_shock = shocks.get("Defensive Safe Haven", default_shock * 0.2)
            else:
                asset_shock = default_shock

            weighted_impact = exp.total_weight * asset_shock
            portfolio_shock += weighted_impact

            asset_impacts.append({
                "ticker": exp.ticker,
                "weight": exp.total_weight,
                "asset_shock_pct": round(asset_shock * 100.0, 1),
                "dollar_impact": round(exp.total_dollar_value * asset_shock, 2),
            })

        dollar_drawdown = portfolio_shock * total_val
        results.append({
            "scenario_id": scenario["id"],
            "name": scenario["name"],
            "category": scenario.get("category", "macro"),
            "category_badge": scenario.get("category_badge", "Macro Risk"),
            "description": scenario["description"],
            "drawdown_pct": round(portfolio_shock * 100.0, 2),
            "dollar_drawdown": round(dollar_drawdown, 2),
            "worst_hit_assets": sorted(asset_impacts, key=lambda a: a["dollar_impact"])[:3],
            "resilient_assets": sorted(asset_impacts, key=lambda a: a["dollar_impact"], reverse=True)[:2],
        })

    return results


def resolve_symbol_or_name(raw: str) -> str:
    """Resolves a ticker symbol, company name, or synonym into a canonical ticker."""
    s = str(raw).strip()
    if not s:
        return "CUSTOM"
    upper = s.upper()
    if upper in ASSET_METADATA:
        return upper
    lower_norm = re.sub(r"\s+", " ", s.lower()).strip()
    if lower_norm in COMPANY_SYNONYMS:
        return COMPANY_SYNONYMS[lower_norm]
    stripped = re.sub(r"[^a-zA-Z0-9.]", "", s).lower()
    if stripped in COMPANY_SYNONYMS:
        return COMPANY_SYNONYMS[stripped]
    for token in lower_norm.split():
        clean_tok = re.sub(r"[^a-zA-Z0-9.]", "", token)
        if clean_tok in COMPANY_SYNONYMS:
            return COMPANY_SYNONYMS[clean_tok]
    for sym, meta in ASSET_METADATA.items():
        name = meta.get("name", "").lower()
        if stripped and (stripped in name or name in stripped or lower_norm in name):
            return sym
    return upper


def simulate_what_if(
    current_portfolio: Portfolio,
    target_allocations: Dict[str, float],  # Ticker or Stock Name -> Target % of portfolio (0 to 100)
) -> Dict[str, Any]:
    """Compares current portfolio against a proposed target allocation with full risk deltas."""
    total_val = current_portfolio.total_value
    if total_val <= 0:
        total_val = 10000.0

    # Build proposed portfolio
    proposed_holdings: List[Holding] = []
    
    # Determine price reference from existing holdings or metadata defaults
    known_prices: Dict[str, float] = {h.ticker.upper(): h.price for h in current_portfolio.holdings}
    default_prices = {
        "NVDA": 130.0, "MSFT": 420.0, "AAPL": 225.0, "AMZN": 185.0, "GOOGL": 165.0,
        "META": 580.0, "TSLA": 220.0, "QQQ": 480.0, "SPY": 550.0, "VOO": 500.0,
        "SMH": 250.0, "TLT": 95.0, "GLD": 230.0, "BND": 72.0, "CASH": 1.0,
    }

    # Resolve targets and handle partial allocations (reweighting existing holdings)
    resolved_targets: Dict[str, float] = {}
    target_ticker_resolved = ""
    for ticker, pct in target_allocations.items():
        t = resolve_symbol_or_name(ticker)
        if not target_ticker_resolved:
            target_ticker_resolved = t
        resolved_targets[t] = resolved_targets.get(t, 0.0) + float(pct)

    # Normalize decimal inputs (e.g. 0.25 -> 25.0)
    if 0 < sum(resolved_targets.values()) <= 1.0:
        resolved_targets = {k: v * 100.0 for k, v in resolved_targets.items()}

    total_target = sum(resolved_targets.values())
    if total_target <= 0:
        total_target = 20.0
        resolved_targets = {"NVDA": 20.0}
        target_ticker_resolved = "NVDA"

    # If target is partial (e.g. allocate 25% to NVDA) and portfolio has existing holdings,
    # reweight the remaining (100 - total_target)% across existing portfolio holdings
    if total_target < 100.0 and current_portfolio.holdings and total_val > 0:
        remaining_pct = 100.0 - total_target
        for h in current_portfolio.holdings:
            can_h = resolve_symbol_or_name(h.ticker)
            cur_wt = (h.shares * h.price) / total_val
            alloc_pct = remaining_pct * cur_wt
            resolved_targets[can_h] = resolved_targets.get(can_h, 0.0) + alloc_pct
        norm_targets = resolved_targets
    else:
        norm_targets = {t: (pct / total_target) * 100.0 for t, pct in resolved_targets.items()}

    for t, pct in norm_targets.items():
        allocated_dollars = (pct / 100.0) * total_val
        price = known_prices.get(t, default_prices.get(t, 100.0))
        shares = allocated_dollars / price
        proposed_holdings.append(Holding(ticker=t, shares=shares, price=price))

    if not target_ticker_resolved and norm_targets:
        target_ticker_resolved = list(norm_targets.keys())[0]

    proposed_portfolio = Portfolio(holdings=proposed_holdings, user_profile=current_portfolio.user_profile)

    # Compute both states
    cur_exp = compute_lookthrough_exposure(current_portfolio)
    prop_exp = compute_lookthrough_exposure(proposed_portfolio)

    cur_metrics = compute_risk_metrics(current_portfolio, cur_exp)
    prop_metrics = compute_risk_metrics(proposed_portfolio, prop_exp)

    # Dynamic stress scenarios tailored to the proposed allocation
    shared_scenarios = select_stress_scenarios(proposed_portfolio, prop_exp, limit=4)
    cur_stress = run_stress_tests(current_portfolio, cur_exp, scenarios=shared_scenarios)
    prop_stress = run_stress_tests(proposed_portfolio, prop_exp, scenarios=shared_scenarios)

    # Risk Profile Alignment Evaluation
    tolerance = current_portfolio.user_profile.get("risk_tolerance", "Moderate")
    max_safe_vol = {
        "Very Low": 8.0,
        "Low": 12.0,
        "Conservative": 12.0,
        "Moderate": 22.0,
        "High": 30.0,
        "Very High": 42.0,
        "Aggressive": 42.0,
    }.get(tolerance, 22.0)
    max_safe_top1 = {
        "Very Low": 10.0,
        "Low": 15.0,
        "Conservative": 15.0,
        "Moderate": 25.0,
        "High": 35.0,
        "Very High": 50.0,
        "Aggressive": 50.0,
    }.get(tolerance, 25.0)

    profile_warnings = []
    if prop_metrics["annual_volatility"] > max_safe_vol:
        profile_warnings.append(
            f"Annual volatility ({prop_metrics['annual_volatility']}%) exceeds the recommended range for a {tolerance} investor (max ~{max_safe_vol}%)."
        )
    if prop_metrics["top_1_concentration"] > max_safe_top1:
        profile_warnings.append(
            f"Top single asset concentration ({prop_metrics['top_1_concentration']}%) breaches the {tolerance} risk threshold ({max_safe_top1}%)."
        )

    return {
        "current_metrics": cur_metrics,
        "proposed_metrics": prop_metrics,
        "metrics_delta": {
            "volatility_change": round(prop_metrics["annual_volatility"] - cur_metrics["annual_volatility"], 2),
            "hhi_change": round(prop_metrics["hhi"] - cur_metrics["hhi"], 1),
            "var_95_dollar_change": round(prop_metrics["var_95_1m_dollar"] - cur_metrics["var_95_1m_dollar"], 2),
            "top_1_change": round(prop_metrics["top_1_concentration"] - cur_metrics["top_1_concentration"], 2),
        },
        "current_lookthrough_top5": [
            {"ticker": e.ticker, "name": e.name, "weight": round(e.total_weight * 100.0, 2)}
            for e in cur_exp[:5]
        ],
        "proposed_lookthrough_top5": [
            {"ticker": e.ticker, "name": e.name, "weight": round(e.total_weight * 100.0, 2)}
            for e in prop_exp[:5]
        ],
        "stress_comparison": [
            {
                "name": cs["name"],
                "current_drawdown_pct": cs["drawdown_pct"],
                "proposed_drawdown_pct": ps["drawdown_pct"],
                "delta_drawdown_pct": round(ps["drawdown_pct"] - cs["drawdown_pct"], 2),
            }
            for cs, ps in zip(cur_stress, prop_stress)
        ],
        "profile_warnings": profile_warnings,
        "is_aligned_with_profile": len(profile_warnings) == 0,
        "resolved_target": {
            "ticker": target_ticker_resolved,
            "name": ASSET_METADATA.get(target_ticker_resolved, {}).get("name", target_ticker_resolved),
        },
    }


def analyze_headline_event(
    portfolio: Portfolio,
    headline: str,
) -> Dict[str, Any]:
    """Analyzes how an external market headline or macro event impacts the portfolio's specific exposures.
    
    Dynamically identifies:
    1. Sentiment polarity (positive/easing/surge vs negative/tightening/slump)
    2. Multi-factor topic domains (rates/fed, AI/chips, energy/oil, defense/space, banks/credit, tariffs, macro)
    3. Direct company/ticker mentions (e.g. SpaceX, Tesla, Nvidia, Apple, etc.)
    """
    exposures = compute_lookthrough_exposure(portfolio)
    headline_lower = headline.lower().strip()

    shocked_themes: Dict[str, float] = {}
    direct_asset_shocks: Dict[str, float] = {}
    rationale: List[str] = []

    def has_any_word(words: List[str]) -> bool:
        return any(re.search(r"\b" + re.escape(w) + r"\b", headline_lower) for w in words)

    positive_cues = [
        "cut", "cuts", "cutting", "slash", "slashes", "slashed", "easing", "eased", "ease",
        "dovish", "pivot", "relief", "falls", "falling", "cools", "cooling", "slows", "slowing",
        "lower", "drop", "drops", "subside", "subsides", "pause", "pauses",
        "beat", "beats", "beating", "surge", "surges", "surged", "soar", "soars", "soared",
        "jump", "jumps", "jumped", "rally", "rallies", "rallied", "gain", "gains", "gained",
        "boom", "booming", "boost", "boosts", "boosted", "record", "growth", "accelerate",
        "accelerates", "accelerated", "outperform", "outperforms", "bullish", "resilient",
        "resilience", "strong", "strength", "optimism", "optimistic", "stimulus", "expand",
        "expands", "expansion", "breakthrough", "approval", "approves", "approved", "contract",
        "win", "wins", "won", "award", "awarded", "success", "successful", "historic"
    ]

    negative_cues = [
        "hike", "hikes", "hiking", "hiked", "increase", "increases", "tightening", "tighten",
        "tightened", "hawkish", "spikes", "spiking", "spiked", "hot", "hotter", "persistent",
        "sticky", "higher", "crash", "crashes", "crashed", "plunge", "plunges", "plunged",
        "slump", "slumps", "slumped", "sink", "sinks", "sank", "tumble", "tumbles", "tumbled",
        "selloff", "rout", "miss", "misses", "missed", "warning", "warns", "warned",
        "layoff", "layoffs", "curb", "curbs", "curbed", "ban", "bans", "banned", "tariff",
        "tariffs", "sanction", "sanctions", "probe", "investigation", "lawsuit", "antitrust",
        "default", "defaults", "defaulted", "bankruptcy", "crisis", "panic", "fear", "freeze",
        "halt", "halts", "halted", "recession", "downgrade", "downgrades", "bearish", "fail",
        "fails", "failed", "grounding", "grounded", "explosion", "loss", "losses", "weak",
        "weakness", "bubble"
    ]

    # 1. Fed / Interest Rates / Inflation / Yields / Central Bank
    rate_terms = ["rate", "rates", "fed", "powell", "inflation", "cpi", "ppi", "yield", "yields", "treasury", "treasuries", "central bank", "basis point", "bps", "bp", "fomc"]
    if has_any_word(rate_terms):
        rate_easing_cues = [
            "cut", "cuts", "cutting", "slash", "slashes", "slashed", "easing", "ease", "dovish",
            "pivot", "relief", "falls", "falling", "cools", "cooling", "slows", "slowing", "lower",
            "drop", "drops", "pause", "pauses", "down", "subside", "soft landing"
        ]
        rate_tightening_cues = [
            "hike", "hikes", "hiking", "increase", "increases", "tightening", "tighten", "hawkish",
            "spikes", "spiking", "surges", "hot", "hotter", "persistent", "sticky", "higher", "accelerate"
        ]
        is_easing = has_any_word(rate_easing_cues)
        is_tightening = has_any_word(rate_tightening_cues)

        if "rate cut" in headline_lower or "cut rate" in headline_lower or "inflation falls" in headline_lower or "inflation cools" in headline_lower or "cpi falls" in headline_lower or "surprise rate cut" in headline_lower:
            is_easing = True
            is_tightening = False
        elif "rate hike" in headline_lower or "inflation spikes" in headline_lower or "hot cpi" in headline_lower or "yields surge" in headline_lower:
            is_tightening = True
            is_easing = False

        # Extract numeric magnitude (e.g. 25 bps, 50 bps, 100 bps, 250 bps, 1000 bps, 1%, 2.5%, etc.)
        rate_mag_multiplier = 1.0
        bps_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bps|basis\s*points?|bp)\b", headline_lower)
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|\bpercent\b)", headline_lower)
        mag_label = ""

        if bps_match:
            bps_val = float(bps_match.group(1))
            # Standard baseline shock is 50 bps (multiplier = 1.0). Scale linearly with soft upper dampener
            rate_mag_multiplier = max(0.2, min(8.0, bps_val / 50.0))
            mag_label = f" ({int(bps_val)} bps)"
        elif pct_match:
            pct_val = float(pct_match.group(1))
            bps_val = pct_val * 100.0
            rate_mag_multiplier = max(0.2, min(8.0, bps_val / 50.0))
            mag_label = f" ({pct_val:g}%)"

        if is_easing:
            shocked_themes["High Growth Tech"] = 0.075 * rate_mag_multiplier
            shocked_themes["Semiconductor Supply Chain"] = 0.07 * rate_mag_multiplier
            shocked_themes["Interest Rate Duration"] = 0.08 * rate_mag_multiplier
            shocked_themes["Consumer Discretionary"] = 0.05 * rate_mag_multiplier
            shocked_themes["AI Infrastructure & Capex"] = 0.06 * rate_mag_multiplier
            shocked_themes["Broad Market Exposure"] = 0.035 * rate_mag_multiplier
            shocked_themes["Banking & Credit"] = -0.015 * rate_mag_multiplier
            rationale.append(f"Rate cuts{mag_label} and cooling inflation reduce the cost of capital and lower bond discount rates, driving valuation multiple expansion across high-growth tech, duration, and consumer equities.")
        else:
            shocked_themes["High Growth Tech"] = -0.08 * rate_mag_multiplier
            shocked_themes["Semiconductor Supply Chain"] = -0.07 * rate_mag_multiplier
            shocked_themes["Interest Rate Duration"] = -0.09 * rate_mag_multiplier
            shocked_themes["Consumer Discretionary"] = -0.05 * rate_mag_multiplier
            shocked_themes["Banking & Credit"] = 0.02 * rate_mag_multiplier
            rationale.append(f"Higher interest rates{mag_label} or persistent inflation compress future valuation multiples and increase bond discount rates across growth and duration assets.")

    # 2. AI / Semiconductor / Tech Hardware / Cloud / Datacenter
    ai_terms = ["ai", "artificial intelligence", "chip", "chips", "semiconductor", "semis", "gpu", "gpus", "capex", "data center", "datacenter", "nvidia", "nvda", "tsmc", "asml", "amd", "blackwell", "cloud", "hyperscaler", "compute"]
    if has_any_word(ai_terms):
        ai_pos = has_any_word(["beat", "beats", "surge", "surges", "record", "boom", "expansion", "breakthrough", "soar", "jump", "rally", "growth", "order", "contract", "demand", "supercycle", "upgrade", "partnership", "all-time high", "revenue surge"])
        ai_neg = has_any_word(["curb", "curbs", "ban", "bans", "restriction", "restrictions", "export", "antitrust", "probe", "pause", "bubble", "slowdown", "delay", "delays", "shortage", "miss", "misses", "slump", "drop", "plunge", "investigation", "halt", "halts", "halted", "attack", "cyberattack", "outage", "disruption", "lawsuit", "defect"])

        if ai_pos and not ai_neg:
            shocked_themes["AI Infrastructure & Capex"] = 0.12
            shocked_themes["Semiconductor Supply Chain"] = 0.14
            shocked_themes["High Growth Tech"] = 0.08
            shocked_themes["Cloud Computing"] = 0.08
            shocked_themes["Nuclear Energy & Power Demand"] = 0.06
            rationale.append("Surging AI datacenter capex, record chip demand, and technological breakthroughs drive strong multiple expansion across hardware, semiconductor, and power infrastructure equities.")
        elif ai_neg:
            shocked_themes["AI Infrastructure & Capex"] = -0.12
            shocked_themes["Semiconductor Supply Chain"] = -0.14
            shocked_themes["High Growth Tech"] = -0.08
            shocked_themes["Cloud Computing"] = -0.08
            rationale.append("Headline signals hardware export curbs, capex scale-backs, or chip demand slowdown, triggering valuation compression across semiconductor and AI infrastructure assets.")

    # 3. Aerospace / Defense / Space / Military
    space_defense_terms = ["spacex", "space x", "starship", "starlink", "space", "aerospace", "defense", "military", "pentagon", "satellite", "boeing", "lockheed", "rtx", "rocket", "launch", "nasa", "orbit", "weapon", "arms", "nato"]
    if has_any_word(space_defense_terms):
        space_pos = has_any_word(["orbit", "success", "successful", "catch", "win", "wins", "won", "award", "awarded", "contract", "budget", "procurement", "soar", "record", "approval", "deal", "expansion", "launch"])
        space_neg = has_any_word(["grounding", "grounded", "explosion", "failure", "fails", "failed", "crash", "delay", "delays", "defect", "probe", "investigation", "cut", "strike"])

        if space_pos and not space_neg:
            shocked_themes["Aerospace & Defense"] = 0.12
            shocked_themes["High Growth Tech"] = 0.03
            rationale.append("Successful space mission milestones, increased defense appropriations, and major government contract awards boost aerospace and defense contractor valuations.")
        elif space_neg:
            shocked_themes["Aerospace & Defense"] = -0.12
            rationale.append("Mission failures, FAA groundings, or defense spending scrutiny create operational headwinds and margin pressure for aerospace and defense holdings.")

    # 4. Energy / Crude Oil / Commodities / OPEC / Middle East
    energy_terms = ["oil", "crude", "gas", "energy", "opec", "petroleum", "middle east", "barrel", "bbl", "drilling", "pipeline"]
    if has_any_word(energy_terms):
        oil_plunge = has_any_word(["plunge", "plunges", "drop", "drops", "falls", "slump", "surplus", "glut", "peace", "deal", "output hike", "cooling"])
        oil_surge = has_any_word(["spike", "spikes", "surge", "surges", "soar", "attack", "embargo", "cut", "cuts", "conflict", "war", "tension"])

        if oil_plunge and not oil_surge:
            shocked_themes["Energy & Commodities"] = -0.10
            shocked_themes["Consumer Discretionary"] = 0.06
            shocked_themes["Broad Market Exposure"] = 0.03
            rationale.append("Falling crude prices provide input margin relief to transport, manufacturing, and consumer discretionary sectors while compressing energy producer cash flows.")
        else:
            shocked_themes["Energy & Commodities"] = 0.12
            shocked_themes["Consumer Discretionary"] = -0.05
            rationale.append("Energy and crude price spikes disproportionately reward energy producers while raising consumer transport costs and retail margin pressure.")

    # 5. Banking / Financials / Credit / CRE
    bank_terms = ["bank", "banks", "banking", "credit", "deposit", "deposits", "cre", "commercial real estate", "loan", "loans", "lending", "financials", "regional bank", "liquidity"]
    if has_any_word(bank_terms):
        bank_neg = has_any_word(["run", "runs", "flight", "default", "defaults", "crisis", "crunch", "contagion", "failure", "panic", "cre", "stress", "bailout"])
        bank_pos = has_any_word(["record", "profit", "nim", "expansion", "beat", "deregulation", "capital", "dividend", "resilient"])

        if bank_neg:
            shocked_themes["Banking & Credit"] = -0.14
            shocked_themes["Broad Market Exposure"] = -0.05
            rationale.append("Credit distress, CRE default concerns, or deposit outflow pressures trigger risk aversion and valuation haircuts across banking and credit sectors.")
        elif bank_pos:
            shocked_themes["Banking & Credit"] = 0.08
            shocked_themes["Broad Market Exposure"] = 0.03
            rationale.append("Strong net interest margins, solid loan performance, and healthy capital ratios support financial institutions and credit stability.")

    # 6. Macro / GDP / Jobs / Recession / Consumer
    macro_terms = ["recession", "gdp", "job", "jobs", "payroll", "unemployment", "layoff", "layoffs", "consumer", "spending", "retail sales", "slowdown"]
    if has_any_word(macro_terms):
        macro_pos = has_any_word(["beat", "beats", "growth", "resilient", "resilience", "boom", "strong", "soft landing", "stimulus"])
        macro_neg = has_any_word(["recession", "layoff", "layoffs", "slowdown", "weakness", "jump in unemployment", "slump", "stagflation"])

        if macro_pos and not macro_neg:
            shocked_themes["Consumer Discretionary"] = 0.06
            shocked_themes["Broad Market Exposure"] = 0.05
            shocked_themes["High Growth Tech"] = 0.04
            rationale.append("Resilient macroeconomic growth, robust consumer spending, and healthy employment numbers support broad market equity multiples.")
        elif macro_neg:
            shocked_themes["Consumer Discretionary"] = -0.08
            shocked_themes["Broad Market Exposure"] = -0.05
            shocked_themes["Defensive Consumer"] = 0.02
            rationale.append("Economic contraction signals and consumer spending weakness compress margins for cyclical businesses while favoring defensive consumer staples.")

    # 7. Tariffs / Trade / Sanctions
    trade_terms = ["tariff", "tariffs", "trade war", "sanction", "sanctions", "embargo", "trade deal"]
    if has_any_word(trade_terms):
        trade_pos = has_any_word(["deal", "pact", "agreement", "removal", "lifted", "roll back", "cut"])
        trade_neg = has_any_word(["tariff", "tariffs", "hike", "war", "sanction", "sanctions", "retaliation"])

        if trade_pos and not trade_neg:
            shocked_themes["Broad Market Exposure"] = 0.04
            shocked_themes["Semiconductor Supply Chain"] = 0.06
            shocked_themes["Consumer Discretionary"] = 0.05
            rationale.append("Easing trade friction, bilateral deals, and tariff rollbacks restore global supply chain fluidity and lower import costs.")
        else:
            shocked_themes["Broad Market Exposure"] = -0.04
            shocked_themes["Semiconductor Supply Chain"] = -0.08
            shocked_themes["Consumer Discretionary"] = -0.05
            rationale.append("Escalating tariffs and trade barriers increase input costs and disrupt international supply chains across manufacturing, chips, and retail.")

    # 8. Direct Company / Ticker Mentions
    for name_or_sym, ticker in COMPANY_SYNONYMS.items():
        if re.search(r"\b" + re.escape(name_or_sym) + r"\b", headline_lower):
            is_pos = has_any_word(positive_cues)
            is_neg = has_any_word(negative_cues)
            company_shock = 0.12 if (is_pos and not is_neg) else (-0.12 if is_neg else (0.05 if is_pos else -0.05))
            direct_asset_shocks[ticker] = company_shock
            meta_name = ASSET_METADATA.get(ticker, {}).get("name", ticker)
            action_desc = "catalyst upside and multiple expansion" if company_shock > 0 else "operational headwinds and valuation compression"
            rationale.append(f"Headline directly spotlights {meta_name} ({ticker}), driving asset-specific {action_desc}.")

    # Fallback for arbitrary / generic headlines
    if not shocked_themes and not direct_asset_shocks:
        is_pos = has_any_word(positive_cues)
        is_neg = has_any_word(negative_cues)
        if is_pos and not is_neg:
            shocked_themes["Broad Market Exposure"] = 0.04
            shocked_themes["High Growth Tech"] = 0.03
            shocked_themes["Consumer Discretionary"] = 0.03
            rationale.append("Headline conveys broad macroeconomic optimism and positive risk sentiment across equity markets.")
        elif is_neg and not is_pos:
            shocked_themes["Broad Market Exposure"] = -0.04
            shocked_themes["High Growth Tech"] = -0.04
            rationale.append("Headline triggers heightened risk aversion and downward equity repricing across broad market benchmarks.")
        else:
            shocked_themes["Broad Market Exposure"] = -0.02
            rationale.append("Headline reflects general market volatility with modest cross-asset sensitivity.")

    # Calculate portfolio sensitivity
    affected_assets = []
    total_dollar_impact = 0.0
    total_val = portfolio.total_value

    avg_theme_shock = (sum(shocked_themes.values()) / len(shocked_themes)) if shocked_themes else 0.0

    for exp in exposures:
        matching_shocks = [shocked_themes[t] for t in exp.themes if t in shocked_themes]
        direct_shock = direct_asset_shocks.get(exp.ticker)

        # If exposure is an ETF, compute lookthrough sensitivity from directly shocked constituents
        if direct_shock is None and exp.ticker in ETF_LOOKTHROUGH and direct_asset_shocks:
            etf_weights = ETF_LOOKTHROUGH[exp.ticker]
            etf_shock = sum(etf_weights.get(sym, 0.0) * direct_asset_shocks[sym] for sym in direct_asset_shocks if sym in etf_weights)
            if abs(etf_shock) > 0.0001:
                direct_shock = etf_shock

        if direct_shock is not None:
            if matching_shocks:
                asset_shock = 0.7 * direct_shock + 0.3 * (sum(matching_shocks) / len(matching_shocks))
            else:
                asset_shock = direct_shock
        elif matching_shocks:
            asset_shock = sum(matching_shocks) / len(matching_shocks)
        elif avg_theme_shock != 0.0 and exp.ticker not in ["CASH", "BND", "TLT"]:
            # Baseline market equity beta sensitivity for macro shocks
            asset_shock = avg_theme_shock * 0.40
        else:
            asset_shock = 0.0

        if abs(asset_shock) > 0.001:
            dollar_impact = exp.total_dollar_value * asset_shock
            total_dollar_impact += dollar_impact
            affected_assets.append({
                "ticker": exp.ticker,
                "name": exp.name,
                "portfolio_weight": round(exp.total_weight * 100.0, 2),
                "estimated_shock_pct": round(asset_shock * 100.0, 2),
                "estimated_dollar_impact": round(dollar_impact, 2),
                "direction": "Negative" if asset_shock < 0 else "Positive",
            })

    affected_assets.sort(key=lambda a: abs(a["estimated_dollar_impact"]), reverse=True)
    net_shock_pct = (total_dollar_impact / total_val * 100.0) if total_val > 0 else 0.0

    return {
        "headline": headline,
        "net_estimated_impact_pct": round(net_shock_pct, 2),
        "net_estimated_dollar_impact": round(total_dollar_impact, 2),
        "rationale": " ".join(rationale) if rationale else "Analysis of external macro sensitivity.",
        "affected_holdings": affected_assets[:6],
    }
