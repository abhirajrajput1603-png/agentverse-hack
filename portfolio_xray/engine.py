"""Core quantitative risk engine for Portfolio X-Ray.

Pure, deterministic financial calculations using standard library only.
Testable without models or external data providers.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from portfolio_xray.data import (
    ETF_LOOKTHROUGH,
    ASSET_METADATA,
    get_asset_info,
    STRESS_SCENARIOS,
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
    """Decomposes all direct and ETF holdings to reveal true underlying company weights."""
    total_val = portfolio.total_value
    if total_val <= 0:
        return []

    # Map holding weights
    holding_weights: Dict[str, float] = {}
    for h in portfolio.holdings:
        t = h.ticker.upper().strip()
        holding_weights[t] = holding_weights.get(t, 0.0) + (h.market_value / total_val)

    # Accumulate exposures
    direct_weights: Dict[str, float] = {}
    indirect_weights: Dict[str, float] = {}
    contributions: Dict[str, Dict[str, float]] = {}  # target -> {etf: weight_contribution}

    for ticker, weight in holding_weights.items():
        if ticker in ETF_LOOKTHROUGH:
            # Distribute across ETF basket
            for constituent, constituent_weight in ETF_LOOKTHROUGH[ticker].items():
                effective_contribution = weight * constituent_weight
                indirect_weights[constituent] = indirect_weights.get(constituent, 0.0) + effective_contribution
                if constituent not in contributions:
                    contributions[constituent] = {}
                contributions[constituent][ticker] = effective_contribution
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


def compute_thematic_exposure(exposures: List[EffectiveExposure]) -> List[Dict[str, Any]]:
    """Sums effective exposures across macroeconomic, industry, and thematic drivers."""
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
            if "US_TREASURY" in exposures[i].ticker or "US_TREASURY" in exposures[j].ticker:
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


def run_stress_tests(portfolio: Portfolio, exposures: List[EffectiveExposure]) -> List[Dict[str, Any]]:
    """Evaluates portfolio drawdown impact across historical and macro shock scenarios."""
    total_val = portfolio.total_value
    results = []

    for scenario in STRESS_SCENARIOS:
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
            elif "US_TREASURY" in exp.ticker:
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
            "description": scenario["description"],
            "drawdown_pct": round(portfolio_shock * 100.0, 2),
            "dollar_drawdown": round(dollar_drawdown, 2),
            "worst_hit_assets": sorted(asset_impacts, key=lambda a: a["dollar_impact"])[:3],
            "resilient_assets": sorted(asset_impacts, key=lambda a: a["dollar_impact"], reverse=True)[:2],
        })

    return results


def simulate_what_if(
    current_portfolio: Portfolio,
    target_allocations: Dict[str, float],  # Ticker -> Target % of portfolio (0 to 100)
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

    # Normalize target allocations
    total_target = sum(target_allocations.values())
    if total_target <= 0:
        norm_targets = {t: 100.0 / len(target_allocations) for t in target_allocations}
    else:
        norm_targets = {t: (pct / total_target) * 100.0 for t, pct in target_allocations.items()}

    for ticker, pct in norm_targets.items():
        t = ticker.upper().strip()
        allocated_dollars = (pct / 100.0) * total_val
        price = known_prices.get(t, default_prices.get(t, 100.0))
        shares = allocated_dollars / price
        proposed_holdings.append(Holding(ticker=t, shares=shares, price=price))

    proposed_portfolio = Portfolio(holdings=proposed_holdings, user_profile=current_portfolio.user_profile)

    # Compute both states
    cur_exp = compute_lookthrough_exposure(current_portfolio)
    prop_exp = compute_lookthrough_exposure(proposed_portfolio)

    cur_metrics = compute_risk_metrics(current_portfolio, cur_exp)
    prop_metrics = compute_risk_metrics(proposed_portfolio, prop_exp)

    cur_stress = run_stress_tests(current_portfolio, cur_exp)
    prop_stress = run_stress_tests(proposed_portfolio, prop_exp)

    # Risk Profile Alignment Evaluation
    tolerance = current_portfolio.user_profile.get("risk_tolerance", "Moderate")
    max_safe_vol = {"Conservative": 12.0, "Moderate": 22.0, "Aggressive": 35.0}.get(tolerance, 22.0)
    max_safe_top1 = {"Conservative": 15.0, "Moderate": 25.0, "Aggressive": 40.0}.get(tolerance, 25.0)

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
    }


def analyze_headline_event(
    portfolio: Portfolio,
    headline: str,
) -> Dict[str, Any]:
    """Analyzes how an external market headline or macro event impacts the portfolio's specific exposures."""
    exposures = compute_lookthrough_exposure(portfolio)
    headline_lower = headline.lower()

    # Identify relevant economic shock themes
    shocked_themes: Dict[str, float] = {}
    rationale = []

    if any(k in headline_lower for k in ["rate", "fed", "powell", "inflation", "yield", "treasury", "central bank", "basis point", "hike"]):
        shocked_themes["High Growth Tech"] = -0.08
        shocked_themes["Semiconductor Supply Chain"] = -0.07
        shocked_themes["Interest Rate Duration"] = -0.09
        shocked_themes["Banking & Credit"] = 0.02
        rationale.append("Higher interest rates or persistent inflation compress future valuation multiples and increase bond discount rates.")

    if any(k in headline_lower for k in ["ai", "capex", "data center", "nvidia", "gpu", "chip", "semiconductor", "tsmc", "asml", "export"]):
        shocked_themes["AI Infrastructure & Capex"] = -0.12
        shocked_themes["Semiconductor Supply Chain"] = -0.14
        shocked_themes["Nuclear Energy & Power Demand"] = -0.08
        rationale.append("Headline directly targets semiconductor hardware, AI infrastructure spending, or supply chain bottlenecks.")

    if any(k in headline_lower for k in ["oil", "gas", "energy", "middle east", "opec", "crude"]):
        shocked_themes["Energy & Commodities"] = 0.12
        shocked_themes["Consumer Discretionary"] = -0.05
        rationale.append("Energy and crude price shocks disproportionately reward energy producers while raising consumer input costs.")

    if any(k in headline_lower for k in ["recession", "gdp", "job", "layoff", "slowdown", "weakness"]):
        shocked_themes["Consumer Discretionary"] = -0.08
        shocked_themes["Broad Market Exposure"] = -0.05
        shocked_themes["Defensive Consumer"] = 0.02
        rationale.append("Economic slowdown signals impact cyclical and consumer discretionary businesses while favoring defensive staples.")

    if not shocked_themes:
        # Default mild broad equity sensitivity
        shocked_themes["Broad Market Exposure"] = -0.03
        rationale.append("Headline reflects general market volatility with modest cross-asset sensitivity.")

    # Calculate portfolio sensitivity
    affected_assets = []
    total_dollar_impact = 0.0
    total_val = portfolio.total_value

    for exp in exposures:
        matching_shocks = [shocked_themes[t] for t in exp.themes if t in shocked_themes]
        if matching_shocks:
            asset_shock = sum(matching_shocks) / len(matching_shocks)
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
        "rationale": " ".join(rationale),
        "affected_holdings": affected_assets[:6],
    }
