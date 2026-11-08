from __future__ import annotations

import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure both project root and v-ui are importable
ROOT_DIR = Path(__file__).resolve().parent.parent
V_UI_DIR = ROOT_DIR / "v-ui"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(V_UI_DIR) not in sys.path:
    sys.path.insert(0, str(V_UI_DIR))

from xray.data_loader import DataLoader
from xray.engine.benchmark import BenchmarkEngine
from xray.engine.concentration import ConcentrationEngine
from xray.engine.correlation import CorrelationAndRiskContributionEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.risk import RiskEngine, VALID_COV_METHODS
from xray.engine.stress import StressAndTailEngine
from xray.engine.themes import ThemeEngine
from xray.engine.whatif import WhatIfSimulator
from xray.evidence import Evidence, EvidenceRegistry
from xray.profile import UserRiskProfile
from xray.safety.audit import AuditLogger
from xray.safety.limits import SafetyLimitsEngine

import data as vui_data
import engine as vui_engine


# Singleton shared DataLoader to avoid re-reading 15Y price matrix on every request
_SHARED_LOADER: Optional[DataLoader] = None


def get_shared_loader() -> DataLoader:
    global _SHARED_LOADER
    if _SHARED_LOADER is None:
        _SHARED_LOADER = DataLoader()
    return _SHARED_LOADER


def map_vui_profile_to_xray(user_profile: Dict[str, Any]) -> UserRiskProfile:
    """
    Maps a v-ui user_profile dict (from the 4-question Risk Profile Questionnaire or preset)
    into an authoritative xray.profile.UserRiskProfile instance with deterministic thresholds.
    """
    raw_tol = str(user_profile.get("risk_tolerance", user_profile.get("title", "Moderate"))).strip()
    raw_horizon = user_profile.get("horizon_years", user_profile.get("horizon", 7.0))
    if isinstance(raw_horizon, (int, float)):
        horizon = float(raw_horizon)
    else:
        h_str = str(raw_horizon)
        if "within 1" in h_str.lower():
            horizon = 1.0
        elif "1" in h_str and "3" in h_str:
            horizon = 2.0
        elif "3" in h_str and "5" in h_str:
            horizon = 4.0
        elif "5" in h_str and "10" in h_str:
            horizon = 7.0
        else:
            try:
                horizon = float(h_str.split()[0])
            except (ValueError, IndexError):
                horizon = 7.0
    capacity_str = str(user_profile.get("loss_capacity", user_profile.get("capacity", "10–20% max")))

    # Map risk tolerance to single-stock limit and max tolerable drawdown
    tol_lower = raw_tol.lower()
    if "very low" in tol_lower or "conservative" in tol_lower or "under 10%" in capacity_str.lower():
        default_single_limit = 0.10
        max_loss_pct = 0.10
        exp_level = "Beginner"
    elif "low" in tol_lower:
        default_single_limit = 0.15
        max_loss_pct = 0.15
        exp_level = "Beginner"
    elif "very high" in tol_lower or "aggressive" in tol_lower or "over 30%" in capacity_str.lower():
        default_single_limit = 0.30
        max_loss_pct = 0.35
        exp_level = "Advanced"
    elif "high" in tol_lower or "growth focus" in tol_lower or "20–30%" in capacity_str.lower():
        default_single_limit = 0.25
        max_loss_pct = 0.25
        exp_level = "Intermediate"
    else:
        default_single_limit = 0.20
        max_loss_pct = 0.20
        exp_level = "Intermediate"

    explicit_limit = user_profile.get("max_single_stock_limit")
    if explicit_limit is not None:
        try:
            val = float(explicit_limit)
            single_limit = val / 100.0 if val > 1.0 else val
        except (ValueError, TypeError):
            single_limit = default_single_limit
    else:
        single_limit = default_single_limit

    return UserRiskProfile(
        name=str(user_profile.get("name", raw_tol or "Everyday Investor")),
        horizon_years=horizon,
        max_tolerable_loss_pct=max_loss_pct,
        max_single_stock_limit=round(single_limit, 4),
        objective=str(user_profile.get("goal", "Long-term wealth building & risk awareness")),
        experience_level=exp_level,
        financial_milestone=user_profile.get("financial_milestone"),
    )


def convert_vui_portfolio_to_xray_holdings(portfolio: vui_engine.Portfolio) -> List[Dict[str, Any]]:
    """
    Converts a v-ui Portfolio (list of Holding(ticker, shares, price)) into
    normalized xray holdings [{'ticker': ..., 'name': ..., 'allocation_gbp': ..., 'weight': ...}]
    whose weights strictly sum to 1.0.
    """
    total_val = portfolio.total_value
    if total_val <= 0 or not portfolio.holdings:
        return [{"ticker": "CASH", "name": "USD Cash", "allocation_gbp": 10000.0, "weight": 1.0}]

    agg_values: Dict[str, float] = {}
    for h in portfolio.holdings:
        t = h.ticker.upper().strip()
        agg_values[t] = agg_values.get(t, 0.0) + h.market_value

    raw_holdings: List[Dict[str, Any]] = []
    for t, val in agg_values.items():
        info = vui_data.get_asset_info(t)
        raw_holdings.append({
            "ticker": t,
            "name": info.get("name", t),
            "allocation_gbp": round(val, 2),
            "weight": val / total_val,
        })

    # Ensure exact weight sum conservation
    w_sum = sum(h["weight"] for h in raw_holdings)
    if w_sum > 0:
        for h in raw_holdings:
            h["weight"] = h["weight"] / w_sum

    return raw_holdings


def build_full_lookthrough_list(
    lt_result: Any,
    total_val: float,
) -> List[Dict[str, Any]]:
    """
    Formats xray.engine.lookthrough.LookThroughResult into the exact frontend schema
    expected by v-ui (`lookthrough_exposure` & `exposures`), while preserving
    100% constituent decomposition (sum of weights == 1.0).
    """
    out: List[Dict[str, Any]] = []
    for exp in lt_result.sorted_exposures(exclude_residual_and_cash=False):
        t = exp.issuer
        info = vui_data.get_asset_info(t)
        contrib_sources = {
            src: round(w * 100.0, 2)
            for src, w in exp.breakdown_by_source.items()
            if src != "DIRECT"
        }
        out.append({
            "ticker": t,
            "company_name": info.get("name", t),
            "name": info.get("name", t),
            "lookthrough_weight": round(exp.total_weight, 6),
            "total_weight": round(exp.total_weight * 100.0, 2),
            "effective_value": round(exp.total_weight * total_val, 2),
            "total_dollar_value": round(exp.total_weight * total_val, 2),
            "direct_weight": round(exp.direct_weight, 6),
            "indirect_weight": round(exp.indirect_weight, 6),
            "direct_weight_pct": round(exp.direct_weight * 100.0, 2),
            "indirect_weight_pct": round(exp.indirect_weight * 100.0, 2),
            "contributing_sources": contrib_sources,
            "sector": info.get("sector", "Diversified"),
            "themes": info.get("themes", []),
            "annual_volatility": round(info.get("annual_volatility", 0.22) * 100.0, 1),
            "is_residual": exp.is_residual,
            "is_cash": exp.is_cash,
            "evidence_id": f"EV-LT-{t}",
        })
    return out


def run_unified_analysis(
    portfolio: vui_engine.Portfolio,
    cov_method: str = "sample",
    lookthrough_mode: str = "overlap",
    user_max_single_stock_limit: Optional[float] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> Dict[str, Any]:
    """
    Executes both Dhillon's v-ui engine and our full institutional xray/ engines,
    returning a unified JSON-serializable payload with traceable Evidence objects (EV-*).
    """
    loader = get_shared_loader()
    audit = audit_logger or AuditLogger()
    registry = EvidenceRegistry()

    cov_clean = cov_method.lower().strip() if cov_method else "sample"
    if cov_clean not in VALID_COV_METHODS:
        cov_clean = "sample"

    # 1. Run v-ui overlap-focused engines
    vui_exposures = vui_engine.compute_lookthrough_exposure(portfolio)
    vui_thematic = vui_engine.compute_thematic_exposure(vui_exposures)
    vui_metrics = vui_engine.compute_risk_metrics(portfolio, vui_exposures)
    vui_stress = vui_engine.run_stress_tests(portfolio, vui_exposures)

    # 2. Convert to xray holdings & profile
    xray_holdings = convert_vui_portfolio_to_xray_holdings(portfolio)
    xray_profile = map_vui_profile_to_xray(portfolio.user_profile or {})
    if user_max_single_stock_limit is not None:
        val = float(user_max_single_stock_limit)
        xray_profile.max_single_stock_limit = val / 100.0 if val > 1.0 else val

    total_val = portfolio.total_value or 32500.0

    # 3. Institutional Look-Through & Concentration
    lt_eng = LookThroughEngine(loader)
    conc_eng = ConcentrationEngine()
    lt_res = lt_eng.analyze(xray_holdings)
    conc_res = conc_eng.analyze(lt_res)
    for ev in lt_res.evidence_list + conc_res.evidence_list:
        registry.register(ev)

    # 4. Non-Additive Economic Themes (data/themes.json) + merged thematic factors
    theme_eng = ThemeEngine(loader)
    theme_res = theme_eng.analyze(lt_res)
    for ev in theme_res.evidence_list:
        registry.register(ev)

    # 5. Real 15-Year Historical Price Risk Engine + Covariance Comparison
    risk_eng = RiskEngine(loader)
    risk_res = risk_eng.analyze(xray_holdings, cov_method=cov_clean)
    cov_cmp = risk_eng.compare_covariance_methods(xray_holdings)
    for ev in risk_res.evidence_list + cov_cmp.evidence_list:
        registry.register(ev)

    # Blend real historical volatility with parametric fallback if any custom unpriced tickers exist
    priced_cov = risk_res.priced_weight_coverage
    if priced_cov >= 0.999:
        effective_ann_vol_dec = risk_res.annualized_volatility
    elif priced_cov <= 0.001:
        effective_ann_vol_dec = vui_metrics.get("annual_volatility", 20.0) / 100.0
    else:
        unpriced_vol_dec = vui_metrics.get("annual_volatility", 20.0) / 100.0
        effective_ann_vol_dec = (
            priced_cov * risk_res.annualized_volatility
            + (1.0 - priced_cov) * unpriced_vol_dec
        )

    # 6. Historical Tail Risk (95%/99% VaR & ES + 1,000-sample Bootstrap CIs) & 4 Crisis Replays
    stress_eng = StressAndTailEngine(loader)
    stress_res = stress_eng.analyze(xray_holdings, n_bootstrap=1000)
    for ev in stress_res.evidence_list:
        registry.register(ev)

    # 7. Correlation Matrix (1Y/3Y) & Euler Risk Decomposition
    corr_eng = CorrelationAndRiskContributionEngine(loader)
    corr_res = corr_eng.analyze(xray_holdings, cov_method=cov_clean)
    for ev in corr_res.evidence_list:
        registry.register(ev)

    # 8. Benchmark Comparison (vs SPY and 60/40 SPY/BND)
    bench_eng = BenchmarkEngine(loader)
    bench_res = bench_eng.analyze(xray_holdings)
    for ev in bench_res.evidence_list:
        registry.register(ev)

    # 9. Hard Safety Circuit Breakers
    safety_eng = SafetyLimitsEngine(audit_logger=audit)
    safety_res = safety_eng.evaluate_portfolio(
        lt_result=lt_res,
        concentration=conc_res,
        user_max_single_stock_limit=xray_profile.max_single_stock_limit,
        persona_name=xray_profile.name,
    )
    for ev in safety_res.evidence_list:
        registry.register(ev)

    # Additional Volatility Regime & Data Integrity checks
    cb_evaluations = [
        {
            "status": e.status,
            "rule_name": e.rule_name,
            "threshold": e.threshold,
            "actual_value": e.actual_value,
            "excess": e.excess,
            "breaching_issuer": e.breaching_issuer,
            "message": e.message,
            "evidence_id": e.evidence_id,
            "timestamp": e.timestamp,
        }
        for e in safety_res.evaluations
    ]

    # Volatility Regime Alert check against user profile max tolerable loss
    vol_threshold = max(0.18, xray_profile.max_tolerable_loss_pct * 1.25)
    vol_status = "BREACH" if effective_ann_vol_dec > vol_threshold else "PASS"
    cb_evaluations.append({
        "status": vol_status,
        "rule_name": "VOLATILITY_REGIME_ALERT",
        "threshold": round(vol_threshold, 4),
        "actual_value": round(effective_ann_vol_dec, 4),
        "excess": round(effective_ann_vol_dec - vol_threshold, 4),
        "breaching_issuer": "PORTFOLIO",
        "message": (
            f"BREACH: Annualized volatility ({effective_ann_vol_dec*100:.1f}%) exceeds "
            f"profile volatility ceiling ({vol_threshold*100:.1f}%)."
            if vol_status == "BREACH"
            else f"PASS: Annualized volatility ({effective_ann_vol_dec*100:.1f}%) is within "
                 f"profile ceiling ({vol_threshold*100:.1f}%)."
        ),
        "evidence_id": "EV-RISK-ANN-VOL",
        "timestamp": cb_evaluations[0]["timestamp"] if cb_evaluations else "",
    })

    # Data Integrity / Coverage Check
    data_status = "PASS" if priced_cov >= 0.50 else "WARNING"
    cb_evaluations.append({
        "status": data_status,
        "rule_name": "DATA_INTEGRITY_COVERAGE",
        "threshold": 0.50,
        "actual_value": round(priced_cov, 4),
        "excess": round(priced_cov - 0.50, 4),
        "breaching_issuer": ", ".join(risk_res.unpriced_holdings) if risk_res.unpriced_holdings else None,
        "message": (
            f"PASS: {priced_cov*100:.1f}% of portfolio weight is backed by 15-year daily market price series."
            if data_status == "PASS"
            else f"WARNING: Unpriced custom tickers {risk_res.unpriced_holdings} use parametric volatility proxy."
        ),
        "evidence_id": "EV-RISK-ANN-VOL",
        "timestamp": cb_evaluations[0]["timestamp"] if cb_evaluations else "",
    })

    # Build both Overlap-Only and Full Look-Through lists
    overlap_lt_list = [
        {
            "ticker": e.ticker,
            "company_name": e.name,
            "name": e.name,
            "lookthrough_weight": round(e.total_weight, 6),
            "total_weight": round(e.total_weight * 100.0, 2),
            "effective_value": round(e.total_dollar_value, 2),
            "total_dollar_value": round(e.total_dollar_value, 2),
            "direct_weight": round(e.direct_weight, 6),
            "indirect_weight": round(e.indirect_weight, 6),
            "direct_weight_pct": round(e.direct_weight * 100.0, 2),
            "indirect_weight_pct": round(e.indirect_weight * 100.0, 2),
            "contributing_sources": {k: round(v * 100.0, 2) for k, v in e.contributing_sources.items()},
            "sector": e.sector,
            "themes": e.themes,
            "annual_volatility": round(e.annual_volatility * 100.0, 1),
            "is_residual": False,
            "is_cash": e.ticker in ("CASH", "CASH_GBP"),
            "evidence_id": f"EV-LT-{e.ticker}",
        }
        for e in vui_exposures
    ]
    full_lt_list = build_full_lookthrough_list(lt_res, total_val)

    active_lt_list = full_lt_list if lookthrough_mode == "full" else overlap_lt_list

    # Merge institutional themes (`data/themes.json`) with v-ui thematic factors
    merged_thematic_factors: List[Dict[str, Any]] = []
    seen_theme_names = set()
    for t in theme_res.themes:
        if t.total_weight >= 0.05:
            seen_theme_names.add(t.name.lower())
            merged_thematic_factors.append({
                "theme_id": t.theme_id,
                "theme": t.name,
                "portfolio_exposure": round(t.total_weight, 4),
                "description": t.description,
                "contributing_assets": [
                    f"{c.issuer} ({c.lookthrough_weight*100:.1f}%)" for c in t.contributing_issuers[:6]
                ],
                "evidence_id": f"EV-THEME-{t.theme_id}",
            })
    for vt in vui_thematic:
        t_name = vt.get("theme", "")
        if t_name.lower() not in seen_theme_names and vt.get("weight", 0.0) >= 0.05:
            seen_theme_names.add(t_name.lower())
            merged_thematic_factors.append({
                "theme_id": f"VUI_{t_name.upper().replace(' ', '_')}",
                "theme": t_name,
                "portfolio_exposure": round(vt.get("weight", 0.0), 4),
                "description": f"Shared economic sensitivity across {t_name} holdings.",
                "contributing_assets": [
                    f"{a['ticker']} ({a['weight']*100:.1f}%)" for a in vt.get("top_assets", [])
                ],
                "evidence_id": "EV-CONC-TOP1-WEIGHT",
            })
    merged_thematic_factors.sort(key=lambda x: x["portfolio_exposure"], reverse=True)

    # Format 4 Historical Crisis Replays
    historical_replays = []
    for scen_id, sc in stress_res.scenarios.items():
        port_ret = sc.portfolio_cumulative_return
        port_dd = sc.portfolio_max_drawdown
        # Use cumulative return or max drawdown (whichever is larger magnitude drop) for crisis impact
        effective_drop = min(port_ret, port_dd)
        historical_replays.append({
            "scenario_id": scen_id,
            "name": sc.name,
            "start_date": sc.start_date,
            "end_date": sc.end_date,
            "description": sc.description,
            "portfolio_cumulative_return_pct": round(port_ret * 100.0, 2),
            "portfolio_max_drawdown_pct": round(port_dd * 100.0, 2),
            "portfolio_dollar_loss": round(abs(effective_drop) * total_val, 2),
            "worst_single_day_pct": round(sc.worst_single_day_return * 100.0, 2),
            "spy_cumulative_return_pct": round(sc.spy_cumulative_return * 100.0, 2),
            "relative_vs_spy_pct": round((port_ret - sc.spy_cumulative_return) * 100.0, 2),
            "holding_returns_pct": {k: round(v * 100.0, 2) for k, v in sc.holding_returns.items()},
            "coverage_pct": round(sc.coverage_weight * 100.0, 1),
            "missing_history_holdings": sc.missing_history_holdings,
            "sample_days": sc.sample_days,
            "evidence_id": f"EV-STRESS-{scen_id}",
        })

    # Format v-ui stress_tests while enriching with historical replay data
    combined_stress_tests = [
        {
            "scenario_name": s.get("name", ""),
            "scenario_id": s.get("scenario_id", ""),
            "category_badge": s.get("category_badge", "Factor Stress"),
            "drawdown_pct": s.get("drawdown_pct", 0.0) / 100.0,
            "dollar_loss": abs(s.get("dollar_drawdown", 0.0)),
            "description": s.get("description", ""),
            "worst_hit": s.get("worst_hit_assets", []),
            "source_type": "dynamic_factor",
        }
        for s in vui_stress
    ]

    # Direct weights map
    direct_weights = {h["ticker"]: round(h["weight"], 6) for h in xray_holdings}

    # Format Tail Risk & Bootstrap CIs (both daily and 1-month scaled for consumer & institutional display)
    tr = stress_res.tail_risk
    sqrt_21 = math.sqrt(21.0)
    daily_var_95_abs = abs(tr.var_95) if tr.total_sample_days > 0 else (vui_metrics.get("var_95_1m_pct", 8.0) / 100.0 / sqrt_21)
    daily_es_95_abs = abs(tr.es_95) if tr.total_sample_days > 0 else (vui_metrics.get("expected_shortfall_95_pct", 10.0) / 100.0 / sqrt_21)
    monthly_var_95_dec = min(0.95, daily_var_95_abs * sqrt_21) if priced_cov >= 0.5 else (vui_metrics.get("var_95_1m_pct", 8.0) / 100.0)
    monthly_cvar_95_dec = min(0.99, daily_es_95_abs * sqrt_21) if priced_cov >= 0.5 else (vui_metrics.get("expected_shortfall_95_pct", 10.0) / 100.0)

    # Top single stock lookthrough weight (excluding residual & cash)
    top_named_lt = conc_res.lookthrough.top_issuers[0] if conc_res.lookthrough.top_issuers else {"issuer": "NONE", "weight": 0.0}
    hhi_10000 = round(conc_res.lookthrough.hhi * 10000.0, 1) if conc_res.lookthrough.hhi > 0 else vui_metrics.get("hhi", 2000.0)

    # Update metrics dict for backward compatibility with v-ui
    enriched_vui_metrics = dict(vui_metrics)
    enriched_vui_metrics["annual_volatility"] = round(effective_ann_vol_dec * 100.0, 2)
    enriched_vui_metrics["var_95_1m_pct"] = round(monthly_var_95_dec * 100.0, 2)
    enriched_vui_metrics["var_95_1m_dollar"] = round(monthly_var_95_dec * total_val, 2)
    enriched_vui_metrics["expected_shortfall_95_pct"] = round(monthly_cvar_95_dec * 100.0, 2)
    enriched_vui_metrics["expected_shortfall_95_dollar"] = round(monthly_cvar_95_dec * total_val, 2)
    enriched_vui_metrics["max_drawdown_3y_pct"] = round(risk_res.max_drawdown * 100.0, 2)
    enriched_vui_metrics["effective_n"] = conc_res.lookthrough.effective_n
    enriched_vui_metrics["direct_effective_n"] = conc_res.direct.effective_n

    evidences_dict = {ev.id: ev.to_dict() for ev in registry.all()}

    return {
        "portfolio_value": round(total_val, 2),
        "lookthrough_mode": lookthrough_mode,
        "covariance_method": cov_clean,
        "metrics": enriched_vui_metrics,
        "risk_metrics": {
            "annualized_volatility": round(effective_ann_vol_dec, 6),
            "daily_volatility": round(risk_res.daily_volatility, 6),
            "var_95_1m": round(monthly_var_95_dec, 6),
            "cvar_95_1m": round(monthly_cvar_95_dec, 6),
            "daily_var_95": round(tr.var_95, 6),
            "daily_var_95_ci_low": round(tr.var_95_ci_low, 6),
            "daily_var_95_ci_high": round(tr.var_95_ci_high, 6),
            "daily_es_95": round(tr.es_95, 6),
            "daily_es_95_ci_low": round(tr.es_95_ci_low, 6),
            "daily_es_95_ci_high": round(tr.es_95_ci_high, 6),
            "daily_var_99": round(tr.var_99, 6),
            "daily_es_99": round(tr.es_99, 6),
            "max_drawdown_3y": round(risk_res.max_drawdown, 6),
            "max_drawdown_peak_date": risk_res.max_drawdown_peak_date,
            "max_drawdown_trough_date": risk_res.max_drawdown_trough_date,
            "concentration_hhi": vui_metrics.get("hhi", hhi_10000),
            "lookthrough_hhi_0_1": conc_res.lookthrough.hhi,
            "lookthrough_hhi_10k": hhi_10000,
            "direct_effective_n": conc_res.direct.effective_n,
            "lookthrough_effective_n": conc_res.lookthrough.effective_n,
            "top_1_issuer": top_named_lt["issuer"],
            "top_1_weight": top_named_lt["weight"],
            "top_3_weight": conc_res.lookthrough.top_3_weight,
            "top_5_weight": conc_res.lookthrough.top_5_weight,
            "sample_size_days": risk_res.sample_size_days,
            "priced_weight_coverage": risk_res.priced_weight_coverage,
            "unpriced_holdings": risk_res.unpriced_holdings,
            "covariance_comparison": {
                "active_method": cov_clean,
                "sample_annualized_vol": round(cov_cmp.sample_annualized_vol, 6),
                "ledoit_wolf_annualized_vol": round(cov_cmp.ledoit_wolf_annualized_vol, 6),
                "ewma_annualized_vol": round(cov_cmp.ewma_annualized_vol, 6),
                "ledoit_wolf_shrinkage_alpha": round(cov_cmp.ledoit_wolf_shrinkage_alpha, 4),
                "ewma_lambda": cov_cmp.ewma_lambda,
            },
        },
        "holdings": [
            {
                "ticker": h.ticker,
                "name": vui_data.ASSET_METADATA.get(h.ticker, {}).get("name", h.ticker),
                "shares": h.shares,
                "price": h.price,
                "value": round(h.shares * h.price, 2),
                "weight": round(direct_weights.get(h.ticker, 0.0), 6),
            }
            for h in portfolio.holdings
        ],
        "direct_weights": direct_weights,
        "exposures": active_lt_list,
        "lookthrough_exposure": active_lt_list,
        "overlap_lookthrough_exposure": overlap_lt_list,
        "full_lookthrough_exposure": full_lt_list,
        "full_lookthrough_weight_sum": round(sum(x["lookthrough_weight"] for x in full_lt_list), 6),
        "thematic": vui_thematic,
        "thematic_factors": merged_thematic_factors,
        "stress_tests": combined_stress_tests,
        "historical_crisis_replays": historical_replays,
        "correlation_and_risk_contribution": {
            "portfolio_volatility": round(corr_res.portfolio_volatility, 6),
            "ccr_sum": round(corr_res.ccr_sum, 6),
            "ccr_sum_check_passed": corr_res.ccr_sum_check_passed,
            "avg_pairwise_corr_1y": corr_res.avg_pairwise_corr_1y,
            "avg_pairwise_corr_3y": corr_res.avg_pairwise_corr_3y,
            "highest_corr_pair_3y": (
                {
                    "ticker_a": corr_res.highest_corr_pair_3y[0],
                    "ticker_b": corr_res.highest_corr_pair_3y[1],
                    "correlation": corr_res.highest_corr_pair_3y[2],
                }
                if corr_res.highest_corr_pair_3y
                else None
            ),
            "corr_matrix_1y": corr_res.corr_matrix_1y,
            "corr_matrix_3y": corr_res.corr_matrix_3y,
            "contributions": [
                {
                    "ticker": c.ticker,
                    "name": vui_data.get_asset_info(c.ticker).get("name", c.ticker),
                    "capital_weight": round(c.capital_weight, 6),
                    "capital_weight_pct": round(c.capital_weight * 100.0, 2),
                    "mcr": round(c.mcr, 6),
                    "ccr": round(c.ccr, 6),
                    "pct_risk_contribution": round(c.pct_risk_contribution, 6),
                    "pct_risk_contribution_pct": round(c.pct_risk_contribution * 100.0, 2),
                    "risk_to_capital_ratio": round(c.risk_to_capital_ratio, 4),
                    "is_disproportionate": c.risk_to_capital_ratio > 1.20,
                }
                for c in corr_res.contributions
            ],
        },
        "benchmark_comparison": {
            "portfolio_beta_vs_spy": bench_res.portfolio_beta_vs_spy,
            "vol_ratio_vs_spy": bench_res.vol_ratio_vs_spy,
            "vol_ratio_vs_60_40": bench_res.vol_ratio_vs_60_40,
            "var_ratio_vs_spy": bench_res.var_ratio_vs_spy,
            "var_ratio_vs_60_40": bench_res.var_ratio_vs_60_40,
            "user_metrics": asdict(bench_res.user_metrics),
            "spy_benchmark": asdict(bench_res.spy_benchmark),
            "balanced_60_40_benchmark": asdict(bench_res.balanced_60_40_benchmark),
            "active_tilts_vs_spy": [
                {
                    "issuer": t.issuer,
                    "name": vui_data.get_asset_info(t.issuer).get("name", t.issuer),
                    "user_weight": round(t.user_weight, 6),
                    "user_weight_pct": round(t.user_weight * 100.0, 2),
                    "spy_weight": round(t.spy_weight, 6),
                    "spy_weight_pct": round(t.spy_weight * 100.0, 2),
                    "active_tilt": round(t.active_tilt, 6),
                    "active_tilt_pct": round(t.active_tilt * 100.0, 2),
                    "multiple_of_spy": t.multiple_of_spy,
                }
                for t in bench_res.active_tilts_vs_spy[:10]
            ],
        },
        "circuit_breakers": {
            "is_safe": safety_res.is_safe and (vol_status == "PASS"),
            "single_stock_limit": xray_profile.max_single_stock_limit,
            "max_tolerable_loss_pct": xray_profile.max_tolerable_loss_pct,
            "evaluations": cb_evaluations,
            "breaches": [e for e in cb_evaluations if e["status"] == "BREACH"],
        },
        "user_risk_profile": xray_profile.to_dict(),
        "evidences": evidences_dict,
        "evidence_list": list(evidences_dict.values()),
    }


def run_unified_what_if(
    portfolio: vui_engine.Portfolio,
    payload: Dict[str, Any],
    audit_logger: Optional[AuditLogger] = None,
) -> Dict[str, Any]:
    """
    Runs a unified What-If simulation combining Dhillon's target-allocation simulation
    with our WhatIfSimulator's realistic execution friction (bid-ask spread, commission,
    quadratic slippage) and before/after circuit-breaker re-verification.
    """
    loader = get_shared_loader()
    audit = audit_logger or AuditLogger()
    whatif_sim = WhatIfSimulator(data_loader=loader, audit_logger=audit)

    xray_holdings = convert_vui_portfolio_to_xray_holdings(portfolio)
    xray_profile = map_vui_profile_to_xray(portfolio.user_profile or {})
    total_val = portfolio.total_value or 32500.0

    # Determine target allocations for v-ui engine
    target_allocations = payload.get("target_allocations", {})
    mode = str(payload.get("mode", "target_allocation")).lower()

    if not target_allocations:
        if "target" in payload or "ticker" in payload:
            target_sym = str(payload.get("target", payload.get("ticker", "NVDA")))
            target_pct = float(payload.get("allocation_change_pct", payload.get("target_pct", 20.0)))
            target_allocations = {target_sym: target_pct, "SPY": max(0.0, 100.0 - target_pct)}
        elif mode == "add_cash":
            target_sym = vui_engine.resolve_symbol_or_name(str(payload.get("target_ticker", "NVDA")))
            cash_amt = float(payload.get("additional_cash", 5000.0))
            new_tot = total_val + cash_amt
            cur_map = {h["ticker"]: h["allocation_gbp"] for h in xray_holdings}
            cur_map[target_sym] = cur_map.get(target_sym, 0.0) + cash_amt
            target_allocations = {k: (v / new_tot) * 100.0 for k, v in cur_map.items()}
        else:
            target_allocations = {"NVDA": 25.0, "SPY": 75.0}
    else:
        # If only 1 ticker is given with < 100%, preserve remaining holdings proportionally
        resolved_targets: Dict[str, float] = {}
        for k, v in target_allocations.items():
            resolved_targets[vui_engine.resolve_symbol_or_name(k)] = float(v)
        sum_pct = sum(resolved_targets.values())
        if len(resolved_targets) == 1 and 0 < sum_pct < 100.0:
            single_t, single_pct = next(iter(resolved_targets.items()))
            rem_pct = 100.0 - single_pct
            other_holdings = [h for h in xray_holdings if h["ticker"] != single_t]
            other_w_sum = sum(h["weight"] for h in other_holdings)
            if other_w_sum > 0:
                for h in other_holdings:
                    resolved_targets[h["ticker"]] = (h["weight"] / other_w_sum) * rem_pct
            else:
                fallback_t = "SPY" if single_t != "SPY" else "BND"
                resolved_targets[fallback_t] = rem_pct
            target_allocations = resolved_targets

    # 1. Run v-ui simulate_what_if
    vui_sim = vui_engine.simulate_what_if(portfolio, target_allocations)

    # 2. Compute before & after states in xray engines + realistic trade friction
    lt_eng = LookThroughEngine(loader)
    conc_eng = ConcentrationEngine()
    safety_eng = SafetyLimitsEngine(audit_logger=audit)

    before_lt = lt_eng.analyze(xray_holdings)
    before_conc = conc_eng.analyze(before_lt)
    before_safety = safety_eng.evaluate_portfolio(
        lt_result=before_lt,
        concentration=before_conc,
        user_max_single_stock_limit=xray_profile.max_single_stock_limit,
        persona_name=f"{xray_profile.name} (Pre-Trade)",
    )

    # Normalize proposed holdings weights
    tot_target = sum(float(v) for v in target_allocations.values()) or 100.0
    after_weights: Dict[str, float] = {}
    for raw_t, pct in target_allocations.items():
        t = vui_engine.resolve_symbol_or_name(raw_t)
        after_weights[t] = after_weights.get(t, 0.0) + (float(pct) / tot_target)

    # Calculate per-asset trade turnover and execution friction
    before_weights = {h["ticker"]: h["weight"] for h in xray_holdings}
    all_trade_tickers = sorted(set(before_weights.keys()) | set(after_weights.keys()))
    frictions = []
    for t in all_trade_tickers:
        w_bef = before_weights.get(t, 0.0)
        w_aft = after_weights.get(t, 0.0)
        delta_val = (w_aft - w_bef) * total_val
        if abs(delta_val) >= 10.0:
            t_type = "BUY" if delta_val > 0 else "SELL"
            f_obj = whatif_sim.compute_friction(t, t_type, abs(delta_val))
            frictions.append(f_obj)

    total_friction_cost = round(sum(f.total_friction_gbp for f in frictions), 2)
    total_spread_cost = round(sum(f.spread_cost_gbp for f in frictions), 2)
    total_commission = round(sum(f.commission_gbp for f in frictions), 2)
    total_slippage = round(sum(f.slippage_gbp for f in frictions), 2)
    total_traded_notional = round(sum(f.trade_value_gbp for f in frictions), 2)
    effective_friction_bps = (
        round((total_friction_cost / total_val) * 10000.0, 2) if total_val > 0 else 0.0
    )

    after_net_val = max(100.0, total_val - total_friction_cost)
    after_xray_holdings = [
        {
            "ticker": t,
            "name": vui_data.get_asset_info(t).get("name", t),
            "allocation_gbp": round(w * after_net_val, 2),
            "weight": w,
        }
        for t, w in after_weights.items()
        if w > 0
    ]

    after_lt = lt_eng.analyze(after_xray_holdings)
    after_conc = conc_eng.analyze(after_lt)
    after_safety = safety_eng.evaluate_portfolio(
        lt_result=after_lt,
        concentration=after_conc,
        user_max_single_stock_limit=xray_profile.max_single_stock_limit,
        persona_name=f"{xray_profile.name} (Simulated What-If)",
    )

    before_breach_issuers = {
        e.breaching_issuer for e in before_safety.evaluations if e.status == "BREACH" and e.breaching_issuer
    }
    after_breach_issuers = {
        e.breaching_issuer for e in after_safety.evaluations if e.status == "BREACH" and e.breaching_issuer
    }
    breaches_resolved = sorted(list(before_breach_issuers - after_breach_issuers))
    new_breaches_created = sorted(list(after_breach_issuers - before_breach_issuers))

    audit.log_event(
        event_type="WHAT_IF_SCENARIO_SIMULATION",
        details={
            "persona": xray_profile.name,
            "target_allocations": {k: round(v * 100.0, 2) for k, v in after_weights.items()},
            "total_friction_cost": total_friction_cost,
            "before_is_safe": before_safety.is_safe,
            "after_is_safe": after_safety.is_safe,
            "breaches_resolved": breaches_resolved,
            "new_breaches_created": new_breaches_created,
        },
    )

    ev_list = (
        before_safety.evidence_list
        + after_safety.evidence_list
        + [
            Evidence(
                id="EV-WHATIF-FRICTION",
                metric="what_if_total_execution_friction",
                value=total_friction_cost,
                formula="sum(spread_cost + fixed_commission + quadratic_slippage)",
                inputs={
                    "total_traded_notional": total_traded_notional,
                    "spread_cost": total_spread_cost,
                    "commission": total_commission,
                    "slippage": total_slippage,
                    "effective_friction_bps": effective_friction_bps,
                },
                data_sources=["config/friction.json"],
                sample_size=len(frictions),
                caveats=["Deducted prior to post-trade weight normalization."],
            )
        ]
    )

    out = dict(vui_sim)
    out["execution_friction"] = {
        "total_traded_notional": total_traded_notional,
        "spread_cost": total_spread_cost,
        "commission": total_commission,
        "slippage": total_slippage,
        "total_friction_cost": total_friction_cost,
        "effective_friction_bps": effective_friction_bps,
        "after_net_portfolio_value": round(after_net_val, 2),
        "trades": [asdict(f) for f in frictions],
        "evidence_id": "EV-WHATIF-FRICTION",
    }
    out["circuit_breaker_comparison"] = {
        "single_stock_limit": xray_profile.max_single_stock_limit,
        "before_is_safe": before_safety.is_safe,
        "after_is_safe": after_safety.is_safe,
        "before_top_issuer": (
            before_conc.lookthrough.top_issuers[0]["issuer"]
            if before_conc.lookthrough.top_issuers
            else "N/A"
        ),
        "before_top_weight_pct": round(before_conc.lookthrough.top_1_weight * 100.0, 2),
        "after_top_issuer": (
            after_conc.lookthrough.top_issuers[0]["issuer"]
            if after_conc.lookthrough.top_issuers
            else "N/A"
        ),
        "after_top_weight_pct": round(after_conc.lookthrough.top_1_weight * 100.0, 2),
        "breaches_resolved": breaches_resolved,
        "new_breaches_created": new_breaches_created,
        "after_evaluations": [asdict(e) for e in after_safety.evaluations],
    }
    out["evidences"] = {ev.id: ev.to_dict() for ev in ev_list}
    return out


def record_human_signoff(
    decision: str,
    scenario_summary: str,
    persona_name: str = "Everyday Investor",
    details: Optional[Dict[str, Any]] = None,
    audit_logger: Optional[AuditLogger] = None,
) -> Dict[str, Any]:
    """
    Records a human-in-the-loop sign-off decision (APPROVED or REJECTED) on a
    simulated portfolio change to the append-only audit log (logs/audit.jsonl).
    """
    audit = audit_logger or AuditLogger()
    clean_decision = str(decision).upper().strip()
    if clean_decision not in ("APPROVED", "REJECTED"):
        clean_decision = "REVIEWED"

    entry = audit.log_event(
        event_type="HUMAN_WHATIF_SIGNOFF",
        details={
            "decision": clean_decision,
            "scenario_summary": scenario_summary,
            "persona": persona_name,
            "metadata": details or {},
        },
    )
    return {
        "status": "recorded",
        "decision": clean_decision,
        "timestamp": entry["timestamp"],
        "event_type": entry["event_type"],
        "scenario_summary": scenario_summary,
        "evidence_id": f"EV-SIGNOFF-{clean_decision}",
    }
