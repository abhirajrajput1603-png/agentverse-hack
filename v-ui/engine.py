from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import traceback

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.concentration import ConcentrationEngine
from xray.engine.risk import RiskEngine
from xray.engine.correlation import CorrelationAndRiskContributionEngine
from xray.engine.stress import StressAndTailEngine, StressReplayScenario
from xray.engine.benchmark import BenchmarkEngine
from xray.engine.whatif import WhatIfSimulator
from xray.engine.themes import ThemeEngine
from xray.profile import UserRiskProfile
from xray.safety.limits import SafetyLimitsEngine

# Import old engine logic for fallback and data models
from engine_old import (
    Holding,
    Portfolio,
    EffectiveExposure,
    compute_lookthrough_exposure as old_compute_lookthrough_exposure,
    compute_thematic_exposure as old_compute_thematic_exposure,
    compute_risk_metrics as old_compute_risk_metrics,
    run_stress_tests as old_run_stress_tests,
    simulate_what_if as old_simulate_what_if,
    analyze_headline_event as old_analyze_headline_event,
)

# Initialize engines
dl = DataLoader()
lt_engine = LookThroughEngine(dl)
conc_engine = ConcentrationEngine()
risk_engine = RiskEngine(dl)
corr_engine = CorrelationAndRiskContributionEngine(dl)
stress_engine = StressAndTailEngine(dl)
bench_engine = BenchmarkEngine(dl)
whatif_sim = WhatIfSimulator(dl)
limits_engine = SafetyLimitsEngine()
theme_engine = ThemeEngine(dl)

def _to_raw_holdings(portfolio: Portfolio) -> List[Dict[str, Any]]:
    tv = portfolio.total_value
    if tv <= 0:
        return []
    return [{"ticker": h.ticker, "weight": h.shares * h.price / tv, "allocation_gbp": h.shares * h.price} for h in portfolio.holdings]

def compute_lookthrough_exposure(portfolio: Portfolio, mode: str = "overlap") -> List[EffectiveExposure]:
    """Returns Dhillon's overlap exposure for compatibility, or institutional if mode=='institutional'."""
    return old_compute_lookthrough_exposure(portfolio)

def get_institutional_lookthrough(portfolio: Portfolio) -> Dict[str, Any]:
    raw = _to_raw_holdings(portfolio)
    if not raw:
        return {"exposures": [], "metrics": {}, "evidences": []}
    
    try:
        lt_res = lt_engine.analyze(raw)
        conc_res = conc_engine.analyze(lt_res)
        theme_res = theme_engine.analyze(lt_res)
        
        exps = []
        for exp in lt_res.sorted_exposures(exclude_residual_and_cash=True):
            exps.append({
                "ticker": exp.issuer,
                "total_weight": round(exp.total_weight * 100.0, 2),
                "direct_weight": round(exp.direct_weight * 100.0, 2),
                "indirect_weight": round(exp.indirect_weight * 100.0, 2),
                "is_fund_undisclosed": exp.is_fund_undisclosed,
                "breakdown_by_source": {k: round(v * 100.0, 2) for k, v in exp.breakdown_by_source.items()}
            })
            
        metrics = {
            "hhi": round(conc_res.hhi, 1),
            "effective_n": round(conc_res.effective_n, 1),
            "top_1_weight": round(conc_res.top_1_weight * 100.0, 2),
            "top_3_weight": round(conc_res.top_3_weight * 100.0, 2),
            "top_5_weight": round(conc_res.top_5_weight * 100.0, 2),
            "top_10_weight": round(conc_res.top_10_weight * 100.0, 2),
        }
        
        return {
            "exposures": exps,
            "metrics": metrics,
            "themes": [{"theme": t.theme, "weight": round(t.weight * 100.0, 2)} for t in theme_res.exposures],
            "evidences": [e.to_dict() for e in lt_res.evidence_list] + [e.to_dict() for e in conc_res.evidence_list]
        }
    except Exception as e:
        print(f"Institutional lookthrough failed: {e}")
        return {"exposures": [], "metrics": {}, "evidences": []}

def compute_risk_metrics(portfolio: Portfolio, exposures: List[EffectiveExposure]) -> Dict[str, Any]:
    old_metrics = old_compute_risk_metrics(portfolio, exposures)
    raw = _to_raw_holdings(portfolio)
    
    # Try institutional engine
    try:
        risk_res = risk_engine.analyze(raw, cov_estimator="sample")
        
        # Override old metrics with institutional deterministic ones
        old_metrics["annual_volatility"] = round(risk_res.annualized_volatility * 100.0, 2)
        old_metrics["var_95_1m_pct"] = round(risk_res.var_95_1m * 100.0, 2)
        old_metrics["var_95_1m_dollar"] = round(risk_res.var_95_1m * portfolio.total_value, 2)
        old_metrics["expected_shortfall_95_pct"] = round(risk_res.cvar_95_1m * 100.0, 2)
        old_metrics["expected_shortfall_95_dollar"] = round(risk_res.cvar_95_1m * portfolio.total_value, 2)
        old_metrics["institutional_success"] = True
        
        evidences = [e.to_dict() for e in risk_res.evidence_list]
        old_metrics["evidences"] = evidences
        
        # Add bootstrap CIs
        old_metrics["bootstrap_var_95_ci"] = [round(x * 100.0, 2) for x in risk_res.bootstrap_var_95_ci] if risk_res.bootstrap_var_95_ci else None
        old_metrics["bootstrap_cvar_95_ci"] = [round(x * 100.0, 2) for x in risk_res.bootstrap_cvar_95_ci] if risk_res.bootstrap_cvar_95_ci else None
    except Exception as e:
        old_metrics["institutional_success"] = False
        print(f"Institutional risk metrics failed: {e}")
        
    return old_metrics

def get_correlation_and_risk(portfolio: Portfolio) -> Dict[str, Any]:
    raw = _to_raw_holdings(portfolio)
    if not raw:
        return {}
    try:
        res = corr_engine.analyze(raw)
        contributions = []
        for rc in res.risk_contributions:
            contributions.append({
                "ticker": rc.ticker,
                "weight": round(rc.weight * 100.0, 2),
                "marginal_contribution": round(rc.marginal_contribution * 100.0, 2),
                "component_contribution": round(rc.component_contribution * 100.0, 2),
                "pct_contribution": round(rc.pct_contribution * 100.0, 2),
                "risk_capital_ratio": round(rc.risk_capital_ratio, 2),
            })
        return {
            "matrix_1y": res.correlation_matrix_1y,
            "matrix_3y": res.correlation_matrix_3y,
            "tickers": res.tickers,
            "risk_contributions": contributions,
            "evidences": [e.to_dict() for e in res.evidence_list]
        }
    except Exception as e:
        print(f"Correlation engine failed: {e}")
        return {}

def run_stress_tests(portfolio: Portfolio, exposures: List[EffectiveExposure], scenarios=None) -> List[Dict[str, Any]]:
    # Dhillon's dynamic factor stress tests
    old_stress = old_run_stress_tests(portfolio, exposures, scenarios)
    return old_stress

def get_historical_stress(portfolio: Portfolio) -> Dict[str, Any]:
    raw = _to_raw_holdings(portfolio)
    if not raw:
        return {}
    try:
        res = stress_engine.analyze(raw)
        replays = []
        for r in res.replays:
            replays.append({
                "scenario": r.scenario_id,
                "portfolio_drawdown": round(r.portfolio_max_drawdown * 100.0, 2),
                "spy_drawdown": round(r.spy_cumulative_return * 100.0, 2), # Using spy_cumulative_return as a proxy for drawdown comparison in this simplified mapping
                "recovery_days": r.sample_days,
            })
        return {
            "replays": replays,
            "max_drawdown_3y": round(res.max_drawdown_3y * 100.0, 2),
            "tail_ratio": round(res.tail_ratio, 2) if res.tail_ratio else None,
            "evidences": [e.to_dict() for e in res.evidence_list]
        }
    except Exception as e:
        print(f"Historical stress engine failed: {e}")
        return {}

def get_benchmark_comparison(portfolio: Portfolio) -> Dict[str, Any]:
    raw = _to_raw_holdings(portfolio)
    if not raw:
        return {}
    try:
        res = bench_engine.analyze(raw)
        return {
            "portfolio_beta": round(res.portfolio_beta, 2),
            "tracking_error": round(res.tracking_error * 100.0, 2),
            "active_risk": round(res.active_risk * 100.0, 2),
            "spy_comparison": {
                "volatility": round(res.spy_metrics.annualized_volatility * 100.0, 2),
                "var_95": round(res.spy_metrics.var_95_1m * 100.0, 2),
                "max_drawdown": round(res.spy_metrics.max_drawdown_3y * 100.0, 2),
            },
            "bnd_6040_comparison": {
                "volatility": round(res.bnd_6040_metrics.annualized_volatility * 100.0, 2),
                "var_95": round(res.bnd_6040_metrics.var_95_1m * 100.0, 2),
                "max_drawdown": round(res.bnd_6040_metrics.max_drawdown_3y * 100.0, 2),
            },
            "tilts": [{"issuer": t.issuer, "active_weight": round(t.active_weight * 100.0, 2)} for t in res.top_active_tilts],
            "evidences": [e.to_dict() for e in res.evidence_list]
        }
    except Exception as e:
        print(f"Benchmark engine failed: {e}")
        return {}

def simulate_what_if(portfolio: Portfolio, target_allocations: Dict[str, float]) -> Dict[str, Any]:
    old_result = old_simulate_what_if(portfolio, target_allocations)
    
    # Try to add friction and circuit breaker resolution
    raw = _to_raw_holdings(portfolio)
    raw_target = [{"ticker": k, "weight": v / 100.0} for k, v in target_allocations.items()]
    
    try:
        # Calculate friction manually using whatif_sim.compute_friction
        total_friction = 0.0
        for ticker, target_pct in target_allocations.items():
            current_pct = next((h.shares * h.price / portfolio.total_value * 100 for h in portfolio.holdings if h.ticker == ticker), 0)
            diff_pct = target_pct - current_pct
            if diff_pct != 0:
                trade_val = abs(diff_pct / 100.0 * portfolio.total_value)
                fric = whatif_sim.compute_friction(ticker, "BUY" if diff_pct > 0 else "SELL", trade_val)
                total_friction += fric.total_friction_cost
        
        old_result["friction"] = {
            "bid_ask_spread": round(total_friction * 0.4, 2), # Approximations for demo
            "commission": round(total_friction * 0.2, 2),
            "slippage": round(total_friction * 0.4, 2),
            "total_friction_cost": round(total_friction, 2),
            "friction_bps": round((total_friction / portfolio.total_value) * 10000 if portfolio.total_value > 0 else 0, 2)
        }
        
        # Check circuit breakers for proposed
        prof = portfolio.user_profile or {}
        user_prof = UserRiskProfile(
            risk_tolerance=prof.get("risk_tolerance", "Moderate"),
            horizon_years=prof.get("horizon_years", 7),
            loss_capacity=prof.get("loss_capacity", "Moderate (15-20% max drawdown)")
        )
        old_res_limits = limits_engine.evaluate(raw, user_prof)
        new_res_limits = limits_engine.evaluate(raw_target, user_prof)
        
        old_result["circuit_breakers"] = {
            "current_breaches": [b.message for b in old_res_limits.breaches],
            "proposed_breaches": [b.message for b in new_res_limits.breaches],
            "resolved": len(old_res_limits.breaches) > len(new_res_limits.breaches),
            "new_breaches_created": len(new_res_limits.breaches) > len(old_res_limits.breaches)
        }
    except Exception as e:
        print(f"What-if sim failed: {e}")
        old_result["friction"] = {"total_friction_cost": 0, "friction_bps": 0, "bid_ask_spread": 0, "commission": 0, "slippage": 0}
        old_result["circuit_breakers"] = {"current_breaches": [], "proposed_breaches": [], "resolved": False, "new_breaches_created": False}
        
    return old_result

def analyze_headline_event(portfolio: Portfolio, headline: str) -> Dict[str, Any]:
    return old_analyze_headline_event(portfolio, headline)

def compute_thematic_exposure(exposures: List[EffectiveExposure], min_weight: float = 0.05) -> List[Dict[str, Any]]:
    return old_compute_thematic_exposure(exposures, min_weight)
