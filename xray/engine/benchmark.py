from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine, LookThroughResult
from xray.engine.risk import RiskEngine
from xray.engine.stress import StressAndTailEngine
from xray.evidence import Evidence


@dataclass
class BenchmarkProfileMetrics:
    benchmark_id: str
    name: str
    label: str
    holdings: List[Dict[str, Any]]
    top_1_issuer: str
    top_1_weight: float
    top_5_weight: float
    lookthrough_hhi: float
    lookthrough_n_eff: float
    annualized_volatility: float
    historical_var_95: float
    expected_shortfall_95: float
    max_drawdown: float


@dataclass
class ActiveIssuerTilt:
    issuer: str
    user_weight: float
    spy_weight: float
    active_tilt: float            # user_weight - spy_weight
    multiple_of_spy: Optional[float]  # user_weight / spy_weight if spy_weight > 0


@dataclass
class BenchmarkComparisonResult:
    user_metrics: BenchmarkProfileMetrics
    spy_benchmark: BenchmarkProfileMetrics
    balanced_60_40_benchmark: BenchmarkProfileMetrics
    portfolio_beta_vs_spy: float
    vol_ratio_vs_spy: float
    vol_ratio_vs_60_40: float
    var_ratio_vs_spy: float
    var_ratio_vs_60_40: float
    active_tilts_vs_spy: List[ActiveIssuerTilt]
    evidence_list: List[Evidence] = field(default_factory=list)


class BenchmarkEngine:
    """
    Deterministic benchmark comparison engine.
    Compares any user portfolio against two canonical institutional baselines:
      1. SPY (100% US Large-Cap Equity Proxy)
      2. 60/40 Portfolio (60% SPY US Large-Cap Equity / 40% BND US Aggregate Bond)
    Computes concentration, volatility, VaR, Expected Shortfall, Max Drawdown,
    Portfolio Beta vs SPY, and Active Look-Through Issuer Tilts.
    """

    SPY_HOLDINGS = [{"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "weight": 1.0}]
    BALANCED_60_40_HOLDINGS = [
        {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "weight": 0.60},
        {"ticker": "BND", "name": "Vanguard Total Bond Market ETF", "weight": 0.40},
    ]

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self.lt_engine = LookThroughEngine(self.loader)
        self.conc_engine = ConcentrationEngine()
        self.risk_engine = RiskEngine(self.loader)
        self.stress_engine = StressAndTailEngine(self.loader)

    def _compute_profile(
        self,
        profile_id: str,
        name: str,
        label: str,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 756,
    ) -> Tuple[BenchmarkProfileMetrics, LookThroughResult]:
        lt_res = self.lt_engine.analyze(holdings)
        conc_res = self.conc_engine.analyze(lt_res)
        risk_res = self.risk_engine.analyze(holdings, lookback_days=lookback_days)
        stress_res = self.stress_engine.analyze(holdings, lookback_days=lookback_days, n_bootstrap=200)

        lt_m = conc_res.lookthrough
        top_1_issuer_name = lt_m.top_issuers[0]["issuer"] if lt_m.top_issuers else "N/A"
        return (
            BenchmarkProfileMetrics(
                benchmark_id=profile_id,
                name=name,
                label=label,
                holdings=holdings,
                top_1_issuer=top_1_issuer_name,
                top_1_weight=round(lt_m.top_1_weight, 6),
                top_5_weight=round(lt_m.top_5_weight, 6),
                lookthrough_hhi=round(lt_m.hhi, 6),
                lookthrough_n_eff=round(lt_m.effective_n, 4),
                annualized_volatility=round(risk_res.annualized_volatility, 6),
                historical_var_95=round(risk_res.historical_var_95, 6),
                expected_shortfall_95=round(stress_res.tail_risk.es_95, 6),
                max_drawdown=round(risk_res.max_drawdown, 6),
            ),
            lt_res,
        )

    def _compute_beta_vs_spy(
        self,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 756,
    ) -> float:
        prices = self.risk_engine.get_prices()
        if "SPY" not in prices.columns:
            return 1.0

        priced_tickers: List[str] = []
        priced_weights: List[float] = []
        cash_weight: float = 0.0

        for h in holdings:
            t = h.get("ticker", "")
            w = float(h.get("weight", 0.0))
            if t in ("CASH", "CASH_GBP", "CASH_USD"):
                cash_weight += w
            elif t in prices.columns:
                priced_tickers.append(t)
                priced_weights.append(w)

        if not priced_tickers:
            return 0.0

        cols = list(dict.fromkeys(priced_tickers + ["SPY"]))
        rets = prices[cols].iloc[-lookback_days:].pct_change(fill_method=None).dropna()
        if len(rets) < 2:
            return 0.0

        if cash_weight > 0:
            rets = rets.copy()
            rets["CASH_GBP"] = 0.0
            eval_cols = priced_tickers + ["CASH_GBP"]
            w_arr = np.array(priced_weights + [cash_weight], dtype=float)
        else:
            eval_cols = priced_tickers
            w_arr = np.array(priced_weights, dtype=float)

        w_norm = w_arr / w_arr.sum()
        port_rets = rets[eval_cols].values @ w_norm
        spy_rets = rets["SPY"].values

        spy_var = float(np.var(spy_rets, ddof=1))
        if spy_var < 1e-12:
            return 0.0
        cov_p_spy = float(np.cov(port_rets, spy_rets, ddof=1)[0, 1])
        return round(cov_p_spy / spy_var, 4)

    def analyze(
        self,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 756,
    ) -> BenchmarkComparisonResult:
        """
        Compares user portfolio against SPY (100% US Equity Proxy) and 60/40 SPY/BND.
        """
        evidence_list: List[Evidence] = []

        user_profile, user_lt = self._compute_profile(
            profile_id="USER_PORTFOLIO",
            name="User Portfolio",
            label="Current Holdings (Look-Through)",
            holdings=holdings,
            lookback_days=lookback_days,
        )

        spy_profile, spy_lt = self._compute_profile(
            profile_id="BENCH_SPY",
            name="SPY (100% US Equity Proxy)",
            label="SPDR S&P 500 ETF Trust (US Large-Cap Proxy)",
            holdings=self.SPY_HOLDINGS,
            lookback_days=lookback_days,
        )

        b6040_profile, _ = self._compute_profile(
            profile_id="BENCH_60_40",
            name="60/40 SPY/BND Balanced",
            label="60% SPY (US Equity) / 40% BND (US Aggregate Bond)",
            holdings=self.BALANCED_60_40_HOLDINGS,
            lookback_days=lookback_days,
        )

        beta_spy = self._compute_beta_vs_spy(holdings, lookback_days=lookback_days)

        vol_ratio_spy = (
            round(user_profile.annualized_volatility / spy_profile.annualized_volatility, 4)
            if spy_profile.annualized_volatility > 1e-9
            else 0.0
        )
        vol_ratio_60_40 = (
            round(user_profile.annualized_volatility / b6040_profile.annualized_volatility, 4)
            if b6040_profile.annualized_volatility > 1e-9
            else 0.0
        )

        var_ratio_spy = (
            round(abs(user_profile.historical_var_95) / abs(spy_profile.historical_var_95), 4)
            if abs(spy_profile.historical_var_95) > 1e-9
            else 0.0
        )
        var_ratio_60_40 = (
            round(abs(user_profile.historical_var_95) / abs(b6040_profile.historical_var_95), 4)
            if abs(b6040_profile.historical_var_95) > 1e-9
            else 0.0
        )

        # Compute Active Issuer Tilts vs SPY across all non-residual/non-cash issuers
        all_issuers = set(user_lt.issuer_exposures.keys()) | set(spy_lt.issuer_exposures.keys())
        tilts: List[ActiveIssuerTilt] = []

        for iss in all_issuers:
            u_exp = user_lt.issuer_exposures.get(iss)
            s_exp = spy_lt.issuer_exposures.get(iss)
            if (u_exp and (u_exp.is_residual or u_exp.is_cash)) or (
                s_exp and (s_exp.is_residual or s_exp.is_cash)
            ):
                continue

            u_w = u_exp.total_weight if u_exp else 0.0
            s_w = s_exp.total_weight if s_exp else 0.0
            diff = u_w - s_w
            mult = round(u_w / s_w, 2) if s_w > 1e-6 else None

            tilts.append(
                ActiveIssuerTilt(
                    issuer=iss,
                    user_weight=round(u_w, 6),
                    spy_weight=round(s_w, 6),
                    active_tilt=round(diff, 6),
                    multiple_of_spy=mult,
                )
            )

        tilts_sorted = sorted(tilts, key=lambda x: abs(x.active_tilt), reverse=True)

        # Generate Evidence Objects
        evidence_list.append(
            Evidence(
                id="EV-BENCH-SPY-COMPARISON",
                metric="portfolio_vs_spy_volatility_ratio",
                value=vol_ratio_spy,
                formula="sigma_user / sigma_SPY",
                inputs={
                    "user_volatility": user_profile.annualized_volatility,
                    "spy_volatility": spy_profile.annualized_volatility,
                    "portfolio_beta_vs_spy": beta_spy,
                    "user_var_95": user_profile.historical_var_95,
                    "spy_var_95": spy_profile.historical_var_95,
                    "var_ratio_vs_spy": var_ratio_spy,
                    "user_max_drawdown": user_profile.max_drawdown,
                    "spy_max_drawdown": spy_profile.max_drawdown,
                },
                data_sources=[
                    "us_market_prices_matrix.csv (3-Year Daily Closes)",
                    "etf_constituents.json (SPY N-PORT Look-Through)",
                ],
                sample_size=lookback_days,
                caveats=["SPY used as labelled US large-cap equity proxy."],
            )
        )

        evidence_list.append(
            Evidence(
                id="EV-BENCH-60-40-COMPARISON",
                metric="portfolio_vs_60_40_volatility_ratio",
                value=vol_ratio_60_40,
                formula="sigma_user / sigma_60_40",
                inputs={
                    "user_volatility": user_profile.annualized_volatility,
                    "balanced_60_40_volatility": b6040_profile.annualized_volatility,
                    "user_var_95": user_profile.historical_var_95,
                    "balanced_60_40_var_95": b6040_profile.historical_var_95,
                    "var_ratio_vs_60_40": var_ratio_60_40,
                    "user_max_drawdown": user_profile.max_drawdown,
                    "balanced_60_40_max_drawdown": b6040_profile.max_drawdown,
                },
                data_sources=["us_market_prices_matrix.csv (60% SPY / 40% BND)"],
                sample_size=lookback_days,
                caveats=["60/40 baseline rebalanced across SPY (60%) and BND (40%)."],
            )
        )

        if tilts_sorted:
            top_tilt = tilts_sorted[0]
            evidence_list.append(
                Evidence(
                    id=f"EV-BENCH-ACTIVE-TILT-{top_tilt.issuer}",
                    metric=f"active_concentration_tilt_{top_tilt.issuer}",
                    value=top_tilt.active_tilt,
                    formula="w_lt_user(issuer) - w_lt_SPY(issuer)",
                    inputs={
                        "issuer": top_tilt.issuer,
                        "user_lookthrough_weight": top_tilt.user_weight,
                        "spy_lookthrough_weight": top_tilt.spy_weight,
                        "active_tilt": top_tilt.active_tilt,
                        "multiple_of_spy": top_tilt.multiple_of_spy,
                    },
                    data_sources=["etf_constituents.json (SEC Form N-PORT)"],
                    sample_size=None,
                    caveats=[],
                )
            )

        return BenchmarkComparisonResult(
            user_metrics=user_profile,
            spy_benchmark=spy_profile,
            balanced_60_40_benchmark=b6040_profile,
            portfolio_beta_vs_spy=beta_spy,
            vol_ratio_vs_spy=vol_ratio_spy,
            vol_ratio_vs_60_40=vol_ratio_60_40,
            var_ratio_vs_spy=var_ratio_spy,
            var_ratio_vs_60_40=var_ratio_60_40,
            active_tilts_vs_spy=tilts_sorted,
            evidence_list=evidence_list,
        )
