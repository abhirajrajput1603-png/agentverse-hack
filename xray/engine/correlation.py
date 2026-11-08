from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xray.data_loader import DataLoader
from xray.evidence import Evidence


@dataclass
class AssetRiskContribution:
    ticker: str
    capital_weight: float          # w_i (normalized across priced + cash holdings)
    mcr: float                     # Marginal Contribution to Risk: (Sigma_ann * w)_i / sigma_p
    ccr: float                     # Component Contribution to Risk: w_i * MCR_i (sums to sigma_p)
    pct_risk_contribution: float   # CCR_i / sigma_p (sums to 1.0)
    risk_to_capital_ratio: float   # pct_risk_contribution / capital_weight


@dataclass
class CorrelationRiskResult:
    portfolio_volatility: float
    ccr_sum: float
    ccr_sum_check_passed: bool
    contributions: List[AssetRiskContribution]
    corr_matrix_1y: Dict[str, Dict[str, float]]
    corr_matrix_3y: Dict[str, Dict[str, float]]
    avg_pairwise_corr_1y: float
    avg_pairwise_corr_3y: float
    highest_corr_pair_3y: Optional[Tuple[str, str, float]]
    top_risk_contributor: Optional[AssetRiskContribution]
    sample_size_1y: int
    sample_size_3y: int
    unpriced_holdings: List[str]
    evidence_list: List[Evidence] = field(default_factory=list)


class CorrelationAndRiskContributionEngine:
    """
    Deterministic correlation and Euler risk decomposition engine.
    Computes:
      1. 1-Year (252d) and 3-Year (756d) return correlation matrices across holdings.
      2. Marginal Contribution to Risk: MCR_i = (Sigma_ann @ w)_i / sigma_p
      3. Component Contribution to Risk: CCR_i = w_i * MCR_i
      4. Euler Homogeneity Check: sum_i(CCR_i) == sigma_p within 1e-8
      5. Capital Weight vs. Risk Weight attribution.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self._prices_cache: Optional[pd.DataFrame] = None

    def get_prices(self) -> pd.DataFrame:
        if self._prices_cache is None:
            self._prices_cache = self.loader.load_prices()
        return self._prices_cache

    @staticmethod
    def _compute_avg_and_max_corr(
        corr_df: pd.DataFrame,
    ) -> Tuple[float, Optional[Tuple[str, str, float]]]:
        """
        Extracts average off-diagonal pairwise correlation and the highest correlated pair.
        """
        tickers = list(corr_df.columns)
        n = len(tickers)
        if n < 2:
            return 1.0 if n == 1 else 0.0, None

        pair_vals: List[float] = []
        max_pair: Optional[Tuple[str, str, float]] = None

        for i in range(n):
            for j in range(i + 1, n):
                val = float(corr_df.iloc[i, j])
                if not np.isnan(val):
                    pair_vals.append(val)
                    if max_pair is None or val > max_pair[2]:
                        max_pair = (tickers[i], tickers[j], round(val, 4))

        avg_corr = float(np.mean(pair_vals)) if pair_vals else 0.0
        return round(avg_corr, 4), max_pair

    def compute_from_returns(
        self,
        returns_3y: pd.DataFrame,
        returns_1y: pd.DataFrame,
        tickers: List[str],
        weights: np.ndarray,
        cash_weight: float = 0.0,
        unpriced_holdings: Optional[List[str]] = None,
    ) -> CorrelationRiskResult:
        """
        Pure mathematical computation from return DataFrames and weights.
        """
        unpriced_holdings = unpriced_holdings or []
        evidence_list: List[Evidence] = []

        # 1. Correlation Matrices (on risky priced tickers)
        corr_3y_df = returns_3y[tickers].corr() if len(tickers) >= 1 else pd.DataFrame()
        corr_1y_df = returns_1y[tickers].corr() if len(tickers) >= 1 else pd.DataFrame()

        avg_corr_3y, max_pair_3y = self._compute_avg_and_max_corr(corr_3y_df)
        avg_corr_1y, _ = self._compute_avg_and_max_corr(corr_1y_df)

        corr_dict_3y = {
            r: {c: round(float(corr_3y_df.loc[r, c]), 4) for c in corr_3y_df.columns}
            for r in corr_3y_df.index
        }
        corr_dict_1y = {
            r: {c: round(float(corr_1y_df.loc[r, c]), 4) for c in corr_1y_df.columns}
            for r in corr_1y_df.index
        }

        # 2. Prepare full return matrix including synthetic zero-return cash if present
        rets_eval = returns_3y[tickers].copy()
        eval_tickers = list(tickers)
        raw_weights = list(weights)

        if cash_weight > 0:
            rets_eval["CASH_GBP"] = 0.0
            eval_tickers.append("CASH_GBP")
            raw_weights.append(cash_weight)

        w_arr = np.array(raw_weights, dtype=float)
        w_sum = w_arr.sum()
        w_norm = w_arr / w_sum if w_sum > 0 else w_arr

        # 3. Annualised Covariance Matrix: Sigma_ann = Cov_daily * 252
        cov_ann = rets_eval.cov().values * 252.0
        port_var = float(w_norm.T @ cov_ann @ w_norm)
        sigma_p = float(np.sqrt(max(port_var, 0.0)))

        # 4. Marginal and Component Risk Contributions
        contributions: List[AssetRiskContribution] = []
        if sigma_p > 1e-12:
            # MCR = (Sigma_ann @ w) / sigma_p
            mcr_vec = (cov_ann @ w_norm) / sigma_p
            # CCR = w * MCR
            ccr_vec = w_norm * mcr_vec
            pct_cr_vec = ccr_vec / sigma_p
        else:
            mcr_vec = np.zeros_like(w_norm)
            ccr_vec = np.zeros_like(w_norm)
            pct_cr_vec = np.zeros_like(w_norm)

        ccr_sum = float(np.sum(ccr_vec))
        ccr_check = abs(ccr_sum - sigma_p) < 1e-8
        if not ccr_check:
            raise AssertionError(
                f"Euler risk decomposition failure: sum(CCR) = {ccr_sum:.10f} != sigma_p = {sigma_p:.10f}"
            )

        for idx, t in enumerate(eval_tickers):
            cap_w = float(w_norm[idx])
            mcr_val = float(mcr_vec[idx])
            ccr_val = float(ccr_vec[idx])
            pct_cr = float(pct_cr_vec[idx])
            ratio = (pct_cr / cap_w) if cap_w > 1e-9 else 0.0

            contributions.append(
                AssetRiskContribution(
                    ticker=t,
                    capital_weight=round(cap_w, 6),
                    mcr=round(mcr_val, 6),
                    ccr=round(ccr_val, 6),
                    pct_risk_contribution=round(pct_cr, 6),
                    risk_to_capital_ratio=round(ratio, 4),
                )
            )

        # Sort contributions by pct_risk_contribution descending
        contributions_sorted = sorted(
            contributions, key=lambda x: x.pct_risk_contribution, reverse=True
        )
        top_contrib = contributions_sorted[0] if contributions_sorted else None

        caveats: List[str] = []
        if unpriced_holdings:
            caveats.append(
                f"Unpriced holdings {unpriced_holdings} excluded from covariance matrix; weights normalized over priced assets."
            )

        # 5. Generate Evidence Objects
        if top_contrib:
            evidence_list.append(
                Evidence(
                    id=f"EV-RISK-CONTRIB-TOP-{top_contrib.ticker}",
                    metric=f"top_risk_contributor_{top_contrib.ticker}",
                    value=round(top_contrib.pct_risk_contribution, 4),
                    formula="CCR_i = w_i * (Sigma_ann @ w)_i / sigma_p; %CR_i = CCR_i / sigma_p",
                    inputs={
                        "ticker": top_contrib.ticker,
                        "capital_weight": round(top_contrib.capital_weight, 4),
                        "mcr": round(top_contrib.mcr, 4),
                        "ccr": round(top_contrib.ccr, 4),
                        "portfolio_volatility": round(sigma_p, 4),
                        "risk_to_capital_ratio": top_contrib.risk_to_capital_ratio,
                    },
                    data_sources=["us_market_prices_matrix.csv (3Y Daily Returns)"],
                    sample_size=len(returns_3y),
                    caveats=caveats,
                )
            )

        evidence_list.append(
            Evidence(
                id="EV-CORR-AVG-3Y",
                metric="average_pairwise_correlation_3y",
                value=avg_corr_3y,
                formula="mean_{i < j}(Corr_3Y(r_i, r_j))",
                inputs={
                    "avg_corr_1y": avg_corr_1y,
                    "avg_corr_3y": avg_corr_3y,
                    "highest_pair_3y": (
                        f"{max_pair_3y[0]}-{max_pair_3y[1]} ({max_pair_3y[2]})"
                        if max_pair_3y
                        else "N/A"
                    ),
                },
                data_sources=["us_market_prices_matrix.csv (1Y & 3Y Daily Returns)"],
                sample_size=len(returns_3y),
                caveats=caveats,
            )
        )

        return CorrelationRiskResult(
            portfolio_volatility=round(sigma_p, 6),
            ccr_sum=round(ccr_sum, 6),
            ccr_sum_check_passed=ccr_check,
            contributions=contributions_sorted,
            corr_matrix_1y=corr_dict_1y,
            corr_matrix_3y=corr_dict_3y,
            avg_pairwise_corr_1y=avg_corr_1y,
            avg_pairwise_corr_3y=avg_corr_3y,
            highest_corr_pair_3y=max_pair_3y,
            top_risk_contributor=top_contrib,
            sample_size_1y=len(returns_1y),
            sample_size_3y=len(returns_3y),
            unpriced_holdings=unpriced_holdings,
            evidence_list=evidence_list,
        )

    def analyze(
        self,
        holdings: List[Dict[str, Any]],
        days_3y: int = 756,
        days_1y: int = 252,
    ) -> CorrelationRiskResult:
        """
        Analyzes 1Y/3Y correlations and Euler risk contributions for a portfolio.
        """
        prices = self.get_prices()

        priced_tickers: List[str] = []
        priced_weights: List[float] = []
        unpriced_holdings: List[str] = []
        cash_weight: float = 0.0

        for h in holdings:
            t = h.get("ticker", "")
            w = float(h.get("weight", 0.0))
            if t in ("CASH", "CASH_GBP", "CASH_USD"):
                cash_weight += w
            elif t in prices.columns:
                priced_tickers.append(t)
                priced_weights.append(w)
            else:
                unpriced_holdings.append(t)

        if not priced_tickers:
            return CorrelationRiskResult(
                portfolio_volatility=0.0,
                ccr_sum=0.0,
                ccr_sum_check_passed=True,
                contributions=[],
                corr_matrix_1y={},
                corr_matrix_3y={},
                avg_pairwise_corr_1y=0.0,
                avg_pairwise_corr_3y=0.0,
                highest_corr_pair_3y=None,
                top_risk_contributor=None,
                sample_size_1y=0,
                sample_size_3y=0,
                unpriced_holdings=unpriced_holdings,
                evidence_list=[],
            )

        rets_all = prices[priced_tickers].pct_change(fill_method=None).dropna()
        rets_3y = rets_all.iloc[-days_3y:]
        rets_1y = rets_all.iloc[-days_1y:]

        return self.compute_from_returns(
            returns_3y=rets_3y,
            returns_1y=rets_1y,
            tickers=priced_tickers,
            weights=np.array(priced_weights, dtype=float),
            cash_weight=cash_weight,
            unpriced_holdings=unpriced_holdings,
        )
