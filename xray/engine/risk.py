from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xray.data_loader import DataLoader
from xray.engine.lookthrough import Holding, LookThroughResult
from xray.evidence import Evidence


VALID_COV_METHODS = ("sample", "ledoit_wolf", "ewma")


def estimate_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = "sample",
    ewma_lambda: float = 0.94,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Deterministic daily covariance matrix estimator.
    Supports:
      - 'sample': Unbiased sample covariance S = (1/(T-1)) Y^T Y (default baseline)
      - 'ledoit_wolf': Analytical Ledoit-Wolf (2004) linear shrinkage towards scaled identity F = mu * I
      - 'ewma': RiskMetrics Exponentially Weighted Moving Average covariance with decay lambda (default 0.94)
    Returns (daily_cov_matrix, metadata_dict).
    """
    method_clean = method.lower().strip()
    if method_clean not in VALID_COV_METHODS:
        raise ValueError(f"Unknown covariance method '{method}'. Expected one of {VALID_COV_METHODS}.")

    X = returns_df.values.astype(float)
    T, N = X.shape
    if T < 2 or N == 0:
        return np.zeros((N, N), dtype=float), {"method": method_clean}

    if method_clean == "sample":
        cov_sample = returns_df.cov().values
        return cov_sample, {"method": "sample", "sample_days": T}

    if method_clean == "ledoit_wolf":
        # Center returns
        Y = X - np.mean(X, axis=0, keepdims=True)
        S_mle = (Y.T @ Y) / float(T)
        S_unbiased = (Y.T @ Y) / float(T - 1)

        # Shrinkage target F = mu * I_N
        mu = float(np.trace(S_mle)) / float(N)
        F = mu * np.eye(N, dtype=float)

        # Distance squared d^2 = ||S_mle - F||_F^2 / N
        d2 = float(np.sum((S_mle - F) ** 2)) / float(N)

        if d2 <= 1e-18:
            alpha = 0.0
        else:
            # Compute b_bar^2 = (1 / (N * T^2)) * sum_t ||y_t y_t^T - S_mle||_F^2
            # Note: sum_t ||y_t y_t^T - S_mle||_F^2 = sum_t (||y_t||_2^4) - T * ||S_mle||_F^2
            row_norms_sq = np.sum(Y ** 2, axis=1)
            sum_fourth = float(np.sum(row_norms_sq ** 2))
            s_frob_sq = float(np.sum(S_mle ** 2))
            b_bar2 = (sum_fourth / float(T ** 2) - s_frob_sq / float(T)) / float(N)
            b2 = min(max(b_bar2, 0.0), d2)
            alpha = float(np.clip(b2 / d2, 0.0, 1.0))

        cov_lw = (1.0 - alpha) * S_unbiased + alpha * F
        return cov_lw, {
            "method": "ledoit_wolf",
            "shrinkage_alpha": round(alpha, 6),
            "target_trace_mu": round(mu, 8),
            "sample_days": T,
        }

    # method_clean == "ewma"
    if not (0.0 < ewma_lambda < 1.0):
        raise ValueError(f"ewma_lambda must be in (0, 1), got {ewma_lambda}")

    # Weights v_t = (1 - lambda) * lambda^(T - 1 - t) for t = 0 .. T-1
    exponents = np.arange(T - 1, -1, -1, dtype=float)
    raw_weights = (1.0 - ewma_lambda) * (ewma_lambda ** exponents)
    norm_weights = raw_weights / np.sum(raw_weights)

    mean_w = np.sum(norm_weights[:, None] * X, axis=0, keepdims=True)
    Y_w = X - mean_w
    cov_ewma = (Y_w * norm_weights[:, None]).T @ Y_w
    return cov_ewma, {
        "method": "ewma",
        "ewma_lambda": ewma_lambda,
        "effective_window_days": round(1.0 / (1.0 - ewma_lambda), 1),
        "sample_days": T,
    }


@dataclass
class CovarianceComparisonResult:
    sample_annualized_vol: float
    ledoit_wolf_annualized_vol: float
    ewma_annualized_vol: float
    ledoit_wolf_shrinkage_alpha: float
    ewma_lambda: float
    sample_size_days: int
    evidence_list: List[Evidence] = field(default_factory=list)


@dataclass
class RiskMetrics:
    annualized_volatility: float          # sigma_p * sqrt(252)
    daily_volatility: float               # sigma_p daily
    historical_var_95: float              # 1-day 95% historical Value at Risk (negative decimal)
    max_drawdown: float                   # maximum peak-to-trough drop (negative decimal)
    max_drawdown_peak_date: Optional[str] # date of peak
    max_drawdown_trough_date: Optional[str] # date of trough
    sample_size_days: int                 # number of trading days used
    priced_weight_coverage: float         # fraction of portfolio weight with price series
    unpriced_holdings: List[str]          # tickers missing price data
    covariance_method: str = "sample"     # 'sample', 'ledoit_wolf', or 'ewma'
    covariance_metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_list: List[Evidence] = field(default_factory=list)


class RiskEngine:
    """
    Deterministic portfolio risk engine adhering to Jane Street math invariants.
    Computes:
      - Annualised Volatility: sigma_p = sqrt(w^T Sigma w) * sqrt(252) using selectable covariance
        ('sample' default, 'ledoit_wolf' shrinkage, or 'ewma')
      - 1-day 95% Historical Value-at-Risk (VaR): empirical 5th percentile of daily portfolio returns
      - Maximum Drawdown: min((wealth - peak) / peak) over the evaluation window
    Returns explicit Evidence objects with sample sizes and data caveats.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self._prices_cache: Optional[pd.DataFrame] = None

    def get_prices(self) -> pd.DataFrame:
        if self._prices_cache is None:
            self._prices_cache = self.loader.load_prices()
        return self._prices_cache

    def analyze_returns_series(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        priced_tickers: List[str],
        window_days: int = 756,
        cov_method: str = "sample",
        ewma_lambda: float = 0.94,
    ) -> Tuple[float, float, float, float, Optional[str], Optional[str]]:
        """
        Pure math helper: calculates annual vol, daily vol, 95% VaR, max DD from a returns DataFrame.
        """
        cov_matrix, _ = estimate_covariance_matrix(
            returns_df, method=cov_method, ewma_lambda=ewma_lambda
        )
        port_var = float(weights.T @ cov_matrix @ weights)
        daily_vol = float(np.sqrt(max(port_var, 0.0)))
        annualized_vol = float(daily_vol * np.sqrt(252))

        port_returns = returns_df.values @ weights
        port_returns_series = pd.Series(port_returns, index=returns_df.index)

        var_95 = float(np.percentile(port_returns_series, 5))

        cum_wealth = (1.0 + port_returns_series).cumprod()
        peak = cum_wealth.cummax()
        drawdown = (cum_wealth - peak) / peak
        max_dd = float(drawdown.min())

        trough_date = None
        peak_date = None
        if not drawdown.empty:
            trough_idx = drawdown.idxmin()
            trough_date = trough_idx.strftime("%Y-%m-%d") if hasattr(trough_idx, "strftime") else str(trough_idx)
            pre_trough_wealth = cum_wealth.loc[:trough_idx]
            peak_idx = pre_trough_wealth.idxmax()
            peak_date = peak_idx.strftime("%Y-%m-%d") if hasattr(peak_idx, "strftime") else str(peak_idx)

        return (
            annualized_vol,
            daily_vol,
            var_95,
            max_dd,
            peak_date,
            trough_date,
        )

    def analyze(
        self,
        holdings: List[Dict[str, Any]],
        window_years: int = 3,
        lookback_days: int = 756,
        cov_method: str = "sample",
        ewma_lambda: float = 0.94,
    ) -> RiskMetrics:
        """
        Analyzes 3-year historical risk of given portfolio holdings.
        Supports selectable covariance estimator: 'sample' (default), 'ledoit_wolf', or 'ewma'.
        """
        prices = self.get_prices()
        evidence_list: List[Evidence] = []

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

        total_priced_weight = sum(priced_weights)
        total_weight = total_priced_weight + cash_weight + sum(
            float(h.get("weight", 0.0)) for h in holdings if h.get("ticker") in unpriced_holdings
        )

        caveats: List[str] = []
        if unpriced_holdings:
            caveats.append(
                f"Tickers {unpriced_holdings} lack historical price series in matrix; risk metrics calculated on available coverage."
            )
        if cash_weight > 0:
            caveats.append(f"Cash allocation of {cash_weight*100:.1f}% treated with 0.0 volatility and 0.0 drawdown.")

        if not priced_tickers:
            return RiskMetrics(
                annualized_volatility=0.0,
                daily_volatility=0.0,
                historical_var_95=0.0,
                max_drawdown=0.0,
                max_drawdown_peak_date=None,
                max_drawdown_trough_date=None,
                sample_size_days=0,
                priced_weight_coverage=0.0,
                unpriced_holdings=unpriced_holdings,
                covariance_method=cov_method,
                covariance_metadata={},
                evidence_list=evidence_list,
            )

        p_window = prices[priced_tickers].iloc[-lookback_days:]
        rets = p_window.pct_change(fill_method=None).dropna()
        sample_size = len(rets)

        if cash_weight > 0:
            rets = rets.copy()
            rets["CASH_SYNTH"] = 0.0
            priced_tickers_with_cash = priced_tickers + ["CASH_SYNTH"]
            weights_arr = np.array(priced_weights + [cash_weight])
            weights_norm = weights_arr / weights_arr.sum()
            rets_eval = rets[priced_tickers_with_cash]
        else:
            weights_arr = np.array(priced_weights)
            weights_norm = weights_arr / weights_arr.sum()
            rets_eval = rets[priced_tickers]

        _, cov_meta = estimate_covariance_matrix(
            rets_eval, method=cov_method, ewma_lambda=ewma_lambda
        )

        ann_vol, daily_vol, var_95, max_dd, peak_d, trough_d = self.analyze_returns_series(
            returns_df=rets_eval,
            weights=weights_norm,
            priced_tickers=priced_tickers,
            window_days=sample_size,
            cov_method=cov_method,
            ewma_lambda=ewma_lambda,
        )

        coverage_pct = round((total_priced_weight + cash_weight) / total_weight, 4) if total_weight > 0 else 0.0

        evidence_list.append(
            Evidence(
                id="EV-RISK-ANN-VOL",
                metric="annualized_portfolio_volatility",
                value=round(ann_vol, 6),
                formula=f"sigma_p = sqrt(w^T * Cov_{cov_method} * w) * sqrt(252)",
                inputs={
                    "weights": {t: round(w, 4) for t, w in zip(priced_tickers, priced_weights)},
                    "lookback_days": sample_size,
                    "covariance_method": cov_method,
                    "covariance_metadata": cov_meta,
                },
                data_sources=["us_market_prices_matrix.csv (3-Year Daily Closes)"],
                sample_size=sample_size,
                caveats=caveats,
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-RISK-VAR-95",
                metric="daily_95_percent_historical_var",
                value=round(var_95, 6),
                formula="VaR_95 = percentile(R_portfolio, 5)",
                inputs={"sample_days": sample_size, "confidence": 0.95},
                data_sources=["us_market_prices_matrix.csv (3-Year Daily Closes)"],
                sample_size=sample_size,
                caveats=caveats,
            )
        )
        evidence_list.append(
            Evidence(
                id="EV-RISK-MAX-DRAWDOWN",
                metric="maximum_peak_to_trough_drawdown",
                value=round(max_dd, 6),
                formula="MDD = min((wealth_t - peak_t) / peak_t)",
                inputs={"peak_date": peak_d, "trough_date": trough_d},
                data_sources=["us_market_prices_matrix.csv (3-Year Daily Closes)"],
                sample_size=sample_size,
                caveats=caveats,
            )
        )

        return RiskMetrics(
            annualized_volatility=round(ann_vol, 6),
            daily_volatility=round(daily_vol, 6),
            historical_var_95=round(var_95, 6),
            max_drawdown=round(max_dd, 6),
            max_drawdown_peak_date=peak_d,
            max_drawdown_trough_date=trough_d,
            sample_size_days=sample_size,
            priced_weight_coverage=coverage_pct,
            unpriced_holdings=unpriced_holdings,
            covariance_method=cov_method,
            covariance_metadata=cov_meta,
            evidence_list=evidence_list,
        )

    def compare_covariance_methods(
        self,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 756,
        ewma_lambda: float = 0.94,
    ) -> CovarianceComparisonResult:
        """
        Computes portfolio annualised volatility across all 3 covariance estimators:
        Sample Covariance (default), Ledoit-Wolf Shrinkage, and EWMA (lambda=0.94).
        """
        res_sample = self.analyze(holdings, lookback_days=lookback_days, cov_method="sample")
        res_lw = self.analyze(holdings, lookback_days=lookback_days, cov_method="ledoit_wolf")
        res_ewma = self.analyze(
            holdings, lookback_days=lookback_days, cov_method="ewma", ewma_lambda=ewma_lambda
        )

        alpha = float(res_lw.covariance_metadata.get("shrinkage_alpha", 0.0))

        ev = Evidence(
            id="EV-RISK-COV-COMPARISON",
            metric="covariance_estimator_volatility_comparison",
            value={
                "sample_vol": res_sample.annualized_volatility,
                "ledoit_wolf_vol": res_lw.annualized_volatility,
                "ewma_vol": res_ewma.annualized_volatility,
            },
            formula="Compare sqrt(w^T * Sigma_m * w) * sqrt(252) for m in {sample, ledoit_wolf, ewma}",
            inputs={
                "ledoit_wolf_shrinkage_alpha": alpha,
                "ewma_lambda": ewma_lambda,
                "sample_days": res_sample.sample_size_days,
            },
            data_sources=["us_market_prices_matrix.csv (3-Year Daily Closes)"],
            sample_size=res_sample.sample_size_days,
            caveats=[],
        )

        return CovarianceComparisonResult(
            sample_annualized_vol=res_sample.annualized_volatility,
            ledoit_wolf_annualized_vol=res_lw.annualized_volatility,
            ewma_annualized_vol=res_ewma.annualized_volatility,
            ledoit_wolf_shrinkage_alpha=alpha,
            ewma_lambda=ewma_lambda,
            sample_size_days=res_sample.sample_size_days,
            evidence_list=[ev],
        )
