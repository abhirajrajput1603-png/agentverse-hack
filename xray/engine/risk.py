from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xray.data_loader import DataLoader
from xray.engine.lookthrough import Holding, LookThroughResult
from xray.evidence import Evidence


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
    evidence_list: List[Evidence] = field(default_factory=list)


class RiskEngine:
    """
    Deterministic portfolio risk engine adhering to Jane Street math invariants.
    Computes:
      - Annualised Volatility: sigma_p = sqrt(w^T Sigma w) * sqrt(252) using sample covariance
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
    ) -> Tuple[float, float, float, float, Optional[str], Optional[str]]:
        """
        Pure math helper: calculates annual vol, daily vol, 95% VaR, max DD from a returns DataFrame.
        """
        # 1. Annualised volatility: sqrt(w^T Sigma w) * sqrt(252)
        cov_matrix = returns_df.cov().values
        port_var = float(weights.T @ cov_matrix @ weights)
        daily_vol = np.sqrt(max(port_var, 0.0))
        annualized_vol = daily_vol * np.sqrt(252)

        # 2. Portfolio return series: R_p(t) = sum_i(w_i * r_i(t))
        port_returns = returns_df.values @ weights
        port_returns_series = pd.Series(port_returns, index=returns_df.index)

        # 3. 1-day 95% Historical VaR: 5th percentile of daily portfolio return
        var_95 = float(np.percentile(port_returns_series, 5))

        # 4. Maximum Drawdown
        cum_wealth = (1.0 + port_returns_series).cumprod()
        peak = cum_wealth.cummax()
        drawdown = (cum_wealth - peak) / peak
        max_dd = float(drawdown.min())

        trough_date = None
        peak_date = None
        if not drawdown.empty:
            trough_idx = drawdown.idxmin()
            trough_date = trough_idx.strftime("%Y-%m-%d") if hasattr(trough_idx, "strftime") else str(trough_idx)
            # Find the peak prior to trough
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
    ) -> RiskMetrics:
        """
        Analyzes 3-year historical risk of given portfolio holdings.
        Weights of unpriced assets (e.g. VST) or safe cash are tracked explicitly.
        """
        prices = self.get_prices()
        evidence_list: List[Evidence] = []

        # Identify priced vs unpriced / cash
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

        # Check if we have sufficient priced securities
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
                evidence_list=evidence_list,
            )

        # Slice 3-year window
        p_window = prices[priced_tickers].iloc[-lookback_days:]
        # Calculate daily returns without forward-filling across gaps
        rets = p_window.pct_change(fill_method=None).dropna()
        sample_size = len(rets)

        # Re-scale weights to priced assets + cash
        # Cash has return 0.0 across all days
        if cash_weight > 0:
            # Add synthetic zero return series for cash
            rets["CASH_SYNTH"] = 0.0
            priced_tickers_with_cash = priced_tickers + ["CASH_SYNTH"]
            weights_arr = np.array(priced_weights + [cash_weight])
            # If total < 1 due to missing stocks (like VST in Case 2), normalize over known assets
            weights_norm = weights_arr / weights_arr.sum()
            rets_eval = rets[priced_tickers_with_cash]
        else:
            weights_arr = np.array(priced_weights)
            weights_norm = weights_arr / weights_arr.sum()
            rets_eval = rets[priced_tickers]

        ann_vol, daily_vol, var_95, max_dd, peak_d, trough_d = self.analyze_returns_series(
            returns_df=rets_eval,
            weights=weights_norm,
            priced_tickers=priced_tickers,
            window_days=sample_size,
        )

        coverage_pct = round((total_priced_weight + cash_weight) / total_weight, 4) if total_weight > 0 else 0.0

        # Construct Evidence objects
        evidence_list.append(
            Evidence(
                id="EV-RISK-ANN-VOL",
                metric="annualized_portfolio_volatility",
                value=round(ann_vol, 6),
                formula="sigma_p = sqrt(w^T * Cov * w) * sqrt(252)",
                inputs={"weights": {t: round(w, 4) for t, w in zip(priced_tickers, priced_weights)}, "lookback_days": sample_size},
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
            evidence_list=evidence_list,
        )
