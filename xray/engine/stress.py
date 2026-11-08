from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from xray.data_loader import DataLoader
from xray.evidence import Evidence


HISTORICAL_SHOCK_WINDOWS: Dict[str, Dict[str, str]] = {
    "MAR_2020_COVID_CRASH": {
        "name": "Mar 2020 COVID-19 Liquidity Crash",
        "start_date": "2020-02-19",
        "end_date": "2020-03-23",
        "description": "Global pandemic market halt and rapid equity/credit liquidation.",
    },
    "YEAR_2022_FED_RATE_SHOCK": {
        "name": "2022 Fed Rate-Hike & Duration Shock",
        "start_date": "2022-01-03",
        "end_date": "2022-12-30",
        "description": "Aggressive Federal Reserve tightening cycle causing simultaneous equity and long-bond drawdowns.",
    },
    "MAR_2023_SVB_CRISIS": {
        "name": "Mar 2023 SVB Banking & Liquidity Shock",
        "start_date": "2023-03-01",
        "end_date": "2023-03-31",
        "description": "Regional banking contagion triggered by Silicon Valley Bank duration losses.",
    },
    "AUG_2024_YEN_TECH_UNWIND": {
        "name": "Aug 2024 Yen-Carry & AI Tech Unwind",
        "start_date": "2024-07-15",
        "end_date": "2024-08-14",
        "description": "Bank of Japan rate hike and crowded AI/semiconductor momentum unwind.",
    },
}


@dataclass
class TailRiskMetrics:
    var_95: float                       # 1-day 95% historical VaR (negative decimal)
    var_95_ci_low: float                # 95% bootstrap CI lower bound (more negative)
    var_95_ci_high: float               # 95% bootstrap CI upper bound (less negative)
    es_95: float                        # 1-day 95% Expected Shortfall (mean loss beyond VaR_95)
    es_95_ci_low: float                 # 95% bootstrap CI lower bound for ES_95
    es_95_ci_high: float                # 95% bootstrap CI upper bound for ES_95
    var_99: float                       # 1-day 99% historical VaR
    es_99: float                        # 1-day 99% Expected Shortfall
    total_sample_days: int              # N trading days in estimation window
    tail_sample_days_95: int            # number of observations <= VaR_95
    tail_sample_days_99: int            # number of observations <= VaR_99


@dataclass
class StressReplayScenario:
    scenario_id: str
    name: str
    start_date: str
    end_date: str
    description: str
    portfolio_cumulative_return: float  # cumulative return over window
    portfolio_max_drawdown: float       # peak-to-trough drawdown inside window
    worst_single_day_return: float      # worst 1-day drop inside window
    spy_cumulative_return: float        # SPY benchmark cumulative return over same window
    holding_returns: Dict[str, float]   # per-ticker cumulative return over window
    missing_history_holdings: List[str] # tickers lacking price history in this window
    coverage_weight: float              # fraction of portfolio weight covered in this replay
    sample_days: int


@dataclass
class StressAndTailResult:
    tail_risk: TailRiskMetrics
    scenarios: Dict[str, StressReplayScenario]
    unpriced_holdings: List[str]
    evidence_list: List[Evidence] = field(default_factory=list)


class StressAndTailEngine:
    """
    Deterministic tail-risk and historical crisis replay engine.
    Computes:
      1. 95% & 99% Historical VaR and Expected Shortfall (ES = E[R | R <= VaR]).
      2. Non-parametric bootstrap 95% confidence intervals (fixed seed for reproducibility).
      3. Historical stress replays across Mar 2020, 2022 Rate Shock, Mar 2023 SVB, and Aug 2024 Unwind.
      4. Explicit tracking of holdings missing price history in any crisis window.
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        self.loader = data_loader or DataLoader()
        self._prices_cache: Optional[pd.DataFrame] = None

    def get_prices(self) -> pd.DataFrame:
        if self._prices_cache is None:
            self._prices_cache = self.loader.load_prices()
        return self._prices_cache

    @staticmethod
    def compute_tail_metrics_from_series(
        port_returns: np.ndarray,
        n_bootstrap: int = 1000,
        random_seed: int = 42,
    ) -> TailRiskMetrics:
        """
        Pure mathematical calculation of VaR (95%, 99%), Expected Shortfall (95%, 99%),
        and deterministic bootstrap 95% confidence intervals.
        """
        n = len(port_returns)
        if n == 0:
            return TailRiskMetrics(
                var_95=0.0,
                var_95_ci_low=0.0,
                var_95_ci_high=0.0,
                es_95=0.0,
                es_95_ci_low=0.0,
                es_95_ci_high=0.0,
                var_99=0.0,
                es_99=0.0,
                total_sample_days=0,
                tail_sample_days_95=0,
                tail_sample_days_99=0,
            )

        var_95 = float(np.percentile(port_returns, 5))
        tail_95 = port_returns[port_returns <= var_95]
        es_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95

        var_99 = float(np.percentile(port_returns, 1))
        tail_99 = port_returns[port_returns <= var_99]
        es_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        # Bootstrap 95% CI (2.5th and 97.5th percentiles of bootstrap distribution)
        rng = np.random.default_rng(random_seed)
        boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
        boot_samples = port_returns[boot_indices]

        boot_var_95 = np.percentile(boot_samples, 5, axis=1)
        var_ci_low = float(np.percentile(boot_var_95, 2.5))
        var_ci_high = float(np.percentile(boot_var_95, 97.5))

        boot_es_95 = np.empty(n_bootstrap, dtype=float)
        for b in range(n_bootstrap):
            sample_b = boot_samples[b]
            thresh_b = boot_var_95[b]
            tail_b = sample_b[sample_b <= thresh_b]
            boot_es_95[b] = np.mean(tail_b) if len(tail_b) > 0 else thresh_b

        es_ci_low = float(np.percentile(boot_es_95, 2.5))
        es_ci_high = float(np.percentile(boot_es_95, 97.5))

        return TailRiskMetrics(
            var_95=round(var_95, 6),
            var_95_ci_low=round(var_ci_low, 6),
            var_95_ci_high=round(var_ci_high, 6),
            es_95=round(es_95, 6),
            es_95_ci_low=round(es_ci_low, 6),
            es_95_ci_high=round(es_ci_high, 6),
            var_99=round(var_99, 6),
            es_99=round(es_99, 6),
            total_sample_days=n,
            tail_sample_days_95=int(len(tail_95)),
            tail_sample_days_99=int(len(tail_99)),
        )

    def replay_historical_window(
        self,
        holdings: List[Dict[str, Any]],
        scenario_id: str,
        scenario_meta: Dict[str, str],
        prices: pd.DataFrame,
    ) -> StressReplayScenario:
        """
        Replays portfolio performance over a specific historical crisis window.
        """
        start_d = scenario_meta["start_date"]
        end_d = scenario_meta["end_date"]
        window_prices = prices.loc[start_d:end_d]

        available_tickers: List[str] = []
        available_weights: List[float] = []
        missing_holdings: List[str] = []
        cash_weight: float = 0.0
        total_input_weight: float = 0.0

        for h in holdings:
            t = h.get("ticker", "")
            w = float(h.get("weight", 0.0))
            total_input_weight += w
            if t in ("CASH", "CASH_GBP", "CASH_USD"):
                cash_weight += w
            elif t in window_prices.columns:
                col_series = window_prices[t].dropna()
                if len(col_series) >= 2:
                    available_tickers.append(t)
                    available_weights.append(w)
                else:
                    missing_holdings.append(t)
            else:
                missing_holdings.append(t)

        covered_weight = sum(available_weights) + cash_weight
        coverage_ratio = (covered_weight / total_input_weight) if total_input_weight > 0 else 0.0

        # Calculate SPY benchmark return in the same window
        spy_cum_ret = 0.0
        if "SPY" in window_prices.columns:
            spy_s = window_prices["SPY"].dropna()
            if len(spy_s) >= 2:
                spy_cum_ret = float((spy_s.iloc[-1] / spy_s.iloc[0]) - 1.0)

        if not available_tickers and cash_weight == 0:
            return StressReplayScenario(
                scenario_id=scenario_id,
                name=scenario_meta["name"],
                start_date=start_d,
                end_date=end_d,
                description=scenario_meta["description"],
                portfolio_cumulative_return=0.0,
                portfolio_max_drawdown=0.0,
                worst_single_day_return=0.0,
                spy_cumulative_return=round(spy_cum_ret, 6),
                holding_returns={},
                missing_history_holdings=missing_holdings,
                coverage_weight=round(coverage_ratio, 4),
                sample_days=0,
            )

        # Compute per-holding cumulative return and daily portfolio path
        holding_returns: Dict[str, float] = {}
        for t in available_tickers:
            s = window_prices[t].dropna()
            holding_returns[t] = round(float((s.iloc[-1] / s.iloc[0]) - 1.0), 6)
        if cash_weight > 0:
            holding_returns["CASH_GBP"] = 0.0

        # Daily return series across available tickers
        rets_df = window_prices[available_tickers].pct_change(fill_method=None).dropna()
        if cash_weight > 0:
            rets_df = rets_df.copy()
            rets_df["CASH_GBP"] = 0.0
            eval_weights = np.array(available_weights + [cash_weight], dtype=float)
        else:
            eval_weights = np.array(available_weights, dtype=float)

        w_norm = eval_weights / eval_weights.sum()

        # Exact buy-and-hold cumulative return over the window: sum(w_norm_i * R_cum_i)
        ordered_tickers = available_tickers + (["CASH_GBP"] if cash_weight > 0 else [])
        port_cum_return = sum(
            float(w_norm[i]) * holding_returns[ordered_tickers[i]]
            for i in range(len(ordered_tickers))
        )

        # Daily wealth path inside the crisis window
        if len(rets_df) > 0:
            daily_port_rets = rets_df[ordered_tickers].values @ w_norm
            worst_day = float(np.min(daily_port_rets))
            wealth_index = np.concatenate([[1.0], np.cumprod(1.0 + daily_port_rets)])
            peaks = np.maximum.accumulate(wealth_index)
            drawdowns = (wealth_index - peaks) / peaks
            max_dd = float(np.min(drawdowns))
        else:
            worst_day = 0.0
            max_dd = min(0.0, port_cum_return)

        return StressReplayScenario(
            scenario_id=scenario_id,
            name=scenario_meta["name"],
            start_date=start_d,
            end_date=end_d,
            description=scenario_meta["description"],
            portfolio_cumulative_return=round(port_cum_return, 6),
            portfolio_max_drawdown=round(max_dd, 6),
            worst_single_day_return=round(worst_day, 6),
            spy_cumulative_return=round(spy_cum_ret, 6),
            holding_returns=holding_returns,
            missing_history_holdings=missing_holdings,
            coverage_weight=round(coverage_ratio, 4),
            sample_days=len(rets_df),
        )

    def analyze(
        self,
        holdings: List[Dict[str, Any]],
        lookback_days: int = 756,
        n_bootstrap: int = 1000,
    ) -> StressAndTailResult:
        """
        Computes 3-year tail risk (VaR_95, ES_95, VaR_99, ES_99 + Bootstrap CIs)
        and replays all 4 historical crisis windows.
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

        caveats: List[str] = []
        if unpriced_holdings:
            caveats.append(
                f"Tickers {unpriced_holdings} lack historical price series; tail risk and stress replays normalized over priced holdings."
            )

        # 1. Compute 3-year daily portfolio returns for Tail Risk
        if priced_tickers:
            p_window = prices[priced_tickers].iloc[-lookback_days:]
            rets = p_window.pct_change(fill_method=None).dropna()
            if cash_weight > 0:
                rets = rets.copy()
                rets["CASH_GBP"] = 0.0
                eval_cols = priced_tickers + ["CASH_GBP"]
                w_arr = np.array(priced_weights + [cash_weight], dtype=float)
            else:
                eval_cols = priced_tickers
                w_arr = np.array(priced_weights, dtype=float)
            w_norm = w_arr / w_arr.sum()
            port_returns = rets[eval_cols].values @ w_norm
        else:
            port_returns = np.array([], dtype=float)

        tail_metrics = self.compute_tail_metrics_from_series(
            port_returns=port_returns,
            n_bootstrap=n_bootstrap,
            random_seed=42,
        )

        evidence_list.append(
            Evidence(
                id="EV-TAIL-ES-95",
                metric="daily_95_percent_expected_shortfall",
                value=tail_metrics.es_95,
                formula="ES_95 = E[R_p | R_p <= VaR_95]",
                inputs={
                    "var_95": tail_metrics.var_95,
                    "es_95": tail_metrics.es_95,
                    "es_95_ci_95": [tail_metrics.es_95_ci_low, tail_metrics.es_95_ci_high],
                    "var_99": tail_metrics.var_99,
                    "es_99": tail_metrics.es_99,
                    "tail_sample_days_95": tail_metrics.tail_sample_days_95,
                },
                data_sources=["us_market_prices_matrix.csv (3-Year Daily Closes)"],
                sample_size=tail_metrics.total_sample_days,
                caveats=caveats,
            )
        )

        # 2. Replay all 4 historical crisis windows
        scenarios: Dict[str, StressReplayScenario] = {}
        for scen_id, scen_meta in HISTORICAL_SHOCK_WINDOWS.items():
            replay = self.replay_historical_window(
                holdings=holdings,
                scenario_id=scen_id,
                scenario_meta=scen_meta,
                prices=prices,
            )
            scenarios[scen_id] = replay

            scen_caveats = list(caveats)
            if replay.missing_history_holdings:
                scen_caveats.append(
                    f"Missing price history in {scen_meta['name']} for: {replay.missing_history_holdings} (coverage: {replay.coverage_weight*100:.1f}%)."
                )

            evidence_list.append(
                Evidence(
                    id=f"EV-STRESS-{scen_id}",
                    metric=f"historical_stress_replay_{scen_id.lower()}",
                    value=replay.portfolio_cumulative_return,
                    formula="sum_i(w_i * (P_{i,end} / P_{i,start} - 1))",
                    inputs={
                        "window": f"{replay.start_date} to {replay.end_date}",
                        "portfolio_cumulative_return": replay.portfolio_cumulative_return,
                        "portfolio_max_drawdown": replay.portfolio_max_drawdown,
                        "spy_cumulative_return": replay.spy_cumulative_return,
                        "coverage_weight": replay.coverage_weight,
                    },
                    data_sources=["us_market_prices_matrix.csv (Historical Crisis Window)"],
                    sample_size=replay.sample_days,
                    caveats=scen_caveats,
                )
            )

        return StressAndTailResult(
            tail_risk=tail_metrics,
            scenarios=scenarios,
            unpriced_holdings=unpriced_holdings,
            evidence_list=evidence_list,
        )
