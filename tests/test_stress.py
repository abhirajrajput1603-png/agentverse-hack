import numpy as np
import pytest

from xray.data_loader import DataLoader
from xray.engine.stress import StressAndTailEngine


def test_tail_risk_ordering_and_bootstrap_ci():
    """
    Verify mathematical invariants of VaR and Expected Shortfall:
    - ES_99 <= VaR_99 <= VaR_95
    - ES_95 <= VaR_95
    - Bootstrap 95% confidence intervals bracket point estimates.
    """
    # Create a known 200-day return distribution
    rng = np.random.default_rng(123)
    returns = rng.normal(loc=0.0005, scale=0.015, size=500)
    # Inject 5 extreme tail days
    returns[:5] = [-0.08, -0.07, -0.065, -0.06, -0.055]

    metrics = StressAndTailEngine.compute_tail_metrics_from_series(
        port_returns=returns,
        n_bootstrap=500,
        random_seed=42,
    )

    assert metrics.total_sample_days == 500
    assert metrics.tail_sample_days_95 == 25
    assert metrics.tail_sample_days_99 == 5

    # Ordering invariants
    assert metrics.es_95 <= metrics.var_95
    assert metrics.var_99 <= metrics.var_95
    assert metrics.es_99 <= metrics.var_99
    assert metrics.es_99 <= metrics.es_95

    # Bootstrap CI brackets point estimate
    assert metrics.var_95_ci_low <= metrics.var_95 <= metrics.var_95_ci_high
    assert metrics.es_95_ci_low <= metrics.es_95 <= metrics.es_95_ci_high


def test_case_03_2022_duration_co_drawdown_trap():
    """
    Verify Case 3 (50% ARKK + 50% TLT) during the 2022 Fed Rate-Hike Shock:
    - Both ARKK and TLT suffer severe negative returns simultaneously in 2022.
    - Portfolio suffers a worse cumulative return than SPY.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c3 = next(p for p in portfolios if p["case_id"] == "CASE-03-RATE-SHOCK-DURATION-TRAP")

    engine = StressAndTailEngine(data_loader=loader)
    res = engine.analyze(c3["holdings"])

    shock_2022 = res.scenarios["YEAR_2022_FED_RATE_SHOCK"]
    assert shock_2022.coverage_weight == 1.0
    assert len(shock_2022.missing_history_holdings) == 0

    arkk_ret = shock_2022.holding_returns["ARKK"]
    tlt_ret = shock_2022.holding_returns["TLT"]

    # Both long-duration tech (ARKK) and long-duration Treasuries (TLT) crashed in 2022
    assert arkk_ret < -0.60  # ARKK dropped ~67% in 2022
    assert tlt_ret < -0.25   # TLT dropped ~31% in 2022
    assert shock_2022.portfolio_cumulative_return < -0.45
    assert shock_2022.portfolio_cumulative_return < shock_2022.spy_cumulative_return


def test_case_01_aug_2024_tech_unwind():
    """
    Verify Case 1 (Liam's AI/Tech heavy portfolio) during Aug 2024 Yen-Carry & Tech Unwind:
    - Portfolio suffers a sharper drawdown and cumulative loss than the SPY benchmark.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    engine = StressAndTailEngine(data_loader=loader)
    res = engine.analyze(c1["holdings"])

    aug_2024 = res.scenarios["AUG_2024_YEN_TECH_UNWIND"]
    assert aug_2024.portfolio_cumulative_return < aug_2024.spy_cumulative_return
    assert aug_2024.portfolio_max_drawdown < -0.15  # Exceeded 15% intra-window peak-to-trough drop


def test_case_02_missing_history_flagged_in_stress_windows():
    """
    Verify Case 2 (with unpriced VST holding at 25% weight):
    - VST is explicitly flagged in missing_history_holdings for every replay window.
    - Coverage weight is explicitly reported as 0.75 (75%).
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c2 = next(p for p in portfolios if p["case_id"] == "CASE-02-AI-CAPEX-ENERGY-THEME")

    engine = StressAndTailEngine(data_loader=loader)
    res = engine.analyze(c2["holdings"])

    assert "VST" in res.unpriced_holdings
    for scen_id, replay in res.scenarios.items():
        assert "VST" in replay.missing_history_holdings
        assert abs(replay.coverage_weight - 0.75) < 1e-4

