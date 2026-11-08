import pytest
import numpy as np
import pandas as pd
from xray.data_loader import DataLoader
from xray.engine.risk import RiskEngine, RiskMetrics


def test_known_handbuilt_series_risk():
    """
    Test risk calculations on a known handbuilt series:
    1. Zero return asset -> vol = 0, VaR = 0, DD = 0
    2. Asset with known drop: [100, 110, 88, 99]
       Peak is 110, trough is 88 -> DD = (88 - 110) / 110 = -22 / 110 = -0.20 (-20%).
    """
    engine = RiskEngine()

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 110.0, 88.0, 99.0]}, index=dates)
    rets = prices.pct_change().dropna()
    weights = np.array([1.0])

    ann_vol, daily_vol, var_95, max_dd, peak_d, trough_d = engine.analyze_returns_series(
        returns_df=rets,
        weights=weights,
        priced_tickers=["A"],
        window_days=3,
    )

    # Max DD must be exactly -20%
    assert abs(max_dd - (-0.20)) < 1e-5
    assert peak_d == "2024-01-02"
    assert trough_d == "2024-01-03"


def test_zero_volatility_series():
    """Asset that never changes in price has zero volatility and zero drawdown."""
    engine = RiskEngine()
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"CASH_MOCK": [100.0] * 10}, index=dates)
    rets = prices.pct_change().dropna()
    weights = np.array([1.0])

    ann_vol, daily_vol, var_95, max_dd, peak_d, trough_d = engine.analyze_returns_series(
        returns_df=rets,
        weights=weights,
        priced_tickers=["CASH_MOCK"],
        window_days=9,
    )

    assert abs(ann_vol) < 1e-9
    assert abs(daily_vol) < 1e-9
    assert abs(var_95) < 1e-9
    assert abs(max_dd) < 1e-9


def test_case_01_diversification_illusion_risk():
    """
    Case 1: 4 tech/index positions over 3 years.
    Annualised vol should be ~28.7%, 95% 1-day VaR ~ -2.77%, Max DD ~ -27.1%.
    Sample size should be ~755 days.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    engine = RiskEngine(loader)
    risk = engine.analyze(c1["holdings"])

    assert risk.sample_size_days >= 750
    assert risk.priced_weight_coverage == 1.0
    assert len(risk.unpriced_holdings) == 0

    assert 0.25 < risk.annualized_volatility < 0.35
    assert -0.04 < risk.historical_var_95 < -0.015
    assert -0.35 < risk.max_drawdown < -0.20

    # Verify Evidence objects generated
    ev_map = {e.id: e for e in risk.evidence_list}
    assert "EV-RISK-ANN-VOL" in ev_map
    assert "EV-RISK-VAR-95" in ev_map
    assert "EV-RISK-MAX-DRAWDOWN" in ev_map
    assert ev_map["EV-RISK-ANN-VOL"].sample_size == risk.sample_size_days


def test_case_02_missing_vst_flagged():
    """
    Case 2 contains VST which lacks price series.
    The engine must cleanly report VST as unpriced, compute risk on priced portion,
    and include an explanatory caveat.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c2 = next(p for p in portfolios if p["case_id"] == "CASE-02-AI-CAPEX-ENERGY-THEME")

    engine = RiskEngine(loader)
    risk = engine.analyze(c2["holdings"])

    assert "VST" in risk.unpriced_holdings
    assert risk.priced_weight_coverage == 0.75
    assert any("VST" in c for c in risk.evidence_list[0].caveats)


def test_case_04_cash_dampening_risk():
    """
    Case 4: Passive portfolio with 20% Cash.
    Cash reduces portfolio volatility and drawdown.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    engine = RiskEngine(loader)
    risk = engine.analyze(c4["holdings"])

    assert risk.priced_weight_coverage == 1.0
    assert len(risk.unpriced_holdings) == 0
    # Volatility should be lower than pure tech portfolio (Case 1)
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")
    c1_risk = engine.analyze(c1["holdings"])
    assert risk.annualized_volatility < c1_risk.annualized_volatility


def test_ledoit_wolf_and_ewma_covariance_estimators():
    """
    Step 3.2: Verify sample, Ledoit-Wolf shrinkage, and EWMA covariance estimators:
    - All three return symmetric positive semi-definite (PSD) matrices.
    - Ledoit-Wolf shrinkage alpha is in [0.0, 1.0] and shrinks extreme off-diagonal covariances.
    - EWMA responds more strongly to a recent volatility spike than equal-weighted sample covariance.
    - Euler CCR check holds across all three covariance methods.
    """
    from xray.engine.risk import estimate_covariance_matrix
    from xray.engine.correlation import CorrelationAndRiskContributionEngine

    loader = DataLoader()
    c1 = loader.load_synthetic_portfolios()[0]
    engine = RiskEngine(loader)

    comp = engine.compare_covariance_methods(c1["holdings"])
    assert comp.sample_annualized_vol > 0.20
    assert comp.ledoit_wolf_annualized_vol > 0.20
    assert comp.ewma_annualized_vol > 0.10
    assert 0.0 <= comp.ledoit_wolf_shrinkage_alpha <= 1.0
    # Because Case 1 holdings are positively correlated, shrinking off-diagonals toward 0 reduces portfolio vol slightly
    assert comp.ledoit_wolf_annualized_vol <= comp.sample_annualized_vol

    # Check EWMA sensitivity to recent volatility regime spike
    rng = np.random.default_rng(99)
    calm = rng.normal(0.0, 0.005, size=(200, 2))
    spike = rng.normal(0.0, 0.040, size=(30, 2))
    df_regime = pd.DataFrame(np.vstack([calm, spike]), columns=["X", "Y"])

    cov_s, _ = estimate_covariance_matrix(df_regime, method="sample")
    cov_lw, meta_lw = estimate_covariance_matrix(df_regime, method="ledoit_wolf")
    cov_ewma, _ = estimate_covariance_matrix(df_regime, method="ewma", ewma_lambda=0.94)

    # PSD check (eigenvalues >= -1e-12)
    for m in (cov_s, cov_lw, cov_ewma):
        eigvals = np.linalg.eigvalsh(m)
        assert np.all(eigvals >= -1e-12)

    # EWMA variance must be significantly larger than equal-weighted sample variance after a recent spike
    assert np.trace(cov_ewma) > np.trace(cov_s) * 1.5

    # Euler CCR check holds for all 3 covariance methods
    corr_eng = CorrelationAndRiskContributionEngine(loader)
    for method in ("sample", "ledoit_wolf", "ewma"):
        res_c = corr_eng.analyze(c1["holdings"], cov_method=method)
        assert res_c.ccr_sum_check_passed is True

