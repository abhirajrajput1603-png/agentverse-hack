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
