import numpy as np
import pandas as pd
import pytest

from xray.data_loader import DataLoader
from xray.engine.correlation import CorrelationAndRiskContributionEngine


def test_hand_calculated_euler_risk_decomposition():
    """
    Analytical test on two uncorrelated assets:
    - Asset A volatility = 20% (variance = 0.04)
    - Asset B volatility = 10% (variance = 0.01)
    - Correlation = 0.0
    - Weights: w_A = 0.50, w_B = 0.50
    Expected:
    - Portfolio variance = 0.25 * 0.04 + 0.25 * 0.01 = 0.0125
    - Portfolio vol sigma_p = sqrt(0.0125) = 0.1118033989
    - %CR_A = 0.01 / 0.0125 = 80.0% (Ratio = 0.80 / 0.50 = 1.60x)
    - %CR_B = 0.0025 / 0.0125 = 20.0% (Ratio = 0.20 / 0.50 = 0.40x)
    - CCR_A + CCR_B == sigma_p
    """
    # Construct orthogonal return series of length 253
    n = 252
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # sin and cos are orthogonal (zero sample covariance)
    s1 = np.sin(t)
    s2 = np.cos(t)
    # Scale to exact sample daily standard deviations: 0.20/sqrt(252) and 0.10/sqrt(252)
    s1 = s1 / np.std(s1, ddof=1) * (0.20 / np.sqrt(252))
    s2 = s2 / np.std(s2, ddof=1) * (0.10 / np.sqrt(252))

    rets_df = pd.DataFrame({"ASSET_A": s1, "ASSET_B": s2})
    engine = CorrelationAndRiskContributionEngine()

    res = engine.compute_from_returns(
        returns_3y=rets_df,
        returns_1y=rets_df,
        tickers=["ASSET_A", "ASSET_B"],
        weights=np.array([0.5, 0.5]),
    )

    assert res.ccr_sum_check_passed is True
    assert abs(res.portfolio_volatility - np.sqrt(0.0125)) < 1e-5
    assert abs(res.ccr_sum - res.portfolio_volatility) < 1e-6

    contrib_map = {c.ticker: c for c in res.contributions}
    assert abs(contrib_map["ASSET_A"].pct_risk_contribution - 0.80) < 1e-4
    assert abs(contrib_map["ASSET_B"].pct_risk_contribution - 0.20) < 1e-4
    assert abs(contrib_map["ASSET_A"].risk_to_capital_ratio - 1.60) < 1e-3
    assert abs(contrib_map["ASSET_B"].risk_to_capital_ratio - 0.40) < 1e-3
    assert abs(res.corr_matrix_3y["ASSET_A"]["ASSET_B"]) < 1e-6


def test_case_01_correlation_and_risk_contribution():
    """
    Verify Case 1 (Liam's Diversification Illusion):
    - SPY, QQQ, NVDA, AAPL, MSFT are highly correlated.
    - NVDA has a risk contribution share significantly greater than its 25% capital weight.
    - Euler check sum(CCR_i) == sigma_p holds.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    engine = CorrelationAndRiskContributionEngine(data_loader=loader)
    res = engine.analyze(c1["holdings"])

    assert res.ccr_sum_check_passed is True
    assert abs(res.ccr_sum - res.portfolio_volatility) < 1e-5

    # Sum of percentage risk contributions must equal 1.0
    pct_sum = sum(c.pct_risk_contribution for c in res.contributions)
    assert abs(pct_sum - 1.0) < 1e-4

    # NVDA punches above its capital weight (25% capital -> ~39% of portfolio risk)
    nvda_contrib = next(c for c in res.contributions if c.ticker == "NVDA")
    assert nvda_contrib.capital_weight == 0.25
    assert nvda_contrib.pct_risk_contribution > 0.35
    assert nvda_contrib.risk_to_capital_ratio > 1.4

    # SPY and QQQ have high correlation (> 0.90)
    assert res.corr_matrix_3y["SPY"]["QQQ"] > 0.90
    assert res.avg_pairwise_corr_3y > 0.60


def test_case_04_cash_zero_risk_contribution():
    """
    Verify Case 4 (Passive Diversified Benchmark with 20% Cash, 60% VTI, 20% SPY):
    - CASH_GBP has 20% capital weight but 0.0 MCR, 0.0 CCR, and 0.0% risk contribution.
    - Risky equity assets (VTI + SPY) account for 100% of portfolio risk (risk_to_capital_ratio > 1.0).
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    engine = CorrelationAndRiskContributionEngine(data_loader=loader)
    res = engine.analyze(c4["holdings"])

    assert res.ccr_sum_check_passed is True
    contrib_map = {c.ticker: c for c in res.contributions}

    assert "CASH_GBP" in contrib_map
    assert abs(contrib_map["CASH_GBP"].capital_weight - 0.20) < 1e-5
    assert contrib_map["CASH_GBP"].mcr == 0.0
    assert contrib_map["CASH_GBP"].ccr == 0.0
    assert contrib_map["CASH_GBP"].pct_risk_contribution == 0.0
    assert contrib_map["CASH_GBP"].risk_to_capital_ratio == 0.0

    # VTI (60% capital) + SPY (20% capital) = 80% capital, but 100% of risk
    equity_risk_share = contrib_map["VTI"].pct_risk_contribution + contrib_map["SPY"].pct_risk_contribution
    assert abs(equity_risk_share - 1.0) < 1e-5
    assert contrib_map["VTI"].risk_to_capital_ratio > 1.20

