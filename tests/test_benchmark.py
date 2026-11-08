import pytest

from xray.data_loader import DataLoader
from xray.engine.benchmark import BenchmarkEngine


def test_spy_self_benchmark_identity():
    """
    Comparing a 100% SPY portfolio against the SPY benchmark must yield:
    - beta == 1.0
    - vol_ratio_vs_spy == 1.0
    - var_ratio_vs_spy == 1.0
    - all active tilts == 0.0
    """
    engine = BenchmarkEngine()
    res = engine.analyze([{"ticker": "SPY", "weight": 1.0}])

    assert abs(res.portfolio_beta_vs_spy - 1.0) < 1e-4
    assert abs(res.vol_ratio_vs_spy - 1.0) < 1e-4
    assert abs(res.var_ratio_vs_spy - 1.0) < 1e-4
    for tilt in res.active_tilts_vs_spy:
        assert abs(tilt.active_tilt) < 1e-6


def test_case_01_benchmark_comparison_and_active_tilts():
    """
    Verify Case 1 (Liam's Diversification Illusion) vs SPY and 60/40:
    - Beta vs SPY > 1.3
    - Volatility > 1.5x SPY and > 2.2x 60/40
    - Top active tilt is NVDA (+28.23% active overweight, 5.36x SPY weight)
    - Look-through N_eff (~7.4) is a fraction of SPY N_eff (~83.5)
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    engine = BenchmarkEngine(data_loader=loader)
    res = engine.analyze(c1["holdings"])

    assert res.portfolio_beta_vs_spy > 1.30
    assert res.vol_ratio_vs_spy > 1.50
    assert res.vol_ratio_vs_60_40 > 2.20
    assert res.user_metrics.lookthrough_n_eff < 10.0
    assert res.spy_benchmark.lookthrough_n_eff > 50.0

    top_tilt = res.active_tilts_vs_spy[0]
    assert top_tilt.issuer == "NVDA"
    assert abs(top_tilt.user_weight - 0.347135) < 1e-5
    assert abs(top_tilt.spy_weight - 0.0648) < 1e-5
    assert abs(top_tilt.active_tilt - 0.282335) < 1e-5
    assert abs(top_tilt.multiple_of_spy - 5.36) < 0.05


def test_case_04_passive_diversified_vs_benchmarks():
    """
    Verify Case 4 (60% VTI, 20% SPY, 20% Cash) vs SPY:
    - Beta vs SPY is ~0.80 (80% broad US equity + 20% cash)
    - Volatility and VaR are lower than 100% SPY (vol_ratio_vs_spy < 0.85)
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    engine = BenchmarkEngine(data_loader=loader)
    res = engine.analyze(c4["holdings"])

    assert 0.75 <= res.portfolio_beta_vs_spy <= 0.85
    assert res.vol_ratio_vs_spy < 0.85
    assert res.var_ratio_vs_spy < 0.85
