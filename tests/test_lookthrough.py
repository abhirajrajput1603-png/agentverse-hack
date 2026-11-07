import pytest
from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine, Holding


def test_simple_lookthrough_formula_example():
    """
    Test the specification example:
    A user holds 10% NVDA direct and 20% in an ETF that holds 8% NVDA.
    Expected true NVDA exposure = 10% + (20% * 8%) = 11.6%.
    """
    engine = LookThroughEngine()
    # Inject synthetic fund with 8% NVDA and 92% residual
    engine.etf_data["MOCK_FUND"] = {
        "fund_name": "Mock Fund",
        "holdings": {"NVDA": 0.08, "OTHER_CONSTITUENTS": 0.92},
        "source": "Mock SEC",
        "as_of_date": "2024-Q3",
    }

    portfolio = [
        {"ticker": "NVDA", "weight": 0.10},
        {"ticker": "MOCK_FUND", "weight": 0.20},
        {"ticker": "CASH_GBP", "weight": 0.70},
    ]

    res = engine.analyze(portfolio)
    assert abs(res.total_weight - 1.0) < 1e-9
    nvda_exp = res.issuer_exposures["NVDA"]
    assert abs(nvda_exp.total_weight - 0.116) < 1e-6
    assert abs(nvda_exp.direct_weight - 0.10) < 1e-6
    assert abs(nvda_exp.indirect_weight - 0.016) < 1e-6


def test_weight_conservation_assert():
    """Verify that any non-conserving portfolio or engine state triggers an error."""
    engine = LookThroughEngine()
    bad_portfolio = [
        {"ticker": "AAPL", "weight": 0.50},
        {"ticker": "MSFT", "weight": 0.40},
    ]
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        engine.analyze(bad_portfolio)


def test_case_01_diversification_illusion_ground_truth():
    """
    Case 1:
      NVDA direct: 25%
      QQQ: 35% -> NVDA: 0.35 * 0.0894 = 0.03129
      SMH: 25% -> NVDA: 0.25 * 0.2245 = 0.056125
      SPY: 15% -> NVDA: 0.15 * 0.0648 = 0.00972
      Total effective NVDA: 0.25 + 0.03129 + 0.056125 + 0.00972 = 0.347135 (rounds to 34.7%)
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    engine = LookThroughEngine(loader)
    res = engine.analyze(c1["holdings"])

    # Total conservation
    assert abs(res.total_weight - 1.0) < 1e-7

    # Ground truth NVDA decomposition
    gt = c1["ground_truth_lookthrough"]
    nvda = res.issuer_exposures["NVDA"]

    assert abs(nvda.direct_weight - gt["NVDA_direct"]) < 1e-5
    assert abs(nvda.breakdown_by_source["QQQ"] - gt["NVDA_from_QQQ"]) < 1e-5
    assert abs(nvda.breakdown_by_source["SMH"] - gt["NVDA_from_SMH"]) < 1e-5
    assert abs(nvda.breakdown_by_source["SPY"] - gt["NVDA_from_SPY"]) < 1e-5
    assert abs(nvda.total_weight - gt["NVDA_total_effective"]) < 1e-3
    assert round(nvda.total_weight, 4) == 0.3471


def test_case_02_thematic_energy_holdings():
    """Case 2: VST is direct and kept intact even without ETF constituent breakdown."""
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c2 = next(p for p in portfolios if p["case_id"] == "CASE-02-AI-CAPEX-ENERGY-THEME")

    engine = LookThroughEngine(loader)
    res = engine.analyze(c2["holdings"])

    assert abs(res.total_weight - 1.0) < 1e-7
    assert res.issuer_exposures["VST"].total_weight == 0.25
    assert res.issuer_exposures["MSFT"].total_weight == 0.25
    # NVDA = direct (0.25) + SMH (0.25 * 0.2245 = 0.056125) = 0.306125
    assert abs(res.issuer_exposures["NVDA"].total_weight - 0.306125) < 1e-5


def test_case_03_rate_shock_duration_trap():
    """Case 3: ARKK decomposed, TLT maintained as aggregate safe/duration asset."""
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c3 = next(p for p in portfolios if p["case_id"] == "CASE-03-RATE-SHOCK-DURATION-TRAP")

    engine = LookThroughEngine(loader)
    res = engine.analyze(c3["holdings"])

    assert abs(res.total_weight - 1.0) < 1e-7
    # TLT has no constituent file, remains aggregate
    assert res.issuer_exposures["TLT"].total_weight == 0.50
    assert res.issuer_exposures["TLT"].is_fund_undisclosed is True
    # TSLA comes from ARKK (0.5 * 0.112 = 0.056)
    assert abs(res.issuer_exposures["TSLA"].total_weight - 0.056) < 1e-5


def test_case_04_passive_diversified_benchmark():
    """
    Case 4: Passive baseline.
    VTI (0.6) + SPY (0.2) + CASH_GBP (0.2).
    NVDA lookthrough: 0.6 * 0.0565 + 0.2 * 0.0648 = 0.04686 (~4.69%).
    Max single stock lookthrough is MSFT: 0.6 * 0.0618 + 0.2 * 0.0712 = 0.05132 (~5.13%).
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    engine = LookThroughEngine(loader)
    res = engine.analyze(c4["holdings"])

    assert abs(res.total_weight - 1.0) < 1e-7
    assert res.issuer_exposures["CASH_GBP"].total_weight == 0.20
    assert abs(res.issuer_exposures["NVDA"].total_weight - 0.04686) < 1e-4

    # Top single-stock lookthrough (excluding cash and OTHER_CONSTITUENTS)
    top_stocks = res.sorted_exposures(exclude_residual_and_cash=True)
    assert top_stocks[0].issuer == "MSFT"
    assert abs(top_stocks[0].total_weight - 0.05132) < 1e-4
    # All stocks well below 10%
    assert top_stocks[0].total_weight < 0.10
