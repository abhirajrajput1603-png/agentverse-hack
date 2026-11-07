import pytest
import pandas as pd
from pathlib import Path
from xray.data_loader import DataLoader
from xray.evidence import Evidence, EvidenceRegistry


def test_evidence_object():
    ev = Evidence(
        id="EV-TEST-001",
        metric="test_metric",
        value=0.42,
        formula="a / b",
        inputs={"a": 42, "b": 100},
        data_sources=["data/test.csv:2024-Q3"],
        sample_size=100,
        caveats=["Test assumption"],
    )
    assert ev.id == "EV-TEST-001"
    assert ev.value == 0.42
    assert "a" in ev.inputs

    registry = EvidenceRegistry()
    registry.register(ev)
    assert registry.get("EV-TEST-001") == ev
    assert len(registry.all()) == 1


def test_data_loader_prices():
    loader = DataLoader()
    prices = loader.load_prices()
    assert isinstance(prices, pd.DataFrame)
    assert len(prices) > 3000
    assert "NVDA" in prices.columns
    assert "SPY" in prices.columns
    assert "QQQ" in prices.columns
    # Check index is DatetimeIndex
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_etf_constituents_weights_sum_to_one():
    loader = DataLoader()
    etfs = loader.load_etf_constituents()
    for ticker, fund in etfs.items():
        holdings = fund["holdings"]
        total_w = sum(holdings.values())
        assert abs(total_w - 1.0) < 1e-4, f"ETF {ticker} does not sum to 1.0: {total_w}"
        assert "OTHER_CONSTITUENTS" in holdings


def test_synthetic_portfolios_loading():
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    assert len(portfolios) == 4
    case_ids = [p["case_id"] for p in portfolios]
    assert "CASE-01-DIVERSIFICATION-ILLUSION" in case_ids
    assert "CASE-02-AI-CAPEX-ENERGY-THEME" in case_ids
    assert "CASE-03-RATE-SHOCK-DURATION-TRAP" in case_ids
    assert "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK" in case_ids


def test_portfolio_holdings_validation():
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()

    # Case 1: NVDA, QQQ, SMH, SPY -> all recognized, 0 missing
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")
    valid, decomposed, missing = loader.validate_holdings(c1["holdings"])
    assert len(missing) == 0
    assert set(decomposed) == {"QQQ", "SMH", "SPY"}

    # Case 2: VST is missing price series (expected per spec)
    c2 = next(p for p in portfolios if p["case_id"] == "CASE-02-AI-CAPEX-ENERGY-THEME")
    valid, decomposed, missing = loader.validate_holdings(c2["holdings"])
    assert "VST" in missing


def test_issuer_mapping():
    loader = DataLoader()
    issuer_map = loader.load_issuer_map()
    assert issuer_map.get("GOOG") == "GOOGL"
    assert issuer_map.get("BRK.B") == "BRK-B"
