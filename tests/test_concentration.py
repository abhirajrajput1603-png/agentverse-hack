import pytest
from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.concentration import ConcentrationEngine, ConcentrationProfile


def test_hand_calculated_concentration():
    """
    Test hand-calculated concentration metrics on a deterministic 4-asset portfolio:
    w = [0.4, 0.3, 0.2, 0.1]
    HHI = 0.4^2 + 0.3^2 + 0.2^2 + 0.1^2 = 0.16 + 0.09 + 0.04 + 0.01 = 0.30
    N_eff = 1 / 0.30 = 3.3333... -> 3.33
    Top 1 = 0.40
    Top 3 = 0.90
    Top 5 = 1.00
    """
    weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    profile = ConcentrationEngine._compute_profile(weights)
    assert abs(profile.hhi - 0.30) < 1e-5
    assert abs(profile.effective_n - 3.33) < 1e-2
    assert abs(profile.top_1_weight - 0.40) < 1e-5
    assert abs(profile.top_3_weight - 0.90) < 1e-5
    assert abs(profile.top_5_weight - 1.00) < 1e-5
    assert profile.total_issuers_count == 4


def test_equal_weighted_portfolio():
    """
    Equal-weighted 10 assets (0.1 each):
    HHI = 10 * 0.01 = 0.10
    N_eff = 1 / 0.10 = 10.0
    """
    weights = {f"STK_{i}": 0.10 for i in range(10)}
    profile = ConcentrationEngine._compute_profile(weights)
    assert abs(profile.hhi - 0.10) < 1e-5
    assert profile.effective_n == 10.0
    assert abs(profile.top_1_weight - 0.10) < 1e-5
    assert abs(profile.top_5_weight - 0.50) < 1e-5


def test_case_01_diversification_illusion_concentration():
    """
    Case 1:
      Direct holdings: NVDA (0.25), QQQ (0.35), SMH (0.25), SPY (0.15)
      Direct HHI = 0.25^2 + 0.35^2 + 0.25^2 + 0.15^2 = 0.0625 + 0.1225 + 0.0625 + 0.0225 = 0.2700
      Direct N_eff = 1 / 0.2700 = 3.70
      
      Lookthrough:
      Top 1 is NVDA with 34.71% concentration!
      Evidence objects correctly generated.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    lt_engine = LookThroughEngine(loader)
    lt_res = lt_engine.analyze(c1["holdings"])

    conc_engine = ConcentrationEngine()
    analysis = conc_engine.analyze(lt_res)

    # Check direct profile
    assert abs(analysis.direct.hhi - 0.2700) < 1e-4
    assert abs(analysis.direct.effective_n - 3.70) < 1e-2

    # Check look-through profile (top single issuer)
    assert analysis.lookthrough.top_issuers[0]["issuer"] == "NVDA"
    assert abs(analysis.lookthrough.top_1_weight - 0.3471) < 1e-3

    # Check evidence traceability
    ev_map = {ev.id: ev for ev in analysis.evidence_list}
    assert "EV-CONC-DIRECT-HHI" in ev_map
    assert "EV-CONC-DIRECT-NEFF" in ev_map
    assert "EV-CONC-LT-HHI" in ev_map
    assert "EV-CONC-LT-NEFF" in ev_map
    assert "EV-CONC-TOP1-WEIGHT" in ev_map
    assert ev_map["EV-CONC-TOP1-WEIGHT"].value == analysis.lookthrough.top_1_weight


def test_case_04_passive_diversified_concentration():
    """
    Case 4:
      Passive diversified baseline with cash.
      MSFT is top holding (~5.13%), followed by AAPL (~4.92%) and NVDA (~4.69%).
      Top 1 weight should be ~5.13% (< 10%).
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c4 = next(p for p in portfolios if p["case_id"] == "CASE-04-PASSIVE-DIVERSIFIED-BENCHMARK")

    lt_engine = LookThroughEngine(loader)
    lt_res = lt_engine.analyze(c4["holdings"])

    conc_engine = ConcentrationEngine()
    analysis = conc_engine.analyze(lt_res)

    assert analysis.lookthrough.top_issuers[0]["issuer"] == "MSFT"
    assert abs(analysis.lookthrough.top_1_weight - 0.05132) < 1e-4
    assert analysis.lookthrough.top_1_weight < 0.10
