import pytest

from xray.data_loader import DataLoader
from xray.engine.lookthrough import LookThroughEngine
from xray.engine.themes import ThemeEngine


def test_themes_json_structure_and_sources():
    """
    Verify every theme in data/themes.json has a name, description, and tickers
    with non-empty one-line evidence notes and sources.
    """
    loader = DataLoader()
    themes_data = loader.load_themes()
    assert "themes" in themes_data
    assert len(themes_data["themes"]) >= 5

    for theme_id, info in themes_data["themes"].items():
        assert info.get("name"), f"{theme_id} missing name"
        assert info.get("description"), f"{theme_id} missing description"
        tickers = info.get("tickers", {})
        assert len(tickers) > 0, f"{theme_id} has zero tickers"
        for t, meta in tickers.items():
            assert meta.get("note"), f"{theme_id}.{t} missing evidence note"
            assert meta.get("source"), f"{theme_id}.{t} missing source"


def test_non_additive_theme_exposure():
    """
    Verify theme exposure is non-additive across themes:
    A 50% NVDA / 50% MSFT portfolio has 100% exposure to AI_DATA_CENTER_AND_POWER_CAPEX,
    50% exposure to SEMICONDUCTOR_SUPPLY_CHAIN, and 50% exposure to CLOUD_HYPERSCALERS_AND_SAAS
    (sum across themes = 2.0 > 1.0).
    """
    loader = DataLoader()
    lt_eng = LookThroughEngine(loader)
    th_eng = ThemeEngine(loader)

    lt_res = lt_eng.analyze(
        [
            {"ticker": "NVDA", "weight": 0.50},
            {"ticker": "MSFT", "weight": 0.50},
        ]
    )
    th_res = th_eng.analyze(lt_res)

    assert abs(th_res.get_theme_weight("AI_DATA_CENTER_AND_POWER_CAPEX") - 1.0) < 1e-6
    assert abs(th_res.get_theme_weight("SEMICONDUCTOR_SUPPLY_CHAIN") - 0.50) < 1e-6
    assert abs(th_res.get_theme_weight("CLOUD_HYPERSCALERS_AND_SAAS") - 0.50) < 1e-6

    sum_across_themes = sum(t.total_weight for t in th_res.themes)
    assert sum_across_themes > 1.99  # Non-additive across overlapping themes


def test_case_02_ai_capex_energy_thematic_ground_truth():
    """
    Verify Case 2 (Sarah's 'Cross-Sector' portfolio: 25% NVDA, 25% MSFT, 25% VST, 25% SMH):
    - Recomputed AI_DATA_CENTER_AND_POWER_CAPEX exposure matches ground truth (0.920375 = 92.04%).
    - Dominant theme is AI_DATA_CENTER_AND_POWER_CAPEX.
    - VST (Utilities), MSFT (Cloud), and NVDA/TSM/AVGO (Hardware) are unified into a single theme.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c2 = next(p for p in portfolios if p["case_id"] == "CASE-02-AI-CAPEX-ENERGY-THEME")

    lt_res = LookThroughEngine(loader).analyze(c2["holdings"])
    th_res = ThemeEngine(loader).analyze(lt_res)

    expected_gt = c2["ground_truth_lookthrough"]["computed_ai_data_center_thematic_exposure"]
    actual_ai_theme = th_res.get_theme_weight("AI_DATA_CENTER_AND_POWER_CAPEX")

    assert abs(actual_ai_theme - 0.920375) < 1e-6
    assert abs(actual_ai_theme - expected_gt) < 1e-6
    assert th_res.dominant_theme is not None
    assert th_res.dominant_theme.theme_id == "AI_DATA_CENTER_AND_POWER_CAPEX"

    ai_issuers = {c.issuer for c in th_res.dominant_theme.contributing_issuers}
    assert {"NVDA", "MSFT", "VST", "TSM", "AVGO"}.issubset(ai_issuers)


def test_case_03_duration_theme_exposure():
    """
    Verify Case 3 (50% ARKK + 50% TLT):
    - Dominant economic theme is RATE_AND_DURATION_SENSITIVE (> 75% look-through exposure).
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c3 = next(p for p in portfolios if p["case_id"] == "CASE-03-RATE-SHOCK-DURATION-TRAP")

    lt_res = LookThroughEngine(loader).analyze(c3["holdings"])
    th_res = ThemeEngine(loader).analyze(lt_res)

    assert th_res.dominant_theme is not None
    assert th_res.dominant_theme.theme_id == "RATE_AND_DURATION_SENSITIVE"
    assert th_res.get_theme_weight("RATE_AND_DURATION_SENSITIVE") > 0.75
