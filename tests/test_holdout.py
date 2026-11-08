import pytest

from xray.data_loader import DataLoader
from xray.engine.concentration import ConcentrationEngine
from xray.engine.lookthrough import LookThroughEngine
from xray.safety.limits import SafetyLimitsEngine


def test_dev_and_holdout_portfolios_counts_and_ground_truth():
    """
    Step 3.3: Verify 5 dev expanded portfolios and 10 holdout portfolios exist,
    conserve weight to 1.0, and match their computed ground truth.
    """
    loader = DataLoader()
    dev_cases = loader.load_dev_expanded_portfolios()
    holdout_cases = loader.load_holdout_portfolios()

    assert len(dev_cases) == 5
    assert len(holdout_cases) == 10

    lt_eng = LookThroughEngine(loader)
    conc_eng = ConcentrationEngine()
    safety_eng = SafetyLimitsEngine()

    for case in dev_cases + holdout_cases:
        holdings = case["holdings"]
        limit = float(case["max_risk_tolerance_single_stock"])
        gt = case["ground_truth_lookthrough"]

        lt_res = lt_eng.analyze(holdings)
        conc_res = conc_eng.analyze(lt_res)
        safety_res = safety_eng.evaluate_portfolio(
            lt_result=lt_res,
            concentration=conc_res,
            user_max_single_stock_limit=limit,
            persona_name=case["persona"],
        )

        assert abs(lt_res.total_weight - 1.0) < 1e-7
        top_named = lt_res.sorted_exposures(exclude_residual_and_cash=True)[0]
        assert top_named.issuer == gt["computed_top_issuer"]
        assert abs(top_named.total_weight - gt["computed_top_issuer_weight"]) < 1e-6
        assert safety_res.evaluations[0].status == gt["computed_circuit_breaker_status"]


def test_dual_share_class_stealth_breaches():
    """
    Verify DEV-05 (GOOG + GOOGL) and HOLDOUT-04 (BRK.B + BRK-B) have zero
    surface ticker breaches, but trigger look-through/issuer-mapped breaches.
    """
    loader = DataLoader()
    dev_cases = loader.load_dev_expanded_portfolios()
    holdout_cases = loader.load_holdout_portfolios()

    dev05 = next(c for c in dev_cases if c["case_id"] == "DEV-05-ALPHABET-DUAL-CLASS-TRAP")
    assert dev05["ground_truth_lookthrough"]["computed_surface_breaches"] == []
    assert "GOOGL" in dev05["ground_truth_lookthrough"]["computed_hidden_breaches"]

    h04 = next(c for c in holdout_cases if c["case_id"] == "HOLDOUT-04-BERKSHIRE-SHARE-CLASS-MERGE")
    assert h04["ground_truth_lookthrough"]["computed_surface_breaches"] == []
    assert "BRK-B" in h04["ground_truth_lookthrough"]["computed_hidden_breaches"]
