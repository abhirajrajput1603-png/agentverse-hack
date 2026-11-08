import pytest
from pathlib import Path
from xray.data_loader import DataLoader
from xray.engine.whatif import WhatIfSimulator
from xray.safety.audit import AuditLogger


def test_friction_calculation():
    """
    Test friction formula on a £10,000 equity trade:
    Spread: 5 bps = 10,000 * 0.0005 = £5.00
    Commission: £4.99
    Slippage: 0.0000005 * 10,000^2 = 0.0000005 * 100,000,000 = £50.00
    Total friction = £5.00 + £4.99 + £50.00 = £59.99
    """
    simulator = WhatIfSimulator()
    friction = simulator.compute_friction("NVDA", "BUY", 10000.0)

    assert abs(friction.spread_cost_gbp - 5.00) < 1e-2
    assert abs(friction.commission_gbp - 4.99) < 1e-2
    assert abs(friction.slippage_gbp - 50.00) < 1e-2
    assert abs(friction.total_friction_gbp - 59.99) < 1e-2


def test_case_01_adversarial_prompt_whatif(tmp_path: Path):
    """
    Case 1 adversarial test:
    Liam: 'I have £15,000 in cash. What if I put all of it into NVDA to boost my returns?'
    - Before: NVDA lookthrough = 34.71% (exceeds 20% limit).
    - After: Adding £15,000 to NVDA pushes total NVDA exposure to ~49.78% (> 49%).
    - Must trigger Circuit Breaker BREACH.
    - Before/After risk, concentration, and friction must be computed deterministically.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    audit = AuditLogger(tmp_path / "whatif_audit.jsonl")
    simulator = WhatIfSimulator(data_loader=loader, audit_logger=audit)

    comparison = simulator.simulate_add_cash(
        current_holdings=c1["holdings"],
        target_ticker="NVDA",
        additional_cash_gbp=15000.0,
        persona_name=c1["persona"],
        user_max_single_stock_limit=c1["max_risk_tolerance_single_stock"],
    )

    # Check friction deducted
    assert comparison.total_friction_gbp > 0
    assert comparison.after_portfolio_value_gbp > comparison.before_portfolio_value_gbp

    # Check Before lookthrough NVDA
    nvda_before = comparison.before_lt.issuer_exposures["NVDA"].total_weight
    assert abs(nvda_before - 0.3471) < 1e-3

    # Check After lookthrough NVDA
    nvda_after = comparison.after_lt.issuer_exposures["NVDA"].total_weight
    assert nvda_after > 0.49  # Jumps to ~49.8%

    # Circuit breaker on simulated portfolio MUST be a BREACH
    assert comparison.after_safety.is_safe is False
    primary_eval = comparison.after_safety.evaluations[0]
    assert primary_eval.status == "BREACH"
    assert primary_eval.breaching_issuer == "NVDA"
    assert primary_eval.actual_value > 0.49
    assert primary_eval.threshold == 0.20
    assert primary_eval.excess > 0.29  # Over 29% excess over policy limit!

    # Audit log check
    entries = audit.get_recent_entries()
    assert len(entries) >= 2  # Circuit breaker + What-if simulation


def test_rebalance_simulation(tmp_path: Path):
    """
    Test rotating capital out of an overweight stock into a diversifying ETF.
    Case 1: Sell £10,000 of NVDA and buy TLT.
    - NVDA direct weight decreases.
    - TLT allocation increases.
    - Two-legged friction is applied.
    """
    loader = DataLoader()
    portfolios = loader.load_synthetic_portfolios()
    c1 = next(p for p in portfolios if p["case_id"] == "CASE-01-DIVERSIFICATION-ILLUSION")

    audit = AuditLogger(tmp_path / "whatif_rebalance_audit.jsonl")
    simulator = WhatIfSimulator(data_loader=loader, audit_logger=audit)

    comparison = simulator.simulate_rebalance(
        current_holdings=c1["holdings"],
        sell_ticker="NVDA",
        sell_gbp=10000.0,
        buy_ticker="TLT",
        persona_name=c1["persona"],
        user_max_single_stock_limit=c1["max_risk_tolerance_single_stock"],
    )

    assert len(comparison.trade_frictions) == 2
    assert comparison.total_friction_gbp > 0
    # Portfolio value slightly lower after friction
    assert comparison.after_portfolio_value_gbp < comparison.before_portfolio_value_gbp

    nvda_before = comparison.before_lt.issuer_exposures["NVDA"].total_weight
    nvda_after = comparison.after_lt.issuer_exposures["NVDA"].total_weight
    assert nvda_after < nvda_before  # Exposure reduced

    # Attempting to sell more than owned should raise ValueError
    with pytest.raises(ValueError):
        simulator.simulate_rebalance(
            current_holdings=c1["holdings"],
            sell_ticker="NVDA",
            sell_gbp=9999999.0,
            buy_ticker="TLT",
            persona_name=c1["persona"],
        )

