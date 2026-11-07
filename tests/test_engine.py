"""Unit tests for Portfolio X-Ray core quantitative risk engine."""

import unittest
from portfolio_xray.engine import (
    Holding,
    Portfolio,
    compute_lookthrough_exposure,
    compute_thematic_exposure,
    compute_risk_metrics,
    run_stress_tests,
    simulate_what_if,
    analyze_headline_event,
)


class TestPortfolioXRayEngine(unittest.TestCase):

    def setUp(self):
        # Accidental Tech Overweight sample portfolio
        # $6,500 NVDA direct (20%)
        # $12,000 QQQ (36.9%)
        # $7,500 VOO (23.1%)
        # $6,500 SMH (20.0%)
        # Total Value: $32,500
        self.portfolio = Portfolio(
            holdings=[
                Holding(ticker="NVDA", shares=50, price=130.0),    # $6,500
                Holding(ticker="QQQ", shares=25, price=480.0),     # $12,000
                Holding(ticker="VOO", shares=15, price=500.0),     # $7,500
                Holding(ticker="SMH", shares=26, price=250.0),     # $6,500
            ],
            user_profile={
                "risk_tolerance": "Moderate",
                "horizon_years": 7,
                "loss_capacity": "Moderate (15-20% max drawdown)",
            }
        )

    def test_lookthrough_decomposition(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        self.assertTrue(len(exposures) > 0)

        # Find NVDA exposure
        nvda_exp = next((e for e in exposures if e.ticker == "NVDA"), None)
        self.assertIsNotNone(nvda_exp)

        # Direct NVDA is 20% ($6,500 / $32,500)
        self.assertAlmostEqual(nvda_exp.direct_weight, 0.20, places=2)

        # Indirect NVDA comes from QQQ (~8.6%), VOO (~6.2%), and SMH (~20.8%)
        self.assertTrue(nvda_exp.indirect_weight > 0.05)
        
        # Effective NVDA weight must be direct + indirect (~28-32%)
        self.assertAlmostEqual(nvda_exp.total_weight, nvda_exp.direct_weight + nvda_exp.indirect_weight, places=4)
        self.assertTrue(nvda_exp.total_weight > 0.25)

        # Verify contributing sources exist
        self.assertIn("QQQ", nvda_exp.contributing_sources)
        self.assertIn("SMH", nvda_exp.contributing_sources)
        self.assertIn("VOO", nvda_exp.contributing_sources)

    def test_risk_metrics_and_concentration(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        metrics = compute_risk_metrics(self.portfolio, exposures)

        self.assertEqual(metrics["portfolio_value"], 32500.0)
        self.assertTrue(metrics["hhi"] > 1000)
        self.assertTrue(metrics["annual_volatility"] > 0)
        self.assertTrue(metrics["var_95_1m_dollar"] > 0)
        self.assertTrue(metrics["expected_shortfall_95_dollar"] > metrics["var_95_1m_dollar"])

    def test_stress_testing(self):
        exposures = compute_lookthrough_exposure(self.portfolio)
        stress_results = run_stress_tests(self.portfolio, exposures)

        self.assertEqual(len(stress_results), 4)
        for s in stress_results:
            self.assertIn("scenario_id", s)
            self.assertIn("drawdown_pct", s)
            self.assertIn("dollar_drawdown", s)
            self.assertIn("worst_hit_assets", s)

    def test_what_if_simulation_breach(self):
        # Moderate investor putting 60% into NVDA should trigger warnings
        sim = simulate_what_if(self.portfolio, {"NVDA": 60.0, "SPY": 40.0})

        self.assertIn("metrics_delta", sim)
        self.assertIn("profile_warnings", sim)
        self.assertFalse(sim["is_aligned_with_profile"])
        self.assertTrue(any("concentration" in w.lower() for w in sim["profile_warnings"]))

    def test_counterexample_all_cash_portfolio(self):
        """Counterexample test: An all-cash portfolio must not trigger false volatility or drawdowns."""
        cash_portfolio = Portfolio(
            holdings=[Holding(ticker="CASH", shares=10000, price=1.0)],
            user_profile={"risk_tolerance": "Conservative", "horizon_years": 1, "loss_capacity": "Low"}
        )
        exposures = compute_lookthrough_exposure(cash_portfolio)
        metrics = compute_risk_metrics(cash_portfolio, exposures)
        stress = run_stress_tests(cash_portfolio, exposures)

        self.assertEqual(metrics["portfolio_value"], 10000.0)
        self.assertAlmostEqual(metrics["annual_volatility"], 0.5, delta=0.2) # ~0% vol
        self.assertTrue(metrics["var_95_1m_dollar"] < 50.0) # minimal VaR

        # In stress tests, cash is protected (drawdown >= 0%)
        for s in stress:
            self.assertTrue(s["drawdown_pct"] >= 0.0)

    def test_headline_event_analyzer(self):
        res = analyze_headline_event(self.portfolio, "Federal Reserve hikes interest rates by 50 basis points to fight inflation")
        self.assertIn("affected_holdings", res)
        self.assertTrue(res["net_estimated_impact_pct"] < 0) # Negative shock on tech portfolio
        self.assertTrue(len(res["affected_holdings"]) > 0)


if __name__ == "__main__":
    unittest.main()
